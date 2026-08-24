"""Taskbar indicator — shows at a glance whether Keylane is working.

States:

``idle``      gateway reachable, nothing running
``busy``      one or more tasks in flight (the icon animates)
``attention`` a task is waiting for the user to approve something
``offline``   the gateway is not answering

This module runs as its **own process**. AppIndicator speaks GTK 3 and the popup
speaks GTK 4, and the two cannot be loaded into one process, so ``launcher.main``
spawns this and the popup separately. Clicking "Open Keylane" re-invokes the
launcher, whose GApplication single-instance lock routes the activation to the
already-running popup.

Run directly with::

    python -m launcher.tray
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "3.0")

from gi.repository import GLib, Gtk  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher.gateway import DEFAULT_GATEWAY, GatewayClient  # noqa: E402

logger = logging.getLogger(__name__)

Indicator: Any = None
INDICATOR_FLAVOUR = ""

for _namespace, _version in (("AyatanaAppIndicator3", "0.1"), ("AppIndicator3", "0.1")):
    try:
        gi.require_version(_namespace, _version)
        _module = __import__("gi.repository", fromlist=[_namespace])
        Indicator = getattr(_module, _namespace)
        INDICATOR_FLAVOUR = _namespace
        break
    except (ValueError, ImportError, AttributeError):
        continue


# Icon names resolved from assets/icons/hicolor.
STATE_ICONS = {
    "idle": "keylane-idle",
    "busy": "keylane-busy",
    "attention": "keylane-attention",
    "offline": "keylane-offline",
}
# Alternated while busy so the panel visibly shows work happening.
BUSY_FRAMES = ("keylane-busy", "keylane-busy-alt")


def _icon_search_path() -> None:
    icons = ROOT / "assets" / "icons"
    if icons.exists():
        Gtk.IconTheme.get_default().append_search_path(str(icons))


def open_popup() -> None:
    """Ask the launcher to show the popup (single-instance activation)."""
    launcher = ROOT / "launcher" / "main.py"
    python = sys.executable or "python3"
    try:
        subprocess.Popen(
            [python, str(launcher), "--toggle"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not open the popup: %s", exc)


class KeylaneTray:
    def __init__(self, client: GatewayClient) -> None:
        self.client = client
        self.state = "offline"
        self._snapshot: dict[str, Any] = {}
        self._stop = threading.Event()
        self._frame = 0
        self._animation_id: int | None = None
        self._indicator = None
        self._status_item: Gtk.MenuItem | None = None

        _icon_search_path()
        self._build_indicator()
        self._start_watcher()

    # ------------------------------------------------------------- indicator

    def _build_indicator(self) -> None:
        if Indicator is None:
            logger.error(
                "No AppIndicator library found, so there is no tray icon. "
                "On Fedora: sudo dnf install libayatana-appindicator-gtk3 "
                "gnome-shell-extension-appindicator, then log out and back in."
            )
            return
        try:
            self._indicator = Indicator.Indicator.new(
                "keylane",
                STATE_ICONS["offline"],
                Indicator.IndicatorCategory.APPLICATION_STATUS,
            )
            self._indicator.set_status(Indicator.IndicatorStatus.ACTIVE)
            self._indicator.set_title("Keylane")
            self._indicator.set_menu(self._build_menu())
            logger.info("Tray indicator active via %s.", INDICATOR_FLAVOUR)
            self._warn_if_no_host()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create the tray indicator: %s", exc)
            self._indicator = None

    @staticmethod
    def _warn_if_no_host() -> None:
        """An indicator with nowhere to draw itself fails silently — say so.

        On GNOME the StatusNotifier host comes from the AppIndicator shell
        extension. Installed-but-disabled looks identical to a broken tray, so
        the log has to name the fix.
        """
        try:
            from gi.repository import Gio

            Gio.DBusProxy.new_for_bus_sync(
                Gio.BusType.SESSION,
                Gio.DBusProxyFlags.DO_NOT_AUTO_START,
                None,
                "org.kde.StatusNotifierWatcher",
                "/StatusNotifierWatcher",
                "org.kde.StatusNotifierWatcher",
                None,
            ).get_cached_property("IsStatusNotifierHostRegistered")
        except Exception:  # noqa: BLE001
            logger.warning(
                "No StatusNotifier host is running, so the icon will not appear. "
                "On GNOME: gnome-extensions enable "
                "appindicatorsupport@rgcjonas.gmail.com  (install "
                "gnome-shell-extension-appindicator first if it is missing)."
            )

    def _build_menu(self) -> Gtk.Menu:
        menu = Gtk.Menu()

        status = Gtk.MenuItem(label="Connecting…")
        status.set_sensitive(False)
        menu.append(status)
        self._status_item = status

        menu.append(Gtk.SeparatorMenuItem())

        open_item = Gtk.MenuItem(label="Open Keylane   (Super+Space)")
        open_item.connect("activate", lambda *_: open_popup())
        menu.append(open_item)

        panel_item = Gtk.MenuItem(label="Control panel…")
        panel_item.connect(
            "activate", lambda *_: webbrowser.open(self.client.base_url + "/")
        )
        menu.append(panel_item)

        docs_item = Gtk.MenuItem(label="Documentation…")
        docs_item.connect(
            "activate", lambda *_: webbrowser.open(self.client.base_url + "/docs/")
        )
        menu.append(docs_item)

        menu.append(Gtk.SeparatorMenuItem())

        quit_item = Gtk.MenuItem(label="Quit tray")
        quit_item.connect("activate", lambda *_: self.quit())
        menu.append(quit_item)

        menu.show_all()
        return menu

    # ---------------------------------------------------------------- polling

    def _start_watcher(self) -> None:
        """Prefer the SSE stream; fall back to polling when it will not connect."""

        def work() -> None:
            backoff = 1.0
            while not self._stop.is_set():
                try:
                    self.client.stream_events(
                        lambda snapshot: GLib.idle_add(self._apply_snapshot, snapshot),
                        self._stop.is_set,
                    )
                    backoff = 1.0
                except Exception:  # noqa: BLE001
                    # Gateway down or SSE refused: poll once, then retry the stream.
                    GLib.idle_add(self._apply_snapshot, None)
                    self._stop.wait(min(backoff, 15.0))
                    backoff = min(backoff * 2, 15.0)

        threading.Thread(target=work, daemon=True, name="keylane-tray-watch").start()

    def _apply_snapshot(self, snapshot: dict[str, Any] | None) -> bool:
        if snapshot is None:
            # Probe directly so a gateway that simply lacks SSE still reports.
            probe = self.client.activity()
            reachable = bool(probe.get("at"))
            self._snapshot = probe
        else:
            reachable = True
            self._snapshot = snapshot

        waiting = int(self._snapshot.get("needs_attention") or 0)
        busy = bool(self._snapshot.get("busy"))

        if not reachable:
            state = "offline"
        elif waiting:
            state = "attention"
        elif busy:
            state = "busy"
        else:
            state = "idle"

        self._set_state(state)
        self._update_menu_label()
        return False

    def _update_menu_label(self) -> None:
        if self._status_item is None:
            return
        active = self._snapshot.get("active") or []
        count = int(self._snapshot.get("active_count") or 0)
        waiting = int(self._snapshot.get("needs_attention") or 0)

        if self.state == "offline":
            text = "Gateway offline"
        elif waiting:
            text = f"{waiting} task{'s' if waiting != 1 else ''} awaiting approval"
        elif count:
            first = active[0] if active else {}
            title = str(first.get("title") or "working")[:48]
            worker = first.get("worker")
            step = first.get("step")
            suffix = f" · {step or worker}" if (step or worker) else ""
            text = f"Working: {title}{suffix}" if count == 1 else f"{count} tasks running"
        else:
            text = "Idle"
        self._status_item.set_label(text)

    # ------------------------------------------------------------------ icon

    def _set_state(self, state: str) -> None:
        if state == self.state:
            return
        previous, self.state = self.state, state
        logger.debug("Tray state %s → %s", previous, state)

        if state == "busy":
            self._start_animation()
        else:
            self._stop_animation()
            self._set_icon(STATE_ICONS.get(state, STATE_ICONS["idle"]))

        if state == "attention" and previous != "attention":
            notify("Keylane needs your approval", "A task is waiting to continue.")

    def _set_icon(self, icon_name: str) -> None:
        if self._indicator is None:
            return
        try:
            self._indicator.set_icon_full(icon_name, "Keylane")
        except Exception:  # noqa: BLE001
            try:
                self._indicator.set_icon(icon_name)
            except Exception:  # noqa: BLE001
                pass

    def _start_animation(self) -> None:
        if self._animation_id is not None:
            return

        def tick() -> bool:
            if self.state != "busy":
                self._animation_id = None
                return False
            self._frame = (self._frame + 1) % len(BUSY_FRAMES)
            self._set_icon(BUSY_FRAMES[self._frame])
            return True

        self._set_icon(BUSY_FRAMES[0])
        self._animation_id = GLib.timeout_add(650, tick)

    def _stop_animation(self) -> None:
        if self._animation_id is not None:
            GLib.source_remove(self._animation_id)
            self._animation_id = None

    # ------------------------------------------------------------------ stop

    def quit(self) -> None:
        self._stop.set()
        self._stop_animation()
        Gtk.main_quit()


def notify(title: str, body: str) -> None:
    binary = shutil.which("notify-send")
    if binary is None:
        return
    try:
        subprocess.Popen(
            [binary, "-a", "Keylane", "-i", "keylane", title, body],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        pass


def tray_available() -> bool:
    return Indicator is not None


def run_tray() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s keylane.tray: %(message)s"
    )
    client = GatewayClient(os.environ.get("KEYLANE_GATEWAY", DEFAULT_GATEWAY))
    tray = KeylaneTray(client)
    if tray._indicator is None and not os.environ.get("KEYLANE_TRAY_FORCE"):
        logger.error("Exiting: no system tray host is available.")
        return 1
    try:
        Gtk.main()
    except KeyboardInterrupt:
        tray.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tray())
