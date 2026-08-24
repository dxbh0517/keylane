#!/usr/bin/env python3
"""Keylane launcher — Spotlight popup plus the taskbar indicator.

Usage::

    python launcher/main.py            # popup app; spawns the tray if possible
    python launcher/main.py --toggle   # show/hide the popup (what Super+Space runs)
    python launcher/main.py --no-tray  # popup only
    python launcher/main.py --tray     # tray only (GTK 3 process)

The popup is a single-instance GApplication: running it again just activates the
one already there, which is what makes a plain keyboard shortcut a toggle.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

# Allow running from a checkout without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("keylane.launcher")

# Overridable so a development checkout can run beside an installed copy.
APP_ID = os.environ.get("KEYLANE_APP_ID", "app.keylane.Launcher")


def _spawn_tray() -> subprocess.Popen | None:
    """Start the tray in its own process — AppIndicator needs GTK 3."""
    if os.environ.get("KEYLANE_NO_TRAY"):
        return None
    python = sys.executable or "python3"
    try:
        return subprocess.Popen(
            [python, "-m", "launcher.tray"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not start the tray indicator: %s", exc)
        return None


def _choose_backend() -> None:
    """Pick the GDK backend that lets the result panel stay on top.

    Wayland gives clients no control over stacking. Compositors that
    implement ``zwlr_layer_shell_v1`` (sway, Hyprland) offer an overlay layer
    instead — but GNOME's Mutter implements neither, so a Wayland window there
    simply cannot be kept above other windows.

    XWayland can: Mutter honours ``_NET_WM_STATE_ABOVE`` on X11 surfaces. So
    when layer-shell is missing we run through XWayland to get that one
    capability. Set ``KEYLANE_BACKEND`` to override.
    """
    if os.environ.get("KEYLANE_BACKEND"):
        os.environ["GDK_BACKEND"] = os.environ["KEYLANE_BACKEND"]
        return
    if os.environ.get("GDK_BACKEND"):
        return
    if os.environ.get("XDG_SESSION_TYPE") != "wayland":
        return

    # The typelib being installed proves nothing: it is present on Fedora
    # GNOME, where is_supported() is False because Mutter does not implement
    # the protocol. Only a real probe answers this, and it needs a display —
    # so run it in a throwaway subprocess rather than opening one here.
    probe = (
        "import gi;"
        "gi.require_version('Gtk','4.0');"
        "gi.require_version('Gtk4LayerShell','1.0');"
        "from gi.repository import Gtk, Gtk4LayerShell as L;"
        "import sys;"
        "sys.exit(0 if L.is_supported() else 1)"
    )
    try:
        have_layer_shell = (
            subprocess.run(
                [sys.executable, "-c", probe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                env={**os.environ, "GDK_BACKEND": "wayland"},
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001
        have_layer_shell = False

    if not have_layer_shell and os.environ.get("DISPLAY"):
        os.environ["GDK_BACKEND"] = "x11"
        logger.info(
            "No layer-shell: using XWayland so the result panel can stay on top."
        )


def run_launcher(*, with_tray: bool = True, background: bool = False) -> int:
    _choose_backend()

    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, Gio, Gtk

    from gi.repository import GLib

    from launcher.gateway import DEFAULT_GATEWAY, GatewayClient
    from launcher.popup import KeylanePopup
    from launcher.result_orb import ResultOrb
    from launcher.theming import install_icon_search_path

    def _open_link(target: str) -> None:
        """Open a path or URL the answer produced."""
        import subprocess as sp

        try:
            sp.Popen(["xdg-open", target], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open %s: %s", target, exc)

    client = GatewayClient(os.environ.get("KEYLANE_GATEWAY", DEFAULT_GATEWAY))

    class LauncherApp(Adw.Application):
        def __init__(self) -> None:
            # No command-line handling: a second launch simply activates the
            # primary instance, and activation is what toggles the popup.
            super().__init__(
                application_id=APP_ID,
                flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
            )
            self.window: KeylanePopup | None = None
            self.orb: ResultOrb | None = None
            self.tray: subprocess.Popen | None = None
            # Starting as a background service must not put the popup on
            # screen; the first activation GApplication sends on run() is
            # swallowed, and every later one is a real hotkey press.
            self._swallow_first_activation = background

        def do_startup(self) -> None:  # noqa: N802
            Adw.Application.do_startup(self)
            install_icon_search_path()
            Gtk.Window.set_default_icon_name("keylane")
            # Hold the app open so the popup survives being closed.
            self.hold()
            if with_tray:
                self.tray = _spawn_tray()

        def do_shutdown(self) -> None:  # noqa: N802
            if self.tray is not None and self.tray.poll() is None:
                self.tray.terminate()
            Adw.Application.do_shutdown(self)

        # ------------------------------------------------------------ orb

        def _ensure_orb(self) -> "ResultOrb":
            corner = "top-right"
            try:
                config = client.gateway_config()
                corner = str(config.get("result_corner") or corner)
            except Exception:  # noqa: BLE001
                pass
            if self.orb is not None and self.orb.corner != corner:
                self.orb.destroy()
                self.orb = None
            if self.orb is None:
                self.orb = ResultOrb(
                    self,
                    client=client,
                    corner=corner,
                    on_open_link=_open_link,
                    on_reopen=self.activate,
                )
            return self.orb

        def submit(self, message: str, payload: dict) -> None:
            """Send a request and let the orb carry it from here."""
            orb = self._ensure_orb()
            orb.start(message)
            self._watch_activity(orb)

            def work() -> None:
                data = client.chat(payload)
                GLib.idle_add(self._on_result, data, payload)

            threading.Thread(target=work, daemon=True).start()

        def _watch_activity(self, orb) -> None:
            """Colour the orb from the gateway's live activity stream.

            The orb otherwise sits on one colour for the whole request, which
            hides the difference between "still thinking" and "Claude Code has
            had this for ninety seconds".
            """
            stop = getattr(self, "_activity_stop", None)
            if stop is not None:
                stop.set()
            stop = threading.Event()
            self._activity_stop = stop

            def on_snapshot(snapshot: dict) -> None:
                active = (snapshot.get("active") or [None])[0]
                if not active:
                    return
                status = str(active.get("status") or "")
                step = str(active.get("step") or "")
                worker = str(active.get("worker") or "")
                state = {
                    "thinking": "thinking",
                    "routing": "thinking",
                    "running": "tool",
                    "verifying": "verifying",
                    "retrying": "tool",
                    "waiting_confirmation": "waiting",
                }.get(status, "thinking")
                if worker and worker not in {"assistant", ""}:
                    state = "delegating"
                GLib.idle_add(orb.set_state, state)
                if step:
                    GLib.idle_add(orb.set_status, step, state)

            def work() -> None:
                try:
                    client.stream_events(on_snapshot, stop.is_set)
                except Exception:  # noqa: BLE001
                    pass  # the orb keeps its last colour

            threading.Thread(target=work, daemon=True).start()

        def _on_result(self, data: dict, payload: dict) -> bool:
            stop = getattr(self, "_activity_stop", None)
            if stop is not None:
                stop.set()
            orb = self._ensure_orb()
            status = str(data.get("status") or "").lower()

            if data.get("requires_confirmation") or status == "waiting_confirmation":
                orb.show_confirmation(
                    data,
                    on_allow=lambda: self._confirm(data, payload),
                    on_cancel=orb.dismiss,
                )
                return False

            failed = status not in {"completed", "success"}
            canvas = data.get("canvas")
            if not canvas:
                text = data.get("result") or data.get("error") or "No answer."
                canvas = {"blocks": [{"type": "text", "text": str(text)}]}
            orb.show_result(canvas, failed=failed)
            return False

        def _confirm(self, data: dict, payload: dict) -> None:
            orb = self._ensure_orb()
            orb.start(payload.get("message", ""))
            body = {**payload, "confirmed": True, "task_id": data.get("task_id")}

            def work() -> None:
                result = client.chat(body)
                GLib.idle_add(self._on_result, result, payload)

            threading.Thread(target=work, daemon=True).start()

        # ---------------------------------------------------------- popup

        def do_activate(self) -> None:  # noqa: N802
            if self._swallow_first_activation:
                self._swallow_first_activation = False
                logger.info("Started in the background; waiting for the hotkey.")
                return
            if self.window is not None and self.window.get_visible():
                self.window.dismiss()
                return
            self.window = KeylanePopup(
                self,
                client,
                on_closed=self._forget_window,
                on_submit=self.submit,
            )
            self.window.present_popup()

        def _forget_window(self) -> None:
            # The popup closes for real, so the next activation builds a fresh
            # one rather than re-showing a stale window.
            self.window = None

    # Pass no argv: our own flags are already parsed, and GApplication would
    # otherwise reject them.
    return LauncherApp().run([])


def main() -> int:
    parser = argparse.ArgumentParser(prog="keylane-launcher", description=__doc__)
    parser.add_argument(
        "--tray", action="store_true", help="run only the taskbar indicator"
    )
    parser.add_argument(
        "--no-tray", action="store_true", help="run the popup without the indicator"
    )
    parser.add_argument(
        "--toggle",
        action="store_true",
        help="show or hide the popup (bind this to Super+Space)",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="start resident without showing the popup (used by the service)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.tray:
        from launcher.tray import run_tray

        return run_tray()

    # --toggle just means "launch again": the single-instance lock forwards
    # the activation to the running popup, which flips its visibility.
    return run_launcher(
        with_tray=not args.no_tray and not args.toggle,
        background=args.background,
    )


if __name__ == "__main__":
    raise SystemExit(main())
