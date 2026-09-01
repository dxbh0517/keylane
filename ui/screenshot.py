"""Screen capture.

Two backends, tried in order:

1. The XDG desktop portal (``org.freedesktop.portal.Screenshot``). This is the
   only route that works on GNOME/Mutter, which does not implement the
   wlr-screencopy protocol — ``grim`` there fails with "compositor doesn't
   support the screen capture protocol".
2. ``grim`` + ``slurp``, for wlroots compositors (Sway, Hyprland, river) where
   the portal may not be installed.

Both are synchronous and safe to call from a worker thread.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

_PORTAL_TIMEOUT = 120


def _tempfile() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        return Path(tmp.name)


# ── XDG portal ───────────────────────────────────────────────────────────


def _portal_screenshot(interactive: bool) -> Path | None:
    """Ask the desktop portal for a screenshot; returns a copy we own."""
    try:
        import gi

        gi.require_version("Gio", "2.0")
        from gi.repository import Gio, GLib
    except (ImportError, ValueError):
        return None

    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except GLib.Error:
        return None

    result: dict[str, object] = {}
    loop = GLib.MainLoop()

    def _on_response(_conn, _sender, _path, _iface, _signal, params) -> None:
        response, results = params.unpack()
        result["response"] = response
        result["uri"] = (results or {}).get("uri", "")
        loop.quit()

    subscription = bus.signal_subscribe(
        "org.freedesktop.portal.Desktop",
        "org.freedesktop.portal.Request",
        "Response",
        None,
        None,
        Gio.DBusSignalFlags.NONE,
        _on_response,
    )
    try:
        bus.call_sync(
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.Screenshot",
            "Screenshot",
            GLib.Variant(
                "(sa{sv})",
                ("", {"interactive": GLib.Variant("b", interactive)}),
            ),
            GLib.VariantType("(o)"),
            Gio.DBusCallFlags.NONE,
            _PORTAL_TIMEOUT * 1000,
            None,
        )

        # Bail out rather than hang forever if the dialog is never answered.
        deadline = time.monotonic() + _PORTAL_TIMEOUT
        GLib.timeout_add_seconds(1, lambda: loop.quit() or False if time.monotonic() > deadline else True)
        loop.run()
    except GLib.Error:
        logger.info("screenshot portal unavailable", exc_info=True)
        return None
    finally:
        bus.signal_unsubscribe(subscription)

    if result.get("response") != 0:
        return None  # user cancelled
    uri = str(result.get("uri") or "")
    if not uri:
        return None

    source = Path(unquote(urlparse(uri).path))
    if not source.is_file():
        return None

    # The portal's file lives in its own cache; copy it somewhere we can delete.
    out = _tempfile()
    try:
        out.write_bytes(source.read_bytes())
    except OSError:
        out.unlink(missing_ok=True)
        return None
    return out


# ── grim / slurp (wlroots) ───────────────────────────────────────────────


def _grim(args: list[str]) -> Path | None:
    if not shutil.which("grim"):
        return None
    out = _tempfile()
    try:
        shot = subprocess.run(["grim", *args, str(out)], capture_output=True, timeout=30, check=False)
        if shot.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            return None
        return out
    except (OSError, subprocess.TimeoutExpired):
        out.unlink(missing_ok=True)
        return None


def _slurp_region() -> str | None:
    if not shutil.which("slurp"):
        return None
    try:
        area = subprocess.run(["slurp"], capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return area.stdout.strip() if area.returncode == 0 and area.stdout.strip() else None


# ── public API ───────────────────────────────────────────────────────────


def capture_region() -> Path | None:
    """Let the user pick an area. Returns a PNG path, or None if cancelled."""
    geometry = _slurp_region()
    if geometry:
        shot = _grim(["-g", geometry])
        if shot:
            return shot
    # The portal's interactive mode covers region selection on GNOME.
    return _portal_screenshot(interactive=True)


def capture_fullscreen() -> Path | None:
    """Capture the whole screen. Returns a PNG path, or None on failure."""
    return _grim([]) or _portal_screenshot(interactive=False)
