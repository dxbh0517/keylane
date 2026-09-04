"""Keep Keylane's panels on top, and put them where they belong.

Two very different worlds:

* **wlr-layer-shell** (Sway, Hyprland, river). A layer surface is genuinely
  always-on-top and can be anchored to a screen edge. Best case.
* **Everything else** — notably **GNOME/Mutter**, which has never implemented
  wlr-layer-shell, so ``Gtk4LayerShell.is_supported()`` is False there. A plain
  Wayland toplevel cannot ask to stay above other windows or place itself, so
  Keylane runs on **XWayland** instead and uses ``_NET_WM_STATE_ABOVE`` plus an
  explicit move — both of which Mutter honours for X11 clients.

In the X11 case each panel is a small window sized to its own content, so the
desktop around it is untouched: no fullscreen surface, and therefore nothing to
punch a click-through hole in.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

_WMCTRL = shutil.which("wmctrl")


def wayland_session() -> bool:
    return os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def forced_backend() -> str:
    """`KEYLANE_BACKEND=layer|x11|auto` overrides detection."""
    return os.environ.get("KEYLANE_BACKEND", "auto").strip().lower()


def surface_xid(window) -> int | None:
    surface = window.get_surface()
    if surface is None:
        return None
    getter = getattr(surface, "get_xid", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except Exception:  # noqa: BLE001
        return None


def _wmctrl(xid: int, *args: str) -> bool:
    if not _WMCTRL:
        return False
    try:
        done = subprocess.run(
            [_WMCTRL, "-i", "-r", str(xid), *args],
            capture_output=True,
            timeout=3,
            check=False,
        )
        return done.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def set_always_on_top(window) -> bool:
    """Ask the window manager to keep this window above the others."""
    xid = surface_xid(window)
    if xid is None:
        return False
    ok = _wmctrl(xid, "-b", "add,above")
    # Keep it out of the alt-tab list and the pager; it is a HUD, not an app.
    _wmctrl(xid, "-b", "add,skip_taskbar")
    _wmctrl(xid, "-b", "add,skip_pager")
    return ok


def move_resize(window, x: int, y: int, width: int, height: int) -> bool:
    """Place the window. Wayland forbids this; X11 allows it, which is why we use X11."""
    xid = surface_xid(window)
    if xid is None:
        return False
    return _wmctrl(xid, "-e", f"0,{int(x)},{int(y)},{int(width)},{int(height)}")


def centered_geometry(width: int, height: int, screen: tuple[int, int]) -> tuple[int, int]:
    """Top-left corner that centres a window of this size on the screen."""
    screen_w, screen_h = screen
    return max((screen_w - width) // 2, 0), max((screen_h - height) // 2, 0)


def scaled_geometry(
    x: int, y: int, width: int, height: int, scale: int
) -> tuple[int, int, int, int]:
    """Turn a logical rectangle into the one ``wmctrl -e`` wants.

    Both the position and the size are scaled. That is easy to talk yourself
    out of, so here is the trap, written down.

    ``wmctrl -lG`` **reports a position double the one actually in effect**,
    while reporting the size correctly. Ask it to put a window at 900 and it
    will report 1800 — and the window is at 900, where you asked for it. Take
    that readback as truth and the arithmetic says every window sits at twice
    the offset it should, which invites "fixing" it by dropping the scale
    factor from the position. Do that and the windows really do move to half
    the offset, into the top-left corner, which is how you find out the
    readback was lying and this line was right all along.

    Trust ``wmctrl -e`` going in. Do not trust ``wmctrl -lG``'s position coming
    back out. Verify placement with a screenshot, never with a query.
    """
    return x * scale, y * scale, width * scale, height * scale


def floating_geometry(
    mode: str,
    width: int,
    height: int,
    screen: tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    """Top-left corner for *mode*, in logical pixels, clamped to the screen."""
    if mode == "spotlight":
        return centered_geometry(width, height, screen)
    screen_w, _ = screen
    return max(screen_w - width - margin, 0), margin


def center_window(window, width: int, height: int, screen: tuple[int, int], scale: int = 1) -> bool:
    """Move an already-realized window to the middle of the screen.

    Wayland gives a client no say in its own position, which is why the
    non-layer-shell path runs on XWayland; there, wmctrl can place it. Under
    layer-shell the compositor owns placement and this is a no-op.
    """
    x, y = centered_geometry(width, height, screen)
    return move_resize(window, *scaled_geometry(x, y, width, height, scale))


def wmctrl_available() -> bool:
    return _WMCTRL is not None
