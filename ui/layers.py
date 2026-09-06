"""Is this compositor one where a client can own the screen?

Two windows now need the answer — the Spotlight surface and the annotation
overlay — and the check is subtle enough that two copies of it would drift:
``LayerShell.is_supported()`` asserts ``GDK_IS_WAYLAND_DISPLAY`` internally, so
calling it on X11 does not return False, it aborts the process.

Everything here is safe to call on any backend.
"""

from __future__ import annotations

import logging

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

try:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell as LayerShell  # type: ignore[attr-defined]

    HAS_LAYER_SHELL = True
except (ImportError, ValueError):  # pragma: no cover - depends on the host
    LayerShell = None  # type: ignore[assignment]
    HAS_LAYER_SHELL = False


def layer_shell_ok() -> bool:
    """True only on a Wayland display whose compositor speaks wlr-layer-shell."""
    if not HAS_LAYER_SHELL:
        return False
    display = Gdk.Display.get_default()
    # Never call is_supported() on X11: it asserts on the display type.
    if display is None or "Wayland" not in type(display).__name__:
        return False
    try:
        return bool(LayerShell.is_supported())
    except Exception:  # noqa: BLE001
        return False


def make_fullscreen_layer(window, *, interactive: bool = False) -> bool:
    """Anchor *window* to all four edges of the overlay layer.

    Returns False when this is not a layer-shell session, which is the caller's
    cue to fall back to a plain always-on-top window.
    """
    if not layer_shell_ok():
        return False
    try:
        LayerShell.init_for_window(window)
        if not LayerShell.is_layer_window(window):
            return False
        LayerShell.set_layer(window, LayerShell.Layer.OVERLAY)
        LayerShell.set_keyboard_mode(
            window,
            LayerShell.KeyboardMode.ON_DEMAND if interactive else LayerShell.KeyboardMode.NONE,
        )
        for edge in (
            LayerShell.Edge.TOP,
            LayerShell.Edge.BOTTOM,
            LayerShell.Edge.LEFT,
            LayerShell.Edge.RIGHT,
        ):
            LayerShell.set_anchor(window, edge, True)
            LayerShell.set_margin(window, edge, 0)
        # -1: do not reserve space, and do not let other surfaces push us.
        LayerShell.set_exclusive_zone(window, -1)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("layer shell init failed for the overlay", exc_info=True)
        return False


def monitor_size() -> tuple[int, int]:
    display = Gdk.Display.get_default()
    if display is None:
        return 1920, 1080
    monitors = display.get_monitors()
    monitor = monitors.get_item(0) if monitors.get_n_items() > 0 else None
    if monitor is None:
        return 1920, 1080
    geom = monitor.get_geometry()
    return geom.width, geom.height
