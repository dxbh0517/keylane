"""The working orb — a corner overlay that spins, then opens into an answer.

Submitting from the popup should not hold you hostage while a worker runs. The
popup closes, a small orb appears in a screen corner and spins, and when the
answer lands the orb grows into a squircle showing the canvas.

Motion follows Apple's fluid-interface rules:

* the panel **expands from the orb**, not from the screen centre, so the
  spatial relationship is obvious (enter and exit share one path);
* the growth is a critically damped spring — ``response`` 0.4, no overshoot,
  because nothing here carried momentum;
* it is **interruptible** — clicking or pressing Escape mid-expansion collapses
  from wherever the animation currently is, never after finishing first.

Placement is the same problem the popup has: GNOME will not let a client
position a window. So the overlay is a screen-sized transparent window whose
*input region* is clipped to the visible orb, letting every click outside it
reach whatever is underneath.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk, GLib, Gtk  # noqa: E402

from launcher.canvas_view import build_canvas  # noqa: E402

logger = logging.getLogger(__name__)

try:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell as LayerShell  # type: ignore

    HAVE_LAYER_SHELL = True
except (ValueError, ImportError):  # pragma: no cover - environment dependent
    LayerShell = None  # type: ignore
    HAVE_LAYER_SHELL = False


ORB_SIZE = 56
PANEL_WIDTH = 460
PANEL_MAX_HEIGHT = 560
EDGE_MARGIN = 24

# Apple's move/reposition spring: critically damped, response 0.4s.
SPRING_RESPONSE = 0.36
SPRING_FPS = 60

CORNERS = {
    "top-right": (Gtk.Align.END, Gtk.Align.START),
    "top-left": (Gtk.Align.START, Gtk.Align.START),
    "bottom-right": (Gtk.Align.END, Gtk.Align.END),
    "bottom-left": (Gtk.Align.START, Gtk.Align.END),
    "center": (Gtk.Align.CENTER, Gtk.Align.CENTER),
}


def _critically_damped(t: float) -> float:
    """Unit step response of a critically damped spring, 0 → 1.

    ``1 - (1 + wt)e^(-wt)`` — settles without overshoot, which is what Apple
    prescribes for motion the user did not throw.
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    w = 6.0  # reaches ~99% at t = 1
    x = w * t
    return 1.0 - (1.0 + x) * math.exp(-x)


class ResultOrb(Gtk.ApplicationWindow):
    """A corner overlay: spinner while working, canvas when done."""

    def __init__(
        self,
        app: Gtk.Application,
        *,
        corner: str = "top-right",
        on_open_link: Callable[[str], None] | None = None,
        on_reopen: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(application=app, title="Keylane result")
        self.corner = corner if corner in CORNERS else "top-right"
        self._on_open_link = on_open_link
        self._on_reopen = on_reopen

        self._expanded = False
        self._progress = 0.0        # 0 = orb, 1 = full panel
        self._target = 0.0
        self._tick: int | None = None
        self._canvas: dict[str, Any] | None = None
        self._actions: Gtk.Widget | None = None
        self._status = "Working…"

        self.set_decorated(False)
        self.set_resizable(False)
        self.set_hide_on_close(True)
        self.add_css_class("keylane-popup")
        self.add_css_class("keylane-orb-window")

        self._setup_layer_shell()
        self._build()

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)
        self.connect("map", lambda *_: GLib.idle_add(self._update_input_region))

    # ------------------------------------------------------------- placement

    def _setup_layer_shell(self) -> None:
        # Set first: every later branch may return early, and _build() reads it.
        self._layer_shell_active = False
        if not HAVE_LAYER_SHELL:
            return
        try:
            if not LayerShell.is_supported():  # type: ignore[union-attr]
                return
            LayerShell.init_for_window(self)  # type: ignore[union-attr]
            LayerShell.set_layer(self, LayerShell.Layer.OVERLAY)  # type: ignore[union-attr]
            LayerShell.set_namespace(self, "keylane-result")  # type: ignore[union-attr]
            LayerShell.set_keyboard_mode(  # type: ignore[union-attr]
                self, LayerShell.KeyboardMode.ON_DEMAND
            )
            top = self.corner.startswith("top")
            right = self.corner.endswith("right")
            LayerShell.set_anchor(self, LayerShell.Edge.TOP, top)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.BOTTOM, not top)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.RIGHT, right)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.LEFT, not right)  # type: ignore[union-attr]
            for edge in (
                LayerShell.Edge.TOP,
                LayerShell.Edge.BOTTOM,
                LayerShell.Edge.LEFT,
                LayerShell.Edge.RIGHT,
            ):
                LayerShell.set_margin(self, edge, EDGE_MARGIN)  # type: ignore[union-attr]
            self._layer_shell_active = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("layer-shell unavailable for the orb: %s", exc)
            self._layer_shell_active = False

    def _screen_size(self) -> tuple[int, int]:
        display = Gdk.Display.get_default()
        if display is None:
            return 1920, 1080
        monitors = display.get_monitors()
        monitor = monitors.get_item(0) if monitors.get_n_items() else None
        if monitor is None:
            return 1920, 1080
        geometry = monitor.get_geometry()
        return geometry.width, geometry.height

    # ---------------------------------------------------------------- build

    def _build(self) -> None:
        self._shell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self._shell.add_css_class("keylane-shell")
        self._shell.add_css_class("keylane-result-shell")

        # Header: spinner or title, always present so the two states share a row.
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        header.set_valign(Gtk.Align.CENTER)

        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(20, 20)
        self._spinner.set_valign(Gtk.Align.CENTER)
        header.append(self._spinner)

        self._status_label = Gtk.Label(label=self._status)
        self._status_label.set_xalign(0.0)
        self._status_label.set_hexpand(True)
        self._status_label.add_css_class("keylane-progress")
        self._status_label.set_ellipsize(3)  # END
        header.append(self._status_label)

        self._close_btn = Gtk.Button()
        self._close_btn.set_icon_name("window-close-symbolic")
        self._close_btn.add_css_class("keylane-icon-btn")
        self._close_btn.set_valign(Gtk.Align.CENTER)
        self._close_btn.set_visible(False)
        self._close_btn.connect("clicked", lambda *_: self.dismiss())
        header.append(self._close_btn)

        self._header = header
        self._shell.append(header)

        self._body = Gtk.ScrolledWindow()
        self._body.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._body.set_propagate_natural_height(True)
        self._body.set_visible(False)
        self._body.add_css_class("keylane-result-view")
        self._shell.append(self._body)

        halign, valign = CORNERS[self.corner]
        self._shell.set_halign(halign)
        self._shell.set_valign(valign)

        if self._layer_shell_active:
            self.set_child(self._shell)
        else:
            # No layer shell: cover the screen with a transparent window and
            # anchor the shell to the requested corner. The input region below
            # keeps the rest of the surface click-through.
            width, height = self._screen_size()
            self.set_default_size(width, height)
            pad = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            pad.set_margin_top(EDGE_MARGIN)
            pad.set_margin_bottom(EDGE_MARGIN)
            pad.set_margin_start(EDGE_MARGIN)
            pad.set_margin_end(EDGE_MARGIN)
            pad.append(self._shell)
            self._shell.set_vexpand(False)
            pad.set_halign(Gtk.Align.FILL)
            pad.set_valign(Gtk.Align.FILL)
            self.set_child(pad)

        self._apply_geometry()

    # ------------------------------------------------------------- geometry

    def _apply_geometry(self) -> None:
        """Size the shell for the current point in the orb → panel animation."""
        t = self._progress
        width = int(ORB_SIZE + (PANEL_WIDTH - ORB_SIZE) * t)
        self._shell.set_size_request(width, -1)

        collapsed = t < 0.02
        self._shell.set_hexpand(False)
        if collapsed:
            self._shell.set_size_request(ORB_SIZE, ORB_SIZE)
            self._shell.add_css_class("is-orb")
        else:
            self._shell.remove_css_class("is-orb")

        self._status_label.set_visible(not collapsed)
        self._close_btn.set_visible(t > 0.9 and self._canvas is not None)
        self._body.set_visible(t > 0.55 and self._canvas is not None)
        self._body.set_opacity(max(0.0, min((t - 0.55) / 0.45, 1.0)))
        self._shell.set_opacity(0.35 + 0.65 * min(t * 4, 1.0) if t < 0.25 else 1.0)
        GLib.idle_add(self._update_input_region)

    def _animate_to(self, target: float) -> None:
        """Spring the shell towards ``target``, interruptibly.

        Starts from the *current* value, so grabbing a half-open panel and
        closing it never jumps.
        """
        self._target = target
        if self._tick is not None:
            return  # a frame loop is already running; it will chase the new target

        start = self._progress
        elapsed = 0.0
        step = 1.0 / SPRING_FPS

        def frame() -> bool:
            nonlocal start, elapsed
            elapsed += step
            fraction = _critically_damped(elapsed / SPRING_RESPONSE)
            self._progress = start + (self._target - start) * fraction
            self._apply_geometry()
            if fraction >= 0.999:
                self._progress = self._target
                self._apply_geometry()
                self._tick = None
                return False
            return True

        self._tick = GLib.timeout_add(int(1000 / SPRING_FPS), frame)

    # ----------------------------------------------------------------- API

    def start(self, message: str) -> None:
        """Show the orb, spinning, for a request that has just been sent."""
        self._canvas = None
        self._clear_actions()
        self._expanded = False
        self._progress = 0.0
        self._status = message[:80] or "Working…"
        self._status_label.set_text(self._status)
        self._set_body(None)
        self._apply_geometry()
        self._spinner.start()
        self.present()

    def set_status(self, text: str) -> None:
        self._status = text[:120]
        self._status_label.set_text(self._status)

    def show_result(
        self, canvas: dict[str, Any] | None, *, title: str = "", failed: bool = False
    ) -> None:
        """Expand the orb into the answer."""
        self._spinner.stop()
        self._clear_actions()
        self._canvas = canvas
        self._status_label.set_text(
            title or (canvas or {}).get("title") or ("Failed" if failed else "Done")
        )
        if failed:
            self._shell.add_css_class("failed")
        else:
            self._shell.remove_css_class("failed")
        self._set_body(canvas)
        self._expanded = True
        self._animate_to(1.0)
        self.present()

    def show_confirmation(
        self,
        data: dict[str, Any],
        *,
        on_allow: Callable[[], None],
        on_cancel: Callable[[], None],
    ) -> None:
        """Expand with an Allow / Cancel choice instead of an answer.

        A gated tool has to be approvable without reopening the bar, or the
        hand-off to the orb would just lose the task.
        """
        self._spinner.stop()
        tool = data.get("pending_tool") or "this action"
        arguments = data.get("pending_arguments") or {}
        self._status_label.set_text("Approval needed")

        canvas: dict[str, Any] = {
            "blocks": [
                {"type": "text", "text": f"Keylane wants to run {tool}."},
            ]
        }
        if arguments:
            canvas["blocks"].append(
                {
                    "type": "table",
                    "columns": ["Argument", "Value"],
                    "rows": [[str(k), str(v)] for k, v in arguments.items()],
                }
            )
        self._canvas = canvas
        self._set_body(canvas)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.set_halign(Gtk.Align.END)
        cancel = Gtk.Button(label="Cancel")
        cancel.connect("clicked", lambda *_: (on_cancel(), self._clear_actions()))
        row.append(cancel)
        allow = Gtk.Button(label="Allow")
        allow.add_css_class("suggested-action")
        allow.connect("clicked", lambda *_: (self._clear_actions(), on_allow()))
        row.append(allow)

        self._clear_actions()
        self._actions = row
        self._shell.append(row)

        self._expanded = True
        self._animate_to(1.0)
        self.present()

    def _clear_actions(self) -> None:
        actions = getattr(self, "_actions", None)
        if actions is not None:
            self._shell.remove(actions)
            self._actions = None

    def _set_body(self, canvas: dict[str, Any] | None) -> None:
        widget = build_canvas(canvas, on_open=self._on_open_link)
        self._body.set_child(widget)
        self._body.set_max_content_height(PANEL_MAX_HEIGHT)

    def dismiss(self) -> None:
        """Collapse and hide. Interruptible at any point in the animation."""
        self._spinner.stop()
        self._expanded = False
        self._animate_to(0.0)

        def hide() -> bool:
            if not self._expanded:
                self.set_visible(False)
                self._progress = 0.0
                self._apply_geometry()
            return False

        GLib.timeout_add(int(SPRING_RESPONSE * 1000) + 40, hide)

    # -------------------------------------------------------------- events

    def _on_key(self, _controller, keyval, _keycode, _state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.dismiss()
            return True
        return False

    def _update_input_region(self) -> None:
        """Only the visible shell should catch clicks."""
        surface = self.get_surface()
        if surface is None:
            return
        try:
            import cairo

            ok, rect = self._shell.compute_bounds(self)
            if not ok:
                return
            region = cairo.Region(
                cairo.RectangleInt(
                    int(rect.origin.x),
                    int(rect.origin.y),
                    max(int(rect.size.width), 1),
                    max(int(rect.size.height), 1),
                )
            )
            surface.set_input_region(region)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not clip the orb input region: %s", exc)
