"""The surface Keylane draws on, over everything else.

One window, two jobs, and they are inverses of each other:

* **annotating** — Keylane paints rings and arrows to show you where to click.
  The window is entirely click-through, so the desktop underneath keeps
  working while the marks sit on top of it.
* **pointing** — you drag a stroke around something to say "this". The window
  takes input for exactly as long as that gesture, then gives it back.

Click-through is the whole trick, and it is the same one the Spotlight surface
already uses: an *empty* input region on a fullscreen surface means every click
falls through to whatever is beneath. Emptying it is not optional — a fullscreen
overlay without it makes the entire desktop unreachable.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

import cairo
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk, GLib, Gtk  # type: ignore[attr-defined]

from ui.annotations import (
    Annotation,
    Shape,
    arrow_head,
    box_from_points,
    from_pixels,
    to_pixels,
    wobbly_circle,
    wobbly_line,
    wobbly_rect,
)
from ui.layers import make_fullscreen_layer, monitor_size
from ui.theme import token_rgb

logger = logging.getLogger(__name__)

FADE_MS = 260
LINE_WIDTH = 3.5
LABEL_PAD = 7
FRAME_MS = 33

# Fallbacks only; the live values come from the active theme's orb tokens, the
# same ones the thinking orb paints itself with.
ACCENT = (0.36, 0.82, 1.00)
ACCENT_ALT = (0.66, 0.52, 1.00)


class AnnotationOverlay(Gtk.ApplicationWindow):
    """A full-screen canvas that is invisible to the mouse unless asked."""

    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app)
        self.set_decorated(False)
        self.set_deletable(False)
        self.add_css_class("annotation-overlay")

        self._annotation: Annotation | None = None
        self._shown_at = 0.0
        self._expire_id = 0
        self._frame_id = 0
        self._interactive = False
        self._stroke: list[tuple[float, float]] = []
        self._on_region: Callable[[tuple[float, float, float, float]], None] | None = None

        self._layered = make_fullscreen_layer(self, interactive=False)
        if not self._layered:
            width, height = monitor_size()
            self.set_default_size(width, height)
            self.set_resizable(False)

        self._canvas = Gtk.DrawingArea()
        self._canvas.set_hexpand(True)
        self._canvas.set_vexpand(True)
        self._canvas.set_draw_func(self._draw, None)
        self.set_child(self._canvas)

        drag = Gtk.GestureDrag.new()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self._canvas.add_controller(drag)

        key = Gtk.EventControllerKey.new()
        key.connect("key-released", self._on_key)
        self.add_controller(key)

        self.connect("map", lambda *_: GLib.idle_add(self._apply_input_region))
        self.set_visible(False)

    # ── click-through ────────────────────────────────────────────────────

    def _apply_input_region(self) -> bool:
        """Empty region while annotating; the whole surface while pointing."""
        if not self.get_realized():
            return False
        surface = self.get_surface()
        if surface is None:
            return False
        if self._interactive:
            surface.set_input_region(None)
        else:
            # An empty region, not a small one: there is nothing here to click.
            surface.set_input_region(cairo.Region())
        return False

    def _raise_above(self) -> None:
        """On X11 the window manager has to be told; layer shell handles itself."""
        if self._layered:
            return
        from ui.placement import move_resize, scaled_geometry, set_always_on_top

        set_always_on_top(self)
        width, height = monitor_size()
        scale = self.get_scale_factor() or 1
        move_resize(self, *scaled_geometry(0, 0, width, height, scale))

    # ── showing marks ────────────────────────────────────────────────────

    def show_annotation(self, annotation: Annotation) -> None:
        """Paint *annotation*, replacing whatever was on screen."""
        self._cancel_expiry()
        if not annotation.shapes and not annotation.caption:
            self.clear()
            return

        self._annotation = annotation
        self._shown_at = time.monotonic()
        self._interactive = False
        self.set_visible(True)
        self.present()
        GLib.idle_add(self._apply_input_region)
        GLib.idle_add(self._raise_above)
        self._start_frames()

        if not annotation.persistent:
            self._expire_id = GLib.timeout_add(annotation.ttl_ms, self._expire)

    def clear(self) -> None:
        self._cancel_expiry()
        self._stop_frames()
        self._annotation = None
        self._stroke = []
        self._interactive = False
        self.set_visible(False)

    def _expire(self) -> bool:
        self._expire_id = 0
        self.clear()
        return False

    def _cancel_expiry(self) -> None:
        if self._expire_id:
            GLib.source_remove(self._expire_id)
            self._expire_id = 0

    def _start_frames(self) -> None:
        # Only for the fade-in; a static annotation does not need a timer.
        if self._frame_id:
            return
        self._frame_id = GLib.timeout_add(FRAME_MS, self._tick)

    def _stop_frames(self) -> None:
        if self._frame_id:
            GLib.source_remove(self._frame_id)
            self._frame_id = 0

    def _tick(self) -> bool:
        self._canvas.queue_draw()
        if time.monotonic() - self._shown_at > FADE_MS / 1000.0 and not self._interactive:
            self._frame_id = 0
            return False
        return True

    # ── pointing ─────────────────────────────────────────────────────────

    def pick_region(self, on_region: Callable[[tuple[float, float, float, float]], None]) -> None:
        """Take the pointer until the user has drawn one stroke.

        The callback gets the stroke's bounding box in grid units, which is
        what a small vision model can actually use — the crop matters far more
        than the shape the user drew around it.
        """
        self._cancel_expiry()
        self._annotation = Annotation(shapes=[], ttl_ms=-1, caption="Circle what you mean")
        self._on_region = on_region
        self._stroke = []
        self._interactive = True
        self._shown_at = time.monotonic()
        self.set_visible(True)
        self.present()
        GLib.idle_add(self._apply_input_region)
        GLib.idle_add(self._raise_above)
        self._start_frames()

    def _on_drag_begin(self, gesture: Gtk.GestureDrag, x: float, y: float) -> None:
        if not self._interactive:
            return
        self._stroke = [(x, y)]
        self._canvas.queue_draw()

    def _on_drag_update(self, gesture: Gtk.GestureDrag, dx: float, dy: float) -> None:
        if not self._interactive or not self._stroke:
            return
        ok, start_x, start_y = gesture.get_start_point()
        if not ok:
            return
        self._stroke.append((start_x + dx, start_y + dy))
        self._canvas.queue_draw()

    def _on_drag_end(self, gesture: Gtk.GestureDrag, dx: float, dy: float) -> None:
        if not self._interactive:
            return
        callback, stroke = self._on_region, list(self._stroke)
        self._on_region = None
        self.clear()
        if callback is None or len(stroke) < 2:
            return

        width = max(self._canvas.get_width(), 1)
        height = max(self._canvas.get_height(), 1)
        x, y, w, h = box_from_points(stroke)
        gx, gy = from_pixels(x, y, width, height)
        gx2, gy2 = from_pixels(x + w, y + h, width, height)
        callback((gx, gy, gx2 - gx, gy2 - gy))

    def _on_key(self, _ctrl, keyval: int, _keycode: int, _state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self._on_region = None
            self.clear()
            return True
        return False

    # ── painting ─────────────────────────────────────────────────────────

    def _colors(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return token_rgb("orb-accent", ACCENT), token_rgb("orb-accent-alt", ACCENT_ALT)

    def _draw(self, _area: Gtk.DrawingArea, ctx: cairo.Context, width: int, height: int, _data) -> None:
        annotation = self._annotation
        if annotation is None:
            return

        elapsed = time.monotonic() - self._shown_at
        alpha = min(1.0, elapsed / (FADE_MS / 1000.0)) if FADE_MS else 1.0
        accent, alt = self._colors()

        ctx.set_line_width(LINE_WIDTH)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        for shape in annotation.shapes:
            self._draw_shape(ctx, to_pixels(shape, width, height), accent, alt, alpha)

        if self._stroke:
            self._draw_stroke(ctx, alt, alpha)

        if annotation.caption:
            self._draw_caption(ctx, annotation.caption, width, height, accent, alpha)

    def _stroke_path(
        self,
        ctx: cairo.Context,
        points: list[tuple[float, float]],
        color: tuple[float, float, float],
        alpha: float,
        *,
        glow: bool = True,
    ) -> None:
        """Draw a path twice: a soft wide pass, then the line itself.

        The halo is what keeps a cyan stroke visible over a light window; a
        single hairline disappears against half the desktops it lands on.
        """
        if len(points) < 2:
            return
        for width_mult, alpha_mult in ((3.0, 0.22), (1.0, 1.0)) if glow else ((1.0, 1.0),):
            ctx.set_line_width(LINE_WIDTH * width_mult)
            ctx.set_source_rgba(*color, alpha * alpha_mult)
            ctx.move_to(*points[0])
            for point in points[1:]:
                ctx.line_to(*point)
            ctx.stroke()
        ctx.set_line_width(LINE_WIDTH)

    def _draw_shape(
        self,
        ctx: cairo.Context,
        shape: Shape,
        accent: tuple[float, float, float],
        alt: tuple[float, float, float],
        alpha: float,
    ) -> None:
        seed = shape.seed
        if shape.kind == "ring":
            self._stroke_path(ctx, wobbly_circle(shape.x, shape.y, shape.r, seed), accent, alpha)
            self._label_near(ctx, shape.text, shape.x, shape.y - shape.r, accent, alpha)
        elif shape.kind == "box":
            self._stroke_path(ctx, wobbly_rect(shape.x, shape.y, shape.w, shape.h, seed), accent, alpha)
            self._label_near(ctx, shape.text, shape.x + shape.w / 2, shape.y, accent, alpha)
        elif shape.kind == "arrow":
            self._stroke_path(
                ctx, wobbly_line(shape.x, shape.y, shape.x2, shape.y2, seed), accent, alpha
            )
            self._stroke_path(
                ctx, arrow_head(shape.x, shape.y, shape.x2, shape.y2), accent, alpha, glow=False
            )
            self._label_near(ctx, shape.text, shape.x, shape.y, accent, alpha)
        elif shape.kind == "label":
            self._label_near(ctx, shape.text, shape.x, shape.y, alt, alpha, anchor="left")

    def _draw_stroke(self, ctx: cairo.Context, color: tuple[float, float, float], alpha: float) -> None:
        self._stroke_path(ctx, self._stroke, color, alpha)

    def _text_extents(self, ctx: cairo.Context, text: str, size: float) -> tuple[float, float]:
        ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(size)
        extents = ctx.text_extents(text)
        return extents.width, extents.height

    def _plate(
        self,
        ctx: cairo.Context,
        text: str,
        x: float,
        y: float,
        color: tuple[float, float, float],
        alpha: float,
        size: float,
    ) -> None:
        """A caption on its own dark plate, because it lands on unknown pixels."""
        width, height = self._text_extents(ctx, text, size)
        left = x - LABEL_PAD
        top = y - height - LABEL_PAD
        box_w = width + LABEL_PAD * 2
        box_h = height + LABEL_PAD * 2

        radius = 6.0
        ctx.new_path()
        ctx.arc(left + radius, top + radius, radius, math.pi, 1.5 * math.pi)
        ctx.arc(left + box_w - radius, top + radius, radius, 1.5 * math.pi, 0)
        ctx.arc(left + box_w - radius, top + box_h - radius, radius, 0, 0.5 * math.pi)
        ctx.arc(left + radius, top + box_h - radius, radius, 0.5 * math.pi, math.pi)
        ctx.close_path()
        ctx.set_source_rgba(0.06, 0.07, 0.10, 0.88 * alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*color, 0.55 * alpha)
        ctx.set_line_width(1.0)
        ctx.stroke()

        ctx.set_source_rgba(*color, alpha)
        ctx.move_to(x, y)
        ctx.show_text(text)
        ctx.set_line_width(LINE_WIDTH)

    def _label_near(
        self,
        ctx: cairo.Context,
        text: str,
        x: float,
        y: float,
        color: tuple[float, float, float],
        alpha: float,
        *,
        anchor: str = "center",
    ) -> None:
        if not text:
            return
        size = 15.0
        width, _height = self._text_extents(ctx, text, size)
        left = x if anchor == "left" else x - width / 2
        # Keep the plate on screen when the target sits near the top edge.
        top = max(y - 12, 34.0)
        self._plate(ctx, text, left, top, color, alpha, size)

    def _draw_caption(
        self,
        ctx: cairo.Context,
        text: str,
        width: int,
        height: int,
        color: tuple[float, float, float],
        alpha: float,
    ) -> None:
        size = 17.0
        text_width, _ = self._text_extents(ctx, text, size)
        self._plate(ctx, text, (width - text_width) / 2, height - 48.0, color, alpha, size)
