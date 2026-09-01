"""Animated activity indicator.

One widget serves two jobs: the large free-floating orb the spotlight
collapses into while working, and — at a smaller size — the inline indicator
in the answer panel's header. Using the same drawing for both is what makes
the collapse read as one object moving rather than two unrelated widgets.
"""

from __future__ import annotations

import math

from gi.repository import GLib, Gtk  # type: ignore[attr-defined]

# Shared accent ramp — keep in step with --kl-accent* in spotlight.css.
ACCENT = (0.36, 0.82, 1.00)      # cyan
ACCENT_ALT = (0.66, 0.52, 1.00)  # violet
IDLE = (0.55, 0.62, 0.75)


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class ThinkingOrb(Gtk.DrawingArea):
    """A pulsing orb. `set_state("done")` settles it instead of stopping dead."""

    ORB_SIZE = 56
    SETTLED_FLOOR = 0.28

    def __init__(self, size: int | None = None) -> None:
        super().__init__()
        self.add_css_class("thinking-orb")
        self._size = size or self.ORB_SIZE
        self.set_content_width(self._size)
        self.set_content_height(self._size)
        self.set_tooltip_text("Keylane is thinking…")
        self._phase = 0.0
        self._state = "thinking"
        # 1 = fully active. It settles to SETTLED_FLOOR, not 0: at zero the orb
        # reads as an empty grey circle rather than a finished indicator.
        self._settle = 1.0
        self._tick_id = 0
        self.set_draw_func(self._draw, None)
        self.connect("map", self._on_map)
        self.connect("unmap", self._on_unmap)

    # ── lifecycle ────────────────────────────────────────────────────────

    def _animations_enabled(self) -> bool:
        settings = Gtk.Settings.get_default()
        if settings is None:
            return True
        try:
            return bool(settings.get_property("gtk-enable-animations"))
        except (TypeError, ValueError):
            return True

    def _on_map(self, *_args: object) -> None:
        if not self._tick_id and self._animations_enabled():
            self._tick_id = GLib.timeout_add(33, self._animate)

    def _on_unmap(self, *_args: object) -> None:
        if self._tick_id:
            GLib.source_remove(self._tick_id)
            self._tick_id = 0

    def _animate(self) -> bool:
        self._phase += 0.10
        if self._state == "done" and self._settle > self.SETTLED_FLOOR:
            self._settle = max(self.SETTLED_FLOOR, self._settle - 0.045)
        self.queue_draw()
        return True

    def set_state(self, state: str) -> None:
        """`thinking` spins; `done` eases down to a calm filled dot."""
        if state == self._state:
            return
        self._state = state
        if state == "thinking":
            self._settle = 1.0
        self.queue_draw()

    # ── drawing ──────────────────────────────────────────────────────────

    def _draw(self, _area: Gtk.DrawingArea, cr, width: int, height: int, _data: object) -> None:
        cx, cy = width / 2, height / 2
        radius = min(width, height) / 2 - max(3.0, self._size * 0.09)
        if radius <= 2:
            return

        scale = self._size / self.ORB_SIZE
        active = self._settle
        accent = _lerp(IDLE, ACCENT, active)
        alt = _lerp(IDLE, ACCENT_ALT, active)

        cr.push_group()

        # Outer bloom — the "glass over anything" halo that keeps the orb
        # readable on a light wallpaper as well as a dark one.
        breathe = 0.5 + 0.5 * math.sin(self._phase * 1.9)
        for i, (spread, alpha) in enumerate(((5.5, 0.16), (3.0, 0.22))):
            cr.arc(cx, cy, radius + spread * scale * (0.7 + 0.3 * breathe), 0, 2 * math.pi)
            cr.set_source_rgba(*accent, alpha * (0.45 + 0.55 * active))
            cr.fill()

        # Core disk
        cr.arc(cx, cy, radius - 1.5 * scale, 0, 2 * math.pi)
        cr.set_source_rgba(0.05, 0.06, 0.09, 0.92)
        cr.fill()

        # Rim light
        cr.arc(cx, cy, radius - 1.5 * scale, 0, 2 * math.pi)
        cr.set_line_width(1.1 * scale)
        cr.set_source_rgba(*accent, 0.30 + 0.25 * active)
        cr.stroke()

        spinning = self._state == "thinking"
        if spinning:
            cr.set_line_cap(1)  # ROUND

            # Primary sweep
            span = math.pi * 1.3
            cr.arc(cx, cy, radius - 0.5 * scale, self._phase, self._phase + span)
            cr.set_line_width(2.3 * scale)
            cr.set_source_rgba(*ACCENT, 0.95)
            cr.stroke()

            # Counter-rotating accent, slightly inside
            start = -self._phase * 1.55 + math.pi
            cr.arc(cx, cy, radius - 4.5 * scale, start, start + span * 0.5)
            cr.set_line_width(1.4 * scale)
            cr.set_source_rgba(*ACCENT_ALT, 0.80)
            cr.stroke()

        if not spinning:
            cr.arc(cx, cy, radius - 0.5 * scale, 0, 2 * math.pi)
            cr.set_line_width(1.8 * scale)
            cr.set_source_rgba(*ACCENT, 0.55)
            cr.stroke()

        # Centre pulse — settles into a steady dot when done.
        wobble = 1.1 * math.sin(self._phase * 2.8) if spinning else 0.0
        pulse = (2.4 + wobble) * scale
        cr.arc(cx, cy, max(pulse, 2.0 * scale), 0, 2 * math.pi)
        cr.set_source_rgba(*_lerp(alt, ACCENT, 0.5), 0.75 + 0.2 * active)
        cr.fill()

        cr.pop_group_to_source()
        cr.paint()
