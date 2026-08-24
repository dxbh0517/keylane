"""The working indicator — three orbiting arcs drawn with Cairo.

``Gtk.Spinner`` is the platform's generic "something is happening" glyph. It is
fine, and it is anonymous: it tells you nothing about *what* is working, and it
sits wherever the theme puts it.

This draws instead. Three concentric arcs rotate at different rates around a
pulsing core — reading as a control plane routing work, which is what Keylane
is actually doing. It is deliberately restrained: one accent colour at varying
alpha, no rainbow, no glow, no bounce.

Motion follows the same rules as the rest of the popup:

* rotation is **continuous and linear** — a spinner that eases is a spinner
  that looks broken;
* the arcs run at coprime rates so the figure never visibly repeats;
* it stops when hidden, because an off-screen animation is a wakeup for
  nothing;
* ``prefers-reduced-motion`` replaces rotation with a slow opacity breath.
"""

from __future__ import annotations

import math
from typing import Any

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import Gdk, GLib, Gtk  # noqa: E402

# Arc geometry as a fraction of the widget's radius, with turns per second.
# The rates are deliberately unequal and non-harmonic.
ARCS = (
    # (radius, sweep in turns, revolutions/sec, line width, alpha)
    (0.92, 0.62, 0.55, 0.10, 0.95),
    (0.68, 0.44, -0.85, 0.09, 0.62),
    (0.44, 0.30, 1.35, 0.08, 0.38),
)
CORE_RADIUS = 0.16
CORE_PULSE_HZ = 0.7
FPS = 60


class OrbitLoader(Gtk.DrawingArea):
    """A compact, self-contained activity indicator."""

    def __init__(self, size: int = 26) -> None:
        super().__init__()
        self._size = size
        self._phase = 0.0
        self._tick: int | None = None
        self._colour: tuple[float, float, float] = (0.31, 0.55, 1.0)

        self.set_content_width(size)
        self.set_content_height(size)
        # Centred in both axes: the old spinner sat top-aligned in its row,
        # which is what put it off-centre inside the collapsed orb.
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.set_draw_func(self._draw)
        self.connect("map", lambda *_: self.start())
        self.connect("unmap", lambda *_: self.stop())

    # ----------------------------------------------------------------- api

    def set_size(self, size: int) -> None:
        self._size = size
        self.set_content_width(size)
        self.set_content_height(size)
        self.queue_draw()

    def set_accent(self, rgba: Gdk.RGBA | None) -> None:
        if rgba is not None:
            self._colour = (rgba.red, rgba.green, rgba.blue)
            self.queue_draw()

    def start(self) -> None:
        if self._tick is not None:
            return
        step = 1.0 / FPS

        def frame() -> bool:
            self._phase = (self._phase + step) % 3600.0
            self.queue_draw()
            return True

        self._tick = GLib.timeout_add(int(1000 / FPS), frame)

    def stop(self) -> None:
        if self._tick is not None:
            GLib.source_remove(self._tick)
            self._tick = None

    # ---------------------------------------------------------------- draw

    @staticmethod
    def _reduced_motion() -> bool:
        settings = Gtk.Settings.get_default()
        if settings is None:
            return False
        try:
            return not settings.get_property("gtk-enable-animations")
        except Exception:  # noqa: BLE001
            return False

    def _draw(self, _area: Gtk.DrawingArea, cr: Any, width: int, height: int) -> None:
        cx, cy = width / 2.0, height / 2.0
        radius = min(width, height) / 2.0 - 1.5
        if radius <= 0:
            return
        r, g, b = self._colour
        t = self._phase

        if self._reduced_motion():
            # A slow breath instead of rotation: still says "working", without
            # the vestibular cost of continuous motion.
            alpha = 0.35 + 0.35 * (0.5 + 0.5 * math.sin(t * 2 * math.pi * 0.4))
            cr.set_source_rgba(r, g, b, alpha)
            cr.arc(cx, cy, radius * 0.55, 0, 2 * math.pi)
            cr.fill()
            return

        cr.set_line_cap(1)  # ROUND
        for frac, sweep, rate, width_frac, alpha in ARCS:
            arc_radius = radius * frac
            if arc_radius <= 0:
                continue
            start = (t * rate) * 2 * math.pi
            cr.set_source_rgba(r, g, b, alpha)
            cr.set_line_width(max(radius * width_frac, 1.2))
            cr.arc(cx, cy, arc_radius, start, start + sweep * 2 * math.pi)
            cr.stroke()

        # The core pulses gently, so the centre reads as alive rather than a dot.
        pulse = 0.82 + 0.18 * math.sin(t * 2 * math.pi * CORE_PULSE_HZ)
        cr.set_source_rgba(r, g, b, 0.95)
        cr.arc(cx, cy, radius * CORE_RADIUS * pulse, 0, 2 * math.pi)
        cr.fill()
