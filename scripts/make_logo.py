#!/usr/bin/env python3
"""Generate every Keylane logo asset from one geometry definition.

The mark is a routing diagram: a chip (the NPU) fanning out to an arrow and two
endpoints (the workers). It is defined once here, in a 96x96 coordinate space,
and emitted as both SVG and PNG so the two can never drift apart.

    python scripts/make_logo.py

Outputs
    assets/keylane-logo.svg              master, full colour, transparent corners
    assets/keylane-symbolic.svg          glyph only, currentColor, for toolbars
    assets/keylane-logo.png              1024px master raster
    assets/logo.png                      512px
    launcher/assets/logo.png             256px
    web/assets/logo.svg                  served to the control panel and docs
    web/assets/logo.png                  256px raster fallback
    web/assets/favicon.png               64px
    assets/icons/hicolor/<n>x<n>/apps/keylane.png
    assets/icons/hicolor/scalable/apps/keylane.svg
    assets/icons/hicolor/scalable/status/keylane-<state>.svg

Rendering uses Pillow with heavy supersampling rather than an SVG rasterizer,
so the build has no system dependency beyond Pillow. Every PNG is written
RGBA with genuinely transparent corners — the previous assets had a
transparency *checkerboard* flattened into the pixels.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- palette --

INK = "#17171B"      # tile
GREEN = "#0F9C6D"    # glyph
BOX = 96.0           # coordinate space

# ------------------------------------------------------- geometry (96x96) --
# Measured from the original 1024px artwork and normalised.

TILE = dict(x=6.0, y=6.0, size=84.0, radius=15.0)

STROKE = 3.0
MID_Y = 48.1                       # the glyph's optical centre line

# The chip: a rounded square holding a 2x2 grid.
CHIP = dict(x=20.8, y=35.5, w=25.3, h=25.2, radius=4.6)
CELL = 4.5
CELL_R = 1.0
CELL_X = (28.3, 34.2)
CELL_Y = (42.9, 48.8)

# Where the three branches leave the chip.
JUNCTION_X = 51.0
CHIP_RIGHT = CHIP["x"] + CHIP["w"] + STROKE / 2

# The straight branch, ending in an arrowhead.
ARROW_TIP = (76.6, MID_Y)
ARROW_BACK_X = 70.2
ARROW_SPREAD = 5.5

# The two curved branches and the endpoints they land on.
END_SIZE = 6.7
END_R = 1.7
END_X = 69.4
END_TOP_CY = 23.7
END_BOTTOM_CY = 71.9
CTRL_X = 60.0                      # bezier control, shapes the S-curve


def _cubic(p0, p1, p2, p3, steps: int = 72):
    """Sample a cubic bezier into a polyline."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def branch_points(end_cy: float):
    """The S-curve from the junction out to one endpoint."""
    return _cubic(
        (JUNCTION_X, MID_Y),
        (CTRL_X, MID_Y),
        (CTRL_X, end_cy),
        (END_X, end_cy),
    )


# --------------------------------------------------------------- SVG out --


def _end_square(cy: float, colour: str, stroke: float, filled: bool) -> str:
    """One branch endpoint, solid or hollow."""
    if filled:
        return (
            f'<rect x="{END_X:.2f}" y="{cy - END_SIZE / 2:.2f}" width="{END_SIZE}" '
            f'height="{END_SIZE}" rx="{END_R}" fill="{colour}"/>'
        )
    inset = stroke / 2
    return (
        f'<rect x="{END_X + inset:.2f}" y="{cy - END_SIZE / 2 + inset:.2f}" '
        f'width="{END_SIZE - stroke:.2f}" height="{END_SIZE - stroke:.2f}" '
        f'rx="{max(END_R - inset, 0.4):.2f}" fill="none" stroke="{colour}" '
        f'stroke-width="{stroke}"/>'
    )


def _svg_glyph(stroke: float, colour: str, ends: str = "both") -> str:
    """The routing mark, as SVG.

    ``ends`` selects which branch endpoints are solid — "both", "top",
    "bottom" or "none". The tray animates by alternating them, which reads as
    traffic on the branches without adding anything foreign to the mark.
    """
    c = CHIP
    cells = "".join(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{CELL}" height="{CELL}" '
        f'rx="{CELL_R}" fill="{colour}"/>'
        for y in CELL_Y
        for x in CELL_X
    )
    ends = "".join(
        _end_square(
            cy,
            colour,
            stroke,
            filled=ends in ("both", name),
        )
        for name, cy in (("top", END_TOP_CY), ("bottom", END_BOTTOM_CY))
    )
    return f"""  <g fill="none" stroke="{colour}" stroke-width="{stroke}" \
stroke-linecap="round" stroke-linejoin="round">
    <rect x="{c['x']}" y="{c['y']}" width="{c['w']}" height="{c['h']}" rx="{c['radius']}"/>
    <path d="M {CHIP_RIGHT:.2f} {MID_Y} H {ARROW_TIP[0]}"/>
    <path d="M {ARROW_BACK_X} {MID_Y - ARROW_SPREAD} L {ARROW_TIP[0]} {MID_Y} \
L {ARROW_BACK_X} {MID_Y + ARROW_SPREAD}"/>
    <path d="M {JUNCTION_X} {MID_Y} C {CTRL_X} {MID_Y}, {CTRL_X} {END_TOP_CY}, \
{END_X} {END_TOP_CY}"/>
    <path d="M {JUNCTION_X} {MID_Y} C {CTRL_X} {MID_Y}, {CTRL_X} {END_BOTTOM_CY}, \
{END_X} {END_BOTTOM_CY}"/>
  </g>
{cells}
{ends}"""


def svg_logo() -> str:
    t = TILE
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{int(BOX)}" height="{int(BOX)}" \
viewBox="0 0 {int(BOX)} {int(BOX)}">
  <title>Keylane</title>
  <rect x="{t['x']}" y="{t['y']}" width="{t['size']}" height="{t['size']}" \
rx="{t['radius']}" fill="{INK}"/>
{_svg_glyph(STROKE, GREEN)}
</svg>
"""


def svg_symbolic(
    extra: str = "",
    stroke: float = 5.0,
    ends: str = "both",
    opacity: float = 1.0,
) -> str:
    """Glyph only, in currentColor, with heavier strokes for small sizes."""
    glyph = _svg_glyph(stroke, "currentColor", ends)
    if opacity < 1.0:
        glyph = f'  <g opacity="{opacity}">\n{glyph}\n  </g>'
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{int(BOX)}" height="{int(BOX)}" \
viewBox="0 0 {int(BOX)} {int(BOX)}">
  <title>Keylane</title>
{glyph}
{extra}
</svg>
"""


# --------------------------------------------------------------- PNG out --


def _supersample_for(size: int) -> int:
    """Small icons need more supersampling; big ones would exhaust memory."""
    if size <= 64:
        return 16
    if size <= 256:
        return 8
    return 4


class Canvas:
    """Supersampled RGBA canvas. Everything is drawn in 96x96 units."""

    def __init__(self, size: int, scale: int | None = None) -> None:
        self.size = size
        self.ss = size * (scale or _supersample_for(size))
        self.k = self.ss / BOX
        self.image = Image.new("RGBA", (self.ss, self.ss), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    def _p(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.k, y * self.k)

    def rounded_rect(self, x, y, w, h, r, fill=None, outline=None, width=0.0):
        box = [*self._p(x, y), *self._p(x + w, y + h)]
        self.draw.rounded_rectangle(
            box,
            radius=r * self.k,
            fill=fill,
            outline=outline,
            width=max(1, round(width * self.k)) if width else 0,
        )

    def stroke(self, points, colour, width):
        """Stroke a polyline by stamping discs along it.

        Pillow's ``line(joint="curve")`` leaves visible notches on the outer
        edge of a curve where segments meet. Stamping overlapping discs gives a
        seamless stroke with round caps and joins for nothing.
        """
        pts = [self._p(*p) for p in points]
        r = max(0.5, width * self.k / 2)
        step = max(r / 6.0, 0.5)

        def stamp(x: float, y: float) -> None:
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)

        stamp(*pts[0])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            for i in range(1, int(dist / step) + 1):
                t = min(i * step / dist, 1.0)
                stamp(x0 + dx * t, y0 + dy * t)
            stamp(x1, y1)

    def finish(self) -> Image.Image:
        return self.image.resize((self.size, self.size), Image.LANCZOS)


def rounded_rect_path(x, y, w, h, r, steps: int = 16):
    """Outline of a rounded rectangle as a closed polyline."""
    pts: list[tuple[float, float]] = []
    corners = (
        (x + w - r, y + r, -90, 0),    # top-right
        (x + w - r, y + h - r, 0, 90),  # bottom-right
        (x + r, y + h - r, 90, 180),    # bottom-left
        (x + r, y + r, 180, 270),       # top-left
    )
    for cx, cy, a0, a1 in corners:
        for i in range(steps + 1):
            a = math.radians(a0 + (a1 - a0) * i / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    return pts


def draw_glyph(c: Canvas, colour: str, stroke: float) -> None:
    chip = CHIP
    c.stroke(
        rounded_rect_path(chip["x"], chip["y"], chip["w"], chip["h"], chip["radius"]),
        colour,
        stroke,
    )
    for cy in CELL_Y:
        for cx in CELL_X:
            c.rounded_rect(cx, cy, CELL, CELL, CELL_R, fill=colour)

    c.stroke([(CHIP_RIGHT, MID_Y), ARROW_TIP], colour, stroke)
    c.stroke(
        [
            (ARROW_BACK_X, MID_Y - ARROW_SPREAD),
            ARROW_TIP,
            (ARROW_BACK_X, MID_Y + ARROW_SPREAD),
        ],
        colour,
        stroke,
    )
    for cy in (END_TOP_CY, END_BOTTOM_CY):
        c.stroke(branch_points(cy), colour, stroke)
        c.rounded_rect(END_X, cy - END_SIZE / 2, END_SIZE, END_SIZE, END_R, fill=colour)


def render_logo(size: int) -> Image.Image:
    c = Canvas(size)
    t = TILE
    c.rounded_rect(t["x"], t["y"], t["size"], t["size"], t["radius"], fill=INK)
    draw_glyph(c, GREEN, STROKE)
    return c.finish()


# ------------------------------------------------------------------ build --

# Tray states, drawn from the same mark so the taskbar matches everything else.
# Tray states, all drawn from the same mark so the taskbar matches everything
# else. Each is (extra markup, which endpoints are solid, glyph opacity).
STATUS_STATES = {
    "idle": ("", "both", 1.0),
    # Busy alternates which branch is "live" — structural, so it stays legible
    # at 16px where a translucent glow would just look like a smudge.
    "busy": ("", "top", 1.0),
    "busy-alt": ("", "bottom", 1.0),
    # A badge in the clear space under the chip. The mark is punched out with
    # fill-rule="evenodd" rather than painted in white, because a symbolic icon
    # is a single colour — a white glyph on a white badge is invisible.
    "attention": (
        '  <path fill="currentColor" fill-rule="evenodd" d="'
        'M 20 65 a 14 14 0 1 0 0.01 0 z '
        'M 17.6 70.5 h 4.8 v 10.5 h -4.8 z '
        'M 20 85.4 a 2.9 2.9 0 1 0 0.01 0 z"/>',
        "both",
        1.0,
    ),
    # Dimmed and struck through: unmistakably "not connected".
    "offline": (
        '  <path d="M 14 82 L 82 14" stroke="currentColor" stroke-width="7" '
        'stroke-linecap="round" fill="none"/>',
        "none",
        0.45,
    ),
}

PNG_TARGETS = [
    (ROOT / "assets" / "keylane-logo.png", 1024),
    (ROOT / "assets" / "logo.png", 512),
    (ROOT / "launcher" / "assets" / "logo.png", 256),
    # The control panel and handbook are served from web/assets, so they need
    # their own copies — this is the one that used to be missed on a refresh.
    (ROOT / "web" / "assets" / "logo.png", 256),
    (ROOT / "web" / "assets" / "favicon.png", 64),
]
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def build() -> None:
    written: list[str] = []

    def write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(str(path.relative_to(ROOT)))

    def write_png(path: Path, size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        render_logo(size).save(path, "PNG", optimize=True)
        written.append(f"{path.relative_to(ROOT)}  ({size}px)")

    # Vector masters.
    write_text(ROOT / "assets" / "keylane-logo.svg", svg_logo())
    write_text(ROOT / "assets" / "keylane-symbolic.svg", svg_symbolic())
    # Served to the browser: an SVG stays crisp at any zoom and weighs nothing.
    write_text(ROOT / "web" / "assets" / "logo.svg", svg_logo())
    write_text(
        ROOT / "assets" / "icons" / "hicolor" / "scalable" / "apps" / "keylane.svg",
        svg_logo(),
    )
    # GTK recolours any icon whose name ends in -symbolic, so the popup can tint
    # the mark with the theme accent instead of pasting a dark tile on a dark bar.
    write_text(
        ROOT / "assets" / "icons" / "hicolor" / "scalable" / "apps" / "keylane-symbolic.svg",
        svg_symbolic(stroke=5.5),
    )

    # Tray states — same glyph, heavier stroke so it survives a 16px panel.
    status_dir = ROOT / "assets" / "icons" / "hicolor" / "scalable" / "status"
    for state, (extra, ends, opacity) in STATUS_STATES.items():
        write_text(
            status_dir / f"keylane-{state}.svg",
            svg_symbolic(extra, stroke=5.5, ends=ends, opacity=opacity),
        )

    # Rasters.
    for path, size in PNG_TARGETS:
        write_png(path, size)
    for size in ICON_SIZES:
        write_png(
            ROOT / "assets" / "icons" / "hicolor" / f"{size}x{size}" / "apps" / "keylane.png",
            size,
        )

    for line in written:
        print(f"  {line}")
    print(f"\nWrote {len(written)} asset(s).")


if __name__ == "__main__":
    build()
