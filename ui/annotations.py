"""Shapes Keylane draws on the desktop, and the arithmetic behind them.

Kept apart from the window that paints them so it can be tested without a
display, and because two of the decisions here are the ones that decide whether
this feature works at all.

**Coordinates arrive normalised.** The model is asked for a 0–1000 grid rather
than pixels, which is the convention Qwen-VL and Gemma grounding are trained
on. Pixels would make every answer depend on the resolution of the screenshot
it happened to see; a fixed grid does not, and it is also the only form we can
sanity-check — anything outside 0–1000 is a hallucinated coordinate, not a
point off the edge of the screen.

**The strokes wobble on purpose.** A geometrically perfect ring drawn over
someone's desktop reads as a compositor glitch. A slightly uneven one reads as
somebody pointing. The jitter is derived from the shape itself, so a given
shape wobbles the *same* way on every frame — jitter re-rolled per frame is a
shape that vibrates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# The grid the model answers in. Not the screen, and not the screenshot.
GRID = 1000.0

KINDS = ("ring", "box", "arrow", "label")

# Beyond this many shapes an annotation is no longer pointing at anything.
MAX_SHAPES = 12
MAX_LABEL_CHARS = 80

DEFAULT_RING_RADIUS = 45.0  # grid units, ≈ a button's worth of screen
MIN_BOX_SIDE = 8.0


@dataclass(frozen=True)
class Shape:
    """One mark, in grid units. :func:`to_pixels` turns it into a drawing."""

    kind: str = "ring"
    x: float = 0.0
    y: float = 0.0
    # ring: radius. box: width/height. arrow: the far end.
    r: float = DEFAULT_RING_RADIUS
    w: float = 0.0
    h: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    text: str = ""

    @property
    def seed(self) -> int:
        """A stable per-shape number, so its wobble never changes."""
        blob = f"{self.kind}:{self.x:.2f}:{self.y:.2f}:{self.r:.2f}:{self.w:.2f}:{self.h:.2f}:{self.text}"
        return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class Annotation:
    """A set of shapes with a lifetime."""

    shapes: list[Shape] = field(default_factory=list)
    ttl_ms: int = 2000
    caption: str = ""

    @property
    def persistent(self) -> bool:
        """A zero or negative TTL means "stay until cleared" — walkthrough steps."""
        return self.ttl_ms <= 0


def _number(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _clamp(value: float, low: float = 0.0, high: float = GRID) -> float:
    return max(low, min(high, value))


def parse_shape(raw: Any) -> Shape | None:
    """One shape from the model's JSON, or None if it is not usable.

    Silently dropping a bad shape is right here: a model that emits four good
    marks and one malformed one should still point at the four.
    """
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", "ring")).strip().lower()
    if kind not in KINDS:
        return None

    x = _clamp(_number(raw.get("x")))
    y = _clamp(_number(raw.get("y")))
    text = str(raw.get("text", "") or raw.get("label", "") or "")[:MAX_LABEL_CHARS]

    if kind == "ring":
        radius = _number(raw.get("r"), DEFAULT_RING_RADIUS)
        # A zero-radius ring points at nothing; a screen-sized one points at
        # everything. Both are what a confused model emits.
        radius = max(6.0, min(radius, GRID / 3))
        return Shape(kind=kind, x=x, y=y, r=radius, text=text)

    if kind == "box":
        w = _number(raw.get("w"))
        h = _number(raw.get("h"))
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            return None
        return Shape(kind=kind, x=x, y=y, w=min(w, GRID - x), h=min(h, GRID - y), text=text)

    if kind == "arrow":
        x2 = _clamp(_number(raw.get("x2")))
        y2 = _clamp(_number(raw.get("y2")))
        if math.hypot(x2 - x, y2 - y) < 4:
            return None
        return Shape(kind=kind, x=x, y=y, x2=x2, y2=y2, text=text)

    if not text:
        return None
    return Shape(kind="label", x=x, y=y, text=text)


def parse_annotation(payload: str | dict) -> Annotation:
    """Parse what the daemon sent the overlay. Never raises."""
    data: Any = payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return Annotation(shapes=[])
    if not isinstance(data, dict):
        return Annotation(shapes=[])

    raw_shapes = data.get("shapes")
    shapes: list[Shape] = []
    if isinstance(raw_shapes, list):
        for entry in raw_shapes[:MAX_SHAPES]:
            shape = parse_shape(entry)
            if shape is not None:
                shapes.append(shape)

    try:
        ttl = int(data.get("ttl_ms", 2000))
    except (TypeError, ValueError):
        ttl = 2000
    return Annotation(
        shapes=shapes,
        ttl_ms=max(-1, min(ttl, 600_000)),
        caption=str(data.get("caption", ""))[:200],
    )


def to_pixels(shape: Shape, width: int, height: int) -> Shape:
    """Grid units to screen pixels.

    A ring's radius is scaled by the *smaller* axis so a circle stays a circle
    on a wide monitor rather than becoming an ellipse that misses its target.
    """
    sx, sy = width / GRID, height / GRID
    scale = min(sx, sy)
    return Shape(
        kind=shape.kind,
        x=shape.x * sx,
        y=shape.y * sy,
        r=shape.r * scale,
        w=shape.w * sx,
        h=shape.h * sy,
        x2=shape.x2 * sx,
        y2=shape.y2 * sy,
        text=shape.text,
    )


def from_pixels(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Screen pixels back to grid units — used when the *user* points."""
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    return _clamp(x / width * GRID), _clamp(y / height * GRID)


def box_from_points(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    """The bounding box of a freehand stroke, as ``(x, y, w, h)``."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


# ── the hand-drawn look ──────────────────────────────────────────────────


def _rand(seed: int, index: int) -> float:
    """A deterministic -1..1 from a seed and a step number."""
    mixed = (seed * 6364136223846793005 + index * 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    unit = (mixed >> 33) / float(1 << 31)  # 0..1
    return unit * 2.0 - 1.0


def wobbly_circle(
    cx: float, cy: float, radius: float, seed: int, *, steps: int = 48, amount: float = 0.05
) -> list[tuple[float, float]]:
    """A closed ring whose radius breathes slightly — a drawn circle, not a path.

    The overshoot at the end is what sells it: a hand coming back round does
    not stop exactly where it started.
    """
    points: list[tuple[float, float]] = []
    span = math.tau * 1.04
    for step in range(steps + 1):
        angle = span * step / steps
        drift = 1.0 + _rand(seed, step) * amount
        points.append((cx + math.cos(angle) * radius * drift, cy + math.sin(angle) * radius * drift))
    return points


def wobbly_line(
    x1: float, y1: float, x2: float, y2: float, seed: int, *, steps: int = 12, amount: float = 0.02
) -> list[tuple[float, float]]:
    """A line that bows off true by a fraction of its own length."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return [(x1, y1), (x2, y2)]
    # Perpendicular unit vector: the direction the wobble pushes into.
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    points: list[tuple[float, float]] = []
    for step in range(steps + 1):
        t = step / steps
        # Zero at both ends, widest in the middle — a bow, not a zigzag.
        envelope = math.sin(math.pi * t)
        offset = _rand(seed, step) * amount * length * envelope
        points.append((x1 + (x2 - x1) * t + nx * offset, y1 + (y2 - y1) * t + ny * offset))
    return points


def wobbly_rect(
    x: float, y: float, w: float, h: float, seed: int, *, amount: float = 0.02
) -> list[tuple[float, float]]:
    """A closed rectangle drawn as four bowed strokes."""
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    points: list[tuple[float, float]] = []
    for index in range(4):
        (ax, ay), (bx, by) = corners[index], corners[index + 1]
        segment = wobbly_line(ax, ay, bx, by, seed + index * 977, steps=8, amount=amount)
        points.extend(segment if index == 0 else segment[1:])
    return points


def arrow_head(
    x1: float, y1: float, x2: float, y2: float, size: float = 16.0, spread: float = 0.42
) -> list[tuple[float, float]]:
    """The two barbs at the ``(x2, y2)`` end, as points to stroke from the tip."""
    angle = math.atan2(y2 - y1, x2 - x1)
    return [
        (x2 - math.cos(angle - spread) * size, y2 - math.sin(angle - spread) * size),
        (x2, y2),
        (x2 - math.cos(angle + spread) * size, y2 - math.sin(angle + spread) * size),
    ]
