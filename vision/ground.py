"""Turn "point at the Export button" into coordinates.

The model is allowed to name a target instead of locating it:

    {"kind": "ring", "target_text": "Export"}

That request is resolved here, against a fresh screenshot and tesseract, and
only the resolved shape reaches the overlay. The model's own ``x``/``y`` are
kept as the fallback for when OCR finds nothing — a ring in roughly the right
place still beats no ring, as long as the caller is told which one it got, so
the model can say "roughly here" rather than "here".

Everything is in the normalised 0–1000 grid from :mod:`ui.annotations`, so
nothing downstream needs to know the screen's resolution.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

from ui.annotations import GRID

logger = logging.getLogger(__name__)

# A ring should sit a little outside the text it circles rather than through it.
RING_PADDING = 1.35
BOX_PADDING_PX = 6
# Where an arrow starts when the model named only its destination: up and to
# the left, far enough to read as an arrow rather than a tick.
ARROW_OFFSET = 120.0


@dataclass
class Grounding:
    """The resolved payload, plus what could not be found."""

    payload: dict[str, Any]
    resolved: list[str]
    unresolved: list[str]
    ocr_available: bool = True

    @property
    def ok(self) -> bool:
        return not self.unresolved


def _image_size(image: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(image)) as img:
        return img.size


def _to_grid(value: float, extent: int) -> float:
    if extent <= 0:
        return 0.0
    return max(0.0, min(GRID, value / extent * GRID))


def resolve(payload: dict[str, Any], screenshot: bytes | None) -> Grounding:
    """Fill in coordinates for any shape that named its target in words."""
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        return Grounding(payload=payload, resolved=[], unresolved=[])

    wanted = [
        str(s.get("target_text", "")).strip()
        for s in shapes
        if isinstance(s, dict) and str(s.get("target_text", "")).strip()
    ]
    if not wanted:
        return Grounding(payload=payload, resolved=[], unresolved=[])

    from vision import ocr

    if not ocr.available() or screenshot is None:
        reason = "tesseract is not installed" if not ocr.available() else "no screenshot"
        logger.info("cannot ground %d target(s): %s", len(wanted), reason)
        return Grounding(
            payload=payload,
            resolved=[],
            unresolved=wanted,
            ocr_available=ocr.available(),
        )

    try:
        width, height = _image_size(screenshot)
    except Exception:  # noqa: BLE001
        logger.info("could not read the screenshot's size", exc_info=True)
        return Grounding(payload=payload, resolved=[], unresolved=wanted)

    # One OCR pass for every target: tesseract over a full screen is the
    # expensive part, and running it per shape would multiply it by five.
    words = ocr.read_words(screenshot)

    resolved: list[str] = []
    unresolved: list[str] = []
    out_shapes: list[Any] = []

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        target = str(shape.get("target_text", "")).strip()
        if not target:
            out_shapes.append(shape)
            continue

        match = ocr.find(words, target)
        updated = {k: v for k, v in shape.items() if k != "target_text"}
        updated.setdefault("text", target)

        if match is None:
            unresolved.append(target)
            # Keep whatever the model guessed; the caller reports the doubt.
            out_shapes.append(updated)
            continue

        cx, cy = match.center
        kind = str(shape.get("kind", "ring")).lower()
        if kind == "box":
            updated["x"] = _to_grid(match.left - BOX_PADDING_PX, width)
            updated["y"] = _to_grid(match.top - BOX_PADDING_PX, height)
            updated["w"] = _to_grid(match.width + BOX_PADDING_PX * 2, width)
            updated["h"] = _to_grid(match.height + BOX_PADDING_PX * 2, height)
        elif kind == "arrow":
            updated["x2"] = _to_grid(cx, width)
            updated["y2"] = _to_grid(cy, height)
            if "x" not in shape or "y" not in shape:
                updated["x"] = max(0.0, updated["x2"] - ARROW_OFFSET)
                updated["y"] = max(0.0, updated["y2"] - ARROW_OFFSET)
        else:
            updated["kind"] = kind if kind in ("ring", "label") else "ring"
            updated["x"] = _to_grid(cx, width)
            updated["y"] = _to_grid(cy, height)
            if kind != "label":
                half = max(match.width, match.height) / 2 * RING_PADDING
                # Scale by the smaller axis, matching ui.annotations.to_pixels,
                # so the ring comes back out the size it was measured at.
                updated["r"] = half / min(width, height) * GRID
        resolved.append(target)
        out_shapes.append(updated)

    grounded = dict(payload)
    grounded["shapes"] = out_shapes
    return Grounding(payload=grounded, resolved=resolved, unresolved=unresolved)
