"""Drawing on the user's screen, as a tool the model can call.

The daemon does not own a display — the UI process does — so this is a thin
client for the control socket in ``ui/main.py``. Everything expensive
(screenshot, OCR, painting) happens over there; what comes back is a verdict,
and the verdict is the point. A tool that always answered "drawn" would let
the model narrate a ring around a button that is not on screen, which is worse
than not drawing at all.

Nothing here is gated by the permission system. Drawing is display-only: it
cannot click, type, read a file, or reach the network, and a mark that the user
does not want is gone within seconds or on the next `screen_clear`.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any

from tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 9101
# Grounding a named target means a screenshot plus a tesseract pass over the
# whole screen, which on a large display is genuinely slow.
CONTROL_TIMEOUT = 60.0

MAX_SECONDS = 120
DEFAULT_SECONDS = 4


def send_control(verb: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    line = verb if payload is None else f"{verb} {json.dumps(payload, ensure_ascii=False)}"
    try:
        with socket.create_connection((CONTROL_HOST, CONTROL_PORT), timeout=5) as sock:
            sock.settimeout(CONTROL_TIMEOUT)
            sock.sendall((line + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
    except OSError as exc:
        return {"ok": False, "error": f"the Keylane UI is not running ({exc})"}

    raw = b"".join(chunks).decode("utf-8", errors="ignore").strip()
    if not raw:
        return {"ok": False, "error": "the UI accepted the request but said nothing"}
    try:
        reply = json.loads(raw.split("\n", 1)[0])
    except json.JSONDecodeError:
        return {"ok": False, "error": f"unreadable reply from the UI: {raw[:120]}"}
    return reply if isinstance(reply, dict) else {"ok": False, "error": "malformed reply"}


def screen_annotate(
    shapes: Any,
    caption: str = "",
    seconds: float = DEFAULT_SECONDS,
) -> str:
    if not isinstance(shapes, list) or not shapes:
        return "Error: shapes must be a non-empty array."

    try:
        ttl = int(float(seconds) * 1000)
    except (TypeError, ValueError):
        ttl = DEFAULT_SECONDS * 1000
    ttl = max(500, min(ttl, MAX_SECONDS * 1000))

    reply = send_control("annotate", {"shapes": shapes, "caption": caption, "ttl_ms": ttl})
    if reply.get("ok"):
        located = reply.get("located") or []
        where = f" on {', '.join(located)}" if located else ""
        return f"Drew {reply.get('shapes', len(shapes))} mark(s){where} for {ttl / 1000:g}s."
    return f"Error: {reply.get('error', 'the annotation could not be drawn')}"


def screen_clear() -> str:
    """Erase the marks, and end a walkthrough if one is running.

    One tool rather than two: "get that off my screen" is a single intent, and
    a model that had to know which of two tools applied would sometimes pick
    the one that left half the drawing behind.
    """
    from seams.walkthrough import get_walkthroughs

    stopped = get_walkthroughs().stop()
    reply = send_control("annotate_clear")
    if not reply.get("ok"):
        return f"Error: {reply.get('error')}"
    return "Stopped the walkthrough and cleared the marks." if stopped else "Cleared the marks."


SHAPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["ring", "box", "arrow", "label"],
            "description": "ring circles a control, box outlines a region, "
            "arrow points from one place to another, label writes a note.",
        },
        "target_text": {
            "type": "string",
            "description": (
                "The visible text of the thing to point at, e.g. \"Export\". "
                "PREFER THIS over coordinates: Keylane reads the screen and "
                "finds it exactly. Only the text as it appears — no description."
            ),
        },
        "x": {"type": "number", "description": "0-1000 across the screen, left to right."},
        "y": {"type": "number", "description": "0-1000 down the screen, top to bottom."},
        "r": {"type": "number", "description": "ring radius in the same 0-1000 units."},
        "w": {"type": "number", "description": "box width, 0-1000 units."},
        "h": {"type": "number", "description": "box height, 0-1000 units."},
        "x2": {"type": "number", "description": "arrow tip, 0-1000 units."},
        "y2": {"type": "number", "description": "arrow tip, 0-1000 units."},
        "text": {"type": "string", "description": "A short caption drawn beside the mark."},
    },
    "required": ["kind"],
}


def register_screen_tools(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="screen_annotate",
            description=(
                "Point at something on the user's screen with a ring, arrow, box or label. "
                "Use it when the answer is a place rather than a sentence. Give "
                "`target_text` with the control's visible label whenever you can: Keylane "
                "then finds it by reading the screen, which is far more accurate than "
                "coordinates you estimate. Marks fade on their own, and this only draws — "
                "it cannot click."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "shapes": {
                        "type": "array",
                        "description": "One to a few marks. More than about four stops pointing at anything.",
                        "items": SHAPE_SCHEMA,
                    },
                    "caption": {
                        "type": "string",
                        "description": "One line shown along the bottom of the screen.",
                    },
                    "seconds": {
                        "type": "number",
                        "description": f"How long the marks stay, 0.5-{MAX_SECONDS}. Default {DEFAULT_SECONDS}.",
                    },
                },
                "required": ["shapes"],
            },
            handler=screen_annotate,
        )
    )

    reg.register(
        Tool(
            name="screen_clear",
            description="Erase Keylane's marks, ending any walkthrough in progress.",
            parameters={"type": "object", "properties": {}},
            handler=screen_clear,
            concurrency_safe=True,
        )
    )


SCREEN_GUIDANCE = """When the user asks *where* something is on their screen, or how to \
do something in an app they are looking at, point at it with `screen_annotate` instead of \
describing a location in words. Name the control with `target_text` — its visible label — \
and Keylane finds it by reading the screen; only fall back to x/y coordinates when the \
target has no readable text. One or two marks land; five do not. Say what you drew in your \
reply, because the marks fade and the answer stays."""


def register_screen_sections(prompt: Any) -> None:
    prompt.section("screen", SCREEN_GUIDANCE, required=False)
