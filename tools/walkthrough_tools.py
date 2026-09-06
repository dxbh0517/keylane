"""Walking the user through something, one drawn step at a time."""

from __future__ import annotations

import logging
from typing import Any

from seams.walkthrough import MAX_STEPS, get_walkthroughs, parse_steps
from tools.registry import Tool, ToolRegistry
from tools.screen_tools import SHAPE_SCHEMA, send_control

logger = logging.getLogger(__name__)


def _show_current(walkthrough) -> str:
    state = walkthrough.describe()
    annotation = state.get("annotation")
    if annotation is None:
        return "The walkthrough is finished."

    step, total = state["step"], state["total"]
    annotation = dict(annotation)
    # The user needs to know where they are in the sequence; the model has
    # already spent its caption on what to do.
    annotation["caption"] = f"{step}/{total}  {state['caption']}"
    annotation["walkthrough"] = True

    reply = send_control("annotate", annotation)
    if not reply.get("ok"):
        detail = reply.get("error", "the step could not be drawn")
        return f"Showing step {step} of {total} failed: {detail}"
    return f"Showing step {step} of {total}: {state['caption']}"


def walkthrough_show(steps: Any, title: str = "") -> str:
    parsed, error = parse_steps(steps)
    if error:
        return f"Error: {error}"
    walkthrough = get_walkthroughs().start(parsed, title=str(title or "")[:100])
    return _show_current(walkthrough)


def walkthrough_next() -> str:
    walkthrough = get_walkthroughs().advance()
    if walkthrough is None:
        return "No walkthrough is running."
    if walkthrough.finished:
        send_control("annotate_clear")
        return "That was the last step; the walkthrough is done."
    return _show_current(walkthrough)


def register_walkthrough_tools(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="walkthrough_show",
            description=(
                "Walk the user through a procedure, drawing each step where it happens. "
                "Use it when the answer is a sequence of clicks rather than one place. "
                f"At most {MAX_STEPS} steps; fewer is better. Each step needs a caption "
                "and the marks to draw for it — prefer `target_text` over coordinates. "
                "Keylane advances the steps at the user's pace, so you need not wait. "
                "End it with `screen_clear`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "What this walkthrough is for."},
                    "steps": {
                        "type": "array",
                        "description": "The steps, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "caption": {
                                    "type": "string",
                                    "description": "What to do at this step — one short line.",
                                },
                                "shapes": {
                                    "type": "array",
                                    "description": "Marks to draw for this step.",
                                    "items": SHAPE_SCHEMA,
                                },
                            },
                            "required": ["caption"],
                        },
                    },
                },
                "required": ["steps"],
            },
            handler=walkthrough_show,
        )
    )

    reg.register(
        Tool(
            name="walkthrough_next",
            description="Move the running walkthrough to its next step.",
            parameters={"type": "object", "properties": {}},
            handler=walkthrough_next,
        )
    )



WALKTHROUGH_GUIDANCE = """For a procedure the user has to carry out in an app — several \
clicks in order — use `walkthrough_show` rather than listing the steps in prose. Each step \
draws itself where it happens, and Keylane moves through them at the user's pace, so you \
do not wait or repeat them. Keep it under about six steps; if the task genuinely needs \
more than that, it is a document, not a walkthrough. Still say in your reply what the \
sequence achieves — the marks fade, your answer does not."""


def register_walkthrough_sections(prompt: Any) -> None:
    prompt.section("walkthrough", WALKTHROUGH_GUIDANCE, required=False)
