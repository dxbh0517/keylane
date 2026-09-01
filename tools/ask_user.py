"""The `ask_user` tool.

The system prompt used to carry rules like "propose it in one sentence and wait
for a yes before creating one" — a rule a 9B model drops exactly when it matters.
A question the model can *call* is a mechanism instead of a hope, and the
plumbing already existed: the permission seam raises a prompt in the HUD and
blocks the call until a human answers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from daemon.permissions import create_prompt, wait_pending
from tools.registry import Tool, ToolRegistry

ASK_TIMEOUT_SECONDS = 300.0


def _normalize_options(options: Any) -> list[dict[str, str]]:
    if not isinstance(options, list):
        return []
    cleaned: list[dict[str, str]] = []
    for option in options:
        if isinstance(option, str):
            cleaned.append({"label": option, "description": ""})
        elif isinstance(option, dict) and option.get("label"):
            cleaned.append(
                {
                    "label": str(option["label"]),
                    "description": str(option.get("description", "")),
                }
            )
    return cleaned


async def ask_user(question: str, options: Any = None, header: str = "") -> str:
    """Put one question to the user and wait for the answer."""
    text = str(question).strip()
    if not text:
        return json.dumps({"error": "question must be a non-empty string", "code": "INVALID_ARGS"})

    # The approval channel carries this: an unanswered question and an
    # unanswered permission prompt are the same shape of wait. It is
    # `create_prompt`, not `create_pending`, because a question is not gated by
    # a permission mode — the prompt is the entire point of the call.
    pending = create_prompt(
        "ask_user",
        {"question": text, "options": _normalize_options(options), "header": header},
    )
    approved = await asyncio.to_thread(wait_pending, pending, timeout=ASK_TIMEOUT_SECONDS)
    if not approved:
        return json.dumps(
            {
                "answered": False,
                "note": "The user did not answer. Proceed with what you have, "
                "or tell them what you need.",
            }
        )
    return json.dumps({"answered": True, "approved": True})


def register_ask_user_tool(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="ask_user",
            description=(
                "Ask the user a short question when you need confirmation or a choice "
                "before acting — before setting up a recurring watcher, before anything "
                "that changes their machine, or when the request is genuinely ambiguous. "
                "Blocks until they answer. Do not use it for questions you can answer "
                "with a tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question, in one sentence.",
                    },
                    "header": {
                        "type": "string",
                        "description": 'Optional short heading, e.g. "Confirm".',
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional choices to offer. Put your recommendation first.",
                    },
                },
                "required": ["question"],
            },
            handler=ask_user,
            timeout_ms=int(ASK_TIMEOUT_SECONDS * 1000) + 5000,
        )
    )


ASK_GUIDANCE = """Use `ask_user` when you need the user's decision before acting — a \
recurring watcher, anything that changes their machine, or a request that could \
reasonably mean two different things. It blocks until they answer. Ask once, with the \
options if there are any; do not ask for information a tool can get you."""


def register_ask_user_sections(prompt: Any) -> None:
    prompt.section("plan", ASK_GUIDANCE, required=False)
