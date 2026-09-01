"""Tool-execution policy, as pipeline hooks.

Permission gating used to be an `if` in the middle of `ToolRegistry.call`,
which meant every new policy — spill, loop guards, timeouts — had to be another
`if` in the same place. Each one is now a hook that can be registered,
inspected, and removed on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from tools.registry import (
    ALLOW,
    ALWAYS_GATED,
    PreDecision,
    ToolCall,
    ToolOutcome,
    ToolRegistry,
    deny,
)

logger = logging.getLogger(__name__)


async def permission_hook(call: ToolCall) -> PreDecision:
    """Gate a dangerous call on the user's answer.

    `auto` runs it, `deny` refuses it, `ask` raises a prompt in the UI and waits.
    Only tools flagged dangerous or named in ALWAYS_GATED are gated at all — the
    mode of an ungated tool is irrelevant, which keeps `recall` from prompting.
    """
    from daemon.config import permission_mode

    gated = call.tool.dangerous or call.name in ALWAYS_GATED
    if not gated:
        return ALLOW

    mode = permission_mode(call.name)
    if mode == "deny":
        return deny(f"permission denied for {call.name}", code="PERMISSION_DENIED")
    if mode != "ask":
        return ALLOW

    from daemon.permissions import create_pending, wait_pending

    pending = create_pending(call.name, call.arguments)
    if not pending or not pending.id:
        return ALLOW

    on_event = getattr(call, "on_event", None)
    if on_event:
        on_event(
            "permission",
            {
                "id": pending.id,
                "tool": call.name,
                "arguments": call.arguments,
                "message": f"Allow {call.name}?",
            },
        )
    approved = await asyncio.to_thread(wait_pending, pending)
    if not approved:
        return deny(f"permission denied for {call.name}", code="PERMISSION_DENIED")
    return ALLOW


_INSTALLED: set[int] = set()


def install_default_policy(registry: ToolRegistry) -> list[Any]:
    """Attach the policy every scope inherits. Returns the disposers.

    Idempotent per registry: registering the permission hook twice would raise
    two prompts for one call, which reads to the user as a bug in the gate.
    """
    if id(registry) in _INSTALLED:
        return []
    _INSTALLED.add(id(registry))
    from tools.guards import repeat_reminder_hook

    return [
        registry.add_pre_hook(permission_hook),
        # Order matters: spill shrinks the result, then the guard appends its
        # reminder to what the model will actually read.
        registry.add_post_hook(spill_hook),
        registry.add_post_hook(repeat_reminder_hook),
    ]


def spill_hook(call: ToolCall, outcome: ToolOutcome) -> ToolOutcome | None:
    """Save an oversized result and replace it with a preview plus a locator.

    Best-effort: if the save fails the original result stays inline, because
    turning a successful call into an error over a storage problem is worse
    than a long result.
    """
    from seams.spill import MAX_INLINE_CHARS, get_spill_store, preview

    if outcome.is_error or len(outcome.content) <= MAX_INLINE_CHARS:
        return None
    try:
        ref = get_spill_store().save_text(
            session_id=call.scope,
            tool_name=call.name,
            content=outcome.content,
        )
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not spill %s result: %s", call.name, exc)
        return None

    outcome.content = (
        f"{preview(outcome.content)}\n\n"
        f"(Result truncated at {MAX_INLINE_CHARS} characters. {ref.retrieval_hint})"
    )
    outcome.meta["spill"] = ref.locator
    return outcome
