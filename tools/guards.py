"""Loop hygiene, as an advisory post-execute hook.

A model that re-runs a failing command or re-reads an unchanged file burns the
iteration budget without making progress. The old check compared one call
against the immediately preceding one, so an A→B→A→B loop was invisible, and
when it did fire it spent an iteration on a scolding tool result.

This keeps a repeat chain per scope instead: same tool, same arguments, however
many other calls happened around it. At chosen counts it *appends* a reminder to
the result — advice, never a block, because a legitimately repeated call must
not be delayed by anything. A new user message clears the chain, so a fresh
instruction is never read as a loop.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from tools.registry import ToolCall, ToolOutcome

# Chosen counts, not every repeat: the third is where a loop starts to look
# like one, and the later two escalate rather than nag.
THRESHOLDS = (3, 5, 8)
ARGUMENT_PREVIEW_CHARS = 500

# Tools whose whole purpose is to be called repeatedly with the same arguments.
EXCLUDED = frozenset(
    {"reminders_list", "inbox_list", "memories_list", "job_list", "skill_list", "get_goal"}
)


def canonical(call: ToolCall) -> str:
    """A stable identity for one call, insensitive to key order."""
    return json.dumps(
        {"name": call.name, "arguments": call.arguments},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


@dataclass
class _Chain:
    signature: str = ""
    count: int = 0


@dataclass
class RepeatTracker:
    """One repeat chain per scope, so one agent's loop never nudges another."""

    _chains: dict[str, _Chain] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self, scope: str) -> None:
        with self._lock:
            self._chains.pop(scope, None)

    def observe(self, call: ToolCall) -> int:
        """Record one call and return how many times it has repeated."""
        signature = canonical(call)
        with self._lock:
            chain = self._chains.setdefault(call.scope, _Chain())
            if chain.signature == signature:
                chain.count += 1
            else:
                chain.signature = signature
                chain.count = 1
            return chain.count


_tracker = RepeatTracker()


def reset_repeat_chain(scope: str) -> None:
    """Called when a new user message arrives — that is not a loop."""
    _tracker.reset(scope)


def _reminder(call: ToolCall, count: int) -> str:
    if count == THRESHOLDS[0]:
        return (
            f"You have now called {call.name} {count} times with the same arguments. "
            "Read the previous result before calling it again — if it did not answer "
            "the question, a different approach will work better than a repeat."
        )
    preview = canonical(call)[:ARGUMENT_PREVIEW_CHARS]
    return (
        f"You have called {call.name} {count} times with identical arguments: {preview}. "
        "This is returning the same thing every time. Either change the approach, use "
        "what you already have, or tell the user what is blocking you."
    )


def repeat_reminder_hook(call: ToolCall, outcome: ToolOutcome) -> ToolOutcome | None:
    """Append a reminder when a call has repeated enough to look stuck."""
    if call.name in EXCLUDED:
        return None
    count = _tracker.observe(call)
    if count not in THRESHOLDS:
        return None
    outcome.content = f"{outcome.content}\n\n<system-reminder>\n{_reminder(call, count)}\n</system-reminder>"
    outcome.meta["repeat_count"] = count
    return outcome
