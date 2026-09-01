"""The three goal tools.

`get_goal` before `update_goal` is not a convention here — the update carries
the revision it read, and a stale one is refused. That is what stops two rounds
of the same goal from overwriting each other's conclusions.
"""

from __future__ import annotations

import json
from typing import Any

from tools.registry import Tool, ToolRegistry


def _service():
    from seams import get_context

    return get_context().goals


class GoalTools:
    """Bound to one session, because a goal belongs to a conversation."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def create(self, objective: str, max_rounds: int | None = None) -> str:
        goal = _service().create(self.session_id, objective, max_rounds)
        return json.dumps(goal.view(), ensure_ascii=False)

    def get(self) -> str:
        goal = _service().get(self.session_id)
        if goal is None:
            return json.dumps({"goal": None, "note": "This session has no goal."})
        return json.dumps(goal.view(), ensure_ascii=False)

    def update(
        self,
        goal_id: str,
        revision: int,
        action: str,
        objective: str = "",
        max_rounds: int | None = None,
        blocked_reason: str = "",
    ) -> str:
        goal = _service().update(
            self.session_id,
            goal_id=str(goal_id),
            revision=int(revision),
            action=str(action),
            objective=objective,
            max_rounds=max_rounds,
            blocked_reason=blocked_reason,
        )
        return json.dumps(goal.view(), ensure_ascii=False)


def register_goal_tools(reg: ToolRegistry, session_id: str) -> None:
    """Register the goal tools into one session's scope."""
    tools = GoalTools(session_id)

    reg.register(
        Tool(
            name="create_goal",
            description=(
                "Create one persisted objective for this session when the request is "
                "long-running work that should continue across rounds rather than finish "
                "in this turn. You may infer that from the request; the user does not "
                "have to say 'goal'. Not for ordinary single-turn work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The concrete completion objective.",
                    },
                    "max_rounds": {
                        "type": "number",
                        "description": "Optional cap on automatic continuation rounds.",
                    },
                },
                "required": ["objective"],
            },
            handler=tools.create,
        )
    )

    reg.register(
        Tool(
            name="get_goal",
            description=(
                "Read this session's goal: its exact id and revision, objective, phase, "
                "rounds completed, and blocker if any. Call this before update_goal and "
                "copy the id and revision it returns."
            ),
            parameters={"type": "object", "properties": {}},
            handler=tools.get,
            concurrency_safe=True,
        )
    )

    reg.register(
        Tool(
            name="update_goal",
            description=(
                "Update the goal, passing the exact goal_id and revision from get_goal. "
                "Mark complete only when the objective is actually achieved. Mark blocked "
                "only when the same concrete condition has persisted across several "
                "rounds — difficulty, uncertainty, or work remaining is not blocked."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "revision": {"type": "number"},
                    "action": {
                        "type": "string",
                        "enum": ["edit", "pause", "resume", "complete", "blocked"],
                    },
                    "objective": {"type": "string", "description": "Only with action edit."},
                    "max_rounds": {"type": "number", "description": "Only with action edit."},
                    "blocked_reason": {
                        "type": "string",
                        "description": "Required with action blocked: what is in the way.",
                    },
                },
                "required": ["goal_id", "revision", "action"],
            },
            handler=tools.update,
        )
    )


GOAL_GUIDANCE = """Use the goal tools for one long-running objective in this session — \
work that continues across rounds rather than finishing in this turn. You may infer that \
from the request without being asked for a goal; do not create one for ordinary \
single-turn work. Call `get_goal` before `update_goal` and copy its exact id and \
revision. Mark it complete only when the objective is actually achieved, and blocked only \
when the same concrete condition has persisted across several rounds."""


def register_goal_sections(prompt: Any) -> None:
    prompt.section("goal", GOAL_GUIDANCE, required=False)


def render_goal(session_id: str) -> str:
    """The active goal, for the session context block."""
    goal = _service().get(session_id)
    if goal is None or goal.phase in {"complete"}:
        return ""
    line = f"## Current goal\n{goal.objective}"
    line += f"\n- **Phase** — {goal.phase} (round {goal.rounds}/{goal.max_rounds})"
    if goal.blocked_reason:
        line += f"\n- **Blocked by** — {goal.blocked_reason}"
    line += f"\n- **Id / revision** — {goal.id} / {goal.revision}"
    return line
