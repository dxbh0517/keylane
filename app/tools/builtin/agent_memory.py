"""Memory and standing-goal tools for the assistant."""

from __future__ import annotations

from typing import Any

from app.agent_goals import (
    AgentGoal,
    default_interval_for,
    get_goal_store,
    infer_kind,
    parse_interval,
)
from app.memory_store import get_memory_store
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    bool_prop,
    int_prop,
    object_schema,
    string_prop,
)


class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = (
        "Search the user's long-term memory vault for people, preferences and "
        "facts that help decide what matters."
    )
    danger = ToolDanger.SAFE
    category = "memory"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "query": string_prop("What to look up."),
                "limit": int_prop("Max notes to return.", default=5),
            },
            required=["query"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        hits = get_memory_store().search(
            str(args.get("query") or ""),
            limit=int(args.get("limit") or 5),
        )
        if not hits:
            return ToolResult.success("No matching memory notes.", data={"hits": []})
        lines = [f"- {h['path']} (score {h['score']}): {h['snippet'][:240]}" for h in hits]
        return ToolResult.success("\n".join(lines), data={"hits": hits})


class MemoryReadTool(BaseTool):
    name = "memory_read"
    description = "Read one note from the memory vault (e.g. preferences.md, people.md)."
    danger = ToolDanger.SAFE
    category = "memory"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {"path": string_prop("Note path relative to memory/, e.g. people.md.")},
            required=["path"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        body = get_memory_store().read(str(args.get("path") or ""))
        if body is None:
            return ToolResult.failure("Memory note not found.")
        return ToolResult.success(body[:8000], data={"path": args.get("path")})


class MemoryWriteTool(BaseTool):
    name = "memory_write"
    description = (
        "Create or update a long-term memory note. Use for lasting facts about "
        "the user, people they care about, or recurring priorities."
    )
    danger = ToolDanger.SENSITIVE
    category = "memory"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "path": string_prop("Note path, e.g. people.md or daily/2026-08-25.md."),
                "content": string_prop("Markdown content to write."),
                "append": bool_prop("Append instead of replace.", default=False),
            },
            required=["path", "content"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            path = get_memory_store().write(
                str(args.get("path") or ""),
                str(args.get("content") or ""),
                append=bool(args.get("append")),
            )
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(f"Wrote memory/{path}.", data={"path": path})


class ListGoalsTool(BaseTool):
    name = "list_goals"
    description = "List standing agent goals (scheduled email/calendar checks, reminders)."
    danger = ToolDanger.SAFE
    category = "agent"

    async def run(self, args: dict[str, Any]) -> ToolResult:
        goals = get_goal_store().list()
        if not goals:
            return ToolResult.success("No standing goals yet.", data={"goals": []})
        lines = []
        for g in goals:
            state = "on" if g.enabled else "paused"
            lines.append(
                f"- [{g.id}] {g.title or g.kind} ({state}, every {g.interval_seconds}s): "
                f"{g.instruction[:120]}"
            )
        return ToolResult.success(
            "\n".join(lines),
            data={"goals": [g.model_dump(mode="json") for g in goals]},
        )


class SetGoalTool(BaseTool):
    name = "set_goal"
    description = (
        "Create or update a standing goal the always-on agent will run on a "
        "schedule (check email, watch calendar, remind the user, …). Prefer "
        "this over rewriting the system prompt."
    )
    danger = ToolDanger.SENSITIVE
    category = "agent"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "instruction": string_prop(
                    "Self-contained instructions for each tick. Include what to "
                    "check and what deserves notifying the user."
                ),
                "title": string_prop("Short label, e.g. 'Unread mail that matters'."),
                "kind": string_prop(
                    "email | calendar | reminder | general",
                    default="general",
                ),
                "interval": string_prop(
                    "How often: 'every 5 minutes', 'hourly', 'daily'. Omit to let "
                    "Keylane pick a default for the kind."
                ),
                "goal_id": string_prop("Existing id to update; omit to create."),
                "session_id": string_prop(
                    "Optional session to continue when the goal notifies."
                ),
            },
            required=["instruction"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        store = get_goal_store()
        goal_id = str(args.get("goal_id") or "").strip()
        existing = store.get(goal_id) if goal_id else None
        instruction = str(args.get("instruction") or "").strip()
        if not instruction:
            return ToolResult.failure("instruction is required.")

        kind = str(args.get("kind") or "").strip().lower() or infer_kind(instruction)
        interval = parse_interval(str(args.get("interval") or ""))
        if interval is None:
            interval = existing.interval_seconds if existing else default_interval_for(kind)

        goal = existing or AgentGoal(instruction=instruction)
        goal.instruction = instruction
        goal.title = str(args.get("title") or goal.title or instruction[:60])
        goal.kind = kind if kind in {"email", "calendar", "reminder", "general"} else "general"
        goal.interval_seconds = interval
        goal.enabled = True
        session_id = str(args.get("session_id") or "").strip()
        if session_id:
            goal.origin_session_id = session_id
        store.upsert(goal)
        return ToolResult.success(
            f"Standing goal '{goal.title}' saved (id={goal.id}, every {goal.interval_seconds}s).",
            data=goal.model_dump(mode="json"),
        )


class PauseGoalTool(BaseTool):
    name = "pause_goal"
    description = "Pause or resume a standing goal without deleting it."
    danger = ToolDanger.SENSITIVE
    category = "agent"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "goal_id": string_prop("Goal id from list_goals."),
                "enabled": bool_prop("false to pause, true to resume.", default=False),
            },
            required=["goal_id"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        store = get_goal_store()
        goal = store.get(str(args.get("goal_id") or ""))
        if goal is None:
            return ToolResult.failure("Unknown goal_id.")
        goal.enabled = bool(args.get("enabled"))
        store.upsert(goal)
        state = "resumed" if goal.enabled else "paused"
        return ToolResult.success(f"Goal {goal.id} {state}.", data=goal.model_dump(mode="json"))


class ForgetGoalTool(BaseTool):
    name = "forget_goal"
    description = "Delete a standing goal permanently."
    danger = ToolDanger.SENSITIVE
    category = "agent"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {"goal_id": string_prop("Goal id from list_goals.")},
            required=["goal_id"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        ok = get_goal_store().delete(str(args.get("goal_id") or ""))
        if not ok:
            return ToolResult.failure("Unknown goal_id.")
        return ToolResult.success("Goal deleted.")


def memory_and_agent_tools() -> list[BaseTool]:
    return [
        MemorySearchTool(),
        MemoryReadTool(),
        MemoryWriteTool(),
        ListGoalsTool(),
        SetGoalTool(),
        PauseGoalTool(),
        ForgetGoalTool(),
    ]
