"""Reminder, watcher, and background-task tools."""

from __future__ import annotations

import json
from typing import Any
import uuid

from scheduler.jobs import (
    cancel_task,
    create_reminder,
    create_watcher,
    list_tasks,
    run_background,
    schedule_at,
    schedule_cron,
)
from tools.registry import Tool, ToolRegistry


def register_scheduler_tools(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="remind_me",
            description=(
                "Remind the user about something at a given time. `when` accepts plain "
                "language: 'in 30 minutes', 'tomorrow at 9am', 'Friday 17:00'. Survives restarts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "What to remind the user about."},
                    "when": {"type": "string", "description": "When to fire, in plain language or ISO."},
                },
                "required": ["text", "when"],
            },
            handler=lambda text, when: json.dumps(create_reminder(text, when)),
        )
    )

    reg.register(
        Tool(
            name="reminders_list",
            description="List active reminders, watchers, and scheduled tasks with their ids.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: json.dumps(list_tasks()),
        )
    )

    reg.register(
        Tool(
            name="reminder_cancel",
            description="Cancel a reminder, watcher, or scheduled task by its id.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            handler=lambda id: json.dumps(cancel_task(id)),
        )
    )

    reg.register(
        Tool(
            name="watch_create",
            description=(
                "Create a recurring background check the user has agreed to — e.g. a morning "
                "sweep for today's calendar events or unanswered mail. Ask before creating one. "
                "cron is 5 fields in local time: 'minute hour day month day_of_week'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short name, e.g. 'morning-briefing'."},
                    "prompt": {"type": "string", "description": "What Keylane should do on each run."},
                    "cron": {"type": "string", "description": "e.g. '0 8 * * 1-5' for weekdays at 08:00."},
                },
                "required": ["name", "prompt", "cron"],
            },
            handler=lambda name, prompt, cron: json.dumps(create_watcher(name, prompt, cron)),
            dangerous=True,
        )
    )

    reg.register(
        Tool(
            name="schedule_task",
            description="Schedule one agent run: pass cron for recurring, or run_at for a one-shot ISO time.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "cron": {"type": "string"},
                    "run_at": {"type": "string"},
                },
                "required": ["prompt"],
            },
            handler=lambda prompt, cron="", run_at="": _schedule(prompt, cron, run_at),
            dangerous=True,
        )
    )

    reg.register(
        Tool(
            name="run_background",
            description=(
                "Run a longer task in the background and return its job id. Use for work "
                "the user should not wait on: tell them it started, then carry on. Read "
                "the result later with job_output, or stop it with job_kill."
            ),
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            handler=lambda prompt: json.dumps({"job_id": run_background(prompt), "status": "started"}),
        )
    )


def _schedule(prompt: str, cron: str = "", run_at: str = "") -> str:
    tid = uuid.uuid4().hex[:8]
    try:
        if cron:
            schedule_cron(f"task-{tid}", cron, prompt)
            return json.dumps({"task_id": f"task-{tid}", "kind": "cron", "cron": cron})
        if run_at:
            schedule_at(f"task-{tid}", run_at, prompt)
            return json.dumps({"task_id": f"task-{tid}", "kind": "at", "run_at": run_at})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {
            "error": "provide either cron or run_at",
            "hint": "For a plain reminder at a spoken time, use remind_me instead.",
        }
    )


SCHEDULE_GUIDANCE = """Use `remind_me` for a plain nudge at a time — say `when` in the \
user's own words ('in 30 minutes', 'tomorrow at 9am'); it is parsed for you. Use \
`run_background` for work that takes a while: tell the user it started, then stop, and \
the result reaches them on its own. `watch_create` sets up a recurring check — call \
`ask_user` to confirm before creating one, never set one up unasked. When a recurring \
check finds nothing worth reporting, stay quiet."""


def register_scheduler_sections(prompt: Any) -> None:
    prompt.section("schedule", SCHEDULE_GUIDANCE, required=False)
