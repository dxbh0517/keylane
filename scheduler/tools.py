"""Scheduler tool registration."""

from __future__ import annotations

import json
import uuid

from scheduler.jobs import run_background, schedule_at, schedule_cron
from tools.registry import Tool, ToolRegistry


def register_scheduler_tools(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="schedule_task",
            description="Schedule a future task (cron or one-shot ISO datetime).",
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
        )
    )

    reg.register(
        Tool(
            name="run_background",
            description="Run an agent task in the background; notify when done.",
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            handler=lambda prompt: json.dumps({"job_id": run_background(prompt)}),
        )
    )


def _schedule(prompt: str, cron: str = "", run_at: str = "") -> str:
    tid = str(uuid.uuid4())[:8]
    if cron:
        schedule_cron(f"task-{tid}", cron, prompt)
        return json.dumps({"task_id": tid, "kind": "cron", "cron": cron})
    if run_at:
        schedule_at(f"task-{tid}", run_at, prompt)
        return json.dumps({"task_id": tid, "kind": "at", "run_at": run_at})
    return json.dumps({"error": "provide cron or run_at"})
