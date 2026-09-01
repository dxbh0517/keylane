"""The three model-facing background-job controls.

One set of controls for every kind of background work — an agent run, a
delegation — because to the model they are the same thing: something with an id
that it can read from and stop.
"""

from __future__ import annotations

import json
from typing import Any

from tools.registry import Tool, ToolRegistry


def _jobs():
    from seams import get_context

    return get_context().jobs


def _job_list() -> str:
    return json.dumps({"jobs": _jobs().list()}, ensure_ascii=False)


def _job_output(job_id: str, wait: bool = False, timeout_seconds: float | None = None) -> str:
    return json.dumps(
        _jobs().output(job_id, wait=bool(wait), timeout_seconds=timeout_seconds),
        ensure_ascii=False,
    )


def _job_kill(job_id: str, reason: str = "") -> str:
    return json.dumps(_jobs().kill(job_id, reason), ensure_ascii=False)


def register_job_tools(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="job_list",
            description="List your background jobs, running and finished, with their ids and status.",
            parameters={"type": "object", "properties": {}},
            handler=_job_list,
            concurrency_safe=True,
        )
    )

    reg.register(
        Tool(
            name="job_output",
            description=(
                "Read a background job by id. Returns its result once it has finished. "
                "Reads do not block unless you set wait, which waits up to a capped "
                "timeout — only do that when you genuinely cannot continue without it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Id returned when the job started."},
                    "wait": {
                        "type": "boolean",
                        "description": "Block until it finishes or the timeout expires.",
                    },
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["job_id"],
            },
            handler=_job_output,
        )
    )

    reg.register(
        Tool(
            name="job_kill",
            description=(
                "Stop a running background job by id. Returns immediately; the job "
                "settles as killed once its work actually stops."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Optional short reason."},
                },
                "required": ["job_id"],
            },
            handler=_job_kill,
        )
    )


JOBS_GUIDANCE = """Track every background job id you start. You are told in this session \
when one finishes — do not poll it and do not wait on it; keep working on the parts that \
do not depend on it. Before you give a final answer, collect any still-relevant job with \
`job_output`, and `job_kill` the ones that stopped mattering."""


def register_job_sections(prompt: Any) -> None:
    prompt.section("jobs", JOBS_GUIDANCE, required=False)
