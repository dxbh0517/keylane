"""The `subagent` tool.

Foreground by default on a 9B model: background delegation is more useful when
the parent has something else to do, and a small model usually does not. Setting
`run_in_background` returns a job id instead, and the job controls read it.
"""

from __future__ import annotations

import json
from typing import Any

from seams.subagent import SubagentRequest
from tools.registry import Tool, ToolRegistry


def _ctx():
    from seams import get_context

    return get_context()


def subagent(description: str, prompt: str, run_in_background: bool = False) -> str:
    label = str(description).strip() or "delegated task"
    request = SubagentRequest(prompt=str(prompt), label=label)

    if not run_in_background:
        result = _ctx().subagents.start(request)
        return json.dumps(result.view(), ensure_ascii=False)

    def _work(job: Any) -> str:
        result = _ctx().subagents.start(
            SubagentRequest(prompt=request.prompt, label=label, cancel=job.cancel)
        )
        return json.dumps(result.view(), ensure_ascii=False)

    job = _ctx().jobs.start(kind="subagent", label=label, work=_work)
    return json.dumps(
        {
            "job_id": job.id,
            "status": "running",
            "note": "Started. Carry on; read it later with job_output.",
        },
        ensure_ascii=False,
    )


def register_subagent_tool(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="subagent",
            description=(
                "Delegate a self-contained task to a subagent — a separate agent working "
                "in its own context — so a long detour does not crowd out this "
                "conversation. It returns its result, not its steps. Give it a complete, "
                "standalone prompt: it cannot see this conversation, so include "
                "everything it needs. Waits for the result unless you set "
                "run_in_background, which returns a job id instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short (3-5 word) label for the delegated task.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The complete, self-contained task for the subagent.",
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": "Return a job id instead of waiting. Defaults to false.",
                    },
                },
                "required": ["description", "prompt"],
            },
            handler=subagent,
        )
    )


SUBAGENT_GUIDANCE = """Use `subagent` for a self-contained piece of work that would \
otherwise fill this conversation — a research detour, a long comparison. Write it a \
complete prompt: it cannot see anything you can see. You get its result, not its steps. \
Do not delegate something you can answer in one step, and do not delegate the whole \
request — that just adds a round trip."""


def register_subagent_sections(prompt: Any) -> None:
    prompt.section("subagent", SUBAGENT_GUIDANCE)
