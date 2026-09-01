"""Register all builtin tools."""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from daemon.config import assistant_settings
from memory.store import (
    forget_memory,
    get_store,
    list_inbox,
    list_memories,
    mark_inbox_read,
    read_memory_md,
    read_user_md,
    save_memory,
    save_skill,
    search_memories,
    write_memory_md,
    write_user_md,
)
from tools.policy import install_default_policy
from tools.registry import Tool, get_registry


def _skill_registry():
    from seams import get_context

    return get_context().skills


def _skill_read(skill_id: str) -> str:
    """Load one skill, refusing before the body is read if policy says no."""
    from seams.skills import NAME_PATTERN, render_skill

    name = str(skill_id).strip()
    if not NAME_PATTERN.match(name):
        return f'Error: invalid skill name "{name}"'
    skill = _skill_registry().get(name)
    if skill is None:
        return f'Error: skill "{name}" is unknown or no longer available'
    if not skill.invocation.model_invocable:
        return f'Error: skill "{name}" is not available for you to load'
    return render_skill(skill)


def register_builtin_tools() -> None:
    reg = get_registry()

    # Policy first: every tool registered below inherits it.
    install_default_policy(reg)

    reg.register(
        Tool(
            name="remember",
            description=(
                "Save one durable fact about the user so it is available in future "
                "conversations. One fact per call, written as a standalone sentence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The fact, e.g. 'Sister's birthday is 3 March'."},
                    "kind": {
                        "type": "string",
                        "enum": ["user", "preference", "fact", "project", "contact"],
                        "description": "user=identity, preference=how to work with them, contact=people.",
                    },
                },
                "required": ["text"],
            },
            handler=lambda text, kind="fact", tags="": json.dumps(save_memory(text, kind=kind, tags=tags)),
        )
    )

    reg.register(
        Tool(
            name="recall",
            description="Search saved memories for facts about the user before answering personal questions.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda query: json.dumps(search_memories(query)),
        )
    )

    reg.register(
        Tool(
            name="forget",
            description="Delete a saved memory by id when the user says it is wrong or no longer true.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            handler=lambda id: json.dumps({"forgotten": id} if forget_memory(id) else {"error": f"no memory {id}"}),
        )
    )

    reg.register(
        Tool(
            name="memories_list",
            description="List saved memories with their ids, newest first.",
            parameters={
                "type": "object",
                "properties": {"kind": {"type": "string"}},
            },
            handler=lambda kind="": json.dumps(list_memories(kind or None)),
        )
    )

    reg.register(
        Tool(
            name="memory_read",
            description="Read the USER.md or MEMORY.md notes file verbatim.",
            parameters={
                "type": "object",
                "properties": {"file": {"type": "string", "enum": ["USER", "MEMORY"]}},
                "required": ["file"],
            },
            handler=lambda file: read_user_md() if file == "USER" else read_memory_md(),
        )
    )

    reg.register(
        Tool(
            name="memory_write",
            description=(
                "Replace the whole USER.md or MEMORY.md file. Rarely correct — to add a "
                "single fact use `remember` instead, which cannot lose the existing ones."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "enum": ["USER", "MEMORY"]},
                    "content": {"type": "string"},
                },
                "required": ["file", "content"],
            },
            handler=lambda file, content: (
                write_user_md(content) or "saved USER.md"
                if file == "USER"
                else write_memory_md(content) or "saved MEMORY.md"
            ),
            dangerous=True,
        )
    )

    reg.register(
        Tool(
            name="inbox_list",
            description="List results from background tasks and reminders the user has not seen yet.",
            parameters={
                "type": "object",
                "properties": {"unread_only": {"type": "boolean"}},
            },
            handler=lambda unread_only=True: json.dumps(list_inbox(bool(unread_only))),
        )
    )

    reg.register(
        Tool(
            name="inbox_mark_read",
            description="Mark one inbox item read, or all of them when no id is given.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
            handler=lambda id="": json.dumps({"marked": mark_inbox_read(id or None)}),
        )
    )

    reg.register(
        Tool(
            name="session_search",
            description="Full-text search across past conversation sessions.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda query: json.dumps(get_store().search_sessions(query)),
        )
    )

    reg.register(
        Tool(
            name="skill_list",
            description="List the skills you can load, with their descriptions.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: json.dumps(
                [
                    {"name": s.name, "description": s.description, "when_to_use": s.when_to_use}
                    for s in _skill_registry().for_model()
                ]
            ),
        )
    )

    reg.register(
        Tool(
            name="skill_read",
            description=(
                "Load the full instructions for an available skill. Call it with the "
                "exact name from the available-skills list before acting on a task that "
                "names or clearly matches that skill."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "The exact skill name from the available skills list.",
                    }
                },
                "required": ["skill_id"],
            },
            handler=_skill_read,
        )
    )

    reg.register(
        Tool(
            name="skill_write",
            description="Create or update a SKILL.md file (agentskills.io frontmatter).",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["skill_id", "content"],
            },
            handler=lambda skill_id, content: save_skill(skill_id, content) or f"saved skill {skill_id}",
        )
    )

    reg.register(
        Tool(
            name="notify_user",
            description="Send a desktop notification; optionally speak via TTS.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "speak": {"type": "boolean"},
                },
                "required": ["title", "body"],
            },
            handler=lambda title, body, speak=False: _notify(title, body, speak),
        )
    )

    reg.register(
        Tool(
            name="desktop_open",
            description="Open a URL or file with xdg-open.",
            parameters={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            handler=lambda target: _desktop_open(target),
        )
    )

    reg.register(
        Tool(
            name="shell",
            description="Run an allowlisted shell command.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["command"],
            },
            handler=lambda command, args=None: _shell(command, args or []),
            dangerous=True,
        )
    )

    from research.tools import register_research_tools
    from tools.ask_user import register_ask_user_tool
    from tools.jobs_tools import register_job_tools
    from tools.subagent_tool import register_subagent_tool
    from tools.todos import register_todo_tools

    register_ask_user_tool(reg)
    register_subagent_tool(reg)
    register_job_tools(reg)
    register_todo_tools(reg)
    register_research_tools(reg)
    try:
        from scheduler.tools import register_scheduler_tools

        register_scheduler_tools(reg)
    except ImportError as exc:
        logger.warning("scheduler tools unavailable: %s", exc)


def _notify(title: str, body: str, speak: bool = False) -> str:
    from notify.desktop import send_notification

    send_notification(title, body)
    if speak:
        from notify.tts_gate import speak_text

        speak_text(body)
    return json.dumps({"sent": True})


def _desktop_open(target: str) -> str:
    subprocess.run(["xdg-open", target], check=False)
    return json.dumps({"opened": target})


def _shell(command: str, args: list[str]) -> str:
    from daemon.paths import ROOT
    from daemon.shellpolicy import CommandNotAllowed, check_command, read_roots

    security = assistant_settings().get("security", {})
    allow = list(security.get("shell_allowlist", []))
    roots = read_roots(security.get("shell_read_roots"))
    try:
        check_command(command, list(args), allowlist=allow, roots=roots)
    except CommandNotAllowed as exc:
        return json.dumps({"error": str(exc)})

    try:
        out = subprocess.run(
            [command, *args],
            capture_output=True,
            text=True,
            timeout=30,
            # A bare `grep -r pattern` searches the working directory, so anchor
            # it inside a permitted root rather than wherever the daemon started.
            cwd=str(roots[0] if roots else ROOT),
        )
        return json.dumps({"stdout": out.stdout, "stderr": out.stderr, "code": out.returncode})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


# ── prompt guidance ──────────────────────────────────────────────────────
#
# One imperative paragraph per capability, registered beside the tools it
# describes. A capability that is not registered contributes no paragraph, so
# the prompt cannot promise a tool the model does not have.

MEMORY_GUIDANCE = """Use `recall` before answering anything about the user's own life — \
"when is my…", "what did I say about…". Use `remember` the moment they tell you something \
durable: their name, where they work, how they like answers, people they mention, \
deadlines, project details. One fact per call, phrased so it still makes sense a month \
from now. Do not save passwords, tokens, one-off chatter, or anything already in this \
conversation. Use `forget` when they correct you or say something is no longer true."""

SKILL_GUIDANCE = """Use `skill_read` to load the full instructions for an available skill. \
Call it with the exact skill id from the available-skills list before acting on a task \
that names or clearly matches that skill. Load every applicable skill, then follow their \
instructions."""

SHELL_GUIDANCE = """Use `shell` only for the allowlisted read-only commands. File \
arguments must sit inside the permitted directories; a refusal names the reason, so read \
it and pick a different path rather than retrying."""

NOTIFY_GUIDANCE = """Use `notify_user` when something needs the user's attention while \
they are not looking at the answer card. Use `desktop_open` to put a URL or file in front \
of them. Neither replaces answering the question."""


def register_builtin_sections(prompt: Any) -> None:
    """Contribute the builtin capabilities' prompt guidance."""
    prompt.section("memory", MEMORY_GUIDANCE)
    prompt.section("skills", SKILL_GUIDANCE)
    prompt.section("shell", SHELL_GUIDANCE)
    prompt.section("mcp", NOTIFY_GUIDANCE)

    from research.tools import register_research_sections
    from tools.ask_user import register_ask_user_sections
    from tools.jobs_tools import register_job_sections
    from tools.goal_tools import register_goal_sections
    from tools.subagent_tool import register_subagent_sections
    from tools.todos import register_todo_sections

    register_ask_user_sections(prompt)
    register_goal_sections(prompt)
    register_subagent_sections(prompt)
    register_job_sections(prompt)
    register_todo_sections(prompt)
    register_research_sections(prompt)
    try:
        from scheduler.tools import register_scheduler_sections

        register_scheduler_sections(prompt)
    except ImportError as exc:
        logger.warning("scheduler guidance unavailable: %s", exc)
