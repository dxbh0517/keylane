"""Register all builtin tools."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daemon.config import assistant_settings
from daemon.paths import TODOS_PATH, ensure_data_dirs
from memory.store import (
    get_store,
    list_skills,
    load_skill,
    read_memory_md,
    read_user_md,
    save_skill,
    write_memory_md,
    write_user_md,
)
from tools.registry import Tool, get_registry


def _load_todos() -> list[dict[str, Any]]:
    ensure_data_dirs()
    if not TODOS_PATH.exists():
        return []
    return json.loads(TODOS_PATH.read_text(encoding="utf-8"))


def _save_todos(items: list[dict[str, Any]]) -> None:
    ensure_data_dirs()
    TODOS_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def register_builtin_tools() -> None:
    reg = get_registry()

    reg.register(
        Tool(
            name="memory_read",
            description="Read USER.md or MEMORY.md persistent memory files.",
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
            description="Overwrite USER.md or MEMORY.md with new content.",
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
            description="List available skills.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: json.dumps(list_skills()),
        )
    )

    reg.register(
        Tool(
            name="skill_read",
            description="Load a skill document by id.",
            parameters={
                "type": "object",
                "properties": {"skill_id": {"type": "string"}},
                "required": ["skill_id"],
            },
            handler=lambda skill_id: load_skill(skill_id),
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
            name="todos_list",
            description="List the user's todo items.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: json.dumps(_load_todos()),
        )
    )

    reg.register(
        Tool(
            name="todos_add",
            description="Add a todo item.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "due_at": {"type": "string"},
                },
                "required": ["title"],
            },
            handler=lambda title, due_at="": _todos_add(title, due_at),
        )
    )

    reg.register(
        Tool(
            name="todos_complete",
            description="Mark a todo done by id.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            handler=lambda id: _todos_complete(id),
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
    from scheduler.tools import register_scheduler_tools

    register_research_tools(reg)
    register_scheduler_tools(reg)


def _todos_add(title: str, due_at: str = "") -> str:
    items = _load_todos()
    tid = f"todo-{len(items)+1}"
    items.append(
        {
            "id": tid,
            "title": title,
            "done": False,
            "due_at": due_at,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_todos(items)
    return json.dumps({"id": tid, "title": title})


def _todos_complete(todo_id: str) -> str:
    items = _load_todos()
    for item in items:
        if item["id"] == todo_id:
            item["done"] = True
    _save_todos(items)
    return json.dumps({"completed": todo_id})


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
    allow = assistant_settings().get("security", {}).get("shell_allowlist", [])
    if command not in allow:
        return json.dumps({"error": f"command not allowlisted: {command}"})
    try:
        out = subprocess.run(
            [command, *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.dumps({"stdout": out.stdout, "stderr": out.stderr, "code": out.returncode})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
