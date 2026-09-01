"""The todo list, as one whole-list write.

Three tools — list, add, complete — meant the model had to read, find an id, and
mutate, which a 9B model gets wrong often enough to matter, and which is where
the colliding `todo-{len+1}` ids came from. DSH's shape removes the problem
rather than fixing it: send the entire list every time, and items need no
identity at all.

The current list is shown in the session context block, so there is nothing to
read before writing.
"""

from __future__ import annotations

import json
from typing import Any

from daemon.paths import TODOS_PATH, ensure_data_dirs
from tools.registry import Tool, ToolRegistry

STATUSES = ("pending", "in_progress", "completed")


def load_todos() -> list[dict[str, str]]:
    """The current list, migrating the old title/done shape on the way."""
    ensure_data_dirs()
    if not TODOS_PATH.exists():
        return []
    try:
        raw = json.loads(TODOS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []

    items: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if "content" in entry and entry.get("status") in STATUSES:
            items.append({"content": str(entry["content"]), "status": str(entry["status"])})
        elif "title" in entry:
            items.append(
                {
                    "content": str(entry["title"]),
                    "status": "completed" if entry.get("done") else "pending",
                }
            )
    return items


def save_todos(items: list[dict[str, str]]) -> None:
    ensure_data_dirs()
    TODOS_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate(todos: Any) -> list[dict[str, str]]:
    if not isinstance(todos, list):
        raise ValueError("todos must be an array of {content, status} objects")
    items: list[dict[str, str]] = []
    for entry in todos:
        if isinstance(entry, str):
            entry = {"content": entry, "status": "pending"}
        if not isinstance(entry, dict):
            raise ValueError("each todo must be an object with content and status")
        content = str(entry.get("content", "")).strip()
        if not content:
            raise ValueError("each todo needs a non-empty content line")
        status = str(entry.get("status", "pending")).strip()
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}, not {status!r}")
        items.append({"content": content, "status": status})
    return items


def todo_write(todos: Any) -> str:
    items = _validate(todos)
    save_todos(items)
    remaining = sum(1 for i in items if i["status"] != "completed")
    return json.dumps({"todos": items, "remaining": remaining}, ensure_ascii=False)


def render_todos() -> str:
    """The list as the model sees it in the session context block."""
    items = load_todos()
    if not items:
        return ""
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    lines = "\n".join(f"{marks[i['status']]} {i['content']}" for i in items)
    return f"## Todo list\n{lines}"


def register_todo_tools(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="todo_write",
            description=(
                "Record and update the task list. Send the ENTIRE list every call — it "
                "REPLACES the previous one; there are no partial updates and no per-item "
                "edits. Use it to plan multi-step work and show progress: add one todo "
                "per concrete step before you start. Mark a todo in_progress when you "
                "begin it and completed the moment it is done — do not batch completions. "
                "While work remains, at least one todo should be in_progress. Skip the "
                "list entirely for trivial single-step tasks. The current list is already "
                "shown to you, so you never need to read it first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The COMPLETE list, replacing any previous one.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "What the task is — a short imperative line.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": list(STATUSES),
                                    "description": "pending | in_progress | completed",
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
            handler=todo_write,
        )
    )


TODO_GUIDANCE = """Use `todo_write` to plan multi-step work and show progress: one todo \
per concrete step, written before you start. Send the whole list every time — it replaces \
the previous one. Mark a step completed the moment it is done rather than batching them \
at the end, and keep one step in_progress while work remains. Skip the list for anything \
that is a single step."""


def register_todo_sections(prompt: Any) -> None:
    prompt.section("todo", TODO_GUIDANCE)
    prompt.context("todos", render_todos)
