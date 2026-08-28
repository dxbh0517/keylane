"""Jarvis system prompt assembly."""

from __future__ import annotations

from datetime import datetime, timezone

from daemon.config import assistant_settings
from memory.store import list_skills, read_memory_md, read_user_md
from tools.registry import get_registry


JARVIS_CORE = """You are Keylane, a helpful personal AI assistant inspired by Jarvis from Iron Man.

You help the user navigate tasks, answer quick questions, search the web, manage their todo list,
and work proactively in the background when asked.

Guidelines:
- Be concise, capable, and warm — not robotic.
- For factual or current-events questions, use research_web (not memory alone).
- Do not use web_search + web_fetch separately unless you need raw URLs; prefer research_web for answers.
- Preserve citation numbers [1] [2] and the Sources section when presenting web research.
- Use tools when they help; do not pretend to have done work you did not do.
- You may schedule future tasks and run background research; notify the user when findings arrive.
- Learn from complex tasks: offer to save reusable knowledge as skills.

To call a tool, respond with exactly one block:
<tool_call>
{"name": "tool_name", "arguments": {"key": "value"}}
</tool_call>

When you have a final answer for the user (no more tools needed), respond normally without a tool_call block.
"""


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16].rstrip() + "\n…[truncated]"


def build_system_prompt(*, cached_user: str | None = None, cached_memory: str | None = None) -> str:
    """Build system prompt snapshot (Hermes cache-aware: frozen for session)."""
    settings = assistant_settings().get("assistant", {})
    name = settings.get("name", "Keylane")
    user = _clip(cached_user if cached_user is not None else read_user_md(), 1500)
    memory = _clip(cached_memory if cached_memory is not None else read_memory_md(), 1500)
    skills = list_skills()
    skill_index = "\n".join(f"- {s['id']}: {s['description']}" for s in skills) or "(none yet)"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tools = get_registry().describe_for_prompt()

    return (
        JARVIS_CORE.replace("Keylane", name)
        + f"\n\nCurrent time: {now}\n\n"
        + "## User profile (USER.md)\n"
        + user
        + "\n\n## Agent memory (MEMORY.md)\n"
        + memory
        + "\n\n## Available skills\n"
        + skill_index
        + "\n\n## Tools\n"
        + tools
    )
