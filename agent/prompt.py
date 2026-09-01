"""The turn's prompt: a stable static prefix plus a dynamic context block.

`register_core_sections()` contributes the parts that belong to the assistant
itself — who it is, how it answers, how it calls a tool. Every capability
contributes its own guidance paragraph next to its tool registration, so this
module never has to know which tools exist.

`assemble_for_turn()` returns both halves. The loop puts the static half in the
system message, where it stays byte-identical across turns, and appends the
dynamic half as a user-role block only when its content changed.
"""

from __future__ import annotations

from datetime import datetime

from daemon.config import assistant_settings
from memory.store import memory_digest, read_user_md
from seams.prompt import Assembly, SystemPrompt
from tools.registry import ToolRegistry, get_registry

IDENTITY = """You are {{assistant_name}}, a personal AI assistant running locally on \
{{user_name}}'s Linux desktop. You are their second brain: you answer quickly, remember \
what matters, and carry out work in the background so they do not have to hold it in \
their head."""

SCOPE = """## Scope
- Answer questions, search the web, manage todos and reminders, and run background tasks.
- You act on this machine only. You have no access to anything the user has not connected.
- If you cannot do something, say so plainly in one sentence and offer the nearest thing you can do.
- Call a tool when it gets a better answer than guessing. Never describe work you did not do.
- If a tool returns an error, read it — it names the reason — and correct the call once.
  Do not repeat an identical call; it will return the same thing."""

OUTPUT = """## Answering
Your reply is rendered as a small formatted card, so write it as a short document, not
as a paragraph of prose.

- **First line is the answer**, in one sentence. It is shown on its own as the headline,
  so it must stand alone and contain the actual result — not "Here is what I found".
- Then, only if there is more worth saying, add structure:
  - `## Short Heading` to label a section (two or three words).
  - `- **Term** — value` for facts, versions, dates, options. Prefer this over prose.
  - `1.` numbered steps for anything sequential.
  - ` ```lang ` fenced blocks for commands or code. Never put a command in a sentence.
  - `> ` for a caveat or prerequisite.
- Keep every line short — the card is narrow. Aim for under 12 words a bullet.
- No filler openers, no restating the question, no offers to help further, no sign-off.
- Do not write a Sources section and do not add [1] style citation markers; the interface
  renders attribution itself.
- Never output reasoning traces, thinking tags, or tool_call markup in a reply to the user."""

TOOL_FORMAT = """## Tool call format
To call a tool, reply with exactly one block and nothing else:
<tool_call>
{"name": "tool_name", "arguments": {"key": "value"}}
</tool_call>

When you are done and have the answer, reply in plain text with no markup."""


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16].rstrip() + "\n…[truncated]"


def _assistant_name() -> str:
    return str(assistant_settings().get("assistant", {}).get("name", "Keylane"))


def _user_name() -> str:
    name = str(assistant_settings().get("assistant", {}).get("user_name", "") or "").strip()
    return name or "the user"


def _now_context() -> str:
    # Local time, because reminders are set in the user's day, not in UTC.
    stamp = datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %Z")
    return f"Local time: {stamp}"


def _profile_context() -> str:
    profile = _clip(read_user_md(), 1200)
    return f"## User profile\n{profile}" if profile else ""


def _memory_context() -> str:
    recall = memory_digest()
    if not recall or recall.startswith("(nothing"):
        return ""
    return f"## What you remember\n{recall}"


def _skills_context() -> str:
    """The catalog: names and descriptions only, never bodies.

    Only model-invocable skills appear. A skill the user can invoke with
    `/name` but the model may not load itself is deliberately absent — listing
    it would invite exactly the call the policy refuses.
    """
    from seams import get_context

    skills = get_context().skills.for_model()
    if not skills:
        return ""
    entries = "\n".join(
        f"- `{s.name}`: {s.description}" + (f" (use when: {s.when_to_use})" if s.when_to_use else "")
        for s in skills
    )
    return (
        "## Available skills\n"
        "A skill is a reusable set of task-specific instructions.\n\n"
        f"{entries}\n\n"
        "If the user names a skill, or the task clearly matches one's description, call "
        "`skill_read` with the exact skill name before taking task actions. This list "
        "contains summaries only; do not infer or follow a skill's instructions until "
        "you have loaded it. If a skill's instructions already appear in this "
        "conversation, follow those and do not load it again."
    )


def register_core_sections(
    prompt: SystemPrompt,
    *,
    registry: ToolRegistry | None = None,
) -> None:
    """Register the sections and contexts the assistant itself owns."""
    registry = registry or get_registry()

    prompt.variable("assistant_name", _assistant_name)
    prompt.variable("user_name", _user_name)

    prompt.section("identity", IDENTITY)
    prompt.section("scope", SCOPE)
    prompt.section("tools", lambda: f"## Tools\n{registry.describe_for_prompt()}")
    prompt.section("output", OUTPUT)
    prompt.section("tool_format", TOOL_FORMAT)

    prompt.context("now", _now_context)
    prompt.context("profile", _profile_context)
    prompt.context("memory", _memory_context)
    prompt.context("skills", _skills_context)


def assemble_for_turn(
    prompt: SystemPrompt | None = None,
    *,
    extra_contexts: list[str] | None = None,
) -> Assembly:
    from seams import get_context

    return (prompt or get_context().prompt).assemble(extra_contexts=extra_contexts)


def build_system_prompt(
    *,
    cached_user: str | None = None,
    cached_memory: str | None = None,
) -> str:
    """The static prompt alone. Kept for callers that want one string."""
    return assemble_for_turn().system
