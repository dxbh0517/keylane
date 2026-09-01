"""System-prompt assembly.

The prompt is built from registered sections rather than written as one string,
for two reasons.

**It cannot drift.** A capability contributes its own paragraph when it
registers. Disable the tool and the paragraph goes with it, so the prompt never
promises something that is not there — which is exactly what a hand-written
`CORE` string cannot guarantee.

**It has a stable prefix.** Facts that change every turn — the clock, the memory
digest, the skill catalog — are not sections. They are *contexts*, materialized
as one user-role block appended after history and re-emitted only when their
content actually changed. Interpolating them into the system message made it
different on every single turn, so no prefix was ever reusable, on a device
where prefill is the expensive part.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

TextSource = str | Callable[[], str]

# Centrally owned placements, so a contributor names a position rather than
# inventing a number and colliding with someone else's.
SECTION_ORDER: dict[str, int] = {
    "identity": 10,
    "persona": 20,
    "scope": 30,
    "plan": 50,
    # Capability guidance: one paragraph per tool family, in the order a turn
    # would naturally reach for them.
    "memory": 100,
    "web": 110,
    "skills": 120,
    "todo": 130,
    "goal": 135,
    "schedule": 140,
    "jobs": 150,
    "subagent": 160,
    "shell": 170,
    "mcp": 180,
    "tools": 800,
    "output": 900,
    "tool_format": 950,
}

CONTEXT_ORDER: dict[str, int] = {
    "now": 10,
    "profile": 20,
    "memory": 30,
    "skills": 40,
    "todos": 50,
    "goal": 55,
    "inbox": 60,
}

CONTEXT_OPEN = "<session_context>"
CONTEXT_CLOSE = "</session_context>"

_VARIABLE = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")


class PromptError(ValueError):
    """A prompt could not be assembled. Always a programming error."""


@dataclass(frozen=True)
class PromptSection:
    """One ordered contribution to the static system prompt."""

    name: str
    order: int
    text: TextSource


@dataclass(frozen=True)
class PromptContext:
    """One ordered contribution to the dynamic per-turn block."""

    name: str
    order: int
    text: TextSource


@dataclass(frozen=True)
class Assembly:
    system: str
    context: str

    @property
    def context_digest(self) -> str:
        return digest_of(self.context)


def digest_of(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _resolve(source: TextSource) -> str:
    return (source() if callable(source) else source) or ""


class SystemPrompt:
    """Registry of prompt sections, dynamic contexts, and variables."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sections: dict[str, PromptSection] = {}
        self._contexts: dict[str, PromptContext] = {}
        self._variables: dict[str, Callable[[], str]] = {}

    # ── registration ─────────────────────────────────────────────────────

    def section(
        self,
        name: str,
        text: TextSource,
        *,
        order: int | str | None = None,
    ) -> Callable[[], None]:
        """Register one static section. Re-registering a name replaces it."""
        with self._lock:
            self._sections[name] = PromptSection(name, _order(SECTION_ORDER, name, order), text)
        return lambda: self._sections.pop(name, None)

    def context(
        self,
        name: str,
        text: TextSource,
        *,
        order: int | str | None = None,
    ) -> Callable[[], None]:
        """Register one dynamic context contribution."""
        with self._lock:
            self._contexts[name] = PromptContext(name, _order(CONTEXT_ORDER, name, order), text)
        return lambda: self._contexts.pop(name, None)

    def variable(self, name: str, provider: Callable[[], str]) -> Callable[[], None]:
        """Register a `{{name}}` value available to every section."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise PromptError(f"invalid prompt variable name: {name!r}")
        with self._lock:
            self._variables[name] = provider
        return lambda: self._variables.pop(name, None)

    def clear(self) -> None:
        with self._lock:
            self._sections.clear()
            self._contexts.clear()
            self._variables.clear()

    # ── assembly ─────────────────────────────────────────────────────────

    def render(self, text: str) -> str:
        """Interpolate `{{variables}}`, failing loudly on an unknown one.

        A missing value that renders as an empty string produces a prompt with
        a hole in it that nobody notices until the model answers oddly.
        """
        with self._lock:
            providers = dict(self._variables)

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            provider = providers.get(key)
            if provider is None:
                raise PromptError(f"prompt references undefined variable {{{{{key}}}}}")
            value = provider()
            if value is None:
                raise PromptError(f"prompt variable {{{{{key}}}}} resolved to nothing")
            return str(value)

        return _VARIABLE.sub(_sub, text)

    def assemble(self, extra_contexts: list[str] | None = None) -> Assembly:
        """Assemble the turn's prompt.

        `extra_contexts` carries per-session facts — the current goal — that
        cannot be registered globally because the registry is process-wide and
        the fact belongs to one conversation.
        """
        with self._lock:
            sections = sorted(self._sections.values(), key=lambda s: (s.order, s.name))
            contexts = sorted(self._contexts.values(), key=lambda c: (c.order, c.name))

        system_parts = [t for t in (_resolve(s.text).strip() for s in sections) if t]
        context_parts = [t for t in (_resolve(c.text).strip() for c in contexts) if t]
        context_parts += [t.strip() for t in (extra_contexts or []) if t and t.strip()]

        system = self.render("\n\n".join(system_parts))
        if not context_parts:
            return Assembly(system=system, context="")

        body = self.render("\n\n".join(context_parts))
        return Assembly(
            system=system,
            context=f"{CONTEXT_OPEN}\n{body}\n{CONTEXT_CLOSE}",
        )


def _order(table: dict[str, int], name: str, order: int | str | None) -> int:
    if isinstance(order, int):
        return order
    key = order if isinstance(order, str) else name
    if key not in table:
        raise PromptError(
            f"no registered placement named {key!r}; pass an explicit order "
            f"or add it to the table (known: {', '.join(sorted(table))})"
        )
    return table[key]


def latest_context_digest(messages: list[dict[str, str]]) -> str:
    """The digest of the newest context block still visible in history.

    Comparing against what the model can actually still see — rather than
    against a counter on this process — means a resumed session, a compacted
    one, and a fresh one all re-emit exactly when they should.
    """
    for message in reversed(messages):
        content = message.get("content", "")
        if CONTEXT_OPEN in content:
            return digest_of(content)
    return ""
