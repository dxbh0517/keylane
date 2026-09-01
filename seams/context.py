"""The capability context.

Every capability Keylane has is reached through a registry here rather than by
importing the one implementation directly. That is the whole of the pattern
worth taking from DSH: a capability is an *interface*, one or more *providers*,
and a *consumer* — usually a model-facing tool. Wiring consumers to the
interface is what lets a provider be swapped, restricted per agent, or stubbed
in a test without every call site knowing.

`get_context()` returns the process-wide context, built once on first use.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from seams.goals import GoalService
from seams.jobs import JobRegistry
from seams.llm import LlmRuntime
from seams.prompt import SystemPrompt
from seams.skills import SkillRegistry
from seams.subagent import SubagentRuntime

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """The registries a running Keylane composes."""

    llm: LlmRuntime = field(default_factory=LlmRuntime)
    prompt: SystemPrompt = field(default_factory=SystemPrompt)
    skills: SkillRegistry = field(default_factory=SkillRegistry)
    jobs: JobRegistry = field(default_factory=JobRegistry)
    subagents: SubagentRuntime = field(default_factory=SubagentRuntime)
    goals: GoalService = field(default_factory=GoalService)

    def status(self) -> dict[str, Any]:
        return {"llm": self.llm.status(), "subagents": self.subagents.status()}


_context: Context | None = None
_lock = threading.RLock()


def _build_llm(ctx: Context) -> None:
    """Register the configured model adapters and route table."""
    from daemon.config import model_settings
    from seams.llm_adapters import NpuAdapter, OpenAiCompatAdapter

    settings = model_settings()

    ctx.llm.register(NpuAdapter())

    for spec in settings.get("adapters", []) or []:
        adapter_id = str(spec.get("id", "")).strip()
        if not adapter_id or adapter_id == "npu":
            continue
        kind = str(spec.get("kind", "openai")).strip()
        if kind != "openai":
            logger.warning("unknown llm adapter kind %r for %r", kind, adapter_id)
            continue
        ctx.llm.register(
            OpenAiCompatAdapter(
                adapter_id=adapter_id,
                base_url=str(spec.get("base_url", "")),
                model=str(spec.get("model", "")),
                api_key=str(spec.get("api_key", "")),
                enabled=bool(spec.get("enabled", False)),
                timeout=float(spec.get("timeout_seconds", 180)),
                auto_unload=bool(spec.get("auto_unload", False)),
                idle_seconds=int(spec.get("idle_seconds", 60)),
            )
        )

    routes = settings.get("routes") or {}
    if routes:
        ctx.llm.configure_routes(dict(routes))


def _build_prompt(ctx: Context) -> None:
    """Register the prompt sections the assistant and its tools contribute.

    Tools register first so the capability paragraphs describe a tool set that
    actually exists — a paragraph about `web_search` in a build where the tool
    is disabled is a promise the model cannot keep.
    """
    from agent.prompt import register_core_sections
    from tools.builtin import register_builtin_tools, register_builtin_sections

    register_builtin_tools()
    register_core_sections(ctx.prompt)
    register_builtin_sections(ctx.prompt)


def _build_skills(ctx: Context) -> None:
    from seams.skills import LocalSkillProvider

    ctx.skills.register_provider(LocalSkillProvider())


def _build_subagents(ctx: Context) -> None:
    from seams.subagent_inproc import InProcessSubagentProvider

    ctx.subagents.register(InProcessSubagentProvider())


def build_context() -> Context:
    """Compose a fresh context. Used by `get_context` and by tests."""
    ctx = Context()
    _build_llm(ctx)
    _build_skills(ctx)
    _build_subagents(ctx)
    _build_prompt(ctx)
    return ctx


def get_context() -> Context:
    global _context
    if _context is None:
        with _lock:
            if _context is None:
                _context = build_context()
    return _context


def reset_context() -> None:
    """Drop the composed context so the next read rebuilds it from settings."""
    global _context
    with _lock:
        _context = None
