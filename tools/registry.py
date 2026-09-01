"""The tool registry and its execution pipeline.

Two ideas from DSH shape this module.

**Scopes.** A registry can have a parent. A child sees everything its ancestors
registered, filtered by its own restriction, plus whatever it registered
itself — and its own registrations are exempt from the filter, so a delegated
child keeps the tools it exists to answer through. This is what makes a
subagent's tool filter one line instead of a second registry.

**A pipeline, not branches.** Execution runs pre-hooks, the handler, then
post-hooks. Permission gating, loop guards, and oversized-result spilling are
all hooks, so none of them is an `if` in the middle of `call()`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Any] | Callable[..., Awaitable[Any]]
EventSink = Callable[[str, dict[str, Any]], None]

# Tools that always go through the permission gate, whatever their flag says.
ALWAYS_GATED = frozenset(
    {"shell", "memory_write", "schedule_task", "watch_create", "skill_write"}
)

# Common model mistakes -> schema field names.
_ARG_ALIASES: dict[str, str] = {
    "query": "question",
    "q": "question",
    "text": "question",
    "prompt": "question",
    "search": "question",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    dangerous: bool = False
    # Cooperative deadline. The handler must actually be interruptible for this
    # to end the work rather than just the wait.
    timeout_ms: int | None = None
    # Whether this call may overlap with sibling calls in one step. Only an
    # explicit True opts in; anything that mutates shared state must not.
    concurrency_safe: bool = False


@dataclass
class ToolCall:
    """One accepted invocation, as the pipeline sees it."""

    name: str
    arguments: dict[str, Any]
    tool: Tool
    scope: str = "root"


@dataclass
class ToolOutcome:
    """The result of one call. `content` is exactly what the model will read."""

    content: str
    is_error: bool = False
    code: str = ""
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreDecision:
    """A pre-hook's verdict. `allow` lets later hooks and the handler run."""

    action: Literal["allow", "deny"]
    reason: str = ""
    code: str = ""


ALLOW = PreDecision("allow")


def deny(reason: str, code: str = "DENIED") -> PreDecision:
    return PreDecision("deny", reason=reason, code=code)


PreHook = Callable[[ToolCall], Any]        # -> PreDecision | None | Awaitable
PostHook = Callable[[ToolCall, ToolOutcome], Any]  # -> ToolOutcome | None | Awaitable


class ToolRegistry:
    def __init__(self, parent: "ToolRegistry | None" = None, *, scope: str = "root") -> None:
        self._parent = parent
        self._scope = scope
        self._tools: dict[str, Tool] = {}
        self._allow: set[str] | None = None
        self._deny: set[str] = set()
        self._pre: list[PreHook] = []
        self._post: list[PostHook] = []

    # ── composition ──────────────────────────────────────────────────────

    def child(self, scope: str) -> "ToolRegistry":
        """A registry scoped to one agent, inheriting this one's tools."""
        return ToolRegistry(self, scope=scope)

    def restrict(
        self,
        *,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
    ) -> Callable[[], None]:
        """Filter what this scope inherits. Its own registrations are exempt.

        A deny-only filter still admits tools registered later; an allow list
        excludes them. Restrictions intersect rather than replace.
        """
        known = set(self._inherited().keys())
        unknown = [n for n in (list(allow or []) + list(deny or [])) if n not in known]
        if unknown:
            # Loud, because a typo that silently restricts nothing is a filter
            # that quietly does not exist.
            raise ValueError(f"cannot restrict unknown tools: {', '.join(sorted(unknown))}")

        previous_allow = None if self._allow is None else set(self._allow)
        previous_deny = set(self._deny)
        if allow is not None:
            self._allow = set(allow) if self._allow is None else (self._allow & set(allow))
        if deny:
            self._deny |= set(deny)

        def _dispose() -> None:
            self._allow = previous_allow
            self._deny = previous_deny

        return _dispose

    def register(self, tool: Tool) -> Callable[[], None]:
        self._tools[tool.name] = tool

        def _dispose() -> None:
            self._tools.pop(tool.name, None)

        return _dispose

    def add_pre_hook(self, hook: PreHook) -> Callable[[], None]:
        """Run before every call in this scope. Return a deny to stop it."""
        self._pre.append(hook)
        return lambda: self._pre.remove(hook) if hook in self._pre else None

    def add_post_hook(self, hook: PostHook) -> Callable[[], None]:
        """Inspect or replace every outcome in this scope."""
        self._post.append(hook)
        return lambda: self._post.remove(hook) if hook in self._post else None

    # ── visibility ───────────────────────────────────────────────────────

    def _inherited(self) -> dict[str, Tool]:
        return self._parent.visible() if self._parent else {}

    def visible(self) -> dict[str, Tool]:
        """Every tool this scope can see, after its restriction."""
        merged: dict[str, Tool] = {}
        for name, tool in self._inherited().items():
            if self._allow is not None and name not in self._allow:
                continue
            if name in self._deny:
                continue
            merged[name] = tool
        merged.update(self._tools)  # own registrations are exempt
        return merged

    def get(self, name: str) -> Tool | None:
        return self.visible().get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.visible().values()
        ]

    def describe_for_prompt(self) -> str:
        """Compact tool list for NPU context limits (~1024 tokens)."""
        lines = []
        for t in self.visible().values():
            props = t.parameters.get("properties", {})
            keys = ", ".join(props.keys()) if props else ""
            lines.append(f"- {t.name}({keys}): {t.description}")
        return "\n".join(lines)

    # ── execution ────────────────────────────────────────────────────────

    def _normalize_arguments(self, tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        props = set(tool.parameters.get("properties", {}).keys())
        out = dict(arguments)
        for alias, canonical in _ARG_ALIASES.items():
            if alias in out and canonical in props and canonical not in out:
                out[canonical] = out.pop(alias)
        return out

    def _hook_chain(self, kind: str) -> list[Any]:
        """This scope's hooks, outermost ancestor first."""
        inherited = self._parent._hook_chain(kind) if self._parent else []
        own = self._pre if kind == "pre" else self._post
        return [*inherited, *own]

    async def _run_pre(self, call: ToolCall) -> PreDecision:
        for hook in self._hook_chain("pre"):
            decision = hook(call)
            if inspect.isawaitable(decision):
                decision = await decision
            if decision is not None and decision.action == "deny":
                return decision
        return ALLOW

    async def _run_post(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        for hook in self._hook_chain("post"):
            replaced = hook(call, outcome)
            if inspect.isawaitable(replaced):
                replaced = await replaced
            if replaced is not None:
                outcome = replaced
        return outcome

    async def _invoke(self, call: ToolCall) -> ToolOutcome:
        tool = call.tool
        try:
            result = tool.handler(**call.arguments)
            if inspect.isawaitable(result):
                if tool.timeout_ms:
                    result = await asyncio.wait_for(result, tool.timeout_ms / 1000)
                else:
                    result = await result
            content = result if isinstance(result, str) else json.dumps(
                result, ensure_ascii=False, default=str
            )
            return ToolOutcome(content=content)
        except asyncio.TimeoutError:
            return ToolOutcome(
                content=json.dumps(
                    {
                        "error": f"{tool.name} did not finish within "
                        f"{tool.timeout_ms} ms",
                        "code": "TOOL_TIMEOUT",
                    }
                ),
                is_error=True,
                code="TOOL_TIMEOUT",
            )
        except TypeError as exc:
            # Name the arguments the handler wanted so the next attempt is a
            # corrected call rather than the same one again.
            sig = inspect.signature(tool.handler)
            expected = [
                p.name
                for p in sig.parameters.values()
                if p.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            ]
            return ToolOutcome(
                content=json.dumps(
                    {
                        "error": str(exc),
                        "code": "INVALID_ARGS",
                        "expected_arguments": expected,
                        "received": list(call.arguments.keys()),
                    }
                ),
                is_error=True,
                code="INVALID_ARGS",
            )
        except Exception as exc:  # noqa: BLE001
            from seams.errors import SeamError

            logger.exception("tool %s failed", tool.name)
            if isinstance(exc, SeamError):
                return ToolOutcome(
                    content=json.dumps(exc.as_dict(), ensure_ascii=False, default=str),
                    is_error=True,
                    code=exc.code,
                )
            return ToolOutcome(
                content=json.dumps({"error": str(exc), "code": "TOOL_FAILED"}),
                is_error=True,
                code="TOOL_FAILED",
            )

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_event: EventSink | None = None,
    ) -> ToolOutcome:
        """Run one call through the pipeline and return its structured outcome."""
        tool = self.get(name)
        if not tool:
            known = ", ".join(sorted(self.visible())) or "none"
            return ToolOutcome(
                content=json.dumps(
                    {
                        "error": f"unknown tool: {name}",
                        "code": "UNKNOWN_TOOL",
                        "available": known,
                    }
                ),
                is_error=True,
                code="UNKNOWN_TOOL",
            )

        call = ToolCall(
            name=name,
            arguments=self._normalize_arguments(tool, arguments),
            tool=tool,
            scope=self._scope,
        )
        # Hooks that need to reach the UI — the permission prompt, above all —
        # read the sink off the call rather than taking another parameter.
        setattr(call, "on_event", on_event)

        decision = await self._run_pre(call)
        if decision.action == "deny":
            return await self._run_post(
                call,
                ToolOutcome(
                    content=json.dumps(
                        {"error": decision.reason, "code": decision.code or "DENIED"}
                    ),
                    is_error=True,
                    code=decision.code or "DENIED",
                ),
            )

        started = time.monotonic()
        outcome = await self._invoke(call)
        outcome.duration_ms = int((time.monotonic() - started) * 1000)
        return await self._run_post(call, outcome)

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_event: EventSink | None = None,
    ) -> str:
        """String-returning shim over :meth:`execute`."""
        outcome = await self.execute(name, arguments, on_event=on_event)
        return outcome.content


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _registry
