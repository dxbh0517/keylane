"""Delegation to a child agent.

The point of a subagent is context: a self-contained piece of work runs in its
own conversation and only its *result* comes back, so a long research detour
does not push the actual question out of a 9B model's window.

Two rules are load-bearing on a small model, and both are checked before a child
exists rather than hoped for afterwards.

**Depth.** A child that can delegate can delegate forever. The cap is shared
with the job registry, because a background run and a delegation are the same
resource.

**Tools.** A child gets a restricted view of the parent's registry — the filter
removes the tool from its prompt *and* refuses to execute it, because visibility
and authority have to be the same thing or the model will keep trying.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from seams.errors import SubagentError

logger = logging.getLogger(__name__)

# A child inherits the parent's tools minus these. Nothing here is arbitrary:
# each either reaches the user (who is not watching the child), changes the
# machine, or lets the child delegate again.
DEFAULT_CHILD_DENY = (
    "subagent",
    "run_background",
    "schedule_task",
    "watch_create",
    "remind_me",
    "reminder_cancel",
    "memory_write",
    "skill_write",
    "ask_user",
    "notify_user",
    "desktop_open",
)

STOP_REASONS = ("completed", "aborted", "error", "refusal", "budget")


@dataclass(frozen=True)
class SubagentCapabilities:
    """What a provider supports, checked before `start` is called."""

    tool_filter: bool = False
    route_choice: bool = False
    persona: bool = False


@dataclass
class SubagentRequest:
    """What a caller asks for. The child sees only `prompt`."""

    prompt: str
    label: str = ""
    # A purpose, resolved by the LLM seam. Delegated work prefers the bigger
    # model precisely because nobody is watching the clock on it.
    route: str = "background"
    tool_deny: tuple[str, ...] = DEFAULT_CHILD_DENY
    tool_allow: tuple[str, ...] | None = None
    persona: str = ""
    cancel: threading.Event | None = None


@dataclass
class SubagentResult:
    """The terminal outcome. A non-completed reason means output may be partial."""

    output: str
    stop_reason: str = "completed"
    diagnostic: str = ""
    session_id: str = ""

    def view(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"output": self.output, "stop_reason": self.stop_reason}
        if self.diagnostic:
            payload["diagnostic"] = self.diagnostic
        return payload


class SubagentProvider(Protocol):
    id: str
    capabilities: SubagentCapabilities

    def available(self) -> bool: ...

    def start(self, request: SubagentRequest) -> SubagentResult: ...


class SubagentRuntime:
    """Named providers, with capability checks that fail loud before start."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._providers: dict[str, SubagentProvider] = {}
        self._preference: list[str] = []

    def register(self, provider: SubagentProvider) -> Callable[[], None]:
        with self._lock:
            if provider.id in self._providers:
                raise SubagentError(
                    "SUBAGENT_DUPLICATE_PROVIDER",
                    f"a subagent provider with id {provider.id!r} is already registered",
                )
            self._providers[provider.id] = provider
            if provider.id not in self._preference:
                self._preference.append(provider.id)
        return lambda: self._providers.pop(provider.id, None)

    def configure_preference(self, order: list[str]) -> None:
        with self._lock:
            self._preference = [str(p) for p in order if str(p).strip()]

    def resolve(self, provider_id: str = "") -> SubagentProvider:
        with self._lock:
            providers = dict(self._providers)
            preference = list(self._preference)

        if provider_id:
            provider = providers.get(provider_id)
            if provider is None:
                raise SubagentError(
                    "SUBAGENT_PROVIDER_MISSING",
                    f"no subagent provider named {provider_id!r}",
                )
            if not provider.available():
                raise SubagentError(
                    "SUBAGENT_PROVIDER_UNAVAILABLE",
                    f"the {provider_id} subagent provider is not ready",
                )
            return provider

        for candidate in preference:
            provider = providers.get(candidate)
            if provider is not None and provider.available():
                return provider
        raise SubagentError(
            "SUBAGENT_UNAVAILABLE",
            "no subagent provider is ready; check that a model is loaded",
        )

    def start(self, request: SubagentRequest, *, provider_id: str = "") -> SubagentResult:
        """Validate capabilities, then delegate. Never accept-then-ignore."""
        from seams.jobs import MAX_DEPTH, current_depth

        depth = current_depth()
        if depth >= MAX_DEPTH:
            raise SubagentError(
                "SUBAGENT_DEPTH_EXCEEDED",
                f"delegation is already {depth} levels deep; do this work directly",
            )

        provider = self.resolve(provider_id)
        caps = provider.capabilities
        if (request.tool_allow is not None or request.tool_deny) and not caps.tool_filter:
            raise SubagentError(
                "SUBAGENT_UNSUPPORTED_CAPABILITY",
                f"the {provider.id} provider cannot restrict a child's tools",
            )
        if request.persona and not caps.persona:
            raise SubagentError(
                "SUBAGENT_UNSUPPORTED_CAPABILITY",
                f"the {provider.id} provider does not support a per-child persona",
            )
        if not str(request.prompt).strip():
            raise SubagentError(
                "SUBAGENT_INVALID_REQUEST",
                "a subagent needs a complete, self-contained prompt — it cannot see "
                "this conversation",
            )
        return provider.start(request)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "providers": {
                    pid: {"available": p.available()} for pid, p in self._providers.items()
                },
                "preference": list(self._preference),
            }
