"""The model-routing seam.

Keylane has one always-on ~9B model on the NPU and, optionally, a larger model
on the GPU. Which one serves a request is a property of *what the request is
for*, not of who is asking: the interactive turn must stay on the NPU so the
HUD answers quickly, while summarising a long transcript or grinding a
background objective should go to the bigger model when there is one.

So callers ask for a **route** — a purpose — and the seam resolves it to an
adapter. Resolution reads a declared preference order from config and never
depends on registration or import order, which is the property that makes a
provider swap safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Protocol, runtime_checkable

from seams.errors import LlmError

logger = logging.getLogger(__name__)

# Purpose, not model size. A new call site picks the route that describes what
# it is doing; the mapping to adapters stays in config.
ROUTES = ("interactive", "background", "utility")

# Used when config names no preference for a route.
_DEFAULT_ROUTES: dict[str, list[str]] = {
    # The HUD is waiting on this one.
    "interactive": ["npu"],
    # Subagents, compaction, scheduled work: latency does not matter, quality does.
    "background": ["gpu", "npu"],
    # Query planning, URL selection, titles — short, frequent, disposable.
    "utility": ["npu", "gpu"],
}


@runtime_checkable
class LlmAdapter(Protocol):
    """One way to reach a model. Implementations are trusted, same-process."""

    id: str

    def available(self) -> bool:
        """A cheap LOCAL check — never a network call.

        It answers "could this serve a request right now", so the seam can pick
        a usable adapter. It is not a health system.
        """

    @property
    def status(self) -> dict[str, Any]:
        """Human-readable state for Settings and /health."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system: str | None = None,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str: ...


class LlmRuntime:
    """Registry of adapters plus the route table that selects between them."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._adapters: dict[str, LlmAdapter] = {}
        self._routes: dict[str, list[str]] = dict(_DEFAULT_ROUTES)

    # ── registration ─────────────────────────────────────────────────────

    def register(self, adapter: LlmAdapter) -> Callable[[], None]:
        """Register one adapter and return the disposer that removes it."""
        with self._lock:
            if adapter.id in self._adapters:
                raise LlmError(
                    "LLM_DUPLICATE_ADAPTER",
                    f"an adapter with id {adapter.id!r} is already registered",
                )
            self._adapters[adapter.id] = adapter

        def _dispose() -> None:
            with self._lock:
                self._adapters.pop(adapter.id, None)

        return _dispose

    def configure_routes(self, routes: dict[str, Any]) -> None:
        """Replace the route table from settings. Unknown routes are ignored."""
        with self._lock:
            merged = dict(_DEFAULT_ROUTES)
            for route, preference in routes.items():
                if route not in ROUTES:
                    logger.warning("ignoring unknown llm route %r", route)
                    continue
                if isinstance(preference, str):
                    preference = [preference]
                merged[route] = [str(p) for p in preference if str(p).strip()]
            self._routes = merged

    def adapters(self) -> list[LlmAdapter]:
        with self._lock:
            return list(self._adapters.values())

    # ── resolution ───────────────────────────────────────────────────────

    def resolve(self, route: str = "interactive") -> LlmAdapter:
        """Pick the adapter serving ``route``, or raise a structured error.

        The preference list is declared, so the answer does not depend on which
        plugin happened to register first.
        """
        with self._lock:
            preference = self._routes.get(route)
            adapters = dict(self._adapters)

        if preference is None:
            raise LlmError("LLM_UNKNOWN_ROUTE", f"no such model route: {route!r}")
        if not adapters:
            raise LlmError("LLM_NO_ADAPTER", "no model adapters are registered")

        unknown: list[str] = []
        unavailable: list[str] = []
        for adapter_id in preference:
            adapter = adapters.get(adapter_id)
            if adapter is None:
                unknown.append(adapter_id)
                continue
            if not adapter.available():
                unavailable.append(adapter_id)
                continue
            return adapter

        if unavailable:
            raise LlmError(
                "LLM_ROUTE_UNAVAILABLE",
                f"the {route} route is served by {', '.join(unavailable)}, "
                "and none of them is ready yet",
                route=route,
            )
        raise LlmError(
            "LLM_ROUTE_MISSING",
            f"the {route} route names {', '.join(unknown) or 'nothing'}, "
            f"but the registered adapters are {', '.join(sorted(adapters)) or 'none'}",
            route=route,
        )

    def is_ready(self, route: str = "interactive") -> bool:
        try:
            self.resolve(route)
        except LlmError:
            return False
        return True

    # ── use ──────────────────────────────────────────────────────────────

    def generate(self, prompt: str, *, route: str = "interactive", **kwargs: Any) -> str:
        return self.resolve(route).generate(prompt, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        route: str = "interactive",
        **kwargs: Any,
    ) -> str:
        return self.resolve(route).chat(messages, **kwargs)

    def status(self) -> dict[str, Any]:
        with self._lock:
            adapters = dict(self._adapters)
            routes = dict(self._routes)
        return {
            "adapters": {
                aid: {"available": a.available(), **a.status} for aid, a in adapters.items()
            },
            "routes": {
                route: {
                    "preference": preference,
                    "resolved": next(
                        (
                            aid
                            for aid in preference
                            if aid in adapters and adapters[aid].available()
                        ),
                        None,
                    ),
                }
                for route, preference in routes.items()
            },
        }
