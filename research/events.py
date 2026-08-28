"""Research progress events via context variable."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

ResearchEventCallback = Callable[[str, dict[str, Any]], None]

_cb: ContextVar[ResearchEventCallback | None] = ContextVar("research_event_cb", default=None)


def set_research_callback(cb: ResearchEventCallback | None) -> None:
    _cb.set(cb)


def emit_research(kind: str, **payload: Any) -> None:
    cb = _cb.get()
    if cb:
        cb(kind, payload)
