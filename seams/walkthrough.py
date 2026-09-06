"""Guidance that takes more than one mark: a sequence of annotated steps.

The state lives in the daemon rather than in the turn that created it, because
the turn ends long before the user finishes step one. The agent sets a
walkthrough up and lets go; the UI drives it forward, one step at a time,
through ``/walkthrough/advance``.

Fifteen steps is the cap, and it is a real limit rather than a defensive
number: past about that many, guidance stops being "here is the next click"
and becomes a document the user would rather read than be walked through.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_STEPS = 15
MAX_CAPTION_CHARS = 200
# A walkthrough nobody has advanced in this long has been abandoned. Without
# this, a forgotten one keeps the UI polling the screen indefinitely.
IDLE_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class Step:
    """One step: what to say, and what to draw while saying it."""

    caption: str
    shapes: tuple[dict[str, Any], ...] = ()

    def payload(self) -> dict[str, Any]:
        """What the overlay is sent. TTL 0 — a step stays until it is passed."""
        return {
            "shapes": [dict(s) for s in self.shapes],
            "caption": self.caption,
            "ttl_ms": 0,
        }


@dataclass
class Walkthrough:
    """An ordered set of steps and how far through them the user is."""

    steps: tuple[Step, ...]
    title: str = ""
    index: int = 0
    started_at: float = field(default_factory=time.monotonic)
    touched_at: float = field(default_factory=time.monotonic)

    @property
    def total(self) -> int:
        return len(self.steps)

    @property
    def finished(self) -> bool:
        return self.index >= self.total

    @property
    def current(self) -> Step | None:
        return self.steps[self.index] if not self.finished else None

    @property
    def idle(self) -> bool:
        return time.monotonic() - self.touched_at > IDLE_TIMEOUT_SECONDS

    def advance(self) -> Step | None:
        """Move to the next step and return it, or None when done."""
        self.index += 1
        self.touched_at = time.monotonic()
        return self.current

    def describe(self) -> dict[str, Any]:
        step = self.current
        return {
            "title": self.title,
            "step": min(self.index + 1, self.total),
            "total": self.total,
            "finished": self.finished,
            "caption": step.caption if step else "",
            "annotation": step.payload() if step else None,
        }


def parse_steps(raw: Any) -> tuple[list[Step], str]:
    """Steps from what the model sent. Returns ``(steps, error)``."""
    if not isinstance(raw, list) or not raw:
        return [], "steps must be a non-empty array"
    if len(raw) > MAX_STEPS:
        return [], f"a walkthrough may have at most {MAX_STEPS} steps, not {len(raw)}"

    steps: list[Step] = []
    for position, entry in enumerate(raw, start=1):
        if isinstance(entry, str):
            entry = {"caption": entry}
        if not isinstance(entry, dict):
            return [], f"step {position} is not an object"
        caption = str(entry.get("caption", "") or entry.get("text", "")).strip()
        if not caption:
            return [], f"step {position} has no caption"
        shapes = entry.get("shapes")
        if shapes is None:
            shapes = []
        if not isinstance(shapes, list):
            return [], f"step {position}: shapes must be an array"
        steps.append(
            Step(
                caption=caption[:MAX_CAPTION_CHARS],
                shapes=tuple(s for s in shapes if isinstance(s, dict)),
            )
        )
    return steps, ""


class WalkthroughRegistry:
    """The one walkthrough in progress, if any.

    Deliberately singular. Two overlapping sets of arrows on one screen point
    at nothing, so starting a walkthrough replaces whatever was running.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: Walkthrough | None = None

    @property
    def active(self) -> Walkthrough | None:
        with self._lock:
            if self._active is not None and self._active.idle:
                logger.info("dropping an abandoned walkthrough")
                self._active = None
            return self._active

    def start(self, steps: list[Step], title: str = "") -> Walkthrough:
        with self._lock:
            self._active = Walkthrough(steps=tuple(steps), title=title)
            return self._active

    def advance(self) -> Walkthrough | None:
        """Step forward. Returns the walkthrough, or None if none is running."""
        with self._lock:
            if self._active is None:
                return None
            self._active.advance()
            if self._active.finished:
                done, self._active = self._active, None
                return done
            return self._active

    def stop(self) -> bool:
        with self._lock:
            had = self._active is not None
            self._active = None
            return had


_registry: WalkthroughRegistry | None = None
_registry_lock = threading.Lock()


def get_walkthroughs() -> WalkthroughRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = WalkthroughRegistry()
    return _registry
