"""Standing goals — Hermes-style scheduled agent work.

Goals are created by the user (or by the assistant when asked to "watch my
email"). The gateway ticks every minute, runs due goals in fresh sessions, and
notifies only when something needs attention. Quiet ticks stay silent.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.config import ROOT

logger = logging.getLogger(__name__)

GOALS_PATH = ROOT / "data" / "agent_goals.json"
SILENT_MARKER = "[SILENT]"

# Default cadences when the user does not specify one.
DEFAULT_INTERVALS = {
    "email": 300,       # 5 minutes
    "calendar": 900,    # 15 minutes
    "reminder": 3600,   # hourly
    "general": 1800,    # 30 minutes
}


class AgentGoal(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:10])
    title: str = ""
    instruction: str
    """Self-contained prompt run on each tick (no chat history)."""

    kind: str = "general"  # email | calendar | reminder | general
    enabled: bool = True
    interval_seconds: int = 300
    """How often to wake. The agent may adjust this when preferences change."""

    allowed_tools: list[str] = Field(default_factory=list)
    """When non-empty, only these tools may run during the goal tick."""

    auto_confirm: list[str] = Field(default_factory=list)
    """Read-only tools that may run unattended for this goal."""

    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    last_result: str = ""
    last_noteworthy: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    origin_session_id: str | None = None
    """If set, noteworthy results continue this conversation (Hermes continuable)."""

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def schedule_next(self, *, from_time: datetime | None = None) -> None:
        base = from_time or datetime.now(timezone.utc)
        seconds = max(60, int(self.interval_seconds or 300))
        self.next_run_at = base + timedelta(seconds=seconds)


class GoalStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or GOALS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._goals: dict[str, AgentGoal] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load agent goals: %s", exc)
            return
        for entry in raw.get("goals") or []:
            try:
                goal = AgentGoal.model_validate(entry)
                self._goals[goal.id] = goal
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping bad goal entry: %s", exc)

    def _save(self) -> None:
        payload = {
            "goals": [g.model_dump(mode="json") for g in self._goals.values()]
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def list(self, *, enabled_only: bool = False) -> list[AgentGoal]:
        with self._lock:
            goals = list(self._goals.values())
        goals.sort(key=lambda g: g.title or g.id)
        if enabled_only:
            goals = [g for g in goals if g.enabled]
        return goals

    def get(self, goal_id: str) -> AgentGoal | None:
        with self._lock:
            return self._goals.get(goal_id)

    def upsert(self, goal: AgentGoal) -> AgentGoal:
        goal.touch()
        if goal.next_run_at is None:
            # New goals are due on the next scheduler tick, then recur.
            goal.next_run_at = datetime.now(timezone.utc)
        with self._lock:
            self._goals[goal.id] = goal
            self._save()
        return goal

    def delete(self, goal_id: str) -> bool:
        with self._lock:
            if goal_id not in self._goals:
                return False
            del self._goals[goal_id]
            self._save()
            return True

    def due(self, *, now: datetime | None = None) -> list[AgentGoal]:
        moment = now or datetime.now(timezone.utc)
        due: list[AgentGoal] = []
        for goal in self.list(enabled_only=True):
            if goal.next_run_at is None or goal.next_run_at <= moment:
                due.append(goal)
        return due

    def mark_ran(
        self,
        goal_id: str,
        *,
        result: str,
        noteworthy: bool,
        now: datetime | None = None,
    ) -> AgentGoal | None:
        moment = now or datetime.now(timezone.utc)
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return None
            goal.last_run_at = moment
            goal.last_result = (result or "")[:4000]
            goal.last_noteworthy = noteworthy
            goal.schedule_next(from_time=moment)
            goal.touch()
            self._save()
            return goal


def infer_kind(instruction: str) -> str:
    text = (instruction or "").lower()
    if re.search(r"\b(email|inbox|mail|mailspring)\b", text):
        return "email"
    if re.search(r"\b(calendar|meeting|agenda|appointment|remind)\b", text):
        return "calendar"
    if re.search(r"\bremind\b", text):
        return "reminder"
    return "general"


def default_interval_for(kind: str) -> int:
    return DEFAULT_INTERVALS.get(kind, DEFAULT_INTERVALS["general"])


def parse_interval(text: str) -> int | None:
    """Parse 'every 5 minutes', 'hourly', 'daily' into seconds."""
    raw = (text or "").strip().lower()
    if not raw:
        return None
    if raw in {"hourly", "every hour"}:
        return 3600
    if raw in {"daily", "every day", "each day"}:
        return 86400
    if raw in {"weekly", "every week"}:
        return 604800
    match = re.search(
        r"(?:every\s+)?(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)",
        raw,
    )
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("sec"):
        return max(60, amount)
    if unit.startswith("min"):
        return max(60, amount * 60)
    if unit.startswith("hr") or unit.startswith("hour"):
        return max(60, amount * 3600)
    if unit.startswith("day"):
        return max(60, amount * 86400)
    return None


def is_silent_result(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if SILENT_MARKER in cleaned.upper().replace(" ", ""):
        return True
    # Also treat near-empty canvases / "nothing new" as silent.
    lowered = cleaned.lower()
    if lowered in {"nothing new", "no new mail", "no changes", "ok", "done"}:
        return True
    if "nothing that needs" in lowered or "no emails that require" in lowered:
        return True
    return False


_goals: GoalStore | None = None


def get_goal_store() -> GoalStore:
    global _goals
    if _goals is None:
        _goals = GoalStore()
    return _goals
