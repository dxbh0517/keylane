"""One long-running objective per session.

A todo list tracks the steps of a task the model is doing now. A goal is the
thing that outlives the turn: "keep working on X until it is done", carried
across background rounds, a restart, or a week.

Two rules make it trustworthy rather than decorative. Updates are
revision-checked, so a model working from a stale read cannot silently overwrite
a change. And `blocked` is refused until the same condition has survived several
rounds — difficulty is not a blocker, and a model that can declare itself
blocked on the first attempt will.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from memory.store import get_store
from seams.errors import SeamError

Phase = Literal["active", "paused", "complete", "blocked"]
ACTIONS = ("edit", "pause", "resume", "complete", "blocked")

# A blocker has to persist to be a blocker. Below this, "blocked" means
# "this turn was hard", which is not the same thing.
MIN_BLOCKED_ROUNDS = 3
DEFAULT_MAX_ROUNDS = 20


class GoalError(SeamError):
    """A goal operation was refused."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Goal:
    id: str
    session_id: str
    objective: str
    phase: Phase = "active"
    revision: int = 1
    rounds: int = 0
    max_rounds: int = DEFAULT_MAX_ROUNDS
    blocked_reason: str = ""

    def view(self) -> dict[str, Any]:
        """What the model reads before it may update. Includes the revision."""
        return {
            "goal_id": self.id,
            "revision": self.revision,
            "objective": self.objective,
            "phase": self.phase,
            "rounds_completed": self.rounds,
            "max_rounds": self.max_rounds,
            "blocked_reason": self.blocked_reason,
            "continuation_armed": self.phase == "active" and self.rounds < self.max_rounds,
        }


class GoalService:
    """One active goal per session, persisted so it survives a restart."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def _row_to_goal(self, row: Any) -> Goal:
        return Goal(
            id=row["id"],
            session_id=row["session_id"],
            objective=row["objective"],
            phase=row["phase"],
            revision=int(row["revision"]),
            rounds=int(row["rounds"]),
            max_rounds=int(row["max_rounds"] or DEFAULT_MAX_ROUNDS),
            blocked_reason=row["blocked_reason"] or "",
        )

    def get(self, session_id: str) -> Goal | None:
        with get_store()._connect() as conn:  # noqa: SLF001
            row = conn.execute(
                "SELECT * FROM goals WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return self._row_to_goal(row) if row else None

    def create(self, session_id: str, objective: str, max_rounds: int | None = None) -> Goal:
        text = str(objective).strip()
        if not text:
            raise GoalError("GOAL_INVALID", "a goal needs a concrete objective")

        existing = self.get(session_id)
        if existing and existing.phase in {"active", "paused"}:
            raise GoalError(
                "GOAL_EXISTS",
                f"this session already has a goal: {existing.objective!r}. "
                "Update it with update_goal rather than creating another.",
                goal_id=existing.id,
            )

        goal = Goal(
            id=f"goal-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            objective=text,
            max_rounds=int(max_rounds or DEFAULT_MAX_ROUNDS),
        )
        now = _utcnow()
        with self._lock, get_store()._connect() as conn:  # noqa: SLF001
            conn.execute(
                "INSERT INTO goals (id, session_id, objective, phase, revision, rounds, "
                "max_rounds, blocked_reason, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    goal.id, goal.session_id, goal.objective, goal.phase, goal.revision,
                    goal.rounds, goal.max_rounds, "", now, now,
                ),
            )
        return goal

    def update(
        self,
        session_id: str,
        *,
        goal_id: str,
        revision: int,
        action: str,
        objective: str = "",
        max_rounds: int | None = None,
        blocked_reason: str = "",
    ) -> Goal:
        if action not in ACTIONS:
            raise GoalError(
                "GOAL_INVALID_ACTION",
                f"action must be one of {', '.join(ACTIONS)}, not {action!r}",
            )

        with self._lock:
            goal = self.get(session_id)
            if goal is None or goal.id != goal_id:
                raise GoalError(
                    "GOAL_UNKNOWN",
                    f"no goal {goal_id!r} in this session; call get_goal first",
                )
            if int(revision) != goal.revision:
                raise GoalError(
                    "GOAL_STALE_REVISION",
                    f"you passed revision {revision} but the goal is at {goal.revision}. "
                    "Call get_goal and retry with the current values.",
                    current=goal.view(),
                )

            if action == "edit":
                if objective.strip():
                    goal.objective = objective.strip()
                if max_rounds is not None:
                    goal.max_rounds = int(max_rounds)
            elif action == "pause":
                goal.phase = "paused"
            elif action == "resume":
                goal.phase = "active"
            elif action == "complete":
                goal.phase = "complete"
            elif action == "blocked":
                if not blocked_reason.strip():
                    raise GoalError(
                        "GOAL_INVALID",
                        "blocked needs a concrete blocked_reason naming what is in the way",
                    )
                if goal.rounds < MIN_BLOCKED_ROUNDS:
                    raise GoalError(
                        "GOAL_TOO_EARLY_TO_BLOCK",
                        f"only {goal.rounds} round(s) have run. A goal is blocked when the "
                        f"same condition has persisted for {MIN_BLOCKED_ROUNDS}; difficulty "
                        "or remaining work is not a blocker. Keep going.",
                    )
                goal.phase = "blocked"
                goal.blocked_reason = blocked_reason.strip()

            goal.revision += 1
            self._persist(goal)
            return goal

    def record_round(self, session_id: str) -> Goal | None:
        """Count one autonomous continuation round against the goal's budget."""
        with self._lock:
            goal = self.get(session_id)
            if goal is None or goal.phase != "active":
                return goal
            goal.rounds += 1
            goal.revision += 1
            self._persist(goal)
            return goal

    def _persist(self, goal: Goal) -> None:
        with get_store()._connect() as conn:  # noqa: SLF001
            conn.execute(
                "UPDATE goals SET objective=?, phase=?, revision=?, rounds=?, "
                "max_rounds=?, blocked_reason=?, updated_at=? WHERE id=?",
                (
                    goal.objective, goal.phase, goal.revision, goal.rounds,
                    goal.max_rounds, goal.blocked_reason, _utcnow(), goal.id,
                ),
            )
