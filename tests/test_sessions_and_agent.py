"""Session store and standing goals."""

from __future__ import annotations

from pathlib import Path

from app.agent_goals import (
    AgentGoal,
    GoalStore,
    default_interval_for,
    is_silent_result,
    parse_interval,
)
from app.memory_store import MemoryStore
from app.sessions import SessionStore


def test_session_history_round_trip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.get_or_create()
    store.append_user(session, "check my mail")
    store.append_assistant(session, "You have 2 unread.")
    again = store.get(session.session_id)
    assert again is not None
    assert len(again.turns) == 2
    history = again.prompt_history()
    assert "check my mail" in history
    assert "2 unread" in history


def test_session_clear(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.get_or_create()
    store.append_user(session, "hello")
    store.clear(session.session_id)
    assert store.get(session.session_id) is None


def test_memory_search_and_write(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory")
    memory.write("people.md", "# People\n\n- Boss: alex@example.com\n")
    hits = memory.search("boss alex")
    assert hits
    assert "people.md" in hits[0]["path"]


def test_goal_schedule_and_silent(tmp_path: Path) -> None:
    store = GoalStore(tmp_path / "goals.json")
    goal = AgentGoal(
        title="Mail watch",
        instruction="Check unread mail that matters.",
        kind="email",
        interval_seconds=default_interval_for("email"),
    )
    store.upsert(goal)
    due = store.due()
    assert any(g.id == goal.id for g in due)
    store.mark_ran(goal.id, result="[SILENT]", noteworthy=False)
    refreshed = store.get(goal.id)
    assert refreshed is not None
    assert refreshed.next_run_at is not None
    assert is_silent_result("[SILENT]")
    assert is_silent_result("nothing new")
    assert not is_silent_result("Meeting invite from Alex tomorrow at 3pm")


def test_parse_interval() -> None:
    assert parse_interval("every 5 minutes") == 300
    assert parse_interval("hourly") == 3600
    assert parse_interval("daily") == 86400
