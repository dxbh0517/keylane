"""Goals: one per session, revision-checked, and hard to declare blocked."""

from __future__ import annotations

import pytest

from seams.goals import MIN_BLOCKED_ROUNDS, GoalError, GoalService


@pytest.fixture()
def goals(tmp_path, monkeypatch):
    """A GoalService over a throwaway database."""
    from memory.store import SessionStore

    store = SessionStore(tmp_path / "test.db")
    # seams.goals binds get_store at import, so the module's own name is what
    # has to be replaced.
    monkeypatch.setattr("seams.goals.get_store", lambda: store)
    return GoalService()


def test_a_goal_is_created_with_a_revision(goals) -> None:
    goal = goals.create("s1", "Migrate the API to v2")
    assert goal.revision == 1
    assert goal.phase == "active"
    assert goals.get("s1").objective == "Migrate the API to v2"


def test_a_session_has_only_one_live_goal(goals) -> None:
    goals.create("s1", "First objective")
    with pytest.raises(GoalError) as exc:
        goals.create("s1", "Second objective")
    assert exc.value.code == "GOAL_EXISTS"


def test_two_sessions_keep_separate_goals(goals) -> None:
    goals.create("s1", "One")
    goals.create("s2", "Two")
    assert goals.get("s1").objective == "One"
    assert goals.get("s2").objective == "Two"


def test_an_empty_objective_is_refused(goals) -> None:
    with pytest.raises(GoalError, match="concrete objective"):
        goals.create("s1", "   ")


def test_a_session_without_a_goal_reads_as_none(goals) -> None:
    assert goals.get("s1") is None


# ── revision checking ────────────────────────────────────────────────────


def test_an_update_advances_the_revision(goals) -> None:
    goal = goals.create("s1", "Ship it")
    updated = goals.update("s1", goal_id=goal.id, revision=goal.revision, action="pause")
    assert updated.phase == "paused"
    assert updated.revision == goal.revision + 1


def test_a_stale_revision_is_refused_with_the_current_state(goals) -> None:
    """Two rounds working from one read must not overwrite each other."""
    goal = goals.create("s1", "Ship it")
    goals.update("s1", goal_id=goal.id, revision=goal.revision, action="pause")

    with pytest.raises(GoalError) as exc:
        goals.update("s1", goal_id=goal.id, revision=goal.revision, action="complete")
    assert exc.value.code == "GOAL_STALE_REVISION"
    assert exc.value.detail["current"]["phase"] == "paused"


def test_an_unknown_goal_id_is_refused(goals) -> None:
    goal = goals.create("s1", "Ship it")
    with pytest.raises(GoalError, match="no goal"):
        goals.update("s1", goal_id="goal-nope", revision=goal.revision, action="pause")


def test_an_unknown_action_names_the_valid_ones(goals) -> None:
    goal = goals.create("s1", "Ship it")
    with pytest.raises(GoalError) as exc:
        goals.update("s1", goal_id=goal.id, revision=goal.revision, action="abandon")
    assert "complete" in exc.value.message


# ── blocking ─────────────────────────────────────────────────────────────


def test_a_goal_cannot_be_blocked_on_the_first_round(goals) -> None:
    """A model that can give up immediately will."""
    goal = goals.create("s1", "Ship it")
    with pytest.raises(GoalError) as exc:
        goals.update(
            "s1",
            goal_id=goal.id,
            revision=goal.revision,
            action="blocked",
            blocked_reason="hard",
        )
    assert exc.value.code == "GOAL_TOO_EARLY_TO_BLOCK"


def test_a_goal_blocks_once_the_condition_has_persisted(goals) -> None:
    goal = goals.create("s1", "Ship it")
    for _ in range(MIN_BLOCKED_ROUNDS):
        goal = goals.record_round("s1")

    blocked = goals.update(
        "s1",
        goal_id=goal.id,
        revision=goal.revision,
        action="blocked",
        blocked_reason="the deploy key is missing",
    )
    assert blocked.phase == "blocked"
    assert blocked.blocked_reason == "the deploy key is missing"


def test_blocking_needs_a_concrete_reason(goals) -> None:
    goal = goals.create("s1", "Ship it")
    for _ in range(MIN_BLOCKED_ROUNDS):
        goal = goals.record_round("s1")
    with pytest.raises(GoalError, match="blocked_reason"):
        goals.update("s1", goal_id=goal.id, revision=goal.revision, action="blocked")


# ── rounds ───────────────────────────────────────────────────────────────


def test_rounds_only_count_while_the_goal_is_active(goals) -> None:
    goal = goals.create("s1", "Ship it")
    goals.record_round("s1")
    goal = goals.get("s1")
    goals.update("s1", goal_id=goal.id, revision=goal.revision, action="pause")
    goals.record_round("s1")
    assert goals.get("s1").rounds == 1


def test_continuation_disarms_at_the_round_cap(goals) -> None:
    goals.create("s1", "Ship it", max_rounds=2)
    for _ in range(2):
        goals.record_round("s1")
    assert goals.get("s1").view()["continuation_armed"] is False


def test_a_completed_goal_leaves_the_context_block(goals, monkeypatch) -> None:
    from tools import goal_tools

    monkeypatch.setattr(goal_tools, "_service", lambda: goals)
    goal = goals.create("s1", "Ship it")
    assert "Ship it" in goal_tools.render_goal("s1")

    goals.update("s1", goal_id=goal.id, revision=goal.revision, action="complete")
    assert goal_tools.render_goal("s1") == ""


def test_the_context_block_carries_the_id_and_revision(goals, monkeypatch) -> None:
    """The model needs both to update, so it should never have to guess."""
    from tools import goal_tools

    monkeypatch.setattr(goal_tools, "_service", lambda: goals)
    goal = goals.create("s1", "Ship it")
    rendered = goal_tools.render_goal("s1")
    assert goal.id in rendered and f"/ {goal.revision}" in rendered
