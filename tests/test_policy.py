"""Pipeline policy: loop guard, spill, and the permission gate."""

from __future__ import annotations

import asyncio
import json

import pytest

from tools.guards import THRESHOLDS, repeat_reminder_hook, reset_repeat_chain
from tools.registry import Tool, ToolCall, ToolOutcome, ToolRegistry


def _call(name: str = "web_fetch", scope: str = "s1", **args) -> ToolCall:
    return ToolCall(
        name=name,
        arguments=args,
        tool=Tool(name=name, description="", parameters={}, handler=lambda: ""),
        scope=scope,
    )


@pytest.fixture(autouse=True)
def _fresh_chains():
    for scope in ("s1", "s2"):
        reset_repeat_chain(scope)
    yield
    for scope in ("s1", "s2"):
        reset_repeat_chain(scope)


def _repeat(times: int, scope: str = "s1", **args) -> list[ToolOutcome]:
    outcomes = []
    for _ in range(times):
        outcome = ToolOutcome(content="same result")
        repeat_reminder_hook(_call(scope=scope, **args), outcome)
        outcomes.append(outcome)
    return outcomes


def test_the_first_repeats_pass_without_comment() -> None:
    for outcome in _repeat(2, url="https://example.com"):
        assert "system-reminder" not in outcome.content


def test_a_reminder_arrives_at_the_first_threshold() -> None:
    outcome = _repeat(THRESHOLDS[0], url="https://example.com")[-1]
    assert "system-reminder" in outcome.content
    assert "Read the previous result" in outcome.content


def test_later_reminders_name_the_repeated_arguments() -> None:
    outcome = _repeat(THRESHOLDS[1], url="https://example.com")[-1]
    assert "https://example.com" in outcome.content
    assert "change the approach" in outcome.content


def test_the_guard_is_advisory_and_never_blocks() -> None:
    outcome = _repeat(THRESHOLDS[0], url="https://example.com")[-1]
    assert outcome.is_error is False
    assert outcome.content.startswith("same result")


def test_different_arguments_break_the_chain() -> None:
    _repeat(2, url="https://a.com")
    outcome = ToolOutcome(content="x")
    repeat_reminder_hook(_call(url="https://b.com"), outcome)
    assert "system-reminder" not in outcome.content


def test_argument_key_order_does_not_start_a_new_chain() -> None:
    """The same call spelled two ways is still the same call."""
    for _ in range(THRESHOLDS[0] - 1):
        repeat_reminder_hook(_call(a="1", b="2"), ToolOutcome(content="x"))
    outcome = ToolOutcome(content="x")
    repeat_reminder_hook(_call(b="2", a="1"), outcome)
    assert "system-reminder" in outcome.content


def test_one_scopes_loop_does_not_nudge_another() -> None:
    _repeat(THRESHOLDS[0] - 1, scope="s1", url="https://a.com")
    outcome = ToolOutcome(content="x")
    repeat_reminder_hook(_call(scope="s2", url="https://a.com"), outcome)
    assert "system-reminder" not in outcome.content


def test_a_new_user_message_clears_the_chain() -> None:
    _repeat(THRESHOLDS[0] - 1, url="https://a.com")
    reset_repeat_chain("s1")
    outcome = ToolOutcome(content="x")
    repeat_reminder_hook(_call(url="https://a.com"), outcome)
    assert "system-reminder" not in outcome.content


@pytest.mark.parametrize("name", ["job_list", "inbox_list", "memories_list", "get_goal"])
def test_polling_a_list_tool_is_not_a_loop(name: str) -> None:
    """Checking the same list repeatedly is what these tools are for."""
    outcomes = _repeat(THRESHOLDS[-1] + 1, name=name)
    assert all("system-reminder" not in o.content for o in outcomes)


# ── spill ────────────────────────────────────────────────────────────────


@pytest.fixture()
def spill_root(tmp_path, monkeypatch):
    from seams import spill

    monkeypatch.setattr(spill, "_store", spill.SpillStore(tmp_path))
    return tmp_path


def test_a_short_result_stays_inline(spill_root) -> None:
    from tools.policy import spill_hook

    outcome = ToolOutcome(content="short")
    assert spill_hook(_call(), outcome) is None
    assert outcome.content == "short"


def test_an_oversized_result_is_saved_not_destroyed(spill_root) -> None:
    from seams.spill import MAX_INLINE_CHARS
    from tools.policy import spill_hook

    body = "x" * (MAX_INLINE_CHARS + 5000) + "TAILMARK"
    outcome = ToolOutcome(content=body)
    replaced = spill_hook(_call(), outcome)

    assert replaced is not None
    saved = list(spill_root.rglob("*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == body


def test_the_preview_keeps_both_ends_and_says_what_is_missing(spill_root) -> None:
    from seams.spill import MAX_INLINE_CHARS
    from tools.policy import spill_hook

    body = "HEADMARK" + "x" * (MAX_INLINE_CHARS + 5000) + "TAILMARK"
    outcome = ToolOutcome(content=body)
    spill_hook(_call(), outcome)

    assert "HEADMARK" in outcome.content
    assert "TAILMARK" in outcome.content
    assert "characters omitted" in outcome.content


def test_the_model_is_told_how_to_read_the_rest(spill_root) -> None:
    from seams.spill import MAX_INLINE_CHARS
    from tools.policy import spill_hook

    outcome = ToolOutcome(content="x" * (MAX_INLINE_CHARS + 100))
    spill_hook(_call(), outcome)
    assert "grep" in outcome.content
    assert outcome.meta["spill"].endswith(".txt")


def test_an_oversized_error_is_left_alone(spill_root) -> None:
    from seams.spill import MAX_INLINE_CHARS
    from tools.policy import spill_hook

    outcome = ToolOutcome(content="x" * (MAX_INLINE_CHARS + 100), is_error=True)
    assert spill_hook(_call(), outcome) is None


def test_two_sessions_spill_to_different_directories(spill_root) -> None:
    from seams.spill import MAX_INLINE_CHARS
    from tools.policy import spill_hook

    for scope in ("s1", "s2"):
        spill_hook(_call(scope=scope), ToolOutcome(content="x" * (MAX_INLINE_CHARS + 100)))
    assert len({p.parent for p in spill_root.rglob("*.txt")}) == 2


# ── the permission gate ──────────────────────────────────────────────────


def _gated_registry():
    from tools.policy import permission_hook

    reg = ToolRegistry(scope="s1")
    reg.register(
        Tool(
            name="shell",
            description="",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ran",
            dangerous=True,
        )
    )
    reg.register(
        Tool(name="recall", description="", parameters={"type": "object", "properties": {}}, handler=lambda: "[]")
    )
    reg.add_pre_hook(permission_hook)
    return reg


def test_a_denied_tool_never_runs(monkeypatch) -> None:
    monkeypatch.setattr("daemon.config.permission_mode", lambda name: "deny")
    outcome = asyncio.run(_gated_registry().execute("shell", {}))
    assert outcome.is_error and outcome.code == "PERMISSION_DENIED"


def test_an_ungated_tool_ignores_the_mode(monkeypatch) -> None:
    """`recall` prompting on every call would make the assistant unusable."""
    monkeypatch.setattr("daemon.config.permission_mode", lambda name: "deny")
    assert asyncio.run(_gated_registry().execute("recall", {})).content == "[]"


def test_auto_mode_runs_a_dangerous_tool(monkeypatch) -> None:
    monkeypatch.setattr("daemon.config.permission_mode", lambda name: "auto")
    assert asyncio.run(_gated_registry().execute("shell", {})).content == "ran"


# ── ask_user ─────────────────────────────────────────────────────────────


def test_a_question_raises_a_prompt_and_waits_for_the_answer() -> None:
    from daemon.permissions import get_pending, respond
    from tools.ask_user import ask_user

    async def _scenario() -> str:
        task = asyncio.create_task(ask_user("Set up a morning sweep?", options=["Yes", "No"]))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if get_pending():
                break
        pending = get_pending()[0]
        assert pending["arguments"]["question"] == "Set up a morning sweep?"
        assert pending["arguments"]["options"][0]["label"] == "Yes"
        respond(pending["id"], True)
        return await task

    assert json.loads(asyncio.run(_scenario())) == {"answered": True, "approved": True}


def test_a_question_is_not_gated_by_a_permission_mode(monkeypatch) -> None:
    """A question is not a dangerous action; `auto` must not skip the prompt."""
    from daemon.permissions import get_pending, respond
    from tools.ask_user import ask_user

    monkeypatch.setattr("daemon.config.permission_mode", lambda name: "auto")

    async def _scenario() -> str:
        task = asyncio.create_task(ask_user("Which one?"))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if get_pending():
                break
        assert get_pending(), "auto mode swallowed the question"
        respond(get_pending()[0]["id"], False)
        return await task

    assert json.loads(asyncio.run(_scenario()))["answered"] is False


def test_an_unanswered_question_tells_the_model_to_carry_on(monkeypatch) -> None:
    import tools.ask_user as module

    monkeypatch.setattr(module, "ASK_TIMEOUT_SECONDS", 0.05)
    payload = json.loads(asyncio.run(module.ask_user("Anyone there?")))
    assert payload["answered"] is False
    assert "Proceed with what you have" in payload["note"]


def test_an_empty_question_is_rejected() -> None:
    from tools.ask_user import ask_user

    payload = json.loads(asyncio.run(ask_user("   ")))
    assert payload["code"] == "INVALID_ARGS"
