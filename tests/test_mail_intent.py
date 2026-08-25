"""Mail intent must use local tools, never fall through to a chat worker."""

from __future__ import annotations

from app.assistant import _heuristic_plan
from app.intent import is_mail_intent, pick_mail_tool


def test_mail_intent_detection() -> None:
    assert is_mail_intent("Do I have any new emails?")
    assert is_mail_intent("check my inbox")
    assert is_mail_intent("unread mail in mailspring")
    assert not is_mail_intent("open firefox")


def test_pick_mail_tool_prefers_search() -> None:
    names = {
        "mailspring.list_folders",
        "mailspring.search_mail",
        "delegate_to_worker",
    }
    assert pick_mail_tool(names) == "mailspring.search_mail"


def test_heuristic_plan_uses_mailspring_for_inbox() -> None:
    plan = _heuristic_plan(
        "do i have any new emails?",
        "Do I have any new emails?",
        None,
        {"mailspring.search_mail", "open_application"},
    )
    assert plan is not None
    tool, args = plan
    assert tool == "mailspring.search_mail"
    assert "query" in args


def test_heuristic_plan_without_mail_tools_returns_none() -> None:
    plan = _heuristic_plan(
        "do i have any new emails?",
        "Do I have any new emails?",
        None,
        {"open_application", "web_search"},
    )
    assert plan is None
