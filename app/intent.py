"""Lightweight intent helpers shared by the assistant, router and agent.

These are keyword heuristics — not a second model. They exist so a degraded
NPU path and the tool-catalog shortlist still recognise "check my email"
without handing the job to LM Studio.
"""

from __future__ import annotations

import re

MAIL_RE = re.compile(
    r"\b("
    r"e-?mails?|inbox|mailspring|unread|mailbox|"
    r"new\s+mail|check\s+(?:my\s+)?(?:e-?mail|inbox|mail)|"
    r"any\s+(?:new\s+)?(?:e-?mails?|mail)|"
    r"read\s+(?:my\s+)?(?:e-?mail|inbox|mail)"
    r")\b",
    re.IGNORECASE,
)

CALENDAR_RE = re.compile(
    r"\b("
    r"calendars?|agenda|appointments?|meetings?|"
    r"schedule|remind(?:er|ers|me)?|"
    r"what(?:'s|\s+is)\s+on\s+(?:my\s+)?(?:calendar|agenda|today|tomorrow)"
    r")\b",
    re.IGNORECASE,
)

# Prefer these MCP / builtin tool name fragments when the intent matches.
MAIL_TOOL_HINTS = ("mailspring.", "search_mail", "list_folders", "get_message", "list_threads")
CALENDAR_TOOL_HINTS = (
    "gnome-calendar.",
    "calendar.",
    "caldav.",
    "list_events",
    "upcoming",
    "create_event",
)


def is_mail_intent(message: str) -> bool:
    return bool(MAIL_RE.search(message or ""))


def is_calendar_intent(message: str) -> bool:
    return bool(CALENDAR_RE.search(message or ""))


def preferred_tool_hints(message: str) -> tuple[str, ...]:
    hints: list[str] = []
    if is_mail_intent(message):
        hints.extend(MAIL_TOOL_HINTS)
    if is_calendar_intent(message):
        hints.extend(CALENDAR_TOOL_HINTS)
    return tuple(hints)


def pick_mail_tool(tool_names: set[str] | list[str]) -> str | None:
    """Choose the best available mail-reading tool."""
    names = set(tool_names)
    for candidate in (
        "mailspring.search_mail",
        "mailspring.list_threads",
        "mailspring.list_folders",
    ):
        if candidate in names:
            return candidate
    for name in sorted(names):
        lowered = name.lower()
        if "mailspring" in lowered and any(
            tip in lowered for tip in ("search", "list", "get", "unread", "inbox")
        ):
            return name
    return None


def pick_calendar_tool(tool_names: set[str] | list[str], *, write: bool = False) -> str | None:
    names = set(tool_names)
    if write:
        for candidate in (
            "gnome-calendar.create_event",
            "calendar.create_event",
            "caldav.create_event",
        ):
            if candidate in names:
                return candidate
    for candidate in (
        "gnome-calendar.list_upcoming",
        "gnome-calendar.list_events",
        "calendar.list_upcoming",
        "calendar.list_events",
        "caldav.list_events",
    ):
        if candidate in names:
            return candidate
    for name in sorted(names):
        lowered = name.lower()
        if any(tip in lowered for tip in ("calendar", "event")) and (
            "list" in lowered or "upcoming" in lowered or (write and "create" in lowered)
        ):
            return name
    return None
