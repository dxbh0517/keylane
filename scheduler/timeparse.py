"""Turn the way people say "when" into an absolute local datetime.

The NPU model is small; asking it to emit correct ISO-8601 for "next Tuesday
at 9" fails often enough that it is worth parsing here instead. Everything
resolves against local time and is returned timezone-aware.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_UNITS = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "day": 86400,
    "days": 86400,
    "week": 604800,
    "weeks": 604800,
}

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_NAMED_TIMES = {
    "morning": (9, 0),
    "noon": (12, 0),
    "midday": (12, 0),
    "afternoon": (14, 0),
    "evening": (18, 0),
    "tonight": (20, 0),
    "night": (21, 0),
    "midnight": (0, 0),
}

_REL = re.compile(r"\bin\s+(\d+(?:\.\d+)?)\s*([a-z]+)", re.IGNORECASE)
_CLOCK = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)


def local_now() -> datetime:
    return datetime.now().astimezone()


def _with_time(day: datetime, hour: int, minute: int) -> datetime:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _find_clock(text: str) -> tuple[int, int] | None:
    for name, (hour, minute) in _NAMED_TIMES.items():
        if re.search(rf"\b{name}\b", text):
            return hour, minute
    match = _CLOCK.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif not meridiem and hour <= 7:
        # "remind me at 6" almost always means this evening, not 6am.
        hour += 12
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def parse_when(text: str, *, now: datetime | None = None) -> datetime | None:
    """Best-effort parse of *text* into an aware datetime, or None."""
    if not text:
        return None
    now = now or local_now()
    raw = text.strip()

    # 1. An explicit ISO timestamp always wins.
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.astimezone()
    except ValueError:
        pass

    lowered = raw.lower()

    # 2. Relative offsets: "in 30 minutes", "in 2 hours".
    match = _REL.search(lowered)
    if match:
        unit = _UNITS.get(match.group(2))
        if unit:
            return now + timedelta(seconds=float(match.group(1)) * unit)

    clock = _find_clock(lowered)

    # 3. Day anchors.
    if "day after tomorrow" in lowered:
        base = now + timedelta(days=2)
    elif "tomorrow" in lowered:
        base = now + timedelta(days=1)
    elif "today" in lowered or "tonight" in lowered:
        base = now
    else:
        base = None
        for name, index in _WEEKDAYS.items():
            if re.search(rf"\b{name}\b", lowered):
                ahead = (index - now.weekday()) % 7
                if ahead == 0 or "next" in lowered:
                    ahead = ahead or 7
                base = now + timedelta(days=ahead)
                break

    if base is not None:
        hour, minute = clock or (9, 0)
        return _with_time(base, hour, minute)

    # 4. A bare time means the next time it comes around.
    if clock:
        candidate = _with_time(now, *clock)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    return None


def to_utc_iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat()


def describe(when: datetime, *, now: datetime | None = None) -> str:
    """Short human phrasing for confirmations, e.g. 'today at 18:30'."""
    now = now or local_now()
    local = when.astimezone()
    delta_days = (local.date() - now.date()).days
    clock = local.strftime("%H:%M")
    if delta_days == 0:
        return f"today at {clock}"
    if delta_days == 1:
        return f"tomorrow at {clock}"
    if 1 < delta_days < 7:
        return f"{local.strftime('%A')} at {clock}"
    return local.strftime("%a %d %b at %H:%M")
