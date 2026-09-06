"""Put text where the caret is.

The one entry point is :func:`type_text`. It picks a provider, applies the
safety rules from :mod:`inject.base`, and returns the name of whatever
actually delivered the string so the caller can say so.

Selection order is a property of the session rather than a preference, with one
exception: on a keyboard layout whose letters are not where ``wtype`` and
``xdotool`` assume, every key-position provider is skipped in favour of the
clipboard, which carries the text itself.
"""

from __future__ import annotations

import logging
from typing import Any

from inject.base import (
    UNKNOWN_TARGET,
    InjectionError,
    InjectionTarget,
    keyboard_layout,
    layout_is_qwerty,
    sanitize,
)
from inject.providers import (
    ClipboardProvider,
    WtypeProvider,
    XdotoolProvider,
    YdotoolProvider,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InjectionError",
    "InjectionTarget",
    "available_providers",
    "describe",
    "select_provider",
    "type_text",
]

# Tried in this order. wtype first where it works at all, because pressing keys
# leaves the user's clipboard alone.
PROVIDER_ORDER = ("wtype", "ydotool", "xdotool", "clipboard")


def _build(name: str) -> Any:
    return {
        "wtype": WtypeProvider,
        "ydotool": YdotoolProvider,
        "xdotool": XdotoolProvider,
        "clipboard": ClipboardProvider,
    }[name]()


def available_providers() -> list[str]:
    """Every provider that could run here, in preference order."""
    found = []
    for name in PROVIDER_ORDER:
        try:
            if _build(name).available():
                found.append(name)
        except Exception:  # noqa: BLE001
            logger.debug("provider %s failed its availability check", name, exc_info=True)
    return found


def select_provider(prefer: str = "") -> Any | None:
    """The provider to use, or None if nothing on this machine can inject.

    ``prefer`` names one explicitly and is honoured whenever it is actually
    available — a user who has configured ydotool on a wlroots session means it.
    """
    if prefer and prefer != "auto":
        if prefer not in PROVIDER_ORDER:
            logger.warning("unknown injection provider %r; falling back to auto", prefer)
        else:
            chosen = _build(prefer)
            if chosen.available():
                return chosen
            logger.warning("preferred injection provider %r is not available here", prefer)

    qwerty = layout_is_qwerty()
    for name in PROVIDER_ORDER:
        candidate = _build(name)
        if not candidate.available():
            continue
        if not qwerty and not candidate.layout_safe:
            logger.info(
                "skipping %s: layout %r is not QWERTY, so key positions would garble the text",
                name,
                keyboard_layout(),
            )
            continue
        return candidate
    return None


def type_text(
    text: str,
    target: InjectionTarget = UNKNOWN_TARGET,
    *,
    prefer: str = "",
) -> str:
    """Deliver *text* to the focused window. Returns the provider that did it.

    Raises :class:`InjectionError` when nothing could, which is a real outcome
    on GNOME without ydotool and must reach the user rather than being logged.
    """
    cleaned = sanitize(text, target)
    if not cleaned:
        return ""

    provider = select_provider(prefer)
    if provider is None:
        raise InjectionError(
            "no way to type into other windows on this session — install wtype "
            "(wlroots), ydotool (GNOME/Wayland), or xdotool (X11)"
        )

    # Only the clipboard provider needs to know where the text is going: it
    # picks Ctrl+Shift+V for a terminal. The rest carry no key combination.
    if isinstance(provider, ClipboardProvider):
        provider.type_text(cleaned, target)
    else:
        provider.type_text(cleaned)
    return provider.name


def describe() -> dict[str, Any]:
    """What Settings and `/settings/health` show about this capability."""
    layout = keyboard_layout()
    found = available_providers()
    chosen = select_provider()
    return {
        "available": found,
        "selected": chosen.name if chosen is not None else "",
        "layout": layout,
        "layout_qwerty": layout_is_qwerty(),
        "ok": chosen is not None,
    }
