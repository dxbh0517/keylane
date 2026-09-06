"""The text-injection interface, and the rules every provider obeys.

Injection is the one capability Keylane has that writes into a window it does
not own. That makes two things non-negotiable, and both live here rather than
in the providers, so a new provider cannot forget them:

**Never synthesise Return.** A newline delivered to a shell is not text, it is
command execution. Trailing newlines are stripped everywhere; inside a terminal
*every* newline is folded to a space, because the second line of a two-line
paste executes just as surely as the first.

**Keycodes lie on a non-QWERTY layout.** ``wtype`` and ``xdotool`` ultimately
press *positions*, and a keymap the user chose (Dvorak, Colemak, AZERTY)
scrambles the result. Providers therefore declare whether they are
layout-safe, and :func:`select_provider` refuses the unsafe ones when the
layout is not one it recognises as QWERTY.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

# Layouts whose letter positions match the ones wtype/xdotool assume. Anything
# else goes to a provider that carries text rather than key positions.
QWERTY_LAYOUTS = frozenset({"us", "gb", "uk", "en", "ca", "au", "nz", "in", "ie"})

# Window classes that mean "this is a shell". Matched case-insensitively as a
# substring, so `org.gnome.Terminal` and `foot` both land.
TERMINAL_HINTS = (
    "term", "konsole", "alacritty", "kitty", "foot", "wezterm",
    "tilix", "ptyxis", "xfce4-terminal", "urxvt", "st-256color",
)

_COMMAND_TIMEOUT = 10


@dataclass(frozen=True)
class InjectionTarget:
    """Where the text is going, as much as the display server will admit."""

    app_id: str = ""
    title: str = ""

    @property
    def is_terminal(self) -> bool:
        """Whether newlines here would execute something.

        Matched against the window *class* only. The title is content the user
        is looking at, not an identity: a browser on a page called "Terms of
        service" contains "term", and folding the newlines out of someone's
        prose because of that would be a strange bug to track down.
        """
        app = self.app_id.lower()
        return any(hint in app for hint in TERMINAL_HINTS)


UNKNOWN_TARGET = InjectionTarget()


class InjectionProvider(Protocol):
    """One way to get a string into the focused window."""

    name: str
    # False when the provider presses key positions rather than carrying text.
    layout_safe: bool

    def available(self) -> bool:
        """Whether this provider can run here, right now."""
        ...

    def type_text(self, text: str) -> None:
        """Deliver *text*. Raise :class:`InjectionError` if it did not land."""
        ...


class InjectionError(RuntimeError):
    """Injection was attempted and failed. The caller should say so."""


# ── the safety rules, applied for every provider ─────────────────────────

_TRAILING_NEWLINES = re.compile(r"[\r\n]+$")
_ANY_NEWLINE = re.compile(r"[\r\n]+")


def sanitize(text: str, target: InjectionTarget = UNKNOWN_TARGET) -> str:
    """Make *text* safe to deliver to *target*.

    Trailing newlines always go: nothing Keylane injects should press Return
    on the user's behalf. In a terminal the interior ones go too — a newline
    mid-string runs everything before it.
    """
    cleaned = _TRAILING_NEWLINES.sub("", text)
    if target.is_terminal and _ANY_NEWLINE.search(cleaned):
        logger.info("folding newlines to spaces: target %r is a terminal", target.app_id)
        cleaned = _ANY_NEWLINE.sub(" ", cleaned)
    return cleaned


def keyboard_layout() -> str:
    """The active keyboard layout's short name, lowercased, or "" if unknown."""
    for env in ("XKB_DEFAULT_LAYOUT", "KEYLANE_KEYBOARD_LAYOUT"):
        value = os.environ.get(env, "").strip()
        if value:
            return value.split(",")[0].strip().lower()
    if shutil.which("setxkbmap"):
        try:
            query = subprocess.run(
                ["setxkbmap", "-query"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        for line in query.stdout.splitlines():
            if line.startswith("layout:"):
                return line.split(":", 1)[1].strip().split(",")[0].lower()
    return ""


def layout_is_qwerty(layout: str | None = None) -> bool:
    """Whether key-position injection will produce the letters we asked for.

    An unknown layout counts as QWERTY: on a machine where nothing answers,
    refusing every position-based provider would leave only the clipboard
    path, which clobbers whatever the user had copied.
    """
    name = keyboard_layout() if layout is None else layout.strip().lower()
    return not name or name in QWERTY_LAYOUTS


def run_with_text(argv: list[str], text: str, *, name: str) -> None:
    """Run *argv*, feeding *text* on stdin. The shared provider body.

    Text goes on stdin rather than in argv on purpose: a transcript beginning
    with "-" is otherwise parsed as a flag, and a long one can exceed the
    argument limit.
    """
    try:
        done = subprocess.run(
            argv,
            input=text,
            text=True,
            capture_output=True,
            timeout=_COMMAND_TIMEOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        raise InjectionError(f"{name} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise InjectionError(f"{name} did not finish within {_COMMAND_TIMEOUT}s") from exc
    except OSError as exc:
        raise InjectionError(f"{name} could not run: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit status {done.returncode}"
        raise InjectionError(f"{name} failed: {reason}")
