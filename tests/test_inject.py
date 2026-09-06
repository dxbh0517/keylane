"""Text injection: the safety rules, and which provider gets picked."""

from __future__ import annotations

import pytest

import inject
from inject.base import (
    InjectionTarget,
    layout_is_qwerty,
    sanitize,
)


class FakeProvider:
    def __init__(self, name: str, *, layout_safe: bool, here: bool = True) -> None:
        self.name = name
        self.layout_safe = layout_safe
        self._here = here
        self.typed: list[str] = []

    def available(self) -> bool:
        return self._here

    def type_text(self, text: str, *_args) -> None:
        self.typed.append(text)


@pytest.fixture()
def providers(monkeypatch):
    """Replace the real providers with fakes, keeping the selection order."""

    def _install(**flags: FakeProvider):
        monkeypatch.setattr(inject, "_build", lambda name: flags[name])
        monkeypatch.setattr(
            inject, "PROVIDER_ORDER", tuple(n for n in inject.PROVIDER_ORDER if n in flags)
        )
        return flags

    return _install


# ── the safety rules ─────────────────────────────────────────────────────


def test_a_trailing_newline_is_never_delivered() -> None:
    """Nothing Keylane injects may press Return on the user's behalf."""
    assert sanitize("send this\n") == "send this"
    assert sanitize("send this\r\n\r\n") == "send this"


def test_a_terminal_loses_every_newline_not_just_the_last() -> None:
    """In a shell the second line of a paste executes as surely as the first."""
    target = InjectionTarget(app_id="org.gnome.Terminal")
    assert sanitize("rm -rf tmp\nls\n", target) == "rm -rf tmp ls"


def test_an_editor_keeps_its_interior_newlines() -> None:
    target = InjectionTarget(app_id="org.gnome.TextEditor")
    assert sanitize("first\nsecond\n", target) == "first\nsecond"


@pytest.mark.parametrize(
    "app_id",
    ["org.gnome.Terminal", "Alacritty", "foot", "kitty", "org.wezfurlong.wezterm"],
)
def test_terminals_are_recognised(app_id: str) -> None:
    assert InjectionTarget(app_id=app_id).is_terminal


def test_a_browser_is_not_a_terminal() -> None:
    assert not InjectionTarget(app_id="firefox", title="Terms of service").is_terminal


# ── layout gating ────────────────────────────────────────────────────────


def test_an_unknown_layout_counts_as_qwerty() -> None:
    """Otherwise the clipboard is the only route on a machine that won't say."""
    assert layout_is_qwerty("")


def test_dvorak_is_not_qwerty() -> None:
    assert not layout_is_qwerty("dvorak")


def test_a_non_qwerty_layout_skips_key_position_providers(providers, monkeypatch) -> None:
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: False)
    monkeypatch.setattr(inject, "keyboard_layout", lambda: "dvorak")
    built = providers(
        wtype=FakeProvider("wtype", layout_safe=False),
        clipboard=FakeProvider("clipboard", layout_safe=True),
    )
    assert inject.select_provider() is built["clipboard"]


def test_qwerty_prefers_key_injection_over_the_clipboard(providers, monkeypatch) -> None:
    """Pressing keys leaves whatever the user had copied alone."""
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: True)
    built = providers(
        wtype=FakeProvider("wtype", layout_safe=False),
        clipboard=FakeProvider("clipboard", layout_safe=True),
    )
    assert inject.select_provider() is built["wtype"]


# ── selection ────────────────────────────────────────────────────────────


def test_an_unavailable_provider_is_skipped(providers, monkeypatch) -> None:
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: True)
    built = providers(
        wtype=FakeProvider("wtype", layout_safe=False, here=False),
        ydotool=FakeProvider("ydotool", layout_safe=False),
    )
    assert inject.select_provider() is built["ydotool"]


def test_an_explicit_preference_wins(providers, monkeypatch) -> None:
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: True)
    built = providers(
        wtype=FakeProvider("wtype", layout_safe=False),
        ydotool=FakeProvider("ydotool", layout_safe=False),
    )
    assert inject.select_provider("ydotool") is built["ydotool"]


def test_an_unavailable_preference_falls_back_rather_than_failing(
    providers, monkeypatch
) -> None:
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: True)
    built = providers(
        wtype=FakeProvider("wtype", layout_safe=False),
        ydotool=FakeProvider("ydotool", layout_safe=False, here=False),
    )
    assert inject.select_provider("ydotool") is built["wtype"]


def test_no_provider_at_all_raises_rather_than_silently_dropping_the_text(
    providers, monkeypatch
) -> None:
    """Losing a dictated sentence quietly is the one unacceptable outcome."""
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: True)
    providers(wtype=FakeProvider("wtype", layout_safe=False, here=False))
    with pytest.raises(inject.InjectionError):
        inject.type_text("hello")


def test_typing_reports_which_provider_delivered_it(providers, monkeypatch) -> None:
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: True)
    built = providers(wtype=FakeProvider("wtype", layout_safe=False))
    assert inject.type_text("hello\n") == "wtype"
    assert built["wtype"].typed == ["hello"]


def test_empty_text_is_not_worth_a_provider(providers, monkeypatch) -> None:
    monkeypatch.setattr(inject, "layout_is_qwerty", lambda: True)
    providers(wtype=FakeProvider("wtype", layout_safe=False, here=False))
    # No provider exists, yet this must not raise: there is nothing to type.
    assert inject.type_text("\n") == ""
