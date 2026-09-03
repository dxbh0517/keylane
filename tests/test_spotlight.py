"""What the Spotlight window shows while the model is still thinking.

The agent emits `replace_answer` with empty text twice per turn: once at the
start of every ReAct iteration, and again the moment a stream turns out to be
a tool call rather than an answer. Both mean "there is nothing to show yet",
and both used to promote the orb into an empty answer card.
"""

from __future__ import annotations

import os

# ui.main re-execs the process at import time to get libgtk4-layer-shell into
# LD_PRELOAD, which has to happen before GTK is loaded and so cannot wait for
# __main__. Importing it from a test therefore restarts pytest — silently, with
# no output and a clean exit. This is the flag that says the priming is done.
os.environ.setdefault("KEYLANE_LAYER_SHELL_PRIMED", "1")

from ui.main import SpotlightWindow  # noqa: E402


class _Revealer:
    def __init__(self) -> None:
        self.revealed: bool | None = None

    def set_reveal_child(self, value: bool) -> None:
        self.revealed = value


class _Orb:
    def __init__(self) -> None:
        self.state: str | None = None

    def set_state(self, state: str) -> None:
        self.state = state


def _window(mode: str) -> SpotlightWindow:
    """A window with only the parts `_replace_corner_answer` touches.

    Building a real one needs a GTK application and a display; the decision
    under test is about which widgets get poked, so the widgets are stubs.
    """
    win = SpotlightWindow.__new__(SpotlightWindow)
    win._mode = mode
    win._canvas_full_answer = "stale"
    win._canvas_summary = "stale"
    win._streaming_answer = True
    win._answer_revealer = _Revealer()
    win._corner_orb = _Orb()
    win._thinking_orb = _Orb()
    win._cleared = False

    def _clear_canvas() -> None:
        win._cleared = True

    win._clear_canvas = _clear_canvas  # type: ignore[method-assign]
    win._promoted = False

    def _promote() -> None:
        win._promoted = True
        win._mode = "corner"

    win._promote_to_corner_panel = _promote  # type: ignore[method-assign]
    win._set_corner_answer_text = lambda *a, **k: None  # type: ignore[method-assign]
    win._resize_corner_panel = lambda: None  # type: ignore[method-assign]
    return win


def test_an_empty_replacement_keeps_the_orb_while_thinking():
    """A new iteration must not turn the orb into a blank answer card."""
    win = _window("thinking")

    win._replace_corner_answer("")

    assert win._promoted is False, "the orb should still be the orb"
    assert win._mode == "thinking"
    assert win._corner_orb.state == "thinking"
    assert win._thinking_orb.state == "thinking"
    assert win._answer_revealer.revealed is False
    assert win._streaming_answer is False
    assert win._cleared is True


def test_whitespace_is_not_an_answer():
    win = _window("thinking")
    win._replace_corner_answer("   \n  ")
    assert win._promoted is False
    assert win._corner_orb.state == "thinking"


def test_a_suppressed_tool_call_returns_the_answer_panel_to_thinking():
    """Mid-stream, a tool call is pulled back — the card must go with it.

    By then the panel is already showing, because real tokens arrived before
    the markup gave the tool call away.
    """
    win = _window("corner")

    win._replace_corner_answer("")

    assert win._answer_revealer.revealed is False
    assert win._corner_orb.state == "thinking"
    assert win._canvas_full_answer == ""


def test_real_text_still_opens_the_answer_panel():
    """The fix must not stop an actual answer from being shown."""
    win = _window("thinking")

    win._replace_corner_answer("Paris is the capital of France.")

    assert win._promoted is True
    assert win._corner_orb.state == "done"
    assert win._answer_revealer.revealed is True
    assert win._streaming_answer is True
