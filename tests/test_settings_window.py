"""Settings window behaviour that made its buttons feel broken.

These build real GTK widgets, so they need a display; without one the whole
module is skipped rather than failing.
"""

from __future__ import annotations

import time

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

if not Gtk.init_check():  # pragma: no cover - depends on the machine
    pytest.skip("no display for GTK", allow_module_level=True)

from ui.settings import FOCUS_SETTLE_MS, SettingsWindow  # noqa: E402


def _pump(ms: int) -> None:
    """Run the main loop for *ms*, so queued timeouts actually fire."""
    context = GLib.MainContext.default()
    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        while context.pending():
            context.iteration(False)
        time.sleep(0.005)


def _unfocused(window: SettingsWindow) -> None:
    """Report the window as not focused, which no display-less test can be.

    `is-active` is the window manager's answer, and there isn't one here, so
    the focus-out path can only be exercised by saying so directly.
    """
    real = window.get_property
    window.get_property = lambda name: False if name == "is-active" else real(name)


def test_settings_is_not_modal_over_its_parent():
    """A modal grab swallows clicks on the launcher — including the gear.

    On GNOME there is no layer shell, so Settings is built transient over the
    launcher. Modal there means the button that opens this window cannot be
    clicked while it is open.
    """
    parent = Gtk.Window()
    window = SettingsWindow(parent, independent=False)
    assert window.get_transient_for() is parent
    assert window.get_modal() is False


def test_present_centered_is_repeatable():
    """Opening Settings twice in a row works.

    A smoke test, deliberately labelled as one: it passes against the old code
    too. The bug it sits next to — `connect(..., once=True)`, which PyGObject
    rejects because `connect()` takes no keyword arguments — lives on a branch
    reached only when `present()` fails to realize the window, and nothing here
    can make that happen. That unreachability is why it survived this long.
    """
    window = SettingsWindow(independent=True)
    window.present_centered()
    window.present_centered()
    assert window.get_visible()


def test_a_close_scheduled_before_a_reopen_does_not_fire():
    """Clicking the gear while Settings is open must not close it.

    The click moves focus off the window, which schedules the dismiss, and
    also re-presents it. Without a guard the dismiss lands a moment later and
    closes the window the click just asked for.
    """
    window = SettingsWindow(independent=True)
    _unfocused(window)
    window.present_centered()
    window._had_focus = True
    window._shown_at = time.monotonic() - 10  # long past the settle grace

    window._on_active_changed()  # focus left: a close is now scheduled
    window.present_centered()  # the same click reopens it

    _pump(FOCUS_SETTLE_MS + 250)
    assert window.get_visible(), "a stale dismiss closed the reopened window"


def test_focus_out_still_closes_the_window():
    """The guard must not disable dismiss-on-click-away, only scope it."""
    window = SettingsWindow(independent=True)
    _unfocused(window)
    window.present_centered()
    window._had_focus = True
    window._shown_at = time.monotonic() - 10

    dismissed: list[bool] = []
    window.set_dismiss_callback(lambda: dismissed.append(True))

    window._on_active_changed()
    _pump(FOCUS_SETTLE_MS + 250)

    assert not window.get_visible()
    assert dismissed == [True]


def test_slow_daemon_calls_do_not_run_on_the_main_loop():
    """/settings/health probes SearXNG and every MCP server; it is not fast.

    Running it inline froze the UI for as long as the daemon took, and a
    frozen window reads as a dead button.
    """
    window = SettingsWindow(independent=True)
    started = time.monotonic()
    window._load_routes()
    assert time.monotonic() - started < 0.25, "_load_routes blocked the caller"


def test_the_download_poll_can_restart_after_it_stops():
    """A finished poll must forget its id, or nothing ever polls again.

    `_tick` returns False when no download is running, which ends the GLib
    source — but the id was left set, so `_ensure_poll` saw a live poll and
    never started another. Download progress then stopped updating for the
    rest of the window's life, and closing the window called source_remove on
    an id GLib had already reclaimed ("Source ID … was not found").
    """
    window = SettingsWindow(independent=True)
    window._models = []
    window._loading_model = False
    # The tick keeps itself alive while the window is hidden, so that a poll
    # armed before a close resumes on the next open. It has to be visible for
    # the "nothing is downloading, stop" path to be the one under test.
    window.present_centered()
    _pump(100)

    window._ensure_poll()
    assert window._poll_id is not None
    first = window._poll_id

    # One tick with nothing downloading ends the source. timeout_add_seconds
    # aligns to the second, so this waits out two of them rather than racing.
    _pump(2600)
    assert window._poll_id is None, "a finished poll must forget its id"

    # And a later download can start a fresh one.
    window._ensure_poll()
    assert window._poll_id is not None
    assert window._poll_id != first
    window._stop_poll()


def test_closing_twice_does_not_raise():
    """Nothing that runs while a window closes may raise; the close must land."""
    window = SettingsWindow(independent=True)
    window._poll_id = 999999  # an id GLib has never heard of
    assert window._on_close() is False
    assert window._poll_id is None
    assert window._on_close() is False


def test_focus_loss_closes_the_window_without_a_notification():
    """The edge is missed on XWayland when focus goes to a Wayland window.

    Dismiss-on-click-away hung entirely off notify::is-active. That fires when
    focus moves between two X11 windows, which is why it tested fine — but
    Keylane runs on XWayland under a Wayland compositor, and clicking a native
    Wayland window is the case that matters. Observed on a real machine: the
    window still open, `_NET_ACTIVE_WINDOW` pointing at something else.

    So no notification is delivered here at all; the poll has to notice.
    """
    window = SettingsWindow(independent=True)
    _unfocused(window)
    window.present_centered()
    window._had_focus = True
    window._shown_at = time.monotonic() - 10

    dismissed: list[bool] = []
    window.set_dismiss_callback(lambda: dismissed.append(True))

    # Deliberately never calling _on_active_changed: nothing tells us.
    _pump(FOCUS_SETTLE_MS + 900)

    assert not window.get_visible()
    assert dismissed == [True]


def test_an_open_dropdown_defers_the_close_rather_than_cancelling_it():
    """A dropdown is this window's own business — until it is closed."""
    window = SettingsWindow(independent=True)
    _unfocused(window)
    window.present_centered()
    window._had_focus = True
    window._shown_at = time.monotonic() - 10

    popover = Gtk.Popover()
    visible = {"open": True}
    popover.get_visible = lambda: visible["open"]  # type: ignore[method-assign]
    window._dropdown_popovers.append(popover)

    _pump(FOCUS_SETTLE_MS + 700)
    assert window.get_visible(), "an open dropdown should hold the window open"

    visible["open"] = False
    _pump(FOCUS_SETTLE_MS + 900)
    assert not window.get_visible(), "closing the dropdown should let it dismiss"
