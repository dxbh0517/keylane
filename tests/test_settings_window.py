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
