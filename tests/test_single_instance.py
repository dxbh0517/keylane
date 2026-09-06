"""One resident UI, and what happens when a second one starts.

A duplicate is not an error the user can see. GTK hands the activation to
whoever owns the bus name and the new process exits 0, which systemd reads as
a clean shutdown and reports as `inactive (dead)` with nothing in the journal
— so an enabled service is simply not running and nothing says why. Worse, the
activation reaches the *other* instance's `do_activate`, so starting a service
pops the launcher onto someone's screen.
"""

from __future__ import annotations

import os

# See tests/test_spotlight.py: importing ui.main re-execs without this.
os.environ.setdefault("KEYLANE_LAYER_SHELL_PRIMED", "1")

from ui.main import APP_ID, EXIT_ALREADY_RUNNING, describe_pid  # noqa: E402


def test_the_conflict_exit_status_is_not_success() -> None:
    """Exit 0 is what made the failure silent; anything else is visible."""
    assert EXIT_ALREADY_RUNNING != 0


def test_the_unit_declines_to_restart_on_that_status() -> None:
    """Restarting cannot help while the other process holds the name.

    Without this systemd retries five times and gives up; with it the unit
    stops at once and reports `failed`, which is the state a person can see.
    """
    from pathlib import Path

    unit = Path(__file__).resolve().parents[1] / "systemd" / "keylane-ui.service"
    text = unit.read_text(encoding="utf-8")
    assert f"RestartPreventExitStatus={EXIT_ALREADY_RUNNING}" in text
    # It must still restart on a genuine crash.
    assert "Restart=on-failure" in text


def test_the_bus_name_matches_the_one_the_remote_commands_use() -> None:
    """`gapplication action <id>` and the app must name the same thing.

    They were two copies of the same string, which is how a rename breaks
    every hotkey and nothing else.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "ui" / "main.py").read_text()
    assert source.count('"app.keylane.Spotlight"') == 1, "the bus name is written twice"
    assert APP_ID == "app.keylane.Spotlight"


def test_a_running_process_can_be_named() -> None:
    """The journal line is only actionable if it can say which pid to stop."""
    described = describe_pid(os.getpid())
    assert described
    assert "python" in described.lower() or "pytest" in described.lower()


def test_a_dead_pid_describes_as_nothing_rather_than_raising() -> None:
    """The owner can exit between the bus answering and the lookup."""
    assert describe_pid(999_999_999) == ""


def test_describe_pid_joins_the_whole_command_line() -> None:
    """/proc separates arguments with NULs; a raw read shows only argv[0]."""
    described = describe_pid(os.getpid())
    assert "\0" not in described
