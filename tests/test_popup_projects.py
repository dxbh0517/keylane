"""The project chip in the popup.

The Spotlight preset renders no meta row, so before this chip existed there was
no way to choose a project in the default theme — and a worker that requires
one refused with an error nothing in the popup could satisfy.

These exercise the selection logic without building a GTK widget tree, which
needs a display.
"""

from __future__ import annotations

import pytest


class FakeChip:
    """Stands in for the parts of the popup the project logic touches."""

    def __init__(self, projects):
        self._projects = projects
        self._project_path = None
        self.label = "No project"
        self.visible = False

    # Copied in behaviour from PopupWindow._sync_project_chip.
    def sync(self):
        self.visible = bool(self._projects)
        name = next(
            (p["name"] for p in self._projects if p["path"] == self._project_path),
            None,
        )
        self.label = name or "No project"

    def apply_projects(self, projects):
        self._projects = projects
        if self._project_path and not any(
            p["path"] == self._project_path for p in projects
        ):
            self._project_path = None
        self.sync()


SANDBOX = {"name": "Sandbox", "path": "/home/u/code/sandbox"}
OTHER = {"name": "Other", "path": "/home/u/code/other"}


def test_chip_is_hidden_when_no_projects_are_configured():
    chip = FakeChip([])
    chip.sync()
    assert chip.visible is False, "nothing to pick, so nothing to show"


def test_chip_appears_once_a_project_exists():
    chip = FakeChip([SANDBOX])
    chip.sync()
    assert chip.visible is True
    assert chip.label == "No project", "visible, but nothing chosen yet"


def test_choosing_a_project_names_it_on_the_chip():
    chip = FakeChip([SANDBOX, OTHER])
    chip._project_path = OTHER["path"]
    chip.sync()
    assert chip.label == "Other"


def test_a_removed_project_does_not_stay_selected():
    """Deleting the selected project in the panel must not leave a stale path.

    Otherwise the popup keeps sending a directory that no longer passes the
    allowed-roots check, and the failure is baffling.
    """
    chip = FakeChip([SANDBOX, OTHER])
    chip._project_path = OTHER["path"]
    chip.sync()
    chip.apply_projects([SANDBOX])
    assert chip._project_path is None
    assert chip.label == "No project"


def test_the_required_project_error_says_how_to_fix_it():
    from app.permissions import PermissionError_, validate_working_directory

    with pytest.raises(PermissionError_) as excinfo:
        validate_working_directory(None, required=True)
    message = str(excinfo.value)
    assert "project chip" in message, "the error must name where to fix it"
    assert "control panel" in message
