"""Keeping Keylane current from its own GitHub releases.

The shape of the install decides everything here, and there are two of them.
A developer runs from a git checkout, where updating is a fetch and a
fast-forward. Everyone else got their copy from ``scripts/install.sh``, which
*rsyncs* the tree and excludes ``.git`` — so an installed Keylane is not a
checkout and cannot pull. That one takes the release tarball, verifies it,
unpacks it beside the running copy, and moves a symlink.

Nothing installs itself. A check writes a note to the inbox; applying an update
goes through the permission gate, because it replaces the code that is running
and restarts the daemon under the user.
"""

from updater.apply import (
    InstallShape,
    UpdateError,
    apply_update,
    detect_shape,
    rollback,
)
from updater.github import Release, check_for_update, latest
from updater.version import VERSION, version_info

__all__ = [
    "VERSION",
    "InstallShape",
    "Release",
    "UpdateError",
    "apply_update",
    "check_for_update",
    "detect_shape",
    "latest",
    "rollback",
    "version_info",
]
