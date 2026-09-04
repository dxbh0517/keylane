"""The one place Keylane's version is written down.

There was no version anywhere: no VERSION file, no pyproject, nothing in
/health. So there was no way to tell whether an update was needed, no way to
report a bug against a build, and nothing for an updater to compare.

Bump this and tag the commit ``vX.Y.Z``; the updater matches releases by that
tag. Everything else reads it from here.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

VERSION = "0.5.1"

# Semantic-ish: major.minor.patch, compared numerically field by field. A tag
# may carry a leading "v" and a suffix like "-rc1"; both are handled below.


def parse(version: str) -> tuple[int, ...]:
    """A version as a comparable tuple. Unparseable parts sort as zero."""
    cleaned = version.strip().lstrip("vV").split("-", 1)[0].split("+", 1)[0]
    parts: list[int] = []
    for chunk in cleaned.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def is_newer(candidate: str, current: str = VERSION) -> bool:
    return parse(candidate) > parse(current)


@lru_cache(maxsize=1)
def git_revision() -> str:
    """The commit this tree is at, when it is a checkout. Empty otherwise.

    Useful in a bug report and the only way to identify a build between tags.
    """
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def version_info() -> dict[str, Any]:
    """What /health and Settings report."""
    from updater.apply import detect_shape

    revision = git_revision()
    return {
        "version": VERSION,
        "revision": revision,
        "install": detect_shape().name,
        # A checkout is a build someone is editing, so the tag alone does not
        # identify it.
        "display": f"{VERSION} ({revision})" if revision else VERSION,
    }
