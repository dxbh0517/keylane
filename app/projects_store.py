"""Read and write the project list shown in the popup's picker.

Projects live in ``config/projects.toml``. Each entry must resolve inside the
configured allowed roots — the picker cannot be used to smuggle a path past the
sandbox.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import ROOT, get_config, reload_config
from app.permissions import PermissionError_, resolve_under_roots

logger = logging.getLogger(__name__)

PROJECTS_TOML = ROOT / "config" / "projects.toml"


class ProjectEntryUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1)


class ProjectsUpdate(BaseModel):
    projects: list[ProjectEntryUpdate] = Field(default_factory=list)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_projects(update: ProjectsUpdate) -> dict:
    """Replace the project list, validating every path against the sandbox."""
    config = get_config()
    roots = config.security.allowed_project_roots
    seen: set[str] = set()
    entries: list[tuple[str, str]] = []
    rejected: list[dict[str, str]] = []

    for entry in update.projects:
        name = entry.name.strip()
        raw = entry.path.strip()
        if not name or not raw:
            continue
        if name.lower() in seen:
            rejected.append({"name": name, "reason": "duplicate name"})
            continue
        try:
            # Keep the user's "~" in the file, but prove it resolves in-sandbox.
            resolve_under_roots(raw, roots)
        except PermissionError_ as exc:
            rejected.append({"name": name, "reason": str(exc)})
            continue
        if not Path(raw).expanduser().is_dir():
            rejected.append({"name": name, "reason": "not a directory"})
            continue
        seen.add(name.lower())
        entries.append((name, raw))

    lines = [
        "# Projects offered in the Keylane popup's picker.",
        "# Every path must sit inside security.allowed_project_roots.",
        "",
    ]
    for name, path in entries:
        lines += ["[[projects]]", f'name = "{_escape(name)}"', f'path = "{_escape(path)}"', ""]

    PROJECTS_TOML.parent.mkdir(parents=True, exist_ok=True)
    PROJECTS_TOML.write_text("\n".join(lines), encoding="utf-8")
    reload_config()

    return {
        "projects": [{"name": n, "path": p} for n, p in entries],
        "rejected": rejected,
        "allowed_roots": roots,
    }
