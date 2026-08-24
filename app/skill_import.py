"""Import skills from a GitHub repository.

A skills repo is rarely one file and is almost never *only* skills — cloning
the whole thing would drag in code, licences and CI config. So this reads the
repository tree through the GitHub API, works out which files are actually
skills, and installs only the ones the user picks.

Recognised layouts:

    skills/<name>/SKILL.md          Claude / Cursor plugin layout
    <name>/SKILL.md                 a repo that is one skill per folder
    skills/<name>.md                a flat folder of markdown skills
    .cursor/rules/<name>.md         Cursor rules used as skills

A candidate must carry front matter with a ``name`` or ``description``, or be
called ``SKILL.md`` — otherwise every README in the repo would look like a
skill.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from app.skills import SkillError, get_skill_registry, load_skill_file

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
USER_AGENT = "Keylane/1.0 (local assistant)"

# Directories that never contain skills, skipped before anything is fetched.
SKIP_DIRS = {
    ".git", ".github", "node_modules", "dist", "build", "__pycache__",
    "tests", "test", "vendor", ".venv", "venv", "site-packages",
}
# Markdown that is documentation about the repo, not a skill.
SKIP_NAMES = {
    "readme.md", "changelog.md", "contributing.md", "license.md",
    "code_of_conduct.md", "security.md", "index.md",
}

MAX_TREE_ENTRIES = 8000
MAX_CANDIDATES = 200
MAX_FILE_BYTES = 512_000

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillImportError(RuntimeError):
    """Raised when a repository cannot be read or has no skills."""


@dataclass
class SkillCandidate:
    path: str
    name: str
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "size": self.size,
        }


def parse_repo(reference: str) -> tuple[str, str, str | None]:
    """Accept a URL, ``owner/repo``, or ``owner/repo@branch``.

    Returns ``(owner, repo, ref)`` with ``ref`` ``None`` for the default branch.
    """
    text = (reference or "").strip()
    if not text:
        raise SkillImportError("Give a GitHub repository, e.g. owner/repo.")

    ref: str | None = None
    # "owner/repo@branch" — but not the "@" in an ssh URL like git@github.com.
    if "@" in text and not text.startswith(("http", "git@", "ssh://")):
        text, _, ref = text.partition("@")

    if text.startswith(("http://", "https://", "git@", "ssh://")):
        if text.startswith("git@"):
            text = text.split(":", 1)[-1]
            parsed_path = text
        else:
            parsed = urlparse(text)
            if parsed.netloc and "github.com" not in parsed.netloc:
                raise SkillImportError("Only github.com repositories are supported.")
            parsed_path = parsed.path
        parts = [p for p in parsed_path.strip("/").split("/") if p]
        # .../owner/repo/tree/<ref>/...
        if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
            ref = ref or parts[3]
            parts = parts[:2]
    else:
        parts = [p for p in text.strip("/").split("/") if p]

    if len(parts) < 2:
        raise SkillImportError(f"Could not read an owner/repo from {reference!r}.")
    owner, repo = parts[0], parts[1]
    return owner, repo.removesuffix(".git"), ref


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_token() -> str | None:
    """Find a token: the environment first, then the gh CLI if it is signed in.

    Unauthenticated GitHub allows 60 requests an hour, which one repository
    scan can exhaust. Borrowing the token the user already has avoids making
    them configure a second one.
    """
    import os
    import shutil
    import subprocess

    for name in ("KEYLANE_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()

    gh = shutil.which("gh")
    if gh:
        try:
            result = subprocess.run(
                [gh, "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            token = result.stdout.strip()
            if result.returncode == 0 and token:
                return token
        except Exception:  # noqa: BLE001
            pass
    return None


def _looks_like_skill(path: str) -> bool:
    lowered = path.lower()
    if not lowered.endswith(".md"):
        return False
    parts = lowered.split("/")
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return False
    filename = parts[-1]
    if filename == "skill.md":
        return True
    if filename in SKIP_NAMES:
        return False
    # A markdown file directly under a skills/rules directory.
    return any(part in {"skills", "skill", "rules", "agents"} for part in parts[:-1])


def _front_matter_fields(block: str) -> dict[str, str]:
    """Parse the front matter we care about, including YAML block scalars.

    Skill repos commonly write ``description: >`` followed by an indented
    paragraph; reading only the first line yields a literal ">".
    """
    fields: dict[str, str] = {}
    lines = block.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.startswith("#") or ":" not in line:
            continue
        if line[:1].isspace():
            continue  # a continuation we already consumed
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if value in {">", "|", ">-", "|-", ">+", "|+"}:
            folded: list[str] = []
            while index < len(lines) and (
                not lines[index].strip() or lines[index][:1].isspace()
            ):
                folded.append(lines[index].strip())
                index += 1
            value = " ".join(part for part in folded if part)
        fields[key] = value.strip().strip("\"'")
    return fields


def _metadata(text: str, fallback: str) -> tuple[str, str, list[str]]:
    match = FRONT_MATTER.match(text)
    name, description, triggers = "", "", []
    if match:
        fields = _front_matter_fields(match.group(1))
        name = fields.get("name", "")
        description = fields.get("description", "")
        raw_triggers = fields.get("triggers") or fields.get("keywords") or ""
        triggers = [v.strip() for v in re.split(r"[,;]", raw_triggers) if v.strip()]
    if not name:
        # skills/deploy/SKILL.md -> "deploy"; skills/deploy.md -> "deploy"
        parts = fallback.split("/")
        stem = parts[-1].removesuffix(".md")
        name = parts[-2] if stem.lower() == "skill" and len(parts) > 1 else stem
    if not description:
        body = FRONT_MATTER.sub("", text).strip()
        first = next((ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")), "")
        description = first[:200]
    return name, description, triggers


async def discover(reference: str) -> dict[str, Any]:
    """List the skills a repository contains, without installing anything."""
    owner, repo, ref = parse_repo(reference)
    token = _github_token()

    async with httpx.AsyncClient(timeout=30.0, headers=_headers(token)) as client:
        if ref is None:
            info = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}")
            if info.status_code == 404:
                raise SkillImportError(
                    f"{owner}/{repo} was not found. Private repositories need "
                    "KEYLANE_GITHUB_TOKEN set in the service environment."
                )
            if info.status_code == 403:
                raise SkillImportError(
                    "GitHub rate-limited this request. Set KEYLANE_GITHUB_TOKEN "
                    "to raise the limit."
                )
            info.raise_for_status()
            ref = info.json().get("default_branch") or "main"

        tree = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{ref}",
            params={"recursive": "1"},
        )
        if tree.status_code == 404:
            raise SkillImportError(f"Branch {ref!r} was not found in {owner}/{repo}.")
        tree.raise_for_status()
        payload = tree.json()

        entries = payload.get("tree") or []
        if payload.get("truncated"):
            logger.warning("GitHub truncated the tree for %s/%s", owner, repo)
        blobs = [
            entry
            for entry in entries[:MAX_TREE_ENTRIES]
            if entry.get("type") == "blob" and _looks_like_skill(entry.get("path", ""))
        ]
        blobs = [b for b in blobs if int(b.get("size") or 0) <= MAX_FILE_BYTES]
        blobs = blobs[:MAX_CANDIDATES]

        if not blobs:
            raise SkillImportError(
                f"No skill files found in {owner}/{repo}. Keylane looks for "
                "SKILL.md files, or markdown under a skills/ or rules/ folder."
            )

        candidates: list[SkillCandidate] = []
        for blob in blobs:
            path = blob["path"]
            text = await _fetch_text(client, owner, repo, ref, path)
            if text is None:
                continue
            # A skill must declare itself, or be named SKILL.md.
            if not FRONT_MATTER.match(text) and not path.lower().endswith("skill.md"):
                continue
            name, description, triggers = _metadata(text, path)
            candidates.append(
                SkillCandidate(
                    path=path,
                    name=name,
                    description=description,
                    triggers=triggers,
                    size=int(blob.get("size") or 0),
                )
            )

    if not candidates:
        raise SkillImportError(
            f"Found markdown in {owner}/{repo} but none of it declares a skill "
            "(front matter with a name or description)."
        )

    return {
        "repo": f"{owner}/{repo}",
        "ref": ref,
        "count": len(candidates),
        "skills": [c.to_dict() for c in candidates],
    }


async def _fetch_text(
    client: httpx.AsyncClient, owner: str, repo: str, ref: str, path: str
) -> str | None:
    response = await client.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}", params={"ref": ref}
    )
    if response.status_code != 200:
        logger.info("Could not read %s from %s/%s", path, owner, repo)
        return None
    data = response.json()
    if data.get("encoding") == "base64":
        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None
    return data.get("content")


async def install(reference: str, paths: list[str]) -> dict[str, Any]:
    """Install the chosen skill files into the local skills directory."""
    if not paths:
        raise SkillImportError("Select at least one skill to install.")

    owner, repo, ref = parse_repo(reference)
    token = _github_token()
    registry = get_skill_registry()

    installed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=30.0, headers=_headers(token)) as client:
        if ref is None:
            info = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}")
            info.raise_for_status()
            ref = info.json().get("default_branch") or "main"

        for path in paths[:MAX_CANDIDATES]:
            text = await _fetch_text(client, owner, repo, ref, path)
            if text is None:
                failed.append({"path": path, "reason": "could not be downloaded"})
                continue

            name, description, triggers = _metadata(text, path)
            body = FRONT_MATTER.sub("", text).strip()
            try:
                skill = registry.save(
                    {
                        "name": name,
                        "description": description,
                        "triggers": triggers,
                        # Imported skills start switched off: a repo should not
                        # silently change how the assistant behaves.
                        "enabled": False,
                        "content": body,
                    }
                )
            except SkillError as exc:
                failed.append({"path": path, "reason": str(exc)})
                continue
            installed.append({"name": skill.name, "path": path})

    return {
        "repo": f"{owner}/{repo}",
        "ref": ref,
        "installed": installed,
        "failed": failed,
        "note": (
            "Imported skills are disabled until you switch them on, so a "
            "repository cannot change the assistant's behaviour on import."
        ),
    }
