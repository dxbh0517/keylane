"""The skill seam: discovery, invocation policy, and loading.

A skill is a reusable set of task-specific instructions kept out of the prompt
until it is needed. The catalog carries names and one-line descriptions; the
body arrives only when something asks for it, which is what keeps a dozen
skills from costing a dozen skills' worth of context on every turn.

Keylane already documented frontmatter — `enabled`, `triggers` — that nothing
read. This implements it, as DSH's normalized pair of independent controls:
whether the *model* may load a skill, and whether the *user* may invoke it by
name. All four combinations mean something.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from daemon.paths import ROOT, SKILLS_DIR

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DESCRIPTION_MAX_CHARS = 500

# Lower rank wins a duplicate name. A skill checked into the project beats the
# user's own, which beats whatever ships with Keylane.
RANK_PROJECT = 100
RANK_USER = 400
RANK_BUNDLED = 600


@dataclass(frozen=True)
class InvocationPolicy:
    """The two independent controls, normalized to positive booleans."""

    model_invocable: bool = True
    user_invocable: bool = True


@dataclass(frozen=True)
class SkillSummary:
    """What the catalog shows: never a body, never a path."""

    name: str
    description: str
    when_to_use: str = ""
    invocation: InvocationPolicy = field(default_factory=InvocationPolicy)
    source: str = "user"
    rank: int = RANK_USER


@dataclass(frozen=True)
class Skill(SkillSummary):
    """A loaded skill: its summary plus the instruction body."""

    content: str = ""
    path: Path | None = None

    @property
    def resource_base(self) -> Path | None:
        """The directory relative paths inside the body resolve against."""
        if self.path is None:
            return None
        return self.path.parent


def summarize(skill: SkillSummary) -> SkillSummary:
    """Project a loaded skill down to what a catalog may carry."""
    return SkillSummary(
        name=skill.name,
        description=skill.description,
        when_to_use=skill.when_to_use,
        invocation=skill.invocation,
        source=skill.source,
        rank=skill.rank,
    )


class SkillProvider(Protocol):
    name: str

    def list(self) -> list[SkillSummary]: ...

    def get(self, name: str) -> Skill | None: ...


# ── frontmatter ──────────────────────────────────────────────────────────


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    raw = text[3:end]
    body = text[end + 4 :].lstrip("\n")

    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, body


def _flag(meta: dict[str, str], key: str, default: bool) -> bool:
    raw = meta.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "no", "0", "off"}


def parse_policy(meta: dict[str, str]) -> InvocationPolicy:
    """Read the invocation controls, with omitted fields meaning permitted.

    `enabled: false` is Keylane's own spelling for "off entirely" and is
    honoured as both controls off — it was documented in the shipped example
    skill and never implemented.
    """
    if not _flag(meta, "enabled", True):
        return InvocationPolicy(model_invocable=False, user_invocable=False)
    return InvocationPolicy(
        model_invocable=not _flag(meta, "disable-model-invocation", False),
        user_invocable=_flag(meta, "user-invocable", True),
    )


def _clip(text: str, limit: int = DESCRIPTION_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ── the local provider ───────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillRoot:
    path: Path
    source: str
    rank: int


def default_roots() -> list[SkillRoot]:
    return [
        SkillRoot(ROOT / ".keylane" / "skills", "project", RANK_PROJECT),
        SkillRoot(SKILLS_DIR, "user", RANK_USER),
        SkillRoot(ROOT / "skills", "bundled", RANK_BUNDLED),
    ]


class LocalSkillProvider:
    """Skills on disk, as a `<name>/SKILL.md` bundle or a flat `<name>.md`."""

    name = "local"

    def __init__(self, roots: list[SkillRoot] | None = None) -> None:
        self._roots = roots if roots is not None else default_roots()

    def _candidate_files(self) -> list[tuple[str, Path, SkillRoot]]:
        found: list[tuple[str, Path, SkillRoot]] = []
        for root in self._roots:
            if not root.path.is_dir():
                continue
            for entry in sorted(root.path.iterdir()):
                if entry.is_dir():
                    bundle = entry / "SKILL.md"
                    if bundle.is_file():
                        found.append((entry.name, bundle, root))
                elif entry.suffix == ".md" and entry.name != "SKILL.md":
                    found.append((entry.stem, entry, root))
        return found

    def _read(self, name: str, path: Path, root: SkillRoot) -> Skill | None:
        if not NAME_PATTERN.match(name):
            logger.warning("skipping skill %r: names must be kebab-case", name)
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read skill %s: %s", path, exc)
            return None
        meta, body = _split_frontmatter(text)
        return Skill(
            name=name,
            description=_clip(meta.get("description", "")),
            when_to_use=_clip(meta.get("when-to-use", "")),
            invocation=parse_policy(meta),
            source=root.source,
            rank=root.rank,
            content=body.strip(),
            path=path,
        )

    def _all(self) -> dict[str, Skill]:
        """Winners by name: lower rank wins, then first seen."""
        winners: dict[str, Skill] = {}
        for name, path, root in self._candidate_files():
            skill = self._read(name, path, root)
            if skill is None:
                continue
            current = winners.get(name)
            if current is None or skill.rank < current.rank:
                winners[name] = skill
        return winners

    def list(self) -> list[SkillSummary]:
        """Summaries only — a catalog that carries bodies is not a catalog."""
        return sorted(
            (summarize(skill) for skill in self._all().values()),
            key=lambda s: s.name,
        )

    def get(self, name: str) -> Skill | None:
        return self._all().get(name)


# ── the registry ─────────────────────────────────────────────────────────


class SkillRegistry:
    """Merges providers and applies invocation policy at the boundary."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._providers: list[SkillProvider] = []

    def register_provider(self, provider: SkillProvider) -> Any:
        with self._lock:
            self._providers.append(provider)
        return lambda: self._providers.remove(provider)

    def list(self) -> list[SkillSummary]:
        """Every skill, invocation-neutral. Consumers apply their own policy."""
        with self._lock:
            providers = list(self._providers)
        winners: dict[str, SkillSummary] = {}
        for provider in providers:
            try:
                found = provider.list()
            except Exception as exc:  # noqa: BLE001
                logger.warning("skill provider %s failed to list: %s", provider.name, exc)
                continue
            for summary in found:
                current = winners.get(summary.name)
                if current is None or summary.rank < current.rank:
                    winners[summary.name] = summary
        return sorted(winners.values(), key=lambda s: s.name)

    def for_model(self) -> list[SkillSummary]:
        return [s for s in self.list() if s.invocation.model_invocable]

    def for_user(self) -> list[SkillSummary]:
        return [s for s in self.list() if s.invocation.user_invocable]

    def get(self, name: str) -> Skill | None:
        with self._lock:
            providers = list(self._providers)
        best: Skill | None = None
        for provider in providers:
            try:
                skill = provider.get(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("skill provider %s failed to load %s: %s", provider.name, name, exc)
                continue
            if skill is not None and (best is None or skill.rank < best.rank):
                best = skill
        return best


def render_skill(skill: Skill) -> str:
    """The envelope a loaded skill arrives in.

    The resource base is part of it: without it a skill cannot reference the
    script or reference file sitting next to it, which is most of what makes a
    skill more than a paragraph of prose.
    """
    base = skill.resource_base
    if base is not None:
        resources = (
            f"Base directory for this skill: {base}\n"
            "Resolve relative paths mentioned by this skill against that directory. "
            "Load referenced resources only as needed."
        )
    else:
        resources = "This skill references no local resources."

    return (
        f'<skill_content name="{skill.name}">\n'
        f"<skill_resources>\n{resources}\n</skill_resources>\n\n"
        f"<skill_instructions>\n{skill.content}\n</skill_instructions>\n"
        f"</skill_content>"
    )


_INVOCATION = re.compile(r"(?:^|\s)/([a-z0-9]+(?:-[a-z0-9]+)*)(?=\s|$)")


def find_invocations(message: str, available: list[SkillSummary]) -> list[str]:
    """Skill names the user invoked with `/name` in their own message.

    This is the sole entry point for a skill the model may not load itself, so
    it checks `user_invocable` rather than the model's control. An unknown or
    user-disabled `/word` stays ordinary prose — the user may simply have been
    talking about a path.
    """
    usable = {s.name for s in available if s.invocation.user_invocable}
    seen: list[str] = []
    for match in _INVOCATION.finditer(message):
        name = match.group(1)
        if name in usable and name not in seen:
            seen.append(name)
    return seen
