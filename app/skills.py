"""Skills — instruction packs that extend the assistant without new code.

A skill is a markdown file with TOML-ish front matter describing when it should
apply. Matching skills are appended to the assistant's system prompt for that
request, so a user can teach Keylane house rules ("always deploy with make
release") by dropping a file into ``skills/``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import ROOT

logger = logging.getLogger(__name__)

SKILLS_DIR = ROOT / "skills"

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class Skill(BaseModel):
    name: str
    description: str = ""
    triggers: list[str] = Field(default_factory=list)
    """Lower-case substrings; an empty list means "always apply"."""

    always: bool = False
    content: str = ""
    source: str = "user"
    path: str = ""
    enabled: bool = True

    def matches(self, message: str) -> bool:
        if not self.enabled:
            return False
        if self.always or not self.triggers:
            return self.always
        lowered = message.lower()
        return any(trigger.lower() in lowered for trigger in self.triggers if trigger)

    def prompt_block(self) -> str:
        header = f"### Skill: {self.name}"
        if self.description:
            header += f"\n{self.description}"
        return f"{header}\n{self.content.strip()}"


def _parse_front_matter(text: str) -> tuple[dict[str, str | bool | list[str]], str]:
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text
    body = text[match.end() :]
    meta: dict[str, str | bool | list[str]] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key in {"triggers", "keywords"}:
            meta["triggers"] = [v.strip() for v in re.split(r"[,;]", value) if v.strip()]
        elif key in {"always", "enabled"}:
            meta[key] = value.lower() in {"true", "yes", "1", "on"}
        else:
            meta[key] = value
    return meta, body


def load_skill_file(path: Path, source: str = "user") -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read skill %s: %s", path, exc)
        return None
    meta, body = _parse_front_matter(text)
    name = str(meta.get("name") or path.stem)
    return Skill(
        name=name,
        description=str(meta.get("description") or ""),
        triggers=list(meta.get("triggers") or []),  # type: ignore[arg-type]
        always=bool(meta.get("always", False)),
        enabled=bool(meta.get("enabled", True)),
        content=body.strip(),
        source=source,
        path=str(path),
    )


# ------------------------------------------------------------------- writing


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,58}[A-Za-z0-9]$")


class SkillError(ValueError):
    """Raised when a skill cannot be written."""


def slug_for(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "skill"


def render_skill(skill: Skill) -> str:
    """Serialise a skill back to markdown with front matter."""
    lines = ["---", f"name: {skill.name}"]
    if skill.description:
        lines.append(f"description: {skill.description}")
    if skill.triggers:
        lines.append(f"triggers: {', '.join(skill.triggers)}")
    if skill.always:
        lines.append("always: true")
    if not skill.enabled:
        lines.append("enabled: false")
    lines += ["---", "", skill.content.strip(), ""]
    return "\n".join(lines)


def _resolve(directory: Path, filename: str) -> Path:
    """Resolve a skill filename inside the skills directory, refusing escapes."""
    target = (directory / filename).resolve()
    root = directory.resolve()
    if target != root and root not in target.parents:
        raise SkillError(f"Refusing a path outside the skills folder: {filename}")
    return target


class SkillRegistry:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or SKILLS_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self.reload()

    # Documentation living beside the skills is not itself a skill.
    NOT_SKILLS = {"readme.md", "index.md", "contributing.md", "license.md"}

    def reload(self) -> None:
        self._skills = {}
        for path in sorted(self.directory.rglob("*.md")):
            if path.name.lower() in self.NOT_SKILLS:
                continue
            skill = load_skill_file(path)
            if skill is not None:
                self._skills[skill.name] = skill

    def add_plugin_skills(self, plugin_id: str, skills: list[Skill]) -> None:
        for skill in skills:
            skill.source = plugin_id
            self._skills[skill.name] = skill

    def list(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name.lower())

    def matching(self, message: str, limit: int = 4) -> list[Skill]:
        hits = [skill for skill in self.list() if skill.matches(message)]
        return hits[:limit]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def save(self, data: dict[str, Any], *, original: str | None = None) -> Skill:
        """Create or update a skill file, then reload."""
        name = str(data.get("name") or "").strip()
        if not SAFE_NAME.match(name):
            raise SkillError(
                "A skill name must be 2-60 characters of letters, numbers, "
                "spaces, hyphens or underscores."
            )

        existing = self._skills.get(original or name)
        if existing is not None and existing.source != "user":
            raise SkillError(
                f"'{existing.name}' comes from the {existing.source} plugin and "
                "cannot be edited here."
            )
        if original is None and name in self._skills:
            raise SkillError(f"A skill called '{name}' already exists.")

        triggers = data.get("triggers") or []
        if isinstance(triggers, str):
            triggers = [t.strip() for t in re.split(r"[,;\n]", triggers)]
        skill = Skill(
            name=name,
            description=str(data.get("description") or ""),
            triggers=[str(t).strip() for t in triggers if str(t).strip()],
            always=bool(data.get("always", False)),
            enabled=bool(data.get("enabled", True)),
            content=str(data.get("content") or ""),
            source="user",
        )

        # Renaming moves the file rather than leaving an orphan behind.
        if existing is not None and existing.path:
            old_path = Path(existing.path)
            new_path = _resolve(self.directory, f"{slug_for(name)}.md")
            new_path.write_text(render_skill(skill), encoding="utf-8")
            if old_path.exists() and old_path.resolve() != new_path.resolve():
                old_path.unlink()
        else:
            new_path = _resolve(self.directory, f"{slug_for(name)}.md")
            if new_path.exists():
                raise SkillError(f"{new_path.name} already exists on disk.")
            new_path.write_text(render_skill(skill), encoding="utf-8")

        self.reload()
        saved = self._skills.get(name)
        if saved is None:
            raise SkillError("The skill was written but could not be read back.")
        return saved

    def delete(self, name: str) -> None:
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Unknown skill: {name}")
        if skill.source != "user":
            raise SkillError(
                f"'{name}' comes from the {skill.source} plugin; disable the "
                "plugin instead."
            )
        if skill.path:
            path = _resolve(self.directory, Path(skill.path).name)
            if path.exists():
                path.unlink()
        self._skills.pop(name, None)

    def set_enabled(self, name: str, enabled: bool) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Unknown skill: {name}")
        data = skill.model_dump()
        data["enabled"] = enabled
        return self.save(data, original=name)

    def prompt_section(self, message: str, limit: int = 4) -> str:
        hits = self.matching(message, limit)
        if not hits:
            return ""
        blocks = "\n\n".join(skill.prompt_block() for skill in hits)
        return f"\n\n## Active skills\n\n{blocks}\n"


_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def reload_skill_registry() -> SkillRegistry:
    global _registry
    _registry = SkillRegistry()
    return _registry
