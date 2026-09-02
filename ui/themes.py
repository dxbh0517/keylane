"""Themes: token tables rendered into the Spotlight stylesheet.

GTK CSS has no custom properties, so a theme cannot be a handful of variables
the stylesheet reads at runtime. Instead ``ui/spotlight.css.in`` carries
``{{token}}`` holes and a theme fills them: ``{{light.panel-bg}}`` and
``{{dark.panel-bg}}`` come from the theme's ``[light]``/``[dark]`` tables, and
``{{radius-panel}}`` from ``[common]``. Both schemes are rendered into one
stylesheet, so switching light/dark stays a class swap with no reload.

Themes live in two places: the ones that ship, next to this file in
``ui/themes/``, and the user's own under ``data/themes/``. A user theme that
sets ``extends`` starts from that theme's tokens and overrides only what it
names, so a new theme can be four lines long.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover — Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).resolve().parent / "themes"
TEMPLATE_PATH = Path(__file__).resolve().parent / "spotlight.css.in"
DEFAULT_THEME_ID = "glass-console"

_HOLE = re.compile(r"\{\{\s*(?:(light|dark)\.)?([a-z0-9-]+)\s*\}\}")


class ThemeError(RuntimeError):
    """A theme is missing tokens, or names a base that does not exist."""


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    description: str = ""
    author: str = ""
    source: str = "built-in"
    path: Path | None = None
    common: dict[str, str] = field(default_factory=dict)
    light: dict[str, str] = field(default_factory=dict)
    dark: dict[str, str] = field(default_factory=dict)

    def scheme(self, dark: bool) -> dict[str, str]:
        """Every token that resolves for one scheme: commons plus that scheme."""
        return {**self.common, **(self.dark if dark else self.light)}


def user_dir() -> Path:
    from daemon.paths import THEMES_DIR

    return THEMES_DIR


def theme_dirs() -> list[Path]:
    return [BUILTIN_DIR, user_dir()]


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _theme_from(path: Path, source: str, seen: tuple[str, ...] = ()) -> Theme:
    raw = _read(path)
    meta = raw.get("theme", {})
    theme_id = str(meta.get("id") or path.stem)
    common = {str(k): str(v) for k, v in (raw.get("common") or {}).items()}
    light = {str(k): str(v) for k, v in (raw.get("light") or {}).items()}
    dark = {str(k): str(v) for k, v in (raw.get("dark") or {}).items()}

    base_id = str(meta.get("extends") or "").strip()
    if not base_id and source == "user":
        # A user theme is a patch on the default unless it says otherwise;
        # writing all 140-odd tokens to change an accent would be absurd.
        base_id = DEFAULT_THEME_ID
    if base_id:
        if base_id == theme_id or base_id in seen:
            raise ThemeError(f"theme {theme_id}: extends loops back to {base_id}")
        base = load_theme(base_id, seen=(*seen, theme_id))
        common = {**base.common, **common}
        light = {**base.light, **light}
        dark = {**base.dark, **dark}

    return Theme(
        id=theme_id,
        name=str(meta.get("name") or theme_id.replace("-", " ").title()),
        description=str(meta.get("description") or ""),
        author=str(meta.get("author") or ""),
        source=source,
        path=path,
        common=common,
        light=light,
        dark=dark,
    )


def list_themes() -> list[Theme]:
    """Every readable theme, built-in first, a user theme shadowing its id."""
    found: dict[str, Theme] = {}
    for directory, source in ((BUILTIN_DIR, "built-in"), (user_dir(), "user")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.toml")):
            try:
                theme = _theme_from(path, source)
            except (ThemeError, OSError, tomllib.TOMLDecodeError):
                logger.exception("skipping unreadable theme %s", path)
                continue
            found[theme.id] = theme
    return sorted(found.values(), key=lambda t: (t.source != "built-in", t.name.lower()))


def load_theme(theme_id: str, *, seen: tuple[str, ...] = ()) -> Theme:
    for directory, source in ((user_dir(), "user"), (BUILTIN_DIR, "built-in")):
        path = directory / f"{theme_id}.toml"
        if path.is_file():
            return _theme_from(path, source, seen)
    raise ThemeError(f"no theme named {theme_id!r}")


def template_tokens() -> tuple[set[str], set[str]]:
    """(scheme tokens, common tokens) the stylesheet asks a theme for."""
    scheme: set[str] = set()
    common: set[str] = set()
    for prefix, name in _HOLE.findall(TEMPLATE_PATH.read_text(encoding="utf-8")):
        (scheme if prefix else common).add(name)
    return scheme, common


def missing_tokens(theme: Theme) -> list[str]:
    scheme_names, common_names = template_tokens()
    missing = [f"common.{n}" for n in sorted(common_names) if n not in theme.common]
    for label, table in (("light", theme.light), ("dark", theme.dark)):
        missing += [f"{label}.{n}" for n in sorted(scheme_names) if n not in table]
    return missing


def render_css(theme: Theme) -> str:
    """Fill the stylesheet's holes from *theme*; every hole must resolve."""
    missing = missing_tokens(theme)
    if missing:
        raise ThemeError(f"theme {theme.id} is missing {len(missing)} tokens: {', '.join(missing[:6])}")

    tables = {"light": theme.light, "dark": theme.dark, "": theme.common}

    def fill(match: re.Match[str]) -> str:
        return tables[match.group(1) or ""][match.group(2)]

    return _HOLE.sub(fill, TEMPLATE_PATH.read_text(encoding="utf-8"))
