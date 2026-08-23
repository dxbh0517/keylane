"""Theme discovery, activation, and community install."""

from __future__ import annotations

import logging
import shutil
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import AppConfig, get_config

logger = logging.getLogger(__name__)

BUILTIN_THEMES: dict[str, dict[str, Any]] = {
    "default": {
        "name": "Default",
        "author": "built-in",
        "description": "Calm zinc surfaces with emerald accent.",
        "colors": {
            "bg": "#f4f4f5",
            "surface": "#ffffff",
            "text": "#18181b",
            "muted": "#71717a",
            "accent": "#059669",
            "border": "#e4e4e7",
            "danger": "#b91c1c",
        },
    },
    "midnight": {
        "name": "Midnight",
        "author": "built-in",
        "description": "Dark charcoal with soft cyan accent.",
        "colors": {
            "bg": "#09090b",
            "surface": "#18181b",
            "text": "#fafafa",
            "muted": "#a1a1aa",
            "accent": "#22d3ee",
            "border": "#27272a",
            "danger": "#f87171",
        },
    },
    "paper": {
        "name": "Paper",
        "author": "built-in",
        "description": "Warm paper light theme with brick accent.",
        "colors": {
            "bg": "#f7f3ee",
            "surface": "#fffdf9",
            "text": "#1c1917",
            "muted": "#78716c",
            "accent": "#c2410c",
            "border": "#e7e5e4",
            "danger": "#b91c1c",
        },
    },
}


class ThemeInfo(BaseModel):
    id: str
    name: str
    author: str = "unknown"
    version: str = "0.1.0"
    description: str = ""
    active: bool = False
    has_web: bool = False
    has_launcher: bool = False
    preview_colors: dict[str, str] = Field(default_factory=dict)
    path: str = ""


def _web_css(colors: dict[str, str]) -> str:
    vars_block = "\n".join(f"  --ag-{k}: {v};" for k, v in colors.items())
    return f""":root {{
{vars_block}
}}
"""


def _launcher_css(colors: dict[str, str]) -> str:
    defines = "\n".join(f"@define-color ag_{k} {v};" for k, v in colors.items())
    return f"""{defines}

window, window.background, .background {{
  background-color: @ag_bg;
  color: @ag_text;
}}

headerbar, headerbar.flat {{
  background-color: @ag_surface;
  color: @ag_text;
  border-bottom-color: @ag_border;
}}

headerbar label, headerbar .title {{
  color: @ag_text;
}}

.ag-shell {{
  background-color: @ag_bg;
}}

entry, textview, textview > text {{
  background-color: @ag_surface;
  color: @ag_text;
  border-color: @ag_border;
  caret-color: @ag_accent;
}}

entry:focus, textview:focus {{
  border-color: @ag_accent;
  box-shadow: 0 0 0 1px @ag_accent;
}}

label {{
  color: @ag_text;
}}

label.dim-label, .dim-label {{
  color: @ag_muted;
}}

button {{
  background-color: @ag_surface;
  color: @ag_text;
  border-color: @ag_border;
}}

button:hover {{
  background-color: mix(@ag_surface, @ag_accent, 0.12);
}}

button.suggested-action, button.suggested-action:hover {{
  background-color: @ag_accent;
  color: @ag_surface;
  border-color: @ag_accent;
}}

button.destructive-action {{
  color: @ag_danger;
  border-color: mix(@ag_danger, @ag_border, 0.5);
}}

checkbutton, checkbutton label {{
  color: @ag_text;
}}

dropdown, popover, popover.menu, popover > contents {{
  background-color: @ag_surface;
  color: @ag_text;
  border-color: @ag_border;
}}

listview, listview row {{
  background-color: @ag_surface;
  color: @ag_text;
}}

scrollbar {{
  background-color: transparent;
}}
"""


class ThemeManager:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.themes_dir = self.config.root / "themes"
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.config.root / "config" / "themes.toml"
        self.active_web_link = self.themes_dir / "active-web.css"
        self.active_launcher_link = self.themes_dir / "active-launcher.css"
        self._active = "default"
        self._ensure_builtins()
        self._load_state()
        self._publish_active()

    def _ensure_builtins(self) -> None:
        for tid, meta in BUILTIN_THEMES.items():
            folder = self.themes_dir / tid
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder / "theme.toml"
            colors = meta["colors"]
            if not manifest.exists():
                manifest.write_text(
                    "\n".join(
                        [
                            f'id = "{tid}"',
                            f'name = "{meta["name"]}"',
                            f'author = "{meta["author"]}"',
                            'version = "1.0.0"',
                            f'description = "{meta["description"]}"',
                            "",
                            "[colors]",
                            *[f'{k} = "{v}"' for k, v in colors.items()],
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            # Always refresh generated CSS for built-ins so launcher/web stay complete.
            (folder / "web.css").write_text(_web_css(colors), encoding="utf-8")
            (folder / "launcher.css").write_text(_launcher_css(colors), encoding="utf-8")

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self._save_state()
            return
        with self.state_path.open("rb") as fh:
            raw = tomllib.load(fh)
        self._active = str(raw.get("active", "default"))
        if not (self.themes_dir / self._active / "theme.toml").exists():
            self._active = "default"

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(f'active = "{self._active}"\n', encoding="utf-8")

    def _publish_active(self) -> None:
        """Copy active CSS to stable paths for the GTK launcher."""
        web = self.web_css_path()
        launcher = self.launcher_css_path()
        self.active_web_link.write_text(web.read_text(encoding="utf-8"), encoding="utf-8")
        self.active_launcher_link.write_text(
            launcher.read_text(encoding="utf-8"), encoding="utf-8"
        )

    def list(self) -> list[ThemeInfo]:
        themes: list[ThemeInfo] = []
        for folder in sorted(self.themes_dir.iterdir()):
            if not folder.is_dir() or folder.name.startswith("active-"):
                continue
            manifest = folder / "theme.toml"
            if not manifest.exists():
                continue
            with manifest.open("rb") as fh:
                data = tomllib.load(fh)
            colors = {str(k): str(v) for k, v in (data.get("colors") or {}).items()}
            tid = str(data.get("id") or folder.name)
            themes.append(
                ThemeInfo(
                    id=tid,
                    name=str(data.get("name") or folder.name),
                    author=str(data.get("author") or "unknown"),
                    version=str(data.get("version") or "0.1.0"),
                    description=str(data.get("description") or ""),
                    active=tid == self._active,
                    has_web=(folder / "web.css").exists(),
                    has_launcher=(folder / "launcher.css").exists(),
                    preview_colors=colors,
                    path=str(folder),
                )
            )
        return themes

    @property
    def active_id(self) -> str:
        return self._active

    def set_active(self, theme_id: str) -> ThemeInfo:
        folder = self.themes_dir / theme_id
        if not (folder / "theme.toml").exists():
            raise KeyError(f"Unknown theme: {theme_id}")
        self._active = theme_id
        self._save_state()
        self._publish_active()
        for info in self.list():
            if info.id == theme_id:
                return info
        raise KeyError(theme_id)

    def web_css_path(self) -> Path:
        path = self.themes_dir / self._active / "web.css"
        if not path.exists():
            path = self.themes_dir / "default" / "web.css"
        return path

    def launcher_css_path(self) -> Path:
        path = self.themes_dir / self._active / "launcher.css"
        if not path.exists():
            path = self.themes_dir / "default" / "launcher.css"
        return path

    def web_css_text(self) -> str:
        return self.web_css_path().read_text(encoding="utf-8")

    def launcher_css_text(self) -> str:
        return self.launcher_css_path().read_text(encoding="utf-8")

    def install_zip(self, zip_bytes: bytes) -> ThemeInfo:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "theme.zip"
            zpath.write_bytes(zip_bytes)
            extract = Path(tmp) / "extracted"
            extract.mkdir()
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(extract)
            manifests = list(extract.rglob("theme.toml"))
            if not manifests:
                raise ValueError("Zip must contain theme.toml")
            manifest = manifests[0]
            with manifest.open("rb") as fh:
                data = tomllib.load(fh)
            tid = str(data.get("id") or manifest.parent.name)
            dest = self.themes_dir / tid
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(manifest.parent, dest)
            colors = {str(k): str(v) for k, v in (data.get("colors") or {}).items()}
            if colors:
                if not (dest / "web.css").exists():
                    (dest / "web.css").write_text(_web_css(colors), encoding="utf-8")
                if not (dest / "launcher.css").exists():
                    (dest / "launcher.css").write_text(
                        _launcher_css(colors), encoding="utf-8"
                    )
        for info in self.list():
            if info.id == tid:
                return info
        raise RuntimeError("Theme installed but not discoverable")

    def uninstall(self, theme_id: str) -> None:
        if theme_id in BUILTIN_THEMES:
            raise ValueError("Built-in themes cannot be removed")
        folder = self.themes_dir / theme_id
        if not folder.exists():
            raise KeyError(f"Unknown theme: {theme_id}")
        if self._active == theme_id:
            self._active = "default"
            self._save_state()
            self._publish_active()
        shutil.rmtree(folder)


_themes: ThemeManager | None = None


def get_theme_manager(config: AppConfig | None = None) -> ThemeManager:
    global _themes
    if _themes is None:
        _themes = ThemeManager(config)
    return _themes


def reload_theme_manager(config: AppConfig | None = None) -> ThemeManager:
    global _themes
    _themes = ThemeManager(config)
    return _themes
