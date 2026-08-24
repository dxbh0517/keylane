"""Theme discovery, activation, and community install.

A Keylane theme controls three surfaces:

``web.css``      the control panel
``launcher.css`` GTK styling for the popup
``[popup]``      the popup's *shape* — bar, panel, window or orb — plus its
                 geometry, position, animation and which parts are visible

The popup spec is what makes the theme system more than a palette: the same
launcher renders as a macOS-Spotlight bar, a full assistant window, or a small
floating orb depending only on the active theme.
"""

from __future__ import annotations

import logging
import shutil
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.config import AppConfig, get_config

logger = logging.getLogger(__name__)

PopupMode = Literal["bar", "panel", "window", "orb"]
PopupPosition = Literal[
    "center",
    "top",
    "top-left",
    "top-right",
    "bottom",
    "bottom-left",
    "bottom-right",
    "left",
    "right",
]

VALID_MODES = {"bar", "panel", "window", "orb"}
VALID_POSITIONS = {
    "center",
    "top",
    "top-left",
    "top-right",
    "bottom",
    "bottom-left",
    "bottom-right",
    "left",
    "right",
}


class PopupSpec(BaseModel):
    """Everything a theme can say about the shape and behaviour of the popup."""

    mode: PopupMode = "panel"
    """bar = one Spotlight-style input row that grows with results.
    panel = input plus a compact result area.
    window = a full assistant window with header and history.
    orb = a small always-visible dot that expands on activation."""

    width: int = 720
    """Popup width in logical pixels."""

    height: int = 0
    """Fixed height, or 0 to size to content."""

    max_height: int = 560
    """Upper bound once results appear."""

    position: PopupPosition = "center"
    offset_x: int = 0
    offset_y: int = 0
    """Nudge from the anchor, in pixels. For 'center', offset_y lifts the popup;
    Spotlight-style themes usually sit above the true centre."""

    corner_radius: int = 16
    padding: int = 14
    """Outer padding inside the popup shell."""

    opacity: float = 1.0
    blur_background: bool = True
    """Ask the compositor for a translucent backdrop where supported."""

    shadow: bool = True
    decorated: bool = False
    """False gives a chromeless popup — the Spotlight look."""

    # What the popup shows
    show_logo: bool = True
    show_title: bool = False
    show_status_chips: bool = True
    show_project_picker: bool = True
    show_hints: bool = True
    show_results: bool = True

    # Behaviour
    dismiss_on_focus_loss: bool = True
    animation: Literal["none", "fade", "scale", "slide"] = "scale"
    animation_ms: int = 140
    input_placeholder: str = "Ask anything…"
    orb_size: int = 72
    """Diameter of the collapsed orb, for mode = orb."""

    font_family: str = ""
    font_size: int = 15
    """Base font size for the prompt entry."""

    @field_validator("mode", mode="before")
    @classmethod
    def _valid_mode(cls, value: Any) -> str:
        text = str(value or "panel").strip().lower()
        return text if text in VALID_MODES else "panel"

    @field_validator("position", mode="before")
    @classmethod
    def _valid_position(cls, value: Any) -> str:
        text = str(value or "center").strip().lower()
        return text if text in VALID_POSITIONS else "center"

    @field_validator("opacity", mode="before")
    @classmethod
    def _clamp_opacity(cls, value: Any) -> float:
        try:
            return max(0.3, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 1.0


# Presets a theme can start from with `preset = "spotlight"`, then override.
POPUP_PRESETS: dict[str, dict[str, Any]] = {
    "spotlight": {
        # Proportioned like macOS Spotlight: a wide, shallow bar. The collapsed
        # row is ~64px tall — padding 10 either side of a 44px entry — which is
        # what keeps it reading as a search field rather than a dialog.
        "mode": "bar",
        "width": 720,
        "max_height": 480,
        "position": "center",
        "offset_y": -150,
        "corner_radius": 18,
        "padding": 10,
        "decorated": False,
        "show_logo": True,
        "show_title": False,
        "show_status_chips": False,
        "show_project_picker": False,
        "show_hints": False,
        "animation": "scale",
        "animation_ms": 140,
        "font_size": 20,
    },
    "panel": {
        "mode": "panel",
        "width": 720,
        "max_height": 520,
        "position": "center",
        "offset_y": -80,
        "corner_radius": 18,
        "padding": 14,
        "decorated": False,
    },
    "window": {
        "mode": "window",
        "width": 860,
        "height": 620,
        "max_height": 900,
        "position": "center",
        "corner_radius": 12,
        "padding": 16,
        "decorated": True,
        "show_title": True,
        "show_status_chips": True,
        "show_project_picker": True,
        "dismiss_on_focus_loss": False,
    },
    "orb": {
        "mode": "orb",
        "width": 420,
        "max_height": 380,
        "position": "bottom-right",
        "offset_x": -32,
        "offset_y": -32,
        "corner_radius": 28,
        "padding": 10,
        "decorated": False,
        "show_logo": True,
        "show_title": False,
        "show_status_chips": False,
        "show_project_picker": False,
        "show_hints": False,
        "orb_size": 68,
        "animation": "scale",
    },
}


BUILTIN_THEMES: dict[str, dict[str, Any]] = {
    "default": {
        "name": "Default",
        "author": "built-in",
        "description": "A translucent Spotlight-style search bar floating above the desktop.",
        "colors": {
            "bg": "#f6f6f7",
            "surface": "#ffffff",
            "text": "#111113",
            "muted": "#6b6b73",
            "accent": "#2563eb",
            "border": "#e2e2e6",
            "danger": "#b91c1c",
        },
        "popup": {"preset": "spotlight"},
    },
    "midnight": {
        "name": "Midnight",
        "author": "built-in",
        "description": "The same Spotlight bar in dark charcoal with a cyan accent.",
        "colors": {
            "bg": "#09090b",
            "surface": "#18181b",
            "text": "#fafafa",
            "muted": "#a1a1aa",
            "accent": "#22d3ee",
            "border": "#27272a",
            "danger": "#f87171",
        },
        "popup": {"preset": "spotlight", "opacity": 0.97},
    },
    "panel": {
        "name": "Panel",
        "author": "built-in",
        "description": "A centred panel with worker chips and a project picker.",
        "colors": {
            "bg": "#f4f4f5",
            "surface": "#ffffff",
            "text": "#18181b",
            "muted": "#71717a",
            "accent": "#059669",
            "border": "#e4e4e7",
            "danger": "#b91c1c",
        },
        "popup": {"preset": "panel"},
    },
    "paper": {
        "name": "Paper",
        "author": "built-in",
        "description": "Warm paper light theme with a brick accent, in panel form.",
        "colors": {
            "bg": "#f7f3ee",
            "surface": "#fffdf9",
            "text": "#1c1917",
            "muted": "#78716c",
            "accent": "#c2410c",
            "border": "#e7e5e4",
            "danger": "#b91c1c",
        },
        "popup": {"preset": "panel"},
    },
    "studio": {
        "name": "Studio",
        "author": "built-in",
        "description": "A full assistant window with a title bar, history and status chips.",
        "colors": {
            "bg": "#0f1116",
            "surface": "#171a21",
            "text": "#e8ecf4",
            "muted": "#9aa3b5",
            "accent": "#8b5cf6",
            "border": "#242938",
            "danger": "#f87171",
        },
        "popup": {"preset": "window"},
    },
    "orb": {
        "name": "Orb",
        "author": "built-in",
        "description": "A small floating orb in the corner that expands when you speak to it.",
        "colors": {
            "bg": "#0b0f14",
            "surface": "#131a22",
            "text": "#eaf2ff",
            "muted": "#8ba0b8",
            "accent": "#38bdf8",
            "border": "#1e2a37",
            "danger": "#fb7185",
        },
        "popup": {"preset": "orb"},
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
    has_popup_css: bool = False
    preview_colors: dict[str, str] = Field(default_factory=dict)
    popup: PopupSpec = Field(default_factory=PopupSpec)
    path: str = ""


def parse_popup(raw: dict[str, Any] | None) -> PopupSpec:
    """Build a popup spec from a theme's ``[popup]`` table, honouring presets."""
    data = dict(raw or {})
    preset_name = str(data.pop("preset", "") or "").strip().lower()
    base: dict[str, Any] = {}
    if preset_name in POPUP_PRESETS:
        base = dict(POPUP_PRESETS[preset_name])
    elif preset_name:
        logger.warning("Unknown popup preset '%s'; using defaults.", preset_name)
    merged = {**base, **data}
    known = set(PopupSpec.model_fields)
    unknown = set(merged) - known
    for key in unknown:
        merged.pop(key)
    if unknown:
        logger.info("Ignoring unknown [popup] keys: %s", ", ".join(sorted(unknown)))
    try:
        return PopupSpec(**merged)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Invalid [popup] section (%s); using defaults.", exc)
        return PopupSpec()


def _web_css(colors: dict[str, str], popup: PopupSpec) -> str:
    vars_block = "\n".join(f"  --ag-{k}: {v};" for k, v in colors.items())
    return f""":root {{
{vars_block}
  --ag-radius: {popup.corner_radius}px;
  --ag-font-size: {popup.font_size}px;
}}
"""


def _launcher_css(colors: dict[str, str], popup: PopupSpec) -> str:
    """GTK4 CSS generated from a theme's colours plus its popup geometry."""
    defines = "\n".join(f"@define-color ag_{k} {v};" for k, v in colors.items())
    radius = popup.corner_radius
    font = f"font-family: {popup.font_family};" if popup.font_family else ""

    # The entry is the popup's whole reason to exist, so it sets the height:
    # roughly 2.2x the type size, floored so small fonts still give a
    # comfortable target. Everything else is measured against it.
    entry_height = max(int(popup.font_size * 2.2), 40)
    inner_radius = max(radius - popup.padding, 8)

    if popup.mode == "bar":
        # A bar has one surface. The entry must be invisible furniture inside
        # it — a second bordered box within the bar is what makes a Spotlight
        # clone look like a form.
        entry_rules = f"""
.keylane-prompt {{
  background: none;
  background-color: transparent;
  border: none;
  outline: none;
  box-shadow: none;
  padding: 0 6px;
  min-height: {entry_height}px;
  font-size: {popup.font_size}px;
  /* Large type reads too loose at default tracking. */
  letter-spacing: -0.011em;
  color: @ag_text;
  caret-color: @ag_accent;
}}
.keylane-prompt:focus,
.keylane-prompt:focus-within,
.keylane-prompt > text {{
  background: none;
  background-color: transparent;
  border: none;
  outline: none;
  box-shadow: none;
}}
"""
    else:
        entry_rules = f"""
.keylane-prompt {{
  background-color: @ag_surface;
  color: @ag_text;
  border: 1px solid @ag_border;
  border-radius: {inner_radius}px;
  caret-color: @ag_accent;
  font-size: {popup.font_size}px;
  min-height: {entry_height}px;
  padding: 0 12px;
}}
.keylane-prompt:focus-within {{
  border-color: @ag_accent;
  box-shadow: 0 0 0 2px alpha(@ag_accent, 0.25);
}}
"""

    return f"""{defines}

/* Generated from [colors] + [popup]. Ship your own launcher.css to override. */

window.keylane-popup {{
  background-color: transparent;
}}

.keylane-shell {{
  background-color: alpha(@ag_bg, {popup.opacity:.2f});
  color: @ag_text;
  border-radius: {radius}px;
  /* A brighter hairline reads as light catching the edge of a real material. */
  border: 1px solid alpha(@ag_border, 0.85);
  padding: {popup.padding}px;
  {font}
}}

.keylane-card {{
  background-color: @ag_surface;
  border-radius: {inner_radius}px;
  border: 1px solid alpha(@ag_border, 0.9);
}}
{entry_rules}
/* The mark sits at the head of the bar, tinted, never a pasted-on tile. */
.keylane-mark {{
  color: @ag_accent;
  margin: 0 4px 0 6px;
}}

.keylane-title {{
  color: @ag_text;
  font-weight: 700;
  letter-spacing: -0.014em;
}}
.keylane-subtitle, .keylane-hint, .keylane-progress {{ color: @ag_muted; }}

.keylane-result {{
  background-color: @ag_surface;
  color: @ag_text;
  border-radius: {inner_radius}px;
  border: 1px solid @ag_border;
}}
.keylane-progress {{
  font-size: {max(popup.font_size - 6, 12)}px;
  padding: 2px 8px 4px;
}}
/* Results are a second region, so the divider only appears when they do. */
.keylane-result-view {{
  border-top: 1px solid alpha(@ag_border, 0.7);
  margin-top: {popup.padding}px;
  padding-top: {popup.padding}px;
}}

.keylane-chip {{
  background-color: alpha(@ag_border, 0.55);
  color: @ag_muted;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 11px;
}}
.keylane-chip.on {{ background-color: alpha(@ag_accent, 0.16); color: @ag_accent; }}
.keylane-chip.warn {{ background-color: alpha(@ag_danger, 0.16); color: @ag_danger; }}

.keylane-orb {{
  background-color: @ag_accent;
  border-radius: {popup.orb_size // 2}px;
  min-width: {popup.orb_size}px;
  min-height: {popup.orb_size}px;
}}

/* Round, quiet, and the same height as the entry so the row optically aligns. */
.keylane-icon-btn {{
  background: none;
  background-color: transparent;
  border: none;
  box-shadow: none;
  border-radius: 999px;
  min-width: {entry_height - 8}px;
  min-height: {entry_height - 8}px;
  padding: 0;
  color: @ag_muted;
}}
.keylane-icon-btn:hover {{
  background-color: alpha(@ag_text, 0.08);
  color: @ag_text;
}}
.keylane-icon-btn:checked,
.keylane-icon-btn.recording {{
  background-color: alpha(@ag_danger, 0.18);
  color: @ag_danger;
}}
.keylane-icon-btn.speaking {{
  background-color: alpha(@ag_accent, 0.18);
  color: @ag_accent;
}}

/* Device chip: status, not a button you are meant to reach for. */
.keylane-device-chip {{
  background: none;
  background-color: transparent;
  border: none;
  box-shadow: none;
  padding: 2px 8px;
  min-height: 0;
  border-radius: 999px;
  color: alpha(@ag_muted, 0.85);
}}
.keylane-device-chip:hover {{
  background-color: alpha(@ag_text, 0.07);
  color: @ag_text;
}}
/* Running somewhere other than your choice is worth a hint of colour. */
.keylane-device-chip.fallback {{ color: alpha(#d97706, 0.95); }}
.keylane-device-text {{
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.07em;
}}
.keylane-device-menu {{ background-color: @ag_surface; }}
.keylane-device-option {{ font-size: 12.5px; color: @ag_text; }}

/* Slash-command list */
.keylane-skill-view {{
  border-top: 1px solid alpha(@ag_border, 0.7);
  margin-top: {popup.padding}px;
  padding-top: 4px;
}}
.keylane-skill-list {{ background-color: transparent; }}
.keylane-skill-list row {{
  border-radius: {max(radius - 6, 6)}px;
  background-color: transparent;
}}
.keylane-skill-list row:selected {{
  background-color: alpha(@ag_accent, 0.16);
}}
.keylane-skill-name {{
  font-family: monospace;
  font-size: 13px;
  font-weight: 600;
  color: @ag_accent;
}}

.keylane-send {{
  border-radius: {inner_radius}px;
  min-height: {entry_height - 6}px;
  padding: 0 16px;
  font-weight: 600;
}}

button.suggested-action, button.suggested-action:hover {{
  background-color: @ag_accent;
  color: @ag_surface;
  border-color: @ag_accent;
}}

entry, textview, textview > text {{
  background-color: @ag_surface;
  color: @ag_text;
  border-color: @ag_border;
  caret-color: @ag_accent;
}}

label {{ color: @ag_text; }}
label.dim-label, .dim-label {{ color: @ag_muted; }}

popover, popover.menu, popover > contents, dropdown, listview, listview row {{
  background-color: @ag_surface;
  color: @ag_text;
  border-color: @ag_border;
}}

headerbar, headerbar.flat {{
  background-color: transparent;
  color: @ag_text;
  border-bottom: none;
  box-shadow: none;
  min-height: 0;
}}

scrollbar {{ background-color: transparent; }}

/* ---------------------------------------------------- working orb + canvas */

window.keylane-orb-window {{ background-color: transparent; }}

.keylane-result-shell {{
  background-color: alpha(@ag_bg, 0.97);
  border: 1px solid alpha(@ag_border, 0.9);
  border-radius: {max(radius + 6, 20)}px;
  padding: {popup.padding + 2}px;
}}
/* Collapsed, it is a circle — the squircle grows out of it. */
.keylane-result-shell.is-orb {{
  border-radius: 999px;
  padding: 0;
  background-color: alpha(@ag_accent, 0.16);
  border-color: alpha(@ag_accent, 0.5);
}}
.keylane-result-shell.failed {{ border-color: alpha(@ag_danger, 0.7); }}
/* Hovering pauses the auto-dismiss; the brighter edge says so. */
.keylane-result-shell.hovered {{ border-color: alpha(@ag_accent, 0.55); }}

.canvas-title {{
  font-size: {popup.font_size - 2}px;
  font-weight: 700;
  letter-spacing: -0.012em;
  color: @ag_text;
}}
.canvas-summary {{ color: @ag_muted; font-size: 13px; }}
.canvas-heading {{
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: @ag_muted;
  margin-top: 4px;
}}
.canvas-text {{ color: @ag_text; font-size: 13.5px; }}
.canvas-bullet {{ color: @ag_accent; font-size: 13.5px; }}
.canvas-source {{ color: alpha(@ag_muted, 0.8); font-size: 11px; }}

.canvas-stat {{
  background-color: @ag_surface;
  border: 1px solid @ag_border;
  border-radius: {max(radius - 6, 6)}px;
  padding: 8px 10px;
}}
.canvas-stat-label {{
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  color: @ag_muted;
}}
.canvas-stat-value {{ font-size: 17px; font-weight: 700; color: @ag_text; }}
.canvas-stat-detail {{ font-size: 11px; color: @ag_muted; }}

.canvas-th {{
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.05em;
  color: @ag_muted;
}}
.canvas-td {{ font-size: 12.5px; color: @ag_text; }}

.canvas-code-view {{
  background-color: @ag_surface;
  border: 1px solid @ag_border;
  border-radius: {max(radius - 6, 6)}px;
}}
.canvas-code {{
  font-family: monospace;
  font-size: 12px;
  color: @ag_text;
  padding: 8px 10px;
}}

.canvas-note {{
  border-radius: {max(radius - 6, 6)}px;
  padding: 8px 11px;
  background-color: alpha(@ag_accent, 0.12);
  border-left: 3px solid @ag_accent;
}}
.canvas-note.success {{
  background-color: alpha(@ag_accent, 0.12);
  border-left-color: @ag_accent;
}}
.canvas-note.warning {{
  background-color: alpha(#d97706, 0.14);
  border-left-color: #d97706;
}}
.canvas-note.danger {{
  background-color: alpha(@ag_danger, 0.14);
  border-left-color: @ag_danger;
}}

.canvas-link {{
  color: @ag_accent;
  padding: 2px 0;
  min-height: 0;
}}
"""


def _manifest_text(tid: str, meta: dict[str, Any]) -> str:
    colors = meta["colors"]
    popup = meta.get("popup") or {}
    lines = [
        f'id = "{tid}"',
        f'name = "{meta["name"]}"',
        f'author = "{meta["author"]}"',
        'version = "1.0.0"',
        f'description = "{meta["description"]}"',
        "",
        "[colors]",
        *[f'{k} = "{v}"' for k, v in colors.items()],
        "",
        "[popup]",
    ]
    for key, value in popup.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f'{key} = "{value}"')
    lines.append("")
    return "\n".join(lines)


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

    # ---------------------------------------------------------------- builtins

    def _ensure_builtins(self) -> None:
        for tid, meta in BUILTIN_THEMES.items():
            folder = self.themes_dir / tid
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder / "theme.toml"
            colors = meta["colors"]
            popup = parse_popup(meta.get("popup"))
            # Built-in manifests and CSS are generated, so always refresh them:
            # an upgrade that changes a preset must actually reach the user.
            # Customising a built-in means copying it to a new id.
            manifest.write_text(_manifest_text(tid, meta), encoding="utf-8")
            (folder / "web.css").write_text(_web_css(colors, popup), encoding="utf-8")
            (folder / "launcher.css").write_text(
                _launcher_css(colors, popup), encoding="utf-8"
            )

    # ------------------------------------------------------------------- state

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
        """Copy the active CSS to stable paths the GTK launcher can read offline."""
        self.active_web_link.write_text(
            self.web_css_path().read_text(encoding="utf-8"), encoding="utf-8"
        )
        self.active_launcher_link.write_text(
            self.launcher_css_path().read_text(encoding="utf-8"), encoding="utf-8"
        )
        (self.themes_dir / "active-popup.json").write_text(
            self.active_popup().model_dump_json(indent=2), encoding="utf-8"
        )

    # -------------------------------------------------------------- discovery

    def _read_manifest(self, folder: Path) -> dict[str, Any] | None:
        manifest = folder / "theme.toml"
        if not manifest.exists():
            return None
        try:
            with manifest.open("rb") as fh:
                return tomllib.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Theme %s has an unreadable manifest: %s", folder.name, exc)
            return None

    def list(self) -> list[ThemeInfo]:
        themes: list[ThemeInfo] = []
        for folder in sorted(self.themes_dir.iterdir()):
            if not folder.is_dir() or folder.name.startswith("active-"):
                continue
            data = self._read_manifest(folder)
            if data is None:
                continue
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
                    has_popup_css=(folder / "popup.css").exists(),
                    preview_colors=colors,
                    popup=parse_popup(data.get("popup")),
                    path=str(folder),
                )
            )
        return themes

    @property
    def active_id(self) -> str:
        return self._active

    def active_popup(self) -> PopupSpec:
        data = self._read_manifest(self.themes_dir / self._active)
        if data is None:
            return PopupSpec()
        return parse_popup(data.get("popup"))

    def active_colors(self) -> dict[str, str]:
        data = self._read_manifest(self.themes_dir / self._active) or {}
        return {str(k): str(v) for k, v in (data.get("colors") or {}).items()}

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

    # ------------------------------------------------------------------- CSS

    def _asset(self, filename: str) -> Path:
        path = self.themes_dir / self._active / filename
        if not path.exists():
            path = self.themes_dir / "default" / filename
        return path

    def web_css_path(self) -> Path:
        return self._asset("web.css")

    def launcher_css_path(self) -> Path:
        return self._asset("launcher.css")

    def popup_css_text(self) -> str:
        """Optional extra GTK CSS applied on top of launcher.css."""
        path = self.themes_dir / self._active / "popup.css"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def web_css_text(self) -> str:
        return self.web_css_path().read_text(encoding="utf-8")

    def launcher_css_text(self) -> str:
        text = self.launcher_css_path().read_text(encoding="utf-8")
        extra = self.popup_css_text()
        return f"{text}\n\n/* popup.css */\n{extra}" if extra else text

    # --------------------------------------------------------------- install

    def install_zip(self, zip_bytes: bytes) -> ThemeInfo:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "theme.zip"
            zpath.write_bytes(zip_bytes)
            extract = Path(tmp) / "extracted"
            extract.mkdir()
            with zipfile.ZipFile(zpath) as zf:
                _safe_extract(zf, extract)
            manifests = list(extract.rglob("theme.toml"))
            if not manifests:
                raise ValueError("The zip must contain a theme.toml")
            manifest = manifests[0]
            with manifest.open("rb") as fh:
                data = tomllib.load(fh)
            tid = str(data.get("id") or manifest.parent.name).strip()
            if not tid or "/" in tid or tid.startswith(".") or tid.startswith("active-"):
                raise ValueError(f"Invalid theme id: {tid!r}")
            if tid in BUILTIN_THEMES:
                raise ValueError(f"'{tid}' is a built-in theme id; choose another.")
            dest = self.themes_dir / tid
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(manifest.parent, dest)

            colors = {str(k): str(v) for k, v in (data.get("colors") or {}).items()}
            popup = parse_popup(data.get("popup"))
            if colors:
                if not (dest / "web.css").exists():
                    (dest / "web.css").write_text(_web_css(colors, popup), encoding="utf-8")
                if not (dest / "launcher.css").exists():
                    (dest / "launcher.css").write_text(
                        _launcher_css(colors, popup), encoding="utf-8"
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
        shutil.rmtree(folder)
        self._publish_active()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a zip, refusing entries that escape the destination."""
    root = destination.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Refusing unsafe archive path: {member.filename}")
    archive.extractall(destination)


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
