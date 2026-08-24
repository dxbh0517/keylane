"""Load the active theme's popup spec and CSS for the GTK launcher.

The gateway is the source of truth. When it is not running we fall back to the
files the theme manager publishes on disk, so the popup still opens with the
right shape after a cold boot.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk, Gtk  # noqa: E402

from launcher.gateway import GatewayClient  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SHARE = Path.home() / ".local" / "share" / "ai-gateway"


@dataclass
class PopupSpec:
    """Mirror of ``app.themes.PopupSpec`` — the theme's control over the popup."""

    mode: str = "bar"
    width: int = 680
    height: int = 0
    max_height: int = 460
    position: str = "center"
    offset_x: int = 0
    offset_y: int = -140
    corner_radius: int = 14
    padding: int = 8
    opacity: float = 1.0
    blur_background: bool = True
    shadow: bool = True
    decorated: bool = False
    show_logo: bool = True
    show_title: bool = False
    show_status_chips: bool = False
    show_project_picker: bool = False
    show_hints: bool = False
    show_results: bool = True
    dismiss_on_focus_loss: bool = True
    animation: str = "scale"
    animation_ms: int = 120
    input_placeholder: str = "Ask anything…"
    orb_size: int = 68
    font_family: str = ""
    font_size: int = 19

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PopupSpec":
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for spec_field in fields(cls):
            key = spec_field.name
            if not data or key not in data:
                continue
            default = getattr(defaults, key)
            value = data[key]
            try:
                # bool before int — bool is a subclass of int.
                if isinstance(default, bool):
                    kwargs[key] = bool(value)
                elif isinstance(default, int):
                    kwargs[key] = int(value)
                elif isinstance(default, float):
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = str(value)
            except (TypeError, ValueError):
                logger.debug("Ignoring bad popup value %s=%r", key, value)
        return cls(**kwargs)


@dataclass
class ActiveTheme:
    theme_id: str = "default"
    popup: PopupSpec = field(default_factory=PopupSpec)
    colors: dict[str, str] = field(default_factory=dict)
    css: str = ""


def _local_popup_spec() -> dict[str, Any] | None:
    for candidate in (
        ROOT / "themes" / "active-popup.json",
        SHARE / "themes" / "active-popup.json",
    ):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return None


def _local_css() -> str:
    for candidate in (
        ROOT / "themes" / "active-launcher.css",
        SHARE / "themes" / "active-launcher.css",
        ROOT / "themes" / "default" / "launcher.css",
        SHARE / "themes" / "default" / "launcher.css",
    ):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


def load_active_theme(client: GatewayClient) -> ActiveTheme:
    remote = client.active_theme()
    if remote:
        return ActiveTheme(
            theme_id=str(remote.get("id") or "default"),
            popup=PopupSpec.from_dict(remote.get("popup")),
            colors={str(k): str(v) for k, v in (remote.get("colors") or {}).items()},
            css=client.launcher_css() or _local_css(),
        )
    return ActiveTheme(
        popup=PopupSpec.from_dict(_local_popup_spec()),
        css=_local_css(),
    )


# Structural CSS the popup always needs, on top of whatever the theme provides.
# Kept minimal: everything visual is the theme's business.
BASE_CSS = """
window.keylane-popup {
  background-color: transparent;
}
window.keylane-popup.decorated .keylane-shell {
  border-radius: 0;
}
.keylane-shell {
  transition: opacity 120ms ease-out;
}
.keylane-result-view {
  background-color: transparent;
}
.keylane-step {
  font-size: 12px;
}
.keylane-busy-dot {
  min-width: 8px;
  min-height: 8px;
  border-radius: 999px;
}
"""

_providers: dict[str, Gtk.CssProvider] = {}


def apply_css(name: str, css: str, priority: int) -> None:
    """Install (or replace) a named CSS provider on the default display."""
    display = Gdk.Display.get_default()
    if display is None:
        return
    existing = _providers.pop(name, None)
    if existing is not None:
        Gtk.StyleContext.remove_provider_for_display(display, existing)
    if not css.strip():
        return
    provider = Gtk.CssProvider()
    try:
        provider.load_from_data(css.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Theme CSS rejected by GTK (%s); skipping.", exc)
        return
    Gtk.StyleContext.add_provider_for_display(display, provider, priority)
    _providers[name] = provider


def apply_theme(theme: ActiveTheme) -> None:
    apply_css("base", BASE_CSS, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 5)
    apply_css("theme", theme.css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 10)


def install_icon_search_path() -> None:
    icons_root = ROOT / "assets" / "icons"
    display = Gdk.Display.get_default()
    if display is None or not icons_root.exists():
        return
    Gtk.IconTheme.get_for_display(display).add_search_path(str(icons_root))


def logo_path() -> Path | None:
    for candidate in (
        ROOT / "launcher" / "assets" / "logo.png",
        ROOT / "assets" / "logo.png",
        ROOT / "assets" / "keylane-logo.png",
        ROOT / "assets" / "icons" / "hicolor" / "256x256" / "apps" / "keylane.png",
    ):
        if candidate.exists():
            return candidate
    return None
