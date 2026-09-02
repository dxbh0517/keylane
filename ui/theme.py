"""Theme and colour-scheme plumbing for the Spotlight windows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gio", "2.0")
from gi.repository import Gdk, Gio, GLib, Gtk  # type: ignore[attr-defined]

from ui.themes import DEFAULT_THEME_ID, Theme, ThemeError, load_theme, render_css

logger = logging.getLogger(__name__)

_scheme_watchers: list[Callable[[bool], None]] = []
_theme_watchers: list[Callable[[], None]] = []
_settings_connected = False
_gnome_settings: Gio.Settings | None = None
_provider: Gtk.CssProvider | None = None
_theme: Theme | None = None


def _settings_path() -> Path:
    from daemon.paths import SETTINGS_PATH

    return SETTINGS_PATH


def user_theme_preference() -> str:
    """User override: system, light, or dark."""
    path = _settings_path()
    if not path.exists():
        return "system"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        theme = str(data.get("ui", {}).get("theme", "system")).lower()
        if theme in {"system", "light", "dark"}:
            return theme
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return "system"


def user_theme_id() -> str:
    """Which theme the user picked; the reference theme until they pick one."""
    path = _settings_path()
    if not path.exists():
        return DEFAULT_THEME_ID
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        theme_id = str(data.get("ui", {}).get("theme_id", "")).strip()
        return theme_id or DEFAULT_THEME_ID
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return DEFAULT_THEME_ID


def current_theme() -> Theme:
    """The active theme, falling back to the reference one if it will not load.

    A broken theme must not leave the user with an unstyled window, so a
    failure here is logged and the default takes over.
    """
    global _theme
    if _theme is not None:
        return _theme
    theme_id = user_theme_id()
    try:
        _theme = load_theme(theme_id)
    except (ThemeError, OSError) as exc:
        if theme_id != DEFAULT_THEME_ID:
            logger.warning("theme %s did not load (%s); using %s", theme_id, exc, DEFAULT_THEME_ID)
        _theme = load_theme(DEFAULT_THEME_ID)
    return _theme


def theme_tokens(dark: bool | None = None) -> dict[str, str]:
    """The active theme's tokens for one scheme — for what CSS cannot reach."""
    return current_theme().scheme(effective_prefers_dark() if dark is None else dark)


def token_rgb(name: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    """A token as 0-1 RGB, for widgets that paint themselves (the orb)."""
    value = theme_tokens().get(name, "").strip().lstrip("#")
    if len(value) == 6:
        try:
            return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            pass
    return fallback


def _gnome_color_scheme() -> str | None:
    global _gnome_settings
    try:
        if _gnome_settings is None:
            _gnome_settings = Gio.Settings.new("org.gnome.desktop.interface")
        return _gnome_settings.get_string("color-scheme")
    except (GLib.Error, AttributeError, TypeError, ValueError):
        return None


def system_prefers_dark() -> bool:
    scheme = _gnome_color_scheme()
    if scheme == "prefer-dark":
        return True
    if scheme == "prefer-light":
        return False

    settings = Gtk.Settings.get_default()
    try:
        gtk_scheme = settings.get_property("gtk-color-scheme")
        if gtk_scheme == Gdk.SystemColorScheme.DARK:
            return True
        if gtk_scheme == Gdk.SystemColorScheme.LIGHT:
            return False
    except (TypeError, AttributeError, ValueError):
        pass

    theme = str(settings.get_property("gtk-theme-name") or "").lower()
    if theme.endswith("-dark") or "dark" in theme:
        return True
    return bool(settings.get_property("gtk-application-prefer-dark-theme"))


def effective_prefers_dark() -> bool:
    pref = user_theme_preference()
    if pref == "light":
        return False
    if pref == "dark":
        return True
    return system_prefers_dark()


def apply_scheme_classes(widget: Gtk.Widget) -> None:
    dark = effective_prefers_dark()
    widget.remove_css_class("style-light")
    widget.remove_css_class("style-dark")
    widget.add_css_class("style-dark" if dark else "style-light")


def _notify_watchers() -> None:
    dark = effective_prefers_dark()
    for watcher in list(_scheme_watchers):
        watcher(dark)


def _on_settings_changed(_settings: Gtk.Settings, pspec) -> None:
    if pspec.name in {"gtk-color-scheme", "gtk-application-prefer-dark-theme"}:
        _notify_watchers()


def _on_gnome_scheme_changed(_settings: Gio.Settings, _key: str) -> None:
    _notify_watchers()


def watch_color_scheme(callback: Callable[[bool], None]) -> None:
    global _settings_connected
    _scheme_watchers.append(callback)
    if not _settings_connected:
        settings = Gtk.Settings.get_default()
        settings.connect("notify::gtk-color-scheme", _on_settings_changed)
        settings.connect("notify::gtk-application-prefer-dark-theme", _on_settings_changed)
        settings.connect("notify::gtk-theme-name", _on_settings_changed)
        try:
            gnome = Gio.Settings.new("org.gnome.desktop.interface")
            gnome.connect("changed::color-scheme", _on_gnome_scheme_changed)
        except (GLib.Error, AttributeError, TypeError, ValueError):
            pass
        _settings_connected = True
    callback(effective_prefers_dark())


def _stylesheet() -> str:
    try:
        return render_css(current_theme())
    except ThemeError:
        logger.exception("rendering theme %s failed; using %s", current_theme().id, DEFAULT_THEME_ID)
        return render_css(load_theme(DEFAULT_THEME_ID))


def apply_spotlight_theme(display: Gdk.Display | None = None) -> None:
    global _provider
    display = display or Gdk.Display.get_default()
    if not display:
        return
    if _provider is None:
        _provider = Gtk.CssProvider()
        Gtk.StyleContext.add_provider_for_display(
            display,
            _provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
    _provider.load_from_string(_stylesheet())


def reload_spotlight_theme() -> None:
    """Re-render after the user picks a theme — every open window restyles."""
    global _theme
    _theme = None
    if _provider is not None:
        _provider.load_from_string(_stylesheet())
    else:
        apply_spotlight_theme()
    for watcher in list(_theme_watchers):
        watcher()


def watch_theme(callback: Callable[[], None]) -> None:
    """Called after a theme change, for colours CSS cannot deliver."""
    _theme_watchers.append(callback)
