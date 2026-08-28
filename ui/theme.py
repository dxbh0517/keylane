"""System color scheme sync for Spotlight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gio", "2.0")
from gi.repository import Gdk, Gio, GLib, Gtk  # type: ignore[attr-defined]

_CSS_PATH = Path(__file__).resolve().parent / "spotlight.css"

_scheme_watchers: list[Callable[[bool], None]] = []
_settings_connected = False
_gnome_settings: Gio.Settings | None = None


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


def apply_spotlight_theme(display: Gdk.Display | None = None) -> None:
    display = display or Gdk.Display.get_default()
    if not display:
        return
    provider = Gtk.CssProvider()
    provider.load_from_path(str(_CSS_PATH))
    Gtk.StyleContext.add_provider_for_display(
        display,
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )
