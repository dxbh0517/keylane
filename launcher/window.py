"""Backwards-compatible shim.

The launcher used to live here as a single decorated window. It is now split
into :mod:`launcher.popup` (the themed Spotlight overlay), :mod:`launcher.tray`
(the taskbar indicator) and :mod:`launcher.main` (the entry point). This module
re-exports the old names so existing service files and scripts keep working.
"""

from __future__ import annotations

from launcher.main import run_launcher
from launcher.popup import KeylanePopup

# Historic alias.
LauncherWindow = KeylanePopup

__all__ = ["KeylanePopup", "LauncherWindow", "run_launcher"]
