"""Runtime paths for Keylane."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("KEYLANE_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("KEYLANE_DATA", ROOT / "data"))
MODELS_DIR = DATA / "models"
CACHE_DIR = DATA / "cache" / "openvino"
DB_PATH = DATA / "keylane.db"
USER_MD = DATA / "memory" / "USER.md"
MEMORY_MD = DATA / "memory" / "MEMORY.md"
SKILLS_DIR = DATA / "skills"
TODOS_PATH = DATA / "todos.json"
TTS_MODEL_DIR = MODELS_DIR / "tts" / "Audio8-TTS-Preview-0.1b"
VOICES_DIR = ROOT / "voices"
CONFIG_DIR = ROOT / "config"
SETTINGS_PATH = DATA / "settings.json"


def ensure_data_dirs() -> None:
    for path in (
        DATA,
        MODELS_DIR,
        CACHE_DIR,
        DATA / "memory",
        SKILLS_DIR,
        TTS_MODEL_DIR.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if not USER_MD.exists():
        USER_MD.write_text(
            "# User profile\n\nPreferences and communication style go here.\n",
            encoding="utf-8",
        )
    if not MEMORY_MD.exists():
        MEMORY_MD.write_text(
            "# Agent memory\n\nFacts learned about projects, environment, and habits.\n",
            encoding="utf-8",
        )
