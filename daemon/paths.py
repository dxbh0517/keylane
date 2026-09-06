"""Runtime paths for Keylane."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("KEYLANE_ROOT", Path(__file__).resolve().parents[1]))


def default_data_dir(root: Path) -> Path:
    """Where this install keeps its data, given where its code lives.

    `root / "data"` is right for a git checkout and wrong for a release, and
    the difference is not cosmetic. An install lays out

        ~/.local/share/keylane/
          releases/<tag>/     the code — never contains data/
          current -> …        what the units follow
          data/               memories, models, settings

    precisely so that replacing the code cannot touch the data. So inside a
    release, `root / "data"` names a directory that by design does not exist:
    `settings.json` is missing, the API token with it, and every authenticated
    call answers 403 — the models list simply comes back empty.

    That was invisible for as long as everything was launched by the systemd
    units, which set `KEYLANE_DATA` explicitly. Anything else — the UI started
    cold by `--toggle`, a script run by hand — got the wrong directory and a
    confusing 403. The layout is unambiguous, so it is read here rather than
    guessed at in nine launchers.
    """
    # Path.resolve() has already followed `current`, so a release always
    # presents as <base>/releases/<tag>.
    if root.parent.name == "releases":
        return root.parent.parent / "data"
    # Belt and braces for an unresolved symlink path.
    if root.name == "current":
        return root.parent / "data"
    return root / "data"


DATA = Path(os.environ.get("KEYLANE_DATA") or default_data_dir(ROOT))
MODELS_DIR = DATA / "models"
CACHE_DIR = DATA / "cache" / "openvino"
DB_PATH = DATA / "keylane.db"
USER_MD = DATA / "memory" / "USER.md"
MEMORY_MD = DATA / "memory" / "MEMORY.md"
SKILLS_DIR = DATA / "skills"
THEMES_DIR = DATA / "themes"
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
        THEMES_DIR,
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
