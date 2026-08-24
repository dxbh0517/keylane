"""Read and write the worker endpoint settings in ``config/workers.toml``.

These are the connection details the control panel used to be unable to reach:
LM Studio's URL, ComfyUI's output directory, the Claude and Cursor commands and
their timeouts, and the microphone capture format.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import ROOT, get_config, reload_config
from app.settings_store import _toml_scalar, set_toml_key

WORKERS_TOML = ROOT / "config" / "workers.toml"


class WorkerEndpoints(BaseModel):
    lmstudio_base_url: str = ""
    lmstudio_default_model: str = ""
    lmstudio_timeout_seconds: int = 120

    lemonade_base_url: str = ""
    lemonade_default_model: str = ""
    lemonade_timeout_seconds: int = 120

    comfyui_base_url: str = ""
    comfyui_output_dir: str = ""
    comfyui_timeout_seconds: int = 600

    claude_command: str = ""
    claude_timeout_seconds: int = 600

    cursor_command: str = ""
    cursor_timeout_seconds: int = 600

    audio_sample_rate: int = 16000
    audio_channels: int = 1


class WorkerEndpointsUpdate(BaseModel):
    lmstudio_base_url: str | None = None
    lmstudio_default_model: str | None = None
    lmstudio_timeout_seconds: int | None = Field(default=None, ge=5, le=3600)

    lemonade_base_url: str | None = None
    lemonade_default_model: str | None = None
    lemonade_timeout_seconds: int | None = Field(default=None, ge=5, le=3600)

    comfyui_base_url: str | None = None
    comfyui_output_dir: str | None = None
    comfyui_timeout_seconds: int | None = Field(default=None, ge=5, le=7200)

    claude_command: str | None = None
    claude_timeout_seconds: int | None = Field(default=None, ge=5, le=7200)

    cursor_command: str | None = None
    cursor_timeout_seconds: int | None = Field(default=None, ge=5, le=7200)

    audio_sample_rate: int | None = Field(default=None, ge=8000, le=48000)
    audio_channels: int | None = Field(default=None, ge=1, le=2)


# field name -> (toml section, toml key)
FIELD_MAP: dict[str, tuple[str, str]] = {
    "lmstudio_base_url": ("lmstudio", "base_url"),
    "lmstudio_default_model": ("lmstudio", "default_model"),
    "lmstudio_timeout_seconds": ("lmstudio", "timeout_seconds"),
    "lemonade_base_url": ("lemonade", "base_url"),
    "lemonade_default_model": ("lemonade", "default_model"),
    "lemonade_timeout_seconds": ("lemonade", "timeout_seconds"),
    "comfyui_base_url": ("comfyui", "base_url"),
    "comfyui_output_dir": ("comfyui", "output_dir"),
    "comfyui_timeout_seconds": ("comfyui", "timeout_seconds"),
    "claude_command": ("claude", "command"),
    "claude_timeout_seconds": ("claude", "timeout_seconds"),
    "cursor_command": ("cursor", "command"),
    "cursor_timeout_seconds": ("cursor", "timeout_seconds"),
    "audio_sample_rate": ("audio", "sample_rate"),
    "audio_channels": ("audio", "channels"),
}


def current_worker_endpoints() -> WorkerEndpoints:
    cfg = get_config()
    return WorkerEndpoints(
        lmstudio_base_url=cfg.lmstudio.base_url,
        lmstudio_default_model=cfg.lmstudio.default_model,
        lmstudio_timeout_seconds=cfg.lmstudio.timeout_seconds,
        lemonade_base_url=cfg.lemonade.base_url,
        lemonade_default_model=cfg.lemonade.default_model,
        lemonade_timeout_seconds=cfg.lemonade.timeout_seconds,
        comfyui_base_url=cfg.comfyui.base_url,
        comfyui_output_dir=cfg.comfyui.output_dir,
        comfyui_timeout_seconds=cfg.comfyui.timeout_seconds,
        claude_command=cfg.claude.command,
        claude_timeout_seconds=cfg.claude.timeout_seconds,
        cursor_command=cfg.cursor.command,
        cursor_timeout_seconds=cfg.cursor.timeout_seconds,
        audio_sample_rate=cfg.audio.sample_rate,
        audio_channels=cfg.audio.channels,
    )


def update_worker_endpoints(update: WorkerEndpointsUpdate) -> dict[str, Any]:
    text = WORKERS_TOML.read_text(encoding="utf-8")
    changed: list[str] = []

    for field, value in update.model_dump(exclude_none=True).items():
        section, key = FIELD_MAP[field]
        text = set_toml_key(text, section, key, _toml_scalar(value))
        changed.append(f"{section}.{key}")

    WORKERS_TOML.write_text(text, encoding="utf-8")
    reload_config()

    return {
        "settings": current_worker_endpoints().model_dump(),
        "changed": changed,
        "note": (
            "Saved. Plugins pick these up on the next request; restart the "
            "service if a worker keeps using the old endpoint."
            if changed
            else "Nothing to change."
        ),
    }
