"""Load gateway configuration from TOML and environment."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent


class GatewaySettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9100
    max_retries: int = 3
    local_only: bool = False
    result_corner: str = "top-right"
    """Where the working orb and result panel appear.

    top-right | top-left | bottom-right | bottom-left | center
    """

    docs_url: str = "/docs"
    """Where the control panel's Docs button points.

    Defaults to the handbook served by this gateway. Set it to your docs
    subdomain (``https://docs.example.com``) to send people there instead.
    """


class NpuSettings(BaseModel):
    model_path: str = "./models/router"
    device: str = "NPU"
    fallback_device: str = "CPU"


class LmStudioSettings(BaseModel):
    base_url: str = "http://127.0.0.1:1234/v1"
    default_model: str = "local-model"
    timeout_seconds: int = 120


class LemonadeSettings(BaseModel):
    """Lemonade Server (OpenAI-compatible LLM), not the clipboard lemond utility."""

    base_url: str = "http://127.0.0.1:13305/api/v1"
    default_model: str = "auto"
    timeout_seconds: int = 120


class ComfyUiSettings(BaseModel):
    base_url: str = "http://127.0.0.1:8188"
    timeout_seconds: int = 600
    output_dir: str = "./outputs"
    default_model: str = "auto"


class ClaudeSettings(BaseModel):
    command: str = "claude"
    timeout_seconds: int = 600


class CursorSettings(BaseModel):
    command: str = "cursor-agent"
    timeout_seconds: int = 600


class SecuritySettings(BaseModel):
    allowed_project_roots: list[str] = Field(default_factory=list)
    require_confirmation_for_modifications: bool = True


class AudioSettings(BaseModel):
    sample_rate: int = 16000
    channels: int = 1


class ProjectEntry(BaseModel):
    name: str
    path: str


class AppConfig(BaseModel):
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    npu: NpuSettings = Field(default_factory=NpuSettings)
    lmstudio: LmStudioSettings = Field(default_factory=LmStudioSettings)
    lemonade: LemonadeSettings = Field(default_factory=LemonadeSettings)
    comfyui: ComfyUiSettings = Field(default_factory=ComfyUiSettings)
    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    cursor: CursorSettings = Field(default_factory=CursorSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    projects: list[ProjectEntry] = Field(default_factory=list)
    root: Path = ROOT

    def resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        return p.resolve()

    @property
    def npu_model_path(self) -> Path:
        return self.resolve_path(self.npu.model_path)

    @property
    def workflows_dir(self) -> Path:
        return self.root / "workflows"

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.comfyui.output_dir)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _expand_home(value: str) -> str:
    return os.path.expanduser(value)


@lru_cache
def get_config() -> AppConfig:
    workers_path = ROOT / "config" / "workers.toml"
    projects_path = ROOT / "config" / "projects.toml"

    raw = _load_toml(workers_path)
    projects_raw = _load_toml(projects_path)

    security = raw.get("security", {})
    roots = [_expand_home(r) for r in security.get("allowed_project_roots", [])]
    security = {**security, "allowed_project_roots": roots}

    projects = []
    for entry in projects_raw.get("projects", []):
        projects.append(
            ProjectEntry(name=entry["name"], path=_expand_home(entry["path"]))
        )

    # Prefer PATH-resolved cursor-agent if present, else known install location.
    cursor_cmd = raw.get("cursor", {}).get("command", "cursor-agent")
    if cursor_cmd == "cursor-agent":
        known = Path.home() / ".local/share/cursor-agent/versions"
        if known.exists():
            versions = sorted(known.iterdir(), reverse=True)
            for version_dir in versions:
                binary = version_dir / "cursor-agent"
                if binary.is_file():
                    cursor_cmd = str(binary)
                    break

    cursor_cfg = {**raw.get("cursor", {}), "command": cursor_cmd}

    return AppConfig(
        gateway=GatewaySettings(**raw.get("gateway", {})),
        npu=NpuSettings(**raw.get("npu", {})),
        lmstudio=LmStudioSettings(**raw.get("lmstudio", {})),
        lemonade=LemonadeSettings(**raw.get("lemonade", {})),
        comfyui=ComfyUiSettings(**raw.get("comfyui", {})),
        claude=ClaudeSettings(**raw.get("claude", {})),
        cursor=CursorSettings(**cursor_cfg),
        security=SecuritySettings(**security),
        audio=AudioSettings(**raw.get("audio", {})),
        projects=projects,
        root=ROOT,
    )


def reload_config() -> AppConfig:
    get_config.cache_clear()
    return get_config()
