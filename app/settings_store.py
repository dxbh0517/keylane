"""Persist gateway settings (port, local-only, retries) to workers.toml."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import ROOT, get_config, reload_config


class GatewaySettingsUpdate(BaseModel):
    host: str | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)
    max_retries: int | None = Field(default=None, ge=0, le=10)
    local_only: bool | None = None
    allowed_project_roots: list[str] | None = None
    require_confirmation_for_modifications: bool | None = None


class GatewaySettingsView(BaseModel):
    host: str
    port: int
    max_retries: int
    local_only: bool
    allowed_project_roots: list[str]
    require_confirmation_for_modifications: bool
    restart_required: bool = False
    note: str = ""


def current_gateway_settings() -> GatewaySettingsView:
    cfg = get_config()
    return GatewaySettingsView(
        host=cfg.gateway.host,
        port=cfg.gateway.port,
        max_retries=cfg.gateway.max_retries,
        local_only=cfg.gateway.local_only,
        allowed_project_roots=list(cfg.security.allowed_project_roots),
        require_confirmation_for_modifications=cfg.security.require_confirmation_for_modifications,
        restart_required=False,
        note="Changing host/port requires restarting ai-gateway.service.",
    )


def update_gateway_settings(update: GatewaySettingsUpdate) -> GatewaySettingsView:
    path = ROOT / "config" / "workers.toml"
    text = path.read_text(encoding="utf-8")
    restart = False

    def replace_bool(section_key: str, key: str, value: bool) -> str:
        nonlocal text
        pattern = rf"(?m)^({key}\s*=\s*)(true|false)\s*$"
        # Only within rough whole-file replace is fine for our flat sections
        text, n = re.subn(pattern, rf"\g<1>{'true' if value else 'false'}", text, count=1)
        if n == 0:
            text += f"\n{key} = {'true' if value else 'false'}\n"
        return text

    def replace_scalar(key: str, value: Any) -> None:
        nonlocal text, restart
        if key in {"host", "port"}:
            restart = True
        if isinstance(value, bool):
            replace_bool("", key, value)
            return
        if isinstance(value, int):
            pattern = rf"(?m)^({key}\s*=\s*)\d+\s*$"
            text2, n = re.subn(pattern, rf"\g<1>{value}", text, count=1)
            text = text2 if n else text + f"\n{key} = {value}\n"
            return
        if isinstance(value, str):
            pattern = rf'(?m)^({key}\s*=\s*)"[^"]*"\s*$'
            text2, n = re.subn(pattern, rf'\g<1>"{value}"', text, count=1)
            text = text2 if n else text + f'\n{key} = "{value}"\n'
            return

    data = update.model_dump(exclude_none=True)
    roots = data.pop("allowed_project_roots", None)
    for key, value in data.items():
        replace_scalar(key, value)

    if roots is not None:
        # Rewrite the allowed_project_roots array block
        block = "allowed_project_roots = [\n" + "".join(
            f'  "{r}",\n' for r in roots
        ) + "]"
        pattern = r"(?ms)^allowed_project_roots\s*=\s*\[.*?\]"
        if re.search(pattern, text):
            text = re.sub(pattern, block, text, count=1)
        else:
            text += "\n" + block + "\n"

    path.write_text(text, encoding="utf-8")
    reload_config()
    view = current_gateway_settings()
    view.restart_required = restart
    if restart:
        view.note = (
            f"Saved. Restart the service to bind {view.host}:{view.port}: "
            "systemctl --user restart ai-gateway.service"
        )
    return view
