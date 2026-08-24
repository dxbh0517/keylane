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
    docs_url: str | None = None
    result_corner: str | None = None
    allowed_project_roots: list[str] | None = None
    require_confirmation_for_modifications: bool | None = None


class GatewaySettingsView(BaseModel):
    host: str
    port: int
    max_retries: int
    local_only: bool
    docs_url: str
    result_corner: str
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
        docs_url=cfg.gateway.docs_url,
        result_corner=cfg.gateway.result_corner,
        allowed_project_roots=list(cfg.security.allowed_project_roots),
        require_confirmation_for_modifications=cfg.security.require_confirmation_for_modifications,
        restart_required=False,
        note="Changing host/port requires restarting ai-gateway.service.",
    )


# Which TOML table each editable key belongs to. Writes are scoped to the
# section so a new key is appended in the right place rather than at EOF, where
# it would silently join whichever table happens to be last.
KEY_SECTIONS = {
    "host": "gateway",
    "port": "gateway",
    "max_retries": "gateway",
    "local_only": "gateway",
    "docs_url": "gateway",
    "result_corner": "gateway",
    "require_confirmation_for_modifications": "security",
    "allowed_project_roots": "security",
}


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _section_bounds(text: str, section: str) -> tuple[int, int] | None:
    """Character range of a TOML table's body, excluding its header line."""
    header = re.search(rf"(?m)^\[{re.escape(section)}\]\s*$", text)
    if header is None:
        return None
    start = header.end()
    following = re.search(r"(?m)^\[", text[start:])
    end = start + following.start() if following else len(text)
    return start, end


def set_toml_key(text: str, section: str, key: str, rendered: str) -> str:
    """Set ``key = rendered`` inside ``[section]``, creating either if needed."""
    bounds = _section_bounds(text, section)
    if bounds is None:
        separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        return f"{text}{separator}[{section}]\n{key} = {rendered}\n"

    start, end = bounds
    body = text[start:end]
    # Match a scalar assignment or a multi-line array for this key.
    pattern = rf"(?ms)^{re.escape(key)}\s*=\s*(?:\[.*?\]|[^\n]*)$"
    if re.search(pattern, body):
        body = re.sub(pattern, f"{key} = {rendered}", body, count=1)
    else:
        body = body.rstrip("\n") + f"\n{key} = {rendered}\n"
        if not body.startswith("\n"):
            body = "\n" + body.lstrip("\n")
    return text[:start] + body + text[end:]


def update_gateway_settings(update: GatewaySettingsUpdate) -> GatewaySettingsView:
    path = ROOT / "config" / "workers.toml"
    text = path.read_text(encoding="utf-8")
    restart = False

    data = update.model_dump(exclude_none=True)
    roots = data.pop("allowed_project_roots", None)

    for key, value in data.items():
        if key in {"host", "port"}:
            restart = True
        section = KEY_SECTIONS.get(key, "gateway")
        text = set_toml_key(text, section, key, _toml_scalar(value))

    if roots is not None:
        rendered = "[\n" + "".join(f'  "{r}",\n' for r in roots) + "]"
        text = set_toml_key(text, "security", "allowed_project_roots", rendered)

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
