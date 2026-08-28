"""Load TOML configuration with user overrides from data/settings.json."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from daemon.paths import CONFIG_DIR, SETTINGS_PATH, ensure_data_dirs

_lock = threading.RLock()

# Sections users may override via PATCH /settings
ALLOWED_SECTIONS = frozenset(
    {"assistant", "notify", "speech", "security", "research", "permissions", "mcp", "ui"}
)


def load_toml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_overrides() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_overrides(data: dict[str, Any]) -> None:
    ensure_data_dirs()
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _defaults() -> dict[str, Any]:
    assistant = load_toml("assistant.toml")
    research_raw = load_toml("research.toml")
    return {
        "assistant": assistant.get("assistant", {}),
        "notify": assistant.get("notify", {}),
        "speech": assistant.get("speech", {}),
        "security": assistant.get("security", {}),
        "research": {
            **research_raw.get("research", {}),
            "search_backend": research_raw.get("research", {}).get("search_backend", "searxng"),
            "extract_backend": research_raw.get("research", {}).get("extract_backend", "local"),
            "keyless_fallback": research_raw.get("research", {}).get("keyless_fallback", True),
            "cache_ttl_minutes": research_raw.get("research", {}).get("cache_ttl_minutes", 15),
            "freshness": research_raw.get("research", {}).get("freshness", "month"),
            "language": research_raw.get("research", {}).get("language", "en"),
        },
        "permissions": {
            "shell": "ask",
            "memory_write": "ask",
            "schedule_task": "ask",
            "run_background": "auto",
            "default": "auto",
        },
        "mcp": {"disabled_tools": [], "servers": []},
        "ui": {"theme": "system"},
    }


def all_settings() -> dict[str, Any]:
    """Merged defaults + user overrides."""
    with _lock:
        defaults = _defaults()
        overrides = _load_overrides()
        merged = _deep_merge(defaults, overrides)
        merged["mcp_servers"] = list_mcp_servers()
        return merged


def get_section(section: str) -> dict[str, Any]:
    return dict(all_settings().get(section, {}))


def save_settings(section: str, patch: dict[str, Any]) -> dict[str, Any]:
    if section not in ALLOWED_SECTIONS:
        raise ValueError(f"unknown settings section: {section}")
    with _lock:
        overrides = _load_overrides()
        current = overrides.get(section, {})
        if not isinstance(current, dict):
            current = {}
        overrides[section] = _deep_merge(current, patch)
        _save_overrides(overrides)
        return all_settings()


def reset_settings(section: str | None = None) -> dict[str, Any]:
    with _lock:
        overrides = _load_overrides()
        if section is None:
            overrides = {}
        elif section in overrides:
            del overrides[section]
        else:
            raise ValueError(f"unknown settings section: {section}")
        _save_overrides(overrides)
        return all_settings()


def assistant_settings() -> dict[str, Any]:
    s = all_settings()
    return {
        "assistant": s.get("assistant", {}),
        "notify": s.get("notify", {}),
        "speech": s.get("speech", {}),
        "security": s.get("security", {}),
    }


def research_settings() -> dict[str, Any]:
    return get_section("research")


def _config_mcp_servers() -> list[dict[str, Any]]:
    return [dict(s, source="config") for s in load_toml("mcp.toml").get("servers", [])]


def list_mcp_servers() -> list[dict[str, Any]]:
    """Merge config/mcp.toml servers with user-added servers from settings.json."""
    by_id: dict[str, dict[str, Any]] = {}
    for srv in _config_mcp_servers():
        sid = str(srv.get("id", ""))
        if sid:
            by_id[sid] = srv
    overrides = _load_overrides()
    mcp = overrides.get("mcp", {})
    user_servers = mcp.get("servers", []) if isinstance(mcp, dict) else []
    for srv in user_servers:
        sid = str(srv.get("id", ""))
        if sid:
            by_id[sid] = {**srv, "source": "user"}
    return list(by_id.values())


def add_mcp_server(server: dict[str, Any]) -> list[dict[str, Any]]:
    sid = str(server.get("id", "")).strip()
    command = str(server.get("command", "")).strip()
    if not sid or not command:
        raise ValueError("id and command are required")
    entry = {
        "id": sid,
        "transport": str(server.get("transport", "stdio")),
        "command": command,
        "args": list(server.get("args", [])),
    }
    if server.get("env"):
        entry["env"] = dict(server["env"])
    with _lock:
        overrides = _load_overrides()
        mcp = overrides.get("mcp", {})
        if not isinstance(mcp, dict):
            mcp = {}
        servers = list(mcp.get("servers", []))
        servers = [s for s in servers if str(s.get("id")) != sid]
        servers.append(entry)
        mcp["servers"] = servers
        overrides["mcp"] = mcp
        _save_overrides(overrides)
    return list_mcp_servers()


def remove_mcp_server(server_id: str) -> list[dict[str, Any]]:
    with _lock:
        overrides = _load_overrides()
        mcp = overrides.get("mcp", {})
        if not isinstance(mcp, dict):
            mcp = {}
        servers = [s for s in mcp.get("servers", []) if str(s.get("id")) != server_id]
        mcp["servers"] = servers
        overrides["mcp"] = mcp
        _save_overrides(overrides)
    return list_mcp_servers()


def mcp_settings() -> dict[str, Any]:
    cfg = load_toml("mcp.toml")
    user = get_section("mcp")
    merged = {**cfg, **user}
    merged["servers"] = list_mcp_servers()
    return merged


def permission_mode(tool_name: str) -> str:
    perms = get_section("permissions")
    if tool_name in perms:
        return str(perms[tool_name])
    return str(perms.get("default", "auto"))
