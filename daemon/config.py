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
    {
        "assistant", "notify", "speech", "security", "research",
        "permissions", "mcp", "ui", "models", "updates",
    }
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
    invalidate_cache()


# ── the merged-settings cache ────────────────────────────────────────────
#
# `all_settings()` re-parses three TOML files and the JSON overrides, and
# `get_section()` goes through it. `ModelEntry.resolve_device()` then calls
# `get_section("models")` once per model, and `get_model()` calls
# `load_catalog()`, which starts the whole thing again — so one `GET /models`
# used to re-read the configuration dozens of times.
#
# The cache is keyed on the mtimes of the files that feed it, so an edit made
# outside the process is still picked up, and writes through this module drop
# it explicitly.

_CACHE_KEY: tuple[Any, ...] | None = None
_CACHE_VALUE: dict[str, Any] | None = None

_CONFIG_FILES = ("assistant.toml", "research.toml", "models.toml", "mcp.toml")


def _mark(path: Path) -> tuple[Any, ...]:
    """The path plus what would have to change for its contents to differ.

    The path is part of the key because a test that points SETTINGS_PATH at a
    fresh tmp_path leaves a file that does not exist yet — and "missing" alone
    is the same mark for every such test, which would let one test read the
    previous one's cached settings.
    """
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), None, None)


def _stamp() -> tuple[Any, ...]:
    """What would have to change for the merged settings to differ."""
    marks = [_mark(CONFIG_DIR / name) for name in _CONFIG_FILES]
    marks.append(_mark(SETTINGS_PATH))
    return tuple(marks)


def invalidate_cache() -> None:
    """Drop the cached merge. Called on every write through this module."""
    global _CACHE_KEY, _CACHE_VALUE
    with _lock:
        _CACHE_KEY = None
        _CACHE_VALUE = None


def _defaults() -> dict[str, Any]:
    assistant = load_toml("assistant.toml")
    research_raw = load_toml("research.toml")
    models_raw = load_toml("models.toml")
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
        # Not a tool — no model can reach it. Set to "deny" to turn off
        # in-app updating entirely.
        "update_apply": "ask",
            "watch_create": "ask",
            "remember": "auto",
            "remind_me": "auto",
            "run_background": "auto",
            "default": "auto",
        },
        "mcp": {"disabled_tools": [], "servers": []},
        "ui": {"theme": "system", "theme_id": "glass-console"},
        "updates": {
            # "stable" follows published releases; "main" follows the branch.
            "channel": "stable",
            # Look once a day and put a note in the inbox. Never install on
            # its own — replacing the running code is the user's decision.
            "check_daily": True,
        },
        "models": {
            "default": models_raw.get("default", ""),
            # Which inference stack Settings browses models for.
            "runtime": models_raw.get("runtime", "openvino"),
            "device": models_raw.get("device", "NPU"),
            # Per-runtime device choice; the runtimes do not offer the same ones.
            "devices": models_raw.get("devices", {}),
            "routes": models_raw.get("routes", {}),
            "adapters": models_raw.get("adapters", []),
            # Models the user added from Hugging Face, appended to the curated list.
            "imported": models_raw.get("imported", []),
        },
    }


def all_settings() -> dict[str, Any]:
    """Merged defaults + user overrides.

    The result is cached against the mtimes of the files behind it, because
    this is on the path of every settings read in the process and re-parsing
    four TOML files per model row is not free.
    """
    global _CACHE_KEY, _CACHE_VALUE
    with _lock:
        key = _stamp()
        if _CACHE_KEY == key and _CACHE_VALUE is not None:
            return deepcopy(_CACHE_VALUE)
        defaults = _defaults()
        overrides = _load_overrides()
        merged = _deep_merge(defaults, overrides)
        merged["mcp_servers"] = list_mcp_servers()
        _CACHE_KEY = key
        _CACHE_VALUE = merged
        # A copy, so a caller that mutates what it got cannot poison the cache.
        return deepcopy(merged)


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


def model_settings() -> dict[str, Any]:
    """Model routes and adapters, merged over config/models.toml."""
    return get_section("models")


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
    """Persist a user MCP server. stdio needs a command, http needs a url."""
    sid = str(server.get("id", "")).strip()
    command = str(server.get("command", "")).strip()
    url = str(server.get("url", "")).strip()
    transport = str(server.get("transport", "")).strip().lower()
    if not transport:
        transport = "http" if url else "stdio"
    if not sid:
        raise ValueError("id is required")

    entry: dict[str, Any] = {"id": sid, "transport": transport}
    if transport in {"http", "streamable-http", "sse"}:
        if not url:
            raise ValueError("url is required for http transport")
        entry["url"] = url
        if server.get("auth_header"):
            entry["auth_header"] = str(server["auth_header"])
        if server.get("headers"):
            entry["headers"] = {str(k): str(v) for k, v in server["headers"].items()}
    else:
        if not command:
            raise ValueError("command is required for stdio transport")
        entry["command"] = command
        entry["args"] = list(server.get("args", []))
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
