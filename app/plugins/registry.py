"""Plugin registry — discover, enable/disable, configure, persist."""

from __future__ import annotations

import json
import logging
import shutil
import tomllib
from pathlib import Path
from typing import Any

from app.config import ROOT, AppConfig, get_config, reload_config
from app.plugins.base import BasePlugin, PluginHealth, PluginInfo, PluginKind
from app.plugins.builtin import create_builtin_plugins
from app.plugins.mcp_plugin import mcp_plugin_from_manifest
from app.schemas import RouteDecision, WorkerResult

logger = logging.getLogger(__name__)

try:
    import tomli_w
except ImportError:  # pragma: no cover
    tomli_w = None  # type: ignore


DEFAULT_ENABLED = {
    "lmstudio": True,
    "claude": True,
    "cursor": True,
    "comfyui": True,
    "comfyui-http": False,
    "lemonade": True,
}


class PluginRegistry:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.state_path = self.config.root / "config" / "plugins.toml"
        self.community_dir = self.config.root / "plugins" / "community"
        self.community_dir.mkdir(parents=True, exist_ok=True)
        self._plugins: dict[str, BasePlugin] = {}
        self._enabled: dict[str, bool] = dict(DEFAULT_ENABLED)
        self._settings: dict[str, dict[str, Any]] = {}
        self._load_state()
        self._discover()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self._write_state()
            return
        with self.state_path.open("rb") as fh:
            raw = tomllib.load(fh)
        plugins = raw.get("plugins", {})
        for pid, entry in plugins.items():
            if isinstance(entry, dict):
                self._enabled[pid] = bool(entry.get("enabled", True))
                settings = entry.get("settings") or {}
                if isinstance(settings, dict):
                    self._settings[pid] = settings

    def _write_state(self) -> None:
        lines = [
            "# Gateway plugin state — edited by the control panel or by hand.",
            "# See docs/PLUGINS.md for the plugin authoring guide.",
            "",
            "[plugins]",
            "",
        ]
        # Prefer tomli_w if available
        data: dict[str, Any] = {"plugins": {}}
        for pid, plugin in sorted(self._plugins.items()) or sorted(
            {**DEFAULT_ENABLED}.items()
        ):
            if isinstance(plugin, BasePlugin):
                data["plugins"][pid] = {
                    "enabled": self._enabled.get(pid, True),
                    "settings": plugin.settings,
                }
            else:
                data["plugins"][pid] = {
                    "enabled": self._enabled.get(pid, True),
                    "settings": self._settings.get(pid, {}),
                }

        if not self._plugins:
            for pid, enabled in DEFAULT_ENABLED.items():
                data["plugins"][pid] = {
                    "enabled": enabled,
                    "settings": self._settings.get(pid, {}),
                }

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if tomli_w is not None:
            with self.state_path.open("wb") as fh:
                tomli_w.dump(data, fh)
            return

        # Minimal TOML writer fallback
        chunks = ["# Gateway plugin state\n"]
        for pid, entry in data["plugins"].items():
            chunks.append(f"[plugins.{pid}]\n")
            chunks.append(f"enabled = {'true' if entry['enabled'] else 'false'}\n")
            settings = entry.get("settings") or {}
            if settings:
                chunks.append(f"[plugins.{pid}.settings]\n")
                for key, value in settings.items():
                    chunks.append(f"{key} = {_toml_literal(value)}\n")
            chunks.append("\n")
        self.state_path.write_text("".join(chunks), encoding="utf-8")

    def _discover(self) -> None:
        builtins = create_builtin_plugins(self.config)
        for pid, plugin in builtins.items():
            if pid in self._settings:
                plugin.update_settings(self._settings[pid])
            self._plugins[pid] = plugin
            self._enabled.setdefault(pid, DEFAULT_ENABLED.get(pid, True))

        for manifest in sorted(self.community_dir.glob("*/plugin.toml")):
            try:
                with manifest.open("rb") as fh:
                    data = tomllib.load(fh)
                pid = data.get("id") or manifest.parent.name
                kind = (data.get("kind") or "mcp").lower()
                settings = self._settings.get(pid, {})
                if kind == "mcp":
                    plugin = mcp_plugin_from_manifest(data, settings)
                else:
                    logger.warning("Skipping non-MCP community plugin %s (native community plugins need a Python entry).", pid)
                    continue
                self._plugins[pid] = plugin
                self._enabled.setdefault(pid, bool(data.get("enabled", True)))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to load community plugin %s: %s", manifest, exc)

    def reload(self) -> None:
        self._plugins.clear()
        self._discover()

    def list(self) -> list[PluginInfo]:
        return [
            plugin.info(enabled=self._enabled.get(pid, True))
            for pid, plugin in sorted(self._plugins.items())
        ]

    async def list_with_health(self) -> list[PluginInfo]:
        out: list[PluginInfo] = []
        for pid, plugin in sorted(self._plugins.items()):
            enabled = self._enabled.get(pid, True)
            health: PluginHealth | None = None
            if enabled:
                try:
                    health = await plugin.health()
                except Exception as exc:  # noqa: BLE001
                    health = PluginHealth(ok=False, detail=str(exc))
            out.append(plugin.info(enabled=enabled, health=health))
        return out

    def get(self, plugin_id: str) -> BasePlugin | None:
        return self._plugins.get(plugin_id)

    def is_enabled(self, plugin_id: str) -> bool:
        return bool(self._enabled.get(plugin_id, False))

    def set_enabled(self, plugin_id: str, enabled: bool) -> PluginInfo:
        if plugin_id not in self._plugins:
            raise KeyError(f"Unknown plugin: {plugin_id}")
        self._enabled[plugin_id] = enabled
        self._write_state()
        return self._plugins[plugin_id].info(enabled=enabled)

    def update_settings(self, plugin_id: str, data: dict[str, Any]) -> PluginInfo:
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            raise KeyError(f"Unknown plugin: {plugin_id}")
        settings = plugin.update_settings(data)
        self._settings[plugin_id] = settings
        self._write_state()
        return plugin.info(enabled=self.is_enabled(plugin_id))

    def install_mcp_manifest(self, data: dict[str, Any]) -> PluginInfo:
        pid = data.get("id")
        if not pid:
            raise ValueError("plugin.toml must include id")
        target = self.community_dir / pid
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "plugin.toml"
        # Write via simple toml
        lines = [f'id = "{pid}"\n', f'name = {_toml_literal(data.get("name") or pid)}\n']
        for key in ("description", "author", "version", "homepage", "kind", "worker_id", "command"):
            if key in data and data[key] is not None:
                lines.append(f"{key} = {_toml_literal(data[key])}\n")
        if "cloud" in data:
            lines.append(f"cloud = {'true' if data['cloud'] else 'false'}\n")
        if "args" in data:
            lines.append(f"args = {_toml_literal(data['args'])}\n")
        for key in ("health_tool", "run_tool"):
            if key in data:
                lines.append(f"{key} = {_toml_literal(data[key])}\n")
        if "env" in data:
            lines.append(f"env = {_toml_literal(data['env'])}\n")
        manifest_path.write_text("".join(lines), encoding="utf-8")
        plugin = mcp_plugin_from_manifest(data, self._settings.get(pid))
        self._plugins[pid] = plugin
        self._enabled[pid] = True
        self._write_state()
        return plugin.info(enabled=True)

    def uninstall(self, plugin_id: str) -> None:
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            raise KeyError(f"Unknown plugin: {plugin_id}")
        if not plugin.removable:
            raise ValueError(f"Plugin '{plugin_id}' is built-in and cannot be removed (disable it instead).")
        community = self.community_dir / plugin_id
        if community.exists():
            shutil.rmtree(community)
        self._plugins.pop(plugin_id, None)
        self._enabled.pop(plugin_id, None)
        self._settings.pop(plugin_id, None)
        self._write_state()

    def enabled_worker_ids(self, *, local_only: bool = False) -> set[str]:
        workers: set[str] = set()
        for pid, plugin in self._plugins.items():
            if not self._enabled.get(pid, False):
                continue
            if plugin.worker_id is None:
                continue
            if local_only and plugin.cloud:
                continue
            workers.add(plugin.worker_id)
        return workers

    async def available_workers(self, *, local_only: bool = False) -> set[str]:
        available: set[str] = set()
        for pid, plugin in self._plugins.items():
            if not self._enabled.get(pid, False):
                continue
            if plugin.worker_id is None:
                continue
            if local_only and plugin.cloud:
                continue
            try:
                health = await plugin.health()
                if health.ok:
                    available.add(plugin.worker_id)
            except Exception:  # noqa: BLE001
                continue
        return available

    async def run_worker(self, decision: RouteDecision) -> WorkerResult:
        worker = decision.worker
        # Prefer enabled plugin that owns this worker_id
        candidates = [
            (pid, p)
            for pid, p in self._plugins.items()
            if self._enabled.get(pid, False) and p.worker_id == worker
        ]
        # Prefer MCP over native HTTP for same worker_id when both enabled
        candidates.sort(key=lambda item: 0 if item[1].kind == PluginKind.MCP else 1)
        if not candidates:
            from app.schemas import WorkerEvidence

            return WorkerResult(
                success=False,
                evidence=WorkerEvidence(
                    worker=worker,
                    action=decision.action,
                    stderr=f"No enabled plugin for worker '{worker}'",
                    exit_code=1,
                ),
                summary=f"No enabled plugin for worker '{worker}'",
            )
        return await candidates[0][1].run(decision)

    async def status_map(self) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for pid, plugin in self._plugins.items():
            if plugin.worker_id is None and plugin.kind != PluginKind.UTILITY:
                continue
            key = plugin.worker_id or pid
            if not self._enabled.get(pid, False):
                status[key] = False
                continue
            try:
                health = await plugin.health()
                status[key] = health.ok
            except Exception:  # noqa: BLE001
                status[key] = False
        return status


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return json.dumps(str(value))


_registry: PluginRegistry | None = None


def get_plugin_registry(config: AppConfig | None = None) -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry(config)
    return _registry


def reload_plugin_registry() -> PluginRegistry:
    global _registry
    reload_config()
    _registry = PluginRegistry(get_config())
    return _registry
