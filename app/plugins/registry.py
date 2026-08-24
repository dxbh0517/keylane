"""Plugin registry — discover, enable/disable, configure, persist."""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterator

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


# Nothing ships installed. A fresh Keylane reaches out to nothing until the
# user installs a plugin from the catalog. Existing installs are migrated in
# _load_state(): anything already in plugins.toml counts as installed.
DEFAULT_ENABLED: dict[str, bool] = {}


class PluginRegistry:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.state_path = self.config.root / "config" / "plugins.toml"
        self.community_dir = self.config.root / "plugins" / "community"
        self.community_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_dir = self.config.root / "plugins" / "catalog"
        self._plugins: dict[str, BasePlugin] = {}
        self._enabled: dict[str, bool] = dict(DEFAULT_ENABLED)
        self._installed: set[str] = set()
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
                # Anything already recorded is installed — this is what carries
                # an upgrade from the old always-on built-ins to the catalog.
                if bool(entry.get("installed", True)):
                    self._installed.add(pid)
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
                    "installed": True,
                    "enabled": self._enabled.get(pid, True),
                    "settings": plugin.settings,
                }
            else:
                data["plugins"][pid] = {
                    "installed": pid in self._installed,
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
            if pid not in self._installed:
                continue  # in the catalog, not installed
            if pid in self._settings:
                plugin.update_settings(self._settings[pid])
            self._plugins[pid] = plugin
            self._enabled.setdefault(pid, True)

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
                    plugin = self._load_python_plugin(manifest.parent, data, settings)
                    if plugin is None:
                        continue
                self._plugins[pid] = plugin
                self._enabled.setdefault(pid, bool(data.get("enabled", True)))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to load community plugin %s: %s", manifest, exc)

    def _load_python_plugin(
        self,
        folder: Path,
        data: dict[str, Any],
        settings: dict[str, Any],
    ) -> BasePlugin | None:
        """Import ``plugin.py`` from a community folder and build its plugin.

        The module must expose ``create_plugin(settings) -> BasePlugin`` or a
        ``Plugin`` class deriving from ``BasePlugin``. Community Python runs in
        this process, so it carries the same trust as any local script.
        """
        pid = str(data.get("id") or folder.name)
        entry = str(data.get("entry") or "plugin.py")
        module_path = folder / entry
        if not module_path.is_file():
            logger.warning(
                "Community plugin %s declares kind=%s but %s is missing.",
                pid,
                data.get("kind"),
                module_path,
            )
            return None

        module_name = f"keylane_community_{pid.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            logger.warning("Could not build an import spec for %s", module_path)
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(module_name, None)
            logger.exception("Community plugin %s failed to import: %s", pid, exc)
            return None

        factory = getattr(module, "create_plugin", None)
        if callable(factory):
            plugin = factory(settings)
        else:
            plugin_cls = getattr(module, "Plugin", None)
            if plugin_cls is None or not isinstance(plugin_cls, type):
                logger.warning(
                    "Community plugin %s exposes neither create_plugin() nor Plugin.", pid
                )
                return None
            plugin = plugin_cls(settings)

        if not isinstance(plugin, BasePlugin):
            logger.warning("Community plugin %s did not return a BasePlugin.", pid)
            return None
        plugin.removable = True
        return plugin

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

    # ------------------------------------------------------------- catalog

    def catalog(self) -> list[dict[str, Any]]:
        """Everything installable, with whether it is installed already."""
        entries: list[dict[str, Any]] = []
        if not self.catalog_dir.is_dir():
            return entries
        for manifest in sorted(self.catalog_dir.glob("*/plugin.toml")):
            try:
                with manifest.open("rb") as fh:
                    data = tomllib.load(fh)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Bad catalog entry %s: %s", manifest, exc)
                continue
            pid = str(data.get("id") or manifest.parent.name)
            entries.append(
                {
                    "id": pid,
                    "name": str(data.get("name") or pid),
                    "description": str(data.get("description") or ""),
                    "kind": str(data.get("kind") or "native"),
                    "worker_id": data.get("worker_id"),
                    "cloud": bool(data.get("cloud", False)),
                    "homepage": data.get("homepage"),
                    "tags": [
                        tag.strip()
                        for tag in str(data.get("tags") or "").split(",")
                        if tag.strip()
                    ],
                    "installed": pid in self._installed,
                    "enabled": bool(self._enabled.get(pid, False)),
                }
            )
        return entries

    def install_from_catalog(self, plugin_id: str) -> PluginInfo:
        """Register a catalog plugin and switch it on."""
        manifest = self.catalog_dir / plugin_id / "plugin.toml"
        if not manifest.exists():
            raise KeyError(f"'{plugin_id}' is not in the catalog")
        with manifest.open("rb") as fh:
            data = tomllib.load(fh)

        entry = str(data.get("entry") or "")
        if entry.startswith("builtin:"):
            builtin_id = entry.split(":", 1)[1]
            builtins = create_builtin_plugins(self.config)
            plugin = builtins.get(builtin_id)
            if plugin is None:
                raise KeyError(f"No built-in implementation named '{builtin_id}'")
            if plugin_id in self._settings:
                plugin.update_settings(self._settings[plugin_id])
        elif (data.get("kind") or "mcp").lower() == "mcp":
            plugin = mcp_plugin_from_manifest(data, self._settings.get(plugin_id))
        else:
            plugin = self._load_python_plugin(
                manifest.parent, data, self._settings.get(plugin_id, {})
            )
            if plugin is None:
                raise ValueError(f"'{plugin_id}' could not be loaded")

        self._plugins[plugin_id] = plugin
        self._installed.add(plugin_id)
        self._enabled[plugin_id] = True
        self._write_state()
        return plugin.info(enabled=True)

    def uninstall_from_catalog(self, plugin_id: str, *, purge: bool = False) -> None:
        """Unregister a plugin, keeping its settings unless purging."""
        if plugin_id not in self._installed:
            raise KeyError(f"'{plugin_id}' is not installed")
        self._plugins.pop(plugin_id, None)
        self._installed.discard(plugin_id)
        self._enabled.pop(plugin_id, None)
        if purge:
            self._settings.pop(plugin_id, None)
        self._write_state()

    def is_installed(self, plugin_id: str) -> bool:
        return plugin_id in self._installed

    def items(self) -> Iterator[tuple[str, BasePlugin]]:
        """Iterate ``(plugin_id, plugin)`` for every discovered plugin."""
        return iter(sorted(self._plugins.items()))

    def contributed_skills(self) -> dict[str, list[Any]]:
        """Skills contributed by enabled plugins, keyed by plugin id."""
        out: dict[str, list[Any]] = {}
        for pid, plugin in self._plugins.items():
            if not self._enabled.get(pid, False):
                continue
            try:
                skills = plugin.skills() or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plugin %s failed to list skills: %s", pid, exc)
                continue
            if skills:
                out[pid] = skills
        return out

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
        """Aggregate health by worker_id.

        Multiple plugins can share a worker_id (e.g. comfyui MCP + HTTP).
        A disabled/unhealthy sibling must not overwrite a healthy enabled one.
        """
        status: dict[str, bool] = {}
        for pid, plugin in self._plugins.items():
            if plugin.worker_id is None and plugin.kind != PluginKind.UTILITY:
                continue
            key = plugin.worker_id or pid
            if not self._enabled.get(pid, False):
                status.setdefault(key, False)
                continue
            try:
                health = await plugin.health()
                if health.ok:
                    status[key] = True
                else:
                    status.setdefault(key, False)
            except Exception:  # noqa: BLE001
                status.setdefault(key, False)
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
