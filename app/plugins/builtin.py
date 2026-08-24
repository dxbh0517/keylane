"""Built-in native + MCP plugin wrappers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import AppConfig, get_config
from app.plugins.base import (
    BasePlugin,
    PluginHealth,
    PluginKind,
    SettingField,
    SettingType,
)
from app.plugins.mcp_plugin import McpPlugin
from app.schemas import RouteDecision, WorkerResult
from app.workers.claude import ClaudeWorker
from app.workers.comfyui import ComfyUiWorker
from app.workers.cursor import CursorWorker
from app.workers.lemonade import LemonadeWorker
from app.workers.lmstudio import LmStudioWorker

logger = logging.getLogger(__name__)


class LmStudioPlugin(BasePlugin):
    id = "lmstudio"
    name = "LM Studio"
    description = "Local OpenAI-compatible LLM inference via LM Studio."
    kind = PluginKind.NATIVE
    worker_id = "lmstudio"
    cloud = False
    homepage = "https://lmstudio.ai"

    def __init__(self, settings: dict[str, Any] | None = None, config: AppConfig | None = None) -> None:
        self._config = config or get_config()
        super().__init__(settings)
        self._worker = LmStudioWorker(self._config)

    def settings_schema(self) -> list[SettingField]:
        return [
            SettingField(
                key="base_url",
                label="Base URL",
                type=SettingType.STRING,
                default=self._config.lmstudio.base_url,
            ),
            SettingField(
                key="default_model",
                label="Default model id",
                type=SettingType.STRING,
                default=self._config.lmstudio.default_model,
                description="Use 'local-model' to auto-pick the first loaded model.",
            ),
        ]

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        settings = super().update_settings(data)
        if "base_url" in settings:
            self._config.lmstudio.base_url = str(settings["base_url"])
        if "default_model" in settings:
            self._config.lmstudio.default_model = str(settings["default_model"])
        self._worker = LmStudioWorker(self._config)
        return settings

    async def health(self) -> PluginHealth:
        ok = await self._worker.health()
        return PluginHealth(ok=ok, detail="LM Studio reachable" if ok else "LM Studio not reachable")

    async def run(self, decision: RouteDecision) -> WorkerResult:
        return await self._worker.run(decision)


class ClaudePlugin(BasePlugin):
    id = "claude"
    name = "Claude Code"
    description = "Controlled Claude Code subprocess for repository work."
    kind = PluginKind.NATIVE
    worker_id = "claude"
    cloud = True
    homepage = "https://docs.anthropic.com/en/docs/claude-code"

    def __init__(self, settings: dict[str, Any] | None = None, config: AppConfig | None = None) -> None:
        self._config = config or get_config()
        super().__init__(settings)
        self._worker = ClaudeWorker(self._config)

    def settings_schema(self) -> list[SettingField]:
        return [
            SettingField(
                key="command",
                label="CLI command",
                type=SettingType.STRING,
                default=self._config.claude.command,
            ),
            SettingField(
                key="timeout_seconds",
                label="Timeout (seconds)",
                type=SettingType.INTEGER,
                default=self._config.claude.timeout_seconds,
            ),
        ]

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        settings = super().update_settings(data)
        if "command" in settings:
            self._config.claude.command = str(settings["command"])
        if "timeout_seconds" in settings:
            self._config.claude.timeout_seconds = int(settings["timeout_seconds"])
        self._worker = ClaudeWorker(self._config)
        return settings

    async def health(self) -> PluginHealth:
        ok = await self._worker.health()
        return PluginHealth(ok=ok, detail="claude CLI found" if ok else "claude CLI missing")

    async def run(self, decision: RouteDecision) -> WorkerResult:
        return await self._worker.run(decision)


class CursorPlugin(BasePlugin):
    id = "cursor"
    name = "Cursor CLI"
    description = "Controlled Cursor Agent subprocess for repository work."
    kind = PluginKind.NATIVE
    worker_id = "cursor"
    cloud = True
    homepage = "https://cursor.com"

    def __init__(self, settings: dict[str, Any] | None = None, config: AppConfig | None = None) -> None:
        self._config = config or get_config()
        super().__init__(settings)
        self._worker = CursorWorker(self._config)

    def settings_schema(self) -> list[SettingField]:
        return [
            SettingField(
                key="command",
                label="CLI command / path",
                type=SettingType.PATH,
                default=self._config.cursor.command,
            ),
            SettingField(
                key="timeout_seconds",
                label="Timeout (seconds)",
                type=SettingType.INTEGER,
                default=self._config.cursor.timeout_seconds,
            ),
        ]

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        settings = super().update_settings(data)
        if "command" in settings:
            self._config.cursor.command = str(settings["command"])
        if "timeout_seconds" in settings:
            self._config.cursor.timeout_seconds = int(settings["timeout_seconds"])
        self._worker = CursorWorker(self._config)
        return settings

    async def health(self) -> PluginHealth:
        ok = await self._worker.health()
        return PluginHealth(ok=ok, detail="cursor-agent found" if ok else "cursor-agent missing")

    async def run(self, decision: RouteDecision) -> WorkerResult:
        return await self._worker.run(decision)


class ComfyUiHttpPlugin(BasePlugin):
    """Legacy direct HTTP ComfyUI worker (optional fallback)."""

    id = "comfyui-http"
    name = "ComfyUI (HTTP)"
    description = "Direct ComfyUI HTTP API with approved workflow templates. Prefer the MCP plugin."
    kind = PluginKind.NATIVE
    worker_id = "comfyui"
    cloud = False
    removable = True

    def __init__(self, settings: dict[str, Any] | None = None, config: AppConfig | None = None) -> None:
        self._config = config or get_config()
        super().__init__(settings)
        self._worker = ComfyUiWorker(self._config)

    def settings_schema(self) -> list[SettingField]:
        return [
            SettingField(
                key="base_url",
                label="ComfyUI URL",
                type=SettingType.STRING,
                default=self._config.comfyui.base_url,
            ),
        ]

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        settings = super().update_settings(data)
        if "base_url" in settings:
            self._config.comfyui.base_url = str(settings["base_url"])
        self._worker = ComfyUiWorker(self._config)
        return settings

    async def health(self) -> PluginHealth:
        ok = await self._worker.health()
        return PluginHealth(ok=ok, detail="ComfyUI HTTP OK" if ok else "ComfyUI HTTP unreachable")

    async def run(self, decision: RouteDecision) -> WorkerResult:
        return await self._worker.run(decision)


class LemonadePlugin(BasePlugin):
    """Lemonade Server — OpenAI-compatible local LLM (default http://127.0.0.1:13305/api/v1)."""

    id = "lemonade"
    name = "Lemonade"
    description = (
        "Lemonade Server OpenAI-compatible LLM. "
        "Note: clipboard lemond often uses port 9000 — keep the gateway on 9100."
    )
    kind = PluginKind.NATIVE
    worker_id = "lemonade"
    cloud = False
    homepage = "https://lemonade-server.ai"

    def __init__(self, settings: dict[str, Any] | None = None, config: AppConfig | None = None) -> None:
        self._config = config or get_config()
        super().__init__(settings)
        self._worker = LemonadeWorker(self._config)

    def settings_schema(self) -> list[SettingField]:
        from app.models_settings import load_models_settings

        models = load_models_settings()
        return [
            SettingField(
                key="base_url",
                label="Base URL",
                type=SettingType.STRING,
                default=models.lemonade_base_url or self._config.lemonade.base_url,
            ),
            SettingField(
                key="default_model",
                label="Default model id",
                type=SettingType.STRING,
                default=models.lemonade_model or self._config.lemonade.default_model,
                description="Use 'auto' to let the NPU router pick per request.",
            ),
        ]

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        settings = super().update_settings(data)
        if "base_url" in settings:
            self._config.lemonade.base_url = str(settings["base_url"])
        if "default_model" in settings:
            self._config.lemonade.default_model = str(settings["default_model"])
        self._worker = LemonadeWorker(self._config)
        return settings

    async def health(self) -> PluginHealth:
        ok = await self._worker.health()
        return PluginHealth(
            ok=ok,
            detail=f"Lemonade reachable at {self._worker.base_url}" if ok else "Lemonade Server not reachable",
            metadata={"base_url": self._worker.base_url},
        )

    async def run(self, decision: RouteDecision) -> WorkerResult:
        return await self._worker.run(decision)


def builtin_comfy_mcp(settings: dict[str, Any] | None = None) -> McpPlugin:
    return McpPlugin(
        plugin_id="comfyui",
        name="ComfyUI (MCP)",
        description=(
            "Official comfy-mcp stdio server. Prefer this over the HTTP plugin — "
            "it uses Comfy's agent tools (server_info, generate_image, run_workflow)."
        ),
        settings={
            "command": "comfy-mcp",
            "args": "[]",
            "health_tool": "server_info",
            "run_tool": "generate_image",
            "env": "{}",
            **(settings or {}),
        },
        author="Comfy-Org / built-in",
        version="0.1.0",
        homepage="https://docs.comfy.org/agent-tools/mcp",
        worker_id="comfyui",
        cloud=False,
    )


def create_builtin_plugins(config: AppConfig | None = None) -> dict[str, BasePlugin]:
    cfg = config or get_config()
    return {
        "lmstudio": LmStudioPlugin(config=cfg),
        "claude": ClaudePlugin(config=cfg),
        "cursor": CursorPlugin(config=cfg),
        "comfyui": builtin_comfy_mcp(),
        "comfyui-http": ComfyUiHttpPlugin(config=cfg),
        "lemonade": LemonadePlugin(config=cfg),
    }
