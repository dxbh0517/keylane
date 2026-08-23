"""Route user requests through the NPU router + plugin registry."""

from __future__ import annotations

import logging

from app.config import AppConfig, get_config
from app.npu.router_model import get_router_model
from app.permissions import PermissionError_, is_local_only, validate_route
from app.plugins.registry import PluginRegistry, get_plugin_registry
from app.schemas import RouteDecision, sync_allowed_workers

logger = logging.getLogger(__name__)


class RouterService:
    def __init__(
        self,
        config: AppConfig | None = None,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.config = config or get_config()
        self.registry = registry or get_plugin_registry(self.config)
        self.router_model = get_router_model(self.config)
        sync_allowed_workers(self.registry.enabled_worker_ids())

    async def available_workers(self, *, local_only: bool = False) -> set[str]:
        return await self.registry.available_workers(local_only=local_only)

    async def route(
        self,
        message: str,
        *,
        project: str | None = None,
        local_only: bool | None = None,
    ) -> RouteDecision:
        local = is_local_only(self.config, local_only)
        available = await self.available_workers(local_only=local)
        sync_allowed_workers(self.registry.enabled_worker_ids(local_only=local))
        if not available:
            available = self.registry.enabled_worker_ids(local_only=local) or {"lmstudio"}

        decision = self.router_model.route(
            message,
            project=project,
            local_only=local,
            available_workers=available,
        )

        try:
            return validate_route(
                decision,
                local_only=local,
                available_workers=None,
                config=self.config,
            )
        except PermissionError_ as exc:
            if "project directory is required" in str(exc).lower():
                raise
            if "lmstudio" in available and decision.worker != "lmstudio":
                logger.info("Route validation failed (%s); falling back to lmstudio.", exc)
                fallback = RouteDecision(
                    intent="general_question",
                    worker="lmstudio",
                    action="answer",
                    instruction=message,
                    working_directory=project,
                    requires_confirmation=False,
                )
                return validate_route(
                    fallback,
                    local_only=local,
                    available_workers=None,
                    config=self.config,
                )
            raise

    async def status(self) -> dict:
        from app.npu.status import npu_status

        local = is_local_only(self.config)
        plugin_status = await self.registry.status_map()
        npu = npu_status()
        return {
            **npu,
            "lmstudio": plugin_status.get("lmstudio", False),
            "comfyui": plugin_status.get("comfyui", False),
            "claude": False if local else plugin_status.get("claude", False),
            "cursor": False if local else plugin_status.get("cursor", False),
            "lemonade": plugin_status.get("lemonade", False),
            "local_only": local,
            **{f"plugin:{k}": v for k, v in plugin_status.items()},
        }
