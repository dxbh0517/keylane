"""Route user requests through the NPU router + plugin registry."""

from __future__ import annotations

import logging

from app.config import AppConfig, get_config
from app.models_settings import load_models_settings
from app.npu.router_model import get_router_model
from app.permissions import PermissionError_, is_local_only, validate_route
from app.plugins.registry import PluginRegistry, get_plugin_registry
from app.schemas import RouteDecision, sync_allowed_workers
from app.worker_models import available_worker_models, resolve_model_for_decision

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

        models_settings = load_models_settings()
        available_models = await available_worker_models(self.config)
        model_modes = {
            "lmstudio": models_settings.lmstudio_mode,
            "lemonade": models_settings.lemonade_mode,
            "comfyui": models_settings.comfyui_mode,
        }

        decision = self.router_model.route(
            message,
            project=project,
            local_only=local,
            available_workers=available,
            available_models=available_models,
            model_modes=model_modes,
            preferred_chat_worker=models_settings.preferred_chat_worker,
        )
        decision = resolve_model_for_decision(decision, available=available_models)

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
            fallback_worker = None
            if "lmstudio" in available:
                fallback_worker = "lmstudio"
            elif "lemonade" in available:
                fallback_worker = "lemonade"
            if fallback_worker and decision.worker != fallback_worker:
                logger.info(
                    "Route validation failed (%s); falling back to %s.",
                    exc,
                    fallback_worker,
                )
                fallback = RouteDecision(
                    intent="general_question",
                    worker=fallback_worker,
                    action="answer",
                    instruction=message,
                    working_directory=project,
                    requires_confirmation=False,
                )
                fallback = resolve_model_for_decision(fallback, available=available_models)
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
