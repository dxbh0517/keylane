"""Lemonade Server OpenAI-compatible worker."""

from __future__ import annotations

import logging

import httpx

from app.config import AppConfig, get_config
from app.models_settings import load_models_settings
from app.schemas import RouteDecision, WorkerEvidence, WorkerResult

logger = logging.getLogger(__name__)


class LemonadeWorker:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    @property
    def base_url(self) -> str:
        settings = load_models_settings()
        return (settings.lemonade_base_url or self.config.lemonade.base_url).rstrip("/")

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code < 500
        except Exception:  # noqa: BLE001
            return False

    async def _resolve_model(self, client: httpx.AsyncClient, decision: RouteDecision) -> str:
        preferred = (decision.model or "").strip()
        if not preferred:
            preferred = str((decision.arguments or {}).get("model") or "").strip()
        settings = load_models_settings()
        configured = preferred or settings.lemonade_model or self.config.lemonade.default_model
        try:
            response = await client.get(f"{self.base_url}/models")
            response.raise_for_status()
            models = [m.get("id") for m in response.json().get("data", []) if m.get("id")]
        except Exception:  # noqa: BLE001
            return configured or "local-model"
        if not models:
            return configured or "local-model"
        if configured in models:
            return configured
        if configured in {"", "auto", "local-model", "local"}:
            return models[0]
        return configured or models[0]

    async def run(self, decision: RouteDecision) -> WorkerResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful local AI assistant running on Lemonade Server. "
                    "Answer clearly and do not claim to have modified files "
                    "or run commands unless explicitly proven."
                ),
            },
            {"role": "user", "content": decision.instruction},
        ]
        try:
            async with httpx.AsyncClient(
                timeout=self.config.lemonade.timeout_seconds
            ) as client:
                model = await self._resolve_model(client, decision)
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3,
                }
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                evidence = WorkerEvidence(
                    worker="lemonade",
                    action=decision.action,
                    response=content,
                    stdout=content,
                    exit_code=0,
                    metadata={"model": data.get("model", model)},
                )
                return WorkerResult(
                    success=True,
                    evidence=evidence,
                    summary=content,
                    raw=data,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lemonade worker failed")
            evidence = WorkerEvidence(
                worker="lemonade",
                action=decision.action,
                stderr=str(exc),
                exit_code=1,
            )
            return WorkerResult(
                success=False,
                evidence=evidence,
                summary=f"Lemonade error: {exc}",
            )
