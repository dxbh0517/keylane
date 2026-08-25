"""LM Studio OpenAI-compatible worker."""

from __future__ import annotations

import logging

import httpx

from app.config import AppConfig, get_config
from app.schemas import RouteDecision, WorkerEvidence, WorkerResult

logger = logging.getLogger(__name__)


class LmStudioWorker:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    @property
    def base_url(self) -> str:
        return self.config.lmstudio.base_url.rstrip("/")

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.25) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code < 500
        except Exception:  # noqa: BLE001
            return False

    async def _resolve_model(self, client: httpx.AsyncClient, decision: RouteDecision) -> str:
        preferred = (decision.model or "").strip()
        if not preferred:
            preferred = str((decision.arguments or {}).get("model") or "").strip()
        configured = preferred or self.config.lmstudio.default_model
        try:
            response = await client.get(f"{self.base_url}/models")
            response.raise_for_status()
            models = [m.get("id") for m in response.json().get("data", []) if m.get("id")]
        except Exception:  # noqa: BLE001
            return configured
        if not models:
            return configured
        if configured in models:
            return configured
        # "local-model" / auto placeholders — use whatever LM Studio has loaded.
        if configured in {"", "local-model", "local", "auto"}:
            return models[0]
        return configured

    async def run(self, decision: RouteDecision) -> WorkerResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful local AI assistant. "
                    "Answer clearly and do not claim to have modified files "
                    "or run commands unless explicitly proven."
                ),
            },
            {"role": "user", "content": decision.instruction},
        ]
        try:
            async with httpx.AsyncClient(
                timeout=self.config.lmstudio.timeout_seconds
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
                    worker="lmstudio",
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
            logger.exception("LM Studio worker failed")
            evidence = WorkerEvidence(
                worker="lmstudio",
                action=decision.action,
                stderr=str(exc),
                exit_code=1,
            )
            return WorkerResult(
                success=False,
                evidence=evidence,
                summary=f"LM Studio error: {exc}",
            )
