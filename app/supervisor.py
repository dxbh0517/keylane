"""Supervisor backend — NPU first, then LM Studio / Lemonade.

The assistant loop needs a model that can emit tool JSON. On machines where
the NPU export is missing or unhealthy, we fall back to a local OpenAI-compatible
chat worker as the supervisor. The user can pin a preference in assistant.toml.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.assistant_settings import load_assistant_settings
from app.config import AppConfig, get_config
from app.npu.pipeline import get_pipeline

logger = logging.getLogger(__name__)


class SupervisorBackend:
    """Generates one assistant turn (system + user → text)."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.pipeline = get_pipeline("router", self.config)

    @property
    def preference(self) -> str:
        settings = load_assistant_settings()
        return (settings.supervisor.backend or "auto").strip().lower()

    @property
    def npu_ready(self) -> bool:
        return bool(self.pipeline.loaded)

    def active_backend(self) -> str:
        pref = self.preference
        if pref == "npu":
            return "npu" if self.npu_ready else "none"
        if pref in {"lmstudio", "lemonade"}:
            return pref
        # auto
        if self.npu_ready:
            return "npu"
        return self._preferred_chat_worker()

    def _preferred_chat_worker(self) -> str:
        settings = load_assistant_settings()
        preferred = (settings.supervisor.fallback_worker or "auto").strip().lower()
        if preferred in {"lmstudio", "lemonade"}:
            return preferred
        # Prefer whichever is configured; lmstudio first to match legacy routing.
        return "lmstudio"

    def generate_chat(
        self,
        system: str,
        user: str,
        *,
        max_new_tokens: int = 384,
    ) -> tuple[str, str]:
        """Return ``(text, backend_id)``.

        ``backend_id`` is ``npu``, ``lmstudio``, ``lemonade``, or ``none``.
        """
        backend = self.active_backend()
        if backend == "npu":
            text = self.pipeline.generate_chat(
                system, user, max_new_tokens=max_new_tokens
            )
            return text, "npu"
        if backend in {"lmstudio", "lemonade"}:
            text = self._openai_chat(backend, system, user, max_tokens=max_new_tokens)
            return text, backend
        return "", "none"

    def _openai_chat(
        self,
        worker: str,
        system: str,
        user: str,
        *,
        max_tokens: int,
    ) -> str:
        if worker == "lemonade":
            base = self.config.lemonade.base_url.rstrip("/")
            model = self.config.lemonade.default_model or "auto"
            timeout = self.config.lemonade.timeout_seconds
        else:
            base = self.config.lmstudio.base_url.rstrip("/")
            model = self.config.lmstudio.default_model or "local-model"
            timeout = self.config.lmstudio.timeout_seconds

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=timeout) as client:
            # Resolve placeholder model ids against /models when possible.
            if model in {"", "auto", "local-model", "local"}:
                try:
                    listed = client.get(f"{base}/models")
                    if listed.status_code < 400:
                        ids = [
                            m.get("id")
                            for m in listed.json().get("data", [])
                            if m.get("id")
                        ]
                        if ids:
                            payload["model"] = ids[0]
                except Exception:  # noqa: BLE001
                    pass
            response = client.post(f"{base}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"] or "")


_supervisor: SupervisorBackend | None = None


def get_supervisor(config: AppConfig | None = None) -> SupervisorBackend:
    global _supervisor
    if _supervisor is None:
        _supervisor = SupervisorBackend(config)
    return _supervisor


def reload_supervisor(config: AppConfig | None = None) -> SupervisorBackend:
    global _supervisor
    _supervisor = SupervisorBackend(config)
    return _supervisor
