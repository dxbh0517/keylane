"""The shipped model adapters.

``npu`` wraps the always-on OpenVINO pipeline Keylane already had. ``gpu``
speaks the OpenAI chat-completions wire format, which is what LM Studio,
llama.cpp's server, Ollama and vLLM all expose — so "a larger model on the GPU"
is a URL in settings rather than a second inference stack in this process.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import httpx

from seams.errors import LlmError

logger = logging.getLogger(__name__)


class NpuAdapter:
    """The local OpenVINO GenAI pipeline."""

    id = "npu"

    def __init__(self) -> None:
        from models.catalog import get_runtime

        self._runtime = get_runtime()

    def available(self) -> bool:
        return bool(self._runtime.status.get("ready"))

    @property
    def status(self) -> dict[str, Any]:
        state = self._runtime.status
        return {
            "kind": "openvino",
            "model": state.get("model_id"),
            "state": state.get("state"),
            "progress": state.get("progress"),
        }

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._runtime.generate(prompt, **kwargs)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self._runtime.chat(messages, **kwargs)


class OpenAiCompatAdapter:
    """A model served over the OpenAI chat-completions API.

    ``available()`` is a local check — is it configured and enabled — never a
    request. Asking the network whether a provider is usable turns every route
    resolution into a round trip, and a resolution happens on every step.
    """

    def __init__(
        self,
        *,
        adapter_id: str,
        base_url: str,
        model: str,
        api_key: str = "",
        enabled: bool = True,
        timeout: float = 180.0,
    ) -> None:
        self.id = adapter_id
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.enabled = enabled
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.enabled and self.base_url and self.model)

    @property
    def status(self) -> dict[str, Any]:
        return {
            "kind": "openai-compatible",
            "model": self.model,
            "base_url": self.base_url,
            "state": "configured" if self.available() else "disabled",
        }

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        if not self.available():
            raise LlmError(
                "LLM_ADAPTER_UNAVAILABLE",
                f"the {self.id} route is not configured (set a base_url and model)",
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LlmError(
                "LLM_TRANSPORT_ERROR",
                f"could not reach the {self.id} model at {self.base_url}: {exc}",
            ) from exc

        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(
                "LLM_BAD_RESPONSE",
                f"the {self.id} model returned a response with no message content",
            ) from exc

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system: str | None = None,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if images:
            logger.debug("%s adapter ignores images; text only", self.id)
        return self._post(messages, max_new_tokens)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        if images:
            logger.debug("%s adapter ignores images; text only", self.id)
        return self._post(list(messages), max_new_tokens)
