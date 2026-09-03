"""Find an OpenAI-compatible server the user is already running.

The ``gpu`` adapter has always shipped pointed at ``127.0.0.1:1234`` with
``enabled = false`` and no model name, so it did nothing until someone went
looking for it in Settings. On a laptop with a discrete GPU and LM Studio
running on that exact port, that is a large, fast model sitting idle behind two
config lines nobody knew to change.

So the daemon asks, once, at startup: is anything answering ``/v1/models`` on
the ports these servers use? Discovery only *offers* — it fills in the model
name and reports what it found. Whether the route actually moves is the user's
decision in Settings, because on battery the NPU is the right answer and only
the user knows they are on battery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# The defaults these servers ship with. Order is the order they are tried.
KNOWN_SERVERS: tuple[tuple[str, str], ...] = (
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("llama.cpp / vLLM", "http://127.0.0.1:8000/v1"),
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
)

# A probe must not delay startup. Nothing here is required for Keylane to work.
PROBE_TIMEOUT = 1.5

# Embedding and reranking models answer /v1/models too, and picking one as the
# chat model produces a baffling failure much later.
_NOT_CHAT = ("embed", "embedding", "rerank", "bge-", "nomic-", "whisper", "clip")


@dataclass
class DiscoveredServer:
    """An OpenAI-compatible server that answered."""

    name: str
    base_url: str
    models: list[str] = field(default_factory=list)

    @property
    def suggested_model(self) -> str:
        return self.models[0] if self.models else ""


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(token in lowered for token in _NOT_CHAT)


def probe(base_url: str, timeout: float = PROBE_TIMEOUT) -> list[str]:
    """The chat models a server offers, or an empty list if it is not there."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return []
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    found = [str(row.get("id", "")) for row in rows if isinstance(row, dict) and row.get("id")]
    return [m for m in found if _is_chat_model(m)]


def discover(timeout: float = PROBE_TIMEOUT) -> list[DiscoveredServer]:
    """Every known port that answers, with the chat models it serves."""
    servers: list[DiscoveredServer] = []
    for name, base_url in KNOWN_SERVERS:
        models = probe(base_url, timeout)
        if models:
            logger.info("found %s at %s with %d model(s)", name, base_url, len(models))
            servers.append(DiscoveredServer(name=name, base_url=base_url, models=models))
    return servers


def suggest_for(base_url: str, timeout: float = PROBE_TIMEOUT) -> DiscoveredServer | None:
    """What is answering at one specific URL, if anything."""
    models = probe(base_url, timeout)
    if not models:
        return None
    name = next((n for n, u in KNOWN_SERVERS if u.rstrip("/") == base_url.rstrip("/")), "server")
    return DiscoveredServer(name=name, base_url=base_url, models=models)
