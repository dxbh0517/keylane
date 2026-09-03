"""Keylane as a provider, not only a consumer.

Keylane already speaks the OpenAI chat-completions API *outward*, to reach a
larger model on a GPU. This is the same wire format pointing the other way: the
resident NPU model, offered to anything else on the machine that knows how to
talk to LM Studio or Ollama — an editor plugin, a script, a shell function.

It is deliberately the raw model rather than the agent. No tools, no memory, no
research: a caller that wanted the assistant would use ``/chat/stream`` and get
its events. This is for the times something just wants a completion out of a
model that is already loaded and costs no VRAM.

The daemon's own token doubles as the API key, so a client configures it in the
field it already has for one.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


class ChatMessage(BaseModel):
    role: str
    # A multimodal client sends a list of parts. Only the text is used here.
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    stream: bool = False
    # Accepted and ignored: the local pipelines are greedy, and silently
    # pretending to honour a temperature would be worse than saying nothing.
    temperature: float | None = None
    top_p: float | None = None


def _text_of(content: Any) -> str:
    """Flatten OpenAI's message content into the string a pipeline wants."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content or "")


def _model_name() -> str:
    from models.catalog import get_runtime

    return str(get_runtime().status.get("model_id") or "keylane-local")


@router.get("/models")
def list_models() -> dict[str, Any]:
    """What this endpoint can serve — the one resident model, if any."""
    from models.catalog import get_runtime

    status = get_runtime().status
    if not status.get("ready"):
        return {"object": "list", "data": []}
    return {
        "object": "list",
        "data": [
            {
                "id": str(status.get("model_id")),
                "object": "model",
                "created": 0,
                "owned_by": "keylane",
            }
        ],
    }


def _envelope(model: str, content: str, finish: str = "stop") -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
    }


def _chunk(model: str, delta: dict[str, Any], finish: str | None = None) -> str:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    """One completion from the resident model, streamed or not."""
    import asyncio

    from seams import get_context
    from seams.errors import LlmError

    if not body.messages:
        raise HTTPException(400, "messages is required")

    llm = get_context().llm
    if not llm.is_ready("interactive"):
        raise HTTPException(
            503, "no model is loaded yet — check GET /health for the warm-up state"
        )

    messages = [{"role": m.role, "content": _text_of(m.content)} for m in body.messages]
    max_tokens = body.max_tokens or 512
    model = _model_name()

    if not body.stream:
        try:
            answer = await asyncio.to_thread(
                llm.chat, messages, route="interactive", max_new_tokens=max_tokens
            )
        except LlmError as exc:
            raise HTTPException(503, str(exc)) from exc
        return _envelope(model, answer)

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _on_token(piece: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, piece)

    async def _run() -> None:
        try:
            await asyncio.to_thread(
                llm.chat,
                messages,
                route="interactive",
                max_new_tokens=max_tokens,
                on_token=_on_token,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("openai-compatible completion failed: %s", exc)
        finally:
            queue.put_nowait(None)

    async def _stream() -> AsyncIterator[str]:
        task = asyncio.create_task(_run())
        yield _chunk(model, {"role": "assistant", "content": ""})
        while True:
            piece = await queue.get()
            if piece is None:
                break
            yield _chunk(model, {"content": piece})
        yield _chunk(model, {}, finish="stop")
        yield "data: [DONE]\n\n"
        await task

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
