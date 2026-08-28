"""Keylane daemon — always-on FastAPI authority on 127.0.0.1:9100."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.loop import AIAgent
from daemon.config import add_mcp_server, all_settings, list_mcp_servers, remove_mcp_server, reset_settings, save_settings
from daemon.health import settings_health
from daemon.paths import ensure_data_dirs
from models.catalog import get_model, get_runtime, load_catalog
from scheduler.jobs import get_scheduler

logger = logging.getLogger(__name__)
HOST = "127.0.0.1"
PORT = 9100


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls: int = 0


class ModelSelectRequest(BaseModel):
    model_id: str


class McpServerRequest(BaseModel):
    id: str
    command: str
    args: list[str] = Field(default_factory=list)
    transport: str = "stdio"
    env: dict[str, str] | None = None


class SettingsPatchRequest(BaseModel):
    section: str
    values: dict[str, Any] = Field(default_factory=dict)


class SettingsResetRequest(BaseModel):
    section: str | None = None


class PermissionRespondRequest(BaseModel):
    id: str
    approved: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    get_scheduler()
    from tools.builtin import register_builtin_tools
    from mcp.client import load_mcp_tools
    from tools.registry import get_registry

    register_builtin_tools()
    try:
        await load_mcp_tools(get_registry())
    except Exception:  # noqa: BLE001
        logger.exception("MCP tool load failed")

    default_id, _, _ = load_catalog()
    runtime = get_runtime()

    def _warm() -> None:
        try:
            runtime.load(default_id, progress=lambda m: logger.info("npu: %s", m))
        except Exception:  # noqa: BLE001
            logger.exception("NPU warm-up failed")

    threading.Thread(target=_warm, daemon=True).start()
    yield
    from mcp.client import shutdown_mcp

    await shutdown_mcp()


app = FastAPI(title="Keylane", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "npu": get_runtime().status}


@app.get("/models")
def list_models() -> dict[str, Any]:
    default, device, entries = load_catalog()
    runtime = get_runtime()
    downloads = runtime.download_status()
    active = runtime.status.get("model_id")
    return {
        "default": default,
        "device": device,
        "active": active,
        "models": [
            {
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "params_b": e.params_b,
                "hf_repo": e.hf_repo,
                "downloaded": e.is_downloaded(),
                "downloading": downloads.get(e.id, {}).get("downloading", False),
                "download_progress": downloads.get(e.id, {}).get("progress", ""),
                "download_percent": downloads.get(e.id, {}).get("percent"),
                "download_error": downloads.get(e.id, {}).get("error", ""),
                "active": e.id == active,
            }
            for e in entries
        ],
    }


@app.post("/models/download")
def download_model_route(body: ModelSelectRequest) -> dict[str, Any]:
    entry = get_model(body.model_id)
    if not entry:
        raise HTTPException(404, "unknown model")
    runtime = get_runtime()
    try:
        return runtime.start_download(body.model_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@app.get("/skills")
def list_skills_route() -> dict[str, Any]:
    from memory.store import list_skills

    return {"skills": list_skills()}


@app.get("/tools")
def list_tools_route() -> dict[str, Any]:
    from tools.registry import get_registry

    reg = get_registry()
    if not reg._tools:  # noqa: SLF001
        from tools.builtin import register_builtin_tools

        register_builtin_tools()
    tools = []
    for tool in reg._tools.values():  # noqa: SLF001
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "dangerous": tool.dangerous,
                "source": "mcp" if tool.name.startswith("mcp.") else "builtin",
            }
        )
    tools.sort(key=lambda t: t["name"])
    return {"tools": tools}


@app.get("/mcp/servers")
async def mcp_servers_route() -> dict[str, Any]:
    from daemon.health import check_mcp_servers

    servers = list_mcp_servers()
    health = await check_mcp_servers()
    health_by_id = {h.get("id"): h for h in health}
    rows = []
    for srv in servers:
        sid = srv.get("id", "mcp")
        rows.append({**srv, "health": health_by_id.get(sid, {})})
    return {"servers": rows}


@app.post("/mcp/servers")
async def mcp_add_server(body: McpServerRequest) -> dict[str, Any]:
    try:
        servers = add_mcp_server(body.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    from mcp.client import reload_mcp_tools
    from tools.builtin import register_builtin_tools
    from tools.registry import get_registry

    register_builtin_tools()
    await reload_mcp_tools(get_registry())
    return {"servers": servers}


@app.delete("/mcp/servers/{server_id}")
async def mcp_remove_server(server_id: str) -> dict[str, Any]:
    servers = remove_mcp_server(server_id)
    from mcp.client import reload_mcp_tools
    from tools.builtin import register_builtin_tools
    from tools.registry import get_registry

    register_builtin_tools()
    await reload_mcp_tools(get_registry())
    return {"servers": servers}


@app.post("/mcp/reload")
async def mcp_reload_route() -> dict[str, Any]:
    from mcp.client import reload_mcp_tools
    from tools.builtin import register_builtin_tools
    from tools.registry import get_registry

    register_builtin_tools()
    count = await reload_mcp_tools(get_registry())
    return {"tools_loaded": count, "servers": list_mcp_servers()}


@app.post("/models/select")
def select_model(body: ModelSelectRequest) -> dict[str, Any]:
    entry = get_model(body.model_id)
    if not entry:
        raise HTTPException(404, "unknown model")
    runtime = get_runtime()
    try:
        return runtime.start_load(body.model_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    agent = AIAgent(session_id=body.session_id)
    result = await agent.run(body.message)
    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        tool_calls=result.tool_calls,
    )


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def run_agent() -> None:
        sources: list[dict[str, Any]] = []
        try:
            agent = AIAgent(session_id=body.session_id)

            def on_event(kind: str, payload: dict[str, Any]) -> None:
                nonlocal sources
                if kind == "sources":
                    sources = payload.get("sources", sources)
                queue.put_nowait({"type": kind, **payload})

            result = await agent.run(body.message, on_event=on_event)
            queue.put_nowait(
                {
                    "type": "done",
                    "answer": result.answer,
                    "session_id": result.session_id,
                    "tool_calls": result.tool_calls,
                    "sources": sources,
                }
            )
        except Exception as exc:  # noqa: BLE001
            queue.put_nowait({"type": "error", "message": str(exc)})
        finally:
            queue.put_nowait(None)

    async def event_stream():
        task = asyncio.create_task(run_agent())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/settings")
def settings_get() -> dict[str, Any]:
    return all_settings()


@app.patch("/settings")
def settings_patch(body: SettingsPatchRequest) -> dict[str, Any]:
    try:
        return save_settings(body.section, body.values)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/settings/reset")
def settings_reset(body: SettingsResetRequest) -> dict[str, Any]:
    try:
        return reset_settings(body.section)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/settings/health")
async def settings_health_route() -> dict[str, Any]:
    return await settings_health()


@app.get("/research/health")
async def research_health() -> dict[str, Any]:
    from daemon.health import check_searxng

    cfg = all_settings().get("research", {})
    searx = await check_searxng()
    return {
        "search_backend": cfg.get("search_backend", "searxng"),
        "extract_backend": cfg.get("extract_backend", "local"),
        "searxng": searx,
    }


@app.post("/settings/test/notification")
def test_notification() -> dict[str, Any]:
    from notify.desktop import send_notification

    send_notification("Keylane", "Test notification from Settings.")
    return {"ok": True}


@app.post("/settings/test/tts")
def test_tts() -> dict[str, Any]:
    from notify.tts_gate import speak_text

    speak_text("Keylane text to speech is working.")
    return {"ok": True}


@app.get("/sessions")
def list_sessions() -> dict[str, Any]:
    from memory.store import get_store

    return {"sessions": get_store().list_sessions()}


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str) -> dict[str, Any]:
    from memory.store import get_store

    return {"messages": get_store().get_messages(session_id)}


@app.post("/permissions/respond")
def permission_respond(body: PermissionRespondRequest) -> dict[str, Any]:
    from daemon.permissions import respond

    ok = respond(body.id, body.approved)
    if not ok:
        raise HTTPException(404, "unknown or resolved permission request")
    return {"ok": True}


@app.get("/permissions/pending")
def permission_pending() -> dict[str, Any]:
    from daemon.permissions import get_pending

    return {"pending": get_pending()}


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("daemon.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
