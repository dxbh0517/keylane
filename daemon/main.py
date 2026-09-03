"""Keylane daemon — always-on FastAPI authority on 127.0.0.1:9100."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import base64
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.loop import AIAgent
from daemon.auth import auth_middleware
from daemon.config import add_mcp_server, all_settings, list_mcp_servers, remove_mcp_server, reset_settings, save_settings
from daemon.health import settings_health
from daemon.paths import ensure_data_dirs
from models.catalog import (
    active_runtime_id,
    default_model_id,
    get_model,
    get_runtime,
    load_catalog,
)
from runtimes import list_backends, status_payload
from scheduler.jobs import get_scheduler, restore_scheduled_tasks

logger = logging.getLogger(__name__)
HOST = "127.0.0.1"
PORT = 9100


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    images: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls: int = 0


class ModelSelectRequest(BaseModel):
    model_id: str


class ModelImportRequest(BaseModel):
    """Add a Hugging Face repo to the catalog.

    Leave `runtime` and `subfolder` unset and Keylane reads the repo's file
    listing and picks the build that suits this machine.
    """

    repo: str
    runtime: str | None = None
    subfolder: str | None = None
    name: str = ""
    device: str = ""
    model_id: str = ""
    download: bool = False


class McpServerRequest(BaseModel):
    id: str
    transport: str = ""
    # stdio
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    # streamable http
    url: str = ""
    auth_header: str = ""
    headers: dict[str, str] | None = None


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
    try:
        restore_scheduled_tasks()
    except Exception:  # noqa: BLE001
        logger.exception("could not restore scheduled tasks")
    from tools.builtin import register_builtin_tools
    from mcpbridge.client import load_mcp_tools
    from tools.registry import get_registry

    register_builtin_tools()
    try:
        await load_mcp_tools(get_registry())
    except Exception:  # noqa: BLE001
        logger.exception("MCP tool load failed")

    default_id = default_model_id()
    runtime = get_runtime()

    def _warm() -> None:
        try:
            entry = get_model(default_id)
            if entry is None:
                logger.warning("Default model %s is unknown — skipping warm-up", default_id)
                return
            installed, detail = entry.backend.installed()
            if not installed:
                logger.warning(
                    "Default model %s needs the %s runtime, which is not installed (%s)"
                    " — skipping warm-up",
                    default_id,
                    entry.backend.info.name,
                    detail,
                )
                return
            if not entry.is_downloaded():
                logger.info("Default model %s not downloaded — skipping warm-up", default_id)
                return
            runtime.load(default_id, progress=lambda m: logger.info("npu: %s", m))
        except Exception:  # noqa: BLE001
            logger.exception("NPU warm-up failed")

    threading.Thread(target=_warm, daemon=True).start()
    yield
    from mcpbridge.client import shutdown_mcp

    await shutdown_mcp()


app = FastAPI(title="Keylane", lifespan=lifespan)
# No CORS. Nothing that talks to this daemon is a browser, and the header that
# used to be here told every website it could read /memories. See daemon/auth.py.
app.middleware("http")(auth_middleware)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "npu": get_runtime().status}


@app.get("/models")
def list_models(runtime: str = "") -> dict[str, Any]:
    """The catalog. Pass ?runtime= to see only what one runtime can load."""
    default, device, entries = load_catalog()
    runtime_filter = runtime.strip().lower()
    rt = get_runtime()
    downloads = rt.download_status()
    active = rt.status.get("model_id")
    startup_default = default_model_id()

    rows = []
    for e in entries:
        if runtime_filter and e.runtime != runtime_filter:
            continue
        info = downloads.get(e.id, {})
        rows.append(
            {
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "params_b": e.params_b,
                "hf_repo": e.hf_repo,
                "runtime": e.runtime,
                "subfolder": e.subfolder,
                "source": e.source,
                "device": e.resolve_device(device),
                "pipeline": e.backend.model_kind(e.model_dir) if e.model_dir.is_dir() else "llm",
                "downloaded": e.is_downloaded(),
                "downloading": info.get("downloading", False),
                "download_progress": info.get("progress", ""),
                "download_percent": info.get("percent"),
                "download_file": info.get("file", ""),
                "download_error": info.get("error", ""),
                "active": e.id == active,
            }
        )

    return {
        "default": startup_default,
        "catalog_default": default,
        "device": device,
        "active": active,
        "runtime": active_runtime_id(),
        # The device chosen per runtime, so Settings does not need a second call.
        "devices": all_settings().get("models", {}).get("devices", {}),
        "runtimes": [status_payload(b) for b in list_backends()],
        "models": rows,
    }


@app.get("/models/servers")
def discover_servers() -> dict[str, Any]:
    """OpenAI-compatible servers answering on this machine right now.

    Settings uses this to fill in the `gpu` adapter rather than making the user
    know that LM Studio's default port is 1234. It only reports — moving a
    route is still a decision, because on battery the NPU is the right answer.
    """
    from seams.discovery import discover

    found = discover()
    configured = {
        str(a.get("id")): str(a.get("base_url", ""))
        for a in all_settings().get("models", {}).get("adapters", []) or []
    }
    return {
        "servers": [
            {
                "name": s.name,
                "base_url": s.base_url,
                "models": s.models,
                "suggested_model": s.suggested_model,
                "configured_as": next(
                    (aid for aid, url in configured.items() if url.rstrip("/") == s.base_url.rstrip("/")),
                    "",
                ),
            }
            for s in found
        ]
    }


@app.get("/runtimes")
def list_runtimes() -> dict[str, Any]:
    """Which inference stacks are installed, and what each can target."""
    _, _, entries = load_catalog()
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.runtime] = counts.get(entry.runtime, 0) + 1
    return {
        "active": active_runtime_id(),
        "runtimes": [
            {**status_payload(b), "models": counts.get(b.info.id, 0)} for b in list_backends()
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
    from seams import get_context

    return {
        "skills": [
            {
                "id": s.name,
                "name": s.name,
                "description": s.description,
                "when_to_use": s.when_to_use,
                "source": s.source,
                "model_invocable": s.invocation.model_invocable,
                "user_invocable": s.invocation.user_invocable,
            }
            for s in get_context().skills.list()
        ]
    }


@app.get("/tools")
def list_tools_route() -> dict[str, Any]:
    from seams import get_context
    from tools.registry import ALWAYS_GATED, get_registry

    get_context()  # composes the registry if this is the first read
    tools = []
    for tool in get_registry().visible().values():
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "dangerous": tool.dangerous,
                # Whether this tool goes through the permission gate at all —
                # the mode of an ungated tool is not consulted.
                "gated": tool.dangerous or tool.name in ALWAYS_GATED,
                "timeout_ms": tool.timeout_ms,
                "concurrency_safe": tool.concurrency_safe,
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
        payload = {k: v for k, v in body.model_dump(exclude_none=True).items() if v != ""}
        servers = add_mcp_server(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    from mcpbridge.client import reload_mcp_tools
    from tools.builtin import register_builtin_tools
    from tools.registry import get_registry

    register_builtin_tools()
    await reload_mcp_tools(get_registry())
    return {"servers": servers}


@app.delete("/mcp/servers/{server_id}")
async def mcp_remove_server(server_id: str) -> dict[str, Any]:
    servers = remove_mcp_server(server_id)
    from mcpbridge.client import reload_mcp_tools
    from tools.builtin import register_builtin_tools
    from tools.registry import get_registry

    register_builtin_tools()
    await reload_mcp_tools(get_registry())
    return {"servers": servers}


@app.post("/mcp/reload")
async def mcp_reload_route() -> dict[str, Any]:
    from mcpbridge.client import reload_mcp_tools
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
    # Say this now rather than letting the load thread fail three steps later.
    installed, detail = entry.backend.installed()
    if not installed:
        raise HTTPException(
            400,
            f"{entry.name} runs on {entry.backend.info.name}, which is not installed "
            f"({detail}). Install it with: {entry.backend.info.install_hint}",
        )
    runtime = get_runtime()
    try:
        return runtime.start_load(body.model_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@app.post("/models/import")
def import_model_route(body: ModelImportRequest) -> dict[str, Any]:
    from models.importer import ImportError_, import_model, inspect_repo

    try:
        inspection = inspect_repo(body.repo)
        entry = import_model(
            body.repo,
            runtime=body.runtime,
            subfolder=body.subfolder,
            name=body.name,
            device=body.device,
            model_id=body.model_id,
        )
    except ImportError_ as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc

    if body.download:
        get_runtime().start_download(entry["id"])

    return {
        "model": entry,
        # Every build found in the repo, so the UI can say which one was taken
        # and offer the others.
        "variants": [
            {
                "runtime": v.runtime,
                "subfolder": v.subfolder,
                "label": v.label,
                "chosen": v.runtime == entry["runtime"] and v.subfolder == entry["subfolder"],
            }
            for v in sorted(inspection.variants, key=lambda v: -v.score)
        ],
    }


@app.delete("/models/imported/{model_id}")
def forget_imported_model(model_id: str, delete_files: bool = False) -> dict[str, Any]:
    from models.importer import remove_imported

    if not remove_imported(model_id, delete_files=delete_files):
        raise HTTPException(404, "not an imported model")
    return {"forgotten": model_id, "files_deleted": delete_files}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    image_bytes: list[bytes] = []
    for encoded in body.images:
        try:
            image_bytes.append(base64.b64decode(encoded))
        except Exception:  # noqa: BLE001
            continue
    agent = AIAgent(session_id=body.session_id)
    result = await agent.run(body.message, images=image_bytes or None)
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
        image_bytes: list[bytes] = []
        for encoded in body.images:
            try:
                image_bytes.append(base64.b64decode(encoded))
            except Exception:  # noqa: BLE001
                continue
        try:
            agent = AIAgent(session_id=body.session_id)

            def on_event(kind: str, payload: dict[str, Any]) -> None:
                nonlocal sources
                if kind == "sources":
                    sources = payload.get("sources", sources)
                queue.put_nowait({"type": kind, **payload})

            result = await agent.run(
                body.message,
                on_event=on_event,
                images=image_bytes or None,
            )
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


# ── Reminders, background work, memory, inbox ────────────────────────────


class ReminderRequest(BaseModel):
    text: str
    when: str


class MemoryRequest(BaseModel):
    text: str
    kind: str = "fact"


@app.get("/tasks")
def tasks_route() -> dict[str, Any]:
    from scheduler.jobs import background_jobs, list_tasks

    return {"scheduled": list_tasks(), "background": background_jobs()}


@app.post("/tasks/reminder")
def create_reminder_route(body: ReminderRequest) -> dict[str, Any]:
    from scheduler.jobs import create_reminder

    result = create_reminder(body.text, body.when)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.delete("/tasks/{task_id}")
def cancel_task_route(task_id: str) -> dict[str, Any]:
    from scheduler.jobs import cancel_task

    result = cancel_task(task_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/memories")
def memories_route(kind: str = "") -> dict[str, Any]:
    from memory.store import list_memories

    return {"memories": list_memories(kind or None)}


@app.post("/memories")
def memory_add_route(body: MemoryRequest) -> dict[str, Any]:
    from memory.store import save_memory

    try:
        return save_memory(body.text, kind=body.kind)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/memories/{memory_id}")
def memory_delete_route(memory_id: str) -> dict[str, Any]:
    from memory.store import forget_memory

    if not forget_memory(memory_id):
        raise HTTPException(404, "unknown memory")
    return {"forgotten": memory_id}


@app.get("/inbox")
def inbox_route(unread_only: bool = True) -> dict[str, Any]:
    from memory.store import list_inbox

    return {"items": list_inbox(unread_only)}


@app.post("/inbox/read")
def inbox_read_route(item_id: str | None = None) -> dict[str, Any]:
    from memory.store import mark_inbox_read

    return {"marked": mark_inbox_read(item_id)}


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("daemon.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
