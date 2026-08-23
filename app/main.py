"""Keylane — FastAPI local AI gateway (bind 127.0.0.1 only)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.audio.transcription import transcribe_wav_bytes
from app.config import ROOT, get_config
from app.orchestrator import GatewayOrchestrator
from app.plugins.registry import get_plugin_registry, reload_plugin_registry
from app.schemas import (
    ChatRequest,
    OpenAIChatRequest,
    ProjectInfo,
    ProjectsResponse,
    RouteRequest,
    StatusResponse,
    TaskResponse,
)
from app.settings_store import (
    GatewaySettingsUpdate,
    current_gateway_settings,
    update_gateway_settings,
)
from app.themes import get_theme_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

orchestrator: GatewayOrchestrator | None = None
WEB_DIR = ROOT / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global orchestrator
    config = get_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    get_theme_manager(config)
    get_plugin_registry(config)
    orchestrator = GatewayOrchestrator(config)
    logger.info(
        "AI Gateway ready on %s:%s (local_only=%s)",
        config.gateway.host,
        config.gateway.port,
        config.gateway.local_only,
    )
    yield
    orchestrator = None


app = FastAPI(
    title="Fedora Local AI Gateway",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")


def _orch() -> GatewayOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    return orchestrator


@app.get("/", response_model=None)
async def root():
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<p>Control panel missing. See /docs</p>")


@app.get("/theme.css")
async def theme_css() -> Response:
    path = get_theme_manager().web_css_path()
    return Response(
        path.read_text(encoding="utf-8"),
        media_type="text/css",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    cfg = get_config()
    data = await _orch().router.status()
    plugins = {
        k.removeprefix("plugin:"): v
        for k, v in data.items()
        if str(k).startswith("plugin:")
    }
    return StatusResponse(
        npu=bool(data.get("npu")),
        npu_driver=bool(data.get("npu_driver")),
        npu_openvino=bool(data.get("npu_openvino")),
        npu_detail=str(data.get("npu_detail") or ""),
        openvino_devices=list(data.get("openvino_devices") or []),
        lmstudio=data["lmstudio"],
        comfyui=data["comfyui"],
        claude=data["claude"],
        cursor=data["cursor"],
        lemonade=bool(data.get("lemonade", False)),
        gateway=True,
        local_only=data.get("local_only", cfg.gateway.local_only),
        plugins=plugins,
    )


@app.get("/api/projects", response_model=ProjectsResponse)
async def projects() -> ProjectsResponse:
    cfg = get_config()
    return ProjectsResponse(
        projects=[ProjectInfo(name=p.name, path=p.path) for p in cfg.projects]
    )


@app.get("/api/config")
async def get_gateway_config() -> dict[str, Any]:
    return current_gateway_settings().model_dump()


@app.put("/api/config")
async def put_gateway_config(update: GatewaySettingsUpdate) -> dict[str, Any]:
    return update_gateway_settings(update).model_dump()


@app.get("/api/plugins")
async def list_plugins(health: bool = True) -> list[dict[str, Any]]:
    registry = get_plugin_registry()
    if health:
        items = await registry.list_with_health()
    else:
        items = registry.list()
    return [i.model_dump() for i in items]


class PluginEnableBody(BaseModel):
    enabled: bool


@app.post("/api/plugins/{plugin_id}/enable")
async def enable_plugin(plugin_id: str, body: PluginEnableBody) -> dict[str, Any]:
    try:
        info = get_plugin_registry().set_enabled(plugin_id, body.enabled)
        return info.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/plugins/{plugin_id}/settings")
async def plugin_settings(plugin_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        info = get_plugin_registry().update_settings(plugin_id, body)
        return info.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class McpInstallBody(BaseModel):
    id: str
    name: str | None = None
    description: str = ""
    command: str
    args: list[str] = Field(default_factory=list)
    health_tool: str = "server_info"
    run_tool: str = "generate_image"
    worker_id: str | None = None
    cloud: bool = False
    author: str = "community"
    version: str = "0.1.0"
    homepage: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


@app.post("/api/plugins/install/mcp")
async def install_mcp_plugin(body: McpInstallBody) -> dict[str, Any]:
    data = body.model_dump()
    data["kind"] = "mcp"
    try:
        info = get_plugin_registry().install_mcp_manifest(data)
        return info.model_dump()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/plugins/{plugin_id}")
async def uninstall_plugin(plugin_id: str) -> dict[str, str]:
    try:
        get_plugin_registry().uninstall(plugin_id)
        return {"status": "removed", "id": plugin_id}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/plugins/reload")
async def reload_plugins() -> dict[str, Any]:
    registry = reload_plugin_registry()
    global orchestrator
    if orchestrator is not None:
        orchestrator = GatewayOrchestrator(get_config())
    return {"status": "reloaded", "count": len(registry.list())}


@app.get("/api/themes")
async def list_themes() -> list[dict[str, Any]]:
    return [t.model_dump() for t in get_theme_manager().list()]


class ThemeSelectBody(BaseModel):
    id: str


@app.put("/api/themes/active")
async def set_theme(body: ThemeSelectBody) -> dict[str, Any]:
    try:
        return get_theme_manager().set_active(body.id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/themes/install")
async def install_theme(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    try:
        return get_theme_manager().install_zip(data).model_dump()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/themes/{theme_id}")
async def uninstall_theme(theme_id: str) -> dict[str, str]:
    try:
        get_theme_manager().uninstall(theme_id)
        return {"status": "removed", "id": theme_id}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/themes/active/launcher.css")
async def launcher_theme_css() -> Response:
    path = get_theme_manager().launcher_css_path()
    return Response(
        path.read_text(encoding="utf-8"),
        media_type="text/css",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/route")
async def route(request: RouteRequest) -> dict[str, Any]:
    try:
        decision = await _orch().route_only(
            request.message,
            project=request.project,
            local_only=request.local_only,
        )
        return decision.model_dump()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat", response_model=TaskResponse)
async def chat(request: ChatRequest) -> TaskResponse:
    return await _orch().chat(request)


@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: ChatRequest) -> TaskResponse:
    return await _orch().chat(request)


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    result = await _orch().get_task(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@app.post("/api/tasks/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(task_id: str) -> TaskResponse:
    result = await _orch().cancel(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    data = await file.read()
    try:
        text = await transcribe_wav_bytes(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"text": text}


@app.post("/v1/chat/completions")
async def openai_chat(request: OpenAIChatRequest) -> dict[str, Any]:
    user_messages = [m.content for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")
    result = await _orch().chat(ChatRequest(message=user_messages[-1]))
    content = result.result or result.error or result.status.value
    return {
        "id": result.task_id,
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
