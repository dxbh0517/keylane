"""Keylane — FastAPI local AI gateway (bind 127.0.0.1 only)."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.activity import get_activity_bus
from app.assistant import get_assistant, reload_assistant
from app.assistant_settings import (
    AssistantSettingsUpdate,
    load_assistant_settings,
    set_tool_policy,
    update_assistant_settings,
)
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
    IncompleteModel,
    StatusResponse,
    TaskResponse,
    ToolCallRequest,
)
from app.projects_store import ProjectsUpdate, write_projects
from app.skills import SkillError, get_skill_registry, reload_skill_registry
from app.workers_store import (
    WorkerEndpointsUpdate,
    current_worker_endpoints,
    update_worker_endpoints,
)
from app.tools.registry import (
    ConfirmationRequired,
    get_tool_registry,
    reload_tool_registry,
)
from app.settings_store import (
    GatewaySettingsUpdate,
    current_gateway_settings,
    update_gateway_settings,
)
from app.models_catalog import installed_router_models
from app.models_settings import (
    ModelsSettingsUpdate,
    models_overview,
    update_models_settings,
)
from app.themes import get_theme_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

orchestrator: GatewayOrchestrator | None = None
WEB_DIR = ROOT / "web"

# Set the moment a stop signal arrives, so long-lived streams end themselves
# instead of blocking uvicorn's graceful shutdown.
shutting_down = asyncio.Event()


def _chain_shutdown_signals() -> list[tuple[int, Any]]:
    """Flag shutdown as soon as SIGINT/SIGTERM lands.

    Uvicorn drains in-flight requests *before* it runs lifespan shutdown, so
    setting the flag in teardown is too late — an endless SSE stream (the tray
    holds one open all session) would block the drain until the graceful
    timeout expires and systemd resorts to SIGABRT. Chaining onto uvicorn's own
    handler lets the stream see the signal immediately and return.
    """
    restore: list[tuple[int, Any]] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(sig)

            def handler(signum, frame, _previous=previous):
                shutting_down.set()
                if callable(_previous):
                    _previous(signum, frame)

            signal.signal(sig, handler)
            restore.append((sig, previous))
        except (ValueError, OSError):
            # Not the main thread (TestClient, embedded use). The uvicorn
            # --timeout-graceful-shutdown backstop covers that case.
            logger.debug("Could not chain a handler for signal %s", sig)
    return restore


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global orchestrator
    shutting_down.clear()
    restore_signals = _chain_shutdown_signals()
    config = get_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    get_theme_manager(config)
    get_plugin_registry(config)
    get_skill_registry()
    tools = get_tool_registry(config)
    orchestrator = GatewayOrchestrator(config)

    # MCP servers each cost a subprocess round-trip, so discover their tools
    # off the startup path rather than blocking the first request.
    async def discover_mcp_tools() -> None:
        try:
            count = await tools.load_mcp_tools()
            if count:
                logger.info("Discovered %s MCP tools for the assistant.", count)
        except Exception as exc:  # noqa: BLE001
            logger.info("MCP tool discovery skipped: %s", exc)

    mcp_task = asyncio.create_task(discover_mcp_tools())

    logger.info(
        "Keylane gateway ready on %s:%s (local_only=%s, tools=%s)",
        config.gateway.host,
        config.gateway.port,
        config.gateway.local_only,
        len(tools.usable_specs()),
    )
    yield

    # Belt and braces: the signal handler normally sets this first.
    shutting_down.set()
    mcp_task.cancel()
    with suppress(asyncio.CancelledError):
        await mcp_task
    for sig, previous in restore_signals:
        with suppress(ValueError, OSError):
            signal.signal(sig, previous)
    orchestrator = None


app = FastAPI(
    title="Keylane local AI gateway",
    version="0.3.0",
    lifespan=lifespan,
    # /docs belongs to the handbook; the OpenAPI explorer moves aside.
    docs_url="/api-docs",
    redoc_url=None,
)

_cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    # Same-origin in practice; the explicit ports keep a browser opened on
    # localhost rather than 127.0.0.1 working too.
    allow_origins=[
        f"http://127.0.0.1:{_cfg.gateway.port}",
        f"http://localhost:{_cfg.gateway.port}",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_DIR = ROOT / "web" / "docs"

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=DOCS_DIR, html=True), name="docs")


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
        assistant=get_assistant().pipeline.loaded,
        assistant_device=get_assistant().pipeline.device,
        assistant_note=(
            get_assistant().pipeline.degraded_reason
            or (None if get_assistant().pipeline.loaded else get_assistant().pipeline.status)
        ),
        tools_enabled=load_assistant_settings().tools.enabled,
        tool_count=len(get_tool_registry().usable_specs()),
        busy=get_activity_bus().snapshot().busy,
        incomplete_models=[
            IncompleteModel(id=m["id"], repo_id=m["repo_id"], missing=m["missing"])
            for m in installed_router_models()
            if not m["ready"]
        ],
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


@app.get("/api/models")
async def get_models_overview() -> dict[str, Any]:
    from app.worker_models import available_worker_models

    data = models_overview()
    try:
        data["available"] = await available_worker_models()
    except Exception as exc:  # noqa: BLE001
        data["available"] = {
            "router": [],
            "lmstudio": [],
            "lemonade": [],
            "comfyui": [],
        }
        data["available_error"] = str(exc)
    # Mark catalog chat suggestions that match a downloaded LM Studio model.
    try:
        lm_ids = [str(m.get("id") or "").lower() for m in (data.get("available") or {}).get("lmstudio") or []]
        for item in ((data.get("recommendations") or {}).get("chat") or []):
            cid = str(item.get("id") or "").lower().replace("_", "-")
            item["available"] = any(cid in mid or mid in cid or cid.split("-")[0] in mid for mid in lm_ids if mid)
    except Exception:  # noqa: BLE001
        pass
    return data


@app.get("/api/devices")
async def list_devices() -> dict[str, Any]:
    """Compute devices the control plane can run on, as the user sees them.

    OpenVINO names devices ``CPU``, ``GPU``, ``GPU.1``, ``NPU``. By convention
    ``GPU.0`` is the integrated adapter and later indices are discrete, but the
    only reliable label is the device's own full name — so that is what is
    shown, with the convention as a fallback.
    """
    from app.models_settings import load_models_settings, resolve_openvino_device

    settings = load_models_settings()
    devices: list[dict[str, Any]] = []
    try:
        import openvino as ov

        core = ov.Core()
        available = list(core.available_devices)
    except Exception as exc:  # noqa: BLE001
        return {"devices": [], "error": str(exc)[:200], "primary": settings.primary_device}

    def describe(name: str) -> tuple[str, str]:
        try:
            full = str(core.get_property(name, "FULL_DEVICE_NAME"))
        except Exception:  # noqa: BLE001
            full = name
        lowered = full.lower()
        if name.startswith("NPU"):
            return "NPU", full
        if name.startswith("CPU"):
            return "CPU", full
        if name.startswith("GPU"):
            if "igpu" in lowered or name in {"GPU", "GPU.0"} and "intel" in lowered:
                kind = "Integrated graphics"
            elif "dgpu" in lowered:
                kind = "Discrete graphics"
            else:
                kind = "Integrated graphics" if name in {"GPU", "GPU.0"} else "Discrete graphics"
            return kind, full
        return name, full

    for name in available:
        label, full = describe(name)
        devices.append({"id": name, "label": label, "name": full})

    router = get_pipeline_safe("router")
    return {
        "devices": devices,
        "primary": settings.primary_device,
        "resolved": resolve_openvino_device(settings),
        "active": router.get("device"),
        "loaded": router.get("loaded", False),
        "model": settings.router_model_id,
    }


def get_pipeline_safe(role: str) -> dict[str, Any]:
    try:
        from app.npu.pipeline import get_pipeline

        pipeline = get_pipeline(role, get_config())
        return {
            "loaded": pipeline.loaded,
            "device": pipeline.device,
            "status": pipeline.status,
        }
    except Exception:  # noqa: BLE001
        return {"loaded": False, "device": None, "status": "unavailable"}


class DeviceChoice(BaseModel):
    primary: str = Field(pattern=r"^(auto|npu|gpu|cpu)$")
    gpu_device: str | None = None


@app.put("/api/devices")
async def choose_device(body: DeviceChoice) -> dict[str, Any]:
    """Switch the control plane to a device and reload straight away."""
    from app.models_settings import ModelsSettingsUpdate, update_models_settings
    from app.npu.pipeline import get_pipeline, reload_pipelines

    update = ModelsSettingsUpdate(primary_device=body.primary)
    if body.gpu_device:
        update.gpu_device = body.gpu_device
    try:
        update_models_settings(update)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    get_pipeline("router", get_config())
    get_pipeline("verifier", get_config())
    reload_pipelines(get_config())
    reload_assistant(get_config())
    router = get_pipeline("router", get_config())
    return {
        "primary": body.primary,
        "active": router.device,
        "loaded": router.loaded,
        "note": router.degraded_reason or router.status,
    }


@app.get("/api/models/available")
async def get_available_worker_models() -> dict[str, Any]:
    from app.worker_models import available_worker_models

    return await available_worker_models()


@app.get("/api/models/hf/targets")
async def get_hf_targets() -> dict[str, Any]:
    from app.hf_hub import targets_info

    return {"targets": targets_info()}


@app.get("/api/models/hf/search")
async def search_hf_models(
    q: str = "",
    target: str = "router",
    limit: int = 12,
) -> dict[str, Any]:
    from app.hf_hub import search_models

    if target not in {"router", "chat", "comfy"}:
        raise HTTPException(status_code=400, detail="target must be router, chat, or comfy")
    try:
        return await search_models(
            q,
            target=target,  # type: ignore[arg-type]
            limit=min(max(limit, 1), 40),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {exc}") from exc


@app.post("/api/models/hf/download")
async def download_hf_model(body: dict[str, Any]) -> dict[str, Any]:
    from app.hf_hub import HfDownloadRequest, start_download

    try:
        req = HfDownloadRequest(**body)
        return start_download(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models/hf/downloads")
async def list_hf_downloads() -> dict[str, Any]:
    from app.hf_hub import list_jobs

    return {"jobs": list_jobs()}


@app.get("/api/models/hf/downloads/{job_id}")
async def get_hf_download(job_id: str) -> dict[str, Any]:
    from app.hf_hub import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Download job not found")
    return job


@app.put("/api/models")
async def put_models_settings(update: ModelsSettingsUpdate) -> dict[str, Any]:
    try:
        return update_models_settings(update)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/reload")
async def reload_models() -> dict[str, Any]:
    """Hot-reload every OpenVINO pipeline after a model or device change."""
    from app.npu.pipeline import get_pipeline, reload_pipelines

    try:
        # Make sure both roles exist before reloading, so a verifier that was
        # never touched still picks up its new path.
        get_pipeline("router", get_config())
        get_pipeline("verifier", get_config())
        status = reload_pipelines(get_config())
        reload_assistant(get_config())
        router = get_pipeline("router", get_config())
        return {
            "status": "reloaded",
            "pipelines": status,
            "loaded": router.loaded,
            "device": router.device,
            "model_path": router.model_path,
            "detail": router.status,
            "note": router.degraded_reason,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # A plugin that just came or went changes what the assistant can do.
    reload_tool_registry(get_config())
    reload_assistant(get_config())
    return info.model_dump()


@app.put("/api/plugins/{plugin_id}/settings")
async def plugin_settings(plugin_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        info = get_plugin_registry().update_settings(plugin_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    reload_tool_registry(get_config())
    return info.model_dump()


@app.get("/api/plugins/catalog")
async def plugin_catalog() -> dict[str, Any]:
    """Everything installable, and whether it is installed."""
    entries = get_plugin_registry().catalog()
    return {
        "plugins": entries,
        "installed": sum(1 for e in entries if e["installed"]),
        "count": len(entries),
    }


@app.post("/api/plugins/catalog/{plugin_id}/install")
async def install_from_catalog(plugin_id: str) -> dict[str, Any]:
    registry = get_plugin_registry()
    try:
        info = registry.install_from_catalog(plugin_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    config = get_config()
    tools = reload_tool_registry(config)
    await tools.load_mcp_tools(force=True)
    reload_assistant(config)
    return {"status": "installed", "plugin": info.model_dump()}


@app.delete("/api/plugins/catalog/{plugin_id}")
async def uninstall_from_catalog(plugin_id: str, purge: bool = False) -> dict[str, str]:
    registry = get_plugin_registry()
    try:
        registry.uninstall_from_catalog(plugin_id, purge=purge)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    config = get_config()
    reload_tool_registry(config)
    reload_assistant(config)
    return {"status": "removed", "id": plugin_id}


# ------------------------------------------------------ skills from GitHub


class SkillDiscoverBody(BaseModel):
    repo: str = Field(min_length=3)


class SkillImportBody(BaseModel):
    repo: str = Field(min_length=3)
    paths: list[str] = Field(default_factory=list)


@app.get("/api/skills/catalog")
async def skill_catalog() -> dict[str, Any]:
    """A curated shortlist of skills worth installing, checked against source."""
    from app.skill_import import catalog_with_status

    try:
        return await catalog_with_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skill catalog check failed: %s", exc)
        from app.skill_import import load_catalog

        return {"skills": load_catalog(), "count": 0, "error": str(exc)[:200]}


@app.post("/api/skills/discover")
async def discover_skills(body: SkillDiscoverBody) -> dict[str, Any]:
    """List the skills in a GitHub repository, without installing anything."""
    from app.skill_import import SkillImportError, discover

    try:
        return await discover(body.repo)
    except SkillImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}") from exc


@app.post("/api/skills/import")
async def import_skills(body: SkillImportBody) -> dict[str, Any]:
    """Install the selected skills from a GitHub repository."""
    from app.skill_import import SkillImportError, install

    try:
        return await install(body.repo, body.paths)
    except SkillImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}") from exc


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
    config = get_config()
    tools = reload_tool_registry(config)
    mcp_tools = await tools.load_mcp_tools(force=True)
    reload_assistant(config)
    skills = reload_skill_registry()
    for plugin_id, contributed in registry.contributed_skills().items():
        skills.add_plugin_skills(plugin_id, contributed)
    global orchestrator
    if orchestrator is not None:
        orchestrator = GatewayOrchestrator(config)
    return {
        "status": "reloaded",
        "count": len(registry.list()),
        "tools": len(tools.specs()),
        "mcp_tools": mcp_tools,
        "skills": len(skills.list()),
    }


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
    return Response(
        get_theme_manager().launcher_css_text(),
        media_type="text/css",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/themes/active")
async def active_theme() -> dict[str, Any]:
    """Everything the launcher needs to draw itself: shape, colours, CSS URL."""
    manager = get_theme_manager()
    return {
        "id": manager.active_id,
        "popup": manager.active_popup().model_dump(),
        "colors": manager.active_colors(),
        "launcher_css_url": "/api/themes/active/launcher.css",
    }


@app.get("/api/themes/active/popup.json")
async def active_popup() -> dict[str, Any]:
    return get_theme_manager().active_popup().model_dump()


@app.get("/api/themes/presets")
async def theme_presets() -> dict[str, Any]:
    from app.themes import POPUP_PRESETS

    return {"presets": POPUP_PRESETS}


# --------------------------------------------------------------- assistant


@app.get("/api/assistant")
async def get_assistant_settings() -> dict[str, Any]:
    settings = load_assistant_settings()
    tools = get_tool_registry()
    return {
        "settings": settings.sanitized(),
        "system_prompt": get_assistant().system_prompt(""),
        "tool_count": len(tools.usable_specs()),
        "model_loaded": get_assistant().pipeline.loaded,
        "device": get_assistant().pipeline.device,
    }


@app.put("/api/assistant")
async def put_assistant_settings(update: AssistantSettingsUpdate) -> dict[str, Any]:
    try:
        settings = update_assistant_settings(update)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    reload_tool_registry(get_config())
    reload_assistant(get_config())
    return {"settings": settings.sanitized(), "status": "saved"}


@app.get("/api/tools")
async def list_tools(available_only: bool = False) -> dict[str, Any]:
    registry = get_tool_registry()
    specs = registry.usable_specs() if available_only else registry.specs()
    return {
        "tools": [spec.model_dump() for spec in specs],
        "count": len(specs),
        "policy": load_assistant_settings().tools.model_dump(),
    }


@app.post("/api/tools/refresh")
async def refresh_tools() -> dict[str, Any]:
    registry = reload_tool_registry(get_config())
    discovered = await registry.load_mcp_tools(force=True)
    reload_assistant(get_config())
    return {
        "status": "reloaded",
        "count": len(registry.specs()),
        "mcp_tools": discovered,
    }


@app.post("/api/tools/call")
async def call_tool(request: ToolCallRequest) -> dict[str, Any]:
    registry = get_tool_registry()
    try:
        result = await registry.call(
            request.tool, request.arguments, confirmed=request.confirmed
        )
    except ConfirmationRequired as exc:
        return {
            "ok": False,
            "requires_confirmation": True,
            "tool": exc.tool,
            "danger": exc.danger.value,
            "arguments": exc.args,
        }
    return {"requires_confirmation": False, **result.model_dump()}


class ToolPolicyBody(BaseModel):
    enabled: bool | None = None
    auto_confirm: bool | None = None


@app.post("/api/tools/{tool_name}/policy")
async def set_tool_policy_endpoint(tool_name: str, body: ToolPolicyBody) -> dict[str, Any]:
    """Switch one tool on or off, or exempt it from confirmation."""
    registry = get_tool_registry()
    if registry.get(tool_name) is None:
        raise HTTPException(status_code=404, detail=f"No tool named '{tool_name}'")
    settings = set_tool_policy(
        tool_name, enabled=body.enabled, auto_confirm=body.auto_confirm
    )
    spec = next((s for s in registry.specs() if s.name == tool_name), None)
    return {
        "tool": spec.model_dump() if spec else None,
        "policy": settings.tools.model_dump(),
    }


class SkillBody(BaseModel):
    name: str
    description: str = ""
    triggers: list[str] | str = Field(default_factory=list)
    always: bool = False
    enabled: bool = True
    content: str = ""


class SkillEnableBody(BaseModel):
    enabled: bool


@app.get("/api/skills")
async def list_skills() -> dict[str, Any]:
    registry = get_skill_registry()
    return {
        "skills": [s.model_dump() for s in registry.list()],
        "directory": str(registry.directory),
    }


@app.post("/api/skills")
async def create_skill(body: SkillBody) -> dict[str, Any]:
    try:
        skill = get_skill_registry().save(body.model_dump())
    except SkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return skill.model_dump()


@app.put("/api/skills/{name}")
async def update_skill(name: str, body: SkillBody) -> dict[str, Any]:
    registry = get_skill_registry()
    if registry.get(name) is None:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {name}")
    try:
        skill = registry.save(body.model_dump(), original=name)
    except SkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return skill.model_dump()


@app.post("/api/skills/{name}/enable")
async def enable_skill(name: str, body: SkillEnableBody) -> dict[str, Any]:
    try:
        return get_skill_registry().set_enabled(name, body.enabled).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/skills/{name}")
async def delete_skill(name: str) -> dict[str, str]:
    try:
        get_skill_registry().delete(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "removed", "name": name}


# ------------------------------------------------- projects + worker config


@app.put("/api/projects")
async def put_projects(body: ProjectsUpdate) -> dict[str, Any]:
    try:
        return write_projects(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/workers")
async def get_worker_endpoints() -> dict[str, Any]:
    return {"settings": current_worker_endpoints().model_dump()}


@app.put("/api/workers")
async def put_worker_endpoints(body: WorkerEndpointsUpdate) -> dict[str, Any]:
    try:
        return update_worker_endpoints(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/system")
async def system_overview() -> dict[str, Any]:
    """Everything the Status page needs about this install, in one call."""
    import platform
    import sys

    from app.hardware import hardware_dict
    from app.npu.pipeline import get_pipeline
    from app.skills import SKILLS_DIR

    cfg = get_config()
    registry = get_plugin_registry()
    router = get_pipeline("router", cfg)
    verifier = get_pipeline("verifier", cfg)

    try:
        hardware = hardware_dict()
    except Exception as exc:  # noqa: BLE001
        hardware = {"error": str(exc)}

    return {
        "version": app.version,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "hardware": hardware,
        "paths": {
            "root": str(cfg.root),
            "config": str(cfg.root / "config"),
            "models": str(cfg.root / "models"),
            "themes": str(cfg.root / "themes"),
            "skills": str(SKILLS_DIR),
            "plugins": str(registry.community_dir),
            "outputs": str(cfg.output_dir),
        },
        "pipelines": {
            "router": {
                "loaded": router.loaded,
                "device": router.device,
                "model_path": router.model_path,
                "status": router.status,
                "note": router.degraded_reason,
            },
            "verifier": {
                "loaded": verifier.loaded,
                "device": verifier.device,
                "model_path": verifier.model_path,
                "status": verifier.status,
                "note": verifier.degraded_reason,
            },
        },
        "counts": {
            "plugins": len(registry.list()),
            "tools": len(get_tool_registry().specs()),
            "skills": len(get_skill_registry().list()),
            "projects": len(cfg.projects),
        },
    }


@app.post("/api/skills/reload")
async def reload_skills() -> dict[str, Any]:
    registry = reload_skill_registry()
    for plugin_id, skills in get_plugin_registry().contributed_skills().items():
        registry.add_plugin_skills(plugin_id, skills)
    return {"status": "reloaded", "count": len(registry.list())}


# ------------------------------------------------------------------ speech


class SpeakBody(BaseModel):
    text: str = Field(min_length=1)
    engine: str = ""
    voice: str = ""
    rate: int | None = Field(default=None, ge=25, le=400)
    pitch: int | None = Field(default=None, ge=0, le=99)


@app.get("/api/speech")
async def speech_options() -> dict[str, Any]:
    """Which speech engines and voices this machine actually has."""
    from app.tts import probe_engines

    settings = load_assistant_settings().speech
    engines = probe_engines()
    return {
        "settings": settings.model_dump(),
        "engines": [e.model_dump() for e in engines],
        "available": any(e.available for e in engines),
    }


@app.post("/api/speech/refresh")
async def refresh_speech() -> dict[str, Any]:
    from app.tts import probe_engines

    engines = probe_engines(refresh=True)
    return {
        "engines": [e.model_dump() for e in engines],
        "available": any(e.available for e in engines),
    }


@app.post("/api/speech/speak")
async def speak_text(body: SpeakBody) -> dict[str, Any]:
    """Read text aloud on the machine running the gateway."""
    from app.tts import SpeakError, speak

    settings = load_assistant_settings().speech
    if not settings.enabled:
        raise HTTPException(
            status_code=400,
            detail="Read aloud is switched off. Enable it under Assistant → Speech.",
        )
    try:
        return await speak(
            body.text,
            engine_id=body.engine or settings.engine,
            voice_id=body.voice or settings.voice,
            rate=body.rate if body.rate is not None else settings.rate,
            pitch=body.pitch if body.pitch is not None else settings.pitch,
        )
    except SpeakError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/speech/stop")
async def stop_speech() -> dict[str, Any]:
    from app.tts import stop

    return {"stopped": await stop()}


# ---------------------------------------------------------------- activity


@app.get("/api/activity")
async def activity() -> dict[str, Any]:
    return get_activity_bus().snapshot().model_dump()


@app.get("/api/events")
async def events() -> StreamingResponse:
    """Server-Sent Events for the tray indicator and the control panel."""
    return StreamingResponse(
        get_activity_bus().sse_stream(shutting_down),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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


@app.post("/api/tasks/{task_id}/approve", response_model=TaskResponse)
async def approve_task(task_id: str) -> TaskResponse:
    """Let a task waiting on confirmation go ahead.

    The same thing a re-POST to /api/chat with ``confirmed`` does, named for
    what it means so the control panel does not have to re-send the original
    message just to say yes.
    """
    return await _orch().approve(task_id)


@app.post("/api/tasks/{task_id}/deny", response_model=TaskResponse)
async def deny_task(task_id: str) -> TaskResponse:
    """Refuse a task waiting on confirmation.

    Distinct from cancel only in what it records: "denied" says a person was
    asked and said no, which is worth telling apart from a task that was
    stopped or timed out.
    """
    result = await _orch().cancel(task_id, reason="Denied")
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
