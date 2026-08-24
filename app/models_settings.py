"""Model selection, primary device, and login autostart settings."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import ROOT, get_config, reload_config
from app.hardware import detect_hardware, hardware_dict
from app.models_catalog import recommendations

logger = logging.getLogger(__name__)

MODELS_TOML = ROOT / "config" / "models.toml"

PrimaryDevice = Literal["auto", "npu", "gpu", "cpu"]
WorkerModelMode = Literal["auto", "fixed"]
PreferredChatWorker = Literal["auto", "lmstudio", "lemonade"]


class ModelsSettings(BaseModel):
    primary_device: PrimaryDevice = "auto"
    npu_device: str = "NPU"
    gpu_device: str = "GPU"
    fallback_device: str = "CPU"
    router_model_id: str = "qwen2.5-1.5b-instruct-int4"
    router_model_path: str = "./models/router"
    verifier_model_id: str = ""
    verifier_model_path: str = ""
    # Legacy aliases kept in sync with lmstudio_* / preferred_chat_worker
    chat_model_id: str = ""
    chat_backend: str = "lmstudio"
    lmstudio_mode: WorkerModelMode = "auto"
    lmstudio_model: str = ""
    lemonade_mode: WorkerModelMode = "auto"
    lemonade_model: str = ""
    lemonade_base_url: str = "http://127.0.0.1:13305/api/v1"
    comfyui_mode: WorkerModelMode = "auto"
    comfyui_model: str = ""
    preferred_chat_worker: PreferredChatWorker = "auto"
    whisper_model: str = "tiny"
    autostart_gateway: bool = True
    autostart_launcher: bool = False
    open_panel_on_login: bool = False


class ModelsSettingsUpdate(BaseModel):
    primary_device: PrimaryDevice | None = None
    npu_device: str | None = None
    gpu_device: str | None = None
    fallback_device: str | None = None
    router_model_id: str | None = None
    router_model_path: str | None = None
    verifier_model_id: str | None = None
    verifier_model_path: str | None = None
    chat_model_id: str | None = None
    chat_backend: str | None = None
    lmstudio_mode: WorkerModelMode | None = None
    lmstudio_model: str | None = None
    lemonade_mode: WorkerModelMode | None = None
    lemonade_model: str | None = None
    lemonade_base_url: str | None = None
    comfyui_mode: WorkerModelMode | None = None
    comfyui_model: str | None = None
    preferred_chat_worker: PreferredChatWorker | None = None
    whisper_model: str | None = Field(default=None, pattern=r"^(tiny|base|small)$")
    autostart_gateway: bool | None = None
    autostart_launcher: bool | None = None
    open_panel_on_login: bool | None = None


def _default_toml() -> str:
    return """# Keylane model + device preferences
[devices]
primary = "auto"          # auto | npu | gpu | cpu
npu_device = "NPU"
gpu_device = "GPU"
fallback = "CPU"

[router]
model_id = "qwen2.5-1.5b-instruct-int4"
model_path = "./models/router"

[verifier]
model_id = ""
model_path = ""

# Preferred chat worker when the router is choosing among local LLMs
[chat]
preferred_worker = "auto"  # auto | lmstudio | lemonade

[lmstudio]
mode = "auto"              # auto = router picks best loaded model | fixed = always use model=
model = ""

[lemonade]
mode = "auto"
model = ""
base_url = "http://127.0.0.1:13305/api/v1"

[comfyui]
mode = "auto"
model = ""                 # checkpoint / UNET / diffusion model filename

[speech]
whisper_model = "tiny"    # tiny | base | small

[startup]
gateway = true
launcher = false
open_panel = false
"""


def _norm_mode(value: Any, default: WorkerModelMode = "auto") -> WorkerModelMode:
    mode = str(value or default).strip().lower()
    return "fixed" if mode == "fixed" else "auto"


def _norm_preferred(value: Any) -> PreferredChatWorker:
    v = str(value or "auto").strip().lower()
    if v in {"lmstudio", "lemonade", "auto"}:
        return v  # type: ignore[return-value]
    if v == "openvino-gpu":
        return "lmstudio"
    return "auto"


def _ensure_file() -> None:
    MODELS_TOML.parent.mkdir(parents=True, exist_ok=True)
    if not MODELS_TOML.exists():
        MODELS_TOML.write_text(_default_toml(), encoding="utf-8")


def _load_raw() -> dict[str, Any]:
    import tomllib

    _ensure_file()
    with MODELS_TOML.open("rb") as fh:
        return tomllib.load(fh)


def load_models_settings() -> ModelsSettings:
    raw = _load_raw()
    devices = raw.get("devices", {})
    router = raw.get("router", {})
    verifier = raw.get("verifier", {})
    chat = raw.get("chat", {})
    lmstudio = raw.get("lmstudio", {})
    lemonade = raw.get("lemonade", {})
    comfyui = raw.get("comfyui", {})
    speech = raw.get("speech", {})
    startup = raw.get("startup", {})
    primary = str(devices.get("primary", "auto")).lower()
    if primary not in {"auto", "npu", "gpu", "cpu"}:
        primary = "auto"

    # Migrate legacy [chat] model_id / backend into lmstudio_* fields
    lm_mode = _norm_mode(lmstudio.get("mode"), "auto")
    lm_model = str(lmstudio.get("model", "") or "")
    legacy_chat_model = str(chat.get("model_id", "") or "")
    if not lm_model and legacy_chat_model:
        lm_model = legacy_chat_model
        if lm_model and lm_mode == "auto" and "mode" not in lmstudio:
            # Old configs pinned a preferred chat model — treat as fixed until user switches
            lm_mode = "fixed"

    preferred = _norm_preferred(
        chat.get("preferred_worker") or chat.get("backend") or "auto"
    )

    return ModelsSettings(
        primary_device=primary,  # type: ignore[arg-type]
        npu_device=str(devices.get("npu_device", "NPU")),
        gpu_device=str(devices.get("gpu_device", "GPU")),
        fallback_device=str(devices.get("fallback", "CPU")),
        router_model_id=str(router.get("model_id", "qwen2.5-1.5b-instruct-int4")),
        router_model_path=str(router.get("model_path", "./models/router")),
        verifier_model_id=str(verifier.get("model_id", "")),
        verifier_model_path=str(verifier.get("model_path", "")),
        chat_model_id=lm_model,
        chat_backend=preferred if preferred != "auto" else "lmstudio",
        lmstudio_mode=lm_mode,
        lmstudio_model=lm_model,
        lemonade_mode=_norm_mode(lemonade.get("mode"), "auto"),
        lemonade_model=str(lemonade.get("model", "") or ""),
        lemonade_base_url=str(
            lemonade.get("base_url", "http://127.0.0.1:13305/api/v1")
            or "http://127.0.0.1:13305/api/v1"
        ),
        comfyui_mode=_norm_mode(comfyui.get("mode"), "auto"),
        comfyui_model=str(comfyui.get("model", "") or ""),
        preferred_chat_worker=preferred,
        whisper_model=str(speech.get("whisper_model", "tiny")),
        autostart_gateway=bool(startup.get("gateway", True)),
        autostart_launcher=bool(startup.get("launcher", False)),
        open_panel_on_login=bool(startup.get("open_panel", False)),
    )


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def save_models_settings(settings: ModelsSettings) -> None:
    # Keep legacy chat aliases aligned
    lm_model = settings.lmstudio_model or settings.chat_model_id
    preferred = settings.preferred_chat_worker
    text = f"""# Keylane model + device preferences
[devices]
primary = "{settings.primary_device}"
npu_device = "{_escape(settings.npu_device)}"
gpu_device = "{_escape(settings.gpu_device)}"
fallback = "{_escape(settings.fallback_device)}"

[router]
model_id = "{_escape(settings.router_model_id)}"
model_path = "{_escape(settings.router_model_path)}"

[verifier]
model_id = "{_escape(settings.verifier_model_id)}"
model_path = "{_escape(settings.verifier_model_path)}"

[chat]
preferred_worker = "{preferred}"

[lmstudio]
mode = "{settings.lmstudio_mode}"
model = "{_escape(lm_model)}"

[lemonade]
mode = "{settings.lemonade_mode}"
model = "{_escape(settings.lemonade_model)}"
base_url = "{_escape(settings.lemonade_base_url)}"

[comfyui]
mode = "{settings.comfyui_mode}"
model = "{_escape(settings.comfyui_model)}"

[speech]
whisper_model = "{_escape(settings.whisper_model)}"

[startup]
gateway = {"true" if settings.autostart_gateway else "false"}
launcher = {"true" if settings.autostart_launcher else "false"}
open_panel = {"true" if settings.open_panel_on_login else "false"}
"""
    _ensure_file()
    MODELS_TOML.write_text(text, encoding="utf-8")


def _systemctl(*args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def _unit_enabled(unit: str) -> bool:
    code, out = _systemctl("is-enabled", unit)
    return code == 0 and "enabled" in out


def _set_unit_enabled(unit: str, enabled: bool) -> str:
    if enabled:
        code, out = _systemctl("enable", "--now", unit)
    else:
        # Keep running if already up; only disable on login
        code, out = _systemctl("disable", unit)
    if code != 0:
        logger.warning("systemctl %s %s failed: %s", "enable" if enabled else "disable", unit, out)
    return out


def _panel_desktop_path() -> Path:
    return Path.home() / ".config" / "autostart" / "keylane-panel.desktop"


def _write_panel_autostart(enabled: bool, port: int) -> None:
    path = _panel_desktop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not enabled:
        if path.exists():
            path.unlink()
        return
    path.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=Keylane Control Panel",
                "Comment=Open the Keylane local control panel in your browser",
                f"Exec=xdg-open http://127.0.0.1:{port}/",
                "X-GNOME-Autostart-enabled=true",
                "Hidden=false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def apply_autostart(settings: ModelsSettings) -> dict[str, Any]:
    cfg = get_config()
    gateway_out = _set_unit_enabled("ai-gateway.service", settings.autostart_gateway)
    launcher_out = _set_unit_enabled("ai-launcher.service", settings.autostart_launcher)
    _write_panel_autostart(settings.open_panel_on_login, cfg.gateway.port)
    return {
        "gateway_enabled": _unit_enabled("ai-gateway.service"),
        "launcher_enabled": _unit_enabled("ai-launcher.service"),
        "panel_autostart": _panel_desktop_path().exists(),
        "detail": {
            "gateway": gateway_out,
            "launcher": launcher_out,
        },
    }


def resolve_openvino_device(settings: ModelsSettings | None = None) -> str:
    """Pick the OpenVINO device for the router/verifier control plane."""
    settings = settings or load_models_settings()
    try:
        import openvino as ov

        available = list(ov.Core().available_devices)
    except Exception:  # noqa: BLE001
        available = []

    primary = settings.primary_device
    npu = settings.npu_device
    gpu = settings.gpu_device
    fallback = settings.fallback_device

    def pick(*candidates: str) -> str | None:
        for name in candidates:
            if name in available:
                return name
        # fuzzy: GPU.0 etc.
        for name in candidates:
            for device in available:
                if device == name or device.startswith(f"{name}."):
                    return device
        return None

    if primary == "npu":
        return pick(npu, fallback, "CPU") or "CPU"
    if primary == "gpu":
        return pick(gpu, fallback, "CPU") or "CPU"
    if primary == "cpu":
        return pick("CPU", fallback) or "CPU"
    # auto: prefer NPU control plane, then GPU, then CPU
    return pick(npu, gpu, fallback, "CPU") or "CPU"


def update_models_settings(update: ModelsSettingsUpdate) -> dict[str, Any]:
    current = load_models_settings()
    data = update.model_dump(exclude_none=True)
    # Alias sync: chat_model_id <-> lmstudio_model, chat_backend <-> preferred
    if "lmstudio_model" in data and "chat_model_id" not in data:
        data["chat_model_id"] = data["lmstudio_model"]
    if "chat_model_id" in data and "lmstudio_model" not in data:
        data["lmstudio_model"] = data["chat_model_id"]
    if "preferred_chat_worker" in data and "chat_backend" not in data:
        pref = data["preferred_chat_worker"]
        data["chat_backend"] = pref if pref != "auto" else "lmstudio"
    if "chat_backend" in data and "preferred_chat_worker" not in data:
        backend = str(data["chat_backend"]).lower()
        if backend in {"lmstudio", "lemonade"}:
            data["preferred_chat_worker"] = backend
        elif backend == "auto":
            data["preferred_chat_worker"] = "auto"
    merged = current.model_copy(update=data)
    # Keep aliases coherent on the merged object
    merged.chat_model_id = merged.lmstudio_model
    if merged.preferred_chat_worker != "auto":
        merged.chat_backend = merged.preferred_chat_worker
    save_models_settings(merged)

    # Keep workers.toml in sync for the running workers
    _sync_workers_npu(merged)
    autostart = apply_autostart(merged)
    reload_config()
    return {
        "settings": merged.model_dump(),
        "resolved_device": resolve_openvino_device(merged),
        "autostart": autostart,
        "restart_required": True,
        "note": (
            "Saved. Restart ai-gateway.service to reload OpenVINO models: "
            "systemctl --user restart ai-gateway.service"
        ),
    }


def _sync_workers_npu(settings: ModelsSettings) -> None:
    path = ROOT / "config" / "workers.toml"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    device = resolve_openvino_device(settings)
    preferred = settings.npu_device if settings.primary_device in {"auto", "npu"} else device

    def repl_in_section(section: str, key: str, value: str) -> None:
        nonlocal text
        pattern = rf'(?ms)(\[{re.escape(section)}\][^\[]*?)(^{re.escape(key)}\s*=\s*)"[^"]*"'
        text2, n = re.subn(pattern, rf'\1\2"{value}"', text, count=1)
        if n:
            text = text2
            return
        # Insert key under section header if missing
        header = f"[{section}]"
        if header in text:
            text = text.replace(header, f'{header}\n{key} = "{value}"', 1)
        else:
            text = text.rstrip() + f'\n\n{header}\n{key} = "{value}"\n'

    repl_in_section("npu", "model_path", settings.router_model_path)
    repl_in_section(
        "npu",
        "device",
        preferred if settings.primary_device != "gpu" else settings.gpu_device,
    )
    repl_in_section("npu", "fallback_device", settings.fallback_device)

    lm_default = (
        settings.lmstudio_model
        if settings.lmstudio_mode == "fixed" and settings.lmstudio_model
        else "local-model"
    )
    repl_in_section("lmstudio", "default_model", lm_default)

    lem_default = (
        settings.lemonade_model
        if settings.lemonade_mode == "fixed" and settings.lemonade_model
        else "auto"
    )
    repl_in_section("lemonade", "default_model", lem_default)
    repl_in_section("lemonade", "base_url", settings.lemonade_base_url)

    comfy_default = (
        settings.comfyui_model
        if settings.comfyui_mode == "fixed" and settings.comfyui_model
        else "auto"
    )
    repl_in_section("comfyui", "default_model", comfy_default)

    path.write_text(text, encoding="utf-8")


def models_overview() -> dict[str, Any]:
    settings = load_models_settings()
    hw = detect_hardware()
    rec = recommendations(hw)
    from app.worker_models import policy_summary

    cfg = get_config()
    return {
        "settings": settings.model_dump(),
        "resolved_device": resolve_openvino_device(settings),
        "hardware": hardware_dict(),
        # Stored paths are relative so the config stays portable, but "./models"
        # reads as the current folder — which is the source checkout, not the
        # install the service runs from. Say where that actually is.
        "paths": {
            "root": str(cfg.root),
            "models": str(cfg.root / "models"),
            "router": str(cfg.root / "models" / "router"),
            "chat": str(cfg.root / "models" / "chat"),
            "resolved_router": str(cfg.resolve_path(settings.router_model_path)),
        },
        "recommendations": rec,
        "worker_policy": policy_summary(),
        "autostart": {
            "gateway_enabled": _unit_enabled("ai-gateway.service"),
            "launcher_enabled": _unit_enabled("ai-launcher.service"),
            "panel_autostart": _panel_desktop_path().exists(),
        },
    }
