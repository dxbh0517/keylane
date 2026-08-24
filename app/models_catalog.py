"""Curated model catalog and hardware-aware recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.config import ROOT
from app.hardware import HardwareProfile, detect_hardware

Role = Literal["router", "verifier", "chat", "vision"]
Backend = Literal["openvino", "lmstudio", "either"]


@dataclass
class CatalogModel:
    id: str
    name: str
    role: Role
    backend: Backend
    size_hint: str
    quant: str
    min_vram_mb: int = 0
    needs_npu: bool = False
    needs_gpu: bool = False
    path_hint: str = ""
    hf_url: str = ""
    gated: bool = False
    """The repository needs a licence accepted on Hugging Face first.

    Offering a download button for one of these gets a 401 and no
    explanation, so the panel links to the model page instead.
    """
    notes: str = ""
    tags: list[str] = field(default_factory=list)


# OpenVINO GenAI exports suitable for NPU / CPU control-plane routing.
ROUTER_MODELS: list[CatalogModel] = [
    CatalogModel(
        id="qwen2.5-1.5b-instruct-int4",
        name="Qwen2.5 1.5B Instruct (INT4)",
        role="router",
        backend="openvino",
        size_hint="~1.0 GB",
        quant="INT4",
        needs_npu=False,
        path_hint="./models/router/qwen2.5-1.5b-instruct-int4",
        hf_url="https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        notes="Best default for Intel NPU routing — fast and small.",
        tags=["recommended", "npu", "default"],
    ),
    CatalogModel(
        id="llama-3.2-1b-instruct-int4",
        name="Llama 3.2 1B Instruct (INT4)",
        role="router",
        backend="openvino",
        size_hint="~0.8 GB",
        quant="INT4",
        path_hint="./models/router/llama-3.2-1b-instruct-int4",
        hf_url="https://huggingface.co/OpenVINO/Llama-3.2-1B-Instruct-int4-ov",
        gated=True,
        notes="Ultra-light router for lowest latency on NPU.",
        tags=["npu", "fast"],
    ),
    CatalogModel(
        id="llama-3.2-3b-instruct-int4",
        name="Llama 3.2 3B Instruct (INT4)",
        role="router",
        backend="openvino",
        size_hint="~1.8 GB",
        quant="INT4",
        path_hint="./models/router/llama-3.2-3b-instruct-int4",
        hf_url="https://huggingface.co/OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
        gated=True,
        notes="Higher-quality routing/verification when NPU has headroom.",
        tags=["npu", "quality"],
    ),
    CatalogModel(
        id="phi-3.5-mini-instruct-int4",
        name="Phi-3.5 Mini Instruct (INT4)",
        role="router",
        backend="openvino",
        size_hint="~2.0 GB",
        quant="INT4",
        path_hint="./models/router/phi-3.5-mini-instruct-int4",
        hf_url="https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-ov",
        notes="Strong instruction following for structured JSON routing.",
        tags=["npu", "structured"],
    ),
    CatalogModel(
        id="qwen2.5-3b-instruct-int4",
        name="Qwen2.5 3B Instruct (INT4)",
        role="router",
        backend="openvino",
        size_hint="~1.9 GB",
        quant="INT4",
        path_hint="./models/router/qwen2.5-3b-instruct-int4",
        hf_url="https://huggingface.co/OpenVINO/Qwen2.5-3B-Instruct-int4-ov",
        gated=True,
        notes="Good NPU/GPU OpenVINO option if 1.5B is too weak.",
        tags=["npu", "gpu"],
    ),
]

CHAT_MODELS: list[CatalogModel] = [
    CatalogModel(
        id="qwen2.5-coder-14b",
        name="Qwen2.5 Coder 14B",
        role="chat",
        backend="lmstudio",
        size_hint="~9–10 GB Q4",
        quant="Q4_K_M",
        min_vram_mb=10000,
        needs_gpu=True,
        hf_url="https://huggingface.co/lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF",
        notes="Excellent local coding model for RTX-class GPUs.",
        tags=["coding", "recommended"],
    ),
    CatalogModel(
        id="qwen2.5-32b-instruct",
        name="Qwen2.5 32B Instruct",
        role="chat",
        backend="lmstudio",
        size_hint="~18–20 GB Q4",
        quant="Q4_K_M",
        min_vram_mb=18000,
        needs_gpu=True,
        hf_url="https://huggingface.co/lmstudio-community/Qwen2.5-32B-Instruct-GGUF",
        notes="High-quality general chat on 20GB+ VRAM.",
        tags=["chat", "quality"],
    ),
    CatalogModel(
        id="llama-3.1-70b-instruct",
        name="Llama 3.1 70B Instruct",
        role="chat",
        backend="lmstudio",
        size_hint="~20–22 GB Q3/Q4",
        quant="Q3_K_M / Q4_K_S",
        min_vram_mb=20000,
        needs_gpu=True,
        hf_url="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
        notes="Fits a 24GB laptop GPU at aggressive quant — flagship local chat.",
        tags=["chat", "flagship", "recommended"],
    ),
    CatalogModel(
        id="deepseek-coder-v2-lite",
        name="DeepSeek Coder V2 Lite",
        role="chat",
        backend="lmstudio",
        size_hint="~10–12 GB Q4",
        quant="Q4_K_M",
        min_vram_mb=10000,
        needs_gpu=True,
        hf_url="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        notes="Strong coding specialist; good LM Studio companion.",
        tags=["coding"],
    ),
    CatalogModel(
        id="mistral-small-3.1",
        name="Mistral Small 3.1",
        role="chat",
        backend="lmstudio",
        size_hint="~12–14 GB Q4",
        quant="Q4_K_M",
        min_vram_mb=12000,
        needs_gpu=True,
        hf_url="https://huggingface.co/lmstudio-community/Mistral-Small-3.1-24B-Instruct-2503-GGUF",
        notes="Balanced speed/quality general assistant.",
        tags=["chat"],
    ),
    CatalogModel(
        id="gemma-3-12b",
        name="Gemma 3 12B",
        role="chat",
        backend="lmstudio",
        size_hint="~8–10 GB Q4",
        quant="Q4_K_M",
        min_vram_mb=9000,
        needs_gpu=True,
        hf_url="https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF",
        notes="Fast mid-size chat model for everyday use.",
        tags=["chat", "fast"],
    ),
    CatalogModel(
        id="qwen2.5-7b-instruct",
        name="Qwen2.5 7B Instruct",
        role="chat",
        backend="either",
        size_hint="~4–5 GB Q4 / INT4",
        quant="Q4 / INT4",
        min_vram_mb=5000,
        hf_url="https://huggingface.co/lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
        notes="Works on lighter GPUs; OpenVINO INT4 can also run on Intel GPU.",
        tags=["chat", "light"],
    ),
]


def missing_weights(path: Path) -> list[str]:
    """Graph files in *path* whose weights are absent.

    An OpenVINO IR is a pair: ``foo.xml`` describes the graph, ``foo.bin``
    holds the numbers. An interrupted download leaves the small .xml behind
    and the large .bin unfinished, so the directory looks like a model and
    cannot load. Naming the missing file is the difference between "download
    a model" and "resume the one you have".
    """
    missing: list[str] = []
    for xml in sorted(path.glob("*.xml")):
        weights = xml.with_suffix(".bin")
        # A few exports genuinely inline their weights; those have no .bin
        # anywhere in the directory, which is not the same as a lost one.
        if not weights.exists() and any(path.glob("*.bin")):
            missing.append(weights.name)
    return missing


def _looks_like_openvino_model(path: Path) -> bool:
    """True when *path* is a usable OpenVINO GenAI export directory."""
    if not path.is_dir():
        return False
    if not _has_graph(path):
        return False
    return not missing_weights(path)


def _has_graph(path: Path) -> bool:
    """True when *path* holds an OpenVINO graph, complete or not."""
    if not path.is_dir():
        return False
    if (path / "openvino_model.xml").exists() or (path / "openvino_language_model.xml").exists():
        return True
    if not any(path.glob("*.xml")):
        return False
    return (path / "config.json").exists() or any(path.glob("*.bin"))


def installed_router_models() -> list[dict[str, Any]]:
    """Scan local disk for downloaded OpenVINO router/verifier models.

    Only returns models that are actually present under ``models/`` — used to
    populate the Models page dropdown (never the curated wish-list catalog).
    """
    catalog_by_id = {m.id: m for m in ROUTER_MODELS}
    candidates: list[Path] = []
    router_root = ROOT / "models" / "router"
    if router_root.is_dir():
        candidates.extend(p for p in router_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    models_root = ROOT / "models"
    if models_root.is_dir():
        skip = {"router", "chat", "comfyui", "comfy", "diffusion_models", "checkpoints", "unet"}
        for p in models_root.iterdir():
            if p.is_dir() and p.name not in skip and not p.name.startswith("."):
                candidates.append(p)

    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(candidates, key=lambda p: p.name.lower()):
        if path.name in seen or not _has_graph(path):
            continue
        seen.add(path.name)
        cat = catalog_by_id.get(path.name)
        rel = path.relative_to(ROOT).as_posix()
        # An interrupted download is listed, not hidden: the folder exists, so
        # hiding it just makes the download look like it did nothing.
        absent = missing_weights(path)
        found.append(
            {
                "id": path.name,
                "name": cat.name if cat else path.name,
                "path": f"./{rel}",
                "absolute": str(path),
                "backend": "openvino",
                "size_hint": cat.size_hint if cat else "",
                "quant": cat.quant if cat else "",
                "installed": True,
                "ready": not absent,
                "missing": absent,
                # Folder names are the repo id with "/" as "__" (see
                # hf_hub.target_dir), so a resume needs no extra state.
                "repo_id": path.name.replace("__", "/"),
            }
        )
    return found


def _installed_models() -> list[dict[str, Any]]:
    return installed_router_models()


def _score_router(model: CatalogModel, hw: HardwareProfile) -> tuple[int, str]:
    score = 50
    reason = []
    if hw.has_npu and "npu" in model.tags:
        score += 25
        reason.append("NPU-friendly")
    if "recommended" in model.tags or "default" in model.tags:
        score += 20
        reason.append("default pick")
    if "fast" in model.tags:
        score += 5
    if not hw.has_npu and model.needs_npu:
        score -= 40
        reason.append("needs NPU")
    return score, ", ".join(reason) or "compatible"


def _score_chat(model: CatalogModel, hw: HardwareProfile) -> tuple[int, str]:
    score = 40
    reason = []
    vram = hw.nvidia_vram_mb or 0
    if model.min_vram_mb and vram and vram >= model.min_vram_mb:
        score += 30
        reason.append(f"fits ~{vram} MB VRAM")
    elif model.min_vram_mb and vram and vram < model.min_vram_mb:
        score -= 50
        reason.append("may not fit VRAM")
    if "recommended" in model.tags:
        score += 15
        reason.append("recommended")
    if hw.tier == "high_vram" and "flagship" in model.tags:
        score += 20
        reason.append("flagship for high VRAM")
    if not hw.has_gpu and model.needs_gpu:
        score -= 40
        reason.append("needs GPU")
    return score, ", ".join(reason) or "compatible"


def recommendations(hw: HardwareProfile | None = None) -> dict[str, Any]:
    hw = hw or detect_hardware()
    router = []
    for model in ROUTER_MODELS:
        score, reason = _score_router(model, hw)
        item = asdict(model)
        item["score"] = score
        item["reason"] = reason
        item["recommended"] = score >= 60
        router.append(item)
    router.sort(key=lambda m: m["score"], reverse=True)

    chat = []
    for model in CHAT_MODELS:
        score, reason = _score_chat(model, hw)
        item = asdict(model)
        item["score"] = score
        item["reason"] = reason
        item["recommended"] = score >= 60
        chat.append(item)
    chat.sort(key=lambda m: m["score"], reverse=True)

    primary_suggestion = "auto"
    if hw.has_npu and hw.has_gpu:
        primary_suggestion = "npu"
    elif hw.has_npu:
        primary_suggestion = "npu"
    elif hw.has_gpu:
        primary_suggestion = "gpu"
    else:
        primary_suggestion = "cpu"

    installed = installed_router_models()
    installed_ids = {m["id"] for m in installed}
    installed_paths = {m["path"] for m in installed}
    for item in router:
        item["installed"] = item["id"] in installed_ids or item.get("path_hint") in installed_paths

    return {
        "hardware": asdict(hw),
        "primary_suggestion": primary_suggestion,
        "router": router,
        "verifier": router,  # same pool; can diverge later
        "chat": chat,
        "installed": installed,
        "guidance": _guidance(hw),
    }


def _guidance(hw: HardwareProfile) -> list[str]:
    tips: list[str] = []
    if hw.has_npu:
        tips.append(
            "Use a 1B–3B INT4 OpenVINO model on the NPU for routing/verification "
            "(control plane). Keep large chat models on the GPU via LM Studio."
        )
    if hw.nvidia_vram_mb and hw.nvidia_vram_mb >= 20000:
        tips.append(
            f"Your GPU has ~{hw.nvidia_vram_mb} MB VRAM — 70B Q3/Q4 or 32B Q4 chat "
            "models are realistic in LM Studio."
        )
    elif hw.nvidia_vram_mb and hw.nvidia_vram_mb >= 10000:
        tips.append(
            f"With ~{hw.nvidia_vram_mb} MB VRAM, prefer 7B–14B Q4 chat models for headroom."
        )
    if not hw.has_npu:
        tips.append(
            "No OpenVINO NPU detected — router will fall back to CPU/heuristic until "
            "NPU userspace is available."
        )
    tips.append(
        "Place OpenVINO GenAI exports under models/router/<name>/ "
        "(openvino_model.xml + .bin). Chat models are loaded in LM Studio separately."
    )
    return tips


def catalog_dict() -> dict[str, Any]:
    return {
        "router": [asdict(m) for m in ROUTER_MODELS],
        "chat": [asdict(m) for m in CHAT_MODELS],
    }
