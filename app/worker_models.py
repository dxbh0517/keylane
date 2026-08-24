"""List and resolve per-worker models (fixed default vs router auto-pick)."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import httpx

from app.config import AppConfig, get_config
from app.models_settings import WorkerModelMode, load_models_settings
from app.schemas import RouteDecision

logger = logging.getLogger(__name__)

ChatWorker = Literal["lmstudio", "lemonade"]


async def list_openai_models(base_url: str, *, timeout: float = 3.0) -> list[dict[str, str]]:
    url = base_url.rstrip("/")
    if not url.endswith("/v1") and not url.endswith("/api/v1"):
        # tolerate bare host
        url = f"{url}/v1"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/models")
            response.raise_for_status()
            data = response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("Model list failed for %s: %s", url, exc)
        return []
    out: list[dict[str, str]] = []
    for item in data:
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        # Skip embedding-only entries for chat pickers
        lower = mid.lower()
        kind = "embedding" if "embed" in lower else "chat"
        out.append({"id": mid, "name": mid, "kind": kind})
    return out


async def list_comfy_models(base_url: str, *, timeout: float = 3.0) -> list[dict[str, str]]:
    root = base_url.rstrip("/")
    collected: list[dict[str, str]] = []
    seen: set[str] = set()
    async with httpx.AsyncClient(timeout=timeout) as client:
        for folder, kind in (
            ("diffusion_models", "diffusion"),
            ("unet", "unet"),
            ("checkpoints", "checkpoint"),
        ):
            try:
                response = await client.get(f"{root}/models/{folder}")
                if response.status_code >= 400:
                    continue
                items = response.json()
                if not isinstance(items, list):
                    continue
                for name in items:
                    mid = str(name).strip()
                    if not mid or mid in seen:
                        continue
                    seen.add(mid)
                    collected.append({"id": mid, "name": mid, "kind": kind, "folder": folder})
            except Exception as exc:  # noqa: BLE001
                logger.debug("Comfy model list %s failed: %s", folder, exc)
    return collected


async def available_worker_models(config: AppConfig | None = None) -> dict[str, list[dict[str, Any]]]:
    """Models that are actually downloaded / exposed by each worker.

    ``router`` comes from on-disk OpenVINO exports; chat/image lists come from
    the live LM Studio / Lemonade / ComfyUI APIs.
    """
    from app.models_catalog import installed_router_models

    cfg = config or get_config()
    settings = load_models_settings()
    lmstudio = await list_openai_models(cfg.lmstudio.base_url)
    lemonade = await list_openai_models(settings.lemonade_base_url or cfg.lemonade.base_url)
    comfyui = await list_comfy_models(cfg.comfyui.base_url)
    return {
        "router": installed_router_models(),
        "lmstudio": [m for m in lmstudio if m.get("kind") != "embedding"],
        "lemonade": [m for m in lemonade if m.get("kind") != "embedding"],
        "comfyui": comfyui,
    }


def _ids(models: list[dict[str, str]]) -> list[str]:
    return [m["id"] for m in models if m.get("id")]


def _score_chat_model(model_id: str, *, intent: str, instruction: str) -> int:
    text = f"{intent} {instruction}".lower()
    mid = model_id.lower()
    score = 10
    if "embed" in mid:
        return -100
    if any(k in text for k in ("code", "coding", "refactor", "bug", "typescript", "react", "implement")):
        if any(k in mid for k in ("coder", "code", "devstral", "deepseek")):
            score += 40
        if "instruct" in mid:
            score += 5
    if any(k in text for k in ("summar", "brainstorm", "analy", "write", "explain")):
        if any(k in mid for k in ("instruct", "chat", "gemma", "llama", "qwen", "mistral")):
            score += 15
    # Prefer mid-size names when unknown; penalize tiny toy ids lightly
    if any(k in mid for k in ("70b", "72b", "65b", "34b", "32b", "35b")):
        score += 8
    if any(k in mid for k in ("1b", "0.5b", "tiny")):
        score -= 5
    return score


def _score_comfy_model(model_id: str, *, intent: str, instruction: str) -> int:
    text = f"{intent} {instruction}".lower()
    mid = model_id.lower()
    score = 10
    if "flux" in mid:
        score += 30
    if "inpaint" in text and "inpaint" in mid:
        score += 20
    if "edit" in text and "edit" in mid:
        score += 15
    if mid.endswith(".safetensors"):
        score += 2
    return score


def heuristic_pick(
    worker: str,
    models: list[dict[str, str]],
    *,
    intent: str = "general_question",
    instruction: str = "",
    preferred: str | None = None,
) -> str | None:
    ids = _ids(models)
    if not ids:
        return preferred if preferred and preferred not in {"", "auto", "local-model", "local"} else None
    if preferred and preferred in ids:
        return preferred
    if worker == "comfyui":
        ranked = sorted(ids, key=lambda m: _score_comfy_model(m, intent=intent, instruction=instruction), reverse=True)
    else:
        ranked = sorted(ids, key=lambda m: _score_chat_model(m, intent=intent, instruction=instruction), reverse=True)
    return ranked[0] if ranked else None


def resolve_model_for_decision(
    decision: RouteDecision,
    *,
    available: dict[str, list[dict[str, str]]] | None = None,
) -> RouteDecision:
    """Apply fixed/auto policy onto decision.model (and comfy arguments)."""
    settings = load_models_settings()
    worker = decision.worker
    if worker not in {"lmstudio", "lemonade", "comfyui"}:
        return decision

    if worker == "lmstudio":
        mode: WorkerModelMode = settings.lmstudio_mode
        fixed = settings.lmstudio_model
    elif worker == "lemonade":
        mode = settings.lemonade_mode
        fixed = settings.lemonade_model
    else:
        mode = settings.comfyui_mode
        fixed = settings.comfyui_model

    models = (available or {}).get(worker) or []
    chosen: str | None = None

    if mode == "fixed":
        chosen = fixed.strip() if fixed.strip() else None
        if not chosen or chosen in {"auto", "local-model", "local"}:
            chosen = heuristic_pick(
                worker,
                models,
                intent=decision.intent,
                instruction=decision.instruction,
            )
        # Otherwise keep the pinned id even if it is not currently listed/loaded.
    else:
        # auto — prefer router-chosen model when valid
        router_pick = (decision.model or "").strip() or str(
            (decision.arguments or {}).get("model")
            or (decision.arguments or {}).get("ckpt_name")
            or (decision.arguments or {}).get("unet_name")
            or ""
        ).strip()
        if router_pick and (not models or router_pick in _ids(models)):
            chosen = router_pick
        else:
            chosen = heuristic_pick(
                worker,
                models,
                intent=decision.intent,
                instruction=decision.instruction,
                preferred=router_pick or None,
            )

    if not chosen:
        return decision

    data = decision.model_dump()
    data["model"] = chosen
    args = dict(data.get("arguments") or {})
    if worker == "comfyui":
        args.setdefault("model", chosen)
        lower = chosen.lower()
        if lower.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".gguf")) or "/" in chosen:
            # Prefer unet/diffusion field for flux-style graphs; also set ckpt for checkpoint loaders
            args.setdefault("unet_name", chosen)
            args.setdefault("ckpt_name", chosen)
    else:
        args.setdefault("model", chosen)
    data["arguments"] = args
    return RouteDecision(**data)


def policy_summary() -> dict[str, Any]:
    s = load_models_settings()
    return {
        "lmstudio": {"mode": s.lmstudio_mode, "model": s.lmstudio_model},
        "lemonade": {
            "mode": s.lemonade_mode,
            "model": s.lemonade_model,
            "base_url": s.lemonade_base_url,
        },
        "comfyui": {"mode": s.comfyui_mode, "model": s.comfyui_model},
        "preferred_chat_worker": s.preferred_chat_worker,
    }
