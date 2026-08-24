"""OpenVINO NPU router model with heuristic fallback."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.npu.pipeline import get_pipeline, model_ready
from app.schemas import RouteDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the routing model for a local Fedora AI gateway.

You do not directly answer the user.

Determine which available worker should handle the request.

Available workers:

lmstudio:
General-purpose local AI (LM Studio). Use for questions, brainstorming,
summarization, analysis and privacy-sensitive tasks.

lemonade:
Local Lemonade Server LLM (OpenAI-compatible). Use like lmstudio when preferred
or when the user asks for Lemonade.

claude:
Advanced coding and repository tasks.

cursor:
Advanced coding and repository tasks.

comfyui:
Image generation and image manipulation.

Rules:
1. Return ONLY valid JSON.
2. Never invent a worker.
3. Never execute commands.
4. Never claim that a task was completed.
5. Choose the minimum number of workers necessary.
6. Respect an explicit user request for Claude or Cursor.
7. Respect local-only mode.
8. Never output arbitrary shell commands.
9. Use the supplied project directory rather than inventing one.
10. Set requires_confirmation=true for modifications unless gateway policy explicitly permits them.

Return:
{
  "intent": "...",
  "worker": "...",
  "action": "...",
  "instruction": "...",
  "working_directory": "...",
  "arguments": {},
  "requires_confirmation": false,
  "model": null
}

When available_models is provided and a worker's mode is auto, set "model" to the
best matching model id from that worker's list (coding → coder models; images →
flux/diffusion checkpoints). When mode is fixed, leave model null — the gateway
applies the pinned default.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


class RouterModel:
    """NPU-backed router with a CPU/heuristic fallback."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.pipeline = get_pipeline("router", self.config)

    def reload(self) -> None:
        """Drop and rebuild the OpenVINO pipeline after a settings change."""
        self.pipeline.reload()

    @property
    def device(self) -> str | None:
        return self.pipeline.device

    @staticmethod
    def _model_ready(path: Path) -> bool:
        return model_ready(path)

    @property
    def npu_available(self) -> bool:
        try:
            import openvino as ov

            return "NPU" in ov.Core().available_devices
        except Exception:  # noqa: BLE001
            return False

    @property
    def model_loaded(self) -> bool:
        return self.pipeline.loaded

    def route(
        self,
        message: str,
        *,
        project: str | None = None,
        local_only: bool = False,
        available_workers: set[str] | None = None,
        available_models: dict[str, list[dict[str, str]]] | None = None,
        model_modes: dict[str, str] | None = None,
        preferred_chat_worker: str = "auto",
    ) -> RouteDecision:
        if self.pipeline.loaded:
            try:
                return self._route_with_model(
                    message,
                    project=project,
                    local_only=local_only,
                    available_workers=available_workers,
                    available_models=available_models,
                    model_modes=model_modes,
                    preferred_chat_worker=preferred_chat_worker,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NPU router failed (%s); using heuristic.", exc)
        return self._heuristic_route(
            message,
            project=project,
            local_only=local_only,
            available_workers=available_workers,
            available_models=available_models,
            preferred_chat_worker=preferred_chat_worker,
        )

    def _route_with_model(
        self,
        message: str,
        *,
        project: str | None,
        local_only: bool,
        available_workers: set[str] | None,
        available_models: dict[str, list[dict[str, str]]] | None = None,
        model_modes: dict[str, str] | None = None,
        preferred_chat_worker: str = "auto",
    ) -> RouteDecision:
        workers = sorted(available_workers or {"lmstudio", "claude", "cursor", "comfyui", "lemonade"})
        # Compact model lists for the prompt (ids only, capped)
        model_snip: dict[str, list[str]] = {}
        for worker, items in (available_models or {}).items():
            ids = [m["id"] for m in items if m.get("id")][:12]
            if ids:
                model_snip[worker] = ids
        modes = model_modes or {}
        prompt = (
            f"{ROUTER_SYSTEM_PROMPT}\n\n"
            f"local_only={local_only}\n"
            f"available_workers={workers}\n"
            f"preferred_chat_worker={preferred_chat_worker}\n"
            f"model_modes={modes}\n"
            f"available_models={model_snip}\n"
            f"project={project or 'none'}\n"
            f"user_request={message}\n"
        )
        text = self.pipeline.generate(prompt, max_new_tokens=320)
        data = _extract_json(text)
        if project and not data.get("working_directory"):
            data["working_directory"] = project
        data.setdefault("arguments", {})
        data.setdefault("requires_confirmation", False)
        return RouteDecision(**data)

    def _heuristic_route(
        self,
        message: str,
        *,
        project: str | None,
        local_only: bool,
        available_workers: set[str] | None,
        available_models: dict[str, list[dict[str, str]]] | None = None,
        preferred_chat_worker: str = "auto",
    ) -> RouteDecision:
        text = message.lower()
        available = available_workers or {"lmstudio", "claude", "cursor", "comfyui", "lemonade"}
        if local_only:
            available = available - {"claude", "cursor"}

        def pick(*candidates: str) -> str:
            for name in candidates:
                if name in available:
                    return name
            if "lmstudio" in available:
                return "lmstudio"
            if "lemonade" in available:
                return "lemonade"
            if available:
                return next(iter(sorted(available)))
            raise RuntimeError("No workers available for routing.")

        def chat_pick() -> str:
            if preferred_chat_worker in {"lmstudio", "lemonade"} and preferred_chat_worker in available:
                return preferred_chat_worker
            if re.search(r"\blemonade\b", text):
                return pick("lemonade", "lmstudio")
            if re.search(r"\blm\s*studio\b", text):
                return pick("lmstudio", "lemonade")
            return pick("lmstudio", "lemonade")

        def with_model(decision: RouteDecision) -> RouteDecision:
            from app.worker_models import heuristic_pick

            if decision.worker not in {"lmstudio", "lemonade", "comfyui"}:
                return decision
            models = (available_models or {}).get(decision.worker) or []
            chosen = heuristic_pick(
                decision.worker,
                models,
                intent=decision.intent,
                instruction=decision.instruction,
            )
            if not chosen:
                return decision
            data = decision.model_dump()
            data["model"] = chosen
            args = dict(data.get("arguments") or {})
            args.setdefault("model", chosen)
            if decision.worker == "comfyui":
                args.setdefault("unet_name", chosen)
                args.setdefault("ckpt_name", chosen)
            data["arguments"] = args
            return RouteDecision(**data)

        # Explicit worker requests
        if re.search(r"\b(use\s+)?claude\b", text):
            worker = pick("claude", "cursor", "lmstudio", "lemonade")
            return with_model(
                RouteDecision(
                    intent="coding",
                    worker=worker,
                    action="modify_project",
                    instruction=message,
                    working_directory=project,
                    requires_confirmation=True,
                )
            )
        if re.search(r"\b(use\s+)?cursor\b", text):
            worker = pick("cursor", "claude", "lmstudio", "lemonade")
            return with_model(
                RouteDecision(
                    intent="coding",
                    worker=worker,
                    action="modify_project",
                    instruction=message,
                    working_directory=project,
                    requires_confirmation=True,
                )
            )
        if re.search(r"\b(local only|do this locally|locally only)\b", text):
            worker = chat_pick()
            if any(k in text for k in ("image", "picture", "generate", "flux")):
                worker = pick("comfyui", "lmstudio", "lemonade")
                return with_model(
                    RouteDecision(
                        intent="image_generation",
                        worker=worker,
                        action="generate_image",
                        instruction=message,
                        working_directory=project,
                        workflow="flux_txt2img",
                        arguments={"prompt": message},
                        requires_confirmation=False,
                    )
                )
            return with_model(
                RouteDecision(
                    intent="general_question",
                    worker=worker,
                    action="answer",
                    instruction=message,
                    working_directory=project,
                    requires_confirmation=False,
                )
            )

        image_tokens = (
            "generate an image",
            "generate a",
            "create an image",
            "draw ",
            "image of",
            "flux",
            "txt2img",
            "img2img",
            "inpaint",
            "upscale",
            "hero artwork",
            "hero background",
            "cyberpunk city",
        )
        if any(tok in text for tok in image_tokens) or (
            "image" in text and any(t in text for t in ("generate", "create", "make", "edit"))
        ):
            worker = pick("comfyui")
            action = "generate_image"
            workflow = "flux_txt2img"
            if "upscale" in text:
                action, workflow = "upscale_image", "upscale"
            elif "inpaint" in text:
                action, workflow = "inpaint_image", "flux_inpaint"
            elif "img2img" in text or "edit image" in text:
                action, workflow = "edit_image", "flux_img2img"
            return with_model(
                RouteDecision(
                    intent="image_generation" if action == "generate_image" else "image_edit",
                    worker=worker,
                    action=action,
                    instruction=message,
                    working_directory=project,
                    workflow=workflow,
                    arguments={"prompt": message},
                    requires_confirmation=False,
                )
            )

        coding_tokens = (
            "fix",
            "bug",
            "refactor",
            "implement",
            "code",
            "typescript",
            "react",
            "component",
            "repository",
            "repo",
            "commit",
            "test",
            "build",
            "lint",
            "authentication",
            "landing page",
            "website",
            "project",
        )
        if any(tok in text for tok in coding_tokens) and project:
            worker = pick("cursor", "claude", "lmstudio", "lemonade")
            return with_model(
                RouteDecision(
                    intent="coding",
                    worker=worker,
                    action="modify_project",
                    instruction=message,
                    working_directory=project,
                    requires_confirmation=True,
                )
            )
        if any(tok in text for tok in coding_tokens):
            worker = pick(chat_pick(), "cursor", "claude")
            action = "answer" if worker in {"lmstudio", "lemonade"} else "inspect_project"
            return with_model(
                RouteDecision(
                    intent="coding",
                    worker=worker,
                    action=action,
                    instruction=message,
                    working_directory=project,
                    requires_confirmation=action == "modify_project",
                )
            )

        worker = chat_pick()
        action = "answer"
        intent = "general_question"
        if "summar" in text:
            action, intent = "summarize", "summarization"
        elif "brainstorm" in text:
            action, intent = "brainstorm", "brainstorming"
        elif "analy" in text:
            action, intent = "analyze", "analysis"

        return with_model(
            RouteDecision(
                intent=intent,
                worker=worker,
                action=action,
                instruction=message,
                working_directory=project,
                requires_confirmation=False,
            )
        )


_router: RouterModel | None = None


def get_router_model(config: AppConfig | None = None) -> RouterModel:
    global _router
    if _router is None:
        _router = RouterModel(config)
    return _router
