"""OpenVINO NPU router model with heuristic fallback."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.schemas import RouteDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the routing model for a local Fedora AI gateway.

You do not directly answer the user.

Determine which available worker should handle the request.

Available workers:

lmstudio:
General-purpose local AI. Use for questions, brainstorming,
summarization, analysis and privacy-sensitive tasks.

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
  "requires_confirmation": false
}
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
    """NPU-backed router with CPU/heuristic fallback."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._pipeline = None
        self._device: str | None = None
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        model_path = self.config.npu_model_path
        if not self._model_ready(model_path):
            logger.warning(
                "Router model not found at %s — using heuristic router.",
                model_path,
            )
            return

        try:
            import openvino as ov
            import openvino_genai as ov_genai

            core = ov.Core()
            devices = list(core.available_devices)
            preferred = self.config.npu.device
            fallback = self.config.npu.fallback_device
            if preferred in devices:
                device = preferred
            elif fallback in devices:
                logger.warning(
                    "NPU unavailable (%s); falling back to %s.", preferred, fallback
                )
                device = fallback
            else:
                logger.warning("No suitable OpenVINO device; using heuristic router.")
                return

            self._pipeline = ov_genai.LLMPipeline(str(model_path), device)
            self._device = device
            logger.info("Router model loaded on %s from %s", device, model_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load OpenVINO router model: %s", exc)
            self._pipeline = None
            self._device = None

    @staticmethod
    def _model_ready(path: Path) -> bool:
        if not path.exists():
            return False
        # OpenVINO IR or GenAI export typically has xml/bin or openvino_model.xml
        markers = (
            "openvino_model.xml",
            "openvino.xml",
            "config.json",
        )
        if any((path / m).exists() for m in markers):
            return True
        return any(path.glob("*.xml"))

    @property
    def npu_available(self) -> bool:
        try:
            import openvino as ov

            return "NPU" in ov.Core().available_devices
        except Exception:  # noqa: BLE001
            return False

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None

    def route(
        self,
        message: str,
        *,
        project: str | None = None,
        local_only: bool = False,
        available_workers: set[str] | None = None,
    ) -> RouteDecision:
        if self._pipeline is not None:
            try:
                return self._route_with_model(
                    message,
                    project=project,
                    local_only=local_only,
                    available_workers=available_workers,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NPU router failed (%s); using heuristic.", exc)
        return self._heuristic_route(
            message,
            project=project,
            local_only=local_only,
            available_workers=available_workers,
        )

    def _route_with_model(
        self,
        message: str,
        *,
        project: str | None,
        local_only: bool,
        available_workers: set[str] | None,
    ) -> RouteDecision:
        assert self._pipeline is not None
        workers = sorted(available_workers or {"lmstudio", "claude", "cursor", "comfyui"})
        prompt = (
            f"{ROUTER_SYSTEM_PROMPT}\n\n"
            f"local_only={local_only}\n"
            f"available_workers={workers}\n"
            f"project={project or 'none'}\n"
            f"user_request={message}\n"
        )
        raw = self._pipeline.generate(prompt, max_new_tokens=256)
        if hasattr(raw, "texts"):
            text = raw.texts[0] if raw.texts else str(raw)
        else:
            text = str(raw)
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
    ) -> RouteDecision:
        text = message.lower()
        available = available_workers or {"lmstudio", "claude", "cursor", "comfyui"}
        if local_only:
            available = available - {"claude", "cursor"}

        def pick(*candidates: str) -> str:
            for name in candidates:
                if name in available:
                    return name
            if "lmstudio" in available:
                return "lmstudio"
            if available:
                return next(iter(sorted(available)))
            raise RuntimeError("No workers available for routing.")

        # Explicit worker requests
        if re.search(r"\b(use\s+)?claude\b", text):
            worker = pick("claude", "cursor", "lmstudio")
            return RouteDecision(
                intent="coding",
                worker=worker,
                action="modify_project",
                instruction=message,
                working_directory=project,
                requires_confirmation=True,
            )
        if re.search(r"\b(use\s+)?cursor\b", text):
            worker = pick("cursor", "claude", "lmstudio")
            return RouteDecision(
                intent="coding",
                worker=worker,
                action="modify_project",
                instruction=message,
                working_directory=project,
                requires_confirmation=True,
            )
        if re.search(r"\b(local only|do this locally|locally only)\b", text):
            worker = pick("lmstudio", "comfyui")
            if any(k in text for k in ("image", "picture", "generate", "flux")):
                worker = pick("comfyui", "lmstudio")
                return RouteDecision(
                    intent="image_generation",
                    worker=worker,
                    action="generate_image",
                    instruction=message,
                    working_directory=project,
                    workflow="flux_txt2img",
                    arguments={"prompt": message},
                    requires_confirmation=False,
                )
            return RouteDecision(
                intent="general_question",
                worker=worker,
                action="answer",
                instruction=message,
                working_directory=project,
                requires_confirmation=False,
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
            return RouteDecision(
                intent="image_generation" if action == "generate_image" else "image_edit",
                worker=worker,
                action=action,
                instruction=message,
                working_directory=project,
                workflow=workflow,
                arguments={"prompt": message},
                requires_confirmation=False,
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
            worker = pick("cursor", "claude", "lmstudio")
            return RouteDecision(
                intent="coding",
                worker=worker,
                action="modify_project",
                instruction=message,
                working_directory=project,
                requires_confirmation=True,
            )
        if any(tok in text for tok in coding_tokens):
            # Coding without project — prefer local read-only analysis if available
            worker = pick("lmstudio", "cursor", "claude")
            action = "answer" if worker == "lmstudio" else "inspect_project"
            return RouteDecision(
                intent="coding",
                worker=worker,
                action=action,
                instruction=message,
                working_directory=project,
                requires_confirmation=action == "modify_project",
            )

        worker = pick("lmstudio", "claude", "cursor")
        action = "answer"
        intent = "general_question"
        if "summar" in text:
            action, intent = "summarize", "summarization"
        elif "brainstorm" in text:
            action, intent = "brainstorm", "brainstorming"
        elif "analy" in text:
            action, intent = "analyze", "analysis"

        return RouteDecision(
            intent=intent,
            worker=worker,
            action=action,
            instruction=message,
            working_directory=project,
            requires_confirmation=False,
        )


_router: RouterModel | None = None


def get_router_model(config: AppConfig | None = None) -> RouterModel:
    global _router
    if _router is None:
        _router = RouterModel(config)
    return _router
