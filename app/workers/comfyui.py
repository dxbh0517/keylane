"""ComfyUI worker with approved workflow templates only."""

from __future__ import annotations

import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.config import AppConfig, get_config
from app.schemas import RouteDecision, WorkerEvidence, WorkerResult

logger = logging.getLogger(__name__)

APPROVED_WORKFLOWS = frozenset(
    {
        "flux_txt2img",
        "flux_img2img",
        "flux_inpaint",
        "upscale",
    }
)

# Only these node input keys may be overwritten from router arguments.
WHITELISTED_FIELDS = frozenset(
    {
        "text",
        "prompt",
        "width",
        "height",
        "seed",
        "steps",
        "cfg",
        "denoise",
        "model",
        "ckpt_name",
        "unet_name",
        "clip_name",
        "vae_name",
    }
)


class ComfyUiWorker:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    @property
    def base_url(self) -> str:
        return self.config.comfyui.base_url.rstrip("/")

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.base_url}/system_stats")
                return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def _load_workflow(self, name: str) -> dict[str, Any]:
        if name not in APPROVED_WORKFLOWS:
            raise ValueError(f"Workflow '{name}' is not approved.")
        path = self.config.workflows_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Workflow file missing: {path}")
        with path.open() as fh:
            return json.load(fh)

    def _apply_arguments(self, workflow: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
        graph = copy.deepcopy(workflow)
        prompt = arguments.get("prompt") or arguments.get("text")
        width = arguments.get("width")
        height = arguments.get("height")
        seed = arguments.get("seed")
        model_name = (
            arguments.get("unet_name")
            or arguments.get("ckpt_name")
            or arguments.get("model")
        )

        for _node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            class_type = node.get("class_type", "")

            if prompt is not None and "text" in inputs and (
                "CLIP" in class_type.upper()
                or class_type
                in {
                    "CLIPTextEncode",
                    "CLIPTextEncodeFlux",
                    "CR Prompt Text",
                }
            ):
                inputs["text"] = prompt

            # Generic whitelist updates for matching keys only.
            for key, value in arguments.items():
                if key in WHITELISTED_FIELDS and key in inputs:
                    inputs[key] = value

            if width is not None and "width" in inputs:
                inputs["width"] = int(width)
            if height is not None and "height" in inputs:
                inputs["height"] = int(height)
            if seed is not None and "seed" in inputs:
                inputs["seed"] = int(seed)

            if model_name:
                if "unet_name" in inputs:
                    inputs["unet_name"] = model_name
                if "ckpt_name" in inputs:
                    inputs["ckpt_name"] = model_name

        # Ensure positive prompt nodes get the prompt when class detection missed.
        if prompt is not None:
            for _node_id, node in graph.items():
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs")
                if isinstance(inputs, dict) and "text" in inputs and node.get("_meta", {}).get("title") in {
                    "Positive Prompt",
                    "Prompt",
                }:
                    inputs["text"] = prompt

        return graph

    async def run(self, decision: RouteDecision) -> WorkerResult:
        workflow_name = decision.workflow or decision.arguments.get("workflow") or "flux_txt2img"
        arguments = dict(decision.arguments or {})
        if decision.model:
            arguments.setdefault("model", decision.model)
            arguments.setdefault("unet_name", decision.model)
            arguments.setdefault("ckpt_name", decision.model)
        try:
            template = self._load_workflow(str(workflow_name))
            graph = self._apply_arguments(template, arguments)
        except Exception as exc:  # noqa: BLE001
            evidence = WorkerEvidence(
                worker="comfyui",
                action=decision.action,
                stderr=str(exc),
                exit_code=1,
            )
            return WorkerResult(success=False, evidence=evidence, summary=str(exc))

        client_id = uuid.uuid4().hex
        payload = {"prompt": graph, "client_id": client_id}

        try:
            async with httpx.AsyncClient(
                timeout=self.config.comfyui.timeout_seconds
            ) as client:
                # Queue prompt
                queued = await client.post(f"{self.base_url}/prompt", json=payload)
                queued.raise_for_status()
                prompt_id = queued.json().get("prompt_id")

                # Poll history
                output_path = None
                dimensions = None
                history_item: dict[str, Any] = {}
                for _ in range(300):
                    hist = await client.get(f"{self.base_url}/history/{prompt_id}")
                    hist.raise_for_status()
                    data = hist.json()
                    if prompt_id in data:
                        history_item = data[prompt_id]
                        break
                    await asyncio_sleep(1.0)

                outputs = history_item.get("outputs", {})
                for node_out in outputs.values():
                    images = node_out.get("images") or []
                    if not images:
                        continue
                    image = images[0]
                    filename = image.get("filename")
                    subfolder = image.get("subfolder", "")
                    img_type = image.get("type", "output")
                    params = {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": img_type,
                    }
                    view = await client.get(f"{self.base_url}/view", params=params)
                    view.raise_for_status()
                    out_dir = self.config.output_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    dest = out_dir / filename
                    dest.write_bytes(view.content)
                    output_path = str(dest)
                    # Best-effort dimensions from request args
                    dimensions = {
                        "width": int(decision.arguments.get("width") or 0) or None,
                        "height": int(decision.arguments.get("height") or 0) or None,
                    }
                    # Drop None values
                    dimensions = {k: v for k, v in dimensions.items() if v}
                    break

                evidence = WorkerEvidence(
                    worker="comfyui",
                    action=decision.action,
                    exit_code=0 if output_path else 1,
                    output_path=output_path,
                    output_dimensions=dimensions or None,
                    metadata={
                        "prompt_id": prompt_id,
                        "workflow": workflow_name,
                        "arguments": {
                            k: v
                            for k, v in decision.arguments.items()
                            if k in WHITELISTED_FIELDS or k in {"prompt", "width", "height"}
                        },
                    },
                )
                if not output_path:
                    return WorkerResult(
                        success=False,
                        evidence=evidence,
                        summary="ComfyUI finished without an image output.",
                    )
                return WorkerResult(
                    success=True,
                    evidence=evidence,
                    summary=f"Generated image: {output_path}",
                    raw={"prompt_id": prompt_id},
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ComfyUI worker failed")
            evidence = WorkerEvidence(
                worker="comfyui",
                action=decision.action,
                stderr=str(exc),
                exit_code=1,
            )
            return WorkerResult(success=False, evidence=evidence, summary=str(exc))


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
