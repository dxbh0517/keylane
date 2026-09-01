"""OpenVINO GenAI pipeline constructor options."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npu.kind import PipelineKind


def _is_npu(device: str) -> bool:
    return device.upper() == "NPU"


def pipeline_init_kwargs(device: str, cache: Path | None, kind: PipelineKind) -> dict[str, Any]:
    """Build keyword args for ``LLMPipeline`` / ``VLMPipeline``."""
    cache_str = str(cache) if cache else ""
    if kind == "vlm":
        device_props: dict[str, Any] = {}
        if _is_npu(device):
            device_props["GENERATE_HINT"] = "FAST_COMPILE"
        if cache_str:
            device_props["CACHE_DIR"] = cache_str
        if not device_props:
            return {}
        return {"config": {"DEVICE_PROPERTIES": {device: device_props}}}

    kwargs: dict[str, Any] = {}
    if _is_npu(device):
        kwargs["MAX_PROMPT_LEN"] = 1024
        kwargs["GENERATE_HINT"] = "FAST_COMPILE"
    if cache_str:
        kwargs["CACHE_DIR"] = cache_str
    return kwargs
