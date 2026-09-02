"""How an OpenVINO GenAI pipeline is constructed — the one place it is.

The options are not decoration: ``MAX_PROMPT_LEN`` is compiled into the model
and is part of the compile cache key, so two callers that spell the options out
differently do not merely configure differently, they compile *different
blobs*. That is why the constructor lives here alongside the options rather
than at each call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npu.kind import PipelineKind
from npu.limits import NPU_MAX_PROMPT_TOKENS


def _is_npu(device: str) -> bool:
    return device.upper() == "NPU"


def pipeline_init_kwargs(device: str, cache: Path | None, kind: PipelineKind) -> dict[str, Any]:
    """Build keyword args for ``LLMPipeline`` / ``VLMPipeline``."""
    cache_str = str(cache) if cache else ""
    if kind == "vlm":
        device_props: dict[str, Any] = {}
        if _is_npu(device):
            device_props["GENERATE_HINT"] = "FAST_COMPILE"
            # Without this the VLM compiles with a 1024-token prompt limit and
            # throws on anything longer — which the system prompt alone exceeds.
            device_props["MAX_PROMPT_LEN"] = NPU_MAX_PROMPT_TOKENS
        if cache_str:
            device_props["CACHE_DIR"] = cache_str
        if not device_props:
            return {}
        return {"config": {"DEVICE_PROPERTIES": {device: device_props}}}

    kwargs: dict[str, Any] = {}
    if _is_npu(device):
        kwargs["MAX_PROMPT_LEN"] = NPU_MAX_PROMPT_TOKENS
        kwargs["GENERATE_HINT"] = "FAST_COMPILE"
    if cache_str:
        kwargs["CACHE_DIR"] = cache_str
    return kwargs


def create_pipeline(
    model_dir: Path,
    device: str,
    cache: Path | None,
    kind: PipelineKind,
) -> Any:
    """Build the pipeline. The subprocess probe and the daemon both come here.

    They used to each write out their own constructor call, and they drifted:
    the probe compiled with ``MAX_PROMPT_LEN`` 1024 — or, for a VLM, whatever
    the default was — while the load asked for 4096. Different value, different
    cache key, so the blob the probe spent its whole run producing was never
    the blob the load could reuse, and a first load paid for two compiles.
    """
    import openvino_genai as ov_genai  # noqa: PLC0415

    kwargs = pipeline_init_kwargs(device, cache, kind)
    pipeline_cls = ov_genai.VLMPipeline if kind == "vlm" else ov_genai.LLMPipeline
    try:
        return pipeline_cls(str(model_dir), device, **kwargs)
    except TypeError:
        # An older GenAI whose constructor does not take these options at all.
        return pipeline_cls(str(model_dir), device)
