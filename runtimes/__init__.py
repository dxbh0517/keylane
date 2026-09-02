"""The runtimes Keylane ships, and how a model finds the right one.

A model entry names its runtime, so most of the time lookup is a dict read. The
interesting case is a model that arrived without saying — an import from Hugging
Face, or a directory already on disk from an older version — and for that
``detect`` asks each runtime whether the export is one of its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

from runtimes.base import LoadedPipeline, RepoVariant, RuntimeBackend, RuntimeInfo, status_payload
from runtimes.onnx_rt import OnnxRuntimeBackend
from runtimes.openvino_rt import OpenVinoBackend

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RUNTIME",
    "LoadedPipeline",
    "RepoVariant",
    "RuntimeBackend",
    "RuntimeInfo",
    "backend_for",
    "detect_runtime",
    "list_backends",
    "normalise_runtime_id",
    "runtime_ids",
    "status_payload",
]

# The one Keylane has always used, and the one whose dependencies are in
# requirements.txt — so an install that changed nothing still works.
DEFAULT_RUNTIME = "openvino"

_BACKENDS: dict[str, RuntimeBackend] = {
    OpenVinoBackend.info.id: OpenVinoBackend(),
    OnnxRuntimeBackend.info.id: OnnxRuntimeBackend(),
}

# Spellings that have shown up in configs and in people's fingers.
_ALIASES = {
    "ov": "openvino",
    "openvino-genai": "openvino",
    "openvino_genai": "openvino",
    "onnx": "onnxruntime",
    "onnxruntime-genai": "onnxruntime",
    "onnxruntime_genai": "onnxruntime",
    "ort": "onnxruntime",
}


def runtime_ids() -> list[str]:
    return list(_BACKENDS)


def list_backends() -> list[RuntimeBackend]:
    return list(_BACKENDS.values())


def normalise_runtime_id(runtime_id: str | None) -> str:
    """Map a stored or typed runtime name onto one we have, else the default."""
    key = (runtime_id or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key in _BACKENDS:
        return key
    if key:
        logger.warning("unknown runtime %r — falling back to %s", runtime_id, DEFAULT_RUNTIME)
    return DEFAULT_RUNTIME


def backend_for(runtime_id: str | None) -> RuntimeBackend:
    return _BACKENDS[normalise_runtime_id(runtime_id)]


def detect_runtime(model_dir: Path) -> str | None:
    """Which runtime claims the export in *model_dir*, if any."""
    for backend in _BACKENDS.values():
        if backend.detect(model_dir):
            return backend.info.id
    return None
