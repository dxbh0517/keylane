"""Detect OpenVINO GenAI export layout on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PipelineKind = Literal["llm", "vlm"]


def model_kind(model_dir: Path) -> PipelineKind:
    """Return ``vlm`` for vision-language exports, else ``llm``."""
    if not model_dir.is_dir():
        return "llm"
    if (model_dir / "openvino_model.xml").is_file():
        return "llm"
    if (model_dir / "openvino_language_model.xml").is_file():
        return "vlm"
    if (model_dir / "openvino_vision_embeddings_model.xml").is_file():
        return "vlm"
    return "llm"
