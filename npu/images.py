"""Convert user image bytes to OpenVINO tensors for VLMPipeline."""

from __future__ import annotations

import io
from typing import Any

import numpy as np


def bytes_to_ov_tensors(image_bytes: list[bytes]) -> list[Any]:
    """RGB uint8 tensors shaped [1, H, W, 3] for openvino_genai.VLMPipeline."""
    if not image_bytes:
        return []

    import openvino as ov
    from PIL import Image

    tensors: list[Any] = []
    for raw in image_bytes:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        tensors.append(ov.Tensor(arr[None]))
    return tensors
