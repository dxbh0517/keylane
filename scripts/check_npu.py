#!/usr/bin/env python3
"""Phase 1 — verify Intel NPU visibility and optional model inference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    section("Kernel / driver")
    print("uname:", subprocess.getoutput("uname -r"))
    print(subprocess.getoutput("lsmod | grep -E 'intel_vpu|ivpu|vpu' || true"))

    section("OpenVINO")
    try:
        import openvino as ov

        print("openvino:", ov.__version__)
        core = ov.Core()
        devices = list(core.available_devices)
        print("devices:", devices)
        if "NPU" not in devices:
            print("WARNING: NPU not listed. Resolve driver/runtime before relying on NPU.")
            npu_ok = False
        else:
            print("NPU is visible to OpenVINO.")
            npu_ok = True
    except Exception as exc:  # noqa: BLE001
        print("OpenVINO import failed:", exc)
        print("Install with: pip install openvino openvino-genai openvino-tokenizers")
        return 1

    section("Router model")
    model_path = ROOT / "models" / "router"
    markers = ["openvino_model.xml", "config.json"]
    ready = model_path.exists() and (
        any((model_path / m).exists() for m in markers) or any(model_path.glob("*.xml"))
    )
    print("model_path:", model_path)
    print("ready:", ready)
    if not ready:
        print(
            "Place a 1B–3B OpenVINO GenAI export in models/router, then re-run.\n"
            "Until then the gateway uses the heuristic router fallback."
        )
        return 0 if npu_ok else 2

    section("Direct inference")
    try:
        import openvino_genai as ov_genai

        device = "NPU" if npu_ok else "CPU"
        pipe = ov_genai.LLMPipeline(str(model_path), device)
        prompt = (
            "Classify this request as coding, image_generation, or general_question: "
            "Fix my React component"
        )
        out = pipe.generate(prompt, max_new_tokens=64)
        print("device:", device)
        print("output:", out)
        print("SUCCESS: inference completed.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("Inference failed:", exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
