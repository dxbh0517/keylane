"""Hardware / accelerator capability detection for model recommendations."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DeviceInfo:
    id: str
    name: str
    kind: str  # cpu | gpu | npu | other
    openvino: bool = False
    vram_mb: int | None = None
    detail: str = ""


@dataclass
class HardwareProfile:
    cpu_name: str = "Unknown CPU"
    cpu_cores: int = 0
    ram_gb: float | None = None
    openvino_devices: list[str] = field(default_factory=list)
    devices: list[DeviceInfo] = field(default_factory=list)
    has_npu: bool = False
    has_gpu: bool = False
    nvidia_vram_mb: int | None = None
    nvidia_name: str | None = None
    tier: str = "cpu"  # cpu | light_npu | npu | gpu | high_vram
    summary: str = ""


def _cpu_name() -> str:
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def _cpu_cores() -> int:
    try:
        return int(os_cpu_count() or 0)
    except Exception:  # noqa: BLE001
        return 0


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _ram_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        text = meminfo.read_text(encoding="utf-8")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.M)
        if match:
            return round(int(match.group(1)) / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        return None
    return None


def _nvidia() -> tuple[str | None, int | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not out:
            return None, None
        # Use first GPU
        line = out.splitlines()[0]
        name, mem = [p.strip() for p in line.split(",", 1)]
        return name, int(float(mem))
    except Exception:  # noqa: BLE001
        return None, None


def _openvino_devices() -> list[tuple[str, str]]:
    try:
        import openvino as ov

        core = ov.Core()
        devices = list(core.available_devices)
        named: list[tuple[str, str]] = []
        for device in devices:
            try:
                full = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001
                full = device
            named.append((device, full))
        return named
    except Exception:  # noqa: BLE001
        return []


def detect_hardware() -> HardwareProfile:
    cpu = _cpu_name()
    cores = _cpu_cores()
    ram = _ram_gb()
    nvidia_name, nvidia_vram = _nvidia()
    ov_named = _openvino_devices()
    ov_ids = [d for d, _ in ov_named]

    devices: list[DeviceInfo] = []
    for device_id, full_name in ov_named:
        kind = "other"
        if device_id == "CPU" or device_id.startswith("CPU"):
            kind = "cpu"
        elif "NPU" in device_id:
            kind = "npu"
        elif "GPU" in device_id:
            kind = "gpu"
        devices.append(
            DeviceInfo(
                id=device_id,
                name=full_name,
                kind=kind,
                openvino=True,
                vram_mb=nvidia_vram if kind == "gpu" and nvidia_vram else None,
            )
        )

    # Surface NVIDIA even if OpenVINO GPU is Intel-only (common on hybrid laptops)
    if nvidia_name and not any(
        "nvidia" in d.name.lower() or "geforce" in d.name.lower() for d in devices if d.kind == "gpu"
    ):
        devices.append(
            DeviceInfo(
                id="NVIDIA",
                name=nvidia_name,
                kind="gpu",
                openvino=False,
                vram_mb=nvidia_vram,
                detail="Use via LM Studio / CUDA workers (not OpenVINO NPU path)",
            )
        )

    has_npu = any(d.kind == "npu" for d in devices) or "NPU" in ov_ids
    has_gpu = any(d.kind == "gpu" for d in devices) or bool(nvidia_name)

    if nvidia_vram and nvidia_vram >= 16000:
        tier = "high_vram"
    elif has_npu and has_gpu:
        tier = "npu_gpu"
    elif has_npu:
        tier = "npu"
    elif has_gpu:
        tier = "gpu"
    else:
        tier = "cpu"

    parts = [cpu]
    if nvidia_name:
        vram = f" ({nvidia_vram} MB)" if nvidia_vram else ""
        parts.append(f"{nvidia_name}{vram}")
    if has_npu:
        parts.append("Intel NPU")
    summary = " · ".join(parts)

    return HardwareProfile(
        cpu_name=cpu,
        cpu_cores=cores,
        ram_gb=ram,
        openvino_devices=ov_ids,
        devices=devices,
        has_npu=has_npu,
        has_gpu=has_gpu,
        nvidia_vram_mb=nvidia_vram,
        nvidia_name=nvidia_name,
        tier=tier,
        summary=summary,
    )


def hardware_dict() -> dict[str, Any]:
    profile = detect_hardware()
    data = asdict(profile)
    return data
