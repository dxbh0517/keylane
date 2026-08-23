"""NPU / accelerator detection helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def accel_device_present() -> bool:
    accel = Path("/dev/accel")
    if accel.is_dir() and any(accel.iterdir()):
        return True
    sysfs = Path("/sys/class/accel")
    return sysfs.is_dir() and any(sysfs.iterdir())


def level_zero_npu_present() -> bool:
    """Intel Level Zero NPU userspace (required by OpenVINO NPU plugin)."""
    names = (
        "libze_intel_npu.so.1",
        "libze_intel_npu.so",
        "libze_intel_vpu.so.1",
        "libze_intel_vpu.so",
    )
    lib_dirs = [
        Path("/usr/lib64"),
        Path("/usr/lib"),
        Path("/usr/local/lib64"),
        Path("/usr/local/lib"),
        Path("/lib64"),
        Path("/lib"),
    ]
    for directory in lib_dirs:
        for name in names:
            if (directory / name).exists():
                return True
    try:
        import subprocess

        out = subprocess.check_output(
            ["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL
        )
        return "libze_intel_npu" in out or "libze_intel_vpu" in out
    except Exception:  # noqa: BLE001
        return False


def user_in_render_group() -> bool:
    try:
        import grp

        render = grp.getgrnam("render")
        return render.gr_gid in os.getgroups() or os.getuid() == 0
    except KeyError:
        return True
    except Exception:  # noqa: BLE001
        return False


def openvino_devices() -> list[str]:
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:  # noqa: BLE001
        return []


def npu_status() -> dict[str, Any]:
    """
    Report both kernel accel presence and OpenVINO NPU visibility.

    On many Fedora + Core Ultra machines the intel_vpu driver and /dev/accel/accel0
    exist, but OpenVINO only exposes NPU after Level Zero NPU userspace is installed
    and the user can access the accel device (often via the `render` group).
    """
    devices = openvino_devices()
    ov_npu = "NPU" in devices
    driver = accel_device_present()
    lz = level_zero_npu_present()
    render_ok = user_in_render_group()

    if ov_npu:
        detail = "OpenVINO NPU device available"
        online = True
    elif driver and not lz:
        detail = (
            "Intel NPU driver present (/dev/accel) but Level Zero NPU userspace is missing "
            f"(OpenVINO devices: {', '.join(devices) or 'none'}). "
            "On Fedora: sudo dnf install intel-npu-driver oneapi-level-zero "
            "&& sudo usermod -aG render $USER (then re-login). "
            "On Ubuntu: install intel-level-zero-npu from "
            "https://github.com/intel/linux-npu-driver/releases, then re-login. "
            "Router uses CPU/heuristic fallback until then."
        )
        online = False
    elif driver and not render_ok:
        detail = (
            "Intel NPU driver present but this user is not in the `render` group. "
            "Run: sudo usermod -aG render $USER  (then log out/in). "
            f"OpenVINO devices: {', '.join(devices) or 'none'}."
        )
        online = False
    elif driver:
        detail = (
            "Intel NPU driver present but OpenVINO does not expose NPU yet "
            f"(devices: {', '.join(devices) or 'none'}). Router uses CPU/heuristic fallback."
        )
        online = False
    else:
        detail = "No Intel NPU accelerator device detected"
        online = False

    return {
        "npu": online,
        "npu_driver": driver,
        "npu_openvino": ov_npu,
        "npu_level_zero": lz,
        "npu_detail": detail,
        "openvino_devices": devices,
    }
