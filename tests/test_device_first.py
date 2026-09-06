"""Choosing hardware first, then the stacks that reach it, then the models.

The old panel asked for the runtime first, which is backwards from how anyone
decides — you know what silicon you want to spend before you know or care which
inference stack reaches it. It also made two questions look independent when
they are not: CUDA is reachable only through ONNX Runtime, so picking OpenVINO
first quietly removed the GPU from the device list without ever saying so.
"""

from __future__ import annotations

import os

os.environ.setdefault("KEYLANE_LAYER_SHELL_PRIMED", "1")

import pytest  # noqa: E402

from ui.settings import SettingsWindow  # noqa: E402


def _panel(**devices_per_runtime: str) -> SettingsWindow:
    """The panel's logic without realising any widget."""
    panel = SettingsWindow.__new__(SettingsWindow)
    panel._model_devices = dict(devices_per_runtime) or {"openvino": "NPU"}
    panel._runtime_id = "openvino"
    panel._runtimes = [
        {
            "id": "openvino",
            "name": "OpenVINO GenAI",
            "default_device": "NPU",
            "devices": ["NPU", "CPU"],
            "all_devices": [
                {"id": "NPU", "label": "NPU — Intel AI Boost", "usable": True, "reason": ""},
                {
                    "id": "GPU",
                    "label": "GPU — NVIDIA RTX 5090",
                    "usable": False,
                    "reason": "not an Intel device; OpenVINO cannot compile for it",
                },
                {"id": "CPU", "label": "CPU — Core Ultra 9", "usable": True, "reason": ""},
            ],
        },
        {
            "id": "onnxruntime",
            "name": "ONNX Runtime GenAI",
            "default_device": "NPU",
            "devices": ["NPU", "CUDA", "CPU", "AUTO"],
            "all_devices": [
                {"id": "NPU", "label": "NPU — Intel AI Boost", "usable": True, "reason": ""},
                {
                    "id": "GPU",
                    "label": "GPU — NVIDIA RTX 5090",
                    "usable": False,
                    "reason": "not an Intel device; OpenVINO cannot compile for it",
                },
                {"id": "CUDA", "label": "CUDA — NVIDIA RTX 5090", "usable": True, "reason": ""},
                {"id": "CPU", "label": "CPU — Core Ultra 9", "usable": True, "reason": ""},
                {"id": "AUTO", "label": "Auto (as exported)", "usable": True, "reason": ""},
            ],
        },
    ]
    return panel


# ── devices are merged across runtimes ───────────────────────────────────


def test_every_device_any_runtime_reaches_is_offered() -> None:
    ids = [r["id"] for r in _panel()._device_rows()]
    assert ids == ["NPU", "CUDA", "GPU", "CPU", "AUTO"]


def test_one_runtimes_refusal_does_not_hide_anothers_support() -> None:
    """The same card is an unusable `GPU` and a usable `CUDA`.

    Merging naively, or asking only the selected runtime, loses the GPU.
    """
    rows = {r["id"]: r for r in _panel()._device_rows()}
    assert rows["CUDA"]["usable"]
    assert rows["CUDA"]["runtimes"] == ["onnxruntime"]


def test_a_device_no_runtime_can_reach_keeps_its_reason() -> None:
    """The user can see the card; hiding it invites the question this answers."""
    rows = {r["id"]: r for r in _panel()._device_rows()}
    assert not rows["GPU"]["usable"]
    assert "cannot compile" in rows["GPU"]["reason"]


def test_devices_are_ordered_fastest_first() -> None:
    rows = [r["id"] for r in _panel()._device_rows()]
    assert rows.index("NPU") < rows.index("CPU")
    assert rows.index("CUDA") < rows.index("CPU")


# ── the runtime follows the device ───────────────────────────────────────


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("NPU", ["openvino", "onnxruntime"]),
        ("CUDA", ["onnxruntime"]),
        ("CPU", ["openvino", "onnxruntime"]),
        ("AUTO", ["onnxruntime"]),
        ("GPU", []),
    ],
)
def test_which_runtimes_reach_a_device(device: str, expected: list[str]) -> None:
    assert _panel()._runtimes_for_device(device) == expected


def test_the_selected_device_is_what_this_runtime_is_set_to() -> None:
    panel = _panel(openvino="CPU", onnxruntime="CUDA")
    assert panel._current_device() == "CPU"
    panel._runtime_id = "onnxruntime"
    assert panel._current_device() == "CUDA"


def test_an_unusable_stored_device_falls_back_rather_than_sticking() -> None:
    """A settings file naming GPU must not leave the panel pointed at nothing."""
    panel = _panel(openvino="GPU")
    assert panel._current_device() != "GPU"
    assert panel._current_device() in ("NPU", "CPU")
