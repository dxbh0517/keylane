"""CUDA as a device of the ONNX runtime, and models that suit a device.

A discrete NVIDIA card is the one piece of hardware Keylane could see and not
use. OpenVINO enumerates it as `GPU` and then cannot compile for it, so it was
correctly refused — and there was no other path, which left a 24 GB GPU idle
next to a 13 TOPS NPU.

CUDA is a device of the ONNX runtime rather than a runtime of its own: it is an
execution provider of that same stack, exactly as OpenVINO is, and a separate
backend would be a second copy of onnx_rt.py.
"""

from __future__ import annotations

import pytest

from models.catalog import ModelEntry
from runtimes import backend_for
from runtimes.onnx_rt import INFO, _variant_score, cuda_status


def _entry(**kw) -> ModelEntry:
    base = {
        "id": "m",
        "name": "M",
        "hf_repo": "org/repo",
        "params_b": 1,
        "runtime": "openvino",
    }
    return ModelEntry(**{**base, **kw})


# ── the device exists and answers honestly ───────────────────────────────


def test_the_onnx_runtime_advertises_cuda() -> None:
    assert "CUDA" in INFO.devices


def test_openvino_does_not_advertise_cuda() -> None:
    """OpenVINO has no CUDA plugin; offering it would be a lie."""
    assert "CUDA" not in backend_for("openvino").info.devices


def test_cuda_is_refused_without_a_driver(monkeypatch) -> None:
    import runtimes.onnx_rt as onnx

    monkeypatch.setattr(onnx, "nvidia_present", lambda: False)
    usable, reason = onnx.cuda_status()
    assert not usable
    assert "driver" in reason


def test_cuda_is_refused_without_the_provider_wheel(monkeypatch) -> None:
    """A driver being present says nothing: the provider is a separate wheel."""
    import runtimes.onnx_rt as onnx

    monkeypatch.setattr(onnx, "nvidia_present", lambda: True)
    monkeypatch.setattr(onnx, "cuda_provider_available", lambda: False)
    usable, reason = onnx.cuda_status()
    assert not usable
    assert "onnxruntime-genai-cuda" in reason


def test_cuda_is_usable_with_both(monkeypatch) -> None:
    import runtimes.onnx_rt as onnx

    monkeypatch.setattr(onnx, "nvidia_present", lambda: True)
    monkeypatch.setattr(onnx, "cuda_provider_available", lambda: True)
    assert onnx.cuda_status() == (True, "")


def test_the_reason_reaches_settings_rather_than_hiding_the_card(monkeypatch) -> None:
    """Hardware the user knows they have is greyed out, never omitted."""
    import runtimes.devices as devices

    monkeypatch.setattr(devices, "_nvidia_name", lambda: "GeForce RTX 5090")
    options = {o.id: o for o in devices.device_options(("CUDA",))}
    assert "CUDA" in options
    assert "5090" in options["CUDA"].label
    if not cuda_status()[0]:
        assert options["CUDA"].reason


def test_cuda_is_not_answered_by_openvinos_device_list(monkeypatch) -> None:
    """The bug this replaces: asking OpenVINO about CUDA says "not present"."""
    import runtimes.devices as devices

    monkeypatch.setattr(devices, "_openvino_devices", lambda: {"CPU": "Intel", "NPU": "Intel"})
    monkeypatch.setattr(devices, "_cuda_option", lambda: devices.DeviceOption("CUDA", "CUDA", True))
    options = {o.id: o for o in devices.device_options(("CUDA", "GPU"))}
    assert options["CUDA"].usable
    assert not options["GPU"].usable  # genuinely absent from OpenVINO's list


# ── picking a build out of a repo ────────────────────────────────────────


def test_a_cuda_build_wins_when_the_card_is_there(monkeypatch) -> None:
    import runtimes.onnx_rt as onnx

    monkeypatch.setattr(onnx, "cuda_status", lambda: (True, ""))
    assert _variant_score("cuda/cuda-int4-rtn-block-32") > _variant_score(
        "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    )


def test_a_cuda_build_is_refused_when_it_is_not(monkeypatch) -> None:
    import runtimes.onnx_rt as onnx

    monkeypatch.setattr(onnx, "cuda_status", lambda: (False, "no card"))
    assert _variant_score("cuda/cuda-int4-rtn-block-32") < 0


@pytest.mark.parametrize("folder", ["directml/directml-int4", "npu/qnn-int4", "rocm/rocm-fp16"])
def test_builds_for_other_vendors_stay_refused(folder: str) -> None:
    """CUDA became conditional; DirectML, QNN and ROCm did not."""
    assert _variant_score(folder) < 0


# ── which models to recommend for a device ───────────────────────────────


def test_an_asymmetric_export_is_not_recommended_on_the_npu() -> None:
    suited, reason = _entry(npu_ready=False).suits_device("NPU")
    assert not suited
    assert "symmetric" in reason


def test_a_symmetric_export_is_recommended_on_the_npu() -> None:
    assert _entry(npu_ready=True).suits_device("NPU") == (True, "")


def test_the_same_asymmetric_export_is_fine_on_cpu_and_gpu() -> None:
    """It is not a bad model — it is the wrong quantization for one device."""
    entry = _entry(npu_ready=False)
    assert entry.suits_device("CPU")[0]
    assert entry.suits_device("GPU")[0]


def test_openvino_ir_is_not_recommended_on_cuda() -> None:
    suited, reason = _entry(runtime="openvino").suits_device("CUDA")
    assert not suited
    assert "ONNX" in reason


def test_an_onnx_export_is_recommended_on_cuda() -> None:
    assert _entry(runtime="onnxruntime").suits_device("CUDA") == (True, "")


def test_auto_imposes_nothing() -> None:
    assert _entry(npu_ready=False).suits_device("AUTO")[0]


# ── the CUDA runtime has to be findable, not merely installed ────────────
#
# Found by installing onnxruntime-genai-cuda and watching it fail. The CUDA
# libraries arrive as separate `nvidia-*` wheels that unpack to
# site-packages/nvidia/<pkg>/lib — a directory no loader searches — so ONNX
# Runtime failed at model init with "Failed to load library: libcublasLt.so.13"
# on an installation that was, by every check Keylane made, complete.


def test_preloading_is_idempotent() -> None:
    """It runs on every CUDA load; doing the work twice would be waste."""
    import runtimes.onnx_rt as onnx

    monkeyed = onnx._preloaded_cuda
    try:
        onnx._preloaded_cuda = False
        first = onnx.preload_cuda_libraries()
        second = onnx.preload_cuda_libraries()
        assert second == 0
        assert first >= 0
    finally:
        onnx._preloaded_cuda = monkeyed


def test_a_listed_provider_is_not_a_loadable_one(monkeypatch) -> None:
    """`get_available_providers` reports what was *compiled in*.

    Believing it gave a Settings panel that offered CUDA and a stack trace
    when it was picked.
    """
    import runtimes.onnx_rt as onnx

    monkeypatch.setattr(onnx, "_cuda_runtime_loadable", lambda: False)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime_genai", _FakeGenai())
    assert not onnx.cuda_provider_available()


class _FakeGenai:
    """Stands in for onnxruntime-genai 0.15, which has no provider list."""


def test_cuda_needs_both_listed_and_loadable(monkeypatch) -> None:
    import sys

    import runtimes.onnx_rt as onnx

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", _FakeGenai())
    monkeypatch.setattr(onnx, "_cuda_runtime_loadable", lambda: True)

    class _Ort:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", _Ort())
    assert onnx.cuda_provider_available()


# ── running out of VRAM is the normal failure on a shared GPU ────────────


def _oom() -> RuntimeError:
    return RuntimeError(
        "Exception during initialization: bfc_arena.cc:359 ... Failed to "
        "allocate memory for requested buffer of size 12582912"
    )


def test_an_out_of_vram_load_says_so(tmp_path, monkeypatch) -> None:
    """ONNX Runtime names the *last* small allocation it tried.

    "failed to allocate 12582912" on a 24 GB card reads as a bug in Keylane
    rather than as a browser and two model servers holding the memory.
    """
    import runtimes.onnx_rt as onnx

    model = tmp_path / "m"
    model.mkdir()
    (model / "model.onnx.data").write_bytes(b"\0" * 2048)
    monkeypatch.setattr(onnx, "free_vram_mb", lambda: 900)

    message = onnx._explain_load_failure(_oom(), model, "CUDA")
    assert "not enough free VRAM" in message
    assert "900 MiB is free" in message
    # The original is kept: it is what a bug report needs.
    assert "bfc_arena" in message


def test_the_vram_message_survives_nvidia_smi_being_absent(tmp_path, monkeypatch) -> None:
    import runtimes.onnx_rt as onnx

    model = tmp_path / "m"
    model.mkdir()
    monkeypatch.setattr(onnx, "free_vram_mb", lambda: None)
    assert "not enough free VRAM" in onnx._explain_load_failure(_oom(), model, "CUDA")


def test_other_errors_are_passed_through_untouched(tmp_path) -> None:
    import runtimes.onnx_rt as onnx

    original = RuntimeError("genai_config.json names no model file")
    assert onnx._explain_load_failure(original, tmp_path, "CUDA") == str(original)


def test_an_allocation_failure_on_cpu_is_not_a_vram_problem(tmp_path) -> None:
    import runtimes.onnx_rt as onnx

    assert "VRAM" not in onnx._explain_load_failure(_oom(), tmp_path, "CPU")
