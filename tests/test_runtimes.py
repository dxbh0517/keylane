"""The runtime seam: which stack claims a model, and what it does with it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtimes import backend_for, detect_runtime, normalise_runtime_id, runtime_ids


@pytest.fixture()
def isolated_config(monkeypatch, tmp_path: Path):
    """Real config/*.toml, throwaway settings.json and models directory."""
    from daemon import config as config_module
    from daemon import paths

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(paths, "CONFIG_DIR", Path(__file__).resolve().parents[1] / "config")
    monkeypatch.setattr(paths, "SETTINGS_PATH", settings)
    # daemon.config binds SETTINGS_PATH at import, so patching paths alone
    # leaves it reading the user's real settings.json.
    monkeypatch.setattr(config_module, "SETTINGS_PATH", settings)
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")

    import models.catalog as catalog

    monkeypatch.setattr(catalog, "MODELS_DIR", tmp_path / "models")
    return settings


def _openvino_export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (root / "openvino_model.bin").write_bytes(b"\0" * 16384)
    return root


def _onnx_export(root: Path, *, vision: bool = False, context: int = 4096) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model: dict[str, object] = {
        "context_length": context,
        "decoder": {"filename": "model.onnx"},
    }
    if vision:
        model["vision"] = {"filename": "vision.onnx"}
    (root / "genai_config.json").write_text(
        json.dumps({"model": model}), encoding="utf-8"
    )
    (root / "model.onnx").write_bytes(b"\0" * 16384)
    (root / "model.onnx.data").write_bytes(b"\0" * 16384)
    if vision:
        (root / "vision.onnx").write_bytes(b"\0" * 16384)
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


# ── which runtime owns an export ─────────────────────────────────────────


def test_runtime_ids_are_the_two_shipped():
    assert runtime_ids() == ["openvino", "onnxruntime"]


def test_detect_runtime_reads_the_layout(tmp_path: Path):
    assert detect_runtime(_openvino_export(tmp_path / "ov")) == "openvino"
    assert detect_runtime(_onnx_export(tmp_path / "onnx")) == "onnxruntime"
    (tmp_path / "empty").mkdir()
    assert detect_runtime(tmp_path / "empty") is None


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("openvino", "openvino"),
        ("OV", "openvino"),
        ("openvino-genai", "openvino"),
        ("onnx", "onnxruntime"),
        ("onnxruntime-genai", "onnxruntime"),
        ("ORT", "onnxruntime"),
        # An unknown or absent runtime falls back rather than raising, so a
        # config from a future version still starts.
        ("tensorrt", "openvino"),
        ("", "openvino"),
        (None, "openvino"),
    ],
)
def test_normalise_runtime_id(given, expected):
    assert normalise_runtime_id(given) == expected


# ── ONNX Runtime: validating a download ──────────────────────────────────


def test_onnx_missing_weights_on_a_complete_export(tmp_path: Path):
    backend = backend_for("onnxruntime")
    assert backend.missing_weights(_onnx_export(tmp_path / "m")) == []


def test_onnx_missing_weights_without_a_config(tmp_path: Path):
    backend = backend_for("onnxruntime")
    (tmp_path / "m").mkdir()
    assert backend.missing_weights(tmp_path / "m") == ["genai_config.json"]


def test_onnx_missing_weights_finds_a_torn_external_data_blob(tmp_path: Path):
    backend = backend_for("onnxruntime")
    model = _onnx_export(tmp_path / "m")
    # The graph arrived; the weights beside it did not finish.
    (model / "model.onnx.data").write_bytes(b"\0" * 10)
    missing = backend.missing_weights(model)
    assert missing == ["model.onnx.data (empty or truncated)"]


def test_onnx_missing_weights_finds_weights_that_never_arrived(tmp_path: Path):
    """The failure the config alone cannot see.

    A genai export keeps its weights in a blob beside the graph that
    genai_config.json never names. Without reading the graph, a download that
    fetched the 170 KB graph and none of the gigabyte beside it looks complete.
    """
    backend = backend_for("onnxruntime")
    model = _onnx_export(tmp_path / "m")
    # A real graph stores the location as a length-prefixed string.
    (model / "model.onnx").write_bytes(
        b"\x08\x09onnx\x0f" + b"model.onnx.data" + b"\x00" * 16384
    )
    assert backend.missing_weights(model) == []

    (model / "model.onnx.data").unlink()
    assert backend.missing_weights(model) == ["model.onnx.data"]


def test_onnx_external_data_scan_skips_an_embedded_graph(tmp_path: Path):
    from runtimes.onnx_rt import _WEIGHTLESS_GRAPH_BYTES, external_data_files

    model = _onnx_export(tmp_path / "m")
    graph = model / "model.onnx"
    graph.write_bytes(b"\x0fmodel.onnx.data" + b"\x00" * 8192)
    assert external_data_files(graph) == {"model.onnx.data"}

    # A graph this size *is* the weights; reading it to look for a pointer
    # would mean loading gigabytes to learn nothing.
    assert _WEIGHTLESS_GRAPH_BYTES > 1_000_000


def test_onnx_missing_weights_finds_an_absent_graph(tmp_path: Path):
    backend = backend_for("onnxruntime")
    model = _onnx_export(tmp_path / "m")
    (model / "model.onnx").unlink()
    assert "model.onnx" in backend.missing_weights(model)


def test_onnx_kind_and_objection_for_a_vision_export(tmp_path: Path):
    backend = backend_for("onnxruntime")
    text_only = _onnx_export(tmp_path / "text")
    multimodal = _onnx_export(tmp_path / "vision", vision=True)

    assert backend.model_kind(text_only) == "llm"
    assert backend.static_objection(text_only) is None
    # Vision ONNX exports load, but Keylane's image path is OpenVINO-only —
    # better to say so before a multi-gigabyte compile than after.
    assert backend.model_kind(multimodal) == "vlm"
    assert "vision" in (backend.static_objection(multimodal) or "").lower()


def test_onnx_prompt_budget_follows_the_declared_context(tmp_path: Path):
    backend = backend_for("onnxruntime")
    small = _onnx_export(tmp_path / "small", context=2048)
    large = _onnx_export(tmp_path / "large", context=16384)

    assert backend.prompt_budget_chars("NPU", "llm", small) < backend.prompt_budget_chars(
        "NPU", "llm", large
    )
    # The budget leaves room for the reply, so it is never the whole context.
    from npu.limits import CHARS_PER_TOKEN

    assert backend.prompt_budget_chars("NPU", "llm", small) < 2048 * CHARS_PER_TOKEN


# ── picking a build out of a Hugging Face repo ───────────────────────────


def test_openvino_repo_variants_prefer_the_root_export():
    backend = backend_for("openvino")
    variants = backend.repo_variants(
        ["openvino_model.xml", "openvino_model.bin", "nested/openvino_model.xml"]
    )
    best = max(variants, key=lambda v: v.score)
    assert best.subfolder == ""
    assert {v.subfolder for v in variants} == {"", "nested"}


def test_onnx_repo_variants_reject_vendor_locked_builds():
    backend = backend_for("onnxruntime")
    variants = backend.repo_variants(
        [
            "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/genai_config.json",
            "cuda/cuda-fp16/genai_config.json",
            "directml/directml-int4-awq-block-128/genai_config.json",
            "README.md",
        ]
    )
    ranked = sorted(variants, key=lambda v: -v.score)
    assert ranked[0].subfolder.startswith("cpu_and_mobile/")
    # A CUDA or DirectML build is not a worse choice, it is the wrong machine —
    # it has to sort below the runnable threshold, not merely last.
    from runtimes.onnx_rt import RUNNABLE_SCORE

    assert [v.subfolder for v in ranked if v.score >= RUNNABLE_SCORE] == [
        "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    ]


def test_onnx_repo_variants_read_qnn_as_a_foreign_target():
    backend = backend_for("onnxruntime")
    # Named for the NPU, but it is Qualcomm's — the folder has to be read
    # segment by segment or "npu/qnn-int4" scores as a win.
    (variant,) = backend.repo_variants(["npu/qnn-int4/genai_config.json"])
    from runtimes.onnx_rt import RUNNABLE_SCORE

    assert variant.score < RUNNABLE_SCORE


def test_allow_patterns_fetch_only_the_chosen_build():
    assert backend_for("onnxruntime").allow_patterns("gpu/int4") == ["gpu/int4/*"]
    # A root-level export is the whole repo, so nothing is filtered out.
    assert backend_for("openvino").allow_patterns("") is None


# ── the importer ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("OpenVINO/Qwen3-8B-int4-ov", "OpenVINO/Qwen3-8B-int4-ov"),
        ("  OpenVINO/Qwen3-8B-int4-ov  ", "OpenVINO/Qwen3-8B-int4-ov"),
        ("https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx", "microsoft/Phi-4-mini-instruct-onnx"),
        (
            "https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx/tree/main/gpu",
            "microsoft/Phi-4-mini-instruct-onnx",
        ),
    ],
)
def test_normalise_repo_accepts_the_ways_people_paste_them(given, expected):
    from models.importer import normalise_repo

    assert normalise_repo(given) == expected


@pytest.mark.parametrize("given", ["", "not-a-repo", "http://example.com/x", "a/b/c/d"])
def test_normalise_repo_rejects_junk(given):
    from models.importer import ImportError_, normalise_repo

    with pytest.raises(ImportError_):
        normalise_repo(given)


def test_suggest_model_id_is_filesystem_safe():
    from models.importer import suggest_model_id

    assert suggest_model_id("OpenVINO/Qwen3-8B-int4-ov") == "qwen3-8b-int4-ov"
    assert (
        suggest_model_id("microsoft/Phi-4-mini-instruct-onnx", "cpu_and_mobile/cpu-int4")
        == "phi-4-mini-instruct-onnx-cpu-int4"
    )


# ── the catalog ──────────────────────────────────────────────────────────


def test_catalog_entries_carry_their_runtime(isolated_config):
    from models.catalog import load_catalog

    _, _, entries = load_catalog()
    by_runtime = {rid: [e for e in entries if e.runtime == rid] for rid in runtime_ids()}
    assert by_runtime["openvino"], "the curated list lost its OpenVINO models"
    assert by_runtime["onnxruntime"], "the curated list has no ONNX Runtime models"
    assert all(e.source == "curated" for e in entries)
    # Every ONNX entry names the build to fetch; the repos ship several.
    assert all(e.subfolder for e in by_runtime["onnxruntime"])


def test_model_dir_descends_into_the_subfolder(isolated_config):
    from models.catalog import get_model

    onnx = get_model("phi-4-mini-onnx")
    assert onnx is not None
    assert onnx.model_dir == onnx.local_path / onnx.subfolder
    assert onnx.local_path.name == "phi-4-mini-onnx"

    ov = get_model("qwen2.5-7b-instruct")
    assert ov is not None and ov.model_dir == ov.local_path


def test_imported_models_join_the_catalog(isolated_config):
    from daemon.config import save_settings
    from models.catalog import get_model, load_catalog

    before = len(load_catalog()[2])
    save_settings(
        "models",
        {
            "imported": [
                {
                    "id": "my-model",
                    "name": "My Model",
                    "hf_repo": "someone/my-model-onnx",
                    "runtime": "onnxruntime",
                    "subfolder": "cpu/int4",
                    "params_b": 3,
                }
            ]
        },
    )
    after = load_catalog()[2]
    assert len(after) == before + 1

    entry = get_model("my-model")
    assert entry is not None
    assert entry.source == "imported"
    assert entry.runtime == "onnxruntime"
    assert entry.model_dir.name == "int4"


def test_imported_model_cannot_shadow_a_curated_one(isolated_config):
    from daemon.config import save_settings
    from models.catalog import get_model

    save_settings(
        "models",
        {
            "imported": [
                {
                    "id": "qwen2.5-7b-instruct",
                    "name": "Impostor",
                    "hf_repo": "someone/else",
                    "runtime": "onnxruntime",
                }
            ]
        },
    )
    entry = get_model("qwen2.5-7b-instruct")
    assert entry is not None
    assert entry.source == "curated"
    assert entry.hf_repo == "OpenVINO/Qwen2.5-7B-Instruct-int4-ov"


def test_resolve_device_prefers_the_model_then_the_runtime_setting(isolated_config):
    from daemon.config import save_settings
    from models.catalog import ModelEntry, get_model

    save_settings("models", {"devices": {"openvino": "GPU", "onnxruntime": "CPU"}})

    ov = get_model("qwen2.5-7b-instruct")
    onnx = get_model("phi-4-mini-onnx")
    assert ov is not None and onnx is not None
    assert ov.resolve_device("NPU") == "GPU"
    assert onnx.resolve_device("NPU") == "CPU"

    # A device on the entry itself wins over both.
    pinned = ModelEntry(
        id="pinned", name="Pinned", hf_repo="a/b", params_b=1, device="NPU", runtime="openvino"
    )
    assert pinned.resolve_device("GPU") == "NPU"


def test_resolve_device_ignores_a_device_the_runtime_cannot_target(isolated_config):
    from daemon.config import save_settings
    from models.catalog import ModelEntry

    # Blank means "no preference", so the catalog default is what gets offered.
    save_settings("models", {"devices": {"openvino": ""}})
    entry = ModelEntry(id="x", name="X", hf_repo="a/b", params_b=1, runtime="openvino")
    assert entry.resolve_device("CPU") == "CPU"
    # OpenVINO GenAI has no AUTO; falling through to its own default beats
    # passing a device string the pipeline constructor would reject.
    assert entry.resolve_device("AUTO") == "NPU"


def test_active_download_file_finds_a_lock_inside_a_subfolder(tmp_path: Path):
    from models.catalog import _active_download_file

    root = tmp_path / "phi"
    lock_dir = root / ".cache/huggingface/download/cpu_and_mobile/cpu-int4"
    lock_dir.mkdir(parents=True)
    (lock_dir / "model.onnx.data.lock").touch()

    # Hugging Face mirrors the repo's own layout under .cache, so the lock for
    # a nested build is nested too.
    assert _active_download_file(root, ["model.onnx.data"]) == "model.onnx.data"
