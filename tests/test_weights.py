from pathlib import Path

from npu.weights import is_model_complete, missing_weights


def test_model_kind_llm_layout(tmp_path: Path):
    from npu.kind import model_kind

    model = tmp_path / "llm"
    model.mkdir()
    (model / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    assert model_kind(model) == "llm"


def test_model_kind_vlm_layout(tmp_path: Path):
    from npu.kind import model_kind

    model = tmp_path / "vlm"
    model.mkdir()
    (model / "openvino_language_model.xml").write_text("<net/>", encoding="utf-8")
    (model / "openvino_vision_embeddings_model.xml").write_text("<net/>", encoding="utf-8")
    assert model_kind(model) == "vlm"


def test_static_objection_allows_vlm(tmp_path: Path):
    from npu.probe import static_objection

    model = tmp_path / "vlm"
    model.mkdir()
    (model / "openvino_tokenizer.xml").write_text(
        '<net><openvino_version value="2024.5"/></net>',
        encoding="utf-8",
    )
    (model / "openvino_language_model.xml").write_text("<net/>", encoding="utf-8")
    assert static_objection(model) is None


def test_warm_timeout_vlm_is_longer():
    from npu.probe import VLM_WARM_TIMEOUT, WARM_TIMEOUT, warm_timeout_for

    assert warm_timeout_for("vlm") == VLM_WARM_TIMEOUT
    assert warm_timeout_for("llm") == WARM_TIMEOUT
    assert VLM_WARM_TIMEOUT > WARM_TIMEOUT


def test_pipeline_init_kwargs_vlm_uses_device_properties():
    from npu.pipeline_config import pipeline_init_kwargs

    kwargs = pipeline_init_kwargs("NPU", Path("/tmp/cache"), "vlm")
    assert "config" in kwargs
    props = kwargs["config"]["DEVICE_PROPERTIES"]["NPU"]
    assert props["GENERATE_HINT"] == "FAST_COMPILE"
    assert props["CACHE_DIR"] == "/tmp/cache"


def test_pipeline_init_kwargs_llm():
    from npu.limits import NPU_MAX_PROMPT_TOKENS
    from npu.pipeline_config import pipeline_init_kwargs

    kwargs = pipeline_init_kwargs("NPU", Path("/tmp/cache"), "llm")
    assert kwargs["MAX_PROMPT_LEN"] == NPU_MAX_PROMPT_TOKENS
    assert kwargs["CACHE_DIR"] == "/tmp/cache"


def test_the_vlm_gets_a_prompt_limit_too():
    """Without one it compiles at 1024 tokens and throws on the system prompt."""
    from npu.limits import NPU_MAX_PROMPT_TOKENS
    from npu.pipeline_config import pipeline_init_kwargs

    props = pipeline_init_kwargs("NPU", Path("/tmp/cache"), "vlm")["config"][
        "DEVICE_PROPERTIES"
    ]["NPU"]
    assert props["MAX_PROMPT_LEN"] == NPU_MAX_PROMPT_TOKENS


def test_the_budget_and_the_compiled_limit_agree():
    """They drifted before: a 10000-char budget against a 1024-token pipeline."""
    from npu.limits import CHARS_PER_TOKEN, NPU_MAX_PROMPT_TOKENS, npu_prompt_budget_chars

    assert npu_prompt_budget_chars() / CHARS_PER_TOKEN < NPU_MAX_PROMPT_TOKENS


def test_pipeline_init_kwargs_vlm_gpu_omits_npu_hints():
    from npu.pipeline_config import pipeline_init_kwargs

    kwargs = pipeline_init_kwargs("GPU", Path("/tmp/cache"), "vlm")
    props = kwargs["config"]["DEVICE_PROPERTIES"]["GPU"]
    assert "GENERATE_HINT" not in props
    assert props["CACHE_DIR"] == "/tmp/cache"


def test_default_model_id_uses_settings_override(tmp_path: Path, monkeypatch):
    from daemon import config as config_module
    from daemon import paths
    from daemon.config import save_settings
    from models.catalog import catalog_default_model_id, default_model_id, load_catalog

    monkeypatch.setattr(paths, "CONFIG_DIR", Path(__file__).resolve().parents[1] / "config")
    monkeypatch.setattr(paths, "SETTINGS_PATH", tmp_path / "settings.json")
    # daemon.config binds SETTINGS_PATH at import, so patching paths alone
    # leaves it reading the user's real settings.json.
    monkeypatch.setattr(config_module, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")

    catalog_default = catalog_default_model_id()
    assert default_model_id() == catalog_default

    save_settings("models", {"default_model_id": "qwen3.5-9b"})
    assert default_model_id() == "qwen3.5-9b"

    save_settings("models", {"default_model_id": "not-a-real-model"})
    assert default_model_id() == catalog_default


def test_downloaded_bytes_counts_files_and_incomplete(tmp_path: Path):
    from models.catalog import _downloaded_bytes

    model = tmp_path / "qwen"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    cache = model / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (cache / "part.incomplete").write_bytes(b"\0" * 2048)

    assert _downloaded_bytes(model) == 2048 + len("{}")


def test_active_download_file_prefers_locked_missing_weight(tmp_path: Path):
    from models.catalog import _active_download_file

    model = tmp_path / "qwen"
    model.mkdir()
    (model / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    cache = model / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (cache / "openvino_model.bin.lock").touch()

    assert _active_download_file(model) == "openvino_model.bin"


def test_missing_weights_detects_partial_download(tmp_path: Path):
    model = tmp_path / "qwen"
    model.mkdir()
    (model / "openvino_model.xml").write_text(
        '<net><layer><data value="openvino_model.bin"/></layer></net>',
        encoding="utf-8",
    )
    assert not is_model_complete(model)
    missing = missing_weights(model)
    assert "openvino_model.bin" in missing


def test_complete_when_bins_present(tmp_path: Path):
    model = tmp_path / "qwen"
    model.mkdir()
    (model / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (model / "openvino_model.bin").write_bytes(b"\0" * 16384)
    assert is_model_complete(model)


def _fake_genai(monkeypatch, seen: dict):
    """Stand in for openvino_genai so construction can be observed, not run."""
    import sys
    import types

    class _Pipe:
        def __init__(self, path, device, **kwargs):
            seen["path"] = path
            seen["device"] = device
            seen["kwargs"] = kwargs

    module = types.ModuleType("openvino_genai")
    module.LLMPipeline = type("LLMPipeline", (_Pipe,), {})
    module.VLMPipeline = type("VLMPipeline", (_Pipe,), {})
    monkeypatch.setitem(sys.modules, "openvino_genai", module)
    return module


def test_create_pipeline_passes_the_declared_options(monkeypatch, tmp_path: Path):
    from npu.pipeline_config import create_pipeline, pipeline_init_kwargs

    seen: dict = {}
    genai = _fake_genai(monkeypatch, seen)
    cache = tmp_path / "cache"

    pipe = create_pipeline(tmp_path / "model", "NPU", cache, "llm")
    assert isinstance(pipe, genai.LLMPipeline)
    assert seen["kwargs"] == pipeline_init_kwargs("NPU", cache, "llm")

    pipe = create_pipeline(tmp_path / "model", "NPU", cache, "vlm")
    assert isinstance(pipe, genai.VLMPipeline)
    assert seen["kwargs"] == pipeline_init_kwargs("NPU", cache, "vlm")


def test_probe_and_load_compile_the_same_thing(monkeypatch, tmp_path: Path):
    """The invariant that broke: one constructor, not two spellings of one.

    MAX_PROMPT_LEN is part of the compile cache key, so a probe that names a
    different value is not a cheaper check of the same thing — it compiles a
    blob the load cannot use, and the load then compiles again from scratch.
    """
    from npu.probe import _probe_source

    for kind in ("llm", "vlm"):
        source = _probe_source(kind)
        assert "create_pipeline" in source
        # Anything spelled out here is a value that can drift from the load's.
        assert "MAX_PROMPT_LEN" not in source
        assert "GENERATE_HINT" not in source
        assert "1024" not in source


def test_probe_child_can_import_the_keylane_tree():
    from daemon.paths import ROOT
    from npu.probe import _probe_env

    assert _probe_env()["PYTHONPATH"].split(":")[0] == str(ROOT)


def test_warm_timeout_brackets_a_real_compile():
    """Generous enough for a slow machine, tight enough to catch a wedge.

    This used to assert `>= 3600`, from a time when a 4096-token compile
    really did take half an hour. Measured 2026-09-03 on OpenVINO 2026.3 and
    an NPU 3720, a cold 7B compile is 60 seconds — so an hour of headroom
    meant a genuinely stuck compile looked like a slow one for an hour.

    The lower bound is ~5x a measured compile so a bigger model on a slower
    NPU still passes; the upper bound is what keeps the deadline useful.
    Re-measure with scripts/npu-bench.py before moving either.
    """
    from npu.probe import VLM_WARM_TIMEOUT, WARM_TIMEOUT

    assert 300 <= WARM_TIMEOUT <= 1800
    # A vision model compiles a vision tower too, and that is genuinely slower.
    assert VLM_WARM_TIMEOUT > WARM_TIMEOUT
