"""The runtime seam: which stack claims a model, and what it does with it."""

from __future__ import annotations

import json
from pathlib import Path

import time

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


# ── talking to a model the way it was trained ────────────────────────────
#
# Messages used to be concatenated as "System: … User: … Assistant:", which is
# off-distribution for every instruct model in the catalog. Each ships its own
# template in the export, so the pipeline is asked to render one.


class _TemplatePipe:
    """A pipeline that knows its model's template, like OpenVINO GenAI does."""

    kind = "llm"

    def __init__(self) -> None:
        self.prompt = ""

    def apply_chat_template(self, messages):
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        ) + "<|im_start|>assistant\n"

    def count_tokens(self, text: str) -> int:
        # One "token" per four characters is wrong in general and exact enough
        # for a test that only cares which turns survive the trim.
        return len(text) // 4

    def generate(self, prompt, *, max_new_tokens=512, images=None, on_token=None):
        self.prompt = prompt
        if on_token:
            for piece in ("Par", "is."):
                on_token(piece)
        return "Paris."

    def close(self) -> None:
        pass


class _PlainPipe(_TemplatePipe):
    """An export with no template at all — the fallback has to still work."""

    def apply_chat_template(self, messages):
        return None

    def count_tokens(self, text: str):
        return None


def _resident(runtime, pipe) -> None:
    runtime._pipe = pipe
    runtime._model_id = "qwen2.5-7b-instruct"
    runtime._runtime_id = "openvino"
    runtime._device = "NPU"
    runtime._pipeline_kind = "llm"
    runtime._status = "ready"


def test_chat_uses_the_models_own_template(isolated_config):
    from models.catalog import LocalModelRuntime

    runtime = LocalModelRuntime()
    pipe = _TemplatePipe()
    _resident(runtime, pipe)

    runtime.chat([
        {"role": "system", "content": "You are Keylane."},
        {"role": "user", "content": "Where is Paris?"},
    ])

    assert "<|im_start|>system" in pipe.prompt
    assert pipe.prompt.endswith("<|im_start|>assistant\n")
    # The shape it used to send, which no instruct model was trained on.
    assert "System: " not in pipe.prompt


def test_chat_falls_back_when_the_export_has_no_template(isolated_config):
    from models.catalog import LocalModelRuntime

    runtime = LocalModelRuntime()
    pipe = _PlainPipe()
    _resident(runtime, pipe)

    runtime.chat([
        {"role": "system", "content": "You are Keylane."},
        {"role": "user", "content": "Where is Paris?"},
    ])

    assert "System: You are Keylane." in pipe.prompt
    assert pipe.prompt.endswith("Assistant:")


def test_the_trim_keeps_the_system_block_and_the_newest_turns(isolated_config):
    from models.catalog import LocalModelRuntime

    runtime = LocalModelRuntime()
    pipe = _TemplatePipe()
    _resident(runtime, pipe)
    runtime.prompt_budget_tokens = lambda: 120  # type: ignore[method-assign]

    history = [{"role": "system", "content": "SYSTEM BLOCK"}]
    for i in range(12):
        history.append({"role": "user", "content": f"question number {i} " + "x" * 60})
    runtime.chat(history)

    assert "SYSTEM BLOCK" in pipe.prompt
    assert "question number 11" in pipe.prompt
    # The oldest turns are what gets dropped, not the newest.
    assert "question number 0 " not in pipe.prompt


def test_chat_streams_every_piece_to_the_callback(isolated_config):
    from models.catalog import LocalModelRuntime

    runtime = LocalModelRuntime()
    _resident(runtime, _TemplatePipe())

    seen: list[str] = []
    answer = runtime.chat([{"role": "user", "content": "hi"}], on_token=seen.append)

    assert seen == ["Par", "is."]
    assert answer == "Paris."


def test_the_token_budget_is_counted_not_guessed(isolated_config):
    """The character budget is a pessimistic conversion; tokens are the truth."""
    from npu.limits import CHARS_PER_TOKEN, prompt_budget_chars, prompt_budget_tokens

    tokens = prompt_budget_tokens("NPU", "llm")
    chars = prompt_budget_chars("NPU", "llm")
    assert chars == int(tokens * CHARS_PER_TOKEN)
    assert tokens > 1000


# ── the curated catalog ──────────────────────────────────────────────────


def test_the_default_model_is_one_the_npu_can_run(isolated_config):
    """The default lands on the NPU, so it has to be a symmetric export.

    Intel's NPU guide requires symmetric INT4 or NF4 at group size -1 or 128.
    Every OpenVINO entry in this catalog was asymmetric once, the default
    included — they load and then run far below the hardware.
    """
    from models.catalog import catalog_default_model_id, get_model

    entry = get_model(catalog_default_model_id())
    assert entry is not None
    assert entry.npu_ready, f"{entry.id} is the default but is not an NPU export"


def test_every_entry_declares_how_it_was_quantized(isolated_config):
    """A claim that can't be checked is worse than no claim."""
    from models.catalog import load_catalog

    _, _, entries = load_catalog()
    for entry in (e for e in entries if e.source == "curated"):
        assert entry.quantization, f"{entry.id} does not say how it was quantized"


def test_npu_ready_entries_are_symmetric_or_purpose_built(isolated_config):
    from models.catalog import load_catalog

    _, _, entries = load_catalog()
    for entry in (e for e in entries if e.npu_ready):
        quant = entry.quantization.upper()
        assert "SYM" in quant or "NF4" in quant or "FOR NPU" in quant, (
            f"{entry.id} claims npu_ready with quantization {entry.quantization!r}"
        )
    for entry in (e for e in entries if not e.npu_ready):
        assert "ASYM" in entry.quantization.upper() or not entry.npu_ready


# ── a failed generate must not break the next one ────────────────────────
#
# A VLM pipeline keeps a tokenized history between generate() calls and
# asserts each new prompt is at least as long as the history it holds. It
# clears that itself at the end of a successful call — but a call that throws
# never gets there, so every later turn died on "Prompt ids size is less than
# tokenized history size", which is the previous prompt still sitting in the
# pipeline. One failure broke the assistant until the daemon was restarted.


class _StatefulPipe:
    """A pipeline that carries state, and refuses a prompt shorter than it.

    This is VLMPipeline's contract, reduced to the part that matters: state
    survives a call, a failure leaves it behind, and finish_chat drops it.
    """

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.history = 0
        self.finish_chat_calls = 0
        self.attempts: list[int] = []
        self._fail_on = fail_on or set()

    def finish_chat(self) -> None:
        self.finish_chat_calls += 1
        self.history = 0

    def generate(self, prompt, *, max_new_tokens=512, streamer=None, **kwargs):
        attempt = len(self.attempts) + 1
        self.attempts.append(len(prompt))
        if len(prompt) < self.history:
            raise RuntimeError("Prompt ids size is less than tokenized history size")
        if attempt in self._fail_on:
            # State is left behind exactly as a real mid-generate failure does.
            self.history = len(prompt)
            raise RuntimeError("L0 pfnAppendGraphExecute result: ZE_RESULT_ERROR_UNINITIALIZED")
        self.history = 0
        return "ok"


def _vlm(pipe):
    from runtimes.openvino_rt import OpenVinoPipeline

    return OpenVinoPipeline(pipe, "vlm")


def test_a_failed_generate_does_not_poison_the_next_turn():
    inner = _StatefulPipe(fail_on={1})
    pipeline = _vlm(inner)

    with pytest.raises(RuntimeError, match="ZE_RESULT_ERROR_UNINITIALIZED"):
        pipeline.generate("a long prompt " * 20)

    # The next turn is short — which is what a new session looks like — and
    # before this fix it died of the previous turn's leftovers.
    assert pipeline.generate("Hi.") == "ok"
    assert inner.finish_chat_calls == 1


def test_the_original_error_is_the_one_raised():
    """The caller must see the real failure, not a cascade from it."""
    inner = _StatefulPipe(fail_on={1})
    pipeline = _vlm(inner)

    with pytest.raises(RuntimeError) as caught:
        pipeline.generate("a long prompt " * 20)
    assert "Prompt ids size" not in str(caught.value)


def test_state_is_only_cleared_after_a_failure():
    """Clearing costs a state reset, so the happy path must not pay for it."""
    inner = _StatefulPipe()
    pipeline = _vlm(inner)

    for _ in range(3):
        pipeline.generate("Hi.")
    assert inner.finish_chat_calls == 0


def test_a_pipeline_that_cannot_clear_still_reports_the_failure():
    """An older GenAI has no finish_chat. That must not mask the real error."""

    class _NoChatApi(_StatefulPipe):
        def finish_chat(self):
            raise AttributeError("finish_chat")

    inner = _NoChatApi(fail_on={1})
    pipeline = _vlm(inner)

    with pytest.raises(RuntimeError, match="ZE_RESULT_ERROR_UNINITIALIZED"):
        pipeline.generate("a long prompt " * 20)
    # The clear failed, so the next call raises the pipeline's own complaint
    # rather than something swallowed here.
    with pytest.raises(RuntimeError, match="Prompt ids size"):
        pipeline.generate("Hi.")


# ── one generation at a time ─────────────────────────────────────────────
#
# There is one pipeline and one NPU behind it. Asking for a second answer
# while the first is running fails with "Infer Request is busy" and takes both
# turns down, not just the second. Reachable from a scheduled job firing
# mid-question, a follow-up sent early, or anything using
# /v1/chat/completions while the HUD is open.


class _SingleUsePipe(_TemplatePipe):
    """A pipeline that refuses to be used twice at once, like the real one."""

    def __init__(self) -> None:
        super().__init__()
        self.busy = False
        self.concurrent = False
        self.calls = 0

    def generate(self, prompt, *, max_new_tokens=512, images=None, on_token=None):
        if self.busy:
            self.concurrent = True
            raise RuntimeError("Infer Request is busy")
        self.busy = True
        try:
            self.calls += 1
            time.sleep(0.05)
            return "ok"
        finally:
            self.busy = False


def test_two_turns_at_once_queue_instead_of_colliding(isolated_config):
    import threading as _threading

    from models.catalog import LocalModelRuntime

    runtime = LocalModelRuntime()
    pipe = _SingleUsePipe()
    _resident(runtime, pipe)

    results: list[str] = []
    errors: list[Exception] = []

    def _turn() -> None:
        try:
            results.append(runtime.chat([{"role": "user", "content": "hi"}]))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [_threading.Thread(target=_turn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not pipe.concurrent, "two generations overlapped on one pipeline"
    assert errors == []
    assert results == ["ok"] * 4
    assert pipe.calls == 4


def test_a_wedged_generation_does_not_strand_the_next_turn_forever(isolated_config):
    """The queue has a deadline, and it says so rather than hanging."""
    import models.catalog as catalog
    from models.catalog import LocalModelRuntime

    runtime = LocalModelRuntime()
    _resident(runtime, _TemplatePipe())
    runtime._infer_lock.acquire()  # stand in for a generation that never ends

    original = catalog.INFER_QUEUE_TIMEOUT
    catalog.INFER_QUEUE_TIMEOUT = 0.05
    try:
        with pytest.raises(RuntimeError, match="did not free up"):
            runtime.chat([{"role": "user", "content": "hi"}])
    finally:
        catalog.INFER_QUEUE_TIMEOUT = original
        runtime._infer_lock.release()
