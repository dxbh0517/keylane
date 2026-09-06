"""ONNX Runtime GenAI, reaching the NPU through the OpenVINO execution provider.

The second local stack. Where OpenVINO GenAI wants an IR export, this one wants
an ONNX export with a ``genai_config.json`` beside it — the format Microsoft
publishes its models in, and the reason this runtime roughly doubles the list of
models Keylane can run.

Device selection is a provider decision rather than a constructor argument: the
model ships a config naming an execution provider, and Keylane replaces it with
the OpenVINO EP pointed at NPU, GPU or CPU. ``build_config`` is the single place
that happens, and the subprocess probe imports it rather than restating it, so
what gets compiled in the child is what gets loaded in the daemon.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from npu.kind import PipelineKind
from npu.limits import CHARS_PER_TOKEN, RESERVE_TOKENS
from npu.thinking import OutputStreamFilter
from runtimes.base import RepoVariant, RuntimeInfo
from runtimes.probe_runner import OK_MARKER, run_probe

logger = logging.getLogger(__name__)

INFO = RuntimeInfo(
    id="onnxruntime",
    name="ONNX Runtime GenAI",
    summary=(
        "Runs ONNX exports (genai_config.json). Reaches Intel silicon through "
        "the OpenVINO execution provider and an NVIDIA card through CUDA, so "
        "it is the one runtime here that can use a discrete GPU."
    ),
    install_hint=(
        "pip install onnxruntime-genai onnxruntime-openvino "
        "(add onnxruntime-genai-cuda for an NVIDIA GPU)"
    ),
    # CUDA is a device of this runtime, not a runtime of its own: it is an
    # execution provider of the same stack, exactly as OpenVINO is, and a
    # separate backend would be a second copy of this module.
    devices=("NPU", "GPU", "CUDA", "CPU", "AUTO"),
    default_device="NPU",
)

# Compile timeouts. ONNX exports are already graph-level artifacts, so the
# OpenVINO EP's compile is far cheaper than compiling an IR from scratch — but
# on the NPU it is still minutes, not seconds, the first time.
WARM_TIMEOUT = float(os.environ.get("KEYLANE_ONNX_WARM_TIMEOUT", "1800"))

# Below this a model file is a stub or a torn download, not weights.
_MIN_MODEL_BYTES = 8192

# Context length assumed when genai_config.json does not declare one.
_FALLBACK_CONTEXT_TOKENS = 4096

# What a *declared* context does not tell you is whether this machine can fill
# it. MiniCPM5-1B declares 131072, which is true of the model and useless as a
# budget: at that length the prompt alone is ~340k characters, and prefilling
# it on a CPU takes minutes per turn. The declared context stays the hard
# ceiling; this is the practical one, and the smaller of the two wins.
# Override with KEYLANE_ONNX_MAX_PROMPT_TOKENS.
MAX_PRACTICAL_PROMPT_TOKENS = int(
    os.environ.get("KEYLANE_ONNX_MAX_PROMPT_TOKENS", "8192")
)

# Models with a hybrid `<think>` mode default to thinking *on*. One ReAct turn
# is several model calls, so on a small local model that is the difference
# between an answer and a wait — and Keylane strips the reasoning before the
# user ever sees it, so the tokens buy nothing here. Set
# KEYLANE_ENABLE_THINKING=1 to leave the model's own default alone.
ENABLE_THINKING = os.environ.get("KEYLANE_ENABLE_THINKING", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")

# Loading a Hugging Face tokenizer costs a second or two; the chat template it
# carries never changes for a directory, so it is loaded once per model.
_HF_TOKENIZERS: dict[str, Any] = {}


def _hf_tokenizer(model_dir: Path) -> Any | None:
    """The transformers tokenizer beside an ONNX export, for its chat template.

    onnxruntime-genai's tokenizer cannot render one, and an export that came
    from an instruct model on the Hub ships tokenizer_config.json next to the
    graph. Absent that, there is no template to apply and None is the answer.
    """
    key = str(model_dir)
    if key in _HF_TOKENIZERS:
        return _HF_TOKENIZERS[key]
    tokenizer = None
    if (model_dir / "tokenizer_config.json").is_file():
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        except Exception:  # noqa: BLE001
            logger.debug("no usable transformers tokenizer in %s", model_dir, exc_info=True)
    _HF_TOKENIZERS[key] = tokenizer
    return tokenizer


# ONNX keeps a large model's weights outside the graph, in a blob the graph
# names but genai_config.json does not. A graph that big is the weights itself,
# so there is nothing to look for and no reason to read it.
_WEIGHTLESS_GRAPH_BYTES = 128 * 1024 * 1024
# The name is preceded by protobuf's length byte, which is not a word
# character — so anchoring on one keeps the match from swallowing whatever
# string happened to be serialised before it.
_EXTERNAL_DATA_RE = re.compile(rb"(?:^|[^\w.\-])([\w][\w.\-]{0,120}\.onnx[._]data)")


# ── model config on disk ─────────────────────────────────────────────────


def read_genai_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "genai_config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _declared_model_files(config: dict[str, Any]) -> list[str]:
    """The .onnx files genai_config.json points at, decoder first."""
    model = config.get("model")
    if not isinstance(model, dict):
        return []
    names: list[str] = []
    for key in ("decoder", "embedding", "vision", "speech", "encoder"):
        section = model.get(key)
        if isinstance(section, dict):
            filename = section.get("filename")
            if isinstance(filename, str) and filename.strip():
                names.append(filename.strip())
    return names


# The scan result, keyed by what would have to change for it to differ. The
# download monitor asks whether a model is complete several times a second, and
# re-reading the graph each time is the one expensive part of the answer.
_SCAN_CACHE: dict[tuple[str, int, int], set[str]] = {}


def external_data_files(graph: Path) -> set[str]:
    """The blobs a graph keeps its weights in, read out of the graph itself.

    Without this an ONNX model whose weights never downloaded reads as
    complete: the graph is a few hundred kilobytes, it is present and valid,
    and genai_config.json never mentions the gigabyte sitting beside it. The
    locations are stored as plain strings in the protobuf, so finding them
    needs no parser — only the discipline not to read a file that is itself
    the weights.
    """
    try:
        stat = graph.stat()
        if stat.st_size > _WEIGHTLESS_GRAPH_BYTES:
            return set()
        key = (str(graph), stat.st_size, int(stat.st_mtime))
        cached = _SCAN_CACHE.get(key)
        if cached is not None:
            return cached
        blob = graph.read_bytes()
    except OSError:
        return set()

    found = {match.decode("utf-8", "ignore") for match in _EXTERNAL_DATA_RE.findall(blob)}
    if len(_SCAN_CACHE) > 32:
        _SCAN_CACHE.clear()
    _SCAN_CACHE[key] = found
    return found


def _blob_present(model_dir: Path, name: str) -> Path | None:
    """The weights file *name* refers to, allowing for a scan that over-read.

    Reading a name out of a protobuf without parsing it can pick up the string
    serialised just before it, so a match may carry a prefix that is not part
    of the filename. Falling back to a file whose name ends the same way keeps
    that mistake from condemning a model that downloaded perfectly.
    """
    exact = model_dir / name
    if exact.is_file():
        return exact
    tail = name.split(".", 1)[-1]
    for candidate in sorted(model_dir.glob(f"*.{tail}")):
        if name.endswith(candidate.name):
            return candidate
    return None


def context_length(config: dict[str, Any]) -> int:
    model = config.get("model")
    if isinstance(model, dict):
        for key in ("context_length", "max_length"):
            value = model.get(key)
            if isinstance(value, int) and value > 0:
                return value
    search = config.get("search")
    if isinstance(search, dict):
        value = search.get("max_length")
        if isinstance(value, int) and value > 0:
            return value
    return _FALLBACK_CONTEXT_TOKENS


def usable_prompt_tokens(config: dict[str, Any]) -> int:
    """The prompt budget: the declared context, capped at what is practical.

    Kept separate from :func:`context_length` because the two answer different
    questions. `context_length` is what the model may attend to and bounds
    generation; this is what Keylane is willing to build, and on a long-context
    model those numbers are three orders of magnitude apart.
    """
    declared = max(context_length(config) - RESERVE_TOKENS, 256)
    return min(declared, MAX_PRACTICAL_PROMPT_TOKENS)


def _is_vision_model(config: dict[str, Any]) -> bool:
    model = config.get("model")
    return isinstance(model, dict) and isinstance(model.get("vision"), dict)


# ── provider selection ───────────────────────────────────────────────────

# Keylane's device names, mapped onto what the OpenVINO EP calls them. AUTO
# means "leave the config exactly as the model shipped it", which is the escape
# hatch for a build that was exported for some other provider entirely.
#
# `GPU` here means an *Intel* GPU through the OpenVINO EP. A discrete NVIDIA
# card is `CUDA` and goes to a different provider entirely — the distinction
# matters because OpenVINO happily enumerates an NVIDIA card as "GPU" and then
# cannot compile for it.
_OPENVINO_DEVICES = {"NPU": "NPU", "GPU": "GPU", "CPU": "CPU"}
CUDA_DEVICE = "CUDA"


def nvidia_present() -> bool:
    """Whether an NVIDIA driver is loaded. Cheap, and no CUDA import."""
    return Path("/proc/driver/nvidia/version").exists() or bool(
        shutil.which("nvidia-smi")
    )


def cuda_provider_available() -> bool:
    """Whether the installed onnxruntime-genai can actually target CUDA.

    The CUDA execution provider ships in a *different wheel* from the default
    one (`onnxruntime-genai-cuda`), so having onnxruntime-genai importable says
    nothing about whether CUDA is reachable. Asking the package is the only
    honest answer; a driver being present is not one.
    """
    try:
        import onnxruntime_genai as og  # noqa: PLC0415
    except ImportError:
        return False
    # Newer builds expose the provider list; older ones do not, and there the
    # append is the only way to find out — so assume yes and let build_config
    # report the real failure rather than refusing up front.
    providers = getattr(og, "get_available_providers", None)
    if providers is None:
        return True
    try:
        return any("cuda" in str(name).lower() for name in providers())
    except Exception:  # noqa: BLE001
        return True


def cuda_status() -> tuple[bool, str]:
    """``(usable, reason)`` for the CUDA device on this machine."""
    if not nvidia_present():
        return False, "no NVIDIA driver on this machine"
    if not cuda_provider_available():
        return False, "onnxruntime-genai has no CUDA provider (pip install onnxruntime-genai-cuda)"
    return True, ""


def build_config(model_dir: Path, device: str, cache: Path | None) -> Any:
    """An ``og.Config`` for this model on this device, or None to use defaults.

    Returning None is not a failure — it means the installed onnxruntime-genai
    predates the Config API, and the model must run with whatever provider its
    own genai_config.json names.
    """
    import onnxruntime_genai as og  # noqa: PLC0415

    if not hasattr(og, "Config"):
        logger.warning(
            "onnxruntime-genai has no Config API; running %s with the provider "
            "declared in genai_config.json",
            model_dir.name,
        )
        return None

    config = og.Config(str(model_dir))
    wanted = device.strip().upper()
    if wanted in {"", "AUTO"}:
        return config

    config.clear_providers()

    if wanted == CUDA_DEVICE:
        usable, reason = cuda_status()
        if not usable:
            raise RuntimeError(f"cannot run on CUDA: {reason}")
        config.append_provider("cuda")
        return config

    if wanted == "CPU" and not _openvino_ep_available():
        # No OpenVINO EP: plain ONNX Runtime already runs on CPU with no
        # provider appended, so leave the list empty rather than fail.
        return config

    ov_device = _OPENVINO_DEVICES.get(wanted)
    if ov_device is None:
        raise ValueError(f"{INFO.name} cannot target device {device!r}")

    config.append_provider("openvino")
    config.set_provider_option("openvino", "device_type", ov_device)
    # Without this the EP treats each token as an independent graph run and
    # the KV cache is rebuilt every step, which is slower than CPU.
    config.set_provider_option("openvino", "enable_causallm", "True")
    if cache:
        try:
            config.set_provider_option("openvino", "cache_dir", str(cache))
        except Exception:  # noqa: BLE001
            logger.debug("this OpenVINO EP build does not take cache_dir")
    return config


def _openvino_ep_available() -> bool:
    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError:
        # onnxruntime-genai bundles its own runtime; assume the EP it was
        # built with is there and let the load say otherwise.
        return True
    return "OpenVINOExecutionProvider" in onnxruntime.get_available_providers()


def open_model(model_dir: Path, device: str, cache: Path | None) -> tuple[Any, Any]:
    """Load the model and its tokenizer. Used by both the probe and the daemon."""
    import onnxruntime_genai as og  # noqa: PLC0415

    config = build_config(model_dir, device, cache)
    model = og.Model(config) if config is not None else og.Model(str(model_dir))
    return model, og.Tokenizer(model)


# ── the resident pipeline ────────────────────────────────────────────────


class OnnxPipeline:
    """A loaded ONNX Runtime GenAI model, driven one token at a time."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        context_tokens: int,
        model_dir: Path | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._context_tokens = context_tokens
        self._model_dir = model_dir
        self.kind: PipelineKind = "llm"

    # ── the model's own idea of a conversation ───────────────────────────

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str | None:
        """Render a conversation with the template shipped beside the export.

        onnxruntime-genai's own tokenizer has no template support, so the
        template is read from ``tokenizer_config.json`` — which every export
        that came from a Hugging Face instruct model carries — and rendered
        with transformers, which is already a dependency. Returns None if
        either half is missing, so the caller falls back rather than fails.
        """
        if self._model_dir is None:
            return None

        if not ENABLE_THINKING:
            # Direct render first: it is the only route that can pass a
            # template variable, and it does not need the tokenizer class to
            # be one this transformers knows.
            from runtimes.chat_template import render_for

            rendered = render_for(self._model_dir, messages, enable_thinking=False)
            if rendered is not None:
                return rendered

        try:
            hf = _hf_tokenizer(self._model_dir)
            if hf is None or not getattr(hf, "chat_template", None):
                return None
            kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if not ENABLE_THINKING:
                # Harmless on a template that does not read it: extra kwargs go
                # into the render context, and one nothing references is unused.
                kwargs["enable_thinking"] = False
            return str(hf.apply_chat_template(list(messages), **kwargs))
        except Exception:  # noqa: BLE001
            logger.debug("apply_chat_template failed; falling back", exc_info=True)
            return None

    def count_tokens(self, text: str) -> int | None:
        try:
            return int(len(self._tokenizer.encode(text)))
        except Exception:  # noqa: BLE001
            logger.debug("token count failed; falling back to characters", exc_info=True)
            return None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        import onnxruntime_genai as og  # noqa: PLC0415

        if images:
            logger.debug("%s runs text-only models; ignoring %d image(s)", INFO.id, len(images))

        tokens = self._tokenizer.encode(prompt)
        # max_length counts the prompt too, so a model asked for more than its
        # context holds fails at the first token rather than truncating.
        limit = min(len(tokens) + max_new_tokens, self._context_tokens)
        if limit <= len(tokens):
            raise RuntimeError(
                f"prompt is {len(tokens)} tokens but the model's context is "
                f"{self._context_tokens} — nothing left to generate into"
            )

        params = og.GeneratorParams(self._model)
        params.set_search_options(max_length=limit, do_sample=False)

        generator = self._append_prompt(params, tokens)
        stream = self._tokenizer.create_stream()
        output_filter = OutputStreamFilter() if on_token is not None else None
        pieces: list[str] = []

        while not generator.is_done():
            self._step(generator)
            piece = stream.decode(generator.get_next_tokens()[0])
            if not piece:
                continue
            pieces.append(piece)
            if on_token is not None:
                visible = output_filter.feed(piece) if output_filter else piece
                if visible:
                    on_token(visible)

        if output_filter is not None and on_token is not None:
            remainder = output_filter.flush()
            if remainder:
                on_token(remainder)

        return "".join(pieces)

    def _append_prompt(self, params: Any, tokens: Any) -> Any:
        """Feed the prompt in, across both generations of the API.

        onnxruntime-genai moved the prompt from ``GeneratorParams.input_ids`` to
        ``Generator.append_tokens``; which one is here depends on the installed
        wheel, and both spellings are still in the wild.
        """
        import onnxruntime_genai as og  # noqa: PLC0415

        generator = og.Generator(self._model, params)
        if hasattr(generator, "append_tokens"):
            generator.append_tokens(tokens)
            return generator
        params.input_ids = tokens
        return og.Generator(self._model, params)

    @staticmethod
    def _step(generator: Any) -> None:
        # compute_logits() was folded into generate_next_token(); calling it on
        # a newer build raises rather than being a harmless no-op.
        if hasattr(generator, "compute_logits"):
            generator.compute_logits()
        generator.generate_next_token()

    def close(self) -> None:
        self._tokenizer = None
        self._model = None


# ── the backend ──────────────────────────────────────────────────────────


class OnnxRuntimeBackend:
    """The ONNX Runtime GenAI runtime."""

    info = INFO

    def installed(self) -> tuple[bool, str]:
        try:
            import onnxruntime_genai as og  # noqa: PLC0415
        except ImportError as exc:
            return False, str(exc)
        version = getattr(og, "__version__", "?")
        if _openvino_ep_available():
            return True, f"onnxruntime-genai {version} with the OpenVINO EP"
        return True, (
            f"onnxruntime-genai {version} — no OpenVINO execution provider, "
            "so NPU and GPU are unavailable"
        )

    def cache_dir(self) -> Path:
        from daemon.paths import CACHE_DIR  # noqa: PLC0415

        path = CACHE_DIR.parent / "onnxruntime"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── recognising a model ──────────────────────────────────────────────

    def detect(self, model_dir: Path) -> bool:
        return model_dir.is_dir() and (model_dir / "genai_config.json").is_file()

    def model_kind(self, model_dir: Path) -> PipelineKind:
        return "vlm" if _is_vision_model(read_genai_config(model_dir)) else "llm"

    def missing_weights(self, model_dir: Path) -> list[str]:
        if not model_dir.is_dir():
            return ["model directory missing"]
        if not (model_dir / "genai_config.json").is_file():
            return ["genai_config.json"]
        config = read_genai_config(model_dir)
        if not config:
            return ["genai_config.json (unreadable)"]

        declared = _declared_model_files(config)
        if not declared:
            return ["genai_config.json names no model file"]

        missing: list[str] = []
        for name in declared:
            graph = model_dir / name
            problems = self._check_file(graph)
            missing.extend(problems)
            if problems:
                continue
            # The graph is here; the weights it points at may not be.
            for blob_name in sorted(external_data_files(graph)):
                blob = _blob_present(model_dir, blob_name)
                missing.extend(self._check_file(blob if blob else model_dir / blob_name))

        # Anything else that arrived half-written, whether the graph named it
        # or not.
        for pattern in ("*.onnx.data", "*.onnx_data", "*.bin"):
            for blob in sorted(model_dir.glob(pattern)):
                if blob.stat().st_size < _MIN_MODEL_BYTES:
                    missing.append(f"{blob.name} (empty or truncated)")

        if not any((model_dir / name).is_file() for name in _TOKENIZER_FILES):
            missing.append("tokenizer.json")
        return missing

    @staticmethod
    def _check_file(path: Path) -> list[str]:
        if not path.is_file():
            return [path.name]
        if path.stat().st_size < _MIN_MODEL_BYTES:
            return [f"{path.name} (empty or truncated)"]
        return []

    def purge_incomplete(self, model_dir: Path) -> None:
        if not model_dir.is_dir():
            return
        for pattern in ("*.onnx", "*.onnx.data", "*.onnx_data", "genai_config.json"):
            for path in model_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass

    def static_objection(self, model_dir: Path) -> str | None:
        config = read_genai_config(model_dir)
        if not config:
            return "no genai_config.json — this is not an ONNX Runtime GenAI export"
        if _is_vision_model(config):
            return (
                "this is a multimodal ONNX export; Keylane runs vision models "
                "on OpenVINO GenAI only"
            )
        return None

    # ── loading ──────────────────────────────────────────────────────────

    def warm_timeout_for(self, kind: PipelineKind) -> float:
        return WARM_TIMEOUT

    def probe(
        self,
        model_dir: Path,
        device: str,
        *,
        cache: Path | None,
        timeout: float | None = None,
        kind: PipelineKind | None = None,
        on_tick: Callable[[float], None] | None = None,
    ) -> tuple[bool, str]:
        from daemon.paths import ROOT  # noqa: PLC0415

        limit = timeout if timeout is not None else self.warm_timeout_for(kind or "llm")
        source = (
            "import sys\n"
            "from pathlib import Path\n"
            "from runtimes.onnx_rt import open_model\n"
            "path, device, cache = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "open_model(Path(path), device, Path(cache) if cache else None)\n"
            f'print("{OK_MARKER}")\n'
        )
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)
        return run_probe(
            [sys.executable, "-c", source, str(model_dir), device, str(cache or "")],
            timeout=limit,
            timeout_message=(
                f"timeout:the OpenVINO EP did not finish compiling within "
                f"{limit / 60:.0f} min (set KEYLANE_ONNX_WARM_TIMEOUT)"
            ),
            on_tick=on_tick,
            env=env,
        )

    def load(
        self,
        model_dir: Path,
        device: str,
        *,
        cache: Path | None,
        kind: PipelineKind,
    ) -> OnnxPipeline:
        model, tokenizer = open_model(model_dir, device, cache)
        return OnnxPipeline(
            model,
            tokenizer,
            context_tokens=context_length(read_genai_config(model_dir)),
            model_dir=model_dir,
        )

    def prompt_budget_tokens(self, device: str, kind: PipelineKind, model_dir: Path) -> int:
        """What this machine can actually prefill, not what the model declares."""
        return usable_prompt_tokens(read_genai_config(model_dir))

    def prompt_budget_chars(self, device: str, kind: PipelineKind, model_dir: Path) -> int:
        return int(usable_prompt_tokens(read_genai_config(model_dir)) * CHARS_PER_TOKEN)

    # ── Hugging Face repos ───────────────────────────────────────────────

    def repo_variants(self, filenames: Iterable[str]) -> list[RepoVariant]:
        """Every genai_config.json in the repo is one build of the model."""
        folders: set[str] = set()
        for name in filenames:
            if name.rsplit("/", 1)[-1] != "genai_config.json":
                continue
            folders.add(name[: -len("genai_config.json")].rstrip("/"))

        variants: list[RepoVariant] = []
        for folder in sorted(folders):
            variants.append(
                RepoVariant(
                    runtime=INFO.id,
                    subfolder=folder,
                    label=folder or "repository root",
                    score=_variant_score(folder),
                )
            )
        return variants

    def allow_patterns(self, subfolder: str) -> list[str] | None:
        if not subfolder:
            return None
        # The tokenizer and config sit beside the graph inside the build's own
        # folder, so the folder alone is a complete model.
        return [f"{subfolder}/*"]


# A build named for hardware this machine does not have. Microsoft's DirectML,
# Qualcomm's QNN, AMD's ROCm and WebGPU exports all sit in the same repos as
# the ones that run here, and none of them will load through any provider
# Keylane can offer.
_FOREIGN_TARGETS = ("directml", "dml", "qnn", "webgpu", "web", "rocm", "trt")

# CUDA is conditional, not foreign. It is unusable on a machine with no NVIDIA
# card and no CUDA provider — and on a machine with both it is the best build
# in the repo by a wide margin, which is the whole reason CUDA was added as a
# device. Judging it by folder name alone is what made a 5090 invisible.
_CUDA_TARGETS = ("cuda", "trt-cuda")

# Below this a variant is not a worse choice, it is the wrong machine.
RUNNABLE_SCORE = 0


def _variant_score(folder: str) -> int:
    """How well one build suits *this* machine, from its folder name alone.

    Repos publish the same model four or five ways and only name the target in
    the path. A CPU int4 build is what the OpenVINO EP wants — on the NPU as
    much as the CPU — while a DirectML or QNN build is not a compromise but
    the wrong machine, so it scores below zero rather than merely last.

    CUDA is the one target that depends on what is plugged in: worthless
    without an NVIDIA card and a CUDA provider, and the best build in the repo
    with them.
    """
    name = folder.lower()
    # Split on every separator these paths use, so "npu/qnn-int4" is seen as a
    # QNN build rather than an NPU one.
    segments = set(re.split(r"[/\-_.]+", name))
    if any(target in segments for target in _FOREIGN_TARGETS):
        return -100
    if any(target in segments for target in _CUDA_TARGETS):
        usable, _reason = cuda_status()
        # Above every OpenVINO-EP build when the card is there: a 24 GB GPU
        # beats an NPU for anything that fits on it.
        return 200 if usable else -100

    score = 50
    for marker, delta in (
        ("openvino", 60),
        ("cpu", 30),
        ("int4", 20),
        ("acc-level-4", 5),
        ("npu", 10),
        ("mobile", 5),
        ("fp16", -10),
    ):
        if marker in name:
            score += delta
    return score
