"""OpenVINO GenAI — the runtime Keylane started with.

The mechanics live in ``npu/`` and predate the runtime interface; this module is
the adapter that presents them through it, so nothing had to move to make room
for a second stack.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

from npu import probe as ov_probe
from npu.images import bytes_to_ov_tensors
from npu.kind import PipelineKind, model_kind
from npu.limits import prompt_budget_chars, prompt_budget_tokens
from npu.pipeline_config import create_pipeline
from npu.thinking import OutputStreamFilter
from npu.weights import missing_weights, purge_incomplete
from runtimes.base import RepoVariant, RuntimeInfo

logger = logging.getLogger(__name__)

INFO = RuntimeInfo(
    id="openvino",
    name="OpenVINO GenAI",
    summary=(
        "Intel's own stack. Loads OpenVINO IR exports (*-int4-ov) and compiles "
        "them for the NPU. Broadest NPU support and the only one here that runs "
        "vision models."
    ),
    install_hint="pip install openvino openvino-genai openvino-tokenizers",
    devices=("NPU", "GPU", "CPU"),
    default_device="NPU",
)


class OpenVinoPipeline:
    """A compiled ``LLMPipeline`` or ``VLMPipeline``, wrapped for the seam."""

    def __init__(self, pipe: Any, kind: PipelineKind) -> None:
        self._pipe = pipe
        self.kind: PipelineKind = kind
        # A VLM pipeline accumulates conversation state across generate()
        # calls, and a call that throws leaves that state behind — see
        # _clear_carried_state.
        self._poisoned = False

    # ── recovering from a failed generate ────────────────────────────────

    def _clear_carried_state(self) -> None:
        """Forget any conversation the pipeline thinks it is in the middle of.

        ``VLMPipeline`` keeps a tokenized history and a KV cache between
        ``generate()`` calls, and asserts that each new prompt is at least as
        long as the history it already holds. It clears that itself at the end
        of a successful call — but a call that *throws* never gets there, so
        the history stays and every later turn dies on

            Prompt ids size is less than tokenized history size

        which is the previous prompt's tokens still sitting in the pipeline.
        One failed generate therefore broke every turn after it until the
        daemon was restarted. ``finish_chat()`` is what drops that state.

        Keylane owns the transcript and sends it whole on every turn, so there
        is never anything here worth keeping.
        """
        pipe = self._pipe
        if pipe is None:
            return
        try:
            pipe.finish_chat()
        except Exception:  # noqa: BLE001
            # An older GenAI, or a pipeline with no chat state to clear.
            logger.debug("could not clear carried pipeline state", exc_info=True)

    # ── the model's own idea of a conversation ───────────────────────────

    def _tokenizer(self) -> Any | None:
        pipe = self._pipe
        if pipe is None:
            return None
        try:
            return pipe.get_tokenizer()
        except Exception:  # noqa: BLE001
            logger.debug("pipeline exposes no tokenizer", exc_info=True)
            return None

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str | None:
        """Render a conversation the way this model was fine-tuned to read it.

        Every instruct model has a template — ``<|im_start|>`` for Qwen,
        ``<|user|>`` for Phi, ``[INST]`` for Mistral — and it ships in the
        export. Concatenating ``"System: … User: …"`` instead is the most
        common reason a good model rambles, ignores a required output format,
        or will not stop.

        Returns None when the export carries no template, so the caller can
        fall back rather than fail.
        """
        tokenizer = self._tokenizer()
        if tokenizer is None:
            return None
        try:
            if not tokenizer.chat_template:
                return None
            return str(tokenizer.apply_chat_template(list(messages), True))
        except Exception:  # noqa: BLE001
            logger.debug("apply_chat_template failed; falling back", exc_info=True)
            return None

    def count_tokens(self, text: str) -> int | None:
        """The real token count, from the tokenizer that will do the encoding."""
        tokenizer = self._tokenizer()
        if tokenizer is None:
            return None
        try:
            return int(tokenizer.encode(text).input_ids.get_shape()[-1])
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
        streamer = None
        output_filter: OutputStreamFilter | None = None
        if on_token is not None:
            output_filter = OutputStreamFilter()

            def streamer(subword: str) -> bool:  # noqa: F811
                visible = output_filter.feed(subword) if output_filter else subword
                if visible:
                    on_token(visible)
                return False

        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if streamer is not None:
            kwargs["streamer"] = streamer

        # A turn after a failed one starts from a pipeline still holding the
        # failed turn's history. Clear it before asking for anything, rather
        # than letting this turn die of the last one's leftovers.
        if self._poisoned:
            logger.info("clearing pipeline state left by a failed generate")
            self._clear_carried_state()
            self._poisoned = False

        try:
            if self.kind == "vlm" and images:
                tensors = bytes_to_ov_tensors(images)
                if len(tensors) == 1:
                    result = self._pipe.generate(prompt, image=tensors[0], **kwargs)
                else:
                    result = self._pipe.generate(prompt, images=tensors, **kwargs)
            else:
                try:
                    result = self._pipe.generate(prompt, **kwargs)
                except TypeError:
                    result = self._pipe.generate(prompt, max_new_tokens=max_new_tokens)
        except Exception:
            # Only VLM pipelines carry state across calls, so only they can be
            # left in this condition — but the flag is cheap and a runtime that
            # starts doing the same thing should not need a second fix.
            self._poisoned = True
            raise

        if output_filter is not None and on_token is not None:
            remainder = output_filter.flush()
            if remainder:
                on_token(remainder)

        texts = getattr(result, "texts", None)
        return str(texts[0]) if texts else str(result)

    def close(self) -> None:
        pipe = self._pipe
        self._pipe = None
        if pipe is not None:
            del pipe
            gc.collect()


class OpenVinoBackend:
    """The OpenVINO GenAI runtime."""

    info = INFO

    def installed(self) -> tuple[bool, str]:
        try:
            import openvino as ov  # noqa: PLC0415

            import openvino_genai  # noqa: F401,PLC0415
        except ImportError as exc:
            return False, str(exc)
        return True, f"OpenVINO {getattr(ov, '__version__', '?')}"

    def cache_dir(self) -> Path:
        return ov_probe.cache_dir()

    # ── recognising a model ──────────────────────────────────────────────

    def detect(self, model_dir: Path) -> bool:
        return model_dir.is_dir() and any(model_dir.glob("openvino*.xml"))

    def model_kind(self, model_dir: Path) -> PipelineKind:
        return model_kind(model_dir)

    def missing_weights(self, model_dir: Path) -> list[str]:
        return missing_weights(model_dir)

    def purge_incomplete(self, model_dir: Path) -> None:
        purge_incomplete(model_dir)

    def static_objection(self, model_dir: Path) -> str | None:
        return ov_probe.static_objection(model_dir)

    # ── loading ──────────────────────────────────────────────────────────

    def warm_timeout_for(self, kind: PipelineKind) -> float:
        return ov_probe.warm_timeout_for(kind)

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
        return ov_probe.probe(
            model_dir,
            device,
            cache=cache,
            timeout=timeout,
            kind=kind,
            on_tick=on_tick,
        )

    def load(
        self,
        model_dir: Path,
        device: str,
        *,
        cache: Path | None,
        kind: PipelineKind,
    ) -> OpenVinoPipeline:
        # Same constructor the probe just ran, so this finds the probe's blob
        # in the cache instead of compiling a second one.
        return OpenVinoPipeline(create_pipeline(model_dir, device, cache, kind), kind)

    def prompt_budget_chars(self, device: str, kind: PipelineKind, model_dir: Path) -> int:
        return prompt_budget_chars(device, kind)

    def prompt_budget_tokens(self, device: str, kind: PipelineKind, model_dir: Path) -> int:
        """The limit in the unit the NPU pipeline actually compiles in."""
        return prompt_budget_tokens(device, kind)

    # ── Hugging Face repos ───────────────────────────────────────────────

    def repo_variants(self, filenames: Iterable[str]) -> list[RepoVariant]:
        """Directories holding an IR export. Nearly always the repo root."""
        found: dict[str, str] = {}
        for name in filenames:
            base = name.rsplit("/", 1)[-1]
            if base not in {"openvino_model.xml", "openvino_language_model.xml"}:
                continue
            folder = name[: -len(base)].rstrip("/")
            found.setdefault(folder, base)

        variants: list[RepoVariant] = []
        for folder, base in sorted(found.items()):
            label = folder or "repository root"
            # A root-level export is the ordinary shape; anything nested is
            # usually a secondary build, so it sorts below.
            score = 100 if not folder else 60
            if base == "openvino_language_model.xml":
                label = f"{label} (vision)"
            variants.append(
                RepoVariant(runtime=INFO.id, subfolder=folder, label=label, score=score)
            )
        return variants

    def allow_patterns(self, subfolder: str) -> list[str] | None:
        if not subfolder:
            return None
        return [f"{subfolder}/*"]
