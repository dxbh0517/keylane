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
from npu.limits import prompt_budget_chars
from npu.pipeline_config import pipeline_init_kwargs
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
        import openvino_genai as ov_genai  # noqa: PLC0415

        kwargs = pipeline_init_kwargs(device, cache, kind)
        pipeline_cls = ov_genai.VLMPipeline if kind == "vlm" else ov_genai.LLMPipeline
        try:
            pipe = pipeline_cls(str(model_dir), device, **kwargs)
        except TypeError:
            pipe = pipeline_cls(str(model_dir), device)
        return OpenVinoPipeline(pipe, kind)

    def prompt_budget_chars(self, device: str, kind: PipelineKind, model_dir: Path) -> int:
        return prompt_budget_chars(device, kind)

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
