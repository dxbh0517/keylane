"""Shared OpenVINO GenAI pipeline loading for the NPU control plane.

The router, the verifier and the assistant all want a small local model on the
NPU. Loading is identical for each, so it lives here once: resolve the path from
model settings, resolve the device (NPU → GPU → CPU per user preference), and
degrade to ``loaded = False`` instead of raising when anything is missing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.config import AppConfig, get_config

logger = logging.getLogger(__name__)

# OpenVINO IR / GenAI exports carry one of these next to the weights.
MODEL_MARKERS = ("openvino_model.xml", "openvino.xml", "config.json")


def model_status(path: Path) -> tuple[bool, str]:
    """Return ``(ready, reason)`` for a model directory.

    An OpenVINO IR is a ``.xml`` graph *plus* a ``.bin`` of weights. Accepting
    the graph alone lets a half-finished download look installed, then fail
    deep inside OpenVINO with "Empty weights data in bin file" — which tells
    the user nothing about what is actually wrong.
    """
    if not path.exists():
        return False, "no such directory"

    graphs = [
        xml for xml in sorted(path.glob("*.xml")) if "tokenizer" not in xml.stem.lower()
    ]
    if not graphs:
        return False, "no OpenVINO graph (*.xml) here"

    graph = graphs[0]
    weights = graph.with_suffix(".bin")
    if not weights.exists():
        partial = list(
            (path / ".cache" / "huggingface" / "download").glob("*.incomplete")
        )
        if partial:
            fetched = sum(f.stat().st_size for f in partial) / 1_000_000
            return False, (
                f"the download has not finished — {weights.name} is missing "
                f"({fetched:.0f} MB fetched). Re-running it resumes where it stopped."
            )
        return False, f"{weights.name} is missing (the graph is here, the weights are not)"

    try:
        if weights.stat().st_size < 4096:
            return False, f"{weights.name} is empty"
    except OSError as exc:
        return False, f"cannot read {weights.name}: {exc}"

    return True, "ready"


def model_ready(path: Path) -> bool:
    return model_status(path)[0]


class NpuPipeline:
    """One OpenVINO GenAI ``LLMPipeline``, loaded lazily and reloadable."""

    def __init__(self, role: str, config: AppConfig | None = None) -> None:
        self.role = role
        self.config = config or get_config()
        self._pipeline = None
        self._device: str | None = None
        self._model_path: Path | None = None
        self.status = "not loaded"
        self.degraded_reason: str | None = None
        self._lock = threading.Lock()
        self._init()

    # ------------------------------------------------------------------ load

    def _resolve_model_path(self) -> Path:
        from app.models_settings import load_models_settings

        settings = load_models_settings()
        if self.role == "verifier":
            configured = settings.verifier_model_path or settings.router_model_path
        else:
            configured = settings.router_model_path
        path = self.config.resolve_path(configured)
        if model_ready(path):
            return path
        # Fall back to the legacy workers.toml location.
        return self.config.npu_model_path

    def _candidate_devices(self) -> tuple[list[str], str | None]:
        """Devices to try, best first, plus the user's stated preference.

        A device being *present* is not the same as it being able to compile
        this model. NPU LLM support depends on the Level Zero driver matching
        the OpenVINO GenAI build, so a perfectly working NPU can still reject a
        model that runs fine on the GPU.
        """
        from app.models_settings import load_models_settings, resolve_openvino_device

        settings = load_models_settings()
        try:
            import openvino as ov

            available = list(ov.Core().available_devices)
        except Exception:  # noqa: BLE001
            return [], None
        if not available:
            return [], None

        def present(name: str | None) -> str | None:
            if not name:
                return None
            for device in available:
                if device == name or device.startswith(f"{name}."):
                    return device
            return None

        preferred = resolve_openvino_device(settings)
        order: list[str] = []
        for name in (preferred, settings.gpu_device, settings.fallback_device, "CPU"):
            resolved = present(name)
            if resolved and resolved not in order:
                order.append(resolved)
        return order, present(preferred) or preferred

    def _init(self) -> None:
        self.status = "not loaded"
        self.degraded_reason = None

        model_path = self._resolve_model_path()
        ready, reason = model_status(model_path)
        if not ready:
            self.status = reason
            logger.warning(
                "%s model at %s is not usable: %s — using the heuristic path.",
                self.role,
                model_path,
                reason,
            )
            return

        candidates, preferred = self._candidate_devices()
        if not candidates:
            self.status = "no OpenVINO device available"
            logger.warning("%s: %s", self.role, self.status)
            return

        failures: list[str] = []
        for device in candidates:
            try:
                import openvino_genai as ov_genai

                pipeline = ov_genai.LLMPipeline(str(model_path), device)
            except Exception as exc:  # noqa: BLE001
                lines = str(exc).strip().splitlines()
                failures.append(f"{device}: {(lines[-1] if lines else str(exc))[:160]}")
                logger.warning("%s could not load on %s — %s", self.role, device, failures[-1])
                continue

            self._pipeline = pipeline
            self._device = device
            self._model_path = model_path
            self.status = "ready"
            if preferred and device != preferred:
                # Loading on a slower device still beats no assistant at all,
                # but the user needs to know why their choice was not honoured.
                self.degraded_reason = (
                    f"{preferred} could not compile this model, so it is running on "
                    f"{device}. {failures[0] if failures else ''}"
                ).strip()
                logger.warning("%s: %s", self.role, self.degraded_reason)
            logger.info("%s model loaded on %s from %s", self.role, device, model_path)
            return

        self.status = "no device could compile this model"
        self.degraded_reason = "; ".join(failures)
        logger.error("%s failed on every device: %s", self.role, self.degraded_reason)

    def reload(self) -> None:
        with self._lock:
            self._pipeline = None
            self._device = None
            self._model_path = None
            self._init()

    # ------------------------------------------------------------------ use

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def model_path(self) -> str | None:
        return str(self._model_path) if self._model_path else None

    def generate_chat(
        self, system: str, user: str, *, max_new_tokens: int = 320
    ) -> str:
        """Generate with the model's own chat template applied.

        An instruct model handed a bare completion string behaves erratically —
        it was fine-tuned on ``<|im_start|>role`` turns and needs to see them.
        Falls back to a plain concatenation for models with no template.
        """
        if self._pipeline is None:
            raise RuntimeError(f"{self.role} pipeline is not loaded")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            tokenizer = self._pipeline.get_tokenizer()
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            prompt = f"{system}\n\n{user}\n"
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def generate(self, prompt: str, *, max_new_tokens: int = 320) -> str:
        """Run the model. Raises when no pipeline is loaded — callers fall back."""
        if self._pipeline is None:
            raise RuntimeError(f"{self.role} pipeline is not loaded")
        # openvino_genai pipelines are not re-entrant; serialise callers.
        with self._lock:
            raw = self._pipeline.generate(prompt, max_new_tokens=max_new_tokens)
        if hasattr(raw, "texts"):
            texts = raw.texts or []
            return texts[0] if texts else str(raw)
        return str(raw)


_pipelines: dict[str, NpuPipeline] = {}
_pipelines_lock = threading.Lock()


def get_pipeline(role: str, config: AppConfig | None = None) -> NpuPipeline:
    with _pipelines_lock:
        pipeline = _pipelines.get(role)
        if pipeline is None:
            pipeline = NpuPipeline(role, config)
            _pipelines[role] = pipeline
        return pipeline


def reload_pipelines(config: AppConfig | None = None) -> dict[str, bool]:
    """Reload every live pipeline; returns ``{role: loaded}``."""
    with _pipelines_lock:
        roles = list(_pipelines)
    status: dict[str, bool] = {}
    for role in roles:
        pipeline = _pipelines[role]
        pipeline.reload()
        status[role] = pipeline.loaded
    return status
