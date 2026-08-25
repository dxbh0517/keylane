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
from app.npu.probe import (
    WARM_TIMEOUT,
    cache_dir,
    failure_detail,
    failure_kind,
    probe,
    static_objection,
)

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
        self._warming: threading.Thread | None = None
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

        # A model the user chose must never be able to kill the gateway, and
        # OpenVINO has more than one way to die without raising. Rule out the
        # known-fatal shapes on sight, then let a subprocess take the risk of
        # the first construction.
        objection = static_objection(model_path)
        if objection is not None:
            self.status = "unsupported model"
            self.degraded_reason = f"{model_path.name} was not loaded because {objection}"
            logger.warning(
                "%s: refusing %s — %s Using the heuristic path.",
                self.role,
                model_path.name,
                objection,
            )
            return

        blob_cache = cache_dir(self.config)

        failures: list[str] = []
        for device in candidates:
            ok, reason = probe(model_path, device, cache=blob_cache)
            if not ok:
                kind = failure_kind(reason)
                detail = failure_detail(reason)
                failures.append(f"{device}: {detail[:160]}")
                if kind == "timeout":
                    # Almost always a cold blob cache rather than a bad model:
                    # an uncached NPU compile of a 4B model runs to minutes.
                    # Warm it off the request path and reload when it lands,
                    # so startup is never held hostage to a compile.
                    logger.info(
                        "%s: %s has not been compiled for %s yet — warming the cache "
                        "in the background.",
                        self.role,
                        model_path.name,
                        device,
                    )
                    self.status = "compiling"
                    self._warm_in_background(model_path, device, blob_cache)
                    return
                logger.warning(
                    "%s could not load on %s — %s", self.role, device, failures[-1]
                )
                continue

            try:
                import openvino_genai as ov_genai

                pipeline = ov_genai.LLMPipeline(
                    str(model_path), device, CACHE_DIR=str(blob_cache)
                )
            except Exception as exc:  # noqa: BLE001
                # The probe just did this successfully, so reaching here means
                # something changed underneath us rather than a bad model.
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
                detail = failures[0] if failures else ""
                if preferred == "NPU":
                    detail = npu_failure_diagnosis() or detail
                self.degraded_reason = (
                    f"{preferred} could not compile this model, so it is running on "
                    f"{device}. {detail}"
                ).strip()
                logger.warning("%s: %s", self.role, self.degraded_reason)
            logger.info("%s model loaded on %s from %s", self.role, device, model_path)
            return

        self.status = "no device could compile this model"
        self.degraded_reason = "; ".join(failures)
        logger.error("%s failed on every device: %s", self.role, self.degraded_reason)

    def _warm_in_background(self, model_path: Path, device: str, blob_cache: Path) -> None:
        """Compile into the blob cache off the request path, then reload.

        The gateway answers on the heuristic path throughout; when the compile
        lands, ``_init`` runs again and finds a cache hit that costs seconds.
        """
        if self._warming is not None and self._warming.is_alive():
            return

        def run() -> None:
            ok, reason = probe(model_path, device, cache=blob_cache, timeout=WARM_TIMEOUT)
            if not ok:
                detail = failure_detail(reason)
                self.status = "no device could compile this model"
                self.degraded_reason = f"{device}: {detail}"
                logger.warning(
                    "%s: warming %s for %s failed — %s",
                    self.role,
                    model_path.name,
                    device,
                    detail,
                )
                return
            logger.info(
                "%s: %s is compiled and cached for %s — loading it now.",
                self.role,
                model_path.name,
                device,
            )
            self.reload()

        self._warming = threading.Thread(
            target=run, name=f"keylane-warm-{self.role}", daemon=True
        )
        self._warming.start()

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


# The NPU driver splits into two pieces: a Level Zero backend that enumerates
# the device, and a compiler that turns an OpenVINO graph into something the
# NPU can run. Distributions package them separately, and with only the first
# installed the NPU looks perfectly healthy right up until a compile is
# attempted — which then fails with whatever configuration key the plugin
# happened to try first. Reporting that key sends people chasing a setting
# when the real answer is a missing package.
_COMPILER_LIBRARIES = (
    "libnpu_driver_compiler.so",          # driver releases up to ~1.32
    "libopenvino_intel_npu_compiler.so",  # renamed in ~1.35
)
_COMPILER_SEARCH = (
    "/usr/lib64",
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib64",
    "/usr/local/lib",
)


def npu_compiler_present() -> bool:
    """True when the NPU driver's compiler library is installed."""
    from pathlib import Path as _Path

    return any(
        (_Path(d) / name).exists()
        for d in _COMPILER_SEARCH
        for name in _COMPILER_LIBRARIES
    )


def _npu_compiler_version() -> int | None:
    try:
        import openvino as ov

        return int(ov.Core().get_property("NPU", "NPU_COMPILER_VERSION"))
    except Exception:  # noqa: BLE001
        return None


def npu_failure_diagnosis() -> str:
    """Explain an NPU compile failure, when the cause is knowable.

    Two causes account for every NPU failure seen so far, and neither is
    visible in the error the plugin raises:

    * the compiler is missing entirely, so nothing compiles;
    * the compiler is present but built for a different OpenVINO generation
      than the runtime, so trivial graphs compile and real models do not.

    The compiler ships inside the NPU driver, not with the OpenVINO wheel, so
    upgrading one without the other is easy to do by accident.
    """
    version = _npu_compiler_version()
    if not npu_compiler_present() or version in (None, 0):
        return (
            "The NPU driver's compiler is not installed, so no model can be "
            "compiled for the NPU. On Fedora: sudo dnf install "
            "intel-npu-compiler — but check that its version matches the "
            "installed OpenVINO, because Fedora versions the two separately."
        )

    try:
        import openvino as ov

        runtime = ov.__version__.split("-")[0]
    except Exception:  # noqa: BLE001
        runtime = "unknown"
    return (
        f"The NPU compiler (interface {version >> 16}.{version & 0xFFFF}) "
        f"rejected the model while OpenVINO {runtime} is in use. The compiler "
        "ships with the NPU driver rather than the OpenVINO package, so the "
        "two have to come from the same release — see "
        "github.com/intel/linux-npu-driver/releases for the pairing."
    )
