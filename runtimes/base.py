"""What an inference runtime has to provide.

Keylane can run a local model through more than one stack. OpenVINO GenAI
compiles an OpenVINO IR export and hands it to the NPU. ONNX Runtime GenAI runs
an ONNX export, reaching the same NPU through the OpenVINO execution provider.
They agree on almost nothing else: what a model looks like on disk, how you
tell a finished download from a torn one, what "compile" costs, how tokens come
back out of it.

So the catalog talks to this interface and never to either stack directly. A
model entry names its runtime, the runtime knows how to recognise, validate,
compile and drive its own kind of export, and adding a third one is a new module
here rather than a new branch in every function that touches a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from npu.kind import PipelineKind


@dataclass(frozen=True)
class RuntimeInfo:
    """What Settings needs to describe a runtime before anything is loaded."""

    id: str
    name: str
    summary: str
    # Shown when the Python package backing this runtime is not importable.
    install_hint: str
    devices: tuple[str, ...]
    default_device: str


@dataclass(frozen=True)
class RepoVariant:
    """One loadable model found inside a Hugging Face repo.

    ONNX repos routinely ship four or five builds of the same model in
    subfolders — cpu-int4, cuda-fp16, directml, qnn — and only some of them can
    run here. A variant is one such build: the subfolder to fetch, which runtime
    claims it, and a score saying how well it suits this machine so the importer
    can pick without asking.
    """

    runtime: str
    subfolder: str
    label: str
    score: int


@runtime_checkable
class LoadedPipeline(Protocol):
    """A model that is compiled, resident, and ready to answer."""

    kind: PipelineKind

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Run one completion. Returns the raw text; the caller sanitises it."""

    def close(self) -> None:
        """Release the device. Called before another model is loaded."""


@runtime_checkable
class RuntimeBackend(Protocol):
    """One inference stack, from "is this mine?" through to a loaded pipeline."""

    info: RuntimeInfo

    # ── availability ─────────────────────────────────────────────────────

    def installed(self) -> tuple[bool, str]:
        """(importable, detail). A local check — never a download or a compile."""

    def cache_dir(self) -> Path:
        """Where this runtime keeps compiled blobs. Created if missing."""

    # ── recognising a model on disk ──────────────────────────────────────

    def detect(self, model_dir: Path) -> bool:
        """Does this directory hold an export this runtime can load?"""

    def model_kind(self, model_dir: Path) -> PipelineKind:
        """``vlm`` for vision-language exports, else ``llm``."""

    def missing_weights(self, model_dir: Path) -> list[str]:
        """Names of weight files the export needs and does not have."""

    def purge_incomplete(self, model_dir: Path) -> None:
        """Remove a partial export so the download can start clean."""

    def static_objection(self, model_dir: Path) -> str | None:
        """A reason not to bother compiling, found without compiling."""

    # ── loading ──────────────────────────────────────────────────────────

    def warm_timeout_for(self, kind: PipelineKind) -> float:
        """How long a first compile of this kind may take before we give up."""

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
        """Compile in a subprocess first, so a crash cannot take the daemon."""

    def load(
        self,
        model_dir: Path,
        device: str,
        *,
        cache: Path | None,
        kind: PipelineKind,
    ) -> LoadedPipeline:
        """Compile and return a resident pipeline."""

    def prompt_budget_chars(self, device: str, kind: PipelineKind, model_dir: Path) -> int:
        """How many characters of prompt this pipeline can actually take."""

    # ── Hugging Face repos ───────────────────────────────────────────────

    def repo_variants(self, filenames: Iterable[str]) -> list[RepoVariant]:
        """Loadable builds this runtime can find in a repo's file listing."""

    def allow_patterns(self, subfolder: str) -> list[str] | None:
        """Which repo files to fetch, or None for the whole repo."""


def status_payload(backend: RuntimeBackend) -> dict[str, Any]:
    """The shape Settings and /health read a runtime as.

    ``devices`` is what this machine can actually be pointed at, not what the
    stack supports in the abstract — see ``runtimes/devices.py``.
    ``all_devices`` keeps the unusable ones so Settings can say why rather than
    quietly hiding hardware the user knows they have.
    """
    from runtimes.devices import device_options  # noqa: PLC0415

    installed, detail = backend.installed()
    options = device_options(backend.info.devices)
    usable = [o.id for o in options if o.usable]
    default = backend.info.default_device
    if usable and default not in usable:
        default = usable[0]
    return {
        "id": backend.info.id,
        "name": backend.info.name,
        "summary": backend.info.summary,
        "installed": installed,
        "detail": detail,
        "install_hint": backend.info.install_hint,
        "devices": usable or list(backend.info.devices),
        "all_devices": [
            {"id": o.id, "label": o.label, "usable": o.usable, "reason": o.reason}
            for o in options
        ],
        "default_device": default,
    }
