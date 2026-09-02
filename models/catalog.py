"""Model catalog — curated models per runtime, with Hugging Face auto-download.

A catalog entry names the runtime it belongs to, because the runtime is a
property of the export rather than a preference: an ``*-int4-ov`` repo is an
OpenVINO IR model and nothing else can load it, and a repo with a
``genai_config.json`` is an ONNX Runtime GenAI model in the same way. Everything
downstream — validating a download, compiling, budgeting a prompt, streaming
tokens — is asked of the entry's runtime rather than decided here.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from daemon.config import load_toml
from daemon.paths import MODELS_DIR
from npu.kind import PipelineKind
from npu.thinking import sanitize_response
from runtimes import DEFAULT_RUNTIME, RuntimeBackend, backend_for, normalise_runtime_id

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    id: str
    name: str
    hf_repo: str
    params_b: int
    description: str = ""
    device: str | None = None
    runtime: str = DEFAULT_RUNTIME
    # ONNX repos ship several builds of the same model side by side, one per
    # execution target. The subfolder says which of them this entry is.
    subfolder: str = ""
    # "curated" comes from config/models.toml; "imported" the user added.
    source: str = "curated"
    size_bytes: int = 0

    @property
    def backend(self) -> RuntimeBackend:
        return backend_for(self.runtime)

    @property
    def local_path(self) -> Path:
        """Where the repo is downloaded to."""
        return MODELS_DIR / self.id

    @property
    def model_dir(self) -> Path:
        """Where the loadable export sits — the repo root unless it is nested."""
        return self.local_path / self.subfolder if self.subfolder else self.local_path

    def resolve_device(self, default: str = "") -> str:
        """Per-model override, then the runtime's setting, then its default."""
        if self.device:
            return self.device

        from daemon.config import get_section

        devices = get_section("models").get("devices", {})
        if isinstance(devices, dict):
            chosen = str(devices.get(self.runtime, "") or "").strip()
            if chosen:
                return chosen

        supported = {d.upper() for d in self.backend.info.devices}
        if default and default.strip().upper() in supported:
            return default.strip().upper()
        return self.backend.info.default_device

    def is_downloaded(self) -> bool:
        return not self.backend.missing_weights(self.model_dir)

    def missing(self) -> list[str]:
        return self.backend.missing_weights(self.model_dir)


def _entry_from_dict(raw: dict[str, Any], *, source: str) -> ModelEntry | None:
    try:
        model_id = str(raw["id"]).strip()
        hf_repo = str(raw["hf_repo"]).strip()
    except (KeyError, TypeError):
        logger.warning("skipping model entry with no id or hf_repo: %r", raw)
        return None
    if not model_id or not hf_repo:
        return None
    return ModelEntry(
        id=model_id,
        name=str(raw.get("name") or model_id),
        hf_repo=hf_repo,
        params_b=int(raw.get("params_b") or 0),
        description=str(raw.get("description", "")),
        device=str(raw["device"]) if raw.get("device") else None,
        runtime=normalise_runtime_id(raw.get("runtime")),
        subfolder=str(raw.get("subfolder", "") or "").strip("/"),
        source=source,
        size_bytes=int(raw.get("size_bytes") or 0),
    )


def load_catalog() -> tuple[str, str, list[ModelEntry]]:
    """(default model id, default device, entries) — curated then imported."""
    raw = load_toml("models.toml")
    default = str(raw.get("default", "qwen2.5-7b-instruct"))
    device = str(raw.get("device", "NPU"))

    entries: list[ModelEntry] = []
    seen: set[str] = set()
    for spec in raw.get("models", []):
        entry = _entry_from_dict(spec, source="curated")
        if entry and entry.id not in seen:
            seen.add(entry.id)
            entries.append(entry)

    for spec in _imported_specs():
        entry = _entry_from_dict(spec, source="imported")
        if entry and entry.id not in seen:
            seen.add(entry.id)
            entries.append(entry)

    return default, device, entries


def _imported_specs() -> list[dict[str, Any]]:
    from daemon.config import get_section

    raw = get_section("models").get("imported", [])
    if not isinstance(raw, list):
        return []
    return [spec for spec in raw if isinstance(spec, dict)]


def get_model(model_id: str) -> ModelEntry | None:
    _, _, entries = load_catalog()
    for entry in entries:
        if entry.id == model_id:
            return entry
    return None


def catalog_default_model_id() -> str:
    default, _, _ = load_catalog()
    return default


def default_model_id() -> str:
    """User override from settings.json, else models.toml default."""
    from daemon.config import get_section

    override = get_section("models").get("default_model_id")
    if override:
        mid = str(override).strip()
        if mid and get_model(mid):
            return mid
    return catalog_default_model_id()


def active_runtime_id() -> str:
    """The runtime Settings is currently browsing models for."""
    from daemon.config import get_section

    return normalise_runtime_id(get_section("models").get("runtime"))


def _repo_total_bytes(hf_repo: str, subfolder: str = "") -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(hf_repo, files_metadata=True)
    except Exception:  # noqa: BLE001
        logger.warning("could not fetch file sizes for %s", hf_repo)
        return 0

    prefix = f"{subfolder}/" if subfolder else ""
    return sum(
        getattr(sibling, "size", 0) or 0
        for sibling in info.siblings
        if not prefix or str(sibling.rfilename).startswith(prefix)
    )


def _downloaded_bytes(model_dir: Path) -> int:
    """Bytes on disk: finished files plus any in-progress .incomplete chunks."""
    if not model_dir.is_dir():
        return 0

    total = 0
    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir)
        if rel.parts[0] == ".cache":
            if path.suffix == ".incomplete":
                total += path.stat().st_size
            continue
        total += path.stat().st_size
    return total


def _active_download_file(model_dir: Path, missing: list[str] | None = None) -> str:
    """Best-effort name of the file Hugging Face is fetching right now.

    *model_dir* is the download root, which for a nested export is the repo
    rather than the model — Hugging Face mirrors the repo's own layout under
    ``.cache``, so the lock files are nested to match.
    """
    dl_dir = model_dir / ".cache/huggingface/download"
    if missing is None:
        from npu.weights import missing_weights

        missing = missing_weights(model_dir)

    if dl_dir.is_dir():
        locks = sorted(dl_dir.rglob("*.lock"), key=lambda p: p.stat().st_mtime, reverse=True)
        locked = {
            lock.relative_to(dl_dir).as_posix().removesuffix(".lock").rsplit("/", 1)[-1]
            for lock in locks
        }
        for name in missing:
            if name in locked:
                return name
        for lock in locks:
            rel = lock.relative_to(dl_dir).as_posix().removesuffix(".lock")
            if rel.endswith(".metadata"):
                continue
            if not (model_dir / rel).is_file():
                return rel.rsplit("/", 1)[-1]
        if any(dl_dir.rglob("*.incomplete")):
            return missing[0] if missing else "finishing download…"

    if missing:
        return missing[0]
    return "preparing…"


class _ByteDownloadMonitor:
    """Poll disk while snapshot_download runs — HF only reports per-file counts."""

    def __init__(
        self,
        entry: ModelEntry,
        total_bytes: int,
        notify: Callable[..., None],
    ) -> None:
        self._entry = entry
        self._total_bytes = total_bytes
        self._notify = notify
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="download-bytes")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(0.4):
            missing = self._entry.backend.missing_weights(self._entry.model_dir)
            name = _active_download_file(self._entry.local_path, missing)
            done = _downloaded_bytes(self._entry.local_path)
            if self._total_bytes > 0:
                pct = min(99, int(100 * done / self._total_bytes))
                self._notify(f"{name} — {pct}%", pct, file=name)
            else:
                self._notify(name, None, file=name)


def download_model(
    entry: ModelEntry,
    *,
    progress: Callable[..., None] | None = None,
    force: bool = False,
) -> Path:
    entry.local_path.mkdir(parents=True, exist_ok=True)
    backend = entry.backend
    missing = backend.missing_weights(entry.model_dir)

    def _notify(
        message: str,
        percent: int | None = None,
        file: str | None = None,
    ) -> None:
        if not progress:
            return
        try:
            progress(message, percent, file)
        except TypeError:
            try:
                progress(message, percent)
            except TypeError:
                progress(message)

    if not missing and not force:
        _notify("already downloaded", 100, file="")
        return entry.model_dir

    if missing and not force:
        lead = missing[0]
        _notify(f"resuming — {lead}", 0, file=lead)
        logger.warning(
            "model %s incomplete — missing %s; resuming download",
            entry.id,
            ", ".join(missing),
        )
    elif force:
        _notify("re-downloading model weights…", 0, file="")
        backend.purge_incomplete(entry.model_dir)
    else:
        _notify(f"downloading {entry.hf_repo}…", 0, file="")

    from huggingface_hub import snapshot_download

    total_bytes = _repo_total_bytes(entry.hf_repo, entry.subfolder)
    monitor = _ByteDownloadMonitor(entry, total_bytes, _notify)
    monitor.start()
    try:
        snapshot_download(
            entry.hf_repo,
            local_dir=str(entry.local_path),
            allow_patterns=backend.allow_patterns(entry.subfolder),
        )
    finally:
        monitor.stop()

    still_missing = backend.missing_weights(entry.model_dir)
    if still_missing:
        raise RuntimeError(
            f"download finished but weights still missing: {', '.join(still_missing)}"
        )

    _notify("download complete", 100, file="")
    return entry.model_dir


class LocalModelRuntime:
    """The always-on local model, whichever runtime is holding it.

    One model is resident at a time. Switching models unloads the old pipeline
    first — on the NPU there is not room for two, and the runtimes do not share
    the device.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pipe: Any = None
        self._model_id: str | None = None
        self._runtime_id: str = DEFAULT_RUNTIME
        self._pipeline_kind: PipelineKind = "llm"
        self._device: str = ""
        self._status = "idle"
        self._error = ""
        self._progress = ""
        self._loading = False
        self._load_thread: threading.Thread | None = None
        self._downloads: dict[str, dict[str, Any]] = {}
        self._download_lock = threading.RLock()

    def _download_info(self, model_id: str) -> dict[str, Any]:
        with self._download_lock:
            return dict(self._downloads.get(model_id, {}))

    def download_status(self) -> dict[str, dict[str, Any]]:
        with self._download_lock:
            return {mid: dict(info) for mid, info in self._downloads.items()}

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "runtime": self._runtime_id,
            "device": self._device,
            "pipeline": self._pipeline_kind,
            "state": self._status,
            "error": self._error,
            "ready": self._pipe is not None and self._status == "ready",
            "loading": self._loading,
            "progress": self._progress,
            "downloads": self.download_status(),
        }

    def start_download(self, model_id: str) -> dict[str, Any]:
        """Download model weights in the background without loading them."""
        entry = get_model(model_id)
        if not entry:
            raise ValueError(f"unknown model: {model_id}")

        if entry.is_downloaded():
            return {
                "model_id": model_id,
                "downloading": False,
                "downloaded": True,
                "progress": "downloaded",
            }

        with self._download_lock:
            current = self._downloads.get(model_id, {})
            if current.get("downloading"):
                return {"model_id": model_id, **current}

        def _worker() -> None:
            with self._download_lock:
                self._downloads[model_id] = {
                    "downloading": True,
                    "progress": "starting download…",
                    "percent": 0,
                    "file": "",
                    "error": "",
                    "downloaded": False,
                }

            def _prog(
                message: str,
                percent: int | None = None,
                file: str | None = None,
            ) -> None:
                with self._download_lock:
                    if model_id not in self._downloads:
                        return
                    self._downloads[model_id]["progress"] = message
                    if percent is not None:
                        self._downloads[model_id]["percent"] = percent
                    if file is not None:
                        self._downloads[model_id]["file"] = file

            try:
                download_model(entry, progress=_prog)
                with self._download_lock:
                    self._downloads[model_id] = {
                        "downloading": False,
                        "progress": "download complete",
                        "percent": 100,
                        "error": "",
                        "downloaded": True,
                    }
            except Exception as exc:  # noqa: BLE001
                logger.exception("model download failed")
                with self._download_lock:
                    self._downloads[model_id] = {
                        "downloading": False,
                        "progress": "",
                        "error": str(exc),
                        "downloaded": False,
                    }

        threading.Thread(target=_worker, daemon=True, name=f"download-{model_id}").start()
        with self._download_lock:
            info = dict(self._downloads.get(model_id, {}))
        info.setdefault("model_id", model_id)
        info.setdefault("downloading", True)
        return info

    def _wait_for_download(self, model_id: str, timeout: float = 3600) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self._download_info(model_id)
            if info.get("downloaded"):
                return
            if info.get("error"):
                raise RuntimeError(info["error"])
            if not info.get("downloading"):
                return
            time.sleep(0.5)
        raise TimeoutError(f"timed out waiting for download of {model_id}")

    def _set_progress(self, message: str, progress: Callable[[str], None] | None = None) -> None:
        self._progress = message
        if progress:
            progress(message)

    def _unload_pipeline(self) -> None:
        with self._lock:
            old = self._pipe
            self._pipe = None
            self._model_id = None
            self._pipeline_kind = "llm"
            self._device = ""
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                logger.debug("pipeline close raised while unloading", exc_info=True)
            del old
            gc.collect()

    def start_load(self, model_id: str) -> dict[str, Any]:
        """Begin loading a model on a background thread; returns immediately."""
        entry = get_model(model_id)
        if not entry:
            raise ValueError(f"unknown model: {model_id}")

        with self._lock:
            if self._loading:
                return self.status
            if self._model_id == model_id and self._pipe is not None:
                return self.status
            self._loading = True
            self._status = "downloading"
            self._error = ""
            self._progress = "starting…"

        def _worker() -> None:
            try:
                self._load_impl(model_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("model load failed")
                self._status = "error"
                self._error = str(exc)
            finally:
                self._loading = False
                self._progress = ""

        self._load_thread = threading.Thread(target=_worker, daemon=True, name=f"load-{model_id}")
        self._load_thread.start()
        return self.status

    def load(
        self,
        model_id: str,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Blocking load — used at daemon warm-up."""
        self._loading = True
        try:
            self._load_impl(model_id, progress=progress)
        finally:
            self._loading = False
            self._progress = ""

    def _load_impl(
        self,
        model_id: str,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        entry = get_model(model_id)
        if not entry:
            raise ValueError(f"unknown model: {model_id}")

        with self._lock:
            if self._model_id == model_id and self._pipe is not None:
                return

        backend = entry.backend
        installed, detail = backend.installed()
        if not installed:
            raise RuntimeError(
                f"{entry.name} needs the {backend.info.name} runtime, which is not "
                f"installed ({detail}). Install it with: {backend.info.install_hint}"
            )

        self._status = "downloading"
        self._error = ""
        self._set_progress("preparing download…", progress)
        dl = self._download_info(model_id)
        if dl.get("downloading"):
            self._set_progress("waiting for background download…", progress)
            self._wait_for_download(model_id)
        try:
            download_model(entry, progress=lambda m: self._set_progress(m, progress))
        except Exception as exc:  # noqa: BLE001
            if "missing" in str(exc).lower() or "bin" in str(exc).lower():
                logger.warning("retrying download after incomplete weights: %s", exc)
                download_model(entry, progress=lambda m: self._set_progress(m, progress), force=True)
            else:
                raise

        path = entry.model_dir
        weights_left = backend.missing_weights(path)
        if weights_left:
            self._status = "error"
            self._error = f"missing weights: {', '.join(weights_left)}"
            raise RuntimeError(self._error)

        objection = backend.static_objection(path)
        if objection:
            self._status = "error"
            self._error = objection
            raise RuntimeError(objection)

        # Free the device before loading a different model.
        self._set_progress("unloading previous model…", progress)
        self._unload_pipeline()

        _, catalog_device, _ = load_catalog()
        device = entry.resolve_device(catalog_device)
        cache = backend.cache_dir()
        pipeline_kind = backend.model_kind(path)

        self._status = "probing"
        label = "VLM" if pipeline_kind == "vlm" else "LLM"
        compile_hint = (
            f"first {label} compile on {device} may take several minutes"
            if pipeline_kind == "vlm" and device == "NPU"
            else f"compiling on {device} via {backend.info.name}"
        )
        self._set_progress(f"compiling {label} for {device} ({compile_hint})…", progress)

        def _probe_tick(elapsed: float) -> None:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self._set_progress(
                f"compiling {label} on {device}… {mins}m {secs:02d}s ({compile_hint})",
                progress,
            )

        warm_timeout = backend.warm_timeout_for(pipeline_kind)
        ok, reason = backend.probe(
            path,
            device,
            cache=cache,
            timeout=warm_timeout,
            kind=pipeline_kind,
            on_tick=_probe_tick,
        )
        # A compile can fail because a weight file is absent or torn rather
        # than because the model is wrong for the device. Ask the runtime
        # rather than reading the failure text, which differs per stack.
        if not ok and (backend.missing_weights(path) or "bin" in reason.lower()):
            logger.warning("probe failed with missing weights — re-downloading: %s", reason)
            download_model(entry, progress=lambda m: self._set_progress(m, progress), force=True)
            ok, reason = backend.probe(
                path,
                device,
                cache=cache,
                timeout=warm_timeout,
                kind=pipeline_kind,
                on_tick=_probe_tick,
            )
        if not ok:
            self._status = "error"
            self._error = reason
            raise RuntimeError(reason)

        self._status = "loading"
        self._set_progress(f"loading {label} into memory…", progress)

        pipe = backend.load(path, device, cache=cache, kind=pipeline_kind)
        with self._lock:
            self._pipe = pipe
            self._model_id = model_id
            self._runtime_id = entry.runtime
            self._device = device
            self._pipeline_kind = pipeline_kind
            self._status = "ready"
            self._error = ""
        self._set_progress("model ready", progress)

    def prompt_budget_chars(self) -> int:
        """Context budget for chat prompts, from the loaded pipeline's own limit.

        This used to grant a VLM 10000 characters on the grounds that vision
        models have roomier contexts. On the NPU the opposite is true: the
        pipeline compiles a fixed prompt length in, and exceeding it throws
        rather than truncates. The budget now comes from the runtime that built
        the pipeline, so the two cannot disagree.
        """
        entry = get_model(self._model_id or "")
        if entry is None:
            return backend_for(self._runtime_id).prompt_budget_chars(
                self._device or "NPU", self._pipeline_kind, Path()
            )
        device = self._device or entry.resolve_device(load_catalog()[1])
        return entry.backend.prompt_budget_chars(device, self._pipeline_kind, entry.model_dir)

    # Kept because the LLM adapter reached for it before it was public.
    _prompt_budget_chars = prompt_budget_chars

    @staticmethod
    def _format_message(msg: dict[str, str]) -> str:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            return f"System: {content}"
        if role == "assistant":
            return f"Assistant: {content}"
        return f"User: {content}"

    def _fit_chat_prompt(self, messages: list[dict[str, str]], max_chars: int) -> str:
        """Build a prompt that keeps the system block and the most recent turns."""
        system_blocks: list[str] = []
        history: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_blocks.append(str(msg.get("content", "")))
            else:
                history.append(msg)

        system_text = "\n\n".join(system_blocks).strip()

        # The system block used to be included whole, whatever the budget, so a
        # system prompt larger than the pipeline's limit could never be trimmed
        # down to fit — it just threw. Clip it, and leave at least a third of
        # the budget for the conversation, since a prompt with no conversation
        # in it cannot answer anything.
        system_cap = int(max_chars * 2 / 3)
        if len(system_text) > system_cap:
            logger.warning(
                "system prompt is %d chars but the pipeline budget is %d; clipping",
                len(system_text),
                system_cap,
            )
            system_text = system_text[:system_cap].rstrip() + "\n…[prompt clipped]"

        formatted = [self._format_message(msg) for msg in history]
        tail: list[str] = []
        overhead = len("\n\nAssistant:")
        if system_text:
            overhead += len(f"System: {system_text}") + 2

        used = overhead
        for block in reversed(formatted):
            chunk_len = len(block) + 2
            if tail and used + chunk_len > max_chars:
                break
            tail.insert(0, block)
            used += chunk_len

        parts: list[str] = []
        if system_text:
            parts.append(f"System: {system_text}")
        parts.extend(tail)
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system: str | None = None,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        with self._lock:
            if self._pipe is None:
                raise RuntimeError("no model loaded")
            pipe = self._pipe
            kind = self._pipeline_kind

        full = f"{system}\n\n{prompt}" if system else prompt
        raw = pipe.generate(
            full,
            max_new_tokens=max_new_tokens,
            images=images if kind == "vlm" else None,
            on_token=on_token,
        )
        return sanitize_response(raw)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Format messages into a single prompt for the active pipeline."""
        budget = self.prompt_budget_chars()
        prompt = self._fit_chat_prompt(messages, budget)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            images=images,
            on_token=on_token,
        )


# The class was named for the device before it could target more than one
# runtime. Both names refer to the same object.
NpuRuntime = LocalModelRuntime

_runtime = LocalModelRuntime()


def get_runtime() -> LocalModelRuntime:
    return _runtime
