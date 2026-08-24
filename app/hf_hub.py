"""Hugging Face Hub search + download into Keylane model folders.

Compatibility scoring uses runtime hardware detection (NPU / VRAM / RAM),
so the same code works on any machine.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from app.config import ROOT
from app.hardware import HardwareProfile, detect_hardware

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/models"

Target = Literal["router", "chat", "comfy"]

TARGET_META: dict[str, dict[str, str]] = {
    "router": {
        "label": "Router (OpenVINO)",
        "folder": "models/router",
        "hint": "INT4/INT8 OpenVINO GenAI exports for NPU/CPU routing.",
    },
    "chat": {
        "label": "Chat (GGUF)",
        "folder": "models/chat",
        "hint": "GGUF weights for LM Studio / Lemonade. Point the app at this folder or copy files in.",
    },
    "comfy": {
        "label": "ComfyUI",
        "folder": "models/comfyui",
        "hint": "Diffusion weights. Symlink or add as an extra ComfyUI models path.",
    },
}


class HfSearchQuery(BaseModel):
    query: str = ""
    target: Target = "router"
    limit: int = Field(default=12, ge=1, le=40)


class HfDownloadRequest(BaseModel):
    repo_id: str = Field(min_length=3)
    target: Target = "router"
    filename: str | None = None


@dataclass
class DownloadJob:
    id: str
    repo_id: str
    target: Target
    dest: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    started_at: str = field(default_factory=lambda: _utcnow())
    finished_at: str | None = None
    bytes_downloaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "target": self.target,
            "dest": self.dest,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "bytes_downloaded": self.bytes_downloaded,
        }


_jobs: dict[str, DownloadJob] = {}
_jobs_lock = threading.Lock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "Keylane/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# LM Studio scans its own directory, laid out <publisher>/<repo>/*.gguf. A
# GGUF anywhere else is invisible to it, so chat models go there rather than
# into Keylane's tree where nothing could load them.
LMSTUDIO_MODEL_DIRS = [
    Path.home() / ".lmstudio" / "models",
    Path.home() / ".cache" / "lm-studio" / "models",
]


def lmstudio_models_dir() -> Path | None:
    for directory in LMSTUDIO_MODEL_DIRS:
        if directory.is_dir():
            return directory
    return None


def target_dir(target: Target, repo_id: str) -> Path:
    meta = TARGET_META[target]
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "_", repo_id.replace("/", "__"))
    base = ROOT / meta["folder"]
    if target == "comfy":
        return base / "diffusion_models" / slug
    if target == "chat":
        lmstudio = lmstudio_models_dir()
        if lmstudio is not None and "/" in repo_id:
            publisher, name = repo_id.split("/", 1)
            return lmstudio / _safe(publisher) / _safe(name)
    return base / slug


def _safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", part)


def _parse_param_b(text: str) -> float | None:
    cleaned = text.replace("-", " ")
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*b(?:illion)?\b", cleaned)
    if match:
        return float(match.group(1))
    match = re.search(r"(?i)[-_](\d+(?:\.\d+)?)b(?:-|_|$)", text)
    if match:
        return float(match.group(1))
    return None


def _score_router(repo_id: str, tags: list[str], hw: HardwareProfile) -> tuple[int, str]:
    tid = repo_id.lower()
    tagset = {t.lower() for t in tags}
    score = 20
    reasons: list[str] = []

    if "openvino" in tagset or "openvino" in tid or repo_id.startswith("OpenVINO/"):
        score += 40
        reasons.append("OpenVINO export")
    else:
        score -= 30
        reasons.append("not an OpenVINO export")

    if any(x in tid for x in ("int4", "int4-ov", "4bit")) or "int4" in " ".join(tagset):
        score += 25
        reasons.append("INT4 (good for NPU/CPU)")
    elif any(x in tid for x in ("int8", "8bit")):
        score += 10
        reasons.append("INT8")

    params = _parse_param_b(tid)
    if params is not None:
        if hw.has_npu:
            if params <= 3:
                score += 20
                reasons.append(f"{params}B fits NPU routing")
            elif params <= 7:
                score += 5
                reasons.append(f"{params}B may be heavy on NPU")
            else:
                score -= 25
                reasons.append(f"{params}B too large for typical NPU router")
        else:
            ram = hw.ram_gb or 8
            if params <= 3:
                score += 15
                reasons.append(f"{params}B OK on CPU")
            elif params <= 7 and ram >= 16:
                score += 5
            else:
                score -= 10

    if "instruct" in tid or "chat" in tid:
        score += 5
    return score, ", ".join(reasons) or "compatible"


def _score_chat(repo_id: str, tags: list[str], hw: HardwareProfile) -> tuple[int, str]:
    tid = repo_id.lower()
    tagset = {t.lower() for t in tags}
    score = 15
    reasons: list[str] = []

    if "gguf" in tagset or "gguf" in tid:
        score += 35
        reasons.append("GGUF")
    else:
        score -= 40
        reasons.append("not GGUF")

    params = _parse_param_b(tid)
    vram = hw.nvidia_vram_mb or 0
    if params is not None:
        need_mb = int(params * 700)
        if vram:
            if need_mb <= vram * 0.85:
                score += 25
                reasons.append(f"~{params}B fits ~{vram} MB VRAM")
            elif need_mb <= vram * 1.2:
                score += 5
                reasons.append(f"~{params}B tight on {vram} MB")
            else:
                score -= 30
                reasons.append(f"~{params}B likely exceeds {vram} MB")
        else:
            ram = hw.ram_gb or 8
            if params <= ram * 0.4:
                score += 10
                reasons.append(f"CPU/RAM path (~{ram} GB)")
            else:
                score -= 15
                reasons.append("no discrete GPU VRAM detected")

    if any(k in tid for k in ("coder", "code")):
        score += 8
        reasons.append("coding-oriented")
    if any(k in tid for k in ("instruct", "chat")):
        score += 5
    return score, ", ".join(reasons) or "compatible"


def _score_comfy(repo_id: str, tags: list[str], hw: HardwareProfile) -> tuple[int, str]:
    tid = repo_id.lower()
    tagset = {t.lower() for t in tags}
    score = 15
    reasons: list[str] = []

    if "diffusers" in tagset or "text-to-image" in tagset or "flux" in tid:
        score += 25
        reasons.append("image model")
    if not hw.has_gpu:
        score -= 20
        reasons.append("GPU recommended for ComfyUI")
    elif (hw.nvidia_vram_mb or 0) >= 12000:
        score += 15
        reasons.append("ample VRAM")
    elif (hw.nvidia_vram_mb or 0) >= 8000:
        score += 8
    return score, ", ".join(reasons) or "compatible"


def _quant_preferences(hw: HardwareProfile) -> list[str]:
    """Quantisations to prefer, best first, for this machine's VRAM."""
    vram = hw.nvidia_vram_mb or 0
    if vram >= 20000:
        return ["Q5_K_M", "Q5_K_S", "Q4_K_M", "Q6_K"]
    if vram >= 10000:
        return ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q3_K_M"]
    return ["Q4_K_S", "Q3_K_M", "Q4_0", "Q2_K", "IQ4"]


def _gguf_allow_patterns(
    hw: HardwareProfile, filename: str | None, repo_files: list[str] | None = None
) -> list[str]:
    """Which files to fetch — exactly one weight file, plus metadata.

    Passing several quant globs downloads several copies of the same model.
    When the repository listing is available the best *available* quant is
    picked and everything else is left behind.
    """
    metadata = ["*.md", "*.json", "LICENSE*"]
    if filename:
        return [filename, *metadata]

    prefs = _quant_preferences(hw)
    if repo_files:
        ggufs = [f for f in repo_files if f.lower().endswith(".gguf")]
        # A projector file belongs with its model, so keep any mmproj.
        projectors = [f for f in ggufs if "mmproj" in f.lower()]
        for quant in prefs:
            match = next(
                (f for f in ggufs if quant.lower() in f.lower() and "mmproj" not in f.lower()),
                None,
            )
            if match:
                return [match, *projectors, *metadata]
        # No preferred quant in this repo: take the smallest weight file.
        weights = [f for f in ggufs if "mmproj" not in f.lower()]
        if weights:
            return [sorted(weights, key=len)[0], *projectors, *metadata]

    # No listing available; fall back to the single best pattern.
    return [f"*{prefs[0]}*.gguf", *metadata]


def _repo_files(repo_id: str, token: str | None) -> list[str]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, token=token)
    except Exception as exc:  # noqa: BLE001
        logger.info("Could not list %s: %s", repo_id, exc)
        return []
    return [
        str(getattr(s, "rfilename", "")) for s in (getattr(info, "siblings", None) or [])
    ]


async def search_models(
    query: str = "",
    *,
    target: Target = "router",
    limit: int = 12,
    hardware: HardwareProfile | None = None,
) -> dict[str, Any]:
    hw = hardware or detect_hardware()
    q = (query or "").strip()

    params: dict[str, Any] = {
        "sort": "downloads",
        "direction": -1,
        "limit": min(max(limit * 3, 20), 60),
    }

    if target == "router":
        params["filter"] = "openvino"
        params["search"] = q or "instruct int4"
    elif target == "chat":
        params["filter"] = "gguf"
        params["search"] = q or "instruct"
    else:
        params["filter"] = "diffusers"
        params["pipeline_tag"] = "text-to-image"
        params["search"] = q or "flux"

    async with httpx.AsyncClient(timeout=30.0, headers=_hf_headers()) as client:
        response = await client.get(HF_API, params=params)
        response.raise_for_status()
        raw = response.json()

    if not isinstance(raw, list):
        raw = []

    scored: list[dict[str, Any]] = []
    for item in raw:
        repo_id = str(item.get("id") or "")
        if not repo_id or "/" not in repo_id:
            continue
        tags = [str(t) for t in (item.get("tags") or [])]
        if target == "router":
            score, reason = _score_router(repo_id, tags, hw)
        elif target == "chat":
            score, reason = _score_chat(repo_id, tags, hw)
        else:
            score, reason = _score_comfy(repo_id, tags, hw)

        dest = target_dir(target, repo_id)
        installed = dest.exists() and any(dest.iterdir())
        try:
            dest_hint = str(dest.relative_to(ROOT))
        except ValueError:
            dest_hint = str(dest)

        scored.append(
            {
                "repo_id": repo_id,
                "name": repo_id.split("/")[-1],
                "author": repo_id.split("/")[0],
                "downloads": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "tags": tags[:12],
                "pipeline_tag": item.get("pipeline_tag"),
                "hf_url": f"https://huggingface.co/{repo_id}",
                "score": score,
                "compatible": score >= 40,
                "reason": reason,
                "target": target,
                "dest_hint": dest_hint,
                "installed": installed,
                "params_b": _parse_param_b(repo_id),
            }
        )

    scored.sort(key=lambda m: (m["compatible"], m["score"], m["downloads"]), reverse=True)
    results = scored[:limit]
    return {
        "query": q,
        "target": target,
        "target_meta": TARGET_META[target],
        "hardware_summary": hw.summary,
        "tier": hw.tier,
        "results": results,
        "count": len(results),
    }


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return job.to_dict() if job else None


def start_download(req: HfDownloadRequest) -> dict[str, Any]:
    repo_id = req.repo_id.strip()
    if not re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", repo_id):
        raise ValueError(f"Invalid Hugging Face repo id: {repo_id}")

    dest = target_dir(req.target, repo_id)
    job_id = uuid.uuid4().hex[:12]
    job = DownloadJob(
        id=job_id,
        repo_id=repo_id,
        target=req.target,
        dest=str(dest),
        status="queued",
        message="Queued",
    )
    with _jobs_lock:
        _jobs[job_id] = job

    threading.Thread(
        target=_run_download,
        args=(job_id, repo_id, req.target, dest, req.filename),
        daemon=True,
        name=f"hf-dl-{job_id}",
    ).start()
    return job.to_dict()


def _dir_size(path: Path) -> int:
    """Bytes on disk, including partial files still being written."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _expected_size(
    repo_id: str, token: str | None, patterns: list[str] | None = None
) -> int:
    """Size of the files this download will actually fetch.

    Counting the whole repository makes the percentage meaningless for a GGUF
    repo holding a dozen quantisations of the same model.
    """
    import fnmatch

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except Exception as exc:  # noqa: BLE001
        logger.info("Could not size %s: %s", repo_id, exc)
        return 0

    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        size = getattr(sibling, "size", None)
        if not isinstance(size, int):
            continue
        name = str(getattr(sibling, "rfilename", ""))
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        total += size
    return total


def _run_download(
    job_id: str,
    repo_id: str,
    target: Target,
    dest: Path,
    filename: str | None,
) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = "running"
        job.message = "Starting download…"
        job.progress = 0.02

    try:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Run: pip install huggingface_hub"
            ) from exc

        dest.mkdir(parents=True, exist_ok=True)
        hw = detect_hardware()
        token = _hf_token()

        def progress(value: float, message: str) -> None:
            with _jobs_lock:
                j = _jobs[job_id]
                j.progress = max(j.progress, min(value, 0.99))
                j.message = message

        progress(0.05, f"Fetching {repo_id}…")

        # snapshot_download gives no callback, so watch the destination grow.
        # Reporting 5% and 0 bytes for the whole download makes a stall look
        # exactly like progress — which is how a half-finished model once
        # looked like a broken NPU.
        listing = _repo_files(repo_id, token) if target == "chat" else []
        size_patterns = (
            _gguf_allow_patterns(hw, filename, listing) if target == "chat" else None
        )
        expected = _expected_size(repo_id, token, size_patterns)
        watcher_stop = threading.Event()

        def watch() -> None:
            while not watcher_stop.wait(1.5):
                fetched = _dir_size(dest)
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job is None or job.status != "running":
                        return
                    job.bytes_downloaded = fetched
                    if expected:
                        # Cap below 1.0: completion is declared by the worker.
                        job.progress = min(0.05 + 0.9 * fetched / expected, 0.95)
                        job.message = (
                            f"{fetched / 1e6:.0f} of {expected / 1e6:.0f} MB"
                        )
                    else:
                        job.message = f"{fetched / 1e6:.0f} MB fetched"

        watcher = threading.Thread(target=watch, daemon=True, name=f"hf-watch-{job_id}")
        watcher.start()

        if target == "chat" and filename:
            saved = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest),
                token=token,
            )
            progress(0.95, f"Saved {filename}")
            local_path = Path(saved).parent
        elif target == "chat":
            patterns = _gguf_allow_patterns(hw, None, listing)
            try:
                local_path = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(dest),
                        allow_patterns=patterns,
                        token=token,
                    )
                )
            except Exception:
                local_path = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(dest),
                        allow_patterns=["*.gguf", "*.md", "*.json"],
                        token=token,
                    )
                )
            progress(0.95, "GGUF download complete")
        elif target == "router":
            local_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(dest),
                    token=token,
                )
            )
            progress(0.95, "OpenVINO model download complete")
        else:
            local_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(dest),
                    allow_patterns=[
                        "*.safetensors",
                        "*.ckpt",
                        "*.pt",
                        "*.pth",
                        "*.json",
                        "*.txt",
                        "*.md",
                        "LICENSE*",
                    ],
                    token=token,
                )
            )
            progress(0.95, "ComfyUI assets download complete")

        watcher_stop.set()

        total = 0
        for path in Path(local_path).rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass

        try:
            rel = str(dest.relative_to(ROOT))
        except ValueError:
            rel = str(dest)

        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "done"
            job.progress = 1.0
            job.bytes_downloaded = total
            job.finished_at = _utcnow()
            job.message = f"Installed to {rel}"
            job.dest = str(dest)

        if target == "router":
            _maybe_set_router_path(rel)

    except Exception as exc:
        try:
            watcher_stop.set()
        except NameError:
            pass
        logger.exception("HF download failed for %s", repo_id)
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "error"
            job.error = str(exc)
            job.message = "Download failed"
            job.finished_at = _utcnow()


def _maybe_set_router_path(rel_path: str) -> None:
    try:
        from app.models_settings import load_models_settings, save_models_settings

        settings = load_models_settings()
        current = (settings.router_model_path or "").strip()
        if current in {"", "./models/router", "models/router"}:
            settings.router_model_path = (
                f"./{rel_path}" if not rel_path.startswith((".", "/")) else rel_path
            )
            settings.router_model_id = Path(rel_path).name.lower()
            save_models_settings(settings)
    except Exception as exc:
        logger.debug("Could not auto-update router path: %s", exc)


def targets_info() -> list[dict[str, Any]]:
    hw = detect_hardware()
    out: list[dict[str, Any]] = []
    for key, meta in TARGET_META.items():
        out.append(
            {
                "id": key,
                **meta,
                "recommended": (
                    key == "router"
                    or (key == "chat" and hw.has_gpu)
                    or (key == "comfy" and hw.has_gpu)
                ),
            }
        )
    return out
