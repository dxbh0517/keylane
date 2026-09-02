"""Import a model from Hugging Face that is not on the curated list.

The curated list exists so the common case needs no decisions. This is the
escape hatch for everything else, and it has one job worth doing carefully:
work out, before anything is downloaded, whether the repo holds something
Keylane can actually run — and if it holds several builds, which one.

That question is answered from the repo's file listing alone. A repo with
``openvino_model.xml`` in it is an OpenVINO IR export; one with
``genai_config.json`` is an ONNX Runtime GenAI export; one with neither is a
PyTorch checkpoint that would need converting first, and saying so up front is
better than a 15 GB download that fails to load.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from typing import Any

from daemon.config import get_section, save_settings
from daemon.paths import MODELS_DIR
from runtimes import RepoVariant, backend_for, list_backends, normalise_runtime_id
from runtimes.onnx_rt import RUNNABLE_SCORE

logger = logging.getLogger(__name__)

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
_PARAMS_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[bB](?![\w])")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ImportError_(ValueError):
    """A repo that cannot become a model entry, with the reason why."""


@dataclass
class RepoInspection:
    """What a Hugging Face repo turned out to contain."""

    repo: str
    variants: list[RepoVariant] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)
    params_b: int = 0
    private: bool = False

    @property
    def best(self) -> RepoVariant | None:
        if not self.variants:
            return None
        return max(self.variants, key=lambda v: (v.score, -len(v.subfolder)))

    @property
    def runnable(self) -> list[RepoVariant]:
        """Builds meant for this machine, rather than for CUDA or DirectML."""
        return [v for v in self.variants if v.score >= RUNNABLE_SCORE]

    def variant(self, runtime: str | None, subfolder: str | None) -> RepoVariant | None:
        """The variant matching an explicit request, or the best one going."""
        if subfolder is None and runtime is None:
            return self.best
        candidates = [
            v
            for v in self.variants
            if (subfolder is None or v.subfolder == subfolder.strip("/"))
            and (runtime is None or v.runtime == normalise_runtime_id(runtime))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda v: (v.score, -len(v.subfolder)))


def normalise_repo(repo: str) -> str:
    """Accept a bare id, a full URL, or something pasted with whitespace."""
    text = (repo or "").strip()
    text = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", text)
    text = text.split("?", 1)[0].rstrip("/")
    # A URL deep-linked into the file browser carries the path along with it.
    text = re.sub(r"/(?:tree|blob|resolve)/[^/]+.*$", "", text)
    if not _REPO_RE.match(text):
        raise ImportError_(
            f"{repo!r} is not a Hugging Face repo id — expected something like "
            "OpenVINO/Qwen2.5-7B-Instruct-int4-ov"
        )
    return text


def _params_from_name(repo: str) -> int:
    match = _PARAMS_RE.search(repo.rsplit("/", 1)[-1])
    if not match:
        return 0
    try:
        return max(1, round(float(match.group(1))))
    except ValueError:
        return 0


def suggest_model_id(repo: str, subfolder: str = "") -> str:
    """A stable, filesystem-safe id — it becomes the download directory."""
    name = repo.rsplit("/", 1)[-1]
    if subfolder:
        name = f"{name}-{subfolder.rsplit('/', 1)[-1]}"
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "imported-model"


def inspect_repo(repo: str) -> RepoInspection:
    """Ask Hugging Face what is in the repo. Network call; no download."""
    repo_id = normalise_repo(repo)
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError as exc:  # pragma: no cover - huggingface_hub is required
        raise ImportError_(f"huggingface_hub is not installed: {exc}") from exc

    try:
        info = HfApi().repo_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError as exc:
        raise ImportError_(
            f"no such model on Hugging Face: {repo_id} (or it is private and "
            "this machine is not logged in)"
        ) from exc
    except GatedRepoError as exc:
        raise ImportError_(
            f"{repo_id} is gated — accept its licence on huggingface.co and run "
            "`huggingface-cli login` first"
        ) from exc
    except HfHubHTTPError as exc:
        raise ImportError_(f"could not read {repo_id} from Hugging Face: {exc}") from exc

    filenames = [str(s.rfilename) for s in info.siblings or []]
    sizes: dict[str, int] = {}
    for sibling in info.siblings or []:
        name = str(sibling.rfilename)
        folder = name.rsplit("/", 1)[0] if "/" in name else ""
        sizes[folder] = sizes.get(folder, 0) + int(getattr(sibling, "size", 0) or 0)

    variants: list[RepoVariant] = []
    for backend in list_backends():
        variants.extend(backend.repo_variants(filenames))

    return RepoInspection(
        repo=repo_id,
        variants=variants,
        sizes=sizes,
        params_b=_params_from_name(repo_id),
        private=bool(getattr(info, "private", False)),
    )


def _no_variant_message(inspection: RepoInspection) -> str:
    return (
        f"{inspection.repo} has no export Keylane can run. It needs either an "
        "OpenVINO IR export (openvino_model.xml — repos named *-int4-ov) or an "
        "ONNX Runtime GenAI export (genai_config.json). A plain PyTorch or "
        "GGUF repo has to be converted first."
    )


def list_imported() -> list[dict[str, Any]]:
    raw = get_section("models").get("imported", [])
    return [dict(entry) for entry in raw if isinstance(entry, dict)]


def import_model(
    repo: str,
    *,
    runtime: str | None = None,
    subfolder: str | None = None,
    name: str = "",
    device: str = "",
    model_id: str = "",
) -> dict[str, Any]:
    """Add a Hugging Face repo to the catalog. Does not download it."""
    inspection = inspect_repo(repo)
    if not inspection.variants:
        raise ImportError_(_no_variant_message(inspection))

    # An explicit subfolder is the user overruling the ranking, which is
    # their call. An automatic pick has to be something that can actually run.
    explicit = subfolder is not None
    if not explicit and not inspection.runnable:
        offered = ", ".join(v.subfolder or "<root>" for v in inspection.variants)
        raise ImportError_(
            f"{inspection.repo} only ships builds for other hardware ({offered}). "
            "None of them will load through the OpenVINO execution provider. "
            "Pass an explicit subfolder to import one anyway."
        )

    variant = inspection.variant(runtime, subfolder)
    if variant is None:
        offered = ", ".join(
            f"{v.runtime}:{v.subfolder or '<root>'}" for v in inspection.variants
        )
        raise ImportError_(
            f"{inspection.repo} has no build matching that runtime and folder. "
            f"It offers: {offered}"
        )

    # The build name only earns a place in the id when the repo ships more
    # than one — otherwise every ONNX import gets a 60-character directory.
    distinguish = variant.subfolder if len(inspection.variants) > 1 else ""
    entry_id = (model_id or suggest_model_id(inspection.repo, distinguish)).strip()
    resolved_device = device.strip() or backend_for(variant.runtime).info.default_device

    entry: dict[str, Any] = {
        "id": entry_id,
        "name": name.strip() or inspection.repo.rsplit("/", 1)[-1],
        "hf_repo": inspection.repo,
        "runtime": variant.runtime,
        "subfolder": variant.subfolder,
        "params_b": inspection.params_b,
        "device": resolved_device,
        "description": f"Imported from {inspection.repo}"
        + (f" ({variant.subfolder})" if variant.subfolder else ""),
        "size_bytes": inspection.sizes.get(variant.subfolder, 0),
    }

    from models.catalog import get_model  # noqa: PLC0415 - avoids an import cycle

    existing = get_model(entry_id)
    if existing is not None and existing.source != "imported":
        raise ImportError_(
            f"{entry_id!r} is already a curated model — pass a different id to "
            "import this one alongside it"
        )

    imported = [e for e in list_imported() if str(e.get("id")) != entry_id]
    imported.append(entry)
    save_settings("models", {"imported": imported})
    logger.info("imported %s as %s (%s)", inspection.repo, entry_id, variant.runtime)
    return entry


def remove_imported(model_id: str, *, delete_files: bool = False) -> bool:
    """Forget an imported model, optionally deleting what it downloaded."""
    imported = list_imported()
    remaining = [e for e in imported if str(e.get("id")) != model_id]
    if len(remaining) == len(imported):
        return False
    save_settings("models", {"imported": remaining})
    if delete_files:
        target = MODELS_DIR / model_id
        # Only ever inside the models directory, and only a directory we named.
        if target.is_dir() and target.parent == MODELS_DIR:
            shutil.rmtree(target, ignore_errors=True)
    return True
