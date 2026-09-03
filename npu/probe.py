"""Subprocess probe before loading OpenVINO GenAI in the daemon process."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from npu.kind import PipelineKind, model_kind
from runtimes.probe_runner import OK_MARKER, last_line, run_probe

logger = logging.getLogger(__name__)

_CACHE_OVERRIDE = os.environ.get("KEYLANE_NPU_CACHE_DIR") or ""
PROBE_TIMEOUT = float(os.environ.get("KEYLANE_NPU_PROBE_TIMEOUT", "90"))

# The probe compiles at the prompt length the load wants, so the first compile
# happens here rather than in the untimed load that followed it, and this
# deadline has to cover it.
#
# These used to be one hour and three hours, set when a 4096-token compile
# really did take half an hour. Measured on 2026-09-03 on an NPU 3720:
#
#   OpenVINO 2026.3 + Fedora's driver 1.32.0 / compiler 2025.1 : 60 s
#   OpenVINO 2026.2.1 + the matched v1.35.0 user-space stack   :  6.8 s
#
# So most of that minute was a driver and a compiler five releases apart, and
# a working stack compiles a 7B in seconds. MAX_PROMPT_LEN barely moves either
# figure — since 2025.3 the pipeline chunks the prefill instead of compiling a
# static graph that wide.
#
# Ten minutes is far more headroom than 6.8 s needs, and deliberately so: this
# ceiling also has to cover a larger model, a slower NPU, and the mismatched
# stack above, which is the configuration someone hits before they know to fix
# it. It is sized to fail visibly rather than to fit this machine. Re-measure
# with `scripts/npu-bench.py` rather than adjusting by feel; it writes what it
# found to data/bench.json so the next person has a number instead of a guess.
WARM_TIMEOUT = float(os.environ.get("KEYLANE_NPU_WARM_TIMEOUT", "600"))
# A VLM compiles a vision tower as well, and that part is genuinely slower.
# Unmeasured here — the one VLM in the catalog was asymmetric and got replaced —
# so this keeps generous headroom rather than pretending to a number.
VLM_WARM_TIMEOUT = float(os.environ.get("KEYLANE_NPU_VLM_WARM_TIMEOUT", "3600"))

_VERSION_RE = re.compile(r'<openvino_version value="(\d+)\.(\d+)')


def warm_timeout_for(kind: PipelineKind) -> float:
    """First NPU compile for VLMs can exceed 30 minutes on laptop NPUs."""
    return VLM_WARM_TIMEOUT if kind == "vlm" else WARM_TIMEOUT


def _probe_source(kind: PipelineKind) -> str:
    """The child builds the pipeline the daemon is about to build.

    It imports the constructor rather than restating it, so the compile the
    probe pays for is the compile the load then finds in the cache. When the
    two were written out separately they drifted apart, and the probe's work
    was thrown away every time.
    """
    return f"""
import sys
from pathlib import Path
from npu.pipeline_config import create_pipeline

path, device, cache = sys.argv[1], sys.argv[2], sys.argv[3]
create_pipeline(Path(path), device, Path(cache) if cache else None, "{kind}")
print("{OK_MARKER}")
"""


def _probe_env() -> dict[str, str]:
    """The child imports from the Keylane tree, so it has to be on the path."""
    from daemon.paths import ROOT

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)
    return env


def cache_dir(root: Path | None = None) -> Path:
    if _CACHE_OVERRIDE:
        path = Path(_CACHE_OVERRIDE).expanduser()
    else:
        from daemon.paths import CACHE_DIR

        path = CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_version() -> tuple[int, int] | None:
    try:
        import openvino as ov

        major, minor = ov.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:  # noqa: BLE001
        return None


def _ir_version(xml: Path) -> tuple[int, int] | None:
    try:
        text = xml.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = _VERSION_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def static_objection(path: Path) -> str | None:
    tokenizer = path / "openvino_tokenizer.xml"
    if tokenizer.exists():
        exported = _ir_version(tokenizer)
        runtime = _runtime_version()
        if exported and runtime and exported > runtime:
            return (
                f"tokenizer exported by OpenVINO {exported[0]}.{exported[1]} "
                f"is newer than installed {runtime[0]}.{runtime[1]}"
            )
    if not any(path.glob("openvino*.xml")):
        return "no OpenVINO IR files found"
    return None


def _timeout_message(kind: PipelineKind, timeout: float) -> str:
    mins = timeout / 60
    if kind == "vlm":
        return (
            f"timeout:NPU compile did not finish within {mins:.0f} min "
            f"(a vision model compiles its vision tower too; raise "
            f"KEYLANE_NPU_VLM_WARM_TIMEOUT if this model is simply large)"
        )
    return (
        f"timeout:compile did not finish within {mins:.0f} min. A 7B compile "
        f"measures around a minute on a working stack, so this usually means a "
        f"driver and compiler that do not match — run scripts/npu-driver-fix.sh"
    )


def probe(
    model_path: Path,
    device: str,
    *,
    cache: Path | None,
    timeout: float | None = None,
    kind: PipelineKind | None = None,
    on_tick: Callable[[float], None] | None = None,
) -> tuple[bool, str]:
    pipeline_kind = kind or model_kind(model_path)
    if timeout is None:
        timeout = warm_timeout_for(pipeline_kind)
    argv = [
        sys.executable,
        "-c",
        _probe_source(pipeline_kind),
        str(model_path),
        device,
        str(cache or ""),
    ]
    return run_probe(
        argv,
        timeout=timeout,
        timeout_message=_timeout_message(pipeline_kind, timeout),
        on_tick=on_tick,
        env=_probe_env(),
    )


_last_line = last_line
