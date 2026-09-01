"""Subprocess probe before loading OpenVINO GenAI in the daemon process."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from npu.kind import PipelineKind, model_kind

logger = logging.getLogger(__name__)

_CACHE_OVERRIDE = os.environ.get("KEYLANE_NPU_CACHE_DIR") or ""
PROBE_TIMEOUT = float(os.environ.get("KEYLANE_NPU_PROBE_TIMEOUT", "90"))
WARM_TIMEOUT = float(os.environ.get("KEYLANE_NPU_WARM_TIMEOUT", "1800"))
VLM_WARM_TIMEOUT = float(os.environ.get("KEYLANE_NPU_VLM_WARM_TIMEOUT", "10800"))

_VERSION_RE = re.compile(r'<openvino_version value="(\d+)\.(\d+)')


def warm_timeout_for(kind: PipelineKind) -> float:
    """First NPU compile for VLMs can exceed 30 minutes on laptop NPUs."""
    return VLM_WARM_TIMEOUT if kind == "vlm" else WARM_TIMEOUT


def _probe_source(kind: PipelineKind) -> str:
    if kind == "vlm":
        body = """
device_props = {}
if device.upper() == "NPU":
    device_props["GENERATE_HINT"] = "FAST_COMPILE"
if cache:
    device_props["CACHE_DIR"] = cache
if device_props:
    config = {"DEVICE_PROPERTIES": {device: device_props}}
    ov_genai.VLMPipeline(path, device, config=config)
else:
    ov_genai.VLMPipeline(path, device)
"""
    else:
        body = """
kwargs = {}
if device.upper() == "NPU":
    kwargs["MAX_PROMPT_LEN"] = 1024
    kwargs["GENERATE_HINT"] = "FAST_COMPILE"
if cache:
    kwargs["CACHE_DIR"] = cache
if kwargs:
    ov_genai.LLMPipeline(path, device, **kwargs)
else:
    ov_genai.LLMPipeline(path, device)
"""
    return f"""
import sys
path, device, cache = sys.argv[1], sys.argv[2], sys.argv[3]
import openvino_genai as ov_genai
{body}
print("KEYLANE_PROBE_OK")
"""


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
            f"(VLM first compile can take 60–90+ min; set KEYLANE_NPU_VLM_WARM_TIMEOUT)"
        )
    return f"timeout:compile did not finish within {mins:.0f} min"


def _signal_failure(returncode: int, stderr: str, stdout: str) -> str:
    sig = -returncode
    names = {6: "SIGABRT", 11: "SIGSEGV", 15: "SIGTERM"}
    signal_name = names.get(sig, str(sig))
    detail = _last_line(stderr) or _last_line(stdout)
    if sig == 15:
        hint = "compile interrupted (daemon restart or timeout)"
        return f"interrupted:{hint}. {detail}".strip()
    return f"crash:loading killed process ({signal_name}). {detail}".strip()


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
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"error:could not start probe: {exc}"

    start = time.time()
    last_tick = 0.0
    stdout = ""
    stderr = ""
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return False, _timeout_message(pipeline_kind, timeout)

        if on_tick and elapsed - last_tick >= 5:
            on_tick(elapsed)
            last_tick = elapsed

        try:
            stdout, stderr = proc.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                break
            continue

    if proc.returncode == 0 and "KEYLANE_PROBE_OK" in stdout:
        return True, "ready"

    if proc.returncode is not None and proc.returncode < 0:
        return False, _signal_failure(proc.returncode, stderr, stdout)

    detail = _last_line(stderr) or f"exit code {proc.returncode}"
    return False, f"error:{detail}"


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if line.startswith(("File ", "Traceback", "  ")):
            continue
        return line[:300]
    return lines[-1][:300]
