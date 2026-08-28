"""Subprocess probe before loading OpenVINO GenAI in the daemon process."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_OVERRIDE = os.environ.get("KEYLANE_NPU_CACHE_DIR") or ""
PROBE_TIMEOUT = float(os.environ.get("KEYLANE_NPU_PROBE_TIMEOUT", "90"))
WARM_TIMEOUT = float(os.environ.get("KEYLANE_NPU_WARM_TIMEOUT", "1800"))

_VERSION_RE = re.compile(r'<openvino_version value="(\d+)\.(\d+)')

_PROBE_SOURCE = """
import sys
path, device, cache = sys.argv[1], sys.argv[2], sys.argv[3]
import openvino_genai as ov_genai
kwargs = {"CACHE_DIR": cache} if cache else {}
ov_genai.LLMPipeline(path, device, **kwargs)
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
    if not (path / "openvino_model.xml").exists():
        if (path / "openvino_language_model.xml").exists():
            return "vision-language export; text pipeline cannot drive it"
    if (path / "openvino_vision_embeddings_model.xml").exists():
        return "multimodal model with vision tower"
    return None


def probe(
    model_path: Path,
    device: str,
    *,
    cache: Path | None,
    timeout: float = PROBE_TIMEOUT,
) -> tuple[bool, str]:
    argv = [sys.executable, "-c", _PROBE_SOURCE, str(model_path), device, str(cache or "")]
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout:compile did not finish within {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"error:could not start probe: {exc}"

    if done.returncode == 0 and "KEYLANE_PROBE_OK" in done.stdout:
        return True, "ready"

    if done.returncode < 0:
        signal_name = {6: "SIGABRT", 11: "SIGSEGV"}.get(-done.returncode, str(-done.returncode))
        detail = _last_line(done.stderr) or _last_line(done.stdout)
        return False, f"crash:loading kills process ({signal_name}). {detail}".strip()

    detail = _last_line(done.stderr) or f"exit code {done.returncode}"
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
