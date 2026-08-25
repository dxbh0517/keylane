"""Load a model somewhere it cannot take the gateway with it.

OpenVINO does not always fail by raising. Two real cases from this machine:

* a tokenizer IR exported by a *newer* OpenVINO than the installed runtime
  walks off the end of a graph in ``MakePaddingSatateful`` and lands on
  SIGSEGV;
* a vision tower the NPU compiler cannot shape aborts the process from inside
  LLVM (``LLVM ERROR: Failed to infer result type(s)``).

Neither is a Python exception, so the ``try/except`` around ``LLMPipeline``
never runs — the whole uvicorn process dies, systemd restarts it, and five
restarts later the unit is failed and Keylane is simply gone. A model the user
picked in the web UI should degrade the assistant to its heuristic path, not
uninstall the gateway.

So the first construction of any model happens in a *subprocess*. There a
segfault is an exit code we can read. Only once that subprocess has come back
clean does the parent load the same model in-process, which by then is a blob
cache hit and costs seconds rather than minutes.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Where OpenVINO keeps compiled blobs. One shared directory is correct: the
# cache is keyed by a hash of graph + device + compile config, so models cannot
# collide, and a single directory is far easier to clear than a tree of them.
_CACHE_OVERRIDE = os.environ.get("KEYLANE_NPU_CACHE_DIR") or ""

# A warm cache turns a 9-14 minute NPU compile into a few seconds, so the
# startup probe can afford to be impatient. When it does time out we assume a
# cold cache rather than a broken model, and hand the compile to a background
# thread (see NpuPipeline._warm_in_background) which gets the long budget.
PROBE_TIMEOUT = float(os.environ.get("KEYLANE_NPU_PROBE_TIMEOUT", "90"))
WARM_TIMEOUT = float(os.environ.get("KEYLANE_NPU_WARM_TIMEOUT", "1800"))


def cache_dir(config=None) -> Path:
    """Directory for OpenVINO's compiled-blob cache, created on demand."""
    if _CACHE_OVERRIDE:
        path = Path(_CACHE_OVERRIDE).expanduser()
    else:
        from app.config import ROOT

        root = getattr(config, "root", None) or ROOT
        path = Path(root) / "cache" / "openvino"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------- static checks

_VERSION_RE = re.compile(r'<openvino_version value="(\d+)\.(\d+)')


def _runtime_version() -> tuple[int, int] | None:
    try:
        import openvino as ov

        major, minor = ov.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:  # noqa: BLE001
        return None


def _ir_version(xml: Path) -> tuple[int, int] | None:
    """The OpenVINO version that exported an IR, from its ``rt_info``."""
    try:
        text = xml.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = _VERSION_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def static_objection(path: Path) -> str | None:
    """A reason to refuse *before* spending a subprocess on it, or ``None``.

    These are the cases that are both cheap to spot and certain to end badly,
    so catching them here buys a precise message instead of "the probe died".
    """
    tokenizer = path / "openvino_tokenizer.xml"
    if tokenizer.exists():
        exported = _ir_version(tokenizer)
        runtime = _runtime_version()
        if exported and runtime and exported > runtime:
            return (
                f"its tokenizer was exported by OpenVINO {exported[0]}.{exported[1]}, "
                f"newer than the installed {runtime[0]}.{runtime[1]} — loading it "
                "crashes the runtime rather than raising. Re-export the tokenizer, "
                "or upgrade OpenVINO."
            )

    # A vision-language export splits the model in two: an encoder, plus a
    # language half that consumes `inputs_embeds` instead of `input_ids`.
    # LLMPipeline feeds input_ids, so it can never drive one of these, and the
    # NPU compiler aborts outright on the vision tower.
    if not (path / "openvino_model.xml").exists():
        if (path / "openvino_language_model.xml").exists():
            return (
                "it is a vision-language export (no openvino_model.xml, only a "
                "split language model) — the text pipeline cannot drive it."
            )
    if (path / "openvino_vision_embeddings_model.xml").exists():
        return (
            "it is a multimodal model with a vision tower, which the text "
            "pipeline cannot load."
        )
    return None


# ------------------------------------------------------------------- the probe

# Kept as source text rather than a function so it runs in a clean interpreter
# with no inherited OpenVINO state.
_PROBE_SOURCE = """
import sys
path, device, cache = sys.argv[1], sys.argv[2], sys.argv[3]
import openvino_genai as ov_genai
kwargs = {"CACHE_DIR": cache} if cache else {}
ov_genai.LLMPipeline(path, device, **kwargs)
print("KEYLANE_PROBE_OK")
"""


def probe(
    model_path: Path,
    device: str,
    *,
    cache: Path | None,
    timeout: float = PROBE_TIMEOUT,
) -> tuple[bool, str]:
    """Try to build the pipeline in a subprocess.

    Returns ``(ok, reason)``. ``ok`` means the same construction is now safe to
    repeat in this process, and — if ``cache`` was given — cheap. The reason
    distinguishes the three ways this goes wrong, because they need different
    responses: a crash means never load this model, a timeout means the compile
    is merely slow, and an exception is the model's own complaint.
    """
    argv = [sys.executable, "-c", _PROBE_SOURCE, str(model_path), device, str(cache or "")]
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            # The probe must never inherit a controlling terminal's death, and
            # must never block on stdin.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout:compile did not finish within {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"error:could not start probe: {exc}"

    if done.returncode == 0 and "KEYLANE_PROBE_OK" in done.stdout:
        return True, "ready"

    if done.returncode < 0:
        # Killed by a signal: SIGSEGV (-11) or SIGABRT (-6) from OpenVINO or
        # the NPU compiler. This is the case that used to take the gateway out.
        signal_name = {6: "SIGABRT", 11: "SIGSEGV"}.get(-done.returncode, str(-done.returncode))
        detail = _last_meaningful_line(done.stderr) or _last_meaningful_line(done.stdout)
        return False, f"crash:loading it kills the process ({signal_name}). {detail}".strip()

    detail = _last_meaningful_line(done.stderr) or f"exit code {done.returncode}"
    return False, f"error:{detail}"


def _last_meaningful_line(text: str) -> str:
    """The most specific-looking line of a failure, trimmed for a log line."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        # Skip Python's own framing; the useful part is OpenVINO's message.
        if line.startswith(("File ", "Traceback", "  ")):
            continue
        return line[:300]
    return lines[-1][:300]


def failure_kind(reason: str) -> str:
    """``crash``, ``timeout`` or ``error`` from a :func:`probe` reason."""
    head, _, _ = reason.partition(":")
    return head if head in {"crash", "timeout", "error"} else "error"


def failure_detail(reason: str) -> str:
    _, _, rest = reason.partition(":")
    return rest or reason
