"""Run a model compile in a subprocess and report why it failed.

Compiling a model for the NPU can abort the process rather than raise — a bad
export, a driver mismatch, or simply not enough memory takes the interpreter
with it. So every runtime compiles once in a child process before the daemon
loads the model itself, and this is the child-process bookkeeping both of them
share: tick while it works, kill it if it overruns, and turn whatever it left
behind into a sentence a person can act on.
"""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable
from typing import Mapping, Sequence

OK_MARKER = "KEYLANE_PROBE_OK"


# The line a traceback ends on: `RuntimeError: something went wrong`. Matched
# so a message that runs to several lines can be taken whole.
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit)?: ")


def last_line(text: str) -> str:
    """The useful part of a traceback: the exception and everything it said.

    Taking the *last* line is right for a one-line message and wrong for one
    written to be read. A load failure explains which model, how much VRAM it
    needed and what is holding the card — several sentences over several lines
    — and reading from the end returned the final fragment while discarding
    every part that made it useful.

    So the exception line is found and taken with its continuation, newlines
    folded to keep this a single line for the caller.
    """
    raw = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not raw:
        return ""

    def _is_frame(line: str) -> bool:
        stripped = line.strip()
        return line.startswith((" ", "\t")) or stripped.startswith(("File ", "Traceback"))

    # The last line that *starts* an exception, not the last line overall. A
    # continuation ("Underlying error: …") is neither frame noise nor a new
    # exception, so scanning for the first non-frame line from the end stops on
    # it and throws away the sentences above.
    for index in range(len(raw) - 1, -1, -1):
        if _is_frame(raw[index]):
            continue
        if _EXCEPTION_LINE.match(raw[index].strip()):
            message = " ".join(part.strip() for part in raw[index:])
            return message[:600]

    for line in reversed(raw):
        if not _is_frame(line):
            return line.strip()[:600]
    return raw[-1].strip()[:600]


def signal_failure(returncode: int, stderr: str, stdout: str) -> str:
    sig = -returncode
    names = {6: "SIGABRT", 11: "SIGSEGV", 15: "SIGTERM"}
    signal_name = names.get(sig, str(sig))
    detail = last_line(stderr) or last_line(stdout)
    if sig == 15:
        hint = "compile interrupted (daemon restart or timeout)"
        return f"interrupted:{hint}. {detail}".strip()
    return f"crash:loading killed process ({signal_name}). {detail}".strip()


def run_probe(
    argv: Sequence[str],
    *,
    timeout: float,
    timeout_message: str,
    on_tick: Callable[[float], None] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Run *argv*, expecting it to print ``KEYLANE_PROBE_OK`` and exit 0."""
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            env=dict(env) if env is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"error:could not start probe: {exc}"

    start = time.time()
    last_reported = 0.0
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
            return False, timeout_message

        if on_tick and elapsed - last_reported >= 5:
            on_tick(elapsed)
            last_reported = elapsed

        try:
            stdout, stderr = proc.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                break
            continue

    if proc.returncode == 0 and OK_MARKER in stdout:
        return True, "ready"

    if proc.returncode is not None and proc.returncode < 0:
        return False, signal_failure(proc.returncode, stderr, stdout)

    detail = last_line(stderr) or f"exit code {proc.returncode}"
    return False, f"error:{detail}"
