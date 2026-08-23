"""Shared git / build evidence helpers for coding workers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


async def _run(cmd: list[str], cwd: Path, timeout: float = 120.0) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            out_b.decode("utf-8", errors="replace"),
            err_b.decode("utf-8", errors="replace"),
        )
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


async def collect_git_evidence(cwd: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "",
        "diff": "",
        "diff_stat": "",
        "build_exit_code": None,
        "test_exit_code": None,
        "lint_exit_code": None,
    }
    if not (cwd / ".git").exists():
        return result

    _, status, _ = await _run(["git", "status", "--porcelain"], cwd)
    _, diff, _ = await _run(["git", "diff"], cwd)
    _, diff_stat, _ = await _run(["git", "diff", "--stat"], cwd)
    result["status"] = status
    result["diff"] = diff
    result["diff_stat"] = diff_stat

    # Optional build/test collection (can be slow). Enable with AI_GATEWAY_RUN_CHECKS=1.
    if os.environ.get("AI_GATEWAY_RUN_CHECKS") == "1" and (cwd / "package.json").exists():
        code, out, err = await _run(
            ["npm", "run", "build", "--if-present"], cwd, timeout=300
        )
        result["build_exit_code"] = code
        if code != 0:
            result["diff"] = (result["diff"] + "\n" + out + "\n" + err)[-12000:]
        code_t, out_t, err_t = await _run(
            ["npm", "test", "--if-present"], cwd, timeout=300
        )
        if "Missing script" not in (out_t + err_t):
            result["test_exit_code"] = code_t

    return result


def list_changed_files(before_diff: str, after_diff: str) -> list[str]:
    """Rough set of files appearing in the newer diff."""
    files: set[str] = set()
    for line in after_diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                files.add(path)
    return sorted(files)
