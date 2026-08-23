"""Claude Code controlled subprocess worker."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from app.config import AppConfig, get_config
from app.schemas import RouteDecision, WorkerEvidence, WorkerResult
from app.workers._git_evidence import collect_git_evidence, list_changed_files

logger = logging.getLogger(__name__)


class ClaudeWorker:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def _command(self) -> str | None:
        cmd = self.config.claude.command
        return shutil.which(cmd) or (cmd if Path(cmd).exists() else None)

    async def health(self) -> bool:
        return self._command() is not None

    async def run(self, decision: RouteDecision) -> WorkerResult:
        binary = self._command()
        if binary is None:
            evidence = WorkerEvidence(
                worker="claude",
                action=decision.action,
                stderr="claude command not found",
                exit_code=127,
            )
            return WorkerResult(success=False, evidence=evidence, summary="Claude CLI missing")

        if not decision.working_directory:
            evidence = WorkerEvidence(
                worker="claude",
                action=decision.action,
                stderr="working_directory required",
                exit_code=2,
            )
            return WorkerResult(
                success=False,
                evidence=evidence,
                summary="Select a project before using Claude Code.",
            )

        cwd = Path(decision.working_directory)
        before = await collect_git_evidence(cwd)

        # Controlled invocation — no permission bypass flags.
        cmd = [
            binary,
            "-p",
            decision.instruction,
            "--output-format",
            "json",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.claude.timeout_seconds,
            )
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            evidence = WorkerEvidence(
                worker="claude",
                action=decision.action,
                stderr="Claude Code timed out",
                exit_code=124,
            )
            return WorkerResult(success=False, evidence=evidence, summary="Claude timed out")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Claude worker failed")
            evidence = WorkerEvidence(
                worker="claude",
                action=decision.action,
                stderr=str(exc),
                exit_code=1,
            )
            return WorkerResult(success=False, evidence=evidence, summary=str(exc))

        after = await collect_git_evidence(cwd)
        changed = list_changed_files(before.get("diff", ""), after.get("diff", ""))
        if not changed and after.get("status"):
            changed = [
                line[3:].strip()
                for line in after["status"].splitlines()
                if line.strip() and not line.startswith("?")
            ]

        evidence = WorkerEvidence(
            worker="claude",
            action=decision.action,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            changed_files=changed,
            git_status=after.get("status", ""),
            git_diff=after.get("diff", ""),
            git_diff_stat=after.get("diff_stat", ""),
            build_exit_code=after.get("build_exit_code"),
            test_exit_code=after.get("test_exit_code"),
            lint_exit_code=after.get("lint_exit_code"),
        )
        summary = stdout.strip() or stderr.strip() or f"Claude exited {exit_code}"
        return WorkerResult(
            success=exit_code == 0,
            evidence=evidence,
            summary=summary[:4000],
            raw={"exit_code": exit_code},
        )
