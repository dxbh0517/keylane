"""Shell tool — run an allowlisted command and return its output.

The model never gets a raw shell. It supplies a program plus argument list; the
program must appear in the configured allowlist, arguments are passed through
``execve`` (no shell interpretation), and the whole thing is danger-gated so the
confirmation policy applies.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from app.assistant_settings import load_assistant_settings
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    int_prop,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)

# Never runnable, even if someone adds them to the allowlist by accident.
FORBIDDEN = {
    "rm",
    "rmdir",
    "shred",
    "mkfs",
    "dd",
    "fdisk",
    "parted",
    "sudo",
    "su",
    "doas",
    "pkexec",
    "chown",
    "chmod",
    "passwd",
    "useradd",
    "userdel",
    "usermod",
    "visudo",
    "reboot",
    "poweroff",
    "shutdown",
    "halt",
    "init",
    "sh",
    "bash",
    "zsh",
    "fish",
    "dash",
    "eval",
    "exec",
}

# Argument shapes that would turn an allowlisted binary into a shell.
DANGEROUS_ARG_PREFIXES = ("-c", "--command", "-e", "--eval", "--exec")


class RunCommandTool(BaseTool):
    name = "run_command"
    description = (
        "Run an allowlisted command on this computer and return stdout/stderr. "
        "Arguments are passed directly to the program — there is no shell, so "
        "pipes, redirection and globbing do not work."
    )
    danger = ToolDanger.DANGEROUS
    category = "system"

    def parameters(self) -> dict[str, Any]:
        allowlist = load_assistant_settings().shell.allowlist
        return object_schema(
            {
                "command": string_prop(
                    "Program to run. Must be allowlisted: " + ", ".join(sorted(allowlist)[:30]),
                ),
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments passed to the program, one per element.",
                },
                "cwd": string_prop("Working directory (defaults to the configured one)."),
                "timeout_seconds": int_prop("Kill the command after this long.", default=60),
            },
            required=["command"],
        )

    def availability(self) -> str | None:
        settings = load_assistant_settings().shell
        if not settings.enabled:
            return "Shell commands are disabled in assistant settings"
        return None

    async def run(self, args: dict[str, Any]) -> ToolResult:
        settings = load_assistant_settings().shell
        if not settings.enabled:
            return ToolResult.failure("Shell commands are disabled in assistant settings.")

        raw_command = str(args.get("command") or "").strip()
        if not raw_command:
            return ToolResult.failure("No command given.")

        argv = args.get("args") or []
        if isinstance(argv, str):
            argv = shlex.split(argv)
        argv = [str(a) for a in argv]

        # Tolerate a model that puts the whole line into `command`.
        if not argv and (" " in raw_command):
            parts = shlex.split(raw_command)
            raw_command, argv = parts[0], parts[1:]

        program = Path(raw_command).name
        if program in FORBIDDEN:
            return ToolResult.failure(
                f"'{program}' is permanently blocked. Ask the user to run it themselves."
            )
        if program not in settings.allowlist:
            return ToolResult.failure(
                f"'{program}' is not in the shell allowlist. "
                f"Allowed: {', '.join(sorted(settings.allowlist))}"
            )
        for arg in argv:
            if arg in DANGEROUS_ARG_PREFIXES:
                return ToolResult.failure(
                    f"Argument '{arg}' would start a nested shell and is not allowed."
                )

        binary = shutil.which(raw_command)
        if binary is None:
            return ToolResult.failure(f"'{raw_command}' is not installed on this system.")

        cwd_raw = str(args.get("cwd") or settings.working_directory or "~")
        cwd = Path(cwd_raw).expanduser()
        if not cwd.is_dir():
            cwd = Path.home()

        timeout = max(1, min(int(args.get("timeout_seconds") or settings.timeout_seconds), 600))

        env = dict(os.environ)
        env["KEYLANE_TOOL_CALL"] = "1"

        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                *argv,
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ToolResult.failure(f"'{program}' timed out after {timeout}s.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("run_command failed")
            return ToolResult.failure(f"Could not run '{program}': {exc}")

        stdout = stdout_b.decode("utf-8", errors="replace")[:20_000]
        stderr = stderr_b.decode("utf-8", errors="replace")[:8_000]
        exit_code = proc.returncode or 0

        body = stdout.strip()
        if stderr.strip():
            body = f"{body}\n[stderr]\n{stderr.strip()}".strip()
        if not body:
            body = f"'{program}' exited with code {exit_code} and no output."

        return ToolResult(
            ok=exit_code == 0,
            output=body,
            error=None if exit_code == 0 else f"exit code {exit_code}",
            data={
                "command": program,
                "args": argv,
                "cwd": str(cwd),
                "exit_code": exit_code,
            },
        )


def shell_tools() -> list[BaseTool]:
    return [RunCommandTool()]
