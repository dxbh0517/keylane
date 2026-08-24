"""File tools — sandboxed to the configured project roots plus the home dir.

Every path is resolved and checked against the sandbox before any I/O happens,
so a model that invents ``/etc/shadow`` gets a clean refusal rather than a read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import get_config
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    bool_prop,
    int_prop,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)

# Directories that are never readable through the file tools, even inside home.
BLOCKED_NAMES = {
    ".ssh",
    ".gnupg",
    ".password-store",
    ".aws",
    ".kube",
    ".docker",
    ".mozilla",
    ".thunderbird",
    "keyrings",
}
BLOCKED_FILENAMES = {".env", ".netrc", ".pgpass", "credentials", "id_rsa", "id_ed25519"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", "dist", "build"}


def sandbox_roots() -> list[Path]:
    config = get_config()
    roots = [Path(r).expanduser().resolve() for r in config.security.allowed_project_roots]
    home = Path.home().resolve()
    for extra in ("Documents", "Downloads", "Pictures", "Desktop", "Music", "Videos"):
        candidate = home / extra
        if candidate.is_dir():
            roots.append(candidate.resolve())
    roots.append(config.output_dir.resolve())
    # De-duplicate while preserving order.
    seen: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.append(root)
    return seen


class SandboxError(ValueError):
    pass


def resolve_in_sandbox(raw: str) -> Path:
    if not raw or not str(raw).strip():
        raise SandboxError("No path given.")
    path = Path(str(raw)).expanduser()
    resolved = path.resolve()
    roots = sandbox_roots()
    inside = any(resolved == root or root in resolved.parents for root in roots)
    if not inside:
        raise SandboxError(
            f"'{resolved}' is outside the allowed folders: "
            + ", ".join(str(r) for r in roots)
        )
    parts = set(resolved.parts)
    if parts & BLOCKED_NAMES or resolved.name in BLOCKED_FILENAMES:
        raise SandboxError(f"'{resolved.name}' is in the protected-paths list.")
    return resolved


def _describe(path: Path) -> str:
    try:
        if path.is_dir():
            return f"{path.name}/"
        size = path.stat().st_size
        if size < 1024:
            return f"{path.name}  ({size} B)"
        if size < 1024**2:
            return f"{path.name}  ({size / 1024:.1f} KB)"
        return f"{path.name}  ({size / 1024**2:.1f} MB)"
    except OSError:
        return path.name


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List the contents of a folder inside the allowed project roots."
    danger = ToolDanger.SAFE
    category = "files"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "path": string_prop("Folder to list."),
                "limit": int_prop("Maximum entries (default 100).", default=100),
            },
            required=["path"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            directory = resolve_in_sandbox(str(args.get("path") or ""))
        except SandboxError as exc:
            return ToolResult.failure(str(exc))
        if not directory.is_dir():
            return ToolResult.failure(f"Not a directory: {directory}")
        limit = max(1, min(int(args.get("limit") or 100), 500))
        entries = sorted(
            directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
        shown = [p for p in entries if not p.name.startswith(".")][:limit]
        listing = "\n".join(_describe(p) for p in shown)
        return ToolResult.success(
            listing or "(empty)",
            data={
                "path": str(directory),
                "count": len(entries),
                "entries": [p.name for p in shown],
            },
        )


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a text file inside the allowed project roots."
    danger = ToolDanger.SAFE
    category = "files"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "path": string_prop("File to read."),
                "max_chars": int_prop("Truncate at this many characters (default 8000).", default=8000),
            },
            required=["path"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            path = resolve_in_sandbox(str(args.get("path") or ""))
        except SandboxError as exc:
            return ToolResult.failure(str(exc))
        if not path.is_file():
            return ToolResult.failure(f"Not a file: {path}")
        max_chars = max(200, min(int(args.get("max_chars") or 8000), 100_000))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult.failure(f"Could not read {path}: {exc}")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n…[truncated]"
        return ToolResult.success(
            text, data={"path": str(path), "truncated": truncated}
        )


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Create or overwrite a text file inside the allowed project roots. "
        "Prefer delegating multi-file code changes to a coding worker instead."
    )
    danger = ToolDanger.SENSITIVE
    category = "files"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "path": string_prop("File to write."),
                "content": string_prop("Full file contents."),
                "append": bool_prop("Append instead of overwriting.", default=False),
            },
            required=["path", "content"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            path = resolve_in_sandbox(str(args.get("path") or ""))
        except SandboxError as exc:
            return ToolResult.failure(str(exc))
        content = str(args.get("content") or "")
        append = bool(args.get("append"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if append and path.exists():
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(content)
            else:
                path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(f"Could not write {path}: {exc}")
        verb = "Appended to" if append else "Wrote"
        return ToolResult.success(
            f"{verb} {path} ({len(content)} characters).",
            data={"path": str(path), "bytes": len(content.encode('utf-8'))},
            artifacts=[str(path)],
        )


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = (
        "Search for text inside files under a folder, or find files by name. "
        "Returns matching paths with line numbers."
    )
    danger = ToolDanger.SAFE
    category = "files"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "path": string_prop("Folder to search under."),
                "query": string_prop("Text to look for inside files."),
                "name_pattern": string_prop("Optional glob such as *.py to restrict files."),
                "limit": int_prop("Maximum matches (default 40).", default=40),
            },
            required=["path"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            root = resolve_in_sandbox(str(args.get("path") or ""))
        except SandboxError as exc:
            return ToolResult.failure(str(exc))
        if not root.is_dir():
            return ToolResult.failure(f"Not a directory: {root}")

        query = str(args.get("query") or "").strip()
        pattern = str(args.get("name_pattern") or "*").strip() or "*"
        limit = max(1, min(int(args.get("limit") or 40), 300))

        matches: list[str] = []
        scanned = 0
        for path in root.rglob(pattern):
            if len(matches) >= limit or scanned > 20_000:
                break
            if any(part in SKIP_DIRS or part in BLOCKED_NAMES for part in path.parts):
                continue
            if not path.is_file():
                continue
            scanned += 1
            if not query:
                matches.append(str(path))
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    for number, line in enumerate(fh, start=1):
                        if query in line:
                            matches.append(f"{path}:{number}: {line.strip()[:200]}")
                            break
            except OSError:
                continue

        if not matches:
            return ToolResult.success(
                "No matches found.", data={"matches": [], "scanned": scanned}
            )
        return ToolResult.success(
            "\n".join(matches), data={"matches": matches, "scanned": scanned}
        )


def file_tools() -> list[BaseTool]:
    return [ListFilesTool(), ReadFileTool(), WriteFileTool(), SearchFilesTool()]
