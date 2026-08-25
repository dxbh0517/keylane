"""Long-term memory — a local markdown vault plus a tiny keyword index.

Designed to be openable in Obsidian (plain ``memory/*.md`` files) while still
giving the small NPU model a cheap retrieval path. Writes are gated by the
tool confirmation policy; reads are free.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ROOT

logger = logging.getLogger(__name__)

MEMORY_DIR = ROOT / "memory"
SAFE_NOTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./-]{0,80}$")
WORD_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)

SEED_NOTES = {
    "preferences.md": (
        "# Preferences\n\n"
        "What matters to this user. Update when they tell you something lasting.\n\n"
        "- (nothing recorded yet)\n"
    ),
    "people.md": (
        "# People\n\n"
        "Who matters in email and calendar decisions.\n\n"
        "- (nothing recorded yet)\n"
    ),
    "projects.md": (
        "# Projects\n\n"
        "Active work and shorthand names.\n\n"
        "- (nothing recorded yet)\n"
    ),
}


class MemoryStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or MEMORY_DIR
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_seed()

    def _ensure_seed(self) -> None:
        for name, body in SEED_NOTES.items():
            path = self.root / name
            if not path.exists():
                path.write_text(body, encoding="utf-8")

    def _resolve(self, name: str) -> Path | None:
        cleaned = (name or "").strip().lstrip("/")
        if not cleaned or ".." in cleaned or cleaned.startswith("/"):
            return None
        if not cleaned.endswith(".md"):
            cleaned = f"{cleaned}.md"
        if not SAFE_NOTE.match(cleaned.replace(".md", "").replace("/", "-")):
            # Allow nested paths like daily/2026-08-25.md
            parts = cleaned.split("/")
            if any(".." in p or not p for p in parts):
                return None
        path = (self.root / cleaned).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return None
        return path

    def list_notes(self) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*.md")):
            rel = str(path.relative_to(self.root))
            notes.append(
                {
                    "path": rel,
                    "bytes": path.stat().st_size,
                    "mtime": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        return notes

    def read(self, name: str) -> str | None:
        path = self._resolve(name)
        if path is None or not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def write(self, name: str, content: str, *, append: bool = False) -> str:
        path = self._resolve(name)
        if path is None:
            raise ValueError(f"Invalid memory note name: {name!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if append and path.exists():
                existing = path.read_text(encoding="utf-8")
                body = existing.rstrip() + "\n\n" + content.strip() + "\n"
            else:
                body = content if content.endswith("\n") else content + "\n"
            path.write_text(body, encoding="utf-8")
        return str(path.relative_to(self.root))

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        tokens = {t.lower() for t in WORD_RE.findall(query or "")}
        if not tokens:
            return []
        scored: list[tuple[int, str, str]] = []
        for path in self.root.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            lowered = text.lower()
            score = sum(lowered.count(token) for token in tokens)
            # Boost filename matches.
            name = path.stem.lower()
            score += sum(3 for token in tokens if token in name)
            if score <= 0:
                continue
            rel = str(path.relative_to(self.root))
            snippet = _snippet(text, tokens)
            scored.append((score, rel, snippet))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            {"path": rel, "score": score, "snippet": snippet}
            for score, rel, snippet in scored[:limit]
        ]

    def prompt_block(self, query: str, *, limit: int = 4, budget: int = 1200) -> str:
        hits = self.search(query, limit=limit)
        if not hits:
            # Always surface preferences + people for agent decisions.
            chunks: list[str] = []
            for name in ("preferences.md", "people.md"):
                body = self.read(name)
                if body and "(nothing recorded yet)" not in body:
                    chunks.append(f"### memory/{name}\n{body.strip()[:400]}")
            return "\n\n".join(chunks)[:budget]
        parts: list[str] = ["## Memory (retrieved)"]
        used = 0
        for hit in hits:
            block = f"### memory/{hit['path']}\n{hit['snippet']}"
            if used + len(block) > budget:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)


def _snippet(text: str, tokens: set[str], radius: int = 180) -> str:
    lowered = text.lower()
    positions = [lowered.find(token) for token in tokens if token in lowered]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return text[: radius * 2].strip()
    start = max(0, min(positions) - radius // 2)
    end = min(len(text), max(positions) + radius)
    chunk = text[start:end].strip()
    if start > 0:
        chunk = "…" + chunk
    if end < len(text):
        chunk = chunk + "…"
    return chunk


_memory: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _memory
    if _memory is None:
        _memory = MemoryStore()
    return _memory
