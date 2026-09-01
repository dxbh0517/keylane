"""Oversized tool results, kept rather than destroyed.

Truncating a result to 2800 characters and appending an ellipsis throws the
rest away: the model is told there was more and given no way to reach it. A
spill writes the full text to a file, hands back a head/tail preview, and says
exactly how to read the remainder — which turns "the answer was cut off" into
one more tool call.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from daemon.paths import DATA

SPILL_DIR = DATA / "spill"

# Below this a result rides inline; above it, the preview plus a locator is
# smaller than the content and more useful than a truncation.
MAX_INLINE_CHARS = 2800
HEAD_CHARS = 1200
TAIL_CHARS = 600

_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."


@dataclass(frozen=True)
class SpillRef:
    """A saved artifact: where it is, how big, and how to read it."""

    locator: str
    chars: int
    retrieval_hint: str


def _safe_name(name: str) -> str:
    cleaned = "".join(c if c in _SAFE else "-" for c in name)[:40]
    return cleaned.strip("-") or "result"


class SpillStore:
    """Session-scoped files on the local disk."""

    def __init__(self, root: Path = SPILL_DIR) -> None:
        self.root = root
        self._lock = threading.Lock()

    def _session_dir(self, session_id: str) -> Path:
        # Hashed so a session id never becomes a path component.
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        return self.root / f"session-{digest}"

    def save_text(self, *, session_id: str, tool_name: str, content: str) -> SpillRef:
        folder = self._session_dir(session_id)
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            folder.chmod(0o700)
            path = folder / f"{uuid.uuid4().hex[:8]}-{_safe_name(tool_name)}.txt"
            path.write_text(content, encoding="utf-8")
            path.chmod(0o600)
        return SpillRef(
            locator=str(path),
            chars=len(content),
            retrieval_hint=(
                f"The full result is saved at {path}. "
                f"Use `shell` with `grep` to search it or `head`/`tail` to read part of it."
            ),
        )


def preview(content: str) -> str:
    """The head and tail of an oversized result, with the gap declared."""
    if len(content) <= HEAD_CHARS + TAIL_CHARS:
        return content
    omitted = len(content) - HEAD_CHARS - TAIL_CHARS
    return (
        f"{content[:HEAD_CHARS]}\n\n"
        f"… [{omitted} characters omitted] …\n\n"
        f"{content[-TAIL_CHARS:]}"
    )


_store: SpillStore | None = None
_store_lock = threading.Lock()


def get_spill_store() -> SpillStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SpillStore()
    return _store
