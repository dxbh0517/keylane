"""Multi-turn conversation sessions.

A session is a rolling transcript that survives until the user clears it.
Follow-ups from the result orb reuse the same ``session_id`` so "make it
fullscreen" knows what "it" was.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.config import ROOT

logger = logging.getLogger(__name__)

SESSIONS_DIR = ROOT / "data" / "sessions"
MAX_TURNS = 40
MAX_OBSERVATION_CHARS = 1500


class SessionTurn(BaseModel):
    role: str  # user | assistant | system
    content: str
    canvas: dict[str, Any] | None = None
    task_id: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    title: str = ""
    turns: list[SessionTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cleared: bool = False

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def prompt_history(self, *, limit: int = 12) -> str:
        """Compact transcript for the assistant context window."""
        useful = [t for t in self.turns if t.role in {"user", "assistant"}][-limit:]
        if not useful:
            return ""
        lines = ["Conversation so far:"]
        for turn in useful:
            label = "User" if turn.role == "user" else "Assistant"
            text = (turn.content or "").strip()
            if len(text) > MAX_OBSERVATION_CHARS:
                text = text[: MAX_OBSERVATION_CHARS - 1] + "…"
            lines.append(f"{label}: {text}")
            if turn.tools_used:
                lines.append(f"  (tools: {', '.join(turn.tools_used[:8])})")
        return "\n".join(lines)


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or SESSIONS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, Session] = {}

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
        return self.root / f"{safe}.json"

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            if session_id in self._cache:
                session = self._cache[session_id]
                return None if session.cleared else session
            path = self._path(session_id)
            if not path.is_file():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load session %s: %s", session_id, exc)
                return None
            self._cache[session_id] = session
            return None if session.cleared else session

    def get_or_create(self, session_id: str | None = None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        session = Session(session_id=session_id or uuid4().hex[:12])
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        session.touch()
        if len(session.turns) > MAX_TURNS:
            session.turns = session.turns[-MAX_TURNS:]
        with self._lock:
            self._cache[session.session_id] = session
            path = self._path(session.session_id)
            path.write_text(
                session.model_dump_json(indent=2),
                encoding="utf-8",
            )

    def append_user(self, session: Session, message: str, *, task_id: str | None = None) -> None:
        if not session.title:
            session.title = (message or "").strip()[:80]
        session.turns.append(
            SessionTurn(role="user", content=message, task_id=task_id)
        )
        self.save(session)

    def append_assistant(
        self,
        session: Session,
        content: str,
        *,
        canvas: dict[str, Any] | None = None,
        task_id: str | None = None,
        tools_used: list[str] | None = None,
    ) -> None:
        session.turns.append(
            SessionTurn(
                role="assistant",
                content=content,
                canvas=canvas,
                task_id=task_id,
                tools_used=list(tools_used or []),
            )
        )
        self.save(session)

    def clear(self, session_id: str) -> None:
        session = self.get(session_id) or Session(session_id=session_id)
        session.cleared = True
        session.turns = []
        self.save(session)
        path = self._path(session_id)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        with self._lock:
            self._cache.pop(session_id, None)

    def list_recent(self, *, limit: int = 20) -> list[Session]:
        sessions: list[Session] = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session.model_validate(data)
            except Exception:  # noqa: BLE001
                continue
            if session.cleared:
                continue
            sessions.append(session)
            if len(sessions) >= limit:
                break
        return sessions


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
