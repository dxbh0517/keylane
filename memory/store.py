"""SQLite session store with FTS5 cross-session search."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daemon.paths import DB_PATH, MEMORY_MD, SKILLS_DIR, USER_MD, ensure_data_dirs


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, path: Path = DB_PATH) -> None:
        ensure_data_dirs()
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    session_id, role, content, content='messages', content_rowid='id'
                );
                CREATE TABLE IF NOT EXISTS todos (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    done INTEGER DEFAULT 0,
                    created_at TEXT,
                    due_at TEXT
                );
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    kind TEXT,
                    schedule TEXT,
                    prompt TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT,
                    last_run TEXT
                );
                CREATE TABLE IF NOT EXISTS background_jobs (
                    id TEXT PRIMARY KEY,
                    prompt TEXT,
                    status TEXT,
                    result TEXT,
                    created_at TEXT,
                    finished_at TEXT
                );
                """
            )

    def new_session(self, title: str = "New chat") -> str:
        sid = str(uuid.uuid4())
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (sid, title, now, now),
            )
        return sid

    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = _utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "INSERT INTO messages_fts (rowid, session_id, role, content) VALUES (?,?,?,?)",
                (cur.lastrowid, session_id, role, content),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (now, session_id),
            )

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def search_sessions(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, snippet(messages_fts, 2, '[', ']', '…', 20) AS snippet
                FROM messages_fts
                WHERE messages_fts MATCH ?
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [{"session_id": r["session_id"], "snippet": r["snippet"]} for r in rows]

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]


def read_user_md() -> str:
    ensure_data_dirs()
    return USER_MD.read_text(encoding="utf-8")


def read_memory_md() -> str:
    ensure_data_dirs()
    return MEMORY_MD.read_text(encoding="utf-8")


def write_memory_md(content: str) -> None:
    ensure_data_dirs()
    MEMORY_MD.write_text(content, encoding="utf-8")


def write_user_md(content: str) -> None:
    ensure_data_dirs()
    USER_MD.write_text(content, encoding="utf-8")


def list_skills() -> list[dict[str, str]]:
    ensure_data_dirs()
    out: list[dict[str, str]] = []
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        name = path.parent.name
        desc = ""
        if text.startswith("---"):
            end = text.find("---", 3)
            if end > 0:
                for line in text[3:end].splitlines():
                    if line.strip().startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip('"')
        out.append({"id": name, "name": name, "description": desc})
    return out


def load_skill(skill_id: str) -> str:
    path = SKILLS_DIR / skill_id / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(skill_id)
    return path.read_text(encoding="utf-8")


def save_skill(skill_id: str, content: str) -> None:
    ensure_data_dirs()
    folder = SKILLS_DIR / skill_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(content, encoding="utf-8")


_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
