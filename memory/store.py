"""SQLite session store with FTS5 cross-session search."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daemon.paths import DB_PATH, MEMORY_MD, SKILLS_DIR, USER_MD, ensure_data_dirs


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


TOOL_RESULT_OPEN = "<tool_result"
_TOOL_CALL_STUB = '{"tool_call"'


def balance_tool_pairs(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim a history window so no tool call is separated from its result.

    A fixed-size window can cut anywhere. A tool result whose originating
    assistant call fell off the front reads to the model as the answer to a
    question it never asked; a trailing call with no result invites the model to
    answer as though it already had one. Both halves are dropped.
    """
    start = 0
    while start < len(messages):
        content = messages[start].get("content", "").lstrip()
        if content.startswith(TOOL_RESULT_OPEN):
            start += 1
            continue
        break

    end = len(messages)
    while end > start:
        last = messages[end - 1]
        content = last.get("content", "").lstrip()
        if last.get("role") == "assistant" and content.startswith(_TOOL_CALL_STUB):
            end -= 1
            continue
        break

    return messages[start:end]


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
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT,
                    text TEXT,
                    norm TEXT,
                    tags TEXT,
                    pinned INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    last_used_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS memories_norm ON memories(norm);
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text, tags, content='memories', content_rowid='rowid'
                );
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    objective TEXT,
                    phase TEXT,
                    revision INTEGER DEFAULT 1,
                    rounds INTEGER DEFAULT 0,
                    max_rounds INTEGER,
                    blocked_reason TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS inbox (
                    id TEXT PRIMARY KEY,
                    kind TEXT,
                    title TEXT,
                    body TEXT,
                    source TEXT,
                    read INTEGER DEFAULT 0,
                    created_at TEXT
                );
                """
            )
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns introduced after a database was first created."""
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(scheduled_tasks)")}
        for name, decl in (("run_at", "TEXT"), ("title", "TEXT"), ("next_run", "TEXT")):
            if name not in cols:
                conn.execute(f"ALTER TABLE scheduled_tasks ADD COLUMN {name} {decl}")

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
        window = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        return balance_tool_pairs(window)

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


def save_skill(skill_id: str, content: str) -> None:
    ensure_data_dirs()
    folder = SKILLS_DIR / skill_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(content, encoding="utf-8")


# ── Structured memory ────────────────────────────────────────────────────
#
# MEMORY.md / USER.md remain the human-editable surface. Individual facts
# live here as rows so the agent can add or drop one without rewriting a
# whole file — a small model handed "overwrite this document" loses the rest
# of it sooner or later.

MEMORY_KINDS = ("user", "preference", "fact", "project", "contact")


def _normalize_memory(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def save_memory(text: str, kind: str = "fact", tags: str = "", pinned: bool = False) -> dict[str, Any]:
    """Insert one fact, or refresh it if an equivalent one already exists."""
    text = text.strip()
    if not text:
        raise ValueError("memory text is empty")
    if kind not in MEMORY_KINDS:
        kind = "fact"
    norm = _normalize_memory(text)
    now = _utcnow()
    store = get_store()
    with store._connect() as conn:  # noqa: SLF001
        row = conn.execute("SELECT id FROM memories WHERE norm=?", (norm,)).fetchone()
        if row:
            conn.execute(
                "UPDATE memories SET text=?, kind=?, tags=?, pinned=?, updated_at=? WHERE id=?",
                (text, kind, tags, int(pinned), now, row["id"]),
            )
            mid = row["id"]
            conn.execute("DELETE FROM memories_fts WHERE rowid=(SELECT rowid FROM memories WHERE id=?)", (mid,))
        else:
            mid = str(uuid.uuid4())[:8]
            conn.execute(
                "INSERT INTO memories (id, kind, text, norm, tags, pinned, created_at, updated_at, last_used_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (mid, kind, text, norm, tags, int(pinned), now, now, now),
            )
        rowid = conn.execute("SELECT rowid FROM memories WHERE id=?", (mid,)).fetchone()["rowid"]
        conn.execute(
            "INSERT INTO memories_fts (rowid, text, tags) VALUES (?,?,?)",
            (rowid, text, tags),
        )
    return {"id": mid, "kind": kind, "text": text}


def search_memories(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """FTS lookup, with a LIKE fallback for queries FTS5 refuses to parse."""
    store = get_store()
    with store._connect() as conn:
        try:
            rows = conn.execute(
                """
                SELECT m.id, m.kind, m.text, m.tags
                FROM memories_fts f JOIN memories m ON m.rowid = f.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(
                "SELECT id, kind, text, tags FROM memories WHERE text LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        if rows:
            conn.execute(
                f"UPDATE memories SET last_used_at=? WHERE id IN ({','.join('?' * len(rows))})",
                (_utcnow(), *[r["id"] for r in rows]),
            )
    return [{"id": r["id"], "kind": r["kind"], "text": r["text"], "tags": r["tags"]} for r in rows]


def list_memories(kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    sql = "SELECT id, kind, text, tags, pinned, updated_at FROM memories"
    params: list[Any] = []
    if kind:
        sql += " WHERE kind=?"
        params.append(kind)
    sql += " ORDER BY pinned DESC, updated_at DESC LIMIT ?"
    params.append(limit)
    with get_store()._connect() as conn:  # noqa: SLF001
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def forget_memory(memory_id: str) -> bool:
    with get_store()._connect() as conn:  # noqa: SLF001
        row = conn.execute("SELECT rowid FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM memories_fts WHERE rowid=?", (row["rowid"],))
        conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    return True


def memory_digest(limit: int = 20, max_chars: int = 1200) -> str:
    """Compact recall block for the system prompt: pinned first, then recent."""
    rows = list_memories(limit=limit)
    if not rows:
        return "(nothing remembered yet)"
    lines: list[str] = []
    total = 0
    for row in rows:
        line = f"- [{row['kind']}] {row['text']}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


# ── Inbox (background results the user has not seen yet) ─────────────────


def push_inbox(title: str, body: str, kind: str = "note", source: str = "") -> str:
    item_id = str(uuid.uuid4())[:8]
    with get_store()._connect() as conn:  # noqa: SLF001
        conn.execute(
            "INSERT INTO inbox (id, kind, title, body, source, read, created_at) VALUES (?,?,?,?,?,0,?)",
            (item_id, kind, title, body, source, _utcnow()),
        )
    return item_id


def list_inbox(unread_only: bool = True, limit: int = 20) -> list[dict[str, Any]]:
    sql = "SELECT id, kind, title, body, source, read, created_at FROM inbox"
    if unread_only:
        sql += " WHERE read=0"
    sql += " ORDER BY created_at DESC LIMIT ?"
    with get_store()._connect() as conn:  # noqa: SLF001
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def mark_inbox_read(item_id: str | None = None) -> int:
    with get_store()._connect() as conn:  # noqa: SLF001
        if item_id:
            cur = conn.execute("UPDATE inbox SET read=1 WHERE id=?", (item_id,))
        else:
            cur = conn.execute("UPDATE inbox SET read=1 WHERE read=0")
        return cur.rowcount


# ── Scheduled tasks ──────────────────────────────────────────────────────


def upsert_scheduled_task(
    task_id: str,
    kind: str,
    prompt: str,
    *,
    schedule: str = "",
    run_at: str = "",
    title: str = "",
) -> None:
    with get_store()._connect() as conn:  # noqa: SLF001
        conn.execute(
            "INSERT OR REPLACE INTO scheduled_tasks"
            " (id, kind, schedule, prompt, enabled, created_at, run_at, title)"
            " VALUES (?,?,?,?,1,?,?,?)",
            (task_id, kind, schedule, prompt, _utcnow(), run_at, title or prompt[:60]),
        )


def list_scheduled_tasks(enabled_only: bool = True) -> list[dict[str, Any]]:
    sql = "SELECT id, kind, schedule, prompt, enabled, created_at, last_run, run_at, title FROM scheduled_tasks"
    if enabled_only:
        sql += " WHERE enabled=1"
    sql += " ORDER BY created_at DESC"
    with get_store()._connect() as conn:  # noqa: SLF001
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def mark_task_run(task_id: str) -> None:
    with get_store()._connect() as conn:  # noqa: SLF001
        conn.execute("UPDATE scheduled_tasks SET last_run=? WHERE id=?", (_utcnow(), task_id))


def disable_scheduled_task(task_id: str) -> bool:
    with get_store()._connect() as conn:  # noqa: SLF001
        cur = conn.execute("UPDATE scheduled_tasks SET enabled=0 WHERE id=?", (task_id,))
        return cur.rowcount > 0


_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
