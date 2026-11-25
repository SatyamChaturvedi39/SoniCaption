"""
SQLite Persistent Session, Call Log, Ambient Watch Alert, and Reminder Store.

Backs the product database under `backend/data/soniccaption.db` so every uploaded
recording, voice agent conversation, live watch alert, and extracted reminder
persists across restarts and can be searched or updated from the React UI.
"""

from __future__ import annotations
from contextlib import closing
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "soniccaption.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or os.environ.get("SONICCAPTION_DB_PATH") or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            source_type TEXT NOT NULL,
            duration REAL NOT NULL,
            speech_source TEXT NOT NULL,
            vad_speech_ratio REAL NOT NULL DEFAULT 0.0,
            snr_before_db REAL NOT NULL DEFAULT 0.0,
            snr_after_db REAL NOT NULL DEFAULT 0.0,
            denoised INTEGER NOT NULL DEFAULT 0,
            summary TEXT NOT NULL DEFAULT '',
            agent_reply TEXT NOT NULL DEFAULT '',
            timeline_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
            title TEXT NOT NULL,
            due_time TEXT NOT NULL DEFAULT 'Soon',
            priority TEXT NOT NULL DEFAULT 'medium',
            trigger_source TEXT NOT NULL DEFAULT 'speech_action',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            event_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL DEFAULT 'medium',
            snippet_text TEXT NOT NULL DEFAULT ''
        );
        """
    )
    conn.commit()


def _db_conn(db_path: Path | str | None = None):
    return closing(_connect(db_path))


def save_session(
    *,
    filename: str,
    source_type: str,
    duration: float,
    speech_source: str,
    vad_speech_ratio: float,
    snr_before_db: float,
    snr_after_db: float,
    denoised: bool,
    summary: str,
    agent_reply: str = "",
    timeline: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    db_path: Path | str | None = None,
) -> int:
    with _db_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions (
                created_at, filename, source_type, duration, speech_source,
                vad_speech_ratio, snr_before_db, snr_after_db, denoised,
                summary, agent_reply, timeline_json, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                filename,
                source_type,
                float(duration),
                speech_source,
                float(vad_speech_ratio),
                float(snr_before_db),
                float(snr_after_db),
                1 if denoised else 0,
                summary,
                agent_reply,
                json.dumps(timeline),
                json.dumps(metrics),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_sessions(search: str = "", limit: int = 50, db_path: Path | str | None = None) -> List[Dict[str, Any]]:
    q = f"%{search.strip().lower()}%"
    with _db_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM sessions
            WHERE lower(filename) LIKE ? OR lower(summary) LIKE ? OR lower(timeline_json) LIKE ?
            ORDER BY id DESC LIMIT ?
            """,
            (q, q, q, limit),
        ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["denoised"] = bool(item["denoised"])
        item["timeline"] = json.loads(item.pop("timeline_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        out.append(item)
    return out


def delete_session(session_id: int, db_path: Path | str | None = None) -> bool:
    with _db_conn(db_path) as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cur.rowcount > 0


def add_reminder(
    *,
    title: str,
    due_time: str = "Today",
    priority: str = "medium",
    trigger_source: str = "speech_action",
    session_id: Optional[int] = None,
    db_path: Path | str | None = None,
) -> Dict[str, Any]:
    now = _utc_now()
    with _db_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO reminders (session_id, title, due_time, priority, trigger_source, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (session_id, title.strip(), due_time.strip() or "Soon", priority, trigger_source, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM reminders WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_reminders(status: Optional[str] = None, db_path: Path | str | None = None) -> List[Dict[str, Any]]:
    with _db_conn(db_path) as conn:
        if status:
            rows = conn.execute("SELECT * FROM reminders WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reminders ORDER BY CASE WHEN status='pending' THEN 0 ELSE 1 END, id DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def update_reminder_status(reminder_id: int, status: str, db_path: Path | str | None = None) -> Optional[Dict[str, Any]]:
    if status not in ("pending", "completed"):
        raise ValueError("status must be 'pending' or 'completed'")
    with _db_conn(db_path) as conn:
        conn.execute("UPDATE reminders SET status = ? WHERE id = ?", (status, reminder_id))
        conn.commit()
        row = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
    return dict(row) if row else None


def delete_reminder(reminder_id: int, db_path: Path | str | None = None) -> bool:
    with _db_conn(db_path) as conn:
        cur = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        conn.commit()
        return cur.rowcount > 0


def log_alert(
    *,
    event_label: str,
    confidence: float,
    severity: str,
    snippet_text: str,
    db_path: Path | str | None = None,
) -> Dict[str, Any]:
    now = _utc_now()
    with _db_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO alerts (created_at, event_label, confidence, severity, snippet_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (now, event_label, float(confidence), severity, snippet_text),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_alerts(limit: int = 40, db_path: Path | str | None = None) -> List[Dict[str, Any]]:
    with _db_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
