"""SQLite persistence for submissions. One table, WAL mode, one connection per
operation — boring on purpose. At friends-scale a queue is a table with a status
column, and SQLite's single-writer model is exactly the concurrency story we
want (the API thread and the worker thread interleave short transactions).
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  hf_id       TEXT NOT NULL,
  kind        TEXT NOT NULL DEFAULT 'auto',      -- auto|base|instruct (resolved in preflight)
  suite       TEXT NOT NULL DEFAULT 'full',      -- quick|full
  submitter   TEXT DEFAULT '',
  note        TEXT DEFAULT '',
  status      TEXT NOT NULL DEFAULT 'queued',    -- queued|preflight|waiting_gpu|waiting_lock|running|done|failed|canceled
  progress    TEXT DEFAULT '',                   -- human string: "3/10 · arc_easy"
  error       TEXT DEFAULT '',
  params      INTEGER,                           -- from HF metadata when known
  vocab       INTEGER,
  batch       INTEGER,
  need_gb     REAL,
  created_at  REAL NOT NULL,
  started_at  REAL,
  finished_at REAL,
  gpu_seconds REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status);
"""

_COLS = ["id", "hf_id", "kind", "suite", "submitter", "note", "status", "progress",
         "error", "params", "vocab", "batch", "need_gb", "created_at", "started_at",
         "finished_at", "gpu_seconds"]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    return c


def init() -> None:
    with closing(_conn()) as c:
        c.executescript(SCHEMA)
        # A worker that died mid-run leaves a phantom 'running' row; on startup no
        # run can be in flight (single process), so re-queue it. Per-task resume
        # means the re-run only repeats the task that was interrupted.
        c.execute("UPDATE submissions SET status='queued', progress='re-queued after restart' "
                  "WHERE status IN ('preflight','waiting_gpu','waiting_lock','running')")
        c.commit()


def add(hf_id: str, kind: str, suite: str, submitter: str, note: str) -> int:
    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO submissions (hf_id, kind, suite, submitter, note, created_at) "
            "VALUES (?,?,?,?,?,?)", (hf_id, kind, suite, submitter, note, time.time()))
        c.commit()
        return int(cur.lastrowid)


def claim_next() -> dict | None:
    """Atomically move the oldest queued row to 'preflight' and return it."""
    with closing(_conn()) as c:
        row = c.execute("SELECT id FROM submissions WHERE status='queued' "
                        "ORDER BY id LIMIT 1").fetchone()
        if not row:
            return None
        sid = row[0]
        hit = c.execute("UPDATE submissions SET status='preflight', started_at=? "
                        "WHERE id=? AND status='queued'", (time.time(), sid))
        c.commit()
        if hit.rowcount != 1:            # raced a cancel — try again next tick
            return None
    return get(sid)


def get(sid: int) -> dict | None:
    with closing(_conn()) as c:
        row = c.execute(f"SELECT {','.join(_COLS)} FROM submissions WHERE id=?",
                        (sid,)).fetchone()
    return dict(zip(_COLS, row)) if row else None


def update(sid: int, **fields) -> None:
    keys = ", ".join(f"{k}=?" for k in fields)
    with closing(_conn()) as c:
        c.execute(f"UPDATE submissions SET {keys} WHERE id=?", (*fields.values(), sid))
        c.commit()


def cancel(sid: int) -> bool:
    """Cancel is only honest for jobs that haven't started; a running lm_eval is
    the worker's to finish (or the operator's to kill)."""
    with closing(_conn()) as c:
        hit = c.execute("UPDATE submissions SET status='canceled', finished_at=? "
                        "WHERE id=? AND status='queued'", (time.time(), sid))
        c.commit()
        return hit.rowcount == 1


def recent(limit: int = 100) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(f"SELECT {','.join(_COLS)} FROM submissions "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(_COLS, r)) for r in rows]
