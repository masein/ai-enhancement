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
CREATE TABLE IF NOT EXISTS truns (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  project     TEXT NOT NULL DEFAULT 'default',
  submitter   TEXT DEFAULT '',
  config      TEXT DEFAULT '{}',                 -- JSON hyperparameters, shown + diffed in the UI
  status      TEXT NOT NULL DEFAULT 'running',   -- running | finished | failed
  hf_prefix   TEXT DEFAULT '',                   -- checkpoint repo/artifact prefix (benchmark join)
  created_at  REAL NOT NULL,
  updated_at  REAL NOT NULL,
  finished_at REAL
);
CREATE TABLE IF NOT EXISTS tmetrics (
  run_id INTEGER NOT NULL,
  step   INTEGER NOT NULL,
  name   TEXT NOT NULL,
  value  REAL NOT NULL,
  ts     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tmetrics ON tmetrics(run_id, name, step);
CREATE TABLE IF NOT EXISTS tevents (
  run_id INTEGER NOT NULL,
  step   INTEGER NOT NULL,
  kind   TEXT NOT NULL,                          -- 'checkpoint'
  detail TEXT DEFAULT '',                        -- the model id it was submitted as
  ts     REAL NOT NULL
);
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
  gpu_seconds REAL DEFAULT 0,
  arch        TEXT                               -- JSON: architecture/hidden/layers/heads/ctx/vocab
);
CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status);
"""

_COLS = ["id", "hf_id", "kind", "suite", "submitter", "note", "status", "progress",
         "error", "params", "vocab", "batch", "need_gb", "created_at", "started_at",
         "finished_at", "gpu_seconds", "arch"]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    return c


def init() -> None:
    with closing(_conn()) as c:
        c.executescript(SCHEMA)
        # migrations for databases created before a column existed — sqlite has no
        # ADD COLUMN IF NOT EXISTS, so probe and tolerate the duplicate error
        try:
            c.execute("ALTER TABLE submissions ADD COLUMN arch TEXT")
        except sqlite3.OperationalError:
            pass
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


# ---------------------------------------------------------------------------
# training runs — the wandb-shaped half. Volume math: logging 5 metrics every
# 10 steps for a 100k-step run is 50k rows; SQLite is comfortable well past
# millions, and inserts arrive pre-batched from the client.
# ---------------------------------------------------------------------------

_TRUN_COLS = ["id", "name", "project", "submitter", "config", "status",
              "hf_prefix", "created_at", "updated_at", "finished_at"]


def trun_create(name, project, submitter, config_json, hf_prefix) -> int:
    now = time.time()
    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO truns (name, project, submitter, config, hf_prefix, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (name, project, submitter, config_json, hf_prefix, now, now))
        c.commit()
        return int(cur.lastrowid)


def trun_get(rid: int) -> dict | None:
    with closing(_conn()) as c:
        row = c.execute(f"SELECT {','.join(_TRUN_COLS)} FROM truns WHERE id=?",
                        (rid,)).fetchone()
    return dict(zip(_TRUN_COLS, row)) if row else None


def trun_log(rid: int, points: list[tuple[int, str, float]]) -> int:
    """points: [(step, metric_name, value), ...] — one transaction per batch."""
    now = time.time()
    with closing(_conn()) as c:
        c.executemany("INSERT INTO tmetrics (run_id, step, name, value, ts) "
                      "VALUES (?,?,?,?,?)",
                      [(rid, s, n, v, now) for s, n, v in points])
        c.execute("UPDATE truns SET updated_at=? WHERE id=?", (now, rid))
        c.commit()
        return len(points)


def trun_event(rid: int, step: int, kind: str, detail: str) -> None:
    now = time.time()
    with closing(_conn()) as c:
        c.execute("INSERT INTO tevents (run_id, step, kind, detail, ts) VALUES (?,?,?,?,?)",
                  (rid, step, kind, detail, now))
        c.execute("UPDATE truns SET updated_at=? WHERE id=?", (now, rid))
        c.commit()


def trun_finish(rid: int, status: str) -> None:
    now = time.time()
    with closing(_conn()) as c:
        c.execute("UPDATE truns SET status=?, finished_at=?, updated_at=? WHERE id=?",
                  (status, now, now, rid))
        c.commit()


def trun_list(project: str | None = None, limit: int = 200) -> list[dict]:
    """Runs plus the summary the list view needs: latest step and last loss."""
    with closing(_conn()) as c:
        q = f"SELECT {','.join(_TRUN_COLS)} FROM truns"
        args: tuple = ()
        if project:
            q += " WHERE project=?"
            args = (project,)
        rows = [dict(zip(_TRUN_COLS, r))
                for r in c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit))]
        for r in rows:
            last = c.execute(
                "SELECT step, value FROM tmetrics WHERE run_id=? AND name='loss' "
                "ORDER BY step DESC LIMIT 1", (r["id"],)).fetchone()
            mx = c.execute("SELECT MAX(step) FROM tmetrics WHERE run_id=?",
                           (r["id"],)).fetchone()
            tok = c.execute(
                "SELECT value FROM tmetrics WHERE run_id=? AND name='tokens' "
                "ORDER BY step DESC LIMIT 1", (r["id"],)).fetchone()
            r["last_step"] = mx[0] if mx and mx[0] is not None else None
            r["last_loss"] = last[1] if last else None
            r["tokens"] = tok[0] if tok else None
            r["n_events"] = c.execute("SELECT COUNT(*) FROM tevents WHERE run_id=?",
                                      (r["id"],)).fetchone()[0]
    return rows


def trun_series(rid: int, max_points: int = 800) -> dict:
    """All metric series for one run, stride-downsampled to <= max_points each
    (last point always kept — it is the number people watch)."""
    with closing(_conn()) as c:
        names = [r[0] for r in c.execute(
            "SELECT DISTINCT name FROM tmetrics WHERE run_id=?", (rid,))]
        out: dict[str, list] = {}
        for n in names:
            pts = c.execute("SELECT step, value FROM tmetrics WHERE run_id=? AND name=? "
                            "ORDER BY step", (rid, n)).fetchall()
            if len(pts) > max_points:
                stride = len(pts) // max_points + 1
                pts = pts[::stride] + [pts[-1]]
            out[n] = [[s, v] for s, v in pts]
        events = [{"step": s, "kind": k, "detail": d}
                  for s, k, d in c.execute(
                      "SELECT step, kind, detail FROM tevents WHERE run_id=? "
                      "ORDER BY step", (rid,))]
    return {"metrics": out, "events": events}
