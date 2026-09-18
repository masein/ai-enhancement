"""SQLite persistence for submissions. One table, WAL mode, one connection per
operation — boring on purpose. At friends-scale a queue is a table with a status
column, and SQLite's single-writer model is exactly the concurrency story we
want (the API thread and the worker thread interleave short transactions).
"""

from __future__ import annotations

import json
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
  finished_at REAL,
  n_updates   INTEGER NOT NULL DEFAULT 0          -- log/event batches received (cadence estimate)
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
  arch        TEXT,                              -- JSON: architecture/hidden/layers/heads/ctx/vocab
  allow_remote_code INTEGER NOT NULL DEFAULT 0,  -- submitter opted in to executing the upload's code
  load_missing TEXT DEFAULT ''                   -- JSON: checkpoint keys transformers had to invent
);
CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status);
-- find-the-gap: an LLM proposes a skill spec from diagnose-half failures, a
-- person approves it, a generator that saw only the spec makes data
CREATE TABLE IF NOT EXISTS proposals (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  model         TEXT NOT NULL,
  task          TEXT NOT NULL,
  category      TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending',  -- pending|proposed|approved|rejected|failed
  spec_text     TEXT DEFAULT '',                  -- the LLM's
  edited_text   TEXT DEFAULT '',                  -- the human's, when edited
  evidence      TEXT DEFAULT '{}',                -- JSON: counts, patterns, the examples shown
  proposer      TEXT DEFAULT '',                  -- LLM id (provider/model)
  requested_by  TEXT DEFAULT '',                  -- who clicked propose
  approver      TEXT DEFAULT '',                  -- who approved or rejected (free text; the tailnet is the auth)
  reject_reason TEXT DEFAULT '',
  batch_id      TEXT DEFAULT '',
  prompt_sha    TEXT DEFAULT '',
  error         TEXT DEFAULT '',
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL,
  approved_at   REAL
);
CREATE TABLE IF NOT EXISTS datasets (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  proposal_id   INTEGER NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending',  -- pending|ready|rejected|failed
  fmt           TEXT NOT NULL DEFAULT 'mc',       -- mc|free
  count         INTEGER NOT NULL,
  requester     TEXT DEFAULT '',
  batch_id      TEXT DEFAULT '',
  provenance    TEXT DEFAULT '{}',                -- JSON, the same record as provenance.json
  error         TEXT DEFAULT '',
  created_at    REAL NOT NULL,
  finished_at   REAL
);
-- who accepted or rejected each drafted exam question, and when. The bank
-- file carries the same name; this is the queryable record of the decisions
CREATE TABLE IF NOT EXISTS exam_curation (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  cid         TEXT NOT NULL,
  topic       TEXT NOT NULL,
  qid         TEXT DEFAULT '',                  -- the accepted question's identity
  decision    TEXT NOT NULL,                    -- accepted|rejected
  approver    TEXT NOT NULL,
  edited      INTEGER NOT NULL DEFAULT 0,
  reason      TEXT DEFAULT '',
  decided_at  REAL NOT NULL
);
-- every batch id, persisted before anything else happens: a restart resumes
-- polling instead of re-submitting
CREATE TABLE IF NOT EXISTS llm_batches (
  batch_id    TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,                      -- proposal|generation
  ref_id      INTEGER NOT NULL,
  n_items     INTEGER NOT NULL,
  provider    TEXT NOT NULL,
  model       TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'submitted',  -- submitted|done|failed
  error       TEXT DEFAULT '',
  created_at  REAL NOT NULL,
  finished_at REAL
);
"""

_COLS = ["id", "hf_id", "kind", "suite", "submitter", "note", "status", "progress",
         "error", "params", "vocab", "batch", "need_gb", "created_at", "started_at",
         "finished_at", "gpu_seconds", "arch", "allow_remote_code", "load_missing"]


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
        for stmt in ("ALTER TABLE submissions ADD COLUMN arch TEXT",
                     "ALTER TABLE truns ADD COLUMN n_updates INTEGER NOT NULL DEFAULT 0",
                     "ALTER TABLE submissions ADD COLUMN allow_remote_code "
                     "INTEGER NOT NULL DEFAULT 0",
                     "ALTER TABLE submissions ADD COLUMN load_missing TEXT DEFAULT ''",
                     "ALTER TABLE truns ADD COLUMN datasets TEXT DEFAULT '[]'",
                     "ALTER TABLE truns ADD COLUMN parent TEXT DEFAULT ''"):
            try:
                c.execute(stmt)
            except sqlite3.OperationalError:
                pass
        # A worker that died mid-run leaves a phantom 'running' row; on startup no
        # run can be in flight (single process), so re-queue it. Per-task resume
        # means the re-run only repeats the task that was interrupted.
        c.execute("UPDATE submissions SET status='queued', progress='re-queued after restart' "
                  "WHERE status IN ('preflight','waiting_gpu','waiting_lock','running')")
        c.commit()


def add(hf_id: str, kind: str, suite: str, submitter: str, note: str,
        allow_remote_code: bool = False) -> int:
    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO submissions (hf_id, kind, suite, submitter, note, created_at, "
            "allow_remote_code) VALUES (?,?,?,?,?,?,?)",
            (hf_id, kind, suite, submitter, note, time.time(), int(allow_remote_code)))
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
              "hf_prefix", "created_at", "updated_at", "finished_at", "datasets", "parent"]


def trun_create(name, project, submitter, config_json, hf_prefix,
                datasets: list[int] | None = None, parent: str = "") -> int:
    now = time.time()
    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO truns (name, project, submitter, config, hf_prefix, "
            "created_at, updated_at, datasets, parent) VALUES (?,?,?,?,?,?,?,?,?)",
            (name, project, submitter, config_json, hf_prefix, now, now,
             json.dumps(sorted(set(int(d) for d in (datasets or [])))), parent))
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
        c.execute("UPDATE truns SET updated_at=?, n_updates=n_updates+1 WHERE id=?", (now, rid))
        c.commit()
        return len(points)


def trun_event(rid: int, step: int, kind: str, detail: str) -> None:
    now = time.time()
    with closing(_conn()) as c:
        c.execute("INSERT INTO tevents (run_id, step, kind, detail, ts) VALUES (?,?,?,?,?)",
                  (rid, step, kind, detail, now))
        c.execute("UPDATE truns SET updated_at=?, n_updates=n_updates+1 WHERE id=?", (now, rid))
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
            # how often this run normally reports, in seconds — the dashboard
            # calls a run idle only after it has been silent for several times
            # its own cadence, so a slow-stepping run is not flagged for logging
            # every 20 minutes while a fast one is not left "running" for hours
            n_up = c.execute("SELECT n_updates FROM truns WHERE id=?", (r["id"],)).fetchone()[0]
            r["cadence_s"] = ((r["updated_at"] - r["created_at"]) / n_up
                              if n_up and n_up >= 3 else None)
    return rows


def trun_series(rid: int, max_points: int = 400) -> dict:
    """All metric series for one run, stride-downsampled to <= max_points each
    (last point always kept — it is the number people watch).

    One query per metric name, which measured faster than a single grouped scan:
    idx_tmetrics(run_id, name, step) makes each one an index range read that
    fetchall pulls at C speed, where the grouped version pays a Python loop per
    row — on a 100-metric, 945-step run that was 285 ms against 194 ms. Left as
    it is on the strength of the measurement, not the shape of the code."""
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


# ---------------------------------------------------------------------------
# find-the-gap: proposals, datasets, batches — and the taint join
# ---------------------------------------------------------------------------

_PROP_COLS = ["id", "model", "task", "category", "status", "spec_text", "edited_text",
              "evidence", "proposer", "requested_by", "approver", "reject_reason",
              "batch_id", "prompt_sha", "error", "created_at", "updated_at", "approved_at"]
_DS_COLS = ["id", "proposal_id", "status", "fmt", "count", "requester", "batch_id",
            "provenance", "error", "created_at", "finished_at"]
_BATCH_COLS = ["batch_id", "kind", "ref_id", "n_items", "provider", "model", "status",
               "error", "created_at", "finished_at"]


def proposal_create(model: str, task: str, category: str, requested_by: str,
                    evidence: dict) -> int:
    now = time.time()
    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO proposals (model, task, category, requested_by, evidence, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (model, task, category, requested_by, json.dumps(evidence), now, now))
        c.commit()
        return int(cur.lastrowid)


def proposal_get(pid: int) -> dict | None:
    with closing(_conn()) as c:
        row = c.execute(f"SELECT {','.join(_PROP_COLS)} FROM proposals WHERE id=?",
                        (pid,)).fetchone()
    return dict(zip(_PROP_COLS, row)) if row else None


def proposal_update(pid: int, **fields) -> None:
    fields["updated_at"] = time.time()
    keys = ", ".join(f"{k}=?" for k in fields)
    with closing(_conn()) as c:
        c.execute(f"UPDATE proposals SET {keys} WHERE id=?", (*fields.values(), pid))
        c.commit()


def proposal_list(status: str | None = None, limit: int = 200) -> list[dict]:
    with closing(_conn()) as c:
        q = f"SELECT {','.join(_PROP_COLS)} FROM proposals"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        rows = c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
    return [dict(zip(_PROP_COLS, r)) for r in rows]


def proposal_active(model: str, task: str, category: str) -> dict | None:
    """A proposal for the same cell that is still in flight or awaiting review."""
    with closing(_conn()) as c:
        row = c.execute(
            f"SELECT {','.join(_PROP_COLS)} FROM proposals WHERE model=? AND task=? AND "
            "category=? AND status IN ('pending','proposed') ORDER BY id DESC LIMIT 1",
            (model, task, category)).fetchone()
    return dict(zip(_PROP_COLS, row)) if row else None


def dataset_create(proposal_id: int, fmt: str, count: int, requester: str,
                   provenance: dict) -> int:
    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO datasets (proposal_id, fmt, count, requester, provenance, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (proposal_id, fmt, count, requester, json.dumps(provenance), time.time()))
        c.commit()
        return int(cur.lastrowid)


def dataset_get(did: int) -> dict | None:
    with closing(_conn()) as c:
        row = c.execute(f"SELECT {','.join(_DS_COLS)} FROM datasets WHERE id=?",
                        (did,)).fetchone()
    return dict(zip(_DS_COLS, row)) if row else None


def dataset_update(did: int, **fields) -> None:
    keys = ", ".join(f"{k}=?" for k in fields)
    with closing(_conn()) as c:
        c.execute(f"UPDATE datasets SET {keys} WHERE id=?", (*fields.values(), did))
        c.commit()


def dataset_list(limit: int = 200) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(f"SELECT {','.join(_DS_COLS)} FROM datasets ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
    return [dict(zip(_DS_COLS, r)) for r in rows]


def batch_add(batch_id: str, kind: str, ref_id: int, n_items: int, provider: str,
              model: str) -> None:
    with closing(_conn()) as c:
        c.execute("INSERT INTO llm_batches (batch_id, kind, ref_id, n_items, provider, model, "
                  "created_at) VALUES (?,?,?,?,?,?,?)",
                  (batch_id, kind, ref_id, n_items, provider, model, time.time()))
        c.commit()


def batch_finish(batch_id: str, status: str, error: str) -> None:
    with closing(_conn()) as c:
        c.execute("UPDATE llm_batches SET status=?, error=?, finished_at=? WHERE batch_id=?",
                  (status, error, time.time(), batch_id))
        c.commit()


def batches_pending() -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(f"SELECT {','.join(_BATCH_COLS)} FROM llm_batches "
                         "WHERE status='submitted' ORDER BY created_at").fetchall()
    return [dict(zip(_BATCH_COLS, r)) for r in rows]


def batches_list(limit: int = 200) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(f"SELECT {','.join(_BATCH_COLS)} FROM llm_batches "
                         "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(_BATCH_COLS, r)) for r in rows]


def llm_items_today() -> int:
    """Batch items submitted since local midnight — the number the spend
    guard compares against LLM_DAILY_ITEM_CAP."""
    lt = time.localtime()
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    with closing(_conn()) as c:
        row = c.execute("SELECT COALESCE(SUM(n_items), 0) FROM llm_batches WHERE created_at>=?",
                        (midnight,)).fetchone()
    return int(row[0] or 0)


def taint_links() -> list[dict]:
    """Every training run that consumed a generated dataset, with the model
    ids its checkpoints were submitted as and its hf_prefix. The join that
    turns 'this run used dataset 3' into 'these checkpoints are tainted'."""
    out = []
    with closing(_conn()) as c:
        rows = c.execute("SELECT id, hf_prefix, datasets, parent, config FROM truns "
                         "WHERE datasets IS NOT NULL AND datasets != '[]'").fetchall()
        for rid, prefix, ds, parent, cfg in rows:
            # the "before" of the comparison: the model the run started from,
            # recorded explicitly, else the base_model its config names
            if not parent:
                try:
                    parent = str(json.loads(cfg or "{}").get("base_model") or "")
                except (ValueError, TypeError):
                    parent = ""
            try:
                ids = [int(x) for x in json.loads(ds or "[]")]
            except (ValueError, TypeError):
                ids = []
            if not ids:
                continue
            ckpts = [r[0] for r in c.execute(
                "SELECT DISTINCT detail FROM tevents WHERE run_id=? AND kind='checkpoint' "
                "AND detail != ''", (rid,)).fetchall()]
            tasks = []
            for did in ids:
                row = c.execute("SELECT provenance FROM datasets WHERE id=?", (did,)).fetchone()
                if row:
                    try:
                        t = json.loads(row[0] or "{}").get("task")
                    except (ValueError, TypeError):
                        t = None
                    if t:
                        tasks.append(t)
            out.append({"run_id": rid, "hf_prefix": prefix or "", "datasets": ids,
                        "checkpoints": ckpts, "tasks": sorted(set(tasks)),
                        "parent": parent or ""})
    return out


def taint_stamp() -> tuple:
    """Changes whenever the taint join could: a cache key for the payload."""
    with closing(_conn()) as c:
        a = c.execute("SELECT COUNT(*), COALESCE(MAX(updated_at), 0) FROM truns "
                      "WHERE datasets IS NOT NULL AND datasets != '[]'").fetchone()
        b = c.execute("SELECT COUNT(*) FROM tevents").fetchone()
        d = c.execute("SELECT COUNT(*), COALESCE(MAX(finished_at), 0) FROM datasets").fetchone()
    return (a[0], a[1], b[0], d[0], d[1])


# ---------------------------------------------------------------------------
# exam curation — the human step of writing the exam
# ---------------------------------------------------------------------------

_CUR_COLS = ["id", "cid", "topic", "qid", "decision", "approver", "edited", "reason", "decided_at"]


def curation_add(cid: str, topic: str, qid: str, decision: str, approver: str,
                 edited: bool = False, reason: str = "") -> int:
    with closing(_conn()) as c:
        cur = c.execute("INSERT INTO exam_curation (cid, topic, qid, decision, approver, edited, "
                        "reason, decided_at) VALUES (?,?,?,?,?,?,?,?)",
                        (cid, topic, qid, decision, approver, int(edited), reason, time.time()))
        c.commit()
        return int(cur.lastrowid)


def curation_list(limit: int = 500) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(f"SELECT {','.join(_CUR_COLS)} FROM exam_curation "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(_CUR_COLS, r)) for r in rows]
