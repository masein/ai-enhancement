"""The HTTP layer: a submit-and-watch API plus the live dashboard.

    uvicorn service.app:app --host <tailscale-ip> --port 8899

Serving GET / reuses the dashboard from scripts/report_lm_eval.py with the data
slot left empty — the page then fetches /api/results and polls /api/submissions,
so the exact same charts run live here and frozen in the emailed report file.
Bind to the Tailscale IP: the tailnet is the auth boundary; nothing here should
face the open internet.
"""

from __future__ import annotations

import json
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import config, db, worker

# the report module is the single source of truth for parsing and for the page
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import report_lm_eval as report  # noqa: E402

_HF_ID_RE = re.compile(r"^[\w.\-]{1,96}/[\w.\-]{1,96}$")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init()
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="benchmark service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# results payload, cached against the tree's shape
# ---------------------------------------------------------------------------

_cache: dict = {"key": None, "payload": None, "at": 0.0}


def _tree_key() -> tuple:
    if not config.OUT_DIR.is_dir():
        return (0, 0.0)
    files = list(config.OUT_DIR.rglob("results*.json"))
    return (len(files), max((f.stat().st_mtime for f in files), default=0.0))


def results_payload() -> dict:
    now = time.time()
    if _cache["payload"] is not None and now - _cache["at"] < 5:
        return _cache["payload"]
    key = _tree_key()
    if key != _cache["key"] or _cache["payload"] is None:
        runs = report.load_results(config.OUT_DIR) if config.OUT_DIR.is_dir() else []
        payload = report.build_payload(report.merge_runs(runs), config.TITLE,
                                       source=str(config.OUT_DIR))
        payload["live"] = True
        _cache.update(key=key, payload=payload)
    _cache["at"] = now
    return _cache["payload"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SubmissionIn(BaseModel):
    hf_id: str
    kind: str = "auto"
    suite: str = "full"
    submitter: str = ""
    note: str = ""


ACTIVE = ("queued", "preflight", "waiting_gpu", "waiting_lock", "running")


@app.post("/api/submissions")
def submit(s: SubmissionIn, x_token: str = Header(default="")):
    if config.SUBMIT_TOKEN and x_token != config.SUBMIT_TOKEN:
        raise HTTPException(401, "bad or missing X-Token header")
    hf_id = s.hf_id.strip()
    if not _HF_ID_RE.match(hf_id):
        raise HTTPException(422, "model id must look like org/name (a Hugging Face repo id)")
    if s.kind not in ("auto", "base", "instruct"):
        raise HTTPException(422, "kind must be auto, base or instruct")
    if s.suite not in ("quick", "full"):
        raise HTTPException(422, "suite must be quick or full")
    for row in db.recent(200):
        if row["hf_id"] == hf_id and row["status"] in ACTIVE:
            return {"id": row["id"], "status": row["status"],
                    "note": "already in the queue — joining the existing run"}
    sid = db.add(hf_id, s.kind, s.suite, s.submitter.strip()[:80], s.note.strip()[:200])
    return {"id": sid, "status": "queued"}


@app.get("/api/submissions")
def submissions(limit: int = 100):
    return db.recent(min(limit, 500))


@app.post("/api/submissions/{sid}/cancel")
def cancel(sid: int):
    if not db.cancel(sid):
        raise HTTPException(409, "only queued submissions can be canceled — a running "
                                 "job finishes its current task")
    return {"id": sid, "status": "canceled"}


@app.get("/api/results")
def results():
    return JSONResponse(results_payload())


@app.get("/api/runs/{sid}/log", response_class=PlainTextResponse)
def run_log(sid: int, tail: int = 200):
    sub = db.get(sid)
    if not sub:
        raise HTTPException(404, "no such submission")
    path = config.LOGS_DIR / f"service_{sid}_{sub['hf_id'].replace('/', '__')}.log"
    if not path.exists():
        return "(no log yet — the run has not reached lm_eval)"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-min(tail, 2000):])


@app.get("/healthz")
def healthz():
    return {"ok": True, "queue": sum(r["status"] in ACTIVE for r in db.recent(200))}


# ---------------------------------------------------------------------------
# the dashboard, live flavour: same page, empty data slot
# ---------------------------------------------------------------------------

_PAGE = (report.TEMPLATE
         .replace("__TITLE__", report.html.escape(config.TITLE))
         .replace("__CSS__", report.CSS)
         .replace("__DATA__", "null")
         .replace("__JS__SLOT__", report.JS))


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE
