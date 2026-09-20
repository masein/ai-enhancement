"""The HTTP layer: a submit-and-watch API plus the live dashboard.

    uvicorn service.app:app --host <tailscale-ip> --port 8899

Serving GET / reuses the dashboard from scripts/report_lm_eval.py with the data
slot left empty — the page then fetches /api/results and polls /api/submissions,
so the exact same charts run live here and frozen in the emailed report file.
Bind to the Tailscale IP: the tailnet is the auth boundary; nothing here should
face the open internet.
"""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import config, db, llm, llm_poller, startup, worker
from . import proposals as prop

# the report module is the single source of truth for parsing and for the page
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import report_lm_eval as report  # noqa: E402
import exam_build  # noqa: E402


def _judge_rubric(task: str) -> str:
    import judge as _judge
    return _judge.rubric_for(task)[0]

_HF_ID_RE = re.compile(r"^[\w.\-]{1,96}/[\w.\-]{1,96}$")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # a file this build does not carry fails at `up`, where the operator is
    # looking — not on the first click that needs it, which is how an image
    # without eval_tasks/fr/ shipped and got as far as a person pressing a button
    startup.check_repo_files()
    db.init()
    llm.startup_check()          # a set-but-broken LLM config fails here, not at a click
    worker.start()
    llm_poller.start()
    yield
    llm_poller.stop()
    worker.stop()


app = FastAPI(title="benchmark service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# results payload, cached against the tree's shape
# ---------------------------------------------------------------------------

_cache: dict = {"key": None, "payload": None, "at": 0.0}


# Files whose appearance or rewrite changes what the dashboard should show.
# diagnose.json belongs here as much as results*.json does: scripts/diagnose.py
# writes it long after the eval finished, and a key that ignores it means the
# payload keeps being served from cache with no diagnosis in it.
_WATCH = ("results*.json", "diagnose.json", "model_meta.json", "judge.json",
          "judge_calibration.json")

# The demo's own report, which the demo script writes and GET /demo serves.
# It is NOT under OUT_DIR — the whole point is that the demo's tree is not the
# live tree — so its mtime joins the cache key by hand. HANDOFF §13: both
# cache bugs so far were a file that changed without the key noticing.
DEMO_REPORT = "report.html"


def demo_report_path() -> Path:
    return config.BENCH_ROOT / "demo" / DEMO_REPORT


def demo_report_stamp() -> float:
    p = demo_report_path()
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _judge_identity() -> dict:
    import judge as _judge
    return _judge.identity()


def _calibration() -> dict | None:
    """results/full/judge_calibration.json, written by scripts/judge_calibrate.py
    import — the judge's agreement with a person, without which nothing judged
    is ranked."""
    p = config.OUT_DIR / "judge_calibration.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _tree_key() -> tuple:
    # the taint join reads the database, so its state is part of the key too:
    # a training run registering a dataset changes what the board should show
    if not config.OUT_DIR.is_dir():
        return (0, 0.0, db.taint_stamp(), demo_report_stamp())
    files = [f for pat in _WATCH for f in config.OUT_DIR.rglob(pat)]
    return (len(files), max((f.stat().st_mtime for f in files), default=0.0),
            db.taint_stamp(), demo_report_stamp())


def taint_for(model_ids) -> dict[str, list[str]]:
    """model id -> tasks its training data was derived from. A checkpoint is
    tied to a run the way the Training tab ties it: submitted under the run as
    a checkpoint event, or its id carries the run's hf_prefix."""
    out: dict[str, set] = {}
    links = db.taint_links()
    for mid in model_ids:
        for link in links:
            pre = link["hf_prefix"]
            if mid in link["checkpoints"] or (pre and (mid == pre or mid.startswith(pre))):
                out.setdefault(mid, set()).update(link["tasks"])
    return {k: sorted(v) for k, v in out.items()}


def trail_for(model_ids) -> dict[str, dict]:
    """tainted model id -> the audit trail behind it: the training run that
    consumed the data, the datasets it consumed, and the proposals those came
    from. One click each, because "what taught this model?" should not be a
    database query."""
    links = db.taint_links()
    props = {d["id"]: d["proposal_id"] for d in db.dataset_list(500)}
    out: dict[str, dict] = {}
    for mid in model_ids:
        for link in links:
            pre = link["hf_prefix"]
            if mid in link["checkpoints"] or (pre and (mid == pre or mid.startswith(pre))):
                ds = list(link["datasets"])
                out[mid] = {"run_id": link["run_id"], "datasets": ds,
                            "proposals": sorted({props[d] for d in ds if d in props})}
                break
    return out


def parents_for(model_ids) -> dict[str, str]:
    """tainted model id -> the model its training run started from (the run's
    `parent`, else its config's base_model). The first run that claims a
    checkpoint wins; a checkpoint belongs to one run."""
    out: dict[str, str] = {}
    links = db.taint_links()
    for mid in model_ids:
        for link in links:
            pre = link["hf_prefix"]
            if mid in link["checkpoints"] or (pre and (mid == pre or mid.startswith(pre))):
                if link["parent"]:
                    out[mid] = link["parent"]
                    break
    return out


def results_payload() -> dict:
    now = time.time()
    if _cache["payload"] is not None and now - _cache["at"] < 5:
        return _cache["payload"]
    key = _tree_key()
    if key != _cache["key"] or _cache["payload"] is None:
        runs = report.load_results(config.OUT_DIR) if config.OUT_DIR.is_dir() else []
        by_model = report.merge_runs(runs)
        payload = report.build_payload(by_model, config.TITLE, source=str(config.OUT_DIR),
                                       taint=taint_for(by_model.keys()),
                                       parents=parents_for(by_model.keys()),
                                       calibration=_calibration(),
                                       judge_identity=_judge_identity())
        payload["live"] = True
        # the loop's audit trail, per tainted model: run, datasets, proposals
        trails = trail_for([m["id"] for m in payload["models"]])
        for m in payload["models"]:
            if trails.get(m["id"]):
                m["taintTrail"] = trails[m["id"]]
        # one link, only when a demo has actually run. The live payload is
        # built from OUT_DIR alone and reads nothing under demo/ — this is a
        # timestamp, not a number from that tree.
        stamp = demo_report_stamp()
        payload["demo"] = {"at": stamp, "href": "/demo"} if stamp else None
        _cache.update(key=key, payload=payload)
    _cache["at"] = now
    return _cache["payload"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SubmissionIn(BaseModel):
    hf_id: str
    kind: str = "auto"
    suite: str = "full"                # quick | full | control (the mmlu_perm experiment)
    submitter: str = ""
    note: str = ""
    allow_remote_code: bool = False    # execute the upload's own modeling code
    # narrow a judged run to these built exam tasks. Empty means the whole
    # suite, which is what it has always meant — one topic at a time is the
    # loop's unit of work, and a person should not have to sit fifteen.
    tasks: list[str] = []


ACTIVE = ("queued", "preflight", "waiting_gpu", "waiting_lock", "running")


@app.post("/api/submissions")
def submit(s: SubmissionIn, x_token: str = Header(default="")):
    if config.SUBMIT_TOKEN and x_token != config.SUBMIT_TOKEN:
        raise HTTPException(401, "bad or missing X-Token header")
    hf_id = s.hf_id.strip()
    if not _HF_ID_RE.match(hf_id):
        raise HTTPException(422, "model id must look like org/name — a Hugging Face repo "
                                 "id, or local/<name> for an uploaded artifact")
    if s.kind not in ("auto", "base", "instruct"):
        raise HTTPException(422, "kind must be auto, base or instruct")
    if s.suite not in ("quick", "full", "control", "judged"):
        raise HTTPException(422, "suite must be quick, full, control (mmlu_perm only) or "
                                 "judged (free response + judge)")
    chosen: list[str] = []
    if s.suite == "judged":
        why = config.judged_blocked()
        if why:
            raise HTTPException(503, why)
        built = config.judged_tasks()
        chosen = [t.strip() for t in s.tasks if t.strip()]
        unknown = [t for t in chosen if t not in built]
        if unknown:
            raise HTTPException(422, f"not built exam tasks: {', '.join(unknown)} — built "
                                     f"tasks are {', '.join(built)}")
    elif s.tasks:
        raise HTTPException(422, "tasks narrows a judged run only; the other suites are "
                                 "fixed lists")
    if s.allow_remote_code:
        # the flag is only meaningful for uploads, and only when the operator has
        # configured the server to run other people's code at all. Checked here
        # so the answer is immediate instead of a queued job that fails later.
        if not hf_id.startswith("local/"):
            raise HTTPException(422, "allow_remote_code applies to uploaded artifacts "
                                     "(local/<name>) only — code from the Hub is never "
                                     "executed on this server")
        blocked = config.remote_code_blocked()
        if blocked:
            raise HTTPException(403, f"remote code is not available: {blocked}. "
                                     f"See SERVICE.md § custom model code.")
    for row in db.recent(200):
        if row["hf_id"] != hf_id or row["status"] not in ACTIVE:
            continue
        # …and asking for the same work. A judged run of one topic is not the
        # run of all fifteen: joining them gave one row two jobs, and the
        # queue could not then say which batch belonged to which.
        try:
            same = sorted(json.loads(row.get("tasks") or "[]")) == sorted(chosen)
        except ValueError:
            same = not chosen
        if row["suite"] == s.suite and same:
            return {"id": row["id"], "status": row["status"],
                    "note": "already in the queue — joining the existing run"}
    sid = db.add(hf_id, s.kind, s.suite, s.submitter.strip()[:80], s.note.strip()[:200],
                 allow_remote_code=s.allow_remote_code, tasks=chosen)
    return {"id": sid, "status": "queued", "tasks": sorted(chosen)}


@app.get("/api/submissions")
def submissions(limit: int = 100):
    """The queue. A judged row carries the judge batch with it: the answers
    are on disk long before the grades are, and 'done' on the GPU half is not
    done — the row should say which topics it sat and how far the judge is."""
    rows = db.recent(min(limit, 500))
    judged = [r for r in rows if r["suite"] == "judged"]
    if judged:
        runs = {run["batch_id"]: run for run in db.judge_runs(200)}
        batches = {b["batch_id"]: b for b in db.batches_list(500) if b["kind"] == "judge"}
        # by the batch THIS submission recorded. Matching on the model gave
        # every judged row of a model the newest run's batch, so a finished
        # 130-answer run showed the 680 of the one still going.
        by_model: dict[str, list[dict]] = {}
        for run in db.judge_runs(200):
            by_model.setdefault(run["model"], []).append(run)
        for r in judged:
            bid = r.get("judge_batch") or ""
            run = runs.get(bid)
            if not run:
                # a row from before the batch was recorded on it: the newest
                # run of that model that started after the row did
                started = r.get("started_at") or r.get("created_at") or 0
                run = next((x for x in by_model.get(r["hf_id"], [])
                            if (x.get("created_at") or 0) >= started), None)
            if not run:
                continue
            b = batches.get(run["batch_id"]) or {}
            r["judge"] = {"batch_id": run["batch_id"], "n_items": run["n_items"],
                          "status": b.get("status") or run.get("status"),
                          "progress": b.get("progress") or "",
                          "judge_id": run.get("judge_id")}
    return rows


@app.post("/api/submissions/{sid}/cancel")
def cancel(sid: int):
    if not db.cancel(sid):
        raise HTTPException(409, "only queued submissions can be canceled — a running "
                                 "job finishes its current task")
    return {"id": sid, "status": "canceled"}


# ---------------------------------------------------------------------------
# training-run tracking — the wandb-shaped API (see API.md § run tracking)
# ---------------------------------------------------------------------------

class TrunIn(BaseModel):
    name: str
    project: str = "default"
    submitter: str = ""
    config: dict = {}
    hf_prefix: str = ""
    datasets: list[int] = []     # generated datasets this run trains on — the taint record
    parent: str = ""             # the model this run started from — the "before" of the comparison


class TrunLogIn(BaseModel):
    metrics: list[dict]          # [{step:int, name:str, value:float}, ...]


class TrunEventIn(BaseModel):
    step: int
    kind: str = "checkpoint"
    detail: str = ""


class TrunFinishIn(BaseModel):
    status: str = "finished"


def _check_token(x_token: str):
    if config.SUBMIT_TOKEN and x_token != config.SUBMIT_TOKEN:
        raise HTTPException(401, "bad or missing X-Token header")


@app.post("/api/truns")
def trun_create(t: TrunIn, x_token: str = Header(default="")):
    _check_token(x_token)
    name = t.name.strip()[:120]
    if not name:
        raise HTTPException(422, "run needs a name")
    for did in t.datasets:
        ds = db.dataset_get(did)
        if not ds or ds["status"] != "ready":
            raise HTTPException(422, f"dataset {did} does not exist or is not ready — a run "
                                     f"can only record data it could actually have trained on")
    rid = db.trun_create(name, t.project.strip()[:80] or "default",
                         t.submitter.strip()[:80],
                         json.dumps(t.config)[:20000], t.hf_prefix.strip()[:200],
                         datasets=t.datasets, parent=t.parent.strip()[:200])
    return {"id": rid}


@app.post("/api/truns/{rid}/log")
def trun_log(rid: int, body: TrunLogIn, x_token: str = Header(default="")):
    _check_token(x_token)
    if not db.trun_get(rid):
        raise HTTPException(404, "no such run")
    if len(body.metrics) > 5000:
        raise HTTPException(422, "batch too large (max 5000 points per call)")
    pts = []
    for p in body.metrics:
        try:
            step, name, value = int(p["step"]), str(p["name"])[:80], float(p["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            pts.append((step, name, value))
    return {"logged": db.trun_log(rid, pts) if pts else 0}


@app.post("/api/truns/{rid}/event")
def trun_event(rid: int, e: TrunEventIn, x_token: str = Header(default="")):
    _check_token(x_token)
    if not db.trun_get(rid):
        raise HTTPException(404, "no such run")
    db.trun_event(rid, e.step, e.kind[:40], e.detail[:300])
    return {"ok": True}


@app.post("/api/truns/{rid}/finish")
def trun_finish(rid: int, f: TrunFinishIn, x_token: str = Header(default="")):
    _check_token(x_token)
    if f.status not in ("finished", "failed"):
        raise HTTPException(422, "status must be finished or failed")
    if not db.trun_get(rid):
        raise HTTPException(404, "no such run")
    db.trun_finish(rid, f.status)
    return {"ok": True}


def _parse_ds(run: dict) -> dict:
    try:
        run["datasets"] = [int(x) for x in json.loads(run.get("datasets") or "[]")]
    except (ValueError, TypeError):
        run["datasets"] = []
    return run


@app.get("/api/truns")
def trun_index(project: str | None = None, limit: int = 200):
    return [_parse_ds(r) for r in db.trun_list(project, min(limit, 500))]


# A finished run's series never changes, and a live one changes only when a log
# batch lands — updated_at moves with it, so it is a free cache key. Without
# this, every 5-second poll on a watched run re-read and re-serialized every
# point of every metric.
_SERIES_CACHE: dict[int, tuple] = {}


@app.get("/api/truns/{rid}")
def trun_detail(rid: int, max_points: int = 400):
    run = db.trun_get(rid)
    if not run:
        raise HTTPException(404, "no such run")
    mp = max(50, min(max_points, 5000))
    key = (run["updated_at"], mp)
    hit = _SERIES_CACHE.get(rid)
    _parse_ds(run)
    if hit and hit[0] == key:
        return {"run": run, **hit[1]}
    series = db.trun_series(rid, mp)
    if len(_SERIES_CACHE) > 16:                  # tiny, bounded, no eviction policy needed
        _SERIES_CACHE.clear()
    _SERIES_CACHE[rid] = (key, series)
    return {"run": run, **series}


# ---------------------------------------------------------------------------
# artifact storage — upload a checkpoint directly, no Hub account needed.
# Raw zip body (stdlib-friendly), streamed to disk, extracted with paranoia.
# ---------------------------------------------------------------------------

_ART_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]{1,80}$")


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0


@app.post("/api/artifacts/{name}")
async def artifact_upload(name: str, request: Request, x_token: str = Header(default="")):
    _check_token(x_token)
    if not _ART_NAME_RE.match(name):
        raise HTTPException(422, "artifact name: letters, digits, dot, dash, underscore only")
    dest = config.ARTIFACTS_DIR / name
    cap = int(config.ARTIFACT_MAX_GB * 1e9)
    quota = int(config.ARTIFACT_QUOTA_GB * 1e9)
    used = _dir_bytes(config.ARTIFACTS_DIR)
    # Refusing BEFORE the body is read makes a mid-send client see a bare
    # connection reset instead of the reason. The client declares Content-Length,
    # so size/quota can be judged up front; for a duplicate name, drain the body
    # (bounded by the declared length ≤ cap) and then answer honestly.
    declared = int(request.headers.get("content-length") or 0)
    if declared > cap:
        raise HTTPException(413, f"upload of {declared / 1e9:.1f} GB exceeds "
                                 f"ARTIFACT_MAX_GB={config.ARTIFACT_MAX_GB:g}")
    if used + declared > quota:
        raise HTTPException(507, "artifact storage quota reached — delete old "
                                 "artifacts (GET /api/artifacts to list)")
    if dest.exists():
        async for _ in request.stream():
            pass
        raise HTTPException(409, f"artifact {name!r} already exists — checkpoints are "
                                 f"immutable; use a new name per checkpoint (or "
                                 f"DELETE /api/artifacts/{name} first)")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.ARTIFACTS_DIR / f".upload-{name}.zip"
    got = 0
    try:
        with open(tmp, "wb") as fh:
            async for chunk in request.stream():
                got += len(chunk)
                if got > cap:
                    raise HTTPException(413, f"upload exceeds ARTIFACT_MAX_GB="
                                             f"{config.ARTIFACT_MAX_GB:g}")
                if used + got > quota:
                    raise HTTPException(507, "artifact storage quota reached — delete "
                                             "old artifacts (GET /api/artifacts to list)")
                fh.write(chunk)
        import zipfile
        with zipfile.ZipFile(tmp) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            if sum(i.file_size for i in infos) > cap * 3:
                raise HTTPException(413, "zip expands past three times the upload cap")
            # strip a single shared top-level directory if the zip has one
            roots = {i.filename.split("/", 1)[0] for i in infos}
            strip = (roots.pop() + "/") if len(roots) == 1 and all(
                "/" in i.filename for i in infos) else ""
            dest.mkdir(parents=True)
            for i in infos:
                rel = i.filename[len(strip):] if i.filename.startswith(strip) else i.filename
                target = (dest / rel).resolve()
                if not str(target).startswith(str(dest.resolve()) + "/"):
                    raise HTTPException(422, f"zip member escapes the artifact dir: {i.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as src, open(target, "wb") as out:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        out.write(buf)
        if not (dest / "config.json").exists():
            raise HTTPException(422, "no config.json at the checkpoint root — zip the "
                                     "directory save_pretrained() produced")
        if list(dest.glob("*.bin")):
            raise HTTPException(422, "pickle-format weights (*.bin) execute code on load "
                                     "and are refused — re-save with safetensors")
        return {"model_id": f"local/{name}", "bytes": _dir_bytes(dest)}
    except HTTPException:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
        raise
    except Exception as e:                       # bad zip, disk error
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(422, f"could not unpack upload: {e}") from e
    finally:
        tmp.unlink(missing_ok=True)


@app.get("/api/artifacts")
def artifact_index():
    out = []
    if config.ARTIFACTS_DIR.is_dir():
        for d in sorted(config.ARTIFACTS_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                out.append({"name": d.name, "model_id": f"local/{d.name}",
                            "bytes": _dir_bytes(d),
                            "created": d.stat().st_mtime})
    return {"artifacts": out,
            "total_bytes": sum(a["bytes"] for a in out),
            "quota_bytes": int(config.ARTIFACT_QUOTA_GB * 1e9)}


@app.delete("/api/artifacts/{name}")
def artifact_delete(name: str, x_token: str = Header(default="")):
    _check_token(x_token)
    if not _ART_NAME_RE.match(name):
        raise HTTPException(422, "bad artifact name")
    mid = f"local/{name}"
    for row in db.recent(200):
        if row["hf_id"] == mid and row["status"] in ACTIVE:
            raise HTTPException(409, "that artifact is queued or being evaluated")
    d = config.ARTIFACTS_DIR / name
    if not d.is_dir():
        raise HTTPException(404, "no such artifact")
    import shutil
    shutil.rmtree(d)
    return {"deleted": name, "note": "its benchmark results stay on the leaderboard"}


@app.get("/api/results")
def results():
    return JSONResponse(results_payload())


# ---------------------------------------------------------------------------
# find the gap, make data for it — see service/proposals.py for the pipeline
# and the one rule every endpoint here enforces
# ---------------------------------------------------------------------------

class ProposalIn(BaseModel):
    model: str
    topic: str                   # an exam topic from scripts/categories.yaml
    requested_by: str = ""


class ApproveIn(BaseModel):
    approver: str
    edited_text: str = ""


class RejectIn(BaseModel):
    approver: str
    reason: str = ""


class GenerateIn(BaseModel):
    requester: str
    count: int = 20
    fmt: str = prop.DEFAULT_FORMAT      # prose documents, not question-and-answer pairs


def _llm_status() -> dict:
    why = llm.blocked()
    used = prop.dir_bytes(config.DATASETS_DIR)
    return {"configured": not why, "reason": why,
            "provider": config.LLM_PROVIDER, "model": config.LLM_MODEL,
            "usage_today": db.llm_items_today(), "daily_cap": config.LLM_DAILY_ITEM_CAP,
            "max_items_per_batch": config.LLM_MAX_ITEMS_PER_BATCH,
            # what THIS provider is asked for per request: a local generator
            # gets one document, because its replies are capped
            "items_per_generation_request": {f: prop.items_per_request(f) for f in prop.FORMATS},
            "formats": list(prop.FORMATS), "default_format": prop.DEFAULT_FORMAT,
            "datasets_bytes": used, "datasets_quota_bytes": int(config.DATASET_QUOTA_GB * 1e9),
            "note": "the tailnet is the auth boundary: approvals record a typed name, "
                    "nothing more"}


@app.get("/api/llm")
def llm_status():
    return _llm_status()


# ---------------------------------------------------------------------------
# the exam: drafted by an LLM, curated by a person, split by qid
# ---------------------------------------------------------------------------

class CandidateAccept(BaseModel):
    approver: str
    prompt: str | None = None
    reference: str | None = None
    notes: str | None = None


class CandidateReject(BaseModel):
    approver: str
    reason: str = ""


@app.get("/api/exam")
def exam_status():
    why = llm.blocked("exam")
    return {"configured": not why, "reason": why,
            "provider": config.EXAM_PROVIDER, "model": config.EXAM_MODEL,
            "root": str(config.EXAM_DIR), "tasks_built": config.judged_tasks(),
            "target_per_topic": exam_build.TARGET_PER_TOPIC,
            "summary": exam_build.summary(config.EXAM_DIR),
            "draft_command": f"python3 scripts/exam_build.py draft --root {config.EXAM_DIR} "
                             f"--per-topic 8",
            # the other way a bank arrives: written by a person, imported whole.
            # Both paths are shown on the tab, because only one of them was.
            "import_command": f"python3 scripts/exam_build.py --root {config.EXAM_DIR} import "
                              f"<questions.json> --topic \"<topic>\" --approver \"<name>\"",
            "migrate_command": f"python3 scripts/exam_build.py --root {config.EXAM_DIR} migrate",
            "note": "the published per-topic score comes from the report half; only the "
                    "diagnose half is ever shown here or placed in a request"}


@app.get("/api/exam/candidates")
def exam_candidates(topic: str | None = None, status: str = "candidate"):
    return exam_build.load_candidates(config.EXAM_DIR, topic or None,
                                      None if status == "all" else status)


@app.get("/api/exam/bank")
def exam_bank(topic: str | None = None):
    """The bank with report-half text withheld — see exam_build.public_bank."""
    return exam_build.public_bank(config.EXAM_DIR, topic or None)


@app.post("/api/exam/candidates/{cid}/accept")
def exam_accept(cid: str, a: CandidateAccept, x_token: str = Header(default="")):
    _check_token(x_token)
    who = _name(a.approver, "accepting a question")
    try:
        rec = exam_build.accept(config.EXAM_DIR, cid, who, a.prompt, a.reference, a.notes)
    except KeyError:
        raise HTTPException(404, "no such candidate") from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    db.curation_add(cid, rec["topic"], rec["qid"], "accepted", who, rec["edited"])
    return {"cid": cid, "qid": rec["qid"], "topic": rec["topic"],
            "half": exam_build.half_of(rec["qid"]), "edited": rec["edited"], "accepted_by": who}


@app.post("/api/exam/candidates/{cid}/reject")
def exam_reject(cid: str, a: CandidateReject, x_token: str = Header(default="")):
    _check_token(x_token)
    who = _name(a.approver, "rejecting a question")
    try:
        rec = exam_build.reject(config.EXAM_DIR, cid, who, a.reason)
    except KeyError:
        raise HTTPException(404, "no such candidate") from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    db.curation_add(cid, rec["topic"], "", "rejected", who, False, rec.get("reason", ""))
    return {"cid": cid, "status": "rejected"}


# ---------------------------------------------------------------------------
# a person delivers a bank, and the rubric that grades it, from the page.
# Two steps every time: a preview that writes nothing, then a commit. The
# commit runs the same code as the CLI, so a bank imported here and one
# imported from a shell are the same records.
# ---------------------------------------------------------------------------

IMPORT_MAX_BYTES = 2 * 1024 * 1024


class ImportIn(BaseModel):
    topic: str
    approver: str                      # who WROTE the questions; it goes on every record
    imported_by: str = ""              # who put them in, when that is someone else
    filename: str = ""                 # the file's own name: the source falls back to its stem
    source: str = ""
    # the file as it is: an array, or an object holding one — physics &
    # engineering arrived as {"questions": [...]}, and a file is not refused
    # over its wrapping
    items: list | dict | None = None
    text: str = ""                     # or the file's text, parsed here


_NOT_A_BANK = ("the file must hold a JSON array of question objects, or an object with one "
               "list in it (\"questions\", \"items\", …)")


def _import_items(body: ImportIn):
    """The file as it arrived — array or object — checked here only for being
    readable at all. exam_build.plan_import does the unwrapping and reports
    which key it came out of, so the page and `exam_build.py import` read a
    delivered file exactly the same way. The page used to have its own
    array-only check in front of that, which is how the one wrapped file of
    the five was refused."""
    raw = body.text or ""
    data = body.items if body.items is not None else None
    if data is None:
        if len(raw.encode("utf-8")) > IMPORT_MAX_BYTES:
            raise HTTPException(413,
                                f"the file is larger than {IMPORT_MAX_BYTES // 1024 // 1024} MB")
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"not valid JSON: {e}") from None
    if not isinstance(exam_build.unwrap_items(data)[0], list):
        raise HTTPException(422, _NOT_A_BANK)
    return data


def _import_preview(plan: dict) -> dict:
    """What the page may show of what would land. A report-half question is
    withheld here exactly as public_bank withholds it: its author wrote it,
    and the page still does not echo it back. Both lists the plan carries —
    the new records and the revisions of ones already in the bank — go
    through the same withholding; neither raw list is in the response."""
    shown = []
    for rec, what in ([(r, "new") for r in plan["records"]]
                      + [(r, "updated") for r in plan["updates"]]):
        half = exam_build.half_of(rec["qid"])
        row = {"qid": rec["qid"], "half": half, "meta": rec["meta"],
               "reference": rec["reference"], "change": what}
        if half == "diagnose":
            row["prompt"] = rec["prompt"]
        else:
            row["prompt"] = None
            row["withheld"] = "report half — never shown, never exported"
        shown.append(row)
    return ({k: v for k, v in plan.items() if k not in ("records", "updates")}
            | {"items": shown})


@app.post("/api/exam/import/preview")
def exam_import_preview(body: ImportIn, x_token: str = Header(default="")):
    _check_token(x_token)
    who = _name(body.approver, "importing a bank")
    try:
        plan = exam_build.plan_import(config.EXAM_DIR, _import_items(body), body.topic.strip(),
                                      who, (body.source or "").strip(),
                                      filename=body.filename, imported_by=body.imported_by.strip())
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return _import_preview(plan)


@app.post("/api/exam/import")
def exam_import(body: ImportIn, x_token: str = Header(default="")):
    _check_token(x_token)
    who = _name(body.approver, "importing a bank")
    items = _import_items(body)
    by = body.imported_by.strip()
    try:
        plan = exam_build.plan_import(config.EXAM_DIR, items, body.topic.strip(), who,
                                      (body.source or "").strip(), filename=body.filename,
                                      imported_by=by)
        out = exam_build.write_import(config.EXAM_DIR, plan)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    # one curation row for the import, the way an accept writes one per question
    db.curation_add(f"import:{out['source']}", out["topic"], "", "imported", by or who,
                    False, f"{out['imported']} imported, {out['updated']} revised, "
                           f"{out['skipped']} already in the bank; written by {who}")
    return out


def _rubric_row(topic: str) -> dict:
    """One topic's instrument, or — when its files cannot be read — a row
    that says so. A board that shows fifteen topics must not disappear
    because one of them has a missing or unreadable file; the endpoints that
    render it are how a person finds out which one."""
    import judge as _judge
    task = exam_build.topic_task(topic)
    base = {"topic": topic, "task": task, "name": "", "own": False, "fallback": False,
            "sha256": "", "version": "", "status": "", "path": "", "scoring": "single",
            "criteria_path": "", "criteria_sha256": "", "criteria_count": 0, "error": ""}
    try:
        r = _judge.rubric_for(task)
    except (_judge.RubricMissing, OSError, ValueError) as e:
        # ValueError covers a criteria file that is not JSON, and
        # CriteriaError (an effect this judge cannot apply) is one
        return {**base, "name": _judge.rubric_name(task), "error": str(e)}
    spec = _judge.rubric_path(r.name, ".criteria.json")
    return {**base, "name": r.name, "own": not r.fallback and r.name != "factual_accuracy",
            "fallback": r.fallback,
            "sha256": r.sha256, "version": r.version, "status": r.status,
            "path": r.path, "scoring": "criteria" if r.criteria else "single",
            "criteria_path": str(spec) if spec else "",
            "criteria_sha256": r.criteria_sha256,
            "criteria_version": (r.criteria or {}).get("version"),
            "criteria_status": (r.criteria or {}).get("status", ""),
            "criteria_count": len(_judge.criteria_ids(r.criteria)) if r.criteria else 0}


def _rubric_store() -> tuple[Path, bool]:
    """Where an uploaded rubric goes, and whether it is the repo's own copy.
    In the container /app is the image, not the checkout, so the writable
    place is under BENCH_ROOT — and judge.rubric_dirs() looks there first."""
    import judge as _judge
    repo_dir = _judge.RUBRIC_DIR
    if os.access(repo_dir, os.W_OK):
        return repo_dir, True
    return Path(config.BENCH_ROOT) / "rubrics", False


@app.get("/api/exam/rubrics")
def exam_rubrics():
    store, in_repo = _rubric_store()
    # the bank beside the instrument: a rubric and a criteria file ship in the
    # repo, so every topic looks equipped here whether or not anyone has
    # written it a single question. Those are different facts and the table
    # has to show both.
    banks = exam_build.summary(config.EXAM_DIR)
    return {"store": str(store), "in_repo": in_repo,
            "banks": {t: {k: (banks.get(t) or {}).get(k, 0)
                          for k in ("accepted", "report", "diagnose", "pending")}
                      for t in exam_build.TOPICS},
            "note": ("uploads land here and the judge reads them before the repo's own copies"
                     if not in_repo else "uploads replace the repo's own copies in this checkout"),
            "topics": [_rubric_row(t) for t in exam_build.TOPICS],
            "changes": db.rubric_changes(20)}


@app.get("/api/exam/rubrics/{name}")
def exam_rubric_file(name: str, kind: str = "rubric"):
    import judge as _judge
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise HTTPException(404, "no such rubric")
    p = _judge.rubric_path(name, ".md" if kind == "rubric" else ".criteria.json")
    if not p:
        raise HTTPException(404, f"no {kind} file for {name}")
    return PlainTextResponse(p.read_text(encoding="utf-8"),
                             media_type="text/markdown" if kind == "rubric"
                             else "application/json")


class RubricIn(BaseModel):
    name: str
    kind: str = "rubric"               # rubric | criteria
    content: str
    approver: str
    note: str = ""


def _rubric_check(body: RubricIn) -> dict:
    """Everything the commit would do, and every reason it should not."""
    import judge as _judge
    if not re.fullmatch(r"[a-z0-9_]+", body.name or ""):
        raise HTTPException(422, "a rubric name is lower-case letters, digits and underscores, "
                                 "and matches the topic's slug")
    if body.kind not in ("rubric", "criteria"):
        raise HTTPException(422, "kind must be rubric or criteria")
    if len(body.content.encode("utf-8")) > IMPORT_MAX_BYTES:
        raise HTTPException(413, "that file is too large")
    problems: list[str] = []
    suffix = ".md" if body.kind == "rubric" else ".criteria.json"
    if body.kind == "criteria":
        try:
            spec = json.loads(body.content)
        except (ValueError, TypeError) as e:
            problems.append(f"not valid JSON: {e}")
        else:
            problems.extend(_judge.validate_criteria(spec))
    else:
        if "# Rubric" not in body.content:
            problems.append("a rubric starts with a '# Rubric — <topic> (version N)' heading; "
                            "the version is recorded in every judge.json")
        for anchor in range(5):
            if f"**{anchor}**" not in body.content:
                problems.append(f"no anchor for {anchor} — the 0-4 scale needs all five")
    current = _judge.rubric_path(body.name, suffix)
    was = current.read_text(encoding="utf-8") if current else ""
    new_sha = hashlib.sha256(body.content.encode("utf-8")).hexdigest()
    store, in_repo = _rubric_store()
    return {
        "name": body.name, "kind": body.kind, "problems": problems,
        "ok": not problems,
        "path": str(store / f"{body.name}{suffix}"), "in_repo": in_repo,
        "sha256": new_sha,
        "was_sha256": hashlib.sha256(was.encode("utf-8")).hexdigest() if was else "",
        "existed": bool(current), "was_path": str(current) if current else "",
        "changed": new_sha != (hashlib.sha256(was.encode("utf-8")).hexdigest() if was else ""),
        "diff": list(difflib.unified_diff(
            was.splitlines(), body.content.splitlines(),
            fromfile=str(current) if current else "(none)", tofile="uploaded", lineterm=""))[:400],
        "warning": ("Committing changes this file's sha256, which is recorded in every "
                    "judge.json. Scores judged before and after it are from different "
                    "instruments and are not comparable; re-run suite=judged for this topic."),
    }


@app.post("/api/exam/rubrics/preview")
def exam_rubric_preview(body: RubricIn, x_token: str = Header(default="")):
    _check_token(x_token)
    _name(body.approver, "changing a rubric")
    return _rubric_check(body)


@app.post("/api/exam/rubrics")
def exam_rubric_commit(body: RubricIn, x_token: str = Header(default="")):
    _check_token(x_token)
    who = _name(body.approver, "changing a rubric")
    check = _rubric_check(body)
    if not check["ok"]:
        raise HTTPException(422, "; ".join(check["problems"]))
    suffix = ".md" if body.kind == "rubric" else ".criteria.json"
    store, _ = _rubric_store()
    store.mkdir(parents=True, exist_ok=True)
    dest = store / f"{body.name}{suffix}"
    dest.write_text(body.content, encoding="utf-8")
    db.rubric_change_add(body.name, body.kind, str(dest), check["sha256"],
                         check["was_sha256"], who, body.note.strip()[:300])
    _cache.update(key=None, payload=None, at=0.0)     # the page shows the sha
    return {**check, "written": str(dest), "approver": who}


@app.post("/api/exam/build")
def exam_build_tasks(x_token: str = Header(default="")):
    """Write the harness tasks from the bank (+ the MMLU control set). No GPU;
    a curator does this after a round of accepting so suite=judged runs the
    new questions."""
    _check_token(x_token)
    try:
        m = exam_build.build(config.OUT_DIR, config.EXAM_DIR)
    except FileNotFoundError as e:
        # a file this build does not carry: say which one, in the answer the
        # button shows, rather than a bare 500 with the reason in the log
        raise HTTPException(500, f"the tasks could not be built: {e}. A file this build "
                                 f"needs is missing — see SERVICE.md § Docker.") from None
    except (ValueError, OSError) as e:
        raise HTTPException(500, f"the tasks could not be built: {e}") from None
    return {"tasks": {t: {k: v[k] for k in ("items", "report", "diagnose")}
                      for t, v in m["tasks"].items()},
            "tasks_dir": str(exam_build.tasks_dir(config.EXAM_DIR))}


@app.get("/api/judge/justifications")
def judge_justifications(model: str, topic: str, limit: int = 8):
    """What the judge wrote about one model's answers on one topic —
    DIAGNOSIS half only, with any exam question text the judge quoted already
    stripped. The same function a proposal is built from, so the page shows
    exactly what the LLM would be given."""
    task = exam_build.topic_task(topic)
    items, counts = prop.justifications_for(config.OUT_DIR / model.replace("/", "__"), task)
    return {"model": model, "topic": topic, "task": task, "counts": counts,
            "items": items[:max(1, min(limit, prop.MAX_JUSTIFICATIONS))],
            "note": "diagnosis half only; any exam question text the judge quoted is removed"}


# ---------------------------------------------------------------------------
# The loop, as one board. Every number here already exists somewhere else —
# the bank, the rubric files, judge.json, the proposals table, the datasets
# table. What this endpoint adds is the ORDER: for one topic, what has been
# done and what the single next step is. The gates are not re-implemented
# here; they are the same objects the API enforces on submit and on propose.
# ---------------------------------------------------------------------------

STEPS = {
    "import": "Import a bank",
    "sit": "Sit the exam",
    "read": "Read the results",
    "propose": "Propose",
    "review": "Review the spec",
    "generate": "Generate",
    "hand": "Hand to training",
}


def _judged_at(model: str) -> float | None:
    """When this model's judge.json was written. The file itself carries no
    timestamp on purpose (it must hash the same for the same inputs), so the
    filesystem is the clock."""
    p = config.OUT_DIR / model.replace("/", "__") / "judge.json"
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _last_judged(payload: dict, task: str) -> dict | None:
    """The most recently written judge.json that contains this topic, as the
    board's 'where does this topic stand' line."""
    best = None
    for m in payload.get("models") or []:
        t = ((m.get("judge") or {}).get("tasks") or {}).get(task)
        if not t:
            continue
        at = _judged_at(m["id"]) or 0
        if best and at <= best["at"]:
            continue
        j = m.get("judge") or {}
        jm = j.get("judge") or {}
        best = {
            "model": m["id"], "at": at,
            "score_report": t.get("score_report"), "n_report": t.get("n_report"),
            "score_diagnose": t.get("score_diagnose"), "n_diagnose": t.get("n_diagnose"),
            "unparseable": t.get("unparseable"),
            "flags": {fid: f.get("n") for fid, f in (t.get("flags") or {}).items()},
            "judge_id": jm.get("id"), "provisional": bool(jm.get("provisional")),
            "provisional_reason": jm.get("provisional_reason") or "",
            "single_provider_loop": bool(jm.get("single_provider_loop")),
            "draft_rubric": task in (jm.get("rubrics_draft") or []),
            "state": m.get("judgeState"), "propose": t.get("propose"),
            "tainted": task in (m.get("tainted") or []),
        }
    return best


def _dataset_kept(d: dict) -> int | None:
    try:
        return ((json.loads(d.get("provenance") or "{}") or {}).get("items") or {}).get("kept")
    except ValueError:
        return None


def _loop_row(topic: str, payload: dict, props: list[dict], sets: list[dict],
              blocked: str) -> dict:
    task = exam_build.topic_task(topic)
    bank = (exam_build.summary(config.EXAM_DIR) or {}).get(topic) or {}
    rub = _rubric_row(topic)
    last = _last_judged(payload, task)
    mine = [p for p in props if p["category"] == topic]
    open_p = next((p for p in mine if p["status"] in ("pending", "proposed", "approved")), None)
    ds = [d for d in sets if d.get("proposal_id") in {p["id"] for p in mine}]
    ready = [d for d in ds if d.get("status") == "ready"]
    gate = (last or {}).get("propose") or {}
    if rub.get("error"):
        # a topic whose instrument cannot be read cannot be sat, judged or
        # proposed from; the row says which file is missing rather than
        # offering a button that would fail later
        step, ok, why = "read", False, rub["error"]
    elif not bank.get("accepted"):
        step, ok, why = "import", True, ""
    elif not last:
        step, ok, why = "sit", not blocked, blocked
    elif open_p and open_p["status"] == "approved":
        step = "hand" if ready else "generate"
        ok, why = True, ""
    elif open_p:
        step, ok, why = "review", True, ""
    elif ready:
        step, ok, why = "hand", True, ""
    else:
        step, ok, why = "read", True, ""
    return {
        "topic": topic, "task": task, "slug": exam_build.task_slug(task),
        "error": rub.get("error", ""),
        "bank": {"accepted": bank.get("accepted", 0), "report": bank.get("report", 0),
                 "diagnose": bank.get("diagnose", 0), "awaiting": bank.get("pending", 0),
                 "floor": report.PROPOSE_MIN_N,
                 "under_floor": bank.get("report", 0) < report.PROPOSE_MIN_N},
        "rubric": {k: rub[k] for k in ("name", "own", "fallback", "version", "status",
                                       "scoring", "sha256", "criteria_count",
                                       "criteria_sha256")},
        "last_judged": last,
        # the same object the propose endpoint enforces — the page never
        # decides for itself whether a topic may be proposed from
        "propose": {"ok": bool(gate.get("ok")), "why": gate.get("why"),
                    "short": gate.get("short"), "caution": gate.get("caution")} if last else None,
        "proposal": {k: open_p[k] for k in ("id", "status", "model", "category",
                                            "created_at", "requested_by")} if open_p else None,
        # kept/dropped live in the provenance record the gate wrote, which is
        # the only place that count is authoritative
        "datasets": [{"id": d["id"], "status": d.get("status"), "count": d.get("count"),
                      "kept": _dataset_kept(d), "created_at": d.get("created_at")} for d in ds],
        "next": {"step": step, "label": STEPS[step], "ok": ok, "why": why},
    }


def _loop_row_safe(topic: str, payload: dict, props: list[dict], sets: list[dict],
                   blocked: str) -> dict:
    """A row that cannot take the board down with it. One topic's files being
    unreadable is a fact about that topic, and the other fourteen rows are
    still the answer to 'what do I do next'."""
    try:
        return _loop_row(topic, payload, props, sets, blocked)
    except Exception as e:                                  # noqa: BLE001
        why = f"this topic could not be read: {e}"
        return {"topic": topic, "task": exam_build.topic_task(topic),
                "slug": exam_build.task_slug(exam_build.topic_task(topic)),
                "bank": {"accepted": 0, "report": 0, "diagnose": 0, "awaiting": 0,
                         "floor": report.PROPOSE_MIN_N, "under_floor": False},
                "rubric": {"name": "", "own": False, "version": "", "status": "",
                           "scoring": "single", "sha256": "", "criteria_count": 0,
                           "criteria_sha256": ""},
                "last_judged": None, "propose": None, "proposal": None, "datasets": [],
                "error": why,
                "next": {"step": "read", "label": "Read the results", "ok": False, "why": why}}


@app.get("/api/loop")
def loop_board():
    """One row per topic: the bank, the rubric, where the last judged run
    left it, and the one next step."""
    payload = results_payload()
    props = db.proposal_list(limit=500)
    sets = db.dataset_list(limit=500)
    blocked = config.judged_blocked()
    return {"topics": [_loop_row_safe(t, payload, props, sets, blocked)
                       for t in exam_build.TOPICS],
            "judged_blocked": blocked, "tasks_built": config.judged_tasks(),
            "floor": report.PROPOSE_MIN_N,
            "gap_dataset_flag": "--gap-dataset"}


@app.get("/api/answers")
def answers(model: str, topic: str, limit: int = 200):
    """One model's DIAGNOSE-half answers on one topic, with what the judge
    made of each. The report half is here as one aggregate line and nothing
    else: not its questions, not its answers, not its qids. An answer quotes
    its question often enough that showing one shows the other."""
    if topic not in exam_build.TOPICS:
        raise HTTPException(404, f"{topic!r} is not an exam topic")
    task = exam_build.topic_task(topic)
    model_dir = config.OUT_DIR / model.replace("/", "__")
    jf = model_dir / "judge.json"
    if not jf.exists():
        raise HTTPException(404, f"{model} has no judged run on file — submit it with "
                                 f"suite=judged first")
    j = json.loads(jf.read_text(encoding="utf-8"))
    t = (j.get("tasks") or {}).get(task)
    if not t:
        raise HTTPException(404, f"{model}'s judged run does not cover {topic!r}")
    import judge as _judge
    spec = _judge.rubric_for(task).criteria or {}
    answers_by_hash = {}
    for rec in _judge._records(model_dir, task):
        answers_by_hash[rec.get("doc_hash")] = ((rec.get("doc") or {}), _judge._answer(rec))
    rows = []
    for it in t.get("items") or []:
        if it.get("half") != "diagnose":
            continue                      # the whole safety property, in one line
        doc, ans = answers_by_hash.get(it.get("doc_hash"), ({}, ""))
        rows.append({
            "qid": it.get("qid"), "score": it.get("score"), "graded": it.get("graded"),
            "meta": it.get("meta") or {}, "answer_words": it.get("answer_words"),
            "prompt": doc.get("prompt") or "", "reference": doc.get("reference") or "",
            "answer": ans, "criteria": it.get("criteria") or {},
            "flags": {k: bool(v) for k, v in (it.get("flags") or {}).items()},
            "effects_applied": (it.get("fold") or {}).get("effects_applied") or [],
            "justification": it.get("justification") or "",
        })
    rows.sort(key=lambda r: (r["score"] if r["score"] is not None else 99, str(r["qid"])))
    return {
        "model": model, "topic": topic, "task": task,
        "criteria": [{"id": c["id"], "label": _judge.label_of(c)}
                     for c in spec.get("criteria") or []],
        "flags": [{"id": f["id"], "label": _judge.label_of(f),
                   "effect_words": _judge.effect_words(_judge.effect_of(f))}
                  for f in spec.get("flags") or []],
        # the published half, as a number and never as rows. Its flag counts
        # are the REPORT half's: the whole-bank figure under this heading
        # says something untrue about the published score
        "report_half": {"n": t.get("n_report"), "mean": t.get("score_report"),
                        "flags": {fid: f.get("n_report", 0)
                                  for fid, f in (t.get("flags") or {}).items()},
                        "flags_whole_bank": {fid: f.get("n", 0)
                                             for fid, f in (t.get("flags") or {}).items()},
                        "note": "report-half questions and answers are never listed — the "
                                "published score is this line"},
        "items": rows[:max(1, min(limit, 500))], "n_diagnose": len(rows),
    }


@app.get("/api/judge")
def judge_status():
    """What the judged suite would run with: the pinned judge, its family,
    the tasks built, and the calibration on file."""
    why = config.judged_blocked()
    import judge as _judge      # scripts/, on sys.path above
    cal = _calibration() or {}
    ident = _judge.identity()
    return {"configured": not why, "reason": why, "judge_model": config.JUDGE_MODEL,
            "judge_provider": config.JUDGE_PROVIDER, "judge_id": ident["id"],
            "judge_family": ident["family"],
            "single_provider_loop": _judge.single_provider_loop(),
            "provider_clash": _judge.provider_clash(),
            "canary_max_drift": config.JUDGE_CANARY_MAX_DRIFT,
            "tasks": config.judged_tasks(), "tasks_dir": str(config.JUDGED_TASKS_DIR),
            "runs": db.judge_runs(20),
            "calibration": {k: cal.get(k) for k in ("kappa", "n", "calibrated", "kappa_min")}
            | ({"judge_id": (cal.get("judge") or {}).get("id")} if cal else {})
            if cal else None}


def _spend_check(n_items: int) -> None:
    if n_items > config.LLM_MAX_ITEMS_PER_BATCH:
        raise HTTPException(422, f"{n_items} batch items exceeds LLM_MAX_ITEMS_PER_BATCH="
                                 f"{config.LLM_MAX_ITEMS_PER_BATCH}")
    used = db.llm_items_today()
    if used + n_items > config.LLM_DAILY_ITEM_CAP:
        raise HTTPException(429, f"today's LLM use ({used} items) plus this request "
                                 f"({n_items}) would pass LLM_DAILY_ITEM_CAP="
                                 f"{config.LLM_DAILY_ITEM_CAP}; try tomorrow or raise the cap")


def _require_llm() -> llm.Backend:
    why = llm.blocked()
    if why:
        raise HTTPException(503, why)
    try:
        return llm.client()
    except llm.LLMError as e:      # a local server that is down, or serves another model
        raise HTTPException(503, str(e)) from None


@app.post("/api/proposals")
def proposal_create(p: ProposalIn, x_token: str = Header(default="")):
    """Ask the LLM what skill is missing, from the judge's written assessments
    of this topic's DIAGNOSE-half answers. The gate is enforced here, not just
    on the button: a preliminary judged suite, a topic under the noise floor,
    or a model that wrote nothing is refused with the same words the page
    shows."""
    _check_token(x_token)
    backend = _require_llm()
    payload = results_payload()
    row = next((m for m in payload["models"] if m["id"] == p.model), None)
    if not row:
        raise HTTPException(404, f"no such model on the board: {p.model}")
    task = exam_build.topic_task(p.topic)
    judge = row.get("judge") or {}
    t = (judge.get("tasks") or {}).get(task)
    if not t:
        raise HTTPException(404, f"{p.model} has no judged answers on file for {p.topic!r} — "
                                 f"submit it with suite=judged first")
    gate = t.get("propose")
    if gate is None:
        raise HTTPException(422, f"{p.topic!r} is not an exam topic")
    if not gate["ok"]:
        raise HTTPException(409, gate["why"])
    dup = db.proposal_active(p.model, task, p.topic)
    if dup:
        raise HTTPException(409, f"proposal #{dup['id']} for this model and topic is already "
                                 f"{dup['status']} — review it in the Review tab")
    _spend_check(1)
    model_dir = config.OUT_DIR / p.model.replace("/", "__")
    justifications, counts = prop.justifications_for(model_dir, task)
    if not justifications:
        raise HTTPException(409, "the judge wrote no assessment of a diagnosis-half answer "
                                 "that fell short on this topic — nothing to propose from")
    jmeta = judge.get("judge") or {}
    evidence = {
        "n_shown": len(justifications), **counts,
        "topic_score_report": t.get("score_report"), "topic_n_report": t.get("n_report"),
        "topic_score_diagnose": t.get("score_diagnose"), "topic_n_diagnose": t.get("n_diagnose"),
        "judge_id": jmeta.get("id"), "mmlu_caution": gate.get("caution"),
        # what the reviewer sees of what the LLM saw — diagnosis half, with any
        # question text the judge quoted already stripped
        "examples": justifications[:prop.EXAMPLES_SHOWN],
    }
    pid = db.proposal_create(p.model, task, p.topic, p.requested_by.strip()[:80], evidence)
    req = prop.proposal_request(pid, p.model, task, p.topic, justifications, counts,
                                _judge_rubric(task), prop.criteria_evidence(model_dir, task),
                                audience=prop.audience_for(p.topic, task))
    try:
        bid = backend.submit([req])
    except llm.LLMError as e:
        db.proposal_update(pid, status="failed", error=str(e)[:400])
        raise HTTPException(502, f"the LLM batch could not be submitted: {e}") from None
    db.batch_add(bid, "proposal", pid, 1, backend.name, backend.model)
    db.proposal_update(pid, batch_id=bid, prompt_sha=llm.prompt_sha(req.system, req.user),
                       judge_run=json.dumps({"judge_id": jmeta.get("id"),
                                             "batch_id": jmeta.get("batch_id"),
                                             "prompt_sha256": jmeta.get("prompt_sha256")}))
    return {"id": pid, "status": "pending", "batch_id": bid, "task": task}


def _proposal_view(r: dict, datasets: list[dict] | None = None) -> dict:
    out = dict(r)
    try:
        out["evidence"] = json.loads(r.get("evidence") or "{}")
    except (ValueError, TypeError):
        out["evidence"] = {}
    out["datasets"] = [{"id": d["id"], "status": d["status"], "fmt": d["fmt"],
                        "count": d["count"], "error": d["error"]}
                       for d in (datasets or []) if d["proposal_id"] == r["id"]]
    return out


@app.get("/api/proposals")
def proposal_index(status: str | None = None, limit: int = 200):
    ds = db.dataset_list(500)
    return [_proposal_view(r, ds) for r in db.proposal_list(status, min(limit, 500))]


@app.get("/api/proposals/{pid}")
def proposal_detail(pid: int):
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    return _proposal_view(r, db.dataset_list(500))


def _name(s: str, what: str) -> str:
    s = s.strip()[:80]
    if not s:
        raise HTTPException(422, f"{what} needs a name — the tailnet is the auth boundary, "
                                 f"so the record of who decided is the name you type")
    return s


@app.post("/api/proposals/{pid}/approve")
def proposal_approve(pid: int, a: ApproveIn, x_token: str = Header(default="")):
    _check_token(x_token)
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    if r["status"] != "proposed":
        raise HTTPException(409, f"proposal #{pid} is {r['status']}, not awaiting review")
    who = _name(a.approver, "approving")
    edited = a.edited_text.strip()[:2000]
    if edited == r["spec_text"].strip():
        edited = ""                                   # approved as written
    db.proposal_update(pid, status="approved", approver=who, edited_text=edited,
                       approved_at=time.time())
    return {"id": pid, "status": "approved", "approver": who, "edited": bool(edited)}


@app.post("/api/proposals/{pid}/reject")
def proposal_reject(pid: int, a: RejectIn, x_token: str = Header(default="")):
    _check_token(x_token)
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    if r["status"] not in ("proposed", "approved"):
        raise HTTPException(409, f"proposal #{pid} is {r['status']} and cannot be rejected")
    who = _name(a.approver, "rejecting")
    db.proposal_update(pid, status="rejected", approver=who, reject_reason=a.reason.strip()[:500])
    return {"id": pid, "status": "rejected"}


@app.post("/api/proposals/{pid}/generate")
def proposal_generate(pid: int, g: GenerateIn, x_token: str = Header(default="")):
    """The generator receives the approved spec text, the category, a count,
    a format and a style constraint. It receives no benchmark item in any
    form — see proposals.generation_requests, and the test that reads the
    recorded request bodies to prove it."""
    _check_token(x_token)
    backend = _require_llm()
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    if r["status"] != "approved":
        raise HTTPException(409, f"proposal #{pid} is {r['status']}; only an approved spec "
                                 f"reaches the generator")
    who = _name(g.requester, "generating")
    if g.fmt not in prop.FORMATS:
        raise HTTPException(422, f"fmt must be one of {', '.join(prop.FORMATS)} — question-"
                                 f"shaped training data teaches the exam more readily than "
                                 f"prose does, so 'doc' is the default and 'mc' is retired")
    if not 1 <= g.count <= 1000:
        raise HTTPException(422, "count must be between 1 and 1000")
    why = prop.quota_blocked()
    if why:
        raise HTTPException(507, why)
    n_items = -(-g.count // prop.items_per_request(g.fmt))
    _spend_check(n_items)
    spec = r["edited_text"] or r["spec_text"]
    did = db.dataset_create(pid, g.fmt, g.count, who, {})
    # the audience travels with the topic: a spec alone never said who asks
    audience = prop.audience_for(r["category"], r["task"])
    reqs = prop.generation_requests(did, spec, r["category"], g.count, g.fmt, seed=did,
                                    audience=audience)
    sha = llm.prompt_sha(*[q.system + "\n" + q.user for q in reqs])
    try:
        bid = backend.submit(reqs)
    except llm.LLMError as e:
        db.dataset_update(did, status="failed", finished_at=time.time(), error=str(e)[:400])
        raise HTTPException(502, f"the LLM batch could not be submitted: {e}") from None
    db.batch_add(bid, "generation", did, len(reqs), backend.name, backend.model)
    # what the generator was actually told about its reader, kept for the
    # poller: recomputing it later would read a bank that may have moved
    db.dataset_update(did, batch_id=bid,
                      provenance=json.dumps({"prompt_sha256": sha, "audience": audience}))
    return {"dataset_id": did, "status": "pending", "batch_id": bid, "items": len(reqs)}


def _dataset_view(d: dict, props: dict[int, dict]) -> dict:
    out = dict(d)
    try:
        out["provenance"] = json.loads(d.get("provenance") or "{}")
    except (ValueError, TypeError):
        out["provenance"] = {}
    p = props.get(d["proposal_id"]) or {}
    out.update({"model": p.get("model"), "task": p.get("task"), "category": p.get("category"),
                "download": f"/api/datasets/{d['id']}/items.jsonl" if d["status"] == "ready"
                else None})
    return out


@app.get("/api/datasets")
def dataset_index(limit: int = 200):
    props = {p["id"]: p for p in db.proposal_list(None, 500)}
    return [_dataset_view(d, props) for d in db.dataset_list(min(limit, 500))]


@app.get("/api/datasets/{did}")
def dataset_detail(did: int):
    d = db.dataset_get(did)
    if not d:
        raise HTTPException(404, "no such dataset")
    return _dataset_view(d, {p["id"]: p for p in db.proposal_list(None, 500)})


@app.get("/api/datasets/{did}/items.jsonl")
def dataset_items(did: int):
    d = db.dataset_get(did)
    if not d:
        raise HTTPException(404, "no such dataset")
    if d["status"] != "ready":
        raise HTTPException(409, f"dataset {did} is {d['status']}" + (f": {d['error']}"
                                                                     if d["error"] else ""))
    path = prop.dataset_dir(did) / "items.jsonl"
    if not path.exists():
        raise HTTPException(404, "items.jsonl is missing on disk")
    return FileResponse(path, media_type="application/x-ndjson",
                        filename=f"dataset-{did}.jsonl")


@app.delete("/api/datasets/{did}")
def dataset_delete(did: int, x_token: str = Header(default="")):
    _check_token(x_token)
    d = db.dataset_get(did)
    if not d:
        raise HTTPException(404, "no such dataset")
    if any(did in link["datasets"] for link in db.taint_links()):
        raise HTTPException(409, "a training run recorded this dataset; its provenance must "
                                 "stay on the record")
    import shutil
    shutil.rmtree(prop.dataset_dir(did), ignore_errors=True)
    db.dataset_update(did, status="deleted", finished_at=time.time())
    return {"deleted": did}


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
         .replace("__BANNER__", "")          # the live board is the real one
         .replace("__CSS__", report.CSS)
         .replace("__DATA__", "null")
         .replace("__JS__SLOT__", report.JS))


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


_NO_DEMO = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>No demo run yet</title></head><body style="font:15px/1.6 system-ui;margin:40px auto;
max-width:46em;padding:0 16px"><h1>No demo run yet</h1>
<p>A demo run writes its own page here — its own exam bank, its own results tree, its own
numbers, none of them on the leaderboard. Run one on the box:</p>
<pre style="background:#f4f4f5;padding:12px;border-radius:6px;overflow:auto">cd $BENCH_ROOT &amp;&amp; python3 aienh/scripts/demo_loop.py \\
    --topic "medicine &amp; health" \\
    --import aienh/eval_tasks/fr/medicine_v2.json \\
    --approver "Dr. Hossein" --model HuggingFaceTB/SmolLM2-360M-Instruct</pre>
<p>It prints this URL when it finishes. See <code>DEMO.md</code> for the rest, including
what a green run does not prove.</p>
<p><a href="/">← the live dashboard</a></p></body></html>"""


@app.get("/demo", response_class=HTMLResponse)
def demo_report():
    """The demo's own dashboard, of the demo's own tree. Deliberately a
    second page rather than a switch on the live one: two trees, two pages,
    one link each way, and no chance of a demo number being read as a result."""
    p = demo_report_path()
    if not p.is_file():
        return HTMLResponse(_NO_DEMO, status_code=404)
    return HTMLResponse(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the friend-facing guide, rendered as a proper page. A focused markdown
# renderer, not a library: it covers exactly what FRIENDS.md uses (headings,
# paragraphs, lists, code fences, bold/italic/inline-code, links, hr) and
# escapes first, so nothing in the file can inject markup.
# ---------------------------------------------------------------------------

_GUIDE_MD = Path(__file__).resolve().parent.parent / "FRIENDS.md"
_CLIENT_PY = Path(__file__).resolve().parent.parent / "clients" / "bench_client.py"

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"<i>\1</i>"),
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+|/[^)\s]*)\)"),
     r'<a href="\2" rel="noopener">\1</a>'),
]


def _md_inline(escaped: str) -> str:
    for rx, rep in _INLINE:
        escaped = rx.sub(rep, escaped)
    return escaped


def _slug(heading: str) -> str:
    """A heading's anchor, the way every markdown host makes one — so the
    page can link to a section of the guide and land on it."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _md_to_html(text: str) -> str:
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []
    code: list[str] | None = None

    def flush():
        if para:
            out.append("<p>" + _md_inline(html.escape(" ".join(para))) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            flush()
            if code is None:
                code = []
            else:
                out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("## "):
            flush(); out.append(f"<h2 id=\"{_slug(s[3:])}\">"
                                + _md_inline(html.escape(s[3:])) + "</h2>")
        elif s.startswith("# "):
            flush(); out.append(f"<h1 id=\"{_slug(s[2:])}\">"
                                + _md_inline(html.escape(s[2:])) + "</h1>")
        elif s == "---":
            flush(); out.append("<hr>")
        elif s.startswith("- "):
            if para:
                flush()
            items.append(_md_inline(html.escape(s[2:])))
        elif items:
            items[-1] += " " + _md_inline(html.escape(s))   # wrapped list item
        else:
            para.append(s)
    flush()
    if code is not None:                                    # unclosed fence
        out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
    return "".join(out)


_GUIDE_CSS = """
:root{color-scheme:light dark}
body{margin:0;background:#f9f9f7;color:#0b0b0b;
  font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
@media(prefers-color-scheme:dark){body{background:#0d0d0d;color:#eee}
  main{background:#1a1a19!important;border-color:rgba(255,255,255,.1)!important}
  pre,code{background:#0d0d0d!important;border-color:rgba(255,255,255,.12)!important}
  a{color:#3987e5!important}hr{border-color:#383835!important}h1,h2{color:#fff}}
main{max-width:780px;margin:28px auto;padding:34px 38px;background:#fcfcfb;
  border:1px solid rgba(11,11,11,.1);border-radius:14px}
h1{font-size:26px;letter-spacing:-.01em;margin:0 0 6px}
h2{font-size:18px;margin:30px 0 8px;letter-spacing:-.01em}
p{margin:10px 0}ul{margin:8px 0;padding-left:22px}li{margin:5px 0}
code{background:#f0efec;border:1px solid rgba(11,11,11,.08);border-radius:5px;
  padding:1px 5px;font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#f0efec;border:1px solid rgba(11,11,11,.08);border-radius:10px;
  padding:14px 16px;overflow-x:auto}
pre code{background:none;border:0;padding:0;font-size:13px;line-height:1.55}
a{color:#2a78d6;text-decoration:none}a:hover{text-decoration:underline}
hr{border:0;border-top:1px solid #e1e0d9;margin:26px 0}
"""


@app.get("/guide", response_class=HTMLResponse)
def guide():
    text = (_GUIDE_MD.read_text(encoding="utf-8")
            if _GUIDE_MD.exists() else "# Guide missing\nFRIENDS.md not found in this build.")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Team benchmark — guide</title><style>" + _GUIDE_CSS + "</style></head>"
            "<body><main>" + _md_to_html(text) + "</main></body></html>")


@app.get("/client")
def client_file():
    """The one-file training client, served from the service itself — friends on
    the tailnet grab it with `curl -O http://…:8899/client` and never need
    access to the git repo (which may be private)."""
    if not _CLIENT_PY.exists():
        raise HTTPException(404, "bench_client.py not found in this build")
    return PlainTextResponse(
        _CLIENT_PY.read_text(encoding="utf-8"),
        media_type="text/x-python",
        headers={"Content-Disposition": 'attachment; filename="bench_client.py"'})
