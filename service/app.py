"""The HTTP layer: a submit-and-watch API plus the live dashboard.

    uvicorn service.app:app --host <tailscale-ip> --port 8899

Serving GET / reuses the dashboard from scripts/report_lm_eval.py with the data
slot left empty — the page then fetches /api/results and polls /api/submissions,
so the exact same charts run live here and frozen in the emailed report file.
Bind to the Tailscale IP: the tailnet is the auth boundary; nothing here should
face the open internet.
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import config, db, llm, llm_poller, worker
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
        return (0, 0.0, db.taint_stamp())
    files = [f for pat in _WATCH for f in config.OUT_DIR.rglob(pat)]
    return (len(files), max((f.stat().st_mtime for f in files), default=0.0),
            db.taint_stamp())


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
    if s.suite == "judged":
        why = config.judged_blocked()
        if why:
            raise HTTPException(503, why)
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
        if row["hf_id"] == hf_id and row["status"] in ACTIVE:
            return {"id": row["id"], "status": row["status"],
                    "note": "already in the queue — joining the existing run"}
    sid = db.add(hf_id, s.kind, s.suite, s.submitter.strip()[:80], s.note.strip()[:200],
                 allow_remote_code=s.allow_remote_code)
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


@app.post("/api/exam/build")
def exam_build_tasks(x_token: str = Header(default="")):
    """Write the harness tasks from the bank (+ the MMLU control set). No GPU;
    a curator does this after a round of accepting so suite=judged runs the
    new questions."""
    _check_token(x_token)
    m = exam_build.build(config.OUT_DIR, config.EXAM_DIR)
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
                                _judge_rubric(task))
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
    reqs = prop.generation_requests(did, spec, r["category"], g.count, g.fmt, seed=did)
    sha = llm.prompt_sha(*[q.system + "\n" + q.user for q in reqs])
    try:
        bid = backend.submit(reqs)
    except llm.LLMError as e:
        db.dataset_update(did, status="failed", finished_at=time.time(), error=str(e)[:400])
        raise HTTPException(502, f"the LLM batch could not be submitted: {e}") from None
    db.batch_add(bid, "generation", did, len(reqs), backend.name, backend.model)
    db.dataset_update(did, batch_id=bid, provenance=json.dumps({"prompt_sha256": sha}))
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
         .replace("__CSS__", report.CSS)
         .replace("__DATA__", "null")
         .replace("__JS__SLOT__", report.JS))


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


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
            flush(); out.append("<h2>" + _md_inline(html.escape(s[3:])) + "</h2>")
        elif s.startswith("# "):
            flush(); out.append("<h1>" + _md_inline(html.escape(s[2:])) + "</h1>")
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
