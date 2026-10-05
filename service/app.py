"""The HTTP layer: a submit-and-watch API plus the live dashboard.

    uvicorn service.app:app --host <tailscale-ip> --port 8899

Serving GET / reuses the dashboard from scripts/report_lm_eval.py with the data
slot left empty — the page then fetches /api/results and polls /api/submissions,
so the exact same charts run live here and frozen in the emailed report file.
Bind to the Tailscale IP: the tailnet is the auth boundary; nothing here should
face the open internet.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
                               StreamingResponse)
from pydantic import BaseModel

from . import (ai_models, builder, chat, config, db, disk, hfmeta, judge_test, llm, llm_poller,
               startup, suggest, worker)
from . import playground
from . import api_v1, downloads, gguf, gpu, phone, reported, served, sizes, uploads
from . import proposals as prop
from . import reader

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
    try:                         # records older code left wrong: once per database
        line = llm_poller.repair_once()
        if line:
            print(f"[judge] restored from the run records (once): {line}")
    except Exception as e:       # noqa: BLE001 — a repair must never stop the service starting
        print(f"[judge] could not restore the judge records, will retry next start: {e!r}")
    llm.startup_check()          # a set-but-broken LLM config fails here, not at a click
    worker.start()
    llm_poller.start()
    chat.start_janitor()         # 12d.1: an idle chat model unloads
    yield
    llm_poller.stop()
    worker.stop()


app = FastAPI(title="benchmark service", lifespan=lifespan)
# 12b.3: the scores payload is the board's biggest answer — 1.4 MB of JSON on
# the live tree, a minute to arrive over the tailnet, and the page draws no
# scores until it has — and JSON compresses several times over. Nothing here
# streams, so compressing whole answers holds nothing back.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def _stamp_build(request: Request, call_next):
    """Every /api/* answer says which build answered it. A page loaded before
    a deploy compares this with its own <meta name="evalboard-build"> and
    offers to reload: a page never reloads its own JavaScript, and one left
    open across #33 ran the old refresh code until someone noticed."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["X-Evalboard-Build"] = BUILD
    return response


# ---------------------------------------------------------------------------
# results payload, cached against the tree's shape
# ---------------------------------------------------------------------------

_cache: dict = {"key": None, "payload": None, "at": 0.0}


# Files whose appearance or rewrite changes what the dashboard should show.
# diagnose.json belongs here as much as results*.json does: scripts/diagnose.py
# writes it long after the eval finished, and a key that ignores it means the
# payload keeps being served from cache with no diagnosis in it.
_WATCH = ("results*.json", "diagnose.json", "model_meta.json", "judge.json",
          "judge_calibration.json", "everyday.json",
          # 12k.2: the judge's marks on Do-Not-Answer and XSTest land after the run
          "safety.json",
          # 12n.2: and its grades on SimpleQA Verified
          "simpleqa.json",
          # 12o.3: MobileAIBench's scores, written after the run
          "mobileaibench.json",
          # 12q.C: a DeviceMark run's row, pilot, parity check, speed test and device speed
          "devicemark*.json")


def files_stamp(files) -> str:
    """12a.8: what a set of files is now — each one's path, time to the
    nanosecond and size, hashed. Any file rewritten changes it, whatever the
    others' times: the newest time alone missed a re-mark once a file dated
    in the future (copied, unpacked or from another clock) was among them"""
    h = hashlib.sha256()
    for f in sorted(files):
        try:
            st = f.stat()
        except OSError:
            continue
        h.update(f"{f}\0{st.st_mtime_ns}\0{st.st_size}\n".encode("utf-8"))
    return h.hexdigest()

# 11h: the dashboard no longer links to, serves or reads anything of the
# demo tree ($BENCH_ROOT/demo). scripts/demo_loop.py stays a command-line
# tool that writes its own page there; DEMO.md says so.

def current_fingerprints() -> dict[str, str] | None:
    """task -> the question set it holds now (None: no exam built here). A
    judged result counts only when it was graded on exactly that set
    (judge.split_by_bank)."""
    return exam_build.current_fingerprints(config.JUDGED_TASKS_DIR)


def judge_now(model: str) -> tuple[dict | None, list[dict]]:
    """A model's judge.json as everything but its model page reads it: only
    the topics graded on the question set the task holds now, and the rest
    as history. Answers, proposals and the gate go through here."""
    import judge as _judge
    p = config.OUT_DIR / model.replace("/", "__") / "judge.json"
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    return _judge.split_by_bank(j, current_fingerprints(), _mtime(p))


# Is the grading model answering, now? #50 answered law (37 GPU-seconds) and
# then failed: vLLM had crashed three hours earlier and nothing was on the
# port. A local judge is asked GET /v1/models with a two-second limit before a
# judged run is queued, and the answer is kept for thirty seconds, so the page
# can poll it and a burst of submissions asks once. An API judge is not
# probed: its batches wait for the provider rather than fail.
JUDGE_HEALTH_TTL = 30.0
JUDGE_HEALTH_TIMEOUT = 2.0
_JUDGE_HEALTH: dict = {"at": 0.0, "value": None}


def judge_health(force: bool = False) -> dict:
    """{ok, checked, provider, url, why}: `why` is a plain sentence naming
    the URL that did not answer."""
    import socket
    import urllib.error
    import urllib.parse
    import urllib.request
    now = time.time()
    if not force and _JUDGE_HEALTH["value"] and now - _JUDGE_HEALTH["at"] < JUDGE_HEALTH_TTL:
        return _JUDGE_HEALTH["value"]
    import judge as _judge
    # 12i.1: the judge the AI models page chose, when it chose one
    prov = _judge.identity()["provider"]
    if prov != "local" or _judge.is_stub():
        v = {"ok": True, "checked": False, "provider": prov, "url": "", "why": ""}
    else:
        url = config.LOCAL_BASE_URL.rstrip("/")
        try:
            req = urllib.request.Request(url + "/models", headers=(
                {"Authorization": f"Bearer {config.JUDGE_API_KEY}"} if config.JUDGE_API_KEY else {}))
            with urllib.request.urlopen(req, timeout=JUDGE_HEALTH_TIMEOUT) as r:
                ok = 200 <= r.status < 300
            why = "" if ok else f"the grading model at {url} answered {r.status}"
        except urllib.error.HTTPError as e:
            ok, why = False, f"the grading model at {url} answered {e.code}"
            if e.code in (401, 403):
                # 12r: its server wants a key the board doesn't send (or not this one)
                why += (": it wants an API key. Set JUDGE_API_KEY in .env to the key its "
                        "server was started with, then `sudo docker compose up -d`")
        except Exception as e:                          # noqa: BLE001 — down is down
            reason = getattr(e, "reason", None) or e
            ok, why = False, f"the grading model isn't answering at {url} ({reason})"
            if isinstance(reason, socket.gaierror):
                # 12r: its name doesn't resolve: the board isn't on its Docker network
                host = urllib.parse.urlsplit(url).hostname or url
                why = (f"the grading model's hostname {host} doesn't resolve from the board's "
                       f"container ({reason.strerror or reason}): the board isn't on its Docker "
                       f"network, {config.JUDGE_NETWORK}. To join it for good, put "
                       "COMPOSE_FILE=docker-compose.yml:docker-compose.judge.yml in .env and "
                       "run `sudo docker compose up -d`. Until then: `sudo docker network "
                       f"connect {config.JUDGE_NETWORK} $(sudo docker compose ps -q bench)`. "
                       "Neither touches the judge's own container")
        v = {"ok": ok, "checked": True, "provider": prov, "url": url, "why": why}
    _JUDGE_HEALTH.update(at=now, value=v)
    return v


@app.get("/api/proposals/{pid}/focus")
def proposal_focus(pid: int, count: int = 20):
    """Where the documents would go. Before Approve, the plan as it stands —
    the proposal card shows it beside the spec, so the spread is part of what
    is approved. After Approve, the plan Approve froze. A proposal approved
    before 11e has no frozen plan, and says so: it keeps 11a's behaviour."""
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, f"no proposal {pid}")
    if not 1 <= count <= 1000:
        raise HTTPException(422, "count must be between 1 and 1000")
    base = {"proposal_id": pid, "count": count}
    if r["status"] == "approved":
        frozen = prop.frozen_focus(r)
        if frozen is None:
            plan = prop.focus_plan(config.OUT_DIR / r["model"].replace("/", "__"), r["task"],
                                   r["category"], count)
            return {**base, "frozen": False, "legacy": True, "mode": "area" if plan else "off",
                    "labels": [f["domain"] for f in plan for _ in range(f["documents"])],
                    "reason": "" if plan else prop.NO_LABELS, "plan": plan}
        return {**base, "frozen": True, "legacy": False, "mode": frozen.get("mode") or "off",
                "labels": frozen.get("labels") or [], "reason": frozen.get("reason") or ""}
    f = _focus_live(r)
    return {**base, "frozen": False, "legacy": False, "mode": f["mode"] or "off",
            "labels": f["labels"], "reason": f["reason"], "failing": f["failing"]}


@app.get("/api/judge/health")
def judge_health_endpoint():
    """For the header's live dot and the judged controls: cheap to poll, the
    probe itself runs at most every thirty seconds."""
    return judge_health()


def _judge_identity() -> dict:
    """the judge this server runs now — 12i.1: with its version's key, which
    decides what judged scores the page shows today"""
    import judge as _judge
    ident = _judge.identity()
    return {**ident, "version": _judge.version(ident)["key"]}


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
    # and so is the built exam: a rebuild with other questions turns every
    # judged result on the old ones into history without touching a judge.json
    exam = _mtime(config.JUDGED_TASKS_DIR / "manifest.json")
    # 12i.1: and the judge version this server runs: a new judge moves every
    # judged score marked by another into History without touching a file
    judge_v = _judge_identity().get("version")
    # 12f.3: and what the host's GGUF worker has written
    gd = config.RESULTS_ROOT / "gguf_results"
    gg = [f for f in gd.glob("*.json")] if gd.is_dir() else []
    ggs = (len(gg), files_stamp(gg), _mtime(config.RESULTS_ROOT / "gguf_data" / "manifest.json"))
    if not config.OUT_DIR.is_dir():
        return (0, 0.0, db.taint_stamp(), exam, judge_v, ggs)
    files = [f for pat in _WATCH for f in config.OUT_DIR.rglob(pat)]
    return (len(files), files_stamp(files), db.taint_stamp(), exam, judge_v, ggs)


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
    marks = {r["id"]: prop.override_of(r) for r in db.proposal_list(limit=500)}
    out: dict[str, dict] = {}
    for mid in model_ids:
        for link in links:
            pre = link["hf_prefix"]
            if mid in link["checkpoints"] or (pre and (mid == pre or mid.startswith(pre))):
                ds = list(link["datasets"])
                pids = sorted({props[d] for d in ds if d in props})
                out[mid] = {"run_id": link["run_id"], "datasets": ds, "proposals": pids}
                # taint follows the data; so does the mark on how it was proposed
                over = {str(p): marks[p] for p in pids if marks.get(p)}
                if over:
                    out[mid]["over_provisional_judge"] = over
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


def _claims(mid: str, run: dict) -> bool:
    """12g.1: a run claims a checkpoint it logged, or one whose id is its
    hf_prefix — alone, or followed by a separator. The taint join above takes
    any id that starts with the prefix, so "run7" also claims "run70"; what a
    model was trained from needs the stricter rule."""
    pre = run["hf_prefix"]
    if mid in run["checkpoints"]:
        return True
    return bool(pre) and (mid == pre or (mid.startswith(pre) and mid[len(pre)] in "-_/."))


def bases_for(model_ids) -> dict[str, dict]:
    """model id -> {base, run}: what the latest training run that claims it
    recorded it started from (12g.1)"""
    runs = db.trun_bases()
    out: dict[str, dict] = {}
    for mid in model_ids:
        for run in reversed(runs):
            if _claims(mid, run) and run["parent"] != mid:
                out[mid] = {"base": run["parent"], "run": run["run_id"]}
                break
    return out


def trained_from_for(model_ids) -> dict[str, dict]:
    """model id -> what it was trained from: a person's word, else the
    training run's record (12g.1). Retests pair a checkpoint with this."""
    people = db.trained_from_all()
    out = {m: {"base": a["base"], "source": "run", "run": a["run"]}
           for m, a in bases_for(model_ids).items()}
    for m in model_ids:
        if m in people and people[m]["base"] != m:
            out[m] = {"base": people[m]["base"], "source": "person", "by": people[m]["by"],
                      "at": people[m]["at"]}
    return out


def exam_gate() -> None:
    """16.5: the Knowledge exam's own endpoints answer, while KNOWLEDGE_EXAM=0,
    that it is switched off — nothing is read, run or written"""
    if not config.KNOWLEDGE_EXAM:
        raise HTTPException(409, config.EXAM_OFF + ".")


EXAM_ONLY = [Depends(exam_gate)]


def exam_task(task: str | None) -> bool:
    """16.5: a proposal's or a dataset's target is a Knowledge exam topic"""
    return str(task or "").startswith(("exam_", "fr_"))


def exam_hidden(task: str | None) -> bool:
    """…and the exam is switched off: kept, and neither shown nor carried on"""
    return exam_task(task) and not config.KNOWLEDGE_EXAM


def _exam_prop_gate(r: dict) -> None:
    if exam_hidden(r.get("task")):
        raise HTTPException(409, config.EXAM_OFF + ".")


def _exam_dataset_gate(did: int) -> None:
    """16.5: a dataset built from an exam topic's proposal, while it is off"""
    d = db.dataset_get(did)
    if d and not config.KNOWLEDGE_EXAM:
        _exam_prop_gate(db.proposal_get(d["proposal_id"]) or {})


def results_payload() -> dict:
    now = time.time()
    if _cache["payload"] is not None and now - _cache["at"] < 5:
        return _cache["payload"]
    key = _tree_key()
    if key != _cache["key"] or _cache["payload"] is None:
        runs = report.load_results(config.OUT_DIR) if config.OUT_DIR.is_dir() else []
        by_model = report.merge_runs(runs)
        # 12g.1: what each checkpoint was trained from is also the "before" of
        # what its training taught: a person's word first, then the run's
        trained = trained_from_for(list(by_model.keys()))
        parents = {**parents_for(by_model.keys()),
                   **{m: t["base"] for m, t in trained.items()}}
        served_map = report.load_served(config.OUT_DIR)
        payload = report.build_payload(by_model, config.TITLE, source=str(config.OUT_DIR),
                                       taint=taint_for(by_model.keys()),
                                       parents=parents,
                                       calibration=_calibration(),
                                       judge_identity=_judge_identity(),
                                       fingerprints=current_fingerprints(),
                                       everyday=report.load_everyday(config.OUT_DIR),
                                       served=served_map,
                                       gguf=report.load_gguf(config.RESULTS_ROOT, config.OUT_DIR,
                                                             served_map),
                                       # 16.1: what people entered, and what the files say
                                       sizes={"entered": db.sizes_all(),
                                              "files": report.load_sizes(config.RESULTS_ROOT)})
        payload["live"] = True
        # 12q.C: each model's DeviceMark runs, for its page and "Open results"
        try:
            payload["devicemark"] = _dm().model_runs(config.OUT_DIR, served.launch_of_id)
        except Exception as e:                      # noqa: BLE001 — the board still loads
            payload["devicemark"] = {"_error": f"the DeviceMark runs couldn't be read: {e}"[:300]}
        # 12g.2: the hidden questions an Everyday group needs before Improve takes it
        if payload.get("everyday"):
            payload["everyday"]["minHidden"] = config.EVERYDAY_MIN_HIDDEN
            # 12n.1: who may open a group's hidden half (and nobody else sees a way to)
            payload["everyday"]["owner"] = config.BOARD_OWNER
        # the loop's audit trail, per tainted model: run, datasets, proposals
        trails = trail_for([m["id"] for m in payload["models"]])
        for m in payload["models"]:
            if trails.get(m["id"]):
                m["taintTrail"] = trails[m["id"]]
            if trained.get(m["id"]):
                m["trainedFrom"] = trained[m["id"]]
        _cache.update(key=key, payload=payload)
    # 16.3: each model's latest finished run of any kind, for Models' Tested —
    # from the runs table, so a run whose files didn't move the tree counts too
    done = db.last_done(judged=config.KNOWLEDGE_EXAM)
    for m in _cache["payload"]["models"]:
        if done.get(m["id"]):
            m["testedAt"] = done[m["id"]]
    _cache["at"] = now
    return _cache["payload"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SubmissionIn(BaseModel):
    hf_id: str
    kind: str = "auto"
    suite: str = "full"                # quick | full | control (the mmlu_perm experiment) | everyday
    submitter: str = ""
    note: str = ""
    allow_remote_code: bool = False    # execute the upload's own modeling code
    # narrow a judged run to these built exam tasks. Empty means the whole
    # suite, which is what it has always meant — one topic at a time is the
    # loop's unit of work, and a person should not have to sit fifteen.
    tasks: list[str] = []
    # 12h.1, the generative suite only: think before answering (a model with
    # a switch; its run is a row of its own), and a seeded MMLU-Pro subset
    # of this many items (0: all 12,032 — the only run comparable to
    # published numbers). 12o.1: thinking in the shared suite too — GPQA
    # measured the way Epoch runs it, with reasoning
    thinking: bool = False
    subset: int = 0
    # 12k.2, the full suite only: BBQ's 29,246 ambiguous questions instead of
    # the seeded 3,000
    bbq_all: bool = False
    # 12q, the devicemark suite only: which part — the battery (full), its
    # 30-item pilot, the MTP parity check, or the speed test (the last three
    # for a served setup) — and, for the parity check, the setup without MTP
    # (hf_id is the one with it)
    part: str = ""
    pair: str = ""


ACTIVE = ("queued", "preflight", "waiting_gpu", "waiting_lock", "running")


@app.post("/api/submissions")
def submit(s: SubmissionIn, x_token: str = Header(default="")):
    if config.SUBMIT_TOKEN and x_token != config.SUBMIT_TOKEN:
        raise HTTPException(401, "bad or missing X-Token header")
    why = disk.blocks_run()                  # 12f.0: a run can't save to a full disk
    if why:
        raise HTTPException(409, why)
    if s.suite == "everyday":
        why = _everyday_paused()             # 12p.1: not without its hidden set
        if why:
            raise HTTPException(409, why + ". Nothing was queued.")
    hf_id = s.hf_id.strip()
    if not _HF_ID_RE.match(hf_id):
        raise HTTPException(422, "model id must look like org/name — a Hugging Face repo "
                                 "id, or local/<name> for an uploaded artifact")
    if s.kind not in ("auto", "base", "instruct"):
        raise HTTPException(422, "kind must be auto, base or instruct")
    # 12q.D: a model this server can't run is refused here, saying why, never queued
    why = hfmeta.catalog.cant_run_here(hf_id)
    if why:
        raise HTTPException(422, f"{hf_id} can't run on this server: {why}. Nothing was queued.")
    if s.suite not in config.SUITES:
        raise HTTPException(422, "suite must be quick, full, control (mmlu_perm only), "
                                 "judged (free response + judge), everyday (Everyday tasks), "
                                 "generative (IFEval, MMLU-Pro, MATH-500), safety "
                                 "(Do-Not-Answer, XSTest), shared (GPQA Diamond, "
                                 "SimpleQA Verified), mobile (MobileAIBench's text sets; "
                                 "part judged: MT-Bench) or devicemark (DeviceMark's battery)")
    if s.suite not in ("generative", "shared", "devicemark") and s.thinking:
        raise HTTPException(422, "thinking is for IFEval, MMLU-Pro and MATH-500 (suite "
                                 "generative), GPQA Diamond and SimpleQA Verified (suite "
                                 "shared) and DeviceMark's battery (suite devicemark) only")
    part = (s.part or "").strip().lower()
    pair = (s.pair or "").strip()
    # 14.1: the mobile suite's two parts — none (no judge), or judged (MT-Bench)
    if s.suite == "mobile" and part not in config.MAB_PARTS:
        raise HTTPException(422, "the mobile suite's part is judged (MT-Bench), trust "
                                 "(Adversarial Instruction, Privacy Leakage, Social Chemistry "
                                 "101), mmlu (Mobile-MMLU-Pro), mmlu_full (the full "
                                 "Mobile-MMLU, non-commercial), or none for its five sets with "
                                 "no judge")
    if s.suite == "mobile" and pair:
        raise HTTPException(422, "pair is for the devicemark suite only")
    if s.suite == "mobile" and part == "trust":
        # 14.2: Privacy Leakage is fetched at deploy (scripts/fetch_data.py)
        why = next((_mab().available(t) for t in _mab().TRUST if _mab().available(t)), "")
        if why:
            raise HTTPException(422, why + ". Nothing was queued.")
    if s.suite == "mobile" and part == "mmlu" and _mmp().available():
        # 14.3: Mobile-MMLU-Pro is fetched at deploy too
        raise HTTPException(422, _mmp().available() + ". Nothing was queued.")
    if s.suite == "mobile" and part == "mmlu_full":
        # 14.4: and the full set, with Pro (its run asks Pro's wording of 9933ec55)
        why = _mmp().available() or _mmp().full_available()
        if why:
            raise HTTPException(422, why + ". Nothing was queued.")
    if s.suite not in ("devicemark", "mobile") and (part or pair):
        raise HTTPException(422, "pair is for the devicemark suite only, and part for it and "
                                 "the mobile suite")
    if s.suite == "devicemark":
        why = _devicemark_check(s.hf_id.strip(), s.kind, part or "full", pair, s.thinking)
        if why:
            raise HTTPException(422, why + " Nothing was queued.")
        part = part or "full"
    if s.suite != "generative" and s.subset:
        raise HTTPException(422, "subset is for MMLU-Pro (suite generative) only")
    if s.suite == "generative" and s.kind == "base":
        raise HTTPException(422, config.GEN_INSTRUCT_ONLY + ". Nothing was queued.")
    if s.bbq_all and s.suite != "full":
        raise HTTPException(422, "bbq_all asks all of BBQ in the full suite; the other suites "
                                 "don't ask BBQ")
    if s.suite == "safety" and s.kind == "base":
        raise HTTPException(422, config.SAFETY_INSTRUCT_ONLY + ". Nothing was queued.")
    if s.suite == "shared" and s.kind == "base":
        raise HTTPException(422, config.SHARED_INSTRUCT_ONLY + ". Nothing was queued.")
    # 14.3: Mobile-MMLU-Pro is scored on the letters' log-likelihoods, as the
    # paper ran it: a base model can sit it (14.4: and the full set)
    if s.suite == "mobile" and s.kind == "base" and part not in ("mmlu", "mmlu_full"):
        raise HTTPException(422, config.MAB_INSTRUCT_ONLY + ". Nothing was queued.")
    total = sum(config.MMLU_PRO_SUBJECTS.values())
    if s.subset and not 0 < s.subset < total:
        raise HTTPException(422, f"subset is a number of MMLU-Pro items, from 1 to {total - 1}; "
                                 f"0 runs all {total}")
    chosen: list[str] = []
    # 16.5: the Knowledge exam switched off: nothing about it is queued
    if s.suite == "judged" and not config.KNOWLEDGE_EXAM:
        raise HTTPException(422, config.EXAM_OFF + ". Nothing was queued.")
    if s.suite == "judged":
        # before a GPU second is spent on answers nobody could grade
        h = judge_health()
        if not h["ok"]:
            raise HTTPException(503, f"{h['why'][:1].upper()}{h['why'][1:]}, so this judged run "
                                     f"would spend GPU time on answers nobody can grade. Start "
                                     f"it, then queue the run — nothing was queued.")
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
    # 12f.1: a model served elsewhere — registered, and asked only what a chat
    # endpoint can answer. Its server is asked at the start of the run
    srv = served.is_served(hf_id)
    if srv:
        if not served.get(hf_id):
            raise HTTPException(422, f"{hf_id} is not registered: add it under Add a model ▸ "
                                     f"Running on a server. Nothing was queued.")
        if s.suite not in served.SUITES:
            raise HTTPException(422, served.LOGLIK_LINE + " Nothing was queued.")
        if s.thinking and s.suite != "devicemark":
            raise HTTPException(422, "A served model thinks as it was registered: register it "
                                     "again to change that. Nothing was queued.")
        # 12m.3: a model from OpenRouter — a run that would pass this month's
        # AI limit is refused here, before it is queued, with its estimate
        rec = served.get(hf_id)
        if served.is_openrouter(rec):
            if not ai_models.has_key():
                raise HTTPException(409, "OpenRouter has no key on this server "
                                         "(OPENROUTER_API_KEY). Nothing was queued.")
            why = served.over_limit_line(served.estimate(rec, s.suite, chosen, s.subset,
                                                         part=part))
            if why:
                raise HTTPException(409, why + " Nothing was queued.")
    # 11i: a checkpoint that ships its own model code is answered HERE, before
    # anything is queued, in the words the page shows beside its disabled
    # button. #56 learned it at start, after the wait — and its Resubmit had
    # no way to ask
    code = {"own_code": False} if srv else hfmeta.remote_code_check(hf_id)
    if code["own_code"]:
        if code["why"]:
            raise HTTPException(422, code["why"] + " Nothing was queued.")
        if not s.allow_remote_code:
            names = ", ".join(f"{f['file']} (sha {f['sha']})" for f in code["files"])
            raise HTTPException(422, f"{hf_id} ships its own model code ({names}). Running "
                                     f"it runs that Python, as the unprivileged "
                                     f"{code['user']} user: tick \"Run this model's own "
                                     f"model code\" (allow_remote_code=true) to queue it. "
                                     f"Nothing was queued.")
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
    # 16b review: "local/.." named BENCH_ROOT here, and was queued
    if hf_id.startswith("local/") and uploads.artifact_dir(hf_id.split("/", 1)[1]) is None:
        # on the board is not on this server: #53 queued a checkpoint whose
        # results came from elsewhere, and the run failed at start, after the
        # person had waited for it. Nothing is queued now instead — and a
        # checkpoint that is not on the board either is refused in preflight's
        # own words, so the two never disagree
        name = hf_id.split("/", 1)[1]
        if (config.OUT_DIR / hf_id.replace("/", "__")).is_dir():
            raise HTTPException(422, f"local/{name} has results on this board but no weights "
                                     f"on this server — upload it first "
                                     f"(POST /api/artifacts/{name}) to run it here.")
        raise HTTPException(422, f"no uploaded artifact named {name!r} — upload it first "
                                 f"(POST /api/artifacts/{name}) or check the name.")
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
        # 12h.1: thinking on is another row, and a subset another run
        same = same and bool(row.get("thinking")) == s.thinking \
            and int(row.get("subset") or 0) == s.subset \
            and bool(row.get("bbq_all")) == s.bbq_all \
            and (row.get("part") or "") == part and (row.get("pair") or "") == pair
        if row["suite"] == s.suite and same:
            return {"id": row["id"], "status": row["status"],
                    "note": "already in the queue — joining the existing run"}
    sid = db.add(hf_id, s.kind, s.suite, s.submitter.strip()[:80], s.note.strip()[:200],
                 allow_remote_code=s.allow_remote_code, tasks=chosen,
                 thinking=s.thinking, subset=s.subset, bbq_all=s.bbq_all,
                 part=part, pair=pair)
    # 14.4.4: a restricted set's run says so as it is queued
    rx = next((e for e in (_restrictions().of(t) for t in config.tasks_for_suite(
        s.suite, part=part)) if e), None) if s.suite == "mobile" else None
    return {"id": sid, "status": "queued", "tasks": sorted(chosen),
            **({"part": part} if part else {}), **({"pair": pair} if pair else {}),
            **({"restriction": rx} if rx else {})}


def _devicemark_check(hf_id: str, kind: str, part: str, pair: str, thinking: bool) -> str:
    """12q: '' when this devicemark run can be queued, else why not"""
    if part not in config.DM_PARTS:
        return f"part must be one of {', '.join(config.DM_PARTS)}."
    if kind == "base":
        return config.DM_INSTRUCT_ONLY + "."
    srv = served.is_served(hf_id)
    rec = served.get(hf_id) if srv else None
    if rec and served.is_openrouter(rec):
        return ("DeviceMark's battery here is for our own served setups and Hugging Face "
                "models, not a model from OpenRouter.")
    if not srv and part != "full":
        return (f"The {part} is for a served setup: a Hugging Face model sits the whole "
                "battery (part full).")
    if part == "speed" and thinking:
        return "The speed test decodes a fixed prompt: thinking doesn't apply to it."
    if part == "parity":
        if not pair:
            return ("The parity check needs pair: the setup without MTP (hf_id is the one "
                    "with it).")
        other = served.get(pair)
        if not served.is_served(pair) or not other or served.is_openrouter(other):
            return f"pair {pair} is not one of our registered served setups."
        if pair == hf_id:
            return "pair is the same setup as hf_id: the parity check compares two."
    elif pair:
        return "pair is for the parity check only."
    return ""


@app.get("/api/submissions/count")
def submissions_count():
    """12z A6: how many runs there are — the Runs list shows the newest 100
    and pages back from there"""
    return {"total": db.count_submissions()}


@app.get("/api/submissions")
def submissions(limit: int = 100, before: int | None = None):
    """The queue. A judged row carries the judge batch with it: the answers
    are on disk long before the grades are, and 'done' on the GPU half is not
    done — the row should say which topics it sat and how far the judge is.
    12z A6: `before` is the next page back — the runs older than that one"""
    gguf.sync()                           # 12f.3: the host worker's progress, into its rows
    rows = db.recent(max(1, min(limit, 500)), before)
    # 12f.5: what a finished GGUF run didn't finish, for Re-run failed benchmarks
    for r in rows:
        if r["suite"] == "gguf" and r["status"] in ("failed", "canceled"):
            r["gguf_left"] = gguf.not_done(r["id"])
    # 12a: a pilot row waits on the judge for one question, and says so the
    # way a judged row does — by the batch it recorded, and nothing else
    # 12k.2: and a Trust & safety row the same way, by its batch
    pilot = [r for r in rows if r["suite"] in ("everyday", "safety", "shared")
             and r.get("judge_batch")]
    if pilot:
        batches = {b["batch_id"]: b for b in db.batches_list(500)
                   if b["kind"] in ("everyday", "safety", "simpleqa")}
        for r in pilot:
            b = batches.get(r["judge_batch"])
            if b:
                r["judge"] = {"batch_id": b["batch_id"], "n_items": b["n_items"],
                              "status": b["status"], "progress": b.get("progress") or "",
                              "error": b.get("error") or ""}
    judged = [r for r in rows if r["suite"] == "judged"]
    if judged:
        runs = {run["batch_id"]: run for run in db.judge_runs(200)}
        batches = {b["batch_id"]: b for b in db.batches_list(500) if b["kind"] == "judge"}
        # by the batch THIS submission recorded when it submitted it, and by
        # nothing else. Any match on the model — even "the newest run that
        # started after this row" — gives every older row of that model the
        # newest run's batch: #45, #46 and #47 all showed #47's 130/130. A
        # row with no batch of its own shows no judge line.
        for r in judged:
            run = runs.get(r.get("judge_batch") or "")
            if not run:
                continue
            b = batches.get(run["batch_id"]) or {}
            r["judge"] = {"batch_id": run["batch_id"], "n_items": run["n_items"],
                          "status": b.get("status") or run.get("status"),
                          "progress": b.get("progress") or "",
                          "judge_id": run.get("judge_id"),
                          "error": b.get("error") or run.get("error") or ""}
        for r in judged:
            # the answers are on disk and only the grading failed: a retry
            # re-grades them (10b's resume answers nothing again)
            r["judge_failed"] = bool(
                (r["status"] == "failed" and str(r.get("error") or "").startswith("judge:"))
                or (r.get("judge") or {}).get("status") == "failed")
    return rows


@app.post("/api/submissions/{sid}/cancel")
def cancel(sid: int):
    """Queued: canceled now. Running: asked to stop — the runner ends the task
    in flight, frees the GPU and marks it canceled; nothing it half-wrote is
    kept, and no judge batch is submitted."""
    st = db.cancel(sid)
    row = db.get(sid) or {}
    if row.get("suite") == "gguf":
        gguf.cancel(sid)                  # 12f.3: the host's worker stops it
    if not st:
        raise HTTPException(409, "only a queued or running submission can be canceled — "
                                 "this one has already finished")
    return {"id": sid, "status": st}


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
    if not uploads.folder_name_ok(name):
        raise HTTPException(422, "artifact name: letters, digits, dot, dash, underscore only, "
                                 "not starting with a dot")
    dest = config.ARTIFACTS_DIR / name
    cap = int(config.ARTIFACT_MAX_GB * 1e9)
    quota = int(config.ARTIFACT_QUOTA_GB * 1e9)
    used = uploads.used()                        # 16b.1: the browser's uploads count too
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
        # 16b.1: unpacked with the browser upload's checks, off the main loop —
        # it stalled every request, chat streams included
        try:
            await asyncio.to_thread(uploads.unpack_zip, tmp, dest, cap)
        except uploads.Refused as e:
            raise HTTPException(e.code, str(e)) from e
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


@app.get("/api/served/up")
def served_up(id: str = ""):
    """16.8: whether a served model's server answers — the Playground's own
    check, asked in the background at most every 30 s: true, false, or null
    before its first answer. Test a model holds Start until it is true"""
    rec = served.get(id)
    if not rec:
        raise HTTPException(404, f"{id} is not registered")
    if served.is_openrouter(rec):
        return {"id": id, "up": True, "why": ""}
    up = chat.served_up(id)
    return {"id": id, "up": up, "why": "" if up else chat.NOT_RUNNING + "."
            if up is False else chat.CHECKING + "."}


# ---------------------------------------------------------------------------
# 16b.1: Add a model ▸ On my computer — a .gguf file, or a model folder as a
# .zip, in pieces that resume (service/uploads.py)
# ---------------------------------------------------------------------------

class UploadIn(BaseModel):
    filename: str
    size: int
    mtime: float = 0
    name: str = ""
    by: str = ""


class UploadAddIn(BaseModel):
    name: str = ""
    based_on: str = ""
    how: str = ""
    total: str | float | None = None
    active: str | float | None = None
    download: bool = True
    by: str = ""


def _refused(e: "uploads.Refused"):
    return HTTPException(e.code, str(e))


@app.get("/api/uploads")
def uploads_index():
    return uploads.storage()


@app.post("/api/uploads")
def upload_start(f: UploadIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return uploads.start(f.model_dump(), f.by)
    except uploads.Refused as e:
        raise _refused(e) from e


_PIECE_LOCKS: dict[str, asyncio.Lock] = {}


@app.put("/api/uploads/{uid}")
async def upload_piece(uid: str, request: Request, offset: int = 0,
                       x_token: str = Header(default="")):
    """one piece at the offset the server has; a cut one keeps what arrived,
    and the next starts there"""
    _check_token(x_token)
    try:
        uploads.get(uid)                          # a well-formed id of an upload, first
    except uploads.Refused as e:
        raise _refused(e) from e
    lock = _PIECE_LOCKS.setdefault(uid, asyncio.Lock())
    if lock.locked():
        raise HTTPException(409, "A piece of this upload is being written: wait for it.")
    async with lock:
        try:
            rec = uploads.at_offset(uid, offset)
        except uploads.Refused as e:
            raise _refused(e) from e
        room = min(uploads.PIECE, rec["size"] - offset)
        got, buf = 0, bytearray()
        fh = await asyncio.to_thread(open, uploads.data_path(uid), "ab")
        try:
            async for chunk in request.stream():
                got += len(chunk)
                if got > room:
                    raise HTTPException(413, f"A piece is at most {room} bytes here.")
                buf += chunk
                if len(buf) >= 4 << 20:
                    await asyncio.to_thread(fh.write, bytes(buf))
                    buf.clear()
            if buf:
                await asyncio.to_thread(fh.write, bytes(buf))
        finally:
            await asyncio.to_thread(fh.close)
            uploads.touch(uid)
    return {"id": uid, "offset": uploads.offset(uid), "size": rec["size"]}


@app.get("/api/uploads/{uid}")
def upload_get(uid: str):
    try:
        return uploads.get(uid)
    except uploads.Refused as e:
        raise _refused(e) from e


@app.post("/api/uploads/{uid}/finish")
def upload_finish(uid: str, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return uploads.finish(uid)
    except uploads.Refused as e:
        raise _refused(e) from e


@app.post("/api/uploads/{uid}/add")
def upload_add(uid: str, f: UploadAddIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        got = uploads.register(uid, f.model_dump(), f.by)
    except uploads.Refused as e:
        raise _refused(e) from e
    _cache.update(key=None, payload=None, at=0.0)
    return got


@app.delete("/api/uploads/{uid}")
def upload_cancel(uid: str, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return uploads.cancel(uid)
    except uploads.Refused as e:
        raise _refused(e) from e


@app.delete("/api/uploads/file/{model_id:path}")
def upload_delete(model_id: str, x_token: str = Header(default="")):
    """a kept upload's file — refused while a run uses it; its results stay"""
    _check_token(x_token)
    try:
        got = uploads.delete(model_id)
    except uploads.Refused as e:
        raise _refused(e) from e
    _cache.update(key=None, payload=None, at=0.0)
    return got


# ---------------------------------------------------------------------------
# 16b.2: Download — a model file on this server, streamed with HTTP Range so
# a download resumes; the board's token; each download logged
# (service/downloads.py)
# ---------------------------------------------------------------------------

class DownloadAllowIn(BaseModel):
    model: str
    allowed: bool
    by: str = ""


class DownloadLinkIn(BaseModel):
    model: str
    by: str = ""


def _dl_refused(e: "downloads.Refused"):
    return HTTPException(e.code, str(e))


@app.get("/api/models/file")
def model_file(id: str = ""):
    """what the model page says: its file's name, size and sha256, whether
    it may be downloaded and by whose word, or why there is no file"""
    got = downloads.info(id)
    return {**got, "log": downloads.log_rows(id, 10)}


@app.post("/api/models/file/allow")
def model_file_allow(f: DownloadAllowIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return downloads.set_allowed(f.model, f.allowed, f.by)
    except downloads.Refused as e:
        raise _dl_refused(e) from e


@app.post("/api/models/file/link")
def model_file_link(f: DownloadLinkIn, x_token: str = Header(default="")):
    """a link for one download from a browser, kept a day — no token in it"""
    _check_token(x_token)
    try:
        return downloads.link(f.model, f.by)
    except downloads.Refused as e:
        raise _dl_refused(e) from e


def _range_start(request: Request) -> int:
    m = re.match(r"bytes=(\d+)-", request.headers.get("range") or "")
    return int(m.group(1)) if m else 0


def _send_file(path: Path, name: str) -> FileResponse:
    # identity: GZip would break a range; the file is streamed from disk
    return FileResponse(path, filename=name, media_type="application/octet-stream",
                        headers={"Content-Encoding": "identity", "Accept-Ranges": "bytes",
                                 "Cache-Control": "no-store"})


@app.get("/api/dl/{tid}/{name}")
def model_file_by_link(tid: str, name: str, request: Request):
    try:
        got = downloads.by_link(tid)
        path, fname = downloads.file_for(got["model"])
    except downloads.Refused as e:
        raise _dl_refused(e) from e
    downloads.logged(got["model"], got["who"], "link", _range_start(request))
    return _send_file(path, fname)


@app.get("/api/download")
def model_file_download(request: Request, model: str = "", x_token: str = Header(default=""),
                        x_who: str = Header(default="")):
    """the same file with the token in a header — curl -C - resumes it. The
    page's command reads the token from $BOARD_TOKEN, never writes it"""
    _check_token(x_token)
    try:
        path, fname = downloads.file_for(model)
    except downloads.Refused as e:
        raise _dl_refused(e) from e
    downloads.logged(model, x_who, "token", _range_start(request))
    return _send_file(path, fname)


# ---------------------------------------------------------------------------
# 16b.3: Use as an API — OpenAI-compatible, behind each person's own key
# (service/api_v1.py). The shared write token makes a key; it is never one
# ---------------------------------------------------------------------------

class ApiKeyIn(BaseModel):
    by: str = ""


def _api_error(e: "api_v1.ApiError") -> JSONResponse:
    return JSONResponse(e.body(), status_code=e.status,
                        headers={"Retry-After": str(e.retry_after)} if e.retry_after else None)


@app.post("/api/keys")
def api_key_make(f: ApiKeyIn, x_token: str = Header(default="")):
    """a key for one person, shown once: the board keeps only its sha256"""
    _check_token(x_token)
    try:
        return api_v1.make_key(f.by)
    except api_v1.ApiError as e:
        return _api_error(e)


@app.get("/api/keys")
def api_keys_list(who: str = ""):
    """each key's who, first characters, counts and last use — never a key"""
    return {"keys": db.apikeys(who), "address": "/v1",
            "max_tokens": config.API_MAX_TOKENS, "wait_s": config.API_WAIT_S}


@app.post("/api/keys/{kid}/revoke")
def api_key_revoke(kid: int, f: ApiKeyIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return api_v1.revoke(kid, f.by)
    except api_v1.ApiError as e:
        return _api_error(e)


@app.get("/api/models/api")
def model_api(id: str = ""):
    """the model page's Use as an API: whether it is offered, and its state"""
    why = api_v1.why_not(id)
    row = chat.model_row(id)
    return {"model": id, "offered": not why, "why": why,
            **({"state": chat.ENGINE.state(row)} if row and not why else {})}


@app.get("/v1/models")
def v1_models(authorization: str = Header(default="")):
    try:
        api_v1.auth(authorization)
    except api_v1.ApiError as e:
        return _api_error(e)
    return api_v1.models_list()


@app.post("/v1/chat/completions")
async def v1_chat(request: Request, authorization: str = Header(default="")):
    try:
        key = api_v1.auth(authorization)
        try:
            body = await request.json()
        except ValueError:
            raise api_v1.ApiError(400, "The body is a JSON object, OpenAI's chat request") \
                from None
        st, ctx, lock = await asyncio.to_thread(api_v1.begin, body, key)
    except api_v1.ApiError as e:
        return _api_error(e)
    if body.get("stream"):
        # identity: the gzip middleware would hold the events back
        return StreamingResponse(api_v1.stream(st, ctx, key, lock), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store",
                                          "Content-Encoding": "identity"})
    try:
        return await asyncio.to_thread(api_v1.complete, st, ctx, key, lock)
    except api_v1.ApiError as e:
        return _api_error(e)


@app.get("/api/models/suggest")
def models_suggest(q: str = ""):
    """What the model-id boxes offer as you type: the models this board knows
    first, then Hugging Face Hub matches (cached, two-second limit). A Hub
    that does not answer leaves the local matches and a footer saying so."""
    arts = []
    if config.ARTIFACTS_DIR.is_dir():
        arts = sorted(d.name for d in config.ARTIFACTS_DIR.iterdir()
                      if d.is_dir() and not d.name.startswith("."))
    local = suggest.local_candidates(results_payload(), db.recent(500), arts,
                                     served.all_public())
    out = suggest.suggest(q[:100], local)
    # 11i: an upload that ships its own model code says so in the list,
    # before it is picked — and whether this server will run it
    for it in out.get("items") or []:
        if it.get("weights") and str(it.get("id", "")).startswith("local/"):
            code = hfmeta.remote_code_check(it["id"])
            if code["own_code"]:
                it["own_code"] = {"runs": not code["why"]}
    return out


# ---------------------------------------------------------------------------
# 12f.1: models served elsewhere. Registered with what their server reports,
# which is pinned; the key is kept here and never returned
# ---------------------------------------------------------------------------

class ServedIn(BaseModel):
    name: str = ""
    base_url: str = ""
    key: str = ""
    based_on: str = ""
    how: str = ""
    flags: str = ""                    # 12z A1: its launch flags, as typed
    env: str = ""                      # 12z A1: its environment, KEY=VALUE
    thinking: str = "auto"
    phone: bool = False                # 12f.2: a phone build
    gguf_path: str = ""                # 12f.3: its GGUF file on the server
    gguf_flags: str = ""
    gguf_setups: str = ""              # 12f.3 addendum: "name: KEY=VALUE --flag", a line each
    # 16.1: its size, as the person registering it confirms it: "35B", "3B"
    size: str = ""
    active: str = ""
    by: str = ""


class LaunchIn(BaseModel):
    flags: str = ""
    env: str = ""


@app.put("/api/served/{model_id:path}/launch")
def served_launch(model_id: str, f: LaunchIn, x_token: str = Header(default="")):
    """12z A1: a served setup's launch flags and environment — what its
    labels (lookahead, MTP) are read from — kept without asking its server"""
    _check_token(x_token)
    try:
        out = {"model": served.set_launch(model_id, f.flags, f.env),
               "launch": served.launch_of_id(model_id)}
    except ValueError as e:
        raise HTTPException(404, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return out


@app.get("/api/served")
def served_list():
    return {"models": served.all_public(), "suites": list(served.SUITES),
            "line": served.LOGLIK_LINE, "thinking": served.THINKING,
            # 12m.3: models from OpenRouter — offered with a key, never the key
            "openrouter": {"has_key": ai_models.has_key(),
                           "subset": config.OPENROUTER_GEN_SUBSET}}


class OpenRouterIn(BaseModel):
    model: str
    by: str = ""


@app.post("/api/served/openrouter")
def served_add_openrouter(f: OpenRouterIn, x_token: str = Header(default="")):
    """12m.3: a model from OpenRouter's list (the one AI models shows), kept
    as a served entry pinned to its dated version and first provider"""
    _check_token(x_token)
    by = _name(f.by, "a model from OpenRouter")
    try:
        out = {"model": served.register_openrouter(f.model, by)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return out


class EstimateIn(BaseModel):
    model: str
    suite: str = "everyday"
    tasks: list[str] = []
    subset: int = 0
    part: str = ""


@app.post("/api/served/estimate")
def served_estimate(a: EstimateIn):
    """12m.3: what a run of a model from OpenRouter would cost, about, before
    Start — and whether this month's AI limit leaves room for it"""
    rec = served.get(a.model)
    if not served.is_openrouter(rec):
        raise HTTPException(422, f"{a.model} is not a model from OpenRouter")
    if a.suite not in served.SUITES:
        raise HTTPException(422, served.LOGLIK_LINE)
    est = served.estimate(rec, a.suite, a.tasks, a.subset, part=a.part)
    cap, month = ai_models.limit(), db.spend_this_month()
    return {**est, "month": round(month, 4), "limit": cap, "left": round(max(0.0, cap - month), 4),
            "refused": served.over_limit_line(est)}


@app.post("/api/served/check")
def served_check(f: ServedIn):
    """Check before saving: what the server at this address reports. Nothing
    is kept"""
    try:
        return {"reported": served.check(f.model_dump())}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


class SameAsIn(BaseModel):
    served: str = ""            # the served entry; "" (from a GGUF's side) clears that setup
    gguf: str = ""              # its GGUF entry; "" guesses again, "none" never joins
    setup: str = "as-built"
    by: str = ""


@app.post("/api/served/same-as")
def served_same_as(a: SameAsIn, x_token: str = Header(default="")):
    """12o.1: a served entry and a GGUF entry's setup are one file, said
    outright — the join never guesses them. From either entry's form"""
    _check_token(x_token)
    _name(a.by, "linking a served model to its GGUF")
    try:
        if not a.served:
            if not a.gguf or a.gguf == "none":
                raise ValueError("which served model: none was named")
            out = {"cleared": served.same_as_clear(a.gguf, a.setup)}
        else:
            out = {"served": a.served, "same_as": served.same_as_set(a.served, a.gguf, a.setup)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return out


@app.post("/api/served")
def served_add(f: ServedIn, x_token: str = Header(default="")):
    """Check the server, pin what it reports, and keep it. A server that
    doesn't answer is said in one line, and nothing is kept"""
    _check_token(x_token)
    try:
        if f.size.strip():
            sizes.check(f.size, f.active)              # before anything is kept
        out = {"model": served.register(f.model_dump(), f.by.strip()[:80])}
        if f.size.strip():
            out["size"] = sizes.set_size(out["model"]["id"], f.size, f.active, f.by)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)       # on Models at the next look
    return out


# ---------------------------------------------------------------------------
# 12f.2: On phone — numbers measured on the phone, as reported, beside what
# the board measured through the served model. Never in a column or average
# ---------------------------------------------------------------------------

class PhoneIn(BaseModel):
    model: str
    device: str = ""
    chip: str = ""
    ram_gb: float | None = None
    decode_median: float | None = None
    decode_best: float | None = None
    repeats: str = ""
    settings: str = ""
    date: str = ""
    by: str = ""                       # who measured it
    quality: list[dict] = []
    source: str = ""
    entered_by: str = ""               # who typed it in


@app.get("/api/phone")
def phone_builds():
    return {"builds": phone.builds(), "readme": phone.README}


# ---------------------------------------------------------------------------
# 12m.2: reported scores from outside the board — their own endpoint, never
# the results payload, so never a column, an average or the frozen report
# ---------------------------------------------------------------------------

@app.get("/api/reported")
def reported_view():
    return reported.view()


class ReportedImportIn(BaseModel):
    by: str = ""


@app.post("/api/reported/import")
def reported_import(a: ReportedImportIn, x_token: str = Header(default="")):
    """Import now: every source, once, each saying what it did in one line"""
    _check_token(x_token)
    _name(a.by, "importing reported scores")
    return {"results": reported.run_all()}


class ReportedCardIn(BaseModel):
    model: str
    maker: str = ""
    benchmark: str
    value: str
    setting: str = ""
    url: str = ""
    date: str = ""
    entered_by: str = ""


@app.post("/api/reported/cards")
def reported_card(f: ReportedCardIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return {"score": reported.card_add(f.model_dump(), f.entered_by)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


class ReportedAliasIn(BaseModel):
    alias: str
    target: str = ""
    by: str = ""


@app.post("/api/reported/aliases")
def reported_alias(a: ReportedAliasIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return reported.alias_set(a.alias, a.target, a.by)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@app.post("/api/reported/aliases/delete")
def reported_alias_delete(a: ReportedAliasIn, x_token: str = Header(default="")):
    _check_token(x_token)
    _name(a.by, "removing an alias")
    if not db.reported_alias_delete(reported.key(a.alias)):
        raise HTTPException(404, "no such alias")
    return {"deleted": reported.key(a.alias)}


@app.post("/api/phone/reports")
def phone_report_add(f: PhoneIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        return {"report": phone.add(f.model_dump(), f.entered_by)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


# ---------------------------------------------------------------------------
# 12f.3: GGUF files, measured by llama-perplexity on the host (gguf_worker.py)
# ---------------------------------------------------------------------------

class GgufIn(BaseModel):
    name: str = ""
    path: str = ""
    based_on: str = ""
    how: str = ""
    flags: str = ""
    setups: str = ""                   # 12f.3 addendum
    size: str = ""                     # 16.1: its size, confirmed: "35B"
    active: str = ""                   # and its active parameters: "3B"
    by: str = ""


class GgufRunIn(BaseModel):
    model: str
    benchmarks: list[str] = []
    subset: int = 0
    setups: list[str] = []             # 12f.3 addendum: none is every one
    by: str = ""


@app.get("/api/gguf")
def gguf_page():
    gguf.sync()
    models = [{"id": r["id"], **gguf.view(r)} for r in db.gguf_all()]
    models += [{"id": r["id"], "name": r["name"], "path": r["gguf_path"],
                "based_on": r.get("based_on", ""), "how": r.get("how", ""),
                "flags": gguf.flags_of(r.get("gguf_flags")), "pin": r.get("gguf_pin") or {},
                "setups": [gguf.gb.AS_BUILT] + (r.get("gguf_setups") or []),
                "served": True} for r in db.served_all() if r.get("gguf_path")]
    man = gguf.manifest()
    # 14.3: a benchmark built apart (Mobile-MMLU-Pro) is offered once its dataset is built.
    # 14.4.5: the full Mobile-MMLU, switched off, isn't offered or shown
    hidden = gguf.hidden()
    order = [b for b in gguf.gb.ORDER if (not gguf.gb.BENCHMARKS[b].get("apart") or b in man)
             and b not in hidden]
    return {"worker": gguf.worker(),
            "benchmarks": {k: v for k, v in gguf.gb.BENCHMARKS.items() if k not in hidden},
            "order": order, "datasets": {k: v for k, v in man.items() if k not in hidden},
            "models": models,
            # 14.4.4: each restricted benchmark's badge and sentence
            "restrictions": _restrictions().sets(),
            "default_flags": " ".join(gguf.gb.DEFAULT_FLAGS), "mtp_line": gguf.gb.MTP_LINE}


@app.post("/api/gguf/models")
def gguf_register(f: GgufIn, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        if f.size.strip():
            sizes.check(f.size, f.active)              # before anything is kept
        rec = gguf.register(f.model_dump(), f.by.strip()[:80])
        got = sizes.set_size(rec["id"], f.size, f.active, f.by) if f.size.strip() else None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return {"model": {"id": rec["id"], **gguf.view(rec)}, **({"size": got} if got else {})}


class SizeIn(BaseModel):
    model: str
    total: str = ""
    active: str = ""
    by: str = ""


@app.post("/api/models/size")
def model_size(a: SizeIn, x_token: str = Header(default="")):
    """16.1: a model's size, as a person enters it on its page — its total and,
    for a mixture of experts, its active parameters. First of the board's
    sources for it; kept with who entered it"""
    _check_token(x_token)
    known = {m["id"] for m in results_payload()["models"]}
    if a.model not in known:
        raise HTTPException(404, f"no such model on the board: {a.model}")
    try:
        got = sizes.set_size(a.model, a.total, a.active, a.by)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return got


@app.post("/api/gguf/estimate")
def gguf_estimate(a: GgufRunIn):
    try:
        # 14.3: "all" is what a run of all asks: every benchmark with a dataset built
        # (14.4: but the full Mobile-MMLU, measured only when named)
        return gguf.estimate(a.model, a.benchmarks or [b for b in gguf.gb.ORDER
                                                       if b in gguf.manifest()
                                                       and not gguf.gb.BENCHMARKS[b].get(
                                                           "kept_apart")] or gguf.gb.DEFAULT,
                             a.subset, a.setups or None)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@app.post("/api/gguf/runs")
def gguf_run(a: GgufRunIn, x_token: str = Header(default="")):
    _check_token(x_token)
    by = _name(a.by, "a GGUF measurement")
    why = disk.blocks_run()
    if why:
        raise HTTPException(409, why)
    try:
        ids = gguf.queue(a.model, a.benchmarks, a.subset, by, setups=a.setups or None)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    w = gguf.worker()
    return {"id": ids[0], "ids": ids, "status": "queued", "worker": w}


class GgufRerunIn(BaseModel):
    by: str = ""


@app.post("/api/gguf/runs/{sid}/rerun")
def gguf_rerun(sid: int, a: GgufRerunIn, x_token: str = Header(default="")):
    """12f.5: Re-run failed benchmarks — a new run of only the benchmarks
    run `sid` didn't finish, in its setup and subset"""
    _check_token(x_token)
    by = _name(a.by, "a GGUF measurement")
    why = disk.blocks_run()
    if why:
        raise HTTPException(409, why)
    try:
        out = gguf.rerun_failed(sid, by)
    except LookupError as e:
        raise HTTPException(404, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {**out, "status": "queued", "worker": gguf.worker()}


@app.get("/api/models/code")
def model_code(id: str = ""):
    """Whether a checkpoint ships its own model code, and whether this server
    will run it (11i) — the page asks before it offers Queue this run. For
    an upload, also whether its weights are here at all (11a)."""
    hf_id = id.strip()
    out = hfmeta.remote_code_check(hf_id)
    name = hf_id[len("local/"):] if hf_id.startswith("local/") else ""
    if name and uploads.folder_name_ok(name):
        out["weights"] = uploads.artifact_dir(name) is not None
    return out


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
    # 16b review: ".." matched the old rule, and this removed BENCH_ROOT
    if not uploads.folder_name_ok(name):
        raise HTTPException(422, "bad artifact name")
    mid = f"local/{name}"
    for row in db.recent(200):
        if row["hf_id"] == mid and row["status"] in ACTIVE:
            raise HTTPException(409, "that artifact is queued or being evaluated")
    d = uploads.artifact_dir(name)
    if d is None:
        raise HTTPException(404, "no such artifact")
    import shutil
    shutil.rmtree(d)
    return {"deleted": name, "note": "its benchmark results stay on the leaderboard"}


# ---------------------------------------------------------------------------
# 12q: DeviceMark's protocol — each row that sat the battery, and a speed
# measured on a device, entered by a person (only that places a row on the
# chart's x-axis; the server's speed test never does)
# ---------------------------------------------------------------------------

def _mab():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import mobileaibench
    return mobileaibench


def _mab_judge() -> dict:
    """14.1: the judge as an estimate needs it: its id and words, whether it
    is this server's, and its prices when it is one from OpenRouter"""
    import judge as _judge
    c = ai_models.choice("judge") or {}
    try:
        ident = _judge.identity()
    except Exception:                                   # noqa: BLE001 — no judge set up
        ident = {"id": ""}
    return {"id": ident.get("id", ""), "label": ai_models.label("judge"),
            "local": ai_models.is_local("judge"),
            "price_in": c.get("price_in") if c.get("kind") == "openrouter" else None,
            "price_out": c.get("price_out") if c.get("kind") == "openrouter" else None}


@app.get("/api/mobileaibench/estimate")
def mab_estimate(model: str):
    """14.1: what each part of a model's MobileAIBench run takes, before
    Start: its answers and time — the model's own measured seconds an answer
    when it has some, else the page's guess — and for the judged part the
    judge's judgements, tokens and (an OpenRouter judge's) cost"""
    if not _HF_ID_RE.match(model):
        raise HTTPException(422, "model must be a model id on the board")
    rec = served.get(model) if served.is_served(model) else None
    each = ((rec or {}).get("speed") or {}).get("secs_each") if rec else _mab_hf_each()
    parts = _mab().estimate(bool(rec), each, _mab_judge())
    # 14.3: Mobile-MMLU-Pro — a letter a question, or four log-likelihoods.
    # 14.4: each counts only what this model has no pick for, by either set's run
    mdir = config.OUT_DIR / model.replace("/", "__")
    parts["mmlu"] = _mmp().run_estimate(bool(rec), each if rec else None, "pro", mdir)
    if _mmp().full_on():                    # 14.4.5: switched off, it isn't offered
        parts["mmlu_full"] = _mmp().run_estimate(bool(rec), each if rec else None, "full", mdir)
    # 14.4.4: and what each set's questions and scores may be used for
    for p, t in (("mmlu", config.MMP_TASK), ("mmlu_full", config.MMF_TASK)):
        rx = _restrictions().of(t)
        if rx and p in parts:
            parts[p]["restriction"] = rx
    return {"model": model, "parts": parts}


def _mab_hf_each() -> float | None:
    """seconds an answer of this server's earlier MobileAIBench runs, on its
    GPU: their GPU time over their answers"""
    rows = [r for r in db.recent(500) if r.get("suite") == "mobile" and r.get("status") == "done"
            and r.get("gpu_seconds") and not served.is_served(r.get("hf_id") or "")
            and (r.get("part") or "") not in ("mmlu", "mmlu_full")]
    if not rows:
        return None
    m = _mab()
    n = sum(m.part_counts(r.get("part") or "")["answers"] for r in rows)
    return round(sum(float(r["gpu_seconds"]) for r in rows) / n, 3) if n else None


class MabJudgeIn(BaseModel):
    by: str = ""


@app.post("/api/mobileaibench/judge")
def mab_judge(a: MabJudgeIn, x_token: str = Header(default="")):
    """14.1: the later judging step — every model's MT-Bench turns that wait
    for the judge, sent now. A judge still offline leaves them waiting, and
    says why"""
    _check_token(x_token)
    _name(a.by, "a judging step")
    m = _mab()
    out = []
    for d in m.awaiting(config.OUT_DIR):
        got = m.start_judge(d)
        out.append({"model": d.name, "batch_id": got.get("batch_id"), "note": got.get("note", ""),
                    "summary": m.summary(got)})
    _cache.update(key=None, payload=None, at=0.0)
    return {"models": out}


# -- 14.3: Mobile-MMLU-Pro and our answer key -----------------------------------

def _restrictions():
    """scripts/restrictions.py: what a restricted set may be used for (14.4.4)"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import restrictions
    return restrictions


def _mmp():
    from . import mmp_key
    return mmp_key.mmp


def _mmp_key_page() -> dict:
    """the key's card on AI models: its labellers, the dry run, the run, the
    key's counts, and the paper's three models beside ours"""
    from . import mmp_key
    st = mmp_key.status()
    m = _mmp()
    st["checks"] = m.paper_checks({mid: m.score(m.predictions(config.OUT_DIR / mid.replace("/", "__")))
                                   for mid in m.manifest()["paper_checks"]["models"]})
    # 14.4: the full set's, on its own key and Table 2's Mobile-MMLU column
    # (14.4.5: none while it is switched off)
    st["checks_full"] = None if not m.full_on() else m.paper_checks(
        {mid: m.score(m.full_predictions(config.OUT_DIR / mid.replace("/", "__")), m.full_key())
         for mid in m.full_manifest()["paper_checks"]["models"]}, "full")
    st["portal"] = m.portal_scores()
    st["credit"] = m.credit()
    # 14.4.4: each set's badge and sentence, for the card and any reader of this page
    st["restrictions"] = {t: rx for t in (config.MMP_TASK, config.MMF_TASK)
                          if (rx := _restrictions().of(t))}
    return st


class MmpByIn(BaseModel):
    by: str = ""


class LabellerIn(BaseModel):
    model: str
    by: str


@app.get("/api/mobile-mmlu/key")
def mmp_key_status():
    return _mmp_key_page()


@app.post("/api/mobile-mmlu/key/start")
def mmp_key_start(a: MmpByIn, x_token: str = Header(default="")):
    """masein's Start, after the dry run: the labellers pinned, and what is
    left to label sent. Started again after a stop, it carries on"""
    from . import mmp_key
    _check_token(x_token)
    _name(a.by, "labelling the key")
    try:
        sent = mmp_key.start(a.by.strip()[:80])
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    return {**sent, "page": _mmp_key_page()}


@app.post("/api/mobile-mmlu/key/stop")
def mmp_key_stop(a: MmpByIn, x_token: str = Header(default="")):
    from . import mmp_key
    _check_token(x_token)
    _name(a.by, "stopping the labelling")
    mmp_key.stop(a.by.strip()[:80])
    return {"page": _mmp_key_page()}


@app.post("/api/ai/labellers/{slot}")
def mmp_labeller_set(slot: str, a: LabellerIn, x_token: str = Header(default="")):
    """a key labeller, pinned on OpenRouter — never local, in-house, a model
    scored on this set, or another labeller's maker"""
    from . import mmp_key
    _check_token(x_token)
    if not a.by.strip():
        raise HTTPException(422, "type your name first — it is recorded with the choice")
    try:
        saved = mmp_key.save(slot, a.model.strip(), a.by.strip()[:80])
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return {"saved": saved, "page": _ai_page(), "key": _mmp_key_page()}


class PortalIn(BaseModel):
    model: str
    score: float
    by: str


@app.post("/api/mobile-mmlu/portal")
def mmp_portal_set(a: PortalIn, x_token: str = Header(default="")):
    """the score the authors' portal gave a model's predictions, typed in"""
    _check_token(x_token)
    _name(a.by, "the portal's score")
    if not _HF_ID_RE.match(a.model) or not 0 <= a.score <= 100:
        raise HTTPException(422, "a model id on the board, and a score from 0 to 100")
    _mmp().set_portal_score(a.model, round(float(a.score), 2), a.by.strip()[:80])
    return _mmp_key_page()


def _hub_public(hf_id: str) -> bool:
    """the model is public on the Hub, asked without this server's token:
    gated is public, private isn't"""
    try:
        from huggingface_hub import model_info
        return not getattr(model_info(hf_id, token=False), "private", True)
    except Exception:                               # noqa: BLE001 — unknown is not public
        return False


def mmp_download_refused(model_id: str, meta: dict | None) -> str:
    """'' when the portal's predictions may be offered for this model: a
    public model on Hugging Face, run here. Never one served elsewhere or
    from OpenRouter, a GGUF, a checkpoint trained here, or an in-house build"""
    from . import mmp_key
    if served.is_served(model_id) or (meta or {}).get("served") or (meta or {}).get("gguf"):
        return "only for a public model on Hugging Face: this one is served"
    if mmp_key.IN_HOUSE.search(model_id) or (meta or {}).get("source") == "artifact" \
            or (meta or {}).get("trained_from"):
        return "never for an in-house model"
    if not _HF_ID_RE.match(model_id):
        return "only for a public model on Hugging Face"
    return ""


@app.get("/api/mobile-mmlu/predictions")
def mmp_predictions(model: str):
    """14.3: a public HF model's picks in the authors' portal format
    (question_id,predicted_answer), for masein to submit — never our key,
    never a served or in-house model's"""
    d = config.OUT_DIR / model.replace("/", "__")
    try:
        meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    why = mmp_download_refused(model, meta)
    if not why and not _hub_public(model):
        why = "only for a model that is public on Hugging Face"
    if why:
        raise HTTPException(403, why)
    preds = _mmp().predictions(d)
    if not preds:
        raise HTTPException(404, f"{model} has no Mobile-MMLU-Pro answers on file")
    name = model.replace("/", "_") + "_predictions.csv"
    return PlainTextResponse(_mmp().portal_csv(preds), media_type="text/csv",
                             headers={"content-disposition": f'attachment; filename="{name}"'})


def _dm():
    from . import devicemark as _devicemark
    return _devicemark.dm()


@app.get("/api/devicemark")
def devicemark_rows():
    """ours, and (12q.B) DeviceMark's own rows from the committed snapshot —
    ranked together, `rank_all`; nothing is fetched from DeviceMark here"""
    d = _dm()
    return {"version": d.VERSION, "whose": d.WHOSE, "cap": d.CAP,
            "server_speed_label": d.SERVER_SPEED_LABEL,
            **d.board(config.OUT_DIR, served.launch_of_id)}


@app.get("/api/devicemark/answers")
def devicemark_answers(model: str, thinking: bool = False, bench: str = "", offset: int = 0,
                       limit: int = 50):
    """12q.C: a DeviceMark row's answers, for the model page's Answers tab —
    public benchmark items, all of them, no answer and wrong first"""
    from . import devicemark as _devicemark
    if not _HF_ID_RE.match(model):
        raise HTTPException(422, "model must be a model id on the board, like served/<name>")
    if bench and bench not in _dm().BENCHES:
        raise HTTPException(422, f"bench is one of {', '.join(_dm().BENCHES)}, or none for all")
    got = _devicemark.answers(model, thinking, bench, max(0, offset), max(1, min(limit, 200)))
    if got is None:
        raise HTTPException(404, f"no DeviceMark answers for {model}"
                                 + (" with thinking on" if thinking else ""))
    return got


class DeviceSpeedIn(BaseModel):
    model: str
    tok_s: float | None = None          # None clears it
    device: str = ""
    source: str = ""                    # "measured by <colleague>, <date>"
    by: str = ""


@app.put("/api/devicemark/device")
def devicemark_device(body: DeviceSpeedIn, x_token: str = Header(default="")):
    _check_token(x_token)
    d = _dm()
    mid = body.model.strip()
    if not _HF_ID_RE.match(mid):
        raise HTTPException(422, "model must be a model id on the board, like served/<name>")
    base = config.OUT_DIR / mid.replace("/", "__")
    f = base / d.DEVICE_NAME
    if body.tok_s is None:
        f.unlink(missing_ok=True)
        return {"model": mid, "device": None}
    if not 0 < body.tok_s < 100000:
        raise HTTPException(422, "tok_s is a decode speed in tokens a second, more than 0")
    if not body.device.strip() or not body.source.strip():
        raise HTTPException(422, "a device speed says the device and where it came from "
                                 "(source: \"measured by <name>, <date>\")")
    if not base.is_dir():
        raise HTTPException(404, f"{mid} has no results on this board")
    rec = {"tok_s": round(float(body.tok_s), 2), "device": body.device.strip()[:80],
           "source": body.source.strip()[:200], "by": body.by.strip()[:80], "at": time.time()}
    f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return {"model": mid, "device": rec}


class DmRawIn(BaseModel):
    model: str
    thinking: bool = False
    url: str = ""
    by: str = ""


@app.put("/api/devicemark/raw")
def devicemark_raw(body: DmRawIn, x_token: str = Header(default="")):
    """15.4: a DeviceMark row's published raw run — the board's "raw" link on
    it (scripts/export_devicemark_raw.py --link sets the same); an empty url
    takes it off"""
    _check_token(x_token)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import export_devicemark_raw as ex
    mid = body.model.strip()
    if not _HF_ID_RE.match(mid):
        raise HTTPException(422, "model must be a model id on the board")
    rows = ex.row_dirs(mid, body.thinking)
    if not rows:
        raise HTTPException(404, f"{mid} has no DeviceMark row "
                                 f"{'with thinking on ' if body.thinking else ''}on this board")
    if body.url.strip() and not body.by.strip():
        raise HTTPException(422, "type your name first — it is recorded with the link")
    try:
        got = ex.set_link(rows[0], body.url.strip() or None, body.by.strip()[:80])
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return {"model": mid, "thinking": body.thinking, "raw": got}


@app.get("/api/results")
def results():
    payload = results_payload()
    # 12f.0: the disk, asked on every request — never cached with the scores
    d = disk.status_check()
    if d:
        payload = {**payload, "checks": [d, *(payload.get("checks") or [])],
                   "warnings": [d["text"], *(payload.get("warnings") or [])]}
    # 12p.1: a set that lives only on the data volume, missing or changed — asked
    # on every request, never cached with the scores
    from . import hidden_store
    payload = {**payload, "alarms": hidden_store.alarms()}
    # 16.5: what the exam switched off hides on Improve, counted, never deleted
    if not config.KNOWLEDGE_EXAM:
        props = db.proposal_list(None, 500)
        exam_ids = {p["id"] for p in props if exam_task(p.get("task"))}
        payload = {**payload, "examHidden": {
            "proposals": len(exam_ids),
            "datasets": sum(1 for d in db.dataset_list(500) if d["proposal_id"] in exam_ids)}}
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# find the gap, make data for it — see service/proposals.py for the pipeline
# and the one rule every endpoint here enforces
# ---------------------------------------------------------------------------

class ProposalIn(BaseModel):
    model: str
    topic: str = ""              # an exam topic from scripts/categories.yaml
    everyday: str = ""           # 12g.2: or an Everyday group ("instructions")
    requested_by: str = ""
    # propose over a judge whose grades are not evidence yet — accepted only
    # when every reason is about the judge, the setting allows it, and a name
    # is recorded; the proposal and everything made from it carry the mark
    override_preliminary: bool = False


class ApproveIn(BaseModel):
    approver: str
    edited_text: str = ""
    # 11e: "Spread the documents over these", on by default
    spread: bool = True


class RejectIn(BaseModel):
    approver: str
    reason: str = ""


class GenerateIn(BaseModel):
    requester: str
    count: int = 20
    # prose documents for an exam topic, chat examples for an Everyday group:
    # empty is the proposal's own default (12g.2)
    fmt: str = ""


def _llm_status() -> dict:
    why = llm.blocked()
    used = prop.dir_bytes(config.DATASETS_DIR)
    return {"configured": not why, "reason": why,
            "provider": config.LLM_PROVIDER, "model": config.LLM_MODEL,
            # the generator's own provider: what Propose and Generate spend
            "usage_today": db.llm_items_today(config.LLM_PROVIDER),
            "daily_cap": config.daily_cap(config.LLM_PROVIDER),
            "usage": llm_usage(),
            "max_items_per_batch": config.LLM_MAX_ITEMS_PER_BATCH,
            # what THIS provider is asked for per request: a local generator
            # gets one document, because its replies are capped
            "items_per_generation_request": {f: prop.items_per_request(f) for f in prop.FORMATS},
            "formats": list(prop.FORMATS), "default_format": prop.DEFAULT_FORMAT,
            "datasets_bytes": used, "datasets_quota_bytes": int(config.DATASET_QUOTA_GB * 1e9),
            # 12i.1: each job's model in words, for the one line on Improve and
            # the Knowledge exam ("AI: judge DeepSeek V4.1 Flash · writer GLM 5.3")
            "ai": {j: ai_models.label(j) for j in ai_models.JOBS},
            # 12z D1: which jobs are on the local server, whose health the page polls
            "ai_local": {j: ai_models.is_local(j) for j in ai_models.JOBS},
            "ai_waiting": ai_models.over_limit(),
            "note": "the tailnet is the auth boundary: approvals record a typed name, "
                    "nothing more"}


@app.get("/api/llm")
def llm_status():
    return _llm_status()


# ---------------------------------------------------------------------------
# 12i.1: the AI models page — which model does each job, what it costs, and
# the judge test. The OpenRouter key is never in a reply: only whether there
# is one
# ---------------------------------------------------------------------------

def _improving() -> list[str]:
    """the models Improve is working on: the ones with proposals"""
    return sorted({p["model"] for p in db.proposal_list(limit=500)})


def _ai_page() -> dict:
    import judge as _judge
    jobs = []
    for k, j in ai_models.JOBS.items():
        c = ai_models.choice(k)
        p, m, _ = llm.identity(j["role"])
        jobs.append({"job": k, "label": j["label"], "does": ai_models.does(k),
                     "suggested": j["suggested"], "why": j["why"],
                     "chosen": c, "from": "page" if c else ("environment" if p else None),
                     "provider": p, "model": m, "now": ai_models.label(k),
                     "blocked": llm.blocked(j["role"]) if (c or p) else ""})
    ident = _judge.identity()
    return {"has_key": ai_models.has_key(), "jobs": jobs,
            "local": {"name": ai_models.local_name(ask=True), "model": ai_models.local_model()},
            "spend": {"month": round(db.spend_this_month(), 4), "limit": ai_models.limit(),
                      "by_job": db.spend_this_month_by_job(), "waiting": ai_models.over_limit()},
            "warnings": ai_models.warnings(_improving()),
            "judge": {"id": ident["id"], "version": _judge.version(ident)}}


@app.get("/api/ai")
def ai_page():
    return _ai_page()


@app.get("/api/ai/models")
def ai_models_list(refresh: int = 0):
    """OpenRouter's text models with their prices, the suggested one per job
    first. Nothing is asked of OpenRouter without a key"""
    if not ai_models.has_key():
        return {"has_key": False, "models": []}
    return {"has_key": True, "models": ai_models.models(refresh=bool(refresh)),
            "suggested": {k: {"id": j["suggested"], "why": j["why"]}
                          for k, j in ai_models.JOBS.items()}}


class AiJobIn(BaseModel):
    model: str
    by: str


def _rejudge_scope() -> dict:
    """what a new judge would re-judge: the Knowledge exam answers on file
    marked by another judge version, and the Everyday judged questions"""
    import everyday as _ev
    import judge as _judge
    key = _judge.version()["key"]
    ident = _judge.identity()
    exam, evd = {}, {}
    for d in sorted(config.OUT_DIR.glob("*")) if config.OUT_DIR.is_dir() else []:
        try:
            j = json.loads((d / "judge.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            j = None
        if j and not j.get("skipped"):
            head = j.get("judge") or {}
            v = (head.get("version") or {}).get("key")
            if not ((v and v == key) or (not v and head.get("id") == ident["id"])):
                n = sum(1 for t in (j.get("tasks") or {}).values()
                        for it in t.get("items") or [] if not it.get("no_answer"))
                if n:
                    exam[j.get("model") or d.name.replace("__", "/", 1)] = {
                        "n": n, "tasks": sorted(j.get("tasks") or {})}
        e = _ev.read(d)
        if e and (e.get("judge") or {}).get("version") != key:
            n = len(_ev.judged_verdicts(e))
            if n:
                evd[e.get("model") or d.name.replace("__", "/", 1)] = n
    return {"exam": exam, "everyday": evd}


def _rejudge_estimate() -> dict:
    s = _rejudge_scope()
    n = sum(x["n"] for x in s["exam"].values()) + sum(s["everyday"].values())
    c = ai_models.choice("judge") or {}
    usd = n * ((c.get("price_in") or 0) * judge_test.TOKENS_IN
               + (c.get("price_out") or 0) * judge_test.TOKENS_OUT) / 1e6
    return {"n": n, "usd": round(usd, 2), "models": len(set(s["exam"]) | set(s["everyday"]))}


@app.post("/api/ai/jobs/{job}")
def ai_job_set(job: str, a: AiJobIn, x_token: str = Header(default="")):
    """Pin a job to a model. For the judge, a new model is a new judge
    version: the reply says how many answers on file another judge marked,
    and about what re-judging them would cost — the page asks first"""
    import judge as _judge
    _check_token(x_token)
    if not a.by.strip():
        raise HTTPException(422, "type your name first — it is recorded with the choice")
    before = _judge.version()["key"]
    try:
        saved = ai_models.save(job, a.model.strip(), a.by.strip()[:80])
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    llm.reset()
    out = {"saved": saved, "page": _ai_page()}
    # 16.5: re-judging is the Knowledge exam's, and asks nothing while it is off
    if job == "judge" and config.KNOWLEDGE_EXAM and _judge.version()["key"] != before:
        out["rejudge"] = _rejudge_estimate()
    return out


class AiLimitIn(BaseModel):
    usd: float
    by: str


@app.post("/api/ai/limit")
def ai_limit_set(a: AiLimitIn, x_token: str = Header(default="")):
    _check_token(x_token)
    if not a.by.strip():
        raise HTTPException(422, "type your name first — it is recorded with the limit")
    if not 0 <= a.usd <= 10000:
        raise HTTPException(422, "the monthly limit is between $0 and $10,000")
    db.ai_set("spend_limit", round(float(a.usd), 2), a.by.strip()[:80])
    return _ai_page()


@app.get("/api/ai/rejudge", dependencies=EXAM_ONLY)
def ai_rejudge_estimate():
    return _rejudge_estimate()


class ByIn(BaseModel):
    by: str


@app.post("/api/ai/rejudge", dependencies=EXAM_ONLY)
def ai_rejudge(a: ByIn, x_token: str = Header(default="")):
    """Re-judge every answer on file another judge marked, with the judge now:
    a judged run per model whose exam answers are on disk (it only grades
    them — no GPU), and an everyday run per model whose judged questions had
    another judge's verdicts (it asks nothing new, only the judge)"""
    import everyday as _ev
    import judge as _judge
    _check_token(x_token)
    if not a.by.strip():
        raise HTTPException(422, "type your name first — it is recorded on each run")
    why = _judge.blocked() or ai_models.over_limit()
    if why:
        raise HTTPException(409, why)
    s = _rejudge_scope()
    queued = []
    for model, x in s["exam"].items():
        queued.append(db.add(model, "auto", "judged", a.by.strip()[:80],
                             "judged again: a new judge", tasks=x["tasks"]))
    for model in s["everyday"]:
        _ev.clear_verdicts(config.OUT_DIR / model.replace("/", "__"))
        queued.append(db.add(model, "instruct", "everyday", a.by.strip()[:80],
                             "judged again: a new judge"))
    return {"queued": queued, "n": _rejudge_estimate()["n"]}


# -- the judge test ------------------------------------------------------------

@app.get("/api/judge-test")
def judge_test_page():
    """the answers to mark, as masein sees them — never a judge's mark — his
    marks so far, and where he got to"""
    every = judge_test.answers()
    shown = [judge_test.shown(x) for x in judge_test.in_use(every)]
    hist = judge_test.history()
    prog = judge_test.progress()
    return {"answers": shown, "marks": db.jt_marks(judge_test.person()), "progress": prog,
            # 16.5: the Knowledge exam's answers in the sample, kept while it is off
            "exam_hidden": len(every) - len(shown),
            "kappa_min": config.JUDGE_KAPPA_MIN, "n_min": config.JUDGE_TEST_MIN,
            # 12f.0: a new sample is a new version; the earlier ones, marks and
            # all, are History — and until the new one has a mark, it says so
            "version": judge_test.version(), "history": hist,
            "changed": bool(hist) and not (prog["marked"] + prog["skipped"])}


class JtMarkIn(BaseModel):
    key: str
    mark: int | str | None = None
    by: str


@app.post("/api/judge-test/mark")
def judge_test_mark(a: JtMarkIn, x_token: str = Header(default="")):
    _check_token(x_token)
    if not a.by.strip():
        raise HTTPException(422, "type your name first — it is recorded with each mark")
    try:
        judge_test.mark(a.key, a.mark, a.by.strip()[:80])
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return judge_test.progress()


@app.get("/api/judge-test/estimate")
def judge_test_estimate(models: str = ""):
    ids = [m for m in models.split(",") if m.strip()]
    try:
        return judge_test.estimate(ids)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


class JtRunIn(BaseModel):
    models: list[str]
    by: str


@app.post("/api/judge-test/run")
def judge_test_run(a: JtRunIn, x_token: str = Header(default="")):
    _check_token(x_token)
    if not a.by.strip():
        raise HTTPException(422, "type your name first — it is recorded with the run")
    try:
        runs = judge_test.run([m.strip() for m in a.models if m.strip()], a.by.strip()[:80])
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    except llm.LLMError as e:
        # 16.2: a candidate's server that isn't answering (a local one: its
        # vLLM is down) is said in its own words, never a bare HTTP 500
        raise HTTPException(503, f"{e} Nothing was sent.") from None
    return {"runs": [{"batch_id": r["batch_id"], "name": r["candidate"]["name"]} for r in runs]}


@app.get("/api/judge-test/result")
def judge_test_result():
    return judge_test.result()


class JtUseIn(BaseModel):
    key: str
    by: str


@app.post("/api/judge-test/use")
def judge_test_use(a: JtUseIn, x_token: str = Header(default="")):
    """make a candidate the judge: a new judge version, as AI models does"""
    import judge as _judge
    _check_token(x_token)
    if not a.by.strip():
        raise HTTPException(422, "type your name first — it is recorded with the choice")
    before = _judge.version()["key"]
    try:
        saved = judge_test.use(a.key, a.by.strip()[:80])
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    llm.reset()
    out = {"saved": saved}
    if _judge.version()["key"] != before:
        out["rejudge"] = _rejudge_estimate()
    return out


# ---------------------------------------------------------------------------
# 12i.2: the question builder — Knowledge exam or Everyday questions, written,
# checked and reviewed in three steps, published as a new bank version
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 12d.1: the Playground — the models on this server, chatting, streamed.
# Chats are a person's own: every request says whose (by, or X-Who), and an
# id that isn't theirs is not found. OpenRouter models are not offered here
# ---------------------------------------------------------------------------

def _who_of(by: str, x_who: str) -> str:
    return _name(by or x_who, "a chat")


@app.get("/api/playground")
def playground_page():
    return {**playground.models(), "status": chat.ENGINE.status(),
            "idle_unload_s": config.CHAT_IDLE_UNLOAD_S}


@app.get("/api/playground/states")
def playground_states():
    """16.2: each model's state in the picker (Ready, Loads on the first message,
    On the CPU, Not now and why) and the GPU's line — asked every few seconds"""
    return playground.states()


@app.get("/api/gpu")
def gpu_status():
    """16.2: GPU memory — total, used and free in GB, and what holds it — from
    what the board knows for certain; "other" for the rest. Cached a few seconds"""
    g = gpu.status()
    return {k: v for k, v in g.items() if k not in ("total", "used", "free")} | {
        "holders": [{"kind": h["kind"], "name": h["name"], "gb": h.get("gb"),
                     "words": gpu.holder_words(h)} for h in g.get("holders") or []]}


_NEED: dict[str, tuple[float, dict]] = {}


@app.get("/api/gpu/need")
def gpu_need(model: str, kind: str = "auto"):
    """16.2: what a model loaded here needs of the GPU, beside what is free now,
    and when its run would start — for Test a model, before Start. A served
    model's memory is its server's; a GGUF file's, the host's llama.cpp"""
    model = model.strip()
    if served.is_served(model) or model.startswith("gguf/"):
        return {"here": False, "line": "its memory is its server's, not this card's"}
    now = time.time()
    hit = _NEED.get(model)
    if not hit or now - hit[0] > 600:
        try:
            meta = hfmeta.preflight(model, kind if kind in ("base", "instruct") else "auto")
            got = {"need_gb": float(meta["need_gb"])}
        except Exception as e:                          # noqa: BLE001 — said, never raised
            got = {"why": str(e)[:200]}
        _NEED[model] = hit = (now, got)
    got = hit[1]
    if "why" in got:
        return {"here": True, "line": "", "why": got["why"]}
    need = gpu.mib_to_bytes(int(got["need_gb"] * 1024) + config.FREE_MARGIN_MIB)
    from . import runner
    g = gpu.status()
    run = runner.run_holding()
    queued = [r for r in db.recent(50) if r.get("status") in ("queued", "preflight", "waiting_gpu",
                                                             "waiting_lock")]
    if run:
        when = f"waits for {gpu.run_words(gpu._run_now() or {'sid': run.get('sid')})}"
    elif queued:
        when = f"waits for {len(queued)} run{'s' if len(queued) > 1 else ''} queued before it"
    elif g.get("ok") and g["free"] < need:
        when = "waits for GPU memory"
    else:
        when = "starts now"
    free = f" · {gpu.gb_text(g['free'])} free now" if g.get("ok") else ""
    return {"here": True, "need_gb": round(need / gpu.GB, 1),
            "free_gb": g.get("free_gb"), "when": when,
            "line": f"Needs about {gpu.gb_text(need)}{free} · {when}"}


@app.get("/api/playground/status")
def playground_status():
    return chat.ENGINE.status()


@app.get("/api/playground/practice")
def playground_practice():
    """practice questions only — a hidden one is never listed"""
    return playground.practice()


@app.get("/api/playground/chats")
def playground_chats(x_who: str = Header(default="")):
    return {"chats": chat.list_chats(_who_of("", x_who))}


class ChatIn(BaseModel):
    model: str
    model2: str = ""                     # 12d.2: a second model, answering the same messages
    by: str = ""
    settings: dict = {}


@app.post("/api/playground/chats")
def playground_new(a: ChatIn, x_token: str = Header(default="")):
    _check_token(x_token)
    by = _who_of(a.by, "")
    try:
        c = chat.new_chat(a.model, by, a.settings, a.model2)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return playground.view(c)


def _chat_or_404(chat_id: str, by: str) -> dict:
    try:
        return chat.get_chat(chat_id, by)
    except KeyError:
        raise HTTPException(404, "no such chat of yours") from None


@app.get("/api/playground/chats/{chat_id}")
def playground_chat(chat_id: str, x_who: str = Header(default="")):
    return playground.view(_chat_or_404(chat_id, _who_of("", x_who)))


class ChatByIn(BaseModel):
    by: str = ""


@app.post("/api/playground/chats/{chat_id}/delete")
def playground_delete(chat_id: str, a: ChatByIn, x_token: str = Header(default="")):
    _check_token(x_token)
    by = _who_of(a.by, "")
    _chat_or_404(chat_id, by)
    playground.delete(chat_id, by)
    return {"deleted": chat_id}


class ChatSettingsIn(BaseModel):
    settings: dict
    by: str = ""


@app.post("/api/playground/chats/{chat_id}/settings")
def playground_settings(chat_id: str, a: ChatSettingsIn, x_token: str = Header(default="")):
    _check_token(x_token)
    by = _who_of(a.by, "")
    _chat_or_404(chat_id, by)
    try:
        return playground.set_settings(chat_id, a.settings, by)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from None


class MessageIn(BaseModel):
    text: str
    by: str = ""
    practice: dict | None = None


@app.post("/api/playground/chats/{chat_id}/messages")
def playground_send(chat_id: str, a: MessageIn, x_token: str = Header(default="")):
    _check_token(x_token)
    by = _who_of(a.by, "")
    _chat_or_404(chat_id, by)
    try:
        return playground.send(chat_id, a.text, by, a.practice)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


class AgainIn(BaseModel):
    n: int
    col: str = "a"                       # 12d.2: "b", the compared model's column
    by: str = ""


@app.post("/api/playground/chats/{chat_id}/again")
def playground_again(chat_id: str, a: AgainIn, x_token: str = Header(default="")):
    _check_token(x_token)
    by = _who_of(a.by, "")
    _chat_or_404(chat_id, by)
    try:
        return playground.again(chat_id, a.n, by, "b" if a.col == "b" else "a")
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@app.post("/api/playground/streams/{stream_id}/stop")
def playground_stop(stream_id: str):
    return {"stopped": chat.ENGINE.stop(stream_id)}


@app.get("/api/playground/streams/{stream_id}")
async def playground_stream(stream_id: str, request: Request):
    """a reply as server-sent events — the id is the key. A closed tab ends
    the reply: it frees the model for the next message"""
    st = chat.ENGINE.streams.get(stream_id)
    if st is None:
        raise HTTPException(404, "no such reply")

    async def events():
        i = 0
        try:
            while True:
                evs = st.events[i:]
                for ev in evs:
                    yield f"data: {json.dumps(ev)}\n\n"
                i += len(evs)
                if st.done and i >= len(st.events):
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.05)
        finally:
            # the tab went away before the reply ended (a disconnect, or the
            # server cancelling this generator): stop it, and free the model
            if not st.done:
                st.stop.set()
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/builder")
def builder_page():
    """what step 1 offers: kinds, topics with suggested subtopics, groups, the
    default writing instructions (their output section locked), each job's
    model, and the drafts so far"""
    exam = config.KNOWLEDGE_EXAM                         # 16.5: Everyday alone while off
    kinds = list(builder.KINDS) if exam else [k for k in builder.KINDS if k != "knowledge"]
    return {"kinds": kinds, "topics": builder.topics() if exam else [],
            "groups": builder.everyday_groups(),
            "levels": list(builder.LEVELS), "reasons": list(builder.REASONS),
            "try_n": builder.TRY_N, "min_reviewed": builder.MIN_REVIEWED,
            "suggest_min": builder.SUGGEST_MIN,
            "prompts": {k: builder.default_prompt(k) for k in kinds},
            "writer": builder.who("writer"), "checker": builder.who("checker"),
            "writer_blocked": builder.blocked("writer"),
            "checker_blocked": builder.blocked("checker"),
            "has_key": ai_models.has_key(),
            # 12o.1: on this server unless set to OpenRouter — then said to send
            # every question out
            "dedup_how": builder.dedup_how(),
            "dedup_warning": builder.REMOTE_WARNING if builder.embeds_remotely() else "",
            # 12i.4: newest first, with who wrote them, for the past batches
            "drafts": sorted([{"id": d["id"], "kind": d["kind"], "spec": d["spec"],
                               "stage": d["stage"], "status": d["status"], "by": d.get("by", ""),
                               "created_at": d.get("created_at"),
                               "writer": (d.get("writer") or {}).get("label", ""),
                               "updated_at": d.get("updated_at"), "published": d.get("published"),
                               "progress": builder.progress(d)} for d in db.qb_list()
                              if exam or d["kind"] != "knowledge"],
                             key=lambda x: -(x["created_at"] or 0))}


class BuildEstimateIn(BaseModel):
    kind: str
    count: int = 0
    group: str = ""
    writer: str = ""
    checker: str = ""


@app.post("/api/builder/estimate")
def builder_estimate(a: BuildEstimateIn):
    try:
        w = builder.who("writer", builder._override(a.writer))
        c = builder.who("checker", builder._override(a.checker))
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if a.kind == "knowledge" and not config.KNOWLEDGE_EXAM:
        raise HTTPException(409, config.EXAM_OFF + ".")             # 16.5
    return builder.estimate(a.kind, a.count, w, c, a.group)


class BuildIn(BaseModel):
    kind: str
    count: int = 0
    topic: str = ""
    subtopics: list[str] = []
    level: str = ""
    group: str = ""
    new_label: str = ""
    new_about: str = ""
    writer: str = ""
    checker: str = ""
    dedup: bool = True
    prompt: str = ""
    by: str = ""
    target: str = ""                      # 12p.3: "hidden", for a new hidden set


@app.post("/api/builder")
def builder_create(a: BuildIn, x_token: str = Header(default="")):
    """step 1 done: a draft, and its first ten being written"""
    _check_token(x_token)
    # 16.5: the Knowledge exam's path, while it is switched off
    if a.kind == "knowledge" and not config.KNOWLEDGE_EXAM:
        raise HTTPException(409, config.EXAM_OFF + ". Nothing was written.")
    by = _name(a.by, "a batch of questions")
    try:
        d = builder.create(a.model_dump(), by)
    except (ValueError, llm.LLMError) as e:
        raise HTTPException(422, str(e)) from None
    return builder.view(d)


def _draft(draft_id: str) -> dict:
    try:
        d = builder.get(draft_id)
    except KeyError:
        raise HTTPException(404, f"no draft {draft_id}") from None
    # 16.5: a Knowledge exam draft is kept, and neither shown nor carried on
    if d.get("kind") == "knowledge" and not config.KNOWLEDGE_EXAM:
        raise HTTPException(409, config.EXAM_OFF + ".")
    return d


def _draft_for(draft_id: str, by: str) -> dict:
    """12p.3: a draft for the hidden set is the board's owner's alone"""
    d = _draft(draft_id)
    if (d.get("spec") or {}).get("target") == "hidden" and not _is_owner(by):
        raise HTTPException(403, "A draft for the hidden set is the board's owner's")
    return d


@app.get("/api/builder/{draft_id}")
def builder_draft(draft_id: str, by: str = ""):
    return builder.view(_draft_for(draft_id, by))


class BuildReviewIn(BaseModel):
    n: int
    verdict: str
    reason: str = ""
    edited: dict = {}
    by: str = ""


@app.post("/api/builder/{draft_id}/review")
def builder_review(draft_id: str, a: BuildReviewIn, x_token: str = Header(default="")):
    _check_token(x_token)
    _draft_for(draft_id, a.by)
    by = _name(a.by, "a review")
    try:
        d = builder.review(draft_id, a.n, a.verdict, by, a.reason, a.edited or None)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return builder.view(d)


class BuildDupIn(BaseModel):
    n: int
    keep: str
    by: str = ""


@app.post("/api/builder/{draft_id}/duplicate")
def builder_duplicate(draft_id: str, a: BuildDupIn, x_token: str = Header(default="")):
    _check_token(x_token)
    _draft_for(draft_id, a.by)
    by = _name(a.by, "a choice between duplicates")
    try:
        d = builder.resolve_dup(draft_id, a.n, a.keep, by)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return builder.view(d)


class BuildByIn(BaseModel):
    by: str = ""


@app.post("/api/builder/{draft_id}/{step}")
def builder_step(draft_id: str, step: str, a: BuildByIn, x_token: str = Header(default="")):
    """rest (make the rest, checked), cancel, resume, publish"""
    _check_token(x_token)
    _draft_for(draft_id, a.by)
    by = _name(a.by, "this step")
    fn = {"rest": builder.rest, "cancel": builder.cancel, "resume": builder.resume,
          "publish": builder.publish}.get(step)
    if fn is None:
        raise HTTPException(404, f"no step {step}")
    try:
        out = fn(draft_id, by)
    except (ValueError, llm.LLMError) as e:
        raise HTTPException(422, str(e)) from None
    if step != "publish":
        return builder.view(out)
    # a new bank version: the exam's tasks rebuilt, the page's data fresh
    if out["kind"] == "knowledge":
        out["build"] = _rebuild_if_idle()
        fp = exam_build.current_fingerprints(config.JUDGED_TASKS_DIR) or {}
        out["version"] = (fp.get(builder._task(_draft(draft_id))) or "")[:12]
    _cache.update(key=None, payload=None, at=0.0)
    return {"published": out, "draft": builder.view(_draft(draft_id))}


# ---------------------------------------------------------------------------
# 12p.3: a new hidden set for Everyday — planned with its cost, written as
# Build questions drafts for the hidden set, staged, and switched to by the
# board's owner when they say
# ---------------------------------------------------------------------------

@app.get("/api/everyday/rotation")
def rotation_plan():
    """per group, what the new set takes; the models and their two rules; the
    cost of all of it — counts and prices, never a question"""
    from . import rotation
    return rotation.plan()


class RotationStartIn(BaseModel):
    by: str = ""
    confirm: bool = False


@app.post("/api/everyday/rotation/start")
def rotation_start(a: RotationStartIn, x_token: str = Header(default="")):
    """the drafts, once the owner has seen the cost"""
    from . import rotation
    _check_token(x_token)
    if a.confirm is not True:
        raise HTTPException(428, "Start writing a new hidden set: " + rotation.plan()[
            "estimate"]["line"])
    try:
        made = rotation.start(a.by)
    except PermissionError as e:
        raise HTTPException(403, str(e)) from None
    except (ValueError, llm.LLMError) as e:
        raise HTTPException(422, str(e)) from None
    return {"drafts": made, "plan": rotation.plan()}


@app.get("/api/everyday/rotation/candidates")
def rotation_candidates(by: str = ""):
    """the old hidden questions worth retiring, with why — the owner's, and an
    opening of the hidden half, so logged"""
    from . import rotation
    if not _is_owner(by):
        raise HTTPException(403, "Only the board's owner opens the hidden half")
    got = rotation.candidates()
    db.hidden_audit_add(by.strip()[:80], "Everyday's hidden set · the switch's review", len(got))
    return {"candidates": got, "warning": AUDIT_WARNING}


class RotationSwitchIn(BaseModel):
    by: str = ""
    confirm: bool = False
    retire: list[str] = []


@app.post("/api/everyday/rotation/switch")
def rotation_switch(a: RotationSwitchIn, x_token: str = Header(default="")):
    from . import rotation
    _check_token(x_token)
    try:
        rec = rotation.switch(a.by, a.retire, a.confirm)
    except PermissionError as e:
        raise HTTPException(403, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _cache.update(key=None, payload=None, at=0.0)
    return rec


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


def _tasks_stale() -> bool:
    """The bank has questions the harness tasks do not: something was accepted
    or imported since the tasks were last written (or they never were)."""
    bank = [p.stat().st_mtime for p in exam_build.bank_dir(config.EXAM_DIR).glob("*.jsonl")
            if p.stat().st_size]
    if not bank:
        return False
    tasks = [p.stat().st_mtime for p in exam_build.tasks_dir(config.EXAM_DIR).glob("exam_*.jsonl")]
    return not tasks or max(bank) > min(tasks)


def _rebuild_if_idle() -> dict:
    """Make the bank sittable: write the harness tasks, the way the old
    "Rebuild the harness tasks from the bank" button did — an implementation
    step a person should never have had to know about. Not while a judged run
    is using the tasks; then the page offers it once that run is done."""
    busy = next((r for r in db.recent(100) if r["suite"] == "judged"
                 and r["status"] in ("preflight", "waiting_gpu", "waiting_lock", "running")),
                None)
    if busy:
        return {"built": False, "why": f"judged run #{busy['id']} is sitting the exam now — "
                                       f"make the new questions sittable when it is done"}
    try:
        m = exam_build.build(config.OUT_DIR, config.EXAM_DIR)
    except (FileNotFoundError, ValueError, OSError) as e:
        return {"built": False, "why": f"the tasks could not be built: {e}"}
    return {"built": True, "tasks": len(m["tasks"])}


@app.get("/api/exam", dependencies=EXAM_ONLY)
def exam_status():
    why = llm.blocked("exam")
    return {"configured": not why, "reason": why,
            "provider": config.EXAM_PROVIDER, "model": config.EXAM_MODEL,
            "root": str(config.EXAM_DIR), "tasks_built": config.judged_tasks(),
            "target_per_topic": exam_build.TARGET_PER_TOPIC,
            "summary": exam_build.summary(config.EXAM_DIR),
            # new questions the harness tasks do not have yet: the page offers
            # "Make new questions sittable" only then
            "tasks_stale": _tasks_stale(),
            "note": "the published per-topic score comes from the report half; only the "
                    "diagnose half is ever shown here or placed in a request"}


@app.get("/api/exam/candidates", dependencies=EXAM_ONLY)
def exam_candidates(topic: str | None = None, status: str = "candidate"):
    return exam_build.load_candidates(config.EXAM_DIR, topic or None,
                                      None if status == "all" else status)


@app.get("/api/exam/bank", dependencies=EXAM_ONLY)
def exam_bank(topic: str | None = None, half: str | None = None):
    """The bank with report-half text withheld — see exam_build.public_bank.
    11g: half=diagnose is what the Reader asks for — one topic's practice
    half, and of the hidden half only a count: no hidden question's qid
    either."""
    if half is not None:
        if half != "diagnose":
            raise HTTPException(422, "only the practice (diagnose) half can be read")
        if not topic or topic not in exam_build.TOPICS:
            raise HTTPException(404, f"no topic {topic!r}")
        return reader.bank_practice(topic)
    return exam_build.public_bank(config.EXAM_DIR, topic or None)


@app.post("/api/exam/candidates/{cid}/accept", dependencies=EXAM_ONLY)
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
            "half": exam_build.half_of(rec["qid"]), "edited": rec["edited"], "accepted_by": who,
            "build": _rebuild_if_idle()}


@app.post("/api/exam/candidates/{cid}/reject", dependencies=EXAM_ONLY)
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
        row = {"qid": rec["qid"], "half": half, "change": what}
        if half == "diagnose":
            row.update(prompt=rec["prompt"], meta=rec["meta"], reference=rec["reference"])
        else:
            # 12p.1: its reference and its meta too — `meta.intent` says what the
            # question asks, in a sentence of its own — as public_bank withholds them
            row.update(prompt=None, meta=None, reference=None,
                       withheld="report half — never shown, never exported")
        shown.append(row)
    return ({k: v for k, v in plan.items() if k not in ("records", "updates")}
            | {"items": shown})


@app.post("/api/exam/import/preview", dependencies=EXAM_ONLY)
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


@app.post("/api/exam/import", dependencies=EXAM_ONLY)
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
    # sittable at once: a person imports questions to have them asked
    return {**out, "build": _rebuild_if_idle()}


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


@app.get("/api/exam/rubrics", dependencies=EXAM_ONLY)
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


@app.get("/api/exam/rubrics/{name}", dependencies=EXAM_ONLY)
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


@app.get("/api/exam/rubrics/{name}/read", dependencies=EXAM_ONLY)
def exam_rubric_read(name: str, kind: str = "rubric"):
    """11g: a rubric or a criteria file shaped for the Reader — the criteria
    through the judge's own normalise_criteria()."""
    if not re.fullmatch(r"[a-z0-9_]+", name) or kind not in ("rubric", "criteria"):
        raise HTTPException(404, "no such rubric")
    try:
        return reader.rubric_view(name) if kind == "rubric" else reader.criteria_view(name)
    except KeyError:
        raise HTTPException(404, f"no {kind} file for {name}") from None
    except ValueError as e:
        raise HTTPException(422, f"{name}.criteria.json is not valid JSON: {e}") from None


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
        # the author's shape, not ours: the 36 rubrics of the 37-topic exam
        # open with '# <Topic> Evaluation Criteria' and anchor with '### 4 —
        # Strong', and they grade. A heading and five anchors are what the
        # judge needs; a version is recorded when there is one
        if not re.search(r"^# \S", body.content, re.M):
            problems.append("a rubric starts with a '# <title>' heading — DRAFT in it is how "
                            "a rubric says it is not signed off yet")
        for anchor in range(5):
            if not re.search(rf"\*\*{anchor}\*\*|^#{{1,6}}\s*{anchor}(?:\s|$)", body.content,
                             re.M):
                problems.append(f"no anchor for {anchor} (a '**{anchor}**' or a '### {anchor}' "
                                f"heading) — the 0-4 scale needs all five")
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
                    "instruments and are not comparable; test this topic again."),
    }


@app.post("/api/exam/rubrics/preview", dependencies=EXAM_ONLY)
def exam_rubric_preview(body: RubricIn, x_token: str = Header(default="")):
    _check_token(x_token)
    _name(body.approver, "changing a rubric")
    return _rubric_check(body)


@app.post("/api/exam/rubrics", dependencies=EXAM_ONLY)
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


@app.post("/api/exam/build", dependencies=EXAM_ONLY)
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


@app.get("/api/judge/justifications", dependencies=EXAM_ONLY)
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

# the loop's steps, in order: Import → Sit → Propose → Review → Generate →
# Train. "Read the results" is not a step — reading is what the topic page is
# for — and the step no longer depends on anything a browser remembers, so
# two people looking at one board see the same next step.
STEPS = {
    "import": "Import a bank",
    "sit": "Sit the exam",
    "propose": "Propose",
    "review": "Review the spec",
    "generate": "Generate",
    "hand": "Train",
    "unreadable": "Rubric unreadable",
}


def _judged_at(model: str, t: dict | None = None) -> float | None:
    """When this topic's grades landed: the `judged_at` the merge wrote on
    it. The file's mtime is only the time of the last merge — a one-topic run
    of economics made last night's law, medicine and physics read 09:25 — so
    it is the answer only for a file from before topics carried their own."""
    if t and t.get("judged_at"):
        return float(t["judged_at"])
    p = config.OUT_DIR / model.replace("/", "__") / "judge.json"
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _last_judged(payload: dict, task: str, model: str | None = None) -> dict | None:
    """Where this topic stands for one model — `model`'s judged run of it —
    or, with no model, the most recently graded one."""
    best = None
    for m in payload.get("models") or []:
        if model and m["id"] != model:
            continue
        t = ((m.get("judge") or {}).get("tasks") or {}).get(task)
        if not t:
            continue
        at = _judged_at(m["id"], t) or 0
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
    return _dataset_items(d).get("kept")


def _dataset_items(d: dict) -> dict:
    """The accounting the poller wrote: requested, kept, missing. A dataset
    made before 11a has no `missing`, and the page says its reasons were not
    recorded rather than saying nothing."""
    try:
        return (json.loads(d.get("provenance") or "{}") or {}).get("items") or {}
    except ValueError:
        return {}


def _loop_row(topic: str, payload: dict, props: list[dict], sets: list[dict],
              blocked: str, model: str | None = None) -> dict:
    task = exam_build.topic_task(topic)
    bank = (exam_build.summary(config.EXAM_DIR) or {}).get(topic) or {}
    rub = _rubric_row(topic)
    # every score on the board is ONE model's: after #48 and #49 one column
    # held the 135M on economics and medicine and the 360M on the rest
    last = _last_judged(payload, task, model)
    mine = [p for p in props if p["category"] == topic]
    open_p = next((p for p in mine if p["status"] in ("pending", "proposed", "approved")), None)
    ds = [d for d in sets if d.get("proposal_id") in {p["id"] for p in mine}]
    ready = [d for d in ds if d.get("status") == "ready"]
    judged_rows = [m for m in payload.get("models") or []
                   if ((m.get("judge") or {}).get("tasks") or {}).get(task)]
    gates = {m["id"]: propose_gate(m, topic) for m in judged_rows}
    gate = gates.get(last["model"]) if last else None
    if rub.get("error"):
        # a topic whose instrument cannot be read cannot be sat, judged or
        # proposed from; the row says which file is missing rather than
        # offering a button that would fail later
        step, ok, why = "unreadable", False, rub["error"]
    elif not bank.get("accepted"):
        step, ok, why = "import", True, ""
    elif open_p and open_p["status"] == "approved":
        step = "hand" if ready else "generate"
        ok, why = True, ""
    elif open_p:
        step, ok, why = "review", True, ""
    elif ready:
        step, ok, why = "hand", True, ""
    elif not last:
        step, ok, why = "sit", not blocked, blocked
    else:
        # proposing opens the topic page; the gate is that model's, and a
        # judge-only refusal still leaves the way in open (Propose… asks)
        g = gate or {}
        step = "propose"
        ok = bool(g.get("ok") or g.get("overridable"))
        why = "" if g.get("ok") else (g.get("why") or "")
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
        # decides for itself whether a topic may be proposed from. One per
        # judged model: the topic page proposes for the model you are reading
        "propose": gates.get(last["model"]) if last else None,
        "propose_by_model": gates,
        "proposal": {**{k: open_p[k] for k in ("id", "status", "model", "category",
                                               "created_at", "requested_by")},
                     "override": prop.override_of(open_p)} if open_p else None,
        # kept/dropped live in the provenance record the gate wrote, which is
        # the only place that count is authoritative
        "datasets": [{"id": d["id"], "status": d.get("status"), "count": d.get("count"),
                      "kept": _dataset_kept(d), "created_at": d.get("created_at"),
                      # the same line the Review card reads (11e): None before 11a
                      "missing": _dataset_items(d).get("missing"),
                      "over_provisional_judge": prop.override_of(
                          next((q for q in mine if q["id"] == d.get("proposal_id")), {}))}
                     for d in ds],
        "next": {"step": step, "ok": ok, "why": why,
                 "label": "Propose…" if step == "propose" and gate and gate.get("overridable")
                 else STEPS[step]},
    }


def _loop_row_safe(topic: str, payload: dict, props: list[dict], sets: list[dict],
                   blocked: str, model: str | None = None) -> dict:
    """A row that cannot take the board down with it. One topic's files being
    unreadable is a fact about that topic, and the other fourteen rows are
    still the answer to 'what do I do next'."""
    try:
        return _loop_row(topic, payload, props, sets, blocked, model)
    except Exception as e:                                  # noqa: BLE001
        why = f"this topic could not be read: {e}"
        return {"topic": topic, "task": exam_build.topic_task(topic),
                "slug": exam_build.task_slug(exam_build.topic_task(topic)),
                "bank": {"accepted": 0, "report": 0, "diagnose": 0, "awaiting": 0,
                         "floor": report.PROPOSE_MIN_N, "under_floor": False},
                "rubric": {"name": "", "own": False, "version": "", "status": "",
                           "scoring": "single", "sha256": "", "criteria_count": 0,
                           "criteria_sha256": ""},
                "last_judged": None, "propose": None, "propose_by_model": {},
                "proposal": None, "datasets": [], "error": why,
                "next": {"step": "unreadable", "label": STEPS["unreadable"], "ok": False,
                         "why": why}}


def _judged_models(payload: dict) -> list[dict]:
    """The models with judged exam topics, most topics first (then the most
    recently graded): the board's model picker, and its default."""
    out = []
    for m in payload.get("models") or []:
        ts = {t: v for t, v in ((m.get("judge") or {}).get("tasks") or {}).items()
              if t in exam_build.TASK_TOPIC}
        if ts and not m.get("duplicateOf"):
            out.append({"id": m["id"], "name": m.get("name") or m["id"], "topics": len(ts),
                        "at": max((_judged_at(m["id"], v) or 0) for v in ts.values())})
    return sorted(out, key=lambda x: (-x["topics"], -x["at"], x["id"]))


PACE_RUNS = 5


def judged_pace() -> dict | None:
    """How long a judged run takes per answer, from the last few that
    answered anything: the GPU time the runner recorded plus the judge
    batch's own time, over the items the judge graded. The Sit and Queue
    panels turn a topic count into minutes with it — never a constant, which
    would be wrong on the next card and the next judge. None before any run."""
    runs = {r["batch_id"]: r for r in db.judge_runs(200)}
    secs = items = used = 0.0
    for s in db.recent(300):
        if used >= PACE_RUNS:
            break
        run = runs.get(s.get("judge_batch") or "")
        if s["suite"] != "judged" or s["status"] != "done" or not (s.get("gpu_seconds") or 0) \
                or not run or run["status"] != "done" or not run.get("finished_at") \
                or not run.get("n_items"):
            continue
        secs += s["gpu_seconds"] + max(0.0, run["finished_at"] - run["created_at"])
        items += run["n_items"]
        used += 1
    return {"sec_per_answer": round(secs / items, 3), "runs": int(used)} if items else None


def built_items() -> dict[str, int]:
    """task -> how many questions the built task holds: what a ticked topic
    costs in answers."""
    try:
        m = json.loads((config.JUDGED_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {t: int(v.get("items") or 0) for t, v in (m.get("tasks") or {}).items()
            if isinstance(v, dict)}


@app.get("/api/loop", dependencies=EXAM_ONLY)
def loop_board(model: str = ""):
    """One row per topic: the bank, the rubric, where `model`'s judged run
    left it (default: the model with the most judged topics), and the one
    next step."""
    payload = results_payload()
    props = db.proposal_list(limit=500)
    sets = db.dataset_list(limit=500)
    blocked = config.judged_blocked()
    models = _judged_models(payload)
    chosen = model if any(m["id"] == model for m in models) else (models[0]["id"] if models
                                                                   else "")
    return {"topics": [_loop_row_safe(t, payload, props, sets, blocked, chosen or None)
                       for t in exam_build.TOPICS],
            "model": chosen, "models": models,
            "judged_blocked": blocked, "tasks_built": config.judged_tasks(),
            "judge_health": judge_health(),
            "built_items": built_items(), "pace": judged_pace(),
            "floor": report.PROPOSE_MIN_N,
            "gap_dataset_flag": "--gap-dataset"}


@app.get("/api/answers", dependencies=EXAM_ONLY)
def answers(model: str, topic: str, limit: int = 200):
    """One model's DIAGNOSE-half answers on one topic, with what the judge
    made of each. The report half is here as one aggregate line and nothing
    else: not its questions, not its answers, not its qids. An answer quotes
    its question often enough that showing one shows the other."""
    if topic not in exam_build.TOPICS:
        raise HTTPException(404, f"{topic!r} is not an exam topic")
    task = exam_build.topic_task(topic)
    model_dir = config.OUT_DIR / model.replace("/", "__")
    if not (model_dir / "judge.json").exists():
        raise HTTPException(404, f"{model} has no judged run on file — test it on the "
                                 f"Knowledge exam first")
    # only answers to the questions the topic holds now: a retired answer
    # shown against a new topic's question would be an answer to another one
    j, earlier = judge_now(model)
    t = ((j or {}).get("tasks") or {}).get(task)
    if not t:
        if any(e.get("task") == task for e in earlier):
            raise HTTPException(404, f"{model} sat {topic!r} only on an earlier question set — "
                                     f"its answers are history; sit the current one")
        raise HTTPException(404, f"{model}'s judged run does not cover {topic!r}")
    import judge as _judge
    spec = _judge.rubric_for(task).criteria or {}
    answers_by_hash = {}
    for rec in _judge._records(model_dir, task):
        answers_by_hash[rec.get("doc_hash")] = ((rec.get("doc") or {}), _judge.answer_parts(rec))
    rows = []
    for it in t.get("items") or []:
        if it.get("half") != "diagnose":
            continue                      # the whole safety property, in one line
        doc, parts = answers_by_hash.get(it.get("doc_hash"), ({}, None))
        parts = parts or {"answer_text": "", "reasoning_text": "", "had_reasoning": False,
                          "reasoning_unterminated": False, "no_answer": False}
        rows.append({
            "qid": it.get("qid"), "score": it.get("score"), "graded": it.get("graded"),
            "meta": it.get("meta") or {}, "answer_words": it.get("answer_words"),
            "prompt": doc.get("prompt") or "", "reference": doc.get("reference") or "",
            # 11l: the answer is what the judge graded — a reasoning block is
            # not part of it. The reasoning rides along for anyone checking
            "answer": parts["answer_text"], "reasoning": parts["reasoning_text"],
            "had_reasoning": parts["had_reasoning"],
            "reasoning_unterminated": parts["reasoning_unterminated"],
            "no_answer": bool(it.get("no_answer") or parts["no_answer"]),
            "criteria": it.get("criteria") or {},
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
        # 11l: answers that never left a reasoning block — counted, not scored
        "no_answer": t.get("no_answer") or 0, "no_score": t.get("no_score"),
    }


@app.get("/api/judge", dependencies=EXAM_ONLY)
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


def _spend_check(n_items: int, provider: str | None = None) -> None:
    """The per-batch limit, and the day's limit for the provider this request
    goes to — its own use only: a judge batch on another provider, or on the
    same local server, is not this request's spend."""
    if n_items > config.LLM_MAX_ITEMS_PER_BATCH:
        raise HTTPException(422, f"{n_items} batch items exceeds LLM_MAX_ITEMS_PER_BATCH="
                                 f"{config.LLM_MAX_ITEMS_PER_BATCH}")
    provider = provider or config.LLM_PROVIDER
    cap = config.daily_cap(provider)
    if cap is None:
        return
    used = db.llm_items_today(provider)
    if used + n_items > cap:
        raise HTTPException(429, f"today's use of {provider} ({used} items) plus this request "
                                 f"({n_items}) would pass {config.daily_cap_name(provider)}="
                                 f"{cap}; try tomorrow or raise the cap")


def llm_usage() -> list[dict]:
    """Today's items per provider, against each one's cap — every provider
    used today and every one configured, so the Review tab shows the judge's
    spend beside the generator's."""
    used = db.llm_items_today_by_provider()
    names = [p for p in (config.LLM_PROVIDER, config.JUDGE_PROVIDER, config.EXAM_PROVIDER)
             if p and p != "stub"]
    names += sorted(p for p in used if p not in names)
    return [{"provider": p, "items": used.get(p, 0), "cap": config.daily_cap(p),
             "cap_name": config.daily_cap_name(p),
             "roles": [r for r, v in (("generator", config.LLM_PROVIDER),
                                      ("judge", config.JUDGE_PROVIDER),
                                      ("exam writer", config.EXAM_PROVIDER)) if v == p]}
            for p in dict.fromkeys(names)]


def _require_llm() -> llm.Backend:
    why = llm.blocked()
    if why:
        raise HTTPException(503, why)
    try:
        return llm.client()
    except llm.LLMError as e:      # a local server that is down, or serves another model
        raise HTTPException(503, str(e)) from None


_EVIDENCE: dict[tuple, bool] = {}


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _has_evidence(model: str, task: str) -> bool:
    """Whether the judge wrote anything a proposal could be built from: an
    assessment of a diagnosis-half answer that fell short. Cached against the
    two files it reads, because the Loop board asks on every poll."""
    model_dir = config.OUT_DIR / model.replace("/", "__")
    key = (model, task, _mtime(model_dir / "judge.json"),
           _mtime(exam_build.tasks_dir(config.EXAM_DIR) / f"{task}.jsonl"))
    if key not in _EVIDENCE:
        if len(_EVIDENCE) > 512:
            _EVIDENCE.clear()
        _EVIDENCE[key] = bool(prop.justifications_for(model_dir, task, limit=1)[0])
    return _EVIDENCE[key]


def propose_gate(row: dict | None, topic: str, evidence: bool = True) -> dict:
    """Whether `row`'s model may propose from `topic`, with every reason, in
    the shape the page and POST /api/proposals both read:
    {ok, overridable, soft, hard, why, short, caution, provisional}.

    `soft` reasons are about the judge and, with ALLOW_PRELIMINARY_OVERRIDE,
    can be overridden behind a warning; `hard` reasons are about the data and
    the queue and never can. With the setting off this is the gate as it was
    before the split, in the same words. One function, so the page never
    keeps its own copy of the rule."""
    task = exam_build.topic_task(topic)
    allow = config.ALLOW_PRELIMINARY_OVERRIDE
    judge = (row or {}).get("judge") or {}
    t = (judge.get("tasks") or {}).get(task)
    g = (t or {}).get("propose") or {}
    hard: list[tuple[str, str]] = []
    if judge.get("skipped"):
        hard.append((f"the judge did not grade this model: {judge['skipped']}",
                     "the judge did not grade this model"))
    elif not t:
        hard.append((f"{(row or {}).get('id', 'this model')} has no judged answers on file for "
                     f"{topic!r} — sit the exam first", "no judged answers on file"))
    hard += [(h["why"], h["short"]) for h in g.get("hard") or []]
    if row and t:
        dup = db.proposal_active(row["id"], task, topic)
        if dup:
            hard.append((f"proposal #{dup['id']} for this model and topic is already "
                         f"{dup['status']} — open it on Improve",
                         f"proposal #{dup['id']} is already {dup['status']}"))
        if evidence and not hard and not _has_evidence(row["id"], task):
            hard.append(("the judge wrote no comment on a practice answer that scored below 3 "
                         "of 4 on this topic — nothing to propose from",
                         "no practice answer scored below 3 of 4 — nothing to propose from"))
    soft = list(g.get("soft") or []) + (list(g.get("soft_extra") or []) if allow else [])
    if not t:
        soft = []
    if not allow and g and not g.get("ok"):
        why, short = g.get("why"), g.get("short")        # the gate as it was, word for word
    elif hard:
        why, short = "; ".join(h[0] for h in hard), hard[0][1]
    elif soft:
        why = ("the judged suite is preliminary, so no topic score is evidence yet: "
               + "; ".join(soft))
        short = ("a provisional judge — proposing marks everything made from it" if allow
                 else "the judged suite is preliminary")
    else:
        why = short = None
    ok = not soft and not hard
    return {"ok": ok, "overridable": bool(allow and soft and not hard),
            "soft": soft, "hard": [h[0] for h in hard], "why": why, "short": short,
            "caution": g.get("caution") if ok else None,
            "provisional": bool(g.get("provisional"))}


IMPROVE_SERVED = ("Improve trains the model's weights, and this one is served elsewhere: there "
                  "are none here to train.")


@app.post("/api/proposals")
def proposal_create(p: ProposalIn, x_token: str = Header(default="")):
    """Ask the LLM what skill is missing, from the judge's written assessments
    of this topic's DIAGNOSE-half answers. The gate is enforced here, not just
    on the button: a preliminary judged suite, a topic under the noise floor,
    or a model that wrote nothing is refused with the same words the page
    shows."""
    _check_token(x_token)
    # 16.5: an exam topic, while the Knowledge exam is switched off
    if p.topic and not p.everyday and not config.KNOWLEDGE_EXAM:
        raise HTTPException(409, config.EXAM_OFF + ". Nothing was proposed.")
    # 14.4.4: a restricted set is never a training target
    why = _restrictions().never_trained([p.topic, p.everyday])
    if why:
        raise HTTPException(422, why + ". Nothing was proposed.")
    if served.is_served(p.model):
        raise HTTPException(422, IMPROVE_SERVED)
    backend = _require_llm()
    payload = results_payload()
    row = next((m for m in payload["models"] if m["id"] == p.model), None)
    if not row:
        raise HTTPException(404, f"no such model on the board: {p.model}")
    if p.everyday.strip():
        return _propose_everyday(p, backend)
    task = exam_build.topic_task(p.topic)
    judge = row.get("judge") or {}
    t = (judge.get("tasks") or {}).get(task)
    if not t:
        raise HTTPException(404, f"{p.model} has no judged answers on file for {p.topic!r} — "
                                 f"test it on the Knowledge exam first")
    if t.get("propose") is None:
        raise HTTPException(422, f"{p.topic!r} is not an exam topic")
    # recomputed on every request: the page's copy is a courtesy, this is the rule
    gate = propose_gate(row, p.topic)
    override = None
    if gate["hard"] or (not config.ALLOW_PRELIMINARY_OVERRIDE and not gate["ok"]):
        raise HTTPException(409, gate["why"])
    if gate["soft"]:
        if not (p.override_preliminary and gate["overridable"]):
            raise HTTPException(409, gate["why"])
        if not p.requested_by.strip():
            raise HTTPException(422, "a name is recorded on a proposal made over a provisional "
                                     "judge — type yours first")
        override = {"by": p.requested_by.strip()[:80], "at": time.time(),
                    "reasons": gate["soft"]}
    _spend_check(1)
    model_dir = config.OUT_DIR / p.model.replace("/", "__")
    justifications, counts = prop.justifications_for(model_dir, task)
    if not justifications:
        raise HTTPException(409, "the judge wrote no comment on a practice answer that scored "
                                 "below 3 of 4 on this topic — nothing to propose from")
    jmeta = judge.get("judge") or {}
    evidence = {
        "n_shown": len(justifications), **counts,
        "topic_score_report": t.get("score_report"), "topic_n_report": t.get("n_report"),
        "topic_score_diagnose": t.get("score_diagnose"), "topic_n_diagnose": t.get("n_diagnose"),
        "judge_id": jmeta.get("id"),
        # what the reviewer sees of what the LLM saw — diagnosis half, with any
        # question text the judge quoted already stripped
        "examples": justifications[:prop.EXAMPLES_SHOWN],
        # 11j: which practice answers it read, so the reviewer can see exactly
        # those — diagnose-half qids, never the hidden half's
        "qids_read": [f["qid"] for f in justifications],
    }
    pid = db.proposal_create(p.model, task, p.topic, p.requested_by.strip()[:80], evidence)
    if override:
        db.proposal_update(pid, override=json.dumps(override))
        print(f"[propose] #{pid} {p.model} · {p.topic}: proposed over a provisional judge by "
              f"{override['by']} — {'; '.join(override['reasons'])}")
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


def _propose_everyday(p: ProposalIn, backend) -> dict:
    """12g.2: a proposal for an Everyday group, from the PRACTICE questions of
    that group the model failed — never a hidden one. The group must have at
    least EVERYDAY_MIN_HIDDEN hidden questions: a score from fewer is noise."""
    import everyday as ev
    g = p.everyday.strip()
    known = ev.groups()
    if g not in known:
        raise HTTPException(422, f"{g!r} is not an Everyday group — the groups are "
                                 + ", ".join(known))
    label = known[g]
    hidden = ev.split_counts().get(g, {}).get("hidden", 0)
    if hidden < config.EVERYDAY_MIN_HIDDEN:
        need = config.EVERYDAY_MIN_HIDDEN - hidden
        raise HTTPException(409, f"{label} has {hidden} hidden questions and Improve takes a "
                                 f"group at {config.EVERYDAY_MIN_HIDDEN}: it needs {need} more")
    task = prop.EVERYDAY_PREFIX + g
    dup = db.proposal_active(p.model, task, label)
    if dup:
        raise HTTPException(409, f"proposal #{dup['id']} for this model and group is already "
                                 f"{dup['status']} — open it on Improve")
    model_dir = config.OUT_DIR / p.model.replace("/", "__")
    failed, counts = prop.everyday_failures(model_dir, g)
    if not counts["current"]:
        raise HTTPException(409, f"{p.model}'s everyday answers are to an earlier version of the "
                                 f"questions — run everyday tasks again first")
    if not failed:
        raise HTTPException(409, f"no practice question in {label} failed — nothing to propose "
                                 f"from")
    _spend_check(1)
    marks = ev.read(model_dir) or {}
    evidence = {
        "kind": "everyday", "group": g, "n_shown": len(failed),
        "practice_total": counts["total"], "practice_failed": counts["failed"],
        # the published score, its hidden half — a count, never a question
        "hidden": (marks.get("groups") or {}).get(g),
        # what the AI read, for the reviewer: practice requests and why each failed
        "failed": [{k: f[k] for k in ("id", "qid", "skill", "prompt", "reason")} for f in failed],
        "skills": prop.everyday_focus(failed),
        "qids_read": [f["qid"] for f in failed],
    }
    pid = db.proposal_create(p.model, task, label, p.requested_by.strip()[:80], evidence)
    req = prop.everyday_proposal_request(pid, p.model, g, label, failed, counts)
    try:
        bid = backend.submit([req])
    except llm.LLMError as e:
        db.proposal_update(pid, status="failed", error=str(e)[:400])
        raise HTTPException(502, f"the LLM batch could not be submitted: {e}") from None
    db.batch_add(bid, "proposal", pid, 1, backend.name, backend.model)
    db.proposal_update(pid, batch_id=bid, prompt_sha=llm.prompt_sha(req.system, req.user))
    return {"id": pid, "status": "pending", "batch_id": bid, "task": task}


def _override_of(r: dict) -> dict | None:
    try:
        return json.loads(r.get("override") or "null") or None
    except (ValueError, TypeError):
        return None


def _proposal_view(r: dict, datasets: list[dict] | None = None) -> dict:
    out = dict(r)
    try:
        out["evidence"] = json.loads(r.get("evidence") or "{}")
    except (ValueError, TypeError):
        out["evidence"] = {}
    out["override"] = _override_of(r)
    out["datasets"] = [{"id": d["id"], "status": d["status"], "fmt": d["fmt"],
                        "count": d["count"], "error": d["error"]}
                       for d in (datasets or []) if d["proposal_id"] == r["id"]]
    return out


@app.get("/api/proposals")
def proposal_index(status: str | None = None, limit: int = 200):
    ds = db.dataset_list(500)
    return [_proposal_view(r, ds) for r in db.proposal_list(status, min(limit, 500))
            if not exam_hidden(r.get("task"))]                       # 16.5


@app.get("/api/proposals/{pid}")
def proposal_detail(pid: int):
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    _exam_prop_gate(r)                                       # 16.5
    return _proposal_view(r, db.dataset_list(500))


# ---------------------------------------------------------------------------
# 12n.1: Everyday questions read and edited where they are read. A hidden
# question is the board's owner's alone, from the audit of its group; every
# opening of a hidden half is logged. An edit is saved beside what the
# question builder published, and every stored answer is marked again at
# once — no model runs; the judge only for the question edited
# ---------------------------------------------------------------------------

HIDDEN_OWNER = ("A hidden question is read and edited only by the board's owner, from its "
                "group's audit")
AUDIT_WARNING = ("These questions are the test. Don’t train on them or write questions toward "
                 "them. This opening is logged.")


def _everyday_paused() -> str:
    """12p.1: '' when Everyday may run and be scored; else the banner's words"""
    import everyday as ev
    return ev.hidden_status()["why"]


def _is_owner(by: str) -> bool:
    return config.is_owner(by)


def _evq(qid: str, by: str):
    """a question to read or edit, as it reads now — or, retired, as it read
    last; a hidden one only for the owner"""
    import everyday as ev
    q = ev.question(qid)
    if q is None:
        last = [r for r in ev.edit_log() if r["id"] == qid]
        q = last[-1]["before"] if last else None
    if q is None:
        raise HTTPException(404, f"No question {qid}")
    if ev.half(q) == ev.HIDDEN and not _is_owner(by):
        raise HTTPException(403, HIDDEN_OWNER)
    return ev, q


def _evq_view(ev, q: dict) -> dict:
    return {**{k: q.get(k) for k in ("id", "group", "prompt", "reference", "checks", "skill",
                                      "edited")},
            "half": ev.half(q), "rubric": (ev.judge_check(q) or {}).get("rubric"),
            "retired": ev.question(q["id"]) is None}


@app.get("/api/everyday/questions/{qid}")
def evd_question(qid: str, by: str = ""):
    ev, q = _evq(qid, by)
    return {"question": _evq_view(ev, q), "history": ev.history(qid),
            "groups": [[k, v] for k, v in ev.groups().items()]}


class EvdEditIn(BaseModel):
    changes: dict = {}
    why: str = ""
    by: str = ""


@app.post("/api/everyday/questions/{qid}/impact")
def evd_impact(qid: str, f: EvdEditIn, x_token: str = Header(default="")):
    _check_token(x_token)
    ev, _ = _evq(qid, f.by)
    try:
        return ev.impact(qid, f.changes, config.OUT_DIR)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


def _evd_saved(ev, rec: dict) -> dict:
    """after a save: every model's answers marked again, the judge asked
    only about this question, and the page's payload rebuilt"""
    note = ""
    try:
        res = ev.remark(config.OUT_DIR, judge=True, only={rec["id"]}, snapshot=False) \
            if config.OUT_DIR.is_dir() else {}
    except RuntimeError as e:                       # the judge isn't set up: marked, waiting
        res, note = {}, str(e)
    _cache.update(key=None, payload=None, at=0.0)
    return {"edit": {k: rec.get(k) for k in ("n", "id", "action", "at", "by", "why", "what")},
            "remarked": len((res or {}).get("models") or {}), "sent": (res or {}).get("sent", 0),
            "note": note, "version": ev.version()}


@app.post("/api/everyday/questions/{qid}/edit")
def evd_edit(qid: str, f: EvdEditIn, x_token: str = Header(default="")):
    _check_token(x_token)
    ev, _ = _evq(qid, f.by)
    try:
        rec = ev.edit(qid, f.changes, f.why, f.by)
    except KeyError:
        raise HTTPException(404, f"No question {qid}") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return _evd_saved(ev, rec)


@app.post("/api/everyday/questions/{qid}/retire")
def evd_retire(qid: str, f: EvdEditIn, x_token: str = Header(default="")):
    _check_token(x_token)
    ev, _ = _evq(qid, f.by)
    try:
        rec = ev.retire(qid, f.why, f.by)
    except KeyError:
        raise HTTPException(404, f"No question {qid}") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return _evd_saved(ev, rec)


@app.post("/api/everyday/questions/{qid}/undo")
def evd_undo(qid: str, f: EvdEditIn, x_token: str = Header(default="")):
    _check_token(x_token)
    ev, _ = _evq(qid, f.by)
    try:
        rec = ev.undo(qid, f.by)
    except KeyError:
        raise HTTPException(404, f"No edit of {qid}") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return _evd_saved(ev, rec)


class AuditIn(BaseModel):
    group: str
    by: str = ""
    confirm: bool = False
    models: list[str] = []


@app.post("/api/everyday/audit")
def evd_audit(a: AuditIn, x_token: str = Header(default="")):
    """12n.1: a group's hidden half, for the board's owner, after the
    warning — logged before anything is shown"""
    _check_token(x_token)
    import everyday as ev
    if not _is_owner(a.by):
        raise HTTPException(403, "Only the board's owner opens the hidden half")
    if a.confirm is not True:
        raise HTTPException(428, AUDIT_WARNING)
    if a.group not in ev.groups():
        raise HTTPException(404, f"No group {a.group}")
    qs = [q for q in ev.load_bank() if q["group"] == a.group and ev.half(q) == ev.HIDDEN]
    db.hidden_audit_add(a.by.strip()[:80], a.group, len(qs))
    ids, now = {q["id"] for q in qs}, ev.version()["hash"]
    answers, changed = {}, {}
    for f in sorted(config.OUT_DIR.glob("*/everyday.json")) if config.OUT_DIR.is_dir() else []:
        e = ev.read(f.parent) or {}
        if (e.get("version") or {}).get("hash") != now or not e.get("model") \
                or (a.models and e["model"] not in a.models):
            continue
        answers[e["model"]] = {it["id"]: {k: it.get(k) for k in (
            "pass", "reason", "failed", "answer_text", "had_reasoning", "reasoning_text",
            "no_answer")} for it in e.get("items") or [] if it["id"] in ids}
        changed[e["model"]] = [i for i in e.get("changed") or [] if i in ids]
    return {"group": a.group, "warning": AUDIT_WARNING,
            "questions": [{**_evq_view(ev, q), "describe": [ev.describe(c) for c in q["checks"]]}
                          for q in qs],
            "answers": answers, "changed": changed}


@app.get("/api/store")
def store_status():
    """12p.1: where Everyday's hidden set and the exam's report half are, and
    the backups — counts and states, never a question"""
    from . import hidden_store
    return hidden_store.overview()


@app.get("/api/everyday/audits")
def evd_audits():
    """every opening of a hidden half: who, when, which group — Data & sources lists them"""
    return {"owner": config.BOARD_OWNER, "audits": db.hidden_audits()}


# ---------------------------------------------------------------------------
# 12o.2: every benchmark's questions, with each chosen model's result on each
# — the halves that may be listed today, and the owner's audit of the others
# ---------------------------------------------------------------------------

def _qtask(task: str) -> None:
    from . import questions
    if questions.GPQA.match(task):
        raise HTTPException(403, questions.NOT_LISTED)
    if questions.kind_of(task) == "exam" and not config.KNOWLEDGE_EXAM:
        raise HTTPException(409, config.EXAM_OFF + ".")             # 16.5
    if task not in questions.tasks():
        if task == questions.MMF:                                     # 14.4.5: off, or not here
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
            import mobile_mmlu
            raise HTTPException(403 if not mobile_mmlu.full_on() else 404,
                                mobile_mmlu.full_available() + ".")
        raise HTTPException(404, f"No questions on file for {task}: "
                                 + questions.NOT_YET[0].lower() + questions.NOT_YET[1:] + ".")


def _qmodels(models: str | list[str] | None) -> list[str] | None:
    if isinstance(models, list):
        return [m for m in models if m] or None
    return [m for m in (models or "").split(",") if m.strip()] or None


@app.get("/api/questions")
def questions_list():
    """every benchmark whose questions can be read — 16.8: from a run's answers
    or from its own file — and why the others can't"""
    from . import questions
    ff = questions.from_file()
    return {"tasks": [{"task": t, "kind": questions.kind_of(t), "from_file": t in ff}
                      for t in questions.tasks()],
            "why": {"gpqa": questions.NOT_LISTED, "not_yet": questions.NOT_YET}}


@app.get("/api/questions/{task}")
def questions_page(task: str, offset: int = 0, limit: int = 50, q: str = "", subject: str = "",
                   models: str = "", f: str = ""):
    """a page of a benchmark's listable half, 50 a page, with each chosen
    model's result on each question"""
    from . import questions
    _qtask(task)
    return questions.page(task, offset=offset, limit=limit, q=q, subject=subject,
                          models=_qmodels(models), f=f)


class QuestionsAuditIn(BaseModel):
    by: str = ""
    confirm: bool = False
    models: list[str] | None = None
    offset: int = 0
    limit: int = 50
    q: str = ""
    subject: str = ""
    f: str = ""


@app.post("/api/questions/{task}/audit")
def questions_audit(task: str, a: QuestionsAuditIn, x_token: str = Header(default="")):
    """the half that is never listed — MMLU's and every lm_eval benchmark's
    report half, the exam's, Everyday's hidden one — for the board's owner,
    after the warning, logged before anything is shown. Never GPQA"""
    from . import questions
    _check_token(x_token)
    if not _is_owner(a.by):
        raise HTTPException(403, "Only the board's owner opens the half that is never listed")
    if a.confirm is not True:
        raise HTTPException(428, AUDIT_WARNING)
    _qtask(task)
    got = questions.page(task, offset=a.offset, limit=a.limit, q=a.q, subject=a.subject,
                         models=_qmodels(a.models), f=a.f, half="report")
    what = ("hidden half" if task == "everyday" else "the exam's report half"
            if questions.kind_of(task) == "exam" else "report half")
    db.hidden_audit_add(a.by.strip()[:80], f"{task} · {what}", got["total"])
    return {**got, "warning": AUDIT_WARNING, "audit": what}


def _name(s: str, what: str) -> str:
    s = s.strip()[:80]
    if not s:
        raise HTTPException(422, f"{what} needs a name — the tailnet is the auth boundary, "
                                 f"so the record of who decided is the name you type")
    return s


@app.get("/api/proposals/{pid}/answers")
def proposal_answers(pid: int):
    """The practice answers this proposal's AI read: the question, the model's
    answer, its score and the judge's comment as the AI received it (any
    question wording it quoted already taken out). The reviewer sees the
    questions so they can check its reading; the AI never did.

    Diagnose half only — the same rule as everywhere: no hidden question's
    text, and no hidden qid, leaves the server."""
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    _exam_prop_gate(r)                                       # 16.5
    try:
        ev = json.loads(r.get("evidence") or "{}")
    except (ValueError, TypeError):
        ev = {}
    model_dir = config.OUT_DIR / r["model"].replace("/", "__")
    task = r["task"]
    items, counts = prop.justifications_for(model_dir, task)
    by_qid = {str(f["qid"]): f for f in items}
    read = [str(q) for q in (ev.get("qids_read") or [])]
    # exactly the ones it read, in the order it read them. A proposal made
    # before 11j recorded no list: the same function is deterministic, so
    # recomputing gives the same answers unless the model was judged again
    chosen = [by_qid[q] for q in read if q in by_qid] if read \
        else items[:ev.get("n_shown") or len(items)]
    import judge as _judge
    j, _earlier = judge_now(r["model"])
    t = ((j or {}).get("tasks") or {}).get(task) or {}
    meta = {str(it.get("qid")): it for it in (t.get("items") or [])
            if it.get("half") == "diagnose"}          # the whole safety property
    answers = {}
    for rec in _judge._records(model_dir, task):
        # 11l: the answer, never the reasoning before it
        answers[rec.get("doc_hash")] = ((rec.get("doc") or {}),
                                        _judge.answer_parts(rec)["answer_text"])
    out = []
    for f in chosen:
        it = meta.get(str(f["qid"]))
        if not it:            # judged again since, on other questions
            continue
        doc, ans = answers.get(it.get("doc_hash"), ({}, ""))
        out.append({"qid": f["qid"], "score": f["score"], "comment": f["justification"],
                    "answer_words": f.get("answer_words"), "answer": ans,
                    "question": doc.get("prompt") or "", "reference": doc.get("reference") or ""})
    return {"proposal": pid, "model": r["model"], "topic": r["category"], "task": task,
            "n": len(out), "n_read": len(read or chosen), "recomputed": not read,
            "gone": len(read or chosen) - len(out),
            "counts": {k: counts[k] for k in ("diagnose_items", "diagnose_weak")},
            "items": out,
            "note": "practice questions only — the AI read the comments with the question "
                    "wording taken out"}


@app.post("/api/proposals/{pid}/approve")
def proposal_approve(pid: int, a: ApproveIn, x_token: str = Header(default="")):
    _check_token(x_token)
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    _exam_prop_gate(r)                                       # 16.5
    if r["status"] != "proposed":
        raise HTTPException(409, f"proposal #{pid} is {r['status']}, not awaiting review")
    who = _name(a.approver, "approving")
    edited = a.edited_text.strip()[:2000]
    if edited == r["spec_text"].strip():
        edited = ""                                   # approved as written
    # 11e: Approve freezes the plan the card showed, so what the approver saw
    # is what the generator gets — whatever the bank or the judge does later
    frozen = _focus_to_freeze(r, a.spread)
    db.proposal_update(pid, status="approved", approver=who, edited_text=edited,
                       approved_at=time.time(), approved_focus=json.dumps(frozen))
    return {"id": pid, "status": "approved", "approver": who, "edited": bool(edited),
            "focus_mode": frozen["mode"]}


def _focus_live(r: dict) -> dict:
    # 12g.2: an Everyday group's plan is its failed skills — one batch each
    if prop.is_everyday(r["task"]):
        try:
            failed = json.loads(r.get("evidence") or "{}").get("failed") or []
        except (ValueError, TypeError):
            failed = []
        labels = prop.everyday_focus(failed)
        return {"mode": "skill" if labels else None, "labels": labels,
                "failing": len(failed), "reason": "" if labels else "no failed skill on file"}
    return prop.focus_for(config.OUT_DIR / r["model"].replace("/", "__"), r["task"],
                          r["category"])


def _focus_to_freeze(r: dict, spread: bool) -> dict:
    if not spread:
        return {"mode": "off", "labels": [], "reason": prop.TURNED_OFF}
    f = _focus_live(r)
    if not f["labels"]:
        return {"mode": "off", "labels": [], "reason": f["reason"]}
    return {"mode": f["mode"], "labels": f["labels"], "reason": ""}


@app.post("/api/proposals/{pid}/reject")
def proposal_reject(pid: int, a: RejectIn, x_token: str = Header(default="")):
    _check_token(x_token)
    r = db.proposal_get(pid)
    if not r:
        raise HTTPException(404, "no such proposal")
    _exam_prop_gate(r)                                       # 16.5
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
    _exam_prop_gate(r)                                       # 16.5
    if r["status"] != "approved":
        raise HTTPException(409, f"proposal #{pid} is {r['status']}; only an approved spec "
                                 f"reaches the generator")
    who = _name(g.requester, "generating")
    fmt = g.fmt or prop.formats_for(r["task"])[0]
    if fmt not in prop.FORMATS:
        raise HTTPException(422, f"fmt must be one of {', '.join(prop.FORMATS)} — question-"
                                 f"shaped training data teaches the exam more readily than "
                                 f"prose does, so 'doc' is the default and 'mc' is retired")
    # 12g.2: chat examples are an Everyday group's; documents an exam topic's
    if fmt not in prop.formats_for(r["task"]):
        raise HTTPException(422, "an Everyday group's data is chat examples ('chat')"
                            if prop.is_everyday(r["task"]) else
                            "chat examples are for Everyday groups; an exam topic takes 'doc' "
                            "(or 'free')")
    g.fmt = fmt
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
    everyday = prop.is_everyday(r["task"])
    audience = prop.CHAT_AUDIENCE if everyday else prop.audience_for(r["category"], r["task"],
                                                                       fmt=g.fmt)
    try:
        failed = (json.loads(r.get("evidence") or "{}").get("failed") or []) if everyday else None
    except (ValueError, TypeError):
        failed = []
    # and so does the corner of the topic each request is for. 11e: the first
    # N labels of the plan Approve froze; a proposal approved before 11e keeps
    # 11a's plan, computed now
    frozen = prop.frozen_focus(r)
    if frozen is not None:
        mode = frozen.get("mode") or "off"
        full = frozen.get("labels") or []
        labels = [full[i % len(full)] for i in range(g.count)] if full else []
        counts: dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        plan = [{"domain": lab, "documents": n} for lab, n in counts.items()]
        reqs = prop.generation_requests(did, spec, r["category"], g.count, g.fmt, seed=did,
                                        audience=audience, labels=labels or None, failed=failed)
    else:
        plan = prop.focus_plan(config.OUT_DIR / r["model"].replace("/", "__"), r["task"],
                               r["category"], g.count)
        mode = "area" if plan else "off"
        labels = [f["domain"] for f in plan for _ in range(f["documents"])]
        reqs = prop.generation_requests(did, spec, r["category"], g.count, g.fmt, seed=did,
                                        audience=audience, plan=plan, failed=failed)
    sha = llm.prompt_sha(*[q.system + "\n" + q.user for q in reqs])
    try:
        bid = backend.submit(reqs)
    except llm.LLMError as e:
        db.dataset_update(did, status="failed", finished_at=time.time(), error=str(e)[:400])
        raise HTTPException(502, f"the LLM batch could not be submitted: {e}") from None
    db.batch_add(bid, "generation", did, len(reqs), backend.name, backend.model)
    # what the generator was actually told about its reader, kept for the
    # poller: recomputing it later would read a bank that may have moved
    # what the generator was told, and what each request was asked for: the
    # poller accounts for every document against this, and recomputing it
    # later would read a bank that may have moved
    db.dataset_update(did, batch_id=bid, provenance=json.dumps(
        {"prompt_sha256": sha, "audience": audience, "focus_plan": plan,
         "focus_mode": mode, "focus_labels": labels,
         "requests": [{"k": k, "count": q.meta.get("count"), "focus": q.meta.get("focus")}
                      for k, q in enumerate(reqs)]}))
    return {"dataset_id": did, "status": "pending", "batch_id": bid, "items": len(reqs)}


def _dataset_view(d: dict, props: dict[int, dict]) -> dict:
    out = dict(d)
    try:
        out["provenance"] = json.loads(d.get("provenance") or "{}")
    except (ValueError, TypeError):
        out["provenance"] = {}
    p = props.get(d["proposal_id"]) or {}
    out.update({"model": p.get("model"), "task": p.get("task"), "category": p.get("category"),
                "over_provisional_judge": prop.override_of(p),
                # 11k: what was approved, for the reader's "The missing skill"
                # — a dataset made before the spec was frozen into provenance
                # still has the proposal's own words
                "spec_text": p.get("edited_text") or p.get("spec_text"),
                "download": f"/api/datasets/{d['id']}/items.jsonl" if d["status"] == "ready"
                else None})
    return out


@app.get("/api/datasets")
def dataset_index(limit: int = 200):
    props = {p["id"]: p for p in db.proposal_list(None, 500)}
    return [_dataset_view(d, props) for d in db.dataset_list(min(limit, 500))
            if not exam_hidden((props.get(d["proposal_id"]) or {}).get("task"))]   # 16.5


@app.get("/api/datasets/{did}")
def dataset_detail(did: int):
    d = db.dataset_get(did)
    if not d:
        raise HTTPException(404, "no such dataset")
    _exam_prop_gate(db.proposal_get(d["proposal_id"]) or {})             # 16.5
    return _dataset_view(d, {p["id"]: p for p in db.proposal_list(None, 500)})


@app.get("/api/datasets/{did}/items")
def dataset_items_read(did: int, offset: int = 0, limit: int = reader.PAGE, q: str = ""):
    """11g: the documents, for the Reader — numbered, each with its focus
    label and word count, the missing ones in their place with their reason.
    The items.jsonl download below is unchanged."""
    _exam_dataset_gate(did)                                   # 16.5
    try:
        return reader.dataset_page(did, offset, limit, q)
    except KeyError:
        raise HTTPException(404, "no such dataset") from None


@app.get("/api/datasets/{did}/items.jsonl")
def dataset_items(did: int):
    _exam_dataset_gate(did)                                   # 16.5
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
    _exam_dataset_gate(did)                                   # 16.5: hidden, so kept
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


# ---- 12g.1: what a checkpoint was trained from --------------------------------
# Set once by a person, from the models on the board; a training run that
# recorded its base fills it when no person has. Improve's Retests pair a
# checkpoint with it, so a checkpoint with none is not in Improve.

class TrainedFromIn(BaseModel):
    model: str
    base: str
    by: str


@app.post("/api/trained-from")
def trained_from_set(t: TrainedFromIn, x_token: str = Header(default="")):
    _check_token(x_token)
    who = _name(t.by, "setting what a model was trained from")
    models = {m["id"]: m for m in results_payload()["models"]}
    if t.model not in models:
        raise HTTPException(404, f"no such model on the board: {t.model}")
    if t.base not in models:
        raise HTTPException(422, f"{t.base} is not on the board — pick a model the board has "
                                 f"tested")
    if t.base == t.model:
        raise HTTPException(422, "a model is not trained from itself")
    # no loop: the base must not itself come from this model, at any remove
    seen, cur = set(), t.base
    while cur and cur not in seen:
        if cur == t.model:
            raise HTTPException(422, f"{t.base} was trained from {t.model}, so {t.model} "
                                     f"cannot be trained from it")
        seen.add(cur)
        cur = ((models.get(cur) or {}).get("trainedFrom") or {}).get("base")
    db.trained_from_set(t.model, t.base, who)
    _cache.update(key=None, payload=None, at=0.0)
    return {"model": t.model, "base": t.base, "by": who}


# ---- 12h.2: saved views of the Models table ---------------------------------
# A table someone built on Models ▸ Standard — its benchmarks and its models —
# named and kept for the whole team. The tailnet is the auth boundary, as for
# every other decision here: the owner is the name typed at the top of the
# page, and only that name renames or deletes the view.

_VIEW_KEY = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class ViewSpec(BaseModel):
    view: str = "standard"             # 12m.1: or "compare", its models side by side
    chip: str = "all"
    cols: list[str] | None = None      # the chosen benchmarks, or the chip's own
    models: list[str] | None = None    # the chosen models, or every one
    # 12o.1: each table's widths and order, as it was left: {table: {w, order, groups}}
    layout: dict[str, dict] | None = None


class ViewIn(BaseModel):
    name: str
    spec: ViewSpec
    by: str


class ViewRename(BaseModel):
    name: str
    by: str


class ViewBy(BaseModel):
    by: str


def _view_name(s: str) -> str:
    s = " ".join((s or "").split())[:60]
    if not s:
        raise HTTPException(422, "a view needs a name")
    return s


def _view_layout(layout: dict | None) -> dict | None:
    """a table's widths in pixels and its columns' order — names, never more"""
    if not layout:
        return None
    if len(layout) > 8:
        raise HTTPException(422, "a view keeps the layout of at most 8 tables")
    names = lambda xs, n, cap: isinstance(xs, list) and len(xs) <= n and all(  # noqa: E731
        isinstance(x, str) and len(x) <= cap for x in xs)
    out = {}
    for key, lay in layout.items():
        if not (isinstance(key, str) and 0 < len(key) <= 80 and isinstance(lay, dict)):
            raise HTTPException(422, "a layout is a table's name and its widths and order")
        w, order, groups = lay.get("w") or {}, lay.get("order") or [], lay.get("groups") or []
        if not (isinstance(w, dict) and len(w) <= 400 and all(
                isinstance(k, str) and len(k) <= 200 and isinstance(px, int)
                and not isinstance(px, bool) and 44 <= px <= 640 for k, px in w.items())):
            raise HTTPException(422, "column widths are 44 to 640 pixels")
        if not names(order, 400, 200) or not names(groups, 60, 120):
            raise HTTPException(422, "a column order is column names")
        # all empty: the table as it comes, which the view opens it as
        out[key] = {"w": w, "order": order, "groups": groups}
    return out or None


def _view_spec(v: ViewSpec) -> dict:
    spec = _view_spec_of(v)
    layout = _view_layout(v.layout)
    return {**spec, "layout": layout} if layout else spec


def _view_spec_of(v: ViewSpec) -> dict:
    cols = v.cols if v.cols else None
    models = v.models if v.models else None
    if v.view not in ("standard", "compare"):
        raise HTTPException(422, "a view is standard or compare")
    if v.view == "compare":
        # 12m.1: a comparison is its models, two to eight of them
        if not models or not 2 <= len(models) <= 8 or not all(0 < len(m) <= 200 for m in models):
            raise HTTPException(422, "a comparison is two to eight model ids")
        return {"view": "compare", "models": models}
    if cols is None and models is None:
        raise HTTPException(422, "nothing to save: choose benchmarks or models first")
    if cols and (len(cols) > 40 or not all(_VIEW_KEY.match(c) for c in cols)):
        raise HTTPException(422, "benchmarks are task names, at most 40")
    if models and (len(models) > 300 or not all(0 < len(m) <= 200 for m in models)):
        raise HTTPException(422, "models are model ids, at most 300")
    chip = v.chip if _VIEW_KEY.match(v.chip or "") else "all"
    return {"chip": chip, "cols": cols, "models": models}


def _own_view(vid: int, by: str, what: str) -> dict:
    view = db.view_get(vid)
    if not view:
        raise HTTPException(404, "no such view — someone may have deleted it")
    who = _name(by, what)
    if who.casefold() != view["saved_by"].casefold():
        raise HTTPException(403, f"only {view['saved_by']}, who saved this view, can {what}")
    return view


@app.get("/api/views")
def views_list():
    return {"views": db.views_list()}


@app.post("/api/views")
def view_save(v: ViewIn, x_token: str = Header(default="")):
    _check_token(x_token)
    who = _name(v.by, "saving a view")
    name = _view_name(v.name)
    spec = _view_spec(v.spec)
    clash = next((x for x in db.views_list() if x["name"].casefold() == name.casefold()), None)
    if clash:
        raise HTTPException(409, f"a view called {clash['name']} already exists, saved by "
                                 f"{clash['saved_by']} — choose another name")
    return db.view_get(db.view_add(name, spec, who))


@app.patch("/api/views/{vid}")
def view_rename(vid: int, v: ViewRename, x_token: str = Header(default="")):
    _check_token(x_token)
    _own_view(vid, v.by, "rename it")
    name = _view_name(v.name)
    clash = next((x for x in db.views_list() if x["id"] != vid
                  and x["name"].casefold() == name.casefold()), None)
    if clash:
        raise HTTPException(409, f"a view called {clash['name']} already exists, saved by "
                                 f"{clash['saved_by']} — choose another name")
    db.view_rename(vid, name)
    return db.view_get(vid)


@app.delete("/api/views/{vid}")
def view_delete(vid: int, v: ViewBy, x_token: str = Header(default="")):
    _check_token(x_token)
    view = _own_view(vid, v.by, "delete it")
    db.view_delete(vid)
    return {"deleted": vid, "name": view["name"]}


@app.get("/api/runs/{sid}/lines")
def run_lines(sid: int, tail: int = 200):
    """11g: the log for the Reader — numbered lines, up to 2,000, a line that
    quotes a hidden question withheld, and whether the run is still going."""
    try:
        return reader.log_lines(sid, tail)
    except KeyError:
        raise HTTPException(404, "no such submission") from None


@app.get("/api/judge/provenance", dependencies=EXAM_ONLY)
def judge_provenance(model: str):
    """11g: how a model was graded, for the Reader — its judge runs (never a
    run's plan) and the head of its judge.json (never an item or a qid)."""
    try:
        return reader.judge_provenance(model)
    except KeyError:
        raise HTTPException(404, f"no judge run recorded for {model}") from None


@app.get("/api/runs/{sid}/log", response_class=PlainTextResponse)
def run_log(sid: int, tail: int = 200):
    sub = db.get(sid)
    if not sub:
        raise HTTPException(404, "no such submission")
    path = config.LOGS_DIR / f"service_{sid}_{sub['hf_id'].replace('/', '__')}.log"
    if not path.exists():
        return "(no log yet — the run has not reached lm_eval)"
    lines = path.read_text(errors="replace").splitlines()
    # 12n.2: the raw text withholds what the Reader's lines do — a hidden exam
    # question, and any line quoting a GPQA question
    qids, texts = reader._hidden_marks()
    return "\n".join("[line withheld — it quotes a hidden question]"
                     if reader._withheld(line, qids, texts) else line
                     for line in lines[-min(tail, 2000):])


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


def build_id(page: str) -> str:
    """The build this process serves: the git short sha when it is known
    (EVALBOARD_BUILD, passed into the image at build time, or the checkout
    the service runs from), plus a hash of the page itself. The hash is what
    makes the check honest: an image built without the sha, or with a stale
    one, still tells an old page apart from a new one."""
    page_hash = hashlib.sha256(page.encode("utf-8")).hexdigest()[:7]
    sha = os.environ.get("EVALBOARD_BUILD", "").strip()[:12]
    if not sha:
        try:
            sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent),
                                  "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            sha = ""
    return f"{sha}+{page_hash}" if sha else page_hash


BUILD = build_id(_PAGE)
_PAGE = _PAGE.replace('<meta charset="utf-8">',
                      f'<meta charset="utf-8">\n<meta name="evalboard-build" content="{BUILD}">', 1)


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


# an address the client may be told it came from: a scheme, a host or an IP, a port
_BASE_OK = re.compile(r"https?://(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:]+\])(?::\d{1,5})?\Z")


@app.get("/client")
def client_file(request: Request):
    """The one-file training client, served from the service itself — friends on
    the tailnet grab it with `curl -O http://…:8899/client` and never need
    access to the git repo (which may be private). It knows where it came
    from: the address it was fetched at becomes its default board, so no
    address is ever written into the repo (the mirror is public)"""
    if not _CLIENT_PY.exists():
        raise HTTPException(404, "bench_client.py not found in this build")
    text = _CLIENT_PY.read_text(encoding="utf-8")
    base = str(request.base_url).rstrip("/")
    if _BASE_OK.match(base):
        text = text.replace('DEFAULT_BASE = ""', f'DEFAULT_BASE = "{base}"', 1)
    return PlainTextResponse(
        text,
        media_type="text/x-python",
        headers={"Content-Disposition": 'attachment; filename="bench_client.py"'})
