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
        raise HTTPException(422, "model id must look like org/name — a Hugging Face repo "
                                 "id, or local/<name> for an uploaded artifact")
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


# ---------------------------------------------------------------------------
# training-run tracking — the wandb-shaped API (see API.md § run tracking)
# ---------------------------------------------------------------------------

class TrunIn(BaseModel):
    name: str
    project: str = "default"
    submitter: str = ""
    config: dict = {}
    hf_prefix: str = ""


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
    rid = db.trun_create(name, t.project.strip()[:80] or "default",
                         t.submitter.strip()[:80],
                         json.dumps(t.config)[:20000], t.hf_prefix.strip()[:200])
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


@app.get("/api/truns")
def trun_index(project: str | None = None, limit: int = 200):
    return db.trun_list(project, min(limit, 500))


@app.get("/api/truns/{rid}")
def trun_detail(rid: int, max_points: int = 800):
    run = db.trun_get(rid)
    if not run:
        raise HTTPException(404, "no such run")
    return {"run": run, **db.trun_series(rid, max(50, min(max_points, 5000)))}


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
    if dest.exists():
        raise HTTPException(409, f"artifact {name!r} already exists — checkpoints are "
                                 f"immutable; use a new name per checkpoint")
    cap = int(config.ARTIFACT_MAX_GB * 1e9)
    quota = int(config.ARTIFACT_QUOTA_GB * 1e9)
    used = _dir_bytes(config.ARTIFACTS_DIR)
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
