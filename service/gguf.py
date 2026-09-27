"""12f.3: GGUF files measured with llama.cpp's llama-perplexity, by the worker
on the host (scripts/gguf_worker.py).

The board can't run the fork's llama-perplexity in its container, so it:
- **registers** a GGUF: a served model's "GGUF file on the server", or a file
  with no server (Test a model ▸ A GGUF file) — a host path, what it's based
  on, how it's built, and llama-perplexity's flags for it;
- **queues** a job as a run row (suite "gguf") and a request file in
  results/gguf_requests/, which the worker picks up under the same run lock
  every run takes;
- **reads back** the worker's result file (results/gguf_results/<id>.json)
  into the row: its progress, its end, and the file's hash, pinned on the model
  after the first job so a changed file stops the next;
- says "The GGUF worker isn't running" when its heartbeat
  (results/gguf_worker.json) is older than a minute.

The results never go where lm_eval's do: the page reads them from
gguf_results/ into a column group of their own, never averaged with lm_eval's.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

from . import config, db, served

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gguf_bench as gb  # noqa: E402

PREFIX = "gguf/"
HEARTBEAT_S = 60
# seconds a task, before a run of this file has measured it: a rough guess
GUESS_S = {"multiple-choice": 0.25, "hellaswag": 0.4, "winogrande": 0.1}
ACTIVE = ("queued", "waiting_lock", "running")
DOWN = "The GGUF worker isn't running."


def root() -> Path:
    return config.RESULTS_ROOT


def _dir(name: str) -> Path:
    """a folder the host's worker writes in too: open to it"""
    d = root() / name
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o777)
    except OSError:
        pass
    return d


def _write(p: Path, data: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(p)
    try:
        os.chmod(p, 0o666)
    except OSError:
        pass


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def worker_command() -> str:
    """HANDOFF § 5d's start command, with this board's results folder"""
    return ("cd ~/benchmarks/aienh && nohup python3 scripts/gguf_worker.py "
            f"--results {root()} --binary ~/llama.cpp-teraformer/build-lda/bin/llama-perplexity "
            "--ld-library-path ~/lda-env/lib:~/llama.cpp-teraformer/build-lda/bin "
            "> ~/gguf-worker.log 2>&1 &")


def worker() -> dict:
    """{alive, at, build, state, line, command}"""
    hb = _read(root() / "gguf_worker.json") or {}
    at = float(hb.get("at") or 0)
    alive = bool(at) and time.time() - at < HEARTBEAT_S
    return {"alive": alive, "at": at or None, "build": hb.get("build") or "",
            "state": hb.get("state") or "", "host": hb.get("host") or "",
            "line": "" if alive else DOWN, "command": worker_command()}


def manifest() -> dict:
    """what gguf_data.py wrote: each benchmark's file, sha256 and count"""
    return (_read(root() / "gguf_data" / "manifest.json") or {}).get("benchmarks") or {}


# ---------------------------------------------------------------------------
# the registry: a served model's GGUF, or a file with no server
# ---------------------------------------------------------------------------

def slug(name: str) -> str:
    return PREFIX + re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")[:80]


def flags_of(text) -> list[str]:
    if isinstance(text, list):
        return [str(x) for x in text]
    text = (text or "").strip()
    return shlex.split(text) if text else list(gb.DEFAULT_FLAGS)


def _check_path(path: str) -> str:
    path = (path or "").strip()
    if not path.startswith(("/", "~/")):
        raise ValueError("The GGUF file is an absolute path on the server, like "
                         "/home/masein/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf")
    return path


def view(rec: dict) -> dict:
    """what the page shows of a GGUF-only model"""
    return {"name": rec["name"], "path": rec["path"], "based_on": rec.get("based_on", ""),
            "how": rec["how"], "flags": rec["flags"], "pin": rec.get("pin") or {}}


def write_meta(rec: dict) -> None:
    d = config.OUT_DIR / rec["id"].replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps(
        {"model": rec["id"], "kind": "instruct", "params": None,
         "kind_reason": "a GGUF file, measured by llama-perplexity", "gguf": view(rec)}),
        encoding="utf-8")


def register(f: dict, by: str) -> dict:
    """Test a model ▸ A GGUF file: kept, the path checked by the worker when
    its first job starts (the board can't see the host's files)"""
    name, how = (f.get("name") or "").strip(), (f.get("how") or "").strip()
    if not slug(name)[len(PREFIX):]:
        raise ValueError("A name: it is shown everywhere")
    if not how:
        raise ValueError("How it's built: the llama.cpp build, the quantisation, what changed. "
                         "It is the record of what was measured")
    rec = {"id": slug(name), "name": name, "path": _check_path(f.get("path")),
           "based_on": (f.get("based_on") or "").strip(), "how": how,
           "flags": flags_of(f.get("flags")), "by": by, "at": time.time()}
    old = db.gguf_get(rec["id"]) or {}
    if old.get("path") == rec["path"] and old.get("pin"):
        rec["pin"] = old["pin"]
    db.gguf_put(rec)
    write_meta(rec)
    return rec


def model(model_id: str) -> dict | None:
    """a GGUF to measure — {id, name, path, flags, based_on, how, pin} — from
    the GGUF registry, or a served model's GGUF file"""
    if model_id.startswith(PREFIX):
        return db.gguf_get(model_id)
    rec = served.get(model_id)
    if rec and rec.get("gguf_path"):
        return {"id": rec["id"], "name": rec["name"], "path": rec["gguf_path"],
                "flags": flags_of(rec.get("gguf_flags")), "based_on": rec.get("based_on", ""),
                "how": rec.get("how", ""), "pin": rec.get("gguf_pin") or {}}
    return None


def _pin(model_id: str, file: dict) -> None:
    """what the first job hashed, kept: the next job checks it"""
    pin = {"sha256": file.get("sha256"), "size": file.get("size"), "name": file.get("name")}
    if model_id.startswith(PREFIX):
        rec = db.gguf_get(model_id)
        if rec and not rec.get("pin"):
            rec["pin"] = pin
            db.gguf_put(rec)
            write_meta(rec)
    else:
        rec = served.get(model_id)
        if rec and not rec.get("gguf_pin"):
            rec["gguf_pin"] = pin
            db.served_put(rec)
            served.write_meta(rec)


# ---------------------------------------------------------------------------
# a job
# ---------------------------------------------------------------------------

def estimate(model_id: str, benchmarks: list[str], subset: int = 0) -> dict:
    """seconds for each benchmark: from the newest run of this file that
    measured it, else a rough guess"""
    m = model(model_id) or {}
    sha = (m.get("pin") or {}).get("sha256")
    man = manifest()
    measured: dict[str, float] = {}
    for res in results():
        if sha and (res.get("file") or {}).get("sha256") != sha:
            continue
        for b, v in (res.get("benchmarks") or {}).items():
            if v.get("seconds") is not None and v.get("done"):
                measured.setdefault(b, v["seconds"] / v["done"])
    out, rough = {}, False
    for b in benchmarks:
        n = subset or int((man.get(b) or {}).get("n") or gb.BENCHMARKS[b]["n"])
        each = measured.get(b)
        if each is None:
            rough = True
            each = GUESS_S[gb.BENCHMARKS[b]["mode"]]
        out[b] = {"n": n, "seconds": round(n * each)}
    total = sum(v["seconds"] for v in out.values())
    return {"by": out, "seconds": total, "rough": rough, "line": _dur(total) + (
        ", a rough guess" if rough else "")}


def _dur(s: float) -> str:
    return f"about {s / 3600:.1f} h" if s >= 5400 else f"about {max(1, round(s / 60))} min"


def queue(model_id: str, benchmarks: list[str], subset: int, by: str,
          time_limit_h: float = 24.0) -> int:
    m = model(model_id)
    if not m:
        raise ValueError(f"{model_id} has no GGUF file registered")
    want = [b for b in gb.ORDER if b in set(benchmarks or gb.ORDER)]
    unknown = [b for b in benchmarks or [] if b not in gb.BENCHMARKS]
    if unknown or not want:
        raise ValueError("benchmarks are " + ", ".join(gb.ORDER))
    man = manifest()
    missing = [gb.BENCHMARKS[b]["label"] for b in want if b not in man]
    if missing:
        raise ValueError(f"No dataset yet for {', '.join(missing)}: run the converter once "
                         "(HANDOFF § 5d)")
    subset = int(subset or 0)
    if subset < 0:
        raise ValueError("a subset is a number of tasks; 0 is every one")
    sid = db.add(model_id, "instruct", "gguf", by, "", tasks=want, subset=subset)
    req = {"id": str(sid), "sid": sid, "model": model_id, "name": m["name"], "path": m["path"],
           "flags": m["flags"], "benchmarks": want, "subset": subset,
           "pin": m.get("pin") or {},
           "datasets": {b: {"sha256": man[b]["sha256"], "n": man[b]["n"]} for b in want},
           "time_limit_s": time_limit_h * 3600, "by": by, "at": time.time()}
    _dir("gguf_results")
    _write(_dir("gguf_requests") / f"{sid}.json", req)
    w = worker()
    db.update(sid, progress="waiting for the GGUF worker" + ("" if w["alive"] else f" · {DOWN}"),
              arch=json.dumps({"gguf": {"path": m["path"], "flags": m["flags"],
                                        "subset": subset, "benchmarks": want}}))
    return sid


def cancel(sid: int) -> None:
    (_dir("gguf_requests") / f"{sid}.cancel").write_text("canceled")


def results() -> list[dict]:
    """every result file, the newest first"""
    d = root() / "gguf_results"
    out = [r for r in (_read(p) for p in d.glob("*.json")) if r] if d.is_dir() else []
    return sorted(out, key=lambda r: -(r.get("finished_at") or r.get("started_at") or 0))


def sync() -> None:
    """the worker's result files, into their run rows"""
    w = worker()
    for row in db.recent(200):
        if row["suite"] != "gguf" or row["status"] not in ACTIVE + ("canceling",):
            continue
        res = _read(root() / "gguf_results" / f"{row['id']}.json")
        if not res or res.get("status") == "queued":
            db.update(row["id"], progress="waiting for the GGUF worker"
                      + ("" if w["alive"] else f" · {DOWN}"))
            continue
        st, line = res.get("status"), res.get("line") or ""
        arch = {"gguf": {"file": res.get("file"), "build": res.get("build"),
                         "flags": res.get("flags"), "subset": res.get("subset"),
                         "datasets": res.get("datasets")}}
        if st == "waiting":
            db.update(row["id"], status="waiting_lock", progress=line)
        elif st == "running":
            db.update(row["id"], status="running", progress=line,
                      started_at=res.get("started_at"), arch=json.dumps(arch))
        else:
            if res.get("file", {}).get("sha256") and st in ("done", "stopped"):
                _pin(row["hf_id"], res["file"])
            scores = " · ".join(f"{gb.BENCHMARKS[b]['label']} {100 * v['acc']:.1f}"
                                for b, v in (res.get("benchmarks") or {}).items()
                                if v.get("status") == "done")
            db.update(row["id"], status="done" if st == "done" else
                      "canceled" if line.startswith("canceled") else "failed",
                      finished_at=res.get("finished_at") or time.time(),
                      progress=scores or line, error="" if st == "done" else line,
                      arch=json.dumps(arch))
