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

12f.5: one queue. A GGUF job's request waits in gguf_requests/held/ while a
board run queued before it hasn't finished, and a board run waits while a
GGUF job queued before it hasn't (db.claim_next): they take turns in the
order they were queued. A run that finds the worker holding the lock says
"waiting for GGUF run #92 (about 40 min left)" and waits it out. And Re-run
failed benchmarks queues only what a run didn't finish.

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
# 12f.4: a job the worker was on when it went quiet this long is released — as
# long as the run lock trusts a worker's heartbeat (runner.LOCK_BEAT_S)
STALE_S = 120
GONE = ("The GGUF worker stopped while this ran{ago}: start it again (HANDOFF § 5d) and queue "
        "this again.")
# 12f.5: a stop asked for while the worker had gone is done: nothing is running
GONE_CANCELED = "canceled; the GGUF worker had stopped{ago}"
NOT_STARTED = "canceled before it started"
LOST = "Its request to the GGUF worker is gone: queue it again."
OLD_DATASET = ("The {label} dataset is the old one, {what}: build it again with the converter "
               "(HANDOFF § 5d). Nothing was queued.")


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


# 12f.3 addendum: setups — "name: KEY=VALUE … --flag …", a line each
ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_MTP = re.compile(r"\bmtp\b|draft|--spec|speculative", re.I)


def parse_setups(text) -> list[dict]:
    """the setups a GGUF is measured in besides "as built": each a name, the
    environment variables and extra flags it passes to llama-perplexity"""
    if isinstance(text, list):
        return [s for s in text if s.get("id") != gb.AS_BUILT["id"]]
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if _MTP.search(line):
            raise ValueError(gb.MTP_LINE)
        name, colon, rest = line.partition(":")
        name = name.strip()
        if not colon or not name:
            raise ValueError(f'A setup is "name: KEY=VALUE --flag …", one a line: {line!r}')
        env, flags = {}, []
        for tok in shlex.split(rest):
            if flags or tok.startswith("-"):
                flags.append(tok)
                continue
            k, eq, v = tok.partition("=")
            if not eq or not ENV_KEY.match(k):
                raise ValueError(f"{name}: {tok!r} is neither KEY=VALUE nor a flag")
            if k in ("LD_LIBRARY_PATH", "PATH"):
                raise ValueError(f"{name}: {k} is the worker's own")
            env[k] = v
        if not env and not flags:
            raise ValueError(f"{name}: nothing set — as built is always measured")
        out.append({"id": gb.setup_id(env, flags), "name": name, "env": env, "flags": flags})
    ids = [s["id"] for s in out]
    if len(set(ids)) != len(ids):
        raise ValueError("two setups with the same settings")
    return out


def setups_text(setups: list[dict]) -> str:
    return "\n".join(f"{s['name']}: " + " ".join([f"{k}={v}" for k, v in s["env"].items()]
                                                  + s["flags"]) for s in setups or [])


def _check_path(path: str) -> str:
    path = (path or "").strip()
    if not path.startswith(("/", "~/")):
        raise ValueError("The GGUF file is an absolute path on the server, like "
                         "/home/masein/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf")
    return path


def view(rec: dict) -> dict:
    """what the page shows of a GGUF-only model"""
    return {"name": rec["name"], "path": rec["path"], "based_on": rec.get("based_on", ""),
            "how": rec["how"], "flags": rec["flags"], "pin": rec.get("pin") or {},
            "setups": [gb.AS_BUILT] + (rec.get("setups") or [])}


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
           "flags": flags_of(f.get("flags")), "setups": parse_setups(f.get("setups")),
           "by": by, "at": time.time()}
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
                "how": rec.get("how", ""), "pin": rec.get("gguf_pin") or {},
                "setups": rec.get("gguf_setups") or []}
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

def setups_of(model_id: str, want: list[str] | None = None) -> list[dict]:
    """as built, then the model's own setups — those asked for, in that order"""
    m = model(model_id) or {}
    have = [gb.AS_BUILT] + list(m.get("setups") or [])
    if not want:
        return have
    unknown = [w for w in want if w not in {s["id"] for s in have}]
    if unknown:
        raise ValueError(f"no setup {', '.join(unknown)} on {model_id}")
    return [s for s in have if s["id"] in want]


def _seconds_each(model_id: str) -> dict[str, float]:
    """seconds a task, by benchmark, from the newest run of this file that
    measured it"""
    sha = ((model(model_id) or {}).get("pin") or {}).get("sha256")
    measured: dict[str, float] = {}
    for res in results():
        if sha and (res.get("file") or {}).get("sha256") != sha:
            continue
        for b, v in (res.get("benchmarks") or {}).items():
            if v.get("seconds") is not None and v.get("done"):
                measured.setdefault(b, v["seconds"] / v["done"])
    return measured


def estimate(model_id: str, benchmarks: list[str], subset: int = 0,
             setups: list[str] | None = None) -> dict:
    """seconds for each benchmark, times the setups chosen: from the newest run
    of this file that measured it, else a rough guess"""
    k = len(setups_of(model_id, setups)) if model(model_id) else 1
    man = manifest()
    measured = _seconds_each(model_id)
    out, rough = {}, False
    for b in benchmarks:
        n = subset or int((man.get(b) or {}).get("n") or gb.BENCHMARKS[b]["n"])
        each = measured.get(b)
        if each is None:
            rough = True
            each = GUESS_S[gb.BENCHMARKS[b]["mode"]]
        out[b] = {"n": n, "seconds": round(n * each * k)}
    total = sum(v["seconds"] for v in out.values())
    return {"by": out, "seconds": total, "rough": rough, "line": _dur(total) + (
        ", a rough guess" if rough else "")}


def _dur(s: float) -> str:
    return f"about {s / 3600:.1f} h" if s >= 5400 else f"about {max(1, round(s / 60))} min"


def time_left(sid: int) -> float | None:
    """12f.5: seconds GGUF job `sid` has left: the running benchmark's at the
    pace it has had, then the ones not started at the pace of this file's
    last run (or the rough guess); None when there is no such job"""
    row = db.get(sid) if sid else None
    if not row or row["suite"] != "gguf":
        return None
    res = _read(root() / "gguf_results" / f"{sid}.json") or {}
    try:
        asked = json.loads(row.get("tasks") or "[]")
    except ValueError:
        asked = []
    benches = res.get("benchmarks") or {b: {"status": "queued"} for b in asked}
    measured, man = _seconds_each(row["hf_id"]), manifest()
    subset, left = int(row.get("subset") or 0), 0.0
    for b, v in benches.items():
        if b not in gb.BENCHMARKS or v.get("status") not in ("queued", "running"):
            continue
        n = subset or int((man.get(b) or {}).get("n") or gb.BENCHMARKS[b]["n"])
        each = measured.get(b) or GUESS_S[gb.BENCHMARKS[b]["mode"]]
        if v.get("status") == "running":
            n = max(0, int(v.get("total") or n) - int(v.get("done") or 0))
            each = v.get("secs_each") or each
        left += n * each
    return left


def waiting_line(sid: int) -> str:
    """12f.5: what a board run waiting on GGUF job `sid` says"""
    if not sid:
        return "waiting for a GGUF run"
    left = time_left(sid)
    return f"waiting for GGUF run #{sid}" + (f" ({_dur(left)} left)" if left else "")


def queue(model_id: str, benchmarks: list[str], subset: int, by: str,
          time_limit_h: float = 24.0, setups: list[str] | None = None) -> list[int]:
    """a run for each setup chosen (every one when none is)"""
    return [_queue_one(model_id, benchmarks, subset, by, time_limit_h, s)
            for s in setups_of(model_id, setups)]


def _queue_one(model_id: str, benchmarks: list[str], subset: int, by: str,
               time_limit_h: float, setup: dict, note: str = "") -> int:
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
    # 12f.5: hours measuring MMLU's cloze file would land in History, not the column
    for b in want:
        form = gb.BENCHMARKS[b].get("format")
        if form and man[b].get("format") != form:
            raise ValueError(OLD_DATASET.format(label=gb.BENCHMARKS[b]["label"],
                                                what="each option's text scored (cloze)"))
    subset = int(subset or 0)
    if subset < 0:
        raise ValueError("a subset is a number of tasks; 0 is every one")
    sid = db.add(model_id, "instruct", "gguf", by, f"setup: {setup['name']}{note}", tasks=want,
                 subset=subset)
    req = {"id": str(sid), "sid": sid, "model": model_id, "name": m["name"], "path": m["path"],
           "flags": m["flags"], "setup": setup, "benchmarks": want, "subset": subset,
           "pin": m.get("pin") or {},
           "datasets": {b: {"sha256": man[b]["sha256"], "n": man[b]["n"]} for b in want},
           "time_limit_s": time_limit_h * 3600, "by": by, "at": time.time()}
    _dir("gguf_results")
    # 12f.5: its turn comes after the board runs queued before it
    ahead = db.board_ahead(sid)
    _write(_dir("gguf_requests/held" if ahead else "gguf_requests") / f"{sid}.json", req)
    w = worker()
    db.update(sid, progress=_turn_line(ahead) if ahead else "waiting for the GGUF worker"
              + ("" if w["alive"] else f" · {DOWN}"),
              arch=json.dumps({"gguf": {"path": m["path"], "flags": m["flags"], "setup": setup,
                                        "subset": subset, "benchmarks": want}}))
    return sid


def _turn_line(ahead: int) -> str:
    return f"waiting for run #{ahead}, queued before it"


def _held(sid: int) -> Path:
    return root() / "gguf_requests" / "held" / f"{sid}.json"


def cancel(sid: int) -> None:
    # 12f.5: a request still waiting its turn never reaches the worker
    try:
        _held(sid).replace(_dir("gguf_requests/done") / f"{sid}.json")
        return
    except OSError:
        pass
    (_dir("gguf_requests") / f"{sid}.cancel").write_text("canceled")


def not_done(sid: int) -> list[str]:
    """12f.5: the benchmarks a finished GGUF run has no score for — failed,
    stopped or not run — in the board's order"""
    res = _read(root() / "gguf_results" / f"{sid}.json")
    if res:
        got = res.get("benchmarks") or {}
        return [b for b in gb.ORDER if b in got and got[b].get("status") != "done"]
    row = db.get(sid) or {}
    try:
        return [b for b in gb.ORDER if b in json.loads(row.get("tasks") or "[]")]
    except ValueError:
        return []


def rerun_failed(sid: int, by: str) -> dict:
    """12f.5: Re-run failed benchmarks — a new run of the same file, setup
    and subset, of only the benchmarks run `sid` didn't finish; the finished
    ones are kept"""
    row = db.get(sid)
    if not row or row["suite"] != "gguf":
        raise LookupError(f"no GGUF run #{sid}")
    if row["status"] in ACTIVE + ("canceling",):
        raise ValueError(f"Run #{sid} hasn't finished yet.")
    left = not_done(sid)
    if not left:
        raise ValueError(f"Every benchmark of run #{sid} finished: nothing to re-run.")
    res = _read(root() / "gguf_results" / f"{sid}.json") or {}
    try:
        was = (json.loads(row.get("arch") or "{}").get("gguf") or {}).get("setup")
    except ValueError:
        was = None
    was = res.get("setup") or was or gb.AS_BUILT
    setup = next((x for x in setups_of(row["hf_id"]) if x["id"] == was.get("id")), None)
    if setup is None:
        raise ValueError(f"Run #{sid}'s setup, {was.get('name')}, is no longer registered with "
                         "those settings: measure it again from Measure on the GGUF.")
    new = _queue_one(row["hf_id"], left, int(res.get("subset") or row.get("subset") or 0), by,
                     24.0, setup, note=f" · what #{sid} didn't finish")
    return {"id": new, "benchmarks": left}


def results() -> list[dict]:
    """every result file, the newest first"""
    d = root() / "gguf_results"
    out = [r for r in (_read(p) for p in d.glob("*.json")) if r] if d.is_dir() else []
    return sorted(out, key=lambda r: -(r.get("finished_at") or r.get("started_at") or 0))


def _left_behind(sid: int) -> bool:
    """12f.5: a worker runs one job at a time, so a job whose result says
    "running" while the worker says it's on another was left by one that
    stopped. Read after the result, then the result again: a job that ended
    while the worker moved on to the next has its end written by then"""
    w = worker()
    on = w["state"]
    if not w["alive"] or not on.startswith("running ") or on == f"running {sid}":
        return False
    return (_read(root() / "gguf_results" / f"{sid}.json") or {}).get("status") == "running"


def _lost(row: dict) -> bool:
    """12f.5: a job with no request for the worker, held or not, and no
    result, a while after it was queued: the worker writes a job's result
    before it moves its request away, so nothing will take it up"""
    sid = row["id"]
    return (time.time() - (row.get("created_at") or 0) > STALE_S and not _held(sid).exists()
            and not (root() / "gguf_requests" / f"{sid}.json").exists()
            and not (root() / "gguf_results" / f"{sid}.json").exists())


def sync() -> None:
    """the worker's result files, into their run rows"""
    w = worker()
    for row in sorted(db.recent(200), key=lambda r: r["id"]):
        if row["suite"] != "gguf" or row["status"] not in ACTIVE + ("canceling",):
            continue
        res = _read(root() / "gguf_results" / f"{row['id']}.json")
        # 12f.5: a stop asked for before the worker took it up is done now:
        # its request is canceled, so the worker skips it
        if row["status"] == "canceling" and (not res or res.get("status") == "queued"):
            cancel(row["id"])
            db.update(row["id"], status="canceled", finished_at=time.time(),
                      progress=NOT_STARTED)
            continue
        # 12f.4: a job never stays "running" after its worker is gone: it fails,
        # saying so, and its request is canceled so a restarted worker skips it
        quiet = time.time() - (w["at"] or 0)
        if res and res.get("status") in ("running", "waiting") and quiet > STALE_S:
            ago = f" (last seen {max(1, round(quiet / 60))} min ago)" if w["at"] else ""
            cancel(row["id"])
            if row["status"] == "canceling":            # 12f.5: stopped, as asked
                db.update(row["id"], status="canceled", finished_at=time.time(),
                          progress=GONE_CANCELED.format(ago=ago))
                continue
            line = GONE.format(ago=ago)
            db.update(row["id"], status="failed", finished_at=time.time(), progress=line,
                      error=line)
            continue
        # 12f.5: a job left "running" by a worker that stopped (#89), or with
        # nothing for the worker to take up, never ends — and holds up the queue
        if (res and res.get("status") == "running" and _left_behind(row["id"])) or \
                (not res and _lost(row)):
            line = gb.RESTARTED if res else LOST
            if res:
                cancel(row["id"])                 # a restarted worker doesn't take it up
            db.update(row["id"], status="failed", finished_at=time.time(), progress=line,
                      error=line)
            continue
        # 12f.5: a request waiting its turn goes to the worker once the board
        # runs queued before it have finished
        if _held(row["id"]).exists():
            ahead = db.board_ahead(row["id"])
            if ahead:
                db.update(row["id"], progress=_turn_line(ahead))
                continue
            try:
                _held(row["id"]).replace(_dir("gguf_requests") / f"{row['id']}.json")
            except OSError:
                pass
        if not res or res.get("status") == "queued":
            db.update(row["id"], progress="waiting for the GGUF worker"
                      + ("" if w["alive"] else f" · {DOWN}"))
            continue
        st, line = res.get("status"), res.get("line") or ""
        arch = {"gguf": {"file": res.get("file"), "build": res.get("build"),
                         "flags": res.get("flags"), "subset": res.get("subset"),
                         "setup": res.get("setup"),
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
