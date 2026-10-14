"""21c: the reports' stats — the thinking tokens and loop examples of a
scorecard (service/reports.py), the lists of a ran-out report
(service/ran_out.py) — worked out in a process of their own and kept on disk.

21 worked them out in a thread of the server's own: reading every stored
answer (gigabytes of them) inside the server held up every other request,
/healthz included. Now:
- the work runs in a separate process (`python -m service.report_stats`), at
  the lowest CPU and disk priority, one at a time (a lock file), started
  after each import or scoring, and by a request that finds something not
  worked out yet;
- what it works out is kept in BENCH_ROOT/reports/stats.json, a slot a stat,
  each with the files it read (their identity) and when it was worked out;
- a request only reads what is kept, with its time: a stat whose files have
  changed since is shown until its new one is in, and one never worked out
  says so ("working these out, back in a minute") — a request never computes.

Nothing here starts a run, an import or a grading batch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, NamedTuple

from . import config

REPO = Path(__file__).resolve().parent.parent
SPAWN_EVERY = 10.0              # a worker that hasn't taken its lock yet is starting
ROUNDS = 5                      # files changed while it worked: looked at again
LOG_MAX = 1_000_000


class Job(NamedTuple):
    """one stat: its slot in the store, the identity of the files it reads,
    its words for the log, and how it is worked out"""
    slot: str
    key: str
    what: str
    fn: Callable[[], object]


def folder() -> Path:
    return Path(config.BENCH_ROOT) / "reports"


def _file() -> Path:
    return folder() / "stats.json"


def identity(f: Path | None) -> str:
    """a file as it is now: its path, when it last changed and its size"""
    if f is None:
        return "-"
    try:
        st = f.stat()
    except OSError:
        return "-"
    return f"{f}|{st.st_mtime_ns}|{st.st_size}"


# ---------------------------------------------------------------------------
# the store: read whole, kept in memory until the file changes; written under
# a lock, a slot at a time
# ---------------------------------------------------------------------------

_mem: dict = {"stamp": None, "data": {}}
_mem_lock = threading.Lock()


def store() -> dict:
    f = _file()
    try:
        st = f.stat()
        stamp = (str(f), st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    with _mem_lock:
        if stamp != _mem["stamp"]:
            data: object = {}
            if stamp:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    data = {}
            _mem.update(stamp=stamp, data=data if isinstance(data, dict) else {})
        return _mem["data"]


def kept(slot: str) -> dict | None:
    """{key, at, value} — or {key, at, error} — as last worked out; None if never"""
    e = store().get(slot)
    return e if isinstance(e, dict) else None


def keep(slot: str, key: str, value: object = None, error: str = "") -> dict:
    d = folder()
    d.mkdir(parents=True, exist_ok=True)
    entry = {"key": key, "at": time.time(), **({"error": error} if error else {"value": value})}
    with open(d / "stats.write.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            f = _file()
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            data[slot] = entry
            tmp = f.with_name(f"stats.{os.getpid()}.{threading.get_ident()}.part")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(f)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    with _mem_lock:
        _mem["stamp"] = None
    return entry


# ---------------------------------------------------------------------------
# what is wanted, and what of it is stale
# ---------------------------------------------------------------------------

def jobs() -> list[Job]:
    """every report's stats, each once"""
    from . import ran_out, reports
    out: dict[str, Job] = {}
    for rid, rep in reports.files().items():
        got = ran_out.jobs(rep) if rep.get("kind") == "ran_out" else reports.jobs(rid, rep)
        for j in got:
            out.setdefault(j.slot, j)
    return list(out.values())


def stale(js: list[Job] | None = None) -> list[Job]:
    """the stats never worked out, or worked out from files changed since"""
    data = store()
    return [j for j in (jobs() if js is None else js)
            if (data.get(j.slot) or {}).get("key") != j.key]


def running() -> bool:
    """a worker holds the lock"""
    p = folder() / "stats.lock"
    if not p.exists():
        return False
    try:
        with open(p, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def progress() -> dict:
    """{todo, done, since} while a worker runs"""
    if not running():
        return {}
    try:
        p = json.loads((folder() / "stats-progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        p = {}
    return {"todo": int(p.get("todo") or 0), "done": int(p.get("done") or 0),
            "since": p.get("since")}


_spawned: dict = {"at": 0.0, "root": None}


def wake(js: list[Job] | None = None) -> bool:
    """start a worker when something is stale and none runs. True while one
    runs (or was just started and is taking its lock)"""
    if running():
        return True
    if not stale(js):
        return False
    if time.time() - _spawned["at"] < SPAWN_EVERY and _spawned["root"] == str(config.BENCH_ROOT):
        return True
    if not config.REPORT_STATS_SPAWN:
        return False
    _spawned.update(at=time.time(), root=str(config.BENCH_ROOT))
    spawn()
    return True


def wake_quietly() -> None:
    """after an import or a scoring: never in its way"""
    try:
        wake()
    except Exception as e:                                  # noqa: BLE001 — said, not raised
        print(f"[report-stats] couldn't start working out the reports' stats: {e!r}")


def spawn() -> subprocess.Popen:
    from . import reports
    d = folder()
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "service.report_stats", "--bench-root", str(config.BENCH_ROOT),
           "--out-dir", str(config.OUT_DIR), "--reports-dir", str(reports.REPORTS_DIR)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        x for x in (str(REPO), str(REPO / "scripts"), os.environ.get("PYTHONPATH", "")) if x)}
    log = d / "stats.log"
    try:
        if log.stat().st_size > LOG_MAX:
            log.replace(log.with_suffix(".log.1"))
    except OSError:
        pass
    with open(log, "ab") as out:
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    # reaped when it ends: waiting holds no lock
    threading.Thread(target=p.wait, name="report-stats-reap", daemon=True).start()
    return p


def working(js: list[Job]) -> dict:
    """for a report's page: {} when its stats are all worked out; else how far
    the worker is ({todo, done}), and `first` when none of them is kept yet"""
    st = stale(js)
    if not st:
        return {}
    data = store()
    first = not any("value" in (data.get(j.slot) or {}) for j in js)
    on = wake(js)
    p = progress() if on else {}
    return {"todo": p.get("todo") or len(st), "done": p.get("done") or 0, "first": first,
            "running": on}


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------

def _progress(**kw) -> None:
    try:
        (folder() / "stats-progress.json").write_text(json.dumps({**kw, "pid": os.getpid()}),
                                                      encoding="utf-8")
    except OSError:
        pass


def work(again: bool = False, say=print) -> dict:
    """work out every stale stat (every stat with `again`), a job at a time,
    each kept as soon as it is out; until nothing is stale"""
    d = folder()
    d.mkdir(parents=True, exist_ok=True)
    lk = open(d / "stats.lock", "a")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lk.close()
        say("[report-stats] another worker is working these out")
        return {"busy": True, "done": 0, "failed": 0, "seconds": 0.0}
    t0 = time.time()
    done = failed = 0
    try:
        # 21's cache, never read again
        try:
            (d / "token-stats.json").unlink()
        except OSError:
            pass
        for rnd in range(ROUNDS):
            js = jobs()
            todo = js if (again and rnd == 0) else stale(js)
            if not todo:
                break
            _progress(todo=len(todo), done=0, since=t0)
            for k, j in enumerate(todo):
                t = time.time()
                try:
                    keep(j.slot, j.key, j.fn())
                    say(f"[report-stats] {j.what}: {time.time() - t:.1f} s")
                except Exception as e:                      # noqa: BLE001 — kept, and said
                    keep(j.slot, j.key, error=f"{type(e).__name__}: {e}"[:300])
                    say(f"[report-stats] {j.what}: couldn't be worked out: {type(e).__name__}: {e}")
                    failed += 1
                done += 1
                _progress(todo=len(todo), done=k + 1, since=t0)
    finally:
        try:
            (d / "stats-progress.json").unlink()
        except OSError:
            pass
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()
    secs = time.time() - t0
    say(f"[report-stats] {done} worked out in {secs:.1f} s, {failed} couldn't be")
    return {"done": done, "failed": failed, "seconds": round(secs, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m service.report_stats",
        description="Work out the reports' stats (thinking tokens, loop examples, ran-out lists) "
                    "and keep them, at the lowest priority. The board starts this itself; run it "
                    "by hand with --again to time it.")
    ap.add_argument("--bench-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--reports-dir")
    ap.add_argument("--again", action="store_true", help="work every stat out again")
    a = ap.parse_args(argv)
    from . import reported, reports
    if a.bench_root:
        config.BENCH_ROOT = Path(a.bench_root)
    if a.out_dir:
        config.OUT_DIR = Path(a.out_dir)
    if a.reports_dir:
        reports.REPORTS_DIR = reported.REPORTS_DIR = Path(a.reports_dir)
    try:
        os.nice(19)
    except (OSError, AttributeError):
        pass
    # the idle disk priority, where the system gives it — the work goes on
    # without it where it doesn't
    if shutil.which("ionice"):
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], capture_output=True,
                       check=False)
    print(f"[report-stats] working, pid {os.getpid()}", flush=True)
    got = work(again=a.again, say=lambda s: print(s, flush=True))
    return 1 if got.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
