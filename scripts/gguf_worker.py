#!/usr/bin/env python3
"""12f.3: the GGUF worker — measures MMLU, HellaSwag, Winogrande, ARC and
TruthfulQA straight from a GGUF file with the fork's llama-perplexity.

It runs on the host, outside Docker, because llama-perplexity is built there
(~/llama.cpp-teraformer/build-lda) against ~/lda-env's CUDA runtime, which the
board's container can't run. The standard library only. masein starts it with
nohup, as he does llama-server (HANDOFF § 5d):

    nohup python3 scripts/gguf_worker.py --results /home/masein/benchmarks/results \\
      --binary ~/llama.cpp-teraformer/build-lda/bin/llama-perplexity \\
      --ld-library-path ~/lda-env/lib:~/llama.cpp-teraformer/build-lda/bin \\
      > ~/gguf-worker.log 2>&1 &

The results folder is bind-mounted, so the host and the board see the same
files:
- results/gguf_requests/<id>.json — what the board queued; <id>.cancel stops it
- results/gguf_results/<id>.json — progress, then the result, as it goes
- results/gguf_worker.json — the heartbeat, every few seconds
- results/.run.lock — the run lock every run takes (mkdir-atomic, a pid file);
  it waits while a board run holds it. Its own lock carries a heartbeat file,
  because the board in its container can't see a host process's pid

For each request: the lock; the model file checked (its sha256 hashed once,
cached by size and mtime) and each dataset's against what the board queued —
a changed one stops the job in one line; each benchmark run and read as it
prints; the lock released. Ctrl+C, a cancel file or the time limit stop the
running benchmark and keep the finished ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_bench as gb  # noqa: E402

POLL_S = 5.0
LOCK_BEAT_S = 120            # a lock whose heartbeat is older is a dead worker's
_stop = {"why": ""}


def now() -> float:
    return time.time()


def write_json(p: Path, data: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(p)


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Worker:
    def __init__(self, results: Path, binary: str, ld_path: str = "", poll: float = POLL_S,
                 time_limit: float = 24 * 3600):
        self.results = Path(results)
        self.binary = os.path.expanduser(binary)
        self.ld_path = ":".join(os.path.expanduser(p) for p in ld_path.split(":") if p)
        self.poll = poll
        self.time_limit = time_limit
        self.requests = self.results / "gguf_requests"
        self.out = self.results / "gguf_results"
        self.data = self.results / "gguf_data"
        self.lock = self.results / ".run.lock"
        self.beat = self.results / "gguf_worker.json"
        self.hashes = self.results / "gguf_hashes.json"
        for d in (self.requests, self.out, self.requests / "done"):
            d.mkdir(parents=True, exist_ok=True)
        self.build = self._build()
        self.state = "idle"

    # -- the heartbeat, the build ------------------------------------------------
    def _build(self) -> str:
        """what the binary says it is: "version: 1234 (91428471f)" """
        try:
            p = subprocess.run([self.binary, "--version"], capture_output=True, text=True,
                               timeout=60, env=self.env())
            line = next((x for x in (p.stderr + p.stdout).splitlines()
                         if x.startswith("version:")), "")
            return line.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def env(self, extra: dict | None = None) -> dict:
        e = dict(os.environ)
        # 12f.3 addendum: the setup's own variables (lookahead routing, say)
        e.update({str(k): str(v) for k, v in (extra or {}).items()})
        if self.ld_path:
            e["LD_LIBRARY_PATH"] = self.ld_path + (
                ":" + e["LD_LIBRARY_PATH"] if e.get("LD_LIBRARY_PATH") else "")
        return e

    def heartbeat(self) -> None:
        write_json(self.beat, {"at": now(), "pid": os.getpid(), "host": socket.gethostname(),
                               "binary": self.binary, "build": self.build, "state": self.state})
        if (self.lock / "owner").exists():
            try:
                (self.lock / "heartbeat").write_text(str(now()))
            except OSError:
                pass

    # -- the run lock ------------------------------------------------------------
    def take_lock(self, sid: int) -> str:
        """'' when taken, else who holds it"""
        try:
            self.lock.mkdir(parents=True)
        except FileExistsError:
            beat = self.lock / "heartbeat"
            if beat.exists() and now() - beat.stat().st_mtime > LOCK_BEAT_S:
                # a worker that died holding it: take it over
                self._rm_lock()
                return self.take_lock(sid)
            held = ""
            try:
                held = (self.lock / "submission").read_text().strip()
            except OSError:
                pass
            return f"run #{held}" if held and held != "0" else "a run from the command line"
        (self.lock / "pid").write_text(str(os.getpid()))
        (self.lock / "submission").write_text(str(sid))
        (self.lock / "owner").write_text(f"gguf-worker {socket.gethostname()} {os.getpid()}")
        (self.lock / "heartbeat").write_text(str(now()))
        return ""

    def _rm_lock(self) -> None:
        for f in ("pid", "submission", "owner", "heartbeat"):
            try:
                (self.lock / f).unlink()
            except OSError:
                pass
        try:
            self.lock.rmdir()
        except OSError:
            pass

    def release_lock(self) -> None:
        if (self.lock / "owner").exists():
            self._rm_lock()

    # -- hashes --------------------------------------------------------------------
    def sha256(self, p: Path) -> str:
        """hashed once, cached by size and mtime: a 23 GB file takes minutes"""
        st = p.stat()
        cache = read_json(self.hashes) or {}
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key in cache:
            return cache[key]
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
        write_json(self.hashes, cache)
        return cache[key]

    # -- a request -------------------------------------------------------------------
    def next_request(self) -> Path | None:
        reqs = sorted(self.requests.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return reqs[0] if reqs else None

    def once(self) -> bool:
        """take one waiting request, if the lock is free: whether one ran"""
        self.heartbeat()
        p = self.next_request()
        if p is None:
            return False
        req = read_json(p)
        if not req:
            p.rename(self.requests / "done" / p.name)
            return False
        sid = int(req.get("sid") or 0)
        res_path = self.out / f"{req['id']}.json"
        res = read_json(res_path) or self._result(req)
        if (self.requests / f"{req['id']}.cancel").exists():
            res.update(status="stopped", line="canceled before it started", finished_at=now())
            write_json(res_path, res)
            self._done(p, req)
            return True
        held = self.take_lock(sid)
        if held:
            res.update(status="waiting", line=f"waiting for the run lock: {held} holds it")
            write_json(res_path, res)
            return False
        try:
            self.state = f"running {req['id']}"
            self.run(req, res, res_path)
        finally:
            self.release_lock()
            self.state = "idle"
            self.heartbeat()
            self._done(p, req)
        return True

    def _done(self, p: Path, req: dict) -> None:
        try:
            p.rename(self.requests / "done" / p.name)
        except OSError:
            pass
        try:
            (self.requests / f"{req['id']}.cancel").unlink()
        except OSError:
            pass

    def _result(self, req: dict) -> dict:
        return {"id": req["id"], "sid": req.get("sid"), "model": req["model"],
                "name": req.get("name", ""), "status": "queued", "line": "",
                "flags": req.get("flags") or [], "subset": int(req.get("subset") or 0),
                # 12f.3 addendum: the setup, pinned with what it measured
                "setup": req.get("setup") or gb.AS_BUILT,
                "benchmarks": {b: {"status": "queued"} for b in req["benchmarks"]},
                "build": self.build, "binary": self.binary, "host": socket.gethostname(),
                "queued_at": req.get("at")}

    def run(self, req: dict, res: dict, res_path: Path) -> None:
        t0 = now()
        res.update(status="running", started_at=t0, line="checking the files")
        write_json(res_path, res)
        model = Path(os.path.expanduser(req["path"]))
        if not model.is_file():
            res.update(status="failed", finished_at=now(),
                       line=f"The GGUF file isn't on this server: {model}")
            write_json(res_path, res)
            return
        sha = self.sha256(model)
        res["file"] = {"path": str(model), "name": model.name, "size": model.stat().st_size,
                       "sha256": sha}
        pin = req.get("pin") or {}
        if pin.get("sha256") and pin["sha256"] != sha:
            res.update(status="failed", finished_at=now(),
                       line="The GGUF file changed since it was registered: its sha256 is "
                            f"{sha[:12]}, not {pin['sha256'][:12]}. Register it again if that's "
                            "intended.")
            write_json(res_path, res)
            return
        res["datasets"] = {}
        for b in req["benchmarks"]:
            f = self.data / gb.BENCHMARKS[b]["data"]
            if not f.is_file():
                res.update(status="failed", finished_at=now(),
                           line=f"The {gb.BENCHMARKS[b]['label']} dataset isn't in "
                                f"{self.data}: run the converter first (HANDOFF § 5d).")
                write_json(res_path, res)
                return
            got = self.sha256(f)
            want = (req.get("datasets") or {}).get(b, {}).get("sha256")
            res["datasets"][b] = {"file": f.name, "sha256": got}
            if want and want != got:
                res.update(status="failed", finished_at=now(),
                           line=f"The {gb.BENCHMARKS[b]['label']} dataset changed since this "
                                f"was queued: its sha256 is {got[:12]}, not {want[:12]}. Queue it "
                                "again to measure on the new one.")
                write_json(res_path, res)
                return
        limit = float(req.get("time_limit_s") or self.time_limit)
        for b in req["benchmarks"]:
            why = self.stop_why(req, t0, limit)
            if why:
                res["benchmarks"][b] = {"status": "not run"}
                continue
            self.bench(req, res, res_path, b, model, t0, limit)
        stopped = [b for b, v in res["benchmarks"].items() if v.get("status") in ("stopped",
                                                                                "not run")]
        failed = [b for b, v in res["benchmarks"].items() if v.get("status") == "failed"]
        done = [b for b, v in res["benchmarks"].items() if v.get("status") == "done"]
        why = self.stop_why(req, t0, limit)
        if stopped:
            res.update(status="stopped", line=f"{why or 'stopped'}: {len(done)} of "
                       f"{len(res['benchmarks'])} benchmarks finished are kept")
        elif failed:
            res.update(status="failed", line="; ".join(
                f"{gb.BENCHMARKS[b]['label']}: {res['benchmarks'][b].get('error')}" for b in failed))
        else:
            res.update(status="done", line=f"{len(done)} benchmarks measured")
        res["finished_at"] = now()
        write_json(res_path, res)

    def stop_why(self, req: dict, t0: float, limit: float) -> str:
        if _stop["why"]:
            return _stop["why"]
        if (self.requests / f"{req['id']}.cancel").exists():
            return "canceled"
        if now() - t0 > limit:
            return f"the time limit ({limit / 3600:g} h) was reached"
        return ""

    def bench(self, req: dict, res: dict, res_path: Path, b: str, model: Path, t0: float,
              limit: float) -> None:
        info = gb.BENCHMARKS[b]
        data = self.data / info["data"]
        n = int(req.get("subset") or 0) or int(((req.get("datasets") or {}).get(b) or {})
                                                .get("n") or info["n"])
        setup = req.get("setup") or gb.AS_BUILT
        cmd = gb.command(b, self.binary, str(model),
                         (req.get("flags") or gb.DEFAULT_FLAGS) + list(setup.get("flags") or []),
                         str(data), n)
        log = self.out / f"{req['id']}.{b}.log"
        cur = {"status": "running", "command": cmd, "env": setup.get("env") or {},
               "log": log.name, "started_at": now(), "done": 0, "total": n}
        res["benchmarks"][b] = cur
        res["line"] = f"{info['label']}: starting"
        write_json(res_path, res)
        mode = info["mode"]
        with open(log, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=self.env(setup.get("env")), start_new_session=True)
            last = 0.0
            while True:
                try:
                    code = proc.wait(timeout=min(self.poll, 2.0))
                    break
                except subprocess.TimeoutExpired:
                    pass
                why = self.stop_why(req, t0, limit)
                if why:
                    self._end(proc)
                    cur.update(status="stopped", why=why)
                    code = None
                    break
                if now() - last >= self.poll:
                    last = now()
                    self._progress(cur, res, res_path, b, log, mode)
                    self.heartbeat()
        got = gb.parse(mode, log.read_text(encoding="utf-8", errors="replace"))
        cur.update(done=got["done"], total=got["total"] or n, finished_at=now(),
                   seconds=round(now() - cur["started_at"], 3))
        if cur["status"] == "stopped":
            cur["partial"] = {"done": got["done"], "acc": got["acc"]}
        elif code == 0 and got["final"] and got["acc"] is not None:
            cur.update(status="done", acc=got["acc"], se=got["se"], chance=got["chance"],
                       n=got["done"], full=int(req.get("subset") or 0) == 0)
        else:
            cur.update(status="failed", error=got["error"] or f"llama-perplexity exited {code} "
                       "without a score — see its log")
        res["line"] = f"{info['label']}: {cur['status']}"
        write_json(res_path, res)

    def _progress(self, cur, res, res_path, b, log, mode) -> None:
        got = gb.parse(mode, log.read_text(encoding="utf-8", errors="replace"))
        cur.update(done=got["done"], total=got["total"] or cur["total"], acc_so_far=got["acc"])
        el = now() - cur["started_at"]
        if got["done"]:
            each = el / got["done"]
            left = max(0, (cur["total"] or 0) - got["done"]) * each
            cur["secs_each"] = round(each, 4)
            res["line"] = (f"{gb.BENCHMARKS[b]['label']}: {got['done']} of {cur['total']} · "
                           f"about {max(1, round(left / 60))} min left")
        write_json(res_path, res)

    @staticmethod
    def _end(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()

    def loop(self) -> None:
        while not _stop["why"]:
            ran = self.once()
            if not ran:
                time.sleep(self.poll)
        self.heartbeat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, help="the results folder the board mounts")
    ap.add_argument("--binary", required=True, help="llama-perplexity, as built on this host")
    ap.add_argument("--ld-library-path", default="", help="its CUDA runtime and libraries")
    ap.add_argument("--poll", type=float, default=POLL_S)
    ap.add_argument("--time-limit-h", type=float, default=24.0,
                    help="a job stops after this, keeping the benchmarks it finished")
    ap.add_argument("--once", action="store_true", help="one request, then exit")
    a = ap.parse_args(argv)

    def stop(sig, frame):          # noqa: ARG001 — Ctrl+C: the running one stops, finished kept
        _stop["why"] = "stopped by Ctrl+C" if sig == signal.SIGINT else "stopped"
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    w = Worker(Path(os.path.expanduser(a.results)), a.binary, a.ld_library_path, a.poll,
               a.time_limit_h * 3600)
    print(f"gguf worker: {w.build or 'llama-perplexity (no version)'} · watching {w.requests}",
          flush=True)
    if a.once:
        w.once()
    else:
        w.loop()
    w.release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
