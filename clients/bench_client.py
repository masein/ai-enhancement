#!/usr/bin/env python3
"""Zero-dependency client for the benchmark service — stdlib only, so you can
vendor this single file into any training repo without touching its environment.

Library:

    from bench_client import Bench
    bench = Bench("http://teraformer-5090-3:8899")          # token="..." if the server wants one

    sid = bench.submit("myorg/run7-step4000", suite="quick",
                       submitter="omar", note="step 4000")   # returns immediately
    info = bench.wait(sid)                                    # blocks until done/failed
    print(bench.scores("myorg/run7-step4000"))
    # {'hellaswag': {'value': 0.412, 'stderr': 0.005, 'metric': 'acc_norm', 'shots': 5}, ...}

CLI (the same four verbs):

    python bench_client.py --base http://teraformer-5090-3:8899 submit myorg/model --suite quick --wait
    python bench_client.py --base ... queue
    python bench_client.py --base ... scores myorg/model
    python bench_client.py --base ... cancel 7

The full API contract lives in API.md; the pattern for calling this from a
training loop (submit at each checkpoint, collect at the end) is in there too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


class BenchError(RuntimeError):
    pass


class Bench:
    def __init__(self, base: str, token: str = "", timeout: float = 30.0):
        self.base = base.rstrip("/")
        self.token = token
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------------
    def _call(self, path: str, payload: dict | None = None, method: str | None = None):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json", "X-Token": self.token},
            method=method or ("POST" if payload is not None else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode()).get("detail", "")
            except Exception:
                detail = ""
            raise BenchError(f"{e.code} on {path}: {detail or e.reason}") from None
        except urllib.error.URLError as e:
            raise BenchError(f"cannot reach {self.base}: {e.reason}") from None

    # -- the API ----------------------------------------------------------------
    def submit(self, hf_id: str, suite: str = "full", kind: str = "auto",
               submitter: str = "", note: str = "") -> int:
        """Queue a model. Returns the submission id. Submitting a model that is
        already queued/running joins the existing run; one that is already fully
        benchmarked completes in seconds (per-task resume)."""
        r = self._call("/api/submissions", {"hf_id": hf_id, "suite": suite, "kind": kind,
                                            "submitter": submitter, "note": note})
        return int(r["id"])

    def status(self, sid: int) -> dict:
        for row in self._call("/api/submissions?limit=500"):
            if row["id"] == sid:
                return row
        raise BenchError(f"submission #{sid} not found")

    def queue(self) -> list[dict]:
        return self._call("/api/submissions?limit=100")

    def cancel(self, sid: int) -> dict:
        return self._call(f"/api/submissions/{sid}/cancel", method="POST")

    def wait(self, sid: int, poll_s: float = 30.0, timeout_s: float = 12 * 3600,
             echo: bool = False) -> dict:
        """Block until the submission reaches done/failed/canceled; returns the row.
        Raises BenchError on failure so `wait()` in a script fails loudly."""
        t0, last = time.time(), ""
        while True:
            row = self.status(sid)
            line = f"{row['status']}: {row.get('progress') or ''}"
            if echo and line != last:
                print(f"  #{sid} {line}", file=sys.stderr)
                last = line
            if row["status"] in ("done", "failed", "canceled"):
                if row["status"] != "done":
                    raise BenchError(f"#{sid} {row['status']}: {row.get('error') or ''}")
                return row
            if time.time() - t0 > timeout_s:
                raise BenchError(f"#{sid} still {row['status']} after {timeout_s:.0f}s")
            time.sleep(poll_s)

    def results(self) -> dict:
        """The full dashboard payload — see API.md § payload schema."""
        return self._call("/api/results")

    def scores(self, hf_id: str) -> dict:
        """Headline metric per task for one model:
        {task: {value, stderr, metric, shots, lower_is_better}}"""
        d = self.results()
        out = {}
        for task, models in d["cells"].items():
            if hf_id in models:
                c = models[hf_id]
                info = d["tasks"].get(task, {})
                out[task] = {"value": c["v"], "stderr": c.get("se"),
                             "metric": info.get("metric"), "shots": c.get("shots"),
                             "lower_is_better": bool(info.get("lower"))}
        return out


# -- CLI ------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="e.g. http://teraformer-5090-3:8899")
    ap.add_argument("--token", default="", help="only if the server sets SUBMIT_TOKEN")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("hf_id")
    s.add_argument("--suite", default="full", choices=["quick", "full"])
    s.add_argument("--kind", default="auto", choices=["auto", "base", "instruct"])
    s.add_argument("--submitter", default=""); s.add_argument("--note", default="")
    s.add_argument("--wait", action="store_true")
    sub.add_parser("queue")
    sc = sub.add_parser("scores"); sc.add_argument("hf_id")
    c = sub.add_parser("cancel"); c.add_argument("sid", type=int)
    a = ap.parse_args()

    b = Bench(a.base, a.token)
    try:
        if a.cmd == "submit":
            sid = b.submit(a.hf_id, a.suite, a.kind, a.submitter, a.note)
            print(f"#{sid} queued")
            if a.wait:
                b.wait(sid, echo=True)
                print(json.dumps(b.scores(a.hf_id), indent=1))
        elif a.cmd == "queue":
            for r in b.queue():
                print(f"#{r['id']:<4} {r['status']:<12} {r['hf_id']:<44} "
                      f"{r.get('progress') or ''}{(' | ' + r['error']) if r.get('error') else ''}")
        elif a.cmd == "scores":
            print(json.dumps(b.scores(a.hf_id), indent=1))
        elif a.cmd == "cancel":
            print(b.cancel(a.sid))
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
