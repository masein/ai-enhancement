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

    # -- run tracking (the wandb-shaped half) -------------------------------------
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


    def init(self, name: str, project: str = "default", config: dict | None = None,
             submitter: str = "", hf_prefix: str = "") -> "Run":
        """Start a tracked training run. Use as a context manager:

            with bench.init("run7", config={"lr": 3e-4}) as run:
                run.log({"loss": loss, "lr": lr}, step=step)
                run.log_checkpoint(step, "local/run7-step200")   # marks + submits

        finish() is called on exit (status "failed" if an exception escaped)."""
        r = self._call("/api/truns", {"name": name, "project": project,
                                      "config": config or {}, "submitter": submitter,
                                      "hf_prefix": hf_prefix})
        return Run(self, int(r["id"]), name, submitter)

    def upload_artifact(self, name: str, checkpoint_dir) -> str:
        """Zip a save_pretrained() directory and upload it as artifact `name`.
        Returns the model id to submit: "local/<name>". No HF account involved."""
        import os
        import tempfile
        import zipfile
        from pathlib import Path as _P
        d = _P(checkpoint_dir)
        if not (d / "config.json").exists():
            raise BenchError(f"{d} does not look like a checkpoint (no config.json)")
        # pre-check the name: artifact names are immutable, and a duplicate makes
        # the server refuse before reading the body — which a mid-upload client
        # experiences as a bare connection reset instead of the real reason
        try:
            existing = {a["name"] for a in self._call("/api/artifacts")["artifacts"]}
            if name in existing:
                raise BenchError(
                    f"artifact {name!r} already exists (names are immutable — an "
                    f"earlier run probably used the same run-name). Pick a fresh "
                    f"name, or delete the old one: DELETE /api/artifacts/{name}")
        except BenchError as e:
            if "already exists" in str(e):
                raise
            # listing failed (old server?) — proceed; the upload itself will say
        # zip to a temp file and stream it — checkpoints are hundreds of MB and
        # do not belong in RAM twice
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
            tmp = tf.name
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(d))
            size = os.path.getsize(tmp)
            with open(tmp, "rb") as fh:
                req = urllib.request.Request(
                    f"{self.base}/api/artifacts/{name}", data=fh,
                    headers={"Content-Type": "application/zip", "X-Token": self.token,
                             "Content-Length": str(size)}, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=max(self.timeout, 1800)) as r:
                        return json.loads(r.read().decode())["model_id"]
                except urllib.error.HTTPError as e:
                    try:
                        detail = json.loads(e.read().decode()).get("detail", "")
                    except Exception:
                        detail = ""
                    raise BenchError(f"{e.code} uploading {name}: "
                                     f"{detail or e.reason}") from None
                except (urllib.error.URLError, OSError) as e:
                    # a mid-send reset usually means the server refused early
                    # (duplicate name, size cap, quota) — surface it as ours, so
                    # callers' BenchError handling keeps training alive
                    raise BenchError(
                        f"connection dropped while uploading {name}: {e}. If this "
                        f"repeats, check GET /api/artifacts (name taken? quota?) "
                        f"and the service logs.") from None
        finally:
            os.unlink(tmp)


class Run:
    """A live training run: buffered metric logging that NEVER raises into the
    training loop, checkpoint markers that also queue the benchmark, finish()."""

    FLUSH_EVERY = 64          # points
    FLUSH_SECS = 10.0

    def __init__(self, bench: Bench, rid: int, name: str, submitter: str = ""):
        self.bench, self.id, self.name = bench, rid, name
        self.submitter = submitter
        self._buf: list[dict] = []
        self._last_flush = time.time()
        self._warned = False

    # context manager: finish cleanly, mark failed if an exception escaped
    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.finish("failed" if exc_type else "finished")
        return False

    def log(self, metrics: dict, step: int) -> None:
        for k, v in metrics.items():
            try:
                self._buf.append({"step": int(step), "name": str(k), "value": float(v)})
            except (TypeError, ValueError):
                continue
        if len(self._buf) >= self.FLUSH_EVERY or \
           time.time() - self._last_flush > self.FLUSH_SECS:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        batch, self._buf = self._buf, []
        self._last_flush = time.time()
        try:
            self.bench._call(f"/api/truns/{self.id}/log", {"metrics": batch})
            self._warned = False
        except BenchError as e:
            if not self._warned:      # complain once, then stay quiet — never kill training
                print(f"[bench] metric logging failing (non-fatal): {e}", file=sys.stderr)
                self._warned = True

    def log_checkpoint(self, step: int, model_id: str, submit: bool = True,
                       suite: str = "quick", note: str = "") -> int | None:
        """Mark a checkpoint at `step` and (by default) queue it for evaluation.
        `model_id` is a Hub repo or a local/<name> artifact id."""
        self.flush()
        try:
            self.bench._call(f"/api/truns/{self.id}/event",
                             {"step": int(step), "kind": "checkpoint", "detail": model_id})
        except BenchError as e:
            print(f"[bench] checkpoint marker failed (non-fatal): {e}", file=sys.stderr)
        if not submit:
            return None
        try:
            return self.bench.submit(model_id, suite=suite, submitter=self.submitter,
                                     note=note or f"{self.name} @ step {step}")
        except BenchError as e:
            print(f"[bench] checkpoint submit failed (non-fatal): {e}", file=sys.stderr)
            return None

    def finish(self, status: str = "finished") -> None:
        self.flush()
        try:
            self.bench._call(f"/api/truns/{self.id}/finish", {"status": status})
        except BenchError:
            pass

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
