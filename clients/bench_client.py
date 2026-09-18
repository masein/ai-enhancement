#!/usr/bin/env python3
"""Zero-dependency client for the benchmark service — stdlib only, so you can
vendor this single file into any training repo without touching its environment.

Library — benchmark a model straight from your disk (no Hugging Face needed):

    from bench_client import Bench
    bench = Bench("http://100.74.89.105:8899")          # token="..." if the server wants one

    mid = bench.upload_artifact("run7-step4000", "ckpt_dir/")   # -> "local/run7-step4000"
    sid = bench.submit(mid, suite="quick", submitter="masein")  # returns immediately
    info = bench.wait(sid)                                      # blocks until done/failed
    print(bench.scores(mid))
    # {'hellaswag': {'value': 0.412, 'stderr': 0.005, 'metric': 'acc_norm', 'shots': 5}, ...}

    (Hub models work the same — bench.submit("myorg/model", ...).)

CLI (same verbs from a shell):

    python bench_client.py --base http://100.74.89.105:8899 upload run7-step4000 ckpt_dir/ --submit --wait
    python bench_client.py --base ... submit myorg/model --suite quick --wait
    python bench_client.py --base ... artifacts        # what's in storage, vs quota
    python bench_client.py --base ... queue
    python bench_client.py --base ... scores local/run7-step4000
    python bench_client.py --base ... delete run7-step4000
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
               submitter: str = "", note: str = "",
               allow_remote_code: bool = False) -> int:
        """Queue a model. Returns the submission id. Submitting a model that is
        already queued/running joins the existing run; one that is already fully
        benchmarked completes in seconds (per-task resume).

        allow_remote_code: for an uploaded artifact whose config.json has an
        auto_map, i.e. a custom architecture whose modeling code ships with the
        checkpoint. Loading it executes that code, so it is opt-in per
        submission and the server has to be configured for it — see API.md
        § custom model code."""
        r = self._call("/api/submissions", {"hf_id": hf_id, "suite": suite, "kind": kind,
                                            "submitter": submitter, "note": note,
                                            "allow_remote_code": allow_remote_code})
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
             submitter: str = "", hf_prefix: str = "", datasets: list[int] | None = None,
             parent: str = "") -> "Run":
        """Start a tracked training run. Use as a context manager:

            with bench.init("run7", config={"lr": 3e-4}) as run:
                run.log({"loss": loss, "lr": lr}, step=step)
                run.log_checkpoint(step, "local/run7-step200")   # marks + submits

        finish() is called on exit (status "failed" if an exception escaped).

        datasets: ids of generated datasets (bench.datasets()) this run trains
        on. Recording them is what marks every checkpoint of the run as trained
        on benchmark-derived data — the task it came from leaves the official
        average and the leaderboard says so. Leave it out and the board has no
        way to know; put it in and the number stays honest.

        parent: the model id this run started from (the base checkpoint). With
        it, the board can show what the training taught: both halves of the
        tainted task before and after, side by side. Defaults to
        config["base_model"] when that is a model id on the board."""
        r = self._call("/api/truns", {"name": name, "project": project,
                                      "config": config or {}, "submitter": submitter,
                                      "hf_prefix": hf_prefix, "datasets": list(datasets or []),
                                      "parent": parent})
        return Run(self, int(r["id"]), name, submitter)

    # -- generated datasets (the find-the-gap pipeline; see API.md § datasets) ----
    def datasets(self) -> list[dict]:
        """Every generated dataset with its status and provenance summary."""
        return self._call("/api/datasets")

    def dataset(self, did: int) -> dict:
        """One dataset's record, provenance in full."""
        return self._call(f"/api/datasets/{did}")

    def pull_dataset(self, did: int, dest) -> "tuple[str, list[dict]]":
        """Download items.jsonl and provenance.json into `dest`; returns
        (path to items.jsonl, the items). Refuses a dataset that is not ready."""
        import os
        from pathlib import Path as _P
        d = _P(dest)
        d.mkdir(parents=True, exist_ok=True)
        rec = self.dataset(did)
        if rec.get("status") != "ready":
            raise BenchError(f"dataset {did} is {rec.get('status')}: {rec.get('error') or ''}")
        req = urllib.request.Request(f"{self.base}/api/datasets/{did}/items.jsonl",
                                     headers={"X-Token": self.token})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raise BenchError(f"{e.code} downloading dataset {did}") from None
        items_path = d / "items.jsonl"
        items_path.write_bytes(body)
        (d / "provenance.json").write_text(json.dumps(rec.get("provenance") or {}, indent=2))
        items = [json.loads(x) for x in body.decode("utf-8").splitlines() if x.strip()]
        return os.fspath(items_path), items

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

    def artifacts(self) -> dict:
        """What the storage holds: {"artifacts": [{"name", "model_id", "bytes",
        "created"}, ...], "total_bytes": ..., "quota_bytes": ...}."""
        return self._call("/api/artifacts")

    def delete_artifact(self, name: str) -> dict:
        """Free an artifact's disk. Its scores stay on the leaderboard; refused
        (409) while that artifact is queued or being evaluated."""
        return self._call(f"/api/artifacts/{name}", method="DELETE")


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
    ap.add_argument("--base", required=True, help="e.g. http://100.74.89.105:8899")
    ap.add_argument("--token", default="", help="only if the server sets SUBMIT_TOKEN")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="queue a model — a Hub id or local/<name>")
    s.add_argument("hf_id")
    s.add_argument("--suite", default="full", choices=["quick", "full", "control", "judged"])
    s.add_argument("--kind", default="auto", choices=["auto", "base", "instruct"])
    s.add_argument("--submitter", default=""); s.add_argument("--note", default="")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--allow-remote-code", action="store_true",
                   help="uploaded artifact with a custom architecture: execute the "
                        "modeling code that ships with it (server must be configured "
                        "for it)")
    u = sub.add_parser("upload",
                       help="upload a save_pretrained() dir to the service's storage "
                            "(no Hugging Face), optionally submit it in one go")
    u.add_argument("name", help="artifact name, e.g. run7-step4000 — immutable, one per checkpoint")
    u.add_argument("checkpoint_dir", help="directory with config.json + *.safetensors")
    u.add_argument("--submit", action="store_true", help="also queue the benchmark")
    u.add_argument("--suite", default="quick", choices=["quick", "full", "control", "judged"])
    u.add_argument("--kind", default="auto", choices=["auto", "base", "instruct"])
    u.add_argument("--submitter", default=""); u.add_argument("--note", default="")
    u.add_argument("--wait", action="store_true", help="implies --submit; block until scored")
    u.add_argument("--allow-remote-code", action="store_true",
                   help="custom architecture: execute the modeling code in the upload")
    sub.add_parser("artifacts", help="list uploaded checkpoints and quota use")
    dl = sub.add_parser("delete", help="free an artifact's disk (scores stay)")
    dl.add_argument("name")
    sub.add_parser("queue")
    sc = sub.add_parser("scores"); sc.add_argument("hf_id")
    c = sub.add_parser("cancel"); c.add_argument("sid", type=int)
    sub.add_parser("datasets", help="generated datasets: id, task, category, status, items")
    pl = sub.add_parser("pull", help="download a generated dataset (items.jsonl + provenance.json)")
    pl.add_argument("dataset_id", type=int); pl.add_argument("dest")
    a = ap.parse_args()

    b = Bench(a.base, a.token)
    try:
        if a.cmd == "submit":
            sid = b.submit(a.hf_id, a.suite, a.kind, a.submitter, a.note,
                           allow_remote_code=a.allow_remote_code)
            print(f"#{sid} queued")
            if a.wait:
                b.wait(sid, echo=True)
                print(json.dumps(b.scores(a.hf_id), indent=1))
        elif a.cmd == "upload":
            mid = b.upload_artifact(a.name, a.checkpoint_dir)
            print(f"uploaded -> {mid}")
            if a.submit or a.wait:
                sid = b.submit(mid, a.suite, a.kind, a.submitter, a.note,
                               allow_remote_code=a.allow_remote_code)
                print(f"#{sid} queued ({a.suite})")
                if a.wait:
                    b.wait(sid, echo=True)
                    print(json.dumps(b.scores(mid), indent=1))
            else:
                print(f"benchmark it with: submit {mid} --suite quick "
                      f"(or paste {mid} into the dashboard)")
        elif a.cmd == "artifacts":
            info = b.artifacts()
            for art in info.get("artifacts", []):
                print(f"{art['model_id']:<48} {art['bytes'] / 1e9:7.2f} GB")
            print(f"{'total':<48} {info.get('total_bytes', 0) / 1e9:7.2f} GB "
                  f"of {info.get('quota_bytes', 0) / 1e9:.0f} GB quota")
        elif a.cmd == "delete":
            r = b.delete_artifact(a.name)
            print(f"deleted {r.get('deleted', a.name)} — {r.get('note', 'disk freed')}")
        elif a.cmd == "queue":
            for r in b.queue():
                print(f"#{r['id']:<4} {r['status']:<12} {r['hf_id']:<44} "
                      f"{r.get('progress') or ''}{(' | ' + r['error']) if r.get('error') else ''}")
        elif a.cmd == "scores":
            print(json.dumps(b.scores(a.hf_id), indent=1))
        elif a.cmd == "cancel":
            print(b.cancel(a.sid))
        elif a.cmd == "datasets":
            for d in b.datasets():
                pv = d.get("provenance") or {}
                kept = (pv.get("items") or {}).get("kept")
                print(f"#{d['id']:<4} {d['status']:<9} {d.get('task') or '—':<12} "
                      f"{d.get('category') or '—':<24} {d['fmt']:<5} "
                      f"{('%d items' % kept) if kept is not None else ''}"
                      f"{(' | ' + d['error']) if d.get('error') else ''}")
        elif a.cmd == "pull":
            path, items = b.pull_dataset(a.dataset_id, a.dest)
            print(f"wrote {len(items)} items to {path} (+ provenance.json)")
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
