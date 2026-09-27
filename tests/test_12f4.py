"""12f.4, from the first night of served and GGUF runs: a missing package is
named in the run's line; the GGUF worker refuses to start without a
llama-perplexity that runs, saying why in one line; a crash ends its job, with
what happened, and the worker goes on; and a job the worker was on when it went
away is released on the board, not left "running · MMLU: starting"."""

from __future__ import annotations

import json
import time

import gguf_worker as gw
from service import config, gguf, runner
from test_gguf_12f3 import svc  # noqa: F401


def test_a_missing_package_is_named():
    tail = ("Traceback (most recent call last):\n  File \"api_models.py\", line 9\n"
            "ModuleNotFoundError: No module named 'tiktoken'")
    assert "(missing package: tiktoken)" in runner.classify(tail)
    assert "(missing package: aiohttp)" in runner.classify(
        "ModuleNotFoundError: No module named 'aiohttp.client'")
    # an ImportError that names no module still says what kind of failure it is
    assert "(missing package)" in runner.classify("ImportError: cannot import name 'x' from 'y'")


def test_the_worker_refuses_to_start_without_its_binary(tmp_path, capsys):
    res = tmp_path / "results"
    code = gw.main(["--results", str(res), "--binary", "/nowhere/llama-perplexity", "--once"])
    assert code == 2
    assert capsys.readouterr().out.strip() == (
        "gguf worker: not started. llama-perplexity isn't at /nowhere/llama-perplexity: build it "
        "(HANDOFF § 5d), or give its path with --binary.")
    # no heartbeat: the board keeps saying it isn't running, with the command
    assert not (res / "gguf_worker.json").exists()


def test_one_that_cant_run_or_doesnt_start_is_refused_too(tmp_path):
    flat = tmp_path / "llama-perplexity"
    flat.write_text("#!/bin/sh\necho hi\n")
    flat.chmod(0o644)
    assert gw.check_binary(str(flat)).endswith(
        "can't be run: make it executable, or give llama-perplexity's path with --binary.")
    broken = tmp_path / "broken"
    broken.write_text("#!/bin/sh\necho 'error while loading shared libraries: libcudart.so.12' >&2\n"
                      "exit 127\n")
    broken.chmod(0o755)
    assert gw.check_binary(str(broken)) == (
        f"{broken} doesn't start: error while loading shared libraries: libcudart.so.12")


def queue(client, box):
    gid = client.post("/api/gguf/models", json={"name": "Qwen3.6 original", "path": str(box["model"]),
                                                "based_on": "fx/good-750m", "how": "unsloth",
                                                "by": "masein"}).json()["model"]["id"]
    return client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mmlu"],
                                               "by": "masein"}).json()["id"]


def test_a_crash_ends_its_job_saying_so_and_the_worker_goes_on(svc, monkeypatch):  # noqa: F811
    client, _, box = svc
    first, second = queue(client, box), queue(client, box)
    w = gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05)
    real = gw.Worker.bench

    def boom(self, *a, **k):
        raise RuntimeError("the model file went away")
    monkeypatch.setattr(gw.Worker, "bench", boom)
    assert w.once() is True
    res = json.loads((config.RESULTS_ROOT / "gguf_results" / f"{first}.json").read_text())
    assert res["status"] == "failed"
    assert res["line"] == "The worker failed on this job: RuntimeError: the model file went away"
    assert not (config.RESULTS_ROOT / ".run.lock").exists()          # the lock is let go
    row = next(x for x in client.get("/api/submissions").json() if x["id"] == first)
    assert row["status"] == "failed" and "went away" in row["error"]
    # the next job runs
    monkeypatch.setattr(gw.Worker, "bench", real)
    assert w.once() is True
    row = next(x for x in client.get("/api/submissions").json() if x["id"] == second)
    assert row["status"] == "done"


def test_a_job_left_running_by_a_gone_worker_is_released(svc):  # noqa: F811
    client, _, box = svc
    sid = queue(client, box)
    out = config.RESULTS_ROOT / "gguf_results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sid}.json").write_text(json.dumps({"id": sid, "status": "running",
                                                 "line": "MMLU: starting", "benchmarks": {}}))
    beat = config.RESULTS_ROOT / "gguf_worker.json"
    # a worker that beat a moment ago is still on it
    beat.write_text(json.dumps({"at": time.time() - 30}))
    row = next(x for x in client.get("/api/submissions").json() if x["id"] == sid)
    assert row["status"] == "running"
    # five minutes quiet: it's gone, and its job is let go, saying so
    beat.write_text(json.dumps({"at": time.time() - 300}))
    row = next(x for x in client.get("/api/submissions").json() if x["id"] == sid)
    assert row["status"] == "failed"
    assert row["error"] == gguf.GONE.format(ago=" (last seen 5 min ago)")
    # and a worker started again doesn't take it up
    assert (config.RESULTS_ROOT / "gguf_requests" / f"{sid}.cancel").exists()
