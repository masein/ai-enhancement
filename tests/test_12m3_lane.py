"""12m.3, a follow-up: a run of a model from OpenRouter never touches the GPU,
yet it waited for and held the run lock and took its turn in the one queue.
Remote runs have a lane of their own: they skip the run lock and the GPU
queue, go one at a time among themselves, and a GPU run or a GGUF job never
waits behind one. The judge they call is any run's.

Against 12m.3's fake OpenAI-compatible server, answering at OpenRouter's
address; nothing calls OpenRouter, and no model runs."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from conftest import make_service
from service import config, db, runner, worker
from test_12m3 import KEY, LUNA, ME, OR, SID, fake, outside  # noqa: F401 — the fake and its guard


def _no_lock(*a, **k):
    raise AssertionError("a run of a model from OpenRouter touched the run lock")


@pytest.fixture
def lane(tmp_path, monkeypatch, outside):  # noqa: F811
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(config, "OPENROUTER_BASE_URL", OR)
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    monkeypatch.setattr(config, "SERVED_RETRY_S", 0.4)
    monkeypatch.setattr(runner, "LOCK", config.RESULTS_ROOT / ".run.lock")
    r = client.post("/api/served/openrouter", json={"model": LUNA, "by": ME})
    assert r.status_code == 200, r.text
    yield client
    client.__exit__(None, None, None)


def board_lock(sid: int) -> Path:
    """the lock as a board run on the GPU holds it: this process's pid"""
    lock = runner.LOCK
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "pid").write_text(str(os.getpid()))
    (lock / "submission").write_text(str(sid))
    return lock


def gguf_lock(sid: int) -> Path:
    """the lock as the host's GGUF worker holds it: a heartbeat, beating"""
    lock = runner.LOCK
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "pid").write_text("999999")
    (lock / "submission").write_text(str(sid))
    (lock / "owner").write_text("gguf-worker host 999999")
    (lock / "heartbeat").write_text("1")
    return lock


def remote_run(monkeypatch, suite="everyday"):
    """a run of the model from OpenRouter, in its own lane, with the lock off limits"""
    monkeypatch.setattr(runner, "acquire_lock", _no_lock)
    monkeypatch.setattr(runner, "release_lock", _no_lock)
    sid = db.add(SID, "instruct", suite, ME, "")
    assert worker.once(remote=True) is True
    return db.get(sid)


def test_it_runs_while_a_board_run_holds_the_lock(lane, fake, monkeypatch):  # noqa: F811
    board = db.add("fx/good-750m", "base", "full", ME, "")
    db.update(board, status="running", progress="3/9 · arc_easy (5-shot)")
    lock = board_lock(board)
    row = remote_run(monkeypatch)
    assert row["status"] == "done", row["error"]
    assert len(fake.requests) > 0                             # asked at OpenRouter's address
    # the board run's lock is as it was: never waited for, never taken, never let go
    assert (lock / "submission").read_text() == str(board)
    assert db.get(board)["status"] == "running"
    # and its tasks were written in a folder of its own: a GPU run's are never rewritten
    assert (config.BENCH_ROOT / "remote" / "everyday-tasks" / "everyday.jsonl").is_file()
    assert not (config.EVERYDAY_TASKS_DIR / "everyday.jsonl").exists()


def test_it_runs_while_a_gguf_job_holds_the_lock(lane, fake, monkeypatch):  # noqa: F811
    g = db.add("gguf/qwen3.6-original", "instruct", "gguf", ME, "setup: as built",
               tasks=["mmlu"])
    db.update(g, status="running", progress="MMLU: 100 of 1,000")
    lock = gguf_lock(g)
    os.utime(lock / "heartbeat", (time.time(), time.time()))
    assert runner.gguf_holding() == g
    row = remote_run(monkeypatch)
    assert row["status"] == "done", row["error"]
    assert (lock / "submission").read_text() == str(g) and runner.gguf_holding() == g


def test_neither_a_gpu_run_nor_a_gguf_job_waits_for_it(lane):
    # queued first, and running: neither lane waits for the other
    remote = db.add(SID, "instruct", "everyday", ME, "")
    g = db.add("gguf/qwen3.6-original", "instruct", "gguf", ME, "setup: as built",
               tasks=["mmlu"])
    board = db.add("fx/good-750m", "base", "quick", ME, "")
    # the GGUF job's request isn't held for the OpenRouter run queued before it
    assert db.board_ahead(g) is None
    # the GPU lane skips it — even with the GGUF worker running (gguf_first)
    got = db.claim_next(gguf_first=True)
    assert got is None or got["id"] != remote            # a GGUF job queued first goes first
    db.update(g, status="done")
    assert db.claim_next(gguf_first=True)["id"] == board
    # and the remote lane takes only its own
    assert db.claim_next(remote=True)["id"] == remote
    assert db.claim_next(remote=True) is None
    # a GPU run takes the lock while the OpenRouter run runs: it never held it
    assert db.get(remote)["status"] == "preflight"
    assert runner.acquire_lock(board) is True
    runner.release_lock()


def test_the_gpu_lane_goes_on_while_one_runs(lane, monkeypatch):
    """a GPU run queued after an OpenRouter one isn't behind it"""
    db.add(SID, "instruct", "everyday", ME, "")
    board = db.add("fx/good-750m", "base", "quick", ME, "")
    seen = []
    monkeypatch.setattr(worker, "run_submission", lambda sub: seen.append(sub["id"]))
    monkeypatch.setattr(worker, "_gguf_first", lambda: False)
    assert worker.once() is True and seen == [board]
    # one remote run at a time: the lane is one thread, taking one row a turn
    assert worker.once(remote=True) is True and len(seen) == 2
    assert worker.once(remote=True) is False


def test_a_served_model_that_isnt_openrouter_keeps_the_lock_and_the_queue(lane):
    """a model served elsewhere can run on this host's own GPU: it keeps its place"""
    from service import served
    rec = {**served.get(SID), "id": "served/phone-build", "via": "", "name": "phone build"}
    db.served_put(rec)
    phone = db.add("served/phone-build", "instruct", "everyday", ME, "")
    assert db.claim_next(remote=True) is None
    assert db.claim_next()["id"] == phone
