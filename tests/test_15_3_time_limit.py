"""15.3: the time limit on the board. A task that writes its answers gets a
limit sized for it — the answers still to write, each at the cap, at the
model's measured pace, with headroom, never under TASK_TIMEOUT_S; a task that
runs past it says it timed out, and what it wrote is kept, so the next run
asks only the rest (#167's IFEval was stopped twice by a fixed 3 hours at 259
and 260 of 300). The pace is measured from a task answered in one go. No
model runs: lm_eval is a stand-in, as in 15.1's tests."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pytest

import devicemark as dm
from service import config, db, runner
from service import devicemark as sdm
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture

QWEN = "Qwen/Qwen3.5-4B"


@pytest.fixture
def hf(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 4.0e9, "vocab": 248320, "batch": 8, "need_gb": 10.0,
        "remote_code": False, "has_template": True, "kind_reason": "chat template",
        "archinfo": {"thinking": "switch", "think_end": "</think>"}})
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "vLLM is not installed here"))
    monkeypatch.setattr(sdm, "prompt_tokens", lambda *a, **k: (911, True))
    monkeypatch.setattr(runner, "_answer_tokens", lambda m, r, texts: (
        sum(len(t) for t in texts) // 4, False))
    return svc


def lm_eval(monkeypatch, stop_after: dict[str, int] | None = None, secs_each: float = 12.0):
    """lm_eval as the runner calls it: each answer into the cache as it is
    written; `stop_after` a task's that many answers, it runs past its limit
    (what _run_task returns then: TIMED_OUT). Records each call's limit"""
    calls = []

    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        out = Path(cmd[cmd.index("--output_path") + 1])
        db_path = Path(cmd[cmd.index("--use_cache") + 1] + "_rank0.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        keys = [json.loads(x)["key"] for x in
                (config.DM_TASKS_DIR / f"{task}.jsonl").read_text().splitlines() if x]
        made = []
        with sqlite3.connect(db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS unnamed (key TEXT PRIMARY KEY, value BLOB)")
            have = dict(c.execute("SELECT key, value FROM unnamed"))
            for k in keys:
                if k in have:
                    continue
                if stop_after and task in stop_after and len(made) >= stop_after[task]:
                    calls.append({"task": task, "limit": runner._TASK_LIMIT.get(sid),
                                  "asked": len(made)})
                    lf.write("\n[service] timed out after 5.6 h: its limit …\n")
                    return runner.TIMED_OUT
                have[k] = "x" * 400 + " \\boxed{A}"           # about 100 tokens
                c.execute("INSERT INTO unnamed VALUES (?, ?)", (k, have[k]))
                c.commit()
                made.append(k)
        calls.append({"task": task, "limit": runner._TASK_LIMIT.get(sid), "asked": len(made)})
        n = len(made)
        lf.write(f"Running generate_until requests: 100%|##| {n}/{n} "
                 f"[{int(n * secs_each) // 60:02d}:{int(n * secs_each) % 60:02d}<00:00, "
                 f"{secs_each:.2f}s/it]\n")
        d = out / QWEN.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"samples_{task}_2026-10-02T12-00-00.jsonl", "w") as fh:
            for k in keys:
                fh.write(json.dumps({"doc": {"bench": dm.BENCH_OF[task], "key": k},
                                     "resps": [[have[k]]]}) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", run_task)
    return calls


def run(*only: str) -> dict:
    sid = db.add(QWEN, "instruct", "devicemark", ME, "", thinking=True, part="full",
                 tasks=list(only) or None)
    runner.run_submission(db.get(sid))
    return db.get(sid)


# ---------------------------------------------------------------------------
# the limit, sized
# ---------------------------------------------------------------------------

def test_a_limit_is_the_answers_left_at_the_cap_at_the_models_pace(monkeypatch):
    monkeypatch.setattr(config, "TASK_TIMEOUT_S", 3 * 3600)
    s, why = runner.task_limit(300, 4096, {"tok_s": 85.0, "sid": 167, "task": "dm_math"})
    assert s == pytest.approx(1.5 * 300 * 4096 / 85)                   # 6.0 h, not 3
    assert why == ("300 answers at the 4,096-token cap at 85 tokens a second (measured on "
                   "#167, dm_math), with 1.5× headroom")
    # no run has measured the model: the guess
    s, why = runner.task_limit(300, 4096, None)
    assert s == pytest.approx(1.5 * 300 * 4096 / config.PACE_GUESS_TOK_S)
    assert "a guess: no run here has measured this model yet" in why
    # a few answers left: never under TASK_TIMEOUT_S
    s, why = runner.task_limit(10, 4096, {"tok_s": 85.0, "sid": 1, "task": "t"})
    assert s == 3 * 3600 and why.endswith(", and never under 3.0 h")
    # a rented GPU's run: no limit at all
    monkeypatch.setattr(config, "TASK_TIMEOUT_S", 0)
    assert runner.task_limit(300, 4096, None) is None


def test_a_task_past_its_limit_is_stopped_and_says_it_timed_out(svc, monkeypatch,  # noqa: F811
                                                                  tmp_path):
    sid = db.add("org/m", "instruct", "devicemark", ME, "")
    monkeypatch.setitem(runner._TASK_LIMIT, sid, (1.0, "300 answers at the cap, with headroom"))
    log = tmp_path / "log.txt"
    t0 = time.time()
    with open(log, "w") as lf:
        status = runner._run_task(sid, [sys.executable, "-c", "import time; time.sleep(60)"],
                                  lf, {}, None, tmp_path)
    assert status == runner.TIMED_OUT and time.time() - t0 < 30
    assert ("[service] timed out after 1 min: its limit, 1 min, was 300 answers at the cap, with "
            "headroom") in log.read_text()


# ---------------------------------------------------------------------------
# on a run: timed out, kept, resumed; and the pace measured
# ---------------------------------------------------------------------------

def test_a_timed_out_task_keeps_its_answers_and_the_next_run_asks_the_rest(hf, monkeypatch):
    calls = lm_eval(monkeypatch, stop_after={"dm_ifeval": 120})
    row = run("dm_ifeval")
    assert row["status"] == "failed"
    assert row["error"].startswith("dm_ifeval timed out: its limit was ")
    assert ("300 answers at the 4,096-token cap at 20 tokens a second (a guess: no run here has "
            "measured this model yet), with 1.5× headroom") in row["error"]
    assert row["error"].endswith("The 120 answers it wrote are kept: resubmit and it asks only "
                                 "the other 180.")
    # the next run is sized for what is left, and asks only that
    calls = lm_eval(monkeypatch)
    row = run("dm_ifeval")
    assert row["status"] == "done", row["error"]
    assert calls[0]["asked"] == 180 and calls[0]["limit"][1].startswith("180 answers at the ")
    assert len(sdm.dm().answered_keys(config.OUT_DIR / (QWEN.replace("/", "__") + "__thinking")
                                      / "dm_ifeval_0shot", "dm_ifeval")) == 300


def test_the_pace_is_measured_and_sizes_the_next_tasks_limit(hf, monkeypatch):
    calls = lm_eval(monkeypatch, secs_each=2.0)
    row = run("dm_math", "dm_ifeval")
    assert row["status"] == "done", row["error"]
    pace = json.loads((config.OUT_DIR / QWEN.replace("/", "__") / runner.PACE_NAME).read_text())
    # each answer about 103 tokens, two seconds each: about 51 a second
    assert pace["sid"] == row["id"] and pace["answers"] in (100, 300)
    assert pace["tok_s"] == pytest.approx(pace["tokens"] / pace["seconds"])
    assert 45 < pace["tok_s"] < 55
    # IFEval ran first, at the guess; MATH after it, at the pace IFEval measured
    assert "a guess" in calls[0]["limit"][1] and calls[0]["task"] == "dm_ifeval"
    assert f"measured on #{row['id']}, dm_ifeval" in calls[1]["limit"][1]
    log = (config.LOGS_DIR / f"service_{row['id']}_{QWEN.replace('/', '__')}.log").read_text()
    assert re.search(r"\[service\] dm_math: its limit is \d+\.\d h — 100 answers at the 4,096-"
                     rf"token cap at \d+ tokens a second \(measured on #{row['id']}, dm_ifeval\)",
                     log)


def test_a_resumed_task_doesnt_measure_the_pace(hf, monkeypatch):
    lm_eval(monkeypatch, stop_after={"dm_math": 40})
    run("dm_math")
    lm_eval(monkeypatch)
    run("dm_math")
    assert runner.model_pace(QWEN) is None                    # 60 of 100 is not the pace
