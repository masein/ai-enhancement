"""12a.5b: IFEval, MMLU-Pro and MATH-500 are slow, and optional; trials stop.

A full MMLU-Pro was about 11½ hours per model on hf, and a trial ran for two
hours after Ctrl+C, holding the GPU. So the three are chosen one by one, with
the time they will take said first; MMLU-Pro is a seeded 1,200 unless the full
12,032 is asked for, and the board keeps a subset apart from a full run; a
slow run says how far it is, and a cancel keeps what finished; and a trial
stops — lm_eval and everything it started killed, the lock freed — on Ctrl+C,
when its terminal goes away, and at --max-minutes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import generative as gen
import report_lm_eval as report
import trial_stop
from conftest import make_service
from generative_fixture import FIX, write_run, write_task
from service import config, db, runner

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-2B"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client
    client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# 5. optional, a subset by default, and an estimate from this server's runs
# ---------------------------------------------------------------------------

def test_a_run_picks_among_the_three_and_mmlu_pro_is_a_subset_unless_full(svc):
    def post(**body):
        return svc.post("/api/submissions", json={"hf_id": MODEL, "suite": "generative", **body})
    r = post(tasks=["ifeval"])
    assert r.status_code == 200 and r.json()["tasks"] == ["ifeval"]
    row = db.get(r.json()["id"])
    assert json.loads(row["tasks"]) == ["ifeval"] and row["subset"] == 0   # no MMLU-Pro in it
    # MMLU-Pro: the seeded 1,200 unless the full run is asked for
    sub = post(tasks=["mmlu_pro", "ifeval"])
    assert db.get(sub.json()["id"])["subset"] == config.GEN_MMLU_PRO_SUBSET == 1200
    full = post(tasks=["mmlu_pro", "ifeval"], subset=0)
    assert full.json()["id"] != sub.json()["id"] and db.get(full.json()["id"])["subset"] == 0
    # all three is the whole suite
    every = svc.post("/api/submissions", json={"hf_id": "Qwen/Qwen3.5-4B", "suite": "generative",
                                               "tasks": ["mmlu_pro", "ifeval", "hendrycks_math500"]})
    assert json.loads(db.get(every.json()["id"])["tasks"]) == []
    bad = post(tasks=["gsm8k"])
    assert bad.status_code == 422 and "not one of the three: gsm8k" in bad.json()["detail"]
    other = svc.post("/api/submissions", json={"hf_id": MODEL, "suite": "full",
                                               "tasks": ["ifeval"]})
    assert other.status_code == 422 and "picks among IFEval" in other.json()["detail"]


def test_the_pace_is_this_servers_or_a_rough_guess(svc):
    pace = svc.get("/api/gen/pace").json()
    assert pace["subset"] == 1200
    assert pace["items"] == {"ifeval": 541, "mmlu_pro": 12032, "hendrycks_math500": 500}
    assert pace["tasks"]["ifeval"] == {"per_b": 7.4, "runs": 0}             # a rough guess
    # two runs here: seconds per item per billion parameters, their median
    for params, secs in ((2_000_000_000, 400.0), (1_000_000_000, 150.0)):
        sid = db.add(MODEL, "instruct", "generative", "t", "")
        db.update(sid, status="done", params=params,
                  task_times=json.dumps({"ifeval": {"s": secs, "n": 20}}))
    got = svc.get("/api/gen/pace").json()["tasks"]
    assert got["ifeval"]["runs"] == 2 and got["ifeval"]["per_b"] == 10.0     # 400/20/2, 150/20/1
    assert got["mmlu_pro"] == {"per_b": 2.0, "runs": 0}


# ---------------------------------------------------------------------------
# on Runs: MMLU-Pro last, how far it is, and a cancel that keeps what finished
# ---------------------------------------------------------------------------

def fake_gpu(monkeypatch, on_task):
    monkeypatch.setattr(runner, "preflight", lambda hf_id, kind, **k: {
        "kind": "instruct", "params": 2_000_000_000, "vocab": 50304, "batch": 8,
        "need_gb": 5.0, "remote_code": False, "has_template": True,
        "archinfo": {"thinking": "switch"}})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "not in this image"))
    seen = []

    def run(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        task = cmd[cmd.index("--tasks") + 1]
        seen.append(task)
        return on_task(sid, task, Path(cmd[cmd.index("--output_path") + 1]), lf, on_poll)
    monkeypatch.setattr(runner, "_run_task", run)
    return seen


def answer(sid, task, out, lf, on_poll, *, subset=0):
    write_task(out / "run", task, margs=f"pretrained={MODEL},dtype=bfloat16", backend="hf",
               subset=subset)
    return 0


def test_mmlu_pro_runs_last_and_says_how_far_it_is(svc, monkeypatch):
    said = []

    def slow(sid, task, out, lf, on_poll):
        if task == "mmlu_pro":
            for done, left in ((0, "?"), (340, "25:21"), (1200, "00:00")):
                lf.write(f"\rRunning generate_until requests:  28%|██▊       | {done}/1200 "
                         f"[10:02<{left},  1.77s/it]")
                lf.flush()
                on_poll()
                said.append(db.get(sid)["progress"])
        return answer(sid, task, out, lf, on_poll, subset=1200)
    seen = fake_gpu(monkeypatch, slow)
    sid = svc.post("/api/submissions", json={"hf_id": MODEL, "suite": "generative"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert seen == ["ifeval", "hendrycks_math500", "mmlu_pro"]
    assert said == ["MMLU-Pro 0 of 1,200", "MMLU-Pro 340 of 1,200 · about 25 min left",
                    "MMLU-Pro 1,200 of 1,200"]
    row = db.get(sid)
    assert row["status"] == "done"
    # each one's seconds and items, for the next estimate
    times = json.loads(row["task_times"])
    assert set(times) == {"ifeval", "hendrycks_math500", "mmlu_pro"}
    assert times["mmlu_pro"]["n"] == 1200 and times["ifeval"]["n"] == len(
        [f for f in FIX if f["task"] == "ifeval"])


def test_the_progress_line_reads_the_harness_bar():
    bar = ("Map: 100%|██████████| 717/717 [00:00<00:00, 9000 examples/s]\n"
           "Running generate_until requests:   5%|▌         | 60/1200 [01:40<1:10:05,  3.5s/it]")
    assert runner.progress_line(bar, "MMLU-Pro") == "MMLU-Pro 60 of 1,200 · about 1 h 10 min left"
    assert runner.progress_line("Map: 100%|█| 717/717 [00:00<00:00]", "MMLU-Pro") is None


def test_a_cancelled_run_keeps_the_finished_benchmarks(svc, monkeypatch):
    def cancel_during_mmlu_pro(sid, task, out, lf, on_poll):
        if task == "mmlu_pro":
            return runner.CANCELED
        return answer(sid, task, out, lf, on_poll)
    fake_gpu(monkeypatch, cancel_during_mmlu_pro)
    sid = svc.post("/api/submissions", json={"hf_id": MODEL, "suite": "generative"}).json()["id"]
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "canceled"
    assert row["progress"] == "cancelled during MMLU-Pro · IFEval and MATH-500 kept"
    # the two finished are read in the board's words, and on the board
    mdir = config.OUT_DIR / MODEL.replace("/", "__")
    assert set(gen.read_json(mdir)["tasks"]) == {"ifeval", "hendrycks_math500"}
    p = report.build_payload(report.merge_runs(report.load_results(config.OUT_DIR)), "t",
                                str(config.OUT_DIR))
    m = next(x for x in p["models"] if x["id"] == MODEL)
    assert set(m["gen"]["tasks"]) == {"ifeval", "hendrycks_math500"}
    assert MODEL in p["cells"]["ifeval"] and MODEL in p["cells"]["hendrycks_math500"]
    assert MODEL not in p["cells"].get("mmlu_pro", {})


def test_a_subset_on_disk_is_not_a_full_run_and_is_asked_again(svc, monkeypatch):
    mdir = config.OUT_DIR / MODEL.replace("/", "__")
    write_task(mdir / "mmlu_pro_5shot" / "run", "mmlu_pro",
               margs=f"pretrained={MODEL},dtype=bfloat16", subset=1200)
    seen = fake_gpu(monkeypatch, answer)
    sid = svc.post("/api/submissions", json={"hf_id": MODEL, "suite": "generative",
                                             "tasks": ["mmlu_pro"], "subset": 0}).json()["id"]
    runner.run_submission(db.get(sid))
    assert seen == ["mmlu_pro"]                                   # not "already done"
    moved = list((config.OUT_DIR.with_name("earlier") / mdir.name).glob("mmlu_pro_5shot-*"))
    assert len(moved) == 1


# ---------------------------------------------------------------------------
# a subset is labelled, and never averaged or compared with a full run
# ---------------------------------------------------------------------------

def test_a_subset_is_a_benchmark_of_its_own_on_the_board(tmp_path):
    out = tmp_path / "full"
    write_run(out, "org/full-2b", backend="hf")
    write_run(out, "org/sub-2b", backend="hf", subset=1200)
    p = report.build_payload(report.merge_runs(report.load_results(out)), "t", str(out))
    assert "mmlu_pro_subset" in p["genTasks"] and "mmlu_pro_subset" in p["accTasks"]
    assert set(p["cells"]["mmlu_pro"]) == {"org/full-2b"}
    assert set(p["cells"]["mmlu_pro_subset"]) == {"org/sub-2b"}
    # labelled, in its name and its tooltip
    meta = p["tasks"]["mmlu_pro_subset"]
    assert "a seeded 1,200 of its 12,032" in meta["desc"]
    assert "never averaged or compared with the full MMLU-Pro" in meta["desc"]
    sub = next(m for m in p["models"] if m["id"] == "org/sub-2b")
    assert "mmlu_pro_subset" in sub["gen"]["tasks"] and "mmlu_pro" not in sub["gen"]["tasks"]
    # the two are never compared: each column's comparisons are its own models'
    for t, ms in (("mmlu_pro", {"org/sub-2b"}), ("mmlu_pro_subset", {"org/full-2b"})):
        assert not {x for row in p["sig"].get(t, []) for x in row[:2]} & ms


# ---------------------------------------------------------------------------
# 6. trials stop
# ---------------------------------------------------------------------------

# a harness that never finishes, and starts a process of its own
SLEEPER = ("import os, subprocess, sys, time\n"
           "kid = subprocess.Popen(['sleep', '300'])\n"
           "open(sys.argv[1], 'w').write(f'{os.getpid()} {kid.pid}')\n"
           "time.sleep(300)\n")


def gone(pid: int, within: float = 10.0) -> bool:
    """the process has ended (a zombie waiting for init counts as ended)"""
    t = time.time() + within
    while time.time() < t:
        try:
            os.kill(pid, 0)
            state = Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][:1]
            if state == "Z":
                return True
        except (ProcessLookupError, FileNotFoundError):
            return True
        time.sleep(0.1)
    return False


def trial_setup(tmp_path, monkeypatch, quick=()):
    """lm_eval replaced: the `quick` tasks answer at once, the rest never do"""
    pids = tmp_path / "pids"
    answers = tmp_path / "answers"
    answers.mkdir()
    for task in quick:
        write_task(answers / task, task, margs=f"pretrained={MODEL},dtype=bfloat16")

    def cmd(margs, task, shots, batch, task_out, **k):
        if task in quick:
            return [sys.executable, "-c", "import shutil, sys; shutil.copytree(sys.argv[1], "
                    "sys.argv[2], dirs_exist_ok=True)", str(answers / task), str(task_out)]
        return [sys.executable, "-c", SLEEPER, str(pids)]
    monkeypatch.setattr(runner, "lm_eval_cmd", cmd)
    monkeypatch.setattr(runner, "LOCK", tmp_path / ".run.lock")
    monkeypatch.setattr("service.hfmeta.preflight", lambda *a, **k: {
        "kind": "instruct", "batch": 8, "remote_code": False, "params": 2e9,
        "archinfo": {"thinking": "switch"}})
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", ""))
    return pids


def when_it_runs(pids: Path, then):
    """once the harness has started, `then`"""
    def go():
        t = time.time() + 30
        while not pids.exists() and time.time() < t:
            time.sleep(0.05)
        time.sleep(0.3)
        then()
    threading.Thread(target=go, daemon=True).start()


def left_nothing(pids: Path):
    parent, kid = (int(x) for x in pids.read_text().split())
    assert gone(parent) and gone(kid), "a harness process outlived the trial"
    assert not runner.LOCK.exists(), "the GPU lock was not freed"


def test_ctrl_c_stops_a_trial_kills_the_harness_and_frees_the_gpu(tmp_path, monkeypatch, capsys):
    import trial_generative as trial
    pids = trial_setup(tmp_path, monkeypatch, quick=("ifeval",))
    before = signal.getsignal(signal.SIGINT)
    when_it_runs(pids, lambda: os.kill(os.getpid(), signal.SIGINT))
    assert trial.main(["--model", MODEL]) == 4
    out = capsys.readouterr().out.splitlines()
    assert out[-2] == "stopped by SIGINT. Finished: IFEval."
    assert out[-1] == "stopped; GPU free"
    left_nothing(pids)
    assert signal.getsignal(signal.SIGINT) is before              # Ctrl+C is the test's again


def test_a_trial_stops_at_its_time_limit_and_says_what_it_finished(tmp_path, monkeypatch, capsys):
    import trial_standard as trial
    pids = trial_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("service.hfmeta.preflight", lambda *a, **k: {
        "kind": "base", "batch": 8, "remote_code": False, "params": 1.35e8})
    t0 = time.time()
    assert trial.main(["--model", "HuggingFaceTB/SmolLM2-135M", "--max-minutes", "0.05"]) == 4
    assert time.time() - t0 < 30
    out = capsys.readouterr().out.splitlines()
    assert out[-2] == "stopped at the 0.05-minute limit (--max-minutes). Finished: nothing."
    assert out[-1] == "stopped; GPU free"
    left_nothing(pids)


@pytest.fixture
def stdin_pipe(monkeypatch):
    """fd 0 is a pipe this test holds the other end of; put back after"""
    monkeypatch.setattr(trial_stop, "QUIET_START_S", 0.2)
    monkeypatch.setattr(trial_stop, "_stdin_is_a_pipe", lambda: True)
    keep = os.dup(0)
    r, w = os.pipe()
    os.dup2(r, 0)
    os.close(r)
    yield w
    os.dup2(keep, 0)
    os.close(keep)


def test_a_trial_stops_when_the_terminal_that_started_it_goes_away(stdin_pipe):
    """`docker compose exec -T` passes no Ctrl+C on; the trial's stdin ends"""
    with trial_stop.Stop(15) as stop:
        time.sleep(0.4)
        assert stop.over() == ""
        os.close(stdin_pipe)                                       # the terminal is gone
        t = time.time() + 5
        while not stop.over() and time.time() < t:
            time.sleep(0.05)
        assert stop.over() == "stopped: the terminal that started it went away"


def test_stdin_that_was_never_a_terminals_does_not_stop_a_trial(stdin_pipe):
    os.close(stdin_pipe)                                           # at its end from the start
    with trial_stop.Stop(15) as stop:
        time.sleep(0.5)
        assert stop.over() == ""


def test_a_trial_asks_two_per_mmlu_pro_subject_and_says_so_first():
    import trial_generative as trial
    order = ["ifeval", "hendrycks_math500", "mmlu_pro"]
    assert list(trial.counts(order, None).values()) == [
        "IFEval: 20 items", "MATH-500: 20 items", "MMLU-Pro: 28 items (2 per subject × 14)"]
    assert trial.counts(["mmlu_pro"], 3)["mmlu_pro"] == "MMLU-Pro: 42 items (3 per subject × 14)"
    per = runner.mmlu_pro_per_subject(2)
    assert len(per) == 14 and all(len(v) == 2 for v in per.values())
    assert per == runner.mmlu_pro_per_subject(2)                    # seeded, the same each time


def test_the_trials_help_lists_the_12h1_models():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "trial_generative.py"), "--help"],
                         capture_output=True, text=True, timeout=60).stdout
    for mid in ("Qwen/Qwen3.5-2B", "tencent/Youtu-LLM-2B", "ibm-granite/granite-4.0-h-1b",
                "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"):
        assert mid in out, mid
    assert "--max-minutes" in out


def test_a_trial_says_its_generating_rate_without_the_loading(capsys):
    """two items are mostly loading the model: lm_eval's bar says how long the
    answers themselves took, and that is the rate to compare"""
    import trial_generative as trial
    log = ("Loading checkpoint shards: 100%|██████████| 2/2 [00:21<00:00]\n"
           "Running generate_until requests: 100%|██████████| 2/2 [00:12<00:00,  6.0s/it]")
    assert runner.bar_seconds(log) == (2, 12)
    recs = [f["rec"] for f in FIX if f["task"] == "ifeval"][:2]
    s = trial.show("ifeval", recs, 40.0, generating=runner.bar_seconds(log))
    assert s["per_item"] == 20.0 and s["per_generating"] == 6.0
    assert "IFEval — 2 items in 40 s, 20.00 s per item · generating 6.00 s per item, loading " \
           "left out" in capsys.readouterr().out
