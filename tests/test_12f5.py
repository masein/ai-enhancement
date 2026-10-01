"""12f.5, from the overnight GGUF and served runs:
- ARC-C, ARC-E and TruthfulQA failed in llama-perplexity, "task N requires a
  higher -np|--parallel value (at least 5)": the worker now passes -np and -c
  sized from the dataset's biggest task. The fake llama-perplexity keeps the
  real one's two limits, so these tests fail on the old command;
- MMLU was scored on each option's text (cloze): it is asked as lm_eval's mmlu
  asks it, lettered, and MMLU measured the old way goes to History, "cloze, not
  comparable";
- Re-run failed benchmarks queues only what a run didn't finish;
- a served run gave up on the run lock the GGUF worker held: it waits it out,
  "waiting for GGUF run #92 (about N min left)", and board runs and GGUF jobs
  take turns in the order they were queued;
- #89 sat "canceling" after its worker crashed: a restarted worker fails the
  job it was left on, "The GGUF worker restarted during this run.", and a
  canceled job with no worker is released.
No model runs: the fake llama-perplexity (tests/fake_llama_perplexity.py)."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import gguf_bench as gb
import gguf_data as gd
import gguf_worker as gw
from conftest import make_service
from service import config, db, gguf, runner
from service import worker as board_worker
from test_gguf_12f3 import docs_of, fake_binary

ARC5 = {"question": "Which is a mammal?",
        "choices": {"text": ["cat", "fish", "ant", "bee", "owl"], "label": list("ABCDE")},
        "answerKey": "A"}
TQA13 = {"question": "Is the earth flat?",
         "mc1_targets": {"choices": [f"Answer number {k}." for k in range(13)],
                         "labels": [1] + [0] * 12}}
LONG = "Read this. " + "The quick brown fox jumps over the lazy dog. " * 70     # ~3.1 kB


def docs(long_mmlu: bool = False):
    """the 12f.3 fixture's documents, with what the real sets have: an ARC
    question with five answers, a TruthfulQA one with thirteen, and (asked for)
    an MMLU question longer than llama-perplexity's default context"""
    base = docs_of()
    mmlu = base("mmlu") + ([{"question": LONG, "choices": ["a", "b", "c", "d"], "answer": 2,
                             "subject": "reading"}] if long_mmlu else [])
    by = {"mmlu": mmlu, "arc_challenge": base("arc_challenge") + [ARC5],
          "arc_easy": base("arc_easy") + [ARC5], "truthfulqa_mc1": base("truthfulqa_mc1") + [TQA13]}
    return lambda name: by.get(name) or base(name)


@pytest.fixture
def box(tmp_path, monkeypatch):
    res = tmp_path / "results"
    gd.build(res / "gguf_data", docs_of=docs(long_mmlu=True))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1000)
    monkeypatch.setattr(gw, "_stop", {"why": ""})
    for k in ("FAKE_PPL_ACC", "FAKE_PPL_DELAY", "FAKE_PPL_FAIL", "FAKE_PPL_LOCK"):
        monkeypatch.delenv(k, raising=False)
    return {"res": res, "model": model, "bin": fake_binary(tmp_path)}


def request(box, rid="1", benchmarks=None):
    man = json.loads((box["res"] / "gguf_data" / "manifest.json").read_text())["benchmarks"]
    # 14.3: every benchmark built by default (Mobile-MMLU-Pro is built apart)
    want = benchmarks or gb.DEFAULT
    req = {"id": rid, "sid": int(rid), "model": "gguf/test", "name": "test",
           "path": str(box["model"]), "flags": ["-ngl", "99", "--cpu-moe"], "benchmarks": want,
           "subset": 0, "pin": {}, "at": time.time(),
           "datasets": {b: {"sha256": man[b]["sha256"], "n": man[b]["n"]} for b in want}}
    d = box["res"] / "gguf_requests"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rid}.json").write_text(json.dumps(req))
    return req


def result(root: Path, rid) -> dict:
    return json.loads((root / "gguf_results" / f"{rid}.json").read_text())


def flag(cmd: list[str], name: str) -> int:
    return int(cmd[cmd.index(name) + 1])


# ---------------------------------------------------------------------------
# 1. -np and -c
# ---------------------------------------------------------------------------

def test_the_fake_refuses_what_llama_perplexity_refuses(box):
    """the command before 12f.5 — no -np, no -c — fails here as it did on the
    server, so the tests below would have caught it"""
    for b, why in (("arc_challenge", "task 4 requires a higher -np|--parallel value (at least 5)"),
                   ("truthfulqa", "task 1 requires a higher -np|--parallel value (at least 13)"),
                   ("mmlu", "does not fit in the context window")):
        info = gb.BENCHMARKS[b]
        cmd = gb.command(b, box["bin"], str(box["model"]), gb.DEFAULT_FLAGS,
                         str(box["res"] / "gguf_data" / info["data"]), 0)
        p = subprocess.run(cmd, capture_output=True, text=True)
        got = gb.parse(info["mode"], p.stdout + p.stderr)
        assert got["error"].startswith(why) and not got["final"], (b, got)


def test_np_and_c_come_from_the_datasets_biggest_task(box):
    man = json.loads((box["res"] / "gguf_data" / "manifest.json").read_text())["benchmarks"]
    # the manifest carries each multiple-choice file's shape
    assert [man[b]["max_answers"] for b in ("mmlu", "arc_challenge", "truthfulqa")] == [4, 5, 13]
    # 12y: ARC after lm_eval's own prompt
    assert man["mmlu"]["format"] == "lettered" and man["arc_easy"]["format"] == "prompted"
    assert "max_answers" not in man["hellaswag"]
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box)
    assert w.once() is True
    r = result(box["res"], 1)
    assert r["status"] == "done", r["line"]
    cmds = {b: v["command"] for b, v in r["benchmarks"].items()}
    assert [flag(cmds[b], "-np") for b in ("mmlu", "arc_challenge", "arc_easy", "truthfulqa")] \
        == [4, 5, 5, 13]
    # -np × -c holds the biggest task: the long MMLU question needs more than
    # llama-perplexity's own 4 × 512; the others keep its 512
    assert flag(cmds["mmlu"], "-c") * 4 >= man["mmlu"]["max_task_tokens"] > 4 * 512
    assert flag(cmds["mmlu"], "-c") % 256 == 0
    assert flag(cmds["truthfulqa"], "-c") == flag(cmds["arc_easy"], "-c") == 512
    # after the model's flags, so they are the ones it runs with
    assert cmds["arc_easy"].index("-np") > cmds["arc_easy"].index("--cpu-moe")
    # HellaSwag and Winogrande are as they were
    assert "-np" not in cmds["hellaswag"] and "-c" not in cmds["winogrande"]
    # what it measured on: the file's shape, kept with the result
    assert r["datasets"]["truthfulqa"]["max_answers"] == 13
    assert r["datasets"]["mmlu"]["format"] == "lettered"


def test_the_flags_bound_any_tokenizer_and_keep_the_default_floor():
    assert gb.mc_flags({"max_answers": 2, "max_task_tokens": 10}) == ["-np", "4", "-c", "512"]
    assert gb.mc_flags({"max_answers": 13, "max_task_tokens": 13 * 600}) == ["-np", "13", "-c",
                                                                              "768"]
    # a token is at least a byte: the bound is over the bytes, and a few more a sequence
    q, answers = "Qü?", ["yes", "no"]
    assert gb.task_tokens(q, answers) == len(q.encode()) + 2 + (3 + 3) + (2 + 3)


def test_a_manifest_from_before_needs_nothing(box):
    """the worker sizes -np and -c from the file itself; a build fills the
    shape in for every multiple-choice file, the ones it doesn't rebuild too"""
    mpath = box["res"] / "gguf_data" / "manifest.json"
    m = json.loads(mpath.read_text())
    shas = {b: v["sha256"] for b, v in m["benchmarks"].items()}
    for v in m["benchmarks"].values():
        for k in ("max_answers", "max_task_tokens", "format"):
            v.pop(k, None)
    mpath.write_text(json.dumps(m))
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box, benchmarks=["arc_easy", "truthfulqa"])
    w.once()
    r = result(box["res"], 1)
    assert r["status"] == "done", r["line"]
    assert flag(r["benchmarks"]["truthfulqa"]["command"], "-np") == 13
    # gguf_data.py --only hellaswag: every multiple-choice entry gets its shape
    # back, and no file changes
    gd.build(box["res"] / "gguf_data", only=["hellaswag"], docs_of=docs(long_mmlu=True))
    m = json.loads(mpath.read_text())["benchmarks"]
    assert m["truthfulqa"]["max_answers"] == 13 and m["arc_challenge"]["max_answers"] == 5
    assert {b: v["sha256"] for b, v in m.items()} == shas


# ---------------------------------------------------------------------------
# 2. MMLU as lm_eval asks it
# ---------------------------------------------------------------------------

def test_mmlu_is_asked_as_lm_evals_mmlu_asks_it(tmp_path):
    doc = {"question": "  Which of these is prime?\n", "choices": ["4", "", "9", "7"], "answer": 3,
           "subject": "high_school_mathematics"}
    t = gd.mmlu_task(doc)
    assert t["question"] == (
        "The following are multiple choice questions (with answers) about high school "
        "mathematics.\n\nWhich of these is prime?\nA. 4\nB. \nC. 9\nD. 7\nAnswer:")
    # llama-perplexity scores question + " " + answer: " A", lm_eval's continuation
    assert t["answers"] == ["A", "B", "C", "D"] and t["labels"] == [0, 0, 0, 1]
    # an empty option is still a letter: nothing is left out
    man = gd.build(tmp_path, only=["mmlu"], docs_of=lambda name: [doc])["benchmarks"]
    assert (man["mmlu"]["n"], man["mmlu"]["skipped"], man["mmlu"]["format"]) == (1, 0, "lettered")
    back = gd.read_mc_binary((tmp_path / "mmlu-test.bin").read_bytes())
    assert back[0]["question"] == t["question"] and back[0]["answers"] == ["A", "B", "C", "D"]
    # ARC and TruthfulQA stay each option's text — 12y: after lm_eval's own prompt
    arc = gd.arc_task(ARC5)
    assert arc["question"] == "Question: Which is a mammal?\nAnswer:" and arc["answers"][0] == "cat"


def svc_box(tmp_path, monkeypatch, docs_fn=None):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_fn or docs())
    model = tmp_path / "orig.gguf"
    model.write_bytes(b"GGUF" + b"\1" * 500)
    monkeypatch.setattr(gw, "_stop", {"why": ""})
    monkeypatch.setattr(runner, "LOCK", config.RESULTS_ROOT / ".run.lock")
    for k in ("FAKE_PPL_ACC", "FAKE_PPL_DELAY", "FAKE_PPL_FAIL", "FAKE_PPL_LOCK"):
        monkeypatch.delenv(k, raising=False)
    gid = client.post("/api/gguf/models", json={
        "name": "Qwen3.6 original", "path": str(model), "based_on": "Qwen/Qwen3.6-35B-A3B",
        "how": "unsloth UD-Q4_K_XL", "by": "masein"}).json()["model"]["id"]
    return client, appmod, {"gid": gid, "model": model, "bin": fake_binary(tmp_path)}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, box = svc_box(tmp_path, monkeypatch)
    yield client, appmod, box
    client.__exit__(None, None, None)


def rows(client) -> dict:
    return {r["id"]: r for r in client.get("/api/submissions").json()}


def results(client, appmod) -> dict:
    appmod._cache.update(key=None, payload=None, at=0.0)
    return client.get("/api/results").json()


def beat(age: float = 0.0) -> None:
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time() - age}))


def test_mmlu_measured_on_the_cloze_file_goes_to_history(svc):
    client, appmod, box = svc
    gid = box["gid"]
    man = gguf.manifest()
    out = config.RESULTS_ROOT / "gguf_results"
    out.mkdir(parents=True, exist_ok=True)
    # run #90 as the worker wrote it before 12f.5: no format with its datasets,
    # on the dataset files it had (MMLU's was the cloze one)
    old = {"id": "90", "sid": 90, "model": gid, "status": "done", "line": "",
           "subset": 0, "finished_at": time.time() - 3600, "setup": gb.AS_BUILT,
           "file": {"name": "orig.gguf", "sha256": "f" * 64},
           "benchmarks": {"mmlu": {"status": "done", "acc": 0.425, "se": 0.004, "n": 14042},
                          "hellaswag": {"status": "done", "acc": 0.79, "se": 0.004, "n": 12}},
           "datasets": {"mmlu": {"file": "mmlu-test.bin", "sha256": "c" * 64},
                        "hellaswag": {"file": "hellaswag-validation.txt",
                                      "sha256": man["hellaswag"]["sha256"]}}}
    (out / "90.json").write_text(json.dumps(old))
    g = results(client, appmod)["gguf"]
    assert set(g["models"][gid]) == {"hellaswag"}               # MMLU is not in the column
    [h] = g["history"][gid]
    assert h["benchmarks"]["mmlu"]["current"] is False
    assert h["benchmarks"]["mmlu"]["earlier"] == "cloze, not comparable"
    assert h["benchmarks"]["hellaswag"]["current"] is True
    # even measured on the file the manifest names, before it was rebuilt
    old["datasets"]["mmlu"]["sha256"] = man["mmlu"]["sha256"]
    (out / "90.json").write_text(json.dumps(old))
    g = results(client, appmod)["gguf"]
    assert "mmlu" not in g["models"][gid]
    assert g["history"][gid][0]["benchmarks"]["mmlu"]["earlier"] == "cloze, not comparable"


def test_the_cloze_file_is_refused_until_it_is_built_again(svc):
    client, _, box = svc
    mpath = config.RESULTS_ROOT / "gguf_data" / "manifest.json"
    m = json.loads(mpath.read_text())
    m["benchmarks"]["mmlu"].pop("format")                       # a manifest from before 12f.5
    mpath.write_text(json.dumps(m))
    r = client.post("/api/gguf/runs", json={"model": box["gid"], "by": "masein"})
    assert r.status_code == 422
    assert r.json()["detail"] == gguf.OLD_DATASET.format(
        label="MMLU", what="each option's text scored (cloze)")
    # the other benchmarks queue as before
    r = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["arc_easy"],
                                            "by": "masein"})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# 3. Re-run failed benchmarks
# ---------------------------------------------------------------------------

def test_rerun_queues_only_the_failed_benchmarks(svc):
    client, appmod, box = svc
    gid = box["gid"]
    sid = client.post("/api/gguf/runs", json={"model": gid, "by": "masein"}).json()["id"]
    # run #90, as it ran: the command without -np and -c
    real = gb.mc_flags
    gb.mc_flags = lambda shape: []
    try:
        gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05).once()
    finally:
        gb.mc_flags = real
    row = rows(client)[sid]
    assert row["status"] == "failed"
    assert "ARC-C: task 4 requires a higher -np|--parallel value (at least 5)" in row["error"]
    assert "TruthfulQA: task 1 requires a higher -np|--parallel value (at least 13)" in row["error"]
    assert row["gguf_left"] == ["arc_challenge", "arc_easy", "truthfulqa"]
    r = client.post(f"/api/gguf/runs/{sid}/rerun", json={"by": "masein"})
    assert r.status_code == 200, r.text
    new = r.json()["id"]
    assert r.json()["benchmarks"] == ["arc_challenge", "arc_easy", "truthfulqa"]
    nrow = rows(client)[new]
    assert json.loads(nrow["tasks"]) == ["arc_challenge", "arc_easy", "truthfulqa"]
    assert nrow["note"] == f"setup: as built · what #{sid} didn't finish"
    req = json.loads((config.RESULTS_ROOT / "gguf_requests" / f"{new}.json").read_text())
    assert req["benchmarks"] == ["arc_challenge", "arc_easy", "truthfulqa"]
    assert req["setup"]["id"] == "as-built" and req["subset"] == 0
    gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05).once()
    assert rows(client)[new]["status"] == "done"
    # the finished ones weren't redone, and the column has all six
    assert set(result(config.RESULTS_ROOT, new)["benchmarks"]) == {"arc_challenge", "arc_easy",
                                                                    "truthfulqa"}
    g = results(client, appmod)["gguf"]
    assert set(g["models"][gid]) == set(gb.DEFAULT)
    assert g["models"][gid]["mmlu"]["sid"] == sid and g["models"][gid]["arc_easy"]["sid"] == new
    # nothing left to re-run, and a run still going can't be
    r = client.post(f"/api/gguf/runs/{new}/rerun", json={"by": "masein"})
    assert r.status_code == 422 and r.json()["detail"] == \
        f"Every benchmark of run #{new} finished: nothing to re-run."
    going = client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mmlu"],
                                                "by": "masein"}).json()["id"]
    r = client.post(f"/api/gguf/runs/{going}/rerun", json={"by": "masein"})
    assert r.status_code == 422 and r.json()["detail"] == f"Run #{going} hasn't finished yet."
    assert client.post("/api/gguf/runs/99999/rerun", json={"by": "masein"}).status_code == 404


# ---------------------------------------------------------------------------
# 4. the run lock, and one queue
# ---------------------------------------------------------------------------

def gguf_lock(sid: int, age: float = 0.0) -> Path:
    lock = runner.LOCK
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "pid").write_text("999999")                     # a host pid the board can't see
    (lock / "submission").write_text(str(sid))
    (lock / "owner").write_text("gguf-worker host 999999")
    (lock / "heartbeat").write_text("1")
    os.utime(lock / "heartbeat", (time.time() - age, time.time() - age))
    return lock


def test_a_run_waits_for_a_live_gguf_worker_saying_for_which(svc, monkeypatch):
    client, _, box = svc
    g = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["mmlu"],
                                            "by": "masein"}).json()["id"]
    out = config.RESULTS_ROOT / "gguf_results"
    (out / f"{g}.json").write_text(json.dumps({
        "id": str(g), "sid": g, "model": box["gid"], "status": "running",
        "benchmarks": {"mmlu": {"status": "running", "done": 100, "total": 1000,
                                "secs_each": 1.2}}}))
    sid = db.add("served/LDA-phone-build", "instruct", "generative", "masein", "")
    lock = gguf_lock(g)
    monkeypatch.setattr(config, "GPU_WAIT_MAX_S", 0)            # it would give up at once
    monkeypatch.setattr(config, "GPU_POLL_S", 0)
    said = []

    def tick(_s):
        said.append((db.get(sid)["status"], db.get(sid)["progress"]))
        os.utime(lock / "heartbeat")                          # the worker beats on
        if len(said) == 3:
            runner.release_lock()                             # its job ends
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=time.time, sleep=tick))
    assert runner.wait_for_lock(sid) is True
    # 900 answers left at 1.2 s each: 18 minutes, and never given up on
    assert said == [("waiting_lock", f"waiting for GGUF run #{g} (about 18 min left)")] * 3
    assert (runner.LOCK / "submission").read_text() == str(sid)
    runner.release_lock()
    # with nothing to estimate from, it says so without one
    assert gguf.waiting_line(0) == "waiting for a GGUF run"
    (out / f"{g}.json").write_text(json.dumps({"id": str(g), "status": "running",
                                               "benchmarks": {"mmlu": {"status": "done"}}}))
    assert gguf.waiting_line(g) == f"waiting for GGUF run #{g}"


def test_a_manual_run_holding_the_lock_is_still_given_up_on(svc, monkeypatch):
    lock = runner.LOCK
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "pid").write_text(str(os.getpid()))              # a live process, no heartbeat
    sid = db.add("fx/good-750m", "base", "quick", "masein", "")
    monkeypatch.setattr(config, "GPU_WAIT_MAX_S", 0)
    monkeypatch.setattr(config, "GPU_POLL_S", 0)
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=time.time, sleep=lambda s: None))
    assert runner.wait_for_lock(sid) is False
    row = db.get(sid)
    assert row["status"] == "failed" and row["error"] == (
        "gave up waiting for the run lock — a manual run has held the GPU for hours; "
        "resubmit later.")
    # and a GGUF worker gone quiet is no reason to wait: its lock is taken over
    runner.release_lock()
    gguf_lock(5, age=300)
    assert runner.gguf_holding() is None and runner.wait_for_lock(sid) is True
    runner.release_lock()


def test_board_runs_and_gguf_jobs_take_turns_in_the_order_queued(svc):
    client, _, box = svc
    beat()                                                    # the worker is running
    a = db.add("fx/good-750m", "base", "quick", "masein", "")
    b = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["mmlu"],
                                            "by": "masein"}).json()["id"]
    c = db.add("served/LDA-phone-build", "instruct", "generative", "masein", "")
    # B waits for A, queued before it: its request isn't the worker's yet
    held = config.RESULTS_ROOT / "gguf_requests" / "held" / f"{b}.json"
    assert held.exists() and not (config.RESULTS_ROOT / "gguf_requests" / f"{b}.json").exists()
    assert rows(client)[b]["progress"] == f"waiting for run #{a}, queued before it"
    assert db.claim_next(gguf_first=True)["id"] == a
    # C waits for B while A runs, and B for A
    assert db.claim_next(gguf_first=True) is None and db.gguf_ahead() == (c, b)
    rows(client)
    assert held.exists()
    db.update(a, status="done", finished_at=time.time())
    rows(client)                                              # sync hands B to the worker
    assert not held.exists() and (config.RESULTS_ROOT / "gguf_requests" / f"{b}.json").exists()
    # the board's queue says what C waits for
    assert board_worker._gguf_first() is True
    assert db.get(c)["progress"].startswith(f"waiting for GGUF run #{b} (about ")
    assert db.claim_next(gguf_first=True) is None
    assert gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05).once() is True
    assert rows(client)[b]["status"] == "done"
    assert db.claim_next(gguf_first=True)["id"] == c


def test_with_no_gguf_worker_board_runs_go_ahead_and_a_held_job_cancels(svc):
    client, _, box = svc
    a = db.add("fx/good-750m", "base", "quick", "masein", "")
    b = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["mmlu"],
                                            "by": "masein"}).json()["id"]
    c = db.add("fx/good-750m", "base", "full", "masein", "")
    beat(age=300)                                             # the worker isn't running
    assert board_worker._gguf_first() is False
    assert db.claim_next(gguf_first=False)["id"] == a
    assert db.claim_next(gguf_first=False)["id"] == c        # nothing to wait for
    # B, still waiting its turn, is canceled at once and never reaches the worker
    assert client.post(f"/api/submissions/{b}/cancel").json()["status"] == "canceled"
    assert not (config.RESULTS_ROOT / "gguf_requests" / "held" / f"{b}.json").exists()
    assert not list((config.RESULTS_ROOT / "gguf_requests").glob("*.json"))


# ---------------------------------------------------------------------------
# 5. a worker restarted mid-job, and a stop asked of a gone one
# ---------------------------------------------------------------------------

def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_a_restarted_worker_fails_the_job_it_was_left_on(svc, capsys):
    client, _, box = svc
    sid = client.post("/api/gguf/runs", json={"model": box["gid"],
                                              "benchmarks": ["mmlu", "hellaswag"],
                                              "by": "masein"}).json()["id"]
    waiting = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["arc_easy"],
                                                  "by": "masein"}).json()["id"]
    out = config.RESULTS_ROOT / "gguf_results"
    # #89: the worker crashed on it; MMLU had finished, HellaSwag was running
    (out / f"{sid}.json").write_text(json.dumps({
        "id": str(sid), "sid": sid, "model": box["gid"], "status": "running",
        "line": "HellaSwag: 3 of 12", "subset": 0, "setup": gb.AS_BUILT,
        "benchmarks": {"mmlu": {"status": "done", "acc": 0.5, "se": 0.1, "n": 6},
                       "hellaswag": {"status": "running", "done": 3, "total": 12}}}))
    (out / f"{waiting}.json").write_text(json.dumps({
        "id": str(waiting), "sid": waiting, "model": box["gid"], "status": "waiting",
        "line": f"waiting for the run lock: run #{sid} holds it",
        "benchmarks": {"arc_easy": {"status": "queued"}}}))
    lock = runner.LOCK
    lock.mkdir(parents=True, exist_ok=True)
    pid = dead_pid()
    for f, v in (("pid", pid), ("submission", sid), ("heartbeat", time.time()),
                 ("owner", f"gguf-worker {socket.gethostname()} {pid}")):
        (lock / f).write_text(str(v))
    beat()                                                    # the new worker is running
    assert rows(client)[sid]["status"] == "running"
    assert client.post(f"/api/submissions/{sid}/cancel").json()["status"] == "canceling"
    # while the worker is alive the board can't tell: it stayed "canceling"
    assert rows(client)[sid]["status"] == "canceling"
    # a restarted worker never keeps a dead one's lock alive
    w = gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05)
    old = (lock / "heartbeat").stat().st_mtime
    os.utime(lock / "heartbeat", (old - 30, old - 30))
    w.heartbeat()
    assert (lock / "heartbeat").stat().st_mtime == old - 30
    # the worker starts: the job it was left on fails, saying so; the finished
    # benchmark is kept; the dead worker's lock goes
    handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    try:
        assert gw.main(["--results", str(config.RESULTS_ROOT), "--binary", box["bin"], "--poll",
                        "0.05", "--once"]) == 0
    finally:
        signal.signal(signal.SIGINT, handlers[0])
        signal.signal(signal.SIGTERM, handlers[1])
    assert f"run #{sid} was left running by a worker that stopped" in capsys.readouterr().out
    r = result(config.RESULTS_ROOT, sid)
    assert r["status"] == "failed" and r["line"] == gw.RESTARTED
    assert r["benchmarks"]["mmlu"]["status"] == "done"
    assert r["benchmarks"]["hellaswag"] == {"status": "failed", "done": 3, "total": 12,
                                            "error": gw.RESTARTED}
    # not run again, and the job only waiting for the lock ran in its place
    assert (config.RESULTS_ROOT / "gguf_requests" / "done" / f"{sid}.json").exists()
    assert not (config.RESULTS_ROOT / "gguf_requests" / f"{sid}.cancel").exists()
    assert result(config.RESULTS_ROOT, waiting)["status"] == "done"
    assert not lock.exists()
    row = rows(client)[sid]
    assert row["status"] == "failed" and row["error"] == gw.RESTARTED
    assert row["gguf_left"] == ["hellaswag"]


def test_a_stop_asked_of_a_gone_worker_is_done(svc):
    client, _, box = svc
    sid = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["mmlu"],
                                              "by": "masein"}).json()["id"]
    out = config.RESULTS_ROOT / "gguf_results"
    (out / f"{sid}.json").write_text(json.dumps({"id": str(sid), "status": "running",
                                                 "benchmarks": {}}))
    beat(age=30)
    assert rows(client)[sid]["status"] == "running"
    assert client.post(f"/api/submissions/{sid}/cancel").json()["status"] == "canceling"
    beat(age=300)
    row = rows(client)[sid]
    assert row["status"] == "canceled"
    assert row["progress"] == gguf.GONE_CANCELED.format(ago=" (last seen 5 min ago)")
    # and one canceled before the worker ever wrote a word of it
    other = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["mmlu"],
                                                "by": "masein"}).json()["id"]
    db.update(other, status="canceling")
    row = rows(client)[other]
    assert row["status"] == "canceled" and row["progress"] == gguf.NOT_STARTED
    assert (config.RESULTS_ROOT / "gguf_requests" / f"{other}.cancel").exists()


def test_the_board_lets_go_of_a_job_the_live_worker_isnt_on(svc):
    """a worker runs one job at a time: #89 still "running" while the worker
    says it's on #90 was left by one that stopped — released without waiting
    for a restart, since it holds up the queue"""
    client, _, box = svc
    first, second = (client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["mmlu"],
                                                         "by": "masein"}).json()["id"]
                     for _ in range(2))
    out = config.RESULTS_ROOT / "gguf_results"
    for rid in (first, second):
        (out / f"{rid}.json").write_text(json.dumps({"id": str(rid), "status": "running",
                                                     "benchmarks": {}}))
    beat()                                   # a worker from before 12f.5 may say nothing
    got = rows(client)
    assert got[first]["status"] == got[second]["status"] == "running"
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps(
        {"at": time.time(), "state": f"running {second}"}))
    got = rows(client)
    assert got[first]["status"] == "failed" and got[first]["error"] == gb.RESTARTED
    assert got[second]["status"] == "running"
    # idle between two jobs says nothing either way
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps(
        {"at": time.time(), "state": "idle"}))
    assert rows(client)[second]["status"] == "running"
    # and a job with neither a request nor a result for the worker never runs
    lost = client.post("/api/gguf/runs", json={"model": box["gid"], "benchmarks": ["mmlu"],
                                               "by": "masein"}).json()["id"]
    (config.RESULTS_ROOT / "gguf_requests" / f"{lost}.json").unlink()
    assert rows(client)[lost]["status"] == "queued"          # just queued: not yet
    db.update(lost, created_at=time.time() - 300)
    row = rows(client)[lost]
    assert row["status"] == "failed" and row["error"] == gguf.LOST
