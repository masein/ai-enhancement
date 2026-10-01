"""12f.3: MMLU, HellaSwag, Winogrande, ARC and TruthfulQA measured straight from
a GGUF by llama-perplexity, through the host's worker — with a fake
llama-perplexity (tests/fake_llama_perplexity.py) that prints what the real one
prints. No model runs.

The worker takes and releases the run lock and waits while a board run holds
it; what it prints is read into scores; Ctrl+C and the time limit keep the
benchmarks that finished; a changed model file or dataset stops a job in one
line; the heartbeat says when it isn't running. The converter's files read back
as llama-perplexity reads them, with lm_eval's questions. And the scores never
reach an lm_eval column or average."""

from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

import gguf_bench as gb
import gguf_data as gd
import gguf_worker as gw
from conftest import make_service
from service import config, db, gguf, runner

HERE = Path(__file__).resolve().parent
FAKE = HERE / "fake_llama_perplexity.py"


def docs_of(n_hs=12):
    """lm_eval's documents, a few of each: the shapes its tasks give"""
    mmlu = [{"question": f"What is {i} + {i}?", "choices": [str(2 * i - 1), str(2 * i),
                                                            str(2 * i + 1), "none"],
             "answer": 1, "subject": "arith"} for i in range(1, 7)]
    hs = [{"query": f"Cooking: He cuts the onion {i}", "choices": ["and cries.", "and flies.",
                                                                   "and sings.", "and swims."],
           "gold": 0} for i in range(n_hs)]
    wg = [{"sentence": "Sarah was better than Maria so _ got the easy cases.", "option1": "Sarah",
           "option2": "Maria", "answer": "2"},
          {"sentence": "When it rained, _ took the umbrella.", "option1": "John",
           "option2": "Bob", "answer": "1"},
          {"sentence": 'He said "hi" to _.', "option1": "A", "option2": "B", "answer": "1"}]
    arc = [{"question": f"Which is a mammal {i}?", "choices": {"text": ["cat", "fish", "ant"],
                                                              "label": ["A", "B", "C"]},
            "answerKey": "A"} for i in range(4)]
    # 12f.5: the question with an empty answer the format can't hold is ARC's
    # now: MMLU's answers are letters
    arc.append({"question": "An empty choice?", "choices": {"text": ["a", "", "c"],
                                                            "label": ["A", "B", "C"]},
                "answerKey": "A"})
    tqa = [{"question": "Is the earth flat?", "mc1_targets": {"choices": ["No.", "Yes."],
                                                             "labels": [1, 0]}}]
    # 12n.2: GPQA's shape, as lm_eval's process_docs leaves it — invented
    # questions, never GPQA's (tests/fixtures/gpqa)
    gpqa = [{"Question": f"An invented lamp draws {i} A at 12 V: its power?",
             "choice1": f"{12 * i} W", "choice2": f"{i} W", "choice3": "7 W", "choice4": "0 W",
             "answer": "(A)"} for i in range(1, 4)]
    by = {"mmlu": mmlu, "hellaswag": hs, "winogrande": wg, "arc_challenge": arc,
          "arc_easy": arc, "truthfulqa_mc1": tqa, "gpqa_diamond_zeroshot": gpqa}
    return lambda name: by[name]


def fake_binary(tmp_path: Path) -> str:
    p = tmp_path / "llama-perplexity"
    p.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    p.chmod(0o755)
    return str(p)


@pytest.fixture
def box(tmp_path, monkeypatch):
    """a results folder with the datasets, a GGUF file and the fake binary"""
    res = tmp_path / "results"
    gd.build(res / "gguf_data", docs_of=docs_of())
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1000)
    monkeypatch.setattr(gw, "_stop", {"why": ""})
    for k in ("FAKE_PPL_ACC", "FAKE_PPL_DELAY", "FAKE_PPL_FAIL", "FAKE_PPL_LOCK"):
        monkeypatch.delenv(k, raising=False)
    return {"res": res, "model": model, "bin": fake_binary(tmp_path)}


def request(box, rid="1", benchmarks=None, **over):
    man = json.loads((box["res"] / "gguf_data" / "manifest.json").read_text())["benchmarks"]
    # 14.3: every benchmark built by default (Mobile-MMLU-Pro is built apart)
    want = benchmarks or gb.DEFAULT
    req = {"id": rid, "sid": int(rid), "model": "gguf/test", "name": "test",
           "path": str(box["model"]), "flags": ["-ngl", "99", "--cpu-moe"], "benchmarks": want,
           "subset": 0, "pin": {}, "datasets": {b: {"sha256": man[b]["sha256"], "n": man[b]["n"]}
                                                 for b in want}, "at": time.time(), **over}
    d = box["res"] / "gguf_requests"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rid}.json").write_text(json.dumps(req))
    return req


def result(box, rid="1"):
    return json.loads((box["res"] / "gguf_results" / f"{rid}.json").read_text())


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------

def test_the_worker_takes_the_run_lock_measures_and_releases_it(box, monkeypatch):
    lock = box["res"] / ".run.lock"
    monkeypatch.setenv("FAKE_PPL_LOCK", str(lock))
    monkeypatch.setenv("FAKE_PPL_ACC", "0.75")
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box)
    assert w.once() is True
    r = result(box)
    # 12n.2: seven with GPQA Diamond
    assert r["status"] == "done" and r["line"] == "7 benchmarks measured"
    assert r["build"] == "version: 9999 (91428471f)"
    assert r["file"]["name"] == "model.gguf" and len(r["file"]["sha256"]) == 64
    # it held the lock while it ran, and gave it back
    log = (box["res"] / "gguf_results" / "1.mmlu.log").read_text()
    assert "lock held" in log and not lock.exists()
    # the request is done with
    assert not list((box["res"] / "gguf_requests").glob("*.json"))
    # each benchmark read into a score, with what it was measured on
    man = json.loads((box["res"] / "gguf_data" / "manifest.json").read_text())["benchmarks"]
    for b, v in r["benchmarks"].items():
        assert v["status"] == "done" and v["full"] is True, (b, v)
        assert v["n"] == man[b]["n"] and abs(v["acc"] - round(0.75 * v["n"]) / v["n"]) < 1e-6
        assert r["datasets"][b]["sha256"] == man[b]["sha256"]
        cmd = v["command"]
        assert cmd[:3] == [box["bin"], "-m", str(box["model"])] and "--cpu-moe" in cmd
    # HellaSwag says how many (its default is 400), the binary file goes in with -bf
    hs = r["benchmarks"]["hellaswag"]["command"]
    assert hs[hs.index("--hellaswag-tasks") + 1] == str(man["hellaswag"]["n"])
    mc = r["benchmarks"]["mmlu"]["command"]
    assert mc[mc.index("-bf") + 1].endswith("mmlu-test.bin")


def test_it_waits_while_a_board_run_holds_the_lock(box):
    lock = box["res"] / ".run.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text("4242")
    (lock / "submission").write_text("7")
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box, benchmarks=["mmlu"])
    assert w.once() is False
    r = result(box)
    assert r["status"] == "waiting" and r["line"] == "waiting for the run lock: run #7 holds it"
    assert (box["res"] / "gguf_requests" / "1.json").exists()
    assert (lock / "pid").read_text() == "4242"             # never taken from a board run
    for f in lock.iterdir():
        f.unlink()
    lock.rmdir()
    assert w.once() is True and result(box)["status"] == "done"


def test_the_time_limit_keeps_the_benchmarks_that_finished(box, monkeypatch):
    gd.build(box["res"] / "gguf_data", only=["hellaswag"], docs_of=docs_of(n_hs=400))
    monkeypatch.setenv("FAKE_PPL_DELAY", "0.01")
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box, benchmarks=["mmlu", "hellaswag", "winogrande"], time_limit_s=1.5)
    w.once()
    r = result(box)
    b = r["benchmarks"]
    assert b["mmlu"]["status"] == "done"
    assert b["hellaswag"]["status"] == "stopped" and 0 < b["hellaswag"]["done"] < 400
    assert b["winogrande"]["status"] == "not run"
    assert r["status"] == "stopped"
    assert r["line"] == "the time limit (0.000416667 h) was reached: 1 of 3 benchmarks finished " \
        "are kept"
    assert not (box["res"] / ".run.lock").exists()


def test_ctrl_c_keeps_the_benchmarks_that_finished(box, monkeypatch):
    gd.build(box["res"] / "gguf_data", only=["hellaswag"], docs_of=docs_of(n_hs=2000))
    request(box, benchmarks=["mmlu", "hellaswag"])
    env = {**os.environ, "FAKE_PPL_DELAY": "0.01",
           "PYTHONPATH": os.pathsep.join([str(HERE.parent / "scripts"), os.environ.get(
               "PYTHONPATH", "")])}
    proc = subprocess.Popen([sys.executable, str(HERE.parent / "scripts" / "gguf_worker.py"),
                             "--results", str(box["res"]), "--binary", box["bin"], "--poll",
                             "0.1", "--once"], env=env)
    t0 = time.time()
    while time.time() - t0 < 30:
        try:
            r = result(box)
        except (OSError, ValueError):
            r = {}
        if (r.get("benchmarks") or {}).get("hellaswag", {}).get("done"):
            break
        time.sleep(0.1)
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=30) == 0
    r = result(box)
    assert r["benchmarks"]["mmlu"]["status"] == "done"
    assert r["benchmarks"]["hellaswag"]["status"] == "stopped"
    assert r["status"] == "stopped" and r["line"].startswith("stopped by Ctrl+C: 1 of 2 ")
    assert not (box["res"] / ".run.lock").exists()


def test_a_changed_model_file_or_dataset_stops_the_job_in_one_line(box):
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box, rid="1", benchmarks=["mmlu"], pin={"sha256": "0" * 64})
    w.once()
    r = result(box, "1")
    assert r["status"] == "failed" and r["line"].startswith(
        "The GGUF file changed since it was registered: its sha256 is ")
    request(box, rid="2", benchmarks=["mmlu"])
    (box["res"] / "gguf_data" / "mmlu-test.bin").write_bytes(gd.mc_binary(
        [{"question": "q", "answers": ["a", "b"], "labels": [1, 0]}]))
    w.once()
    r = result(box, "2")
    assert r["status"] == "failed" and r["line"].startswith(
        "The MMLU dataset changed since this was queued: its sha256 is ")
    assert "\n" not in r["line"]


def test_a_run_that_prints_no_score_says_why(box, monkeypatch):
    monkeypatch.setenv("FAKE_PPL_FAIL", "1")
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box, benchmarks=["mmlu"])
    w.once()
    r = result(box)
    assert r["status"] == "failed"
    assert r["line"] == "MMLU: does not fit in the context window"


# ---------------------------------------------------------------------------
# what llama-perplexity prints, and the converter's files
# ---------------------------------------------------------------------------

def test_the_output_is_read_as_perplexity_cpp_prints_it():
    mc = ("multiple_choice_score: there are 14042 tasks in prompt\n\ntask\tacc_norm\n"
          "1\t100.00000000\n2\t50.00000000\n14042\t81.12000000\n\n"
          "Final result: 81.1200 +/- 0.3300\nRandom chance: 25.0000 +/- 0.3700\n")
    got = gb.parse("multiple-choice", mc)
    assert (got["done"], got["total"], got["final"]) == (14042, 14042, True)
    assert (got["acc"], got["se"], got["chance"]) == (0.8112, 0.0033, 0.25)
    hs = ("hellaswag_score : loaded 10042 tasks from prompt.\n"
          "hellaswag_score : selecting 10042 randomized tasks.\n"
          "\ntask\tacc_norm\t95% confidence interval\n"
          "1\t100.00000000%\t[20.6549%, 100.0000%]\n10042\t79.50000000%\t[78.7%, 80.3%]\n")
    got = gb.parse("hellaswag", hs)
    assert got["final"] and got["acc"] == 0.795 and got["done"] == 10042
    assert abs(got["se"] - (0.795 * 0.205 / 10041) ** 0.5) < 1e-9
    wg = ("winogrande_score : loaded 1267 tasks from prompt.\n"
          "1267\t75.1400\t -1.234567   -2.345678  1  1\n\n"
          "Final Winogrande score(1267 tasks): 75.1400 +/- 1.2100\n")
    got = gb.parse("winogrande", wg)
    assert (got["acc"], got["se"], got["done"], got["chance"]) == (0.7514, 0.0121, 1267, 0.5)
    # a run cut short: its progress, no score
    got = gb.parse("multiple-choice", mc.split("14042\t")[0])
    assert got["done"] == 2 and not got["final"]


def test_the_multiple_choice_binary_reads_back_as_llama_perplexity_reads_it():
    tasks = [{"question": "Q1?", "answers": ["a", "bb"], "labels": [0, 1]},
             {"question": "Qü 2", "answers": ["x", "y", "z"], "labels": [1, 0, 0]}]
    data = gd.mc_binary(tasks)
    (n,) = struct.unpack_from("<I", data, 0)
    assert n == 2
    back = gd.read_mc_binary(data)
    assert [(t["question"], t["answers"], t["labels"], t["mc2"]) for t in back] == \
        [(t["question"], t["answers"], t["labels"], []) for t in tasks]


def test_the_converters_questions_are_lm_evals(tmp_path):
    docs = docs_of()
    man = gd.build(tmp_path, docs_of=docs)["benchmarks"]
    mmlu = gd.read_mc_binary((tmp_path / "mmlu-test.bin").read_bytes())
    # the same questions, in lm_eval's order — 12f.5: as lm_eval's mmlu asks
    # them, lettered (test_12f5 has the whole prompt)
    assert [t["question"].split("\n\n")[1].split("\n")[0] for t in mmlu] == \
        [d["question"] for d in docs("mmlu")]
    assert all(t["labels"] == [0, 1, 0, 0] and t["answers"] == ["A", "B", "C", "D"] for t in mmlu)
    assert (man["mmlu"]["n"], man["mmlu"]["skipped"], man["mmlu"]["of"]) == (6, 0, 6)
    arc = gd.read_mc_binary((tmp_path / "arc-challenge-test.bin").read_bytes())
    assert arc[0]["answers"] == ["cat", "fish", "ant"] and arc[0]["labels"] == [1, 0, 0]
    # the one it can't hold, counted
    assert (man["arc_challenge"]["n"], man["arc_challenge"]["skipped"]) == (4, 1)
    tqa = gd.read_mc_binary((tmp_path / "truthfulqa-mc1-validation.bin").read_bytes())
    assert tqa[0]["labels"] == [1, 0]
    # six lines a HellaSwag task: the context, the gold index, four endings
    lines = (tmp_path / "hellaswag-validation.txt").read_text().rstrip("\n").split("\n")
    assert len(lines) == 6 * 12 and lines[:2] == ["Cooking: He cuts the onion 0", "0"]
    # Winogrande as its reader parses it, the quoted one kept whole, the last row kept
    wg = gd.read_winogrande((tmp_path / "winogrande-validation.csv").read_text())
    assert [r["sentence"] for r in wg] == [d["sentence"] for d in docs("winogrande")[:2]]
    assert man["winogrande"]["skipped"] == 1
    assert all(len(v["sha256"]) == 64 for v in man.values())


def test_the_converter_matches_lm_evals_own_documents():
    """on the server image, lm_eval's task gives the documents it converts"""
    pytest.importorskip("lm_eval")
    docs = gd.lm_eval_docs("arc_easy")
    assert len(docs) == gb.BENCHMARKS["arc_easy"]["n"]


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    model = tmp_path / "orig.gguf"
    model.write_bytes(b"GGUF" + b"\1" * 500)
    monkeypatch.setattr(gw, "_stop", {"why": ""})
    yield client, appmod, {"model": model, "bin": fake_binary(tmp_path)}
    client.__exit__(None, None, None)


def test_a_gguf_is_registered_measured_and_shown_in_its_own_columns(svc, monkeypatch):
    client, appmod, box = svc
    monkeypatch.setenv("FAKE_PPL_ACC", "0.8")
    before = client.get("/api/results").json()
    r = client.post("/api/gguf/models", json={"name": "Qwen3.6 original", "path": str(box["model"]),
                                              "based_on": "fx/good-750m", "how": "unsloth "
                                              "UD-Q4_K_XL", "by": "masein"})
    assert r.status_code == 200, r.text
    gid = r.json()["model"]["id"]
    assert gid == "gguf/Qwen3.6-original" and r.json()["model"]["flags"] == gb.DEFAULT_FLAGS
    # a row before any result: its page opens
    rows = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    assert rows[gid]["name"] == "Qwen3.6 original"
    # the worker isn't running: the job says so, with the command
    r = client.post("/api/gguf/runs", json={"model": gid, "by": "masein"})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert r.json()["worker"]["alive"] is False and "gguf_worker.py" in r.json()["worker"]["command"]
    row = next(x for x in client.get("/api/submissions").json() if x["id"] == sid)
    assert row["suite"] == "gguf" and row["status"] == "queued"
    assert row["progress"] == "waiting for the GGUF worker · The GGUF worker isn't running."
    # the board's own queue never takes it
    assert db.claim_next() is None
    # the host's worker measures it
    w = gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05)
    assert w.once() is True
    row = next(x for x in client.get("/api/submissions").json() if x["id"] == sid)
    assert row["status"] == "done" and row["progress"].startswith("MMLU 83.3 · HellaSwag 83.3")
    # its hash, pinned after the first job
    assert len(db.gguf_get(gid)["pin"]["sha256"]) == 64
    # its own column group; nothing in lm_eval's cells or averages moved
    appmod._cache.update(key=None, payload=None, at=0.0)
    j = client.get("/api/results").json()
    g = j["gguf"]["models"][gid]
    assert set(g) == set(gb.DEFAULT) and g["mmlu"]["full"] and abs(g["mmlu"]["v"] - 5 / 6) < 1e-6
    assert j["gguf"]["group"] == "Measured on the GGUF · llama.cpp, 0-shot"
    assert j["cells"] == before["cells"]
    for m in before["models"]:
        after = next(x for x in j["models"] if x["id"] == m["id"])
        assert (after["avg"], after["partialAvg"]) == (m["avg"], m["partialAvg"])
    row = next(m for m in j["models"] if m["id"] == gid)
    assert row["avg"] is None and not any(gid in c for c in j["cells"].values())
    assert [h["status"] for h in j["gguf"]["history"][gid]] == ["done"]


def test_the_heartbeat_says_when_the_worker_is_down(svc):
    w = gguf.worker()
    assert w["alive"] is False and w["line"] == "The GGUF worker isn't running."
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time()}))
    assert gguf.worker()["alive"] is True and gguf.worker()["line"] == ""
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time() - 90}))
    assert gguf.worker()["alive"] is False


def test_the_board_never_takes_a_live_workers_lock(svc):
    lock = runner.LOCK
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "pid").write_text("999999")            # a host pid this container can't see
    (lock / "owner").write_text("gguf-worker host 999999")
    (lock / "heartbeat").write_text("1")
    assert runner.acquire_lock(5) is False
    # a heartbeat two minutes old is a dead worker's: taken over
    old = time.time() - 300
    os.utime(lock / "heartbeat", (old, old))
    assert runner.acquire_lock(5) is True
    assert (lock / "submission").read_text() == "5"
    runner.release_lock()


def test_two_ggufs_of_one_base_are_compared_and_a_small_gap_is_not_a_clear_one(svc,
                                                                                monkeypatch):
    client, appmod, box = svc
    other = box["model"].with_name("lda.gguf")
    other.write_bytes(b"GGUF" + b"\2" * 500)
    ids = []
    for name, path, acc in (("orig", box["model"], "0.8"), ("lda", other, "0.8")):
        ids.append(client.post("/api/gguf/models", json={
            "name": name, "path": str(path), "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "x",
            "by": "masein"}).json()["model"]["id"])
        monkeypatch.setenv("FAKE_PPL_ACC", acc)
        client.post("/api/gguf/runs", json={"model": ids[-1], "benchmarks": ["mmlu"],
                                             "by": "masein"})
        gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05).once()
    client.get("/api/submissions")
    appmod._cache.update(key=None, payload=None, at=0.0)
    [pair] = client.get("/api/results").json()["gguf"]["pairs"]
    assert {pair["a"], pair["b"]} == set(ids)
    assert pair["by"]["mmlu"]["diff"] == 0 and pair["by"]["mmlu"]["clear"] is False


# ---------------------------------------------------------------------------
# the addendum: setups — environment variables and flags a run of their own
# ---------------------------------------------------------------------------

LOOK = "lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"


def test_setups_are_read_and_mtp_is_not_one():
    [s] = gguf.parse_setups(LOOK)
    assert s["name"] == "lookahead 1" and s["flags"] == []
    assert s["env"] == {"LLAMA_MOE_ROUTE_MODE": "lookahead", "LLAMA_MOE_ROUTE_LOOKAHEAD": "1"}
    assert s["id"] == gb.setup_id(s["env"], []) != gb.AS_BUILT["id"]
    two = gguf.parse_setups(LOOK + "\nbig batch: -b 4096")
    assert two[1]["flags"] == ["-b", "4096"] and two[1]["env"] == {}
    for bad, why in (("mtp 3: --mtp 3", gb.MTP_LINE), ("draft: -md x.gguf", gb.MTP_LINE),
                     ("nothing:", "nothing set"), ("x: lowercase=1", "neither KEY=VALUE"),
                     ("no colon here", "one a line"), ("p: PATH=/x", "the worker's own")):
        with pytest.raises(ValueError) as e:
            gguf.parse_setups(bad)
        assert why in str(e.value), (bad, str(e.value))


def test_the_worker_passes_a_setups_variables_and_flags(box):
    [s] = gguf.parse_setups("lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead "
                            "LLAMA_MOE_ROUTE_LOOKAHEAD=1 --no-warmup")
    w = gw.Worker(box["res"], box["bin"], poll=0.05)
    request(box, benchmarks=["mmlu"], setup=s)
    w.once()
    r = result(box)
    log = (box["res"] / "gguf_results" / "1.mmlu.log").read_text()
    assert "env LLAMA_MOE_ROUTE_MODE=lookahead" in log and "env LLAMA_MOE_ROUTE_LOOKAHEAD=1" in log
    cmd = r["benchmarks"]["mmlu"]["command"]
    assert cmd[cmd.index("--cpu-moe") + 1] == "--no-warmup"       # after the model's own flags
    assert r["setup"] == s and r["benchmarks"]["mmlu"]["env"] == s["env"]


def test_a_gguf_is_measured_in_each_setup_and_they_sit_side_by_side(svc, monkeypatch):
    client, appmod, box = svc
    monkeypatch.setenv("FAKE_PPL_ACC", "0.5")
    monkeypatch.setenv("FAKE_PPL_ACC_LOOKAHEAD", "0.8")
    r = client.post("/api/gguf/models", json={"name": "LDA", "path": str(box["model"]),
                                              "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                              "setups": LOOK, "by": "masein"})
    assert r.status_code == 200, r.text
    gid = r.json()["model"]["id"]
    assert [x["name"] for x in r.json()["model"]["setups"]] == ["as built", "lookahead 1"]
    # MTP is refused where setups are entered, in the one line
    bad = client.post("/api/gguf/models", json={"name": "x", "path": "/x.gguf", "how": "x",
                                                "setups": "mtp: --mtp 3", "by": "masein"})
    assert bad.status_code == 422 and bad.json()["detail"] == gb.MTP_LINE
    # one run a setup
    j = client.post("/api/gguf/runs", json={"model": gid, "benchmarks": ["mmlu", "arc_easy"],
                                            "by": "masein"}).json()
    assert len(j["ids"]) == 2
    rows = {x["id"]: x for x in client.get("/api/submissions").json()}
    assert [rows[i]["note"] for i in j["ids"]] == ["setup: as built", "setup: lookahead 1"]
    for _ in j["ids"]:
        gw.Worker(config.RESULTS_ROOT, box["bin"], poll=0.05).once()
    client.get("/api/submissions")
    appmod._cache.update(key=None, payload=None, at=0.0)
    g = client.get("/api/results").json()["gguf"]
    sets = {x["name"]: x for x in g["setups"][gid]}
    assert abs(sets["as built"]["benches"]["mmlu"]["v"] - 3 / 6) < 1e-6
    assert abs(sets["lookahead 1"]["benches"]["mmlu"]["v"] - 5 / 6) < 1e-6
    assert sets["lookahead 1"]["env"]["LLAMA_MOE_ROUTE_MODE"] == "lookahead"
    # Models shows it as built; the pairing puts the setup against as built
    assert g["models"][gid]["mmlu"]["setup"] == "as built"
    [pair] = [p for p in g["pairs"] if p["a"] == p["b"] == gid]
    assert (pair["setup_a"], pair["setup_b"]) == ("lookahead 1", "as built")
    assert abs(pair["by"]["mmlu"]["diff"] - 2 / 6) < 1e-6
    # each run's settings, pinned in its History
    hist = {h["setup"]["name"]: h for h in g["history"][gid]}
    assert hist["lookahead 1"]["setup"]["env"] == sets["lookahead 1"]["env"]
    # the setup's settings changed: another setup, and the old results say so
    client.post("/api/gguf/models", json={"name": "LDA", "path": str(box["model"]),
                                          "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                          "setups": LOOK.replace("=1", "=2"), "by": "masein"})
    appmod._cache.update(key=None, payload=None, at=0.0)
    g = client.get("/api/results").json()["gguf"]
    names = [(x["name"], x["current"], bool(x["benches"])) for x in g["setups"][gid]]
    assert names == [("as built", True, True), ("lookahead 1", True, False),
                     ("lookahead 1", False, True)]
