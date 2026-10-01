"""12z A: wrong data and labels, from the 1 Oct QA walk of the live board.

A1. A served setup's lookahead and MTP come from its launch — its flags and
    environment, the GGUF setup it serves the same file as, the server saying
    its slots draft tokens — never from words in "How it's served": "routing
    local (no lookahead)" is not lookahead, "MTP-GGUF" in a file's name is not
    MTP. A row already on disk is labelled by its setup's launch as registered
    now.
A3. DeviceMark scores its rows' quality on a Mac: "int8, scored on a Mac;
    speed on iPhone 17 Pro".
A4. The plain phone build (no MTP, no lookahead) has its retention over the
    original.
A5. Our hf run's card carries DeviceMark's own row beside it.
A6. The Runs list pages back past the newest 100.
A7. A GGUF run's time is estimated at this server's measured pace, and the
    running line says the whole run's time left.
A8. An out-of-memory error says which process grew (12q.G, #129): pinned in
    tests/test_12q_g_hf_length_batch.py, and checked here once more.

Fake llama-servers and stand-ins: no model runs."""

from __future__ import annotations

import json
import time

import pytest

import devicemark as dm
import gguf_bench as gb
from fake_openai import FakeServer
from service import config, db, gguf, runner, served
from service import devicemark as sdm
from test_12q_devicemark_board import HF, SERVED, _row
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_12q_e_hf_pairs import _medians

ACC = {"ifeval": 0.8, "mmlu_pro": 0.7, "math": 0.6}
ORIG_ACC = {"ifeval": 0.8, "mmlu_pro": 0.7, "math": 0.6}
# the five setups as registered on the server, by port: what each "How it's served"
# says in words, and what its launch is
FIVE = [
    ("8090", "Qwen3.6-35B-A3B k4-LDA phone build", True,
     "llama.cpp fork, routing local (no lookahead)", "-ngl 99 --n-cpu-moe 21",
     "LLAMA_MOE_ROUTE_MODE=local", "phone build (k4-LDA) · thinking off"),
    ("8091", "Qwen3.6-35B-A3B Q4 original k=8", False,
     "unsloth MTP-GGUF, k=8 as built", "-ngl 99 --n-cpu-moe 21", "",
     "original (k=8) · thinking off"),
    ("8092", "Qwen3.6-35B-A3B k4-LDA lookahead", True, "llama.cpp fork, lookahead 1",
     "-ngl 99 --n-cpu-moe 21", "LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1",
     "phone build (k4-LDA) · lookahead · thinking off"),
    ("8094", "Qwen3.6-35B-A3B k4-LDA MTP", True, "llama.cpp fork, MTP 3",
     "-ngl 99 --n-cpu-moe 21 --spec-type draft-mtp", "LLAMA_MOE_ROUTE_MODE=local",
     "phone build (k4-LDA) · MTP · thinking off"),
    ("8096", "Qwen3.6-35B-A3B k4-LDA lookahead MTP", True, "llama.cpp fork, lookahead and MTP",
     "-ngl 99 --n-cpu-moe 21 --spec-type=draft-mtp",
     "LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1",
     "phone build (k4-LDA) · MTP · lookahead · thinking off")]


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, lookahead, mtp, said", [
    ("llama.cpp fork, routing local (no lookahead)", False, False, False),
    ("unsloth MTP-GGUF", False, False, False),
    ("lookahead 1, MTP 3, speculative", False, False, False),
    ("LLAMA_MOE_ROUTE_MODE=local", False, False, True),
    ("LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1", True, False, True),
    ("(LLAMA_MOE_ROUTE_MODE=lookahead)", True, False, True),
    ("LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=0", False, False, True),
    ("-ngl 99 --spec-type draft-mtp", False, True, True),
    ("--spec-type=draft-mtp", False, True, True),
    ("--spec-type ngram", False, False, True),
])
def test_lookahead_and_mtp_are_read_from_flags_and_environment_never_words(text, lookahead,
                                                                           mtp, said):
    assert dm.launch_of(text) == {"lookahead": lookahead, "mtp": mtp, "said": said}


def _five(out, fake: FakeServer) -> dict[str, str]:
    """the five setups registered with their launch, and a row each as a run
    recorded it before 12z — its lookahead and MTP guessed from the words. The
    original's file is unsloth's, "MTP-GGUF" in its name"""
    ids = {}
    orig = FakeServer()
    orig.model_path = "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP-GGUF.gguf"
    fake.closers = getattr(fake, "closers", []) + [orig]
    for port, name, phone, how, flags, env, _ in FIVE:
        base = orig.base if port == "8091" else fake.base
        rec = served.register({"name": name, "base_url": base, "how": how, "flags": flags,
                               "env": env, "thinking": "off", "phone": phone,
                               "based_on": "Qwen/Qwen3.6-35B-A3B"}, ME)
        ids[port] = rec["id"]
        guessed = {"lookahead": "lookahead" in (name + how).lower(),
                   "mtp": "mtp" in (name + how).lower()}
        _row(out, rec["id"], ORIG_ACC if port == "8091" else ACC,
             {**SERVED, **guessed, "phone": phone, "name": name, "server_flags": how})
    return ids


def test_the_five_registered_setups_get_the_right_labels(svc):  # noqa: F811
    fake = FakeServer()
    try:
        ids = _five(config.OUT_DIR, fake)
        rows = {r["id"]: r for r in svc.get("/api/devicemark").json()["rows"]}
        for port, *_, label in FIVE:
            assert rows[ids[port]]["label"] == label, port
        # and the model page's runs, which "Its scores" reads
        runs = sdm.dm().model_runs(config.OUT_DIR, served.launch_of_id)
        assert runs[ids["8090"]]["off"]["row"]["label"] == FIVE[0][-1]
        # A4: the plain phone build has its retention over the original now
        assert rows[ids["8090"]]["retention"] == {"of": ids["8091"], "ifeval": 1.0,
                                                  "mmlu_pro": 1.0, "math": 1.0}
        # lookahead never has one
        assert rows[ids["8092"]]["retention"] is None
    finally:
        for f in [fake, *getattr(fake, "closers", [])]:
            f.close()


def test_a_new_run_records_its_setup_from_its_launch(svc):  # noqa: F811
    fake = FakeServer()
    try:
        ids = _five(config.OUT_DIR, fake)
        for port, *_, label in FIVE:
            su = sdm.setup_of(served.get(ids[port]), False)
            assert dm.setup_label(su, False) == label, port
            assert su["launch"]["flags"].startswith("-ngl 99")
    finally:
        for f in [fake, *getattr(fake, "closers", [])]:
            f.close()


def test_the_launch_is_set_without_its_server_and_the_label_follows(svc):  # noqa: F811
    fake = FakeServer()
    try:
        ids = _five(config.OUT_DIR, fake)
        fake.close()                               # only one fits on the card: it's down now
        r = svc.put(f"/api/served/{ids['8090']}/launch",
                    json={"flags": "-ngl 99 --spec-type draft-mtp", "env": ""})
        assert r.status_code == 200, r.text
        assert r.json()["launch"]["mtp"] is True
        rows = {x["id"]: x for x in svc.get("/api/devicemark").json()["rows"]}
        assert rows[ids["8090"]]["label"] == "phone build (k4-LDA) · MTP · thinking off"
        assert svc.put("/api/served/served/nobody/launch", json={}).status_code == 404
    finally:
        for f in [fake, *getattr(fake, "closers", [])]:
            f.close()


def test_the_gguf_setup_it_serves_and_the_servers_slots_count_too(svc, monkeypatch):  # noqa: F811
    rec = {"id": "served/x", "name": "x", "how": "llama.cpp fork", "base_url": "http://h:1/v1",
           "same_as": {"gguf": "gguf/x", "setup": "s1"}, "pin": {}}
    monkeypatch.setattr(db, "gguf_get", lambda gid: {"setups": [
        {"id": "s1", "name": "lookahead 1",
         "env": {"LLAMA_MOE_ROUTE_MODE": "lookahead", "LLAMA_MOE_ROUTE_LOOKAHEAD": "1"},
         "flags": []}]})
    assert served.launch(rec)["lookahead"] is True and served.launch(rec)["mtp"] is False
    # a server whose slots draft tokens is drafting, whatever the words say
    assert served.launch({**rec, "speculative": True})["mtp"] is True
    # a model from OpenRouter has no launch
    assert served.launch({**rec, "via": served.OPENROUTER}) is None


# ---------------------------------------------------------------------------
# A3, A5
# ---------------------------------------------------------------------------

def test_their_rows_say_where_they_were_scored():
    rows = {r["id"]: r for r in dm.external_rows()}
    assert rows["lfm2.5-1.2b__int8hu__aimodel"]["label"] == \
        "int8, scored on a Mac; speed on iPhone 17 Pro"
    assert rows["gemma-4-e2b__int4__litertlm"]["label"] == \
        "int4, scored on a Mac; speed on iPhone 17 Pro"


def test_our_hf_runs_card_carries_their_row_and_the_modes_note(tmp_path):
    out = tmp_path / "full"
    _row(out, "Qwen/Qwen3.5-4B", {"ifeval": 0.3, "mmlu_pro": 0.2, "math": 0.3}, HF)
    _medians(out, "Qwen/Qwen3.5-4B", 400)                   # theirs 3,917: another mode
    card = dm.model_runs(out)["Qwen/Qwen3.5-4B"]["off"]["row"]
    theirs = {r["id"]: r for r in dm.external_rows()}["qwen3.5-4b__int8hu__aimodel"]
    assert card["paired"] == "qwen3.5-4b__int8hu__aimodel"
    assert card["theirs"]["composite"] == theirs["composite"]
    assert card["theirs"]["label"] == "int8, scored on a Mac; speed on iPhone 17 Pro"
    assert card["theirs"]["mode_differs"].startswith("the modes differ: ours answered")
    assert card["theirs"]["calibration"] is False


# ---------------------------------------------------------------------------
# A6
# ---------------------------------------------------------------------------

def test_the_runs_list_pages_back_past_the_newest_100(svc):  # noqa: F811
    for i in range(130):
        db.add(f"org/m{i}", "base", "quick", ME, "")
    newest = svc.get("/api/submissions?limit=100").json()
    assert len(newest) == 100
    assert svc.get("/api/submissions/count").json() == {"total": 130}
    older = svc.get(f"/api/submissions?limit=100&before={min(r['id'] for r in newest)}").json()
    assert len(older) == 30 and max(r["id"] for r in older) < min(r["id"] for r in newest)
    assert {r["id"] for r in newest} | {r["id"] for r in older} == set(range(1, 131))


# ---------------------------------------------------------------------------
# A7
# ---------------------------------------------------------------------------

def _result(sid: int, sha: str, benches: dict, status: str = "done") -> None:
    d = config.RESULTS_ROOT / "gguf_results"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(json.dumps({
        "id": str(sid), "sid": sid, "model": "gguf/other", "status": status,
        "file": {"name": "x.gguf", "sha256": sha}, "finished_at": time.time(),
        "benchmarks": benches}))


def test_a_benchmark_new_to_a_file_is_estimated_at_this_servers_pace(svc, monkeypatch):  # noqa: F811
    # another file measured ARC-C at 0.7 s a task (85 a minute): ARC-E, new to this
    # file and to this server, goes at ARC-C's pace — a benchmark of the same kind
    monkeypatch.setattr(gguf, "model", lambda mid: {"pin": {"sha256": "a" * 64}} if mid == "gguf/new"
                        else None)
    _result(7, "b" * 64, {"arc_challenge": {"status": "done", "seconds": 820.4, "done": 1172}})
    est = gguf.estimate("gguf/new", ["arc_challenge", "arc_easy", "truthfulqa"])
    assert est["by"]["arc_challenge"]["from"] == "this server"
    assert est["by"]["arc_easy"]["each"] == pytest.approx(0.7, abs=0.001)
    assert est["line"].endswith(", at the pace of this server's runs of other files")
    assert est["seconds"] == pytest.approx((1172 + 2376 + 817) * 0.7, rel=0.01)
    # this file's own pace wins once it has one
    _result(8, "a" * 64, {"arc_easy": {"status": "done", "seconds": 1188.0, "done": 2376}})
    est = gguf.estimate("gguf/new", ["arc_easy"])
    assert (est["by"]["arc_easy"]["from"], est["by"]["arc_easy"]["each"]) == ("this file", 0.5)
    assert not est["line"].endswith("other files") and not est["rough"]
    # nothing measured on this server: the rough guess, 85 a minute now
    monkeypatch.setattr(gguf, "results", lambda: [])
    est = gguf.estimate("gguf/new", ["arc_challenge"])
    assert est["rough"] and est["by"]["arc_challenge"]["each"] == gguf.GUESS_S["multiple-choice"] \
        == 0.7


def test_the_running_line_says_the_whole_runs_time_left(svc, monkeypatch):  # noqa: F811
    sid = db.add("gguf/new", "instruct", "gguf", ME, "setup: as built",
                 tasks=["arc_challenge", "arc_easy", "truthfulqa"])
    db.update(sid, status="running")
    monkeypatch.setattr(gguf, "model", lambda mid: {"pin": {"sha256": "a" * 64}})
    res = {"id": str(sid), "sid": sid, "model": "gguf/new", "status": "running",
           "line": "ARC-C: 1,100 of 1,172 · about 1 min left",
           "benchmarks": {"arc_challenge": {"status": "running", "done": 1100, "total": 1172,
                                            "secs_each": 0.7},
                          "arc_easy": {"status": "queued"}, "truthfulqa": {"status": "queued"}}}
    d = config.RESULTS_ROOT / "gguf_results"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(json.dumps(res))
    # the worker's heartbeat, on this job
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps(
        {"at": time.time(), "state": f"running {sid}"}))
    gguf.sync()
    left = (72 + 2376 + 817) * 0.7
    assert db.get(sid)["progress"] == (f"ARC-C: 1,100 of 1,172 · about 1 min left · "
                                       f"{gguf._dur(left)} left in all")
    # the last benchmark: its own line is the run's
    res["benchmarks"] = {"truthfulqa": {"status": "running", "done": 800, "total": 817,
                                        "secs_each": 0.7}}
    res["line"] = "TruthfulQA: 800 of 817 · about 1 min left"
    (d / f"{sid}.json").write_text(json.dumps(res))
    gguf.sync()
    assert db.get(sid)["progress"] == "TruthfulQA: 800 of 817 · about 1 min left"


# ---------------------------------------------------------------------------
# A8
# ---------------------------------------------------------------------------

def test_an_out_of_memory_error_says_which_process_grew():
    o = runner.oom_said("CUDA out of memory. Tried to allocate 6.00 GiB. GPU 0 has a total capacity "
                        "of 31.36 GiB of which 2.37 GiB is free. Process 4121 has 12.90 GiB memory "
                        "in use. Including non-PyTorch memory, this process has 15.90 GiB memory "
                        "in use.")
    assert runner.oom_line(o, int(12.9 * 1024)).endswith(
        "as when the task started: the run grew, not the card")
    assert "busier" not in runner._FRIENDLY[0][1]
    assert gb.BENCHMARKS                       # the GGUF table is imported as the worker's
