"""12q.C (the brief's): a DeviceMark run on its model's page. "Open results"
landed on a model page with nothing about DeviceMark. Now each model's runs
come with the results — by thinking mode, its full row as the board ranks it
(and its row on the On-device chart), its pilot with the cap-check line, its
MTP parity check with the pairs that differ, and its speed test with the
trials — and its answers have an endpoint: every item (public benchmark
items), no answer and wrong first, one bench or all. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import devicemark as dm
from conftest import make_service
from service import config
from test_12q_devicemark_board import HF, SERVED, _row

MTP, PLAIN = "served/DM-phone-build-MTP", "served/DM-phone-build"
NEMO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def _dir(out: Path, model: str, thinking: bool = False) -> Path:
    d = out / (model.replace("/", "__") + ("__thinking" if thinking else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(d: Path, name: str, obj) -> None:
    (d / name).write_text(json.dumps(obj) if not isinstance(obj, list)
                          else "".join(json.dumps(x) + "\n" for x in obj), encoding="utf-8")


PARITY = {"n": 50, "identical": 47, "same_answer": 49, "need": 48, "passes": True,
          "mtp": MTP, "plain": PLAIN, "thinking": False, "version": dm.VERSION, "at": 5.0,
          "items": [{"bench": "math", "key": "m1", "identical": False, "same_answer": False},
                    {"bench": "mmlu_pro", "key": "p1", "identical": False, "same_answer": True},
                    {"bench": "ifeval", "key": "i1", "identical": False, "same_answer": True},
                    {"bench": "math", "key": "m2", "identical": True, "same_answer": True}]}
SPEED = {"decode_tok_s": 31.5, "prompt_tokens": 128, "decode_tokens": 256,
         "label": "server: RTX 5090 + CPU experts", "at": 6.0,
         "trials": [{"warmup": True, "prompt_n": 128, "predicted_n": 256, "decode_tok_s": 29.0,
                     "prefill_tok_s": 900.0},
                    {"warmup": False, "prompt_n": 128, "predicted_n": 256, "decode_tok_s": 31.0,
                     "prefill_tok_s": 910.0},
                    {"warmup": False, "prompt_n": 128, "predicted_n": 256, "decode_tok_s": 32.0,
                     "prefill_tok_s": 905.0}]}
CAP = {"ok": True, "line": "thinking on, max_tokens 64: stopped at 64 tokens (length), all of it "
                           "thinking — the cap counts the thinking"}


def runs_here(out: Path) -> None:
    """an MTP setup with every kind of run, and its partner without MTP"""
    _row(out, MTP, {"ifeval": 0.7, "mmlu_pro": 0.65, "math": 0.6},
         {**SERVED, "phone": True, "name": "DM phone build, MTP"}, speed=31.5,
         device={"tok_s": 14.2, "device": "iPhone 17 Pro", "source": "measured by Sam, 29 Sep",
                 "at": 1.0})
    _row(out, MTP, {"ifeval": 0.5, "mmlu_pro": 0.7, "math": 0.8},
         {**SERVED, "phone": True, "name": "DM phone build, MTP"}, thinking=True)
    m = _dir(out, MTP)
    pilot = json.loads((m / dm.OUT_NAME).read_text())
    _write(m, dm.PILOT_NAME, {**pilot, "part": "pilot", "n": 30, "items": [{"bench": "math"}],
                              "cap_check": CAP})
    _write(m, dm.SPEED_NAME, {**SPEED, "setup": {}})
    for model, tag in ((MTP, "mtp"), (PLAIN, "plain")):
        d = _dir(out, model)
        _write(d, dm.PARITY_NAME, PARITY)
        _write(d, dm.PARITY_ANSWERS.format(tag), [
            {"bench": "math", "key": "m1", "gen_tokens": 90 if tag == "mtp" else 95,
             "text": "<think>so</think> \\boxed{2}" if tag == "mtp" else "\\boxed{3}"},
            {"bench": "mmlu_pro", "key": "p1", "gen_tokens": 40,
             "text": "The answer is (C)" if tag == "mtp" else "so \\boxed{C}"},
            {"bench": "ifeval", "key": "i1", "gen_tokens": 12, "text": "hello"}])


# ---------------------------------------------------------------------------
# a model's runs, as its page reads them
# ---------------------------------------------------------------------------

def test_each_models_runs_by_thinking_mode(tmp_path):
    out = tmp_path / "full"
    runs_here(out)
    got = dm.model_runs(out)
    assert set(got) == {MTP, PLAIN} and set(got[MTP]) == {"off", "on"}
    off = got[MTP]["off"]
    row = off["row"]
    assert row["composite"] == json.loads((_dir(out, MTP) / dm.OUT_NAME).read_text())["composite"]
    assert row["rank"] and row["chart_id"] == MTP and row["items"] is True
    assert row["label"] == "phone build · MTP · thinking off"
    assert (row["server_tok_s"], row["device"]["tok_s"]) == (31.5, 14.2)
    assert set(row["benches"]) == {"ifeval", "mmlu_pro", "math"}
    assert set(row["benches"]["math"]) == set(dm.BENCH_KEYS)
    # the pilot: its numbers and the cap check, never its items
    assert off["pilot"]["n"] == 30 and off["pilot"]["cap_check"] == CAP
    assert "items" not in off["pilot"]
    # the parity check: the counts, and each pair that differs with what each setup said
    par = off["parity"]
    assert (par["identical"], par["same_answer"], par["n"], par["passes"]) == (47, 49, 50, True)
    assert par["differ"] == [
        {"bench": "math", "key": "m1", "identical": False, "same_answer": False, "mtp": "2",
         "plain": "3", "mtp_tokens": 90, "plain_tokens": 95},
        {"bench": "mmlu_pro", "key": "p1", "identical": False, "same_answer": True, "mtp": "C",
         "plain": "C", "mtp_tokens": 40, "plain_tokens": 40},
        {"bench": "ifeval", "key": "i1", "identical": False, "same_answer": True, "mtp": None,
         "plain": None, "mtp_tokens": 12, "plain_tokens": 12}]
    # the speed test: the trials and the decode speed; the thinking card has none
    assert off["speed"]["decode_tok_s"] == 31.5 and len(off["speed"]["trials"]) == 3
    on = got[MTP]["on"]
    assert on["speed"] is None and on["pilot"] is None and on["row"]["chart_id"] == \
        f"{MTP} · thinking"
    # the setup without MTP: its quality from the MTP run, said so, and no answers of its own
    plain = got[PLAIN]["off"]
    assert plain["row"]["inherited"]["line"] == "quality from MTP run, parity 49/50"
    assert plain["row"]["items"] is False and plain["parity"]["mtp"] == MTP


def test_our_run_of_their_model_opens_their_row_on_the_chart(tmp_path):
    out = tmp_path / "full"
    _row(out, NEMO, {"ifeval": 0.6, "mmlu_pro": 0.6, "math": 0.8}, HF)
    row = dm.model_runs(out)[NEMO]["off"]["row"]
    assert (row["paired"], row["chart_id"], row["rank"]) == ("nemotron-4b__int8hu__aimodel",
                                                             "nemotron-4b__int8hu__aimodel", None)


def test_the_results_carry_them_and_refresh_when_a_run_lands(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    try:
        assert client.get("/api/results").json()["devicemark"] == {}
        runs_here(config.OUT_DIR)
        appmod._cache.update(at=0.0)                    # the five seconds, not the files
        got = client.get("/api/results").json()["devicemark"]
        assert set(got) == {MTP, PLAIN} and got[MTP]["off"]["speed"]["decode_tok_s"] == 31.5
        # a device speed entered later is a new file the board watches
        _write(_dir(config.OUT_DIR, MTP), dm.DEVICE_NAME,
               {"tok_s": 20.0, "device": "Pixel 10", "source": "measured by Sam", "at": 2.0})
        appmod._cache.update(at=0.0)
        got = client.get("/api/results").json()["devicemark"]
        assert got[MTP]["off"]["row"]["device"]["device"] == "Pixel 10"
    finally:
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# its answers
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"bench": "ifeval", "key": "i1", "prompt": "Write two lines about rain, all lowercase."},
    {"bench": "mmlu_pro", "key": "p1", "question": "Which is a prime?", "options": ["4", "6", "7"],
     "answer": "C", "category": "math"},
    {"bench": "mmlu_pro", "key": "p2", "question": "Which is even?", "options": ["3", "4"],
     "answer": "B", "category": "math"},
    {"bench": "math", "key": "test/algebra/1.json", "problem": "What is 1 + 1?", "answer": "2"},
    {"bench": "math", "key": "test/algebra/2.json", "problem": "What is 2 + 2?", "answer": "4"}]
ITEMS = [
    {"bench": "ifeval", "key": "i1", "text": "rain falls\nsoftly now", "answered": True,
     "correct": True, "gen_tokens": 8, "capped": False,
     "ifeval": {"strict": {"prompt": True, "inst": [True]}, "loose": {"prompt": True,
                                                                        "inst": [True]}}},
    {"bench": "mmlu_pro", "key": "p1", "text": "<think>7 is prime</think>The answer is (C)",
     "answered": True, "correct": True, "parsed": "C", "how": "the answer is (X)", "gold": "C",
     "gen_tokens": 30, "capped": False},
    {"bench": "mmlu_pro", "key": "p2", "text": "The answer is (A)", "answered": True,
     "correct": False, "parsed": "A", "how": "the answer is (X)", "gold": "B", "gen_tokens": 9,
     "capped": False},
    {"bench": "math", "key": "test/algebra/1.json", "text": "<think>" + "so " * 50,
     "answered": False, "correct": False, "parsed": None, "how": "", "gold": "2",
     "gen_tokens": 4096, "capped": True},
    {"bench": "math", "key": "test/algebra/2.json", "text": "\\boxed{4}", "answered": True,
     "correct": True, "parsed": "4", "how": "boxed", "gold": "4", "gen_tokens": 12,
     "capped": False}]


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.DM_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    config.DM_ITEMS.write_text(json.dumps({"version": dm.VERSION}) + "\n"
                               + "".join(json.dumps(q) + "\n" for q in QUESTIONS))
    _write(_dir(config.OUT_DIR, MTP), dm.ITEMS_NAME, ITEMS)
    yield client
    client.__exit__(None, None, None)


def test_every_answer_no_answer_and_wrong_first(svc):
    j = svc.get("/api/devicemark/answers", params={"model": MTP}).json()
    assert [(x["bench"], x["key"]) for x in j["items"]] == [
        ("math", "test/algebra/1.json"),                      # no answer
        ("mmlu_pro", "p2"),                                   # wrong
        ("ifeval", "i1"), ("mmlu_pro", "p1"), ("math", "test/algebra/2.json")]   # right
    assert j["total"] == 5 and j["counts"] == {
        "ifeval": {"n": 1, "no_answer": 0, "wrong": 0, "capped": 0},
        "mmlu_pro": {"n": 2, "no_answer": 0, "wrong": 1, "capped": 0},
        "math": {"n": 2, "no_answer": 1, "wrong": 0, "capped": 1}}
    first, wrong, right = j["items"][0], j["items"][1], j["items"][3]
    # ran out of room, still thinking: no output, the thinking apart
    assert (first["capped"], first["answered"], first["answer"], first["tokens"]) == (
        True, False, "", 4096)
    assert first["thinking"].startswith("so so") and first["q"] == "What is 1 + 1?"
    assert (wrong["parsed"], wrong["gold"], wrong["ok"]) == ("A", "B", False)
    assert wrong["options"] == ["3", "4"]
    # the output after the thinking, the thinking apart
    assert (right["answer"], right["thinking"]) == ("The answer is (C)", "7 is prime")
    ife = j["items"][2]
    assert ife["verdict"].startswith("strict: 1 of 1 instructions followed")


def test_one_bench_and_a_page_at_a_time(svc):
    j = svc.get("/api/devicemark/answers", params={"model": MTP, "bench": "math"}).json()
    assert [x["key"] for x in j["items"]] == ["test/algebra/1.json", "test/algebra/2.json"]
    assert j["total"] == 2 and j["counts"]["mmlu_pro"]["n"] == 2       # the counts are every bench's
    j = svc.get("/api/devicemark/answers", params={"model": MTP, "limit": 2}).json()
    assert len(j["items"]) == 2 and j["total"] == 5
    j = svc.get("/api/devicemark/answers", params={"model": MTP, "offset": 4}).json()
    assert [x["key"] for x in j["items"]] == ["test/algebra/2.json"]


def test_what_it_refuses(svc):
    assert svc.get("/api/devicemark/answers", params={"model": "a/b/c"}).status_code == 422
    # a name that looks like a way out is one folder's name ("..__etc"), and not there
    assert svc.get("/api/devicemark/answers", params={"model": "../etc"}).status_code == 404
    assert svc.get("/api/devicemark/answers",
                   params={"model": MTP, "bench": "gsm8k"}).status_code == 422
    r = svc.get("/api/devicemark/answers", params={"model": MTP, "thinking": True})
    assert r.status_code == 404 and "with thinking on" in r.json()["detail"]
    assert svc.get("/api/devicemark/answers", params={"model": "org/none"}).status_code == 404
