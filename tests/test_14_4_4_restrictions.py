"""14.4.4: a restricted set says what its questions and scores may be used for,
the same way wherever it appears. Driven by each set's manifest (`licence`,
`restriction`, `covers`): the full Mobile-MMLU "Non-commercial", Mobile-MMLU-Pro
"Internal use", and Artificial Analysis's numbers "Internal only" in the board's
own words. The API and every export carry the licence and restriction (a CSV as
columns); an export that leaves the server never holds a non-commercial set;
and a restricted set is never a training target, a few-shot example or a
source of questions. Invented rows, picks written in; nothing calls anyone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mobile_mmlu as mmp
import restrictions
from conftest import make_service
from service import config, db, runner
from test_14_3_mobile_mmlu import RIGHT, put_invented, sit_picks
from test_14_4_1_mobile_mmlu_data import put_full
from test_14_4_3_mobile_mmlu_full_run import pool_key, right, sit_full

NC = ("CC BY-NC-ND 4.0. For internal research only. Don't use these questions or scores in "
      "anything commercial: product claims, marketing, sales material or customer reports. "
      "Don't publish the questions or our answer key.")
IU = "CC BY-ND 4.0. Its scores may be used. The questions and our answer key must not be published."
AA = ("Artificial Analysis’s free data is for internal use: this board is on the tailnet, and its "
      "numbers are never in the single-file report.")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    put_invented(monkeypatch, config.MMP_DIR)
    put_full(monkeypatch, config.MMF_DIR)
    yield client, appmod
    client.__exit__(None, None, None)


def sat(model: str = "org/both-2b") -> Path:
    """a model that sat the full set — so Pro's score too"""
    d = config.OUT_DIR / model.replace("/", "__")
    sit_full(d, {q["lid"]: right(q) for q in mmp.pool()})
    mmp.collect(d)
    return d


def results(client, appmod) -> dict:
    appmod._cache.update(key=None, payload=None, at=0.0)
    return client.get("/api/results").json()


# ---------------------------------------------------------------------------
# what each says, from its manifest
# ---------------------------------------------------------------------------

def test_each_sets_badge_and_sentence_come_from_its_manifest():
    got = restrictions.sets()
    full, pro = got["mobile_mmlu_full"], got["mobile_mmlu_pro"]
    assert (full["name"], full["licence"], full["restriction"], full["badge"]) == (
        "Mobile-MMLU (full)", "CC BY-NC-ND 4.0", "non-commercial", "Non-commercial")
    assert full["sentence"] == NC
    assert (pro["name"], pro["restriction"], pro["badge"], pro["sentence"]) == (
        "Mobile-MMLU-Pro", "internal-use", "Internal use", IU)
    for f in ("mobile_mmlu", "mobile_mmlu_pro"):
        m = json.loads((restrictions.MANIFESTS / f / "manifest.json").read_text())
        assert m["restriction"] and m["covers"] and m["board_name"]


def test_any_manifest_with_a_restriction_is_a_restricted_set(tmp_path, monkeypatch):
    """general: a new set needs only its manifest's three fields"""
    (tmp_path / "newset").mkdir()
    (tmp_path / "newset" / "manifest.json").write_text(json.dumps({
        "name": "A new set", "board_name": "New set", "licence": "CC BY-NC 4.0",
        "restriction": "non-commercial", "covers": ["new_set", "new_set_cot"]}))
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "manifest.json").write_text(json.dumps({"name": "No restriction"}))
    monkeypatch.setattr(restrictions, "MANIFESTS", tmp_path)
    got = restrictions.sets()
    assert set(got) == {"new_set", "new_set_cot"} and got["new_set"]["badge"] == "Non-commercial"
    assert got["new_set"]["sentence"].startswith("CC BY-NC 4.0. For internal research only.")
    with pytest.raises(ValueError, match="unknown restriction"):
        restrictions.entry("x", "X", "", "secret")


def test_artificial_analysis_is_internal_only_in_the_boards_own_words(svc):
    client, _ = svc
    r = client.get("/api/reported").json()["restrictions"]["aa"]
    assert (r["badge"], r["restriction"], r["sentence"]) == ("Internal only", "internal-only", AA)
    assert set(client.get("/api/reported").json()["restrictions"]) == {"aa"}


# ---------------------------------------------------------------------------
# the API carries it
# ---------------------------------------------------------------------------

def test_the_api_carries_licence_and_restriction(svc):
    client, appmod = svc
    pool_key()
    sat()
    data = results(client, appmod)
    assert data["restrictions"]["mobile_mmlu_full"]["sentence"] == NC
    assert data["restrictions"]["mobile_mmlu_pro"]["badge"] == "Internal use"
    assert (data["mmp"]["licence"], data["mmp"]["restriction"]) == ("CC BY-ND 4.0", "internal-use")
    assert (data["mmf"]["licence"], data["mmf"]["restriction"]) == ("CC BY-NC-ND 4.0",
                                                                    "non-commercial")
    est = client.get("/api/mobileaibench/estimate", params={"model": "org/x"}).json()["parts"]
    assert est["mmlu_full"]["restriction"]["badge"] == "Non-commercial"
    assert est["mmlu"]["restriction"]["badge"] == "Internal use"
    assert "restriction" not in est["none"]
    key = client.get("/api/mobile-mmlu/key").json()
    assert set(key["restrictions"]) == {"mobile_mmlu_pro", "mobile_mmlu_full"}
    gg = client.get("/api/gguf").json()
    assert gg["restrictions"]["mobile_mmlu_full"]["badge"] == "Non-commercial"
    # the queue's reply says so as the run is queued
    r = client.post("/api/submissions", json={"hf_id": "org/c", "suite": "mobile",
                                              "part": "mmlu_full", "kind": "instruct"}).json()
    assert r["restriction"]["sentence"] == NC
    r = client.post("/api/submissions", json={"hf_id": "org/c", "suite": "mobile",
                                              "kind": "instruct"}).json()
    assert "restriction" not in r


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_the_command_lines_csv_carries_licence_and_restriction_as_columns(svc, tmp_path,
                                                                          monkeypatch):
    import report_lm_eval as rep
    pool_key()
    sit_picks(config.OUT_DIR / "org__pro-1b", RIGHT)
    mmp.collect(config.OUT_DIR / "org__pro-1b")
    out = tmp_path / "r.csv"
    monkeypatch.setattr("sys.argv", ["report_lm_eval.py", str(config.OUT_DIR), "--out",
                                     str(tmp_path / "r.html"), "--csv", str(out)])
    assert rep.main() == 0
    rows = out.read_text().splitlines()
    head = rows[0].split(",")
    assert head[-2:] == ["licence", "restriction"]
    pro = [r for r in rows if ",mobile_mmlu_pro,key_acc," in r]
    assert pro and all(r.endswith(",CC BY-ND 4.0,Internal use") for r in pro)
    assert any(r.endswith(",,") for r in rows[1:])            # an unrestricted task's


def test_the_single_file_report_never_holds_a_non_commercial_set(svc, tmp_path):
    import report_lm_eval as rep
    pool_key()
    sat()
    runs = rep.load_results(config.OUT_DIR)
    out = rep.build_report(runs, tmp_path / "board.html", "board")
    html = out.read_text()
    blob = json.loads(html.split('<script id="data" type="application/json">', 1)[1]
                      .split("</script>")[0].replace("<\\/", "</"))
    assert blob["mmf"] is None and not any(m.get("mmf") for m in blob["models"])
    assert "mobile_mmlu_full" not in json.dumps({k: v for k, v in blob.items()
                                                 if k in ("cells", "extra", "restrictions")})
    # Pro's scores may be used: its cell stays, with its badge
    assert "org/both-2b" in blob["cells"]["mobile_mmlu_pro"]
    assert blob["restrictions"]["mobile_mmlu_pro"]["badge"] == "Internal use"


def test_for_export_strips_the_full_sets_gguf_cells_too():
    import report_lm_eval as rep
    cell = {"v": 0.5}
    payload = {"restrictions": restrictions.sets(), "mmf": {"name": "x"},
               "models": [{"id": "g", "mmf": {"acc": 0.5}}], "cells": {}, "extra": [],
               "gguf": {"order": ["mmlu", "mobile_mmlu_pro"],
                        "benchmarks": {"mmlu": {}, "mobile_mmlu_full": {}},
                        "models": {"g": {"mmlu": cell, "mobile_mmlu_full": cell}},
                        "setups": {"g": [{"benches": {"mobile_mmlu_full": cell}}]},
                        "history": {"g": [{"benchmarks": {"mobile_mmlu_full": cell}}]}}}
    out = rep.for_export(payload)
    assert "mobile_mmlu_full" not in json.dumps(out)
    assert out["gguf"]["models"]["g"] == {"mmlu": cell} and "mobile_mmlu_pro" in out["gguf"]["order"]


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_an_export_that_leaves_the_server_refuses_a_non_commercial_set(monkeypatch):
    import export_devicemark_raw as ex
    import remote_bundle as rb
    assert ex.refused(["dm_ifeval", "dm_mmlu_pro", "dm_math"]) == ""
    assert "Mobile-MMLU (full) is non-commercial: it never leaves this server" in ex.refused(
        ["mobile_mmlu_full"])
    assert ex.refused(["mobile_mmlu_pro"]) == ""           # internal use: its scores may go
    # no suite a bundle can carry holds one: checked here, where the manifests are (a
    # rented box has none of them)
    assert all(rb.refused(s) == "" for s in rb.SUITES)
    monkeypatch.setitem(rb.SUITES, "mmlu_full", {"prefix": "mmf", "tasks": ("mobile_mmlu_full",)})
    with pytest.raises(ValueError, match="never leaves this server"):
        rb.bundle_name("mmlu_full", "org/m", False)


def test_never_a_training_target(svc, monkeypatch):
    client, _ = svc
    for topic in ("mobile_mmlu_full", "mobile_mmlu_pro"):
        r = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": topic})
        assert r.status_code == 422, r.text
        assert "never a training target" in r.json()["detail"]
        assert r.json()["detail"].endswith("Nothing was proposed.")


def test_never_a_few_shot_example(svc, monkeypatch):
    client, _ = svc
    from test_14_4_3_mobile_mmlu_full_run import fake_lm_eval
    pool_key()
    seen = fake_lm_eval(monkeypatch)
    monkeypatch.setitem(config.NFEWSHOT, "mobile_mmlu_full", 5)
    sid = client.post("/api/submissions", json={"hf_id": "org/fs-2b", "suite": "mobile",
                                                "part": "mmlu_full", "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert not seen and db.get(sid)["status"] == "failed"
    log = (config.LOGS_DIR / f"service_{sid}_org__fs-2b.log").read_text()
    assert "never a training target, a few-shot example or a source of questions" in log


def test_never_a_source_of_questions(svc, monkeypatch):
    """the question builder refuses a drafted question that copies a benchmark's"""
    from service import builder
    full = mmp.load_full()
    q = next(x for x in full if len(x["question"].split()) >= 14)
    d = {"id": "d1", "spec": {"kind": "everyday", "dedup": False}, "items": [
        {"n": 1, "q": {"prompt": q["question"]}, "auto": False, "verdict": None, "flags": [],
         "sample": False},
        {"n": 2, "q": {"prompt": "Write a polite note asking a neighbour to move their car."},
         "auto": False, "verdict": None, "flags": [], "sample": False}]}
    monkeypatch.setattr(builder, "_text", lambda d, q: q["prompt"])
    builder._bench_gate(d)
    a, b = d["items"]
    assert a["verdict"] == "reject" and a["flags"][0]["text"] == builder.BENCH_COPY
    assert b["verdict"] is None and not b["flags"]


def test_its_questions_are_read_from_its_file_never_a_runs_answers(svc, monkeypatch):
    """16.8: listed from its own file — Non-commercial, with no right answer
    (the authors hold theirs back) and no model's result (a run's lm_eval
    target is a stand-in) — and refused while it is switched off"""
    client, _ = svc
    sat("org/m")
    from service import questions
    questions._ff.clear()
    got = client.get("/api/questions/mobile_mmlu_full", params={"limit": 200}).json()
    assert got["from_file"] is True and got["models"] == []
    assert got["rows"] and all(r["answer_idx"] is None and r["results"] == {}
                               for r in got["rows"])
    assert got["meta"]["licence"].startswith("CC BY-NC-ND 4.0 — Non-commercial")
    # the halves, as everywhere: part of the set is listed
    assert got["listed"] + got["other"] == len(mmp.load_full()) and got["other"] > 0
    monkeypatch.setattr(config, "MOBILE_MMLU_FULL", False)
    questions._ff.clear()
    r = client.get("/api/questions/mobile_mmlu_full")
    assert r.status_code == 403 and "switched off" in r.json()["detail"]
    assert "mobile_mmlu_full" not in questions.tasks()
