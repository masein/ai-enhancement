"""Close the loop: what did the training teach? Both halves, before and
after, and the sentence derived from them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import make_fixture
import report_lm_eval as report
from conftest import fresh, make_service
from service import llm_poller

REPO = Path(__file__).resolve().parents[1]


def test_the_sentence_on_every_outcome():
    # taught the test: diagnosis half up 20 points (5 SE), leaderboard half flat
    v, text, ratio = report.taint_verdict("mmlu", 0.004, 0.02, 0.20, 0.04)
    assert v == "test" and "taught the test" in text and "within noise" in text
    assert ratio == pytest.approx(50.0) and "50× the leaderboard-half change" in text
    # taught the skill: both up, both significant, same sign
    v, text, ratio = report.taint_verdict("mmlu", 0.15, 0.02, 0.17, 0.03)
    assert v == "skill" and "taught the skill" in text and "moved together" in text
    # nothing measurable
    v, text, _ = report.taint_verdict("mmlu", 0.005, 0.02, -0.01, 0.03)
    assert v == "none" and "changed nothing measurable" in text
    # the odd ones: leaderboard half moved alone, or the halves moved apart
    v, text, _ = report.taint_verdict("mmlu", 0.10, 0.02, 0.01, 0.03)
    assert v == "mixed" and "unusual pattern" in text
    v, _, _ = report.taint_verdict("mmlu", 0.10, 0.02, -0.12, 0.03)
    assert v == "mixed"
    # a diagnosis-half rise with a flat leaderboard half and NO change at all there
    v, text, ratio = report.taint_verdict("mmlu", 0.0, 0.02, 0.20, 0.04)
    assert v == "test" and ratio is None and "did not move at all" in text


def test_compare_needs_both_halves_of_both_models():
    t = {"tasks": {"mmlu": {"score_report": .5, "n_report": 100, "score_diagnose": .6,
                            "n_diagnose": 100, "categories": {}}}}
    assert report.taint_compare("mmlu", t, {"tasks": {}}, "p") is None
    assert report.taint_compare("mmlu", t, {"tasks": {"mmlu": {"score_report": .5,
                                                                "n_report": 100}}}, "p") is None
    c = report.taint_compare("mmlu", t, t, "p")
    assert c["verdict"] == "none" and c["dReport"] == 0 and c["parent"] == "p"


def test_fixture_children_show_both_outcomes(payload, tree, diag):
    test = next(m for m in payload["models"] if m["id"] == "fx/good-750m-tuned-test")
    skill = next(m for m in payload["models"] if m["id"] == "fx/good-750m-tuned-skill")
    parent = diag["fx/good-750m"]["tasks"]["mmlu"]
    for m in (test, skill):
        assert m["tainted"] == ["mmlu"] and m["official"] is False and m["avg"] is None
        c = m["taintCompare"]["mmlu"]
        assert c["parent"] == "fx/good-750m"
        assert c["before"]["report"]["v"] == parent["score_report"]
        assert c["before"]["diagnose"]["v"] == parent["score_diagnose"]
        assert c["before"]["report"]["n"] == parent["n_report"]
        assert c["after"]["report"]["v"] == diag[m["id"]]["tasks"]["mmlu"]["score_report"]
        assert set(c["categories"]) == set(parent["categories"])
        for name, v in c["categories"].items():
            assert v["before"]["score_report"] == parent["categories"][name]["score_report"]
            assert v["dDiagnose"] is not None
    ct, cs = test["taintCompare"]["mmlu"], skill["taintCompare"]["mmlu"]
    assert ct["verdict"] == "test" and "taught the test" in ct["text"]
    assert ct["dDiagnose"] > 0.1 and abs(ct["dReport"]) < 1.96 * ct["seReport"]
    assert ct["ratio"] is None or abs(ct["ratio"]) > 1.5
    assert cs["verdict"] == "skill" and "taught the skill" in cs["text"]
    assert cs["dReport"] > 0.1 and cs["dDiagnose"] > 0.1
    # by category, the half the training never saw moved for the skill child and not the other
    econ_t, econ_s = ct["categories"]["economics"], cs["categories"]["economics"]
    assert econ_s["dReport"] > econ_t["dReport"]
    assert any("taught the test, not the skill" in w for w in payload["warnings"])
    # the untainted parent carries no comparison
    assert next(m for m in payload["models"] if m["id"] == "fx/good-750m")["taintCompare"] is None


def test_diagnose_groups_and_categories_carry_both_halves(diag):
    for mid, d in diag.items():
        t = d["tasks"]["mmlu"]
        for g in t["groups"].values():
            assert g["n_report"] + g["n_diagnose"] == g["n"]
            assert g["score_diagnose"] is not None
        for c in t["categories"].values():
            assert c["n_report"] + c["n_diagnose"] == c["n"]
            assert c["score_diagnose"] is not None


def _ready_dataset(client):
    r = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": "economics",
                                            "requested_by": "t"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
    did = client.post(f"/api/proposals/{pid}/generate",
                      json={"requester": "Omar", "count": 10}).json()["dataset_id"]
    llm_poller.tick()
    assert client.get(f"/api/datasets/{did}").json()["status"] == "ready"
    return did


def test_join_dataset_to_run_to_checkpoint_to_model_picks_the_right_parent(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    try:
        did = _ready_dataset(client)
        # run A: explicit parent, checkpoint by event
        a = client.post("/api/truns", json={"name": "a", "datasets": [did],
                                            "parent": "fx/good-750m"}).json()["id"]
        client.post(f"/api/truns/{a}/event", json={"step": 1, "detail": "fx/good-750m-tuned-test"})
        # run B: a different parent, checkpoint by hf_prefix
        client.post("/api/truns", json={"name": "b", "datasets": [did],
                                            "parent": "fx/chance-160m",
                                            "hf_prefix": "fx/good-750m-tuned-skill"}).json()["id"]
        # run C: no parent field, but base_model in its config
        c = client.post("/api/truns", json={"name": "c", "datasets": [did],
                                            "config": {"base_model": "fx/good-750m"}}).json()["id"]
        client.post(f"/api/truns/{c}/event", json={"step": 1, "detail": "fx/skewed-360m"})
        # run D: a parent that is not on the board
        d = client.post("/api/truns", json={"name": "d", "datasets": [did],
                                            "parent": "org/never-evaluated"}).json()["id"]
        client.post(f"/api/truns/{d}/event", json={"step": 1, "detail": "fx/short-pick-410m"})
        # run E: no parent anywhere
        e = client.post("/api/truns", json={"name": "e", "datasets": [did]}).json()["id"]
        client.post(f"/api/truns/{e}/event", json={"step": 1, "detail": "fx/one-option-70m"})
        assert client.get(f"/api/truns/{a}").json()["run"]["parent"] == "fx/good-750m"
        fresh(appmod)
        rows = {m["id"]: m for m in client.get("/api/results").json()["models"]}
        ids = list(rows)
        # the join itself: each checkpoint finds the parent its own run recorded —
        # explicitly, by hf_prefix, or from the base_model in its config
        assert appmod.parents_for(ids) == {
            "fx/good-750m-tuned-test": "fx/good-750m",
            "fx/good-750m-tuned-skill": "fx/chance-160m",
            "fx/skewed-360m": "fx/good-750m",
            "fx/short-pick-410m": "org/never-evaluated"}
        # run E recorded no parent at all, so its checkpoint has none to find
        assert "fx/one-option-70m" not in appmod.parents_for(ids)
        # and every one of them is tainted on the topic the dataset came from
        tainted = appmod.taint_for(ids)
        assert set(tainted) == {"fx/good-750m-tuned-test", "fx/good-750m-tuned-skill",
                                "fx/skewed-360m", "fx/short-pick-410m", "fx/one-option-70m"}
        assert all(v == ["exam_economics"] for v in tainted.values())
        assert rows["fx/one-option-70m"]["tainted"] == ["exam_economics"]
        # the halves comparison is for a multiple-choice task; this dataset came
        # from an exam topic, so there is nothing for it to compare (C5 adds the
        # exam view of the same before/after)
        assert rows["fx/good-750m-tuned-test"]["taintCompare"] is None
        # the parents themselves are untouched
        for pid in ("fx/good-750m", "fx/chance-160m"):
            assert rows[pid]["tainted"] == [] and rows[pid]["taintCompare"] is None
    finally:
        client.__exit__(None, None, None)


def test_example_and_client_record_parent_and_dataset():
    src = (REPO / "examples" / "train_and_benchmark.py").read_text()
    assert "parent=args.base_model" in src and "datasets=[args.gap_dataset]" in src
    assert "--gap-dataset" in src and "--gap-ratio" in src and '"gap_ratio": args.gap_ratio' in src
    import bench_client
    import inspect
    sig = inspect.signature(bench_client.Bench.init)
    assert "parent" in sig.parameters and "datasets" in sig.parameters


def test_frozen_report_carries_the_comparison(tree):
    html = tree["report"].read_text(encoding="utf-8")
    blob = html.split('<script id="data" type="application/json">', 1)[1].split("</script>")[0]
    data = json.loads(blob.replace("<\\/", "</"))
    m = next(x for x in data["models"] if x["id"] == "fx/good-750m-tuned-test")
    assert m["taintCompare"]["mmlu"]["verdict"] == "test"
    assert make_fixture.TAINT == tree["taint"]
