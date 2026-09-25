"""12g.1 in the service: what a checkpoint was trained from, and the rule
that the Standard benchmarks are never a training target.

Trained from is set by a person from the models on the board, or recorded by
a training run that logged the checkpoint; a person's word wins. Improve's
Retests pair a checkpoint with it.

masein's decision (2026-09-25): the Knowledge exam (and Everyday tasks) are
what Improve trains toward; no Standard benchmark — IFEval, MMLU-Pro and
MATH-500 included — may reach a proposal, a generator request or a training
dataset. Asserted over the recorded request bodies and the dataset's items,
the way the split is asserted in test_gap.py."""

from __future__ import annotations

import json
import re

import pytest

import report_lm_eval as report
from conftest import fresh, make_service
from service import contamination as ct
from service import db, llm, llm_poller

BASE = "fx/good-750m"
CK = "fx/good-750m-tuned-skill"
TOPIC = "Economics"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, manifest = make_service(tmp_path, monkeypatch)
    yield client, appmod, manifest
    client.__exit__(None, None, None)


def row(client, appmod, mid):
    fresh(appmod)
    return next(m for m in client.get("/api/results").json()["models"] if m["id"] == mid)


def run_with(parent, prefix="", checkpoints=(), datasets=(), config=None):
    rid = db.trun_create("t", "p", "tester", json.dumps(config or {}), prefix,
                         datasets=list(datasets), parent=parent)
    for i, c in enumerate(checkpoints):
        db.trun_event(rid, i, "checkpoint", c)
    return rid


# ---------------------------------------------------------------------------
# Trained from
# ---------------------------------------------------------------------------

def test_a_person_sets_what_a_checkpoint_was_trained_from(svc):
    client, appmod, _ = svc
    assert "trainedFrom" not in row(client, appmod, CK)          # nothing links it today
    r = client.post("/api/trained-from", json={"model": CK, "base": BASE, "by": "masein"})
    assert r.status_code == 200 and r.json() == {"model": CK, "base": BASE, "by": "masein"}
    tf = row(client, appmod, CK)["trainedFrom"]
    assert (tf["base"], tf["source"], tf["by"]) == (BASE, "person", "masein")


@pytest.mark.parametrize("body,status,words", [
    ({"model": "fx/nope", "base": BASE, "by": "m"}, 404, "no such model on the board"),
    ({"model": CK, "base": "Qwen/Qwen3-0.6B", "by": "m"}, 422, "is not on the board"),
    ({"model": CK, "base": CK, "by": "m"}, 422, "not trained from itself"),
    ({"model": CK, "base": BASE, "by": "  "}, 422, "needs a name"),
])
def test_trained_from_is_a_model_on_the_board_and_a_name(svc, body, status, words):
    client, _, _ = svc
    r = client.post("/api/trained-from", json=body)
    assert r.status_code == status and words in r.json()["detail"]


def test_trained_from_makes_no_loop(svc):
    client, appmod, _ = svc
    assert client.post("/api/trained-from",
                       json={"model": CK, "base": BASE, "by": "m"}).status_code == 200
    r = client.post("/api/trained-from", json={"model": BASE, "base": CK, "by": "m"})
    assert r.status_code == 422
    assert r.json()["detail"] == f"{CK} was trained from {BASE}, so {BASE} cannot be trained from it"


def test_a_training_run_that_logged_the_checkpoint_fills_it(svc):
    client, appmod, _ = svc
    rid = run_with(BASE, checkpoints=[CK])
    tf = row(client, appmod, CK)["trainedFrom"]
    assert tf == {"base": BASE, "source": "run", "run": rid}
    # its config's base_model, when the run named no parent
    rid2 = run_with("", checkpoints=["fx/good-750m-tuned-test"],
                    config={"base_model": "fx/skewed-360m"})
    assert row(client, appmod, "fx/good-750m-tuned-test")["trainedFrom"]["run"] == rid2
    # …and a person's word wins over the run's
    client.post("/api/trained-from", json={"model": CK, "base": "fx/chance-160m", "by": "omar"})
    tf = row(client, appmod, CK)["trainedFrom"]
    assert (tf["base"], tf["source"], tf["by"]) == ("fx/chance-160m", "person", "omar")


def test_a_run_claims_by_prefix_only_up_to_a_separator():
    from service.app import _claims
    run = {"hf_prefix": "fx/good-750m-tuned", "checkpoints": []}
    assert _claims("fx/good-750m-tuned", run)
    assert _claims("fx/good-750m-tuned-skill", run)
    assert not _claims("fx/good-750m-tunedX", run)                # the taint join would say yes
    assert _claims("anything", {"hf_prefix": "", "checkpoints": ["anything"]})


def test_trained_from_is_the_before_of_what_the_training_taught(svc):
    """a tainted checkpoint's comparison uses the same link"""
    client, appmod, _ = svc
    client.post("/api/trained-from", json={"model": CK, "base": BASE, "by": "masein"})
    assert row(client, appmod, CK)["trainedFrom"]["base"] == BASE
    import service.app as app
    assert app.trained_from_for([CK])[CK]["base"] == BASE


# ---------------------------------------------------------------------------
# the rule: no Standard benchmark in any proposal, request or dataset
# ---------------------------------------------------------------------------

def standard_names(payload) -> set[str]:
    """every Standard benchmark, by its task name and the name the board shows"""
    tasks = set(payload["accTasks"]) | set(payload["pplTasks"]) | set(payload.get("genTasks", []))
    names = {t for t in tasks if not t.startswith(("exam_", "fr_", "everyday"))}
    shown = {"MMLU", "HellaSwag", "ARC-C", "ARC-E", "ARC", "Winogrande", "PIQA", "TruthfulQA",
             "GSM8K", "IFEval", "MMLU-Pro", "MATH-500", "MATH"}
    return names | shown


def mentions(text: str, names: set[str]) -> list[str]:
    return sorted(n for n in names
                  if re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", text, re.I))


def test_no_standard_benchmark_reaches_a_proposal_a_request_or_a_dataset(svc):
    client, appmod, tree = svc
    fake = llm.client()
    # a Standard benchmark is not a topic a proposal can be about
    for topic in ("mmlu", "MMLU", "ifeval", "hellaswag", "MMLU-Pro"):
        r = client.post("/api/proposals", json={"model": BASE, "topic": topic,
                                                "requested_by": "m"})
        assert r.status_code in (404, 422), (topic, r.status_code)
    # the whole loop, for an exam topic
    r = client.post("/api/proposals", json={"model": BASE, "topic": TOPIC, "requested_by": "m"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert llm_poller.tick() == 1
    p = client.get(f"/api/proposals/{pid}").json()
    assert "mmlu_caution" not in p["evidence"]          # MMLU no longer rides on a proposal
    assert client.post(f"/api/proposals/{pid}/approve",
                       json={"approver": "m", "edited_text": ""}).status_code == 200
    r = client.post(f"/api/proposals/{pid}/generate", json={"requester": "m", "count": 4})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]
    assert llm_poller.tick() == 1
    assert client.get(f"/api/datasets/{did}").json()["status"] == "ready"

    fresh(appmod)
    names = standard_names(client.get("/api/results").json())
    ix = ct.index(tree["out_dir"])                       # every benchmark item on disk
    bodies = fake.recorded()
    # the proposal's, the generator's — and the exam drafting's, which the
    # fixture's bank went through: every request this board sends an AI
    assert {"proposal", "gen"} <= {q["custom_id"].split(":")[0] for q in bodies}
    for q in bodies:
        body = q["system"] + "\n" + q["user"] + "\n" + json.dumps(q["meta"])
        assert mentions(body, names) == [], (q["custom_id"], mentions(body, names))
        # no 13 words of any benchmark item — for Improve's own requests; the
        # exam drafting's carry practice questions of the exam, which the
        # index holds too, by design
        if q["custom_id"].split(":")[0] in ("proposal", "gen"):
            assert ix.hits(q["system"] + "\n" + q["user"]) == [], q["custom_id"]
    items = client.get(f"/api/datasets/{did}/items.jsonl").text
    assert items.strip()
    for line in items.splitlines():
        text = json.dumps(json.loads(line))
        assert mentions(text, names) == []
        assert ix.hits(text) == []
    # and the proposal and dataset records, as stored
    stored = json.dumps(client.get(f"/api/proposals/{pid}").json()) \
        + json.dumps(client.get(f"/api/datasets/{did}").json())
    assert mentions(stored, names - {"MATH", "ARC"}) == []


def test_the_board_offers_no_standard_benchmark_to_propose_from(payload):
    """the gate that opens Propose exists only on exam topics"""
    std = set(payload["accTasks"]) | set(payload["pplTasks"])
    for m in payload["models"]:
        for t, v in ((m.get("judge") or {}).get("tasks") or {}).items():
            if v.get("propose") is not None:
                assert t.startswith("exam_") and t not in std, t
    assert report.GEN_TASKS and not any(t.startswith("exam_") for t in report.GEN_TASKS)
