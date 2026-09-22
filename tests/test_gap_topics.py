"""C3: the gap comes from the judge's written assessments, not from MMLU.

The safety property this file exists for: a proposal request may carry the
judge's reasoning about DIAGNOSE-half answers, and no exam question text of
either half — the justification is about the answer, not the question.
"""

from __future__ import annotations

import json

import pytest

import diagnose as dx
import exam_build as eb
from conftest import assert_no_report_half_text, make_service
from service import llm, llm_poller
from service import proposals as prop

TOPIC = "Economics"
TASK = "exam_economics"


@pytest.fixture
def gap(tmp_path, monkeypatch):
    client, appmod, manifest = make_service(tmp_path, monkeypatch)
    yield client, appmod, manifest
    client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# reading the judge
# ---------------------------------------------------------------------------

def test_weak_topics_ranks_every_topic_from_the_report_half(tree):
    rows = prop.weak_topics(tree["models"]["fx/good-750m"]["dir"])
    assert rows and all(r["task"].startswith("exam_") for r in rows)
    assert {r["topic"] for r in rows} <= set(eb.TOPICS)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    scores = [r["score_report"] for r in rows if r["score_report"] is not None]
    assert scores == sorted(scores)                       # weakest first
    econ = next(r for r in rows if r["task"] == TASK)
    assert econ["n_report"] >= 30 and econ["n_diagnose"] > 0
    assert econ["of"] == len(rows) and econ["weak_diagnose"] > 0
    assert econ["answers"]["n"] == econ["n"] and econ["answers"]["distinct"] > 1
    # the control task is not a topic, and a model with no judge.json has none
    assert not any(r["task"] == eb.CONTROL_TASK for r in rows)
    assert prop.weak_topics(tree["models"][tree["nodiag"]]["dir"]) == []


def test_justifications_are_diagnose_half_and_below_the_line(tree):
    d = tree["models"]["fx/good-750m"]["dir"]
    items, counts = prop.justifications_for(d, TASK)
    assert items and counts["diagnose_weak"] <= counts["diagnose_items"]
    assert len(items) <= prop.MAX_JUSTIFICATIONS
    raw = json.loads((d / "judge.json").read_text())["tasks"][TASK]["items"]
    diagnose_qids = {it["qid"] for it in raw if it["half"] == "diagnose"}
    report_qids = {it["qid"] for it in raw if it["half"] == "report"}
    assert report_qids and diagnose_qids
    for it in items:
        assert it["qid"] in diagnose_qids and it["qid"] not in report_qids
        assert dx.split_of(it["qid"]) == "diagnose"       # the rule, from the hash itself
        assert it["score"] < prop.WEAK_SCORE and it["justification"]
    assert [it["qid"] for it in items] == sorted(it["qid"] for it in items)   # deterministic
    assert prop.justifications_for(tree["models"][tree["nodiag"]]["dir"], TASK)[0] == []


# ---------------------------------------------------------------------------
# the planted quotation
# ---------------------------------------------------------------------------

def test_strip_question_quotes():
    q = "Explain why a binding price ceiling produces a shortage rather than a surplus."
    grams = prop.question_ngrams([q])
    keep = "The answer names the mechanism but stops before the consequence."
    assert prop.strip_question_quotes(keep, grams) == keep
    quoted = ("The model restated the question — " + q + " — and then wrote nothing.")
    out = prop.strip_question_quotes(quoted, grams)
    assert prop.REDACTED in out
    for frag in ("binding price ceiling produces", "rather than a surplus"):
        assert frag not in out
    assert "The model restated the question" in out and "then wrote nothing" in out
    # five words are not a quotation; six are
    five = "binding price ceiling produces a"
    assert len(five.split()) == 5 and prop.strip_question_quotes(five, grams) == five
    six = "a binding price ceiling produces a"
    assert len(six.split()) == 6 and prop.strip_question_quotes(six, grams) == prop.REDACTED
    assert prop.strip_question_quotes("", grams) == ""
    assert prop.strip_question_quotes("anything at all here", set()) == "anything at all here"


def test_a_quoted_question_never_reaches_the_request(gap):
    """Plant both halves' question text in the judge's justifications and
    require that neither survives into the recorded request body."""
    client, _, tree = gap
    d = tree["models"]["fx/good-750m"]["dir"]
    bank = {b["qid"]: b["prompt"] for b in eb.load_bank(tree["judged"]["exam_root"])[TOPIC]}
    report_q = next(p for q, p in sorted(bank.items()) if eb.half_of(q) == "report")
    diagnose_q = next(p for q, p in sorted(bank.items()) if eb.half_of(q) == "diagnose")
    j = json.loads((d / "judge.json").read_text())
    weak = [it for it in j["tasks"][TASK]["items"]
            if it["half"] == "diagnose" and it["score"] < prop.WEAK_SCORE]
    assert len(weak) >= 2
    weak[0]["justification"] = f"The answer ignores the question, which asked: {diagnose_q}"
    weak[1]["justification"] = f"Compare this with the other item: {report_q} — it answered neither."
    (d / "judge.json").write_text(json.dumps(j, sort_keys=True))

    r = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": TOPIC,
                                             "requested_by": "tester"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    body = next(q["system"] + "\n" + q["user"] for q in llm.client().recorded()
                if q["custom_id"] == f"proposal:{pid}")
    for q in bank.values():
        assert q not in body
    for frag in (" ".join(report_q.split()[:6]), " ".join(diagnose_q.split()[:6])):
        assert frag not in body
    assert assert_no_report_half_text(body, eb.load_bank(tree["judged"]["exam_root"])[TOPIC])
    assert body.count(prop.REDACTED) == 2
    assert "The answer ignores the question" in body and "it answered neither" in body
    # and the reviewer sees the same redacted text, never the question
    ev = client.get(f"/api/proposals/{pid}").json()["evidence"]
    for e in ev["examples"]:
        assert report_q not in e["justification"] and diagnose_q not in e["justification"]


def test_the_request_carries_the_rubric_and_no_answers(gap):
    client, _, tree = gap
    pid = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": TOPIC,
                                              "requested_by": "t"}).json()["id"]
    body = next(q["system"] + "\n" + q["user"] for q in llm.client().recorded()
                if q["custom_id"] == f"proposal:{pid}")
    # the topic's own rubric when it has one — economics does now
    assert "# Economics Evaluation Rubric" in body and "Topic: Economics" in body
    assert "scored 0 of 4" in body or "scored 1 of 4" in body or "scored 2 of 4" in body
    # the model's own answers are not in the request: the judge's reading of
    # them is what the proposal is built from
    raw = json.loads((tree["models"]["fx/good-750m"]["dir"] / "judge.json").read_text())
    assert raw["tasks"][TASK]["items"]
    assert "In short, that is the answer." not in body      # the fixture's answer boilerplate
    assert "must not invent them" in body or "must not" in body


def test_the_proposal_records_its_judge_run(gap):
    client, _, _ = gap
    pid = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": TOPIC,
                                              "requested_by": "t"}).json()["id"]
    llm_poller.tick()
    p = client.get(f"/api/proposals/{pid}").json()
    jr = json.loads(p["judge_run"])
    assert jr["judge_id"] == "stub/overlap-v1" and jr["batch_id"] == "stub"
    assert len(jr["prompt_sha256"]) == 64
    # and it reaches provenance, even for a proposal that predates the column
    assert prop.judge_run_of({"judge_run": json.dumps(jr)}) == jr
    assert prop.judge_run_of({})["judge_id"] == "unrecorded"
    assert prop.provenance_complete({"identities": {}, "judge_run": prop.judge_run_of({})}) == [
        "identities"]
