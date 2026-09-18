"""The whole loop, once, against the fake backend — the definition of done.

    write the exam → curate it → sit it → judge it → find the weak topic →
    propose → approve → generate documents → gate → provenance → taint →
    before and after

Each stage asserts the artefact the next one needs, and the rule is checked
where it applies: no report-half question, of the benchmarks or of our own
exam, ever reaches a human or a model.
"""

from __future__ import annotations

import json

import pytest

import diagnose as dx
import exam_build as eb
import judge as jd
import report_lm_eval as report
from conftest import fresh, make_service
from service import llm, llm_poller
from service import proposals as prop

TOPIC = "economics"
TASK = "exam_economics"


def test_the_whole_loop(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    try:
        exam_root = tree["judged"]["exam_root"]

        # 1. WRITE THE EXAM — an LLM drafts, and drafts are not the bank
        before_bank = len(eb.load_bank(exam_root)[TOPIC])
        out = eb.draft(exam_root, llm.client("exam"), [TOPIC], per_topic=2, wait=True, poll_s=0)
        assert out["written"] == {TOPIC: 2}
        assert len(eb.load_bank(exam_root)[TOPIC]) == before_bank
        cands = client.get(f"/api/exam/candidates?topic={TOPIC}").json()
        assert len(cands) == 2

        # 2. CURATE — a person accepts, under a name, and the question is split
        r = client.post(f"/api/exam/candidates/{cands[0]['cid']}/accept",
                        json={"approver": "Omar"})
        assert r.status_code == 200 and r.json()["half"] in ("report", "diagnose")
        assert r.json()["half"] == dx.split_of(r.json()["qid"])     # the benchmarks' own split
        client.post(f"/api/exam/candidates/{cands[1]['cid']}/reject",
                    json={"approver": "Omar", "reason": "recall, not understanding"})
        assert len(eb.load_bank(exam_root)[TOPIC]) == before_bank + 1
        built = client.post("/api/exam/build").json()
        assert built["tasks"][TASK]["items"] == before_bank + 1

        # 3 & 4. SIT AND JUDGE — the fixture's models have sat it; the judge
        # records each item's half and what the model wrote
        j = json.loads((tree["models"]["fx/good-750m"]["dir"] / "judge.json").read_text())
        t = j["tasks"][TASK]
        assert t["n_report"] + t["n_diagnose"] == t["n"]
        assert all(it["half"] == dx.split_of(it["qid"]) for it in t["items"])
        assert j["canary"]["n"] == 30 and j["judge"]["id"] == "stub/overlap-v1"

        # 5. THE WEAK TOPIC — from the judge's own numbers, gated
        rows = prop.weak_topics(tree["models"]["fx/good-750m"]["dir"])
        assert rows[0]["rank"] == 1 and rows[0]["score_report"] is not None
        payload = client.get("/api/results").json()
        me = next(m for m in payload["models"] if m["id"] == "fx/good-750m")
        assert me["judgeState"]["ok"] and me["judge"]["tasks"][TASK]["propose"]["ok"]

        # 6. PROPOSE — the judge's words about diagnose-half answers, no questions
        pid = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": TOPIC,
                                                  "requested_by": "Omar"}).json()["id"]
        assert llm_poller.tick() == 1
        p = client.get(f"/api/proposals/{pid}").json()
        assert p["status"] == "proposed" and p["spec_text"]
        req = next(q for q in llm.client().recorded() if q["custom_id"] == f"proposal:{pid}")
        body = req["system"] + "\n" + req["user"]
        assert all(dx.split_of(q) == "diagnose" for q in req["meta"]["qids"])
        for b in eb.load_bank(exam_root)[TOPIC]:              # THE RULE, both halves
            assert b["prompt"] not in body
        assert json.loads(p["judge_run"])["judge_id"] == "stub/overlap-v1"

        # 7. APPROVE — the airlock: a name, and the text the generator will get
        edited = p["spec_text"] + " Emphasise direction of effect."
        assert client.post(f"/api/proposals/{pid}/approve",
                           json={"approver": "Omar", "edited_text": edited}).status_code == 200

        # 8. GENERATE DOCUMENTS — the spec and nothing else
        did = client.post(f"/api/proposals/{pid}/generate",
                          json={"requester": "Omar", "count": 6}).json()["dataset_id"]
        gen = [q for q in llm.client().recorded() if q["custom_id"].startswith(f"gen:{did}:")]
        assert len(gen) == 3 and all(q["meta"]["format"] == "doc" for q in gen)
        for q in gen:
            gbody = q["system"] + "\n" + q["user"]
            assert edited in gbody and "fx/good-750m" not in gbody
            for b in eb.load_bank(exam_root)[TOPIC]:
                assert b["prompt"] not in gbody
        assert llm_poller.tick() == 1

        # 9. GATE AND PROVENANCE
        d = client.get(f"/api/datasets/{did}").json()
        assert d["status"] == "ready", d["error"]
        pv = d["provenance"]
        assert pv["format"] == "doc" and pv["items"]["kept"] == 6
        assert pv["gate"]["rejected"] is False and pv["gate"]["exam_questions"] > 0
        assert pv["identities"] == {"exam_writer": "fake/fake-exam", "judge": "stub/overlap-v1",
                                    "generator": "fake/fake-1", "single_provider_loop": False}
        assert pv["judge_run"]["judge_id"] == "stub/overlap-v1"
        assert prop.provenance_complete(pv) == []
        items = [json.loads(x) for x in
                 client.get(f"/api/datasets/{did}/items.jsonl").text.splitlines()]
        assert len(items) == 6 and all(set(it) == {"title", "text"} for it in items)

        # 10. TAINT — a run that consumes it, with the parent it started from
        rid = client.post("/api/truns", json={"name": "loop", "datasets": [did],
                                              "parent": "fx/good-750m"}).json()["id"]
        client.post(f"/api/truns/{rid}/event", json={"step": 1, "detail": "fx/skewed-360m"})
        fresh(appmod)
        after = client.get("/api/results").json()
        child = next(m for m in after["models"] if m["id"] == "fx/skewed-360m")
        assert child["tainted"] == [TASK]
        # the topic it trained on is shown but no longer counts toward the average
        assert child["judgedAvg"] == report.judged_avg(child["judge"], [TASK])
        assert child["judgedAvg"] != report.judged_avg(child["judge"], [])
        assert child["judge"]["tasks"][TASK]["score_report"] is not None

        # 11. BEFORE AND AFTER — both halves of the topic, on the rubric scale
        cmp = child["taintCompare"][TASK]
        assert cmp["scale"] == "rubric" and cmp["parent"] == "fx/good-750m"
        assert cmp["judge"] == "stub/overlap-v1"
        assert cmp["verdict"] in ("skill", "test", "none", "mixed")
        assert cmp["before"]["report"]["n"] and cmp["after"]["report"]["n"]
        assert "of 4" in cmp["text"] and "points" not in cmp["text"]
        # the "after" report half is the child's own report-half mean, computed
        # here from the per-item file rather than from the same summary
        raw = json.loads((tree["models"]["fx/skewed-360m"]["dir"] / "judge.json").read_text())
        rep = [it["score"] for it in raw["tasks"][TASK]["items"] if it["half"] == "report"]
        assert cmp["after"]["report"]["n"] == len(rep)
        assert cmp["after"]["report"]["v"] == pytest.approx(sum(rep) / len(rep), abs=1e-3)
        assert jd.rubric_for(TASK)[2] == "1"
    finally:
        client.__exit__(None, None, None)
