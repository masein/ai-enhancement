"""Find the gap, generate data, keep a human in the loop — end to end against
the fake backend, and the rule twice."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import diagnose as dx
import exam_build as eb
import judge as jd
import report_lm_eval as report
from service import proposals as prop_mod
from conftest import assert_no_report_half_text, fresh, make_service
from service import contamination as ct
from service import llm, llm_poller, runner

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gap(tmp_path, monkeypatch):
    client, appmod, manifest = make_service(tmp_path, monkeypatch)
    yield client, appmod, manifest
    client.__exit__(None, None, None)


def _row(client, mid):
    return next(m for m in client.get("/api/results").json()["models"] if m["id"] == mid)


TOPIC = "Economics"          # the fixture's one topic above the 30-question floor
TASK = "exam_economics"


def _propose(client, model="fx/good-750m", topic=TOPIC, who="tester"):
    return client.post("/api/proposals", json={"model": model, "topic": topic,
                                                "requested_by": who})


# ---------------------------------------------------------------------------
# off by default; on with the fake
# ---------------------------------------------------------------------------

def test_off_unless_configured(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, llm_provider="", tree=False)
    try:
        st = client.get("/api/llm").json()
        assert st["configured"] is False and "LLM_PROVIDER is unset" in st["reason"]
        r = _propose(client)
        assert r.status_code == 503 and "LLM_PROVIDER" in r.json()["detail"]
    finally:
        client.__exit__(None, None, None)


def test_startup_refuses_a_half_configured_provider(monkeypatch):
    from service import config
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    with pytest.raises(RuntimeError, match="LLM_MODEL is unset"):
        llm.startup_check()
    monkeypatch.setattr(config, "LLM_MODEL", "claude-x")
    with pytest.raises(RuntimeError, match="LLM_API_KEY is unset"):
        llm.startup_check()
    monkeypatch.setattr(config, "LLM_PROVIDER", "mistral")
    assert "not one of" in llm.blocked()
    monkeypatch.setattr(config, "LLM_PROVIDER", "")
    llm.startup_check()                       # unset is fine: the feature is simply off


def test_llm_status_reports_usage_and_quota(gap):
    client, _, _ = gap
    st = client.get("/api/llm").json()
    assert st["configured"] and st["provider"] == "fake" and st["usage_today"] == 0
    assert st["daily_cap"] == 2000 and st["datasets_quota_bytes"] == 20_000_000_000
    assert "tailnet" in st["note"]


# ---------------------------------------------------------------------------
# the gate, on the payload and on the API
# ---------------------------------------------------------------------------

def test_topic_gate_on_the_payload(payload):
    """The gate is per EXAM TOPIC now: the judged suite must be believable,
    the topic must have enough report-half questions, and the model must have
    written something. MMLU rides along as a caution, never as a gate."""
    def topics(mid):
        row = next(m for m in payload["models"] if m["id"] == mid)
        return {t: v["propose"] for t, v in (row["judge"] or {})["tasks"].items()
                if t.startswith("exam_")}
    g = topics("fx/good-750m")
    assert g[TASK]["ok"] is True and g[TASK]["why"] is None
    thin = g["exam_law"]
    assert thin["ok"] is False and "under the 30" in thin["why"] and "Exam tab" in thin["why"]
    assert thin["short"] and len(thin["short"]) < len(thin["why"])
    assert thin["caution"] is None                       # noise on a row already refused
    # the model that wrote the same sentence every time: the output collapsed
    c = topics("fx/chance-160m")[TASK]
    assert c["ok"] is False and "same answer on nearly every question" in c["why"]
    # MMLU's finding for the matching category is the caution, and only that
    assert g[TASK]["caution"] is None                     # good-750m has no mmlu finding
    skew = topics("fx/skewed-360m")[TASK]
    assert skew["caution"] and "answer positions" in skew["caution"]
    # the control task is not a topic and carries no action
    row = next(m for m in payload["models"] if m["id"] == "fx/good-750m")
    assert "propose" not in row["judge"]["tasks"]["fr_control_mmlu"]
    # MMLU's own gate still computes — it is where the caution comes from
    assert row["diag"]["tasks"]["mmlu"]["propose"]["categories"]["Economics"]["ok"] is True


def test_topic_gate_refuses_a_preliminary_suite_and_an_empty_writer(tree):
    import report_lm_eval as rep
    ok_state = {"ok": True, "reasons": [], "current": True}
    good = {"n_report": 40, "answers": {"n": 80, "empty": 0, "short": 1, "distinct": 70}}
    assert rep.topic_gate(TASK, good, ok_state, None)["ok"] is True
    bad = rep.topic_gate(TASK, good, {"ok": False, "reasons": ["kappa 0.4 is below 0.6"]}, None)
    assert bad["ok"] is False and "preliminary" in bad["why"] and "0.4" in bad["why"]
    assert rep.topic_gate(TASK, good, None, None)["ok"] is False
    thin = rep.topic_gate(TASK, {**good, "n_report": 12}, ok_state, None)
    assert "12 hidden questions" in thin["why"]              # 11h: plain words
    empty = rep.topic_gate(TASK, {"n_report": 40, "answers": {"n": 40, "empty": 18, "short": 4,
                                                              "distinct": 20}}, ok_state, None)
    assert "wrote nothing usable on 22 of 40" in empty["why"] and "multiple choice" in empty["why"]
    same = rep.topic_gate(TASK, {"n_report": 40, "answers": {"n": 40, "empty": 0, "short": 0,
                                                             "distinct": 1}}, ok_state, None)
    assert "same answer on nearly every question" in same["why"]
    assert rep.topic_gate(TASK, good, ok_state, "picks by option length")["caution"] == \
        "picks by option length"


def test_api_enforces_the_same_gate(gap):
    client, _, tree = gap
    r = _propose(client, "fx/chance-160m")
    assert r.status_code == 409 and "same answer on nearly every question" in r.json()["detail"]
    r = _propose(client, topic="Law")
    assert r.status_code == 409 and "under the 30" in r.json()["detail"]
    r = _propose(client, topic="astrology")
    assert r.status_code == 404 and "no judged answers on file" in r.json()["detail"]
    assert _propose(client, "nobody/nothing").status_code == 404
    r = _propose(client, tree["nodiag"])
    assert r.status_code == 404 and "suite=judged" in r.json()["detail"]
    assert client.get("/api/proposals").json() == []          # nothing was recorded


# ---------------------------------------------------------------------------
# the whole flow, and the rule twice
# ---------------------------------------------------------------------------

def test_propose_approve_generate_gate_provenance_taint(gap):
    client, appmod, tree = gap
    fake = llm.client()
    assert isinstance(fake, llm.FakeBatches)

    # -- propose: the judge's reasoning about diagnose-half answers only ---------
    r = _propose(client)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert r.json()["status"] == "pending" and r.json()["batch_id"].startswith("fake_")
    assert r.json()["task"] == TASK
    assert _propose(client).status_code == 409                # already pending
    assert client.get("/api/llm").json()["usage_today"] == 1
    assert llm_poller.tick() == 1
    p = client.get(f"/api/proposals/{pid}").json()
    assert p["status"] == "proposed" and p["proposer"] == "fake/fake-1"
    assert p["task"] == TASK and p["category"] == TOPIC
    assert f"introductory {TOPIC}" in p["spec_text"]
    ev = p["evidence"]
    assert 0 < ev["n_shown"] <= 60 and ev["diagnose_weak"] <= ev["diagnose_items"]
    assert 1 <= len(ev["examples"]) <= 8 and ev["patterns"] and ev["share_explained"] == 0.6
    assert ev["topic_n_report"] >= 30 and ev["judge_id"] == "stub/overlap-v1"
    assert all(e["justification"] and e["score"] < 3 for e in ev["examples"])
    # the proposal row records which judge run it was derived from
    assert json.loads(p["judge_run"]) == {"judge_id": "stub/overlap-v1", "batch_id": "stub",
                                          "prompt_sha256": jd.prompt_sha()}

    # THE RULE (a): every qid behind the request splits to diagnose, and NO exam
    # question — of either half — appears anywhere in the request body
    reqs = [q for q in fake.recorded() if q["custom_id"] == f"proposal:{pid}"]
    assert len(reqs) == 1
    req = reqs[0]
    qids = req["meta"]["qids"]
    assert qids and all(dx.split_of(q) == "diagnose" for q in qids)
    body = req["system"] + "\n" + req["user"]
    bank = eb.load_bank(tree["judged"]["exam_root"])[TOPIC]
    halves = {eb.half_of(b["qid"]) for b in bank}
    assert halves == {"report", "diagnose"}               # both are on file…
    for b in bank:                                       # …and neither is in the request
        assert b["prompt"] not in body, b["qid"]
    assert assert_no_report_half_text(body, bank)          # nor anything else of the report half
    assert TOPIC in body and "rubric the judge graded against" in body
    assert p["prompt_sha"] == llm.prompt_sha(req["system"], req["user"])

    # -- the human: a name is the record; generation waits for approval ---------
    assert client.post(f"/api/proposals/{pid}/generate",
                       json={"requester": "t", "count": 10}).status_code == 409
    r = client.post(f"/api/proposals/{pid}/approve", json={"approver": "  ", "edited_text": ""})
    assert r.status_code == 422 and "name" in r.json()["detail"]
    edited = p["spec_text"] + " Emphasise direction-of-effect reasoning."
    r = client.post(f"/api/proposals/{pid}/approve",
                    json={"approver": "Omar", "edited_text": edited})
    assert r.status_code == 200 and r.json()["edited"] is True
    p = client.get(f"/api/proposals/{pid}").json()
    assert p["status"] == "approved" and p["approver"] == "Omar" and p["edited_text"] == edited
    assert client.post(f"/api/proposals/{pid}/approve",
                       json={"approver": "x"}).status_code == 409

    # -- generate: only the spec goes ---------------------------------------------
    r = client.post(f"/api/proposals/{pid}/generate",
                    json={"requester": "Omar", "count": 6})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]
    assert r.json()["items"] == 3                              # documents are long: two per request
    assert client.get("/api/llm").json()["usage_today"] == 4
    gen = [q for q in fake.recorded() if q["custom_id"].startswith(f"gen:{did}:")]
    assert len(gen) == 3
    assert all(q["meta"]["format"] == "doc" for q in gen)
    assert all("never write a question-and-answer pair" in q["system"] for q in gen)
    ix = ct.index(tree["out_dir"])
    assert ix.grams
    for q in gen:
        body = q["system"] + "\n" + q["user"]
        assert edited in body and TOPIC in body
        # THE RULE (b): no 13-gram of any benchmark item, either half
        assert ix.hits(body) == [], "a generation request carries benchmark text"
        assert not any(d["q"] in body for docs in tree["docs"].values() for d in docs if d["q"])
        assert "fx/good-750m" not in body and "doc_hash" not in body
        assert_no_report_half_text(body, bank)
        assert "doc_hashes" not in q["meta"]
    assert client.get(f"/api/datasets/{did}/items.jsonl").status_code == 409   # not yet

    assert llm_poller.tick() == 1
    d = client.get(f"/api/datasets/{did}").json()
    assert d["status"] == "ready", d["error"]
    pv = d["provenance"]
    # every document asked for is accounted for (11a): six asked, six kept
    assert pv["items"] == {"requested": 6, "generated": 6, "dropped": 0, "kept": 6,
                           "missing": []}
    assert pv["format"] == "doc"
    assert pv["gate"]["rejected"] is False and pv["gate"]["ngram"] == 13
    assert pv["approver"] == "Omar" and pv["requester"] == "Omar"
    assert pv["approved_spec"] == edited and pv["spec_text"] == p["spec_text"]
    assert pv["generator"] == {"provider": "fake", "model": "fake-1",
                               "batch_id": d["batch_id"], "id": "fake/fake-1"}
    assert pv["split"] == "diagnose" and pv["split_salt"] == dx.SPLIT_SALT
    assert pv["source_model"] == "fx/good-750m" and pv["task"] == TASK
    assert pv["category"] == TOPIC and pv["proposal_id"] == pid
    assert pv["judge_run"]["judge_id"] == "stub/overlap-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", pv["prompt_sha256"])
    from service import proposals as prop
    assert prop.provenance_complete(pv) == []
    items_path = tree["root"] / "datasets" / str(did) / "items.jsonl"
    body = items_path.read_bytes()
    import hashlib
    assert pv["items_sha256"] == hashlib.sha256(body).hexdigest()
    assert json.loads((items_path.parent / "provenance.json").read_text()) == pv
    items = [json.loads(x) for x in body.decode().splitlines()]
    assert len(items) == 6
    # prose documents, not question-and-answer pairs
    assert all(set(it) == {"title", "text"} for it in items)
    assert all(len(it["text"].split()) >= prop_mod.DOC_MIN_WORDS for it in items)
    assert len({it["title"] for it in items}) == 6
    dl = client.get(f"/api/datasets/{did}/items.jsonl")
    assert dl.status_code == 200 and dl.content == body
    lst = client.get("/api/datasets").json()
    assert lst[0]["id"] == did and lst[0]["download"] == f"/api/datasets/{did}/items.jsonl"
    assert lst[0]["task"] == TASK and lst[0]["category"] == TOPIC

    # -- taint: through truns -> tevents -> model, and by hf_prefix ---------------
    before = {m["id"]: (m["official"], m["avg"], m["judgedAvg"])
              for m in client.get("/api/results").json()["models"]}
    assert before["fx/good-750m"][0] is True
    r = client.post("/api/truns", json={"name": "gap-run", "datasets": [did],
                                        "hf_prefix": "fx/skewed"})
    rid = r.json()["id"]
    client.post(f"/api/truns/{rid}/event", json={"step": 100, "detail": "fx/good-750m"})
    assert client.get(f"/api/truns/{rid}").json()["run"]["datasets"] == [did]
    fresh(appmod)
    after = client.get("/api/results").json()
    good = next(m for m in after["models"] if m["id"] == "fx/good-750m")
    # the data came from an EXAM topic, so the multiple-choice average is
    # untouched — and that topic stops counting toward the judged average
    assert good["tainted"] == [TASK] and good["official"] is True
    assert good["avg"] == before["fx/good-750m"][1]
    assert good["judge"]["tasks"][TASK]["score_report"] is not None      # the score stays
    assert good["judgedAvg"] != before["fx/good-750m"][2]
    assert good["judgedAvg"] == report.judged_avg(good["judge"], [TASK])
    # the exam's own before/after wants a parent; this run recorded none
    cmp = good["taintCompare"][TASK]
    assert cmp["scale"] == "rubric" and cmp["parent"] is None
    assert "recorded no parent" in cmp["missing"]
    skewed = next(m for m in after["models"] if m["id"] == "fx/skewed-360m")
    assert skewed["tainted"] == [TASK]                         # via hf_prefix
    for mid in ("fx/chance-160m", "fx/below-135m-it", "fx/short-pick-410m"):
        m = next(x for x in after["models"] if x["id"] == mid)
        assert m["tainted"] == [] and (m["official"], m["avg"], m["judgedAvg"]) == before[mid], mid
    assert any("trained on data derived from benchmark diagnostics" in w for w in after["warnings"])
    # a run may only record data it could have trained on
    assert client.post("/api/truns", json={"name": "x", "datasets": [999]}).status_code == 422
    # and a referenced dataset cannot be deleted from under its provenance
    assert client.delete(f"/api/datasets/{did}").status_code == 409


def test_generation_free_response_and_unreferenced_delete(gap):
    client, appmod, tree = gap
    pid = _propose(client).json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
    p = client.get(f"/api/proposals/{pid}").json()
    assert p["edited_text"] == ""                              # approved as written
    bad = client.post(f"/api/proposals/{pid}/generate",
                      json={"requester": "Omar", "count": 12, "fmt": "mc"})
    assert bad.status_code == 422 and "retired" in bad.json()["detail"]
    did = client.post(f"/api/proposals/{pid}/generate",
                      json={"requester": "Omar", "count": 12, "fmt": "free"}).json()["dataset_id"]
    llm_poller.tick()
    d = client.get(f"/api/datasets/{did}").json()
    assert d["status"] == "ready" and d["provenance"]["items"]["kept"] == 12
    assert d["provenance"]["format"] == "free"
    items = [json.loads(x) for x in client.get(f"/api/datasets/{did}/items.jsonl").text.splitlines()]
    assert all("choices" not in it and it["question"] for it in items)
    assert d["provenance"]["approved_spec"] == d["provenance"]["spec_text"]
    assert client.delete(f"/api/datasets/{did}").json() == {"deleted": did}
    assert client.get(f"/api/datasets/{did}").json()["status"] == "deleted"
    assert not (tree["root"] / "datasets" / str(did)).exists()


def test_reject_records_who_and_why(gap):
    client, _, _ = gap
    pid = _propose(client).json()["id"]
    llm_poller.tick()
    assert client.post(f"/api/proposals/{pid}/reject", json={"approver": ""}).status_code == 422
    r = client.post(f"/api/proposals/{pid}/reject",
                    json={"approver": "Omar", "reason": "names the questions, not the skill"})
    assert r.status_code == 200
    p = client.get(f"/api/proposals/{pid}").json()
    assert p["status"] == "rejected" and p["approver"] == "Omar"
    assert p["reject_reason"] == "names the questions, not the skill"
    assert client.post(f"/api/proposals/{pid}/generate",
                       json={"requester": "x", "count": 5}).status_code == 409
    # a rejected proposal frees the cell for a new one
    assert _propose(client).status_code == 200


def test_parse_items_on_the_document_format():
    doc = {"title": "Margins first", "text": "word " * 200}
    assert prop_mod.parse_items(json.dumps([doc]), "doc") == [
        {"title": "Margins first", "text": ("word " * 200).strip()}]
    # malformed: no title, no body, a body too short to teach anything, not a
    # dict, not an array, and a rogue question-shaped item
    bad = [{"text": "word " * 200}, {"title": "t"}, {"title": "t", "text": "too short"},
           "a string", 7, {"question": "q?", "answer": "a"}]
    assert prop_mod.parse_items(json.dumps(bad), "doc") == []
    assert prop_mod.parse_items("not json at all", "doc") == []
    # a lone object IS one document: a request for a single document (what a
    # local generator gets) comes back bare under JSON mode, not in an array
    assert prop_mod.parse_items(json.dumps({"title": "t", "text": "word " * 200}), "doc") == [
        {"title": "t", "text": ("word " * 200).strip()}]
    assert prop_mod.parse_items(json.dumps({"nothing": "doc shaped"}), "doc") == []
    # `body` is accepted as a synonym, and the title is capped
    alt = {"title": "T" * 400, "body": "word " * 200}
    got = prop_mod.parse_items(json.dumps([alt]), "doc")
    assert len(got) == 1 and len(got[0]["title"]) == 300
    # the free format is unchanged; mc is gone
    free = {"question": "q?", "answer": "a", "rationale": "r"}
    assert prop_mod.parse_items(json.dumps([free]), "free") == [free]
    assert prop_mod.FORMATS == ("doc", "free") and prop_mod.DEFAULT_FORMAT == "doc"
    assert prop_mod.items_per_request("doc") == 2 and prop_mod.items_per_request("free") == 10


def test_the_generation_request_asks_for_prose_and_nothing_exam_shaped():
    reqs = prop_mod.generation_requests(7, "The model cannot separate a rule from its purpose.",
                                        "Law", 5, "doc", seed=7)
    assert [r.meta["count"] for r in reqs] == [2, 2, 1]
    body = reqs[0].system + "\n" + reqs[0].user
    assert "never write a question-and-answer pair" in body and "not a quiz" in body
    assert "Topic: Law" in body and "Write 2 documents" in body
    assert "The model cannot separate a rule from its purpose." in body
    assert "multiple-choice" in body and str(prop_mod.DOC_TARGET_WORDS) in body
    # the spec and the topic are all it gets: no model, no score, no question
    assert "fx/" not in body and "qid" not in body


# ---------------------------------------------------------------------------
# the gate on real output, the spend guard, the quota, the poller
# ---------------------------------------------------------------------------

def _echoing_responder(tree, share, source="mmlu"):
    """A generator that buries a benchmark question — or an exam question — in
    the body of `share` of its documents, the way a leak would actually look."""
    if source == "mmlu":
        qs = [d["q"] for d in tree["docs"]["mmlu"]]
    else:
        qs = [b["prompt"] for b in eb.load_bank(tree["judged"]["exam_root"])[TOPIC]]
    assert qs

    def responder(req):
        if not req.custom_id.startswith("gen:"):
            return llm.default_responder(req)
        items = json.loads(llm.default_responder(req))
        start = int(req.meta.get("start", 0))
        for i, it in enumerate(items):
            if (start + i) % round(1 / share) == 0:
                q = qs[(start + i) % len(qs)]
                it["text"] = (it["text"].split("\n\n")[0] + "\n\nA question of the kind this "
                              f"teaches: {q}\n\n" + "\n\n".join(it["text"].split("\n\n")[1:]))
        return json.dumps(items)
    return responder


def test_dataset_echoing_the_benchmark_is_rejected(gap, monkeypatch):
    client, _, tree = gap
    pid = _propose(client).json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(_echoing_responder(tree, 0.10)))
    did = client.post(f"/api/proposals/{pid}/generate",
                      json={"requester": "Omar", "count": 100}).json()["dataset_id"]
    llm_poller.tick()
    d = client.get(f"/api/datasets/{did}").json()
    assert d["status"] == "rejected" and "echoing the test" in d["error"]
    g = d["provenance"]["gate"]
    assert g["rejected"] is True and g["dropped_benchmark"] == 10 and g["offending_ngrams"]
    assert g["exam_questions"] > 0                     # the exam is indexed too
    it = d["provenance"]["items"]
    assert (it["requested"], it["generated"], it["dropped"], it["kept"]) == (100, 100, 100, 0)
    # the ten the gate dropped one by one are named; the rest went with the
    # whole batch, which the error above says in words
    assert [m["why"] for m in it["missing"]] == ["dropped by the gate: benchmark"] * 10
    assert not (tree["root"] / "datasets" / str(did) / "items.jsonl").exists()
    assert client.get(f"/api/datasets/{did}/items.jsonl").status_code == 409
    # a single echo in fifty is dropped, recorded, and the rest kept
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(_echoing_responder(tree, 0.02)))
    did2 = client.post(f"/api/proposals/{pid}/generate",
                       json={"requester": "Omar", "count": 50}).json()["dataset_id"]
    llm_poller.tick()
    d2 = client.get(f"/api/datasets/{did2}").json()
    assert d2["status"] == "ready"
    it2 = d2["provenance"]["items"]
    assert (it2["requested"], it2["generated"], it2["dropped"], it2["kept"]) == (50, 50, 1, 49)
    assert [m["why"] for m in it2["missing"]] == ["dropped by the gate: benchmark"]
    assert d2["provenance"]["gate"]["dropped_benchmark"] == 1


def test_spend_guard(gap, monkeypatch):
    client, _, _ = gap
    from service import config
    monkeypatch.setattr(config, "LLM_DAILY_ITEM_CAP", 1)
    assert _propose(client).status_code == 200
    r = _propose(client, "fx/skewed-360m")          # a different cell, so the dup check passes
    assert r.status_code == 429 and "LLM_DAILY_ITEM_CAP" in r.json()["detail"]
    monkeypatch.setattr(config, "LLM_DAILY_ITEM_CAP", 2000)
    monkeypatch.setattr(config, "LLM_MAX_ITEMS_PER_BATCH", 2)
    llm_poller.tick()
    pid = client.get("/api/proposals").json()[0]["id"]
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
    r = client.post(f"/api/proposals/{pid}/generate", json={"requester": "Omar", "count": 50})
    assert r.status_code == 422 and "LLM_MAX_ITEMS_PER_BATCH" in r.json()["detail"]
    assert client.post(f"/api/proposals/{pid}/generate",
                       json={"requester": "Omar", "count": 4}).status_code == 200


def test_quota_refusal(gap, monkeypatch):
    client, _, _ = gap
    from service import config
    pid = _propose(client).json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
    monkeypatch.setattr(config, "DATASET_QUOTA_GB", 0.0)
    r = client.post(f"/api/proposals/{pid}/generate", json={"requester": "Omar", "count": 10})
    assert r.status_code == 507 and "quota" in r.json()["detail"]
    assert client.get("/api/datasets").json() == []          # nothing half-made


def test_poller_resumes_a_persisted_batch_after_a_restart(gap, monkeypatch):
    client, appmod, _ = gap
    monkeypatch.setattr(llm.FakeBatches, "polls_to_done", 3)
    pid = _propose(client).json()["id"]
    n_sent = len(llm.client().recorded())
    assert llm_poller.tick() == 0                             # still processing
    assert client.get(f"/api/proposals/{pid}").json()["status"] == "pending"
    # "restart": a fresh process has no client object and no in-memory state,
    # only the llm_batches row and the provider's own record of the batch
    llm.reset()
    from service import db
    assert [b["batch_id"] for b in db.batches_pending()] == [
        client.get(f"/api/proposals/{pid}").json()["batch_id"]]
    assert llm_poller.tick() == 0
    assert llm_poller.tick() == 1
    assert client.get(f"/api/proposals/{pid}").json()["status"] == "proposed"
    assert len(llm.client().recorded()) == n_sent             # resumed, never re-submitted
    assert db.batches_pending() == []


def test_a_failed_batch_marks_the_proposal_failed(gap, monkeypatch):
    client, _, _ = gap

    def boom(req):
        raise RuntimeError("provider says no")
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(boom))
    pid = _propose(client).json()["id"]
    llm_poller.tick()
    p = client.get(f"/api/proposals/{pid}").json()
    assert p["status"] == "failed" and "provider says no" in p["error"]


# ---------------------------------------------------------------------------
# secrets
# ---------------------------------------------------------------------------

KEY_PATTERNS = [r"sk-ant-[A-Za-z0-9_\-]{10,}", r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}"]


def test_the_key_is_interpolated_never_literal():
    compose = (REPO / "docker-compose.yml").read_text()
    assert re.search(r"LLM_API_KEY:\s*\$\{LLM_API_KEY", compose)
    for pat in KEY_PATTERNS:
        assert not re.search(pat, compose)
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split()
    for f in tracked:
        p = REPO / f
        if not p.is_file() or p.suffix in (".png",):
            continue
        text = p.read_text(errors="ignore")
        for pat in KEY_PATTERNS:
            assert not re.search(pat, text), f"key-shaped string in {f}"
    assert ".env" in (REPO / ".gitignore").read_text().split()


def test_the_key_is_stripped_from_evaluation_subprocesses(monkeypatch):
    assert "LLM_API_KEY" in runner.SECRET_ENV_VARS and "ANTHROPIC_API_KEY" in runner.SECRET_ENV_VARS
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    env = runner._child_env(remote_code=True)
    assert "LLM_API_KEY" not in env and "HF_TOKEN" not in env and env["HF_HUB_OFFLINE"] == "1"
    # a stock (non-remote-code) job keeps its environment; the key only matters
    # where someone else's code runs — and that is documented in SERVICE.md
    assert "docker inspect" in (REPO / "SERVICE.md").read_text()


def test_backends_shape_the_provider_requests_without_the_network(monkeypatch):
    """The real backends are HTTP glue; check the bodies they would send."""
    sent = []

    def fake_http(method, url, headers, body=None, timeout=60.0):
        sent.append((method, url, headers, body))
        if url.endswith("/messages/batches") and method == "POST":
            return 200, b'{"id": "msgbatch_1"}'
        if "/files" in url and method == "POST":
            return 200, b'{"id": "file_1"}'
        if url.endswith("/batches") and method == "POST":
            return 200, b'{"id": "batch_1"}'
        raise AssertionError(url)
    monkeypatch.setattr(llm, "_http", fake_http)
    # a proposal request, which parses JSON and so asks for it (proposal_request)
    req = llm.Request("proposal:1", "sys", "user text", 100, {"doc_hashes": ["x"]}, json=True)
    a = llm.AnthropicBatches("claude-x", "key-a")
    assert a.submit([req]) == "msgbatch_1"
    body = json.loads(sent[-1][3])
    assert body["requests"][0]["custom_id"] == "proposal:1"
    assert body["requests"][0]["params"]["model"] == "claude-x"
    assert "meta" not in json.dumps(body) and "doc_hashes" not in json.dumps(body)
    assert sent[-1][2]["x-api-key"] == "key-a"
    o = llm.OpenAIBatches("gpt-x", "key-o")
    assert o.submit([req]) == "batch_1"
    upload = sent[-2][3].decode()
    assert '"custom_id": "proposal:1"' in upload and "doc_hashes" not in upload
    assert '"response_format": {"type": "json_object"}' in upload
    assert sent[-2][2]["authorization"] == "Bearer key-o"
    assert llm.extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json("[1, 2] trailing") == [1, 2]
    assert llm.extract_json("no json here") is None
    # a one-element array of objects is an array, not the object inside it —
    # a request for a single document returns exactly this
    assert llm.extract_json('[{"title": "t"}]') == [{"title": "t"}]
    assert llm.extract_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]
    assert llm.extract_json('{"items": [1]}') == {"items": [1]}
