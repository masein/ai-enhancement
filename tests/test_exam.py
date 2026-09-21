"""C1: the exam is the instrument. Drafted by an LLM, curated by a person,
split by qid — and no report-half question ever leaves the bank."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest

import categories
import diagnose as dx
import exam_build as eb
from conftest import make_service
from service import llm

REPO = Path(__file__).resolve().parents[1]


def test_qid_is_the_normalised_text_and_splits_deterministically():
    a = eb.qid_of("Why does a price ceiling cause a shortage?")
    assert a == eb.qid_of("  why does a PRICE ceiling cause a shortage ")
    assert a != eb.qid_of("Why does a price floor cause a surplus?")
    assert eb.half_of(a) == dx.split_of(a) and eb.half_of(a) in ("report", "diagnose")
    # the split is the benchmarks' own: same function, same salt
    assert dx.SPLIT_SALT == "evalboard-split-v1"
    qids = [hashlib.sha256(f"q{i}".encode()).hexdigest() for i in range(5000)]
    share = sum(eb.half_of(q) == "diagnose" for q in qids) / len(qids)
    assert 0.48 < share < 0.52


def test_bank_halves_are_near_even_and_topics_match_categories(tree):
    root = tree["judged"]["exam_root"]
    bank = eb.load_bank(root)
    assert set(bank) == set(categories.category_order())            # exactly the yaml's topics
    files = {p.stem for p in (root / "bank").glob("*.jsonl")}
    assert files <= {categories.topic_slug(t) for t in categories.category_order()}
    rows = [r for rs in bank.values() for r in rs]
    assert len(rows) >= 100
    for r in rows:
        assert r["topic"] in bank and r["qid"] == eb.qid_of(r["prompt"])
        assert r["accepted_by"] and r["accepted_at"] and r["reference"]
    halves = collections.Counter(eb.half_of(r["qid"]) for r in rows)
    assert 0.35 <= halves["diagnose"] / len(rows) <= 0.65
    assert len({r["qid"] for r in rows}) == len(rows)                # no duplicates


def test_the_rule_no_report_half_question_in_any_request(tree):
    """The drafting requests are the only LLM calls C1 makes. Their example
    questions come from the bank; every one must be diagnose-half, and no
    report-half prompt may appear anywhere in a request body."""
    root = tree["judged"]["exam_root"]
    fake = llm.FakeBatches("fake-exam", tree["root"])
    reqs = [q for q in fake.recorded() if q["custom_id"].startswith("exam:")]
    assert len(reqs) >= len(categories.category_order())
    bank = [r for rs in eb.load_bank(root).values() for r in rs]
    report = [r["prompt"] for r in bank if eb.half_of(r["qid"]) == "report"]
    diagnose = {r["prompt"] for r in bank if eb.half_of(r["qid"]) == "diagnose"}
    assert report and diagnose
    quoted = 0
    for q in reqs:
        body = q["system"] + "\n" + q["user"]
        for p in report:
            assert p not in body, "a REPORT-half question left the bank in a request"
        for ex in q["meta"].get("example_qids", []):
            assert dx.split_of(ex) == "diagnose"
        quoted += sum(1 for p in diagnose if p in body)
    # General & Multidisciplinary had the migrated questions when drafting ran,
    # so examples were used
    assert quoted > 0


def test_examples_for_drafting_come_from_the_diagnose_half(tmp_path):
    root = tmp_path / "exam"
    eb.migrate_seeds(root)
    ex = eb.diagnose_half_examples(root, categories.OTHER, k=50)
    other = eb.load_bank(root)[categories.OTHER]
    assert ex and all(eb.half_of(eb.qid_of(p)) == "diagnose" for p in ex)
    assert len(ex) == sum(1 for r in other if eb.half_of(r["qid"]) == "diagnose")
    assert eb.diagnose_half_examples(root, "Law") == []
    reqs = eb.draft_requests(root, categories.OTHER, 9, per_request=4)
    assert [r.meta["count"] for r in reqs] == [4, 4, 1]
    assert all(f"Topic: {categories.OTHER}" in r.user and "Rubric" in r.user for r in reqs)
    assert all(eb.qid_of(e) for e in ex) and "do not copy" in reqs[0].user


def test_a_candidate_never_reaches_the_bank_without_a_name(tmp_path):
    root = tmp_path / "exam"
    fake = llm.FakeBatches("fake-exam", tmp_path)
    out = eb.draft(root, fake, ["Law"], per_topic=3, wait=True, poll_s=0)
    assert out["written"] == {"Law": 3}
    cands = eb.load_candidates(root, "Law", "candidate")
    assert len(cands) == 3 and all(c["drafted_by"] == "fake/fake-exam" for c in cands)
    assert eb.load_bank(root)["Law"] == []                              # drafts are not the bank
    with pytest.raises(ValueError, match="needs a name"):
        eb.accept(root, cands[0]["cid"], approver="  ")
    with pytest.raises(ValueError, match="needs a name"):
        eb.reject(root, cands[0]["cid"], approver="")
    assert eb.load_bank(root)["Law"] == []
    rec = eb.accept(root, cands[0]["cid"], approver="Omar")
    assert rec["accepted_by"] == "Omar" and rec["edited"] is False and rec["source"] == "llm-draft"
    assert eb.load_bank(root)["Law"][0]["qid"] == rec["qid"]
    # an edit is a new question: the accepted text is what gets hashed
    rec2 = eb.accept(root, cands[1]["cid"], approver="Omar",
                     prompt=cands[1]["prompt"] + " Give one counterexample.")
    assert rec2["edited"] is True and rec2["qid"] != cands[1]["qid"]
    assert rec2["qid"] == eb.qid_of(cands[1]["prompt"] + " Give one counterexample.")
    with pytest.raises(ValueError, match="already accepted"):
        eb.accept(root, cands[0]["cid"], approver="Omar")
    r = eb.reject(root, cands[2]["cid"], approver="Omar", reason="recall, not understanding")
    assert r["status"] == "rejected" and r["reason"] == "recall, not understanding"
    assert eb.load_candidates(root, "Law", "candidate") == []
    assert eb.summary(root)["Law"]["accepted"] == 2 and eb.summary(root)["Law"]["rejected"] == 1
    # fetching the same batch twice writes nothing twice; a re-draft continues
    # past what the topic already has and never re-offers a bank question
    assert eb.fetch(root, fake, out["batch_id"])["written"] == {}
    out2 = eb.draft(root, fake, ["Law"], per_topic=3, wait=True, poll_s=0)
    assert out2["written"] == {"Law": 3}
    bank_q = {r["qid"] for r in eb.load_bank(root)["Law"]}
    new = eb.load_candidates(root, "Law", "candidate")
    assert len(new) == 3 and not any(c["qid"] in bank_q for c in new)
    assert {c["qid"] for c in new}.isdisjoint({c["qid"] for c in cands})


def test_manifest_hashes_match_the_files(tree):
    m = tree["judged"]["manifest"]
    td = tree["judged"]["tasks_dir"]
    assert m["split_salt"] == dx.SPLIT_SALT
    for task, v in m["tasks"].items():
        p = td / f"{task}.jsonl"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == v["sha256"]
        items = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        assert len(items) == v["items"] == v["report"] + v["diagnose"]
        assert (td / f"{task}.yaml").exists()
        if task.startswith("exam_"):
            assert v["topic"] == eb.TASK_TOPIC[task] and m["topics"][task] == v["topic"]
            assert all(it["qid"] == eb.qid_of(it["prompt"]) and it["topic"] == v["topic"]
                       for it in items)
            assert all("half" not in it and "accepted_by" not in it for it in items)
    for name, sha in m["rubrics"].items():
        assert hashlib.sha256((REPO / "eval_tasks" / "fr" / "rubrics" / f"{name}.md").read_bytes()
                              ).hexdigest() == sha
    assert "exam" in m["rubrics"]
    assert set(m["tasks"]) <= set(eb.ALL_TASKS)
    assert (td / "manifest.json").exists()


def test_build_skips_empty_topics_and_removes_their_stale_files(tree, tmp_path):
    root = tmp_path / "exam"
    eb.migrate_seeds(root)                  # only General & Multidisciplinary has questions
    m = eb.build(tree["out_dir"], root)
    assert set(m["tasks"]) == {"exam_general_multidisciplinary", "fr_control_mmlu"}
    assert not (eb.tasks_dir(root) / "exam_law.jsonl").exists()
    (eb.tasks_dir(root) / "exam_law.jsonl").write_text("stale\n")
    (eb.tasks_dir(root) / "exam_law.yaml").write_text("stale\n")
    eb.build(tree["out_dir"], root)
    assert not (eb.tasks_dir(root) / "exam_law.jsonl").exists()


def test_public_bank_withholds_report_half_text(tree):
    root = tree["judged"]["exam_root"]
    rows = eb.public_bank(root)
    assert rows and {r["half"] for r in rows} == {"report", "diagnose"}
    for r in rows:
        if r["half"] == "report":
            assert r["prompt"] is None and r["reference"] is None and "withheld" in r
        else:
            assert r["prompt"] and r["reference"]
    law = eb.public_bank(root, "Law")
    assert law and all(r["topic"] == "Law" for r in law)


def test_curation_through_the_service(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judged=True)
    try:
        st = client.get("/api/exam").json()
        assert st["configured"] and st["provider"] == "fake"
        assert set(st["summary"]) == set(categories.category_order())
        assert st["summary"][categories.OTHER]["accepted"] >= 40 and st["tasks_built"]
        assert "report half" in st["note"]
        # draft more into this service's exam root, then curate over the API
        root = tree["judged"]["exam_root"]
        # the topic name carries '&' and spaces, so it goes in the query encoded
        topic = {"topic": "History & Archaeology"}
        out = eb.draft(root, llm.client("exam"), [topic["topic"]], per_topic=2, wait=True,
                       poll_s=0)
        assert out["written"] == {"History & Archaeology": 2}
        cands = client.get("/api/exam/candidates", params=topic).json()
        assert len(cands) == 2 and all(c["status"] == "candidate" for c in cands)
        cid = cands[0]["cid"]
        r = client.post(f"/api/exam/candidates/{cid}/accept", json={"approver": ""})
        assert r.status_code == 422
        r = client.post(f"/api/exam/candidates/{cid}/accept",
                        json={"approver": "Omar", "prompt": cands[0]["prompt"] + " Be specific."})
        assert r.status_code == 200 and r.json()["edited"] is True and r.json()["half"] in ("report", "diagnose")
        assert client.post(f"/api/exam/candidates/{cid}/accept", json={"approver": "Omar"}).status_code == 409
        assert client.post("/api/exam/candidates/nope/accept", json={"approver": "Omar"}).status_code == 404
        r = client.post(f"/api/exam/candidates/{cands[1]['cid']}/reject",
                        json={"approver": "Omar", "reason": "too vague"})
        assert r.status_code == 200
        assert client.get("/api/exam/candidates", params=topic).json() == []
        from service import db
        cur = db.curation_list()
        assert [c["decision"] for c in cur] == ["rejected", "accepted"]
        assert all(c["approver"] == "Omar" for c in cur) and cur[1]["edited"] == 1
        assert cur[1]["qid"] == r.json().get("qid", cur[1]["qid"])
        # the bank over the API never shows a report-half question
        bank = client.get("/api/exam/bank", params=topic).json()
        assert bank and all((b["prompt"] is None) == (b["half"] == "report") for b in bank)
        # rebuilding the tasks picks the new question up
        built = client.post("/api/exam/build").json()
        assert built["tasks"]["exam_history_archaeology"]["items"] == \
            eb.summary(root)["History & Archaeology"]["accepted"]
        assert "exam_history_archaeology" in client.get("/api/judge").json()["tasks"]
    finally:
        client.__exit__(None, None, None)


def test_config_identities_are_separate(monkeypatch):
    from service import config
    monkeypatch.setattr(config, "EXAM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "EXAM_MODEL", "")
    monkeypatch.setattr(config, "EXAM_API_KEY", "")
    assert "EXAM_MODEL is unset" in llm.blocked("exam")
    with pytest.raises(RuntimeError, match="EXAM misconfigured"):
        llm.startup_check()
    monkeypatch.setattr(config, "EXAM_PROVIDER", "")
    assert "EXAM_PROVIDER is unset" in llm.blocked("exam")
    assert llm.identity("exam") == ("", "", "")
    from service import runner
    assert "EXAM_API_KEY" in runner.SECRET_ENV_VARS and "JUDGE_API_KEY" in runner.SECRET_ENV_VARS
    compose = (REPO / "docker-compose.yml").read_text()
    assert "EXAM_API_KEY: ${EXAM_API_KEY:-}" in compose
    assert "EXAM_PROVIDER" in (REPO / ".env.example").read_text()
    assert not (REPO / "scripts" / "fr_build.py").exists()
