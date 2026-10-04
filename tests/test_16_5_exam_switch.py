"""16.5: the Knowledge exam switched off — KNOWLEDGE_EXAM, default 0.

- **Off, it appears nowhere:** /api/results carries none of it (examOn
  false, no judged result on any model, no exam topics, none of its checks or
  alarms).
- **Nothing about it is run, asked or generated:**
  - its endpoints answer 409 in one line;
  - a judged submission is refused, and one queued before the switch skips
    its tasks and says so;
  - the question builder offers Everyday alone;
  - a new judge-test sample is Everyday's alone, and the current sample's
    exam answers are neither shown nor asked;
  - Improve's exam proposals and datasets are hidden, counted, and kept.
- **Nothing is deleted,** and no number on the board changes.
- **On** (`KNOWLEDGE_EXAM=1`), everything is back as it was: every other test
  runs with it on (conftest)."""

from __future__ import annotations

import time

import pytest

from conftest import make_service
from service import config, db, judge_test, runner


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client, appmod


def results(client, appmod, on: bool, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_EXAM", on)
    appmod._cache.update(key=None, payload=None, at=0.0)
    return client.get("/api/results").json()


def off(monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_EXAM", False)


def test_the_switch_is_off_by_default_and_reads_1_as_on(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_EXAM", raising=False)
    assert config._switch("KNOWLEDGE_EXAM", "0") is False
    monkeypatch.setenv("KNOWLEDGE_EXAM", "1")
    assert config._switch("KNOWLEDGE_EXAM", "0") is True


def test_off_the_payload_carries_none_of_it_and_no_number_changes(svc, monkeypatch):
    client, appmod = svc
    on = results(client, appmod, True, monkeypatch)
    assert on["examOn"] is True and on["judged"]["exam"]
    assert any((m.get("judge") or {}).get("tasks") for m in on["models"])
    gone = results(client, appmod, False, monkeypatch)
    assert gone["examOn"] is False
    assert gone["judged"]["exam"] == [] and gone["judged"]["topics"] == {}
    assert gone["judged"]["tasks"] == [] and gone["judged"]["calibration"] is None
    for m in gone["models"]:
        assert m.get("judge") is None and m.get("judgedAvg") is None and not m.get("judgedEarlier")
    assert not [c for c in gone.get("checks") or [] if (c.get("show") or {}).get("tab") == "exam"]
    assert not [a for a in gone.get("alarms") or [] if a["key"] == "exam-report"]
    # no number on the board moves: the Avg, its error, the rank's inputs, every cell
    keys = ("avg", "avgRaw", "nhave", "nreq", "params")
    a = {m["id"]: {k: m.get(k) for k in keys} for m in on["models"]}
    b = {m["id"]: {k: m.get(k) for k in keys} for m in gone["models"]}
    assert a == b
    assert on["cells"] == gone["cells"] and on["accTasks"] == gone["accTasks"]


def test_off_its_endpoints_say_so_and_do_nothing(svc, monkeypatch):
    client, _ = svc
    off(monkeypatch)
    for path in ("/api/exam", "/api/exam/candidates", "/api/exam/bank", "/api/exam/rubrics",
                 "/api/loop", "/api/judge", "/api/judge/justifications", "/api/ai/rejudge"):
        r = client.get(path)
        assert r.status_code == 409, path
        assert r.json()["detail"].startswith("The Knowledge exam is switched off on this server")
    r = client.post("/api/exam/build", json={"by": "masein"})
    assert r.status_code == 409
    # on, the same answer as before
    monkeypatch.setattr(config, "KNOWLEDGE_EXAM", True)
    assert client.get("/api/exam").status_code == 200


def test_off_a_judged_run_is_refused_and_one_queued_before_skips_its_tasks(svc, monkeypatch):
    client, _ = svc
    off(monkeypatch)
    r = client.post("/api/submissions", json={"hf_id": "fx/good-750m", "suite": "judged",
                                              "submitter": "masein"})
    assert r.status_code == 422 and r.json()["detail"].endswith("Nothing was queued.")
    # queued while it was on
    sid = db.add("fx/good-750m", "instruct", "judged", "masein", "")
    from service import disk
    monkeypatch.setattr(disk, "blocks_run", lambda: None)
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed"
    assert row["error"].endswith("Its exam tasks were skipped: nothing was run.")


def test_off_its_questions_are_not_listed(svc, monkeypatch):
    client, _ = svc
    from service import questions
    exam = [t for t in questions.tasks() if questions.kind_of(t) == "exam"]
    off(monkeypatch)
    assert not [t for t in questions.tasks() if questions.kind_of(t) == "exam"]
    if exam:
        assert client.get(f"/api/questions/{exam[0]}").status_code == 409


def test_off_the_builder_offers_everyday_alone(svc, monkeypatch):
    client, _ = svc
    off(monkeypatch)
    page = client.get("/api/builder").json()
    assert page["kinds"] == ["everyday"] and page["topics"] == []
    assert set(page["prompts"]) == {"everyday"}
    r = client.post("/api/builder", json={"kind": "knowledge", "count": 10, "topic": "Law",
                                          "by": "masein"})
    assert r.status_code == 409 and r.json()["detail"].endswith("Nothing was written.")
    assert client.post("/api/builder/estimate", json={"kind": "knowledge", "count": 10}).status_code \
        == 409


def test_off_improves_exam_proposals_and_datasets_are_hidden_counted_and_kept(svc, monkeypatch):
    client, appmod = svc
    pe = db.proposal_create("fx/good-750m", "exam_law", "Law", "masein", {})
    pv = db.proposal_create("fx/good-750m", "everyday:instructions", "instructions", "masein", {})
    de = db.dataset_create(pe, "doc", 10, "masein", {})
    off(monkeypatch)
    ids = [p["id"] for p in client.get("/api/proposals").json()]
    assert pe not in ids and pv in ids
    assert de not in [d["id"] for d in client.get("/api/datasets").json()]
    assert client.get(f"/api/proposals/{pe}").status_code == 409
    assert client.get(f"/api/datasets/{de}").status_code == 409
    assert client.delete(f"/api/datasets/{de}").status_code == 409        # kept, never deleted
    r = results(client, appmod, False, monkeypatch)
    assert r["examHidden"] == {"proposals": 1, "datasets": 1}
    # a new proposal on an exam topic: refused
    r = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": "Law",
                                            "requested_by": "masein"})
    assert r.status_code == 409 and r.json()["detail"].endswith("Nothing was proposed.")
    # on: all there, as before
    monkeypatch.setattr(config, "KNOWLEDGE_EXAM", True)
    assert pe in [p["id"] for p in client.get("/api/proposals").json()]
    assert db.proposal_get(pe) and db.dataset_get(de)


def test_off_a_new_judge_test_sample_is_everydays_and_the_exam_answers_are_kept(svc, monkeypatch):
    client, _ = svc
    evd = [{"key": f"everyday:m|q{i}", "kind": "everyday", "model": "m", "answer": f"an answer {i}",
            "question": f"q {i}", "task": f"q{i}", "topic": "Writing", "half": "practice", "reference": "", "judge": 4, "rubric": "r"}
           for i in range(5)]
    exam = [{"key": f"exam:{i}", "kind": "exam", "model": "m", "answer": f"a long answer {i}",
             "question": f"what {i}", "task": "exam_law", "topic": "Law", "half": "diagnose", "reference": "ref", "judge": 2}
            for i in range(30)]
    monkeypatch.setattr(judge_test, "_everyday_rows", lambda: list(evd))
    monkeypatch.setattr(judge_test, "_exam_rows", lambda: list(exam))
    monkeypatch.setattr(judge_test, "_instruct", lambda: {"m"})
    monkeypatch.setattr(judge_test, "degenerate", lambda a, q="": False)
    monkeypatch.setattr(config, "JUDGE_TEST_N", 14)
    # on: a sample of both, kept
    before = judge_test.answers(n=14, rebuild=True)
    assert {a["kind"] for a in before} == {"everyday", "exam"}
    off(monkeypatch)
    # the sample stays; its exam answers are neither shown nor asked of a judge
    page = client.get("/api/judge-test").json()
    assert page["exam_hidden"] == sum(a["kind"] == "exam" for a in before)
    assert {a["key"] for a in page["answers"]} <= {a["key"] for a in before if a["kind"] == "everyday"}
    assert judge_test.answers() == before                               # not redrawn
    # a new one is Everyday's alone
    fresh = judge_test.answers(n=14, rebuild=True)
    assert fresh and {a["kind"] for a in fresh} == {"everyday"}


def test_off_the_ai_jobs_and_the_playground_say_nothing_of_it(svc, monkeypatch):
    client, _ = svc
    off(monkeypatch)
    jobs = {j["job"]: j["does"] for j in client.get("/api/ai").json()["jobs"]}
    assert "Knowledge exam" not in " ".join(jobs.values())
    assert client.get("/api/playground/practice").json().get("knowledge") in ([], None)


def test_off_a_judged_run_is_no_models_test_date(svc, monkeypatch):
    sid = db.add("fx/good-750m", "instruct", "judged", "masein", "", status="done")
    db.update(sid, finished_at=time.time())
    assert "fx/good-750m" in db.last_done(judged=True)
    assert "fx/good-750m" not in db.last_done(judged=False)
