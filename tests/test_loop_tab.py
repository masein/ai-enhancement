"""Phase 8e P6a: the loop as one board, and one topic at a time.

The board asks the server what the next step is for each topic; the server
answers with the same gates the API enforces on submit and on propose. A
judged run can be narrowed to one topic, and the answers panel shows the
diagnosis half — never the report half, in any field of any response."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import make_service

REPO = Path(__file__).resolve().parents[1]
TOPIC = "Medicine & Clinical Health"
TASK = "exam_medicine_clinical_health"
LAW = "exam_law"
HISTORY = "History & Archaeology"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def rows(client, model: str = "") -> dict:
    j = client.get("/api/loop", params={"model": model} if model else None).json()
    return {r["topic"]: r for r in j["topics"]}


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------

def test_the_board_has_a_row_per_topic_with_its_bank_rubric_and_last_run(svc):
    client, _, _ = svc
    j = client.get("/api/loop").json()
    assert len(j["topics"]) == len(eb.TOPICS)
    med = rows(client)[TOPIC]
    assert med["task"] == TASK and med["slug"] == "medicine_clinical_health"
    assert med["bank"]["accepted"] == med["bank"]["report"] + med["bank"]["diagnose"]
    assert med["bank"]["under_floor"] is True and med["bank"]["floor"] == 30
    assert med["rubric"]["name"] == "medicine_clinical_health" and med["rubric"]["own"] is True
    assert med["rubric"]["criteria_count"] == 20
    last = med["last_judged"]
    assert last["model"] and last["score_report"] is not None
    # the delivered rubric's heading carries no DRAFT, so nothing stamps it one
    assert last["judge_id"] == "stub/overlap-v1" and last["draft_rubric"] is False
    assert set(last["flags"]) == {"critical_medicine_clinical_health_error"}
    # every topic is graded by its own — Arts too, since 2026-09-22
    assert rows(client)["Arts"]["rubric"]["name"] == "arts"



def test_a_draft_rubric_is_stamped_on_the_topics_row(svc):
    """No delivered rubric is a draft any more, so the stamp is exercised with
    one: the topic's own rubric with DRAFT in its heading, where the page's
    rubric upload would put it, and the topic re-judged over it."""
    import conftest
    from service import config
    client, appmod, tree = svc
    store = Path(config.BENCH_ROOT) / "rubrics"
    store.mkdir(parents=True, exist_ok=True)
    head, rest = (REPO / "eval_tasks" / "fr" / "rubrics" / "medicine_clinical_health.md"
                  ).read_text(encoding="utf-8").split("\n", 1)
    (store / "medicine_clinical_health.md").write_text(f"{head} (DRAFT)\n{rest}",
                                                        encoding="utf-8")
    assert jd.rubric_for(TASK).status == "draft"
    model = client.get("/api/loop").json()["model"]
    mdir = tree["models"][model]["dir"]
    jd.write_judge(mdir, jd.merge_judged(mdir, jd.run_stub(mdir, tree["out_dir"], only=[TASK])))
    conftest.fresh(appmod)
    # this model's board, asked for by name: which model the board opens on
    # is the one judged most recently, and three runs of one second are a
    # coin toss — this test is about the stamp, not about that order
    board = rows(client, model)
    last = board[TOPIC]["last_judged"]
    assert last["model"] == model and last["draft_rubric"] is True
    # and only this topic: the others were graded by rubrics nobody marked
    assert board["Law"]["last_judged"]["draft_rubric"] is False

def test_the_next_step_walks_the_loop_in_order(svc, tmp_path, monkeypatch):
    client, appmod, _ = svc
    from service import config, db
    # a topic nobody has written questions for: the step is to get a bank
    (eb.bank_dir(config.EXAM_DIR) / "sociology.jsonl").unlink()
    empty = next(r for r in client.get("/api/loop").json()["topics"]
                 if r["topic"] == "Sociology")
    assert empty["bank"]["accepted"] == 0
    assert empty["next"]["step"] == "import" and empty["next"]["ok"] is True
    # a bank with no judged run: sit the exam — and the reason it cannot be
    # sat is the API's own refusal, not a sentence the page made up
    med = rows(client)[TOPIC]
    # the fixture has judged runs: the next step is to propose — reading is
    # what the topic page is for, and no longer a step a browser remembers
    assert med["next"]["step"] == "propose"
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    blocked = client.get("/api/loop").json()["judged_blocked"]
    assert blocked and blocked == config.judged_blocked()
    # a topic with questions that no model has sat yet
    for d in config.OUT_DIR.iterdir():
        jf = d / "judge.json"
        if not jf.exists():
            continue
        j = json.loads(jf.read_text(encoding="utf-8"))
        j["tasks"].pop("exam_history_archaeology", None)
        jf.write_text(json.dumps(j), encoding="utf-8")
    appmod._cache.update(key=None, payload=None, at=0.0)
    fresh = next(r for r in client.get("/api/loop").json()["topics"]
                 if r["topic"] == HISTORY)
    assert fresh["bank"]["accepted"] and fresh["last_judged"] is None
    assert fresh["next"] == {"step": "sit", "label": "Sit the exam", "ok": False,
                             "why": blocked}
    # a proposal in flight moves the step to reviewing it
    pid = db.proposal_create("fx/good-750m", TASK, TOPIC, "omar", {"n_shown": 1})
    assert rows(client)[TOPIC]["next"]["step"] == "review"
    assert rows(client)[TOPIC]["proposal"]["id"] == pid
    # approved, nothing generated yet
    db.proposal_update(pid, status="approved")
    assert rows(client)[TOPIC]["next"]["step"] == "generate"
    # and a dataset out the other end
    did = db.dataset_create(pid, "prose", 20, "omar", "")
    db.dataset_update(did, status="ready",
                      provenance=json.dumps({"items": {"generated": 20, "kept": 18,
                                                       "dropped": 2}}))
    med = rows(client)[TOPIC]
    assert med["next"]["step"] == "hand"
    assert med["datasets"][0] == {"id": did, "status": "ready", "count": 20, "kept": 18,
                                  "created_at": med["datasets"][0]["created_at"],
                                  "over_provisional_judge": None}


def test_the_propose_gate_is_the_apis_own_words(svc):
    client, _, _ = svc
    med = rows(client)[TOPIC]
    # the fixture's medicine bank is under the floor, and that is the reason
    # the propose endpoint gives too
    assert med["propose"]["ok"] is False
    assert "under the 30" in med["propose"]["why"]
    r = client.post("/api/proposals", json={"model": med["last_judged"]["model"],
                                            "topic": TOPIC, "requested_by": "omar"})
    assert r.status_code in (409, 503)
    if r.status_code == 409:
        assert r.json()["detail"] == med["propose"]["why"]


# ---------------------------------------------------------------------------
# sitting one topic
# ---------------------------------------------------------------------------

def test_a_judged_run_can_be_narrowed_to_one_topic(svc, monkeypatch):
    client, _, _ = svc
    from service import config, db
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    r = client.post("/api/submissions", json={"hf_id": "org/m", "suite": "judged",
                                              "tasks": [LAW]})
    assert r.status_code == 200, r.text
    assert r.json()["tasks"] == [LAW]
    row = db.get(r.json()["id"])
    assert json.loads(row["tasks"]) == [LAW] and row["suite"] == "judged"
    # the whole suite is still what an empty list means
    r2 = client.post("/api/submissions", json={"hf_id": "org/m2", "suite": "judged"})
    assert json.loads(db.get(r2.json()["id"])["tasks"]) == []
    # only built exam tasks, and only for a judged run
    bad = client.post("/api/submissions", json={"hf_id": "org/m3", "suite": "judged",
                                                "tasks": ["exam_astrology"]})
    assert bad.status_code == 422 and "not built exam tasks" in bad.json()["detail"]
    wrong = client.post("/api/submissions", json={"hf_id": "org/m4", "suite": "full",
                                                  "tasks": [LAW]})
    assert wrong.status_code == 422 and "judged run only" in wrong.json()["detail"]


def test_the_judge_grades_only_the_narrowed_tasks(svc, tmp_path):
    """The answers to the other topics are on disk from an earlier run; a
    narrowed run must not pay to grade them again."""
    client, _, tree = svc
    from service import config
    model_dir = config.OUT_DIR / "fx__good-750m"
    reqs, plan = jd.plan_requests(model_dir, "stub")
    narrowed_reqs, narrowed = jd.plan_requests(model_dir, "stub", only=[LAW])
    assert set(plan["tasks"]) > {LAW} and set(narrowed["tasks"]) == {LAW}
    assert len(narrowed_reqs) < len(reqs)
    assert all(r.meta.get("task") in (LAW, None) for r in narrowed_reqs)


def test_a_narrowed_run_keeps_the_other_topics_it_did_not_regrade(svc):
    client, _, _ = svc
    from service import config
    model_dir = config.OUT_DIR / "fx__good-750m"
    before = json.loads((model_dir / "judge.json").read_text(encoding="utf-8"))
    out = jd.run_stub(model_dir, config.OUT_DIR, record=False, only=[LAW])
    assert set(out["tasks"]) == {LAW}
    merged = jd.merge_judged(model_dir, out)
    assert set(merged["tasks"]) == set(before["tasks"])       # nothing lost
    assert merged["tasks"][LAW] == out["tasks"][LAW]          # and the new grades are the new ones
    assert "replaced" not in merged["judge"]
    # a different judge's earlier work is NOT carried under this one's heading
    stale = {**before, "judge": {**before["judge"], "id": "anthropic/claude-x"}}
    (model_dir / "judge.json").write_text(json.dumps(stale), encoding="utf-8")
    merged2 = jd.merge_judged(model_dir, out)
    assert set(merged2["tasks"]) == {LAW}
    assert merged2["judge"]["replaced"] == sorted(t for t in before["tasks"] if t != LAW)
    assert "re-run suite=judged" in merged2["judge"]["replaced_why"]


# ---------------------------------------------------------------------------
# the answers
# ---------------------------------------------------------------------------

def test_the_answers_are_the_diagnosis_half_and_only_that(svc):
    client, _, _ = svc
    from service import config
    model = "fx/good-750m"
    j = client.get("/api/answers", params={"model": model, "topic": TOPIC}).json()
    assert j["items"] and j["n_diagnose"] == len(j["items"])
    assert all(eb.half_of(it["qid"]) == "diagnose" for it in j["items"])
    assert [c["id"] for c in j["criteria"]] == jd.criteria_ids(jd.rubric_for(TASK).criteria)
    assert j["flags"][0]["effect_words"] == "sets the whole score to 0"
    # the report half is one line: a count and a mean, no rows, no qids
    assert j["report_half"]["n"] and j["report_half"]["mean"] is not None
    body = json.dumps(j)
    bank = eb.load_bank(config.EXAM_DIR)[TOPIC]
    report = [b for b in bank if eb.half_of(b["qid"]) == "report"]
    assert report
    for b in report:
        assert b["prompt"] not in body and b["prompt"][:60] not in body
        assert b["qid"] not in body
    # and what IS shown is the question, the answer and the judge's reading
    one = j["items"][0]
    assert one["prompt"] and set(one["criteria"]) == set(c["id"] for c in j["criteria"])
    assert isinstance(one["flags"], dict) and "justification" in one


def test_the_answers_endpoint_refuses_what_it_cannot_show(svc):
    client, _, _ = svc
    assert client.get("/api/answers", params={"model": "fx/nope", "topic": TOPIC}
                      ).status_code == 404
    assert client.get("/api/answers", params={"model": "fx/good-750m", "topic": "astrology"}
                      ).status_code == 404
    # a topic this model has not been judged on
    from service import config
    j = json.loads((config.OUT_DIR / "fx__good-750m" / "judge.json").read_text())
    del j["tasks"]["exam_history_archaeology"]
    (config.OUT_DIR / "fx__good-750m" / "judge.json").write_text(json.dumps(j))
    assert client.get("/api/answers", params={"model": "fx/good-750m", "topic": HISTORY}
                      ).status_code == 404


def test_the_queue_row_carries_the_judge_batch(svc, monkeypatch):
    client, _, _ = svc
    from service import config, db
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    sid = client.post("/api/submissions", json={"hf_id": "fx/good-750m", "suite": "judged",
                                                "tasks": [LAW]}).json()["id"]
    rid = db.judge_run_create("fx/good-750m", "batch_1", 40, "stub/overlap-v1", "{}")
    db.batch_add("batch_1", "judge", rid, 40, "local", "chat")
    db.batch_progress("batch_1", "18/40 done")
    db.update(sid, judge_batch="batch_1")      # as judge.start_run records it on submit
    row = next(r for r in client.get("/api/submissions").json() if r["id"] == sid)
    assert json.loads(row["tasks"]) == [LAW]
    assert row["judge"]["progress"] == "18/40 done" and row["judge"]["n_items"] == 40


# ---------------------------------------------------------------------------
# a topic with no rubric of its own, and a board that survives it
# ---------------------------------------------------------------------------

def test_a_topic_without_its_own_rubric_is_graded_by_the_shared_one(svc, arts_without_rubric):
    """Arts arrived without its files, as thirteen of the first fifteen topics
    did; it has them now, so this test takes them out of the repo copy the
    judge reads. The fallback is rubrics/exam.md, as P4a specified — not
    rubrics/<slug>.md, which does not exist, and which took both of these
    endpoints down on the live tree the moment anyone opened them."""
    client, _, _ = svc
    rubrics = client.get("/api/exam/rubrics")
    assert rubrics.status_code == 200
    rows = {t["topic"]: t for t in rubrics.json()["topics"]}
    arts = rows["Arts"]
    assert arts["name"] == "exam" and arts["fallback"] is True and arts["own"] is False
    assert arts["error"] == "" and arts["sha256"] and arts["path"].endswith("exam.md")
    med = rows[TOPIC]
    assert med["name"] == "medicine_clinical_health" and med["fallback"] is False
    # the loop board says the same thing, for every topic
    loop = client.get("/api/loop")
    assert loop.status_code == 200
    lrows = {t["topic"]: t for t in loop.json()["topics"]}
    assert len(lrows) == len(eb.TOPICS)
    assert lrows["Arts"]["rubric"]["fallback"] is True
    assert lrows[TOPIC]["rubric"]["fallback"] is False
    assert all(not t["error"] for t in lrows.values())
    # and the judge reads the same file the page names
    assert jd.rubric_for("exam_arts").path == arts["path"]
    assert jd.rubric_for("exam_arts").name == "exam"


def test_neither_board_dies_when_a_rubric_file_is_missing(svc, tmp_path, monkeypatch):
    """One unreadable topic is a fact about that topic. Both endpoints
    iterate every topic, and before this both returned 500 for all fifteen
    because of one."""
    client, _, _ = svc
    empty = tmp_path / "no-rubrics-here"
    empty.mkdir()
    monkeypatch.setattr(jd, "RUBRIC_DIR", empty)       # not even exam.md
    rubrics = client.get("/api/exam/rubrics")
    assert rubrics.status_code == 200
    rows = {t["topic"]: t for t in rubrics.json()["topics"]}
    assert len(rows) == len(eb.TOPICS)
    hist = rows[HISTORY]
    assert "no rubric file for exam_history_archaeology" in hist["error"]
    assert "history_archaeology.md" in hist["error"] and "exam.md" in hist["error"]
    assert hist["sha256"] == ""                        # nothing invented
    loop = client.get("/api/loop")
    assert loop.status_code == 200
    lrows = {t["topic"]: t for t in loop.json()["topics"]}
    assert len(lrows) == len(eb.TOPICS)
    bad = lrows[HISTORY]
    assert "no rubric file for exam_history_archaeology" in bad["error"]
    # and the row offers no button that would fail later
    assert bad["next"]["ok"] is False and bad["next"]["why"] == bad["error"]
    # every topic is in the same boat here, and all thirty-seven rows still render
    assert all(t["error"] for t in lrows.values())


def test_the_missing_rubric_is_named_not_guessed(svc, tmp_path, monkeypatch):
    empty = tmp_path / "empty-rubrics"
    empty.mkdir()
    monkeypatch.setattr(jd, "RUBRIC_DIR", empty)
    with pytest.raises(jd.RubricMissing) as e:
        jd.rubric_for("exam_history_archaeology")
    assert "history_archaeology.md" in str(e.value) and "exam.md" in str(e.value)
    assert str(empty) in str(e.value)


# ---------------------------------------------------------------------------
# what a person waiting on a judged run is told
# ---------------------------------------------------------------------------

def test_a_batch_in_flight_reports_how_far_it_is(svc, monkeypatch):
    """"judging 40/130", not "judging 130 answers". Every provider reports
    counts while the batch runs; the poller records what it was told."""
    from service import config, db, llm
    client, _, _ = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    sid = client.post("/api/submissions", json={"hf_id": "fx/good-750m", "suite": "judged",
                                                "tasks": [LAW]}).json()["id"]
    rid = db.judge_run_create("fx/good-750m", "b_1", 130, "stub/overlap-v1", "{}")
    db.batch_add("b_1", "judge", rid, 130, "anthropic", "claude-x")
    db.update(sid, judge_batch="b_1")
    # what the provider says while it runs, in the shape every backend returns
    assert llm._counted({"processing": 90, "succeeded": 40, "total": 130}) == "40/130 done"
    assert llm._counted({"succeeded": 38, "errored": 2, "total": 130}) == "40/130 done, 2 failed"
    assert llm._counted({}, "in_progress") == "in_progress"
    db.batch_progress("b_1", llm._counted({"succeeded": 40, "total": 130}))
    row = next(r for r in client.get("/api/submissions").json() if r["id"] == sid)
    assert row["judge"]["progress"] == "40/130 done" and row["judge"]["n_items"] == 130


def test_the_poller_records_what_the_provider_said(tmp_path, monkeypatch):
    """The poller asks for status anyway; recording the answer is the whole
    feature. It must not write on every tick when nothing moved."""
    from service import config, db, llm, llm_poller
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "s.sqlite3")
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    monkeypatch.setattr(config, "LLM_PROVIDER", "fake")
    monkeypatch.setattr(config, "LLM_MODEL", "fake-1")
    monkeypatch.setattr(llm.FakeBatches, "polls_to_done", 3)
    llm.reset()
    db.init()
    backend = llm.client()
    bid = backend.submit([llm.Request(custom_id=f"x{i}", system="", user="hi", max_tokens=8)
                          for i in range(4)])
    db.batch_add(bid, "proposal", 1, 4, backend.name, backend.model)
    llm_poller.tick()
    row = next(b for b in db.batches_list(10) if b["batch_id"] == bid)
    assert row["progress"] == "1/4 done"          # a third of the way, recorded
    llm_poller.tick()
    row = next(b for b in db.batches_list(10) if b["batch_id"] == bid)
    assert row["progress"] == "3/4 done"
