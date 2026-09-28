"""12o.1: the judge test shows each answer's question to the person marking
it, so it draws only from questions that may be shown — the Knowledge exam's
diagnose half (its report half is never listed) and Everyday's practice half
(12n.1). A sample drawn before either rule is screened when it is next read:
what can't be shown leaves it, and the rest keep their version and marks.

Fixtures and the stand-in judge only."""

from __future__ import annotations

import json

import pytest

import everyday as ev
import judge_calibrate as jc
from conftest import make_service
from service import config, db, judge_test
from test_12n1_everyday import _answered


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    _answered("org/answers-right", lambda q: q.get("reference") or "yes")
    ev.remark(config.OUT_DIR, judge=True, snapshot=False)
    yield client
    client.__exit__(None, None, None)


def judged():
    return jc._judged_rows(config.OUT_DIR, set())


def test_the_exam_rows_are_the_diagnose_half_only(svc):
    halves = {r["half"] for r in judged()}
    assert {"report", "diagnose"} <= halves                 # both are on file
    rows = judge_test._exam_rows()
    assert rows and {r["half"] for r in rows} == {"diagnose"}
    report = {r["prompt"] for r in judged() if r["half"] == "report"}
    assert not report & {r["question"] for r in rows}
    # and Everyday's, the practice half (12n.1)
    assert {r["half"] for r in judge_test._everyday_rows()} == {ev.PRACTICE}


def test_a_new_sample_holds_no_question_that_is_never_shown(svc):
    client = svc
    page = client.get("/api/judge-test").json()
    assert page["answers"]
    shown = {a["question"] for a in page["answers"]}
    report = {r["prompt"] for r in judged() if r["half"] == "report"}
    hidden = {q["prompt"] for q in ev.load_bank() if ev.half(q) == ev.HIDDEN}
    assert not shown & report and not shown & hidden


def test_a_sample_drawn_before_is_screened_and_keeps_its_version_and_marks(svc):
    client = svc
    # as 12f.0's builder drew it: both exam halves, both Everyday halves, no half on a row
    exam = [{"key": "exam:" + r["id"], "kind": "exam", "model": r["model"], "task": r["task"],
             "topic": r["category"], "question": r["prompt"], "reference": r["reference"],
             "answer": r["answer"], "judge": r["judge_score"]} for r in judged()][:40]
    assert {r["half"] for r in judged()[:40]} == {"report", "diagnose"}
    qs = {q["id"]: q for q in ev.load_bank()}
    d = config.OUT_DIR / "org__answers-right"
    evd = [{"key": f"everyday:{d.name}|{it['id']}", "kind": "everyday", "model": "org/answers-right",
            "task": it["id"], "topic": "t", "question": qs[it["id"]]["prompt"], "reference": "",
            "rubric": "", "answer": it["answer_text"], "judge": 4}
           for it in ev.read(d)["items"]][:20]
    assert {ev.half(qs[a["task"]]) for a in evd} == {ev.HIDDEN, ev.PRACTICE}
    old = exam + evd
    p = judge_test._sample_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n": config.JUDGE_TEST_N, "at": 1.0, "answers": old,
                             "builder": judge_test.BUILDER, "version": "v1234abcd"}))
    for a in old:
        db.jt_mark(judge_test.person("v1234abcd"), a["key"], 3, "masein")
    page = client.get("/api/judge-test").json()
    # the same version: nothing was drawn again, and the marks stand
    assert page["version"] == "v1234abcd" and not page["history"]
    keys = {a["key"] for a in page["answers"]}
    report = {"exam:" + r["id"] for r in judged() if r["half"] == "report"}
    hidden = {a["key"] for a in evd if ev.half(qs[a["task"]]) == ev.HIDDEN}
    assert keys and not keys & report and not keys & hidden
    assert keys == {a["key"] for a in old} - report - hidden
    assert page["progress"]["marked"] == len(keys)
    # written back with each row's half: screened once
    on_file = json.loads(p.read_text())["answers"]
    assert all(a["half"] in ("diagnose", ev.PRACTICE) for a in on_file)
    report_q = {r["prompt"] for r in judged() if r["half"] == "report"}
    assert not report_q & {a["question"] for a in page["answers"]}


def test_a_row_whose_half_cant_be_told_leaves_the_sample(svc):
    p = judge_test._sample_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    gone = {"key": "exam:org__deleted|exam_law|abc", "kind": "exam", "model": "org/deleted",
            "task": "exam_law", "topic": "Law", "question": "q", "reference": "r",
            "answer": "a", "judge": 2}
    p.write_text(json.dumps({"n": config.JUDGE_TEST_N, "at": 1.0, "answers": [gone],
                             "builder": judge_test.BUILDER, "version": "v1"}))
    assert judge_test.answers() == []
