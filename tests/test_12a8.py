"""12a.8: after the rules change, a re-mark reaches both halves, and every
reader shows the marks it wrote. On the server the practice half's readers
kept 12a.6's verdicts after 12a.7's re-mark. The page's data was cached on
the newest time among the result files, so one file dated in the future hid
every write after it; now any file rewritten changes the key. And
`--question` sends the judge one question's answers, after its rubric or
reference changed — the school run plan's reference, fixed here.

Fixtures and the stand-in judge only."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

import everyday as ev
from service import config, questions
from service.app import files_stamp
from test_12n1_everyday import A, _dir, _answered, item, svc  # noqa: F401

SCHOOL_RUN = "everyday-summarising-07"
FIXED = ("Tomorrow: she takes Zain to football at 4; you pick Layla up at 5:30; everyone home by "
         "6:30 for dinner. Your mum arrives at 7. Get Zain's cake and top up Layla's transit card.")


def judged(half_: str, model: str = A) -> dict:
    """a Summarise question of this half the model's answer took to the judge"""
    out = ev.read(_dir(model))
    qs = {q["id"]: q for q in ev.load_bank()}
    return next(qs[it["id"]] for it in out["items"]
                if it["group"] == "summarising" and it["half"] == half_ and it.get("rubric")
                and ev.grade(qs[it["id"]], it["answer_text"])[0] is None)


def new_rubric(monkeypatch, *ids) -> None:
    """the rules change: these questions' rubrics, as a deploy would bring them"""
    real = ev.load_bank

    def bank(path=None):
        rows = real(path)
        if path is None:
            for q in rows:
                if q["id"] in ids:
                    c = ev.judge_check(q)
                    c["rubric"] = c["rubric"] + "\n7. (a changed rubric)"
        return rows
    monkeypatch.setattr(ev, "load_bank", bank)


def now(qid: str) -> dict:
    """the question as the bank reads it now"""
    return next(q for q in ev.load_bank() if q["id"] == qid)


def plant(qid: str, reason: str, model: str = A) -> None:
    """the marks from before the change, as they sat on the server"""
    f = _dir(model) / ev.OUT_NAME
    out = json.loads(f.read_text(encoding="utf-8"))
    for it in out["items"]:
        if it["id"] == qid:
            it.update({"pass": False, "reason": reason, "score": 1})
    f.write_text(json.dumps(out), encoding="utf-8")


def future(model: str = A) -> None:
    """a result file dated a day ahead — copied, unpacked or from another clock"""
    f = _dir(model) / "model_meta.json"
    t = time.time() + 86400
    os.utime(f, (t, t))


def page_item(client, appmod, qid: str, model: str = A) -> dict | None:
    appmod._cache["at"] = 0.0                    # past its five seconds: the key decides
    ev_ = client.get("/api/results").json()["everyday"]
    return next((it for it in ev_["models"][model]["items"] if it["id"] == qid), None)


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["everyday.py", str(config.OUT_DIR), *argv])
    return ev.main()


def test_a_rubric_change_and_a_remark_reach_both_halves_and_the_page(svc, monkeypatch):  # noqa: F811
    client, appmod = svc
    hq, pq = judged(ev.HIDDEN), judged(ev.PRACTICE)
    for q in (hq, pq):
        plant(q["id"], "an old verdict")
    future()
    new_rubric(monkeypatch, hq["id"], pq["id"])
    # the page as it was after the deploy, before the re-mark
    assert page_item(client, appmod, pq["id"])["reason"] == "an old verdict"
    assert run(monkeypatch, "--judge") == 0
    for q in (now(hq["id"]), now(pq["id"])):
        it = item(A, q["id"])
        assert it["rubric"] == ev.rubric_key(q) and it["reason"] != "an old verdict"
        want = ev.stub_for(q, it["answer_text"])
        assert (it["pass"], it["reason"]) == (want["pass"], want["reason"])
    # and the page shows the new marks, whatever the other files' times
    shown = page_item(client, appmod, pq["id"])
    assert (shown["reason"], shown["pass"]) == (item(A, pq["id"])["reason"], item(A, pq["id"])["pass"])
    # the question browser too
    got = client.get("/api/questions/everyday", params={"models": A, "limit": 200}).json()
    row = next(r for r in got["rows"] if r["id"] == pq["id"])
    assert row["results"][A] and "an old verdict" not in json.dumps(row["results"][A])


def test_the_pages_key_moves_when_any_watched_file_is_rewritten(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("1")
    b.write_text("2")
    t = time.time() + 86400
    os.utime(b, (t, t))                          # b dated a day ahead
    before = files_stamp([a, b])
    a.write_text("3")                            # rewritten now: still older than b
    assert files_stamp([a, b]) != before
    assert files_stamp([a, b]) == files_stamp([b, a])      # order doesn't matter
    assert files_stamp([a, b, tmp_path / "gone.json"]) == files_stamp([a, b])


def test_the_question_browsers_stamp_moves_too(svc):  # noqa: F811
    d = {A: _dir(A)}
    (_dir(A) / "judge.json").write_text("{}")
    t = time.time() + 86400
    os.utime(_dir(A) / "judge.json", (t, t))
    before = questions._stamp("mmlu", d)
    (_dir(A) / "generative.json").write_text("{}")
    assert questions._stamp("mmlu", d) != before


def test_question_sends_the_judge_one_questions_answers(svc, monkeypatch):  # noqa: F811
    client, appmod = svc
    q1, q2 = judged(ev.PRACTICE), judged(ev.HIDDEN)
    new_rubric(monkeypatch, q1["id"], q2["id"])
    assert run(monkeypatch, "--judge", "-q", q1["id"]) == 0
    assert item(A, q1["id"])["rubric"] == ev.rubric_key(now(q1["id"]))
    assert item(A, q1["id"])["pass"] is not None
    # the other waits for the judge; and no "before" is written for --compare
    assert item(A, q2["id"])["pass"] is None
    assert not (config.OUT_DIR / ev.BEFORE_NAME).exists()


def test_an_unknown_question_is_refused(svc, monkeypatch, capsys):  # noqa: F811
    assert run(monkeypatch, "--judge", "-q", "everyday-nope") == 2
    assert "no such question: everyday-nope" in capsys.readouterr().err


def test_the_school_run_plans_reference_is_hers_and_his():
    q = next(q for q in ev.load_bank() if q["id"] == SCHOOL_RUN)
    assert "I'll take Zain to football at 4" in q["prompt"]           # she says so
    assert q["reference"] == FIXED
    rubric = ev.judge_check(q)["rubric"]
    assert rubric == ev.summarise_rubric(q)
    assert f'A good summary, for reference: "{FIXED}"' in rubric and "you take Zain" not in rubric
    # the facts it asks for are as they were
    assert ev._facts_of(q) == ([["5:30", "5.30"], ["mum", "mother"], ["transit card"], ["cake"]], 4)


@pytest.mark.parametrize("fname", ["mobileaibench.json"])
def test_mobileaibench_scores_are_watched(fname):
    from service import app
    assert fname in app._WATCH
