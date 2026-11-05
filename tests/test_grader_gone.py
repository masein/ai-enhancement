"""A grader whose chosen model is gone holds its own benchmark alone
(masein, 5 Nov). HLE's judge, o3-mini-2025-01-31, was shut down by OpenAI on
23 Oct, and Start refused every grader for it — SimpleQA, MATH and OTIS too.
Start now sends the others; the gone grader's answers wait as they are, none
marked ungraded or counted wrong, its rows shown on the dry run as not sent
and left out of what Start costs; with nothing else to send, Start is refused
as before and a stop stays. Once a model is chosen for it again, the next
Start sends them. Invented questions and a stand-in for OpenRouter only:
nothing is sent anywhere."""

from __future__ import annotations

import pytest

from service import ai_models, config
from service import frontier as sf
from service import frontier_grade as fgr
from test_17_grading import ROW, drain, rec, results, svc  # noqa: F401 — svc is the fixture
from test_17b_grading import GPT, O3, grades, more_data, plain, stub, write_runs

GONE = "openai/o3-mini is no longer on OpenRouter's list — choose the job's model again on AI models"
JUDGE = "Humanity's Last Exam's judge"


def hle(monkeypatch):
    """four HLE answers waiting for the judge, the judge chosen and gone"""
    more_data(monkeypatch, {"hle_text_cais": [
        {"id": f"h{k}", "question": f"Hard question {k}?", "answer": f"K{k}",
         "answer_type": "exactMatch", "subject": "x"} for k in range(4)]})
    write_runs("hle_text_cais", [(f"h{k}", f"I believe K{k}.") for k in range(4)])
    fgr.save("hle", O3, "masein")
    monkeypatch.setattr(ai_models, "drifted",
                        lambda pin: GONE if (pin or {}).get("id") == O3 else "")


def test_start_sends_the_others_and_leaves_the_gone_judges_answers_as_they_are(
        svc, monkeypatch):  # noqa: F811
    hle(monkeypatch)
    asked = stub(monkeypatch, lambda m, r: ("correct: yes", "stop", "") if m != GPT
                 and "algebra/" not in r["custom_id"] else plain(m, r))
    # the dry run: HLE's row shown, not sent, and left out of what Start costs
    est = svc.get("/api/frontier/grading").json()["estimate"]
    row = next(r for r in est["rows"] if r["slot"] == "hle")
    assert row["answers"] == 4 and row["usd"] is None
    assert row["waits"] == (f"Not sent: {GONE}. Its 4 answers wait as they are — none marked "
                            "ungraded or counted wrong — and Start sends them once a model is "
                            "chosen for it again")
    assert "hle" not in est["graders"] and est["answers"] == 7        # SimpleQA 5, MATH 2
    got = fgr.start("masein")
    assert {x["slot"] for x in got["sent"]} == {"simpleqa", "math"}
    assert got["skipped"] == [f"{JUDGE}: {GONE}"]
    drain()
    assert {m for m, _ in asked} == {GPT, "google/gemini-2.5-flash"}  # nothing to the judge
    assert results("simpleqa_epoch") is not None
    # HLE: no grade, no no-grade, no try counted — its four answers wait
    g = grades("hle_text_cais")
    assert not g.get("items") and not g.get("refused")
    m = sf.marks(config.OUT_DIR / ROW, "hle_text_cais")
    flat = [r for rs in m["runs"].values() for r in rs]
    assert len(flat) == 4 and all(r["ok"] is None and not r.get("ungraded") for r in flat)
    assert sf.score_task(config.OUT_DIR / ROW, "hle_text_cais", rec()) == {
        "waiting": 4, "of": 4, "label": "Humanity's Last Exam"}
    assert results("hle_text_cais") is None
    # its model served again (as one chosen again would be): the next Start
    # sends them, from no tries
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    got = fgr.start("masein")
    assert [(x["slot"], x["n"]) for x in got["sent"]] == [("hle", 4)] and "skipped" not in got
    drain()
    assert sum(1 for m, _ in asked if m == O3) == 4
    r = results("hle_text_cais")
    assert r["results"]["hle_text_cais"]["acc,none"] == pytest.approx(1.0)


def test_with_only_the_gone_judge_to_send_start_is_refused_and_a_stop_stays(
        svc, monkeypatch):  # noqa: F811
    stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    hle(monkeypatch)
    fgr.stop("masein")
    with pytest.raises(ValueError) as e:
        fgr.start("masein")
    assert str(e.value) == f"{JUDGE}: {GONE}"
    assert fgr.stopped() and not fgr.pending()
    assert not grades("hle_text_cais").get("refused")
    # the card: nothing for Start to send, HLE's row said
    page = svc.get("/api/frontier/grading").json()
    assert page["estimate"]["answers"] == 0
    assert [r["slot"] for r in page["estimate"]["rows"]] == ["hle"]
