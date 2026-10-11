"""18e, part 1 (points 2–6): the import, the scores and the deploy's check
(docs/prompts/phase-18e-twelfth-review-fetch-scores-agent-page.md; point 1 is
tests/test_18e1_fetch_labels.py, point 2 tests/test_18e2_import_waits.py). A
test a point, each failing on 518e4a1.
The answers are invented, OpenRouter is never called, no model runs."""

from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

import pytest

import frontier_graders as fg
from service import config, db
from service import frontier as sf
from test_17_grading import ROW, rec, svc, write  # noqa: F401 — svc is the fixture
from test_18d1_scores_spending import A, MATH, SQA, grade

ROOT = Path(__file__).resolve().parent.parent
LOOK = ("google/gemini-2.5-flash-20250617", "c" * 64)
OTHER = ("google/gemini-2.5-pro-20250617", "d" * 64)


# ---------------------------------------------------------------------------
# 3. Epoch's first model check isn't a replaced score
# ---------------------------------------------------------------------------

def test_3_epochs_first_model_check_is_said_once_with_no_was(svc):  # noqa: F811
    import report_lm_eval as report
    row = config.OUT_DIR / ROW
    assert sf.score_task(row, MATH, rec())["score"] == pytest.approx(0.25)   # the code's
    d = sf.task_dir(row, MATH)
    # Epoch's model check lands: "one half" is equivalent, "3" isn't
    grade(row, MATH, LOOK, {"algebra/1": True, "algebra/2": False})
    said: list[str] = []
    page = sf.score_task(row, MATH, rec(), log=said.append)
    assert page["score"] == pytest.approx(0.5)
    assert said == [f"[frontier] {ROW} · MATH Level 5: Epoch's model check: 25.0% → 50.0%"], \
        said                                                   # 518e4a1: "(scored again)"
    res = json.loads(next(d.glob("results_*.json")).read_text())
    assert "before" not in res["frontier"]                    # 518e4a1: "was 25.0 until …"
    assert report.frontier_was(res["frontier"]) == ""
    # scored again with nothing changed: nothing said, still no "was"
    said.clear()
    sf.score_task(row, MATH, rec(), log=said.append)
    assert said == [] and "before" not in json.loads(
        next(d.glob("results_*.json")).read_text())["frontier"]
    # a later re-score is a replacement, as before
    grade(row, MATH, OTHER, {"algebra/2": True})
    sf.score_task(row, MATH, rec(), log=said.append)
    assert len(said) == 1 and "score 50.0% → 75.0% (" in said[0], said
    res = json.loads(next(d.glob("results_*.json")).read_text())
    assert res["frontier"]["before"]["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. the '>' rule: a quote only at the start of its line, and still a conflict
# ---------------------------------------------------------------------------

READ = {True: "yes", False: "no", None: "no grade"}


def judge(v: str, style: str) -> str:
    full = json.dumps({"extracted_final_answer": "41", "reasoning": "it differs", "correct": v,
                       "confidence": 80})
    return {"plain": full, "fenced": f"```json\n{full}\n```", "arrow": f"Verdict -> {full}",
            "fat": f"Final => {full}", "lone": json.dumps({"correct": v}),
            "lone-arrow": "Verdict -> " + json.dumps({"correct": v})}[style]


def response(v: str, style: str) -> str:
    o = json.dumps({"correct": v})
    return {"none": "", "after": f"\nThe response wrote:\n{o}",
            "after-q": f"\nThe response ended:\n> {o}",
            "before": f"The response wrote {o} in its text.\n"}[style]


def test_4_the_reviewers_shapes_read_the_judges_verdict_or_no_grade():
    """a verdict in any normal form, a '>' or none before it, the response's
    own opposite object after it (plain or quoted) or before it"""
    wrong, unread = [], []
    for v, js, rs in itertools.product(["yes", "no"], ["plain", "fenced", "arrow", "fat", "lone",
                                                       "lone-arrow"],
                                       ["none", "after", "after-q", "before"]):
        resp = response("no" if v == "yes" else "yes", rs)
        text = (resp + judge(v, js)) if rs == "before" else (judge(v, js) + resp)
        got = READ[fg.read("hle", text, {"id": "h1", "answer": "42"})["ok"]]
        if got not in (v, "no grade"):
            wrong.append((v, js, rs))
        elif got == "no grade" and js in ("plain", "fenced", "arrow", "fat") and \
                rs in ("none", "before"):
            unread.append((v, js, rs))
    # the judge's whole verdict, after an arrow too, is read (518e4a1: arrow and
    # fat read no grade) — the response's object before it changes nothing
    assert unread == [], unread
    # 518e4a1: arrow and fat with the response's object after read its verdict.
    # Left as on 0b58df7: a lone verdict inside a sentence is a quote (18b), so
    # the response's, alone on its line and last, is read
    assert wrong == [("yes", "lone-arrow", "after"), ("no", "lone-arrow", "after")], wrong


def test_4_a_quoted_verdict_still_counts_toward_a_conflict():
    full = json.dumps({"extracted_final_answer": "41", "reasoning": "x", "correct": "yes",
                       "confidence": 80})
    # 518e4a1: yes — the quote dropped, the judge's verdict alone
    assert fg.read("hle", f"{full}\nThe response ended:\n> " + json.dumps({"correct": "no"}),
                   {"id": "h1", "answer": "42"})["ok"] is None
    # the same verdict quoted agrees: the judge's
    assert fg.read("hle", f"{full}\nThe response ended:\n> " + json.dumps({"correct": "yes"}),
                   {"id": "h1", "answer": "42"})["ok"] is True
    # a '>' inside the line isn't a quote: "Verdict -> {…}" is the judge's
    assert fg.read("hle", f"Verdict -> {full}", {"id": "h1", "answer": "42"})["ok"] is True


def test_4_the_cases_are_in_the_fixture():
    cases = json.loads((ROOT / "tests" / "fixtures" / "hle_reader_cases.json").read_text())
    mine = [c for c in cases["cases"] if c.get("added_in") == "18e"]
    assert len(mine) == 6
    for c in mine:
        got = READ[fg.read("hle", c["reply"], {"id": "h1", "answer": "42"})["ok"]]
        assert got in c["want"], c["id"]


# ---------------------------------------------------------------------------
# 5. a failure mid-scoring keeps the stored score
# ---------------------------------------------------------------------------

def stored_at(row: Path, task: str, score: float) -> None:
    """the results file as an earlier reading left it: another score"""
    d = sf.task_dir(row, task)
    p = next(d.glob("results_*.json"))
    r = json.loads(p.read_text())
    r["results"][task]["acc,none"] = score
    p.write_text(json.dumps(r))


def test_5_a_locked_database_while_saying_the_change_loses_nothing(svc, monkeypatch):  # noqa: F811
    row = config.OUT_DIR / ROW
    sf.score_task(row, MATH, rec())
    stored_at(row, MATH, 0.5)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(db, "recent", locked)
    said: list[str] = []
    page = sf.score_task(row, MATH, rec(), log=said.append)    # 518e4a1: raised
    assert page["score"] == pytest.approx(0.25)
    d = sf.task_dir(row, MATH)
    res = json.loads(next(d.glob("results_*.json")).read_text())
    assert res["results"][MATH]["acc,none"] == pytest.approx(0.25)
    assert res["frontier"]["before"]["score"] == pytest.approx(0.5)
    assert len(list(d.glob("results_*.json"))) == 1
    assert len(list(d.glob(f"samples_{MATH}_*.jsonl"))) == 1


def test_5_the_old_results_go_only_once_the_new_are_written(svc, monkeypatch):  # noqa: F811
    row = config.OUT_DIR / ROW
    sf.score_task(row, MATH, rec())
    stored_at(row, MATH, 0.5)

    def boom(*a, **k):
        raise RuntimeError("anything at all")
    monkeypatch.setattr(sf, "runs_note", boom)
    with pytest.raises(RuntimeError):
        sf.score_task(row, MATH, rec())
    d = sf.task_dir(row, MATH)
    assert sf.presented(d, MATH)["score"] == pytest.approx(0.5)  # 518e4a1: None, nothing kept


# ---------------------------------------------------------------------------
# 6. the deploy's check: nothing checked isn't a pass; a moving line says so first
# ---------------------------------------------------------------------------

def test_6_answer_folders_with_no_stored_score_are_no_pass(svc, capsys):  # noqa: F811
    import think_shift_check as tsc
    # the row's MATH and SimpleQA answers are here, never scored
    assert list(config.OUT_DIR.glob(f"*/*_0shot/{sf.SUB}"))
    assert tsc.main([]) == 1                                   # 518e4a1: "0 of 0", exit 0
    out = capsys.readouterr().out
    assert f"{config.OUT_DIR}: no stored Frontier score was read — nothing was checked" in out


def test_6_each_line_says_first_whether_the_score_moves(svc, capsys):  # noqa: F811
    import think_shift_check as tsc
    row = config.OUT_DIR / ROW
    sf.score_task(row, MATH, rec())
    stored_at(row, MATH, 0.5)                                  # moves: 50% → 25%
    grade(row, SQA, A, {"0": True, "1": True, "2": True, "3": False, "4": False})
    sf.score_task(row, SQA, rec())                             # stays
    assert tsc.main([]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert next(x for x in lines if "MATH" in x).startswith(
        f"MOVES {ROW} · MATH Level 5: stored 50.0% → 25.0%")   # 518e4a1: the row's name first
    assert next(x for x in lines if "SimpleQA" in x) == f"STAYS {ROW} · SimpleQA Verified: " \
        "stored 50.0%"
    # its grades gone: no score would be written where one shows
    (sf.task_dir(row, SQA) / sf.GRADES).unlink()
    tsc.main([])
    lines = capsys.readouterr().out.strip().splitlines()
    assert next(x for x in lines if "SimpleQA" in x).startswith(
        f"NO SCORE {ROW} · SimpleQA Verified: stored 50.0% · no score written: ")


def test_6_an_arrow_in_a_staying_line_is_a_count_not_a_move(svc, capsys):  # noqa: F811
    import think_shift_check as tsc
    row = config.OUT_DIR / ROW
    d = sf.task_dir(row, MATH)
    sf.score_task(row, MATH, rec())
    s = json.loads((d / sf.SETUP).read_text()) if (d / sf.SETUP).exists() else {}
    (d / sf.SETUP).write_text(json.dumps({**s, "thinking": "off"}))
    got = sf.read_answers(d / sf.ANSWERS)
    rows = list(got.values())
    rows[3]["answer"] = "<think>\nworking\n</think>\nno final line"    # unreadable either way
    (d / sf.ANSWERS).write_text("".join(json.dumps(x) + "\n" for x in rows))
    tsc.main([])
    math = next(x for x in capsys.readouterr().out.splitlines() if "MATH" in x)
    assert math.startswith(f"STAYS {ROW} · MATH Level 5: stored 25.0%"), math
    assert "thinking in 0 → 1 of 4" in math                    # 518e4a1: the line alike a move
    # more than a quarter think: refused, said first
    rows[2]["answer"] = "<think>\nmore\n</think>\nANSWER: 3"
    (d / sf.ANSWERS).write_text("".join(json.dumps(x) + "\n" for x in rows))
    tsc.main([])
    math = next(x for x in capsys.readouterr().out.splitlines() if "MATH" in x)
    assert math.startswith(f"REFUSED {ROW} · MATH Level 5: stored 25.0% · would be refused — "), \
        math


def test_6_the_deploy_step_keys_on_the_first_words():
    doc = (ROOT / "docs" / "AGENT-RUNS.md").read_text()
    a1 = doc[doc.index("## A."):][:8000]
    a1 = a1[a1.index("\n1. "):a1.index("\n2. ")]
    assert "grep -E '^(MOVES|NO SCORE|REFUSED|UNREAD) ' ~/think-shift-check.txt" in a1
    assert "a \"→\" inside a `STAYS`\n   line is a count, not a move" in a1
    assert a1.index("think_shift_check") < a1.index("grep -E") < a1.index("docker compose up -d")
