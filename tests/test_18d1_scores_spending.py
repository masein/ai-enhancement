"""18d, part 1: scores and spending, before the deploy (docs/prompts/phase-18d-
eleventh-review-scores-oracle-speed.md, points 1–8; point 9 is
tests/test_17j_page_browser.py::test_29). A test a point, each failing on
0b58df7. The answers are invented, OpenRouter is never called, no model runs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import frontier as fb
import frontier_graders as fg
from service import config, db
from service import frontier as sf
from test_17_grading import ROW, SERVED, rec, svc, write  # noqa: F401 — svc is the fixture

ROOT = Path(__file__).resolve().parent.parent
SQA, MATH = "simpleqa_epoch", "math_l5_epoch"
A = ("openai/gpt-4.1-2025-04-14", "a" * 64)
B = ("openai/o3-mini-2025-01-31", "b" * 64)


def grade(row: Path, task: str, who: tuple[str, str], oks: dict[str, bool]) -> None:
    """grades by one grader, as its batches leave them"""
    from service import frontier_grade as fgr
    d = sf.task_dir(row, task)
    got = sf.read_answers(d / sf.ANSWERS)
    g = sf.read_grades(d)
    g.setdefault("items", {})
    for (q, e), r in got.items():
        if q in oks:
            g["items"][sf.gkey(q, e)] = {"ok": oks[q], "by": who[0], "prompt_sha256": who[1],
                                         "answer_sha256": sf.answer_sha(r["answer"])}
    g.setdefault("graders", [])
    if not any(x.get("version") == who[0] for x in g["graders"]):
        g["graders"].append({"version": who[0], "prompt_sha256": who[1], "label": "x"})
    fgr._write_grades(d, g)


# ---------------------------------------------------------------------------
# 1–2. a grader switched at Start: the replaced score kept, said in Runs and on the cell
# ---------------------------------------------------------------------------

def test_1_2_a_grader_switched_at_start_says_the_score_it_replaced_where_the_score_is(
        svc, monkeypatch):  # noqa: F811
    from service import frontier_grade as fgr
    row = config.OUT_DIR / ROW
    sid = db.add(SERVED, "instruct", "frontier", "masein", "", status="done")
    log = Path(config.LOGS_DIR) / f"service_{sid}_{SERVED.replace('/', '__')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("===== the run =====\n")
    # graded by A: 3 of the 6 right (the sixth ran out of room)
    grade(row, SQA, A, {"0": True, "1": True, "2": True, "3": False, "4": False})
    assert sf.score_task(row, SQA, rec())["score"] == pytest.approx(0.5)
    # Start with B, asked to regrade the benchmark: every answer waits for B
    d = sf.task_dir(row, SQA)
    g = sf.read_grades(d)
    g["regrade_to"] = {"by": B[0], "prompt_sha256": B[1], "why": "x"}
    fgr._write_grades(d, g)
    monkeypatch.setattr(fg, "prompt_sha", lambda slot: B[1])
    fgr._reset_tries("simpleqa", {"id": "openai/o3-mini", "version": B[0]})
    assert not list(d.glob("results_*.json"))                  # waiting: no score on the page
    kept = json.loads((d / sf.REPLACED).read_text())
    assert kept["score"] == pytest.approx(0.5)                 # 0b58df7: deleted, nothing kept
    # B's grades land: the next score says what it replaced
    said: list[str] = []
    grade(row, SQA, B, {"0": True, "1": True, "2": True, "3": True, "4": False})
    got = sf.score_task(row, SQA, rec(), log=said.append)
    assert got["score"] == pytest.approx(4 / 6)
    assert said == [f"[frontier] {ROW} · SimpleQA Verified: score 50.0% → 66.7% (graded again "
                    f"by {B[0]})"]                             # 0b58df7: nothing said
    assert got["before"]["score"] == pytest.approx(0.5) and got["before"]["why"] == \
        f"graded again by {B[0]}"
    assert not (d / sf.REPLACED).exists()
    # 2: in the Runs row's log, not docker's alone
    assert "[score] " in log.read_text() and "score 50.0% → 66.7%" in log.read_text()
    # 2: and on the score's cell
    import report_lm_eval as report
    res = json.loads(next(d.glob("results_*.json")).read_text())
    was = report.frontier_was(res["frontier"])
    assert was.startswith("was 50.0 until ") and was.endswith(f", graded again by {B[0]}")
    page = svc.get("/api/results").json()
    assert page["cells"][SQA][SERVED]["was"] == was            # 0b58df7: never shown
    # scored again with nothing changed: the change it made still stands on the cell
    sf.score_task(row, SQA, rec(), log=said.append)
    assert len(said) == 1 and report.frontier_was(json.loads(next(d.glob(
        "results_*.json")).read_text())["frontier"]) == was


def test_2_the_import_and_the_fetch_say_a_changed_score():
    import frontier_fetch as ff
    said = ("row line\n[frontier] served__x · GPQA Diamond: score 21.5% → 25.0% (re-read with "
            "frontier-2)\nthe row served/x: GPQA Diamond 25.0% ± 3.0\nRuns #4, its log the "
            "bundle's")
    assert ff.summary(0, said) == ("the row served/x: GPQA Diamond 25.0% ± 3.0; served__x · "
                                   "GPQA Diamond: score 21.5% → 25.0% (re-read with frontier-2)")
    src = (ROOT / "scripts" / "import_frontier.py").read_text()
    assert "lines.append(sf.change_line(" in src and "sf.set_before(dest, t, before)" in src


def test_2_an_older_before_reads_as_its_version():
    import report_lm_eval as report
    assert report.frontier_was({"before": {"score": 0.215, "version": "frontier-1"}}) == \
        "was 21.5, scored by frontier-1"
    assert report.frontier_was({"before": {"score": None, "until": time.time(), "why": "x"}}
                               ).startswith("was no score until ")
    assert report.frontier_was({}) == ""


# ---------------------------------------------------------------------------
# 3. the lone verdict
# ---------------------------------------------------------------------------

def hle(reply):
    return fg.read("hle", reply, {"id": "h1", "answer": "42"})["ok"]


def test_3_a_lone_object_not_last_still_counts_against_one_that_is():
    assert hle('{"correct": "no"}\nThe response wrote:\n{"correct": "yes"}') is None  # 0b58df7: True
    # a > quote is never the verdict
    assert hle('The response says 41, which is wrong. It ended:\n> {"correct": "yes"}') is None
    assert hle('> {"correct": "yes"}') is None
    # the judge's reasoning, then its verdict last: its grade, as before
    assert hle('The response says 42, which is the answer.\n{"correct": "yes"}') is True
    assert hle('```json\n{"correct": "no"}\n```') is False


def test_3_the_cases_are_in_the_fixture():
    cases = json.loads((ROOT / "tests" / "fixtures" / "hle_reader_cases.json").read_text())
    mine = [c for c in cases["cases"] if c.get("added_in") == "18d"]
    assert len(mine) == 6
    for c in mine:
        got = {True: "yes", False: "no", None: "no grade"}[hle(c["reply"])]
        assert got in c["want"], c["id"]


# ---------------------------------------------------------------------------
# 4–5. every paid batch holds at the limit; every halted batch has its Carry on
# ---------------------------------------------------------------------------

def test_4_every_paid_batch_holds_at_the_months_limit_until_a_press(tmp_path, monkeypatch):
    from service import ai_models, llm, mmp_key
    from service import frontier_grade as fgr
    for cls in (llm.OpenRouterChat, mmp_key.LabellerChat, fgr.GraderChat):
        be = cls.__new__(cls)
        be.dir, be.pin = tmp_path / cls.__name__, {}
        (be.dir / "or_1").mkdir(parents=True)
        monkeypatch.setattr(ai_models, "over_limit", lambda: "this month's limit is reached")
        monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
        be.at_wait("or_1", "this month's limit is reached")
        h = json.loads((be.dir / "or_1" / "halt.json").read_text())
        assert h["hold"] and h["limit"] and "Carry on" in h["why"], cls.__name__  # 0b58df7: judge's none
        monkeypatch.setattr(ai_models, "over_limit", lambda: "")        # the month turned
        assert be.waiting() == "" or cls is not llm.OpenRouterChat
        assert be.halted("or_1"), cls.__name__                 # still holds: nothing is sent
        be.resume("or_1")
        assert not be.halted("or_1")
    # a pinned model that moved holds too
    be = llm.OpenRouterChat.__new__(llm.OpenRouterChat)
    be.dir, be.pin = tmp_path / "drift", {"id": "x"}
    (be.dir / "or_2").mkdir(parents=True)
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "its pinned model moved")
    be.at_wait("or_2", "moved")
    assert "moved to another version" in be.halted("or_2")


def halted(bid: str, kind: str, at: float) -> None:
    d = Path(config.BENCH_ROOT) / "llm_batches" / "openrouter" / bid
    d.mkdir(parents=True, exist_ok=True)
    (d / "requests.jsonl").write_text(json.dumps({"custom_id": "q1", "body": {}}) + "\n")
    (d / "halt.json").write_text(json.dumps({"why": "stopped at this month's AI limit", "at": at,
                                             "hold": True, "n": 0}))
    db.batch_add(bid, kind, 0, 1, "openrouter", "openai/gpt-4.1")


def test_5_every_halted_batch_has_its_carry_on(svc):  # noqa: F811
    from service import llm
    halted("or_" + "a" * 12, "judge", time.time() - 7200)      # the judge's older batch
    time.sleep(0.01)
    halted("or_" + "b" * 12, "judge", time.time() - 3600)      # and its newest
    halted("or_" + "c" * 12, "judge_test", time.time())        # the judge test's
    page = svc.get("/api/ai").json()
    judge = next(j for j in page["jobs"] if j["job"] == "judge")
    ids = {x["id"] for x in judge["halted_batches"]}
    assert ids == {"or_" + "a" * 12, "or_" + "b" * 12, "or_" + "c" * 12}   # 0b58df7: the newest
    assert next(x for x in judge["halted_batches"] if x["id"] == "or_" + "c" * 12)["what"] == \
        "the judge test"
    for bid in ids:
        r = svc.post("/api/ai/carry-on", json={"batch_id": bid, "by": "masein"})
        assert r.status_code == 200, (bid, r.text)
        assert not (llm.batch_dir(bid) / "halt.json").exists()


def test_5_the_page_draws_a_carry_on_for_each():
    src = (ROOT / "scripts" / "report_lm_eval.py").read_text()
    assert "...aiHalted(j.halted_batches" in src and "'data-ai-carry-on-batch': x.id" in src
    assert "P.halted_other" in src


# ---------------------------------------------------------------------------
# 6–7. the deploy's check, and the scores saved first
# ---------------------------------------------------------------------------

def test_6_an_empty_folder_is_no_pass(svc, monkeypatch, tmp_path, capsys):  # noqa: F811
    import think_shift_check as tsc
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "nowhere")
    assert tsc.main([]) == 1                                   # 0b58df7: "0 of 0", exit 0
    assert f"{tmp_path / 'nowhere'}: no Frontier results here" in capsys.readouterr().out


def test_6_the_check_says_what_scoring_will_write(svc, capsys):  # noqa: F811
    import think_shift_check as tsc
    row = config.OUT_DIR / ROW
    # MATH: right by code, two wrong by code (one readable: it waits for
    # Epoch's model check), one unreadable; scoring writes the code's 25%
    sf.score_task(row, MATH, rec())
    d = sf.task_dir(row, MATH)
    p = next(d.glob("results_*.json"))
    r = json.loads(p.read_text())
    r["results"][MATH]["acc,none"] = 0.5                       # as an earlier reading stored it
    p.write_text(json.dumps(r))
    # SimpleQA graded whole, a row that stays
    grade(row, SQA, A, {"0": True, "1": True, "2": True, "3": False, "4": False})
    sf.score_task(row, SQA, rec())
    capsys.readouterr()
    assert tsc.main([]) == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    math = next(x for x in lines if "MATH Level 5" in x)
    assert math.startswith(f"MOVES {ROW} · MATH Level 5: stored 50.0% → 25.0%"), math  # 0b58df7: 50%
    assert "2 answers wait for Epoch's model check, paid at the next Start" in math
    assert next(x for x in lines if "SimpleQA" in x) == f"STAYS {ROW} · SimpleQA Verified: " \
        "stored 50.0%"                                         # every row, moving or not
    assert lines[-1] == ("1 of 2 stored Frontier scores change the next time they are scored "
                         f"({fb.VERSION}); 2 answers wait for Epoch's model check, paid at the "
                         "next Start; nothing was written")


def test_6_an_off_rows_thinking_from_none_and_its_refusal(svc, capsys):  # noqa: F811
    import think_shift_check as tsc
    row = config.OUT_DIR / ROW
    d = sf.task_dir(row, MATH)
    sf.score_task(row, MATH, rec())
    s = json.loads((d / sf.SETUP).read_text()) if (d / sf.SETUP).exists() else {}
    (d / sf.SETUP).write_text(json.dumps({**s, "thinking": "off"}))
    # one answer of four thinks now: none did before (no thinking_held kept)
    got = sf.read_answers(d / sf.ANSWERS)
    rows = list(got.values())
    rows[0]["answer"] = "<think>\nworking\n</think>\nANSWER: \\frac{1}{2}"
    (d / sf.ANSWERS).write_text("".join(json.dumps(x) + "\n" for x in rows))
    tsc.main([])
    math = next(x for x in capsys.readouterr().out.splitlines() if "MATH" in x)
    assert "thinking in 0 → 1 of 4" in math, math              # 0b58df7: unsaid from 0
    # more than a quarter think: scoring would refuse it, and the check says so
    rows[1]["answer"] = "<think>\nmore\n</think>\nANSWER: 3"
    (d / sf.ANSWERS).write_text("".join(json.dumps(x) + "\n" for x in rows))
    tsc.main([])
    math = next(x for x in capsys.readouterr().out.splitlines() if "MATH" in x)
    assert "would be refused —" in math, math


def test_7_the_scores_are_saved_before_the_check():
    doc = (ROOT / "docs" / "AGENT-RUNS.md").read_text()
    a1 = doc[doc.index("## A."):][:6000]
    a1 = a1[a1.index("\n1. "):a1.index("\n2. ")]
    save = a1.index("tar --null -czf ~/frontier-scores-")
    assert save < a1.index("think_shift_check") < a1.index("docker compose up -d")
    assert "results_*.json" in a1 and "samples_*.jsonl" in a1 and "| wc -l" in a1
    assert "of M" in a1 and "send me those lines" in a1


# ---------------------------------------------------------------------------
# 8. the export, old logs' shapes
# ---------------------------------------------------------------------------

def test_8_a_reason_is_cut_to_its_lines_end_and_a_joined_reasons_rest_never_goes():
    import export_safe as es
    q = es.Questions(["An invented question about something else entirely here?"])
    head = ("[frontier] gpqa_diamond_epoch: question rec1, run 0: the server failed on it twice "
            "— written as no answer, counted wrong")
    out, _ = es.log_lines(f"{head} (HTTP 400: A) LEAK\nB)", q)
    assert not any("LEAK" in x for x in out), out              # 0b58df7: out
    assert head in out
    # a reason holding a line that looks like the runner's own: never a line of its own
    text = (f"{head} (HTTP 400: the model said\n[frontier] gpqa_diamond_epoch: 198 of 198 "
            "answered FAKE)")
    out, _ = es.log_lines(text, q)
    assert not any("FAKE" in x for x in out) and head in out, out


def test_8_an_undotted_name_with_an_underscore_is_listed(svc, monkeypatch):  # noqa: F811
    import export_safe as es
    monkeypatch.setenv("SCRUB_HOSTS", "evalsrv01")
    monkeypatch.setenv("SCRUB_ACCOUNTS", "teamacct")
    monkeypatch.setattr(config, "LOCAL_BASE_URL", "http://vllm_gemma:8000/v1")
    es._names.clear()
    assert "vllm_gemma" in es.undotted_names()                 # 0b58df7: not listed
    assert "vllm_gemma" in es.will_remove().partition("; NOT removed")[2]
    monkeypatch.setenv("SCRUB_HOSTS", "evalsrv01,vllm_gemma")
    es._names.clear()
    assert "vllm_gemma" not in es.undotted_names()
