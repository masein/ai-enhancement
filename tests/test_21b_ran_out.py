"""21b: the report of the answers step4000 and step1000 ran out of room on
(docs/prompts/phase-21b-ran-out-of-room-report.md). Its counts are the board's;
"had the right answer before it ran out" reads clear statements only; the
hidden half is shown in this report alone — the question browser, the
diagnosis, 21's scorecard and a report file without the flag still refuse
it; GPQA Diamond's question text never appears; nothing here reaches a
training, generation or Improve path. Invented questions and answers; no
model runs, nothing is fetched or graded."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import frontier as fb
from conftest import make_service
from service import config, ran_out, reported, reports

ROOT = Path(__file__).resolve().parent.parent
REP = json.loads((reported.REPORTS_DIR / "ran-out-of-room.json").read_text())
MMLU, GPQA, MATH = "mmlupro_tiger", "gpqa_diamond_epoch", "math_l5_epoch"
OPTS = [f"invented option {c}" for c in "ABCDEFGHIJ"]
GPQA_Q = "An invented question about the spin of an imagined particle trapped in a box?"
ITEMS = {
    MMLU: [{"id": "486", "question": "Invented: how many bits does the invented code need?",
            "options": OPTS, "answer": "E", "category": "computer science"},
           {"id": "1", "question": "Invented: which invented clause of the imagined statute holds?",
            "options": OPTS, "answer": "B", "category": "law"},
           {"id": "3", "question": "Invented: which invented sum is the largest of them?",
            "options": OPTS, "answer": "C", "category": "math"}],
    GPQA: [{"id": "recQ1", "question": GPQA_Q, "right": "Spin one half",
            "wrong": ["Spin zero", "Spin one", "Spin three halves"]}],
    MATH: [{"id": "algebra/1", "question": "Invented: what is the sum of the roots?",
            "answer": "\\frac{1}{2}", "subject": "Algebra"}],
}
LOOP = "\n".join(["Wait, let me re-check option E against the question once more."] * 40)


def think(body: str, closed: bool = False, after: str = "") -> str:
    return "<think>\n" + body + ("\n</think>\n" + after if closed else "")


# the runs: (question, ran out, right, answer, tokens)
STEP4000 = {
    MMLU: [("486", True, False, think("Counting: 7 bits. So the answer is (E).\n" + LOOP), 32768),
           ("1", True, False, think("Hmm.\n" + LOOP), 32768),
           ("3", False, True, think("Sum.", True, "ANSWER: C"), 900)],
    GPQA: [("recQ1", True, False, think(GPQA_Q + "\nLet me think about the spin.\n" + LOOP),
            81920)],
    MATH: [("algebra/1", True, False, think("So \\boxed{\\frac{1}{2}} it is.\n" + LOOP), 65536)],
}
STEP1000 = {
    MMLU: [("486", True, False, think(LOOP), 32768), ("1", False, False,
                                                       think("x", True, "ANSWER: A"), 500),
           ("3", True, False, think("The answer is (A).\n" + LOOP), 32768)],
    GPQA: [("recQ1", False, True, think("y", True, "ANSWER: A"), 700)],
    MATH: [("algebra/1", True, False, think("Perhaps 2.\n" + LOOP), 65536)],
}
Q4 = {
    MMLU: [("486", False, True, think("7 bits.", True, "ANSWER: E"), 2500),
           ("1", False, False, think("b", True, "ANSWER: A"), 2400),
           ("3", False, True, think("c", True, "ANSWER: C"), 2300)],
    GPQA: [("recQ1", True, False, think("long"), 81920)],
    MATH: [("algebra/1", False, True, think("half", True, "\\boxed{\\frac{1}{2}}"), 3000)],
}


def row_dir(label: str) -> Path:
    o = next(x for x in REP["ours"] if x["label"] == label)
    return reports.row_dir(o["row"])


def write_row(label: str, runs: dict) -> None:
    from service import frontier as sf
    mid = next(x for x in REP["ours"] if x["label"] == label)["row"]
    for task, rs in runs.items():
        d = sf.task_dir(row_dir(label), task)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"samples_{task}_2026-10-14T07-20-00.000000.jsonl").write_text("".join(
            json.dumps({"doc_id": k, "doc": {"id": q}, "resps": [[a]],
                        "filtered_resps": ["" if out else (fb.visible(a)[-1:] or "")],
                        "frontier": [{"epoch": 0, "tokens": t, "ran_out": out, "ok": ok,
                                      "read": None if out else fb.visible(a)[-1:]}]}) + "\n"
            for k, (q, out, ok, a, t) in enumerate(rs)))
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": q, "epoch": 0, "answer": a, "finish": "length" if out else "stop",
             "tokens": t}) + "\n" for q, out, ok, a, t in rs))
        # the board's own count, as its results say it
        (d / "results_2026-10-14T07-20-00.000000.json").write_text(json.dumps({
            "results": {task: {"alias": task, "acc,none": 0.5, "acc_stderr,none": 0.1}},
            "config": {"model": "frontier",
                       "model_args": f"pretrained={mid.removesuffix(' · thinking')},"
                                     "enable_thinking=True"},
            "date": time.time(),
            "frontier": {"answers": len(rs), "ran_out": sum(1 for r in rs if r[1]),
                         "scored_by": "code"}}))


def world() -> None:
    for task, items in ITEMS.items():
        p = fb._cache(config.BENCH_ROOT, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"items": items, "extra": {"shots": {}}}))
    for label, runs in (("step4000", STEP4000), ("step1000", STEP1000),
                        ("Q4 (UD-Q4_K_XL)", Q4)):
        write_row(label, runs)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    world()
    yield client
    client.__exit__(None, None, None)


def page(svc) -> dict:
    """the report, its stats worked out"""
    reports.report("ran-out-of-room", compute=True)
    return svc.get("/api/reports/ran-out-of-room").json()


# ---------------------------------------------------------------------------
# the counts, and the right answer before the limit
# ---------------------------------------------------------------------------

def test_the_counts_are_the_boards_for_step4000_and_step1000(svc):
    got = page(svc)
    assert got["insightOnly"] and got["showHidden"] and got["decided"] == "masein, 14 Oct 2026"
    # 22: and Ornith-1.5, measured here on its card's settings, to set beside them
    assert got["picks"] == ["step4000", "step1000", "Ornith-1.5 Q4_K_M"]
    for label in ("step4000", "step1000"):
        for x in got["models"][label]["summary"]:
            if x["board"] is not None:
                assert x["n"] == x["board"], (label, x["key"], x["n"], x["board"])
    s4 = {x["key"]: x for x in got["models"]["step4000"]["summary"]}
    assert (s4["mmlupro"]["n"], s4["gpqa"]["n"], s4["math"]["n"]) == (2, 1, 1)
    assert s4["mmlupro"]["share"] == pytest.approx(2 / 3)
    # Q4 on the same questions: 486 right, 1 wrong; GPQA's it ran out on too
    assert s4["mmlupro"]["against"] == {"right": 1, "wrong": 1, "ran_out": 0, "of": 2}
    assert s4["gpqa"]["against"]["ran_out"] == 1
    assert sorted(s["subject"] for s in s4["mmlupro"]["subjects"]) == ["computer science", "law"]
    s1 = {x["key"]: x for x in got["models"]["step1000"]["summary"]}
    assert (s1["mmlupro"]["n"], s1["math"]["n"]) == (2, 1)


def test_had_the_right_answer_reads_clear_statements_only():
    item = ITEMS[MMLU][0]
    early = think("Counting: 7 bits. So the answer is (E).\n" + LOOP)
    at = ran_out.first_right(MMLU, item, early)
    assert at is not None and early[at:].lower().startswith("the answer is (e)")
    assert ran_out.first_right(MMLU, item, think("Hmm.\n" + LOOP)) is None     # never states one
    assert ran_out.first_right(MMLU, item, think("The answer is (A).\n" + LOOP)) is None
    # the right option's letter with its text
    assert ran_out.first_right(MMLU, item, "So (E) invented option E fits.") is not None
    # MATH and OTIS: the exact final answer
    m = ITEMS[MATH][0]
    assert ran_out.first_right(MATH, m, "So \\boxed{\\frac{1}{2}} it is.") is not None
    assert ran_out.first_right(MATH, m, "the answer is \\frac{1}{2}.") is not None
    assert ran_out.first_right(MATH, m, "the answer is \\frac{1}{3}") is None
    # never read on HLE, SimpleQA or ARC-AGI-2
    assert ran_out.first_right("hle_text_cais", {"answer": "x"}, "the answer is x") is None


def test_the_list_says_where_and_at_which_token(svc):
    rows = {(r["task"], r["id"]): r for r in page(svc)["models"]["step4000"]["rows"]}
    r = rows[(MMLU, "486")]
    assert r["had"] is True and 0 < r["at_token"] < 32768
    assert r["waits"] == 40 and r["line_n"] == 40 and r["line"].startswith("Wait, let me re-check")
    assert r["q"].startswith("Invented: how many bits") and r["right"] == "(E) invented option E"
    assert r["against"]["verdict"].startswith("read as")
    assert rows[(MMLU, "1")]["had"] is False
    assert rows[(MATH, "algebra/1")]["had"] is True


# ---------------------------------------------------------------------------
# the hidden half: this report alone
# ---------------------------------------------------------------------------

def test_this_report_lists_a_hidden_half_question_and_opens_it(svc):
    assert not reports.shown_half(MMLU, "1")
    rows = page(svc)["models"]["step4000"]["rows"]
    hidden = next(r for r in rows if r["id"] == "1")
    assert hidden["half"] == "hidden"
    got = svc.get("/api/reports/ran-out-of-room/answer", params={
        "model": "step4000", "task": MMLU, "id": "1", "epoch": 0}).json()
    assert got["question"]["q"].startswith("Invented: which invented clause")
    assert got["warning"] == ran_out.WARNING


def test_the_rest_of_the_board_still_refuses_it(svc, monkeypatch):
    import diagnose as dx
    # the question browser lists the shown half alone
    listed = svc.get(f"/api/questions/{MMLU}").json()
    keys = {str(r.get("key") or r.get("id")) for r in listed.get("rows") or []}
    assert "1" not in keys and "486" in keys
    # the diagnosis: its split, unchanged
    assert dx.split_of("1") == "report"
    # 21's scorecard
    sc = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())
    with pytest.raises(reports.Hidden):
        reports.loop_example(MMLU, "1", None, None)
    assert "refused" in reports.loops({**sc, "loops": {**sc["loops"], "examples": [
        {"id": "1", "note": ""}]}}, reports.our_rows(sc, {}))[0]
    # a report file without the flag
    blind = {k: v for k, v in REP.items() if k != "show_hidden"}
    got = ran_out.rows_of(blind, "step4000", MMLU)
    assert [r["id"] for r in got["rows"]] == ["486"] and got["hidden"] == 1
    with pytest.raises(ran_out.Refused):
        ran_out.one(blind, "step4000", MMLU, "1", 0)
    monkeypatch.setattr(reports, "files", lambda: {"ran-out-of-room": blind})
    r = svc.get("/api/reports/ran-out-of-room/answer", params={
        "model": "step4000", "task": MMLU, "id": "1", "epoch": 0})
    assert r.status_code == 403 and "hidden half" in r.json()["detail"]


# ---------------------------------------------------------------------------
# GPQA's question text, and what the report never offers
# ---------------------------------------------------------------------------

def test_gpqa_question_text_never_appears(svc):
    got = page(svc)
    assert GPQA_Q[:40] not in json.dumps(got)
    r = next(x for x in got["models"]["step4000"]["rows"] if x["task"] == GPQA)
    assert r["q"] is None and r["right"] is None and "authors ask" in r["withheld"]
    one = svc.get("/api/reports/ran-out-of-room/answer", params={
        "model": "step4000", "task": GPQA, "id": "recQ1", "epoch": 0}).json()
    blob = json.dumps(one)
    assert GPQA_Q[:40] not in blob and "spin of an imagined particle" not in blob
    assert ran_out.WITHHELD_LINE in one["thinking"] and one["question"] is None
    assert "Let me think about the spin." in one["thinking"]   # the rest of its thinking


def test_nothing_from_it_reaches_a_training_or_improve_path():
    for f in ("service/proposals.py", "service/builder.py", "scripts/diagnose.py",
              "scripts/everyday.py", "scripts/exam_build.py", "service/playground.py",
              "service/chat.py"):
        src = (ROOT / f).read_text()
        import re
        assert not re.search(r"\bimport\s+(?:ran_out|reports)\b|from\s+\S+\s+import\s+[^\n]*"
                             r"\b(?:ran_out|reports)\b|api/reports", src), f
    page_src = (ROOT / "scripts" / "report_lm_eval.py").read_text()
    ro = page_src[page_src.index("function vRanOut("):page_src.index("function benchLabel(")]
    for word in ("download", "export", "Blob(", "Improve", "propose", "training data"):
        assert word not in ro, word                              # no bulk download, no path out


def test_the_list_card_says_insight_only(svc):
    got = {r["id"]: r for r in svc.get("/api/reports").json()["reports"]}
    assert got["ran-out-of-room"]["insight"] is True
    assert got["ran-out-of-room"]["ours"] == ["step4000", "step1000", "Ornith-1.5 Q4_K_M"]
    assert not got["step-checkpoints"].get("insight")


def test_the_script_prints_the_top_numbers_without_a_question(svc, capsys):
    import report_ran_out as rro
    assert rro.main(["ran-out-of-room"]) == 0
    out = capsys.readouterr().out
    assert "  MMLU-Pro: 2 ran out (2) of 3, 66.7% · Q4 (UD-Q4_K_XL): 1 right, 1 wrong, 0 also ran " \
           "out · had it: 1 of 2" in out
    assert "Invented" not in out and GPQA_Q[:20] not in out
