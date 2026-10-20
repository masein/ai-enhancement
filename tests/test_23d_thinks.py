"""23d: how it thinks (service/thinks.py) — where the thinking first states
the right answer, the tokens after it, the checking share, "had it, then
lost it", ran out with it stated and second thoughts, each worked by hand on
invented answers; the ran-out report's number given again; the notes as the
brief words them. Nothing is run, fetched or graded."""

from __future__ import annotations

import json

import pytest

import frontier as fb
from conftest import make_service
from service import config, ran_out, report_stats, reports, thinks
from service import frontier as sf

MMLU = "mmlupro_tiger"
OPTS = ["alpha one", "beta two", "gamma three", "delta four"]
LOOP = "\n".join(["Let me look at the options once more before I answer."] * 30)


def item(qid: str) -> dict:
    return {"id": qid, "question": f"Invented question {qid}?", "options": OPTS, "answer": "B",
            "category": "invented"}


def think(body: str, reply: str = "", closed: bool = True) -> str:
    return "<think>" + body + ("</think>\n\n" + reply if closed else "")


def write(d, task: str, runs: list[tuple[str, bool, object, str, int]], cache: list[str]) -> None:
    """runs as (question, ran out, ok, text, tokens), one run a question; the
    questions in `cache` in this machine's question cache"""
    p = fb._cache(config.BENCH_ROOT, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"items": [item(q) for q in cache], "extra": {"shots": {}}}))
    t = sf.task_dir(d, task)
    t.mkdir(parents=True, exist_ok=True)
    by: dict[str, list] = {}
    for q, out, ok, text, n in runs:
        by.setdefault(q, []).append((out, ok, text, n))
    (t / f"samples_{task}_2026-10-21T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": q}, "resps": [[x[2]] for x in rs],
                    "filtered_resps": [""] * len(rs),
                    "frontier": [{"epoch": e, "ok": x[1], "ran_out": x[0], "tokens": x[3]}
                                 for e, x in enumerate(rs)]}) + "\n"
        for k, (q, rs) in enumerate(by.items())))


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield client
    client.__exit__(None, None, None)


# r1: stated at a known character — "x"×200, "So the answer is (B).", "y"×299:
# the statement found where "the answer" begins, 203 of the thinking's 520;
# its reply "ANSWER: B" 9 more
R1 = think("x" * 200 + "So the answer is (B)." + "y" * 299, "ANSWER: B")
R5 = think("So the answer is (B)." + "y" * 79, "ANSWER: B")          # at 3 of 100
NEVER = think("z" * 100, "ANSWER: B")
REPLY_ONLY = think("w" * 100, "The answer is (B).\nANSWER: B")
WEIGHED = think("Hmm, maybe (B)? Or (C)? Let me weigh them.", "ANSWER: B")


def test_first_stated_as_worked_by_hand():
    it = item("q1")
    got = thinks.one(MMLU, it, R1, 1000)
    # position: 203 / 520; after: (520 − 203 + 9) / 529 of 1,000 tokens = 616.26 → 616
    assert got["stated"] and got["pos"] == pytest.approx(203 / 520)
    assert got["after_share"] == pytest.approx(326 / 529)
    assert (got["after"], got["before"]) == (616, 384)
    assert got["think_tokens"] == round(1000 * 520 / 529) == 983
    for text in (NEVER, REPLY_ONLY, WEIGHED):                 # never; in the reply only; weighed
        x = thinks.one(MMLU, it, text, 500)
        assert not x["stated"] and "pos" not in x
    assert ran_out.first_right(MMLU, it, REPLY_ONLY) is not None   # the whole text would have it


def test_the_shares_as_worked_by_hand(svc):
    d = reports.row_dir("served/step4000-lda-2610 · thinking")
    runs = [("q1", False, True, R1, 1000), ("q2", False, True, NEVER, 500),
            ("q3", False, True, REPLY_ONLY, 500), ("q4", False, True, WEIGHED, 500),
            ("q5", False, True, R5, 218),
            # wrong: one had stated (B) first, one stated another, one stated none
            ("q6", False, False, think("So the answer is (B). But no, it can't be.", "ANSWER: C"),
             300),
            ("q7", False, False, think("The answer is (C).", "ANSWER: C"), 300),
            ("q8", False, False, think("hmm", "ANSWER: D"), 300),
            # ran out: one had stated it before the limit, one never
            ("q9", True, False, think("The answer is (B). " + LOOP, closed=False), 32768),
            ("q10", True, False, think(LOOP, closed=False), 32768),
            ("q11", False, None, think("So the answer is (B).", "ANSWER: B"), 100),  # ungraded
            ("q99", False, True, R1, 1000), ("q99", False, False, NEVER, 500)]      # not cached
    write(d, MMLU, runs, [f"q{k}" for k in range(1, 12)])
    got = thinks.measures(reports.samples_file(d, MMLU), MMLU)
    assert (got["answers"], got["ungraded"], got["no_item"], got["no_item_answers"]) == (
        13, 1, 1, 2)
    r = got["right"]
    # stated in the thinking: q1 and q5 of the five right
    assert (r["n"], r["stated"], r["stated_share"]) == (5, 2, 0.4)
    # q5: at 3 of 100, after (100 − 3 + 9) / 109 of 218 = 212, before 6
    assert r["pos_median"] == pytest.approx((203 / 520 + 3 / 100) / 2)
    assert (r["after_median"], r["before_median"]) == ((616 + 212) / 2, (384 + 6) / 2)
    assert r["checking"] == pytest.approx((326 / 529 + 106 / 109) / 2)
    assert got["wrong"] == {"n": 3, "lost": 1, "lost_share": pytest.approx(1 / 3)}
    assert got["ran_out"] == {"n": 2, "stated": 1, "stated_share": 0.5}
    # kept by the worker, read from what it kept
    assert report_stats.work(say=lambda s: None)["failed"] == 0
    kept = thinks.kept(d, MMLU)
    assert kept["fresh"] and kept["right"] == got["right"]


SECOND_TEXT = ("First step. Wait, check it.\nHmm, the sum is 4. I will wait here. It was Actually "
               "fine.\n- Actually, no. Alternatively we try 5? Wait! Done. Waiting helps.")


def test_second_thoughts_at_a_sentence_start_only(svc):
    # Wait twice, Hmm, Actually and Alternatively once each at a sentence's start; "wait",
    # "was Actually" and "Waiting" not
    assert thinks.second_thoughts(SECOND_TEXT) == {"Wait": 2, "Alternatively": 1, "Hmm": 1,
                                                   "Actually": 1}
    d = reports.row_dir("served/step4000-lda-2610 · thinking")
    write(d, MMLU, [("q1", False, True, think(SECOND_TEXT), 500),                # all thinking
                    ("q2", False, False, think("Wait. Wait.", "ANSWER: C"), 400)],
          ["q1", "q2"])
    s = thinks.measures(reports.samples_file(d, MMLU), MMLU)["second"]
    # right: 5 over 500 thinking tokens = 10.0 per 1,000; Wait 4.0
    assert s["right"]["think_tokens"] == 500 and s["right"]["all_per_1k"] == pytest.approx(10.0)
    assert s["right"]["per_1k"] == {"Wait": 4.0, "Alternatively": 2.0, "Hmm": 2.0,
                                    "Actually": 2.0}
    # wrong: its thinking 11 of 20 characters of 400 tokens = 220; 2 Waits = 9.09 per 1,000
    assert s["wrong"]["think_tokens"] == 220
    assert s["wrong"]["all_per_1k"] == pytest.approx(2000 / 220)
    assert thinks.SECOND == ("Wait", "Alternatively", "Hmm", "Actually")


def test_ran_out_with_it_stated_agrees_with_the_ran_out_report(svc):
    from test_21b_ran_out import world
    world()
    got = reports.report("ran-out-of-room", compute=True)
    checked = 0
    for label in ("step4000", "step1000"):
        rep = reports.files()["ran-out-of-room"]
        mid = next(o["row"] for o in rep["ours"] if o["label"] == label)
        for x in got["models"][label]["summary"]:
            if x["task"] not in thinks.TASKS or not x["read"]:
                continue
            m = thinks.measures(reports.samples_file(reports.row_dir(mid), x["task"]), x["task"])
            assert (m["ran_out"]["stated"], m["ran_out"]["n"]) == (x["had"], x["read"]), (
                label, x["task"])
            checked += 1
    assert checked >= 4


def test_the_scorecard_and_the_model_page_read_what_is_kept(svc):
    d = reports.row_dir("served/step4000-lda-2610 · thinking")
    write(d, MMLU, [("q1", False, True, R1, 1000), ("q6", False, False, think(
        "So the answer is (B). But no.", "ANSWER: C"), 300)], ["q1", "q6"])
    from service import app
    r = reports.report("step-checkpoints", app.results_payload())
    assert "mmlupro" not in (r["thinks"].get("step4000") or {})          # not worked out yet
    report_stats.work(say=lambda s: None)
    r = reports.report("step-checkpoints", app.results_payload())
    x = r["thinks"]["step4000"]["mmlupro"]
    assert x["right"]["stated"] == 1 and x["wrong"]["lost_share"] == 1.0
    assert r["secondWords"] == ["Wait", "Alternatively", "Hmm", "Actually"]
    got = svc.get("/api/beyond", params={"model": "served/step4000-lda-2610 · thinking"}).json()
    assert got["thinks"][MMLU]["right"]["pos_median"] == pytest.approx(203 / 520)
    assert got["label"] == "step4000"


def test_the_notes_as_the_brief_words_them():
    notes = reports.NOTES
    assert ("First states the right answer means the first place the thinking says it clearly: "
            "\"the answer is (B)\", the right option's letter with its text, or the exact final "
            "answer on MATH Level 5 and OTIS — the rule the ran-out report uses. A model that "
            "only weighs an option (\"maybe (B)?\") doesn't count; one that states it and then "
            "checks it does. Not read on HLE, SimpleQA and ARC-AGI-2.") in notes
    assert ("Checking share is how much of a right answer comes after the model first stated "
            "it: re-checking, or second-guessing. Some checking is good; a lot of it is what "
            "makes answers long and sometimes run out of room (Chen et al. 2024, \"Do NOT Think "
            "That Much for 2+3=?\").") in notes
    assert ("Had it, then lost it is the share of wrong answers whose thinking had stated the "
            "right answer before giving a wrong one. A high number means the model can find the "
            "answer but doesn't trust it (Wang et al. 2025, \"Thoughts Are All Over the "
            "Place\").") in notes
    assert ("Second thoughts counts \"Wait\", \"Alternatively\", \"Hmm\" and \"Actually\" at the "
            "start of a sentence, per 1,000 thinking tokens. It's a rough sign of how often the "
            "model changes course, not a measure of quality.") in notes
