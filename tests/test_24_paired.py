"""24: the two-builds report's reader — each listed question's subject and
both builds' runs, an answer with what the scorer read, the right answer
under the reader's rules (a letter alone where the question is withheld,
nothing on HLE and MATH), its thinking's tokens; a benchmark waiting for
grading said so. Invented questions and answers; nothing is run, fetched or
graded."""

from __future__ import annotations

import json

import frontier as fb
from service import config, paired, reports
from service import frontier as sf
from test_23_paired import GPQA, MMLU, Q4, S4, row_dir, svc, world  # noqa: F401

HLE, MATH = "hle_text_cais", "math_l5_epoch"


def subjects() -> None:
    """GPQA's questions with a subdomain, as a copy that has one"""
    p = fb._cache(config.BENCH_ROOT, GPQA)
    got = json.loads(p.read_text())
    for k, it in enumerate(got["items"]):
        it["subdomain"] = ("Quantum Mechanics", "Organic Chemistry")[k % 2]
    p.write_text(json.dumps(got))


def test_the_list_carries_each_questions_subject_and_both_builds_runs(svc):  # noqa: F811
    subjects()
    reports.report("paired", compute=True, a=S4, b=Q4)
    got = svc.get("/api/reports/paired", params={"a": S4, "b": Q4, "bench": "mmlupro",
                                                 "side": "a"}).json()
    assert got["runs"] == 1 and got["unit"] == "questions" and got["withheld"] == ""
    assert got["warning"].startswith("For insight only — not for training.")
    x = got["items"][0]
    assert x["subject"] == "math"
    assert x["a_runs"] == [{"epoch": 0, "ok": True, "ran_out": False}]
    assert x["b_runs"] == [{"epoch": 0, "ok": False, "ran_out": False}]
    assert all(reports.shown_half(MMLU, i["id"]) for i in got["items"])
    g = svc.get("/api/reports/paired", params={"a": S4, "b": Q4, "bench": "gpqa",
                                               "side": "a"}).json()
    assert g["runs"] == 4 and g["withheld"] and all(len(i["a_runs"]) == 4 for i in g["items"])
    assert {i["subject"] for i in g["items"]} <= {"Quantum Mechanics", "Organic Chemistry"}
    assert "invented" not in json.dumps(g).lower()                 # no question text


def test_an_answer_says_what_was_read_and_the_right_answer_as_the_reader_may(svc):  # noqa: F811
    reports.report("paired", compute=True, a=S4, b=Q4)
    m = paired.one(reports.files()["paired"], S4, MMLU, "1000", 0)
    assert m["read"] is None or isinstance(m["read"], str)
    assert m["right"] == {"letter": "A", "text": "invented option A"}
    assert m["subject"] == "math" and m["question"]["q"].startswith("Invented MMLU-Pro question")
    # thinking tokens: 100 output tokens × the thinking's share of the characters —
    # "thinking about 1000" (19) beside "ANSWER: A" (9): round(100 × 19 / 28) = 68
    assert m["think_tokens"] == 68
    q = next(i for i in range(30) if reports.shown_half(GPQA, f"q{i}"))
    g = paired.one(reports.files()["paired"], S4, GPQA, f"q{q}", 0)
    assert g["question"] is None and g["withheld"]
    assert set(g["right"]) == {"letter"}                         # the letter alone
    assert "Spin one half" not in json.dumps(g)


def test_a_benchmark_waiting_for_grading_says_so(svc):  # noqa: F811
    d = sf.task_dir(row_dir(S4), HLE)
    d.mkdir(parents=True, exist_ok=True)
    (d / sf.ANSWERS).write_text("".join(json.dumps({"id": f"h{k}", "epoch": 0, "answer": "x",
                                                    "finish": "stop"}) + "\n"
                                        for k in range(2098)))
    got = reports.report("paired", compute=True, a=S4, b=Q4)
    hle = next(r for r in got["rows"] if r["key"] == "hle")
    assert hle["missing"] == f"waiting for grading — 2,098 answers from {S4}"
    math = next(r for r in got["rows"] if r["key"] == "math")
    assert math["missing"] == f"no answers from {S4}"
