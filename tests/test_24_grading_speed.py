"""24: the AI models page's grading card (GET /api/frontier/grading) in
under two seconds on a large fixture. Its dry run read every graded row's
answers in full, scored each and built every grader's message on every
request (61.8 s and 120.6 s on 21 Oct); it now reads each answers file's
index (service/frontier.answer_index), worked out once by the stats worker
and kept until the answers, the questions or the grader's prompt change —
and gives the same numbers. Invented questions and answers; nothing is sent
anywhere."""

from __future__ import annotations

import json
import random
import time

import pytest

import frontier as fb
from conftest import make_service
from service import config, report_stats
from service import frontier as sf
from service import frontier_grade as fgr

N = {"math_l5_epoch": 1324, "otis_aime_epoch": 45, "hle_text_cais": 2158, "simpleqa_epoch": 1000}
ROWS = 6


def items(task: str) -> list[dict]:
    if task == "math_l5_epoch":
        return [{"id": f"algebra/{k}", "question": f"Invented problem {k}.",
                 "answer": f"\\frac{{{k}}}{{7}}", "subject": "Algebra"} for k in range(N[task])]
    if task == "otis_aime_epoch":
        return [{"id": f"o{k}", "question": f"Invented problem {k}.", "answer": str(100 + k)}
                for k in range(N[task])]
    return [{"id": f"q{k}", "question": f"Invented question {k}?", "answer": f"Answer {k}",
             "subject": "x"} for k in range(N[task])]


@pytest.fixture
def big(tmp_path, monkeypatch):
    """six rows, each with every answer of the four graded benchmarks, each
    answer 2,000 characters of thinking and its reply; a quarter graded"""
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    monkeypatch.setattr(fb, "_fetch", lambda task: items(task))
    monkeypatch.setattr(fb, "expected", lambda task: None)
    rng = random.Random(24)
    think = "<think>" + ("Let me work through it once more. " * 60) + "</think>\n\n"
    for r in range(ROWS):
        row = config.OUT_DIR / f"served__big{r}__thinking"
        row.mkdir(parents=True)
        (row / "model_meta.json").write_text(json.dumps({"model": f"served/big{r} · thinking"}))
        for task in N:
            d = sf.task_dir(row, task)
            d.mkdir(parents=True)
            graded = {}
            with open(d / sf.ANSWERS, "w") as fh:
                for it in items(task):
                    for e in range(fb.BENCH[task]["epochs"]):
                        if task == "math_l5_epoch":
                            a = think + f"So \\boxed{{\\frac{{{rng.randint(1, 9)}}}{{7}}}}"
                        elif task == "otis_aime_epoch":
                            a = think + f"ANSWER: {rng.randint(100, 150)}"
                        elif task == "hle_text_cais":
                            a = think + "Explanation: x\nExact Answer: y\nConfidence: 80%"
                        else:
                            a = think + "It is X."
                        fh.write(json.dumps({"id": it["id"], "epoch": e, "answer": a,
                                             "finish": "stop", "tokens": 900}) + "\n")
                        if rng.random() < 0.25:
                            graded[sf.gkey(it["id"], e)] = {"ok": True,
                                                            "answer_sha256": sf.answer_sha(a)}
            (d / "grades.json").write_text(json.dumps({"items": graded}))
    yield client
    client.__exit__(None, None, None)


def test_the_grading_card_answers_in_under_two_seconds(big, monkeypatch, capsys):
    t = time.time()
    old = fgr._estimate(fgr.waiting(view=True, hold={}, light=False))   # as it was worked out
    took_old = time.time() - t
    assert old["answers"] > 10_000                         # answers still to grade
    report_stats.work(say=lambda s: None)                    # the worker keeps each index
    sf._index_memo.clear()

    def never(*a, **k):
        raise AssertionError("a request worked an index out")
    monkeypatch.setattr(sf, "build_index", never)
    t = time.time()
    r = big.get("/api/frontier/grading")
    first = time.time() - t
    t = time.time()
    big.get("/api/frontier/grading")
    again = time.time() - t
    assert r.status_code == 200
    assert first < 2.0 and again < 2.0, (first, again)
    # the same numbers as before
    est = r.json()["estimate"]
    assert est["answers"] == old["answers"] and est["rows"] == old["rows"]
    assert est["graders"] == old["graders"]
    with capsys.disabled():
        print(f"\n[24] the dry run as it was: {took_old:.1f} s; the route now: {first:.2f} s, "
              f"then {again:.2f} s")


def test_an_answers_file_that_changes_is_read_again(big):
    row = config.OUT_DIR / "served__big0__thinking"
    task = "simpleqa_epoch"
    first = sf.answer_index(row, task)
    f = sf.task_dir(row, task) / sf.ANSWERS
    lines = f.read_text().splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), "answer": "A different answer."})
    f.write_text("\n".join(lines) + "\n")
    again = sf.answer_index(row, task)
    assert again["q0#0"]["sha"] == sf.answer_sha("A different answer.") != first["q0#0"]["sha"]
