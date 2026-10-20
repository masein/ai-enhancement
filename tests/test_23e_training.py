"""23e: our checkpoints across training (service/training.py) — each
segment's verdict read from the two-builds report's kept pairs, the sentence
worked by hand, a checkpoint unscored on a benchmark leaving its line
shorter; the answers behind "had it, then lost it" and "ran out with it
already stated", the shown half listed and opened with its marks, the hidden
half refused; near the ceiling at 90.0% and not at 89.9%; the notes.
Invented answers; nothing is run, fetched or graded."""

from __future__ import annotations

import json

import pytest

import frontier as fb
from conftest import make_service
from service import config, paired, ran_out, report_stats, reports, thinks, training
from service import frontier as sf

MMLU, MATH, GPQA = "mmlupro_tiger", "math_l5_epoch", "gpqa_diamond_epoch"
ROWS = {"step1000": "served/step1000-lda-2610 · thinking",
        "step4000": "served/step4000-lda-2610 · thinking",
        "step8804": "served/step8804-lda-2610 · thinking",
        "Q4 (UD-Q4_K_XL)": "served/Qwen3.6-35B-A3B-Q4-lda-2610 · thinking",
        "BF16": "served/Qwen3.6-35B-A3B-BF16-lda-2610 · thinking"}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired._tables.clear()
    yield client
    client.__exit__(None, None, None)


def right_on(label: str, task: str, k: int, n: int = 100) -> None:
    """`label` right on the first k of n invented questions, one run each"""
    d = sf.task_dir(reports.row_dir(ROWS[label]), task)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{task}_2026-10-21T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": q, "doc": {"id": f"q{q}"}, "resps": [["ANSWER: A"]],
                    "filtered_resps": ["A"],
                    "frontier": [{"epoch": 0, "ok": q < k, "ran_out": False}]}) + "\n"
        for q in range(n)))


def rows(**scores) -> list[dict]:
    """the scorecard's rows as reports.our_rows gives them: {label: {key: v}}"""
    return [{"label": lb, "cells": {k: {"v": v, "se": 0.02} for k, v in sc.items()}}
            for lb, sc in scores.items()]


def world() -> list[dict]:
    """MMLU-Pro: step1000 40, step4000 70, step8804 72, Q4 90 of 100; MATH
    Level 5: step4000 80, step8804 60, Q4 90 — step1000 not on it"""
    for lb, k in (("step1000", 40), ("step4000", 70), ("step8804", 72), ("Q4 (UD-Q4_K_XL)", 90)):
        right_on(lb, MMLU, k)
    for lb, k in (("step4000", 80), ("step8804", 60), ("Q4 (UD-Q4_K_XL)", 90)):
        right_on(lb, MATH, k)
    return rows(step1000={"mmlupro": .40}, step4000={"mmlupro": .70, "math": .80},
                step8804={"mmlupro": .72, "math": .60},
                # Q4's board scores: MATH Level 5 at 90% (near the ceiling), MMLU-Pro at 89.9%
                **{"Q4 (UD-Q4_K_XL)": {"mmlupro": .899, "math": .90}, "BF16": {"mmlupro": .88}})


def test_the_checkpoints_and_their_segments(svc):
    rep = reports.files()["step-checkpoints"]
    assert [(o["label"], o.get("step")) for o in rep["ours"][:3]] == [
        ("step8804", 8804), ("step4000", 4000), ("step1000", 1000)]
    assert rep["training"] == {"label": "Training steps", "references": ["Q4 (UD-Q4_K_XL)", "BF16"]}
    got = training.of_report(rep, world(), compute=True)
    assert [c["label"] for c in got["checkpoints"]] == ["step1000", "step4000", "step8804"]
    assert got["references"] == ["Q4 (UD-Q4_K_XL)", "BF16"] and got["paired"] == "paired"
    # MMLU-Pro: 1000 → 4000, 30 questions up and none down (McNemar p < .001); 4000 → 8804,
    # two up (p = 0.5)
    mm = got["segments"]["mmlupro"]
    assert [(s["a"], s["b"], s["verdict"]) for s in mm] == [
        ("step1000", "step4000", "higher"), ("step4000", "step8804", "can't tell")]
    assert mm[0]["d"] == pytest.approx(30.0) and mm[1]["d"] == pytest.approx(2.0)
    # MATH Level 5: step1000 not scored — one segment, 20 down
    assert [(s["a"], s["b"], s["verdict"]) for s in got["segments"]["math"]] == [
        ("step4000", "step8804", "lower")]
    assert got["segments"]["gpqa"] == []


def test_the_sentence_as_worked_by_hand(svc):
    rep = reports.files()["step-checkpoints"]
    got = training.of_report(rep, world(), compute=True)
    # Q4 scores 90% on MATH Level 5: near the ceiling, said after its "lower"
    assert got["sentence"] == ("From step4000 to step8804: can't tell on MMLU-Pro; lower on "
                               "MATH Level 5 (near the ceiling).")
    # against Q4: 18 and 30 questions the other way on both — lower on 2 of 2
    assert got["against"] == ("Against Q4, step8804 is still lower on 2 of 2 (MATH Level 5 near "
                              "the ceiling).")
    # one checkpoint only: no section
    one = {**rep, "ours": [{k: v for k, v in o.items() if k != "step" or o["label"] == "step8804"}
                           for o in rep["ours"]]}
    assert training.of_report(one, world()) is None


def test_a_pair_the_worker_hasnt_kept_has_no_label(svc):
    rep = reports.files()["step-checkpoints"]
    w = world()
    got = training.of_report(rep, w)                          # nothing kept, nothing computed
    assert got["segments"]["mmlupro"] == [{"a": "step1000", "b": "step4000"},
                                          {"a": "step4000", "b": "step8804"}]
    assert got["sentence"] == "" and got["against"] == ""
    report_stats.work(say=lambda s: None)                     # the worker keeps the pairs
    assert training.of_report(rep, w)["segments"]["mmlupro"][0]["verdict"] == "higher"


def test_near_the_ceiling_at_90_and_not_at_89_9(svc):
    rep = reports.files()["step-checkpoints"]
    got = training.near_ceiling(rep, rows(**{"Q4 (UD-Q4_K_XL)": {"math": 0.900, "gpqa": 0.899}}))
    assert got["math"] is True and got["gpqa"] is False and got["hle"] is False
    assert training.CEILING == 0.90
    # the two-builds report's rows and sentence say it too
    w = world()
    report_stats.work(say=lambda s: None)
    from service import app
    p = reports.report("paired", app.results_payload(), a="step8804", b="step4000")
    assert "ceiling" in p["rows"][0] and p["ceilingWords"] == training.CEILING_WORDS
    # MATH Level 5 marked as Q4's 90% would mark it; the pairs as the worker kept them
    rws = [{**r, "ceiling": r["key"] == "math"} for r in p["rows"]]
    assert paired.sentence("step8804", "step4000", rws) == (
        "step8804 is higher than step4000 on 0 benchmarks, lower on 1 (MATH Level 5 near the "
        "ceiling), and the two can't be told apart on 1.")
    del w


# ---------------------------------------------------------------------------
# the examples behind two of 23d's numbers
# ---------------------------------------------------------------------------

def gpqa_world() -> tuple[list[str], list[str]]:
    """30 invented GPQA questions: step4000 right on q0–q9, wrong on the rest,
    q10–q24 having stated the right answer first; q25–q29 ran out, q25–q27
    with it stated. Returns the lost and ran-out ids"""
    items = [{"id": f"g{k}", "question": f"Invented question {k}?", "right": f"right {k}",
              "wrong": [f"wrong {k}{c}" for c in "abc"]} for k in range(30)]
    p = fb._cache(config.BENCH_ROOT, GPQA)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"items": items, "extra": {}}))
    lines = []
    for k, it in enumerate(items):
        letter = ran_out.right_of(GPQA, it)["letter"]
        other = next(c for c in "ABCD" if c != letter)
        say = f"So the answer is ({letter})."
        if k < 10:
            text, ok, out, read = f"<think>Some steps. {say} Checked.</think>\n\nANSWER: {letter}", \
                True, False, letter
        elif k < 25:
            text = (f"<think>{'x' * 100}{say} But wait, maybe not.</think>\n\nThe options "
                    f"weighed.\nANSWER: {other}")
            ok, out, read = False, False, other
        else:
            text, ok, out, read = "<think>" + (say if k < 28 else "") + " again." * 200, False, \
                True, None
        lines.append(json.dumps({"doc_id": k, "doc": {"id": it["id"]}, "resps": [[text]],
                                 "filtered_resps": [read or ""],
                                 "frontier": [{"epoch": 0, "ok": ok, "ran_out": out,
                                               "read": read, "tokens": 1000}]}) + "\n")
    d = sf.task_dir(reports.row_dir(ROWS["step4000"]), GPQA)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{GPQA}_2026-10-21T08-00-00.000000.jsonl").write_text("".join(lines))
    return [f"g{k}" for k in range(10, 25)], [f"g{k}" for k in range(25, 28)]


def test_the_list_holds_the_shown_half_and_counts_the_rest(svc):
    lost, out = gpqa_world()
    report_stats.work(say=lambda s: None)
    got = svc.get("/api/reports/step-checkpoints/thinks",
                  params={"model": "step4000", "bench": "gpqa", "which": "lost"}).json()
    shown = [q for q in lost if reports.shown_half(GPQA, q)]
    assert 0 < len(shown) < len(lost)                          # both halves in the fixture
    assert [x["id"] for x in got["items"]] == shown
    assert (got["n"], got["listed"], got["hidden"]) == (15, len(shown), 15 - len(shown))
    assert got["warning"] == ran_out.WARNING
    x = got["items"][0]
    assert set(x) == {"id", "epoch", "ran_out", "pos", "read"}  # no question text
    assert x["pos"] == pytest.approx(103 / len("x" * 100 + "So the answer is (A). But wait, "
                                                "maybe not."))
    assert x["ran_out"] is False and x["read"] in "ABCD"
    ro = svc.get("/api/reports/step-checkpoints/thinks",
                 params={"model": "step4000", "bench": "gpqa", "which": "ran_out"}).json()
    assert [y["id"] for y in ro["items"]] == [q for q in out if reports.shown_half(GPQA, q)]
    assert all(y["ran_out"] and y["read"] is None for y in ro["items"])
    # the scorecard has the sizes, not the lists
    from service import app
    r = reports.report("step-checkpoints", app.results_payload())
    t = r["thinks"]["step4000"]["gpqa"]
    assert "lists" not in t and t["listed"]["lost"] == {"listed": len(shown),
                                                       "hidden": 15 - len(shown)}
    assert svc.get("/api/reports/step-checkpoints/thinks", params={
        "model": "step4000", "bench": "hle", "which": "lost"}).status_code == 404


def test_an_answer_opens_with_its_marks_and_a_hidden_one_is_refused(svc):
    lost, _ = gpqa_world()
    report_stats.work(say=lambda s: None)
    shown = next(q for q in lost if reports.shown_half(GPQA, q))
    hidden = next(q for q in lost if not reports.shown_half(GPQA, q))
    a = svc.get("/api/reports/paired/answer", params={"model": "step4000", "task": GPQA,
                                                      "id": shown, "epoch": 0}).json()
    s, e = a["marks"]["first"]
    assert a["thinking"][s:e].startswith("the answer is (") and a["thinking"][e - 1] == "."
    fs, fe = a["marks"]["final"]
    assert a["reply"][fs:fe].startswith("ANSWER: ") and a["reply"][fe - 1] in "ABCD"
    # GPQA: no question text, the right option's letter alone
    assert a["question"] is None and a["withheld"] and a["right_letter"] in "ABCD"
    assert "right " + shown[1:] not in json.dumps(a)            # the right option's text
    r = svc.get("/api/reports/paired/answer", params={"model": "step4000", "task": GPQA,
                                                      "id": hidden, "epoch": 0})
    assert r.status_code == 403 and "hidden half" in r.json()["detail"]


def test_the_notes_as_the_brief_words_them():
    assert ("Across training draws our checkpoints of the same training run in order of their "
            "training step. The label on each segment is the question-by-question verdict "
            "between the two checkpoints, from the two-builds report: it can say \"higher\" or "
            "\"lower\" where the two scores' ± overlap, because it compares the same "
            "questions.") in reports.NOTES
    assert ("Near the ceiling marks a benchmark where our reference build scores 90% or more. "
            "There, a few questions decide the difference.") in reports.NOTES
    assert f"{training.CEILING:.0%}" == "90%"
    assert thinks.WAY == "thinks-2"
