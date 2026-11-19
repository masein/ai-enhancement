"""Two things the step2388 rows showed that were never about their answers (masein, 19 Nov):

- IFEval and IFBench said "N with no answer read" on every row, N being every answer that didn't
  run out of room: their score checks each instruction on the reply and reads no answer out of
  it. Such an answer is no longer counted, nor said for a score written before.
- step2388's High and Instant rows said "35 of 40 chunks in" and "4 of 40 chunks in", every
  benchmark scored: plan U wrote both rows, and each was counted against all 40 boxes. A row now
  counts the plan's part that writes to it — its thinking's steps, of the Frontier benchmarks
  this registry records, never a box asked with another model (U40, the original Q2's).

The IF world is 25's (invented prompts); the registries are invented, shaped as plan U's."""

from __future__ import annotations

import json

import pytest

import frontier_box as fbx
from service import frontier as sf
from service import reports
from test_25_board import IFB, IFE, PLAN, row_of, svc  # noqa: F401 — the fixture, by name

GPQA, MATH, MMLU, HLE, OTIS = ("gpqa_diamond_epoch", "math_l5_epoch", "mmlupro_tiger",
                               "hle_text_cais", "otis_aime_epoch")
ARC, SQA, EASY_MMLU, GSM8K = "arc_agi2_public", "simpleqa_epoch", "mmlu_chat", "gsm8k_chat"
U = [f"U{i}" for i in range(1, 41)]


def test_instruction_following_answers_are_never_unread(svc):  # noqa: F811
    """25's builds, scored: IFEval's and IFBench's results count none unread"""
    for label, task, th in PLAN:
        d = sf.task_dir(row_of(label, th == "on")[1], task)
        res = json.loads(next(d.glob("results_*.json")).read_text())
        assert res["frontier"]["unread"] == 0, (label, task, th)
    assert not sf.reads_an_answer(IFE) and not sf.reads_an_answer(IFB)
    assert sf.reads_an_answer(GPQA) and sf.reads_an_answer(MATH) and sf.reads_an_answer(HLE)


@pytest.mark.parametrize("task,said", [(IFB, False), (IFE, False), (GPQA, True)])
def test_a_score_written_before_says_no_unread_for_them(task, said):
    """a results file written before keeps its count: the line never says it for IFEval or
    IFBench, and still does for a benchmark that reads an answer"""
    sc = {"score": 0.637, "se": 0.028, "epochs": 1, "questions": 300, "ran_out": 15,
          "unread": 285}
    line = sf.words(task, sc)
    assert "15 ran out of room" in line
    assert ("285 with no answer read" in line) is said, line


def _whole(*boxes: str) -> dict:
    return {"box": boxes[0], "boxes": list(boxes)}


def _sharded(n: int, boxes: list[str]) -> dict:
    return {"shards": n, "boxes": boxes}


# High's row: every thinking-on step of plan U's 34 boxes, merged whole or from its shards;
# and, by name, the four benchmarks U35 asks with thinking off — what made it "35 of 40"
HIGH = {"tasks": {HLE: _sharded(15, U[0:15]), MMLU: _sharded(7, U[15:22]),
                  MATH: _sharded(7, U[22:29]), GPQA: _sharded(7, U[22:29]),
                  OTIS: _sharded(3, U[29:32]), ARC: _sharded(2, ["U33", "U34"]),
                  SQA: _whole("U33"), "ifeval_google": _whole("U34"),
                  "ifbench_test": _whole("U34")}}
# Instant's row: U35's four, U36 and U37's MMLU-Pro halves, U38's easy set
INSTANT = {"tasks": {GPQA: _whole("U35"), MATH: _whole("U35"), "ifeval_google": _whole("U35"),
                     "ifbench_test": _whole("U35"), MMLU: _sharded(2, ["U36", "U37"]),
                     EASY_MMLU: _whole("U38"), GSM8K: _whole("U38")}}


def test_each_row_counts_the_part_of_its_plan_that_writes_to_it():
    assert set(fbx.PLANS["U"]) == set(U) and fbx.BOX_MODELS["U40"] != "served/step2388-k3k6-2611"
    # as before: every box of the plan, either row
    assert reports.chunks(HIGH) == (35, 40)
    assert reports.chunks(INSTANT) == (4, 40)
    # each row its own: High the 34 boxes asked with thinking on (U39's DeviceMark and
    # showcase aren't Frontier benchmarks); Instant U35 to U38 (U38's DeviceMark aside, its
    # easy set), not U40, the original Q2's
    assert reports.chunks(HIGH, "served/step2388-k3k6-2611 · thinking") == (34, 34)
    assert reports.chunks(INSTANT, "served/step2388-k3k6-2611") == (4, 4)
    # a box still to come is counted out of the row's own: MMLU-Pro's second half (U37)
    short = {"tasks": {k: v for k, v in INSTANT["tasks"].items() if k != MMLU},
             "shards": {MMLU: {"n": 2, "have": {"1": {"box": "U36"}}}}}
    assert reports.chunks(short, "served/step2388-k3k6-2611") == (3, 4)


def test_the_rows_status_says_complete_when_every_box_is_in(svc, monkeypatch, tmp_path):  # noqa: F811
    """our_rows hands the row to chunks: no "of 40 chunks" on a row whose part is all in"""
    rep = {"ours": [{"label": "step2388 · High", "row": "served/step2388-k3k6-2611 · thinking"},
                    {"label": "step2388 · Instant", "row": "served/step2388-k3k6-2611"}],
           "benchmarks": [{"key": "gpqa", "task": GPQA, "label": "GPQA Diamond"}]}
    regs = {"served/step2388-k3k6-2611 · thinking": HIGH, "served/step2388-k3k6-2611": INSTANT}
    monkeypatch.setattr(reports, "row_dir", lambda mid: tmp_path / mid.replace("/", "_"))
    monkeypatch.setattr(reports, "_registry", lambda d: next(
        v for k, v in regs.items() if d.name == k.replace("/", "_")))
    cells = {GPQA: {r: {"v": 0.8, "se": 0.02, "answers": 792} for r in regs}}
    got = {r["label"]: (r["chunks"], r["status"]) for r in reports.our_rows(rep, cells)}
    assert got == {"step2388 · High": ([34, 34], "Complete"),
                   "step2388 · Instant": ([4, 4], "Complete")}
