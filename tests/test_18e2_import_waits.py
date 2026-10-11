"""18e point 2: an import whose answers wait for a grader (HLE, SimpleQA)
keeps the score the page showed and says it's withdrawn; the grades landing
say what replaced it (docs/prompts/phase-18e-twelfth-review-fetch-scores-
agent-page.md). Fails on 518e4a1. The box is tests/fixtures/
fake_llama_server.py; the questions are invented, no model runs."""

from __future__ import annotations

import json

import pytest

import frontier as fb
from service import config, served
from service import frontier as sf
from test_17_gguf_box import ROW as BOX_ROW, SERVED as BOX_SERVED, TASK, box  # noqa: F401
from test_17_gguf_box import bundle_of, register, run_box
from test_12q_devicemark_runs import svc  # noqa: F401 — box's fixture
from test_17b_review import answers_name, imported, rewrite
from test_18d1_scores_spending import A, SQA, grade


# ---------------------------------------------------------------------------
# 2. an import whose answers wait for a grader keeps the score it withdrew, and says it
# ---------------------------------------------------------------------------

def test_2_an_import_that_waits_for_a_grader_keeps_and_says_the_score_it_withdrew(
        box, monkeypatch):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "run"))[0] == 0
    row = config.OUT_DIR / BOX_ROW
    d = sf.task_dir(row, TASK)
    shown = json.loads(next(d.glob("results_*.json")).read_text())["results"][TASK]["acc,none"]
    assert shown == pytest.approx(0.75)
    # GPQA set to wait for a grader, as HLE's and SimpleQA's answers do
    monkeypatch.setitem(fb.BENCH, TASK, {**fb.BENCH[TASK],
                                         "grader": fb.BENCH[SQA]["grader"]})

    def other(files):
        lines = files[answers_name(BOX_ROW, TASK)].decode().splitlines()
        r = json.loads(lines[0])
        r["answer"] += "\n(asked again)"
        files[answers_name(BOX_ROW, TASK)] = ("\n".join([json.dumps(r), *lines[1:]])
                                              + "\n").encode()
    code, said = imported(rewrite(bundle_of(box, "run"), other))
    assert code == 0, said
    assert not list(d.glob("results_*.json"))                  # waiting: no score on the page
    assert f"{TASK}: its score 75.0% is withdrawn until the new answers' grades land, which " \
        "say what replaced it" in said, said                   # 518e4a1: unsaid
    assert json.loads((d / sf.REPLACED).read_text())["score"] == pytest.approx(0.75)
    # the grades land, every answer right: the line says what it replaced, the cell its "was"
    got = sf.read_answers(d / sf.ANSWERS)
    grade(row, TASK, A, {q: True for q, _ in got})
    lines: list[str] = []
    page = sf.score_task(row, TASK, served.get(BOX_SERVED), log=lines.append)
    assert page["score"] == pytest.approx(1.0)
    assert lines == [f"[frontier] {BOX_ROW} · GPQA Diamond: score 75.0% → 100.0% "
                     "(scored again)"], lines                  # 518e4a1: nothing said
    res = json.loads(next(d.glob("results_*.json")).read_text())
    assert res["frontier"]["before"]["score"] == pytest.approx(0.75)
    assert not (d / sf.REPLACED).exists()
