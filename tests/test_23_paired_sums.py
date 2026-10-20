"""23: the sums of two builds compared question by question, against
hand-worked fixtures (scripts/paired_stats.py)."""

from __future__ import annotations

from math import sqrt

import pytest

import frontier as fb
import paired_stats as ps


def test_one_run_and_the_exact_mcnemar_test():
    got = ps.one_run(7, 1, 50)
    assert got["d"] == pytest.approx(0.12)                      # +12.0 points
    assert got["p"] == pytest.approx(18 / 256) and round(got["p"], 4) == 0.0703
    assert got["verdict"] == "can't tell"
    # SE: √((8 − 50·0.0144)/49)/√50
    assert got["se"] == pytest.approx(sqrt((8 - 50 * 0.12 ** 2) / 49) / sqrt(50))
    got = ps.one_run(9, 1, 50)
    assert round(got["p"], 4) == 0.0215 and got["verdict"] == "A higher"
    assert ps.one_run(1, 9, 50)["verdict"] == "B higher"


def test_no_question_differs_is_cant_tell_and_never_divides_by_zero():
    got = ps.one_run(0, 0, 40)
    assert got["d"] == 0 and got["se"] == 0 and got["p"] == 1.0
    assert got["verdict"] == "can't tell"
    assert ps.one_run(0, 0, 0)["verdict"] == "can't tell"
    assert ps.several_runs([], [])["verdict"] == "can't tell"


def test_several_runs_paired_se_by_hand():
    # 4 questions, 4 runs each: A right 4, 3, 2, 0; B right 2, 3, 1, 0
    sa, sb = [1.0, 0.75, 0.5, 0.0], [0.5, 0.75, 0.25, 0.0]
    got = ps.several_runs(sa, sb)
    # differences 0.5, 0, 0.25, 0: mean 0.1875; deviations² sum 0.171875; var /3
    d = [0.5, 0.0, 0.25, 0.0]
    mu = sum(d) / 4
    se = sqrt(sum((x - mu) ** 2 for x in d) / 3 / 4)
    assert got["d"] == pytest.approx(0.1875) and got["se"] == pytest.approx(se)
    assert got["lo"] == pytest.approx(0.1875 - 1.96 * se)
    assert got["verdict"] == ("A higher" if 0.1875 - 1.96 * se > 0 else "can't tell")


def test_arc_by_task_as_summary_scores_it():
    grids = {"t1#0": [True, False], "t1#1": [False, False], "t2#0": [False, True]}
    group = {"t1#0": "t1", "t1#1": "t1", "t2#0": "t2"}
    tasks = ps.arc_tasks(grids, group)
    assert tasks == {"t1": 0.5, "t2": 1.0}
    board = fb.summary(grids, "arc_agi2_public", group)
    assert board["score"] == pytest.approx(sum(tasks.values()) / len(tasks))


def test_how_often_a_build_disagrees_with_itself():
    assert ps.self_disagree(4, 2) == pytest.approx(2 * 2 * 2 / (4 * 3))
    assert round(ps.self_disagree(4, 2), 3) == 0.667
    assert ps.self_disagree(4, 4) == 0 and ps.self_disagree(4, 0) == 0
    assert ps.self_disagree(1, 1) is None
    assert ps.cross_disagree(0.5, 0.5) == 0.5 and ps.cross_disagree(1.0, 0.0) == 1.0
