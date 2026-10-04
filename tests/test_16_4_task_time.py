"""16.4: the catalogue's "A run here" — how long a benchmark's run takes on
this server: each task's own lm_eval run time (one invocation a task), the
median over the models run here, a served model's left out (its time is its
server's)."""

from __future__ import annotations

import copy
import statistics

import make_fixture
import report_lm_eval as report


def runs_of(tmp_path):
    make_fixture.build(tmp_path)
    return report.load_results(tmp_path / "results" / "full")


def test_each_tasks_time_is_the_median_of_its_runs_here(tmp_path):
    runs = runs_of(tmp_path)
    payload = report.build_payload(report.merge_runs(copy.deepcopy(runs)), "t", source="")
    times = payload["taskTime"]
    assert "mmlu" in times
    # a model's last run of a task is its time (merge_runs goes in date order)
    last: dict[str, float] = {}
    for r in sorted(runs, key=lambda r: str(r.get("date") or "")):
        if "mmlu" in r["tasks"] and r.get("eval_seconds"):
            last[r["model"]] = float(r["eval_seconds"])
    assert times["mmlu"] == {"secs": round(statistics.median(last.values())), "n": len(last)}
    assert set(times) <= set(payload["accTasks"]) | set(payload["pplTasks"])


def test_a_served_models_run_is_left_out(tmp_path):
    runs = runs_of(tmp_path)
    for r in runs:
        r["served"] = {"name": "elsewhere"}
    payload = report.build_payload(report.merge_runs(runs), "t", source="")
    assert payload["taskTime"] == {}
