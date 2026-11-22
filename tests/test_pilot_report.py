"""A pilot's report on Reports (kind "pilot", service/pilot_report.py): benchmarks tried on
rented boxes outside the board, read from the pilot's own summary under
$BENCH_ROOT/pilot/ — internal, nothing worked out here, nothing started. The summary
here is made up (made_up()): invented models, tests and numbers; no pilot's real figures
are in the repo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import make_service
from service import config, db, pilot_report, report_stats, reported, reports

RID = "pilot-2026-11"
REP = json.loads((reported.REPORTS_DIR / f"{RID}.json").read_text())


def made_up() -> dict:
    """a pilot's summary as board_summary.py writes it: three invented tests, two of ours
    and three invented rivals, every number made up"""
    models = [("high", "High", "High", True), ("instant", "Instant", "Instant", True),
              ("ra", "Rival-A-9B", "A", False), ("rb", "Rival-B-8B", "B", False),
              ("rc", "Rival-C-2B", "C", False)]
    keys = [m[0] for m in models]

    def cells(scores, se=4.0, at_limit=None):
        return {k: {"score": v, "se": se if v else 0.0, "right": int(v), "n": 100,
                    "at_limit": at_limit[i] if at_limit else None} for i, (k, v) in enumerate(zip(keys, scores))}
    gamma = cells([30.0, 12.0, 9.0, 4.0, 3.0], at_limit=[2, 0, 40, 11, 7])
    gamma["ra"].update(score=None, se=None, words="limit-bound", extra="stopped at 14 asked, 12 at limit",
                       excluded=True)
    beta = cells([0.0, 1.0, 0.0, 0.0, 12.5])
    for k in keys:
        beta[k]["extra"] = "checks met 20.0%"
    return {
        "schema": 1, "pilot": "test", "intro": "Three invented tests, one try per item.",
        "models": [{"key": k, "label": lab, "short": sh, "ours": ours, "setup": f"{lab}'s invented setup"}
                   for k, lab, sh, ours in models],
        "rule": ["it runs cleanly;", "both between 10% and 90%;", "frontier numbers exist."],
        "rule_note": "Keep a test if all three hold.",
        "verdicts": [
            {"test": "Alpha", "row": "alpha", "clean": "yes", "high": 61.0,
             "best": {"model": "rc", "score": 66.0}, "in_band": "yes", "frontier": "yes", "verdict": "keep",
             "words": "after a fix"},
            {"test": "Beta", "row": "beta", "clean": "yes", "high": 0.0, "best": {"model": "rc", "score": 12.5},
             "in_band": "no: High at the floor", "frontier": "yes", "verdict": "drop", "words": ""},
            {"test": "Gamma", "row": "gamma", "clean": "Rival A limit-bound and excluded", "high": 30.0,
             "best": {"model": "rb", "score": 4.0}, "in_band": "no", "frontier": "yes", "verdict": "drop",
             "words": "borderline"}],
        "scores": [{"key": "alpha", "name": "Alpha", "items": "40", "cells": cells([61.0, 50.0, 55.0, 40.0, 66.0])},
                   {"key": "beta", "name": "Beta", "items": "60", "cells": beta},
                   {"key": "gamma", "name": "Gamma", "items": "100", "cells": gamma}],
        "score_notes": ["Each cell is the percentage right ± its standard error.",
                        {"text": "Gamma's share at the limit:", "items": ["High 2%", "Rival B 11%"]}],
        "per_test": [{"test": t, "clean": {k: 100.0 if t != "Gamma" else 92.5 for k in keys},
                      "clean_words": "", "errors": "none",
                      "usd_item": {k: "board" if (k, t) == ("high", "Gamma") else 0.0123 for k in keys},
                      "usd_words": ""} for t in ("Alpha", "Beta", "Gamma")],
        "per_test_notes": ["Clean means a scorable answer."],
        "full_run": {"rows": [{"test": "Alpha", "full": "400", "dropped": False,
                               "cells": {k: 4.2 for k in keys}, "later_each": 4.0, "all_nine": 37.0},
                              {"test": "Beta", "full": "600", "dropped": True,
                               "cells": {k: "done" for k in keys}, "later_each": 1.0, "all_nine": 4.0}],
                     "later": ["Later-1", "Later-2"], "kept_total": 37.0, "notes": ["Per item × the full set."]},
        "audit": {"drawn": ["10 runs, drawn with a fixed seed."],
                  "unblinded": {k: {"packets": 2, "harness_ok": 2, "consistent": 1 if k == "high" else 2,
                                    "flags": 1, "changes_score": 1 if k == "rc" else 0} for k in keys},
                  "changes": [{"title": "Rival C, Alpha 7: simulator fault", "text": "It ended early."}],
                  "early_stop": {"text": "The simulator ends 3 early.", "after": "Decide before a full run.",
                                 "lost": {"high": 1, "instant": 0, "ra": 2, "rb": 0, "rc": 0},
                                 "most": {"high": 2.5, "instant": 0.0, "ra": 5.0, "rb": 0.0, "rc": 0.0}},
                  "other": [{"text": "Alpha", "items": ["A check that never counts."]}]},
        "boxes": {"rows": [{"box": "A", "role": "High", "rate": 1.0, "hours": 2.5, "usd": 2.5,
                            "destroyed": "1 Jan 10:00"},
                           {"box": "B", "role": "the rivals", "rate": 0.8, "hours": 5.0, "usd": 4.0,
                            "destroyed": "1 Jan 12:30"}], "hours": 7.5, "usd": 6.5,
                  "teardown": "Each box copied its results home, then destroyed itself."},
        "caps": {"gpu": {"used": 6.5, "cap": 10}, "api": {"used": 1.25, "cap": 5, "before": 10.0, "after": 11.25},
                 "time": {"used": "3 h 30 min", "cap": "5 h", "from": "1 Jan 09:00", "to": "1 Jan 12:30"}},
        "api_notes": ["The key read $10.00 before and $11.25 after."],
        "credentials": "No key in any task container.",
        "disclosures": ["One run restarted."],
        "decisions": ["Alpha: fix the simulator.", "Beta stays dropped."],
    }


def put(s, rel: str = "pilot/2026-11/summary.json") -> Path:
    p = config.BENCH_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s if isinstance(s, str) else json.dumps(s), encoding="utf-8")
    return p


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield client
    client.__exit__(None, None, None)


def test_the_report_file_names_its_summary_under_bench_root_pilot():
    assert REP["kind"] == "pilot" and REP["ours"] == [] and REP["benchmarks"] == []
    assert REP["what"].startswith("Internal: ")
    assert pilot_report.path(REP) == (config.BENCH_ROOT / "pilot" / "2026-11" / "summary.json").resolve()


def test_before_the_summary_is_put_here_it_says_so(svc):
    got = svc.get(f"/api/reports/{RID}").json()
    assert got["kind"] == "pilot" and got["internal"] is True and got["summary"] is None
    assert got["sources"]["ours"]["words"] == "nothing imported yet"
    card = {c["id"]: c for c in svc.get("/api/reports").json()["reports"]}[RID]
    assert card["internal"] is True and card["line"] == "nothing imported yet" and card["changed"] == ""


def test_the_report_is_the_summary_as_kept(svc):
    s = made_up()
    p = put(s)
    os.utime(p, (1_764_000_000, 1_764_000_000))
    got = svc.get(f"/api/reports/{RID}").json()
    assert got["summary"] == s                  # as the pilot wrote it: nothing worked out here
    assert got["title"] == REP["title"] and got["what"] == REP["what"]
    assert got["sources"]["ours"]["at"] == 1_764_000_000
    assert got["sources"]["ours"]["words"] == f"the pilot's summary, put here {reports.when(1_764_000_000)}"
    card = {c["id"]: c for c in svc.get("/api/reports").json()["reports"]}[RID]
    assert card["line"] == "step2388 High and Instant against Rival-A-9B, Rival-B-8B and Rival-C-2B · " \
                           "1 of 3 tests kept"
    assert card["changed"] == reports.when(1_764_000_000) and card["internal"] is True
    # last on the list, after the other internal report
    ids = [c["id"] for c in svc.get("/api/reports").json()["reports"]]
    assert ids[-2:] == ["switching-mode", RID]


@pytest.mark.parametrize("bad", ["{not json", json.dumps([1, 2]), json.dumps({"schema": 2}),
                                 json.dumps({"verdicts": []})])
def test_a_summary_it_cannot_read_is_none(svc, bad):
    put(bad)
    assert svc.get(f"/api/reports/{RID}").json()["summary"] is None
    card = {c["id"]: c for c in svc.get("/api/reports").json()["reports"]}[RID]
    assert card["line"] == "nothing imported yet"


@pytest.mark.parametrize("data", ["", "pilot", "pilot/", "results/summary.json", "../pilot/x.json",
                                  "pilot/../results/x.json", "/etc/passwd", "pilot/../../x.json"])
def test_a_summary_outside_bench_root_pilot_is_never_read(data):
    assert pilot_report.path({**REP, "data": data}) is None


def test_a_summary_elsewhere_under_pilot_is_read(svc, monkeypatch):
    s = made_up()
    put(s, "pilot/other/run.json")
    rep = {**REP, "id": "pilot-other", "data": "pilot/other/run.json"}
    monkeypatch.setattr(reports, "files", lambda: {"pilot-other": rep})
    assert svc.get("/api/reports/pilot-other").json()["summary"] == s


def test_nothing_is_worked_out_or_started_for_a_pilot(svc, monkeypatch):
    put(made_up())

    def no(*a, **k):
        raise AssertionError("a pilot's report starts nothing")
    monkeypatch.setattr(db, "add", no)
    monkeypatch.setattr(reported, "import_epoch", no)
    assert svc.get(f"/api/reports/{RID}").status_code == 200
    assert svc.get("/api/reports").status_code == 200
    # the reports' stats worker has nothing to do for it
    assert not [j for j in report_stats.jobs() if RID in str(j)]
