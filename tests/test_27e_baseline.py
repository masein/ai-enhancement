"""27e: Q2 as our same-size baseline
(docs/prompts/phase-27e-top-bar-q2-baseline-and-small-fixes.md, Part 2).

On-device: Q2's row with its line, what it keeps of our BF16, and on IFEval
and IFBench — which our BF16 hasn't run — our Q4 standing in for BF16, said
on every row it serves, until BF16's own come in. Step checkpoints: the
baseline the viewer compares against, Q2 by default or Q4, and the lines
that name it; every number the same either way. Invented scores, chosen so
the shares come out as the brief's; nothing runs, nothing is fetched."""

from __future__ import annotations

import json

import pytest

from conftest import make_service
from service import config, ondevice, paired, reported, reports

OD = json.loads((reported.REPORTS_DIR / "on-device.json").read_text())
ROW = {o["label"]: o["row"] for o in OD["ours"]}
TASK = {b["key"]: b["task"] for b in OD["benchmarks"]}
Q2, Q4, BF16 = "Q2 (UD-Q2_K_XL)", "Q4 (UD-Q4_K_XL)", "BF16"
# invented: {label: {benchmark: score}}
SCORES = {BF16: {"gpqa": 0.848, "mmlupro": 0.850},
          Q4: {"gpqa": 0.848, "mmlupro": 0.8525, "ifeval": 0.900, "ifbench": 0.600},
          Q2: {"gpqa": 0.818, "mmlupro": 0.838},
          "step8804": {"gpqa": 0.80, "mmlupro": 0.82, "ifeval": 0.883, "ifbench": 0.568},
          "step4000": {"gpqa": 0.79, "mmlupro": 0.81, "ifeval": 0.8577, "ifbench": 0.520}}
LINE = "of our Q4, standing in for BF16 (Q4 keeps 100.0–100.3% of BF16 elsewhere)"


def cells(scores: dict) -> dict:
    out: dict = {}
    for label, got in scores.items():
        for k, v in got.items():
            out.setdefault(TASK[k], {})[ROW[label]] = {"v": v, "se": 0.01, "answers": 100}
    return out


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, _, _ = make_service(tmp_path, monkeypatch, tree=False)
    yield client
    client.__exit__(None, None, None)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def test_q2_is_a_row_of_ours_with_its_line_and_what_it_keeps_of_our_bf16(svc):
    d = ondevice.report(OD, cells(SCORES))
    rows = {r["label"]: r for r in d["rows"]}
    q2 = rows[Q2]
    assert q2["ours"] and q2["row"] == "served/Qwen3.6-35B-A3B-Q2-lda-2610 · thinking"
    assert q2["what"] == "the original, squeezed to 2 bits — our same-size baseline"
    assert q2["colour"] == 3 and q2["source"] == "measured here"
    # of our BF16, like our other builds
    assert (pct(q2["retention"]["gpqa"]), pct(q2["retention"]["mmlupro"])) == ("96.5", "98.6")
    assert q2["keepOf"] == {}
    # no IFEval or IFBench yet: a "–", not run yet
    assert q2["notRun"] == ["ifeval", "ifbench"]
    assert "ifeval" not in q2["retention"] and "ifbench" not in q2["retention"]
    # the others' line is their kind and source, as before
    assert "what" not in rows["step8804"]


def test_ifeval_and_ifbench_against_our_q4_standing_in_for_bf16(svc):
    d = ondevice.report(OD, cells(SCORES))
    rows = {r["label"]: r for r in d["rows"]}
    si = d["standIn"]
    assert (si["label"], si["for"], si["benchmarks"]) == (Q4, BF16, ["ifeval", "ifbench"])
    assert si["line"] == LINE
    assert si["elsewhere"] == "Q4 keeps 100.0–100.3% of BF16 elsewhere"
    assert [pct(x) for x in si["keeps"]] == ["100.0", "100.3"]
    for label, ife, ifb in (("step8804", "98.1", "94.7"), ("step4000", "95.3", "86.7")):
        r = rows[label]
        assert (pct(r["retention"]["ifeval"]), pct(r["retention"]["ifbench"])) == (ife, ifb), label
        assert r["keepOf"] == {"ifeval": "stand_in", "ifbench": "stand_in"}, label
        # GPQA and MMLU-Pro still of our BF16
        assert r["retention"]["gpqa"] == pytest.approx(SCORES[label]["gpqa"] / 0.848)
    # the stand-in isn't measured against itself
    q4 = rows[Q4]
    assert "ifeval" not in q4["retention"] and q4["keepOf"] == {} and q4["notRun"] == []
    assert pct(q4["retention"]["mmlupro"]) == "100.3"
    # BF16 is still the base and never a row
    assert BF16 not in rows


def test_bf16s_own_scores_replace_the_stand_in_by_themselves(svc):
    got = {**SCORES, BF16: {**SCORES[BF16], "ifeval": 0.90, "ifbench": 0.64}}
    d = ondevice.report(OD, cells(got))
    rows = {r["label"]: r for r in d["rows"]}
    assert d["standIn"] is None
    s88 = rows["step8804"]
    assert s88["keepOf"] == {}
    assert s88["retention"]["ifbench"] == pytest.approx(0.568 / 0.64)
    # and Q4 is measured against BF16 there too
    assert rows[Q4]["retention"]["ifeval"] == pytest.approx(1.0)
    # one of the two in: the stand-in serves the other alone
    half = {**SCORES, BF16: {**SCORES[BF16], "ifeval": 0.90}}
    d = ondevice.report(OD, cells(half))
    assert d["standIn"]["benchmarks"] == ["ifbench"]
    assert {r["label"]: r for r in d["rows"]}["step8804"]["keepOf"] == {"ifbench": "stand_in"}


# ---- Step checkpoints: the baseline switch, on 25's instruction-following world ----

@pytest.fixture
def ifsvc(tmp_path, monkeypatch):
    import test_25_board as t25
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired._tables.clear()
    t25.world()
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield client
    client.__exit__(None, None, None)


def test_the_step_checkpoints_baseline_is_q2_by_default_or_q4(ifsvc):
    d = reports.report("step-checkpoints", compute=True)
    B = d["baseline"]
    assert B["options"] == [Q2, Q4] and B["default"] == Q2
    q2, q4 = B["by"][Q2], B["by"][Q4]
    assert (q2["baselineSel"], q4["baselineSel"]) == (Q2, Q4)
    # the report's own lines are the default's
    assert d["training"] == q2["training"] and d["follow"] == q2["follow"]
    assert d["tokenLine"] == q2["tokenLine"]
    # Across training: the baseline the dashed line, BF16 beside it, as before
    assert q2["training"]["references"] == [Q2, BF16]
    assert q4["training"]["references"] == [Q4, BF16]
    # every number the same either way: the segments, the scores, the sentence
    assert q2["training"]["segments"] == q4["training"]["segments"]
    assert q2["training"]["sentence"] == q4["training"]["sentence"]
    assert q2["follow"]["rows"] == q4["follow"]["rows"]
    # the paired verdicts follow it: Q4's as 25's, Q2's not measured (no answers here)
    assert q4["follow"]["sentence"].startswith(
        "step8804 against Q4 (UD-Q4_K_XL) on IFEval: lower, by 50.0 points")
    assert q2["follow"]["sentence"].startswith(
        "step8804 against Q2 (UD-Q2_K_XL) on IFEval: not measured yet")
    # near the ceiling stays by the report's reference, Q4
    assert d["reference"] == Q4


def test_a_report_without_two_baselines_has_no_switch(ifsvc):
    rep = json.loads((reported.REPORTS_DIR / "step-checkpoints.json").read_text())
    rows = reports.our_rows(rep, {})
    assert reports.baselines({**rep, "baselines": [Q2]}, rows, {}, {}) is None
    assert reports.baselines({**rep, "baselines": [Q2, "no such build"]}, rows, {}, {}) is None
