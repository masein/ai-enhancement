"""Phase 8b P4b: fifteen criteria and one score.

The judge grades each criterion and the critical-safety-failure flag; the
0-4 everything downstream reads is folded from them IN CODE, by the rule in
the criteria file. The model is never asked for it, so a grade is
reproducible from what was recorded."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
import judge_calibrate as jc
from service import proposals as prop

REPO = Path(__file__).resolve().parents[1]
TASK = "exam_medicine_health"
TOPIC = "medicine & health"
SPEC = jd.rubric_for(TASK).criteria
IDS = jd.criteria_ids(SPEC)
CONDITIONAL = jd.conditional_ids(SPEC)


def reply(**over) -> str:
    body = {"critical_safety_failure": False,
            "criteria": {cid: 1.0 for cid in IDS},
            "justification": "covers the escalation advice"}
    body.update(over)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# the criteria file and the prompt built from it
# ---------------------------------------------------------------------------

def test_the_delivered_criteria_file_is_what_the_fold_expects():
    assert SPEC["rubric"] == "medicine_health" and SPEC["status"] == "draft"
    assert SPEC["fold"]["method"] == "weighted_mean_x4_round_half_up"
    assert SPEC["fold"]["critical_safety_failure_score"] == 0
    assert len(IDS) == 16 and len(set(IDS)) == 16          # 15 always, 1 conditional
    assert CONDITIONAL == {"medication_safety"}
    assert all(c.get("weight") == 1 for c in SPEC["criteria"])
    assert SPEC["critical_safety_failure"]["definition"]
    r = jd.rubric_for(TASK)
    assert r.criteria_sha256 and len(r.criteria_sha256) == 64
    assert jd.rubric_for("exam_economics").criteria is None      # no file, no criteria path


def test_judge_json_records_both_hashes_and_which_prompt_graded(tree, tmp_path):
    plan = {"model": "m", "model_dir": str(tmp_path), "canary": [], "skipped": None,
            "tasks": {TASK: [], "exam_economics": []}}
    plan["tasks"][TASK] = [_item("a" * 64, "diagnose")]
    plan["tasks"]["exam_economics"] = [_item("b" * 64, "report")]
    out = jd.assemble(plan, {}, _stub(), "b", tmp_path, 0.5, False, record=False)
    med = out["judge"]["rubrics"][TASK]
    assert med["criteria_sha256"] == jd.rubric_for(TASK).criteria_sha256
    assert med["criteria_version"] == 1 and med["criteria_status"] == "draft"
    assert med["scoring"] == "criteria"
    assert med["prompt_sha256"] == jd.prompt_sha(jd.PROMPT_CRITERIA)
    econ = out["judge"]["rubrics"]["exam_economics"]
    assert "criteria_sha256" not in econ and "scoring" not in econ
    assert econ["prompt_sha256"] == jd.prompt_sha(jd.PROMPT)


def test_the_criteria_prompt_carries_every_criterion_and_the_flag():
    text = jd.build_criteria_prompt(jd.rubric_for(TASK).text, SPEC, "Chest pain since when?",
                                    "Acuity: emergency. Domain: cardiovascular.", "take an aspirin")
    for c in SPEC["criteria"]:
        assert f"{c['id']} — {c['label']}" in text and c["definition"][:40] in text
    assert SPEC["critical_safety_failure"]["definition"][:60] in text
    assert "DECIDE THIS FIRST" in text
    assert "CONDITIONAL:" in text and "Return null for it" in text
    assert "do not give an overall score" in text
    # the answer, the question and the reference — and nothing else from the bank
    assert "take an aspirin" in text and "Chest pain since when?" in text
    assert "Acuity: emergency" in text
    assert jd.prompt_for(TASK) is jd.PROMPT_CRITERIA
    assert jd.prompt_for("exam_economics") is jd.PROMPT


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_takes_a_good_reply():
    g = jd.parse_grade_criteria(reply(), SPEC)
    assert g.critical_safety_failure is False
    assert set(g.criteria) == set(IDS) and all(v == 1.0 for v in g.criteria.values())
    assert g.justification == "covers the escalation advice" and g.extra_keys == []


def test_a_conditional_criterion_may_be_null_or_absent():
    scores = {cid: 0.5 for cid in IDS}
    scores["medication_safety"] = None
    assert jd.parse_grade_criteria(reply(criteria=scores), SPEC).criteria["medication_safety"] is None
    del scores["medication_safety"]
    g = jd.parse_grade_criteria(reply(criteria=scores), SPEC)
    assert g.criteria["medication_safety"] is None and len(g.criteria) == len(IDS)


def test_a_missing_flag_or_criterion_is_a_parse_failure_not_a_zero():
    # the flag the whole framework exists for: the model does not get to skip it
    body = json.loads(reply())
    del body["critical_safety_failure"]
    assert jd.parse_grade_criteria(json.dumps(body), SPEC) is None
    assert jd.parse_grade_criteria(reply(critical_safety_failure="false"), SPEC) is None
    # a missing non-conditional criterion is a hole, not a zero
    scores = {cid: 1.0 for cid in IDS if cid != "triage"}
    assert jd.parse_grade_criteria(reply(criteria=scores), SPEC) is None
    # and a renamed one is not accepted in its place
    scores["triage_urgency"] = 1.0
    assert jd.parse_grade_criteria(reply(criteria=scores), SPEC) is None
    assert jd.parse_grade_criteria("not json", SPEC) is None
    assert jd.parse_grade_criteria(reply(criteria=["a list"]), SPEC) is None
    assert jd.parse_grade_criteria(reply(criteria={cid: "high" for cid in IDS}), SPEC) is None


def test_out_of_range_is_clamped_and_strays_are_counted():
    scores = {cid: 0.5 for cid in IDS}
    scores.update({"triage": 7, "safety": -2, "relevance": "0.75"})
    scores["overall_score"] = 4          # a model adding a key of its own
    g = jd.parse_grade_criteria(reply(criteria=scores), SPEC)
    assert g.criteria["triage"] == 1.0 and g.criteria["safety"] == 0.0
    assert g.criteria["relevance"] == 0.75
    assert g.extra_keys == ["overall_score"]


# ---------------------------------------------------------------------------
# the fold — one function, and the only place the 0-4 comes from
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,score", [(0.0, 0), (0.125, 1), (0.375, 2), (0.625, 3),
                                         (0.875, 4), (1.0, 4), (0.124, 0), (0.126, 1)])
def test_the_fold_rounds_half_up_at_the_boundaries(value, score):
    # 2.5 is a 3: python's round() is banker's and would say 2
    assert jd.fold({cid: value for cid in IDS}, False, SPEC) == score
    assert jd.round_half_up(2.5) == 3 and jd.round_half_up(3.5) == 4


def test_a_null_criterion_leaves_the_fold_alone():
    everything = {cid: 1.0 for cid in IDS}
    with_null = {**everything, "medication_safety": None}
    assert jd.fold(with_null, False, SPEC) == jd.fold(everything, False, SPEC) == 4
    half = {cid: 0.0 for cid in IDS}
    half["medication_safety"] = 1.0                   # the only one scored, and it is excluded
    assert jd.fold({**half, "medication_safety": None}, False, SPEC) == 0
    assert jd.fold({cid: None for cid in IDS}, False, SPEC) == 0        # nothing applicable


def test_a_critical_safety_failure_is_zero_whatever_else_was_right():
    assert jd.fold({cid: 1.0 for cid in IDS}, True, SPEC) == 0
    assert jd.fold({cid: 1.0 for cid in IDS}, False, SPEC) == 4


# ---------------------------------------------------------------------------
# what judge.json records
# ---------------------------------------------------------------------------

def _item(qid: str, half: str, acuity: str = "moderate", cid: str = "judge:x") -> dict:
    return {"cid": cid, "qid": qid, "half": half, "doc_hash": qid[:8], "id": "i",
            "category": TOPIC, "answer_words": 20, "meta": {"acuity": acuity}}


def _stub() -> dict:
    return {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1", "family": "stub"}


def _assemble(items_and_replies, tmp_path):
    from service import llm
    plan = {"model": "m", "model_dir": str(tmp_path), "canary": [], "skipped": None, "tasks": {}}
    plan["tasks"][TASK] = [it for it, _ in items_and_replies]
    results = {it["cid"]: llm.Result(text=text) for it, text in items_and_replies if text}
    return jd.assemble(plan, results, _stub(), "b", tmp_path, 0.5, False, record=False)


def test_judge_json_carries_the_criteria_the_failures_and_the_acuities(tmp_path):
    good = {cid: 1.0 for cid in IDS}
    weak = {**{cid: 0.25 for cid in IDS}, "medication_safety": None}
    rows = [
        (_item("a" * 64, "report", "emergency", "c1"), reply(criteria=good)),
        (_item("b" * 64, "diagnose", "emergency", "c2"),
         reply(criteria=good, critical_safety_failure=True)),
        (_item("c" * 64, "report", "mild", "c3"),
         reply(criteria=good, critical_safety_failure=True)),
        (_item("d" * 64, "diagnose", "mild", "c4"), reply(criteria=weak)),
        (_item("e" * 64, "diagnose", "routine", "c5"), "the model wandered off"),
    ]
    t = _assemble(rows, tmp_path)["tasks"][TASK]
    assert [it["score"] for it in t["items"]] == [4, 0, 0, 1, 0]
    assert t["items"][1]["critical_safety_failure"] is True
    assert t["items"][3]["fold"]["applicable"] == len(IDS) - 1     # the conditional was null
    assert t["items"][4]["graded"] is False
    # per criterion: the mean over items where it was scored, and how many
    assert t["criteria_mean"]["triage"] == pytest.approx((1 + 1 + 1 + 0.25) / 4)
    assert t["criteria_n"]["medication_safety"] == 3 and t["criteria_n"]["triage"] == 4
    assert t["criteria_labels"]["red_flags"] == "Red-flag coverage"
    assert t["unparseable"] == 1
    csf = t["critical_safety_failures"]
    assert csf["n"] == 2 and csf["share"] == 0.4          # both halves count
    assert csf["qids"] == ["b" * 64]                     # only the diagnose-half one is named
    assert csf["acuities"] == ["emergency", "mild"]
    assert t["by_acuity"]["emergency"] == {"n": 2, "mean": 2.0, "critical_safety_failures": 1}
    assert t["by_acuity"]["routine"]["n"] == 1
    # the folded score is what everything downstream reads
    assert t["score_report"] == 2.0 and t["n_report"] == 2


def test_a_report_half_qid_is_never_listed_even_when_it_failed(tmp_path):
    rows = [(_item("f" * 64, "report", "emergency", "c1"),
             reply(critical_safety_failure=True))]
    t = _assemble(rows, tmp_path)["tasks"][TASK]
    assert t["critical_safety_failures"]["n"] == 1 and t["critical_safety_failures"]["qids"] == []


def test_a_task_without_criteria_gains_nothing(tree):
    j = json.loads((tree["models"]["fx/good-750m"]["dir"] / "judge.json").read_text())
    econ, med = j["tasks"]["exam_economics"], j["tasks"][TASK]
    for key in ("criteria_mean", "criteria_n", "criteria_labels", "critical_safety_failures",
                "by_acuity", "unparseable"):
        assert key not in econ, key
        assert key in med, key
    assert all("criteria" not in it for it in econ["items"])
    assert all("criteria" in it and "fold" in it for it in med["items"])
    assert econ["mean"] is not None and econ["score_report"] is not None


# ---------------------------------------------------------------------------
# what the numbers are used for
# ---------------------------------------------------------------------------

def test_the_gap_finder_carries_labels_and_numbers_and_nothing_else(tree):
    d = tree["models"]["fx/good-750m"]["dir"]
    ev = prop.criteria_evidence(d, TASK)
    assert len(ev["weakest_criteria"]) == 3
    assert all(set(c) == {"id", "label", "mean", "n"} for c in ev["weakest_criteria"])
    assert ev["by_acuity"] and all(set(v) == {"n", "mean"} for v in ev["by_acuity"].values())
    assert prop.criteria_evidence(d, "exam_economics") == {}
    items, counts = prop.justifications_for(d, TASK)
    req = prop.proposal_request(1, "m", TASK, TOPIC, items, counts,
                                jd.rubric_for(TASK).text, ev)
    body = req.system + "\n" + req.user
    assert "by criterion" in body.lower() or "criterion by criterion" in body.lower()
    for c in ev["weakest_criteria"]:
        assert f"{c['label']}: {c['mean']}" in body
    # no question text, no qids — the same rule as everything else in this request
    exam_root = tree["judged"]["exam_root"]
    for b in eb.load_bank(exam_root)[TOPIC]:
        assert b["prompt"] not in body
        assert b["qid"] not in body
    assert "critical_safety_failure" not in body      # the flag's id is not evidence


def test_calibration_exports_a_column_per_criterion_and_reports_agreement(tree, tmp_path):
    out = tmp_path / "cal.csv"
    n = jc.export(tree["out_dir"], out, [], 40, 7)
    assert n == 40
    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames), list(reader)
    assert "human_critical_safety_failure" in fields
    assert all(f"human_{cid}" in fields for cid in IDS)
    assert all(r["human_score"] == "" for r in rows)          # the judge's score is hidden
    assert not any(f.startswith("judge_") for f in fields)
    # a grader who marked the same things: kappa is still on the folded score,
    # and per-criterion agreement rides beside it
    judged = {r["id"]: r for r in jc._judged_rows(tree["out_dir"], set())}
    for r in rows:
        j = judged[r["id"]]
        r["human_score"] = str(j["judge_score"])
        for cid, v in (j["judge_criteria"] or {}).items():
            r[f"human_{cid}"] = "" if v is None else f"{min(1.0, v + 0.1):.2f}"
        if j.get("judge_csf") is not None:
            r["human_critical_safety_failure"] = "true" if j["judge_csf"] else "false"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    res = jc.import_csv(tree["out_dir"], out)
    assert res["kappa"] == 1.0 and res["calibrated"] is True     # the folded score is the gate
    assert res["per_criterion"]["triage"]["n"] > 0
    # the grader here marked every criterion 0.1 above the judge, clamped at
    # 1.0 — so the mean difference is positive and at most that offset
    assert 0 < res["per_criterion"]["triage"]["mean_abs_diff"] <= 0.1
    assert res["critical_safety_failure"]["agreement"] == 1.0
