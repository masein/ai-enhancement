"""Phases 8b and 8d: the criteria file is the author's, and the judge reads it.

The judge scores each criterion 0–1 and answers true or false for each flag;
the 0–4 everything downstream reads is folded from them IN CODE — the mean
of the criteria, then each true flag's effect in the file's order. The model
is never asked for the 0–4, so a grade is reproducible from what was
recorded. Nothing about medicine or law is special-cased: both files are
read by one loader, and a topic's tables come from the metadata its own
items carry."""

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
LAW_TASK = "exam_law"
SPEC = jd.rubric_for(TASK).criteria
LAW = jd.rubric_for(LAW_TASK).criteria
IDS = jd.criteria_ids(SPEC)
FLAGS = jd.flag_ids(SPEC)


def reply(spec=SPEC, **over) -> str:
    body = {"criteria": {cid: 1.0 for cid in jd.criteria_ids(spec)},
            "flags": {fid: False for fid in jd.flag_ids(spec)},
            "justification": "covers the escalation advice"}
    body.update(over)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# the author's schema, loaded as delivered
# ---------------------------------------------------------------------------

def test_both_delivered_files_load_and_say_what_they_grade():
    assert jd.validate_criteria(SPEC) == [] and jd.validate_criteria(LAW) == []
    assert len(IDS) == 15 and len(set(IDS)) == 15
    assert len(jd.criteria_ids(LAW)) == 23
    assert FLAGS == ["critical_safety_failure"]
    assert jd.flag_ids(LAW) == ["critical_legal_error", "fabricated_authority"]
    # `topic` is informational — the FILE NAME decides which task it grades
    assert isinstance(SPEC["topic"], str) and isinstance(LAW["topic"], str)
    assert jd.rubric_name(TASK) == "medicine_health" and jd.rubric_name(LAW_TASK) == "law"
    assert jd.rubric_for("exam_history").criteria is None      # no file, no criteria path
    for r in (jd.rubric_for(TASK), jd.rubric_for(LAW_TASK)):
        assert len(r.criteria_sha256) == 64
        assert r.status == "draft"          # the prose anchors are not signed off yet
    # a label is optional in the schema; the page still has words for every id
    assert jd.criteria_labels(SPEC)["red_flag_coverage"] == "Red flag coverage"
    assert jd.criteria_labels(LAW)["jurisdiction_awareness"] == "Jurisdiction awareness"


def test_equal_weights_with_the_key_and_without_it():
    assert SPEC["weights"] == "equal" and LAW["weights"] == "equal"
    assert all(jd.weight_of(SPEC, cid) == 1.0 for cid in IDS)
    bare = {"criteria": [{"id": "a", "definition": "d"}, {"id": "b", "definition": "d"}]}
    assert jd.validate_criteria(bare) == []                    # no weights key at all
    assert jd.weight_of(bare, "a") == 1.0
    heavier = {"criteria": [{"id": "a", "definition": "d", "weight": 3},
                            {"id": "b", "definition": "d"}]}
    assert jd.weight_of(heavier, "a") == 3.0 and jd.weight_of(heavier, "b") == 1.0
    # a file that asks for a weighting scheme this loader does not implement
    assert any("weights" in p for p in jd.validate_criteria({**bare, "weights": "softmax"}))


def test_an_effect_this_judge_cannot_apply_refuses_to_load(tmp_path, monkeypatch):
    spec = json.loads(json.dumps(LAW))
    spec["flags"][1]["effect"] = "cap_at_9_of_4"
    assert any("cap_at_9_of_4" in p and "fabricated_authority" in p
               for p in jd.validate_criteria(spec))
    spec["flags"][1]["effect"] = "melt_the_score"
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "rubrics" / "law.md").write_text(jd.rubric_for(LAW_TASK).text, encoding="utf-8")
    (tmp_path / "rubrics" / "law.criteria.json").write_text(json.dumps(spec), encoding="utf-8")
    from service import config
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    with pytest.raises(jd.CriteriaError) as e:
        jd.rubric_for(LAW_TASK)
    # the file and the string it could not apply, both named
    assert "melt_the_score" in str(e.value) and "law.criteria.json" in str(e.value)
    assert "fabricated_authority" in str(e.value)
    # every effect the two delivered files ask for is one it can apply
    assert {jd.effect_of(f) for f in LAW["flags"]} == {"zero_score", "cap_at_1_of_4"}
    assert jd.effect_words("zero_score") == "sets the whole score to 0"
    assert jd.effect_words("cap_at_1_of_4") == "caps the whole score at 1 of 4"


def test_a_criterion_and_a_flag_may_share_an_id_and_stay_apart():
    """Law scores `fabricated_authority` 0–1 AND flags it: the criterion is
    about invention at all, the flag about invention that carries the
    answer."""
    assert "fabricated_authority" in jd.criteria_ids(LAW)
    assert "fabricated_authority" in jd.flag_ids(LAW)
    text = jd.build_criteria_prompt(jd.rubric_for(LAW_TASK).text, LAW, "Can he evict me?",
                                    "Acuity: urgent. Difficulty: 4.", "no idea")
    crit_block = text.split("CRITERIA (0.0–1.0 each)\n", 1)[1].split("\n\nFLAGS", 1)[0]
    flag_block = text.split("FLAGS (true/false each)\n", 1)[1].split("\n\nQUESTION", 1)[0]
    assert "fabricated_authority — Whether the response avoids invented" in crit_block
    assert "fabricated_authority — fabricated legal authority materially affects" in flag_block
    assert "caps the whole score at 1 of 4" in flag_block
    # and they are answered in two objects, so a parse cannot conflate them
    g = jd.parse_grade_criteria(reply(LAW, flags={"critical_legal_error": False,
                                                  "fabricated_authority": True}), LAW)
    assert g.criteria["fabricated_authority"] == 1.0
    assert g.flags["fabricated_authority"] is True


def test_the_criteria_prompt_carries_every_criterion_and_every_flag():
    text = jd.build_criteria_prompt(jd.rubric_for(TASK).text, SPEC, "Chest pain since when?",
                                    "Acuity: emergency. Domain: cardiovascular.", "take an aspirin")
    for c in SPEC["criteria"]:
        assert f"{c['id']} — {c['definition'][:40]}" in text
    for f in SPEC["flags"]:
        assert f"{f['id']} — {f['condition']}" in text
    assert "CRITERIA (0.0–1.0 each)" in text and "FLAGS (true/false each)" in text
    assert "do not give an overall score" in text
    # the answer, the question and the reference — and nothing else from the bank
    assert "take an aspirin" in text and "Chest pain since when?" in text
    assert "Acuity: emergency" in text
    assert jd.prompt_for(TASK) is jd.PROMPT_CRITERIA
    assert jd.prompt_for("exam_history") is jd.PROMPT      # no criteria file of its own


def test_a_conditional_criterion_is_told_when_to_return_null():
    """Neither delivered file has one today; the path is kept, and the prompt
    says what to do even when the author wrote no applies_when."""
    spec = {"criteria": [{"id": "a", "definition": "always"},
                         {"id": "meds", "definition": "medication safety",
                          "conditional": True},
                         {"id": "b", "definition": "when asked", "conditional": True,
                          "applies_when": "the question is about a child"}]}
    block = jd.criteria_block(spec)
    assert "meds — medication safety CONDITIONAL. Return null for it" in block
    assert "b — when asked CONDITIONAL: applies when the question is about a child." in block
    assert jd.conditional_ids(spec) == {"meds", "b"}
    body = json.dumps({"criteria": {"a": 1.0, "meds": None}, "flags": {}, "justification": ""})
    g = jd.parse_grade_criteria(body, spec)
    assert g.criteria["meds"] is None and g.criteria["b"] is None    # null or absent
    assert jd.fold({"a": 1.0, "meds": None, "b": None}, {}, spec) == 4


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_takes_a_good_reply():
    g = jd.parse_grade_criteria(reply(), SPEC)
    assert g.flags == {"critical_safety_failure": False}
    assert set(g.criteria) == set(IDS) and all(v == 1.0 for v in g.criteria.values())
    assert g.justification == "covers the escalation advice" and g.extra_keys == []


def test_a_missing_flag_or_criterion_is_a_parse_failure_not_a_zero():
    # the fields the framework exists for: the model does not get to skip one
    body = json.loads(reply(LAW))
    del body["flags"]["fabricated_authority"]
    assert jd.parse_grade_criteria(json.dumps(body), LAW) is None
    body = json.loads(reply(LAW))
    del body["flags"]
    assert jd.parse_grade_criteria(json.dumps(body), LAW) is None
    assert jd.parse_grade_criteria(reply(flags={"critical_safety_failure": "false"}),
                                   SPEC) is None
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
    g = jd.parse_grade_criteria(reply(criteria=scores,
                                      flags={"critical_safety_failure": False,
                                             "vibes": True}), SPEC)
    assert g.criteria["triage"] == 1.0 and g.criteria["safety"] == 0.0
    assert g.criteria["relevance"] == 0.75
    assert g.extra_keys == ["overall_score", "flags.vibes"]
    assert "vibes" not in g.flags


# ---------------------------------------------------------------------------
# the fold — one function, and the only place the 0-4 comes from
# ---------------------------------------------------------------------------

NO_FLAGS = {"critical_safety_failure": False}


@pytest.mark.parametrize("value,score", [(0.0, 0), (0.125, 1), (0.375, 2), (0.625, 3),
                                         (0.875, 4), (1.0, 4), (0.124, 0), (0.126, 1)])
def test_the_fold_rounds_half_up_at_the_boundaries(value, score):
    # 2.5 is a 3: python's round() is banker's and would say 2
    assert jd.fold({cid: value for cid in IDS}, NO_FLAGS, SPEC) == score
    assert jd.round_half_up(2.5) == 3 and jd.round_half_up(3.5) == 4


def test_a_null_criterion_leaves_the_fold_alone():
    spec = {"criteria": [{"id": "a", "definition": "d"}, {"id": "b", "definition": "d"},
                         {"id": "c", "definition": "d", "conditional": True}]}
    assert jd.fold({"a": 1.0, "b": 1.0, "c": None}, {}, spec) == 4
    assert jd.fold({"a": 0.0, "b": 0.0, "c": 1.0}, {}, spec) == 1     # c counts when scored
    assert jd.fold({"a": 0.0, "b": 0.0, "c": None}, {}, spec) == 0
    assert jd.fold({"a": None, "b": None, "c": None}, {}, spec) == 0  # nothing applicable
    # a weight that is not 1 is honoured
    heavy = {"criteria": [{"id": "a", "definition": "d", "weight": 3},
                          {"id": "b", "definition": "d"}]}
    assert jd.fold({"a": 1.0, "b": 0.0}, {}, heavy) == 3              # 4 × 3/4 = 3


def test_the_flag_effects_land_after_the_fold_and_in_file_order():
    good = {cid: 1.0 for cid in jd.criteria_ids(LAW)}
    assert jd.fold(good, {"critical_legal_error": False, "fabricated_authority": False},
                   LAW) == 4
    # a cap brings a 4 down to its ceiling
    capped = {"critical_legal_error": False, "fabricated_authority": True}
    assert jd.fold(good, capped, LAW) == 1
    assert jd.effects_applied(capped, LAW) == ["fabricated_authority"]
    # a zero beats everything, and both together is still 0: zero_score comes
    # first in the file, and a cap cannot raise a score
    both = {"critical_legal_error": True, "fabricated_authority": True}
    assert jd.fold(good, both, LAW) == 0
    assert jd.effects_applied(both, LAW) == ["critical_legal_error", "fabricated_authority"]
    assert jd.fold({cid: 1.0 for cid in IDS}, {"critical_safety_failure": True}, SPEC) == 0
    # order is the file's: a cap written after a zero still cannot raise it
    reversed_file = {"criteria": LAW["criteria"], "flags": list(reversed(LAW["flags"]))}
    assert jd.fold(good, both, reversed_file) == 0
    # a weak answer that is not flagged keeps its folded score
    weak = {cid: 0.3 for cid in jd.criteria_ids(LAW)}
    assert jd.fold(weak, {"critical_legal_error": False, "fabricated_authority": True},
                   LAW) == 1


# ---------------------------------------------------------------------------
# what judge.json records
# ---------------------------------------------------------------------------

def _item(qid: str, half: str, meta: dict | None = None, cid: str = "judge:x") -> dict:
    return {"cid": cid, "qid": qid, "half": half, "doc_hash": qid[:8], "id": "i",
            "category": TOPIC, "answer_words": 20,
            "meta": meta if meta is not None else {"acuity": "moderate"}}


def _stub() -> dict:
    return {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1", "family": "stub"}


def _assemble(items_and_replies, tmp_path, task=TASK):
    from service import llm
    plan = {"model": "m", "model_dir": str(tmp_path), "canary": [], "skipped": None, "tasks": {}}
    plan["tasks"][task] = [it for it, _ in items_and_replies]
    results = {it["cid"]: llm.Result(text=text) for it, text in items_and_replies if text}
    return jd.assemble(plan, results, _stub(), "b", tmp_path, 0.5, False, record=False)


def test_judge_json_carries_the_criteria_the_flags_and_the_breakdowns(tmp_path):
    good = {cid: 1.0 for cid in IDS}
    weak = {cid: 0.25 for cid in IDS}
    fail = {"critical_safety_failure": True}
    rows = [
        (_item("a" * 64, "report", {"acuity": "emergency", "difficulty": 3}, "c1"),
         reply(criteria=good)),
        (_item("b" * 64, "diagnose", {"acuity": "emergency", "difficulty": 3}, "c2"),
         reply(criteria=good, flags=fail)),
        (_item("c" * 64, "report", {"acuity": "mild", "difficulty": 1}, "c3"),
         reply(criteria=good, flags=fail)),
        (_item("d" * 64, "diagnose", {"acuity": "mild", "difficulty": 1}, "c4"),
         reply(criteria=weak)),
        (_item("e" * 64, "diagnose", {"acuity": "routine", "difficulty": 2}, "c5"),
         "the model wandered off"),
    ]
    t = _assemble(rows, tmp_path)["tasks"][TASK]
    assert [it["score"] for it in t["items"]] == [4, 0, 0, 1, 0]
    assert t["items"][1]["flags"] == fail
    assert t["items"][1]["fold"]["effects_applied"] == ["critical_safety_failure"]
    assert t["items"][0]["fold"] == {"method": jd.FOLD_METHOD, "applicable": len(IDS),
                                     "effects_applied": []}
    assert t["items"][4]["graded"] is False
    # per criterion: the mean over items where it was scored, and how many
    assert t["criteria_mean"]["triage"] == pytest.approx((1 + 1 + 1 + 0.25) / 4)
    assert t["criteria_n"]["triage"] == 4
    assert t["criteria_labels"]["red_flag_coverage"] == "Red flag coverage"
    assert t["unparseable"] == 1
    flag = t["flags"]["critical_safety_failure"]
    assert flag["n"] == 2 and flag["share"] == 0.4          # both halves count
    assert flag["qids"] == ["b" * 64]                       # only the diagnose-half one
    assert flag["effect"] == "zero_score" and flag["label"] == "Critical safety failure"
    assert flag["effect_words"] == "sets the whole score to 0"
    # a table per field the items carry, each cell counting every flag
    assert list(t["breakdowns"]) == ["acuity", "difficulty"]
    assert t["breakdowns"]["acuity"]["emergency"] == {
        "n": 2, "mean": 2.0, "flags": {"critical_safety_failure": 1}}
    assert t["breakdowns"]["difficulty"]["1"]["n"] == 2
    # acuity most severe first, difficulty in numeric order
    assert list(t["breakdowns"]["acuity"]) == ["emergency", "mild", "routine"]
    assert list(t["breakdowns"]["difficulty"]) == ["1", "2", "3"]
    # the folded score is what everything downstream reads
    assert t["score_report"] == 2.0 and t["n_report"] == 2


def test_a_report_half_qid_is_never_listed_even_when_it_was_flagged(tmp_path):
    rows = [(_item("f" * 64, "report", {"acuity": "emergency"}, "c1"),
             reply(flags={"critical_safety_failure": True}))]
    t = _assemble(rows, tmp_path)["tasks"][TASK]
    assert t["flags"]["critical_safety_failure"]["n"] == 1
    assert t["flags"]["critical_safety_failure"]["qids"] == []


def test_a_law_task_is_tabulated_by_its_own_fields_and_both_flags(tmp_path):
    good = {cid: 1.0 for cid in jd.criteria_ids(LAW)}
    rows = [
        (_item("a" * 64, "diagnose", {"difficulty": 5, "intent": "procedural_guidance",
                                      "acuity": "urgent"}, "c1"),
         reply(LAW, criteria=good, flags={"critical_legal_error": False,
                                          "fabricated_authority": True})),
        (_item("b" * 64, "diagnose", {"difficulty": 1, "intent": "legal_information",
                                      "acuity": "routine"}, "c2"),
         reply(LAW, criteria=good)),
    ]
    t = _assemble(rows, tmp_path, LAW_TASK)["tasks"][LAW_TASK]
    assert [it["score"] for it in t["items"]] == [1, 4]          # the cap, then no flag
    assert set(t["flags"]) == {"critical_legal_error", "fabricated_authority"}
    assert t["flags"]["fabricated_authority"]["effect_words"] == "caps the whole score at 1 of 4"
    assert t["flags"]["critical_legal_error"]["n"] == 0
    assert list(t["breakdowns"]) == ["acuity", "difficulty", "intent"]
    assert t["breakdowns"]["difficulty"]["5"] == {
        "n": 1, "mean": 1.0, "flags": {"critical_legal_error": 0, "fabricated_authority": 1}}
    assert list(t["breakdowns"]["intent"]) == ["legal_information", "procedural_guidance"]


def test_a_task_without_criteria_gains_nothing(tree):
    j = json.loads((tree["models"]["fx/good-750m"]["dir"] / "judge.json").read_text())
    econ, med = j["tasks"]["exam_history"], j["tasks"][TASK]
    for key in ("criteria_mean", "criteria_n", "criteria_labels", "flags", "breakdowns",
                "unparseable"):
        assert key not in econ, key
        assert key in med, key
    assert all("criteria" not in it for it in econ["items"])
    assert all("criteria" in it and "fold" in it for it in med["items"])
    assert econ["mean"] is not None and econ["score_report"] is not None


def test_every_folded_score_in_the_fixture_recomputes_from_what_was_recorded(tree):
    """The fold is the one thing that must not drift: the score on disk is
    recomputed here from the item's own criteria and flags."""
    checked = 0
    for name, m in tree["models"].items():
        jf = m["dir"] / "judge.json"
        if not jf.exists():
            continue
        j = json.loads(jf.read_text(encoding="utf-8"))
        for task, t in j["tasks"].items():
            spec = jd.rubric_for(task).criteria
            if not spec:
                continue
            for it in t["items"]:
                if not it.get("graded"):
                    continue
                assert it["score"] == jd.fold(it["criteria"], it["flags"], spec), (name, task)
                assert it["fold"]["effects_applied"] == jd.effects_applied(it["flags"], spec)
                checked += 1
    assert checked > 20


# ---------------------------------------------------------------------------
# what the numbers are used for
# ---------------------------------------------------------------------------

def test_the_gap_finder_carries_labels_and_numbers_and_nothing_else(tree):
    d = tree["models"]["fx/good-750m"]["dir"]
    ev = prop.criteria_evidence(d, TASK)
    assert len(ev["weakest_criteria"]) == 3
    assert all(set(c) == {"id", "label", "mean", "n"} for c in ev["weakest_criteria"])
    assert ev["breakdowns"]["acuity"]
    assert all(set(v) == {"n", "mean"} for v in ev["breakdowns"]["acuity"].values())
    assert prop.criteria_evidence(d, "exam_history") == {}
    items, counts = prop.justifications_for(d, TASK)
    req = prop.proposal_request(1, "m", TASK, TOPIC, items, counts,
                                jd.rubric_for(TASK).text, ev)
    body = req.system + "\n" + req.user
    assert "by criterion" in body.lower() or "criterion by criterion" in body.lower()
    for c in ev["weakest_criteria"]:
        assert f"{c['label']}: {c['mean']}" in body
    assert "Mean score of 4 by acuity:" in body
    # no question text, no qids — the same rule as everything else in this request
    exam_root = tree["judged"]["exam_root"]
    for b in eb.load_bank(exam_root)[TOPIC]:
        assert b["prompt"] not in body
        assert b["qid"] not in body
    assert "critical_safety_failure" not in body      # the flag's id is not evidence


def test_calibration_exports_a_column_per_criterion_and_per_flag(tree, tmp_path):
    out = tmp_path / "cal.csv"
    n = jc.export(tree["out_dir"], out, [], 40, 7)
    assert n == 40
    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames), list(reader)
    # a flag column is prefixed, because a criterion may carry the same id
    assert "human_flag_critical_safety_failure" in fields
    assert all(f"human_{cid}" in fields for cid in IDS)
    if any(r["task"] == LAW_TASK for r in rows):
        assert "human_flag_fabricated_authority" in fields
        assert "human_fabricated_authority" in fields
    assert all(r["human_score"] == "" for r in rows)          # the judge's score is hidden
    assert not any(f.startswith("judge_") for f in fields)
    # a grader who marked the same things: kappa is still on the folded score,
    # and per-criterion and per-flag agreement ride beside it
    judged = {r["id"]: r for r in jc._judged_rows(tree["out_dir"], set())}
    for r in rows:
        j = judged[r["id"]]
        r["human_score"] = str(j["judge_score"])
        for cid, v in (j["judge_criteria"] or {}).items():
            r[f"human_{cid}"] = "" if v is None else f"{min(1.0, v + 0.1):.2f}"
        for fid, v in (j["judge_flags"] or {}).items():
            r[f"human_flag_{fid}"] = "true" if v else "false"
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
    flag = res["per_flag"]["critical_safety_failure"]
    assert flag["agreement"] == 1.0 and flag["label"] == "Critical safety failure"
