"""Phases 8b and 8d: the criteria file is the author's, and the judge reads it.

The judge scores each criterion 0–1 and answers true or false for each flag;
the 0–4 everything downstream reads is folded from them IN CODE — the mean
of the criteria, then each true flag's effect in the file's order. The model
is never asked for the 0–4, so a grade is reproducible from what was
recorded. Nothing about medicine or law is special-cased: both files are
read by one loader, and a topic's tables come from the metadata its own
items carry.

Phase 10 replaced both files. The new Medicine & Clinical Health file is
the one the fixture's topic is graded by, so it is SPEC here. The law file
delivered in 8d is retired, but it is still the only file with two flags —
one zeroing, one capping — and a flag that shares its id with a criterion,
so every test of that path reads it from where it was retired to (nothing
in the service reads that folder)."""

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
TASK = "exam_medicine_clinical_health"
TOPIC = "Medicine & Clinical Health"
LAW_TASK = "exam_law"
SPEC = jd.rubric_for(TASK).criteria
NEW_LAW = jd.rubric_for(LAW_TASK).criteria
RETIRED = REPO / "eval_tasks" / "fr" / "retired" / "rubrics"
LAW_FILE = RETIRED / "law.criteria.json"
# read the way rubric_for reads a file, path and all, so an effect it could
# not apply would refuse here too
LAW = jd.normalise_criteria(json.loads(LAW_FILE.read_text(encoding="utf-8")), LAW_FILE)
LAW_PROSE = (RETIRED / "law.md").read_text(encoding="utf-8")
IDS = jd.criteria_ids(SPEC)
FLAGS = jd.flag_ids(SPEC)
# the topic whose files the shared-rubric tests hide (conftest.arts_without_rubric):
# every topic has its own since Arts arrived, 2026-09-22
NO_TOPIC = "exam_arts"


def reply(spec=SPEC, **over) -> str:
    body = {"criteria": {cid: 1.0 for cid in jd.criteria_ids(spec)},
            "flags": {fid: False for fid in jd.flag_ids(spec)},
            "justification": "covers the escalation advice"}
    body.update(over)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# the author's schema, loaded as delivered
# ---------------------------------------------------------------------------

def test_both_delivered_files_load_and_say_what_they_grade(arts_without_rubric):
    assert jd.validate_criteria(SPEC) == [] and jd.validate_criteria(NEW_LAW) == []
    assert len(IDS) == 20 and len(set(IDS)) == 20
    assert len(jd.criteria_ids(NEW_LAW)) == 20
    assert FLAGS == ["critical_medicine_clinical_health_error"]
    assert jd.flag_ids(NEW_LAW) == ["critical_legal_error"]
    # the retired law file still loads, with the second flag the tests below need
    assert jd.validate_criteria(LAW) == [] and len(jd.criteria_ids(LAW)) == 23
    assert jd.flag_ids(LAW) == ["critical_legal_error", "fabricated_authority"]
    # what a file says about itself is informational — the FILE NAME decides
    # which task it grades. The retired files said `topic`; the new ones say
    # `task` or `benchmark`, or nothing, and grade the same way
    assert LAW["topic"] == "law" and "topic" not in SPEC and NEW_LAW["benchmark"] == "Law"
    assert jd.rubric_name(TASK) == "medicine_clinical_health"
    assert jd.rubric_name(LAW_TASK) == "law"
    assert jd.rubric_for(NO_TOPIC).criteria is None      # no file, no criteria path
    assert jd.rubric_for(NO_TOPIC).fallback is True
    for r in (jd.rubric_for(TASK), jd.rubric_for(LAW_TASK)):
        assert len(r.criteria_sha256) == 64
        # the author wrote these anchors himself: nothing is stamped DRAFT
        assert r.status == ""
    # a label is optional in the schema; the page still has words for every id
    assert jd.criteria_labels(LAW)["jurisdiction_awareness"] == "Jurisdiction awareness"
    assert jd.criteria_labels(SPEC)["triage_and_urgency"] == "Triage and urgency"


def test_equal_weights_with_the_key_and_without_it():
    # the retired law file says "equal" and puts 1.0 on every row; the new
    # medicine file leaves the key out and puts 0.05 on every row — equal too
    assert LAW["weights"] == "equal" and "weights" not in SPEC
    assert all(jd.weight_of(LAW, cid) == 1.0 for cid in jd.criteria_ids(LAW))
    assert all(jd.weight_of(SPEC, cid) == 0.05 for cid in IDS)
    # so twenty weights of a twentieth fold exactly as twenty of one would
    halves = {cid: float(i % 2) for i, cid in enumerate(IDS)}
    unweighted = {"criteria": [{"id": cid, "definition": "d"} for cid in IDS]}
    assert jd.fold(halves, NO_FLAGS, SPEC) == jd.fold(halves, {}, unweighted) == 2
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
    # installed where an upload from the page lands, which the judge reads
    # before the repo's own copy
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "rubrics" / "law.md").write_text(LAW_PROSE, encoding="utf-8")
    (tmp_path / "rubrics" / "law.criteria.json").write_text(json.dumps(spec), encoding="utf-8")
    from service import config
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    with pytest.raises(jd.CriteriaError) as e:
        jd.rubric_for(LAW_TASK)
    # the file and the string it could not apply, both named
    assert "melt_the_score" in str(e.value) and "law.criteria.json" in str(e.value)
    assert "fabricated_authority" in str(e.value)
    # every effect the retired law file asks for is one it can apply, and the
    # new files' "score=0" is read as the same zero
    assert {jd.effect_of(f) for f in LAW["flags"]} == {"zero_score", "cap_at_1_of_4"}
    assert [jd.effect_of(f) for f in SPEC["flags"]] == ["zero_score"]
    assert jd.effect_words("zero_score") == "sets the whole score to 0"
    assert jd.effect_words("cap_at_1_of_4") == "caps the whole score at 1 of 4"


def test_a_criterion_and_a_flag_may_share_an_id_and_stay_apart():
    """The retired law file scores `fabricated_authority` 0–1 AND flags it:
    the criterion is about invention at all, the flag about invention that
    carries the answer. No new file does, and the loader still may not
    conflate the two when one arrives."""
    assert "fabricated_authority" in jd.criteria_ids(LAW)
    assert "fabricated_authority" in jd.flag_ids(LAW)
    text = jd.build_criteria_prompt(LAW_PROSE, LAW, "Can he evict me?",
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


def test_the_criteria_prompt_carries_every_criterion_and_every_flag(arts_without_rubric):
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
    assert jd.prompt_for(NO_TOPIC) is jd.PROMPT      # no criteria file of its own


def test_a_conditional_criterion_is_told_when_to_return_null():
    """The new files mark most criteria conditional and never say when one
    applies; the prompt says what to do either way."""
    lines = dict(ln.split(" — ", 1) for ln in jd.criteria_block(SPEC).splitlines())
    assert len(jd.conditional_ids(SPEC)) == 13
    assert "CONDITIONAL. Return null for it" in lines["triage_and_urgency"]
    assert "CONDITIONAL" not in lines["relevance"]
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
    assert g.flags == {"critical_medicine_clinical_health_error": False}
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
    assert jd.parse_grade_criteria(reply(flags={FLAGS[0]: "false"}), SPEC) is None
    # a missing non-conditional criterion is a hole, not a zero
    scores = {cid: 1.0 for cid in IDS if cid != "medical_accuracy"}
    assert jd.parse_grade_criteria(reply(criteria=scores), SPEC) is None
    # and a renamed one is not accepted in its place
    scores["accuracy_medical"] = 1.0
    assert jd.parse_grade_criteria(reply(criteria=scores), SPEC) is None
    # (a missing conditional one is the null the file allows)
    scores = {cid: 1.0 for cid in IDS if cid != "triage_and_urgency"}
    assert jd.parse_grade_criteria(reply(criteria=scores), SPEC).criteria[
        "triage_and_urgency"] is None
    assert jd.parse_grade_criteria("not json", SPEC) is None
    assert jd.parse_grade_criteria(reply(criteria=["a list"]), SPEC) is None
    assert jd.parse_grade_criteria(reply(criteria={cid: "high" for cid in IDS}), SPEC) is None


def test_out_of_range_is_clamped_and_strays_are_counted():
    scores = {cid: 0.5 for cid in IDS}
    scores.update({"medical_accuracy": 7, "clinical_reasoning": -2, "relevance": "0.75"})
    scores["overall_score"] = 4          # a model adding a key of its own
    g = jd.parse_grade_criteria(reply(criteria=scores,
                                      flags={FLAGS[0]: False, "vibes": True}), SPEC)
    assert g.criteria["medical_accuracy"] == 1.0 and g.criteria["clinical_reasoning"] == 0.0
    assert g.criteria["relevance"] == 0.75
    assert g.extra_keys == ["overall_score", "flags.vibes"]
    assert "vibes" not in g.flags


# ---------------------------------------------------------------------------
# the fold — one function, and the only place the 0-4 comes from
# ---------------------------------------------------------------------------

NO_FLAGS = {FLAGS[0]: False}


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
    assert jd.fold({cid: 1.0 for cid in IDS}, {FLAGS[0]: True}, SPEC) == 0
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
    fail = {FLAGS[0]: True}
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
    assert t["items"][1]["fold"]["effects_applied"] == [FLAGS[0]]
    assert t["items"][0]["fold"] == {"method": jd.FOLD_METHOD, "applicable": len(IDS),
                                     "effects_applied": []}
    assert t["items"][4]["graded"] is False
    # per criterion: the mean over items where it was scored, and how many
    assert t["criteria_mean"]["medical_accuracy"] == pytest.approx((1 + 1 + 1 + 0.25) / 4)
    assert t["criteria_n"]["medical_accuracy"] == 4
    assert t["criteria_labels"]["triage_and_urgency"] == "Triage and urgency"
    assert t["unparseable"] == 1
    flag = t["flags"][FLAGS[0]]
    assert flag["n"] == 2 and flag["share"] == 0.4          # both halves count
    assert flag["qids"] == ["b" * 64]                       # only the diagnose-half one
    # the flag carries no label of its own, so its id is the words
    assert flag["effect"] == "zero_score"
    assert flag["label"] == "Critical medicine clinical health error"
    assert flag["effect_words"] == "sets the whole score to 0"
    # a table per field the items carry, each cell counting every flag
    assert list(t["breakdowns"]) == ["acuity", "difficulty"]
    assert t["breakdowns"]["acuity"]["emergency"] == {
        "n": 2, "mean": 2.0, "flags": {FLAGS[0]: 1}}
    assert t["breakdowns"]["difficulty"]["1"]["n"] == 2
    # acuity most severe first, difficulty in numeric order
    assert list(t["breakdowns"]["acuity"]) == ["emergency", "mild", "routine"]
    assert list(t["breakdowns"]["difficulty"]) == ["1", "2", "3"]
    # the folded score is what everything downstream reads
    assert t["score_report"] == 2.0 and t["n_report"] == 2


def test_a_report_half_qid_is_never_listed_even_when_it_was_flagged(tmp_path):
    rows = [(_item("f" * 64, "report", {"acuity": "emergency"}, "c1"),
             reply(flags={FLAGS[0]: True}))]
    t = _assemble(rows, tmp_path)["tasks"][TASK]
    assert t["flags"][FLAGS[0]]["n"] == 1
    assert t["flags"][FLAGS[0]]["qids"] == []


def test_a_law_task_is_tabulated_by_its_own_fields_and_both_flags(tmp_path, monkeypatch):
    # the retired law file, installed where an upload from the page lands:
    # the task is graded by the file the judge finds, and only that one caps
    from service import config
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "rubrics" / "law.md").write_text(LAW_PROSE, encoding="utf-8")
    (tmp_path / "rubrics" / "law.criteria.json").write_bytes(LAW_FILE.read_bytes())
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    assert jd.rubric_for(LAW_TASK).criteria == LAW
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
    # every topic the fixture sits now has a criteria file; the control set,
    # graded by the factual rubric alone, is the task that has none
    plain, med = j["tasks"][eb.CONTROL_TASK], j["tasks"][TASK]
    assert jd.rubric_for(eb.CONTROL_TASK).criteria is None
    for key in ("criteria_mean", "criteria_n", "criteria_labels", "flags", "breakdowns",
                "unparseable"):
        assert key not in plain, key
        assert key in med, key
    assert all("criteria" not in it for it in plain["items"])
    assert all("criteria" in it and "fold" in it for it in med["items"])
    # its plain 0-4 is still there — on the diagnose half, the only one the
    # control set has
    assert plain["mean"] is not None and plain["score_diagnose"] is not None


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
    assert prop.criteria_evidence(d, eb.CONTROL_TASK) == {}
    items, counts = prop.justifications_for(d, TASK)
    rubric = jd.rubric_for(TASK).text
    req = prop.proposal_request(1, "m", TASK, TOPIC, items, counts, rubric, ev)
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
    # the flag's id is not evidence: the new rubric's own prose names it, and
    # nothing the evidence adds does
    assert FLAGS[0] in rubric and FLAGS[0] not in body.replace(rubric.strip(), "")


def test_calibration_exports_a_column_per_criterion_and_per_flag(tree, tmp_path):
    out = tmp_path / "cal.csv"
    # the sample is round-robin over (topic, score) in alphabetical order: with
    # 36 topics, Law and Medicine only come round after some seventy rows
    n = jc.export(tree["out_dir"], out, [], 120, 7)
    assert n == 120
    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames), list(reader)
    assert {LAW_TASK, TASK} <= {r["task"] for r in rows}
    # a flag column is prefixed, because a criterion may carry the same id
    # (the retired law file's fabricated_authority did)
    assert f"human_flag_{FLAGS[0]}" in fields and f"human_{FLAGS[0]}" not in fields
    assert all(f"human_{cid}" in fields for cid in IDS)
    assert "human_flag_critical_legal_error" in fields
    assert all(f"human_{cid}" in fields for cid in jd.criteria_ids(NEW_LAW))
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
    assert res["per_criterion"]["medical_accuracy"]["n"] > 0
    # the grader here marked every criterion 0.1 above the judge, clamped at
    # 1.0 — so the mean difference is positive and at most that offset
    assert 0 < res["per_criterion"]["medical_accuracy"]["mean_abs_diff"] <= 0.1
    flag = res["per_flag"][FLAGS[0]]
    assert flag["agreement"] == 1.0
    assert flag["label"] == "Critical medicine clinical health error"
