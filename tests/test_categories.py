"""scripts/categories.yaml: every MMLU subject, exactly once, rolled up right."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import categories
import diagnose as dx
import make_fixture

# lm_eval/tasks/mmlu/default/mmlu_<subject>.yaml at tag v0.4.12 — the harness
# this board is pinned to. If the harness list ever changes, this list is what
# a reviewer updates, and categories.yaml must follow in the same commit.
HARNESS_SUBJECTS_0_4_12 = sorted("""
abstract_algebra anatomy astronomy business_ethics clinical_knowledge college_biology
college_chemistry college_computer_science college_mathematics college_medicine
college_physics computer_security conceptual_physics econometrics electrical_engineering
elementary_mathematics formal_logic global_facts high_school_biology high_school_chemistry
high_school_computer_science high_school_european_history high_school_geography
high_school_government_and_politics high_school_macroeconomics high_school_mathematics
high_school_microeconomics high_school_physics high_school_psychology
high_school_statistics high_school_us_history high_school_world_history human_aging
human_sexuality international_law jurisprudence logical_fallacies machine_learning
management marketing medical_genetics miscellaneous moral_disputes moral_scenarios
nutrition philosophy prehistory professional_accounting professional_law
professional_medicine professional_psychology public_relations security_studies
sociology us_foreign_policy virology world_religions
""".split())

CATEGORIES = ["economics", "law", "medicine & health", "mathematics", "computer science",
              "physics & engineering", "chemistry & biology", "history",
              "philosophy & religion", "politics & government", "psychology & sociology",
              "business & accounting", "geography & world facts", "language & logic", "other"]


def test_the_harness_has_57_subjects():
    assert len(HARNESS_SUBJECTS_0_4_12) == 57


def test_every_subject_is_mapped_exactly_once():
    m = categories.load()
    assert sorted(m) == HARNESS_SUBJECTS_0_4_12
    # exactly once is what the parser enforces; a duplicate is a ValueError
    # (test_parser_refuses_a_subject_listed_twice), so the set equality above
    # is the whole claim
    assert set(m.values()) <= set(CATEGORIES)


def test_the_categories_are_the_agreed_fifteen():
    assert categories.category_order() == CATEGORIES
    assert categories.category_order()[-1] == categories.OTHER
    assert all(categories.categorize(s) for s in HARNESS_SUBJECTS_0_4_12)
    assert categories.categorize("underwater_basketweaving") is None


def test_matches_the_installed_harness():
    """When lm_eval is importable (the server, not CI), its own MMLU task index
    is the authority — pinned lists drift, package contents do not."""
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    subjects = sorted(name[5:] for name, meta in tm.task_index.items()
                      if name.startswith("mmlu_")
                      and "tasks/mmlu/default/" in str(meta.get("yaml_path", ""))
                      and meta.get("type") == "task")
    assert subjects == HARNESS_SUBJECTS_0_4_12


def test_parser_refuses_a_subject_listed_twice():
    with pytest.raises(ValueError, match="already under"):
        categories.parse("a:\n  - x\nb:\n  - x\n")
    with pytest.raises(ValueError, match="listed twice"):
        categories.parse("a:\n  - x\na:\n  - y\n")
    with pytest.raises(ValueError, match="expected"):
        categories.parse("a:\n  - x\n  nested: deeper\n")
    with pytest.raises(ValueError, match="expected"):
        categories.parse("  - orphan\n")
    assert categories.parse("# only a comment\n\n") == {}
    assert categories.parse("a b:\n  - x  # trailing\n") == {"a b": ["x"]}


def test_fixture_subjects_all_map():
    m = categories.load()
    assert {m[s] for s in make_fixture.MMLU_SUBJECTS} == {
        "mathematics", "medicine & health", "economics", "politics & government",
        "philosophy & religion", "law"}


def test_rollup_arithmetic_matches_the_groups(diag):
    checked = 0
    for mid, d in diag.items():
        v = d["tasks"]["mmlu"]
        assert v["unmapped"] == [], mid
        placed = [s for c in v["categories"].values() for s in c["groups"]]
        assert sorted(placed) == sorted(v["groups"]), mid       # every subject, once
        for cat, c in v["categories"].items():
            subs = [v["groups"][s] for s in c["groups"]]
            assert c["n"] == sum(g["n"] for g in subs)
            assert c["n_report"] == sum(g["n_report"] for g in subs)
            hits = sum(g["score_report"] * g["n_report"] for g in subs)
            assert c["score_report"] == pytest.approx(hits / c["n_report"], abs=1e-5)
            for b in set(k for g in subs for k in g["buckets"]):
                assert c["buckets"][b] == sum(g["buckets"].get(b, 0) for g in subs)
            assert categories.categorize(c["groups"][0]) == cat
            checked += 1
        # file order, not alphabetical, so the page lays categories out consistently
        order = categories.category_order()
        assert list(v["categories"]) == [c for c in order if c in v["categories"]]
        # the control is a flat group of subjects and rolls up too
        if "mmlu_perm" in d["tasks"]:
            assert set(d["tasks"]["mmlu_perm"]["categories"]) == {
                "medicine & health", "economics", "politics & government",
                "philosophy & religion"}
    assert checked >= 6 * 6


def test_noise_floor_is_shared_by_page_and_file(diag):
    assert dx.MIN_GROUP_N == 30
    for d in diag.values():
        assert d["thresholds"]["min_group_n"] == 30
    # the fixture puts two-subject categories above the floor and one-subject ones below
    v = diag["fx/good-750m"]["tasks"]["mmlu"]
    above = {c for c, g in v["categories"].items() if g["n_report"] >= 30}
    below = {c for c, g in v["categories"].items() if g["n_report"] < 30}
    assert above == {"economics", "medicine & health"} and len(below) == 4


def _rec(subject: str, correct: int, pick: int) -> dict:
    import hashlib
    import math
    probs = [0.6 if i == pick else 0.4 / 3 for i in range(4)]
    doc = {"question": f"q {subject} {correct} {pick}", "subject": subject,
           "choices": ["a", "bb", "ccc", "dddd"], "answer": correct}
    return {"doc": doc, "target": correct, "acc": float(pick == correct), "filter": "none",
            "metrics": ["acc"], "doc_hash": hashlib.sha256(json.dumps(doc).encode()).hexdigest(),
            "filtered_resps": [[f"{math.log(p):.4f}", str(i == pick)] for i, p in enumerate(probs)]}


def test_unmapped_groups_roll_into_other_and_are_listed(tmp_path: Path):
    recs = [_rec("anatomy", i % 4, (i * 3) % 4) for i in range(40)]
    recs += [_rec("underwater_basketweaving", i % 4, i % 4) for i in range(40)]
    recs += [_rec("nutrition", i % 4, 0) for i in range(40)]
    f = tmp_path / "samples_t_2026-01-01T00-00-00.000000.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = dx.diagnose_task([f])
    assert out["unmapped"] == ["underwater_basketweaving"]
    assert set(out["categories"]) == {"medicine & health", "other"}
    assert out["categories"]["medicine & health"]["groups"] == ["anatomy", "nutrition"]
    assert out["categories"]["other"]["groups"] == ["underwater_basketweaving"]
    assert out["categories"]["other"]["n"] == 40
    assert list(out["categories"]) == ["medicine & health", "other"]     # other is last


def test_groups_that_are_not_subjects_get_no_categories_block(tmp_path: Path):
    recs = [_rec("alpha", i % 4, i % 4) for i in range(20)]
    recs += [_rec("beta", i % 4, 0) for i in range(20)]
    f = tmp_path / "samples_t_2026-01-01T00-00-00.000000.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = dx.diagnose_task([f])
    assert out["groups"] and "categories" not in out and "unmapped" not in out


def test_payload_carries_the_categories(payload, diag):
    assert payload["meta"]["categories"] == CATEGORIES
    row = next(m for m in payload["models"] if m["id"] == "fx/good-750m")
    t = row["diag"]["tasks"]["mmlu"]
    assert t["categories"] == diag["fx/good-750m"]["tasks"]["mmlu"]["categories"]
    assert t["unmapped"] == []
