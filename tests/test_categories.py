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

# the 37 folders Omar delivered (phase 10), named exactly as written, in file
# order — with the fallback, General & Multidisciplinary, last
CATEGORIES = [
    "Agriculture", "AI & Machine Learning", "Anthropology & Human Geography",
    "Architecture & Built Environment", "Arts", "Biology & Life Sciences",
    "Business & Management", "Chemistry & Materials Science", "Computer Science",
    "Data & Information Science", "Design", "Earth & Environmental Sciences", "Economics",
    "Education", "Engineering", "Ethics & Religion", "Finance & Accounting",
    "Food & Veterinary Sciences", "Government & Public Policy", "History & Archaeology", "IT",
    "Language & Literature", "Law", "Manufacturing & Applied Sciences",
    "Mathematics & Statistics", "Media & Communication", "Medicine & Clinical Health",
    "Philosophy", "Physics & Astronomy", "Political Science & International Relations",
    "Psychology & Cognitive Sciences", "Public Health & Wellness", "Sociology",
    "Software Engineering & Programming", "Systems & Cybersecurity", "Technology",
    "General & Multidisciplinary"]

# the brief's §2 table, as written (Omar can change it; this is the proposal)
MAPPING = {
    "AI & Machine Learning": ["machine_learning"],
    "Anthropology & Human Geography": ["high_school_geography"],
    "Biology & Life Sciences": ["college_biology", "high_school_biology"],
    "Business & Management": ["management", "marketing"],
    "Chemistry & Materials Science": ["college_chemistry", "high_school_chemistry"],
    "Computer Science": ["college_computer_science", "high_school_computer_science"],
    "Economics": ["econometrics", "high_school_macroeconomics", "high_school_microeconomics"],
    "Engineering": ["electrical_engineering"],
    "Ethics & Religion": ["business_ethics", "moral_disputes", "moral_scenarios",
                          "world_religions"],
    "Finance & Accounting": ["professional_accounting"],
    "General & Multidisciplinary": ["global_facts", "miscellaneous"],
    "Government & Public Policy": ["high_school_government_and_politics"],
    "History & Archaeology": ["high_school_european_history", "high_school_us_history",
                              "high_school_world_history", "prehistory"],
    "Law": ["international_law", "jurisprudence", "professional_law"],
    "Mathematics & Statistics": ["abstract_algebra", "college_mathematics",
                                 "elementary_mathematics", "high_school_mathematics",
                                 "high_school_statistics"],
    "Media & Communication": ["public_relations"],
    "Medicine & Clinical Health": ["anatomy", "clinical_knowledge", "college_medicine",
                                   "medical_genetics", "professional_medicine", "virology"],
    "Philosophy": ["philosophy", "formal_logic", "logical_fallacies"],
    "Physics & Astronomy": ["astronomy", "college_physics", "conceptual_physics",
                            "high_school_physics"],
    "Political Science & International Relations": ["security_studies", "us_foreign_policy"],
    "Psychology & Cognitive Sciences": ["high_school_psychology", "professional_psychology"],
    "Public Health & Wellness": ["human_aging", "human_sexuality", "nutrition"],
    "Sociology": ["sociology"],
    "Systems & Cybersecurity": ["computer_security"],
}
NO_MMLU = ["Agriculture", "Architecture & Built Environment", "Arts",
           "Data & Information Science", "Design", "Earth & Environmental Sciences", "Education",
           "Food & Veterinary Sciences", "IT", "Language & Literature",
           "Manufacturing & Applied Sciences", "Software Engineering & Programming", "Technology"]


def test_the_harness_has_57_subjects():
    assert len(HARNESS_SUBJECTS_0_4_12) == 57


def test_every_subject_is_mapped_exactly_once():
    m = categories.load()
    assert sorted(m) == HARNESS_SUBJECTS_0_4_12
    # exactly once is what the parser enforces; a duplicate is a ValueError
    # (test_parser_refuses_a_subject_listed_twice), so the set equality above
    # is the whole claim
    assert set(m.values()) <= set(CATEGORIES)


def test_the_topics_are_the_37_folders_and_the_mapping_is_the_briefs():
    assert len(CATEGORIES) == 37
    assert categories.category_order() == CATEGORIES
    assert categories.OTHER == "General & Multidisciplinary"
    table = categories._table(categories.YAML_PATH)
    assert {c: s for c, s in table.items() if s} == MAPPING
    assert sorted(c for c, s in table.items() if not s) == sorted(NO_MMLU)
    assert categories.with_subjects() == [c for c in CATEGORIES if c in MAPPING]
    assert len(categories.with_subjects()) == 24
    # every topic name becomes exactly the slug the delivered files use
    import os
    delivered = {f[:-len("_v1.json")] for f in os.listdir(
        Path(__file__).resolve().parents[1] / "eval_tasks" / "fr" / "banks")}
    assert delivered == {categories.topic_slug(c) for c in CATEGORIES}
    assert categories.topic_slug("Medicine & Clinical Health") == "medicine_clinical_health"
    assert categories.topic_slug("IT") == "it"
    assert categories.category_order()[-1] == categories.OTHER
    assert all(categories.categorize(s) for s in HARNESS_SUBJECTS_0_4_12)
    assert categories.categorize("underwater_basketweaving") is None


def _field(meta, name):
    """task_index's values were dicts; the installed 0.4.12 gives Entry
    objects (the live check, 2026-09-24: 'Entry' object has no attribute
    'get'). Read either."""
    return meta.get(name) if isinstance(meta, dict) else getattr(meta, name, None)


def _is_plain_task(meta) -> bool:
    """"task" — not a group, a tag or a python task — as a string or an enum"""
    t = _field(meta, "type")
    if t is None:
        t = _field(meta, "kind")
    said = {str(t).lower(), str(getattr(t, "value", "")).lower(),
            str(getattr(t, "name", "")).lower()}
    return "task" in said or any(x.endswith(".task") for x in said)


def test_matches_the_installed_harness():
    """When lm_eval is importable (the server's image, not a laptop), its own
    MMLU task index is the authority — pinned lists drift, package contents
    do not. Deploy step 3 is where this runs."""
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    index = tm.task_index
    mmlu = {name: meta for name, meta in index.items() if name.startswith("mmlu_")}
    # a shape this cannot read fails here, and says what it got, rather than
    # comparing an empty list
    sample = next(iter(mmlu.values()), None)
    assert sample is not None and _field(sample, "yaml_path") is not None, repr(sample)
    subjects = sorted(name[5:] for name, meta in mmlu.items()
                      if "tasks/mmlu/default/" in str(_field(meta, "yaml_path") or "")
                      and _is_plain_task(meta))
    assert subjects == HARNESS_SUBJECTS_0_4_12


@pytest.mark.parametrize("meta", [
    {"yaml_path": "/x/tasks/mmlu/default/mmlu_law.yaml", "type": "task"},
    type("Entry", (), {"yaml_path": "/x/tasks/mmlu/default/mmlu_law.yaml", "type": "task"})(),
    type("Entry", (), {"yaml_path": "/x/tasks/mmlu/default/mmlu_law.yaml",
                       "kind": type("Kind", (), {"name": "TASK", "value": 1})()})(),
])
def test_the_harness_index_is_read_in_either_shape(meta):
    """the reader above, without lm_eval: a dict, an object, an enum kind"""
    assert _field(meta, "yaml_path").endswith("mmlu_law.yaml")
    assert _is_plain_task(meta)
    group = {"yaml_path": "/x/tasks/mmlu/default/_mmlu.yaml", "type": "group"}
    assert not _is_plain_task(group)


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
        "Mathematics & Statistics", "Medicine & Clinical Health", "Economics",
        "Political Science & International Relations", "Ethics & Religion", "Law"}


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
                "Medicine & Clinical Health", "Economics",
                "Political Science & International Relations", "Ethics & Religion"}
    assert checked >= 6 * 6


def test_noise_floor_is_shared_by_page_and_file(diag):
    assert dx.MIN_GROUP_N == 30
    for d in diag.values():
        assert d["thresholds"]["min_group_n"] == 30
    # the fixture puts two-subject categories above the floor and one-subject ones below
    v = diag["fx/good-750m"]["tasks"]["mmlu"]
    above = {c for c, g in v["categories"].items() if g["n_report"] >= 30}
    below = {c for c, g in v["categories"].items() if g["n_report"] < 30}
    assert above == {"Economics", "Medicine & Clinical Health"} and len(below) == 4


def _rec(subject: str, correct: int, pick: int) -> dict:
    import hashlib
    import math
    probs = [0.6 if i == pick else 0.4 / 3 for i in range(4)]
    doc = {"question": f"q {subject} {correct} {pick}", "subject": subject,
           "choices": ["a", "bb", "ccc", "dddd"], "answer": correct}
    return {"doc": doc, "target": correct, "acc": float(pick == correct), "filter": "none",
            "metrics": ["acc"], "doc_hash": hashlib.sha256(json.dumps(doc).encode()).hexdigest(),
            "filtered_resps": [[f"{math.log(p):.4f}", str(i == pick)] for i, p in enumerate(probs)]}


def test_unmapped_groups_roll_into_the_fallback_and_are_listed(tmp_path: Path):
    recs = [_rec("anatomy", i % 4, (i * 3) % 4) for i in range(40)]
    recs += [_rec("underwater_basketweaving", i % 4, i % 4) for i in range(40)]
    recs += [_rec("clinical_knowledge", i % 4, 0) for i in range(40)]
    recs += [_rec("global_facts", i % 4, 0) for i in range(40)]
    f = tmp_path / "samples_t_2026-01-01T00-00-00.000000.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = dx.diagnose_task([f])
    med, gen = "Medicine & Clinical Health", "General & Multidisciplinary"
    assert out["unmapped"] == ["underwater_basketweaving"]
    assert set(out["categories"]) == {med, gen}
    assert out["categories"][med]["groups"] == ["anatomy", "clinical_knowledge"]
    # `other` is not a topic any more: the fallback is a real topic, and an
    # unknown subject lands beside the two MMLU subjects mapped there
    assert out["categories"][gen]["groups"] == ["global_facts", "underwater_basketweaving"]
    assert out["categories"][gen]["n"] == 80
    assert list(out["categories"]) == [med, gen]                         # the fallback is last


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
