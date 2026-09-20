"""Phase 8g A–B: three more topics, and a third way of writing a criteria file.

The decision behind these tests (Omar, 2026-09-20) is that the loader adapts
to whatever the author sends — his files go in verbatim and the platform
reads them. So the test is: every delivered file, in whatever shape, comes
out of the loader in ONE internal shape, and that shape is what the prompt,
the fold and the page are built from. docs/CRITERIA-SCHEMA.md is the table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd

REPO = Path(__file__).resolve().parents[1]
RUBRICS = REPO / "eval_tasks" / "fr" / "rubrics"
DELIVERED = {
    "computer science": "computer_science",
    "economics": "economics",
    "physics & engineering": "physics_engineering",
    "law": "law",
    "medicine & health": "medicine_health",
}
NEW = ["computer_science", "economics", "physics_engineering"]


def raw(slug: str) -> dict:
    return json.loads((RUBRICS / f"{slug}.criteria.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the files, as delivered
# ---------------------------------------------------------------------------

def test_the_topics_slugs_reach_the_delivered_file_names():
    """The file name is how a rubric finds its topic; if these disagree the
    topic is silently graded by the shared rubric."""
    for topic, slug in DELIVERED.items():
        assert eb.task_slug(eb.topic_task(topic)) == slug
        assert (RUBRICS / f"{slug}.md").is_file()
        assert (RUBRICS / f"{slug}.criteria.json").is_file()
        r = jd.rubric_for(eb.topic_task(topic))
        assert r.name == slug and r.fallback is False and r.criteria is not None


def test_the_three_new_rubrics_are_the_authors_own_and_not_drafts():
    """He wrote the 0–4 anchors this time, so nothing is stamped DRAFT."""
    for slug in NEW:
        r = jd.rubric_for(eb.topic_task(
            next(t for t, s in DELIVERED.items() if s == slug)))
        assert r.status == "", slug
        for anchor in range(5):
            assert str(anchor) in r.text, slug


def test_every_delivered_file_normalises_to_one_shape():
    """The table in docs/CRITERIA-SCHEMA.md, asserted."""
    for slug in DELIVERED.values():
        spec = jd.normalise_criteria(raw(slug), RUBRICS / f"{slug}.criteria.json")
        assert len(spec["criteria"]) >= 15, slug
        for c in spec["criteria"]:
            assert isinstance(c["id"], str) and c["id"] == c["id"].lower(), (slug, c)
            assert c["id"].replace("_", "").isalnum(), (slug, c["id"])
            assert c["name"] and isinstance(c["name"], str)
            assert c["definition"]
            assert jd.weight_of(spec, c["id"]) > 0
        assert isinstance(spec["flags"], list) and spec["flags"], slug
        for f in spec["flags"]:
            assert f["id"] and f["condition"]
            assert jd.normalise_effect(f["effect"]) == f["effect"], (slug, f["effect"])
            assert isinstance(f["examples"], list) and isinstance(f["not_critical"], list)
        assert isinstance(spec["evaluation_principles"], list)
        assert jd.validate_criteria(raw(slug)) == [], slug


def test_a_row_number_is_not_a_criterion_id():
    """Physics numbers its criteria 1..20 and puts the slug in `name`. The
    model has to repeat the id back, so the id must be the word."""
    spec = jd.normalise_criteria(raw("physics_engineering"))
    assert [c["id"] for c in raw("physics_engineering")["criteria"]][:3] == [1, 2, 3]
    ids = jd.criteria_ids(spec)
    assert ids[0] == "relevance" and all(not i.isdigit() for i in ids)
    assert jd.criteria_labels(spec)["relevance"] == "Relevance"
    # and computer science, which writes it the other way round, is unchanged
    cs = jd.normalise_criteria(raw("computer_science"))
    assert jd.criteria_ids(cs)[0] == "relevance"
    assert jd.criteria_labels(cs)["relevance"] == "Relevance"


def test_a_single_flag_object_under_any_of_its_names_becomes_the_list():
    assert "critical_flag" in raw("computer_science")
    assert "critical_error_flag" in raw("economics")
    assert "flags" in raw("law")
    for slug, fid in (("computer_science", "critical_technical_error"),
                      ("economics", "critical_economic_error"),
                      ("physics_engineering", "critical_physics_engineering_error")):
        spec = jd.normalise_criteria(raw(slug))
        assert jd.flag_ids(spec) == [fid]
        assert jd.effect_of(spec["flags"][0]) == "zero_score"     # written "score=0"
    assert jd.flag_ids(jd.normalise_criteria(raw("law"))) == [
        "critical_legal_error", "fabricated_authority"]


@pytest.mark.parametrize("written,read_as", [
    ("zero_score", "zero_score"), ("score=0", "zero_score"), ("score = 0", "zero_score"),
    ("cap_at_1_of_4", "cap_at_1_of_4"), ("cap=1", "cap_at_1_of_4"),
    ("score=2", "set_at_2_of_4"), ("set_at_3_of_4", "set_at_3_of_4"),
])
def test_the_effect_spellings_he_has_used(written, read_as):
    assert jd.normalise_effect(written) == read_as


def test_an_effect_nobody_can_apply_still_refuses_to_load(tmp_path, monkeypatch):
    spec = raw("computer_science")
    spec["critical_flag"]["effect"] = "halve_it"
    with pytest.raises(jd.CriteriaError) as e:
        jd.normalise_criteria(spec, tmp_path / "computer_science.criteria.json")
    assert "halve_it" in str(e.value) and "computer_science.criteria.json" in str(e.value)
    assert "score=N" in str(e.value)
    assert any("halve_it" in p for p in jd.validate_criteria(spec))


def test_set_at_n_makes_the_score_that_number():
    spec = {"criteria": [{"id": "a", "definition": "d"}, {"id": "b", "definition": "d"}],
            "flags": [{"id": "f", "condition": "c", "effect": "score=2"}]}
    spec = jd.normalise_criteria(spec)
    assert jd.fold({"a": 1.0, "b": 1.0}, {"f": False}, spec) == 4
    assert jd.fold({"a": 1.0, "b": 1.0}, {"f": True}, spec) == 2
    assert jd.fold({"a": 0.0, "b": 0.0}, {"f": True}, spec) == 2     # set, not cap
    assert jd.effect_words("set_at_2_of_4") == "makes the whole score 2 of 4"


# ---------------------------------------------------------------------------
# what the judge is actually sent
# ---------------------------------------------------------------------------

def test_the_authors_calibration_travels_in_the_request():
    """His examples of what counts as critical, what does not, and how the
    topic is graded are the difference between a flag that fires on anything
    and one that means something. They belong in the request."""
    for slug, topic in (("physics_engineering", "physics & engineering"),
                        ("economics", "economics")):
        r = jd.rubric_for(eb.topic_task(topic))
        text = jd.build_criteria_prompt(r.text, r.criteria, "a question?",
                                        "Domain: x. Style: y.", "an answer")
        f = r.criteria["flags"][0]
        assert f"{f['id']} — {f['condition'][:40]}" in text
        for ex in f["examples"][:2]:
            assert f"counts as {f['id']}: {ex}" in text
        for ex in f["not_critical"][:2]:
            assert f"does NOT count as {f['id']}: {ex}" in text
        for p in r.criteria["evaluation_principles"][:2]:
            assert p in text
        if r.criteria["evaluation_principles"]:
            assert "HOW THIS TOPIC IS GRADED" in text
            assert text.index("HOW THIS TOPIC IS GRADED") < text.index("CRITERIA (0.0")
        # a conditional criterion is told what to do, one line each
        cond = jd.conditional_ids(r.criteria)
        assert len(cond) >= 10
        for cid in sorted(cond)[:3]:
            line = next(x for x in text.splitlines() if x.startswith(cid + " — "))
            assert "Return null for it when it does not apply" in line


def test_the_judge_grades_the_new_topics_end_to_end(tmp_path):
    """The stub grader answers the prompt these files build, and the fold
    turns it into a 0–4 — the path that was silently broken while physics
    numbered its criteria."""
    from service import llm
    for topic in ("computer science", "economics", "physics & engineering"):
        task = eb.topic_task(topic)
        spec = jd.rubric_for(task).criteria
        prompt = jd.build_criteria_prompt(jd.rubric_for(task).text, spec, "q?",
                                          "Domain: x.", "an answer about x")
        reply = jd.StubGrader.reply(prompt)
        g = jd.parse_grade_criteria(reply, spec)
        assert g is not None, topic
        assert set(g.criteria) == set(jd.criteria_ids(spec))
        assert set(g.flags) == set(jd.flag_ids(spec))
        assert 0 <= jd.fold(g.criteria, g.flags, spec) <= 4
        assert llm is not None


# ---------------------------------------------------------------------------
# the banks, and the tables they earn
# ---------------------------------------------------------------------------

BANKS = {"computer science": "computer_science_v1", "economics": "economics_v1",
         "physics & engineering": "physics_engineering_v1",
         "law": "law_v2", "medicine & health": "medicine_v2"}


@pytest.fixture(scope="module")
def five(tmp_path_factory) -> Path:
    """Every delivered bank in one exam root."""
    root = tmp_path_factory.mktemp("five") / "exam"
    for topic, stem in BANKS.items():
        eb.import_bank(root, REPO / "eval_tasks" / "fr" / f"{stem}.json", topic,
                       "Dr. Hossein", stem)
    return root


def test_a_wrapped_questions_file_imports_like_a_bare_one(tmp_path):
    """Physics & engineering arrived as {"questions": [...]}. A file is not
    refused over its wrapping, and the preview says what it found."""
    src = REPO / "eval_tasks" / "fr" / "physics_engineering_v1.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and list(data) == ["questions"]
    r = eb.import_bank(tmp_path / "exam", src, "physics & engineering", "Dr. Hossein", "p")
    assert r["imported"] == 100 and r["wrapper"] == "questions"
    plan = eb.plan_import(tmp_path / "bare", data["questions"], "physics & engineering", "x")
    assert plan["imported"] == 100 and plan["wrapper"] == ""
    # and one that wraps nothing usable is still refused, in words
    with pytest.raises(ValueError, match="not a JSON array"):
        eb.import_bank(tmp_path / "e", {"a": [], "b": []}, "economics", "x")


def test_the_three_new_banks_land_whole(five):
    for topic, stem in BANKS.items():
        rows = eb.load_bank(five)[topic]
        assert len(rows) == 100, topic
        assert {r["source"] for r in rows} == {stem}
        assert all(r["accepted_by"] == "Dr. Hossein" for r in rows)
        halves = {eb.half_of(r["qid"]) for r in rows}
        assert halves == {"report", "diagnose"}
        # every topic clears the floor a proposal needs
        import report_lm_eval as report
        n_report = sum(1 for r in rows if eb.half_of(r["qid"]) == "report")
        assert n_report >= report.PROPOSE_MIN_N, (topic, n_report)


def test_the_tables_a_topic_earns_are_the_ones_that_split_it(five):
    """By what the field does, not by its name: a field with one value on
    every item is a sentence, a field with a hundred is nothing at all."""
    want = {
        # the new three: acuity is routine throughout and intent is a
        # sentence per question, so difficulty and domain carry them
        "computer science": ["difficulty", "domain"],
        "economics": ["difficulty", "domain"],
        "physics & engineering": ["difficulty", "domain"],
        # and the two that were already tabulated keep exactly their tables
        "law": ["acuity", "difficulty", "jurisdiction_required", "intent"],
        "medicine & health": ["acuity", "difficulty", "intent"],
    }
    for topic, expect in want.items():
        items = [{"meta": r["meta"]} for r in eb.load_bank(five)[topic]]
        spec = jd.rubric_for(eb.topic_task(topic)).criteria
        fields, constant = jd.breakdown_fields(items, spec)
        assert fields == expect, topic
        assert len(fields) <= jd.BREAKDOWN_MAX_TABLES
        if topic in ("computer science", "economics", "physics & engineering"):
            assert constant == ["acuity", "jurisdiction_required"], topic
        else:
            assert constant == [], topic


def test_a_constant_field_is_said_once_instead_of_drawn(five):
    spec = jd.rubric_for("exam_computer_science").criteria
    items = [{"meta": r["meta"], "score": 2, "graded": True,
              "criteria": {c: 0.5 for c in jd.criteria_ids(spec)},
              "flags": {f: False for f in jd.flag_ids(spec)}, "half": "diagnose",
              "qid": f"{i:064d}"}
             for i, r in enumerate(eb.load_bank(five)["computer science"][:20])]
    blocks = jd._criteria_blocks(items, spec)
    assert set(blocks["breakdowns"]) == {"difficulty", "domain"}
    assert blocks["breakdowns_constant"] == {"acuity": "routine",
                                             "jurisdiction_required": "False"}


def test_the_reference_line_carries_what_the_judge_must_read(five):
    """These topics state the intent as a sentence; it is the author's
    statement of what the question tests, so the judge sees it."""
    one = eb.load_bank(five)["computer science"][0]
    assert one["reference"].startswith("Acuity: ")
    for label in ("Intent: ", "Domain: ", "Difficulty: ", "Style: "):
        assert label in one["reference"], label
    assert len(one["meta"]["intent"].split()) > 3        # a sentence, not a token
    assert one["meta"]["intent"] in one["reference"]


def test_no_report_half_question_of_any_bank_leaves_it(five, tmp_path, monkeypatch):
    """Five banks now, one rule, over the bodies actually sent."""
    from service import config, llm, proposals as prop
    monkeypatch.setattr(config, "EXAM_DIR", five)
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    fake = llm.FakeBatches("fake-exam", tmp_path)
    for topic in BANKS:
        task = eb.topic_task(topic)
        fake.submit(eb.draft_requests(five, topic, 2))
        fake.submit([prop.proposal_request(1, "m", task, topic,
                                           [{"qid": "q", "score": 1, "justification": "vague"}],
                                           {"diagnose_items": 1, "diagnose_weak": 1}, "rubric",
                                           audience=prop.audience_for(topic, task))])
    sent = "\n".join(r["system"] + "\n" + r["user"] for r in fake.recorded())
    for topic in BANKS:
        rows = eb.load_bank(five)[topic]
        report = [r for r in rows if eb.half_of(r["qid"]) == "report"]
        assert report, topic
        for r in report:
            assert r["prompt"] not in sent and r["prompt"][:60] not in sent
            assert r["qid"] not in sent
