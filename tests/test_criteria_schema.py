"""Phase 8g A–B: three more topics, and a third way of writing a criteria file.

The decision behind these tests (Omar, 2026-09-20) is that the loader adapts
to whatever the author sends — his files go in verbatim and the platform
reads them. So the test is: every delivered file, in whatever shape, comes
out of the loader in ONE internal shape, and that shape is what the prompt,
the fold and the page are built from. docs/CRITERIA-SCHEMA.md is the table.

Phase 10 delivered 36 topics at once and retired the five before them; Arts,
the 37th, followed a day later. The 37 are what the judge reads now; the
retired five are kept whole in
eval_tasks/fr/retired/, and they are still the only files with some of the
layouts the loader learned — a criterion numbered by its row, a flag list
with a cap in it, a bank wrapped in {"questions": [...]} — so the tests of
those paths read them from there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import assert_no_report_half_text

REPO = Path(__file__).resolve().parents[1]
RUBRICS = REPO / "eval_tasks" / "fr" / "rubrics"
RETIRED = REPO / "eval_tasks" / "fr" / "retired"
# every topic arrived with a rubric and a criteria file — Arts last, 2026-09-22
DELIVERED = list(eb.TOPICS)
# the 37 the judge reads, then the 5 it no longer does
CRITERIA = sorted(RUBRICS.glob("*.criteria.json")) + sorted(
    (RETIRED / "rubrics").glob("*.criteria.json"))


def retired(slug: str) -> dict:
    return json.loads((RETIRED / "rubrics" / f"{slug}.criteria.json").read_text(
        encoding="utf-8"))


# ---------------------------------------------------------------------------
# the files, as delivered
# ---------------------------------------------------------------------------

def test_the_topics_slugs_reach_the_delivered_file_names(monkeypatch, tmp_path):
    """The file name is how a rubric finds its topic; if these disagree the
    topic is silently graded by the shared rubric."""
    # the names with capitals, "&" and an acronym in them, spelled out by hand
    for topic, slug in (("Medicine & Clinical Health", "medicine_clinical_health"),
                        ("AI & Machine Learning", "ai_machine_learning"),
                        ("IT", "it"), ("Law", "law"),
                        ("Political Science & International Relations",
                         "political_science_international_relations")):
        assert eb.task_slug(eb.topic_task(topic)) == slug
    # every delivered file belongs to a topic, and every topic has one
    stems = {p.name[:-len(".criteria.json")] for p in RUBRICS.glob("*.criteria.json")}
    assert stems == {eb.task_slug(eb.topic_task(t)) for t in DELIVERED}
    for topic in DELIVERED:
        slug = eb.task_slug(eb.topic_task(topic))
        assert (RUBRICS / f"{slug}.md").is_file()
        r = jd.rubric_for(eb.topic_task(topic))
        assert r.name == slug and r.fallback is False and r.criteria is not None, topic
    # and a topic whose files are missing is graded by the shared rubric, not
    # by nothing and not by a stale name
    from conftest import without_its_own_rubric
    without_its_own_rubric(monkeypatch, tmp_path, "arts")
    assert jd.rubric_for(eb.topic_task("Arts")).fallback is True


def test_the_37_new_rubrics_are_the_authors_own_and_not_drafts():
    """He wrote the 0–4 anchors, so nothing is stamped DRAFT."""
    for topic in DELIVERED:
        r = jd.rubric_for(eb.topic_task(topic))
        assert r.status == "", topic
        for anchor in range(5):
            assert str(anchor) in r.text, topic


def test_every_delivered_file_normalises_to_one_shape():
    """The table in docs/CRITERIA-SCHEMA.md, asserted row by row over all 42
    files: whatever layout a file arrived in, what the loader reads is what
    the file said. (tests/test_37_topics.py checks the same files come out
    carrying every key of the one shape; this checks the values are his.)"""
    assert len(CRITERIA) == 42
    layouts: dict[str, int] = {}
    for p in CRITERIA:
        where = p.relative_to(REPO)
        raw = json.loads(p.read_text(encoding="utf-8"))
        spec = jd.normalise_criteria(raw, p)
        assert jd.validate_criteria(raw) == [], where
        assert len(spec["criteria"]) >= 15, where
        # a criterion: the slug the model repeats back, and words for a person
        for c, given in zip(spec["criteria"], raw["criteria"], strict=True):
            if isinstance(given["id"], int) or str(given["id"]).isdigit():
                # a row number, and the slug in `name`
                assert (c["id"], c["name"]) == (given["name"],
                                                jd.label_of({"id": given["name"]})), where
            elif given.get("name"):
                assert (c["id"], c["name"]) == (given["id"], given["name"]), where
            else:
                assert (c["id"], c["name"]) == (given["id"], jd.label_of(given)), where
            assert c["definition"] == given["definition"], where
            assert jd.weight_of(spec, c["id"]) == float(given.get("weight", 1)) > 0, where
        # the flags: one layout per file, read as one list in the file's order
        keys = [k for k in jd._FLAG_KEYS if k in raw]
        assert len(keys) == 1, where
        layouts[keys[0]] = layouts.get(keys[0], 0) + 1
        flags = raw[keys[0]] if isinstance(raw[keys[0]], list) else [raw[keys[0]]]
        assert [f["id"] for f in spec["flags"]] == [f["id"] for f in flags], where
        for f, given in zip(spec["flags"], flags):
            assert f["condition"] == given["condition"], where
            assert f["effect"] == jd.normalise_effect(given["effect"]) is not None, where
            assert f["examples"] == given.get("examples", []), where
            assert f["not_critical"] == next((given[k] for k in jd._NOT_CRITICAL_KEYS
                                              if k in given), []), where
        # his principles, under whichever name, one sentence each
        said = [k for k in jd._PRINCIPLE_KEYS if k in raw]
        n_said = len(raw[said[0]]) if said else 0
        assert len(spec["evaluation_principles"]) == n_said, where
        # and everything the table does not name rides along untouched
        for k in set(raw) - {"criteria", *jd._FLAG_KEYS, *jd._PRINCIPLE_KEYS}:
            assert spec[k] == raw[k], (where, k)
    # the "seen in" column: medicine and law (retired) wrote a list; computer
    # science and physics (retired) `critical_flag`; economics (retired) and
    # 31 of the 37 `critical_error_flag`; the other six the two new layouts
    assert layouts == {"flags": 2, "critical_flag": 2, "critical_error_flag": 32,
                       "critical_error": 4, "critical_flags": 2}


def test_a_row_number_is_not_a_criterion_id():
    """Physics & engineering numbered its criteria 1..20 and put the slug in
    `name`. The model has to repeat the id back, so the id must be the word.
    The file is retired, and still the only one written that way."""
    spec = jd.normalise_criteria(retired("physics_engineering"))
    assert [c["id"] for c in retired("physics_engineering")["criteria"]][:3] == [1, 2, 3]
    ids = jd.criteria_ids(spec)
    assert ids[0] == "relevance" and all(not i.isdigit() for i in ids)
    assert jd.criteria_labels(spec)["relevance"] == "Relevance"
    # and computer science, which wrote it the other way round, is unchanged
    cs = jd.normalise_criteria(retired("computer_science"))
    assert jd.criteria_ids(cs)[0] == "relevance"
    assert jd.criteria_labels(cs)["relevance"] == "Relevance"


def test_a_single_flag_object_under_any_of_its_names_becomes_the_list():
    # the layouts of the retired files; the two the 36 added are in
    # tests/test_37_topics.py
    assert "critical_flag" in retired("computer_science")
    assert "critical_error_flag" in retired("economics")
    assert "flags" in retired("law")
    for slug, fid in (("computer_science", "critical_technical_error"),
                      ("economics", "critical_economic_error"),
                      ("physics_engineering", "critical_physics_engineering_error")):
        spec = jd.normalise_criteria(retired(slug))
        assert jd.flag_ids(spec) == [fid]
        assert jd.effect_of(spec["flags"][0]) == "zero_score"     # written "score=0"
    assert jd.flag_ids(jd.normalise_criteria(retired("law"))) == [
        "critical_legal_error", "fabricated_authority"]


@pytest.mark.parametrize("written,read_as", [
    ("zero_score", "zero_score"), ("score=0", "zero_score"), ("score = 0", "zero_score"),
    ("cap_at_1_of_4", "cap_at_1_of_4"), ("cap=1", "cap_at_1_of_4"),
    ("score=2", "set_at_2_of_4"), ("set_at_3_of_4", "set_at_3_of_4"),
])
def test_the_effect_spellings_he_has_used(written, read_as):
    assert jd.normalise_effect(written) == read_as


def test_an_effect_nobody_can_apply_still_refuses_to_load(tmp_path, monkeypatch):
    spec = retired("computer_science")
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
    and one that means something. They belong in the request — for every
    topic he delivered."""
    with_principles = []
    for topic in DELIVERED:
        r = jd.rubric_for(eb.topic_task(topic))
        text = jd.build_criteria_prompt(r.text, r.criteria, "a question?",
                                        "Domain: x. Style: y.", "an answer")
        for f in r.criteria["flags"]:
            assert f"{f['id']} — {f['condition'][:40]}" in text, topic
            assert f["examples"], topic
            for ex in f["examples"][:2]:
                assert f"counts as {f['id']}: {ex}" in text, topic
            for ex in f["not_critical"][:2]:
                assert f"does NOT count as {f['id']}: {ex}" in text, topic
        for p in r.criteria["evaluation_principles"][:2]:
            assert p in text, topic
        if r.criteria["evaluation_principles"]:
            with_principles.append(topic)
            assert "HOW THIS TOPIC IS GRADED" in text
            assert text.index("HOW THIS TOPIC IS GRADED") < text.index("CRITERIA (0.0")
        # a conditional criterion is told what to do, one line each
        cond = jd.conditional_ids(r.criteria)
        assert len(cond) >= 10, topic
        for cid in sorted(cond)[:3]:
            line = next(x for x in text.splitlines() if x.startswith(cid + " — "))
            assert "Return null for it when it does not apply" in line
    # principles arrived in six of the 37, under three names
    assert len(with_principles) == 6


def test_the_judge_grades_the_new_topics_end_to_end(tmp_path):
    """The stub grader answers the prompt these files build, and the fold
    turns it into a 0–4 — the path that was silently broken while physics
    numbered its criteria."""
    from service import llm
    for topic in DELIVERED:
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

# the five banks of phases 8b–8g, each under the topic that replaced its own:
# their metadata is what the table rules below were written against
BANKS = {"Computer Science": "computer_science_v1", "Economics": "economics_v1",
         "Physics & Astronomy": "physics_engineering_v1",
         "Law": "law_v2", "Medicine & Clinical Health": "medicine_v2"}


@pytest.fixture(scope="module")
def five(tmp_path_factory) -> Path:
    """Every retired bank in one exam root."""
    root = tmp_path_factory.mktemp("five") / "exam"
    for topic, stem in BANKS.items():
        eb.import_bank(root, RETIRED / f"{stem}.json", topic, "Dr. Hossein", stem)
    return root


def test_a_wrapped_questions_file_imports_like_a_bare_one(tmp_path):
    """Physics & engineering arrived as {"questions": [...]}. A file is not
    refused over its wrapping, and the preview says what it found."""
    src = RETIRED / "physics_engineering_v1.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and list(data) == ["questions"]
    r = eb.import_bank(tmp_path / "exam", src, "Physics & Astronomy", "Dr. Hossein", "p")
    assert r["imported"] == 100 and r["wrapper"] == "questions"
    plan = eb.plan_import(tmp_path / "bare", data["questions"], "Physics & Astronomy", "x")
    assert plan["imported"] == 100 and plan["wrapper"] == ""
    # and one that wraps nothing usable is still refused, in words
    with pytest.raises(ValueError, match="not a JSON array"):
        eb.import_bank(tmp_path / "e", {"a": [], "b": []}, "Economics", "x")


def test_the_retired_banks_land_whole_under_their_new_topics(five):
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
        # the three of phase 8g: acuity is routine throughout and intent is a
        # sentence per question, so difficulty and domain carry them
        "Computer Science": ["difficulty", "domain"],
        "Economics": ["difficulty", "domain"],
        "Physics & Astronomy": ["difficulty", "domain"],
        # and the two that were already tabulated keep exactly their tables
        "Law": ["acuity", "difficulty", "jurisdiction_required", "intent"],
        "Medicine & Clinical Health": ["acuity", "difficulty", "intent"],
    }
    for topic, expect in want.items():
        items = [{"meta": r["meta"]} for r in eb.load_bank(five)[topic]]
        spec = jd.rubric_for(eb.topic_task(topic)).criteria
        fields, constant = jd.breakdown_fields(items, spec)
        assert fields == expect, topic
        assert len(fields) <= jd.BREAKDOWN_MAX_TABLES
        if topic in ("Computer Science", "Economics", "Physics & Astronomy"):
            assert constant == ["acuity", "jurisdiction_required"], topic
        else:
            assert constant == [], topic


def test_a_constant_field_is_said_once_instead_of_drawn(five):
    spec = jd.rubric_for("exam_computer_science").criteria
    items = [{"meta": r["meta"], "score": 2, "graded": True,
              "criteria": {c: 0.5 for c in jd.criteria_ids(spec)},
              "flags": {f: False for f in jd.flag_ids(spec)}, "half": "diagnose",
              "qid": f"{i:064d}"}
             for i, r in enumerate(eb.load_bank(five)["Computer Science"][:20])]
    blocks = jd._criteria_blocks(items, spec)
    assert set(blocks["breakdowns"]) == {"difficulty", "domain"}
    assert blocks["breakdowns_constant"] == {"acuity": "routine",
                                             "jurisdiction_required": "False"}


def test_the_reference_line_carries_what_the_judge_must_read(five):
    """These topics state the intent as a sentence; it is the author's
    statement of what the question tests, so the judge sees it."""
    one = eb.load_bank(five)["Computer Science"][0]
    assert one["reference"].startswith("Acuity: ")
    for label in ("Intent: ", "Domain: ", "Difficulty: ", "Style: "):
        assert label in one["reference"], label
    assert len(one["meta"]["intent"].split()) > 3        # a sentence, not a token
    assert one["meta"]["intent"] in one["reference"]


def test_no_report_half_question_of_any_bank_leaves_it(five, tmp_path, monkeypatch):
    """Every bank delivered — the five retired ones and the 37 of the exam
    now — one rule, over the bodies actually sent."""
    from service import config, llm, proposals as prop
    current = tmp_path / "exam"
    eb.import_dir(current, REPO / "eval_tasks" / "fr" / "banks", "Dr. Hossein")
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    for root in (five, current):
        # the audience line is counted from whichever bank is the service's
        monkeypatch.setattr(config, "EXAM_DIR", root)
        topics = [t for t, rows in eb.load_bank(root).items() if rows]
        assert len(topics) == (5 if root == five else 37)
        fake = llm.FakeBatches("fake-exam", tmp_path / f"fake-{len(topics)}")
        for topic in topics:
            task = eb.topic_task(topic)
            fake.submit(eb.draft_requests(root, topic, 2))
            fake.submit([prop.proposal_request(1, "m", task, topic,
                                               [{"qid": "q", "score": 1,
                                                 "justification": "vague"}],
                                               {"diagnose_items": 1, "diagnose_weak": 1},
                                               "rubric",
                                               audience=prop.audience_for(topic, task))])
        sent = "\n".join(r["system"] + "\n" + r["user"] for r in fake.recorded())
        for topic in topics:
            rows = eb.load_bank(root)[topic]
            report = [r for r in rows if eb.half_of(r["qid"]) == "report"]
            assert report, topic
            # every free text of a report-half item, under any name: the
            # 37-topic banks' `intent` is a sentence per question, and the
            # audience line carried four of them until 10b
            assert assert_no_report_half_text(sent, rows) >= len(report), topic


# ---------------------------------------------------------------------------
# the wrapped file, through both doors
# ---------------------------------------------------------------------------

# retired, and still the one wrapped bank there has been; it goes in under the
# physics half of the topic it was
PHYSICS = RETIRED / "physics_engineering_v1.json"
PHYS_TOPIC = "Physics & Astronomy"


def test_the_real_wrapped_file_imports_from_the_page(tmp_path, monkeypatch):
    """It did not: the page had an array-only check in front of the unwrapping,
    so the one wrapped file of the five was refused with "the file must hold a
    JSON array of question objects" while the other four went in."""
    from conftest import make_service
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    try:
        from service import config
        text = PHYSICS.read_text(encoding="utf-8")
        assert isinstance(json.loads(text), dict)          # the shape that was refused
        body = {"topic": PHYS_TOPIC, "approver": "Dr. Hossein",
                "source": "physics_engineering_v1", "text": text}
        pre = client.post("/api/exam/import/preview", json=body)
        assert pre.status_code == 200, pre.text
        assert pre.json()["imported"] == 100
        # the preview says where it found them
        assert pre.json()["wrapper"] == "questions"
        assert len(pre.json()["items"]) == 100
        got = client.post("/api/exam/import", json=body)
        assert got.status_code == 200, got.text
        assert got.json()["imported"] == 100 and got.json()["wrapper"] == "questions"
        mine = [r for r in eb.load_bank(config.EXAM_DIR)[PHYS_TOPIC]
                if r.get("source") == "physics_engineering_v1"]
        assert len(mine) == 100
        # the committed file stays as the author sent it, so re-importing the
        # same file is a no-op: same prompts, same qids, nothing written
        again = client.post("/api/exam/import", json=body)
        assert (again.json()["imported"], again.json()["updated"],
                again.json()["skipped"]) == (0, 0, 100)
        assert len([r for r in eb.load_bank(config.EXAM_DIR)[PHYS_TOPIC]
                    if r.get("source") == "physics_engineering_v1"]) == 100
        # and the array the page sends when it has already parsed the file
        parsed = client.post("/api/exam/import/preview",
                             json={**body, "text": "", "items": json.loads(text)})
        assert parsed.status_code == 200 and parsed.json()["skipped"] == 100
    finally:
        client.__exit__(None, None, None)


def test_the_cli_and_the_page_read_the_wrapped_file_identically(tmp_path, monkeypatch):
    """Two doors, one reading. The records must match field for field."""
    from conftest import make_service
    client, _, _ = make_service(tmp_path, monkeypatch)
    try:
        from service import config
        text = PHYSICS.read_text(encoding="utf-8")
        client.post("/api/exam/import", json={"topic": PHYS_TOPIC, "approver": "Dr. Hossein",
                                              "source": "physics_engineering_v1", "text": text})
        from_page = [r for r in eb.load_bank(config.EXAM_DIR)[PHYS_TOPIC]
                     if r.get("source") == "physics_engineering_v1"]
        cli_root = tmp_path / "cli-exam"
        eb.import_bank(cli_root, PHYSICS, PHYS_TOPIC, "Dr. Hossein", "physics_engineering_v1")
        from_cli = eb.load_bank(cli_root)[PHYS_TOPIC]
        assert len(from_page) == len(from_cli) == 100
        drop = lambda rows: [{k: v for k, v in r.items() if k != "accepted_at"}     # noqa: E731
                             for r in sorted(rows, key=lambda x: x["qid"])]
        assert drop(from_page) == drop(from_cli)
    finally:
        client.__exit__(None, None, None)


def test_a_file_that_wraps_nothing_usable_is_still_refused(tmp_path, monkeypatch):
    from conftest import make_service
    client, _, _ = make_service(tmp_path, monkeypatch)
    try:
        body = {"topic": "Economics", "approver": "x"}
        for text in ('{"a": [], "b": []}', '{"questions": {"not": "a list"}}', '"a string"'):
            r = client.post("/api/exam/import/preview", json={**body, "text": text})
            assert r.status_code == 422, text
            assert "array of question objects" in r.json()["detail"]
            assert "one list in it" in r.json()["detail"]
    finally:
        client.__exit__(None, None, None)
