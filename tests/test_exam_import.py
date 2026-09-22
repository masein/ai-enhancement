"""Phases 8b P4a and 8d: a human-written bank, and a rubric per topic.

Dr. Hossein's 100 consumer health questions come with metadata instead of
reference answers, and with their own rubric. These tests are over the real
delivered files — if the file changes, they are what says so.

Phase 10 retired that bank with its topic; the file is kept byte for byte in
eval_tasks/fr/retired/, and it is still the one bank whose numbers (nine
emergencies, a first delivery of fifty inside the second) these tests know.
So the import machinery is tested on it, into the topic that replaced
"medicine & health"."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from service import llm
from conftest import assert_no_report_half_text

REPO = Path(__file__).resolve().parents[1]
RETIRED = REPO / "eval_tasks" / "fr" / "retired"
MEDICINE = RETIRED / "medicine_v2.json"
LAW_V2 = RETIRED / "law_v2.json"
TOPIC = "Medicine & Clinical Health"
TASK = "exam_medicine_clinical_health"
LAW = "Law"
NO_RUBRIC = "exam_arts"          # the one topic delivered without a rubric of its own


@pytest.fixture
def bank(tmp_path) -> tuple[Path, dict]:
    root = tmp_path / "exam"
    r = eb.import_bank(root, MEDICINE, TOPIC, "Dr. Hossein", "medicine_v1")
    return root, r


# ---------------------------------------------------------------------------
# the import
# ---------------------------------------------------------------------------

def test_the_delivered_file_is_what_the_import_expects():
    items = json.loads(MEDICINE.read_text(encoding="utf-8"))
    assert isinstance(items, list) and len(items) == 100
    assert all(isinstance(it, dict) and it.get("prompt") for it in items)
    assert not any(it.get("reference") for it in items)      # metadata instead
    assert {it["acuity"] for it in items} == {"emergency", "urgent", "moderate", "mild", "routine"}
    # v2 gave every item the author's own difficulty level
    assert all(isinstance(it.get("difficulty"), int) for it in items)
    law = json.loads(LAW_V2.read_text("utf-8"))
    assert len(law) == 100 and all(it.get("prompt") and it.get("difficulty") for it in law)
    assert sorted({it["difficulty"] for it in law}) == [1, 2, 3, 4, 5]


def test_import_round_trips_the_whole_file_and_is_idempotent(bank):
    root, r = bank
    assert (r["imported"], r["skipped"], r["invalid"]) == (100, 0, 0)
    assert r["report"] + r["diagnose"] == 100 and r["report"] > 20 and r["diagnose"] > 20
    assert sum(r["acuity"].values()) == 100 and r["acuity"]["emergency"] == 9
    rows = eb.load_bank(root)[TOPIC]
    assert len(rows) == 100
    one = next(x for x in rows if x["meta"]["id"] == 1)
    assert one["source"] == "medicine_v1" and one["accepted_by"] == "Dr. Hossein"
    assert one["edited"] is False and one["notes"] == "" and one["accepted_at"] > 0
    assert one["qid"] == eb.qid_of(one["prompt"]) and one["topic"] == TOPIC
    # his metadata is kept whole, under meta
    assert one["meta"] == {"id": 1, "intent": "symptom_assessment_triage", "subject": "child",
                           "age_group": "5-12", "sex": "male", "acuity": "moderate",
                           "domain": "respiratory_infectious", "style": "conversational",
                           "difficulty": 2}
    again = eb.import_bank(root, MEDICINE, TOPIC, "Dr. Hossein", "medicine_v1")
    assert (again["imported"], again["skipped"]) == (0, 100)
    assert len(eb.load_bank(root)[TOPIC]) == 100


def test_a_second_delivery_adds_only_what_is_new(tmp_path):
    """v2 is v1's fifty questions unchanged plus fifty more. On a bank that
    already holds the first delivery, the second imports half and skips half
    — which is the idempotence the qid is for, and the shape every later
    delivery from the author will have."""
    root = tmp_path / "exam"
    items = json.loads(MEDICINE.read_text(encoding="utf-8"))
    first, second = tmp_path / "v1.json", MEDICINE
    first.write_text(json.dumps(items[:50]), encoding="utf-8")
    r1 = eb.import_bank(root, first, TOPIC, "Dr. Hossein", "medicine_v1")
    assert (r1["imported"], r1["skipped"]) == (50, 0)
    r2 = eb.import_bank(root, second, TOPIC, "Dr. Hossein", "medicine_v2")
    assert (r2["imported"], r2["skipped"]) == (50, 50)
    rows = eb.load_bank(root)[TOPIC]
    assert len(rows) == 100
    # the first fifty keep the source they arrived under, and their qids and
    # halves did not move: the published score stays comparable across the
    # delivery, which is the whole point of hashing the prompt
    kept = [x for x in rows if x["source"] == "medicine_v1"]
    assert len(kept) == 50
    assert {x["qid"] for x in kept} == {eb.qid_of(it["prompt"]) for it in items[:50]}


def test_the_reference_is_the_metadata_in_a_stable_order(bank):
    root, _ = bank
    one = next(x for x in eb.load_bank(root)[TOPIC] if x["meta"]["id"] == 1)
    assert one["reference"] == ("Acuity: moderate. Intent: symptom_assessment_triage. "
                                "Domain: respiratory_infectious. Difficulty: 2. "
                                "Subject: child (male, 5-12). Style: conversational.")
    # field order is fixed, not dict order, so the same item always hashes the same
    scrambled = {"style": "telegraphic", "domain": "cardiovascular", "sex": "male",
                 "acuity": "emergency", "age_group": "45-59", "subject": "self",
                 "difficulty": 4, "intent": "symptom_assessment_triage"}
    assert eb.metadata_reference(scrambled) == (
        "Acuity: emergency. Intent: symptom_assessment_triage. Domain: cardiovascular. "
        "Difficulty: 4. Subject: self (male, 45-59). Style: telegraphic.")
    # only what is there appears
    assert eb.metadata_reference({"acuity": "mild"}) == "Acuity: mild."
    assert eb.metadata_reference({"subject": "self"}) == "Subject: self."
    assert eb.metadata_reference({"sex": "female", "age_group": "25-34"}) == \
        "Subject: female, 25-34."
    assert eb.metadata_reference({}) == ""


def test_an_item_with_its_own_reference_keeps_it(tmp_path):
    src = tmp_path / "own.json"
    src.write_text(json.dumps([
        {"prompt": "A question long enough to be a question about chest pain.",
         "reference": "Send to emergency care now.", "acuity": "emergency", "notes": "n"}]))
    r = eb.import_bank(tmp_path / "exam", src, TOPIC, "Dr. Hossein")
    assert r["imported"] == 1
    row = eb.load_bank(tmp_path / "exam")[TOPIC][0]
    assert row["reference"] == "Send to emergency care now. Acuity: emergency."
    assert row["notes"] == "n" and row["source"] == "import"


def test_the_import_refuses_what_it_should(tmp_path):
    with pytest.raises(ValueError, match="needs a name"):
        eb.import_bank(tmp_path / "exam", MEDICINE, TOPIC, "  ")
    with pytest.raises(ValueError, match="not an exam topic"):
        eb.import_bank(tmp_path / "exam", MEDICINE, "astrology", "Dr. Hossein")
    # nor, now, the name this file was first delivered under
    with pytest.raises(ValueError, match="not an exam topic"):
        eb.import_bank(tmp_path / "exam", MEDICINE, "medicine & health", "Dr. Hossein")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"prompt": "not an array"}))
    with pytest.raises(ValueError, match="not a JSON array"):
        eb.import_bank(tmp_path / "exam", bad, TOPIC, "Dr. Hossein")
    short = tmp_path / "short.json"
    short.write_text(json.dumps([{"prompt": "too short"}, {"nope": 1}, "a string"]))
    r = eb.import_bank(tmp_path / "exam", short, TOPIC, "Dr. Hossein")
    assert (r["imported"], r["invalid"]) == (0, 3)


def test_the_cli_imports_and_says_what_it_did(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "exam_build.py"),
                        "--root", str(tmp_path / "exam"), "import", str(MEDICINE),
                        "--topic", TOPIC, "--approver", "Dr. Hossein", "--source", "medicine_v1"],
                       capture_output=True, text=True, timeout=120, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "imported 100, skipped 0" in r.stdout and "report" in r.stdout
    assert "acuity: emergency 9" in r.stdout
    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "exam_build.py"),
                          "--root", str(tmp_path / "exam2"), "import", str(MEDICINE),
                          "--topic", TOPIC, "--approver", " "],
                         capture_output=True, text=True, timeout=120, cwd=REPO)
    assert bad.returncode == 2 and "needs a name" in bad.stderr


def test_the_built_task_carries_the_metadata_the_judge_reads(bank, tmp_path):
    root, _ = bank
    (tmp_path / "results").mkdir(exist_ok=True)     # no MMLU there: no control set
    m = eb.build(tmp_path / "results", root)
    items = [json.loads(x) for x in
             (eb.tasks_dir(root) / f"{TASK}.jsonl").read_text().splitlines() if x.strip()]
    assert m["tasks"][TASK]["items"] == 100 and len(items) == 100
    assert all(it["meta"]["acuity"] and it["meta"]["difficulty"] for it in items)
    assert items[0]["prompt"] and items[0]["reference"].startswith("Acuity: ")
    assert "Difficulty: " in items[0]["reference"]


def test_a_report_half_import_is_as_withheld_as_any_other_question(bank):
    root, _ = bank
    public = eb.public_bank(root, TOPIC)
    assert len(public) == 100
    for row in public:
        if row["half"] == "report":
            assert row["prompt"] is None and row["reference"] is None and row["meta"] is None
            assert "never shown" in row["withheld"]
        else:
            assert row["prompt"] and row["meta"]["acuity"]


# ---------------------------------------------------------------------------
# a rubric per topic
# ---------------------------------------------------------------------------

def test_the_rubric_follows_the_topic_and_falls_back(tmp_path):
    slug = "medicine_clinical_health"
    assert eb.task_slug(TASK) == slug == eb.topic_task(TOPIC)[len("exam_"):]
    assert jd.rubric_name(TASK) == slug
    assert jd.rubric_name(NO_RUBRIC) == "exam"            # no rubric of its own
    assert jd.rubric_name(eb.CONTROL_TASK) == "factual_accuracy"
    med, exam = jd.rubric_for(TASK), jd.rubric_for(NO_RUBRIC)
    assert "Medicine & Clinical Health question" in med.text and med.sha256 != exam.sha256
    # the 37-topic anchors are the author's own: nothing awaits his sign-off
    assert med.status == "" and exam.status == ""
    # the old rubric that did is retired with its topic, not read as this one's
    assert "DRAFT" in (RETIRED / "rubrics" / "medicine_health.md").read_text("utf-8")
    assert not (jd.RUBRIC_DIR / "medicine_health.md").exists()
    # law keeps its slug through the rename, and its file is the new one
    assert jd.rubric_name(eb.topic_task(LAW)) == "law"
    assert jd.rubric_for("exam_law").criteria is not None
    assert jd.rubric_for("exam_law").criteria_sha256 != eb.sha256_file(
        RETIRED / "rubrics" / "law.criteria.json")
    # the canary is graded with the shared rubric, as before
    assert jd.rubric_for("exam_x").sha256 == exam.sha256


def test_judge_json_records_which_rubric_graded_each_task(tree, tmp_path, monkeypatch):
    """The medicine task gets the medicine rubric's sha; a topic without one
    keeps exam.md's, and the draft stamp is on the file."""
    from service import config
    # no rubric in the 37-topic delivery is a draft, so the one that was — the
    # retired medicine & health pair — is put where an upload from the page
    # lands, under the slug of the topic that replaced it
    store = tmp_path / "rubrics"
    store.mkdir()
    for suffix in (".md", ".criteria.json"):
        (store / f"medicine_clinical_health{suffix}").write_bytes(
            (RETIRED / "rubrics" / f"medicine_health{suffix}").read_bytes())
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    d = tree["models"]["fx/chance-160m"]["dir"]
    reqs, plan = jd.plan_requests(d, "claude")
    plan["tasks"][TASK] = [{"cid": "judge:x", "qid": "a" * 64, "half": "diagnose",
                            "doc_hash": "h", "id": "i", "category": TOPIC, "answer_words": 10}]
    # every topic the fixture sits has a rubric of its own; Arts, delivered
    # empty, is the one that would be graded by the shared file
    other = NO_RUBRIC
    plan["tasks"][other] = [{"cid": "judge:y", "qid": "b" * 64, "half": "diagnose",
                             "doc_hash": "h2", "id": "j", "category": "Arts",
                             "answer_words": 10}]
    out = jd.assemble(plan, {}, jd.STUB_IDENT if hasattr(jd, "STUB_IDENT") else
                      {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1",
                       "family": "stub"}, "b", tmp_path, 0.5, False, record=False)
    rub = out["judge"]["rubrics"]
    assert rub[TASK]["sha256"] == jd.rubric_for(TASK).sha256
    assert rub[TASK]["name"] == "medicine_clinical_health"
    assert rub[TASK]["status"] == "draft"
    assert rub[other]["sha256"] == jd.rubric_for(other).sha256
    assert rub[other]["name"] == "exam" and "status" not in rub[other]
    # the file says a draft rubric graded it, and which topic's — and no
    # other, because none of the delivered ones is
    assert out["judge"]["rubric_status"] == "draft"
    assert out["judge"]["rubrics_draft"] == [TASK]


def test_a_bank_with_no_draft_rubric_carries_no_draft_stamp(tmp_path):
    plan = {"model": "m", "model_dir": str(tmp_path), "canary": [], "skipped": None,
            "tasks": {"exam_economics": [{"cid": "judge:x", "qid": "a" * 64, "half": "report",
                                          "doc_hash": "h", "id": "i",
                                          "category": "Economics", "answer_words": 10}]}}
    out = jd.assemble(plan, {}, {"provider": "stub", "model": "overlap-v1",
                                 "id": "stub/overlap-v1", "family": "stub"},
                      "b", tmp_path, 0.5, False, record=False)
    assert set(out["judge"]["rubrics"]) == {"exam_economics"}
    assert "rubric_status" not in out["judge"] and "rubrics_draft" not in out["judge"]


# ---------------------------------------------------------------------------
# the split holds for an imported bank too
# ---------------------------------------------------------------------------

def test_both_delivered_banks_keep_their_report_half_out_of_every_request(bank, tmp_path,
                                                                            monkeypatch):
    """Two topics, two criteria files, one rule: over the bodies actually
    sent, no report-half question from either bank appears."""
    from service import config, proposals as prop
    root, _ = bank
    eb.import_bank(root, LAW_V2, LAW, "Dr. Hossein", "medicine_v1")
    monkeypatch.setattr(config, "EXAM_DIR", root)
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    fake = llm.FakeBatches("fake-exam", tmp_path)
    for topic, task in ((TOPIC, TASK), (LAW, "exam_law")):
        fake.submit(eb.draft_requests(root, topic, 2))
        fake.submit([prop.proposal_request(1, "m", task, topic,
                                           [{"qid": "q", "score": 1,
                                             "justification": "vague on escalation"}],
                                           {"diagnose_items": 1, "diagnose_weak": 1}, "rubric")])
    sent = "\n".join(r["system"] + "\n" + r["user"] for r in fake.recorded())
    for topic in (TOPIC, LAW):
        rows = eb.load_bank(root)[topic]
        report = [r for r in rows if eb.half_of(r["qid"]) == "report"]
        assert report, f"the split put nothing in the report half of {topic}"
        # the prompt, the reference line and any phrase in its metadata
        assert assert_no_report_half_text(sent, rows) >= len(report)
        assert any(r["prompt"] in sent for r in rows if eb.half_of(r["qid"]) == "diagnose")


def test_an_imported_report_half_question_never_leaves_the_bank(bank, tmp_path, monkeypatch):
    """The same rule as any other question, over the bodies actually sent:
    the judge may see it, and nothing else may."""
    from service import config, proposals as prop
    root, _ = bank
    rows = eb.load_bank(root)[TOPIC]
    report = [r for r in rows if eb.half_of(r["qid"]) == "report"]
    assert report, "the split put nothing in the report half"
    monkeypatch.setattr(config, "EXAM_DIR", root)
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    fake = llm.FakeBatches("fake-exam", tmp_path)
    # what the exam writer would be shown as examples for this topic
    fake.submit(eb.draft_requests(root, TOPIC, 2))
    # and what a proposal would carry: the judge's words, nothing of the bank
    fake.submit([prop.proposal_request(1, "m", TASK, TOPIC,
                                       [{"qid": "q", "score": 1, "justification": "vague on "
                                                                                  "escalation"}],
                                       {"diagnose_items": 1, "diagnose_weak": 1}, "rubric")])
    sent = "\n".join(r["system"] + "\n" + r["user"] for r in fake.recorded())
    assert assert_no_report_half_text(sent, rows) >= len(report)
    # the diagnose half is what may be shown as an example, and is
    assert any(r["prompt"] in sent for r in rows if eb.half_of(r["qid"]) == "diagnose")
