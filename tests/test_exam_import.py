"""Phase 8b P4a: a human-written bank, and a rubric per topic.

Dr. Hossein's 50 consumer health questions come with metadata instead of
reference answers, and with their own rubric. These tests are over the real
delivered files — if the file changes, they are what says so."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from service import llm

REPO = Path(__file__).resolve().parents[1]
MEDICINE = REPO / "eval_tasks" / "fr" / "hossein_medicine_v1.json"
TOPIC = "medicine & health"
TASK = "exam_medicine_health"


@pytest.fixture
def bank(tmp_path) -> tuple[Path, dict]:
    root = tmp_path / "exam"
    r = eb.import_bank(root, MEDICINE, TOPIC, "Dr. Hossein", "hossein_v1")
    return root, r


# ---------------------------------------------------------------------------
# the import
# ---------------------------------------------------------------------------

def test_the_delivered_file_is_what_the_import_expects():
    items = json.loads(MEDICINE.read_text(encoding="utf-8"))
    assert isinstance(items, list) and len(items) == 50
    assert all(isinstance(it, dict) and it.get("prompt") for it in items)
    assert not any(it.get("reference") for it in items)      # metadata instead
    assert {it["acuity"] for it in items} == {"emergency", "urgent", "moderate", "mild", "routine"}


def test_import_round_trips_fifty_items_and_is_idempotent(bank):
    root, r = bank
    assert (r["imported"], r["skipped"], r["invalid"]) == (50, 0, 0)
    assert r["report"] + r["diagnose"] == 50 and r["report"] > 10 and r["diagnose"] > 10
    assert sum(r["acuity"].values()) == 50 and r["acuity"]["emergency"] == 3
    rows = eb.load_bank(root)[TOPIC]
    assert len(rows) == 50
    one = next(x for x in rows if x["meta"]["id"] == 1)
    assert one["source"] == "hossein_v1" and one["accepted_by"] == "Dr. Hossein"
    assert one["edited"] is False and one["notes"] == "" and one["accepted_at"] > 0
    assert one["qid"] == eb.qid_of(one["prompt"]) and one["topic"] == TOPIC
    # his metadata is kept whole, under meta
    assert one["meta"] == {"id": 1, "intent": "symptom_assessment_triage", "subject": "child",
                           "age_group": "5-12", "sex": "male", "acuity": "moderate",
                           "domain": "respiratory_infectious", "style": "conversational"}
    again = eb.import_bank(root, MEDICINE, TOPIC, "Dr. Hossein", "hossein_v1")
    assert (again["imported"], again["skipped"]) == (0, 50)
    assert len(eb.load_bank(root)[TOPIC]) == 50


def test_the_reference_is_the_metadata_in_a_stable_order(bank):
    root, _ = bank
    one = next(x for x in eb.load_bank(root)[TOPIC] if x["meta"]["id"] == 1)
    assert one["reference"] == ("Acuity: moderate. Intent: symptom_assessment_triage. "
                                "Domain: respiratory_infectious. Subject: child (male, 5-12). "
                                "Style: conversational.")
    # field order is fixed, not dict order, so the same item always hashes the same
    scrambled = {"style": "telegraphic", "domain": "cardiovascular", "sex": "male",
                 "acuity": "emergency", "age_group": "45-59", "subject": "self",
                 "intent": "symptom_assessment_triage"}
    assert eb.metadata_reference(scrambled) == (
        "Acuity: emergency. Intent: symptom_assessment_triage. Domain: cardiovascular. "
        "Subject: self (male, 45-59). Style: telegraphic.")
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
                        "--topic", TOPIC, "--approver", "Dr. Hossein", "--source", "hossein_v1"],
                       capture_output=True, text=True, timeout=120, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "imported 50, skipped 0" in r.stdout and "report" in r.stdout
    assert "acuity: emergency 3" in r.stdout
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
    assert m["tasks"][TASK]["items"] == 50 and len(items) == 50
    assert all(it["meta"]["acuity"] for it in items)
    assert items[0]["prompt"] and items[0]["reference"].startswith("Acuity: ")


def test_a_report_half_import_is_as_withheld_as_any_other_question(bank):
    root, _ = bank
    public = eb.public_bank(root, TOPIC)
    assert len(public) == 50
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
    assert eb.task_slug(TASK) == "medicine_health" == eb.topic_task(TOPIC)[len("exam_"):]
    assert jd.rubric_name(TASK) == "medicine_health"
    assert jd.rubric_name("exam_economics") == "exam"          # no rubric of its own
    assert jd.rubric_name(eb.CONTROL_TASK) == "factual_accuracy"
    med, exam = jd.rubric_for(TASK), jd.rubric_for("exam_economics")
    assert "consumer health question" in med.text and med.sha256 != exam.sha256
    assert med.version == "1" and med.status == "draft"        # until its author signs it off
    assert exam.status == ""
    # the canary is graded with the shared rubric, as before
    assert jd.rubric_for("exam_x").sha256 == exam.sha256


def test_judge_json_records_which_rubric_graded_each_task(tree, tmp_path, monkeypatch):
    """The medicine task gets the medicine rubric's sha; everything else
    keeps exam.md's, and the draft stamp is on the file."""
    d = tree["models"]["fx/chance-160m"]["dir"]
    reqs, plan = jd.plan_requests(d, "claude")
    plan["tasks"][TASK] = [{"cid": "judge:x", "qid": "a" * 64, "half": "diagnose",
                            "doc_hash": "h", "id": "i", "category": TOPIC, "answer_words": 10}]
    out = jd.assemble(plan, {}, jd.STUB_IDENT if hasattr(jd, "STUB_IDENT") else
                      {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1",
                       "family": "stub"}, "b", tmp_path, 0.5, False, record=False)
    rub = out["judge"]["rubrics"]
    assert rub[TASK]["sha256"] == jd.rubric_for(TASK).sha256
    assert rub[TASK]["name"] == "medicine_health" and rub[TASK]["status"] == "draft"
    other = next(t for t in rub if t.startswith("exam_") and t != TASK)
    assert rub[other]["sha256"] == jd.rubric_for("exam_economics").sha256
    assert rub[other]["name"] == "exam" and "status" not in rub[other]
    # the file says a draft rubric graded it, and which topic's
    assert out["judge"]["rubric_status"] == "draft"
    assert TASK in out["judge"]["rubrics_draft"]      # and any other draft topic


def test_a_bank_with_no_draft_rubric_carries_no_draft_stamp(tmp_path):
    plan = {"model": "m", "model_dir": str(tmp_path), "canary": [], "skipped": None,
            "tasks": {"exam_economics": [{"cid": "judge:x", "qid": "a" * 64, "half": "report",
                                          "doc_hash": "h", "id": "i", "category": "economics",
                                          "answer_words": 10}]}}
    out = jd.assemble(plan, {}, {"provider": "stub", "model": "overlap-v1",
                                 "id": "stub/overlap-v1", "family": "stub"},
                      "b", tmp_path, 0.5, False, record=False)
    assert set(out["judge"]["rubrics"]) == {"exam_economics"}
    assert "rubric_status" not in out["judge"] and "rubrics_draft" not in out["judge"]


# ---------------------------------------------------------------------------
# the split holds for an imported bank too
# ---------------------------------------------------------------------------

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
    for r in report:
        assert r["prompt"] not in sent
        assert r["prompt"][:60] not in sent
    # the diagnose half is what may be shown as an example, and is
    assert any(r["prompt"] in sent for r in rows if eb.half_of(r["qid"]) == "diagnose")
