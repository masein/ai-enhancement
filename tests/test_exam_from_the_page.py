"""Phase 8c P5c: a person delivers a bank and the rubric that grades it,
from the page.

Dr. Hossein is writing about 100 questions per topic with a criteria file
each. Until now the questions went in by CLI and the rubrics by git commit,
and he has neither a shell on the box nor a checkout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import make_service

REPO = Path(__file__).resolve().parents[1]
MEDICINE = REPO / "eval_tasks" / "fr" / "hossein_medicine_v1.json"
TOPIC = "medicine & health"
ITEMS = json.loads(MEDICINE.read_text(encoding="utf-8"))


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    # a test never writes into the checkout: on a developer's machine the
    # repo's rubrics directory IS writable, which is what the service prefers
    from service import config
    monkeypatch.setattr(appmod, "_rubric_store",
                        lambda: (Path(config.BENCH_ROOT) / "rubrics", False))
    yield client, appmod
    client.__exit__(None, None, None)


def post(client, path, **body):
    return client.post(path, json=body)


# ---------------------------------------------------------------------------
# importing a bank
# ---------------------------------------------------------------------------

def test_preview_writes_nothing_and_says_what_would_land(svc):
    client, appmod = svc
    from service import config
    before = len(eb.load_bank(config.EXAM_DIR).get(TOPIC, []))
    r = post(client, "/api/exam/import/preview", topic=TOPIC, approver="Dr. Hossein",
             source="hossein_v1", items=ITEMS)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["imported"] == 50 and p["skipped"] == 0 and p["invalid"] == 0
    assert p["report"] + p["diagnose"] == 50 and p["report"] > 10
    assert p["acuity"]["emergency"] == 3 and p["intent"]["symptom_assessment_triage"] == 17
    assert len(p["items"]) == 50
    assert len(eb.load_bank(config.EXAM_DIR).get(TOPIC, [])) == before      # nothing written


def test_the_preview_withholds_a_report_half_prompt(svc):
    client, _ = svc
    p = post(client, "/api/exam/import/preview", topic=TOPIC, approver="Dr. Hossein",
             items=ITEMS).json()
    report_rows = [it for it in p["items"] if it["half"] == "report"]
    diagnose_rows = [it for it in p["items"] if it["half"] == "diagnose"]
    assert report_rows and diagnose_rows
    assert all(it["prompt"] is None and "never shown" in it["withheld"] for it in report_rows)
    assert all(it["prompt"] for it in diagnose_rows)
    # not in the body anywhere — not under another key, not in the reference
    body = json.dumps(p)
    for it in ITEMS:
        if eb.half_of(eb.qid_of(it["prompt"])) == "report":
            assert it["prompt"] not in body
    # the metadata line still shows, for every half: it is the ground truth
    assert all(it["reference"].startswith("Acuity: ") for it in p["items"])
    assert all(it["meta"]["acuity"] for it in p["items"])


def test_commit_writes_the_same_records_as_the_cli(svc, tmp_path):
    client, _ = svc
    from service import config
    r = post(client, "/api/exam/import", topic=TOPIC, approver="Dr. Hossein",
             source="hossein_v1", items=ITEMS)
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 50
    # the fixture's bank already holds drafted questions for this topic; the
    # imported ones are the ones to compare
    from_page = [r for r in eb.load_bank(config.EXAM_DIR)[TOPIC]
                 if r.get("source") == "hossein_v1"]
    # the same file through the CLI's own entry point, into a fresh bank
    cli_root = tmp_path / "cli-exam"
    eb.import_bank(cli_root, MEDICINE, TOPIC, "Dr. Hossein", "hossein_v1")
    from_cli = eb.load_bank(cli_root)[TOPIC]
    assert len(from_page) == len(from_cli) == 50
    drop = lambda rows: [{k: v for k, v in r.items() if k != "accepted_at"}      # noqa: E731
                         for r in sorted(rows, key=lambda x: x["qid"])]
    assert drop(from_page) == drop(from_cli)
    # and the decision is on the record, the way an accept is
    from service import db
    row = db.curation_list(5)[0]
    assert row["decision"] == "imported" and row["approver"] == "Dr. Hossein"
    assert row["topic"] == TOPIC and "50 imported" in row["reason"]


def test_a_second_import_of_the_same_file_changes_nothing(svc):
    client, _ = svc
    post(client, "/api/exam/import", topic=TOPIC, approver="Dr. Hossein", items=ITEMS)
    again = post(client, "/api/exam/import", topic=TOPIC, approver="Dr. Hossein",
                 items=ITEMS).json()
    assert (again["imported"], again["skipped"]) == (0, 50)
    pre = post(client, "/api/exam/import/preview", topic=TOPIC, approver="Dr. Hossein",
               items=ITEMS).json()
    assert pre["skipped"] == 50 and len(pre["duplicates"]) == 50


def test_what_the_import_refuses(svc):
    client, _ = svc
    assert post(client, "/api/exam/import/preview", topic=TOPIC, approver="  ",
                items=ITEMS).status_code == 422
    assert post(client, "/api/exam/import/preview", topic="astrology",
                approver="X", items=ITEMS).status_code == 422
    assert post(client, "/api/exam/import/preview", topic=TOPIC, approver="X",
                text='{"prompt": "not an array"}').status_code == 422
    assert post(client, "/api/exam/import/preview", topic=TOPIC, approver="X",
                text="not json at all").status_code == 422
    big = post(client, "/api/exam/import/preview", topic=TOPIC, approver="X",
               text="[" + "0," * 1_200_000 + "0]")
    assert big.status_code == 413
    # an item with no prompt is counted and named, not silently dropped
    p = post(client, "/api/exam/import/preview", topic=TOPIC, approver="X",
             items=[{"id": 7, "acuity": "mild"}, {"prompt": "short"}] + ITEMS[:2]).json()
    assert p["invalid"] == 2 and p["imported"] == 2
    assert p["invalid_items"][0]["index"] == 0 and p["invalid_items"][0]["id"] == 7


# ---------------------------------------------------------------------------
# the rubric and the criteria file
# ---------------------------------------------------------------------------

def test_the_page_says_which_rubric_grades_each_topic(svc):
    client, _ = svc
    r = client.get("/api/exam/rubrics").json()
    rows = {t["topic"]: t for t in r["topics"]}
    med = rows[TOPIC]
    assert med["name"] == "medicine_health" and med["own"] is True
    assert med["status"] == "draft" and med["scoring"] == "criteria"
    assert med["criteria_count"] == 16 and len(med["criteria_sha256"]) == 64
    assert rows["history"]["name"] == "exam" and rows["history"]["own"] is False
    assert rows["history"]["scoring"] == "single"
    assert r["store"] and isinstance(r["in_repo"], bool)
    # and both files come back whole
    md = client.get("/api/exam/rubrics/medicine_health")
    assert md.status_code == 200 and md.text.startswith("# Rubric — medicine & health")
    spec = client.get("/api/exam/rubrics/medicine_health?kind=criteria")
    assert json.loads(spec.text)["rubric"] == "medicine_health"
    assert client.get("/api/exam/rubrics/nothing_here").status_code == 404


def test_a_criteria_upload_is_checked_before_it_can_be_saved(svc):
    client, _ = svc
    good = json.loads(jd.rubric_path("medicine_health", ".criteria.json").read_text())
    bad = {**good, "criteria": [
        {"id": "Relevance", "label": "", "definition": "", "weight": 0},
        {"id": "triage", "label": "T", "definition": "d", "weight": 1, "conditional": True},
        {"id": "triage", "label": "T", "definition": "d", "weight": "heavy"}]}
    bad.pop("critical_safety_failure")
    bad["fold"] = {"method": "vibes"}
    r = post(client, "/api/exam/rubrics/preview", name="medicine_health", kind="criteria",
             content=json.dumps(bad), approver="Dr. Hossein").json()
    joined = " | ".join(r["problems"])
    assert r["ok"] is False
    assert "fold.method 'vibes' is not one this judge knows" in joined
    assert "critical_safety_failure needs a definition" in joined
    assert "an id is lower-case letters" in joined and "duplicate id" in joined
    assert "has no label" in joined and "weight must be greater than 0" in joined
    assert "weight is not a number" in joined and "needs applies_when" in joined
    # and the commit refuses too — the preview is not the only gate
    assert post(client, "/api/exam/rubrics", name="medicine_health", kind="criteria",
                content=json.dumps(bad), approver="Dr. Hossein").status_code == 422
    assert post(client, "/api/exam/rubrics/preview", name="medicine_health", kind="criteria",
                content="{not json", approver="Dr. Hossein").json()["problems"][0].startswith(
        "not valid JSON")


def test_a_rubric_upload_wants_a_heading_and_five_anchors(svc):
    client, _ = svc
    r = post(client, "/api/exam/rubrics/preview", name="medicine_health", kind="rubric",
             content="Just some prose.", approver="Dr. Hossein").json()
    assert r["ok"] is False
    assert any("# Rubric" in p for p in r["problems"])
    assert sum("no anchor for" in p for p in r["problems"]) == 5


def test_committing_a_rubric_changes_what_the_judge_reads_and_is_recorded(svc):
    client, appmod = svc
    from service import config, db
    before = jd.rubric_for("exam_medicine_health")
    text = jd.rubric_path("medicine_health").read_text(encoding="utf-8")
    signed_off = text.replace(", DRAFT — awaiting Dr. Hossein's sign-off", "")
    pre = post(client, "/api/exam/rubrics/preview", name="medicine_health", kind="rubric",
               content=signed_off, approver="Dr. Hossein").json()
    assert pre["ok"] and pre["changed"] and pre["existed"]
    assert pre["was_sha256"] == before.sha256 and pre["sha256"] != before.sha256
    assert any(line.startswith("-# Rubric") for line in pre["diff"])
    assert "not comparable" in pre["warning"]
    r = post(client, "/api/exam/rubrics", name="medicine_health", kind="rubric",
             content=signed_off, approver="Dr. Hossein", note="signed off in the meeting")
    assert r.status_code == 200, r.text
    written = Path(r.json()["written"])
    assert written.is_file() and written.read_text(encoding="utf-8") == signed_off
    # the judge reads the new one, and the draft stamp is gone
    after = jd.rubric_for("exam_medicine_health")
    assert after.sha256 == pre["sha256"] and after.status == ""
    assert after.criteria is not None                     # the criteria file is untouched
    row = db.rubric_changes(1)[0]
    assert row["name"] == "medicine_health" and row["kind"] == "rubric"
    assert row["approver"] == "Dr. Hossein" and row["note"] == "signed off in the meeting"
    assert row["sha256"] == after.sha256 and row["was_sha256"] == before.sha256
    assert client.get("/api/exam/rubrics").json()["changes"][0]["id"] == row["id"]
    assert str(config.BENCH_ROOT) in row["path"] or str(jd.RUBRIC_DIR) in row["path"]


def test_the_store_is_the_checkout_when_it_can_be_written_and_bench_root_otherwise(monkeypatch,
                                                                                    tmp_path):
    """In the container /app is the image, not the checkout: the upload has
    to land somewhere the judge will look, and judge.rubric_dirs() looks
    under BENCH_ROOT first for exactly this reason."""
    import os as _os
    from service import app as appmod, config
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    assert appmod._rubric_store() == (jd.RUBRIC_DIR, True)          # a writable checkout
    monkeypatch.setattr(_os, "access", lambda p, mode: False)
    assert appmod._rubric_store() == (tmp_path / "rubrics", False)   # a read-only image
    assert jd.rubric_dirs()[0] == tmp_path / "rubrics"


def test_an_uploaded_rubric_beats_the_repo_copy(svc):
    """A topic with no rubric of its own gains one from outside the repo, and
    the judge reads it without the checkout being touched."""
    client, _ = svc
    from service import config
    body = "# Rubric — history (version 9)\n" + "".join(
        f"- **{i}** — anchor {i}\n" for i in range(5)) + "\nLength: short.\n"
    r = post(client, "/api/exam/rubrics", name="history", kind="rubric", content=body,
             approver="Dr. Hossein")
    assert r.status_code == 200, r.text
    assert str(config.BENCH_ROOT) in r.json()["written"]
    # history had no rubric of its own; now it does, from outside the repo
    assert jd.rubric_name("exam_history") == "history"
    got = jd.rubric_for("exam_history")
    assert got.version == "9" and got.text == body
    assert (REPO / "eval_tasks" / "fr" / "rubrics" / "history.md").exists() is False


def test_a_page_imported_report_half_question_never_leaves_the_bank(svc, tmp_path):
    """The rule does not care how the question arrived. Over the bodies
    actually sent: the judge may see a report-half prompt, and nothing else."""
    from service import config, llm, proposals as prop
    client, _ = svc
    assert post(client, "/api/exam/import", topic=TOPIC, approver="Dr. Hossein",
                source="hossein_v1", items=ITEMS).status_code == 200
    rows = eb.load_bank(config.EXAM_DIR)[TOPIC]
    report = [r for r in rows if eb.half_of(r["qid"]) == "report"]
    assert report, "the split put nothing in the report half"
    fake = llm.FakeBatches("fake-exam", tmp_path)
    fake.submit(eb.draft_requests(config.EXAM_DIR, TOPIC, 2))
    fake.submit([prop.proposal_request(1, "m", "exam_medicine_health", TOPIC,
                                       [{"qid": "q", "score": 1,
                                         "justification": "vague on escalation"}],
                                       {"diagnose_items": 1, "diagnose_weak": 1}, "rubric")])
    sent = "\n".join(r["system"] + "\n" + r["user"] for r in fake.recorded())
    for r in report:
        assert r["prompt"] not in sent and r["prompt"][:60] not in sent
    # what the page itself serves for this topic is withheld the same way
    body = client.get(f"/api/exam/bank?topic={TOPIC}").text
    for r in report:
        assert r["prompt"][:60] not in body
