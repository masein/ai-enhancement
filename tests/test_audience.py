"""Phase 8f: the generator was writing for the wrong reader.

Both demo runs of 2026-09-20 passed the gate and produced the wrong register
— clinical case notes for medicine, advisory memoranda for law — because the
generator sees the approved spec and nothing else, and the spec never carried
the audience. It could not: the proposal step did not have it either. These
tests are over what travels: labels and percentages from the bank's own
metadata, and nothing else from the bank."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
from conftest import make_service
from service import proposals as prop

REPO = Path(__file__).resolve().parents[1]
LAW = REPO / "eval_tasks" / "fr" / "law_v2.json"
TOPIC = "medicine & health"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# the line itself
# ---------------------------------------------------------------------------

def test_the_audience_is_built_from_the_banks_own_labels(svc, tmp_path, monkeypatch):
    from service import config
    root = tmp_path / "bank-exam"
    items = [{"prompt": f"A question about a rash on my arm, number {i}, long enough to count.",
              "style": "conversational" if i < 8 else "telegraphic",
              "subject": "self" if i < 6 else "child",
              "intent": "symptom_assessment_triage"} for i in range(10)]
    eb.import_bank(root, items, TOPIC, "Dr. Hossein", "t")
    monkeypatch.setattr(config, "EXAM_DIR", root)
    line = prop.audience_for(TOPIC)
    assert line.startswith("Audience: members of the public asking about their own situation")
    assert "styles: conversational 80%, telegraphic 20%" in line
    assert "subjects: self 60%, child 40%" in line
    assert "intents: symptom_assessment_triage 100%" in line
    assert "guidance a layperson can read and act on" in line
    assert "not clinical notes, case files, legal memoranda" in line
    # labels and percentages — no question, no qid
    for it in items:
        assert it["prompt"] not in line
    for r in eb.load_bank(root)[TOPIC]:
        assert r["qid"] not in line


def test_a_topic_whose_bank_says_nothing_about_its_reader_gets_no_line(svc, tmp_path,
                                                                       monkeypatch):
    from service import config
    root = tmp_path / "bare-exam"
    eb.import_bank(root, [{"prompt": "A question with no metadata at all, long enough."}],
                   "other", "someone", "t")
    monkeypatch.setattr(config, "EXAM_DIR", root)
    assert prop.audience_for("other") == ""
    # and the fixture's own legacy topic, for the same reason
    monkeypatch.setattr(config, "EXAM_DIR", Path(config.BENCH_ROOT) / "exam")


def test_the_author_may_write_the_register_sentence_herself(svc, tmp_path, monkeypatch):
    from service import config
    spec = json.loads(jd.rubric_path("law", ".criteria.json").read_text(encoding="utf-8"))
    spec["audience"] = "plain-language guidance for a tenant or an employee, never a memo."
    (tmp_path / "rubrics").mkdir(exist_ok=True)
    (tmp_path / "rubrics" / "law.md").write_text(jd.rubric_for("exam_law").text, encoding="utf-8")
    (tmp_path / "rubrics" / "law.criteria.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    line = prop.audience_for("law")
    assert "plain-language guidance for a tenant or an employee" in line
    assert "not clinical notes" not in line          # the default is replaced, not appended
    assert line.startswith("Audience: ")             # the counts are still the bank's


# ---------------------------------------------------------------------------
# where it travels
# ---------------------------------------------------------------------------

def test_the_audience_reaches_both_requests_and_nothing_else_does(svc):
    """The bodies actually sent: the labels are there, and no question from
    either half of the bank is."""
    client, _, _ = svc
    from service import config, llm
    audience = prop.audience_for(TOPIC)
    assert audience
    task = eb.topic_task(TOPIC)
    model_dir = config.OUT_DIR / "fx__good-750m"
    items, counts = prop.justifications_for(model_dir, task)
    fake = llm.FakeBatches("fake-1", config.BENCH_ROOT)
    fake.submit([prop.proposal_request(1, "m", task, TOPIC, items, counts, "rubric",
                                       audience=audience)])
    fake.submit(prop.generation_requests(1, "a spec", TOPIC, 2, "doc", 7, audience=audience))
    sent = "\n".join(r["system"] + "\n" + r["user"] for r in fake.recorded())
    assert sent.count("Audience: members of the public") == 2      # both requests
    assert "Register for documents:" in sent
    assert "Who asks the questions on this topic:" in sent         # the proposal's heading
    for r in eb.load_bank(config.EXAM_DIR)[TOPIC]:
        assert r["prompt"] not in sent and r["prompt"][:60] not in sent
        assert r["qid"] not in sent


def test_the_generation_prompt_asks_for_something_a_person_would_read():
    reqs = prop.generation_requests(1, "spec", "law", 1, "doc", 3, audience="Audience: X\nY")
    body = reqs[0].user
    assert "Audience: X" in body
    assert "what a person with that question would be helped by reading" in body
    assert "never shaped as a question followed by its answer" in body
    # and the rule it sits beside is still there
    assert "no answer keys" in body


def test_provenance_records_the_audience_the_generator_was_given(svc, tmp_path):
    client, _, _ = svc
    from service import db
    pid = db.proposal_create("fx/good-750m", eb.topic_task(TOPIC), TOPIC, "omar", {"n": 1})
    db.proposal_update(pid, status="approved", spec_text="a spec", approver="omar",
                       approved_at=1.0, proposer="fake/fake-1")
    did = db.dataset_create(pid, "doc", 2, "omar", "")
    p = prop.provenance(db.proposal_get(pid), db.dataset_get(did), "fake/fake-1", "b1", "sha",
                        {"rejected": False}, "s", 2, 2, audience=prop.audience_for(TOPIC))
    assert p["audience"].startswith("Audience: members of the public")
    assert "audience" not in prop.provenance_complete(p)
    # a topic with no audience is not a hole in the record
    p2 = prop.provenance(db.proposal_get(pid), db.dataset_get(did), "fake/fake-1", "b1", "sha",
                         {"rejected": False}, "s", 2, 2, audience="")
    assert prop.provenance_complete(p2) == prop.provenance_complete(p)


# ---------------------------------------------------------------------------
# the per-role token caps
# ---------------------------------------------------------------------------

def test_each_role_has_its_own_cap_with_one_fallback(monkeypatch):
    from service import config
    for k in ("LOCAL_MAX_TOKENS", "LOCAL_MAX_TOKENS_LLM", "LOCAL_MAX_TOKENS_JUDGE",
              "LOCAL_MAX_TOKENS_EXAM"):
        monkeypatch.setattr(config, k, 0)
    # the defaults: only generation needs the headroom
    assert config.local_max_tokens("llm") == 1536
    assert config.local_max_tokens("judge") == 1024
    assert config.local_max_tokens("exam") == 1024
    # the old single knob still caps every role that has none of its own
    monkeypatch.setattr(config, "LOCAL_MAX_TOKENS", 800)
    assert config.local_max_tokens("llm") == 800 and config.local_max_tokens("judge") == 800
    # and a role's own knob wins
    monkeypatch.setattr(config, "LOCAL_MAX_TOKENS_LLM", 2048)
    assert config.local_max_tokens("llm") == 2048 and config.local_max_tokens("judge") == 800


def test_the_local_backend_takes_the_cap_of_the_role_it_serves(monkeypatch, tmp_path):
    from service import config, llm
    for k, v in (("LOCAL_MAX_TOKENS", 0), ("LOCAL_MAX_TOKENS_LLM", 0),
                 ("LOCAL_MAX_TOKENS_JUDGE", 0), ("LOCAL_MAX_TOKENS_EXAM", 0)):
        monkeypatch.setattr(config, k, v)
    monkeypatch.setattr(llm.LocalOpenAI, "_check_served", lambda self: ["chat"])
    gen = llm.LocalOpenAI("chat", "", tmp_path, role="llm")
    judge = llm.LocalOpenAI("chat", "", tmp_path, role="judge")
    assert gen.max_tokens == 1536 and judge.max_tokens == 1024
    # a truncation names the knob that actually capped it
    assert gen.cap_name == "LOCAL_MAX_TOKENS_LLM (default)"
    monkeypatch.setattr(config, "LOCAL_MAX_TOKENS_LLM", 2048)
    assert llm.LocalOpenAI("chat", "", tmp_path, role="llm").cap_name == "LOCAL_MAX_TOKENS_LLM"


# ---------------------------------------------------------------------------
# the author's revised law bank
# ---------------------------------------------------------------------------

def test_the_revised_law_bank_updates_metadata_in_place(tmp_path):
    """v2 is v1's hundred prompts with his own difficulty levels and a
    jurisdiction flag. The prompt is the identity: the qids and the halves do
    not move, and the records are revised rather than skipped."""
    root = tmp_path / "exam"
    v2 = json.loads(LAW.read_text(encoding="utf-8"))
    v1 = [{k: v for k, v in it.items() if k != "jurisdiction_required"} for it in v2]
    for it in v1:                       # the id-range mapping v1 carried
        i = it["id"]
        it["difficulty"] = 1 if i <= 20 else 2 if i <= 45 else 3 if i <= 65 else 4 if i <= 85 else 5
    first = eb.import_bank(root, v1, "law", "Dr. Hossein", "medicine_v1")
    assert (first["imported"], first["updated"], first["skipped"]) == (100, 0, 0)
    halves = {r["qid"]: eb.half_of(r["qid"]) for r in eb.load_bank(root)["law"]}
    second = eb.import_bank(root, LAW, "law", "Dr. Hossein", "law_v2")
    assert (second["imported"], second["updated"], second["skipped"]) == (0, 100, 0)
    rows = eb.load_bank(root)["law"]
    assert len(rows) == 100
    assert {r["source"] for r in rows} == {"law_v2"}
    assert {r["accepted_by"] for r in rows} == {"Dr. Hossein"}
    assert {r["qid"]: eb.half_of(r["qid"]) for r in rows} == halves     # nothing moved
    assert all("jurisdiction_required" in r["meta"] for r in rows)
    assert sum(1 for r in rows if r["meta"]["jurisdiction_required"]) == 85
    assert sum(1 for r in rows if r["meta"]["difficulty"] == 2) == 33
    # and a third import of the same file changes nothing
    again = eb.import_bank(root, LAW, "law", "Dr. Hossein", "law_v2")
    assert (again["imported"], again["updated"], again["skipped"]) == (0, 0, 100)


def test_the_judge_reads_the_jurisdiction_flag(tmp_path):
    items = json.loads(LAW.read_text(encoding="utf-8"))
    yes = next(it for it in items if it["jurisdiction_required"])
    no = next(it for it in items if not it["jurisdiction_required"])
    assert "Difficulty: " in eb.metadata_reference(yes)
    assert eb.metadata_reference(yes).index("Jurisdiction required: yes.") \
        > eb.metadata_reference(yes).index("Difficulty:")
    assert "Jurisdiction required: no." in eb.metadata_reference(no)
    # the criterion that reads it is the author's own
    spec = jd.rubric_for("exam_law").criteria
    assert "jurisdiction_awareness" in jd.criteria_ids(spec)


def test_law_is_tabulated_by_jurisdiction_and_medicine_is_not(tmp_path):
    spec = jd.rubric_for("exam_law").criteria
    law_items = [{"cid": f"c{i}", "qid": f"{i:064d}", "half": "diagnose", "doc_hash": str(i),
                  "id": i, "category": "law", "answer_words": 10, "score": 2, "graded": True,
                  "criteria": {c: 0.5 for c in jd.criteria_ids(spec)},
                  "flags": {f: False for f in jd.flag_ids(spec)},
                  "meta": {"acuity": "routine" if i % 2 else "urgent",
                           "difficulty": 1 + i % 3,
                           "jurisdiction_required": i % 2 == 0}}
                 for i in range(4)]
    blocks = jd._criteria_blocks(law_items, spec)
    assert "jurisdiction_required" in blocks["breakdowns"]
    assert list(blocks["breakdowns"]["jurisdiction_required"]) == ["True", "False"]
    assert blocks["breakdowns"]["jurisdiction_required"]["True"]["n"] == 2
    # the order a person reads them in, for the fields that split this bank
    assert list(blocks["breakdowns"]) == ["acuity", "difficulty", "jurisdiction_required"]
    # medicine's items carry no such field, so no such table
    med_spec = jd.rubric_for("exam_medicine_health").criteria
    med_items = [{**law_items[0], "criteria": {c: 0.5 for c in jd.criteria_ids(med_spec)},
                  "flags": {f: False for f in jd.flag_ids(med_spec)},
                  "meta": {"acuity": "mild", "difficulty": 1}}]
    med = jd._criteria_blocks(med_items, med_spec)
    # one item carries one value of everything, so nothing splits it: no
    # tables at all, and the constant fields are named instead
    assert "breakdowns" not in med
    assert set(med["breakdowns_constant"]) == {"acuity", "difficulty"}
