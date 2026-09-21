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
# the author's revised bank of the old exam, kept for what it proves about
# the import: v1's hundred prompts re-delivered with new metadata. Nothing
# reads eval_tasks/fr/retired/ at run time; the test imports it into the
# current topic by hand.
LAW_V2 = REPO / "eval_tasks" / "fr" / "retired" / "law_v2.json"
# the Law bank the current exam is built from
LAW = REPO / "eval_tasks" / "fr" / "banks" / "law_v1.json"
TOPIC = "Medicine & Clinical Health"


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
                   "General & Multidisciplinary", "someone", "t")
    monkeypatch.setattr(config, "EXAM_DIR", root)
    assert prop.audience_for("General & Multidisciplinary") == ""
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
    line = prop.audience_for("Law")
    assert "plain-language guidance for a tenant or an employee" in line
    assert "not clinical notes" not in line          # the default is replaced, not appended
    assert line.startswith("Audience: ")             # the counts are still the bank's


BANKS = REPO / "eval_tasks" / "fr" / "banks"
RETIRED = REPO / "eval_tasks" / "fr" / "retired"
EXPLAINER = "set the problem up, state the assumptions, carry units and check limiting cases"


def _registers(tmp_path, monkeypatch) -> dict[str, str]:
    """The line for four real banks: two of the 37-topic exam, whose styles
    are kinds of reasoning, and the two retired ones written as people asking
    about their own situation, imported into their topics by hand."""
    from service import config
    out = {}
    for name, root, src, topic in (
            ("physics", "new", BANKS / "physics_astronomy_v1.json", "Physics & Astronomy"),
            ("economics", "new", BANKS / "economics_v1.json", "Economics"),
            ("medicine", "old", RETIRED / "medicine_v2.json", "Medicine & Clinical Health"),
            ("law", "old", RETIRED / "law_v2.json", "Law")):
        eb.import_bank(tmp_path / root, src, topic, "masein")
        monkeypatch.setattr(config, "EXAM_DIR", tmp_path / root)
        out[name] = prop.audience_for(topic)
    return out


def test_a_topic_of_worked_problems_gets_the_explainer_not_the_layperson(svc, tmp_path,
                                                                          monkeypatch):
    """Proposal #1 on the box, physics & engineering: the spec asked for
    governing equations and dimensional consistency, and document 1 opened
    "When you notice your heating or cooling system isn't performing…" —
    the layperson register 8f wrote for medicine and law, applied to every
    topic whose items carry a style."""
    lines = _registers(tmp_path, monkeypatch)
    for name in ("physics", "economics"):
        line = lines[name]
        assert line.startswith("Audience: people learning or practising the subject (styles: ")
        assert EXPLAINER in line and "name the common mistake" in line
        assert "Prose, never question-and-answer pairs" in line
        assert "layperson" not in line and "members of the public" not in line
    assert "styles: quantitative 15%, conceptual 14%, physical_explanation 13%" in lines["physics"]
    for name in ("medicine", "law"):
        line = lines[name]
        assert line.startswith("Audience: members of the public asking about their own situation")
        assert "guidance a layperson can read and act on" in line and EXPLAINER not in line
    assert "styles: conversational 85%" in lines["medicine"]
    assert "intents: legal_assessment 44%" in lines["law"]              # labels: a mix


def test_a_sentence_per_question_is_not_a_label_and_never_travels(svc, tmp_path, monkeypatch):
    """The 37-topic banks' `intent` is a different sentence on every item —
    a description of that question. The top four of them would put four
    questions, report half included, into a generator's request."""
    lines = _registers(tmp_path, monkeypatch)
    assert "intents:" not in lines["physics"] and "intents:" not in lines["economics"]
    items = json.loads((BANKS / "physics_astronomy_v1.json").read_text(encoding="utf-8"))
    for it in items:
        assert it["intent"] not in lines["physics"]
    assert prop.is_label_set({"quantitative": 3, "case_analysis": 2})
    assert not prop.is_label_set({"Distinguish path length from vector displacement": 1})


def test_the_criteria_files_audience_replaces_the_explainer_too(svc, tmp_path, monkeypatch):
    from service import config
    eb.import_bank(tmp_path / "exam", BANKS / "physics_astronomy_v1.json",
                   "Physics & Astronomy", "masein")
    monkeypatch.setattr(config, "EXAM_DIR", tmp_path / "exam")
    spec = json.loads(jd.rubric_path("physics_astronomy", ".criteria.json")
                      .read_text(encoding="utf-8"))
    spec["audience"] = "an observing log a first-year student would keep."
    (tmp_path / "rubrics").mkdir(exist_ok=True)
    (tmp_path / "rubrics" / "physics_astronomy.md").write_text(
        jd.rubric_for("exam_physics_astronomy").text, encoding="utf-8")
    (tmp_path / "rubrics" / "physics_astronomy.criteria.json").write_text(
        json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    line = prop.audience_for("Physics & Astronomy")
    assert "an observing log a first-year student would keep" in line and EXPLAINER not in line
    assert line.startswith("Audience: people learning or practising the subject")


def test_the_explainer_line_reaches_both_requests_and_the_provenance(svc, tmp_path,
                                                                    monkeypatch):
    client, _, _ = svc
    from service import config, db, llm
    audience = _registers(tmp_path, monkeypatch)["physics"]
    fake = llm.FakeBatches("fake-1", config.BENCH_ROOT)
    bids = {fake.submit([prop.proposal_request(
                1, "m", "exam_physics_astronomy", "Physics & Astronomy", [],
                {"diagnose_items": 0, "diagnose_weak": 0, "scores": {}}, "rubric",
                audience=audience)]),
            fake.submit(prop.generation_requests(1, "a spec", "Physics & Astronomy", 2, "doc",
                                                 7, audience=audience))}
    sent = [r["system"] + "\n" + r["user"] for r in fake.recorded() if r["batch_id"] in bids]
    assert len(sent) >= 2 and all(EXPLAINER in body for body in sent)
    pid = db.proposal_create("fx/good-750m", "exam_physics_astronomy", "Physics & Astronomy",
                             "omar", {"n": 1})
    db.proposal_update(pid, status="approved", spec_text="a spec", approver="omar",
                       approved_at=1.0, proposer="fake/fake-1")
    did = db.dataset_create(pid, "doc", 2, "omar", "")
    p = prop.provenance(db.proposal_get(pid), db.dataset_get(did), "fake/fake-1", "b1", "sha",
                        {"rejected": False}, "s", 2, 2, audience=audience)
    assert EXPLAINER in p["audience"]


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
    reqs = prop.generation_requests(1, "spec", "Law", 1, "doc", 3, audience="Audience: X\nY")
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
    v2 = json.loads(LAW_V2.read_text(encoding="utf-8"))
    v1 = [{k: v for k, v in it.items() if k != "jurisdiction_required"} for it in v2]
    for it in v1:                       # the id-range mapping v1 carried
        i = it["id"]
        it["difficulty"] = 1 if i <= 20 else 2 if i <= 45 else 3 if i <= 65 else 4 if i <= 85 else 5
    first = eb.import_bank(root, v1, "Law", "Dr. Hossein", "medicine_v1")
    assert (first["imported"], first["updated"], first["skipped"]) == (100, 0, 0)
    halves = {r["qid"]: eb.half_of(r["qid"]) for r in eb.load_bank(root)["Law"]}
    second = eb.import_bank(root, LAW_V2, "Law", "Dr. Hossein", "law_v2")
    assert (second["imported"], second["updated"], second["skipped"]) == (0, 100, 0)
    rows = eb.load_bank(root)["Law"]
    assert len(rows) == 100
    assert {r["source"] for r in rows} == {"law_v2"}
    assert {r["accepted_by"] for r in rows} == {"Dr. Hossein"}
    assert {r["qid"]: eb.half_of(r["qid"]) for r in rows} == halves     # nothing moved
    assert all("jurisdiction_required" in r["meta"] for r in rows)
    assert sum(1 for r in rows if r["meta"]["jurisdiction_required"]) == 85
    assert sum(1 for r in rows if r["meta"]["difficulty"] == 2) == 33
    # and a third import of the same file changes nothing
    again = eb.import_bank(root, LAW_V2, "Law", "Dr. Hossein", "law_v2")
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
    assert "jurisdiction_and_authority" in jd.criteria_ids(spec)


def test_law_is_tabulated_by_jurisdiction_and_medicine_is_not(tmp_path):
    spec = jd.rubric_for("exam_law").criteria
    law_items = [{"cid": f"c{i}", "qid": f"{i:064d}", "half": "diagnose", "doc_hash": str(i),
                  "id": i, "category": "Law", "answer_words": 10, "score": 2, "graded": True,
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
    # medicine's items carry the field but it is never set — every item of
    # the delivered bank says no — so no such table
    med_spec = jd.rubric_for("exam_medicine_clinical_health").criteria
    med_items = [{**law_items[0], "criteria": {c: 0.5 for c in jd.criteria_ids(med_spec)},
                  "flags": {f: False for f in jd.flag_ids(med_spec)},
                  "meta": {"acuity": "routine", "difficulty": 1,
                           "jurisdiction_required": False}}]
    med = jd._criteria_blocks(med_items, med_spec)
    # one item carries one value of everything, so nothing splits it: no
    # tables at all, and the constant fields are named instead
    assert "breakdowns" not in med
    assert set(med["breakdowns_constant"]) == {"acuity", "difficulty", "jurisdiction_required"}
    assert med["breakdowns_constant"]["jurisdiction_required"] == "False"
