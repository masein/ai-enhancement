"""11g: what the Reader reads, from the service's side.

Each file a person can read in the page has its own JSON, shaped for reading
— and each keeps the rule that does not bend: no hidden (report-half)
question's text, answer or qid reaches the page. The bank reader gets the
practice half and a count; the log reader withholds a line that quotes a
hidden question; "how this was graded" never carries an item.
"""

from __future__ import annotations

import hashlib
import json

import pytest

import exam_build as eb
from conftest import assert_no_report_half_text, label_domains, make_service
from service import config, db, llm, llm_poller
from service import proposals as prop

TOPIC = "Economics"
MODEL = "fx/good-750m"
SHORT = "A short note. " * 20               # 60 words: under DOC_MIN_WORDS


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, tree
    client.__exit__(None, None, None)


def no_hidden(body: str, rows: list[dict]) -> int:
    """No hidden question's qid, and none of its text whole. (The fixture's
    questions are written from templates, so a hidden one and a practice one
    can open with the same sixty characters; the whole text is its own.)"""
    from conftest import report_half_text
    fields = report_half_text(rows)
    assert not [t[:60] for _, _, t in fields if t in body]
    assert not [r["qid"] for r in rows if eb.half_of(r["qid"]) == "report" and r["qid"] in body]
    return len(fields)


def bank_rows(tree):
    return [r for rs in eb.load_bank(tree["judged"]["exam_root"]).values() for r in rs]


def make_dataset(client, tree, monkeypatch, count=12):
    """Propose, approve (spread by area) and generate — one request's reply
    two documents too short, so two are missing with their reason."""
    label_domains(tree["judged"]["exam_root"], TOPIC)

    def responder(req):
        if req.custom_id.startswith("gen:") and req.custom_id.rsplit(":", 1)[1] == "1":
            return json.dumps([{"title": "Too brief", "text": SHORT}] * 2)
        return llm.default_responder(req)
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(responder))
    pid = client.post("/api/proposals", json={"model": MODEL, "topic": TOPIC,
                                              "requested_by": "tester"}).json()["id"]
    assert llm_poller.tick() == 1
    assert client.post(f"/api/proposals/{pid}/approve",
                       json={"approver": "Omar"}).status_code == 200
    r = client.post(f"/api/proposals/{pid}/generate", json={"requester": "Omar", "count": count})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]
    assert llm_poller.tick() == 1
    assert client.get(f"/api/datasets/{did}").json()["status"] == "ready"
    return did


# ---------------------------------------------------------------------------
# the dataset reader
# ---------------------------------------------------------------------------

def test_the_dataset_reader_numbers_every_document_and_places_the_missing(svc, monkeypatch):
    client, tree = svc
    did = make_dataset(client, tree, monkeypatch)
    page = client.get(f"/api/datasets/{did}/items").json()
    lines = (prop.dataset_dir(did) / "items.jsonl").read_text().splitlines()
    docs = [e for e in page["entries"] if e["type"] == "doc"]
    gone = [e for e in page["entries"] if e["type"] == "missing"]
    assert page["kept"] == len(lines) == len(docs) == 10
    assert [e["n"] for e in docs] == list(range(1, 11))
    assert page["requested"] == 12 and page["missing"] == 2
    assert [e["why"] for e in gone] == ["too short (60 words)"] * 2
    # each document carries the focus label its request was for, and the
    # missing ones sit right after their request's documents
    pv = client.get(f"/api/datasets/{did}").json()["provenance"]
    assert page["labels_recorded"] and all(e["focus"] for e in docs)
    assert {e["focus"] for e in docs} <= set(pv["focus_labels"])
    reqs = [e["request"] for e in page["entries"]]
    assert reqs == sorted(reqs)                              # in request order
    asked = {q["k"]: q["focus"] for q in pv.get("requests") or []} or None
    assert all(e["focus"] for e in gone)
    if asked:
        assert all(asked[e["request"]] == e["focus"] for e in page["entries"])
    assert all(e["words"] == len(json.loads(lines[e["n"] - 1])["text"].split()) for e in docs)
    # a search narrows to the documents that hold it, and none of the missing
    word = docs[3]["title"].split()[0]
    found = client.get(f"/api/datasets/{did}/items", params={"q": word}).json()["entries"]
    assert found and all(e["type"] == "doc" and word.lower() in (e["title"] + e["text"]).lower()
                         for e in found)
    # paged, 50 at most a page
    assert client.get(f"/api/datasets/{did}/items", params={"limit": 3}).json()["entries"] \
        == page["entries"][:3]
    assert client.get("/api/datasets/999/items").status_code == 404


def test_the_labels_are_written_beside_the_documents_and_rebuilt_without_them(svc, monkeypatch):
    client, tree = svc
    did = make_dataset(client, tree, monkeypatch)
    side = prop.dataset_dir(did) / "items.meta.json"
    written = json.loads(side.read_text())
    assert len(written) == 10 and all("focus" in x and "request" in x for x in written)
    before = client.get(f"/api/datasets/{did}/items").json()
    # a dataset made before 11g has no such file: its provenance rebuilds it
    side.unlink()
    after = client.get(f"/api/datasets/{did}/items").json()
    assert after["labels_recorded"] and after["entries"] == before["entries"]
    # the download is unchanged: one document per line, nothing added
    first = json.loads((prop.dataset_dir(did) / "items.jsonl").read_text().splitlines()[0])
    assert set(first) == {"title", "text"}


def test_a_dataset_from_before_11a_says_its_reasons_were_not_recorded(svc):
    client, tree = svc
    pid = db.proposal_create(MODEL, "exam_physics_astronomy", "Physics & Astronomy", "tester", {})
    did = db.dataset_create(pid, "doc", 20, "Omar", "")
    d = prop.dataset_dir(did)
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.jsonl").write_text("".join(json.dumps({"title": f"Doc {i}", "text": "word " * 200})
                                           + "\n" for i in range(18)))
    db.dataset_update(did, status="ready", provenance=json.dumps(
        {"count_requested": 20, "items": {"generated": 18, "dropped": 0, "kept": 18}}))
    page = client.get(f"/api/datasets/{did}/items").json()
    gone = [e for e in page["entries"] if e["type"] == "missing"]
    assert page["kept"] == 18 and page["missing"] == 2 and not page["labels_recorded"]
    assert [e["why"] for e in gone] == ["reasons not recorded (made before 11a)"] * 2


# ---------------------------------------------------------------------------
# the bank: the practice half, and a count
# ---------------------------------------------------------------------------

def test_the_bank_reader_gets_the_practice_half_and_only_a_count_of_the_hidden(svc):
    client, tree = svc
    rows = bank_rows(tree)
    mine = [r for r in rows if r["topic"] == TOPIC]
    r = client.get("/api/exam/bank", params={"topic": TOPIC, "half": "diagnose"})
    assert r.status_code == 200
    j = r.json()
    assert j["topic"] == TOPIC and j["half"] == "diagnose"
    assert {q["qid"] for q in j["questions"]} == {x["qid"] for x in mine
                                                   if eb.half_of(x["qid"]) == "diagnose"}
    assert j["report_count"] == sum(eb.half_of(x["qid"]) == "report" for x in mine) > 0
    assert no_hidden(r.text, rows) > 0
    # the hidden half cannot be asked for, and a topic has to exist
    assert client.get("/api/exam/bank", params={"topic": TOPIC, "half": "report"}).status_code == 422
    assert client.get("/api/exam/bank", params={"topic": "Nope", "half": "diagnose"}).status_code == 404
    # without the parameter the endpoint is what it was
    assert len(client.get("/api/exam/bank", params={"topic": TOPIC}).json()) == len(mine)


# ---------------------------------------------------------------------------
# rubric and criteria
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["arts", "architecture_built_environment",
                                  "biology_life_sciences"])
def test_the_criteria_reader_is_the_judges_own_reading(svc, name):
    """Arts carries its flag as critical_error_flag, architecture as
    critical_error, biology as critical_flags: all three read the same."""
    import judge
    client, _ = svc
    r = client.get(f"/api/exam/rubrics/{name}/read", params={"kind": "criteria"})
    assert r.status_code == 200, r.text
    j = r.json()
    raw = judge.rubric_path(name, ".criteria.json").read_bytes()
    want = judge.normalise_criteria(json.loads(raw))
    assert [(c["id"], c["name"]) for c in j["criteria"]] == \
        [(c["id"], c["name"]) for c in want["criteria"]]
    assert len(j["criteria"]) == 20
    assert [f["id"] for f in j["flags"]] == [f["id"] for f in want["flags"]] and j["flags"]
    assert all(f["effect_words"] for f in j["flags"])
    assert j["sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads(j["raw"]) == json.loads(raw)


def test_the_rubric_reader_gives_its_text_version_and_sha(svc):
    import judge
    client, _ = svc
    j = client.get("/api/exam/rubrics/arts/read").json()
    text = judge.rubric_path("arts").read_text(encoding="utf-8")
    assert j["text"] == text and j["file"] == "arts.md"
    assert j["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert client.get("/api/exam/rubrics/nope/read").status_code == 404
    assert client.get("/api/exam/rubrics/arts/read", params={"kind": "other"}).status_code == 404


# ---------------------------------------------------------------------------
# the log, and how a model was graded
# ---------------------------------------------------------------------------

def test_the_log_reader_numbers_lines_and_withholds_one_that_quotes_a_hidden_question(svc):
    client, tree = svc
    rows = bank_rows(tree)
    hidden = next(r for r in rows if eb.half_of(r["qid"]) == "report")
    sid = db.add(MODEL, "auto", "judged", "omar", "")
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"step {i}" for i in range(2500)]
    lines[2400] = f"graded {hidden['qid']} score 3"
    lines[2450] = "prompt was: " + hidden["prompt"]
    (config.LOGS_DIR / f"service_{sid}_{MODEL.replace('/', '__')}.log").write_text("\n".join(lines))
    r = client.get(f"/api/runs/{sid}/lines", params={"tail": 200})
    j = r.json()
    assert j["total"] == 2500 and j["first"] == 2301 and len(j["lines"]) == 200
    assert j["withheld"] == 2 and j["lines"][2400 - 2300] == "[line withheld — it quotes a hidden question]"
    assert_no_report_half_text(r.text, rows)
    # at most 2,000 lines, however many are asked for
    assert len(client.get(f"/api/runs/{sid}/lines", params={"tail": 9999}).json()["lines"]) == 2000
    assert client.get("/api/runs/9999/lines").status_code == 404


def test_how_a_model_was_graded_never_carries_an_item_or_a_qid(svc):
    client, tree = svc
    rows = bank_rows(tree)
    r = client.get("/api/judge/provenance", params={"model": MODEL})
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == MODEL and j["tasks"]
    assert all("items" not in t for t in j["tasks"].values())
    assert not any(x["qid"] in r.text for x in rows)          # no qid of either half
    assert_no_report_half_text(r.text, rows)
    assert client.get("/api/judge/provenance", params={"model": "org/none"}).status_code == 404
