"""12g.2 in the service: Everyday tasks join Improve.

The bank is split as the exam is — every question's qid and half by
diagnose.split_of with the exam's salt, nothing by hand. The hidden half
scores the model and is never shown; the practice half is shown and is the
only half Improve reads. A group joins Improve only when its hidden half has
EVERYDAY_MIN_HIDDEN questions. Its training data is chat examples, each kept
only if its reply passes its own checks, and gated against the whole bank."""

from __future__ import annotations

import json

import pytest

import diagnose as dx
import everyday as ev
import exam_build as eb
import report_lm_eval as report
from conftest import fresh, make_service
from service import config, contamination as ct, llm, llm_poller
from service import proposals as prop
from test_improve_12g1 import mentions, standard_names

MODEL = "fx/skewed-360m"           # the fixture's: every other everyday answer is "not sure"
GROUP = "instructions"

# what the split puts in each half — said in the PR, and pinned here. 12a.5:
# the short summaries are "shorten" (same ids, same halves), "summarising" is
# the 45 long ones, and Honesty's ten new questions take it from 19 hidden to 24
SPLIT = {"understanding": (27, 18), "writing": (24, 24), "shorten": (26, 37),
         "summarising": (26, 19), "transform": (25, 21), "quick_maths": (24, 21),
         "instructions": (24, 21), "honesty": (24, 27)}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, manifest = make_service(tmp_path, monkeypatch)
    yield client, appmod, manifest
    client.__exit__(None, None, None)


def hidden_questions():
    return [q for q in ev.load_bank() if ev.half(q) == ev.HIDDEN]


def practice_questions():
    return [q for q in ev.load_bank() if ev.half(q) == ev.PRACTICE]


def leaks(text: str) -> list[str]:
    """every hidden question whose text — whole, or 13 words of it — is in `text`"""
    grams = {w for w in ct.windows(ct.normalize(text))}
    flat = " ".join(ct.normalize(text))
    out = []
    for q in hidden_questions():
        words = ct.normalize(q["prompt"])
        if any(w in grams for w in ct.windows(words)) or (
                len(words) >= 6 and " ".join(words) in flat):
            out.append(q["id"])
    return out


def _asked(model_dir, questions, answer):
    task = model_dir / "everyday_0shot" / "pretrained__x"
    task.mkdir(parents=True, exist_ok=True)
    with open(task / "samples_everyday_2026-09-25T00-00-00.jsonl", "w", encoding="utf-8") as fh:
        for i, q in enumerate(questions):
            fh.write(json.dumps({"doc_id": i, "doc": q, "resps": [[answer(q)]],
                                 "filtered_resps": [answer(q)],
                                 "arguments": [["", {"max_gen_toks": 512}]]}) + "\n")


# ---------------------------------------------------------------------------
# 5. the split
# ---------------------------------------------------------------------------

def test_every_everyday_question_is_split_by_the_exams_function_and_salt():
    bank = ev.load_bank()
    assert ev.SPLIT == dx.SPLIT_SALT
    for q in bank:
        assert ev.qid(q) == eb.qid_of(q["prompt"])
        assert ev.half(q) == dx.split_of(eb.qid_of(q["prompt"])) in (ev.HIDDEN, ev.PRACTICE)
    counts = ev.split_counts()
    assert {g: (c["hidden"], c["practice"]) for g, c in counts.items()} == SPLIT
    assert sum(c["hidden"] for c in counts.values()) == 200
    # 12a.5: no group is under the line now; Honesty was, by one
    assert [g for g, c in counts.items() if c["hidden"] < 20] == []


def test_no_hidden_question_reaches_the_page(tmp_path):
    mdir = tmp_path / "org__m"
    _asked(mdir, ev.load_bank(), lambda q: q["reference"])
    ev.write(mdir, ev.mark(mdir))
    e = report.load_everyday(tmp_path)
    hidden = {q["id"] for q in hidden_questions()}
    # the questions list and the answers are the practice half's, and say how
    # many hidden ones there are
    assert {q["id"] for q in e["questions"]} == {q["id"] for q in practice_questions()}
    assert not hidden & {it["id"] for it in e["models"]["org/m"]["items"]}
    assert e["hidden"] == {g: h for g, (h, _) in SPLIT.items()}
    assert e["practice"] == {g: p for g, (_, p) in SPLIT.items()}
    assert leaks(json.dumps(e)) == []
    # the published score is the hidden half's
    m = e["models"]["org/m"]
    assert m["total"] == 200 and {g: x["total"] for g, x in m["groups"].items()} == \
        {g: h for g, (h, _) in SPLIT.items()}
    assert {g: x["total"] for g, x in m["practice"].items()} == \
        {g: p for g, (_, p) in SPLIT.items()}


def test_results_from_before_the_split_are_kept_labelled_and_never_mixed(tmp_path):
    mdir = tmp_path / "org__m"
    bank = ev.load_bank()
    mdir.mkdir(parents=True)
    # marked before 12g.2: every question counted, the wording's hash alone
    (mdir / "everyday.json").write_text(json.dumps({
        "model": "org/m", "passed": 300, "total": 333, "marked_at": 1.0,
        "version": {"hash": ev.wording_hash(bank), "date": ev.WORDING_DATE},
        "items": [{"id": q["id"], "group": q["group"], "pass": True, "reason": ""}
                  for q in bank]}), encoding="utf-8")
    e = report.load_everyday(tmp_path)
    assert "org/m" not in e["models"]                              # in no score…
    assert e["earlier"]["org/m"]["label"] == "all questions, before the split"   # …but kept
    assert (e["earlier"]["org/m"]["passed"], e["earlier"]["org/m"]["total"]) == (300, 333)


# ---------------------------------------------------------------------------
# 6. a group opens only when it's big enough; 7. its proposal reads practice only
# ---------------------------------------------------------------------------

def test_a_group_under_the_line_is_refused_and_says_how_many_more(svc, monkeypatch):
    client, _, _ = svc
    # 12a.5: Honesty has 24 hidden questions now, so the line moves to find one
    monkeypatch.setattr(config, "EVERYDAY_MIN_HIDDEN", 25)
    r = client.post("/api/proposals", json={"model": MODEL, "everyday": "honesty",
                                            "requested_by": "m"})
    assert r.status_code == 409
    assert r.json()["detail"] == ("Honesty has 24 hidden questions and Improve takes a group at "
                                  "25: it needs 1 more")
    # the line is one setting: at 24, honesty is taken
    monkeypatch.setattr(config, "EVERYDAY_MIN_HIDDEN", 24)
    r = client.post("/api/proposals", json={"model": MODEL, "everyday": "honesty",
                                            "requested_by": "m"})
    assert r.status_code == 200, r.text
    assert client.post("/api/proposals", json={"model": MODEL, "everyday": "nope",
                                               "requested_by": "m"}).status_code == 422


def test_an_everyday_group_goes_propose_approve_generate_on_practice_only(svc):
    client, appmod, tree = svc
    fake = llm.client()
    r = client.post("/api/proposals", json={"model": MODEL, "everyday": GROUP,
                                            "requested_by": "masein"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert r.json()["task"] == "everyday:instructions"
    assert llm_poller.tick() == 1
    p = client.get(f"/api/proposals/{pid}").json()
    assert p["status"] == "proposed" and p["category"] == "Instructions"
    ev_ = p["evidence"]
    practice = {q["id"]: q for q in practice_questions() if q["group"] == GROUP}
    # what it read: this group's practice questions the model failed, each with why
    assert ev_["failed"] and all(f["id"] in practice for f in ev_["failed"])
    assert all(f["reason"] for f in ev_["failed"])
    assert ev_["hidden"]["total"] == SPLIT[GROUP][0]
    assert ev_["skills"] == prop.everyday_focus(ev_["failed"])
    # the request: those practice requests, and no hidden question at all
    req = next(q for q in fake.recorded() if q["custom_id"] == f"proposal:{pid}")
    body = req["system"] + "\n" + req["user"]
    assert all(f["prompt"] in body for f in ev_["failed"])
    assert leaks(body) == []
    assert all(dx.split_of(q) == "diagnose" for q in req["meta"]["qids"])
    # approve (the plan: its failed skills), then generate — chat, by default
    assert client.post(f"/api/proposals/{pid}/approve",
                       json={"approver": "masein", "edited_text": ""}).status_code == 200
    focus = client.get(f"/api/proposals/{pid}/focus?count=10").json()
    assert focus["frozen"] and focus["mode"] == "skill" and focus["labels"] == ev_["skills"]
    assert client.post(f"/api/proposals/{pid}/generate",
                       json={"requester": "m", "count": 10, "fmt": "doc"}).status_code == 422
    r = client.post(f"/api/proposals/{pid}/generate", json={"requester": "m", "count": 10})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]
    gen = [q for q in fake.recorded() if q["custom_id"].startswith(f"gen:{did}:")]
    assert gen and all(q["meta"]["format"] == "chat" for q in gen)
    for q in gen:
        body = q["system"] + "\n" + q["user"]
        assert "Everyday group: Instructions" in body and "Focus: " in body
        assert leaks(body) == [], q["custom_id"]
    assert llm_poller.tick() == 1
    d = client.get(f"/api/datasets/{did}").json()
    assert d["status"] == "ready", d["error"]
    items = [json.loads(x) for x in
             client.get(f"/api/datasets/{did}/items.jsonl").text.splitlines()]
    assert len(items) == 10
    for it in items:
        assert set(it) == {"user", "assistant", "checks"}
        assert all(ev.run_check(c, it["assistant"], it["user"])[0] is True for c in it["checks"])
    assert d["provenance"]["format"] == "chat"
    assert d["provenance"]["gate"]["everyday_questions"] == 388
    # 12g.1's rule holds here too: no Standard benchmark in any of it
    fresh(appmod)
    names = standard_names(client.get("/api/results").json())
    for q in fake.recorded():
        if q["custom_id"].split(":")[0] in ("proposal", "gen"):
            body = q["system"] + "\n" + q["user"] + "\n" + json.dumps(q["meta"])
            assert mentions(body, names) == [], (q["custom_id"], mentions(body, names))
    assert mentions(json.dumps(items), names) == []


def test_chat_is_an_everyday_groups_format_and_documents_an_exam_topics(svc):
    client, _, _ = svc
    r = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": "Economics",
                                            "requested_by": "m"})
    pid = r.json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "m", "edited_text": ""})
    r = client.post(f"/api/proposals/{pid}/generate",
                    json={"requester": "m", "count": 4, "fmt": "chat"})
    assert r.status_code == 422 and "Everyday groups" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 7. every example is marked by its own checks; the gate covers the whole bank
# ---------------------------------------------------------------------------

def test_a_chat_example_that_fails_its_own_checks_is_dropped_with_its_reason():
    reply = json.dumps([
        {"user": "wats 12 plus 30 for the bus fares", "assistant": "That comes to 42.",
         "checks": [{"type": "number", "value": 42, "tolerance": 0}]},
        {"user": "wats 12 plus 31 for the bus fares", "assistant": "That comes to 42.",
         "checks": [{"type": "number", "value": 43, "tolerance": 0}]},
        {"user": "is this polite", "assistant": "Yes.",
         "checks": [{"type": "judge", "rubric": "polite"}]},
        {"user": "say hi", "assistant": "hi", "checks": []},
        {"user": "say hi", "assistant": "hi", "checks": [{"type": "sparkle"}]},
    ])
    kept, why = prop.read_reply(reply, "chat", 5)
    assert [k["assistant"] for k in kept] == ["That comes to 42."] and len(kept) == 1
    assert why == ["failed its own checks: didn't say 43",
                   "a judge check, which a script cannot mark", "no checks",
                   "a check it cannot use: unknown check type 'sparkle'"]


def test_a_failing_example_is_counted_missing_on_the_dataset(svc, monkeypatch):
    client, _, _ = svc

    def one_bad(req):
        if not req.custom_id.startswith("gen:"):
            return llm.default_responder(req)
        items = json.loads(llm.default_responder(req))
        items[0]["assistant"] = "no idea"                  # fails its own check
        return json.dumps(items)
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(one_bad))
    pid = client.post("/api/proposals", json={"model": MODEL, "everyday": GROUP,
                                              "requested_by": "m"}).json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "m", "edited_text": ""})
    did = client.post(f"/api/proposals/{pid}/generate",
                      json={"requester": "m", "count": 10}).json()["dataset_id"]
    llm_poller.tick()
    d = client.get(f"/api/datasets/{did}").json()
    it = d["provenance"]["items"]
    assert it["kept"] + len(it["missing"]) == it["requested"] == 10
    assert it["missing"] and all(m["why"].startswith("failed its own checks: ")
                                 for m in it["missing"])


def test_the_gate_drops_an_example_copying_any_everyday_question_either_half(tmp_path):
    ix = ct.index(tmp_path / "none", None)
    long_ = [q for q in ev.load_bank() if len(ct.normalize(q["prompt"])) >= 20]
    hid = next(q for q in long_ if ev.half(q) == ev.HIDDEN)
    pra = next(q for q in long_ if ev.half(q) == ev.PRACTICE)
    short = next(q for q in ev.load_bank() if len(ct.normalize(q["prompt"])) < 13)
    thirteen = lambda q: " ".join(q["prompt"].split()[:13])            # noqa: E731
    items = [{"user": "ok so " + thirteen(hid), "assistant": "Sure."},
             {"user": "hey " + thirteen(pra) + " thanks", "assistant": "Sure thing."},
             {"user": short["prompt"], "assistant": "Here you go."},
             {"user": "whats a good name for a pet snail", "assistant": "Gary."}]
    gate = ct.check(items, ix)
    assert [d["index"] for d in gate["dropped"]] == [0, 1, 2]
    assert all(d["source"] == "everyday" for d in gate["dropped"])
    assert [k["assistant"] for k in gate["kept"]] == ["Gary."]
    assert gate["report"]["everyday_questions"] == 388
