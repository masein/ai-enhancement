"""12i.2: the question builder, through the service with a fake writer,
checker and judge (service.llm's FakeBatches) and a fake OpenRouter for the
embeddings. Nothing leaves the machine and no model runs."""

from __future__ import annotations

import json

import pytest

from conftest import fresh, make_service
from fake_openrouter import FakeOpenRouter
from service import builder, config, db, llm, llm_poller

BY = "masein"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    # the checker has no environment of its own; the AI models page sets it
    monkeypatch.setattr(config, "CHECKER_PROVIDER", "fake", raising=False)
    monkeypatch.setattr(config, "CHECKER_MODEL", "fake-checker", raising=False)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    llm.reset()
    yield client, appmod, tree
    client.__exit__(None, None, None)


def respond(monkeypatch, fn):
    """answer some requests by hand, the rest as the fake does"""
    def r(req):
        got = fn(req)
        return llm.default_responder(req) if got is None else got
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(r))


def drain() -> None:
    for _ in range(20):
        if not db.batches_pending():
            return
        llm_poller.tick()
    raise AssertionError("batches still pending")


def create(client, **spec) -> dict:
    body = {"count": 20, "dedup": True, "by": BY, **spec}
    r = client.post("/api/builder", json=body)
    assert r.status_code == 200, r.text
    drain()
    return client.get(f"/api/builder/{r.json()['id']}").json()


def review_all(client, d: dict, verdict="accept", only=None) -> dict:
    for it in d["items"]:
        if it["auto"] or it["verdict"] or (only and not only(it)):
            continue
        r = client.post(f"/api/builder/{d['id']}/review",
                        json={"n": it["n"], "verdict": verdict, "by": BY})
        assert r.status_code == 200, r.text
    return client.get(f"/api/builder/{d['id']}").json()


def rest(client, d: dict) -> dict:
    r = client.post(f"/api/builder/{d['id']}/rest", json={"by": BY})
    assert r.status_code == 200, r.text
    drain()
    return client.get(f"/api/builder/{d['id']}").json()


def writer_prompts() -> list[str]:
    fake = llm.FakeBatches("", config.BENCH_ROOT)
    return [r["user"] for r in fake.recorded() if r["custom_id"].startswith("qbw:")]


# ---------------------------------------------------------------------------
# step 1
# ---------------------------------------------------------------------------

def test_step_one_offers_topics_groups_and_the_instructions_with_their_output_locked(svc):
    client, _, _ = svc
    page = client.get("/api/builder").json()
    econ = next(t for t in page["topics"] if t["name"] == "Economics")
    assert econ["subtopics"][:2] == ["micro and macro fundamentals", "incentives"]
    assert [g["id"] for g in page["groups"]][:3] == ["understanding", "writing", "shorten"]
    for kind in ("knowledge", "everyday"):
        p = page["prompts"][kind]
        assert p["locked"].startswith("## Output") and "## Output" not in p["editable"]
    assert page["prompts"]["everyday"]["path"] == "docs/prompts/phase-12i/everyday-question-prompt.md"
    assert page["writer"]["label"] == "fake fake-exam" and page["checker_blocked"] == ""
    assert page["drafts"] == []


def test_the_output_section_is_always_the_default_whatever_was_typed(svc):
    """masein edits the instructions; an output section typed into them is
    replaced by the locked one, so the result can always be read"""
    client, _, _ = svc
    default = builder.default_prompt("knowledge")
    edited = default["editable"].replace("a curious adult", "a curious teenager") \
        + "\n\n## Output\nWrite prose, not JSON."
    d = create(client, kind="knowledge", topic="Economics", level="general public",
               prompt=edited)
    assert d["prompt"]["edited"] is True and "a curious teenager" in d["prompt"]["editable"]
    sent = writer_prompts()[0]
    assert "a curious teenager" in sent and "Write prose, not JSON." not in sent
    assert sent.rstrip().endswith(default["locked"].rstrip())
    assert "TOPIC: Economics" in sent and "COUNT: 10" in sent and "{topic}" not in sent
    # the default is untouched: Reset to default restores it
    assert builder.default_prompt("knowledge") == default


def test_the_count_is_free(svc):
    client, _, _ = svc
    d = create(client, kind="everyday", group="quick_maths", count=7)
    assert d["spec"]["count"] == 7 and d["progress"]["written"] == 7
    assert d["status"] == "review" and d["progress"]["line"] == "7 of 7 written"
    assert client.post("/api/builder", json={"kind": "everyday", "group": "quick_maths",
                                             "count": 0, "by": BY}).status_code == 422
    est = client.post("/api/builder/estimate", json={"kind": "everyday", "count": 60}).json()
    assert est["line"] == "free: every step runs on the local model"


# ---------------------------------------------------------------------------
# Everyday
# ---------------------------------------------------------------------------

def test_an_everyday_question_whose_reference_fails_its_own_checks_is_set_aside(svc, monkeypatch):
    client, _, _ = svc
    bad = {"group": "quick_maths", "skill": "tip", "difficulty": 2,
           "prompt": "dinner was $80 and we tip 10% how much in all",
           "reference": "$90 in all.", "checks": [{"type": "number", "value": 88, "tolerance": 0}],
           "notes": ""}
    paste = {"group": "shorten", "skill": "tldr", "difficulty": 1,
             "prompt": "tldr pls: the bins go out on thursday not friday this week",
             "reference": "Bins on Thursday.", "checks": [{"type": "contains_any",
                                                          "values": ["thursday"]}], "notes": ""}

    def writer(req):
        if req.custom_id.startswith("qbw:"):
            good = json.loads(llm.default_responder(req).splitlines()[0])
            return "\n".join(json.dumps(x) for x in (good, bad, paste))
    respond(monkeypatch, writer)
    d = create(client, kind="everyday", group="quick_maths", count=10)
    auto = [it["auto"] for it in d["items"]]
    assert auto[0] == ""
    assert auto[1] == "its reference fails its own checks: didn't say 88"
    # the paste-back rule is the Shorten group's: here it is read as quick maths
    d2 = create(client, kind="everyday", group="shorten", count=10)
    assert d2["items"][2]["auto"] == "pasting the message back passes its checks"
    # set aside before review: it cannot be reviewed, and is never published
    r = client.post(f"/api/builder/{d['id']}/review", json={"n": 2, "verdict": "accept", "by": BY})
    assert r.status_code == 422 and "set aside before review" in r.json()["detail"]


def test_an_everyday_question_the_checker_answers_wrongly_is_flagged(svc, monkeypatch):
    client, _, _ = svc

    def checker(req):
        if req.custom_id.startswith("qbc:") and req.custom_id.endswith(":12"):
            return "I think it's 3."
    respond(monkeypatch, checker)
    d = create(client, kind="everyday", group="quick_maths", count=20)
    d = rest(client, review_all(client, d))
    assert d["status"] == "review" and d["progress"]["written"] == 20
    flagged = [it for it in d["items"] if it["flags"]]
    assert [it["n"] for it in flagged] == [12]
    text = flagged[0]["flags"][0]["text"]
    assert text.startswith("the checker's answer failed: ") and "not 3" not in text[:30]
    assert d["progress"]["line"] == "20 of 20 written · 1 flagged"
    # the checker never saw the checks or the reference: only the question
    fake = llm.FakeBatches("", config.BENCH_ROOT)
    sent = next(r for r in fake.recorded() if r["custom_id"].endswith(":12")
                and r["custom_id"].startswith("qbc:"))
    it12 = next(it for it in d["items"] if it["n"] == 12)
    assert sent["user"] == it12["q"]["prompt"] and sent["system"] == ""


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------

def test_a_knowledge_question_the_judge_marks_two_of_four_is_flagged(svc, monkeypatch):
    client, _, _ = svc
    import judge
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    llm.reset()
    spec = judge.rubric_for("exam_economics").criteria
    half = {"criteria": {c: 0.5 for c in judge.criteria_ids(spec)},
            "flags": {f: False for f in judge.flag_ids(spec)}, "justification": "half of it"}
    assert judge.fold(half["criteria"], half["flags"], spec) == 2

    def two(req):
        if req.custom_id.startswith("qbj:") and req.custom_id.endswith(":14"):
            return json.dumps(half)
        if req.custom_id.startswith("qbc:") and req.custom_id.endswith(":15"):
            return json.dumps({"answer": req.meta["ref"], "concerns": ["time-sensitive"],
                               "why": "rates change every year"})
    respond(monkeypatch, two)
    d = create(client, kind="knowledge", topic="Economics", level="specialist", count=20)
    d = rest(client, review_all(client, d))
    by_n = {it["n"]: it for it in d["items"]}
    f14 = by_n[14]["flags"]
    assert by_n[14]["mark"] == 2 and len(f14) == 1
    assert f14[0]["text"].startswith("the checker answered “")
    assert "(2/4); the criteria expect: " + "; ".join(by_n[14]["q"]["criteria"]) in f14[0]["text"]
    assert [f["text"] for f in by_n[15]["flags"]] == \
        ["the checker finds it time-sensitive: rates change every year"]
    assert all(not it["flags"] for n, it in by_n.items() if n not in (14, 15))
    # the checker answered without the reference
    fake = llm.FakeBatches("", config.BENCH_ROOT)
    sent = next(r for r in fake.recorded() if r["custom_id"] == f"qbc:{d['id']}:14")
    assert sent["user"] == by_n[14]["q"]["question"]
    assert by_n[14]["q"]["reference"] not in sent["user"] + sent["system"]


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

def _near_copy_writer(monkeypatch, text: str):
    def w(req):
        if req.custom_id.startswith("qbw:") and req.meta.get("start") == 10:
            rows = [json.loads(x) for x in llm.default_responder(req).splitlines()]
            rows[0]["prompt"] = text
            rows[0]["reference"] = "3 pm."
            rows[0]["checks"] = [{"type": "contains_any", "values": ["3 pm", "3pm", "15:00"]}]
            return "\n".join(json.dumps(x) for x in rows)
    respond(monkeypatch, w)


def test_a_near_duplicate_of_a_bank_question_is_flagged_and_keep_old_drops_it(svc, monkeypatch):
    client, _, _ = svc
    import everyday as ev
    bank = next(q for q in ev.load_bank() if len(q["prompt"].split()) > 16)
    _near_copy_writer(monkeypatch, bank["prompt"] + " thx")      # the same 13 words in a row
    d = create(client, kind="everyday", group="quick_maths", count=20)
    d = rest(client, review_all(client, d))
    it = next(it for it in d["items"] if it["n"] == 11)
    dup = next(f for f in it["flags"] if f["kind"] == "dup")
    assert dup["text"] == f"looks like {bank['id']}" and dup["how"] == "13 words in a row"
    assert dup["other"]["text"] == bank["prompt"] and d["dedup_how"] == "13-gram"
    # "keep new" can't retire a bank question
    r = client.post(f"/api/builder/{d['id']}/duplicate", json={"n": 11, "keep": "new", "by": BY})
    assert r.status_code == 422
    d = client.post(f"/api/builder/{d['id']}/duplicate",
                    json={"n": 11, "keep": "old", "by": BY}).json()
    it = next(it for it in d["items"] if it["n"] == 11)
    assert it["verdict"] == "reject" and it["dup"] == "old"
    assert 11 not in [x["n"] for x in builder.publishable(builder.get(d["id"]))]


def test_embeddings_find_a_reworded_duplicate_and_the_toggle_turns_it_off(svc, monkeypatch):
    client, _, _ = svc
    fake = FakeOpenRouter.install(monkeypatch)
    import everyday as ev
    bank = next(q for q in ev.load_bank() if 10 < len(q["prompt"].split()) < 30)
    words = bank["prompt"].split()
    reworded = " ".join(words[:6] + ["uh"] + words[6:12] + ["pls"] + words[12:])
    _near_copy_writer(monkeypatch, reworded)
    d = create(client, kind="everyday", group="quick_maths", count=20)
    d = rest(client, review_all(client, d))
    dup = next(f for it in d["items"] if it["n"] == 11 for f in it["flags"] if f["kind"] == "dup")
    assert dup["how"].startswith("cosine ") and float(dup["how"].split()[1]) >= 0.9
    assert d["dedup_how"] == "13-gram and embeddings" and fake.embedded
    assert fake.embedded[0]["model"] == config.OPENROUTER_EMBED_MODEL
    # the bank is embedded once: the second draft asks only for its own
    n_first = sum(len(e["input"]) for e in fake.embedded)
    # toggle off: nothing is flagged as a duplicate, and nothing is embedded
    fake.embedded.clear()
    d2 = create(client, kind="everyday", group="quick_maths", count=20, dedup=False)
    d2 = rest(client, review_all(client, d2))
    assert not [f for it in d2["items"] for f in it["flags"] if f["kind"] == "dup"]
    assert fake.embedded == [] and n_first > 20


# ---------------------------------------------------------------------------
# the first ten teach the rest; cancel; resume
# ---------------------------------------------------------------------------

def test_make_the_rest_needs_five_reviewed_and_carries_their_reasons(svc):
    client, _, _ = svc
    d = create(client, kind="everyday", group="quick_maths", count=20)
    assert d["can_rest"] == "review 5 more of the first ten first"
    assert client.post(f"/api/builder/{d['id']}/rest", json={"by": BY}).status_code == 422
    for n, (v, why) in enumerate([("reject", "too easy"), ("reject", "trivia"),
                                  ("accept", ""), ("accept", ""), ("reject", "unclear")], 1):
        client.post(f"/api/builder/{d['id']}/review",
                    json={"n": n, "verdict": v, "reason": why, "by": BY})
    edited = {"prompt": d["items"][5]["q"]["prompt"] + " thanks"}
    r = client.post(f"/api/builder/{d['id']}/review",
                    json={"n": 6, "verdict": "edit", "edited": edited, "by": BY})
    assert r.status_code == 200 and r.json()["can_rest"] == ""
    # an edit that breaks its own checks is refused
    bad = client.post(f"/api/builder/{d['id']}/review",
                      json={"n": 7, "verdict": "edit", "by": BY,
                            "edited": {"reference": "no idea"}})
    assert bad.status_code == 422 and "its reference fails its own checks" in bad.json()["detail"]
    d = rest(client, r.json())
    later = writer_prompts()[1]
    assert "Avoid questions like these:\n- too easy: “" in later
    assert "- trivia: “" in later and "- unclear: “" in later
    assert "Do more of these:" in later and "as it was edited" in later
    # the output section still comes last, locked
    assert later.rstrip().endswith(builder.default_prompt("everyday")["locked"].rstrip())
    assert d["stage"] == "rest" and d["progress"]["written"] == 20


def test_a_cancelled_batch_keeps_what_is_written_and_resumes(svc):
    client, _, _ = svc
    r = client.post("/api/builder", json={"kind": "everyday", "group": "quick_maths",
                                          "count": 40, "by": BY})
    did = r.json()["id"]
    drain()
    review_all(client, client.get(f"/api/builder/{did}").json())
    client.post(f"/api/builder/{did}/rest", json={"by": BY})
    llm_poller.tick()                                  # ten more land, the next ten are asked
    d = client.post(f"/api/builder/{did}/cancel", json={"by": BY}).json()
    assert d["status"] == "cancelled"
    drain()                                            # the ten in flight still land
    d = client.get(f"/api/builder/{did}").json()
    assert d["status"] == "cancelled" and d["progress"]["written"] == 30
    assert d["can_publish"] == "the questions are still being written or checked"
    d = client.post(f"/api/builder/{did}/resume", json={"by": BY}).json()
    drain()
    d = client.get(f"/api/builder/{did}").json()
    assert d["status"] == "review" and d["progress"]["written"] == 40


# ---------------------------------------------------------------------------
# publish: a new bank version, with provenance on every question
# ---------------------------------------------------------------------------

def _to_publish(client, d: dict) -> dict:
    d = review_all(client, d, only=lambda it: it["flags"] or it["sample"])
    assert d["can_publish"] == "", d["can_publish"]
    r = client.post(f"/api/builder/{d['id']}/publish", json={"by": BY})
    assert r.status_code == 200, r.text
    return r.json()


def test_publishing_everyday_questions_makes_a_new_bank_version(svc):
    client, appmod, _ = svc
    import everyday as ev
    before = ev.version()["hash"]
    d = create(client, kind="everyday", group="quick_maths", count=20)
    for n in (1, 2):
        client.post(f"/api/builder/{d['id']}/review",
                    json={"n": n, "verdict": "reject", "reason": "too easy", "by": BY})
    d = rest(client, review_all(client, client.get(f"/api/builder/{d['id']}").json()))
    sampled = [it for it in d["items"] if it["sample"]]
    assert len(sampled) == 1                            # a tenth of the ten unreviewed
    out = _to_publish(client, d)
    pub = out["published"]
    assert pub["added"] == 18 and pub["version"] != before and pub["before"] == before
    assert ev.version()["hash"] == pub["version"]
    rows = [json.loads(x) for x in ev.built_path().read_text().splitlines()]
    assert len(rows) == 18
    for row in rows:
        assert row["written_by"] == "fake fake-exam (fake/fake-exam)"
        assert row["checked_by"] == "fake fake-checker (fake/fake-checker)"
        assert row["approved_by"] == BY and row["batch_id"] == d["id"]
        assert len(row["prompt_sha256"]) == 64 and row["group"] == "quick_maths"
    # they are in the bank the runs read, split into its two halves as usual
    bank = {q["id"]: q for q in ev.load_bank()}
    assert all(r["id"] in bank for r in rows)
    assert {ev.half(bank[r["id"]]) for r in rows} == {ev.HIDDEN, ev.PRACTICE}
    assert out["draft"]["status"] == "published"
    assert client.post(f"/api/builder/{d['id']}/publish", json={"by": BY}).status_code == 422
    # the page's data carries the new version
    fresh(appmod)
    assert client.get("/api/results").json()["everyday"]["version"]["hash"] == pub["version"]


def test_publishing_knowledge_questions_makes_a_new_bank_version(svc):
    client, _, _ = svc
    import exam_build as eb
    before = (eb.current_fingerprints(config.JUDGED_TASKS_DIR) or {}).get("exam_economics")
    n_bank = len(eb.load_bank(config.EXAM_DIR)["Economics"])
    d = create(client, kind="knowledge", topic="Economics", level="general public", count=12)
    d = rest(client, review_all(client, d))
    out = _to_publish(client, d)
    assert out["published"]["added"] == 12 and out["published"]["build"]["built"] is True
    rows = eb.load_bank(config.EXAM_DIR)["Economics"]
    assert len(rows) == n_bank + 12
    new = [r for r in rows if r.get("batch_id") == d["id"]]
    for r in new:
        assert r["source"] == "question builder" and r["approved_by"] == BY
        assert r["written_by"] == "fake fake-exam (fake/fake-exam)"
        assert r["checked_by"].startswith("fake fake-checker")
        assert "\n\nA full answer:\n- " in r["reference"] and len(r["meta"]["criteria"]) == 3
        assert len(r["prompt_sha256"]) == 64 and r["qid"] == eb.qid_of(r["prompt"])
    after = eb.current_fingerprints(config.JUDGED_TASKS_DIR)["exam_economics"]
    assert after != before and out["published"]["version"] == after[:12]


# ---------------------------------------------------------------------------
# a writer for this batch only; a new Everyday group
# ---------------------------------------------------------------------------

def test_a_writer_chosen_for_this_batch_is_pinned_and_its_spend_counted(svc, monkeypatch):
    import time
    client, _, _ = svc
    fake = FakeOpenRouter.install(monkeypatch)

    def lines(req):
        user = req["messages"][-1]["content"]
        n = int(user.split("COUNT: ", 1)[1].split()[0])
        return "\n".join(json.dumps(q) for q in llm._fake_qb_questions(
            {"draft_kind": "knowledge", "start": 0, "count": n, "topic": "Economics"}))
    fake.reply = lines
    r = client.post("/api/builder", json={"kind": "knowledge", "topic": "Economics",
                                          "count": 10, "writer": "z-ai/glm-5.3", "by": BY})
    assert r.status_code == 200, r.text
    end = time.time() + 10
    while time.time() < end and client.get(f"/api/builder/{r.json()['id']}").json()["status"] \
            == "writing":
        llm_poller.tick()
        time.sleep(0.05)
    d = client.get(f"/api/builder/{r.json()['id']}").json()
    assert d["writer"]["label"] == "GLM 5.3" and d["writer"]["chosen_here"] is True
    assert d["writer"]["id"] == "z-ai/glm-5.3-20260816"
    assert d["progress"]["written"] == 10 and not any(it["auto"] for it in d["items"])
    assert [(c["model"], c["provider"]) for c in fake.chat] == \
        [("z-ai/glm-5.3", {"order": ["inference-net"], "allow_fallbacks": False})]
    assert db.spend_this_month_by_job() == {"writer": pytest.approx(0.001)}
    # the AI models page's own choice for the job is untouched
    from service import ai_models
    assert ai_models.choice("writer") is None


def test_a_new_everyday_group_joins_the_bank_with_its_questions(svc):
    client, _, _ = svc
    import everyday as ev
    r = client.post("/api/builder", json={"kind": "everyday", "group": "new", "count": 10,
                                          "new_label": "Travel plans", "by": BY})
    assert r.status_code == 422 and "a name and one line" in r.json()["detail"]
    d = create(client, kind="everyday", group="new", count=10, new_label="Travel plans",
               new_about="booking, packing and timetables", dedup=False)
    assert d["spec"]["group"] == "travel_plans"
    assert "GROUP: travel_plans — booking, packing and timetables" in writer_prompts()[0]
    d = rest(client, review_all(client, d))
    _to_publish(client, d)
    assert ev.groups()["travel_plans"] == "Travel plans"
    assert {q["group"] for q in ev.load_bank() if q.get("batch_id") == d["id"]} == \
        {"travel_plans"}
    assert "travel_plans" in ev.split_counts()


def test_a_reply_with_nothing_readable_stops_and_resume_asks_again(svc, monkeypatch):
    client, _, _ = svc
    prose = {"on": True}

    def writer(req):
        if req.custom_id.startswith("qbw:") and prose["on"]:
            return "Here are some great questions about maths! 1. What is 2+2?"
    respond(monkeypatch, writer)
    d = create(client, kind="everyday", group="quick_maths", count=10)
    assert d["status"] == "failed" and d["items"] == []
    assert d["error"] == "nothing in the writer's reply could be read as questions — Resume asks again"
    prose["on"] = False
    client.post(f"/api/builder/{d['id']}/resume", json={"by": BY})
    drain()
    d = client.get(f"/api/builder/{d['id']}").json()
    assert d["status"] == "review" and d["progress"]["written"] == 10
