"""12n.1, Everyday questions read and edited where they are read — in the
service. An edit is saved beside what the question builder published, never
the repo's bank; its impact is shown first and matches the marks after it;
every stored answer is marked again with no model run; a reworded question
leaves every score until it is asked again; a retired one leaves the counts;
Undo gives the version before back, and its marks. A practice question is
never made hidden. The hidden half is the owner's alone, after a warning,
and every opening is logged — and still reaches no other endpoint, page or
writer. Fixtures and the fake judge only; nothing is asked of a model."""

from __future__ import annotations

import hashlib
import json

import pytest

import everyday as ev
from conftest import fresh, make_service
from service import config, db, llm, llm_poller, playground
from test_builder_12i2 import create, writer_prompts
from test_everyday_improve_12g2 import leaks

OWNER, OTHER = "masein", "sam"
A, B = "org/answers-right", "org/answers-wrong"


def _dir(model: str):
    return config.OUT_DIR / model.replace("/", "__")


def _answered(model: str, answer) -> None:
    """a model that answered every question of the bank: its answers kept,
    and marked"""
    d = _dir(model)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps({"model": model, "kind": "instruct"}))
    rows = []
    for q in ev.load_bank():
        raw = answer(q)
        rows.append({"key": ev.answer_key(q), "id": q["id"], "prompt_hash": ev.prompt_hash(q["prompt"]),
                     "raw": raw, "at": "2026-09-28T00-00-00"})
    (d / ev.ANSWERS_NAME).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ev.write(d, ev.mark(d))


def scripted(half_: str, group: str | None = None) -> dict:
    """a question of this half whose checks are all the script's, with a
    reference that passes them"""
    return next(q for q in ev.load_bank() if ev.half(q) == half_
                and (group is None or q["group"] == group)
                and not ev.judge_check(q) and q.get("reference")
                and ev.grade(q, q["reference"])[0] is True)


def item(model: str, qid: str) -> dict | None:
    return next((it for it in ev.read(_dir(model))["items"] if it["id"] == qid), None)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "BOARD_OWNER", OWNER)
    _answered(A, lambda q: q.get("reference") or "yes")
    _answered(B, lambda q: "I'm not sure.")
    ev.remark(config.OUT_DIR, judge=True, snapshot=False)     # the stand-in judge's verdicts
    yield client, appmod
    client.__exit__(None, None, None)


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# an edit: its impact, then every answer marked again — no model run
# ---------------------------------------------------------------------------

def test_a_checks_edit_marks_every_answer_again_as_its_impact_said(svc, monkeypatch):
    client, appmod = svc
    q = scripted(ev.PRACTICE)
    assert item(A, q["id"])["pass"] is True and item(B, q["id"])["pass"] is False
    # nothing may run a model: the runner's entry point fails the test if it's called
    import service.runner as runner
    monkeypatch.setattr(runner, "run_submission", lambda *a, **k: pytest.fail("a model ran"))
    was = ev.version()
    bank_sha, retired_sha = sha(ev.BANK_PATH), sha(ev.RETIRED_PATH)
    # "not sure" passes now, and the reference fails: both flip
    checks = [{"type": "contains_any", "values": ["not sure"]}]
    imp = client.post(f"/api/everyday/questions/{q['id']}/impact",
                      json={"changes": {"checks": json.dumps(checks)}, "by": OTHER}).json()
    assert imp["answers"] >= 2 and imp["unasked"] == [] and imp["judge"] == []
    flips = {f["model"]: (f["from"], f["to"]) for f in imp["flips"]}
    assert flips[A] == (True, False) and flips[B] == (False, True)
    assert imp["line"].startswith(f"{len(imp['flips'])} would flip: ")
    assert "answers-right ✓→✗" in imp["line"]
    # a reason is required
    r = client.post(f"/api/everyday/questions/{q['id']}/edit",
                    json={"changes": {"checks": checks}, "by": OTHER})
    assert r.status_code == 422 and r.json()["detail"] == "Why: every edit keeps its reason"
    r = client.post(f"/api/everyday/questions/{q['id']}/edit",
                    json={"changes": {"checks": checks}, "why": "it asks for doubt", "by": OTHER})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["edit"]["what"] == ["checks"] and j["remarked"] >= 2 and j["sent"] == 0
    # the marks now are the impact's, model by model
    for m, (_, to) in flips.items():
        assert item(m, q["id"])["pass"] is to, m
    # a new version of the bank, each model marked on it — and the repo's files untouched
    now = ev.version()
    assert now["hash"] != was["hash"] and ev.read(_dir(A))["version"]["hash"] == now["hash"]
    assert sha(ev.BANK_PATH) == bank_sha and sha(ev.RETIRED_PATH) == retired_sha
    assert ev.edits_path().is_file() and ev.edits_path().parent == config.BENCH_ROOT / "everyday"
    # who, when and why, on the question and in its history
    got = client.get(f"/api/everyday/questions/{q['id']}?by={OTHER}").json()
    assert got["question"]["edited"]["by"] == OTHER and got["question"]["edited"]["why"] == \
        "it asks for doubt"
    assert [h["action"] for h in got["history"]] == ["edit"]
    fresh(appmod)
    page = client.get("/api/results").json()["everyday"]
    assert next(x for x in page["questions"] if x["id"] == q["id"])["edited"]["by"] == OTHER
    assert page["version"]["hash"] == now["hash"] and A in page["models"]


def test_undo_gives_the_version_before_back_and_its_marks(svc):
    client, _ = svc
    q = scripted(ev.PRACTICE)
    before = {m: item(m, q["id"])["pass"] for m in (A, B)}
    was = ev.version()["hash"]
    client.post(f"/api/everyday/questions/{q['id']}/edit",
                json={"changes": {"checks": [{"type": "contains_any", "values": ["not sure"]}],
                                  "reference": "not sure"}, "why": "try", "by": OTHER})
    assert ev.version()["hash"] != was
    r = client.post(f"/api/everyday/questions/{q['id']}/undo", json={"by": OTHER})
    assert r.status_code == 200, r.text
    assert ev.version()["hash"] == was
    assert {m: item(m, q["id"])["pass"] for m in (A, B)} == before
    assert ev.question(q["id"])["checks"] == q["checks"]
    h = client.get(f"/api/everyday/questions/{q['id']}?by={OTHER}").json()["history"]
    assert [(x["action"], x["undone"]) for x in h] == [("undo", False), ("edit", True)]
    r = client.post(f"/api/everyday/questions/{q['id']}/undo", json={"by": OTHER})
    assert r.status_code == 422 and r.json()["detail"] == "Nothing left to undo"


def test_undo_of_a_rubric_gives_the_judges_marks_back_without_asking_it(svc, monkeypatch):
    client, _ = svc
    q = next(x for x in ev.load_bank() if ev.half(x) == ev.PRACTICE and ev.judge_check(x)
             and item(A, x["id"]) and item(A, x["id"])["pass"] is not None)
    before = item(A, q["id"])["pass"]
    asked = []
    real = ev.stub_for
    monkeypatch.setattr(ev, "stub_for", lambda *a: asked.append(a) or real(*a))
    client.post(f"/api/everyday/questions/{q['id']}/edit",
                json={"changes": {"rubric": "Passes only if it says the word banana."},
                      "why": "a stricter rubric", "by": OTHER})
    n = len(asked)
    assert n >= 1                                # the new rubric went to the judge
    client.post(f"/api/everyday/questions/{q['id']}/undo", json={"by": OTHER})
    assert len(asked) == n                       # the old one's verdicts came back, unasked
    assert item(A, q["id"])["pass"] is before


def test_a_reworded_question_leaves_every_score_until_asked_again(svc):
    client, appmod = svc
    p, h = scripted(ev.PRACTICE), scripted(ev.HIDDEN)
    total = ev.read(_dir(A))["total"]
    r = client.post(f"/api/everyday/questions/{p['id']}/edit",
                    json={"changes": {"prompt": p["prompt"] + " Please."}, "why": "clearer",
                          "by": OTHER})
    assert r.status_code == 200, r.text
    # it keeps its half, whatever its new words hash to
    assert ev.half(ev.question(p["id"])) == ev.PRACTICE
    e = ev.read(_dir(A))
    assert p["id"] not in {it["id"] for it in e["items"]} and p["id"] in e["changed"]
    assert p["id"] in {q["id"] for q in ev.unanswered(_dir(A))}       # a run asks it again
    # a hidden one: the owner's, and it leaves the score's count
    imp = client.post(f"/api/everyday/questions/{h['id']}/impact",
                      json={"changes": {"prompt": h["prompt"] + " Thanks."}, "by": OWNER}).json()
    assert A in imp["unasked"] and imp["line"].startswith("The words change: ")
    client.post(f"/api/everyday/questions/{h['id']}/edit",
                json={"changes": {"prompt": h["prompt"] + " Thanks."}, "why": "typo", "by": OWNER})
    assert ev.half(ev.question(h["id"])) == ev.HIDDEN
    assert ev.read(_dir(A))["total"] == total - 1
    fresh(appmod)
    m = client.get("/api/results").json()["everyday"]["models"][A]
    assert p["id"] in m["changed"] and m["changedHidden"] == 1 and m["total"] == total - 1


def test_retire_recounts_and_keeps_the_reason(svc):
    client, _ = svc
    h = scripted(ev.HIDDEN)
    total, n = ev.read(_dir(A))["total"], len(ev.load_bank())
    r = client.post(f"/api/everyday/questions/{h['id']}/retire", json={"why": "", "by": OWNER})
    assert r.status_code == 422
    r = client.post(f"/api/everyday/questions/{h['id']}/retire",
                    json={"why": "two right answers", "by": OWNER})
    assert r.status_code == 200, r.text
    assert len(ev.load_bank()) == n - 1 and ev.question(h["id"]) is None
    assert ev.read(_dir(A))["total"] == total - 1
    [row] = [json.loads(x) for x in ev.retired_here_path().read_text().splitlines()]
    assert row["id"] == h["id"] and row["why"] == "two right answers" and row["by"] == OWNER
    # and undone, it is back
    client.post(f"/api/everyday/questions/{h['id']}/undo", json={"by": OWNER})
    assert ev.question(h["id"]) is not None and ev.read(_dir(A))["total"] == total
    assert ev.retired_here_path().read_text() == ""


def test_a_practice_question_is_never_made_hidden(svc):
    client, _ = svc
    p, h = scripted(ev.PRACTICE), scripted(ev.HIDDEN)
    r = client.post(f"/api/everyday/questions/{p['id']}/edit",
                    json={"changes": {"half": "report"}, "why": "hide it", "by": OWNER})
    assert r.status_code == 422 and r.json()["detail"] == ev.PRACTICE_STAYS
    # hidden → practice is allowed, by the owner: it is revealed on purpose
    r = client.post(f"/api/everyday/questions/{h['id']}/edit",
                    json={"changes": {"half": "diagnose"}, "why": "reveal it", "by": OWNER})
    assert r.status_code == 200, r.text
    assert ev.half(ev.question(h["id"])) == ev.PRACTICE


# ---------------------------------------------------------------------------
# the hidden half: the owner's, after a warning, logged
# ---------------------------------------------------------------------------

def test_a_hidden_question_is_the_owners_alone(svc):
    client, _ = svc
    h = scripted(ev.HIDDEN)
    for path, body in ((f"/api/everyday/questions/{h['id']}/impact", {"changes": {}}),
                       (f"/api/everyday/questions/{h['id']}/edit",
                        {"changes": {"reference": "x"}, "why": "w"}),
                       (f"/api/everyday/questions/{h['id']}/retire", {"why": "w"})):
        r = client.post(path, json={**body, "by": OTHER})
        assert r.status_code == 403, path
    r = client.get(f"/api/everyday/questions/{h['id']}?by={OTHER}")
    assert r.status_code == 403 and h["prompt"] not in r.text
    assert client.get(f"/api/everyday/questions/{h['id']}?by={OWNER}").status_code == 200


def test_the_audit_needs_the_owner_and_the_warning_and_is_logged(svc):
    client, _ = svc
    g = scripted(ev.HIDDEN)["group"]
    r = client.post("/api/everyday/audit", json={"group": g, "by": OTHER, "confirm": True})
    assert r.status_code == 403 and db.hidden_audits() == []
    r = client.post("/api/everyday/audit", json={"group": g, "by": OWNER})
    assert r.status_code == 428 and r.json()["detail"] == (
        "These questions are the test. Don’t train on them or write questions toward them. "
        "This opening is logged.")
    assert db.hidden_audits() == []
    r = client.post("/api/everyday/audit", json={"group": g, "by": OWNER, "confirm": True,
                                                  "models": [A]})
    assert r.status_code == 200, r.text
    j = r.json()
    hidden = [q for q in ev.load_bank() if q["group"] == g and ev.half(q) == ev.HIDDEN]
    assert [q["id"] for q in j["questions"]] == [q["id"] for q in hidden]
    assert list(j["answers"]) == [A] and set(j["answers"][A]) == {q["id"] for q in hidden}
    [log] = db.hidden_audits()
    assert (log["by"], log["group"], log["n"]) == (OWNER, g, len(hidden))
    listed = client.get("/api/everyday/audits").json()
    assert listed["owner"] == OWNER and listed["audits"][0]["group"] == g


def test_hidden_questions_reach_no_other_endpoint_writer_or_page(svc, monkeypatch):
    client, appmod = svc
    h = scripted(ev.HIDDEN)
    g = h["group"]
    # the owner opens the half and edits one of it: still the test's
    client.post("/api/everyday/audit", json={"group": g, "by": OWNER, "confirm": True})
    client.post(f"/api/everyday/questions/{h['id']}/edit",
                json={"changes": {"reference": h.get("reference", "") + " ok"}, "why": "w",
                      "by": OWNER})
    fresh(appmod)
    # every page's payload, and every endpoint a page reads without a name
    for path in ("/api/results", "/api/playground/practice", "/api/judge-test",
                 "/api/everyday/audits", "/api/submissions", "/api/proposals", "/api/builder"):
        r = client.get(path)
        assert r.status_code in (200, 404), (path, r.status_code)
        assert leaks(r.text) == [], path
    # the Playground: a hidden id is no practice question
    assert playground.practice_item({"kind": "everyday", "id": h["id"]}) is None
    # the question writer: what it is sent for a new batch of this group
    monkeypatch.setattr(config, "CHECKER_PROVIDER", "fake", raising=False)
    monkeypatch.setattr(config, "CHECKER_MODEL", "fake-checker", raising=False)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    llm.reset()
    create(client, kind="everyday", group=g, count=10, dedup=False)
    assert writer_prompts() and all(leaks(p) == [] for p in writer_prompts())
    # Improve, and the training-data writer: a proposal on the group, then its data
    fake = llm.client()
    r = client.post("/api/proposals", json={"model": "fx/skewed-360m", "everyday": g,
                                            "requested_by": OWNER})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": OWNER, "edited_text": ""})
    client.post(f"/api/proposals/{pid}/generate", json={"requester": OWNER, "count": 6})
    sent = [q for q in fake.recorded() if q["custom_id"].split(":")[0] in ("proposal", "gen")]
    assert sent
    for q in sent:
        assert leaks(q["system"] + "\n" + q["user"]) == [], q["custom_id"]
    # the judge test's answers: the practice half only
    rows = [r for r in __import__("service.judge_test", fromlist=["x"])._everyday_rows()]
    assert rows and leaks(json.dumps(rows)) == []
    # and the page itself
    assert leaks(client.get("/").text) == []
