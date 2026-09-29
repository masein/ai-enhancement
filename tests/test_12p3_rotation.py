"""12p.3: a new hidden set for Everyday, built now and switched to when the
board's owner says — never before the demo, since it changes every score.

The plan says, per group, what to write (its hidden count, and 40% more for
what review sets aside), checks the two rules — a writer that is not a Qwen
model, a checker from another maker — and prices it all before anything
starts. The drafts are the owner's; what they publish is staged, scoring
nothing, until the switch: the new set in, the old one to practice but for
the ones retired (the weak or flagged, listed with why), every score to
History as "scored on the retired hidden set". Fixtures and the fake models
only; the tests' hidden set is the invented one."""

from __future__ import annotations

import json
import math

import pytest

import everyday as ev
from conftest import make_service
from service import builder, config, db, hidden_store, llm, rotation
from test_builder_12i2 import drain

OWNER = "masein"
GLM = {"label": "GLM 5.3", "id": "z-ai/glm-5.3-20260816", "kind": "openrouter",
       "price_in": 0.5, "price_out": 2.0}
LUNA = {"label": "GPT-6 Luna", "id": "openai/gpt-6-luna-20260901", "kind": "openrouter",
        "price_in": 1.0, "price_out": 4.0}
QWEN = {"label": "Qwen3.5 72B", "id": "qwen/qwen3.5-72b", "kind": "openrouter",
        "price_in": 0.5, "price_out": 2.0}


FLASH = {"label": "DeepSeek V4.1 Flash", "id": "deepseek/deepseek-v4.1-flash-20260801",
         "kind": "openrouter", "price_in": 0.2, "price_out": 0.8}


def as_models(monkeypatch, writer=GLM, checker=LUNA, judge=FLASH) -> None:
    """who writes, checks and judges, as the AI models page would name them;
    the fake backends still answer"""
    real = builder.who

    def who(job, override=None):
        return {"writer": writer, "checker": checker, "judge": judge}.get(job) \
            or real(job, override)
    monkeypatch.setattr(builder, "who", who)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "BOARD_OWNER", OWNER)
    monkeypatch.setattr(config, "CHECKER_PROVIDER", "fake", raising=False)
    monkeypatch.setattr(config, "CHECKER_MODEL", "fake-checker", raising=False)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    llm.reset()
    as_models(monkeypatch)
    yield client, appmod
    client.__exit__(None, None, None)


def get(client, did):
    return client.get(f"/api/builder/{did}", params={"by": OWNER}).json()


def review_all(client, d):
    """the owner's review of every question waiting on one"""
    for it in d["items"]:
        if it["auto"] or it["verdict"]:
            continue
        r = client.post(f"/api/builder/{d['id']}/review",
                        json={"n": it["n"], "verdict": "accept", "by": OWNER})
        assert r.status_code == 200, r.text
    return get(client, d["id"])


def stage_all(extra=0) -> dict[str, int]:
    """every group's new questions staged, as published drafts leave them"""
    got = {}
    rows = []
    for g, n in rotation.targets().items():
        for k in range(n + extra):
            rows.append({"id": f"everyday-{g}-hnew-{k:02d}", "group": g, "skill": "new",
                         "difficulty": 1, "prompt": f"a new hidden {g} question {k}, invented",
                         "reference": "ok", "checks": [{"type": "contains_any", "values": ["ok"]}],
                         "notes": "", "half": ev.HIDDEN})
        got[g] = n + extra
    rotation.staged_path().write_text("".join(json.dumps(r) + "\n" for r in rows),
                                      encoding="utf-8")
    return got


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

def test_the_plan_per_group_its_rules_and_its_cost(svc):
    client, _ = svc
    p = client.get("/api/everyday/rotation").json()
    hid = {g: sum(q["group"] == g for q in ev._raw_rows(ev.hidden_path())) for g in ev.groups()}
    assert {g["group"]: g["target"] for g in p["groups"]} == {g: n for g, n in hid.items() if n}
    assert all(g["write"] == math.ceil(g["target"] * 1.4) and g["staged"] == 0
               for g in p["groups"])
    assert [r["ok"] for r in p["rules"]] == [True, True]
    assert p["rules"][0]["line"] == "writer: GLM 5.3 (Z.ai)"
    assert p["rules"][1]["line"] == "checker: GPT-6 Luna (OpenAI)"
    # priced before anything starts: the sum of each group's estimate
    per = sum(g["estimate"]["usd"] for g in p["groups"])
    assert p["estimate"]["usd"] == round(per, 2) > 0
    written = sum(g["write"] for g in p["groups"])
    assert p["estimate"]["line"] == (f"about ${per:,.2f} for {written} questions written, "
                                     f"{sum(g['target'] for g in p['groups'])} kept")
    assert p["can_start"] is True and p["can_switch"] is False
    # counts and prices: no question
    assert not [q for q in ev._raw_rows(ev.hidden_path()) if q["prompt"] in json.dumps(p)]


@pytest.mark.parametrize("writer,checker,bad", [
    (QWEN, LUNA, "the writer must not be a Qwen model"),
    (GLM, {**GLM, "label": "GLM 5.3 Air", "id": "z-ai/glm-5.3-air"},
     "the checker must be from another maker than the writer"),
])
def test_a_qwen_writer_or_a_same_maker_checker_is_refused(svc, monkeypatch, writer, checker, bad):
    client, _ = svc
    as_models(monkeypatch, writer, checker)
    p = client.get("/api/everyday/rotation").json()
    assert p["can_start"] is False and bad in p["why"]
    r = client.post("/api/everyday/rotation/start", json={"by": OWNER, "confirm": True})
    assert r.status_code == 422 and bad in r.json()["detail"]
    # nor a draft for the hidden set by hand
    r = client.post("/api/builder", json={"kind": "everyday", "group": "honesty", "count": 5,
                                          "target": "hidden", "by": OWNER})
    assert r.status_code == 422 and bad in r.json()["detail"]


def test_start_shows_its_cost_first_and_is_the_owners(svc):
    client, _ = svc
    r = client.post("/api/everyday/rotation/start", json={"by": OWNER})
    assert r.status_code == 428 and r.json()["detail"].startswith(
        "Start writing a new hidden set: about $")
    r = client.post("/api/everyday/rotation/start", json={"by": "sam", "confirm": True})
    assert r.status_code == 403
    r = client.post("/api/everyday/rotation/start", json={"by": OWNER, "confirm": True})
    assert r.status_code == 200, r.text
    made = r.json()["drafts"]
    drain()
    drafts = [builder.get(i) for i in made]
    assert sorted(d["spec"]["group"] for d in drafts) == sorted(rotation.targets())
    assert all(d["spec"]["target"] == "hidden" and d["by"] == OWNER for d in drafts)
    assert all(d["spec"]["count"] == math.ceil(rotation.targets()[d["spec"]["group"]] * 1.4)
               for d in drafts)
    # under way: not twice
    p = client.get("/api/everyday/rotation").json()
    assert p["can_start"] is False and p["why"] == "a hidden set's drafts are under way"
    assert all(g["running"] for g in p["groups"])


# ---------------------------------------------------------------------------
# the drafts: the owner's, staged
# ---------------------------------------------------------------------------

def test_a_hidden_draft_is_the_owners_and_what_it_publishes_is_staged(svc):
    client, _ = svc
    r = client.post("/api/builder", json={"kind": "everyday", "group": "quick_maths",
                                          "count": 12, "target": "hidden", "by": OWNER})
    assert r.status_code == 200, r.text
    did = r.json()["id"]
    drain()
    # someone else: no questions
    assert client.get(f"/api/builder/{did}").status_code == 403
    assert client.get(f"/api/builder/{did}", params={"by": "sam"}).status_code == 403
    assert client.post(f"/api/builder/{did}/cancel", json={"by": "sam"}).status_code == 403
    d = get(client, did)
    assert d["items"]
    bank, version = ev.load_bank(), ev.version()
    review_all(client, d)
    assert client.post(f"/api/builder/{did}/rest", json={"by": OWNER}).status_code == 200
    drain()
    review_all(client, get(client, did))
    r = client.post(f"/api/builder/{did}/publish", json={"by": OWNER})
    assert r.status_code == 200, r.text
    got = r.json()["published"]
    assert got["staged"] is True and got["added"] > 0
    rows = rotation.staged()
    assert len(rows) == got["added"] and {q["half"] for q in rows} == {ev.HIDDEN}
    assert all(q["id"].startswith("everyday-quick_maths-h") for q in rows)
    # it scores nothing and is shown nowhere: the bank and its version as they were
    assert [q["id"] for q in ev.load_bank()] == [q["id"] for q in bank]
    assert ev.version() == version
    page = json.dumps(client.get("/api/results").json())
    assert not [q for q in rows if q["prompt"] in page]
    qm = next(g for g in client.get("/api/everyday/rotation").json()["groups"]
              if g["group"] == "quick_maths")
    assert qm["staged"] == len(rows)


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

def test_the_switch_waits_for_every_group(svc):
    client, _ = svc
    r = client.post("/api/everyday/rotation/switch", json={"by": OWNER, "confirm": True})
    assert r.status_code == 422 and r.json()["detail"].startswith(
        "every group needs its new questions first: Understanding 0 of ")


def test_the_candidates_are_the_owners_with_why_and_logged(svc, monkeypatch):
    client, _ = svc
    stage_all()
    assert client.get("/api/everyday/rotation/candidates").status_code == 403
    r = client.get("/api/everyday/rotation/candidates", params={"by": OWNER})
    assert r.status_code == 200
    got = r.json()["candidates"]
    # the fixture's good model passes most hidden questions and the skewed one half:
    # the ones both pass are "every model passes it"
    assert got and all(c["why"] for c in got)
    assert any(w.startswith("every model passes it") for c in got for w in c["why"])
    [log] = db.hidden_audits()
    assert log["group"] == "Everyday's hidden set · the switch's review" and log["n"] == len(got)


def test_the_switch_the_new_set_in_the_old_to_practice_every_score_to_history(svc, monkeypatch):
    client, appmod = svc
    old = ev._raw_rows(ev.hidden_path())
    before_v = ev.version()["hash"]
    staged = stage_all()
    retire = [old[0]["id"], old[1]["id"]]
    r = client.post("/api/everyday/rotation/switch", json={"by": OWNER, "retire": retire})
    assert r.status_code == 422 and "confirm" in r.json()["detail"]
    n_backups = len(hidden_store.backups())
    r = client.post("/api/everyday/rotation/switch", json={"by": OWNER, "confirm": True,
                                                          "retire": retire})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["count"] == sum(staged.values()) and rec["to_practice"] == len(old) - 2
    assert rec["retired"] == sorted(retire) and rec["old_version"] == before_v
    # the new set is the hidden set, and accepted without a manifest commit
    assert {q["id"] for q in ev._raw_rows(ev.hidden_path())} == {
        f"everyday-{g}-hnew-{k:02d}" for g, n in staged.items() for k in range(n)}
    assert rotation.staged() == []                                   # staged: none left
    st = ev.hidden_status()
    assert st["ok"] and st["state"] == "store" and st["count"] == sum(staged.values())
    bank = {q["id"]: q for q in ev.load_bank()}
    # the old set: practice now, but for the two retired
    assert all(ev.half(bank[q["id"]]) == ev.PRACTICE for q in old if q["id"] not in retire)
    assert not set(retire) & set(bank)
    assert sum(ev.half(q) == ev.HIDDEN for q in bank.values()) == sum(staged.values())
    assert ev.version()["hash"] == rec["new_version"] != before_v
    # backed up before and after; the old file kept beside; logged
    assert len(hidden_store.backups()) == n_backups + 2
    assert list(ev.built_dir().glob("hidden.before-*.jsonl"))
    assert db.hidden_audits()[0]["group"].startswith("Everyday's hidden set switched")
    # every model's score: History, scored on the retired hidden set
    appmod._cache.update(key=None, at=0.0)
    e = client.get("/api/results").json()["everyday"]
    assert e["models"] == {}
    assert e["earlier"] and {x["label"] for x in e["earlier"].values()} <= {
        "scored on the retired hidden set", "all questions, before the split",
        "an earlier wording"}
    assert "scored on the retired hidden set" in {x["label"] for x in e["earlier"].values()}
    # and the plan says it was done
    p = client.get("/api/everyday/rotation").json()
    assert p["switches"][-1]["new_digest"] == rec["new_digest"]
