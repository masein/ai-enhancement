"""Phase 9a: the five things Omar asked for — model search, answers that fit,
pages that outlive a deploy, Propose with a warning, one pager.

The API half here; the browser half is in test_the_five_asks_browser.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import fresh, make_service
from service import llm_poller

REPO = Path(__file__).resolve().parents[1]
MODEL = "fx/good-750m"
PROVISIONAL = "graded by a local model — not a pinned benchmark"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def make_provisional(model: str = MODEL) -> None:
    """The live board today: a local judge, whose every file is stamped."""
    from service import config
    jf = config.OUT_DIR / model.replace("/", "__") / "judge.json"
    j = json.loads(jf.read_text(encoding="utf-8"))
    j["judge"].update({"provisional": True, "provisional_reason": PROVISIONAL,
                       "base_url": "http://localhost:8000/v1", "served_model": "chat"})
    jf.write_text(json.dumps(j), encoding="utf-8")


def loop_row(client, topic: str) -> dict:
    return next(r for r in client.get("/api/loop").json()["topics"] if r["topic"] == topic)


# ---------------------------------------------------------------------------
# 1. model search
# ---------------------------------------------------------------------------

def test_the_boards_own_models_come_before_any_hub_result(svc, monkeypatch):
    client, appmod, _ = svc
    from service import suggest
    real = appmod.results_payload

    def with_smol():
        p = real()
        base = next(m for m in p["models"] if m["id"] == MODEL)
        extra = [dict(base, id=i, name=i.split("/")[-1], kind="instruct")
                 for i in ("HuggingFaceTB/SmolLM2-360M-Instruct",
                           "HuggingFaceTB/SmolLM2-135M-Instruct")]
        return {**p, "models": p["models"] + extra}
    monkeypatch.setattr(appmod, "results_payload", with_smol)
    calls = []

    def hub(q):
        calls.append(q)
        return [{"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "params": 1.7e9},
                {"id": "HuggingFaceTB/SmolLM2-360M-Instruct", "params": 3.6e8},  # already listed
                {"id": "HuggingFaceTB/giant-70B", "params": 7e10}]
    monkeypatch.setattr(suggest, "hub_search", hub)
    suggest._hub_cache.clear()
    got = client.get("/api/models/suggest", params={"q": "hugg"}).json()
    ids = [it["id"] for it in got["items"]]
    board = [i for i, it in enumerate(got["items"]) if it["source"] == "board"]
    hubs = [i for i, it in enumerate(got["items"]) if it["source"] == "hub"]
    assert board and hubs and max(board) < min(hubs)          # the board's own first
    assert ids.count("HuggingFaceTB/SmolLM2-360M-Instruct") == 1
    smol = got["items"][0]
    assert smol["on_board"] and smol["kind"] == "instruct" and smol["judged"] > 0
    big = next(it for it in got["items"] if it["id"] == "HuggingFaceTB/giant-70B")
    assert big["over_cap"] is True                            # greyed, still offered
    small = next(it for it in got["items"] if it["id"] == "HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert small["over_cap"] is False and small["kind"] == "instruct" and small["kind_guessed"]
    assert got["hub_ok"] is True and got["footer"] == ""
    # matching is a case-insensitive substring of the whole id
    for q, want in (("smol", "HuggingFaceTB/SmolLM2-360M-Instruct"),
                    ("360", "HuggingFaceTB/SmolLM2-360M-Instruct"), ("GOOD-750", MODEL)):
        assert want in [it["id"] for it in
                        client.get("/api/models/suggest", params={"q": q}).json()["items"]], q
    # cached: the same query does not ask the Hub again
    n = len(calls)
    client.get("/api/models/suggest", params={"q": "hugg"})
    assert len(calls) == n


def test_a_hub_that_does_not_answer_leaves_the_local_matches_and_says_so(svc, monkeypatch):
    client, _, _ = svc
    from service import suggest

    def down(q):
        raise OSError("the Hub is down")
    monkeypatch.setattr(suggest, "hub_search", down)
    suggest._hub_cache.clear()
    got = client.get("/api/models/suggest", params={"q": "good"}).json()
    assert [it["id"] for it in got["items"]][:1] == [MODEL]
    assert got["hub_ok"] is False and "Hub search unavailable" in got["footer"]
    # and one character is not a search
    assert client.get("/api/models/suggest", params={"q": "g"}).json()["items"] == []


def test_a_slow_hub_never_holds_the_answer_past_two_seconds(svc, monkeypatch):
    import time
    client, _, _ = svc
    from service import suggest
    monkeypatch.setattr(suggest, "HUB_TIMEOUT_S", 0.3)
    monkeypatch.setattr(suggest, "hub_search", lambda q: time.sleep(2) or [])
    suggest._hub_cache.clear()
    t0 = time.time()
    got = client.get("/api/models/suggest", params={"q": "good"}).json()
    assert time.time() - t0 < 1.5
    assert got["hub_ok"] is False and got["items"]


def test_queued_models_and_uploaded_checkpoints_are_found_too(svc):
    client, _, _ = svc
    from service import config, db
    db.add("org/queued-only-model", "auto", "quick", "omar", "")
    (config.ARTIFACTS_DIR / "my-ckpt-step900").mkdir(parents=True)
    ids = [it["id"] for it in client.get("/api/models/suggest",
                                         params={"q": "queued-only"}).json()["items"]]
    assert ids[0] == "org/queued-only-model"
    ids = [it["id"] for it in client.get("/api/models/suggest",
                                         params={"q": "step900"}).json()["items"]]
    assert ids[0] == "local/my-ckpt-step900"


# ---------------------------------------------------------------------------
# 3. the build every answer came from
# ---------------------------------------------------------------------------

def test_every_api_answer_names_the_build_the_page_was_built_as(svc):
    client, appmod, _ = svc
    page = client.get("/").text
    assert f'<meta name="evalboard-build" content="{appmod.BUILD}">' in page
    for path in ("/api/results", "/api/submissions", "/api/loop",
                 "/api/models/suggest?q=go"):
        assert client.get(path).headers["X-Evalboard-Build"] == appmod.BUILD, path
    # a page is not an API answer
    assert "X-Evalboard-Build" not in client.get("/").headers


def test_the_build_changes_with_the_page_even_without_a_sha(monkeypatch):
    from service import app
    monkeypatch.setenv("EVALBOARD_BUILD", "")
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    a, b = app.build_id("<page one>"), app.build_id("<page two>")
    assert a != b and len(a) == 7
    monkeypatch.setenv("EVALBOARD_BUILD", "472d5a5")
    assert app.build_id("<page one>").startswith("472d5a5+")


# ---------------------------------------------------------------------------
# 4. propose over a provisional judge
# ---------------------------------------------------------------------------

def test_every_reason_is_evaluated_and_sorted_into_judge_and_data(svc):
    """topic_gate returned on the first preliminary reason and never looked
    at the data. Now both, always."""
    client, appmod, _ = svc
    make_provisional()
    fresh(appmod)
    med = loop_row(client, "medicine & health")               # under the floor in the fixture
    g = med["propose_by_model"][MODEL]
    assert any(PROVISIONAL in r for r in g["soft"])
    assert any("under the 30" in r for r in g["hard"])
    assert g["ok"] is False and g["overridable"] is False
    econ = loop_row(client, "economics")["propose_by_model"][MODEL]
    assert econ["hard"] == [] and econ["soft"] and econ["overridable"] is True
    assert econ["provisional"] is True
    assert set(econ) >= {"ok", "overridable", "soft", "hard", "why", "short", "caution"}


def test_soft_reasons_alone_propose_with_the_override_and_carry_the_mark(svc):
    client, appmod, _ = svc
    make_provisional()
    fresh(appmod)
    body = {"model": MODEL, "topic": "economics", "requested_by": "Omar"}
    r = client.post("/api/proposals", json=body)
    assert r.status_code == 409 and PROVISIONAL in r.json()["detail"]      # as today
    r = client.post("/api/proposals", json={**body, "override_preliminary": True,
                                            "requested_by": ""})
    assert r.status_code == 422 and "name" in r.json()["detail"]
    r = client.post("/api/proposals", json={**body, "override_preliminary": True})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    p = client.get(f"/api/proposals/{pid}").json()
    over = p["override"]
    assert over["by"] == "Omar" and over["at"] > 0
    assert any(PROVISIONAL in x for x in over["reasons"])
    # the Loop board's proposal carries it too
    assert loop_row(client, "economics")["proposal"]["override"]["by"] == "Omar"
    # and so does the dataset made from it: the provenance record, and the list
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
    did = client.post(f"/api/proposals/{pid}/generate",
                      json={"requester": "Omar", "count": 10}).json()["dataset_id"]
    llm_poller.tick()
    d = client.get(f"/api/datasets/{did}").json()
    assert d["status"] == "ready", d.get("error")
    from service import config
    prov = json.loads((config.DATASETS_DIR / str(did) / "provenance.json").read_text("utf-8"))
    assert prov["proposed_over_provisional_judge"]["by"] == "Omar"
    assert prov["proposed_over_provisional_judge"]["reasons"] == over["reasons"]
    assert d["over_provisional_judge"]["by"] == "Omar"
    # and a model trained on it: the taint trail says how its data was proposed
    run = client.post("/api/truns", json={"name": "demo", "datasets": [did],
                                          "parent": MODEL}).json()["id"]
    client.post(f"/api/truns/{run}/event", json={"step": 1, "detail": "fx/good-750m-tuned-test"})
    fresh(appmod)
    m = next(x for x in client.get("/api/results").json()["models"]
             if x["id"] == "fx/good-750m-tuned-test")
    assert m["taintTrail"]["over_provisional_judge"][str(pid)]["by"] == "Omar"


def test_a_data_reason_refuses_whatever_the_override_says(svc):
    client, appmod, _ = svc
    make_provisional()
    fresh(appmod)
    r = client.post("/api/proposals", json={"model": MODEL, "topic": "medicine & health",
                                            "requested_by": "Omar",
                                            "override_preliminary": True})
    assert r.status_code == 409
    assert "under the 30" in r.json()["detail"]
    assert r.json()["detail"] == loop_row(client, "medicine & health")[
        "propose_by_model"][MODEL]["why"]


def test_with_the_override_off_the_gate_is_what_it_was(svc, monkeypatch):
    client, appmod, _ = svc
    from service import config
    import report_lm_eval as report
    monkeypatch.setattr(config, "ALLOW_PRELIMINARY_OVERRIDE", False)
    make_provisional()
    fresh(appmod)
    g = loop_row(client, "economics")["propose_by_model"][MODEL]
    assert g["overridable"] is False and g["ok"] is False
    # the old words, exactly: what topic_gate said before it knew about soft reasons
    row = next(m for m in client.get("/api/results").json()["models"] if m["id"] == MODEL)
    legacy = row["judge"]["tasks"]["exam_economics"]["propose"]
    assert g["why"] == legacy["why"] and legacy["why"].startswith(
        "the judged suite is preliminary, so no topic score is evidence yet")
    r = client.post("/api/proposals", json={"model": MODEL, "topic": "economics",
                                            "requested_by": "Omar",
                                            "override_preliminary": True})
    assert r.status_code == 409 and r.json()["detail"] == legacy["why"]
    # a draft rubric or a single-provider loop is a caveat, not a gate, as before
    t = {"n_report": 40, "answers": {"n": 20, "distinct": 20}}
    ok_state = {"ok": True, "reasons": []}
    head = {"single_provider_loop": True, "rubrics_draft": ["exam_law"]}
    old = report.topic_gate("exam_law", t, ok_state, None, head)
    assert old["ok"] is True and old["soft"] == [] and len(old["soft_extra"]) == 2


def test_the_override_setting_defaults_on():
    """It is on so the live board can demo the loop today; 0 restores the gate."""
    import os
    from service import config
    if "ALLOW_PRELIMINARY_OVERRIDE" not in os.environ:
        assert config.ALLOW_PRELIMINARY_OVERRIDE is True
    src = (REPO / "service" / "config.py").read_text(encoding="utf-8")
    assert 'os.environ.get("ALLOW_PRELIMINARY_OVERRIDE", "1") == "1"' in src
