"""12i.1: AI models, and the judge test — in the service, with a fake
OpenRouter (tests/fixtures/fake_openrouter.py): nothing here reaches the
network, and no model runs.

Each job's model is chosen from OpenRouter's live list, with its prices, the
suggested one marked; saved as its dated version and its first provider, and
every request sends that provider with allow_fallbacks false. Warnings show
only when they hold. At the monthly limit, jobs wait and say why. A new judge
is a new judge version: the old judged scores leave today's views for History.
The judge test's agreement numbers are checked by hand, and "provisional"
comes off only at a weighted kappa of 0.7 on 100 answers. With no key, only
Local is offered, and nothing breaks."""

from __future__ import annotations

import json
import time

import pytest

from conftest import fresh, make_service
from fake_openrouter import KEY, FakeOpenRouter
from service import ai_models, config, db, judge_test, llm, llm_poller


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client, appmod, tree
    client.__exit__(None, None, None)


@pytest.fixture
def fake(monkeypatch):
    return FakeOpenRouter.install(monkeypatch)


def save(client, job, model, by="masein"):
    r = client.post(f"/api/ai/jobs/{job}", json={"model": model, "by": by})
    assert r.status_code == 200, r.text
    return r.json()


def wait_done(backend, bid, t=10.0):
    end = time.time() + t
    while time.time() < end:
        state, detail = backend.status(bid)
        if state != "pending":
            return state, detail
        time.sleep(0.05)
    return backend.status(bid)


# ---------------------------------------------------------------------------
# the model list, pinning, and the requests
# ---------------------------------------------------------------------------

def test_the_model_list_loads_with_prices_and_marks_the_suggested(svc, fake):
    client, _, _ = svc
    got = client.get("/api/ai/models").json()
    ids = [m["id"] for m in got["models"]]
    # text models only: no image model, no free one
    assert "acme/painter-1" not in ids and "acme/free-chat:free" not in ids
    ds = next(m for m in got["models"] if m["id"] == "deepseek/deepseek-v4.1-flash")
    # prices per million tokens, and the dated version
    assert (ds["price_in"], ds["price_out"]) == (0.14, 0.42)
    assert ds["version"] == "deepseek/deepseek-v4.1-flash-20260910"
    assert got["suggested"]["judge"] == {"id": "deepseek/deepseek-v4.1-flash",
                                        "why": "cheap and strong as a judge when given a "
                                               "reference answer"}
    assert {k: v["id"] for k, v in got["suggested"].items()} == {
        "judge": "deepseek/deepseek-v4.1-flash", "writer": "z-ai/glm-5.3",
        "data": "z-ai/glm-5.3", "checker": "openai/gpt-6-luna"}
    # kept a day: a second look asks OpenRouter nothing
    n = len(fake.calls)
    client.get("/api/ai/models")
    assert len(fake.calls) == n
    client.get("/api/ai/models?refresh=1")
    assert len(fake.calls) == n + 1


def test_saving_a_model_pins_its_version_and_provider_and_requests_never_fall_back(svc, fake):
    client, _, _ = svc
    saved = save(client, "data", "z-ai/glm-5.3")["saved"]
    assert saved == {"kind": "openrouter", "id": "z-ai/glm-5.3", "version": "z-ai/glm-5.3-20260816",
                     "name": "Z.ai: GLM 5.3", "provider": "inference-net",
                     "provider_name": "InferenceNet", "precision": "fp8",
                     "price_in": 0.1, "price_out": 0.4}
    assert db.ai_get("job:data") == saved
    # the training-data writer's requests go to OpenRouter, pinned
    llm.reset()
    be = llm.client("llm")
    assert isinstance(be, llm.OpenRouterChat) and be.model == "z-ai/glm-5.3"
    bid = be.submit([llm.Request(custom_id=f"r{i}", system="s", user="write") for i in range(2)])
    assert wait_done(be, bid)[0] == "done"
    assert len(fake.chat) == 2
    for req in fake.chat:
        assert req["model"] == "z-ai/glm-5.3"
        assert req["provider"] == {"order": ["inference-net"], "allow_fallbacks": False}
    # its cost counts against the month, as OpenRouter reported it
    assert db.spend_this_month() == pytest.approx(0.002)
    assert db.spend_this_month_by_job() == {"data": pytest.approx(0.002)}
    # the page says which model does the job, where, and its price — and never the key
    page = client.get("/api/ai").json()
    row = next(j for j in page["jobs"] if j["job"] == "data")
    assert row["chosen"]["provider_name"] == "InferenceNet" and row["from"] == "page"
    assert KEY not in json.dumps(page) and KEY not in json.dumps(client.get("/api/llm").json())


def test_the_key_never_reaches_a_submitted_models_code(monkeypatch):
    from service import runner
    assert "OPENROUTER_API_KEY" in runner.SECRET_ENV_VARS
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    assert "OPENROUTER_API_KEY" not in runner._child_env(remote_code=True)


def test_a_pinned_id_that_moved_to_another_version_waits(svc, fake, monkeypatch):
    client, _, _ = svc
    save(client, "data", "z-ai/glm-5.3")
    import fake_openrouter as fo
    moved = [dict(m, canonical_slug="z-ai/glm-5.3-20261001") if m["id"] == "z-ai/glm-5.3" else m
             for m in fo.MODELS]
    monkeypatch.setattr(fo, "MODELS", moved)
    client.get("/api/ai/models?refresh=1")
    assert llm.blocked("llm") == ("z-ai/glm-5.3 now points to z-ai/glm-5.3-20261001, not the "
                                  "z-ai/glm-5.3-20260816 this job was pinned to — choose it again "
                                  "on AI models")


# ---------------------------------------------------------------------------
# warnings
# ---------------------------------------------------------------------------

def texts(client):
    return [w["text"] for w in client.get("/api/ai").json()["warnings"]]


def test_each_warning_shows_only_when_it_holds(svc, fake):
    client, _, _ = svc
    save(client, "judge", "deepseek/deepseek-v4.1-flash")
    save(client, "writer", "z-ai/glm-5.3")
    save(client, "data", "z-ai/glm-5.3-flash")
    save(client, "checker", "openai/gpt-6-luna")
    assert texts(client) == []                                   # the suggested four: none
    # the judge and a writer in one family
    save(client, "judge", "z-ai/glm-5.3-flash")
    assert texts(client) == [
        "The judge and the question writer are both GLM: a judge tends to favour its own "
        "family's style.",
        "The judge and the training-data writer are both GLM: a judge tends to favour its own "
        "family's style."]
    save(client, "judge", "deepseek/deepseek-v4.1-flash")
    # the checker and the writer in one family
    save(client, "checker", "z-ai/glm-5.3")
    assert texts(client) == ["The checker and the question writer are both GLM: a checker "
                             "shares its writer's blind spots."]
    save(client, "checker", "openai/gpt-6-luna")
    # a training-data writer whose terms restrict training on its output
    save(client, "data", "openai/gpt-6-luna")
    assert texts(client) == ["Check the terms — its output becomes training data. Open-weight "
                             "models (GLM, DeepSeek, Qwen) avoid this."]
    save(client, "data", "z-ai/glm-5.3-flash")
    # a writer of the family of a model being improved
    save(client, "writer", "qwen/qwen3.5-72b-instruct")
    db.proposal_create("Qwen/Qwen3-1.7B", "exam_law", "Law", "masein", {})
    assert texts(client) == ["The question writer is Qwen, like Qwen3-1.7B, which is being "
                             "improved: its data carries its own family's habits."]


# ---------------------------------------------------------------------------
# the monthly limit
# ---------------------------------------------------------------------------

def test_at_the_spend_limit_jobs_wait_and_the_page_says_why(svc, fake):
    client, _, _ = svc
    save(client, "data", "z-ai/glm-5.3")
    r = client.post("/api/ai/limit", json={"usd": 5, "by": "masein"})
    assert r.status_code == 200 and r.json()["spend"]["limit"] == 5
    db.spend_add("data", "z-ai/glm-5.3", "InferenceNet", 10, 10, 5.25)
    why = ("waiting: this month's AI spend has reached its $5.00 limit — raise it on AI models, "
           "or wait for next month")
    assert client.get("/api/ai").json()["spend"]["waiting"] == why
    assert client.get("/api/llm").json()["ai_waiting"] == why
    llm.reset()
    be = llm.client("llm")
    bid = be.submit([llm.Request(custom_id="r0", system="", user="write")])
    time.sleep(0.3)
    state, detail = be.status(bid)
    assert state == "pending" and detail.endswith(why)
    assert fake.chat == []                                       # nothing went out
    # a higher limit, and it goes
    client.post("/api/ai/limit", json={"usd": 20, "by": "masein"})
    assert wait_done(be, bid)[0] == "done" and len(fake.chat) == 1


# ---------------------------------------------------------------------------
# a new judge is a new judge version
# ---------------------------------------------------------------------------

def test_a_new_judge_version_moves_old_judged_scores_to_history(svc, fake):
    client, appmod, _ = svc
    judged = [m["id"] for m in client.get("/api/results").json()["models"] if m.get("judge")]
    assert judged
    r = save(client, "judge", "deepseek/deepseek-v4.1-flash")
    # it asks first: how many answers another judge marked, and about what re-judging costs
    assert r["rejudge"]["n"] > 0 and r["rejudge"]["usd"] > 0
    fresh(appmod)
    now = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    for mid in judged:
        assert now[mid]["judge"] is None and now[mid]["judgedAvg"] is None
        assert now[mid]["judgedEarlier"]["by"] == "overlap-v1"
    # Later: nothing queued, the old scores stay in History. Yes: a run per model
    q = client.post("/api/ai/rejudge", json={"by": "masein"})
    assert q.status_code == 200 and len(q.json()["queued"]) >= len(judged)
    rows = {r["id"]: r for r in client.get("/api/submissions").json()}
    notes = {rows[i]["suite"] for i in q.json()["queued"]}
    assert "judged" in notes and all(rows[i]["note"] == "judged again: a new judge"
                                     for i in q.json()["queued"])


def test_every_judged_answer_records_its_judge_version(svc):
    _, _, tree = svc
    import judge
    for d in tree["out_dir"].glob("*/judge.json"):
        j = json.loads(d.read_text(encoding="utf-8"))
        if j.get("skipped"):
            continue
        key = j["judge"]["version"]["key"]
        assert key == judge.version()["key"]
        assert all(it["judge_version"] == key for t in j["tasks"].values() for it in t["items"])


# ---------------------------------------------------------------------------
# the judge test
# ---------------------------------------------------------------------------

def test_agreement_is_what_a_hand_count_gives():
    """person 4 4 3 2 0 against judge 4 3 3 2 1: 3 of 5 the same, all 5 within
    one. Quadratic weights (i-j)²/16: observed 1/16 + 1/16 = 0.125; expected,
    from the margins (person 0:1 2:1 3:1 4:2, judge 1:1 2:1 3:2 4:1), is
    82/16/5 = 1.025; so kappa = 1 - 0.125/1.025 = 0.878"""
    ag = judge_test.agreement({"a": 4, "b": 4, "c": 3, "d": 2, "e": 0, "skipped": None},
                              {"a": 4, "b": 3, "c": 3, "d": 2, "e": 1, "skipped": 2})
    assert ag == {"n": 5, "exact": 0.6, "within1": 1.0, "kappa": 0.878}
    assert judge_test.weighted_kappa([2, 2], [2, 2]) == 1.0           # no spread, full agreement
    assert judge_test.weighted_kappa([0, 4], [4, 0]) == -1.0          # opposite ends
    assert judge_test.weighted_kappa([], []) is None


def _synthetic(monkeypatch, n, judge_marks):
    keys = [f"exam:k{i}" for i in range(n)]
    monkeypatch.setattr(judge_test, "answers", lambda *a, **k: [{"key": k} for k in keys])
    monkeypatch.setattr(judge_test, "current_marks", lambda: dict(zip(keys, judge_marks)))
    return keys


def test_provisional_comes_off_only_at_kappa_07_on_100_answers(svc, monkeypatch):
    person = [i % 5 for i in range(120)]
    # one point off on every sixth answer: kappa well over 0.7
    close = [p if i % 6 else min(4, p + 1) for i, p in enumerate(person)]
    keys = _synthetic(monkeypatch, 120, close)
    for k, v in zip(keys[:99], person):
        db.jt_mark(judge_test.PERSON, k, v, "masein")
    cal = judge_test.calibrate()
    assert cal["n"] == 99 and cal["kappa"] >= 0.7 and cal["calibrated"] is False   # 99: not yet
    for k, v in zip(keys[99:], person[99:]):
        db.jt_mark(judge_test.PERSON, k, v, "masein")
    cal = judge_test.calibrate()
    assert cal["n"] == 120 and cal["kappa"] >= 0.7 and cal["calibrated"] is True
    on_file = json.loads((config.OUT_DIR / "judge_calibration.json").read_text(encoding="utf-8"))
    assert on_file["calibrated"] is True and on_file["judge"]["id"] == "stub/overlap-v1"
    # far apart: never, however many
    _synthetic(monkeypatch, 120, [4 - p for p in person])
    assert judge_test.calibrate()["calibrated"] is False


def test_the_judge_test_marks_runs_a_candidate_and_can_make_it_the_judge(svc, fake):
    client, _, _ = svc
    page = client.get("/api/judge-test").json()
    assert page["answers"] and page["progress"]["marked"] == 0
    # the judges' marks are never in what he marks from
    assert all("judge" not in a for a in page["answers"])
    first = page["answers"][0]
    r = client.post("/api/judge-test/mark", json={"key": first["key"], "mark": 3, "by": "masein"})
    assert r.status_code == 200 and r.json()["marked"] == 1
    assert client.post("/api/judge-test/mark", json={"key": first["key"], "mark": 7,
                                                     "by": "masein"}).status_code == 422
    est = client.get("/api/judge-test/estimate?models=deepseek/deepseek-v4.1-flash").json()
    assert est["n"] == len(page["answers"]) and est["usd"] > 0
    run = client.post("/api/judge-test/run", json={"models": ["deepseek/deepseek-v4.1-flash"],
                                                   "by": "masein"})
    assert run.status_code == 200, run.text
    end = time.time() + 10
    while time.time() < end and llm_poller.tick() == 0:
        time.sleep(0.05)
    assert all(req["provider"]["allow_fallbacks"] is False for req in fake.chat)
    res = client.get("/api/judge-test/result").json()
    cand = next(r for r in res["rows"] if r["name"] == "DeepSeek V4.1 Flash")
    assert cand["n"] == 1 and cand["per_1000"] == pytest.approx(1000 * 0.001, rel=0.01)
    use = client.post("/api/judge-test/use", json={"key": cand["key"], "by": "masein"})
    assert use.status_code == 200 and use.json()["saved"]["id"] == "deepseek/deepseek-v4.1-flash"
    assert ai_models.choice("judge")["provider"] == "inference-net"


# ---------------------------------------------------------------------------
# no key
# ---------------------------------------------------------------------------

def test_with_no_key_only_local_is_offered_and_nothing_breaks(svc, monkeypatch):
    client, _, _ = svc
    called = []
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm, "_http", lambda *a, **k: called.append(a) or (_ for _ in ()).throw(
        AssertionError("OpenRouter was asked with no key")))
    page = client.get("/api/ai").json()
    assert page["has_key"] is False and page["warnings"] == []
    assert client.get("/api/ai/models").json() == {"has_key": False, "models": []}
    r = client.post("/api/ai/jobs/judge", json={"model": "deepseek/deepseek-v4.1-flash",
                                                "by": "masein"})
    assert r.status_code == 422 and "only Local can be chosen" in r.json()["detail"]
    assert client.post("/api/ai/jobs/writer", json={"model": "local", "by": "masein"}) \
        .status_code == 200
    assert client.get("/api/llm").status_code == 200
    # 12i.3: the local server may be asked its model's name; OpenRouter never
    assert not [c for c in called if c[1].startswith(config.OPENROUTER_BASE_URL)]
