"""16c, parts 2 to 7: what OpenRouter said, said; "Pro only" a choice on the
card; AI models current; /api/judge-test/result with either shape of a judge
record; no page or batch waiting on OpenRouter's model list; "How it's served"
corrected from the page; and a served model's first Mobile estimate from
another build of the same model.

The key reached its own spending limit at about 12:38 on 5 Oct: Gemini's batch
then had 2,714 of 9,497 refused, each recorded as failed, the card said
nothing, the third labeller was blamed for a data policy it doesn't have, and
Carry on said "Refused" after it had sent a batch. Invented rows, an
OpenRouter double and fake replies; nothing calls OpenRouter."""

from __future__ import annotations

import json
import threading
import time

import pytest

import mobile_mmlu as mmp
from conftest import make_service
from fake_openrouter import KEY, FakeOpenRouter
from service import ai_models, config, db, judge_test, llm, llm_poller, mmp_key, served
from test_14_3_mobile_mmlu import fake_pin, put_invented
from test_14_4_1_mobile_mmlu_data import put_full

ME = "masein"
LIMIT_BODY = json.dumps({"error": {"code": 402, "message": "Key limit exceeded",
                                   "metadata": {"limit_source": "openrouter_key_limit"}}})
LIMIT_WORDS = ("OpenRouter refused the key: it has reached its own spending limit, or the account "
               "is out of credit. That is fixed on OpenRouter’s side: raise the key’s limit or add "
               "credit there")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch)
    put_invented(monkeypatch, config.MMP_DIR)
    monkeypatch.setattr(llm.LocalOpenAI, "BACKOFF", (0, 0, 0))
    yield client, appmod
    client.__exit__(None, None, None)


@pytest.fixture
def both(svc, monkeypatch):
    put_full(monkeypatch, config.BENCH_ROOT / "data" / "mobile_mmlu")
    return svc


def drain(timeout: float = 15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        llm_poller.tick()
        if not mmp_key.pending():
            return
        time.sleep(0.05)
    pytest.fail("the labellers' batches never finished")


class Chat:
    """OpenRouter's chat completions, as llm._http meets them: a label, or the
    error `fail(model, n)` names for the n-th request to that model"""

    def __init__(self, fail=lambda model, n: None):
        self.fail, self.n, self.sent = fail, {}, []
        self.lock = threading.Lock()

    def __call__(self, method, url, headers, body=None, timeout=60.0):
        assert url.endswith("/chat/completions"), url
        req = json.loads(body)
        with self.lock:
            n = self.n[req["model"]] = self.n.get(req["model"], 0) + 1
            self.sent.append(req["model"])
        err = self.fail(req["model"], n)
        if err:
            status, text = err
            raise llm.LLMError(f"POST {url}: HTTP {status}: {text}", status=status)
        return 200, json.dumps({
            "choices": [{"message": {"content": json.dumps({"answer": "A",
                                                            "depends_on_now": False})},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.01}}).encode()


def labelling(monkeypatch, chat: Chat):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(ai_models, "pin", fake_pin)
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    monkeypatch.setattr(llm, "_http", chat)


# ---------------------------------------------------------------------------
# part 2: what OpenRouter said
# ---------------------------------------------------------------------------

def test_a_probe_refused_for_the_keys_limit_for_data_policy_and_timed_out_says_each(
        svc, monkeypatch):
    fake = FakeOpenRouter.install(monkeypatch)
    probes = {"how": None}

    def http(method, url, headers, body=None, timeout=60.0):
        if url.endswith("/chat/completions") and json.loads(body).get("max_tokens") == 1:
            if probes["how"] == "limit":
                raise llm.LLMError(f"POST {url}: HTTP 402: {LIMIT_BODY}", status=402)
            if probes["how"] == "timeout":
                raise llm.LLMError(f"POST {url}: no response within {timeout:g}s")
        return fake(method, url, headers, body, timeout)
    monkeypatch.setattr(llm, "_http", http)
    probes["how"] = "limit"
    with pytest.raises(ValueError) as e:
        ai_models.pin("z-ai/glm-5.3")
    assert str(e.value) == "z-ai/glm-5.3 couldn’t be pinned: " + LIMIT_WORDS
    probes["how"] = None
    fake.collects = {"inference-net", "morph/fp8"}
    with pytest.raises(ValueError) as e:
        ai_models.pin("z-ai/glm-5.3")
    assert str(e.value) == ("every provider running z-ai/glm-5.3 may store or train on prompts: "
                            "choose another model")
    fake.collects = set()
    probes["how"] = "timeout"
    with pytest.raises(ValueError) as e:
        ai_models.pin("z-ai/glm-5.3")
    assert str(e.value) == "z-ai/glm-5.3 couldn’t be pinned: OpenRouter didn’t answer in time"
    # never the key, never a header
    assert KEY not in LIMIT_WORDS and "Bearer" not in ai_models.refusal(402, f"Bearer {KEY}")[1]


def test_start_with_a_labeller_that_cant_be_pinned_sends_nothing_and_says_why_once(
        svc, monkeypatch):
    client, _ = svc
    chat = Chat()
    labelling(monkeypatch, chat)

    def pin(model_id, job="pin"):
        if model_id == mmp.DEFAULT_LABELLERS["third"]["id"]:
            raise ValueError(f"{model_id} couldn’t be pinned: {LIMIT_WORDS}")
        return fake_pin(model_id)
    monkeypatch.setattr(ai_models, "pin", pin)
    r = client.post("/api/mobile-mmlu/key/start", json={"by": ME})
    assert r.status_code == 409
    third = mmp.DEFAULT_LABELLERS["third"]["id"]
    assert r.json()["detail"] == f"Third labeller: {third} couldn’t be pinned: {LIMIT_WORDS}"
    # nothing sent: no batch, no request
    assert not db.batches_list() and not chat.sent
    assert "choose another" not in r.json()["detail"]


def test_a_batch_that_part_fails_and_one_that_fails_whole_are_on_the_card(svc, monkeypatch):
    client, _ = svc
    first, second = (mmp.DEFAULT_LABELLERS[s]["id"] for s in ("first", "second"))

    def fail(model, n):
        if model == first and n % 4 == 1:            # 3 of its 12
            return 400, json.dumps({"error": {"code": 400, "message": "Bad request"}})
        if model == second:                          # every one: the key's limit
            return 402, LIMIT_BODY
        return None
    labelling(monkeypatch, Chat(fail))
    assert client.post("/api/mobile-mmlu/key/start", json={"by": ME}).status_code == 200
    drain()
    page = client.get("/api/mobile-mmlu/key").json()
    last = page["last"]
    assert last["first"]["failed"] and last["first"]["line"] == (
        "First labeller’s last batch: 12 sent · 9 answered · 3 failed. The first failure: "
        "OpenRouter refused it (HTTP 400): Bad request.")
    assert last["second"]["status"] == "failed"
    assert last["second"]["line"] == ("Second labeller’s last batch: 12 sent · 0 answered · 12 "
                                      "failed. The first failure: " + LIMIT_WORDS + ".")
    # the same line serves every job that uses OpenRouter
    tl = llm.tally(last["second"]["batch_id"])
    assert llm.batch_line(tl) == last["second"]["line"].split("’s last batch: ", 1)[1]


def test_twenty_refusals_in_a_row_pause_a_batch_and_mark_none_failed(svc, monkeypatch):
    state = {"refuse": True}
    chat = Chat(lambda model, n: (402, LIMIT_BODY) if state["refuse"] else None)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(llm, "_http", chat)
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    be = llm.OpenRouterChat("x/y", KEY, config.BENCH_ROOT, pin=fake_pin("x/y"), role="judge")
    bid = be.submit([llm.Request(f"r:{i}", "", f"q{i}", max_tokens=10) for i in range(50)])
    end = time.time() + 10
    while time.time() < end and not be.halted(bid):
        time.sleep(0.02)
    time.sleep(0.2)
    tl = llm.tally(bid)
    # 20 sent, all refused: the batch waits — nothing recorded failed
    assert len(chat.sent) == 20 and (tl["answered"], tl["failed"]) == (0, 0)
    assert tl["halted"] == ("waiting: OpenRouter refused 20 requests in a row — " + LIMIT_WORDS
                            + ". It tries again in 10 minutes")
    state_, detail = be.status(bid)
    assert state_ == "pending" and detail.endswith(tl["halted"])
    # fixed on OpenRouter's side: taken up again, and every one answered
    state["refuse"] = False
    be.resume(bid)
    end = time.time() + 10
    while time.time() < end and be.status(bid)[0] == "pending":
        time.sleep(0.02)
    tl = llm.tally(bid)
    assert (tl["sent"], tl["answered"], tl["failed"], tl["halted"]) == (50, 50, 0, "")


def test_a_labellers_batch_paused_by_refusals_says_so_and_carry_on_takes_it_up(
        svc, monkeypatch):
    client, _ = svc
    monkeypatch.setattr(mmp_key.LabellerChat, "HALT_AFTER", 5)
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    state = {"refuse": True}
    labelling(monkeypatch, Chat(lambda model, n: (402, LIMIT_BODY) if state["refuse"] else None))
    assert client.post("/api/mobile-mmlu/key/start", json={"by": ME}).status_code == 200
    end = time.time() + 10
    while time.time() < end and len(client.get("/api/mobile-mmlu/key").json()["running"]) < 2:
        time.sleep(0.05)
    # 17d: up to 15 s — two batches' five refusals each, on a busy CI runner,
    # took longer than the 5 s this waited
    for _ in range(300):
        page = client.get("/api/mobile-mmlu/key").json()
        lines = [r["progress"] for r in page["running"]]
        if lines and all("refused 5 requests in a row" in x for x in lines):
            break
        time.sleep(0.05)
    # 18b: it waits for Carry on — it took itself up after 10 minutes
    assert lines and all(x.endswith("It waits for Carry on: nothing more is sent until then")
                         for x in lines), lines
    assert all(x.startswith("0 of 12 · ") for x in lines)
    # Carry on: taken up again
    state["refuse"] = False
    assert client.post("/api/mobile-mmlu/key/start", json={"by": ME}).status_code == 200
    drain()
    assert mmp.current_key()["counts"]["all"]["kept"] == 12


def test_a_rate_limit_is_retried(svc, monkeypatch):
    chat = Chat(lambda model, n: (429, '{"error": {"message": "rate limited"}}') if n < 3 else None)
    monkeypatch.setattr(llm, "_http", chat)
    be = llm.OpenRouterChat("x/y", KEY, config.BENCH_ROOT, pin=fake_pin("x/y"), role="judge")
    rec = be._complete({"custom_id": "a", "system": "", "user": "hi", "max_tokens": 10,
                        "json": False})
    assert rec["error"] == "" and rec["attempts"] == 3 and "answer" in rec["text"]


def test_the_keys_own_allowance_is_on_the_page_and_a_dearer_job_is_warned(svc, monkeypatch):
    client, _ = svc
    fake = FakeOpenRouter.install(monkeypatch)
    for s in mmp.SLOTS:                              # priced: each labeller pinned
        db.ai_set(f"labeller:{s}", fake_pin(mmp.DEFAULT_LABELLERS[s]["id"]), ME)
    fake.key_data = {"limit": 50.0, "limit_remaining": 0.0005, "usage": 49.9995}
    for _ in range(100):
        k = client.get("/api/ai").json()["spend"]["key"]
        if k and k.get("remaining") is not None:
            break
        time.sleep(0.02)
    assert (k["limit"], k["remaining"]) == (50.0, 0.0005)
    page = client.get("/api/mobile-mmlu/key").json()
    usd = page["sends"]["usd"]
    assert usd > 0.0005 and page["key_warning"] == (
        f"About ${usd:,.2f} is more than the OpenRouter key may still spend ($0.00): raise the "
        "key’s limit on OpenRouter first, or what is sent stops where the key runs out")
    # kept a minute: asked once, however often the page asks
    asked = sum(1 for _, p in fake.calls if p == "/key")
    for _ in range(5):
        client.get("/api/ai")
    assert sum(1 for _, p in fake.calls if p == "/key") == asked


# ---------------------------------------------------------------------------
# part 3: Pro only, a choice on the card
# ---------------------------------------------------------------------------

def test_pro_only_by_default_sends_none_of_the_full_set_and_the_choice_changed_sends_it(
        both, monkeypatch):
    client, _ = both
    chat = Chat()
    labelling(monkeypatch, chat)
    for s in mmp.SLOTS:                              # priced: each labeller pinned
        db.ai_set(f"labeller:{s}", fake_pin(mmp.DEFAULT_LABELLERS[s]["id"]), ME)
    page = client.get("/api/mobile-mmlu/key").json()
    assert page["scope"]["value"] == "pro" and not page["scope"]["chosen"]
    assert page["sends"]["words"] == "Pro only"
    assert page["sends"]["usd"] == page["estimate"]["sets"]["pro"]["usd"] \
        < page["estimate"]["sets"]["all"]["usd"]
    r = client.post("/api/mobile-mmlu/key/start", json={"by": ME})
    assert r.status_code == 200 and {x["n"] for x in r.json()["sent"].values()} == {12}
    drain()
    page = client.get("/api/mobile-mmlu/key").json()
    # Pro's key is whole: said, and nothing more to send — none of the full set's
    assert page["done"] is True and page["sets"]["pro"]["counts"]["all"]["kept"] == 12
    assert len(chat.sent) == 24 and not mmp_key.pending()
    # the choice changed: kept with who and when, and the next Start sends the rest
    r = client.post("/api/mobile-mmlu/key/scope", json={"scope": "all", "by": ME})
    assert r.status_code == 200
    page = r.json()["page"]
    assert (page["scope"]["value"], page["scope"]["by"]) == ("all", ME) and page["scope"]["at"]
    assert page["done"] is False and page["sends"]["words"] == "Pro, then the full set"
    r = client.post("/api/mobile-mmlu/key/start", json={"by": ME})
    assert {x["n"] for x in r.json()["sent"].values()} == {7}      # the full set's own, and inv00004
    drain()
    assert client.get("/api/mobile-mmlu/key").json()["sets"]["full"]["counts"]["all"]["kept"] == 18


def test_a_waiting_full_set_batch_is_cancelled_under_pro_only_and_never_sent(both, monkeypatch):
    client, _ = both
    chat = Chat()
    labelling(monkeypatch, chat)
    db.ai_set("mmp:scope", "all", ME)
    # the month's limit reached: what is out waits — as on 5 Oct
    monkeypatch.setattr(ai_models, "over_limit", lambda: "waiting: this month's AI spend")
    db.ai_set("labeller:first", fake_pin(mmp.DEFAULT_LABELLERS["first"]["id"]), ME)
    full_only = [q["lid"] for q in mmp.pool() if q["sets"] == ["full"]]
    bid = mmp_key._submit("first", mmp_key.chosen("first"), full_only, ME)
    time.sleep(0.2)
    assert not chat.sent and mmp_key.pending()
    r = client.post("/api/mobile-mmlu/key/scope", json={"scope": "pro", "by": ME})
    assert r.status_code == 200
    # cancelled, never failed; the batch done, and nothing sent when the month turns
    monkeypatch.setattr(ai_models, "over_limit", lambda: "")
    llm_poller.tick()
    tl = llm.tally(bid)
    assert (tl["cancelled"], tl["failed"], tl["answered"]) == (len(full_only), 0, 0)
    assert not mmp_key.pending()
    time.sleep(0.2)
    assert not chat.sent


# ---------------------------------------------------------------------------
# part 5: either shape of a judge record
# ---------------------------------------------------------------------------

def test_the_judge_test_table_reads_either_shape_and_one_odd_file_never_breaks_it(
        tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    try:
        assert judge_test.judge_of({"id": "j", "version": {"key": "k1"}}) == ("j", "k1")
        assert judge_test.judge_of({"id": "j", "version": "k1"}) == ("j", "k1")
        assert judge_test.judge_of("deepseek/deepseek-v4.1-flash") == (
            "deepseek/deepseek-v4.1-flash", "")
        assert judge_test.judge_of(["odd"]) == ("", "")
        # every model's files with the shapes the server had: a string version
        # (everyday.json), a string judge, and a judge that is a list
        for d in (p for p in config.OUT_DIR.iterdir() if p.is_dir()):
            for name, head in (("everyday.json", {"id": "x", "version": "abc"}),
                               ("judge.json", "a-judge-by-name")):
                f = d / name
                try:
                    got = json.loads(f.read_text())
                except (OSError, ValueError):
                    continue
                got["judge"] = head
                f.write_text(json.dumps(got))
        r = client.get("/api/judge-test/result")
        assert r.status_code == 200, r.text
        appmod._rejudge_scope()                      # a new judge's scope reads them too
    finally:
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# part 6: no page or batch waits on OpenRouter's list
# ---------------------------------------------------------------------------

def test_with_openrouter_hanging_the_pages_answer_and_a_batch_never_waits_on_the_list(
        svc, monkeypatch):
    client, appmod = svc
    fake = FakeOpenRouter.install(monkeypatch)
    pin = ai_models.pin("z-ai/glm-5.3")
    db.ai_set("job:judge", pin, ME)
    # the day-old list has expired; OpenRouter now hangs on everything but chat
    p = ai_models._cache_path()
    got = json.loads(p.read_text())
    p.write_text(json.dumps({**got, "at": time.time() - ai_models.CACHE_S - 60}))
    asked = []

    def hang(method, url, headers, body=None, timeout=60.0):
        if not url.endswith("/chat/completions"):
            asked.append(url)
            time.sleep(min(timeout, 3))
            raise llm.LLMError(f"{method} {url}: no response within {timeout:g}s")
        return fake(method, url, headers, body, timeout)
    monkeypatch.setattr(llm, "_http", hang)
    for path in ("/healthz", "/api/ai", "/api/llm", "/api/mobile-mmlu/key", "/api/ai/models"):
        t0 = time.time()
        assert client.get(path).status_code == 200, path
        assert time.time() - t0 < 1.0, (path, time.time() - t0)
    # a batch's requests: none waits on the list
    be = llm.OpenRouterChat(pin["id"], KEY, config.BENCH_ROOT, pin=pin, role="judge")
    t0 = time.time()
    bid = be.submit([llm.Request(f"r:{i}", "", f"q{i}", max_tokens=10) for i in range(30)])
    while be.status(bid)[0] == "pending" and time.time() - t0 < 10:
        time.sleep(0.02)
    assert be.status(bid)[0] == "done" and time.time() - t0 < 2.0
    # the list was asked once, in the background, and the failure remembered
    time.sleep(3.2)
    lists = [u for u in asked if u.endswith("/models")]
    assert len(lists) == 1
    ai_models.models()
    ai_models.drifted(pin)
    assert len([u for u in asked if u.endswith("/models")]) == 1


# ---------------------------------------------------------------------------
# part 7: "How it's served", and a first estimate from another build
# ---------------------------------------------------------------------------

def _served(rid, name, based_on="Qwen/Qwen3.6-35B-A3B", **kw):
    rec = {"id": rid, "name": name, "base_url": "http://127.0.0.1:9/v1", "how": "k4, --cpu-moe",
           "based_on": based_on, "thinking": "auto", "pin": {"model": "m", "file": "f.gguf"},
           "flags": "--n-cpu-moe 21", "env": "", **kw}
    db.served_put(rec)
    served.write_meta(rec)
    return rec


def test_how_its_served_is_corrected_from_the_page_and_what_it_said_is_kept(svc):
    client, _ = svc
    _served("served/orig", "Qwen3.6 original")
    assert served.launch_check(served.get("served/orig"))["mismatch"] == [
        "“How it’s served” says --cpu-moe; the launch flags registered don’t"]
    r = client.put("/api/served/served/orig/launch", json={"how": "k4, --n-cpu-moe 21",
                                                           "flags": "--n-cpu-moe 21", "env": "",
                                                           "by": ME})
    assert r.status_code == 200, r.text
    rec = served.get("served/orig")
    assert rec["how"] == "k4, --n-cpu-moe 21" and rec["how_was"][0]["how"] == "k4, --cpu-moe"
    assert rec["how_was"][0]["by"] == ME
    assert served.launch_check(rec)["mismatch"] == []
    # a name, and some words, are needed
    assert client.put("/api/served/served/orig/launch", json={"how": "x"}).status_code == 422
    assert client.put("/api/served/served/orig/launch",
                      json={"how": " ", "by": ME}).status_code == 422


def test_a_first_mobile_estimate_uses_another_builds_pace_and_says_whose(svc):
    client, _ = svc
    _served("served/orig", "Qwen3.6 original")
    _served("served/lda", "Qwen3.6 k4-LDA", speed_by={"off": {"secs_each": 0.96, "n": 5000,
                                                               "at": time.time()}})
    _served("served/other", "Another model", based_on="org/other",
            speed_by={"off": {"secs_each": 9.0, "n": 50, "at": time.time()}})
    got = client.get("/api/mobileaibench/estimate", params={"model": "served/orig"}).json()
    assert got["parts"]["none"]["line"] == (
        "5,000 answers, about 80 min, at 1.0 s an answer as Qwen3.6 k4-LDA measured it with "
        "thinking off — another build of the same model; this one hasn’t been measured yet")
    # its own, once measured, comes first
    rec = served.get("served/orig")
    rec["speed_by"] = {"off": {"secs_each": 2.0, "n": 100, "at": time.time()}}
    db.served_put(rec)
    got = client.get("/api/mobileaibench/estimate", params={"model": "served/orig"}).json()
    assert "at its measured 2.0 s an answer with thinking off" in got["parts"]["none"]["line"]
