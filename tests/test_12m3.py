"""12m.3: frontier models measured here, through OpenRouter — against the fake
OpenAI-compatible server (tests/fake_openai.py), which answers at
OpenRouter's address for these tests: every urllib request in the process
goes through `outside` below, OpenRouter's is sent to the fake, and any other
address outside this machine fails the test. Nothing calls OpenRouter, and
no model runs.

A model from OpenRouter's list becomes a served entry pinned as the judge is
(its dated version and first provider, no fallbacks), checked at every run's
start. The estimate shows before Start and a run that would pass the month's
AI limit is refused; each answer's cost, thinking included, counts toward the
limit as it lands, the running total is in the progress, and a run stops at
the limit keeping its answers. The key is never in any reply, file or log."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

import everyday as ev
from conftest import make_service
from fake_openai import FakeServer
from service import chat, config, db, playground, runner, served

OR = "https://openrouter.ai/api/v1"
KEY = "sk-or-12m3-not-a-real-key"
ME = "masein"
LUNA = "openai/gpt-6-luna"
LUNA_V = "openai/gpt-6-luna-20260922"
SID = "served/openrouter-openai-gpt-6-luna"
PLAIN = "mistralai/mistral-medium-4"          # one that doesn't think
PLAIN_SID = "served/openrouter-mistralai-mistral-medium-4"


def _m(mid, version, name, pin, pout, reasons=True):
    return {"id": mid, "canonical_slug": version, "name": name, "context_length": 400000,
            "pricing": {"prompt": pin, "completion": pout},
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["max_tokens", "temperature", "stop"]
            + (["reasoning", "include_reasoning"] if reasons else [])}


CATALOG = [
    _m(LUNA, LUNA_V, "OpenAI: GPT-6 Luna", "0.000001", "0.000008"),
    _m("google/gemini-4-flash", "google/gemini-4-flash-20260901", "Google: Gemini 4 Flash",
       "0.0000003", "0.0000025"),
    _m("anthropic/claude-sonnet-5.1", "anthropic/claude-sonnet-5.1-20260915",
       "Anthropic: Claude Sonnet 5.1", "0.000003", "0.000015"),
    _m(PLAIN, "mistralai/mistral-medium-4-20260801", "Mistral: Mistral Medium 4",
       "0.0000001", "0.0000003", reasons=False),
]
ENDPOINTS = [
    {"provider_name": "OpenAI", "tag": "openai", "quantization": "unknown", "status": 0,
     "pricing": {"prompt": "0.000001", "completion": "0.000008"}},
    {"provider_name": "Azure", "tag": "azure", "quantization": "unknown", "status": 0,
     "pricing": {"prompt": "0.000001", "completion": "0.000008"}},
]


@pytest.fixture
def fake():
    s = FakeServer()
    s.key = KEY
    s.catalog = [dict(m) for m in CATALOG]
    s.endpoints = [dict(e) for e in ENDPOINTS]
    s.provider = "OpenAI"
    yield s
    s.close()


@pytest.fixture
def outside(monkeypatch, fake):
    """OpenRouter's address answered by the fake server; any other address
    outside this machine is refused and fails the test"""
    real = urllib.request.urlopen
    tried: list[str] = []

    def urlopen(req, *a, **k):
        if not isinstance(req, urllib.request.Request):
            req = urllib.request.Request(req)
        if req.full_url.startswith(OR + "/"):
            req.full_url = fake.base + req.full_url[len(OR):]
        if urllib.parse.urlsplit(req.full_url).hostname not in ("127.0.0.1", "localhost"):
            tried.append(req.full_url)
            raise urllib.error.URLError(f"a test reached outside: {req.full_url}")
        return real(req, *a, **k)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    yield tried
    assert tried == []


@pytest.fixture
def svc(tmp_path, monkeypatch, outside):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(config, "OPENROUTER_BASE_URL", OR)
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    monkeypatch.setattr(config, "SERVED_RETRY_S", 0.4)
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)

    def no_vram():
        raise AssertionError("a served run waits for no VRAM here")
    monkeypatch.setattr(runner, "gpu_free_mib", no_vram)
    yield client, appmod
    client.__exit__(None, None, None)


def add(client, model=LUNA):
    r = client.post("/api/served/openrouter", json={"model": model, "by": ME})
    assert r.status_code == 200, r.text
    return r.json()["model"]


def queue(suite, tasks=None, sid=SID, subset=0):
    return db.add(sid, "instruct", suite, ME, "", tasks=tasks or [], subset=subset)


def run(sid):
    runner.run_submission(db.get(sid))
    return db.get(sid)


def model_dir(sid=SID) -> Path:
    return config.OUT_DIR / sid.replace("/", "__")


def exam_task():
    task = config.judged_tasks()[0]
    items = [json.loads(x) for x in (config.JUDGED_TASKS_DIR / f"{task}.jsonl").read_text()
             .splitlines() if x.strip()]
    return task, items


def limit(usd):
    db.ai_set("spend_limit", usd, ME)


# ---------------------------------------------------------------------------
# adding one, and its pin
# ---------------------------------------------------------------------------

def test_adding_a_model_from_openrouter_pins_it_as_the_judge_is(svc, fake, monkeypatch):
    client, _ = svc
    m = add(client)
    assert m["id"] == SID and m["name"] == "GPT-6 Luna" and m["via"] == "openrouter"
    # OpenRouter's address, as a string: nothing was asked there but its lists
    assert m["base_url"] == OR and fake.requests == []
    assert m["maker"] == "OpenAI"
    # the dated id and the first provider, with that provider's prices
    assert m["pin"] == {"model": LUNA, "version": LUNA_V, "provider": "openai",
                        "provider_name": "OpenAI", "precision": "unknown",
                        "price_in": 1.0, "price_out": 8.0, "ctx": 400000}
    assert m["how"] == f"OpenRouter · {LUNA_V} · on OpenAI · no fallbacks"
    # it thinks, as OpenRouter says: the thinking budgets; one that doesn't, not
    assert m["thinking"] == "auto" and add(client, PLAIN)["thinking"] == "off"
    got = client.get("/api/served").json()
    assert {x["id"] for x in got["models"]} == {SID, PLAIN_SID}
    assert got["openrouter"] == {"has_key": True, "subset": config.OPENROUTER_GEN_SUBSET}
    # the page reads it from its model_meta.json, as any served model's
    meta = json.loads((model_dir() / "model_meta.json").read_text())
    assert meta["served"]["via"] == "openrouter" and meta["served"]["maker"] == "OpenAI"
    assert meta["served"]["pin"]["version"] == LUNA_V
    # one line when it can't be added
    r = client.post("/api/served/openrouter", json={"model": "acme/nothing", "by": ME})
    assert r.status_code == 422 and r.json()["detail"] == \
        "acme/nothing is not one of OpenRouter's text models"
    assert client.post("/api/served/openrouter", json={"model": LUNA}).status_code == 422
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    r = client.post("/api/served/openrouter", json={"model": LUNA, "by": ME})
    assert r.status_code == 422 and r.json()["detail"] == \
        "OpenRouter has no key on this server (OPENROUTER_API_KEY)"


def test_every_question_carries_the_pin_and_a_changed_pin_stops_the_next_run(svc, fake):
    client, _ = svc
    add(client)
    task, items = exam_task()
    row = run(queue("judged", [task]))
    assert row["status"] == "done", row["error"]
    assert len(fake.requests) == len(items)
    for body in fake.requests:
        assert body["model"] == LUNA
        assert body["provider"] == {"order": ["openai"], "allow_fallbacks": False}
        assert body["usage"] == {"include": True}
        assert "chat_template_kwargs" not in body         # OpenRouter has no template switch
        assert body["max_tokens"] == config.REASONING_MAX_GEN_TOKS    # it thinks
    # with AI models' key, at OpenRouter's address (answered by the fake)
    assert set(fake.auth) == {f"Bearer {KEY}"}
    # each answer records the pin it was asked under
    [samples] = model_dir().glob(f"{task}_0shot/served/samples_{task}_*.jsonl")
    first = json.loads(samples.read_text().splitlines()[0])
    assert first["served"]["pin"]["version"] == LUNA_V and first["served"]["via"] == "openrouter"

    # the id moved to another dated version: the next run stops at its start
    fake.catalog[0] = dict(fake.catalog[0], canonical_slug="openai/gpt-6-luna-20261001")
    fake.requests.clear()
    task2 = config.judged_tasks()[1]
    row = run(queue("judged", [task2]))
    assert row["status"] == "failed" and fake.requests == []
    assert row["error"] == ("openai/gpt-6-luna now points to openai/gpt-6-luna-20261001, not the "
                            "openai/gpt-6-luna-20260922 it was added with — add it again under "
                            "Test a model ▸ A model from OpenRouter if that's intended")
    # added again, it is pinned to what OpenRouter lists now
    assert add(client)["pin"]["version"] == "openai/gpt-6-luna-20261001"
    # the pinned provider no longer runs it: no fallbacks, so the run stops
    fake.endpoints = [dict(ENDPOINTS[0], status=-1), ENDPOINTS[1]]
    row = run(queue("judged", [task2]))
    assert row["status"] == "failed" and fake.requests == []
    assert row["error"] == ("OpenAI doesn't run openai/gpt-6-luna on OpenRouter now, and it is "
                            "pinned there with no fallbacks — add it again under Test a model ▸ "
                            "A model from OpenRouter if that's intended")
    fake.endpoints = [dict(e) for e in ENDPOINTS]
    row = run(queue("judged", [task2]))
    assert row["status"] == "done", row["error"]


def test_the_key_is_never_returned_by_any_endpoint(svc, fake, monkeypatch):
    client, _ = svc
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)            # the service's own environment
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    said = [client.post("/api/served/openrouter", json={"model": LUNA, "by": ME}).text]
    limit(1000.0)
    task, _ = exam_task()
    assert run(queue("judged", [task]))["status"] == "done"
    import generative
    monkeypatch.setattr(generative, "mark", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(runner, "_run_task", lm_eval_asking(3, seen))
    row = run(queue("generative", subset=100))
    assert row["status"] == "done", row["error"]
    # lm_eval's child: no key in its environment or its command — its relay has it
    for s in seen:
        assert "OPENAI_API_KEY" not in s["env"] and "OPENROUTER_API_KEY" not in s["env"]
        assert KEY not in " ".join(s["cmd"])
        assert s["margs"]["base_url"].startswith("http://127.0.0.1:")
    said += [client.get(p).text for p in (
        "/api/served", "/api/results", "/api/submissions", "/api/ai", "/api/ai/models",
        "/api/llm", "/api/playground", f"/api/runs/{row['id']}/log")]
    said.append(client.post("/api/served/estimate", json={"model": SID, "suite": "everyday"}).text)
    for d in (config.OUT_DIR, config.LOGS_DIR):
        said += [f.read_text(encoding="utf-8", errors="replace")
                 for f in d.rglob("*") if f.is_file()]
    assert said and not [s for s in said if KEY in s]
    # and the page's registry row never kept it
    assert served.get(SID)["key"] == ""


# ---------------------------------------------------------------------------
# the estimate, and the limit at the start
# ---------------------------------------------------------------------------

def test_tokens_are_counted_by_the_rough_rule():
    assert [served.tokens_of(t) for t in ("", "abcd", "abcde", "x" * 400)] == [0, 1, 2, 100]
    assert served.about(0.4) == "about $0.40" and served.about(1240) == "about $1,240.00"
    assert served.about(0.001) == "under $0.01"


def test_the_estimate_is_the_questions_tokens_at_the_models_prices(svc, fake):
    client, _ = svc
    add(client)
    add(client, PLAIN)

    def est(**body):
        r = client.post("/api/served/estimate", json=body)
        assert r.status_code == 200, r.text
        return r.json()
    # the Knowledge exam: each prompt as sent, a token a four characters, and
    # each answer at its whole budget — a thinking model's
    task, items = exam_task()
    tin = sum(-(-len(it["prompt"] + "\n\nAnswer:") // 4) for it in items)
    tout = len(items) * config.REASONING_MAX_GEN_TOKS
    want = (tin * 1.0 + tout * 8.0) / 1e6
    got = est(model=SID, suite="judged", tasks=[task])
    assert (got["n"], got["tokens_in"], got["tokens_out"]) == (len(items), tin, tout)
    assert got["usd"] == round(want, 2) and got["line"] == f"about ${want:,.2f}"
    assert got["limit"] == 20.0 and got["month"] == 0 and got["refused"] == ""
    # Everyday: the questions it has no answer to, at the thinking budget
    bank = ev.load_bank()
    tin = sum(-(-len(q["prompt"]) // 4) for q in bank)
    tout = len(bank) * config.EVERYDAY_REASONING_MAX_GEN_TOKS
    got = est(model=SID, suite="everyday")
    assert (got["n"], got["tokens_in"], got["tokens_out"]) == (len(bank), tin, tout)
    assert got["usd"] == round((tin + tout * 8.0) / 1e6, 2)
    # a model that doesn't think gets the plain budgets
    got = est(model=PLAIN_SID, suite="judged", tasks=[task])
    assert got["tokens_out"] == len(items) * runner.EXAM_MAX_GEN_TOKS
    assert est(model=PLAIN_SID, suite="everyday")["tokens_out"] == len(bank) * ev.MAX_GEN_TOKS
    # the generative three: lm_eval's prompts, said by size; a seeded MMLU-Pro subset
    got = est(model=SID, suite="generative", subset=1000)
    n = 541 + 1000 + 500
    assert got["n"] == n and got["tokens_in"] == 541 * 90 + 1000 * 2300 + 500 * 110
    assert got["tokens_out"] == n * config.GEN_THINKING_MAX_GEN_TOKS
    assert est(model=PLAIN_SID, suite="generative")["n"] == 541 + 12032 + 500
    # a task already answered isn't counted: the run won't ask it
    assert run(queue("judged", [task]))["status"] == "done"
    assert est(model=SID, suite="judged", tasks=[task])["n"] == 0
    # only for a model from OpenRouter, and only what a served model can sit
    assert client.post("/api/served/estimate", json={"model": "fx/good-750m"}).status_code == 422
    assert client.post("/api/served/estimate",
                       json={"model": SID, "suite": "full"}).status_code == 422


def test_a_run_that_would_pass_the_limit_is_refused_in_one_line(svc, fake):
    client, _ = svc
    add(client)
    limit(1.0)
    db.spend_add("judge", "deepseek/deepseek-v4.1-flash", "InferenceNet", 0, 0, 0.95)
    est = client.post("/api/served/estimate", json={"model": SID, "suite": "everyday"}).json()
    line = (f"This run could cost {est['line']}, more than the $0.05 left of this month's $1.00 "
            f"AI limit — test fewer questions, or raise the limit on AI models.")
    assert est["refused"] == line and est["left"] == 0.05
    r = client.post("/api/submissions", json={"hf_id": SID, "suite": "everyday",
                                              "submitter": ME})
    assert r.status_code == 409 and r.json()["detail"] == line + " Nothing was queued."
    assert "\n" not in r.json()["detail"]
    assert not [x for x in db.recent(50) if x["hf_id"] == SID]
    # queued some other way, it is refused at its start, asking nothing
    row = run(queue("everyday"))
    assert row["status"] == "failed" and row["error"] == line and fake.requests == []
    # with room, it is queued
    limit(100.0)
    r = client.post("/api/submissions", json={"hf_id": SID, "suite": "everyday",
                                              "submitter": ME})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# the running total, and the stop at the limit
# ---------------------------------------------------------------------------

def test_the_running_total_stops_a_run_at_the_limit_and_keeps_the_answers(svc, fake):
    client, _ = svc
    add(client)
    n = len(ev.load_bank())
    # each answer costs a dollar, as OpenRouter reports it: at $20 a month,
    # the 21st question could pass the limit
    fake.cost = 1.0
    progress = []
    fake.on_request = lambda body: progress.append(db.get(sid)["progress"])
    sid = queue("everyday")
    row = run(sid)
    assert row["status"] == "failed"
    assert row["error"] == (f"everyday: stopped at 20 of {n}: the next question could pass this "
                            f"month's $20.00 AI limit · $20.00 spent on this run · the 20 "
                            f"answered are kept and marked")
    assert len(fake.requests) == 20
    # counted toward the month, under Test a model's job, and never past the limit
    assert db.spend_this_month() == pytest.approx(20.0)
    assert db.spend_of_batch(f"run-{sid}") == pytest.approx(20.0)
    ai = client.get("/api/ai").json()["spend"]
    assert ai["by_job"] == {"tests": pytest.approx(20.0)} and ai["month"] == 20.0
    # the running total in the progress, and the row's total at the end
    assert any(re.search(r"· 19 of \d+ · [\d.]+ s an answer · [^·]+ · \$19\.00 of about "
                         r"\$[\d,.]+ so far · limit \$20\.00, \$1\.00 left$", p or "")
               for p in progress), progress[-3:]
    assert row["progress"].endswith("$20.00 on OpenRouter")
    # what it answered is kept and marked; the rest waits
    out = json.loads((model_dir() / "everyday.json").read_text())
    assert len(out["items"]) == 20 and out["unasked"] == n - 20
    # the limit raised, the next run asks only the rest
    limit(100.0)
    fake.cost = 0.001
    fake.requests.clear()
    row = run(queue("everyday"))
    assert row["status"] == "done", row["error"]
    assert len(fake.requests) == n - 20


def test_reasoning_tokens_are_counted_in_the_total(svc, fake):
    client, _ = svc
    add(client)
    # no cost in the reply: its tokens at the pinned prices, thinking included —
    # OpenRouter's completion_tokens hold the reasoning tokens
    fake.reasoning = "Let me think about this carefully before I answer it."
    fake.reasoning_field = "reasoning"                    # as OpenRouter says it
    fake.reasoning_tokens = lambda body: 10
    task, items = exam_task()
    sid = queue("judged", [task])
    assert run(sid)["status"] == "done"
    think, text = 10, len("A plain answer, from the served model.".split())
    each = (10 * 1.0 + (think + text) * 8.0) / 1e6
    assert db.spend_of_batch(f"run-{sid}") == pytest.approx(len(items) * each)
    # the thinking is kept in its tags, as a local run's is
    [samples] = model_dir().glob(f"{task}_0shot/served/samples_{task}_*.jsonl")
    first = json.loads(samples.read_text().splitlines()[0])
    assert first["resps"][0][0].startswith("<think>\nLet me think about this carefully")
    assert first["tokens"] == think + text
    pin = served.get(SID)["pin"]
    # reasoning reported apart from the completion is added to it
    assert served.reply_cost(pin, {"prompt_tokens": 10, "completion_tokens": 7,
                                   "completion_tokens_details": {"reasoning_tokens": 50}}) == \
        (10, 57, pytest.approx((10 + 57 * 8) / 1e6))
    # inside it, counted once
    assert served.reply_cost(pin, {"prompt_tokens": 10, "completion_tokens": 57,
                                   "completion_tokens_details": {"reasoning_tokens": 50}})[1] == 57
    # OpenRouter's own cost, when it reports one
    assert served.reply_cost(pin, {"prompt_tokens": 10, "completion_tokens": 57,
                                   "cost": 0.0123})[2] == 0.0123


# ---------------------------------------------------------------------------
# IFEval, MMLU-Pro and MATH-500: lm_eval through the board's relay
# ---------------------------------------------------------------------------

def lm_eval_asking(n_items, seen):
    """lm_eval's local-chat-completions, as far as a run sees it: `n_items`
    requests a task to the base_url it is given, its progress bar, and
    on_poll between answers; a refusal is retried until the runner stops it"""
    def run_task(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        margs = dict(kv.split("=", 1) for kv in cmd[cmd.index("--model_args") + 1].split(","))
        task = cmd[cmd.index("--tasks") + 1]
        budget = int(cmd[cmd.index("--gen_kwargs") + 1].split("=", 1)[1])
        seen.append({"cmd": cmd, "env": env, "margs": margs})
        for i in range(n_items):
            req = urllib.request.Request(margs["base_url"], method="POST", data=json.dumps({
                "model": margs["model"], "max_tokens": budget, "temperature": 0, "seed": 1234,
                "messages": [{"role": "user", "content": f"{task} question {i}"}]}).encode(),
                headers={"content-type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=10).read()
            except urllib.error.HTTPError as e:
                lf.write(f"HTTPError: {e.code} — retrying\n")
                lf.flush()
                why = on_poll()
                assert why, "the runner stops lm_eval at the limit"
                lf.write(f"\n[service] {why}\n")
                return -1
            lf.write(f"Requesting API: {i + 1}/{n_items} [00:{i + 1:02d}<00:10,  1.00s/it]\n")
            lf.flush()
            on_poll()
        out = Path(cmd[cmd.index("--output_path") + 1]) / margs["model"].replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        (out / "results_2026-09-28T10-00-00.000000.json").write_text(json.dumps({
            "results": {task: {"alias": task, "exact_match,none": 0.5}},
            "config": {"model": "local-chat-completions",
                       "model_args": cmd[cmd.index("--model_args") + 1]},
            "n-samples": {task: {"original": n_items, "effective": n_items}}, "date": 1.0}))
        return 0
    return run_task


def test_the_generative_three_are_metered_through_the_relay(svc, fake, monkeypatch):
    client, _ = svc
    fake.endpoints = [{"provider_name": "Mistral", "tag": "mistral", "quantization": "fp8",
                       "status": 0, "pricing": {"prompt": "0.0000001", "completion": "0.0000003"}}]
    add(client, PLAIN)
    import generative
    monkeypatch.setattr(generative, "mark", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(runner, "_run_task", lm_eval_asking(4, seen))
    fake.cost = 0.25
    sid = queue("generative", sid=PLAIN_SID, subset=200)
    row = run(sid)
    assert row["status"] == "done", row["error"]
    assert [s["cmd"][s["cmd"].index("--tasks") + 1] for s in seen] == list(config.GEN_TASKS)
    # every request reached OpenRouter (the fake) pinned, with the key the relay added
    assert len(fake.requests) == 12
    for body in fake.requests:
        assert body["model"] == PLAIN
        assert body["provider"] == {"order": ["mistral"], "allow_fallbacks": False}
        assert body["usage"] == {"include": True}
    assert set(fake.auth) == {f"Bearer {KEY}"}
    for s in seen:
        margs = s["cmd"][s["cmd"].index("--model_args") + 1]
        assert s["margs"]["model"] == PLAIN and "num_concurrent=1" in margs
    [mmlu] = [s["cmd"] for s in seen if "mmlu_pro" in s["cmd"]]
    assert "--samples" in mmlu                            # the seeded subset
    assert db.spend_of_batch(f"run-{sid}") == pytest.approx(3.0)
    assert row["progress"].endswith("$3.00 on OpenRouter")
    # at the limit the relay sends nothing on, and the runner stops lm_eval:
    # $3 spent, and room for three more at 30 cents but not a fourth
    limit(3.9003)
    fake.requests.clear()
    seen.clear()
    import shutil
    for d in model_dir(PLAIN_SID).glob("*shot"):           # asked afresh
        shutil.rmtree(d)
    fake.cost = 0.3
    sid = queue("generative", sid=PLAIN_SID, subset=100)
    row = run(sid)
    assert row["status"] == "failed"
    assert len(fake.requests) == 3
    assert row["error"] == ("ifeval: stopped at 3 of 4: the next question could pass this "
                            "month's $3.90 AI limit · $0.90 spent on this run · the answers it "
                            "gave are kept: the next run asks only the rest")
    assert db.spend_this_month() == pytest.approx(3.9)
    assert row["progress"] == "failed on: ifeval · $0.90 on OpenRouter"
    # the relay is gone with the run
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(seen[0]["margs"]["base_url"], data=b"{}", timeout=2)


def test_a_child_is_stopped_when_its_poll_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "cancel_requested", lambda sid: False)
    log = tmp_path / "log"
    with open(log, "w") as lf:
        status = runner._run_task(1, [sys.executable, "-c", "import time; time.sleep(60)"], lf,
                                  {}, None, tmp_path,
                                  on_poll=lambda: "stopped at this month's AI limit")
    assert status == -1
    assert "[service] stopped at this month's AI limit" in log.read_text()


# ---------------------------------------------------------------------------
# where it isn't offered
# ---------------------------------------------------------------------------

def test_it_sits_what_a_served_model_sits_and_not_the_playground(svc, fake):
    client, _ = svc
    add(client)
    r = client.post("/api/submissions", json={"hf_id": SID, "suite": "full"})
    assert r.status_code == 422 and "served elsewhere" in r.json()["detail"]
    [row] = [m for m in chat.board_models() if m["id"] == SID]
    assert row["chat"] is False
    pg = playground.models()
    assert SID not in {m["id"] for m in pg["models"]}
    assert playground.OPENROUTER_LINE in pg["left_out"]
    r = client.post("/api/proposals", json={"model": SID, "topic": "Economics", "by": ME})
    assert r.status_code == 422 and "served elsewhere" in r.json()["detail"]
    # OpenRouter typed in as a server of ours would go unmetered: refused, nothing asked
    n = len(fake.auth)
    for path in ("/api/served/check", "/api/served"):
        r = client.post(path, json={"name": "GPT by hand", "base_url": OR, "key": KEY,
                                    "how": "OpenRouter", "by": ME})
        assert r.status_code == 422 and r.json()["detail"] == (
            "That is OpenRouter: add its models under Test a model ▸ A model from OpenRouter, "
            "where what they cost is counted")
    assert len(fake.auth) == n and served.get("served/GPT-by-hand") is None
