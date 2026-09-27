"""12f.1: models served elsewhere, against a fake OpenAI-compatible server
(tests/fake_openai.py) — no model runs, and nothing calls OpenRouter.

Registered with what the server reports, pinned; a different file stops the
next run in one line. Everyday and the exam are asked with the settings a
local run uses, from the same functions; the generative three go through
lm_eval's local-chat-completions. A server that stops answering gives a
partial result that says so. Log-likelihood tasks aren't offered, Improve
isn't either, the key never leaves the server, and a served model's scores
are never in another model's average."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import everyday as ev
from conftest import make_service
from fake_openai import LDA_FILE, FakeServer, nothing_listening
from service import chat, config, db, runner, served

NAME = "Qwen3.6-35B-A3B k4-LDA (phone build)"
SID = "served/Qwen3.6-35B-A3B-k4-LDA-phone-build"
HOW = "llama.cpp fork teraformer/lda-2026-09-22 @ 91428471f, --cpu-moe, lookahead 1, fusion off"
BASE = "fx/good-750m"
KEY = "lda-key-3f9a1c77e2"
ME = "masein"


@pytest.fixture
def fake():
    s = FakeServer()
    yield s
    s.close()


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "SERVED_RETRY_S", 0.4)
    # a served run holds the run lock, and never looks for free VRAM
    locks = []
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: locks.append(sid) or True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)

    def no_vram():
        raise AssertionError("a served run waits for no VRAM here")
    monkeypatch.setattr(runner, "gpu_free_mib", no_vram)
    yield client, appmod, locks
    client.__exit__(None, None, None)


def register(client, fake, **over):
    body = {"name": NAME, "base_url": fake.base, "key": "", "based_on": BASE, "how": HOW,
            "thinking": "off", "by": ME, **over}
    return client.post("/api/served", json=body)


def queue(suite, tasks=None):
    return db.add(SID, "instruct", suite, ME, "", tasks=tasks or [])


def model_dir() -> Path:
    return config.OUT_DIR / SID.replace("/", "__")


# ---------------------------------------------------------------------------
# registering
# ---------------------------------------------------------------------------

def test_adding_a_served_model_reads_and_pins_its_details(svc, fake):
    client, _, _ = svc
    # Check: what the server reports, and nothing kept
    r = client.post("/api/served/check", json={"base_url": fake.base})
    assert r.status_code == 200, r.text
    rep = r.json()["reported"]
    assert rep["file"] == "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
    assert (rep["size"], rep["ctx"], rep["build"]) == (22_900_000_000, 16384, "b6500-91428471f")
    assert rep["answered"] == ["models", "props", "health"]
    assert client.get("/api/served").json()["models"] == []
    # Save: the same, pinned with it
    r = register(client, fake)
    assert r.status_code == 200, r.text
    m = r.json()["model"]
    assert m["id"] == SID and m["name"] == NAME and m["how"] == HOW
    assert m["pin"] == {"model": "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf",
                        "file": "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf",
                        "size": 22_900_000_000, "ctx": 16384, "build": "b6500-91428471f"}
    assert [x["id"] for x in client.get("/api/served").json()["models"]] == [SID]
    # the page reads it from its model_meta.json, as it reads any model's
    meta = json.loads((model_dir() / "model_meta.json").read_text())
    assert meta["kind"] == "instruct" and meta["served"]["pin"] == m["pin"]
    # how it's served is required, and a name
    assert register(client, fake, how="  ").status_code == 422
    assert register(client, fake, name="").status_code == 422


def test_nothing_answering_is_said_in_one_line_and_nothing_is_saved(svc):
    client, _, _ = svc
    gone = nothing_listening()
    for path in ("/api/served/check", "/api/served"):
        r = client.post(path, json={"name": NAME, "base_url": gone, "how": HOW})
        assert r.status_code == 422
        assert r.json()["detail"].startswith(f"Nothing answered at {gone}")
        assert "\n" not in r.json()["detail"]
    assert client.get("/api/served").json()["models"] == []
    assert not model_dir().exists()


def test_the_key_is_never_returned_by_any_endpoint(svc, fake, monkeypatch):
    client, _, _ = svc
    fake.key = KEY
    # a server with a key refuses a check without it, in one line
    r = client.post("/api/served/check", json={"base_url": fake.base})
    assert r.status_code == 422 and "refused the key" in r.json()["detail"]
    r = register(client, fake, key=KEY)
    assert r.status_code == 200 and r.json()["model"]["has_key"] is True
    # Everyday, then the generative three through lm_eval's own client
    fake.auth.clear()
    runner.run_submission(db.get(queue("everyday")))
    assert fake.auth and all(h == f"Bearer {KEY}" for h in fake.auth)
    seen = {}

    def lm_eval(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        seen["cmd"], seen["key"] = cmd, env.get("OPENAI_API_KEY")
        lf.write(" ".join(cmd) + "\n")
        return 0
    monkeypatch.setattr(runner, "_run_task", lm_eval)
    runner.run_submission(db.get(queue("generative")))
    assert seen["key"] == KEY and KEY not in " ".join(seen["cmd"])
    # re-registering without the key keeps it
    assert register(client, fake).json()["model"]["has_key"] is True
    said = [client.get("/api/served").text, client.get("/api/results").text,
            client.get("/api/submissions").text,
            client.get("/api/models/suggest", params={"q": "k4-LDA"}).text,
            register(client, fake).text,
            client.post("/api/served/check", json={"name": NAME, "base_url": fake.base}).text]
    for f in config.OUT_DIR.rglob("*"):
        if f.is_file():
            said.append(f.read_text(encoding="utf-8", errors="replace"))
    for f in config.LOGS_DIR.rglob("*"):
        if f.is_file():
            said.append(f.read_text(encoding="utf-8", errors="replace"))
    assert not [s for s in said if KEY in s]


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------

def test_a_changed_model_file_stops_the_next_run_with_the_line(svc, fake):
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    fake.model_path = "/home/masein/Qwen3.6-35B-A3B-k4-LDA-UD-Q5_K_M.gguf"
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed"
    assert row["error"] == ("The server now serves a different file than the one registered. "
                            "Register it again if that's intended.")
    assert fake.requests == []                                # not one question asked
    # registered again, the next run goes ahead
    assert register(client, fake).status_code == 200
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done"


def test_everyday_goes_through_the_shared_settings_function(svc, fake, monkeypatch):
    client, _, locks = svc
    assert register(client, fake, thinking="on").status_code == 200
    calls = []
    real = ev.run_settings

    def spy(arch):
        calls.append(arch)
        return real(arch)
    monkeypatch.setattr(ev, "run_settings", spy)
    fake.reasoning = "Let me think about it."
    progress = []
    fake.on_request = lambda body: progress.append(db.get(sid)["progress"])
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert locks == [sid]                                     # the same run lock
    # one message a question, with a local run's settings for a thinking model
    want = real({"reasoning_template": True})
    assert want["max_gen_toks"] == config.EVERYDAY_REASONING_MAX_GEN_TOKS
    assert len(fake.requests) == len(ev.load_bank())
    for body in fake.requests:
        assert body["max_tokens"] == want["max_gen_toks"] and body["stop"] == want["until"]
        assert body["temperature"] == want["temperature"]
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
    assert any(a and a.get("reasoning_template") for a in calls)
    # the prompts are the bank's, as the task's doc_to_text gives them
    bank = {q["id"]: q["prompt"] for q in ev.load_bank()}
    assert {b["messages"][0]["content"] for b in fake.requests} == set(bank.values())
    # marked like any model's answers, the thinking put back in its tags
    out = json.loads((model_dir() / "everyday.json").read_text())
    assert len(out["items"]) == len(bank) and out["unasked"] == 0
    assert all(it.get("had_reasoning") for it in out["items"] if it.get("answer_text"))
    # each answer records how it was asked: the pinned file and the settings
    [samples] = model_dir().glob("everyday_0shot/served/samples_everyday_*.jsonl")
    first = json.loads(samples.read_text().splitlines()[0])
    assert first["served"]["pin"]["file"] == "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
    assert first["served"]["settings"]["max_tokens"] == want["max_gen_toks"]
    # progress with the time left, from the seconds each answer took
    assert any(re.search(r"\d+ of 340 · [\d.]+ s an answer · (about|under)", p or "")
               for p in progress)
    # on Models: its own row, called what it was registered as
    rows = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    assert rows[SID]["name"] == NAME and rows[SID]["served"]["how"] == HOW


def test_the_knowledge_exam_goes_through_the_shared_settings_function(svc, fake, monkeypatch):
    client, _, _ = svc
    assert register(client, fake, thinking="off").status_code == 200
    calls = []
    real = runner._exam_settings

    def spy(meta):
        calls.append(meta)
        return real(meta)
    monkeypatch.setattr(runner, "_exam_settings", spy)
    task = config.judged_tasks()[0]
    items = [json.loads(x) for x in (config.JUDGED_TASKS_DIR / f"{task}.jsonl").read_text()
             .splitlines() if x.strip()]
    sid = queue("judged", [task])
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert calls and len(fake.requests) == len(items)
    for body, it in zip(fake.requests, items):
        assert body["messages"][0]["content"] == it["prompt"] + "\n\nAnswer:"
        assert body["max_tokens"] == runner.EXAM_MAX_GEN_TOKS and body["stop"] == ["\n\n\n"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # the judge marked them, as any model's
    j = json.loads((model_dir() / "judge.json").read_text())
    assert task in j["tasks"]
    rows = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    assert task in (rows[SID]["judge"] or {}).get("tasks", {})


def test_a_server_that_stops_mid_run_gives_a_partial_labelled_result(svc, fake):
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    fake.stop_after = 140
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    n = len(ev.load_bank())
    assert row["status"] == "failed"
    assert row["error"] == (f"everyday: the server stopped answering at 140 of {n} · the 140 "
                            f"answered are kept and marked")
    # what it answered is kept and marked; the rest is what it hasn't been asked
    out = json.loads((model_dir() / "everyday.json").read_text())
    assert len(out["items"]) == 140 and out["unasked"] == n - 140
    [res] = model_dir().glob("everyday_0shot/served/results_*.json")
    blob = json.loads(res.read_text())
    assert blob["n-samples"]["everyday"] == {"original": n, "effective": 140}
    assert blob["served"]["partial"] == {"answered": 140, "of": n}
    # back up: the next run asks only the rest
    fake.stop_after = None
    fake.requests.clear()
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done"
    assert len(fake.requests) == n - 140


def test_requests_go_one_or_two_at_a_time(svc, fake, monkeypatch):
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    fake.delay_s = 0.01
    # a topic each: one it has answered is not asked again
    for n, task in zip((1, 2), config.judged_tasks()):
        monkeypatch.setattr(config, "SERVED_CONCURRENCY", n)
        fake.max_in_flight = 0
        runner.run_submission(db.get(queue("judged", [task])))
        assert fake.max_in_flight == n


def test_the_generative_three_go_through_lm_evals_local_chat_completions(svc, fake, monkeypatch):
    client, _, _ = svc
    assert register(client, fake, key=KEY).status_code == 200
    import generative
    monkeypatch.setattr(generative, "mark", lambda *a, **k: None)
    seen = []

    def lm_eval(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        seen.append(cmd)
        out = Path(cmd[cmd.index("--output_path") + 1]) / "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
        out.mkdir(parents=True, exist_ok=True)
        task = cmd[cmd.index("--tasks") + 1]
        margs = cmd[cmd.index("--model_args") + 1]
        (out / "results_2026-09-27T10-00-00.000000.json").write_text(json.dumps({
            "results": {task: {"alias": task, "exact_match,none": 0.5}},
            "config": {"model": "local-chat-completions", "model_args": margs},
            "n-samples": {task: {"original": 10, "effective": 10}}, "date": 1.0}))
        return 0
    monkeypatch.setattr(runner, "_run_task", lm_eval)
    sid = queue("generative")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert [c[c.index("--tasks") + 1] for c in seen] == list(config.GEN_TASKS)
    for cmd in seen:
        assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
        assert "--device" not in cmd and cmd[cmd.index("--batch_size") + 1] == "1"
        assert "--apply_chat_template" in cmd and "--use_cache" in cmd
        margs = cmd[cmd.index("--model_args") + 1]
        assert f"base_url={fake.base}/chat/completions" in margs
        assert "num_concurrent=1" in margs and KEY not in margs
    # its results are the served row's, and say how it was served
    rows = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    assert rows[SID]["name"] == NAME
    assert json.loads(next(model_dir().glob("ifeval_0shot/*/results_*.json")).read_text()
                      )["served"]["how"] == HOW


def test_a_server_gone_under_lm_eval_says_where_and_keeps_its_answers(svc, fake, monkeypatch):
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    progress = []

    def lm_eval(sid, cmd, lf, env, run_as, cwd, on_poll=None):
        lf.write("Requesting API:  26%|██▌       | 140/541 [10:02<28:44,  4.30s/it]\n")
        lf.flush()
        on_poll()
        progress.append(db.get(sid)["progress"])
        lf.write("requests.exceptions.ConnectionError: HTTPConnectionPool: "
                 "[Errno 111] Connection refused\n")
        return 1
    monkeypatch.setattr(runner, "_run_task", lm_eval)
    sid = queue("generative")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed"
    assert row["error"] == ("ifeval: the server stopped answering at 140 of 541 · the answers "
                            "it gave are kept: the next run asks only the rest")
    assert progress == ["1/3 · ifeval (0-shot) · 140 of 541 · 4.3 s an answer · about 29 min left"]


def test_time_left_is_the_seconds_each_answer_took():
    assert runner.time_left(140, 200, 4.1) == "140 of 200 · 4.1 s an answer · about 4 min left"
    assert runner.time_left(10, 12000, 2.0) == "10 of 12000 · 2.0 s an answer · about 6.7 h left"
    assert runner.time_left(199, 200, 4.0) == "199 of 200 · 4.0 s an answer · under a minute left"


# ---------------------------------------------------------------------------
# what it is not offered
# ---------------------------------------------------------------------------

def test_log_likelihood_tasks_are_not_offered_and_the_line_says_why(svc, fake):
    client, _, _ = svc
    line = "Multiple-choice benchmarks need the model loaded here; this one is served elsewhere."
    r = client.post("/api/submissions", json={"hf_id": SID, "suite": "everyday"})
    assert r.status_code == 422 and "is not registered" in r.json()["detail"]
    assert register(client, fake).status_code == 200
    for suite in ("full", "quick", "control"):
        r = client.post("/api/submissions", json={"hf_id": SID, "suite": suite})
        assert r.status_code == 422 and r.json()["detail"] == line + " Nothing was queued."
    assert client.get("/api/served").json()["line"] == line
    # queued some other way, the run says the same at its start
    sid = db.add(SID, "instruct", "full", ME, "")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["error"] == line and fake.requests == []
    # the three it can sit are queued
    for suite in ("everyday", "generative"):
        assert client.post("/api/submissions", json={"hf_id": SID, "suite": suite}).status_code == 200


def test_improve_leaves_it_out_and_the_playground_chats_through_its_server(svc, fake):
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    r = client.post("/api/proposals", json={"model": SID, "topic": "Economics", "by": ME})
    assert r.status_code == 422 and "served elsewhere" in r.json()["detail"]
    # 12d.3: the Playground chats with it, through its server, once
    rows = [m for m in chat.board_models() if m["id"] == SID]
    assert len(rows) == 1 and rows[0]["served"] and rows[0]["chat"]


def test_the_compose_file_lets_the_container_reach_the_host():
    text = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text()
    # the bench service's block: from its name to the next service at its indent
    bench = re.search(r"(?ms)^  bench:\n(.*?)(?=^  \S|\Z)", text).group(1)
    assert re.search(r'(?m)^    extra_hosts:\n      - "host\.docker\.internal:host-gateway"$',
                     bench)


def test_a_served_models_score_is_never_in_another_models_average(svc, fake):
    client, appmod, _ = svc
    j0 = client.get("/api/results").json()
    before = {m["id"]: m for m in j0["models"]}
    assert register(client, fake).status_code == 200
    runner.run_submission(db.get(queue("everyday")))
    # and a Standard score of its own, from the generative three
    d = model_dir() / "ifeval_0shot" / "served"
    d.mkdir(parents=True)
    (d / "results_2026-09-27T10-00-00.000000.json").write_text(json.dumps({
        "results": {"ifeval": {"alias": "ifeval", "prompt_level_strict_acc,none": 0.99}},
        "config": {"model": "local-chat-completions", "model_args": f"pretrained={SID},x=1"},
        "served": served.view(served.get(SID)), "date": 2.0}))
    appmod._cache.update(key=None, payload=None, at=0.0)
    j = client.get("/api/results").json()
    after = {m["id"]: m for m in j["models"]}
    for k in ("avg", "avgRaw", "partialAvg", "nhave", "judgedAvg"):
        assert after[BASE][k] == before[BASE][k], k
    assert {t: c[BASE] for t, c in j["cells"].items() if BASE in c} == \
        {t: c[BASE] for t, c in j0["cells"].items() if BASE in c}
    assert after[SID]["avg"] is None and after[SID]["served"]["based_on"] == BASE
    assert j["everyday"]["models"][BASE] == j0["everyday"]["models"][BASE]
    assert SID in j["everyday"]["models"]
    assert j["served"][SID]["name"] == NAME
    assert LDA_FILE.endswith(after[SID]["served"]["pin"]["file"])
