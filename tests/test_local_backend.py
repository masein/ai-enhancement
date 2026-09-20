"""Phase 8 P1: a `local` provider — vLLM's OpenAI-compatible server presented
through the batch interface — and the provisional stamp on everything a local
identity produces.

Every test here talks to a stub HTTP server on 127.0.0.1, never to vLLM: CI
has no GPU, and none of this may need one."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

import exam_build as eb
import judge as jd
import report_lm_eval as report
import vllm_stub
from conftest import fresh, make_service
from service import config, db, llm, llm_poller, proposals

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = vllm_stub.WEIGHTS
REASON = "graded by a local model — not a pinned benchmark"


@pytest.fixture
def vllm(monkeypatch):
    stub, srv = vllm_stub.serve()
    monkeypatch.setattr(config, "LOCAL_BASE_URL", stub.url)
    monkeypatch.setattr(config, "LOCAL_CONCURRENCY", 2)
    monkeypatch.setattr(config, "LOCAL_MAX_TOKENS", 1024)
    monkeypatch.setattr(llm.LocalOpenAI, "BACKOFF", (0, 0, 0))
    yield stub
    if stub.gate is not None:
        stub.gate.set()
    srv.shutdown()
    srv.server_close()


def settle(backend, bid: str, timeout: float = 20.0) -> tuple[str, str]:
    """Poll like the poller does until the batch leaves 'pending'."""
    t0 = time.monotonic()
    while True:
        state, detail = backend.status(bid)
        if state != "pending":
            return state, detail
        assert time.monotonic() - t0 < timeout, f"still pending: {detail}"
        threading.Event().wait(0.02)


def dead_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def req(user: str, **kw) -> llm.Request:
    return llm.Request(custom_id=kw.pop("cid", user), system=kw.pop("system", ""), user=user,
                       max_tokens=kw.pop("max_tokens", 50), **kw)


# ---------------------------------------------------------------------------
# the batch interface over synchronous completions
# ---------------------------------------------------------------------------

def test_the_batch_interface_round_trips_without_blocking_submit(vllm, tmp_path):
    vllm.gate = threading.Event()                      # every completion waits until released
    b = llm.LocalOpenAI("chat", "", tmp_path)
    assert b.served_models == ["chat"] and b.id == "local/chat"
    t0 = time.monotonic()
    bid = b.submit([req(f"question {i}", cid=f"r{i}") for i in range(5)])
    assert time.monotonic() - t0 < 1.0 and re.fullmatch(r"local_[0-9a-f]{12}", bid)
    assert b.status(bid) == ("pending", "0/5 done")
    with pytest.raises(llm.LLMError, match="not finished: 0/5 done"):
        b.fetch(bid)
    d = tmp_path / "llm_batches" / "local" / bid
    assert [json.loads(x)["custom_id"] for x in (d / "requests.jsonl").read_text().splitlines()] \
        == [f"r{i}" for i in range(5)]
    vllm.gate.set()
    assert settle(b, bid) == ("done", "5/5 done")
    got = b.fetch(bid)
    assert {k: v.text for k, v in got.items()} == {f"r{i}": f"echo: question {i}" for i in range(5)}
    assert all(not v.error for v in got.values())
    # the results are on disk: a fresh client (a restarted process) reads the same
    again = llm.LocalOpenAI("chat", "", tmp_path).fetch(bid)
    assert {k: v.text for k, v in again.items()} == {k: v.text for k, v in got.items()}
    for bad in ("local_000000000000", "../../etc"):
        with pytest.raises(llm.LLMError, match="unknown local batch"):
            b.status(bad)


@pytest.mark.parametrize("n", [1, 2])
def test_concurrency_is_bounded(vllm, tmp_path, monkeypatch, n):
    monkeypatch.setattr(config, "LOCAL_CONCURRENCY", n)
    vllm.delay = 0.05
    b = llm.LocalOpenAI("chat", "", tmp_path)
    settle(b, b.submit([req(f"q{i}") for i in range(8)]))
    assert vllm.max_inflight == n and sum(vllm.hits.values()) == 8


def test_a_500_retries_then_degrades_to_that_one_error(vllm, tmp_path):
    vllm.always["always-500"] = (500, "internal server error")
    vllm.always["oom"] = (400, "CUDA out of memory. Tried to allocate 2.00 GiB")
    vllm.first["flaky"] = [(503, "busy"), (500, "engine restarting")]
    b = llm.LocalOpenAI("chat", "", tmp_path)
    bid = b.submit([req("fine one"), req("always-500"), req("flaky"), req("oom"), req("fine two")])
    assert settle(b, bid) == ("done", "5/5 done, 2 failed")    # the batch completes
    got = b.fetch(bid)
    assert "HTTP 500" in got["always-500"].error and vllm.hits["always-500"] == 4
    assert "out of memory" in got["oom"].error and vllm.hits["oom"] == 4   # OOM reads transient
    assert got["flaky"].text == "echo: flaky" and vllm.hits["flaky"] == 3
    assert got["fine one"].text == "echo: fine one" and got["fine two"].text == "echo: fine two"


def test_the_backoff_is_2_8_30(vllm, tmp_path, monkeypatch):
    monkeypatch.setattr(llm.LocalOpenAI, "BACKOFF", (2, 8, 30))
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    vllm.always["down"] = (502, "bad gateway")
    b = llm.LocalOpenAI("chat", "", tmp_path)
    settle(b, b.submit([req("down")]))
    assert waits == [2, 8, 30] and vllm.hits["down"] == 4


def test_a_4xx_is_recorded_without_retrying(vllm, tmp_path):
    vllm.always["too-long"] = (400, "This model's maximum context length is 8192 tokens")
    vllm.always["no-such-model"] = (404, "The model `gemma` does not exist.")
    b = llm.LocalOpenAI("chat", "", tmp_path)
    bid = b.submit([req("too-long"), req("no-such-model"), req("fine")])
    assert settle(b, bid) == ("done", "3/3 done, 2 failed")
    got = b.fetch(bid)
    assert "HTTP 400" in got["too-long"].error and vllm.hits["too-long"] == 1
    assert "HTTP 404" in got["no-such-model"].error and vllm.hits["no-such-model"] == 1
    assert got["fine"].text == "echo: fine"


def test_a_batch_fails_only_when_every_request_did(vllm, tmp_path):
    vllm.always["bad"] = (400, "malformed")
    b = llm.LocalOpenAI("chat", "", tmp_path)
    state, detail = settle(b, b.submit([req("bad a"), req("bad b")]))
    assert state == "failed" and "HTTP 400" in detail and "malformed" in detail


def test_a_restart_resumes_instead_of_re_running(vllm, tmp_path):
    b = llm.LocalOpenAI("chat", "", tmp_path)
    bid = b.submit([req(f"q{i}", cid=f"r{i}") for i in range(4)])
    settle(b, bid)
    res = tmp_path / "llm_batches" / "local" / bid / "results.jsonl"
    lines = res.read_text().splitlines()
    kept = {json.loads(x)["custom_id"] for x in lines[:2]}
    # the process died with two results unwritten, one of them half-way through its line
    res.write_text(lines[0] + "\n" + lines[1] + "\n" + lines[2][:15])
    before = Counter(vllm.hits)
    again = llm.LocalOpenAI("chat", "", tmp_path)        # the next process's client
    assert again.status(bid)[0] == "pending"
    assert settle(again, bid) == ("done", "4/4 done")
    rerun = vllm.hits - before
    assert set(rerun) == {f"q{i}" for i in range(4) if f"r{i}" not in kept}
    assert all(v == 1 for v in rerun.values())
    assert {k: v.text for k, v in again.fetch(bid).items()} == {f"r{i}": f"echo: q{i}" for i in range(4)}


def test_one_worker_per_batch_however_many_clients_poll_it(vllm, tmp_path):
    vllm.gate = threading.Event()
    a = llm.LocalOpenAI("chat", "", tmp_path)
    bid = a.submit([req("one"), req("two")])
    other = llm.LocalOpenAI("chat", "", tmp_path)
    assert other._ensure_worker(bid) is False             # the first worker holds the lock
    assert other.status(bid)[0] == "pending"
    vllm.gate.set()
    settle(other, bid)
    assert dict(vllm.hits) == {"one": 1, "two": 1}


def test_json_mode_and_the_token_cap_on_the_wire(vllm, tmp_path):
    b = llm.LocalOpenAI("chat", "", tmp_path)
    settle(b, b.submit([req("wants json", system="sys", max_tokens=8192, json=True),
                        req("wants prose", max_tokens=100)]))
    sent = {x["messages"][-1]["content"]: x for x in vllm.bodies}
    assert sent["wants json"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in sent["wants prose"]
    assert sent["wants json"]["max_tokens"] == 1024        # LOCAL_MAX_TOKENS: the card is shared
    assert sent["wants prose"]["max_tokens"] == 100
    assert sent["wants json"]["model"] == "chat"
    assert sent["wants json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert [m["role"] for m in sent["wants prose"]["messages"]] == ["user"]   # no empty system turn


def test_json_mode_follows_the_flag_for_openai_and_is_ignored_by_anthropic(monkeypatch):
    sent = []

    def fake_http(method, url, headers, body=None, timeout=60.0):
        sent.append(body)
        if "/files" in url:
            return 200, b'{"id": "file_1"}'
        return 200, b'{"id": "batch_1"}'
    monkeypatch.setattr(llm, "_http", fake_http)
    reqs = [req("a", system="s", json=True), req("b", system="s")]
    llm.OpenAIBatches("gpt-x", "k").submit(reqs)
    upload = sent[0].decode().split("application/jsonl\r\n\r\n", 1)[1].split("\r\n--", 1)[0]
    bodies = {json.loads(x)["custom_id"]: json.loads(x)["body"] for x in upload.split("\n")}
    assert bodies["a"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in bodies["b"]
    llm.AnthropicBatches("claude-x", "k").submit(reqs)
    assert "response_format" not in sent[-1].decode()


def test_every_caller_that_parses_json_asks_for_it(tree, tmp_path):
    prop = proposals.proposal_request(1, "m", "exam_economics", "economics", [],
                                      {"diagnose_items": 0, "diagnose_weak": 0}, "rubric")
    gens = proposals.generation_requests(1, "spec", "economics", 4, "doc", 7)
    drafts = eb.draft_requests(tmp_path / "exam", "economics", 8)
    judged, _ = jd.plan_requests(tree["models"]["fx/chance-160m"]["dir"], "claude")
    assert prop.json and gens and drafts and judged
    assert all(r.json for r in gens + drafts + judged)
    fb = llm.FakeBatches("fake-1", tmp_path)
    fb.submit([prop, req("prose, please")])
    assert [r["json"] for r in fb.recorded()] == [True, False]


def test_a_local_generator_is_asked_for_one_document_at_a_time(monkeypatch):
    # two ~600-word documents is ~1700 tokens and LOCAL_MAX_TOKENS caps the
    # reply at 1024: asking for two truncates the JSON and nothing parses
    monkeypatch.setattr(config, "LLM_PROVIDER", "local")
    assert proposals.items_per_request("doc") == 1
    assert proposals.items_per_request("free") == 10          # short; two fit easily
    reqs = proposals.generation_requests(1, "spec", "economics", 4, "doc", 7)
    assert len(reqs) == 4 and all("Write 1 document." in r.user for r in reqs)
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    assert proposals.items_per_request("doc") == 2
    reqs = proposals.generation_requests(1, "spec", "economics", 4, "doc", 7)
    assert len(reqs) == 2 and all("Write 2 documents." in r.user for r in reqs)
    # and one document comes back as the object itself, not in an array
    doc = {"title": "Margins first", "text": "word " * 200}
    assert proposals.parse_items(json.dumps(doc), "doc") == [
        {"title": "Margins first", "text": ("word " * 200).strip()}]


def test_the_container_reaches_the_box_through_the_host_gateway():
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"host.docker.internal:host-gateway"' in compose
    assert "LOCAL_BASE_URL: ${LOCAL_BASE_URL:-http://host.docker.internal:8000/v1}" in compose
    # host networking would reach it too and would undo the ${BIND} publish line
    assert not re.search(r"(?m)^\s*network_mode:", compose)
    assert config.LOCAL_BASE_URL == "http://localhost:8000/v1"       # the code default: the host
    env = (REPO / ".env.example").read_text(encoding="utf-8")
    assert "LOCAL_BASE_URL" in env and "LOCAL_CONCURRENCY" in env and "LOCAL_MAX_TOKENS" in env


def test_a_json_mode_object_wrapping_the_array_reads_as_the_array():
    assert llm.extract_array('{"documents": [{"title": "t"}]}') == [{"title": "t"}]
    assert llm.extract_array('[{"a": 1}]') == [{"a": 1}]
    assert llm.extract_array('{"title": "t", "text": "x"}') is None       # an object is not an array
    assert llm.extract_array('{"a": [1], "b": [2]}') is None              # ambiguous: which one?
    assert llm.extract_array("no json") is None
    doc = {"title": "Margins first", "text": "word " * 200}
    assert proposals.parse_items(json.dumps({"documents": [doc]}), "doc")[0]["title"] == "Margins first"
    cand = {"prompt": "Explain why a binding constraint moves the margin first.", "reference": "r"}
    assert eb.parse_candidates(json.dumps({"questions": [cand]}))[0]["reference"] == "r"


# ---------------------------------------------------------------------------
# configuration: keys, models, and a server that is not what it should be
# ---------------------------------------------------------------------------

def test_a_local_provider_needs_a_model_but_no_key(monkeypatch):
    def cfg(**kw):
        for k, v in {"LLM_PROVIDER": "", "LLM_MODEL": "", "LLM_API_KEY": "",
                     "EXAM_PROVIDER": "", "EXAM_MODEL": "", "EXAM_API_KEY": "", **kw}.items():
            monkeypatch.setattr(config, k, v)
    cfg(LLM_PROVIDER="local", LLM_MODEL="chat", EXAM_PROVIDER="local", EXAM_MODEL="chat")
    assert llm.blocked() == "" and llm.blocked("exam") == ""
    llm.startup_check()                  # and startup never asks the server: vLLM may still be loading
    cfg(LLM_PROVIDER="local")
    assert "LLM_MODEL is unset" in llm.blocked() and "/models lists them" in llm.blocked()
    with pytest.raises(RuntimeError, match="LLM_MODEL is unset"):
        llm.startup_check()
    # a missing key is still fatal for a provider that needs one
    cfg(LLM_PROVIDER="openai", LLM_MODEL="gpt-x")
    with pytest.raises(RuntimeError, match="LLM_API_KEY is unset"):
        llm.startup_check()
    assert "local" in llm.PROVIDERS and not llm.needs_key("local") and llm.needs_key("openai")


def test_a_local_judge_runs_provisional_instead_of_being_refused(monkeypatch):
    def cfg(**kw):
        for k, v in {"JUDGE_PROVIDER": "local", "JUDGE_MODEL": "chat", "JUDGE_API_KEY": "",
                     "EXAM_PROVIDER": "openai", "EXAM_MODEL": "gpt-x", "EXAM_API_KEY": "k",
                     "LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-x", "LLM_API_KEY": "k",
                     "ALLOW_SINGLE_PROVIDER_LOOP": False, **kw}.items():
            monkeypatch.setattr(config, k, v)
    cfg()
    assert jd.blocked() == ""                     # undated and keyless: a stamp, not a refusal
    assert jd.identity() == {"provider": "local", "model": "chat", "id": "local/chat",
                             "family": "chat"}
    cfg(JUDGE_MODEL="")
    assert "JUDGE_MODEL is unset" in jd.blocked()
    # every other refusal stands
    cfg(JUDGE_PROVIDER="anthropic", JUDGE_MODEL="claude-sonnet-4-5", JUDGE_API_KEY="k")
    assert "floating alias" in jd.blocked()
    cfg(JUDGE_PROVIDER="anthropic", JUDGE_MODEL="claude-sonnet-4-5-20250929")
    assert "JUDGE_API_KEY is unset" in jd.blocked()
    # all three roles local is a provider clash, and still takes someone typing the override
    cfg(EXAM_PROVIDER="local", EXAM_MODEL="chat", LLM_PROVIDER="local", LLM_MODEL="chat")
    assert "same as the exam writer" in jd.blocked()
    cfg(EXAM_PROVIDER="local", EXAM_MODEL="chat", LLM_PROVIDER="local", LLM_MODEL="chat",
        ALLOW_SINGLE_PROVIDER_LOOP=True)
    assert jd.blocked() == "" and jd.single_provider_loop() is True


def test_a_server_serving_another_id_is_named_at_construction(vllm, tmp_path):
    with pytest.raises(llm.LLMError) as e:
        llm.LocalOpenAI("gemma", "", tmp_path)
    assert f"vLLM is up at {vllm.url} but serves ['chat'], not 'gemma'" in str(e.value)
    assert not isinstance(e.value, llm.LocalUnreachable)
    # nothing listening is a different failure: one to wait out, with the tunnel named
    with pytest.raises(llm.LocalUnreachable, match="ssh -L 8000:localhost:8000"):
        llm.LocalOpenAI("chat", "", tmp_path, base_url=dead_url())


def test_http_errors_carry_their_status(vllm):
    vllm.always["boom"] = (503, "overloaded")
    body = json.dumps({"model": "chat", "messages": [{"role": "user", "content": "boom"}]}).encode()
    with pytest.raises(llm.LLMError) as e:
        llm._http("POST", f"{vllm.url}/chat/completions", {"content-type": "application/json"}, body)
    assert e.value.status == 503 and "HTTP 503" in str(e.value) and "overloaded" in str(e.value)
    with pytest.raises(llm.LLMError) as e:
        llm._http("GET", f"{dead_url()}/models", {})
    assert e.value.status is None
    vllm.delay = 1.0
    with pytest.raises(llm.LLMError, match="no response within") as e:
        llm._http("POST", f"{vllm.url}/chat/completions", {"content-type": "application/json"},
                  body.replace(b"boom", b"slow"), timeout=0.2)
    assert e.value.status is None


def test_the_poller_waits_out_a_server_that_is_down(tmp_path, monkeypatch):
    client, _, _ = make_service(tmp_path, monkeypatch, tree=False)
    try:
        monkeypatch.setattr(config, "LLM_PROVIDER", "local")
        monkeypatch.setattr(config, "LLM_MODEL", "chat")
        monkeypatch.setattr(config, "LOCAL_BASE_URL", dead_url())
        llm.reset()
        db.batch_add("local_0123456789ab", "proposal", 1, 1, "local", "chat")
        assert llm_poller.tick() == 0
        assert [b["batch_id"] for b in db.batches_pending()] == ["local_0123456789ab"]
    finally:
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# the stamp: every artefact from a local identity, always
# ---------------------------------------------------------------------------

def test_provisional_is_a_preliminary_reason_no_calibration_lifts():
    tasks = {t: {"n_report": 40, "score_report": 3.0} for t in report.EXAM_TASKS[:3]}
    j = {"judge": {"id": "local/chat", "provisional": True, "provisional_reason": REASON},
         "preliminaryReasons": [], "tasks": tasks}
    cal = {"kappa": 0.9, "n": 60, "calibrated": True, "judge_id": "local/chat"}
    # calibrated, for this very judge, and the judge this server runs — still not counted
    assert report.judged_state(j, cal, "local/chat") == {"ok": False, "reasons": [REASON],
                                                         "current": True}
    assert report.judged_avg(j) is None
    assert report.judged_avg({**j, "judge": {"id": "local/chat"}}) == 3.0


def test_a_local_judge_writes_a_provisional_judge_json_and_the_page_greys_it(vllm, tmp_path,
                                                                              monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judged=True, judge_model="")
    try:
        monkeypatch.setattr(config, "JUDGE_PROVIDER", "local")
        monkeypatch.setattr(config, "JUDGE_MODEL", "chat")
        monkeypatch.setattr(config, "JUDGE_API_KEY", "")
        llm.reset()
        assert config.judged_blocked() == ""
        d = tree["models"]["fx/chance-160m"]["dir"]
        (d / "judge.json").unlink()                                    # the stub's file
        jr = jd.start_run(d, tree["out_dir"])
        assert jr["mode"] == "batch" and jr["batch_id"].startswith("local_")
        settle(llm.client("judge"), jr["batch_id"])
        assert llm_poller.tick() == 1
        j = json.loads((d / "judge.json").read_text())
        jj = j["judge"]
        assert jj["provisional"] is True and jj["provisional_reason"] == REASON
        assert jj["base_url"] == vllm.url and jj["served_model"] == "chat"
        assert jj["weights"] == WEIGHTS
        assert jj["id"] == "local/chat" and jj["family"] == "gemma"   # the weights' family, not 'chat'
        assert j["preliminary_reasons"][0] == REASON
        assert all(v["ungraded"] == 0 for v in j["tasks"].values())
        assert all(b.get("response_format") == {"type": "json_object"} for b in vllm.bodies)
        fresh(appmod)
        p = client.get("/api/results").json()
        row = next(m for m in p["models"] if m["id"] == "fx/chance-160m")
        assert row["judge"]["judge"]["provisional"] is True
        assert row["judgeState"]["ok"] is False and REASON in row["judgeState"]["reasons"]
        assert row["judgedAvg"] is None                                # never in any average
        assert any("graded by a local model" in w for w in p["warnings"])
        # nothing a local judge wrote can pick a topic to train
        gate = row["judge"]["tasks"]["exam_economics"]["propose"]
        assert gate["ok"] is False and REASON in gate["why"]
    finally:
        client.__exit__(None, None, None)


def test_a_local_proposer_and_generator_stamp_the_spec_and_the_dataset(vllm, tmp_path,
                                                                       monkeypatch):
    # the stub judge graded the board and is the judge this server runs, so
    # the gate lets a proposal through; the generator is the local model
    client, _, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    try:
        monkeypatch.setattr(config, "LLM_PROVIDER", "local")
        monkeypatch.setattr(config, "LLM_MODEL", "gemma")
        llm.reset()
        body = {"model": "fx/good-750m", "topic": "economics", "requested_by": "t"}
        r = client.post("/api/proposals", json=body)
        assert r.status_code == 503 and "serves ['chat'], not 'gemma'" in r.json()["detail"]
        monkeypatch.setattr(config, "LLM_MODEL", "chat")
        llm.reset()
        r = client.post("/api/proposals", json=body)
        assert r.status_code == 200, r.text
        pid, bid = r.json()["id"], r.json()["batch_id"]
        settle(llm.client(), bid)
        assert llm_poller.tick() == 1
        p = client.get(f"/api/proposals/{pid}").json()
        assert p["status"] == "proposed" and p["proposer"] == "local/chat"
        ev = p["evidence"]
        assert ev["provisional"] is True and ev["provisional_reason"] == \
            "proposed by a local model — not a pinned benchmark"
        assert ev["served_model"] == "chat" and ev["weights"] == WEIGHTS
        client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
        g = client.post(f"/api/proposals/{pid}/generate", json={"requester": "Omar", "count": 4}).json()
        settle(llm.client(), g["batch_id"])
        assert llm_poller.tick() == 1
        ds = client.get(f"/api/datasets/{g['dataset_id']}").json()
        assert ds["status"] == "ready", ds["error"]
        prov = ds["provenance"]
        # the documents came back wrapped in an object, as JSON mode returns them
        assert prov["items"]["generated"] == 4
        assert prov["provisional"] is True
        assert prov["provisional_reason"] == "a local model was the generator — not a pinned benchmark"
        assert prov["local_models"] == {"generator": {"base_url": vllm.url, "served_model": "chat",
                                                      "weights": WEIGHTS}}
        assert proposals.provenance_complete(prov) == []
        assert all(b["response_format"] == {"type": "json_object"} for b in vllm.bodies)
    finally:
        client.__exit__(None, None, None)


def test_a_pinned_loop_carries_no_stamp(tmp_path, monkeypatch):
    client, _, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    try:
        pid = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": "economics",
                                                  "requested_by": "t"}).json()["id"]
        llm_poller.tick()
        assert "provisional" not in client.get(f"/api/proposals/{pid}").json()["evidence"]
        client.post(f"/api/proposals/{pid}/approve", json={"approver": "Omar"})
        did = client.post(f"/api/proposals/{pid}/generate",
                          json={"requester": "Omar", "count": 4}).json()["dataset_id"]
        llm_poller.tick()
        prov = client.get(f"/api/datasets/{did}").json()["provenance"]
        assert "provisional" not in prov and "local_models" not in prov
    finally:
        client.__exit__(None, None, None)


def test_a_local_exam_writer_stamps_its_candidates_and_the_bank_keeps_it(vllm, tmp_path):
    b = llm.LocalOpenAI("chat", "", tmp_path)
    root = tmp_path / "exam"
    r = eb.draft(root, b, ["economics"], per_topic=4, wait=True, poll_s=0.02)
    assert r["written"] == {"economics": 4}
    cands = eb.load_candidates(root, "economics")
    assert len(cands) == 4 and all(c["drafted_by"] == "local/chat" for c in cands)
    assert all(c["provisional"] is True and c["weights"] == WEIGHTS for c in cands)
    assert cands[0]["provisional_reason"] == "drafted by a local model — not a pinned benchmark"
    rec = eb.accept(root, cands[0]["cid"], "Omar")
    assert rec["provisional"] is True and rec["served_model"] == "chat" and rec["accepted_by"] == "Omar"
