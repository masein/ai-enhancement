"""12d.3: the Playground chats with models served elsewhere — the phone build
and its setups, the original — through their OpenAI-compatible address,
streamed: thinking (`reasoning_content`) into the fold, the server's tokens and
MTP's drafts under the reply, a run testing that same model waited for (and
nothing sent to it meanwhile), a server that doesn't answer said in one line,
Compare across a local and a served model, practice questions marked as for a
local one. The server is tests/fake_openai.py; no model runs."""

from __future__ import annotations

import shutil
import threading
import time

import pytest

import everyday as ev
from fake_openai import FakeServer
from service import chat, runner
from test_playground_12d1 import ME, SMALL, events, new_chat, say, svc  # noqa: F401

LDA = "served/k4-LDA-MTP"
HOW = "llama.cpp fork, k=4 + LDA, MTP n_max 3, phone build"


@pytest.fixture
def fake():
    s = FakeServer()
    s.reply = lambda body: "The train arrives at 9:40, on platform 2."
    s.reasoning = "The timetable says 9:40."
    s.timings = lambda body: {"predicted_n": 12, "draft_n": 165, "draft_n_accepted": 135}
    yield s
    s.close()


def register(client, fake, name="k4-LDA MTP", thinking="on", **over):
    r = client.post("/api/served", json={"name": name, "base_url": fake.base, "key": "",
                                         "based_on": "Qwen/Qwen3.6-35B-A3B", "how": HOW,
                                         "thinking": thinking, "by": ME, **over})
    assert r.status_code == 200, r.text
    return r.json()["model"]["id"]


def test_a_served_model_is_offered_with_its_tags(svc, fake):  # noqa: F811
    client = svc
    sid = register(client, fake)
    ms = {m["id"]: m for m in client.get("/api/playground").json()["models"]}
    assert ms[sid]["served"] is True and ms[sid]["phone"] is True
    assert ms[SMALL]["served"] is False and ms[SMALL]["phone"] is False
    # its scored settings are the one function's, as for a local model
    assert ms[sid]["scored"]["thinking_on"] is True and ms[sid]["scored"]["can_think"] is True
    # nothing loads for it, so the runs popover never says it is loaded
    c = new_chat(client, model=sid)
    say(client, c, "when does the train arrive?")
    assert client.get("/api/playground/status").json()["loaded"] == []


def test_a_served_reply_streams_thinking_folded_and_mtp_kept(svc, fake):  # noqa: F811
    client = svc
    sid = register(client, fake)
    c = new_chat(client, model=sid)
    evs, c = say(client, c, "when does the train arrive?")
    kinds = [e["t"] for e in evs]
    assert kinds[0] == "place" and evs[0]["device"] == "served" and kinds[-1] == "done"
    assert kinds.count("text") > 3 and kinds.count("think") >= 1
    reply = c["messages"][1]["replies"][0]
    assert reply["text"] == "The train arrives at 9:40, on platform 2."
    assert reply["thinking"].strip() == "The timetable says 9:40."
    # the server's own count (usage), thinking included
    assert reply["tokens"] == len(reply["text"].split()) + len(reply["thinking"].split())
    assert reply["draft"] == {"n": 165, "accepted": 135}
    # its server's speed is not the phone's: no words a second is kept
    assert reply["wps"] is None and reply["secs"] > 0
    # asked as a run asks it: streamed, its settings, thinking as registered
    body = fake.requests[-1]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["temperature"] == 0.0 and body["max_tokens"] == c["scored"]["max_gen_toks"]
    assert body["messages"] == [{"role": "user", "content": "when does the train arrive?"}]
    # the next message carries the reply shown, never its thinking
    say(client, c, "and the platform?")
    assert fake.requests[-1]["messages"][1] == {"role": "assistant", "content": reply["text"]}


def test_a_key_is_sent_to_the_server_and_to_no_page(svc, fake):  # noqa: F811
    client = svc
    fake.key = "lda-key-3f9a1c77e2"
    sid = register(client, fake, key=fake.key)
    c = new_chat(client, model=sid)
    evs, c = say(client, c, "hello")
    assert fake.auth[-1] == "Bearer lda-key-3f9a1c77e2"
    for page in (client.get("/api/playground").text, client.get(
            f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).text, str(evs)):
        assert "lda-key-3f9a1c77e2" not in page


def test_while_a_run_tests_that_model_chat_waits_and_sends_nothing(svc, fake):  # noqa: F811
    client = svc
    sid = register(client, fake)
    other = register(client, fake, name="k4-LDA no MTP")
    r = client.post("/api/submissions", json={"hf_id": sid, "suite": "everyday"})
    assert r.status_code == 200, r.text
    run = r.json()["id"]
    from service import db
    db.update(run, status="running", progress="140 of 388 · 4.1 s an answer · about 20 min left")
    runner.LOCK.mkdir(parents=True, exist_ok=True)
    (runner.LOCK / "submission").write_text(str(run))
    try:
        c = new_chat(client, model=sid)
        before = len(fake.requests)
        evs, c = say(client, c, "hello?")
        line = (f"Being tested right now (run #{run}, about 20 min left). "
                "Chat starts when it's done.")
        assert evs[-1] == {"t": "refused", "why": line}
        assert c["messages"][1]["refused"] == line
        assert len(fake.requests) == before              # nothing was sent to it
        st = client.get("/api/playground/status").json()
        assert st["testing"] == sid and st["testing_line"] == line
        # another served model stays chattable
        evs, _ = say(client, new_chat(client, model=other), "hello?")
        assert evs[-1]["t"] == "done" and len(fake.requests) == before + 1
    finally:
        shutil.rmtree(runner.LOCK, ignore_errors=True)
    # the run is over: asked again, it answers
    r = client.post(f"/api/playground/chats/{c['id']}/again", json={"n": 1, "by": ME})
    assert events(client, r.json()["stream"])[-1]["t"] == "done"


def test_a_run_testing_it_stops_its_chat_reply_first(svc, fake, monkeypatch):  # noqa: F811
    client = svc
    sid = register(client, fake)
    fake.stream_delay_s = 0.05
    c = new_chat(client, model=sid)
    r = client.post(f"/api/playground/chats/{c['id']}/messages", json={"text": "hi", "by": ME})
    st = chat.ENGINE.streams[r.json()["stream"]]
    t0 = time.time()
    while not any(e["t"] == "text" for e in st.events) and time.time() - t0 < 5:
        time.sleep(0.02)
    took = chat.ENGINE.yield_served(sid)
    assert took < 5 and st.done               # ended before the run goes on
    reply = client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).json()[
        "messages"][1]["replies"][0]
    assert reply["cut"] == chat.CUT_RUN and reply["text"]
    assert not chat.ENGINE.loaded[sid].busy.locked()


def test_a_server_that_does_not_answer_is_one_line(svc, fake):  # noqa: F811
    client = svc
    sid = register(client, fake)
    port = fake.port
    fake.close()
    c = new_chat(client, model=sid)
    evs, c = say(client, c, "hello?")
    line = f"The server at :{port} isn't answering."
    assert evs[-1] == {"t": "error", "why": line}
    assert c["messages"][1]["refused"] == line and "pending" not in c["messages"][1]


def test_compare_a_local_and_a_served_model_streams_both_at_once(svc, fake):  # noqa: F811
    client = svc
    sid = register(client, fake)
    fake.stream_delay_s = 0.02
    r = client.post("/api/playground/chats", json={"model": SMALL, "model2": sid, "by": ME})
    assert r.status_code == 200, r.text
    c = r.json()
    r = client.post(f"/api/playground/chats/{c['id']}/messages", json={"text": "hi", "by": ME})
    j = r.json()
    got = {}
    ts = [threading.Thread(target=lambda k, s: got.__setitem__(k, events(client, s)), args=(k, s))
          for k, s in (("a", j["stream"]), ("b", j["stream_b"]))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    # a served model takes no memory here: neither waits for the other
    assert all(e["t"] != "wait" for e in got["a"] + got["b"])
    c = client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).json()
    a, b = c["messages"][1]["replies"][0], c["messages"][1]["b"]["replies"][0]
    assert a["text"].startswith("You asked: hi") and a["device"] == "cuda"
    assert b["text"] == "The train arrives at 9:40, on platform 2." and b["device"] == "served"


def test_a_practice_question_is_marked_for_a_served_model_too(svc, fake):  # noqa: F811
    client = svc
    q = next(x for x in ev.load_bank() if ev.half(x) == ev.PRACTICE
             and not any(c["type"] == "judge" for c in x["checks"]))
    fake.reply = lambda body: q["reference"]
    fake.reasoning = ""
    sid = register(client, fake, thinking="off")
    evs, c = say(client, new_chat(client, model=sid), q["prompt"],
                 practice={"kind": "everyday", "id": q["id"]})
    assert c["messages"][1]["replies"][0]["mark"] == {"kind": "everyday", "pass": True,
                                                     "words": "✓ passes"}
    assert fake.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}
