"""16b.3: Use as an API — an OpenAI-compatible address, behind each person's
own key, through the Playground's engine.

- Keys: made with the board's token, shown once, kept as a sha256; a revoked
  one answers nothing; the shared write token is never an API key.
- /v1/models lists what can be asked; a base model, a GGUF with no server and
  a model from OpenRouter are not offered, each with why.
- /v1/chat/completions, streamed and not: a model that loads, a served model
  through its server (its key never returned), counts kept for the key and no
  prompt or reply stored.
- Taking turns: a run holding the GPU is 503 with Retry-After and one line;
  two requests at once — the second waits, then 429; a reply's cap; a
  conversation too long; a server that is down.

The fake backend; a fake server. Nothing runs."""

from __future__ import annotations

import json
import threading
import time


from fake_openai import FakeServer
from service import chat, config, db
from test_playground_12d1 import BASE, BIG, SMALL, run_holds, svc  # noqa: F401

ME = "masein"


def key(client, who=ME) -> str:
    r = client.post("/api/keys", json={"by": who})
    assert r.status_code == 200, r.text
    return r.json()["key"]


def ask(client, k, model=SMALL, stream=False, **body):
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {k}"},
                       json={"model": model, "stream": stream,
                             "messages": [{"role": "user", "content": "When is the train?"}],
                             **body})


def test_a_key_is_shown_once_kept_as_its_hash_and_revoked(svc, monkeypatch):  # noqa: F811
    client = svc
    monkeypatch.setattr(config, "SUBMIT_TOKEN", "s3cret")
    assert client.post("/api/keys", json={"by": ME}).status_code == 401      # the board's token
    r = client.post("/api/keys", json={"by": ME}, headers={"X-Token": "s3cret"})
    k = r.json()["key"]
    assert k.startswith("ebk_") and r.json()["line"].startswith("Copy it now: it is shown once")
    listed = client.get("/api/keys", params={"who": ME}).json()["keys"]
    assert [x["prefix"] for x in listed] == [k[:10]] and k not in json.dumps(listed)
    import sqlite3
    with sqlite3.connect(config.DB_PATH) as c:
        assert not c.execute("SELECT 1 FROM api_keys WHERE hash=?", (k,)).fetchone()
    # the shared write token is not a key
    r = client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 401 and r.json()["error"]["message"] == \
        "That API key isn't one of this board's."
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {k}"}).status_code == 200
    # someone else can't revoke it; its owner can, and it answers nothing after
    kid = listed[0]["id"]
    r = client.post(f"/api/keys/{kid}/revoke", json={"by": "omar"}, headers={"X-Token": "s3cret"})
    assert r.status_code == 403
    r = client.post(f"/api/keys/{kid}/revoke", json={"by": ME}, headers={"X-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["revoked_at"]
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {k}"})
    assert r.status_code == 401 and r.json()["error"]["message"] == "That API key was revoked."


def test_the_models_offered_and_why_the_others_arent(svc):  # noqa: F811
    client = svc
    k = key(client)
    ids = [m["id"] for m in client.get("/v1/models",
                                       headers={"Authorization": f"Bearer {k}"}).json()["data"]]
    assert SMALL in ids and BIG in ids and BASE not in ids
    r = ask(client, k, BASE)
    assert r.status_code == 404 and r.json()["error"]["message"] == (
        f"{BASE}: not offered through the API — a base model, with no chat template: it isn't "
        "asked as a chat.")
    assert client.get("/api/models/api", params={"id": SMALL}).json()["offered"] is True


def test_a_reply_whole_and_streamed_its_counts_kept_and_nothing_stored(svc):  # noqa: F811
    client = svc
    k = key(client)
    r = ask(client, k)
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["object"] == "chat.completion" and got["model"] == SMALL
    assert got["choices"][0]["message"]["content"].startswith("You asked: When is the train?")
    assert got["choices"][0]["finish_reason"] == "stop" and got["usage"]["total_tokens"] > 0
    with client.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {k}"},
                       json={"model": SMALL, "stream": True,
                             "messages": [{"role": "user", "content": "hello"}]}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [x[6:] for x in r.iter_lines() if x.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text.startswith("You asked: hello") and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # counts, for the key; no prompt or reply anywhere in the database
    [row] = db.apikeys(ME)
    assert row["requests"] == 2 and row["tokens"] > 0 and row["last_used"]
    import sqlite3
    with sqlite3.connect(config.DB_PATH) as c:
        dump = "\n".join(c.iterdump())
    assert "When is the train?" not in dump and "You asked" not in dump


def test_while_a_run_holds_the_gpu_a_model_that_would_load_is_503(svc, run_holds):  # noqa: F811
    client = svc
    k = key(client)
    r = ask(client, k, BIG)
    assert r.status_code == 503 and r.headers["retry-after"] == "60"
    assert r.json()["error"]["message"].startswith("Run #")
    # a small one answers on the CPU, as in the Playground
    assert ask(client, k, SMALL).status_code == 200


def test_two_at_once_the_second_waits_then_is_told_to_try_again(svc, monkeypatch):  # noqa: F811
    client = svc
    k = key(client)
    monkeypatch.setattr(config, "API_WAIT_S", 0.5)
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.2)          # a slow first reply
    first = {}
    t = threading.Thread(target=lambda: first.update(r=ask(client, k)))
    t.start()
    time.sleep(0.3)
    r = ask(client, k)
    assert r.status_code == 429 and r.headers["retry-after"] == "10"
    assert "answering another request" in r.json()["error"]["message"]
    t.join()
    assert first["r"].status_code == 200
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.0)
    assert ask(client, k).status_code == 200                       # its turn now


def test_a_replys_cap_and_a_conversation_too_long(svc, monkeypatch):  # noqa: F811
    client = svc
    k = key(client)
    seen = {}
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        lambda m, msgs, s: seen.update(s=s) or "ok")
    assert ask(client, k, max_tokens=99999).status_code == 200
    assert seen["s"]["max_gen_toks"] == config.API_MAX_TOKENS
    r = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {k}"},
                    json={"model": SMALL, "messages": [{"role": "user",
                                                        "content": "word " * 20000}]})
    assert r.status_code == 400 and r.json()["error"]["type"] == "context_length_exceeded"


def test_a_served_model_through_its_server_and_one_that_is_down(svc):  # noqa: F811
    client = svc
    k = key(client)
    fake = FakeServer()
    try:
        r = client.post("/api/served", json={"name": "LDA api", "base_url": fake.base,
                                             "how": "k4", "thinking": "off", "by": ME,
                                             "key": "server-secret"})
        mid = r.json()["model"]["id"]
        r = ask(client, k, mid)
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["message"]["content"]
        assert "server-secret" not in r.text
        assert fake.requests[-1]["messages"][-1]["content"] == "When is the train?"
    finally:
        fake.close()
    # its server gone: one line, not a hang
    r = ask(client, k, mid)
    assert r.status_code in (500, 502) and r.json()["error"]["message"]
