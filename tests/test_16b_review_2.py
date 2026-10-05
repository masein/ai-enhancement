"""16b's review, the medium and low points.

3. The upload quota counts the other uploads under way — their unwritten
   bytes too — so several started together can't pass ARTIFACT_QUOTA_GB.
4. A check a restart cut short starts again (at start-up, and on a sweep),
   instead of "checking" for ever with its part held.
5. An untrusted GGUF header keeps only the fields the board reads: a string
   to 256 characters, never an array; a malformed one gives None.
6. /v1 streamed: a client that leaves stops its reply and frees its model.
7. /v1 not streamed: the same.
8. A served model's own error text never reaches the API's caller.

No model runs; fake files, a fake backend and a fake server."""

from __future__ import annotations

import asyncio
import json
import struct

import time

import pytest

import gguf_header as gh
from conftest import make_service
from fake_openai import FakeServer
from service import api_v1, chat, config, uploads
from test_16_1_sizes import gguf_bytes
from test_playground_12d1 import SMALL, svc  # noqa: F401

ME = "masein"


# ---------------------------------------------------------------------------
# 3, 4: uploads
# ---------------------------------------------------------------------------

@pytest.fixture
def up(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(uploads, "PIECE", 4096)
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "UPLOAD_FREE_GB", 0.0)
    yield client
    client.__exit__(None, None, None)


def test_the_quota_counts_the_uploads_still_coming(up, monkeypatch):
    client = up
    monkeypatch.setattr(uploads, "used", lambda: 0)
    monkeypatch.setattr(config, "ARTIFACT_QUOTA_GB", 100_000 / 1e9)       # 100 kB in all
    a = client.post("/api/uploads", json={"filename": "a.gguf", "size": 60_000, "by": ME})
    assert a.status_code == 200
    # nothing of a is on the disk yet: its 60 kB still count
    r = client.post("/api/uploads", json={"filename": "b.gguf", "size": 60_000, "by": ME})
    assert r.status_code == 507
    assert "60.0 kB" not in r.text and "more are on their way" in r.json()["detail"]
    # its own bytes are never counted twice as it comes
    assert client.put(f"/api/uploads/{a.json()['id']}?offset=0",
                      content=b"x" * 4096).status_code == 200


def test_a_check_a_restart_cut_short_starts_again(up):
    client = up
    data = gguf_bytes() + b"\0" * 5000
    rec = client.post("/api/uploads", json={"filename": "p.gguf", "size": len(data),
                                            "by": ME}).json()
    client.put(f"/api/uploads/{rec['id']}?offset=0", content=data[:4096])
    client.put(f"/api/uploads/{rec['id']}?offset=4096", content=data[4096:])
    # the restart: "checking" on disk, and no thread in this process doing it
    r = json.loads(uploads._rec_path(rec["id"]).read_text())
    r["state"] = "checking"
    uploads._rec_path(rec["id"]).write_text(json.dumps(r))
    assert client.get(f"/api/uploads/{rec['id']}").json()["state"] == "checking"
    assert uploads.resume_checks() == [rec["id"]]
    for _ in range(200):
        got = client.get(f"/api/uploads/{rec['id']}").json()
        if got["state"] != "checking":
            break
        time.sleep(0.02)
    assert got["state"] == "ready" and got["header"]["arch"] == "qwen3moe"
    # and the service does this when it starts
    import inspect

    import service.app as appmod
    assert "uploads.resume_checks()" in inspect.getsource(appmod.lifespan)


# ---------------------------------------------------------------------------
# 5: an untrusted header
# ---------------------------------------------------------------------------

def _s(x: bytes) -> bytes:
    return struct.pack("<Q", len(x)) + x


def _kv(k: str, t: int, v: bytes) -> bytes:
    return _s(k.encode()) + struct.pack("<I", t) + v


def header(kvs: list[bytes], n_tensors: int = 0) -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) + \
        struct.pack("<Q", len(kvs)) + b"".join(kvs)


def test_an_untrusted_header_keeps_only_what_the_board_reads(tmp_path):
    big = b"x" * (8 * 1024 * 1024)                       # 8 MB of something
    f = tmp_path / "big.gguf"
    f.write_bytes(header([
        _kv("general.architecture", 8, _s(b"llama")),
        _kv("general.name", 8, _s(b"n" * 5000)),
        _kv("general.description", 8, _s(big)),
        # an array of 200,000 strings, and the chat template
        _kv("tokenizer.ggml.tokens", 9, struct.pack("<I", 8) + struct.pack("<Q", 200_000)
            + _s(b"tok") * 200_000),
        _kv("tokenizer.chat_template", 8, _s(b"{{ messages }}" * 1000)),
        _kv("llama.context_length", 4, struct.pack("<I", 8192))]))
    got = gh.read(f)
    assert got["arch"] == "llama" and got["context_length"] == 8192
    assert got["chat_template"] is True
    assert len(got["name"]) == gh.MAX_KEEP                 # kept to 256 characters
    assert len(json.dumps(got)) < 2000                     # nothing big came back


@pytest.mark.parametrize("kvs", [
    # arrays nested deeper than any model's
    [_kv("x.deep", 9, struct.pack("<I", 9) + struct.pack("<Q", 1)
         + struct.pack("<I", 9) + struct.pack("<Q", 1) + struct.pack("<I", 9)
         + struct.pack("<Q", 1) + struct.pack("<I", 9) + struct.pack("<Q", 1)
         + struct.pack("<I", 4) + struct.pack("<Q", 0))],
    # a string that says it is longer than the file
    [_kv("general.name", 8, struct.pack("<Q", 10_000_000) + b"short")],
    # a value type that doesn't exist
    [_kv("general.name", 99, b"")]])
def test_a_malformed_header_gives_none(tmp_path, kvs):
    f = tmp_path / "bad.gguf"
    f.write_bytes(header(kvs))
    assert gh.read(f) is None


# ---------------------------------------------------------------------------
# 6, 7, 8: /v1
# ---------------------------------------------------------------------------

def _key(client) -> dict:
    k = client.post("/api/keys", json={"by": ME}).json()["key"]
    return api_v1.auth("Bearer " + k)


def test_a_streamed_client_that_leaves_stops_its_reply_and_frees_the_model(svc, monkeypatch):  # noqa: F811
    client = svc
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.05)
    key = _key(client)
    st, ctx, turn = api_v1.begin({"model": SMALL, "messages": [
        {"role": "user", "content": "a long answer please"}]}, key)
    lock = api_v1._model_lock(SMALL)
    assert lock.locked()
    seen = {"n": 0}

    async def gone():
        seen["n"] += 1
        return seen["n"] > 2                             # the client goes after a moment

    async def read():
        return [c async for c in api_v1.astream(st, ctx, key, turn, gone)]
    asyncio.run(read())
    assert not lock.locked() and st.stop.is_set()
    for _ in range(100):
        if st.done:
            break
        time.sleep(0.02)
    assert st.done                                       # the reply stopped, not run out
    turn.end()                                           # once is enough: again is nothing


def test_a_whole_reply_whose_client_leaves_stops_and_frees_the_model(svc, monkeypatch):  # noqa: F811
    from service import app as appmod
    client = svc
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.05)
    k = client.post("/api/keys", json={"by": ME}).json()["key"]

    class Left:
        """a request whose client goes half a second in"""
        def __init__(self):
            self.t0 = time.time()

        async def json(self):
            return {"model": SMALL, "messages": [{"role": "user", "content": "go on"}]}

        async def is_disconnected(self):
            return time.time() - self.t0 > 0.5
    t0 = time.time()
    asyncio.run(appmod.v1_chat(Left(), authorization=f"Bearer {k}"))
    assert time.time() - t0 < 5
    assert not api_v1._model_lock(SMALL).locked()


def test_a_served_models_own_error_never_reaches_the_caller(svc):  # noqa: F811
    client = svc
    k = client.post("/api/keys", json={"by": ME}).json()["key"]
    fake = FakeServer()
    try:
        r = client.post("/api/served", json={"name": "LDA err", "base_url": fake.base,
                                             "how": "k4", "thinking": "off", "by": ME})
        mid = r.json()["model"]["id"]
        fake.chat_error = lambda body: (500, "SECRET upstream detail at /home/masein/lda")
        r = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {k}"},
                        json={"model": mid, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 502
        assert r.json()["error"]["message"] == (
            "LDA err's server failed on this request. Try again; if it goes on, its server may "
            "need a look.")
        assert "SECRET" not in r.text and "/home" not in r.text
    finally:
        fake.close()

