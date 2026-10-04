"""12d.2: two models side by side, with the fake backend — one message to
both; at once when both fit in free memory, one after the other when they
don't ("waiting for GPU memory"); Stop stops both; practice questions marked
for both; a trained model under its base."""

from __future__ import annotations

import time

import pytest

import everyday as ev
from service import chat, config, db, playground
from test_playground_12d1 import BIG, ME, SMALL, events, plant, svc  # noqa: F401

BIG2 = "fx/other-1.7b-it"
TUNED = "local/chat-1.7b-tuned"


@pytest.fixture
def two(svc):  # noqa: F811
    plant(config.OUT_DIR, BIG2, 1_700_000_000)
    plant(config.OUT_DIR, TUNED, 1_700_000_000)
    yield svc


def compare(client, a, b, by=ME):
    r = client.post("/api/playground/chats", json={"model": a, "model2": b, "by": by})
    assert r.status_code == 200, r.text
    return r.json()


def send(client, c, text, practice=None):
    r = client.post(f"/api/playground/chats/{c['id']}/messages",
                    json={"text": text, "by": ME, "practice": practice})
    assert r.status_code == 200, r.text
    return r.json()


def test_one_message_goes_to_both_and_both_stream(two):
    client = two
    c = compare(client, SMALL, BIG)
    assert c["name2"] == "chat-1.7b-it"
    r = send(client, c, "say hi")
    ea, eb = events(client, r["stream"]), events(client, r["stream_b"])
    assert ea[-1]["t"] == eb[-1]["t"] == "done"
    assert "".join(e.get("d", "") for e in eb if e["t"] == "text").endswith("a word at a time.")
    c = client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).json()
    m = c["messages"][1]
    assert "below-135m-it" in m["replies"][0]["text"] and "chat-1.7b-it" in m["b"]["replies"][0]["text"]
    assert not m.get("pending") and not m["b"].get("pending")
    # both fitted: neither waited for the other
    assert not any(e["t"] == "wait" for e in ea + eb)
    # the chat list names both
    listed = client.get("/api/playground/chats", headers={"X-Who": ME}).json()["chats"][0]
    assert listed["names"] == "below-135m-it vs chat-1.7b-it"


def _until(ok, what: str, timeout: float = 5.0) -> None:
    end = time.time() + timeout
    while not ok():
        assert time.time() < end, f"timed out waiting for {what}"
        time.sleep(0.01)


def _second_started() -> bool:
    return any(st.model == BIG2 and st.events for st in list(chat.ENGINE.streams.values()))


def test_two_models_that_dont_both_fit_answer_one_after_the_other(two, monkeypatch):
    client = two
    # room for one 1.7B model (3.4 GB + the 2 GB margin), not two
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 6 * 10 ** 9)

    # the first answers only once the second has looked: it finds the first
    # still answering, every time, however the threads are scheduled
    def first_holds(mid, msgs, s):
        if mid == BIG:
            _until(_second_started, "the second reply to start")
        return "a short reply"
    monkeypatch.setattr(chat.FakeBackend, "responder", staticmethod(first_holds))
    c = compare(client, BIG, BIG2)
    r = send(client, c, "hello both")
    eb = events(client, r["stream_b"])
    assert eb[0] == {"t": "wait", "why": chat.WAIT_GPU}
    assert eb[-1]["t"] == "done" and eb[-1]["reply"]["words"] > 0
    assert events(client, r["stream"])[-1]["t"] == "done"
    # the first gave its memory back before the second loaded
    assert chat.FakeBackend.loaded == [(BIG, "cuda"), (BIG2, "cuda")]
    assert BIG not in chat.ENGINE.loaded


def test_the_first_gives_its_memory_back_even_when_it_finished_before_the_second_looked(
        two, monkeypatch):
    """the other order: the first reply is over before the second's thread
    looks. Nothing to wait for, but the first model must still be unloaded
    — left loaded, on the real GPU the second would be refused for memory"""
    client = two
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 6 * 10 ** 9)
    run = chat.Engine._run

    def late(self, st, messages, settings, row, on_done, after=None):
        if after is not None:
            _until(lambda: after.done, "the first reply to finish")
        return run(self, st, messages, settings, row, on_done, after)
    monkeypatch.setattr(chat.Engine, "_run", late)
    c = compare(client, BIG, BIG2)
    r = send(client, c, "hello both")
    eb = events(client, r["stream_b"])
    assert eb[0]["t"] == "place" and eb[-1]["t"] == "done" and eb[-1]["reply"]["words"] > 0
    assert not any(e["t"] == "wait" for e in eb)
    assert chat.FakeBackend.loaded == [(BIG, "cuda"), (BIG2, "cuda")]
    assert BIG not in chat.ENGINE.loaded


def test_stop_stops_both(two, monkeypatch):
    client = two
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.02)
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: " ".join(["w"] * 400)))
    c = compare(client, SMALL, BIG)
    r = send(client, c, "go")
    time.sleep(0.2)
    for sid in (r["stream"], r["stream_b"]):
        client.post(f"/api/playground/streams/{sid}/stop")
    ra, rb = events(client, r["stream"])[-1]["reply"], events(client, r["stream_b"])[-1]["reply"]
    assert ra["cut"] == rb["cut"] == "stopped" and ra["words"] < 400 and rb["words"] < 400


def test_again_asks_one_column_again(two):
    client = two
    c = compare(client, SMALL, BIG)
    r = send(client, c, "hi")
    events(client, r["stream"]), events(client, r["stream_b"])
    r = client.post(f"/api/playground/chats/{c['id']}/again", json={"n": 1, "col": "b", "by": ME})
    assert "stream_b" in r.json() and r.json()["stream"] == r.json()["stream_b"]
    events(client, r.json()["stream_b"])
    m = client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).json()["messages"][1]
    assert len(m["replies"]) == 1 and len(m["b"]["replies"]) == 2 and m["b"]["shown"] == 1
    # each model is sent its own replies, never the other's
    seen = []
    chat.FakeBackend.responder = staticmethod(lambda mid, msgs, s: seen.append((mid, msgs)) or "ok")
    r = send(client, c, "and then?")
    events(client, r["stream"]), events(client, r["stream_b"])
    by_model = dict(seen)
    assert "below-135m-it" in by_model[SMALL][1]["content"]
    assert "chat-1.7b-it" in by_model[BIG][1]["content"]


def test_a_practice_question_is_marked_for_both(two, monkeypatch):
    client = two
    q = next(x for x in ev.load_bank() if ev.half(x) == ev.PRACTICE
             and not any(c["type"] == "judge" for c in x["checks"]))
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda mid, msgs, s: q["reference"] if mid == SMALL
                                     else "no idea"))
    c = compare(client, SMALL, BIG)
    r = send(client, c, q["prompt"], {"kind": "everyday", "id": q["id"]})
    events(client, r["stream"]), events(client, r["stream_b"])
    m = client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).json()["messages"][1]
    assert m["replies"][0]["mark"]["words"] == "✓ passes"
    assert m["b"]["replies"][0]["mark"]["pass"] is False


def test_two_at_most_and_two_different_ones(two):
    client = two
    r = client.post("/api/playground/chats", json={"model": SMALL, "model2": SMALL, "by": ME})
    assert r.status_code == 422 and "two different models" in r.json()["detail"]
    r = client.post("/api/playground/chats", json={"model": SMALL, "model2": "fx/good-750m",
                                                   "by": ME})
    assert r.status_code == 422


def test_a_trained_model_is_listed_under_its_base(two):
    client = two
    db.trained_from_set(TUNED, BIG, ME)
    models = client.get("/api/playground").json()["models"]
    ids = [m["id"] for m in models]
    assert ids.index(TUNED) == ids.index(BIG) + 1
    tuned = models[ids.index(TUNED)]
    assert tuned["trained_from"] == BIG and len(tuned["trained_on"]) == 10
    # and it loads through the same loader: its directory, as a run's does
    from service import runner
    assert runner.load_spec(TUNED, {})["pretrained"] == \
        str((config.ARTIFACTS_DIR / "chat-1.7b-tuned").resolve())
    assert [m["id"] for m in playground.models()["models"]].count(TUNED) == 1
