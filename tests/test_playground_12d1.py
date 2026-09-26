"""12d.1: the Playground's engine and API, with the fake backend — a model
that streams canned text. No model runs: every real chat happens on the
server, after deploy.

The engine loads through the runs' load_spec, streams over server-sent
events, answers one message per model at a time, stops at once, unloads
when idle, and gives the GPU to a run: a GPU load is refused while a run
holds the lock, a small model answers on the CPU, and a run taking the lock
unloads chat within CHAT_YIELD_WAIT_S. The settings are the Everyday run's,
from one function. Hidden questions are never reachable."""

from __future__ import annotations

import json
import time

import pytest

import everyday as ev
from conftest import make_service
from service import chat, config, db, playground, runner

ME, OTHER = "masein", "omar"
SMALL = "fx/below-135m-it"                 # the fixture's instruct model: 135M, CPU-able
BIG = "fx/chat-1.7b-it"                    # a 1.7B instruct model, planted below
THINKER = "fx/thinker-0.6b"                # a reasoning model, planted below
BASE = "fx/good-750m"                      # a base model: no chat format


def plant(out_dir, mid, params, **arch):
    d = out_dir / mid.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps({
        "model": mid, "kind": "instruct", "params": params, "ctx": 4096,
        "tmpl_sha": "abc123", **arch}), encoding="utf-8")


def test_a_reply_budget_as_long_as_the_context_still_leaves_room_to_ask():
    row = {"archinfo": {"ctx": 4096}}
    assert chat.context_words(row, {"max_gen_toks": 4096}) == 1536
    assert chat.context_words(row, {"max_gen_toks": 512}) == 2688


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "CHAT_BACKEND", "fake")
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.0)
    monkeypatch.setattr(chat.FakeBackend, "responder", None)
    monkeypatch.setattr(chat.FakeBackend, "loaded", [])
    monkeypatch.setattr(chat, "ENGINE", chat.Engine())
    # the loader's decision without the Hub: no model's own code
    from service import hfmeta
    monkeypatch.setattr(hfmeta, "preflight", lambda hf_id, kind="auto", **k: {
        "remote_code": False, "revision": None, "archinfo": {}})
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 40 * 10 ** 9)
    plant(config.OUT_DIR, BIG, 1_700_000_000)
    plant(config.OUT_DIR, THINKER, 600_000_000, reasoning_template=True, thinking="always",
          ctx=32768)
    yield client
    client.__exit__(None, None, None)


def new_chat(client, model=SMALL, by=ME, **settings):
    r = client.post("/api/playground/chats", json={"model": model, "by": by, "settings": settings})
    assert r.status_code == 200, r.text
    return r.json()


def events(client, stream_id) -> list[dict]:
    out = []
    with client.stream("GET", f"/api/playground/streams/{stream_id}") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def say(client, c, text, by=ME, practice=None):
    r = client.post(f"/api/playground/chats/{c['id']}/messages",
                    json={"text": text, "by": by, "practice": practice})
    assert r.status_code == 200, r.text
    evs = events(client, r.json()["stream"])
    return evs, client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": by}).json()


# ---------------------------------------------------------------------------
# the fake backend: it streams, Stop cuts it off, a closed tab frees it
# ---------------------------------------------------------------------------

def test_a_reply_streams_and_is_kept_with_its_stats(svc):
    client = svc
    c = new_chat(client)
    evs, c = say(client, c, "can u make this shorter pls")
    kinds = [e["t"] for e in evs]
    assert kinds[0] == "place" and kinds[-1] == "done" and kinds.count("text") > 5
    streamed = "".join(e["d"] for e in evs if e["t"] == "text")
    reply = c["messages"][1]["replies"][0]
    assert streamed == reply["text"] and reply["text"].startswith("You asked: can u make")
    assert reply["words"] == len(reply["text"].split()) and reply["wps"] > 0 and reply["secs"] > 0
    assert c["title"] == "can u make this shorter pls" and "pending" not in c["messages"][1]
    # loaded through the runs' load_spec, where the size says: 135M on the GPU when it's free
    assert chat.FakeBackend.loaded == [(SMALL, "cuda")]


def test_stop_cuts_a_reply_off_and_keeps_what_it_wrote(svc, monkeypatch):
    client = svc
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.02)
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: " ".join(f"word{i}" for i in range(400))))
    c = new_chat(client)
    r = client.post(f"/api/playground/chats/{c['id']}/messages", json={"text": "go on", "by": ME})
    sid = r.json()["stream"]
    time.sleep(0.2)
    assert client.post(f"/api/playground/streams/{sid}/stop").json() == {"stopped": True}
    evs = events(client, sid)
    reply = evs[-1]["reply"]
    assert reply["cut"] == "stopped" and 0 < reply["words"] < 400
    # and the model is free for the next message
    assert not chat.ENGINE.loaded[SMALL].busy.locked()


def test_one_reply_per_model_at_a_time_the_second_waits_and_says_so(svc, monkeypatch):
    client = svc
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.01)
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: " ".join(["w"] * 60)))
    a, b = new_chat(client), new_chat(client, by=OTHER)
    ra = client.post(f"/api/playground/chats/{a['id']}/messages", json={"text": "one", "by": ME})
    time.sleep(0.1)
    rb = client.post(f"/api/playground/chats/{b['id']}/messages", json={"text": "two", "by": OTHER})
    eb = events(client, rb.json()["stream"])
    assert {"t": "wait", "why": "answering another message, yours is next"} in eb
    assert eb[-1]["t"] == "done" and eb[-1]["reply"]["words"] == 60
    assert events(client, ra.json()["stream"])[-1]["reply"]["words"] == 60


# ---------------------------------------------------------------------------
# runs come first
# ---------------------------------------------------------------------------

@pytest.fixture
def run_holds(svc):
    """a run holds the GPU lock: Qwen3.5-2B's Standard tests"""
    r = svc.post("/api/submissions", json={"hf_id": "Qwen/Qwen3.5-2B", "suite": "full"})
    assert r.status_code == 200, r.text
    runner.LOCK.mkdir(parents=True, exist_ok=True)
    (runner.LOCK / "submission").write_text(str(r.json()["id"]))
    yield
    import shutil
    shutil.rmtree(runner.LOCK, ignore_errors=True)


def test_with_the_run_lock_held_a_gpu_load_is_refused_in_one_line(svc, run_holds):
    client = svc
    c = new_chat(client, BIG)
    evs, c = say(client, c, "hello")
    assert evs == [{"t": "refused", "why": "The GPU is running Qwen3.5-2B's Standard tests. "
                                           "Chat starts when it's done."}]
    assert chat.FakeBackend.loaded == []
    assert "pending" not in c["messages"][1] and c["messages"][1]["refused"].startswith("The GPU")
    assert client.get("/api/playground/status").json()["run"].startswith("The GPU is running")


def test_a_small_model_answers_on_the_cpu_meanwhile(svc, run_holds):
    client = svc
    evs, c = say(client, new_chat(client, SMALL), "hello")
    assert evs[0] == {"t": "place", "device": "cpu",
                      "note": "on the CPU while a run uses the GPU — slower"}
    assert c["messages"][1]["replies"][0]["device"] == "cpu"
    assert chat.FakeBackend.loaded == [(SMALL, "cpu")]


def test_a_run_taking_the_lock_unloads_chat_within_the_wait(svc, monkeypatch):
    """fake clock: a reply on the GPU is stopped — kept, "cut short" — and the
    GPU model unloads; the run waits no longer than CHAT_YIELD_WAIT_S"""
    client = svc
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.02)
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: " ".join(f"w{i}" for i in range(500))))
    c = new_chat(client, BIG)
    r = client.post(f"/api/playground/chats/{c['id']}/messages", json={"text": "go", "by": ME})
    time.sleep(0.2)
    clock = {"t": 1000.0}
    monkeypatch.setattr(chat, "now", lambda: clock["t"])

    def sleep(s):
        clock["t"] += s
        time.sleep(0.01)
    took = chat.ENGINE.yield_gpu(sleep=sleep)
    assert took <= config.CHAT_YIELD_WAIT_S
    evs = events(client, r.json()["stream"])
    assert evs[-1]["reply"]["cut"] == "a run started" and evs[-1]["reply"]["words"] < 500
    assert BIG not in chat.ENGINE.loaded


def test_a_run_waits_no_longer_than_the_limit_even_if_chat_never_lets_go(svc, monkeypatch):
    """a reply that ignores its stop (a stuck kernel, say): the run goes ahead
    after CHAT_YIELD_WAIT_S of fake time, never longer"""
    clock = {"t": 0.0}
    monkeypatch.setattr(chat, "now", lambda: clock["t"])
    ld = chat.Loaded(BIG, "cuda", {}, 0.0)
    ld.busy.acquire()
    chat.ENGINE.loaded[BIG] = ld

    def sleep(s):
        clock["t"] += s
    assert chat.ENGINE.yield_gpu(sleep=sleep) == pytest.approx(config.CHAT_YIELD_WAIT_S, abs=0.1)
    # and with nothing loaded, it takes no time at all
    chat.ENGINE.loaded.clear()
    assert chat.ENGINE.yield_gpu(sleep=sleep) == 0


def test_a_run_asks_chat_to_let_go_before_the_lock_and_is_never_delayed(svc, monkeypatch):
    client = svc
    from test_everyday_12a import fake_gpu, queue_pilot
    fake_gpu(monkeypatch)
    order = []
    monkeypatch.setattr(chat.ENGINE, "yield_gpu", lambda *a, **k: order.append("yield") or 0.0)
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: order.append("lock") or True)
    sid = queue_pilot(client)
    t0 = time.time()
    runner.run_submission(db.get(sid))
    assert order[:2] == ["yield", "lock"] and db.get(sid)["status"] == "done"
    assert time.time() - t0 < 30
    # a chat that raises cannot break a run
    monkeypatch.setattr(chat.ENGINE, "yield_gpu", lambda *a, **k: 1 / 0)
    sid = queue_pilot(client)
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done"


def test_an_idle_model_unloads_after_the_configured_time(svc, monkeypatch):
    client = svc
    clock = {"t": 5000.0}
    monkeypatch.setattr(chat, "now", lambda: clock["t"])
    say(client, new_chat(client), "hi")
    assert SMALL in chat.ENGINE.loaded
    clock["t"] += config.CHAT_IDLE_UNLOAD_S - 1
    assert chat.ENGINE.sweep() == [] and SMALL in chat.ENGINE.loaded
    clock["t"] += 1
    assert chat.ENGINE.sweep() == [SMALL] and SMALL not in chat.ENGINE.loaded
    assert config.CHAT_IDLE_UNLOAD_S == 600


def test_the_gpu_memory_check_says_why_in_one_line(svc, monkeypatch):
    client = svc
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 3 * 10 ** 9)
    evs, _ = say(client, new_chat(client, BIG), "hi")
    assert evs == [{"t": "refused", "why": "chat-1.7b-it needs about 5 GB of GPU memory and "
                                           "3.0 GB is free now."}]


# ---------------------------------------------------------------------------
# hidden questions are never reachable
# ---------------------------------------------------------------------------

def test_no_playground_endpoint_ever_gives_a_hidden_question(svc):
    client = svc
    import exam_build as eb
    hidden_evd = [q for q in ev.load_bank() if ev.half(q) == ev.HIDDEN]
    hidden_exam = [r for rows in eb.load_bank(config.EXAM_DIR).values() for r in rows
                   if eb.half_of(r["qid"]) == "report"]
    assert hidden_evd and hidden_exam
    body = json.dumps(client.get("/api/playground/practice").json())
    body += json.dumps(client.get("/api/playground").json())
    for q in hidden_evd:
        assert q["id"] not in body and q["prompt"] not in body
    for r in hidden_exam:
        assert r["qid"] not in body and r["prompt"] not in body
    # by id, whatever the parameters: never a practice question, never marked
    c = new_chat(client)
    for ref, prompt, ref_text in [({"kind": "everyday", "id": hidden_evd[0]["id"]},
                                   hidden_evd[0]["prompt"], hidden_evd[0]["reference"]),
                                  ({"kind": "knowledge", "id": hidden_exam[0]["qid"]},
                                   hidden_exam[0]["prompt"], hidden_exam[0]["reference"])]:
        assert playground.practice_item(ref) is None
        evs, cc = say(client, c, prompt, practice=ref)
        assert evs[-1]["reply"]["mark"] is None
        assert ref_text not in json.dumps(cc) and ref_text not in json.dumps(evs)
    for bad in ({"kind": "everyday", "id": None}, {"kind": "exam", "id": hidden_exam[0]["qid"]},
                {"id": hidden_evd[0]["id"]}, "everyday", None):
        assert playground.practice_item(bad) is None


# ---------------------------------------------------------------------------
# practice questions, marked as the run marks them
# ---------------------------------------------------------------------------

def test_an_everyday_practice_question_sent_unedited_is_marked_as_the_run_marks(svc, monkeypatch):
    client = svc
    q = next(x for x in ev.load_bank() if ev.half(x) == ev.PRACTICE
             and not any(c["type"] == "judge" for c in x["checks"]))
    monkeypatch.setattr(chat.FakeBackend, "responder", staticmethod(lambda m, msgs, s: q["reference"]))
    ref = {"kind": "everyday", "id": q["id"]}
    evs, c = say(client, new_chat(client), q["prompt"], practice=ref)
    assert c["messages"][1]["replies"][0]["mark"] == {"kind": "everyday", "pass": True,
                                                     "words": "✓ passes"}
    # a wrong answer: the checks' own plain words
    monkeypatch.setattr(chat.FakeBackend, "responder", staticmethod(lambda m, msgs, s: "no idea"))
    evs, c = say(client, new_chat(client), q["prompt"], practice=ref)
    mk = c["messages"][1]["replies"][0]["mark"]
    assert mk["pass"] is False and mk["words"].startswith("✗ ") and mk["checks"]
    assert mk["words"] == "✗ " + "; ".join(f["why"] for f in ev.failures(q, "no idea"))
    # edited: not marked, and it says why
    evs, c = say(client, new_chat(client), q["prompt"] + " thx", practice=ref)
    assert c["messages"][1]["replies"][0]["mark"] == {"unmarked": "edited, so not marked"}
    # the settings changed: not marked either
    evs, c = say(client, new_chat(client, temperature=0.7), q["prompt"], practice=ref)
    assert c["messages"][1]["replies"][0]["mark"] == {"unmarked": "settings changed, so not marked"}
    # never a score: the board's Everyday results are untouched
    assert client.get("/api/results").json()["everyday"]["models"].get(SMALL) is None


def test_a_knowledge_practice_question_shows_its_reference_folded_and_is_not_judged(svc):
    client = svc
    import exam_build as eb
    r = next(x for rows in eb.load_bank(config.EXAM_DIR).values() for x in rows
             if eb.half_of(x["qid"]) == "diagnose")
    evs, c = say(client, new_chat(client), r["prompt"], practice={"kind": "knowledge",
                                                                    "id": r["qid"]})
    assert c["messages"][1]["replies"][0]["mark"] == {
        "kind": "knowledge", "reference": r["reference"],
        "note": "not marked here; the exam's judge marks it in runs"}


# ---------------------------------------------------------------------------
# settings: the Everyday run's, from one function
# ---------------------------------------------------------------------------

def test_the_default_settings_are_the_everyday_runs_from_the_same_function(svc, tmp_path):
    client = svc
    page = client.get("/api/playground").json()
    by = {m["id"]: m for m in page["models"]}
    for mid in (SMALL, BIG, THINKER):
        row = chat.model_row(mid)
        s = ev.run_settings(row["archinfo"])
        assert by[mid]["scored"] == {"system": s["system"], "thinking": s["thinking"],
                                     "temperature": s["temperature"],
                                     "max_gen_toks": s["max_gen_toks"],
                                     "thinking_on": s["thinking_on"], "can_think": s["can_think"]}
    assert by[SMALL]["scored"]["max_gen_toks"] == 512 and by[SMALL]["scored"]["system"] == ""
    assert by[THINKER]["scored"]["max_gen_toks"] == config.EVERYDAY_REASONING_MAX_GEN_TOKS
    assert by[THINKER]["scored"]["thinking_on"] is True
    # …and the run reads the same function: its budget, and the task it writes
    assert runner._everyday_settings({"archinfo": {"reasoning_template": True}})["max_gen_toks"] \
        == config.EVERYDAY_REASONING_MAX_GEN_TOKS
    ev.build_task(tmp_path / "t")
    y = (tmp_path / "t" / "everyday.yaml").read_text(encoding="utf-8")
    s = ev.run_settings(None)
    assert f"max_gen_toks: {s['max_gen_toks']}" in y and "do_sample: false" in y
    assert f"temperature: {s['temperature']}" in y and 'until: ["\\n\\n\\n\\n"]' in y


def test_a_changed_setting_is_said_and_a_chat_reopens_with_its_own(svc):
    client = svc
    c = new_chat(client)
    assert c["is_scored"] is True and c["settings"] == {}
    r = client.post(f"/api/playground/chats/{c['id']}/settings",
                    json={"settings": {"temperature": 0.7, "system": "Be brief."}, "by": ME})
    assert r.json()["is_scored"] is False
    again = client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).json()
    assert again["settings"] == {"temperature": 0.7, "system": "Be brief."}
    # reset: the scored settings again
    r = client.post(f"/api/playground/chats/{c['id']}/settings", json={"settings": {}, "by": ME})
    assert r.json()["is_scored"] is True
    # thinking on, for a model scored thinking, is still the scored setting
    t = new_chat(client, THINKER)
    r = client.post(f"/api/playground/chats/{t['id']}/settings",
                    json={"settings": {"thinking": True}, "by": ME})
    assert r.json()["is_scored"] is True
    # and its system message reaches the model first
    seen = []
    chat.FakeBackend.responder = staticmethod(lambda m, msgs, s: seen.append(msgs) or "ok")
    client.post(f"/api/playground/chats/{c['id']}/settings",
                json={"settings": {"system": "Be brief."}, "by": ME})
    say(client, c, "hi")
    assert seen[0][0] == {"role": "system", "content": "Be brief."}


# ---------------------------------------------------------------------------
# chats: per person
# ---------------------------------------------------------------------------

def test_chats_are_per_person_and_another_s_cannot_be_read_or_deleted(svc):
    client = svc
    mine = new_chat(client)
    say(client, mine, "my own question")
    theirs = new_chat(client, by=OTHER)
    assert [c["id"] for c in client.get("/api/playground/chats",
                                        headers={"X-Who": ME}).json()["chats"]] == [mine["id"]]
    assert client.get(f"/api/playground/chats/{mine['id']}",
                      headers={"X-Who": OTHER}).status_code == 404
    assert client.post(f"/api/playground/chats/{mine['id']}/delete",
                       json={"by": OTHER}).status_code == 404
    assert client.post(f"/api/playground/chats/{mine['id']}/messages",
                       json={"text": "x", "by": OTHER}).status_code == 404
    # a name is a person whatever its case
    assert client.get(f"/api/playground/chats/{mine['id']}",
                      headers={"X-Who": "Masein"}).status_code == 200
    assert client.post(f"/api/playground/chats/{mine['id']}/delete",
                       json={"by": ME}).json() == {"deleted": mine["id"]}
    assert client.get("/api/playground/chats", headers={"X-Who": ME}).json()["chats"] == []
    assert client.get("/api/playground/chats", headers={"X-Who": OTHER}).json()["chats"][0]["id"] \
        == theirs["id"]


def test_again_keeps_both_replies(svc, monkeypatch):
    client = svc
    n = {"i": 0}
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: n.update(i=n["i"] + 1) or f"reply {n['i']}"))
    c = new_chat(client)
    say(client, c, "hi")
    r = client.post(f"/api/playground/chats/{c['id']}/again", json={"n": 1, "by": ME})
    events(client, r.json()["stream"])
    c = client.get(f"/api/playground/chats/{c['id']}", headers={"X-Who": ME}).json()
    assert [x["text"] for x in c["messages"][1]["replies"]] == ["reply 1", "reply 2"]
    assert c["messages"][1]["shown"] == 1


# ---------------------------------------------------------------------------
# thinking, the picker, long input
# ---------------------------------------------------------------------------

def test_a_reasoning_models_thinking_goes_in_the_fold_and_the_reply_is_without_it(svc):
    client = svc
    evs, c = say(client, new_chat(client, THINKER), "what is 2+2")
    think = "".join(e.get("d", "") for e in evs if e["t"] == "think")
    assert "Let me think this through" in think
    reply = c["messages"][1]["replies"][0]
    assert "Let me think" in reply["thinking"] and "Let me think" not in reply["text"]
    assert reply["text"].startswith("You asked: what is 2+2")


def test_base_models_are_not_offered_and_the_line_says_why(svc):
    client = svc
    page = client.get("/api/playground").json()
    ids = [m["id"] for m in page["models"]]
    assert SMALL in ids and BIG in ids and BASE not in ids
    assert page["left_out"] == ["Base models aren't listed: they have no chat format."]
    r = client.post("/api/playground/chats", json={"model": BASE, "by": ME})
    assert r.status_code == 422 and "base models have no chat format" in r.json()["detail"]


def test_a_message_too_long_for_the_model_is_refused_before_sending(svc):
    client = svc
    c = new_chat(client)                              # ctx 2048, a reply budget of 512
    r = client.post(f"/api/playground/chats/{c['id']}/messages",
                    json={"text": "word " * 2000, "by": ME})
    assert r.status_code == 422
    assert r.json()["detail"] == "Too long for this model: about 1,152 words fits."
