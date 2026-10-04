"""16.2: GPU memory — one status, said the same way everywhere. GET /api/gpu:
the card's total, used and free in GB, and what holds it, from what the board
knows for certain ("other" for the rest, never a guess shown as a fact). A
state for each model in the Playground's picker, worked out without loading
anything. Running out of memory while a chat model loads or answers is
caught: the half-loaded model freed, the message kept, one line and Try
again. Runs wait in the same words, and a card that can't be read is said in
one line. Fakes only: nvidia-smi's answers, free-memory values, a loader that
runs out, a served model that doesn't answer. No GPU."""

from __future__ import annotations

import json
import os
import time

import pytest

from service import chat, config, db, gpu, judge_test, llm, runner, served
from test_playground_12d1 import BIG, SMALL, events, new_chat, run_holds, say, svc  # noqa: F401

MIB = 1024 ** 2


def fake_smi(monkeypatch, card="32607, 19866, 12741", apps: str | None = "",
             cmdlines: dict[int, str] | None = None):
    """nvidia-smi's answers: the card's line, and each process's (None: it fails)"""
    def smi(args, timeout=10.0):
        if args[0].startswith("--query-gpu"):
            if card is None:
                raise FileNotFoundError("nvidia-smi")
            return card + "\n"
        if apps is None:
            raise FileNotFoundError("nvidia-smi")
        return apps
    monkeypatch.setattr(gpu, "_smi", smi)
    monkeypatch.setattr(gpu, "_cmdline", lambda pid: (cmdlines or {}).get(pid, ""))
    monkeypatch.setattr(gpu, "_parent", lambda pid: 1)
    gpu._cache.clear()


# ---------------------------------------------------------------------------
# GET /api/gpu
# ---------------------------------------------------------------------------

def test_the_card_and_each_process_named_by_what_it_runs(svc, monkeypatch, run_holds):  # noqa: F811
    client = svc
    me = os.getpid()
    rec = {"id": "served/lda", "name": "LDA phone build", "base_url": "http://x:8090/v1",
           "how": "k4", "based_on": "", "thinking": "off", "pin": {}}
    db.served_put(rec)
    fake_smi(monkeypatch, apps="\n".join([f"{me}, 3100", "41, 13200", "52, 2600", "63, 400"]),
             cmdlines={41: "python -m vllm.entrypoints.openai.api_server --model gemma",
                       52: "/home/m/llama-server -m x.gguf --port 8090 -ngl 99",
                       63: "some other thing"})
    g = client.get("/api/gpu").json()
    assert g["ok"] and g["line"] == "GPU memory: 19.4 of 32 GB in use · 12.4 GB free"
    assert (g["total_gb"], g["used_gb"], g["free_gb"]) == (31.8, 19.4, 12.4)
    words = {h["kind"]: h["words"] for h in g["holders"]}
    assert words["judge"] == "the judge 12.9 GB"
    assert words["served"] == "LDA phone build 2.5 GB"           # by its port
    assert words["chat"] == "the Playground 3.0 GB"
    assert words["other"] == "other: 0.4 GB"
    # the run holding the lock, though none of its processes is on the card yet
    assert words["run"].startswith("run #") and "Qwen3.5-2B" in words["run"]
    assert g["per_process"] is True


def test_without_memory_for_each_process_the_rest_is_other_never_a_guess(svc, monkeypatch):  # noqa: F811
    client = svc
    fake_smi(monkeypatch, apps="1234, [N/A]\n")
    chat.ENGINE.loaded["fx/chat-1.7b-it"] = chat.Loaded("fx/chat-1.7b-it", "cuda", {}, time.time())
    monkeypatch.setattr(gpu, "_chat_bytes", lambda: 3.5 * 1024 ** 3)
    g = client.get("/api/gpu").json()
    assert g["per_process"] is False
    assert [h["words"] for h in g["holders"]] == ["the Playground: chat-1.7b-it 3.5 GB",
                                                  "other: 15.9 GB"]
    # no judge, no server: the board can't tell them apart without each process's memory
    assert not any(h["kind"] in ("judge", "served") for h in g["holders"])


def test_a_card_that_cant_be_read_is_said_in_one_line(svc, monkeypatch):  # noqa: F811
    fake_smi(monkeypatch, card=None)
    g = svc.get("/api/gpu").json()
    assert g["ok"] is False and g["line"] == "GPU memory can't be read on this server."


def test_it_is_cached_a_few_seconds_and_never_asked_twice_meanwhile(svc, monkeypatch):  # noqa: F811
    asked = []
    fake_smi(monkeypatch)
    real = gpu._smi
    monkeypatch.setattr(gpu, "_smi", lambda a, timeout=10.0: asked.append(a[0]) or real(a))
    for _ in range(5):
        svc.get("/api/gpu")
    assert asked.count("--query-gpu=memory.total,memory.used,memory.free") == 1


# ---------------------------------------------------------------------------
# runs: the same words, and a card that can't be read
# ---------------------------------------------------------------------------

def _waiting_run(monkeypatch, free_mib):
    from test_14_1_mab_text import fake_gpu
    fake_gpu(monkeypatch)
    monkeypatch.setattr(config, "GPU_POLL_S", 0)
    said = []
    real = db.update

    def update(sid, **kw):
        said.append(kw)
        return real(sid, **kw)
    monkeypatch.setattr(db, "update", update)
    asked = iter([False, True])
    monkeypatch.setattr(db, "cancel_requested", lambda sid: next(asked, True))
    monkeypatch.setattr(runner, "gpu_free_mib", free_mib)
    return said


def test_a_run_waits_for_gpu_memory_in_gb(svc, monkeypatch):  # noqa: F811
    client = svc
    said = _waiting_run(monkeypatch, lambda: 1000)
    sid = client.post("/api/submissions", json={"hf_id": "fx/good-750m", "suite": "quick"}).json()["id"]
    runner.run_submission(db.get(sid))
    waits = [kw["progress"] for kw in said if kw.get("status") == "waiting_gpu"]
    # the fake preflight's 2 GB and the 512 MiB margin; 1,000 MiB free
    assert waits == ["Waiting for GPU memory: needs 2.5 GB, 1.0 GB is free."]
    assert db.get(sid)["progress"] == "canceled by request while waiting for GPU memory"


def test_a_card_that_cant_be_read_fails_the_run_in_one_line(svc, monkeypatch):  # noqa: F811
    client = svc

    def broken():
        raise FileNotFoundError("[Errno 2] No such file or directory: 'nvidia-smi'")
    _waiting_run(monkeypatch, broken)
    sid = client.post("/api/submissions", json={"hf_id": "fx/good-750m", "suite": "quick"}).json()["id"]
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed"
    assert row["error"].startswith("GPU memory can't be read on this server (nvidia-smi: ")
    assert row["error"].endswith("Nothing was run.") and "internal error" not in row["error"]


# ---------------------------------------------------------------------------
# the Playground's picker: a state for each model
# ---------------------------------------------------------------------------

def states(client) -> dict:
    return client.get("/api/playground/states").json()["states"]


def test_each_model_has_a_state_worked_out_without_loading_it(svc, monkeypatch):  # noqa: F811
    client = svc
    fake_smi(monkeypatch)
    s = states(client)
    assert s[BIG] == {"state": "loads", "why": "it fits now"}
    assert chat.FakeBackend.loaded == []                       # nothing was loaded
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 2 * 10 ** 9)
    s = states(client)
    assert s[BIG]["state"] == "not_now"
    assert s[BIG]["why"] == "chat-1.7b-it needs about 5.2 GB of GPU memory and 1.9 GB is free now."
    assert s[SMALL]["state"] == "cpu"                          # small: on the CPU, slower
    # loaded: Ready
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 40 * 10 ** 9)
    say(client, new_chat(client, BIG), "hello")
    assert states(client)[BIG] == {"state": "ready", "why": "loaded"}
    # the GPU's line comes with them
    got = client.get("/api/playground/states").json()["gpu"]
    assert got["line"].startswith("GPU memory: ")


def test_a_run_on_the_gpu_is_not_now_with_its_number(svc, run_holds):  # noqa: F811
    s = states(svc)
    assert s[BIG]["state"] == "not_now"
    assert s[BIG]["why"].startswith("Run #1 is using the GPU")
    assert s[SMALL]["state"] == "cpu"


def test_while_chat_hands_the_gpu_to_a_run_not_now_says_so(svc):  # noqa: F811
    chat.ENGINE.yielding = True
    try:
        s = states(svc)
    finally:
        chat.ENGINE.yielding = False
    assert s[BIG] == {"state": "not_now",
                      "why": "A run is starting on the GPU. Chat starts when it's done."}
    assert s[SMALL]["state"] == "cpu"


def test_a_served_model_whose_server_isnt_running_is_not_now(svc, monkeypatch):  # noqa: F811
    client = svc
    rec = {"id": "served/down", "name": "down one", "base_url": "http://127.0.0.1:9/v1", "how": "k4",
           "based_on": "", "thinking": "off", "pin": {}}
    db.served_put(rec)
    served.write_meta(rec)
    up = {"v": False}
    monkeypatch.setattr(chat, "ping", lambda r: up["v"])
    monkeypatch.setattr(chat, "_health", {})
    chat.served_up(rec["id"])                                  # the first ask starts a check
    for _ in range(50):
        if chat._health.get(rec["id"], {}).get("ok") is not None:
            break
        time.sleep(0.02)
    assert states(client)[rec["id"]] == {"state": "not_now", "why": "Its server isn't running."}
    # its key is never in what the picker gets
    assert "key" not in json.dumps(client.get("/api/playground").json())
    up["v"] = True
    chat._health[rec["id"]]["at"] = 0                          # stale: asked again
    chat.served_up(rec["id"])
    for _ in range(50):
        if chat._health[rec["id"]].get("ok"):
            break
        time.sleep(0.02)
    assert states(client)[rec["id"]]["state"] == "ready"


# ---------------------------------------------------------------------------
# running out of GPU memory in the middle
# ---------------------------------------------------------------------------

def test_running_out_while_loading_frees_it_keeps_the_message_and_the_next_chat_works(
        svc, monkeypatch):  # noqa: F811
    client = svc
    monkeypatch.setattr(chat.FakeBackend, "oom", "load")
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 6 * 10 ** 9)
    evs, c = say(client, new_chat(client, BIG), "hello")
    assert evs[0]["t"] == "place" and evs[-1]["t"] == "error" and evs[-1]["oom"] is True
    line = ("Ran out of GPU memory while loading chat-1.7b-it (it needed about 5.2 GB; 5.6 GB "
            "was free). Your message is kept.")
    assert evs[-1]["why"] == line
    m = c["messages"]
    assert m[0]["text"] == "hello" and m[1]["refused"] == line   # kept, with its line
    assert BIG not in chat.ENGINE.loaded                        # nothing half-loaded kept
    # the board's process is fine: the next chat works
    monkeypatch.setattr(chat.FakeBackend, "oom", None)
    evs, _ = say(client, new_chat(client, BIG), "again")
    assert evs[-1]["t"] == "done"


def test_running_out_while_answering_unloads_it_and_says_so(svc, monkeypatch):  # noqa: F811
    client = svc
    monkeypatch.setattr(chat.FakeBackend, "oom", "reply")
    evs, c = say(client, new_chat(client, BIG), "hello")
    assert evs[-1] == {"t": "error", "oom": True, "why": (
        "Ran out of GPU memory while chat-1.7b-it was answering; it was unloaded to give the "
        "memory back. Your message is kept.")}
    for _ in range(50):
        if BIG not in chat.ENGINE.loaded:
            break
        time.sleep(0.02)
    assert BIG not in chat.ENGINE.loaded


def test_in_a_two_model_chat_one_running_out_leaves_the_others_answer(svc, monkeypatch):  # noqa: F811
    client = svc
    calls = {"n": 0}
    real = chat.FakeBackend.load

    def load(self, model_id, spec, device):
        calls["n"] += 1
        if model_id == BIG:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(self, model_id, spec, device)
    monkeypatch.setattr(chat.FakeBackend, "load", load)
    r = client.post("/api/playground/chats", json={"model": SMALL, "model2": BIG, "by": "masein"})
    c = r.json()
    r = client.post(f"/api/playground/chats/{c['id']}/messages", json={"text": "hi", "by": "masein"})
    ea, eb = events(client, r.json()["stream"]), events(client, r.json()["stream_b"])
    assert ea[-1]["t"] == "done" and eb[-1]["t"] == "error" and eb[-1]["oom"] is True


def test_a_reply_that_fails_inside_generate_ends_never_hangs(monkeypatch):
    """HF's generate runs in a thread: its error ends the stream and is raised
    (before 16.2 the stream waited for ever, the model busy). torch and
    transformers stand-ins: the threading is what is tested"""
    import queue
    import sys
    import types

    class Streamer:
        """TextIteratorStreamer's shape: a queue, ended by end()"""
        def __init__(self, tok, skip_prompt=True, skip_special_tokens=True, timeout=None):
            self.q, self.timeout = queue.Queue(), timeout

        def end(self):
            self.q.put(None)

        def __iter__(self):
            while True:
                x = self.q.get(timeout=self.timeout)
                if x is None:
                    return
                yield x
    tr = types.ModuleType("transformers")
    tr.TextIteratorStreamer, tr.StoppingCriteria = Streamer, object
    tr.StoppingCriteriaList = list
    th = types.ModuleType("torch")
    th.inference_mode = lambda: (lambda f: f)
    monkeypatch.setitem(sys.modules, "transformers", tr)
    monkeypatch.setitem(sys.modules, "torch", th)

    class Ids:
        def to(self, device):
            return self

    class Tok:
        def apply_chat_template(self, *a, **k):
            return {"input_ids": Ids(), "attention_mask": Ids()}

    class Model:
        device = "cpu"

        def generate(self, **kw):
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="out of memory"):
        chat.HFBackend().generate({"tok": Tok(), "model": Model()}, [{"role": "user", "content": "x"}],
                                  {"max_gen_toks": 8}, lambda: False, lambda p: None)
    assert time.time() - t0 < 5


# ---------------------------------------------------------------------------
# Test a model: what it needs, beside what is free, and when it starts
# ---------------------------------------------------------------------------

def test_test_a_model_says_what_it_needs_whats_free_and_when_it_starts(svc, monkeypatch):  # noqa: F811
    client = svc
    from service import app as appmod
    from service import hfmeta
    monkeypatch.setattr(hfmeta, "preflight", lambda hf_id, kind="auto", **k: {"need_gb": 9.0})
    monkeypatch.setattr(appmod, "_NEED", {})
    fake_smi(monkeypatch)
    j = client.get("/api/gpu/need", params={"model": "org/m-4b"}).json()
    assert j["line"] == "Needs about 9.5 GB · 12.4 GB free now · starts now"
    fake_smi(monkeypatch, card="32607, 26000, 6607")
    j = client.get("/api/gpu/need", params={"model": "org/m-4b"}).json()
    assert j["when"] == "waits for GPU memory"
    assert client.get("/api/gpu/need", params={"model": "served/x"}).json()["here"] is False


def test_test_a_model_names_the_run_it_would_wait_for(svc, monkeypatch, run_holds):  # noqa: F811
    from service import app as appmod
    from service import hfmeta
    monkeypatch.setattr(hfmeta, "preflight", lambda hf_id, kind="auto", **k: {"need_gb": 2.0})
    monkeypatch.setattr(appmod, "_NEED", {})
    fake_smi(monkeypatch)
    j = svc.get("/api/gpu/need", params={"model": "org/m-1b"}).json()
    assert j["when"].startswith("waits for run #1") and "Qwen3.5-2B" in j["when"]


# ---------------------------------------------------------------------------
# other sections: never a bare error
# ---------------------------------------------------------------------------

def test_the_judge_test_with_a_local_server_down_says_so(svc, monkeypatch):  # noqa: F811
    def down(*a, **k):
        raise llm.LocalUnreachable("nothing is answering at http://localhost:8000/v1 (refused)")
    monkeypatch.setattr(judge_test, "run", down)
    r = svc.post("/api/judge-test/run", json={"models": ["local"], "by": "masein"})
    assert r.status_code == 503
    assert r.json()["detail"].startswith("nothing is answering at http://localhost:8000/v1")
    assert r.json()["detail"].endswith("Nothing was sent.")
