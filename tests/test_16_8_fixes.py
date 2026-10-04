"""16.8: what the deployed board showed after phase 16.

- **A served model's thinking row takes its model's size** (16.1 missed it):
  sizes are entered for the model, "served/X", and the thinking row,
  "served/X · thinking", looked for one of its own. It now takes its model's
  row's — the row its folder names (base_model), or its id without
  " · thinking" — right after a size entered for itself, and a served row's
  name is never read, in the served lookup or not (12f.1).
- **/api/gpu names a host's process by nvidia-smi's own name for it** (16.2
  missed it): the container can't read a host process's /proc, so the judge's
  vLLM read as "other". The lines below are the server's.

No model runs; a fake server."""

from __future__ import annotations

import json

import pytest

import report_lm_eval as rep
from conftest import make_service
from fake_openai import FakeServer
from generative_fixture import write_run
from service import config, served, sizes

ME = "masein"


# ---------------------------------------------------------------------------
# a thinking row's size
# ---------------------------------------------------------------------------

def test_a_thinking_row_takes_its_models_size_and_never_its_name():
    entered = {"served/lda": {"total": 35e9, "active": 3e9, "by": ME}}
    lookup = {"served/lda": {"based_on": "Qwen/Qwen3.6-35B-A3B"},
              "served/lda · thinking": {"based_on": "Qwen/Qwen3.6-35B-A3B"}}
    by_model = {"org/m-2b": {"num_params": 2e9}}

    def s(mid, r=None, lk=lookup, e=entered):
        return rep.size_of(mid, r or {}, lk, e, {}, by_model)
    want = {"total": 35e9, "active": 3e9, "src": "entered", "by": ME}
    # the row its folder names, or its id without " · thinking"
    assert s("served/lda · thinking", {"archinfo": {"base_model": "served/lda"}}) == \
        {**want, "of": "served/lda"}
    assert s("served/lda · thinking") == {**want, "of": "served/lda"}
    # in the served lookup or not
    assert s("served/lda · thinking", lk={}) == {**want, "of": "served/lda"}
    # a size entered for the thinking row itself comes first
    own = {**entered, "served/lda · thinking": {"total": 36e9, "by": "omar"}}
    assert s("served/lda · thinking", e=own)["total"] == 36e9
    # its model has none: no size, its name never read — in the lookup or not
    for lk in (lookup, {}):
        for mid in ("served/Qwen3.6-35B-A3B-Q4-original-k-8 · thinking",
                    "served/Qwen3.6-35B-A3B-Q4-original-k-8"):
            assert s(mid, lk=lk)["total"] is None, (mid, lk)
    # a Hub model's thinking row: its model's count
    assert s("org/m-2b · thinking")["total"] == 2e9


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client, appmod
    client.__exit__(None, None, None)


@pytest.fixture
def fake():
    s = FakeServer()
    yield s
    s.close()


def rows(client, appmod) -> dict:
    appmod._cache.update(key=None, payload=None, at=0.0)
    return {m["id"]: m for m in client.get("/api/results").json()["models"]}


def thinking_row(rec: dict, in_lookup: bool) -> str:
    """the thinking row a run writes for a served model (runner.py): its folder
    names its model; with the served view, it is in the served lookup too"""
    mdir = write_run(config.OUT_DIR, rec["id"], thinking=True, backend="local-chat-completions")
    meta = {"model": rec["id"] + " · thinking", "base_model": rec["id"], "kind": "instruct",
            "params": None}
    if in_lookup:
        meta.update(served.archinfo(rec))
    (mdir / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return rec["id"] + " · thinking"


@pytest.mark.parametrize("in_lookup", [True, False])
def test_the_deploy_steps_sizes_reach_the_served_thinking_rows(svc, fake, capsys, in_lookup):
    client, appmod = svc
    recs = [client.post("/api/served", json={
        "name": n, "base_url": fake.base, "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
        "thinking": "off", "by": ME}).json()["model"] for n in
        ("Qwen3.6-k4-LDA-MTP", "Qwen3.6-35B-A3B-Q4-original-k-8")]
    think = [thinking_row(served.get(r["id"]), in_lookup) for r in recs]
    m = rows(client, appmod)
    assert all(m[t]["params"] is None for t in think)          # never 35B from its name
    assert sizes.main(["set", "--based-on", "Qwen/Qwen3.6-35B-A3B", "--total", "35B",
                       "--active", "3B", "--by", ME]) == 0
    # the list is the served and GGUF models: a thinking row is its model's
    assert "2 model(s) set" in capsys.readouterr().out
    m = rows(client, appmod)
    for r, t in zip(recs, think):
        assert (m[t]["params"], m[t]["activeParams"], m[t]["paramsSrc"], m[t]["paramsBy"],
                m[t]["paramsOf"]) == (35e9, 3e9, "entered", ME, r["id"]), t
        assert r["id"] not in m or m[r["id"]]["paramsOf"] is None


# ---------------------------------------------------------------------------
# /api/gpu: a host's process named by nvidia-smi's name for it
# ---------------------------------------------------------------------------

# the server's own answer (4 Oct), as `--format=csv` writes it: a header, MiB
SERVER_APPS = """pid, used_gpu_memory [MiB], process_name
3064888, 13238 MiB, VLLM::EngineCore
"""
LLAMA = "/home/masein/llama.cpp-teraformer/build-lda/bin/llama-server"


def holders(client) -> dict:
    return {h["kind"]: h["words"] for h in client.get("/api/gpu").json()["holders"]}


def test_the_judge_is_named_by_nvidia_smis_name_when_its_proc_cant_be_read(svc, monkeypatch):
    from test_16_2_gpu import fake_smi
    client, _ = svc
    import os
    me = os.getpid()
    # the container reads no host process's /proc: every command line is empty
    fake_smi(monkeypatch, apps=SERVER_APPS + f"{me}, 512 MiB, python\n")
    g = client.get("/api/gpu").json()
    assert g["per_process"] is True
    assert [h["words"] for h in g["holders"]] == ["the judge 12.9 GB", "the Playground 0.5 GB"]
    # and in the form the board asks for: no header, no units
    fake_smi(monkeypatch, apps="3064888, 13238, VLLM::EngineCore\n")
    assert holders(client) == {"judge": "the judge 12.9 GB"}


@pytest.mark.parametrize("answering,named", [
    (["served/lda"], "LDA phone build 2.5 GB"),          # one server answers: it is that one
    (["served/lda", "served/lda-think"], "LDA phone build / LDA, thinking 2.5 GB"),  # one server
    (["served/lda", "served/other"], "llama-server 2.5 GB"),     # two answer: which, unknown
    ([], "llama-server 2.5 GB")])
def test_a_llama_server_by_its_name_is_the_one_served_model_answering(svc, monkeypatch,
                                                                      answering, named):
    from test_16_2_gpu import fake_smi
    from service import chat, db
    client, _ = svc
    for mid, name, url in (("served/lda", "LDA phone build", "http://h:8090/v1"),
                           ("served/lda-think", "LDA, thinking", "http://h:8090/v1/"),
                           ("served/other", "Original k-8", "http://h:8091/v1")):
        db.served_put({"id": mid, "name": name, "base_url": url, "how": "k4", "based_on": "",
                       "thinking": "off", "pin": {}})
    monkeypatch.setattr(chat, "served_up", lambda mid: mid in answering)
    fake_smi(monkeypatch, apps=SERVER_APPS + f"77, 2600 MiB, {LLAMA}\n"
                                             "78, 1024 MiB, llama-perplexity\n")
    got = holders(client)
    assert got["served"] == named
    assert got["judge"] == "the judge 12.9 GB" and got["gguf"] == "a GGUF run (llama.cpp) 1.0 GB"


# ---------------------------------------------------------------------------
# the Playground: a served model is never "ready" before its server answers
# ---------------------------------------------------------------------------

def test_a_served_model_is_checking_until_its_server_first_answers(svc, monkeypatch):
    import threading
    import time
    from service import chat, db
    client, _ = svc
    rec = {"id": "served/slow", "name": "slow one", "base_url": "http://127.0.0.1:9/v1",
           "how": "k4", "based_on": "", "thinking": "off", "pin": {}}
    db.served_put(rec)
    served.write_meta(rec)
    gate = threading.Event()
    monkeypatch.setattr(chat, "ping", lambda r: gate.wait(5) or True)
    monkeypatch.setattr(chat, "_health", {})

    def state():
        return client.get("/api/playground/states").json()["states"][rec["id"]]
    # its first check is under way: a state of its own, never "ready"
    assert state() == {"state": "checking", "why": "Checking that its server answers."}
    gate.set()
    for _ in range(100):
        if chat._health.get(rec["id"], {}).get("ok"):
            break
        time.sleep(0.02)
    assert state() == {"state": "ready", "why": ""}
