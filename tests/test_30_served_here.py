"""30: step8804's Mobile and DeviceMark sets on this board's server. A model run
only on rented GPUs (registered by its import, no address) is given a server
here under its own id and name — registering it again by name made another
row — with its runtime, its file's sha256 and how it ran on rented GPUs kept;
another file than the one registered is refused with nothing changed. And the
Mobile suite narrowed to some of its sets: HotpotQA and SQL with thinking on,
as Q4's row has them. The server is the fake llama-server; nothing runs."""

from __future__ import annotations

import json
import time

import pytest

import mobileaibench as mab
from conftest import make_service
from fake_openai import FakeServer
from service import config, db, runner, served

ME = "masein"
MID = "served/step8804-lda-2610"
SHA = "c" * 64
RENTED_HOW = ("step8804.gguf on rented GPUs, llama.cpp 9954 (a4a76a751): llama-server -m "
              "step8804.gguf -c 671744 -np 8 -ngl 99 --jinja -ctk q8_0 -ctv q8_0")
HERE_HOW = ("step8804.gguf on the team's RTX 5090 + CPU experts, llama.cpp 9954 (a4a76a751), "
            "LOOKAHEAD_MOE_ROUTER_NORM=1, -ngl 99 --n-cpu-moe 21 --no-mmap -c 16384 "
            "--parallel 1 --jinja")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    monkeypatch.setattr(config, "SERVED_RETRY_S", 0.3)
    yield client, appmod
    client.__exit__(None, None, None)


@pytest.fixture
def fake():
    s = FakeServer()
    s.model_path = "/home/masein/step8804.gguf"
    s.size = 15_700_000_000                  # its weights, as the server counts them
    s.build = "b9954-a4a76a751"
    yield s
    s.close()


def rented() -> dict:
    """as import_frontier.registered_here keeps a model run only on rented GPUs"""
    rec = {"id": MID, "name": "step8804 (lda runtime)", "base_url": "", "key": "",
           "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto", "phone": False,
           "how": RENTED_HOW, "gguf_path": "", "gguf_flags": "", "gguf_setups": [],
           "pin": {"model": "step8804.gguf", "file": "step8804.gguf", "size": 15_776_378_720,
                   "ctx": 83968, "build": 9954},
           "file_sha256": {"sha256": SHA, "by": ME, "at": 1.0, "name": "step8804.gguf",
                           "size": 15_776_378_720},
           "rented_only": True, "answered": [], "by": ME, "at": 1.0,
           "flags": "-ctk q8_0 -ctv q8_0 --flash-attn on --presence-penalty 1.5",
           "env": "LOOKAHEAD_MOE_ROUTER_NORM=1", "runtime": "lda-2610"}
    db.served_put(rec)
    return rec


def address(client, fake, **k):
    body = {"base_url": fake.base, "how": HERE_HOW, "env": "LOOKAHEAD_MOE_ROUTER_NORM=1",
            "flags": "-ngl 99 --n-cpu-moe 21 --no-mmap -c 16384 --parallel 1 --jinja", "by": ME,
            **k}
    return client.put(f"/api/served/{MID}/address", json=body)


def test_a_model_run_on_rented_gpus_is_served_here_under_its_own_id(svc, fake):
    client, _ = svc
    was = rented()
    r = address(client, fake)
    assert r.status_code == 200, r.text
    rec = served.get(MID)
    # the same row: its id and name, its runtime and its file's sha256
    assert (rec["id"], rec["name"], rec["runtime"]) == (MID, was["name"], "lda-2610")
    assert rec["file_sha256"] == was["file_sha256"]
    # asked over this server now, pinned as it reports itself
    assert rec["base_url"] == fake.base and "rented_only" not in rec
    assert rec["pin"] == {"model": "step8804.gguf", "file": "step8804.gguf",
                          "size": 15_700_000_000, "ctx": 16384, "build": "b9954-a4a76a751"}
    assert served.check_pin(rec) == ""
    assert rec["flags"].startswith("-ngl 99") and rec["env"] == "LOOKAHEAD_MOE_ROUTER_NORM=1"
    assert rec["here"]["by"] == ME
    # how it ran on rented GPUs kept beside it, and both said on hover
    assert rec["rented"] == {k: was[k] for k in ("how", "pin", "flags", "env", "by", "at")}
    assert served.view(rec)["how"] == f"{HERE_HOW} · {served.RENTED_WORDS}{RENTED_HOW}"
    # the queue takes its Mobile and DeviceMark sets
    for suite, part in (("mobile", ""), ("devicemark", "speed")):
        q = client.post("/api/submissions", json={"hf_id": MID, "suite": suite, "part": part})
        assert q.status_code == 200, q.text
    # given another address later (its server started again), the rented record stays
    assert address(client, fake, how=HERE_HOW + " · again").status_code == 200
    assert served.get(MID)["rented"]["how"] == RENTED_HOW


@pytest.mark.parametrize("path,size", [("/home/masein/step4000.gguf", 15_700_000_000),
                                       ("/home/masein/step8804.gguf", 13_592_481_632)],
                         ids=["another name", "another size"])
def test_another_file_is_refused_and_nothing_changes(svc, fake, path, size):
    client, _ = svc
    was = rented()
    fake.model_path, fake.size = path, size
    r = address(client, fake)
    assert r.status_code == 422
    assert "the file registered for served/step8804-lda-2610: another file" in r.json()["detail"]
    assert served.get(MID) == was


def test_an_address_needs_how_its_served_a_name_and_a_registered_model(svc, fake):
    client, _ = svc
    rented()
    assert address(client, fake, how=" ").status_code == 422
    assert address(client, fake, by="").status_code == 422
    r = client.put("/api/served/served/nobody/address", json={"base_url": fake.base,
                                                               "how": HERE_HOW, "by": ME})
    assert r.status_code == 422 and "no served llama-server served/nobody" in r.json()["detail"]
    assert served.get(MID)["base_url"] == ""


def test_a_mobile_run_narrowed_to_hotpotqa_and_sql_with_thinking_on(svc, fake, monkeypatch):
    client, appmod = svc
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    rented()
    assert address(client, fake).status_code == 200
    bad = client.post("/api/submissions", json={"hf_id": MID, "suite": "mobile",
                                                 "tasks": ["mobile_mmlu_pro"]})
    assert bad.status_code == 422 and "not sets of this part of the Mobile suite" in \
        bad.json()["detail"]
    r = client.post("/api/submissions", json={"hf_id": MID, "suite": "mobile", "thinking": True,
                                               "tasks": [mab.SQL, mab.HOTPOT]})
    assert r.status_code == 200, r.text
    assert r.json()["tasks"] == sorted([mab.HOTPOT, mab.SQL])
    fake.reasoning = "Let me think."
    t0 = time.time()
    runner.run_submission(db.get(r.json()["id"]))
    row = db.get(r.json()["id"])
    assert row["status"] == "done", row["error"]
    # the two sets alone, each with its system line, the switch said on
    systems = {b["messages"][0]["content"] for b in fake.requests}
    assert systems == {mab.SYSTEM[mab.HOTPOT], mab.SYSTEM[mab.SQL]}
    assert len(fake.requests) == 2000
    assert all(b["chat_template_kwargs"] == {"enable_thinking": True} for b in fake.requests)
    assert all(b["max_tokens"] == config.EVERYDAY_REASONING_MAX_GEN_TOKS and
               b["temperature"] == 0.0 for b in fake.requests)
    think = config.OUT_DIR / (MID.replace("/", "__") + "__thinking")
    assert {p.name for p in think.iterdir() if p.is_dir()} == {
        f"{mab.HOTPOT}_0shot", f"{mab.SQL}_0shot"}
    assert time.time() - t0 < 120
    appmod._cache.update(key=None, payload=None, at=0.0)
    m = {x["id"]: x for x in client.get("/api/results").json()["models"]}
    assert set(m[MID + " · thinking"]["mab"]) == {mab.HOTPOT, mab.SQL}
    assert json.loads(json.dumps(m[MID + " · thinking"]["mab"][mab.HOTPOT]))["n"] == 1000


def test_a_checkpoint_on_devicemark_is_named_by_it_and_is_never_the_original(tmp_path):
    import devicemark as dm
    from test_12q_devicemark_board import SERVED, _row
    out = tmp_path / "full"
    plain = {**SERVED, "mtp": False}
    _row(out, "served/Qwen3.6-35B-A3B-k4-LDA-phone-build", {"ifeval": 0.6, "mmlu_pro": 0.5,
                                                             "math": 0.4},
         {**plain, "phone": True, "name": "Qwen3.6-35B-A3B k4-LDA (phone build)"})
    _row(out, "served/Qwen3.6-35B-A3B-Q4-original-k-8", {"ifeval": 0.8, "mmlu_pro": 0.5,
                                                          "math": 0.5},
         {**plain, "phone": False, "name": "Qwen3.6-35B-A3B Q4 (original, k=8)"})
    # an id that sorts before the original's: still not its original
    _row(out, "served/A-step8804-lda-2610", {"ifeval": 0.9, "mmlu_pro": 0.9, "math": 0.9},
         {**plain, "phone": False, "name": "step8804 (lda runtime)", "file": "step8804.gguf"})
    _row(out, "served/A-step8804-lda-2610", {"ifeval": 0.9, "mmlu_pro": 0.9, "math": 0.9},
         {**plain, "phone": False, "name": "step8804 (lda runtime)", "file": "step8804.gguf"},
         thinking=True)
    got = {r["id"]: r for r in dm.rows(out)}
    assert got["served/A-step8804-lda-2610"]["label"] == "step8804 · thinking off"
    assert got["served/A-step8804-lda-2610 · thinking"]["label"] == "step8804 · thinking on"
    assert got["served/A-step8804-lda-2610"]["retention"] is None
    assert got["served/Qwen3.6-35B-A3B-k4-LDA-phone-build"]["retention"] == {
        "of": "served/Qwen3.6-35B-A3B-Q4-original-k-8", "ifeval": 0.75, "mmlu_pro": 1.0,
        "math": 0.8}
    assert got["served/Qwen3.6-35B-A3B-k4-LDA-phone-build"]["label"] == \
        "phone build (k4-LDA) · thinking off"
    assert got["served/Qwen3.6-35B-A3B-Q4-original-k-8"]["label"] == \
        "original (k=8) · thinking off"
