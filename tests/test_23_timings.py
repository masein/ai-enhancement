"""23 part 2: llama-server's timings kept on every answer from the next run
on — prompt_n, prompt_ms, predicted_n, predicted_ms and cache_n as its reply
gives them, prompt_tokens from its usage, and wall_s as the board saw the
request — through the bundle, the fetch (which copies the file as it is) and
the import into the board's stored answers; each task's setup.json saying the
answers in flight and the server's slots. Old lines without them import as
before. The server is the stand-in llama-server; nothing is fetched or run."""

from __future__ import annotations

import json

import agent_relay
import import_frontier as imf
import import_remote as ir
import remote_bundle as rb
from service import config, served
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (ROW, TASK, box, bundle_of, register,  # noqa: F401
                              run_box)

KEYS = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms")


def test_the_same_timings_as_the_agent_relay_keeps():
    assert served.TIMINGS == agent_relay.TIMINGS == KEYS
    assert served.timings_of({"prompt_n": 4, "prompt_ms": 1.5, "draft_n": 3, "cache_n": True,
                              "predicted_ms": "9"}) == {"prompt_n": 4, "prompt_ms": 1.5}


def test_timings_go_from_the_box_through_the_import_onto_the_boards_answers(box, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_TIMINGS", "1")
    assert run_box(box, "t") == 0
    path = bundle_of(box, "t")
    b = rb.read(path)
    name = f"results/{ROW}/{TASK}_0shot/{sf.SUB}/{sf.ANSWERS}"
    lines = [json.loads(x) for x in b["files"][name].decode().splitlines() if x.strip()]
    for r in lines:
        assert {k: r[k] for k in KEYS} == {"cache_n": 8, "prompt_n": 42, "prompt_ms": 31.5,
                                           "predicted_n": 12, "predicted_ms": 240.25}
        assert r["prompt_tokens"] == 50 and isinstance(r["wall_s"], float) and r["wall_s"] >= 0
        assert "prompt_per_second" not in r and not imf.bad_line(r, 4)
    tsetup = json.loads(b["files"][f"results/{ROW}/{TASK}_0shot/{sf.SUB}/{sf.SETUP}"])
    assert tsetup["concurrency"] == {"in_flight": 3, "server_slots": 3}
    # the fetch copies the file as it is; the import keeps every field
    copy = box["root"] / "home" / path.name
    copy.parent.mkdir()
    copy.write_bytes(path.read_bytes())
    register(box["sha"])
    assert ir.import_bundle(copy, "masein", lambda s: None) == 0
    stored = sf.read_answers(sf.task_dir(config.OUT_DIR / ROW, TASK) / sf.ANSWERS)
    assert len(stored) == len(lines)
    assert all(r["prompt_ms"] == 31.5 and r["predicted_n"] == 12 and "wall_s" in r
               for r in stored.values())


def test_an_old_bundle_without_timings_imports_as_before(box):  # noqa: F811
    assert run_box(box, "old") == 0
    b = rb.read(bundle_of(box, "old"))
    name = f"results/{ROW}/{TASK}_0shot/{sf.SUB}/{sf.ANSWERS}"
    for raw in b["files"][name].decode().splitlines():
        r = json.loads(raw)
        old = {k: v for k, v in r.items() if k not in (*KEYS, "prompt_tokens", "wall_s")}
        assert imf.bad_line(old, 4) == ""
    register(box["sha"])
    assert ir.import_bundle(bundle_of(box, "old"), "masein", lambda s: None) == 0
    assert sf.read_answers(sf.task_dir(config.OUT_DIR / ROW, TASK) / sf.ANSWERS)


def test_a_timing_that_isnt_a_number_is_refused():
    line = {"id": "q", "epoch": 0, "answer": "x"}
    assert imf.bad_line({**line, "prompt_ms": 12.5, "wall_s": 3.0, "cache_n": 0}, 1) == ""
    assert imf.bad_line({**line, "prompt_ms": "12"}, 1) == "has a prompt_ms that isn't a number"
    assert imf.bad_line({**line, "wall_s": -1}, 1) == "has a wall_s that isn't a number"
    assert imf.bad_line({**line, "predicted_n": True}, 1) == \
        "has a predicted_n that isn't a number"
