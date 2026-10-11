"""18d, part 3: the speed test, for the next time it runs (docs/prompts/phase-
18d-eleventh-review-scores-oracle-speed.md, points 20–24). A test a point,
each failing on 0b58df7, against a stand-in server that errors mid-stream,
dies, and reports a short slot context. No model runs; nothing is fetched."""

from __future__ import annotations

import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import frontier as fb
import speed_test as sp
from fake_openai import FakeServer
from test_19_speed import gpqa_items, mmlu_items, right_answers, smi

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "REMOTE-RUNS.md").read_text()


@pytest.fixture
def items(monkeypatch):
    monkeypatch.setattr(fb, "shots_for", lambda task, cat: [])
    return {sp.GPQA: gpqa_items(), sp.MMLU: mmlu_items()}


@pytest.mark.parametrize("how", ["error", "data"])
def test_20_an_error_mid_stream_is_the_requests_never_an_answer(items, how):
    srv = FakeServer()
    try:
        right_answers(srv, words=40, think=10)
        srv.stream_error = (how, 5)
        test = sp.Test([sp.Endpoint(srv.base, 2)], items, "qwen", warmup=0, gpqa=0.4, mmlu=0)
        r = test.go(tick=0.01)
        assert r["gpqa"]["answers"] == 0                          # 0b58df7: each an answer
        assert r["errors"] >= 1 and r["sanity"][sp.GPQA]["of"] == 0
        assert "the server said mid-stream" in r["error_words"]
    finally:
        srv.close()
    line = sp.line_of({"box": "b", "price": 0.5, "engine": "e", "file": "f", "setting": "s",
                       "result": r})
    assert f"{r['errors']} errors (the first: the server said mid-stream" in line


def test_21_an_engine_that_dies_ends_its_setting_with_its_last_words(items, tmp_path):
    srv = FakeServer()
    try:
        right_answers(srv, words=40, think=0)
        srv.stream_delay_s = 0.002
        at = {"t": time.monotonic() + 0.3}
        dead = lambda: ("its engine stopped (exit 9): CUDA error" if time.monotonic() > at["t"]  # noqa: E731
                        else "")
        test = sp.Test([sp.Endpoint(srv.base, 2)], items, "qwen", warmup=0, gpqa=30, mmlu=30,
                       dead=dead)
        t0 = time.monotonic()
        r = test.go(tick=0.01)
        assert time.monotonic() - t0 < 10                        # 0b58df7: waited 60 s
        assert r["stopped"] == "its engine stopped (exit 9): CUDA error"
    finally:
        srv.close()
    line = sp.line_of({"box": "b", "price": 0.5, "engine": "e", "file": "f", "setting": "s",
                       "result": r})
    assert line.startswith("STOPPED EARLY — its engine stopped (exit 9): CUDA error")
    # a real process that exited: said with its log's last lines
    log = tmp_path / "e.log"
    log.write_text("loading\nggml_cuda_init: an illegal memory access\n")
    p = sp.Proc(["true"], {}, log, "http://127.0.0.1:9/v1")
    p.p = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    p.p.wait()
    assert sp.dead_of([p]) == ("its engine stopped (exit 3): loading / ggml_cuda_init: an "
                               "illegal memory access")


def test_22_a_shared_setting_whose_slots_hold_less_than_an_answer_is_refused(tmp_path,
                                                                              monkeypatch,
                                                                              capsys):
    import remote_gguf as rg
    srv = FakeServer()
    srv.ctx = 42_900                                             # pool ÷ slots, as the fork gives
    try:
        assert sp.slot_context(srv.base) == 42_900
        monkeypatch.setattr(sp, "run", smi([("NVIDIA GeForce RTX 5090", 32_607)]))
        monkeypatch.setattr(sp, "questions", lambda task, root: [])
        (tmp_path / "u" / "llama" / "bin").mkdir(parents=True)
        monkeypatch.setattr(rg, "fetch", lambda src, into, say: tmp_path / "x")
        monkeypatch.setattr(rg, "unpack_server", lambda tb, into: {
            "bin": tmp_path / "u" / "llama" / "bin" / "llama-server", "libs": [],
            "root": tmp_path / "u"})
        monkeypatch.setattr(rg, "sha256_cached", lambda *a, **k: "a" * 64)
        monkeypatch.setattr(rg, "version_of", lambda *a, **k: {"build": 1, "commit": "x"})
        monkeypatch.setattr(sp, "llama_pool", lambda *a, **k: (16 * 86_016, "x"))

        class Up(sp.Proc):
            def start(self, timeout):
                self.base = srv.base
                return ""

            def stop(self):
                pass
        monkeypatch.setattr(sp, "Proc", Up)
        monkeypatch.setattr(sp.Test, "go", lambda self, tick=0.05: pytest.fail("it ran"))
        assert sp.main(["5090", "--price", "0.5", "--gguf", "x.gguf", "--server", "s.tar.gz",
                        "--only", "2", "--out", str(tmp_path / "o"),
                        "--files", str(tmp_path / "f")]) == 1
        out = capsys.readouterr().out
        assert "slots 16 shared: not run — each slot holds 42,900 tokens, not the pool's " \
               "1,376,256: a shared setting needs 83,968 a slot" in out   # 0b58df7: ran short
    finally:
        srv.close()


def test_23_the_cuda_a_box_needs_is_its_settings(monkeypatch, capsys):
    monkeypatch.setattr(sp, "run", smi([("NVIDIA H100 80GB HBM3", 81_559)], cuda="12.8"))
    plan = sp.PLANS["h100"]
    llama = [s for s in plan["settings"] if s["engine"] == "llama"]
    assert sp.check_box(plan, sp.gpus(), sp.driver_cuda(), llama) == ""   # 0b58df7: refused
    assert "needs CUDA 13.0" in sp.check_box(plan, sp.gpus(), sp.driver_cuda(), plan["settings"])
    assert sp.main(["h100", "--price", "2.55", "--gguf", "x", "--server", "y", "--only", "1,2",
                    "--dry-run"]) == 0
    b = DOC[DOC.index("**(b) 4× V100"):DOC.index("**(c) H100")]
    assert "max CUDA 12.8+" in b


def test_24_a_preempted_requests_tokens_come_off(items):
    test = sp.Test([], items, "qwen", warmup=0, gpqa=1, mmlu=0)
    test.phase = "gpqa"
    ep = sp.Endpoint("http://127.0.0.1:9/v1", 2)
    ask = sp.Ask(test, ep, sp.GPQA, {"messages": []}, {}, ({"id": "q"}, 0))
    ep.flight.append(ask)
    test.streamed(ask, 50)
    assert test.tokens["gpqa"] == 50 and ask.counted == {"gpqa": 50}
    ask.cut, ask.again = "the shared pool was nearly full", True
    test.done(ask)
    assert test.tokens["gpqa"] == 0                             # 0b58df7: 50, counted again
    # a request cut when its window closed keeps what it streamed
    other = sp.Ask(test, ep, sp.GPQA, {"messages": []}, {}, ({"id": "r"}, 0))
    ep.flight.append(other)
    test.streamed(other, 30)
    other.cut = "the run's windows closed"
    test.done(other)
    assert test.tokens["gpqa"] == 30


def test_24_the_small_ones(tmp_path):
    ep = sp.Endpoint("http://127.0.0.1:9/v1", 64, pool=200_000)
    # 191,000 in flight: room for 2,048 more under 97% of the pool, not 64 × 64
    ep.flight = [types.SimpleNamespace(used=lambda: 191_000)]
    assert ep.room(10) == "memory"                              # 0b58df7: 2,048 kept, not 4,096
    argv = sp.vllm_argv(Path("/v"), Path("/w"), 8100, {"seqs": 32, "tp": 1}, "fp8")
    assert "--enable-prompt-tokens-details" in argv
    assert sp.VLLM_START_S == 1200
    assert sp.OOM.search("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3 MiB on "
                         "device 0: cudaMalloc failed: out of memory")
    assert not sp.OOM.search("error while loading shared libraries: libcudart.so.12")
    line = sp.line_of({"box": "b", "price": 0.5, "engine": "e", "file": "f", "setting": "s",
                       "result": {"gpqa": {"tokens_s": 1, "answers_h": 1},
                                  "mmlu": {"tokens_s": 1, "answers_h": 1, "prompt_s": 1,
                                           "prompt_read_s": 1},
                                  "waited": 0, "cut": 0, "refused": 0, "errors": 0,
                                  "sanity": {sp.GPQA: {"right": 0, "of": 0},
                                             sp.MMLU: {"right": 0, "of": 0}}}})
    assert "peak" not in line                                   # 0b58df7: "peak " and nothing
    src = (ROOT / "scripts" / "speed_test.py").read_text()
    assert 'rb.scrub(f"the weights couldn\'t be fetched' in src


def test_24_admission_for_a_shared_pool_on_the_box_is_said():
    s = DOC[DOC.index("# Speed tests"):]
    assert "Admission control in the box runner" in s
