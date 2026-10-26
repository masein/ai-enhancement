"""29: our own boxes on the DGX Stations, and NVIDIA Nemotron 3.5 Bolt E4B
measured here (docs/prompts/phase-29-own-boxes-and-bolt.md) — the runtime
card-bolt and its 64K context, a GPU shared with colleagues (never started
while another process uses it, paused while one does, shown on Runs), the
attention a V100 takes, plan D and its pilot, the station's launcher and the
fetch reading the station from the server's config. The build is the
stand-in llama-server (tests/fixtures/fake_llama_server.py) saying it is
upstream's b11555; nvidia-smi is invented; the questions are invented: no
model runs, nothing is fetched, no paid API is called."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import card_check as cc
import frontier as fb
import frontier_box as fx
import frontier_fetch as ff
import gguf_header as gh
import gpu_free
import import_frontier as imf
import remote_bundle as rb
import remote_gguf as rg
from service import config, runner
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import FIX, TASK, box, free_port, invented, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_22_ornith import MMLU, UPSTREAM, VERSION_LINE, invented_mmlu, upstream_tarball

ROOT = Path(__file__).resolve().parent.parent
RT = "card-bolt"
Q4 = "served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M"
Q8 = "served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0"
BF16 = "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"
TEMPLATE = "{# Bolt's own template, invented here #}"
V100 = "Tesla V100-DGXS-32GB"
BUSY = "GPU 0 of D3 (Tesla V100-DGXS-32GB) has another process: pid 4242, python, 1,234 MiB"


@pytest.fixture
def bolt(box, card, monkeypatch):  # noqa: F811
    """a Bolt box: the stand-in says it is upstream's b11555 and serves Bolt's
    template; GPQA Diamond's and MMLU-Pro's questions invented"""
    monkeypatch.setenv("FAKE_LLAMA_VERSION", VERSION_LINE)
    monkeypatch.setenv("FAKE_LLAMA_TEMPLATE", TEMPLATE)
    monkeypatch.setitem(fb.RUNTIMES[RT], "template_sha256",
                        hashlib.sha256(TEMPLATE.encode()).hexdigest())
    monkeypatch.setattr(fb, "_fetch", lambda task: invented_mmlu() if task == MMLU
                        else invented())
    box["tarball"] = upstream_tarball(box["root"])
    return box


class Busy:
    """gpu_free.busy, stood in for: free for its first `free` readings, then
    someone else's process on GPU 0 — each reading counted (the questions in
    flight read it side by side)"""

    def __init__(self, free: int):
        import threading
        self.free, self.n, self.lock = free, 0, threading.Lock()

    def __call__(self, label: str = "", own=frozenset(), baseline_mib=None, run=None) -> str:
        with self.lock:
            self.n += 1
            return "" if self.n <= self.free else BUSY


def run_b(b: dict, out: str, *more: str, served_as: str = Q4, only=(TASK,),
          thinking: str = "on") -> int:
    """remote_gguf.py as a Bolt box on our own GPU runs it — the server's
    config put back after"""
    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR"]}
    lock, env = runner.LOCK, {v: os.environ.get(v) for v in rg.ENV.values()}
    pause = sf.PAUSE
    try:
        return rg.main(["--as", served_as, "--gguf", str(b["gguf"]), "--server", str(b["tarball"]),
                        "--thinking", thinking, "--slots", "3", "--port", str(free_port()),
                        "--load-timeout", "60", "--runtime", RT,
                        *[x for t in only for x in ("--only", t)],
                        "--out", str(b["root"] / out), *more])
    finally:
        sf.PAUSE = pause
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def progress_of(b: dict, out: str) -> dict:
    return json.loads((b["root"] / out / "progress.json").read_text())


def task_dir(b: dict, out: str, task: str = TASK, served_as: str = Q4, on: bool = True) -> Path:
    """a task's folder in the box's own results, under the step's --out"""
    row = served_as.replace("/", "__") + ("__thinking" if on else "")
    return b["root"] / out / "bench" / "results" / "full" / row / f"{task}_0shot" / "frontier"


def answers_of(b: dict, out: str, task: str = TASK) -> list[dict]:
    p = task_dir(b, out, task) / "answers.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ---------------------------------------------------------------------------
# Bolt's runtime: its card, its template, its 64K
# ---------------------------------------------------------------------------

def test_bolts_runtime_is_nvidias_card_and_shipped_defaults_on_upstream_b11555():
    rt = fb.RUNTIMES[RT]
    # NVIDIA's card: temperature 1.0 and top-p 0.95, thinking on or off; its
    # generation_config.json's top-k 64 (masein, 26 Oct); the rest neutral —
    # min-p 0, no presence or repeat penalty — each value's source kept
    assert rt["sampling"] == {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0,
                              "presence_penalty": 0.0, "repeat_penalty": 1.0}
    assert rt["thinking"] == ("on",) and rt["off_for"] == ("ifeval_google", "ifbench_test")
    assert rt["flags"] == ["-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on"]
    assert rt["build"]["tag"] == "b11555" and rt["build"]["commit"] == UPSTREAM
    assert rt["models"] == {Q4: BF16, Q8: BF16} and rt["context"] == 65_536
    assert rt["refuses_env"] == ("LOOKAHEAD_",) and rt["env"] == {}
    # the GGUFs' own template: the BF16 repository's chat_template.jinja
    assert rt["template_sha256"] == \
        "35b7d0641b6fb4ee41c3bc623e85bdb7d583c40f6ae9f24514ae58a9d57119f3"
    words = fb.runtime_words(RT)
    assert set(rt["sampling_from"]) == set(rt["sampling"])
    assert rt["sampling_from"]["top_k"].startswith("generation_config.json, "
                                                   "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16")
    assert rt["sampling_from"]["min_p"].startswith("neutral")
    assert fb.runtime_flags(RT)[6:10] == ["--temp", "1.0", "--top-k", "64"]
    assert "temperature 1.0, top-k 64, top-p 0.95, min-p 0.0, presence penalty 0.0, repeat " \
           "penalty 1.0" in words and "its 64K context" in words
    # a Bolt build runs on it alone, and it runs Bolt alone
    assert fb.runtime_of_model(Q4) == RT and fb.runtime_of_model(Q8) == RT
    assert "alone" in fb.runtime_model_refused("card-ornith", Q4)
    assert "alone" in fb.runtime_model_refused(RT, "served/step4000-lda-2610")
    # thinking off of instruction following alone
    assert fb.runtime_thinking_refused(RT, "off", ["ifeval_google", "ifbench_test"]) == ""
    assert "on only" in fb.runtime_thinking_refused(RT, "off", [TASK])


def test_each_request_says_the_thinking_switch_the_template_defaults_off():
    """Bolt's template reads enable_thinking with a default of off: every
    request says it, on or off"""
    rec = {"id": Q4, "name": Q4, "based_on": BF16, "runtime": RT,
           "base_url": "http://127.0.0.1:9/v1", "pin": {"file": "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"}}
    on, off = sf.settings(rec, TASK, True), sf.settings(rec, "ifeval_google", False)
    assert on["chat_template_kwargs"] == {"enable_thinking": True}
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert on["top_k"] == off["top_k"] == 64 and off["temperature"] == 1.0
    assert sf.family_of(rec) == ""
    # the budget is the benchmark's; each answer's own limit is capped below it
    assert on["max_tokens"] == 81_920


def test_the_capped_limit_is_the_benchmarks_or_what_64k_leaves_after_the_prompt():
    assert fb.context_cap(RT) == 65_536 and fb.context_cap("lda-2610") is None
    assert fb.limit_of(81_920, 1_000, 65_536) == 64_536
    assert fb.limit_of(32_768, 1_000, 65_536) == 32_768                  # the benchmark's
    assert fb.limit_of(81_920, 70_000, 65_536) == 0                      # no room at all
    assert fb.limit_of(81_920, 1_000, None) == 81_920                    # no cap: as before
    # a slot holds 64K, never our 83,968 or ARC-AGI-2's 114,688
    assert fb.slot_context([TASK], True, 65_536) == 65_536
    assert fb.slot_context(["arc_agi2_public"], True, 65_536) == 65_536
    assert fb.slot_context(["ifeval_google"], False, 65_536) == 4_096 + 1_024
    assert fb.slot_context([TASK], True) == 83_968


def test_a_box_records_each_answers_capped_limit(bolt, monkeypatch, tmp_path):
    """max_tokens is min(the benchmark's limit, 65,536 − the prompt), in each
    request and in each answer's line"""
    b = bolt
    launched = tmp_path / "launched.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(launched))
    assert run_b(b, "run") == 0
    reqs = [r for r in requests(b) if r.get("messages")]
    got = answers_of(b, "run")
    assert len(got) == len(reqs) == 12 * fb.BENCH[TASK]["epochs"]
    # every request's max_tokens is 65,536 less its prompt as the server counts it
    for r in reqs:
        text = " ".join(m["content"] for m in r["messages"])
        prompt_n = len(("<bos> <|think|> " + text + " <turn|>").split())
        assert r["max_tokens"] == 65_536 - prompt_n < 81_920
    # and each answer's line keeps the limit it had
    assert sorted(x["limit"] for x in got) == sorted(r["max_tokens"] for r in reqs)
    assert all(x["capped"] is True for x in got)
    setup = json.loads((task_dir(b, "run") / "setup.json").read_text())
    assert setup["context_cap"] == 65_536 and setup["budget"] == 81_920
    # the run's settings: top-k 64 in every request, and where each value came from
    assert all(r["top_k"] == 64 and r["temperature"] == 1.0 and r["top_p"] == 0.95
               for r in reqs)
    assert setup["sampling"]["top_k"] == 64
    assert setup["sampling_from"] == fb.RUNTIMES[RT]["sampling_from"]
    bundle = rb.read(b["root"] / "run" / progress_of(b, "run")["bundle"]["name"])
    assert bundle["setup"]["tasks"][TASK]["sampling_from"]["top_k"].startswith(
        "generation_config.json")
    # the server's slots hold 64K each: -c is 3 of them
    argv = json.loads(launched.read_text().splitlines()[-1])["argv"]
    assert argv[argv.index("-c") + 1] == str(3 * 65_536)


def test_a_prompt_that_fills_the_context_and_one_that_runs_out_are_wrong(bolt, monkeypatch):
    """a context of 400 tokens: one question's prompt fills it — written as no
    room (finish "length", limit 0, counted wrong, never asked); the others
    run out at the limit left after their prompts, and are wrong too"""
    b = bolt
    def items(task):
        got = invented()
        got[5]["question"] += " filler" * 500
        return got
    monkeypatch.setattr(fb, "_fetch", items)
    monkeypatch.setitem(fb.RUNTIMES[RT], "context", 400)
    monkeypatch.setenv("FAKE_LLAMA_OUT_TOKENS", "1000")
    monkeypatch.setattr(rg, "kv_fit", lambda *x, **k: (None, "invented"))
    assert run_b(b, "tiny", "--slots-fit") == 0
    got = answers_of(b, "tiny")
    assert len(got) == 12 * fb.BENCH[TASK]["epochs"]
    assert all(x["finish"] == "length" and x["limit"] < 400 and x["capped"] for x in got)
    none = [x for x in got if x.get("no_room")]
    assert {x["id"] for x in none} == {"rec005"}
    assert all(x["limit"] == 0 and x["answer"] == "" and x["prompt_tokens"] >= 400 for x in none)
    asked = [r for r in requests(b) if r.get("messages")]
    assert len(asked) == len(got) - len(none) and all(0 < r["max_tokens"] < 400 for r in asked)
    # each is wrong, as an answer that runs out of room is for everyone
    assert all(fb.ran_out(x["answer"], x["finish"]) for x in got)


# ---------------------------------------------------------------------------
# the GPU is colleagues' too
# ---------------------------------------------------------------------------

class Smi:
    """nvidia-smi on a station's container, invented: one V100, and whatever
    processes `apps` lists on it"""

    def __init__(self, apps: list[tuple[int, str, int]] | None = None, used: int = 0):
        self.apps, self.used = list(apps or []), used

    def __call__(self, args: list[str]) -> str:
        if args[0].startswith("--query-gpu"):
            return f"0, GPU-aaaa, {V100}, {self.used}, 32768\n"
        return "".join(f"GPU-aaaa, {pid}, /usr/bin/{name}, {mib}\n" for pid, name, mib in self.apps)


def test_a_busy_gpu_is_named_and_a_free_one_says_nothing():
    assert gpu_free.busy("D3.0", run=Smi()) == ""
    said = gpu_free.busy("D3.0", run=Smi([(4242, "python", 1234)], used=1300))
    assert said.startswith("GPU 0 of D3 (Tesla V100-DGXS-32GB) has another process: pid 4242")
    assert "python" in said and "1,234 MiB" in said
    # memory in use with no process to show for it, before a step starts
    assert "MiB in use, by no process this box can see" in gpu_free.busy(
        "D3.2", run=Smi(used=2048))
    assert gpu_free.busy("D3.2", run=Smi(used=12)) == ""        # an idle card's own
    # once a step runs: its own server's process isn't someone else's
    assert gpu_free.busy("D3.0", own={77}, baseline_mib=9000,
                         run=Smi([(77, "llama-server", 9000)], used=9100)) == ""
    assert "pid 4242" in gpu_free.busy("D3.0", own={77}, baseline_mib=9000,
                                       run=Smi([(77, "llama-server", 9000), (4242, "python", 500)],
                                               used=9600))
    # memory beyond the step's own that no process seen explains
    assert "more in use than this step's own" in gpu_free.busy(
        "D3.0", own={77}, baseline_mib=9000, run=Smi(used=12000))
    # a card that can't be read pauses, saying so
    def broken(args):
        raise OSError("nvidia-smi exit 9: no devices")
    assert "couldn't be read" in gpu_free.busy("D3.0", run=broken)


def test_a_gate_reads_the_card_fresh_between_benchmarks_and_once_a_minute_otherwise():
    smi = Smi()
    calls = []
    gate = gpu_free.Gate("D3.0", {77}, 0, run=lambda a: calls.append(a) or smi(a), every=60)
    assert gate() == "" and gate() == ""
    n = len(calls)
    assert gate() == "" and len(calls) == n                     # within the minute: read once
    smi.apps = [(4242, "python", 100)]
    assert gate() == ""                                         # not read again yet
    assert "pid 4242" in gate(fresh=True)                       # between benchmarks: fresh


def test_a_box_whose_gpu_is_busy_refuses_and_says_which(bolt, monkeypatch, capsys):
    b = bolt
    monkeypatch.setattr(gpu_free, "busy", Busy(0))
    with pytest.raises(SystemExit) as e:
        run_b(b, "busy", "--own", "--label", "D3.0")
    assert BUSY in str(e.value.code) and "Nothing was fetched" in str(e.value.code)
    p = progress_of(b, "busy")
    assert p["state"] == "stopped" and "GPU 0 of D3" in p["why"]
    assert not requests(b)                                      # nothing asked, nothing started
    assert not (b["root"] / "files").exists()
    # an own box takes no price, and needs its label
    for more in (["--own", "--label", "D3.0", "--price", "0.5"], ["--own"],
                 ["--own", "--label", "L3"]):
        with pytest.raises(SystemExit) as e:
            run_b(b, "x", *more)
        assert e.value.code == 2
    assert "--own" in capsys.readouterr().err


def test_someone_appearing_between_benchmarks_pauses_the_run_its_answers_kept(bolt, monkeypatch):
    """free at the start and through GPQA Diamond; someone else's process
    before MMLU-Pro: the run pauses (exit 75), progress says why, GPQA's
    answers are in the bundle, MMLU-Pro isn't started; once free, the same line
    carries on and asks MMLU-Pro alone"""
    b = bolt
    busy = Busy(2)          # the start's check, and the one before GPQA Diamond
    monkeypatch.setattr(gpu_free, "busy", busy)
    assert run_b(b, "two", "--own", "--label", "D3.0", only=(TASK, MMLU)) == rg.PAUSED
    p = progress_of(b, "two")
    assert p["state"] == "paused" and BUSY in p["why"] and p["label"] == "D3.0"
    assert "paused before MMLU-Pro" in p["why"]
    assert len(answers_of(b, "two")) == 12 * fb.BENCH[TASK]["epochs"]
    assert not answers_of(b, "two", MMLU)
    bundle = rb.read(b["root"] / "two" / p["bundle"]["name"])
    assert bundle["setup"]["own_box"] == {"station": "D3", "label": "D3.0", "gpu": 0}
    assert bundle["setup"]["where"] == "our own GPU" and "price" not in bundle["setup"]
    # free again: the rest, and nothing asked twice
    n = len(requests(b))
    monkeypatch.setattr(gpu_free, "busy", Busy(10_000))
    assert run_b(b, "two", "--own", "--label", "D3.0", only=(TASK, MMLU)) == 0
    asked = [r for r in requests(b)[n:] if r.get("messages")]
    assert len(asked) == 9 and progress_of(b, "two")["state"] == "whole"


def test_someone_appearing_mid_benchmark_lets_the_answers_in_flight_finish(bolt, monkeypatch):
    b = bolt
    monkeypatch.setattr(gpu_free, "EVERY_S", 0)        # read before every question
    monkeypatch.setattr(gpu_free, "busy", Busy(6))
    assert run_b(b, "mid", "--own", "--label", "D3.0") == rg.PAUSED
    p = progress_of(b, "mid")
    assert p["state"] == "paused" and "GPQA Diamond: paused at" in p["why"]
    kept = answers_of(b, "mid")
    assert len(kept) == 4           # the four that read the card free; none after it
    # every request sent was answered and kept: none cut off
    assert len([r for r in requests(b) if r.get("messages")]) == len(kept)


def test_a_paused_step_shows_on_runs_as_paused_on_our_own_gpu(svc, monkeypatch):  # noqa: F811
    import time
    now = time.time()
    step = {"label": "D3.0", "model": Q4, "step": "D3.0-1", "thinking": "on",
            "tasks": ["hle_text_cais"], "shard": "1/4", "state": "paused",
            "why": BUSY + " — waiting for it to be free", "line": "", "at": now - 30,
            "seen_at": now, "reachable": True, "safe": False, "box_id": "b" * 16, "own": True,
            "started_at": "2026-10-26T08:00:00Z"}
    assert imf.store_boxes([{"asked": ["b" * 16], "reached": ["b" * 16]}, step]) == 1
    got = imf.read_boxes(now)
    run = got["runs"][0]
    assert run["status"] == "paused" and BUSY in run["status_words"]
    assert run["own"] is True and run["where"].startswith("our own GPU · box D3.0")
    assert got["boxes"][0]["own"] is True


# ---------------------------------------------------------------------------
# the attention a V100 takes
# ---------------------------------------------------------------------------

def test_flash_attention_with_q8_0_is_checked_and_a_refused_one_falls_back_recorded(
        bolt, monkeypatch):
    b = bolt
    monkeypatch.setenv("FAKE_LLAMA_REFUSE_CACHE", "q8_0")
    assert run_b(b, "fa") == 0
    p = progress_of(b, "fa")
    bundle = rb.read(b["root"] / "fa" / p["bundle"]["name"])
    srv = bundle["setup"]["server"]
    assert srv["flags"][:6] == ["-ctk", "f16", "-ctv", "f16", "--flash-attn", "on"]
    assert srv["attention"]["words"] == "flash attention on with a f16 cache"
    assert [r["words"] for r in srv["attention"]["refused"]] == \
        ["flash attention on with a q8_0 cache"]
    assert "no kernel for a q8_0 cache" in srv["attention"]["refused"][0]["why"]
    # the import takes the fallback as the runtime's own, and nothing else
    assert imf.runtime_problem(bundle["setup"], {"id": Q4, "runtime": RT}) == ""
    bad = {**bundle["setup"], "server": {**srv, "flags": ["-ctk", "q4_0", "-ctv", "q4_0",
                                                         *srv["flags"][4:]]}}
    assert "or one of its 2 fallbacks" in imf.runtime_problem(bad, {"id": Q4, "runtime": RT})
    # the next step on this box runs what this one found, without asking the card again
    memo = json.loads((b["root"] / "files" / rg.ATTN_FILE).read_text())
    assert [x["flags"][:6] for x in memo.values()] == [["-ctk", "f16", "-ctv", "f16",
                                                        "--flash-attn", "on"]]
    assert [x["flags"] for x in memo.values()] == [srv["flags"]]


def test_the_card_check_names_the_card_for_the_record():
    rec = {"card": {"name": V100, "memory_mib": 32768, "driver": "580.178.04"},
           "probe": {"alone": {"prompt_n": 512, "prompt_tok_s": 2345.0, "output_tok_s": 95.0},
                     "long": {"prompt_n": 16384, "prompt_tok_s": 1890.0},
                     "together": {"n": 8, "output_tok_s": 520.0}}}
    line = cc.record_line(rec, "flash attention on with a q8_0 cache", "7.0")
    assert line == ("the card: Tesla V100-DGXS-32GB · sm_70 · 32,768 MiB · driver 580.178.04 · "
                    "flash attention on with a q8_0 cache · 512 tokens in at 2,345 tok/s, out at "
                    "95 alone and 520 at 8 · 16,384 in at 1,890 tok/s")
    assert cc.compute_cap(lambda cmd: "7.0\n") == "7.0"
    assert cc.compute_cap(lambda cmd: (_ for _ in ()).throw(OSError("old driver"))) == ""


def test_an_own_boxs_spent_is_marked_own_box_never_a_price():
    setup = {"own_box": {"station": "D3", "label": "D3.0", "gpu": 0},
             "tasks": {TASK: {"energy_wh": 12.5, "server_s": 600.0}},
             "card_checks": [{"card": {"name": V100}, "probe": {"alone": {"output_tok_s": 95.0}}}]}
    got = imf.spent_of(setup, TASK, 48)
    assert got["own_box"] is True and "price_per_hour" not in got and "cost_usd" not in got
    assert got["card"]["name"] == V100


# ---------------------------------------------------------------------------
# Bolt's memory: its sliding window and shared caches
# ---------------------------------------------------------------------------

def gemma4_header(tmp: Path) -> Path:
    """a GGUF header shaped as Gemma 4 E4B's: 42 layers, two KV heads, global
    heads of 512 every sixth layer, a 512-token window of 256-wide heads on the
    others, the last 18 sharing earlier layers' caches"""
    import struct

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, s("gemma4")),
          ("gemma4.block_count", 4, struct.pack("<I", 42)),
          ("gemma4.attention.head_count", 4, struct.pack("<I", 8)),
          ("gemma4.attention.head_count_kv", 4, struct.pack("<I", 2)),
          ("gemma4.attention.key_length", 4, struct.pack("<I", 512)),
          ("gemma4.attention.value_length", 4, struct.pack("<I", 512)),
          ("gemma4.attention.key_length_swa", 4, struct.pack("<I", 256)),
          ("gemma4.attention.value_length_swa", 4, struct.pack("<I", 256)),
          ("gemma4.attention.sliding_window", 4, struct.pack("<I", 512)),
          ("gemma4.attention.shared_kv_layers", 4, struct.pack("<I", 18)),
          ("gemma4.attention.sliding_window_pattern", 9,
           struct.pack("<IQ", 7, 42) + bytes((i + 1) % 6 != 0 for i in range(42))),
          ("gemma4.embedding_length", 4, struct.pack("<I", 2560))]
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, t, v in kv:
        body += s(k) + struct.pack("<I", t) + v
    p = tmp / "bolt.gguf"
    p.write_bytes(body)
    return p


def test_the_memory_check_counts_bolts_full_layers_alone(tmp_path):
    shape = gh.shape(open(gemma4_header(tmp_path), "rb"))
    assert shape["arch"] == "gemma4" and shape["kv_layers"] == 24
    assert shape["swa"]["window"] == 512 and sum(shape["swa"]["layers"]) == 35
    # of the 24 layers with a cache of their own, 4 hold the whole context
    per = rg.kv_per_token(shape, "q8_0", "q8_0")
    assert per == 4 * 2 * (512 + 512) * 34 / 32
    assert rg.swa_per_slot(shape, "q8_0", "q8_0") == 20 * 2 * (256 + 256) * 34 / 32 * (512 + 512)
    # 8 slots of 64K: under 5 GB of cache, where every layer counted made 48
    assert per * 65_536 * 8 < 5e9 < 42 * 2 * 1024 * 34 / 32 * 65_536 * 8


# ---------------------------------------------------------------------------
# plan D, its pilot, its estimate
# ---------------------------------------------------------------------------

LINE = ["--as", Q4, "--gguf", "hf://nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB@1db54d5b12533ff1bcaa8e42"
        "cf4c667988a1f1be/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf", "--server-dir", "/opt/llama"]


def test_plan_d_puts_a_quarter_of_every_benchmark_on_each_gpu_longest_first():
    assert list(fx.PLANS["D"]) == ["D3.0", "D3.1", "D3.2", "D3.3"]
    for g, (_label, steps) in enumerate(fx.PLANS["D"].items()):
        assert [s[1] for s in steps] == [*fx.D_ORDER, ("ifeval_google", "ifbench_test")]
        assert {s[2] for s in steps} == {f"{g + 1}/4"} and {s[3] for s in steps} == {8}
        assert [s[0] for s in steps] == ["on"] * 8 + ["off"]
    assert fx.D_ORDER[0] == ("hle_text_cais",) and fx.D_ORDER[-2] == ("simpleqa_epoch",)
    covered = {t for s in fx.PLANS["D"]["D3.0"] if s[0] == "on" for t in s[1]}
    assert covered == set(fb.TASKS)                        # every benchmark, thinking on
    lines = fx.listing("D")
    assert lines[-1].endswith("our own GPUs: no price") and "a guess" in "\n".join(lines)
    assert fx.gpu_refused("D3.0", {"names": [V100]}) == ""
    assert "V100 only" in fx.gpu_refused("D3.0", {"names": ["NVIDIA GeForce RTX 5090"]})


def test_a_plan_d_line_runs_bolt_on_its_runtime_own_and_pinned():
    a = fx.parser().parse_args(["D3.2", *LINE])
    argv = fx.argv_of("D3.2", 1, fx.PLANS["D"]["D3.2"][0], a)
    assert argv[argv.index("--runtime") + 1] == RT and "--own" in argv
    assert argv[argv.index("--server-dir") + 1] == "/opt/llama" and "--price" not in argv
    assert argv[argv.index("--based-on") + 1] == BF16
    assert argv[argv.index("--label") + 1] == "D3.2" and argv[argv.index("--shard") + 1] == "3/4"
    assert argv[argv.index("--pin") + 1] == ("Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf=c8af2b8aea10a690a1d"
                                             "9491db744a9446660173cb38fc489612b556615a69612")
    assert argv[argv.index("--flags") + 1] == ""
    assert argv[-1].endswith("/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M/D3.2-1")
    assert fx.PLAN_PINS["D"]["NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf"] == \
        "64504effb9a71a991e487ac4e4cf827485e78bf577efb0a6edf67ce7e1ba84a8"
    # an own box takes no price
    with pytest.raises(SystemExit):
        fx.main(["D3.0", *LINE, "--price", "0.5"])


def test_a_plan_d_line_doesnt_start_on_a_busy_gpu(monkeypatch):
    monkeypatch.setattr(fx, "gpu_refused", lambda label, info=None: "")
    monkeypatch.setattr(gpu_free, "busy", Busy(0))
    ran = []
    monkeypatch.setattr(fx, "run_box", lambda a, label: ran.append(label) or 0)
    with pytest.raises(SystemExit, match=re.escape(f"D3.0: {BUSY} — nothing is started")):
        fx.main(["D3.0", *LINE])
    assert not ran


def test_a_step_waits_for_the_gpu_and_a_paused_one_is_asked_again(monkeypatch, tmp_path, capsys):
    a = fx.parser().parse_args(["D3.1", *LINE, "--root", str(tmp_path)])
    monkeypatch.setattr(fx, "PLANS", {**fx.PLANS, "D": {"D3.1": [("on", (TASK,), "2/4", 8)]}})
    # free at the step's start; paused, then busy twice, then free
    readings = iter(["", BUSY, BUSY, ""])
    monkeypatch.setattr(gpu_free, "busy", lambda *x, **k: next(readings))
    monkeypatch.setattr(fx.time, "sleep", lambda s: None)
    codes = iter([fx.PAUSED, 0])
    calls = []

    class Done:
        def __init__(self, code):
            self.returncode = code
    monkeypatch.setattr(fx.subprocess, "run", lambda cmd: calls.append(cmd) or Done(next(codes)))
    assert fx.run_box(a, "D3.1") == 0
    assert len(calls) == 2 and calls[0] == calls[1]
    out = capsys.readouterr().out
    assert f"D3.1: {BUSY} — waiting for it to be free" in out
    p = json.loads((tmp_path / "NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M" / "D3.1-1" /
                    "progress.json").read_text())
    assert p["state"] == "paused" and BUSY in p["why"]


def test_the_own_pilot_asks_five_of_each_on_and_ifeval_ifbench_off(monkeypatch, tmp_path):
    a = fx.parser().parse_args(["D3.0", *LINE, "--root", str(tmp_path), "--pilot"])
    on = fx.own_pilot_argv(a, "D3.0", "on", list(fb.TASKS))
    off = fx.own_pilot_argv(a, "D3.0", "off", ["ifeval_google", "ifbench_test"])
    assert on[on.index("--thinking") + 1] == "on" and off[off.index("--thinking") + 1] == "off"
    assert on[on.index("--pilot-each") + 1] == "5" and "--pilot" in on and "--own" in on
    assert [on[i + 1] for i, x in enumerate(on) if x == "--only"] == list(fb.TASKS)
    assert [off[i + 1] for i, x in enumerate(off) if x == "--only"] == ["ifeval_google",
                                                                       "ifbench_test"]
    assert on[-1].endswith("/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M/pilot")


def test_the_own_pilots_estimate_is_both_builds_on_four_gpus():
    paces = {t: {"pace_s": 10.0, "answers": 5, "at_once": 5, "tok_s": 400.0} for t in fb.TASKS}
    got = {"on": {"paces": paces, "server": {
        "slot_context": 65_536, "kv_estimate": {"fit": 23},
        "attention": {"words": "flash attention on with a q8_0 cache"},
        "card_check": {"card": {"name": V100},
                       "probe": {"together": {"n": 8, "output_tok_s": 520.0}}}}},
           "off": {"paces": {"ifeval_google": {"pace_s": 2.0, "answers": 5, "at_once": 5}}}}
    lines = fx.own_estimate(got)
    q4 = sum(fx.ANSWERS[t] * 10.0 for t in fb.TASKS) / 3600 + fx.ANSWERS["ifeval_google"] * 2 / 3600
    assert lines[0] == ("the card: Tesla V100-DGXS-32GB · flash attention on with a q8_0 cache · "
                        "each slot 65,536 tokens")
    assert "tokens a second on one GPU, 8 at once: 520" in lines[1]
    assert "slots of 65,536 tokens this card holds with Q4_K_M: 23" in lines[2]
    assert lines[-1].startswith(f"the full set: Q4_K_M about {q4:.0f} GPU-hours, Q8_0 about "
                                f"{q4 * fx.Q8_GUESS:.0f}")
    assert f"both after about {q4 * (1 + fx.Q8_GUESS) / 4:.0f} h" in lines[-1]


def test_an_own_pilot_with_thinking_off_wants_none():
    r = {"thinking": "off", "thinking_apart": False, "thinking_chars": 0, "reply_chars": 40,
         "reply_tags": False}
    assert rg.not_apart(r) == ""
    assert rg.not_apart({**r, "thinking_apart": True}) == "thinking, with thinking off"
    assert rg.not_apart({**r, "reply_chars": 0}) == "an empty reply"


def test_the_pilot_picks_n_of_every_benchmark(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    got = fb.pilot_items(TASK, tmp_path, 5)
    assert len(got) == 5 and got == fb.pilot_items(TASK, tmp_path, 5)
    assert len(fb.pilot_items(TASK, tmp_path)) == 12        # the 22 pilot's 20, of 12 here


# ---------------------------------------------------------------------------
# the station: its launcher, and the fetch reading it from the server's config
# ---------------------------------------------------------------------------

def fake_bin(tmp: Path, busy: bool) -> Path:
    """docker and nvidia-smi on the station, invented — each call written down"""
    d = tmp / "bin"
    d.mkdir(exist_ok=True)
    log = tmp / "calls.log"
    (d / "docker").write_text(f"""#!/usr/bin/env bash
echo "docker $*" >> {log}
case "$1" in
  inspect) [ -f {tmp}/running ] && {{ echo true; exit 0; }}; exit 1 ;;
  image) exit 0 ;;
  top) echo PID; exit 0 ;;
  *) exit 0 ;;
esac
""")
    apps = "GPU-aaaa, 4242, /usr/bin/python, 1234" if busy else ""
    (d / "nvidia-smi").write_text(f"""#!/usr/bin/env bash
echo "nvidia-smi $*" >> {log}
case "$*" in
  *query-gpu=uuid*) echo GPU-aaaa ;;
  *query-compute-apps*) [ -n "{apps}" ] && echo "{apps}" ;;
  *memory.used*) echo {1300 if busy else 3} ;;
esac
exit 0
""")
    for f in d.iterdir():
        f.chmod(0o755)
    return d


def own_box(tmp: Path, *args: str, busy: bool = False) -> subprocess.CompletedProcess:
    d = fake_bin(tmp, busy)
    return subprocess.run(["bash", str(ROOT / "scripts" / "own_box.sh"), *args],
                          capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PATH": f"{d}:{os.environ['PATH']}",
                               "OWN_WORK": str(tmp / "work"), "HF_TOKEN": "hf_invented29"})


def test_the_launcher_never_starts_on_a_busy_gpu_and_says_whose(tmp_path):
    r = own_box(tmp_path, "check", "D3.0", busy=True)
    assert r.returncode == 1 and "GPU 0 of D3 is busy: pid 4242" in r.stdout
    assert "python, 1234 MiB" in r.stdout
    r = own_box(tmp_path, "start", "D3.0", busy=True)
    assert r.returncode == 1 and "docker run" not in (tmp_path / "calls.log").read_text()
    r = own_box(tmp_path, "check", "D3.1")
    assert r.returncode == 0 and "GPU 1 of D3 is free" in r.stdout


def test_the_launcher_runs_its_own_container_on_its_gpu_as_the_account(tmp_path):
    r = own_box(tmp_path, "start", "D3.2")
    assert r.returncode == 0, r.stderr
    run = next(x for x in (tmp_path / "calls.log").read_text().splitlines()
               if x.startswith("docker run"))
    for want in ("--name evalboard-D3.2", "--gpus device=2", "--pid=host",
                 f"--user {os.getuid()}:{os.getgid()}", "--cap-drop ALL",
                 "--security-opt no-new-privileges", f"-v {tmp_path}/work/workspace:/workspace",
                 "ghcr.io/masein/evalboard-runner:v100", "sleep infinity"):
        assert want in run, want
    (tmp_path / "running").write_text("")
    r = own_box(tmp_path, "run", "D3.2", "--", "--as", Q4, "--gguf", "x.gguf", "--then",
                "--as", Q8, "--gguf", "y.gguf")
    assert r.returncode == 0, r.stderr
    ex = next(x for x in (tmp_path / "calls.log").read_text().splitlines()
              if x.startswith("docker exec"))
    assert "-e HF_TOKEN " in ex and "hf_invented29" not in ex       # by name, never its value
    assert ex.count("python scripts/frontier_box.py D3.2") == 2 and Q8 in ex
    r = own_box(tmp_path, "stop", "D3.2")
    rm = [x for x in (tmp_path / "calls.log").read_text().splitlines() if " rm " in f" {x} "]
    assert rm and all("evalboard-D3.2" in x for x in rm)


def test_the_launcher_never_uses_sudo_or_touches_the_station():
    text = (ROOT / "scripts" / "own_box.sh").read_text()
    code = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
    for word in ("sudo", "reboot", "systemctl", "kill ", "pkill", "apt-get", "netbird",
                 "nvidia-smi -r", "--privileged", "modprobe"):
        assert word not in code, word
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / "own_box.sh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / "own_tunnel.sh")]).returncode == 0


def test_the_fetch_reads_the_station_from_the_servers_config_never_printing_it(tmp_path):
    env = {"EVALBOARD_OWN_D3": "evalboard@203.0.113.29:22:/raid/evalboard/workspace"}
    st = ff.own_station("D3", env=env, dotenv=tmp_path / "none")
    assert ff.box_name(st) == "D3" and ff.root_of(st) == "/raid/evalboard/workspace"
    assert (st["user"], st["host"], st["port"]) == ("evalboard", "203.0.113.29", "22")
    dot = tmp_path / ".env"
    dot.write_text("OTHER=1\nEVALBOARD_OWN_D3='evalboard@203.0.113.29:2222:/raid/b/w'\n")
    assert ff.own_station("D3", env={}, dotenv=dot)["port"] == "2222"
    with pytest.raises(SystemExit, match="EVALBOARD_OWN_D3 isn't set"):
        ff.own_station("D3", env={}, dotenv=tmp_path / "none")
    # a rented box as before
    m = ff._BOX.fullmatch("203.0.113.7:41022=L1,L2")
    assert ff.box_name(m) == "203.0.113.7:41022" and ff.root_of(m) == "/workspace"
    assert ff._BOX.fullmatch("evalboard@203.0.113.29:22=D3.0,D3.1")


def test_the_listing_reads_an_own_stations_work_folder(tmp_path):
    root = tmp_path / "raid" / "workspace"
    step = root / "NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M" / "D3.0-1"
    step.mkdir(parents=True)
    (root / "files").mkdir()
    (step / "progress.json").write_text(json.dumps({"label": "D3.0", "state": "paused"}))
    (step / "frontier-x.tar.gz").write_bytes(b"bundle")
    (step.parent / "D3.0.line.json").write_text(json.dumps({"box": "D3.0", "state": "running"}))
    out = subprocess.run([sys.executable, "-"], input=f"ROOT = {str(root)!r}\n" + ff.LISTING,
                         capture_output=True, text=True, check=True).stdout
    got = json.loads(out)
    assert got["steps"] == [{"build": "NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M", "step": "D3.0-1"}]
    assert got["progress"][0]["state"] == "paused" and got["bundles"][0]["path"].endswith(".tar.gz")
    assert got["lines"][0]["build"] == "NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M"


def test_an_own_station_is_done_never_safe_to_destroy_and_its_steps_are_own(tmp_path,
                                                                          monkeypatch):
    root = "/raid/evalboard/workspace"
    d = f"{root}/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M/D3.0-9"
    listing = {"bundles": [], "parity": [], "steps": [{"build": "NVIDIA-Nemotron-3.5-Bolt-E4B-"
                                                                "Q4_K_M", "step": "D3.0-9"}],
               "progress": [{"label": "D3.0", "model": Q4, "state": "paused", "why": BUSY,
                             "tasks": [TASK], "thinking": "on", "dir": d}], "lines": []}
    monkeypatch.setattr(ff, "listing", lambda station, key: (listing, ""))
    st = {"user": "evalboard", "host": "203.0.113.29", "port": "22", "labels": None,
          "root": root, "own": "D3"}
    a = fx.parser().parse_args(["D3.0", *LINE])
    # the board says it holds none of the build's steps (never asked here)
    a.abandoned, a.abandoned_matched, a.by, a.imported = set(), set(), "masein", {Q4: {}}
    safe, failed, lines, steps, got = ff.one_box(st, a, "k", tmp_path / "b", {}, {}, {}, {})
    assert got and not safe and lines[0].startswith("D3 (D3.0): not done — ")
    assert "safe to destroy" not in "\n".join(lines) and "203.0.113.29" not in "\n".join(lines)
    assert steps[0]["own"] is True and steps[0]["state"] == "paused"


# ---------------------------------------------------------------------------
# the mirror holds no address and no key
# ---------------------------------------------------------------------------

KEY = re.compile(r"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY|ssh-(?:ed25519|rsa) AAAA[0-9A-Za-z+/]{40}")
OWN_SET = re.compile(r"EVALBOARD_OWN_[A-Z]\d{1,2}=\S*@")


def test_no_tracked_file_holds_a_key_or_a_stations_address():
    from conftest import needs_checkout  # noqa: F401
    try:
        names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                               check=True).stdout.decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (deploy step 3's copy)")
    found = []
    for n in names:
        p = ROOT / n
        if not n or not p.is_file() or p.stat().st_size > 30_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for k, line in enumerate(text.splitlines(), 1):
            # this file's own made-up lines aside (203.0.113.0/24 is for documentation)
            if KEY.search(line) or (OWN_SET.search(line) and "203.0.113." not in line):
                found.append(f"{n}:{k}")
    assert not found, "a key or a station's address in a tracked file: " + ", ".join(found)


def test_the_check_would_catch_one():
    made = "ssh-ed25519 " + "AAAAC3NzaC1lZDI1NTE5" + "A" * 48 + " evalboard@server"
    assert KEY.search(made) and not KEY.search("ssh-ed25519 AAAA… you@machine")
    # built here, so this file holds no such line itself
    set_line = "EVALBOARD_OWN_D3" + "=" + "evalboard@" + ".".join(["10", "1", "2", "3"]) + ":22:/r"
    assert OWN_SET.search(set_line)
    assert not OWN_SET.search("EVALBOARD_OWN_D3 in the server's .env")


# ---------------------------------------------------------------------------
# the runbook's lines run as written
# ---------------------------------------------------------------------------

def doc_blocks() -> list[str]:
    text = (ROOT / "docs" / "BOLT-RUN.md").read_text()
    return [b.strip() for b in re.findall(r"```bash\n(.*?)```", text, re.S)]


def test_the_runbooks_lines_are_plan_ds_with_the_pinned_files():
    blocks = doc_blocks()
    lines = [b for b in blocks if b.startswith("bash own_box.sh ") and " -- " in b]
    assert {b.split()[2] for b in lines} >= {"pilot", "run", "serve"}
    for b in lines:
        args = b.split(" -- ", 1)[1]
        for part in args.split(" --then "):
            argv = part.split()
            if b.split()[2] == "serve":
                continue
            a = fx.parser().parse_args([b.split()[3], *argv])
            assert a.served_as in (Q4, Q8) and a.server_dir == "/opt/llama"
            name = a.gguf.rsplit("/", 1)[1]
            assert fx.PLAN_PINS["D"][name] and fx.pins_of(a, a.box) == [
                f"{name}={fx.PLAN_PINS['D'][name]}"]
    fetch = next(b for b in blocks if "frontier_fetch.py" in b)
    assert "--own D3" in fetch
    for name, sha in fx.PLAN_PINS["D"].items():
        assert sha in fetch
    tunnel = next(b for b in blocks if "own_tunnel.sh" in b)
    assert tunnel.split()[2:4] == ["D3", "18090"]
    # no address and no token in it
    assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", "\n".join(blocks))
    assert "hf_" not in "\n".join(blocks).replace("hf://", "")
