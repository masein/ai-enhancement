"""19: the speed test. The box is checked before anything downloads; the
workload is Frontier's own requests; one streaming client counts tokens as
they come, against one server a GPU or a shared pool, waiting and cutting
instead of letting a shared pool fail; one line a setting, and speed.json;
the engines started as each setting says; vLLM pinned with hashes, 14 days
old. Stand-in servers (FakeServer) for llama-server and vLLM, invented
questions; no model runs, nothing is fetched."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import pytest

import frontier as fb
import speed_test as sp
from fake_openai import FakeServer

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "REMOTE-RUNS.md").read_text()


def smi(cards: list[tuple[str, int]], cuda: str = "12.9", cap: str = "12.0"):
    """nvidia-smi, stood in for: these GPUs, this driver"""
    def run(cmd, timeout=60, env=None):
        if cmd == ["nvidia-smi"]:
            return 0, f"| NVIDIA-SMI 575.57    Driver Version: 575.57    CUDA Version: {cuda}  |"
        if cmd[:2] == ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap"]:
            return 0, "\n".join(f"{i}, {n}, {m}, {cap}" for i, (n, m) in enumerate(cards))
        if cmd[:2] == ["nvidia-smi", "--query-gpu=index,memory.used"]:
            return 0, "\n".join(f"{i}, {m - 1000}" for i, (_, m) in enumerate(cards))
        return 0, ""
    return run


# ---------------------------------------------------------------------------
# the box, before anything downloads
# ---------------------------------------------------------------------------

def test_a_box_that_isnt_what_was_rented_stops_before_anything_downloads(monkeypatch, capsys):
    import remote_gguf as rg
    monkeypatch.setattr(rg, "fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(sp, "run", smi([("Tesla V100-SXM2-16GB", 16_384)] * 4, cap="7.0"))
    assert sp.main(["v100x4", "--price", "0.65", "--gguf", "hf://u/r/x.gguf",
                    "--server", "hf://u/r/s.tar.gz"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("this box: 4× Tesla V100-SXM2-16GB (16,384 MiB, 16,384 MiB, ")
    assert "stopping — this box isn't a 4× V100 32 GB: " in out and "Nothing was fetched" in out
    # a driver too old for vLLM's torch: said with what to rent
    monkeypatch.setattr(sp, "run", smi([("NVIDIA H100 80GB HBM3", 81_559)], cuda="12.8"))
    assert sp.main(["h100", "--price", "2.55", "--gguf", "x", "--server", "y"]) == 2
    out = capsys.readouterr().out
    assert "its driver runs CUDA 12.8; this plan needs CUDA 13.0 or newer" in out
    # the box it is: its plan, and nothing more with --dry-run
    monkeypatch.setattr(sp, "run", smi([("NVIDIA GeForce RTX 5090", 32_607)]))
    assert sp.main(["5090", "--price", "0.50", "--gguf", "hf://u/r/q4.gguf",
                    "--server", "hf://u/r/s.tar.gz", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "5090: RTX 5090 at $0.50/h — 4 settings, each 3 + 20 + 5 minutes" in out
    for s in ("slots 8", "slots 16 shared", "slots 24 shared", "slots 8, MTP"):
        assert f"  · {s}\n" in out


def test_the_tarballs_architectures_are_read_from_its_version(tmp_path):
    (tmp_path / "llama").mkdir()
    (tmp_path / "llama" / "VERSION").write_text("commit abc\ncuda_archs 80;86;89;90;120\n")
    assert sp.tarball_archs(tmp_path) == [80, 86, 89, 90, 120]
    assert 70 not in sp.tarball_archs(tmp_path)               # a V100 can't run it


def test_a_v100_box_refuses_a_tarball_without_sm70_before_the_gguf(tmp_path, monkeypatch, capsys):
    import remote_gguf as rg
    fetched = []
    monkeypatch.setattr(sp, "run", smi([("Tesla V100-SXM2-32GB", 32_768)] * 4, cap="7.0"))
    monkeypatch.setattr(sp, "questions", lambda task, root: [])
    monkeypatch.setattr(rg, "fetch", lambda src, into, say: fetched.append(src) or tmp_path / "t")
    (tmp_path / "u" / "llama").mkdir(parents=True)
    (tmp_path / "u" / "llama" / "VERSION").write_text("cuda_archs 80;86;89;90;120\n")
    monkeypatch.setattr(rg, "unpack_server", lambda t, into: {"root": tmp_path / "u",
                                                               "bin": tmp_path / "x", "libs": []})
    assert sp.main(["v100x4", "--price", "0.65", "--gguf", "hf://u/r/q4.gguf", "--server",
                    "hf://u/r/s.tar.gz", "--files", str(tmp_path / "f")]) == 2
    assert "not this GPU's sm 70: build one with LLAMA_CUDA_ARCHS='70;80;86;89;90;120'" in \
        capsys.readouterr().out
    assert fetched == ["hf://u/r/s.tar.gz"]                    # the GGUF never fetched


# ---------------------------------------------------------------------------
# the workload: Frontier's own requests
# ---------------------------------------------------------------------------

def gpqa_items(n: int = 30) -> list[dict]:
    return [{"id": f"g{i}", "question": f"Invented question {i}?", "right": f"right {i}",
             "wrong": [f"wrong {i}a", f"wrong {i}b", f"wrong {i}c"]} for i in range(n)]


def mmlu_items(n: int = 60) -> list[dict]:
    return [{"id": f"m{i}", "question": f"Invented question {i}?", "category": "invented",
             "options": [f"right {i}", f"wrong {i}"], "answer": "A"} for i in range(n)]


def test_the_workload_is_frontiers_own_requests(monkeypatch):
    monkeypatch.setattr(fb, "shots_for", lambda task, cat: [
        {"question": "Shot?", "options": ["a", "b"], "cot_content": "A: Because."}] * 5)
    g = gpqa_items(1)[0]
    body, need = sp.request_of(sp.GPQA, g, 2, "qwen")
    assert body["max_tokens"] == 81_920 and body["stream"] is True
    assert {k: body[k] for k in ("temperature", "top_p", "top_k", "presence_penalty")} == \
        fb.PRESETS["qwen3.6"]["on"]                            # the card's, thinking on
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["seed"] == fb.seed_of(sp.GPQA, "g0", 2)
    assert body["messages"] == [{"role": "user", "content": fb.prompt(sp.GPQA, g)[0]}]
    assert body["stream_options"]["include_usage"] is True
    m = mmlu_items(1)[0]
    body, need = sp.request_of(sp.MMLU, m, 0, "qwen")
    assert body["max_tokens"] == 8_192 and body["chat_template_kwargs"] == {
        "enable_thinking": False}
    assert body["temperature"] == fb.PRESETS["qwen3.6"]["off"]["temperature"]
    assert body["messages"][0]["content"].count("Question: ") == 6        # 5-shot
    assert need["key"] == "A"


# ---------------------------------------------------------------------------
# one streaming client: tokens as they come, one server a GPU
# ---------------------------------------------------------------------------

def right_answers(srv: FakeServer, words: int = 40, think: int = 30) -> None:
    """a stand-in model that thinks, then answers each question right"""
    def reply(body):
        text = body["messages"][-1]["content"]
        if "ANSWER: LETTER" in text:                          # GPQA: the right option's letter
            q = re.search(r"Invented question (\d+)\?", text)[1]
            letter = re.search(rf"(?m)^([ABCD])\) right {q}$", text)[1]
            return " ".join(["so"] * words) + f"\nANSWER: {letter}"
        return " ".join(["so"] * 10) + " The answer is (A)"
    srv.reply = reply
    srv.reasoning = lambda body: (" ".join(["hmm"] * think)
                                  if body["chat_template_kwargs"]["enable_thinking"] else "")


@pytest.fixture
def items(monkeypatch):
    monkeypatch.setattr(fb, "shots_for", lambda task, cat: [])
    return {sp.GPQA: gpqa_items(), sp.MMLU: mmlu_items()}


def test_one_client_counts_tokens_as_they_stream_across_one_server_a_gpu(items):
    srvs = [FakeServer(), FakeServer()]
    try:
        for s in srvs:
            right_answers(s)
            s.stream_delay_s = 0.004
        eps = [sp.Endpoint(s.base, 3) for s in srvs]
        test = sp.Test(eps, items, "qwen", warmup=0.3, gpqa=1.2, mmlu=0.6)
        r = test.go(tick=0.01)
        g, m = r["gpqa"], r["mmlu"]
        assert g["tokens"] > 0 and g["answers"] > 0 and m["answers"] > 0
        assert g["tokens_s"] == pytest.approx(g["tokens"] / 1.2)
        assert g["answers_h"] == pytest.approx(g["answers"] / (1.2 / 3600))
        # the workload shared between the servers, each at its slots
        assert all(s.answered > 0 and s.max_in_flight <= 3 for s in srvs)
        assert r["sanity"][sp.GPQA]["right"] == r["sanity"][sp.GPQA]["of"] > 0
        assert r["sanity"][sp.MMLU]["right"] == r["sanity"][sp.MMLU]["of"] > 0
        assert (r["waited"], r["cut"], r["refused"], r["errors"]) == (0, 0, 0, 0)
    finally:
        for s in srvs:
            s.close()


def test_an_answer_still_running_when_its_window_closes_counts_its_part(items):
    srvs = [FakeServer()]
    try:
        right_answers(srvs[0], words=2000, think=0)
        srvs[0].stream_delay_s = 0.002
        test = sp.Test([sp.Endpoint(srvs[0].base, 4)], items, "qwen", warmup=0, gpqa=0.5,
                       mmlu=0)
        r = test.go(tick=0.01)
        assert r["gpqa"]["answers"] == 0 and r["gpqa"]["tokens"] > 50       # none finished
    finally:
        for s in srvs:
            s.close()


def test_a_shared_pool_waits_and_cuts_instead_of_failing(items, monkeypatch):
    """llama-server fails every request in flight when its shared pool
    fills: the client admits only what fits, and cuts the request admitted
    last before the pool fills — asked again from its start"""
    monkeypatch.setattr(sp, "NEW_ROOM", 10)
    monkeypatch.setattr(sp, "STEP_ROOM", 1)
    srv = FakeServer()
    try:
        right_answers(srv, words=80, think=0)
        srv.reasoning = None
        srv.stream_delay_s = 0.004
        e = sp.prompt_estimate(sp.request_of(sp.GPQA, items[sp.GPQA][0], 0, "qwen")[0])
        ep = sp.Endpoint(srv.base, 8, pool=2 * e + 50)
        test = sp.Test([ep], items, "qwen", warmup=0, gpqa=2.5, mmlu=0)
        r = test.go(tick=0.005)
        assert r["waited"] >= 1                                # a slot free, no room: it waited
        assert r["cut"] >= 1                                   # cut before the pool filled
        seeds = [b["seed"] for b in srv.requests]
        assert len(seeds) > len(set(seeds))                    # and asked again from its start
        assert r["refused"] == r["errors"] == 0
        assert srv.max_in_flight <= 2
        assert r["gpqa"]["answers"] >= 2                       # and the run went on
    finally:
        srv.close()


def test_a_refusal_for_memory_is_counted(items):
    srv = FakeServer()
    try:
        # llama-server's words when a shared pool fills
        srv.chat_error = lambda body: (500, "Context size has been exceeded.")
        right_answers(srv)
        test = sp.Test([sp.Endpoint(srv.base, 2)], items, "qwen", warmup=0, gpqa=0.4, mmlu=0)
        r = test.go(tick=0.01)
        assert r["refused"] >= 1 and r["errors"] == 0
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# one line a setting, and speed.json
# ---------------------------------------------------------------------------

def record(tokens_s: float = 400.0, right: int = 40, of: int = 50, mtp=None) -> dict:
    return {"box": "RTX 5090 (1× NVIDIA GeForce RTX 5090)", "price": 0.50,
            "engine": "llama.cpp 7000 (abc1234)", "file": "q4.gguf sha256 0123456789abcdef…",
            "setting": "slots 16 shared (a pool of 600,000 tokens a GPU)", "peak_mib": [31_800],
            "mtp": mtp is not None,
            "result": {"gpqa": {"tokens_s": tokens_s, "answers_h": 11.0},
                       "mmlu": {"tokens_s": 900.0, "answers_h": 2400.0, "prompt_s": 12_000.0,
                                "prompt_read_s": 3_000.0},
                       "waited": 3, "cut": 1, "refused": 0, "errors": 0,
                       "mtp_acceptance": mtp,
                       "sanity": {sp.GPQA: {"right": right, "of": of},
                                  sp.MMLU: {"right": 200, "of": 260}}}}


def test_one_line_a_setting_with_dollars_per_million_output_tokens(tmp_path):
    line = sp.line_of(record())
    assert line.startswith("RTX 5090 (1× NVIDIA GeForce RTX 5090) · $0.50/h · llama.cpp 7000 "
                           "(abc1234) · q4.gguf sha256 0123456789abcdef… · slots 16 shared")
    for words in ("GPQA 400 output tokens/s, 11.0 answers/h",
                  "MMLU-Pro 900 output tokens/s, 2,400 answers/h, 12,000 prompt tokens/s "
                  "(3,000 read, the rest from the cache)",
                  "$0.35 per million output tokens",                      # 0.50 / 1.44
                  "peak 31,800 MiB", "3 waited for memory, 1 cut and asked again, 0 refused",
                  "sanity GPQA 40/50, MMLU-Pro 200/260"):
        assert words in line, words
    assert "BROKEN" not in line.upper()
    assert "FAR BELOW OUR RUNS: this setting is broken" in sp.line_of(record(right=5))
    assert "MTP accepted 62% of its drafts" in sp.line_of(record(mtp=0.62))
    # the break-evens in the brief: 4× V100 at $0.65 over ~410 tokens/s
    assert sp.dollars_per_million(0.65, 410) == pytest.approx(0.44, abs=0.005)
    assert sp.dollars_per_million(2.55, 1600) == pytest.approx(0.44, abs=0.005)
    sp.keep(tmp_path, record())
    sp.keep(tmp_path, {**record(tokens_s=500.0)})               # the same setting again
    got = json.loads((tmp_path / "speed.json").read_text())
    assert len(got) == 1 and got[0]["result"]["gpqa"]["tokens_s"] == 500.0


# ---------------------------------------------------------------------------
# the engines, as each setting says
# ---------------------------------------------------------------------------

def test_llama_server_as_each_setting_says():
    per = sp.llama_argv("/s/llama-server", "/f/q4.gguf", 8090, {"slots": 8}, 0)
    assert per[per.index("-c") + 1] == str(83_968 * 8) and "--kv-unified" not in per
    assert per[per.index("-np") + 1] == "8" and "-ngl" in per
    for flag in ("-ctk", "q8_0", "-ctv", "--flash-attn"):
        assert flag in per
    shared = sp.llama_argv("/s/llama-server", "/f/q4.gguf", 8091,
                           {"slots": 24, "shared": True}, 1_200_000)
    assert shared[shared.index("-c") + 1] == "1200000" and "--kv-unified" in shared
    mtp = sp.llama_argv("/s/llama-server", "/f/q4.gguf", 8090, {"slots": 8, "mtp": True}, 0)
    assert mtp[-2:] == ["--spec-type", "draft-mtp"]
    for argv in (per, shared, mtp):                             # never split across GPUs
        assert not {"--tensor-split", "-ts", "--split-mode", "-sm"} & set(argv)


def test_several_gpus_run_one_server_each(tmp_path, monkeypatch, capsys, items):
    import remote_gguf as rg
    started = []

    class Proc:
        def __init__(self, argv, env, log, base):
            self.argv, self.env, self.log, self.base = argv, env, log, base

        def start(self, timeout):
            started.append(self)
            return ""

        def stop(self):
            pass
    monkeypatch.setattr(sp, "Proc", Proc)
    monkeypatch.setattr(sp, "run", smi([("Tesla V100-SXM2-32GB", 32_768)] * 4, cap="7.0"))
    monkeypatch.setattr(sp, "questions", lambda task, root: items[task])
    monkeypatch.setattr(rg, "fetch", lambda src, into, say: tmp_path / Path(src).name)
    (tmp_path / "u" / "llama").mkdir(parents=True)
    (tmp_path / "u" / "llama" / "VERSION").write_text("cuda_archs 70;80;86;89;90;120\n")
    monkeypatch.setattr(rg, "unpack_server", lambda t, into: {
        "root": tmp_path / "u", "bin": tmp_path / "llama-server", "libs": []})
    monkeypatch.setattr(rg, "sha256_cached", lambda *a, **k: "ab" * 32)
    monkeypatch.setattr(rg, "version_of", lambda exe, env: {"build": 7000, "commit": "abc1234"})
    monkeypatch.setattr(sp.Test, "go", lambda self, tick=0.05: {
        "gpqa": {"tokens_s": 600.0, "answers_h": 9.0, "tokens": 1, "answers": 1},
        "mmlu": {"tokens_s": 1.0, "answers_h": 1.0, "prompt_s": 1.0, "prompt_read_s": 1.0},
        "waited": 0, "cut": 0, "refused": 0, "errors": 0, "mtp_acceptance": None,
        "sanity": {sp.GPQA: {"right": 1, "of": 1}, sp.MMLU: {"right": 1, "of": 1}}})
    monkeypatch.setattr(sp.Peak, "start", lambda self: None)
    assert sp.main(["v100x4", "--price", "0.65", "--gguf", "hf://u/r/q4.gguf", "--server",
                    "hf://u/r/s.tar.gz", "--files", str(tmp_path / "f"), "--out",
                    str(tmp_path / "o"), "--only", "1"]) == 0
    assert len(started) == 4
    assert [p.env["CUDA_VISIBLE_DEVICES"] for p in started] == ["0", "1", "2", "3"]
    assert [p.argv[p.argv.index("--port") + 1] for p in started] == ["8090", "8091", "8092",
                                                                     "8093"]
    out = capsys.readouterr().out
    assert "4× V100 32 GB (4× Tesla V100-SXM2-32GB) · $0.65/h" in out     # one combined line
    assert "$0.30 per million output tokens" in out
    got = json.loads((tmp_path / "o" / "speed.json").read_text())
    assert got[0]["servers"] == 4


def test_vllm_as_each_setting_says(tmp_path):
    argv = sp.vllm_argv(tmp_path / "venv", tmp_path / "w", 8100,
                        {"seqs": 64, "tp": 2}, "fp8")
    assert argv[:2] == [str(tmp_path / "venv" / "bin" / "vllm"), "serve"]
    for k, v in (("--max-model-len", "83968"), ("--max-num-seqs", "64"),
                 ("--tensor-parallel-size", "2"), ("--kv-cache-dtype", "fp8"),
                 ("--reasoning-parser", "qwen3"), ("--host", "127.0.0.1")):
        assert argv[argv.index(k) + 1] == v, k
    log = ("INFO GPU KV cache size: 412,416 tokens\nINFO Maximum concurrency for 83,968 tokens "
           "per request: 4.91x\n")
    assert sp.vllm_fits(log) == {"kv_tokens": 412_416, "full_length_at_once": 4.91}


def test_vllm_is_pinned_fourteen_days_old_with_hashes():
    assert sp.VLLM_VERSION == "0.30.0"
    text = sp.VLLM_LOCK.read_text()
    pins = dict(re.findall(r"(?m)^([a-z0-9][a-z0-9._-]*)==(\S+) \\$", text))
    assert pins["vllm"] == "0.30.0" and pins["torch"] == "2.13.0" and len(pins) == 196
    blocks = re.split(r"\n(?=[a-z0-9])", text.split("\n\n", 1)[1].strip())
    assert len(blocks) == len(pins)
    assert all(re.search(r"--hash=sha256:[0-9a-f]{64}", b) for b in blocks)
    # 0.30.0 was published on 22 Sep 2026: 17 days before the test on 9 Oct
    assert (datetime.date(2026, 10, 9) - datetime.date(2026, 9, 22)).days >= 14
    assert "--exclude-newer 2026-09-24" in text
    assert re.fullmatch(r"[0-9a-f]{40}", sp.BF16["revision"])
    assert sp.VLLM_LOCK.parent == ROOT / "scripts"              # in the runner image


def test_speed_results_never_reach_the_board():
    src = (ROOT / "scripts" / "speed_test.py").read_text()
    for never in ("from service", "import service", "import_remote", "make_bundle", "/api/",
                  "import_frontier"):
        assert never not in src, never


# ---------------------------------------------------------------------------
# the box lines, as the docs give them
# ---------------------------------------------------------------------------

BOXES = {"5090": [("NVIDIA GeForce RTX 5090", 32_607)],
         "v100x4": [("Tesla V100-SXM2-32GB", 32_768)] * 4,
         "h100": [("NVIDIA H100 80GB HBM3", 81_559)],
         "h100x2": [("NVIDIA H100 80GB HBM3", 81_559)] * 2}


def test_each_box_line_in_the_docs_runs(monkeypatch, capsys):
    s = DOC[DOC.index("# Speed tests"):]
    lines = [x.strip() for x in s.splitlines() if x.strip().startswith("python scripts/speed_test.py")]
    boxes = [x.split()[2] for x in lines]
    assert boxes == ["5090", "v100x4", "h100", "h100x2"]
    for line, box in zip(lines, boxes):
        monkeypatch.setattr(sp, "run", smi(BOXES[box], cuda="13.0"))
        argv = line.split()[2:] + ["--dry-run"]
        assert sp.main(argv) == 0, line
        out = capsys.readouterr().out
        assert out.startswith("this box: ") and f"{box}: " in out
