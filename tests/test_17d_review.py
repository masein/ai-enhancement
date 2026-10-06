"""17d, the third review round (docs/prompts/phase-17d-third-review-fixes.md),
re-run on 0f7c943. Part 1, before the full run. Each test fails on 0f7c943.
The box is tests/fixtures/fake_llama_server.py; the server's served model
tests/fake_openai.py. Questions invented; nothing is fetched and no model
runs."""

from __future__ import annotations

import json
import re
import struct
import time
from pathlib import Path

import pytest

import frontier as fb
import gguf_header
import remote_bundle as rb
import remote_gguf as rg
from fake_openai import FakeServer
from service import config, db, served
from service import frontier as sf
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (GPU, N, RUNS, SERVED, TASK, box,  # noqa: F401
                              bundle_of, invented, register, run_box)
from test_17b_review import imported, rewrite

REPO = Path(__file__).resolve().parents[1]


def gguf(path: Path, arch: str = "qwen3moe", **kv) -> Path:
    """a GGUF header and nothing else: its metadata, no tensors"""
    meta = {"general.architecture": arch, f"{arch}.block_count": 48,
            f"{arch}.attention.head_count": 32, f"{arch}.attention.head_count_kv": 4,
            f"{arch}.attention.key_length": 128, f"{arch}.attention.value_length": 128,
            f"{arch}.embedding_length": 2048, **{f"{arch}.{k}": v for k, v in kv.items()}}

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta))
    for k, v in meta.items():
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str)
                       else struct.pack("<II", 4, v))
    path.write_bytes(out + b"\0" * 4096)
    return path


def a_served(fake: FakeServer, **more) -> dict:
    fake.ctx = 40960
    return served.register({"name": "board box", "base_url": fake.base, "how": "x",
                            "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off", **more}, ME)


# ---------------------------------------------------------------------------
# 1: the KV cache from the GGUF's header, before the download
# ---------------------------------------------------------------------------

def test_1_the_kv_cache_is_worked_out_from_the_header_and_the_card(box, monkeypatch, tmp_path):  # noqa: F811
    path = gguf(tmp_path / "m.gguf")
    shape = gguf_header.shape(open(path, "rb"))
    assert (shape["layers"], shape["kv_heads"][:2], shape["key_length"]) == (48, [4, 4], 128)
    # 48 layers × 4 heads × 128 × 2 (K and V) × q8_0's 34/32 bytes: 52,224 a token
    assert rg.kv_per_token(shape, "q8_0", "q8_0") == 52224
    hybrid = gguf_header.shape(open(gguf(tmp_path / "h.gguf", full_attention_interval=4), "rb"))
    assert sum(1 for h in hybrid["kv_heads"] if h) == 12
    # an 8 GB card: 8 slots of GPQA's 34,816 — under the stated limit, more than it holds
    monkeypatch.setattr(rb, "gpu_info", lambda: {**GPU, "memory_mib": 8192})
    with pytest.raises(SystemExit, match=r"8 slots of 34,816 tokens \(GPQA Diamond's, thinking "
                                         r"on\) need about 13.5 GB of KV cache \(q8_0/q8_0, 51.0 "
                                         r"KB a token.*at most 3 slots fit — give --slots 3"):
        run_box(box, "run", "--gguf", str(path), "--slots", "8",
                "--flags", "-ctk q8_0 -ctv q8_0 --flash-attn on")
    assert not (box["root"] / "files").exists() or not any((box["root"] / "files").iterdir())
    # a header that can't be read before the fetch: the stated limit, as before
    assert rg.header_of("hf://nobody/nothing/x.gguf") == (None, None)


# ---------------------------------------------------------------------------
# 2–5: thinking, failures near the end, a timeout
# ---------------------------------------------------------------------------

def test_2_an_off_run_whose_first_answers_think_stops_in_words(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "<think>\nweighing it\n</think>\n\nANSWER: A"
        rec = a_served(fake)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        with pytest.raises(served.ServerStopped, match="thinking was off, and 2\\d of the first "
                                                       "20 answers hold thinking"):
            sf.ask_task(rec, TASK, row, False)
        n = len(sf.read_answers(sf.task_dir(row, TASK) / sf.ANSWERS))
        assert n < N * RUNS
        # the next run says so before it asks anything
        asked = len(fake.requests)
        with pytest.raises(served.ServerStopped):
            sf.ask_task(rec, TASK, row, False)
        assert len(fake.requests) == asked
    finally:
        fake.close()


def test_3_the_one_percent_rule_is_the_whole_benchmarks(box):  # noqa: F811
    """an off shard with a thinking answer was refused, though the whole might pass"""
    assert run_box(box, "s1", "--shard", "1/2", thinking="off") == 0
    register(box["sha"])
    path = bundle_of(box, "s1", (1, 2), thinking=False)
    row = path.name.split("-thinking")[0].replace("frontier-", "")
    name = f"results/{row}/{TASK}_0shot/frontier/answers.jsonl"

    def one_thinks(files):
        lines = files[name].decode().splitlines()
        r = json.loads(lines[0])
        r["answer"] = "<think>\nhmm\n</think>\n\n" + r["answer"]
        files[name] = ("\n".join([json.dumps(r), *lines[1:]]) + "\n").encode()
    code, said = imported(rewrite(path, one_thinks))
    assert code == 0, said
    assert any(f"1 of this shard's {N * RUNS // 2} answers hold thinking" in x
               for x in said), said


def test_4_a_server_that_goes_down_near_the_end_writes_no_answer_off(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"
        rec = a_served(fake)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        down = {"after": N * RUNS - 2}

        def broken(body):
            return (500, "the server crashed") if fake.answered >= down["after"] else None
        fake.chat_error = fake.raw_error = broken
        with pytest.raises(served.ServerStopped, match="the server stopped answering"):
            sf.ask_task(rec, TASK, row, False)
        got = sf.read_answers(sf.task_dir(row, TASK) / sf.ANSWERS)
        assert not any(r.get("unanswered") for r in got.values())
        # back up: the next run asks what is left, and the task is whole
        fake.chat_error = fake.raw_error = None
        assert sf.ask_task(rec, TASK, row, False) == (N * RUNS, N * RUNS)
    finally:
        fake.close()


def test_5_a_timeout_from_a_server_that_is_up_is_that_questions_own(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    monkeypatch.setattr(served, "timeout_for", lambda s: 0.5)
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"

        def slow(body):
            if re.search(r"Invented question 3\b", json.dumps(body)) and \
                    (body.get("max_tokens") or 0) > 16:
                time.sleep(1.5)
        fake.on_request = slow
        rec = a_served(fake)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        assert sf.ask_task(rec, TASK, row, False) == (N * RUNS, N * RUNS)
        got = sf.read_answers(sf.task_dir(row, TASK) / sf.ANSWERS)
        never = [r for r in got.values() if r.get("unanswered")]
        assert [r["id"] for r in never] == ["rec003"] * RUNS
        assert "no answer within" in never[0]["unanswered"]
    finally:
        fake.close()


# ---------------------------------------------------------------------------
# 6–10
# ---------------------------------------------------------------------------

def test_6_a_box_with_no_only_is_refused_thinking_off_too(tmp_path, capsys):
    with pytest.raises(SystemExit):
        rg.main(["--as", SERVED, "--gguf", "hf://me/private/m.gguf", "--server",
                 "hf://me/private/s.tar.gz", "--thinking", "off", "--out", str(tmp_path / "o")])
    assert "--only: name this box's benchmarks" in capsys.readouterr().err


def test_7_9_the_boards_resume_knows_what_changes_answers(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"
        rec = a_served(fake)
        db.served_put({**rec, "speculative": False, "flags": "-ctk q8_0"})
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        assert sf.ask_task(served.get(rec["id"]), TASK, row, False)[0] == N * RUNS
        n = len(fake.requests)
        # 7: its /slots didn't answer this time: unknown, not another setup
        db.served_put({**served.get(rec["id"]), "speculative": None})
        sf.ask_task(served.get(rec["id"]), TASK, row, False)
        assert len(fake.requests) == n
        # 9: flags that change answers do count
        for flags in ("-ctk f16", "-ctk f16 --chat-template-file other.jinja",
                      "-ctk f16 --chat-template-file other.jinja --reasoning-budget 0"):
            db.served_put({**served.get(rec["id"]), "flags": flags})
            sf.ask_task(served.get(rec["id"]), TASK, row, False)
            assert len(fake.requests) == n + N * RUNS, flags
            n = len(fake.requests)
    finally:
        fake.close()


def test_8_visible_is_linear_in_leading_whitespace():
    t0 = time.time()
    for _ in range(3):
        assert fb.visible("\n" * 20000 + "x") == "x"
    assert time.time() - t0 < 0.5
    assert fb.visible("weighing\n</think>\nB") == "B"


def test_10_the_tarball_script_stops_in_words_where_it_stopped_silently():
    script = (REPO / "scripts" / "build_llama_tarball.sh").read_text()
    assert "if ! grep -E 'LLAMA_(BUILD_NUMBER|COMMIT) =' \"$info\" > /out/build-info; then" \
        in script
    assert "-exec cp -P -t /out/llama/lib/ {} +" in script and "-exec cp -P {}" not in script
    assert "|| { echo \"couldn't pack $so: nothing was packed\" >&2; exit 1; }" in script
