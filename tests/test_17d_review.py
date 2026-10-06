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


# ---------------------------------------------------------------------------
# part 2: before the imports
# ---------------------------------------------------------------------------

def test_11_bundles_whose_shards_were_set_aside_can_be_imported_again(box):  # noqa: F811
    """they answered "imported already" for good"""
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2", "--flags", "-ctk q8_0") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    code, said = imported(bundle_of(box, "s2", (2, 2)), aside=True)
    assert code == 0 and any(f"came from {bundle_of(box, 's1', (1, 2)).name} — import it again"
                             in x for x in said), said
    # shard 2 run again the first box's way; then the first box's bundle again
    assert run_box(box, "s2b", "--shard", "2/2") == 0
    assert imported(bundle_of(box, "s2b", (2, 2)), aside=True)[0] == 0
    code, said = imported(bundle_of(box, "s1", (1, 2)))
    assert code == 0 and not any("imported already" in x for x in said), said
    assert any("every shard is in (2 of 2)" in x for x in said), said


def test_12_saving_a_served_models_page_keeps_its_file_hash(svc):  # noqa: F811
    fake = FakeServer()
    try:
        rec = a_served(fake)
        db.served_put({**rec, "file_sha256": {"sha256": "ab" * 32, "by": "masein"}})
        served.register({"name": "board box", "base_url": fake.base, "how": "llama-server, "
                         "edited", "thinking": "off"}, ME)
        assert served.get(rec["id"])["file_sha256"]["sha256"] == "ab" * 32
        # another file served now: the hash isn't carried to it
        fake.model_path = "/models/another.gguf"
        served.register({"name": "board box", "base_url": fake.base, "how": "x",
                         "thinking": "off"}, ME)
        assert "file_sha256" not in served.get(rec["id"])
    finally:
        fake.close()


def test_13_the_counts_that_lower_a_score_are_on_its_cell():
    import report_lm_eval as report
    assert report.frontier_how({"scored_by": "code", "unanswered": 3}) == (
        "by code · 3 the server never answered, counted wrong")
    g = {"version": "openai/o3-mini-2025-01-31", "prompt_words": "CAIS's judge prompt",
         "prompt_sha256": "ab" * 32}
    d = {"scored_by": "grader", "grader": g, "ungraded": 2, "final": False}
    assert report.frontier_how(d) == ("graded by openai/o3-mini-2025-01-31 with CAIS's judge "
                                      "prompt (abababab): not final · 2 its grader gave no grade, "
                                      "counted wrong")
    assert report.frontier_setting(d) is None                     # ranked with nothing


def test_15_a_bundle_json_of_the_wrong_types_is_refused_in_words(box):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])

    def odd(files):
        bundle = json.loads(files["bundle.json"])
        bundle["row"], bundle["model"] = ["a", "b"], 7
        files["bundle.json"] = json.dumps(bundle).encode()
    code, said = imported(rewrite(bundle_of(box, "run"), odd))
    assert code == 2, said
    assert any("bundle.json's row is list, not str" in x for x in said), said
    assert any("bundle.json's model is int, not str" in x for x in said), said


# ---------------------------------------------------------------------------
# part 3: before Start on grading (the stand-in grader of tests/test_17_grading.py)
# ---------------------------------------------------------------------------

from test_17_grading import ROW as GROW  # noqa: E402
from test_17_grading import drain, results  # noqa: E402
from test_17_grading import svc as gsvc  # noqa: E402,F401 — the grading fixture
from test_17b_grading import GEMINI, GPT, plain, stub  # noqa: E402


def grades_of(task: str) -> dict:
    return sf.read_grades(sf.task_dir(config.OUT_DIR / GROW, task))


def test_16_a_grader_that_cant_be_reached_uses_no_tries_and_another_asks_again(
        gsvc, monkeypatch):  # noqa: F811
    from service import frontier_grade as fgr
    down = {"on": True}
    stub(monkeypatch, lambda m, r: ("", "", "HTTP 503: the provider is down")
         if m == GPT and down["on"] else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()
    ref = grades_of("simpleqa_epoch")["refused"]
    assert ref and all(x["tries"] == 0 for x in ref.values())       # 0f7c943: 3, ungraded
    assert {r["slot"]: r["answers"] for r in gsvc.get("/api/frontier/grading").json()
            ["estimate"]["rows"]} == {"simpleqa": 5}
    # a reply that came and isn't a grade counts; another grader starts again
    down["on"] = False
    stub(monkeypatch, lambda m, r: ("I would rather not say", "stop", "")
         if m == GPT and r["custom_id"].endswith(":2#0") else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()
    assert grades_of("simpleqa_epoch")["refused"]["2#0"]["tries"] == 3
    fgr.save("simpleqa", GEMINI, "masein")
    assert grades_of("simpleqa_epoch")["refused"]["2#0"]["tries"] == 0
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    drain()
    assert [c for m, c in asked if m == GEMINI and "algebra/" not in c] == ["frgr:2#0"]


def test_17_a_grader_answering_in_prose_leaves_no_score(gsvc, monkeypatch):  # noqa: F811
    from service import frontier_grade as fgr
    stub(monkeypatch, lambda m, r: ("Let me think about whether this is right…", "stop", "")
         if m == GPT else plain(m, r))
    for _ in range(3):
        fgr.start("masein")
        drain()
    assert results("simpleqa_epoch") is None                  # 0f7c943: 0.0%, "graded"
    sc = sf.score_task(config.OUT_DIR / GROW, "simpleqa_epoch",
                       served.get("served/lda-box"))
    assert sc["no_score"].startswith("SimpleQA Verified: its grader gave no grade on 5 of the 5")
    card = gsvc.get("/api/frontier/grading").json()["refused"]
    assert [(x["slot"], x["ungraded"], x["form"]) for x in card] == [("simpleqa", 5, True)]


def test_18_hles_verdict_is_the_field_on_its_own_line():
    import frontier_graders as fg
    got = fg.read("hle", "reasoning: is 4 correct: no, it's 5\ncorrect: yes\nconfidence: 90%",
                  {})
    assert got["ok"] is True and got["confidence"] == 90
    assert fg.read("hle", "correct: yes, with caveats", {})["ok"] is None
    assert fg.read("hle", "correct: yes\r", {})["ok"] is True
    assert fg.read("hle", '{"correct": "no", "confidence": 40}', {})["ok"] is False


def test_19_a_batch_that_cant_land_stays_and_is_never_paid_twice(gsvc, monkeypatch):  # noqa: F811
    from service import frontier_grade as fgr
    from service import llm_poller
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    real = fb.load
    broken = {"on": True}
    monkeypatch.setattr(fb, "load", lambda task, root: (_ for _ in ()).throw(
        OSError("the disk is full")) if broken["on"] else real(task, root))
    end = time.time() + 10
    while time.time() < end and any((llm_tally(p) or {}).get("answered", 0) < (llm_tally(p) or
                                    {}).get("sent", 1) for p in fgr.pending()):
        time.sleep(0.05)
    for _ in range(31):                                       # 0f7c943: failed at 30
        llm_poller.tick()
    assert len(fgr.pending()) == 2
    assert any("tried 31 times" in w["why"] for w in fgr.waits())
    broken["on"] = False
    drain()
    fgr.start("masein")
    drain()
    assert len(asked) == 7 and results("simpleqa_epoch") is not None


def llm_tally(p: dict) -> dict | None:
    from service import llm
    return llm.tally(p["batch_id"])


def test_20_a_refused_carry_on_leaves_the_stop(gsvc, monkeypatch):  # noqa: F811
    """after Stop the grader moved, Carry on was refused — and the poller then
    sent the other grader's held requests"""
    import threading
    from service import ai_models
    from service import frontier_grade as fgr
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    gate = threading.Event()
    seen: list = []

    def answer(model, row):
        seen.append(model)
        if model == GPT and seen.count(GPT) == 1:
            gate.wait(10)
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    end = time.time() + 10
    while time.time() < end and GPT not in seen:
        time.sleep(0.05)
    fgr.stop("masein")
    gate.set()
    monkeypatch.setattr(ai_models, "drifted",
                        lambda pin: "moved" if (pin or {}).get("id") == GPT else "")
    with pytest.raises(ValueError, match="moved"):
        fgr.start("masein")
    assert fgr.stopped()                                      # 0f7c943: lifted


def test_21_requests_in_flight_land_before_start_sends_them_again(gsvc, monkeypatch):  # noqa: F811
    import threading
    from service import frontier_grade as fgr
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 2)
    gate = threading.Event()
    seen: list = []

    def answer(model, row):
        seen.append((model, row["custom_id"]))
        if model == GPT:
            gate.wait(10)                    # two in flight, held
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    end = time.time() + 10
    while time.time() < end and sum(1 for m, _ in seen if m == GPT) < 2:
        time.sleep(0.05)
    fgr.save("simpleqa", GEMINI, "masein")
    threading.Timer(1.0, gate.set).start()                   # they land a second later
    fgr.start("masein")
    drain()
    keys = [c for _, c in seen if "algebra/" not in c]
    assert len(keys) == len(set(keys)) == 5                   # 0f7c943: the two asked twice


def test_22_the_estimate_says_the_most_three_tries_could_cost(gsvc):  # noqa: F811
    est = gsvc.get("/api/frontier/grading").json()["estimate"]
    g = est["graders"]["simpleqa"]
    assert g["usd_max"] == pytest.approx(
        3 * (g["tokens_in"] * 2.0 + g["tokens_out_max"] * 8.0) / 1e6, abs=1e-4)
