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
# 28 (sent mid-round): the parity check's sizes
# ---------------------------------------------------------------------------

def test_28_parity_compares_the_files_by_sha256_and_sizes_only_like_with_like(tmp_path, capsys):
    """every real pair was refused: "the server's file is 22,843,343,360 bytes, the
    box's 22,854,339,808" — llama-server's count of the weights against the
    file on disk"""
    import frontier_parity as fp
    sha = "ab" * 32
    launch = {"env": {}, "spec": [], "drafts": False}
    rows = [{"id": str(k), "key": "A", "answer": "the answer is (A)"} for k in range(50)]

    def files(server_file: dict, box_file: dict, tag: str) -> tuple[str, str]:
        s = tmp_path / f"server-{tag}.jsonl"
        b = tmp_path / f"box-{tag}.jsonl"
        for path, head in ((s, {"side": "server", "as": SERVED, "file": server_file,
                                "launch": launch}),
                           (b, {"side": "box", "as": SERVED, "file": box_file,
                                "server": {"flags": [], "env": {}}})):
            path.write_text("".join(json.dumps(x) + "\n" for x in [{"parity_of": head},
                                                                    *rows]))
        return str(s), str(b)
    name = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"
    # as this round writes them: the server's weights, the box's file and weights
    s, b = files({"name": name, "weights": 22_843_343_360, "sha256": ""},
                 {"name": name, "size": 22_854_339_808, "sha256": sha,
                  "weights": 22_843_343_360}, "new")
    assert fp.main(["compare", s, b, "--file-sha256", sha]) == 0
    assert capsys.readouterr().out.startswith("The same: ")
    # as the pilot wrote them (c43cdbe): one "size" each, of two kinds — not compared
    s, b = files({"name": name, "size": 22_843_343_360, "sha256": ""},
                 {"name": name, "size": 22_854_339_808, "sha256": sha}, "old")
    assert fp.main(["compare", s, b, "--file-sha256", sha]) == 0
    # the files whole, by sha256: another file is refused
    assert fp.main(["compare", s, b, "--file-sha256", "cd" * 32]) == 1
    assert "the box's file has sha256" in capsys.readouterr().out
    # weights on both sides that differ: refused
    s, b = files({"name": name, "weights": 22_843_343_360, "sha256": ""},
                 {"name": name, "size": 22_854_339_808, "sha256": sha,
                  "weights": 21_000_000_000}, "other")
    assert fp.main(["compare", s, b, "--file-sha256", sha]) == 1
    assert "llama-server counts the server's weights as 22,843,343,360 bytes" in \
        capsys.readouterr().out


# ---------------------------------------------------------------------------
# 29 (sent mid-round): the parity check decided on accuracy
# ---------------------------------------------------------------------------

def test_29_parity_is_decided_on_accuracy_over_n_questions(svc, monkeypatch, tmp_path, capsys):  # noqa: F811
    """letters agreed on 42 to 46 of 49 between the server and the box, and on
    43 of 50 between two runs on one box: letter agreement over 50 measured
    run-to-run noise, not the setup"""
    import frontier_parity as fp

    def rows(right: set[int], n: int = 500, wrong: str = "B", again: bool = False,
             say: str = "the answer is ({})") -> list[dict]:
        out = []
        for k in range(n):
            r = {"id": str(k), "key": "A", "answer": say.format("A" if k in right else wrong)}
            if again:                           # its second run: two letters of ten part
                r["answer2"] = r["answer"] if k % 10 > 1 else say.format("D")
            out.append(r)
        return out
    server = rows(set(range(210)))
    # 500 questions, 15 more right on the box and 15 fewer, the wrong ones
    # another letter, every reply worded otherwise: the same — 90% interval
    # about ±1.8 points
    boxed = rows(set(range(15, 225)), wrong="C", again=True, say="The answer is ({}).")
    got = fb.parity_compare(server, boxed)
    assert got["ok"] and (got["right_server"], got["right_box"]) == (210, 210)
    assert -0.02 < got["lo"] < 0 < got["hi"] < 0.02
    assert got["words"].startswith("The same: the box answers 42.0% right and the server 42.0% "
                                   "on the same 500 questions — a difference of +0.0 points, 90% "
                                   "interval -1.8 to +1.8, inside ±5 points")
    # the letters, information only: 195 the same (the wrong ones part), and
    # the box against itself
    assert got["letters"] == 195 and got["self"] == (500, 400, 400)
    assert ("For information: the same letter on 195 of the 500 read on both sides, 0 "
            "identical replies; the box against a second run of itself: the same letter on "
            "400 of 500, 400 identical") in got["words"]
    # ten points lower on the box: not the same
    got = fb.parity_compare(server, rows(set(range(50, 210))))
    assert not got["ok"] and got["hi"] < -0.05 and got["words"].startswith("Not the same: ")
    # the pilot's 50: 41 right on the server and 44 on the box — at 50 the
    # interval can't fit inside ±5 points, which is why it is 500
    got = fb.parity_compare(rows(set(range(41)), 50), rows(set(range(2, 46)), 50))
    assert not got["ok"] and got["lo"] < -0.02 and got["hi"] > 0.14

    # --n on both sides, and compare refuses two that differ
    monkeypatch.setattr(fb, "_fetch", lambda task: {"items": [
        {"id": str(k), "question": f"Q{k}?", "options": ["w", "x", "y"], "answer": "A",
         "category": "law"} for k in range(60)], "extra": {"shots": {}}})
    fake = FakeServer()
    try:
        fake.reply = lambda body: "the answer is (A)"
        rec = served.register({"name": "parity", "base_url": fake.base, "how": "x",
                               "thinking": "off"}, ME)
        # the time it takes, said before it asks: this server's pace thinking off
        db.served_put({**served.get(rec["id"]),
                       "speed_by": {"off": {"secs_each": 3.6, "n": 120}}})
        out = tmp_path / "server.jsonl"
        assert fp.main(["ask", "--as", rec["id"], "--out", str(out), "--n", "20"]) == 0
        said = capsys.readouterr().out
        assert said.startswith("20 questions, thinking off: about 1 min at 3.6 s an answer "
                               "(this server's pace with thinking off)")
        assert re.search(r"\n20 answers in \d+ min \([\d.]+ s an answer\)", said), said
        sh, srows = fp.read(out)
        assert sh["n"] == 20 and len(srows) == 20 and "answer2" not in srows[0]
        # 500 by default; a box asked 30 is refused beside it
        assert fb.PARITY["n"] == 500 and fb.PARITY["margin"] == 0.05
        box_file = tmp_path / "box.jsonl"
        sf.parity_ask(rec, box_file, identity={"side": "box", "as": rec["id"]}, n=30,
                      twice=True)
        bh, brows = fp.read(box_file)
        assert bh["n"] == 30 and bh["twice"] and all("answer2" in r for r in brows)
    finally:
        fake.close()
    assert fp.main(["compare", str(out), str(box_file)]) == 1
    assert "the server was asked 20 questions and the box 30: ask both with the same --n" \
        in capsys.readouterr().out
