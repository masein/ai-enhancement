"""17b, parts 1 to 5 of the review of 1388058..62ec4d9 — what blocks the
pilot. Each test fails on 62ec4d9:
- part 1: the box runs the setup it says it runs (the port, the child, the
  file, the launch against the registered one, the setup across a resume and
  across shards, the board's own resume, thinking on or off, the parity check);
- part 2: a failed answer is not an answer (never kept, asked again, the
  limit; the fallback's sampling; the timeout from the budget);
- part 3: the import treats a bundle as untrusted (--register, each line, the
  row untouched by a failed import, the board's own model record, the size
  caps);
- part 4: the commands masein pastes (the token never on a command line, the
  tarball script's own errors);
- part 5: scoring by code (OTIS's integers, each benchmark's count on load).
The box is tests/fixtures/fake_llama_server.py; the board's served model is
tests/fake_openai.py. Questions invented; nothing is fetched and no model runs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

import frontier as fb
import frontier_parity as fp
import import_remote as ir
import remote_bundle as rb
import remote_gguf as rg
from fake_openai import FakeServer
from service import config, db, served
from service import frontier as sf
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (FIX, GGUF_NAME, N, ROW, RUNS, SERVED, TASK, box,  # noqa: F401
                              bundle_of, free_port, invented, record, register, run_box)

REPO = Path(__file__).resolve().parents[1]


def imported(path: Path, **kw) -> tuple[int, list[str]]:
    said: list[str] = []
    return ir.import_bundle(path, "masein", said.append, **kw), said


def rewrite(path: Path, fix) -> Path:
    """the bundle with its files changed by `fix(files)`, made whole again —
    a bundle someone altered, its checksums agreeing"""
    b = rb.read(path)
    files = dict(b["files"])
    fix(files)
    bundle = json.loads(files["bundle.json"])
    bundle["files"] = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()
                       if n != "bundle.json"}
    files["bundle.json"] = json.dumps(bundle).encode()
    return rb.write(path.with_name("altered-" + path.name), files)


def answers_name(row: str = ROW, task: str = TASK) -> str:
    return f"results/{row}/{task}_0shot/frontier/answers.jsonl"


def results_of(row: str = ROW, task: str = TASK) -> dict | None:
    fs = list(sf.task_dir(config.OUT_DIR / row, task).glob("results_*.json"))
    return json.loads(fs[0].read_text()) if fs else None


# ---------------------------------------------------------------------------
# part 1: the box runs the setup it says it runs
# ---------------------------------------------------------------------------

def test_1_a_server_already_on_the_port_is_refused(box, tmp_path):  # noqa: F811
    """an earlier llama-server on the port answered for the new one: its
    file's answers under the new file's sha256, and exit 0"""
    port = free_port()
    old = subprocess.Popen([sys.executable, str(FIX / "fake_llama_server.py"), "-m",
                            "/x/earlier.gguf", "--port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            with socket.socket() as sk:
                if sk.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        with pytest.raises(SystemExit, match="something already answers on 127.0.0.1"):
            run_box(box, "run", "--port", str(port))
        assert not bundle_of(box, "run").exists()
        assert not (box["log"].exists() and box["log"].read_text())   # nothing was asked
    finally:
        old.terminate()
        old.wait(10)


def test_1_a_server_serving_another_file_is_refused(box, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_MODEL_PATH", "/models/another.gguf")
    with pytest.raises(SystemExit, match="serves another.gguf, not " + GGUF_NAME.replace(".", r"\.")):
        run_box(box, "run")
    assert not (box["log"].exists() and box["log"].read_text())


def test_1_the_launch_recorded_is_the_one_launched(box, monkeypatch, tmp_path):  # noqa: F811
    """the bundle's argv and environment are what llama-server was started
    with, as the server saw them"""
    seen = tmp_path / "argv.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(seen))
    assert run_box(box, "run") == 0
    got = json.loads(seen.read_text().splitlines()[-1])
    argv = got["argv"]
    rec = rb.read(bundle_of(box, "run"))["setup"]["server"]
    assert argv[argv.index("-c") + 1] == str(3 * (81920 + 2048))          # 17f: the new limit
    assert argv[argv.index("-np") + 1] == "3" and "--jinja" in argv
    assert argv[argv.index("--flash-attn") + 1] == "on"
    # what is recorded is what was launched: the same flags, the same environment
    assert rec["argv"][rec["argv"].index("-m") + 2:] == argv[argv.index("-m") + 2:]
    assert got["env"].get("LLAMA_MOE_ROUTE_MODE") == "lookahead" == rec["env"][
        "LLAMA_MOE_ROUTE_MODE"]


def test_2_the_import_refuses_a_launch_that_isnt_the_registered_one(box):  # noqa: F811
    """one GGUF, four setups on the board: the sha256 passes for all four. The
    phone build ("routing local (no lookahead)") must not get lookahead answers"""
    assert run_box(box, "run") == 0                      # with LLAMA_MOE_ROUTE_MODE=lookahead
    rec = record(box["sha"])
    rec.update(env="", flags="", how="routing local (no lookahead)")
    db.served_put(rec)
    code, said = imported(bundle_of(box, "run"))
    assert code == ir.REFUSED
    assert any("routing: the box ran with LLAMA_MOE_ROUTE_MODE=lookahead; served/lda-box is "
               "registered with none" in x for x in said), said
    assert not (config.OUT_DIR / ROW / f"{TASK}_0shot").exists()
    # an MTP setup's row from a box without --spec-type: refused too
    rec.update(env="LLAMA_MOE_ROUTE_MODE=lookahead", flags="--spec-type mtp --draft-max 3")
    db.served_put(rec)
    code, said = imported(bundle_of(box, "run"))
    assert code == ir.REFUSED and any("speculative decoding: the box ran with none" in x
                                      for x in said), said
    # memory, context, slots, the cache type, flash attention may differ: the
    # registered --cpu-moe against the box's --flash-attn on, accepted
    register(box["sha"])
    assert imported(bundle_of(box, "run"))[0] == 0


def test_3_a_resume_with_another_setup_is_refused(box, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_DIE_AFTER", "5")
    assert run_box(box, "run") == 1
    monkeypatch.delenv("FAKE_LLAMA_DIE_AFTER")
    for more, what in ((["--flags", "-ctk q8_0"], "flags"),
                       (["--env", "LLAMA_MOE_ROUTE_MODE=local"], "env")):
        with pytest.raises(SystemExit, match=f"another setup — its {what}"):
            run_box(box, "run", *more)
    # another server build: another binary in another tarball
    tb2 = box["root"] / "llama-server-other.tar.gz"
    with tarfile.open(tb2, "w:gz") as tar:
        data = (FIX / "fake_llama_server.py").read_bytes() + b"\n# another build\n"
        ti = tarfile.TarInfo("llama/bin/llama-server")
        ti.size, ti.mode = len(data), 0o755
        tar.addfile(ti, io.BytesIO(data))
    with pytest.raises(SystemExit, match="binary_sha256"):
        run_box({**box, "tarball": tb2}, "run")
    # the same setup carries on; the slots may change, and each session's are kept
    assert run_box(box, "run", "--slots", "2") == 0
    srv = rb.read(bundle_of(box, "run"))["setup"]["server"]
    assert [x["slots"] for x in srv["sessions_slots"]] == [3, 2]


def test_3_shards_made_with_other_setups_are_never_merged(box):  # noqa: F811
    assert run_box(box, "s1", "--shard", "1/2") == 0
    assert run_box(box, "s2", "--shard", "2/2", "--flags", "-ctk q8_0") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "s1", (1, 2)))[0] == 0
    code, said = imported(bundle_of(box, "s2", (2, 2)))
    assert code == ir.REFUSED
    assert any("shard 1 here was made with another setup (flags" in x for x in said), said
    assert results_of() is None


def test_4_the_boards_own_resume_sets_aside_answers_made_with_another_setup(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.reply = lambda body: "ANSWER: A"
        fake.ctx = 40960
        rec = served.register({"name": "board box", "base_url": fake.base, "how": "x",
                               "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off"}, ME)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        logged: list[str] = []
        assert sf.ask_task(rec, TASK, row, False, log=logged.append) == (N * RUNS, N * RUNS)
        n = len(fake.requests)
        # the same setup: nothing asked again
        assert sf.ask_task(rec, TASK, row, False, log=logged.append) == (N * RUNS, N * RUNS)
        assert len(fake.requests) == n
        # its server now serves another file: every answer asked again, the
        # earlier ones set aside
        rec["pin"] = {**rec["pin"], "file": "another.gguf"}
        db.served_put(rec)
        assert sf.ask_task(rec, TASK, row, False, log=logged.append) == (N * RUNS, N * RUNS)
        assert len(fake.requests) == n + N * RUNS
        assert any("made with another setup (server)" in x for x in logged), logged
        assert list((config.OUT_DIR.with_name("earlier") / row.name).glob(
            f"{TASK}_0shot-frontier-another-setup-*"))
    finally:
        fake.close()


def test_5_a_thinking_row_with_no_thinking_is_refused(box, monkeypatch):  # noqa: F811
    """a box whose server never thought (a --reasoning-budget 0) filled a
    thinking row"""
    monkeypatch.setenv("FAKE_LLAMA_THINK", "never")
    # 17d: the box stops after its first 20 answers, saying why — not three
    # hours later at the import (tests/test_17d_review.py has the import's)
    assert run_box(box, "run") == 1
    assert not bundle_of(box, "run").exists()
    own = (box["root"] / "run" / rg.OWN_LOG).read_text()
    assert "thinking was asked for, and none of the first 20 answers holds any" in own


def test_5_an_off_row_with_thinking_is_not_scored(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    rec = record(None)
    db.served_put(rec)
    row = config.OUT_DIR / "served__lda-box"
    d = sf.task_dir(row, TASK)
    d.mkdir(parents=True)
    (d / sf.SETUP).write_text(json.dumps({"thinking": "off"}))
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": it["id"], "epoch": e, "answer": "<think>\nweighing it\n</think>\n\nANSWER: A",
         "finish": "stop"}) + "\n" for it in invented() for e in range(RUNS)))
    sc = sf.score_task(row, TASK, rec)
    assert "thinking was off, and 48 of its 48 answers hold thinking" in sc["refused"]
    assert not list(d.glob("results_*.json"))
    # an empty block, as a template told not to think writes, is no thinking
    assert not sf.thought("<think>\n\n</think>\n\nANSWER: A") and sf.thought("<think>x")


def test_6_the_parity_check_compares_the_server_and_the_box(box, monkeypatch, svc):  # noqa: F811
    monkeypatch.setitem(fb.PARITY, "floor", 50)         # 17e: 500 on a real pair
    items = [{"id": str(k), "question": f"Q{k}?", "options": ["w", "x", "y"],
              "answer": "A", "category": "law"} for k in range(60)]
    monkeypatch.setattr(fb, "_fetch", lambda task: {"items": items, "extra": {"shots": {}}})
    # the box: --parity asks the 50, thinking off, and makes no bundle
    log = box["log"]
    assert run_box(box, "parity", "--parity", "--n", "50") == 0
    _, box_lines = fp.read(box["root"] / "parity" / "parity.jsonl")
    assert len(box_lines) == 50 and not list((box["root"] / "parity").glob("*.tar.gz"))
    asked = [json.loads(x) for x in log.read_text().splitlines()]
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False}
               and r["temperature"] == 0.0 and r["top_k"] == 1 for r in asked)
    # the server: the same 50, through its served model
    fake = FakeServer()
    try:
        fake.reply = lambda body: "the answer is (A)"
        rec = served.register({"name": "parity", "base_url": fake.base, "how": "x",
                               "thinking": "off"}, ME)
        out = box["root"] / "server.jsonl"
        assert fp.main(["ask", "--as", rec["id"], "--out", str(out), "--n", "50"]) == 0
        _, server_lines = fp.read(out)
    finally:
        fake.close()
    assert [r["id"] for r in server_lines] == [r["id"] for r in box_lines]
    got = fb.parity_compare(server_lines, box_lines)
    # 17d: decided on accuracy — the same questions right on both sides
    assert got["ok"] and got["letters"] == 50 and got["identical"] == 0
    assert got["words"].startswith("The same: the box answers 100.0% right (the mean of its "
                                   "two runs) and the server 100.0% on the same 50 questions")
    # ten the server got wrong: twenty points apart, not the same
    other = [{**r, "answer": "the answer is (B)"} if i < 10 else r
             for i, r in enumerate(server_lines)]
    assert fb.parity_compare(other, box_lines)["ok"] is False
    # 17c: compare refuses these two — the server's answered as another model
    # (tests/test_17c_review.py has the pair that compares)
    assert fp.main(["compare", str(out), str(box["root"] / "parity" / "parity.jsonl")]) == 1


# ---------------------------------------------------------------------------
# part 2: a failed answer is not an answer
# ---------------------------------------------------------------------------

def test_7_a_server_that_fails_every_question_leaves_nothing_answered(box, monkeypatch):  # noqa: F811
    """500 to everything was "48 of 48 answered", exit 0, scored 0.0%"""
    monkeypatch.setenv("FAKE_LLAMA_STATUS", "500")
    assert run_box(box, "run") == 1
    answers = box["root"] / "run" / "bench" / "results" / "full" / ROW / f"{TASK}_0shot" / \
        "frontier" / "answers.jsonl"
    assert not answers.exists() or not answers.read_text().strip()
    assert not bundle_of(box, "run").exists()
    own = (box["root"] / "run" / rg.OWN_LOG).read_text()
    assert "the server failed on 4 questions" in own
    # the same command once the server answers: every question asked
    monkeypatch.delenv("FAKE_LLAMA_STATUS")
    n = len(box["log"].read_text().splitlines())
    assert run_box(box, "run") == 0
    assert len(box["log"].read_text().splitlines()) - n == N * RUNS


def test_7_a_bundle_holding_failed_lines_is_refused(box):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])

    def fail_one(files):
        lines = files[answers_name()].decode().splitlines()
        r = json.loads(lines[0])
        r["error"] = {"chat": "HTTP 500: it failed", "fallback": "HTTP 404"}
        files[answers_name()] = ("\n".join([json.dumps(r), *lines[1:]]) + "\n").encode()
    code, said = imported(rewrite(bundle_of(box, "run"), fail_one))
    assert code == ir.REFUSED
    assert any("line 1 is a question the server failed on, not an answer" in x for x in said)


def test_7_and_8_the_board_keeps_no_failed_answer_and_the_fallback_samples(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented())
    fake = FakeServer()
    try:
        fake.ctx = 40960
        fake.chat_error = lambda body: (500, "it failed")
        fake.raw_error = lambda body: (500, "and this")
        rec = served.register({"name": "failing", "base_url": fake.base, "how": "x",
                               "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off"}, ME)
        row = config.OUT_DIR / rec["id"].replace("/", "__")
        with pytest.raises(served.ServerStopped, match="the server failed on 4 questions"):
            sf.ask_task(rec, TASK, row, True)
        assert not sf.read_answers(sf.task_dir(row, TASK) / sf.ANSWERS)
        # 8: the raw way round sends the card's sampling, the seed among it
        sent = fake.completions[0]
        assert (sent["temperature"], sent["top_p"], sent["top_k"], sent["presence_penalty"]) \
            == (1.0, 0.95, 20, 1.5) and isinstance(sent["seed"], int)
        # …and a model with no preset sends none, where it raised KeyError
        fake.raw_error = None
        fake.completion_reply = lambda body: {"content": "ANSWER: A", "tokens_predicted": 3,
                                              "stop_type": "eos"}
        bare = {**rec, "id": "served/bare", "based_on": "", "name": "bare",
                "pin": {**rec["pin"], "file": "plain.gguf", "model": "plain"}}
        a = served.answer_one(bare, "Q?", sf.settings(bare, TASK, False))
        assert str(a) == "ANSWER: A" and a.fallback
        assert "temperature" not in fake.completions[-1]
    finally:
        fake.close()


def test_9_the_timeout_follows_the_budget(monkeypatch):
    seen = {}

    def http(method, url, key="", body=None, timeout=None):
        seen["timeout"] = timeout
        return 200, json.dumps({"choices": [{"message": {"content": "x"},
                                             "finish_reason": "stop"}]}).encode()
    monkeypatch.setattr(served, "_http", http)
    rec = {"id": "served/x", "name": "x", "base_url": "http://127.0.0.1:9/v1", "key": "",
           "pin": {"model": "m"}}
    served.ask(rec, "Q", {"max_tokens": 32768})
    assert seen["timeout"] == pytest.approx(32768 / 10 + 120)        # 57 minutes, not 15
    served.ask(rec, "Q", {"max_tokens": 1024})
    assert seen["timeout"] == config.SERVED_TIMEOUT_S


# ---------------------------------------------------------------------------
# part 3: the import treats a bundle as untrusted
# ---------------------------------------------------------------------------

def test_10_register_never_takes_the_bundles_word(box):  # noqa: F811
    assert run_box(box, "run") == 0
    path = bundle_of(box, "run")
    # no --file-sha256: refused (the sha256 check compared the bundle with itself)
    code, said = imported(path, register="Mine")
    assert code == ir.REFUSED and any("--register needs --file-sha256" in x for x in said)
    # a row already here: refused
    (config.OUT_DIR / ROW).mkdir(parents=True)
    code, said = imported(path, register="Mine", file_sha=box["sha"])
    assert code == ir.REFUSED and any("has a row here already" in x for x in said), said
    # an id holding "__", naming another row: refused
    assert run_box(box, "evil", "--as", "served/lda-box__x") == 0
    evil = box["root"] / "evil" / rb.bundle_name("frontier", "served/lda-box__x", True,
                                                 parts=["gpqa"])
    code, said = imported(evil, register="Evil", file_sha=box["sha"])
    assert code == ir.REFUSED and any("can't be a new model's id" in x for x in said), said
    assert served.get("served/lda-box__x") is None


def test_11_a_line_that_isnt_an_answer_is_refused_and_the_row_stays(box):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "run"))[0] == 0
    before = results_of()

    def five(files):
        lines = files[answers_name()].decode().splitlines()
        r = json.loads(lines[3])
        r["answer"] = 5
        files[answers_name()] = ("\n".join([*lines[:3], json.dumps(r), *lines[4:]]) + "\n").encode()
    code, said = imported(rewrite(bundle_of(box, "run"), five))
    assert code == ir.REFUSED
    assert any("line 4 has an answer that isn't text" in x for x in said), said
    assert results_of() == before                         # the row kept its score


def test_11_the_rows_model_record_is_the_boards(box):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])

    def evil_meta(files):
        files[f"results/{ROW}/model_meta.json"] = json.dumps({"model": "someone else"}).encode()
    assert imported(rewrite(bundle_of(box, "run"), evil_meta))[0] == 0
    meta = json.loads((config.OUT_DIR / ROW / "model_meta.json").read_text())
    assert meta["model"] == SERVED + " · thinking" and meta["base_model"] == SERVED


def test_11_a_task_that_cant_be_scored_leaves_the_row_as_it_was(box, monkeypatch):  # noqa: F811
    assert run_box(box, "run") == 0
    register(box["sha"])
    assert imported(bundle_of(box, "run"))[0] == 0
    before = results_of()
    assert run_box(box, "again") == 0

    def boom(*a, **k):
        raise RuntimeError("the scorer broke")
    monkeypatch.setattr(sf, "score_task", boom)

    # 17f: other answers (the same ones change nothing, and need no scoring)
    def other(files):
        lines = files[answers_name()].decode().splitlines()
        r = json.loads(lines[0])
        r["answer"] += "\n(asked again)"
        files[answers_name()] = ("\n".join([json.dumps(r), *lines[1:]]) + "\n").encode()
    code, said = imported(rewrite(bundle_of(box, "again"), other))
    assert code == 1 and any("not imported — scoring it failed" in x for x in said), said
    assert results_of() == before
    assert not list(config.OUT_DIR.with_name("staging").glob("*"))


def test_12_a_bundle_is_capped_before_a_byte_is_read(tmp_path, monkeypatch):
    path = rb.write(tmp_path / "b.tar.gz", {"bundle.json": b"{}", "setup.json": b"{}",
                                            "results/r/t_0shot/frontier/answers.jsonl": b"x" * 4000})
    monkeypatch.setattr(rb, "MAX_MEMBER", 1000)
    with pytest.raises(ValueError, match="unpacks to .* more than a bundle's file can be"):
        rb.read(path)
    monkeypatch.setattr(rb, "MAX_MEMBER", 10 ** 6)
    monkeypatch.setattr(rb, "MAX_UNPACKED", 3000)
    with pytest.raises(ValueError, match="more than a bundle can be"):
        rb.read(path)


# ---------------------------------------------------------------------------
# part 4: the commands masein pastes
# ---------------------------------------------------------------------------

def test_13_the_token_is_read_never_on_a_command_line():
    assert "read -rs HF_TOKEN && export HF_TOKEN" in rg.__doc__
    assert "HF_TOKEN=…" not in rg.__doc__ and "HF_TOKEN=..." not in rg.__doc__


def test_15_the_tarball_script_says_when_git_cant_read_the_checkout(tmp_path):
    src = tmp_path / "llama.cpp"
    src.mkdir()
    (src / "CMakeLists.txt").write_text("project(x)\n")
    r = subprocess.run(["bash", str(REPO / "scripts" / "build_llama_tarball.sh"), str(src),
                        str(tmp_path / "out.tar.gz")], capture_output=True, text=True,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)})
    assert r.returncode == 1 and f"git can't read {src}" in r.stderr, (r.returncode, r.stderr)
    script = (REPO / "scripts" / "build_llama_tarball.sh").read_text()
    assert "safe.directory" in script and "doesn't know its commit" in script
    assert 'DOCKER="sudo docker"' in script and "ldd" in script


# ---------------------------------------------------------------------------
# part 5: scoring by code
# ---------------------------------------------------------------------------

def test_16_otis_reads_an_integer_only_when_nothing_follows_it():
    need = {"answer": "3"}
    for text, key in (("ANSWER: 3.5", "3"), ("ANSWER: 3/4", "3"), ("ANSWER: 2^{10}", "2"),
                      ("ANSWER: 12 or 13", "12")):
        sc = fb.score("otis_aime_epoch", text, "stop", {"answer": key})
        assert sc["read"] is None and sc["ok"] is False, text
    assert fb.score("otis_aime_epoch", "ANSWER: 3.", "stop", need)["ok"] is True
    assert fb.score("otis_aime_epoch", "**ANSWER: $\\boxed{3}$**", "stop", need)["ok"] is True


def test_17_a_benchmark_that_loads_another_count_asks_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "expected", fb._expected_n)      # the real count (conftest)
    monkeypatch.setattr(fb, "_fetch_source", lambda task: invented()[:3])
    with pytest.raises(ValueError, match="GPQA Diamond: 3 questions came from Idavidrein/gpqa "
                                         "at 83022cefff93, and this board expects 198"):
        fb.load(TASK, tmp_path)
    assert not fb.cached(TASK, tmp_path)                 # and nothing is kept
    # MMLU-Pro's 12,032, one short
    rows = {"items": [{"id": str(k), "category": "law"} for k in range(12031)], "extra": {}}
    monkeypatch.setattr(fb, "_fetch_source", lambda task: rows)
    with pytest.raises(ValueError, match="12,031 questions .* expects 12,032"):
        fb.load("mmlupro_tiger", tmp_path)
    # ARC-AGI-2 counted by its 120 tasks, not its test grids
    grids = [{"id": f"t{k}#{i}", "group": f"t{k}"} for k in range(120) for i in range(2)]
    monkeypatch.setattr(fb, "_fetch_source", lambda task: grids)
    assert len(fb.load("arc_agi2_public", tmp_path)) == 240
