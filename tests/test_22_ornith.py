"""22: Ornith-1.5-35B-A3B (Q4_K_M) on the Frontier set
(docs/prompts/phase-22-ornith-on-frontier.md) — the runtime card-ornith
(upstream llama.cpp b11555 and the card's settings, never the lda runtime),
its pinned file, the pilot, the box lines, the import, and its rows on the
reports. The build is the stand-in llama-server
(tests/fixtures/fake_llama_server.py) told to say it is upstream's b11555; the
questions are invented: nothing is fetched, no model runs, no paid API is
called."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import tarfile
from pathlib import Path

import pytest

import frontier as fb
import frontier_box as fx
import import_frontier as imf
import import_remote as ir
import remote_bundle as rb
import remote_gguf as rg
from service import config, reports, runner, served
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import FIX, TASK, box, free_port, invented, requests  # noqa: F401
from test_20_step4000 import card, lda_tarball  # noqa: F401

RT = "card-ornith"
ORNITH = "served/Ornith-1.5-35B-A3B-Q4_K_M"
ROW = "served__Ornith-1.5-35B-A3B-Q4_K_M__thinking"
BASED_ON = "ornith-ai/Ornith-1.5-35B-A3B"
GGUF = ("hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/"
        "Ornith-1.5-35B-Q4_K_M.gguf")
SHA = "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"
UPSTREAM = "1cf7767765b872806e1bb74cfb55614f4334f02f"
VERSION_LINE = ("version: 0.6.0-dev (build 11555, commit 1cf776776)\n"
                "built with GNU 11.4.0 for Linux x86_64")
TEMPLATE = "{# Ornith-1.5's own template, invented here #}"
MMLU = "mmlupro_tiger"
ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "ORNITH-RUN.md"


def upstream_tarball(root: Path, commit: str = UPSTREAM, changes: str = "no",
                     lookahead: bool = False, name: str = "llama-server-b11555.tar.gz") -> Path:
    """upstream's b11555 packed as build_llama_tarball.sh packs a build:
    bin/, lib/ (no LOOKAHEAD_* in it, unless `lookahead`), VERSION"""
    tb = root / name
    with tarfile.open(tb, "w:gz") as tar:
        def add(n: str, data: bytes, mode: int = 0o644) -> None:
            ti = tarfile.TarInfo(n)
            ti.size, ti.mode = len(data), mode
            tar.addfile(ti, io.BytesIO(data))
        add("llama/bin/llama-server", (FIX / "fake_llama_server.py").read_bytes(), 0o755)
        add("llama/lib/libllama.so", b"\x7fELF qwen35moe "
            + (b"LOOKAHEAD_MOE_ROUTER_NORM" if lookahead else b""))
        add("llama/lib/libggml-cuda.so", b"\x7fELF not really")
        add("llama/VERSION", f"commit {commit}\nuncommitted_changes {changes}\nbuild 11555\n"
                             "build_commit 1cf77677\ncuda_archs 120\n".encode())
    return tb


@pytest.fixture
def upstream(monkeypatch):
    """the stand-in says it is upstream's b11555 and serves Ornith's own
    template — the runtime's template sha256 is the stand-in's"""
    monkeypatch.setenv("FAKE_LLAMA_VERSION", VERSION_LINE)
    monkeypatch.setenv("FAKE_LLAMA_TEMPLATE", TEMPLATE)
    monkeypatch.setitem(fb.RUNTIMES[RT], "template_sha256",
                        hashlib.sha256(TEMPLATE.encode()).hexdigest())


def run_o(b: dict, out: str, tb: Path, *more: str, served_as: str = ORNITH,
          runtime: str | None = RT, only: bool = True) -> int:
    """remote_gguf.py from a tarball as an Ornith box runs it — the server's
    config put back after"""
    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR"]}
    lock, env = runner.LOCK, {v: os.environ.get(v) for v in rg.ENV.values()}
    try:
        return rg.main(["--as", served_as, "--gguf", str(b["gguf"]), "--server", str(tb),
                        "--based-on", BASED_ON, "--thinking", "on", "--slots", "3",
                        "--port", str(free_port()), "--load-timeout", "60",
                        *(["--runtime", runtime] if runtime else []),
                        *(["--only", TASK] if only else []),
                        "--out", str(b["root"] / out), *more])
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# the runtime
# ---------------------------------------------------------------------------

def test_the_runtime_is_the_cards_settings_on_upstream_llamacpp():
    rt = fb.RUNTIMES[RT]
    assert rt["env"] == {} and rt["thinking"] == ("on",)
    assert rt["sampling"] == {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                              "presence_penalty": 0.0, "repeat_penalty": 1.0}
    assert fb.runtime_flags(RT) == ["-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on",
                                    "--temp", "1.0", "--top-k", "20", "--top-p", "0.95",
                                    "--min-p", "0.0", "--presence-penalty", "0.0",
                                    "--repeat-penalty", "1.0"]
    assert rt["build"] == {"repo": "ggml-org/llama.cpp", "tag": "b11555", "number": 11555,
                           "commit": UPSTREAM}
    assert rt["models"] == {ORNITH: BASED_ON} and rt["refuses_env"] == ("LOOKAHEAD_",)
    # the GGUF's own template, read from its header at 12393612
    assert rt["template_sha256"] == \
        "f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b"
    words = fb.runtime_words(RT)
    assert words.startswith("runtime card-ornith (upstream llama.cpp b11555, Ornith-1.5's card "
                            "settings") and ": no variables · temperature 1.0" in words
    assert "presence penalty 0.0, repeat penalty 1.0 · -ctk q8_0" in words
    assert "LOOKAHEAD" not in words
    # the lda runtime as it was: its words and flags unchanged
    assert fb.runtime_words("lda-2610").startswith(
        "runtime lda-2610 (the team's lda branch: router norm on, no reasoning budget, no "
        "speculative decoding, thinking on): LOOKAHEAD_MOE_ROUTER_NORM=1 · temperature 1.0")


def test_each_request_carries_the_cards_sampling_at_our_limits():
    rec = {"id": ORNITH, "name": ORNITH, "based_on": BASED_ON, "runtime": RT,
           "base_url": "http://127.0.0.1:9/v1", "pin": {"file": "Ornith-1.5-35B-Q4_K_M.gguf"}}
    want = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True}}
    assert sf.settings(rec, TASK, True) == {"max_tokens": 81920, **want}
    assert sf.settings(rec, "math_l5_epoch", True)["max_tokens"] == 65536
    assert sf.settings(rec, MMLU, True)["max_tokens"] == 32768
    assert sf.settings(rec, "simpleqa_epoch", True)["max_tokens"] == 32768
    # no card preset of Qwen's under it: the runtime's values alone
    assert sf.family_of(rec) == ""


def test_ornith_runs_on_its_runtime_alone_and_the_runtime_runs_ornith_alone(tmp_path, capsys):
    assert fb.runtime_of_model(ORNITH) == RT and fb.runtime_of_model("served/step4000-lda-2610") == ""
    assert fb.runtime_model_refused(RT, ORNITH) == ""
    assert "runs on runtime card-ornith alone — not lda-2610" in \
        fb.runtime_model_refused("lda-2610", ORNITH)
    assert "give --runtime card-ornith" in fb.runtime_model_refused("", ORNITH)
    assert "runs served/Ornith-1.5-35B-A3B-Q4_K_M alone" in \
        fb.runtime_model_refused(RT, "served/step4000-lda-2610")
    assert fb.runtime_model_refused("lda-2610", "served/step4000-lda-2610") == ""
    for as_, rt in ((ORNITH, "lda-2610"), (ORNITH, None), ("served/step4000-lda-2610", RT)):
        with pytest.raises(SystemExit) as e:
            rg.main(["--as", as_, "--gguf", str(tmp_path / "x.gguf"), "--server",
                     str(tmp_path / "s.tar.gz"), "--thinking", "on", "--only", TASK,
                     *(["--runtime", rt] if rt else []), "--out", str(tmp_path / "o")])
        assert e.value.code == 2 and "alone" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()                    # refused before anything
    # and a box line, before the card is read or anything fetched
    with pytest.raises(SystemExit, match="L13: served/Ornith-1.5-35B-A3B-Q4_K_M runs on runtime "
                                         "card-ornith alone — not lda-2610. Nothing was fetched"):
        fx.main(["L13", "--as", ORNITH, "--gguf", GGUF, "--server", "s.tar.gz",
                 "--runtime", "lda-2610"])


@pytest.mark.parametrize("more, words", [
    (["--env", "LOOKAHEAD_MOE_ROUTER_NORM=1"], "never gives its server LOOKAHEAD_*"),
    (["--env", "LOOKAHEAD_ANYTHING=2"], "never gives its server LOOKAHEAD_*"),
    (["--flags", "--chat-template-file /tmp/t.jinja"], "--chat-template-file"),
    (["--flags", "--reasoning-format none"], "--reasoning-format"),
    (["--flags", "--presence-penalty 1.5"], "--presence-penalty"),
])
def test_the_runtime_refuses_the_lda_variables_and_another_template(tmp_path, capsys, more,
                                                                    words):
    with pytest.raises(SystemExit) as e:
        rg.main(["--as", ORNITH, "--gguf", str(tmp_path / "x.gguf"), "--server",
                 str(tmp_path / "s.tar.gz"), "--runtime", RT, "--thinking", "on",
                 "--only", TASK, "--out", str(tmp_path / "o"), *more])
    assert e.value.code == 2 and words in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_upstreams_version_line_is_read():
    """b11555 says "version: 0.6.0-dev (build 11555, commit 1cf776776)"; the
    lda build "version: 9954 (a4a76a75)" — both read"""
    rd = lambda text: re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text) \
        or re.search(r"\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\)", text)       # noqa: E731
    assert rd(VERSION_LINE).groups() == ("11555", "1cf776776")
    assert rd("version: 9954 (a4a76a75)\nbuilt with cc").groups() == ("9954", "a4a76a75")
    assert fb.runtime_build_refused(RT, 11555, "1cf776776") == ""
    assert fb.runtime_build_refused(RT, 11555, "1cf776776",
                                    {"commit": UPSTREAM, "uncommitted_changes": "no"}) == ""
    assert "this llama-server is build 9954 (a4a76a75)" in \
        fb.runtime_build_refused(RT, 9954, "a4a76a75")
    assert "this llama-server is build 11556" in fb.runtime_build_refused(RT, 11556, "1cf776776")
    assert "VERSION says commit a4a76a751" in fb.runtime_build_refused(
        RT, 11555, "1cf776776", {"commit": "a4a76a751" + "0" * 31})
    assert "changes not committed" in fb.runtime_build_refused(
        RT, 11555, "1cf776776", {"commit": UPSTREAM, "uncommitted_changes": "yes"})
    assert fb.runtime_build_refused("lda-2610", 9954, "a4a76a75") == ""     # names none


def test_a_box_runs_upstream_b11555_and_records_it(box, card, upstream, monkeypatch, tmp_path):  # noqa: F811
    launched = tmp_path / "launched.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(launched))
    monkeypatch.setenv("LOOKAHEAD_MOE_ROUTER_NORM", "1")     # in the box's shell: never passed on
    tb = upstream_tarball(tmp_path)
    assert run_o(box, "o", tb, "--pin", f"{tb.name}={rb.sha256_file(tb)}") == 0
    b = rb.read(box["root"] / "o" / rb.bundle_name("frontier", ORNITH, True, None,
                                                   parts=[fb.BENCH[TASK]["short"]]))
    srv = b["setup"]["server"]
    assert b["setup"]["runtime"] == RT and srv["runtime"] == RT
    assert (srv["build"], srv["commit"]) == (11555, "1cf776776")
    assert srv["flags"] == fb.runtime_flags(RT) and srv["env"] == {}
    assert srv["chat_template_sha256"] == hashlib.sha256(TEMPLATE.encode()).hexdigest()
    assert b["setup"]["based_on"] == BASED_ON
    got = json.loads(launched.read_text().splitlines()[-1])
    assert got["env"] == {} and "--jinja" in got["argv"]
    assert all(r["temperature"] == 1.0 and r["top_p"] == 0.95 and r["top_k"] == 20
               and r["min_p"] == 0.0 and r["presence_penalty"] == 0.0
               and r["repeat_penalty"] == 1.0 and "repeat_last_n" not in r
               and r["chat_template_kwargs"] == {"enable_thinking": True}
               for r in requests(box))
    # the import takes it, registered with its runtime
    assert imf.runtime_problem(b["setup"], {"id": ORNITH, "runtime": RT}) == ""
    assert ir.main([str(box["root"] / "o" / rb.bundle_name("frontier", ORNITH, True, None,
                                                            parts=[fb.BENCH[TASK]["short"]])),
                    "--by", "masein", "--register", "Ornith-1.5-35B-A3B Q4_K_M (card settings)",
                    "--file-sha256", box["sha"]]) == 0
    rec = served.get(ORNITH)
    assert rec["name"] == "Ornith-1.5-35B-A3B Q4_K_M (card settings)" and rec["runtime"] == RT
    assert "runtime card-ornith" in rec["how"] and rec["based_on"] == BASED_ON


@pytest.mark.parametrize("kind, words", [
    ("lda", "never runs a build that reads LOOKAHEAD_MOE_ROUTER_NORM (the lda branch's)"),
    ("version", "runs upstream llama.cpp b11555 (build 11555, 1cf776776), and this llama-server "
                "is build 9954 (a4a76a75)"),
    ("commit", "this tarball's VERSION says commit a4a76a751000"),
    ("changes", "built from a tree with changes not committed"),
])
def test_a_build_that_isnt_upstreams_b11555_is_refused_before_the_gguf(box, card, upstream,  # noqa: F811
                                                                        monkeypatch, tmp_path,
                                                                        kind, words):
    if kind == "lda":
        tb = upstream_tarball(tmp_path, lookahead=True)
    elif kind == "version":
        monkeypatch.setenv("FAKE_LLAMA_VERSION", "version: 9954 (a4a76a75)\nbuilt with cc")
        tb = upstream_tarball(tmp_path)
    elif kind == "commit":
        tb = upstream_tarball(tmp_path, commit="a4a76a751" + "0" * 31)
    else:
        tb = upstream_tarball(tmp_path, changes="yes")
    fetched = []
    real = rg.fetch
    monkeypatch.setattr(rg, "fetch", lambda src, into, say: fetched.append(src) or real(src, into,
                                                                                    say))
    with pytest.raises(SystemExit, match=re.escape(words)):
        run_o(box, "bad", tb)
    assert fetched == [str(tb)] and not requests(box)       # the GGUF never fetched
    # the lda tarball itself, the same way
    if kind == "lda":
        with pytest.raises(SystemExit, match="never runs a build that reads LOOKAHEAD_MOE"):
            run_o(box, "lda", lda_tarball(tmp_path))


def test_a_server_with_another_template_is_refused_before_a_question(box, card, upstream,  # noqa: F811
                                                                     monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_LLAMA_TEMPLATE", "{# Qwen's own, not Ornith's #}")
    with pytest.raises(SystemExit, match=r"asks with the GGUF's own chat template \(sha256 "):
        run_o(box, "t", upstream_tarball(tmp_path))
    assert not requests(box)


def test_the_import_refuses_ornith_asked_any_other_way(box, card, upstream, tmp_path):  # noqa: F811
    assert run_o(box, "o", upstream_tarball(tmp_path)) == 0
    b = rb.read(box["root"] / "o" / rb.bundle_name("frontier", ORNITH, True, None,
                                                   parts=[fb.BENCH[TASK]["short"]]))
    rec = {"id": ORNITH, "runtime": RT}
    assert imf.runtime_problem(b["setup"], rec) == ""

    def bad(**srv):
        s = json.loads(json.dumps(b["setup"]))
        s["server"].update(srv)
        return imf.runtime_problem(s, rec)
    assert "no LOOKAHEAD_MOE_ROUTER_NORM" in bad(env={"LOOKAHEAD_MOE_ROUTER_NORM": "1"})
    assert "this llama-server is build 9954" in bad(build=9954, commit="a4a76a75")
    assert "the GGUF's own chat template" in bad(chat_template_sha256="0" * 64)
    # asked with the lda runtime, or registered with it: never imported
    s = json.loads(json.dumps(b["setup"]))
    s["runtime"] = "lda-2610"
    assert "runs on runtime card-ornith alone" in imf.runtime_problem(s, {"id": ORNITH,
                                                                          "runtime": "lda-2610"})
    assert "runs served/Ornith-1.5-35B-A3B-Q4_K_M alone" in imf.runtime_problem(
        b["setup"], {"id": "served/step4000-lda-2610", "runtime": RT})


# ---------------------------------------------------------------------------
# the pilot
# ---------------------------------------------------------------------------

def invented_mmlu() -> dict:
    """MMLU-Pro's shape, invented: a question, ten options, its letter"""
    items = [{"id": str(100 + k), "question": f"Invented MMLU-Pro question {k}?",
              "options": [f"option {k}{c}" for c in "abcdefghij"], "answer": "AB"[k % 2],
              "category": "invented"} for k in range(9)]
    return {"items": items, "extra": {"shots": {"invented": []}}}


@pytest.fixture
def pilot_world(box, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fb, "_fetch", lambda task: invented_mmlu() if task == MMLU
                        else invented())
    return box


def test_the_pilot_asks_its_questions_and_says_last_whether_they_came_apart(
        pilot_world, card, upstream, tmp_path, capsys):  # noqa: F811
    b = pilot_world
    assert fb.PILOT["n"] == {TASK: 20, MMLU: 5}
    out = b["root"] / "Ornith-1.5-35B-A3B-Q4_K_M" / "pilot"
    assert run_o(b, out.relative_to(b["root"]).as_posix(), upstream_tarball(tmp_path),
                 "--pilot", only=False) == 0
    said = [x for x in capsys.readouterr().out.splitlines() if x.strip()]
    lines = [x for x in said if " of 12 (question rec" in x or " of 5 (question 1" in x]
    # GPQA's 12 invented (fewer than 20 here), MMLU-Pro's 5 of 9
    assert len(lines) == 17
    one = next(x for x in lines if "(question rec003)" in x)
    assert re.search(r"GPQA Diamond \d+ of 12 \(question rec003\): thinking 29 characters · 12 "
                     r"tokens in all · read [ABCD] · right [ABCD] ✗ · \d+ s$", one)
    assert any(re.search(r"MMLU-Pro \d of 5 \(question 1\d\d\): .* read A · right A ✓", x)
               for x in lines)
    assert re.search(r"^GPQA Diamond: 9 of 12 right · thinking 29 characters on average · "
                     r"\d+ s an answer, 3 at a time", next(x for x in said if x.startswith(
                         "GPQA Diamond: ") and " right · " in x))
    assert said[-1].startswith("Thinking and answer separated correctly: all 17 answers that "
                               "finished came back with the thinking apart")
    rows = [json.loads(x) for x in (out / "pilot.jsonl").read_text().splitlines()]
    assert len(rows) == 17 and all(r["thinking_apart"] and not r["reply_tags"] for r in rows)
    summary = json.loads((out / "pilot.json").read_text())
    assert summary["separated"] is True and summary["runtime"] == RT
    assert summary["paces"][TASK]["answers"] == 12 and summary["paces"][MMLU]["answers"] == 5
    # no bundle, no progress file: the fetch never takes it for a step
    assert not list(out.glob("frontier-*.tar.gz")) and not (out / "progress.json").exists()
    assert "-" not in out.name
    # each question asked once, thinking on, with the card's settings
    reqs = requests(b)
    assert len(reqs) == 17 and all(r["chat_template_kwargs"] == {"enable_thinking": True}
                                   and r["temperature"] == 1.0 for r in reqs)


@pytest.mark.parametrize("think, why", [
    ("never", "no thinking apart from the reply (reasoning_content empty)"),
    ("inline", "no thinking apart from the reply (reasoning_content empty)"),
])
def test_a_pilot_whose_thinking_isnt_apart_says_so_last(pilot_world, card, upstream,  # noqa: F811
                                                         monkeypatch, tmp_path, capsys, think,
                                                         why):
    monkeypatch.setenv("FAKE_LLAMA_THINK", think)
    assert run_o(pilot_world, "pilot", upstream_tarball(tmp_path), "--pilot", only=False) == 1
    said = [x for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert said[-1].startswith("Thinking and answer NOT separated correctly in 17 of 17 answers")
    assert why in said[-1] and said[-1].endswith("Don't rent the other boxes yet")


def test_the_pilot_verdict_never_judges_an_answer_that_ran_out():
    ok = {"task": TASK, "id": "a", "thinking_apart": True, "reply_chars": 9, "reply_tags": False}
    out = {"task": TASK, "id": "b", "ran_out": True, "thinking_apart": True, "reply_chars": 0}
    assert rg.pilot_verdict([ok, out]) == (True, (
        "Thinking and answer separated correctly: all 1 answers that finished came back with "
        "the thinking apart (reasoning_content) and the answer in the reply, no think tags in "
        "it; 1 ran out of room before replying, not judged"))
    assert rg.pilot_verdict([out])[0] is False
    assert "think tags in the reply" in rg.pilot_verdict([{**ok, "reply_tags": True}])[1]
    assert "an empty reply" in rg.pilot_verdict([{**ok, "reply_chars": 0}])[1]
    assert "without the server's chat parsing" in rg.pilot_verdict([{**ok, "fallback": True}])[1]


def test_a_pilot_is_a_runtimes_thinking_on_and_asks_its_own_questions(tmp_path, capsys):
    for more, words in ((["--thinking", "off", "--runtime", RT], "--pilot: a runtime's pilot"),
                        (["--thinking", "on", "--runtime", RT, "--only", TASK],
                         "--pilot asks its own questions")):
        with pytest.raises(SystemExit):
            rg.main(["--as", ORNITH, "--gguf", "g", "--server", "s", "--pilot",
                     "--out", str(tmp_path / "o"), *more])
        assert words in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# the box lines
# ---------------------------------------------------------------------------

def test_an_ornith_box_line_runs_its_runtime_and_pins_its_file():
    assert fx.PLAN_PINS["L"]["Ornith-1.5-35B-Q4_K_M.gguf"] == SHA
    for more in ([], ["--runtime", RT]):
        a = fx.parser().parse_args(["L13", "--as", ORNITH, "--gguf", GGUF, "--server",
                                    "hf://me/p@c/llama-server-b11555.tar.gz", *more])
        argv = fx.argv_of("L13", 1, fx.box_of("L13")[0], a)
        assert argv[argv.index("--runtime") + 1] == RT
        assert argv[argv.index("--based-on") + 1] == BASED_ON
        assert argv[argv.index("--flags") + 1] == ""
        assert [argv[k + 1] for k, x in enumerate(argv) if x == "--pin"] == [
            f"Ornith-1.5-35B-Q4_K_M.gguf={SHA}"]
        assert argv[argv.index("--out") + 1] == "/workspace/Ornith-1.5-35B-A3B-Q4_K_M/L13-1"
    # the pilot: the same files and runtime, its own folder, the plan's 8 slots
    argv = fx.argv_of("L1", 0, fx.PILOT_STEP, a)
    assert "--pilot" in argv and "--only" not in argv and "--shard" not in argv
    assert argv[argv.index("--slots") + 1] == "8" and argv[argv.index("--thinking") + 1] == "on"
    assert argv[argv.index("--out") + 1] == "/workspace/Ornith-1.5-35B-A3B-Q4_K_M/pilot"
    # step4000's lines as before: the plan's runtime and Qwen
    a = fx.parser().parse_args(["L13", "--as", "served/step4000-lda-2610", "--gguf", "g",
                                "--server", "s"])
    argv = fx.argv_of("L13", 1, fx.box_of("L13")[0], a)
    assert argv[argv.index("--runtime") + 1] == "lda-2610"
    assert argv[argv.index("--based-on") + 1] == "Qwen/Qwen3.6-35B-A3B"


def test_the_pilot_line_prints_the_plan_at_its_pace_then_its_verdict_last(monkeypatch, capsys,
                                                                          tmp_path):
    monkeypatch.setattr(fx, "gpu_refused", lambda label: "")

    def ran(cmd):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "pilot.json").write_text(json.dumps({
            "separated": True, "verdict": "Thinking and answer separated correctly: all 25",
            "paces": {TASK: {"answers": 20, "pace_s": 19.5, "at_once": 8},
                      MMLU: {"answers": 5, "pace_s": 2.1, "at_once": 5}}}))
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(fx.subprocess, "run", ran)
    assert fx.main(["L1", "--pilot", "--as", ORNITH, "--gguf", GGUF, "--server", "s.tar.gz",
                    "--root", str(tmp_path)]) == 0
    said = [x for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert said[0] == ("L1: the pilot — 20 GPQA Diamond and 5 MMLU-Pro questions, thinking on, "
                       "each asked once, before the other boxes are rented")
    assert said[1] == fb.runtime_words(RT)
    assert any(x.startswith("plan L at GPQA Diamond's pace here: about ") and "box-hours, $" in x
               and "at $0.50 an hour — from 20 answers, a rough guide" in x for x in said)
    assert said[-1] == "Thinking and answer separated correctly: all 25"
    # no line file: the fetch never waits for a pilot
    assert not list(tmp_path.rglob("*.line.json"))
    with pytest.raises(SystemExit):
        fx.main(["L1", "L2", "--pilot", "--as", ORNITH, "--gguf", GGUF, "--server", "s"])


def doc_lines() -> list[list[str]]:
    text = DOC.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    return [shlex.split(b.strip()) for b in blocks if "frontier_box.py" in b]


def test_the_docs_box_lines_are_the_pilot_and_L1_to_L21_whole():
    lines = doc_lines()
    pilot = [x for x in lines if "--pilot" in x]
    boxes = [x for x in lines if "--pilot" not in x]
    assert len(pilot) == 1 and pilot[0][2] == "L1"
    assert [x[2] for x in boxes] == [f"L{i}" for i in range(1, 22)]
    for x in lines:
        a = fx.parser().parse_args(x[2:])
        assert a.served_as == ORNITH and a.gguf == GGUF and a.runtime == RT
        assert a.server == "hf://<you>/evalboard-private@<commit>/llama-server-b11555.tar.gz"
        assert not a.based_on and not a.flags and not a.env
    text = DOC.read_text(encoding="utf-8")
    # the first import, with --register and the file's sha256; the fetch with it
    assert (f'--register "Ornith-1.5-35B-A3B Q4_K_M (card settings)" --file-sha256 {SHA}'
            in text)
    assert f"--sha {ORNITH}={SHA}" in text
    # the tarball, built from the tag as the runtime names it
    assert "git -C ~/llama.cpp-upstream checkout b11555" in text
    assert ('LLAMA_CUDA_ARCHS=120 LLAMA_CMAKE_ARGS="-DGGML_CUDA_CCCL_VERSION=v3.4.3 '
            '-DLLAMA_OPENSSL=OFF" scripts/build_llama_tarball.sh ~/llama.cpp-upstream '
            '~/llama-server-b11555.tar.gz') in text
    assert f"`{SHA}`" in text and "21,713,463,040 bytes" in text


# ---------------------------------------------------------------------------
# the reports
# ---------------------------------------------------------------------------

def test_ornith_is_ours_on_latest_and_its_cards_claim_sits_beside_it(svc):  # noqa: F811
    from test_21c_page import world
    world()
    rep = json.loads((reports.REPORTS_DIR / "step-checkpoints.json").read_text())
    o = next(x for x in rep["ours"] if x["label"] == "Ornith-1.5 Q4_K_M")
    assert o == {"label": "Ornith-1.5 Q4_K_M", "row": "served/Ornith-1.5-35B-A3B-Q4_K_M · thinking",
                 "colour": 5, "pick": True, "glance": True,
                 "note": "measured here, on its card's settings; not the team's model"}
    # the next free colour: no other row of ours has it
    assert [x.get("colour") for x in rep["ours"] if x is not o].count(5) == 0
    assert max(x["colour"] for x in rep["ours"] if x is not o and x.get("colour") is not None) == 4
    t = next(x for x in rep["theirs"] if x["label"] == "Ornith-1.5-35B-A3B, its card")
    url = ("https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/blob/"
           "12393612fd4f730ff5aadc23e9b8f9648aa49ceb/README.md")
    assert (t["gpqa"]["value"], t["hle"]["value"]) == (89.2, 25.6)
    assert t["gpqa"]["url"] == t["hle"]["url"] == url
    assert t["gpqa"]["note"] == t["hle"]["note"] == "5-run average"
    assert "no tools" in t["hle"]["setting"] and t["glance"] is True and t["epoch_version"] is None
    assert not {"otis", "math", "mmlupro", "simpleqa", "arc"} & set(t)
    got = svc.get("/api/reports/step-checkpoints").json()
    mine = next(r for r in got["ours"] if r["label"] == "Ornith-1.5 Q4_K_M")
    assert mine["glance"] and mine["pick"] and mine["colour"] == 5
    assert "Ornith-1.5 Q4_K_M" in got["picks"]
    claim = next(r for r in got["theirs"] if r["label"] == "Ornith-1.5-35B-A3B, its card")
    assert claim["glance"] and claim["source"] == "Ornith's model card (5-run average)"
    assert claim["cells"]["gpqa"]["v"] == pytest.approx(0.892)
    assert claim["cells"]["hle"]["v"] == pytest.approx(0.256)
    # Latest's rows: ours marked glance (every picked one), Ornith among them
    from service import app
    meta = app.results_payload()["glance"]
    assert meta["rows"][-1] == "served/Ornith-1.5-35B-A3B-Q4_K_M · thinking"
    assert meta["rows"] == [x["row"] for x in rep["ours"] if x.get("glance")]
    # and the ran-out report's picker
    ro = json.loads((reports.REPORTS_DIR / "ran-out-of-room.json").read_text())
    assert [x["label"] for x in ro["ours"] if x.get("pick")] == [
        "step4000", "step1000", "Ornith-1.5 Q4_K_M"]
    assert next(x for x in ro["ours"] if x["label"] == "Ornith-1.5 Q4_K_M")["colour"] == 5
