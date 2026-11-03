"""31.1: step2388, the team's joint k3/k6 checkpoint — runtime k3k6-2611, its
settings check and plan U. One GGUF, two expert paths: k3 (no thinking) and
k6 (thinking), each path's sampling and thinking stored in the file and
applied by llama-server, so a request carries its path and never a sampling
value or a thinking switch. The server is the stand-in
(tests/fixtures/fake_llama_server.py, FAKE_LLAMA_PATHS=1), the file a GGUF
header written here, the questions invented; nothing is fetched and no model
runs."""

from __future__ import annotations

import io
import json
import os
import re
import struct
import tarfile
from pathlib import Path

import pytest

import frontier as fb
import frontier_box as fx
import gguf_header
import import_frontier as imf
import import_remote as ir
import path_check
import remote_bundle as rb
import remote_gguf as rg
from service import config, runner, served
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import FIX, TASK, box, free_port, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401

RT = "k3k6-2611"
STEP = "served/step2388-k3k6-2611"
NAME = "step2388.k3k6.gguf"
SHA = "a2594a68a810ec526061f83cb88c7e469e1c3889912e8da4b1d598c72271faf6"
VERSION_LINE = "version: 7000 (3ad38b7cc)\nbuilt with GNU 11.4.0 for Linux x86_64"
GPQA, MATH, MMLU = "gpqa_diamond_epoch", "math_l5_epoch", "mmlupro_tiger"
ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "REMOTE-RUNS.md"


# ---------------------------------------------------------------------------
# a file of expert paths, as its header holds them
# ---------------------------------------------------------------------------

def _s(x: bytes) -> bytes:
    return struct.pack("<Q", len(x)) + x


def _kv(k: str, t: int, v: bytes) -> bytes:
    return _s(k.encode()) + struct.pack("<I", t) + v


def _val(v) -> tuple[int, bytes]:
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, int):
        return 5, struct.pack("<i", v)
    if isinstance(v, list):
        return 9, struct.pack("<I", 5) + struct.pack("<Q", len(v)) + b"".join(
            struct.pack("<i", x) for x in v)
    return 8, _s(str(v).encode())


STORED = {"teraformer.paths.top_k": [3, 6], "teraformer.paths.default": 3,
          "teraformer.paths.k3.sampling.temp": 0.8, "teraformer.paths.k3.sampling.top_p": 0.95,
          "teraformer.paths.k3.sampling.top_k": 20, "teraformer.paths.k3.sampling.min_p": 0.0,
          "teraformer.paths.k3.sampling.presence_penalty": 0.0,
          "teraformer.paths.k3.thinking": False,
          "teraformer.paths.k6.sampling.temp": 0.8, "teraformer.paths.k6.sampling.top_p": 0.95,
          "teraformer.paths.k6.sampling.top_k": 20, "teraformer.paths.k6.sampling.min_p": 0.0,
          "teraformer.paths.k6.sampling.presence_penalty": 0.75,
          "teraformer.paths.k6.sampling.penalty_last_n": -1,
          "teraformer.paths.k6.sampling.penalty_prompt_tokens": 0,
          "teraformer.paths.k6.thinking": True}


def write_gguf(path: Path, **change) -> Path:
    """a GGUF header with step2388's path entries (`change`: key suffix → value,
    k6__sampling__presence_penalty=0.5), the architecture's name, and no tensors"""
    kv = {"general.architecture": "qwen35moe", **STORED,
          "teraformer.reference_gguf": "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"}
    for k, v in change.items():
        kv["teraformer.paths." + k.replace("__", ".")] = v
    entries = [_kv(k, *_val(v)) for k, v in kv.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", len(entries)) + b"".join(entries) + b"\0" * 4096)
    return path


def k3k6_tarball(root: Path, commit: str = "3ad38b7cc", changes: str = "yes",
                 changed: str = "Makefile", reads: bool = True) -> Path:
    """the team's build packed as build_llama_tarball.sh packs it: bin/, lib/
    (that reads the router norm and the CPU fusion switch), VERSION — the tree
    with its Makefile changed, which CMake never reads"""
    tb = root / "llama-server-k3k6-3ad38b7.tar.gz"
    with tarfile.open(tb, "w:gz") as tar:
        def add(n: str, data: bytes, mode: int = 0o644) -> None:
            ti = tarfile.TarInfo(n)
            ti.size, ti.mode = len(data), mode
            tar.addfile(ti, io.BytesIO(data))
        add("llama/bin/llama-server", (FIX / "fake_llama_server.py").read_bytes(), 0o755)
        add("llama/lib/libllama.so", b"\x7fELF " + (b"LOOKAHEAD_MOE_ROUTER_NORM" if reads else b""))
        add("llama/lib/libggml-cpu.so", b"\x7fELF GGML_CPU_DISABLE_FUSION")
        add("llama/lib/libggml-cuda.so", b"\x7fELF not really")
        add("llama/VERSION", (f"commit {commit}{'0' * 31}\nuncommitted_changes {changes}\n"
                              + (f"changed_files {changed}\n" if changed else "")
                              + "build 7000\ncuda_archs 120\n").encode())
    return tb


@pytest.fixture(autouse=True)
def step_kept():
    """remote_gguf.py's step state as it was: --check marks the step a pilot's
    (no progress file), and a later test's own rg.progress would write none"""
    was = dict(rg.STEP)
    yield
    rg.STEP.clear()
    rg.STEP.update(was)


@pytest.fixture
def k3k6(box, card, monkeypatch, tmp_path):  # noqa: F811
    """a box given step2388's file and the team's build, its server serving
    the two paths"""
    monkeypatch.setenv("FAKE_LLAMA_PATHS", "1")
    monkeypatch.setenv("FAKE_LLAMA_VERSION", VERSION_LINE)
    gguf = write_gguf(tmp_path / "models" / NAME)
    return {**box, "gguf": gguf, "sha": rb.sha256_file(gguf), "tarball": k3k6_tarball(tmp_path)}


def run_k(b: dict, out: str, *more: str, thinking: str = "on", only: bool = True,
          gguf: Path | None = None) -> int:
    """remote_gguf.py as a plan U box runs it — the server's config put back after"""
    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR"]}
    lock, env = runner.LOCK, {v: os.environ.get(v) for v in rg.ENV.values()}
    try:
        return rg.main(["--as", STEP, "--gguf", str(gguf or b["gguf"]), "--server",
                        str(b["tarball"]), "--based-on", "Qwen/Qwen3.6-35B-A3B",
                        "--thinking", thinking, "--slots", "3", "--port", str(free_port()),
                        "--load-timeout", "60", "--runtime", RT,
                        *(["--only", TASK] if only else []), "--out", str(b["root"] / out),
                        *more])
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bundle_at(b: dict, out: str, thinking: bool = True) -> Path:
    return b["root"] / out / rb.bundle_name("frontier", STEP, thinking, None,
                                            parts=[fb.BENCH[TASK]["short"]])


CHECK_QUESTIONS = set(path_check.QUESTIONS)


def asked(b: dict) -> tuple[list[dict], list[dict]]:
    """the requests the server answered: the settings check's, then the benchmark's"""
    got = requests(b)
    chk = [r for r in got if (r.get("messages") or [{}])[-1].get("content") in CHECK_QUESTIONS]
    return chk, [r for r in got if r not in chk]


# ---------------------------------------------------------------------------
# the runtime
# ---------------------------------------------------------------------------

def test_the_runtime_sends_no_sampling_and_asks_each_path_by_its_thinking():
    rt = fb.RUNTIMES[RT]
    assert rt["env"] == {"LOOKAHEAD_MOE_ROUTER_NORM": "1", "GGML_CPU_DISABLE_FUSION": "1"}
    # no sampling flag: one would replace both paths' own values
    assert fb.runtime_flags(RT) == ["-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on"]
    assert fb.path_of(RT, True) == 6 and fb.path_of(RT, "off") == 3
    assert fb.path_of("lda-2610", True) is None
    # the team's values, as the box's check holds the file and the server to them
    assert fb.PATHS[3] == {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                           "presence_penalty": 0.0, "thinking": False}
    assert fb.PATHS[6] == {**fb.PATHS[3], "presence_penalty": 0.75, "penalty_last_n": -1,
                           "penalty_prompt_tokens": 0, "thinking": True}
    rec = {"id": STEP, "name": "step2388", "based_on": "Qwen/Qwen3.6-35B-A3B", "runtime": RT,
           "base_url": "http://127.0.0.1:9/v1", "pin": {}}
    # a request: its limit and its path, never the qwen3.6 preset or the switch
    assert sf.settings(rec, GPQA, True) == {"max_tokens": 81920, "expert_path": 6}
    for t, limit in ((GPQA, 16384), (MATH, 8192), (MMLU, 8192)):
        assert sf.settings(rec, t, False) == {"max_tokens": limit, "expert_path": 3}
    # k3 is the whitepaper's thinking-off set alone
    assert fb.runtime_thinking_refused(RT, "off", [GPQA, MATH, MMLU, "ifeval_google"]) == ""
    assert fb.runtime_thinking_refused(RT, "off", ["hle_text_cais"]).startswith(
        "--runtime k3k6-2611 asks thinking on only")
    # said on each step's line: the path's values, the file's, never sent
    w = fb.runtime_words(RT, "on")
    assert "LOOKAHEAD_MOE_ROUTER_NORM=1 GGML_CPU_DISABLE_FUSION=1" in w
    assert ("expert path k6: temperature 0.8, top-p 0.95, top-k 20, min-p 0.0, presence "
            "penalty 0.75, penalty last-n -1, penalty prompt tokens 0, thinking on — the file's, "
            "never sent") in w
    assert "expert path k3: " in fb.runtime_words(RT, "off")
    # step2388 runs on this runtime alone, and the runtime runs step2388 alone
    assert fb.runtime_model_refused(RT, STEP) == ""
    assert fb.runtime_model_refused("lda-2610", STEP).startswith(f"{STEP} runs on runtime {RT}")
    assert "runs served/step2388-k3k6-2611 alone" in fb.runtime_model_refused(
        RT, "served/step8804-lda-2610")


def test_a_request_on_a_path_carries_its_messages_limit_path_and_seed_alone(monkeypatch):
    sent = []
    monkeypatch.setattr(served, "_post", lambda url, key, body, **k: sent.append(body) or {
        "choices": [{"message": {"content": "ANSWER: A"}, "finish_reason": "stop"}]})
    rec = {"id": STEP, "name": "step2388", "base_url": "http://127.0.0.1:9/v1", "key": "",
           "pin": {"model": NAME}, "runtime": RT}
    served.ask(rec, "A question?", {**sf.settings(rec, GPQA, True), "seed": 7})
    assert sorted(sent[0]) == ["expert_path", "max_tokens", "messages", "seed"]
    assert served.thinking_key({"expert_path": 6}) == "on"
    assert served.thinking_key({"expert_path": 3}) == "off"


def test_an_answer_on_a_path_is_never_asked_without_its_chat_endpoint(monkeypatch):
    calls = []

    def failing(rec, text, s, how=None):
        calls.append(how)
        raise served.ItemError("HTTP 500: it failed")
    monkeypatch.setattr(served, "_ask_patiently", failing)
    a = served.answer_one({"id": STEP}, "A question?", {"max_tokens": 9, "expert_path": 6})
    assert str(a) == "" and calls == [None, None]           # twice, never the raw completion
    assert a.error["fallback"] == "not asked: an expert path is asked through its chat endpoint alone"


@pytest.mark.parametrize("flags, env, words", [
    ("--temp 0.7", "", "--temp"),
    ("--reasoning-format none", "", "--reasoning-format"),
    ("--expert-path 6", "", "--expert-path"),
    ("--slot-save-path /tmp/s", "", "--slot-save-path"),
    ("--spec-draft-n-max 4", "", "--spec-draft-n-max"),
    ("", "LLAMA_ARG_EXPERT_PATH=6", "LLAMA_ARG_EXPERT_PATH"),
    ("", "LLAMA_ARG_TEMP=0.7", "LLAMA_ARG_TEMP"),
    ("", "GGML_CUDA_DISABLE_FUSION=1", "GGML_CUDA_DISABLE_FUSION"),
])
def test_nothing_that_would_replace_a_paths_own_reaches_its_server(tmp_path, capsys, flags, env,
                                                                    words):
    with pytest.raises(SystemExit):
        rg.main(["--as", STEP, "--gguf", str(tmp_path / NAME), "--server", str(tmp_path / "t.tgz"),
                 "--runtime", RT, "--thinking", "on", "--only", GPQA,
                 *(["--flags", flags] if flags else []), *(["--env", env] if env else []),
                 "--out", str(tmp_path / "o")])
    assert words in capsys.readouterr().err


def test_the_build_is_the_teams_commit_with_its_makefile_alone_changed():
    vf = {"commit": "3ad38b7cc" + "0" * 31, "uncommitted_changes": "yes",
          "changed_files": "Makefile"}
    assert fb.runtime_build_refused(RT, 7000, "3ad38b7cc", vf) == ""
    # d98b64741, the commit before (the README alone differs), runs too
    assert fb.runtime_build_refused(RT, 7000, "d98b6474", {"commit": "d98b64741"}) == ""
    assert fb.runtime_build_refused(RT, 9954, "a4a76a75").startswith(
        "runtime k3k6-2611 runs the team's llama.cpp-teraformer at 3ad38b7cc (d98b64741's code), "
        "and this llama-server is build 9954 (a4a76a75)")
    assert "changes not committed (its VERSION: src/llama.cpp)" in fb.runtime_build_refused(
        RT, 7000, "3ad38b7cc", {**vf, "changed_files": "Makefile src/llama.cpp"})
    # changes the VERSION doesn't name: never taken for the Makefile
    assert "changes not committed (its VERSION: unnamed)" in fb.runtime_build_refused(
        RT, 7000, "3ad38b7cc", {"commit": "3ad38b7cc", "uncommitted_changes": "yes"})
    # upstream's runtimes are as they were: no change allowed at all
    assert "as released" in fb.runtime_build_refused(
        "card-ornith", 11555, "1cf776776", {"commit": "1cf7767765b872806e1bb74cfb55614f4334f02f",
                                            "uncommitted_changes": "yes",
                                            "changed_files": "Makefile"})


# ---------------------------------------------------------------------------
# the file's own values
# ---------------------------------------------------------------------------

def test_the_header_gives_the_paths_entries_and_nothing_else(tmp_path):
    f = write_gguf(tmp_path / NAME)
    got = gguf_header.keys(f, "teraformer.paths.")
    assert got["teraformer.paths.top_k"] == [3, 6] and got["teraformer.paths.default"] == 3
    assert abs(got["teraformer.paths.k6.sampling.presence_penalty"] - 0.75) < 1e-6
    assert got["teraformer.paths.k6.thinking"] is True
    assert "general.architecture" not in got and "teraformer.reference_gguf" not in got
    st = path_check.stored(f)
    assert path_check.compare(fb.PATHS[6], st["paths"][6], "k6") == []
    assert path_check.compare(fb.PATHS[3], st["paths"][3], "k3") == []
    assert gguf_header.keys(tmp_path / "missing.gguf", "x") is None
    (tmp_path / "bad.gguf").write_bytes(b"NOPE" * 10)
    assert gguf_header.keys(tmp_path / "bad.gguf", "x") is None


def test_the_servers_words_are_read_by_any_of_their_names():
    lines = ("srv  load_model: expert path k3 (default): temp = 0.800, top_k = 20, top_p = 0.950\n"
             "srv  load_model: expert path k6: temp = 0.800, presence_penalty = 0.750, "
             "penalty_last_n = -1, thinking = true\n")
    got = path_check.startup(lines)
    assert got[3]["default"] and not got[6]["default"]
    assert got[6]["values"] == {"temperature": 0.8, "presence_penalty": 0.75,
                                "penalty_last_n": -1, "thinking": True}
    # /props as a list, or keyed by the path
    props = {"expert_paths": [{"top_k": 6, "params": {"temperature": 0.8, "repeat_last_n": 32768},
                               "thinking": True}]}
    assert path_check.props_paths(props)[6]["params"] == {
        "temperature": 0.8, "penalty_last_n": 32768, "thinking": True}
    assert path_check.props_paths({"expert_paths": {"k3": {"params": {"temp": 0.8}}}})[3][
        "params"] == {"temperature": 0.8}
    # -1 is the whole context: a server may say the context it resolved it to
    assert path_check.compare({"penalty_last_n": -1}, {"penalty_last_n": 32768}, "x", 16384) == []
    assert path_check.compare({"penalty_last_n": -1}, {"penalty_last_n": 64}, "x", 16384) == [
        "x: penalty last-n 64, not -1"]
    assert path_check.compare({"top_k": 20}, {}, "x") == ["x gives no top-k"]
    assert path_check.compare({"top_k": 20}, {}, "x", all_of=False) == []


# ---------------------------------------------------------------------------
# the check, on the box
# ---------------------------------------------------------------------------

def test_a_k6_box_checks_its_path_then_asks_with_the_path_alone(k3k6, monkeypatch, tmp_path):
    launched = tmp_path / "launched.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(launched))
    assert run_k(k3k6, "u") == 0
    chk_file = k3k6["root"] / "u" / "settings-check-k6.json"
    chk = json.loads(chk_file.read_text())
    assert chk["pass"] and chk["mismatches"] == [] and chk["path"] == 6
    assert chk["prompt_penalty"]["verdict"] == "not counted"
    assert all(a["thinking"] for a in chk["answers"])
    # the server: our flags, the runtime's two variables, --jinja, no sampling flag
    got = json.loads(launched.read_text().splitlines()[-1])
    assert got["env"] == {"LOOKAHEAD_MOE_ROUTER_NORM": "1", "GGML_CPU_DISABLE_FUSION": "1"}
    assert "--jinja" in got["argv"] and not {"--temp", "--top-k", "--presence-penalty",
                                             "--reasoning-format"} & set(got["argv"])
    # the check's two questions, then the benchmark's: each its path, nothing that samples
    chk_reqs, bench = asked(k3k6)
    assert len(chk_reqs) == 2 and all(sorted(r) == ["expert_path", "max_tokens", "messages"]
                                      for r in chk_reqs)
    assert bench and all(sorted(r) == ["expert_path", "max_tokens", "messages", "seed"]
                         and r["expert_path"] == 6 and r["max_tokens"] == 81920 for r in bench)
    # the bundle records it all: the check, the file's and the server's values
    b = rb.read(bundle_at(k3k6, "u"))
    srv = b["setup"]["server"]
    assert srv["runtime"] == RT and (srv["build"], srv["commit"]) == (7000, "3ad38b7cc")
    assert srv["paths_check"]["pass"] and srv["paths_check"]["server"]["startup"]["6"]["line"] \
        .startswith("expert path k6: temp = 0.800")
    assert b["setup"]["tasks"][TASK]["sampling"] == {"expert_path": 6}
    assert b["setup"]["tasks"][TASK]["family"] == "the file's expert path k6"
    assert b["setup"]["card_checks"][0]["extra"] == {"expert_path": 6}
    # the import takes it onto step2388's · thinking row, registered with its runtime
    assert imf.runtime_problem(b["setup"], {"id": STEP, "runtime": RT}) == ""
    assert ir.main([str(bundle_at(k3k6, "u")), "--by", "masein", "--register",
                    "step2388 (k3/k6)", "--file-sha256", k3k6["sha"]]) == 0
    rec = served.get(STEP)
    assert rec["runtime"] == RT and "runtime k3k6-2611" in rec["how"]


def test_a_k3_box_asks_its_thinking_off_set_on_k3_with_no_thinking(k3k6):
    assert run_k(k3k6, "u3", thinking="off") == 0
    chk = json.loads((k3k6["root"] / "u3" / "settings-check-k3.json").read_text())
    assert chk["pass"] and chk["prompt_penalty"] is None     # k3 has no presence penalty
    assert not any(a["thinking"] for a in chk["answers"])
    _, bench = asked(k3k6)
    assert bench and all(r["expert_path"] == 3 and r["max_tokens"] == 16384 for r in bench)


@pytest.mark.parametrize("bad, words", [
    ({"FAKE_LLAMA_PATHS_BAD": "6.presence_penalty=0.5"},
     "the server's k6 (/props): presence penalty 0.5, not 0.75"),
    ({"FAKE_LLAMA_PATHS_BAD": "6.temp=0.6"}, "the server's start-up line for k6: temperature 0.6"),
    ({"FAKE_LLAMA_PATHS_LOG": "0"}, "the server's start-up lines name no expert path k6"),
    ({"FAKE_LLAMA_PROMPT_PENALTY": "1"}, "on k6 the presence penalty counts the prompt's tokens"),
    ({"FAKE_LLAMA_THINK": "never"}, "question 1 on k6: no thinking block, where k6 thinks"),
])
def test_a_setting_that_doesnt_match_stops_the_box_before_a_question(k3k6, monkeypatch, bad,
                                                                     words):
    for k, v in bad.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(SystemExit, match=re.escape(words)):
        run_k(k3k6, "bad")
    _, bench = asked(k3k6)
    assert bench == []                                      # nothing of the benchmark asked
    chk = json.loads((k3k6["root"] / "bad" / "settings-check-k6.json").read_text())
    assert not chk["pass"] and any(words in m for m in chk["mismatches"])


def test_a_file_storing_other_values_stops_the_box(k3k6, tmp_path):
    other = write_gguf(tmp_path / "other" / NAME, k6__sampling__presence_penalty=0.5,
                       default=6)
    with pytest.raises(SystemExit) as e:
        run_k(k3k6, "f", gguf=other)
    assert "the file's k6: presence penalty 0.5" in str(e.value)
    assert "the file's default path (teraformer.paths.default) is 6, not 3" in str(e.value)


def test_the_check_alone_asks_no_benchmark_and_leaves_no_bundle(k3k6, capsys):
    assert run_k(k3k6, "chk", "--check", thinking="off", only=False) == 0
    out = k3k6["root"] / "chk"
    assert (out / "settings-check-k3.json").exists()
    assert not list(out.glob("*.tar.gz")) and not (out / "progress.json").exists()
    chk_reqs, bench = asked(k3k6)
    assert len(chk_reqs) == 2 and bench == []
    said = (out / rg.OWN_LOG).read_text()
    assert "settings check, expert path k3: pass" in said
    assert "the settings check of expert path k3 passes" in said


def test_the_check_is_for_a_runtime_of_paths_and_asks_nothing_else(tmp_path, capsys):
    for more in (["--runtime", "lda-2610", "--as", "served/step8804-lda-2610"],
                 ["--runtime", RT, "--as", STEP, "--only", GPQA]):
        with pytest.raises(SystemExit):
            rg.main(["--gguf", str(tmp_path / NAME), "--server", str(tmp_path / "t.tgz"),
                     "--thinking", "on", "--check", "--out", str(tmp_path / "o"), *more])
    err = capsys.readouterr().err
    assert "--check: the settings check of a runtime of expert paths (--runtime k3k6-2611)" in err
    assert "--check asks no benchmark" in err


def test_the_import_refuses_a_bundle_whose_check_didnt_pass(k3k6):
    assert run_k(k3k6, "u") == 0
    setup = rb.read(bundle_at(k3k6, "u"))["setup"]
    rec = {"id": STEP, "runtime": RT}

    def problem(**chk) -> str:
        s = json.loads(json.dumps(setup))
        if chk.get("drop"):
            s["server"].pop("paths_check")
        else:
            s["server"]["paths_check"].update(chk)
        return imf.runtime_problem(s, rec)
    assert "its settings check (none in the bundle)" in problem(drop=True)
    assert "a settings check that passed (the server's k6 (/props): top-k 40, not 20)" in problem(
        **{"pass": False, "mismatches": ["the server's k6 (/props): top-k 40, not 20"]})
    assert "its settings check on expert path k6 (the bundle's was k3)" in problem(path=3)
    assert "the file's expert path k6 as temperature 0.8" in problem(
        file={"paths": {"6": {**fb.PATHS[6], "presence_penalty": 0.5}}})


# ---------------------------------------------------------------------------
# plan U
# ---------------------------------------------------------------------------

def test_plan_u_is_the_whitepapers_set_on_k6_and_its_thinking_off_set_on_k3():
    plan = fx.PLANS["U"]
    assert len(plan) == 37 and fx.PLAN_RUNTIME["U"] == RT
    assert fx.PLAN_GPUS["U"] == fx.PLAN_GPUS["L"]                     # the sm_120 tarball's
    assert fx.PLAN_PINS["U"] == {NAME: SHA}
    # each box one path: k3 and k6 never on the same box
    assert all(len({s[0] for s in steps}) == 1 for steps in plan.values())
    # every question of each benchmark once, in its mode: every shard of n there
    have: dict = {}
    for steps in plan.values():
        for th, tasks, shard, _ in steps:
            for t in tasks:
                have.setdefault((th, t), []).append(shard)
    for (th, t), shards in have.items():
        n = fb.parse_shard(shards[0])[1] if shards[0] else 1
        assert sorted(shards) == sorted(f"{i}/{n}" if n > 1 else "" for i in range(1, n + 1)), t
    on = {t for th, t in have if th == "on"}
    off = {t for th, t in have if th == "off"}
    assert on == set(fb.TASKS) and off == set(fb.RUNTIMES[RT]["off_for"])
    # step8804's measured paces: no box over about 5¼ hours
    lines = fx.listing("U")
    assert lines[-1] == "37 boxes a build · 172.0 box-hours a build · the longest 5.2 h"
    assert all("(step8804's measured pace)" in ln for ln in lines if ln.startswith("  "))


def test_a_plan_u_line_runs_its_path_and_pins_step2388s_file():
    a = fx.parser().parse_args(["U35", "--as", STEP, "--gguf", f"hf://me/p@c/{NAME}",
                                "--server", "hf://me/p@c/llama-server-k3k6-3ad38b7.tar.gz",
                                "--price", "0.45"])
    first, second = fx.box_of("U35")
    argv = fx.argv_of("U35", 1, first, a)
    assert argv[argv.index("--runtime") + 1] == RT and argv[argv.index("--thinking") + 1] == "off"
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "--only"] == [GPQA, MATH]
    assert argv[argv.index("--flags") + 1] == ""                       # the runtime's own
    assert fx.pins_of(a, "U35") == [f"{NAME}={SHA}"]
    k6 = fx.argv_of("U1", 1, fx.box_of("U1")[0], a)
    assert k6[k6.index("--thinking") + 1] == "on" and k6[k6.index("--shard") + 1] == "1/15"


def test_the_runbook_gives_the_check_and_a_box_end_to_end():
    text = DOC.read_text()
    at = text.index("# step2388 on plan U (31.1)")
    part = text[at:]
    assert "scripts/teraformer-expert-paths-server.py --skip-parity" in part
    assert "--runtime k3k6-2611 --thinking off --check" in part
    assert "python scripts/frontier_box.py U35 --as served/step2388-k3k6-2611" in part
    assert SHA in part
