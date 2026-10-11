"""20: step4000 against Q4 and BF16 on rented boxes, with the server out of
reach (docs/prompts/phase-20-step4000-benchmarks-off-server.md). A build
directory run as it is (--server-dir), the lda runtime (--runtime lda-2610),
plan L, the scores on the box (score_bundle.py), the upload to the private
repository (box_upload.py) and the import of a folder of bundles. The build
is the stand-in llama-server (tests/fixtures/fake_llama_server.py), GPQA's
questions are invented, Hugging Face is a stand-in: nothing is fetched, no
model runs, no paid API is called."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

import box_upload as bu
import frontier as fb
import frontier_box as fx
import import_frontier as imf
import import_remote as ir
import remote_bundle as rb
import remote_gguf as rg
import score_bundle as sb
from service import config, runner, served
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import (FIX, N, RUNS, TASK, box, expected, free_port,  # noqa: F401
                              invented, requests)

RT = "lda-2610"
STEP = "served/step4000-lda-2610"
ROW = "served__step4000-lda-2610__thinking"
COMMIT = "1234567890abcdef1234567890abcdef12345678"


def build_dir(root: Path, archs: str | None = "120") -> Path:
    """a llama.cpp build directory as the team leaves one on a box: bin/ with
    llama-server and its libraries, its CMakeCache.txt, inside a checkout of
    the lda branch"""
    d = root / "llama.cpp" / "build"
    (d / "bin").mkdir(parents=True)
    exe = d / "bin" / "llama-server"
    exe.write_bytes((FIX / "fake_llama_server.py").read_bytes())
    exe.chmod(0o755)
    (d / "bin" / "libggml-cuda.so").write_bytes(b"\x7fELF not really")
    # the lda branch reads its router norm in libllama.so
    (d / "bin" / "libllama.so").write_bytes(b"\x7fELF not really either LOOKAHEAD_MOE_ROUTER_NORM")
    if archs is not None:
        (d / "CMakeCache.txt").write_text(f"CMAKE_CUDA_ARCHITECTURES:STRING={archs}\n")
    g = root / "llama.cpp" / ".git"
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "HEAD").write_text("ref: refs/heads/lda\n")
    (g / "refs" / "heads" / "lda").write_text(COMMIT + "\n")
    return d


def run_dir(b: dict, out: str, build: Path, *more: str, served_as: str = STEP,
            runtime: bool = True) -> int:
    """remote_gguf.py on the "box" with --server-dir — the server's config put back after"""
    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR"]}
    lock = runner.LOCK
    env = {v: os.environ.get(v) for v in rg.ENV.values()}
    try:
        return rg.main(["--as", served_as, "--gguf", str(b["gguf"]), "--server-dir", str(build),
                        "--based-on", "Qwen/Qwen3.6-35B-A3B", "--thinking", "on",
                        "--slots", "3", "--port", str(free_port()), "--load-timeout", "60",
                        *(["--runtime", RT] if runtime else ["--flags", "--flash-attn on"]),
                        "--only", TASK, "--out", str(b["root"] / out), *more])
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bundle_at(b: dict, out: str, served_as: str = STEP,
              shard: tuple[int, int] | None = None) -> Path:
    return b["root"] / out / rb.bundle_name("frontier", served_as, True, shard,
                                            parts=[fb.BENCH[TASK]["short"]])


def lda_tarball(root: Path, archs: str = "120", reads: bool = True) -> Path:
    """the lda build packed as build_llama_tarball.sh packs one: bin/, lib/
    with libllama.so (that reads the router norm), VERSION"""
    import io
    import tarfile
    tb = root / "llama-server-lda-a4a76a7.tar.gz"
    with tarfile.open(tb, "w:gz") as tar:
        def add(name: str, data: bytes, mode: int = 0o644) -> None:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), mode
            tar.addfile(ti, io.BytesIO(data))
        add("llama/bin/llama-server", (FIX / "fake_llama_server.py").read_bytes(), 0o755)
        add("llama/lib/libllama.so", b"\x7fELF " + (b"LOOKAHEAD_MOE_ROUTER_NORM" if reads
                                                    else b"nothing of it"))
        add("llama/lib/libggml-cuda.so", b"\x7fELF not really")
        add("llama/VERSION", f"commit a4a76a7\nbuild 9954\ncuda_archs {archs}\n"
                             "cmake_args -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF\n"
                             .encode())
    return tb


def run_tb(b: dict, out: str, tb: Path, *more: str, served_as: str = STEP) -> int:
    """remote_gguf.py from a tarball with the runtime — the server's config put back after"""
    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR"]}
    lock, env = runner.LOCK, {v: os.environ.get(v) for v in rg.ENV.values()}
    try:
        return rg.main(["--as", served_as, "--gguf", str(b["gguf"]), "--server", str(tb),
                        "--based-on", "Qwen/Qwen3.6-35B-A3B", "--thinking", "on", "--slots", "3",
                        "--port", str(free_port()), "--load-timeout", "60", "--runtime", RT,
                        "--only", TASK, "--out", str(b["root"] / out), *more])
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def card(monkeypatch):
    """the box's card: a 5090, sm_120"""
    monkeypatch.setattr(rg, "compute_cap", lambda: 120)


# ---------------------------------------------------------------------------
# 1–2: a build directory, run as it is, with the lda runtime
# ---------------------------------------------------------------------------

def test_a_build_directory_runs_with_the_runtime_and_is_recorded(box, card, monkeypatch, tmp_path):  # noqa: F811
    launched = tmp_path / "launched.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(launched))
    build = build_dir(box["root"])
    assert run_dir(box, "run", build) == 0
    b = rb.read(bundle_at(box, "run"))
    setup, srv = b["setup"], b["setup"]["server"]
    # the runtime by name, every flag and value recorded
    assert setup["runtime"] == RT and srv["runtime"] == RT
    assert srv["flags"] == fb.runtime_flags(RT)
    assert srv["flags"][:6] == ["-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on"]
    assert srv["flags"][srv["flags"].index("--min-p") + 1] == "0.0"
    assert srv["env"] == {"LOOKAHEAD_MOE_ROUTER_NORM": "1"}
    # the build: its version, the binary's sha256 and its libraries', its branch and commit
    assert (srv["build"], srv["commit"]) == (6543, "abc1234")
    assert srv["binary_sha256"] == rb.sha256_file(build / "bin" / "llama-server")
    assert srv["tarball_sha256"] is None and len(srv["libs_sha256"]) == 64
    d = srv["dir"]
    assert set(d["libs"]) == {"libggml-cuda.so", "libllama.so"}
    assert d["libs"]["libllama.so"] == rb.sha256_file(build / "bin" / "libllama.so")
    assert (d["git"]["branch"], d["git"]["commit"]) == ("lda", COMMIT)
    assert d["cuda_archs"] == ["120"] and d["compute_cap"] == 120
    assert d["devices"] == ["CUDA0: NVIDIA GeForce RTX 5090 (32109 MiB, 31588 MiB free)"]
    # llama-server got the runtime's flags and its variable, with its own folder's libraries
    got = json.loads(launched.read_text().splitlines()[-1])
    assert got["argv"][got["argv"].index("--repeat-last-n") + 1] == "64"
    assert got["env"]["LOOKAHEAD_MOE_ROUTER_NORM"] == "1"
    # each request carries the runtime's sampling — min-p too
    reqs = requests(box)
    assert len(reqs) == N * RUNS
    assert all(r["min_p"] == 0.0 and r["repeat_penalty"] == 1.0 and r["repeat_last_n"] == 64
               and r["temperature"] == 1.0 and r["top_k"] == 20 and r["top_p"] == 0.95
               and r["presence_penalty"] == 1.5 and r["max_tokens"] == 81920
               and r["chat_template_kwargs"] == {"enable_thinking": True} for r in reqs)
    assert setup["tasks"][TASK]["sampling"] == {**fb.RUNTIMES[RT]["sampling"]}
    # and the step's line said it, the repeat penalty among it
    log = (box["root"] / "run" / rg.OWN_LOG).read_text()
    assert "runtime lda-2610" in log and "repeat penalty 1.0" in log and "min-p 0.0" in log
    assert f"git lda at {COMMIT[:12]}" in log
    # nothing of the build in the bundle
    assert not any(n.endswith(".so") or "llama-server" == n.rsplit("/", 1)[-1] for n in b["files"])
    # a library rebuilt under answers already given: refused, not mixed
    (build / "bin" / "libllama.so").write_bytes(b"rebuilt, LOOKAHEAD_MOE_ROUTER_NORM")
    with pytest.raises(SystemExit, match="libs_sha256"):
        run_dir(box, "run", build)


def test_a_build_that_isnt_there_or_doesnt_run_here_is_refused(tmp_path, monkeypatch, card):
    with pytest.raises(SystemExit, match="no bin/llama-server there"):
        rg.server_dir(str(tmp_path))
    d = build_dir(tmp_path)
    srv = rg.server_dir(str(d))
    said: list[str] = []
    got = rg.check_runs_here(srv, rg.server_env(srv, {}), said.append)
    assert got["cuda_archs"] == ["120"] and got["devices"] and "runs here" in said[0]
    # a library missing on this box
    monkeypatch.setenv("FAKE_LLAMA_BROKEN", "1")
    with pytest.raises(SystemExit, match="doesn't run here.*libcudart"):
        rg.check_runs_here(srv, rg.server_env(srv, {}), said.append)
    monkeypatch.delenv("FAKE_LLAMA_BROKEN")
    # no CUDA device it can use: a driver older than its CUDA
    monkeypatch.setenv("FAKE_LLAMA_DEVICES", "none")
    with pytest.raises(SystemExit, match="sees no CUDA device.*insufficient"):
        rg.check_runs_here(srv, rg.server_env(srv, {}), said.append)
    monkeypatch.delenv("FAKE_LLAMA_DEVICES")
    # built for other cards only
    (d / "CMakeCache.txt").write_text("CMAKE_CUDA_ARCHITECTURES:STRING=86-real;89-real\n")
    with pytest.raises(SystemExit, match="won't run here.*CMAKE_CUDA_ARCHITECTURES=120"):
        rg.check_runs_here(srv, rg.server_env(srv, {}), said.append)
    # a packed build (build_llama_tarball.sh's layout) says its architectures in VERSION
    (d / "CMakeCache.txt").unlink()
    (d / "VERSION").write_text("commit abc\ncuda_archs 80;86;89;90;120\n")
    assert rg.check_runs_here(srv, rg.server_env(srv, {}), said.append)["cuda_archs"] == [
        "80", "86", "89", "90", "120"]
    # CUDA's runtime and cuBLAS: the runner image's, after the build's own folders —
    # unless the build carries its own
    monkeypatch.setattr(rg, "cuda_lib_dirs", lambda: ["/opt/py/nvidia/cublas/lib"])
    got = rg.server_dir(str(d))
    here = str((d / "bin").resolve())
    assert got["libs"] == [here, "/opt/py/nvidia/cublas/lib"]
    (d / "bin" / "libcudart.so.12").write_bytes(b"\x7fELF CUDA's")
    assert rg.server_dir(str(d))["libs"] == [here]
    # no library beside it: a binary copied alone
    for so in (d / "bin").glob("*.so*"):
        so.unlink()
    with pytest.raises(SystemExit, match="no shared library"):
        rg.server_dir(str(d))


def test_which_architectures_run_on_a_card():
    assert rg.archs_run_on(["120"], 120) is True
    assert rg.archs_run_on(["86"], 120) is True               # PTX the driver compiles
    assert rg.archs_run_on(["86-real", "89-real"], 120) is False
    assert rg.archs_run_on(["120a-real"], 120) is True
    assert rg.archs_run_on(["90a"], 120) is False
    assert rg.archs_run_on(["native"], 120) is None


@pytest.mark.parametrize("more, words", [
    (["--thinking", "off"], "thinking on only"),
    (["--flags", "--spec-type draft-mtp"], "--spec-type"),
    (["--flags", "--temp 0.7"], "--temp"),
    (["--flags=--moe-stream"], "--moe-stream"),
    (["--flags", "--reasoning-budget 0"], "--reasoning-budget"),
    (["--env", "LOOKAHEAD_MOE_ROUTER_NORM=0"], "LOOKAHEAD_MOE_ROUTER_NORM"),
    (["--env", "LLAMA_ARG_REASONING_BUDGET=0"], "LLAMA_ARG_REASONING_BUDGET"),
])
def test_the_runtime_refuses_what_would_change_it(tmp_path, capsys, more, words):
    with pytest.raises(SystemExit) as e:
        rg.main(["--as", STEP, "--gguf", str(tmp_path / "x.gguf"), "--server-dir", str(tmp_path),
                 "--runtime", RT, "--thinking", "on", "--only", TASK,
                 "--out", str(tmp_path / "o"), *more])
    assert e.value.code == 2 and words in capsys.readouterr().err
    assert not (tmp_path / "o").exists()                    # refused before anything


def test_the_runtimes_sampling_goes_in_each_request():
    rec = {"id": STEP, "name": "step4000", "based_on": "", "runtime": RT,
           "base_url": "http://127.0.0.1:9/v1", "pin": {}}
    assert sf.settings(rec, TASK, True) == {
        "max_tokens": 81920, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "presence_penalty": 1.5, "min_p": 0.0, "repeat_penalty": 1.0, "repeat_last_n": 64,
        "chat_template_kwargs": {"enable_thinking": True}}
    # a model without one: the card's, as before
    assert "min_p" not in sf.settings({**rec, "runtime": "", "based_on": "Qwen/Qwen3.6-35B-A3B"},
                                      TASK, True)
    # the runtime's flags and variable go out in an export as they are
    import export_safe as es
    assert es.flags(fb.runtime_flags(RT)) == (fb.runtime_flags(RT), 0)
    assert es.env({"LOOKAHEAD_MOE_ROUTER_NORM": "1"}) == ({"LOOKAHEAD_MOE_ROUTER_NORM": "1"}, 0)


# ---------------------------------------------------------------------------
# 3: plan L
# ---------------------------------------------------------------------------

def test_plan_L_is_thinking_on_every_benchmark_whole_and_no_box_over_6_hours():  # noqa: F811
    L = fx.PLANS["L"]
    assert len(L) == 21 and list(L)[0] == "L1" and list(L)[-1] == "L21"
    assert max(sum(fx.hours(s) for s in st) for st in L.values()) <= 6.0
    assert all(s[0] == "on" for st in L.values() for s in st)
    cover: dict[str, list] = {}
    for st in L.values():
        for _, tasks, shard, _ in st:
            for t in tasks:
                cover.setdefault(t, []).append(fb.parse_shard(shard))
    assert set(cover) == set(fb.TASKS)
    for t, shards in cover.items():
        if shards == [None]:
            continue
        ns = {n for _, n in shards}
        assert len(ns) == 1 and sorted(i for i, _ in shards) == list(range(1, ns.pop() + 1)), t
    assert fx.listing("L")[-1].startswith("21 boxes a build · 114.1 box-hours")


def test_the_runtime_from_a_tarball_as_any_build(box, card, monkeypatch, tmp_path):  # noqa: F811
    """the main path: the lda tarball fetched as G2's, the runtime by name,
    each file pinned by its sha256"""
    launched = tmp_path / "launched.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(launched))
    tb = lda_tarball(tmp_path)
    pins = ["--pin", f"{tb.name}={rb.sha256_file(tb)}",
            "--pin", f"{box['gguf'].name}={box['sha']}"]
    assert run_tb(box, "tb", tb, *pins) == 0
    srv = rb.read(bundle_at(box, "tb"))["setup"]["server"]
    assert srv["runtime"] == RT and srv["flags"] == fb.runtime_flags(RT)
    assert srv["tarball_sha256"] == rb.sha256_file(tb) and "dir" not in srv
    assert json.loads(launched.read_text().splitlines()[-1])["env"][
        "LOOKAHEAD_MOE_ROUTER_NORM"] == "1"
    assert all(r["min_p"] == 0.0 and r["repeat_last_n"] == 64 for r in requests(box))


def test_a_file_that_isnt_the_pinned_one_is_refused(box, card, tmp_path):  # noqa: F811
    tb = lda_tarball(tmp_path)
    with pytest.raises(SystemExit, match=f"{tb.name} here has sha256 .*not the 000000"):
        run_tb(box, "p1", tb, "--pin", f"{tb.name}={'0' * 64}")
    with pytest.raises(SystemExit, match=f"{box['gguf'].name} here has sha256"):
        run_tb(box, "p2", tb, "--pin", f"{box['gguf'].name}={'1' * 64}")
    with pytest.raises(SystemExit, match="NAME=SHA256"):
        run_tb(box, "p3", tb, "--pin", "step4000.gguf=abc")
    assert not requests(box)                        # nothing was asked


def test_a_build_that_doesnt_read_the_runtimes_variable_is_refused(box, card, tmp_path):  # noqa: F811
    tb = lda_tarball(tmp_path, reads=False)
    with pytest.raises(SystemExit, match="sets LOOKAHEAD_MOE_ROUTER_NORM, and this build .* "
                                         "never reads it"):
        run_tb(box, "nr", tb)
    assert not requests(box)


def test_a_tarball_for_other_cards_is_refused_before_it_loads(box, card, tmp_path):  # noqa: F811
    tb = lda_tarball(tmp_path, archs="86-real;89-real")
    with pytest.raises(SystemExit, match="built for CUDA architectures 86-real;89-real .* sm_120"):
        run_tb(box, "ar", tb)
    assert not requests(box)


def test_plan_L_checks_the_card_first(monkeypatch, capsys):
    ok = {"name": "NVIDIA GeForce RTX 5090", "names": ["NVIDIA GeForce RTX 5090"]}
    pro = {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
           "names": ["NVIDIA RTX PRO 6000 Blackwell Server Edition"]}
    assert fx.gpu_refused("L3", ok) == "" and fx.gpu_refused("L3", pro) == ""
    assert "RTX 4090 — plan L runs on an RTX 5090 or an RTX PRO 6000 only" in fx.gpu_refused(
        "L3", {"names": ["NVIDIA GeForce RTX 4090"]})
    assert "H100" in fx.gpu_refused("L3", {"names": ["NVIDIA GeForce RTX 5090",
                                                     "NVIDIA H100 80GB HBM3"]})
    assert "no GPU could be read" in fx.gpu_refused("L3", {"name": None})
    assert fx.gpu_refused("A3", {"names": ["NVIDIA GeForce RTX 4090"]}) == ""   # plan A: any
    # the box line on another card: one line, before anything is fetched or run
    import remote_bundle
    monkeypatch.setattr(remote_bundle, "gpu_info", lambda: {"names": ["NVIDIA A100-SXM4-80GB"]})
    monkeypatch.setattr(fx.subprocess, "run", lambda *a, **k: pytest.fail("a step ran"))
    with pytest.raises(SystemExit) as e:
        fx.main(["L3", "--as", STEP, "--gguf", "hf://me/p/step4000.gguf",
                 "--server", "hf://me/p/llama-server-lda-a4a76a7.tar.gz"])
    assert "this box has NVIDIA A100-SXM4-80GB — plan L runs on an RTX 5090 or an RTX PRO " \
           "6000 only" in str(e.value.code)


def test_the_build_script_takes_cmake_args_and_leaves_the_web_page_out(tmp_path):
    import subprocess
    import tarfile
    from test_17c_board_six import CONTAINER
    src = tmp_path / "llama.cpp"
    src.mkdir()
    (src / "CMakeLists.txt").write_text("project(x)\n")
    git = ["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "CMakeLists.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "x"], check=True)
    stub = tmp_path / "docker"
    log = tmp_path / "docker.log"
    stub.write_text(CONTAINER.replace("#!/bin/sh\n", f'#!/bin/sh\necho "$*" >> {log}\n', 1))
    stub.chmod(0o755)

    def build(**env) -> str:
        out = tmp_path / "t.tar.gz"
        r = subprocess.run(["bash", str(FIX.parents[1] / "scripts" / "build_llama_tarball.sh"),
                            str(src), str(out)], capture_output=True, text=True,
                           env={**os.environ, "DOCKER": str(stub), **env})
        assert r.returncode == 0, r.stderr[-800:]
        with tarfile.open(out) as tar:
            return tar.extractfile("llama/VERSION").read().decode()
    v = build(LLAMA_CUDA_ARCHS="120", LLAMA_CMAKE_ARGS="-DGGML_CUDA_FA_ALL_QUANTS=ON")
    assert "cuda_archs 120" in v and "web_ui no" in v
    assert "cmake_args -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF " \
           "-DGGML_CUDA_FA_ALL_QUANTS=ON" in v
    assert "EXTRA=-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON" \
        in log.read_text()
    v = build(LLAMA_WEB_UI="1")
    assert "web_ui yes" in v and "cmake_args none" in v
    script = (FIX.parents[1] / "scripts" / "build_llama_tarball.sh").read_text()
    assert '-DCMAKE_CUDA_ARCHITECTURES="$ARCHS" $EXTRA >/dev/null' in script


def test_a_plan_L_line_runs_the_runtime_from_the_tarball_its_files_pinned():
    a = fx.parser().parse_args(["L13", "--as", STEP,
                                "--gguf", "hf://me/evalboard-private/step4000.gguf",
                                "--server", "hf://me/evalboard-private/"
                                            "llama-server-lda-a4a76a7.tar.gz"])
    argv = fx.argv_of("L13", 1, fx.box_of("L13")[0], a)
    assert argv[argv.index("--runtime") + 1] == RT and "--server-dir" not in argv
    assert argv[argv.index("--server") + 1].endswith("llama-server-lda-a4a76a7.tar.gz")
    assert argv[argv.index("--flags") + 1] == ""
    pins = [argv[k + 1] for k, x in enumerate(argv) if x == "--pin"]
    assert pins == ["llama-server-lda-a4a76a7.tar.gz="
                    "5b538290644d67139c6c996fdaa3e04cce82fd0b343cce66a4ff234648afb618",
                    "step4000.gguf="
                    "630c408a3836f82afa1f79629b06d9390d3f736f6140791baa42e180a640565e"]
    # a file of another name (the Q4's) has no pin
    a = fx.parser().parse_args(["L13", "--as", "served/Qwen3.6-35B-A3B-Q4-lda-2610",
                                "--gguf", "hf://me/p/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                                "--server", "hf://me/p/llama-server-lda-a4a76a7.tar.gz"])
    argv = fx.argv_of("L13", 1, fx.box_of("L13")[0], a)
    assert [argv[k + 1].split("=")[0] for k, x in enumerate(argv) if x == "--pin"] == [
        "llama-server-lda-a4a76a7.tar.gz"]


def test_a_plan_L_line_runs_the_runtime_and_the_build_directory():
    a = fx.parser().parse_args(["L13", "--as", STEP, "--gguf", "/workspace/s/step4000.gguf",
                                "--server-dir", "/workspace/llama.cpp/build"])
    argv = fx.argv_of("L13", 1, fx.box_of("L13")[0], a)
    assert argv[argv.index("--runtime") + 1] == RT
    assert argv[argv.index("--server-dir") + 1] == "/workspace/llama.cpp/build"
    assert "--server" not in argv and argv[argv.index("--flags") + 1] == ""
    assert argv.count("--only") == 2 and argv[argv.index("--shard") + 1] == "1/4"
    assert argv[argv.index("--out") + 1] == "/workspace/step4000-lda-2610/L13-1"
    # plan A as before: our flags, no runtime
    a = fx.parser().parse_args(["A5", "--as", "served/x", "--gguf", "g", "--server", "s"])
    argv = fx.argv_of("A5", 1, fx.box_of("A5")[0], a)
    assert "--runtime" not in argv and argv[argv.index("--flags") + 1] == fx.FLAGS
    with pytest.raises(SystemExit):
        fx.main(["L1", "--as", STEP, "--gguf", "g", "--server", "s", "--server-dir", "d"])


# ---------------------------------------------------------------------------
# 4: scores on the box
# ---------------------------------------------------------------------------

def test_scores_on_the_box_are_the_boards_and_side_by_side(box, card, capsys):  # noqa: F811
    build = build_dir(box["root"])
    other = "served/Qwen3.6-35B-A3B-Q4-lda-2610"
    assert run_dir(box, "a", build) == 0
    assert run_dir(box, "b", build, served_as=other) == 0
    before = sorted(p for p in config.OUT_DIR.rglob("*")) if config.OUT_DIR.exists() else []
    bench = config.BENCH_ROOT
    capsys.readouterr()
    assert sb.main([str(bundle_at(box, "a")), str(bundle_at(box, "b", other))]) == 0
    out = capsys.readouterr().out
    assert (f"GPQA Diamond · step4000-lda-2610: {100 * expected():.1f}% ± " in out
            and f"{N * RUNS} answers · 0 ran out of room" in out)
    assert "GPQA Diamond, side by side:" in out
    # the file each box hashed, whole: what the import's --file-sha256 is given
    assert f"step4000-lda-2610: its GGUF {box['gguf'].name} · sha256 {box['sha']}" in out
    assert "step4000-lda-2610 − Qwen3.6-35B-A3B-Q4-lda-2610: +0.0 points" in out
    # nothing written where the board keeps its results, and its config as it was
    after = sorted(p for p in config.OUT_DIR.rglob("*")) if config.OUT_DIR.exists() else []
    assert after == before and config.BENCH_ROOT == bench


def test_ran_out_and_loops_are_counted_by_17es_rule(box, tmp_path):  # noqa: F811
    items, q, tmp = invented(), tmp_path / "q", tmp_path / "t"
    words = ["".join(random.Random(k).choice("abcdefghij") for _ in range(6)) for k in range(60)]
    rng = random.Random(7)
    answers = {}
    for k, it in enumerate(items):
        _, need = fb.prompt(TASK, it)
        for e in range(RUNS):
            r = {"id": it["id"], "epoch": e, "answer": f"ANSWER: {need['key']}", "finish": "stop"}
            if k == 0:              # cut, the last passage repeating: a loop
                r.update(answer="<think>\n" + "Let me check the options again. " * 900,
                         finish="length")
            if k == 1:              # cut, no passage repeating
                r.update(answer="<think>\n" + " ".join(rng.choice(words) for _ in range(5000)),
                         finish="length")
            answers[(it["id"], e)] = r
    got = {"answers": answers, "shards": set(), "whole": True, "bundles": ["x"], "setup": b"{}"}
    was = sb.configure(tmp, q)
    try:
        sc = sb.score(TASK, got, q, tmp, "m")
    finally:
        sb.restore(was)
    assert (sc["answers"], sc["ran_out"], sc["loops"]) == (N * RUNS, 2 * RUNS, RUNS)
    assert sc["score"] == pytest.approx((N - 2) / N)
    line = sb.line_of(TASK, "m", sc)
    assert f"{2 * RUNS} ran out of room" in line and f"{RUNS} of them ending in a loop" in line
    # a graded benchmark: its answers and its ran-out only, graded on the board later
    hle = sb.line_of("hle_text_cais", "m", {**sc, "score": None})
    assert "graded on the board later" in hle and "%" not in hle.split("·")[1]
    # MATH and OTIS: the code's score, said
    assert "before Epoch's model check" in sb.line_of("math_l5_epoch", "m", sc)


# ---------------------------------------------------------------------------
# 5: the upload, and the import of a folder
# ---------------------------------------------------------------------------

@dataclass
class Op:
    path_in_repo: str
    path_or_fileobj: str


class FakeHub:
    """Hugging Face, standing in: a repository's files, by path"""

    def __init__(self, private: bool = True):
        self.private, self.files, self.messages = private, {}, []

    def repo_info(self, repo, repo_type=None):
        return types.SimpleNamespace(private=self.private)

    def create_commit(self, repo, operations, repo_type=None, commit_message=""):
        self.messages.append(commit_message)
        for op in operations:
            self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()

    def list_repo_tree(self, repo, path_in_repo="", repo_type=None, recursive=False,
                       expand=False):
        for p, data in self.files.items():
            if p.endswith(".tar.gz"):
                yield types.SimpleNamespace(path=p, lfs={"sha256": hashlib.sha256(data).hexdigest()})
            else:
                yield types.SimpleNamespace(path=p, lfs=None, blob_id=bu.blob_sha1(data))


def workspace(root: Path) -> tuple[Path, Path]:
    """/workspace as a plan L box leaves it: a step whole, one stopped, one
    whose bundle isn't the one its progress file records"""
    build = root / "step4000-lda-2610"
    whole = build / "L13-1"
    whole.mkdir(parents=True)
    bundle = whole / "frontier-served__step4000-lda-2610-thinking-on-gpqa+math-l5-shard-1-of-4.tar.gz"
    bundle.write_bytes(b"a bundle")
    (whole / "progress.json").write_text(json.dumps({"state": "whole", "bundle": {
        "name": bundle.name, "sha256": rb.sha256_file(bundle)}}))
    (build / "L13-2").mkdir()
    (build / "L13-2" / "progress.json").write_text(json.dumps({"state": "stopped",
                                                               "why": "stopped by hand"}))
    (build / "L14-1").mkdir()
    other = build / "L14-1" / "frontier-x.tar.gz"
    other.write_bytes(b"changed since")
    (build / "L14-1" / "progress.json").write_text(json.dumps({"state": "whole", "bundle": {
        "name": other.name, "sha256": "0" * 64}}))
    (build / "L13.line.json").write_text('{"box": "L13", "state": "ended"}')
    (root / "files").mkdir()
    (root / "files" / "big.gguf").write_bytes(b"GGUF")
    return build, bundle


def test_the_upload_sends_whole_bundles_and_progress_to_a_private_repo_only(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        CommitOperationAdd=Op))
    monkeypatch.setenv("HF_TOKEN", "hf_FIXTUREwrite20box0123456789")
    build, bundle = workspace(tmp_path)
    hub = FakeHub()
    to = "hf://me/evalboard-private/runs/step4000-2610"
    assert bu.main(["--to", to, "--root", str(tmp_path)], api=hub) == 0
    base = "runs/step4000-2610/step4000-lda-2610"
    assert set(hub.files) == {f"{base}/L13-1/progress.json", f"{base}/L13-1/{bundle.name}",
                              f"{base}/L13-2/progress.json", f"{base}/L14-1/progress.json",
                              f"{base}/L13.line.json"}
    out = capsys.readouterr().out
    assert "5 of 5 files in me/evalboard-private as they are here" in out
    assert "isn't the bundle its progress file records" in out and "no bundle yet" in out
    assert "hf_FIXTURE" not in out and hub.messages == ["L13, L14: bundles and progress"]
    # a repository that isn't private: nothing goes
    hub = FakeHub(private=False)
    with pytest.raises(SystemExit, match="isn't private"):
        bu.main(["--to", to, "--root", str(tmp_path)], api=hub)
    assert hub.files == {}
    # no token: said, nothing goes
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(SystemExit, match="HF_TOKEN isn't set"):
        bu.main(["--to", to, "--root", str(tmp_path)], api=FakeHub())


def test_a_folder_of_one_builds_bundles_is_imported_and_registers_its_runtime(box, card, tmp_path):  # noqa: F811
    build = build_dir(box["root"])
    assert run_dir(box, "s1", build, "--shard", "1/2") == 0
    assert run_dir(box, "s2", build, "--shard", "2/2") == 0
    home = tmp_path / "home" / "runs" / "step4000-2610" / "step4000-lda-2610"
    for out, sh in (("s1", (1, 2)), ("s2", (2, 2))):
        (home / out).mkdir(parents=True)
        shutil.copy(bundle_at(box, out, shard=sh), home / out)
    assert ir.main([str(home), "--by", "masein", "--register", "step4000 (lda runtime)",
                    "--file-sha256", box["sha"]]) == 0
    rec = served.get(STEP)
    assert rec["name"] == "step4000 (lda runtime)" and rec["runtime"] == RT
    assert "runtime lda-2610" in rec["how"] and rec["file_sha256"]["sha256"] == box["sha"]
    res = json.loads(next(sf.task_dir(config.OUT_DIR / ROW, TASK).glob("results_*.json"))
                     .read_text())
    assert res["results"][TASK]["acc,none"] == pytest.approx(expected())
    assert res["frontier"]["sampling"]["min_p"] == 0.0
    # --register over bundles of two models: refused before anything
    assert run_dir(box, "q4", build, served_as="served/Qwen3.6-35B-A3B-Q4-lda-2610") == 0
    with pytest.raises(SystemExit):
        ir.main([str(home), str(bundle_at(box, "q4", "served/Qwen3.6-35B-A3B-Q4-lda-2610")),
                 "--by", "masein", "--register", "x", "--file-sha256", box["sha"]])


def test_a_runtimes_rows_never_mix_with_anothers(box, card):  # noqa: F811
    build = build_dir(box["root"])
    assert run_dir(box, "run", build) == 0
    b = rb.read(bundle_at(box, "run"))
    # registered without the runtime (an earlier row of the same file): refused, in words
    rec = {"id": STEP, "name": "step4000", "base_url": "", "key": "", "thinking": "auto",
           "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "", "phone": False, "gguf_path": "",
           "gguf_flags": "", "gguf_setups": [], "flags": "", "env": "LOOKAHEAD_MOE_ROUTER_NORM=1",
           "pin": {"file": box["gguf"].name, "model": box["gguf"].name},
           "file_sha256": {"sha256": box["sha"]}, "answered": [], "by": "masein", "at": 0}
    said = imf.checks(b, rec)
    assert any("a runtime's rows never mix" in x for x in said)
    # registered with it: nothing to say
    assert imf.runtime_problem(b["setup"], {**rec, "runtime": RT}) == ""
    # a box that didn't run the runtime as this board defines it
    bad = json.loads(json.dumps(b["setup"]))
    bad["server"]["env"] = {}
    assert "didn't run runtime lda-2610" in imf.runtime_problem(bad, {**rec, "runtime": RT})
    bad["runtime"] = "lda-9999"
    assert "isn't one this board knows" in imf.runtime_problem(bad, {**rec, "runtime": RT})
    # the router norm is compared as routing is
    assert any("LOOKAHEAD_MOE_ROUTER_NORM=1" in x for x in imf.launch_differs(
        {**rec, "env": ""}, b["setup"]["server"]))
