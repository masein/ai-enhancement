"""17c, six fixes found on the deployed board (0a81fc2) and on the server:
1. build_llama_tarball.sh read git's changes through head under pipefail: on a
   tree with many untracked files git died of SIGPIPE and the script exited
   141 before a word;
2.–3. Mobile-MMLU's card is the full set's, and a full run's time is the full
   set's (it was filed as Pro's, and the full set's card had no "A run here");
6. the Frontier suite's benchmarks each have what their card says;
7. the tarball's link leaves libcuda's symbols to the box (the build container
   has no driver), and it is built for no particular CPU.
The page's side of 2–6 is tests/test_17c_board_six_browser.py. Nothing runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import report_lm_eval as report

REPO = Path(__file__).resolve().parents[1]


def test_1_the_tarball_script_reads_a_tree_with_many_untracked_files(tmp_path):
    """the server's checkout: exit 141, no output, before the first line"""
    src = tmp_path / "llama.cpp"
    src.mkdir()
    (src / "CMakeLists.txt").write_text("project(x)\n")
    git = ["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "CMakeLists.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "x"], check=True)
    # more than a pipe holds (64 KiB) of `git status --porcelain`
    for i in range(6000):
        (src / f"an-untracked-file-of-the-build-tree-{i:05d}.o").write_bytes(b"")
    # docker, stood in for: `info` answers, `run` stops the build here
    stub = tmp_path / "docker"
    stub.write_text("#!/bin/sh\n[ \"$1\" = info ] && exit 0\necho \"stub docker: $1\" >&2\nexit 3\n")
    stub.chmod(0o755)
    r = subprocess.run(["bash", str(REPO / "scripts" / "build_llama_tarball.sh"), str(src),
                        str(tmp_path / "out.tar.gz")], capture_output=True, text=True,
                       env={**os.environ, "DOCKER": str(stub)})
    assert r.returncode == 3, (r.returncode, r.stderr[-500:])
    assert f"note: {src} has changes not committed — recorded in VERSION" in r.stderr
    assert "building llama-server at " in r.stderr and "stub docker: run" in r.stderr


def test_2_a_full_mobile_mmlu_runs_time_is_the_full_sets():
    """its results give Pro's cell, and its time was Pro's"""
    def run(model, date, secs, full):
        return {"model": model, "date": date, "eval_seconds": secs, "mmf_run": full,
                "tasks": {report.MMP_TASK: {"alias": report.MMP_TASK}},
                "n_shot": {}, "n_samples": {}}
    got = report.merge_runs([run("a", "2026-10-01", 300, False), run("b", "2026-10-02", 900, True)])
    assert got["a"]["task_secs"] == {report.MMP_TASK: 300}
    assert got["b"]["task_secs"] == {report.MMF_TASK: 900}


def test_6_each_frontier_benchmark_says_its_set_source_and_licence():
    import frontier as fb
    meta = report.frontier_bench_meta()
    assert set(meta) == set(fb.BENCH)
    for t, m in meta.items():
        src = fb.BENCH[t]["source"]
        assert m["n"] == fb.BENCH[t]["n"] and m["licence"] == src["licence"], t
        assert m["url"] == (f"https://huggingface.co/datasets/{src['hf']}" if src.get("hf")
                            else f"https://github.com/{src['github']}"), t
    assert meta["mmlupro_tiger"]["n"] == 12032 and meta["hle_text_cais"]["gated"] is True
    assert meta["hle_text_cais"]["grader"] == "o3-mini-2025-01-31"


def test_7_the_tarball_links_without_a_driver_and_for_no_particular_cpu():
    """on the server: "libcuda.so.1, needed by libggml-cuda.so, not found" at the
    last link; and a native build can die of "Illegal instruction" on a box"""
    script = (REPO / "scripts" / "build_llama_tarball.sh").read_text()
    cmake = script[script.index("cmake -S . -B build-tarball"):]
    cmake = cmake[:cmake.index(">/dev/null")]
    assert "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined" in cmake
    assert "-DGGML_NATIVE=OFF" in cmake
    # and nothing in it reads through a pipe that can close early under pipefail
    assert "| head" not in script
    assert "{ ldd " in script and "|| true; } | awk" in script
