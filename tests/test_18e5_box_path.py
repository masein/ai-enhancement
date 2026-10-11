"""18e, part 4: the box path, for the next image (docs/prompts/phase-18e-
twelfth-review-fetch-scores-agent-page.md, points 17–22; plan L's boxes stay
on 9310719). A test a point, each failing on 518e4a1: a drafting server
refused at the probe, the tarball checked before the GGUF is fetched, each
file's commit said, a saved page keeping its runtime, more slots where they
fit, and the small ones. The box is tests/fixtures/fake_llama_server.py;
Hugging Face is a stand-in; no model runs."""

from __future__ import annotations

import json
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

import frontier as fb
import frontier_box as fbx
import remote_gguf as rg
import score_bundle as sb
from fake_openai import FakeServer
from service import db, served
from test_12q_devicemark_runs import ME, svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import box, requests, run_box  # noqa: F401
from test_20_step4000 import RT, bundle_at, card, lda_tarball, run_tb  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
REMOTE = (ROOT / "docs" / "REMOTE-RUNS.md").read_text()
STEP4000 = (ROOT / "docs" / "STEP4000-RUN.md").read_text()
SHA = "0123456789abcdef0123456789abcdef01234567"


def own_log(b: dict, out: str) -> str:
    return (b["root"] / out / rg.OWN_LOG).read_text()


# ---------------------------------------------------------------------------
# 17. a server that drafts tokens, under a runtime: refused at the probe
# ---------------------------------------------------------------------------

def test_17_a_drafting_server_is_refused_before_a_question(box, card, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_DRAFTS", "1")
    tb = lda_tarball(tmp_path)
    with pytest.raises(SystemExit, match=f"runtime {RT} runs no speculative decoding, and this "
                                         "server's slots say they draft tokens"):
        run_tb(box, "dr", tb)                                  # 518e4a1: asked, refused at import
    assert not requests(box)
    monkeypatch.delenv("FAKE_LLAMA_DRAFTS")
    assert run_tb(box, "ok", tb) == 0


# ---------------------------------------------------------------------------
# 18. the tarball fetched and run before the GGUF
# ---------------------------------------------------------------------------

def hf_world(monkeypatch, sources: dict[str, Path], commit: str = "") -> list:
    """Hugging Face's client, stood in for: each download recorded"""
    calls: list = []

    def download(repo, path, repo_type=None, revision=None, token=None, local_dir=None):
        calls.append((path, revision))
        dest = Path(local_dir) / path
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(sources[path], dest)
        return str(dest)

    class Api:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo, repo_type=None, revision=None):
            if not commit:
                raise RuntimeError("no")
            return types.SimpleNamespace(sha=commit)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=download, HfApi=Api))
    return calls


@pytest.mark.parametrize("why", ["archs", "broken"])
def test_18_a_tarball_that_wont_run_here_is_refused_before_the_gguf_is_fetched(
        box, card, tmp_path, monkeypatch, why):  # noqa: F811
    tb = lda_tarball(tmp_path, archs="86-real;89-real" if why == "archs" else "120")
    if why == "broken":
        monkeypatch.setenv("FAKE_LLAMA_BROKEN", "1")
    calls = hf_world(monkeypatch, {box["gguf"].name: box["gguf"], tb.name: tb})
    with pytest.raises(SystemExit, match="86-real;89-real" if why == "archs" else
                       f"{tb.name}: llama-server doesn't run here"):
        run_tb(box, "f", f"hf://me/private/{tb.name}", "--gguf",
               f"hf://me/private/{box['gguf'].name}")
    assert [p for p, _ in calls] == [tb.name]                  # 518e4a1: the GGUF first


# ---------------------------------------------------------------------------
# 19. the private repository's files at a commit, said
# ---------------------------------------------------------------------------

def test_19_each_file_says_the_commit_it_came_from(box, tmp_path, monkeypatch):  # noqa: F811
    calls = hf_world(monkeypatch, {"step4000.gguf": box["gguf"]}, commit=SHA)
    said: list[str] = []
    got = rg.fetch("hf://me/evalboard-private/step4000.gguf", tmp_path / "files", said.append)
    assert got.name == "step4000.gguf"
    assert said == [f"fetching step4000.gguf from me/evalboard-private: commit {SHA} — give "
                    f"@{SHA} to fetch this one again"]         # 518e4a1: no commit said
    assert calls == [("step4000.gguf", SHA)]                   # fetched at that commit
    said.clear()
    rg.fetch(f"hf://me/evalboard-private@{SHA}/step4000.gguf", tmp_path / "files", said.append)
    assert said == [f"fetching step4000.gguf from me/evalboard-private at {SHA}: commit {SHA}"]


def test_19_plan_ls_lines_fetch_the_private_files_at_a_commit():
    plan = REMOTE[REMOTE.index("**The files, each pinned on plan L's boxes"):
                  REMOTE.index("5. Once every step reads whole")]
    lines = re.findall(r"hf://<you>/evalboard-private\S*", plan)
    assert lines and all(x.startswith("hf://<you>/evalboard-private@<commit>/") for x in lines)
    assert "Files and versions ▸ History" in plan


# ---------------------------------------------------------------------------
# 20. saving a rented model's page keeps its runtime
# ---------------------------------------------------------------------------

def test_20_saving_the_page_keeps_the_runtime(svc):  # noqa: F811
    srv = FakeServer()
    try:
        form = {"name": "step box", "base_url": srv.base, "how": "x",
                "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "on"}
        rec = served.register(form, ME)
        db.served_put({**served.get(rec["id"]), "runtime": RT})   # as the import registers it
        served.register(form, ME)                              # the page saved
        assert served.get(rec["id"]).get("runtime") == RT      # 518e4a1: dropped
        srv.model_path = "/models/another.gguf"                # another file: not carried
        served.register(form, ME)
        assert "runtime" not in served.get(rec["id"])
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# 21. more slots where the memory check confirms they fit
# ---------------------------------------------------------------------------

FIT = {"used": 40 * 1024 ** 3, "kv": 20 * 1024 ** 3, "per_token": 34 * 1024, "per_slot": 1,
       "file": 20 * 1024 ** 3, "card": 95 * 1024 ** 3, "fit": 25, "types": "q8_0/q8_0",
       "arch": "qwen35moe", "basis": "the GGUF's header"}


def test_21_more_slots_run_where_they_fit(box, tmp_path, monkeypatch):  # noqa: F811
    launched = tmp_path / "launched.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(launched))
    monkeypatch.setattr(rg, "kv_fit", lambda src, flags, slots, ctx, mem: (dict(FIT), ""))
    assert run_box(box, "ms", "--max-slots", "25") == 0         # 518e4a1: no such option
    argv = json.loads(launched.read_text().splitlines()[-1])["argv"]
    assert argv[argv.index("-np") + 1] == "25"
    assert "3 slots asked, and up to 25 of" in own_log(box, "ms") and "25 run (--max-slots 25)" \
        in own_log(box, "ms")
    # no more than --max-slots, nor than fit
    assert run_box(box, "ms2", "--max-slots", "10") == 0
    argv = json.loads(launched.read_text().splitlines()[-1])["argv"]
    assert argv[argv.index("-np") + 1] == "10"
    # the box line passes it on (on an image after 18e)
    a = fbx.parser().parse_args(["L13", "--as", "served/Qwen3.6-35B-A3B-BF16-lda-2610",
                                 "--gguf", "hf://me/p/x.gguf", "--server", "hf://me/p/t.tar.gz",
                                 "--max-slots", "25"])
    got = fbx.argv_of("L13", 1, fbx.box_of("L13")[0], a)
    assert got[got.index("--max-slots") + 1] == "25"


# ---------------------------------------------------------------------------
# 22. the small ones
# ---------------------------------------------------------------------------

def test_22_the_memory_line_says_why_it_wasnt_worked_out(box, card, tmp_path):  # noqa: F811
    assert run_tb(box, "mem", lda_tarball(tmp_path)) == 0
    line = next(x for x in own_log(box, "mem").splitlines() if "memory not worked out" in x)
    assert "not worked out: []" not in line and "couldn't be read" in line, line


def test_22_the_docs_example_is_what_score_bundle_prints():
    spec = fb.BENCH["gpqa_diamond_epoch"]
    assert (spec["n"], spec["epochs"]) == (198, 4)
    sc = {"missing": 0, "score": 0.616, "se": 0.031, "answers": 792, "part": "", "ran_out": 12,
          "loops": 9, "unanswered": 0}
    words = " ".join(REMOTE.split())
    line = sb.line_of("gpqa_diamond_epoch", "step4000-lda-2610", sc)
    assert f'"{line}"' in words, line                          # 518e4a1: "198 answers (shards…"
    part = sb.line_of("gpqa_diamond_epoch", "x", {**sc, "answers": 200,
                                                  "part": "shards 1 of 4: 50 of its 198 questions"})
    assert '"200 answers (shards 1 of 4: 50 of its 198 questions)"' in words
    assert "200 answers (shards 1 of 4: 50 of its 198 questions)" in part
    assert '"graded on the board later"' in words


def test_22_the_step4000_tarball_and_its_check():
    s = " ".join(STEP4000.split())
    assert "for the A100" not in s and "built for sm_120 alone" in s   # 518e4a1: the A100
    check = next(x for x in STEP4000.splitlines() if x.endswith("llama-server --version"))
    assert "nvidia.cuda_runtime" in check and "nvidia.cublas" in check
    assert check.startswith('LD_LIBRARY_PATH="/workspace/lda-build/bin:$(python -c ')
