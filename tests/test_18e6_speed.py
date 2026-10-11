"""18e, part 5: the speed test (docs/prompts/phase-18e-twelfth-review-fetch-
scores-agent-page.md, points 23–24). Each fails on 518e4a1: a token cut in
half before the scrub, and a setting that STOPPED EARLY exiting 0. A
stand-in server and stand-in weights; no model runs, nothing is fetched."""

from __future__ import annotations

import sys
import types

import pytest

import speed_test as sp
from fake_openai import FakeServer
from test_19_speed import smi

TOKEN = "hf_" + "Q" * 34                     # invented


def test_23_the_scrub_reads_the_whole_error_before_it_is_cut(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    def snapshot_download(*a, **k):
        raise RuntimeError("y" * 290 + " sent " + TOKEN)       # the token across the cut
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=snapshot_download))
    got, why = sp.fetch_bf16(tmp_path)
    assert got is None and why.startswith("the weights couldn't be fetched: yyy")
    assert "hf_Q" not in why and "QQQ" not in why, why[-60:]   # 518e4a1: "hf_Q" kept


def result(stopped: str = "") -> dict:
    return {"gpqa": {"tokens_s": 10, "answers_h": 1}, "mmlu": {"tokens_s": 10, "answers_h": 1,
                                                                "prompt_s": 1, "prompt_read_s": 1},
            "waited": 0, "cut": 0, "refused": 0, "errors": 0,
            "sanity": {sp.GPQA: {"right": 0, "of": 0}, sp.MMLU: {"right": 0, "of": 0}},
            **({"stopped": stopped} if stopped else {})}


@pytest.mark.parametrize("stopped", ["", "its engine stopped (exit 9): CUDA error"])
def test_24_a_setting_that_stopped_early_is_no_pass(tmp_path, monkeypatch, capsys, stopped):
    import remote_gguf as rg
    srv = FakeServer()
    try:
        monkeypatch.setattr(sp, "run", smi([("NVIDIA GeForce RTX 5090", 32_607)]))
        monkeypatch.setattr(sp, "questions", lambda task, root: [])
        (tmp_path / "u" / "llama" / "bin").mkdir(parents=True)
        monkeypatch.setattr(rg, "fetch", lambda src, into, say: tmp_path / "x")
        monkeypatch.setattr(rg, "unpack_server", lambda tb, into: {
            "bin": tmp_path / "u" / "llama" / "bin" / "llama-server", "libs": [],
            "root": tmp_path / "u"})
        monkeypatch.setattr(rg, "sha256_cached", lambda *a, **k: "a" * 64)
        monkeypatch.setattr(rg, "version_of", lambda *a, **k: {"build": 1, "commit": "x"})

        class Up(sp.Proc):
            def start(self, timeout):
                self.base = srv.base
                return ""

            def stop(self):
                pass
        monkeypatch.setattr(sp, "Proc", Up)
        monkeypatch.setattr(sp, "slot_context", lambda base: sp.SLOT_CTX)
        monkeypatch.setattr(sp.Test, "go", lambda self, tick=0.05: result(stopped))
        one = next(k for k, s in enumerate(sp.PLANS["5090"]["settings"], 1)
                   if s["engine"] == "llama" and not s.get("shared"))
        code = sp.main(["5090", "--price", "0.5", "--gguf", "x.gguf", "--server", "s.tar.gz",
                        "--only", str(one), "--out", str(tmp_path / "o"),
                        "--files", str(tmp_path / "f")])
        out = capsys.readouterr().out
        if stopped:
            assert "STOPPED EARLY — its engine stopped" in out
            assert code == 1                                   # 518e4a1: 0
        else:
            assert code == 0, out
    finally:
        srv.close()
