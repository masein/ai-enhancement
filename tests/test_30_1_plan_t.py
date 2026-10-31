"""30.1: plan T — step8804's thinking-off column for the whitepaper: GPQA
Diamond, MATH Level 5 and MMLU-Pro asked as the thinking-on rows were, on the
lda runtime, with thinking off at their thinking-off limits. Three boxes,
MMLU-Pro in halves; the answers land in the build's plain row; step8804's file
pinned. Nothing runs."""

from __future__ import annotations

import pytest

import frontier as fb
import frontier_box as fbx
import remote_gguf as rg

from service import frontier as sf

GPQA, MATH, MMLU = "gpqa_diamond_epoch", "math_l5_epoch", "mmlupro_tiger"
STEP8804 = "c7a719f8c37d85bcd24f5a0bf103323e0102938be01dea3e0992465923188f3a"
LINE = ["T3", "--as", "served/step8804-lda-2610", "--gguf", "step8804.gguf",
        "--server", "llama-server-lda-a4a76a7.tar.gz", "--price", "0.42"]


def test_plan_t_is_three_boxes_mmlu_pro_in_halves_thinking_off():
    assert fbx.PLANS["T"] == {"T1": [("off", (MMLU,), "1/2", 8)],
                              "T2": [("off", (MMLU,), "2/2", 8)],
                              "T3": [("off", (GPQA, MATH), "", 8)]}
    assert fbx.PLAN_RUNTIME["T"] == "lda-2610"
    assert fbx.PLAN_GPUS["T"] == fbx.PLAN_GPUS["L"]                 # sm_120 alone
    # every step fits eight slots of a 5090 at its thinking-off limits
    assert all(fbx.slots_run(s) == 8 for st in fbx.PLANS["T"].values() for s in st)
    assert {t: fb.BENCH[t]["budget"]["off"] for t in (GPQA, MATH, MMLU)} == {
        GPQA: 16384, MATH: 8192, MMLU: 8192}
    lines = fbx.listing("T")
    assert lines[-1].startswith("3 boxes a build · 11.3 box-hours a build · the longest 3.9 h")


def test_the_lda_runtime_asks_these_three_with_thinking_off_and_the_rest_never():
    assert fb.runtime_thinking_refused("lda-2610", "off", [GPQA, MATH]) == ""
    assert fb.runtime_thinking_refused("lda-2610", "off", [MMLU]) == ""
    for t in ("otis_aime_epoch", "hle_text_cais", "simpleqa_epoch", "arc_agi2_public"):
        assert fb.runtime_thinking_refused("lda-2610", "off", [GPQA, t]).startswith(
            "--runtime lda-2610 asks thinking on only")
    # Bolt's runtime is unchanged: instruction following alone
    assert fb.runtime_thinking_refused("card-bolt", "off", [GPQA]).endswith(
        "(off for IFEval and IFBench alone)")


def test_each_question_is_asked_with_the_runtimes_sampling_and_the_switch_off():
    rec = {"id": "served/step8804-lda-2610", "name": "step8804", "based_on": "Qwen/Qwen3.6-35B-A3B",
           "runtime": "lda-2610", "base_url": "http://127.0.0.1:9/v1", "pin": {}}
    lda = fb.RUNTIMES["lda-2610"]["sampling"]
    for t, limit in ((GPQA, 16384), (MATH, 8192), (MMLU, 8192)):
        # the same sampling as thinking on — not the qwen3.6 preset's off values
        assert sf.settings(rec, t, False) == {"max_tokens": limit, **lda,
                                              "chat_template_kwargs": {"enable_thinking": False}}
        assert sf.settings(rec, t, True)["temperature"] == sf.settings(rec, t, False)["temperature"]


def test_the_box_line_runs_its_step_and_pins_step8804s_file():
    a = fbx.parser().parse_args(LINE)
    (step,) = fbx.box_of("T3")
    argv = fbx.argv_of("T3", 1, step, a)
    assert argv[argv.index("--runtime") + 1] == "lda-2610"
    assert argv[argv.index("--thinking") + 1] == "off"
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "--only"] == [GPQA, MATH]
    assert argv[argv.index("--slots") + 1] == "8" and argv[argv.index("--price") + 1] == "0.42"
    assert argv[-1].endswith("step8804-lda-2610/T3-1")
    assert sorted(fbx.pins_of(a, "T3")) == [
        "llama-server-lda-a4a76a7.tar.gz=5b538290644d67139c6c996fdaa3e04cce82fd0b343cce66a4ff234648afb618",
        f"step8804.gguf={STEP8804}"]
    # T1's shard of MMLU-Pro
    (t1,) = fbx.box_of("T1")
    a1 = fbx.argv_of("T1", 1, t1, fbx.parser().parse_args(["T1", *LINE[1:]]))
    assert a1[a1.index("--shard") + 1] == "1/2" and a1[a1.index("--only") + 1] == MMLU


def test_a_box_asked_thinking_off_of_the_others_still_stops_before_anything_is_fetched(tmp_path,
                                                                                         capsys):
    with pytest.raises(SystemExit) as e:
        rg.main(["--as", "served/step8804-lda-2610", "--gguf", str(tmp_path / "x.gguf"),
                 "--server-dir", str(tmp_path), "--runtime", "lda-2610", "--thinking", "off",
                 "--only", GPQA, "--only", "otis_aime_epoch", "--out", str(tmp_path / "o")])
    assert e.value.code == 2 and "asks thinking on only" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
