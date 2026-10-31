"""25: plan I — instruction following, one box a build on the lda runtime:
IFEval and IFBench with thinking on, then the same with thinking off, at
SimpleQA's token limits; the off step's answers in the build's plain row,
the on step's in its · thinking row; the box line with --price. The lda
runtime asks thinking off of these two alone. Nothing runs."""

from __future__ import annotations

import pytest

import frontier as fb
import frontier_box as fbx
import remote_gguf as rg

from service import frontier as sf

IFEVAL, IFBENCH = "ifeval_google", "ifbench_test"
LINE = ["I1", "--as", "served/step8804-lda-2610", "--gguf", "step8804.gguf",
        "--server", "llama-server-lda-a4a76a7.tar.gz", "--price", "0.45"]


def test_I1_is_two_steps_on_then_off_on_the_lda_runtime():
    steps = fbx.box_of("I1")
    assert steps == [("on", (IFEVAL, IFBENCH), "", 8), ("off", (IFEVAL, IFBENCH), "", 8)]
    assert list(fbx.PLANS["I"]) == ["I1"]                          # one box a build
    assert fbx.PLAN_RUNTIME["I"] == "lda-2610"
    assert fbx.PLAN_GPUS["I"] == fbx.PLAN_GPUS["L"]                 # sm_120 alone
    assert fbx.PLAN_PINS["I"] is fbx.PLAN_PINS["L"]                 # the same files, checked
    # both steps fit eight slots of a 5090, at SimpleQA's limits
    assert [fbx.slots_run(s) for s in steps] == [8, 8]
    assert {t: fb.BENCH[t]["budget"] for t in (IFEVAL, IFBENCH)} == {
        IFEVAL: fb.BENCH["simpleqa_epoch"]["budget"], IFBENCH: fb.BENCH["simpleqa_epoch"]["budget"]}
    lines = fbx.listing("I")
    assert lines[1] == "  1. thinking on · IFEval, IFBench · 8 slots · about 1.2 h (a guess)"
    assert lines[2] == "  2. thinking off · IFEval, IFBench · 8 slots · about 0.2 h (a guess)"
    assert lines[-1].startswith("1 box a build · 1.4 box-hours a build")
    # one run a prompt, all of each
    assert (fb.BENCH[IFEVAL]["n"], fb.BENCH[IFBENCH]["n"]) == (541, 300)
    assert fb.BENCH[IFEVAL]["epochs"] == fb.BENCH[IFBENCH]["epochs"] == 1


def test_the_box_line_is_accepted_with_its_price_and_runs_each_step():
    a = fbx.parser().parse_args(LINE)
    on, off = (fbx.argv_of("I1", k, s, a) for k, s in enumerate(fbx.box_of("I1"), 1))
    for argv, th in ((on, "on"), (off, "off")):
        assert argv[argv.index("--runtime") + 1] == "lda-2610"
        assert argv[argv.index("--thinking") + 1] == th
        assert argv[argv.index("--price") + 1] == "0.45"
        assert [argv[i + 1] for i, x in enumerate(argv) if x == "--only"] == [IFEVAL, IFBENCH]
        assert argv[argv.index("--slots") + 1] == "8"
    assert on[-1].endswith("step8804-lda-2610/I1-1") and off[-1].endswith("step8804-lda-2610/I1-2")
    # the files its line names are pinned as plan L's are
    assert sorted(p.split("=")[0] for p in fbx.pins_of(a, "I1")) == [
        "llama-server-lda-a4a76a7.tar.gz"]


def test_the_lda_runtime_asks_thinking_off_of_the_benchmarks_it_names_alone(tmp_path, capsys):
    assert fb.runtime_thinking_refused("lda-2610", "on", [IFEVAL, IFBENCH]) == ""
    assert fb.runtime_thinking_refused("lda-2610", "off", [IFEVAL, IFBENCH]) == ""
    assert fb.runtime_thinking_refused("lda-2610", "off", [IFEVAL]) == ""
    # 30.1: and of GPQA Diamond, MATH Level 5 and MMLU-Pro (plan T) — never the rest
    for tasks in ([IFEVAL, "hle_text_cais"], ["otis_aime_epoch"], []):
        assert fb.runtime_thinking_refused("lda-2610", "off", tasks) == (
            "--runtime lda-2610 asks thinking on only (off for GPQA Diamond, MATH Level 5, "
            "MMLU-Pro, IFEval and IFBench alone)")
    # card-ornith names none: thinking on only, as before
    assert fb.runtime_thinking_refused("card-ornith", "off", [IFEVAL]).startswith(
        "--runtime card-ornith asks thinking on only")
    # the box refuses the rest before anything is fetched or made
    with pytest.raises(SystemExit) as e:
        rg.main(["--as", "served/step8804-lda-2610", "--gguf", str(tmp_path / "x.gguf"),
                 "--server-dir", str(tmp_path), "--runtime", "lda-2610", "--thinking", "off",
                 "--only", IFEVAL, "--only", "hle_text_cais", "--out", str(tmp_path / "o")])
    assert e.value.code == 2 and "MMLU-Pro, IFEval and IFBench alone" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_each_step_is_asked_as_its_row_says_and_lands_in_it():
    rec = {"id": "served/step8804-lda-2610", "name": "step8804", "based_on": "",
           "runtime": "lda-2610", "base_url": "http://127.0.0.1:9/v1", "pin": {}}
    lda = fb.RUNTIMES["lda-2610"]["sampling"]
    for t in (IFEVAL, IFBENCH):
        assert sf.settings(rec, t, True) == {"max_tokens": 32768, **lda,
                                             "chat_template_kwargs": {"enable_thinking": True}}
        assert sf.settings(rec, t, False) == {"max_tokens": 4096, **lda,
                                              "chat_template_kwargs": {"enable_thinking": False}}
        assert fb.slot_context([t], True) == 32768 + 1024
    # an off row's answers that thought anyway are held, as on every off row
    # (more than a quarter refused): the switch off is the import's to check
    assert sf.thinking_refused(IFEVAL, "off", ["<think>\nx\n</think>\n\nreply"] * 4)
    assert not sf.thinking_refused(IFEVAL, "off", ["reply"] * 4)
