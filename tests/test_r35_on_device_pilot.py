"""r35: plan O, the on-device pilot — Qwen3.5-4B, Gemma 4 E4B and Gemma 4 E2B on
GPQA Diamond and MMLU-Pro under our protocol: Unsloth's Q4_K_M of each at its
pinned commit, on step2388's build, thinking on at each card's settings, up to
98,304 tokens an answer on both. masein's yes (16 Nov) is for the pilot alone:
32 questions of each benchmark a box. Nothing is fetched and no model runs."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

import frontier as fb
import frontier_box as fx
from service import frontier as sf
from test_31_1_k3k6 import DOC

GPQA, MMLU = "gpqa_diamond_epoch", "mmlupro_tiger"
Q4B, E4B, E2B = ("served/Qwen3.5-4B-Q4_K_M-k3k6-2611", "served/gemma-4-E4B-it-Q4_K_M-k3k6-2611",
                 "served/gemma-4-E2B-it-Q4_K_M-k3k6-2611")
RT_Q, RT_G = "card-qwen35-4b-k3k6", "card-gemma4-k3k6"
TB = "llama-server-k3k6-3ad38b7.tar.gz"
FILES = {  # name: (repository, commit, sha256) — Hugging Face's LFS listing, read 16 Nov
    "Qwen3.5-4B-Q4_K_M.gguf": ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523",
                               "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
    "gemma-4-E4B-it-Q4_K_M.gguf": ("unsloth/gemma-4-E4B-it-GGUF", "bfc15c382204943c3a8fff0c750b94ae2364d7a3",
                                   "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87"),
    "gemma-4-E2B-it-Q4_K_M.gguf": ("unsloth/gemma-4-E2B-it-GGUF", "0314792d7f1f7e229411f620751375812bb9faf2",
                                   "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8")}
BOXES = {"O1": (Q4B, "Qwen3.5-4B-Q4_K_M.gguf", RT_Q, 12), "O2": (E4B, "gemma-4-E4B-it-Q4_K_M.gguf", RT_G, 16),
         "O3": (E2B, "gemma-4-E2B-it-Q4_K_M.gguf", RT_G, 16)}
GEMMA = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0,
         "repeat_penalty": 1.0}


def line(box: str, *more: str) -> list[str]:
    served, name, _, _ = BOXES[box]
    repo, rev, _ = FILES[name]
    return [box, "--as", served, "--gguf", f"hf://{repo}@{rev}/{name}", "--server",
            f"hf://me/evalboard-private@{fx.PLAN_REVISIONS['U'][TB]}/{TB}", *more]


def test_the_4b_runs_on_step2388s_build_at_its_cards_thinking_settings_as_the_9b():
    rt, u, q9 = fb.RUNTIMES[RT_Q], fb.RUNTIMES["k3k6-2611"], fb.RUNTIMES["card-qwen35-9b-k3k6"]
    assert (rt["env"], rt["flags"], rt["build"]) == (u["env"], u["flags"], u["build"])
    assert rt["models"] == {Q4B: "Qwen/Qwen3.5-4B"} and rt["thinking"] == ("on",) and "paths" not in rt
    assert rt["sampling"] == q9["sampling"]                     # the card's line is the 9B's word for word
    assert set(rt["sampling_from"]) == set(rt["sampling"])
    assert all("Qwen/Qwen3.5-4B's model card at 851bf6e8" in v for v in rt["sampling_from"].values())
    assert fb.runtime_of_model(Q4B) == RT_Q and fb.runtime_model_refused(RT_Q, Q4B) == ""
    assert fb.runtime_model_refused("card-qwen35-9b-k3k6", Q4B)


def test_gemma_4_runs_on_step2388s_build_at_googles_card_with_its_thinking_switch_checked():
    rt, u = fb.RUNTIMES[RT_G], fb.RUNTIMES["k3k6-2611"]
    assert (rt["env"], rt["flags"], rt["build"]) == (u["env"], u["flags"], u["build"])
    assert rt["models"] == {E4B: "google/gemma-4-E4B-it", E2B: "google/gemma-4-E2B-it"}
    assert rt["sampling"] == GEMMA and set(rt["sampling_from"]) == set(GEMMA)
    assert all(rt["sampling_from"][k].startswith("Google's model cards") for k in ("temperature", "top_p", "top_k"))
    assert all(rt["sampling_from"][k].startswith("neutral") for k in ("min_p", "presence_penalty", "repeat_penalty"))
    # the cards' trigger, checked on the box's server before a question, and Bolt's attention fallbacks
    assert rt["think_marker"] == "<|think|>"
    assert rt["attn_fallbacks"] == fb.RUNTIMES["card-bolt"]["attn_fallbacks"]
    assert [s[:6] for s in fb.runtime_flag_sets(RT_G)] == [
        ["-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on"], ["-ctk", "f16", "-ctv", "f16", "--flash-attn", "on"],
        ["-ctk", "f16", "-ctv", "f16", "--flash-attn", "off"]]
    assert not fb.context_cap(RT_G)                              # 131,072 holds 98,304 and its prompt
    for m in (E4B, E2B):
        assert fb.runtime_of_model(m) == RT_G and fb.runtime_model_refused(RT_G, m) == ""


@pytest.mark.parametrize("served, rt", [(Q4B, RT_Q), (E4B, RT_G), (E2B, RT_G)])
@pytest.mark.parametrize("task", [GPQA, MMLU])
def test_both_benchmarks_are_asked_up_to_98304_with_the_card_and_thinking_on(served, rt, task):
    rec = {"id": served, "name": served, "runtime": rt, "base_url": "http://box:8090/v1", "pin": {}}
    s = sf.settings(rec, task, True)
    assert s["max_tokens"] == 98304 and s["chat_template_kwargs"] == {"enable_thinking": True}
    assert {k: s[k] for k in fb.RUNTIMES[rt]["sampling"]} == fb.RUNTIMES[rt]["sampling"]
    assert fb.budget_of(task, True, rt) == 98304
    assert fb.slot_context([GPQA, MMLU], True, runtime=rt) == 98304 + max(fb.BENCH[t]["room"] for t in (GPQA, MMLU))
    # the benchmarks' own limits, for every other model, as before
    assert fb.budget_of(MMLU, True) == 32768 and fb.budget_of(GPQA, True) == 81920


def test_plan_o_is_a_box_a_model_on_unsloths_pinned_files_and_plan_us_server():
    assert {b: st for b, st in fx.PLANS["O"].items()} == {
        b: [("on", (GPQA, MMLU), "", slots)] for b, (_, _, _, slots) in BOXES.items()}
    for name, (_, rev, sha) in FILES.items():
        assert (fx.PLAN_PINS["O"][name], fx.PLAN_REVISIONS["O"][name]) == (sha, rev)
    assert (fx.PLAN_PINS["O"][TB], fx.PLAN_REVISIONS["O"][TB]) == (fx.PLAN_PINS["U"][TB], fx.PLAN_REVISIONS["U"][TB])
    assert fx.PLAN_GPUS["O"] == fx.PLAN_GPUS["U"] and "O" in fx.GUESSED_PLANS
    assert fx.listing("O")[-1].startswith("3 boxes a build")
    for box, (served, _, rt, _) in BOXES.items():
        a = fx.parser().parse_args(line(box, "--pilot"))
        assert fx.runtime_of(a, box) == rt and fx.model_refused(box, served) == ""
        assert fx.revision_refused(a, box) == ""


@pytest.mark.parametrize("box, other", [("O1", E4B), ("O2", Q4B), ("O3", E4B)])
def test_each_box_of_plan_o_runs_its_own_model_alone(box, other):
    assert f"{box} runs {BOXES[box][0]}, not {other}" == fx.model_refused(box, other)
    assert "on O" in fx.model_refused("U35", BOXES[box][0])


def test_a_box_line_of_plan_o_at_another_commit_is_refused_before_anything_is_fetched():
    x = line("O2", "--pilot")
    x[x.index("--gguf") + 1] = "hf://unsloth/gemma-4-E4B-it-GGUF@main/gemma-4-E4B-it-Q4_K_M.gguf"
    why = fx.revision_refused(fx.parser().parse_args(x), "O2")
    assert "gemma-4-E4B-it-Q4_K_M.gguf at main, where plan O fetches it at bfc15c38" in why


@pytest.mark.parametrize("box", list(BOXES))
def test_the_pilot_asks_32_of_each_benchmark_at_the_boxs_slots_on_its_pinned_files(box):
    served, name, rt, slots = BOXES[box]
    a = fx.parser().parse_args(line(box, "--pilot"))
    step = fx.pilot_step(box)
    assert step == ("pilot", (GPQA, MMLU), "", slots)
    argv = fx.argv_of(box, 0, step, a)
    assert "--pilot" in argv and argv[argv.index("--pilot-each") + 1] == "32"
    assert [argv[k + 1] for k, x in enumerate(argv) if x == "--only"] == [GPQA, MMLU]
    assert argv[argv.index("--slots") + 1] == str(slots) and argv[argv.index("--min-slots") + 1] == str(slots - 1)
    assert argv[argv.index("--thinking") + 1] == "on" and argv[argv.index("--runtime") + 1] == rt
    assert [argv[k + 1] for k, x in enumerate(argv) if x == "--pin"] == sorted(
        [f"{name}={FILES[name][2]}", f"{TB}={fx.PLAN_PINS['U'][TB]}"])
    assert argv[argv.index("--out") + 1] == f"/workspace/{served.split('/', 1)[1]}/pilot"
    assert "--price" not in argv
    # another plan's pilot as before: 20 and 5, 8 slots, no --only
    assert fx.pilot_step("L1") == fx.PILOT_STEP


def test_a_line_of_plan_o_without_pilot_is_refused(capsys):
    with pytest.raises(SystemExit):
        fx.main(line("O1"))
    assert "plan O runs its pilot alone (masein's yes, 16 Nov): add --pilot" in capsys.readouterr().err


def test_the_pilot_line_says_its_32_questions_and_what_it_measured_then_its_verdict(monkeypatch, capsys,
                                                                                  tmp_path):
    monkeypatch.setattr(fx, "gpu_refused", lambda label: "")

    def ran(cmd):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "pilot-thinking-on.json").write_text(json.dumps({
            "separated": True, "verdict": "Thinking and answer separated correctly: all 64",
            "paces": {GPQA: {"answers": 32, "tokens": 480_000, "tok_s": 710.4, "at_once": 12, "ran_out": 1,
                             "loops": 0},
                      MMLU: {"answers": 32, "tokens": 160_000, "tok_s": 802.0, "at_once": 12, "ran_out": 0,
                             "loops": 0}}}))
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(fx.subprocess, "run", ran)
    assert fx.main(line("O1", "--pilot", "--root", str(tmp_path))) == 0
    said = [x for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert said[0] == ("O1: the pilot — 32 GPQA Diamond and 32 MMLU-Pro questions, thinking on, each asked "
                       "once, before the other boxes are rented")
    assert said[1] == fb.runtime_words(RT_Q)
    assert ("GPQA Diamond: 32 answers, about 15,000 tokens an answer, 710 tokens a second together at 12 at a "
            "time · 1 ran out of room · 0 ended in a loop") in said
    assert ("MMLU-Pro: 32 answers, about 5,000 tokens an answer, 802 tokens a second together at 12 at a time · "
            "0 ran out of room · 0 ended in a loop") in said
    assert not any("pace here" in x for x in said)
    assert said[-1] == "Thinking and answer separated correctly: all 64"


def test_the_runbook_gives_plan_os_three_pilot_lines_and_their_pins():
    part = DOC.read_text(encoding="utf-8")
    part = part[part.index("# O. On-device models: plan O's pilot (r35)"):]
    part = part[:part.index("\n# ", 1)]
    lines = [shlex.split(b.strip()) for b in re.findall(r"```bash\n(.*?)```", part, re.S)]
    assert [x[2] for x in lines] == ["O1", "O2", "O3"]
    for x in lines:
        assert x[:2] == ["python", "scripts/frontier_box.py"] and "--pilot" in x
        served, name, _, _ = BOXES[x[2]]
        repo, rev, sha = FILES[name]
        assert x[x.index("--as") + 1] == served
        assert x[x.index("--gguf") + 1] == f"hf://{repo}@{rev}/{name}"
        assert x[x.index("--server") + 1] == f"hf://<you>/evalboard-private@{fx.PLAN_REVISIONS['U'][TB]}/{TB}"
        assert sha in part and fx.revision_refused(fx.parser().parse_args(x[2:]), x[2]) == ""
