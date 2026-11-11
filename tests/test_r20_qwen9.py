"""r20: Qwen3.5-9B on GPQA Diamond under our protocol (plan Q) — Unsloth's
Q4_K_M at its pinned commit, on step2388's build, thinking on at its card's
thinking settings, 4 runs of the 198 questions, up to 98,304 tokens an answer.
Nothing is fetched and no model runs."""

from __future__ import annotations

import pytest

import frontier as fb
import frontier_box as fx
from service import frontier as sf
from test_31_1_k3k6 import DOC

RT = "card-qwen35-9b-k3k6"
Q9 = "served/Qwen3.5-9B-Q4_K_M-k3k6-2611"
GPQA = "gpqa_diamond_epoch"
FILE = "Qwen3.5-9B-Q4_K_M.gguf"
REC = {"id": Q9, "name": Q9, "runtime": RT, "base_url": "http://box:8090/v1", "pin": {}}
CARD = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 1.5,
        "repeat_penalty": 1.0}


def test_qwen9_runs_on_step2388s_build_at_its_cards_thinking_settings():
    rt, u = fb.RUNTIMES[RT], fb.RUNTIMES["k3k6-2611"]
    assert (rt["env"], rt["flags"], rt["build"]) == (u["env"], u["flags"], u["build"])
    assert rt["models"] == {Q9: "Qwen/Qwen3.5-9B"} and rt["thinking"] == ("on",)
    assert rt["sampling"] == CARD and "paths" not in rt
    assert fb.runtime_of_model(Q9) == RT and fb.runtime_model_refused(RT, Q9) == ""
    assert fb.runtime_model_refused("k3k6-2611", Q9)


def test_gpqa_is_asked_up_to_98304_with_the_card_and_thinking_on():
    s = sf.settings(REC, GPQA, True)
    assert s["max_tokens"] == 98304 and s["chat_template_kwargs"] == {"enable_thinking": True}
    assert {k: s[k] for k in CARD} == CARD
    assert fb.budget_of(GPQA, True, RT) == 98304 and fb.budget_of(GPQA, True) == 81920
    assert fb.slot_context([GPQA], True, runtime=RT) == 98304 + fb.BENCH[GPQA]["room"]


def test_plan_q_is_gpqa_in_halves_on_unsloths_pinned_file():
    assert fx.PLANS["Q"] == {"Q1": [("on", (GPQA,), "1/2", 8)], "Q2": [("on", (GPQA,), "2/2", 8)]}
    assert fx.PLAN_RUNTIME["Q"] == RT and fx.PLAN_GPUS["Q"] == fx.PLAN_GPUS["U"]
    assert fx.PLAN_PINS["Q"][FILE] == "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
    assert fx.PLAN_REVISIONS["Q"][FILE] == "3885219b6810b007914f3a7950a8d1b469d598a5"
    tb = "llama-server-k3k6-3ad38b7.tar.gz"
    assert (fx.PLAN_PINS["Q"][tb], fx.PLAN_REVISIONS["Q"][tb]) == (
        fx.PLAN_PINS["U"][tb], fx.PLAN_REVISIONS["U"][tb])
    lines = fx.listing("Q")
    assert lines[-1].startswith("2 boxes a build")
    assert all("(a guess, not measured on a box yet)" in ln for ln in lines if ln.startswith("  "))


@pytest.mark.parametrize("label, model, why", [("Q1", "served/step2388-k3k6-2611", "Q1 runs " + Q9),
                                               ("U35", Q9, Q9 + " runs on Q1, Q2 alone")])
def test_plan_qs_boxes_run_qwen9_alone(label, model, why):
    assert why in fx.model_refused(label, model)


def test_the_runbook_gives_plan_qs_line_and_its_pins():
    part = DOC.read_text()
    part = part[part.index("# Q. Qwen3.5-9B on GPQA Diamond: plan Q (r20)"):]
    part = part[:part.index("\n# ", 1)]
    assert f"python scripts/frontier_box.py Q1 --as {Q9}" in part
    assert "hf://unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5/" + FILE in part
    assert f"--sha {Q9}=03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8" in part
