"""r20 (masein, 12 Nov): Medium's presence penalty, a setting of its own
runtime row — k3k6-2611-medium at 0 (plan M, as asked), k3k6-2611-medium-075 at
0.75 (High's, the app's Medium as Kevin's section 3 has it; plan N) — so a run
at 0.75 is a new row, not a code change. Plan N asks the thinking-off set at
Instant's final limit, 32,768, at once. The server is the stand-in (tests/
fixtures/fake_llama_server.py, FAKE_LLAMA_PATHS=1: a request's enable_thinking
over its path), the questions invented; nothing is fetched and no model runs."""

from __future__ import annotations

import pytest

import frontier as fb
import frontier_box as fx
import import_frontier as imf
import path_check
import remote_bundle as rb
from service import devicemark as sdm
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import TASK, box, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import DOC, NAME, SHA, k3k6, run_k  # noqa: F401

R0, R75 = "k3k6-2611-medium", "k3k6-2611-medium-075"
M0, M75 = "served/step2388-k3k6-2611-medium", "served/step2388-k3k6-2611-medium-075"
REC0 = {"id": M0, "name": M0, "runtime": R0, "base_url": "http://box:8090/v1", "pin": {}}
REC75 = {"id": M75, "name": M75, "runtime": R75, "base_url": "http://box:8090/v1", "pin": {}}
GPQA, MATH, MMLU = "gpqa_diamond_epoch", "math_l5_epoch", "mmlupro_tiger"
IFEVAL, IFBENCH = "ifeval_google", "ifbench_test"
OFF = (GPQA, MMLU, MATH, IFEVAL, IFBENCH)


def test_each_medium_row_says_its_own_presence_penalty():
    a, b = fb.RUNTIMES[R0], fb.RUNTIMES[R75]
    assert (a["presence_penalty"], b["presence_penalty"]) == (0.0, 0.75)
    assert fb.send_of(R0) == {"chat_template_kwargs": {"enable_thinking": False},
                              "presence_penalty": 0.0}
    assert fb.send_of(R75) == {"chat_template_kwargs": {"enable_thinking": False},
                               "presence_penalty": 0.75}
    # one row each, never mixed
    assert list(a["models"]) == [M0] and list(b["models"]) == [M75]
    # the same build, file, flags, variables and paths; the penalty, its words and
    # (at 0.75) Instant's final limit apart
    same = ("env", "flags", "build", "paths", "thinking", "off_for", "mode", "sampling",
            "refuses_env", "refuses_flags")
    assert {k: a.get(k) for k in same} == {k: b.get(k) for k in same}
    assert "presence penalty 0.75" in b["words"] and "no presence penalty" in a["words"]
    assert fb.path_expects(R75, 6)["presence_penalty"] == 0.75 == fb.PATHS[6]["presence_penalty"]
    assert fb.path_expects(R0, 6)["presence_penalty"] == 0.0
    assert fb.path_expects(R75, 6)["thinking"] is False is fb.path_expects(R0, 6)["thinking"]


@pytest.mark.parametrize("task", OFF)
def test_a_075_question_says_path_6_thinking_off_and_075_at_32768(task):
    got = sf.settings(REC75, task, False)
    assert (got["expert_path"], got["chat_template_kwargs"], got["presence_penalty"]) == (
        6, {"enable_thinking": False}, 0.75)
    assert got["max_tokens"] == fb.budget_of(task, False, R75) == 32768
    # Medium at 0 keeps the thinking-off limits it was asked at
    was = sf.settings(REC0, task, False)
    assert was["presence_penalty"] == 0.0 and was["max_tokens"] == fb.BENCH[task]["budget"]["off"]
    assert {k: v for k, v in got.items() if k not in ("presence_penalty", "max_tokens")} == {
        k: v for k, v in was.items() if k not in ("presence_penalty", "max_tokens")}


def test_devicemark_asks_both_medium_rows_alike_greedy_with_no_penalty():
    assert sdm.settings(False, rec=REC75) == sdm.settings(False, rec=REC0)
    got = sdm.settings(False, rec=REC75)
    assert (got["temperature"], got["presence_penalty"], got["expert_path"]) == (0.0, 0.0, 6)
    # so plan N doesn't ask it: M4's row is Medium's DeviceMark either way
    assert all(fx.DM not in tasks for steps in fx.PLANS["N"].values() for _, tasks, _, _ in steps)


def test_a_075_box_checks_path_6_without_thinking_at_075_then_asks(k3k6):  # noqa: F811
    assert run_k(k3k6, "n", "--runtime", R75, served_as=M75, thinking="off") == 0
    got = requests(k3k6)
    chk = [r for r in got if (r.get("messages") or [{}])[-1].get("content")
           in set(path_check.QUESTIONS)]
    bench = [r for r in got if r not in chk and "max_tokens" in r and "messages" in r]
    assert len(chk) == 2 and all(r["expert_path"] == 6 and r["presence_penalty"] == 0.75
                                 and r["chat_template_kwargs"] == {"enable_thinking": False}
                                 for r in chk)
    assert bench and all(r["expert_path"] == 6 and r["presence_penalty"] == 0.75
                         and r["chat_template_kwargs"] == {"enable_thinking": False}
                         and r["max_tokens"] == 32768 for r in bench)
    b = rb.read(k3k6["root"] / "n" / rb.bundle_name("frontier", M75, False, None,
                                                    parts=[fb.BENCH[TASK]["short"]]))
    pc = b["setup"]["server"]["paths_check"]
    assert pc["pass"] and pc["path"] == 6 and pc["asked"]["thinking"] is False
    assert pc["asked"]["presence_penalty"] == 0.75 and pc["sent"]["presence_penalty"] == 0.75
    # its own row: imported onto it, refused onto Medium's at 0
    assert imf.runtime_problem(b["setup"], {"id": M75, "runtime": R75}) == ""
    assert "it was asked with runtime k3k6-2611-medium-075" in imf.runtime_problem(
        b["setup"], {"id": M0, "runtime": R0})


def test_plan_n_asks_the_thinking_off_set_in_three_boxes_at_8_slots():
    plan = fx.PLANS["N"]
    assert fx.PLAN_RUNTIME["N"] == R75 and fx.PLAN_GPUS["N"] == fx.PLAN_GPUS["U"]
    assert fx.PLAN_PINS["N"] == fx.PLAN_PINS["U"] and fx.PLAN_PINS["N"][NAME] == SHA
    have: dict = {}
    for steps in plan.values():
        for th, tasks, shard, slots in steps:
            assert th == "off" and slots == 8
            for t in tasks:
                have.setdefault(t, []).append(shard)
    assert set(have) == set(OFF) and sorted(have[MMLU]) == ["1/2", "2/2"]
    lines = fx.listing("N")
    assert lines[-1].startswith("3 boxes a build")
    assert all("8 slots · about 2.6 h" in ln for ln in lines if ln.startswith("  "))
    # paced at plan M's boxes as they ran, the answers M ran out on writing on
    assert fx.PLAN_PACE["N"][("off", MMLU)] > fx.MEDIUM_RAN_PACE[MMLU]
    assert fx.PLAN_PACE["N"][("off", GPQA)] == fx.MEDIUM_RAN_PACE[GPQA]


@pytest.mark.parametrize("label, model, words", [
    ("N1", M0, "plan M's boxes, and they alone"),       # Medium at 0 on a plan N box
    ("M1", M75, "plan M's boxes, and they alone"),      # 0.75 on a plan M box
    ("U35", M75, "plan N's boxes, and they alone"),     # 0.75 on any other plan's
])
def test_plan_ns_boxes_alone_ask_at_075(label, model, words):
    with pytest.raises(SystemExit, match=words):
        fx.main([label, "--as", model, "--gguf", f"hf://me/p@c/{NAME}", "--server",
                 "hf://me/p@c/llama-server-k3k6-3ad38b7.tar.gz", "--price", "0.45"])


def test_the_runbook_gives_plan_ns_line():
    part = DOC.read_text()
    part = part[part.index("# N. Medium at presence penalty 0.75: plan N (r20)"):]
    part = part[:part.index("\n# ", 1)]
    assert f"python scripts/frontier_box.py N1 --as {M75}" in part
    assert f"--sha {M75}={SHA}" in part and '"presence_penalty": 0.75' in part
    assert '--register "step2388 Medium, presence 0.75 (k3k6 runtime)"' in part
