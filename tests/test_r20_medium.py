"""r20: the app's Medium mode — step2388 on its 6-expert path with thinking
off (runtime k3k6-2611-medium), plan M. Each request names path 6 and says
thinking off and no presence penalty; the settings check holds its two answers
to that. The server is the stand-in (tests/fixtures/fake_llama_server.py,
FAKE_LLAMA_PATHS=1: a request's enable_thinking over its path, as the team's
server has it), the questions invented; nothing is fetched and no model runs."""

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
from test_31_1_k3k6 import DOC, NAME, SHA, STEP, k3k6, run_k, step_kept  # noqa: F401

RT = "k3k6-2611-medium"
MED = "served/step2388-k3k6-2611-medium"
GPQA, MATH, MMLU = "gpqa_diamond_epoch", "math_l5_epoch", "mmlupro_tiger"
IFEVAL, IFBENCH = "ifeval_google", "ifbench_test"
OFF = (GPQA, MMLU, MATH, IFEVAL, IFBENCH)
REC = {"id": MED, "name": MED, "runtime": RT, "base_url": "http://box:8090/v1", "pin": {}}
SAYS = {"chat_template_kwargs": {"enable_thinking": False}, "presence_penalty": 0.0}


def test_medium_is_plan_us_server_on_path_6_with_thinking_off():
    rt, u = fb.RUNTIMES[RT], fb.RUNTIMES["k3k6-2611"]
    assert (rt["env"], rt["flags"], rt["build"]) == (u["env"], u["flags"], u["build"])
    assert rt["models"] == {MED: "Qwen/Qwen3.6-35B-A3B"} and rt["mode"] == "Medium"
    assert fb.path_of(RT, "off") == 6 and fb.path_of(RT, "on") is None
    assert fb.send_of(RT) == SAYS and fb.send_of("k3k6-2611") == {}
    # its answers: no thinking, no presence penalty; the rest path 6's own, path 3's values
    want = fb.path_expects(RT, 6)
    assert want["thinking"] is False and want["presence_penalty"] == 0.0
    assert {k: want[k] for k in ("temperature", "top_p", "top_k", "min_p")} == {
        k: fb.PATHS[3][k] for k in ("temperature", "top_p", "top_k", "min_p")}
    assert fb.path_expects("k3k6-2611", 6) == fb.PATHS[6]
    assert all(fb.runtime_thinking_refused(RT, "off", [t]) == "" for t in OFF)
    assert fb.runtime_thinking_refused(RT, "on", [GPQA])


@pytest.mark.parametrize("task", OFF)
def test_every_question_says_path_6_and_thinking_off_at_the_thinking_off_limit(task):
    s = sf.settings(REC, task, False)
    assert s == {"max_tokens": fb.BENCH[task]["budget"]["off"], "expert_path": 6, **SAYS}


def test_devicemark_asks_greedy_on_path_6_with_thinking_off():
    s = sdm.settings(False, rec=REC)
    assert (s["expert_path"], s["temperature"], s["presence_penalty"]) == (6, 0.0, 0.0)
    assert s["chat_template_kwargs"] == {"enable_thinking": False}
    # Instant's and High's DeviceMark requests are as they were
    k3 = sdm.settings(False, rec={**REC, "id": STEP, "runtime": "k3k6-2611"})
    assert "chat_template_kwargs" not in k3 and k3["expert_path"] == 3


def test_a_medium_box_checks_path_6_without_thinking_then_asks(k3k6):  # noqa: F811
    assert run_k(k3k6, "m", "--runtime", RT, served_as=MED, thinking="off") == 0
    got = requests(k3k6)
    chk = [r for r in got if (r.get("messages") or [{}])[-1].get("content")
           in set(path_check.QUESTIONS)]
    bench = [r for r in got if r not in chk and "max_tokens" in r and "messages" in r]
    assert len(chk) == 2 and all(r["expert_path"] == 6 and r["chat_template_kwargs"] ==
                                 {"enable_thinking": False} for r in chk)
    assert bench and all(r["expert_path"] == 6 and r["presence_penalty"] == 0.0
                         and r["chat_template_kwargs"] == {"enable_thinking": False}
                         for r in bench)
    b = rb.read(k3k6["root"] / "m" / rb.bundle_name("frontier", MED, False, None,
                                                    parts=[fb.BENCH[TASK]["short"]]))
    pc = b["setup"]["server"]["paths_check"]
    assert pc["pass"] and pc["path"] == 6 and pc["asked"]["thinking"] is False
    assert all(a["thinking"] is False for a in pc["answers"])
    assert imf.runtime_problem(b["setup"], {"id": MED, "runtime": RT}) == ""


def test_a_server_that_thinks_on_medium_stops_the_box_before_a_question(
        k3k6, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_PATH_ALONE", "1")          # it thinks on path 6 whatever is said
    with pytest.raises(SystemExit, match="settings check of expert path k6 failed"):
        run_k(k3k6, "mt", "--runtime", RT, served_as=MED, thinking="off")


def test_plan_m_asks_the_thinking_off_set_once_8_slots_a_box():
    plan = fx.PLANS["M"]
    assert fx.PLAN_RUNTIME["M"] == RT and fx.PLAN_GPUS["M"] == fx.PLAN_GPUS["U"]
    assert fx.PLAN_PINS["M"] == fx.PLAN_PINS["U"] and fx.PLAN_PINS["M"][NAME] == SHA
    have: dict = {}
    for label, steps in plan.items():
        if label == "M5":                   # 12 Nov: its answers that ran out, continued
            continue
        for th, tasks, shard, slots in steps:
            assert th == "off" and slots == 8
            for t in tasks:
                have.setdefault(t, []).append(shard)
    assert set(have) == {*OFF, fx.DM} and sorted(have[MMLU]) == ["1/2", "2/2"]
    assert all(have[t] == [""] for t in have if t != MMLU)
    lines = fx.listing("M")
    assert lines[-1].startswith("5 boxes a build") and "box-hours a build" in lines[-1]
    assert all("(a guess, not measured on a box yet)" in ln or "at the most" in ln
               for ln in lines if ln.startswith("  "))


@pytest.mark.parametrize("label, model", [("M1", STEP), ("U35", MED)])
def test_plan_ms_boxes_alone_ask_in_medium(label, model):
    with pytest.raises(SystemExit, match="plan M's boxes, and they alone, ask in the app's Medium"):
        fx.main([label, "--as", model, "--gguf", f"hf://me/p@c/{NAME}", "--server",
                 "hf://me/p@c/llama-server-k3k6-3ad38b7.tar.gz", "--price", "0.45"])


def test_the_runbook_gives_plan_ms_line():
    part = DOC.read_text()
    part = part[part.index("# M. The app's Medium mode: plan M (r20)"):]
    part = part[:part.index("\n# ", 1)]
    assert f"python scripts/frontier_box.py M1 --as {MED}" in part
    assert f"--sha {MED}={SHA}" in part and '"enable_thinking": false' in part
