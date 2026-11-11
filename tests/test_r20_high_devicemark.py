"""r20: High on DeviceMark's battery as the app runs it (plan H) — step2388's
6-expert path with thinking on, greedy, the thinking stopped by the server at
1,024 tokens and the reply cut at 2,048, inside DeviceMark's cap of 4,096
(runtime k3k6-2611-app). The server is the stand-in (FAKE_LLAMA_PATHS=1,
FAKE_LLAMA_THINK_WORDS for thinking past the budget), the items invented."""

from __future__ import annotations

import json

import pytest

import devicemark as dm
import frontier_box as fx
import import_frontier as imf
import remote_bundle as rb
from service import config
from service import devicemark as sdm
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import TASK, box, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import NAME, SHA, STEP, k3k6, run_k, step_kept  # noqa: F401
from test_31_2_boxes import dm_bundle, dm_requests, dmbox  # noqa: F401
import devicemark_box as dmb

RT = "k3k6-2611-app"
APP = "served/step2388-k3k6-2611-app"
REC = {"id": APP, "name": APP, "runtime": RT, "base_url": "http://box:8090/v1", "pin": {}}
HIGH = {"max_tokens": 3100, "temperature": 0.0, "presence_penalty": 0.0, "seed": 0,
        "expert_path": 6, "thinking_budget_tokens": 1024, "reply_limit": 2048}


def test_high_asks_devicemark_greedy_with_the_apps_limits_inside_its_cap():
    assert sdm.settings(True, rec=REC) == HIGH
    assert HIGH["max_tokens"] <= dm.CAP == 4096
    # with thinking off, and on the server-limit runtime, as they were
    assert "thinking_budget_tokens" not in sdm.settings(False, rec=REC)
    on = sdm.settings(True, rec={**REC, "id": STEP, "runtime": "k3k6-2611"})
    assert on["max_tokens"] == 4096 and "reply_limit" not in on


def test_a_high_box_checks_the_budget_then_asks_devicemark_at_the_apps_limits(
        dmbox, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_THINK_WORDS", "1500")             # thinks past the budget
    assert run_k(dmbox, "dmh", "--devicemark", "--runtime", RT, served_as=APP, thinking="on",
                 only=False) == 0
    asked = dm_requests(dmbox)
    assert len(asked) == 596
    assert all({k: r[k] for k in HIGH if k != "reply_limit"} ==
               {k: v for k, v in HIGH.items() if k != "reply_limit"}
               and "reply_limit" not in r for r in asked)
    path = dm_bundle(dmbox, "dmh", APP, thinking=True)
    b = rb.read(path)
    srv = b["setup"]["server"]
    assert srv["paths_check"]["pass"] and srv["app_check"]["pass"]
    assert b["setup"]["devicemark"]["request"] == HIGH
    # each answer: the thinking stopped at the budget, counted
    name = f"results/{b['bundle']['row']}/{sdm.ANSWERS_NAME}"
    lines = [json.loads(x) for x in b["files"][name].decode().splitlines() if x.strip()]
    assert lines and all(x["app"]["budget_reached"] and x["app"]["thinking_tokens"] == 1024
                         for x in lines)
    rec = imf._would_be(b, dmbox["sha"])
    battery = dm.battery_hashes(dm.load_items(config.DM_ITEMS))
    assert dmb.checks(b, rec, dmbox["sha"], battery) == []
    # without the budget check passed, the import refuses it
    c = json.loads(json.dumps({k: v for k, v in b.items() if k != "files"}))
    c["files"] = dict(b["files"])
    c["setup"]["server"]["app_check"]["pass"] = False
    assert "its check that the server held the thinking budget, passed" in dmb.checks(
        c, rec, dmbox["sha"], battery)


def test_plan_h_is_one_box_of_high_on_devicemark():
    assert fx.PLANS["H"] == {"H1": [("on", (fx.DM,), "", 8)]}
    assert fx.PLAN_RUNTIME["H"] == RT and fx.PLAN_PINS["H"] == fx.PLAN_PINS["U"]
    assert fx.PLAN_PINS["H"][NAME] == SHA
    lines = fx.listing("H")
    assert lines[-1].startswith("1 boxes a build") or lines[-1].startswith("1 box a build")
    assert "(a guess, not measured on a box yet)" in lines[1]


@pytest.mark.parametrize("label, model", [("H1", STEP), ("U39", APP)])
def test_plan_hs_box_alone_asks_devicemark_at_the_apps_limits(label, model):
    with pytest.raises(SystemExit, match="plan P's and H's boxes, and they alone, ask under the"):
        fx.main([label, "--as", model, "--gguf", f"hf://me/p@c/{NAME}", "--server",
                 "hf://me/p@c/llama-server-k3k6-3ad38b7.tar.gz", "--price", "0.45"])
