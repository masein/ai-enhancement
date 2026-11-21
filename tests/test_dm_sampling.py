"""19 Nov (masein): DeviceMark's rows say how their answers were sampled. A row
on a runtime of expert paths recorded what each request sent: no temperature in
it is the model file's own sampling for the path; every other row of ours ran
DeviceMark's protocol, greedy. A greedy run of a checkpoint of our own model
(step2388, step8804) is hidden on Mobile ▸ DeviceMark and unranked; the phone
builds of Qwen3.6 and the open models stay. Two runs of one file never share a
label. Fixtures only: the five step2388 runs as their rows recorded them (runs
419, 420, 428, 448, 449), step8804's two, a phone build and an hf model."""

from __future__ import annotations

import pytest

from test_12q_devicemark_board import _row
import devicemark as dm

S = {"runtime": "llama-server", "lookahead": True, "mtp": False, "phone": False,
     "file": "step2388.k3k6.gguf"}
ACC = {"ifeval": 0.8, "mmlu_pro": 0.7, "math": 0.9}
# what each run's requests sent, as its row recorded it (the seed, the path, the rest)
GREEDY_K3 = {"temperature": 0.0, "presence_penalty": 0.0, "seed": 0, "expert_path": 3}
GREEDY_K6 = {"temperature": 0.0, "presence_penalty": 0.0, "seed": 0, "expert_path": 6}
FILE_K3 = {"seed": 0, "expert_path": 3}
MEDIUM = {"chat_template_kwargs": {"enable_thinking": False}, "presence_penalty": 0.0,
          "temperature": 0.0, "seed": 0, "expert_path": 6}
APP = {"temperature": 0.0, "presence_penalty": 0.0, "seed": 0, "expert_path": 6,
       "thinking_budget_tokens": 1024, "reply_limit": 2048}
RUNS = {  # id: (setup, thinking)
    "served/step2388-k3k6-2611": ({**S, "name": "step2388 (k3k6 runtime)", "expert_path": 3,
                                   "request": GREEDY_K3}, False),                       # 419
    "served/step2388-k3k6-2611 · thinking": ({**S, "name": "step2388 (k3k6 runtime)",
                                              "expert_path": 6, "request": GREEDY_K6}, True),  # 420
    "served/step2388-k3k6-2611-file": ({**S, "name": "step2388 (k3k6 runtime, file's own settings)",
                                        "expert_path": 3, "request": FILE_K3}, False),  # 428
    "served/step2388-k3k6-2611-medium": ({**S, "name": "step2388 Medium (k3k6 runtime)",
                                          "expert_path": 6, "request": MEDIUM}, False),  # 448
    "served/step2388-k3k6-2611-app · thinking": ({**S, "name": "step2388 at the app's limits",
                                                  "expert_path": 6, "request": APP}, True),  # 449
    "served/step8804-lda-2610": ({**S, "name": "step8804 (lda runtime)",
                                  "file": "step8804.gguf"}, False),
    "served/step8804-lda-2610 · thinking": ({**S, "name": "step8804 (lda runtime)",
                                             "file": "step8804.gguf"}, True),
    "served/Qwen3.6-k4-LDA-lookahead": ({**S, "phone": True, "name": "Qwen3.6 k4-LDA lookahead",
                                         "file": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"}, False),
    "LiquidAI/LFM2.5-1.2B-Instruct": ({"runtime": "hf transformers (lm_eval)",
                                       "dtype": "bfloat16", "lookahead": False}, False)}
HIDDEN = {"served/step2388-k3k6-2611", "served/step2388-k3k6-2611 · thinking",
          "served/step2388-k3k6-2611-medium", "served/step2388-k3k6-2611-app · thinking",
          "served/step8804-lda-2610", "served/step8804-lda-2610 · thinking"}


@pytest.fixture
def out(tmp_path):
    o = tmp_path / "full"
    for rid, (su, thinks) in RUNS.items():
        _row(o, rid.removesuffix(" · thinking"), ACC, su, thinking=thinks)
    return o


def test_each_row_says_its_sampling_from_what_its_requests_sent(out):
    got = {r["id"]: r["sampling"] for r in dm.rows(out)}
    assert {k for k, v in got.items() if v["key"] == "file"} == {"served/step2388-k3k6-2611-file"}
    assert got["served/step2388-k3k6-2611-file"]["words"] == (
        "the model file's own sampling for expert path k3: nothing sent but the path and seed 0")
    assert got["served/step2388-k3k6-2611"]["words"] == (
        "greedy, DeviceMark's protocol: temperature 0, seed 0, presence penalty 0")
    assert got["served/step2388-k3k6-2611-medium"]["words"].endswith("thinking off by the request")
    assert got["served/step2388-k3k6-2611-app · thinking"]["words"].endswith(
        "the thinking stopped at 1,024 tokens and the reply cut at 2,048 (the app's limits)")
    # no request recorded: DeviceMark's protocol, greedy — the phone build's and an hf model's too
    for rid in ("served/step8804-lda-2610", "served/Qwen3.6-k4-LDA-lookahead",
                "LiquidAI/LFM2.5-1.2B-Instruct"):
        assert got[rid] == {"key": "greedy", "short": "greedy",
                            "words": "greedy, DeviceMark's protocol: temperature 0, seed 0"}, rid


def test_greedy_runs_of_our_checkpoints_are_hidden_and_unranked(out):
    rows = {r["id"]: r for r in dm.rows(out)}
    assert {k for k, r in rows.items() if r["hidden"]} == HIDDEN
    assert "step2388" in rows["served/step2388-k3k6-2611"]["hidden"]
    assert "Show greedy runs of our models" in rows["served/step8804-lda-2610"]["hidden"]
    # the shown rows are ranked among themselves; a hidden row never
    assert all(rows[k]["rank"] is None for k in HIDDEN)
    shown = [k for k in rows if k not in HIDDEN]
    assert all(rows[k]["rank"] for k in shown) and len(shown) == 3
    b = dm.board(out)
    ranked = {r["id"]: r["rank_all"] for r in b["rows"]}
    assert all(ranked[k] is None for k in HIDDEN)
    assert ranked["served/step2388-k3k6-2611-file"] is not None


def test_two_runs_of_one_file_never_share_a_label(out):
    """419 and 428 apart: the path, and greedy or the file's sampling"""
    labels = {r["id"]: r["label"] for r in dm.rows(out)}
    assert labels["served/step2388-k3k6-2611"] == "step2388 · k3 · lookahead · thinking off · greedy"
    assert labels["served/step2388-k3k6-2611-file"] == (
        "step2388 · k3 · lookahead · thinking off · the file's sampling")
    assert labels["served/step2388-k3k6-2611-medium"] == (
        "step2388 · k6 · lookahead · thinking off · greedy")
    assert labels["served/step2388-k3k6-2611-app · thinking"] == (
        "step2388 · k6 · lookahead · thinking on · the app's limits · greedy")
    step = [v for k, v in labels.items() if "step2388" in k]
    assert len(set(step)) == len(step) == 5
    # a phone build's and an hf model's labels are as they were
    assert labels["served/Qwen3.6-k4-LDA-lookahead"] == "phone build (k4-LDA) · lookahead · thinking off"
    assert labels["LiquidAI/LFM2.5-1.2B-Instruct"] == "LiquidAI/LFM2.5-1.2B-Instruct"


def test_a_models_runs_carry_their_sampling_and_why_hidden(out):
    """the model page's cards and the table's columns read these"""
    runs = dm.model_runs(out)
    off = runs["served/step2388-k3k6-2611"]["off"]["row"]
    assert off["sampling"]["key"] == "greedy" and "step2388" in off["hidden"]
    f = runs["served/step2388-k3k6-2611-file"]["off"]["row"]
    assert f["sampling"]["key"] == "file" and f["hidden"] is None
    assert runs["served/Qwen3.6-k4-LDA-lookahead"]["off"]["row"]["hidden"] is None
