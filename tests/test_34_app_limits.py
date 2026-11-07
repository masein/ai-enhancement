"""34: plan P — step2388 asked under the apps' limits (Kevin's Section 3.4):
runtime k3k6-2611-app, the thinking stopped by the server at 1,024 tokens
(thinking_budget_tokens), the reply cut at 2,048 when it comes back, 2,048 the
limit with thinking off. The server is the stand-in
(tests/fixtures/fake_llama_server.py: FAKE_LLAMA_PATHS=1, and
FAKE_LLAMA_THINK_WORDS for thinking long enough to reach a budget), the
questions invented; nothing is fetched and no model runs."""

from __future__ import annotations

import json
import re

import pytest

import app_check
import frontier as fb
import frontier_box as fx
import import_frontier as imf
import import_remote as ir
import path_check
import remote_bundle as rb
from service import served
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import TASK, box, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import DOC, NAME, SHA, STEP, k3k6, run_k, step_kept  # noqa: F401

RT = "k3k6-2611-app"
APP = "served/step2388-k3k6-2611-app"
GPQA, MATH, MMLU = "gpqa_diamond_epoch", "math_l5_epoch", "mmlupro_tiger"
OTIS, IFEVAL, IFBENCH = "otis_aime_epoch", "ifeval_google", "ifbench_test"
SIX = (GPQA, MMLU, MATH, OTIS, IFEVAL, IFBENCH)
REC = {"id": APP, "name": APP, "runtime": RT, "base_url": "http://box:8090/v1", "pin": {}}


def bundle_at(b: dict, out: str, thinking: bool = True):
    return b["root"] / out / rb.bundle_name("frontier", APP, thinking, None,
                                            parts=[fb.BENCH[TASK]["short"]])


def asked(b: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """the requests the server answered: the settings check's, the budget
    check's, then the benchmark's"""
    got = requests(b)
    last = [(r.get("messages") or [{}])[-1].get("content") for r in got]
    chk = [r for r, q in zip(got, last) if q in set(path_check.QUESTIONS)]
    budget = [r for r, q in zip(got, last) if q == app_check.QUESTION]
    return chk, budget, [r for r in got if r not in chk and r not in budget]


def answer_lines(bundle) -> list[dict]:
    b = rb.read(bundle)
    return [json.loads(x) for name, raw in b["files"].items() if name.endswith(".jsonl")
            for x in raw.decode().splitlines() if x.strip() and '"answer"' in x]


# ---------------------------------------------------------------------------
# the runtime and its settings
# ---------------------------------------------------------------------------

def test_the_app_runtime_is_plan_us_server_asked_at_the_apps_limits():
    # the same build, flags, variables and paths as k3k6-2611: only the limits differ
    u, a = fb.RUNTIMES["k3k6-2611"], fb.RUNTIMES[RT]
    same = set(u) - {"models", "words", "off_for"}
    assert same <= set(a) and all(a[k] == u[k] for k in same)
    assert {k: fb.app_of(RT)[k] for k in ("thinking_budget", "reply", "room")} == {
        "thinking_budget": 1024, "reply": 2048, "room": 28}
    # masein's three assumptions, recorded with the runtime
    said = fb.app_of(RT)["assumed"]
    assert "no message" in said and "reply alone" in said and "sample as the file stores" in said
    assert fb.app_of("k3k6-2611") is None and fb.app_of(None) is None
    # a model of its own, so its rows never mix with plan U's; both modes of all six
    assert fb.runtime_of_model(APP) == RT and fb.runtime_of_model(STEP) == "k3k6-2611"
    assert set(a["off_for"]) == set(SIX)


@pytest.mark.parametrize("task", SIX)
def test_every_benchmark_is_asked_at_the_apps_limits_in_both_modes(task):
    # thinking on: the budget sent, the reply's limit kept for the cut, the
    # request's limit the two and the forced end; off: 2,048 the request's
    assert sf.settings(REC, task, True) == {"max_tokens": 3100, "expert_path": 6,
                                            "thinking_budget_tokens": 1024, "reply_limit": 2048}
    assert sf.settings(REC, task, False) == {"max_tokens": 2048, "expert_path": 3}
    # a slot holds the limits and the prompt alone, never past the apps' 16,384
    room = fb.BENCH[task]["room"]
    assert fb.slot_context([task], True, runtime=RT) == 3100 + room <= 16_384
    assert fb.slot_context([task], False, runtime=RT) == 2048 + room
    # plan U's own budgets are unchanged
    u = {"id": STEP, "runtime": "k3k6-2611"}
    assert sf.settings(u, task, True) == {"max_tokens": fb.BENCH[task]["budget"]["on"],
                                          "expert_path": 6}
    assert fb.slot_context([task], True, runtime="k3k6-2611") == fb.slot_context([task], True)


# ---------------------------------------------------------------------------
# one answer: the budget sent, the reply counted and cut
# ---------------------------------------------------------------------------

def server_of(think_words: int, reply_words: int, sent: list):
    """a server's reply and its /tokenize with pieces, a word a token"""
    def post(url, key, body, item=True, timeout=None):
        sent.append((url, body))
        if url.endswith("/tokenize"):
            return {"tokens": [{"id": k, "piece": w}
                               for k, w in enumerate(re.findall(r"\s*\S+", body["content"]))]}
        think = " ".join(f"t{i}" for i in range(think_words))
        reply = " ".join(f"r{i}" for i in range(reply_words)) + "\nANSWER: B"
        return {"choices": [{"message": {"reasoning_content": think, "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": think_words + reply_words + 2}}
    return post


def test_a_request_sends_the_budget_and_never_the_reply_limit(monkeypatch):
    sent: list = []
    monkeypatch.setattr(served, "_post", server_of(300, 40, sent))
    a = served.ask(REC, "Which option?", sf.settings(REC, GPQA, True))
    (url, body), *counted = sent
    assert url == "http://box:8090/v1/chat/completions"
    assert sorted(body) == ["expert_path", "max_tokens", "messages", "thinking_budget_tokens"]
    assert body["thinking_budget_tokens"] == 1024 and body["max_tokens"] == 3100
    # the reply and the thinking counted by the server's own tokenizer
    assert [u for u, _ in counted] == ["http://box:8090/tokenize"] * 2
    assert all(b["with_pieces"] and b["add_special"] is False for _, b in counted)
    # a short answer kept whole
    assert a.app == {"thinking_tokens": 300, "reply_tokens": 42, "budget": 1024,
                     "budget_reached": False, "reply_limit": 2048}
    assert a.content.endswith("ANSWER: B") and a.finish == "stop"


def test_a_reply_past_2048_tokens_is_cut_there_and_ran_out_of_room(monkeypatch):
    monkeypatch.setattr(served, "_post", server_of(1024, 3000, []))
    a = served.ask(REC, "Which option?", sf.settings(REC, GPQA, True))
    assert a.app == {"thinking_tokens": 1024, "reply_tokens": 3002, "budget": 1024,
                     "budget_reached": True, "reply_limit": 2048, "reply_cut": True}
    assert a.content == " ".join(f"r{i}" for i in range(2048))
    assert "ANSWER" not in a and a.finish == "length"
    assert a.startswith("<think>\nt0 t1 ")                     # the thinking kept as it came


def test_thinking_off_counts_nothing_its_limit_the_requests(monkeypatch):
    sent: list = []
    monkeypatch.setattr(served, "_post", server_of(0, 40, sent))
    a = served.ask(REC, "Which option?", sf.settings(REC, GPQA, False))
    assert len(sent) == 1 and sent[0][1]["max_tokens"] == 2048
    assert "thinking_budget_tokens" not in sent[0][1] and a.app is None


def test_a_character_split_across_tokens_is_cut_whole_or_not_at_all(monkeypatch):
    pieces = [{"id": 1, "piece": "caf"}, {"id": 2, "piece": [195]}, {"id": 3, "piece": [169]},
              {"id": 4, "piece": " au"}]
    monkeypatch.setattr(served, "_pieces", lambda rec, text: pieces)
    assert served.app_limits(REC, "", "café au", {"reply_limit": 3})[0] == "café"
    assert served.app_limits(REC, "", "café au", {"reply_limit": 2})[0] == "caf"
    text, app = served.app_limits(REC, "", "café au", {"reply_limit": 4})
    assert text == "café au" and "reply_cut" not in app


# ---------------------------------------------------------------------------
# the box's budget check
# ---------------------------------------------------------------------------

def check_server(holds: bool):
    def post(base, path, body, timeout=900):
        if path == "/tokenize":
            return {"tokens": body["content"].split()}
        assert (body["thinking_budget_tokens"], body["max_tokens"], body["expert_path"]) == \
            (64, 464, 6)
        n = 64 if holds else 464
        return {"choices": [{"message": {"reasoning_content": " ".join(["w"] * n),
                                         "content": "The sum is 22140." if holds else ""},
                             "finish_reason": "stop" if holds else "length"}]}
    return post


def test_the_budget_check_passes_when_the_server_holds_it_and_says_why_when_not(monkeypatch):
    monkeypatch.setattr(path_check, "_post", check_server(True))
    ok = app_check.check("http://box:8090", 6, fb.APP_ROOM)
    assert ok["pass"] and ok["thinking_tokens"] == 64 and ok["why"] == ""
    assert app_check.words(ok).startswith("the thinking budget holds on expert path k6")
    monkeypatch.setattr(path_check, "_post", check_server(False))
    bad = app_check.check("http://box:8090", 6, fb.APP_ROOM)
    assert not bad["pass"]
    assert bad["why"] == ("the thinking ran to 464 tokens, past the budget of 64; "
                          "no reply followed the thinking")


# ---------------------------------------------------------------------------
# a box, end to end
# ---------------------------------------------------------------------------

def test_a_k6_box_checks_the_budget_then_asks_at_the_apps_limits(
        k3k6, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_THINK_WORDS", "1500")         # thinks past any budget
    assert run_k(k3k6, "p", "--runtime", RT, served_as=APP) == 0
    chk, budget, bench = asked(k3k6)
    assert len(chk) == 2 and "thinking_budget_tokens" not in chk[0]    # the paths' check first
    assert len(budget) == 1 and budget[0]["thinking_budget_tokens"] == 64
    assert bench and all(sorted(r) == ["expert_path", "max_tokens", "messages", "seed",
                                       "thinking_budget_tokens"]
                         and (r["expert_path"], r["max_tokens"], r["thinking_budget_tokens"])
                         == (6, 3100, 1024) for r in bench)
    b = rb.read(bundle_at(k3k6, "p"))
    srv = b["setup"]["server"]
    assert srv["runtime"] == RT and srv["paths_check"]["pass"]
    assert srv["app_check"]["pass"] and srv["app_check"]["thinking_tokens"] == 64
    task = b["setup"]["tasks"][TASK]
    assert task["budget"] == 3100 and task["sampling"] == {
        "expert_path": 6, "thinking_budget_tokens": 1024, "reply_limit": 2048}
    # each answer: the thinking stopped at the budget, the reply whole after it
    lines = answer_lines(bundle_at(k3k6, "p"))
    assert lines and all(x["app"]["thinking_tokens"] == 1024 and x["app"]["budget_reached"]
                         and "reply_cut" not in x["app"] for x in lines)
    # the import takes it onto its own row, registered with its runtime
    assert imf.runtime_problem(b["setup"], {"id": APP, "runtime": RT}) == ""
    assert ir.main([str(bundle_at(k3k6, "p")), "--by", "masein", "--register",
                    "step2388 (k3/k6), the apps' limits", "--file-sha256", k3k6["sha"]]) == 0
    assert served.get(APP)["runtime"] == RT


def test_a_reply_cut_at_2048_on_a_box_is_marked_as_it_came(k3k6, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_THINK_WORDS", "1500")
    monkeypatch.setenv("FAKE_LLAMA_REPLY_WORDS", "2100")         # its answer line past 2,048
    assert run_k(k3k6, "pc", "--runtime", RT, served_as=APP) == 0
    lines = answer_lines(bundle_at(k3k6, "pc"))
    assert lines and all(x["app"]["reply_cut"] and x["app"]["reply_tokens"] > 2048
                         and "ANSWER" not in x["answer"] for x in lines)


def test_a_k3_box_asks_with_2048_and_no_budget(k3k6):  # noqa: F811
    assert run_k(k3k6, "p3", "--runtime", RT, served_as=APP, thinking="off") == 0
    _, budget, bench = asked(k3k6)
    assert budget == []                                           # no budget check off
    assert bench and all(sorted(r) == ["expert_path", "max_tokens", "messages", "seed"]
                         and (r["expert_path"], r["max_tokens"]) == (3, 2048) for r in bench)
    setup = rb.read(bundle_at(k3k6, "p3", thinking=False))["setup"]
    assert "app_check" not in setup["server"]
    assert imf.runtime_problem(setup, {"id": APP, "runtime": RT}) == ""


def test_a_server_that_ignores_the_budget_stops_the_box_before_a_question(
        k3k6, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_THINK_WORDS", "1500")
    monkeypatch.setenv("FAKE_LLAMA_BUDGET", "ignore")
    with pytest.raises(SystemExit, match=re.escape(
            "the thinking budget doesn't hold on expert path k6: the thinking ran to 464 "
            "tokens, past the budget of 64; no reply followed the thinking. Nothing was asked")):
        run_k(k3k6, "pi", "--runtime", RT, served_as=APP)
    _, budget, bench = asked(k3k6)
    assert len(budget) == 1 and bench == []
    assert not bundle_at(k3k6, "pi").exists()


def test_the_import_refuses_a_thinking_bundle_without_a_budget_check_that_passed(
        k3k6, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FAKE_LLAMA_THINK_WORDS", "1500")
    assert run_k(k3k6, "pr", "--runtime", RT, served_as=APP) == 0
    setup = rb.read(bundle_at(k3k6, "pr"))["setup"]
    rec = {"id": APP, "runtime": RT}
    words = "its check that the server held the thinking budget, passed"
    s = json.loads(json.dumps(setup))
    s["server"].pop("app_check")
    assert words in imf.runtime_problem(s, rec)
    s["server"]["app_check"] = {**setup["server"]["app_check"], "pass": False}
    assert words in imf.runtime_problem(s, rec)
    assert imf.runtime_problem(setup, rec) == ""


# ---------------------------------------------------------------------------
# plan P
# ---------------------------------------------------------------------------

def test_plan_p_asks_the_six_benchmarks_once_in_each_mode_16_slots_a_box():
    plan = fx.PLANS["P"]
    assert fx.PLAN_RUNTIME["P"] == RT and fx.PLAN_GPUS["P"] == fx.PLAN_GPUS["U"]
    assert fx.PLAN_PINS["P"] == fx.PLAN_PINS["U"] and fx.PLAN_PINS["P"][NAME] == SHA
    have: dict = {}
    for steps in plan.values():
        assert len({s[0] for s in steps}) == 1                   # one path a box
        for th, tasks, shard, slots in steps:
            assert slots == 16
            for t in tasks:
                have.setdefault((th, t), []).append(shard)
    for (th, t), shards in have.items():
        n = fb.parse_shard(shards[0])[1] if shards[0] else 1
        assert sorted(shards) == sorted(f"{i}/{n}" if n > 1 else "" for i in range(1, n + 1)), t
    assert {t for th, t in have if th == "on"} == {t for th, t in have if th == "off"} == set(SIX)
    assert all(fx.slots_run(s, False, "P") == 16 for st in plan.values() for s in st)
    lines = fx.listing("P")
    assert lines[-1] == "6 boxes a build · 17.5 box-hours a build · the longest 4.5 h"
    assert all("(a guess, not measured on a box yet)" in ln for ln in lines if ln.startswith("  "))


def test_a_plan_p_line_runs_the_app_runtime_on_plan_us_files():
    a = fx.parser().parse_args(["P1", "--as", APP, "--gguf", f"hf://me/p@c/{NAME}",
                                "--server", "hf://me/p@c/llama-server-k3k6-3ad38b7.tar.gz",
                                "--price", "0.45"])
    argv = fx.argv_of("P1", 1, fx.box_of("P1")[0], a)
    assert argv[argv.index("--runtime") + 1] == RT and argv[argv.index("--thinking") + 1] == "on"
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "--only"] == [MMLU]
    assert argv[argv.index("--shard") + 1] == "1/3" and argv[argv.index("--slots") + 1] == "16"
    assert fx.pins_of(a, "P1") == fx.pins_of(a, "U35")
    off = fx.argv_of("P6", 1, fx.box_of("P6")[0], a)
    assert off[off.index("--thinking") + 1] == "off"
    assert {off[i + 1] for i, x in enumerate(off) if x == "--only"} == set(SIX) - {MMLU}


@pytest.mark.parametrize("label, model", [("P1", STEP), ("U35", APP)])
def test_plan_ps_boxes_alone_ask_at_the_apps_limits(label, model):
    with pytest.raises(SystemExit, match="plan P's boxes, and they alone, ask under the apps"):
        fx.main([label, "--as", model, "--gguf", f"hf://me/p@c/{NAME}", "--server",
                 "hf://me/p@c/llama-server-k3k6-3ad38b7.tar.gz", "--price", "0.45"])


def test_the_runbook_gives_plan_ps_line_and_its_assumptions():
    part = DOC.read_text()
    part = part[part.index("# P. The apps' limits: plan P (34)"):]
    part = part[:part.index("\n# ", 1)]
    assert "python scripts/frontier_box.py P1 --as served/step2388-k3k6-2611-app" in part
    assert f"--sha served/step2388-k3k6-2611-app={SHA}" in part
    assert "add no message when the thinking budget runs out" in part
    assert "2,048 on the reply alone" in part and "sample as the file stores" in part


@pytest.mark.parametrize("more", [["--showcase", "own.json", "--slots", "1"],
                                  ["--devicemark"], ["--continue", "cont-k6.jsonl"]])
def test_the_app_runtime_asks_the_frontier_benchmarks_alone(tmp_path, capsys, more):
    with pytest.raises(SystemExit):
        run_k({"gguf": tmp_path / NAME, "tarball": tmp_path / "t.tgz", "root": tmp_path}, "x",
              "--runtime", RT, *more, served_as=APP, only=False)
    assert ("--runtime k3k6-2611-app asks the Frontier benchmarks under the apps' limits alone: "
            "no --showcase, --devicemark or --continue") in capsys.readouterr().err
