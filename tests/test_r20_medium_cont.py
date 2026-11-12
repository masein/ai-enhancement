"""r20 (masein, 12 Nov): Medium's answers that ran out of room, continued to
32,768 as Instant's (k3's) were in U-cont, so the three modes are on one
protocol — box M5 of plan M, on Medium's own runtime (k3k6-2611-medium): each
request names path 6 and says thinking off and no presence penalty, as Medium
asked them; its answers are replies, continued as k3's, with no penalty window
and no penalty tests. The server is the stand-in (tests/fixtures/
fake_llama_server.py, FAKE_LLAMA_CONTINUE=1, FAKE_LLAMA_PATHS=1: a request's
enable_thinking over its path), the limits shrunk so its answers run out; the
questions invented; nothing is fetched and no model runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import continue_box as cb
import frontier as fb
import frontier_box as fx
import import_remote as ir
import remote_bundle as rb
from service import config
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import HARD, N, box, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import DOC, SHA, STEP, k3k6, run_k  # noqa: F401

RT = "k3k6-2611-medium"
MED = cb.MEDIUM
GPQA = "gpqa_diamond_epoch"
ROW = "served__step2388-k3k6-2611-medium"
OLD, NEW = 20, 40               # Medium's thinking-off limit and Instant's continued one, shrunk
SPEC = cb.spec_of(MED, False)
REC = {"id": MED, "name": MED, "runtime": RT, "base_url": "http://box:8090/v1", "pin": {}}
SAYS = {"expert_path": 6, "chat_template_kwargs": {"enable_thinking": False},
        "presence_penalty": 0.0}


def test_mediums_answers_are_continued_as_instants_to_32768_on_path_6_as_replies():
    assert MED == fx.CONT_MODEL[fx.CONT_MEDIUM][0] == "served/step2388-k3k6-2611-medium"
    assert (SPEC["key"], SPEC["limit"], SPEC["path"]) == ("medium", 32768, 6)
    # Instant's continued benchmarks, without the easy set Medium never asked
    assert SPEC["tasks"] == tuple(t for t in cb.spec_of(STEP, False)["tasks"]
                                  if t not in cb.EASY)
    assert cb.spec_of(MED, True) is None and cb.context(SPEC) == cb.context(
        cb.spec_of(STEP, False)) == 36864
    # its answers are replies, as k3's and the Q2's; k6's alone think
    assert not cb.thinks(SPEC) and cb.form_path(SPEC) is None
    assert not cb.thinks(cb.spec_of(STEP, False)) and not cb.thinks(cb.spec_of(cb.Q2, False))
    assert cb.thinks(cb.spec_of(STEP, True)) and cb.form_path(cb.spec_of(STEP, True)) == 6


@pytest.mark.parametrize("task", cb.SPECS[(cb.MEDIUM, False)]["tasks"])
def test_a_medium_continuation_says_what_medium_was_asked_with_its_reply_as_the_turn(task):
    asked = sf.settings(REC, task, False)
    assert {k: asked[k] for k in SAYS} == SAYS
    line = {"task": task, "id": "q7", "epoch": 1, "tokens": 8192}
    got = cb.request(line, cb.messages(task, "A question?"), "A reply so far", SPEC, asked)
    # everything it was asked with but its limit; no penalty window (its
    # presence penalty is none, said), the cut reply the assistant's turn
    assert {k: v for k, v in got.items() if k not in ("messages", "max_tokens", "seed")} == {
        k: v for k, v in asked.items() if k != "max_tokens"}
    assert "penalty_prompt_tokens" not in got
    assert got["messages"][-1] == {"role": "assistant", "content": "A reply so far"}
    assert (got["max_tokens"], got["seed"]) == (32768 - 8192, fb.seed_of(task, "q7+cont", 1))
    # a k6 continuation keeps its window and its thinking turn
    k6 = cb.request(line, cb.messages(task, "A question?"), "half a thou", cb.spec_of(STEP, True))
    assert k6["penalty_prompt_tokens"] == 8192 and "reasoning_content" in k6["messages"][-1]


def test_a_mediums_cut_reply_is_checked_as_one_reply():
    assert cb.continuity("A reply", "A reply, carried on", "reply", "", cb.form_path(SPEC)) == []
    assert cb.continuity("A reply", "Another reply", "reply", "", cb.form_path(SPEC)) == [
        "the reply doesn't begin with the cut-off one"]


def test_plan_ms_m5_continues_mediums_file_on_one_box_on_mediums_runtime():
    assert fx.PLANS["M"]["M5"] == [("off", (fx.CONT_MEDIUM,), "", 30)]
    assert fx.CONT_SIZE[("off", fx.CONT_MEDIUM)]["answers"] == 155
    listed = fx.listing("M")
    m5 = next(ln for ln in listed if "Medium's answers that ran out of room" in ln)
    assert "thinking off (k6)" in m5 and "30 slots of 36,864 tokens" in m5
    assert "281 tokens a second written" in m5 and listed[-1].startswith("5 boxes a build")
    sha = "a" * 64
    a = fx.parser().parse_args([
        "M5", "--as", MED, "--gguf",
        "hf://o/r@fe12fa27825b3a0cc6cde3dedec3e838bd3e6de4/step2388.k3k6.gguf", "--server",
        "hf://o/r@fe12fa27825b3a0cc6cde3dedec3e838bd3e6de4/llama-server-k3k6-3ad38b7.tar.gz",
        "--continue", "/workspace/cont-medium.jsonl", "--continue-sha256", sha])
    cmd = fx.argv_of("M5", 1, fx.box_of("M5")[0], a)
    assert cmd[cmd.index("--runtime") + 1] == RT and cmd[cmd.index("--thinking") + 1] == "off"
    assert cmd[cmd.index("--continue") + 1] == "/workspace/cont-medium.jsonl"
    assert f"cont-medium.jsonl={sha}" in cmd and "--part" not in cmd
    assert (cmd[cmd.index("--slots") + 1], cmd[cmd.index("--min-slots") + 1]) == ("30", "15")


def test_m5_without_its_file_is_refused(capsys):
    with pytest.raises(SystemExit):
        fx.main(["M5", "--as", MED, "--gguf", "hf://o/r@c/step2388.k3k6.gguf", "--server",
                 "hf://o/r@c/llama-server-k3k6-3ad38b7.tar.gz"])
    assert "give --continue" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# on a box, and home: Medium's answers cut, continued, imported
# ---------------------------------------------------------------------------

@pytest.fixture
def med(k3k6, monkeypatch):  # noqa: F811
    """a Medium box whose stand-in writes 22 tokens a GPQA answer, Medium's
    limit 20 and the continuation's 40"""
    monkeypatch.setenv("FAKE_LLAMA_CONTINUE", "1")
    monkeypatch.setenv("FAKE_LLAMA_OUT_TOKENS", "22")
    monkeypatch.setitem(fb.BENCH[GPQA]["budget"], "off", OLD)
    monkeypatch.setitem(cb.SPECS[(MED, False)], "limit", NEW)
    return k3k6


def cont_requests(b: dict) -> list[dict]:
    return [r for r in requests(b) if (r.get("messages") or [{}])[-1].get("role") == "assistant"
            and "verbose" in r]


def test_mediums_cut_answers_are_continued_on_its_runtime_and_imported(med):
    # plan M's box: Medium asks GPQA Diamond, every answer cut at OLD
    assert run_k(med, "m3", "--runtime", RT, served_as=MED, thinking="off") == 0
    got = next((med["root"] / "m3").glob("frontier-*.tar.gz"))
    assert ir.main([str(got), "--by", "masein", "--register", "step2388 Medium (k3k6 runtime)",
                    "--file-sha256", med["sha"]]) == 0
    row = config.OUT_DIR / ROW
    first = sf.read_answers(sf.task_dir(row, GPQA) / sf.ANSWERS)
    assert len(first) == N * 4 and all(a["finish"] == "length" and a["tokens"] == OLD
                                       for a in first.values())
    assert all("<think>" not in a["answer"] for a in first.values())
    # the server's inputs: Medium's file alone
    made = cb.inputs(med["root"] / "u-cont", say=lambda x: None, models=[MED])
    assert set(made) == {"medium"}
    f, sha = Path(made["medium"]["file"]), made["medium"]["sha256"]
    assert f.name == "cont-medium.jsonl" and len(cb.read_input(f)) == N * 4
    assert {r["row"] for r in cb.read_input(f)} == {ROW}
    # M5: its settings check first, then each answer continued as Medium asked it
    before = len(requests(med))
    assert run_k(med, "m5", "--runtime", RT, "--continue", str(f), "--pin", f"{f.name}={sha}",
                 served_as=MED, thinking="off", only=False) == 0
    checked = [r for r in requests(med)[before:] if r not in cont_requests(med)
               and "messages" in r and "max_tokens" in r]
    assert checked and all(r["expert_path"] == 6 and r["chat_template_kwargs"] ==
                           {"enable_thinking": False} for r in checked)
    asked = cont_requests(med)
    assert len(asked) == N * 4
    assert all(sorted(r) == ["chat_template_kwargs", "expert_path", "max_tokens", "messages",
                             "presence_penalty", "seed", "stream", "verbose"] for r in asked)
    assert {(r["max_tokens"], r["expert_path"], r["presence_penalty"]) for r in asked} == {
        (NEW - OLD, 6, 0.0)}
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False}
               and set(r["messages"][-1]) == {"role", "content"} for r in asked)
    p = next((med["root"] / "m5").glob("frontier-cont-*.tar.gz"))
    assert p.name == "frontier-cont-served__step2388-k3k6-2611-medium-thinking-off.tar.gz"
    b = rb.read(p)
    cont = b["setup"]["continuation"]
    assert (cont["limit"], cont["path"], cont["key"]) == (NEW, 6, "medium")
    assert "tests" not in cont                     # no penalty tests: it doesn't think
    assert b["setup"]["runtime"] == RT and b["setup"]["thinking"] == "off"
    rows = [json.loads(x) for x in
            b["files"][f"results/{ROW}/{GPQA}_0shot/frontier/{cb.DONE}"].decode().splitlines()]
    r = rows[0]
    assert r["status"] == "continued" and r["form"] == "reply"
    assert (r["was"]["tokens"], r["new_tokens"], r["tokens"], r["finish"]) == (OLD, 2, 22, "stop")
    assert r["penalty_prompt_tokens"] is None and r["settings"]["presence_penalty"] == 0.0
    # home: imported onto Medium's row, the old limit's score on record
    said: list[str] = []
    assert ir.import_bundle(p, "masein", said.append) == 0
    now = sf.read_answers(sf.task_dir(row, GPQA) / sf.ANSWERS)
    assert len(now) == N * 4 and all(
        a["tokens"] == 22 and a["finish"] == "stop" and a["continued"]["from"]["tokens"] == OLD
        and "<think>" not in a["answer"] for a in now.values())
    setup = json.loads((sf.task_dir(row, GPQA) / sf.SETUP).read_text())
    assert setup["budget"] == NEW and setup["continued"]["from_budget"] == OLD
    res = json.loads(next(sf.task_dir(row, GPQA).glob("results_*.json")).read_text())
    lim = res["frontier"]["limits"]
    assert res["results"][GPQA]["acc,none"] == pytest.approx((N - len(HARD)) / N)
    assert lim["at_old_limit"]["score"] == 0.0 and lim["at_old_limit"]["ran_out"] == N * 4
    assert (lim["continued"], lim["still_ran_out"]) == (N * 4, 0)
    # and the next inputs file holds nothing more to continue
    again = cb.inputs(med["root"] / "u-cont", say=lambda x: None, models=[MED])
    assert cb.read_input(Path(again["medium"]["file"])) == []


def test_a_continuation_on_instants_runtime_is_refused_onto_medium(med):
    assert run_k(med, "m3", "--runtime", RT, served_as=MED, thinking="off") == 0
    got = next((med["root"] / "m3").glob("frontier-*.tar.gz"))
    assert ir.main([str(got), "--by", "masein", "--register", "step2388 Medium (k3k6 runtime)",
                    "--file-sha256", med["sha"]]) == 0
    made = cb.inputs(med["root"] / "u-cont", say=lambda x: None, models=[MED])
    f, sha = Path(made["medium"]["file"]), made["medium"]["sha256"]
    # the file is Medium's: a box asking as Instant (k3k6-2611, k3) won't take it
    with pytest.raises(SystemExit, match="not of served__step2388-k3k6-2611"):
        run_k(med, "c4", "--continue", str(f), "--pin", f"{f.name}={sha}", thinking="off",
              only=False)


def test_the_runbook_gives_m5s_line_and_the_inputs_step():
    part = DOC.read_text()
    part = part[part.index("# M. The app's Medium mode: plan M (r20)"):]
    part = part[:part.index("\n# ", 1)]
    assert "continue_box.py inputs --as served/step2388-k3k6-2611-medium" in part
    assert f"python scripts/frontier_box.py M5 --as {MED}" in part
    assert "--continue /workspace/cont-medium.jsonl --continue-sha256" in part
