"""15.7: a reply read as DeviceMark reads one (scoring v2). With the thinking
on, only the answer after the closed thinking is scored, and a reply capped
inside its thinking has no answer on all three tests — as their Youtu and
Nanbeige rows show; with it off, the whole visible reply is the answer, as
their Qwen3.5 rows show. Replies shaped as each family writes them — Qwen3.5
(its template opens <think>), Youtu (writes <think> itself), Gemma 4
(<|channel>thought … <channel|>, kept by lm_eval_whole.py) and a served model
(the server splits the thinking off) — each closed, capped inside the
thinking, and capped inside the answer. Then the rows scored before 15.7:
scored again where their saved text allows, and marked as needing a new run
where it doesn't. The battery is invented; nothing runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import devicemark as dm
from service import config, runner
from service import devicemark as sdm
from test_12q_devicemark_runs import write_items

THINKING = "Let me see, carefully, step by step: maybe \\boxed{B}, or \\boxed{3}"
ANSWER = {"ifeval": "a note without any commas", "mmlu_pro": "So the answer is \\boxed{A}",
          "math": "It is \\boxed{\\frac{1}{2}}"}
FAMILIES = {
    "qwen": {"on": True, "marks": list(dm.THINK), "opens": True},
    "youtu": {"on": True, "marks": list(dm.THINK), "opens": False},
    "gemma": {"on": True, "marks": list(dm.CHANNEL), "opens": False},
}


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    return write_items(tmp_path_factory.mktemp("dm") / "items.jsonl")


def key_of(bench: str) -> str:
    """the first key of a test — for MMLU-Pro, one whose answer is A (an even id)"""
    return next(k for b, k in dm.keys_for("full")
                if b == bench and (b != "mmlu_pro" or int(k) % 2 == 0))


def reply(family: str, bench: str, state: str) -> dict:
    """one reply as the family writes it, kept whole: closed, capped inside
    the thinking, or capped inside the answer (cut short)"""
    ans = ANSWER[bench] if state != "cut answer" else ANSWER[bench] + " and then more and"
    if family == "qwen":                       # the template opened <think>: none in the reply
        text = THINKING + ("" if state == "cut thinking" else "\n</think>\n\n" + ans)
        text += "<|im_end|>" if state == "closed" else ""
    elif family == "youtu":
        text = "<think>\n" + THINKING + ("" if state == "cut thinking" else "\n</think>\n\n" + ans)
    else:
        text = "<|channel>thought\n" + THINKING + (
            "" if state == "cut thinking" else "<channel|>" + ans)
        text += "<turn|>" if state == "closed" else ""
    return {"bench": bench, "key": key_of(bench), "text": text, "form": "whole",
            "capped": state != "closed", "gen_tokens": 4096 if state != "closed" else 900}


def served(bench: str, state: str) -> dict:
    """one reply as ask() keeps a served one: the server's split, put back"""
    think = THINKING
    content = "" if state == "cut thinking" else (
        ANSWER[bench] + (" and then more and" if state == "cut answer" else ""))
    return {"bench": bench, "key": key_of(bench), "answer": content,
            "text": f"<think>\n{think}\n</think>\n\n{content}", "thinking_chars": len(think),
            "capped": state != "closed", "gen_tokens": 4096 if state != "closed" else 900}


STATES = ("closed", "cut thinking", "cut answer")


@pytest.mark.parametrize("family", [*FAMILIES, "served"])
@pytest.mark.parametrize("state", STATES)
def test_each_family_is_read_and_scored_on_its_answer(items, family, state):
    reading = FAMILIES.get(family, {"on": True})
    records = [served(b, state) if family == "served" else reply(family, b, state)
               for b in dm.BENCHES]
    scored = {s["bench"]: s for s in dm.score_items(records, items, reading)}
    for b in dm.BENCHES:
        s, got = scored[b], dm.read_reply(records[dm.BENCHES.index(b)], reading)
        assert "maybe \\boxed{B}" in got["thinking"] and "<" not in got["thinking"][:2]
        if state == "cut thinking":
            # no answer, on every test: the thinking is never read
            assert (got["answer"], got["closed"], s["cap_in"]) == ("", False, "thinking")
            assert not s["answered"] and not s["correct"]
        else:
            assert got["answer"].startswith(ANSWER[b]) and got["closed"] is True
            assert s["scored_answer"] == got["answer"]
            assert s["cap_in"] == (None if state == "closed" else "answer")
    if state != "cut thinking":
        # the answer, not the thinking's \boxed{B} or \boxed{3}, nor its commas
        assert scored["mmlu_pro"]["parsed"] == "A" and scored["mmlu_pro"]["correct"]
        assert scored["math"]["correct"]
        assert scored["ifeval"]["ifeval"]["strict"]["inst"] == [True]


def test_a_thinking_on_reply_with_no_thinking_is_all_answer():
    # a model whose template doesn't open the thinking, that chose not to think
    got = dm.split_reply("a note without any commas", True, dm.THINK, opens=False)
    assert (got["answer"], got["closed"]) == ("a note without any commas", None)


def test_thinking_off_the_whole_reply_is_the_answer(items):
    # DeviceMark's Qwen3.5 rows, thinking off: a "Thinking Process:" preamble
    # and a literal </think> are part of the answer, and are scored
    text = ("Thinking Process:\n1. Analyse the request, then the rules.\n</think>\n\n"
            "a note without any commas")
    rec = {"bench": "ifeval", "key": key_of("ifeval"), "text": text, "gen_tokens": 300}
    got = dm.read_reply(rec, {"on": False})
    assert got["answer"] == text and got["closed"] is None
    v2 = dm.score_items([rec], items, {"on": False})[0]
    v1 = dm.score_items([rec], items)[0]
    assert v2["ifeval"]["strict"]["inst"] == [False]           # the preamble's commas count
    assert v1["ifeval"]["strict"]["inst"] == [True]            # v1 read only after </think>
    # an empty thinking block a template or a server leaves is markup, not answer
    assert dm.split_reply("<think>\n\n</think>\n\nhi<|im_end|>", False)["answer"] == "hi"
    assert dm.split_reply("<|channel>thought\n<channel|>hi", False, dm.CHANNEL)["answer"] == "hi"


# ---------------------------------------------------------------------------
# replies saved before 15.7: cut by lm_eval at the end of the thinking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,text,capped,want", [
    # Qwen3.5: lm_eval kept the answer after </think> — or, capped inside the
    # thinking, the whole of it with no marker (4B: "Thinking Process: …")
    ("qwen", "a note without any commas", False, ("answer", True)),
    ("qwen", "Thinking Process:\n1. **Analyze the Request:** …", True, ("thinking", False)),
    # Youtu writes <think> itself: an unclosed reply still begins with it
    ("youtu", "<think>\nstill thinking, when the cap came", True, ("thinking", False)),
    ("youtu", "a note without any commas", False, ("answer", True)),
    # Gemma 4: the decode dropped <|channel> and <channel|>: thinking and answer can't be told
    ("gemma", "thought\nLet me see.The answer is \\boxed{A}", False, ("unreadable", None)),
])
def test_a_reply_saved_before_15_7_is_read_where_its_text_allows(family, text, capped, want):
    got = dm.read_reply({"text": text, "capped": capped}, FAMILIES[family])   # no form: cut
    kind, closed = want
    if kind == "unreadable":
        assert got["readable"] is False and got["answer"] == ""
    else:
        assert got["closed"] is closed and got["readable"]
        assert (got["answer"] == text) is (kind == "answer")


# ---------------------------------------------------------------------------
# the rows on the board: scored again, or marked for a new run
# ---------------------------------------------------------------------------

def v1_row(model: str, thinking: bool, runtime: str, records: list[dict], items) -> Path:
    """a row as the reading before 15.7 left it"""
    d = config.OUT_DIR / (model.replace("/", "__") + ("__thinking" if thinking else ""))
    d.mkdir(parents=True, exist_ok=True)
    dm.mark(d, items, {"model": model, "runtime": runtime, "thinking": thinking}, records)
    return d


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "full")
    monkeypatch.setattr(config, "DM_ITEMS", tmp_path / "items.jsonl")
    return write_items(config.DM_ITEMS)


def cut_records(items, n_capped: int) -> list[dict]:
    """every item answered right — the first n of each test only inside a
    thinking the cap cut, as lm_eval kept it before 15.7 (all of it, no
    marker): the Qwen3.5 case, where v1 scored such thinking as the answer"""
    right = {"ifeval": lambda k: "a note without any commas",
             "mmlu_pro": lambda k: "\\boxed{" + items[("mmlu_pro", k)]["answer"] + "}",
             "math": lambda k: "\\boxed{\\frac{1}{2}}"}
    out = []
    for b, k in dm.keys_for("full"):
        i = sum(1 for r in out if r["bench"] == b)
        cut = i < n_capped
        out.append({"bench": b, "key": k, "capped": cut, "gen_tokens": 4095 if cut else 600,
                    "text": ("Thinking Process: it looks like " if cut else "") + right[b](k)})
    return out


def test_a_qwen_row_is_scored_again_from_what_it_saved(board):
    hf = "hf transformers (lm_eval)"
    d = v1_row("Qwen/Qwen3.5-4B", True, hf, cut_records(board, 40), board)
    before = json.loads((d / dm.OUT_NAME).read_text())
    assert before["setup"]["scoring"] == "v1"
    assert dm.provisional_why(before["setup"]).startswith("scored by the reading before 15.7")
    r = sdm.rescore(d)
    assert r["status"] == "re-scored"
    # the 40 a test capped inside the thinking were scored right before, from
    # the thinking; now they're no answer
    assert r["before"] == {"composite": 1.0, "ifeval": 1.0, "mmlu_pro": 1.0, "math": 1.0}
    assert r["after"]["ifeval"] == round(260 / 300, 4)
    assert r["after"]["mmlu_pro"] == round(156 / 196, 4) and r["after"]["math"] == 0.6
    assert r["caps"]["ifeval"] == {"capped": 40, "capped_in_thinking": 40}
    after = json.loads((d / dm.OUT_NAME).read_text())
    assert after["setup"]["scoring"] == dm.SCORING and dm.provisional_why(after["setup"]) is None
    assert after["setup"]["reading"] == {"on": True, "marks": list(dm.THINK), "opens": True}
    assert after["benches"]["ifeval"]["capped_in_thinking"] == 40
    # and again changes nothing
    assert sdm.rescore(d)["status"] == f"{dm.SCORING} already"


def test_a_gemma_row_keeps_its_numbers_and_needs_a_new_run(board):
    hf = "hf transformers (lm_eval)"
    recs = [{"bench": b, "key": k, "text": "thought\nLet me see." + ANSWER[b], "capped": False,
             "gen_tokens": 700} for b, k in dm.keys_for("full")]
    d = v1_row("google/gemma-4-E2B-it", True, hf, recs, board)
    before = json.loads((d / dm.OUT_NAME).read_text())
    r = sdm.rescore(d)
    assert r["status"] == "needs a new run" and r["after"] is None
    assert r["why"].startswith("needs a new run: 596 of its answers can't be told from its "
                               "thinking")
    after = json.loads((d / dm.OUT_NAME).read_text())
    assert after["composite"] == before["composite"]                      # kept
    assert dm.provisional_why(after["setup"]) == r["why"]
    rows = {x["id"]: x for x in dm.rows(config.OUT_DIR)}
    assert rows["google/gemma-4-E2B-it · thinking"]["provisional"] == r["why"]


def test_a_served_row_reads_the_servers_split_and_says_what_was_cut(board, capsys):
    import rescore_devicemark as rs
    recs = []
    for b, k in dm.keys_for("full"):
        i = sum(1 for r in recs if r["bench"] == b)
        recs.append({**served(b, "cut thinking" if i < 5 else "cut answer" if i < 8 else
                              "closed"), "key": k})
    v1_row("served/qwen36-phone", True, "llama-server", recs, board)
    assert rs.main([]) == 0
    out = capsys.readouterr().out
    assert "served/qwen36-phone · thinking on |" in out and "re-scored" in out
    assert ("capped: IFEval 8 (5 inside the thinking), MMLU-Pro 8 (5 inside the thinking), "
            "MATH 8 (5 inside the thinking) · 9 cut with an answer: 9 after thinking the "
            "server split off") in out


# ---------------------------------------------------------------------------
# a new run: replies kept whole, and said so beside them
# ---------------------------------------------------------------------------

def test_a_thinking_on_devicemark_task_runs_lm_eval_whole(tmp_path):
    cmd = runner.lm_eval_cmd("pretrained=x,enable_thinking=True,think_end_token=</think>",
                             "dm_math", 0, 1, tmp_path, chat=True, whole=True)
    assert cmd[:2] == [sys.executable, str(runner.WHOLE_SCRIPT)]
    assert cmd[cmd.index("--model") + 1] == "hf-whole"
    assert runner.WHOLE_SCRIPT.exists()
    plain = runner.lm_eval_cmd("pretrained=x", "dm_math", 0, 1, tmp_path, chat=True)
    assert plain[0] == "lm_eval" and plain[plain.index("--model") + 1] == "hf"


def test_answers_kept_cut_are_set_aside_before_a_whole_run(tmp_path, monkeypatch):
    task_out = tmp_path / "dm_math_0shot"
    (task_out / "lm-cache").mkdir(parents=True)
    (task_out / "lm-cache" / "answers_rank0.db").write_bytes(b"cut")
    runner.keep_whole(task_out)
    assert dm.reply_form(task_out) == "whole"
    assert not (task_out / "lm-cache").exists()
    assert [p.name.startswith("lm-cache-cut-") for p in task_out.iterdir()
            if p.is_dir()] == [True]
    # a whole cache stays: the same command carries on from it
    (task_out / "lm-cache").mkdir()
    (task_out / "lm-cache" / "answers_rank0.db").write_bytes(b"whole")
    runner.keep_whole(task_out)
    assert (task_out / "lm-cache" / "answers_rank0.db").read_bytes() == b"whole"


def test_the_catalogue_knows_which_templates_open_the_thinking():
    from service import catalog
    assert {m: catalog.thinking_of(m, None)["opens"] for m in (
        "Qwen/Qwen3.5-4B", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "tencent/Youtu-LLM-2B",
        "google/gemma-4-E2B-it", "Nanbeige/Nanbeige4.1-3B")} == {
        "Qwen/Qwen3.5-4B": True, "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": True,
        "tencent/Youtu-LLM-2B": False, "google/gemma-4-E2B-it": False,
        "Nanbeige/Nanbeige4.1-3B": False}
    assert sdm.reading_of("google/gemma-4-E2B-it", True) == {
        "on": True, "marks": list(dm.CHANNEL), "opens": False}
