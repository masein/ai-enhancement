"""31.3 (brief 31d): step2388 at the model team's token limits — each answer
plan U cut at its limit continued from where it stopped, on a rented box, as
one reply, to 98,304 tokens on k6 and 32,768 on k3; the box's check of five
before U-cont; the import onto the rows they came from, the old limit's score
on record; and plan C's box lines. The server is the stand-in (tests/fixtures/
fake_llama_server.py, FAKE_LLAMA_CONTINUE=1: a reply a list of tokens, a cut
one continued the rest of it), the limits shrunk so its answers run out; the
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
import remote_gguf as rg
import report_lm_eval as report
from service import config
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import HARD, N, box, requests  # noqa: F401
from test_20_step4000 import card  # noqa: F401
from test_31_1_k3k6 import STEP, k3k6, run_k, step_kept  # noqa: F401

GPQA = "gpqa_diamond_epoch"
ROW = "served__step2388-k3k6-2611__thinking"
OLD, NEW = 20, 40               # plan U's limit and the team's, shrunk for the stand-in
K6, K3, Q2 = cb.spec_of(STEP, True), cb.spec_of(STEP, False), cb.spec_of(cb.Q2, False)
# 31.3b: how U40 asked the Q2 (service/frontier.settings on card-qwen-k3k6)
Q2_ASKED = {"max_tokens": 4096, "temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0,
            "presence_penalty": 1.5, "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}}


# ---------------------------------------------------------------------------
# the method: the forms an answer is continued in, its two checks, the request
# ---------------------------------------------------------------------------

def test_an_answer_is_split_as_service_served_keeps_it():
    assert cb.split("<think>\nA thought.\n\n</think>\n\nThe reply") == (
        "A thought.\n", "The reply", True)
    assert cb.split("<think>\nCut mid-thought\n</think>\n\n") == ("Cut mid-thought", "", True)
    assert cb.split("A reply alone") == ("", "A reply alone", False)


def test_the_forms_put_the_likeliest_first_and_give_back_what_the_model_wrote():
    # cut in its thinking — or right after closing it
    assert cb.forms("<think>\nhalf a thou\n</think>\n\n", 6) == [
        ("open", "half a thou"),
        ("closed, 2 newlines after </think>", "half a thou</think>\n\n"),
        ("closed, 1 newline after </think>", "half a thou</think>\n"),
        ("closed, nothing after </think>", "half a thou</think>")]
    # its thinking closed: the newline the parser kept before </think> stays,
    # and no other is added (the server's own reply form adds one)
    assert cb.forms("<think>\nDone.\n\n</think>\n\nThe ans", 6)[0] == (
        "closed, 2 newlines after </think>", "Done.\n</think>\n\nThe ans")
    assert cb.forms("A reply, cut", 3) == [("reply", "A reply, cut")]


def test_a_continuation_carries_the_turn_its_limit_seed_and_path_and_on_k6_the_window():
    line = {"task": GPQA, "id": "rec001", "epoch": 2, "tokens": 81920}
    msgs = cb.messages(GPQA, "A question?")
    k6 = cb.request(line, msgs, "thinking so far", K6)
    assert sorted(k6) == ["expert_path", "max_tokens", "messages", "penalty_prompt_tokens",
                          "seed"]
    assert k6["messages"][-1] == {"role": "assistant", "reasoning_content": "thinking so far",
                                  "content": ""}
    assert (k6["max_tokens"], k6["penalty_prompt_tokens"], k6["expert_path"]) == (
        98304 - 81920, 81920, 6)
    # a fresh draw, its own seed, the same on every machine
    assert k6["seed"] == fb.seed_of(GPQA, "rec001+cont", 2) != fb.seed_of(GPQA, "rec001", 2)
    k3 = cb.request({**line, "tokens": 8192}, msgs, "a reply so far", K3)
    assert sorted(k3) == ["expert_path", "max_tokens", "messages", "seed"]       # no penalty
    assert k3["messages"][-1] == {"role": "assistant", "content": "a reply so far"}
    assert k3["max_tokens"] == 32768 - 8192
    # HLE's system message, as it was asked
    assert cb.messages("hle_text_cais", "Q")[0]["role"] == "system"
    assert (K6["limit"], K3["limit"], cb.context(K6), cb.context(K3)) == (
        98304, 32768, 131072, 36864)


class Stub(cb.Server):
    """a server whose template and tokenizer are the stand-in's: words, the
    prompt ending in a space, a final turn written straight after it"""

    def __init__(self, extra_newline: bool = False):
        super().__init__("http://stub")
        self.extra = extra_newline

    def post(self, path, body, timeout=None):
        if path == "/apply-template":
            msgs = body["messages"]
            last = msgs[-1]
            q = msgs[:-1] if last["role"] == "assistant" else msgs
            p = "<bos> " + " ".join(m["content"] for m in q) + " <turn|> "
            if last["role"] == "assistant":
                text = last.get("reasoning_content") or last.get("content") or ""
                p += text.replace("</think>", "\n</think>") if self.extra else text
            return {"prompt": p}
        if path == "/tokenize":
            return {"tokens": list(range(len(body["content"].split())))}
        raise AssertionError(path)


def test_the_form_whose_prompt_gives_back_the_replys_tokens_is_taken():
    msgs = cb.messages(GPQA, "Which one?")
    p0 = "<bos> Which one? <turn|> "
    open_ = {"task": GPQA, "id": "q", "epoch": 0, "answer": "<think>\nw1 w2 w3\n</think>\n\n",
             "tokens": 3, "prompt_tokens": len(p0.split())}
    got = cb.prepare(Stub(), open_, msgs, K6)
    assert (got["form"], got["text"], got["prompt_tokens"], got["continued_tokens"]) == (
        "open", "w1 w2 w3", 4, 7)
    closed = {**open_, "answer": "<think>\nw1 w2\n\n</think>\n\nThe options", "tokens": 5}
    got = cb.prepare(Stub(), closed, msgs, K6)
    assert got["form"] == "closed, 2 newlines after </think>"
    assert got["text"] == "w1 w2\n</think>\n\nThe options"


def test_an_answer_whose_text_doesnt_give_back_its_tokens_isnt_continued():
    msgs = cb.messages(GPQA, "Which one?")
    line = {"task": GPQA, "id": "q", "epoch": 0, "answer": "<think>\nw1 w2 w3\n</think>\n\n",
            "tokens": 5, "prompt_tokens": 4}
    why = cb.prepare(Stub(), line, msgs, K6)["why"]
    assert why.startswith("the reply's text doesn't give back its 5 tokens") and "open: 3" in why
    # another question or template: its prompt's tokens differ
    assert "not the same question" in cb.prepare(Stub(), {**line, "prompt_tokens": 9}, msgs,
                                                 K6)["why"]
    # a server that writes a reply's turn with a newline of its own: the text
    # isn't the question's and the reply's, character for character
    closed = {**line, "answer": "<think>\nw1\n\n</think>\n\nThe", "tokens": 2}
    assert "the prompt isn't the question's and the reply's text" in cb.prepare(
        Stub(extra_newline=True), closed, msgs, K6)["why"]


@pytest.mark.parametrize("final, form, wrong", [
    ("<think>\nw1 w2 w3 w4\n</think>\n\nANSWER: A", "open", None),
    ("<think>\nw9 w2 w3 w4\n</think>\n\nANSWER: A", "open", "doesn't begin"),
    ("<think>\nw1 w2 w3 <think> again\n</think>\n\nA", "open", "opens its thinking again"),
    ("<think>\nw1 w2 w3 <|im_start|>user\n</think>\n\n", "open", "writes a turn"),
    ("<think>\nw1 w2 w3 Which of these four options is the right one, A to D, and why\n"
     "</think>\n\n", "open", "writes the question again"),
])
def test_a_continued_answer_is_held_to_being_one_reply(final, form, wrong):
    was = "<think>\nw1 w2 w3\n</think>\n\n"
    got = cb.continuity(was, final, form, "Which of these four options is the right one, A to D,"
                        " and why", 6)
    assert (got == []) if wrong is None else any(wrong in x for x in got)
    # a closed one: the same thinking, its reply carried on
    assert cb.continuity("<think>\nT\n</think>\n\nThe op", "<think>\nT\n</think>\n\nThe options",
                         "closed, 2 newlines after </think>", "", 6) == []
    assert cb.continuity("A reply", "A reply, carried on", "reply", "", 3) == []


def test_each_box_gets_an_even_share_by_the_room_left():
    lines = [{"task": GPQA, "id": f"q{k}", "epoch": 0, "tokens": t, "answer": ""}
             for k, t in enumerate([90, 10, 50, 70, 30, 20])]
    parts = [cb.part_of(lines, (i, 3), 100) for i in (1, 2, 3)]
    assert sorted(r["id"] for p in parts for r in p) == sorted(r["id"] for r in lines)
    room = [sum(100 - r["tokens"] for r in p) for p in parts]
    assert max(room) - min(room) <= 40
    assert cb.part_of(lines, None, 100)[0]["id"] == "q1"            # the most room first


def test_the_inputs_file_is_held_to_its_form(tmp_path):
    f = tmp_path / "cont-k6.jsonl"
    good = {"task": GPQA, "id": "q", "epoch": 0, "answer": "x", "tokens": 3}
    f.write_text(json.dumps(good) + "\n")
    assert cb.read_input(f) == [good]
    f.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(ValueError, match="given twice"):
        cb.read_input(f)
    f.write_text(json.dumps({**good, "task": "no_such"}) + "\n")
    with pytest.raises(ValueError, match="isn't an answer to continue"):
        cb.read_input(f)


# ---------------------------------------------------------------------------
# penalty_prompt_tokens' test
# ---------------------------------------------------------------------------

class Probe(cb.Server):
    """a server for the penalty's test: greedy replies of words, continued as
    the rest; probabilities with the presence penalty on the tokens it counts"""

    def __init__(self, counts: str = "window", part_at: int | None = None):
        super().__init__("http://probe")
        self.counts, self.part_at = counts, part_at

    def post(self, path, body, timeout=None):
        if path == "/apply-template":
            msgs = body["messages"]
            p = "<bos> " + msgs[0]["content"] + " <turn|> "
            return {"prompt": p + (msgs[1]["reasoning_content"] if len(msgs) > 1 else "")}
        if path == "/tokenize":
            return {"tokens": [int(cb.answer_sha(w)[:6], 16) for w in body["content"].split()]}
        if path == "/v1/chat/completions":
            msgs = body["messages"]
            done = len(msgs[1]["reasoning_content"].split()) if len(msgs) > 1 else 0
            words = [f" w{done + i}" for i in range(body["max_tokens"])]
            if self.part_at is not None and len(msgs) > 1 and body.get("penalty_prompt_tokens"):
                words[self.part_at] = " parted"
            return {"choices": [{"message": {"reasoning_content": "".join(words).strip(),
                                             "content": ""},
                                 "logprobs": {"content": [
                                     {"token": w, "logprob": -0.1,
                                      "top_logprobs": [{"token": w, "logprob": -0.1},
                                                       {"token": "x", "logprob": -0.2}]}
                                     for w in words]}}],
                    "usage": {"completion_tokens": len(words),
                              "prompt_tokens": 9 + done}}
        if path == "/completion":
            ids = [int(cb.answer_sha(w)[:6], 16) for w in body["prompt"].split()]
            n = body["penalty_prompt_tokens"]
            if self.counts == "all":
                n = -1
            tail = set(ids) if n < 0 else set(ids[len(ids) - n:]) if n else set()
            first = [i for i in dict.fromkeys(ids) if i not in set(ids[len(ids) - max(n, 0):])]
            last = [i for i in dict.fromkeys(reversed(ids)) if i not in set(ids[:len(ids) - n])] \
                if n > 0 else [ids[-1]]
            cands = first[:2] + last[:2] + [1, 2]
            raw = dict(zip(cands, [-0.4, -1.1, -0.7, -1.5, -0.9, -1.8]))
            if not body["post_sampling_probs"]:
                return {"completion_probabilities": [{"top_logprobs": [
                    {"id": i, "logprob": x} for i, x in raw.items()]}]}
            import math
            temp = float(body.get("temperature", 0.8))
            post = {i: (x - (0.75 if i in tail else 0)) / temp for i, x in raw.items()}
            z = math.log(sum(math.exp(x) for x in post.values()))
            return {"completion_probabilities": [{"top_probs": [
                {"id": i, "prob": math.exp(x - z)} for i, x in post.items()]}]}
        raise AssertionError(path)


@pytest.fixture
def small_probe(monkeypatch):
    monkeypatch.setattr(cb, "PENALTY_CUT", 8)
    monkeypatch.setattr(cb, "PENALTY_MORE", 10)
    monkeypatch.setattr(cb, "FILLER_TOKENS", 30)
    monkeypatch.setattr(cb, "FILLER_HEAD_S", 0.0)
    monkeypatch.setattr(cb, "Server", lambda base: Probe())


def test_the_penalty_test_passes_when_the_window_is_the_replys(small_probe, monkeypatch):
    got = cb.penalty_test("http://probe", 6)
    assert got["verdict"] == "pass"
    assert got["greedy"]["counted"]["same_tokens"] == 10 and got["greedy"]["counted"]["read_whole"]
    assert got["greedy"]["counted"]["max_diff"] == 0.0
    w = got["window"]
    # its samplers opened for the probe alone: the path's top-p kept one token on C1
    assert w["sampling"] == {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}
    assert abs(w["turn"]["turn"] - 0.75) < 1e-6 and abs(w["turn"]["question"]) < 1e-6
    assert abs(w["none"]["turn"]) < 1e-6 and abs(w["all"]["question"] - 0.75) < 1e-6
    words = cb.penalty_words(got)
    assert words[-1] == "penalty_prompt_tokens' window: pass"
    assert any("cut and continued with penalty_prompt_tokens 8 (the cut reply's): the same for "
               "10 of 10 tokens; where they agree their log probabilities are at most 0.000 "
               "apart" in x for x in words)
    assert any("alone and among 8 others (batching alone): the same for 18 of 18" in x
               for x in words)


def test_the_penalty_test_fails_when_the_question_is_counted(small_probe, monkeypatch):
    monkeypatch.setattr(cb, "Server", lambda base: Probe(counts="all"))
    got = cb.penalty_test("http://probe", 6)
    assert got["window"]["turn"]["verdict"] == "fail" and got["verdict"] == "fail"


def test_where_a_continuation_parts_its_two_likeliest_are_said(small_probe, monkeypatch):
    monkeypatch.setattr(cb, "Server", lambda base: Probe(part_at=3))
    got = cb.penalty_test("http://probe", 6)
    assert got["greedy"]["counted"]["same_tokens"] == 3
    assert got["greedy"]["counted"]["parted"]["first"] == [(" w11", -0.1), ("x", -0.2)]
    assert any("they part where the first's two likeliest were" in x
               for x in cb.penalty_words(got))


def test_two_uncut_replies_alone_and_among_others_say_how_far_apart_batching_puts_them(
        small_probe):
    class Batched(Probe):
        """the reply among others: its log probabilities 0.2 off, its 12th token another"""
        busy = 0

        def post(self, path, body, timeout=None):
            filler = "Count from one" in body["messages"][0]["content"]
            if filler:
                Batched.busy += 1
            got = super().post(path, body, timeout)
            if path == "/v1/chat/completions" and not filler and Batched.busy:
                for k, x in enumerate(got["choices"][0]["logprobs"]["content"]):
                    x["logprob"] -= 0.2
                    if k == 11:
                        x["token"] = " other"
            return got
    import time as _t
    b = cb.batching_test(Batched(), 6)
    assert (b["fillers"], b["same_tokens"], b["of"]) == (8, 11, 18)
    assert abs(b["max_diff"] - 0.2) < 1e-9 and abs(b["median_diff"] - 0.2) < 1e-9
    assert b["parted"]["first"][0][0] == " w11" and _t.time()


def test_an_answer_left_at_its_old_limit_by_the_rule_is_said_not_failed():
    rule = {"status": "refused", "why": "the reply's text doesn't give back its 81,920 tokens "
                                        "(open: 81,917 tokens)"}
    assert cb.left_by_rule(rule)
    assert not cb.left_by_rule({"status": "refused", "why": "it opens its thinking again"})
    assert not cb.left_by_rule({"status": "continued"})
    r = {"status": "continued", "id": "q", "epoch": 0, "was": {"tokens": 81920}, "tokens": 82407,
         "new_tokens": 487, "ran_out": False, "limit": 98304, "form": "open",
         "tokens_evaluated": 82290, "prompt_tokens": 370, "settings": {"seed": 1},
         "timings": {"prompt_n": 82290, "prompt_ms": 20572.5, "predicted_n": 487,
                     "predicted_ms": 9740.0}}
    assert "read 82,290 tokens at 4,000 a second and wrote 487 tokens at 50 a second" in \
        cb.check_words(r)


# ---------------------------------------------------------------------------
# on a box, and home: plan U's answers cut, continued, imported
# ---------------------------------------------------------------------------

@pytest.fixture
def cont(k3k6, monkeypatch):  # noqa: F811
    """a k6 box whose stand-in writes 22 tokens a GPQA answer, plan U's limit
    20 (cut just after its thinking closed) and the team's 40"""
    monkeypatch.setenv("FAKE_LLAMA_CONTINUE", "1")
    monkeypatch.setenv("FAKE_LLAMA_OUT_TOKENS", "22")
    monkeypatch.setitem(fb.BENCH[GPQA]["budget"], "on", OLD)
    monkeypatch.setitem(cb.SPECS[(STEP, True)], "limit", NEW)
    monkeypatch.setattr(cb, "PENALTY_CUT", 8)
    monkeypatch.setattr(cb, "PENALTY_MORE", 6)
    monkeypatch.setattr(cb, "FILLER_TOKENS", 30)
    monkeypatch.setattr(cb, "FILLER_HEAD_S", 0.0)
    return k3k6


def plan_u_row(b: dict) -> Path:
    """plan U's box asks GPQA Diamond, every answer cut at OLD; its bundle
    imported, step2388 registered"""
    assert run_k(b, "u") == 0
    got = next((b["root"] / "u").glob("frontier-*.tar.gz"))
    assert ir.main([str(got), "--by", "masein", "--register", "step2388 (k3k6 runtime)",
                    "--file-sha256", b["sha"]]) == 0
    return got


def inputs_of(b: dict) -> tuple[Path, str]:
    got = cb.inputs(b["root"] / "u-cont", say=lambda x: None, models=[STEP])
    return Path(got["k6"]["file"]), got["k6"]["sha256"]


def cont_requests(b: dict) -> list[dict]:
    return [r for r in requests(b) if (r.get("messages") or [{}])[-1].get("role") == "assistant"
            and "verbose" in r]


def test_plan_us_cut_answers_are_continued_and_imported_with_the_old_limits_score_kept(cont):
    original = plan_u_row(cont)
    before = rb.sha256_file(original)
    row = config.OUT_DIR / ROW
    first = sf.read_answers(sf.task_dir(row, GPQA) / sf.ANSWERS)
    assert len(first) == N * 4 and all(a["finish"] == "length" and a["tokens"] == OLD
                                       for a in first.values())
    f, sha = inputs_of(cont)
    lines = cb.read_input(f)
    assert len(lines) == N * 4 and {r["row"] for r in lines} == {ROW}
    assert inputs_of(cont)[1] == sha and (cont["root"] / "u-cont" / "cont-k3.jsonl").read_text() == ""
    # two boxes, a half each
    for i in (1, 2):
        assert run_k(cont, f"c{i}", "--continue", str(f), "--pin", f"{f.name}={sha}", "--part",
                     f"{i}/2", only=False) == 0
    asked = cont_requests(cont)
    assert len(asked) == N * 4
    # each its turn, its limit, a seed of its own, the path and the window — nothing that samples
    assert all(sorted(r) == ["expert_path", "max_tokens", "messages", "penalty_prompt_tokens",
                             "seed", "stream", "verbose"] for r in asked)
    assert {(r["max_tokens"], r["penalty_prompt_tokens"], r["expert_path"]) for r in asked} == {
        (NEW - OLD, OLD, 6)}
    assert all(r["messages"][-1]["content"] == ""
               and r["messages"][-1]["reasoning_content"].endswith("</think>\n\nThe options compared.")
               for r in asked)
    got = [rb.read(p) for k in (1, 2) for p in (cont["root"] / f"c{k}").glob("frontier-cont-*.tar.gz")]
    assert len(got) == 2 and {x["bundle"]["shard"]["i"] for x in got} == {1, 2}
    rows = [json.loads(x) for g in got for x in
            g["files"][f"results/{ROW}/{GPQA}_0shot/frontier/{cb.DONE}"].decode().splitlines()]
    r = rows[0]
    assert r["status"] == "continued" and r["form"] == "closed, 2 newlines after </think>"
    assert (r["was"]["tokens"], r["new_tokens"], r["tokens"], r["finish"], r["ran_out"]) == (
        OLD, 2, 22, "stop", False)
    assert r["tokens_evaluated"] == r["prompt_tokens"] + OLD
    assert r["settings"]["penalty_prompt_tokens"] == OLD and r["settings"]["presence_penalty"] == 0.75
    assert got[0]["setup"]["continuation"]["limit"] == NEW
    # 31.3c: each k6 box tested the window and batching first, kept in its bundle
    tests = got[0]["setup"]["continuation"]["tests"]
    assert tests["verdict"] == "pass" and tests["batching"]["fillers"] == cb.FILLERS
    # home: both imported, the row's answers the continued ones, the old limit's score on record
    paths = [p for k in (1, 2) for p in (cont["root"] / f"c{k}").glob("frontier-cont-*.tar.gz")]
    said: list[str] = []
    assert ir.import_bundle(paths[0], "masein", said.append) == 0
    assert any("15 answers continued" in x or "answers continued" in x for x in said)
    assert ir.import_bundle(paths[1], "masein", said.append) == 0
    now = sf.read_answers(sf.task_dir(row, GPQA) / sf.ANSWERS)
    assert len(now) == N * 4
    assert all(a["tokens"] == 22 and a["finish"] == "stop" and a["continued"]["from"]["tokens"]
               == OLD for a in now.values())
    setup = json.loads((sf.task_dir(row, GPQA) / sf.SETUP).read_text())
    assert setup["budget"] == NEW and setup["continued"]["from_budget"] == OLD
    assert setup["continued"]["answers"] == N * 4 and len(setup["continued"]["bundles"]) == 2
    res = json.loads(next(sf.task_dir(row, GPQA).glob("results_*.json")).read_text())
    lim = res["frontier"]["limits"]
    assert res["results"][GPQA]["acc,none"] == pytest.approx((N - len(HARD)) / N)
    assert lim["at_old_limit"]["score"] == 0.0 and lim["at_old_limit"]["ran_out"] == N * 4
    assert lim["at_new_limit"]["score"] == pytest.approx((N - len(HARD)) / N)
    assert (lim["continued"], lim["still_ran_out"], lim["tokens_before"], lim["tokens_after"]) == (
        N * 4, 0, N * 4 * OLD, N * 4 * 22)
    how = report.frontier_how(res["frontier"])
    assert f"ran out of room at {OLD:,} tokens continued to {NEW:,}" in how and f"at {OLD}: 0.0" in how
    # the answers before kept beside the row; the original bundle untouched
    assert list(config.OUT_DIR.with_name("earlier").rglob(sf.ANSWERS))
    assert rb.sha256_file(original) == before
    # imported again: nothing changes
    said.clear()
    assert ir.import_bundle(paths[0], "masein", said.append) == 0
    assert any("imported already" in x for x in said)
    # and the next inputs file holds nothing more to continue
    assert cb.read_input(inputs_of(cont)[0]) == []


def test_a_continuation_onto_a_row_whose_answer_changed_is_refused(cont):
    plan_u_row(cont)
    f, sha = inputs_of(cont)
    assert run_k(cont, "c", "--continue", str(f), "--pin", f"{f.name}={sha}", only=False) == 0
    d = sf.task_dir(config.OUT_DIR / ROW, GPQA)
    lines = (d / sf.ANSWERS).read_text().splitlines()
    one = json.loads(lines[0])
    one["answer"] += " changed"
    (d / sf.ANSWERS).write_text("\n".join([json.dumps(one), *lines[1:]]) + "\n")
    said: list[str] = []
    p = next((cont["root"] / "c").glob("frontier-cont-*.tar.gz"))
    assert ir.import_bundle(p, "masein", said.append) == cb.REFUSED
    assert any("the row's answer isn't the one continued" in x for x in said)
    assert "nothing was imported" in said
    assert not any(json.loads(x).get("continued") for x in (d / sf.ANSWERS).read_text().splitlines())


def test_a_box_wont_continue_an_unpinned_file_or_another_paths(cont):
    plan_u_row(cont)
    f, sha = inputs_of(cont)
    with pytest.raises(SystemExit, match="give cont-k6.jsonl's sha256"):
        run_k(cont, "c", "--continue", str(f), only=False)
    with pytest.raises(SystemExit, match="doesn't continue|not of"):
        run_k(cont, "c3", "--continue", str(f), "--pin", f"{f.name}={sha}", only=False,
              thinking="off")


def test_the_check_continues_the_first_answers_and_tests_the_window_with_no_bundle(cont):
    plan_u_row(cont)
    f, sha = inputs_of(cont)
    assert run_k(cont, "chk", "--continue", str(f), "--pin", f"{f.name}={sha}",
                 "--continue-check", "5") == 0
    out = cont["root"] / "chk"
    got = json.loads((out / "continue-check.json").read_text())
    assert len(got["answers"]) == 5 and all(r["status"] == "continued" for r in got["answers"])
    assert got["penalty_test"]["verdict"] == "pass"
    log = (out / rg.OWN_LOG).read_text()
    assert "every answer continued is one reply" in log
    assert "applied: temperature 0.8" in log and "penalty_prompt_tokens 20" in log
    assert not list(out.glob("*.tar.gz")) and not (out / "progress.json").exists()
    # no question text in what it says
    assert "Invented question" not in log


# ---------------------------------------------------------------------------
# plan C and the fetch
# ---------------------------------------------------------------------------

def test_plan_c_is_k6_in_thirds_k3_on_one_and_the_q2s_easy_set_on_its_own():
    assert {b: s for b, s in fx.PLANS["C"].items()} == {
        "C1": [("on", (fx.CONT,), "1/3", 9)], "C2": [("on", (fx.CONT,), "2/3", 9)],
        "C3": [("on", (fx.CONT,), "3/3", 9)], "C4": [("off", (fx.CONT,), "", 30)],
        "C5": [("off", (fx.CONT_Q2,), "", 40)]}
    listed = fx.listing("C")
    assert listed[-1].startswith("5 boxes a build") and "9 slots of 131,072 tokens" in listed[1]
    assert "30 slots of 36,864 tokens" in listed[-4]
    assert "Q2's easy-set answers" in listed[-2] and "40 slots of 33,792 tokens" in listed[-2]
    # as many as a 5090 holds beside each file
    assert fx.fits(131072, fx.STEP2388_FILE) == 9 and fx.fits(36864, fx.STEP2388_FILE) == 30
    assert fx.fits(33792, fx.Q2_FILE) == 40
    assert fx.BOX_MODELS["C5"] == cb.Q2 and fx.model_refused("C5", STEP)
    assert fx.PLAN_PINS["C"] == fx.PLAN_PINS["U"] and fx.PLAN_RUNTIME["C"] == "k3k6-2611"


def line(*more: str) -> list[str]:
    return ["--as", STEP, "--gguf", "hf://o/r@fe12fa27825b3a0cc6cde3dedec3e838bd3e6de4/"
            "step2388.k3k6.gguf", "--server",
            "hf://o/r@fe12fa27825b3a0cc6cde3dedec3e838bd3e6de4/llama-server-k3k6-3ad38b7.tar.gz",
            *more]


def test_a_u_cont_boxs_line_continues_its_part_of_its_paths_file():
    sha = "a" * 64
    a = fx.parser().parse_args(line("C1", "--continue", "/workspace/cont-k6.jsonl",
                                    "--continue-sha256", sha))
    cmd = fx.argv_of("C1", 1, fx.box_of("C1")[0], a)
    assert cmd[cmd.index("--continue") + 1] == "/workspace/cont-k6.jsonl"
    assert f"cont-k6.jsonl={sha}" in cmd and cmd[cmd.index("--part") + 1] == "1/3"
    assert (cmd[cmd.index("--slots") + 1], cmd[cmd.index("--min-slots") + 1]) == ("9", "4")
    assert cmd[cmd.index("--runtime") + 1] == "k3k6-2611" and cmd[cmd.index("--thinking") + 1] == "on"
    a4 = fx.parser().parse_args(line("C4", "--continue", "/workspace/cont-k3.jsonl",
                                     "--continue-sha256", sha))
    cmd = fx.argv_of("C4", 1, fx.box_of("C4")[0], a4)
    assert "--part" not in cmd and cmd[cmd.index("--thinking") + 1] == "off"
    # the check on C1: GPQA Diamond's first five, in a folder the fetch never reads
    ac = fx.parser().parse_args(line("C1", "--continue", "/workspace/cont-k6.jsonl",
                                     "--continue-sha256", sha, "--continue-check", "5"))
    cmd = fx.argv_of("C1", 1, fx.box_of("C1")[0], ac)
    assert cmd[cmd.index("--continue-check") + 1] == "5" and cmd[cmd.index("--only") + 1] == GPQA
    assert cmd[cmd.index("--out") + 1].endswith("/contcheck") and "--part" not in cmd


@pytest.mark.parametrize("argv, words", [
    (["C1"], "give --continue"),
    (["U1", "--continue", "/w/cont-k6.jsonl", "--continue-sha256", "a" * 64], "U-cont's boxes"),
    (["C4", "--continue", "/w/cont-k3.jsonl", "--continue-sha256", "a" * 64,
      "--continue-check", "5"], "one box of the k6 path"),
])
def test_a_line_without_its_file_or_on_another_box_is_refused(argv, words, capsys):
    with pytest.raises(SystemExit):
        fx.main(line(*argv))
    assert words in capsys.readouterr().err


def test_the_fetch_and_import_name_a_continuations_bundle_by_its_model():
    name = rb.bundle_name(cb.SUITE, STEP, True, (1, 3))
    assert name == "frontier-cont-served__step2388-k3k6-2611-thinking-on-shard-1-of-3.tar.gz"
    assert ir.model_of_name(name) == STEP
    assert rb.SUITES[cb.SUITE]["format"] == 5


def test_the_limits_words_on_a_cell():
    assert report.limits_words({"limit": 98304, "from": 81920, "continued": 9,
                                "not_continued": 1, "still_ran_out": 2,
                                "at_old_limit": {"score": 0.761, "se": 0.023}}) == (
        " · its 9 answers that ran out of room at 81,920 tokens continued to 98,304 (1 not), "
        "2 still ran out · at 81,920: 76.1 ± 2.3")
    assert report.limits_words(None) == ""


def test_the_showcase_asks_at_the_teams_limit_in_one_go():
    import showcase_box as scb
    assert (scb.LIMIT, scb.CONTEXT) == (K6["limit"], K6["limit"] + 2048)



# ---------------------------------------------------------------------------
# 31.3b: the original Qwen3.6 Q2's easy set, continued to the same 32,768
# ---------------------------------------------------------------------------

def test_the_q2_is_continued_as_it_was_asked_its_cards_sampling_and_the_switch():
    assert (Q2["key"], Q2["limit"], Q2["path"], Q2["tasks"]) == (
        "q2", 32768, None, ("mmlu_chat", "gsm8k_chat"))
    assert cb.spec_of(cb.Q2, True) is None and cb.context(Q2) == 33792
    line = {"task": "mmlu_chat", "id": "q7", "epoch": 0, "tokens": 4096}
    msgs = cb.messages("mmlu_chat", "A question?")
    got = cb.request(line, msgs, "A reply so far", Q2, Q2_ASKED)
    # everything it was asked with but its limit — and no window: its requests
    # never set one, and the server's default counts the prompt with the reply
    assert {k: v for k, v in got.items() if k not in ("messages", "max_tokens", "seed")} == {
        k: v for k, v in Q2_ASKED.items() if k != "max_tokens"}
    assert "penalty_prompt_tokens" not in got and "expert_path" not in got
    assert got["messages"][-1] == {"role": "assistant", "content": "A reply so far"}
    assert (got["max_tokens"], got["seed"]) == (32768 - 4096, fb.seed_of("mmlu_chat", "q7+cont", 0))


def test_the_q2s_template_is_told_the_thinking_switch_not_a_path():
    told = []

    class Told(Stub):
        def post(self, path, body, timeout=None):
            told.append({k: v for k, v in body.items() if k not in ("messages", "content")})
            return super().post(path, body, timeout)

    msgs = cb.messages("mmlu_chat", "Which one?")
    line = {"task": "mmlu_chat", "id": "q", "epoch": 0, "answer": "x1 x2 x3", "tokens": 3,
            "prompt_tokens": 4}
    got = cb.prepare(Told(), line, msgs, Q2, Q2_ASKED)
    assert got["form"] == "reply" and got["continued_tokens"] == 7
    assert {"chat_template_kwargs": {"enable_thinking": False}} in told
    assert not any("expert_path" in x for x in told)


EASY_N = 6


def easy_invented() -> list[dict]:
    return [{"id": f"mm{k:03d}", "question": f"Invented easy question {k}?",
             "options": [f"right answer {k}", f"wrong {k}a", f"wrong {k}b", f"wrong {k}c"],
             "answer": "A"} for k in range(EASY_N)]


def run_q2(b: dict, out: str, *more: str, only: bool = True) -> int:
    """remote_gguf.py as U40 (and C5) run the original Q2, on its own runtime"""
    import os

    keep = {k: getattr(config, k) for k in [*rg.ENV, "OUT_DIR", "DM_ITEMS"]}
    env = {v: os.environ.get(v) for v in rg.ENV.values()}
    from service import runner
    lock = runner.LOCK
    try:
        from test_17_gguf_box import free_port
        return rg.main(["--as", cb.Q2, "--gguf", str(b["gguf"]), "--server", str(b["tarball"]),
                        "--based-on", "Qwen/Qwen3.6-35B-A3B", "--thinking", "off", "--slots", "3",
                        "--port", str(free_port()), "--load-timeout", "60",
                        "--runtime", "card-qwen-k3k6",
                        *(["--only", "mmlu_chat"] if only else []), "--out", str(b["root"] / out),
                        *more])
    finally:
        for k, v in keep.items():
            setattr(config, k, v)
        runner.LOCK = lock
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_the_q2s_easy_set_is_continued_to_the_same_limit_on_its_own_runtime(cont, monkeypatch):
    monkeypatch.setattr(fb, "_fetch", lambda task: easy_invented() if task == "mmlu_chat"
                        else __import__("test_17_gguf_box").invented())
    monkeypatch.setitem(fb.BENCH["mmlu_chat"]["budget"], "off", OLD)
    monkeypatch.setitem(cb.SPECS[(cb.Q2, False)], "limit", NEW)
    # U40: the Q2 asked the easy set at its (shrunk) limit, every answer cut
    assert run_q2(cont, "u40") == 0
    got = next((cont["root"] / "u40").glob("frontier-*.tar.gz"))
    assert ir.main([str(got), "--by", "masein", "--register", "Qwen3.6-35B-A3B Q2 (k3k6 build)",
                    "--file-sha256", cont["sha"]]) == 0
    row = config.OUT_DIR / "served__Qwen3.6-35B-A3B-Q2-k3k6-2611"
    first = sf.read_answers(sf.task_dir(row, "mmlu_chat") / sf.ANSWERS)
    assert len(first) == EASY_N and all(a["tokens"] == OLD for a in first.values())
    made = cb.inputs(cont["root"] / "u-cont", say=lambda x: None, models=[cb.Q2])
    f, sha = Path(made["q2"]["file"]), made["q2"]["sha256"]
    assert f.name == "cont-q2.jsonl" and len(cb.read_input(f)) == EASY_N
    # C5: continued as it was asked — its card's sampling and the switch, no path, no window
    assert run_q2(cont, "c5", "--continue", str(f), "--pin", f"{f.name}={sha}", only=False) == 0
    asked = cont_requests(cont)
    assert len(asked) == EASY_N
    assert all("expert_path" not in r and "penalty_prompt_tokens" not in r
               and r["chat_template_kwargs"] == {"enable_thinking": False}
               and r["presence_penalty"] == 1.5 and r["temperature"] == 0.7
               and r["max_tokens"] == NEW - OLD for r in asked)
    p = next((cont["root"] / "c5").glob("frontier-cont-*.tar.gz"))
    assert p.name == "frontier-cont-served__Qwen3.6-35B-A3B-Q2-k3k6-2611-thinking-off.tar.gz"
    said: list[str] = []
    assert ir.import_bundle(p, "masein", said.append) == 0
    now = sf.read_answers(sf.task_dir(row, "mmlu_chat") / sf.ANSWERS)
    assert all(a["tokens"] == 22 and a["finish"] == "stop" for a in now.values())
    res = json.loads(next(sf.task_dir(row, "mmlu_chat").glob("results_*.json")).read_text())
    assert res["results"]["mmlu_chat"]["acc,none"] == 1.0
    assert res["frontier"]["limits"]["at_old_limit"]["score"] == 0.0
    assert json.loads((sf.task_dir(row, "mmlu_chat") / sf.SETUP).read_text())["budget"] == NEW
