"""17c, part 4 of the second review round — before Start on grading. Each
test fails on 0a81fc2:
- 22: a batch is never failed with its replies on disk paid for — a poll with
  no key, or a record that fails, waits and tries again;
- 23: an answer whose reply is never a grade is asked GRADE_TRIES times, then
  ungraded, counted wrong, and never sent again;
- 24–26: SimpleQA, MATH and HLE read a grade only in its exact form;
- 27: a grade carries the prompt it was sent with;
- 28: a batch whose grader OpenRouter moved says why, and Start sends its
  unsent answers to the grader pinned now;
- 29: an unread HLE reply never shows its words;
- 30: after Stop, what the batches out hold and what Carry on costs.
Invented questions and a stand-in for OpenRouter only: nothing is sent
anywhere."""

from __future__ import annotations

import json
import threading
import time

import pytest

import frontier as fb
import frontier_graders as fg
from service import ai_models, config, llm, llm_poller
from service import frontier as sf
from service import frontier_grade as fgr
from test_17_grading import ROW, drain, rec, results, svc  # noqa: F401 — svc is the fixture
from test_17b_grading import GEMINI, GPT, plain, stub

ASKED_ONCE = 7                          # five SimpleQA answers, two of MATH's


def landed(timeout: float = 10.0) -> None:
    """every request of every batch out answered on disk"""
    end = time.time() + timeout
    while time.time() < end:
        tls = [llm.tally(p["batch_id"]) or {} for p in fgr.pending()]
        if tls and all(t.get("answered", 0) + t.get("failed", 0) >= t.get("sent", 1)
                       for t in tls):
            return
        time.sleep(0.05)
    pytest.fail("the batches never landed")


# ---------------------------------------------------------------------------
# 22: replies paid for are never dropped
# ---------------------------------------------------------------------------

def test_22_a_poll_with_no_key_keeps_the_batch_and_its_replies(svc, monkeypatch):  # noqa: F811
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    landed()
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    llm_poller.tick()                           # 0a81fc2: failed, its replies never read
    assert len(fgr.pending()) == 2
    assert any("no key" in w["why"] for w in svc.get("/api/frontier/grading").json()["waits"])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    drain()
    fgr.start("masein")
    drain()
    assert len(asked) == ASKED_ONCE             # each answer paid for once
    assert results("simpleqa_epoch")["results"]["simpleqa_epoch"]["acc,none"] == \
        pytest.approx(5 / 6)


def test_22_a_record_that_fails_is_tried_again_and_nothing_paid_twice(svc, monkeypatch):  # noqa: F811
    asked = stub(monkeypatch, plain)
    fgr.start("masein")
    landed()
    real, broken = fb.load, {"on": True}

    def load(task, root):
        if broken["on"]:
            broken["on"] = False
            raise OSError("the disk was full")
        return real(task, root)
    monkeypatch.setattr(fb, "load", load)
    llm_poller.tick()                           # 0a81fc2: failed, 5 or 2 replies dropped
    assert fgr.pending() and any("couldn't be recorded yet" in w["why"]
                                 for w in fgr.waits())
    drain()
    fgr.start("masein")
    drain()
    assert len(asked) == ASKED_ONCE
    assert results("math_l5_epoch") and results("simpleqa_epoch")


def test_22_a_failed_batch_has_what_landed_read(svc, monkeypatch):  # noqa: F811
    """failed() read nothing: the next Start paid again for every reply"""
    stub(monkeypatch, plain)
    fgr.start("masein")
    landed()
    bid = next(p["batch_id"] for p in fgr.pending() if p["slot"] == "simpleqa")
    fgr.failed(bid, "a test's failure")
    g = sf.read_grades(sf.task_dir(config.OUT_DIR / ROW, "simpleqa_epoch"))
    assert len(g.get("items") or {}) == 5


# ---------------------------------------------------------------------------
# 23: never readable: asked GRADE_TRIES times, then ungraded
# ---------------------------------------------------------------------------

def test_23_a_reply_never_readable_is_asked_three_times_then_ungraded(svc, monkeypatch):  # noqa: F811
    asked = stub(monkeypatch, lambda m, r: ("I would rather not say", "stop", "")
                 if r["custom_id"].endswith(":2#0") else plain(m, r))
    for _ in range(4):
        fgr.start("masein")
        drain()
    assert sum(1 for _, c in asked if c.endswith(":2#0")) == sf.GRADE_TRIES   # 0a81fc2: 4
    page = svc.get("/api/frontier/grading").json()
    assert [(x["slot"], x["n"], x["ungraded"]) for x in page["refused"]] == [("simpleqa", 1, 1)]
    assert page["tries"] == sf.GRADE_TRIES and not page["estimate"]["rows"]
    # the score is final: the ungraded one counted wrong, and said
    s = results("simpleqa_epoch")
    assert s["results"]["simpleqa_epoch"]["acc,none"] == pytest.approx(4 / 6)
    assert s["frontier"]["ungraded"] == 1
    sc = sf.score_task(config.OUT_DIR / ROW, "simpleqa_epoch", rec())
    assert "1 its grader gave no grade 3 times, counted wrong" in sf.words("simpleqa_epoch", sc)


# ---------------------------------------------------------------------------
# 24–26, 29: a grade only in its exact form
# ---------------------------------------------------------------------------

def test_24_simpleqa_reads_a_lone_grade_only():
    for text in ("not correct", "The correct answer is Paris, so B",
                 "B: the correct answer differs", "B\n\nExplanation: … CORRECT in spirit",
                 "A: INCORRECT", "a"):
        assert fg.read("simpleqa", text, {})["ok"] is None, text
    for text, g in (("**B**", "B"), ("(A)", "A"), ("C.", "C"), ("B: INCORRECT", "B"),
                    ("Not attempted", "C"), ("`CORRECT`", "A")):
        assert fg.read("simpleqa", text, {})["grade"] == g, text


def test_25_math_reads_yes_or_no_and_asks_again_otherwise():
    for text, ok in (("Yes.", True), ("**Yes**", True), ("no", False), ("No.", False)):
        assert fg.read("math", text, {})["ok"] is ok, text
    for text in ("As an AI, I can't compare these", "Yes, they are equivalent", "Maybe"):
        assert fg.read("math", text, {})["ok"] is None, text


def test_26_29_hle_reads_a_lone_yes_or_no_and_never_shows_an_unread_reply():
    assert fg.read("hle", "correct: yes/no", {})["ok"] is None
    assert fg.read("hle", "extracted_final_answer: 7\ncorrect: **yes**", {})["ok"] is True
    quoted = "The question asks: Invented gated question text? I can't say.\ncorrect: maybe"
    got = fg.read("hle", quoted, {})
    assert got["ok"] is None and "Invented gated" not in json.dumps(got)
    assert fg.read("math", "Invented problem text", {}).get("said") == ""


# ---------------------------------------------------------------------------
# 27, 28: the prompt sent, and a grader that moved
# ---------------------------------------------------------------------------

def test_27_a_grade_carries_the_prompt_it_was_sent_with(svc, monkeypatch):  # noqa: F811
    stub(monkeypatch, plain)
    fgr.start("masein")
    sent = fg.prompt_sha("simpleqa")
    landed()
    monkeypatch.setattr(fg, "prompt_sha", lambda slot: "f" * 64)    # the file changed since
    drain()
    g = sf.read_grades(sf.task_dir(config.OUT_DIR / ROW, "simpleqa_epoch"))
    assert {x["prompt_sha256"] for x in g["items"].values()} == {sent}
    assert g["grader"]["prompt_sha256"] == sent


def test_28_a_batch_whose_grader_moved_says_why_and_start_sends_its_unsent_again(
        svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    gate = threading.Event()
    seen: list = []

    def answer(model, row):
        seen.append((model, row["custom_id"]))
        if model == GPT and sum(1 for m, _ in seen if m == GPT) == 1:
            gate.wait(10)
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    for _ in range(200):
        if any(m == GPT for m, _ in seen):
            break
        time.sleep(0.05)
    moved = "openai/gpt-4.1 now points to openai/gpt-4.1-2026-01-01, not the version pinned"
    monkeypatch.setattr(ai_models, "drifted",
                        lambda pin: moved if (pin or {}).get("id") == GPT else "")
    gate.set()
    time.sleep(0.5)
    llm_poller.tick()
    waits = svc.get("/api/frontier/grading").json()["waits"]
    assert {"why": f"SimpleQA Verified's grader: {moved}",
            "carry": "Start sends its 4 unsent answers to the grader pinned now."} in waits
    # another grader chosen: Start cancels the old batch's unsent, sends them anew
    fgr.save("simpleqa", GEMINI, "masein")
    got = fgr.start("masein")
    assert got["moved"] == 4
    drain()
    assert sum(1 for m, c in seen if m == GPT) == 1
    assert sorted(c for m, c in seen if m == GEMINI and "algebra/" not in c) == \
        [f"frgr:{k}#0" for k in range(1, 5)]
    assert results("simpleqa_epoch") is not None


# ---------------------------------------------------------------------------
# 30: the card after Stop
# ---------------------------------------------------------------------------

def test_30_after_stop_the_card_says_what_is_held_and_what_carry_on_costs(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    gate = threading.Event()
    seen: list = []

    def answer(model, row):
        seen.append(row["custom_id"])
        if model == GPT and len([c for c in seen if "algebra/" not in c]) == 1:
            gate.wait(10)
        return plain(model, row)
    stub(monkeypatch, answer)
    fgr.start("masein")
    for _ in range(200):
        if any("algebra/" not in c for c in seen):
            break
        time.sleep(0.05)
    fgr.stop("masein")
    gate.set()
    time.sleep(0.5)
    page = svc.get("/api/frontier/grading").json()
    # nothing new waits; four are held out, unsent — with their cost
    assert page["estimate"]["answers"] == 0
    assert page["held"]["answers"] == 4 and page["held"]["usd"] > 0
    assert page["held"]["usd_known"] is True
