"""12a.8 (the brief's): Summarise — the judge reports findings, the code
computes the score. After 12a.7, 18 of 145 practice answers failed; ten were
"several versions" that the rubric makes −1 and the judge (gemma) gave 0 to
2, and one was the judge saying "Your mom arrives" at 7:00 PM wasn't there.
The judge finds what is wrong well and adds up points badly, so it reports
findings; the code scores them — 4, a missing fact −1 (2 at most), invented
or wrong −2, several versions −1, a length asked and not kept −1 — and drops a
claim the answer disproves, saying so.

The exact cases the brief names, and the retry. Nothing runs a model: the
judge's replies are written here. 12a.9: a generated rubric is a checklist now
(test_12a9_checklist.py); these are the findings a rubric someone edited
under 12a.8 still asks for, marked as they were."""

from __future__ import annotations

import json

import pytest

import everyday as ev
from conftest import make_service
from service import config, db, llm
from test_everyday_12a5 import _asked

BANK = {q["id"]: q for q in ev.load_bank()}
LEAVE, SCHOOL_RUN, BANK_SMS = "everyday-summarising-15", "everyday-summarising-07", \
    "everyday-summarising-05"


def said(**f) -> str:
    """the judge's reply: findings, and no score"""
    return json.dumps({"missing_facts": [], "invented_or_wrong": [], "several_versions": False,
                       "length_ok": "not asked", "note": "", **f})


def findings(q: dict, need: int | None = None) -> dict:
    """12a.9: the question's judge check as 12a.8 generated it — findings on
    its facts, `need` of them if given"""
    facts, n = ev._facts_of(q)
    was = {**q, "checks": [{"type": "facts", "values": facts, "n": need or n}]}
    return {"type": "judge", "scale": 4, "pass_at": 3, "findings": True,
            "rubric": ev.summarise_rubric(was, ev.RUBRIC_12A8)}


def verdict(qid: str, answer: str, **f) -> dict:
    q = BANK[qid]
    return ev.parse_verdict(said(**f), findings(q), q, answer)


LEAVE_3 = """Option 1: Annual leave 3–10 November requested; the Hamilton file is with Mariam, and I'm reachable by mobile. Form attached.
Option 2: Requesting leave from 3 to 10 November. Work done, Hamilton file handed to Mariam; call my mobile if urgent.
Option 3: Leave request, 3 Nov – 10 Nov. Hamilton file: Mariam. Reachable on mobile. Form attached for approval."""

SCHOOL_RUN_ANSWER = ("Tomorrow: She takes Zain to football at 4:00 PM. You pick up Layla at 5:30 PM. "
                     "Everyone home by 6:30 PM for dinner. Your mom arrives at 7:00 PM. Get Zain's "
                     "birthday cake and top up Layla's transit card.")

SMS_NO_BALANCE = ("$2,450.00 was taken from your account ending 4821 at FreshMart City Mall on "
                  "12 Oct. Not you? Call 800 1234.")


def test_three_versions_with_every_fact_is_3_and_passes():
    v = verdict(LEAVE, LEAVE_3, several_versions=True, length_ok="yes")
    assert (v["pass"], v["score"]) == (True, 3)
    assert v["reason"] == "3 of 4: several versions (−1)"
    assert v["findings"]["several_versions"] is True


def test_a_missing_fact_the_answer_has_is_dropped_and_it_passes():
    # the judge on the school run plan (lookahead + MTP): "misses that the mother is coming"
    v = verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER, missing_facts=["mum / mother"])
    assert (v["pass"], v["score"]) == (True, 4)
    assert v["reason"] == "4 of 4: all key facts, one version"
    assert v["dropped"] == [{"kind": "missing", "claim": "mum / mother",
                             "text": "the judge said it missed “mum / mother”; the answer has "
                                     "it"}]
    # said in its own words, it is still that fact — and the answer still has it
    v = verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER, missing_facts=["the mother is coming"])
    assert v["score"] == 4 and v["dropped"][0]["text"] == (
        "the judge said it missed “the mother is coming”; the answer has it")
    # one that is none of the listed facts is not a missing fact
    v = verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER, missing_facts=["the football kit"])
    assert v["score"] == 4 and v["dropped"][0]["text"] == (
        "the judge said it missed “the football kit”, which isn’t one of the key facts")
    # and the answer that really hasn't got her
    v = verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER.replace(" Your mom arrives at 7:00 PM.", ""),
                missing_facts=["mum / mother"])
    assert (v["score"], v["reason"]) == (3, "3 of 4: missing: mum / mother (−1)")
    assert "dropped" not in v


def test_a_missing_balance_is_3():
    v = verdict(BANK_SMS, SMS_NO_BALANCE, missing_facts=["8,320.75 / 8320.75"], length_ok="yes")
    assert (v["pass"], v["score"]) == (True, 3)
    assert v["reason"] == "3 of 4: missing: 8,320.75 / 8320.75 (−1)"


def test_an_invented_number_costs_2():
    ans = SMS_NO_BALANCE + " Balance $8,320.75. A £50 fee applies."
    v = verdict(BANK_SMS, ans, invented_or_wrong=["A £50 fee applies"], length_ok="yes")
    assert (v["pass"], v["score"]) == (False, 2)
    assert v["reason"] == "2 of 4: invented or wrong: “A £50 fee applies” (−2)"
    # one that doesn't quote the answer is dropped, and said
    v = verdict(BANK_SMS, ans, invented_or_wrong=["a fee of fifty pounds"], length_ok="yes")
    assert v["score"] == 4 and v["dropped"][0]["text"] == (
        "the judge said “a fee of fifty pounds” is invented or wrong; the answer doesn’t say that")


def test_style_alone_is_4():
    ans = ("Here's a concise summary:\n- " + SCHOOL_RUN_ANSWER.replace(". ", ".\n- ")
           + "\nLet me know if you'd like it any shorter!")
    v = verdict(SCHOOL_RUN, ans)
    assert (v["pass"], v["score"], v["reason"]) == (True, 4, "4 of 4: all key facts, one version")
    # and a judge that counts the lead-in or the offer anyway is overruled
    v = verdict(SCHOOL_RUN, ans, invented_or_wrong=["Here's a concise summary:",
                                                    "Let me know if you'd like it any shorter!"])
    assert v["score"] == 4 and [d["kind"] for d in v["dropped"]] == ["style", "style"]


@pytest.mark.parametrize("length_ok,score", [("no", 3), ("yes", 4), ("not asked", 4)])
def test_a_length_counts_only_where_the_request_asks_one(length_ok, score):
    # the bank SMS asks for it "in short"; the school run plan asks none
    ans = SMS_NO_BALANCE + " Balance $8,320.75."
    assert verdict(BANK_SMS, ans, length_ok=length_ok)["score"] == score
    assert verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER, length_ok=length_ok)["score"] == 4


def test_everything_wrong_floors_at_0_and_the_judges_reply_is_kept():
    v = verdict(SCHOOL_RUN, "Option 1: dinner at 9.\nOption 2: dinner at 10.",
                missing_facts=["5:30 / 5.30", "mum / mother", "cake"],
                invented_or_wrong=["dinner at 9"], several_versions=True)
    assert (v["pass"], v["score"]) == (False, 0)
    assert v["reason"] == ("0 of 4: missing: 5:30 / 5.30; mum / mother; cake (−2) · invented or "
                           "wrong: “dinner at 9” (−2) · several versions (−1)")
    assert v["judge_raw"]["missing_facts"] == ["5:30 / 5.30", "mum / mother", "cake"]


def test_a_fact_it_may_leave_out_costs_nothing():
    """a rubric that keeps "at least n of these k": one short of k is fine"""
    q = BANK["everyday-summarising-long-01"]
    facts, _ = ev._facts_of(q)
    first = ev._show(facts[0])
    v = ev.parse_verdict(said(missing_facts=[first]), findings(q, need=5), q, "x")
    assert v["score"] == 4 and v["reason"] == f"4 of 4: missing: {first} (it may leave that out)"


def test_the_text_given_back_is_0_whatever_the_judge_finds():
    q = BANK[BANK_SMS]
    text = q["prompt"][q["prompt"].index('"') + 1:q["prompt"].rindex('"')]
    v = ev.parse_verdict(said(), findings(q), q, text)
    assert (v["pass"], v["score"], v["reason"]) == (False, 0,
                                                    "0 of 4: the text given back, not a summary")


@pytest.mark.parametrize("reply", ['{"score": 3, "reason": "a lead-in"}', "not json at all",
                                   '{"missing_facts": "5:30", "invented_or_wrong": []}',
                                   '{"missing_facts": [], "invented_or_wrong": [], '
                                   '"length_ok": "maybe"}'])
def test_a_reply_that_isnt_findings_is_not_read(reply):
    q = BANK[SCHOOL_RUN]
    assert ev.parse_verdict(reply, findings(q), q, SCHOOL_RUN_ANSWER) is None


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    yield client
    client.__exit__(None, None, None)


def test_an_unreadable_reply_is_asked_again_then_waits(svc):
    q = BANK[SCHOOL_RUN]
    mdir = config.OUT_DIR / "org__m"
    _asked(mdir, [q], answer=lambda q: SCHOOL_RUN_ANSWER)
    ev.write(mdir, ev.mark(mdir))
    assert ev.read(mdir)["items"][0]["pass"] is None                   # waiting on the judge
    # the reply can't be read: asked again, once, as a re-mark batch
    ev.finish(mdir, {f"everyday:0:{q['id']}": llm.Result(text="I think it's fine!")})
    [it] = ev.read(mdir)["items"]
    assert (it["pass"], it["reason"]) == (None, ev.ASKED_AGAIN)
    [b] = [b for b in db.batches_pending() if b["kind"] == "everyday_remark"]
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].endswith(":retry:" + q["id"])]
    assert len(sent) == 1 and sent[0]["user"] == ev.judge_prompt(q, SCHOOL_RUN_ANSWER)
    # the second can't be read either: neither pass nor fail, and counted waiting
    ev.finish_remark(config.OUT_DIR, {sent[0]["custom_id"]: llm.Result(text="{\"ok\": 1}")})
    out = ev.read(mdir)
    [it] = out["items"]
    assert (it["pass"], it["reason"]) == (None, ev.UNREADABLE) and out["waiting"] == 1
    assert len([b for b in db.batches_pending() if b["kind"] == "everyday_remark"]) == 1  # no third
    # a second asking that can be read lands as any verdict does (12a.9: a checklist)
    ev.finish_remark(config.OUT_DIR, {sent[0]["custom_id"]: llm.Result(text=json.dumps(
        {**json.loads(ev.stub_reply(ev.judge_prompt(q, SCHOOL_RUN_ANSWER))),
         "checklist": [{"fact": "mum / mother", "status": "missing"}]}))})
    [it] = ev.read(mdir)["items"]
    assert (it["pass"], it["score"]) == (True, 4)
    assert it["dropped"][0]["text"].endswith("the answer has it") and it["judge_raw"]


def test_the_remark_asks_every_summarise_answer_again(svc, monkeypatch):
    """a verdict on 12a.7's rubric is on another rubric: both halves are asked again"""
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    qs = [q for q in BANK.values() if q["group"] == "summarising"]
    mdir = config.OUT_DIR / "org__m"
    _asked(mdir, qs)
    # the key of the verdicts 12a.7's re-mark wrote: its rubric's
    old = {q["id"]: ev.rubric_key({"checks": [{"type": "judge", "rubric": ev.summarise_rubric(
        q, ev.RUBRIC_12A7)}]}) for q in qs}
    assert not set(old.values()) & {ev.rubric_key(q) for q in qs}
    ev.write(mdir, {"model": "org/m", "items": [
        {"id": q["id"], "pass": False, "reason": "1 of 4: several versions", "score": 1,
         "rubric": old[q["id"]], "answer_text": q["reference"]} for q in qs]})
    ev.remark(config.OUT_DIR, judge=True, snapshot=False)
    items = ev.read(mdir)["items"]
    assert {ev.half(BANK[it["id"]]) for it in items} == {ev.HIDDEN, ev.PRACTICE}
    assert all(it["rubric"] == ev.rubric_key(BANK[it["id"]]) and it["reason"] != "1 of 4: several "
               "versions" and "findings" in it for it in items)
