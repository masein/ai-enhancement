"""12a.9 (the brief's): Summarise — the findings judge was too lenient on the
small models. It reported "all key facts" of a fact stated with the wrong
person (gemma-3-270m-it on the school run plan: "She … will pick up Layla",
"Her mum is coming at 7", "She is also reminded to get the cake") and of a
fact without its number (Qwen3-0.6B on the work project's email thread: no
10 hours for the overtime cap, no 9:30 daily call). Both were 4 of 4.

Now the judge fills in a checklist — every key fact correct, wrong (quoting
the answer) or missing, who does what being part of the fact — and the code
checks it: a fact called correct whose number, time, amount or name isn't in
the answer is missing, and said; a fact called missing that the answer has is
correct, as 12a.8 dropped it. Scored as 12a.8: a missing fact −1 (2 at
most), anything wrong −2, several versions −1, a length asked and not kept
−1. Every listed fact counts where the request sets no length.

The two answers are the ones the server stored, both on the practice half
(a hidden question's answer never comes into the repo), each judged 4 of 4 by
12a.8. Nothing runs a model: the judge's replies are written here."""

from __future__ import annotations

import json

import pytest

import everyday as ev
from conftest import make_service
from make_fixture import HIDDEN_FIXTURE
from service import config, llm
from test_everyday_12a5 import _asked

BANK = {q["id"]: q for q in ev.load_bank()}
LEAVE, SCHOOL_RUN, EMAIL = ("everyday-summarising-15", "everyday-summarising-07",
                            "everyday-summarising-long-01")

# gemma-3-270m-it, the school run plan, as stored: you pick Layla up, it's your
# mum, you get the cake — the wrong person each time
GEMMA_SCHOOL_RUN = (
    "Here's a summary of the message from your wife:\nYour wife is planning a day with Zain and "
    "Layla. She is planning to take Zain to a football game at 4 and will pick up Layla from her "
    "friend's house at 5:30. They will then meet at home by 6:30 for dinner. Her mum is coming at "
    "7. She is also reminded to get the cake for Zain's birthday. Additionally, she is planning to "
    "top up the transit card for Layla, as she needs it for the metro tomorrow.")

# Qwen3-0.6B, the email thread (work project), as stored: the overtime cap
# without its 10 hours, no daily call at 9:30
QWEN_EMAIL = (
    "**Summary of Email Thread:**  \n\n- **Timeline Update:** The Northgate Outfitters website "
    "will launch on **October 6** instead of **October 20**, reducing the timeline by two weeks.  "
    "\n- **Mobile Checkout:** The mobile checkout redesign will be moved to **Phase 2 "
    "(November)**, with product pages and search filters retained.  \n- **Budget:** The project "
    "remains at **$18,000**.  \n- **Deadlines:**  \n  - **Tomasz:** Filters will be completed by "
    "**29 September**.  \n  - **Aisha:** Design review will be moved to **Wednesday, October 1, "
    "2:00 PM**.  \n  - **Mei:** Full build will be completed by **3 October**.  \n- **Team "
    "Flexibility:** The team is flexible and agrees to the changes, with clear deadlines and a "
    "capped overtime limit.  \n\n**Key Takeaway:** The timeline and tasks are adjusted, with "
    "deadlines set for each contributor.")

LEAVE_3 = """Option 1: Annual leave 3–10 November requested; the Hamilton file is with Mariam, and I'm reachable by mobile. Form attached.
Option 2: Requesting leave from 3 to 10 November. Work done, Hamilton file handed to Mariam; call my mobile if urgent.
Option 3: Leave request, 3 Nov – 10 Nov. Hamilton file: Mariam. Reachable on mobile. Form attached for approval."""

SCHOOL_RUN_ANSWER = ("Tomorrow: She takes Zain to football at 4:00 PM. You pick up Layla at 5:30 PM. "
                     "Everyone home by 6:30 PM for dinner. Your mom arrives at 7:00 PM. Get Zain's "
                     "birthday cake and top up Layla's transit card.")


def facts(qid: str) -> list[str]:
    """the question's key facts, as its rubric lists them"""
    return [ev._show(f) for f in ev._facts_of(BANK[qid])[0]]


def listed(qid: str, **status) -> str:
    """the judge's reply: every fact correct but those given — a status, or
    ("wrong", the answer's words)"""
    rows = []
    for fact in facts(qid):
        st = status.get(fact, "correct")
        st, quote = st if isinstance(st, tuple) else (st, "")
        rows.append({"fact": fact, "status": st, **({"quote": quote} if quote else {})})
    return json.dumps({"checklist": rows, "invented_or_wrong": status.get("invented", []),
                       "several_versions": status.get("several", False),
                       "length_ok": status.get("length", "not asked"), "note": ""})


def verdict(qid: str, answer: str, reply: str) -> dict:
    q = BANK[qid]
    return ev.parse_verdict(reply, ev.judge_check(q), q, answer)


# ---------------------------------------------------------------------------
# the brief's two answers fail; the three-option leave request stays at 3
# ---------------------------------------------------------------------------

def test_the_school_run_plan_with_the_wrong_person_fails():
    assert facts(SCHOOL_RUN) == ["5:30 / 5.30", "mum / mother", "transit card", "cake"]
    reply = listed(SCHOOL_RUN, **{
        "5:30 / 5.30": ("wrong", "will pick up Layla from her friend's house at 5:30"),
        "mum / mother": ("wrong", "Her mum is coming at 7"),
        "cake": ("wrong", "She is also reminded to get the cake")})
    v = verdict(SCHOOL_RUN, GEMMA_SCHOOL_RUN, reply)
    assert (v["pass"], v["score"]) == (False, 2)
    assert v["reason"] == (
        "2 of 4: wrong: 5:30 / 5.30 (“will pick up Layla from her friend's house at 5:30”); "
        "mum / mother (“Her mum is coming at 7”); cake (“She is also reminded to get the cake”) "
        "(−2)")
    assert [r["status"] for r in v["findings"]["checklist"]] == ["wrong", "wrong", "correct",
                                                                 "wrong"]
    assert ev.half(BANK[SCHOOL_RUN]) == ev.PRACTICE
    # the judge's words cut short with "…", as the brief quotes it, still quote the answer
    reply = listed(SCHOOL_RUN, **{"5:30 / 5.30": ("wrong", "She is planning to take Zain … and "
                                                            "will pick up Layla")})
    v = verdict(SCHOOL_RUN, GEMMA_SCHOOL_RUN, reply)
    assert (v["pass"], v["score"]) == (False, 2) and "dropped" not in v


def test_the_email_thread_without_its_numbers_fails_whatever_the_judge_says():
    assert facts(EMAIL)[-2:] == ["10 hours / overtime", "9:30 / 9.30"]
    assert ev.half(BANK[EMAIL]) == ev.PRACTICE
    # the judge as the brief found it: every fact correct
    v = verdict(EMAIL, QWEN_EMAIL, listed(EMAIL))
    assert (v["pass"], v["score"]) == (False, 2)
    assert v["reason"] == "2 of 4: missing: 10 hours / overtime; 9:30 / 9.30 (−2)"
    assert [d["text"] for d in v["dropped"]] == [
        "the judge said it has “10 hours / overtime”; the answer doesn’t say “10 hours”",
        "the judge said it has “9:30 / 9.30”; the answer doesn’t say “9:30”"]
    assert {r["fact"]: r["by"] for r in v["findings"]["checklist"]
            if r["status"] == "missing"} == {"10 hours / overtime": "code", "9:30 / 9.30": "code"}
    # with its numbers, it has every fact
    whole = QWEN_EMAIL.replace("a capped overtime limit", "overtime capped at 10 hours, and a "
                               "daily status call at 9:30")
    v = verdict(EMAIL, whole, listed(EMAIL))
    assert (v["pass"], v["score"], v["reason"]) == (True, 4, "4 of 4: all key facts, one version")


def test_the_three_option_leave_request_stays_at_3():
    v = verdict(LEAVE, LEAVE_3, listed(LEAVE, several=True, length="yes"))
    assert (v["pass"], v["score"]) == (True, 3)
    assert v["reason"] == "3 of 4: several versions (−1)"
    assert "dropped" not in v


# ---------------------------------------------------------------------------
# what the code checks
# ---------------------------------------------------------------------------

def test_a_false_missing_is_still_dropped():
    """12a.8's rule kept: "mum / mother" called missing of an answer with "Your mom arrives" """
    v = verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER, listed(SCHOOL_RUN, **{"mum / mother": "missing"}))
    assert (v["pass"], v["score"]) == (True, 4)
    assert v["dropped"] == [{"kind": "missing", "claim": "mum / mother",
                             "text": "the judge said it missed “mum / mother”; the answer has "
                                     "it"}]
    # …but "overtime" alone doesn't have "10 hours / overtime": the claim stands
    v = verdict(EMAIL, QWEN_EMAIL, listed(EMAIL, **{"10 hours / overtime": "missing",
                                                     "9:30 / 9.30": "missing"}))
    assert (v["score"], "dropped" in v) == (2, False)


def test_a_wrong_that_doesnt_quote_the_answer_is_decided_by_the_code():
    reply = listed(SCHOOL_RUN, **{"cake": ("wrong", "she bakes the cake herself")})
    v = verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER, reply)
    assert (v["score"], v["dropped"][0]["text"]) == (4, "the judge said “cake” is wrong, quoting "
                                                        "“she bakes the cake herself”; the answer "
                                                        "doesn’t say that")
    # and with no cake in the answer, the code finds it missing
    v = verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER.replace("birthday cake and ", ""), reply)
    assert (v["score"], v["reason"]) == (3, "3 of 4: missing: cake (−1)")


def test_a_fact_the_judge_skips_or_invents():
    rows = json.loads(listed(SCHOOL_RUN))
    rows["checklist"] = [r for r in rows["checklist"] if r["fact"] != "transit card"] + [
        {"fact": "the football kit", "status": "missing"}]
    ans = SCHOOL_RUN_ANSWER.replace(" and top up Layla's transit card", "")
    v = verdict(SCHOOL_RUN, ans, json.dumps(rows))
    assert (v["score"], v["reason"]) == (3, "3 of 4: missing: transit card (−1)")
    assert [d["text"] for d in v["dropped"]] == [
        "the judge checked “the football kit”, which isn’t one of the key facts",
        "the judge didn’t check “transit card”; the answer hasn’t got it"]


def test_every_listed_fact_counts_unless_the_request_limits_the_length():
    """12a.8 let a long text's summary leave out 3 of 8 whatever was asked. A
    request that sets no length has every key fact: a missing one costs 1, 2 at
    most. One that limits it (bullets, words, "short") still lets the summary
    choose, as far as the question's own "at least n" allows"""
    q = BANK[EMAIL]
    assert ev.stated_length(q["prompt"]) == ""
    assert "It should keep all 9 of these:" in ev.judge_check(q)["rubric"]
    no_checkout = QWEN_EMAIL.replace("**Mobile Checkout:** The mobile checkout redesign",
                                     "The redesign")
    v = verdict(EMAIL, no_checkout, listed(EMAIL, **{"checkout": "missing"}))
    assert v["reason"] == ("2 of 4: missing: checkout; 10 hours / overtime; 9:30 / 9.30 (−2)")
    one = QWEN_EMAIL.replace("a capped overtime limit", "overtime capped at 10 hours")
    assert verdict(EMAIL, one, listed(EMAIL))["reason"] == "3 of 4: missing: 9:30 / 9.30 (−1)"
    # "give me the main points as 4 bullets": the reference leaves out the boiler, still undecided
    four = BANK["everyday-summarising-long-14"]
    assert ev.stated_length(four["prompt"]) == "4 bullets"         # 12a.9: "bullets" is a length
    assert "It should keep at least 5 of these 8:" in ev.judge_check(four)["rubric"]
    v = ev.stub_for(four, four["reference"])
    assert (v["score"], v["reason"]) == (4, "4 of 4: missing: boiler / heating (it may leave that "
                                            "out)")


def test_an_invented_number_and_a_wrong_fact_cost_2_once():
    ans = GEMMA_SCHOOL_RUN + " Dinner is at 9."
    reply = listed(SCHOOL_RUN, **{"mum / mother": ("wrong", "Her mum is coming at 7"),
                                  "invented": ["Dinner is at 9."]})
    v = verdict(SCHOOL_RUN, ans, reply)
    assert (v["score"], v["reason"]) == (2, "2 of 4: wrong: mum / mother (“Her mum is coming at "
                                            "7”) · invented or wrong: “Dinner is at 9.” (−2)")


@pytest.mark.parametrize("answer,fact,has", [
    ("overtime is capped", ["10 hours", "overtime"], False),
    ("overtime is capped at 10 hours", ["10 hours", "overtime"], True),
    ("overtime capped at ten hours", ["10 hours", "overtime"], True),
    ("the group is capped at 12", ["twelve", "cap"], True),
    ("drinks at 6", ["6 pm", "6:00 pm"], True),
    ("drinks at 6:00 PM", ["6 pm", "6:00 pm"], True),
    ("moved to phase 2", ["phase 2", "november"], True),
    ("moved to November", ["phase 2", "november"], True),
    ("moved to a later phase", ["phase 2", "november"], False),
    ("sign the card", ["card", "nadia"], False),
    ("Nadia has the card", ["card", "nadia"], True),
    ("her mum comes at 7", ["mum", "mother"], True),
])
def test_a_fact_with_a_number_or_a_name_needs_it(answer, fact, has):
    named = ev.names('The Hamilton file is with Mariam. The card is with Nadia. Phase 2 is in '
                     'November. Your mum comes at 7.')
    assert ev.has_fact(answer, fact, named) is has


def test_the_names_of_a_text():
    named = ev.names(BANK[LEAVE]["prompt"])
    assert {"mariam", "hamilton", "nov"} <= named
    assert "mum" not in ev.names(BANK[SCHOOL_RUN]["prompt"])       # "Your mum" isn't a name
    # a month is its name or its short form, and nothing that starts like one
    assert ev.key_words("Mariam's novel, 3rd November, Sept 29") >= {"mariam", "novel", "3",
                                                                      "nov", "sep", "29"}


@pytest.mark.parametrize("answer,fact,has", [
    # a word that starts like a month is not that month…
    ("Mariam has it", ["3 mar", "3 march"], False),
    ("Mariam 3", ["3 mar", "3 march"], False),
    ("a novel, chapter 12", ["12 nov", "12 november"], False),
    ("the novel's 12 chapters", ["november 12"], False),
    # …and a month's name or its short form still is
    ("Mar 3", ["3 mar", "3 march"], True),
    ("3 March", ["3 mar", "3 march"], True),
    ("Nov 12", ["12 nov", "12 november"], True),
    ("12 November", ["november 12"], True),
    ("Sept 29", ["29 september"], True),
])
def test_a_month_is_its_name_or_short_form_only(answer, fact, has):
    assert ev.has_fact(answer, fact, ev.names("Mariam reads a novel from 3 March to 12 November."))\
        is has


@pytest.mark.parametrize("reply", ['{"missing_facts": [], "invented_or_wrong": []}',
                                   '{"checklist": [{"fact": "cake", "status": "maybe"}]}',
                                   '{"checklist": "all correct"}', "all 8 key facts"])
def test_a_reply_that_isnt_a_checklist_is_not_read(reply):
    assert verdict(SCHOOL_RUN, SCHOOL_RUN_ANSWER, reply) is None


# ---------------------------------------------------------------------------
# the rubric, on both halves
# ---------------------------------------------------------------------------

def test_the_rubric_is_a_checklist_with_its_worked_examples():
    c = ev.judge_check(BANK[SCHOOL_RUN])
    assert c["checklist"] and not c.get("findings")
    r = c["rubric"]
    assert "It should keep all 4 of these:\n- 5:30 / 5.30\n- mum / mother\n- transit card\n- cake\n" \
        in r
    assert '"She picks up Layla" is wrong when the text says you pick her up' in r
    assert '("overtime is capped" for "overtime is capped at 10 hours")' in r
    assert '{"fact": "Sam / booking", "status": "wrong", "quote": "You\'re booking the table"}' \
        in r
    p = ev.judge_prompt(BANK[SCHOOL_RUN], "x")
    assert p.startswith(ev.CHECKLIST_PROMPT.split("{{", 1)[0]) and '"checklist": [' in p
    assert ev.judge_tokens(BANK[SCHOOL_RUN]) == ev.CHECKLIST_TOKENS
    # the repo's practice rows carry it
    rows = [json.loads(x) for x in ev.BANK_PATH.read_text(encoding="utf-8").splitlines() if x]
    assert all(ev.judge_check(q).get("checklist") for q in rows if q["group"] == "summarising")


def test_the_hidden_rows_12a8_wrote_take_the_checklist_as_they_load():
    """the server's hidden set carries 12a.8's rubric in its store; it is
    today's as it loads, as the invented one here shows — nothing rewritten"""
    raw = [json.loads(x) for x in HIDDEN_FIXTURE.read_text(encoding="utf-8").splitlines() if x]
    summ = [q for q in raw if q["group"] == "summarising"]
    assert summ and all(ev.judge_check(q).get("findings") for q in summ)
    for q in summ:
        now = BANK[q["id"]]
        assert ev.half(now) == ev.HIDDEN
        assert ev.judge_check(now)["checklist"] and ev.judge_check(now)["rubric"] == \
            ev.summarise_rubric(now)
    # a rubric someone wrote stays theirs, as it did
    own = {**summ[0], "checks": [{"type": "judge", "scale": 4, "pass_at": 3, "findings": True,
                                  "rubric": "my own rubric"}]}
    assert ev._valid(json.loads(json.dumps(own)), "x")["checks"][0]["rubric"] == "my own rubric"


def test_every_reference_passes_its_own_checklist():
    """the stand-in judge on each Summarise question's reference, both halves"""
    for q in BANK.values():
        if q["group"] == "summarising":
            v = ev.stub_for(q, q["reference"])
            assert v["pass"], (q["id"], v["reason"])


# ---------------------------------------------------------------------------
# the re-mark
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    yield client
    client.__exit__(None, None, None)


def test_the_remark_asks_every_summarise_answer_again_as_a_checklist(svc):
    """a verdict on 12a.8's rubric is on another rubric: both halves are asked
    again, with room for a checklist, and the marks before are kept"""
    qs = [q for q in BANK.values() if q["group"] == "summarising"]
    mdir = config.OUT_DIR / "org__m"
    answer = lambda q: QWEN_EMAIL if q["id"] == EMAIL else q["reference"]  # noqa: E731
    _asked(mdir, qs, answer=answer)
    was = {q["id"]: ev.rubric_key({"checks": [{"type": "judge", "rubric": ev.summarise_rubric(
        q, ev.RUBRIC_12A8)}]}) for q in qs}
    ev.write(mdir, {"model": "org/m", "items": [
        {"id": q["id"], "pass": True, "reason": "4 of 4: all key facts, one version", "score": 4,
         "rubric": was[q["id"]], "answer_text": answer(q)} for q in qs]})
    res = ev.remark(config.OUT_DIR, judge=True)
    assert res["sent"] == len(qs) and (config.OUT_DIR / ev.BEFORE_NAME).exists()
    assert ev.BEFORE_NAME == "everyday_before_12a9.json"
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith(ev.REMARK + ":")]
    assert {r["custom_id"].rsplit(":", 1)[1] for r in sent} == {q["id"] for q in qs}
    assert {r["max_tokens"] for r in sent} == {ev.CHECKLIST_TOKENS}
    assert all(r["user"].startswith(ev.CHECKLIST_PROMPT.split("{{", 1)[0]) for r in sent)
    # the replies land: the email thread's, as the brief found it, fails
    replies = {r["custom_id"]: llm.Result(text=ev.stub_reply(r["user"])) for r in sent}
    cid = next(c for c in replies if c.endswith(":" + EMAIL))
    replies[cid] = llm.Result(text=listed(EMAIL))
    ev.finish_remark(config.OUT_DIR, replies)
    items = {it["id"]: it for it in ev.read(mdir)["items"]}
    assert {ev.half(BANK[i]) for i in items} == {ev.HIDDEN, ev.PRACTICE}
    assert all(it["rubric"] == ev.rubric_key(BANK[i]) and it["findings"]["checklist"]
               for i, it in items.items())
    assert (items[EMAIL]["pass"], items[EMAIL]["score"]) == (False, 2)


def test_compare_says_which_change_moved_which_scores(svc):
    """the brief: split --compare by cause. On one stored checklist each, the
    columns add 12a.9's changes in turn — the school run plan fails on the
    checklist, a four-bullet answer too long on the length now read, the email
    thread on every fact counting — and the marks before are each answer's"""
    qs = [q for q in BANK.values() if q["group"] == "summarising"]
    FOUR = "everyday-summarising-long-14"                          # "as 4 bullets"
    special = {SCHOOL_RUN: GEMMA_SCHOOL_RUN, EMAIL: QWEN_EMAIL}
    answer = lambda q: special.get(q["id"], q["reference"])  # noqa: E731
    mdir = config.OUT_DIR / "org__m"
    _asked(mdir, qs, answer=answer)
    ev.write(mdir, {"model": "org/m", "items": [
        {"id": q["id"], "group": "summarising", "pass": True, "score": 4,
         "reason": "4 of 4: all key facts, one version", "answer_text": answer(q),
         "rubric": ev.rubric_key({"checks": [{"type": "judge", "rubric": ev.summarise_rubric(
             q, ev.RUBRIC_12A8)}]})} for q in qs]})
    ev.remark(config.OUT_DIR, judge=True)
    snap = json.loads((config.OUT_DIR / ev.BEFORE_NAME).read_text())["summarise"]["org/m"]
    assert set(snap) == {q["id"] for q in qs} and all(snap.values())
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith(ev.REMARK + ":org__m:")]
    replies = {r["custom_id"]: llm.Result(text=ev.stub_reply(r["user"])) for r in sent}
    by = {c.rsplit(":", 1)[1]: c for c in replies}
    replies[by[SCHOOL_RUN]] = llm.Result(text=listed(SCHOOL_RUN, **{
        "5:30 / 5.30": ("wrong", "will pick up Layla from her friend's house at 5:30"),
        "mum / mother": ("wrong", "Her mum is coming at 7")}))
    replies[by[EMAIL]] = llm.Result(text=listed(EMAIL))
    replies[by[FOUR]] = llm.Result(text=listed(FOUR, several=True, length="no"))
    ev.finish_remark(config.OUT_DIR, replies)
    items = {it["id"]: it for it in ev.read(mdir)["items"]}
    assert [items[i]["score"] for i in (SCHOOL_RUN, FOUR, EMAIL)] == [2, 2, 2]
    out = ev.compare(config.OUT_DIR).splitlines()
    k = out.index("Summarise, what moved it: each column adds one change to the column before")
    assert out[k + 2] == ("| model | half | before | the checklist | + lengths “as 4 bullets” "
                          "asks | + every fact where no length is set (now) | waiting |")
    n = {h: sum(1 for q in qs if ev.half(q) == h) for h in (ev.HIDDEN, ev.PRACTICE)}
    p = n[ev.PRACTICE]
    assert f"| org/m | practice | {p} of {p} | {p - 1} (−1) | {p - 2} (−1) | {p - 3} (−1) | 0 |" \
        in out
    h = n[ev.HIDDEN]
    assert f"| org/m | hidden | {h} of {h} | {h} | {h} | {h} | 0 |" in out
    # which questions each change reaches: practice ones named, hidden ones counted
    assert out[-2:] == [
        "Lengths “as 4 bullets” asks reach: long-14, long-16, long-30, long-33, long-44 in the "
        "practice half; 0 hidden.",
        "Every fact where no length is set reaches: long-01, long-08, long-11, long-15, long-22, "
        "long-28 in the practice half; 0 hidden."]


def test_a_hidden_question_a_change_reaches_is_counted_never_named(tmp_path, monkeypatch):
    x = {"id": "everyday-summarising-invented-x", "group": "summarising", "half": ev.HIDDEN,
         "prompt": 'give me the main points as 3 bullets: "Fixture text, invented for the tests."',
         "reference": "Fixture text."}
    x["checks"] = ev.summarise_checks({**x, "checks": [
        {"type": "facts", "values": [["fixture"], ["text"], ["invented"], ["tests"]], "n": 2}]})
    real = ev.load_bank
    monkeypatch.setattr(ev, "load_bank", lambda *a, **k: real(*a, **k) + [x])
    (tmp_path / ev.BEFORE_NAME).write_text(json.dumps({"models": {}, "summarise": {"org/x": {}}}))
    out = ev.moved(tmp_path)
    assert "invented-x" not in out
    assert "Lengths “as 4 bullets” asks reach: long-14, long-16, long-30, long-33, long-44 in the " \
           "practice half; 1 hidden." in out
