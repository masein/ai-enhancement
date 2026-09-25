"""12a.3: Everyday tasks round 3 — the bank grew from 111 to 333 questions,
with two new check types and time ranges read as both times. 12a.4 reworded
the bank and replaced the reference checker again: the probes, the port
against checks.py, the references and the bank itself are
tests/test_everyday_12a4.py's now. What round 3 added and still holds is
here: its rules, its two check types' words, its import's refusals, and its
summaries and writing questions failing an answer that only echoes."""

from __future__ import annotations

import json

import pytest

import everyday as ev

BANK = {q["id"]: q for q in ev.load_bank()}
ROUND3 = [q for q in BANK.values() if "-r3-" in q["id"]]


def test_round3_is_222_of_the_bank():
    assert len(ROUND3) == 222 and all(q["written_by"] for q in ROUND3)


def test_pasting_the_message_back_fails_every_round3_summary_on_its_word_limit():
    summaries = [q for q in ROUND3 if q["group"] == "summarising"]
    assert len(summaries) == 46
    for q in summaries:
        message = q["prompt"][q["prompt"].index('"') + 1:q["prompt"].rindex('"')]
        failed = [c["type"] for c in q["checks"]
                  if ev.run_check(c, message, q["prompt"])[0] is False]
        assert failed == ["max_words"], (q["id"], failed)      # it keeps every fact
        assert ev.grade(q, message)[0] is False


def test_repeating_the_request_back_fails_every_round3_writing_question():
    writing = [q for q in ROUND3 if q["group"] == "writing"]
    assert len(writing) == 32
    for q in writing:
        ok, why = ev.grade(q, q["prompt"])
        assert ok is False and why, q["id"]


def test_the_changes_the_brief_names():
    # a range says each of its times, in every check that matches times
    for text, t in (("open 7–11 am", "7 am"), ("open 7-11 am", "11 am"), ("from 2–3pm", "2 pm"),
                    ("9:30–11:30 am slot", "9:30 am"), ("9:30–11:30 am slot", "11:30 am")):
        assert ev.run_check({"type": "contains_all", "values": [t]}, text, "")[0], (text, t)
    assert not ev.run_check({"type": "contains_all", "values": ["7 pm"]}, "7–11 am", "")[0]
    assert ev.run_check({"type": "in_order", "values": ["7 am", "11 am"]}, "7-11 am", "")[0]
    # in_order reads am/pm as the contains checks do
    assert ev.run_check({"type": "in_order", "values": ["8 am", "pickup"]}, "8am pickup", "")[0]
    # no_invented: the question's own number, price or link repeated back is not invented
    q = "this missed call was from 050 712 8841 who is it"
    phone = {"type": "no_invented", "what": "phone"}
    assert ev.run_check(phone, "I can't tell who 050 712 8841 is.", q) == (True, "")
    assert ev.run_check(phone, "It is Etisalat on 050 712 8000.", q) == \
        (False, "made up a phone")
    url = {"type": "no_invented", "what": "url"}
    assert ev.run_check(url, "see www.rta.ae", "is www.rta.ae down")[0]
    assert not ev.run_check(url, "see www.rta.ae", "is the metro down")[0]
    # facts: at least n, and the reason names what went missing
    facts = {"type": "facts", "n": 4,
             "values": [["saturday"], ["7 am", "7am"], ["mirdif"], ["gate 3"], ["bus"]]}
    assert ev.run_check(facts, "Saturday, gate 3, by bus.", "") == \
        (False, "kept 3 of 5 key facts, needs 4 (missing: 7 am, mirdif)")
    assert ev.run_check(facts, "Saturday 7–9 am at gate 3, by bus.", "")[0] is True
    # first_mention: the right one before any wrong one
    pick = {"type": "first_mention", "right": ["Nadia"], "wrong": ["Nabil"]}
    assert ev.run_check(pick, "Message Nadia, not Nabil.", "") == (True, "")
    assert ev.run_check(pick, "Nabil.", "") == (False, "never names Nadia")
    assert ev.run_check(pick, "Nabil, then Nadia.", "") == (False, "names Nabil first")


def test_the_new_checks_say_what_they_look_for():
    assert ev.describe({"type": "facts", "n": 4, "values": [["the date"], ["7 am", "7am"],
                                                            ["mirdif"]]}) == \
        'keeps at least 4 of: "the date", "7 am", "mirdif"'
    assert ev.describe({"type": "first_mention", "right": ["Nadia"], "wrong": ["Nabil"]}) == \
        "picks Nadia, not Nabil"
    # every one in the bank has its words
    for q in BANK.values():
        for c in q["checks"]:
            if c["type"] in ("facts", "first_mention"):
                assert ev.describe(c).startswith(("keeps at least", "picks ")), q["id"]


def test_the_import_refuses_a_misshapen_new_check_and_names_the_line(tmp_path):
    def line(check):
        return json.dumps({"id": "a", "group": "writing", "prompt": "p", "checks": [check]})
    cases = [
        ({"type": "facts", "n": 2, "values": ["saturday", "mirdif"]},
         "line 1: facts needs values: a list of facts, each a list of ways to say it"),
        ({"type": "facts", "n": 3, "values": [["saturday"], ["mirdif"]]},
         "line 1: facts needs n between 1 and 2"),
        ({"type": "facts", "values": [["saturday"]]}, "line 1: facts needs n"),
        ({"type": "first_mention", "right": "Nadia", "wrong": ["Nabil"]},
         "line 1: first_mention needs right: a list of names"),
        ({"type": "first_mention", "right": ["Nadia"]}, "line 1: first_mention needs wrong"),
    ]
    for check, want in cases:
        f = tmp_path / "bank.jsonl"
        f.write_text(line(check), encoding="utf-8")
        with pytest.raises(ValueError, match=want):
            ev.load_bank(f)
