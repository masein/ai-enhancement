"""12a.3: Everyday tasks round 3 — the bank grows from 111 to 333 questions,
and docs/prompts/phase-12a3/checks.py replaces 12a.2's reference checker.
scripts/everyday.py must reach its verdict on every one of the 366 probes in
everyday_probes_all.jsonl (round 2's 180 and round 3's 186), and decide every
check exactly as it does."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import everyday as ev

REPO = Path(__file__).resolve().parents[1]
BRIEF = REPO / "docs" / "prompts" / "phase-12a3"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


PROBES = _jsonl(BRIEF / "everyday_probes_all.jsonl")
ROUND3 = _jsonl(BRIEF / "everyday_round3.jsonl")
ROUND2 = _jsonl(REPO / "docs" / "prompts" / "phase-12b3" / "everyday_round2.jsonl")
BANK = {q["id"]: q for q in ev.load_bank()}
# the brief's table: round 2 and the pilot, plus round 3
SIZES = {"understanding": 45, "writing": 48, "summarising": 63, "transform": 46,
         "quick_maths": 45, "instructions": 45, "honesty": 41}


def reference_checker():
    spec = importlib.util.spec_from_file_location("checks_12a3", BRIEF / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def script_verdict(item, answer):
    """what the script checks say, the judge left out: a probe on a judged
    question expects the script checks' verdict"""
    return all(ev.run_check(c, answer, item["prompt"])[0]
               for c in item["checks"] if c["type"] != "judge")


def test_the_bank_is_333_questions_in_seven_groups():
    qs = ev.load_bank()
    assert len(qs) == 333
    assert {g: sum(q["group"] == g for q in qs) for g in ev.GROUPS} == SIZES
    # round 3's 222 as they were written, written_by and all; round 2's 106
    # and the pilot's five still there
    assert len(ROUND3) == 222 and all(BANK[q["id"]] == q for q in ROUND3)
    assert {q["id"] for q in ROUND2} <= set(BANK) and len(ROUND2) == 106
    assert sum(q["id"].startswith("everyday-pilot-") for q in qs) == 5
    assert all(q["written_by"] for q in ROUND3)
    # in the groups' order, each group's older questions before round 3's:
    # the pilot's first is still question 16
    groups = [q["group"] for q in qs]
    assert groups == sorted(groups, key=list(ev.GROUPS).index)
    for g in ev.GROUPS:
        r3 = ["-r3-" in q["id"] for q in qs if q["group"] == g]
        assert r3 == sorted(r3)
    assert [q["id"] for q in qs].index("everyday-pilot-01") == 15
    # no split yet: 12g.2 assigns the halves
    assert not any("split" in q for q in qs)


@pytest.mark.parametrize("i", range(len(PROBES)), ids=[f"{p['id']}#{n}" for n, p in enumerate(PROBES)])
def test_every_probe_gets_its_verdict(i):
    p = PROBES[i]
    item = BANK[p["id"]]
    assert script_verdict(item, p["answer"]) == (p["expected"] == "pass"), (p["answer"], [
        ev.run_check(c, p["answer"], item["prompt"]) for c in item["checks"]])
    # a pass never fails a script check, and a judged question's pass waits on
    # the judge; a fail says why
    ok, why = ev.grade(item, p["answer"])
    if p["expected"] == "pass":
        assert ok is (None if any(c["type"] == "judge" for c in item["checks"]) else True), why
    else:
        assert ok is False and why


def test_the_port_decides_as_checks_py_does():
    """every check of every question, on every probe and on the question's own
    reference: the same verdict and the same reason"""
    ref = reference_checker()
    n = 0
    for p in PROBES + [{"id": q["id"], "answer": q["reference"]} for q in BANK.values()]:
        item = BANK[p["id"]]
        for c in item["checks"]:
            assert ev.run_check(c, p["answer"], item["prompt"]) == \
                ref.run(c, p["answer"], item["prompt"]), (p["id"], c, p["answer"])
            n += 1
    assert n > 1000


def test_every_reference_passes_its_own_script_checks():
    assert [q["id"] for q in BANK.values() if not script_verdict(q, q["reference"])] == []


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
