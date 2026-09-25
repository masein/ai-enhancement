"""12a.2: Everyday tasks round 2 — one bank in seven groups, marked by a
general check vocabulary. 12a.3 replaced the reference checker
(docs/prompts/phase-12a3/checks.py) and grew the bank to 333: the probes —
round 2's 180 among the 366 — the port against checks.py and the references
are tests/test_everyday_12a3.py's now. What 12a.2 added and still holds is
here: its rules, its import's refusals, and every check in plain words."""

from __future__ import annotations

import json

import pytest

import everyday as ev

BANK = {q["id"]: q for q in ev.load_bank()}


def test_the_rules_the_brief_names():
    # whole words, case-insensitive unless asked
    two = {"type": "contains_any", "values": ["2 november"]}
    assert ev.run_check(two, "on 22 November", "")[0] is False
    assert ev.run_check(two, "On 2 November.", "")[0] is True
    cs = {"type": "contains_all", "values": ["NASA"], "case_sensitive": True}
    assert ev.run_check(cs, "nasa said", "")[0] is False
    # am/pm on both sides
    assert ev.run_check({"type": "contains_all", "values": ["12:30 pm"]}, "at 12:30pm", "")[0]
    # thousands commas
    assert ev.run_check({"type": "number", "value": 1284.5, "tolerance": 0.01}, "AED 1,284.50", "")[0]
    # counts ignore fences and one lead-in line
    assert ev.run_check({"type": "line_count", "n": 2}, "Here you go:\n```\na\nb\n```", "")[0]
    # json: an array, fenced
    assert ev.run_check({"type": "json", "required_values": [["dubai"]]},
                        '```json\n[{"city": "Dubai"}]\n```', "")[0]
    # in_order: alternatives, anywhere
    assert ev.run_check({"type": "in_order", "values": [["first", "1st"], "then"]},
                        "1st do this, then that", "")[0]
    # no_invented price, both ways round and in millions
    for s in ("AED 80", "80 AED", "1.2 million AED"):
        assert ev.run_check({"type": "no_invented", "what": "price"}, f"about {s}", "")[0] is False


def test_the_import_refuses_a_bad_bank_and_names_the_line(tmp_path):
    good = json.dumps({"id": "a", "group": "writing", "prompt": "p",
                       "checks": [{"type": "max_words", "n": 3}]})
    cases = [
        (good + "\n{not json", "line 2: not valid JSON"),
        (good + "\n" + good, "line 2: a is there twice"),
        (good.replace('"max_words"', '"telepathy"'), "line 1: unknown check type 'telepathy'"),
        (good.replace('"writing"', '"poetry"'), "line 1: unknown group 'poetry'"),
        (good.replace(', "n": 3', ''), "line 1: max_words needs n"),
        (json.dumps({"id": "b", "group": "writing", "prompt": "p", "checks": []}), "line 1: no checks"),
    ]
    for text, want in cases:
        f = tmp_path / "bank.jsonl"
        f.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=want.replace("(", r"\(")):
            ev.load_bank(f)


def test_every_check_says_what_it_looks_for_in_plain_words():
    for q in BANK.values():
        for c in q["checks"]:
            words = ev.describe(c)
            assert words and "_" not in words.split(":")[0], (q["id"], c, words)
    assert ev.describe({"type": "number", "value": 29, "tolerance": 0}) == "says 29"
    assert ev.describe({"type": "no_invented", "what": "phone"}) == "no invented phone"
