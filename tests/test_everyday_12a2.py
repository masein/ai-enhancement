"""12a.2: Everyday tasks round 2 — one bank of 111 questions in seven groups,
marked by a general check vocabulary. docs/prompts/phase-12b3/checks.py is the
reference checker; scripts/everyday.py must reach its verdict on every probe
in everyday_round2_probes.jsonl, and every question's reference must pass its
own script checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import everyday as ev

REPO = Path(__file__).resolve().parents[1]
BRIEF = REPO / "docs" / "prompts" / "phase-12b3"
PROBES = [json.loads(line) for line in
          (BRIEF / "everyday_round2_probes.jsonl").read_text(encoding="utf-8").splitlines()
          if line.strip()]
BANK = {q["id"]: q for q in ev.load_bank()}


def reference_checker():
    spec = importlib.util.spec_from_file_location("checks_ref", BRIEF / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def script_verdict(item, answer):
    """what the script checks say, the judge left out: a probe on a judged
    question expects the script checks' verdict"""
    res = [ev.run_check(c, answer, item["prompt"]) for c in item["checks"] if c["type"] != "judge"]
    return all(ok for ok, _ in res)


def test_the_bank_is_111_questions_in_seven_groups():
    qs = ev.load_bank()
    assert len(qs) == 111
    assert list(ev.GROUPS) == ["understanding", "writing", "summarising", "transform",
                               "quick_maths", "instructions", "honesty"]
    assert list(ev.GROUPS.values()) == ["Understanding", "Writing", "Summarising", "Transform",
                                        "Quick maths", "Instructions", "Honesty"]
    sizes = {g: sum(1 for q in qs if q["group"] == g) for g in ev.GROUPS}
    assert all(15 <= n <= 17 for n in sizes.values()), sizes
    # the pilot's five joined it, and the round's 106 are all there
    assert [q["id"] for q in qs if q["id"].startswith("everyday-pilot-")] == \
        [f"everyday-pilot-0{i}" for i in (1, 4, 3, 2, 5)]           # in the groups' order
    round2 = [json.loads(line)["id"] for line in
              (BRIEF / "everyday_round2.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert set(round2) <= set(BANK) and len(round2) == 106
    # no split yet: every question readable; adding one later is a data change
    assert not any("split" in q for q in qs)


@pytest.mark.parametrize("i", range(len(PROBES)), ids=[f"{p['id']}#{n}" for n, p in enumerate(PROBES)])
def test_every_probe_gets_its_verdict(i):
    p = PROBES[i]
    item = BANK[p["id"]]
    got = script_verdict(item, p["answer"])
    assert got == (p["expected"] == "pass"), (p["answer"], [
        ev.run_check(c, p["answer"], item["prompt"]) for c in item["checks"]])
    # a probe expecting a pass does not fail on a script check, and a judge
    # question's pass waits on the judge
    ok, why = ev.grade(item, p["answer"])
    if p["expected"] == "pass":
        assert ok is not False, why
        assert ok is (None if any(c["type"] == "judge" for c in item["checks"]) else True)
    else:
        assert ok is False and why


def test_the_port_decides_as_checks_py_does_on_every_probe():
    ref = reference_checker()
    for p in PROBES:
        item = BANK[p["id"]]
        for c in item["checks"]:
            mine = ev.run_check(c, p["answer"], item["prompt"])
            theirs = ref.run(c, p["answer"], item["prompt"])
            assert mine == theirs, (p["id"], c, p["answer"])


def test_every_reference_passes_its_own_script_checks():
    bad = [q["id"] for q in BANK.values() if not script_verdict(q, q["reference"])]
    assert bad == []


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
