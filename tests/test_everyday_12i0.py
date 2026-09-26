"""12i.0 #12: two rules for numbers_from_source, from the long-summary runs of
2026-09-26 — a note of the answer's own length ("(109 words)") isn't a number
it claims, and "end of October" in the source gives October 31.
docs/prompts/phase-12i/checks.py is the reference checker now: it and
scripts/everyday.py agree on every probe of docs/prompts/phase-12a5/ and on
docs/prompts/phase-12i/probes_12i0.jsonl, whose probes test
numbers_from_source alone. The bank's words don't change, so its version
doesn't: re-marking the stored answers is enough, and no model runs again."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import everyday as ev

REPO = Path(__file__).resolve().parents[1]
BRIEF = REPO / "docs" / "prompts" / "phase-12i"
NUMBERS = {"type": "numbers_from_source"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


PROBES_12A5 = _jsonl(REPO / "docs" / "prompts" / "phase-12a5" / "everyday_probes_12a5.jsonl")
PROBES_12I0 = _jsonl(BRIEF / "probes_12i0.jsonl")
BANK = {q["id"]: q for q in ev.load_bank()}


def reference_checker():
    spec = importlib.util.spec_from_file_location("checks_12i", BRIEF / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_port_decides_as_12i_checks_py_does():
    """every check of every question, on every 12a.5 probe and every
    reference, and numbers_from_source on the 12i.0 probes: the same verdict
    and the same reason"""
    ref = reference_checker()
    n = 0
    for p in PROBES_12A5 + [{"id": q["id"], "answer": q["reference"]} for q in BANK.values()]:
        item = BANK[p["id"]]
        for c in item["checks"]:
            assert ev.run_check(c, p["answer"], item["prompt"]) == \
                ref.run(c, p["answer"], item["prompt"]), (p["id"], c, p["answer"])
            n += 1
    for p in PROBES_12I0:
        prompt = BANK[p["id"]]["prompt"]
        assert ev.run_check(NUMBERS, p["answer"], prompt) == ref.run(NUMBERS, p["answer"], prompt)
        n += 1
    assert n == 3044 + 3


@pytest.mark.parametrize("i", range(len(PROBES_12I0)),
                         ids=[f"{p['id']}#{n}" for n, p in enumerate(PROBES_12I0)])
def test_every_12i0_probe_gets_its_verdict(i):
    p = PROBES_12I0[i]
    assert p["checks"] == "numbers_from_source only"
    ok, why = ev.run_check(NUMBERS, p["answer"], BANK[p["id"]]["prompt"])
    assert ok is (p["expected"] == "pass"), why


PROMPT = ("summarise pls: \"Rent for 14 Maple Park rises to £1,210 a month from 1 February. "
          "Sign the new lease by the end of October, and pay the £200 deposit then.\"")


@pytest.mark.parametrize("note", [
    "(109 words)", "(Word count: 89)", "Word count: 89", "(word count = 12)",
    "— 12 words", "12 words", "(12 words)"])
def test_a_note_of_the_answers_own_length_is_not_a_claim(note):
    answer = f"Rent at 14 Maple Park goes up to £1,210 a month from 1 February. {note}"
    assert ev.run_check(NUMBERS, answer, PROMPT) == (True, "")


def test_a_number_that_is_not_a_length_note_still_counts():
    ok, why = ev.run_check(NUMBERS, "Rent goes up to £1,210 from 1 February 2025.", PROMPT)
    assert (ok, why) == (False, "invented 2025")
    # "words" before the end, or a count that isn't the answer's own, is a claim
    ok, why = ev.run_check(NUMBERS, "Rent rises by 12 words' worth: £1,210 from 1 February, "
                                    "and 7 more things.", PROMPT)
    assert ok is False


def test_end_of_a_month_gives_its_last_day():
    assert ev.run_check(NUMBERS, "Sign the lease by October 31 and pay £200.", PROMPT) == (True, "")
    assert ev.run_check(NUMBERS, "Sign the lease by 30 October.", PROMPT) == \
        (False, "invented 30")
    feb = PROMPT.replace("end of October", "end of February")
    assert ev.run_check(NUMBERS, "Sign by 28 February.", feb) == (True, "")
    jun = PROMPT.replace("end of October", "end of June")
    assert ev.run_check(NUMBERS, "Sign by June 30.", jun) == (True, "")
    assert ev.run_check(NUMBERS, "Sign by June 31.", jun)[0] is False


def test_the_banks_version_does_not_change():
    """a checker change re-marks the answers on file; the words are the same"""
    assert ev.version()["hash"] == "32432393"
