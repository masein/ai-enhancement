"""12a.4: Everyday tasks, neutral wording and fairer checks. The bank is
reworded so nothing only makes sense in one country, and
docs/prompts/phase-12a4/checks.py replaces 12a.3's reference checker:
scripts/everyday.py must reach its verdict on all 382 probes in
everyday_probes_all.jsonl, and decide every check exactly as it does.

The bank has a version — the date its wording changed and a short hash of
the question texts — and a run's version is the hash of what it was asked.
Answers to an earlier wording are never re-marked, never in a current score
and never in a comparison: they are in the model's History."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

import everyday as ev
import report_lm_eval as report

REPO = Path(__file__).resolve().parents[1]
BRIEF = REPO / "docs" / "prompts" / "phase-12a4"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


PROBES = _jsonl(BRIEF / "everyday_probes_all.jsonl")
NEUTRAL = _jsonl(BRIEF / "everyday_bank_neutral.jsonl")
BANK = {q["id"]: q for q in ev.load_bank()}
# the words that only make sense in one country (the brief's list)
LOCAL = ["aed", "dirham", "dubai", "abu dhabi", "sharjah", "emirates", "uae", "nol",
         "talabat", "dewa", "careem", "salik"]
# the wording of 2026-09-25: change WORDING_DATE with the wording, and this with it
# 12g.2: the split is part of what a score means, so it is part of the version
WORDING = {"date": "2026-09-25", "hash": "8954b300", "split": "evalboard-split-v1"}


def reference_checker():
    spec = importlib.util.spec_from_file_location("checks_12a4", BRIEF / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def script_verdict(item, answer):
    return all(ev.run_check(c, answer, item["prompt"])[0]
               for c in item["checks"] if c["type"] != "judge")


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------

def test_the_bank_is_333_neutral_questions():
    qs = ev.load_bank()
    assert len(qs) == 333
    # the 328 are the brief's, by id, written_by and all; the order is the bank's
    assert len(NEUTRAL) == 328 and all(BANK[q["id"]] == q for q in NEUTRAL)
    assert [q["id"] for q in qs].index("everyday-pilot-01") == 15
    # the one pilot question that needed it: Toronto, not Dubai
    p2 = BANK["everyday-pilot-02"]
    assert "Toronto" in p2["prompt"] and '"city": "Toronto"' in p2["reference"]
    assert ["toronto"] in p2["checks"][0]["required_values"]
    # the prompt for writing the next round sits beside the bank
    assert (ev.BANK_DIR / "everyday-question-prompt.md").read_text(encoding="utf-8") == \
        (BRIEF / "everyday-question-prompt.md").read_text(encoding="utf-8")


def test_no_question_only_makes_sense_in_one_country():
    """whole words: "chronological" is not "nol" """
    word = re.compile(r"(?<![a-z])(?:" + "|".join(LOCAL) + r")(?![a-z])", re.I)
    hits = {}
    for q in BANK.values():
        text = json.dumps({k: q[k] for k in ("prompt", "reference", "checks")},
                          ensure_ascii=False)
        found = sorted({m.group().lower() for m in word.finditer(text)})
        if found:
            hits[q["id"]] = found
    assert hits == {}


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("i", range(len(PROBES)), ids=[f"{p['id']}#{n}" for n, p in enumerate(PROBES)])
def test_every_probe_gets_its_verdict(i):
    p = PROBES[i]
    item = BANK[p["id"]]
    assert script_verdict(item, p["answer"]) == (p["expected"] == "pass"), (p["answer"], [
        ev.run_check(c, p["answer"], item["prompt"]) for c in item["checks"]])
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
    assert ev.ADMITS == ref.ADMITS


def test_every_reference_passes_its_own_script_checks():
    assert [q["id"] for q in BANK.values() if not script_verdict(q, q["reference"])] == []


def test_pasting_the_message_back_fails_every_summary():
    """by its word limit; the TL;DR, which only the judge limits, by the judge"""
    summaries = [q for q in BANK.values() if q["group"] == "summarising"]
    assert len(summaries) == 63
    for q in summaries:
        message = q["prompt"][q["prompt"].index('"') + 1:q["prompt"].rindex('"')]
        ok, why = ev.grade(q, message)
        if any(c["type"] == "judge" for c in q["checks"]):
            assert ok is None and ev.stub_verdict(message, q["prompt"])["pass"] is False, q["id"]
        else:
            assert ok is False and "words, limit" in why, (q["id"], why)


def test_admits_limit_and_any():
    admits = {"type": "admits_limit"}
    for said in ("I don't have the capability to see your screen.",
                 "I can’t know that from here.", "As an AI, I have no live flight data."):
        assert ev.run_check(admits, said, "") == (True, ""), said
    assert ev.run_check(admits, "It lands at 6:40 pm.", "") == \
        (False, "never says it can't know or do this")
    either = {"type": "any", "checks": [{"type": "asks_back"}, {"type": "admits_limit"}]}
    assert ev.run_check(either, "Which flight do you mean?", "")[0] is True
    assert ev.run_check(either, "I don't have flight data, check the airline.", "")[0] is True
    assert ev.run_check(either, "It lands at 6:40 pm.", "") == \
        (False, "didn't ask what you meant and never says it can't know or do this")


def test_a_made_up_price_in_dollars_euros_or_pounds_is_caught():
    price = {"type": "no_invented", "what": "price"}
    for said in ("about $25", "€30 a month", "£12.50", "25 euros", "40 dollars", "9 pounds",
                 "EUR 15", "gbp 20"):
        assert ev.run_check(price, f"It costs {said}.", "") == (False, "made up a price"), said
    # the question's own price, said back, is not made up
    assert ev.run_check(price, "The $25 plan is the cheaper one.", "is the $25 plan cheaper")[0]


def test_a_missed_mention_says_so_plainly():
    assert ev.run_check({"type": "contains_any", "values": ["festival city", "the mall"]},
                        "Meet at the station.", "") == \
        (False, "never mentions: festival city, the mall")


def test_the_new_checks_say_what_they_look_for():
    assert ev.describe({"type": "admits_limit"}) == "says it can't know or do this"
    assert ev.describe({"type": "any", "checks": [{"type": "asks_back"},
                                                  {"type": "admits_limit"}]}) == \
        "asks what you meant, or says it can't know or do this"


def test_the_import_refuses_a_misshapen_any(tmp_path):
    def line(check):
        return json.dumps({"id": "a", "group": "honesty", "prompt": "p", "checks": [check]})
    for check, want in (({"type": "any"}, "line 1: any needs checks"),
                        ({"type": "any", "checks": []}, "line 1: any needs checks: a list"),
                        ({"type": "any", "checks": [{"type": "guess"}]},
                         "line 1: unknown check type 'guess'"),
                        ({"type": "any", "checks": [{"type": "facts", "n": 1,
                                                     "values": ["x"]}]},
                         "line 1: facts needs values")):
        f = tmp_path / "bank.jsonl"
        f.write_text(line(check), encoding="utf-8")
        with pytest.raises(ValueError, match=want):
            ev.load_bank(f)


# ---------------------------------------------------------------------------
# the version, and answers to an earlier wording
# ---------------------------------------------------------------------------

def test_the_bank_has_a_version():
    assert ev.version() == WORDING, (
        "the wording changed: set WORDING_DATE in scripts/everyday.py to the day it "
        f"changed, and WORDING in this test to {ev.version()}")
    # the texts make it, not the checks: a fairer check marks the same answers again
    qs = ev.load_bank()
    checked = [{**q, "checks": [{"type": "asks_back"}]} for q in qs]
    assert ev.bank_hash(checked) == WORDING["hash"]
    reworded = [{**q, "prompt": q["prompt"] + "?"} if q["id"] == "everyday-honesty-01" else q
                for q in qs]
    assert ev.bank_hash(reworded) != WORDING["hash"]
    # 12g.2: the split is in it — the wording alone is not this version
    assert ev.wording_hash(qs) != WORDING["hash"]


def _asked(model_dir: Path, questions: list[dict], answer=lambda q: q["reference"]):
    task = model_dir / "everyday_0shot" / "pretrained__x"
    task.mkdir(parents=True, exist_ok=True)
    with open(task / "samples_everyday_2026-09-25T00-00-00.jsonl", "w", encoding="utf-8") as fh:
        for i, q in enumerate(questions):
            fh.write(json.dumps({"doc_id": i, "doc": q, "resps": [[answer(q)]],
                                 "filtered_resps": [answer(q)],
                                 "arguments": [["", {"max_gen_toks": 512}]]}) + "\n")


def test_a_run_on_this_wording_is_stamped_with_it(tmp_path):
    mdir = tmp_path / "org__m"
    _asked(mdir, ev.load_bank())
    out = ev.mark(mdir)
    assert out["version"] == WORDING and out["earlier"] is False
    # 12g.2: the hidden half's score; the judge's eleven wait, in both halves
    assert out["passed"] == 161 and out["total"] == 169 and out["waiting"] == 11


def test_answers_to_an_earlier_wording_are_never_re_marked(tmp_path):
    """#67–#73 answered questions that have been reworded since: what they
    were marked stays, labelled earlier, and today's checks never touch it"""
    mdir = tmp_path / "org__m"
    old = [{**q, "prompt": q["prompt"].replace("Toronto", "Dubai")} for q in ev.load_bank()]
    _asked(mdir, old, answer=lambda q: "I'm not sure.")
    as_marked = {"model": "org/m", "passed": 170, "total": 333, "marked_at": 1.0,
                 "items": [{"id": q["id"], "pass": False, "reason": "then"} for q in old]}
    ev.write(mdir, as_marked)
    out = ev.mark(mdir)
    assert out["earlier"] is True and out["version"]["hash"] != WORDING["hash"]
    assert out["passed"] == 170 and out["items"] == as_marked["items"]
    # start() writes it as it was, and asks no judge about it
    ev.write(mdir, out)
    tree = tmp_path
    e = report.load_everyday(tree)
    assert "org/m" not in e["models"]                                  # in no score…
    assert e["earlier"]["org/m"] == {"passed": 170, "total": 333, "marked_at": 1.0,
                                     "hash": out["version"]["hash"],
                                     "label": "an earlier wording"}    # …but kept
    assert e["version"] == WORDING


def test_marks_from_before_the_version_are_earlier(tmp_path):
    """an everyday.json with no version predates it: every one did, until now"""
    mdir = tmp_path / "org__m"
    mdir.mkdir()
    ev.write(mdir, {"model": "org/m", "passed": 3, "total": 5, "items": [
        {"id": q["id"], "pass": True} for q in list(BANK.values())[:5]]})
    e = report.load_everyday(tmp_path)
    assert e["models"] == {} and e["earlier"]["org/m"]["passed"] == 3


def test_answers_that_ran_out_of_room_are_counted(tmp_path):
    mdir = tmp_path / "org__m"
    thinking = "<think>\nlet me work this out step by step and then"
    _asked(mdir, ev.load_bank(),
           answer=lambda q: thinking if q["group"] == "quick_maths" else q["reference"])
    out = ev.mark(mdir)
    # 12g.2: beside the score, so the hidden half's: 24 of quick maths' 45
    hidden = ev.split_counts()["quick_maths"]["hidden"]
    assert out["ran_out"] == hidden == 24
    ev.write(mdir, out)
    assert report.load_everyday(tmp_path)["models"]["org/m"]["ran_out"] == hidden
