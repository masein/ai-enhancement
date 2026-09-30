"""12a.7: Summarise, marked on what it says. Read across all 29 practice
questions and five served setups, about 50 of 145 answers failed, some two
thirds of them wrongly:

- the judge took a point each for a lead-in and a closing offer, so a complete,
  accurate summary scored 2 of 4, and called a bulleted summary "not a single
  summary". The rubric now scores content: a missing key fact costs 1 (2 at
  most, and the judge names it), anything invented or wrong 2, several versions
  1, and style nothing — with two worked examples;
- numbers_from_source rejected correct answers. It flags only a number the
  text can't account for: a bare hour is that hour on the clock, an option's
  number is a label, a note on the answer's own length is not a fact, number
  words count, and a number worked out from the text passes to the judge.

Every case below is from the practice half. Nothing runs a model or a judge:
the stand-in judge only."""

from __future__ import annotations

import json
import sys

import pytest

import everyday as ev
from conftest import make_service
from service import config
from test_everyday_12a5 import _asked

BANK = {q["id"]: q for q in ev.load_bank()}
SUMMARISE = [q for q in BANK.values() if q["group"] == "summarising"]
GATE = {"type": "numbers_from_source"}


def as_12a6(q: dict, template: str | None = None) -> dict:
    """the question as 12a.6 wrote it: its rubric generated from the same facts,
    request and reference with 12a.6's words (12a.8: or 12a.7's; 12a.9: or
    12a.8's, which asked for findings)"""
    old = json.loads(json.dumps(q))
    j = ev.judge_check(old)
    j["rubric"] = ev.summarise_rubric(q, template or ev.RUBRIC_12A6)
    j.pop("findings", None)
    j.pop("checklist", None)
    if template is ev.RUBRIC_12A8:
        j["findings"] = True
    return old


# ---------------------------------------------------------------------------
# 1. the rubric scores content — 12a.8: the judge reports findings on it, and
# the code scores them (test_12a8_findings); 12a.9: a checklist of its key
# facts (test_12a9_checklist); what counts is as 12a.7 set it
# ---------------------------------------------------------------------------

def test_every_summarise_rubric_scores_content_not_style():
    assert len(SUMMARISE) == 51                    # 12p.2: the practice 29, the tests' hidden 22
    for q in SUMMARISE:
        c = ev.judge_check(q)
        r = c["rubric"]
        assert c["checklist"] is True and (c["scale"], c["pass_at"]) == (4, 3), q["id"]
        assert r.startswith("Check the answer as a summary of the text in the question: its "
                            "content, not its style. Give no score"), q["id"]
        assert "1. The checklist. Check each of the key facts against the text and the " \
               "answer." in r
        # 12a.10: against the text in the question, never the reference
        assert "List each thing the answer says that the text doesn't give, or gets wrong: a " \
               "number, person, day, time, place or amount" in r
        assert 'Several versions: true if it gives several versions instead of one ' \
               '("Option 1 / Option 2"). A summary in bullets is one summary.' in r
        # 12a.10: an option's name and a tip are style too
        assert ("Style is never a finding: a lead-in (\"Here's a concise summary:\"), a heading "
                "or an option's name (\"Option 2: Bulleted\"), a closing offer (\"Let me know if "
                "you'd like it shorter\"), a tip on using it (\"Replace the name with your "
                "manager's\"), bullets, bold and emoji.") in r
        # the worked examples, as checklists
        assert "Example 1: every fact is correct, and the lead-in, the bullets and the offer " \
               "are style." in r
        assert "Here's a concise summary:\n    - Lunch moves to Friday" in r
        assert 'Example 2: the same answer with "- You\'re booking the table"' in r \
            and '{"fact": "£15 / cash", "status": "missing"}' in r
        # 12a.6's style deductions are gone
        assert "wrapped in a lead-in" not in r and "not one summary" not in r


def test_each_keeps_its_facts_length_and_reference():
    for q in SUMMARISE:
        now = ev.judge_check(q)["rubric"]
        for template in (ev.RUBRIC_12A6, ev.RUBRIC_12A7, ev.RUBRIC_12A8):
            was = as_12a6(q, template)
            # the round trip: 12a.6's, 12a.7's or 12a.8's rubric, read as the bank is read,
            # is today's
            assert ev.judge_check(ev._valid(was, "earlier"))["rubric"] == now, q["id"]
            assert ev._facts_in(now) == ev._facts_in(ev.judge_check(was)["rubric"]), q["id"]
        assert ev._facts_in(now)[0], q["id"]
        length = ev.stated_length(q["prompt"])
        if length and length != "shorter":
            assert f'The request asks for "{length}"' in now, q["id"]
        assert f'A good summary, for reference: "{q["reference"]}"' in now
    # a length only when the request states one, as before
    assert 'The request asks for it shorter' in ev.judge_check(BANK["everyday-pilot-03"])["rubric"]


def test_the_judge_is_asked_to_name_what_is_missing():
    q = BANK["everyday-summarising-09"]
    p = ev.judge_prompt(q, "Order 88291 is late.")
    # 12a.8: each missing fact quoted from the list, and no score; 12a.9: every fact, checked
    assert "one entry for every key fact in the rubric's list, in its order" in p
    assert '"status": <"correct", "wrong" or "missing">' in p
    assert "Give no score" in p and '"score"' not in p
    assert "Three worked examples, on another text" in p


def test_a_rubric_someone_wrote_stays_theirs_and_a_generated_one_is_replaced(tmp_path):
    for template in (ev.RUBRIC_12A6, ev.RUBRIC_12A7, ev.RUBRIC_12A8):
        old = as_12a6(BANK["everyday-summarising-04"], template)
        # one the question builder published with 12a.6's, 12a.7's or 12a.8's rubric: today's
        q = ev._valid(json.loads(json.dumps(old)), "built")
        assert ev.judge_check(q) == ev.judge_check(BANK["everyday-summarising-04"])
    # one someone edited: theirs, and scored as it was
    mine = as_12a6(BANK["everyday-summarising-04"])
    ev.judge_check(mine)["rubric"] += "\nAlso: the plumber's time must be there."
    kept = ev.judge_check(ev._valid(mine, "edited"))
    assert kept["rubric"].endswith("the plumber's time must be there.")
    assert "not its style" not in kept["rubric"] and not kept.get("findings")
    assert not kept.get("checklist")


# ---------------------------------------------------------------------------
# 2. numbers_from_source flags only what the text can't account for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qid,answer", [
    # a bare hour in the text is that hour on the clock
    ("everyday-summarising-04", "Tomorrow: kids to school by 7:30, meeting at head office at 9, "
                                "back ~2 PM. Buy milk, eggs, bread and fruit. Plumber at 5."),
    ("everyday-summarising-04", "Back around 14:00; plumber at 5 pm."),
    ("everyday-summarising-09", "Order 88291 was due at 7:15 and it's 8:00 now, still preparing."),
    # an option's or a list's number is a label
    ("everyday-summarising-09", "Option 1: Order 88291 is late.\nOption 2: You want a refund or a "
                                "new order to Oak Park."),
    ("everyday-summarising-09", "What you want: 1) a refund, or 2) a new order sent now."),
    # a note on the answer's own length is about the answer
    ("everyday-summarising-09", "Order 88291 is late; refund or resend to Oak Park. (Reduced "
                                "from ~48 to 33 words.)"),
    ("everyday-summarising-09", "Order 88291 late: refund or new order. ~30% shorter."),
    # number words count, and a number worked out from the text passes to the judge
    ("everyday-summarising-long-10", "Amara's gallery now loads in half the time."),
    ("everyday-summarising-long-10", "Amara's gallery loads 50% faster."),
    ("everyday-summarising-long-08", "Total £900; business cards would add £200, making "
                                     "£1,100."),
    ("everyday-summarising-09", "Due 7:15, still not here at 8:00: a 45-minute delay."),
])
def test_what_the_text_accounts_for_passes_the_gate(qid, answer):
    assert ev.run_check(GATE, answer, BANK[qid]["prompt"]) == (True, "")
    assert ev.half(BANK[qid]) == ev.PRACTICE


@pytest.mark.parametrize("qid,answer,why", [
    ("everyday-summarising-04", "Back at 3 PM; plumber at 5.", "invented the time 3 PM"),
    ("everyday-summarising-09", "Due 7:15, now 9:30.", "invented the time 9:30"),
    ("everyday-summarising-09", "Order 88291, flat 305.", "invented 305"),
    ("everyday-summarising-long-08", "Extra revision rounds cost £175.", "invented 175"),
])
def test_a_number_nothing_accounts_for_is_still_flagged(qid, answer, why):
    assert ev.run_check(GATE, answer, BANK[qid]["prompt"]) == (False, why)


def test_every_reference_still_passes_the_gate():
    for q in SUMMARISE:
        assert ev.run_check(GATE, q["reference"], q["prompt"])[0] is True, q["id"]


# ---------------------------------------------------------------------------
# 3. every stored answer re-marked, no model runs; before and after
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client
    client.__exit__(None, None, None)


def test_the_re_mark_judges_on_the_new_rubric_and_compares_with_12a6s_marks(
        svc, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    root = tmp_path / "full"
    d = root / "org__a"
    _asked(d, list(BANK.values()))
    ev.write(d, ev.mark(d))
    # 12a.6's, 12a.7's and 12a.8's snapshots are on the server already: this round keeps its own
    (root / "everyday_before_12a6.json").write_text(json.dumps({"models": {"org/a": {}}}))
    (root / "everyday_before_12a7.json").write_text(json.dumps({"models": {"org/a": {}}}))
    (root / "everyday_before_12a8.json").write_text(json.dumps({"models": {"org/a": {}}}))
    (root / "everyday_before_12a9.json").write_text(json.dumps({"models": {"org/a": {}}}))
    ev.remark(root, judge=True)
    now = ev.read(d)
    rubric = {q["id"]: ev.rubric_key(q) for q in SUMMARISE}
    judged = [it for it in now["items"] if it["group"] == "summarising" and it.get("score")]
    assert judged and all(it["rubric"] == rubric[it["id"]] for it in judged)
    assert (root / ev.BEFORE_NAME).exists() and ev.BEFORE_NAME == "everyday_before_12a11.json"
    monkeypatch.setattr(sys, "argv", ["everyday.py", str(root), "--compare"])
    assert ev.main() == 0
    table = capsys.readouterr().out.strip().splitlines()
    assert table[0].startswith("| model | before | after | Summarise before | Summarise after")
    assert table[2].startswith("| org/a | ")
