"""12a.11 (the brief's, rule B): 12a.10 dropped as style every claim with no
number, name, date, amount or place — and 14 of 20 practice-half claims it
dropped for gemma-3-270m-it and SmolLM2-135M-Instruct were wrong or invented
statements, plain prose: "The landlord is also responsible for replacing light
bulbs and smoke alarms", when the text says the tenants now are.

Now style is a shape — a lead-in or heading, a closing offer, a tip on using
the answer, a line about the text itself. A claim tied to a line of the text
stands when it shares a number or name with it, or two words it is about; a
claim that is the text's own words isn't invented or wrong; a "not in the
source" claim with no fact is dropped when a sentence of the text holds most
of its words. `--rescore` scores the stored checklists again by today's code,
no judge asked. The twenty claims below are the server's, with the line the
judge gave; each answer is written around its claim. Nothing runs a model."""

from __future__ import annotations

import json

import pytest

import everyday as ev
from conftest import make_service
from service import config, llm
from test_everyday_12a5 import _asked

BANK = {q["id"]: q for q in ev.load_bank()}
NS = "not in the source"
Q = "everyday-summarising-"

# (question, the claim, the judge's source, what the code does with it)
REAL = [
    # gemma-3-270m-it
    ("04", "The voice note is a brief, informal message from a person to their family.", NS,
     "style"),
    # the text says it: the judge called a true line wrong, and it counts — B's one false fail
    ("04", "They also mention needing groceries, milk, eggs, bread, and fruit.",
     "We need milk, eggs, bread and some fruit.", "kept"),
    ("long-01", "The client agreed to the mobile checkout redesign.",
     "The client agreed. Mobile checkout moves to phase 2 in November, and we launch on 6 October "
     "with product pages and search filters.", "kept"),
    ("long-01", "The client is flexible and willing to work with the team.", NS, "kept"),
    ("long-10", "They are meeting to discuss the client review.", NS, "kept"),
    ("long-14", "Fire safety is a concern.", NS, "kept"),
    ("long-28", "Read the entire text carefully.", NS, "style"),
    ("long-28", "Identify the key changes:", NS, "style"),
    # wrong, but it shares one word with its line: B can't tell it from a stray word
    ("long-29", "pet-free living",
     "pets may only be kept with the landlord's written permission, which must be requested in "
     "advance; this applies to any new pet, not to anything already agreed.", "unsupported"),
    ("long-29", "The landlord is also responsible for replacing light bulbs and smoke alarms.",
     "tenants will now be responsible for replacing light bulbs and smoke alarm batteries inside "
     "the flat, although the landlord will still handle any electrical faults.", "kept"),
    # SmolLM2-135M-Instruct
    ("09", "I don't have the capability to assist with delivery or delivery apps.", NS, "kept"),
    ("long-01", "I've also confirmed that the team will be working on the staging server for "
     "the new site.", "Also, small thing, the staging server was slow again on Friday. I'll "
     "restart it tonight, no action needed from anyone.", "kept"),
    # wrong, but the line the judge gave shares nothing with it
    ("long-01", "I've also confirmed that the client will be working on the design review for "
     "the product pages and search filters.",
     "Aisha Rahman ... Can we move it to Wednesday 1 October at 2pm?", "unsupported"),
    # the text says it ("I'd keep it confidential, members just email me directly.")
    ("long-07", "I'll keep it confidential", NS, "given"),
    ("long-08", "Thanks for the feedback! I'll make sure to include the sourdough and business "
     "cards. I'll also include the sourdough and business cards for the first half of the "
     "invoice.", NS, "kept"),
    ("long-08", "I'll also include the sourdough and business cards for the second half of the "
     "invoice.", NS, "kept"),
    ("long-14", "the clubhouse staff were in favour of the new fees.", NS, "kept"),
    ("long-18", "We're keeping the rotating host system, and the host picks the snacks.",
     "We're keeping the rotating host system, and the host picks the snacks.", "given"),
    ("long-22", "I’ve been thinking about the new systems and dates, and I’m really excited "
     "about them.", NS, "kept"),
    ("long-22", "I’m really looking forward to the new floor and the new team dinner.", NS,
     "kept"),
]


def outcome(qid: str, claim: str, source: str) -> tuple[str, list[dict]]:
    dropped: list[dict] = []
    kept = ev.claims_kept([{"quote": claim, "source": source}], BANK[Q + qid],
                          "Here it is. " + claim, dropped)
    return ("kept" if kept else dropped[0]["kind"]), dropped


@pytest.mark.parametrize("qid,claim,source,want", REAL,
                         ids=[f"{r[0]}-{i}" for i, r in enumerate(REAL)])
def test_the_servers_twenty_claims(qid, claim, source, want):
    assert ev.half(BANK[Q + qid]) == ev.PRACTICE
    if source != NS:
        assert ev._quoted(source, BANK[Q + qid]["prompt"])          # the judge's line is the text's
    got, dropped = outcome(qid, claim, source)
    assert got == want, dropped


def test_what_they_add_up_to():
    """14 of the 20 are wrong or invented, and 12a.10 dropped them all as style;
    now 12 of the 14 count, style is the three that are, and two of the three
    true lines are dropped — the third counts (the judge called it wrong)"""
    got = [outcome(q, c, s)[0] for q, c, s, _ in REAL]
    assert (got.count("style"), got.count("kept"), got.count("given"),
            got.count("unsupported")) == (3, 13, 2, 2)


@pytest.mark.parametrize("claim,shape", [
    ("Here are a few shortened options:", "a lead-in or heading"),
    ("Option 2: Bulleted version", "a lead-in or heading"),
    ("Identify the key changes:", "a lead-in or heading"),
    ("Let me know if you'd like it even shorter!", "a closing offer"),
    ("Replace 'Sir' with your manager's actual name", "a tip on using the answer"),
    ("Attach the leave form before you send it.", "a tip on using the answer"),
    ("Read the entire text carefully.", "a tip on using the answer"),
    ("The voice note is a brief, informal message from a person to their family.",
     "a line about the text itself"),
    # what a summary says is never style
    ("Bring £15 cash", ""), ("Keep it secret from Elena", ""),
    ("The landlord is also responsible for replacing light bulbs and smoke alarms.", ""),
    ("The notice says the school closes at 12:30", ""),
])
def test_style_is_a_shape(claim, shape):
    assert ev.style_shape(claim, ev.claim_facts(claim)) == shape


def test_a_short_phrase_of_the_text_is_still_judged_on_its_line():
    """"the text's own words" is a claim of six words or more: "Friday the 18th"
    is in the stand-up too, and wrong as the release date"""
    q = BANK[Q + "long-10"]
    assert ev._quoted("Friday the 18th", q["prompt"])
    dropped: list[dict] = []
    ev.claims_kept([{"quote": "Friday the 18th", "source": NS}], q,
                   "The release is on Friday the 18th.", dropped)
    assert dropped[0]["kind"] == "given" and "the text gives" in dropped[0]["text"]
    assert "says that itself" not in dropped[0]["text"]


# ---------------------------------------------------------------------------
# --rescore: the stored checklists, scored again — no judge asked
# ---------------------------------------------------------------------------

LEASE = Q + "long-29"
WRONG = "The landlord is also responsible for replacing light bulbs and smoke alarms."
LINE = ("tenants will now be responsible for replacing light bulbs and smoke alarm batteries "
        "inside the flat, although the landlord will still handle any electrical faults.")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    yield client
    client.__exit__(None, None, None)


def _marked_by_12a10(name: str) -> None:
    """a model's Summarise answers as 12a.10 left them: the lease's wrong line
    in its answer and in the judge's checklist, and dropped as style"""
    qs = [q for q in BANK.values() if q["group"] == "summarising" and ev.half(q) == ev.PRACTICE]
    answer = lambda q: q["reference"] + (" " + WRONG if q["id"] == LEASE else "")  # noqa: E731
    d = config.OUT_DIR / name
    _asked(d, qs, answer=answer)
    items = []
    for q in qs:
        c = ev.judge_check(q)
        raw = ev.stub_checklist(c["rubric"], answer(q), q["prompt"])
        if q["id"] == LEASE:
            raw["invented_or_wrong"] = [{"quote": WRONG, "source": LINE}]
        items.append({"id": q["id"], "group": "summarising", "pass": True, "score": 4,
                      "reason": "4 of 4: all key facts, one version", "rubric": ev.rubric_key(q),
                      "answer_text": answer(q), "judge_raw": raw})
    # one on another rubric: never scored again
    items[1]["rubric"] = "0" * 12
    ev.write(d, {"model": name.replace("__", "/"), "items": items})


def test_rescore_counts_the_wrong_line_with_no_judge_asked(svc):
    _marked_by_12a10("org__small")
    got = ev.rescore(config.OUT_DIR)
    assert got == {"org/small": 28}                                 # 29 practice, one on another rubric
    item = {it["id"]: it for it in ev.read(config.OUT_DIR / "org__small")["items"]}[LEASE]
    assert (item["pass"], item["score"]) == (False, 2)
    assert item["reason"].endswith(f"invented or wrong: “{WRONG}” (−2)")
    assert item["findings"]["invented_or_wrong"] == [{"quote": WRONG, "source": LINE}]
    # no judge was asked
    assert not [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
                if r["custom_id"].startswith(ev.REMARK)]
    # the marks before it are kept, for --compare
    was = json.loads((config.OUT_DIR / ev.BEFORE_NAME).read_text())
    assert was["summarise"]["org/small"][LEASE] is True
    # again: the same, and the before kept as it was
    assert ev.rescore(config.OUT_DIR) == {"org/small": 28}
    assert json.loads((config.OUT_DIR / ev.BEFORE_NAME).read_text()) == was


def test_the_before_file_takes_each_model_the_first_time(svc):
    """a model re-marked alone first leaves the others to be added later, never
    written over"""
    _marked_by_12a10("org__one")
    ev.rescore(config.OUT_DIR, {"org__one"})
    first = json.loads((config.OUT_DIR / ev.BEFORE_NAME).read_text())
    assert set(first["models"]) == {"org/one"}
    _marked_by_12a10("org__two")
    ev.rescore(config.OUT_DIR)
    both = json.loads((config.OUT_DIR / ev.BEFORE_NAME).read_text())
    assert set(both["models"]) == {"org/one", "org/two"}
    assert both["summarise"]["org/one"] == first["summarise"]["org/one"]


def test_the_command_line(svc, capsys, monkeypatch):
    _marked_by_12a10("org__small")
    monkeypatch.setattr("sys.argv", ["everyday.py", str(config.OUT_DIR), "--rescore"])
    assert ev.main() == 0
    out = capsys.readouterr().out
    assert "org/small: 28 Summarise answer(s) scored again from the judge's checklist" in out
    assert "no judge asked" in out
