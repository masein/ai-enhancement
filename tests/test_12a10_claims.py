"""12a.10 (the brief's): Summarise — 12a.9's checklist failed good summaries.
Of the 18 practice fails for the original and the phone build, about 11 were
the judge listing as "invented or wrong" either style (a lead-in, an option's
heading, a closing offer, a tip) or true lines of the text the reference
leaves out, and the code kept them: they quoted the answer.

Now each claim gives the answer's words and the line of the text it
contradicts, or "not in the source", and the code decides: a claim that
states no number, name, date, amount or place is style, and dropped; "not in
the source" stands only if none of its facts is anywhere in the text; a line
it contradicts stands only if that line is the text's, word for word, and
shares a fact with the claim. Each one dropped is said, greyed, as 12a.8's
are. The judge's reply is held to a JSON schema (constrained decoding), so
every reply parses.

The five cases are the brief's, on the practice half (a hidden question's
answer never comes into the repo): each answer is written here around the
line the brief quotes. Nothing runs a model: the judge's replies are written
here too."""

from __future__ import annotations

import json

import pytest

import everyday as ev
from conftest import make_service
from service import config, llm
from test_everyday_12a5 import _asked

BANK = {q["id"]: q for q in ev.load_bank()}
SCHOOL, LEAVE = "everyday-summarising-01", "everyday-summarising-15"
SEND_OFF, STAND_UP, LOGO = ("everyday-summarising-long-44", "everyday-summarising-long-10",
                            "everyday-summarising-long-08")
NOT_IN = "not in the source"

# the school notice: three options, a lead-in and a closing offer
SCHOOL_3 = """Here are a few shortened options:

Option 1: School closes at 12:30 pm on Thu 16 Oct for parent-teacher meetings. Buses leave at 12:45 pm; no after-school clubs. Normal hours from Sun 19 Oct.

Option 2: Bulleted version
- Thu 16 Oct: school closes 12:30 pm (parent-teacher meetings)
- Buses leave 12:45 pm
- After-school clubs cancelled
- Back to normal Sun 19 Oct

Option 3: Early close 12:30 pm Thu 16 Oct, buses 12:45 pm, no clubs; normal from Sun 19 Oct.

Let me know if you'd like it even shorter!"""

# the leave request: two versions, and tips on sending it
LEAVE_TIPS = """Formal:
Dear Sir, I'd like to request annual leave from 3 to 10 November. My pending work is done and the Hamilton file is with Mariam. I'm reachable on my mobile for anything urgent. The leave form is attached for your approval.

Casual:
Hi, requesting leave 3–10 November. Work's done, Hamilton file handed to Mariam, reachable on mobile. Form attached.

Tips:
- Replace 'Sir' with your manager's actual name if you know it.
- Attach the leave form before you send it."""

# the send-off chat, as 3 bullets, with the coffee line
SEND_OFF_3 = """- Surprise send-off for Elena on Thursday 16 October at 6 pm in the back room of The Copper Kettle; keep it secret.
- Gift: $10 each to Ravi by Wednesday 15th for a big plant and a garden centre voucher; Nadia has the card for everyone to sign.
- Ravi will say a few words, and Nadia offered to buy coffee this week."""

# the stand-up, with Kofi's birthday
STAND_UP_TLDR = (
    "Release moves from Friday the 18th to Monday the 21st; code freeze is now Thursday at 5 pm. "
    "Tomasz is fixing the discount code bug, Amara is waiting on the new logo (the old logo is "
    "used for the demo if it's not in by Thursday), Kofi is running his test cases, and Lena "
    "reviews Hye-jin's help pages by 3 pm. Hye-jin has a day off on Friday. It's Kofi's birthday "
    "tomorrow.")

# the logo emails: business cards listed as decided while Hamid hasn't confirmed them
LOGO_DECIDED = """Decided: Concept B, changed to deep green. Ines sends three font options by 9 October as the first revision round. The total is £900: £450 is paid and the other £450 is due within 14 days of the final files.
Decided: matching business cards for £200, confirmed by 16 October.
Still open: which font to use."""
CARDS = "Decided: matching business cards for £200"
CARDS_LINE = "Business cards would be an extra £200, including print-ready files for both sides."


def facts(qid: str) -> list[str]:
    return [ev._show(f) for f in ev._facts_of(BANK[qid])[0]]


def reply(qid: str, claims=(), several=False, length="not asked", **status) -> str:
    """the judge's reply, in 12a.10's schema: every key fact correct but
    those given, and each claim (quote, source)"""
    rows = []
    for fact in facts(qid):
        st, quote = (status.get(fact) if isinstance(status.get(fact), tuple)
                     else (status.get(fact, "correct"), ""))
        rows.append({"fact": fact, "status": st, "quote": quote})
    return json.dumps({"checklist": rows,
                       "invented_or_wrong": [c if isinstance(c, str) else
                                             {"quote": c[0], "source": c[1]} for c in claims],
                       "several_versions": several, "length_ok": length, "note": ""})


def verdict(qid: str, answer: str, text: str) -> dict:
    q = BANK[qid]
    return ev.parse_verdict(text, ev.judge_check(q), q, answer)


def kinds(v: dict) -> list[str]:
    return [d["kind"] for d in v.get("dropped") or []]


# ---------------------------------------------------------------------------
# the brief's five practice cases
# ---------------------------------------------------------------------------

def test_the_school_notice_with_three_options_a_lead_in_and_an_offer_scores_3_and_passes():
    assert ev.half(BANK[SCHOOL]) == ev.PRACTICE
    assert ev.grade(BANK[SCHOOL], SCHOOL_3)[0] is None          # the options' numbers pass the gate
    style = [("Here are a few shortened options:", NOT_IN),
             ("Option 2: Bulleted version", NOT_IN),
             ("Let me know if you'd like it even shorter!", NOT_IN)]
    v = verdict(SCHOOL, SCHOOL_3, reply(SCHOOL, style, several=True, length="yes"))
    assert (v["pass"], v["score"], v["reason"]) == (True, 3, "3 of 4: several versions (−1)")
    assert v["findings"]["invented_or_wrong"] == []
    assert kinds(v) == ["style"] * 3
    # 12a.11: style by its shape — a heading, a lead-in, a closing offer
    assert v["dropped"][1]["text"] == (
        "the judge counted “Option 2: Bulleted version” as invented or wrong; it is a lead-in "
        "or heading: style is never a finding")
    assert v["dropped"][2]["text"].endswith("it is a closing offer: style is never a finding")
    # the same claims as 12a.9's judge wrote them, bare quotes: dropped the same
    old = verdict(SCHOOL, SCHOOL_3, reply(SCHOOL, [c for c, _ in style], several=True,
                                          length="yes"))
    assert (old["score"], kinds(old)) == (3, ["style"] * 3)


def test_the_leave_request_with_tips_scores_3():
    assert ev.half(BANK[LEAVE]) == ev.PRACTICE
    tips = [("Replace 'Sir' with your manager's actual name", NOT_IN),
            ("Attach the leave form before you send it.", NOT_IN)]
    v = verdict(LEAVE, LEAVE_TIPS, reply(LEAVE, tips, several=True, length="yes"))
    assert (v["pass"], v["score"], v["reason"]) == (True, 3, "3 of 4: several versions (−1)")
    # 12a.11: both are tips on using the answer — style, by their shape
    assert kinds(v) == ["style", "style"]
    assert v["dropped"][0]["text"] == ("the judge counted “Replace 'Sir' with your manager's "
                                       "actual name” as invented or wrong; it is a tip on using "
                                       "the answer: style is never a finding")


def test_the_coffee_line_is_dropped():
    assert ev.half(BANK[SEND_OFF]) == ev.PRACTICE
    line = "Nadia offered to buy coffee this week"
    v = verdict(SEND_OFF, SEND_OFF_3, reply(SEND_OFF, [(line, NOT_IN)], length="yes"))
    assert (v["pass"], v["score"]) == (True, 4)
    assert v["dropped"] == [{"kind": "given", "claim": line,
                             "text": f"the judge said “{line}” isn’t in the text; the text gives "
                                     "Nadia"}]
    # given a line of the text to contradict that shares nothing with it: dropped as well
    tom = "Tom: Same. Six years, wow, she was here before the coffee machine"
    assert tom in BANK[SEND_OFF]["prompt"]
    v = verdict(SEND_OFF, SEND_OFF_3, reply(SEND_OFF, [(line, tom)], length="yes"))
    assert (v["score"], kinds(v)) == (4, ["unsupported"])
    assert v["dropped"][0]["text"].endswith("they share no number, name, date, amount or place, "
                                            "and at most one word")


def test_kofis_birthday_is_dropped():
    assert ev.half(BANK[STAND_UP]) == ev.PRACTICE
    v = verdict(STAND_UP, STAND_UP_TLDR,
                reply(STAND_UP, [("Kofi's birthday tomorrow", NOT_IN)], length="yes"))
    assert (v["pass"], v["score"], v["reason"]) == (True, 4, "4 of 4: all key facts, one version")
    assert v["dropped"][0]["text"] == ("the judge said “Kofi's birthday tomorrow” isn’t in the "
                                       "text; the text gives Kofi")


def test_business_cards_decided_while_still_open_are_kept():
    assert ev.half(BANK[LOGO]) == ev.PRACTICE and CARDS_LINE in BANK[LOGO]["prompt"]
    v = verdict(LOGO, LOGO_DECIDED, reply(LOGO, [(CARDS, CARDS_LINE)]))
    assert (v["pass"], v["score"]) == (False, 2)
    assert v["reason"] == f"2 of 4: invented or wrong: “{CARDS}” (−2)"
    assert v["findings"]["invented_or_wrong"] == [{"quote": CARDS, "source": CARDS_LINE}]
    assert "dropped" not in v
    # 12a.9's own rule stands beside it: the fact called wrong, quoting the answer
    v = verdict(LOGO, LOGO_DECIDED, reply(LOGO, **{"business cards / business card":
                                                   ("wrong", CARDS)}))
    assert (v["score"], v["reason"]) == (2, f"2 of 4: wrong: business cards / business card "
                                            f"(“{CARDS}”) (−2)")


# ---------------------------------------------------------------------------
# the rules, one by one
# ---------------------------------------------------------------------------

SCHOOL_ONE = ("School closes at 12:30 pm on Thu 16 Oct for parent-teacher meetings. Buses leave at "
              "12:45 pm. Normal timings resume on Friday 17 October.")
WRONG_DAY = "Normal timings resume on Friday 17 October"
ITS_LINE = "Normal timings resume on Sunday 19 October."


def test_a_wrong_claim_stands_on_its_line_of_the_text_word_for_word_sharing_a_fact():
    v = verdict(SCHOOL, SCHOOL_ONE, reply(SCHOOL, [(WRONG_DAY, ITS_LINE)], length="yes"))
    assert (v["score"], v["findings"]["invented_or_wrong"]) == (
        2, [{"quote": WRONG_DAY, "source": ITS_LINE}])
    # a line the text doesn't have
    v = verdict(SCHOOL, SCHOOL_ONE, reply(SCHOOL, [(WRONG_DAY, "Normal timings resume on Monday "
                                                               "20 October.")], length="yes"))
    assert (v["score"], kinds(v)) == (4, ["unsupported"])
    assert "the text doesn’t say that" in v["dropped"][0]["text"]
    # the text's line, sharing no fact with the claim
    v = verdict(SCHOOL, SCHOOL_ONE, reply(SCHOOL, [(WRONG_DAY, "After-school clubs are cancelled "
                                                               "that day.")], length="yes"))
    assert (v["score"], kinds(v)) == (4, ["unsupported"])


def test_an_invented_claim_stands_only_if_the_text_has_none_of_its_facts():
    campus = SCHOOL_ONE.replace("meetings.", "meetings at the Riverside campus.")
    claim = "at the Riverside campus"
    v = verdict(SCHOOL, campus, reply(SCHOOL, [(claim, NOT_IN)], length="yes"))
    assert (v["score"], v["reason"]) == (2, f"2 of 4: invented or wrong: “{claim}” (−2)")
    assert v["findings"]["invented_or_wrong"] == [{"quote": claim, "source": None}]
    # a wrong day called "not in the source": October is, so it needed its line
    v = verdict(SCHOOL, SCHOOL_ONE, reply(SCHOOL, [(WRONG_DAY, NOT_IN)], length="yes"))
    assert (v["score"], kinds(v)) == (4, ["given"])
    assert v["dropped"][0]["text"].endswith("the text gives October")


def test_a_claim_must_still_quote_the_answer():
    v = verdict(SCHOOL, SCHOOL_ONE, reply(SCHOOL, [("closes on Riverside Road", NOT_IN)],
                                          length="yes"))
    assert (v["score"], kinds(v)) == (4, ["invented"])


@pytest.mark.parametrize("qid,claim,has", [
    (SCHOOL, "Here are a few shortened options:", set()),
    (SCHOOL, "Option 2: Bulleted version", set()),
    (SCHOOL, "Let me know if you'd like it even shorter!", set()),
    (LEAVE, "Attach the leave form before you send it.", set()),
    (LEAVE, "Replace 'Sir' with your manager's actual name", {"sir"}),
    (STAND_UP, "Kofi's birthday tomorrow", {"kofi"}),
    ("everyday-summarising-long-07", "Grace keeps her fee at £45", {"grace", "45"}),
    (SEND_OFF, "Nadia offered to buy coffee this week", {"nadia"}),
    (STAND_UP, "a 45-minute delay", {"45"}),
    (LOGO, CARDS, {"200"}),
    (SCHOOL, "Normal timings resume on Friday 17 October", {"friday", "17", "oct"}),
])
def test_the_facts_a_claim_states(qid, claim, has):
    """a number, an amount, a date, a day, or a name — a word written with a
    capital where a sentence doesn't start, or one of the text's names"""
    assert ev.claim_facts(claim, ev.names(BANK[qid]["prompt"])) == has


def test_a_lead_in_with_a_number_is_still_style():
    q = BANK[SCHOOL]
    dropped: list[dict] = []
    kept = ev.claims_kept([{"quote": "Here's the 12:30 version:", "source": NOT_IN}], q,
                          "Here's the 12:30 version:\nSchool closes at 12:30 on 16 Oct.", dropped)
    assert kept == [] and kinds({"dropped": dropped}) == ["style"]


# ---------------------------------------------------------------------------
# every reply parses: the judge is held to the checklist's schema
# ---------------------------------------------------------------------------

def test_the_schema_is_the_checklist_with_its_facts_and_statuses():
    s = ev.judge_schema(BANK[SCHOOL])
    assert s["additionalProperties"] is False and set(s["required"]) == {
        "checklist", "invented_or_wrong", "several_versions", "length_ok", "note"}
    row = s["properties"]["checklist"]["items"]
    assert row["properties"]["fact"]["enum"] == facts(SCHOOL) == ["16 oct / oct 16",
                                                                  "12:30 / 12.30"]
    assert row["properties"]["status"]["enum"] == ["correct", "wrong", "missing"]
    assert set(row["required"]) == {"fact", "status", "quote"} and not row["additionalProperties"]
    claim = s["properties"]["invented_or_wrong"]["items"]
    assert set(claim["required"]) == set(claim["properties"]) == {"quote", "source"}
    assert s["properties"]["length_ok"]["enum"] == ["yes", "no", "not asked"]
    # a reply in its shape is read, an empty quote beside a correct fact and all
    assert verdict(SCHOOL, SCHOOL_ONE, reply(SCHOOL, length="yes"))["score"] == 4
    # only the checklist has one
    assert ev.judge_schema({"id": "x", "checks": [{"type": "judge", "rubric": "r"}]}) is None
    assert ev.judge_schema({"id": "x", "checks": [{"type": "contains", "values": ["a"]}]}) is None


def test_the_schema_goes_on_the_wire_to_every_backend(tmp_path, monkeypatch):
    import vllm_stub
    schema = ev.judge_schema(BANK[SCHOOL])
    want = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True,
                                                   "schema": schema}}
    r = llm.Request(custom_id="c", system="", user="u", json=True, schema=schema)
    # OpenAI's batch body
    assert llm.OpenAIBatches.body("gpt-x", r)["response_format"] == want
    # a local server (vLLM, llama-server): guided decoding
    stub, srv = vllm_stub.serve()
    monkeypatch.setattr(config, "LOCAL_BASE_URL", stub.url)
    monkeypatch.setattr(llm.LocalOpenAI, "BACKOFF", (0, 0, 0))
    try:
        b = llm.LocalOpenAI("chat", "", tmp_path)
        b._complete({"custom_id": "c", "system": "", "user": "u", "max_tokens": 10,
                     "json": True, "schema": schema})
        b._complete({"custom_id": "d", "system": "", "user": "v", "max_tokens": 10,
                     "json": True})
    finally:
        srv.shutdown()
        srv.server_close()
    assert stub.bodies[-2]["response_format"] == want
    assert stub.bodies[-1]["response_format"] == {"type": "json_object"}   # none: JSON mode


def test_the_judge_test_asks_for_the_schema_on_a_checklist_only():
    from service import judge_test
    assert judge_test.schema_for({"kind": "everyday", "task": SCHOOL}) == \
        ev.judge_schema(BANK[SCHOOL])
    assert judge_test.schema_for({"kind": "everyday", "task": "no-such-question"}) is None
    assert judge_test.schema_for({"kind": "exam", "task": "exam_law"}) is None


def test_openrouter_is_sent_the_schema(tmp_path, monkeypatch):
    from fake_openrouter import FakeOpenRouter
    from service import ai_models
    client, _, _ = make_service(tmp_path, monkeypatch, judge_model="stub")
    try:
        fake = FakeOpenRouter.install(monkeypatch)
        pin = ai_models.pin("z-ai/glm-5.3")
        b = llm.OpenRouterChat("z-ai/glm-5.3", config.OPENROUTER_API_KEY, config.BENCH_ROOT,
                               pin=pin)
        schema = ev.judge_schema(BANK[SCHOOL])
        b._complete({"custom_id": "x", "system": "", "user": "hi", "max_tokens": 10,
                     "json": True, "schema": schema})
        assert fake.chat[-1]["response_format"]["json_schema"]["schema"] == schema
    finally:
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# the re-mark: 12a.9's rubrics and 12a.8's, whichever length reading made them
# ---------------------------------------------------------------------------

def test_a_12a8_rubric_made_with_either_length_reading_takes_todays():
    """12a.9 read "as 4 bullets" as a length, which changed what 12a.8's
    rubric generates for such a question; five hidden questions' stored 12a.8
    rubrics stopped matching, weren't upgraded, and kept their 12a.8 verdicts
    through 12a.9's re-mark"""
    q = BANK["everyday-summarising-long-14"]
    assert ev.stated_length(q["prompt"]) == "4 bullets"
    assert ev.stated_length(q["prompt"], ev.STATED_LENGTH_12A8) == ""
    for rubric, flag in ((ev.summarise_rubric(q, ev.RUBRIC_12A8, ev.STATED_LENGTH_12A8),
                          "findings"),
                         (ev.summarise_rubric(q, ev.RUBRIC_12A8), "findings"),
                         (ev.summarise_rubric(q, ev.RUBRIC_12A9), "checklist")):
        old = {**q, "checks": [{"type": "numbers_from_source"},
                               {"type": "judge", "scale": 4, "pass_at": 3, flag: True,
                                "at_least": 5, "rubric": rubric}]}
        now = ev._valid(json.loads(json.dumps(old)), "x")
        assert ev.judge_check(now)["rubric"] == ev.summarise_rubric(q)
        assert ev.judge_check(now)["checklist"] is True
    assert 'The request asks for "4 bullets"' in ev.summarise_rubric(q)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    yield client
    client.__exit__(None, None, None)


ANSWERS = {SCHOOL: SCHOOL_3, LOGO: LOGO_DECIDED}


def test_the_remark_sends_the_schema_and_compare_says_what_was_dropped(svc):
    qs = [q for q in BANK.values() if q["group"] == "summarising"]
    mdir = config.OUT_DIR / "org__m"
    answer = lambda q: ANSWERS.get(q["id"], q["reference"])  # noqa: E731
    _asked(mdir, qs, answer=answer)
    ev.write(mdir, {"model": "org/m", "items": [
        {"id": q["id"], "group": "summarising", "pass": True, "score": 4,
         "reason": "4 of 4: all key facts, one version",
         "rubric": ev.rubric_key({"checks": [{"type": "judge", "rubric": ev.summarise_rubric(
             q, ev.RUBRIC_12A9)}]}), "answer_text": answer(q)} for q in qs]})
    res = ev.remark(config.OUT_DIR, judge=True)
    assert res["sent"] == len(qs) and ev.BEFORE_NAME == "everyday_before_12a11.json"
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith(ev.REMARK + ":")]
    by_id = {r["custom_id"].rsplit(":", 1)[1]: r for r in sent}
    assert set(by_id) == {q["id"] for q in qs}
    assert all(r["schema"] == ev.judge_schema(BANK[i]) and r["schema"] for i, r in by_id.items())
    replies = {r["custom_id"]: llm.Result(text=ev.stub_reply(r["user"])) for r in sent}
    replies[by_id[SCHOOL]["custom_id"]] = llm.Result(text=reply(
        SCHOOL, [("Here are a few shortened options:", NOT_IN),
                 ("Option 2: Bulleted version", NOT_IN)], several=True, length="yes"))
    replies[by_id[LOGO]["custom_id"]] = llm.Result(text=reply(LOGO, [(CARDS, CARDS_LINE)]))
    ev.finish_remark(config.OUT_DIR, replies)
    items = {it["id"]: it for it in ev.read(mdir)["items"]}
    assert (items[SCHOOL]["pass"], items[SCHOOL]["score"]) == (True, 3)
    assert (items[LOGO]["pass"], items[LOGO]["score"]) == (False, 2)
    out = ev.compare(config.OUT_DIR)
    assert ("| model | half | before | now | claims kept | dropped: style | dropped: the text "
            "gives it | dropped: no line of the text behind it | dropped: not the answer's words "
            "| waiting |") in out
    practice = [q for q in qs if ev.half(q) == ev.PRACTICE]
    n = len(practice)
    assert f"| org/m | practice | {n} of {n} | {n - 1} (−1) | 1 | 2 | 0 | 0 | 0 | 0 |" in out
    hidden = [q for q in qs if ev.half(q) == ev.HIDDEN]
    assert f"| org/m | hidden | {len(hidden)} of {len(hidden)} |" in out
    # a hidden question is counted, never named
    assert hidden and not any(q["id"] in out for q in hidden)
    assert "Lengths “as 4 bullets” asks reach" not in out           # 12a.9's table is gone


def test_waiting_is_an_answer_with_no_verdict_and_nothing_else(svc):
    """12a.9's table counted as waiting every answer without a checklist — a
    script check's fail, no answer, and the five 12a.8 verdicts kept. Now it
    is what the first table counts: no verdict yet (a reply unreadable twice)"""
    qs = [q for q in BANK.values() if q["group"] == "summarising" and ev.half(q) == ev.PRACTICE]
    mdir = config.OUT_DIR / "org__m"
    wrong_number = {SCHOOL: "School closes at 12:30 pm on Thu 16 Oct; 97 buses leave at 12:45 pm."}
    answer = lambda q: wrong_number.get(q["id"], q["reference"])  # noqa: E731
    _asked(mdir, qs, answer=answer)
    ev.write(mdir, {"model": "org/m", "items": [
        {"id": q["id"], "group": "summarising", "pass": True, "score": 4, "reason": "",
         "rubric": "", "answer_text": answer(q)} for q in qs]})
    ev.remark(config.OUT_DIR, judge=True)
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith(ev.REMARK + ":")]
    assert not any(r["custom_id"].endswith(":" + SCHOOL) for r in sent)   # the gate failed it
    replies = {r["custom_id"]: llm.Result(text=ev.stub_reply(r["user"])) for r in sent}
    cid = next(c for c in replies if c.endswith(":" + LOGO))
    replies[cid] = llm.Result(text="the answer is fine")                   # can't be read
    ev.finish_remark(config.OUT_DIR, replies)
    # asked once more, held to the schema again
    again = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
             if ":retry:" in r["custom_id"]]
    assert [r["custom_id"].rsplit(":", 1)[1] for r in again] == [LOGO]
    assert again[0]["schema"] == ev.judge_schema(BANK[LOGO])
    ev.finish_remark(config.OUT_DIR, {again[0]["custom_id"]: llm.Result(text="still fine")})
    items = {it["id"]: it for it in ev.read(mdir)["items"]}
    assert items[LOGO]["pass"] is None and items[LOGO]["reason"] == ev.UNREADABLE
    assert items[SCHOOL]["pass"] is False                                   # the gate's, not waiting
    out = ev.compare(config.OUT_DIR)
    first = next(x for x in out.splitlines() if x.startswith("| org/m |"))
    assert first.endswith("| 1 |")                                        # the first table: 1
    row = next(x for x in out.splitlines() if x.startswith("| org/m | practice |"))
    assert row.endswith("| 1 |")                                          # and this one: the same
