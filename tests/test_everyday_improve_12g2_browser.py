"""12g.2 on the page: the Everyday bank split — the practice half shown, the
hidden half counted and scoring — and Everyday groups in Improve: in Weak
spots beside the exam topics, labelled; greyed with how many more hidden
questions they need when under the line; proposed from, as chat examples;
retested before → after on the hidden half, or asked to retest across two
versions of the questions."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

import everyday as ev
from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12g2"
MODEL = "fx/skewed-360m"        # every other everyday answer "not sure": failures everywhere
GOOD = "fx/good-750m"
CK = "fx/good-750m-tuned-skill"


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def served(page, edit):
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def pipeline(page, base, model=MODEL, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=improve&sub=model&model=" + quote(model, safe=""))
    page.wait_for_selector(f"[data-pipeline='{model}']")
    page.wait_for_function("() => state.rv.loaded")
    page.wait_for_selector("[data-stages]")
    more = page.locator("[data-stage-more='weak']")
    if more.count():
        more.click()


def hidden_ids():
    return {q["id"] for q in ev.load_bank() if ev.half(q) == ev.HIDDEN}


# ---------------------------------------------------------------------------
# 5. the split on the page
# ---------------------------------------------------------------------------

def test_the_everyday_page_shows_the_practice_half_and_counts_the_hidden(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-table]")
    assert page.locator("[data-evd-bank-split]").get_attribute("data-evd-bank-split") == "200|188"
    shown = {e.get_attribute("data-evd-bank-q") for e in page.locator("[data-evd-bank-q]").all()}
    assert len(shown) == 188 and not shown & hidden_ids()
    assert page.locator("[data-evd-group-split='honesty']").inner_text() == \
        "24 hidden · 27 practice"                                       # 12a.5: was 19 · 22
    # no hidden question's text anywhere in what the page holds
    held = page.evaluate("JSON.stringify(DATA.everyday)")
    for q in ev.load_bank():
        if q["id"] in hidden_ids() and len(q["prompt"]) > 30:
            assert q["prompt"] not in held, q["id"]
    # a model's score is its hidden half's
    tot = page.evaluate(f"DATA.everyday.models[{json.dumps(GOOD)}].total")
    assert tot == 200
    assert page.locator(f"[data-evd-count='{GOOD}']").inner_text().endswith(" of 200")
    shot(page, "12g2-everyday-1400-light.png", full_page=True)
    assert page.errors == []


def test_a_groups_answers_are_its_practice_half_and_say_so(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=benchmarks&sub=everyday")
    page.locator(f"[data-evd-cell='{GOOD}|honesty']").click()
    note = page.locator("[data-evd-split-note='honesty']")
    assert note.inner_text().startswith("Practice questions · ")
    assert note.inner_text().endswith("The 24 hidden ones score it and are not shown.")
    rows = {r.get_attribute("data-evd-row") for r in page.locator("[data-evd-row]").all()}
    assert rows and not rows & hidden_ids()
    assert page.errors == []


def test_results_from_before_the_split_are_in_history_labelled(live, page):
    def before(body):
        e = body["everyday"]
        e["models"].pop(GOOD, None)
        e["earlier"][GOOD] = {"passed": 300, "total": 333, "marked_at": 1790000000,
                              "hash": "20efe555", "label": "all questions, before the split"}
    served(page, before)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + quote(GOOD, safe=""))
    page.wait_for_selector("[data-model-hero]")
    assert page.locator("[data-evd-earlier-note]").inner_text() == \
        "scored on all questions, before the split · in History"
    page.locator("[data-mtab='history']").click()
    card = page.locator(f"[data-evd-earlier='{GOOD}']")
    assert card.locator("[data-earlier-badge]").inner_text() == "all questions, before the split"
    assert "300 of 333" in card.inner_text() and "hidden half" in card.inner_text()
    assert page.errors == []


# ---------------------------------------------------------------------------
# 6. Everyday groups in Weak spots
# ---------------------------------------------------------------------------

def test_weak_spots_mix_exam_topics_and_everyday_groups_each_labelled(live, page):
    # 12a.5: every group has 20 hidden questions now; one is served under the
    # line, as Honesty was with 19
    served(page, lambda b: b["everyday"]["hidden"].update(honesty=19))
    pipeline(page, live["base"])
    kinds = [(e.get_attribute("data-weak-kind"), e.locator(".imp-kind").inner_text().lower())
             for e in page.locator("[data-stage='weak'] [data-weak]").all()]
    assert {"exam", "everyday"} <= {k for k, _ in kinds}
    assert all(k == t for k, t in kinds)                 # the label says which
    # a group under the line: once, greyed, how many more — and no Propose
    honesty = page.locator("[data-weak='everyday:honesty']")
    assert honesty.get_attribute("data-weak-need") == "1"
    assert "greyed" in honesty.get_attribute("class")
    assert honesty.inner_text().endswith("Honesty · needs 1 more hidden question to improve on")
    assert page.locator("[data-weak-propose='everyday:honesty']").count() == 0
    # a group over it offers Propose
    assert page.locator("[data-weak-propose='everyday:instructions']").count() == 1
    shot(page, "12g2-weak-spots-1400-light.png", full_page=True)
    assert page.errors == []


def test_with_twenty_hidden_questions_a_group_offers_propose(live, page):
    served(page, lambda b: b["everyday"]["hidden"].update(honesty=20))
    pipeline(page, live["base"])
    honesty = page.locator("[data-weak='everyday:honesty']")
    assert honesty.get_attribute("data-weak-need") is None
    assert page.locator("[data-weak-propose='everyday:honesty']").count() == 1
    assert page.errors == []


def test_with_the_bank_as_it_is_every_group_offers_propose(live, page):
    """12a.5: Honesty's ten new questions take it to 24 hidden"""
    pipeline(page, live["base"])
    assert page.locator("[data-stage='weak'] [data-weak-need]").count() == 0
    assert page.locator("[data-weak-propose='everyday:honesty']").count() == 1
    assert page.errors == []


def test_propose_on_a_group_makes_its_proposal_and_its_card_reads_practice(live, page):
    pipeline(page, live["base"])
    set_name(page, "masein")
    page.locator("[data-weak-propose='everyday:instructions']").click()
    page.wait_for_selector("[data-toast='propose']")
    item = page.locator("[data-stage='proposals'] [data-prop]").first
    item.wait_for()
    assert item.locator(".imp-kind").inner_text().lower() == "everyday"
    assert "Instructions" in item.inner_text()
    pid = item.get_attribute("data-prop")
    # the group is out of Weak spots while its proposal is open
    assert page.locator("[data-weak='everyday:instructions']").count() == 0
    # the fake AI answers within a poll or two: then it waits for a person
    page.wait_for_function(f"() => {{ const s = document.querySelector(\"[data-rv-status='{pid}']\");"
                           " return s && s.textContent === 'waiting for you'; }", timeout=30000)
    item.locator("[data-prop-act]").click()
    page.wait_for_selector("#reader[data-ready='1']")
    why = page.locator(f"[data-why-line='{pid}']").inner_text()
    assert "practice requests failed · Instructions" in why and why.endswith(
        "of 24 (hidden questions)")
    block = page.locator(f"[data-evd-read-block='{pid}']")
    block.locator("summary").click()
    read = {e.get_attribute("data-evd-read-q") for e in block.locator("[data-evd-read-q]").all()}
    assert read and not read & hidden_ids()
    shot(page, "12g2-everyday-card-1400-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 8. Retests on the hidden half, and across two versions of the questions
# ---------------------------------------------------------------------------

def _retest(case):
    def edit(body):
        e = body["everyday"]
        for m in body["models"]:
            if m["id"] == CK:
                m["trainedFrom"] = {"base": MODEL, "source": "person", "by": "masein"}
                m["tainted"] = ["everyday:instructions"]
        base = e["models"][MODEL]
        ck = json.loads(json.dumps(base))
        ck["groups"]["instructions"] = {"passed": 20, "total": 24}
        if case == "same":
            e["models"][CK] = ck
        else:                                   # answered before the current version
            e["earlier"][CK] = {"passed": 200, "total": 333, "marked_at": 1.0, "hash": "x",
                                "label": "an earlier wording"}
    return edit


def test_a_retest_shows_the_group_before_after_on_the_hidden_half(live, page):
    served(page, _retest("same"))
    pipeline(page, live["base"])
    line = page.locator(f"[data-retest='{CK}'] [data-retest-group='instructions']")
    a = page.evaluate(f"DATA.everyday.models[{json.dumps(MODEL)}].groups.instructions")
    assert line.inner_text() == f"Instructions {a['passed']} of 24 → 20 of 24"
    # and the Standard watch under it, as 12g.1 left it
    assert page.locator(f"[data-retest='{CK}'] [data-watch-line='{CK}']").count() == 1
    shot(page, "12g2-retest-1400-light.png")
    assert page.errors == []


def test_a_retest_across_two_versions_asks_to_retest(live, page):
    served(page, _retest("earlier"))
    pipeline(page, live["base"])
    line = page.locator(f"[data-retest='{CK}'] [data-retest-group='instructions']")
    assert line.inner_text() == "Instructions: retest on the current questions · Test"
    assert line.get_attribute("data-retest-stale") == CK
    assert page.errors == []


# ---------------------------------------------------------------------------
# 7. a chat dataset reads as request, reply and checks; the model page's Improve
# ---------------------------------------------------------------------------

def test_a_chat_dataset_reads_as_request_reply_and_checks(live, page):
    from service import db
    from service import proposals as prop
    pid = db.proposal_create(MODEL, "everyday:writing", "Writing", "masein",
                             {"kind": "everyday", "group": "writing"})
    db.proposal_update(pid, status="approved", spec_text="write plainly")
    did = db.dataset_create(pid, "chat", 2, "masein", {})
    d = prop.dataset_dir(did)
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"user": "shorten: the meeting is moved to 3pm on friday", "assistant":
         "Meeting moved to 3pm Friday.", "checks": [{"type": "max_words", "n": 8}]},
        {"user": "one line pls: bring snacks", "assistant": "Please bring snacks.",
         "checks": [{"type": "line_count", "n": 1}]}]) + "\n", encoding="utf-8")
    db.dataset_update(did, status="ready", provenance=json.dumps(
        {"format": "chat", "items": {"kept": 2, "requested": 2}}))
    try:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto(live["base"] + "/#tab=improve&sub=model&model=" + quote(MODEL, safe="")
                  + f"&read=dataset:{did}")
        page.wait_for_selector("#reader[data-ready='1']")
        ex = page.locator("[data-chat-example]")
        ex.wait_for()
        text = ex.inner_text()
        low = text.lower()                           # the labels are set in capitals
        assert "request" in low and "reply" in low and "Meeting moved to 3pm Friday." in text
        assert page.locator("[data-chat-checks]").inner_text() == "at most 8 words"
        assert "chat examples, each passed its own checks" in page.locator("#reader").inner_text()
        shot(page, "12g2-chat-dataset-1400-light.png")
        assert page.errors == []
    finally:
        c = db._conn()                        # noqa: SLF001
        c.execute("DELETE FROM datasets WHERE id = ?", (did,))
        c.execute("DELETE FROM proposals WHERE id = ?", (pid,))
        c.commit()
        c.close()


def test_the_model_pages_improve_tab_shows_the_same_stages(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + quote(MODEL, safe=""))
    page.wait_for_selector("[data-model-hero]")
    page.wait_for_function("() => state.rv.loaded")
    page.locator("[data-mtab='improve']").click()
    panel = page.locator("[data-mtab-panel='improve']")
    panel.locator("[data-stages]").wait_for()
    assert [s.get_attribute("data-stage") for s in panel.locator("[data-stage]").all()] == \
        ["weak", "proposals", "data", "retests"]
    assert panel.locator("[data-weak-kind='everyday']").count() >= 1
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1400])
def test_the_screens(live, page, width):
    pipeline(page, live["base"], width=width)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12g2-pipeline-{width}-light.png", full_page=True)
    assert page.errors == []
