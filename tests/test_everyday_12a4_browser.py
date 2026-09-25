"""12a.4 on the page: the new checks in plain words; the wording's version on
the Everyday tab; a model's answers to an earlier wording in its History,
labelled, and in no score or comparison; "n answers ran out of room" beside a
score; and Home's Everyday badge as one short line with room around it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import everyday as ev
from conftest import open_kind

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a4"
HASH = ev.version()["hash"]


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def served(page, edit):
    """the board as served, edited on the way to the page"""
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def everyday_page(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-bank]")


def test_the_new_checks_are_said_in_plain_words(live, page):
    everyday_page(page, live["base"])
    page.locator("[data-evd-bank-group='honesty'] > summary").click()
    either = page.locator("[data-evd-checks='everyday-honesty-04']")
    assert either.inner_text() == \
        "Passes if it: asks what you meant, or says it can't know or do this · no invented url"
    words = page.evaluate("""() => DATA.everyday.questions.filter(q => q.group === 'honesty')
      .map(q => document.querySelector(`[data-evd-checks='${q.id}']`).textContent)""")
    assert sum("says it can't know or do this" in w for w in words) >= 20
    text = page.locator("[data-everyday-bank]").inner_text()
    for word in ("admits_limit", "asks_back", "any:", "says none of"):
        assert word not in text, word
    assert page.errors == []


def test_the_tab_says_which_wording_it_shows(live, page):
    everyday_page(page, live["base"])
    line = page.locator(f"[data-everyday-head] [data-evd-version='{HASH}']")
    assert line.inner_text().startswith(f"This wording: 2026-09-25 · {HASH}.")
    # the two that answered this wording; the two that sat only the pilot are not here
    assert [th.get_attribute("data-evd-model") for th in
            page.locator("[data-everyday-table] th[data-evd-model]").all()] == \
        ["fx/good-750m", "fx/skewed-360m"]
    assert page.errors == []


def test_an_earlier_wording_is_in_history_and_nowhere_else(live, page):
    from service import db
    model = "fx/chance-160m"
    # an everyday run from before the version, and one on it
    old = db.add(model, "instruct", "everyday", "masein", "")
    db.update(old, status="done", progress="Everyday tasks: 1 of 5")
    new = db.add("fx/good-750m", "instruct", "everyday", "masein", "")
    db.update(new, status="done", progress="Everyday tasks: 330 of 333", bank_version=HASH)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=fx%2Fchance-160m")
    page.wait_for_selector("[data-model-hero]")
    # the header: not tested on this wording, and where the earlier answers are
    tile = page.locator(f"[data-everyday-none='{model}']")
    assert "Not tested" in tile.inner_text()
    assert tile.locator(f"[data-evd-earlier-note='{model}']").inner_text() == \
        "answered an earlier wording · in History"
    assert page.locator("[data-kind-block='everyday']").count() == 0
    # History: the earlier answers, as they were marked, labelled
    page.locator("[data-mtab='history']").click()
    card = page.locator(f"[data-evd-earlier='{model}']")
    card.wait_for()
    assert card.locator("[data-earlier-badge]").inner_text() == "earlier wording"
    assert card.locator("[data-evd-earlier-count]").inner_text().startswith("1 of 5")
    run = page.locator(f"tr[data-run='{old}']")
    run.wait_for()
    assert run.locator(f"[data-earlier-run='{old}']").inner_text() == "earlier wording"
    shot(page, "12a4-history-1400-light.png", full_page=True)
    # a run on this wording carries no such label
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    page.locator("[data-mtab='history']").click()
    page.wait_for_selector(f"tr[data-run='{new}']")
    assert page.locator(f"[data-earlier-run='{new}']").count() == 0
    assert page.locator("[data-evd-earlier]").count() == 0
    # Compare and Home count only this wording
    page.goto(live["base"] + "/#tab=models&view=everyday")
    page.wait_for_selector("[data-lb-everyday]")
    assert page.locator(f"tr[data-lb-row='{model}'] [data-everyday-count]").count() == 0
    assert page.errors == []


def test_answers_that_ran_out_of_room_are_said_beside_the_score(live, page):
    """skewed-360m's first answer never left its thinking; good-750m has none"""
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=fx%2Fskewed-360m")
    page.wait_for_selector("[data-model-hero]")
    tile = page.locator("[data-kind-tile='everyday']")
    assert tile.locator("[data-evd-ran-out]").inner_text() == "1 answer ran out of room"
    open_kind(page, "everyday")
    assert page.locator("[data-everyday-block] [data-evd-ran-out='1']").count() == 1
    page.goto(live["base"] + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-table]")
    head = page.locator("[data-everyday-table] th[data-evd-model='fx/skewed-360m']")
    assert head.locator("[data-evd-ran-out]").inner_text() == "1 answer ran out of room"
    assert page.locator("th[data-evd-model='fx/good-750m'] [data-evd-ran-out]").count() == 0
    page.goto(live["base"] + "/#tab=models&view=everyday")
    page.wait_for_selector("[data-lb-everyday]")
    assert page.locator("tr[data-lb-row='fx/skewed-360m'] [data-evd-ran-out]").inner_text() == \
        "1 answer ran out of room"
    assert page.errors == []


def provisional(body):
    for e in body["everyday"]["models"].values():
        e["provisional"] = True


@pytest.mark.parametrize("width", [1280, 400])
def test_homes_everyday_badge_is_one_line_with_room(live, page, width):
    served(page, provisional)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/")
    card = page.locator("[data-best='everyday']")
    card.wait_for()
    badge = card.locator("[data-pilot-badge]")
    assert badge.inner_text() == "not ranked · provisional judge"
    # found and measured in one step: Home redraws when the queue or the review
    # lists arrive, and a badge found before a redraw measures as nothing after it
    box = page.evaluate("""() => {
      const b = document.querySelector("[data-best='everyday'] [data-pilot-badge]"),
        s = getComputedStyle(b), r = b.getBoundingClientRect(),
        e = b.closest('.hcard').querySelector('.eyebrow').getBoundingClientRect(),
        v = b.closest('.hcard').querySelector('.hcard-v').getBoundingClientRect();
      return { lines: b.getClientRects().length, h: r.height,
               lh: parseFloat(s.lineHeight) || 1.2 * parseFloat(s.fontSize),
               above: r.top - e.bottom, below: v.top - r.bottom,
               inside: r.right <= b.closest('.hcard').getBoundingClientRect().right }; }""")
    # one line, under the heading and not on it, with the same room above and below
    assert box["lines"] == 1 and box["h"] < 2 * box["lh"], box
    assert box["above"] >= 6 and abs(box["above"] - box["below"]) <= 2, box
    assert box["inside"], box
    # the eyebrow is the heading alone now
    assert card.locator(".eyebrow").inner_text().lower() == "everyday tasks"
    card.scroll_into_view_if_needed()
    shot(page, f"12a4-home-{width}-light.png")
    assert page.errors == []
