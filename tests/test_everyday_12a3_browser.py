"""12a.3 on the page: the question list is the 333 questions, each group
with its count, and round 3's two new checks say what they look for in plain
words — "keeps at least 4 of: …" and "picks door, not desk"."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a3"
SIZES = {"understanding": 45, "writing": 48, "summarising": 63, "transform": 46,
         "quick_maths": 45, "instructions": 45, "honesty": 41}


def everyday_page(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-bank]")


def test_the_question_list_is_333_with_each_groups_count(live, page):
    everyday_page(page, live["base"])
    bank = page.locator("[data-everyday-bank]")
    assert bank.locator("[data-evd-bank-q]").count() == 333
    assert "333 questions in seven groups" in bank.inner_text()
    for g, n in SIZES.items():
        head = bank.locator(f"[data-evd-bank-group='{g}'] > summary")
        assert head.inner_text().endswith(f"· {n} questions"), g
        assert bank.locator(f"[data-evd-bank-group='{g}'] [data-evd-bank-q]").count() == n
    # the results table says the same count beside each group
    for g, n in SIZES.items():
        assert page.locator(f"[data-everyday-table] tr[data-evd-g='{g}'] .evq-group") \
            .inner_text() == f"{n} questions"
    assert page.errors == []


def test_the_new_checks_are_said_in_plain_words(live, page):
    everyday_page(page, live["base"])
    page.locator("[data-evd-bank-group='understanding'] > summary").click()
    # first_mention: the right one, not the wrong one
    picks = page.locator("[data-evd-checks='everyday-understanding-r3-04']")
    assert picks.inner_text() == 'Passes if it: says "door" · picks door, not desk'
    # facts: how many of the key facts, and which
    page.locator("[data-evd-bank-group='summarising'] > summary").click()
    words = page.evaluate("""() => DATA.everyday.questions
      .filter(q => q.group === 'summarising' && q.id.includes('-r3-'))
      .map(q => document.querySelector(`[data-evd-checks='${q.id}']`).textContent)""")
    assert len(words) == 46
    assert all(w.startswith("Passes if it: at most ") and " · keeps at least " in w
               for w in words), words[:3]
    # no check is said by its type
    text = page.locator("[data-everyday-bank]").inner_text()
    for word in ("first_mention", "facts:", "max_words", "in_order", "not_contains"):
        assert word not in text, word
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1400])
def test_the_screens(live, page, width):
    everyday_page(page, live["base"], width)
    page.locator("[data-evd-bank-group='summarising'] > summary").click()
    page.locator("[data-evd-bank-q='everyday-summarising-r3-01']").scroll_into_view_if_needed()
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(200)
    page.screenshot(path=SCREENS / f"12a3-questions-{width}-light.png")
    page.locator("[data-evd-cell='fx/good-750m|summarising']").click()
    page.wait_for_selector("[data-evd-panel]")
    page.locator("[data-everyday-results]").screenshot(
        path=SCREENS / f"12a3-results-{width}-light.png")
    assert page.errors == []
