"""12a.3 on the page: the question list is the bank's questions (12a.5:
388 in eight groups), each group with its count, and round 3's two new checks say what they look for in plain
words — "keeps at least 4 of: …" and "picks door, not desk"."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a3"
# 12a.5: the short summaries are "shorten", the long texts "summarising"
SIZES = {"understanding": 45, "writing": 48, "shorten": 63, "summarising": 45,
         "transform": 46, "quick_maths": 45, "instructions": 45, "honesty": 51}
# 12g.2: each group's hidden half (scores, never shown) and practice half (shown)
SPLIT = {"understanding": (27, 18), "writing": (24, 24), "shorten": (26, 37),
         "summarising": (26, 19), "transform": (25, 21), "quick_maths": (24, 21),
         "instructions": (24, 21), "honesty": (24, 27)}


def everyday_page(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-bank]")


def test_the_question_list_is_388_with_each_groups_count(live, page):
    everyday_page(page, live["base"])
    bank = page.locator("[data-everyday-bank]")
    # 12g.2: the 388, split — the practice half listed, the hidden half counted
    assert bank.locator("[data-evd-bank-q]").count() == 188
    assert "188 practice questions in eight groups; 200 more are hidden" in bank.inner_text()
    for g, n in SIZES.items():
        hidden, practice = SPLIT[g]
        assert hidden + practice == n
        head = bank.locator(f"[data-evd-bank-group='{g}'] > summary")
        assert head.inner_text().endswith(f"· {practice} practice · {hidden} hidden"), g
        assert bank.locator(f"[data-evd-bank-group='{g}'] [data-evd-bank-q]").count() == practice
    # the results table says the same beside each group
    for g, (hidden, practice) in SPLIT.items():
        assert page.locator(f"[data-everyday-table] tr[data-evd-g='{g}'] .evq-group") \
            .inner_text() == f"{hidden} hidden · {practice} practice"
    assert page.errors == []


def test_the_new_checks_are_said_in_plain_words(live, page):
    everyday_page(page, live["base"])
    page.locator("[data-evd-bank-group='understanding'] > summary").click()
    # first_mention: the right one, not the wrong one
    picks = page.locator("[data-evd-checks='everyday-understanding-r3-04']")
    assert picks.inner_text() == 'Passes if it: says "door" · picks door, not desk'
    # facts: how many of the key facts, and which
    page.locator("[data-evd-bank-group='shorten'] > summary").click()
    words = page.evaluate("""() => DATA.everyday.questions
      .filter(q => q.group === 'shorten' && q.id.includes('-r3-'))
      .map(q => document.querySelector(`[data-evd-checks='${q.id}']`).textContent)""")
    assert len(words) == 26                      # 12g.2: round 3's practice ones
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
    page.locator("[data-evd-bank-group='shorten'] > summary").click()
    page.locator("[data-evd-bank-q='everyday-summarising-r3-02']").scroll_into_view_if_needed()
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(200)
    page.screenshot(path=SCREENS / f"12a3-questions-{width}-light.png")
    page.locator("[data-evd-cell='fx/good-750m|shorten']").click()
    page.wait_for_selector("[data-evd-panel]")
    page.locator("[data-everyday-results]").screenshot(
        path=SCREENS / f"12a3-results-{width}-light.png")
    assert page.errors == []
