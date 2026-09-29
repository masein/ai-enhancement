"""12a.3 on the page: the question list is the bank's questions (12a.5:
340 in seven groups since 12a.6), each group with its count, and round 3's two new checks say what they look for in plain
words — "keeps at least 4 of: …" and "picks door, not desk"."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a3"
# 12a.6: one Summarise group — the long texts and the fifteen short ones kept
# 12g.2: each group's hidden half (scores, never shown) and practice half (shown);
# 12p.2: the hidden half is the tests' invented one
SPLIT = {"understanding": (21, 18), "writing": (21, 24), "summarising": (22, 29),
         "transform": (22, 21), "quick_maths": (21, 21),
         "instructions": (21, 21), "honesty": (21, 27)}
SIZES = {g: h + p for g, (h, p) in SPLIT.items()}


def everyday_page(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-bank]")


def test_the_question_list_is_340_with_each_groups_count(live, page):
    everyday_page(page, live["base"])
    bank = page.locator("[data-everyday-bank]")
    # 12g.2: the 340 (12a.6), split — the practice half listed, the hidden half counted
    assert bank.locator("[data-evd-bank-q]").count() == 161
    assert "161 practice questions in seven groups; 149 more are hidden" in bank.inner_text()
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
    # facts: how many of the key facts, and which — 12a.6: Writing's, where they
    # stay; round 3's short summaries are retired
    page.locator("[data-evd-bank-group='writing'] > summary").click()
    words = page.locator("[data-evd-bank-group='writing'] [data-evd-checks]").all_inner_texts()
    assert any(" keeps at least " in w for w in words), words[:3]
    # 12a.6: a Summarise question says its gate and what the judge's rubric asks
    page.locator("[data-evd-bank-group='summarising'] > summary").click()
    assert page.locator("[data-evd-checks='everyday-summarising-01']").inner_text() == (
        # 12a.7: on what it says; 12a.8: the judge's findings, scored in code
        "Passes if it: no number the question doesn’t give · the judge's findings, scored in "
        "code (0 to 4, passing at 3): a missing key fact −1 (2 at most), anything invented or "
        "wrong −2, several versions −1, a length asked and not kept −1 — never the style")
    # no check is said by its type
    text = page.locator("[data-everyday-bank]").inner_text()
    for word in ("first_mention", "facts:", "max_words", "in_order", "not_contains"):
        assert word not in text, word
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1400])
def test_the_screens(live, page, width):
    everyday_page(page, live["base"], width)
    page.locator("[data-evd-bank-group='summarising'] > summary").click()
    page.locator("[data-evd-bank-q='everyday-summarising-01']").scroll_into_view_if_needed()
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(200)
    page.screenshot(path=SCREENS / f"12a3-questions-{width}-light.png")
    page.locator("[data-evd-cell='fx/good-750m|summarising']").click()
    page.wait_for_selector("[data-evd-panel]")
    page.locator("[data-everyday-results]").screenshot(
        path=SCREENS / f"12a3-results-{width}-light.png")
    assert page.errors == []
