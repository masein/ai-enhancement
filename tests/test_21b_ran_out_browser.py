"""21b in a real browser, at 1280 and 400 px: the report of the answers that
ran out of room — its "Insight only" card, the warning at the top and above
an opened question, the picker and the filters, an opened answer's thinking
loaded with its repeated lines and its first right answer marked, GPQA's
question text nowhere, nothing scrolling sideways; and the scorecard's links
into it. Invented questions and answers (tests/test_21b_ran_out.py's world)."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21b_ran_out import GPQA_Q, world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase21b"


def no_sideways(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def opened(page, live, width: int) -> None:
    page.set_viewport_size({"width": width, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports")
    card = page.locator("[data-report-card='ran-out-of-room']")
    card.wait_for()
    assert card.locator("[data-report-insight]").inner_text() == "Insight only"
    card.click()
    page.wait_for_selector("[data-report='ran-out-of-room'] [data-ro-warning]")


@pytest.mark.parametrize("width", [1280, 400])
def test_the_warning_the_picker_the_filters_and_an_opened_answer(live, page, width):
    world()
    opened(page, live, width)
    warn = page.locator("[data-report='ran-out-of-room'] [data-ro-warning]")
    assert warn.is_visible() and warn.inner_text().startswith(
        "For insight only — not for training.")
    assert warn.evaluate("e => e.getBoundingClientRect().top") < 900      # the top
    # the list, once worked out on the server
    page.wait_for_selector("[data-ro-row]", timeout=30000)
    assert page.locator("[data-ro-list] [data-ro-warning]").is_visible()
    s4 = page.locator("[data-ro-row]").count()
    assert s4 == 4                                            # 486, 1 (hidden half), GPQA, MATH
    assert page.locator("[data-ro-summary] [data-ro-bench='mmlupro']").inner_text().startswith(
        "MMLU-Pro")
    # the picker
    page.locator("[data-ro-pick='step1000']").click()
    page.wait_for_function("document.querySelectorAll('[data-ro-row]').length === 3")
    page.locator("[data-ro-pick='step4000']").click()
    page.wait_for_function("document.querySelectorAll('[data-ro-row]').length === 4")
    # the filters: a benchmark, then a subject
    page.locator("[data-ro-filter-bench]").select_option("mmlupro")
    page.wait_for_function("document.querySelectorAll('[data-ro-row]').length === 2")
    page.locator("[data-ro-filter-subject]").select_option("law")
    page.wait_for_function("document.querySelectorAll('[data-ro-row]').length === 1")
    assert "hidden half" in page.locator("[data-ro-row]").first.inner_text()
    page.locator("[data-ro-filter-subject]").select_option("")
    page.locator("[data-ro-sort]").select_option("waits")
    assert no_sideways(page)
    # GPQA's question text nowhere on the page
    page.locator("[data-ro-filter-bench]").select_option("")
    page.wait_for_function("document.querySelectorAll('[data-ro-row]').length === 4")
    assert GPQA_Q[:30] not in page.content()
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.set_viewport_size({"width": width, "height": 1800})
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(150)
    page.screenshot(path=SCREENS / f"ran-out-{width}.png")
    page.set_viewport_size({"width": width, "height": 900})
    # one question opened: the warning above it, its thinking loaded and marked
    page.locator("[data-ro-row='mmlupro_tiger:486:0']").click()
    page.wait_for_selector("[data-ro-open] [data-ro-thinking]", timeout=30000)
    assert page.evaluate("location.hash").endswith("&open=mmlupro_tiger%3A486%3A0")
    assert page.locator("[data-ro-open] [data-ro-warning]").is_visible()
    think = page.locator("[data-ro-thinking]")
    assert "Wait, let me re-check" in think.inner_text()
    assert think.locator("mark.ro-rep").count() >= 3
    assert think.locator("[data-ro-first]").inner_text().lower().startswith("the answer is (e)")
    assert "Right answer: (E)" in page.locator("[data-ro-right]").inner_text()
    assert no_sideways(page)
    # GPQA's opened: its thinking, the quoted line withheld, no question text
    page.go_back()
    page.wait_for_selector("[data-ro-row='gpqa_diamond_epoch:recQ1:0']")
    page.locator("[data-ro-row='gpqa_diamond_epoch:recQ1:0']").click()
    page.wait_for_selector("[data-ro-open] [data-ro-thinking]", timeout=30000)
    assert "[a line quoting the question, withheld]" in page.locator(
        "[data-ro-thinking]").inner_text()
    assert page.locator("[data-ro-withheld]").count() == 1
    assert GPQA_Q[:30] not in page.content()
    assert page.errors == []


def test_the_scorecard_links_into_it(live, page):
    world()
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    link = page.locator("[data-ro-link='step4000|mmlupro']")
    link.wait_for(timeout=30000)
    link.click()
    page.wait_for_selector("[data-report='ran-out-of-room']")
    page.wait_for_selector("[data-ro-row]", timeout=30000)
    assert page.locator("[data-ro-filter-bench]").input_value() == "mmlupro"
    assert "model=step4000" in page.evaluate("location.hash")
    # a loop example opens its whole answer there
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    page.locator("[data-ro-open-link='486']").click(timeout=30000)
    page.wait_for_selector("[data-ro-open='mmlupro_tiger:486:0'] [data-ro-thinking]",
                           timeout=30000)
    assert page.errors == []
