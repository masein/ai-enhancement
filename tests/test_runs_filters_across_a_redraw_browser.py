"""Runs' filters across a whole-page redraw. The judge's health answers on its
own, and its first answer redraws the whole page on Runs; when the probe is
slow it lands after the list is drawn — while someone types in the filter or
has the where filter open. The typing went to the page's body and was lost,
and an option chosen from the open list redrew the list the redraw had
replaced, so the rows on the page stayed as they were (CI's Runs tests timed
out on both, now and then, late in a shard). Here the judge's answer is held
until that moment, then let through."""

from __future__ import annotations

import pytest

from test_17i_runs_browser import run_row
from test_17j_page_browser import HLE, world

pytestmark = pytest.mark.dashboard


def held_judge(page) -> list:
    """the judge's health held back until the test lets it through"""
    held: list = []
    page.route("**/api/judge/health", lambda route: held.append(route))
    return held


def let_through(page, held: list) -> None:
    for route in held:
        route.continue_()
    page.unroute("**/api/judge/health")                       # and any asked since
    page.wait_for_function("!!state.judgeHealth")             # its answer drawn


def runs(page, live) -> None:
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("[data-queue-table]")


def test_typing_in_the_filter_survives_the_redraw(live, page):
    world()
    held = held_judge(page)
    runs(page, live)
    box = page.get_by_label("filter queue")
    # as a person's typing (and Playwright's fill) goes: the field focused,
    # then the letters — the redraw lands in between
    box.focus()
    let_through(page, held)
    page.keyboard.insert_text("17j-")
    page.wait_for_function("state.qQ === '17j-'", timeout=5000)
    assert page.get_by_label("filter queue").input_value() == "17j-"
    assert page.evaluate("document.activeElement.getAttribute('aria-label')") == "filter queue"
    run_row(page, "served/17j-bf16", HLE).wait_for(timeout=5000)
    assert page.errors == []


def test_an_option_chosen_after_the_redraw_filters_the_list_on_the_page(live, page):
    world()
    held = held_judge(page)
    runs(page, live)
    page.get_by_label("filter queue").fill("17j-")
    page.wait_for_function("state.qQ === '17j-'")
    page.wait_for_selector("[data-rented-run]")
    page.get_by_label("where filter").click()
    options = page.locator("[role=listbox][aria-label='where filter']")
    options.wait_for()
    let_through(page, held)                                   # the list open meanwhile
    options.locator("[data-value='here']").click()
    # the choice itself redraws the list on the page — not a poll some
    # seconds on (before, the four rented rows stayed until one did)
    assert page.evaluate("[state.qWhere, document.querySelectorAll('[data-rented-run]')"
                         ".length]") == ["here", 0]
    assert page.get_by_label("where filter").inner_text().startswith("this server")
    assert page.locator("[data-queue-row]").count() == 4      # the four runs here
    assert page.errors == []
