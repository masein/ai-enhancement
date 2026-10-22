"""27's Thinking filter, "as each runs": each model shows the setting with the
most scores in the columns shown, a tie to thinking on. Since plan I a
build's thinking-off row holds only IFEval and IFBench; it showed in place of
the full thinking-on row, every other column "–", and sank to the bottom of
Ours. The world is 25's: 21c's scorecard (our builds thinking on, every
Frontier benchmark) with IFEval and IFBench scored thinking on and off on top;
every question and answer in it is invented."""

from __future__ import annotations

import pytest

from test_21c_page import SPEC
from test_25_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
ON = SPEC["Q4 (UD-Q4_K_XL)"]["row"]
OFF = ON.removesuffix(" · thinking")


@pytest.fixture(scope="module", autouse=True)
def registered_off(board):  # noqa: F811
    """Q4 (lda) served as on the board: registered thinking off, so "as each
    runs" used to show its thinking-off row"""
    import json

    import service.app as appmod
    from service import db, reports
    db.served_put({**db.served_get(OFF), "thinking": "off"})
    # the board reads how a row was served from each of its runs' results
    for f in reports.row_dir(OFF).rglob("results_*.json"):
        got = json.loads(f.read_text())
        if isinstance(got.get("served"), dict):
            got["served"]["thinking"] = "off"
            f.write_text(json.dumps(got))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def frontier(page, live, mode=None):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/")
    page.evaluate("m => m ? localStorage.setItem('bench-mt-thinking', m) "
                  ": localStorage.removeItem('bench-mt-thinking')", mode)
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&view=frontier")
    page.wait_for_selector("[data-frontier-table] tbody tr.mt-row", timeout=30000)


def scored(page, mid: str) -> int:
    return page.locator(f"tr.mt-row[data-fr-row='{mid}'] td[data-fr-here]").count()


def test_as_each_runs_shows_the_setting_with_the_most_scores(live, page):
    assert ON.endswith(" · thinking")
    frontier(page, live)
    # Q4 (lda): its thinking-on row, every Frontier benchmark; its thinking-off
    # row (IFEval and IFBench only) not shown
    assert page.locator(f"tr.mt-row[data-fr-row='{ON}']").count() == 1
    assert page.locator(f"tr.mt-row[data-fr-row='{OFF}']").count() == 0
    assert scored(page, ON) >= 6
    gpqa = page.locator(f"tr.mt-row[data-fr-row='{ON}'] td[data-fr-cell='gpqa'] .mt-n")
    assert gpqa.inner_text() != "–"
    # …so it no longer sinks: it sorts above every row of Ours with no GPQA
    ours = page.eval_on_selector_all(
        "[data-frontier-table] tr.mt-row[data-mt-group='ours']:not(.mt-sub-row)",
        "rs => rs.map(r => [r.dataset.frRow, !!r.querySelector(\"td[data-fr-cell='gpqa'] .mt-n\")])")
    at = [r[0] for r in ours].index(ON)
    assert all(has for _, has in ours[:at]), ours
    assert page.errors == []


def test_off_and_both_still_show_the_other_setting(live, page):
    # off: the thinking-off row, with its two benchmarks
    frontier(page, live, "off")
    assert page.locator(f"tr.mt-row[data-fr-row='{OFF}']").count() == 1
    assert page.locator(f"tr.mt-row[data-fr-row='{ON}']").count() == 0
    assert scored(page, OFF) == 2
    # both: the one with the most scores, the other as its sub-row
    frontier(page, live, "both")
    row = page.locator(f"tr.mt-row[data-fr-row='{ON}']")
    assert row.count() == 1 and "mt-sub-row" not in row.get_attribute("class")
    sub = page.locator(f"tr.mt-row.mt-sub-row[data-fr-row='{OFF}']")
    assert sub.count() == 1
    assert page.errors == []


@pytest.mark.parametrize("width", [1280, 400])
def test_frontier_as_each_runs_screens(live, page, width):
    """the reply's screenshot: Frontier with Thinking "as each runs", dark"""
    from pathlib import Path
    page.emulate_media(color_scheme="dark")
    frontier(page, live)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&view=frontier")
    page.wait_for_selector("[data-lb-card] [data-mt-card]" if width < 640
                           else "[data-frontier-table] tbody tr.mt-row", timeout=30000)
    if width < 640:
        assert page.locator(f"[data-mt-card='{ON}']").count() == 1
        assert page.locator(f"[data-mt-card='{OFF}']").count() == 0
    screens = Path(__file__).resolve().parent / "_screens" / "phase27c"
    screens.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(300)
    page.locator("[data-lb-card]").screenshot(path=str(screens / f"frontier-{width}-dark.png"))
    assert page.errors == []
