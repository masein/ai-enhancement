"""15.7 on the page: a DeviceMark row's numbers are provisional until it's
scored by today's reading, or past it when it needs a new run — the On-device
chart's table marks it, with why on hover, and the model page's card says why
in one line and which reading scored it. At 1400 and 375 px, with no sideways
scroll. Rows written as runs and re-scoring leave them; nothing runs."""

from __future__ import annotations

from pathlib import Path

import pytest

import devicemark as dm
from service import config
from test_12q_devicemark_board import HF, _row
from test_12q_devicemark_model_page_browser import model_page, open_block

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase15_7"
GOOD = "fx/good-750m"
WHY = ("needs a new run: 596 of its answers can't be told from its thinking — the run before "
       "15.7 kept neither its thinking markers nor where the thinking ended")


@pytest.fixture(scope="module", autouse=True)
def rows(live):
    import service.app as appmod
    _row(config.OUT_DIR, GOOD, {"ifeval": 0.4, "mmlu_pro": 0.5, "math": 0.7},
         {**HF, "scoring": "v1", "provisional": WHY}, thinking=True)
    _row(config.OUT_DIR, GOOD, {"ifeval": 0.8, "mmlu_pro": 0.5, "math": 0.6}, HF)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    for suffix in ("", "__thinking"):
        for f in (config.OUT_DIR / (GOOD.replace("/", "__") + suffix)).glob("devicemark*"):
            f.unlink()
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width", [1400, 375])
def test_a_provisional_row_says_why_and_which_reading_scored_it(live, page, width):
    model_page(page, live, GOOD, width)
    on = open_block(page, "dm_thinking")
    assert on.locator("[data-dm-card-provisional='on']").inner_text() == f"Provisional: {WHY}"
    assert on.locator("[data-dm-card-scoring='on']").inner_text() == "Scoring v1"
    off = open_block(page, "dm")
    assert off.locator("[data-dm-card-provisional='off']").count() == 0
    assert off.locator("[data-dm-card-scoring='off']").inner_text().startswith(
        f"Scoring {dm.SCORING}: thinking on, the answer after the closed thinking")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    SCREENS.mkdir(parents=True, exist_ok=True)
    on.locator("[data-dm-card]").screenshot(path=SCREENS / f"card-{width}.png")
    # the chart's table marks it, why on hover; the row scored by today's isn't marked
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&chip=ondevice")
    page.wait_for_selector("[data-dm-table]")
    badge = page.locator(f"[data-dm-provisional='{GOOD} · thinking']")
    assert badge.inner_text() == "provisional" and WHY in badge.get_attribute("data-tip")
    assert page.locator(f"[data-dm-provisional='{GOOD}']").count() == 0
    page.locator(f"tr[data-dm-row='{GOOD} · thinking']").screenshot(
        path=SCREENS / f"chart-row-{width}.png")
    assert page.errors == []
