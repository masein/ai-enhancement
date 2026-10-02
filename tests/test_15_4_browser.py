"""15.4 on the page: a DeviceMark row's "raw" link — set on its model page's
card (the form PUTs it, as export_devicemark_raw.py --link does), shown there
and beside the row in the On-device chart's table, as DeviceMark's own rows
link theirs. At 1400 and 375 px. A row written as a run leaves one."""

from __future__ import annotations

from pathlib import Path

import pytest

import devicemark as dm
from conftest import set_name
from service import config
from test_12q_devicemark_board import HF, _row
from test_12q_devicemark_model_page_browser import model_page, open_block

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase15_4"
GOOD = "fx/good-750m"
URL = "https://huggingface.co/datasets/masein/evalboard-devicemark-raw/tree/main/fx__good-750m__thinking"


@pytest.fixture(scope="module", autouse=True)
def rows(live):
    import service.app as appmod
    _row(config.OUT_DIR, GOOD, {"ifeval": 0.6, "mmlu_pro": 0.5, "math": 0.7}, HF, thinking=True)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    d = config.OUT_DIR / (GOOD.replace("/", "__") + "__thinking")
    for f in d.glob("devicemark*"):
        f.unlink()
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width", [1400, 375])
def test_the_card_sets_its_raw_link_and_the_chart_shows_it(live, page, width):
    (config.OUT_DIR / (GOOD.replace("/", "__") + "__thinking") / dm.RAW_NAME).unlink(
        missing_ok=True)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    model_page(page, live, GOOD, width)
    card = open_block(page, "dm_thinking")
    card.locator("[data-dm-raw-edit='on']").click()
    card.locator("[data-dm-raw-input='on']").fill(URL)
    card.locator("[data-dm-raw-save='on']").click()
    link = card.locator("[data-dm-card-raw='on']")
    link.wait_for()
    assert link.get_attribute("href") == URL and link.inner_text() == "raw"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    SCREENS.mkdir(parents=True, exist_ok=True)
    card.locator("[data-dm-card]").screenshot(path=SCREENS / f"card-{width}.png")
    # beside the row in the On-device chart's table, as DeviceMark's rows link theirs
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&chip=ondevice")
    page.wait_for_selector("[data-dm-table]")
    raw = page.locator(f"[data-dm-raw='{GOOD} · thinking']")
    assert raw.get_attribute("href") == URL
    page.locator(f"tr[data-dm-row='{GOOD} · thinking']").screenshot(
        path=SCREENS / f"chart-row-{width}.png")
    assert page.errors == []
