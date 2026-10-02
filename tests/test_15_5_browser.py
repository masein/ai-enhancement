"""15.5 on the page: a DeviceMark row answered in shards says so on its
model's page — "Run on a rented GPU (…), in 3 shards", or, on shards that ran
on different cards, "IFEval run on rented GPUs (… and …), in 2 shards;
MMLU-Pro and MATH on this server". At 1400 and 375 px, with no sideways
scroll. Rows written as an import of shards leaves them; nothing runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from service import config
from test_12q_devicemark_board import HF, _row
from test_12q_devicemark_model_page_browser import model_page, open_block

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase15_5"
GOOD = "fx/good-750m"
WHOLE = "run on a rented GPU (NVIDIA GeForce RTX 4090), in 3 shards"
PART = ("IFEval run on rented GPUs (NVIDIA GeForce RTX 4090 and NVIDIA GeForce RTX 5090), in 2 "
        "shards; MMLU-Pro and MATH on this server")


@pytest.fixture(scope="module", autouse=True)
def rows(live):
    import service.app as appmod
    _row(config.OUT_DIR, GOOD, {"ifeval": 0.6, "mmlu_pro": 0.5, "math": 0.7},
         {**HF, "where": WHOLE}, thinking=True)
    _row(config.OUT_DIR, GOOD, {"ifeval": 0.55, "mmlu_pro": 0.45, "math": 0.6},
         {**HF, "where": PART})
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    for suffix in ("", "__thinking"):
        for f in (config.OUT_DIR / (GOOD.replace("/", "__") + suffix)).glob("devicemark*"):
            f.unlink()
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width", [1400, 375])
def test_the_model_page_says_the_row_ran_in_shards(live, page, width):
    model_page(page, live, GOOD, width)
    on = open_block(page, "dm_thinking")
    assert on.locator("[data-dm-card-where='on']").inner_text() == WHOLE[0].upper() + WHOLE[1:]
    off = open_block(page, "dm")
    assert off.locator("[data-dm-card-where='off']").inner_text() == PART
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    SCREENS.mkdir(parents=True, exist_ok=True)
    on.locator("[data-dm-card]").screenshot(path=SCREENS / f"card-{width}.png")
    assert page.errors == []
