"""12o.1 on the page: the Frontier view's calibration line says whether our
number was measured thinking — "measured here 36.0 (thinking off) · Epoch
38.0" — and a thinking row, its model asked to think, is calibrated against
the same reported number, shown beside it and never ranked a second time.
Test a model offers thinking for the shared suite.

Invented GPQA-shaped results and Epoch's trimmed file only."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from service import reported
from test_12n2 import sit_gpqa
from test_reported_12m2 import epoch_fetch, epoch_zip

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12o1"
GOOD = "fx/good-750m"
THINKING = GOOD + " · thinking"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    tree = live["tree"]
    good = tree["models"][GOOD]["dir"]
    # a model with a thinking switch: asked with it off, and once with it on
    meta = json.loads((good / "model_meta.json").read_text())
    (good / "model_meta.json").write_text(json.dumps({**meta, "thinking": "switch"}))
    sit_gpqa(good, acc=0.36)
    row = good.with_name(good.name + "__thinking")
    sit_gpqa(row, acc=0.40)
    res = next(row.glob("gpqa_diamond_cot_zeroshot_0shot/*/results_*.json"))
    blob = json.loads(res.read_text())
    blob["config"]["model_args"] = f"pretrained={GOOD},dtype=bfloat16,enable_thinking=True"
    res.write_text(json.dumps(blob))
    (row / "model_meta.json").write_text(json.dumps({**meta, "model": THINKING,
                                                     "base_model": GOOD, "thinking": "switch"}))
    reported.alias_set("alibaba/qwen3-1.7b", GOOD, "masein")
    reported.import_epoch(fetch=epoch_fetch(epoch_zip()))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def test_the_calibration_line_says_thinking_off_or_on(live, page):
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    off = page.locator(f"[data-fr-cal='{GOOD}'][data-fr-cell='gpqa diamond']")
    assert off.inner_text().replace("\n", "") == "measured here 36.0 (thinking off) · Epoch 38.0"
    assert off.get_attribute("data-fr-thinking") == "off"
    on = page.locator(f"[data-fr-cal='{THINKING}'][data-fr-cell='gpqa diamond']")
    assert on.inner_text().replace("\n", "") == "measured here 40.0 (thinking on) · Epoch 38.0"
    # Epoch's number is ranked once, on its own row: beside the thinking row it
    # is the calibration only
    assert on.locator("[data-fr-borrowed]").count() == 1
    assert on.locator("[data-fr-set^='epoch|']").count() == 0
    reps = page.locator("[data-fr-cell='gpqa diamond'] [data-fr-set^='epoch|'], "
                        "[data-fr-cell='gpqa diamond'][data-fr-set^='epoch|']")
    vals = [reps.nth(i).get_attribute("data-fr-v") for i in range(reps.count())]
    assert vals.count("38") + vals.count("0.38") + vals.count("38.0") <= 1
    page.locator("[data-frontier-table]").scroll_into_view_if_needed()
    SCREENS.mkdir(parents=True, exist_ok=True)
    off.screenshot(path=SCREENS / "calibration-thinking-off.png")
    assert page.errors == []


def test_test_a_model_offers_thinking_for_the_shared_suite(live, page):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='shared']").click()
    box = page.locator("[data-shared-opts] [data-think-switch] input")
    box.wait_for()
    assert not box.is_checked()
    assert page.locator("[data-gen-opts]").count() == 0     # no MMLU-Pro subset here
    assert page.errors == []
