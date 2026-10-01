"""12z A on the page: the setups table's thinking row (A2), DeviceMark's own
row beside our hf run (A5), the rank in words (A9) and the Runs list past its
newest 100 (A6) — at 1400 and 375 px. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from service import config
from test_12q_devicemark_board import HF, _row
from test_12q_devicemark_model_page_browser import MTP, PLAIN, model_page, runs  # noqa: F401
from test_12x_models_devicemark_browser import _dm_results

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12z"
QWEN = "Qwen/Qwen3.5-4B"
FITS = """el => [...el.querySelectorAll('td, th')].every(c => c.scrollWidth <= c.clientWidth + 1)
  && el.closest('.lb-wrap').scrollWidth >= el.scrollWidth"""


def shot(loc, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    loc.screenshot(path=SCREENS / name)


@pytest.mark.parametrize("width", [1400, 375])
def test_the_setups_table_has_a_row_per_setup_and_mode_and_no_cell_spills(
        live, page, runs, width):  # noqa: F811
    model_page(page, live, PLAIN, width=width)
    table = page.locator("[data-served-setups]")
    table.wait_for()
    names = table.locator("tbody tr td:first-child").all_inner_texts()
    # each once: the thinking one its own row, named so
    assert sorted(names) == sorted(["DM phone build", "DM phone build, MTP",
                                    "DM phone build, MTP · thinking"])
    assert len(set(names)) == len(names)
    off = table.locator(f"[data-served-dm='{MTP}|off']").inner_text()
    on = table.locator(f"[data-served-dm='{MTP}|on']").inner_text()
    assert off != on and "·" not in off and "·" not in on
    # nothing runs over its neighbour: every cell holds its text, the table scrolls
    assert table.locator("table").evaluate(FITS)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(table, f"setups-{width}.png")
    assert page.errors == []


@pytest.fixture(scope="module")
def qwen(live):
    """our hf run of Qwen3.5-4B, beside DeviceMark's own row: another mode"""
    import service.app as appmod
    _dm_results(QWEN, False)
    _row(config.OUT_DIR, QWEN, {"ifeval": 0.3, "mmlu_pro": 0.2, "math": 0.3}, HF)
    f = config.OUT_DIR / QWEN.replace("/", "__") / "devicemark.json"
    row = json.loads(f.read_text())
    row["median_tokens"] = 400                                      # theirs 3,917
    f.write_text(json.dumps(row))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    for p in (config.OUT_DIR / QWEN.replace("/", "__")).glob("devicemark*"):
        p.unlink()
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width", [1400, 375])
def test_our_hf_runs_card_shows_their_row_and_the_modes_note(live, page, qwen, width):
    model_page(page, live, QWEN, width=width)
    page.locator("[data-kind-tile='dm']").click()
    card = page.locator("[data-dm-card-theirs='off']")
    card.wait_for()
    text = card.inner_text()
    assert text.startswith("DeviceMark’s own row (int8, scored on a Mac; speed on iPhone 17 Pro): ")
    assert text.endswith(" · modes differ · not a calibration point")
    assert card.get_attribute("title").startswith("the modes differ: ours answered")
    # the tile's line says their number too
    assert "beside DeviceMark’s own row (" in page.locator(
        "[data-kind-tile='dm']").inner_text()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-dm-card='Qwen/Qwen3.5-4B|off']"), f"theirs-{width}.png")
    assert page.errors == []


def _runs(ids):
    return [{"id": i, "hf_id": f"org/m{i}", "kind": "base", "suite": "quick", "status": "done",
             "submitter": "masein", "note": "", "progress": "done", "error": "", "part": "",
             "created_at": 1.7e9 + i, "started_at": None, "finished_at": None, "gpu_seconds": 1.0}
            for i in ids]


@pytest.mark.parametrize("width", [1400, 375])
def test_the_runs_list_pages_back_past_its_newest_100(live, page, width):
    asked = []

    def handle(route):
        url = route.request.url
        asked.append(url)
        if "/api/submissions/count" in url:
            return route.fulfill(json={"total": 166})
        body = _runs(range(66, 0, -1)) if "before=67" in url else _runs(range(166, 66, -1))
        return route.fulfill(json=body)
    page.route("**/api/submissions**", handle)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("[data-queue-table]")
    note = page.locator("[data-queue-count]")
    page.wait_for_function("document.querySelector('[data-queue-count]').textContent.includes('166')")
    assert note.inner_text() == "100 of 166 · the latest 100"
    older = page.locator("[data-queue-older]")
    assert older.inner_text() == "Show 66 older runs"
    older.click()
    page.wait_for_function(
        "document.querySelector('[data-queue-count]').textContent === '166 of 166'")
    assert any("before=67" in u for u in asked)
    assert page.locator("[data-queue-older]").count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert page.errors == []
