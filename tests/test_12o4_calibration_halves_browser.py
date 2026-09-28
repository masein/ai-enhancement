"""12o.4 on the page: an export of the judge calibration sheet with the
exam's report half in it is listed under Data & sources with the audits —
who, when, which topics. The list is answered here, so no other test's
audit log is touched. Fixtures only."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import go_tab

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12o4"
LOG = {"id": 3, "by": "masein", "n": 12, "at": 1790000000.0,
       "group": "judge calibration export · the exam's report half · Economics, Law"}


@pytest.mark.parametrize("width", [1400, 400])
def test_data_and_sources_lists_the_export(live, page, width):
    page.route("**/api/everyday/audits",
               lambda r: r.fulfill(json={"owner": "masein", "audits": [LOG]}))
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/")
    go_tab(page, "Provenance")
    card = page.locator("[data-audits='1']")
    card.wait_for()
    assert "or the exam’s report half in a judge calibration sheet" in card.inner_text()
    assert card.locator("[data-audit-row='3']").inner_text() == (
        "masein · 2026-09-21 14:13 · judge calibration export · the exam's report half · "
        "Economics, Law · 12 questions")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    SCREENS.mkdir(parents=True, exist_ok=True)
    card.screenshot(path=SCREENS / f"audits-{width}.png")
    assert page.errors == []
