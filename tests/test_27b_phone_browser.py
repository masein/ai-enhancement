"""27b in a real browser: On a phone and the testers' medians in 27's style.
The phone build measured on our testers' phones shows its median — the
row's filled dot, decode tok/s, the phone and its tests under it — and not
the speed typed in by hand, which is on hover; the medians per phone and
model are modelTable's, a row a model on a phone. The world is 26's: nine
made-up tests (made-up ids, testers and replies) and a speed typed in by
hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21d_browser import no_sideways
from test_26_browser import NAME, PHONE
from test_26_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27b"


def go(page, live, rid, wait, width=1280, scheme="dark"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=" + rid)
    page.wait_for_selector(wait, state="attached", timeout=30000)
    page.wait_for_timeout(300)


def test_a_measured_median_wins_the_phone_cell(live, page):
    go(page, live, "on-device", "[data-od-table]")
    label = f"{NAME} (phone build)"
    cell = page.locator(f"td[data-od-fact='{label}|phone']")
    assert cell.get_attribute("data-od-measured") == "1"
    assert cell.locator("[data-mt-cell-mark='here']").count() == 1
    # the phone with the most tests: OnePlus 15, four in its median; the other on hover
    assert cell.locator(".mt-n").inner_text() == "13.0 tok/s"
    assert cell.locator(".mt-sub").inner_text() == "OnePlus 15 · 4 tests · +1 phone"
    tip = cell.get_attribute("data-tip")
    assert "Galaxy S25 Ultra: 9.8 tok/s" in tip
    # typed in by hand: on hover, never the cell's number
    assert "13.5" not in cell.inner_text() and "reported by hand: 13.5 tok/s" in tip
    assert page.errors == []


@pytest.mark.parametrize("width", [1280, 400])
def test_the_medians_per_phone_are_27s_table(live, page, width):
    go(page, live, "testers", "[data-pt-per]", width=width)
    t = page.locator("table[data-pt-per]")
    assert t.get_attribute("data-model-table") == "pt-per"
    names = t.locator("tbody tr.mt-row [data-mt-name]").all_text_contents()
    assert names[0] == f"{NAME} · OnePlus 15"                    # the fastest first
    assert set(names) == {f"{NAME} · OnePlus 15", f"{NAME} · Galaxy S25 Ultra",
                          f"{NAME} · 23078PND5G"}
    row = t.locator("tr[data-pt-per-row='CPH2747|model-004']")
    assert row.locator("[data-pt-per-decode] .mt-n").inner_text() == "13.0"
    assert row.locator("[data-pt-per-n]").inner_text() == "4 of 5"
    # its line: the phone's memory and the tests' dates; it opens the build's page
    assert row.locator("[data-mt-line]").inner_text().endswith("2026-10-17 to 2026-10-20")
    assert row.locator("a[data-mt-name]").get_attribute("href") == "#model=" + PHONE.replace(
        "/", "%2F")
    # an id the names file doesn't name yet is its id, and its line says so
    bare = t.locator("tr[data-pt-per-row='23078PND5G|model-004'] [data-mt-line]")
    assert bare.inner_text().startswith("phone not named yet")
    if width < 640:
        assert page.locator("[data-mt-cards='pt-per'] [data-mt-card]").first.is_visible()
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-report-section='pt-per-phone']").screenshot(
        path=str(SCREENS / f"testers-per-phone-{width}-dark.png"))
