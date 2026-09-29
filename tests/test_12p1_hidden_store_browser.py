"""12p.1 on the page: when Everyday's hidden set is missing or changed, every
page shows it in red with the command that restores it, and the board still
works; Data & sources says where each set is and the backups. Fixtures only:
the live tree's store is written and put back by the test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import everyday as ev
from conftest import go_tab

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12p1"


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name)


@pytest.mark.parametrize("width", [1400, 400])
def test_a_changed_set_is_red_on_every_page(live, page, width):
    import service.app as appmod
    # 12p.2: the store's set (the tests' invented one), changed — and put back after
    p = ev.hidden_path()
    was = p.read_bytes()
    rows = ev._raw_rows(p)
    rows[0] = {**rows[0], "prompt": rows[0]["prompt"] + " (changed)"}
    try:
        p.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows),
                     encoding="utf-8")
        appmod._cache.update(key=None, payload=None, at=0.0)
        page.set_viewport_size({"width": width, "height": 900})
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=models")
        banner = page.locator("#alarms [data-alarm='hidden']")
        banner.wait_for()
        assert banner.inner_text().startswith(
            "Everyday's hidden set is missing or changed: restore it · ")
        assert page.locator("[data-alarm-command='hidden']").inner_text() == ev.RESTORE
        assert banner.get_attribute("role") == "alert"
        # red: the critical colour, not the page's text
        assert page.evaluate("getComputedStyle(document.querySelector('.alarm')).color") != \
            page.evaluate("getComputedStyle(document.body).color")
        # the board works: the Models table is there, and on another place too
        page.wait_for_selector("[data-lb-table]")
        if width == 1400:
            shot(page.locator("#alarms"), "banner-1400.png")
            go_tab(page, "Provenance")
            page.locator("#alarms [data-alarm='hidden']").wait_for()
            card = page.locator("[data-store-card='1']")
            card.wait_for()
            assert card.locator("[data-store='hidden']").get_attribute(
                "data-store-state") == "changed"
            assert "Backups: none yet in " in card.locator("[data-store='backup']").inner_text()
            shot(card, "store-card.png")
        else:
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            shot(page.locator("#alarms"), "banner-400.png")
        assert page.errors == []
    finally:
        p.write_bytes(was)
        appmod._cache.update(key=None, payload=None, at=0.0)


def test_nothing_red_when_the_set_is_as_committed(live, page):
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    assert page.locator("#alarms .alarm").count() == 0
    assert page.errors == []
