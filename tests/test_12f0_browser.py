"""12f.0 on the page: a full disk turns the status dot red, with its line."""

from __future__ import annotations

from pathlib import Path

import pytest

from service import disk

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12f0"


@pytest.mark.parametrize("level, gb, dot", [("red", 2.1, "red"), ("amber", 7.5, "amber")])
def test_the_status_dot_says_the_disk_is_filling(live, page, monkeypatch, level, gb, dot):
    line = f"The server's disk has {gb:.1f} GB free. Runs may fail to save."
    monkeypatch.setattr(disk, "check", lambda: {"free_gb": gb, "level": level, "line": line})
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    btn = page.locator("#warnings [data-warn-summary]")
    btn.wait_for()
    assert btn.get_attribute("data-dot") == dot
    btn.click()
    row = page.locator("#pop-checks [data-check='disk']")
    assert row.locator(".check-short").inner_text() == line
    assert row.get_attribute("data-severity") == ("error" if level == "red" else "warning")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / f"disk-{level}.png", clip={"x": 700, "y": 0, "width": 700,
                                                             "height": 420})
    assert page.errors == []
