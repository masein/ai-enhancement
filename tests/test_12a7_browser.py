"""12a.7 on the page: an Everyday reader — a group's questions side by side,
a model's answers — is 70% of the page wide, draggable by its left edge (or
←/→ once focused, a double-click back to 70%), its width remembered in this
browser; under 800px it is the whole screen. Other readers are as they were.
Fixtures only."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_12n1_everyday_browser import everyday, open_group

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a7"
KEY = "bench-reader-width"


def width(page) -> float:
    return page.evaluate("document.querySelector('#reader .reader').getBoundingClientRect().width")


def test_the_everyday_reader_is_wide_draggable_and_remembered(live, page):
    everyday(page, live, width=1400)
    page.evaluate(f"localStorage.removeItem('{KEY}')")
    open_group(page)
    page.wait_for_function("document.querySelector('#reader.open')")
    page.wait_for_timeout(300)                                    # its slide in
    assert abs(width(page) - 0.7 * 1400) <= 2
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / "group-reader-1400-70.png")
    # drag its left edge 140px further left: 80%
    grip = page.locator("#reader [data-reader-grip]")
    b = grip.bounding_box()
    page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    page.mouse.down()
    page.mouse.move(b["x"] + b["width"] / 2 - 140, b["y"] + b["height"] / 2, steps=6)
    page.mouse.up()
    assert abs(width(page) - 0.8 * 1400) <= 3
    kept = float(page.evaluate(f"localStorage.getItem('{KEY}')"))
    assert abs(kept - 80) <= 0.3
    # kept across a reload
    page.reload()
    page.wait_for_function("document.querySelector('#reader.open')")
    page.wait_for_timeout(300)
    assert abs(width(page) - kept / 100 * 1400) <= 2
    # the keyboard: ← wider by 2%, → narrower
    page.locator("#reader [data-reader-grip]").focus()
    page.keyboard.press("ArrowRight")
    assert abs(float(page.evaluate(f"localStorage.getItem('{KEY}')")) - (kept - 2)) <= 0.3
    # a double-click: 70% again, and nothing kept
    page.locator("#reader [data-reader-grip]").dblclick()
    assert page.evaluate(f"localStorage.getItem('{KEY}')") is None
    assert abs(width(page) - 0.7 * 1400) <= 2
    # only an Everyday reader is wide
    assert page.evaluate("rdWideKind('group') && rdWideKind('everyday') && !rdWideKind('log') "
                         "&& !rdWideKind('dataset') && !rdWideKind('bank')")
    assert page.errors == []


def test_under_800px_it_is_the_whole_screen_and_has_no_handle(live, page):
    everyday(page, live, width=760)
    page.evaluate(f"localStorage.setItem('{KEY}', '50')")
    page.reload()
    page.wait_for_selector("[data-everyday-table]")
    open_group(page)
    page.wait_for_function("document.querySelector('#reader.open')")
    page.wait_for_timeout(300)
    assert abs(width(page) - page.evaluate("innerWidth")) <= 1
    assert not page.locator("#reader [data-reader-grip]").is_visible()
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / "group-reader-760.png")
    assert page.errors == []
