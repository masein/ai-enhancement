"""18d, part 4: point 25 in a real browser, at 400 and 1440 px, light and dark —
the header's list keeps each run's name whole, a rented line's description
"rented GPU". The rented runs are 18c.4's invented boxes; no model runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_14_3_browser import steady_shot
from test_18c4_page_browser import rented_world, runs_page

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18d"


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_25_the_headers_list_keeps_each_name_whole(live, page, scheme):
    from service import db
    rented_world()
    db.add("Qwen/Qwen3.5-2B-18d4-a-rather-long-name", "instruct", "frontier", "masein", "",
           tasks=["gpqa_diamond_epoch"], status="running")
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        runs_page(page, live, width, q="18c4")
        page.wait_for_function("rentedRunning() > 0")
        page.locator("[data-runs]").first.click()
        lst = page.locator("#pop-runs [data-runs-list]")
        lst.wait_for()
        lines = lst.locator("[data-run-rented]")
        assert lines.count() >= 2
        for i in range(lines.count()):
            ln = lines.nth(i)
            name = ln.locator(".runname")
            w = name.evaluate("e => [e.getBoundingClientRect().width, e.scrollWidth]")
            assert w[0] > 20 and w[0] + 1 >= w[1], (width, w)   # 0b58df7: 0 px at 400
            assert ln.locator(".runwhat").inner_text() == "rented GPU"
        here = lst.locator(".runline:not([data-run-rented])").first
        w = here.locator(".runname").evaluate(
            "e => [e.getBoundingClientRect().width, e.scrollWidth]")
        assert w[0] > 20 and w[0] + 1 >= w[1], w
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(lst, SCREENS / f"header-list-{width}-{scheme}.png")
        page.keyboard.press("Escape")
    assert page.errors == []
