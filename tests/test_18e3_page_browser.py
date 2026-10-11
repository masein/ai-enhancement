"""18e point 7 in a real browser, at 400 and 1440 px, light and dark: the
header's list with the board's long names — a GGUF's "… · GGUF · lookahead
routing + MTP draft" and a rented "Qwen3.6-35B-A3B-UD-Q4_K_XL-lookahead-norm"
— stays inside its popover, each long name cut with an ellipsis and whole in
its hover, and a rented line keeps "rented GPU". Invented runs; no model runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_14_3_browser import steady_shot
from test_17i_runs_browser import post, step
from test_18c4_page_browser import HLE, runs_page

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18e"
RENTED = "served/Qwen3.6-35B-A3B-UD-Q4_K_XL-lookahead-norm"
GGUF = "served/Qwen3.6-35B-A3B-UD-Q4_K_XL-lookahead-norm-18e"
SETUP = "lookahead routing + MTP draft"


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_7_long_names_stay_inside_the_headers_list(live, page, scheme):
    from service import db
    post([step(RENTED, "L1", "L1-1", tasks=(HLE,),
               line="Humanity's Last Exam 12 of 240 · 30 s an answer · 2.0 h left")])
    db.add(GGUF, "instruct", "gguf", "masein", f"setup: {SETUP} · the GGUF's own file",
           status="running")
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        runs_page(page, live, width, q="lookahead")
        page.wait_for_function("rentedRunning() > 0")
        page.locator("[data-runs]").first.click()
        lst = page.locator("#pop-runs [data-runs-list]")
        lst.wait_for()
        # nothing scrolls sideways: the list, the popover, the page
        box = lst.evaluate("e => [e.scrollWidth, e.clientWidth, e.getBoundingClientRect().right]")
        assert box[0] <= box[1] + 1, (width, box)                 # 518e4a1: 765 in 374
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        names = {}
        for sel in ("[data-run-rented]", "[data-run-line]:not([data-run-rented])"):
            ln = lst.locator(sel).first
            assert ln.evaluate("e => e.getBoundingClientRect().right") <= box[2] + 1, (width, sel)
            name = ln.locator(".runname")
            got = name.evaluate("""e => ({w: e.getBoundingClientRect().width, full: e.scrollWidth,
                right: e.getBoundingClientRect().right, cut: getComputedStyle(e).textOverflow,
                title: e.title, text: e.textContent})""")
            assert got["w"] > 40 and got["right"] <= box[2] + 1, (width, got)
            if got["full"] > got["w"] + 1:
                assert got["cut"] == "ellipsis", (width, got)        # 518e4a1: cut at the edge
            assert got["text"] in got["title"], got                  # whole in its hover
            names[sel] = got
        # the rented line's description whole: "rented GPU" (518e4a1: 0 px)
        what = lst.locator("[data-run-rented] .runwhat").first
        w = what.evaluate("e => [e.getBoundingClientRect().width, e.scrollWidth]")
        assert what.inner_text() == "rented GPU" and w[0] > 20 and w[0] + 1 >= w[1], (width, w)
        assert names["[data-run-rented]"]["text"] == RENTED.split("/")[-1]
        assert names["[data-run-line]:not([data-run-rented])"]["text"].endswith(
            f" · GGUF · {SETUP}")
        if width == 400:
            assert any(n["full"] > n["w"] + 1 for n in names.values())   # long enough to cut
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(lst, SCREENS / f"header-list-long-{width}-{scheme}.png")
        page.keyboard.press("Escape")
    assert page.errors == []
