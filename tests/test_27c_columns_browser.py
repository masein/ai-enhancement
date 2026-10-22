"""27c, Part 2: Columns ▾ changes a tab's current set, never its default —
each click kept, the columns in one order before and after any change, and
the choice kept across a reload and in the page's link — on every tab with
the picker. Before, each click started again from the default: ticking
IFBench after IFEval unticked IFEval. The world is 21d's (21c's scorecard on
the fixture board); every number in it is invented."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21d_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27c"
TABS = {"latest": "view=latest", "frontier": "view=frontier", "everyday": "view=everyday",
        "standard": "view=standard"}


def go(page, live, hash_, width=1280, scheme="dark"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&" + hash_)
    page.wait_for_selector("[data-lb-card] table[data-model-table] tbody tr.mt-row",
                           state="attached", timeout=30000)


# the columns a view always has, whatever the picker says
FIXED = {"name", "params", "avg", "cavg", "rank", "src"}


def heads(page) -> list[str]:
    return [k for k in page.eval_on_selector_all(
        "[data-lb-card] table[data-model-table] thead tr.mt-hrow th[data-mt-col]",
        "ts => ts.map(t => t.dataset.mtCol)") if k not in FIXED]


def menu(page) -> list[tuple[str, bool]]:
    """the menu's boxes, in its order: (column, ticked)"""
    page.locator("#pill-columns").click()
    page.wait_for_selector("#pop-columns [data-column]")
    return page.eval_on_selector_all("#pop-columns [data-column]",
                                     "bs => bs.map(b => [b.dataset.column, b.checked])")


def click(page, key: str) -> None:
    box = page.locator(f"#pop-columns [data-column='{key}']")
    if not box.is_visible():
        page.locator("#pill-columns").click()
    page.locator(f"#pop-columns [data-column='{key}']").click()
    page.wait_for_timeout(250)


@pytest.mark.parametrize("tab", list(TABS))
def test_each_click_is_kept_and_a_reload_keeps_them_in_order(live, page, tab):
    go(page, live, TABS[tab])
    page.evaluate("localStorage.removeItem('bench-mt-cols'); localStorage.removeItem('bench-lb-shown')")
    go(page, live, TABS[tab])
    boxes = [(k, t) for k, t in menu(page) if k not in FIXED]
    order = [k for k, _ in boxes]
    on = [k for k, ticked in boxes if ticked]
    off = [k for k, ticked in boxes if not ticked]
    assert len(on) >= 2, boxes
    before = heads(page)
    # the default is in the menu's order (27c: one order for both)
    assert [k for k in before if k in order] == [k for k in order if k in on], (before, on)
    page.keyboard.press("Escape")
    if len(off) >= 2:
        # tick two, one after the other; untick one of the default's
        a, b, gone = off[0], off[1], on[0]
        click(page, a)
        assert a in heads(page)
        click(page, b)
        assert a in heads(page) and b in heads(page)       # the first click is kept
        click(page, gone)
        now = heads(page)
        assert a in now and b in now and gone not in now    # and both before it
        want = [k for k in order if k in (set(on) | {a, b}) - {gone}]
    else:
        # every column is the default (Everyday): untick two, then tick one back
        a, b = on[0], on[1]
        click(page, a)
        assert a not in heads(page)
        click(page, b)
        assert a not in heads(page) and b not in heads(page)  # the first click is kept
        click(page, a)
        now = heads(page)
        assert a in now and b not in now
        want = [k for k in order if k in set(on) - {b}]
    assert [k for k in now if k in order] == want, (now, want)
    # a reload, and the page's link alone: the same three changes, the same order
    if tab != "standard":
        assert "cols=" in page.evaluate("location.hash")
    link = page.evaluate("location.hash")
    page.reload()
    page.wait_for_selector("[data-lb-card] table[data-model-table] tbody tr.mt-row",
                           state="attached", timeout=30000)
    assert [k for k in heads(page) if k in order] == want
    if tab != "standard":
        page.evaluate("localStorage.removeItem('bench-mt-cols')")
        page.goto("about:blank")
        page.goto(live["base"] + "/" + link)
        page.wait_for_selector("[data-lb-card] table[data-model-table] tbody tr.mt-row",
                               state="attached", timeout=30000)
        assert [k for k in heads(page) if k in order] == want
    assert page.errors == []


@pytest.mark.parametrize("width", [1280, 400])
def test_latest_after_two_changes_and_a_reload(live, page, width):
    """the reply's screenshot: Latest after two column changes and a reload"""
    go(page, live, TABS["latest"], width=width)
    page.evaluate("localStorage.removeItem('bench-mt-cols')")
    go(page, live, TABS["latest"], width=width)
    # 27c, Part 3: IFEval and IFBench are in Latest's default now
    assert {"ifeval", "ifbench"} <= {k for k, ticked in menu(page) if ticked}
    click(page, "math")
    click(page, "gpqa")
    page.keyboard.press("Escape")
    page.reload()
    page.wait_for_selector("[data-lb-card] table[data-model-table] tbody tr.mt-row",
                           state="attached", timeout=30000)
    shown = heads(page)
    assert "math" in shown and "gpqa" not in shown and "ifeval" in shown
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(300)
    page.locator("[data-lb-card]").screenshot(path=str(SCREENS / f"latest-columns-{width}-dark.png"))
    assert page.errors == []
