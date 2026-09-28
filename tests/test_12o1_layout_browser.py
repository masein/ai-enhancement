"""12o.1 on the page: every column's width and place. A header's right edge
widens or narrows it (a pointer, ←/→, a double-click to fit); a header drags
to another place in its group, a group's header moves the group; where the
columns are models (Compare, the Everyday table) any model goes anywhere; the
header's ⋯ says the same for a keyboard or a finger, and Reset layout. #,
Model and Avg stay at the left, and the rank never moves with a column. Kept
per table in this browser, and in a saved view. Under 600px, no handles.

Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12o1"
GOOD, SKEWED, SMALL = "fx/good-750m", "fx/skewed-360m", "fx/below-135m-it"
ALL = "models:standard:all"
DEFAULT = ["mmlu", "hellaswag", "winogrande", "piqa", "arc_challenge", "arc_easy"]


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def go(page, live, hash_, sel, width=1280, scheme="light", fresh=()):
    """the page at this address, with these tables' layouts forgotten first"""
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    if fresh:
        page.evaluate("ks => ks.forEach(k => localStorage.removeItem('bench-layout-' + k))",
                      list(fresh))
        page.reload()
    page.wait_for_selector(sel)


def kept(page, key):
    return json.loads(page.evaluate(f"localStorage.getItem('bench-layout-{key}')") or "null")


def heads(page, table="[data-lb-table]"):
    return page.locator(f"{table} thead tr:last-child th[data-lkey]").evaluate_all(
        "xs => xs.map(x => x.dataset.lkey)")


def all_heads(page, table="[data-lb-table]"):
    return page.locator(f"{table} thead tr:last-child th").evaluate_all(
        "xs => xs.map(x => x.dataset.lcol || '')")


def group_heads(page):
    return page.locator("[data-lb-table] thead tr.grp th").evaluate_all(
        "xs => xs.map(x => x.textContent)")


def rows(page):
    return page.locator("[data-lb-table] tbody tr[data-lb-row]").evaluate_all(
        "xs => xs.map(x => [x.dataset.lbRow, x.children[0].textContent])")


def cell(page, model, col, table="[data-lb-table]"):
    """a row's cell under a column, wherever the column is now"""
    return page.evaluate("""([t, m, c]) => {
      const tb = document.querySelector(t);
      const i = [...tb.querySelector('thead tr:last-child').children]
        .findIndex(th => th.dataset.lcol === c);
      return tb.querySelector(`tbody tr[data-lb-row="${m}"]`).children[i].textContent; }""",
                         [table, model, col])


def menu(page, table, col, what):
    """the header's ⋯, and one of its items"""
    page.locator(f"{table} thead th [data-col-more='{col}']").click()
    item = "[data-layout-reset]" if what == "reset" else f"[data-col-move='{what}']"
    page.locator(f"[id='pop-col-{col}'] {item}").click()


def drag(page, src, dst, after):
    b = dst.bounding_box()
    src.drag_to(dst, target_position={"x": b["width"] * (0.7 if after else 0.3),
                                      "y": b["height"] / 2})


def test_a_column_moves_in_its_group_and_a_group_as_a_block_never_a_rank(live, page):
    T = "[data-lb-table]"
    go(page, live, "tab=models", T, fresh=[ALL])
    assert heads(page) == DEFAULT
    fixed, ranks = all_heads(page)[:4], rows(page)
    assert fixed == ["rank", "name", "params", "avg"]
    before = {c: cell(page, GOOD, c) for c in DEFAULT}
    # the menu: to the start of its group, then right — never out of it
    menu(page, T, "piqa", "start")
    page.wait_for_function("document.querySelector('[data-lb-table] thead th[data-lkey=piqa]')"
                           ".previousElementSibling.dataset.lkey === 'mmlu'")
    assert heads(page) == ["mmlu", "piqa", "hellaswag", "winogrande", "arc_challenge", "arc_easy"]
    menu(page, T, "piqa", "right")
    page.wait_for_function("document.querySelector('[data-lb-table] thead th[data-lkey=piqa]')"
                           ".previousElementSibling.dataset.lkey === 'hellaswag'")
    menu(page, T, "arc_easy", "left")
    page.wait_for_function("document.querySelector('[data-lb-table] thead th[data-lkey=arc_easy]')"
                           ".nextElementSibling.dataset.lkey === 'arc_challenge'")
    assert heads(page) == ["mmlu", "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"]
    # a drag: winogrande before hellaswag
    drag(page, page.locator(f"{T} th[data-lkey='winogrande']"),
         page.locator(f"{T} th[data-lkey='hellaswag']"), after=False)
    page.wait_for_function("document.querySelector('[data-lb-table] thead th[data-lkey=winogrande]')"
                           ".previousElementSibling.dataset.lkey === 'mmlu'")
    # …and never out of its group: MMLU dropped on ARC-C stays where it was
    drag(page, page.locator(f"{T} th[data-lkey='mmlu']"),
         page.locator(f"{T} th[data-lkey='arc_challenge']"), after=True)
    page.wait_for_timeout(300)
    assert heads(page) == ["mmlu", "winogrande", "hellaswag", "piqa", "arc_easy", "arc_challenge"]
    # a group, as a block: Reasoning before Knowledge
    drag(page, page.locator(f"{T} th[data-lgroup-head='Reasoning']"),
         page.locator(f"{T} th[data-lgroup-head='Knowledge']"), after=False)
    page.wait_for_function("document.querySelector('[data-lb-table] thead tr.grp th:nth-child(2)')"
                           ".textContent === 'Reasoning'")
    moved = ["arc_easy", "arc_challenge", "mmlu", "winogrande", "hellaswag", "piqa"]
    assert heads(page) == moved
    assert group_heads(page) == ["", "Reasoning", "Knowledge", "Commonsense"]
    # #, Model, Params and Avg where they were; the rows, their ranks, and each
    # number under its own column
    assert all_heads(page)[:4] == fixed and rows(page) == ranks
    assert {c: cell(page, GOOD, c) for c in DEFAULT} == before
    assert kept(page, ALL)["order"] == moved
    shot(page.locator("[data-lb-card]"), "models-moved-1280-light.png")
    # kept in this browser
    page.reload()
    page.wait_for_selector(T)
    assert heads(page) == moved and rows(page) == ranks
    # Reset layout, in any header's ⋯
    menu(page, T, "mmlu", "reset")
    page.wait_for_function("document.querySelector('[data-lb-table] thead th[data-lkey]')"
                           ".dataset.lkey === 'mmlu'")
    assert heads(page) == DEFAULT and kept(page, ALL) is None
    # a fixed column's ⋯ has no moves, only the reset
    page.locator(f"{T} thead th [data-col-more='avg']").click()
    page.wait_for_selector("[id='pop-col-avg'] [data-layout-reset]")
    assert page.locator("[id='pop-col-avg'] [data-col-move]").count() == 0
    assert page.errors == []


def test_any_column_resizes_kept_across_a_reload_and_fits_on_a_double_click(live, page):
    T = "[data-lb-table]"
    go(page, live, "tab=models", T, fresh=[ALL])
    th = page.locator(f"{T} thead th[data-lkey='hellaswag']")
    w0 = th.bounding_box()["width"]
    grip = th.locator("[data-col-grip='hellaswag']")
    b = grip.bounding_box()
    page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    page.mouse.down()
    page.mouse.move(b["x"] + 90, b["y"] + b["height"] / 2, steps=6)
    page.mouse.up()
    w1 = th.bounding_box()["width"]
    assert w1 > w0 + 60
    assert abs(kept(page, ALL)["w"]["hellaswag"] - w1) <= 2
    # every cell of the column takes it
    body = page.locator(f"{T} tbody tr[data-lb-row] td[data-lw]").evaluate_all(
        "xs => [...new Set(xs.map(x => x.dataset.lw))]")
    assert body == [str(kept(page, ALL)["w"]["hellaswag"])]
    page.reload()
    page.wait_for_selector(f"{T} thead th[data-lkey='hellaswag'][data-lw]")
    assert abs(page.evaluate("document.querySelector('[data-lb-table] thead th[data-lkey=hellaswag]')"
                             ".getBoundingClientRect().width") - w1) <= 2
    # ←/→ once focused, 16px a press
    w = kept(page, ALL)["w"]["hellaswag"]
    page.locator(f"{T} thead [data-col-grip='hellaswag']").focus()
    page.keyboard.press("ArrowRight")
    assert kept(page, ALL)["w"]["hellaswag"] == w + 16
    # a double-click fits the widest content shown: the header's name, uncut
    page.locator(f"{T} thead [data-col-grip='arc_challenge']").dblclick()
    fit = kept(page, ALL)["w"]["arc_challenge"]
    name = page.locator(f"{T} thead th[data-lkey='arc_challenge'] .hname")
    assert name.evaluate("x => x.scrollWidth <= x.clientWidth + 1")
    assert fit < 200
    # the sort did not change on any of it
    assert page.locator(f"{T} thead th[aria-sort='descending']").get_attribute("data-col") == "avg"
    assert page.errors == []


def test_everydays_model_columns_move_anywhere_and_its_groups_on_models(live, page):
    T = "[data-everyday-table]"
    go(page, live, "tab=benchmarks&sub=everyday", T, fresh=["everyday-page"])
    assert heads(page, T) == [GOOD, SKEWED]
    menu(page, T, SKEWED, "start")
    page.wait_for_function(f"document.querySelector('{T} thead th[data-lkey]').dataset.lkey "
                           f"=== {json.dumps(SKEWED)}")
    # the counts move with their model
    assert page.locator(f"{T} tbody tr").first.locator("[data-evd-cell]").first.get_attribute(
        "data-evd-cell").startswith(SKEWED + "|")
    drag(page, page.locator(f"{T} th[data-lkey='{SKEWED}']"),
         page.locator(f"{T} th[data-lkey='{GOOD}']"), after=True)
    page.wait_for_function(f"document.querySelector('{T} thead th[data-lkey]').dataset.lkey "
                           f"=== {json.dumps(GOOD)}")
    menu(page, T, GOOD, "right")
    page.wait_for_function(f"document.querySelector('{T} thead th[data-lkey]').dataset.lkey "
                           f"=== {json.dumps(SKEWED)}")
    page.reload()
    page.wait_for_selector(T)
    assert heads(page, T) == [SKEWED, GOOD]
    shot(page.locator("[data-everyday-results]"), "everyday-moved-1280-light.png")
    # Models ▸ Everyday: its group columns anywhere
    E = "[data-lb-everyday]"
    go(page, live, "tab=models&view=everyday", E, fresh=["models:everyday"])
    groups = heads(page, E)
    last = groups[-1]
    menu(page, E, last, "start")
    page.wait_for_function(f"document.querySelector('{E} thead th[data-lkey]').dataset.lkey "
                           f"=== {json.dumps(last)}")
    assert heads(page, E) == [last, *groups[:-1]]
    assert all_heads(page, E)[0] == "name" and all_heads(page, E)[len(groups) + 1] == "total"
    assert page.errors == []


def test_compares_model_columns_move_and_the_address_keeps_their_order(live, page):
    T = "[data-cmp-table]"
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, SKEWED, SMALL)}", "[data-compare='3']",
       fresh=["compare"])
    assert heads(page, T) == [GOOD, SKEWED, SMALL]
    menu(page, T, SMALL, "start")
    page.wait_for_function(f"location.hash.includes('m={ids(SMALL, GOOD, SKEWED)}')")
    page.wait_for_selector(f"{T} thead th[data-lkey='{SMALL}']:nth-child(2)")
    assert heads(page, T) == [SMALL, GOOD, SKEWED]
    drag(page, page.locator(f"{T} th[data-lkey='{SMALL}']"),
         page.locator(f"{T} th[data-lkey='{SKEWED}']"), after=True)
    page.wait_for_function(f"location.hash.includes('m={ids(GOOD, SKEWED, SMALL)}')")
    # the Benchmark column stays; its width is its own
    page.locator(f"{T} thead th [data-col-more='bench']").click()
    page.wait_for_selector("[id='pop-col-bench'] [data-layout-reset]")
    assert page.locator("[id='pop-col-bench'] [data-col-move]").count() == 0
    page.keyboard.press("Escape")
    page.locator(f"{T} thead [data-col-grip='bench']").focus()
    page.keyboard.press("Shift+ArrowRight")
    assert kept(page, "compare")["w"]["bench"] > 0
    assert page.errors == []


def test_a_saved_view_opens_with_its_widths_and_order_and_reset_restores(live, page):
    T = "[data-lb-table]"
    key = "models:standard:all"
    go(page, live, f"tab=models&models={ids(GOOD, SKEWED)}", T, fresh=[key])
    menu(page, T, "arc_easy", "start")
    page.wait_for_function("document.querySelector('[data-lb-table] thead th[data-lkey=arc_easy]')"
                           ".nextElementSibling.dataset.lkey === 'arc_challenge'")
    page.locator(f"{T} thead [data-col-grip='mmlu']").focus()
    page.keyboard.press("Shift+ArrowRight")
    saved = kept(page, key)
    set_name(page, "masein")
    page.locator("[data-save-view]").click()
    page.fill("[data-view-name]", "Wide MMLU")
    page.locator("[data-action='view-save']").click()
    page.wait_for_selector("[data-toast]")
    spec = page.evaluate("fetch('api/views').then(r => r.json())")["views"][-1]["spec"]
    assert spec["layout"] == {key: saved}
    # forgotten here, then the view: as it was saved
    menu(page, T, "mmlu", "reset")
    page.wait_for_function("document.querySelector('[data-lb-table] thead th[data-lkey]')"
                           ".dataset.lkey === 'mmlu'")
    assert kept(page, key) is None
    page.locator("[data-saved-view]", has_text="Wide MMLU").click()
    page.wait_for_selector(f"{T} thead th[data-lkey='mmlu'][data-lw]")
    assert kept(page, key) == saved
    assert heads(page)[-2:] == ["arc_easy", "arc_challenge"]
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_on_a_phone_no_handles_and_names_wrap(live, page, scheme):
    go(page, live, "tab=models", "[data-lb-table]", width=400, scheme=scheme)
    for t in ("[data-lb-table]",):
        assert page.locator(f"{t} .colgrip").count() > 0
        assert page.locator(f"{t} .colgrip:visible, {t} .col-more:visible").count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page, f"models-400-{scheme}.png", full_page=True)
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, SKEWED)}", "[data-compare='2']",
       width=400, scheme=scheme)
    assert page.locator("[data-cmp-table] .colgrip:visible").count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    go(page, live, "tab=benchmarks&sub=everyday", "[data-everyday-table]", width=400,
       scheme=scheme)
    assert page.locator("[data-everyday-table] .colgrip:visible").count() == 0
    shot(page.locator("[data-everyday-results]"), f"everyday-400-{scheme}.png")
    go(page, live, "tab=models", "[data-lb-table]", width=1280, scheme=scheme)
    shot(page.locator("[data-lb-card]"), f"models-1280-{scheme}.png")
    assert page.errors == []
