"""Models tables that scroll sideways (masein's bug on Standard ▸ Knowledge):
the "scroll →" hint sat on the last column's header, so a click sorted by a
column nobody could see, its hover showed that column's tooltip half off the
screen, it never scrolled, and it hid that column's numbers. Now it is a
button above the table, over no column, that scrolls about a screen; "←
scroll" comes once scrolled, each goes when there is nothing more that way,
and neither shows when the table fits. The fade takes no clicks and is a
sliver; a column's tooltip near the right edge opens to the left; a sparse
model's "908M act" is a line under its "2.3B", not beside the name; and the
"Not tested on this" line stays in view while the table scrolls."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_live_check_11e_browser import LB, served

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "scroll-hint"
BOX = "[data-hfade='lb']"
RIGHT, LEFT = f"{BOX} [data-scroll='right']", f"{BOX} [data-scroll='left']"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def sparse(body):
    """row 1 a sparse model, as on masein's board: 2.3B, 908M of it active"""
    top = max((m for m in body["models"] if m.get("avg") is not None), key=lambda m: m["avg"])
    top["name"] = "qwen35-delta-moe-7d560104-step945-v2"
    top.setdefault("archinfo", {}).update(active_params=908_000_000, experts=64,
                                           experts_per_tok=8)
    top["params"] = 2_300_000_000


def knowledge(page, live, width=1024):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&chip=knowledge")
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")
    page.mouse.move(2, 2)


def scroll_left(page):
    return page.evaluate("document.querySelector('[data-hkeep=\"lb\"]').scrollLeft")


def test_the_hint_is_a_button_over_no_column_that_scrolls_and_leaves_the_sort(live, page):
    knowledge(page, live)
    assert page.locator(BOX).get_attribute("data-more") == "1"
    assert page.locator(RIGHT).is_visible() and not page.locator(LEFT).is_visible()
    # above the table, and over no header cell
    geo = page.evaluate(f"""() => {{
      const b = document.querySelector("{RIGHT}").getBoundingClientRect();
      const t = document.querySelector('{LB}').getBoundingClientRect();
      const hit = [...document.querySelectorAll('{LB} thead th')].filter(th => {{
        const r = th.getBoundingClientRect();
        return r.left < b.right && r.right > b.left && r.top < b.bottom && r.bottom > b.top; }});
      return {{ below: b.bottom <= t.top + 0.5, over: hit.length,
                at: document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2)
                  .closest('[data-scroll]') !== null }}; }}""")
    assert geo == {"below": True, "over": 0, "at": True}, geo
    shot(page.locator("[data-lb-card]"), "knowledge-1024-at-rest.png")
    hash_before = page.evaluate("location.hash")
    sorted_before = page.locator(f"{LB} thead th .dir").evaluate_all("xs => xs.map(x => x.closest('th').dataset.col)")
    rows_before = page.locator(f"{LB} tbody tr[data-lb-row]").evaluate_all("xs => xs.map(x => x.dataset.lbRow)")
    width = page.evaluate("document.querySelector('[data-hkeep=\"lb\"]').clientWidth")
    page.locator(RIGHT).click()
    page.wait_for_function("document.querySelector('[data-hkeep=\"lb\"]').scrollLeft > 0")
    page.wait_for_timeout(600)                              # a smooth scroll settles
    moved = scroll_left(page)
    end = page.evaluate("(s => s.scrollWidth - s.clientWidth)(document.querySelector('[data-hkeep=\"lb\"]'))")
    # about a screen, less the pinned # and Model — or to the end if that is nearer
    assert moved >= min(end, width * 0.4) - 1, (moved, end, width)
    # the sort and the rows are what they were
    assert page.evaluate("location.hash") == hash_before
    assert page.locator(f"{LB} thead th .dir").evaluate_all(
        "xs => xs.map(x => x.closest('th').dataset.col)") == sorted_before
    assert page.locator(f"{LB} tbody tr[data-lb-row]").evaluate_all(
        "xs => xs.map(x => x.dataset.lbRow)") == rows_before
    # scrolled: "← scroll" on the left; at the end, "scroll →" goes
    assert page.locator(LEFT).is_visible()
    if moved >= end - 2:
        assert not page.locator(RIGHT).is_visible()
    # the Not tested line spans every column: its words stay in view
    tog = page.locator("[data-not-tested-toggle]").bounding_box()
    sc = page.locator("[data-hkeep='lb']").bounding_box()
    assert tog["x"] >= sc["x"], (tog, sc)
    shot(page.locator("[data-lb-card]"), "knowledge-1024-scrolled.png")
    page.locator(LEFT).click()
    page.wait_for_function("document.querySelector('[data-hkeep=\"lb\"]').scrollLeft === 0")
    page.wait_for_function(f"!document.querySelector(\"{LEFT}\").checkVisibility({{visibilityProperty: true}})")
    assert page.locator(RIGHT).is_visible()
    assert page.errors == []


def test_no_buttons_when_the_table_fits_and_the_fade_takes_no_clicks(live, page):
    knowledge(page, live, width=1600)
    assert page.locator(BOX).get_attribute("data-wide") == "0"
    assert not page.locator(RIGHT).is_visible() and not page.locator(LEFT).is_visible()
    assert page.locator(f"{BOX} > .hnav").evaluate("n => getComputedStyle(n).display") == "none"
    knowledge(page, live)
    fade = page.locator(BOX).evaluate(
        "b => { const s = getComputedStyle(b, '::after'); return [s.pointerEvents, parseFloat(s.width)]; }")
    assert fade[0] == "none" and fade[1] <= 8, fade
    # the last column wholly in view: its numbers are not under the fade
    box_right = page.evaluate("document.querySelector('[data-hkeep=\"lb\"]').getBoundingClientRect().right")
    tds = page.locator(f"{LB} tbody tr[data-lb-row] td.tcell")
    ends = tds.evaluate_all(f"""xs => xs.map(td => {{ const r = document.createRange();
        r.selectNodeContents(td); const b = r.getBoundingClientRect(), c = td.getBoundingClientRect();
        return [c.right, b.right]; }}).filter(([c]) => c <= {box_right})""")
    assert ends and max(b for _, b in ends) <= box_right - 8, (ends, box_right)
    assert page.errors == []


def test_a_columns_tooltip_near_the_right_edge_opens_to_the_left(live, page):
    knowledge(page, live)
    box = page.locator(BOX).bounding_box()
    heads = page.locator(f"{LB} thead tr:last-child th[data-tip]")
    first = heads.first.bounding_box()
    y = first["y"] + first["height"] / 2
    # the pointer along the header row, every 8px to the table's right edge:
    # each column's tooltip stays inside the table's box and on the screen
    seen, x = 0, first["x"] + 4
    while x < box["x"] + box["width"] - 2:
        page.mouse.move(x, y)
        tip = page.evaluate("""() => { const t = document.getElementById('tip');
          const r = t.getBoundingClientRect();
          return [getComputedStyle(t).opacity, r.left, r.right, innerWidth]; }""")
        if tip[0] == "1":
            seen += 1
            assert tip[1] >= 0 and tip[2] <= box["x"] + box["width"] + 0.5, (x, tip, box)
        x += 8
    assert seen > 20, seen
    page.mouse.move(box["x"] + box["width"] - 6, y)
    shot(page, "tooltip-right-edge-1024.png")
    assert page.errors == []


def test_a_sparse_models_active_count_is_a_line_under_its_total(live, page):
    served(page, sparse)
    knowledge(page, live)
    row = page.locator(f"{LB} tbody tr[data-lb-row]").first
    cell = row.locator("td:has(.act)")
    assert cell.inner_text() == "2.3B\n908M act"
    geo = row.evaluate("""tr => {
      const name = tr.querySelector('td.model').getBoundingClientRect();
      const td = tr.querySelector('td:has(.act)'), act = td.querySelector('.act');
      const r = document.createRange(); r.selectNodeContents(td);
      const text = r.getBoundingClientRect(), a = act.getBoundingClientRect();
      const tot = document.createRange(); tot.selectNodeContents(td.firstChild);
      return { gap: text.left - name.right, under: a.top >= tot.getBoundingClientRect().bottom - 1,
               title: td.title }; }""")
    assert geo["gap"] >= 6 and geo["under"], geo
    assert "908M active" in geo["title"]
    shot(row, "sparse-params-1024.png")
    assert page.errors == []
