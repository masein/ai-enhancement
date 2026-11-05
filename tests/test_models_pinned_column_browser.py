"""The Models table scrolled sideways: the pinned Model column stays above
every scrolling cell in every state — a row hovered, the sorted column, a
cell selected or focused — and a group's heading stays at the left edge of
the table's visible part, as the Model column does. In a real browser,
light and dark: at 1280 px, and at 700 px (a phone on its side, the
narrowest the table is drawn at, where it scrolls furthest); under 640 px
the Models tabs are cards, with no table to scroll. On 27's world (every
model and number invented)."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_27_models_table_browser import board, go  # noqa: F401 — board is the fixture

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "models-pinned"
# the tab and the width: Frontier's table scrolls at both, Latest's (its
# group heading "measured here, same questions · …") at the narrower
CASES = [(tab, width, scheme) for tab, width in (("frontier", 1280), ("frontier", 700),
                                                 ("latest", 700))
         for scheme in ("light", "dark")]

# for each body row in sight: the pinned cells' right part, and whatever the
# browser draws on top at points there — a data cell there is one drawn over
# them; and the pinned cells' own background, which must hide what scrolls
# under them in every state (a row's tint laid on it see-through, as the
# hovered row's was, showed the sorted column's cell under the name)
OVER = """box => {
  const opaque = td => { const m = getComputedStyle(td).backgroundColor.match(/[\\d.]+/g);
    return m && (m.length < 4 || Number(m[3]) === 1); };
  const pinned = el => el && el.closest('td.mt-name, td.mt-rank, tr.mt-group th');
  const out = [];
  // the rows in sight: under the header (pinned too), above the window's bottom
  const b = box.getBoundingClientRect(), head = box.querySelector('thead').getBoundingClientRect();
  const top = Math.max(b.top, head.bottom), bottom = Math.min(b.bottom, innerHeight - 60);
  for (const tr of box.querySelectorAll('tbody tr.mt-row')) {
    const nm = tr.querySelector('td.mt-name');
    const r = nm.getBoundingClientRect();
    if (r.top < top || r.bottom > bottom) continue;
    // a row under the page's own bar (pinned to the window's top) is out of sight
    const left = document.elementFromPoint(r.left + 4, r.top + r.height / 2);
    if (!left || !tr.contains(left)) continue;
    for (const td of tr.querySelectorAll('td.mt-name, td.mt-rank')) {
      if (!opaque(td)) out.push({ row: tr.dataset.mtRow, seeThrough: td.className,
        background: getComputedStyle(td).backgroundColor });
    }
    for (const f of [0.25, 0.5, 0.75]) {
      for (const dx of [2, 12, 40]) {
        const x = r.right - dx, y = r.top + r.height * f;
        if (x <= r.left) continue;
        const hit = document.elementFromPoint(x, y);
        // a tooltip or the page's own overlays float above the table: only
        // the table's own cells are held to it
        if (!hit || !box.contains(hit)) continue;
        if (!tr.contains(hit) || pinned(hit) !== nm) {
          const td = hit && hit.closest('td, th');
          out.push({ row: tr.dataset.mtRow, at: [Math.round(x), Math.round(y)],
            top: td ? (td.className || td.tagName) + ' ' + (td.textContent || '').trim().slice(0, 24)
              : (hit ? hit.tagName : 'nothing') });
          break;
        }
      }
    }
  }
  return out;
}"""
# how many rows the probe looked at (a check that looks at none passes)
SEEN = OVER.replace("out.push(", "void (").replace("return out;", "return seen;").replace(
    "const out = [];", "const out = []; let seen = 0;").replace(
    "for (const f of [0.25, 0.5, 0.75]) {", "seen++; for (const f of [0.25, 0.5, 0.75]) {")

# a group's heading: its label in view, at the left edge of the visible part
GROUP = """box => {
  const b = box.getBoundingClientRect();
  return [...box.querySelectorAll('tbody tr.mt-group th')].map(th => {
    const s = th.querySelector('span').getBoundingClientRect();
    return { left: Math.round(s.left - b.left), right: Math.round(b.right - s.right),
      text: th.textContent.trim().slice(0, 40) };
  });
}"""


def scrolled(page):
    """the table, sorted by GPQA Diamond (▼), scrolled to its far right"""
    box = page.locator("[data-lb-card] .mt-box").first
    head = page.locator("[data-lb-card] table[data-model-table] thead th[data-mt-col='gpqa']")
    for _ in range(2):
        if "▼" in head.inner_text():
            break
        head.click()
        page.wait_for_timeout(150)
    assert "▼" in head.inner_text()
    box.evaluate("b => { b.scrollLeft = b.scrollWidth; }")
    page.wait_for_timeout(150)
    assert box.evaluate("b => b.scrollLeft") > 0
    return box


@pytest.mark.parametrize("tab,width,scheme", CASES)
def test_no_cell_is_drawn_over_the_pinned_column_when_scrolled_sideways(live, page, tab, width,
                                                                         scheme):
    go(page, live, f"view={tab}", width=width, scheme=scheme)
    box = scrolled(page)
    rows = page.locator("[data-lb-card] table[data-model-table] tbody tr.mt-row")
    assert rows.count() >= 3
    # nothing hovered, then each row hovered in turn (its sorted-column cell
    # first: the cell that sorts highlighted, under the pointer), then a cell
    # focused from the keyboard
    assert box.evaluate(SEEN) >= 3
    assert box.evaluate(OVER) == [], "at rest"
    for k in range(rows.count()):
        row = rows.nth(k)
        cell = row.locator("td[data-mt-col], td.tcell").last
        cell.hover(force=True)
        page.wait_for_timeout(30)
        got = box.evaluate(OVER)
        assert got == [], (k, got)
    focusable = rows.first.locator("td.tcell[tabindex]")
    if focusable.count():
        focusable.last.focus()
        assert box.evaluate(OVER) == [], "focused"
    # the group's heading at the left edge of what is in view, whole
    for g in box.evaluate(GROUP):
        assert 0 <= g["left"] <= 24 and g["right"] >= 0, g
    assert page.errors == []
    if scheme == "dark":
        SCREENS.mkdir(parents=True, exist_ok=True)
        rows.nth(1).hover(force=True)
        box.screenshot(path=str(SCREENS / f"{tab}-scrolled-{width}-{scheme}.png"))


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_on_a_phone_the_models_tab_is_cards_with_no_table_to_scroll(live, page, scheme):
    go(page, live, "view=frontier", width=400, scheme=scheme)
    assert page.locator("[data-lb-card] [data-mt-card]").first.is_visible()
    assert not page.locator("[data-lb-card] .mt-box").first.is_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert page.errors == []
