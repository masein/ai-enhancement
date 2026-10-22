"""27b in a real browser: the Reports pages in 27's style. The On-device table
on modelTable — nothing cut off at 1280 (every column in view or reached by
scrolling its own box), no "published by" in a score cell, what each keeps of
its base the one line under a score, Memory one number with the size on disk
under it only when it differs, 27's cards at 400 — and every scatter chart's
labels from one placer: no two labels meet, none sits on a point, a point
left bare has its name on hover and under its panel. The world is 25c's
(21c's scorecard, four builds with their files' sizes, every card entered);
every model, question and number in it is invented."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21d_browser import no_sideways
from test_25c_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27b"
E0 = "Edge0-35B-A3B-preview"


def report(page, live, rid="on-device", width=1280, scheme="dark", wait="[data-od-table]"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=" + rid)
    page.wait_for_selector(wait, state="attached", timeout=30000)
    page.wait_for_timeout(300)


def shot(page, sel: str, name: str) -> None:
    SCREENS.mkdir(parents=True, exist_ok=True)
    loc = page.locator(sel).first
    b = loc.evaluate("e => { const r = e.getBoundingClientRect(); return {x: r.left + scrollX, "
                     "y: r.top + scrollY, width: r.width, height: r.height}; }")
    page.screenshot(path=str(SCREENS / name), full_page=True, clip=b)


def test_the_on_device_table_is_27s_and_nothing_is_cut_off(live, page):
    report(page, live)
    t = page.locator("table[data-od-table]")
    assert t.get_attribute("data-model-table") == "od"
    heads = t.locator("thead tr.mt-hrow th[data-mt-col]").evaluate_all(
        "ts => ts.map(t => t.dataset.mtCol)")
    # GPQA Diamond, MMLU-Pro, IFEval, IFBench, Memory, On a phone; the base is in the line
    assert heads == ["name", "gpqa", "mmlupro", "ifeval", "ifbench", "memory", "phone"]
    # every column in view, or reached by scrolling inside its box; the page never scrolls
    geo = page.evaluate("""() => {
      const box = document.querySelector('[data-hkeep="od-on-device"]');
      const b = box.getBoundingClientRect();
      return { sw: box.scrollWidth, cw: box.clientWidth, out: [...box.querySelectorAll(
        'thead tr.mt-hrow th')].map(th => th.getBoundingClientRect())
        .filter(r => r.right - b.left > box.scrollWidth + 1).length,
        cut: [...box.querySelectorAll('thead .mt-h1, tbody .mt-n')].filter(e =>
          e.scrollWidth > e.clientWidth + 1).length }; }""")
    assert geo["out"] == 0 and geo["cut"] == 0, geo
    assert no_sideways(page) and page.errors == []
    # a row: its mark, one name, its line — the base, what it is, its source once, linked
    row = t.locator(f"tr[data-od-row='{E0}']")
    assert row.locator("[data-mt-mark='pub']").count() == 1
    line = row.locator("[data-mt-line]").inner_text()
    assert line.startswith("of Qwen3.6-35B-A3B · int4") and line.count("published by Edge0") == 1
    assert row.locator("[data-od-link]").inner_text() == "README"
    ours = t.locator("tr[data-od-row='step4000']")
    assert ours.locator("[data-mt-mark='here']").count() == 1
    assert ours.locator("[data-mt-line]").inner_text().endswith("measured here")


def test_a_score_cell_says_its_source_by_its_letter_and_keeps_one_line(live, page):
    report(page, live)
    cells = page.locator("table[data-od-table] td[data-od-cell]")
    assert cells.count() > 20
    assert all("published by" not in x for x in cells.all_inner_texts())
    e0 = page.locator(f"td[data-od-cell='{E0}|gpqa']")
    assert e0.locator(".mt-n").inner_text() == "79.8X"
    assert e0.locator(".mt-n.mt-pub").count() == 1 and e0.locator("[data-mt-src='X']").count() == 1
    # what it keeps of its base: the one line under a score that has a base
    rows = page.evaluate("RPS().data['on-device'].rows.map(r => [r.label, r.retention || {}])")
    n = 0
    for label, keep in rows:
        for key, v in keep.items():
            sub = page.locator(f"td[data-od-cell='{label}|{key}'] [data-od-keep]")
            assert sub.inner_text() == f"{100 * v:.1f}% of its base", (label, key)
            assert page.locator(f"td[data-od-cell='{label}|{key}'] .mt-sub").count() == 1
            n += 1
    assert n >= 10
    assert e0.locator("[data-od-keep]").inner_text() == "97.6% of its base"
    # ours keep their ± small after the number
    assert page.locator("td[data-od-cell='step4000|gpqa'] .mt-se").inner_text().startswith("±")


def test_memory_is_one_number_with_the_disk_under_it_when_it_differs(live, page):
    report(page, live)
    e0 = page.locator(f"td[data-od-fact='{E0}|memory']")
    assert e0.locator(".mt-n").inner_text() == "2.9 GiBX"
    assert e0.locator("[data-od-disk]").inner_text() == "19.6 GB on disk"
    # ours: the file is what it needs, one number
    s4 = page.locator("td[data-od-fact='step4000|memory']")
    assert s4.locator(".mt-n").inner_text() == "15.8 GB" and s4.locator(".mt-sub").count() == 0
    # the same twice is one number too
    b9 = page.locator("td[data-od-fact='Ternary Bonsai 2 9B|memory']")
    assert b9.locator(".mt-n").inner_text() == "2 GBP" and b9.locator(".mt-sub").count() == 0
    # On a phone: a published speed in italics with its letter, the phone under it
    ph = page.locator("td[data-od-fact='Ternary Bonsai 2 9B|phone']")
    assert ph.locator(".mt-n.mt-pub").inner_text() == "34 tok/sP"
    assert ph.locator(".mt-sub").inner_text() == "iPhone 18 Pro Max"
    # none: "–", its long note on hover, not in the cell
    none = page.locator(f"td[data-od-fact='{E0}|phone']")
    assert none.inner_text() == "–" and "still maturing" in none.get_attribute("data-tip")


@pytest.mark.parametrize("width", [1280, 400])
def test_at_400_the_table_is_27s_cards(live, page, width):
    report(page, live, width=width, wait="[data-mt-cards='od']")
    cards = page.locator("[data-mt-cards='od'] [data-mt-card]")
    if width < 640:
        assert cards.first.is_visible() and not page.locator("table[data-od-table]").is_visible()
        assert cards.count() >= 10
        assert "Edge0" in page.locator(f"[data-mt-card='{E0}']").inner_text()
    else:
        assert not cards.first.is_visible()
    assert no_sideways(page) and page.errors == []


# every label, every mark and every leader of a chart's panels
LABELS = """(sel) => [...document.querySelectorAll(sel)].map(p => {
  const r = e => { const b = e.getBoundingClientRect();
    return [b.left, b.top, b.right, b.bottom]; };
  return { labels: [...p.querySelectorAll('[data-chart-label]')].map(l => ({
      id: l.dataset.chartFor, r: r(l) })),
    marks: [...p.querySelectorAll('[data-chart-point]')].map(m => ({ id: m.dataset.chartPoint,
      ours: m.tagName === 'circle', best: m.dataset.odBest === '1',
      title: (m.querySelector('title') || {}).textContent || '', r: r(m) })),
    leads: [...p.querySelectorAll('[data-chart-leader]')].map(l => Math.hypot(
      l.x2.baseVal.value - l.x1.baseVal.value, l.y2.baseVal.value - l.y1.baseVal.value)) };
})"""


def meet(a, b, pad=0.0):
    return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad


def far(r, m):
    """a label's box to a mark's centre"""
    cx, cy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
    return ((max(r[0] - cx, 0, cx - r[2])) ** 2 + (max(r[1] - cy, 0, cy - r[3])) ** 2) ** 0.5


def check_panels(panels, memory=False):
    assert panels
    for p in panels:
        ls, ms = p["labels"], p["marks"]
        by = {m["id"]: m for m in ms}
        for i, a in enumerate(ls):
            for b in ls[i + 1:]:
                assert not meet(a["r"], b["r"]), (a, b)
            for m in ms:
                assert not meet(a["r"], m["r"], 0.5), (a, m)
            # 27c: a label is nearer its own point than any other
            own = far(a["r"], by[a["id"]]["r"])
            assert all(own < far(a["r"], m["r"]) for m in ms if m["id"] != a["id"]), a
            # On-device's: ours, and the published on the best-for-their-memory line
            if memory:
                assert by[a["id"]]["ours"] or by[a["id"]]["best"], a
        # no leader longer than 24; every point named on hover
        assert all(x <= 24.5 for x in p["leads"]), p["leads"]
        assert all(m["title"] for m in ms), ms


@pytest.mark.parametrize("width", [1280, 400])
def test_the_charts_labels_never_meet_nor_cover_a_point(live, page, width):
    report(page, live, width=width, wait="[data-od-mem-panel]")
    panels = page.evaluate(LABELS, "[data-od-mem-panel]")
    assert len(panels) == 4
    check_panels(panels, memory=True)
    # fewer labels, by design: not every point is labelled; ours are where they fit
    assert 0 < sum(len(p["labels"]) for p in panels) < sum(len(p["marks"]) for p in panels)
    # one legend above the four: each family once, in its hue
    fams = page.eval_on_selector_all("[data-od-leg-family]", "ls => ls.map(l => l.dataset.odLegFamily)")
    assert fams == ["edge0", "bonsai", "gemma"]
    assert page.locator("[data-od-mem-frontier]").count() == 4
    report(page, live, "step-checkpoints", width=width, wait="[data-sz-panel]")
    sz = page.evaluate(LABELS, "[data-sz-panel]")
    check_panels(sz)
    assert page.errors == []


@pytest.mark.parametrize("width", [1280, 400])
def test_screens(live, page, width):
    report(page, live, width=width)
    page.wait_for_selector("[data-od-mem-panel]")
    shot(page, "[data-report-section='od-table']", f"on-device-table-{width}-dark.png")
    shot(page, "[data-report-section='od-memory']", f"on-device-memory-{width}-dark.png")
    report(page, live, "step-checkpoints", width=width, wait="[data-sz-panel]")
    shot(page, "[data-size-chart]", f"file-size-chart-{width}-dark.png")
    shot(page, "[data-report-section='ours']", f"step-checkpoints-ours-{width}-dark.png")
    shot(page, "[data-report-section='every']", f"step-checkpoints-every-{width}-dark.png")
    assert page.errors == []
