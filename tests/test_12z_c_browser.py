"""12z C on the page, on the board's own fixtures: the Answers tab's switch
and chips (C1), the Everyday tile's badge and its ran-out count (C2), Home's
Everyday tile (C8) and the checks popover (C9) — at 1400 and 375 px. The
rest of C is tested beside its fixtures: the setups table and DeviceMark's
answers (test_12q_devicemark_model_page_browser), the On-device chart
(test_12q_devicemark_chart_browser), Models ▸ DeviceMark
(test_12x_models_devicemark_browser), the GGUF panels' labels and Running now
(test_12n1_browser), and the GGUF group (test_gguf_12f3_browser)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conftest import set_name, pick_answers

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12z"
MODEL = "fx/good-750m"
WIDTHS = [1400, 375]


def shot(loc, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    loc.screenshot(path=SCREENS / name)


def clip(page, sel, name):
    """a picture of the page where `sel` is: a poll may draw the element again
    while a screenshot of the element itself waits for it to hold still"""
    box = page.evaluate("""s => { const r = document.querySelector(s).getBoundingClientRect();
      return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height }; }""", sel)
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, clip=box, full_page=True)


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


# gaps between boxes: the next one's left minus this one's right, along a row
GAPS = """sel => { const xs = [...document.querySelectorAll(sel)].map(e => e.getBoundingClientRect());
  const out = []; for (let i = 1; i < xs.length; i++)
    if (Math.abs(xs[i].top - xs[i - 1].top) < 2) out.push(xs[i].left - xs[i - 1].right);
  return out; }"""
# the space between two boxes, top to bottom
VGAP = """([a, b]) => document.querySelector(b).getBoundingClientRect().top
  - document.querySelector(a).getBoundingClientRect().bottom"""


@pytest.mark.parametrize("width", WIDTHS)
def test_the_answers_tab_has_a_switch_then_spaced_chips(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    page.locator("[data-mtab='answers']").click()
    seg = page.locator("[data-answers-kinds]")
    seg.wait_for()
    # the source is one control, not more chips: a group of one picker (16c:
    # there may be twenty sources, so not a segmented row)
    assert seg.get_attribute("role") == "group"
    assert seg.get_attribute("aria-label") == "which answers"
    assert seg.locator("button.sel[data-answers-pick]").count() == 1
    assert seg.locator("button").count() == 1 and seg.locator(".chip-btn").count() == 0
    pick_answers(page, "everyday")
    page.wait_for_selector("[data-answers-groups] [data-answers-group]")
    # 8px between chips, 12px between the rows and under the description
    gaps = page.evaluate(GAPS, "[data-answers-groups] [data-answers-group]")
    assert gaps and all(g >= 7.5 for g in gaps), gaps
    assert page.evaluate(VGAP, ["[data-model-answers] > .sub", "[data-answers-kinds]"]) >= 11.5
    assert page.evaluate(VGAP, ["[data-answers-kinds]", "[data-answers-groups]"]) >= 11.5
    # the two controls don't look alike: the picker is one bordered box
    looks = page.evaluate("""() => {
      const s = getComputedStyle(document.querySelector('[data-answers-pick]'));
      const c = getComputedStyle(document.querySelector('[data-answers-group]'));
      return [s.borderTopWidth, c.borderRadius, s.borderRadius]; }""")
    assert looks[0] != "0px" and looks[1] != looks[2]
    no_sideways(page)
    # the controls alone: no question's text in a screenshot
    box = page.evaluate("""() => { const c = document.querySelector('[data-model-answers]')
        .getBoundingClientRect(), g = document.querySelector('[data-answers-groups]')
        .getBoundingClientRect();
      return { x: c.left, y: c.top + scrollY, width: c.width, height: g.bottom - c.top + 8 }; }""")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / f"answers-switch-{width}.png", clip=box, full_page=True)
    assert page.errors == []


def _provisional_ran_out(route):
    r = route.fetch()
    body = r.json()
    for e in body["everyday"]["models"].values():
        e["provisional"] = True
    body["everyday"]["models"][MODEL]["ran_out"] = 2
    route.fulfill(response=r, body=json.dumps(body))


@pytest.mark.parametrize("width", WIDTHS)
def test_the_everyday_tiles_badge_sits_at_its_edge_on_one_line(live, page, width):
    page.route("**/api/results*", _provisional_ran_out)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    tile = page.locator("[data-kind-tile='everyday']")
    tile.wait_for()
    badge = tile.locator("[data-pilot-badge]")
    assert badge.inner_text() == "not ranked · provisional"
    box = page.evaluate("""() => {
      const t = document.querySelector("[data-kind-tile='everyday']");
      const b = t.querySelector('[data-pilot-badge]').getBoundingClientRect();
      const k = t.querySelector('.ktile-k').getBoundingClientRect();
      return { dx: b.left - k.left, h: b.height,
               lh: parseFloat(getComputedStyle(t.querySelector('[data-pilot-badge]')).lineHeight) || 20 }; }""")
    assert abs(box["dx"]) <= 1, box                     # at the tile's edge, as its heading
    assert box["h"] <= box["lh"] + 4, box               # one line
    # ran out, said with its base, as everywhere
    out = tile.locator("[data-evd-ran-out]")
    assert re.fullmatch(r"2 of \d+ ran out of room", out.inner_text())
    assert "hidden answers that score it" in out.get_attribute("title")
    no_sideways(page)
    clip(page, "[data-kind-tile='everyday']", f"everyday-tile-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_homes_everyday_tile_says_most_passed_not_best(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    card = page.locator("[data-best='everyday']")
    card.wait_for()
    assert card.locator(".eyebrow").inner_text().lower() == "everyday · most passed"   # 16.7
    assert card.locator("[data-pilot-badge]").inner_text().startswith("not ranked")
    no_sideways(page)
    assert page.errors == []


CONTRAST = """el => {
  const rgb = c => (c.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
  const lum = c => { const [r, g, b] = rgb(c).map(v => { v /= 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
  let bg = null;
  for (let e = el; e && !bg; e = e.parentElement) {
    const c = getComputedStyle(e).backgroundColor;
    if (c && !/rgba\\(.*, 0\\)$/.test(c) && c !== 'transparent') bg = c;
  }
  const a = lum(getComputedStyle(el).color), b = lum(bg || 'rgb(255,255,255)');
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05); }"""


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_checks_open_under_the_line_clicked_in_full_ink(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    link = page.locator("[data-needs='checks'] a")
    link.wait_for()
    link.click()
    pop = page.locator("#pop-checks")
    pop.wait_for()
    page.wait_for_timeout(200)
    at = page.evaluate("""() => {
      const a = document.querySelector("[data-needs='checks'] a").getBoundingClientRect();
      const p = document.querySelector('#pop-checks').getBoundingClientRect();
      return { dy: p.top - a.bottom, flipped: a.top - p.bottom, dx: p.left - a.left }; }""")
    # just under the link (or just over it, near the bottom), not at the dot up the page
    assert 0 <= at["dy"] <= 12 or 0 <= at["flipped"] <= 12, at
    assert abs(at["dx"]) <= 20 or width < 600, at
    for el in pop.locator(".check-short").all()[:4]:
        assert el.evaluate(CONTRAST) >= 4.5
    no_sideways(page)
    shot(page, f"checks-popover-{width}-{scheme}.png")
    assert page.errors == []
