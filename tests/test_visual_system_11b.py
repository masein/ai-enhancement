"""11b: the visual system. Tokens, type, the sticky bar with its LIVE badge,
numbered sections, one table component, back to top, and the tooltip.

The 9d tests in test_one_visual_system.py still hold — the scale, the contrast
in every theme, and the table component — and this file adds what 11b brings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import go_tab


pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11"

# the tokens, read by painting them: a custom property's computed value is the
# token stream ("color-mix(…)"), the painted colour is what a reader sees
PAINT = """keys => keys.map(k => {
  const p = document.createElement('div');
  p.style.color = `var(${k})`;
  document.body.append(p);
  const c = getComputedStyle(p).color;
  p.remove();
  return c;
})"""

LIGHT = {
    "--plane": "rgb(246, 248, 252)", "--surface-1": "rgb(255, 255, 255)",
    "--text-primary": "rgb(20, 33, 61)", "--text-secondary": "rgb(61, 75, 102)",
    "--muted": "rgb(90, 107, 133)", "--border": "rgb(228, 233, 242)",
    "--grid": "rgb(228, 233, 242)", "--axis": "rgb(201, 210, 227)",
    "--accent": "rgb(47, 84, 235)", "--live-text": "rgb(11, 122, 90)",
    "--live-dot": "rgb(18, 184, 134)",
}
DARKISH = {"--accent": "rgb(124, 155, 255)", "--live-text": "rgb(47, 211, 154)"}


def paint(page, keys):
    return dict(zip(keys, page.evaluate(PAINT, list(keys))))


@pytest.mark.parametrize("theme", ["light", "dark", "dim"])
def test_the_tokens_are_the_new_palette(live, browser, theme):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, reduced_motion="reduce")
    ctx.add_init_script(f"localStorage.setItem('bench-theme', '{theme}');")
    pg = ctx.new_page()
    try:
        pg.goto(live["base"] + "/")
        pg.wait_for_selector("#view .card")
        want = LIGHT if theme == "light" else {**LIGHT, **DARKISH}
        keys = list(LIGHT) if theme == "light" else list(DARKISH)
        got = paint(pg, keys)
        for k in keys:
            assert got[k] == want[k], f"{theme} {k}: {got[k]}"
        # the five heat steps run from palest to strongest, and none of them is
        # the card surface itself
        heat = pg.evaluate(PAINT, [f"--heat-{i}" for i in range(1, 6)])
        assert len(set(heat)) == 5, heat
        surface = paint(pg, ["--surface-1"])["--surface-1"]
        assert heat[0] != surface
        # ink on the strongest step still reads
        ratio = pg.evaluate("""([fg, bg]) => {
          // color-mix() computes to color(srgb r g b) with 0–1 channels
          const rgb = c => { const n = (c.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
            return /^color\\(/.test(c) ? n.map(v => v * 255) : n; };
          const lum = ([r, g, b]) => { const f = v => (v /= 255) <= 0.03928 ? v / 12.92
            : Math.pow((v + 0.055) / 1.055, 2.4); return .2126*f(r) + .7152*f(g) + .0722*f(b); };
          const a = lum(rgb(fg)), b = lum(rgb(bg));
          return (Math.max(a, b) + .05) / (Math.min(a, b) + .05); }""",
          [paint(pg, ["--text-primary"])["--text-primary"], heat[4]])
        assert ratio >= 4.5, ratio
    finally:
        ctx.close()


def test_the_data_and_the_labels_are_mono(live, page):
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("table[data-lb-table] tbody tr")
    mono = page.evaluate("""() => {
      const fam = e => getComputedStyle(e).fontFamily;
      const isMono = f => /ui-monospace|SF Mono|Menlo|Consolas|monospace/i.test(f);
      const one = sel => { const e = document.querySelector(sel); return e ? isMono(fam(e)) : null; };
      return {th: one('table[data-lb-table] thead th'),
              num: one('table[data-lb-table] tbody td.num'),
              status: one('[data-statusline]'),
              badge: one('#view .badge'),
              footer: isMono(fam(document.querySelector('footer'))),
              live: one('#liveBadge'),
              prose: isMono(fam(document.querySelector('#view p.sub')))}; }""")
    # 11h: the status line and the footer are words, so sans
    assert mono["th"] and mono["num"] and mono["live"]
    assert mono["status"] is False and mono["footer"] is False
    assert mono["badge"] in (True, None)
    assert mono["prose"] is False                 # prose stays in the sans face
    assert page.errors == []


def test_the_bar_is_sticky_and_56px(live, page):
    page.set_viewport_size({"width": 1280, "height": 860})
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("table[data-lb-table] tbody tr")
    bar = page.locator("#bar")
    # 56px of bar, plus the hairline under it
    assert bar.bounding_box()["height"] <= 57.5
    page.evaluate("window.scrollTo(0, 3000)")
    page.wait_for_timeout(120)
    box = bar.bounding_box()
    assert box["y"] <= 0.5, box                   # still at the top of the window
    assert page.evaluate("getComputedStyle(document.querySelector('#bar')).position") == "sticky"
    # the tabs are in the bar, so a tab is reachable without scrolling back up
    assert page.locator("#bar #tabs button[role=tab]").count() >= 5
    assert page.errors == []


def test_the_live_badge_says_the_state_and_only_a_live_page_has_one(live, page, tree):
    page.goto(live["base"] + "/")
    badge = page.locator("#liveBadge")
    badge.wait_for()
    page.wait_for_selector("#liveBadge[data-fresh='ok'] .dot.ok")
    assert badge.text_content().startswith("LIVE · ")
    assert badge.is_visible()
    # when the data last changed is on the badge, for anyone who wants it (11h)
    assert "data last changed" in badge.get_attribute("title")
    # a frozen report is not live, and says nothing about being live
    page.goto(tree["report"].as_uri())
    page.wait_for_selector("#view .card")
    assert page.locator("#liveBadge").is_hidden()
    assert "generated" in page.locator("#metaChips").text_content()
    assert page.errors == []


def test_reduced_motion_stops_the_pulse(live, browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 860},
                              reduced_motion="reduce")
    pg = ctx.new_page()
    try:
        pg.goto(live["base"] + "/")
        pg.wait_for_selector("#liveBadge .dot")
        name = pg.evaluate(
            "getComputedStyle(document.querySelector('#liveBadge .dot')).animationName")
        assert name == "none"
    finally:
        ctx.close()


@pytest.mark.parametrize("label", ["Overview", "Leaderboard", "Models", "Queue"])
def test_the_sections_are_numbered_in_order(live, page, label):
    page.goto(live["base"] + "/")
    page.wait_for_selector("#view .card")
    go_tab(page, label)
    page.wait_for_selector("#view .card h2")
    # a card's heading may sit in its .sechead, beside the card's actions
    ix = page.evaluate("""() => [...document.querySelectorAll(
        '#view > .card > h2[data-ix], #view > .card > .sechead > h2[data-ix]')]
      .map(e => e.dataset.ix)""")
    assert ix == [f"{i:02d}" for i in range(1, len(ix) + 1)], (label, ix)
    assert len(ix) >= 1
    # the index is drawn by CSS, so the section's name is still its own
    assert page.evaluate("""() => { const h = document.querySelector('#view > .card > h2[data-ix]');
      return [h.textContent.trim().slice(0, 2),
              getComputedStyle(h, '::before').content]; }""")[0] != ix[0]
    assert page.errors == []


def test_back_to_top_appears_after_two_screens_and_moves_focus_to_the_bar(live, page):
    # a short window, so two screens of this board is a scroll it really has
    page.set_viewport_size({"width": 1280, "height": 300})
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("table[data-lb-table] tbody tr")
    top = page.locator("#toTop")
    assert top.is_hidden()
    page.evaluate("window.scrollTo(0, innerHeight * 2 + 40)")
    assert page.evaluate("scrollY >= innerHeight * 2"), "the board is shorter than two screens"
    page.wait_for_selector("#toTop:not([hidden])")
    top.click()
    page.wait_for_function("window.scrollY < 4")
    assert page.evaluate("document.querySelector('#bar').contains(document.activeElement)")
    assert page.errors == []


def test_the_tooltip_inverts_and_is_wired_to_what_it_describes(live, page):
    page.goto(live["base"] + "/#tab=tasks")
    page.wait_for_selector("#view svg [data-tip]")
    style = page.evaluate("""() => { const t = getComputedStyle(document.querySelector('#tip'));
      return [t.backgroundColor, t.color, t.maxWidth, t.borderRadius]; }""")
    ink, surface = page.evaluate(PAINT, ["--text-primary", "--surface-1"])
    assert style[0] == ink and style[1] == surface        # it inverts, in every theme
    assert style[2] == "280px" and style[3] == "6px"
    # it shows on focus as well as hover, and says what it describes
    page.evaluate("document.querySelector('#view svg [data-tip]').focus()")
    page.wait_for_function("getComputedStyle(document.querySelector('#tip')).opacity === '1'")
    assert page.evaluate("document.activeElement.getAttribute('aria-describedby')") == "tip"
    page.evaluate("document.activeElement.blur()")
    page.wait_for_function("getComputedStyle(document.querySelector('#tip')).opacity === '0'")
    assert page.evaluate("!document.querySelector('[aria-describedby=\"tip\"]')")
    assert page.errors == []


def test_screenshots_for_the_pr(live, page):
    SCREENS.mkdir(parents=True, exist_ok=True)
    base = live["base"]
    where = [("overview", "/#tab=overview", "#view .card"),
             ("leaderboard", "/#tab=leaderboard", "table[data-lb-table] tbody tr"),
             ("model", "/#model=fx%2Fgood-750m", ".card h2:has-text('Judged free response')")]
    for theme in ("light", "dark", "dim"):
        for width in (1280, 400):
            page.set_viewport_size({"width": width, "height": 900})
            for name, hash_, ready in where:
                page.goto(base + hash_)
                page.wait_for_selector(ready, timeout=30000)
                # the theme is applied in the page: a goto that only changes
                # the hash does not reload it, so localStorage alone would not
                page.evaluate("t => applyTheme(t)", theme)
                page.wait_for_timeout(400)
                page.screenshot(path=SCREENS / f"{name}-{width}-{theme}.png", full_page=True)
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
                ), f"{name} {width} {theme} overflows sideways"
    assert page.errors == []
