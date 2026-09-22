"""Phase 9d: one visual system — tokens, buttons that say what they are, one
badge component with three tones and one warning per row, one table
component, empty states with the action that fills them, skeletons for
loading, a focus ring and keyboard reach for every action, three themes."""

from __future__ import annotations

import pytest

from test_review_ui import law_under_its_draft_rubric

pytestmark = pytest.mark.dashboard

ALLOWED = {12.0, 14.0, 16.0, 20.0, 28.0}


def test_the_tokens_are_the_scale(live, page):
    page.goto(live["base"] + "/")
    page.wait_for_selector("#view .card")
    tok = page.evaluate("""() => { const s = getComputedStyle(document.body);
      return ['--sp-1','--sp-2','--sp-3','--sp-4','--sp-5','--sp-6','--fs-1','--fs-2','--fs-3',
              '--fs-4','--fs-5','--r-1','--r-2'].map(k => s.getPropertyValue(k).trim()); }""")
    assert tok == ["4px", "8px", "12px", "16px", "24px", "32px", "12px", "14px", "16px", "20px",
                   "28px", "6px", "10px"]
    radii = page.evaluate("""() => [getComputedStyle(document.querySelector('.card')).borderRadius,
      getComputedStyle(document.querySelector('#themeBtn')).borderRadius]""")
    assert radii == ["10px", "6px"]
    assert page.errors == []


@pytest.mark.parametrize("where", ["#tab=overview", "#tab=leaderboard",
                                   "#topic=medicine_clinical_health", "#tab=exam", "#tab=queue"])
def test_every_piece_of_text_is_on_the_type_scale(live, page, where):
    page.goto(live["base"] + "/" + where)
    page.wait_for_selector("#view > *")
    page.wait_for_timeout(1500)
    sizes = page.evaluate("""() => {
      const out = new Set();
      for (const e of document.querySelectorAll('.wrap *')) {
        if (e.closest('svg')) continue;                         // chart labels scale with the chart
        if (![...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) continue;
        const r = e.getBoundingClientRect(); if (!r.width) continue;
        out.add(parseFloat(getComputedStyle(e).fontSize));
      }
      return [...out]; }""")
    off = sorted(x for x in sizes if x not in ALLOWED and x >= 10.5)
    assert not off, f"{where}: text at {off}px is off the 12/14/16/20/28 scale"


def test_a_disabled_button_looks_disabled_and_says_why(live, page):
    page.goto(live["base"] + "/#topic=medicine_clinical_health")
    btn = page.locator("[data-propose='medicine_clinical_health']")
    btn.wait_for()
    assert btn.is_disabled()
    look = btn.evaluate("""b => { const s = getComputedStyle(b);
      return [s.cursor, parseFloat(s.opacity)]; }""")
    assert look[0] == "not-allowed" and look[1] < 1
    why = page.locator("[data-topic-page] [data-why='propose']")
    assert why.is_visible() and "under the 30" in why.text_content()
    # the four kinds of button look like four kinds
    page.goto(live["base"] + "/#tab=queue")
    primary = page.get_by_role("button", name="Submit model")
    bg = primary.evaluate("b => getComputedStyle(b).backgroundColor")
    accent = page.evaluate("getComputedStyle(document.body).getPropertyValue('--accent').trim()")
    assert bg != "rgba(0, 0, 0, 0)" and accent
    assert page.errors == []


def test_one_badge_three_tones_and_one_warning_per_row(live, page):
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("table[data-lb-table] tbody tr")
    per_row = page.evaluate("""() => [...document.querySelectorAll('table[data-lb-table] tbody tr')]
      .map(tr => tr.querySelectorAll('.badge.taint, .badge.prelim, .badge.over, .badge.warn').length)""")
    assert max(per_row) <= 1
    tones = page.evaluate("""() => { const probe = (cls) => { const b = document.createElement('span');
        b.className = cls; document.body.append(b); const c = getComputedStyle(b).color; b.remove();
        return c; };
      return [probe('badge'), probe('badge warn'), probe('badge taint'), probe('badge danger')]; }""")
    assert tones[1] == tones[2] and tones[0] != tones[1] and tones[3] not in (tones[0], tones[1])
    # the Loop board: what is true of every score is said once, above the board.
    # Every rubric delivered with the 37 topics is signed off, so nothing would
    # be: law is graded under its retired draft for this, as it was until then
    with law_under_its_draft_rubric(live):
        page.goto(live["base"] + "/#tab=loop")
        page.wait_for_selector("table[data-loop-table] tbody tr")
        rows = page.evaluate("""() => [...document.querySelectorAll('table[data-loop-table] tbody tr')]
          .map(tr => tr.querySelectorAll('.badge.taint').length - tr.querySelectorAll('td:nth-child(3) .badge').length)""")
        assert max(rows) <= 1
        assert page.locator("[data-loop-caveats]").count() == 1
    assert page.errors == []


def test_one_table_component(live, page):
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("table[data-models-table] tbody tr")
    th = page.locator("table[data-models-table] thead th").first
    assert th.evaluate("e => getComputedStyle(e).position") == "sticky"
    heights = page.evaluate("""() => [...document.querySelectorAll('table[data-models-table] tbody tr')]
      .map(tr => tr.getBoundingClientRect().height)""")
    assert min(heights) >= 39.5
    num = page.locator("table[data-models-table] tbody td.num").first
    assert num.evaluate("e => getComputedStyle(e).textAlign") == "right"
    assert "tabular-nums" in num.evaluate("e => getComputedStyle(e).fontVariantNumeric")
    assert page.errors == []


def test_an_empty_state_offers_what_fills_it(live, page):
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("table[data-models-table] tbody tr")
    page.locator("#view input[type=search]").first.fill("zzz-no-such-model")
    empty = page.locator("[data-empty]")
    empty.wait_for()
    assert "No model matches" in empty.text_content()
    empty.locator("[data-empty-action]").click()
    page.wait_for_selector("table[data-models-table] tbody tr")
    assert page.errors == []


def test_loading_is_a_skeleton_not_a_word(live, page):
    import time
    page.route("**/api/loop*", lambda route: (time.sleep(1.5), route.continue_())[1])
    page.goto(live["base"] + "/#tab=loop")
    page.wait_for_selector("[data-loading='loop'][aria-busy='true']", timeout=10000)
    assert "Loading…" not in page.locator("#view").text_content()
    page.wait_for_selector("table[data-loop-table] tbody tr", timeout=20000)
    page.unroute("**/api/loop*")


def test_every_action_is_reachable_from_the_keyboard(live, page):
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("table[data-models-table] thead th[data-sort='params']")
    th = page.locator("table[data-models-table] thead th[data-sort='params']")
    assert th.get_attribute("tabindex") == "0"
    before = page.evaluate("JSON.stringify(state.mdl.sort)")
    th.focus()
    page.keyboard.press("Enter")
    page.wait_for_function("b => JSON.stringify(state.mdl.sort) !== b", arg=before)
    # a visible focus ring on a control reached by Tab
    page.goto(live["base"] + "/#tab=queue")
    page.wait_for_selector("[data-ms='submit'] input")
    page.locator("[data-ms='submit'] input").focus()
    page.keyboard.press("Tab")                                   # to the next control
    ring = page.evaluate("""() => { const s = getComputedStyle(document.activeElement);
      return [s.outlineStyle, parseFloat(s.outlineWidth)]; }""")
    assert ring[0] != "none" and ring[1] >= 2
    assert page.errors == []


@pytest.mark.parametrize("theme", ["light", "dark", "dim"])
def test_the_text_reads_in_every_theme(live, browser, theme):
    ctx = browser.new_context(viewport={"width": 1512, "height": 900})
    ctx.add_init_script(f"localStorage.setItem('bench-theme', '{theme}');")
    pg = ctx.new_page()
    try:
        for where in ("#tab=overview", "#tab=leaderboard", "#topic=law"):
            pg.goto(live["base"] + "/" + where)
            pg.wait_for_selector("#view .card")
            pg.wait_for_timeout(800)
            worst = pg.evaluate("""() => {
              // color-mix() computes to color(srgb r g b) with 0–1 channels (11c's tint)
              const rgb = c => { const n = (c.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
                return /^color\\(/.test(c) ? n.map(v => v * 255) : n; };
              const lum = ([r, g, b]) => { const f = v => (v /= 255) <= 0.03928 ? v / 12.92
                : Math.pow((v + 0.055) / 1.055, 2.4); return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
              // the background a reader sees: translucent layers composited over
              // whatever is under them, down to the first opaque one
              const bgOf = e => { const layers = [];
                for (; e; e = e.parentElement) { const c = getComputedStyle(e).backgroundColor;
                  const v = (c.match(/[\\d.]+/g) || []).map(Number);
                  const al = v.length > 3 ? v[3] : 1;
                  const ch = /^color\\(/.test(c) ? v.slice(0, 3).map(x => x * 255) : v.slice(0, 3);
                  if (al > 0) { layers.push([ch, al]); if (al >= 1) break; } }
                let out = [255, 255, 255];
                for (const [c, al] of layers.reverse())
                  out = out.map((x, i) => c[i] * al + x * (1 - al));
                return out; };
              let worst = 21;
              for (const e of document.querySelectorAll('#view p, #view td, #view h2, #view th, #view a, '
                                                     + '#view .badge, #view .se, #view .small')) {
                if (!e.textContent.trim() || !e.getBoundingClientRect().width) continue;
                if (e.closest('.dim, [disabled], button:disabled')) continue;   // greyed on purpose
                const a = lum(rgb(getComputedStyle(e).color)), b = lum(bgOf(e));
                worst = Math.min(worst, (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05));
              }
              return worst; }""")
            # WCAG AA for normal text: 4.5:1 — badges and muted notes included
            assert worst >= 4.5, f"{theme} {where}: contrast {worst:.2f}"
    finally:
        ctx.close()
