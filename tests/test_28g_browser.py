"""28g in a real browser: the Ask panel redesigned (board E, "Ask panel — four
states", docs/prompts/phase-28g/ask-panel-mock.dc.html). Its four states at
400 px, the comparison wide (720 px) and on a phone (390 px), dark and light;
masein's case (step8804 and Ternary Bonsai 2 2B on six benchmarks) with no
sideways scroll and every header whole; five models at 400 and 720; a chip
taken out or added runs again on the board with no model; "as a chart"; the
thumbs-down form; the status and Stop; ↑ for the last question; and the page
behind never redrawn or scrolled by the panel. The world and the fake model
are test_28_browser's."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from service import assistant
from test_21d_browser import no_sideways
from test_28_browser import CTRL, answered, ask, board, board_at, calm  # noqa: F401

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase28g"
SCHEMES = ("dark", "light")


def shot(page, name: str):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-ask-panel]").screenshot(path=str(SCREENS / name))


def opened(page, live, width=1280, scheme="dark", hash_="#tab=home"):
    board_at(page, live, width, scheme, hash_)
    page.wait_for_function("ASK.on")
    if width > 720:
        page.locator("[data-ask-open]").click()
    else:
        page.locator("#menuBtn").click()
        page.locator("[data-place='ask']").click()
    page.locator("[data-ask-panel]").wait_for()
    page.wait_for_function("!!ASK.choices")


def whole(page, sel: str) -> list[str]:
    """the cells whose text is cut: wider than they show, or clipped"""
    return page.evaluate(f"""() => [...document.querySelectorAll({json.dumps(sel)})]
      .filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.textContent)""")


def unscrolled(page) -> bool:
    return page.evaluate("""() => { const P = document.querySelector('[data-ask-panel]');
      return [...P.querySelectorAll('*')].every(e => e.getBoundingClientRect().right
        <= P.getBoundingClientRect().right + 1) && P.scrollWidth <= P.clientWidth + 1
        && document.querySelector('[data-ask-msgs]').scrollWidth
          <= document.querySelector('[data-ask-msgs]').clientWidth + 1; }""")


# ---------------------------------------------------------------------------
# board E's four states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
def test_board_e_four_states(live, page, scheme):
    opened(page, live, scheme=scheme)
    # 1: opened, nothing asked yet
    assert page.locator("[data-ask-empty] h2").inner_text() == "Ask about the board"
    assert page.locator("[data-ask-status]").inner_text() == "Q4 · our server"
    shot(page, f"1-empty-400-{scheme}.png")
    # 2: a follow-up with a table — masein's case
    ask(page, "step8804 on the frontier benchmarks", "table")
    got = ask(page, "Ternary Bonsai 2 2B", "table")
    assert got.locator("[data-ask-say]").inner_text() == "Compared 2 models on 6 benchmarks"
    assert page.locator("[data-ask-you] .ask-note").last.inner_text() == (
        "· added to the last answer")
    shot(page, f"2-table-400-{scheme}.png")
    # 3: a number, then asking back
    page.locator("[data-ask-clear]").click()
    got = ask(page, "q2's gpqa?", "table")
    assert got.get_attribute("data-ask-card") == "number"
    assert got.locator("[data-ask-number-sub]").inner_text().startswith("Measured here")
    got = ask(page, "how does the step one do on HLE", "clarify")
    assert got.locator("[data-ask-say]").inner_text() == "Which checkpoint?"
    assert got.locator("[data-ask-why]").inner_text() == "“The step one” fits three of ours."
    assert got.locator("[data-ask-all]").inner_text() == "All three, side by side"
    assert got.locator("[data-ask-choice]").count() == 3
    assert page.locator("[data-ask-input]").get_attribute("placeholder") == "Or type which one…"
    shot(page, f"3-number-ask-400-{scheme}.png")
    # 4: what it understood, a thumbs-down; slower while a run uses the model
    page.locator("[data-ask-clear]").click()
    got = ask(page, "frontier, last 7 days, sorted by OTIS", "open")
    assert got.locator("[data-ask-chip]").all_inner_texts() == [
        "Tab: Frontier\n×", "Tested: last 7 days\n×"]
    assert got.locator("[data-ask-add]").inner_text() == "+ Sort, Show, Columns…"
    got.locator("[data-ask-down]").click()
    got.locator("[data-ask-reason='Missed something I asked']").click()
    got.locator("[data-ask-note]").fill("sorted by OTIS")
    CTRL.update(slow=4.0, busy=True)
    box = page.locator("[data-ask-input]")
    box.fill("best of ours on HLE")
    box.press("Enter")
    page.locator("[data-ask-status='slow']").wait_for(timeout=5000)
    page.locator("[data-ask-stop]").wait_for()
    assert page.locator("[data-ask-waiting]").count() == 1
    shot(page, f"4-understood-400-{scheme}.png")
    answered(page, 2)
    assert unscrolled(page) and page.errors == []


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_comparison_wide_and_on_a_phone(live, page, scheme):
    opened(page, live, scheme=scheme)
    page.locator("[data-ask-wide]").click()
    w = page.locator("[data-ask-panel]").evaluate("e => e.getBoundingClientRect().width")
    assert w == 720
    ask(page, "step8804, step4000, step1000, q4 and q2 on gpqa and hle", "table")
    assert page.locator("[data-ask-col]").count() == 5
    shot(page, f"wide-compare-720-{scheme}.png")
    # remembered in this browser
    page.reload()
    page.wait_for_function("ASK.on")
    assert page.evaluate("ASK.wide") is True
    page.evaluate("ASK.wide = false; localStorage.setItem('ask-wide', '0')")
    # a phone: the whole width, the same table rules
    opened(page, live, width=390, scheme=scheme)
    assert page.locator("[data-ask-panel]").evaluate("e => e.getBoundingClientRect().width") == 390
    ask(page, "step8804 on the frontier benchmarks", "table")
    ask(page, "Ternary Bonsai 2 2B", "table")
    assert unscrolled(page) and no_sideways(page) and whole(page, "[data-ask-col]") == []
    shot(page, f"phone-390-{scheme}.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------

def test_masein_case_models_across_benchmarks_down(live, page):
    """step8804 and Ternary Bonsai 2 2B on six benchmarks at 400 px: one and a
    half columns, "OTIS" cut and a sideways scroll before"""
    opened(page, live)
    ask(page, "step8804 on the frontier benchmarks", "table")
    got = ask(page, "Ternary Bonsai 2 2B", "table")
    assert got.locator("[data-ask-col]").count() == 2
    assert got.locator("tbody tr").count() == 6
    assert got.locator("tbody th").all_inner_texts() == [
        "GPQA Diamond", "OTIS Mock AIME", "MMLU-Pro", "HLE", "IFEval", "IFBench"]
    assert unscrolled(page)
    assert whole(page, "[data-ask-col], .ask-grid tbody th, .ask-grid td") == []
    # the published number italic with its letter, the missing a muted dash
    pub = got.locator(".ask-grid .mt-pub")
    assert pub.count() >= 1 and got.locator(".ask-grid .mt-src").count() == pub.count()
    assert got.locator(".ask-none").count() >= 1
    assert "never ranked with ours" in got.locator("[data-ask-pubnote]").inner_text()
    assert page.errors == []


def test_five_models_show_three_columns_at_400_and_five_at_720(live, page):
    opened(page, live)
    got = ask(page, "step8804, step4000, step1000, q4 and q2 on gpqa and hle", "table")
    assert got.locator("[data-ask-col]").count() == 3
    assert got.locator("[data-ask-more]").inner_text() == "+2 more · Open in Compare ▸"
    assert got.locator("[data-ask-more] a").get_attribute("href").startswith(
        "#tab=models&view=compare&m=")
    assert unscrolled(page)
    page.locator("[data-ask-wide]").click()
    got = page.locator("[data-ask-bot]").last
    assert got.locator("[data-ask-col]").count() == 5 and got.locator("[data-ask-more]").count() == 0
    page.locator("[data-ask-wide]").click()
    assert page.errors == []


# ---------------------------------------------------------------------------
# chips and suggestions: the board, never the model
# ---------------------------------------------------------------------------

def test_taking_out_a_model_chip_runs_again_without_the_model(live, page):
    opened(page, live)
    got = ask(page, "step8804 vs q4 on gpqa and hle", "table")
    calls = CTRL["calls"]
    got.locator("[data-ask-chip^='models:'] [data-ask-x]").last.click()
    page.wait_for_function("!ASK.msgs.some(m => m.busy) && ASK.msgs.at(-1).edited")
    got = page.locator("[data-ask-bot]").last
    assert CTRL["calls"] == calls                         # no request to the model
    assert got.locator("[data-ask-col]").count() == 1
    assert got.locator("[data-ask-edited]").inner_text() == "edited"
    edits = assistant.edits()
    assert edits[-1]["why"] == "edit" and edits[-1]["q"] == "step8804 vs q4 on gpqa and hle"
    assert len(edits[-1]["before"]["models"]) == 2 and len(edits[-1]["after"]["models"]) == 1
    assert page.errors == []


def test_adding_a_sort_on_an_opened_frontier_page_changes_its_link(live, page):
    opened(page, live)
    got = ask(page, "frontier, last 7 days, sorted by OTIS", "open")
    page.wait_for_function("location.hash.includes('view=frontier')")
    calls = CTRL["calls"]
    got.locator("[data-ask-add]").click()
    page.locator("[data-ask-field='sort']").click()
    page.locator("[data-ask-picker='sort'] [data-ask-opt='otis']").click()
    page.wait_for_function("location.hash.includes('sort=otis')")
    assert CTRL["calls"] == calls
    got = page.locator("[data-ask-bot]").last
    assert "Sort: OTIS Mock AIME" in got.locator("[data-ask-chip]").all_inner_texts()[-1]
    assert page.errors == []


def test_as_a_chart_after_a_compare_is_the_same_models_and_benchmarks(live, page):
    opened(page, live)
    ask(page, "step8804 vs q4 on gpqa and hle", "table")
    sug = page.locator("[data-ask-sug]")
    assert "as a chart" in sug.all_inner_texts() and sug.count() <= 3
    calls = CTRL["calls"]
    page.locator("[data-ask-sug='as a chart']").click()
    got = answered(page, 2)
    assert CTRL["calls"] == calls
    assert got.get_attribute("data-ask-card") == "chart"
    assert [g.get_attribute("data-ask-chart-bench") for g in
            got.locator("[data-ask-chart-bench]").all()] == ["gpqa", "hle"]
    names = set(got.locator(".ask-dots .nm").all_inner_texts())
    assert names == {"step8804", "Q4 (UD-Q4_K_XL)"}
    assert page.locator("[data-ask-q]").last.inner_text() == "as a chart"
    # an open page: no suggestions, never filler
    ask(page, "frontier tab, ours only, sorted by HLE", "open")
    assert page.locator("[data-ask-suggest]").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# the status, Stop, the keys, and the page behind
# ---------------------------------------------------------------------------

def test_stop_abandons_a_request_and_the_panel_stays_usable(live, page):
    opened(page, live)
    CTRL.update(slow=3.0)
    box = page.locator("[data-ask-input]")
    box.fill("best of ours on HLE")
    box.press("Enter")
    page.locator("[data-ask-stop]").click()
    assert page.locator("[data-ask-stopped]").count() == 1
    assert page.locator("[data-ask-waiting]").count() == 0
    CTRL.update(slow=0.0)
    ask(page, "what's running?", "list")
    time.sleep(3.2)                        # the abandoned answer comes, and is dropped
    assert page.evaluate("ASK.msgs.filter(m => m.role === 'bot').length") == 1
    # ↑ in an empty box: the last question
    box = page.locator("[data-ask-input]")
    box.focus()
    box.press("ArrowUp")
    assert box.input_value() == "what's running?"
    assert page.errors == []


def test_the_panel_never_redraws_or_scrolls_the_page_behind(live, page):
    board_at(page, live, hash_="#tab=models&view=frontier")
    page.locator("[data-frontier-table]").wait_for(state="attached", timeout=20000)
    page.mouse.wheel(0, 600)
    page.wait_for_timeout(300)
    y = page.evaluate("scrollY")
    page.evaluate("window._view = document.getElementById('view').firstElementChild")
    page.locator("[data-ask-open]").click()
    page.wait_for_function("!!ASK.choices")
    ask(page, "step8804 vs q4 on gpqa and hle", "table")
    ask(page, "q2's gpqa?", "table")
    ask(page, "step8804, step4000, step1000, q4 and q2 on gpqa and hle", "table")
    page.locator("[data-ask-unfold]").first.click()
    page.locator("[data-ask-wide]").click()
    page.locator("[data-ask-wide]").click()
    got = page.locator("[data-ask-bot]").last
    got.locator("[data-ask-chip^='models:'] [data-ask-x]").last.click()
    page.wait_for_function("!ASK.msgs.some(m => m.busy) && ASK.msgs.at(-1).edited")
    assert page.evaluate("window._view.isConnected") and page.evaluate("scrollY") == y
    assert page.errors == []


CONTRAST = """() => {
  const P = document.querySelector('[data-ask-panel]');
  // a colour as the browser resolves it: rgb(), rgba() or color(srgb …)
  const rgba = s => {
    let m = s.match(/rgba?\\(([^)]+)\\)/);
    if (m) { const v = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
      return [v[0], v[1], v[2], v[3] ?? 1]; }
    m = s.match(/color\\(srgb ([^)]+)\\)/);
    if (m) { const v = m[1].split(/[ /]+/).filter(Boolean).map(Number);
      return [v[0] * 255, v[1] * 255, v[2] * 255, v[3] ?? 1]; }
    return [0, 0, 0, 0];
  };
  const over = (a, b) => [0, 1, 2].map(i => a[i] * a[3] + b[i] * (1 - a[3])).concat(1);
  const ground = e => {                  // the background behind e, composited
    const layers = [];
    for (let x = e; x; x = x.parentElement) {
      const c = rgba(getComputedStyle(x).backgroundColor);
      if (c[3] > 0) layers.push(c);
      if (c[3] >= 1) break;
    }
    return layers.reverse().reduce((b, c) => over(c, b), [255, 255, 255, 1]);
  };
  const lum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92
    : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]); };
  const bad = [];
  for (const e of P.querySelectorAll('*')) {
    const own = [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    if (!own || !e.getClientRects().length || e.closest('svg')) continue;
    const st = getComputedStyle(e);
    const g = ground(e), t = over(rgba(st.color), g);
    const [a, b] = [lum(t), lum(g)].sort((x, y) => y - x);
    const r = (a + 0.05) / (b + 0.05);
    if (r < 4.5) bad.push(e.textContent.trim().slice(0, 30) + ' ' + r.toFixed(2));
  }
  return bad;
}"""


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_panels_text_is_readable(live, page, scheme):
    """every word in the panel 4.5:1 or better against what is behind it"""
    opened(page, live, scheme=scheme)
    assert page.evaluate(CONTRAST) == []
    ask(page, "step8804 on the frontier benchmarks", "table")
    got = ask(page, "Ternary Bonsai 2 2B", "table")
    got.locator("[data-ask-down]").click()
    got.locator("[data-ask-reason='Wrong model']").click()
    got.locator("[data-ask-add]").click()
    assert page.evaluate(CONTRAST) == []
    ask(page, "how does the step one do on HLE", "clarify")
    ask(page, "q2's gpqa?", "table")
    assert page.evaluate(CONTRAST) == []
