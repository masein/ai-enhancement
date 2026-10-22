"""27c, Part 5 in a real browser: On our testers' phones.

- Per phone and model: Tint tints it; its legend says only what it uses; each
  row the model, as names.json names it, then the phone, its dot the model's
  board colour; a phone with no finished test of a model is in What failed.
- Every test: 27's table, its columns grouped under Speed, Memory and
  Battery; no engine build, thermal max, charging or first load (they are in
  the test's panel); a crash "–" for its times; a flagged value struck
  through, its reason on hover, and the "flagged" word under the date too.
- A test opens in a panel from the right: the page neither moves nor redraws,
  the row stays marked; Esc, × or a click outside closes it; ↑ and ↓ move;
  the link says &test=<id> and opens it with the page at its row.

The world is 26's: nine made-up tests from made-up testers (made-up ids and
replies), three of them crashes."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21d_browser import no_sideways
from test_26_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27c"
MODEL = "k4 phone build, Q4 (Qwen3.6-35B-A3B)"


def place(page, live, width=1280, scheme="dark", rest="", height=700):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": height})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=testers" + rest)
    page.wait_for_selector("[data-pt-table] tbody tr[data-pt-row]", state="attached",
                           timeout=30000)
    page.wait_for_timeout(300)


def test_per_phone_tints_says_only_what_it_uses_and_leaves_crashes_out(live, page):
    place(page, live)
    per = page.locator("[data-mt='pt-per']")
    legend = per.locator("[data-mt-legend]").inner_text()
    assert "published" not in legend and "±" not in legend and "standard error" not in legend
    rows = page.eval_on_selector_all("table[data-pt-per] tr.mt-row",
                                     "rs => rs.map(r => r.dataset.ptPerRow)")
    # 23078PND5G: two crashes, no finished test — in What failed, not here
    assert sorted(rows) == ["CPH2747|model-004", "SM-S938B|model-004"]
    assert page.locator("[data-pt-failed='23078PND5G|1.4']").count() == 1
    # the model as names.json names it, then the phone (a long one cut in its
    # middle, the whole on hover)
    names = page.eval_on_selector_all("table[data-pt-per] td.mt-name",
                                      "ts => ts.map(t => [t.innerText.split('\\n')[0], t.title])")
    assert [n.split(" · ")[-1] for n, _ in names] == ["OnePlus 15", "Galaxy S25 Ultra"]
    assert all(n.startswith("k4 phone build") and t.startswith(MODEL) for n, t in names), names
    dot = page.locator("table[data-pt-per] tr[data-pt-per-row='CPH2747|model-004'] .mt-mark")
    assert dot.get_attribute("style").startswith("background:var(--s")   # the model's colour
    # Tint on: the columns tinted, no bars
    per.locator("[data-tint]").check()
    page.wait_for_selector("table[data-pt-per] td[data-mt-heat]")
    assert "tint" in page.locator("[data-mt='pt-per'] [data-mt-legend]").inner_text()
    per = page.locator("[data-mt='pt-per']")
    per.locator("[data-tint]").uncheck()
    assert page.errors == []


def test_every_test_is_27s_table_grouped_without_the_engine(live, page):
    place(page, live)
    t = page.locator("table[data-pt-table]")
    assert t.get_attribute("data-model-table") == "pt-tests"
    groups = t.locator("thead tr.grp th[data-mt-colgroup]").evaluate_all(
        "ts => ts.map(t => t.dataset.mtColgroup)")
    assert groups == ["Speed", "Memory", "Battery"]
    heads = t.locator("thead tr.mt-hrow th").evaluate_all("ts => ts.map(t => t.title || '')")
    for gone in ("App · engine", "Thermal max", "Charging", "First load"):
        assert gone not in heads, heads
    assert "engine" not in t.locator("td[data-pt-cell='t1|app']").inner_text()
    # a crash: "–" for its times, never "0.0 s"
    assert t.locator("td[data-pt-cell='x1|ttft']").inner_text() == "–"
    # flagged: struck through, the reason on hover; and the word under the date
    dec = t.locator("td[data-pt-cell='t4|decode']")
    assert "pt-off" in dec.get_attribute("class")
    assert dec.locator(".mt-n").evaluate("e => getComputedStyle(e).textDecorationLine") == (
        "line-through")
    assert dec.get_attribute("title").startswith("decode 400 tok/s: no phone does that")
    word = t.locator("[data-pt-flag-why='t4']")
    assert word.inner_text() == "flagged" and word.get_attribute("title").startswith("decode 400")
    # numbers right-aligned in monospace; the names column stays at the left
    assert dec.evaluate("e => getComputedStyle(e).textAlign") == "right"
    assert "monospace" in dec.locator(".mt-n").evaluate("e => getComputedStyle(e).fontFamily") \
        or "Mono" in dec.locator(".mt-n").evaluate("e => getComputedStyle(e).fontFamily")
    assert t.locator("tbody td.mt-name").first.evaluate("e => getComputedStyle(e).position") == (
        "sticky")
    assert no_sideways(page) and page.errors == []


def test_a_test_opens_in_a_panel_and_the_page_never_moves(live, page):
    place(page, live)
    row = page.locator("table[data-pt-table] tr[data-pt-row='t5']")
    row.scroll_into_view_if_needed()
    page.evaluate("window.scrollBy(0, 40)")
    y = page.evaluate("scrollY")
    page.evaluate("document.querySelector('[data-pt-table]').dataset.seen = 'yes'")
    row.locator("td.mt-name").click()
    panel = page.locator("[data-pt-panel]")
    panel.wait_for()
    assert panel.get_attribute("data-pt-test") == "t5"
    page.wait_for_selector("[data-pt-panel] [data-pt-check]", timeout=30000)
    assert page.evaluate("scrollY") == y
    # no redraw: the table is the same element, the row marked
    assert page.evaluate("document.querySelector('[data-pt-table]').dataset.seen") == "yes"
    assert "pt-on" in row.get_attribute("class")
    assert page.evaluate("location.hash").endswith("&test=t5")
    assert panel.evaluate("e => e.getBoundingClientRect().width") == 480
    # its conditions: what left the table
    assert panel.locator("[data-pt-cond='Thermal max']").inner_text() == "1"
    assert panel.locator("[data-pt-cond='Charging']").inner_text() == "yes"
    assert panel.locator("[data-pt-cond='Engine build']").inner_text() == "engine, made-up build"
    # ↓ the next test, ↑ back
    page.keyboard.press("ArrowDown")
    page.wait_for_function("document.querySelector('[data-pt-panel]').dataset.ptTest === 't6'")
    page.keyboard.press("ArrowUp")
    page.wait_for_function("document.querySelector('[data-pt-panel]').dataset.ptTest === 't5'")
    # Esc closes; the page hasn't moved
    page.keyboard.press("Escape")
    assert panel.count() == 0 and page.evaluate("scrollY") == y
    assert "test=" not in page.evaluate("location.hash")
    assert "pt-on" not in (row.get_attribute("class") or "")
    # × and a click outside close it too
    row.locator("td.mt-name").click()
    page.locator("[data-pt-panel] [data-pt-close]").click()
    assert page.locator("[data-pt-panel]").count() == 0
    row.locator("td.mt-name").click()
    page.locator("[data-pt-panel]").wait_for()
    page.mouse.click(20, 300)
    assert page.locator("[data-pt-panel]").count() == 0 and page.evaluate("scrollY") == y
    assert page.errors == []


@pytest.mark.parametrize("width", [1280, 400])
def test_a_link_opens_its_test_with_the_page_at_its_row(live, page, width):
    place(page, live, width=width, rest="&test=x2", height=800)
    panel = page.locator("[data-pt-panel]")
    panel.wait_for(timeout=30000)
    assert panel.get_attribute("data-pt-test") == "x2"
    page.wait_for_selector("[data-pt-panel] [data-pt-check], [data-pt-panel] .small",
                           timeout=30000)
    if width >= 640:
        # the page at its row: in sight, below the header
        box = page.locator("table[data-pt-table] tr[data-pt-row='x2']").bounding_box()
        assert 80 <= box["y"] <= 700, box
    else:
        # on a phone the panel is the whole screen
        assert panel.evaluate("e => e.getBoundingClientRect().width") == width
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(300)
    page.screenshot(path=str(SCREENS / f"testers-panel-{width}-dark.png"))
    assert page.errors == []
