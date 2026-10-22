"""27d in a real browser: search on every Models tab, one filter row with
Models ▾ on Latest and its chips, When ▾'s calendar, rows one line high, and
Pick to compare first in every row with its bar.

The world is 25's (21c's scorecard; step8804, step4000 and Q4 with IFEval and
IFBench, thinking on and off), with Standard's three generative benchmarks
for step8804 (on and off), Q4 and BF16 so they have rows on Standard ▸
Instruction & maths, and their files' sizes (24b). Every question, answer and
number in it is invented."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from service import db
from test_21c_page import SPEC
from test_21d_browser import no_sideways
from test_24b_browser import FILES
from test_25_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27d"
S88 = "served/step8804-lda-2610"
Q4 = SPEC["Q4 (UD-Q4_K_XL)"]["row"].removesuffix(" · thinking")
BF16 = SPEC["BF16"]["row"].removesuffix(" · thinking")
TABS = {"latest": "view=latest", "standard": "view=standard&group=instruction",
        "mobile": "view=mobile", "everyday": "view=everyday", "frontier": "view=frontier"}
ROWS = "[data-lb-card] table[data-model-table] tbody tr.mt-row"


@pytest.fixture(scope="module", autouse=True)
def more(board, live):  # noqa: F811
    import service.app as appmod
    from generative_fixture import write_run
    out = live["tree"]["out_dir"]
    write_run(out, S88, thinking=False)
    write_run(out, S88, thinking=True)
    write_run(out, Q4, thinking=True)
    write_run(out, BF16, thinking=True)
    for label, extra in FILES.items():
        mid = SPEC[label]["row"].removesuffix(" · thinking")
        db.served_put({**(db.served_get(mid) or {}), **extra})
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def go(page, live, hash_, width=1280, scheme="dark", fresh=True):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/")
    if fresh:
        page.evaluate("localStorage.removeItem('bench-cmp-picks'); "
                      "localStorage.removeItem('bench-mt-cols')")
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&" + hash_)
    page.wait_for_selector("[data-lb-card] [data-lb-search]", timeout=30000)
    page.wait_for_timeout(400)


def row_ids(page) -> list[str]:
    return page.eval_on_selector_all(ROWS, "rs => rs.map(r => r.dataset.lbRow || r.dataset.frRow"
                                     " || r.dataset.latestRows || r.dataset.latestRow || r.dataset.mtRow)")


def search(page, text: str) -> None:
    box = page.locator("[data-lb-search]")
    box.fill("")
    box.type(text, delay=30)
    page.wait_for_function(f"(location.hash.match(/[&]q=([^&]*)/) || [])[1] === {json.dumps(text.replace(' ', '%20'))}"
                           if text else "!/[&]q=/.test(location.hash)")
    page.wait_for_timeout(250)


# ---------------------------------------------------------------------------
# Part 1: search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tab", ["standard", "latest", "frontier"])
def test_8804_finds_step8804s_rows_and_nothing_else(live, page, tab):
    go(page, live, TABS[tab])
    page.wait_for_selector(ROWS, state="attached")
    assert len(row_ids(page)) > 1
    search(page, "8804")
    ids = row_ids(page)
    assert ids and all("8804" in i for i in ids), ids
    found = page.locator("[data-lb-search-count]").inner_text()
    assert re.fullmatch(r"\d+ of \d+ models? match", found), found
    # the matching letters marked in the name
    assert page.locator(f"{ROWS} mark.mt-hit").first.inner_text() == "8804"
    assert page.errors == []


def test_qwen36_q4_finds_the_q4_rows(live, page):
    go(page, live, TABS["latest"])
    search(page, "qwen36 q4")
    names = page.eval_on_selector_all(f"{ROWS} [data-mt-name]", "ns => ns.map(n => n.textContent)")
    assert names and all("Q4" in n for n in names), names
    assert not any("BF16" in n or "Q2" in n for n in names), names
    assert page.errors == []


def test_no_match_esc_clears_and_a_reload_keeps_it(live, page):
    go(page, live, TABS["latest"])
    search(page, "zzzz")
    assert page.locator("[data-lb-nomatch]").inner_text().startswith("No model matches ‘zzzz’ · ")
    page.locator("[data-lb-nomatch-clear]").click()
    page.wait_for_selector(ROWS)
    assert page.locator("[data-lb-search]").input_value() == ""
    search(page, "step")
    page.reload()
    page.wait_for_selector(ROWS, state="attached", timeout=30000)
    assert page.locator("[data-lb-search]").input_value() == "step"
    assert all("step" in n for n in page.eval_on_selector_all(f"{ROWS} [data-mt-name]",
                                                             "ns => ns.map(n => n.textContent)"))
    page.locator("[data-lb-search]").focus()
    page.keyboard.press("Escape")
    page.wait_for_function("!/[&]q=/.test(location.hash)")
    assert page.locator("[data-lb-search]").input_value() == ""
    # "/" anywhere puts the cursor in the search
    page.locator("h2").first.click()
    page.keyboard.press("/")
    assert page.evaluate("document.activeElement.dataset.lbSearch") == "1"
    assert page.errors == []


def test_typing_never_redraws_the_page_nor_moves_it(live, page):
    go(page, live, TABS["frontier"], scheme="light")
    page.evaluate("window.scrollTo(0, 180)")
    page.evaluate("document.querySelector('[data-lb-search]').dataset.seen = 'yes'; "
                  "document.querySelector('header').dataset.seen = 'yes'")
    box = page.locator("[data-lb-search]")
    box.focus()                                 # (a focus may bring the box into sight)
    page.wait_for_timeout(100)
    y = page.evaluate("scrollY")
    ys = []
    for ch in "qwen":
        page.keyboard.type(ch)
        page.wait_for_timeout(220)
        ys.append(page.evaluate("scrollY"))
    assert ys == [y] * 4, (y, ys)
    # the same box, still holding the cursor; the page around it never drawn again
    assert page.evaluate("document.activeElement.dataset.seen") == "yes"
    assert page.evaluate("document.querySelector('header').dataset.seen") == "yes"
    assert page.evaluate("document.activeElement.selectionStart") == 4
    SCREENS.mkdir(parents=True, exist_ok=True)
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_standard_while_typing_qwen(live, page, scheme):
    """the reply's screenshot"""
    go(page, live, TABS["standard"], scheme=scheme)
    box = page.locator("[data-lb-search]")
    box.focus()
    page.keyboard.type("qwen", delay=40)
    page.wait_for_function("location.hash.includes('q=qwen')")
    page.wait_for_timeout(300)
    ids = row_ids(page)
    assert ids and all("Qwen" in i for i in ids), ids
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-lb-card]").screenshot(path=str(SCREENS / f"standard-qwen-1280-{scheme}.png"))


# ---------------------------------------------------------------------------
# Part 2: the same filter row; Models ▾ on Latest; chips
# ---------------------------------------------------------------------------

ORDER = ["data-lb-search", "data-models-menu", "data-pill=\"when\"", "data-filters",
         "data-columns-menu", "data-mt-sort"]


@pytest.mark.parametrize("tab", list(TABS))
def test_one_filter_row_in_one_order(live, page, tab):
    go(page, live, TABS[tab])
    got = page.evaluate("""() => {
      const kinds = [['search', '[data-lb-search]'], ['models', '[data-models-menu]'],
        ['when', '[data-pill="when"]'], ['filters', '[data-filters]'],
        ['thinking', '#pill-thinking'], ['columns', '[data-columns-menu]'], ['sort', '[data-mt-sort]']];
      const row = document.querySelector('[data-lb-card] [data-pickers]');
      return [...row.querySelectorAll(kinds.map(k => k[1]).join(', '))].map(e =>
        kinds.find(k => e.matches(k[1]))[0]); }""")
    want = ["search", "models", "when", "filters", "thinking", "columns", "sort"]
    assert got == [w for w in want if w in got], got
    assert {"search", "models", "when", "filters", "sort"} <= set(got), got
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_latest_models_ours_shows_only_ours(live, page, scheme):
    go(page, live, TABS["latest"], scheme=scheme)
    assert {"ours", "open"} <= set(page.eval_on_selector_all(ROWS, "rs => rs.map(r => r.dataset.mtGroup)"))
    page.locator("#pill-models").click()
    page.wait_for_selector("#pop-models [data-models-show='ours']")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(200)
    page.screenshot(path=str(SCREENS / f"latest-models-menu-1280-{scheme}.png"))
    page.locator("#pop-models [data-models-show='ours']").click()
    page.wait_for_function("location.hash.includes('only=ours')")
    groups = set(page.eval_on_selector_all(ROWS, "rs => rs.map(r => r.dataset.mtGroup)"))
    assert groups == {"ours"}, groups
    chip = page.locator("[data-on='group']")
    assert chip.inner_text().startswith("Models: ours")
    assert page.locator("[data-clear-all]").count() == 1
    chip.locator("[data-on-remove]").click()
    page.wait_for_function("!location.hash.includes('only=')")
    assert "open" in set(page.eval_on_selector_all(ROWS, "rs => rs.map(r => r.dataset.mtGroup)"))
    assert page.errors == []


# ---------------------------------------------------------------------------
# Part 3: the date picker
# ---------------------------------------------------------------------------

def open_when(page):
    page.locator("#pill-when").click()
    page.wait_for_selector("#pop-when [data-cal]")


def test_the_presets_and_a_picked_range_filter_as_the_old_fields_did(live, page):
    go(page, live, TABS["frontier"])
    # Last 7 days, from its row, against the link the old menu made
    open_when(page)
    page.locator("#pop-when [data-choice='7d']").click()
    page.wait_for_function("location.hash.includes('tested=7d')")
    seven = row_ids(page)
    go(page, live, TABS["frontier"] + "&tested=7d", fresh=False)
    assert row_ids(page) == seven
    # a range picked on the calendar, against the link the old date fields made
    go(page, live, TABS["frontier"])
    open_when(page)
    days = page.eval_on_selector_all("#pop-when [data-cal-day]", "ds => ds.map(d => d.dataset.calDay)")
    a, b = days[3], days[20]
    page.locator(f"#pop-when [data-cal-day='{a}']").click()
    page.locator(f"#pop-when [data-cal-day='{b}']").hover()
    assert page.locator("#pop-when .cal-day.cal-prev").count() > 5            # the hover previews
    page.locator(f"#pop-when [data-cal-day='{b}']").click()
    words = page.locator("#pop-when [data-cal-words]").inner_text()
    assert words.endswith("· 18 days"), words
    assert page.locator("#pop-when .cal-day.cal-inr").count() == 18
    page.locator("#pop-when [data-cal-apply]").click()
    page.wait_for_function(f"location.hash.includes('tested={a}..{b}')")
    picked = row_ids(page)
    go(page, live, TABS["frontier"] + f"&tested={a}..{b}", fresh=False)
    assert row_ids(page) == picked
    assert page.errors == []


def test_the_calendar_works_with_the_keyboard_and_a_typed_date(live, page):
    go(page, live, TABS["latest"])
    open_when(page)
    first = page.locator("#pop-when [data-cal-day][tabindex='0']")
    first.focus()
    start = first.get_attribute("data-cal-day")
    page.keyboard.press("Enter")
    for _ in range(3):
        page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    words = page.locator("#pop-when [data-cal-words]").inner_text()
    assert words.endswith("· 11 days"), (start, words)
    # a typed date
    page.locator("#pop-when [data-cal-clear]").click()
    typed = page.locator("#pop-when [data-cal-typed]")
    typed.fill("14 oct")
    typed.press("Enter")
    page.locator("#pop-when [data-cal-typed]").fill("2026-10-20")
    page.locator("#pop-when [data-cal-typed]").press("Enter")
    assert page.locator("#pop-when [data-cal-words]").inner_text() == "14 Oct – 20 Oct 2026 · 7 days"
    page.keyboard.press("Escape")
    assert page.locator("#pop-when").count() == 0
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_date_picker_with_a_range_chosen(live, page, scheme):
    """the reply's screenshot"""
    go(page, live, TABS["latest"], scheme=scheme)
    open_when(page)
    page.locator("#pop-when [data-cal-typed]").fill("14 oct")
    page.locator("#pop-when [data-cal-typed]").press("Enter")
    page.locator("#pop-when [data-cal-typed]").fill("22 oct")
    page.locator("#pop-when [data-cal-typed]").press("Enter")
    assert page.locator("#pop-when [data-cal-words]").inner_text() == "14 Oct – 22 Oct 2026 · 9 days"
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(200)
    page.locator("#pop-when").screenshot(path=str(SCREENS / f"date-picker-{scheme}.png"))
    assert page.errors == []


# ---------------------------------------------------------------------------
# Part 4: shorter rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_latests_rows_are_short_and_each_line_one_line(live, page, scheme):
    go(page, live, TABS["latest"], scheme=scheme)
    got = page.evaluate("""() => [...document.querySelectorAll(
        '[data-latest-table] tr.mt-row[data-mt-group=ours]')].map(r => {
      const l = r.querySelector('[data-mt-line]');
      return { h: r.getBoundingClientRect().height, line: l ? l.getBoundingClientRect().height : 0,
               nowrap: l ? getComputedStyle(l).whiteSpace : 'nowrap',
               title: l ? l.title : '' }; })""")
    assert got and all(r["h"] <= 72 for r in got), got
    assert all(r["line"] <= 18 and r["nowrap"] == "nowrap" for r in got), got
    # words every row of Ours shares are on its heading, not in the rows
    head = page.locator("[data-latest-table] [data-mt-gnote='ours']").inner_text()
    assert head.startswith("measured here, same questions") and "thinking on" in head
    lines = page.eval_on_selector_all("[data-latest-table] tr.mt-row[data-mt-group=ours] [data-mt-line]",
                                      "ls => ls.map(l => l.textContent)")
    assert not any("thinking on" in x for x in lines), lines
    # the header: "⋯" only on hover or focus
    more = page.locator("[data-lb-card] th .col-more").first
    if more.count():
        assert more.evaluate("e => getComputedStyle(e).opacity") == "0"
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-lb-card]").screenshot(path=str(SCREENS / f"latest-rows-after-1280-{scheme}.png"))
    assert page.errors == []


# ---------------------------------------------------------------------------
# Part 5: Pick to compare
# ---------------------------------------------------------------------------

def test_two_picks_on_latest_show_the_bar_and_compare_opens_as_on_standard(live, page):
    go(page, live, TABS["latest"])
    ticks = page.locator(f"{ROWS} [data-cmp-tick]")
    assert ticks.count() >= 2
    # first in its row, before the dot; "Pick to compare" on hover
    first = page.locator(f"{ROWS} .mt-nrow").first
    assert first.evaluate("n => n.firstElementChild.matches('[data-cmp-tick]')")
    assert ticks.first.get_attribute("title") == "Pick to compare"
    a, b = ticks.nth(0).get_attribute("data-cmp-tick"), ticks.nth(1).get_attribute("data-cmp-tick")
    ticks.nth(0).check()
    bar = page.locator("[data-cmp-bar]")
    assert bar.get_attribute("data-cmp-bar") == "1"
    assert page.locator("[data-cmp-go]").is_disabled()
    ticks.nth(1).check()
    assert bar.inner_text().startswith("2 picked")
    assert not page.locator("[data-cmp-go]").is_disabled()
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / "compare-bar-1280-dark.png"))
    # they carry over to another tab, and survive a reload
    page.locator("[data-models-view='frontier']").click()
    page.wait_for_selector("[data-frontier-table]")
    assert page.locator("[data-cmp-bar]").get_attribute("data-cmp-bar") == "2"
    page.reload()
    page.wait_for_selector("[data-cmp-bar]", timeout=30000)
    page.locator("[data-cmp-go]").click()
    page.wait_for_function("location.hash.includes('view=compare')")
    latest = page.evaluate("location.hash")
    assert all(x in page.evaluate("decodeURIComponent(location.hash)") for x in (a, b)), (latest, a, b)
    # the same two picked on Standard open the same comparison
    go(page, live, TABS["standard"])
    page.evaluate(f"lbS().ticks = {json.dumps([a, b])}; localStorage.setItem('bench-cmp-picks', "
                  f"JSON.stringify({json.dumps([a, b])})); cmpSync()")
    page.locator("[data-cmp-go]").click()
    page.wait_for_function("location.hash.includes('view=compare')")
    assert page.evaluate("location.hash") == latest
    assert page.errors == []


def test_the_header_box_picks_every_row_shown_or_none(live, page):
    go(page, live, TABS["latest"])
    allbox = page.locator("[data-lb-card] th.model [data-cmp-all]")
    n = len(json.loads(allbox.get_attribute("data-ids")))
    allbox.check()
    assert page.locator("[data-cmp-bar]").get_attribute("data-cmp-bar") == str(n)
    if n > 8:
        assert page.locator("[data-cmp-most]").inner_text() == "at most 8"
    allbox.uncheck()
    assert page.locator("[data-cmp-bar]").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# every tab at 1280 and 400, light and dark
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tab", list(TABS))
@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_every_tab(live, page, tab, width, scheme):
    go(page, live, TABS[tab], width=width, scheme=scheme)
    assert page.locator("[data-lb-search]").is_visible()
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-lb-card]").screenshot(path=str(SCREENS / f"{tab}-{width}-{scheme}.png"))
