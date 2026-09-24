"""12b.1 on the page: the board organised as places. Four in the header and
nothing that opens a list of more; the run counter, Test a model, the status
dot and the name menu on the right; Improve and Benchmarks with their
switches; Models as one table. The brief's §1 table is the contract — every
row reachable from the new navigation — and every old address in §8 lands."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12b"
MODEL = "fx/good-750m"


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def home(page, base, width=1512):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/")
    page.wait_for_selector("#tabs [role=tab]", state="attached")
    page.wait_for_selector("#view > *")


def place(page, pid, sub=None):
    page.locator(f"#tabs [role=tab][data-tab='{pid}']").click()
    if sub:
        page.locator(f"[data-subswitch] [data-sub='{sub}']").click()


def name_menu(page, item):
    page.locator("#who button.who").click()
    page.locator(f"#pop-who [data-menu='{item}']").click()


# ---------------------------------------------------------------------------
# §1: every row of the table, reached from the new navigation
# ---------------------------------------------------------------------------

def reach_all_runs(page):
    page.locator("#runs [data-runs]").click()
    page.locator("#pop-runs [data-all-runs-link]").click()


def reach_test(page):
    page.locator("[data-test-model]").click()


def reach_theme(page):
    page.locator("#who button.who").click()


def reach_topic(page):
    place(page, "improve", "topics")
    page.locator("[data-loop-table] a[href^='#topic=']").first.click()


def reach_status(page):
    page.locator("#warnings summary[data-warn-summary]").click()


def reach_facts(page):
    place(page, "models")
    page.locator("[data-filters]").click()
    page.locator("[data-columns-menu]").click()


def reach_lm(page):
    place(page, "models")
    page.locator("[data-chip='lm']").click()


def reach_model_tab(tab):
    """a Models row opens the model page (12b.1); its tabs are 12b.2's"""
    def go(page):
        place(page, "models")
        page.locator(f"tr[data-lb-row='{MODEL}'] td.num").first.click()
        page.locator(f"[data-model-tabs] [data-mtab='{tab}']").click()
    return go


CONTRACT = [
    # (the row today, how it is reached now, what shows it arrived)
    ("Overview", lambda p: place(p, "home"),
     "[data-needs-you] ~ [data-running-now] ~ [data-best-by-kind]"),
    ("Leaderboard", lambda p: place(p, "models"), "[data-lb-table]"),
    ("Leaderboard ▸ Insights", lambda p: place(p, "models"), "[data-lb-card] ~ [data-insights]"),
    ("Leaderboard ▸ About these benchmarks", lambda p: place(p, "benchmarks", "standard"),
     ".about"),
    ("Models tab (its facts)", reach_facts, "#pop-columns [data-column-group='facts']"),
    ("Loop", lambda p: place(p, "improve", "topics"), "[data-loop-table]"),
    ("More ▸ Review", lambda p: place(p, "improve", "review"), "[data-review-head] [data-rv-view]"),
    ("More ▸ Training", lambda p: place(p, "improve", "training"),
     "#view h2:text-is('Training runs')"),
    ("More ▸ Exam", lambda p: place(p, "benchmarks", "exam"), "[data-panel='rubrics']"),
    ("Topic pages", reach_topic, "[data-topic-back]"),
    ("More ▸ Tasks", lambda p: place(p, "benchmarks", "standard"), ".panels"),
    ("More ▸ Perplexity & Loss", reach_lm, "[data-chip='lm'][aria-pressed='true']"),
    ("More ▸ Provenance ▸ Run provenance", reach_model_tab("history"),
     f"[data-model-prov='{MODEL}']"),
    ("More ▸ Provenance ▸ Query, Export", lambda p: name_menu(p, "data"),
     "#view h2:text-is('Query every metric')"),
    ("Queue", reach_all_runs, "[data-all-runs]"),
    ("Queue ▸ Submit a model", reach_test, "[data-dialog='test'] [data-submit-form]"),
    ("#everyday", lambda p: place(p, "benchmarks", "everyday"), "[data-everyday-table]"),
    ("7 checks ▾", reach_status, "#warnings .checklist [data-show-me]"),
    ("Theme ▾", reach_theme, "#pop-who [data-theme='dark']"),
    ("guide, the loop, How to read", lambda p: name_menu(p, "help"),
     "[data-help-guide], [data-how-to-read]"),
    ("Model page: 01 Judged · 02 Results · 03 Diagnose", reach_model_tab("scores"),
     "[data-kind-block='standard'], [data-kind-block='exam']"),
    ("Model page: 04 Provenance · 05 Runs", reach_model_tab("history"),
     "[data-model-prov] ~ [data-model-graded]"),
    ("Overview ▸ Judge steadiness", reach_status, "#warnings .checklist [data-judge-steady]"),
]


@pytest.mark.parametrize("row, reach, arrived", CONTRACT, ids=[c[0] for c in CONTRACT])
def test_every_row_of_the_table_is_reachable(live, page, row, reach, arrived):
    home(page, live["base"])
    reach(page)
    page.locator(arrived).first.wait_for()
    assert page.errors == []


def test_the_topic_back_link_reads_knowledge_exam(live, page):
    home(page, live["base"])
    reach_topic(page)
    back = page.locator("[data-topic-back]")
    assert back.text_content() == "← Knowledge exam"
    back.click()
    page.wait_for_selector("[data-panel='rubrics']")
    assert page.evaluate("location.hash") == "#tab=benchmarks&sub=exam"
    assert page.errors == []


# ---------------------------------------------------------------------------
# §8: old links land on their new home, with their sub-state
# ---------------------------------------------------------------------------

OLD = [
    ("#tab=overview", "#tab=home", "[data-needs-you]"),
    ("#tab=leaderboard", "#tab=models", "[data-models-view='standard'][aria-selected='true']"),
    ("#tab=models", "#tab=models", "[data-models-view='standard'][aria-selected='true']"),
    ("#tab=leaderboard&chip=math", "#tab=models&chip=math", "[data-chip='math'][aria-pressed='true']"),
    ("#tab=leaderboard&chip=judged", "#tab=models&view=exam",
     "[data-models-view='exam'][aria-selected='true']"),
    ("#tab=loop", "#tab=improve&sub=topics", "[data-loop-table]"),
    ("#tab=review", "#tab=improve&sub=review", "[data-review-head]"),
    ("#tab=review&view=datasets", "#tab=improve&sub=review&view=datasets",
     "[data-rv-view='datasets'][aria-selected='true']"),
    ("#tab=training", "#tab=improve&sub=training", "#view h2:text-is('Training runs')"),
    ("#tab=queue", "#tab=runs", "[data-all-runs]"),
    ("#tab=submit", "#tab=runs", "[data-dialog='test'] [data-submit-form]"),
    ("#tab=exam", "#tab=benchmarks&sub=exam", "[data-panel='rubrics']"),
    ("#tab=tasks", "#tab=benchmarks&sub=standard", ".panels"),
    ("#tab=perplexity", "#tab=models&chip=lm", "[data-chip='lm'][aria-pressed='true']"),
    ("#tab=provenance", "#tab=data", "#view h2:text-is('Run provenance')"),
    ("#everyday", "#tab=benchmarks&sub=everyday", "[data-everyday-table]"),
]


@pytest.mark.parametrize("old, new, arrived", OLD, ids=[o[0] for o in OLD])
def test_every_old_address_lands(live, page, old, new, arrived):
    page.set_viewport_size({"width": 1512, "height": 1000})
    page.goto(live["base"] + "/" + old)
    page.locator(arrived).first.wait_for()
    assert page.evaluate("location.hash") == new
    assert page.errors == []


def test_model_topic_and_reader_addresses_are_unchanged(live, page):
    page.set_viewport_size({"width": 1512, "height": 1000})
    base = live["base"]
    page.goto(base + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    assert page.evaluate("location.hash") == "#model=fx%2Fgood-750m"
    page.goto(base + "/#topic=economics")
    page.wait_for_selector("[data-topic-back]")
    assert page.evaluate("location.hash") == "#topic=economics"
    page.goto(base + "/#tab=exam&read=criteria:economics")
    page.wait_for_selector("#reader[data-kind='criteria'][data-ready='1']")
    assert page.evaluate("location.hash") == "#tab=benchmarks&sub=exam&read=criteria:economics"
    assert page.errors == []


# ---------------------------------------------------------------------------
# §2 and §3: the header
# ---------------------------------------------------------------------------

def test_the_header_holds_four_places_and_no_more_menu(live, page):
    home(page, live["base"])
    tabs = page.locator("#tabs [role=tab]")
    assert [t.text_content() for t in tabs.all()] == ["Home", "Models", "Improve", "Benchmarks"]
    assert page.locator("#moreBtn, [data-tab='more'], #barMore, #themeBtn").count() == 0
    bar = page.locator("#bar")
    assert bar.locator("button.primary").count() == 1
    assert bar.locator("button.primary").inner_text() == "Test a model"
    assert "checks" not in bar.text_content()
    assert page.locator("#liveBadge").is_visible()
    shot(page, "12b-header-1512-light.png", clip={"x": 0, "y": 0, "width": 1512, "height": 60})
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 720])
def test_below_720_the_places_are_one_menu_and_the_bar_is_one_line(live, page, width):
    home(page, live["base"], width)
    assert not page.locator("#tabs").is_visible()
    menu = page.locator("#menuBtn")
    assert menu.is_visible()
    boxes = [page.locator(s).bounding_box() for s in
             ("#menuBtn", "#runs [data-runs]", "[data-test-model]",
              "#warnings summary[data-warn-summary]", "#who button.who")]
    assert all(b for b in boxes)
    mid = [b["y"] + b["height"] / 2 for b in boxes]
    assert max(mid) - min(mid) < 4                        # one line
    assert max(b["x"] + b["width"] for b in boxes) <= width
    assert page.locator("#bar").bounding_box()["height"] <= 64
    # at a phone's width the name is its initial, not "Wh…"
    who = page.locator("#who button.who").inner_text().strip()
    assert who == ("Who are you? ▾" if width > 480 else "? ▾"), who
    set_name(page, "masein")
    who = page.locator("#who button.who").inner_text().strip()
    assert who == ("masein ▾" if width > 480 else "M ▾"), who
    menu.click()
    page.locator("#pop-places [data-place-sub='review']").click()
    page.wait_for_selector("[data-review-head]")
    shot(page, f"12b-header-{width}-light.png", clip={"x": 0, "y": 0, "width": width, "height": 60})
    assert page.errors == []


def test_the_status_dot_is_green_alone_or_amber_with_a_count(live, page):
    home(page, live["base"])
    page.evaluate("DATA.checks = []; _warnSig = null; renderWarnings();")
    dot = page.locator("#warnings summary[data-warn-summary]")
    assert dot.get_attribute("data-warn-summary") == "0"
    assert dot.text_content().strip() == ""
    assert dot.locator(".dot.ok").count() == 1
    dot.click()
    assert page.locator("[data-checks-none]").text_content() == "No problems."   # 12b.3
    page.evaluate("DATA.checks = [{key: 'a', severity: 'info', short: 'first', text: 'x', show: null},"
                  " {key: 'b', severity: 'warning', short: 'second', text: 'y', show: null}];"
                  " _warnSig = null; renderWarnings();")
    dot = page.locator("#warnings summary[data-warn-summary]")
    assert dot.text_content().strip() == "2"
    assert dot.locator(".dot.warn").count() == 1
    assert page.errors == []


def test_the_run_counter_lists_runs_and_leads_to_all_runs(live, page):
    base = live["base"]
    sid = api(base, "/api/submissions", {"hf_id": "fx/one-option-70m", "suite": "quick"})["id"]
    try:
        home(page, base)
        page.wait_for_function("(state.queue || []).some(r => r.status === 'queued')")
        pill = page.locator("#runs [data-runs]")
        assert pill.text_content() == "Runs"                 # queued is not running
        pill.click()
        line = page.locator(f"#pop-runs [data-run-line='{sid}']")
        assert "queued" in line.text_content() and "waiting its turn" in line.text_content()
        page.keyboard.press("Escape")
        # a running one: the dot pulses and the count says so
        page.evaluate(f"state.queue.find(r => r.id === {sid}).status = 'running'; "
                      "_runsSig = null; renderRuns();")
        assert page.locator("#runs [data-runs]").text_content() == "1 running"
        assert page.locator("#runs .dot.pulse").count() == 1
        page.locator("#runs [data-runs]").click()
        shot(page, "12b-run-counter-1512-light.png")
        page.locator("#pop-runs [data-all-runs-link]").click()
        page.wait_for_selector(f"[data-all-runs] tr[data-queue-row='{sid}']")
        assert page.evaluate("location.hash") == "#tab=runs"
    finally:
        api(base, f"/api/submissions/{sid}/cancel", {})
    assert page.errors == []


def test_test_a_model_opens_the_form_and_from_a_model_page_fills_it_in(live, page):
    home(page, live["base"])
    page.locator("[data-test-model]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    assert dlg.locator("h2").text_content() == "Test a model"
    assert dlg.locator("[data-suite-help]").count() == 1         # today's form, unchanged
    shot(page, "12b-test-a-model-1512-light.png")
    page.keyboard.press("Escape")
    dlg.wait_for(state="detached")
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    page.locator("[data-test-model]").click()
    inp = page.locator("[data-dialog='test'] [data-ms='submit'] input")
    inp.wait_for()
    assert inp.input_value() == MODEL
    page.locator("[data-dialog-close]").click()
    page.wait_for_selector("[data-dialog='test']", state="detached")
    assert page.errors == []


def test_the_name_menu_holds_theme_data_and_help(live, page):
    home(page, live["base"])
    page.locator("#who button.who").click()
    pop = page.locator("#pop-who")
    assert pop.locator("[data-who-input]").count() == 1           # the name, at the top
    pop.locator("[data-theme='dark']").click()
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    assert pop.locator("[data-theme='dark']").get_attribute("aria-checked") == "true"
    shot(page, "12b-name-menu-1512-dark.png")
    pop.locator("[data-theme='auto']").click()
    assert [b.text_content() for b in pop.locator("[data-menu]").all()] == \
        ["Data & sources", "Help"]
    assert page.errors == []


# ---------------------------------------------------------------------------
# §4 and §5: Benchmarks, Models
# ---------------------------------------------------------------------------

def test_the_switches_remember_the_last_choice(live, page):
    home(page, live["base"])
    place(page, "benchmarks", "exam")
    place(page, "home")
    place(page, "benchmarks")
    page.wait_for_selector("[data-panel='rubrics']")
    assert page.locator("[data-sub='exam'][aria-selected='true']").count() == 1
    place(page, "models")
    page.locator("[data-models-view='everyday']").click()
    place(page, "home")
    place(page, "models")
    page.wait_for_selector("[data-lb-everyday]")
    page.evaluate("localStorage.removeItem('bench-models-view'); "
                  "localStorage.removeItem('bench-benchmarks-sub')")
    assert page.errors == []


def test_models_is_one_table_and_a_row_opens_the_model_page(live, page):
    home(page, live["base"])
    place(page, "models")
    table = page.locator("[data-lb-table]")
    assert [b.text_content() for b in page.locator("[data-models-view]").all()] == \
        ["Standard", "Knowledge exam", "Everyday tasks"]
    assert [b.text_content() for b in page.locator("[data-chip]").all()] == \
        ["All tasks", "Knowledge", "Commonsense", "Reasoning", "Math", "Truthfulness",
         "Language modelling"]
    # Kind, Size, Status, Columns, Models and Scale are in Filters ▾
    assert page.locator(".lbbar > .pills").count() == 0
    page.locator("[data-filters]").click()
    sheet = page.locator("[data-filter-sheet]")
    for pill in ("Kind", "Size", "Status", "Columns", "Models", "Scale"):
        assert pill in sheet.text_content(), pill
    # no row expands: a click on a row is the model page
    assert page.locator("[data-open-row], [data-lb-detail]").count() == 0
    page.locator("[data-filters-done]").click()
    page.locator("[data-chip='math']").click()
    page.wait_for_function("location.hash === '#tab=models&chip=math'")
    table.locator(f"tr[data-lb-row='{MODEL}'] td.num").first.click()
    page.wait_for_selector("[data-model-hero]")
    assert page.evaluate("state.model") == MODEL
    # ← Back to Models is Back: the same entry, the chip it was on
    back = page.locator(".backlink")
    assert back.text_content() == "← Back to Models"
    assert back.get_attribute("href") == "#tab=models&chip=math"
    n = page.evaluate("history.length")
    back.click()
    page.wait_for_function("location.hash === '#tab=models&chip=math'")
    assert page.evaluate("history.length") == n                     # went back, pushed nothing
    assert page.errors == []


def test_a_model_with_nothing_in_the_view_is_one_line_not_a_row_of_dashes(live, page):
    home(page, live["base"])
    page.goto(live["base"] + "/#tab=models&view=exam")
    page.wait_for_selector("[data-lb-table]")
    rows = [r.get_attribute("data-lb-row") for r in page.locator("[data-lb-row]").all()]
    judged = page.evaluate("DATA.models.filter(m => Object.keys((m.judge || {}).tasks || {})"
                           ".some(t => t.startsWith('exam_'))).map(m => m.id)")
    assert sorted(rows) == sorted(judged)
    # it sorts by the judged average, and says so
    assert "sorted by Judged avg ▼" in page.locator("[data-lb-card]").inner_text()
    n = page.evaluate("DATA.models.length") - len(judged)
    line = page.locator("[data-not-tested]")
    assert line.get_attribute("data-not-tested") == str(n)
    assert line.text_content() == f"Not tested on this ({n}) ▸"
    assert page.locator("[data-not-tested-row]").count() == 0      # collapsed
    line.locator("button").click()
    assert page.locator("[data-not-tested-row]").count() == n
    first = page.locator("[data-not-tested-row]").first
    first.locator("[data-not-tested-test]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    assert dlg.locator("[data-ms='submit'] input").input_value() == \
        first.get_attribute("data-not-tested-row")
    assert dlg.locator("[data-select='suite']").get_attribute("data-value") == "judged"
    page.keyboard.press("Escape")
    shot(page, "12b-models-exam-1512-light.png", full_page=True)
    assert page.errors == []


def test_the_everyday_view_shows_the_pilot_rows_with_one_badge(live, page):
    home(page, live["base"])
    page.goto(live["base"] + "/#tab=models&view=everyday")
    page.wait_for_selector("[data-lb-everyday]")
    assert page.locator("[data-pilot-badge]").count() == 1
    assert page.locator("[data-lb-card] h2 [data-pilot-badge]").count() == 1
    row = page.locator(f"[data-lb-row='{MODEL}']")
    assert row.locator("[data-everyday-count]").text_content() == "5 of 5"
    assert row.locator("[data-evd-mark]").count() == 5
    assert page.locator("[data-not-tested]").count() == 1
    shot(page, "12b-models-everyday-1512-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# no page scrolls sideways; screenshots of each place
# ---------------------------------------------------------------------------

PLACES = [("home", "#tab=home"), ("models", "#tab=models"), ("improve", "#tab=improve"),
          ("benchmarks", "#tab=benchmarks&sub=standard"), ("runs", "#tab=runs"),
          ("data", "#tab=data"), ("help", "#tab=help")]


@pytest.mark.parametrize("width", [400, 1512])
def test_no_place_scrolls_sideways(live, browser, width):
    ctx = browser.new_context(viewport={"width": width, "height": 1000}, reduced_motion="reduce")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        for name, h in PLACES:
            page.goto(live["base"] + "/" + h)
            page.wait_for_selector("#view > *")
            # the shot is of the place, not of its skeleton
            page.wait_for_function("!document.querySelector('#view [aria-busy=true]')", timeout=20000)
            page.wait_for_timeout(250)
            assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth"), (name, width)
            shot(page, f"12b-{name}-{width}-light.png")
        assert errors == []
    finally:
        ctx.close()
