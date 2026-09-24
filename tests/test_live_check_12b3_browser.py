"""12b.3 on the page: what the live check of 2026-09-24 found. The board's own
address showed its title, its footer and nothing else; the status dot was
always amber; a model with judged topics showed a dash; the model page had two
filled buttons."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from test_live_check_11k import clear, plant_proposal

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12b"
MODEL = "fx/good-750m"
PLACES = ["Home", "Models", "Improve", "Benchmarks"]


def served(page, edit, delay=0.0):
    """the board as served, edited (and held back) on the way to the page"""
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        if delay:
            time.sleep(delay)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def fresh_page(browser, width=1512):
    """a new browser context: nothing rendered before, nothing in storage"""
    ctx = browser.new_context(viewport={"width": width, "height": 900}, reduced_motion="reduce")
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.errors = errors
    return ctx, pg


# ---------------------------------------------------------------------------
# A1: the bare address
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["/", "/?token=abc123"])
def test_the_bare_address_opens_home_cold(live, browser, url):
    """Loaded cold — a new context, no hash — not by changing the hash of a
    page that has already rendered, which is the case that worked."""
    ctx, pg = fresh_page(browser)
    try:
        pg.goto(live["base"] + url)
        pg.wait_for_selector("[data-needs-you]")
        assert pg.locator("#tabs > button[role=tab]").all_inner_texts() == PLACES
        heads = pg.evaluate("[...document.querySelectorAll('#view > .card h2')].map(h => h.textContent)")
        assert heads == ["Needs you", "Running now", "Best in each kind of test"]
        # the address says where it is, and keeps its query
        assert pg.evaluate("location.hash") == "#tab=home"
        assert pg.evaluate("location.search") == url[1:]
        assert pg.errors == []
    finally:
        ctx.close()


def test_the_header_is_drawn_before_the_scores_arrive(live, browser):
    """The scores are the board's biggest answer; the header needs none of
    them, so it is there while they are on their way."""
    ctx, pg = fresh_page(browser)
    try:
        served(pg, lambda b: None, delay=3.0)
        t0 = time.time()
        pg.goto(live["base"] + "/")
        pg.wait_for_selector("#tabs > button[role=tab]", timeout=2500)
        assert time.time() - t0 < 2.5
        assert pg.locator("#tabs > button[role=tab]").all_inner_texts() == PLACES
        assert pg.locator("header [data-test-model]").is_visible()
        assert pg.locator("#view [data-loading='results']").count() == 1   # the skeleton, still
        # a place clicked before the scores land is where the page opens
        pg.locator("#tabs [data-tab='benchmarks']").click()
        pg.wait_for_selector("[data-subswitch='benchmarks']", timeout=10000)
        assert pg.errors == []
    finally:
        ctx.close()


@pytest.mark.parametrize("typed, place, arrived", [
    ("#home", "Home", "[data-needs-you]"),
    ("#models", "Models", "[data-lb-table]"),
    ("#benchmarks", "Benchmarks", "[data-subswitch='benchmarks']"),
])
def test_a_bare_place_name_lands_on_its_place(live, browser, typed, place, arrived):
    ctx, pg = fresh_page(browser)
    try:
        pg.goto(live["base"] + "/" + typed)
        pg.wait_for_selector(arrived)
        assert pg.locator("#tabs button[aria-selected='true']").inner_text() == place
        assert pg.evaluate("location.hash").startswith("#tab=" + typed[1:])
        assert pg.errors == []
    finally:
        ctx.close()


@pytest.mark.parametrize("typed", ["#nonsense", "#tab=nonsense", "#model=nobody%2Fnothing"])
def test_an_address_the_router_does_not_know_is_home(live, browser, typed):
    ctx, pg = fresh_page(browser)
    try:
        pg.goto(live["base"] + "/" + typed)
        pg.wait_for_selector("[data-needs-you]")
        assert pg.evaluate("location.hash") == "#tab=home"
        assert pg.errors == []
    finally:
        ctx.close()


def test_an_error_drawing_the_board_is_said_not_swallowed(live, browser):
    """The first paint ran inside a catch that said nothing: a page that could
    not be drawn showed its title and footer, and no word why."""
    ctx, pg = fresh_page(browser)
    logged = []
    pg.on("console", lambda m: logged.append(m.text) if m.type == "error" else None)
    try:
        pg.route("**/api/results*", lambda route: route.fulfill(
            status=200, content_type="application/json", body="{}"))
        pg.goto(live["base"] + "/")
        box = pg.locator("[data-draw-failed]")
        box.wait_for()
        assert box.inner_text().startswith("The board could not be drawn.")
        assert "Reload the page" in box.inner_text()
        assert logged                                       # and the console has it
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# A4: problems and known limits
# ---------------------------------------------------------------------------

def only_limits(body):
    body["checks"] = [c for c in body["checks"] if c["limit"]]


def test_the_dot_counts_problems_and_the_limits_are_one_folded_line(live, page):
    page.goto(live["base"] + "/")
    page.wait_for_selector("#warnings [data-warn-summary]")
    # the service's board: the stub grader is its one problem; chat templates
    # and preliminary models are its known limits
    assert page.evaluate("DATA.checks.map(c => [c.key, c.limit])") == \
        [["chat_templates", True], ["preliminary", True], ["judge_stub", False]]
    summary = page.locator("#warnings summary[data-warn-summary]")
    assert summary.get_attribute("data-warn-summary") == "1"
    assert summary.locator(".statusn").inner_text() == "1"
    summary.click()
    # problems first, then one line for the limits, folded
    rows = page.locator("#warnings details.checks > .checklist > li[data-check]")
    assert rows.count() == 1 and rows.first.get_attribute("data-limit") is None
    assert rows.first.get_attribute("data-check") == "judge_stub"
    limits = page.locator("#warnings [data-known-limits]")
    assert limits.get_attribute("data-known-limits") == "2"
    assert limits.locator(":scope > details > summary").inner_text() == "Known limits (2) ▸"
    assert not limits.locator("li[data-check]").first.is_visible()
    limits.locator(":scope > details > summary").click()
    assert limits.locator("li[data-check][data-limit]").count() == 2
    # opened, they are rows in the panel, under their line — not a second
    # panel floating below it (12a.2: the list took the panel's position)
    panel = page.locator("#warnings details.checks > .checklist").bounding_box()
    head = limits.locator(":scope > details > summary").bounding_box()
    for r in limits.locator("li[data-check][data-limit]").all():
        assert r.is_visible()
        b = r.bounding_box()
        assert b["y"] > head["y"] and b["y"] + b["height"] <= panel["y"] + panel["height"] + 1
    # one arrow: the line's own ▸, no browser marker beside it
    assert limits.locator(":scope > details > summary").evaluate(
        "e => getComputedStyle(e).listStyleType") == "none"
    # Home's Needs you counts the problems, not the limits
    assert "1 check is not green" in page.locator("[data-needs-you]").inner_text()
    assert page.errors == []


def test_with_only_known_limits_the_dot_is_green_and_home_says_nothing(live, page):
    served(page, only_limits)
    page.goto(live["base"] + "/")
    page.wait_for_selector("#warnings [data-warn-summary]")
    summary = page.locator("#warnings summary[data-warn-summary]")
    assert summary.get_attribute("data-warn-summary") == "0"
    assert summary.locator(".dot.ok").count() == 1 and summary.locator(".statusn").count() == 0
    assert page.locator("[data-needs='checks']").count() == 0
    # add one problem — a duplicate row — and the dot says 1
    page.unroute("**/api/results*")

    def one_problem(body):
        only_limits(body)
        body["checks"].append({"key": "duplicates", "severity": "info", "show": None,
                               "short": "1 duplicate row: the same run submitted twice",
                               "text": "…", "judged": False, "limit": False})
    served(page, one_problem)
    page.reload()
    page.wait_for_selector("#warnings summary[data-warn-summary='1']")
    assert page.locator("#warnings .statusn").inner_text() == "1"
    assert page.locator("[data-needs='checks']").inner_text() == "1 check is not green"
    assert page.errors == []


# ---------------------------------------------------------------------------
# A5: Demo only datasets
# ---------------------------------------------------------------------------

def test_needs_you_does_not_count_demo_only_datasets(live, page):
    from service import db
    clear()
    try:
        demo = db.dataset_create(plant_proposal(status="approved"), "free", 10, "masein",
                                 {"provisional": True, "provisional_reason": "a local judge"})
        db.dataset_update(demo, status="ready")
        page.goto(live["base"] + "/")
        page.wait_for_selector("[data-needs-you]")
        page.wait_for_function("state.rv.loaded && state.trLoaded")
        assert page.locator("[data-needs='datasets']").count() == 0
        # it is still in Improve ▸ Review, with its badge
        page.goto(live["base"] + "/#tab=improve&sub=review&view=datasets")
        row = page.locator(f"[data-ds-row='{demo}']")
        row.wait_for()
        assert row.locator("[data-demo-only]").count() == 1
        # one that is not Demo only is counted
        real = db.dataset_create(plant_proposal(status="approved"), "free", 10, "masein", {})
        db.dataset_update(real, status="ready")
        page.goto(live["base"] + "/")
        page.wait_for_selector("[data-needs='datasets']")
        assert page.locator("[data-needs='datasets']").inner_text() == \
            "1 dataset made but not used in training"
        assert page.errors == []
    finally:
        clear()


# ---------------------------------------------------------------------------
# A6: judged topics, no average
# ---------------------------------------------------------------------------

def provisional(body):
    body["judged"]["calibration"] = None
    for m in body["models"]:
        if m.get("judgeState"):
            m["judgeState"] = {**m["judgeState"], "ok": False,
                               "reasons": ["a local judge, not calibrated — provisional"]}
        m["judgedAvg"] = None


def test_a_model_with_judged_topics_shows_them_not_a_dash(live, page):
    served(page, provisional)
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    tile = page.locator("[data-kind-tile='exam']")
    tile.wait_for()
    n, N, topic, v = page.evaluate(f"""() => {{ const m = DATA.models.find(x => x.id === '{MODEL}');
      const w = weakestTopic(m);
      const n = Object.entries(m.judge.tasks).filter(([t, x]) => DATA.judged.exam.includes(t)
        && pubScore(x) != null).length;
      return [n, DATA.judged.exam.length, frName(w.task), num(w.v, 2)]; }}""")
    assert tile.locator("[data-kind-value='exam']").inner_text() == f"{n} of {N}"
    sub = tile.locator(".ktile-sub").inner_text()
    assert sub.startswith(f"topics judged · weakest: {topic} {v} / 4")
    assert tile.locator("[data-provisional-badge]").count() == 1      # once, on the tile
    assert "—" not in tile.inner_text()
    # the block header says the same
    head = page.locator("[data-kind-block='exam'] > summary")
    assert head.locator(".kblock-v").inner_text() == f"{n} of {N}"
    assert f"weakest: {topic} {v} / 4" in head.inner_text()
    # Home: the weakest topic across the board, with one provisional badge
    page.goto(live["base"] + "/")
    card = page.locator("[data-best='exam']")
    card.wait_for()
    assert card.get_attribute("data-best-weakest") == "1"
    low = page.evaluate("""() => { let w = null;
      for (const m of DATA.models.filter(x => !x.duplicateOf)) { const t = weakestTopic(m);
        if (t && (!w || t.v < w.v)) w = { ...t, name: m.name }; }
      return [num(w.v, 2) + ' / 4', frName(w.task) + ' · ' + w.name]; }""")
    assert page.locator("[data-best-value='exam']").inner_text() == low[0]
    assert page.locator("[data-best-name='exam']").inner_text() == low[1]
    assert page.locator("[data-best-by-kind] [data-best-caveat]").count() == 1
    assert page.errors == []


# ---------------------------------------------------------------------------
# A7, A8
# ---------------------------------------------------------------------------

def test_one_filled_button_and_it_is_the_headers(live, page):
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    filled = page.locator("button.primary:visible")
    assert filled.count() == 1
    btn = page.locator("header [data-test-model]")
    assert btn.inner_text() == "Test this model" and btn.get_attribute("data-test-this") == MODEL
    btn.click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    assert dlg.locator("[data-ms='submit'] input").input_value() == MODEL
    page.keyboard.press("Escape")
    # everywhere else it is Test a model
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    assert btn.inner_text() == "Test a model" and btn.get_attribute("data-test-this") is None
    assert page.errors == []


def test_home_has_no_section_numbers_and_its_links_say_see_all_models(live, page):
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-best] .hcard-link")
    assert page.locator("#view h2[data-ix]").count() == 0
    assert set(page.locator("[data-best] .hcard-link").all_inner_texts()) == {"See all models →"}
    # a place that is a sequence keeps its numbers
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("#view h2[data-ix]")
    assert page.errors == []


def test_the_judged_block_says_judge_steady_and_keeps_the_numbers_behind_a_click(live, page):
    page.goto(live["base"] + "/#model=fx%2Fskewed-360m")
    from conftest import open_kind
    open_kind(page, "exam")
    block = page.locator("#sec-judged")
    steady = block.locator("[data-canary='steady']")
    assert steady.inner_text() == "Judge steady."
    assert "mean absolute deviation" not in block.inner_text()          # not in the main view
    how = block.locator("[data-how-judged]")
    how.locator("summary").click()
    assert "fixed scripts re-graded: mean absolute deviation" in how.inner_text()
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1512])
def test_the_screens(live, browser, width):
    ctx, pg = fresh_page(browser, width)
    try:
        pg.goto(live["base"] + "/")
        pg.wait_for_selector("[data-best] .hcard-link")
        SCREENS.mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=SCREENS / f"12b3-home-{width}-light.png", full_page=True)
        pg.locator("#warnings summary[data-warn-summary]").click()
        pg.locator("#warnings [data-known-limits] > details > summary").click()
        pg.screenshot(path=SCREENS / f"12b3-status-{width}-light.png")
        pg.keyboard.press("Escape")
        pg.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
        pg.wait_for_selector("[data-kind-tiles]")
        pg.screenshot(path=SCREENS / f"12b3-model-{width}-light.png")
        assert pg.errors == []
    finally:
        ctx.close()
