"""11c: the Leaderboard. Topic-group chips, "Label: Value ▾" pills on the
shared popover, two header rows, one-line cells tinted by their rank on the
whole board, rows that open the model page (12b; they opened in place until
then), and Insights under the table.

The rules that do not bend are asserted here too: no provisional score is
tinted, averaged or put on a frontier, and a tie the z-test cannot break is
shown as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import open_filters

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11"
LB = "table[data-lb-table]"
TOP = "fx/good-750m-tuned-skill"


def open_lb(page, base, frag=""):
    page.goto(base + "/#tab=leaderboard" + frag)
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")


def heads(page):
    return page.evaluate(f"""() => [...document.querySelectorAll('{LB} thead tr:not(.grp) th')]
      .map(t => t.getAttribute('data-col') || t.childNodes[0].textContent.trim())""")


def uncalibrated(page):
    """The same board as served, with the judge not calibrated: every judged
    score on it is provisional."""
    def handle(route):
        r = route.fetch()
        body = r.json()
        body["judged"]["calibration"] = None
        for m in body["models"]:
            if m.get("judgeState"):
                m["judgeState"] = {**m["judgeState"], "ok": False,
                                   "reasons": ["a local judge, not calibrated — provisional"]}
            m["judgedAvg"] = None
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


# ---------------------------------------------------------------------------
# chips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chip,want", [
    ("commonsense", ["hellaswag", "piqa", "winogrande"]),
    ("reasoning", ["arc_challenge", "arc_easy"]),
    ("math", ["gsm8k"]),
    ("truthfulness", ["truthfulqa_mc2"]),
])
def test_a_chip_shows_exactly_its_groups_columns(live, page, chip, want):
    open_lb(page, live["base"])
    page.locator(f"[data-chip='{chip}']").click()
    page.wait_for_selector(f"[data-chip='{chip}'][aria-pressed='true']")
    tasks = page.evaluate(f"""() => [...document.querySelectorAll('{LB} thead th[data-task]')]
      .map(t => t.dataset.task)""")
    have = page.evaluate("DATA.accTasks")
    assert tasks == [t for t in want if t in have]
    # 11f: a chip is one group, so it has no group row — that row is All tasks'
    assert page.locator(f"{LB} thead tr.grp").count() == 0
    assert f"chip={chip}" in page.evaluate("location.hash")
    assert page.errors == []


def test_knowledge_shows_mmlu_and_mmlu_by_area(live, page):
    open_lb(page, live["base"])
    page.locator("[data-chip='knowledge']").click()
    page.wait_for_selector(f"{LB} thead th[data-area]")
    areas = page.evaluate(f"""() => [...document.querySelectorAll('{LB} thead th[data-area]')]
      .map(t => t.dataset.area)""")
    # 11e: an area with no number for any model on the page is not a column
    # (the fixture's MMLU subjects reach five of the eight)
    assert areas == page.evaluate("""() => Object.keys(DATA.meta.areas)
      .filter(a => visible().some(m => areaMmlu(m, a)))""") and len(areas) >= 1
    assert page.locator(f"{LB} thead th[data-task='mmlu']").count() == 1
    # the 24 per-topic MMLU columns are one tick away under Columns
    open_filters(page)                                     # 12b: in Filters ▾
    page.locator("#pill-columns").click()
    page.wait_for_selector("#pop-columns [data-column-group='cats'] input[data-column]")
    assert page.locator("#pop-columns [data-column-group='cats'] input[data-column]").count() >= 1
    page.keyboard.press("Escape")
    # an area's value is the item-weighted mean of its topics' MMLU scores
    ok = page.evaluate("""() => { const m = DATA.models.find(x => mmluCats(x));
      const a = Object.keys(DATA.meta.areas).find(a => areaMmlu(m, a));
      let s = 0, n = 0;
      for (const t of DATA.meta.areas[a]) { const g = mmluCats(m)[t];
        if (g && g.n_report) { s += g.score_report * g.n_report; n += g.n_report; } }
      return Math.abs(areaMmlu(m, a).v - s / n) < 1e-12; }""")
    assert ok
    assert page.errors == []


def test_judged_topics_is_live_with_a_calibrated_judge(live, page):
    # 12b: Judged topics is the Knowledge exam view, on the switch above the chips
    open_lb(page, live["base"])
    page.locator("[data-models-view='exam']").click()
    page.wait_for_selector(f"{LB} thead th[data-jarea]")
    assert page.locator(f"{LB} thead th[data-jarea]").count() == 8
    # an area mean needs half its topics judged; below that it says how many
    cells = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody td[data-jarea-cell]')]
      .map(t => [t.textContent.trim(), t.title])""")
    assert cells and all(t == "—" and "topics judged" in why or t != "—" for t, why in cells)
    assert page.errors == []


def test_judged_topics_is_disabled_and_says_why_while_provisional(live, page):
    # 12b: the Knowledge exam view is offered — the scores exist — and says in
    # one line why none is ranked, instead of an empty table
    uncalibrated(page)
    open_lb(page, live["base"])
    page.locator("[data-models-view='exam']").click()
    why = page.locator("[data-exam-off]")
    why.wait_for()
    assert "once a person has agreed with the judge" in why.text_content()
    assert "not calibrated" in why.text_content()
    assert page.locator(f"{LB}").count() == 0
    # an old pasted link asking for the chip lands on the same line
    page.goto(live["base"] + "/#tab=leaderboard&chip=judged")
    page.wait_for_selector("[data-exam-off]")
    open_lb(page, live["base"])
    # and the radar's judged source is off for the same reason
    src = page.locator("[data-radar-src='judged']")
    assert src.is_disabled() and "not calibrated" in src.get_attribute("title")
    assert page.errors == []


# ---------------------------------------------------------------------------
# the rank tint
# ---------------------------------------------------------------------------

def leads(page, col):
    """11f: which rows lead a column — its best, or inside its noise."""
    return page.evaluate(f"""() => Object.fromEntries([...document.querySelectorAll(
      '{LB} tbody tr[data-lb-row]')].map(tr => {{
        const i = [...tr.parentElement.closest('table').querySelectorAll('thead tr.names th')]
          .findIndex(th => th.dataset.col === '{col}');
        const td = tr.children[i];
        return [tr.dataset.lbRow, !!(td && td.dataset.lead)]; }}))""")


def test_the_leaders_are_the_whole_boards_and_a_filter_never_changes_them(live, page):
    open_lb(page, live["base"])
    before = leads(page, "hellaswag")
    assert before[TOP] and 1 <= sum(before.values()) < len(before)
    # narrow the rows to the bottom of the board: nobody there becomes a leader
    open_filters(page)
    page.locator("#pill-size").click()
    page.locator("#pop-size [data-choice='s']").click()
    page.wait_for_selector("#pill-size[data-value='s']")
    after = leads(page, "hellaswag")
    assert after and all(before[k] == v for k, v in after.items())
    assert TOP not in after
    assert "size=s" in page.evaluate("location.hash")
    assert page.errors == []


def test_a_cell_the_z_test_cannot_tell_from_the_best_is_a_leader_too(live, page):
    open_lb(page, live["base"])
    found = page.evaluate(f"""() => {{
      const out = [];
      for (const td of document.querySelectorAll('{LB} tbody td.lead[data-tip]'))
        if (JSON.parse(td.dataset.tip).includes('within the noise of the best'))
          out.push([(td.getAttribute('style') || ''), getComputedStyle(td.querySelector('b')).fontWeight]);
      return out; }}""")
    assert found, "the fixture has a pair inside the noise at the top of a column"
    assert all("--heat-3" in st and int(w) >= 700 for st, w in found)
    # and the z-test agrees: the pair is in DATA.sig as not significant
    assert page.evaluate("""() => DATA.accTasks.some(t => (DATA.sig[t] || []).some(r => !r[4]))""")
    assert page.errors == []


def test_no_provisional_cell_is_ever_tinted(live, page):
    uncalibrated(page)
    open_lb(page, live["base"])
    for chip in ("all", "knowledge"):
        page.locator(f"[data-chip='{chip}']").click()
        page.wait_for_selector(f"[data-chip='{chip}'][aria-pressed='true']")
        tinted = page.evaluate(f"""() => [...document.querySelectorAll(
          '{LB} tbody td[data-judged-avg], {LB} tbody td[data-jarea-cell]')]
          .filter(td => td.dataset.lead || (td.getAttribute('style') || '').includes('--heat')).length""")
        assert tinted == 0
    # no judged average exists for a provisional judge at all: judged_avg()
    assert page.evaluate("DATA.models.every(m => m.judgedAvg == null)")
    assert page.errors == []


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def test_every_row_is_one_line(live, page):
    open_lb(page, live["base"])
    hs = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody tr[data-lb-row]')]
      .map(tr => tr.getBoundingClientRect().height)""")
    assert hs and min(hs) >= 39.5 and max(hs) <= 48, hs
    assert page.errors == []


def test_a_row_opens_the_model_page_from_the_keyboard(live, page):
    # 12b: a row opens the model page, not a panel in place — by the mouse or
    # by the keyboard, and Back returns to the table
    open_lb(page, live["base"])
    row = page.locator(f"tr[data-lb-row='{TOP}']")
    assert row.get_attribute("tabindex") == "0"
    row.focus()
    page.keyboard.press("Enter")
    page.wait_for_selector("[data-model-hero]")
    assert page.evaluate("state.model") == TOP
    page.go_back()
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")
    assert page.errors == []


# ---------------------------------------------------------------------------
# popovers and the toolbar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pill,panel", [("#pill-columns", "#pop-columns"),
                                        ("#pill-models", "#pop-models"),
                                        ("#pill-kind", "#pop-kind")])
def test_a_popover_stays_open_across_a_poll(live, page, pill, panel):
    open_lb(page, live["base"])
    open_filters(page)                                     # 12b: the pills are in Filters ▾
    page.locator(pill).click()
    page.wait_for_selector(panel)
    page.wait_for_timeout(6000)
    assert page.locator(panel).count() == 1
    assert page.locator(pill).get_attribute("aria-expanded") == "true"
    assert page.errors == []


def test_the_models_popover_narrows_the_rows_and_says_how_many(live, page):
    open_lb(page, live["base"])
    open_filters(page)
    page.locator("#pill-models").click()
    page.wait_for_selector("#pop-models")
    assert page.locator("[data-models-foot]").text_content() == "All models shown"
    page.locator("#pop-models").get_by_role("button", name="Clear").click()
    page.locator(f"#pop-models [data-model-pick='{TOP}']").check()
    page.locator("#pop-models [data-model-pick='fx/good-750m']").check()
    n = page.evaluate("DATA.models.length")
    assert page.locator("[data-models-foot]").text_content() == f"2 of {n} shown"
    # each model carries its family's colour — the legend of the family bar
    assert page.locator("#pop-models .famdot").count() == n
    page.locator("[data-models-apply]").click()
    page.wait_for_function(f"document.querySelectorAll('{LB} tbody tr[data-lb-row]').length === 2")
    assert page.locator("#pill-models").text_content() == "Models: 2 ▾"     # 12h.2
    assert page.errors == []


def test_the_family_bar_is_never_colour_alone(live, page):
    open_lb(page, live["base"])
    td = page.locator(f"tr[data-lb-row='{TOP}'] td.model")
    assert "--fam" in (td.get_attribute("style") or "")
    assert "family:" in td.get_attribute("title")
    # 12b: and in words, as the Family column under Filters ▾ ▸ Columns
    open_filters(page)
    page.locator("#pill-columns").click()
    page.locator("#pop-columns input[data-column='family']").check()
    fam = page.locator(f"tr[data-lb-row='{TOP}'] td[data-fact='family']")
    fam.wait_for()
    assert fam.text_content() == page.evaluate(f"famOf(DATA.models.find(m => m.id === '{TOP}'))")
    page.evaluate("localStorage.removeItem('bench-lb-facts')")
    assert page.errors == []


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

def test_the_frontier_ignores_gaps_inside_the_noise(live, page):
    open_lb(page, live["base"])
    on = page.evaluate("""() => frontierOf([
        {m: {id: 'small'}, x: 1e8, y: 0.50, se: 0.03},
        {m: {id: 'big'},   x: 1e9, y: 0.46, se: 0.03}]).map(p => p.m.id)""")
    assert on == ["small", "big"]                  # 4 points, well inside the noise
    off = page.evaluate("""() => frontierOf([
        {m: {id: 'small'}, x: 1e8, y: 0.50, se: 0.005},
        {m: {id: 'big'},   x: 1e9, y: 0.40, se: 0.005}]).map(p => p.m.id)""")
    assert off == ["small"]                        # a real gap: the bigger one is off it
    # without standard errors nothing can be called real, and nothing is dropped
    assert page.evaluate("""() => frontierOf([{m: {id: 'a'}, x: 1, y: 0.9, se: null},
        {m: {id: 'b'}, x: 2, y: 0.1, se: null}]).length""") == 2
    assert page.errors == []


def test_insights_has_the_frontier_weakest_topics_and_the_radar(live, page):
    open_lb(page, live["base"])
    ins = page.locator("[data-insights]")
    ins.wait_for()
    assert ins.locator("h2").text_content() == "Insights"
    assert ins.locator("[data-frontier] svg [data-point]").count() >= 5
    assert ins.locator("[data-frontier-line]").count() == 1
    # preliminary models are not plotted, and the caption says how many
    n_pre = page.evaluate("DATA.models.filter(m => officialAvg(m) == null && !m.duplicateOf).length")
    if n_pre:
        assert f"{n_pre} preliminary not shown" in ins.locator("[data-frontier]").text_content()
    # a point is reachable by keyboard and selects on Enter
    pt = ins.locator("[data-point]").first
    pt.focus()
    page.keyboard.press("Enter")
    page.wait_for_selector("[data-shade]")
    assert "bigger and score" in page.locator("[data-frontier-caption]").text_content()
    page.keyboard.press("Escape")
    page.wait_for_selector("[data-shade]", state="detached")
    # weakest topics: one model's own topics, weakest first, each a link
    bars = ins.locator("[data-weakest] [data-weak-topic]")
    assert bars.count() >= 1
    vals = page.evaluate("""() => [...document.querySelectorAll('[data-weakest] [data-weak-topic]')]
      .map(a => +JSON.parse(a.dataset.tip)[1].split(' ')[0])""")
    assert vals == sorted(vals)
    assert bars.first.get_attribute("href").startswith("#topic=")
    # the radar: chips, not a compare column
    assert page.locator(f"{LB} th.cmp, {LB} input[aria-label^='compare']").count() == 0
    page.locator("#pill-radar-add").click()
    page.locator(f"#pop-radar-add [data-radar-pick='{TOP}']").click()
    page.wait_for_selector(f"[data-radar-chip='{TOP}']")
    assert ins.locator("[data-radar] svg.radar").count() == 1
    page.locator("[data-radar-src='areas']").click()
    page.wait_for_selector("[data-radar-src='areas'][aria-pressed='true']")
    assert page.errors == []


def test_weakest_topics_says_so_when_nothing_is_judged(live, page):
    def handle(route):
        r = route.fetch()
        body = r.json()
        for m in body["models"]:
            m["judge"] = None
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)
    open_lb(page, live["base"])
    assert page.locator("[data-weakest-empty]").text_content() == \
        "No model has been judged yet — Improve ▸ By topic ▸ Sit the exam"
    assert page.errors == []


# ---------------------------------------------------------------------------
# the phone, and the pictures
# ---------------------------------------------------------------------------

def test_at_400px_the_model_column_is_pinned_and_nothing_overflows(live, page):
    page.set_viewport_size({"width": 400, "height": 860})
    open_lb(page, live["base"])
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    # the header sits above the first row, not on top of it
    head = page.locator(f"{LB} thead tr:not(.grp)").bounding_box()
    first = page.locator(f"{LB} tbody tr[data-lb-row]").first.bounding_box()
    assert head["y"] + head["height"] <= first["y"] + 1, (head, first)
    td = page.locator(f"tr[data-lb-row='{TOP}'] td.model")
    rank = page.locator(f"tr[data-lb-row='{TOP}'] td.rank")
    assert td.evaluate("e => getComputedStyle(e).position") == "sticky"
    before = (rank.bounding_box()["x"], td.bounding_box()["x"])
    wrap = page.locator(".lb-wrap.stick")
    assert wrap.evaluate("e => e.scrollWidth > e.clientWidth")    # there is something to scroll
    wrap.evaluate("e => e.scrollLeft = 400")
    page.wait_for_timeout(100)
    # the rank and the model stay exactly where they were; the scores move
    assert abs(rank.bounding_box()["x"] - before[0]) < 1
    assert abs(td.bounding_box()["x"] - before[1]) < 1
    assert page.errors == []


def test_screenshots_for_the_pr(live, page):
    SCREENS.mkdir(parents=True, exist_ok=True)
    for width in (1280, 400):
        page.set_viewport_size({"width": width, "height": 900})
        open_lb(page, live["base"])
        page.wait_for_timeout(300)
        page.screenshot(path=SCREENS / f"leaderboard-{width}-light.png", full_page=True)
        open_lb(page, live["base"], "&chip=knowledge")
        page.wait_for_timeout(300)
        page.screenshot(path=SCREENS / f"leaderboard-knowledge-{width}-light.png", full_page=True)
    page.set_viewport_size({"width": 1280, "height": 900})
    open_lb(page, live["base"])
    open_filters(page)
    page.locator("#pill-columns").click()
    page.wait_for_selector("#pop-columns")
    page.screenshot(path=SCREENS / "leaderboard-columns-1280-light.png")
    page.keyboard.press("Escape")
    page.locator("[data-insights]").scroll_into_view_if_needed()
    page.locator("[data-point]").first.click()
    page.wait_for_selector("[data-shade]")
    page.locator("[data-insights]").screenshot(path=SCREENS / "leaderboard-insights-1280-light.png")
    assert page.errors == []
