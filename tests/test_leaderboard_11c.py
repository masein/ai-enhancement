"""11c: the Leaderboard. Topic-group chips, "Label: Value ▾" pills on the
shared popover, two header rows, one-line cells tinted by their rank on the
whole board, rows that open in place, and Insights under the table.

The rules that do not bend are asserted here too: no provisional score is
tinted, averaged or put on a frontier, and a tie the z-test cannot break is
shown as one.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

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
    # the group sits over its columns, in the first header row
    grp = page.locator(f"{LB} thead tr.grp th:not(.nogrp)").all_text_contents()
    assert grp == [{"commonsense": "Commonsense", "reasoning": "Reasoning", "math": "Math",
                    "truthfulness": "Truthfulness"}[chip]] if tasks else True
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
    open_lb(page, live["base"])
    chip = page.locator("[data-chip='judged']")
    assert chip.is_enabled()
    chip.click()
    page.wait_for_selector(f"{LB} thead th[data-jarea]")
    assert page.locator(f"{LB} thead th[data-jarea]").count() == 8
    # an area mean needs half its topics judged; below that it says how many
    cells = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody td[data-jarea-cell]')]
      .map(t => [t.textContent.trim(), t.title])""")
    assert cells and all(t == "—" and "topics judged" in why or t != "—" for t, why in cells)
    assert page.errors == []


def test_judged_topics_is_disabled_and_says_why_while_provisional(live, page):
    uncalibrated(page)
    open_lb(page, live["base"])
    chip = page.locator("[data-chip='judged']")
    # 11e: aria-disabled, so a click can say why; the reason is the tooltip,
    # the description, and a note the click opens — not a permanent line
    assert chip.get_attribute("aria-disabled") == "true"
    why = chip.get_attribute("title")
    assert "once a person has agreed with the judge" in why and "not calibrated" in why
    assert page.locator("[data-why='judged-chip']").text_content() == why
    assert chip.get_attribute("aria-describedby") == "why-judged-chip"
    look = chip.evaluate("b => [getComputedStyle(b).cursor, +getComputedStyle(b).opacity]")
    assert look[0] == "not-allowed" and look[1] < 1
    # a pasted hash asking for it lands on All tasks, not on an empty table
    open_lb(page, live["base"], "&chip=judged")
    assert page.locator("[data-chip='all'][aria-pressed='true']").count() == 1
    # and the radar's judged source is off for the same reason
    src = page.locator("[data-radar-src='judged']")
    assert src.is_disabled() and "not calibrated" in src.get_attribute("title")
    assert page.errors == []


# ---------------------------------------------------------------------------
# the rank tint
# ---------------------------------------------------------------------------

def steps(page, col):
    return page.evaluate(f"""() => Object.fromEntries([...document.querySelectorAll(
      '{LB} tbody tr[data-lb-row]')].map(tr => {{
        const i = [...tr.parentElement.closest('table').querySelectorAll('thead tr:not(.grp) th')]
          .findIndex(th => th.dataset.col === '{col}');
        const td = tr.children[i];
        return [tr.dataset.lbRow, td ? td.dataset.step || null : null]; }}))""")


def test_the_tint_is_the_whole_boards_rank_and_a_filter_never_changes_it(live, page):
    open_lb(page, live["base"])
    before = steps(page, "avg")
    ranked = [k for k, v in before.items() if v]
    assert len(ranked) >= 5
    assert before[TOP] == "5"                              # the leader is the top step
    assert set(before.values()) - {None} <= {"1", "2", "3", "4", "5"}
    # narrow the rows to the bottom of the board: the colours stay
    page.locator("#pill-size").click()
    page.locator("#pop-size [data-choice='s']").click()
    page.wait_for_selector("#pill-size[data-value='s']")
    after = steps(page, "avg")
    assert after and all(before[k] == v for k, v in after.items())
    assert TOP not in after
    assert "size=s" in page.evaluate("location.hash")
    assert page.errors == []


def test_a_cell_the_z_test_cannot_tell_from_the_best_shares_the_top_step(live, page):
    open_lb(page, live["base"])
    found = page.evaluate(f"""() => {{
      const out = [];
      for (const td of document.querySelectorAll('{LB} tbody td.tiebest[data-step]'))
        out.push([td.dataset.step, getComputedStyle(td.querySelector('b')).fontWeight]);
      return out; }}""")
    assert found, "the fixture has a pair inside the noise at the top of a column"
    assert all(s == "5" and int(w) >= 700 for s, w in found)
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
          .filter(td => td.dataset.step).length""")
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


def test_an_opened_row_survives_polls_paging_sorting_and_a_pasted_hash(live, page):
    open_lb(page, live["base"])
    row = page.locator(f"tr[data-lb-row='{TOP}']")
    row.locator("td.model .se, td.num").first.click()
    page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']")
    assert row.get_attribute("class").split().count("open") == 1
    btn = row.locator("button.disclose")
    assert btn.get_attribute("aria-expanded") == "true"
    assert btn.get_attribute("aria-controls") == page.locator(
        f"tr.detail[data-lb-detail='{TOP}']").get_attribute("id")
    # the detail row holds its blocks
    detail = page.locator(f"tr.detail[data-lb-detail='{TOP}']")
    text = detail.text_content()
    for eyebrow in ("Tasks", "Links"):
        assert eyebrow in text
    assert "Open model page →" in text and "#1/" in text
    # two polls
    page.wait_for_timeout(11000)
    assert page.locator(f"tr.detail[data-lb-detail='{TOP}']").count() == 1
    # a re-sort
    page.locator(f"{LB} thead th[data-col='params']").click()
    page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']")
    # a page change, and back (ten fixture models fit one page, so the page
    # size is set the way the pager's own select sets it)
    page.evaluate("state.pg.leaderboard.size = 5; state.pg.leaderboard.page = 2; render()")
    page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']", state="detached")
    page.evaluate("state.pg.leaderboard.page = 1; render()")
    page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']")
    page.evaluate("state.pg.leaderboard.size = 25; render()")
    # the hash holds it: a pasted link opens it again, in a fresh page
    h = page.evaluate("location.hash")
    assert f"open={quote(TOP, safe='')}" in h
    page.goto("about:blank")
    page.goto(live["base"] + "/" + h)
    page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']")
    # a click on a link keeps the link's own job
    assert page.locator(f"tr[data-lb-row='{TOP}'] a.mname").get_attribute("href").startswith("#model=")
    assert page.errors == []


def test_a_row_opens_from_the_keyboard(live, page):
    open_lb(page, live["base"])
    btn = page.locator(f"tr[data-lb-row='{TOP}'] button.disclose")
    btn.focus()
    page.keyboard.press("Enter")
    page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']")
    page.locator(f"tr[data-lb-row='{TOP}'] button.disclose").focus()
    page.keyboard.press(" ")
    page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']", state="detached")
    assert page.errors == []


def test_a_provisional_row_shows_its_topics_grey_with_the_line_once_and_no_area_mean(live, page):
    uncalibrated(page)
    open_lb(page, live["base"])
    judged = page.evaluate("""() => DATA.models.find(m => Object.keys((m.judge || {}).tasks || {})
      .some(t => t.startsWith('exam_'))).id""")
    page.locator(f"tr[data-lb-row='{judged}'] button.disclose").click()
    detail = page.locator(f"tr.detail[data-lb-detail='{judged}']")
    detail.wait_for()
    assert detail.locator("[data-provisional-line]").count() == 1
    assert "provisional" in detail.locator("[data-provisional-line]").text_content()
    assert "not ranked" in detail.locator("[data-provisional-line]").text_content()
    chips = detail.locator(".tchip")
    assert chips.count() >= 1 and chips.count() == detail.locator(".tchip.grey").count()
    # no area mean: the chips are topics, each with its own score, and nothing else
    assert not detail.locator("[data-area-mean]").count()
    assert page.errors == []


# ---------------------------------------------------------------------------
# popovers and the toolbar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pill,panel", [("#pill-columns", "#pop-columns"),
                                        ("#pill-models", "#pop-models"),
                                        ("#pill-kind", "#pop-kind")])
def test_a_popover_stays_open_across_a_poll(live, page, pill, panel):
    open_lb(page, live["base"])
    page.locator(pill).click()
    page.wait_for_selector(panel)
    page.wait_for_timeout(6000)
    assert page.locator(panel).count() == 1
    assert page.locator(pill).get_attribute("aria-expanded") == "true"
    assert page.errors == []


def test_the_models_popover_narrows_the_rows_and_says_how_many(live, page):
    open_lb(page, live["base"])
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
    assert "2 of" in page.locator("#pill-models").text_content()
    assert page.errors == []


def test_the_family_bar_is_never_colour_alone(live, page):
    open_lb(page, live["base"])
    td = page.locator(f"tr[data-lb-row='{TOP}'] td.model")
    assert "--fam" in (td.get_attribute("style") or "")
    assert "family:" in td.get_attribute("title")
    page.locator(f"tr[data-lb-row='{TOP}'] button.disclose").click()
    assert "family" in page.locator(f"tr.detail[data-lb-detail='{TOP}']").text_content()
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
        "No model has been judged yet — Loop ▸ Sit the exam"
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
        open_lb(page, live["base"], f"&open={quote(TOP, safe='')}")
        page.wait_for_selector(f"tr.detail[data-lb-detail='{TOP}']")
        page.wait_for_timeout(300)
        page.screenshot(path=SCREENS / f"leaderboard-open-{width}-light.png", full_page=True)
        open_lb(page, live["base"], "&chip=knowledge")
        page.wait_for_timeout(300)
        page.screenshot(path=SCREENS / f"leaderboard-knowledge-{width}-light.png", full_page=True)
    page.set_viewport_size({"width": 1280, "height": 900})
    open_lb(page, live["base"])
    page.locator("#pill-columns").click()
    page.wait_for_selector("#pop-columns")
    page.screenshot(path=SCREENS / "leaderboard-columns-1280-light.png")
    page.keyboard.press("Escape")
    page.locator("[data-insights]").scroll_into_view_if_needed()
    page.locator("[data-point]").first.click()
    page.wait_for_selector("[data-shade]")
    page.locator("[data-insights]").screenshot(path=SCREENS / "leaderboard-insights-1280-light.png")
    assert page.errors == []
