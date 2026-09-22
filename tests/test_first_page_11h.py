"""11h: the first page, and the findings of the 11e live check.

One font rule for the board, a compact hero with no repeated title, highlight
cards with a sans number; one scale for an area's number wherever it is; no
page that scrolls sideways; the phone's filters behind one button; a LIVE
time that moves with each check; plain words on every main tab; and no
"Sandbox run".
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.request
from pathlib import Path

import pytest

from conftest import label_domains

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11h"
LB = "table[data-lb-table]"
MODEL = "fx/good-750m"
BANNED = ["diagnosis half", "DIAGNOSIS", "report half", "report-half", "over a provisional judge",
          "LLM", "13-gram", "batch items", "exam_", "local/chat", "judge.json", "κ"]
# where mono is allowed: numbers in tables, cards and charts, column headers,
# eyebrows and section indices, badges, model ids, the LIVE badge and the log
MONO_OK = "td.num, th, .eyebrow, .secidx, .badge, .mid, .livebadge, .rd-log, svg"


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def new_page(browser, width, height=900, **kw):
    ctx = browser.new_context(viewport={"width": width, "height": height},
                              reduced_motion="reduce", **kw)
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.errors = errors
    return ctx, pg


# ---------------------------------------------------------------------------
# 1. the font rule, the hero, the cards
# ---------------------------------------------------------------------------

MONO_TEXT = """ok => { const out = [];
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let t; while ((t = w.nextNode())) {
    if (!t.textContent.trim()) continue;
    const e = t.parentElement;
    if (!e || !e.getClientRects().length || e.closest('script, style, #tip')) continue;
    if (!/monospace|Menlo|SF Mono|Consolas/i.test(getComputedStyle(e).fontFamily)) continue;
    if (e.closest(ok)) continue;
    out.push(e.tagName.toLowerCase() + '.' + [...e.classList].join('.') + ': '
      + t.textContent.trim().slice(0, 30)); }
  return out; }"""


@pytest.mark.parametrize("where", ["/", "/#tab=leaderboard", "/#model=fx%2Fgood-750m"])
def test_mono_is_only_for_numbers_ids_labels_and_the_log(live, page, where):
    page.set_viewport_size({"width": 1512, "height": 900})
    page.goto(live["base"] + where)
    page.wait_for_selector("#view .card")
    page.wait_for_timeout(1200)
    assert page.evaluate(MONO_TEXT, MONO_OK) == []
    # and words really are sans: the stats line, the pills, a card's link
    fams = page.evaluate("""() => ['[data-statline]', '.pill', '.chip-btn', '.hcard-link', 'footer',
        '#themeBtn'].map(s => document.querySelector(s)).filter(Boolean)
        .map(e => getComputedStyle(e).fontFamily)""")
    assert fams and not [f for f in fams if re.search("monospace|Menlo", f)]
    assert page.errors == []


@pytest.mark.parametrize("width", [1280, 1512])
def test_the_overview_opens_on_a_compact_hero_the_cards_and_the_top_models(live, browser, width):
    ctx, page = new_page(browser, width, 868)
    try:
        page.goto(live["base"] + "/")
        page.wait_for_selector("[data-highlights] [data-hl='judge']")
        geo = page.evaluate("""() => { const h = document.getElementById('pagehero');
          const r = el => el.getBoundingClientRect();
          return { hero: r(h).height, h1: !!h.querySelector('h1') && r(h.querySelector('h1')).width > 1,
            cards: [...document.querySelectorAll('[data-hl]')].map(c => [r(c).bottom, r(c).height]),
            links: [...document.querySelectorAll('.hcard-link')].map(l => Math.round(r(l).bottom)),
            row: r(document.querySelector('[data-top-models] tbody tr')).bottom,
            submit: r(document.querySelector('[data-submit-model]')), eb: r(document.getElementById('heroEyebrow')) }; }""")
        assert geo["hero"] <= 120, geo
        assert not geo["h1"]                             # the bar carries the title
        # Submit a model sits on the hero's right, beside its text
        assert geo["submit"]["left"] > geo["eb"]["left"] + 400
        assert all(b <= 868 for b, _ in geo["cards"]) and geo["row"] <= 868, geo
        assert len({round(h) for _, h in geo["cards"]}) == 1                 # one height
        assert len(set(geo["links"])) == 1                                   # one baseline
        look = page.evaluate("""() => { const c = e => getComputedStyle(document.querySelector(e));
          const v = c('.hcard-v'), n = c('.hcard-name'), d = c('.hcard-verdict'), l = c('.hcard-link');
          return [v.fontSize, v.fontWeight, v.fontVariantNumeric, v.fontFamily, n.fontSize, n.fontWeight,
                  n.textOverflow, d.fontSize, l.textDecorationLine, l.fontFamily]; }""")
        assert look[0] == "28px" and look[1] == "700" and "tabular-nums" in look[2]
        assert not re.search("monospace", look[3])
        assert look[4] == "14px" and look[5] == "600" and look[6] == "ellipsis"
        assert look[7] == "14px" and look[8] == "none" and not re.search("monospace", look[9])
        assert page.errors == []
    finally:
        ctx.close()


def test_a_top_models_row_is_washed_end_to_end_on_hover(live, page):
    page.goto(live["base"] + "/")
    row = page.locator("[data-top-models] tbody tr").nth(2)            # not a leader
    row.hover()
    bgs = row.evaluate("tr => [...tr.children].map(td => getComputedStyle(td).backgroundColor)")
    assert len(set(bgs)) == 1 and bgs[0] not in ("rgba(0, 0, 0, 0)", "transparent"), bgs
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2. one scale for an area's number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", ["chance", "raw"])
def test_an_areas_number_is_the_same_in_its_column_and_its_bar(live, page, scale):
    page.set_viewport_size({"width": 1512, "height": 900})
    model = "fx/good-750m-tuned-skill"
    page.goto(live["base"] + f"/#tab=leaderboard&chip=knowledge&open={model}")
    page.wait_for_selector(f"{LB} tr[data-lb-detail] [data-area-bar]")
    if scale == "raw":
        page.locator("#pill-scale").click()
        page.locator("#pop-scale [data-choice='raw']").click()
        page.wait_for_selector("[data-bars-scale='raw']")
    pairs = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tr[data-lb-detail] [data-area-bar]')]
      .map(b => {{ const a = b.dataset.areaBar;
        const td = document.querySelector(`{LB} tr[data-lb-row="{model}"] td[data-area-cell="${{a}}"]`);
        return [a, b.querySelector('.mb-v').textContent, td && td.textContent.trim()]; }})""")
    assert pairs and all(bar == col for _, bar, col in pairs), pairs
    words = "raw accuracy" if scale == "raw" else "above chance"
    assert words in page.locator("[data-bars-scale]").text_content()
    tip = page.locator(f"{LB} thead th[data-area]").first.get_attribute("data-tip")
    assert words in tip
    assert page.errors == []


# ---------------------------------------------------------------------------
# 3. no page scrolls sideways
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [1280, 1512, 1920])
def test_no_chip_makes_the_page_scroll_sideways(live, browser, width):
    ctx, page = new_page(browser, width)
    try:
        for chip in ("all", "knowledge", "commonsense", "reasoning", "math", "truthfulness", "judged"):
            page.goto(live["base"] + f"/#tab=leaderboard&chip={chip}")
            page.wait_for_selector(f"{LB} tbody tr")
            page.wait_for_timeout(150)
            assert page.evaluate("document.documentElement.scrollWidth") <= width, chip
        # every column shown on Knowledge: the table is wider than the card, so
        # it scrolls in its own box, with # and Model pinned, and the fade says so
        page.goto(live["base"] + "/#tab=leaderboard&chip=knowledge")
        page.locator("#pill-columns").click()
        page.locator("#pop-columns [data-show-all]").click()
        page.keyboard.press("Escape")
        page.wait_for_selector("[data-hkeep='lb'].hscroll")
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        assert page.locator("[data-hfade='lb']").get_attribute("data-more") == "1"
        pinned = page.evaluate(f"""() => {{ const s = document.querySelector('[data-hkeep=lb]');
          const td = document.querySelector('{LB} tbody td.model'), x0 = td.getBoundingClientRect().left;
          s.scrollLeft = 400; return [x0, td.getBoundingClientRect().left]; }}""")
        assert abs(pinned[0] - pinned[1]) <= 1
        assert page.errors == []
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# 4. the phone's filters
# ---------------------------------------------------------------------------

def test_on_a_phone_the_filters_are_one_button_and_the_table_starts_near_the_top(live, browser):
    ctx, page = new_page(browser, 400, 860, is_mobile=True, has_touch=True)
    try:
        page.goto(live["base"] + "/#tab=leaderboard")
        page.wait_for_selector(f"{LB} tbody tr")
        gap = page.evaluate(f"""() => document.querySelector('{LB} tbody tr').getBoundingClientRect().top
          - document.querySelector('[data-lb-card]').getBoundingClientRect().top""")
        assert gap <= 320, gap
        assert page.locator("#pill-kind").count() == 0            # folded away
        chips = page.evaluate("""() => [...document.querySelectorAll('.lbbar .chip-btn')]
          .map(b => Math.round(b.getBoundingClientRect().top))""")
        assert len(set(chips)) == 1                                # one row, that scrolls
        btn = page.locator("[data-filters]")
        assert btn.text_content() == "Filters ▾"
        btn.click()
        sheet = page.locator("[data-filter-sheet]")
        sheet.wait_for()
        for pill in ("#pill-kind", "#pill-size", "#pill-status", "#pill-columns", "#pill-models",
                     "#pill-scale"):
            assert sheet.locator(pill).count() == 1, pill
        shot(page, "11h-4-filters-400-light.png")
        sheet.locator("#pill-kind").click()
        page.locator("#pop-kind [data-choice='base']").click()
        page.wait_for_selector("[data-filters='1']")
        assert page.locator("[data-filters]").text_content() == "Filters · 1 ▾"
        page.locator("[data-filters-done]").click()
        page.wait_for_selector("[data-filter-sheet]", state="detached")
        assert page.errors == []
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# 5. the LIVE time moves with each check
# ---------------------------------------------------------------------------

def test_the_live_badge_moves_with_each_check_and_turns_stale(live, browser):
    ctx, page = new_page(browser, 1400)
    try:
        start = dt.datetime(2026, 9, 22, 17, 44, 0)
        page.clock.install(time=start)
        page.goto(live["base"] + "/")
        page.wait_for_selector("#liveBadge:not([hidden])")
        page.wait_for_function("liveBadge.textContent.includes('17:44')")
        assert "data last changed" in page.locator("#liveBadge").get_attribute("title")
        # two minutes on: a check lands, and the badge says so
        page.clock.fast_forward("02:00")
        page.wait_for_function("liveBadge.textContent.includes('17:46')", timeout=15000)
        assert page.locator("#liveBadge").text_content().startswith("LIVE · ")
        # the service goes away: after two missed intervals, amber and STALE
        page.route("**/api/**", lambda route: route.abort())
        for _ in range(6):
            page.clock.fast_forward("00:06")
            page.wait_for_timeout(100)
        page.wait_for_function("liveBadge.dataset.fresh === 'stale'", timeout=15000)
        assert page.locator("#liveBadge").text_content() == "STALE · 17:46"
        assert page.errors == [] or all("Failed to fetch" in e or "net::" in e for e in page.errors)
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# 7. plain words, and 8. no Sandbox run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_board(live):
    """Proposals, a dataset and runs on the board, so every main tab has its
    busiest words on screen."""
    from service import db
    base = live["base"]
    label_domains(live["root"] / "exam", "Economics")
    pid = api(base, "/api/proposals", {"model": MODEL, "topic": "Economics", "requested_by": "Omar"})["id"]
    for _ in range(150):
        if api(base, f"/api/proposals/{pid}")["status"] == "proposed":
            break
        time.sleep(0.15)
    api(base, "/api/proposals", {"model": "fx/skewed-360m", "topic": "Economics", "requested_by": "Omar"})
    api(base, f"/api/proposals/{pid}/approve", {"approver": "Omar"})
    did = api(base, f"/api/proposals/{pid}/generate", {"requester": "Omar", "count": 4})["dataset_id"]
    for _ in range(200):
        if api(base, f"/api/datasets/{did}")["status"] == "ready":
            break
        time.sleep(0.15)
    s1 = db.add(MODEL, "auto", "judged", "omar", "", tasks=["exam_law"])
    db.update(s1, status="failed", progress="failed on: judge", error="judge: LocalUnreachable: nothing")
    s2 = db.add("org/running", "auto", "quick", "omar", "")
    db.update(s2, status="running", progress="1/4 · arc_easy")
    return {"pid": pid, "did": did}


@pytest.mark.parametrize("where", ["/", "/#tab=loop", "/#tab=models", "/#tab=leaderboard",
                                   "/#tab=queue", "/#tab=review", "/#topic=economics",
                                   "/#model=fx%2Fgood-750m"])
def test_no_banned_phrase_is_on_a_main_tab(live, page, full_board, where):
    """The words a person reads, in plain language (11h §7). Details, tooltips,
    the Reader's raw views and the Provenance tab keep the technical names."""
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + where)
    page.wait_for_selector("#view .card")
    page.wait_for_timeout(2200)
    text = page.evaluate("document.body.innerText")
    found = [b for b in BANNED
             if re.search(re.escape(b), text, 0 if b in ("DIAGNOSIS", "LLM") else re.I)]
    assert found == [], [(b, text[max(0, text.lower().find(b.lower()) - 60):][:140]) for b in found]
    assert page.errors == []


def test_there_is_no_sandbox_run(live, page):
    page.goto(live["base"] + "/")
    page.locator("#moreBtn").click()
    page.wait_for_selector("[data-pop='more']")
    assert "Sandbox run" not in page.locator("[data-pop='more']").text_content()
    assert page.locator("#demoItem, [data-demo]").count() == 0
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(live["base"] + "/demo", timeout=10)
    assert e.value.code == 404
    assert page.errors == []


# ---------------------------------------------------------------------------
# the screenshots for the PR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1280, 1512, 400])
def test_screenshots_for_the_pr(live, browser, width, theme):
    ctx, page = new_page(browser, width, 868 if width > 500 else 860)
    try:
        for name, where in (("overview", "/"), ("leaderboard", "/#tab=leaderboard"),
                            ("knowledge", "/#tab=leaderboard&chip=knowledge"),
                            ("model", "/#model=fx%2Fgood-750m")):
            page.goto(live["base"] + where)
            page.wait_for_selector("#view .card")
            page.wait_for_timeout(700)
            page.evaluate(f"applyTheme('{theme}')")
            page.wait_for_timeout(200)
            shot(page, f"11h-{name}-{width}-{theme}.png")
        assert page.errors == []
    finally:
        ctx.close()
