"""11e: the second live check, in the browser.

Where the documents go is shown before Approve; the bar holds its tabs at
every desktop width; the badge tells the time; the model cell and the opened
row keep their text to themselves; a stale diagnosis says so instead of
drawing empty bars; a poll keeps focus in an open panel; the frontier counts
models of the same size; the highlight values are one number each; a phone
sees scores, a short header and readable charts; and the polish.

The phase-11 rules still hold here: only diagnose-half items are ever read
for a generation request, and no provisional judged score is averaged,
tinted, ranked or put on a frontier.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

from conftest import label_domains, set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11e"
LB = "table[data-lb-table]"
TOP = "fx/good-750m-tuned-skill"
E2E_MS = 30000
OLD15 = ["economics", "law", "medicine & health", "mathematics", "computer science",
         "physics & engineering", "chemistry & biology", "history", "philosophy & religion",
         "politics & government", "psychology & sociology", "business & accounting",
         "geography & world facts", "language & logic", "other"]


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def served(page, edit, pattern="**/api/results*"):
    """The same payload as served, edited on the way to the page."""
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route(pattern, handle)


def open_lb(page, base, frag=""):
    page.goto(base + "/#tab=leaderboard" + frag)
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def boxes_intersect(a, b, slack=0.5):
    return not (a["x"] + a["width"] <= b["x"] + slack or b["x"] + b["width"] <= a["x"] + slack
                or a["y"] + a["height"] <= b["y"] + slack or b["y"] + b["height"] <= a["y"] + slack)


def wait_proposed(page, base, pid):
    for _ in range(150):
        if api(base, f"/api/proposals/{pid}")["status"] == "proposed":
            return
        page.wait_for_timeout(200)
    raise AssertionError(f"proposal {pid} never reached 'proposed'")


# ---------------------------------------------------------------------------
# 1. the plan is shown before Approve, and Approve freezes it
# ---------------------------------------------------------------------------

def test_the_plan_is_shown_before_approve_and_approve_freezes_it(live, page):
    base, root = live["base"], live["root"]
    path, was = label_domains(root / "exam", "Economics")
    try:
        pid = api(base, "/api/proposals", {"model": "fx/good-750m", "topic": "Economics",
                                           "requested_by": "Omar"})["id"]
        wait_proposed(page, base, pid)
        live_plan = api(base, f"/api/proposals/{pid}/focus?count=20")
        assert not live_plan["frozen"] and live_plan["mode"] == "area"
        want = Counter(live_plan["labels"][:20])

        page.set_viewport_size({"width": 1280, "height": 900})
        page.goto(base + "/#tab=review")
        page.wait_for_selector(".card h2:has-text('Review')")
        set_name(page, "Omar")
        box = page.locator(f"[data-plan-decide='{pid}']")
        box.wait_for(timeout=E2E_MS)
        line = box.locator("[data-plan-stage='decide']")
        page.wait_for_function("el => el.textContent.length > 0", arg=line.element_handle(),
                               timeout=E2E_MS)
        # masein's condition: the spread is on the card before anyone approves
        text = line.text_content()
        assert text.startswith(f"Documents will cover {len(want)} areas: ")
        for d, n in want.items():
            assert f"{d} {n}" in text
        assert line.get_attribute("data-focus-mode") == "area"
        spread = box.locator(f"[data-spread='{pid}']")
        assert spread.is_checked() and spread.is_enabled()
        assert "Spread the documents over these" in box.text_content()
        box.scroll_into_view_if_needed()
        shot(page, "11e-1-plan-before-approve-1280-light.png")

        # approved as shown: the plan is frozen, exactly the one on the card
        page.locator(f".rv[data-proposal='{pid}'] button:has-text('Approve this spec')").click()
        page.wait_for_selector(f"[data-focus-plan='{pid}'][data-plan-stage='generate']",
                               timeout=E2E_MS)
        frozen = api(base, f"/api/proposals/{pid}/focus?count=20")
        assert frozen["frozen"] and frozen["mode"] == "area"
        assert frozen["labels"] == live_plan["labels"]
        assert page.errors == []
    finally:
        path.write_bytes(was)


def test_unticking_spread_approves_a_plan_with_no_focus(live, page):
    base, root = live["base"], live["root"]
    path, was = label_domains(root / "exam", "Economics")
    try:
        pid = api(base, "/api/proposals", {"model": "fx/good-750m", "topic": "Economics",
                                           "requested_by": "Omar"})["id"]
        wait_proposed(page, base, pid)
        page.goto(base + "/#tab=review")
        page.wait_for_selector(".card h2:has-text('Review')")
        set_name(page, "Omar")
        spread = page.locator(f"[data-spread='{pid}']")
        spread.wait_for(timeout=E2E_MS)
        page.wait_for_function("el => !el.disabled", arg=spread.element_handle(), timeout=E2E_MS)
        spread.uncheck()
        page.wait_for_timeout(5500)                          # a poll: the box stays unticked
        assert not page.locator(f"[data-spread='{pid}']").is_checked()
        page.locator(f".rv[data-proposal='{pid}'] button:has-text('Approve this spec')").click()
        line = page.locator(f"[data-focus-plan='{pid}'][data-plan-stage='generate']")
        line.wait_for(timeout=E2E_MS)
        page.wait_for_function("el => el.textContent.length > 0", arg=line.element_handle(),
                               timeout=E2E_MS)
        frozen = api(base, f"/api/proposals/{pid}/focus?count=20")
        assert frozen["mode"] == "off" and frozen["labels"] == []
        assert frozen["reason"] == "the approver turned spreading off"
        assert line.text_content() == "Not spread — the approver turned spreading off."
        assert page.errors == []
    finally:
        path.write_bytes(was)


# ---------------------------------------------------------------------------
# 2. the bar at every desktop width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [1280, 1440, 1512, 1920])
def test_every_tab_is_itself_at_its_centre_and_the_strip_never_scrolls(live, page, width):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/")
    page.wait_for_selector("#tabs [role='tab']")
    page.wait_for_selector("#warnings summary[data-warn-summary]")
    hits = page.evaluate("""() => [...document.querySelectorAll('#tabs [role="tab"], #moreBtn')]
      .map(t => { const r = t.getBoundingClientRect();
        const at = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
        return [t.textContent.trim(), !!at && (at === t || t.contains(at))]; })""")
    assert hits and all(ok for _, ok in hits), hits
    sw, cw = page.evaluate("[tabs.scrollWidth, tabs.clientWidth]")
    assert sw == cw
    # nothing in the bar overlaps anything else
    parts = page.evaluate("""() => ['.bar-title', '#liveBadge', '#tabs', '#warnings', '#who',
        '#themeBtn'].map(s => document.querySelector(s)).filter(e => e && e.offsetParent)
      .map(e => { const r = e.getBoundingClientRect();
        return {s: e.id || e.className, x: r.x, y: r.y, width: r.width, height: r.height}; })""")
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            assert not boxes_intersect(a, b), (a, b)
    # the pill is one button among the bar's buttons: the same height and type
    pill = page.locator("#warnings summary[data-warn-summary]")
    n = int(pill.get_attribute("data-warn-summary"))
    assert pill.text_content().strip() == f"{n} check{'s' if n != 1 else ''} ▾"
    look = page.evaluate("""() => [document.querySelector('#warnings summary'), themeBtn,
        document.querySelector('#who button')].filter(Boolean).map(e => { const c = getComputedStyle(e);
        return [Math.round(e.getBoundingClientRect().height), c.fontFamily, c.fontSize, c.fontWeight,
                c.borderTopLeftRadius, c.borderTopWidth]; })""")
    assert len({json.dumps(x) for x in look}) == 1, look
    # and they sit on one line: the same top and bottom
    ys = page.evaluate("""() => [document.querySelector('#warnings summary'), themeBtn,
        document.querySelector('#who button')].filter(Boolean)
      .map(e => [Math.round(e.getBoundingClientRect().top), Math.round(e.getBoundingClientRect().bottom)])""")
    assert len({json.dumps(y) for y in ys}) == 1, ys
    if width == 1512:
        shot(page, "11e-2-bar-1512-light.png", clip={"x": 0, "y": 0, "width": width, "height": 60})
        page.locator("#warnings summary[data-warn-summary]").click()
        judged = page.locator("[data-checks-judged]")
        assert re.fullmatch(r"\d+ of \d+ are about the judged suite", judged.text_content().strip())
        shot(page, "11e-2-checks-open-1512-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 3. the badge tells the time
# ---------------------------------------------------------------------------

def test_the_live_badge_and_the_status_line_show_the_refresh_time(live, page):
    served(page, lambda b: b.update(generated="2026-09-22 15:17 +04"))
    open_lb(page, live["base"])
    page.wait_for_function("() => liveBadge.textContent.includes('15:17')")
    badge = page.locator("#liveBadge").text_content()
    assert "15:17" in badge and "+04" not in badge
    if page.locator("#liveBadge").get_attribute("data-fresh") == "ok":
        assert badge.strip() == "LIVE · 15:17"
    line = page.locator(".statusline").first.text_content()
    assert "live 15:17" in line and "+04" not in line
    assert page.errors == []


# ---------------------------------------------------------------------------
# 4 and 5. the model cell and the opened row keep their text to themselves
# ---------------------------------------------------------------------------

def _with_a_long_duplicate(body):
    ranked = sorted((m for m in body["models"] if m.get("avg") is not None),
                    key=lambda m: -m["avg"])
    twin, dup = ranked[0], ranked[1]
    twin["name"] = "qwen35-delta-moe-7d560104-step945-v2"
    dup.update(name="qwen35-delta-moe-7d560104-step945-v2-resubmitted",
               duplicateOf=twin["id"], duplicateOfName=twin["name"],
               duplicateWhy="every score and item count is identical")
    twin.setdefault("archinfo", {}).update(active_params=908_000_000, experts=64,
                                           experts_per_tok=8)
    twin["params"] = 2_300_000_000


@pytest.mark.parametrize("width", [1280, 1512])
def test_the_model_cells_children_stay_inside_the_cell(live, page, width):
    served(page, _with_a_long_duplicate)
    page.set_viewport_size({"width": width, "height": 900})
    open_lb(page, live["base"], "&status=all")
    page.wait_for_selector(f"{LB} [data-dup-toggle]")
    out = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody td.model')].flatMap(td => {{
        const c = td.getBoundingClientRect();
        return [...td.querySelectorAll('.mcell > *')].filter(k => k.offsetParent).map(k => {{
          const r = k.getBoundingClientRect();
          return [td.dataset.model, k.className || k.tagName,
                  r.left >= c.left - 0.5 && r.right <= c.right + 0.5]; }}); }})""")
    assert out and all(ok for *_, ok in out), [x for x in out if not x[2]]
    # and the duplicate toggle is inside the name's cell, not over Params
    tog = page.locator(f"{LB} [data-dup-toggle]").first
    td = tog.locator("xpath=ancestor::td[1]")
    assert "model" in td.get_attribute("class")
    if width == 1512:
        shot(page, "11e-4-duplicate-row-1512-light.png",
             clip={"x": 0, "y": 0, "width": width, "height": 700})
    assert page.errors == []


@pytest.mark.parametrize("width", [1280, 1512])
def test_the_opened_rows_blocks_never_intersect(live, page, width):
    served(page, _with_a_long_duplicate)
    page.set_viewport_size({"width": width, "height": 1000})
    open_lb(page, live["base"], "&status=all&open=" + TOP)
    page.wait_for_selector(f"{LB} tr[data-lb-detail] .dblock")
    blocks = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tr[data-lb-detail] .dblock')]
      .map(b => {{ const r = b.getBoundingClientRect();
        return {{e: b.querySelector('.eyebrow')?.textContent, x: r.x, y: r.y, width: r.width,
                 height: r.height}}; }})""")
    assert len(blocks) >= 3
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            assert not boxes_intersect(a, b), (a, b)
    # nothing inside a block paints past its right edge
    over = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tr[data-lb-detail] .dblock')]
      .filter(b => b.scrollWidth > b.clientWidth + 1).map(b => b.querySelector('.eyebrow')?.textContent)""")
    assert over == []
    # the opened duplicate twin says so in its Links
    open_lb(page, live["base"], "&status=all")
    tw = page.evaluate("DATA.models.find(m => m.name === 'qwen35-delta-moe-7d560104-step945-v2').id")
    page.goto(live["base"] + "/#tab=leaderboard&status=all&open=" + tw)
    line = page.locator(f"[data-dup-line='{tw}']")
    line.wait_for()
    assert line.text_content().startswith("Same run as qwen35-delta-moe-7d560104-step945-v2-resubmitted")
    if width == 1512:
        page.locator(f"{LB} tr[data-lb-detail]").scroll_into_view_if_needed()
        shot(page, "11e-5-opened-row-1512-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 6. a stale diagnosis says so
# ---------------------------------------------------------------------------

def _old_categories(body):
    for m in body["models"]:
        mm = (((m.get("diag") or {}).get("tasks") or {}).get("mmlu") or {})
        if mm.get("categories"):
            vals = list(mm["categories"].values())
            mm["categories"] = {name: vals[i % len(vals)] for i, name in enumerate(OLD15)}


def test_a_stale_diagnosis_says_so_once_and_draws_nothing_empty(live, page):
    served(page, _old_categories)
    page.set_viewport_size({"width": 1512, "height": 1000})
    open_lb(page, live["base"], "&chip=knowledge")
    note = page.locator("[data-lb-card] [data-stale-diag='1']")
    note.wait_for()
    assert note.text_content() == ("MMLU by area needs a fresh diagnosis — the diagnoses on this "
                                   "board were made with the old 15 categories.")
    assert page.locator("[data-stale-diag='1']").count() == 1
    # no area column, since every model's would be empty
    assert page.locator(f"{LB} thead th[data-area]").count() == 0
    # an opened row says it in place of eight empty bars
    shot(page, "11e-6-stale-diagnosis-1512-light.png", clip={"x": 0, "y": 0, "width": 1512, "height": 700})
    page.goto(live["base"] + "/#tab=leaderboard&chip=knowledge&open=" + TOP)
    blk = page.locator(f"{LB} tr[data-lb-detail] [data-stale-diag='{TOP}']")
    blk.wait_for()
    assert blk.text_content() == ("MMLU by area needs a fresh diagnosis — this one was made with "
                                  "the old 15 categories.")
    assert page.locator(f"{LB} tr[data-lb-detail] .minibars [data-area-bar], "
                        f"{LB} tr[data-lb-detail] .minibars").count() == 0
    assert page.errors == []


def test_a_fresh_diagnosis_shows_every_area_it_has_a_number_for(live, page):
    """The fixture's MMLU subjects reach five of the eight areas; the live
    board's reach all eight. Either way, an area column is there exactly when
    some model on the page has a number in it."""
    open_lb(page, live["base"], "&chip=knowledge")
    page.wait_for_selector(f"{LB} thead th[data-area]")
    shown = page.evaluate(f"""() => [...document.querySelectorAll('{LB} thead th[data-area]')]
      .map(t => t.dataset.area)""")
    have = page.evaluate("""() => Object.keys(DATA.meta.areas)
      .filter(a => visible().some(m => areaMmlu(m, a)))""")
    assert shown == have and len(shown) >= 1
    for a in shown:
        vals = page.evaluate(f"""() => {{ const i = [...document.querySelectorAll('{LB} thead tr:not(.grp) th')]
            .findIndex(t => t.dataset.area === {json.dumps(a)});
          return [...document.querySelectorAll('{LB} tbody tr[data-lb-row]')]
            .map(r => r.children[i].textContent.trim()); }}""")
        assert any(v != "—" for v in vals), a
    assert page.locator("[data-stale-diag]").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# 7. a poll keeps focus inside an open panel
# ---------------------------------------------------------------------------

def test_a_render_keeps_focus_on_the_same_checkbox_in_columns(live, page):
    open_lb(page, live["base"])
    page.locator("#pill-columns").click()
    page.wait_for_selector("[data-pop='columns'] input[data-column]")
    box = page.locator("[data-pop='columns'] input[data-column]").nth(2)
    key = box.get_attribute("data-column")
    box.focus()
    page.evaluate("document.activeElement._mark = 'mine'")
    page.evaluate("render()")
    assert page.evaluate("document.activeElement.dataset.column") == key
    assert page.evaluate("document.activeElement._mark") == "mine"       # the same node
    # and the keyboard goes on working after a render: Space toggles it,
    # the table follows, the focus stays
    was = page.evaluate("document.activeElement.checked")
    page.keyboard.press("Space")
    page.wait_for_function("was => document.activeElement.checked === !was", arg=was)
    page.evaluate("render()")
    assert page.evaluate("document.activeElement.dataset.column") == key
    assert page.evaluate("document.activeElement.checked") == (not was)
    shown = page.evaluate(f"[...document.querySelectorAll('{LB} thead th[data-col]')].map(t => t.dataset.col)")
    assert (key in shown) == (not was)
    assert page.errors == []


# ---------------------------------------------------------------------------
# 8. the frontier counts models of the same size
# ---------------------------------------------------------------------------

def test_of_two_models_the_same_size_only_the_better_is_on_the_frontier(live, page):
    open_lb(page, live["base"])
    on = page.evaluate("""() => frontierOf([
        {m: {id: 'a'}, x: 7.5e8, y: 0.60, se: 0.01},
        {m: {id: 'b'}, x: 7.5e8, y: 0.50, se: 0.01},
        {m: {id: 'c'}, x: 1.5e9, y: 0.70, se: 0.01}]).map(p => p.m.id)""")
    assert on == ["a", "c"]
    # inside the noise, both stay marked
    tie = page.evaluate("""() => frontierOf([
        {m: {id: 'a'}, x: 7.5e8, y: 0.51, se: 0.03},
        {m: {id: 'b'}, x: 7.5e8, y: 0.50, se: 0.03}]).map(p => p.m.id)""")
    assert tie == ["a", "b"]
    assert page.errors == []


def test_the_frontier_line_goes_through_the_best_point_at_each_size(live, page):
    def same_size(body):
        ranked = sorted((m for m in body["models"] if m.get("avg") is not None
                         and not m.get("duplicateOf") and m.get("params")), key=lambda m: -m["avg"])
        a, b, c = ranked[0], ranked[1], ranked[2]
        for m in (a, b, c):
            m["params"] = 750_000_000
        a["avg"], b["avg"], c["avg"] = 0.60, 0.52, 0.50
        for m in (a, b, c):
            m["avgSe"] = 0.01
    served(page, same_size)
    page.set_viewport_size({"width": 1280, "height": 900})
    open_lb(page, live["base"])
    page.wait_for_selector("[data-frontier] [data-point]")
    at = page.evaluate("""() => { const pts = [...document.querySelectorAll('[data-frontier] [data-point]')]
        .filter(p => DATA.models.find(m => m.id === p.dataset.point).params === 750000000);
      return pts.map(p => [p.dataset.point, !!p.dataset.onFrontier, +p.getAttribute('cy')]); }""")
    assert len(at) == 3
    assert sum(1 for _, on, _ in at if on) == 1
    best = min(at, key=lambda x: x[2])                    # the highest point
    assert best[1]
    # the line never runs straight up: one point per size
    d = page.locator("[data-frontier-line]").get_attribute("d")
    xs = [p.split(",")[0] for p in d[1:].split("L")]
    assert len(xs) == len(set(xs))
    # the biggest ranked model: no model here is bigger and scores lower
    big = page.evaluate("""() => { const ps = [...document.querySelectorAll('[data-frontier] [data-point]')];
      return ps.sort((a, b) => +b.getAttribute('cx') - +a.getAttribute('cx'))[0].dataset.point; }""")
    page.locator(f"[data-frontier] [data-point='{big}']").click()
    name = page.evaluate(f"DATA.models.find(m => m.id === {json.dumps(big)}).name")
    page.wait_for_function(
        "want => document.querySelector('[data-frontier-caption]').textContent === want",
        arg=f"No model here is bigger and scores lower than {name}.")
    page.locator("[data-frontier]").scroll_into_view_if_needed()
    shot(page, "11e-8-frontier-1280-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 9. the highlight values: the number, then the name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [1280, 1512])
def test_every_highlight_value_is_one_line(live, page, width):
    served(page, lambda b: next(m for m in sorted((m for m in b["models"] if m.get("avg") is not None
                                                   and not m.get("duplicateOf")), key=lambda m: -m["avg"]))
           .update(name="qwen35-delta-moe-7d560104-step945-v2"))
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-highlights] [data-hl='judge']")
    vals = page.evaluate("""() => [...document.querySelectorAll('[data-hl-value]')].map(v => {
        const c = getComputedStyle(v);
        return {k: v.dataset.hlValue, t: v.textContent, h: v.getBoundingClientRect().height,
                lh: parseFloat(c.lineHeight) || 1.2 * parseFloat(c.fontSize),
                fs: c.fontSize, ff: c.fontFamily, clipped: v.scrollWidth > v.clientWidth + 1}; })""")
    assert {v["k"] for v in vals} == {"best", "weakest", "loop", "judge"}
    for v in vals:
        assert v["h"] < 1.5 * v["lh"], v
        assert not v["clipped"], v
        assert v["fs"] == "28px" and "mono" in v["ff"].lower(), v
    by = {v["k"]: v["t"] for v in vals}
    assert re.fullmatch(r"\d+\.\d", by["best"])
    assert re.fullmatch(r"\d\.\d\d / 4", by["weakest"])
    assert re.fullmatch(r"\d+ / \d+", by["loop"])
    assert re.fullmatch(r"\d+ / \d+", by["judge"])
    # the name: its own line, cut with an ellipsis, whole in the tooltip
    name = page.locator("[data-hl-name='best']")
    assert name.text_content() == "qwen35-delta-moe-7d560104-step945-v2"
    look = name.evaluate("e => [getComputedStyle(e).fontSize, getComputedStyle(e).textOverflow, e.title]")
    assert look[0] == "16px" and look[1] == "ellipsis" and look[2].startswith("fx/")
    # the canary: two decimals
    v = page.locator("[data-verdict='judge']").text_content()
    for x in re.findall(r"(\d+\.\d+) from the (?:human marks|last run)", v):
        assert re.fullmatch(r"\d+\.\d\d", x), v
    if width == 1512:
        page.locator("[data-highlights]").scroll_into_view_if_needed()
        shot(page, "11e-9-highlights-1512-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 10. 400 px
# ---------------------------------------------------------------------------

@pytest.fixture
def phone(browser):
    ctx = browser.new_context(viewport={"width": 400, "height": 860}, device_scale_factor=2,
                              reduced_motion="reduce",
                              is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None)
    pg.errors = errors
    yield pg
    ctx.close()


def test_at_400px_the_header_is_two_rows_and_the_rest_is_behind_one_menu(live, phone):
    page = phone
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")
    h = page.evaluate("document.getElementById('bar').getBoundingClientRect().height")
    assert h <= 96, h
    # the checks, the name and the theme are one ⋯ away
    assert not page.locator("#themeBtn").is_visible()
    more = page.locator("#barMore")
    assert more.is_visible() and more.get_attribute("aria-expanded") == "false"
    shot(page, "11e-10-header-400-light.png", clip={"x": 0, "y": 0, "width": 400, "height": 120})
    more.click()
    assert more.get_attribute("aria-expanded") == "true"
    for sel in ("#warnings summary", "#who button", "#themeBtn"):
        assert page.locator(sel).first.is_visible(), sel
    shot(page, "11e-10-header-menu-400-light.png")
    page.keyboard.press("Escape")
    assert not page.locator("#themeBtn").is_visible()
    assert page.evaluate("document.activeElement.id") == "barMore"
    # the bar did not grow for having been opened
    assert page.evaluate("document.getElementById('bar').getBoundingClientRect().height") <= 96
    assert page.errors == []


def test_at_400px_the_leaderboard_shows_a_score(live, phone):
    page = phone
    served(page, _with_a_long_duplicate)
    page.goto(live["base"] + "/#tab=leaderboard&status=all")
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")
    geo = page.evaluate(f"""() => {{
      const sc = document.querySelector('[data-hkeep="lb"]'), s = sc.getBoundingClientRect();
      const row = document.querySelector('{LB} tbody tr[data-lb-row]');
      const pin = row.querySelector('td.model').getBoundingClientRect();
      const rank = row.querySelector('td.rank').getBoundingClientRect();
      // a score: a tinted score cell, not Params
      const nums = [...row.querySelectorAll('td.tcell')].map(td => td.getBoundingClientRect())
        .filter(r => r.left >= pin.right - 0.5 && r.right <= s.right + 0.5);
      return {{scroller: s.width, pinned: pin.right - rank.left, visible: nums.length}}; }}""")
    assert geo["pinned"] <= 0.45 * geo["scroller"] + 1, geo
    assert geo["visible"] >= 1, geo
    # of the badges, only "prelim" stays
    shown = page.evaluate(f"""() => [...document.querySelectorAll('{LB} td.model .mcell > *')]
      .filter(e => e.offsetParent && !e.classList.contains('mname')).map(e => e.className)""")
    assert all("prelim" in c for c in shown), shown
    # the right edge fades and says there is more
    fade = page.locator("[data-hfade='lb']")
    assert fade.get_attribute("data-more") == "1"
    assert page.locator("[data-hfade='lb'] .scrollhint").is_visible()
    assert page.locator("[data-hfade='lb'] .scrollhint").text_content() == "scroll →"
    shot(page, "11e-10-leaderboard-400-light.png")
    # scrolled to the end, the hint goes; a poll keeps the scroll
    page.evaluate("const s = document.querySelector('[data-hkeep=\"lb\"]'); s.scrollLeft = s.scrollWidth")
    page.wait_for_function("document.querySelector('[data-hfade=\"lb\"]').dataset.more === '0'")
    x = page.evaluate("document.querySelector('[data-hkeep=\"lb\"]').scrollLeft")
    page.evaluate("render()")
    assert page.evaluate("document.querySelector('[data-hkeep=\"lb\"]').scrollLeft") == x
    assert page.errors == []


def test_at_400px_no_chart_text_is_under_12px(live, phone):
    page = phone
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("[data-insights] [data-frontier] svg")
    small = page.evaluate("""() => [...document.querySelectorAll('#view svg text')]
      .filter(t => t.getClientRects().length && t.closest('svg').getBoundingClientRect().width)
      .map(t => [t.textContent, parseFloat(getComputedStyle(t).fontSize) * t.getScreenCTM().a])
      .filter(([, px]) => px < 11.95)""")
    assert small == [], small[:5]
    # weakest topics are HTML rows here, not a shrunken SVG
    assert not page.locator("[data-weakest] svg.weakest").is_visible()
    rows = page.locator("[data-weakest] [data-weak-row]")
    assert rows.count() >= 1 and rows.first.is_visible()
    fs = rows.first.evaluate("r => Math.min(...[...r.querySelectorAll('span')].filter(s => s.textContent)"
                             ".map(s => parseFloat(getComputedStyle(s).fontSize)))")
    assert fs >= 12
    # the scatter keeps its width in its own sideways scroller
    w = page.evaluate("document.querySelector('[data-frontier] svg').getBoundingClientRect().width")
    assert w >= 520
    assert page.evaluate("document.documentElement.scrollWidth") <= 400
    page.locator("[data-insights]").scroll_into_view_if_needed()
    shot(page, "11e-10-insights-400-light.png", full_page=False)
    assert page.errors == []


def test_at_desktop_no_chart_text_is_under_12px(live, page):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("[data-insights] [data-frontier] svg")
    small = page.evaluate("""() => [...document.querySelectorAll('[data-insights] svg text')]
      .filter(t => t.getClientRects().length)
      .map(t => [t.textContent, parseFloat(getComputedStyle(t).fontSize) * t.getScreenCTM().a])
      .filter(([, px]) => px < 11.95)""")
    assert small == [], small[:5]
    # and the charts do not scroll at a desktop width, two to a row
    assert page.evaluate("""() => [...document.querySelectorAll('[data-insights] .chartscroll')]
      .filter(e => e.offsetParent).every(e => e.scrollWidth <= e.clientWidth + 1)""")
    tops = page.evaluate("""() => ['[data-frontier]', '[data-weakest]']
      .map(s => Math.round(document.querySelector(s).getBoundingClientRect().top))""")
    assert tops[0] == tops[1]
    # the radar, with the longest labels (the areas): every label inside the chart
    page.evaluate(f"state.cmpSel = [{json.dumps(TOP)}]; lbSet({{radarSrc: 'areas'}})")
    page.wait_for_selector("[data-radar] svg.radar text")
    out = page.evaluate("""() => [...document.querySelectorAll('[data-insights] svg text')]
      .filter(t => t.getClientRects().length).filter(t => { const b = t.getBBox(),
        vb = t.closest('svg').viewBox.baseVal;
        return b.x < -0.5 || b.x + b.width > vb.width + 0.5; }).map(t => t.textContent)""")
    assert out == []
    assert page.errors == []


# ---------------------------------------------------------------------------
# 11. a dataset made before 11a says what is missing
# ---------------------------------------------------------------------------

def test_a_dataset_from_before_11a_says_two_are_missing_on_the_topic_page(live, page):
    def old_dataset(body):
        for r in body.get("topics") or []:
            if r.get("task") == "exam_economics":
                r["datasets"] = [{"id": 2, "status": "ready", "count": 20, "kept": 18,
                                  "missing": None, "over_provisional_judge": None}]
    served(page, old_dataset, "**/api/loop*")
    page.goto(live["base"] + "/#topic=economics")
    line = page.locator("[data-datasets-table] [data-doc-line='2']")
    line.wait_for(timeout=E2E_MS)
    assert line.text_content() == "18 of 20 documents · 2 missing — reasons not recorded (made before 11a)"
    assert page.errors == []


# ---------------------------------------------------------------------------
# 12. polish
# ---------------------------------------------------------------------------

def test_the_spacing_the_badge_and_the_links(live, page):
    page.set_viewport_size({"width": 1512, "height": 900})
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-statline]")
    gap = page.evaluate("""() => document.querySelector('[data-statline]').getBoundingClientRect().top
      - document.querySelector('[data-submit-model]').getBoundingClientRect().bottom""")
    assert abs(gap - 12) <= 1, gap
    open_lb(page, live["base"], "&open=" + TOP)
    ins = page.evaluate("""() => { const c = document.querySelector('[data-insights]');
      return c.querySelector('.igrid').getBoundingClientRect().top
        - c.querySelector(':scope > p.sub').getBoundingClientRect().bottom; }""")
    assert abs(ins - 16) <= 1, ins
    # Links: the buttons are links to look at, flush with the links above
    links = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tr[data-lb-detail] .dlinks > *')]
      .map(e => [e.tagName, Math.round(e.getBoundingClientRect().left), getComputedStyle(e).paddingLeft])""")
    assert len(links) >= 3
    assert len({x for _, x, _ in links}) == 1, links
    assert all(p == "0px" for t, _, p in links if t == "BUTTON"), links
    assert page.errors == []


def test_the_provisional_badge_is_the_standard_badge(live, page):
    def local_judge(body):
        for m in body["models"]:
            if m.get("judgeState"):
                m["judgeState"] = {**m["judgeState"], "ok": False,
                                   "reasons": ["a local judge, not calibrated — provisional"]}
    served(page, local_judge)
    open_lb(page, live["base"])
    b = page.locator("[data-provisional='weakest']")
    b.wait_for()
    h, fs = b.evaluate("e => [e.getBoundingClientRect().height, getComputedStyle(e).fontSize]")
    ref = page.evaluate(f"document.querySelector('{LB} .badge').getBoundingClientRect().height")
    assert fs == "12px" and h <= ref + 1, (h, ref)
    assert page.errors == []


def test_the_unavailable_judged_chip_says_why_on_click(live, page):
    def uncal(body):
        body["judged"]["calibration"] = None
        for m in body["models"]:
            if m.get("judgeState"):
                m["judgeState"] = {**m["judgeState"], "ok": False,
                                   "reasons": ["a local judge, not calibrated — provisional"]}
            m["judgedAvg"] = None
    served(page, uncal)
    open_lb(page, live["base"])
    chip = page.locator("[data-chip='judged']")
    # aria-disabled, not disabled: it still takes the click
    assert chip.get_attribute("aria-disabled") == "true"
    assert chip.evaluate("b => b.disabled") is False
    why = chip.get_attribute("title")
    assert "once a person has agreed with the judge" in why
    note = page.locator("#" + chip.get_attribute("aria-describedby"))
    assert note.text_content() == why
    # no permanent line under the toolbar
    assert not note.is_visible()
    chip.click(force=True)
    assert note.is_visible()
    # under the chip: below its row of chips, above the pills
    nb, cb = note.bounding_box(), chip.bounding_box()
    pb = page.locator("#pill-columns").bounding_box()
    assert cb["y"] + cb["height"] <= nb["y"] + 0.5 and nb["y"] + nb["height"] <= pb["y"] + 0.5
    h, lh = note.evaluate("e => [e.getBoundingClientRect().height, parseFloat(getComputedStyle(e).lineHeight)]")
    assert h < 1.5 * lh
    assert page.locator("[data-chip='all'][aria-pressed='true']").count() == 1
    shot(page, "11e-12-judged-chip-note-1280-light.png", clip={"x": 0, "y": 0, "width": 1240, "height": 420})
    chip.click(force=True)
    assert not note.is_visible()
    assert page.errors == []


# ---------------------------------------------------------------------------
# the screenshots for the PR: both themes, the three widths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, page, theme):
    page.set_viewport_size({"width": 1512, "height": 1000})
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-highlights] [data-hl='judge']")
    page.evaluate(f"applyTheme('{theme}')")
    shot(page, f"11e-overview-1512-{theme}.png")
    open_lb(page, live["base"], "&open=" + TOP)
    page.evaluate(f"applyTheme('{theme}')")
    shot(page, f"11e-leaderboard-1512-{theme}.png", full_page=True)
    assert page.errors == []
