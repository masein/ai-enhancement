"""11f: masein's seven — a board that is simple, calm and smooth to work in.

The one rule changed on purpose: the ± no longer sits on every cell. It is
the cell's tooltip, it is in the opened row, and "Show ± errors" puts it back
on every cell. Bold still means "best in the column or within its noise" —
the z-test's set, exactly — and no provisional score is bold, tinted,
averaged or ranked.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import date
from pathlib import Path

import pytest

from conftest import choice, model_tab, open_filters, open_submit

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11f"
LB = "table[data-lb-table]"
TOP = "fx/good-750m-tuned-skill"


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def served(page, edit, pattern="**/api/results*"):
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route(pattern, handle)


def open_lb(page, base, frag=""):
    page.goto(base + "/#tab=leaderboard" + frag)
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")


def results(base):
    with urllib.request.urlopen(base + "/api/results", timeout=20) as r:
        return json.loads(r.read())


def new_page(browser, **ctx):
    c = browser.new_context(viewport={"width": 1280, "height": 900}, **ctx)
    pg = c.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None)
    pg.errors = errors
    return c, pg


# every class a view's #view element or a .dwrap was given, as it happened
RECORD = """() => { window.__cls = [];
  const note = e => window.__cls.push((e.id === 'view' ? 'view:' : 'dwrap:') + e.className);
  new MutationObserver(ms => ms.forEach(m => {
    if (m.type === 'attributes') note(m.target);
    m.addedNodes && m.addedNodes.forEach(n => n.querySelectorAll
      && n.querySelectorAll('.dwrap').forEach(note));
  })).observe(document.body, { subtree: true, childList: true, attributes: true,
                               attributeFilter: ['class'] }); }"""


# ---------------------------------------------------------------------------
# 1a. a header one line tall, under a quiet group row
# ---------------------------------------------------------------------------

def test_the_header_is_one_line_of_names_under_a_quiet_group_row(live, page):
    page.set_viewport_size({"width": 1280, "height": 900})
    open_lb(page, live["base"])
    h = page.evaluate(f"document.querySelector('{LB} thead').getBoundingClientRect().height")
    assert h <= 64, h
    names = page.evaluate(f"""() => [...document.querySelectorAll('{LB} thead tr.names th')]
      .map(t => [t.textContent.trim(), t.getBoundingClientRect().height, t.dataset.tip || '',
                 getComputedStyle(t, '::after').content, t.dataset.col])""")
    for text, height, tip, after, col in names:
        assert "ⓘ" not in text and after in ("none", "normal", '""'), (text, after)
        assert height <= 40, (text, height)                  # one line
        assert tip, text                                     # every name has its setup
        assert " " not in text.replace(" ▼", "").replace(" ▲", "") or col.startswith(("area:", "j")), text
    tips = {col: json.loads(tip) for _, _, tip, _, col in names}
    for col in ("mmlu", "hellaswag", "arc_challenge"):
        if col in tips:
            assert re.search(r"\d+-shot", tips[col][0]), tips[col]
    assert "ARC-C" in [n[0] for n in names]
    # 12b: Updated is one of the model facts, a column under Filters ▾, off by default
    assert "UPDATED" not in [n[0].upper() for n in names]
    assert "above chance" in tips["avg"][0]                  # the scale left the header
    assert not page.locator(f"{LB} thead .unit").count()
    # the group row: muted, not the accent — and only on All tasks
    grp = page.locator(f"{LB} thead tr.grp th:not(.nogrp)").first
    muted = page.evaluate("""() => { const d = document.createElement('i');
      d.style.color = 'var(--muted)'; document.body.append(d);
      const c = getComputedStyle(d).color; d.remove(); return c; }""")
    assert grp.evaluate("e => getComputedStyle(e).color") == muted
    page.locator("[data-chip='commonsense']").click()
    page.wait_for_selector("[data-chip='commonsense'][aria-pressed='true']")
    assert page.locator(f"{LB} thead tr.grp").count() == 0
    # the caption says what bold means, once, under the table
    cap = page.locator("[data-lb-caption]")
    assert cap.text_content().startswith("Bold = best in the column or within its noise")
    assert page.errors == []


def test_dates_carry_a_year_only_when_it_is_not_this_one(live, page):
    open_lb(page, live["base"])
    # 12b: Last evaluated is a model fact under Filters ▾, off by default
    page.evaluate("localStorage.setItem('bench-lb-facts', '[\"date\"]'); render()")
    page.wait_for_selector(f"{LB} thead th[data-col='date']")
    i = page.evaluate(f"""[...document.querySelectorAll('{LB} thead tr.names th')]
      .findIndex(t => t.dataset.col === 'date')""")
    dates = page.evaluate(f"""i => [...document.querySelectorAll('{LB} tbody tr[data-lb-row]')]
      .map(r => r.children[i].textContent.trim())""", i)
    this_year = date.today().year
    for d in dates:
        m = re.fullmatch(r"(\d{1,2}) ([A-Z][a-z]{2})(?: (\d{4}))?", d)
        assert m, d
        assert (m.group(3) is None) == (int(m.group(3) or this_year) == this_year), d
    page.evaluate("localStorage.removeItem('bench-lb-facts')")
    assert page.errors == []


def test_the_judged_column_is_absent_while_every_value_would_be_a_dash(live, page):
    def none(body):
        for m in body["models"]:
            m["judgedAvg"] = None
    served(page, none)
    open_lb(page, live["base"])
    assert page.locator(f"{LB} thead th[data-col='javg']").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# 1b. cells: the number, bold for the leaders, ± a hover away
# ---------------------------------------------------------------------------

def test_no_error_on_a_cell_by_default_and_the_switch_puts_it_back(live, page):
    page.evaluate("() => { try { localStorage.removeItem('bench-lb-se'); } catch (e) {} }")
    open_lb(page, live["base"])
    cells = page.locator(f"{LB} tbody td.tcell")
    assert "±" not in page.locator(f"{LB} tbody").text_content()
    # …it is in the tooltip, on hover and on focus
    tips = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody td.tcell[data-tip]')]
      .map(td => JSON.parse(td.dataset.tip)[0])""")
    assert tips and sum("±" in t for t in tips) >= len(tips) // 2
    cells.first.focus()
    page.wait_for_function("getComputedStyle(document.querySelector('#tip')).opacity === '1'")
    assert "±" in page.locator("#tip").text_content()
    # the switch, in Columns (12b: under Filters ▾), remembered in this browser
    open_filters(page)
    page.locator("#pill-columns").click()
    page.locator("#pop-columns [data-show-se]").check()
    page.wait_for_function(f"document.querySelector('{LB} tbody').textContent.includes('±')")
    page.keyboard.press("Escape")
    page.reload()
    page.wait_for_selector(f"{LB} tbody tr[data-lb-row]")
    assert "±" in page.locator(f"{LB} tbody").text_content()
    open_filters(page)
    page.locator("#pill-columns").click()
    page.locator("#pop-columns [data-show-se]").uncheck()
    page.wait_for_function(f"!document.querySelector('{LB} tbody').textContent.includes('±')")
    assert page.errors == []


def test_the_bold_cells_are_exactly_the_z_tests_best_or_tied_set(live, page):
    page.set_viewport_size({"width": 1600, "height": 900})
    open_lb(page, live["base"])
    body = results(live["base"])
    ranked = {m["id"] for m in body["models"] if not m.get("duplicateOf")}
    heads = page.evaluate(f"""() => [...document.querySelectorAll('{LB} thead tr.names th')]
      .map(t => t.dataset.col)""")
    rows = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody tr[data-lb-row]')]
      .map(r => [r.dataset.lbRow, [...r.children].map(td => !!td.dataset.lead)])""")
    checked = 0
    for t in body["accTasks"]:
        if t not in heads:
            continue
        cells = body["cells"].get(t, {})
        pool = {m: c["v"] for m, c in cells.items() if m in ranked and c.get("v") is not None}
        best = max(pool, key=pool.get)
        tied = {best} | {b if a == best else a for a, b, _d, _z, ok in body["sig"].get(t, [])
                         if best in (a, b) and not ok}
        i = heads.index(t)
        bold = {mid for mid, flags in rows if flags[i]}
        assert bold == {m for m in tied if m in {r[0] for r in rows}}, (t, bold, tied)
        checked += 1
    assert checked >= 3
    # only the bold cells are tinted, and every bold cell is (Tint is on)
    look = page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody td.tcell')]
      .map(td => {{ const b = td.querySelector('b');
        return [!!td.dataset.lead, (td.getAttribute('style') || '').includes('--heat'),
                +(b ? getComputedStyle(b) : getComputedStyle(td)).fontWeight]; }})""")
    assert all(lead == tinted for lead, tinted, _w in look)
    assert all((w >= 700) == lead for lead, _t, w in look)
    # no glyphs in cells any more
    assert not re.search("[●≈]", page.locator(f"{LB} tbody").text_content())
    assert page.errors == []


# ---------------------------------------------------------------------------
# 4. a new page starts at the top; Back returns to where you were
# ---------------------------------------------------------------------------

def test_a_new_page_starts_at_the_top_and_back_returns_to_where_you_were(live, page):
    # 12b: rows no longer open in place. 12g.1: the tall page with topic links
    # is the Knowledge exam's rubrics table, 37 topics long (By topic is gone)
    page.set_viewport_size({"width": 1280, "height": 600})
    page.goto(live["base"] + "/#tab=benchmarks&sub=exam")
    page.wait_for_selector("[data-topic-link]")
    # a link well down the page, brought to 300px from the top: clear of the
    # sticky bar, so the click does not make the browser scroll first
    y = page.evaluate("""() => { const a = [...document.querySelectorAll('[data-topic-link]')].pop();
      return Math.round(a.getBoundingClientRect().top + scrollY - 300); }""")
    assert y > 600, y
    page.evaluate(f"window.scrollTo(0, {y})")
    page.wait_for_timeout(300)                            # the scroll is saved on the entry
    assert abs(page.evaluate("Math.round(scrollY)") - y) <= 2
    page.locator("[data-topic-link]").last.click()
    page.wait_for_selector("[data-topic-back]")
    assert page.evaluate("Math.round(scrollY)") == 0
    page.go_back()
    page.wait_for_selector("[data-topic-link]")
    page.wait_for_function(f"Math.abs(scrollY - {y}) <= 50", timeout=5000)
    assert page.errors == []


def test_a_sort_a_page_or_a_poll_does_not_move_the_scroll(live, page):
    page.set_viewport_size({"width": 1280, "height": 700})
    open_lb(page, live["base"])
    page.evaluate("window.scrollTo(0, 300)")
    y = page.evaluate("Math.round(scrollY)")
    page.evaluate("state.sort = { key: 'params', dir: -1 }; render()")
    assert page.evaluate("Math.round(scrollY)") == y
    page.evaluate("render()")
    assert page.evaluate("Math.round(scrollY)") == y
    page.evaluate("lbSet({ chip: 'commonsense' })")
    assert abs(page.evaluate("Math.round(scrollY)") - y) <= 2
    assert page.errors == []


# ---------------------------------------------------------------------------
# 5. one motion system
# ---------------------------------------------------------------------------

def test_the_motion_tokens_exist_and_are_zero_under_reduced_motion(live, browser):
    c0, page = new_page(browser, reduced_motion="no-preference")
    page.goto(live["base"] + "/")
    toks = page.evaluate("""() => { const c = getComputedStyle(document.body);
      return ['--dur-1', '--dur-2', '--dur-3', '--ease'].map(k => c.getPropertyValue(k).trim()); }""")
    assert toks == ["120ms", "180ms", "240ms", "cubic-bezier(.2, .8, .2, 1)"]
    c0.close()
    ctx, pg = new_page(browser, reduced_motion="reduce")
    try:
        pg.goto(live["base"] + "/")
        zero = pg.evaluate("""() => { const c = getComputedStyle(document.body);
          return ['--dur-1', '--dur-2', '--dur-3'].map(k => c.getPropertyValue(k).trim()); }""")
        assert all(z in ("0ms", "0s") for z in zero), zero
    finally:
        ctx.close()


def test_every_transition_moves_only_opacity_transform_rows_or_colour(live, page):
    page.goto(live["base"] + "/")
    props = page.evaluate("""() => { const out = [];
      const walk = rules => { for (const r of rules) {
        if (r.cssRules) walk(r.cssRules);
        if (r.style && r.style.transitionProperty) out.push([r.selectorText, r.style.transitionProperty]);
      } };
      for (const s of document.styleSheets) walk(s.cssRules);
      return out; }""")
    ok = {"opacity", "transform", "grid-template-rows", "color", "background-color", "border-color",
          "none", "all"}
    bad = [(sel, p) for sel, ps in props for p in (x.strip() for x in ps.split(","))
           if p not in ok]
    assert props and not bad, bad
    # "all" appears nowhere: only the theme cross-fade names its colours
    assert not [sel for sel, ps in props if "all" in ps.split(", ")]


def test_a_navigation_plays_the_entrance_and_a_poll_does_not(live, browser):
    ctx, page = new_page(browser, reduced_motion="no-preference")
    page.goto(live["base"] + "/#tab=overview")
    page.wait_for_selector("[data-needs-you]")
    page.evaluate(RECORD)
    page.locator("#tabs [data-tab='models']").click()
    page.wait_for_selector(f"{LB} tbody tr")
    assert any(c.startswith("view:") and "view-enter" in c for c in page.evaluate("window.__cls"))
    page.wait_for_timeout(400)
    page.evaluate("window.__cls = []")
    page.evaluate("render()")
    assert not any("view-enter" in c for c in page.evaluate("window.__cls"))
    assert page.errors == []
    ctx.close()


def test_a_value_a_poll_changed_is_washed_and_one_it_did_not_is_not(live, page):
    n = {"polls": 0}

    def bump(body):
        n["polls"] += 1
        if n["polls"] > 1:
            body["cells"]["hellaswag"][TOP]["v"] = 0.5
        body["generated"] = f"{body['generated']} #{n['polls']}"
    served(page, bump)
    open_lb(page, live["base"])
    page.evaluate("""() => { window.__changed = new Set();
      new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(n => n.querySelectorAll
        && n.querySelectorAll('.changed').forEach(e => window.__changed.add(e.dataset.watch)))))
        .observe(document.getElementById('view'), { subtree: true, childList: true });
      const o = new MutationObserver(ms => ms.forEach(m => m.target.classList.contains('changed')
        && window.__changed.add(m.target.dataset.watch)));
      o.observe(document.getElementById('view'), { subtree: true, attributes: true,
                                                   attributeFilter: ['class'] }); }""")
    # the board is fetched again when something finished: ask for it as a poll does
    page.evaluate("refreshResults()")
    page.wait_for_function("window.__changed.size > 0", timeout=15000)
    changed = page.evaluate("[...window.__changed]")
    assert changed == [f"lb|{TOP}|hellaswag"], changed
    assert page.errors == []


# ---------------------------------------------------------------------------
# 6. one pattern for row actions
# ---------------------------------------------------------------------------

@pytest.fixture
def queue_rows(live, monkeypatch):
    from service import config, db
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    rows = {}
    rows["queued"] = db.add("org/queued-one", "auto", "quick", "omar", "")
    rows["running"] = db.add("org/running-one", "auto", "quick", "omar", "")
    db.update(rows["running"], status="running", progress="1/4 · arc_easy", gpu_seconds=75)
    rows["failed"] = db.add("org/failed-one", "auto", "quick", "omar", "")
    db.update(rows["failed"], status="failed", gpu_seconds=372,
              error=("arc_easy: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.65 GiB "
                     "total capacity; 21.10 GiB already allocated; 1.02 GiB free; 21.50 GiB "
                     "reserved in total by PyTorch) If reserved memory is >> allocated memory "
                     "try setting max_split_size_mb to avoid fragmentation. See documentation "
                     "for Memory Management and PYTORCH_CUDA_ALLOC_CONF"))
    rows["judge"] = db.add("fx/good-750m", "auto", "judged", "omar", "")
    db.update(rows["judge"], status="failed", progress="failed on: judge", tasks='["exam_law"]',
              error="judge: LocalUnreachable: nothing is answering at http://host.docker.internal:8000/v1")
    rows["done"] = db.add("fx/good-750m", "base", "quick", "omar", "")
    db.update(rows["done"], status="done", gpu_seconds=61)
    return rows


def test_every_queue_row_shows_one_button_and_a_menu(live, page, queue_rows):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(live["base"] + "/#tab=queue")
    page.wait_for_selector(f"tr[data-queue-row='{queue_rows['done']}']")
    rows = page.evaluate("""() => [...document.querySelectorAll('[data-queue-table] tbody tr')]
      .map(tr => { const c = tr.querySelector('td.rowacts');
        const bs = [...c.querySelectorAll('button')].filter(b => b.offsetParent);
        const menu = bs.filter(b => b.dataset.rowMenu), main = bs.filter(b => !b.dataset.rowMenu);
        return { id: tr.dataset.queueRow, main: main.map(b => b.textContent), menu: menu.length,
                 looks: bs.map(b => { const s = getComputedStyle(b);
                   return [Math.round(b.getBoundingClientRect().height), s.borderTopLeftRadius]; }),
                 filled: main.filter(b => b.classList.contains('primary')).map(b => b.textContent) }; })""")
    assert len(rows) >= 5
    for r in rows:
        assert len(r["main"]) <= 1 and r["menu"] <= 1, r
        assert all(lk == [32, "6px"] for lk in r["looks"]), r
    filled = [f for r in rows for f in r["filled"]]
    assert filled == ["Retry grading"]
    by = {r["id"]: r for r in rows}
    assert by[str(queue_rows["queued"])]["main"] == ["Cancel"]
    assert by[str(queue_rows["failed"])]["main"] == ["Resubmit"]
    assert by[str(queue_rows["done"])]["main"] == ["Open results"]
    # the GPU cell is one line: "1 min", "6 min" or "—"
    gpu = page.evaluate("""() => [...document.querySelectorAll('[data-queue-table] tbody tr')]
      .map(tr => { const td = tr.children[7], r = document.createRange();
        r.selectNodeContents(td);
        return [td.textContent, new Set([...r.getClientRects()].map(x => Math.round(x.top))).size]; })""")
    for text, lines in gpu:
        assert re.fullmatch("\\d+\u00a0min|—", text), text
        assert lines == 1, text
    # a long failure is two lines and "details ▸"
    fail = page.locator(f"tr[data-queue-row='{queue_rows['failed']}'] .clamp")
    lines = fail.evaluate("e => e.getBoundingClientRect().height / parseFloat(getComputedStyle(e).lineHeight)")
    assert lines <= 2.2, lines
    page.locator(f"[data-clamp='q{queue_rows['failed']}']").click()
    page.wait_for_function(f"document.querySelector(\"tr[data-queue-row='{queue_rows['failed']}'] .clamp\")"
                           ".classList.contains('open')")
    shot(page, "11f-6-queue-1280-light.png", full_page=True)
    assert page.errors == []


def test_the_row_menu_passes_the_popover_checks(live, page, queue_rows):
    for width in (1280, 400):
        page.set_viewport_size({"width": width, "height": 820})
        page.goto(live["base"] + "/#tab=queue")
        sid = queue_rows["done"]
        btn = page.locator(f"[data-row-menu='q{sid}']")
        btn.wait_for()
        # the page's sticky header must not sit over the row
        btn.evaluate("b => b.scrollIntoView({ block: 'center' })")
        btn.click()
        panel = page.locator(f"[data-pop='q{sid}']")
        panel.wait_for()
        page.wait_for_timeout(250)
        assert btn.get_attribute("aria-expanded") == "true"
        items = page.evaluate(f"""() => [...document.querySelectorAll("[data-pop='q{sid}'] [role=menuitem]")]
          .map(it => {{ const r = it.getBoundingClientRect();
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            return [it.textContent, !!hit && (hit === it || it.contains(hit)), r.height]; }})""")
        # 11g: Log opens the reader; the raw log is the next item
        assert [i[0] for i in items] == ["Log", "Open raw log ↗", "Resubmit", "Copy id",
                                         "Open model page"]
        assert all(ok and h >= 20 for _t, ok, h in items), items
        box = panel.bounding_box()
        assert box["x"] >= 7 and box["x"] + box["width"] <= width - 7
        if width == 1280:
            shot(page, "11f-6-row-menu-1280-light.png")
        page.keyboard.press("Escape")
        page.wait_for_selector(f"[data-pop='q{sid}']", state="detached")
        assert page.evaluate("document.activeElement.dataset.rowMenu") == f"q{sid}"
    assert page.errors == []


# ---------------------------------------------------------------------------
# 7. no native dropdowns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("where", ["", "#tab=leaderboard", "#tab=loop", "#tab=queue", "#tab=review",
                                   "#tab=exam", "#tab=models", "#tab=provenance", "#tab=runs",
                                   "#model=fx%2Fgood-750m", "#topic=economics", "#topic=law"])
def test_there_is_no_native_select_in_the_view(live, page, where):
    page.goto(live["base"] + "/" + where)
    page.wait_for_selector("#view > *")
    page.wait_for_timeout(1200)                    # what loads after the first paint
    assert page.locator("#view select").count() == 0
    assert page.errors == []


def test_the_select_keyboard_path(live, page, queue_rows):
    page.goto(live["base"] + "/#tab=queue")
    sel = page.get_by_label("status filter")
    sel.wait_for()
    assert choice(sel) == "all"
    sel.focus()
    page.keyboard.press("ArrowDown")                # opens, on the chosen option
    page.wait_for_selector("[role=listbox][aria-label='status filter']")
    assert page.evaluate("document.activeElement.dataset.value") == "all"
    page.keyboard.press("ArrowDown")
    assert page.evaluate("document.activeElement.dataset.value") == "active"
    page.keyboard.press("End")
    assert page.evaluate("document.activeElement.dataset.value") == "canceled"
    page.keyboard.press("Home")
    assert page.evaluate("document.activeElement.dataset.value") == "all"
    page.keyboard.type("fa")                        # typeahead
    assert page.evaluate("document.activeElement.dataset.value") == "failed"
    page.keyboard.press("Enter")
    page.wait_for_selector("[role=listbox][aria-label='status filter']", state="detached")
    assert choice(sel) == "failed"
    assert page.evaluate("document.activeElement.getAttribute('aria-label')") == "status filter"
    page.wait_for_function("[...document.querySelectorAll('[data-queue-table] tbody tr')].length === "
                           "Math.min(25, state.queue.filter(r => r.status === 'failed').length)")
    # Esc closes it and leaves the value alone
    page.keyboard.press("ArrowDown")
    page.wait_for_selector("[role=listbox][aria-label='status filter']")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Escape")
    page.wait_for_selector("[role=listbox][aria-label='status filter']", state="detached")
    assert choice(sel) == "failed"
    # the look: 36px, radius 6
    look = sel.evaluate("b => [b.getBoundingClientRect().height, getComputedStyle(b).borderTopLeftRadius]")
    assert look == [36, "6px"]
    assert page.errors == []


# 11m removed the by-criterion block's topic picker; the model page's answers
# picker is the same Combobox, grouped and scored the same way
TOPIC_BOX, TOPIC_POP = "[data-combobox='answers topic']", "#pop-cb-answers-topic"


def test_the_topic_combobox_narrows_as_you_type_and_is_grouped_by_area(live, page):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    model_tab(page, "answers")                          # 12b.2: the Answers tab
    box = page.locator(TOPIC_BOX)
    box.wait_for()
    box.click()
    page.wait_for_selector(f"{TOPIC_POP} [role=option]")
    groups = page.locator(f"{TOPIC_POP} [role=group]").evaluate_all(
        "gs => gs.map(g => g.getAttribute('aria-label'))")
    areas = page.evaluate("Object.keys(DATA.meta.areas)")
    assert groups and all(g in areas + ["Other"] for g in groups)
    shot(page, "11f-7-topic-combobox-1280-light.png")
    box.fill("eco")
    page.wait_for_function(f"document.querySelectorAll('{TOPIC_POP} [role=option]').length === 1")
    only = page.locator(f"{TOPIC_POP} [role=option]")
    assert only.get_attribute("data-value") == "Economics"
    assert re.search(r"\d\.\d\d / 4", only.text_content())
    # the keyboard: ↓ makes it active, Enter picks it
    page.keyboard.press("ArrowDown")
    act = box.get_attribute("aria-activedescendant")
    assert act and page.locator(f"#{act}").get_attribute("data-value") == "Economics"
    page.keyboard.press("Enter")
    page.wait_for_function("state.ans.topic === 'Economics'")
    assert choice(page.locator(TOPIC_BOX)) == "Economics"
    # the choice survives a poll
    page.evaluate("render()")
    assert choice(page.locator(TOPIC_BOX)) == "Economics"
    assert page.locator(TOPIC_BOX).input_value() == "Economics"
    assert page.errors == []


def test_a_chosen_value_survives_a_poll_while_the_list_is_open(live, page):
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    model_tab(page, "answers")
    box = page.locator(TOPIC_BOX)
    box.wait_for()
    box.click()
    box.fill("la")
    page.wait_for_selector(f"{TOPIC_POP} [role=option]")
    page.evaluate("render()")                       # a poll, mid-typing
    assert page.locator(TOPIC_BOX).input_value() == "la"
    assert page.evaluate("document.activeElement.dataset.combobox") == "answers topic"
    page.locator(f"{TOPIC_POP} [role=option][data-value='Law']").click()
    page.wait_for_function("state.ans.topic === 'Law'")
    assert page.errors == []


# ---------------------------------------------------------------------------
# the screenshots for the PR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1512, 400])
def test_screenshots_for_the_pr(live, browser, theme, width):
    ctx, page = new_page(browser)
    try:
        page.set_viewport_size({"width": width, "height": 1000 if width > 500 else 860})
        open_lb(page, live["base"])
        page.evaluate(f"applyTheme('{theme}')")
        page.wait_for_timeout(300)
        shot(page, f"11f-leaderboard-{width}-{theme}.png", full_page=width > 500)
        page.goto(live["base"] + "/#tab=queue")
        page.wait_for_selector("#view .card")
        page.evaluate(f"applyTheme('{theme}')")
        shot(page, f"11f-queue-{width}-{theme}.png")
        open_submit(page, live["base"])
        page.get_by_label("suite").click()
        page.wait_for_selector("[role=listbox][aria-label='suite']")
        page.wait_for_timeout(250)
        shot(page, f"11f-select-{width}-{theme}.png")
        assert page.errors == []
    finally:
        ctx.close()
