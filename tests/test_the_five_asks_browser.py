"""Phase 9a in the browser: model search, answer cards that fit, the version
bar, the override dialog, the pager — against a live service."""

from __future__ import annotations

import json
import re

import pytest

from test_the_five_asks import PROVISIONAL, make_provisional

pytestmark = pytest.mark.dashboard
MODEL = "fx/good-750m"


# ---------------------------------------------------------------------------
# 1. model search
# ---------------------------------------------------------------------------

def open_sit(page, base, topic="law"):
    page.goto(f"{base}/#topic={topic}")
    page.wait_for_selector("[data-panel='sit'] [data-ms='sit'] input")
    return page.locator("[data-ms='sit'] input")


def test_a_keyboard_pick_fills_the_box_and_sets_the_kind(live, page, monkeypatch):
    from service import suggest
    monkeypatch.setattr(suggest, "hub_search", lambda q: [
        {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "params": 1.35e8}])
    suggest._hub_cache.clear()
    box = open_sit(page, live["base"])
    assert box.get_attribute("role") == "combobox"
    box.click()
    box.type("tuned-sk", delay=30)
    page.wait_for_selector("#ms-list-sit:not([hidden]) [role='option']")
    opts = page.locator("#ms-list-sit [role='option']")
    first = opts.first.get_attribute("data-ms-item")
    assert first == "fx/good-750m-tuned-skill"             # the board's own, first
    assert "on the board" in opts.first.text_content()
    assert opts.last.get_attribute("data-ms-item") == "HuggingFaceTB/SmolLM2-135M-Instruct"
    box.press("ArrowDown")
    assert box.get_attribute("aria-activedescendant") == "ms-sit-0"
    assert opts.first.get_attribute("aria-selected") == "true"
    box.press("Enter")
    page.wait_for_selector("#ms-list-sit[hidden]", state="attached")
    assert page.locator("[data-ms='sit'] input").input_value() == "fx/good-750m-tuned-skill"
    assert page.get_by_label("kind", exact=True).first.input_value() == "base"
    # a Hub model with an instruct name: picked with Tab, the kind follows its name
    box = page.locator("[data-ms='sit'] input")
    box.fill("")
    box.type("smollm2", delay=30)
    page.wait_for_selector("#ms-list-sit:not([hidden]) [data-ms-item='HuggingFaceTB/SmolLM2-135M-Instruct']")
    box.press("ArrowDown")
    box.press("Tab")
    assert page.locator("[data-ms='sit'] input").input_value() == \
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert page.get_by_label("kind", exact=True).first.input_value() == "instruct"
    assert page.errors == []


def test_esc_closes_and_a_dead_hub_leaves_the_local_matches(live, page, monkeypatch):
    from service import suggest

    def down(q):
        raise OSError("down")
    monkeypatch.setattr(suggest, "hub_search", down)
    suggest._hub_cache.clear()
    box = open_sit(page, live["base"])
    box.click()
    box.type("good", delay=30)
    page.wait_for_selector("#ms-list-sit:not([hidden]) [data-ms-footer]")
    assert "Hub search unavailable" in page.locator("[data-ms-footer]").text_content()
    assert page.locator("#ms-list-sit [role='option']").count() >= 3
    box.press("Escape")
    page.wait_for_selector("#ms-list-sit[hidden]", state="attached")
    assert box.get_attribute("aria-expanded") == "false"
    assert page.errors == []


def test_a_poll_mid_typing_keeps_the_text_the_caret_and_the_list(live, page):
    box = open_sit(page, live["base"])
    box.click()
    box.type("good-750", delay=30)
    page.wait_for_selector("#ms-list-sit:not([hidden]) [role='option']")
    box.press("ArrowLeft")
    box.press("ArrowLeft")
    before = page.evaluate("document.querySelector('[data-ms=sit] input').selectionStart")
    page.evaluate("render()")                               # what every poll does
    page.evaluate("render()")
    got = page.evaluate("""() => { const i = document.querySelector('[data-ms=sit] input');
      return { v: i.value, c: i.selectionStart, focus: document.activeElement === i,
               open: !document.getElementById('ms-list-sit').hidden,
               n: document.querySelectorAll('#ms-list-sit [role=option]').length } }""")
    assert got["v"] == "good-750" and got["c"] == before == 6
    assert got["focus"] and got["open"] and got["n"] >= 1
    assert page.errors == []


def test_submit_has_the_same_search(live, page):
    page.goto(live["base"] + "/#tab=queue")
    box = page.locator("[data-ms='submit'] input")
    box.wait_for()
    box.click()
    box.type("chance", delay=30)
    page.wait_for_selector("#ms-list-submit:not([hidden]) [data-ms-item='fx/chance-160m']")
    box.press("ArrowDown")
    box.press("Enter")
    assert page.locator("[data-ms='submit'] input").input_value() == "fx/chance-160m"
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2. the answers fit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [1280, 1512])
@pytest.mark.parametrize("slug,criteria", [("medicine_health", 15), ("law", 23)])
def test_the_answers_never_scroll_sideways(live, browser, width, slug, criteria):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    pg = ctx.new_page()
    try:
        pg.goto(f"{live['base']}/#topic={slug}")
        pg.wait_for_selector("[data-answers-table] [data-answer]")
        m = pg.evaluate("""() => { const l = document.querySelector('[data-answers-table]');
          const d = document.documentElement;
          return { list: l.scrollWidth <= l.clientWidth, page: d.scrollWidth <= d.clientWidth,
                   cells: l.querySelector('[data-answer]').querySelectorAll('[data-criterion]').length,
                   cellW: Math.max(...[...l.querySelectorAll('[data-criterion]')]
                                     .map(c => c.getBoundingClientRect().width)) } }""")
        assert m["list"] and m["page"], m
        assert m["cells"] == criteria
        assert m["cellW"] <= 14.5                          # fixed cells, not 170 px bars
        card = pg.locator("[data-answer]").first
        assert card.get_attribute("data-half") == "diagnose"
        assert card.locator("[data-judge-note]").count() == 1
        assert "weakest:" in card.locator("[data-weakest]").text_content() \
            or "every criterion met" in card.locator("[data-weakest]").text_content()
        cell = card.locator("[data-criterion]").first
        title = cell.get_attribute("title")
        assert re.fullmatch(r".+: (\d(\.\d+)?|not applicable)", title), title
        assert cell.get_attribute("aria-label") == title
    finally:
        ctx.close()


def test_under_800_px_the_score_drops_below_the_text(live, browser):
    ctx = browser.new_context(viewport={"width": 700, "height": 900})
    pg = ctx.new_page()
    try:
        pg.goto(f"{live['base']}/#topic=law")
        pg.wait_for_selector("[data-answers-table] [data-answer]")
        box = pg.evaluate("""() => { const c = document.querySelector('[data-answer]');
          const a = c.querySelector('.main').getBoundingClientRect();
          const b = c.querySelector('.side').getBoundingClientRect();
          return { below: b.top >= a.bottom - 1,
                   fits: document.documentElement.scrollWidth <= document.documentElement.clientWidth } }""")
        assert box["below"] and box["fits"]
    finally:
        ctx.close()


def test_a_criterion_below_half_is_a_filter(live, page):
    page.goto(live["base"] + "/#topic=law")
    page.wait_for_selector("[data-answers-table] [data-answer]")
    before = int(page.locator("[data-answer-count]").first.get_attribute("data-answer-count"))
    sel = page.get_by_label("criterion filter")
    assert "any criterion below 0.5" in sel.text_content()
    sel.select_option("any")
    page.wait_for_function("n => +document.querySelector('[data-answer-count]').dataset.answerCount <= n",
                           arg=before)
    assert page.errors == []


# ---------------------------------------------------------------------------
# 3. a page from an older build
# ---------------------------------------------------------------------------

def stale(page, base, reload_ms=None):
    """The page as an older build served it: same code, a different stamp."""
    def old(route):
        r = route.fetch()
        body = re.sub(r'(<meta name="evalboard-build" content=")[^"]*', r"\1stale0+0000000",
                      r.text())
        route.fulfill(response=r, body=body)
    page.route(re.compile(re.escape(base) + r"/(#.*)?$"), old)
    if reload_ms:
        page.add_init_script(f"window.__evalboardReloadMs = {reload_ms};")


def test_a_page_from_an_older_build_offers_the_new_one(live, page):
    base = live["base"]
    stale(page, base)
    page.goto(base + "/#tab=models")
    bar = page.locator("[data-build-bar]")
    bar.wait_for(timeout=20000)
    assert "The dashboard was updated" in bar.text_content()
    assert bar.locator("button[data-reload]").is_visible()
    from service import app
    assert bar.get_attribute("data-build-bar") == app.BUILD
    assert page.errors == []


def test_it_never_reloads_while_someone_is_typing(live, page):
    base = live["base"]
    stale(page, base, reload_ms=1200)
    loads = []
    page.on("load", lambda _: loads.append(1))
    page.goto(base + "/#tab=queue")
    page.locator("[data-build-bar]").wait_for(timeout=20000)
    page.locator("[data-ms='submit'] input").click()      # typing
    page.wait_for_timeout(3500)
    assert len(loads) == 1
    assert "once you stop typing" in page.locator("[data-build-bar]").text_content()
    page.locator("h2").first.click()                      # stopped
    page.wait_for_function("() => performance.getEntriesByType('navigation')[0] && "
                           "document.readyState === 'complete'")
    page.wait_for_timeout(2500)
    assert len(loads) >= 2
    assert page.url.endswith("#tab=queue")                # the hash survives


# ---------------------------------------------------------------------------
# 4. Propose… over a provisional judge
# ---------------------------------------------------------------------------

def open_topic_for(page, base, slug, model=MODEL):
    page.goto(f"{base}/#topic={slug}")
    page.wait_for_selector("[data-panel='answers'] select[aria-label='model']")
    page.locator("[data-panel='answers'] select[aria-label='model']").select_option(model)
    # picking a model re-fetches its answers, and their landing re-renders the
    # page: wait for that, or a click can land on a button being replaced
    page.wait_for_function("m => state.ans.model === m && state.ans.rows && !state.ans.loading",
                           arg=model, timeout=30000)
    page.wait_for_selector("[data-answers-table] [data-answer]")
    btn = page.locator(f"[data-propose='{slug}'][data-propose-model='{model}']")
    btn.wait_for()
    return btn


def test_propose_over_a_provisional_judge_asks_first_and_marks_it(live, page):
    from service import app
    make_provisional()
    app._cache.update(key=None, payload=None, at=0.0)
    base = live["base"]
    btn = open_topic_for(page, base, "economics")
    assert btn.get_attribute("data-gate") == "overridable"
    assert btn.text_content() == "Propose…" and btn.is_enabled()
    btn.click()
    dlg = page.locator("[role='dialog']")
    dlg.wait_for()
    assert dlg.get_attribute("aria-modal") == "true"
    assert "This judge is a small local model" in dlg.text_content()
    assert PROVISIONAL in dlg.locator("[data-dialog-reasons]").text_content()
    assert "proposed over a provisional judge" in dlg.text_content()
    go = dlg.locator("[data-dialog-go]")
    assert go.is_disabled()
    # Esc cancels, and focus goes back to the button
    page.keyboard.press("Escape")
    page.wait_for_selector("[role='dialog']", state="detached")
    assert page.evaluate("document.activeElement.dataset.propose") == "economics"
    # again: focus stays inside; the box must be ticked; then it goes
    btn.click()
    dlg.wait_for()
    for _ in range(8):
        page.keyboard.press("Tab")
        assert page.evaluate("!!document.activeElement.closest('[role=dialog]')")
    dlg.get_by_label("your name").fill("Omar")
    assert go.is_disabled()
    dlg.locator("[data-dialog-ack]").check()
    assert go.is_enabled()
    go.click()
    page.wait_for_selector("[role='dialog']", state="detached")
    ok = page.locator("[data-action-ok='propose:economics']")
    ok.wait_for(timeout=20000)
    assert "Proposal #" in ok.text_content() and "provisional judge" in ok.text_content()
    import urllib.request
    with urllib.request.urlopen(base + "/api/proposals") as r:
        props = json.loads(r.read())
    p = next(x for x in props if x["model"] == MODEL and x["category"] == "economics")
    assert p["override"]["by"] == "Omar"
    # the topic page's proposal line carries the mark
    page.wait_for_selector("[data-panel='output'] [data-over-provisional]", timeout=20000)
    assert page.errors == []


def test_a_data_reason_is_a_disabled_button_and_no_dialog(live, page):
    from service import app
    make_provisional()
    app._cache.update(key=None, payload=None, at=0.0)
    btn = open_topic_for(page, live["base"], "medicine_health")
    assert btn.get_attribute("data-gate") == "hard"
    assert btn.is_disabled()
    why = page.locator("[data-topic-page] [data-why='propose']").first.text_content()
    assert "under the 30" in why
    btn.click(force=True)
    page.wait_for_timeout(300)
    assert page.locator("[role='dialog']").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# 5. one pager
# ---------------------------------------------------------------------------

def test_the_queue_pages_and_a_poll_keeps_the_page(live, page):
    from service import db
    have = len(db.recent(500))
    for i in range(49 - have):
        db.add(f"org/pager-model-{i:02d}", "auto", "quick", "omar", "")
    for r in db.recent(500)[:12]:
        db.update(r["id"], status="done")
    page.goto(live["base"] + "/#tab=queue")
    pager = page.locator("[data-pager='queue']")
    pager.wait_for()
    assert "1–25 of 49" in pager.text_content()
    assert pager.locator("button[data-page]").count() == 2        # two pages
    assert pager.locator("[aria-current='page']").text_content() == "1"
    pager.locator("button[data-page='2']").click()
    assert "26–49 of 49" in page.locator("[data-pager='queue']").text_content()
    assert page.locator("[data-queue-table] tbody tr").count() == 24
    # a poll lands (a new row arrives) — still page 2
    db.add("org/arrives-mid-read", "auto", "quick", "omar", "")
    page.wait_for_function("document.querySelector('[data-pager=queue]').textContent.includes('of 50')",
                           timeout=20000)
    assert page.locator("[data-pager='queue'] [aria-current='page']").text_content() == "2"
    # a filter is a different list: back to page 1
    page.get_by_label("status filter").select_option("done")
    page.wait_for_function("document.querySelector('[data-queue-table] tbody tr')")
    assert page.locator("[data-pager='queue'] [aria-current='page']").count() == 1
    assert page.locator("[data-pager='queue'] [aria-current='page']").text_content() == "1"
    assert page.errors == []
