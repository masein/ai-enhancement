"""12b.2 on the page. Home is three blocks — Needs you, Running now, and the
best in each kind of test — with no hero and no stats line. The model page is
a header (the name, one line of facts, Test this model, one tile per kind of
test) over four tabs: Scores · Answers · Improve · History, remembered per
viewer, Improve only when there is something in it."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest

from test_live_check_11k import clear, plant_proposal, sql
from test_page_recovery import Live

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12b"
MODEL = "fx/good-750m"                        # Standard, the exam and the pilot
STANDARD_ONLY = "fx/good-750m-tuned-skill"    # no exam, no pilot


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def home(page, base, width=1512):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/")
    page.wait_for_selector("[data-needs-you]")


def open_model(page, base, mid=MODEL, width=1512):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#model=" + mid.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")


def plant_dataset(pid, status="ready"):
    from service import db
    did = db.dataset_create(pid, "free", 10, "masein", {})
    db.dataset_update(did, status=status)
    return did


@pytest.fixture
def tidy():
    """proposals, datasets, runs and training runs this test plants go again"""
    clear()
    sql(("DELETE FROM truns",))
    yield
    clear()
    sql(("DELETE FROM truns",))


# ---------------------------------------------------------------------------
# §6 Home
# ---------------------------------------------------------------------------

def test_home_is_three_blocks_in_order_with_no_hero(live, page):
    home(page, live["base"])
    heads = page.evaluate("[...document.querySelectorAll('#view > .card')].map(c => "
                          "(c.querySelector('h2') || {}).textContent)")
    assert heads == ["Needs you", "Running now", "Best in each kind of test"]
    # no hero paragraph, no stats line, and nothing of the old Overview's
    for gone in ("#pagehero", "[data-statline]", "[data-highlights]", "[data-top-models]",
                 "[data-overview-loop]", "[data-prelim-count]"):
        assert page.locator(gone).count() == 0, gone
    assert page.errors == []


def test_needs_you_lists_what_waits_and_each_line_goes_there(live, page, tidy):
    from service import db
    pid = plant_proposal()
    did = plant_dataset(plant_proposal(status="approved"))
    sid = db.add("org/broken", "auto", "quick", "omar", "")
    db.update(sid, status="failed", finished_at=time.time())
    old = db.add("org/long-ago", "auto", "quick", "omar", "")
    db.update(old, status="failed", finished_at=time.time() - 8 * 86400)    # not this week
    home(page, live["base"])
    page.wait_for_selector("[data-needs='datasets']")
    # 12b.3: the problems, not the known limits — this board's one is the stub grader
    assert page.evaluate("boardProblems(DATA.checks).map(c => c.key)") == ["judge_stub"]
    lines = page.locator("[data-needs-you] li").all_inner_texts()
    assert lines == ["1 proposal waiting for review", "1 dataset made but not used in training",
                     "1 run failed in the last seven days", "1 check is not green"]
    # each line is a link to where it is dealt with
    page.locator("[data-needs='proposals'] a").click()
    page.wait_for_selector(f"[data-rv-list='review'] [data-rv-row='{pid}']")
    assert page.evaluate("location.hash").startswith("#tab=improve&sub=review")
    home(page, live["base"])
    page.locator("[data-needs='datasets'] a").click()
    page.wait_for_selector(f"[data-rv-list='datasets'] [data-ds-row='{did}']")
    home(page, live["base"])
    page.locator("[data-needs='failed'] a").click()
    page.wait_for_selector(f"[data-all-runs] tr[data-queue-row='{sid}']")
    home(page, live["base"])
    page.locator("[data-needs='checks'] a").click()
    assert page.locator("#warnings details.checks").get_attribute("open") is not None
    # a dataset a training run used is not waiting
    db.trun_create("run-a", "p", "omar", "{}", "", datasets=[did])
    home(page, live["base"])
    page.wait_for_selector("[data-needs='proposals']")
    page.wait_for_timeout(600)
    assert page.locator("[data-needs='datasets']").count() == 0
    assert page.errors == []


def test_nothing_needs_you_and_nothing_running_say_so(browser, payload):
    """A board with nothing waiting: no checks, and an API with no proposals,
    datasets, runs or training runs."""
    p = copy.deepcopy(payload)
    p["checks"], p["warnings"] = [], []
    ctx = browser.new_context(viewport={"width": 1512, "height": 900}, reduced_motion="reduce")
    s = Live(ctx, p, fail=False)
    try:
        pg = s.open()
        pg.wait_for_selector("[data-needs-none]")
        assert pg.locator("[data-needs-none]").inner_text() == "Nothing needs you."
        assert pg.locator("[data-running-none]").inner_text() == "Nothing running · Test a model"
        pg.locator("[data-running-test]").click()
        pg.wait_for_selector("[data-dialog='test'] [data-submit-form]")
        assert s.errors == []
    finally:
        ctx.close()


def test_running_now_is_the_run_counters_list(live, page, tidy):
    from service import db
    sid = db.add(MODEL, "auto", "quick", "omar", "")
    db.update(sid, status="running", progress="1 of 4 tasks")
    home(page, live["base"])
    page.wait_for_selector("[data-running-now='1']")
    block = page.locator("[data-running-now] .runsfull")
    page.locator("#runs [data-runs]").click()
    pop = page.locator("#pop-runs")
    pop.wait_for()
    # the same list: the same rows, in the same words
    assert block.inner_text() == pop.inner_text()
    assert "good-750m" in block.inner_text()
    assert page.errors == []


def test_best_in_each_kind_is_one_card_per_kind_with_data(live, page):
    home(page, live["base"])
    cards = page.locator("[data-best]")
    assert [c.get_attribute("data-best") for c in cards.all()] == ["standard", "exam", "everyday"]
    want = page.evaluate("""() => {
      const std = DATA.models.filter(m => officialAvg(m) != null && !m.duplicateOf)
        .sort((a, b) => officialAvg(b) - officialAvg(a))[0];
      const ex = DATA.models.filter(m => m.judgedAvg != null && !m.duplicateOf)
        .sort((a, b) => b.judgedAvg - a.judgedAvg)[0];
      return [[(100 * officialAvg(std)).toFixed(1), std.name],
              [num(ex.judgedAvg, 2) + ' / 4', ex.name]]; }""")
    for key, (value, name) in zip(("standard", "exam"), want):
        assert page.locator(f"[data-best-value='{key}']").inner_text() == value
        assert page.locator(f"[data-best-name='{key}']").inner_text() == name
    assert page.locator("[data-best-value='everyday']").inner_text() == "108 of 111"   # 12a.2
    assert page.locator("[data-best='everyday'] [data-pilot-badge]").count() == 1
    # one link each, to that kind on Models
    for key in ("standard", "exam", "everyday"):
        home(page, live["base"])
        assert page.locator(f"[data-best='{key}'] a").count() == 1
        page.locator(f"[data-best='{key}'] a").click()
        page.wait_for_selector("[data-lb-card]")
        assert page.evaluate("lbS().view") == key
    # a calibrated judge: no caveat
    home(page, live["base"])
    assert page.locator("[data-best-caveat]").count() == 0
    assert page.errors == []


def test_a_provisional_judge_is_said_once_in_the_blocks_header(live, page):
    def handle(route):
        r = route.fetch()
        body = r.json()
        body["judged"]["calibration"] = None
        for m in body["models"]:
            if m.get("judgeState"):
                m["judgeState"] = {**m["judgeState"], "ok": False,
                                   "reasons": ["the judge is not calibrated"]}
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)
    home(page, live["base"])
    page.wait_for_selector("[data-best='exam']")
    caveat = page.locator("[data-best-caveat]")
    assert caveat.count() == 1 and caveat.inner_text() == "Knowledge exam: provisional judge"
    assert page.locator("[data-best-by-kind] .sechead [data-best-caveat]").count() == 1
    assert page.errors == []


def test_a_kind_with_no_data_has_no_card(browser, payload):
    p = copy.deepcopy(payload)
    for m in p["models"]:
        m["judgedAvg"] = None
        m["judge"] = None           # 12b.3: judged topics alone make a card, the weakest topic
    p["everyday"] = None
    ctx = browser.new_context(viewport={"width": 1512, "height": 900}, reduced_motion="reduce")
    s = Live(ctx, p, fail=False)
    try:
        pg = s.open()
        pg.wait_for_selector("[data-best-by-kind]")
        assert [c.get_attribute("data-best") for c in pg.locator("[data-best]").all()] == ["standard"]
        assert s.errors == []
    finally:
        ctx.close()


def test_judge_steadiness_is_a_line_under_the_checks(live, page):
    home(page, live["base"])
    page.locator("#warnings summary[data-warn-summary]").click()
    line = page.locator("#warnings [data-judge-steady]")
    line.wait_for()
    assert line.inner_text().startswith("Judge steadiness: steady — 30 fixed scripts re-graded")
    # it is not one of the checks: the dot's count is the checks' (12b.3: the problems)
    n = page.evaluate("boardProblems(DATA.checks).length")
    assert page.locator("#warnings [data-warn-summary]").get_attribute("data-warn-summary") == str(n)
    assert page.errors == []


# ---------------------------------------------------------------------------
# §7 the model page
# ---------------------------------------------------------------------------

def test_the_header_is_the_name_the_facts_one_action_and_a_tile_per_kind(live, page):
    open_model(page, live["base"])
    hero = page.locator("[data-model-hero]")
    assert hero.locator("h1.mtitle").inner_text() == "good-750m"
    assert hero.locator("[data-model-facts]").inner_text() == "750M · base · fx"
    assert MODEL not in hero.inner_text()                      # no ids in the main view
    # 12b.3: one filled button, and it is the header's, reading Test this model
    assert page.locator("#view button.primary").count() == 0
    assert page.locator("header [data-test-model]").inner_text() == "Test this model"
    tiles = hero.locator("[data-kind-tile]")
    assert [t.get_attribute("data-kind-tile") for t in tiles.all()] == \
        ["standard", "exam", "everyday"]
    avg, jav, r = page.evaluate(f"""() => {{ const m = DATA.models.find(x => x.id === '{MODEL}');
      return [(100 * officialAvg(m)).toFixed(1), num(m.judgedAvg, 2), rankOf(m)]; }}""")
    assert hero.locator("[data-kind-value='standard']").inner_text() == avg
    assert hero.locator("[data-kind-tile='standard'] .ktile-sub").inner_text() == \
        f"above chance · 7 of 7 tasks · #{r['n']} of {r['of']}"
    assert hero.locator("[data-kind-value='exam']").inner_text() == jav
    assert hero.locator("[data-kind-value='everyday']").inner_text() == "108 of 111"  # 12a.2
    assert hero.locator("[data-kind-tile='everyday'] [data-pilot-badge]").count() == 1
    # the main action: the Test a model dialog, this model filled in
    page.locator("[data-test-this]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    assert dlg.locator("[data-ms='submit'] input").input_value() == MODEL
    assert page.errors == []


def test_a_kind_not_taken_is_a_not_tested_tile_and_no_block(live, page):
    open_model(page, live["base"], STANDARD_ONLY)
    for kind in ("exam", "everyday"):
        tile = page.locator(f"[data-kind-tile='{kind}']")
        assert " ".join(tile.inner_text().split()) == \
            ("KNOWLEDGE EXAM" if kind == "exam" else "EVERYDAY TASKS") + " Not tested · Test"
    assert [b.get_attribute("data-kind-block") for b in page.locator("[data-kind-block]").all()] \
        == ["standard"]
    # the exam's Test is its own topic picker, on this page
    page.locator("[data-kind-tile='exam'] [data-sit-open]").click()
    page.wait_for_selector("[data-panel='msit']")
    assert page.errors == []


def test_scores_opens_the_newest_kind_and_folds_the_others(live, page):
    open_model(page, live["base"])
    newest = page.evaluate(f"""() => {{ const ks = modelKinds(DATA.models.find(x => x.id === '{MODEL}'))
      .filter(k => k.taken); return ks.reduce((a, b) => (b.at || 0) > (a.at || 0) ? b : a).kind; }}""")
    blocks = page.locator("[data-kind-block]")
    assert [b.get_attribute("data-kind-block") for b in blocks.all()] == \
        ["standard", "exam", "everyday"]
    for b in blocks.all():
        k = b.get_attribute("data-kind-block")
        is_open = b.get_attribute("open") is not None
        assert is_open == (k == newest), k
        if not is_open:            # a closed block is its summary alone, built when opened
            assert b.evaluate("d => d.children.length") == 1, k
    # Standard: Results, and Diagnose under "What the score can't show"
    page.locator("[data-kind-block='standard'] > summary").click()
    std = page.locator("[data-kind-block='standard']")
    std.locator("#sec-results").wait_for()
    fold = std.locator("[data-cant-show]")
    assert page.locator("[data-cant-show] > summary").inner_text() == "What the score can’t show ▸"
    assert fold.locator("#sec-diagnose").count() == 1
    # the exam: score against length folds under More detail; no judge ids here
    page.locator("[data-kind-block='exam'] > summary").click()
    exam = page.locator("[data-kind-block='exam']")
    exam.locator("#sec-judged").wait_for()
    more = exam.locator("[data-more-detail='judged']")
    assert "Score against answer length" in more.text_content()
    assert "Judge stub/" not in exam.inner_text()
    # an opened block stays open across a poll
    page.evaluate("render()")
    assert page.locator("[data-kind-block='standard']").get_attribute("open") is not None
    # a tile opens its block
    page.locator("[data-kind-block='standard'] > summary").click()
    page.locator("[data-kind-tile='standard']").click()
    page.wait_for_selector("[data-kind-block='standard'][open] #sec-results")
    assert page.errors == []


def test_the_tabs_are_remembered_and_improve_is_absent_when_empty(live, page, tidy):
    open_model(page, live["base"])
    tabs = page.locator("[data-model-tabs] [role=tab]")
    assert tabs.all_inner_texts() == ["Scores", "Answers", "History"]
    page.locator("[data-mtab='history']").click()
    page.wait_for_selector("[data-model-prov]")
    # remembered per viewer: across a reload and another model's page
    page.reload()
    page.wait_for_selector("[data-mtab='history'][aria-selected='true']")
    open_model(page, live["base"], STANDARD_ONLY)
    page.wait_for_selector("[data-mtab='history'][aria-selected='true']")
    # the arrow keys move along the tabs
    page.locator("[data-mtab='history']").focus()
    page.keyboard.press("ArrowLeft")
    page.wait_for_selector("[data-mtab='answers'][aria-selected='true']")
    page.wait_for_function("document.activeElement.dataset.mtab === 'answers'")
    assert page.errors == []


def test_improve_is_this_models_proposals_and_datasets(live, page, tidy):
    """The brief names SmolLM2-360M-Instruct's fixtures; that model is not on
    this test board, so its proposal and dataset are planted on good-750m."""
    pid = plant_proposal(model=MODEL)
    did = plant_dataset(plant_proposal(model=MODEL, status="approved"))
    open_model(page, live["base"])
    page.wait_for_selector("[data-mtab='improve']")
    assert page.locator("[data-model-tabs] [role=tab]").all_inner_texts() == \
        ["Scores", "Answers", "Improve", "History"]
    page.locator("[data-mtab='improve']").click()
    page.wait_for_selector(f"[data-model-proposals] [data-rv-row='{pid}']")
    assert page.locator(f"[data-model-datasets] [data-ds-row='{did}']").count() == 1
    # another model's page has none of it, and no Improve
    open_model(page, live["base"], STANDARD_ONLY)
    page.wait_for_selector("[data-model-tabs]")
    page.wait_for_timeout(600)
    assert page.locator("[data-mtab='improve']").count() == 0
    assert page.locator("[data-mtab='scores'][aria-selected='true']").count() == 1
    assert page.errors == []


def test_answers_are_by_kind_then_topic_or_group(live, page):
    open_model(page, live["base"])
    page.locator("[data-mtab='answers']").click()
    kinds = page.locator("[data-answers-kind]")
    assert kinds.all_inner_texts() == ["Knowledge exam", "Everyday tasks"]
    page.wait_for_selector("[data-panel='model-answers'] .anscard")
    page.locator("[data-answers-kind='everyday']").click()
    # 12a.2: one group at a time, the first to begin with — Understanding's 16
    qs = page.locator("[data-answers-q]")
    assert qs.count() == 16
    group = page.locator("[data-answers-group='summarising']")
    assert group.inner_text() == "Summarising · 17 of 17"
    group.click()
    page.wait_for_function("document.querySelectorAll('[data-answers-q]').length === 17")
    assert set(page.locator("[data-answers-q] .evgroup").all_inner_texts()) == {"Summarising"}
    assert page.locator("[data-answers-q] [data-evd-question]").count() == 17
    # a model that has written nothing says so in one line
    open_model(page, live["base"], STANDARD_ONLY)
    page.locator("[data-mtab='answers']").click()
    assert page.locator("[data-answers-none]").inner_text().startswith("No written answers yet")
    assert page.errors == []


def test_history_holds_the_runs_run_provenance_and_how_it_was_graded(live, page):
    open_model(page, live["base"])
    page.locator("[data-mtab='history']").click()
    prov = page.locator(f"[data-model-prov='{MODEL}']")
    prov.wait_for()
    facts = dict(zip(prov.locator("dt").all_inner_texts(), prov.locator("dd").all_inner_texts()))
    assert facts["hub id"] == MODEL and facts["harness"] and facts["limit"] == "full"
    graded = page.locator(f"[data-model-graded='{MODEL}']")
    assert graded.locator("dt").all_inner_texts() == ["Standard", "Knowledge exam", "Everyday tasks"]
    assert "Judge stub/overlap-v1" in graded.inner_text()
    graded.locator("[data-how-graded]").click()
    page.wait_for_selector("#reader[data-ready='1']")
    # Data & sources keeps what is true of the whole board, and says where
    # each model's provenance went
    page.keyboard.press("Escape")
    page.goto(live["base"] + "/#tab=data")
    page.wait_for_selector("[data-builds-card]")
    assert page.locator("[data-prov-table]").count() == 0
    assert "under History" in page.locator("[data-builds-card]").inner_text()
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1512])
def test_no_page_scrolls_sideways_and_the_screens(live, browser, width):
    ctx = browser.new_context(viewport={"width": width, "height": 1000}, reduced_motion="reduce")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(live["base"] + "/")
        page.wait_for_selector("[data-best-by-kind]")
        page.wait_for_timeout(300)
        assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth"), "home"
        shot(page, f"12b-home-{width}-light.png", full_page=True)
        for tab in ("scores", "answers", "history"):
            page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
            page.wait_for_selector("[data-model-tabs]")
            page.locator(f"[data-mtab='{tab}']").click()
            page.wait_for_selector(f"[data-mtab-panel='{tab}']")
            page.wait_for_function("!document.querySelector('#view [aria-busy=true]')",
                                   timeout=20000)
            page.wait_for_timeout(300)
            assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth"), tab
            shot(page, f"12b-model-{tab}-{width}-light.png", full_page=True)
        assert errors == []
    finally:
        ctx.close()
