"""16.3 on the page: the Models toolbar.

- **Row 1:** which tests — Standard · Mobile · Everyday · Frontier (and the
  Knowledge exam while it has results) — and Table or Chart.
- **Row 2:** Group ▾ in place of the twelve chips, each group with how many
  benchmarks it holds; Models ▾; Filters ▾ (Size, Source, Type, Tested; More:
  Status, Scale) with its count; one Columns ▾ in place of two, keeping both
  jobs; and the saved views.
- **Row 3:** a chip for each filter on, Clear all, "9 of 56 models".
- **Mobile:** All (a headline a suite), DeviceMark (its Chart the On-device
  chart), MobileAIBench in its three parts, Mobile-MMLU with Pro's categories
  and the full set kept apart.
- A benchmark in one place only; the Chart is the ranked bars, for what the
  toolbar keeps; a view with no chart says why; Table or Chart is remembered
  for each view and is in the address.

At 1400 and 375 px. Results and answers written in: nothing runs."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

import mobile_mmlu as mmp
import mobileaibench as mab
from conftest import choose_chip, pick_view, set_name
from test_12o3 import sit_mab
from test_12x_models_devicemark_browser import QWEN, SETUP, rows  # noqa: F401
from test_14_1_mab_text import sit
from test_14_3_browser import no_sideways, steady_shot
from test_14_3_mobile_mmlu import FIXTURE, RIGHT, a_key, pin_fixture, sit_picks

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16_3"
GOOD, SKEWED, LOCAL = "fx/good-750m", "fx/skewed-360m", "local/nodiag-step400"
LB = "[data-lb-table]"
WIDTHS = [1400, 375]


@pytest.fixture(scope="module", autouse=True)
def board(live, rows):  # noqa: F811
    """MobileAIBench's no-judge and judged parts on good-750m (14.1's), Pro on
    our key for good-750m and skewed-360m (14.3's), DeviceMark's rows (12x's),
    and good-750m tested yesterday by a run of the board's"""
    import service.app as appmod
    from service import config, db
    tree = live["tree"]
    good, skewed = tree["models"][GOOD]["dir"], tree["models"][SKEWED]["dir"]
    sit_mab(good)
    for d, words in ((good, 120), (skewed, 30)):
        sit(d, mab.MTB1, answer=lambda q, w=words: "word " * w)
        sit(d, mab.MTB2, docs=mab.load(mab.MTB2), answer=lambda q, w=words: "word " * w)
    now = {"id": "judge-test-1", "version": mab._judge_now()["version"]}
    mab._record(good, {(it["id"], it["turn"]): float(4 + it["turn"])
                       for it in mab.mt_answers(good)}, now)
    for d in (good, skewed):
        mab.write(d, mab.mark(d))
    sid = db.add(GOOD, "instruct", "everyday", "masein", "", status="done")
    db.update(sid, finished_at=time.time() - 86400)
    with pytest.MonkeyPatch.context() as mp:
        d = Path(config.MMP_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes())
        pin_fixture(mp)
        a_key()
        for mid, picks in ((GOOD, dict(RIGHT, inv00001="C")), (SKEWED, RIGHT)):
            md = tree["models"][mid]["dir"]
            sit_picks(md, picks)
            mmp.collect(md)
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


def go(page, live, hash_="", width=1400, sel=f"{LB} tbody tr"):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models" + (("&" + hash_) if hash_ else ""))
    page.wait_for_selector(sel)


def heads(page):
    return page.evaluate(f"""() => [...document.querySelectorAll('{LB} thead tr.names th[data-col]')]
      .map(t => t.dataset.col)""")


def groups(page):
    return page.locator(f"{LB} thead tr.grp th").all_text_contents()


def row_ids(page):
    return page.evaluate(f"""() => [...document.querySelectorAll('{LB} tbody tr[data-lb-row]')]
      .map(t => t.dataset.lbRow)""")


# ---------------------------------------------------------------------------
# Row 1 and Row 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTHS)
def test_row_1_says_which_tests_and_row_2_holds_the_menus(live, page, width):
    go(page, live, width=width)
    assert page.locator("[data-models-view]").all_inner_texts() == \
        ["Standard", "Mobile", "Everyday", "Frontier", "Knowledge exam"]
    assert page.locator("[data-lb-show] [data-show]").all_inner_texts() == ["Table", "Chart"]
    assert page.locator("[data-lb-show]").get_attribute("data-lb-show") == "table"
    # Row 2's menus, and the count where Row 3 would be with nothing on
    picks = page.locator("[data-pickers]")
    for pill in ("#pill-group", "#pill-models", "[data-filters]", "#pill-columns"):
        assert picks.locator(pill).count() == 1, pill
    n = page.evaluate("DATA.models.length")
    assert page.locator("[data-lb-count]").inner_text() == f"{n} of {n} models"
    assert page.locator("[data-lb-active]").count() == 0
    # Group ▾: one choice, each with how many benchmarks it holds
    page.locator("#pill-group").click()
    menu = page.locator("#pop-group")
    texts = menu.locator("[data-choice]").all_inner_texts()
    assert [t.split(" · ")[0] for t in texts] == ["All", "Knowledge", "Commonsense", "Reasoning",
                                                  "Math", "Trust & safety",
                                                  "Instruction & maths", "Language modelling"]
    assert texts[2] == "Commonsense · 3 benchmarks" and texts[1] == "Knowledge · 1 benchmark"
    # All counts as the groups count: at least their sum, each benchmark once
    counts = [int(t.split(" · ")[1].split()[0]) for t in texts]
    assert counts[0] >= sum(counts[1:7])
    no_sideways(page)
    shot(page.locator("[data-lb-card] .lbbar").locator(".."), f"toolbar-{width}.png")
    page.keyboard.press("Escape")
    assert page.errors == []


def test_mobiles_groups_and_a_benchmark_in_one_place_only(live, page):
    go(page, live)
    pick_view(page, "mobile")
    page.locator("#pill-group").click()
    menu = page.locator("#pop-group")
    names = menu.locator("[data-choice]").evaluate_all("es => es.map(e => e.firstChild.textContent)")
    assert names == ["All", "DeviceMark", "MobileAIBench", "Mobile-MMLU"]
    assert menu.locator("[data-choice='mobileaibench'] [data-group-count]").inner_text() == \
        " · 9 benchmarks"
    # 14.4.4's badge on the menu item whose sets carry a restriction
    assert menu.locator("[data-choice='mmlu'] [data-group-restriction]").count() >= 1
    page.keyboard.press("Escape")
    # Standard ▸ All, every column shown: none of Mobile's, none of Frontier's
    pick_view(page, "standard", "all")
    page.locator("#pill-columns").click()
    page.locator("#pop-columns [data-show-all]").click()
    page.keyboard.press("Escape")
    cols = heads(page)
    assert not [c for c in cols if c.startswith(("mab_", "mobile_mmlu", "gpqa", "simpleqa"))], cols
    # and Trust & safety holds ours alone: MobileAIBench's three are under Mobile
    pick_view(page, "standard", "trust")
    page.locator("#pill-columns").click()
    assert not page.locator("#pop-columns [data-column^='mab_']").count()
    page.keyboard.press("Escape")
    assert not [c for c in heads(page) if c.startswith("mab_")]
    assert page.errors == []


# ---------------------------------------------------------------------------
# Mobile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTHS)
def test_mobile_all_is_one_headline_a_suite(live, page, width):
    go(page, live, "view=mobile", width)
    assert page.locator("#pill-group").get_attribute("data-value") == "all"
    assert heads(page) == ["name", "params", "dm:composite", "mab_mtbench", "mobile_mmlu_pro"]
    assert groups(page) == ["", "DeviceMark", "MobileAIBench", "Mobile-MMLU"]
    # the full set is never a column beside Pro's (14.4)
    assert page.locator(f"{LB} th[data-col='mobile_mmlu_full']").count() == 0
    # no chart: different suites, each on its own scale — the button says why
    chart = page.locator("[data-lb-show] [data-show='chart']")
    assert chart.get_attribute("aria-disabled") == "true"
    assert "choose one under Group" in chart.get_attribute("title")
    chart.dispatch_event("click")              # aria-disabled: it takes the click, and says why
    why = page.locator("#why-no-chart")
    assert why.is_visible() and why.inner_text().startswith("Mobile ▸ All sets different suites")
    assert page.locator("[data-lb-show]").get_attribute("data-lb-show") == "table"
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"mobile-all-{width}.png")
    assert page.errors == []


def test_mobileaibench_in_its_parts(live, page):
    go(page, live, "view=mobile&group=mobileaibench")
    assert "mab_mtbench" in heads(page) and "mab_hotpotqa" in heads(page)
    assert "mobile_mmlu_pro" not in heads(page)                       # Mobile-MMLU's own group
    g = groups(page)
    assert "MobileAIBench · no judge" in g and "MobileAIBench · judged" in g
    assert g.index("MobileAIBench · no judge") < g.index("MobileAIBench · judged")
    shot(page.locator("[data-lb-card]"), "mobileaibench-1400.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_mobile_mmlu_is_pro_with_its_categories(live, page, width):
    go(page, live, "view=mobile&group=mmlu", width)
    cats = page.evaluate("MMPD().categories")
    assert heads(page) == ["name", "params", "mobile_mmlu_pro", *[f"mmpcat:{c}" for c in cats]]
    th = page.locator(f"{LB} th[data-col='mobile_mmlu_pro']")
    assert th.locator("[data-mmp-provisional]").inner_text() == "provisional key"
    assert th.locator("[data-col-restriction]").count() == 1         # its badge
    # …and its categories', once, on their heading — not on each of them
    assert page.locator(f"{LB} [data-group-head-restriction]").count() == 1
    assert page.locator(f"{LB} th[data-col^='mmpcat:'] [data-col-restriction]").count() == 0
    # each category: good-750m's, as its page says it
    c0 = cats[0]
    want = page.evaluate(f"(DATA.models.find(m => m.id === '{GOOD}').mmp.by_category[{json.dumps(c0)}] "
                         "|| {}).acc")
    cell = page.locator(f"tr[data-lb-row='{GOOD}'] [data-mmpcat-cell={json.dumps(c0)}]")
    assert cell.inner_text() == ("—" if want is None else f"{100 * want:.1f}")
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"mobile-mmlu-{width}.png")
    assert page.errors == []


def test_devicemarks_table_and_its_chart_remembered_for_that_view(live, page):
    go(page, live, "view=mobile&group=devicemark")
    assert "dm:composite" in heads(page) and SETUP in row_ids(page)
    page.locator("[data-lb-show] [data-show='chart']").click()
    page.wait_for_selector("[data-ondevice]")
    assert page.evaluate("location.hash") == "#tab=models&view=mobile&group=devicemark&show=chart"
    shot(page.locator("[data-lb-card]"), "devicemark-chart-1400.png")
    # Standard opens on its own choice, a table; Mobile again: DeviceMark, its chart
    pick_view(page, "standard")
    page.wait_for_selector(f"{LB} tbody tr")
    assert page.locator("[data-lb-show]").get_attribute("data-lb-show") == "table"
    pick_view(page, "mobile")
    page.wait_for_selector("[data-ondevice]")
    assert page.locator("#pill-group").get_attribute("data-value") == "devicemark"
    assert page.errors == []


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTHS)
def test_the_chart_is_the_ranked_bars_for_what_the_toolbar_keeps(live, page, width):
    go(page, live, "group=commonsense", width)
    page.locator("[data-lb-show] [data-show='chart']").click()
    panels = page.locator("[data-chart-panels]")
    panels.wait_for()
    have = page.evaluate("['hellaswag', 'piqa', 'winogrande'].filter(t => "
                         "lbFilter(DATA.models).some(m => cell(t, m.id))).length")
    assert panels.get_attribute("data-chart-panels") == str(have) and have >= 1
    assert page.evaluate("location.hash") == "#tab=models&group=commonsense&show=chart"
    n_all = page.locator("[data-chart-panels] .panel").count()
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"chart-{width}.png")
    # the filters narrow it as they narrow the table: base models only
    page.locator("[data-filters]").click()
    page.locator("#pill-type").click()
    page.locator("#pop-type [data-choice='base']").click()
    page.wait_for_selector("[data-on='type']")
    assert page.locator("[data-chart-panels] .panel").count() <= n_all
    assert page.errors == []


def test_a_view_with_no_chart_says_why(live, page):
    go(page, live, "view=everyday", sel="[data-lb-everyday]")
    chart = page.locator("[data-lb-show] [data-show='chart']")
    assert chart.get_attribute("aria-disabled") == "true"
    chart.dispatch_event("click")
    assert page.locator("#why-no-chart").inner_text().startswith("Everyday has no chart")
    assert page.locator("[data-lb-everyday]").count() == 1
    assert page.errors == []


# ---------------------------------------------------------------------------
# Filters and Row 3
# ---------------------------------------------------------------------------

def test_source_type_and_row_3(live, page):
    go(page, live)
    n = page.evaluate("DATA.models.length")
    page.locator("[data-filters]").click()
    page.locator("#pill-source").click()
    items = page.locator("#pop-source [data-choice]").all_inner_texts()
    assert items[0] == "All" and [i.split(" (")[0] for i in items[1:]] == \
        ["Hugging Face", "Uploaded here", "GGUF file", "Served"]
    page.locator("#pop-source [data-choice='local']").click()
    page.wait_for_selector("[data-on='source']")
    assert page.locator("[data-filters]").inner_text() == "Filters · 1 ▾"
    assert page.locator("[data-on='source']").inner_text().startswith("Source: Uploaded here")
    kept = page.evaluate("lbFilter(DATA.models).map(m => m.id)")
    assert kept and all(k.startswith("local/") for k in kept)
    count = page.locator("[data-lb-count]")
    assert count.inner_text() == f"{len(kept)} of {n} models"
    # Type with it: an uploaded instruct model — none here: said, and Clear all
    page.locator("#pill-type").click()
    page.locator("#pop-type [data-choice='instruct']").click()
    page.wait_for_selector("[data-no-match]")
    assert page.locator("[data-no-match]").inner_text().startswith("No models match these filters.")
    assert page.locator("[data-on]").count() == 2
    shot(page.locator("[data-lb-card]"), "nothing-matches-1400.png")
    # a chip's × takes that one filter off
    page.locator("[data-on-remove='type']").click()
    page.wait_for_selector("[data-on='type']", state="detached")
    assert page.locator(f"{LB} tbody tr").count() >= 1
    page.locator("[data-clear-all]").click()
    page.wait_for_selector("[data-on]", state="detached")
    assert page.locator("[data-lb-count]").inner_text() == f"{n} of {n} models"
    assert page.evaluate("location.hash") == "#tab=models"
    assert page.errors == []


def test_tested_last_7_days_between_two_dates_and_no_date_said(live, page):
    go(page, live)
    page.locator("[data-filters]").click()
    page.locator("#pill-tested").click()
    page.locator("#pop-tested [data-choice='7d']").click()
    page.wait_for_selector("[data-on='tested']")
    assert page.locator("[data-on='tested']").inner_text().startswith("Tested: last 7 days")
    kept = page.evaluate("lbFilter(DATA.models).map(m => m.id)")
    assert GOOD in kept                                     # tested yesterday, by a run
    week = page.evaluate("Date.now() - 7 * 864e5")
    assert all(page.evaluate(f"testedMs(DATA.models.find(m => m.id === {json.dumps(k)}))") >= week
               for k in kept)
    # a model with no test date is never hidden silently
    nodate = page.evaluate("DATA.models.filter(m => testedMs(m) == null).length")
    if nodate:
        line = page.locator("[data-tested-hidden]")
        assert line.get_attribute("data-tested-hidden") == str(nodate)
        line.locator("[data-tested-show]").click()
        page.wait_for_function("lbS().tested === '7d,none'")
    shot(page.locator("[data-lb-card]"), "tested-7d-1400.png")
    # between two dates: one model's own day
    day = page.evaluate(f"testedDay(DATA.models.find(m => m.id === {json.dumps(SKEWED)}))")
    page.locator("#pill-tested").click()
    page.locator("#pop-tested [data-tested-from]").fill(day)
    page.locator("#pop-tested [data-tested-to]").fill(day)
    page.locator("#pop-tested [data-tested-apply]").click()
    page.wait_for_function(f"lbS().tested.startsWith({json.dumps(day + '..' + day)})")
    assert SKEWED in page.evaluate("lbFilter(DATA.models).map(m => m.id)")
    assert f"tested={day}..{day}" in page.evaluate("location.hash")
    assert page.errors == []


# ---------------------------------------------------------------------------
# Columns ▾
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTHS)
def test_columns_is_one_menu_that_keeps_both_jobs(live, page, width):
    go(page, live, width=width)
    page.locator("#pill-columns").click()
    menu = page.locator("#pop-columns")
    assert [s.get_attribute("data-columns-section") for s in
            menu.locator("[data-columns-section]").all()] == ["benchmarks", "details", "display"]
    mode = menu.locator("[data-columns-mode]")
    assert mode.get_attribute("data-columns-mode") == "shown"
    assert "the Avg stays the board’s" in mode.inner_text()
    no_sideways(page)
    shot(menu, f"columns-{width}.png")
    # showing a column leaves the board's Avg as it was
    hidden = menu.locator("[data-column-group='tasks'] input[data-column]:not(:checked)").first
    key = hidden.get_attribute("data-column")
    hidden.check()
    page.wait_for_selector(f"{LB} th[data-col='{key}']")
    assert page.locator(f"{LB} th[data-col='avg']").count() == 1
    # a table of the ticked ones, with their own Avg
    menu.locator("[data-columns-mode-pick='built']").check()
    page.wait_for_selector(f"{LB} th[data-col='cavg']")
    assert page.locator("#pill-columns").inner_text().startswith("Columns · Avg of ")
    assert page.locator(f"{LB} th[data-col='avg']").count() == 0
    menu.locator("[data-columns-mode-pick='shown']").check()
    page.wait_for_selector(f"{LB} th[data-col='avg']")
    page.keyboard.press("Escape")
    assert page.errors == []


def test_model_details_type_source_tested_and_size(live, page):
    go(page, live)
    page.locator("#pill-columns").click()
    details = page.locator("#pop-columns [data-columns-section='details']")
    assert details.locator("label").all_inner_texts() == \
        ["Size", "Family", "Type", "Source", "Tested", "Flags"]
    for k in ("kind", "source", "date"):
        details.locator(f"input[data-column='{k}']").check()
    page.keyboard.press("Escape")
    src = page.locator(f"tr[data-lb-row='{LOCAL}'] td[data-fact='source']")
    assert src.inner_text() == "Uploaded here"
    assert page.locator(f"tr[data-lb-row='{GOOD}'] td[data-fact='kind']").inner_text() in (
        "base", "instruct")                                  # never "checkpoint" (16.7)
    # Tested sorts
    page.locator(f"{LB} th[data-col='date']").click()
    assert page.locator(f"{LB} th[data-col='date']").get_attribute("aria-sort") != "none"
    # Size is shown until hidden
    page.locator("#pill-columns").click()
    details.locator("input[data-column='params']").uncheck()
    page.wait_for_selector(f"{LB} th[data-col='params']", state="detached")
    details.locator("input[data-column='params']").check()
    page.wait_for_selector(f"{LB} th[data-col='params']")
    page.keyboard.press("Escape")
    page.evaluate("localStorage.removeItem('bench-lb-facts')")
    assert page.errors == []


# ---------------------------------------------------------------------------
# keep working
# ---------------------------------------------------------------------------

def test_an_old_chip_address_and_a_saved_view_open_as_they_did(live, page):
    # 12h.2's saved view: its chip, its benchmarks, its models
    body = {"name": "Math pair", "by": "masein",
            "spec": {"chip": "math", "cols": None, "models": [GOOD, SKEWED]}}
    req = urllib.request.Request(live["base"] + "/api/views", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    vid = json.loads(urllib.request.urlopen(req, timeout=30).read())["id"]
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "chip=mobile")                           # Mobile tasks, before 16.3
    page.wait_for_function("location.hash === '#tab=models&view=mobile&group=mobileaibench'")
    pick_view(page, "standard")
    chip = page.locator(f".savedviews [data-saved-view='{vid}']")
    chip.wait_for()
    chip.click()
    page.wait_for_selector("#pill-group[data-value='math']")
    assert sorted(row_ids(page)) == sorted([GOOD, SKEWED])
    assert chip.get_attribute("aria-pressed") == "true"
    # an old Frontier chip: Frontier, on Row 1
    choose_chip(page, "frontier")
    page.wait_for_selector("[data-models-view='frontier'][aria-selected='true']")
    assert page.errors == []


def test_a_focused_tab_keeps_its_focus_across_a_poll(live, page):
    """found by 12b.2's tab test flaking on CI: a poll's redraw replaced the
    focused tab, and the arrow keys went nowhere"""
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + GOOD.replace("/", "%2F"))
    page.wait_for_selector("[data-model-tabs]")
    page.locator("[data-model-tabs] [aria-selected='true']").focus()
    page.evaluate("render()")                               # what a poll does
    assert page.evaluate("document.activeElement.dataset.mtab") is not None
    go(page, live)
    page.locator("[data-models-view='mobile']").focus()
    page.evaluate("render()")
    assert page.evaluate("document.activeElement.dataset.modelsView") == "mobile"
    assert page.errors == []
