"""16.8 on the page: what the deployed board showed after phase 16.

- **A name uses the width its column has** (16.3): each row's name block
  was capped on its own (a name at 190px), so "Qwen3.6-35B-…" was cut beside
  empty room. The widest row makes the column, and every row's name may use
  it; a name still cut shows its short form, so two rows on screen never read
  the same (lookahead, and lookahead + MTP, both read "Qwen3.6 k4-LDA · loo…").

At 1512 px, as on the server: Mobile, Standard, Everyday and Frontier. Served
setups named as the server's are, with DeviceMark rows written as runs write
them; nothing runs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from conftest import pick_view
from fake_openai import FakeServer
from service import config
from test_12q_devicemark_board import SERVED, _row
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16_8"
WIDE = 1512
# the server's setups (4 Oct): two that differ only at their end
SETUPS = [("Qwen3.6 k4-LDA · lookahead", {"phone": True, "lookahead": True, "mtp": False}),
          ("Qwen3.6 k4-LDA · lookahead + MTP", {"phone": True, "lookahead": True, "mtp": True}),
          ("Qwen3.6-35B-A3B-Q4-original-k-8", {"phone": False, "lookahead": False, "mtp": False}),
          ("Qwen3.6-35B-A3B-Q4-original-k-8 MTP", {"phone": False, "lookahead": False,
                                                  "mtp": True})]
ACC = {"ifeval": 0.6, "mmlu_pro": 0.5, "math": 0.4}


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def setups(live):
    import service.app as appmod
    fake = FakeServer()
    ids = []
    for name, su in SETUPS:
        got = api(live, "/api/served", {"name": name, "base_url": fake.base, "thinking": "off",
                                        "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                        "phone": su["phone"], "by": "masein"})
        mid = (got.get("model") or got)["id"]
        for thinking in (False, True):
            _row(config.OUT_DIR, mid, ACC, {**SERVED, **su, "name": name}, thinking=thinking)
        ids.append(mid)
    # Frontier: Epoch's reported scores, as 12n.2 imports them
    from test_reported_12m2 import epoch_fetch, epoch_zip
    from service import reported
    reported.import_epoch(fetch=epoch_fetch(epoch_zip()))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield ids
    fake.close()
    for mid in ids:
        for suffix in ("", "__thinking"):
            for f in (config.OUT_DIR / (mid.replace("/", "__") + suffix)).glob("devicemark*"):
                f.unlink()
    appmod._cache.update(key=None, payload=None, at=0.0)


NAMES = """() => [...document.querySelectorAll('[data-lb-card] table.lb tbody td.model')].map(td => {
  const n = td.querySelector('.mname'), b = td.querySelector('.mcell');
  const cut = n.dataset.cut === '1', s = n.querySelector('.mn-short');
  const shown = cut ? s.textContent : (n.querySelector('.mn-full') || n).textContent;
  const cs = getComputedStyle(td);
  return { id: td.dataset.model, shown, cut,
    over: n.scrollWidth > n.clientWidth + 1,
    room: td.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight),
    block: b ? b.getBoundingClientRect().width : null };
})"""


def names(page) -> list[dict]:
    page.wait_for_function("document.querySelector('[data-lb-card] table.lb tbody td.model')")
    page.wait_for_timeout(200)
    return page.evaluate(NAMES)


def check(got: list[dict], view: str) -> None:
    assert got, view
    for r in got:
        # a name is whole, or its block fills its column and its short form is shown, whole
        if r["cut"]:
            assert r["block"] >= r["room"] - 2, (view, r)
            assert not r["over"], (view, r)
        else:
            assert not r["over"], (view, r)
    shown = [r["shown"] for r in got]
    assert len(set(shown)) == len(shown), (view, shown)          # never two the same


@pytest.mark.parametrize("view", ["mobile", "standard", "everyday", "frontier"])
def test_a_name_uses_its_columns_width_and_two_never_read_the_same(live, page, setups, view):
    page.set_viewport_size({"width": WIDE, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    pick_view(page, view)
    got = names(page)
    check(got, view)
    if view == "mobile":
        by = {r["id"]: r for r in got}
        look, look_mtp = setups[0], setups[1]
        # the two that differ at their end read apart
        assert by[look]["shown"] != by[look_mtp]["shown"]
        assert "MTP" in by[look_mtp]["shown"] and "MTP" not in by[look]["shown"]
        # a name with room is whole: the original's
        assert by[setups[2]]["shown"] == "Qwen3.6-35B-A3B-Q4-original-k-8"
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(page.locator("[data-lb-table]"), SCREENS / "mobile-names-1512.png")
    assert page.errors == []


def test_the_fit_follows_the_window(live, page, setups):
    page.set_viewport_size({"width": WIDE, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    pick_view(page, "mobile")
    names(page)
    page.set_viewport_size({"width": 1024, "height": 1000})
    page.wait_for_timeout(300)
    check(names(page), "mobile at 1024")
    # narrow enough that names are cut: each shows its short form, whole,
    # and the two that differ at their end still read apart
    page.set_viewport_size({"width": 640, "height": 1000})
    page.wait_for_timeout(300)
    got = names(page)
    check(got, "mobile at 640")
    assert any(r["cut"] for r in got), got
    by = {r["id"]: r["shown"] for r in got}
    assert by[setups[0]] != by[setups[1]] and "MTP" in by[setups[1]]
    # a cut name keeps its front as far as it fits, and its end whole
    for i, end in ((2, "k-8"), (3, "k-8 MTP")):
        assert by[setups[i]].startswith("Qwen3.6") and by[setups[i]].endswith(end), by
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(page.locator("[data-lb-table]"), SCREENS / "mobile-names-640.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# one count; "prelim" on Standard alone; the Filters panel; a served model
# being checked
# ---------------------------------------------------------------------------

def counts(page) -> dict:
    c = page.locator("[data-lb-count]")
    table, nt, dup, setup, cannot = (int(x) for x in c.get_attribute("data-lb-counts").split("|"))
    n = int(c.get_attribute("data-lb-count").split("|")[0])
    return {"text": c.inner_text(), "n": n, "table": table, "not": nt, "rest": dup + setup + cannot}


@pytest.mark.parametrize("view,group", [("mobile", None), ("mobile", "mmlu"), ("standard", None)])
def test_one_count_says_where_each_model_is(live, page, setups, view, group):
    page.set_viewport_size({"width": WIDE, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table], [data-empty]")
    pick_view(page, view, group)
    page.wait_for_timeout(300)
    c = counts(page)
    # they add up: in the table, under "Not tested on this", and the rest said
    assert c["table"] + c["not"] + c["rest"] == c["n"], c
    assert c["text"].startswith(f"{c['n']} models: {c['table']} in the table"), c
    rows = page.locator("[data-lb-table] tbody tr[data-lb-row]")
    assert rows.count() == c["table"]
    nt = page.locator("[data-not-tested]")
    assert (int(nt.get_attribute("data-not-tested")) if nt.count() else 0) == c["not"]
    if c["not"]:
        assert f"{c['not']} not tested on this" in c["text"]
    # no second count beside it: the status line says the page only when paged
    status = page.locator("[data-statusline='models']")
    assert "Showing" not in (status.inner_text() if status.count() else "")
    empty = page.locator("[data-empty]")
    if empty.count():
        assert not any(ch.isdigit() for ch in empty.locator("p").inner_text().split("·")[0])
    if group:
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(page.locator("[data-lb-card]"), SCREENS / f"count-{view}-{group}-1512.png")
    assert page.errors == []


def test_prelim_is_standards_alone(live, page, setups):
    page.set_viewport_size({"width": WIDE, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    table = page.locator("[data-lb-table]")
    assert table.locator(".badge.prelim").count() >= 1           # Standard: its required seven
    pick_view(page, "mobile")
    page.wait_for_timeout(300)
    assert page.locator("[data-lb-table] .badge.prelim").count() == 0
    assert "prelim" not in page.locator("[data-lb-table]").inner_text()
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_the_filters_panel_is_one_row_of_compact_menus_that_wraps(live, page, width):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    page.locator("[data-filters]").click()
    sheet = page.locator("[data-filter-sheet]")
    sheet.wait_for()
    pills = sheet.locator(".pills > .pill, .pills > * > .pill, .fmore > .pill, .fmore > * > .pill")
    boxes = pills.evaluate_all("""ps => ps.map(p => p.getBoundingClientRect())
        .map(r => [r.left, r.top, r.width])""")
    sw = sheet.bounding_box()["width"]
    assert len(boxes) == 6, boxes
    assert all(w < sw * 0.6 for _, _, w in boxes), (sw, boxes)        # compact, never full width
    lines = sorted({round(top) for _, top, _ in boxes})
    assert len(lines) <= (1 if width >= 1400 else 3), lines            # one row, wrapping
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(sheet, SCREENS / f"filters-{width}.png")
    assert page.errors == []


def test_a_served_model_being_checked_holds_send_until_it_answers(live, page, setups,
                                                                  monkeypatch):
    from conftest import pg_choose, set_name
    from service import chat
    mid = setups[0]
    seen = {"up": None}
    real = chat.served_up
    monkeypatch.setattr(chat, "served_up", lambda m: seen["up"] if m == mid else real(m))
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=playground")
    page.wait_for_selector("[data-pg-main]")
    pg_choose(page, mid)
    assert page.locator(f"[data-pg-state-of='{mid}']").first.inner_text() == "Checking its server"
    assert page.locator("[data-pg-send]").is_disabled()
    assert page.locator(f"[data-pg-checking='{mid}']").inner_text().endswith(
        "checking that its server answers. Send waits for the answer.")
    assert page.locator(f"[data-pg-notnow='{mid}']").count() == 0       # not "Not now"
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(page.locator("[data-pg-main]"), SCREENS / "checking-1400.png")
    # the answer is in: Send comes back on its own
    seen["up"] = True
    page.wait_for_selector("[data-pg-send]:not([disabled])", timeout=20000)
    assert page.locator(f"[data-pg-checking='{mid}']").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# Read the questions: never a dead end
# ---------------------------------------------------------------------------

def test_read_the_questions_is_offered_where_they_can_be_read_and_says_why_where_not(live, page):
    from service import questions
    questions._ff.clear()
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=benchmarks")
    page.wait_for_selector("[data-catalog]")
    page.wait_for_function("state.qxList instanceof Set")
    listable = set(page.evaluate("[...state.qxList]"))
    # every link the catalogue offers leads to questions — none to a 404
    offered = page.locator("[data-cat-read]").evaluate_all("xs => xs.map(x => x.dataset.catRead)")
    assert offered and set(offered) <= listable
    # MobileAIBench's parts: read from their files though no model has answered
    assert "mab_hotpotqa" in listable and "mab_mtbench" in listable
    # where they can't be, the card says why in the link's place
    whys = page.locator("[data-cat-noread]").all_text_contents()     # some in folded suites
    assert whys and all(w in (questions.NOT_YET + ".", questions.NOT_LISTED + ".") for w in whys)
    assert questions.NOT_LISTED + "." in whys                       # GPQA, never
    # a part no model has run: its questions, from its file, said so
    page.goto(live["base"] + "/#tab=benchmarks&q=mab_xsum")
    page.wait_for_selector("[data-qx-counts]")
    if page.locator("[data-qx-from-file]").count():
        assert page.locator("[data-qx-from-file]").inner_text().startswith(
            "Read from its file on this server: no model has answered it yet")
    assert page.locator("[data-qx-error]").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# Models: the row boxes say what they are for
# ---------------------------------------------------------------------------

def test_the_compare_boxes_say_what_they_are_for_and_one_tick_says_what_next(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    tip = page.locator("[data-lb-table] th[data-col='name']").get_attribute("data-tip")
    assert "tick two to eight models, then Compare ▸ lays them side by side" in tip
    boxes = page.locator("[data-cmp-tick]")
    assert boxes.first.get_attribute("title") == "Tick two to eight models to compare them"
    boxes.nth(0).check()
    one = page.locator("[data-cmp-one]")
    one.wait_for()
    assert one.inner_text() == "Tick one more to compare"
    boxes.nth(1).check()
    page.wait_for_selector("[data-cmp-go='2']")
    assert page.locator("[data-cmp-one]").count() == 0
    boxes.nth(0).uncheck()
    boxes.nth(1).uncheck()
    assert page.errors == []


def test_insights_compare_shapes_says_nothing_of_the_exam_while_it_is_off(live, page,
                                                                         monkeypatch):
    import service.app as appmod
    from service import config as cfg
    monkeypatch.setattr(cfg, "KNOWLEDGE_EXAM", False)
    appmod._cache.update(key=None, payload=None, at=0.0)
    try:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto(live["base"] + "/#tab=models")
        page.wait_for_selector("[data-radar]")
        assert page.locator("[data-radar-judged-why]").count() == 0
        assert "Judged by area" not in page.locator("[data-radar]").inner_text()
    finally:
        appmod._cache.update(key=None, payload=None, at=0.0)
    assert page.errors == []


# ---------------------------------------------------------------------------
# Test a model: a served model's server, and the judge, before Start
# ---------------------------------------------------------------------------

def test_start_waits_for_a_served_models_server_and_an_offline_judge_is_said(live, page,
                                                                            setups,
                                                                            monkeypatch):
    from conftest import set_name
    from service import chat
    mid = setups[0]
    up = {"v": False}
    real = chat.served_up
    monkeypatch.setattr(chat, "served_up", lambda m: up["v"] if m == mid else real(m))
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto(live["base"] + "/#model=" + mid.replace("/", "%2F"))
    page.wait_for_selector(f"[data-test-model][data-test-this='{mid}']")
    page.locator("[data-test-model]").click()
    dlg = page.locator("[data-dialog='test']")
    why = dlg.locator("[data-why='server-down']")
    why.wait_for()
    page.wait_for_function("document.querySelector(\"[data-why='server-down']\")"
                           ".dataset.srvUp === 'false'")
    assert why.inner_text() == "Its server isn’t running."
    start = dlg.locator("[data-submit-form] button.primary")
    assert start.is_disabled()
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(dlg.locator("[data-submit-form]"), SCREENS / "server-down-1400.png")
    # its server starts: Start comes back on its own
    up["v"] = True
    page.wait_for_selector("[data-dialog='test'] [data-why='server-down']", state="detached",
                           timeout=15000)
    assert not start.is_disabled()
    # the judge offline, and a part it marks: said before Start
    page.evaluate("setJudgeHealth({ok: false, why: 'the grading model is not answering'})")
    page.evaluate("state.sub.suite = 'mobile'; state.sub.part = 'judged'; render()")
    line = dlg.locator("[data-judge-needed]")
    line.wait_for()
    assert line.inner_text() == ("The judge marks these answers, and it isn’t answering now: the "
                                 "grading model is not answering. The run can start; its marks "
                                 "wait until the judge answers.")
    assert page.errors == []
