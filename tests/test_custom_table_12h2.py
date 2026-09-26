"""12h.2 on the page: Models ▸ Standard, a table you build. Benchmarks ▾ and
Models ▾ sit beside Filters; the average is over the chosen benchmarks only
("Avg of 3"), with the chosen columns' errors combined and the same z-test for
bold; a chosen model missing one is not averaged but listed with what it is
missing and a Test. The address holds the view, a view can be saved for the
team (only the name that saved it renames or deletes it), and ⋯ copies the
table as CSV. At phone width the pickers stack under the chips and the table
scrolls in its own box."""

from __future__ import annotations

import csv
import io
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from generative_fixture import write_run

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12h2"
MODEL = "fx/below-135m-it"                  # the fixture's instruct model
THINK = MODEL + " · thinking"
SHORT = "fx/small-it-1b"                     # sat IFEval and MMLU-Pro, not MATH-500
BASE = "fx/good-750m"                        # a base model: instruct only for the three
THREE = ["ifeval", "mmlu_pro", "hendrycks_math500"]
THREE_URL = "cols=ifeval,mmlu_pro,math500"


@pytest.fixture(scope="module", autouse=True)
def generative_runs(live):
    """the instruct model sat the three, thinking off and on (on, a better
    IFEval); another instruct model sat two of them"""
    import service.app as appmod
    out = live["tree"]["out_dir"]
    write_run(out, MODEL, thinking=False)
    write_run(out, MODEL, thinking=True, ifeval=0.9)
    write_run(out, SHORT, tasks=("ifeval", "mmlu_pro"))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def models_tab(page, base, hash_rest="", width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=models" + ("&" + hash_rest if hash_rest else ""))
    page.wait_for_selector("[data-lb-card] [data-pickers]")


def choose(page, *tasks):
    """Benchmarks ▾ → Clear → tick each"""
    page.locator("#pill-benchmarks").click()
    page.locator("#pop-benchmarks").wait_for()
    page.locator("#pop-benchmarks [data-bench-clear]").click()
    page.wait_for_selector("[data-no-bench]")
    for t in tasks:
        page.locator(f"#pop-benchmarks [data-bench='{t}']").check()
    page.keyboard.press("Escape")
    page.wait_for_selector("th[data-col='cavg']" if tasks else "[data-no-bench]")


def rows(page):
    return [r.get_attribute("data-lb-row") for r in page.locator("tr[data-lb-row]").all()]


def tip_number(td):
    """a cell's number and its ±, from its tooltip: "76.9 ± 3.1" """
    first = json.loads(td.get_attribute("data-tip"))[0]
    v, _, se = first.partition(" ± ")
    return v, se


def expected_avg(page, mid, tasks):
    """the mean above chance, and √Σse²/k, from the payload's own cells"""
    got = page.evaluate("([m, ts]) => ts.map(t => [DATA.cells[t][m].v, DATA.cells[t][m].se, "
                        "DATA.tasks[t].chance])", [mid, tasks])
    vs, v2 = [], 0.0
    for v, se, c in got:
        k = 1 / (1 - c) if c is not None and 0 < c < 1 else 1.0
        vs.append(max(0.0, (v - c) * k) if k != 1.0 else v)
        v2 += (se * k) ** 2
    return sum(vs) / len(vs), math.sqrt(v2) / len(vs)


# ---------------------------------------------------------------------------
# 1. the average follows the choice
# ---------------------------------------------------------------------------

def test_three_benchmarks_make_an_average_of_those_three(live, page):
    models_tab(page, live["base"])
    assert page.locator("#pill-benchmarks").inner_text() == "Benchmarks: 6 ▾"    # today's six
    choose(page, *THREE)
    assert page.locator("#pill-benchmarks").inner_text() == "Benchmarks: 3 ▾"
    heads = [h.lower() for h in page.locator("[data-lb-table] thead tr.names th .hname")
             .all_inner_texts()]
    # 12i.0: it says what it is
    assert heads == ["model", "params", "avg above chance", "ifeval", "mmlu-pro", "math-500"]
    assert page.locator("th[data-col='avg']").count() == 0        # not the board's Avg
    # the rows: the models with all three, ranked on this average
    assert rows(page) == [THINK, MODEL]                             # thinking: the better IFEval
    assert page.locator(f"tr[data-lb-row='{THINK}'] td.rank").inner_text() == "1"
    # the number and its ±, as the chosen columns make them
    for mid in (MODEL, THINK):
        v, se = tip_number(page.locator(f"td[data-cavg='{mid}']"))
        want_v, want_se = expected_avg(page, mid, THREE)
        assert (v, se) == (f"{100 * want_v:.1f}", f"{100 * want_se:.1f}"), mid
    # …which is not the board's Avg of the required tasks
    assert page.evaluate(f"DATA.models.find(m => m.id === {json.dumps(MODEL)}).avg") \
        != pytest.approx(expected_avg(page, MODEL, THREE)[0])
    # bold by the same z-test as Avg: the best, and whatever its noise cannot tell apart
    (va, sa), (vb, sb) = (expected_avg(page, m, THREE) for m in (THINK, MODEL))
    tied = abs(va - vb) / math.sqrt(sa * sa + sb * sb) <= 1.96
    assert page.locator(f"td[data-cavg='{THINK}']").get_attribute("data-lead") == "1"
    assert (page.locator(f"td[data-cavg='{MODEL}']").get_attribute("data-lead") == "1") == tied
    # one line says what is shown
    assert page.locator("[data-custom-what]").inner_text() == \
        "Custom · IFEval, MMLU-Pro, MATH-500 · 2 models"
    assert page.evaluate("location.hash") == "#tab=models&" + THREE_URL
    assert page.errors == []


def test_a_model_missing_one_is_not_averaged_and_says_what_is_missing(live, page):
    models_tab(page, live["base"], THREE_URL)
    assert SHORT not in rows(page)
    page.locator("[data-not-tested-toggle]").click()
    short = page.locator(f"[data-not-tested-row='{SHORT}']")
    name = page.evaluate(f"DATA.models.find(m => m.id === {json.dumps(SHORT)}).name")
    assert short.inner_text() == f"{name} · no MATH-500 · Test"
    # a base model cannot sit them: said, and no Test
    base = page.locator(f"[data-missing='{BASE}']")
    assert base.inner_text() == " · no IFEval, MMLU-Pro, MATH-500 · instruct only"
    assert page.locator(f"[data-not-tested-test='{BASE}']").count() == 0
    shot(page, "12h2-custom-1400-light.png", full_page=True)
    # Test opens the form, filled in for what is missing
    page.locator(f"[data-not-tested-test='{SHORT}']").click()
    page.locator("[data-dialog='test']").wait_for()
    assert page.evaluate("[state.sub.hf_id, state.sub.suite]") == [SHORT, "generative"]
    page.keyboard.press("Escape")
    assert page.errors == []


def test_a_chip_fills_the_checklist_and_a_tick_makes_it_custom(live, page):
    models_tab(page, live["base"])
    assert page.locator("[data-custom-line]").count() == 0          # today's table: no line
    page.locator("[data-chip='instruction']").click()
    page.wait_for_selector("th[data-col='ifeval']")
    assert page.locator("#pill-benchmarks").inner_text() == "Benchmarks: 3 ▾"
    page.locator("#pill-benchmarks").click()
    ticked = page.locator("#pop-benchmarks [data-bench]:checked")
    assert sorted(ticked.evaluate_all("es => es.map(e => e.dataset.bench)")) == sorted(THREE)
    # every Standard benchmark, in its chip groups — no Everyday, no exam, no perplexity
    groups = page.locator("#pop-benchmarks [data-bench-group]").evaluate_all(
        "es => es.map(e => e.dataset.benchGroup)")
    # 12i.0: a group nothing has run yet (Math: GSM8K) is listed too, greyed
    assert groups == ["knowledge", "commonsense", "reasoning", "math", "truthfulness",
                      "instruction"]
    offered = page.locator("#pop-benchmarks [data-bench]").evaluate_all(
        "es => es.map(e => e.dataset.bench)")
    assert not [t for t in offered if t.startswith(("exam_", "fr_", "everyday")) or t == "mmlu_perm"]
    shot(page, "12h2-benchmarks-1400-light.png")
    # the search narrows the list
    page.locator("#pop-benchmarks input[type=search]").fill("math")
    assert page.locator("#pop-benchmarks [data-bench]").evaluate_all(
        "es => es.map(e => e.dataset.bench)") == ["hendrycks_math500"]
    page.locator("#pop-benchmarks input[type=search]").fill("")
    page.locator("#pop-benchmarks [data-bench='hendrycks_math500']").uncheck()
    page.keyboard.press("Escape")
    page.wait_for_selector("th[data-col='cavg']")
    assert page.locator("th[data-col='cavg'] .hname").inner_text().lower() == "avg above chance"
    assert page.locator("#pill-benchmarks").inner_text() == "Benchmarks: 2 ▾"
    assert page.locator("[data-chip='instruction']").get_attribute("aria-pressed") == "false"
    # the chip again: today's view, no line
    page.locator("[data-chip='instruction']").click()
    page.wait_for_selector("[data-custom-line]", state="detached")
    assert page.locator("th[data-col='cavg']").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2. choosing models
# ---------------------------------------------------------------------------

def test_a_model_subset_shows_only_those_rows_and_all_ranked_brings_the_default(live, page):
    models_tab(page, live["base"])
    n = len(rows(page))
    page.locator("#pill-models").click()
    panel = page.locator("#pop-models")
    panel.wait_for()
    # grouped as the board groups them
    assert panel.locator("[data-model-group]").evaluate_all(
        "es => es.map(e => e.dataset.modelGroup)")[:2] == ["instruct", "base"]
    shot(page, "12h2-models-1400-light.png")
    # 12i.0: each tick applies at once — there is no Apply
    panel.get_by_role("button", name="Clear").click()
    page.wait_for_selector("[data-no-models]")
    panel.locator(f"[data-model-pick='{MODEL}']").check()
    panel.locator(f"[data-model-pick='{BASE}']").check()
    page.wait_for_function("document.querySelectorAll('tr[data-lb-row]').length === 2")
    page.keyboard.press("Escape")
    assert sorted(rows(page)) == sorted([MODEL, BASE])
    assert page.locator("#pill-models").inner_text() == "Models: 2 ▾"
    assert page.locator("[data-custom-what]").inner_text() == "Custom · All tasks · 2 models"
    assert "models=" + quote(MODEL, safe="") in page.evaluate("location.hash")
    # one click on a name still opens the model
    page.locator(f"tr[data-lb-row='{BASE}'] a.mname").click()
    page.wait_for_selector("[data-model-hero]")
    page.go_back()
    page.wait_for_selector("tr[data-lb-row]")
    page.locator("#pill-models").click()
    page.locator("#pop-models [data-models-default]").click()
    page.wait_for_function(f"document.querySelectorAll('tr[data-lb-row]').length === {n}")
    assert page.locator("#pill-models").inner_text() == "Models: all ▾"
    assert page.locator("[data-custom-line]").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# 3. the address holds the view
# ---------------------------------------------------------------------------

def test_the_url_round_trips(live, page, browser):
    models_tab(page, live["base"])
    choose(page, *THREE)
    page.locator("#pill-models").click()
    page.locator("#pop-models").get_by_role("button", name="Clear").click()
    page.locator(f"#pop-models [data-model-pick='{MODEL}']").check()
    page.locator(f"#pop-models [data-model-pick='{SHORT}']").check()
    page.keyboard.press("Escape")
    page.wait_for_function("document.querySelectorAll('tr[data-lb-row]').length === 1")
    url = page.url
    assert url.endswith("#tab=models&" + THREE_URL + "&models="
                        + ",".join(quote(m, safe="") for m in (MODEL, SHORT)))

    def table(p):
        return {"heads": p.locator("[data-lb-table] thead tr.names th .hname").all_inner_texts(),
                "rows": rows(p), "avg": [tip_number(td) for td in p.locator("td[data-cavg]").all()],
                "line": p.locator("[data-custom-what]").inner_text(),
                "missing": p.locator("[data-not-tested]").get_attribute("data-not-tested")}
    first = table(page)
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, reduced_motion="reduce")
    try:
        other = ctx.new_page()
        other.goto(url)
        other.wait_for_selector("td[data-cavg]")
        assert table(other) == first
        assert other.url == url
        # the harness's name for MATH-500 opens it too, and the address says it the short way
        other.goto(live["base"] + "/#tab=models&cols=ifeval,mmlu_pro,hendrycks_math500")
        other.wait_for_selector("th[data-col='cavg']")
        other.wait_for_function("location.hash.endsWith('cols=ifeval,mmlu_pro,math500')")
        # a name the board does not know is dropped; none left is today's table
        other.goto(live["base"] + "/#tab=models&cols=nonesuch")
        other.wait_for_selector("[data-lb-table]")
        assert other.locator("th[data-col='avg']").count() == 1
        assert other.locator("[data-custom-line]").count() == 0
    finally:
        ctx.close()
    assert page.errors == []


# ---------------------------------------------------------------------------
# 4. saved views: the team's, changed only by the name that saved them
# ---------------------------------------------------------------------------

def named(browser, who):
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, reduced_motion="reduce")
    ctx.add_init_script(f"localStorage.setItem('bench-name', {json.dumps(who)})")
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.errors.append(f"console.error: {m.text}")
          if m.type == "error" else None)
    return ctx, pg


def call(base, method, path, body):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_a_saved_view_is_the_teams_and_only_its_owner_changes_it(live, browser):
    base = live["base"]
    ca, pa = named(browser, "masein")
    cb, pb = named(browser, "omar")
    try:
        pa.goto(base + "/#tab=models&" + THREE_URL)
        pa.wait_for_selector("td[data-cavg]")
        pa.locator("[data-save-view]").click()
        pa.locator("[data-view-name]").fill("Phone shortlist")
        pa.locator("[data-action='view-save']").click()
        chip = pa.locator("[data-saved-view]", has_text="Phone shortlist")
        chip.wait_for()
        assert chip.get_attribute("aria-pressed") == "true"         # it is the view shown
        assert pa.locator(".lbbar .chipdiv").count() == 1             # after the groups
        vid = chip.get_attribute("data-saved-view")
        assert pa.locator(f"[data-view-menu='{vid}']").count() == 1   # its owner's ⋯
        pa.locator(f"[data-view-menu='{vid}']").click()
        shot(pa, "12h2-saved-view-1400-light.png")
        pa.keyboard.press("Escape")

        # someone else: the chip, who saved it, and no ⋯
        pb.goto(base + "/#tab=models")
        other = pb.locator(f"[data-saved-view='{vid}']")
        other.wait_for()
        assert other.get_attribute("title") == "saved by masein"
        assert pb.locator(f"[data-view-menu='{vid}']").count() == 0
        other.click()
        pb.wait_for_selector("td[data-cavg]")
        assert pb.evaluate("location.hash") == "#tab=models&" + THREE_URL
        # …and the service refuses them, whatever the page shows
        status, j = call(base, "PATCH", f"/api/views/{vid}", {"name": "Mine now", "by": "omar"})
        assert status == 403 and j["detail"] == "only masein, who saved this view, can rename it"
        status, j = call(base, "DELETE", f"/api/views/{vid}", {"by": "omar"})
        assert status == 403 and j["detail"] == "only masein, who saved this view, can delete it"

        # its owner renames it, and everyone sees the new name
        pa.locator(f"[data-view-menu='{vid}']").click()
        pa.locator(f"[data-view-rename='{vid}']").click()
        pa.locator("[data-view-name]").fill("Phone picks")
        pa.locator("[data-action='view-rename']").click()
        pa.locator(f"[data-saved-view='{vid}']", has_text="Phone picks").wait_for()
        pb.reload()
        pb.locator(f"[data-saved-view='{vid}']", has_text="Phone picks").wait_for()
        # …and deletes it, after saying so
        pa.locator(f"[data-view-menu='{vid}']").click()
        pa.locator(f"[data-view-delete='{vid}']").click()
        assert pa.locator("[data-view-form='delete']").inner_text().startswith(
            "Delete “Phone picks” for the whole team?")
        pa.locator("[data-action='view-delete']").click()
        pa.wait_for_selector(f"[data-saved-view='{vid}']", state="detached")
        assert pa.locator(".lbbar .chipdiv").count() == 0
        assert pa.errors == [] and pb.errors == []
    finally:
        ca.close()
        cb.close()


# ---------------------------------------------------------------------------
# 5. Copy as CSV
# ---------------------------------------------------------------------------

def test_the_csv_is_the_table_with_the_average_and_its_errors(live, page):
    models_tab(page, live["base"], THREE_URL)
    page.wait_for_selector("td[data-cavg]")
    got = list(csv.reader(io.StringIO(page.evaluate("lbCsv()"))))
    assert got[0] == ["#", "Model", "Params", "Avg above chance", "Avg above chance ±", "IFEval",
                      "IFEval ±",
                      "MMLU-Pro", "MMLU-Pro ±", "MATH-500", "MATH-500 ±"]
    want = []
    for tr in page.locator("tr[data-lb-row]").all():
        mid = tr.get_attribute("data-lb-row")
        params = tr.locator("td").nth(2).inner_text()
        line = [tr.locator("td.rank").inner_text(), tr.locator("a.mname").inner_text(),
                "" if params == "—" else params]                  # a dash is an empty field
        for td in [page.locator(f"td[data-cavg='{mid}']"),
                   *(tr.locator(f"td[data-gen-cell='{t}']") for t in THREE)]:
            line.extend(tip_number(td))
        want.append(line)
    assert got[1:] == want and len(want) == 2
    page.locator("[data-custom-more]").click()
    page.locator("[data-copy-csv]").click()
    page.wait_for_selector("[data-toast='copy']")
    assert "Copied 2 rows as CSV" in page.locator("[data-toast='copy']").inner_text()
    assert page.errors == []


# ---------------------------------------------------------------------------
# 6. a phone
# ---------------------------------------------------------------------------

def test_at_phone_width_the_pickers_stack_and_the_table_scrolls_in_its_box(live, page):
    seven = "cols=mmlu,hellaswag,arc_challenge,arc_easy,winogrande,piqa,truthfulqa_mc2"
    models_tab(page, live["base"], seven, width=400)
    page.wait_for_selector("td[data-cavg]")
    box = page.evaluate("""() => {
      const r = s => document.querySelector(s).getBoundingClientRect();
      const w = document.querySelector('.lb-wrap');
      return { chipsBottom: r('.lbbar .chips').bottom, pickTop: r('[data-pickers]').top,
               pickRight: Math.max(...[...document.querySelectorAll('[data-pickers] .pill')]
                 .map(b => b.getBoundingClientRect().right)),
               page: document.scrollingElement.scrollWidth, vw: innerWidth,
               wrap: [w.scrollWidth, w.clientWidth, getComputedStyle(w).overflowX] }; }""")
    assert box["pickTop"] >= box["chipsBottom"] - 1          # under the chips, a row of their own
    assert box["pickRight"] <= box["vw"]
    assert box["page"] <= box["vw"]                           # the page never scrolls sideways…
    sw, cw, ov = box["wrap"]
    assert sw > cw and ov in ("auto", "scroll")               # …the table does, in its own box
    shot(page, "12h2-phone-400-light.png", full_page=True)
    page.locator("#pill-benchmarks").click()
    page.locator("#pop-benchmarks").wait_for()
    shot(page, "12h2-phone-benchmarks-400-light.png")
    assert page.errors == []
