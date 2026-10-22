"""21d in a real browser, at 1280 and 400 px, light and dark: Models opens on
Latest when the link names no view — exactly the report's rows marked
"glance" (22: ours too, and Ornith-1.5's two), on GPQA Diamond, OTIS Mock AIME,
MMLU-Pro and HLE,
no rank and no average, every number the report's; When ▾ beside Group and
Models, by the date tested or added, kept in the link; the report's token
table with its definitions, and its score lines in questions × runs. The
world is 21c's (the 14 Oct scorecard as the board would hold it, invented
questions) on the fixture board; nothing is fetched, run or graded."""

from __future__ import annotations

import json
import time
from contextlib import closing
from pathlib import Path

import pytest

from test_21c_page import SPEC, world
from test_21d_glance import GLANCE, PICKED

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase21d"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
# our six rows, tested now — 22: Ornith-1.5's isn't run in this world
LDA = [SPEC[x]["row"] for x in SPEC if x not in ("Ornith-1.5 Q4_K_M", "step8804")]  # 23b


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """the scorecard's world; good-750m first added 40 days ago and tested
    yesterday, skewed-360m first added yesterday and never tested since"""
    import service.app as appmod
    from service import db
    world()
    good = db.add(GOOD, "instruct", "full", "masein", "", status="done")
    skewed = db.add(SKEWED, "instruct", "full", "masein", "", status="failed")
    with closing(db._conn()) as c:
        c.execute("UPDATE submissions SET created_at=?, finished_at=? WHERE id=?",
                  (time.time() - 40 * 86400, time.time() - 86400, good))
        c.execute("UPDATE submissions SET created_at=? WHERE id=?",
                  (time.time() - 86400, skewed))
        c.commit()
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, hash_="", width=1280, scheme="light", sel=None):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models" + (("&" + hash_) if hash_ else ""))
    # 27: under 640 px a card a model, else the table's rows
    page.wait_for_selector(sel or ("[data-lb-card] [data-mt-card]" if width < 640
                                   else "[data-lb-card] table tbody tr.mt-row"), timeout=30000)


# 27: Latest's rows by their report labels (a row two sources make one, each),
# each benchmark's own number, and the Source column
LATEST = """() => [...document.querySelectorAll('[data-latest-table] tbody tr.mt-row')].map(tr => ({
  labels: (tr.dataset.latestRows || tr.dataset.latestRow).split('|'), ours: tr.dataset.latestOurs,
  name: tr.querySelector('[data-mt-name]').textContent, group: tr.dataset.mtGroup,
  mark: tr.querySelector('.mt-mark').dataset.mtMark,
  cells: Object.fromEntries([...tr.querySelectorAll('td[data-latest-cell]')].map(td => {
    const n = td.querySelector('.mt-n');
    return [td.dataset.latestCell, n ? n.firstChild.textContent : td.textContent.trim()]; })),
  se: Object.fromEntries([...tr.querySelectorAll('td[data-latest-cell]')].map(td =>
    [td.dataset.latestCell, !!td.querySelector('.mt-se')])),
  src: (tr.querySelector('[data-latest-source]') || {}).textContent || '' }))"""


def no_sideways(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=str(SCREENS / name))


def first_text(page, sel: str) -> list[list[str]]:
    """each row: its model's name, then each cell's own number (its first line);
    27b: a report's table is modelTable's, a row by each label it answers to"""
    from values27b import MT_FIRST
    return page.eval_on_selector_all(sel, MT_FIRST)


# ---------------------------------------------------------------------------
# Models ▸ Latest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_models_opens_on_latest(live, page, width, scheme):
    go(page, live, width=width, scheme=scheme)
    assert page.evaluate("location.hash") == "#tab=models"           # the link names no view
    chip = page.locator("[data-models-view='latest']")
    assert chip.get_attribute("aria-selected") == "true"
    assert page.locator("[data-models-view]").first.get_attribute("data-models-view") == "latest"
    # 27: each column's short name, never cut, what it tests and how many under it
    heads = page.eval_on_selector_all("[data-latest-table] thead tr.mt-hrow th .mt-h1",
                                      "ts => ts.map(t => t.textContent.replace(/[▲▼]/g, '').trim())")
    # 27c: IFEval and IFBench too, in the menu's one order
    assert heads == ["Model", "GPQA", "HLE", "OTIS", "MMLU-Pro", "IFEval", "IFBench", "Source"]
    assert page.locator("[data-latest-table] th[data-latest-col='gpqa'] .mt-h2").text_content() \
        == "science · 198 × 4"
    rows = page.evaluate(LATEST)
    # 27: Qwen's report and Epoch's run of one model are one row; step8804 (23b: not
    # scored yet) and Ornith-1.5 (22: not run in this world) have no score shown, so
    # are hidden and counted under the table, a click away
    assert sorted(x for r in rows for x in r["labels"]) == sorted([*PICKED, *GLANCE])
    hid = page.locator("[data-latest] [data-mt-empty]")
    assert hid.inner_text() == "2 more models have no score in these columns · show them"
    assert len({r["name"] for r in rows}) == len(rows)          # never two rows with one name
    # no rank and no average
    assert not any(h.lower() in ("#", "rank", "avg", "average") for h in heads)
    # three groups, ours first; in each, GPQA Diamond's highest first, a dash last
    groups = [r["group"] for r in rows]
    assert groups == sorted(groups, key=["ours", "open", "closed"].index)
    for gname in set(groups):
        g = [float(r["cells"]["gpqa"].rstrip("†")) if r["cells"]["gpqa"] != "–" else -1
             for r in rows if r["group"] == gname]
        assert g == sorted(g, reverse=True), (gname, g)
    # ours with a filled dot, the published with a ring; each published row says where it is from
    assert sum(r["ours"] == "true" for r in rows) == 5     # 22: Ornith-1.5, 23b: step8804, hidden
    assert all((r["mark"] == "here") == (r["ours"] == "true") for r in rows)
    src = {x: r["src"] for r in rows for x in r["labels"]}
    assert src["GPT-6 Astra (max)"] == "Epoch AI"
    assert src["DeepSeek-V4-Pro (max)"] == "Epoch AI; HLE, MMLU-Pro: DeepSeek card"
    assert src["o4-mini (high)"] == "Epoch AI; MMLU-Pro: Qwen card (medium effort)"
    # 25: the report's IFEval and IFBench not in yet (plan I)
    assert src["step4000"].startswith("Our run · 7 of 9 benchmarks in")
    # a card's number: italics, its letter linking the card, no ±; ours always has one
    assert page.locator("[data-latest-table] [data-card-link]").count() >= 6
    assert all(r["se"]["gpqa"] for r in rows if set(r["labels"]) & set(PICKED)
               and r["cells"]["gpqa"] != "–")
    # the score in a cell, its ± small beside it; how many questions, on hover
    s4 = page.locator("[data-latest-row='step4000'] td[data-latest-cell='gpqa']")
    assert s4.locator(".mt-n").inner_text() == "74.2" and s4.locator(".mt-se").inner_text() == "±2.6"
    assert {"198 questions × 4", "3 answers ran out"} <= set(json.loads(s4.get_attribute("data-tip")))
    o4 = page.locator("[data-latest-row='o4-mini (high)'] td[data-latest-cell='mmlupro']")
    assert o4.locator(".mt-se").count() == 0 and o4.locator("[data-mt-src]").inner_text() == "Q"
    line = page.locator("[data-latest-from]").inner_text()
    assert line.startswith("From the step-checkpoints report · as of ") and "Open the report" in line
    assert "Score: the average over every answer" in page.locator(
        "[data-latest-legend]").inner_text()
    assert no_sideways(page)
    assert page.errors == []
    page.evaluate("window.scrollTo(0, 0)")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / f"latest-{width}-{scheme}.png"), full_page=True)


def test_latest_equals_the_report_and_a_named_view_stays(live, page):
    go(page, live)
    latest = page.evaluate(LATEST)
    page.locator("[data-latest-open]").click()
    page.wait_for_selector("[data-rp-every] tbody tr")
    assert page.evaluate("location.hash") == "#tab=reports&report=step-checkpoints"
    every = {r[0]: r[1:] for r in first_text(page, "[data-rp-every] tbody tr")}
    order = ["gpqa", "hle", "otis", "math", "mmlupro", "simpleqa", "arc"]
    num = lambda x: x.split(" ")[0].replace("—", "–")             # noqa: E731
    for r in latest:
        # a row is its label's numbers; a row two sources make shows one of theirs a
        # cell. 27b: the report's table makes one row of a model's two sources
        # where Latest shows one (its glance rows): an empty cell there may be
        # the other source's number here
        by = [dict(zip(order, every[x])) for x in r["labels"]]
        for k in ("gpqa", "otis", "mmlupro", "hle"):
            assert r["cells"][k] == "–" or r["cells"][k] in {num(b[k]) for b in by}, (
                r["labels"], k)
    # Standard, named in the link, stays Standard on a reload
    go(page, live)
    page.locator("[data-models-view='standard']").click()
    page.wait_for_selector("[data-lb-table]")
    assert "view=standard" in page.evaluate("location.hash")
    page.reload()
    page.wait_for_selector("[data-lb-table]")
    assert page.locator("[data-models-view='standard']").get_attribute("aria-selected") == "true"
    # a plain link opens on Latest again
    go(page, live)
    assert page.locator("[data-models-view='latest']").get_attribute("aria-selected") == "true"
    assert page.errors == []


# ---------------------------------------------------------------------------
# When ▾
# ---------------------------------------------------------------------------

def kept(page) -> set[str]:
    return set(page.evaluate("lbFilter(DATA.models).map(m => m.id)"))


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (400, "dark")])
def test_when_by_tested_and_by_added_and_kept_in_the_link(live, page, width, scheme):
    go(page, live, "view=standard", width=width, scheme=scheme)
    pill = page.locator("[data-pickers] #pill-when")
    assert pill.inner_text().startswith("When: any time")
    # beside Group and Models, never inside Filters
    order = page.eval_on_selector_all("[data-pickers] > *", "ns => ns.map(n => n.id)")
    assert order.index("pill-when") == order.index("pill-models") + 1
    pill.click()
    page.locator("#pop-when [data-choice='7d']").click()
    page.wait_for_selector("[data-on='tested']")
    assert pill.inner_text().startswith("When: tested in the last 7 days")
    # tested: good-750m's run finished yesterday, and our six rows today
    assert kept(page) == {GOOD, *LDA}, kept(page) ^ {GOOD, *LDA}
    if width == 1280:
        page.locator("#pill-when").click()
        page.wait_for_selector("#pop-when [data-when-date='added']")
        shot(page.locator("[data-lb-card]"), f"when-{width}-{scheme}.png")
        page.keyboard.press("Escape")
    # added: skewed-360m's first submission was yesterday, good-750m's 40 days ago
    page.locator("#pill-when").click()
    page.locator("#pop-when [data-when-date='added']").click()
    page.wait_for_function("lbS().date === 'added'")
    assert page.locator("#pill-when").inner_text().startswith("When: added in the last 7 days")
    assert kept(page) == {SKEWED}, kept(page)
    assert "tested=7d" in page.evaluate("location.hash") and "date=added" in page.evaluate(
        "location.hash")
    assert page.locator("[data-tested-hidden]").inner_text().startswith(
        f"{page.evaluate('DATA.models.filter(m => addedMs(m) == null).length')} models have no "
        "date added")
    # the choice survives a reload from the link
    page.reload()
    page.wait_for_selector("[data-lb-table]", state="attached")
    assert page.locator("#pill-when").inner_text().startswith("When: added in the last 7 days")
    assert kept(page) == {SKEWED}
    assert no_sideways(page)
    assert page.errors == []


# ---------------------------------------------------------------------------
# the report: its token table's definitions, its score lines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_reports_numbers_say_what_they_are(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    page.wait_for_selector("[data-rp-tok]", timeout=30000)
    # a score cell: "74.2", its ± small; 27b: "198 questions × 4 · 3 answers ran
    # out" on its hover, the column's head "science · 198 × 4"
    cell = page.locator("[data-rp-row='step4000'] td[data-mt-col='gpqa']")
    assert cell.locator(".mt-n").inner_text() == "74.2"
    assert cell.locator(".mt-se").inner_text() == "±2.6"
    assert "198 questions × 4 · 3 answers ran out" in cell.get_attribute("data-tip").replace(
        '","', " · ")
    assert page.locator("[data-rp-ours] th[data-mt-col='gpqa'] .mt-h2").inner_text() == (
        "science · 198 × 4")
    arc = page.locator("[data-rp-row='step4000'] td[data-mt-col='arc']").get_attribute("data-tip")
    assert "167 tasks × 2" in arc and "3 answers ran out" in arc
    assert "792 answers" not in page.locator("[data-rp-ours]").inner_text()
    # 21d.1: a loop example's length is its thinking, as the page says it
    page.wait_for_selector("[data-rp-loop='486'] [data-rp-thought='against']", timeout=30000)
    assert page.locator("[data-rp-loop='486'] [data-rp-thought='against']").inner_text() \
        .startswith("Q4 (UD-Q4_K_XL) thought for ")
    assert page.locator("[data-rp-loop='486'] [data-rp-thought='first']").inner_text() \
        .startswith("step4000 thought for ")
    assert page.locator("[data-rp-score-legend]").inner_text().startswith(
        "Score: the average over every answer, in percent. ±: its standard error")
    # the token table: the page's names, each with its definition
    heads = page.eval_on_selector_all("[data-rp-tokens] thead th", "ts => ts.map(t => "
                                      "t.dataset.rpTokCol)")
    # 23b: and "Ended in a loop", by the board's loop rule
    assert heads == ["Benchmark", "Answers", "Average thinking", "Median thinking", "Output p90",
                     "Average output, right", "Average output, wrong", "Ran out", "Output total",
                     "Ended in a loop"]
    qs = page.locator("[data-rp-tokens] thead .rp-q")
    assert qs.count() == 9                                   # every column but Benchmark
    defs = dict(zip(page.eval_on_selector_all("[data-rp-tok-defs] dt", "d => d.map(x => "
                                              "x.textContent)"),
                    page.eval_on_selector_all("[data-rp-tok-defs] dd", "d => d.map(x => "
                                              "x.textContent)")))
    assert len(defs) == 6
    for i in range(8):
        assert qs.nth(i).get_attribute("title") in defs.values()
    p90 = page.locator("[data-rp-q='Output p90']")
    p90.scroll_into_view_if_needed()
    p90.click()
    pop = page.locator("#pop-rpq-output-p90")
    pop.wait_for()
    assert pop.inner_text().startswith("Output p90") and defs["Output p90"] in pop.inner_text()
    if width == 1280 or scheme == "light":
        shot(page.locator("[data-report-section='tokens']"), f"tokens-{width}-{scheme}.png")
    page.keyboard.press("Escape")
    assert no_sideways(page)
    assert page.errors == []
