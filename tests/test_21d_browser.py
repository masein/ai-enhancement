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

import re
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


def go(page, live, hash_="", width=1280, scheme="light", sel="[data-latest-table] tbody tr"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models" + (("&" + hash_) if hash_ else ""))
    page.wait_for_selector(sel, timeout=30000)


def no_sideways(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=str(SCREENS / name))


def first_text(page, sel: str) -> list[list[str]]:
    """each row: its model's name, then each cell's own number (its first line)"""
    return page.eval_on_selector_all(sel, """rows => rows.map(r => [r.children[0].title
      || r.children[0].textContent.trim(), ...[...r.children].slice(1).map(c => c.firstChild
      && c.firstChild.nodeType === 3 ? c.firstChild.textContent.trim() : c.textContent.trim())])""")


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
    heads = page.eval_on_selector_all("[data-latest-table] thead th",
                                      "ts => ts.map(t => t.textContent.trim())")
    assert heads == ["Model", "GPQA Diamond", "OTIS Mock AIME", "MMLU-Pro", "HLE", "Source"]
    rows = first_text(page, "[data-latest-table] tbody tr")
    # 23b: and step8804, not scored yet
    assert sorted(r[0] for r in rows) == sorted(["step8804", *PICKED, "Ornith-1.5 Q4_K_M",
                                                 *GLANCE])
    # no rank and no average: a name opens every row, no column is either
    assert not any(re.fullmatch(r"#?\d+", r[0]) for r in rows)
    assert not any(h.lower() in ("#", "rank", "avg", "average") for h in heads)
    # sorted by GPQA Diamond, the highest first, a dash last
    g = [float(r[1].split(" ")[0].rstrip("†")) if r[1] != "—" else -1 for r in rows]
    assert g == sorted(g, reverse=True), g
    # our rows bold; each published row says where it is from
    ours = page.locator("[data-latest-ours='true'] td:first-child")
    assert ours.count() == 7                       # 22: and Ornith-1.5; 23b: and step8804
    assert ours.first.evaluate("e => +getComputedStyle(e).fontWeight") >= 600
    src = {r[0]: r[-1] for r in rows}
    assert src["GPT-6 Astra (max)"] == "Epoch AI"
    assert src["DeepSeek-V4-Pro (max)"] == "Epoch AI; HLE, MMLU-Pro: DeepSeek card"
    assert src["o4-mini (high)"] == "Epoch AI; MMLU-Pro: Qwen card (medium effort)"
    assert src["step4000"].startswith("Our run · Complete")
    assert page.locator("[data-latest-table] [data-card-link]").count() >= 8
    # a card's number has no ±; ours always has one
    assert all("±" in r[1] for r in rows if r[0] in PICKED and r[1] != "—")
    # the score alone in a cell; how many questions, on hover
    s4 = page.locator("[data-latest-row='step4000'] td").nth(1)
    assert s4.inner_text() == "74.2 ± 2.6"
    assert s4.get_attribute("title") == "198 questions × 4 · 3 answers ran out"
    o4 = next(r for r in rows if r[0] == "o4-mini (high)")
    assert "±" not in o4[3]                                            # MMLU-Pro, from a card
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
    latest = {r[0]: r[1:5] for r in first_text(page, "[data-latest-table] tbody tr")}
    page.locator("[data-latest-open]").click()
    page.wait_for_selector("[data-rp-every] tbody tr")
    assert page.evaluate("location.hash") == "#tab=reports&report=step-checkpoints"
    every = {r[0]: r[1:] for r in first_text(page, "[data-rp-every] tbody tr")}
    order = ["gpqa", "hle", "otis", "math", "mmlupro", "simpleqa", "arc"]
    for name, cells in latest.items():
        by = dict(zip(order, every[name]))
        assert cells == [by["gpqa"], by["otis"], by["mmlupro"], by["hle"]], name
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
    go(page, live, "view=standard", width=width, scheme=scheme, sel="[data-lb-table] tbody tr")
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
    page.wait_for_selector("[data-lb-table]")
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
    # a score cell: "74.2 ± 2.6", then "198 questions × 4 · 3 answers ran out"
    cell = page.locator("[data-rp-row='step4000'] td").nth(1)
    assert cell.inner_text().startswith("74.2 ± 2.6")
    assert cell.locator("[data-rp-asked]").inner_text() == "198 questions × 4 · 3 answers ran out"
    arc = page.locator("[data-rp-row='step4000'] td").nth(7).locator("[data-rp-asked]")
    assert arc.inner_text() == "167 tasks × 2 · 3 answers ran out"
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
