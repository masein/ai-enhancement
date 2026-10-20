"""23e in a real browser, at 1280 and 400 px, light and dark: "Across
training" — a panel a benchmark, wrapping to one column on a phone, a
checkpoint not scored on one said under it, a segment's label opening the
two-builds report on that pair; "near the ceiling" where the report names
MATH Level 5; and the answers behind "had it, then lost it", one opened with
its marks; nothing scrolls sideways. The world is 21c's scorecard, step8804
scored on three benchmarks, and invented HLE and GPQA answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from service import reports
from service import frontier as sf
from test_21c_page import results
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_23e_training import GPQA, ROWS, gpqa_world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase23e"
HLE = "hle_text_cais"


def hle(label: str, k: int, n: int = 60) -> None:
    """`label` right on the first k of n invented HLE questions"""
    d = sf.task_dir(reports.row_dir(ROWS[label]), HLE)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{HLE}_2026-10-21T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": q, "doc": {"id": f"h{q}"}, "resps": [["Exact Answer: x"]],
                    "filtered_resps": ["x"],
                    "frontier": [{"epoch": 0, "ok": q < k, "ran_out": False}]}) + "\n"
        for q in range(n)))


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import report_stats
    scorecard()
    row = reports.row_dir(ROWS["step8804"])
    mid = ROWS["step8804"].removesuffix(" · thinking")
    for task, v in ((GPQA, 0.62), (HLE, 0.105), ("mmlupro_tiger", 0.80)):
        results(row, mid, task, v, 0.02, 1000, 0)
    # and step1000's HLE, which 21c's fixture hasn't
    r1 = reports.row_dir(ROWS["step1000"])
    results(r1, ROWS["step1000"].removesuffix(" · thinking"), HLE, 0.05, 0.015, 2158, 0)
    for lb, k in (("step1000", 3), ("step4000", 12), ("step8804", 13)):
        hle(lb, k)
    gpqa_world()
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(page, loc, path: Path) -> None:
    """the element as the page lays it out, the sticky header kept off it: a
    full-page capture clipped to it"""
    b = loc.evaluate("e => { const r = e.getBoundingClientRect(); return {x: r.left + scrollX, "
                     "y: r.top + scrollY, width: r.width, height: r.height}; }")
    page.screenshot(path=str(path), full_page=True, clip=b)


def open_report(page, live, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    sec = page.locator("[data-report-section='training']")
    sec.wait_for(timeout=30000)
    return sec


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_across_training_draws_and_a_segment_opens_its_pair(live, page, width, scheme):
    sec = open_report(page, live, width, scheme)
    order = page.eval_on_selector_all("[data-report-section]", "s => s.map(x => x.dataset"
                                      ".reportSection)")
    assert order[order.index("ours") + 1] == "training"          # right after Our runs
    assert page.locator("[data-tr-panel]").count() == 7
    # HLE: the three checkpoints, their whiskers, the line, Q4 and BF16 dashed
    assert page.locator("[data-tr-dot^='hle:']").count() == 3
    assert page.locator("[data-tr-whisker^='hle:']").count() == 3
    assert page.locator("[data-tr-ref^='hle:']").count() == 2
    # OTIS Mock AIME: step8804 not scored — its line shorter, and said
    assert page.locator("[data-tr-dot^='otis:']").count() == 2
    assert page.locator("[data-tr-missing='otis']").inner_text() == "step8804: not scored yet"
    # each segment's verdict: 1000 → 4000, 9 questions up (p = 0.004); 4000 → 8804, one
    seg = page.locator("[data-tr-seg='hle:step1000>step4000'] text")
    assert seg.text_content() == "+15.0, higher"
    assert page.locator("[data-tr-seg='hle:step4000>step8804'] text").text_content() == (
        "+1.7, can't tell")
    said = page.locator("[data-tr-sentence]").inner_text()
    assert said.startswith("From step4000 to step8804: ") and "can't tell on HLE" in said
    # near the ceiling where the report names MATH Level 5 (Q4 99.2)
    mark = page.locator("[data-rp-ours] th [data-ceil='math']")
    assert mark.inner_text() == "near the ceiling" and "handful of questions" in mark.get_attribute(
        "title")
    assert page.locator("[data-tr-panel='math'] [data-ceil='math']").count() == 1
    if width == 400:                                             # one column on a phone
        xs = page.eval_on_selector_all("[data-tr-panel]", "ps => ps.map(p => Math.round("
                                       "p.getBoundingClientRect().left))")
        assert len(set(xs)) == 1, xs
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, sec, SCREENS / f"training-{width}-{scheme}.png")
    # the segment's label opens the two-builds report on that pair
    page.locator("[data-tr-seg='hle:step1000>step4000']").click()
    page.wait_for_selector("[data-pd-pill='a'][data-value='step4000']", timeout=30000)
    assert page.locator("[data-pd-pill='b']").get_attribute("data-value") == "step1000"
    assert no_sideways(page) and page.errors == []


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_an_example_opens_with_its_marks(live, page, width, scheme):
    open_report(page, live, width, scheme)
    page.wait_for_selector("[data-th-table='gpqa']", timeout=30000)
    page.locator("[data-th-row='step4000'] [data-th-open='lost']").click()
    lst = page.locator("[data-th-list]")
    lst.wait_for()
    page.wait_for_selector("[data-th-list-head]", timeout=30000)
    head = page.locator("[data-th-list-head]").inner_text()
    n = page.locator("[data-th-item]").count()
    assert head == f"15 answers; {n} listed, {15 - n} in the hidden half." and 0 < n < 15
    assert lst.locator("[data-ro-warning]").inner_text().startswith("For insight only")
    assert "Invented question" not in lst.inner_text()           # GPQA: no question text
    lst.locator("[data-th-read]").first.click()
    ans = page.locator("[data-pd-answer]")
    ans.wait_for(timeout=30000)
    first = ans.locator("[data-pd-mark='first']")
    assert first.inner_text().startswith("the answer is (") and first.is_visible()
    assert ans.locator("[data-pd-mark='final']").inner_text().startswith("ANSWER: ")
    assert ans.locator("[data-pd-right]").inner_text().startswith("The right option: (")
    assert ans.locator("[data-pd-withheld]").count() == 1
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, lst, SCREENS / f"example-{width}-{scheme}.png")
