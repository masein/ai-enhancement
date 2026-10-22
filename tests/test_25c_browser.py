"""25c in a real browser, at 1280 and 400 px, light and dark: the On-device
models report — every row's cells saying where they come from, what each
keeps of its own base, score against the memory each needs, a measured dot
a filled circle and a published one a hollow diamond — and a model known
only as published, its own page and its Frontier row. The world is 21c's
scorecard, four of our builds given their files' sizes (24b's), every card
entered by masein."""

from __future__ import annotations

from pathlib import Path

import pytest

from service import db
from test_21c_page import SPEC
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_24b_browser import FILES
from test_25c_ondevice import E0

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase25c"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import enter_report_cards as erc
    import service.app as appmod
    scorecard()                                          # enters step-checkpoints' cards
    for label, more in FILES.items():
        mid = SPEC[label]["row"].removesuffix(" · thinking")
        db.served_put({**(db.served_get(mid) or {}), **more})
    assert erc.main(["on-device", "--by", "masein"]) == 0
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(page, loc, path: Path) -> None:
    for _ in range(20):
        page.wait_for_timeout(100)
        b = loc.evaluate("e => { const r = e.getBoundingClientRect(); return {x: r.left + "
                         "scrollX, y: r.top + scrollY, width: r.width, height: r.height}; }")
        if b["width"] > 0 and b["height"] > 0:
            break
    page.screenshot(path=str(path), full_page=True, clip=b)


def report(page, live, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=on-device")
    page.wait_for_selector("[data-od-table]", timeout=30000)


@pytest.mark.parametrize("width,scheme", SIZES)
def test_the_on_device_report(live, page, width, scheme):
    report(page, live, width, scheme)
    order = page.eval_on_selector_all("[data-report-section]",
                                      "ss => ss.map(s => s.dataset.reportSection)")
    assert order == ["od-table", "od-keep", "od-memory", "notes"]
    heads = page.eval_on_selector_all("[data-od-table] thead th", "ts => ts.map(t => t.textContent)")
    assert heads == ["Model", "Base model", "Size on disk", "Memory it needs", "Phone speed",
                     "Measured on our testers’ phones", "GPQA Diamond", "MMLU-Pro", "IFEval",
                     "IFBench"]
    rows = page.eval_on_selector_all("[data-od-row]", "rs => rs.map(r => r.dataset.odRow)")
    assert rows[:4] == ["step8804", "step4000", "step1000", "Q4 (UD-Q4_K_XL)"]
    assert rows[4:] == ["Edge0-35B-A3B-preview", "Edge0-8B-A1B-preview", "Bonsai 27B (1-bit)",
                        "Ternary Bonsai 2 2B", "Ternary Bonsai 2 4B", "Ternary Bonsai 2 9B",
                        "Gemma 4 E4B", "Gemma 4 E2B"]
    # every cell says where it is from: measured here, or published by … with its link
    e0 = page.locator("[data-od-cell='Edge0-35B-A3B-preview|gpqa']")
    assert e0.inner_text().split("\n")[0] == "79.8"
    assert e0.locator("[data-od-link]").get_attribute("href") == E0
    assert e0.locator("[data-od-link]").inner_text() == "published by Edge0"
    assert e0.locator("[data-od-keep]").inner_text() == "keeps 97.6% of its base"
    assert page.locator("[data-od-keep='Ternary Bonsai 2 9B|ifbench']").inner_text() == (
        "keeps 99.4% of its base")
    s4 = page.locator("[data-od-cell='step4000|gpqa']")
    assert s4.locator("[data-od-src]").inner_text() == "measured here"
    assert page.locator("[data-od-fact='Edge0-35B-A3B-preview|phone']").inner_text().startswith(
        "no phone numbers yet")
    assert page.locator("[data-od-fact='Bonsai 27B (1-bit)|phone']").inner_text().startswith(
        "about 11 tok/s decode")
    assert "iPhone 17 Pro Max" in page.locator("[data-od-fact='Bonsai 27B (1-bit)|phone']").inner_text()
    assert page.locator("[data-od-fact='Edge0-35B-A3B-preview|disk']").inner_text().startswith(
        "19.6 GB")
    assert page.locator("[data-od-fact='step4000|disk']").inner_text().startswith("15.8 GB")
    # the charts: ours a filled circle, theirs a hollow diamond — shape and fill
    for chart in ("keep", "mem"):
        marks = page.eval_on_selector_all(
            f"[data-od-{chart}-dot]", "ds => ds.map(d => [d.dataset.od" + chart.capitalize()
            + "Dot.split(':')[1], d.tagName, d.getAttribute('fill')])")
        assert marks, chart
        ours = {"step8804", "step4000", "step1000", "Q4 (UD-Q4_K_XL)"}
        for label, tag, fill in marks:
            if label in ours:
                assert tag == "circle" and fill.startswith("var(--s"), (chart, label, tag, fill)
            else:
                assert tag == "path" and fill == "var(--surface-1)", (chart, label, tag, fill)
    assert page.locator("[data-od-keep-dot='gpqa:Edge0-35B-A3B-preview']").count() == 1
    assert page.locator("[data-od-base='gpqa:Edge0-35B-A3B-preview']").evaluate(
        "e => e.firstChild.textContent") == (
        "of Qwen3.6-35B-A3B fp16")
    notes = page.locator("[data-report-section='notes']").inner_text()
    assert "Edge0-35B is built from the same base as our builds" in notes
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    for sec in ("od-table", "od-keep", "od-memory"):
        shot(page, page.locator(f"[data-report-section='{sec}']"),
             SCREENS / f"{sec}-{width}-{scheme}.png")


def test_a_published_model_has_its_own_page_and_frontier_row(live, page):
    report(page, live, 1280, "light")
    page.locator("[data-od-page='Edge0-35B-A3B-preview']").click()
    page.wait_for_selector("[data-reported-model]", timeout=30000)
    assert page.locator("h1.mtitle").inner_text() == "Edge0-35B-A3B-preview"
    assert "never in an average or a rank" in page.locator("[data-rm-said]").inner_text()
    assert page.locator("[data-rm-fact='base']").inner_text().startswith("Qwen3.6-35B-A3B")
    assert page.locator("[data-rm-fact='disk']").inner_text().startswith("19.6 GB")
    assert "Mac mini M4 Pro" in page.locator("[data-rm-fact='speed']").inner_text()
    pub = page.locator("[data-rm-published='GPQA-Diamond']")
    assert pub.inner_text() == "published by Edge0" and pub.get_attribute("href") == E0
    assert "AIME 2026: 86.6 (its base 92.7)" in page.locator("[data-rm-also]").inner_text()
    assert page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / "edge0-page-1280-light.png"), full_page=True)
    # on Models: a row of its own in Frontier, its number published by Edge0 — no rank
    page.goto(live["base"] + "/#tab=models&chip=frontier")
    page.wait_for_selector("[data-frontier-table]", timeout=30000)
    row = page.locator("[data-fr-row^='reported/'][data-fr-row*='edge0-35b']")
    assert row.count() == 1
    # 27: its number in italics, Edge0's letter linking its page; the row opens its own page
    cell = row.locator("[data-fr-published]").first
    assert cell.get_attribute("data-fr-published") == "Edge0"
    assert cell.locator("[data-mt-src]").inner_text() == "X"
    assert row.get_attribute("data-fr-page").startswith("reported/")
    assert row.locator("a[data-mt-name]").get_attribute("href").startswith("#model=reported%2F")
    assert row.locator("[data-fr-ours]").count() == 0 and page.errors == []
