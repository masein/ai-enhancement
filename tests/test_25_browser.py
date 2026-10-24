"""25 in a real browser, at 1280 and 400 px, light and dark: the
step-checkpoints report's "Instruction following" — a row a build, IFEval
strict and loose and IFBench loose, each thinking on beside thinking off,
with ±, under the sentence from the paired verdicts — after "Scores by
benchmark"; the two kept out of "Where we stand" and "Every model"; and the
two-builds reader marking each instruction ✓ or ✗ under each reply. The
world is 21c's scorecard with test_25_board's answers scored on top."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import frontier as fb
from service import config
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_25_board import IFB, IFE, PLAN, Q4, S40, S88, build, items, reply

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase25"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import paired, report_stats
    scorecard()
    for task in (IFE, IFB):
        p = fb._cache(config.BENCH_ROOT, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"items": items(task), "extra": {}}))
    for key in PLAN:
        build(*key)
    paired._tables.clear()
    report_stats.work(say=lambda s: None)
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


def report(page, live, width, scheme, hash_="report=step-checkpoints", wait="[data-fl-table]"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&" + hash_)
    page.wait_for_selector(wait, state="attached", timeout=30000)


def heads(page, sel: str) -> list[str]:
    # 27b: a modelTable column by its whole name (its title), as the old head read
    return page.eval_on_selector_all(f"{sel} thead th", "ts => ts.map(t => t.dataset.mtCol "
                                     "&& t.title || t.textContent.trim())")


@pytest.mark.parametrize("width,scheme", SIZES)
def test_instruction_following_on_and_off_side_by_side(live, page, width, scheme):
    # 27e: against Q4, as before — Q2, the default baseline, hasn't run them here
    report(page, live, width, scheme, hash_="report=step-checkpoints&base=Q4%20(UD-Q4_K_XL)")
    order = page.eval_on_selector_all("[data-report-section]",
                                      "ss => ss.map(s => s.dataset.reportSection)")
    assert order.index("scores") + 1 == order.index("follow") < order.index("beyond")
    assert page.locator("[data-fl-sentence]").inner_text() == (
        "step8804 against Q4 (UD-Q4_K_XL) on IFEval: lower, by 50.0 points; with thinking "
        "off, can't tell. step8804 against Q4 (UD-Q4_K_XL) on IFBench: can't tell; with "
        "thinking off, lower, by 50.0 points.")
    assert heads(page, "[data-fl-table]") == [
        "Model", "IFEval strict", "IFEval loose", "IFBench loose",
        "thinking on", "thinking off", "thinking on", "thinking off", "thinking on",
        "thinking off"]

    def cell(label, key, w):
        return page.locator(f"[data-fl-cell='{label}|{key}|{w}']").inner_text()
    assert cell(S88, "ifeval_google:strict", "on") == "30.0 ± 10.5"
    assert cell(S88, "ifeval_google:strict", "off") == "60.0 ± 11.2"
    assert cell(S88, "ifeval_google:loose", "on") == "40.0 ± 11.2"
    assert cell(S88, "ifbench_test:loose", "on") == "50.0 ± 11.5"
    assert cell(Q4, "ifbench_test:loose", "off") == "60.0 ± 11.2"
    assert cell(S40, "ifeval_google:strict", "off") == "—"          # no thinking-off answers
    assert cell("BF16", "ifbench_test:loose", "on") == "—"           # not measured
    # on its left the thinking row, on its right the plain one
    assert page.locator(f"[data-fl-cell='{S88}|ifeval_google:strict|off']").get_attribute(
        "title") == "served/step8804-lda-2610"
    assert "DeviceMark" in page.locator("[data-fl-legend]").inner_text()
    # in Our runs; 25b: and, compared with Google's Gemma 4 numbers, in Where we
    # stand and Every model (tests/test_25b_browser.py)
    assert {"IFEval", "IFBench"} <= set(heads(page, "[data-rp-ours]"))
    assert {"IFEval", "IFBench"} <= set(heads(page, "[data-rp-every]"))
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, page.locator("[data-report-section='follow']"),
         SCREENS / f"follow-{width}-{scheme}.png")


@pytest.mark.parametrize("width,scheme", SIZES)
def test_the_reader_marks_each_instruction_under_each_reply(live, page, width, scheme):
    from service import reports
    report(page, live, width, scheme, f"report=paired&a={S88}&b={Q4}", "[data-pd-table]")
    rows = page.eval_on_selector_all("[data-pd-row]", "rs => rs.map(r => r.dataset.pdRow)")
    assert rows[-2:] == ["ifeval", "ifbench"]
    assert page.locator("[data-pd-verdict='ifeval']").inner_text() == f"{Q4} higher"
    page.locator("[data-pd-part='ifeval:b']").click()
    page.wait_for_selector("[data-pd-follow]")
    # the first prompt of the shown half Q4 followed and step8804 didn't
    first = next(it for it in items(IFE)[6:16] if reports.shown_half(IFE, it["id"]))
    k = int(first["id"]) - 1000
    assert page.locator("[data-pd-card]").get_attribute("data-pd-card") == first["id"]
    assert page.locator("[data-pd-question]").inner_text() == f"Invented prompt {k}."
    assert page.locator("[data-pd-right]").count() == 0             # no right answer: its checks
    iid = first["instructions"][0]
    for who, ok in ((S88, False), (Q4, True)):
        col = page.locator(f"[data-pd-col-of='{who}']")
        item = col.locator(f"[data-pd-inst='{iid}']")
        assert item.get_attribute("data-pd-inst-ok") == str(ok).lower()
        assert item.inner_text().startswith(("✓ " if ok else "✗ ") + iid)
        assert col.locator("[data-pd-final]").inner_text() == (
            f"{int(ok)} of 1 followed (strict)")
        want = PLAN[(who, IFE, "on")][k]
        assert col.locator("[data-pd-reply]").inner_text() == reply(IFE, k, want)
    # step8804's reply followed it once its lead-in line is set aside: loose says so
    if PLAN[(S88, IFE, "on")][k] == "l":
        assert page.locator(f"[data-pd-col-of='{S88}'] [data-pd-inst='{iid}']").inner_text() \
            .endswith("· loose ✓")
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, page.locator("[data-pd-card]"), SCREENS / f"reader-{width}-{scheme}.png")


def test_the_model_page_says_the_four_numbers(live, page):
    """the build's page: each benchmark under instruction following, its
    authors' measure named, the other three under it"""
    page.set_viewport_size({"width": 1280, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + "served/step8804-lda-2610%20%C2%B7%20thinking")
    page.wait_for_selector("[data-result-row='ifeval_google']", timeout=30000)
    assert page.locator("[data-result-name='ifeval_google']").inner_text() == (
        "IFEval (the reply alone) · prompt-level strict")
    assert page.locator("[data-result-name='ifbench_test']").inner_text() == (
        "IFBench · prompt-level loose")
    assert page.locator("[data-result-follow='ifeval_google']").inner_text() == (
        "↑ prompt-level loose 40.0 · instruction-level strict 30.0, loose 40.0 — over 20 "
        "prompts and 20 instructions")
    assert page.locator("[data-result-follow='ifbench_test']").inner_text().startswith(
        "↑ prompt-level strict 25.0 · ")
    group = page.locator("[data-result-row='ifeval_google']").evaluate(
        "r => { let p = r.previousElementSibling; while (p && !p.classList.contains('domrow')) "
        "p = p.previousElementSibling; return p ? p.textContent : ''; }")
    assert group == "instruction following"
    assert page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    box = page.locator("[data-result-row='ifeval_google']").evaluate(
        "r => { const c = r.closest('.card').getBoundingClientRect(); return {x: c.left + "
        "scrollX, y: c.top + scrollY, width: c.width, height: c.height}; }")
    page.screenshot(path=str(SCREENS / "model-page-1280-light.png"), full_page=True, clip=box)
