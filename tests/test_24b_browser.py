"""24b in a real browser, at 1280 and 400 px, light and dark: "Score against
file size" in the step-checkpoints report — a dot a build with a size, the
frontier starred and joined, log ticks at round sizes, a build with no size
named under it — and each build's quant and size under its name, in Our
runs, Across training's legend and the two-builds report's picker and table.
The world is 21c's scorecard, four of its builds given what their files say."""

from __future__ import annotations

from pathlib import Path

import pytest

from service import db
from test_21c_page import SPEC
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase24b"
S4, Q4, Q2, BF16 = "step4000", "Q4 (UD-Q4_K_XL)", "Q2 (UD-Q2_K_XL)", "BF16"
FILES = {S4: {"gguf_file": {"size": 15_800_000_000, "file_type": "Q4_K_M"}},
         Q4: {"gguf_file": {"size": 22_000_000_000, "file_type": "Q4_K_M"}},
         Q2: {"pin": {"size": 13_000_000_000}},
         BF16: {"gguf_file": {"size": 70_000_000_000, "file_type": "BF16"}}}


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    scorecard()
    for label, more in FILES.items():
        mid = SPEC[label]["row"].removesuffix(" · thinking")
        db.served_put({**(db.served_get(mid) or {}), **more})
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
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    page.wait_for_selector("[data-size-chart]", timeout=30000)


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_score_against_file_size(live, page, width, scheme):
    report(page, live, width, scheme)
    gpqa = page.locator("[data-sz-panel='gpqa']")
    stars = dict(gpqa.locator("[data-sz-dot]").evaluate_all(
        "ds => ds.map(d => [d.dataset.szDot.split(':')[1], d.dataset.szStar])"))
    # 13 GB at 81.8 and 22 GB at 85.1: no smaller build scores higher; 15.8 GB at 74.2 and
    # 70 GB at 84.8 each have one that does
    assert stars == {Q2: "true", Q4: "true", S4: "false", BF16: "false"}
    assert gpqa.locator("[data-sz-frontier='gpqa']").count() == 1
    assert gpqa.locator("[data-sz-label='gpqa:Q2 (UD-Q2_K_XL)']").text_content() == "★ Q2"
    ticks = gpqa.locator("[data-sz-tick]").evaluate_all("ts => ts.map(t => t.textContent)")
    assert ticks == ["10", "20", "50"]
    assert page.locator("[data-sz-star-note]").inner_text() == (
        "★ = no smaller build scores higher on this benchmark.")
    assert page.locator("[data-sz-missing]").inner_text() == "step1000: size not known"
    if width == 400:                                              # one column on a phone
        xs = page.eval_on_selector_all("[data-sz-panel]", "ps => ps.map(p => Math.round("
                                       "p.getBoundingClientRect().left))")
        assert len(set(xs)) == 1, xs
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, page.locator("[data-size-chart]"), SCREENS / f"size-chart-{width}-{scheme}.png")


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_each_builds_quant_and_size_under_its_name(live, page, width, scheme):
    report(page, live, width, scheme)
    meta = dict(page.eval_on_selector_all(
        "[data-rp-ours] [data-rp-row]", "rs => rs.map(r => [r.dataset.rpRow, "
        "(r.querySelector('[data-rp-meta]') || {}).textContent || ''])"))
    assert meta[S4] == "Q4_K_M · 15.8 GB" and meta[Q4] == "Q4_K_M · 22.0 GB"
    assert meta[Q2] == "13.0 GB" and meta["step1000"] == ""      # only what is known
    assert meta[BF16] == "BF16 · 70.0 GB"
    # Across training's legend
    page.wait_for_selector("[data-tr-legend]")
    assert page.locator(f"[data-tr-key='{S4}'] [data-rp-meta]").inner_text() == "Q4_K_M · 15.8 GB"
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, page.locator("[data-report-section='ours']"), SCREENS / f"labels-{width}-{scheme}.png")
    # the two-builds report: its picker's menu and its table's heads
    page.goto(live["base"] + f"/#tab=reports&report=paired&a={S4}&b={Q4}")
    page.wait_for_selector("[data-pd-table]", timeout=30000)
    assert page.locator("[data-pd-col='a'] [data-rp-meta]").inner_text() == "Q4_K_M · 15.8 GB"
    page.locator("[data-pd-pill='b']").click()
    item = page.locator(f"#pop-pd-b [data-pd-choice='{Q2}'] [data-rp-meta]")
    assert item.inner_text() == "13.0 GB"
    assert no_sideways(page) and page.errors == []
