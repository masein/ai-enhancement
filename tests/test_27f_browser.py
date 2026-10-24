"""27f in a real browser, at 1280 and 400 px, light and dark: NVIDIA's
Nemotron-3.5-Bolt-E4B on the On-device report — its row's line with the
card's data table on hover, its IFBench as NVIDIA publishes it with N, the
three it doesn't give as "–", what it keeps of Gemma 4 E4B-it, its 4-bit
file's size, a family colour of its own on the charts, and "Also
published" — and on Models, its own page and Frontier row. The world is
25c's: 21c's scorecard, 24b's file sizes, every card entered by masein."""

from __future__ import annotations

from pathlib import Path

import pytest

from service import db
from test_21c_page import SPEC
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_24b_browser import FILES
from test_27f_nemotron import CARD

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27f"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]
NV = "NVIDIA-Nemotron-3.5-Bolt-E4B"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import enter_report_cards as erc
    import service.app as appmod
    scorecard()
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
    page.wait_for_selector("[data-report-section='od-keep']", timeout=30000)


@pytest.mark.parametrize("width,scheme", SIZES)
def test_nemotron_on_the_on_device_report(live, page, width, scheme):
    report(page, live, width, scheme)
    if width > 700:
        row = page.locator(f"tr[data-od-row='{NV}']")
        assert row.get_attribute("data-mt-group") == "open"
        what = row.locator(f"[data-od-what='{NV}']")
        assert what.inner_text() == "trained further from Gemma 4 E4B, not compressed"
        assert what.get_attribute("title") == ("its training seed data includes IFEval prompts "
                                               "(the model card's data table)")
        assert "published by NVIDIA" in row.locator("[data-mt-line]").inner_text()
        ifb = page.locator(f"[data-od-cell='{NV}|ifbench']")
        assert ifb.locator(".mt-n").inner_text() == "63.8N"
        assert ifb.locator("[data-card-link]").get_attribute("href") == CARD
        assert page.locator(f"[data-od-keep='{NV}|ifbench']").inner_text() == (
            "148.9% of its base")
        for k in ("gpqa", "mmlupro", "ifeval"):
            assert page.locator(f"[data-od-cell='{NV}|{k}']").inner_text().strip() == "–", k
        assert page.locator(f"[data-od-fact='{NV}|memory'] .mt-n").inner_text().startswith(
            "6.18 GB")
    # What each keeps: IFBench alone, of Gemma 4 E4B-it as NVIDIA ran it
    dot = page.locator(f"[data-od-keep-dot='ifbench:{NV}']")
    assert dot.count() == 1 and dot.get_attribute("stroke") == "var(--fam-nvidia)"
    assert page.locator(f"[data-od-keep-dot='gpqa:{NV}']").count() == 0
    assert page.locator(f"[data-od-base='ifbench:{NV}'] title").text_content() == (
        "of Gemma 4 E4B-it (NVIDIA's run)")
    # the memory chart: its own family, once in the legend
    assert page.locator(f"[data-od-mem-dot='ifbench:{NV}']").get_attribute("stroke") == (
        "var(--fam-nvidia)")
    fam = page.locator("[data-od-leg-family='nvidia']")
    assert fam.count() == 1 and "Nemotron Bolt-E4B" in fam.inner_text()
    colours = page.evaluate("""() => { const s = getComputedStyle(document.querySelector('.viz-root'));
        return ['edge0', 'bonsai', 'gemma', 'nvidia'].map(f => s.getPropertyValue('--fam-' + f)
          .trim()); }""")
    assert len(set(colours)) == 4 and all(colours), colours
    # Also published, each with its value as NVIDIA's card gives it
    also = page.locator(f"[data-od-also-of='{NV}']").inner_text()
    assert "BFCL-v3 68.51" in also and "Tau2-Bench 55.10" in also
    # the axis labels never touch, the IFBench panel's now to 150%
    gaps = page.evaluate("""() => [...document.querySelectorAll('[data-od-keep-panel]')]
        .map(p => { const r = [...p.querySelectorAll('[data-od-keep-tick]')]
          .map(t => t.getBoundingClientRect()).sort((a, b) => a.left - b.left);
          return Math.min(...r.slice(1).map((x, i) => x.left - r[i].right)); })""")
    assert min(gaps) > 0, gaps
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    for sec in ("od-table", "od-keep", "od-memory"):
        shot(page, page.locator(f"[data-report-section='{sec}']"),
             SCREENS / f"{sec}-{width}-{scheme}.png")


def test_nemotron_has_its_own_page_and_frontier_row(live, page):
    report(page, live, 1280, "light")
    page.locator(f"tr[data-od-row='{NV}'] a[data-mt-name]").click()
    page.wait_for_selector("[data-reported-model]", timeout=30000)
    assert page.locator("h1.mtitle").inner_text() == NV
    assert page.locator("[data-rm-fact='base']").inner_text().startswith("Gemma 4 E4B-it")
    pub = page.locator("[data-rm-published='IFBench']")
    assert pub.inner_text() == "published by NVIDIA" and pub.get_attribute("href") == CARD
    assert "BFCL-v3: 68.51" in page.locator("[data-rm-also]").inner_text()
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / "nemotron-page-1280-light.png"), full_page=True)
    # its one number is IFBench's: shown here as on the board, where IFBench is measured
    page.goto(live["base"] + "/#tab=models&view=frontier&cols=gpqa,ifbench")
    page.wait_for_selector("[data-frontier-table]", timeout=30000)
    row = page.locator("[data-fr-row^='reported/'][data-fr-row*='nemotron']")
    assert row.count() == 1
    cell = row.locator("[data-fr-published]").first
    assert cell.get_attribute("data-fr-published") == "NVIDIA"
    assert cell.locator("[data-mt-src]").inner_text() == "N"
    assert row.locator("[data-fr-ours]").count() == 0 and page.errors == []
