"""21c: the scorecard's page as it is, in a real browser — at 1280 and 400 px,
light and dark: no console error, no request to another host, the picker
changes the tiles, the head-to-head table's header stays in view while it
scrolls, nothing scrolls sideways; its numbers are 21's view's, cell for
cell; the Reports card and 21's view open it in a tab of its own. The world
is tests/test_21c_page.py's: the 14 Oct scorecard as the board would hold it,
invented questions for the loop examples. No model runs; nothing is fetched."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from test_21c_page import world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase21c"


def until(page, expr: str, timeout: float = 30) -> None:
    """wait until `expr` holds on the page — polled from here: the page's
    Content-Security-Policy refuses the eval Playwright's own waiting uses"""
    end = time.time() + timeout
    while not page.evaluate(expr):
        assert time.time() < end, f"never true: {expr}"
        time.sleep(0.1)


def no_sideways(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def open_page(page, live, width: int, scheme: str) -> list[str]:
    """the page, its data in; every URL it asked for"""
    asked: list[str] = []
    page.on("request", lambda r: asked.append(r.url))
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/reports/step-checkpoints/page")
    page.wait_for_selector("#ours-body tr")
    page.wait_for_selector("#stand-tiles .sc-tile .sc-strip svg")
    page.wait_for_selector("#tok-body tr", timeout=30000)
    page.wait_for_selector("#loops-list .sc-loop", timeout=30000)
    page.wait_for_selector("#compare-body tr")
    until(page, "document.querySelectorAll('[data-dash-source]').length === 4")
    # the stats worker done with what the world changed: the header line says so
    until(page, "!document.querySelector('[data-dash-fresh]').textContent"
                ".includes('being worked out')")
    return asked


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_page_draws_from_our_server_alone(live, page, width, scheme):
    world()
    asked = open_page(page, live, width, scheme)
    here = urlsplit(live["base"]).netloc
    assert asked and all(urlsplit(u).netloc == here for u in asked), \
        [u for u in asked if urlsplit(u).netloc != here]
    # the theme: the viewer's setting
    bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == ("rgb(247, 247, 245)" if scheme == "light" else "rgb(13, 13, 13)"), bg
    # our runs, each with its status
    rows = page.locator("#ours-body tr")
    assert rows.count() == 7                          # 22: and Ornith-1.5's, not run yet
    assert rows.nth(0).locator("td").nth(1).inner_text() == "Complete · 21 of 21 chunks"
    assert rows.nth(1).locator("td").nth(1).inner_text() == "Running · 20 of 21 chunks in"
    # the takeaway, worked out by the page's own calcs from our data
    assert page.locator("#takeaway .dash-skeleton").count() == 0
    # the picker changes the line and the tiles
    gpqa = page.locator("#stand-gpqa .sc-tile-v").inner_text()
    line = page.locator("#stand-line").inner_text()
    assert gpqa.startswith("74.2") and line.startswith("step4000 is higher than ")
    page.locator("#stand-picks .sc-pick", has_text="BF16").click()
    until(page, "document.querySelector('#stand-line').textContent.startsWith('BF16')")
    assert page.locator("#stand-gpqa .sc-tile-v").inner_text().startswith("84.8")
    assert page.locator("#stand-picks .sc-pick[aria-pressed=true]").inner_text() == "BF16"
    # the head-to-head table: its header stays in view while it scrolls
    box = page.locator("#stands .sc-tall")
    box.scroll_into_view_if_needed()
    box.evaluate("e => e.scrollTop = 500")
    page.wait_for_timeout(100)
    assert box.evaluate("e => e.scrollTop") > 0
    top = box.evaluate("e => e.getBoundingClientRect().top")
    th = page.locator("#stand-table thead th").first.evaluate(
        "e => e.getBoundingClientRect().top")
    ref = page.locator("#stand-ref td").first.evaluate("e => e.getBoundingClientRect().top")
    assert abs(th - top) <= 2 and 0 < ref - top <= 60, (top, th, ref)
    # the header line: when each source last changed
    fresh = page.locator("[data-dash-fresh]").inner_text()
    assert "Our scores: as of the last import, " in fresh and "Epoch AI: imported " in fresh
    assert no_sideways(page)
    assert page.errors == []
    if width == 1280 or scheme == "light":
        SCREENS.mkdir(parents=True, exist_ok=True)
        page.locator("#stand-picks .sc-pick", has_text="step4000").click()
        until(page, "document.querySelector('#stand-line').textContent.startsWith('step4000')")
        box.evaluate("e => e.scrollTop = 0")
        h = page.evaluate("document.documentElement.scrollHeight")
        page.set_viewport_size({"width": width, "height": h})
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        page.screenshot(path=SCREENS / f"page-{width}-{scheme}.png", full_page=True)


def cells(page, sel: str) -> list[list[str]]:
    return page.eval_on_selector_all(sel, "rows => rows.map(r => [...r.children].map("
                                     "c => c.textContent.trim()))")


def test_the_numbers_are_21s_cell_for_cell(live, page):
    world()
    open_page(page, live, 1280, "light")
    mine = {r[0]: r[2:] for r in cells(page, "#ours-body tr")}
    h2h = {}
    for pick in ("step4000", "BF16"):
        page.locator("#stand-picks .sc-pick", has_text=pick).click()
        until(page, f"document.querySelector('#stand-line').textContent.startsWith({pick!r})")
        h2h[pick] = page.eval_on_selector_all(
            "#stand-body tr", "rows => rows.map(r => [...r.children].map(c => [c.textContent"
            ".trim(), c.className]))")
    compare = {r[0]: r[2:8] for r in cells(page, "#compare-body tr")}
    tokens = {(r[1], r[0]): r[2:9] for r in cells(page, "#tok-body tr")}
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    page.wait_for_selector("[data-rp-ours] [data-rp-row='step4000']")
    page.wait_for_selector("[data-rp-tok]", timeout=30000)
    benches = ["gpqa", "hle", "otis", "math", "mmlupro", "simpleqa", "arc"]
    # our runs: 21 shows "74.2 ± 2.6"; the page "74.2"
    theirs = page.eval_on_selector_all(
        "[data-rp-ours] tbody tr",
        "rows => rows.map(r => [r.dataset.rpRow, ...[...r.children].slice(1).map("
        "c => c.firstChild && c.firstChild.nodeType === 3 ? c.firstChild.textContent.trim() "
        ": c.textContent.trim())])")
    assert len(theirs) == 7                           # 22: and Ornith-1.5's
    for row in theirs:
        name, vals = row[0], row[1:]
        got = [v.split(" ± ")[0] for v in vals]
        assert got == mine[name], (name, got, mine[name])
    # where it stands: each head-to-head cell, its gap and its verdict
    state = {"sc-above": "st-higher", "sc-below": "st-lower", "sc-level": "st-level"}
    for pick, rows in h2h.items():
        page.locator(f"[data-rp-pick='{pick}']").click()
        page.wait_for_function(f"document.querySelector('[data-rp-summary]').textContent"
                               f".startsWith({pick!r})")
        ref = {r[0]: r[1:] for r in page.eval_on_selector_all(
            "[data-rp-h2h-row]", "rows => rows.map(r => [r.dataset.rpH2hRow, ...[...r.children]"
            ".slice(1).map(c => [c.textContent.trim(), c.className])])")}
        # the page sets ours against the lda runtime's runs and the popular
        # models; 21 against "Q4, earlier run" too. 22: both against
        # Ornith-1.5's card; 21 against Ornith-1.5's own row too, one of ours
        # off the lda runtime as "Q4, earlier run" is
        assert len(ref) == 48 and len(rows) == 46 and not {"Q4, earlier run",
                                                           "Ornith-1.5 Q4_K_M"} & {
            r[0][0] for r in rows}, (pick, len(ref), len(rows))
        for r in rows:
            name, got = r[0][0], dict(zip(["gpqa", "hle", "otis", "math", "simpleqa", "mmlupro"],
                                          r[1:7]))
            want = dict(zip(["gpqa", "hle", "otis", "math", "mmlupro", "simpleqa"], ref[name]))
            for k in got:
                g, w = got[k], want[k]
                assert g[0] == w[0], (pick, name, k, g, w)
                assert state.get(g[1].split()[-1], "dash") == next(
                    (c for c in w[1].split() if c.startswith("st-")), "dash"), (pick, name, k, g, w)
            assert [r[7][0], r[8][0]] == [ref[name][6][0], ref[name][7][0]], (pick, name)
    # every other model: the page's compare table, 21's "every model" table
    every = page.eval_on_selector_all(
        "[data-rp-every] tbody tr",
        "rows => rows.map(r => [r.children[0].title, ...[...r.children].slice(1).map("
        "c => c.firstChild && c.firstChild.nodeType === 3 ? c.firstChild.textContent.trim() "
        ": c.textContent.trim())])")
    order = ["gpqa", "hle", "otis", "math", "simpleqa", "mmlupro"]
    checked = 0
    for row in every:
        name = row[0]
        if name not in compare:
            continue
        by = dict(zip(benches, row[1:]))
        want = [by[k].split(" ± ")[0].rstrip("†") for k in order]
        assert want == [x.rstrip("†") for x in compare[name]], (name, want, compare[name])
        checked += 1
    assert checked == len(compare) == 47                    # 22: and Ornith-1.5's card
    # the thinking tokens: answers, thinking mean and median, p90, right and
    # wrong means, ran out
    for row in page.eval_on_selector_all(
            "[data-rp-tok]", "rows => rows.map(r => [r.dataset.rpTok, ...[...r.children]"
            ".map(c => c.textContent.trim())])"):
        label, task = row[0].split("|")
        bench = {"gpqa_diamond_epoch": "GPQA Diamond", "hle_text_cais": "HLE",
                 "otis_aime_epoch": "OTIS Mock AIME", "math_l5_epoch": "MATH Level 5",
                 "mmlupro_tiger": "MMLU-Pro", "simpleqa_epoch": "SimpleQA Verified",
                 "arc_agi2_public": "ARC-AGI-2"}[task]
        p = tokens[(label, bench)]
        assert [row[2], row[3], row[4], row[5], row[6].split(" (")[0], row[7].split(" (")[0],
                row[8]] == [p[0], p[1], p[2], p[3], p[4], p[5], p[6]], (label, bench, row, p)
    assert page.errors == []


def test_the_reports_card_and_21s_view_open_it_in_a_tab_of_its_own(live, page):
    world()
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(live["base"] + "/#tab=reports")
    link = page.locator("[data-report-page='step-checkpoints']")
    link.wait_for()
    assert link.get_attribute("target") == "_blank"
    assert page.locator("[data-report-page='ran-out-of-room']").count() == 0
    with page.context.expect_page() as tab:
        link.click()
    new = tab.value
    new.wait_for_selector("#ours-body tr")
    assert new.url.endswith("/reports/step-checkpoints/page")
    new.close()
    page.locator("[data-report-card='step-checkpoints']").click()
    page.wait_for_selector("[data-report='step-checkpoints'] [data-report-page]")
    assert page.locator("[data-report='step-checkpoints'] [data-report-page]").get_attribute(
        "href") == "reports/step-checkpoints/page"
    # the board's theme setting is the page's too
    page.evaluate("localStorage.setItem('bench-theme', 'dark')")
    page.goto(live["base"] + "/reports/step-checkpoints/page")
    page.wait_for_selector("#ours-body tr")
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(13, 13, 13)"
    page.evaluate("localStorage.removeItem('bench-theme')")
    assert page.errors == []


def test_the_speed_script_times_the_board(live, capsys):
    """scripts/report_speed.py, as masein runs it on the server: each report
    path answered, and how long it took"""
    import report_speed
    world()
    assert report_speed.main(["--base", live["base"]]) == 0
    out = capsys.readouterr().out
    for path in report_speed.PATHS:
        line = next(x for x in out.splitlines() if x.endswith(" /" + path))
        assert "HTTP 200" in line or (path.endswith(("tokens", "loops")) and "HTTP 202" in line), \
            line
