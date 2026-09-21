"""Phase 8c P5b: what the page does when it cannot reach the service, and
what it shows when it can.

The browser half of these drives the LIVE page with the API stubbed, because
the failure this fixes — 27 requests in a few seconds and "Loading results…"
forever — only exists in live mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import go_tab

import report_lm_eval as report

pytestmark = pytest.mark.dashboard

REPO = Path(__file__).resolve().parents[1]
SCREENS = Path(__file__).resolve().parent / "_screens"


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def live_page(payload: dict | None = None) -> str:
    """The page as the service serves it: an empty data slot, so the JS
    fetches. Same assembly as service/app.py."""
    return (report.TEMPLATE
            .replace("__TITLE__", "Test board")
            .replace("__BANNER__", "")
            .replace("__CSS__", report.CSS)
            .replace("__DATA__", "null")
            .replace("__JS__SLOT__", report.JS))


class Live:
    """A live page whose API can be made to fail, and that counts requests."""

    def __init__(self, ctx, payload: dict, fail: bool = True):
        self.calls: list[str] = []
        self.errors: list[str] = []
        self.fail = fail
        self.payload = payload
        self.page = ctx.new_page()
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.route("**/api/**", self._api)
        self.page.route("https://board.test/", lambda r: r.fulfill(
            status=200, content_type="text/html", body=live_page()))

    def _api(self, route):
        self.calls.append(route.request.url.split("/api/")[-1])
        if self.fail:
            route.fulfill(status=503, content_type="application/json",
                          body='{"detail":"service restarting"}')
        elif "results" in route.request.url:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(self.payload))
        else:
            route.fulfill(status=200, content_type="application/json", body="[]")

    def open(self):
        self.page.goto("https://board.test/")
        return self.page


@pytest.fixture
def live(browser, payload):
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    yield lambda **kw: Live(ctx, payload, **kw)
    ctx.close()


def test_a_blocked_api_backs_off_and_says_so(live):
    """Found on 2026-09-20: 'Loading results…' forever, and 27 requests to
    /api/submissions in a few seconds."""
    s = live(fail=True)
    pg = s.open()
    pg.wait_for_selector("[data-net='down']", timeout=15000)
    text = pg.locator("[data-net='down']").text_content()
    assert "Not reaching the service" in text
    assert "api/" in text and "HTTP 503" in text
    assert "Retrying" in text or "retrying" in text
    # the boot message is replaced by something that says what is wrong
    assert "Loading results…" not in pg.locator("#view").text_content()
    assert "Cannot reach" in pg.locator("#view").text_content()
    # and the retries are paced: a handful over several seconds, not dozens
    pg.wait_for_timeout(6000)
    assert len(s.calls) <= 8, s.calls
    first = len(s.calls)
    pg.wait_for_timeout(4000)
    assert len(s.calls) - first <= 2, s.calls         # backed off to seconds apart
    SCREENS.mkdir(exist_ok=True)
    pg.screenshot(path=SCREENS / "page-api-down.png", clip={"x": 0, "y": 0,
                                                            "width": 1240, "height": 320})
    assert s.errors == []


def test_recovery_clears_the_message(live):
    s = live(fail=True)
    pg = s.open()
    pg.wait_for_selector("[data-net='down']", timeout=15000)
    s.fail = False                                    # the service comes back
    pg.wait_for_selector("[data-net='down']", state="detached", timeout=40000)
    pg.wait_for_selector("#view .card", timeout=20000)
    assert "Cannot reach" not in pg.locator("#view").text_content()
    assert s.errors == []


def test_the_board_checks_are_one_line_on_every_tab(live):
    """Phase 9b-1: the checks filled the first screen of Overview and the
    Leaderboard. Now one line everywhere — "N checks · M about the judged
    suite" — and, opened, one line per check with a severity dot, a Show me
    to the rows it concerns, and the long text behind a disclosure."""
    s = live(fail=False)
    pg = s.open()
    pg.wait_for_selector("#view .card", timeout=20000)
    for label in ("Overview", "Leaderboard", "Exam", "Provenance"):
        go_tab(pg, label)
        pg.wait_for_selector("[data-warnings='collapsed']", timeout=20000)
        fold = pg.locator("[data-warnings='collapsed']")
        assert fold.count() == 1, label
        assert not fold.locator(".checklist").is_visible(), label       # one line, closed
    said = pg.locator("[data-warnings='collapsed'] > summary").text_content()
    n = pg.evaluate("DATA.checks.length")
    assert n == pg.evaluate("DATA.warnings.length") and f"{n} check" in said
    judged = pg.evaluate("DATA.checks.filter(c => c.judged).length")
    if judged:
        assert f"{judged} about the judged suite" in said
    # folded, never dismissed: they open, one line each
    pg.locator("[data-warnings='collapsed'] > summary").click()
    rows = pg.locator("[data-warnings='collapsed'] li[data-check]")
    assert rows.count() == n
    first = rows.first
    assert first.locator(".dot").count() == 1
    assert first.get_attribute("data-severity") in ("warning", "info")
    assert not first.locator(".check-more > p").is_visible()            # the long text waits
    first.locator(".check-more > summary").click()
    assert first.locator(".check-more > p").is_visible()
    # "Show me" goes to the rows: the preliminary check lands on Models, filtered
    prelim = pg.locator("li[data-check='preliminary'] [data-show-me]")
    if prelim.count():
        prelim.click()
        pg.wait_for_selector("input[data-filter='preliminary']:checked")
        assert pg.evaluate("location.hash") == "#tab=models"
    # and exactly once per page: the Provenance tab used to print the same
    # findings again under its own "Warnings" heading
    go_tab(pg, "Provenance")
    pg.wait_for_selector("#view .card")
    assert pg.locator("#view .warn").count() == 0
    assert pg.locator("#view h2", has_text="Warnings").count() == 0
    assert s.errors == []


def test_the_exam_tab_shows_both_ways_a_bank_arrives(live):
    s = live(fail=False)
    pg = s.open()
    pg.wait_for_selector("#view .card", timeout=20000)
    go_tab(pg, "Exam")
    pg.wait_for_selector("#view .card", timeout=20000)
    text = pg.locator("#view").text_content()
    # phase 9b-9: the product's words, not the shell's — the import panel is
    # how a person's bank arrives, and drafting is asked of whoever runs it
    assert pg.locator("[data-panel='import']").count() == 1
    assert "exam_build.py" not in text and "--root" not in text and "/app/" not in text
    # an empty bank is not a page bug, but the page is where someone finds out
    empty = pg.locator("[data-bank='empty']")
    assert empty.count() == 1
    assert "Import a person's bank below" in empty.text_content()
    assert "draft candidates" in empty.text_content()
    assert s.errors == []
