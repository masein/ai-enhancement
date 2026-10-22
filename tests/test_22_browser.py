"""22 in a real browser, at 1280 and 400 px, light and dark: Ornith-1.5 on
Models ▸ Latest — our row, measured here, saying it isn't the team's model,
in the next free colour, and its card's claim beside it, linked to the card at
its commit; in the report's runs and its picker; in the ran-out report's
picker. The world is 21c's (the 14 Oct scorecard as the board would hold it,
invented questions), with Ornith-1.5 not run yet; nothing is fetched, run or
graded."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21c_page import world
from test_21d_browser import LATEST, go, no_sideways

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase22"
OURS, CARD = "Ornith-1.5 Q4_K_M", "Ornith-1.5-35B-A3B, its card"
COMMIT = "12393612fd4f730ff5aadc23e9b8f9648aa49ceb"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    world()
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light")])
def test_latest_shows_ornith_measured_here_and_its_cards_claim_beside_it(live, page, width,
                                                                         scheme):
    go(page, live, width=width, scheme=scheme)
    # 27: ours, not run yet, has no score shown: hidden and counted, a click away
    page.locator("[data-latest] [data-mt-empty-toggle]").click()
    page.wait_for_selector(f"[data-latest-row='{OURS}']", state="attached")
    rows = {x: r for r in page.evaluate(LATEST) for x in r["labels"]}
    assert OURS in rows and CARD in rows
    # ours: a dash in every cell, and what it is in its Source
    assert list(rows[OURS]["cells"].values()) == ["–", "–", "–", "–"]
    assert rows[OURS]["src"] == ("Measured here, on its card's settings; not the team's model · "
                                 "Not scored yet")
    me = page.locator(f"[data-latest-row='{OURS}']")
    assert me.get_attribute("data-latest-ours") == "true"
    assert me.locator(".mt-mark").get_attribute("style") == "background:var(--s6)"
    # its card's claim: GPQA Diamond and HLE, no ±, each letter linked to the card at its commit
    card = page.locator(f"[data-latest-row='{CARD}']")
    assert card.get_attribute("data-latest-ours") == "false"
    assert rows[CARD]["cells"]["gpqa"] == "89.2" and rows[CARD]["cells"]["hle"] == "25.6"
    assert not rows[CARD]["se"]["gpqa"] and rows[CARD]["cells"]["otis"] == "–" \
        and rows[CARD]["cells"]["mmlupro"] == "–"
    links = card.locator("[data-card-link]")
    assert links.count() == 2
    for k in range(2):
        assert f"/blob/{COMMIT}/README.md" in links.nth(k).get_attribute("href")
    assert rows[CARD]["src"] == "Ornith's model card (5-run average)"
    # the team's rows as before — 25: instruction following not in yet (plan I)
    assert rows["step4000"]["src"].startswith("Our run · 7 of 9 benchmarks in")
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(SCREENS / f"latest-{width}-{scheme}.png"), full_page=True)


@pytest.mark.parametrize("width", [1280, 400])
def test_the_report_lists_it_in_our_runs_and_its_picker(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    # 27b: modelTable's row (on a phone, its card's; the row is in the page)
    page.wait_for_selector(f"[data-rp-row='{OURS}']", state="attached", timeout=30000)
    row = page.locator(f"[data-rp-row='{OURS}']")
    assert row.locator("[data-mt-mark='here']").get_attribute("style") == "background:var(--s6)"
    assert page.locator(f"[data-rp-status='{OURS}']").inner_text() == "Not scored yet"
    assert "measured here, on its card's settings; not the team's model" in row.inner_text()
    picks = page.eval_on_selector_all("[data-rp-pick]", "bs => bs.map(b => b.textContent)")
    assert picks == ["step8804", "step4000", "step1000", "Q4 (UD-Q4_K_XL)", "BF16",
                     "Q2 (UD-Q2_K_XL)", OURS]                          # 23b: step8804 first
    page.locator(f"[data-rp-pick='{OURS}']").click()
    page.wait_for_selector(f"[data-rp-pick='{OURS}'][aria-checked='true']")
    assert page.locator("[data-rp-summary]").inner_text()
    # its card's row among the other models
    assert page.locator(f"[data-rp-every] tr[data-rp-every-row='{CARD}'], "
                        f"[data-rp-every] tr[data-rp-every-rows*='{CARD}']").count() == 1
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-report-section='ours']").screenshot(
        path=str(SCREENS / f"report-ours-{width}.png"))


def test_the_ran_out_report_has_it_in_its_picker(live, page):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=ran-out-of-room")
    page.wait_for_selector("[data-ro-picker]", timeout=30000)
    picks = page.eval_on_selector_all("[data-ro-pick]", "bs => bs.map(b => b.textContent)")
    assert picks == ["step4000", "step1000", OURS]
    page.locator(f"[data-ro-pick='{OURS}']").click()
    page.wait_for_selector(f"[data-ro-pick='{OURS}'][aria-checked='true']")
    assert page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / "ran-out-picker-1280.png"))
