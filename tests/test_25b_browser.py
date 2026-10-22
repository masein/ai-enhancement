"""25b in a real browser, at 1280 light and 400 dark: Gemma 4 E4B and E2B in
"Every model", sorted with the rest by GPQA Diamond, Google's note once under
each name and a link on each of their numbers; IFEval and IFBench columns
there now; and their tiles in "Where we stand" with the two Gemma dots. The
world is test_25_browser's: 21c's scorecard, the cards entered by masein,
the instruction-following answers scored on top."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21d_browser import no_sideways
from test_25_browser import board, heads, report, shot  # noqa: F401 — board: the world
from test_25b_gemma import ARXIV, CARD, NOTE

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase25b"


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (400, "dark")])
def test_gemma_in_every_model_and_their_tiles(live, page, width, scheme):
    report(page, live, width, scheme, wait="[data-rp-every]")
    assert {"IFEval", "IFBench"} <= set(heads(page, "[data-rp-every]"))
    names = page.eval_on_selector_all("[data-rp-every] tbody tr td:first-child",
                                      "ts => ts.map(t => t.title)")
    i4, i2 = names.index("Gemma 4 E4B"), names.index("Gemma 4 E2B")
    assert i4 < i2                                       # by GPQA Diamond: 58.6, then 43.4
    for label in ("Gemma 4 E4B", "Gemma 4 E2B"):
        assert page.locator(f"[data-their-note='{label}']").inner_text() == NOTE
    row = page.locator("[data-rp-every] tbody tr", has=page.locator(
        "[data-their-note='Gemma 4 E4B']"))
    cells = row.locator("td").evaluate_all("ts => ts.map(t => t.textContent.trim())")
    keys = heads(page, "[data-rp-every]")
    got = dict(zip(keys, cells))
    assert got["GPQA Diamond"].startswith("58.6") and got["MMLU-Pro"].startswith("69.4")
    assert got["IFEval"].startswith("96.7") and got["IFBench"].startswith("44.0")
    assert got["HLE"] == got["OTIS Mock AIME"] == got["ARC-AGI-2"] == "–"          # 27's dash
    links = row.locator("[data-card-link]").evaluate_all("as => as.map(a => a.href)")
    assert links == [CARD["Gemma 4 E4B"], CARD["Gemma 4 E4B"], ARXIV, ARXIV]
    # Where we stand: IFEval's and IFBench's tiles, step8804's number in each
    tiles = page.eval_on_selector_all("[data-rp-tile]", "ts => ts.map(t => t.dataset.rpTile)")
    assert {"ifeval", "ifbench"} <= set(tiles)
    assert page.locator("[data-rp-tile='ifeval'] .rp-big").inner_text() == "30.0 ± 10.5"
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, page.locator("[data-rp-tile='ifbench']"), SCREENS / f"tile-{width}-{scheme}.png")
    # the row is inside Every model's own scroller: shot as the element, scrolled to
    both = page.locator("[data-rp-every] tbody tr").filter(has=page.locator("[data-their-note]"))
    if width >= 640:                                     # 27b: on a phone, 27's cards
        both.first.scroll_into_view_if_needed()
    page.locator("[data-report-section='every']").screenshot(
        path=str(SCREENS / f"every-{width}-{scheme}.png"))
