"""A pilot's report on Reports in a real browser, at 1280 and 400 px, light and dark:
internal, empty until the pilot's summary is put under $BENCH_ROOT/pilot/, then its tiles,
the verdicts (keep and drop), what a full run needs decided, the scores (± its standard
error, at the limit, a score not counted), each test's clean runs and cost, the full-run
estimate, the blind audit unblinded per model, and the boxes, time and money. The summary
is made up (test_pilot_report.made_up); no pilot's real figures are in the repo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from service import config
from test_21d_browser import no_sideways
from test_pilot_report import RID, made_up

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "pilot"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]


def summary() -> Path:
    return config.BENCH_ROOT / "pilot" / "2026-11" / "summary.json"


def open_report(page, live, width: int, scheme: str, wait: str) -> None:
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=reports&report={RID}")
    page.wait_for_selector(wait, state="attached", timeout=30000)


def test_before_the_summary_is_put_here_it_says_so(live, page):
    summary().unlink(missing_ok=True)
    open_report(page, live, 1280, "light", "[data-pl-none]")
    assert page.inner_text("[data-pl-none]") == ("Nothing imported yet: the pilot’s summary isn’t "
                                                 "on this server.")
    assert page.inner_text("[data-pl-source]") == "nothing imported yet"
    page.goto(live["base"] + "/#tab=reports")
    page.wait_for_selector(f"[data-report-card='{RID}']", timeout=30000)
    assert page.inner_text(f"[data-report-internal='{RID}']") == "Internal"
    assert page.inner_text(f"[data-report-models='{RID}']") == "nothing imported yet"
    assert page.errors == []


@pytest.mark.parametrize("width,scheme", SIZES)
def test_the_report_shows_the_pilots_summary(live, page, width, scheme):
    s = made_up()
    summary().parent.mkdir(parents=True, exist_ok=True)
    summary().write_text(json.dumps(s), encoding="utf-8")
    open_report(page, live, width, scheme, "[data-pl-verdicts]")
    tiles = {t.get_attribute("data-pl-tile"): t.inner_text().split("\n")
             for t in page.locator("[data-pl-tile]").all()}
    assert tiles["kept"] == ["Tests kept", "1 of 3"]
    assert tiles["full"] == ["Full run, kept tests", "~$40", "GPU, all 7 models"]
    assert tiles["gpu"] == ["GPU", "$6.50", "of the $10 cap"]
    assert tiles["api"] == ["API", "$1.25", "of the $5 cap"]
    assert tiles["time"] == ["Time", "3 h 30 min", "of the 5 h cap"]
    assert page.locator("[data-pl-model]").count() == 5
    assert page.inner_text("[data-pl-model='ra']") == "Rival-A-9B: Rival-A-9B's invented setup"
    # the verdicts: keep and drop, the best rival by its short name, the drops dimmed
    assert page.locator("[data-pl-verdict]").count() == 3
    assert page.inner_text("[data-pl-says='alpha']") == "keep"
    assert page.inner_text("[data-pl-says='beta']") == "drop"
    assert page.inner_text("[data-pl-best='alpha']") == "C 66.0"
    assert page.get_attribute("[data-pl-verdict='beta']", "class") == "dim"
    assert page.get_attribute("[data-pl-verdict='alpha']", "class") is None
    assert "after a fix" in page.inner_text("[data-pl-verdict='alpha']")
    assert page.locator("[data-pl-decisions] li").all_inner_texts() == [
        "Alpha: fix the simulator.", "Beta stays dropped."]
    # the scores: ± its standard error, none at zero, at the limit, a score not counted
    assert page.inner_text("[data-pl-cell='alpha|high']") == "61.0 ±4.0"
    assert page.inner_text("[data-pl-cell='beta|high']") == "0.0\nchecks met 20.0%"
    assert page.inner_text("[data-pl-cell='gamma|rb']") == "4.0 ±4.0\n11 at limit"
    assert page.inner_text("[data-pl-cell='gamma|ra']") == "limit-bound\nstopped at 14 asked, 12 at limit"
    assert "Gamma's share at the limit:" in page.inner_text("[data-pl-score-notes]")
    # each test's clean runs and cost; High's stored answers cost no GPU
    assert page.locator("[data-pl-clean-row]").count() == 3
    assert page.inner_text("[data-pl-clean-row='Gamma'] td:nth-child(2)") == "92.5"
    assert page.inner_text("[data-pl-clean-row='Alpha'] td:nth-child(2)") == "100"
    assert page.inner_text("[data-pl-usd-row='Gamma'] td:nth-child(2)") == "board"
    assert page.inner_text("[data-pl-usd-row='Alpha'] td:nth-child(2)") == "$0.012"
    # the full-run estimate, the dropped dimmed, the kept total
    assert page.inner_text("[data-pl-all='Alpha']") == "$37"
    assert page.get_attribute("[data-pl-full-row='Beta']", "class") == "dim"
    assert page.inner_text("[data-pl-full-row='Beta'] td:nth-child(3)") == "done"
    assert page.inner_text("[data-pl-full-total]").split("\n")[0].split("\t") == [
        "The 1 kept test, all 7 models", "$37"]
    # the blind audit, unblinded per model
    assert page.locator("[data-pl-unblinded-row]").count() == 5
    assert page.inner_text("[data-pl-unblinded-row='rc']").split("\t") == ["Rival-C-2B", "2", "2", "2", "1", "1"]
    assert page.inner_text("[data-pl-changes]").startswith("Rival C, Alpha 7: simulator fault. It ended early.")
    assert page.inner_text("[data-pl-early-row='ra']").split("\t") == ["Rival-A-9B", "2", "+5.0"]
    assert page.inner_text("[data-pl-early-row='rb']").split("\t") == ["Rival-B-8B", "0", "—"]
    assert "A check that never counts." in page.inner_text("[data-pl-other]")
    # the boxes, time and money
    assert page.inner_text("[data-pl-box='B']").split("\t") == ["B", "the rivals", "0.800", "5.00", "4.00",
                                                                "1 Jan 12:30"]
    assert page.inner_text("[data-pl-box='all']").split("\t")[3:5] == ["7.50", "6.50"]
    caps = page.inner_text("[data-pl-caps]")
    assert "GPU: $6.50 of the $10 cap" in caps and "API: $1.25 of the $5 cap" in caps
    assert "Time: 3 h 30 min, 1 Jan 09:00 to 1 Jan 12:30, against the 5 h cap" in caps
    assert "The key read $10.00 before and $11.25 after." in caps
    assert page.inner_text("[data-pl-creds]") == "No key in any task container."
    assert "One run restarted." in page.inner_text("[data-report-section='pl-disclosures']")
    assert no_sideways(page), "the page scrolls sideways"
    assert page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / f"pilot-{width}-{scheme}.png"), full_page=True)
    # and the list's card says whose, and how many kept
    page.goto(live["base"] + "/#tab=reports")
    page.wait_for_selector(f"[data-report-card='{RID}']", timeout=30000)
    assert page.inner_text(f"[data-report-models='{RID}']") == (
        "step2388 High and Instant against Rival-A-9B, Rival-B-8B and Rival-C-2B · 1 of 3 tests kept")
