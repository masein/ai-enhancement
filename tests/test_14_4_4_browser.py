"""14.4.4 on the page: each restricted set's badge — in words, in a box, never
colour alone — beside its name wherever it shows, close enough to the
numbers that a screenshot takes it in, and its sentence on a banner where it
has a section of its own: the full Mobile-MMLU "Non-commercial", Pro "Internal
use", Artificial Analysis "Internal only". The results table, its tooltip and
Copy as CSV; the model page; Compare; Test a model's part and the queue's
confirmation; AI models' Outside data card and the Frontier view. At 1400 and
375 px. Invented rows, picks written in, Artificial Analysis's canned file;
nothing runs, nothing calls anyone."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from test_14_3_browser import go, ids, no_sideways, steady_shot
from test_14_4_3_browser import CHANCE, GOOD, SHORT, board  # noqa: F401 — the module's board
from test_14_4_4_restrictions import AA, IU, NC

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase14_4_4"
WIDTHS = [1400, 375]


@pytest.fixture(scope="module", autouse=True)
def reported_aa(live):
    """Artificial Analysis's canned file, imported as 12n.1's tests do"""
    import service.app as appmod
    from service import config, reported
    from test_reported_12m2 import KEY, canned
    saved = config.AA_API_KEY
    config.AA_API_KEY = KEY
    try:
        reported.import_aa(fetch=canned())
    finally:
        config.AA_API_KEY = saved
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


def badge(loc) -> str:
    return loc.inner_text().strip()


@pytest.mark.parametrize("width", WIDTHS)
def test_the_results_table_and_the_sets_own_section(live, page, width):
    go(page, live, "tab=models&view=mobile&group=mmlu", "[data-mmf-card]", width)
    # Pro's column: its badge beside its name, in the header a screenshot takes
    th = page.locator("th[data-col='mobile_mmlu_pro']")
    assert badge(th.locator("[data-col-restriction='mobile_mmlu_pro']")) == "Internal use"
    tip = json.loads(th.get_attribute("data-tip"))
    assert tip[0].endswith(" · Internal use") and tip[1] == IU
    # the full set's own table: its banner, its sentence above its numbers
    card = page.locator("[data-mmf-card]")
    assert card.locator("h2").inner_text() == "Mobile-MMLU (full) Non-commercial"
    banner = card.locator("[data-restriction-banner='mobile_mmlu_full']")
    assert banner.inner_text() == "Non-commercial " + NC
    assert banner.locator(".rbadge").get_attribute("data-restriction") == "non-commercial"
    # in words, in a box: never colour alone
    assert page.evaluate("""() => { const s = getComputedStyle(document.querySelector(
      "[data-mmf-card] .rbadge")); return s.borderStyle !== 'none' && parseFloat(s.borderWidth) > 0 }""")
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"results-table-{width}.png")
    shot(card, f"full-sets-section-{width}.png")
    assert page.errors == []


def test_copy_as_csv_carries_each_restricted_columns_licence_and_restriction(live, page):
    go(page, live, f"tab=models&view=mobile&group=mmlu&models={ids(GOOD, CHANCE)}",
       "th[data-col='mobile_mmlu_pro']")
    got = list(csv.reader(io.StringIO(page.evaluate("lbCsv()"))))
    head = got[0]
    i = head.index("Mobile-MMLU-Pro licence")
    assert head[i + 1] == "Mobile-MMLU-Pro restriction"
    assert {(r[i], r[i + 1]) for r in got[1:]} == {("CC BY-ND 4.0", "Internal use")}
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_model_page(live, page, width):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    page.locator(f"[data-mmf-line='{GOOD}']").wait_for()
    assert badge(page.locator(f"[data-mmp-restriction='{GOOD}']")) == "Internal use"
    assert badge(page.locator(f"[data-mmf-restriction='{GOOD}']").first) == "Non-commercial"
    # and the results list names it with its badge
    assert badge(page.locator("[data-result-restriction='mobile_mmlu_pro']")) == "Internal use"
    no_sideways(page)
    part = page.locator(f"[data-mmp-line='{GOOD}']").locator("xpath=..")
    shot(part, f"model-page-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_compare(live, page, width):
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, CHANCE)}", "[data-compare='2']", width)
    assert badge(page.locator("[data-cmp-restriction='mmf']")) == "Non-commercial"
    assert badge(page.locator("[data-cmp-row-restriction='mobile_mmlu_pro']")) == "Internal use"
    no_sideways(page)
    shot(page.locator("[data-cmp-body='mmf']"), f"compare-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_submit_form_and_the_queues_confirmation(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    queued = []

    def submit(route):
        queued.append(json.loads(route.request.post_data))
        route.fulfill(json={"id": 9001, "status": "queued", "tasks": ["mobile_mmlu_full"],
                            "part": "mmlu_full"})
    page.route("**/api/submissions", submit)
    go(page, live, "model=" + quote(SHORT, safe=""), "[data-model-hero]", width)
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    part = page.locator("[data-mab-part='mmlu_full']")
    part.wait_for()
    assert badge(part.locator("[data-mab-part-restriction='mobile_mmlu_full']")) == "Non-commercial"
    assert badge(page.locator("[data-mab-part-restriction='mobile_mmlu_pro']")) == "Internal use"
    page.wait_for_function("document.querySelector(\"[data-mab-part-est='mmlu_full']\")"
                           "?.textContent.includes('answers')")
    part.locator("input").check()
    assert page.locator("[data-mab-part-sentence='mobile_mmlu_full']").inner_text() == NC
    no_sideways(page)
    shot(page.locator("[data-dialog='test'] .dlg"), f"submit-form-{width}.png")
    page.locator("[data-dialog='test'] button.primary", has_text="Start test").click()
    toast = page.locator("[data-toast='submit']")
    toast.wait_for()
    assert queued and queued[0]["part"] == "mmlu_full"
    assert f"Run #9001 queued · Mobile-MMLU (full): Non-commercial. {NC} —" in toast.inner_text()
    shot(toast, f"queued-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_artificial_analysis_is_internal_only_where_its_numbers_are(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "tab=ai", "[data-outside]", width)
    out = page.locator("[data-outside]")
    page.wait_for_selector("[data-restriction-banner='reported:aa']")
    assert out.locator("[data-restriction-banner='reported:aa']").inner_text() == "Internal only " + AA
    assert badge(out.locator("[data-rep-restriction='aa']")) == "Internal only"
    shot(out, f"outside-data-{width}.png")
    go(page, live, "tab=models&chip=frontier", "[data-frontier-credit]", width)
    credit = page.locator("[data-frontier-credit]")
    assert badge(credit.locator("[data-frontier-restriction='aa']")) == "Internal only"
    assert "Data: Artificial Analysis" in credit.inner_text()
    no_sideways(page)
    shot(credit, f"frontier-credit-{width}.png")
    assert page.errors == []
