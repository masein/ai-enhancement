"""14.4.5 on the page: with MOBILE_MMLU_FULL=0 the full Mobile-MMLU is nowhere
— no table under Mobile tasks, no line on the model page, no group in
Compare, no part in Test a model, Pro alone on the key's card — and switched
back on, all of it is there again from what stayed on disk. Invented rows,
picks written in: nothing runs, nothing calls OpenRouter."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from test_14_3_browser import go, ids, no_sideways, steady_shot
from test_14_4_3_browser import CHANCE, GOOD, SHORT, board  # noqa: F401 — the module's board

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase14_4_5"


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


@pytest.fixture
def switched_off(live):
    import service.app as appmod
    from service import config
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "MOBILE_MMLU_FULL", False)
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def test_switched_off_it_is_nowhere_on_the_page(live, page, switched_off):
    go(page, live, "tab=models&view=mobile&group=mmlu", "th[data-col='mobile_mmlu_pro']")
    assert page.locator("[data-mmf-card]").count() == 0
    assert "Mobile-MMLU (full)" not in page.locator("#view").inner_text()
    # not badged either: nothing on the page is non-commercial
    assert page.locator("[data-restriction='non-commercial']").count() == 0
    no_sideways(page)
    shot(page.locator("#view"), "mobile-tasks-off.png")
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    page.locator(f"[data-mmp-line='{GOOD}']").wait_for()
    assert page.locator(f"[data-mmf-line='{GOOD}']").count() == 0
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, CHANCE)}", "[data-compare='2']")
    assert page.locator("[data-cmp-group='mmf']").count() == 0
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(SHORT, safe=""), "[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    page.locator("[data-mab-part='mmlu']").wait_for()
    assert page.locator("[data-mab-part='mmlu_full']").count() == 0
    assert "full Mobile-MMLU" not in page.locator("[data-dialog='test']").inner_text()
    shot(page.locator("[data-dialog='test'] .dlg"), "test-a-model-off.png")
    go(page, live, "tab=ai", "[data-mmp-key]:not([data-mmp-key='loading'])")
    card = page.locator("[data-mmp-key]")
    assert card.locator("[data-mmp-full-counts]").count() == 0
    assert card.locator("[data-mmp-full-est]").count() == 0
    assert card.locator("[data-mmp-full-check]").count() == 0
    assert "full Mobile-MMLU" not in card.inner_text()
    assert page.errors == []


def test_switched_back_on_it_is_all_there_again(live, page):
    go(page, live, "tab=models&view=mobile&group=mmlu", "[data-mmf-card]")
    assert page.locator(f"[data-mmf-cell='{GOOD}']").inner_text() == "94.4%"
    assert page.errors == []
