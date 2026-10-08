"""18c part 2 on the page (points 9 and 10): SWE-bench Multilingual's card
names the tasks left out — their reference solution doesn't pass here, as
the oracle run read them — and says what isn't comparable with the
published numbers, at 400 and 1440 px, light and dark. An invented oracle
run; nothing is fetched."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import agent_run as ar
from agent18 import run_json, trial
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18c"
NAMES = ["inv__oracle-1", "inv__oracle-2", "inv__oracle-3"]


@pytest.fixture
def oracle_run(live):
    import service.app as appmod
    from service import config
    od = ar.run_dir(Path(config.BENCH_ROOT), "swebench-multilingual", "", 1, True)
    run_json(od, NAMES, model="")
    trial(od, NAMES[0], agent="oracle", exit_status="")
    trial(od, NAMES[1], result="unresolved", agent="oracle", exit_status="")
    trial(od, NAMES[2], agent="oracle", exit_status="")
    appmod._cache.update(key=None, payload=None, at=0.0)
    try:
        yield od
    finally:                                    # the other pages' runs never see it
        shutil.rmtree(od, ignore_errors=True)
        appmod._cache.update(key=None, payload=None, at=0.0)


# the card's lines: none wider than the card, none over another
FITS = """(card) => {
  const bad = [];
  const box = card.getBoundingClientRect();
  for (const e of card.querySelectorAll('[data-agent-left-out], [data-agent-not-comparable], '
                                        + '[data-agent-ref-left-out]')) {
    const r = e.getBoundingClientRect();
    if (r.right > box.right + 1 || e.scrollWidth > e.clientWidth + 1) bad.push(e.textContent);
  }
  return bad;
}"""


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_card_names_the_tasks_left_out_and_what_isnt_comparable(live, page, oracle_run,
                                                                   scheme):
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        page.set_viewport_size({"width": width, "height": 1200})
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=benchmarks")
        card = page.locator("[data-agent-card='swebench-multilingual']")
        card.wait_for()
        left = card.locator("[data-agent-left-out='swebench-multilingual']")
        left.wait_for()
        assert left.inner_text() == ("1 task left out: their reference solution doesn’t pass "
                                     "here")                               # d94f7f6: none
        assert left.get_attribute("title") == NAMES[1]
        assert card.locator("[data-agent-not-comparable]").inner_text() == (
            "A task that runs past its 50 minutes counts as not resolved here; the published "
            "runs had no such limit.")
        assert card.locator("[data-agent-ref-left-out]").inner_text() == (
            "They count all 300 tasks; ours leaves out the 1 above.")
        deep = page.locator("[data-agent-card='deepswe']")
        assert deep.locator("[data-agent-left-out], [data-agent-not-comparable]").count() == 0
        assert card.evaluate(FITS) == [], (width, scheme)
        SCREENS.mkdir(parents=True, exist_ok=True)
        page.set_viewport_size({"width": width, "height": 2400})
        steady_shot(card, SCREENS / f"card-{width}-{scheme}.png")
    assert page.errors == []
