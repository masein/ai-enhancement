"""17d on the page: at the month's limit the grading card says the limit once
— beside its disabled button, where it printed it twice. Nothing runs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.dashboard
LIMIT = "this month's AI spend has reached its limit of $20.00"


def test_26_the_limit_is_said_once(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=ai")
    page.wait_for_selector("[data-frontier-grading]")
    page.evaluate("""limit => {
      const G = JSON.parse(JSON.stringify(state.ai.frg || {}));
      G.has_key = true;
      G.estimate = { ...(G.estimate || {}), answers: 3, usd: 0.01, usd_known: true,
        over_limit: limit, rows: [] };
      G.running = [{ slot: 'simpleqa', task: 'simpleqa_epoch', model: 'm', n: 3, progress: '' }];
      G.waits = [{ why: limit, carry: 'Carry on waits until the limit is raised.' }];
      state.ai.frg = G; render();
    }""", LIMIT)
    card = page.locator("[data-frontier-grading]")
    assert card.inner_text().count(LIMIT) == 1
    assert card.locator("[data-frontier-start]").is_disabled()
    assert page.errors == []
