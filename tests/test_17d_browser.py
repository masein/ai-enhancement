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
    # set, drawn and read in one go: the page's own poll redraws the card
    # from the server's state between two steps
    got = page.evaluate("""limit => {
      const G = JSON.parse(JSON.stringify(state.ai.frg || {}));
      G.has_key = true;
      G.estimate = { ...(G.estimate || {}), answers: 3, usd: 0.01, usd_known: true,
        over_limit: limit, rows: [] };
      G.running = [{ slot: 'simpleqa', task: 'simpleqa_epoch', model: 'm', n: 3, progress: '' }];
      G.waits = [{ why: limit, carry: 'Carry on waits until the limit is raised.' }];
      state.ai.frg = G; render();
      const card = document.querySelector('[data-frontier-grading]');
      const start = card.querySelector('[data-frontier-start]');
      return { text: card.innerText, disabled: !!(start && start.disabled) };
    }""", LIMIT)
    assert got["text"].count(LIMIT) == 1, got["text"]
    assert got["disabled"]
    assert page.errors == []
