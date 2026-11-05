"""The grading card at the key's limit (masein, 5 Nov), in a real browser at
1400 and 375 px, light and dark: SimpleQA's batch waiting at the key's limit
in plain words, Carry on priced at what it sends (never the HLE batches out
at OpenRouter), the key warning with each part, and the last press kept on the
card. The card's data is the grading page as the board would give it, set
here; nothing calls OpenRouter."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "grading-key-limit"
HALT = ("waiting: OpenRouter refused 20 requests in a row — OpenRouter refused the key: it has "
        "reached its own spending limit, or the account is out of credit. That is fixed on "
        "OpenRouter’s side: raise the key’s limit or add credit there (the key has used $336.15 "
        "of its $350.00 limit). It waits for Start: nothing more is sent until then")
WARN = ("About $47.83 — what Carry on sends $2.62, the batches out at OpenRouter $45.21 — is more "
        "than the OpenRouter key may still spend ($13.85: the key has used $336.15 of its "
        "$350.00 limit): raise the key’s limit on OpenRouter first, or what is sent stops where "
        "the key runs out")


def page_of(real: dict) -> dict:
    """the board's page with SimpleQA waiting at the key's limit and HLE's two
    batches out at OpenRouter"""
    sqa = "served/step2388-k3k6-2611 · thinking"
    return {**real,
            "estimate": {**real["estimate"], "rows": [], "answers": 0, "usd": 0.0,
                         "usd_max": 0.0, "graders": {}},
            "running": [{"slot": "simpleqa", "task": "simpleqa_epoch", "model": sqa, "n": 983,
                         "progress": f"263 of 983 · $0.29 so far · {HALT}"},
                        {"slot": "hle", "task": "hle_text_cais", "model": sqa, "n": 2071,
                         "progress": "0 of 2,071 · $0.00 so far"},
                        {"slot": "hle", "task": "hle_text_cais",
                         "model": "served/step8804-lda-2610 · thinking", "n": 2098,
                         "progress": "0 of 2,098 · $0.00 so far"}],
            "waits": [{"why": f"SimpleQA Verified's grader: {HALT}",
                       "carry": "Carry on sends the rest."}],
            "held": {"answers": 720, "usd": 2.62, "usd_known": True,
                     "out": {"answers": 4169, "usd": 45.21, "usd_known": True}},
            "key_warning": WARN,
            "last_press": {"what": "Carry on", "by": "masein", "refused": False,
                           "at": time.time() - 300,
                           "words": "nothing new to send · took SimpleQA Verified's grader’s "
                                    "batch up again"}}


@pytest.mark.parametrize("width,scheme", [(1400, "light"), (1400, "dark"), (375, "light"),
                                          (375, "dark")])
def test_the_card_says_the_keys_limit_prices_carry_on_and_keeps_the_last_press(
        live, page, monkeypatch, width, scheme):
    from service import config
    from service import frontier_grade as fgr
    real = fgr.status
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(fgr, "status", lambda: page_of(real()))
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=ai")
    set_name(page, "masein")
    page.wait_for_selector("[data-frontier-grading]:not([data-frontier-grading='loading'])")
    card = page.locator("[data-frontier-grading]")
    assert card.locator("[data-frontier-key-warning]").inner_text() == WARN
    assert card.locator("[data-frontier-start]").inner_text() == "Carry on: about $2.62"
    assert HALT in card.locator("[data-frontier-progress='simpleqa|served/step2388-k3k6-2611 "
                                "· thinking']").inner_text()
    lp = card.locator("[data-frontier-last-press='done']").inner_text()
    assert lp.startswith("Last press: Carry on, ") and lp.endswith(
        ", by masein — nothing new to send · took SimpleQA Verified's grader’s batch up again")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    card.screenshot(path=str(SCREENS / f"grading-key-limit-{width}-{scheme}.png"))
