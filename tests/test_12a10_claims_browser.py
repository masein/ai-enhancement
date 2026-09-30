"""12a.10 (the brief's) on the page: the judge's "invented or wrong" checked
against the text. The school notice's lead-in and option heading are style,
dropped and greyed under the verdict, which is 3 for its several versions; the
business cards listed as decided stand, with the line of the text they
contradict, and the judge's checklist is folded beneath. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import everyday as ev
from test_12a10_claims import (CARDS, CARDS_LINE, LOGO, LOGO_DECIDED, NOT_IN, SCHOOL, SCHOOL_3,
                               reply)
from test_12n1_everyday_browser import GOOD, everyday, open_group

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a10"
STYLE = [("Here are a few shortened options:", NOT_IN), ("Option 2: Bulleted version", NOT_IN)]


def _marked(out: dict, qid: str, answer: str, text: str) -> None:
    it = next(x for x in out["items"] if x["id"] == qid)
    q = next(q for q in ev.load_bank() if q["id"] == qid)
    v = ev.parse_verdict(text, ev.judge_check(q), q, answer)
    it.update({"answer_text": answer,
               **{k: v[k] for k in ("pass", "reason", "score", "findings", "judge_raw")},
               "dropped": v.get("dropped") or []})


@pytest.mark.parametrize("width", [1400, 400])
def test_style_is_dropped_and_greyed_and_a_wrong_claim_stands(live, page, width):
    import service.app as appmod
    d = Path(live["tree"]["models"][GOOD]["dir"])
    f = d / ev.OUT_NAME
    was = f.read_bytes()
    out = json.loads(was)
    _marked(out, SCHOOL, SCHOOL_3, reply(SCHOOL, STYLE, several=True, length="yes"))
    _marked(out, LOGO, LOGO_DECIDED, reply(LOGO, [(CARDS, CARDS_LINE)]))
    try:
        f.write_text(json.dumps(out), encoding="utf-8")
        appmod._cache.update(at=0.0)
        everyday(page, live, width=width)
        open_group(page, "summarising")
        card = page.locator(f"[data-grp-answer='{GOOD}|{SCHOOL}']")
        assert card.locator("[data-grp-decided]").inner_text() == \
            "the judge: 3 of 4: several versions (−1)"
        assert card.locator("[data-evd-dropped='style']").all_inner_texts() == [
            f"the judge counted “{c}” as invented or wrong; it is a lead-in or heading: style is "
            "never a finding" for c, _ in STYLE]
        grey = page.evaluate("getComputedStyle(document.querySelector('.evdropped')).color")
        body = page.evaluate("getComputedStyle(document.querySelector('.evans')).color")
        assert grey != body
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        SCREENS.mkdir(parents=True, exist_ok=True)
        card.screenshot(path=SCREENS / f"style-dropped-{width}.png")
        cards = page.locator(f"[data-grp-answer='{GOOD}|{LOGO}']")
        assert cards.locator("[data-grp-decided]").inner_text() == \
            f"2 of 4: invented or wrong: “{CARDS}” (−2)"
        assert cards.locator("[data-evd-dropped]").count() == 0
        cards.screenshot(path=SCREENS / f"cards-kept-{width}.png")
        fold = cards.locator(f"[data-evd-judge-raw='{GOOD}']")
        fold.locator("summary").click()
        raw = json.loads(fold.locator("pre").inner_text())
        assert raw["invented_or_wrong"] == [{"quote": CARDS, "source": CARDS_LINE}]
        assert page.errors == []
    finally:
        f.write_bytes(was)
        appmod._cache.update(key=None, payload=None, at=0.0)
