"""12a.8 (the brief's) on the page: a Summarise answer's verdict is built
from the judge's findings; a claim the answer disproves is shown greyed
under it, and the judge's own reply is folded beneath, for the audit.
Fixtures only. 12a.9: findings on a rubric 12a.8 wrote, as one someone edited
keeps them (test_12a9_checklist_browser.py has the checklist)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import everyday as ev
from test_12a8_findings import findings
from test_12n1_everyday_browser import GOOD, everyday, open_group

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a8b"


@pytest.mark.parametrize("width", [1400, 400])
def test_a_dropped_claim_is_greyed_and_the_judges_reply_folded(live, page, width):
    import service.app as appmod
    d = Path(live["tree"]["models"][GOOD]["dir"])
    f = d / ev.OUT_NAME
    was = f.read_bytes()
    qid = "everyday-summarising-07"
    raw = {"missing_facts": ["mum / mother"], "invented_or_wrong": [], "several_versions": False,
           "length_ok": "not asked", "note": "misses that the mother is coming"}
    out = json.loads(was)
    it = next(x for x in out["items"] if x["id"] == qid)
    q = next(q for q in ev.load_bank() if q["id"] == qid)
    v = ev.parse_verdict(json.dumps(raw), findings(q), q, it["answer_text"]
                         + " Your mom arrives at 7:00 PM.")
    it.update({k: v[k] for k in ("pass", "reason", "score", "findings", "dropped", "judge_raw")})
    try:
        f.write_text(json.dumps(out), encoding="utf-8")
        appmod._cache.update(at=0.0)
        everyday(page, live, width=width)
        open_group(page, "summarising")
        card = page.locator(f"[data-grp-answer='{GOOD}|{qid}']")
        assert card.locator("[data-grp-decided]").inner_text() == \
            "the judge: 4 of 4: all key facts, one version"
        drop = card.locator("[data-evd-dropped='missing']")
        assert drop.inner_text() == "the judge said it missed “mum / mother”; the answer has it"
        grey = page.evaluate("getComputedStyle(document.querySelector('.evdropped')).color")
        body = page.evaluate("getComputedStyle(document.querySelector('.evans')).color")
        assert grey != body                                       # greyed, not the text's colour
        fold = card.locator(f"[data-evd-judge-raw='{GOOD}']")
        assert fold.get_attribute("open") is None
        fold.locator("summary").click()
        assert json.loads(fold.locator("pre").inner_text()) == raw
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        SCREENS.mkdir(parents=True, exist_ok=True)
        card.screenshot(path=SCREENS / f"dropped-claim-{width}.png")
        assert page.errors == []
    finally:
        f.write_bytes(was)
        appmod._cache.update(key=None, payload=None, at=0.0)
