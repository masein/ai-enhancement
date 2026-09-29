"""12a.9 (the brief's) on the page: a Summarise answer marked on the judge's
checklist — the email thread without its numbers, which the judge called
complete. What the code decided instead is greyed under the verdict, and the
judge's checklist is folded beneath, for the audit. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import everyday as ev
from test_12a9_checklist import EMAIL, QWEN_EMAIL, listed
from test_12n1_everyday_browser import GOOD, everyday, open_group

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a9"


@pytest.mark.parametrize("width", [1400, 400])
def test_a_fact_without_its_number_is_missing_and_said(live, page, width):
    import service.app as appmod
    d = Path(live["tree"]["models"][GOOD]["dir"])
    f = d / ev.OUT_NAME
    was = f.read_bytes()
    out = json.loads(was)
    it = next(x for x in out["items"] if x["id"] == EMAIL)
    q = next(q for q in ev.load_bank() if q["id"] == EMAIL)
    v = ev.parse_verdict(listed(EMAIL), ev.judge_check(q), q, QWEN_EMAIL)
    it.update({"answer_text": QWEN_EMAIL,
               **{k: v[k] for k in ("pass", "reason", "score", "findings", "dropped", "judge_raw")}})
    try:
        f.write_text(json.dumps(out), encoding="utf-8")
        appmod._cache.update(at=0.0)
        everyday(page, live, width=width)
        open_group(page, "summarising")
        card = page.locator(f"[data-grp-answer='{GOOD}|{EMAIL}']")
        assert card.locator("[data-grp-decided]").inner_text() == \
            "2 of 4: missing: 10 hours / overtime; 9:30 / 9.30 (−2)"
        notes = card.locator("[data-evd-dropped='correct']").all_inner_texts()
        assert notes == [
            "the judge said it has “10 hours / overtime”; the answer doesn’t say “10 hours”",
            "the judge said it has “9:30 / 9.30”; the answer doesn’t say “9:30”"]
        grey = page.evaluate("getComputedStyle(document.querySelector('.evdropped')).color")
        body = page.evaluate("getComputedStyle(document.querySelector('.evans')).color")
        assert grey != body
        fold = card.locator(f"[data-evd-judge-raw='{GOOD}']")
        assert fold.get_attribute("open") is None
        assert fold.locator("summary").inner_text() == "the judge’s checklist ▸"
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        SCREENS.mkdir(parents=True, exist_ok=True)
        card.screenshot(path=SCREENS / f"number-missing-{width}.png")
        fold.locator("summary").click()
        raw = json.loads(fold.locator("pre").inner_text())
        assert {r["status"] for r in raw["checklist"]} == {"correct"}   # as the judge said it
        assert page.errors == []
    finally:
        f.write_bytes(was)
        appmod._cache.update(key=None, payload=None, at=0.0)
