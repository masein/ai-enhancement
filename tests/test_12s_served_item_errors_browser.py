"""12s on the page: an Everyday question a served model's server failed on
says so in the answer's place (never "the model wrote nothing"), with both of
the server's errors under it; one asked without the server's chat parsing is
an answer like any other, with a "raw fallback" badge and what the server
said. On the model's Answers tab and in the group's side-by-side. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import pick_answers

import everyday as ev
import judge
from test_12a10_claims import LEAVE, SCHOOL
from test_12n1_everyday_browser import GOOD, everyday, open_group
from test_home_and_model_page_12b2_browser import open_model

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12s"
SAID = "HTTP 500: The model produced output that does not match the expected peg-native format"
ERR = {"chat": SAID, "fallback": "HTTP 500: the completion failed too"}


@pytest.mark.parametrize("width", [1400, 400])
def test_a_server_failure_and_a_raw_fallback_say_so(live, page, width):
    import service.app as appmod
    f = Path(live["tree"]["models"][GOOD]["dir"]) / ev.OUT_NAME
    was = f.read_bytes()
    out = json.loads(was)
    failed = next(x for x in out["items"] if x["id"] == SCHOOL)
    failed.update({"answer_text": "", "pass": False, "reason": judge.server_failed(ERR),
                   "server_error": ERR})
    failed.pop("failed", None)
    raw = next(x for x in out["items"] if x["id"] == LEAVE)
    raw.update({"raw_fallback": {"error": SAID}})
    try:
        f.write_text(json.dumps(out), encoding="utf-8")
        appmod._cache.update(at=0.0)
        # the model's Answers tab
        open_model(page, live["base"], GOOD, width=width)
        page.locator("[data-mtab='answers']").click()
        pick_answers(page, "everyday")
        page.locator("[data-answers-group='summarising']").click()
        row = page.locator(f"[data-answers-q='{SCHOOL}']")
        row.wait_for()
        assert row.locator(".evreason").inner_text() == judge.server_failed(ERR)
        ans = row.locator(f"[data-evd-answer='{SCHOOL}']")
        # in the answer's place, whole: the row's reason above it is cut to one line
        assert ans.inner_text().split("\n") == [
            "No answer: its server failed on this question.",
            f"asked twice, the server said: {SAID}",
            "without its chat parsing: HTTP 500: the completion failed too"]
        assert ans.get_attribute("data-server-error") == "1"
        assert "wrote nothing" not in row.inner_text()
        badge = page.locator(f"[data-answers-q='{LEAVE}'] [data-evd-raw='{LEAVE}']")
        assert badge.inner_text() == "raw fallback"
        tip = json.loads(badge.get_attribute("data-tip"))
        assert "/apply-template and /completion" in tip[0] and tip[1] == f"the server said: {SAID}"
        # its answer is there, marked as any answer
        assert page.locator(f"[data-answers-q='{LEAVE}'] [data-evd-answer='{LEAVE}']"
                            ).inner_text() == raw["answer_text"]
        assert page.locator("[data-evd-raw]").count() == 1
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        SCREENS.mkdir(parents=True, exist_ok=True)
        row.screenshot(path=SCREENS / f"no-answer-{width}.png")
        page.locator(f"[data-answers-q='{LEAVE}']").screenshot(
            path=SCREENS / f"raw-fallback-{width}.png")

        # the group's side-by-side: the reason, and nothing in the answer's place
        everyday(page, live, width=width)
        open_group(page, "summarising")
        card = page.locator(f"[data-grp-answer='{GOOD}|{SCHOOL}']")
        assert card.locator("[data-grp-decided]").inner_text() == judge.server_failed(ERR)
        assert card.locator("[data-grp-text]").count() == 0
        side = page.locator(f"[data-grp-answer='{GOOD}|{LEAVE}']")
        assert side.locator(f"[data-evd-raw='{GOOD}']").inner_text() == "raw fallback"
        assert side.locator("[data-grp-text]").inner_text() == raw["answer_text"]
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert page.errors == []
    finally:
        f.write_bytes(was)
        appmod._cache.update(key=None, payload=None, at=0.0)
