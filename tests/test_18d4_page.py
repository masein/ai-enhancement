"""18d, part 4: the page's words (docs/prompts/phase-18d-eleventh-review-scores-
oracle-speed.md, points 26 and 27; 25 is tests/test_18d4_page_browser.py). Each
fails on 0b58df7. A stand-in server; no model runs."""

from __future__ import annotations

from pathlib import Path

import pytest

import agent_bench as ab
from fake_openai import FakeServer
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture

ROOT = Path(__file__).resolve().parent.parent


def test_26_a_server_that_gives_no_window_says_nothing_not_the_models_training_one(svc):  # noqa: F811
    from service import db, served
    srv = FakeServer()
    try:
        mid = "served/window-18d"
        db.served_put({"id": mid, "name": "w", "base_url": srv.base, "key": "",
                       "based_on": "", "how": "x", "thinking": "auto", "phone": False,
                       "pin": served.pin_of(served.probe(srv.base, "")), "by": "masein",
                       "at": 0})
        srv.ctx = 0                       # its /props says n_ctx 0; its model 262,144 trained
        with pytest.raises(ValueError) as e:
            served.use_new_window(mid, "masein")
        assert "(it says nothing)" in str(e.value) and "262,144" not in str(e.value)
        got = svc.get("/api/served/pin", params={"id": mid}).json()
        assert got["window"] is None
        assert "It says nothing" in got["why"] or "(it says nothing)" in got["why"], got
        assert "262,144" not in got["why"]                     # 0b58df7: "it says 262,144"
    finally:
        srv.close()


def test_27_the_row_and_the_summary_count_the_same_tries():
    why = "Docker couldn't pull, build or start its container (RuntimeError: x)"
    rows = [{"task": "a", "attempt": 1, "result": "error", "why": why, "tries": 3,
             "again": [why, why]},                         # asked 3 times, each: why
            {"task": "b", "attempt": 1, "result": "resolved", "why": "", "tries": 2,
             "again": [why]}]                               # asked twice: once before
    words = ab.again_words(rows)
    kind = ab.again_kind(why)
    assert f"{kind} (2 tasks, 4 tries)" in words, words      # 0b58df7: 3 tries, the row's 3 + 1
    src = (ROOT / "scripts" / "report_lm_eval.py").read_text()
    assert "asked ${r.tries} times, each: ${r.why}" in src   # the row: every try it ended in
