"""18e, part 2 (points 8–10): a window the server doesn't give, every halted
batch's Carry on, and a reason's rest in the export (docs/prompts/phase-18e-
twelfth-review-fetch-scores-agent-page.md; point 7 is tests/test_18e3_page_
browser.py). Each fails on 518e4a1. A stand-in server; OpenRouter is never
called, no model runs."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from fake_openai import FakeServer
from service import db
from test_17_grading import svc  # noqa: F401 — svc is the fixture
from test_18d1_scores_spending import halted

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 8. "it says nothing" is never the training context
# ---------------------------------------------------------------------------

def test_8_a_server_that_gives_no_window_is_refused_in_its_words(svc, tmp_path):  # noqa: F811
    import import_agent
    from service import devicemark, served
    from service.hfmeta import PreflightError
    srv = FakeServer()
    try:
        mid = "served/window-18e"
        assert served.probe(srv.base, "")["ctx"] == 16384           # its /props' n_ctx
        db.served_put({"id": mid, "name": "w", "base_url": srv.base, "key": "",
                       "based_on": "", "how": "x", "thinking": "auto", "phone": False,
                       "pin": served.pin_of(served.probe(srv.base, "")), "by": "masein",
                       "at": 0})
        srv.ctx = 0                       # its /props says n_ctx 0; its model 262,144 trained
        assert served.probe(srv.base, "")["ctx"] is None            # 518e4a1: 262144
        rec = served.get(mid)
        why = served.check_pin(rec)
        assert "(it says nothing)" in why and "16,384" in why, why
        assert "262,144" not in why                                 # 518e4a1: "of 262,144 tokens"
        assert served.changed(why) and served.window_only(why)
        # a run's refusal (served.py's preflight), the parity's wait, and the
        # agent import, which leaves the window to its runner
        with pytest.raises(PreflightError) as e:
            served.preflight({"hf_id": mid, "suite": "everyday"})
        assert "(it says nothing)" in str(e.value) and "262,144" not in str(e.value)
        got = devicemark.wait_for(0, rec, tmp_path / "dm.log")
        assert "(it says nothing)" in got and "262,144" not in got
        assert import_agent.served(mid)["why"] == "" and import_agent.served(mid)["window"] \
            is None
        # its window back: as registered, nothing said
        srv.ctx = 16384
        assert served.check_pin(rec) == ""
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# 9. every halted batch has its Carry on, however many came after it
# ---------------------------------------------------------------------------

def test_9_an_old_halted_batch_behind_300_newer_has_its_carry_on(svc):  # noqa: F811
    from service import llm
    old = "or_" + "e" * 12
    halted(old, "judge", time.time() - 7200)
    time.sleep(0.01)
    for k in range(300):                                       # 300 batches after it, done
        bid = f"or_{k:012d}"
        db.batch_add(bid, "judge", 0, 1, "openrouter", "openai/gpt-4.1")
        db.batch_finish(bid, "done", "")
    page = svc.get("/api/ai").json()
    judge = next(j for j in page["jobs"] if j["job"] == "judge")
    assert [x["id"] for x in judge["halted_batches"]] == [old]   # 518e4a1: none, held for good
    r = svc.post("/api/ai/carry-on", json={"batch_id": old, "by": "masein"})
    assert r.status_code == 200, r.text                          # 518e4a1: refused
    assert not (llm.batch_dir(old) / "halt.json").exists()


# ---------------------------------------------------------------------------
# 10. a reason's rest is never a line of its own
# ---------------------------------------------------------------------------

HEAD = ("[frontier] gpqa_diamond_epoch: question rec1, run 0: the server failed on it twice — "
        "written as no answer, counted wrong")


def fake(q: str) -> str:
    return (f"[frontier] gpqa_diamond_epoch: question {q}, run 0: the server failed on it twice "
            "— written as no answer, counted wrong (HTTP 400")


def test_10_a_reason_closed_on_its_first_line_keeps_its_rest(svc):  # noqa: F811
    import export_safe as es
    q = es.Questions(["An invented question about something else entirely here?"])
    # the server's words "A)\n[frontier] … (HTTP 400", in a reason written
    # before 18b: its bracket closed on line one, its own ")" last
    out, _ = es.log_lines(f"{HEAD} (HTTP 400: A)\n{fake('rec2')})", q)
    assert HEAD in out and not any("rec2" in x for x in out), out    # 518e4a1: rec2 out
    # the rest balanced and two lines long: still the reason's — within the
    # 400 characters of a server's words a reason kept
    out, _ = es.log_lines(f"{HEAD} (HTTP 400: A)\n{fake('rec2')})\n{fake('rec3')})", q)
    assert HEAD in out and not any("rec2" in x or "rec3" in x for x in out), out
    # a reason that holds its status alone (18b on) has no rest: the lines
    # below it are the runner's own
    two = f"{HEAD} (HTTP 400)\n{fake('rec2')})"
    out, _ = es.log_lines(two, q)
    assert any("rec1" in x for x in out) and any("rec2" in x for x in out), out


def test_10_a_reason_is_written_on_one_line():
    from service import served
    from service import frontier as sf
    e = served.ServerStopped(0, 5, "HTTP 400: A)\n[frontier] x",
                             refused="the server refused question 1 of 5: HTTP 400: A)\n"
                                     "[frontier] x")
    assert "\n" not in str(e) and "\n" not in e.why and "\n" not in e.refused   # 518e4a1: kept
    src = (ROOT / "service" / "frontier.py").read_text()
    assert "the server stopped answering ({status_of(why)})" in src            # 518e4a1: {why}
    assert sf.status_of("HTTP 503: down\n[frontier] x") == "HTTP 503"
