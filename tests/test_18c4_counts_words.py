"""18c part 4, the counts and words behind the page (points 21 and 22): a
run's Runs line counts the tasks attempted, errors of ours among them, as
its page and score do; a window the board can't take said in words, never
"None"; the tasks asked again said by tasks and by tries. Invented trials;
a stand-in llama-server; no model runs."""

from __future__ import annotations

import time

import pytest

import agent_bench as ab
import agent_run as ar
from agent18 import trial
from fake_openai import FakeServer
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture


def test_21_the_runs_line_counts_what_its_page_counts(tmp_path):
    rd = tmp_path / "r"
    tasks = [f"inv__c-{k}" for k in range(1, 11)]
    trial(rd, tasks[0])
    trial(rd, tasks[1], result="unresolved")
    trial(rd, tasks[2], result="", exc="RuntimeError", started=False,
          message="Docker compose command failed: pull")                # an error of ours
    b = ab.bench("swebench-multilingual")
    p = ar.progress(rd, b, tasks, 1, time.time() - 3600)
    s = ab.score(ar.results(rd, tasks, 1), 300)
    assert p["line"].startswith("3 of 10 · 1 resolved")                # d94f7f6: 2 of 10
    assert s["n"] == 3 and ab.score_words(s, 300).count("pilot: 3 of 300") == 1
    assert p["done"] == 3


@pytest.mark.parametrize("said, words", [
    (None, "it says nothing"), ("", "it says nothing"), (512, "it says 512"),
    (1024, "it says 1,024"), ("65536", "it says '65536'"), (True, "it says True")])
def test_22_a_window_the_board_cant_take_said_in_words(said, words):
    from service import served
    got = served.unusable_window(said)
    assert got == f"its server doesn't report a usable context window ({words})"
    assert "None" not in got                                            # d94f7f6: "of None"


def test_22_the_window_refusals_say_it_so(svc):  # noqa: F811
    import service.app as appmod
    from service import db, served
    srv = FakeServer()
    try:
        mid = "served/window-words-18c4"
        db.served_put({"id": mid, "name": "w", "base_url": srv.base, "key": "",
                       "based_on": "x/y", "how": "-c 16384", "thinking": "auto", "phone": False,
                       "pin": served.pin_of(served.probe(srv.base, "")), "by": "masein",
                       "at": 0})
        srv.ctx = None
        got = appmod.served_pin(id=mid)
        assert got["why"].startswith("Its server doesn't report a usable context window") \
            or got["why"] == "" or "context window" in got["why"]
        assert "None" not in got["why"]                                   # d94f7f6: of None
        srv.ctx = 1024
        got = appmod.served_pin(id=mid)
        assert "(it says 1,024): the board takes a whole number of at least 4,096 tokens" in \
            got["why"], got["why"]
        with pytest.raises(ValueError) as e:
            served.use_new_window(mid, "masein")
        assert str(e.value).startswith("its server doesn't report a usable context window "
                                       "(it says 1,024)")
    finally:
        srv.close()


def test_22_tasks_asked_again_said_by_tasks_and_tries():
    rs = [{"task": "a", "attempt": 1, "again": ["RuntimeError: Docker compose command failed",
                                                "RuntimeError: Docker compose command failed"]},
          {"task": "b", "attempt": 1, "again": ["RuntimeError: Docker compose command failed"]},
          {"task": "c", "attempt": 1, "again": [ab.DOWN_WORDS + " (ServerDown: x)"]}]
    assert ab.again_words(rs) == (
        "3 tasks asked again after an error of ours: Docker couldn't pull, build or start its "
        "container (2 tasks, 3 tries) · the model's server went down (1 task, 1 try)")
