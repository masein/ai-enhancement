"""15.6: remote_run.py's progress lines — each task's pace is its own. A task's
clock starts when the task before it ended, not when the session began: on
gemma-1, dm_math's fourth answer said "1255.1 s an answer · 10.5 h left" 84
minutes in, the first two tasks' hours counted as its own. The time left on
the run is each task's answers left at that task's pace. The answers and the
clock are given; nothing runs."""

from __future__ import annotations

import pytest

import remote_run as rr
from service import db

TOTAL = {"dm_ifeval": 100, "dm_mmlu_pro": 66, "dm_math": 34}


@pytest.fixture
def watch(monkeypatch):
    have = dict.fromkeys(TOTAL, 0)
    monkeypatch.setattr(rr.Watch, "_answered", lambda self, t: have[t])
    monkeypatch.setattr(rr.Watch, "_total", lambda self, t: TOTAL[t])
    monkeypatch.setattr(db, "get", lambda sid: {})
    said: list[str] = []
    w = rr.Watch(1, list(TOTAL), said.append)
    w.started = 0.0
    return w, have, said


def answer(w, have, task: str, at: float) -> None:
    have[task] += 1
    w.tick(now=at)


def test_each_task_is_timed_from_when_the_one_before_it_ended(watch):
    w, have, said = watch
    for i in range(1, 101):                     # IFEval: an answer every 10 s, done at 1,000 s
        answer(w, have, "dm_ifeval", 10.0 * i)
    # done: nothing left on it, and the 100 answers of the other two at the only pace measured
    assert said[-1] == ("dm_ifeval  100/100 · 10.0 s an answer · 0 min left on this task · "
                        "17 min on the run")
    # MMLU-Pro: the model loads again, its first answer at 1,100 s, then one every 40 s
    answer(w, have, "dm_mmlu_pro", 1100.0)
    assert said[-1].startswith("dm_mmlu_pro  1/66 · 100.0 s an answer · ")   # from IFEval's end
    for i in range(1, 66):
        answer(w, have, "dm_mmlu_pro", 1100.0 + 40 * i)        # done at 3,700 s
    # MATH: its first answer 100 s after MMLU-Pro's last, then one every 50 s
    for i in range(4):
        answer(w, have, "dm_math", 3800.0 + 50 * i)
    # its own pace — (3,950 − 3,700) / 4 — not 3,950 s over its 4 answers
    assert said[-1] == "dm_math  4/34 · 62.5 s an answer · 31 min left on this task · 31 min on the run"


def test_the_time_left_on_the_run_is_each_tasks_own_pace(watch):
    w, have, said = watch
    for i in range(1, 101):
        answer(w, have, "dm_ifeval", 10.0 * i)
    answer(w, have, "dm_mmlu_pro", 1060.0)
    answer(w, have, "dm_mmlu_pro", 1120.0)
    # MMLU-Pro at 60 s an answer: its 64 left, and MATH's 34 at that pace (not begun)
    assert said[-1] == ("dm_mmlu_pro  2/66 · 60.0 s an answer · 1.1 h left on this task · "
                        "1.6 h on the run")
    # IFEval's 10 s doesn't stand in for the others: 98 answers at 60 s, not 10
    assert w.pace_of("dm_ifeval") == 10.0 and w.pace_of("dm_math") == 60.0


def test_a_session_resumed_part_way_starts_its_clock_with_the_session(monkeypatch):
    have = {"dm_ifeval": 300, "dm_mmlu_pro": 40, "dm_math": 0}
    monkeypatch.setattr(rr.Watch, "_answered", lambda self, t: have[t])
    monkeypatch.setattr(rr.Watch, "_total", lambda self, t: {**TOTAL, "dm_ifeval": 300}[t])
    monkeypatch.setattr(db, "get", lambda sid: {})
    said: list[str] = []
    w = rr.Watch(1, list(TOTAL), said.append)
    w.started = 0.0
    have["dm_mmlu_pro"] += 1
    w.tick(now=45.0)                        # the 41st answer, 45 s into this session
    assert said == ["dm_mmlu_pro  41/66 · 45.0 s an answer · 19 min left on this task · "
                    "44 min on the run"]
