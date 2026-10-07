"""17i, part 5: how a run's row is worked out from its boxes — one stopped or
unreachable among several (import_frontier.rented_runs). Steps invented."""

from __future__ import annotations

import time

import import_frontier as imf

HLE = "hle_text_cais"


def box(label, state="asking", n=100, of=540, left_h=4.0, **more):
    now = time.time()
    return {"label": label, "model": "served/m", "step": f"{label}-1", "thinking": "on",
            "tasks": [HLE], "state": state, "at": now - 60, "seen_at": now, "heard": now,
            "reachable": True, "n": n, "of": of, "task_finish": now + left_h * 3600,
            "box_finish": now + left_h * 3600, "quiet_min": None, **more}


def one(*boxes):
    got = imf.rented_runs(list(boxes), time.time(), time.time())
    assert len(got) == 1, got
    return got[0]


def test_one_box_stopped_or_unreached_among_running_ones():
    now = time.time()
    r = one(box("A1"), box("A2", left_h=6), box("A3", state="stopped", why="llama-server died"),
            box("A4", reachable=False, heard=now - 3 * 3600))
    # still Running, counted with every box's last known count, the odd ones said
    assert (r["status"], r["status_words"]) == ("running", "Running")
    assert (r["n"], r["of"], r["boxes_n"]) == (400, 2160, 4)
    assert r["attention"] == ["A3 · A3-1: Stopped: llama-server died",
                              "A4 · A4-1: No contact for 3 h"]
    # the benchmark's finish waits on the stopped and the unreached box: unknown
    assert r["finish"] is None and r["behind"]
    # every box running: the last of their finishes
    r = one(box("A1"), box("A2", left_h=6))
    assert abs(r["finish"] - (now + 6 * 3600)) < 5 and not r["behind"]


def test_with_no_box_running():
    now = time.time()
    assert one(box("A1", state="whole"), box("A2", state="whole"))["status_words"] == "Done"
    assert one(box("A1", state="starting"), box("A2", state="stopped"))["status"] == "loading"
    r = one(box("A1", state="whole"), box("A2", reachable=False, heard=now - 7200))
    assert r["status_words"] == "No contact for 2 h"
    r = one(box("A1", state="whole"), box("A2", state="stopped", why="out of memory"),
            box("A3", reachable=False, heard=now - 7200))
    assert r["status_words"] == "Stopped: out of memory"
    r = one(box("A1", quiet_min=180, at=now - 3 * 3600))
    assert r["status_words"] == "Stopped? No word for 3 h" and abs(r["heard"] - (now - 10800)) < 5
