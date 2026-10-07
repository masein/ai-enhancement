"""17i on the page, part 5 (points 27 to 31): the runs on rented GPUs are rows
of Runs itself — the same columns, status chip and progress bar as a run
here, one row a model, benchmark and thinking setting with its boxes merged,
in plain words, with two named times, one line saying when the list was read.
The boxes are invented, as the fetch posts them; nothing is fetched."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase17i"
HLE, GPQA = "hle_text_cais", "gpqa_diamond_epoch"
REPO = Path(__file__).resolve().parents[1]


def step(model, label, step_, state="asking", tasks=(HLE,), thinking="on", line="", **more):
    now = time.time()
    return {"label": label, "model": model, "step": step_, "thinking": thinking,
            "tasks": list(tasks), "state": state, "at": now - 60, "seen_at": now,
            "started_at": "2026-10-07T06:00:00Z", "line": line, "sessions": 1,
            "reachable": True, "safe": False, "box_id": label.lower(), **more}


def post(steps, asked=None, reached=None, every=180):
    import import_frontier as imf
    asked = asked if asked is not None else sorted({s["box_id"] for s in steps})
    imf.store_boxes([{"asked": asked, "reached": reached if reached is not None else asked,
                      "every": every}, *steps])


def runs_page(page, live):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("[data-queue-table]")
    # 17j: a rented run pages with the rest, by when it started: these
    # started this morning, under the day's runs here — found as a person
    # finds them, by where they ran
    where(page, "rented")


def where(page, value):
    """the where filter, chosen as a person chooses it — and seen taken (on CI
    the option's click once fell while the list redrew, and nothing was
    chosen)"""
    for _ in range(3):
        page.get_by_label("where filter").click()
        page.locator(f"[role=listbox][aria-label='where filter'] [data-value='{value}']").click()
        try:
            page.wait_for_function(f"state.qWhere === '{value}'", timeout=4000)
            return
        except Exception:                               # noqa: BLE001 — chosen again
            page.keyboard.press("Escape")
    raise AssertionError(f"the where filter never took {value!r}")


def run_row(page, model, tasks, thinking="on"):
    return page.locator(f"[data-rented-run='rented:{model}|{tasks}|{thinking}']")


# ---------------------------------------------------------------------------
# 27: one row a run, its boxes merged; the import's row once imported
# ---------------------------------------------------------------------------

def test_27_a_run_on_four_boxes_is_one_row_of_runs_and_its_import_replaces_it(live, page):
    import import_frontier as imf
    from service import config, db
    m = "served/four-box"
    counts = [(103, 540), (104, 540), (103, 539), (104, 539)]
    post([step(m, f"A{k}", f"A{k}-1", shard=f"{k}/4",
               line=f"Humanity's Last Exam {n} of {of} · 30 s an answer · 4.5 h left")
          for k, (n, of) in enumerate(counts, 1)])
    runs_page(page, live)
    row = run_row(page, m, HLE)
    row.wait_for()                                       # 511854e: a table apart
    assert row.locator("td").count() == page.locator("[data-queue-table] thead th").count()
    assert row.locator("[data-rented-count]").inner_text() == "414 of 2,158"
    assert row.locator("[data-rented-open]").inner_text() == "4 boxes ▸"   # 17j
    assert row.locator("[data-stage]").inner_text() == "running"
    assert row.locator("[data-run-bar]").count() == 1
    assert row.locator("[data-run-where]").inner_text() == "rented GPU · boxes A1, A2, A3, A4"
    row.locator("[data-rented-open]").click()
    page.wait_for_selector("[data-rented-boxes] [data-rented-box='A4|A4-1']")
    assert page.locator("[data-rented-boxes] li").count() == 4
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(page.locator("[data-all-runs]"), SCREENS / "runs-rented-merged.png")
    # the where filter: this server hides it, rented GPUs shows it
    where(page, "here")
    page.wait_for_function("!document.querySelector('[data-rented-run]')")
    where(page, "rented")
    row.wait_for()
    # shard 1 imported: its Runs row stands for it, never a second row for that box
    sid = db.add(m, "instruct", "frontier", "masein", "imported from a rented GPU (RTX 5090)"
                 " · shard 1 of 4", status="done")
    db.update(sid, where_ran="rented GPU · RTX 5090 · box A1")
    d = config.OUT_DIR / "served__four-box__thinking"
    d.mkdir(parents=True, exist_ok=True)
    (d / imf.REGISTRY).write_text(json.dumps({"imports": [
        {"sid": sid, "tasks": [HLE], "box": "A1", "shard": [1, 4], "gpu": "RTX 5090"}]}))
    page.evaluate("state.boxesAt = 0; loadQueue()")
    sel = f"[data-rented-run='rented:{m}|{HLE}|on'] [data-rented-count]"
    page.wait_for_function(f"(document.querySelector(\"{sel}\") || {{}}).textContent"
                           " === '311 of 1,618'", timeout=10000)
    assert run_row(page, m, HLE).locator("[data-rented-open]").inner_text() == "3 boxes ▾"  # open
    page.wait_for_selector(f"[data-queue-row='{sid}']")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 28: plain words; a step stopped by hand says so, never running with a finish
# ---------------------------------------------------------------------------

def test_28_each_state_in_plain_words(live, page):
    now = time.time()
    post([step("served/w-done", "B1", "B1-1", state="whole", line="all answered"),
          step("served/w-run", "B2", "B2-1", line="Humanity's Last Exam 5 of 540 · 1 h left"),
          step("served/w-load", "B3", "B3-1", state="starting"),
          step("served/w-stop", "B4", "B4-1", state="stopped", why="llama-server didn't come up"),
          {**step("served/w-gone", "B5", "B5-1"), "reachable": False, "seen_at": now - 3 * 3600},
          # the BF16 box's A3-2, stopped by hand: asking, and silent for three hours
          {**step("served/w-bf16", "A3", "A3-2", line="Humanity's Last Exam 25 of 539 · "
                  "2.4 h left"), "at": now - 3 * 3600}])
    runs_page(page, live)
    # 17j: spelt as a run here is
    want = {"served/w-done": "done", "served/w-run": "running",
            "served/w-load": "loading the model",
            "served/w-stop": "stopped: llama-server didn't come up",
            "served/w-gone": "no contact for 3 h", "served/w-bf16": "stopped? no word for 3 h"}
    for m, words in want.items():
        chip = run_row(page, m, HLE).locator("[data-stage]")
        chip.wait_for()
        assert chip.inner_text() == words, m             # 511854e: whole, asking, starting…
    bf16 = run_row(page, "served/w-bf16", HLE)
    assert bf16.locator("[data-rented-finish]").count() == 0   # 511854e: a finish time
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(page.locator("[data-queue-table]"), SCREENS / "runs-rented-states.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 29: two times, each named
# ---------------------------------------------------------------------------

def test_29_this_benchmarks_finish_and_its_boxs_are_two_times(live, page):
    # A9's plan: GPQA, then Humanity's Last Exam after it
    post([step("served/two-times", "A9", "A9-1", tasks=(GPQA,),
               line="GPQA Diamond 300 of 792 · 18 s an answer · 2.4 h left")])
    runs_page(page, live)
    row = run_row(page, "served/two-times", GPQA)
    row.wait_for()
    finish = page.evaluate("state.boxes.runs.find(r => r.hf_id === 'served/two-times').finish")
    box = page.evaluate("state.boxes.boxes.find(b => b.model === 'served/two-times')")
    assert abs(finish - (box["at"] + 2.4 * 3600)) < 5     # 511854e: the box's, later steps in
    assert box["box_finish"] > box["task_finish"] == finish
    # 17j: beside the count; one box has nothing to open — its box's time on
    # its where line
    assert row.locator("[data-rented-finish]").inner_text().startswith("→ ")
    assert row.locator("[data-rented-open]").count() == 0
    assert row.locator("[data-run-where] [data-box-finish]").inner_text().startswith(
        " · its box → ")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 30: when the list was read, once; "last heard" only for a row behind
# ---------------------------------------------------------------------------

def test_30_the_list_says_when_it_was_read_and_a_row_only_when_it_is_behind(live, page):
    now = time.time()
    post([step("served/fresh", "C1", "C1-1", line="Humanity's Last Exam 9 of 540 · 4 h left"),
          {**step("served/late", "C2", "C2-1", line="Humanity's Last Exam 9 of 540 · 4 h left"),
           "reachable": False, "seen_at": now - 2 * 3600}], reached=["c1"])
    runs_page(page, live)
    line = page.locator("[data-rented-line]")
    line.wait_for()                                       # 511854e: "13m ago" on every row
    words = line.inner_text()
    assert words.startswith("Rented boxes read ") and "next in about 3 min" in words   # 17j
    assert run_row(page, "served/fresh", HLE).locator("[data-rented-heard]").count() == 0
    late = run_row(page, "served/late", HLE).locator("[data-rented-heard]")
    assert late.inner_text().startswith("heard ")
    # the docs' fetch line reads every 3 minutes
    docs = (REPO / "docs" / "REMOTE-RUNS.md").read_text()
    g3 = next(x for x in docs.splitlines() if x.startswith("python3 scripts/frontier_fetch.py"))
    assert "--every 3m" in g3 and "--every 15m" not in g3   # 511854e: 15m
    assert page.errors == []


# ---------------------------------------------------------------------------
# 31: a build's steps gone from a box the fetch reached leave at once
# ---------------------------------------------------------------------------

def test_31_steps_gone_from_a_reached_box_leave_the_list_at_once(live, page):
    post([step("served/gone-build", "D1", "D1-1", line="Humanity's Last Exam 9 of 540 · 4 h left"),
          step("served/kept-build", "D1", "D1-2", tasks=(GPQA,),
               line="GPQA Diamond 9 of 792 · 4 h left")])
    runs_page(page, live)
    run_row(page, "served/gone-build", HLE).wait_for()
    # the next reading reached D1, and the first build's step isn't on it
    post([step("served/kept-build", "D1", "D1-2", tasks=(GPQA,),
               line="GPQA Diamond 19 of 792 · 4 h left")])
    page.evaluate("state.boxesAt = 0; state.queueRedraw()")
    page.wait_for_function("!document.querySelector(\"[data-rented-run^='rented:served/gone-build']\")",
                           timeout=10000)                 # 511854e: "not reached" for a day
    assert run_row(page, "served/kept-build", GPQA).count() == 1
    assert page.errors == []
