"""17j on the page, part 5 (points 29 to 35): a run on rented GPUs is one line
of Runs as a run here is — sorted, paged and found with them, in the same
words — a finished parity step and an abandoned step said on their box's
line, a step's two benchmarks both counted, and the table fits a phone. The
boxes are invented, as the fetch posts them; nothing is fetched."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from test_14_3_browser import steady_shot
from test_17i_runs_browser import post, run_row, step, where

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase17j"
HLE, GPQA, OTIS, MMLU = "hle_text_cais", "gpqa_diamond_epoch", "otis_aime_epoch", "mmlupro_tiger"


def world():
    """four rented runs and four runs here, each model's name holding "17j-"
    so a search shows these eight alone"""
    import import_frontier as imf
    from service import db
    imf.boxes_path().unlink(missing_ok=True)             # this test's boxes alone
    from contextlib import closing
    with closing(db._conn()) as c:                        # and its runs here
        c.execute("DELETE FROM submissions WHERE hf_id LIKE 'served/17j-%'")
        c.commit()
    now = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 3 * 3600))
    post([
        # Humanity's Last Exam on four boxes; A3's parity step done before it
        *[step("served/17j-bf16", f"A{k}", f"A{k}-{2 if k == 3 else 1}", shard=f"{k}/4",
               started_at=started,
               line=f"Humanity's Last Exam {100 + k} of 540 · 30 s an answer · 4.5 h left")
          for k in range(1, 5)],
        step("served/17j-bf16", "A3", "A3-1", state="whole", tasks=(), parity=True,
             thinking="off", started_at=started, line="parity 1,000 of 1,000"),
        # A1's second step: GPQA then OTIS, in one step
        step("served/17j-q4", "A1", "A1-2", tasks=(GPQA, OTIS), thinking="off",
             started_at=started, line="GPQA Diamond 120 of 198 · 18 s an answer · 2.1 h left"),
        # one box, done; and one whose step was abandoned beside a running one
        step("served/17j-k8", "A5", "A5-1", state="whole", tasks=(MMLU,),
             started_at=started, line="MMLU-Pro: 12,032 of 12,032 answered"),
        step("served/17j-lda", "A6", "A6-1", tasks=(MMLU,), started_at=started,
             line="MMLU-Pro 3,000 of 12,032 · 5.3 s an answer · 13.3 h left"),
        step("served/17j-lda", "A6", "A6-2", state="abandoned", tasks=(HLE,),
             started_at=started),
    ])
    here = []
    for model, status, progress in (
            ("served/17j-here-a", "running", "GPQA Diamond 120 of 198 · 2.1 s an answer · 3.4 h left"),
            ("served/17j-here-b", "done", "GPQA Diamond: 61.4"),
            ("served/17j-here-c", "failed", "the server stopped answering"),
            ("served/17j-here-d", "canceled", "stopped at 40 of 198")):
        sid = db.add(model, "instruct", "frontier", "masein", "", tasks=[GPQA], status=status)
        db.update(sid, progress=progress)
        here.append(sid)
    return here


def runs(page, live, width=1440, q="17j-"):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("[data-queue-table]")
    box = page.get_by_label("filter queue")
    box.fill(q)
    page.wait_for_function(f"state.qQ === {q!r}")
    page.wait_for_selector("[data-rented-run]")


def test_29_a_rented_run_is_one_line_as_its_neighbours(live, page):
    here = world()
    runs(page, live)
    hle = run_row(page, "served/17j-bf16", HLE)
    hle.wait_for()
    mine = page.locator(f"[data-queue-row='{here[0]}']")
    # as tall as a run here with its bar (0abb757: 118 px against 56)
    h_rented, h_here = hle.bounding_box()["height"], mine.bounding_box()["height"]
    assert h_rented <= h_here * 1.2, (h_rented, h_here)
    # the count, the bar and the finish on one line
    parts = [hle.locator(s).bounding_box() for s in
             ("[data-rented-count]", "[data-run-bar]", "[data-rented-finish]")]
    mids = [b["y"] + b["height"] / 2 for b in parts]
    assert max(mids) - min(mids) < 8, mids
    assert hle.locator("[data-rented-finish]").inner_text().startswith("this part done ")  # 18b
    assert hle.locator("[data-rented-open]").inner_text() == "4 boxes ▸"
    one = run_row(page, "served/17j-k8", MMLU)
    assert one.locator("[data-rented-open]").count() == 0     # nothing to open
    # the board's name, no served/…, no instruct badge
    model = hle.locator("td").nth(2)
    assert not model.inner_text().startswith("served/")
    assert hle.locator(".badge").count() == 0
    # sorted with the others: by model, the rented rows among the runs here
    page.locator("[data-queue-table] th", has_text="model").click()
    order = page.evaluate("[...document.querySelectorAll('[data-queue-table] tbody tr')]"
                          ".map(t => t.querySelector('td:nth-child(3)').textContent)")
    names = [x.split("rented GPU")[0].split("this server")[0] for x in order]
    assert names == sorted(names, key=str.lower) or names == sorted(names, key=str.lower,
                                                                    reverse=True), names
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-queue-table] th", has_text="#").click()
    steady_shot(page.locator("[data-all-runs]"), SCREENS / "runs-1440.png")
    assert page.errors == []


def test_30_31_a_parity_step_done_and_an_abandoned_step_are_said_on_their_box(live, page):
    world()
    runs(page, live)
    run_row(page, "served/17j-bf16", HLE).wait_for()
    # neither is a row (0abb757: a row each, "Done" and "No contact")
    assert page.locator("[data-rented-run^='rented:served/17j-bf16|parity']").count() == 0
    assert page.locator("[data-rented-run^='rented:served/17j-lda|hle']").count() == 0
    # the abandoned step on its one-box run's line
    lda = run_row(page, "served/17j-lda", MMLU)
    assert "a step abandoned" in lda.locator("[data-run-where]").inner_text()   # 18c: no id
    # the parity check on its box's line
    run_row(page, "served/17j-bf16", HLE).locator("[data-rented-open]").click()
    a3 = page.locator("[data-rented-box='A3|A3-2']")
    a3.wait_for()
    assert a3.locator("[data-box-notes]").inner_text() == " · parity done"
    assert page.errors == []


@pytest.mark.parametrize("words", ["Humanity's Last Exam", "GPQA Diamond", "thinking off",
                                   "Frontier"])
def test_32_the_search_finds_rented_runs_by_their_words(live, page, words):
    world()
    runs(page, live, q=words)
    got = page.locator("[data-rented-run]").count()
    assert got >= 1, words                                  # 0abb757: none of 9
    texts = page.locator("[data-rented-run]").all_inner_texts()
    assert all(words.lower() in t.lower() for t in texts), texts


def test_33_a_step_asking_two_benchmarks_counts_both(live, page):
    world()
    runs(page, live)
    q4 = run_row(page, "served/17j-q4", f"{GPQA},{OTIS}", thinking="off")
    q4.wait_for()
    assert q4.locator("[data-rented-count]").inner_text() == \
        "GPQA Diamond 120 of 198 · OTIS Mock AIME 2024–2025 next"    # 0abb757: 120 of 198
    run = page.evaluate("state.boxes.runs.find(r => r.hf_id === 'served/17j-q4')")
    assert run["of"] > 198 and run["finish"] > time.time() + 2.1 * 3600   # OTIS after it
    assert page.errors == []


def test_34_one_spelling_no_contact_its_own_status_and_the_top_line(live, page):
    from service import db
    here = world()
    now = time.time()
    post([{**step("served/17j-gone", "B5", "B5-1"), "reachable": False,
           "seen_at": now - 3 * 3600}], asked=["b5"], reached=[])
    runs(page, live)
    chip = run_row(page, "served/17j-bf16", HLE).locator("[data-stage]")
    assert chip.inner_text() == page.locator(
        f"[data-queue-row='{here[0]}'] [data-stage]").inner_text().split(" ")[0] == "running"
    assert run_row(page, "served/17j-k8", MMLU).locator("[data-stage]").inner_text() == \
        page.locator(f"[data-queue-row='{here[1]}'] [data-stage]").inner_text() == "done"
    gone = run_row(page, "served/17j-gone", HLE)
    assert gone.locator("[data-stage]").inner_text() == "no contact for 3 h"
    assert page.evaluate("state.boxes.runs.find(r => r.hf_id === 'served/17j-gone').status") \
        == "unreached"
    # "no contact" is no run running: the status filter "active" keeps it,
    # and none of done, failed or canceled shows it
    for status in ("done", "failed", "canceled"):
        page.get_by_label("status filter").click()
        page.locator(f"[role=listbox][aria-label='status filter'] [data-value='{status}']").click()
        page.wait_for_function(f"state.qStatus === '{status}'")
        assert gone.count() == 0, status
    line = page.locator("[data-rented-line]").inner_text()
    assert line.startswith("Rented boxes read ") and " · next in about 3 min" in line, line
    assert "frontier_fetch" not in line
    assert db.get(here[0])["status"] == "running"
    assert page.errors == []


def test_35_at_400_px_the_table_fits(live, page):
    world()
    runs(page, live, width=400)
    table = page.locator("[data-queue-table]")
    wrap = page.locator("[data-all-runs]")
    assert table.bounding_box()["width"] <= wrap.bounding_box()["width"] + 1, (
        table.bounding_box(), wrap.bounding_box())             # 0abb757: 873 in 334
    hle = run_row(page, "served/17j-bf16", HLE)
    for cell in ("[data-stage]", "[data-rented-count]"):
        b = hle.locator(cell).bounding_box()
        assert b and b["x"] + b["width"] <= 400, (cell, b)
    assert page.evaluate("document.documentElement.scrollWidth") <= 400
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(wrap, SCREENS / "runs-400.png")
    assert page.errors == []


def test_where_rented_still_lists_each_rented_run(live, page):
    world()
    runs(page, live)
    where(page, "rented")
    assert page.locator("[data-rented-run]").count() == 4
    where(page, "here")
    page.wait_for_function("!document.querySelector('[data-rented-run]')")
    assert page.errors == []
