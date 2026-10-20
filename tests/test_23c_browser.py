"""23c in a real browser, at 1280 and 400 px, light and dark: a model page's
"Speed and energy" part in Beyond the score (its sentence, its card's speeds,
each benchmark's seconds, tokens a second, energy and dollars), the line
under the calibration chart, the scorecard's two new columns (a dash and
"measured from the next run on" for a build without), and Runs' rented list
with a slow card flagged, its numbers on hover; nothing scrolls sideways.
The world is 23b's, step4000's answers given timings and its imports card
checks, energy and a price — all invented."""

from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.parse import quote

import pytest

from test_17i_runs_browser import run_row
from test_17j_page_browser import runs as runs_page
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_23b_beyond import GPQA, HLE, hle_rows, write
from test_23b_browser import S4, S4_ROW, answers, runs
from test_23c_card import card, progress, steps

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase23c"
BUILD = "served/23c-step"


def timed(rows: list[dict], rng: random.Random, secs: tuple[float, float]) -> list[dict]:
    """each run given the seconds it took and llama-server's timings"""
    for r in rows:
        for run in r["runs"]:
            run.update(wall_s=round(rng.uniform(*secs), 2), predicted_n=4000,
                       predicted_ms=round(4000 / rng.uniform(40, 60) * 1000, 1))
    return rows


def imports() -> None:
    """step4000's imports as 23c's import keeps them: GPQA Diamond in two
    shards and HLE on one box, each with its card check, energy and price"""
    from service import reports
    d = reports.row_dir(S4_ROW)
    gpqa = {"answers": 120, "shard_bundles": [
        {"shard": 1, "box": "L13", "answers": 60, "energy_wh": 42.0, "server_s": 1800.0,
         "cost_usd": 0.25, "card": card(142.0, "L13")},
        {"shard": 2, "box": "L14", "answers": 60, "energy_wh": 48.0, "server_s": 2100.0,
         "cost_usd": 0.29, "card": card(138.0, "L14")}]}
    hle = {"answers": 261, "box": "L1", "energy_wh": 2375.1, "server_s": 30000.0,
           "cost_usd": 4.17, "card": card(145.0, "L1")}
    (d / "frontier_imports.json").write_text(json.dumps({"tasks": {GPQA: gpqa, HLE: hle}}))
    # and how they were asked: eight at once (23's setup.json)
    from service import frontier as sf
    for t in (GPQA, HLE):
        (sf.task_dir(d, t) / "setup.json").write_text(json.dumps(
            {"concurrency": {"in_flight": 8, "server_slots": 8}}))


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import report_stats
    scorecard()
    answers()
    rng = random.Random(23)
    write(S4, GPQA, timed(runs(30, 4, 18, 3), rng, (20.0, 90.0)))
    write(S4, HLE, timed(hle_rows(260, random.Random(7)), rng, (90.0, 200.0)))
    imports()
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def open_block(page, live, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + quote(S4_ROW, safe=""))
    blk = page.locator("[data-kind-block='beyond']")
    blk.wait_for(timeout=30000)
    page.wait_for_function("document.querySelector('[data-by-head]').textContent"
                           ".startsWith('calibration error ')", timeout=30000)
    blk.locator("summary").click()
    page.wait_for_selector("[data-by-part='speed']")
    return blk


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_speed_and_energy_part_draws(live, page, width, scheme):
    blk = open_block(page, live, width, scheme)
    part = page.locator("[data-by-part='speed']")
    said = part.locator("[data-by-speed-sentence]").inner_text()
    assert said.startswith("On an RTX 5090, step4000 writes 142 tokens a second for one user "
                           "and 568 in all for 8; an HLE answer takes "), said
    assert said.endswith(" minutes and 9.1 Wh.") or said.endswith(" seconds and 9.1 Wh."), said
    # the card's speeds: the median over three boxes, the range
    cardt = part.locator("[data-by-card] tbody tr")
    assert cardt.count() == 5
    assert cardt.nth(0).inner_text().split("\t")[1:] == ["142", "138–145"]
    # each benchmark: seconds, tokens a second, energy and dollars an answer
    assert page.locator(f"[data-by-wh='{GPQA}']").inner_text() == "0.75 Wh"      # 90 Wh / 120
    assert page.locator(f"[data-by-usd='{GPQA}']").inner_text() == "$0.0045"     # $0.54 / 120
    assert page.locator(f"[data-by-wh='{HLE}']").inner_text() == "9.1 Wh"
    assert part.locator("h4").nth(1).inner_text() == "Each benchmark, with 8 answers at once"
    # 23b's chart, its last group said: 260 graded answers, 160 left out
    left = page.locator("[data-by-cal-left]")
    assert left.inner_text() == "the error leaves out the 160 most confident answers, as CAIS’s does."
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.evaluate("e => window.scrollTo(0, e.getBoundingClientRect().top + scrollY - 70)")
    page.screenshot(path=str(SCREENS / f"model-speed-{width}-{scheme}.png"), full_page=True)
    part.screenshot(path=str(SCREENS / f"model-speed-part-{width}-{scheme}.png"))
    del blk


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (400, "dark")])
def test_the_scorecards_two_columns(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    sec = page.locator("[data-report-section='beyond']")
    sec.wait_for(timeout=30000)
    alone = page.locator(f"[data-by-cell='{S4}:alone']")
    assert alone.inner_text() == "142"
    assert alone.get_attribute("title") == ("output tokens a second for one user: the median "
                                            "over 3 boxes (138–145)")
    assert page.locator(f"[data-by-cell='{S4}:gpqa-wh']").inner_text() == "0.75 Wh"
    for c in ("alone", "gpqa-wh"):                    # step8804: from the next run on
        x = page.locator(f"[data-by-cell='step8804:{c}']")
        assert x.inner_text() == "—" and x.get_attribute("title") == "measured from the next run on"
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    sec.screenshot(path=str(SCREENS / f"report-columns-{width}-{scheme}.png"))


def rented() -> None:
    """four boxes of one build asking HLE, the fourth's card slow — flagged
    by the fetch's round as it posts them"""
    import time

    import frontier_fetch as ff
    import import_frontier as imf
    imf.boxes_path().unlink(missing_ok=True)
    got = steps(*[progress(lb, v, n, pace, model=BUILD) for lb, v, n, pace in (
        ("L1", 140, 600, 12.0), ("L2", 145, 600, 11.0), ("L3", 150, 600, 10.0),
        ("L4", 100, 200, 20.0))])
    ff.slow_cards(got)
    now = time.time()
    rows = [{**{k: v for k, v in x.items() if not k.startswith("_")},
             **{k: x["_p"][k] for k in ("model", "thinking", "tasks", "shard", "line", "at")},
             "started_at": "2026-10-20T06:00:00Z", "sessions": 1, "seen_at": now,
             "safe": False, "card": "card: 142 tok/s alone, 568 at 8 · 575 W limit · PCIe 5 ×16",
             "use": "312 W · 96% busy"} for x in got]
    ids = [x["box_id"] for x in got]
    imf.store_boxes([{"asked": ids, "reached": ids, "every": 180}, *rows])


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_box_list_flags_the_slow_card(live, page, width, scheme):
    rented()
    page.emulate_media(color_scheme=scheme)
    runs_page(page, live, width=width, q="23c-")
    row = run_row(page, BUILD, HLE)
    row.wait_for()
    flag = row.locator("[data-rented-slow]")
    assert flag.inner_text() == ("L4: slow card — 100 tok/s against the others' 145; replacing it "
                                 "now would save about 4 hours")
    tip = flag.get_attribute("title")
    assert tip.startswith("L4: slow card") and "the build's other boxes: 140, 145, 150 tok/s" in tip
    row.locator("[data-rented-open]").click()
    page.wait_for_selector("[data-rented-boxes]")
    box = page.locator("[data-rented-box='L4|L4-1'] [data-box-slow]")
    assert box.inner_text().startswith(" · slow card — 100 tok/s")
    assert "left at its pace about 10.9 h" in box.get_attribute("title")
    assert page.locator("[data-box-slow]").count() == 1                  # only the slow one
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("table[data-queue-table]").screenshot(
        path=str(SCREENS / f"box-list-{width}-{scheme}.png"))
