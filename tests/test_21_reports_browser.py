"""21: the Reports place in a real browser, at 1280 and 400 px: the list and
the report both open, at their own links, and Back returns to the list; the
picker changes the tiles and the head-to-head table; the table's header and
our reference row stay visible while it scrolls; nothing scrolls sideways.
The world: our six rows' scores as the board stores them (invented), Epoch
AI's data as it was on 14 Oct (a trimmed copy, CC BY 4.0), the card numbers
entered through the board's card entry, and a few stored answers for the
token stats and the loop examples. No model runs; nothing is fetched."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase21"
EPOCH = Path(__file__).resolve().parent / "fixtures" / "epoch-2026-10-14"
TASKS = ["gpqa_diamond_epoch", "hle_text_cais", "otis_aime_epoch", "math_l5_epoch",
         "mmlupro_tiger", "simpleqa_epoch", "arc_agi2_public"]
# our rows' scores and errors (invented), as the board would store them; a
# missing one is a benchmark not in yet
OURS = {
    "served/step4000-lda-2610": [(.616, .031), (.112, .007), (.533, .05), (.912, .008),
                                 (.781, .004), (.052, .007), (.03, .009)],
    "served/step1000-lda-2610": [(.648, .030), None, (.580, .05), (.918, .008), None,
                                 (.061, .008), (.04, .01)],
    "served/Qwen3.6-35B-A3B-Q4-lda-2610": [(.851, .022), (.205, .009), (.86, .04), (.95, .006),
                                           (.842, .003), (.093, .009), (.05, .012)],
    "served/Qwen3.6-35B-A3B-BF16-lda-2610": [(.862, .021), (.214, .009), (.88, .04),
                                             (.955, .006), (.851, .003), (.097, .009),
                                             (.06, .013)],
    "served/Qwen3.6-35B-A3B-Q2-lda-2610": [(.781, .026), (.162, .008), (.74, .05), (.93, .007),
                                           (.80, .004), (.075, .008), (.03, .009)],
    "served/Qwen3.6-35B-A3B-Q4-original-k-8": [(.833, .024), (.199, .009), (.84, .04),
                                               (.947, .006), (.838, .003), (.09, .009),
                                               (.04, .011)],
}
ANSWERS = {"gpqa_diamond_epoch": 792, "hle_text_cais": 2158, "otis_aime_epoch": 360,
           "math_l5_epoch": 1324, "mmlupro_tiger": 12032, "simpleqa_epoch": 1000,
           "arc_agi2_public": 334}


def think(n_think: int, rest: str = "ANSWER: E") -> str:
    lines = [f"Wait, let me check option E against the question again ({i})." for i in range(6)]
    body = "\n".join(lines * max(1, n_think // 400))
    return f"<think>\n{body}\n</think>\n{rest}"


def results(row: Path, mid: str, task: str, v: float, se: float, ran_out: int) -> None:
    d = row / f"{task}_0shot" / "frontier"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_2026-10-14T07-20-00.000000.json").write_text(json.dumps({
        "results": {task: {"alias": task, "acc,none": v, "acc_stderr,none": se}},
        "config": {"model": "frontier", "model_args": f"pretrained={mid},enable_thinking=True"},
        "date": time.time(),
        "frontier": {"version": "frontier-2", "answers": ANSWERS[task], "ran_out": ran_out,
                     "scored_by": "code", "where": "rented GPU · RTX 5090"}}))


def samples(row: Path, task: str, runs: list[tuple[str, str, int, bool, bool]]) -> None:
    d = row / f"{task}_0shot" / "frontier"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{task}_2026-10-14T07-20-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": qid}, "resps": [[text]],
                    "filtered_resps": ["E" if not out else ""],
                    "frontier": [{"epoch": 0, "tokens": tok, "ran_out": out, "ok": ok,
                                  "read": None if out else "E"}]}) + "\n"
        for k, (qid, text, tok, out, ok) in enumerate(runs)))


def world() -> None:
    from service import config, db, reported
    import enter_report_cards as erc
    for mid, cells in OURS.items():
        row = config.OUT_DIR / (mid.replace("/", "__") + "__thinking")
        row.mkdir(parents=True, exist_ok=True)
        db.served_put({"id": mid, "name": mid.split("/")[1], "base_url": "", "key": "",
                       "how": "rented GPU", "based_on": "Qwen/Qwen3.6-35B-A3B",
                       "thinking": "on", "pin": {}, "by": "masein", "at": 0})
        for task, c in zip(TASKS, cells):
            if c:
                results(row, mid, task, c[0], c[1], ran_out=12 if task != "arc_agi2_public" else 3)
    # a few stored answers: GPQA and MMLU-Pro for step4000 and Q4, the loops among them
    for mid, scale in (("served/step4000-lda-2610", 3), ("served/Qwen3.6-35B-A3B-Q4-lda-2610", 1)):
        row = config.OUT_DIR / (mid.replace("/", "__") + "__thinking")
        samples(row, "gpqa_diamond_epoch", [(f"g{i}", think(900 * scale), 3000 * scale + i,
                                              False, i % 3 != 0) for i in range(12)])
        samples(row, "mmlupro_tiger", [(q, think(4000 * scale if scale > 1 else 900,
                                                 "" if scale > 1 else "ANSWER: E"),
                                         32768 if scale > 1 else 2500 + i, scale > 1, scale == 1)
                                        for i, q in enumerate(["486", "10707", "11648", "3", "7"])])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(EPOCH.glob("*.csv")):
            z.writestr(f.name, f.read_text(encoding="utf-8"))
    blob = buf.getvalue()
    reported.import_epoch(fetch=lambda url, headers=None, timeout=60: blob, board=[])
    erc.main(["step-checkpoints", "--by", "masein"])


def open_list(page, live, width: int) -> None:
    page.set_viewport_size({"width": width, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports")
    page.wait_for_selector("[data-reports-list] [data-report-card='step-checkpoints']")


def no_sideways(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


@pytest.mark.parametrize("width", [1280, 400])
def test_the_list_and_the_report_open_and_the_picker_moves_the_tiles(live, page, width):
    world()
    open_list(page, live, width)
    card = page.locator("[data-report-card='step-checkpoints']")
    assert "Step checkpoints vs popular models" in card.inner_text()
    assert "step4000" in card.locator("[data-report-models]").inner_text()
    assert card.locator("[data-report-changed]").inner_text().startswith("Numbers last changed ")
    assert no_sideways(page)
    card.click()
    page.wait_for_selector("[data-report='step-checkpoints']")
    assert page.evaluate("location.hash") == "#tab=reports&report=step-checkpoints"
    # 1: our runs, each with its status, a dash where a benchmark isn't in
    ours = page.locator("[data-rp-ours]")
    assert ours.locator("[data-rp-status='step4000']").inner_text() == "Complete"
    assert "61.6 ± 3.1" in ours.locator("[data-rp-row='step4000']").inner_text()
    assert ours.locator("[data-rp-row='step1000'] td.dash").count() == 2
    # a note under a name wraps inside its column, never cut at its edge
    note = ours.locator("[data-rp-row='Q4, earlier run'] td").first
    assert note.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
    # sources, each with its own time; Epoch's credit
    fresh = page.locator("[data-report-fresh]").inner_text()
    assert "Our scores: as of the last import, " in fresh and "Epoch AI: imported " in fresh
    assert page.locator("[data-credit='epoch']").inner_text() == "Data: Epoch AI, CC BY 4.0"
    # 2: the picker changes the summary, the tiles and the head-to-head table
    line = page.locator("[data-rp-summary]").inner_text()
    assert line.startswith("step4000 is higher than ") and "of the " in line
    tile = page.locator("[data-rp-tile='gpqa'] .rp-big").inner_text()
    first = page.locator("[data-rp-ref] td").nth(1).inner_text()
    page.locator("[data-rp-pick='BF16']").click()
    page.wait_for_function("document.querySelector('[data-rp-summary]').textContent"
                           ".startsWith('BF16 is higher than')")
    assert page.locator("[data-rp-tile='gpqa'] .rp-big").inner_text() != tile
    assert page.locator("[data-rp-ref] td").first.inner_text() == "BF16"
    assert page.locator("[data-rp-ref] td").nth(1).inner_text() != first
    assert page.locator("[data-rp-h2h-row='step4000']").count() == 1      # ours among the others
    # the header and our reference row stay visible while the table scrolls
    box = page.locator("[data-rp-h2h-scroll]")
    box.scroll_into_view_if_needed()
    box.evaluate("e => e.scrollTop = 600")
    page.wait_for_timeout(100)
    top = box.evaluate("e => e.getBoundingClientRect().top")
    th = page.locator("[data-rp-h2h] thead th").first.evaluate("e => e.getBoundingClientRect().top")
    ref = page.locator("[data-rp-ref] td").first.evaluate("e => e.getBoundingClientRect().top")
    assert abs(th - top) <= 2 and 0 < ref - top <= 40, (top, th, ref)
    assert box.evaluate("e => e.scrollTop") > 0
    cell = page.locator("[data-rp-h2h-row='GPT-4.1'] td").nth(1)
    assert "BF16 " in cell.get_attribute("title") and "GPT-4.1 66.9" in cell.get_attribute("title")
    # 4 and 5: the stats are worked out on the server, then shown
    page.wait_for_selector("[data-rp-token-line]", timeout=30000)
    assert page.locator("[data-rp-token-line]").inner_text().startswith(
        "step4000 thinks ")
    page.wait_for_selector("[data-rp-loop='486'] .rp-excerpt", timeout=30000)
    assert "“Wait”" in page.locator("[data-rp-loop='486']").inner_text()
    # 6 and 7: every model, ours in bold; the cards' links beside their numbers
    every = page.locator("[data-rp-every]")
    assert every.locator("td.rp-ours").count() == 6
    assert every.locator("td.rp-ours").first.evaluate(
        "e => +getComputedStyle(e).fontWeight") >= 600                  # ours in bold
    assert every.locator("[data-card-link]").count() >= 19
    assert page.locator("[data-rp-cards] a").count() == 6
    assert no_sideways(page)
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-rp-pick='step4000']").click()
    page.wait_for_function("document.querySelector('[data-rp-summary]').textContent"
                           ".startsWith('step4000')")
    # the report's top — its sources, our runs, where step4000 stands
    page.set_viewport_size({"width": width, "height": 2000})
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(150)
    page.screenshot(path=SCREENS / f"report-{width}.png")
    # Back returns to the list
    page.go_back()
    page.wait_for_selector("[data-reports-list]")
    assert page.evaluate("location.hash") == "#tab=reports"
    assert page.errors == []


def test_the_report_in_dark_and_the_place_in_the_bar(live, page):
    world()
    page.emulate_media(color_scheme="dark")
    open_list(page, live, 1280)
    assert page.locator("#tabs [role=tab][data-tab='reports']").get_attribute(
        "aria-selected") == "true"
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    page.wait_for_selector("[data-report='step-checkpoints']")
    page.wait_for_selector("[data-rp-tiles] [data-rp-tile='gpqa']")
    assert no_sideways(page)
    assert page.errors == []
