"""12i.0: polish from the live check of #71–#73.

The chosen benchmarks' average says what it is ("Avg above chance"); the
benchmark picker names each benchmark by its own name, and lists the ones
nothing has run yet, greyed; both pickers apply at once and stay inside the
page card; a table someone built numbers its rows 1, 2, 3 and says when a
chosen model has no scores; and Improve says what it reads, and names its AI
in plain words."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import pytest

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12i0"
NAMES = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "piqa": "PIQA", "winogrande": "WinoGrande",
         "arc_challenge": "ARC-Challenge", "arc_easy": "ARC-Easy", "gsm8k": "GSM8K",
         "truthfulqa_mc2": "TruthfulQA", "ifeval": "IFEval", "mmlu_pro": "MMLU-Pro",
         "hendrycks_math500": "MATH-500"}


def shot(page_or_part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page_or_part.screenshot(path=SCREENS / name, **kw)


def models_tab(page, base, rest="", width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=models" + ("&" + rest if rest else ""))
    page.wait_for_selector("[data-lb-card] [data-pickers]")


def rows(page):
    return [r.get_attribute("data-lb-row") for r in page.locator("tr[data-lb-row]").all()]


def test_the_average_says_it_is_above_chance(live, page):
    models_tab(page, live["base"], "cols=arc_easy,hellaswag")
    th = page.locator("th[data-col='cavg']")
    th.wait_for()
    assert th.locator(".hname").inner_text().lower() == "avg above chance"
    assert json.loads(th.get_attribute("data-tip")) == [
        "0 = guessing, 100 = perfect, so a 25% guess on a 4-option test counts as 0"]
    assert page.errors == []


def test_the_picker_names_each_benchmark_and_lists_the_ones_not_run_yet(live, page):
    models_tab(page, live["base"])
    page.locator("#pill-benchmarks").click()
    menu = page.locator("#pop-benchmarks")
    menu.wait_for()
    labels = {r.get_attribute("data-bench-row"): r for r in menu.locator("[data-bench-row]").all()}
    # every one the board knows, by its own name and nothing else
    assert set(labels) == set(NAMES)
    for t, name in NAMES.items():
        text = " ".join(labels[t].inner_text().split()).replace(" · not run yet", "")
        assert text == name, (t, text)
    # a benchmark nothing on this board has run is there, greyed, and can't be ticked
    ran = set(page.evaluate("DATA.accTasks"))
    for t in NAMES:
        box = menu.locator(f"[data-bench='{t}']")
        if t in ran:
            assert box.is_enabled() and menu.locator(f"[data-not-run='{t}']").count() == 0
        else:
            assert box.is_disabled(), t
            assert menu.locator(f"[data-not-run='{t}']").inner_text() == "· not run yet"
    assert set(NAMES) - ran, "the fixture should leave some benchmark not run"
    shot(menu, "12i0-benchmarks-picker-1400-light.png")
    assert page.errors == []


@pytest.mark.parametrize("width", [1024, 1400])
def test_both_pickers_stay_inside_the_page_card(live, page, width):
    models_tab(page, live["base"], width=width)
    card = page.locator("[data-lb-card]").bounding_box()
    for pill, pop in (("#pill-benchmarks", "#pop-benchmarks"), ("#pill-models", "#pop-models")):
        page.locator(pill).click()
        box = page.locator(pop).bounding_box()
        assert box["x"] >= card["x"] - 0.5, (pop, box, card)
        assert box["x"] + box["width"] <= card["x"] + card["width"] + 0.5, (pop, box, card)
        page.keyboard.press("Escape")
    assert page.errors == []


def test_the_models_picker_applies_at_once(live, page):
    models_tab(page, live["base"])
    page.locator("#pill-models").click()
    menu = page.locator("#pop-models")
    menu.wait_for()
    assert menu.locator("[data-models-apply]").count() == 0          # no Apply
    first = rows(page)[:2]
    menu.locator("[data-models-clear]").click()
    page.wait_for_selector("[data-no-models]")
    for mid in first:
        menu.locator(f"[data-model-pick='{mid}']").check()
    # while the list is still open, the table already shows the two
    page.wait_for_function("document.querySelectorAll('tr[data-lb-row]').length === 2")
    assert menu.is_visible() and sorted(rows(page)) == sorted(first)
    assert page.locator("#pill-models").inner_text() == "Models: 2 ▾"
    assert page.errors == []


def served(page, edit):
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


NEW = "fx/never-scored-1b"


def a_model_with_no_scores(body):
    """a model on the board that has sat nothing Standard yet"""
    m = dict(body["models"][0], id=NEW, name="never-scored-1b", avg=None, avgRaw=None,
             avgSe=None, avgRawSe=None)
    body["models"].append(m)


def test_a_built_table_numbers_its_rows_and_says_when_one_has_no_scores(live, page):
    served(page, a_model_with_no_scores)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("tr[data-lb-row]")
    board = [m for m in rows(page) if m != NEW]
    # three from further down the board, and one with no scores at all
    picked = board[3:6] + [NEW]
    models_tab(page, live["base"], "models=" + ",".join(quote(m, safe="") for m in picked))
    page.wait_for_function("document.querySelectorAll('tr[data-lb-row]').length === 3")
    # 1, 2, 3 — not their ranks on the whole board; the one with no scores is
    # under the table's "not tested" line, and the line above says so
    assert page.locator("tr[data-lb-row] [data-row-n]").all_inner_texts() == ["1", "2", "3"]
    assert NEW not in rows(page)
    assert page.locator("[data-custom-what]").inner_text().endswith(" · 4 chosen · 3 tested")
    shot(page.locator("[data-lb-card]"), "12i0-built-table-1400-light.png")
    assert page.errors == []


def test_improve_says_what_it_reads_and_names_its_ai_plainly(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=improve")
    page.wait_for_selector("[data-pipeline]")
    sub = page.locator("[data-pipeline] p.sub").inner_text()
    assert sub.startswith("What the Knowledge exam and Everyday tasks say this model is missing")
    ai = page.locator("[data-imp-ai]")
    ai.wait_for()
    line = ai.inner_text()
    assert re.fullmatch(r"AI: .+ · [\d,]+ requests? today( of [\d,]+.*)?", line), line
    for word in ("/", "(generator", "exam writer", "items", "no daily limit"):
        assert word not in line, word
    assert page.errors == []
