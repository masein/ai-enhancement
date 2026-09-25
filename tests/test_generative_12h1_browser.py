"""12h.1 on the page: Models ▸ Standard has an Instruction & maths group —
IFEval, MMLU-Pro and MATH-500 — where a thinking run is a row of its own, a
base model is "instruct only" once in the Not tested list and never an empty
cell, and each cell says what else there is to know: answers that ran out of
room, IFEval's instruction-level score, a subset, a score far below what the
model's makers publish. The model page says how they were asked; the submit
form offers the suite, a thinking switch where the model has one, and a
seeded MMLU-Pro subset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generative_fixture import write_run

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12h1"
MODEL = "fx/below-135m-it"          # the fixture's instruct model
THINK = MODEL + " · thinking"


@pytest.fixture(scope="module", autouse=True)
def generative_runs(live):
    """the fixture's instruct model sat the three, thinking off and on"""
    import service.app as appmod
    out = live["tree"]["out_dir"]
    write_run(out, MODEL, thinking=False)
    write_run(out, MODEL, thinking=True)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def served(page, edit):
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def instruction_view(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    page.locator("[data-chip='instruction']").click()
    page.wait_for_selector("th[data-col='ifeval']")


def test_instruction_and_maths_is_a_group_of_three(live, page):
    instruction_view(page, live["base"])
    # the headers are set in capitals by the stylesheet; no rank and no Avg here
    heads = [h.lower() for h in page.locator("[data-lb-table] thead tr.names th .hname")
             .all_inner_texts()]
    assert heads == ["model", "params", "ifeval", "mmlu-pro", "math-500"]
    # a thinking run is a row of its own, labelled; never merged with the other
    rows = [r.get_attribute("data-lb-row") for r in page.locator("tr[data-lb-row]").all()]
    assert MODEL in rows and THINK in rows
    assert page.locator(f"[data-thinking-badge='{THINK}']").inner_text() == "thinking"
    assert page.locator(f"[data-thinking-badge='{MODEL}']").count() == 0
    # the column says who scores it and how it is asked
    tip = json.loads(page.locator("th[data-col='mmlu_pro']").get_attribute("data-tip"))
    assert any(t.startswith("scored by exact match on the letter A–J") for t in tip)
    # 12h.2: never in the board's Avg; Benchmarks ▾ can average it with others
    assert ("instruct models only, and never part of the board's Avg — choose it under "
            "Benchmarks to average it with others") in tip
    assert any(t.startswith("published by the makers") and "Qwen3.5-2B 55.3 (no thinking)" in t
               for t in tip)
    assert page.errors == []


def test_each_cell_says_what_else_there_is(live, page):
    instruction_view(page, live["base"])
    row = page.locator(f"tr[data-lb-row='{MODEL}']")
    mmlu = row.locator("td[data-gen-cell='mmlu_pro']")
    assert mmlu.inner_text().startswith("76.9")                   # 10 of 13, the board's reading
    assert mmlu.locator("[data-ran-out]").inner_text() == "1 ran out of room"
    ife = row.locator("td[data-gen-cell='ifeval']")
    # instruction-level, from the harness's per-answer verdicts: 5 of 8
    assert ife.locator("[data-inst-level]").inner_text() == "62.5 by instruction"
    tip = json.loads(mmlu.get_attribute("data-tip"))
    assert any("thinking off · on vllm" in t and "1 answer ran out of room" in t for t in tip)
    assert page.errors == []


def test_a_base_model_is_instruct_only_once_and_never_an_empty_cell(live, page):
    instruction_view(page, live["base"])
    nt = page.locator("[data-not-tested]")
    nt.locator("button").click()
    base = page.locator("[data-instruct-only='fx/good-750m']")
    assert base.inner_text() == " · instruct only"
    assert page.locator("tr[data-lb-row='fx/good-750m']").count() == 0   # not a row of dashes
    assert page.locator("[data-not-tested-test='fx/good-750m']").count() == 0
    assert page.errors == []


def test_a_score_far_below_published_is_flagged_on_its_cell(live, page):
    def far(body):
        for m in body["models"]:
            if m["id"] == MODEL:
                m["gen"]["far"] = {"ifeval": {"published": 86.2, "note": "", "gap": 36.2}}
    served(page, far)
    instruction_view(page, live["base"])
    cellx = page.locator(f"tr[data-lb-row='{MODEL}'] td[data-gen-cell='ifeval']")
    assert cellx.locator("[data-far-below]").inner_text() == \
        "far below published, check extraction"
    assert any("far below published (86.2): check extraction" in t
               for t in json.loads(cellx.get_attribute("data-tip")))
    assert page.errors == []


def test_the_model_page_says_how_they_were_asked(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    from conftest import open_kind
    open_kind(page, "standard")
    line = page.locator(f"[data-gen-mode='{MODEL}']")
    assert line.inner_text() == ("IFEval, MMLU-Pro and MATH-500: thinking off · on vllm · "
                                 "answers that ran out of room: 1 on IFEval, 1 on MMLU-Pro, "
                                 "1 on MATH-500.")
    assert page.errors == []


def test_the_form_offers_the_three_a_thinking_switch_and_a_subset(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    page.wait_for_selector("header [data-test-model]")
    page.locator("header [data-test-model]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    page.get_by_label("suite").click()
    opt = page.locator("#pop-sel-submit-suite [role=option][data-value='generative']")
    # 12a.5b: slow, each chosen on its own (test_slow_benchmarks_12a5b_browser.py)
    assert opt.inner_text().startswith("Instruction & maths — slow, instruct models only")
    opt.click()
    # a model with a switch: the switch; one without: none
    page.evaluate("state.sub.hf_id = 'Qwen/Qwen3.5-2B'; render()")
    dlg.locator("[data-gen-opts]").wait_for()
    assert dlg.locator("[data-think-switch]").count() == 1
    # 12a.5b: MMLU-Pro is the seeded 1,200 unless Full is chosen
    dlg.locator("[data-gen-pick='mmlu_pro'] input").check()
    assert dlg.locator("[data-mmlu-size='subset'] input").is_checked()
    page.evaluate("state.sub.hf_id = 'LiquidAI/LFM2.5-1.2B-Instruct'; render()")
    assert dlg.locator("[data-think-switch]").count() == 0
    shot(page, "12h1-form-1400-light.png")
    page.keyboard.press("Escape")
    assert page.errors == []


def test_test_from_a_thinking_row_asks_the_model_to_think(live, page):
    def switch(body):
        for m in body["models"]:
            if m["id"] in (MODEL, THINK):
                m["archinfo"] = {**(m.get("archinfo") or {}), "thinking": "switch"}
    served(page, switch)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    page.evaluate(f"openTest({json.dumps(THINK)})")
    assert page.evaluate("[state.sub.hf_id, state.sub.suite, state.sub.thinking]") == \
        [MODEL, "generative", True]
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1400])
def test_the_screens(live, page, width):
    instruction_view(page, live["base"], width)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12h1-instruction-{width}-light.png", full_page=True)
    assert page.errors == []
