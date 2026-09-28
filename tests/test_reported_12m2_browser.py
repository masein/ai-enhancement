"""12m.2 on the page: models known only as reported are in Models ▾ under
"Reported (not run here)", by maker; chosen, a row of their own with the
"Reported · <source>" columns, never a leader, an average or a rank; Compare
gives each source its own group, credited; Benchmarks puts a reported panel
beside the lm_eval one of the same benchmark; and AI models ▸ Outside data
says what each source holds."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from service import config, reported
from test_reported_12m2 import KEY, canned, epoch_fetch, epoch_zip

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12m2"
GOOD = "fx/good-750m"
F55 = "reported/openai/frontier-test-5.5"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


@pytest.fixture(scope="module", autouse=True)
def imported(live):
    saved = config.AA_API_KEY
    config.AA_API_KEY = KEY
    try:
        reported.import_aa(fetch=canned())
    finally:
        config.AA_API_KEY = saved
    reported.import_epoch(fetch=epoch_fetch(epoch_zip()))
    # a number from a model card: MMLU, the benchmark the board runs too
    reported.card_add({"model": "Frontier Test 5.5", "maker": "OpenAI", "benchmark": "MMLU",
                       "value": "91.2%", "setting": "5-shot", "url": "https://example.org/card",
                       "date": "2026-09-20"}, "masein")


def tip(loc) -> list[str]:
    return json.loads(loc.get_attribute("data-tip"))


def test_models_menu_has_the_reported_models_by_maker_and_a_chosen_one_is_a_row(live, page):
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    page.locator("#pill-models").click()
    page.wait_for_selector("#pop-models [data-model-group='reported · OpenAI']")
    head = page.locator("#pop-models [data-model-group='reported · OpenAI']").inner_text()
    assert head.startswith("Reported (not run here) · OpenAI")
    # not among "all", which is the board's
    assert page.locator(f"#pop-models [data-model-pick='{F55}']").is_checked() is False
    page.locator("[data-models-clear]").click()
    for mid in (GOOD, F55):
        page.locator(f"#pop-models [data-model-pick='{mid}']").check()
    page.keyboard.press("Escape")
    row = page.locator(f"tr[data-lb-row='{F55}']")
    row.wait_for()
    # its numbers in their own group, credited, never a leader, no rank, no average
    cell = row.locator("[data-rep2-cell='rep2:aa:MMLU-Pro']")
    assert cell.inner_text().split("\n")[0] == "87.3"
    t = tip(cell)
    assert "reported by Artificial Analysis · Artificial Analysis's own run" in t
    assert "Data: Artificial Analysis" in t
    assert page.locator("[data-rep2-cell]").evaluate_all(
        "xs => xs.every(x => !x.dataset.lead)")
    # a built table numbers its rows; the model has no average, so no rank
    assert page.evaluate(f"officialAvg(anyModel({json.dumps(F55)}))") is None
    groups = page.locator(f"{'[data-lb-table]'} thead tr.grp th").all_inner_texts()
    assert any(g.upper().startswith("REPORTED · ARTIFICIAL ANALYSIS") for g in groups)
    shot(page.locator("[data-lb-card]"), "models-reported-row.png")
    assert page.errors == []


def test_compare_gives_each_source_its_group_credited(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&view=compare&m=" + ",".join(
        quote(m, safe="") for m in (GOOD, F55, "reported/anthropic/claude-test-5")))
    page.wait_for_selector("[data-compare='3']")
    assert page.locator(f"[data-cmp-kind='{F55}']").inner_text() == " · reported only · OpenAI"
    g = page.locator("[data-cmp-group='rep:aa']")
    assert g.inner_text().startswith("▾ Reported · Artificial Analysis")
    assert page.locator("[data-cmp-credit='rep:aa']").inner_text() == " · Data: Artificial Analysis"
    assert page.locator("[data-cmp-main='rep:aa:MMLU-Pro']").inner_text() == \
        "reported by Artificial Analysis · Artificial Analysis's own run"
    # the card's MMLU is its own group, never the lm_eval row (folded: one model has it)
    page.locator("[data-cmp-fold='standard']").click()
    assert page.locator(f"[data-cmp-row='mmlu'] [data-cmp-cell='{F55}']").inner_text() == \
        "not measured"
    page.locator("[data-cmp-fold='rep:card']").click()
    assert page.locator(f"[data-cmp-row='rep:card:MMLU'] [data-cmp-cell='{F55}']").inner_text() \
        .startswith("91.2")
    shot(page.locator("[data-compare]"), "compare-reported.png")
    assert page.errors == []


def test_a_reported_panel_sits_beside_its_benchmark_credited(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=benchmarks&sub=standard")
    page.wait_for_selector("[data-panel='rep:card:mmlu']")
    keys = page.locator("[data-panel]").evaluate_all("xs => xs.map(x => x.dataset.panel)")
    at = keys.index("mmlu")
    assert "rep:card:mmlu" in keys[at + 1:at + 3]
    panel = page.locator("[data-panel='rep:card:mmlu']")
    assert panel.locator("h3").inner_text() == "MMLU · reported by model card"
    assert "as the model card or paper reports it" in panel.locator("[data-panel-method]").inner_text()
    hit = panel.locator(f"rect.hit[data-model='{F55}']")
    assert "as the model card or paper reports it" in tip(hit)
    assert page.errors == []


def test_outside_data_says_what_each_source_holds(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=ai")
    set_name(page, "masein")
    card = page.locator("[data-outside]")
    card.wait_for()
    aa = card.locator("[data-rep-line='aa']").inner_text()
    assert aa.startswith("Artificial Analysis: 39 scores for 13 models, imported ")
    assert "· Data: Artificial Analysis" in aa
    ep = card.locator("[data-rep-line='epoch']").inner_text()
    assert ep.startswith("Epoch AI: ") and ep.endswith("· Data: Epoch AI, CC BY 4.0")
    assert "internal use" in card.inner_text()
    # a number from a card, refused in one line, then added
    card.locator("[data-rep-card-form] > summary").click()
    card.locator("[data-rep-in='model']").fill("Claude Test 5")
    card.locator("[data-rep-in='benchmark']").fill("MMLU-Pro")
    card.locator("[data-rep-in='value']").fill("85%")
    card.locator("[data-action='rep-card']").click()
    page.wait_for_selector("[data-action-error='rep-card']")
    assert "Their setting" in card.locator("[data-action-error='rep-card']").inner_text()
    shot(card, "outside-data.png")
    # the refusal's own 422, as the browser logs it, and nothing else
    assert [e for e in page.errors if "status of 422" not in e] == []


def test_epochs_numbers_are_their_own_group_credited(live, page):
    ids = ("reported/openai/gpt-5.5", "reported/anthropic/claude-opus-5")
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&view=compare&m=" + ",".join(quote(m, safe="") for m in ids))
    page.wait_for_selector("[data-compare='2']")
    assert page.locator("[data-cmp-credit='rep:epoch']").inner_text() == " · Data: Epoch AI, CC BY 4.0"
    assert page.locator("[data-cmp-main='rep:epoch:GPQA diamond']").inner_text() == \
        "reported by Epoch AI · Epoch AI's own run"
    assert page.locator(f"[data-cmp-kind='{ids[0]}']").inner_text() == " · reported only · OpenAI"
    names = page.locator("[data-cmp-chip]").all_inner_texts()
    assert names[0].startswith("GPT-5.5 (xhigh)")
    # two models, one method: the Δ, z-tested on their own standard errors
    d = page.locator("[data-cmp-delta='rep:epoch:GPQA diamond']").inner_text()
    assert d.endswith("· clear") or d.endswith("· not a clear difference"), d
    shot(page.locator("[data-compare]"), "compare-epoch.png")
    assert page.errors == []
