"""12m.2 on the page: models known only as reported are in Models ▾ under
"Reported (not run here)", by maker — 12n.1: folded, and chosen, a row of
the Frontier view's, never of Standard's; Compare gives each source its own
group, credited; Benchmarks draws a reported number as a tick on the panel
of the same benchmark; and AI models ▸ Outside data says what each source
holds."""

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


def shot(part, name):
    # 17h: taken again when the page's poll redraws the part mid-shot
    # (test_14_3_browser.steady_shot) — the AI models page answers later since
    # 17h.1, and its outside-data card was caught being replaced
    from test_14_3_browser import steady_shot
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


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
    # 12n.1: Reported (not run here) is folded until opened, a maker a group
    page.locator("#pop-models [data-rep-fold]").click()
    page.wait_for_selector("#pop-models [data-model-group='reported · OpenAI']")
    head = page.locator("#pop-models [data-model-group='reported · OpenAI']").inner_text()
    assert head.strip().startswith("OpenAI all · none · only these")
    # not among "all", which is the board's
    assert page.locator(f"#pop-models [data-model-pick='{F55}']").is_checked() is False
    page.locator("[data-models-clear]").click()
    for mid in (GOOD, F55):
        page.locator(f"#pop-models [data-model-pick='{mid}']").check()
    page.keyboard.press("Escape")
    # 12n.1: never a row of Standard's; its row is Frontier's, credited in its header
    page.wait_for_selector("[data-rep-hidden='1']")
    assert page.locator(f"tr[data-lb-row='{F55}']").count() == 0
    page.locator("[data-rep-hidden-go]").click()
    page.locator("[data-frontier-all]").click()
    cell = page.locator(f"[data-fr-rep='{F55}'][data-fr-cell='mmlu-pro']")
    cell.wait_for()
    assert cell.inner_text() == "87.3"
    t = tip(cell)
    assert "reported by Artificial Analysis · Artificial Analysis's own run" in t[1]
    assert "Data: Artificial Analysis" in page.locator("[data-frontier-credit]").inner_text()
    # the model has no average, so no rank
    assert page.evaluate(f"officialAvg(anyModel({json.dumps(F55)}))") is None
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


def test_a_reported_number_is_a_tick_on_its_benchmark_credited(live, page):
    """12n.1: a reference tick on MMLU's panel, not a panel of bars beside it"""
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=benchmarks&sub=standard&models={quote(GOOD, safe='')},"
              + quote(F55, safe=""))
    page.wait_for_selector(f"[data-panel='mmlu'] g.reftick[data-ref='{F55}']")
    assert page.locator("[data-panel='rep:card:mmlu']").count() == 0
    t = page.locator(f"[data-panel='mmlu'] g.reftick[data-ref='{F55}']")
    assert t.locator("text").text_content() == "Frontier Test 5.5 91.2 · card"
    assert "as the model card or paper reports it" in tip(t.locator("rect.hit"))
    assert page.errors == []


def test_outside_data_says_what_each_source_holds(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=ai")
    set_name(page, "masein")
    card = page.locator("[data-outside]")
    card.wait_for()
    # it draws at once and fills in when api/reported has answered
    page.wait_for_selector("[data-outside] [data-rep-lines='1']")
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
