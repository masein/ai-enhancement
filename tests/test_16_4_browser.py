"""16.4 on the page: Benchmarks is a catalogue — "what is this test?".

- **Sections** Standard · Mobile · Everyday · Frontier (and the Knowledge
  exam's tools while it has them).
- **A card a benchmark:** its line, its questions, how an answer is marked,
  who made it (a link), its licence as its dataset's card states it ("not
  stated" where it states none), its run time here where measured, how many
  models have a score, and whether it counts in the Avg.
- **See scores ▸** opens Models on its group; **Read the questions ▸** where
  the viewer may, never for a hidden set.
- **Suites** (DeviceMark, MobileAIBench, Mobile-MMLU) are one card that opens
  to their parts; DeviceMark's states its protocol.
- **Manage questions ▸** reaches Everyday's and the exam's tools, which lead
  back to the catalogue.
- **The ranked bars are Models ▸ Chart:** an old link to Benchmarks ▸
  Standard lands there, its models and highlights kept.

At 1400 and 375 px. Nothing runs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from test_14_3_browser import no_sideways, steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16_4"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
WIDTHS = [1400, 375]


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


def go(page, live, hash_="tab=benchmarks", width=1400, sel="[data-catalog]"):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def fact(card, k):
    return card.locator(f"[data-cat-fact='{k}']")


@pytest.mark.parametrize("width", WIDTHS)
def test_the_sections_and_a_card_a_benchmark(live, page, width):
    go(page, live, width=width)
    page.wait_for_selector("[data-cat-section='agent']")          # 18: Agent tasks, once asked
    assert page.locator("[data-cat-section]").evaluate_all("xs => xs.map(x => x.dataset.catSection)") \
        == ["standard", "mobile", "everyday", "frontier", "agent", "exam"]
    assert page.locator("[data-cat-nav] [data-cat-jump]").all_inner_texts() == \
        ["Standard", "Mobile", "Everyday", "Frontier", "Agent tasks", "Knowledge exam"]
    std = page.locator("[data-cat-section='standard'] > .catgrid > [data-cat-card]")
    assert std.evaluate_all("xs => xs.map(x => x.dataset.catCard)")[:4] == \
        ["mmlu", "hellaswag", "winogrande", "piqa"]
    # no scores on Benchmarks: no table of models, no bars
    assert page.locator("[data-lb-table], [data-panel]").count() == 0
    no_sideways(page)
    shot(page.locator("[data-cat-section='standard']"), f"standard-{width}.png")
    assert page.errors == []


def test_a_card_says_what_the_test_is_and_who_made_it(live, page):
    go(page, live)
    mmlu = page.locator("[data-cat-card='mmlu']")
    assert mmlu.locator("h3").inner_text() == "MMLU"
    assert mmlu.locator(".catline").inner_text().startswith("Four-choice questions across 57")
    n = page.evaluate("Math.max(...Object.values(DATA.cells.mmlu).map(c => c.n))")
    assert fact(mmlu, "questions").inner_text() == f"{n:,}"
    assert fact(mmlu, "marked").inner_text().startswith("options scored")
    by = fact(mmlu, "by").locator("a")
    assert by.inner_text().startswith("Hendrycks et al.")
    assert by.get_attribute("href") == "https://huggingface.co/datasets/cais/mmlu"
    assert fact(mmlu, "licence").inner_text() == "MIT"
    # its run time here, where runs measured it, with how many on hover
    t = page.evaluate("DATA.taskTime.mmlu")
    assert t and fact(mmlu, "time").inner_text().startswith("about ")
    assert f"median of {t['n']} run" in fact(mmlu, "time").get_attribute("title")
    nm = page.evaluate("Object.keys(DATA.cells.mmlu).length")
    assert fact(mmlu, "models").inner_text() == f"{nm} models"
    assert fact(mmlu, "avg").inner_text() == (
        "counts in the Avg" if page.evaluate("DATA.required.includes('mmlu')") else "never in the Avg")
    # a dataset whose card states no licence says so — never one from memory
    assert fact(page.locator("[data-cat-card='hellaswag']"), "licence").inner_text() == \
        "not stated on its dataset card"
    # one nothing has run: said, and no time
    gsm = page.locator("[data-cat-card='mmlu_pro']")
    assert fact(gsm, "questions").inner_text() == "not run here yet"
    assert fact(gsm, "time").count() == 0
    assert page.errors == []


def test_see_scores_opens_models_on_its_group(live, page):
    go(page, live)
    page.locator("[data-cat-card='hellaswag'] [data-cat-see]").click()
    page.wait_for_selector("#pill-group[data-value='commonsense']")
    assert page.evaluate("location.hash") == "#tab=models&group=commonsense"
    go(page, live)
    page.locator("[data-cat-card='mobileaibench'] > .catacts [data-cat-see]").click()
    page.wait_for_selector("#pill-group[data-value='mobileaibench']")
    assert page.errors == []


def test_read_the_questions_where_the_viewer_may_never_a_hidden_set(live, page):
    go(page, live)
    page.locator("[data-cat-card='mmlu'] [data-cat-read='mmlu']").click()
    page.wait_for_selector("[data-qx-q]")
    assert page.evaluate("location.hash").startswith("#tab=benchmarks&q=mmlu")
    go(page, live)
    # GPQA's questions are never shown
    assert page.locator("[data-cat-card='gpqa'] [data-cat-read]").count() == 0
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_a_suite_is_one_card_that_opens_to_its_parts(live, page, width):
    go(page, live, width=width)
    mab = page.locator("[data-cat-card='mobileaibench']")
    assert mab.locator(":scope > .catfacts [data-cat-fact='by'] a").get_attribute("href") == \
        "https://github.com/SalesforceAIResearch/MobileAIBench"
    assert mab.locator(":scope > .catacts [data-cat-read]").count() == 0     # read from its parts
    mab.locator("[data-cat-parts] > summary").click()
    parts = mab.locator("[data-cat-parts] [data-cat-card]")
    assert parts.count() == 9
    hot = mab.locator("[data-cat-card='mab_hotpotqa']")
    assert fact(hot, "licence").inner_text() == "CC BY-SA 4.0"
    assert fact(hot, "by").locator("a").get_attribute("href") == \
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa"
    # DeviceMark's card states its protocol
    dm = page.locator("[data-cat-card='devicemark'] [data-cat-protocol]")
    assert dm.inner_text().startswith("Its protocol: 0-shot through the chat template, greedy, a cap "
                                      "of 4,096 generated tokens")
    assert "thinking off unless a row says thinking, scoring v2" in dm.inner_text()
    no_sideways(page)
    shot(page.locator("[data-cat-section='mobile']"), f"mobile-{width}.png")
    assert page.errors == []


def test_manage_questions_reaches_the_tools_and_they_lead_back(live, page):
    go(page, live)
    page.locator("[data-cat-card='everyday'] [data-cat-manage]").click()
    page.wait_for_selector("[data-everyday-table]")
    back = page.locator("[data-cat-back]")
    assert back.inner_text() == "← Benchmarks" and back.get_attribute("href") == "#tab=benchmarks"
    back.click()
    page.locator("[data-cat-card='exam'] [data-cat-manage]").click()
    page.wait_for_selector("[data-panel='rubrics']")
    assert page.evaluate("location.hash").startswith("#tab=benchmarks&sub=exam")
    # Benchmarks in the header opens the catalogue, whichever tool was open
    page.locator("#tabs [role=tab][data-tab='benchmarks']").click()
    page.wait_for_selector("[data-catalog]")
    assert page.evaluate("location.hash") == "#tab=benchmarks"
    assert page.errors == []


@pytest.mark.parametrize("old", ["tab=tasks", "tab=benchmarks&sub=standard"])
def test_an_old_link_to_the_bars_lands_on_models_chart(live, page, old):
    ids = ",".join(quote(m, safe="") for m in (GOOD, SKEWED))
    go(page, live, f"{old}&models={ids}&hl={quote(GOOD, safe='')}", sel="[data-lb-chart]")
    assert page.evaluate("location.hash") == f"#tab=models&show=chart&models={ids}&hl={quote(GOOD, safe='')}"
    # the highlighted model in colour in every panel, with Highlight ▾ on Row 2
    assert page.locator(f"[data-panel='mmlu'] path.bar[data-model='{GOOD}']").get_attribute(
        "data-hl") == "0"
    assert page.locator("[data-pickers] #pill-highlight").inner_text() == "Highlight: 1 ▾"
    assert page.errors == []
