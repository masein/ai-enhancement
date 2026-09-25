"""12a.2 on the page: Everyday tasks round 2 — the bank by the seven groups,
each question readable with its checks in plain words; the results as models
across the top and groups down the side, n of k, each cell opening that
model's answers in that group; the model page's block by group; Models'
Everyday view with a column per group and the total, the badge once. 12a.3:
the bank is 333 questions. 12a.4: the badge says "not ranked", and only the
models that answered this wording are here — the two that sat only the pilot
answered an earlier one. 12a.5: 388 questions in eight groups."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import open_kind

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a2"
GROUPS = ["Understanding", "Writing", "Shorten a message", "Summarise", "Transform",
          "Quick maths", "Instructions", "Honesty"]
KEYS = ["understanding", "writing", "shorten", "summarising", "transform", "quick_maths",
        "instructions", "honesty"]


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def everyday_page(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-table]")


def test_the_results_are_groups_by_models_n_of_k(live, page):
    everyday_page(page, live["base"])
    rows = page.locator("[data-everyday-table] tbody tr")
    assert [r.get_attribute("data-evd-g") for r in rows.all()] == KEYS
    assert page.locator("[data-everyday-table] .evq-short").all_inner_texts() == GROUPS
    models = [th.get_attribute("data-evd-model")
              for th in page.locator("[data-everyday-table] th[data-evd-model]").all()]
    assert models == ["fx/good-750m", "fx/skewed-360m"]
    # 12g.2: the hidden half's score
    assert page.locator("[data-evd-count='fx/good-750m']").inner_text() == "199 of 200"
    # k is the group's hidden half
    assert page.locator("[data-evd-cell='fx/good-750m|quick_maths']").inner_text() == "23 of 24"
    # 12a.4: one that sat the pilot only answered an earlier wording: not here
    assert page.locator("[data-evd-count='fx/chance-160m']").count() == 0
    # one badge, and the one action
    assert page.locator("[data-pilot-badge]").inner_text() == "not ranked"
    assert page.locator("[data-everyday-run]").inner_text() == "Run everyday tasks"
    assert page.errors == []


def test_a_cell_opens_that_models_answers_in_that_group(live, page):
    everyday_page(page, live["base"])
    page.locator("[data-evd-cell='fx/skewed-360m|writing']").click()
    panel = page.locator("[data-evd-panel='fx/skewed-360m|writing']")
    panel.wait_for()
    assert "skewed-360m · Writing" in panel.inner_text()
    rows = panel.locator("[data-evd-row]")
    want = page.evaluate("""() => DATA.everyday.questions.filter(q => q.group === 'writing')
      .map(q => q.id)""")
    assert [r.get_attribute("data-evd-row") for r in rows.all()] == want      # one row a question
    marks = [r.locator("[data-evd-mark]").get_attribute("data-evd-mark") for r in rows.all()]
    assert marks.count("ok") == 12 and marks.count("no") == 12     # 12g.2: its 24 practice
    # each row says why, and opens the whole answer in the side panel
    no = rows.nth(marks.index("no"))
    assert no.locator(".evreason").inner_text()
    no.click()
    page.wait_for_selector("#reader[data-ready='1'] [data-evd-answer]")
    page.keyboard.press("Escape")
    # another cell is another model and group; the same cell closes it
    page.locator("[data-evd-cell='fx/good-750m|quick_maths']").click()
    page.wait_for_selector("[data-evd-panel='fx/good-750m|quick_maths']")
    assert page.locator("[data-evd-panel]").count() == 1
    # 12g.2: its one miss here is a hidden question — counted, never a row
    assert page.locator("[data-evd-row='everyday-maths-01']").count() == 0
    assert page.locator("[data-evd-split-note='quick_maths']").inner_text().endswith(
        "The 24 hidden ones score it and are not shown.")
    page.locator("[data-evd-cell='fx/good-750m|quick_maths']").click()
    assert page.locator("[data-evd-panel]").count() == 0
    assert page.errors == []


def test_the_bank_is_readable_by_group_with_its_checks_in_plain_words(live, page):
    everyday_page(page, live["base"])
    groups = page.locator("[data-everyday-bank] [data-evd-bank-group]")
    assert [g.get_attribute("data-evd-bank-group") for g in groups.all()] == KEYS
    page.locator("[data-evd-bank-group='transform'] > summary").click()
    # 12g.2: a practice question (the pilot's JSON one is hidden now)
    q = page.locator("[data-evd-bank-q='everyday-transform-07']")
    assert q.locator(".evq").inner_text().startswith("convert this bank sms into json")
    assert q.locator("[data-evd-checks]").inner_text() == \
        "Passes if it: valid JSON with 4091, 84.50, freshmart"
    # every practice question is there, every one readable; the hidden are counted
    assert page.locator("[data-evd-bank-q]").count() == 188
    assert page.locator("[data-evd-bank-q='everyday-pilot-02']").count() == 0
    # the page's cards are not steps: no section numbers
    assert page.locator("#view h2[data-ix]").count() == 0
    assert page.errors == []


def test_the_model_page_block_is_the_eight_groups(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    open_kind(page, "everyday")
    block = page.locator("[data-everyday-block='fx/good-750m']")
    assert [b.get_attribute("data-evd-group") for b in block.locator("[data-evd-group]").all()] == KEYS
    # 12g.2: the hidden half's count, and the practice half's answers
    assert block.locator("[data-evd-group-count='honesty']").inner_text() == "24 of 24"
    block.locator("[data-evd-group='honesty']").click()
    rows = page.locator("[data-evd-answers='fx/good-750m|honesty'] [data-evd-row]")
    rows.first.wait_for()
    assert rows.count() == 27
    assert page.locator("[data-kind-tile='everyday'] [data-kind-value='everyday']").inner_text() == \
        "199 of 200"
    # 12a.4: a model that sat the pilot answered an earlier wording — no
    # block, and its header says where the answers are
    page.goto(live["base"] + "/#model=fx%2Fbelow-135m-it")
    page.wait_for_selector("[data-model-hero]")
    assert page.locator("[data-kind-block='everyday']").count() == 0
    assert page.locator("[data-evd-earlier-note='fx/below-135m-it']").is_visible()
    assert page.errors == []


def test_models_everyday_is_a_column_per_group_and_the_total(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=models&view=everyday")
    t = page.locator("[data-lb-everyday]")
    t.wait_for()
    heads = t.locator("thead th").all_inner_texts()
    # the headers are set in capitals by the stylesheet
    assert [h.lower() for h in heads[1:-1]] == [g.lower() for g in GROUPS]
    assert heads[-1].lower() == "total"
    row = t.locator("tr[data-lb-row='fx/good-750m']")
    assert row.locator("[data-evd-g='writing']").inner_text() == "24 of 24"      # 12g.2
    assert row.locator("[data-everyday-count]").inner_text() == "199 of 200"
    assert page.locator("[data-lb-card] [data-pilot-badge]").count() == 1
    assert page.errors == []


def test_run_everyday_tasks_and_the_suite_say_388(live, page):
    everyday_page(page, live["base"])
    page.locator("[data-everyday-run]").click()
    dlg = page.locator("[data-dialog='everyday']")
    dlg.wait_for()
    assert dlg.locator("h2").inner_text() == "Run everyday tasks"
    # 12a.5: a run asks what the model has no answer to — all 388 the first time
    assert "all 388 the first time" in dlg.inner_text()
    page.keyboard.press("Escape")
    page.locator("header [data-test-model]").click()
    page.get_by_label("suite").click()
    opt = page.locator("#pop-sel-submit-suite [role=option][data-value='everyday']")
    assert opt.inner_text() == "Everyday tasks — 388 questions, a few minutes"
    page.keyboard.press("Escape")
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1400])
def test_the_screens(live, page, width):
    everyday_page(page, live["base"], width)
    page.locator("[data-evd-cell='fx/good-750m|quick_maths']").click()
    page.wait_for_selector("[data-evd-panel]")
    page.wait_for_timeout(200)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12a2-everyday-{width}-light.png", full_page=True)
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    open_kind(page, "everyday")
    page.locator("[data-evd-group='writing']").click()
    page.wait_for_timeout(200)
    shot(page, f"12a2-model-{width}-light.png", full_page=True)
    assert page.errors == []
