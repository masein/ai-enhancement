"""11l §1 on the page: a topic whose answers never finished shows a dash and
says why; an answer card shows the answer and keeps the reasoning one click
away; the checks bar names the model; How this was graded says what the
answers were generated with."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import judge as jd
from test_thinking_models_11l import FINISHED, UNFINISHED, rewrite_answers

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11l"
MODEL = "fx/good-750m"
VOID, DONE = "exam_economics", "exam_sociology"


@pytest.fixture(scope="module")
def reasoned(live):
    """good-750m as a reasoning model: Economics never left its reasoning
    (graded before the split, then re-checked, like run #60), and Sociology
    finished its reasoning and answered (graded after it)."""
    import service.app as appmod
    mdir = live["tree"]["models"][MODEL]["dir"]
    rewrite_answers(mdir, VOID, lambda i: UNFINISHED)
    assert jd.revalidate(mdir)
    rewrite_answers(mdir, DONE, lambda i: FINISHED)
    for f in mdir.glob(f"{DONE}_*shot/**/results_*.json"):
        blob = json.loads(f.read_text(encoding="utf-8"))
        blob["configs"][DONE]["generation_kwargs"] = {"until": ["\n\n\n"], "max_gen_toks": 2048}
        blob["config"]["gen_kwargs"] = "max_gen_toks=2048"
        f.write_text(json.dumps(blob), encoding="utf-8")
    jd.write_judge(mdir, jd.merge_judged(mdir, jd.run_stub(mdir, only=[DONE])))
    appmod._cache["at"] = 0.0
    appmod._cache["key"] = None
    return mdir


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def open_model(page, base):
    page.goto(base + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("table[data-judged-topics]")


def test_a_topic_that_never_answered_shows_a_dash_and_says_why(live, page, reasoned):
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_model(page, live["base"])
    row = page.locator("table[data-judged-topics] tr[data-topic='Economics']")
    row.wait_for()
    score = row.locator("[data-topic-score='Economics']")
    assert score.text_content().startswith("—")
    n = row.locator("[data-no-answer]").get_attribute("data-no-answer")
    assert row.locator("[data-no-answer]").text_content() == f"{n} answers, {n} no answer"
    # it stands on no hidden question: a dash, not "0 · under 30"
    assert row.locator("td").nth(3).text_content() == "—"
    assert "never finished answering" in row.text_content()
    assert "not scored" in row.text_content()
    note = page.locator("[data-not-scored]")
    assert note.get_attribute("data-not-scored") == "1"
    assert "The model never finished answering" in note.text_content()
    # and a topic that finished is scored as usual
    soc = page.locator("table[data-judged-topics] tr[data-topic='Sociology']")
    assert " / 4" in soc.locator("[data-topic-score='Sociology']").text_content()
    row.scroll_into_view_if_needed()
    shot(page, "11l-1-not-scored-1400-light.png")
    assert page.errors == []


def choose_model_on_topic(page, base, slug):
    page.goto(base + f"/#topic={slug}")
    page.wait_for_selector("[data-panel='answers']")
    page.evaluate("m => { state.ans.model = m; state.ans.rows = null; render(); }", MODEL)
    page.wait_for_selector("[data-panel='answers'] [data-answer]")


def test_an_answer_card_shows_the_answer_and_the_reasoning_behind_a_click(live, page, reasoned):
    page.set_viewport_size({"width": 1400, "height": 1000})
    choose_model_on_topic(page, live["base"], "sociology")
    card = page.locator("[data-panel='answers'] [data-answer]").first
    assert card.locator("[data-answer-text]").text_content() == "The answer is 6."
    assert "<think>" not in card.text_content()
    why = card.locator("[data-reasoning]")
    assert why.evaluate("e => e.open") is False
    assert "The model's reasoning (7 words)" in why.locator("summary").text_content()
    why.locator("summary").click()
    assert "Six units." in why.text_content()
    shot(page, "11l-2-answer-and-reasoning-1400-light.png")
    # a topic whose answers never finished: no answer, no score, the reasoning
    # still there to read
    choose_model_on_topic(page, live["base"], "economics")
    card = page.locator("[data-panel='answers'] [data-answer]").first
    assert card.get_attribute("data-no-answer") == "1"
    assert card.locator("[data-answer-text]").text_content().startswith("No answer:")
    assert "no answer — not scored" in card.text_content()
    assert "it never finished" in card.locator("[data-reasoning] summary").text_content()
    assert "not scored" in page.locator("[data-report-half]").text_content()
    card.scroll_into_view_if_needed()
    page.evaluate("window.scrollBy(0, -160)")
    shot(page, "11l-3-no-answer-card-1400-light.png")
    assert page.errors == []


def test_the_checks_bar_names_the_model(live, page, reasoned):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/")
    pill = page.locator("#warnings summary[data-warn-summary]")
    pill.wait_for()
    pill.click()
    item = page.get_by_text("Answers that never finished: good-750m", exact=False).first
    item.wait_for()
    shot(page, "11l-4-checks-1400-light.png")
    page.locator("[data-show-me='judge_unfinished']").click()
    page.wait_for_selector("table[data-judged-topics]")
    assert page.evaluate("state.model") == MODEL
    assert page.errors == []


def test_how_this_was_graded_says_what_the_answers_were_generated_with(live, page, reasoned):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + f"/#model={MODEL.replace('/', '%2F')}&read=provenance:judge:{MODEL}")
    page.wait_for_selector("#reader[data-ready='1']")
    line = page.locator("#reader [data-answer-budget]")
    assert "2,048" in line.text_content()
    assert "more than the usual 256, because this model reasons before it answers" \
        in line.text_content()
    assert "answers never finished: they were not scored" in line.text_content()
    shot(page, "11l-5-how-graded-1400-light.png")
    assert page.errors == []


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, browser, reasoned, theme):
    for width in (1400, 400):
        ctx = browser.new_context(viewport={"width": width, "height": 1000},
                                  reduced_motion="reduce")
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            open_model(page, live["base"])
            page.evaluate(f"applyTheme('{theme}')")
            page.locator("[data-not-scored]").scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            shot(page, f"11l-model-{width}-{theme}.png")
            assert errors == []
        finally:
            ctx.close()
