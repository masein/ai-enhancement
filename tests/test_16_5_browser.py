"""16.5 on the page: KNOWLEDGE_EXAM=0, and the Knowledge exam is nowhere —
Models' Row 1, Benchmarks, Home, a model's page (its kinds, answers and
History), Test a model, Improve (with what it hides, counted), AI models and
the question builder; its old addresses land elsewhere. At 1400 and 375 px.
The fixture's judged results are on disk throughout; nothing runs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from service import config, db
from test_14_3_browser import no_sideways, steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16_5"
GOOD = "fx/good-750m"
SAID = ("Knowledge exam", "the written exam", "Sit the exam", "judged topics")


@pytest.fixture(scope="module", autouse=True)
def exam_off(live):
    import service.app as appmod
    pe = db.proposal_create(GOOD, "exam_law", "Law", "masein", {})
    db.dataset_create(pe, "doc", 10, "masein", {})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "KNOWLEDGE_EXAM", False)
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def said(page) -> list[str]:
    """what the view says of the exam — but the lines that say what it hides,
    which the brief asks for"""
    text = page.evaluate("""() => { const v = document.querySelector('#view').cloneNode(true);
      v.querySelectorAll('[data-exam-hidden], [data-jt-exam-hidden]').forEach(e => e.remove());
      document.body.append(v); const t = v.innerText; v.remove(); return t; }""")
    return [w for w in SAID if w.lower() in text.lower()]


@pytest.mark.parametrize("width", [1400, 375])
def test_no_place_shows_it(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.wait_for_function("DATA !== null")
    assert page.evaluate("DATA.examOn") is False
    places = [("tab=home", "[data-best-by-kind]"), ("tab=models", "[data-lb-table]"),
              ("tab=benchmarks", "[data-catalog]"),
              ("model=" + quote(GOOD, safe=""), "[data-model-hero]"),
              ("tab=improve", "[data-pipeline]"), ("tab=ai", "[data-ai-page] [data-ai-jobs]"),
              ("tab=build", "[data-qb-what]")]
    for h, sel in places:
        go(page, live, h, sel, width)
        page.wait_for_timeout(300)
        assert said(page) == [], h
        no_sideways(page)
    # Models: Row 1 without it
    go(page, live, "tab=models", "[data-lb-table]", width)
    assert "exam" not in page.locator("[data-models-view]").evaluate_all(
        "xs => xs.map(x => x.dataset.modelsView)")
    shot(page.locator("[data-lb-card] .lbrow1"), f"models-row1-{width}.png")
    # a model's page: no exam tile, and no exam row in How it was graded
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    assert page.locator("[data-kind-tile='exam'], [data-kind-block='exam']").count() == 0
    assert page.errors == []


def test_its_old_addresses_land_elsewhere(live, page):
    go(page, live, "tab=models&view=exam", "[data-lb-table]")
    assert page.evaluate("location.hash") == "#tab=models"
    go(page, live, "tab=benchmarks&sub=exam", "[data-catalog]")
    assert page.evaluate("location.hash") == "#tab=benchmarks"
    assert page.locator("[data-cat-section='exam']").count() == 0
    page.goto(live["base"] + "/#topic=law")
    page.wait_for_selector("#view > *")
    assert page.locator("[data-topic-back]").count() == 0
    assert page.errors == []


def test_test_a_model_offers_no_judged_suite(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=submit")
    page.wait_for_selector("[data-dialog='test'] [data-submit-form]")
    page.get_by_label("suite").click()
    opts = page.locator("[role=listbox][aria-label='suite'] [role=option]")
    opts.first.wait_for()
    assert "judged" not in [o.get_attribute("data-value") for o in opts.all()]
    assert page.errors == []


def test_improve_says_what_it_hides_and_works_from_everyday(live, page):
    go(page, live, "tab=improve", "[data-pipeline]")
    line = page.locator("[data-exam-hidden]").first
    assert line.inner_text() == ("1 proposal and 1 dataset built from Knowledge exam results are "
                                 "hidden while the exam is switched off — kept, not deleted.")
    # no exam topic to propose from; Everyday's groups are Improve's now
    assert page.locator("[data-imp-propose], [data-weak-kind='exam']").count() == 0
    shot(page.locator("[data-pipeline]").first, "improve-1400.png")
    assert page.errors == []


def test_the_builder_offers_everyday_alone(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "tab=build", "[data-qb-what]")
    assert page.locator("[data-qb-kind]").evaluate_all("xs => xs.map(x => x.dataset.qbKind)") == \
        ["everyday"]
    assert page.locator("[data-qb-what]").get_attribute("data-qb-what") == "everyday"
    assert page.errors == []
