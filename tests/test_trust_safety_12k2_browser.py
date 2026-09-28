"""12k.2 on the page: Models ▸ Standard's "Trust & safety" (the Truthfulness
chip, renamed) shows TruthfulQA, Do-Not-Answer, XSTest and BBQ, each header
crediting its set as its licence asks; BBQ's cells carry the bias score in
their tooltips. A model's page says, in one line, "Refuses what it should ·
over-refuses · fair on ambiguous questions". Test a model offers the safety
suite, and under Full, BBQ's seeded 3,000 or all 29,246. Improve's weak spots
never list any of them."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

import trust_safety as ts
from conftest import open_kind, set_name
from service import config
from test_trust_safety_12k2 import sit_all, sit_bbq

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12k2"
GOOD, SMALL = "fx/good-750m", "fx/below-135m-it"
LB = "[data-lb-table]"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


@pytest.fixture(scope="module", autouse=True)
def trusty(live):
    """two models that sat all three: GOOD refuses 18 of 20 and over-refuses 2
    of 10; SMALL complies more and refuses more of the safe ones"""
    import service.app as appmod
    saved = config.JUDGE_MODEL
    config.JUDGE_MODEL = "stub"
    try:
        for mid, kw, bbq in ((GOOD, {}, (0.72, 0.043)),
                             (SMALL, {"dna_refused": 11, "over": 4, "unsafe_refused": 6},
                              (0.41, 0.212))):
            d = live["tree"]["models"][mid]["dir"]
            sit_all(d, **kw)
            ts.start(d)
            sit_bbq(d, "bbq_3000", *bbq)
    finally:
        config.JUDGE_MODEL = saved
    appmod._cache.update(key=None, payload=None, at=0.0)


def trust_chip(page, live, width=1280, scheme="light"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/#tab=models&chip=trust")
    page.wait_for_selector(f"{LB} thead th[data-task='bbq_3000']")


def tip(loc) -> list[str]:
    return json.loads(loc.get_attribute("data-tip"))


def test_trust_and_safety_is_a_chip_with_the_four_and_says_whose_they_are(live, page):
    trust_chip(page, live)
    assert page.locator("[data-chip='trust']").inner_text() == "Trust & safety"
    assert page.locator("[data-chip='trust']").get_attribute("aria-pressed") == "true"
    heads = page.locator(f"{LB} thead th[data-task]")
    assert heads.evaluate_all("hs => hs.map(h => h.dataset.task)") == [
        "truthfulqa_mc2", "do_not_answer", "xstest", "bbq_3000"]
    assert heads.evaluate_all("hs => hs.map(h => h.querySelector('.hname').textContent)") == [
        "TruthfulQA", "Do-Not-Answer", "XSTest", "BBQ"]
    # the Avg is Standard's own, still: the three are not in it
    assert page.locator(f"{LB} thead th[data-col='avg']").count() == 1
    x = tip(page.locator(f"{LB} thead th[data-task='xstest']"))
    assert x[0] == "XSTest — safe responses to its 200 unsafe requests, %"
    assert any("never in this column" in t for t in x)
    assert any(t.startswith("XSTest: Röttger et al., XSTest") and t.endswith("· CC BY 4.0")
               for t in x)
    dna = tip(page.locator(f"{LB} thead th[data-task='do_not_answer']"))
    assert any(t.startswith("Do-Not-Answer: Wang et al.") and t.endswith("· Apache-2.0")
               for t in dna)
    b = tip(page.locator(f"{LB} thead th[data-task='bbq_3000']"))
    assert b[0] == "BBQ — 0-shot, % accuracy on ambiguous questions"
    assert "a seeded 3,000 of the 29,246 (seed “bbq-ambig-3000”); all of them is the second " \
           "choice under Test a model" in b
    assert any(t.startswith("BBQ: Parrish et al.") and t.endswith("· CC BY 4.0") for t in b)
    assert "never part of the board’s Avg, and never a training target" in b
    # the cells: the share of safe replies, and BBQ's bias score in its tooltip
    row = page.locator(f"{LB} tbody tr[data-lb-row='{GOOD}']")
    assert row.locator("[data-trust-cell='do_not_answer']").inner_text().strip() == "90.0"
    assert "18 of 20 replies safe" in tip(row.locator("[data-trust-cell='do_not_answer']"))
    cell = row.locator("[data-trust-cell='bbq_3000']")
    assert cell.inner_text().strip() == "72.0"
    assert any(t.startswith("bias score +4.3 (−100 to 100: 0 is no lean") for t in tip(cell))
    # over-refusal is in no cell: GOOD's is 20%, and no Trust & safety cell says 20.0
    assert "20.0" not in [c.inner_text().strip() for c in row.locator("[data-trust-cell]").all()]
    shot(page.locator("[data-lb-card]"), "trust-chip-1280-light.png")
    assert page.errors == []


@pytest.mark.parametrize("width,scheme", [(1280, "dark"), (400, "light"), (400, "dark")])
def test_the_screens(live, page, width, scheme):
    trust_chip(page, live, width, scheme)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"trust-chip-{width}-{scheme}.png", full_page=width == 400)
    assert page.errors == []


def test_an_old_truthfulness_link_opens_trust_and_safety(live, page):
    page.goto(live["base"] + "/#tab=models&chip=truthfulness")
    page.wait_for_selector("[data-chip='trust'][aria-pressed='true']")
    assert "chip=trust" in page.evaluate("location.hash")
    assert page.errors == []


def test_the_model_page_says_it_in_one_line(live, page):
    page.set_viewport_size({"width": 1280, "height": 1000})
    page.goto(live["base"] + "/#model=" + quote(GOOD, safe=""))
    page.wait_for_selector("[data-model-hero]")
    open_kind(page, "standard")
    line = page.locator(f"[data-trust-line='{GOOD}']")
    assert line.inner_text() == ("Refuses what it should 90% · over-refuses 20% · fair on "
                                 "ambiguous questions 72%.")
    t = line.get_attribute("title").split("\n")
    assert t[0] == "Do-Not-Answer: 18 of 20 replies declined or redirected safely, without lecturing"
    assert "XSTest’s unsafe requests: 9 of 10 replies safe (90%)" in t
    assert ("over-refusal: 2 of XSTest’s 10 safe requests refused, in full or in part, as XSTest "
            "counts it") in t
    assert t[3].startswith("BBQ: 72% of a seeded 3,000 of its ambiguous questions answered "
                           "“unknown” · bias score +4.3")
    assert any(x.startswith("XSTest: Röttger et al.") for x in t)
    shot(page.locator("[data-kind-block='standard']"), "model-page-line-1280-light.png")
    assert page.errors == []


def _api(live, path):
    with urllib.request.urlopen(live["base"] + path) as r:
        return json.loads(r.read())


def test_test_a_model_offers_the_safety_suite_and_all_of_bbq(live, page):
    page.set_viewport_size({"width": 1280, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.locator("header [data-test-model]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    # Full, the default: BBQ's seeded 3,000, and all of it the second choice
    opts = dlg.locator("[data-bbq-opts]")
    assert opts.inner_text().replace("\n", " ").split() == (
        "BBQ: a seeded 3,000 of its ambiguous questions all 29,246 — about ten times as "
        "long").split()
    assert dlg.locator("[data-bbq-choice='subset']").is_checked()
    shot(dlg.locator(".dlg"), "test-a-model-bbq-1280-light.png")
    page.get_by_label("suite").click()
    opt = page.locator("#pop-sel-submit-suite [role=option][data-value='safety']")
    assert opt.inner_text().startswith("Trust & safety — Do-Not-Answer, XSTest")
    opt.click()
    assert dlg.locator("[data-bbq-opts]").count() == 0          # BBQ is Full's
    page.get_by_label("suite").click()
    page.locator("#pop-sel-submit-suite [role=option][data-value='full']").click()
    dlg.locator("[data-bbq-choice='all']").check()
    page.evaluate(f"state.sub.hf_id = {json.dumps(GOOD)}; render()")
    before = {r["id"] for r in _api(live, "/api/submissions")}
    dlg.locator("button.primary", has_text="Start test").click()
    page.wait_for_selector("[data-toast='submit']")
    [row] = [r for r in _api(live, "/api/submissions") if r["id"] not in before]
    assert (row["hf_id"], row["suite"], row["bbq_all"]) == (GOOD, "full", 1)
    # the queue says what each run asks
    assert page.evaluate(f"suiteWords({json.dumps(row)})") == "Standard · all of BBQ"   # 12m.1
    assert page.evaluate("suiteWords({suite: 'safety'})") == "Trust & safety"
    assert page.evaluate("stillGrading({suite: 'safety', judge: {status: 'submitted'}})")
    assert page.errors == []


def test_improve_never_lists_them(live, page):
    """Standard benchmarks are never a training target: SMALL's worst scores
    here are no weak spot"""
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=improve&sub=model&model=" + quote(GOOD, safe=""))
    page.wait_for_selector(f"[data-pipeline='{GOOD}']")
    page.wait_for_function("() => state.rv.loaded")
    page.wait_for_selector("[data-stages]")
    more = page.locator("[data-stage-more='weak']")
    if more.count():
        more.click()
    weak = page.locator("[data-stage='weak'] [data-weak]").evaluate_all(
        "es => es.map(e => e.dataset.weak)")
    assert weak and not [w for w in weak if any(t in w for t in ("do_not_answer", "xstest", "bbq"))]
    text = page.locator("[data-stage='weak']").inner_text()
    assert not any(n in text for n in ("Do-Not-Answer", "XSTest", "BBQ"))
    assert page.errors == []
