"""12n.2 on the page: GPQA Diamond and SimpleQA Verified measured here, beside
what Epoch AI reports — three GPQA methods, three cells, never ranked or
averaged together or against the reported number; SimpleQA's share correct
with not attempted apart; the Frontier view's columns, Compare's "Shared with
the frontier", the Benchmarks panels with their reference ticks, the model
page's line, and Test a model's new suite. And no GPQA question anywhere.

Invented GPQA-shaped results (tests/fixtures/gpqa), SimpleQA's pinned
questions answered by hand and graded by the stand-in judge, and Epoch's
trimmed file — its Qwen3-1.7B aliased to a board model, for the calibration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import pytest

import simpleqa as sq
from service import reported
from test_12n2 import leaks, sit_gpqa, sit_simpleqa, twenty
from test_reported_12m2 import epoch_fetch, epoch_zip

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12n2"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
GPT55 = "reported/openai/gpt-5.5"
CAL = ("Same questions, different prompt and settings. A large gap means our method differs, "
       "not the model.")


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def tip(loc) -> list[str]:
    return json.loads(loc.get_attribute("data-tip"))


def graded(model_dir: Path) -> None:
    """SimpleQA's answers graded as the stand-in judge grades them"""
    qs = {q["id"]: q for q in sq.load()}
    out = sq.mark(model_dir)
    sq.write(model_dir, sq.mark(model_dir, {it["id"]: sq.stub_grade(qs[it["id"]], it["answer_text"])
                                            for it in out["items"] if it["grade"] is None}))


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    tree = live["tree"]
    good, skewed = tree["models"][GOOD]["dir"], tree["models"][SKEWED]["dir"]
    sit_gpqa(good, acc=0.36)
    sit_gpqa(good, "gpqa_diamond_zeroshot", 0.30)
    sit_gpqa(skewed, acc=0.28)
    sit_simpleqa(good, twenty())
    graded(good)
    sit_simpleqa(skewed, {k: "I'm not sure." for k in twenty()})
    graded(skewed)
    # Epoch's Qwen3-1.7B, as the GPQA board model: the calibration (before the
    # import, which keeps a model it can place on the board)
    reported.alias_set("alibaba/qwen3-1.7b", GOOD, "masein")
    reported.import_epoch(fetch=epoch_fetch(epoch_zip()))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def go(page, live, hash_, sel, width=1400, scheme="light"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def test_the_frontier_view_has_both_with_our_cells_tagged_and_the_calibration(live, page):
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    cols = page.locator("[data-fr-col]").evaluate_all("xs => xs.map(x => x.dataset.frCol)")
    assert "gpqa diamond" in cols and "simpleqa verified" in cols
    # the model in both: ours and Epoch's, side by side, with the one-line tooltip
    cal = page.locator(f"[data-fr-cal='{GOOD}'][data-fr-cell='gpqa diamond']")
    assert cal.inner_text().replace("\n", "") == "measured here 36.0 · Epoch 38.0"
    assert tip(cal) == [CAL]
    # ours alone: tagged with how, chain of thought first
    ours = page.locator(f"[data-fr-here='{SKEWED}'][data-fr-cell='gpqa diamond']")
    assert ours.locator(".fr-tag").inner_text() == "measured here · CoT, 0-shot"
    sqa = page.locator(f"[data-fr-here='{GOOD}'][data-fr-cell='simpleqa verified']")
    assert sqa.locator(".fr-tag").inner_text() == \
        "measured here · graded by the judge, the dataset’s grader"
    # ranked within one setting: our two CoT cells with each other, Epoch's with Epoch's
    here = page.locator("[data-fr-cell='gpqa diamond'] [data-fr-set^='here|'], "
                        "[data-fr-cell='gpqa diamond'][data-fr-set^='here|']")
    lead = [here.nth(i).get_attribute("data-fr-v") for i in range(here.count())
            if here.nth(i).get_attribute("data-fr-lead")]
    assert "0.36" in lead                        # the best of ours (the other is within its noise)
    gpt = page.locator(f"[data-fr-rep='{GPT55}'][data-fr-cell='gpqa diamond']")
    assert gpt.get_attribute("data-fr-set") == "epoch|Epoch AI's own run"
    shot(page.locator("[data-frontier]"), "frontier-gpqa-1400.png")
    assert page.errors == []


def test_compare_puts_ours_and_epochs_on_one_line_never_ranked_together(live, page):
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, SKEWED, GPT55)}", "[data-compare='3']")
    g = page.locator("[data-cmp-group='shared']")
    assert g.inner_text().startswith("▾ Shared with the frontier")
    row = page.locator("[data-cmp-row='fr:gpqa_diamond_cot_zeroshot']")
    assert row.locator("[data-cmp-main]").inner_text() == "CoT, 0-shot"
    # Epoch's number in the same row: grey, its own tag, never the best
    gpt = row.locator(f"[data-cmp-cell='{GPT55}']")
    assert "cmp-off" in gpt.get_attribute("class") and gpt.get_attribute("data-best") is None
    assert "reported by Epoch AI · Epoch AI's own run" in gpt.inner_text()
    # ours ranked with ours; the model Epoch ran too shows its number beside
    assert row.locator(f"[data-cmp-cell='{GOOD}']").get_attribute("data-best") == "1"
    assert row.locator(f"[data-cmp-cal='{GOOD}']").inner_text() == "Epoch 38.0"
    # the three GPQA methods: rows of their own, each its own tag
    assert page.locator("[data-cmp-main='fr:gpqa_diamond_zeroshot']").inner_text() == \
        "4 options scored, 0-shot"
    # SimpleQA: the share correct, and not attempted apart — never "best"
    assert page.locator("[data-cmp-row='fr:sq-na'] [data-best]").count() == 0
    assert page.locator("[data-cmp-row='fr:simpleqa_verified'] [data-best]").count() == 1
    # Epoch's GPQA is here, not in the Reported group too
    assert page.locator("[data-cmp-row='rep:epoch:GPQA diamond']").count() == 0
    shot(page.locator("[data-compare]"), "compare-shared-1400.png")
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, SKEWED, GPT55)}", "[data-compare='3']",
       width=400, scheme="dark")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-compare]"), "compare-shared-400-dark.png")
    assert page.errors == []


def test_benchmarks_panels_with_epochs_ticks(live, page):
    go(page, live, "tab=benchmarks&sub=standard", "[data-panel='gpqa_diamond_cot_zeroshot']")
    cot = page.locator("[data-panel='gpqa_diamond_cot_zeroshot']")
    labels = cot.locator("g.reftick text").all_text_contents()
    assert labels and all(re.fullmatch(r".+ \d+\.\d · Epoch", x) for x in labels), labels
    assert cot.locator("path.bar[data-model^='reported/']").count() == 0
    assert page.locator("[data-panel='simpleqa_verified'] path.bar").count() == 2
    assert page.locator("[data-panel='gpqa_diamond_zeroshot']").count() == 1
    shot(cot, "gpqa-cot-panel.png")
    assert page.errors == []


def test_gpqas_forms_are_never_averaged_together(live, page):
    go(page, live, f"tab=models&cols=gpqa_diamond_cot_zeroshot,gpqa_diamond_zeroshot&models="
       f"{ids(GOOD)}", "[data-lb-table]")
    assert page.locator("[data-gpqa-mix]").inner_text() == (
        " · no Avg: GPQA Diamond measured 2 ways is one benchmark, never averaged with itself")
    assert page.locator(f"[data-cavg='{GOOD}']").inner_text() == "—"
    # one form with another benchmark averages as any two columns do
    go(page, live, f"tab=models&cols=gpqa_diamond_cot_zeroshot,simpleqa_verified&models="
       f"{ids(GOOD)}", "[data-lb-table]")
    assert page.locator("[data-gpqa-mix]").count() == 0
    assert page.locator(f"[data-cavg='{GOOD}']").inner_text() != "—"
    assert page.errors == []


def test_the_model_page_line_and_simpleqas_cell(live, page):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    page.locator("[data-kind-block='standard'] > summary").first.click() if page.locator(
        "[data-kind-block='standard']:not([open])").count() else None
    line = page.locator(f"[data-shared-line='{GOOD}']")
    line.wait_for()
    assert line.locator("p").first.inner_text() == (
        "GPQA Diamond 36.0 (CoT) · GPQA Diamond 30.0 (4 options scored) · "
        "SimpleQA Verified 30.0 · not attempted 30%")
    assert page.locator(f"[data-shared-cal='{GOOD}']").inner_text().startswith(
        "Epoch AI reports GPQA diamond 38.0 — same questions, different prompt and settings")
    assert "honest, not wrong" in line.inner_text()
    shot(line, "model-page-line.png")
    go(page, live, f"tab=models&cols=simpleqa_verified&models={ids(GOOD, SKEWED)}",
       "[data-lb-table]")
    cell = page.locator(f"tr[data-lb-row='{SKEWED}'] [data-sq-na]")
    assert cell.inner_text() == "100.0% not attempted"
    assert page.errors == []


def test_test_a_model_offers_the_suite(live, page):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    opt = page.locator("[role='option'][data-value='shared']")
    opt.wait_for()
    assert opt.inner_text().startswith("Shared with the frontier — GPQA Diamond (CoT), SimpleQA")
    assert page.errors == []


def test_no_gpqa_question_is_on_any_page(live, page):
    for h, sel in (("tab=models&chip=frontier", "[data-frontier-table]"),
                   (f"tab=models&view=compare&m={ids(GOOD, SKEWED)}", "[data-compare]"),
                   ("tab=benchmarks&sub=standard", "[data-panel]"),
                   ("model=" + quote(GOOD, safe=""), "[data-model-hero]")):
        go(page, live, h, sel)
        assert leaks(page.content()) == [], h
        assert leaks(json.dumps(page.evaluate("DATA"))) == [], h
    assert page.errors == []
