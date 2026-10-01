"""14.1 on the page: the Mobile tasks chip's new columns — Dolly (F1),
CNN/DailyMail and XSum (ROUGE-L), MT-Bench (out of 10, or "awaiting judge")
— each tooltip with its limits, credits and judge; the model page's line and
its Judge now; Test a model's two parts, each with what it takes; MT-Bench's
questions, a row a turn; its panel out of 10. At 1400 and 375 px. Answers
written in, the fake judge — no model runs, nothing calls OpenRouter."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

import mobileaibench as mab
from conftest import set_name
from test_12o3 import sit_mab
from test_14_1_mab_text import sit

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase14_1"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
WIDTHS = [1400, 375]


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """good-750m: every set, MT-Bench rated; skewed-360m: MT-Bench answered,
    awaiting the judge"""
    import service.app as appmod
    tree = live["tree"]
    good, skewed = tree["models"][GOOD]["dir"], tree["models"][SKEWED]["dir"]
    sit_mab(good)
    for d, words in ((good, 120), (skewed, 30)):
        sit(d, mab.MTB1, answer=lambda q, w=words: "word " * w)
        sit(d, mab.MTB2, docs=mab.load(mab.MTB2), answer=lambda q, w=words: "word " * w)
    now = {"id": "judge-test-1", "version": mab._judge_now()["version"]}
    mab._record(good, {(it["id"], it["turn"]): float(4 + it["turn"]) for it in mab.mt_answers(good)},
                now)
    for d in (good, skewed):
        mab.write(d, mab.mark(d))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


@pytest.mark.parametrize("width", WIDTHS)
def test_the_chip_has_every_set_mt_bench_out_of_10_or_awaiting(live, page, width):
    go(page, live, "tab=models&chip=mobile", "[data-lb-table]", width)
    cols = page.locator("[data-lb-table] thead th[data-col]").evaluate_all(
        "xs => xs.map(x => x.dataset.col)")
    assert [c for c in cols if c.startswith("mab_")] == ["mab_hotpotqa", "mab_sql", "mab_dolly",
                                                         "mab_cnndm", "mab_xsum", "mab_mtbench"]
    # MT-Bench: the judge's mean, out of 10 — and awaiting the judge where it waits
    good = page.locator(f"tr[data-lb-row='{GOOD}'] [data-mab-mtbench='{GOOD}']")
    assert good.inner_text().startswith("5.50")
    wait = page.locator(f"tr[data-lb-row='{SKEWED}'] [data-mab-awaiting='{SKEWED}']")
    assert wait.inner_text() == "awaiting judge"
    assert wait.get_attribute("title").startswith("160 of 160 turns wait for the judge")
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"chip-{width}.png")
    assert page.errors == []


def test_each_columns_tooltip_says_its_limits_credits_and_judge(live, page):
    go(page, live, "tab=models&chip=mobile", "[data-lb-table]")
    tip = lambda t: json.loads(page.locator(f"th[data-col='{t}']").get_attribute("data-tip"))  # noqa: E731
    cnn = tip("mab_cnndm")
    assert cnn[0] == "CNN/DailyMail (MobileAIBench) — ROUGE-L, as MobileAIBench scores it"
    assert cnn[1].startswith("word overlap with one reference summary; a good summary in other "
                             "words scores low")
    assert any("Apache-2.0" in x for x in cnn)
    assert "XSum" in tip("mab_xsum")[0] and any("not stated" in x for x in tip("mab_xsum"))
    dolly = tip("mab_dolly")
    assert "scored by overlap with one human answer" in dolly[1]
    assert any("CC BY-SA 3.0" in x for x in dolly)
    mt = tip("mab_mtbench")
    assert mt[0] == "MT-Bench (MobileAIBench) — score out of 10, as MobileAIBench scores it"
    assert "it needs a judge" in mt[1] and "their paper's by GPT-4" in mt[1]
    assert any("FastChat" in x for x in mt)
    assert any("reported (paper), never ranked with ours" in x for x in mt)
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_model_page_line_rated_or_awaiting(live, page, width):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    line = page.locator(f"[data-mab-line='{GOOD}']")
    line.wait_for()
    assert line.inner_text().startswith("HotpotQA F1 1.00 · SQL")
    assert "CNN/DailyMail ROUGE-L 1.00 · XSum ROUGE-L 1.00" in line.inner_text()
    assert page.locator(f"[data-mab-mtbench-line='{GOOD}']").inner_text().startswith(
        "MT-Bench 5.50 of 10 (turn 1 5.00, turn 2 6.00) · judged by judge-test-1 · their "
        "paper's by GPT-4 — judge differs")
    no_sideways(page)
    shot(line, f"model-line-{width}.png")
    # the other waits, and says so
    go(page, live, "model=" + quote(SKEWED, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    w = page.locator(f"[data-mab-mtbench-line='{SKEWED}']")
    w.wait_for()
    assert w.inner_text().startswith("MT-Bench: 160 of 160 turns awaiting judge")
    assert w.locator(f"[data-mab-judge='{SKEWED}']").inner_text() == "Judge now"
    no_sideways(page)
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_test_a_model_offers_two_parts_each_with_what_it_takes(live, page, width):
    posted = []
    page.route("**/api/submissions", lambda r: (posted.append(json.loads(r.request.post_data))
                                                or r.fulfill(json={"id": 990, "status": "queued"}))
               if r.request.method == "POST" else r.continue_())
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    opts = page.locator("[data-mab-opts]")
    opts.wait_for()
    page.wait_for_function("document.querySelector(\"[data-mab-part-est='none']\")"
                           "?.textContent.includes('answers')")
    assert page.locator("[data-mab-part-est='none']").inner_text().strip() == \
        "· 5,000 answers, about 83 min, a rough guess"
    assert page.locator("[data-mab-part-est='judged']").inner_text().strip().startswith(
        "· 160 answers")
    j = page.locator("[data-mab-judge-est]").inner_text()
    assert j.startswith("The judge (") and "160 judgements · about " in j and "judge tokens" in j
    assert j.endswith("each answer guessed at 400 tokens")
    no_sideways(page)
    shot(page.locator("[data-dialog='test'] .dlg"), f"test-parts-{width}.png")
    page.locator("[data-mab-part='judged'] input").check()
    page.locator("[data-dialog='test'] button.primary", has_text="Start test").click()
    page.wait_for_function("document.querySelector('[data-toast]')")
    assert posted and posted[-1]["suite"] == "mobile" and posted[-1]["part"] == "judged"
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_mt_benchs_questions_a_row_a_turn_and_its_panel_out_of_10(live, page, width):
    go(page, live, f"tab=benchmarks&q=mab_mtbench&models={ids(GOOD, SKEWED)}", "[data-qx-q]",
       width)
    first = page.locator("[data-qx-q]").first
    assert first.locator(f"[data-qx-res='{GOOD}'] .qx-verdict").inner_text().strip().startswith(
        "· rated ")
    assert first.locator(f"[data-qx-res='{SKEWED}'] .qx-verdict").inner_text().strip() == \
        "· awaiting judge"
    src = page.locator("[data-qx-source]").inner_text()
    assert "MT-Bench" in src and "FastChat" in src and "Apache-2.0" in src
    no_sideways(page)
    shot(page.locator("[data-qx-q]").first, f"mtbench-question-{width}.png")
    go(page, live, f"tab=benchmarks&sub=standard&models={ids(GOOD)}", "[data-panel='mab_mtbench']",
       width)
    panel = page.locator("[data-panel='mab_mtbench']")
    assert "5.50" in panel.inner_text() and "%" not in panel.locator("svg").text_content()
    no_sideways(page)
    assert page.errors == []


def test_judge_now_rates_what_waits_and_the_line_fills(live, page, monkeypatch):
    """last in this file: it rates skewed-360m's turns, which the tests above
    find awaiting"""
    import judge as _judge
    monkeypatch.setattr(_judge, "is_stub", lambda: True)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(SKEWED, safe=""), "[data-model-hero]")
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    page.locator(f"[data-mab-judge='{SKEWED}']").click()
    page.wait_for_function(
        f"document.querySelector(\"[data-mab-mtbench-line='{SKEWED}']\")?.textContent"
        ".startsWith('MT-Bench 2.00 of 10')")
    assert page.evaluate(f"DATA.cells.mab_mtbench['{SKEWED}'].v") == pytest.approx(0.2)
    assert page.errors == []
