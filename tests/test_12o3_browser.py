"""12o.3 on the page: a Mobile tasks chip with HotpotQA (F1) and SQL
(SQLParser F1), their other numbers and credits on hover; a panel each on
Benchmarks with its questions; Compare's group of their own; the model page's
line; Test a model's new suite. Answers written in, scored by the ported
metrics — no model runs."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

import mobileaibench as mab
from test_12o3 import sit_mab

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12o3"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    tree = live["tree"]
    good, skewed = tree["models"][GOOD]["dir"], tree["models"][SKEWED]["dir"]
    sit_mab(good)                                   # the references: near-perfect
    sit_mab(skewed, answer=lambda q: "```sql\nSELECT 1 FROM t\n```" if q["id"].startswith(
        "sql-") else "Answer: I don't know")
    for d in (good, skewed):
        mab.write(d, mab.mark(d))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def go(page, live, hash_, sel, width=1400, scheme="light"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def test_the_mobile_tasks_chip_has_both_columns_with_their_metrics(live, page):
    go(page, live, "tab=models&chip=mobile", "[data-lb-table]")
    heads = page.locator("[data-lb-table] thead th[data-col] .hname").evaluate_all(
        "xs => xs.map(x => x.textContent)")
    assert "HotpotQA" in heads and "SQL" in heads
    tip = json.loads(page.locator("th[data-col='mab_hotpotqa']").get_attribute("data-tip"))
    assert tip[0] == "HotpotQA (MobileAIBench) — F1, as MobileAIBench scores it"
    assert "short answers score best: the reference is a few words" in tip[1]
    assert any("CC BY-SA 4.0" in x for x in tip) and any("Apache-2.0" in x for x in tip)
    sql = json.loads(page.locator("th[data-col='mab_sql']").get_attribute("data-tip"))
    assert "else its first line starting with SELECT" in sql[1]
    assert any("CC BY 4.0" in x for x in sql)
    cells = page.evaluate("[DATA.cells.mab_hotpotqa['fx/good-750m'].v, "
                          "DATA.cells.mab_sql['fx/skewed-360m'].v]")
    assert cells[0] > 0.99 and cells[1] < 0.5
    shot(page.locator("[data-lb-card]"), "models-mobile-chip.png")
    assert page.errors == []


def test_the_model_page_line_and_compares_group(live, page):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    line = page.locator(f"[data-mab-line='{GOOD}']")
    line.wait_for(state="attached")
    assert line.inner_text() == "HotpotQA F1 1.00 · SQL 1.00 (MobileAIBench’s 1,000 each)" \
        or line.inner_text().startswith("HotpotQA F1 1.00 · SQL 0.9")
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, SKEWED)}", "[data-compare='2']")
    g = page.locator("[data-cmp-group='mobile']")
    assert g.inner_text().startswith("▾ Mobile tasks (MobileAIBench)")
    row = page.locator("[data-cmp-row='mab_hotpotqa']")
    assert row.locator(f"[data-cmp-cell='{GOOD}']").get_attribute("data-best") == "1"
    # never under lm_eval's group
    assert page.locator("[data-cmp-group='standard'] [data-cmp-row='mab_sql']").count() == 0
    shot(page.locator("[data-compare]"), "compare-mobile.png")
    assert page.errors == []


def test_benchmarks_panels_and_their_questions(live, page):
    go(page, live, f"tab=benchmarks&sub=standard&models={ids(GOOD, SKEWED)}",
       "[data-panel='mab_hotpotqa'] [data-q-open='mab_hotpotqa']")
    assert page.locator("[data-panel='mab_sql']").count() == 1
    page.locator("[data-panel='mab_hotpotqa'] [data-q-open='mab_hotpotqa']").click()
    page.wait_for_selector("[data-qx-q]")
    first = page.locator("[data-qx-q]").first
    assert first.locator("[data-qx-context] summary").inner_text() == "the passages ▸"
    v = first.locator(f"[data-qx-res='{GOOD}'] .qx-verdict").inner_text()
    assert v.startswith(" · F1 ") and "· EM " in v and "· BLEU " in v
    src = page.locator("[data-qx-source]").inner_text()
    assert "HotpotQA" in src and "CC BY-SA 4.0" in src
    shot(page, "hotpotqa-questions.png")
    go(page, live, f"tab=benchmarks&q=mab_sql&models={ids(SKEWED)}", "[data-qx-q]")
    v = page.locator("[data-qx-q]").first.locator(f"[data-qx-res='{SKEWED}'] .qx-verdict")
    assert "SQL from code block" in v.inner_text()
    assert page.errors == []


def test_test_a_model_offers_the_suite(live, page):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    opt = page.locator("[role='option'][data-value='mobile']")
    opt.wait_for()
    assert opt.inner_text().startswith("Mobile tasks (MobileAIBench) — HotpotQA, SQL")
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_at_400px(live, page, scheme):
    go(page, live, "tab=models&chip=mobile", "[data-lb-table]", width=400, scheme=scheme)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-lb-card]"), f"models-mobile-400-{scheme}.png")
    assert page.errors == []


def test_the_chip_stands_on_its_own_two(live, page):
    """as Instruction & maths does: the Standard rank and Avg are Standard's"""
    go(page, live, "tab=models&chip=mobile", "[data-lb-table]")
    cols = page.locator("[data-lb-table] thead th[data-col]").evaluate_all(
        "xs => xs.map(x => x.dataset.col)")
    assert "avg" not in cols and "rank" not in cols and "mab_sql" in cols
    assert page.errors == []
