"""14.4.3 on the page: the full Mobile-MMLU kept apart — its own table under
Mobile tasks' (no column of the table above, no rank, no average), the model
page's line with its 9 categories and 80 fields, its own group in Compare,
and Test a model's part with its estimate. At 1400 and 375 px. Invented rows,
picks written in: nothing runs, nothing calls OpenRouter."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import quote

import pytest

import mobile_mmlu as mmp
from conftest import set_name
from test_14_3_browser import go, ids, no_sideways, steady_shot
from test_14_3_mobile_mmlu import FIXTURE, RIGHT, pin_fixture, sit_picks
from test_14_4_1_mobile_mmlu_data import FIXTURE as FULL, pin_full
from test_14_4_3_mobile_mmlu_full_run import lids, right, sit_full

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase14_4_3"
GOOD, CHANCE, SHORT = "fx/good-750m", "fx/chance-160m", "fx/short-pick-410m"
WIDTHS = [1400, 375]


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """both sets on this server and one key for them; good-750m sat the full
    set (one wrong), chance-160m too (right only where the full set has its own wording), and
    short-pick-410m Pro alone — everything put back after"""
    import service.app as appmod
    from service import config
    kd = mmp.key_dir()
    saved = {f: (kd / f).read_bytes() for f in ("labels.json", "key.json") if (kd / f).exists()}
    tree = live["tree"]
    dirs = {m: Path(tree["models"][m]["dir"]) for m in (GOOD, CHANCE, SHORT)}
    with pytest.MonkeyPatch.context() as mp:
        d = Path(config.MMP_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes())
        pin_fixture(mp)
        full_dir = Path(config.MMF_DIR)
        (full_dir / "test").mkdir(parents=True, exist_ok=True)
        for p in (FULL / "test").glob("*.csv"):
            (full_dir / "test" / p.name).write_bytes(p.read_bytes())
        pin_full(mp)
        from test_14_4_3_mobile_mmlu_full_run import pool_key
        pool_key()
        rows = mmp.pool()
        good = {q["lid"]: right(q) for q in rows}
        good[lids("full")["inv00014"]] = "B"
        sit_full(dirs[GOOD], good)
        sit_full(dirs[CHANCE], {q["lid"]: right(q) if q["sets"] == ["full"] else "D" if
                                right(q) != "D" else "A" for q in rows})
        sit_picks(dirs[SHORT], RIGHT)
        for m in dirs.values():
            mmp.collect(m)
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
        for m in dirs.values():
            for f in ("mobile_mmlu_full.json", "mobile_mmlu_pro.json"):
                (m / f).unlink(missing_ok=True)
            for t in ("mobile_mmlu_full_0shot", "mobile_mmlu_pro_0shot"):
                shutil.rmtree(m / t, ignore_errors=True)
        for f in ("labels.json", "key.json"):
            if f in saved:
                (kd / f).write_bytes(saved[f])
            else:
                (kd / f).unlink(missing_ok=True)
        shutil.rmtree(full_dir / "test", ignore_errors=True)
        mmp._keyc.clear()
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


def shot_clear(page, part, name):
    """the part whole, below the page's sticky header"""
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.evaluate("e => window.scrollTo(0, e.getBoundingClientRect().top + scrollY - 90)",
                  part.element_handle())
    b = part.bounding_box()
    page.screenshot(path=SCREENS / name, clip={"x": max(0, b["x"] - 6), "y": b["y"] - 6,
                                               "width": b["width"] + 12,
                                               "height": b["height"] + 12})


@pytest.mark.parametrize("width", WIDTHS)
def test_mobile_tasks_has_the_full_sets_own_table_and_no_column_of_it(live, page, width):
    go(page, live, "tab=models&chip=mobile", "[data-mmf-card]", width)
    # never a column of the table above, and nothing ranks or averages it
    assert page.locator("th[data-col='mobile_mmlu_full']").count() == 0
    assert page.locator("th[data-col='mobile_mmlu_pro']").count() == 1
    card = page.locator("[data-mmf-card]")
    assert card.locator("h2").inner_text() == "Mobile-MMLU (full)"
    rows = card.locator("[data-mmf-row]")
    assert rows.evaluate_all("xs => xs.map(x => x.dataset.mmfRow)") == [GOOD, CHANCE]
    assert card.locator(f"[data-mmf-cell='{GOOD}']").inner_text() == "94.4%"      # 17 of 18
    assert card.locator(f"[data-mmf-cell='{CHANCE}']").inner_text() == "38.9%"    # 7 of 18
    about = card.locator("[data-mmf-about]").inner_text()
    assert "never in any average, rank or overall" in about and "provisional key" in about
    assert "CC BY-NC-ND 4.0" in card.locator("[data-mmf-credit]").inner_text()
    heads = card.locator("thead th").evaluate_all("xs => xs.map(x => x.textContent)")
    assert "Rank" not in heads and "Avg" not in heads and heads[:2] == ["Model", "Accuracy"]
    no_sideways(page)
    shot(card, f"table-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_model_page_has_the_full_sets_line_categories_and_fields(live, page, width):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    line = page.locator(f"[data-mmf-line='{GOOD}']")
    line.wait_for()
    assert line.locator("p").first.inner_text() == (
        "Mobile-MMLU (full) 94.4% on 18 kept questions · log-likelihood · provisional key")
    # one run, two scores: Pro's line from the same picks
    assert page.locator(f"[data-mmp-line='{GOOD}'] p").first.inner_text().startswith(
        "Mobile-MMLU-Pro 100.0% on 12 kept questions")
    line.locator(f"[data-mmf-cats='{GOOD}'] > summary").click()
    names = line.locator("[data-mmf-cat]").evaluate_all("xs => xs.map(x => x.dataset.mmfCat)")
    assert names and set(names) <= set(mmp.CATEGORIES)
    line.locator(f"[data-mmf-fields='{GOOD}'] > summary").click()
    fields = line.locator("[data-mmf-field]").evaluate_all("xs => xs.map(x => x.dataset.mmfField)")
    assert len(fields) == 12 and fields == sorted(fields)
    q14 = mmp.full_by_id()["inv00014"]["field"]
    assert line.locator(f"[data-mmf-field='{q14}'] td").nth(1).inner_text() != "100.0%"
    no_sideways(page)
    page.set_viewport_size({"width": width, "height": 2400})      # the whole line, opened
    shot_clear(page, line, f"model-line-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_compare_has_the_full_set_as_a_group_of_its_own(live, page, width):
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, CHANCE)}", "[data-compare='2']", width)
    g = page.locator("[data-cmp-group='mmf']")
    assert g.inner_text().startswith("▾ Mobile-MMLU (full)")
    assert "CC BY-NC-ND 4.0 · kept apart: never averaged" in g.inner_text()
    body = page.locator("[data-cmp-body='mmf']")
    row = body.locator("[data-cmp-row='mmf']")
    assert row.locator(f"[data-cmp-cell='{GOOD}']").inner_text().startswith("94.4")
    assert row.locator(f"[data-cmp-cell='{CHANCE}']").inner_text().startswith("38.9")
    assert body.locator("[data-cmp-row^='mmf:']").count() >= 1
    no_sideways(page)
    shot(body, f"compare-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_test_a_model_offers_the_full_set_with_what_it_takes(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(SHORT, safe=""), "[data-model-hero]", width)
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    page.wait_for_function("document.querySelector(\"[data-mab-part-est='mmlu_full']\")"
                           "?.textContent.includes('answers')")
    part = page.locator("[data-mab-part='mmlu_full']")
    # its Pro run counts: only the seven it hasn't answered
    assert part.inner_text().strip().startswith(
        "multiple choice, the full set — Mobile-MMLU (full) · 7 answers")
    assert "12 answered already, by its Mobile-MMLU-Pro run" in part.inner_text()
    assert part.locator("input").is_enabled()
    # and Pro's own part, answered whole, has nothing to ask
    assert page.locator("[data-mab-part-est='mmlu']").inner_text().strip() == (
        "· nothing to ask: all 12 answered already, by its Mobile-MMLU-Pro run")
    # Pro keeps its name and stays first
    parts = page.locator("[data-mab-part]").evaluate_all("xs => xs.map(x => x.dataset.mabPart)")
    assert parts.index("mmlu") < parts.index("mmlu_full")
    no_sideways(page)
    shot(page.locator("[data-dialog='test'] .dlg"), f"test-part-{width}.png")
    assert page.errors == []
