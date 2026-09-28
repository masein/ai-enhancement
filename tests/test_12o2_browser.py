"""12o.2 on the page: Questions ▸ from a Benchmarks panel, a Models column's
⋯ and a Compare row opens a benchmark's questions at #tab=benchmarks&q=<task>
with the Models ▾ choice — 50 a page, each with each chosen model's pick,
✓/✗ and margin, "not run" for a model with no run, the filters, the GGUF
line, the other half as a count and a line, and the owner's audit behind the
warning. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

import diagnose as dx
from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12o2"
GOOD, CHANCE, NODIAG = "fx/good-750m", "fx/chance-160m", "local/nodiag-step400"


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


def listed(page):
    return page.locator("[data-qx-q]").evaluate_all("xs => xs.map(x => x.dataset.qxQ)")


def test_a_panels_questions_link_opens_its_questions_with_each_models_result(live, page):
    go(page, live, f"tab=benchmarks&sub=standard&models={ids(GOOD, CHANCE)}",
       "[data-panel='mmlu'] [data-q-open='mmlu']")
    page.locator("[data-panel='mmlu'] [data-q-open='mmlu']").click()
    page.wait_for_selector("[data-qx-q]")
    assert page.evaluate("location.hash") == f"#tab=benchmarks&q=mmlu&models={ids(GOOD, CHANCE)}"
    assert page.locator("[data-qx-head] h2").inner_text().startswith("MMLU · questions")
    counts = page.locator("[data-qx-counts]")
    n, other = map(int, counts.get_attribute("data-qx-counts").split("|"))
    assert counts.inner_text().startswith(f"{n:,} questions listed · {other:,} more are the "
                                          "report half: never shown")
    # 50 a page, every one from the half that may be listed
    qs = listed(page)
    assert len(qs) == 50 and all(dx.split_of(q) == "diagnose" for q in qs)
    first = page.locator("[data-qx-q]").first
    assert first.locator("[data-qx-opt]").count() == 4
    assert first.locator(".qx-right").count() == 1
    for m in (GOOD, CHANCE):
        res = first.locator(f"[data-qx-res='{m}']")
        assert res.get_attribute("data-qx-state") in ("right", "wrong")
        assert res.locator("[data-qx-pick]").count() == 1
        assert 0 <= float(res.locator("[data-qx-margin]").get_attribute("data-qx-margin")) <= 1
    # the GGUF's line
    assert page.locator("[data-qx-gguf='mmlu']").inner_text() == \
        "Measured on the GGUF: llama.cpp records only the total"
    shot(page, "mmlu-questions-1400.png")
    # the next page
    assert page.locator("[data-qx-pager]").first.get_attribute("data-qx-pager").startswith("1|50|")
    page.locator("[data-qx-next]").first.click()
    page.wait_for_function("document.querySelector('[data-qx-pager]').dataset.qxPager"
                           ".startsWith('51|')")
    assert not set(listed(page)) & set(qs)
    assert page.errors == []


def test_the_filters_narrow_the_list(live, page):
    go(page, live, f"tab=benchmarks&q=mmlu&models={ids(GOOD, CHANCE)}", "[data-qx-q]")
    page.locator("[data-select='which']").click()
    page.locator("[role='option'][data-value='disagree']").click()
    page.wait_for_function("document.querySelector('[data-qx-pager]').dataset.qxPager"
                           ".split('|')[2] < 158")
    for q in page.locator("[data-qx-q]").all()[:10]:
        states = {q.locator(f"[data-qx-res='{m}']").get_attribute("data-qx-state")
                  for m in (GOOD, CHANCE)}
        assert states == {"right", "wrong"}
    page.locator("[data-select='subject']").click()
    page.locator("[role='option'][data-value='anatomy']").click()
    page.wait_for_function("[...document.querySelectorAll('.qx-subj')].every(x => "
                           "x.textContent === 'anatomy')")
    assert page.errors == []


def test_a_chosen_model_with_no_run_is_not_run(live, page):
    go(page, live, f"tab=benchmarks&q=mmlu&models={ids(GOOD, NODIAG)}", "[data-qx-q]")
    cell = page.locator("[data-qx-q]").first.locator(f"[data-qx-res='{NODIAG}']")
    assert cell.get_attribute("data-qx-state") == "none"
    assert cell.inner_text().endswith("not run")
    assert page.errors == []


def test_the_models_column_menu_and_compares_row_open_it_too(live, page):
    go(page, live, "tab=models", "[data-lb-table]")
    page.locator("[data-lb-table] thead th[data-col='hellaswag'] [data-col-more='hellaswag']").click()
    page.locator("[role='menuitem'][data-q-open='hellaswag']").click()
    page.wait_for_selector("[data-qx-head='hellaswag']")
    assert page.evaluate("location.hash").startswith("#tab=benchmarks&q=hellaswag")
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, CHANCE)}", "[data-compare='2']")
    page.locator("[data-cmp-row='arc_easy'] [data-q-open='arc_easy']").click()
    page.wait_for_selector("[data-qx-head='arc_easy']")
    # back to Benchmarks
    page.locator("[data-qx-back]").click()
    page.wait_for_selector("[data-panel]")
    assert "q=" not in page.evaluate("location.hash")
    assert page.errors == []


def test_the_owners_audit_behind_the_warning(live, page):
    go(page, live, "tab=benchmarks&q=hellaswag", "[data-qx-q]")
    assert page.locator("[data-qx-audit-open]").count() == 0       # no name, no audit
    set_name(page, "sam")
    go(page, live, "tab=benchmarks&q=hellaswag", "[data-qx-q]")
    assert page.locator("[data-qx-audit-open]").count() == 0       # not the owner, no audit
    set_name(page, "masein")
    go(page, live, "tab=benchmarks&q=hellaswag", "[data-qx-audit-open='hellaswag']")
    page.locator("[data-qx-audit-open='hellaswag']").click()
    dlg = page.locator("[data-audit-dialog='hellaswag']")
    assert dlg.locator("[data-audit-warning]").inner_text() == (
        "These questions are the test. Don’t train on them or write questions toward them. "
        "This opening is logged.")
    dlg.locator("[data-audit-go]").click()
    page.wait_for_selector("[data-qx-audit-banner]")
    page.wait_for_function("[...document.querySelectorAll('[data-qx-q]')].length > 0")
    assert all(dx.split_of(q) == "report" for q in listed(page))
    assert page.locator("[data-qx-audit]").inner_text() == "report half · audit"
    shot(page, "hellaswag-audit-1400.png")
    page.locator("[data-qx-audit-close]").click()
    page.wait_for_selector("[data-qx-audit-banner]", state="detached")
    page.wait_for_function("[...document.querySelectorAll('[data-qx-q]')].length > 0")
    assert all(dx.split_of(q) == "diagnose" for q in listed(page))
    audits = json.loads(page.evaluate("fetch('api/everyday/audits').then(r => r.text())"))
    assert audits["audits"][0]["group"] == "hellaswag · report half"
    assert page.errors == []


def test_gpqa_is_never_linked_or_listed(live, page):
    go(page, live, "tab=benchmarks&q=gpqa_diamond_cot_zeroshot", "[data-qx-error]")
    assert "never shown" in page.locator("[data-qx-error]").inner_text()
    assert page.evaluate("canBrowse('gpqa_diamond_cot_zeroshot')") is False
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_at_400px(live, page, scheme):
    go(page, live, f"tab=benchmarks&q=mmlu&models={ids(GOOD, CHANCE)}", "[data-qx-q]",
       width=400, scheme=scheme)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page, f"mmlu-questions-400-{scheme}.png")
    assert page.errors == []
