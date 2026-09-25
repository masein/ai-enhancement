"""12a.5a on the page: the bank's eight groups — "Shorten a message" and
"Summarise" where "Summarising" was; what a model's last marking was, in one
line ("55 new questions · 333 re-marked"); Run everyday tasks ticking a model
that has questions it was not asked; and the answer reader listing each
check a failed answer failed, with that check's plain words."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import open_kind

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a5"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"


def shot(page_or_part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page_or_part.screenshot(path=SCREENS / name, **kw)


def served(page, edit):
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def everyday_page(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=benchmarks&sub=everyday")
    page.wait_for_selector("[data-everyday-table]")


def test_the_page_has_eight_groups(live, page):
    everyday_page(page, live["base"])
    assert page.locator("[data-everyday-table] .evq-short").all_inner_texts() == [
        "Understanding", "Writing", "Shorten a message", "Summarise", "Transform",
        "Quick maths", "Instructions", "Honesty"]
    assert "in eight groups" in page.locator("[data-everyday-head]").inner_text()
    assert "practice questions in eight groups" in \
        page.locator("[data-everyday-bank]").inner_text()
    # the long texts: each group's split beside it
    assert page.locator("[data-evd-group-split='summarising']").inner_text() == \
        "26 hidden · 19 practice"
    assert page.locator("[data-evd-group-split='shorten']").inner_text() == \
        "26 hidden · 37 practice"
    shot(page, "12a5-everyday-1400-light.png", full_page=True)
    assert page.errors == []


def test_the_model_page_says_what_the_last_marking_was(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + GOOD.replace("/", "%2F"))
    open_kind(page, "everyday")
    # the fixture's answers were marked, not asked by a run
    assert page.locator(f"[data-evd-marking='{GOOD}']").inner_text() == "388 re-marked"
    assert page.errors == []


def test_a_run_and_a_re_mark_say_it_in_one_line(live, page):
    def after(body):
        body["everyday"]["models"][GOOD].update(marking="55 new questions · 333 re-marked")
        body["everyday"]["models"][SKEWED].update(marking="333 re-marked · 55 not asked yet",
                                                  unasked=55)
    served(page, after)
    page.set_viewport_size({"width": 1400, "height": 1000})
    for mid, line in ((GOOD, "55 new questions · 333 re-marked"),
                      (SKEWED, "333 re-marked · 55 not asked yet")):
        page.goto(live["base"] + "/#model=" + mid.replace("/", "%2F"))
        open_kind(page, "everyday")
        assert page.locator(f"[data-evd-marking='{mid}']").inner_text() == line
    shot(page.locator(f"[data-everyday-block='{SKEWED}']"), "12a5-model-block-1400-light.png")
    assert page.errors == []


def test_run_everyday_tasks_ticks_a_model_with_questions_not_asked(live, page):
    def asked(body):
        m = body["everyday"]["models"]
        m["Qwen/Qwen3-1.7B"] = {**m[GOOD], "unasked": 55}
        m["Qwen/Qwen3-0.6B"] = {**m[GOOD], "unasked": 0}
    served(page, asked)
    everyday_page(page, live["base"])
    page.locator("[data-everyday-run]").click()
    dlg = page.locator("[data-dialog='everyday']")
    dlg.wait_for()
    new = dlg.locator("[data-evd-pick='Qwen/Qwen3-1.7B']")
    assert new.locator("input").is_checked()
    assert new.locator("[data-evd-done]").inner_text() == "55 not asked yet · asks only those"
    done = dlg.locator("[data-evd-pick='Qwen/Qwen3-0.6B']")
    assert not done.locator("input").is_checked()
    assert done.locator("[data-evd-done]").inner_text() == "all answered · marks them again"
    assert "It asks the questions the model has no answer to on today's words" in \
        dlg.inner_text()
    shot(dlg.locator(".dlg"), "12a5-run-dialog-1400-light.png")
    page.keyboard.press("Escape")
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1400])
def test_the_reader_lists_each_check_a_failed_answer_failed(live, page, width):
    everyday_page(page, live["base"], width)
    it = page.evaluate(f"""() => DATA.everyday.models[{json.dumps(SKEWED)}].items
      .find(it => it.pass === false && (it.failed || []).length > 1)""")
    q = page.evaluate(f"""() => DATA.everyday.questions.find(q => q.id === {json.dumps(it['id'])})""")
    page.locator(f"[data-evd-cell='{SKEWED}|{q['group']}']").click()
    page.locator(f"[data-evd-row='{q['id']}']").click()
    page.wait_for_selector("#reader[data-ready='1'] [data-evd-fail]")
    rows = page.locator("#reader [data-evd-fail]")
    assert rows.count() == len(it["failed"]) > 1
    for n, f in enumerate(it["failed"]):
        assert rows.nth(n).locator("[data-evd-fail-why]").inner_text() == f["why"]
        assert rows.nth(n).locator("[data-evd-fail-check]").inner_text() == \
            "The check: " + f["check"]
    # the plain words are the question list's
    assert {f["check"] for f in it["failed"]} <= set(q["checks"])
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12a5-reader-fails-{width}-light.png")
    assert page.errors == []
