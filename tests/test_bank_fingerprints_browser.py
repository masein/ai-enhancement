"""Phase 10b on the page: a grade on a retired question set is history, the
Sit panel ticks the topic whose page it is on, and "Review the spec" lands
on the proposal it is about."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import pytest

from conftest import choose, open_kind

pytestmark = pytest.mark.dashboard

MODEL = "fx/good-750m"


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.loads(r.read())


def test_a_retired_grade_is_under_earlier_exams_and_not_on_the_judged_card(live, page):
    judge = live["tree"]["models"][MODEL]["dir"] / "judge.json"
    kept = judge.read_bytes()
    j = json.loads(kept)
    t = j["tasks"]["exam_law"]
    t.pop("bank_sha256", None)                     # graded before fingerprints existed
    t.pop("topic", None)
    for it in t.get("items") or []:
        it["category"] = "law"
    judge.write_text(json.dumps(j), encoding="utf-8")
    try:
        page.goto(live["base"] + "/#model=fx%2Fgood-750m")
        # 12b.2: in the exam block, under More detail
        open_kind(page, "exam")
        page.locator("[data-more-detail='judged'] > summary").click()
        earlier = page.locator("[data-earlier]")
        earlier.wait_for()
        assert "Earlier exams (retired question sets)" in earlier.text_content()
        row = earlier.locator("tr[data-earlier-task='exam_law']")
        assert row.count() == 1
        cells = row.locator("td").all_text_contents()
        # under the name it was sat under, with its score and its date
        assert cells[0] == "law" and cells[1].endswith("/ 4")
        assert re.fullmatch(r"\d{4}-\d\d-\d\d", cells[3]), cells
        # the current Law is not on the judged card: this model has not sat it
        assert page.locator("tr[data-topic='Law']").count() == 0
        # it is inside the exam's block, where the sub-nav used to point
        assert page.locator("[data-kind-block='exam'] [data-earlier]").count() == 1
        assert page.errors == []
    finally:
        judge.write_bytes(kept)


def test_a_topic_page_ticks_exactly_its_own_topic(live, page):
    """On the physics topic page, law was still ticked from the page before."""
    base = live["base"]
    page.goto(base + "/#topic=law")
    page.wait_for_selector("[data-panel='sit'] [data-sit-task]")
    ticked = lambda: sorted(page.eval_on_selector_all(          # noqa: E731
        "[data-panel='sit'] [data-sit-task]",
        "els => els.filter(e => e.checked).map(e => e.dataset.sitTask)"))
    assert ticked() == ["exam_law"]
    page.locator("[data-sit-task='exam_economics']").check()   # someone adds one here…
    assert ticked() == ["exam_economics", "exam_law"]
    page.goto(base + "/#topic=physics_astronomy")               # …and moves on
    page.wait_for_selector("[data-panel='sit'] [data-sit-task='exam_physics_astronomy']")
    assert ticked() == ["exam_physics_astronomy"]
    assert page.errors == []


def test_review_the_spec_lands_on_the_proposal_and_marks_it(live, page):
    from service import llm_poller
    base = live["base"]
    model = MODEL                  # the board's default, chance-160m, has collapsed answers
    req = urllib.request.Request(base + "/api/proposals", data=json.dumps(
        {"model": model, "topic": "Economics", "requested_by": "tester"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req).read())["id"]
    except urllib.error.HTTPError as e:
        raise AssertionError(f"{model}: {e.read().decode()}") from None
    llm_poller.tick()
    assert _get(base, f"/api/proposals/{pid}")["status"] == "proposed"
    page.set_viewport_size({"width": 1240, "height": 700})
    page.goto(base + "/#tab=loop")
    sel = page.locator("[aria-label='results for']")
    sel.wait_for()
    choose(sel, model)
    btn = page.locator("tr[data-loop-row='economics'] button[data-step='review']")
    btn.wait_for()
    btn.click()
    # 11j: the step opens that proposal's own card in the reader's sheet, and
    # says so in the address — no hunting for it among the others
    page.wait_for_selector("#reader[data-ready='1']")
    assert page.locator("#reader").get_attribute("data-key") == f"proposal:{pid}"
    assert page.evaluate("location.hash").endswith(f"read=proposal:{pid}")
    assert f"#{pid}" in page.locator("#reader .rd-src").text_content()
    assert page.locator(f"[data-why-line='{pid}']").count() == 1
    # Esc closes it and the tab is behind it, on To review
    page.keyboard.press("Escape")
    page.wait_for_selector("#reader", state="detached")
    assert page.locator("[data-rv-view='review'][aria-selected='true']").count() == 1
    assert page.errors == []
