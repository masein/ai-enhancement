"""12a on the page: #everyday shows the fixture's four models against the
five questions and every mark opens the answer; the model page has its
Everyday block, or one line with a Test button; Run the pilot queues one run
per ticked model; one Pilot · not ranked badge per page, never per row; and
nothing of the pilot reaches the Leaderboard."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a"
FOUR = ["below-135m-it", "chance-160m", "good-750m", "skewed-360m"]    # by name, not by score
LABELS = ["Typos", "JSON", "Arabic", "Fix the email", "Just 3 names"]
UNTESTED = "fx/one-option-70m"


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def cancel_pilot_rows(base):
    for r in api(base, "/api/submissions"):
        if r["suite"] == "everyday" and r["status"] == "queued":
            api(base, f"/api/submissions/{r['id']}/cancel", {})


def open_pilot(page, base):
    page.goto(base + "/#everyday")
    page.wait_for_selector("[data-everyday-table]")


# ---------------------------------------------------------------------------
# #everyday
# ---------------------------------------------------------------------------

def test_the_pilot_page_shows_four_models_by_five_questions(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_pilot(page, live["base"])
    heads = page.locator("[data-everyday-table] thead th[data-evd-model]")
    assert [h.locator("a, span").first.text_content() for h in heads.all()] == FOUR
    # the n of 5 in the header row, from the marks themselves
    counts = {h.get_attribute("data-evd-model"): h.locator("[data-evd-count]").text_content()
              for h in heads.all()}
    assert counts == {"fx/good-750m": "5 of 5", "fx/below-135m-it": "1 of 5",
                      "fx/skewed-360m": "2 of 5", "fx/chance-160m": "1 of 5"}
    rows = page.locator("[data-everyday-table] tbody tr")
    assert [r.locator(".evq-short").text_content() for r in rows.all()] == LABELS
    # the full question under its short label, typos and all
    assert rows.first.locator(".evq-full").text_content() == \
        "hey can u tell me hwo many days is in febuary in a leep yaer"
    assert rows.first.locator(".evq-group").text_content() == "Understanding"
    cells = page.locator("[data-evd-cell]")
    assert cells.count() == 20
    marks = {c.get_attribute("data-evd-cell"): c.text_content() for c in cells.all()}
    assert marks["fx/good-750m|everyday-pilot-01"] == "✓"
    assert marks["fx/skewed-360m|everyday-pilot-01"] == "✗"
    assert marks["fx/below-135m-it|everyday-pilot-02"] == "✓"
    # one badge on the page, in the header; no row repeats it
    assert page.locator("[data-pilot-badge]").count() == 1
    assert page.locator("[data-everyday-head] [data-pilot-badge]").text_content() \
        == "Pilot · not ranked"
    body = page.locator("[data-everyday-table]").text_content()
    assert "not ranked" not in body and "Pilot" not in body
    # one main action, top right, filled
    run = page.locator("[data-everyday-head] button.primary")
    assert run.count() == 1 and run.text_content() == "Run the pilot"
    assert page.locator("#view button.primary").count() == 1
    # no system words: no ids, no check names, no file or suite names
    view = page.locator("#view").text_content()
    for word in ("everyday-pilot-", "everyday_pilot", ".jsonl", "contains", "judge", "suite"):
        assert word not in view, word
    shot(page, "12a-1-pilot-page-1400-light.png")
    assert page.errors == []


def test_a_mark_opens_the_answer_and_the_panel_moves_along_a_row_and_a_column(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_pilot(page, live["base"])
    page.locator("[data-evd-cell='fx/good-750m|everyday-pilot-01']").click()
    page.wait_for_selector("#reader[data-kind='everyday'][data-ready='1']")
    assert "read=everyday:fx/good-750m:1" in page.evaluate("decodeURIComponent(location.hash)")
    rd = page.locator("#reader")
    assert rd.locator("#readerTitle").text_content() == "good-750m · Typos"
    # the question above the answer, the answer after the thinking, the thinking folded
    q = rd.locator("[data-evd-question]")
    a = rd.locator("[data-evd-answer]")
    assert q.text_content().startswith("hey can u tell me")
    assert q.bounding_box()["y"] < a.bounding_box()["y"]
    assert a.text_content() == "A leap year February has 29 days."
    th = rd.locator("[data-evd-thinking]")
    assert th.evaluate("e => e.open") is False
    assert th.locator("summary").text_content() == "thinking ▸ 10 words"
    assert rd.locator("[data-evd-verdict]").text_content() == "✓says 29"
    shot(page, "12a-2-answer-panel-1400-light.png")
    # ↓ the next question for this model, → the next model on this question
    page.keyboard.press("ArrowDown")
    page.wait_for_function("document.querySelector('#readerTitle').textContent === "
                           "'good-750m · JSON'")
    page.keyboard.press("ArrowRight")
    page.wait_for_function("document.querySelector('#readerTitle').textContent === "
                           "'skewed-360m · JSON'")
    assert "missing: Dubai, March 2021" in rd.locator("[data-evd-verdict]").text_content()
    page.locator("[data-evd-prev-q]").click()
    page.wait_for_function("document.querySelector('#readerTitle').textContent === "
                           "'skewed-360m · Typos'")
    # an answer that never left its thinking says so, and the thinking is there
    assert rd.locator("[data-no-answer]").text_content() == \
        "No answer: the model was still thinking when it ran out of room."
    assert "never finished answering" in rd.locator("[data-evd-verdict]").text_content()
    assert rd.locator("[data-evd-thinking]").count() == 1
    page.keyboard.press("Escape")
    page.wait_for_selector("#reader", state="detached")
    assert page.errors == []


def test_no_answer_to_every_question_leaves_nothing_empty(live, page):
    """The pilot page before anyone has sat it: one line and the button."""
    import service.app as appmod
    out_dir = live["tree"]["out_dir"]
    moved = []
    for f in out_dir.glob("*/everyday.json"):
        f.rename(f.with_suffix(".json.bak"))
        moved.append(f)
    appmod._cache.update(key=None, payload=None, at=0.0)
    try:
        page.goto(live["base"] + "/#everyday")
        box = page.locator("[data-everyday-empty]")
        box.wait_for()
        assert "No model has taken the pilot yet." in box.text_content()
        assert box.locator("button").text_content() == "Run the pilot"
        assert page.locator("[data-everyday-table]").count() == 0
    finally:
        for f in moved:
            f.with_suffix(".json.bak").rename(f)
        appmod._cache.update(key=None, payload=None, at=0.0)
    assert page.errors == []


# ---------------------------------------------------------------------------
# the model page
# ---------------------------------------------------------------------------

def open_model(page, base, mid):
    page.goto(base + "/#model=" + mid.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")


def test_the_model_page_has_its_everyday_block_above_the_exam(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_model(page, live["base"], "fx/good-750m")
    block = page.locator("[data-everyday-block='fx/good-750m']")
    block.wait_for()
    judged = page.locator("#sec-judged")
    assert block.bounding_box()["y"] < judged.bounding_box()["y"]
    assert block.locator("[data-pilot-badge]").count() == 1
    assert page.locator("[data-pilot-badge]").count() == 1
    assert block.locator("[data-everyday-count]").text_content() == "5 of 5"
    rows = block.locator("[data-evd-row]")
    assert [r.locator(".evgroup").text_content() for r in rows.all()] == \
        ["Understanding", "Transform", "Language", "Writing", "Behaviour"]
    assert [r.locator(".evmark").text_content() for r in rows.all()] == ["✓"] * 5
    assert rows.nth(1).locator(".evreason").text_content() == "valid JSON, all five values"
    # no criteria strip, no score bar, no ids
    assert block.locator(".critrow, [data-bar], .scorebar").count() == 0
    assert "everyday-pilot-" not in block.text_content()
    # a row opens the question and the whole answer under it
    rows.nth(4).click()
    opened = block.locator("[data-evd-open='everyday-pilot-05']")
    opened.wait_for()
    assert "give me 3 names" in opened.locator("[data-evd-question]").text_content()
    assert opened.locator("[data-evd-answer]").text_content() == \
        "1. Bean There\n2. Daily Grind\n3. Brew Haven"
    assert opened.locator("[data-evd-thinking] summary").text_content() == "thinking ▸ 3 words"
    # a poll keeps it open
    page.evaluate("render()")
    assert page.locator("[data-evd-open='everyday-pilot-05']").count() == 1
    block.scroll_into_view_if_needed()
    shot(page, "12a-3-model-block-1400-light.png")
    # Compare models → is the pilot page
    block.locator("[data-everyday-compare]").click()
    page.wait_for_selector("[data-everyday-table]")
    assert page.evaluate("location.hash") == "#tab=benchmarks&sub=everyday"      # 12b
    assert page.errors == []


def test_a_model_that_has_not_sat_the_pilot_shows_one_line_and_test_queues_it(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    cancel_pilot_rows(live["base"])
    open_model(page, live["base"], UNTESTED)
    line = page.locator(f"[data-everyday-none='{UNTESTED}']")
    line.wait_for()
    assert " ".join(line.text_content().split()) == "Not tested on everyday tasks · Test"
    assert page.locator("[data-everyday-block]").count() == 0
    assert page.locator("[data-model-nav] a[data-nav='everyday']").count() == 0
    line.scroll_into_view_if_needed()
    shot(page, "12a-4-not-tested-1400-light.png")
    try:
        line.locator("[data-everyday-test]").click()
        toast = page.locator("[data-toast='submit']")
        toast.wait_for()
        rows = [r for r in api(live["base"], "/api/submissions")
                if r["hf_id"] == UNTESTED and r["suite"] == "everyday"]
        assert len(rows) == 1 and rows[0]["status"] == "queued"
        assert toast.text_content().startswith(f"Run #{rows[0]['id']} queued —")
        assert toast.locator("[data-toast-link]").text_content() == "follow it →"
    finally:
        cancel_pilot_rows(live["base"])
    assert page.errors == []


# ---------------------------------------------------------------------------
# Run the pilot
# ---------------------------------------------------------------------------

def test_run_the_pilot_queues_one_run_per_ticked_model(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    cancel_pilot_rows(live["base"])
    open_pilot(page, live["base"])
    page.locator("[data-everyday-run]").click()
    dlg = page.locator("[data-dialog='everyday']")
    dlg.wait_for()
    picks = dlg.locator("[data-evd-pick]")
    ids = [p.get_attribute("data-evd-pick") for p in picks.all()]
    assert ids[:4] == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B",
                       "HuggingFaceTB/SmolLM2-360M-Instruct", "google/gemma-3-270m-it"]
    ticked = [p.get_attribute("data-evd-pick") for p in picks.all()
              if p.locator("input").is_checked()]
    assert ticked == ids[:4]
    # an instruct model on the board that already sat it: done, unticked
    done = dlg.locator("[data-evd-pick='fx/below-135m-it']")
    assert done.locator("[data-evd-done]").text_content() == "done · run again"
    assert not done.locator("input").is_checked()
    go = dlg.locator("[data-dialog-go]")
    assert go.text_content() == "Queue 4 runs"
    shot(page, "12a-5-run-the-pilot-1400-light.png")
    try:
        go.click()
        toast = page.locator("[data-toast='submit']")
        toast.wait_for()
        assert toast.locator(".toast-text").text_content() == "4 runs queued ·"
        assert toast.locator("[data-toast-link]").text_content() == "follow them →"
        assert page.locator("[data-dialog='everyday']").count() == 0
        rows = [r for r in api(live["base"], "/api/submissions")
                if r["suite"] == "everyday" and r["status"] == "queued"]
        assert sorted(r["hf_id"] for r in rows) == sorted(ids[:4])
        toast.locator("[data-toast-link]").click()
        page.wait_for_selector("[data-queue-row]")
        assert page.locator(f"tr[data-queue-row='{rows[0]['id']}'] [data-suite-cell]") \
            .first.text_content() == "everyday pilot"
    finally:
        cancel_pilot_rows(live["base"])
    assert page.errors == []


def test_it_is_reached_from_benchmarks_and_test_a_model_offers_it(live, page):
    """12b: More ▸ Everyday pilot is Benchmarks ▸ Everyday tasks, and the
    Submit form is the Test a model dialog."""
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=runs")
    page.locator("#tabs [data-tab='benchmarks']").click()
    page.locator("[data-subswitch] [data-sub='everyday']").click()
    page.wait_for_selector("[data-everyday-table]")
    assert page.evaluate("location.hash") == "#tab=benchmarks&sub=everyday"
    assert page.locator("#tabs [data-tab='benchmarks']").get_attribute("aria-selected") == "true"
    page.locator("header [data-test-model]").click()
    page.wait_for_selector("[data-dialog='test'] [data-suite-help]")
    page.get_by_label("suite").click()
    opt = page.locator("#pop-sel-submit-suite [role=option][data-value='everyday']")
    assert opt.text_content() == "Everyday tasks — 5 questions, minutes"
    page.keyboard.press("Escape")
    assert page.errors == []


def test_the_pilot_is_on_no_leaderboard(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("table.lb")
    cols = page.evaluate("[...document.querySelectorAll('table.lb thead th[data-col]')]"
                         ".map(th => th.dataset.col)")
    assert not any("everyday" in c for c in cols)
    assert "everyday" not in page.evaluate("JSON.stringify([DATA.models, DATA.accTasks, "
                                           "DATA.pplTasks, DATA.cells, DATA.required])")
    open_model(page, live["base"], "fx/good-750m")
    assert "everyday" not in page.locator("#sec-results").text_content().lower()
    assert page.errors == []


def test_at_400px_the_table_scrolls_inside_its_card(live, browser):
    ctx = browser.new_context(viewport={"width": 400, "height": 900}, reduced_motion="reduce")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        open_pilot(page, live["base"])
        wrap = page.locator("[data-hkeep='everyday']")
        assert wrap.evaluate("e => e.scrollWidth > e.clientWidth")
        assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
        shot(page, "12a-6-pilot-page-400-light.png", full_page=True)
        assert errors == []
    finally:
        ctx.close()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, browser, theme):
    for width in (1400, 400):
        ctx = browser.new_context(viewport={"width": width, "height": 1000},
                                  reduced_motion="reduce")
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            open_pilot(page, live["base"])
            page.evaluate(f"applyTheme('{theme}')")
            page.wait_for_timeout(200)
            shot(page, f"12a-pilot-{width}-{theme}.png")
            open_model(page, live["base"], "fx/skewed-360m")
            page.evaluate(f"applyTheme('{theme}')")
            page.locator("[data-evd-row='everyday-pilot-01']").click()
            page.wait_for_timeout(200)
            page.evaluate("document.querySelector('[data-everyday-block]').scrollIntoView()")
            page.wait_for_timeout(200)
            shot(page, f"12a-model-{width}-{theme}.png")
            assert errors == []
        finally:
            ctx.close()

