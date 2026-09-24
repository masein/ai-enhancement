"""12a on the page, moved to the bank in 12a.2: #everyday shows the fixture's
four models by name, each count from its own marks, and every answer opens
in the side panel, whose keys walk the bank and the models; the model page
has its Everyday block, or one line with a Test button; Run everyday tasks
queues one run per ticked model; one Round 2 · not ranked badge per page,
never per row; and nothing of it reaches the Leaderboard. The groups-by-models
table itself is test_everyday_12a2_browser's."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from conftest import open_kind

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a"
FOUR = ["below-135m-it", "chance-160m", "good-750m", "skewed-360m"]    # by name, not by score
GROUPS = ["Understanding", "Writing", "Summarising", "Transform", "Quick maths", "Instructions",
          "Honesty"]
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

def test_the_page_names_the_models_and_has_one_action_and_one_badge(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_pilot(page, live["base"])
    heads = page.locator("[data-everyday-table] thead th[data-evd-model]")
    assert [h.locator("a, span").first.text_content() for h in heads.all()] == FOUR
    # each model's count in the header row, from its own marks: two sat the
    # bank, two only the pilot's five
    counts = {h.get_attribute("data-evd-model"): h.locator("[data-evd-count]").text_content()
              for h in heads.all()}
    assert counts == {"fx/good-750m": "108 of 111", "fx/below-135m-it": "2 of 5",
                      "fx/skewed-360m": "57 of 111", "fx/chance-160m": "1 of 5"}
    # one badge on the page, in the header; no row repeats it
    assert page.locator("[data-pilot-badge]").count() == 1
    assert page.locator("[data-everyday-head] [data-pilot-badge]").text_content() \
        == "Round 2 · not ranked"
    body = page.locator("[data-everyday-table]").text_content()
    assert "not ranked" not in body and "Round 2" not in body
    # one main action, top right, filled
    run = page.locator("[data-everyday-head] button.primary")
    assert run.count() == 1 and run.text_content() == "Run everyday tasks"
    assert page.locator("#view button.primary").count() == 1
    # no system words: no ids, no check types, no file or suite names — the
    # questions' checks are said in plain words
    view = page.locator("#view").text_content()
    for word in ("everyday-", "everyday_", ".jsonl", "contains_", "not_contains", "line_count",
                 "max_words", "no_invented", "in_order", "suite"):
        assert word not in view, word
    shot(page, "12a-1-pilot-page-1400-light.png")
    assert page.errors == []


def test_an_answer_opens_in_the_panel_and_the_keys_walk_the_bank_and_the_models(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_pilot(page, live["base"])
    page.locator("[data-evd-cell='fx/good-750m|understanding']").click()
    page.locator("[data-evd-panel] [data-evd-row='everyday-pilot-01']").click()
    page.wait_for_selector("#reader[data-kind='everyday'][data-ready='1']")
    # the pilot's first question is the bank's sixteenth, the last in Understanding
    assert "read=everyday:fx/good-750m:16" in page.evaluate("decodeURIComponent(location.hash)")
    rd = page.locator("#reader")
    title = "document.querySelector('#readerTitle').textContent === "
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
    assert rd.locator("[data-evd-verdict]").text_content() == \
        '✓says "29" or "twenty-nine" or "twenty nine"'
    shot(page, "12a-2-answer-panel-1400-light.png")
    # ↓ the next question in the bank for this model, → the next model on it
    page.keyboard.press("ArrowDown")
    page.wait_for_function(title + "'good-750m · Landlord repair message'")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(title + "'skewed-360m · Landlord repair message'")
    assert rd.locator("[data-evd-verdict]").text_content().startswith('✓says "kitchen", "tap"')
    page.locator("[data-evd-prev-q]").click()
    page.wait_for_function(title + "'skewed-360m · Typos'")
    # an answer that never left its thinking says so, and the thinking is there
    assert rd.locator("[data-no-answer]").text_content() == \
        "No answer: the model was still thinking when it ran out of room."
    assert "never finished answering" in rd.locator("[data-evd-verdict]").text_content()
    assert rd.locator("[data-evd-thinking]").count() == 1
    # ← twice: a model that sat only the pilot, on its question and on one it
    # was never asked
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(title + "'good-750m · Typos'")
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(title + "'chance-160m · Typos'")
    assert rd.locator("[data-evd-answer]").text_content() == \
        "Twenty-nine days, because it is a leap year."
    page.keyboard.press("ArrowDown")
    page.wait_for_function(title + "'chance-160m · Landlord repair message'")
    assert rd.locator("[data-not-asked]").text_content() == \
        "Not asked: this model\u2019s run did not include this question."
    assert rd.locator("[data-evd-verdict]").count() == 0          # nothing to mark
    page.keyboard.press("Escape")
    page.wait_for_selector("#reader", state="detached")
    assert page.errors == []


def test_no_answer_to_every_question_leaves_nothing_empty(live, page):
    """The page before anyone has sat it: one line and the button, and the
    questions, readable before any model has answered them."""
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
        assert "No model has taken everyday tasks yet." in box.text_content()
        assert box.locator("button").text_content() == "Run everyday tasks"
        assert page.locator("[data-everyday-table]").count() == 0
        assert page.locator("[data-everyday-bank] [data-evd-bank-q]").count() == 111
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


def test_the_model_page_has_its_everyday_block_after_the_exam(live, page):
    """12a put the pilot above the exam; 12b.2 names the kinds in one order
    everywhere — Standard, Knowledge exam, Everyday tasks — and the badge is
    said once, on the header's tile."""
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_model(page, live["base"], "fx/good-750m")
    kinds = [b.get_attribute("data-kind-block") for b in page.locator("[data-kind-block]").all()]
    assert kinds.index("exam") < kinds.index("everyday")
    open_kind(page, "everyday")
    block = page.locator("[data-everyday-block='fx/good-750m']")
    block.wait_for()
    assert page.locator("[data-pilot-badge]:visible").count() == 1
    assert page.locator("[data-kind-tile='everyday'] [data-pilot-badge]").is_visible()
    assert block.locator("[data-everyday-count]").text_content() == "108 of 111"
    # 12a.2: a row a group, each its n of k
    groups = block.locator("[data-evd-group]")
    assert [g.locator(".evgroup").text_content() for g in groups.all()] == GROUPS
    # no criteria strip, no score bar, no ids
    assert block.locator(".critrow, [data-bar], .scorebar").count() == 0
    assert "everyday-" not in block.text_content()
    # a group opens its answers, a row a question with its mark and why
    block.locator("[data-evd-group='instructions']").click()
    row = block.locator("[data-evd-answers='fx/good-750m|instructions'] "
                        "[data-evd-row='everyday-pilot-05']")
    row.wait_for()
    assert row.locator("[data-evd-mark]").get_attribute("data-evd-mark") == "ok"
    assert row.locator(".evreason").text_content() == "3 lines · at most 15 words"
    # a poll keeps it open
    page.evaluate("render()")
    assert page.locator("[data-evd-answers='fx/good-750m|instructions']").count() == 1
    # the row opens the question and the whole answer in the side panel
    row.click()
    rd = page.locator("#reader[data-ready='1']")
    rd.locator("[data-evd-answer]").wait_for()
    assert "give me 3 names" in rd.locator("[data-evd-question]").text_content()
    assert rd.locator("[data-evd-answer]").text_content() == \
        "1. Bean There\n2. Daily Grind\n3. Brew Haven"
    assert rd.locator("[data-evd-thinking] summary").text_content() == "thinking ▸ 3 words"
    page.keyboard.press("Escape")
    page.wait_for_selector("#reader", state="detached")
    block.scroll_into_view_if_needed()
    shot(page, "12a-3-model-block-1400-light.png")
    # Compare models → is the Everyday tasks page
    block.locator("[data-everyday-compare]").click()
    page.wait_for_selector("[data-everyday-table]")
    assert page.evaluate("location.hash") == "#tab=benchmarks&sub=everyday"      # 12b
    assert page.errors == []


def test_a_model_that_has_not_sat_the_pilot_shows_one_line_and_test_queues_it(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    cancel_pilot_rows(live["base"])
    open_model(page, live["base"], UNTESTED)
    # 12b.2: the one line is the Everyday tile's, in the header
    line = page.locator(f"[data-everyday-none='{UNTESTED}']")
    line.wait_for()
    assert " ".join(line.inner_text().split()) == "EVERYDAY TASKS Not tested · Test"
    assert page.locator("[data-everyday-block], [data-kind-block='everyday']").count() == 0
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
# Run everyday tasks
# ---------------------------------------------------------------------------

def test_run_everyday_tasks_queues_one_run_per_ticked_model(live, page):
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
    # an instruct model on the board that sat the pilot: what it was asked, unticked
    done = dlg.locator("[data-evd-pick='fx/below-135m-it']")
    assert done.locator("[data-evd-done]").text_content() == "asked 5 of 111 · run all 111"
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
            .first.text_content() == "everyday tasks"
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
    assert opt.text_content() == "Everyday tasks — 111 questions, a few minutes"
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
    open_kind(page, "standard")
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
            open_kind(page, "everyday")
            page.locator("[data-evd-group='understanding']").click()
            page.wait_for_timeout(200)
            page.evaluate("document.querySelector('[data-everyday-block]').scrollIntoView()")
            page.wait_for_timeout(200)
            shot(page, f"12a-model-{width}-{theme}.png")
            assert errors == []
        finally:
            ctx.close()

