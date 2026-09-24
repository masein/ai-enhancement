"""11m on the page: the Exam tab opens a topic's practice questions; the
model page has no by-criterion block and the answer cards keep their strip;
the three header pills share one rule; a failed dataset says 0 of n and what
went wrong, with every reason one click away; the suite options say what
each one gets you."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exam_build as eb
from conftest import assert_no_report_half_text, open_kind, open_submit

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11l"
MODEL = "fx/good-750m"
PLANTED: list[int] = []


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def ready(page, kind):
    page.wait_for_selector(f"#reader[data-kind='{kind}'][data-ready='1']")
    return page.locator("#reader")


def open_exam(page, base):
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='rubrics'] [data-read-bank-count]")


# ---------------------------------------------------------------------------
# §2: the Exam tab opens the bank
# ---------------------------------------------------------------------------

def test_the_exam_tab_opens_the_practice_half_for_three_topics(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    base = live["base"]
    banks = eb.load_bank(Path(live["root"]) / "exam")
    open_exam(page, base)
    slugs = [x.get_attribute("data-read-bank-count")
             for x in page.locator("[data-read-bank-count]").all()][:3]
    assert len(slugs) == 3
    # three ways in, one per topic: the count, "read the practice half", and
    # the questions entry beside rubric and criteria
    openers = ["[data-read-bank-count='{s}']", "[data-read-bank='{s}']",
               "[data-read-questions='{s}']"]
    for slug, sel in zip(slugs, openers):
        open_exam(page, base)
        row = page.locator(f"[data-read-bank-count='{slug}']").locator("xpath=ancestor::tr")
        topic = row.get_attribute("data-rubric-row")
        mine = banks[topic]
        rep = sum(1 for r in mine if eb.half_of(r["qid"]) == "report")
        dia = len(mine) - rep
        cell = row.locator("[data-bank]").text_content()
        assert cell.startswith(f"{len(mine)} — {rep} hidden / {dia} practice · read the practice half")
        if slug == slugs[0]:
            row.scroll_into_view_if_needed()
            shot(page, "11m-1-exam-bank-cell-1400-light.png")
        page.locator(sel.format(s=slug)).first.click()
        box = ready(page, "bank")
        assert f"read=bank:{slug}" in page.evaluate("location.hash")
        page.wait_for_selector("#reader [data-bank-q]")
        shown = [q.get_attribute("data-bank-q") for q in box.locator("[data-bank-q]").all()]
        assert shown and all(eb.half_of(q) == "diagnose" for q in shown), topic
        assert box.locator("[data-bank-hidden]").get_attribute("data-bank-hidden") == str(rep)
        assert_no_report_half_text(box.text_content(), mine)
        if slug == slugs[0]:
            shot(page, "11m-2-exam-bank-reader-1400-light.png")
        page.keyboard.press("Escape")
        page.wait_for_selector("#reader", state="detached")
    assert page.errors == []


# ---------------------------------------------------------------------------
# §3: no by-criterion block; the answer cards keep their strip
# ---------------------------------------------------------------------------

def test_the_model_page_has_no_by_criterion_block_and_the_cards_keep_their_strip(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    open_kind(page, "exam")
    page.wait_for_selector("table[data-judged-topics]")
    for gone in ("[data-topic-switch]", "[data-criteria-table]", "[data-flag-topic]",
                 "[data-breakdown-table]"):
        assert page.locator(gone).count() == 0, gone
    assert "by criterion" not in page.locator("#view").text_content().lower()
    shot(page, "11m-3-model-page-1400-light.png")
    # the strip inside an answer card stays: the squares and "weakest: …"
    page.goto(live["base"] + "/#topic=economics")
    page.wait_for_selector("[data-panel='answers']")
    page.evaluate("m => { state.ans.model = m; state.ans.rows = null; render(); }", MODEL)
    page.wait_for_selector("[data-panel='answers'] [data-answer] .critrow")
    strip = page.locator("[data-panel='answers'] [data-answer] .critrow").first
    assert strip.locator("[data-weakest]").text_content()
    assert page.errors == []


# ---------------------------------------------------------------------------
# §4: one rule for the three pills
# ---------------------------------------------------------------------------

# 12b: the bar's pills are the run counter, the status dot and the name menu —
# Theme is inside the name menu now
PILLS = {"checks": "#warnings summary[data-warn-summary]", "who": "button.who",
         "theme": "#runs [data-runs]"}


def test_the_three_header_pills_share_one_padding(live, page):
    page.set_viewport_size({"width": 1512, "height": 900})
    page.goto(live["base"] + "/")
    page.wait_for_selector(PILLS["checks"])
    got = {k: page.locator(sel).evaluate(
        "e => { const s = getComputedStyle(e); return [s.paddingTop, s.paddingRight, "
        "s.paddingBottom, s.paddingLeft, s.fontSize, s.minHeight, s.borderTopWidth]; }")
        for k, sel in PILLS.items()}
    assert got["checks"] == got["who"] == got["theme"], got
    assert got["checks"][:4] == ["4px", "12px", "4px", "12px"]
    # the text no longer touches its border
    pad = page.locator(PILLS["checks"]).evaluate(
        "e => { const r = e.getBoundingClientRect(), t = document.createRange(); "
        "t.selectNodeContents(e); const q = t.getBoundingClientRect(); return q.left - r.left; }")
    assert pad >= 12
    boxes = [page.locator(sel).bounding_box() for sel in PILLS.values()]
    x0, y0 = min(b["x"] for b in boxes) - 16, min(b["y"] for b in boxes) - 10
    x1 = max(b["x"] + b["width"] for b in boxes) + 16
    y1 = max(b["y"] + b["height"] for b in boxes) + 10
    shot(page, "11m-4-header-pills-1512-light.png",
         clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
    assert page.errors == []


# ---------------------------------------------------------------------------
# §8: a failed dataset says what went wrong
# ---------------------------------------------------------------------------

def plant_failed(fmt, count, missing, audience, error):
    from service import db
    pid = db.proposal_create(MODEL, "exam_government_public_policy",
                             "Government & Public Policy", "masein",
                             {"n_shown": 3, "judge_id": "stub/overlap-v1"})
    db.proposal_update(pid, status="approved", spec_text="the missing skill")
    PLANTED.append(pid)
    did = db.dataset_create(pid, fmt, count, "masein", {})
    db.dataset_update(did, status="failed", error=error,
                      provenance=json.dumps({"audience": audience, "format": fmt, "items": {
                          "requested": count, "generated": 0, "dropped": 0, "kept": 0,
                          "missing": missing}}))
    return did


def clear_planted():
    from service import db
    c = db._conn()                                          # noqa: SLF001
    try:
        for pid in PLANTED:
            c.execute("DELETE FROM datasets WHERE proposal_id = ?", (pid,))
            c.execute("DELETE FROM proposals WHERE id = ?", (pid,))
        c.commit()
    finally:
        c.close()
    PLANTED.clear()


PROSE = ("Audience: people learning or practising the subject (styles: policy_analysis 30%).\n"
         "Register for documents: explain the reasoning step by step. Prose, never "
         "question-and-answer pairs.")


def test_a_failed_dataset_says_0_of_n_and_what_went_wrong(live, page):
    """Dataset #9, as it is on the box: asked for question-and-answer items,
    handed the prose register, 26 items missing for "no question or no
    answer" — and one that failed for another reason, said in its own words."""
    clear_planted()
    nine = plant_failed("free", 26, [{"request": k // 10, "focus": f"area {k % 3}",
                                      "why": "no question or no answer"} for k in range(26)],
                        PROSE, "the generator returned no parseable items")
    other = plant_failed("doc", 4, [{"request": 0, "focus": None, "why": "reply not JSON"},
                                    {"request": 0, "focus": None, "why": "reply not JSON"},
                                    {"request": 1, "focus": None, "why": "empty reply"},
                                    {"request": 1, "focus": None, "why": "empty reply"}],
                         "", "the generator returned no parseable items")
    try:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto(live["base"] + "/#tab=review&view=datasets")
        page.wait_for_selector(f"[data-ds-row='{nine}']")
        # the count where a count belongs; the failure in the status
        assert page.locator(f"[data-doc-line='{nine}']").text_content() == "0 of 26"
        assert page.locator(f"[data-ds-failed='{nine}']").text_content() == "Failed"
        assert page.locator(f"[data-ds-why-line='{nine}']").text_content() == (
            "0 of 26 kept — asked for question-and-answer items, but the generator was given "
            "the prose register.")
        assert page.locator(f"[data-ds-why-line='{other}']").text_content() == (
            "0 of 4 kept — 2 reply not JSON, 2 empty reply.")
        # every reason, paged, one click away
        det = page.locator(f"[data-ds-details='{nine}']")
        assert det.evaluate("e => e.open") is False
        assert det.locator("summary").text_content() == "Details — 26 items, one reason each ▸"
        det.locator("summary").click()
        items = det.locator("[data-ds-missing]")
        assert items.count() == 10
        assert items.first.text_content() == "request 0 · area 0 · no question or no answer"
        det.locator("[data-page-next]").click()
        page.wait_for_function(f"document.querySelectorAll(\"[data-ds-details='{nine}'] "
                               "[data-ds-missing]\")[0].textContent.includes('area 1')")
        assert det.locator("ol").get_attribute("start") == "11"
        # a poll keeps it open, on its page
        page.evaluate("render()")
        det = page.locator(f"[data-ds-details='{nine}']")
        assert det.evaluate("e => e.open") is True
        assert det.locator("ol").get_attribute("start") == "11"
        page.locator(f"[data-ds-row='{nine}']").scroll_into_view_if_needed()
        shot(page, "11m-5-failed-dataset-1400-light.png")
        assert page.errors == []
    finally:
        clear_planted()


# ---------------------------------------------------------------------------
# §9: the suite options
# ---------------------------------------------------------------------------

def test_each_suite_option_says_what_it_gets_you(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    open_submit(page, live["base"])
    help_ = page.locator("[data-suite-help]")
    help_.wait_for()
    assert help_.text_content() == (
        "full and judged are separate runs, not one inside the other: a model needs both to "
        "have an average and a judged score. Resubmitting is free — each run does only the "
        "tasks still missing, which is also how a quick run becomes a full one.")
    page.get_by_label("suite").click()
    page.wait_for_selector("[role=listbox][aria-label='suite']")
    opts = page.locator("#pop-sel-submit-suite [role=option]")
    assert [o.get_attribute("data-value") for o in opts.all()] == \
        ["full", "quick", "control", "judged", "everyday"]
    # 12a: the pilot's option is its one short line, and 12c replaces the list
    assert opts.nth(4).text_content() == "Everyday tasks — 5 questions, minutes"
    for o in opts.all()[:4]:
        v = o.get_attribute("data-value")
        sub = o.locator(f"[data-opt-sub='{v}']").text_content()
        assert len(sub.split()) >= 8, v
    assert "average" in page.locator("[data-opt-sub='full']").text_content()
    assert "judged score" in page.locator("[data-opt-sub='judged']").text_content()
    shot(page, "11m-6-suite-options-1400-light.png")
    # the picked one still reads as a name, not a paragraph
    page.locator("#pop-sel-submit-suite [data-value='quick']").click()
    assert page.get_by_label("suite").text_content().startswith("quick — three tasks, minutes")
    assert page.errors == []


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, browser, theme):
    for width in (1400, 400):
        ctx = browser.new_context(viewport={"width": width, "height": 1000},
                                  reduced_motion="reduce")
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            open_exam(page, live["base"])
            page.evaluate(f"applyTheme('{theme}')")
            page.wait_for_timeout(200)
            page.evaluate("document.querySelector(\"[data-panel='rubrics']\").scrollIntoView()")
            page.wait_for_timeout(200)
            shot(page, f"11m-exam-{width}-{theme}.png")
            open_submit(page, live["base"])
            page.locator("[data-suite-help]").wait_for()
            page.evaluate(f"applyTheme('{theme}')")
            page.get_by_label("suite").click()
            page.wait_for_selector("[role=listbox][aria-label='suite']")
            page.wait_for_timeout(200)
            shot(page, f"11m-suite-{width}-{theme}.png")
            assert errors == []
        finally:
            ctx.close()
