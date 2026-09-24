"""11k: the nine things masein and I found on the board on 22 Sep, after
11f–11j went out.

Propose → led to a page with no Propose on it; a run said "done" while the
judge was still grading it; the reader's missing-skill box was empty; the
Queue's table painted over its card; the page slid sideways on a phone; "0 of
5" meant ticked and read as judged; the answers list was one 14,000px scroll;
a mouse click left a focus ring; and a list's chosen row looked like a text
field.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11k"
MODEL = "fx/good-750m"
TOPIC = "Economics"
TASK = "exam_economics"
WIDTHS = [1280, 1440, 1512, 1920]
PAGES = [("overview", "/"), ("loop", "/#tab=loop"), ("models", "/#tab=models"),
         ("leaderboard", "/#tab=leaderboard"), ("queue", "/#tab=queue"),
         ("review", "/#tab=review"), ("topic", "/#topic=economics"),
         ("model", "/#model=fx%2Fgood-750m")]


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def sql(*stmts):
    from service import db
    c = db._conn()                                    # noqa: SLF001 — the test owns this tree
    try:
        for q, *args in stmts:
            c.execute(q, args)
        c.commit()
    finally:
        c.close()


def clear():
    sql(("DELETE FROM proposals",), ("DELETE FROM datasets",), ("DELETE FROM submissions",),
        ("DELETE FROM judge_runs",), ("DELETE FROM llm_batches",))


def plant_proposal(status="proposed", model=MODEL, topic=TOPIC, task=TASK, **kw):
    from service import db
    pid = db.proposal_create(model, task, topic, "masein",
                             {"n_shown": 3, "diagnose_items": 44, "diagnose_weak": 36,
                              "topic_score_report": 1.46, "judge_id": "stub/overlap-v1"})
    kw.setdefault("spec_text", "the missing skill")
    db.proposal_update(pid, status=status, **kw)
    return pid


def new_page(browser, width, height=900, **kw):
    ctx = browser.new_context(viewport={"width": width, "height": height},
                              reduced_motion="reduce", **kw)
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.errors = errors
    return ctx, pg


# ---------------------------------------------------------------------------
# 1. Propose → opens the dialog, wherever it is
# ---------------------------------------------------------------------------

def judged_row(page, topic=TOPIC):
    return page.locator(f"table[data-judged-topics] tr[data-topic='{topic}']")


@pytest.mark.dashboard
def test_propose_opens_the_dialog_without_leaving_the_model_page(live, page):
    clear()
    try:
        page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
        page.wait_for_selector("table[data-judged-topics]")
        page.wait_for_function("() => state.rv.loaded")
        btn = judged_row(page).locator("[data-propose-link]")
        btn.wait_for()
        assert btn.text_content() == "Propose →"
        assert (btn.get_attribute("title") or "") == ""      # the old one pointed away
        btn.click()
        dlg = page.locator("[data-dialog='propose']")
        dlg.wait_for()
        # filled in, and the page under it did not move
        assert dlg.locator("[data-combobox='model']").get_attribute("data-value") == MODEL
        assert dlg.locator(f"[data-np-topic='{TOPIC}'] input").is_checked()
        assert page.evaluate("location.hash").startswith("#model=")
        shot(page, "11k-1-propose-dialog-1240-light.png")
        page.keyboard.press("Escape")
        page.wait_for_selector("[data-dialog='propose']", state="detached")
        assert page.errors == []
    finally:
        clear()


@pytest.mark.dashboard
def test_a_topic_with_an_open_proposal_says_review_it(live, page):
    clear()
    pid = plant_proposal()
    try:
        page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
        page.wait_for_selector("table[data-judged-topics]")
        link = judged_row(page).locator("[data-review-link]")
        link.wait_for()
        assert link.text_content() == "Review it →"
        assert link.get_attribute("href") == f"#tab=improve&sub=review&read=proposal:{pid}"   # 12b
        assert judged_row(page).locator("[data-propose-link]").count() == 0
        link.click()
        page.wait_for_selector("#reader[data-ready='1']")
        assert page.locator("#reader").get_attribute("data-key") == f"proposal:{pid}"
        # the topic page says the same thing
        page.goto(live["base"] + "/#topic=economics")
        page.wait_for_selector("[data-topic-page='economics']")
        page.wait_for_function("() => state.rv.loaded")
        gate = page.locator("[data-propose='economics']")
        if gate.count():
            assert gate.get_attribute("data-gate") == "open"
            assert gate.text_content() == "Review it →"
        assert page.errors == []
    finally:
        clear()


@pytest.mark.dashboard
def test_a_tooltip_at_the_right_edge_stays_in_the_window(live, page):
    page.set_viewport_size({"width": 1240, "height": 900})
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("table.lb thead th[data-tip]")
    heads = page.locator("table.lb thead th[data-tip]")
    box = heads.nth(heads.count() - 1).bounding_box()      # the rightmost column
    page.mouse.move(box["x"] + box["width"] - 2, box["y"] + box["height"] / 2)
    page.wait_for_function("() => tip.style.opacity === '1'")
    r = page.evaluate("""() => { const b = tip.getBoundingClientRect();
      return { left: b.left, right: b.right, top: b.top, bottom: b.bottom,
               w: innerWidth, h: innerHeight }; }""")
    assert r["right"] <= r["w"] - 8 + 0.5 and r["left"] >= 7.5, r
    assert r["bottom"] <= r["h"] - 8 + 0.5 and r["top"] >= 7.5, r
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2. a run that is still being graded says so
# ---------------------------------------------------------------------------

def grading_row(n_done=240, n=570):
    """A judged run whose answers are in and whose grades are not."""
    from service import db
    sid = db.add(MODEL, "base", "judged", "masein", "", tasks=[TASK])
    bid = "fake_grading_1"
    db.batch_add(bid, "judge", sid, n, "fake", "stub")
    db.judge_run_create(MODEL, bid, n, "stub/overlap-v1", "{}")
    db.batch_progress(bid, f"{n_done}/{n} done")
    # the service's own poller finishes a 'submitted' batch within a second,
    # and this test is about the row while the judge is still working: hold it
    # in flight where the poller does not reach (it takes status='submitted')
    sql(("UPDATE llm_batches SET status='pending' WHERE batch_id=?", bid))
    db.update(sid, status="done", judge_batch=bid,
              progress=f"judge batch {bid} submitted ({n} answers)")
    return sid, bid


@pytest.mark.dashboard
def test_a_run_being_graded_says_grading_and_never_shows_an_empty_cell(live, page):
    from service import db
    clear()
    sid, bid = grading_row()
    try:
        page.set_viewport_size({"width": 1400, "height": 900})
        page.goto(live["base"] + "/#tab=queue")
        row = page.locator(f"tr[data-queue-row='{sid}']")
        row.wait_for()
        chip = row.locator("[data-stage]")
        assert chip.get_attribute("data-stage") == "grading"
        assert chip.text_content() == "grading 240/570"
        assert row.locator(f"[data-grading='{sid}']").text_content() == "Grading… 240/570"
        assert row.locator("[data-row-open]").count() == 0
        assert row.locator("[data-row-menu]").count() == 1      # Log is still there
        row.locator("[data-row-menu]").click()
        assert page.locator("[role=menuitem][data-act='log']").count() == 1
        page.keyboard.press("Escape")
        shot(page, "11k-2-grading-1400-light.png")
        # the grades land: done, and Open results
        db.batch_finish(bid, "done", "")
        page.evaluate("loadQueue()")
        page.wait_for_function("id => document.querySelector(`tr[data-queue-row='${id}'] "
                               "[data-stage]`).dataset.stage === 'done'", arg=sid)
        assert row.locator("[data-stage]").text_content() == "done"
        assert row.locator("[data-row-open]").count() == 1
        assert row.locator("[data-grading]").count() == 0
        assert page.errors == []
    finally:
        clear()


# ---------------------------------------------------------------------------
# 3. the reader's missing skill
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_the_dataset_reader_shows_the_missing_skill(live, page):
    from service import db, proposals as prop
    clear()
    spec = ("Say which way an effect runs, and name the mechanism that carries it, "
            "before giving the number.")
    pid = plant_proposal("approved", spec_text=spec)
    did = db.dataset_create(pid, "doc", 2, "masein", {})
    d = prop.dataset_dir(did)
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.jsonl").write_text(
        json.dumps({"text": "Marginal cost is the cost of one more unit. " * 12}) + "\n",
        encoding="utf-8")
    db.dataset_update(did, status="ready", provenance=json.dumps(
        {"items": {"kept": 1, "requested": 2}, "approved_spec": spec}))
    try:
        page.set_viewport_size({"width": 1400, "height": 900})
        page.goto(live["base"] + f"/#tab=review&view=datasets&read=dataset:{did}")
        page.wait_for_selector("#reader[data-ready='1']")
        det = page.locator(f"[data-dataset-spec='{did}']")
        assert det.locator("summary").text_content() == "The missing skill ▸"
        closed = det.bounding_box()["height"]
        assert closed < 40, closed
        det.locator("summary").click()
        assert spec[:40] in det.text_content()
        shot(page, "11k-3-missing-skill-1400-light.png")
        # a dataset whose provenance predates the frozen spec falls back to the
        # proposal's own words
        db.dataset_update(did, provenance=json.dumps({"items": {"kept": 1, "requested": 2}}))
        page.reload()
        page.wait_for_selector("#reader[data-ready='1']")
        page.locator(f"[data-dataset-spec='{did}'] summary").click()
        assert spec[:40] in page.locator(f"[data-dataset-spec='{did}']").text_content()
        assert page.errors == []
    finally:
        clear()


# ---------------------------------------------------------------------------
# 4 and 5. nothing paints over its card, and nothing slides sideways
# ---------------------------------------------------------------------------

# a table either fits inside its scroller or the scroller clips and scrolls
# it; what it may never do is paint across the card it sits in
TABLES = """() => [...document.querySelectorAll('#view table')].map(t => {
  const sc = t.closest('.lb-wrap') || t.parentElement;
  const tr = t.getBoundingClientRect(), sr = sc.getBoundingClientRect();
  const ov = getComputedStyle(sc).overflowX;
  return { spill: Math.round(tr.right - sr.right), ov: ov, cls: sc.className,
           clips: ov === 'auto' || ov === 'scroll' || ov === 'hidden' || ov === 'clip',
           scrolls: sc.scrollWidth > sc.clientWidth + 1 };
})"""


@pytest.mark.dashboard
@pytest.mark.parametrize("width", WIDTHS)
def test_no_table_paints_over_its_card(live, browser, width):
    from service import db
    clear()
    for i in range(3):
        sid = db.add(MODEL, "base", "judged" if i else "full", "masein", "a note", tasks=[TASK])
        db.update(sid, status="done", progress="2/2 · exam_economics (0-shot) — done")
    pid = plant_proposal("approved")
    did = db.dataset_create(pid, "doc", 20, "masein", {})
    db.dataset_update(did, status="ready",
                      provenance=json.dumps({"items": {"kept": 20, "requested": 20}}))
    ctx, page = new_page(browser, width)
    try:
        for where in ("/#tab=queue", "/#tab=models", "/#tab=review&view=datasets",
                      "/#tab=review", "/#tab=loop"):
            page.goto(live["base"] + where)
            page.wait_for_selector("#view table", state="attached")
            page.wait_for_timeout(400)
            for t in page.evaluate(TABLES):
                assert t["clips"] or t["spill"] <= 1, (where, t)
            wide = page.evaluate("[document.documentElement.scrollWidth, innerWidth]")
            assert wide[0] <= wide[1], (where, wide)
        assert page.errors == []
    finally:
        ctx.close()
        clear()


@pytest.mark.dashboard
def test_no_page_slides_sideways_on_a_phone(live, browser):
    from service import db
    clear()
    pid = plant_proposal()
    did = db.dataset_create(pid, "doc", 20, "masein", {})
    db.dataset_update(did, status="ready",
                      provenance=json.dumps({"items": {"kept": 18, "requested": 20}}))
    sid = db.add(MODEL, "base", "judged", "masein", "", tasks=[TASK])
    db.update(sid, status="running", progress="1/4 · exam_economics (0-shot)")
    ctx, page = new_page(browser, 400, 860)
    try:
        for name, where in PAGES:
            page.goto(live["base"] + where)
            page.wait_for_selector("#view .card, #view table", timeout=20000)
            page.wait_for_timeout(500)
            wide = page.evaluate("[document.documentElement.scrollWidth, innerWidth]")
            assert wide[0] <= wide[1], (name, wide)
        # and with the reader open over it
        page.goto(live["base"] + f"/#tab=review&read=proposal:{pid}")
        page.wait_for_selector("#reader[data-ready='1']")
        wide = page.evaluate("[document.documentElement.scrollWidth, innerWidth]")
        assert wide[0] <= wide[1], wide
        shot(page, "11k-5-phone-review-400-light.png")
        assert page.errors == []
    finally:
        ctx.close()
        clear()


# ---------------------------------------------------------------------------
# 6. "0 ticked · 2 of 5 judged"
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_the_area_heading_counts_ticks_and_judged_topics(live, page):
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    page.locator(f"[data-sit-open='{MODEL}']").click()
    page.wait_for_selector("[data-exam-picker='msit']")
    area = page.locator("[data-panel='msit'] [data-area]").first
    topics = area.locator("[data-exam-topic]")
    judged = area.locator("[data-exam-topic][data-status='judged']").count()
    head = area.locator(".excount")
    assert head.text_content() == f"0 ticked · {judged} of {topics.count()} judged"
    boxes = area.locator("input[data-exam-task]:not([disabled])")
    boxes.first.check()
    assert head.text_content() == f"1 ticked · {judged} of {topics.count()} judged"
    area.locator("[data-area-box]").check()
    assert head.text_content() == (f"{boxes.count()} ticked · {judged} of "
                                   f"{topics.count()} judged")
    shot(page, "11k-6-sit-headings-1240-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 7. the answers list is paged, and each answer is clamped
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_the_answers_list_pages_and_clamps(live, page):
    clear()
    j = api(live["base"], f"/api/judge/justifications?model={MODEL.replace('/', '%2F')}"
                          f"&topic={TOPIC}&limit=60")
    from service import db
    pid = db.proposal_create(MODEL, TASK, TOPIC, "masein",
                             {"n_shown": len(j["items"]), "diagnose_items": 39,
                              "diagnose_weak": 6, "topic_score_report": 1.46,
                              "qids_read": [it["qid"] for it in j["items"]]})
    db.proposal_update(pid, status="proposed", spec_text="the missing skill")
    try:
        page.set_viewport_size({"width": 1400, "height": 900})
        page.goto(live["base"] + f"/#tab=review&read=proposal:{pid}")
        page.wait_for_selector("#reader[data-ready='1']")
        det = page.locator(f"[data-answers-read='{pid}']")
        det.locator("summary").click()
        page.wait_for_selector("[data-answers-count]")
        shown = det.locator("[data-answer-qid]")
        n = api(live["base"], f"/api/proposals/{pid}/answers")["n"]
        assert shown.count() == min(10, n)
        # each answer is clamped until it is asked for; the question and the
        # judge's comment are whole
        one = shown.first
        text = one.locator("[data-answer-text]")
        assert "clamp3" in (text.get_attribute("class") or "") or \
            len(text.text_content()) <= 200
        if "clamp3" in (text.get_attribute("class") or ""):
            box = text.bounding_box()
            assert box["height"] < 90, box
            one.locator("[data-answer-toggle]").click()
            assert "clamp3" not in (one.locator("[data-answer-text]").get_attribute("class") or "")
            assert one.locator("[data-answer-toggle]").text_content() == "Show less"
        assert det.locator(".rd-q").first.text_content().startswith("The question.")
        shot(page, "11k-7-answers-paged-1400-light.png")
        assert page.errors == []
    finally:
        clear()


# ---------------------------------------------------------------------------
# 8 and 9. the ring, and a list row
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_a_mouse_click_leaves_no_ring_and_a_tab_does(live, page):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/")
    pill = page.locator("#warnings summary[data-warn-summary]")
    pill.wait_for()
    pill.click()                                   # opens
    page.wait_for_timeout(200)
    pill.click()                                   # and closes, focus comes back
    page.wait_for_timeout(200)
    ring = page.evaluate("""() => { const a = document.activeElement;
      const cs = getComputedStyle(a);
      return { tag: a.tagName, width: cs.outlineWidth, style: cs.outlineStyle,
               noring: a.dataset ? a.dataset.noring : null }; }""")
    assert ring["style"] == "none" or ring["width"] == "0px", ring
    shot(page, "11k-8-no-ring-1400-light.png")
    # the keyboard still gets one
    page.keyboard.press("Tab")
    page.keyboard.press("Shift+Tab")
    ring = page.evaluate("""() => { const a = document.activeElement;
      const cs = getComputedStyle(a);
      return { width: cs.outlineWidth, style: cs.outlineStyle,
               visible: a.matches(':focus-visible') }; }""")
    assert ring["visible"] and ring["style"] == "solid" and ring["width"] == "2px", ring
    assert page.errors == []


@pytest.mark.dashboard
def test_a_list_row_is_not_a_text_field(live, page):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("#pill-weak")
    page.locator("#pill-weak").click()
    page.wait_for_selector("#pop-weak")
    rows = page.evaluate("""() => [...document.querySelectorAll('#pop-weak [role=menuitem]')]
      .map(o => { const cs = getComputedStyle(o);
        return { text: o.textContent, current: o.getAttribute('aria-current'),
                 border: [cs.borderTopWidth, cs.borderRightWidth, cs.borderBottomWidth,
                          cs.borderLeftWidth].join(' '),
                 deco: cs.textDecorationLine, bg: cs.backgroundColor,
                 tick: getComputedStyle(o, '::after').content }; })""")
    assert rows, "the judged-model list is empty"
    for r in rows:
        assert r["border"] == "0px 0px 0px 0px", r
        assert r["deco"] == "none", r
    chosen = [r for r in rows if r["current"] == "true"]
    assert len(chosen) == 1, rows
    assert "✓" in chosen[0]["tick"], chosen[0]
    assert chosen[0]["bg"] not in ("rgba(0, 0, 0, 0)", "transparent"), chosen[0]
    shot(page, "11k-9-model-list-1400-light.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# screenshots
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, browser, theme):
    from service import db
    clear()
    sid, bid = grading_row()
    pid = plant_proposal()
    did = db.dataset_create(pid, "doc", 20, "masein", {})
    db.dataset_update(did, status="ready",
                      provenance=json.dumps({"items": {"kept": 20, "requested": 20},
                                             "approved_spec": "Name the mechanism."}))
    for width in (1400, 400):
        ctx, page = new_page(browser, width, 1000)
        try:
            for name, where in (("queue", "/#tab=queue"),
                                ("model", "/#model=" + MODEL.replace("/", "%2F")),
                                ("review", "/#tab=review&view=datasets")):
                page.goto(live["base"] + where)
                page.wait_for_selector("#view .card")
                page.wait_for_timeout(500)
                page.evaluate(f"applyTheme('{theme}')")
                page.wait_for_timeout(150)
                shot(page, f"11k-{name}-{width}-{theme}.png")
            assert page.errors == []
        finally:
            ctx.close()
    clear()
