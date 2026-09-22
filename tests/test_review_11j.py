"""11j: the Review tab, rebuilt.

masein: "If I want to start a new review, or want to review the pending ones,
the approved ones and the generated datasets — I think the UX is bad." Four
views with their counts, one line per row, + New proposal from the tab itself,
and a proposal that opens as a short card in the reader's sheet: the missing
skill, one line of why (the count that used to be blank), every practice
answer the AI read — with its question, for the reviewer — the focus plan as
chips, and one Demo only badge instead of four warning boxes.

No hidden question's text or qid reaches any of it.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import pytest

import exam_build as eb
from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11j"
MODEL = "fx/good-750m"
TOPIC = "Economics"
TASK = "exam_economics"
E2E_MS = 90000


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def propose(base, model=MODEL, topic=TOPIC, who="masein"):
    """A real proposal, answered by the fake AI: the evidence, the qids it
    read and the spec are the ones the card shows."""
    j = api(base, "/api/proposals", {"model": model, "topic": topic, "requested_by": who})
    for _ in range(int(E2E_MS / 500)):
        p = api(base, f"/api/proposals/{j['id']}")
        if p["status"] != "pending":
            return p
        time.sleep(0.5)
    raise AssertionError("the proposal never came back")


PLANTED = []


def plant(status, topic=TOPIC, task=TASK, model=MODEL, **kw):
    """A proposal straight into the database, for the views' counts: the ones
    the AI answered are made by propose() above."""
    from service import db
    ev = {"n_shown": 3, "diagnose_items": 44, "diagnose_weak": 36,
          "topic_score_report": 1.46, "topic_n_report": 56, "judge_id": "stub/overlap-v1",
          **kw.pop("evidence", {})}
    pid = db.proposal_create(model, task, topic, kw.pop("requested_by", "masein"), ev)
    db.proposal_update(pid, status=status, spec_text=kw.pop("spec_text", "the missing skill"),
                       **kw)
    PLANTED.append(pid)
    return pid


def sql(*stmts):
    """The test owns this tree: it plants rows and clears them again."""
    from service import db
    c = db._conn()                            # noqa: SLF001
    try:
        for q, *args in stmts:
            c.execute(q, args)
        c.commit()
    finally:
        c.close()


def clear_proposals():
    for pid in PLANTED:
        sql(("DELETE FROM datasets WHERE proposal_id = ?", pid),
            ("DELETE FROM proposals WHERE id = ?", pid))
    PLANTED.clear()


def views(page):
    return {b.get_attribute("data-rv-view"): b.text_content()
            for b in page.locator("[data-rv-view]").all()}


def counts(page):
    return {k: int(re.search(r"\((\d+)\)", v).group(1)) for k, v in views(page).items()}


def open_review(page, base, view=""):
    page.goto(base + "/#tab=review" + (f"&view={view}" if view else ""))
    page.wait_for_selector("[data-review-head]")
    page.wait_for_function("() => state.rv.loaded")
    return page.locator("[data-review-head]")


# ---------------------------------------------------------------------------
# the four views
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_the_four_views_hold_the_right_rows_and_their_counts(live, page):
    from service import db
    clear_proposals()
    a = plant("proposed")
    b = plant("approved", topic="Law", task="exam_law")
    c = plant("rejected", topic="Physics & Astronomy", task="exam_physics_astronomy",
              reject_reason="names a fact, not a skill")
    d = plant("pending", topic="Sociology", task="exam_sociology")
    did = db.dataset_create(b, "doc", 20, "masein", {})
    db.dataset_update(did, status="ready",
                      provenance=json.dumps({"items": {"kept": 18, "requested": 20,
                                                       "missing": [{"why": "too short"},
                                                                   {"why": "not JSON"}]}}))
    try:
        open_review(page, live["base"])
        assert counts(page) == {"review": 2, "ready": 1, "datasets": 1, "history": 1}
        # it opens on what is waiting
        assert page.locator("[data-rv-view='review'][aria-selected='true']").count() == 1
        rows = page.locator("[data-rv-row]")
        assert sorted(int(r.get_attribute("data-rv-row")) for r in rows.all()) == sorted([a, d])
        assert "Waiting for the AI" in page.locator(f"[data-rv-status='{d}']").text_content()
        assert "To review" in page.locator(f"[data-rv-status='{a}']").text_content()
        page.locator("[data-rv-view='ready']").click()
        assert [r.get_attribute("data-rv-row") for r in page.locator("[data-rv-row]").all()] \
            == [str(b)]
        page.locator("[data-rv-view='history']").click()
        assert [r.get_attribute("data-rv-row") for r in page.locator("[data-rv-row]").all()] \
            == [str(c)]
        page.locator("[data-rv-view='datasets']").click()
        row = page.locator(f"[data-ds-row='{did}']")
        assert row.count() == 1
        assert row.locator("[data-doc-line]").text_content() == "18 of 20 · 2 missing"
        assert row.locator("[data-ds-read]").count() == 1
        assert row.locator("[data-ds-flag]").count() == 1
        assert page.errors == []
    finally:
        clear_proposals()


@pytest.mark.dashboard
def test_the_view_is_in_the_hash_and_survives_a_poll(live, page):
    clear_proposals()
    plant("proposed")
    try:
        open_review(page, live["base"], "datasets")
        assert page.locator("[data-rv-view='datasets'][aria-selected='true']").count() == 1
        assert page.locator("[data-rv-list='datasets']").count() == 1
        page.evaluate("loadReview()")
        page.wait_for_timeout(600)
        assert page.locator("[data-rv-view='datasets'][aria-selected='true']").count() == 1
        # a click writes it into the address, and Back goes where it came from
        page.locator("[data-rv-view='ready']").click()
        assert "view=ready" in page.evaluate("location.hash")
        page.go_back()
        page.wait_for_selector("[data-rv-list='datasets']")
        # with nothing waiting it opens on the datasets
        clear_proposals()
        page.goto(live["base"] + "/#tab=review")
        page.evaluate("loadReview()")
        page.wait_for_function("() => document.querySelector(\"[data-rv-view='datasets']\")"
                               ".getAttribute('aria-selected') === 'true'")
        assert page.errors == []
    finally:
        clear_proposals()


# ---------------------------------------------------------------------------
# + New proposal
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_new_proposal_disables_each_blocked_topic_with_its_reason(live, page):
    clear_proposals()
    open_ = plant("proposed", topic="Sociology", task="exam_sociology")
    try:
        open_review(page, live["base"])
        set_name(page, "masein")
        page.locator("[data-new-proposal]").click()
        dlg = page.locator("[data-dialog='propose']")
        dlg.wait_for()
        assert dlg.locator("[data-np-topic]").count() == 0      # a model first
        dlg.locator("[data-combobox='model']").click()
        page.locator(f"[role=option][data-value='{MODEL}']").click()
        page.wait_for_selector("[data-np-topic]")
        why = {t.get_attribute("data-np-topic"): t.get_attribute("data-np-why")
               for t in dlg.locator("[data-np-topic]").all()}
        assert why["Law"] and "under 30 hidden questions" in why["Law"]
        assert why["Sociology"] == f"proposal #{open_} is open"
        assert why["Arts"] == "not sat yet"
        assert why[TOPIC] is None                                # this one can be proposed
        assert dlg.locator("[data-np-topic='Law'] input").is_disabled()
        assert dlg.locator("[data-dialog-go]").is_disabled()     # nothing picked yet
        dlg.locator(f"[data-np-topic='{TOPIC}'] input").check()
        assert dlg.locator("[data-dialog-go]").is_enabled()
        shot(page, "11j-new-proposal-1400-light.png")
        dlg.locator("[data-dialog-go]").click()
        page.wait_for_selector("[data-toast='propose']")
        made = [p for p in api(live["base"], "/api/proposals") if p["id"] != open_]
        assert len(made) == 1
        assert made[0]["model"] == MODEL and made[0]["category"] == TOPIC
        assert made[0]["requested_by"] == "masein"
        # and the tab came back on To review, with it in the list
        assert page.locator(f"[data-rv-row='{made[0]['id']}']").count() == 1
        assert page.errors == []
    finally:
        # the one this test asked the AI for goes too: an open proposal on a
        # topic is exactly what stops the next one
        sql(("DELETE FROM proposals WHERE model = ? AND category = ?", MODEL, TOPIC))
        clear_proposals()


@pytest.mark.dashboard
def test_the_topic_pages_propose_opens_the_same_dialog_filled_in(live, page):
    clear_proposals()
    try:
        page.goto(live["base"] + "/#topic=economics")
        page.wait_for_selector("[data-topic-page='economics']")
        # the topic page's Propose… hands the dialog its model and topic
        page.evaluate("() => npDialog({ model: 'fx/good-750m', topic: 'Economics' })")
        dlg = page.locator("[data-dialog='propose']")
        dlg.wait_for()
        assert dlg.locator("[data-combobox='model']").get_attribute("data-value") == MODEL
        assert dlg.locator(f"[data-np-topic='{TOPIC}'] input").is_checked()
        page.keyboard.press("Escape")
        assert dlg.count() == 0
        assert page.errors == []
    finally:
        clear_proposals()


# ---------------------------------------------------------------------------
# the card in the sheet
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def proposed(live):
    """One real proposal, made by the API and answered by the fake AI."""
    p = propose(live["base"])
    yield p
    sql(("DELETE FROM proposals WHERE id = ?", p["id"]))


@pytest.mark.dashboard
def test_the_card_says_why_in_one_line_with_the_count_that_was_blank(live, page, proposed):
    p = proposed
    ev = p["evidence"]
    assert ev["diagnose_weak"] and ev["diagnose_items"]          # the fixture has both
    open_review(page, live["base"])
    page.locator(f"[data-rv-row='{p['id']}'] [data-rv-open]").click()
    page.wait_for_selector("#reader[data-ready='1']")
    sheet = page.locator("#reader")
    assert sheet.locator(".rd-title").text_content() == f"{TOPIC} · good-750m"
    assert f"#{p['id']}" in sheet.locator(".rd-src").text_content()
    line = sheet.locator(f"[data-why-line='{p['id']}']").text_content()
    score = f"{ev['topic_score_report']:.2f}".rstrip("0").rstrip(".")   # the board's format
    assert line == (f"{ev['diagnose_weak']} of {ev['diagnose_items']} practice answers scored "
                    f"below 3 of 4 · {TOPIC} score {score} / 4 (hidden questions)")
    assert "—" not in line
    assert sheet.locator(f"[data-spec-edit='{p['id']}']").input_value() == p["spec_text"]
    # the plan as chips, not a paragraph — or, with nothing to spread over, why
    plan = sheet.locator(f"[data-focus-plan='{p['id']}']")
    page.wait_for_function("id => { const b = document.querySelector(`[data-focus-plan='${id}']`);"
                           " return b && !/loading/.test(b.textContent); }", arg=str(p["id"]))
    assert plan.locator(".chip-static").count() >= 1 or "Not spread" in plan.text_content()
    shot(page, "11j-card-1400-light.png")
    assert page.errors == []


def many_read(base):
    """A model and topic the judge wrote more than eight comments about: the
    card lists every one, where the old one stopped at eight."""
    for model in ("fx/chance-160m", "fx/skewed-360m", "fx/good-750m"):
        for topic in ("Economics", "Law", "Sociology", "Philosophy", "Medicine & Clinical Health",
                      "History & Archaeology", "Education", "Business & Management"):
            j = api(base, f"/api/judge/justifications?model={model.replace('/', '%2F')}"
                          f"&topic={topic.replace(' ', '%20').replace('&', '%26')}&limit=60")
            if len(j["items"]) > 8:
                return model, topic, j
    raise AssertionError("no topic in the fixture has more than eight comments")


@pytest.mark.dashboard
def test_the_answers_it_read_lists_every_one_with_its_practice_question(live, page, proposed):
    p = proposed
    read = api(live["base"], f"/api/proposals/{p['id']}/answers")
    assert read["n"] == p["evidence"]["n_shown"]
    page.goto(live["base"] + f"/#tab=review&read=proposal:{p['id']}")
    page.wait_for_selector("#reader[data-ready='1']")
    sheet = page.locator("#reader")
    det = sheet.locator(f"[data-answers-read='{p['id']}']")
    assert det.locator("summary").text_content() == f"The answers it read ({read['n']}) ▸"
    det.locator("summary").click()
    page.wait_for_selector("[data-answers-count]")
    items = det.locator("[data-answer-qid]")
    assert items.count() == read["n"]                    # all of them, not eight
    first = read["items"][0]
    one = det.locator(f"[data-answer-qid='{first['qid']}']")
    assert first["question"][:40] in one.text_content()
    assert first["answer"][:40] in one.text_content()
    assert f"Scored {first['score']} of 4" in one.text_content()
    assert first["comment"][:40] in one.text_content()
    assert "wording taken out" in det.text_content()
    assert page.errors == []


@pytest.mark.dashboard
def test_the_list_holds_all_of_them_where_the_old_card_stopped_at_eight(live, page):
    import exam_build as _eb
    model, topic, j = many_read(live["base"])
    pid = plant("proposed", topic=topic, task=_eb.topic_task(topic), model=model,
                evidence={"n_shown": len(j["items"]),
                          "qids_read": [it["qid"] for it in j["items"]],
                          "diagnose_items": j["counts"]["diagnose_items"],
                          "diagnose_weak": j["counts"]["diagnose_weak"]})
    try:
        read = api(live["base"], f"/api/proposals/{pid}/answers")
        assert read["n"] == len(j["items"]) > 8
        page.goto(live["base"] + f"/#tab=review&read=proposal:{pid}")
        page.wait_for_selector("#reader[data-ready='1']")
        det = page.locator(f"[data-answers-read='{pid}']")
        assert det.locator("summary").text_content() == f"The answers it read ({read['n']}) ▸"
        det.locator("summary").click()
        page.wait_for_selector("[data-answers-count]")
        assert det.locator("[data-answer-qid]").count() == read["n"]
        assert page.errors == []
    finally:
        clear_proposals()


@pytest.mark.dashboard
def test_no_hidden_question_reaches_the_card_or_its_answers(live, page, proposed):
    """Every response the page receives while the card and the list open."""
    rows = [r for rs in eb.load_bank(Path(live["root"]) / "exam").values() for r in rs]
    hidden = [r for r in rows if eb.half_of(r["qid"]) == "report"]
    # whole questions, not the first eighty characters: the fixture writes
    # both halves from the same templates, so a prefix matches a practice
    # question too and would fail on text the page is allowed to show
    practice = {str(r.get("prompt") or "") for r in rows if eb.half_of(r["qid"]) == "diagnose"}
    texts = [str(r["prompt"]) for r in hidden if r.get("prompt")
             and str(r["prompt"]) not in practice]
    qids = [r["qid"] for r in hidden]
    assert texts and qids
    bodies = []

    def keep(resp):
        try:
            if resp.request.resource_type in ("fetch", "xhr", "document"):
                bodies.append((resp.url, resp.text()))
        except Exception:
            pass
    page.on("response", keep)
    page.goto(live["base"] + f"/#tab=review&read=proposal:{proposed['id']}")
    page.wait_for_selector("#reader[data-ready='1']")
    page.locator(f"[data-answers-read='{proposed['id']}'] summary").click()
    page.wait_for_selector("[data-answers-count]")
    page.wait_for_timeout(400)
    assert len(bodies) >= 3
    for url, body in bodies:
        assert not [t for t in texts if t in body], url
        assert not [q for q in qids if q in body], url
    assert page.errors == []


@pytest.mark.dashboard
def test_one_demo_only_badge_and_no_repeated_warning(live, page):
    clear_proposals()
    pid = plant("proposed", override=json.dumps(
        {"by": "masein", "at": time.time(),
         "reasons": ["the judge is not calibrated", "a local model serves the judge",
                     "the same provider wrote and graded"]}),
        evidence={"provisional": True, "provisional_reason": "graded by a local model",
                  "served_model": "chat", "base_url": "http://localhost:8000/v1"})
    try:
        open_review(page, live["base"])
        row = page.locator(f"[data-rv-row='{pid}']")
        assert row.locator("[data-demo-only]").count() == 1
        page.locator(f"[data-rv-row='{pid}'] [data-rv-open]").click()
        page.wait_for_selector("#reader[data-ready='1']")
        sheet = page.locator("#reader")
        badges = sheet.locator("[data-demo-only]")
        assert badges.count() == 1
        tip = badges.get_attribute("title")
        for words in ("no person has checked the judge yet", "the judge is a small local AI",
                      "the same AI did more than one step"):
            assert words in tip
        assert sheet.locator(".warn").count() == 0
        # the server's own sentences are the record, in Details
        sheet.locator(f"[data-rv-details='{pid}'] summary").click()
        assert "a local model serves the judge" in sheet.text_content()
        shot(page, "11j-demo-only-1400-light.png")
        assert page.errors == []
    finally:
        clear_proposals()


# ---------------------------------------------------------------------------
# dataset rows, and the height of the tab
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_dataset_rows_read_and_hand_to_training(live, page):
    from service import db
    clear_proposals()
    pid = plant("approved")
    from service import proposals as prop
    did = db.dataset_create(pid, "doc", 2, "masein", {})
    d = prop.dataset_dir(did)
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.jsonl").write_text(
        json.dumps({"text": "Marginal cost is the cost of one more unit. " * 12}) + "\n"
        + json.dumps({"text": "Elasticity measures how quantity answers price. " * 12}) + "\n",
        encoding="utf-8")
    db.dataset_update(did, status="ready",
                      provenance=json.dumps({"items": {"kept": 2, "requested": 2}}))
    try:
        open_review(page, live["base"], "datasets")
        row = page.locator(f"[data-ds-row='{did}']")
        row.wait_for()
        assert row.locator("[data-doc-line]").text_content() == "2 of 2"
        row.locator("[data-ds-flag]").click()
        page.wait_for_selector("[data-toast='copy']")
        assert f"--gap-dataset {did}" in page.locator("[data-toast='copy']").text_content()
        assert page.locator("[data-ds-flag]").first.get_attribute("title").startswith(
            "Pass this to your training run.")
        row.locator("[data-ds-read]").click()
        page.wait_for_selector("#reader[data-ready='1']")
        assert page.locator("#reader").get_attribute("data-kind") == "dataset"
        assert "Marginal cost" in page.locator("#reader").text_content()
        assert page.errors == []
    finally:
        clear_proposals()


@pytest.mark.dashboard
def test_the_tab_is_short_with_five_proposals_and_eight_datasets(live, browser):
    from service import db
    clear_proposals()
    for i in range(5):
        plant("proposed" if i % 2 else "approved")
    pid = plant("approved")
    for _ in range(8):
        did = db.dataset_create(pid, "doc", 20, "masein", {})
        db.dataset_update(did, status="ready",
                          provenance=json.dumps({"items": {"kept": 20, "requested": 20}}))
    ctx = browser.new_context(viewport={"width": 1512, "height": 900}, reduced_motion="reduce")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        for view in ("review", "ready", "datasets"):
            open_review(page, live["base"], view)
            page.wait_for_timeout(300)
            tall = page.evaluate("document.documentElement.scrollHeight")
            assert tall <= 1600, (view, tall)
            assert page.evaluate(
                "document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert errors == []
    finally:
        ctx.close()
        clear_proposals()


# ---------------------------------------------------------------------------
# screenshots
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, browser, proposed, theme):
    from service import db
    pid = plant("approved", spec_text="Say which way an effect runs, and name the mechanism.")
    did = db.dataset_create(pid, "doc", 20, "masein", {})
    db.dataset_update(did, status="ready",
                      provenance=json.dumps({"items": {"kept": 20, "requested": 20}}))
    for width in (1400, 400):
        ctx = browser.new_context(viewport={"width": width, "height": 1000},
                                  reduced_motion="reduce")
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            open_review(page, live["base"], "review")
            page.evaluate(f"applyTheme('{theme}')")
            page.wait_for_timeout(200)
            shot(page, f"11j-review-{width}-{theme}.png")
            open_review(page, live["base"], "datasets")
            page.evaluate(f"applyTheme('{theme}')")
            page.wait_for_timeout(200)
            shot(page, f"11j-datasets-{width}-{theme}.png")
            page.goto(live["base"] + f"/#tab=review&read=proposal:{proposed['id']}")
            page.wait_for_selector("#reader[data-ready='1']")
            page.evaluate(f"applyTheme('{theme}')")
            page.locator(f"[data-answers-read='{proposed['id']}'] summary").click()
            page.wait_for_selector("[data-answers-count]")
            page.wait_for_timeout(200)
            shot(page, f"11j-card-{width}-{theme}.png")
            assert errors == []
        finally:
            ctx.close()
    sql(("DELETE FROM datasets WHERE id = ?", did), ("DELETE FROM proposals WHERE id = ?", pid))
