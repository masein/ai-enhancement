"""11j: the Review tab, rebuilt. 12g.1: the tab is Improve's pipeline for
one model now; its proposal card, generating and the dataset reader are the
same, reached from the pipeline's Proposals and Training data.

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


def open_review(page, base):
    """12g.1: Improve's pipeline, on the model these proposals are for"""
    page.goto(base + "/#tab=improve&sub=model&model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector(f"[data-pipeline='{MODEL}']")
    page.wait_for_function("() => state.rv.loaded")
    return page.locator(f"[data-pipeline='{MODEL}']")


# ---------------------------------------------------------------------------
# the four views
# ---------------------------------------------------------------------------

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
        # 12g.1: the pipeline's Propose, with its model filled in
        page.locator("[data-imp-propose]").click()
        dlg = page.locator("[data-dialog='propose']")
        dlg.wait_for()
        assert dlg.locator("[data-combobox='model']").get_attribute("data-value") == MODEL
        page.wait_for_selector("[data-np-topic]")
        why = {t.get_attribute("data-np-topic"): t.get_attribute("data-np-why")
               for t in dlg.locator("[data-np-topic]").all()}
        assert why["Law"] and "under 30 hidden questions" in why["Law"]
        assert why["Sociology"] == f"proposal #{open_} is open"
        assert why["Arts"] == "not sat yet"
        assert why[TOPIC] is None                                # this one can be proposed
        assert dlg.locator("[data-np-topic='Law'] input").is_disabled()
        # it opens on the weakest topic a proposal can be made from
        assert dlg.locator(f"[data-np-topic='{TOPIC}'] input").is_checked()
        assert dlg.locator("[data-dialog-go]").is_enabled()
        shot(page, "11j-new-proposal-1400-light.png")
        dlg.locator("[data-dialog-go]").click()
        page.wait_for_selector("[data-toast='propose']")
        made = [p for p in api(live["base"], "/api/proposals") if p["id"] != open_]
        assert len(made) == 1
        assert made[0]["model"] == MODEL and made[0]["category"] == TOPIC
        assert made[0]["requested_by"] == "masein"
        # and it is in the pipeline's Proposals
        page.locator(f"[data-prop='{made[0]['id']}']").wait_for()
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
    page.locator(f"[data-prop-act='{p['id']}']").click()
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
def test_the_list_holds_all_of_them_ten_at_a_time(live, page):
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
        # 11k: ten at a time, and the pager reaches the rest — the whole list
        # in one scroll was 14,000px
        assert det.locator("[data-answer-qid]").count() == 10
        pager = det.locator("[data-pager='rv-answers']")
        assert pager.count() == 1
        assert pager.locator("[data-page-range]").get_attribute("data-page-range") == "1-10"
        seen = set(det.locator("[data-answer-qid]").evaluate_all(
            "xs => xs.map(x => x.dataset.answerQid)"))
        while pager.locator("[data-page-next]").is_enabled():
            pager.locator("[data-page-next]").click()
            seen |= set(det.locator("[data-answer-qid]").evaluate_all(
                "xs => xs.map(x => x.dataset.answerQid)"))
        assert len(seen) == read["n"] == len({str(i["qid"]) for i in read["items"]})
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
def test_what_the_reader_opened_stays_open_through_a_poll(live, page):
    """11m: masein opened Details and it closed itself a couple of seconds
    later — the card was repainted on every poll of the Review tab, and the
    disclosure came back closed."""
    clear_proposals()
    pid = plant("proposed")
    try:
        page.goto(live["base"] + f"/#tab=review&read=proposal:{pid}")
        page.wait_for_selector("#reader[data-ready='1']")
        det = page.locator(f"[data-rv-details='{pid}']")
        det.locator("summary").click()
        assert det.evaluate("e => e.open")
        answers = page.locator(f"[data-answers-read='{pid}']")
        answers.locator("summary").click()
        page.wait_for_selector("[data-answers-note]")
        # a poll of the tab behind it, and then a repaint of the card itself
        page.wait_for_timeout(6000)
        page.evaluate("loadReview()")
        page.wait_for_timeout(400)
        assert page.locator(f"[data-rv-details='{pid}']").evaluate("e => e.open"), \
            "Details closed itself"
        assert page.locator(f"[data-answers-read='{pid}']").evaluate("e => e.open")
        page.evaluate("() => readFetch(state.read)")
        page.wait_for_timeout(500)
        assert page.locator(f"[data-rv-details='{pid}']").evaluate("e => e.open"), \
            "Details closed when the card was rebuilt"
        assert page.errors == []
    finally:
        clear_proposals()


@pytest.mark.dashboard
def test_generate_closes_the_card_and_lands_on_the_new_dataset(live, page):
    """11m: pressing Generate is the end of this card's work — the sheet
    closes and the Datasets view opens on the row being written. It used to
    leave the card open, saying "#9 Waiting for the AI"."""
    clear_proposals()
    pid = plant("approved")
    try:
        # the name goes in first: the sheet covers the header
        open_review(page, live["base"])
        set_name(page, "masein")
        page.goto(live["base"] + f"/#tab=review&read=proposal:{pid}")
        page.wait_for_selector("#reader[data-ready='1']")
        page.locator(f"[data-generate='{pid}']").click()
        page.wait_for_selector("#reader", state="detached")
        assert page.evaluate("location.hash").startswith(
            "#tab=improve&sub=model&model=" + MODEL.replace("/", "%2F"))
        assert "read=" not in page.evaluate("location.hash")
        assert page.locator("[data-stage='data']").count() == 1
        toast = page.locator("[data-toast='review']")
        toast.wait_for()
        did = int(re.search(r"Dataset #(\d+)", toast.text_content()).group(1))
        row = page.locator(f"[data-ds-item='{did}']")
        row.wait_for()
        assert "landed" in (row.get_attribute("class") or "")
        assert "being written" in row.text_content() or "of" in row.text_content()
        # Back returns to the card it came from
        page.go_back()
        page.wait_for_selector("#reader[data-ready='1']")
        assert page.locator("#reader").get_attribute("data-key") == f"proposal:{pid}"
        assert page.errors == []
    finally:
        clear_proposals()


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
        row = page.locator(f"[data-prop='{pid}']")
        assert row.locator("[data-demo-only]").count() == 1
        page.locator(f"[data-prop-act='{pid}']").click()
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
        open_review(page, live["base"])
        row = page.locator(f"[data-ds-item='{did}']")
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
def test_the_pipeline_is_short_with_five_proposals_and_eight_datasets(live, browser):
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
        # 12g.1: one page for all of them — five items a stage, then "+ n more"
        open_review(page, live["base"])
        page.wait_for_timeout(300)
        tall = page.evaluate("document.documentElement.scrollHeight")
        assert tall <= 1600, tall
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert errors == []
    finally:
        ctx.close()
        clear_proposals()
