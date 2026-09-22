"""The Review tab and the propose buttons, against a live service with the
fake LLM: the LIVE dashboard fetches, so this needs a real HTTP server."""

from __future__ import annotations

import contextlib
import json
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import all_rows, choose, go_tab, set_name


pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens"


def model_url(base, mid):
    return f"{base}/#model={quote(mid, safe='')}"


def open_mmlu(pg, base, mid):
    pg.goto(model_url(base, mid))
    pg.wait_for_selector("details.dx")
    det = pg.locator("details.dx", has=pg.locator(".dxname", has_text=re.compile(r"^mmlu[^_]"))).first
    det.locator("> summary").click()
    return det


def open_topics(pg, base, mid):
    """The Judged card's per-topic table — where the loop's action lives."""
    pg.goto(model_url(base, mid))
    pg.wait_for_selector(".card h2:has-text('Judged free response')")
    return pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))


def test_propose_buttons_carry_their_reasons(live, page):
    """The action is on the exam topic, and every refusal says why on the row.
    Phase 9a: one entry point, with the reason in words beside it. 11k: that
    entry point is the New proposal dialog, opened where the person is."""
    base = live["base"]
    card = open_topics(page, base, "fx/good-750m")
    econ = card.locator("tr[data-topic='Economics']")
    assert econ.locator(".propose").count() == 1
    assert econ.locator(".propwhy").count() == 0
    law = card.locator("tr[data-topic='Law']")
    assert "under the 30" in law.locator(".propwhy").first.text_content()
    assert "no proposal" in law.locator(".propwhy").first.text_content()
    # a model that wrote the same sentence every time has revealed no topic gap
    card = open_topics(page, base, "fx/chance-160m")
    econ = card.locator("tr[data-topic='Economics']")
    assert "same answer on nearly every question" in econ.locator(".propwhy").first.text_content()
    # MMLU's finding for the same category rides along as a caution, not a gate
    card = open_topics(page, base, "fx/skewed-360m")
    econ = card.locator("tr[data-topic='Economics']")
    assert econ.locator(".propose").count() == 1
    assert "caution — MMLU for this category" in econ.text_content()
    assert "answer positions" in econ.locator(".propwhy").first.text_content()
    # 11k: it opens the dialog here, with this model and topic in it, instead
    # of sending the person to a topic page with nothing to do
    econ.locator(".propose").click()
    dlg = page.locator("[data-dialog='propose']")
    dlg.wait_for()
    assert dlg.locator("[data-combobox='model']").get_attribute("data-value") == "fx/skewed-360m"
    assert dlg.locator("[data-np-topic='Economics'] input").is_checked()
    assert page.evaluate("location.hash").startswith("#model=")
    page.keyboard.press("Escape")
    page.wait_for_selector("[data-dialog='propose']", state="detached")
    # and the Diagnose section no longer offers one: MMLU does not pick the topic
    det = open_mmlu(page, base, "fx/good-750m")
    assert det.locator("button.propose, a.propose").count() == 0
    assert page.errors == []


def test_exam_curation_in_the_browser(live, page):
    """The Exam tab: the bank per topic, candidates awaiting a decision, accept
    with an edit under a name, reject with a reason, rebuild the tasks."""
    import exam_build as eb
    from service import llm
    base, root = live["base"], live["root"]
    out = eb.draft(root / "exam", llm.FakeBatches("fake-exam", root),
                   ["Law", "History & Archaeology"], per_topic=2, wait=True, poll_s=0)
    assert out["written"] == {"Law": 2, "History & Archaeology": 2}
    page.goto(base + "/#tab=exam")
    page.wait_for_selector(".card h2:has-text('Exam')")
    page.wait_for_selector(".rv[data-candidate]", timeout=15000)
    text = page.locator("#view").text_content()
    assert "fake/fake-exam" in text and "awaiting curation 4" in text
    # "By topic" is the rubrics table now (9c-4): every topic with questions
    # has its row there, and clicking the topic filters the list below
    page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
    rows = page.locator("[data-panel='rubrics'] tr[data-rubric-row]")
    # one per topic in categories.yaml that has questions: all 37 but Arts,
    # delivered empty, which folds into the one row that names it (9b-7)
    all_rows(page, "rubrics")                        # 25 a page since 36 topics (10c)
    assert rows.count() == len(eb.TOPICS) - 1 == 36
    rtext = page.locator("[data-panel='rubrics']").text_content()
    assert "General & Multidisciplinary" in rtext and "Economics" in rtext
    # filter to one topic by clicking it
    page.locator("[data-panel='rubrics'] a[data-filter-topic='Law']").click()
    page.wait_for_selector(".card h2:has-text('Awaiting curation — Law')")
    # the list is dropped with the filter and re-fetched, so wait for it to
    # land rather than counting whatever is on screen this frame
    page.wait_for_function("document.querySelectorAll('.rv[data-candidate]').length === 2")
    cards = page.locator(".rv[data-candidate]")
    assert cards.count() == 2
    card = cards.first
    cid = card.get_attribute("data-candidate")
    q = card.get_by_label("question")
    q.fill(q.input_value() + " Give one counterexample.")
    set_name(page, "Omar")                               # the one name, in the header
    card = page.locator(f".rv[data-candidate='{cid}']")
    card.get_by_role("button", name="Accept into the bank").click()
    page.wait_for_selector("[data-toast='curate']")                 # a toast says so (9c-1)
    assert "Accepted into Law" in page.locator("[data-toast='curate']").first.text_content()
    # the toast lands before the list is fetched again: wait for the card to go
    page.wait_for_selector(f".rv[data-candidate='{cid}']", state="detached")
    bank = eb.load_bank(root / "exam")["Law"]
    # the law bank also holds imported items, which carry no candidate id
    assert any(r.get("cid") == cid and r["edited"] and r["accepted_by"] == "Omar"
               for r in bank)
    card = page.locator(".rv[data-candidate]").first
    card.get_by_label("reject reason").fill("recall, not understanding")
    card.get_by_role("button", name="Reject").click()
    page.wait_for_function("[...document.querySelectorAll('[data-toast=curate]')]"
                           ".some(t => t.textContent.includes('Rejected'))")
    page.wait_for_function("document.querySelectorAll('.rv[data-candidate]').length === 0")
    # 9c-4: accepting made the question sittable by itself — the harness task
    # holds it, and there is no rebuild step left for a person to know about
    assert (root / "exam" / "tasks" / "exam_law.yaml").exists()
    accepted = next(r for r in eb.load_bank(root / "exam")["Law"] if r.get("cid") == cid)
    tasks = (root / "exam" / "tasks" / "exam_law.jsonl").read_text(encoding="utf-8")
    assert accepted["qid"] in tasks or accepted["prompt"][:40] in tasks
    assert page.locator("[data-action='exbuild']").count() == 0
    # Law holds one question more now, so the judged models' Law answers are
    # to other questions and count as history (10b); they sit it again, as
    # they would on the box, before anything below reads Law's results
    import make_fixture
    make_fixture.sit_again(root, root / "results" / "full", ["exam_law"])
    # screenshots: the Exam tab, light and dark, desktop and phone
    SCREENS.mkdir(exist_ok=True)
    eb.draft(root / "exam", llm.FakeBatches("fake-exam", root), ["Economics"], per_topic=2,
             wait=True, poll_s=0)
    for scheme in ("light", "dark"):
        page.emulate_media(color_scheme=scheme)
        for width in (1240, 430):
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(base + "/#tab=exam")
            page.reload()
            page.wait_for_selector(".rv[data-candidate]", timeout=15000)
            page.screenshot(path=SCREENS / f"exam-{scheme}-{width}.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.errors == []


# propose → approve → generate → gate → ready is four batch round trips and
# two poll intervals; on a loaded CI runner that is ordinary, not a flake, so
# the waits below are generous on purpose rather than tight and re-run.
E2E_MS = 90000


def test_review_flow_in_the_browser(live, page):
    base = live["base"]
    # the LLM card says what is configured and what today has cost
    page.goto(base + "/#tab=review")
    page.wait_for_selector(".card h2:has-text('Review')")
    # the AI's line fills in when the service answers
    page.wait_for_function("document.querySelector('#view').textContent.includes('fake/fake-1')")
    # 11j: with nothing waiting it opens on Datasets, and every view is empty
    assert page.locator("[data-rv-view='datasets'][aria-selected='true']").count() == 1
    assert [b.text_content() for b in page.locator("[data-rv-view]").all()] == [
        "To review (0)", "Ready to generate (0)", "Datasets (0)", "History (0)"]
    assert "No datasets yet" in page.locator("#view").text_content()
    # a name, once, in the header: every decision on the page records it
    set_name(page, "Omar")

    # 11k: Propose → on the model page opens the dialog over it
    card = open_topics(page, base, "fx/good-750m")
    card.locator("tr[data-topic='Economics'] .propose").click()
    dlg = page.locator("[data-dialog='propose']")
    dlg.wait_for()
    assert dlg.locator("[data-np-topic='Economics'] input").is_checked()
    dlg.locator("[data-dialog-go]").click()
    page.wait_for_selector("[data-toast='propose']", timeout=E2E_MS)
    assert "Proposal #" in page.locator("[data-toast='propose']").text_content()
    # 11j: the tab lists it in To review, and it opens as a card in the sheet
    page.goto(base + "/#tab=review")
    page.wait_for_selector("[data-review-head]")
    row = page.locator("[data-rv-row]").first
    row.wait_for(timeout=E2E_MS)
    pid = row.get_attribute("data-rv-row")
    assert row.locator("td").first.text_content() == "Economics"
    row.locator("[data-rv-open]").click()
    page.wait_for_selector("#reader[data-ready='1']", timeout=E2E_MS)
    card = page.locator("#reader")
    page.wait_for_selector(f"[data-spec-edit='{pid}']", timeout=E2E_MS)
    text = card.text_content()
    # the missing skill is the editable box's own text while it is to review
    assert "introductory Economics" in card.locator(f"[data-spec-edit='{pid}']").input_value()
    assert card.locator(".rd-title").text_content() == "Economics · good-750m"
    # 11h/11j: plain words, and the count that used to be blank
    assert "practice answers scored below 3 of 4" in text and "hidden questions" in text
    assert "—" not in card.locator(f"[data-why-line='{pid}']").text_content()
    # the answers it read: every one, with the question the reviewer checks it
    # against — graded by Economics' own criteria, 6 of good-750m's fell short
    card.locator(f"[data-answers-read='{pid}'] summary").click()
    page.wait_for_selector("[data-answers-count]", timeout=E2E_MS)
    assert card.locator("[data-answer-qid]").count() == 6
    assert "wording taken out" in card.text_content()
    # approve, edited
    ta = card.locator(f"[data-spec-edit='{pid}']")
    ta.fill(ta.input_value() + " Emphasise direction of effect.")
    card.locator(f"[data-approve='{pid}']").click()
    page.wait_for_selector(f"#reader [data-generate='{pid}']", timeout=E2E_MS)
    card.locator(f"[data-rv-details='{pid}'] summary").click()
    assert "Omar" in card.locator(f"[data-rv-details='{pid}']").text_content()
    assert "Approved as edited" in card.text_content()
    # generate
    page.locator("#reader input[type=number]").fill("20")
    card.locator(f"[data-generate='{pid}']").click()
    page.wait_for_selector(f"#reader [data-rv-datasets='{pid}']", timeout=E2E_MS)
    page.goto(base + "/#tab=review&view=datasets")
    ds = page.locator("[data-ds-row]").first
    ds.wait_for(timeout=E2E_MS)
    did_attr = ds.get_attribute("data-ds-row")
    page.wait_for_function("id => document.querySelector(`[data-doc-line='${id}']`)"
                           ".textContent === '20 of 20'", arg=did_attr, timeout=E2E_MS)
    # 11g: Read opens it in the page; the download is in the row's ⋯ menu, and
    # it is the same file
    assert ds.locator("[data-ds-read]").count() == 1
    ds.locator("[data-row-menu]").click()
    href = page.locator("[role=menuitem][data-act='download']").get_attribute("href")
    page.keyboard.press("Escape")
    with urllib.request.urlopen(f"{base}/{href}") as r:
        items = [json.loads(x) for x in r.read().decode().splitlines()]
    assert len(items) == 20
    # the whole provenance is in the reader, where the documents are
    ds.locator("[data-row-menu]").click()
    page.locator("[role=menuitem][data-act='provenance']").click()
    page.wait_for_selector("#reader[data-ready='1']", timeout=E2E_MS)
    prov = page.locator("#reader").text_content()
    assert "items_sha256" in prov and "approver" in prov
    page.keyboard.press("Escape")
    # taint it through the API the way a training run would, then look at the board
    did = int(did_attr)
    req = urllib.request.Request(f"{base}/api/truns", data=json.dumps(
        {"name": "gap-run", "datasets": [did]}).encode(),
        headers={"Content-Type": "application/json"})
    rid = json.loads(urllib.request.urlopen(req).read())["id"]
    urllib.request.urlopen(urllib.request.Request(f"{base}/api/truns/{rid}/event", data=json.dumps(
        {"step": 1, "detail": "fx/good-750m"}).encode(), headers={"Content-Type": "application/json"}))
    time.sleep(5.2)                                            # the payload debounce
    # the live page re-fetches results when a queue job finishes, not on a taint
    # change — a reload is how a viewer sees it, so that is what the test does
    page.goto(base + "/#tab=leaderboard")
    page.reload()
    page.wait_for_selector("table.lb")
    row = page.locator("table.lb tbody tr",
                       has=page.locator("a.mname", has_text=re.compile(r"^good-750m$"))).first
    # 11h: "trained on this topic's practice data", in plain words
    assert "trained on Economics practice data" in row.locator(".badge.taint").text_content()
    page.goto(model_url(base, "fx/good-750m"))
    page.wait_for_selector(".backlink")
    head = page.locator("#view .card").first.text_content()
    assert "derived from Economics diagnostics" in head and "never ranked" in head
    # the topic itself is badged on the Judged card and drops out of the judged average
    econ = page.locator("tr[data-topic='Economics']")
    assert "trained on it" in econ.locator(".badge.taint").text_content()
    assert "excluding Economics" in page.locator("#view").text_content()
    # screenshots for the PR: the Review tab, light and dark, desktop and phone
    SCREENS.mkdir(exist_ok=True)
    for scheme in ("light", "dark"):
        page.emulate_media(color_scheme=scheme)
        for width in (1240, 430):
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(base + "/#tab=review&view=datasets")
            page.wait_for_selector("[data-ds-row]")
            page.screenshot(path=SCREENS / f"review-{scheme}-{width}.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
            # the model page, exam first, with the before/after on the rubric scale
            page.goto(base + "/#model=fx%2Fskewed-360m")
            page.wait_for_selector(".card h2:has-text('Judged free response')")
            page.screenshot(path=SCREENS / f"model-exam-first-{scheme}-{width}.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.errors == []


REPO = Path(__file__).resolve().parents[1]
# the files the first five topics were delivered as, kept byte-identical when
# the 37-topic exam replaced them: these tests are about the import machinery
# and its numbers (100 questions, wrapped or bare), not about today's banks
RETIRED = REPO / "eval_tasks" / "fr" / "retired"
MEDICINE = RETIRED / "medicine_v2.json"


def upload(pg, label, name, mime, text):
    pg.get_by_label(label).set_input_files(
        {"name": name, "mimeType": mime, "buffer": text.encode("utf-8")})


def test_a_bank_arrives_from_the_page_with_its_report_half_withheld(live, page):
    """Dr. Hossein has no shell on the box: he delivers his file from the Exam
    tab, sees what would land before anything is written, and the half that
    becomes the published score is never printed back to him."""
    import exam_build as eb
    base, root = live["base"], live["root"]
    topic, raw = "Medicine & Clinical Health", MEDICINE.read_text(encoding="utf-8")
    items = json.loads(raw)
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='import']")
    panel = page.locator("[data-panel='import']")
    upload(page, "questions file", "medicine_v2.json", "application/json", raw)
    choose(panel.get_by_label("topic"), topic)
    panel.get_by_label("written by").fill("Dr. Hossein")
    # the source is the file's own name, shown as text; "change" opens a box (9c-4)
    assert panel.locator("[data-source]").get_attribute("data-source") == "medicine_v2"
    panel.locator("[data-source-change]").click()
    panel.get_by_label("source").fill("medicine_v1")
    set_name(page, "Omar")
    before = len(eb.load_bank(root / "exam").get(topic, []))
    panel.get_by_role("button", name="Preview").click()
    page.wait_for_selector("[data-preview='counts']")
    counts = panel.locator("[data-preview='counts']").text_content()
    assert "would import 100" in counts and "unusable 0" in counts
    # a preview writes nothing: that is the whole point of two steps
    assert len(eb.load_bank(root / "exam").get(topic, [])) == before
    # the report half is on the table as a qid, and its text is nowhere on the
    # page — not in a row, not in a title, not in a data attribute
    assert panel.locator("tr[data-half='report'] .se:has-text('withheld')").count() > 0
    assert panel.locator("tr[data-half='diagnose']").count() > 0
    html = page.content()
    withheld = [it["prompt"] for it in items
                if eb.half_of(eb.qid_of(it["prompt"])) == "report"]
    assert withheld and not any(q[:60] in html for q in withheld)
    # commit, and the bank on disk holds his records under his name
    panel.locator("button[data-commit='import']").click()
    page.wait_for_selector("[data-toast='import']", timeout=30000)
    assert "Imported 100" in page.locator("[data-toast='import']").text_content()
    said = page.locator("[data-action-ok='eximport']").text_content()
    assert "report" in said and "diagnose" in said
    bank = eb.load_bank(root / "exam")[topic]
    mine = [r for r in bank if r.get("source") == "medicine_v1"]
    assert len(mine) == 100
    assert all(r["accepted_by"] == "Dr. Hossein" and not r["edited"] for r in mine)
    assert {eb.half_of(r["qid"]) for r in mine} == {"report", "diagnose"}
    assert page.errors == []
    # the topic holds other questions now, so the judged models' answers on
    # it are history (10b); they sit it again, as they would on the box, and
    # the tests after this one read a topic that was sat on what it holds
    import make_fixture
    make_fixture.sit_again(root, root / "results" / "full", [eb.topic_task(topic)])


def test_a_rubric_is_replaced_from_the_page_and_says_what_that_costs(live, page,
                                                                    arts_without_rubric):
    """The file that grades a topic, changed by the person who wrote it: a bad
    criteria file cannot be committed at all, and a good one is committed only
    after the page has said the sha it is recorded under changes."""
    import judge as jd
    base, root = live["base"], live["root"]
    # every rubric delivered with the 37 topics is signed off, and signing one
    # off is what this test does — so the medicine topic is graded, for this
    # test, by the retired medicine & health pair (a DRAFT prose rubric, v2,
    # and its 15-criterion file), installed where the page's upload puts a
    # rubric. Removed again at the end, so the rest of the module reads the
    # repo's own.
    slug = "medicine_clinical_health"
    store = root / "rubrics"
    store.mkdir(exist_ok=True)
    for suffix in (".md", ".criteria.json"):
        (store / f"{slug}{suffix}").write_bytes(
            (RETIRED / "rubrics" / f"medicine_health{suffix}").read_bytes())
    repo_md = REPO / "eval_tasks" / "fr" / "rubrics" / f"{slug}.md"
    repo_text = repo_md.read_text(encoding="utf-8")
    try:
        page.goto(base + "/#tab=exam")
        page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
        panel = page.locator("[data-panel='rubrics']")
        all_rows(page, "rubrics")                    # Medicine is on page 2 of 25
        row = panel.locator("tr[data-rubric-row='Medicine & Clinical Health']")
        assert f"{slug}.md v2" in row.text_content()
        assert "15 criteria" in row.text_content()
        # the prose rubric is a draft; the criteria file is the author's own and
        # carries no status at all, so nothing claims it is one
        assert row.locator(".badge.taint", has_text="DRAFT").count() == 1
        # a topic without a rubric of its own says which one grades it instead —
        # once, in words; the file's name and sha in the tooltip (phase 9b-9).
        # Arts, with its files hidden for this test (every topic has its own
        # since 2026-09-22) and no questions in the fixture, so its row is
        # behind the empty-topics fold
        panel.locator("tr[data-empty-topics] [data-show-empty]").click()
        arts = panel.locator("tr[data-rubric-row='Arts'] [data-fallback]")
        arts.wait_for()
        assert arts.text_content().strip() == "shared rubric"
        assert "exam.md" in arts.get_attribute("title")
        # a criteria file the judge would refuse never gets a commit button
        row.get_by_role("link", name="replace").click()
        page.wait_for_selector(f"[data-upload='{slug}']")
        up = page.locator(f"[data-upload='{slug}']")
        choose(up.get_by_label("which file"), "criteria")
        spec = json.loads(jd.rubric_path(slug, ".criteria.json").read_text("utf-8"))
        spec["flags"][0]["effect"] = "melt_the_score"
        spec["criteria"][0]["id"] = "Relevance"
        upload(page, "new file", f"{slug}.criteria.json", "application/json",
               json.dumps(spec))
        set_name(page, "Dr. Hossein")
        up.get_by_role("button", name="Check it").click()
        page.wait_for_selector("[data-problems]")
        problems = up.locator("[data-problems]").text_content()
        assert "effect 'melt_the_score' is not one this judge can apply" in problems
        assert "an id is lower-case letters" in problems
        assert up.locator("button[data-commit='rubric']").count() == 0
        # the prose rubric, signed off: valid, changed, and the page says the cost
        choose(up.get_by_label("which file"), "rubric")
        text = jd.rubric_path(slug).read_text(encoding="utf-8")
        signed = text.split(", DRAFT")[0] + ")" + text.split(")", 1)[1]
        assert signed != text and "DRAFT" not in signed.split("\n", 1)[0]
        upload(page, "new file", f"{slug}.md", "text/markdown", signed)
        up.get_by_label("note").fill("signed off in the meeting")
        up.get_by_role("button", name="Check it").click()
        page.wait_for_selector("[data-sha-warning]")
        assert "not comparable" in up.locator("[data-sha-warning]").text_content()
        assert up.locator("[data-problems]").count() == 0
        up.locator("button[data-commit='rubric']").click()
        page.wait_for_selector("[data-action-ok='exrubric']", timeout=30000)
        said = page.locator("[data-action-ok='exrubric']").text_content()
        assert "Written to" in said and "re-run suite=judged" in said
        # written outside the checkout, read by the judge, and on the record
        written = store / f"{slug}.md"
        assert written.read_text(encoding="utf-8") == signed
        assert jd.rubric_for("exam_medicine_clinical_health").status == ""
        assert repo_md.read_text(encoding="utf-8") == repo_text
        from service import db
        change = db.rubric_changes(1)[0]
        assert change["approver"] == "Dr. Hossein" and change["kind"] == "rubric"
        assert change["note"] == "signed off in the meeting"
        # and the table it came from now shows the new sha, with the rubric's own
        # DRAFT gone and the criteria file's still there
        sha = jd.rubric_for("exam_medicine_clinical_health").sha256[:10]
        # the table is dropped and re-fetched after a commit, so for a tick there
        # is no row at all — waiting on its text must tolerate that, not throw
        page.wait_for_function(
            "sha => { const r = document.querySelector("
            "\"tr[data-rubric-row='Medicine & Clinical Health']\");"
            "  return !!r && r.innerHTML.includes(sha); }", arg=sha)   # in the tooltip (9b-9)
        assert row.locator(".badge.taint", has_text="DRAFT").count() == 0
        assert page.errors == []
    finally:
        for suffix in (".md", ".criteria.json"):
            (store / f"{slug}{suffix}").unlink(missing_ok=True)


@contextlib.contextmanager
def law_under_its_draft_rubric(live):
    """Law graded, for the Loop board's model, by the retired law v2 pair —
    a DRAFT prose rubric and its criteria file — as it was before the 37-topic
    exam: every rubric delivered with the 37 is signed off, and a draft's
    stamp needs one that is not. The pair goes where the page's rubric upload
    puts a file ($BENCH_ROOT/rubrics, which the judge reads first), and that
    model's law answers are graded again under it, as a judged run would —
    which is what records the draft in its judge.json. Put back on the way
    out. Yields the model."""
    import judge as jd
    from service import app, config
    store = live["root"] / "rubrics"
    store.mkdir(exist_ok=True)
    for suffix in (".md", ".criteria.json"):
        (store / f"law{suffix}").write_bytes((RETIRED / "rubrics" / f"law{suffix}").read_bytes())
    with urllib.request.urlopen(live["base"] + "/api/loop") as r:
        model = json.loads(r.read())["model"]
    d = config.OUT_DIR / model.replace("/", "__")
    kept = (d / "judge.json").read_bytes()
    try:
        jd.write_judge(d, jd.merge_judged(d, jd.run_stub(d, config.OUT_DIR, only=["exam_law"])))
        app._cache.update(key=None, payload=None, at=0.0)
        yield model
    finally:
        (d / "judge.json").write_bytes(kept)
        for suffix in (".md", ".criteria.json"):
            (store / f"law{suffix}").unlink(missing_ok=True)
        app._cache.update(key=None, payload=None, at=0.0)


def test_the_loop_tab_is_one_row_per_topic_with_the_next_step(live, page):
    """Phase 8e P6a: the loop, as a board. Every row says where the topic
    stands and the one thing to do next — and a refusal says why in words."""
    base = live["base"]
    with law_under_its_draft_rubric(live):
        page.goto(base + "/#tab=loop")
        page.wait_for_selector("table.jd[data-loop-table] tbody tr")
        # every topic in categories.yaml: a row for each of the 36 with
        # questions, and Arts, delivered empty, named in the one folded row
        assert page.locator("table.jd[data-loop-table] tbody tr[data-loop-row]").count() == 25
        all_rows(page, "loop")                       # the shared pager, 25 a page (10c)
        assert page.locator("table.jd[data-loop-table] tbody tr[data-loop-row]").count() == 36
        fold = page.locator("table.jd[data-loop-table] tbody tr[data-empty-topics]")
        assert fold.count() == 1 and "Arts" in fold.text_content()
        med = page.locator("tr[data-loop-row='medicine_clinical_health']")
        assert "medicine_clinical_health.md" in med.text_content()
        assert "20 criteria" in med.text_content()
        assert "/ 4" in med.text_content()                     # the last judged score
        # a rubric its author has not signed off is stamped on the row that uses it
        law = page.locator("tr[data-loop-row='law']")
        assert "law.md" in law.text_content() and "DRAFT" in law.text_content()
        # and its judged run is named once, above the board, not as a badge per row
        # (9d: at most one warning badge on a row)
        caveats = page.locator("[data-loop-caveats]").text_content()
        assert "draft rubric" in caveats and "Law" in caveats
        assert "draft rubric" not in law.text_content()
    # no judge is configured in this fixture, so a topic nobody has sat says
    # so on the button rather than offering it
    assert page.locator("[data-loop-blocked]").count() == 1
    sit = page.locator("button[data-step='sit']").first
    if sit.count():
        assert sit.is_disabled()
    # the step for a judged topic is to propose — the same for everyone who
    # looks — and the button opens the topic page, where Propose lives
    btn = med.locator("button[data-step]")
    assert btn.get_attribute("data-step") == "propose"
    med.locator("a[data-read]").click()
    page.wait_for_selector("[data-topic-page='medicine_clinical_health']")
    assert "#topic=medicine_clinical_health" in page.url
    SCREENS.mkdir(exist_ok=True)
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    page.screenshot(path=SCREENS / "loop-board.png", full_page=True)
    page.goto(base + "/#topic=medicine_clinical_health")
    page.wait_for_selector("[data-panel='answers'] [data-answers-table] [data-answer]")
    page.screenshot(path=SCREENS / "loop-topic.png", full_page=True)
    assert page.errors == []


def test_the_topic_page_shows_the_answers_and_never_the_report_half(live, page):
    """The panel a person reads before proposing anything: the diagnosis half
    in full, the report half as one line and not one row."""
    import exam_build as eb
    base, root = live["base"], live["root"]
    page.goto(base + "/#topic=medicine_clinical_health")   # deep link, cold
    page.wait_for_selector("[data-panel='answers'] [data-answers-table] [data-answer]")
    rows = page.locator("[data-answers-table] [data-answer]")
    assert rows.count() > 0
    assert rows.count() == page.locator("[data-answer][data-half='diagnose']").count()
    # the published half is a sentence, and the only sentence
    line = page.locator("[data-report-half]").first.text_content()
    assert "The hidden questions." in line and "published score" in line
    # and no report-half question is anywhere in the document
    bank = eb.load_bank(root / "exam")["Medicine & Clinical Health"]
    report = [b for b in bank if eb.half_of(b["qid"]) == "report"]
    assert report
    html = page.content()
    for b in report:
        assert b["prompt"][:60] not in html and b["qid"] not in html
    # a diagnose-half card carries the question, the answer, the score and the
    # judge's words — and a cell per criterion
    first = rows.first
    assert first.locator("[data-answer-text]").count() == 1
    assert first.locator("[data-criterion]").count() == 20
    assert "/ 4" in first.text_content() or "unreadable" in first.text_content()
    # filters narrow it without a reload
    before = int(page.locator("[data-answer-count]").first.get_attribute("data-answer-count"))
    choose(page.get_by_label("score filter"), "weak")
    page.wait_for_function(
        "n => +document.querySelector('[data-answer-count]').dataset.answerCount <= n",
        arg=before)
    assert page.errors == []


def test_sitting_one_topic_from_its_page(live, page):
    """The Sit panel queues a judged run narrowed to the topics ticked — and
    says why it cannot when the judge is not configured."""
    base = live["base"]
    page.goto(base + "/#topic=law")
    page.wait_for_selector("[data-panel='sit']")
    panel = page.locator("[data-panel='sit']")
    # this topic is pre-ticked, the others are there to add
    assert panel.locator("input[data-sit-task='exam_law']").is_checked()
    assert panel.locator("input[data-sit-task='exam_economics']").is_checked() is False
    # no judge configured in the fixture: the button says so instead of failing later
    assert panel.locator("[data-sit-blocked]").count() == 1
    assert panel.locator("button[data-sit]").is_disabled()
    # the queue's suite drop-down offers judged all the same, disabled with the reason
    page.goto(base + "/#tab=queue")
    suite = page.get_by_label("suite")
    suite.click()
    opt = page.locator("#pop-sel-submit-suite [role=option][data-value='judged']")
    assert opt.count() == 1 and opt.get_attribute("aria-disabled") == "true"
    assert "unavailable" in opt.text_content()
    page.keyboard.press("Escape")
    assert page.errors == []


def test_the_tabs_are_named_once_and_ordered_by_how_often_they_are_opened(live, page):
    """Phase 8e P6c, 9b-3: one name per tab, the hash equal to it, the old
    hashes still landing — and six tabs, the rest under More ▾."""
    base = live["base"]
    page.goto(base + "/")
    page.wait_for_selector("#tabs button[role=tab]")
    labels = page.locator("#tabs > button[role=tab]").all_text_contents()
    assert labels == ["Overview", "Loop", "Models", "Leaderboard", "Queue"]
    more = page.locator("#moreBtn")
    assert more.text_content() == "More ▾" and more.get_attribute("aria-haspopup") == "menu"
    more.click()
    # the panel lives on the body now (11a's popover), so the tab strip's own
    # sideways scroller cannot clip it
    items = page.locator("#pop-more [role=menuitem]").all_text_contents()
    assert items[:6] == ["Exam", "Review", "Training", "Tasks", "Perplexity & Loss", "Provenance"]
    page.keyboard.press("Escape")
    assert page.locator("#pop-more").count() == 0
    # the hash is the label, and the page said so in SERVICE.md
    for label, want in (("Loop", "loop"), ("Models", "models"),
                        ("Queue", "queue"), ("Provenance", "provenance"), ("Exam", "exam")):
        go_tab(page, label)
        page.wait_for_selector("#view > *")
        assert page.evaluate("location.hash") == f"#tab={want}", label
    # a tab under More names itself on the More button while it is open
    assert page.locator("#moreBtn").text_content() == "Exam ▾"
    assert page.locator("#moreBtn").get_attribute("aria-selected") == "true"
    # the hashes people already pasted somewhere
    for old, label in (("runs", "Provenance ▾"), ("submit", "Queue"),
                       ("evals", "Provenance ▾"), ("review", "Review ▾")):
        page.goto(f"{base}/#tab={old}")
        page.wait_for_selector("#view > *")
        assert page.locator("#tabs button[aria-selected='true']").inner_text() == label, old
    # the header says what it is, and the theme button says what it does
    page.goto(base + "/")
    page.wait_for_selector("[data-stamp]")
    # 11b: the live chip is the bar's badge now — "● LIVE · 12:33"
    assert re.fullmatch(r"LIVE · \d\d:\d\d",
                        page.locator("[data-stamp]").text_content())
    assert page.locator("[data-stamp] .dot.ok").count() == 1
    assert page.locator("#themeBtn").text_content().startswith("Theme")
    assert ":" not in page.locator("#themeBtn").text_content()
    assert "theme:" in page.locator("#themeBtn").get_attribute("title")
    assert page.errors == []


def test_the_loop_tab_says_what_failed_instead_of_loading_forever(live, page):
    """The live tree's /api/loop returned 500 and the board said 'Loading…'
    until someone opened the console. Every other tab already had the 8c
    error line; this one now does too."""
    base = live["base"]
    page.route("**/api/loop*", lambda route: route.fulfill(
        status=500, content_type="application/json", body='{"detail":"boom"}'))
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("[data-loop-failed]", timeout=20000)
    line = page.locator("[data-loop-failed]").text_content()
    assert "Not reaching the service." in line
    assert "api/loop — HTTP 500" in line
    assert "Retrying" in line and "backing off" in line
    assert "Nothing has loaded yet." in line
    assert "Loading…" not in page.locator("#view").text_content()
    # and when the service comes back, the board does
    page.unroute("**/api/loop*")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr", timeout=60000)
    assert page.locator("[data-loop-failed]").count() == 0
    # the 500 we injected is the only thing the console should have to say
    assert all("500" in e for e in page.errors), page.errors


def test_a_topic_on_the_shared_rubric_says_so_on_both_boards(live, page, arts_without_rubric):
    """Thirteen topics had no rubric of their own; of the 37 none has now, so
    this one takes Arts' files out of the repo copy the judge reads — and the
    page says which file grades it rather than implying each has one. The
    fixture leaves Arts without questions, so on both boards its row is behind
    the empty-topics fold."""
    base = live["base"]
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    page.locator("tr[data-empty-topics] [data-show-empty]").click()
    arts = page.locator("tr[data-loop-row='arts']")
    arts.wait_for()
    assert arts.locator("[data-fallback]").count() == 1
    assert "shared rubric" in arts.text_content()
    assert "exam.md" in arts.locator("[data-fallback]").get_attribute("title")
    assert page.locator("tr[data-loop-row='law'] [data-fallback]").count() == 0
    # every other topic is graded by its own
    assert page.locator("tr[data-loop-row] [data-fallback]").count() == 1
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
    page.locator("[data-panel='rubrics'] tr[data-empty-topics] [data-show-empty]").click()
    row = page.locator("tr[data-rubric-row='Arts']")
    row.wait_for()
    assert "shared rubric" in row.text_content()
    assert "exam.md" in row.locator("[data-fallback]").get_attribute("title")
    assert "(fallback)" not in row.text_content()            # said once, in words
    assert page.locator("[data-rubric-error]").count() == 0
    assert page.errors == []


def make_stale(root):
    """The bank one step ahead of the harness tasks: what a judged run in
    progress leaves behind when an import could not rebuild them."""
    import os
    import time as _t
    import exam_build as eb
    earlier = _t.time() - 3600
    for p in eb.tasks_dir(root / "exam").glob("exam_*.jsonl"):
        os.utime(p, (earlier, earlier))


def test_a_button_that_calls_the_api_says_what_happened(live, page):
    """The rebuild button answered "refused: 500" in small grey text under
    itself, which the person read as "nothing happens" — and behind it was a
    file missing from the image. Every button that calls the API now says it
    is working, says what happened, and on a refusal says what the SERVER
    said."""
    base = live["base"]
    make_stale(live["root"])                     # 9c-4: the button shows only when needed
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-action='exbuild']")
    btn = page.locator("[data-action='exbuild']")
    assert btn.text_content() == "Make new questions sittable"
    # the refusal: the server's own sentence, in the error style, not a status
    page.route("**/api/exam/build", lambda route: route.fulfill(
        status=500, content_type="application/json",
        body='{"detail":"the tasks could not be built: no such file: _fr_template_yaml"}'))
    btn.click()
    page.wait_for_selector("[data-action-error='exbuild']")
    err = page.locator("[data-action-error='exbuild']")
    assert "Refused." in err.text_content()
    assert "_fr_template_yaml" in err.text_content()
    assert "warn" in (err.get_attribute("class") or "")     # the 8c error style
    assert "500" not in err.text_content()                  # the message, not the code
    # and the success: what was built, where, in a line a person can read
    page.unroute("**/api/exam/build")
    btn.click()
    # the success is a toast in words (9c-1), and the button goes: nothing is stale
    page.wait_for_selector("[data-toast='build']", timeout=30000)
    ok = page.locator("[data-toast='build']").text_content()
    assert "can be sat now" in ok and "exam_" not in ok
    page.wait_for_selector("[data-action='exbuild']", state="detached", timeout=30000)
    assert page.locator("[data-action-error='exbuild']").count() == 0
    # the 500 we injected is the only thing the console should have to say
    assert all("500" in e for e in page.errors), page.errors


def test_a_button_that_is_working_says_so_and_cannot_be_pressed_twice(live, page):
    base = live["base"]
    make_stale(live["root"])
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-action='exbuild']")
    # hold the request open: the button must say it is working meanwhile
    page.route("**/api/exam/build", lambda route: page.wait_for_timeout(1500) or route.fulfill(
        status=200, content_type="application/json", body='{"tasks":{},"tasks_dir":"/x"}'))
    btn = page.locator("[data-action='exbuild']")
    btn.click()
    assert btn.inner_text().strip() == "working…"
    assert btn.is_disabled()
    assert page.locator("[data-action-busy='exbuild']").count() == 1
    page.wait_for_selector("[data-toast='build']", timeout=30000)
    assert page.locator("[data-action-busy='exbuild']").count() == 0
    page.unroute("**/api/exam/build")
    assert page.errors == []


def test_typing_survives_the_poll(live, page):
    """Phase 8g D1/D3, 9b-4: the board refreshes every five seconds, and the
    name is now one box, in the header. A person part way through typing it
    must not lose it, or the caret, or the field — and the name must still be
    there on the next page they open, and after a reload."""
    base = live["base"]
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    # 11b: the name is one control in the sticky bar, and its box is the
    # popover's — a poll rebuilds the button, never the open panel
    page.locator("#who button[data-who]").click()
    page.wait_for_selector("#pop-who input")
    name = page.locator("#pop-who input")
    name.fill("")
    name.type("Omar")
    page.evaluate("refreshResults()")              # a results refresh redraws the header
    page.wait_for_timeout(6500)                    # and two polls land here
    assert page.evaluate("document.activeElement === document.querySelector('#pop-who input')")
    name.type(" Affifi")
    assert name.input_value() == "Omar Affifi"
    name.press("Enter")
    page.wait_for_selector("#who button[data-who='Omar Affifi']")
    # the same name on the topic page, and after a reload
    page.goto(base + "/#topic=law")
    page.wait_for_selector("[data-topic-page='law']")
    assert page.locator("#who button[data-who]").get_attribute("data-who") == "Omar Affifi"
    page.reload()
    page.wait_for_selector("[data-topic-page='law']")
    assert page.locator("#who button[data-who]").get_attribute("data-who") == "Omar Affifi"
    assert page.errors == []


def test_the_loop_board_does_not_rebuild_itself_when_nothing_moved(live, page):
    """The same fix as the tab bar's: a poll that changes nothing must not
    hand the person a new DOM. A reference taken to a row survives."""
    base = live["base"]
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    row = page.locator("tr[data-loop-row='law']")
    page.evaluate("document.querySelector(\"tr[data-loop-row='law']\").dataset.marked = 'yes'")
    page.wait_for_timeout(6500)
    assert row.get_attribute("data-marked") == "yes"     # the same node, two polls later
    assert page.errors == []


def test_the_queue_row_counts_the_judge_batch_up(live, page, monkeypatch):
    """Phase 8g D2: "judging 40/130", not "judging 130 answers"."""
    from service import db, llm_poller
    base = live["base"]
    # a batch in flight, held there: this fixture has no judge, so the live
    # poller would fail the batch on its next tick — and a row whose grading
    # failed says that instead of a count (10c), which made this a race
    monkeypatch.setattr(llm_poller, "tick", lambda: 0)
    sid = db.add("fx/good-750m", "auto", "judged", "omar", "", tasks=["exam_law"])
    rid = db.judge_run_create("fx/good-750m", "b_live", 130, "stub/overlap-v1", "{}")
    db.batch_add("b_live", "judge", rid, 130, "anthropic", "claude-x")
    db.batch_progress("b_live", "40/130 done")
    db.update(sid, judge_batch="b_live")       # as judge.start_run records it on submit
    page.goto(base + "/#tab=queue")
    page.wait_for_selector(f"table.jd tbody tr:has-text('#{sid}'), table tbody tr")
    page.wait_for_selector("[data-judge-progress]", timeout=20000)
    cell = page.locator("[data-judge-progress]").first
    assert cell.text_content().strip() == "judging 40/130"
    # and the topic page says it too, where the person is waiting
    page.goto(base + "/#topic=law")
    page.wait_for_selector("[data-judging]", timeout=20000)
    assert "judging 40/130" in page.locator("[data-judging]").text_content()
    assert page.errors == []


# the retired physics & engineering file: still the one delivered wrapped in
# {"questions": [...]}, which is what this test is about (the new banks are
# bare arrays); its classical mechanics goes into Physics & Astronomy
PHYSICS_FILE = RETIRED / "physics_engineering_v1.json"


def test_a_refusal_replaces_the_last_success_rather_than_sitting_under_it(live, page):
    """After computer science imported, the physics attempt showed the CS
    success line with "refused" beneath it, which reads as a partial import.
    One result line per panel."""
    base = live["base"]
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='import']")
    panel = page.locator("[data-panel='import']")
    set_name(page, "Omar")
    panel.get_by_label("written by").fill("Dr. Hossein")
    # a good import first
    upload(page, "questions file", "computer_science_v1.json", "application/json",
           (RETIRED / "computer_science_v1.json").read_text(encoding="utf-8"))
    choose(panel.get_by_label("topic"), "Computer Science")
    panel.get_by_role("button", name="Preview").click()
    page.wait_for_selector("[data-action-ok='eximport']", timeout=30000)
    panel.locator("button[data-commit='import']").click()
    # the answer is a toast (9c-1), and under the button what it did to the halves
    page.wait_for_selector("[data-toast='import']", timeout=30000)
    assert "Imported 100" in page.locator("[data-toast='import']").text_content()
    page.wait_for_function(
        "() => (document.querySelector(\"[data-action-ok='eximport']\") || {}).textContent"
        "?.startsWith('report')", timeout=30000)
    # then one the server refuses: the success must be gone, not above it
    upload(page, "questions file", "broken.json", "application/json", '{"a": [], "b": []}')
    # the file's onchange is async (it reads the file), so wait for the answer
    # to the LAST file to go rather than racing it
    page.wait_for_selector("[data-action-ok='eximport']", state="detached", timeout=20000)
    panel.get_by_role("button", name="Preview").click()
    page.wait_for_selector("[data-action-error='eximport']", timeout=30000)
    assert page.locator("[data-action-ok='eximport']").count() == 0
    assert panel.locator("[data-action-error], [data-action-ok]").count() == 1
    said = panel.locator("[data-action-error='eximport']").text_content()
    assert "Refused." in said and "array of question objects" in said
    assert "Imported 100" not in panel.text_content()
    # the 422 we asked for is the only thing the console should have to say
    assert all("422" in e for e in page.errors), page.errors


def test_the_page_imports_the_wrapped_file_the_author_sent(live, page):
    """The physics file, as delivered, through the panel that refused it."""
    import exam_build as eb
    base, root = live["base"], live["root"]
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='import']")
    panel = page.locator("[data-panel='import']")
    set_name(page, "Omar")
    panel.get_by_label("written by").fill("Dr. Hossein")
    upload(page, "questions file", "physics_engineering_v1.json", "application/json",
           PHYSICS_FILE.read_text(encoding="utf-8"))
    choose(panel.get_by_label("topic"), "Physics & Astronomy")
    # the file's own name is the source already — nothing to type
    assert panel.locator("[data-source]").get_attribute("data-source") == "physics_engineering_v1"
    panel.get_by_role("button", name="Preview").click()
    page.wait_for_selector("[data-action-ok='eximport']", timeout=30000)
    said = panel.locator("[data-action-ok='eximport']").text_content()
    assert "Read 100 questions" in said and 'from "questions" in the file' in said
    panel.locator("button[data-commit='import']").click()
    page.wait_for_selector("[data-toast='import']", timeout=30000)
    assert "Imported 100 questions" in page.locator("[data-toast='import']").text_content()
    mine = [r for r in eb.load_bank(root / "exam")["Physics & Astronomy"]
            if r.get("source") == "physics_engineering_v1"]
    assert len(mine) == 100
    assert page.errors == []


def test_the_rubrics_table_says_whether_a_topic_has_questions(live, page):
    """A rubric and a criteria file ship in the repo, so every topic looked
    equipped whether or not anyone had written it a question — and someone
    reasonably asked whether the banks had already been imported."""
    import exam_build as eb
    base, root = live["base"], live["root"]
    # a topic with the files and no questions: exactly the case that misled
    # (Arts is empty already, but it has no rubric files of its own)
    bank = eb.bank_dir(root / "exam") / "sociology.jsonl"
    kept = bank.read_bytes()
    bank.unlink()
    try:
        page.goto(base + "/#tab=exam")
        page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
        panel = page.locator("[data-panel='rubrics']")
        all_rows(page, "rubrics")
        med = panel.locator("tr[data-rubric-row='Medicine & Clinical Health']")
        assert med.locator("[data-bank]").first.get_attribute("data-bank") != "0"
        assert "report" in med.text_content() and "diagnose" in med.text_content()
        # a topic without questions folds into one row that names it (9b-7) …
        fold = panel.locator("tr[data-empty-topics]")
        assert fold.count() == 1 and "Sociology" in fold.text_content()
        assert panel.locator("tr[data-rubric-row='Sociology']").count() == 0
        # … and opens to its own row, which still says it has no questions
        fold.locator("[data-show-empty]").click()
        empty = panel.locator("tr[data-rubric-row='Sociology'] [data-bank='0']")
        empty.wait_for()
        assert empty.text_content() == "no questions yet"
        # and an unversioned heading is not printed as an error
        assert "v?" not in panel.text_content()
        assert panel.locator("[data-no-version]").count() >= 1
        assert "no version" in panel.text_content()
        assert page.errors == []
    finally:
        bank.write_bytes(kept)
