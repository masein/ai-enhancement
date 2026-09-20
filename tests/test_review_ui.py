"""The Review tab and the propose buttons, against a live service with the
fake LLM: the LIVE dashboard fetches, so this needs a real HTTP server."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

import make_fixture

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens"


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    import uvicorn

    from service import config, llm, llm_poller, worker
    import service.app as appmod
    root = tmp_path_factory.mktemp("live")
    tree = make_fixture.build(root)
    saved = {k: getattr(config, k) for k in (
        "BENCH_ROOT", "RESULTS_ROOT", "OUT_DIR", "DB_PATH", "ARTIFACTS_DIR", "LOGS_DIR",
        "DATASETS_DIR", "SUBMIT_TOKEN", "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_POLL_S",
        "EXAM_DIR", "EXAM_PROVIDER", "EXAM_MODEL", "EXAM_API_KEY", "JUDGED_TASKS_DIR")}
    for k, v in {"BENCH_ROOT": root, "RESULTS_ROOT": root / "results",
                 "OUT_DIR": root / "results" / "full", "DB_PATH": root / "service.sqlite3",
                 "ARTIFACTS_DIR": root / "artifacts", "LOGS_DIR": root / "logs",
                 "DATASETS_DIR": root / "datasets", "SUBMIT_TOKEN": "",
                 "LLM_PROVIDER": "fake", "LLM_MODEL": "fake-1", "LLM_API_KEY": "",
                 "LLM_POLL_S": 0.3, "EXAM_DIR": root / "exam", "EXAM_PROVIDER": "fake",
                 "EXAM_MODEL": "fake-exam", "EXAM_API_KEY": "",
                 "JUDGED_TASKS_DIR": root / "exam" / "tasks"}.items():
        setattr(config, k, v)
    worker_start = worker.start
    worker.start = lambda: None
    llm.reset()
    appmod._cache.update(key=None, payload=None, at=0.0)
    # an upload from the page must not land in the developer's checkout: the
    # service prefers the repo's rubrics directory when it can write there,
    # and on this machine it can
    rubric_store = appmod._rubric_store
    appmod._rubric_store = lambda: (root / "rubrics", False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("live server did not come up")
    yield {"base": base, "tree": tree, "root": root}
    server.should_exit = True
    th.join(5)
    llm_poller.stop()
    worker.start = worker_start
    appmod._rubric_store = rubric_store
    for k, v in saved.items():
        setattr(config, k, v)
    llm.reset()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.error: {m.text}")
          if m.type == "error" else None)
    pg.errors = errors
    yield pg
    ctx.close()


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
    """The action is on the exam topic, and every refusal says why on the row."""
    base = live["base"]
    card = open_topics(page, base, "fx/good-750m")
    econ = card.locator("tr[data-topic='economics']")
    assert econ.locator("button.propose").is_enabled()
    assert econ.locator(".propwhy").count() == 0
    law = card.locator("tr[data-topic='law']")
    assert law.locator("button.propose").is_disabled()
    assert "under the 30" in law.locator(".propwhy").first.text_content()
    # a model that wrote the same sentence every time has revealed no topic gap
    card = open_topics(page, base, "fx/chance-160m")
    econ = card.locator("tr[data-topic='economics']")
    assert econ.locator("button.propose").is_disabled()
    assert "same answer on nearly every question" in econ.locator(".propwhy").first.text_content()
    # MMLU's finding for the same category rides along as a caution, not a gate
    card = open_topics(page, base, "fx/skewed-360m")
    econ = card.locator("tr[data-topic='economics']")
    assert econ.locator("button.propose").is_enabled()
    assert "caution — MMLU for this category" in econ.text_content()
    assert "answer positions" in econ.locator(".propwhy").first.text_content()
    # and the Diagnose section no longer offers one: MMLU does not pick the topic
    det = open_mmlu(page, base, "fx/good-750m")
    assert det.locator("button.propose").count() == 0
    assert page.errors == []


def test_exam_curation_in_the_browser(live, page):
    """The Exam tab: the bank per topic, candidates awaiting a decision, accept
    with an edit under a name, reject with a reason, rebuild the tasks."""
    import exam_build as eb
    from service import llm
    base, root = live["base"], live["root"]
    out = eb.draft(root / "exam", llm.FakeBatches("fake-exam", root), ["law", "history"],
                   per_topic=2, wait=True, poll_s=0)
    assert out["written"] == {"law": 2, "history": 2}
    page.goto(base + "/#tab=exam")
    page.wait_for_selector(".card h2:has-text('Exam')")
    page.wait_for_selector(".rv[data-candidate]", timeout=15000)
    text = page.locator("#view").text_content()
    assert "fake/fake-exam" in text and "awaiting curation 4" in text
    rows = page.locator("table.jd[data-topics-table] tbody tr")
    assert rows.count() == 15                                      # one per topic in categories.yaml
    assert "other" in text and "economics" in text
    # filter to one topic by clicking it
    page.locator("table.jd[data-topics-table] tbody tr a", has_text=re.compile(r"^law$")).click()
    page.wait_for_selector(".card h2:has-text('Awaiting curation — law')")
    # the list is dropped with the filter and re-fetched, so wait for it to
    # land rather than counting whatever is on screen this frame
    page.wait_for_function("document.querySelectorAll('.rv[data-candidate]').length === 2")
    cards = page.locator(".rv[data-candidate]")
    assert cards.count() == 2
    card = cards.first
    cid = card.get_attribute("data-candidate")
    q = card.get_by_label("question")
    q.fill(q.input_value() + " Give one counterexample.")
    card.get_by_label("your name").fill("Omar")
    card.get_by_role("button", name="Accept into the bank").click()
    page.wait_for_function("document.querySelector('#view').textContent.includes('accepted →')")
    assert page.locator(f".rv[data-candidate='{cid}']").count() == 0
    bank = eb.load_bank(root / "exam")["law"]
    # the law bank also holds imported items, which carry no candidate id
    assert any(r.get("cid") == cid and r["edited"] and r["accepted_by"] == "Omar"
               for r in bank)
    card = page.locator(".rv[data-candidate]").first
    card.get_by_label("your name").fill("Omar")
    card.get_by_label("reject reason").fill("recall, not understanding")
    card.get_by_role("button", name="Reject").click()
    page.wait_for_function("document.querySelector('#view').textContent.includes('rejected')")
    assert page.locator(".rv[data-candidate]").count() == 0
    # rebuild the harness tasks from the bank, from the page
    page.get_by_role("button", name="Rebuild the harness tasks from the bank").click()
    page.wait_for_selector("[data-action-ok='exbuild']", timeout=30000)
    built = page.locator("[data-action-ok='exbuild']").text_content()
    assert built.startswith("Built") and "law (" in built
    assert (root / "exam" / "tasks" / "exam_law.yaml").exists()
    # screenshots: the Exam tab, light and dark, desktop and phone
    SCREENS.mkdir(exist_ok=True)
    eb.draft(root / "exam", llm.FakeBatches("fake-exam", root), ["economics"], per_topic=2,
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


def test_the_review_tab_starts_from_a_topic(live, page):
    """Pick a topic, see where every model stands on it, read what the judge
    wrote about the diagnosis-half answers, then propose."""
    base = live["base"]
    page.goto(base + "/#tab=review")
    page.wait_for_selector(".card h2:has-text('Pick a topic')")
    picker = page.locator(".card", has=page.locator("h2", has_text="Pick a topic"))
    rows = picker.locator("tbody tr")
    assert rows.count() >= 10
    # weakest first, and the weakest model on each topic is named
    scores = [float(x.split("/")[0]) for x in
              picker.locator("tbody tr td:nth-child(3)").all_text_contents()]
    assert scores == sorted(scores)
    econ = picker.locator("tr[data-pick='economics']")
    assert econ.locator("a.mlink").count() == 1
    econ.get_by_role("button", name="choose").click()
    detail = page.locator("[data-topic-detail='economics']")
    detail.wait_for()
    assert "across the board" in detail.text_content()
    assert detail.locator("[data-topic-model]").count() >= 3
    first = detail.locator("[data-topic-model]").first
    assert "report-half questions" in first.text_content()
    # the judge's own words, fetched on demand and labelled
    first.locator("details.dxex summary").click()
    page.wait_for_selector("[data-topic-model] details.dxex li", timeout=15000)
    li = first.locator("details.dxex li").first
    assert "scored" in li.text_content()
    assert "question text removed" in first.text_content()
    assert "diagnosis-half answers scored below 3 of 4" in first.text_content()
    # and the propose button for the model that can be proposed from
    good = detail.locator("[data-topic-model='fx/good-750m']")
    assert good.locator("button.propose").is_enabled()
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
    assert "fake/fake-1" in page.locator("#view").text_content()
    assert "Nothing waiting" in page.locator("#view").text_content()
    # a name, remembered for every decision on this page
    page.get_by_label("your name").first.fill("Omar")

    card = open_topics(page, base, "fx/good-750m")
    card.locator("tr[data-topic='economics'] button.propose").click()
    page.wait_for_selector(".card h2:has-text('Review')")
    assert "proposal #" in page.locator("#view").text_content()
    card = page.locator(".rv[data-proposal]").first
    card.locator("textarea").wait_for(timeout=E2E_MS)         # the poller and the 5 s poll
    text = card.text_content()
    assert "introductory economics" in text
    assert "judge assessments the LLM saw" in text and "question text removed" in text
    assert "report-half questions" in text and "fell short" in text
    assert "fx/good-750m · exam_economics · economics" in text
    assert "judge stub/overlap-v1" in text
    card.locator("details summary").first.click()
    assert card.locator(".ex").count() == 8
    assert "scored" in card.locator(".ex").first.text_content()
    # approve, edited
    ta = card.locator("textarea")
    ta.fill(ta.input_value() + " Emphasise direction of effect.")
    card.get_by_label("your name").fill("Omar")
    card.get_by_role("button", name="Approve this spec").click()
    page.wait_for_selector(".rv[data-proposal] :text('Approved as edited')", timeout=E2E_MS)
    card = page.locator(".rv[data-proposal]").first
    assert "approved by Omar" in card.text_content()
    # generate
    card.get_by_label("item count").fill("20")
    card.get_by_label("your name").fill("Omar")
    card.get_by_role("button", name="Generate data").click()
    page.wait_for_selector(".rv[data-dataset]:has-text('ready')", timeout=E2E_MS)
    ds = page.locator(".rv[data-dataset]").first
    ds.locator("summary").click()
    dtext = ds.text_content()
    assert "20 kept of 20 generated" in dtext and "Contamination gate" in dtext
    assert "Provenance, in full" in dtext and "approver" in dtext and "items_sha256" in dtext
    href = ds.locator("a:has-text('items.jsonl')").get_attribute("href")
    with urllib.request.urlopen(f"{base}/{href}") as r:
        items = [json.loads(x) for x in r.read().decode().splitlines()]
    assert len(items) == 20
    # taint it through the API the way a training run would, then look at the board
    did = int(re.search(r"dataset #(\d+)", dtext).group(1))
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
    assert "trained on economics diagnostics" in row.locator(".badge.taint").text_content()
    page.goto(model_url(base, "fx/good-750m"))
    page.wait_for_selector(".backlink")
    head = page.locator("#view .card").first.text_content()
    assert "derived from economics diagnostics" in head and "never ranked" in head
    # the topic itself is badged on the Judged card and drops out of the judged average
    econ = page.locator("tr[data-topic='economics']")
    assert "trained on it" in econ.locator(".badge.taint").text_content()
    assert "excluding economics" in page.locator("#view").text_content()
    # screenshots for the PR: the Review tab, light and dark, desktop and phone
    SCREENS.mkdir(exist_ok=True)
    for scheme in ("light", "dark"):
        page.emulate_media(color_scheme=scheme)
        for width in (1240, 430):
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(base + "/#tab=review")
            page.wait_for_selector(".rv[data-dataset]")
            # the button toggles, so only choose when it is not already chosen
            if page.locator("[data-topic-detail='economics']").count() == 0:
                page.locator("tr[data-pick='economics'] button").click()
            page.wait_for_selector("[data-topic-detail='economics']")
            if page.locator("[data-topic-model] details.dxex[open]").count() == 0:
                page.locator("[data-topic-model] details.dxex summary").first.click()
                page.wait_for_selector("[data-topic-model] details.dxex li", timeout=15000)
            if page.locator(".rv[data-dataset][open]").count() == 0:
                page.locator(".rv[data-dataset] summary").first.click()
            page.screenshot(path=SCREENS / f"review-{scheme}-{width}.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
            # the model page, exam first, with the before/after on the rubric scale
            page.goto(base + "/#model=fx%2Fskewed-360m")
            page.wait_for_selector(".card h2:has-text('Judged free response')")
            page.screenshot(path=SCREENS / f"model-exam-first-{scheme}-{width}.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.errors == []


REPO = Path(__file__).resolve().parents[1]
MEDICINE = REPO / "eval_tasks" / "fr" / "medicine_v2.json"


def upload(pg, label, name, mime, text):
    pg.get_by_label(label).set_input_files(
        {"name": name, "mimeType": mime, "buffer": text.encode("utf-8")})


def test_a_bank_arrives_from_the_page_with_its_report_half_withheld(live, page):
    """Dr. Hossein has no shell on the box: he delivers his file from the Exam
    tab, sees what would land before anything is written, and the half that
    becomes the published score is never printed back to him."""
    import exam_build as eb
    base, root = live["base"], live["root"]
    topic, raw = "medicine & health", MEDICINE.read_text(encoding="utf-8")
    items = json.loads(raw)
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='import']")
    panel = page.locator("[data-panel='import']")
    upload(page, "questions file", "medicine_v2.json", "application/json", raw)
    panel.get_by_label("topic").select_option(topic)
    panel.get_by_label("written by").fill("Dr. Hossein")
    panel.get_by_label("source").fill("medicine_v1")
    panel.get_by_label("your name").first.fill("Omar")
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
    page.wait_for_selector("[data-action-ok='eximport']", timeout=30000)
    said = page.locator("[data-action-ok='eximport']").text_content()
    assert "Imported 100" in said and "report" in said and "diagnose" in said
    bank = eb.load_bank(root / "exam")[topic]
    mine = [r for r in bank if r.get("source") == "medicine_v1"]
    assert len(mine) == 100
    assert all(r["accepted_by"] == "Dr. Hossein" and not r["edited"] for r in mine)
    assert {eb.half_of(r["qid"]) for r in mine} == {"report", "diagnose"}
    assert page.errors == []


def test_a_rubric_is_replaced_from_the_page_and_says_what_that_costs(live, page):
    """The file that grades a topic, changed by the person who wrote it: a bad
    criteria file cannot be committed at all, and a good one is committed only
    after the page has said the sha it is recorded under changes."""
    import judge as jd
    base, root = live["base"], live["root"]
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
    panel = page.locator("[data-panel='rubrics']")
    row = panel.locator("tr[data-rubric-row='medicine & health']")
    assert "medicine_health.md v2" in row.text_content()
    assert "15 criteria" in row.text_content()
    # the prose rubric is a draft; the criteria file is the author's own and
    # carries no status at all, so nothing claims it is one
    assert row.locator(".badge.taint", has_text="DRAFT").count() == 1
    # a topic without a rubric of its own says which one grades it instead
    assert "exam.md" in panel.locator("tr[data-rubric-row='history']").text_content()
    # a criteria file the judge would refuse never gets a commit button
    row.get_by_role("link", name="replace").click()
    page.wait_for_selector("[data-upload='medicine_health']")
    up = page.locator("[data-upload='medicine_health']")
    up.get_by_label("which file").select_option("criteria")
    spec = json.loads(jd.rubric_path("medicine_health", ".criteria.json").read_text("utf-8"))
    spec["flags"][0]["effect"] = "melt_the_score"
    spec["criteria"][0]["id"] = "Relevance"
    upload(page, "new file", "medicine_health.criteria.json", "application/json",
           json.dumps(spec))
    up.get_by_label("your name").first.fill("Dr. Hossein")
    up.get_by_role("button", name="Check it").click()
    page.wait_for_selector("[data-problems]")
    problems = up.locator("[data-problems]").text_content()
    assert "effect 'melt_the_score' is not one this judge can apply" in problems
    assert "an id is lower-case letters" in problems
    assert up.locator("button[data-commit='rubric']").count() == 0
    # the prose rubric, signed off: valid, changed, and the page says the cost
    up.get_by_label("which file").select_option("rubric")
    text = jd.rubric_path("medicine_health").read_text(encoding="utf-8")
    signed = text.split(", DRAFT")[0] + ")" + text.split(")", 1)[1]
    assert signed != text and "DRAFT" not in signed.split("\n", 1)[0]
    upload(page, "new file", "medicine_health.md", "text/markdown", signed)
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
    written = root / "rubrics" / "medicine_health.md"
    assert written.read_text(encoding="utf-8") == signed
    assert jd.rubric_for("exam_medicine_health").status == ""
    assert (REPO / "eval_tasks" / "fr" / "rubrics" / "medicine_health.md"
            ).read_text(encoding="utf-8") == text
    from service import db
    change = db.rubric_changes(1)[0]
    assert change["approver"] == "Dr. Hossein" and change["kind"] == "rubric"
    assert change["note"] == "signed off in the meeting"
    # and the table it came from now shows the new sha, with the rubric's own
    # DRAFT gone and the criteria file's still there
    sha = jd.rubric_for("exam_medicine_health").sha256[:10]
    # the table is dropped and re-fetched after a commit, so for a tick there
    # is no row at all — waiting on its text must tolerate that, not throw
    page.wait_for_function(
        "sha => { const r = document.querySelector(\"tr[data-rubric-row='medicine & health']\");"
        "  return !!r && r.textContent.includes(sha); }", arg=sha)
    assert row.locator(".badge.taint", has_text="DRAFT").count() == 0
    assert page.errors == []


def test_the_loop_tab_is_one_row_per_topic_with_the_next_step(live, page):
    """Phase 8e P6a: the loop, as a board. Every row says where the topic
    stands and the one thing to do next — and a refusal says why in words."""
    base = live["base"]
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    rows = page.locator("table.jd[data-loop-table] tbody tr")
    assert rows.count() == 15                              # every topic in categories.yaml
    med = page.locator("tr[data-loop-row='medicine_health']")
    assert "medicine_health.md" in med.text_content()
    assert "15 criteria" in med.text_content()
    assert "/ 4" in med.text_content()                     # the last judged score
    # a rubric its author has not signed off is stamped on the row that uses it
    law = page.locator("tr[data-loop-row='law']")
    assert "law.md" in law.text_content() and "DRAFT" in law.text_content()
    assert "draft rubric" in law.text_content()            # and on its judged run
    # no judge is configured in this fixture, so a topic nobody has sat says
    # so on the button rather than offering it
    assert page.locator("[data-loop-blocked]").count() == 1
    sit = page.locator("button[data-step='sit']").first
    if sit.count():
        assert sit.is_disabled()
    # the step for a judged topic is to read what the judge made of it
    btn = med.locator("button[data-step]")
    assert btn.get_attribute("data-step") == "read"
    btn.click()
    page.wait_for_selector("[data-topic-page='medicine_health']")
    assert "#topic=medicine_health" in page.url
    SCREENS.mkdir(exist_ok=True)
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    page.screenshot(path=SCREENS / "loop-board.png", full_page=True)
    page.goto(base + "/#topic=medicine_health")
    page.wait_for_selector("[data-panel='answers'] table.jd[data-answers-table]")
    page.screenshot(path=SCREENS / "loop-topic.png", full_page=True)
    assert page.errors == []


def test_the_topic_page_shows_the_answers_and_never_the_report_half(live, page):
    """The panel a person reads before proposing anything: the diagnosis half
    in full, the report half as one line and not one row."""
    import exam_build as eb
    base, root = live["base"], live["root"]
    page.goto(base + "/#topic=medicine_health")            # deep link, cold
    page.wait_for_selector("[data-panel='answers'] table.jd[data-answers-table]")
    rows = page.locator("table.jd[data-answers-table] tbody tr")
    assert rows.count() > 0
    assert rows.count() == page.locator("tr[data-half='diagnose']").count()
    # the published half is a sentence, and the only sentence
    line = page.locator("[data-report-half]").first.text_content()
    assert "The report half." in line and "published score" in line
    # and no report-half question is anywhere in the document
    bank = eb.load_bank(root / "exam")["medicine & health"]
    report = [b for b in bank if eb.half_of(b["qid"]) == "report"]
    assert report
    html = page.content()
    for b in report:
        assert b["prompt"][:60] not in html and b["qid"] not in html
    # a diagnose-half row carries the question, the answer, the score and the
    # judge's words — and a bar per criterion
    first = rows.first
    assert first.locator("[data-answer-text]").count() == 1
    assert first.locator("[data-criterion]").count() == 15
    assert "/ 4" in first.text_content() or "unreadable" in first.text_content()
    # filters narrow it without a reload
    before = int(page.locator("[data-answer-count]").first.get_attribute("data-answer-count"))
    page.get_by_label("score filter").select_option("weak")
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
    page.wait_for_selector("select")
    opt = page.locator("option[value='judged']")
    assert opt.count() == 1 and opt.is_disabled()
    assert "unavailable" in opt.text_content()
    assert page.errors == []


def test_the_tabs_are_named_once_and_ordered_by_how_often_they_are_opened(live, page):
    """Phase 8e P6c: one name per tab, the hash equal to it, the old hashes
    still landing, and the loop's tabs where the eye lands."""
    base = live["base"]
    page.goto(base + "/")
    page.wait_for_selector("#tabs button")
    labels = page.locator("#tabs button").all_text_contents()
    assert labels[:4] == ["Overview", "Loop", "Models", "Leaderboard"]
    assert "Evals" not in labels and "Provenance" in labels
    # the hash is the label, and the page said so in SERVICE.md
    for label, want in (("Loop", "loop"), ("Models", "models"),
                        ("Submit & Queue", "queue"), ("Provenance", "provenance")):
        page.get_by_role("tab", name=label, exact=True).click()
        page.wait_for_selector("#view > *")
        assert page.evaluate("location.hash") == f"#tab={want}", label
    # the hashes people already pasted somewhere
    for old, label in (("runs", "Provenance"), ("submit", "Submit & Queue"),
                       ("evals", "Provenance")):
        page.goto(f"{base}/#tab={old}")
        page.wait_for_selector("#view > *")
        assert page.locator("#tabs button[aria-selected='true']").inner_text() == label, old
    # the header says what it is, and the theme button says what it does
    page.goto(base + "/")
    page.wait_for_selector("[data-stamp]")
    assert "live · refreshed" in page.locator("[data-stamp]").text_content()
    assert page.locator("#themeBtn").text_content().startswith("Theme")
    assert ":" not in page.locator("#themeBtn").text_content()
    assert "theme:" in page.locator("#themeBtn").get_attribute("title")
    assert page.errors == []


def test_the_loop_tab_says_what_failed_instead_of_loading_forever(live, page):
    """The live tree's /api/loop returned 500 and the board said 'Loading…'
    until someone opened the console. Every other tab already had the 8c
    error line; this one now does too."""
    base = live["base"]
    page.route("**/api/loop", lambda route: route.fulfill(
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
    page.unroute("**/api/loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr", timeout=60000)
    assert page.locator("[data-loop-failed]").count() == 0
    # the 500 we injected is the only thing the console should have to say
    assert all("500" in e for e in page.errors), page.errors


def test_a_topic_on_the_shared_rubric_says_so_on_both_boards(live, page):
    """Thirteen topics have no rubric of their own, and the page says which
    file grades them rather than implying each has one."""
    base = live["base"]
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    hist = page.locator("tr[data-loop-row='history']")
    assert "exam.md" in hist.text_content()
    assert hist.locator("[data-fallback]").count() == 1
    assert page.locator("tr[data-loop-row='law'] [data-fallback]").count() == 0
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
    row = page.locator("tr[data-rubric-row='history']")
    assert "exam.md" in row.text_content() and "(fallback)" in row.text_content()
    assert page.locator("[data-rubric-error]").count() == 0
    assert page.errors == []


def test_a_button_that_calls_the_api_says_what_happened(live, page):
    """The rebuild button answered "refused: 500" in small grey text under
    itself, which the person read as "nothing happens" — and behind it was a
    file missing from the image. Every button that calls the API now says it
    is working, says what happened, and on a refusal says what the SERVER
    said."""
    base = live["base"]
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-action='exbuild']")
    btn = page.locator("[data-action='exbuild']")
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
    page.wait_for_selector("[data-action-ok='exbuild']", timeout=30000)
    ok = page.locator("[data-action-ok='exbuild']").text_content()
    assert "Built" in ok and "task" in ok
    assert "exam_" not in ok                     # topic names, not task ids
    assert "medicine & health (" in ok or "law (" in ok
    assert page.locator("[data-action-error='exbuild']").count() == 0
    # the 500 we injected is the only thing the console should have to say
    assert all("500" in e for e in page.errors), page.errors


def test_a_button_that_is_working_says_so_and_cannot_be_pressed_twice(live, page):
    base = live["base"]
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
    page.wait_for_selector("[data-action-ok='exbuild']", timeout=30000)
    assert btn.is_disabled() is False
    page.unroute("**/api/exam/build")
    assert page.errors == []


def test_typing_survives_the_poll(live, page):
    """Phase 8g D1/D3: the board refreshes every five seconds. A person part
    way through typing their name must not lose it, or the caret, or the
    field — and the name must still be there on the next page they open."""
    base = live["base"]
    page.goto(base + "/#tab=loop")
    page.wait_for_selector("table.jd[data-loop-table] tbody tr")
    name = page.locator("[data-panel], .card").first.get_by_label("your name").first
    name.click()
    name.type("Omar")
    page.wait_for_timeout(6500)                    # two polls land here
    assert page.evaluate("document.activeElement && document.activeElement.dataset.keep"
                         ) is not None             # still in the field
    name.type(" Affifi")
    assert name.input_value() == "Omar Affifi"
    # and it is the same name on the topic page, and after a reload
    page.goto(base + "/#topic=law")
    page.wait_for_selector("[data-topic-page='law']")
    assert page.get_by_label("your name").first.input_value() == "Omar Affifi"
    page.reload()
    page.wait_for_selector("[data-topic-page='law']")
    assert page.get_by_label("your name").first.input_value() == "Omar Affifi"
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


def test_the_queue_row_counts_the_judge_batch_up(live, page):
    """Phase 8g D2: "judging 40/130", not "judging 130 answers"."""
    from service import db
    base = live["base"]
    sid = db.add("fx/good-750m", "auto", "judged", "omar", "", tasks=["exam_law"])
    rid = db.judge_run_create("fx/good-750m", "b_live", 130, "stub/overlap-v1", "{}")
    db.batch_add("b_live", "judge", rid, 130, "anthropic", "claude-x")
    db.batch_progress("b_live", "40/130 done")
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


PHYSICS_FILE = REPO / "eval_tasks" / "fr" / "physics_engineering_v1.json"


def test_a_refusal_replaces_the_last_success_rather_than_sitting_under_it(live, page):
    """After computer science imported, the physics attempt showed the CS
    success line with "refused" beneath it, which reads as a partial import.
    One result line per panel."""
    base = live["base"]
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='import']")
    panel = page.locator("[data-panel='import']")
    panel.get_by_label("your name").first.fill("Omar")
    panel.get_by_label("written by").fill("Dr. Hossein")
    # a good import first
    upload(page, "questions file", "computer_science_v1.json", "application/json",
           (REPO / "eval_tasks" / "fr" / "computer_science_v1.json").read_text(encoding="utf-8"))
    panel.get_by_label("topic").select_option("computer science")
    panel.get_by_role("button", name="Preview").click()
    page.wait_for_selector("[data-action-ok='eximport']", timeout=30000)
    panel.locator("button[data-commit='import']").click()
    page.wait_for_function(
        "() => (document.querySelector(\"[data-action-ok='eximport']\") || {}).textContent"
        "?.startsWith('Imported')", timeout=30000)
    assert "Imported 100" in panel.locator("[data-action-ok='eximport']").text_content()
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
    panel.get_by_label("your name").first.fill("Omar")
    panel.get_by_label("written by").fill("Dr. Hossein")
    upload(page, "questions file", "physics_engineering_v1.json", "application/json",
           PHYSICS_FILE.read_text(encoding="utf-8"))
    panel.get_by_label("topic").select_option("physics & engineering")
    panel.get_by_label("source").fill("physics_engineering_v1")
    panel.get_by_role("button", name="Preview").click()
    page.wait_for_selector("[data-action-ok='eximport']", timeout=30000)
    said = panel.locator("[data-action-ok='eximport']").text_content()
    assert "Read 100 questions" in said and 'from "questions" in the file' in said
    panel.locator("button[data-commit='import']").click()
    page.wait_for_function(
        "() => (document.querySelector(\"[data-action-ok='eximport']\") || {}).textContent"
        "?.startsWith('Imported')", timeout=30000)
    mine = [r for r in eb.load_bank(root / "exam")["physics & engineering"]
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
    bank = eb.bank_dir(root / "exam") / "geography_world_facts.jsonl"
    kept = bank.read_bytes()
    bank.unlink()
    try:
        page.goto(base + "/#tab=exam")
        page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
        panel = page.locator("[data-panel='rubrics']")
        med = panel.locator("tr[data-rubric-row='medicine & health']")
        assert med.locator("[data-bank]").first.get_attribute("data-bank") != "0"
        assert "report" in med.text_content() and "diagnose" in med.text_content()
        empty = panel.locator("tr[data-rubric-row='geography & world facts'] [data-bank='0']")
        assert empty.count() == 1
        assert empty.text_content() == "no questions yet"
        # and an unversioned heading is not printed as an error
        assert "v?" not in panel.text_content()
        assert panel.locator("[data-no-version]").count() >= 1
        assert "no version" in panel.text_content()
        assert page.errors == []
    finally:
        bank.write_bytes(kept)
