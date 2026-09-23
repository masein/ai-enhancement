"""Phase 9c: every action answers, every table fits.

Toasts for what changed, row actions in the queue (including stopping a
running job), topics for a judged Submit, an import panel that makes its
questions sittable by itself, a Leaderboard and a Provenance table that fit,
a model page that leads with numbers, Training opening on a run, and a theme
menu that is a menu."""

from __future__ import annotations

import copy
import json
import re
import os
import sys
import threading
import time

import pytest

from conftest import choose, go_tab, make_service, set_name
from test_page_recovery import Live

MODEL = "fx/good-750m"


# ---------------------------------------------------------------------------
# the API half
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def test_a_queued_job_cancels_and_a_running_one_is_asked_to_stop(svc):
    client, _, _ = svc
    from service import db
    q = db.add("org/queued", "auto", "quick", "omar", "")
    assert client.post(f"/api/submissions/{q}/cancel").json()["status"] == "canceled"
    r = db.add("org/running", "auto", "quick", "omar", "")
    db.update(r, status="running", progress="2/4 · arc_easy")
    assert client.post(f"/api/submissions/{r}/cancel").json()["status"] == "canceling"
    assert db.get(r)["status"] == "canceling" and db.cancel_requested(r)
    # the runner's next progress line does not undo the request
    db.update(r, status="running", progress="3/4 · piqa")
    assert db.get(r)["status"] == "canceling" and db.get(r)["progress"] == "3/4 · piqa"
    # a finished job has nothing to cancel
    d = db.add("org/done", "auto", "quick", "omar", "")
    db.update(d, status="done")
    assert client.post(f"/api/submissions/{d}/cancel").status_code == 409
    # a stop in flight when the service went down is simply done
    db.init()
    assert db.get(r)["status"] == "canceled"


def test_the_runner_stops_a_task_it_was_asked_to_stop(svc, tmp_path):
    from service import db, runner
    sid = db.add("org/long", "auto", "quick", "omar", "")
    db.update(sid, status="running")
    got = {}

    def run():
        with open(tmp_path / "log.txt", "a") as lf:
            got["status"] = runner._run_task(
                sid, [sys.executable, "-c", "import time; time.sleep(60)"], lf,
                dict(os.environ), None)
    t0 = time.time()
    th = threading.Thread(target=run)
    th.start()
    time.sleep(1)
    db.cancel(sid)
    th.join(30)
    assert got["status"] == runner.CANCELED and time.time() - t0 < 20
    assert "canceled by request" in (tmp_path / "log.txt").read_text()


def test_an_import_makes_its_questions_sittable_by_itself(svc):
    client, _, _ = svc
    from service import config
    import exam_build as eb
    body = {"topic": "Law", "approver": "Dr. Hossein", "imported_by": "Omar",
            "filename": "extra.json",
            "items": [{"prompt": f"A fresh law question number {i} about the doctrine of "
                                 f"estoppel in contract formation?",
                       "reference": "Promissory estoppel prevents going back on a promise."}
                      for i in range(3)]}
    got = client.post("/api/exam/import", json=body).json()
    assert got["build"]["built"] is True
    tasks = (eb.tasks_dir(config.EXAM_DIR) / "exam_law.jsonl").read_text(encoding="utf-8")
    assert "doctrine of estoppel" in tasks
    assert client.get("/api/exam").json()["tasks_stale"] is False


def test_it_waits_while_a_judged_run_is_sitting_the_exam(svc):
    client, _, _ = svc
    from service import db
    sid = db.add(MODEL, "auto", "judged", "omar", "")
    db.update(sid, status="running")
    body = {"topic": "Law", "approver": "Dr. Hossein", "imported_by": "Omar",
            "items": [{"prompt": "What does consideration mean in the formation of a "
                                 "contract, and why does it matter?",
                       "reference": "Something of value exchanged."}]}
    got = client.post("/api/exam/import", json=body).json()
    assert got["build"]["built"] is False and f"#{sid}" in got["build"]["why"]
    assert client.get("/api/exam").json()["tasks_stale"] is True        # the page will offer it


# ---------------------------------------------------------------------------
# the browser half
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_submit_answers_with_a_toast_and_no_line_that_stays(live, page):
    page.goto(live["base"] + "/#tab=queue")
    set_name(page, "Omar")
    box = page.locator("[data-ms='submit'] input")
    box.fill("org/toast-me")
    page.get_by_role("button", name="Submit model").click()
    t = page.locator("[data-toast='submit']")
    t.wait_for()
    # 11l: the confirmation names the run it made and links to its row
    assert re.match(r"^Run #\d+ queued —", t.locator(".toast-text").text_content())
    assert t.locator("[data-toast-link]").text_content() == "follow it →"
    assert page.locator("[role=status] [data-toast='submit']").count() == 1   # a live region
    # a toast with a link stays eight seconds (10c: four was gone before
    # anyone reached "see it")
    page.wait_for_timeout(5000)
    assert t.count() == 1
    page.wait_for_selector("[data-toast='submit']", state="detached", timeout=8000)
    assert "queued" not in (page.locator("#view .card").first.text_content() or "").lower() \
        or "Queued #" not in page.locator("#view").text_content()
    assert page.errors == []


@pytest.mark.dashboard
def test_queue_rows_have_the_actions_their_state_allows(live, page):
    from service import db
    base = live["base"]
    q = db.add("org/cancel-me", "auto", "quick", "omar", "")
    r = db.add("org/stop-me", "auto", "quick", "omar", "")
    db.update(r, status="running", progress="1/4 · arc_easy")
    f = db.add("org/failed-one", "instruct", "quick", "omar", "a note")
    db.update(f, status="failed", error="arc_easy: out of memory")
    d = db.add(MODEL, "base", "judged", "omar", "", tasks=["exam_law"])
    rid = db.judge_run_create(MODEL, "b_open", 42, "stub/overlap-v1", "{}")
    db.judge_run_update(rid, status="done", finished_at=time.time())
    db.batch_add("b_open", "judge", rid, 42, "local", "chat")
    db.batch_finish("b_open", "done", "")
    db.update(d, status="done", judge_batch="b_open")          # answered and judged
    page.goto(base + "/#tab=queue")
    page.wait_for_selector(f"[data-row-cancel='{q}']")
    # queued: Cancel, at once
    page.locator(f"[data-row-cancel='{q}']").click()
    page.wait_for_selector("[data-toast='cancel']")
    assert db.get(q)["status"] == "canceled"
    # running: Cancel asks first
    page.locator(f"[data-row-cancel='{r}']").click()
    stop = page.locator(f"[data-row-stop='{r}']")
    stop.wait_for()
    assert "Stop this run?" in stop.locator("xpath=..").text_content()
    stop.click()
    page.wait_for_function(f"state.queue.some(x => x.id === {r} && x.status === 'canceling')")
    assert db.get(r)["status"] == "canceling"
    # failed: the reason is on the row, and Resubmit is one click
    row = page.locator(f"tr:has([data-row-resubmit='{f}'])")
    assert "out of memory" in row.text_content()
    n = len(db.recent(500))
    row.locator(f"[data-row-resubmit='{f}']").click()
    page.wait_for_selector("[data-toast='resubmit']")
    again = db.recent(500)
    assert len(again) == n + 1 and again[0]["hf_id"] == "org/failed-one" \
        and again[0]["suite"] == "quick"
    # done, judged on one topic: Open results lands on that topic's answers
    page.locator(f"[data-row-open='{d}']").click()
    page.wait_for_selector("[data-topic-page='law']")
    page.wait_for_function("m => state.ans.model === m", arg=MODEL)
    assert page.errors == []


@pytest.mark.dashboard
def test_a_queued_submission_clears_the_form_and_links_to_its_row(live, page, monkeypatch):
    """11l: on success the form goes back to where it started, and the
    confirmation takes you to the row it made."""
    from service import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")          # the judged suite is on
    page.goto(live["base"] + "/#tab=queue")
    set_name(page, "Omar")
    choose(page.get_by_label("suite"), "judged")
    picker = page.locator("[data-exam-picker='submit']")
    picker.wait_for()
    picker.locator("[data-quick='none']").click()               # not the default ticks
    picker.locator("input[data-exam-task='exam_law']").check()
    ctl = picker.locator("[data-exam-control]")
    if ctl.count():
        ctl.check()
    box = page.locator("[data-ms='submit'] input")
    box.fill("org/clears-itself")
    note = page.get_by_label("note")
    note.fill("a note that goes")
    page.get_by_role("button", name="Submit model").click()
    t = page.locator("[data-toast='submit']")
    t.wait_for()
    rid = int(re.search(r"Run #(\d+)", t.locator(".toast-text").text_content()).group(1))
    # follow it → marks that row and brings it into view. Straight away: the
    # mark is for finding it, and it clears itself after a few seconds
    t.locator("[data-toast-link]").click()
    row = page.locator(f"tr[data-queue-row='{rid}']")
    row.wait_for()
    # the scroll lands on the next frame, so wait for it rather than guess
    page.wait_for_function("id => { const e = document.querySelector("
                           "`tr[data-queue-row='${id}']`); if (!e) return false;"
                           " const r = e.getBoundingClientRect();"
                           " return r.bottom > 0 && r.top < innerHeight; }", arg=rid)
    assert "landed" in (row.get_attribute("class") or "")
    # the model box and the note are empty, and the button can be pressed again
    assert box.input_value() == ""
    assert note.input_value() == ""
    assert page.get_by_role("button", name="Submit model").is_enabled()
    # and the ticks are the ones the form opens with: every topic, no control
    boxes = picker.locator("input[data-exam-task]:not([disabled])")
    assert boxes.count() > 1
    assert all(boxes.nth(i).is_checked() for i in range(boxes.count()))
    if ctl.count():
        assert not ctl.is_checked()
    assert page.errors == []


@pytest.mark.dashboard
def test_the_submit_button_is_held_while_the_request_is_in_flight(live, page):
    page.goto(live["base"] + "/#tab=queue")
    set_name(page, "Omar")
    page.locator("[data-ms='submit'] input").fill("org/held-while-in-flight")
    # read the button one tick after the click, while the POST is out
    page.evaluate("""() => { const b = [...document.querySelectorAll('button')]
        .find(x => x.textContent === 'Submit model');
      window.__held = [];
      b.addEventListener('click', () => setTimeout(
        () => window.__held.push([b.textContent, b.disabled]), 0));
      b.click(); }""")
    page.wait_for_selector("[data-toast='submit']")
    held = page.evaluate("window.__held")
    assert held and held[0] == ["Queueing…", True], held
    assert page.get_by_role("button", name="Submit model").is_enabled()
    assert page.errors == []


@pytest.mark.dashboard
def test_a_refused_submission_keeps_every_field_and_says_why(live, page):
    page.goto(live["base"] + "/#tab=queue")
    set_name(page, "Omar")
    box = page.locator("[data-ms='submit'] input")
    box.fill("not-a-model-id")
    note = page.get_by_label("note")
    note.fill("keep me")
    page.get_by_role("button", name="Submit model").click()
    msg = page.locator("[data-qmsg]")
    msg.wait_for()
    assert msg.text_content().startswith("Refused. ")
    assert "org/name" in msg.text_content()            # the server's own words
    assert box.input_value() == "not-a-model-id"       # nothing was cleared
    assert note.input_value() == "keep me"
    assert page.get_by_role("button", name="Submit model").is_enabled()
    assert page.locator("[data-toast='submit']").count() == 0
    # the refusal itself is the 422 the browser logs; nothing else
    assert all("422" in e for e in page.errors), page.errors


@pytest.mark.dashboard
def test_a_judged_submit_chooses_its_topics(live, page, monkeypatch):
    from service import config, db
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")         # the judged suite is on
    page.goto(live["base"] + "/#tab=queue")
    set_name(page, "Omar")
    choose(page.get_by_label("suite"), "judged")
    # 11i: the model page's grouped picker; every topic with questions starts ticked
    boxes = page.locator("[data-submit-topics] input[data-exam-task]:not([disabled])")
    boxes.first.wait_for()
    assert all(boxes.nth(i).is_checked() for i in range(boxes.count()))   # the whole exam
    page.locator("[data-exam-picker='submit'] [data-quick='none']").click()
    page.locator("[data-exam-picker='submit'] input[data-exam-task='exam_law']").check()
    page.locator("[data-ms='submit'] input").fill("org/judged-law-only")
    page.get_by_role("button", name="Submit model").click()
    page.wait_for_selector("[data-toast='submit']")
    row = next(x for x in db.recent(50) if x["hf_id"] == "org/judged-law-only")
    assert json.loads(row["tasks"]) == ["exam_law"] and row["suite"] == "judged"
    assert page.errors == []


@pytest.mark.dashboard
def test_the_import_panel_labels_its_fields_and_says_when_nothing_is_new(live, page):
    from conftest import ROOT as REPO
    base = live["base"]
    # law as first delivered, retired with the 37-topic exam: this is about the
    # panel, and any file of law questions will do
    raw = (REPO / "eval_tasks" / "fr" / "retired" / "law_v2.json").read_text(encoding="utf-8")
    page.goto(base + "/#tab=exam")
    panel = page.locator("[data-panel='import']")
    panel.wait_for()
    labels = panel.locator(".fld-label").all_text_contents()
    assert labels == ["Questions file", "Topic", "Written by", "Source"]
    set_name(page, "Omar")
    panel.get_by_label("written by").fill("Dr. Hossein")
    items = json.loads(raw)[:3]

    def upload(content):
        page.locator("[data-panel='import'] input[type=file]").set_input_files(
            files=[{"name": "law_v2.json", "mimeType": "application/json",
                    "buffer": json.dumps(content).encode()}])
        page.wait_for_selector("[data-source='law_v2']")
    upload(items)
    choose(panel.get_by_label("topic"), "Law")
    panel.get_by_role("button", name="Preview").click()
    page.wait_for_selector("[data-panel='import'] [data-commit='import']")
    btn = panel.locator("[data-commit='import']")
    if btn.is_enabled():
        btn.click()
        page.wait_for_selector("[data-toast='import']")
        assert "can be sat now" in page.locator("[data-toast='import']").text_content()
        upload(items)
        panel.get_by_role("button", name="Preview").click()
        page.wait_for_function("(document.querySelector(\"[data-commit='import']\") || {})"
                               ".textContent === 'Nothing new to import'")
    btn = panel.locator("[data-commit='import']")
    assert btn.text_content() == "Nothing new to import" and btn.is_disabled()
    assert page.errors == []


@pytest.mark.dashboard
@pytest.mark.parametrize("width", [1280, 1512])
def test_no_tab_scrolls_sideways(live, browser, width):
    ctx = browser.new_context(viewport={"width": width, "height": 900}, reduced_motion="reduce")
    pg = ctx.new_page()
    try:
        pg.goto(live["base"] + "/")
        pg.wait_for_selector("#view > *")
        for label in ("Overview", "Loop", "Models", "Leaderboard", "Queue", "Exam", "Review",
                      "Training", "Tasks", "Perplexity & Loss", "Provenance"):
            go_tab(pg, label)
            pg.wait_for_selector("#view > *")
            pg.wait_for_timeout(300)
            fits = pg.evaluate("document.documentElement.scrollWidth <= "
                               "document.documentElement.clientWidth")
            assert fits, f"{label} scrolls sideways at {width}px"
        # the paged tables fit their own container, too
        for label, sel in (("Leaderboard", "table[data-lb-table]"), ("Provenance", "table.prov"),
                           ("Models", "table[data-models-table]"), ("Queue", "table[data-queue-table]")):
            go_tab(pg, label)
            pg.wait_for_selector("#view > *")
            t = pg.locator(sel)
            # an empty queue hides its table: nothing to measure, and a hidden
            # parent has no width, which reads as nan
            if t.count() and t.first.is_visible():
                ok = t.first.evaluate("t => t.parentElement.scrollWidth <= t.parentElement.clientWidth + 1")
                assert ok, f"{label}'s table is wider than its card at {width}px"
                # and with room to spare: the narrowest the table can be, at most 95%
                # of its card — fonts on another machine run wider (CI's did)
                share = t.first.evaluate("""t => { const w = t.style.width;
                  t.style.width = 'min-content'; const n = t.scrollWidth; t.style.width = w;
                  return n / t.parentElement.clientWidth; }""")
                assert share <= 0.95, f"{label}'s table needs {share:.0%} of its card at {width}px"
    finally:
        ctx.close()


@pytest.mark.dashboard
def test_the_leaderboard_shows_six_task_columns_and_says_how_many_are_hidden(browser, payload):
    p = copy.deepcopy(payload)
    ranked = sorted((m for m in p["models"] if m.get("avg") is not None), key=lambda m: -m["avg"])
    ranked[1].update(duplicateOf=ranked[0]["id"], duplicateOfName=ranked[0]["name"],
                     duplicateWhy="every score and item count is identical")
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, reduced_motion="reduce")
    s = Live(ctx, p, fail=False)
    try:
        pg = s.open()
        go_tab(pg, "Leaderboard")
        pg.wait_for_selector("table[data-lb-table]")
        # 11c: two header rows — the group over its columns, then the names
        heads = pg.locator("table[data-lb-table] thead tr:not(.grp) th")
        task_cols = pg.locator("table[data-lb-table] thead th[data-task]").count()
        assert task_cols <= 6
        # every task-like column (benchmarks, perplexity, judged topics) is in
        # the Columns popover; what is not shown is counted out loud on the pill
        pg.locator("[data-columns-menu]").click()
        pg.wait_for_selector("#pop-columns")
        n_cols = pg.locator("#pop-columns input[data-column]").count()
        n_shown = pg.locator("#pop-columns input[data-column]:checked").count()
        assert n_shown <= 6 and n_cols > n_shown
        hidden = pg.locator("[data-hidden-tasks]")
        assert hidden.get_attribute("data-hidden-tasks") == str(n_cols - n_shown)
        assert "hidden" in hidden.text_content()
        pg.keyboard.press("Escape")
        # the rank comes first, then the model, which stays put; there is no
        # compare column any more — the radar's model chips are the comparison
        assert heads.nth(0).get_attribute("data-col") == "rank"
        assert heads.nth(1).get_attribute("data-col") == "name"
        assert pg.locator("table[data-lb-table] thead th.cmp").count() == 0
        assert pg.locator("table[data-lb-table] td.model").first.evaluate(
            "e => getComputedStyle(e).position") == "sticky"
        # the duplicate folds under its twin
        assert pg.locator(f"tr[data-lb-row='{ranked[1]['id']}']").count() == 0
        pg.locator(f"[data-dup-toggle='{ranked[0]['id']}']").click()
        pg.wait_for_selector(f"tr.duprow[data-lb-row='{ranked[1]['id']}']")
        # 11b removed the comfortable/compact switch: the one-line row IS the
        # compact one, so there is nothing left to choose between
        assert pg.get_by_role("button", name="compact").count() == 0
        # the Columns popover brings a column back, and stays open to say so
        pg.locator("[data-columns-menu]").click()
        pg.locator("#pop-columns [data-show-all]").click()
        pg.wait_for_function("!document.querySelector('[data-hidden-tasks]')")
        assert pg.locator("#pop-columns input[data-column]:not(:checked)").count() == 0
        assert s.errors == []
    finally:
        ctx.close()


@pytest.mark.dashboard
def test_the_model_page_leads_with_numbers(live, page):
    from service import app, config
    base = live["base"]
    jf = config.OUT_DIR / "fx__good-750m" / "judge.json"
    kept = jf.read_bytes()
    j = json.loads(kept)
    j["judge"].update({"provisional": True, "provisional_reason": "graded by a local model"})
    for t in j["tasks"].values():
        t["judged_at"] = 1790000000                        # 2026-09-21
    jf.write_text(json.dumps(j), encoding="utf-8")
    app._cache.update(key=None, payload=None, at=0.0)
    try:
        page.goto(base + "/#model=" + MODEL.replace("/", "%2F"))
        card = page.locator(".card", has=page.locator("h2", has_text="Judged free response"))
        card.wait_for()
        line = card.locator("[data-caveats]")
        assert line.count() == 1
        assert line.locator("[data-caveat='provisional']").count() == 1
        why = line.locator("details.caveat-why")
        assert why.get_attribute("open") is None
        assert not card.locator("[data-provisional='judge']").is_visible()   # behind "why?"
        why.locator("summary").click()
        assert card.locator("[data-provisional='judge']").is_visible()
        # one topic's tables at a time, picked from a searchable list (10c: 36
        # of them; 11f: a Combobox)
        box = card.locator("[data-topic-switch]")
        box.click()
        sw = page.locator("#pop-cb-mdl-topic [role=option]")
        assert sw.count() >= 2
        assert card.locator("[data-criteria-table]").count() == 1
        second = [o.get_attribute("data-value") for o in sw.all()
                  if o.get_attribute("aria-selected") != "true"][0]
        page.keyboard.press("Escape")
        choose(box, second)
        page.wait_for_function("document.querySelectorAll('[data-criteria-table]').length === 1 "
                               "&& document.querySelector('[data-topic-switch]').dataset.value === "
                               f"'{second}'")
        # no "Training compute: Unknown" — the hero's cards (11d) say it only when known
        assert "Training compute" not in page.locator("[data-model-hero]").text_content()
        # last evaluated counts the judged run
        assert "Last evaluated 2026-09-21" in page.locator("#view").text_content()
        assert page.errors == []
    finally:
        jf.write_bytes(kept)
        app._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.dashboard
def test_training_opens_on_the_most_recent_run(live, page):
    import urllib.request

    def post(path, body):
        req = urllib.request.Request(live["base"] + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())
    post("/api/truns", {"name": "older-run"})
    time.sleep(1.1)
    newest = post("/api/truns", {"name": "newest-run"})["id"]
    page.goto(live["base"] + "/#tab=training")
    page.wait_for_function("id => state.trSel.includes(id)", arg=newest)
    assert "Select a run on the left" not in page.locator("#view").text_content()
    assert page.errors == []


@pytest.mark.dashboard
def test_the_theme_is_a_menu(live, page):
    page.goto(live["base"] + "/")
    btn = page.locator("#themeBtn")
    btn.click()
    # on the shared popover since 11a: the panel is on the body, keyed pop-theme
    items = page.locator("#pop-theme [role=menuitemradio]")
    assert items.count() == 4
    assert [i.get_attribute("aria-checked") for i in items.all()].count("true") == 1
    page.locator("#pop-theme [data-theme='dark']").click()
    assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "dark"
    btn.click()
    assert page.locator("#pop-theme [data-theme='dark']").get_attribute("aria-checked") == "true"
    page.keyboard.press("Escape")
    assert page.locator("#pop-theme").count() == 0
    assert page.evaluate("document.activeElement.id") == "themeBtn"   # Esc gives it back
    assert page.errors == []


def test_two_checkpoints_of_one_run_have_different_labels():
    import subprocess
    js = ("const midTrunc = (s, n) => s.length <= n ? s : s.slice(0, Math.ceil((n - 1) * 0.45))"
          " + '…' + s.slice(s.length - Math.floor((n - 1) * 0.55));"
          "console.log(midTrunc('qwen35-delta-moe-7d560104-step945', 18));"
          "console.log(midTrunc('qwen35-delta-moe-7d560104-step945-v2', 18));")
    import report_lm_eval as report
    assert "const midTrunc = (s, n) => s.length <= n ? s" in report.JS
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True).stdout.split()
    assert len(out) == 2 and out[0] != out[1] and out[0].endswith("step945")
