"""11i: sit the exam from the model page.

masein: "it should be from the model page, so I can click to start on all or
some of the topics for a model." The panel lists every topic under its area
with where this model stands on it; quick picks tick exactly their set; the
MMLU control is its own box, off by default; the summary follows the ticks;
Queue this run sends exactly what is ticked. The Queue form's judged suite
uses the same picker. A checkpoint that ships its own model code says so
before it is queued — the box when the server runs it, the reason when it
does not, and a 422 from the API either way, never a failure at start. A
judged row's Suite cell is one line.
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from conftest import choose, make_service, set_name, open_submit

SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11i"
MODEL = "fx/good-750m"                  # judged on every topic with questions
FRESH = "fx/short-pick-410m"            # never judged
LOCAL = "local/nodiag-step400"          # on the board, under a local id
UPLOAD = "custom-arch"                  # an upload that is not on the board
AUTO_MAP = {"AutoConfig": "configuration_custom.CustomConfig",
            "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"}
MODELING = "class CustomForCausalLM:\n    pass\n"


def own_code_upload(artifacts: Path, name: str) -> str:
    """An uploaded checkpoint whose config.json carries an auto_map, and the
    sha the server gives its one .py file."""
    d = artifacts / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"auto_map": AUTO_MAP, "vocab_size": 128,
                                               "architectures": ["CustomForCausalLM"]}))
    (d / "modeling_custom.py").write_text(MODELING)
    return hashlib.sha256(MODELING.encode()).hexdigest()[:12]


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def round_about(n: int) -> int:
    return n if n < 100 else round(n / 10) * 10 if n < 1000 else round(n / 100) * 100


# ---------------------------------------------------------------------------
# the API: an upload's own code is answered before anything is queued
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    from service import config
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    yield client, config
    client.__exit__(None, None, None)


def test_own_code_on_a_server_that_does_not_run_it_is_a_422_and_nothing_is_queued(svc):
    from service import db
    client, config = svc
    own_code_upload(config.ARTIFACTS_DIR, UPLOAD)
    before = len(db.recent(500))
    code = client.get("/api/models/code", params={"id": f"local/{UPLOAD}"}).json()
    assert code["own_code"] and code["weights"] is True
    assert [f["file"] for f in code["files"]] == ["modeling_custom.py"]
    for flag in (False, True):
        r = client.post("/api/submissions", json={"hf_id": f"local/{UPLOAD}", "suite": "quick",
                                                  "allow_remote_code": flag})
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        # the two settings and where to read about them — the page's words
        for words in ("ALLOW_REMOTE_CODE=1", "EVAL_USER", "SERVICE.md § custom model code",
                      "Nothing was queued"):
            assert words in detail
        assert detail.startswith(code["why"])
    assert len(db.recent(500)) == before


def test_own_code_on_a_server_that_runs_it_needs_the_box(svc, monkeypatch):
    from service import db
    client, config = svc
    sha = own_code_upload(config.ARTIFACTS_DIR, UPLOAD)
    monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", True)
    monkeypatch.setattr(config, "EVAL_USER", getpass.getuser())
    code = client.get("/api/models/code", params={"id": f"local/{UPLOAD}"}).json()
    assert code["own_code"] and code["why"] == "" and code["files"][0]["sha"] == sha
    assert code["user"] == getpass.getuser()
    r = client.post("/api/submissions", json={"hf_id": f"local/{UPLOAD}", "suite": "quick"})
    assert r.status_code == 422 and "Run this checkpoint's own model code" in r.json()["detail"]
    assert sha in r.json()["detail"]
    r = client.post("/api/submissions", json={"hf_id": f"local/{UPLOAD}", "suite": "quick",
                                              "allow_remote_code": True})
    assert r.status_code == 200, r.text
    row = db.get(r.json()["id"])
    assert row["allow_remote_code"] == 1 and row["status"] == "queued"


def test_a_sha_off_the_allowlist_is_named_with_the_sha_to_add(svc, monkeypatch):
    client, config = svc
    sha = own_code_upload(config.ARTIFACTS_DIR, UPLOAD)
    monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", True)
    monkeypatch.setattr(config, "EVAL_USER", getpass.getuser())
    monkeypatch.setattr(config, "REMOTE_CODE_SHAS", {"000000000000"})
    code = client.get("/api/models/code", params={"id": f"local/{UPLOAD}"}).json()
    assert f"modeling_custom.py (sha {sha})" in code["why"]
    assert f"add {sha} to REMOTE_CODE_SHAS" in code["why"]
    r = client.post("/api/submissions", json={"hf_id": f"local/{UPLOAD}", "suite": "quick",
                                              "allow_remote_code": True})
    assert r.status_code == 422 and sha in r.json()["detail"]
    # on the list, it runs
    monkeypatch.setattr(config, "REMOTE_CODE_SHAS", {sha})
    assert client.get("/api/models/code", params={"id": f"local/{UPLOAD}"}).json()["why"] == ""


def test_the_search_says_so_and_a_plain_upload_or_a_path_is_not_asked_about(svc):
    client, config = svc
    own_code_upload(config.ARTIFACTS_DIR, UPLOAD)
    plain = config.ARTIFACTS_DIR / "plain-ckpt"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"vocab_size": 128}))
    items = client.get("/api/models/suggest", params={"q": "arch"}).json()["items"]
    it = next(i for i in items if i["id"] == f"local/{UPLOAD}")
    assert it["own_code"] == {"runs": False}
    items = client.get("/api/models/suggest", params={"q": "plain"}).json()["items"]
    assert "own_code" not in next(i for i in items if i["id"] == "local/plain-ckpt")
    assert client.get("/api/models/code", params={"id": "local/plain-ckpt"}).json() == {
        "own_code": False, "weights": True}
    # a name, never a path: nothing under the bench root is hashed for a caller
    for sneaky in ("local/..", "local/../results", "local/.hidden"):
        j = client.get("/api/models/code", params={"id": sneaky}).json()
        assert j.get("own_code") is False and "files" not in j and "weights" not in j
    assert client.get("/api/models/code", params={"id": "org/model"}).json() == {
        "own_code": False}


# ---------------------------------------------------------------------------
# the model page's panel
# ---------------------------------------------------------------------------

@pytest.fixture
def judged_on(monkeypatch):
    from service import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")         # the judged suite is on


def open_panel(page, base, model=MODEL):
    page.goto(base + "/#model=" + model.replace("/", "%2F"))
    page.locator(f"[data-sit-open='{model}']").click()
    page.wait_for_selector("[data-panel='msit'] [data-exam-picker='msit']")
    return page.locator("[data-panel='msit']")


def ticked(page, key="msit"):
    return sorted(page.eval_on_selector_all(
        f"[data-exam-picker='{key}'] input[data-exam-task]:checked",
        "xs => xs.map(x => x.dataset.examTask)"))


def statuses(page, key="msit"):
    return page.eval_on_selector_all(
        f"[data-exam-picker='{key}'] [data-exam-topic]",
        "xs => Object.fromEntries(xs.map(x => [x.dataset.examTopic, "
        "[x.dataset.status, x.querySelector('[data-exam-status]').textContent, "
        "x.querySelector('input').disabled]]))")


def cleanup_rows(model):
    from service import db
    for r in db.recent(500):
        if r["hf_id"] == model and r["status"] in ("queued", "running"):
            db.update(r["id"], status="canceled")


@pytest.mark.dashboard
def test_the_panel_holds_every_topic_under_its_area_with_this_models_standing(
        live, page, judged_on):
    import urllib.request
    base = live["base"]
    panel = open_panel(page, base)
    # the hero's button opened it here, scrolled to
    assert page.evaluate("document.querySelector('[data-panel=msit]').getBoundingClientRect()"
                         ".top < innerHeight")
    assert panel.locator("[data-exam-topic]").count() == 37
    assert panel.locator("[data-area]").count() == 8
    payload = json.loads(urllib.request.urlopen(base + "/api/results").read())
    m = next(x for x in payload["models"] if x["id"] == MODEL)
    loop = json.loads(urllib.request.urlopen(base + "/api/loop").read())
    built = set(loop["tasks_built"])
    st = statuses(page)
    for task, v in (m["judge"]["tasks"] or {}).items():
        if not task.startswith("exam_"):
            continue
        score = v.get("score_report")
        kind, text, _ = st[task]
        assert kind == "judged", (task, st[task])
        # the board's number format: 0.79 / 4, and 4 / 4 rather than 4.00; a
        # topic with no hidden questions yet was graded on practice ones only
        assert text.split(" / 4")[0] == f"{score:.2f}".rstrip("0").rstrip(".") \
            if score is not None else text.startswith("no hidden questions yet"), (task, text)
        day = dt.datetime.fromtimestamp(v["judged_at"]).strftime("%-d %b") if v.get(
            "judged_at") else None
        assert (text.endswith(f" · {day}") if day else " · " not in text), (task, text)
    for task, (kind, text, off) in st.items():
        if task not in built:
            assert (kind, text, off) == ("nobank", "no questions yet", True)
        elif task not in m["judge"]["tasks"]:
            assert (kind, text, off) == ("notsat", "not sat", False)
    # the model's history: a topic whose grades are on questions the exam no
    # longer holds (10b) moves from judge.tasks to history
    older = next(t for t, (k, _, _) in st.items() if k == "judged")
    page.evaluate("t => { const j = DATA.models.find(m => m.id === 'fx/good-750m').judge; "
                  "delete j.tasks[t]; j.history = [{ task: t, score_report: 2.5 }]; "
                  "state.msitRedraw(); }", older)
    assert statuses(page)[older][:2] == ["older", "on older questions"]
    assert page.errors == []


@pytest.mark.dashboard
def test_each_quick_pick_ticks_exactly_its_set_and_the_summary_follows(live, page, judged_on):
    import urllib.request
    base = live["base"]
    items = json.loads(urllib.request.urlopen(base + "/api/loop").read())["built_items"]
    # a model judged on every topic: nothing left to sit, the five weakest to sit again
    panel = open_panel(page, base, MODEL)
    st = statuses(page)
    open_ = sorted(t for t, (_, _, off) in st.items() if not off)
    judged = sorted((float(txt.split(" / ")[0]), t) for t, (k, txt, off) in st.items()
                    if k == "judged" and not off and " / 4" in txt)
    assert len(judged) >= 5
    assert ticked(page) == []                                   # nothing, to begin with
    assert panel.locator("[data-quick='notsat']").text_content() == "Not sat yet (0)"
    assert panel.locator("[data-quick='notsat']").is_disabled()
    panel.locator("[data-quick='all']").click()
    assert ticked(page) == open_
    assert panel.locator("[data-quick='all']").text_content() == f"All {len(open_)}"
    panel.locator("[data-quick='weakest']").click()
    assert ticked(page) == sorted(t for _, t in judged[:5])
    summary = panel.locator("[data-exam-summary]")
    # the same questions are graded again: no answers, no GPU, and the rows say so
    assert summary.text_content() == "5 topics · 5 re-grade only · no GPU, grading only"
    for _, t in judged[:5]:
        assert panel.locator(f"[data-regrade='{t}']").text_content() == \
            "same questions — re-grade only, no GPU"
        assert panel.locator(f"[data-regrade='{t}']").is_visible()
    panel.locator("[data-quick='none']").click()
    assert ticked(page) == []
    assert summary.text_content() == "Tick the topics this model should sit."
    # a model never judged: everything is not sat, and there is no weakest
    panel = open_panel(page, base, FRESH)
    st = statuses(page)
    open_ = sorted(t for t, (_, _, off) in st.items() if not off)
    assert all(st[t][:2] == ["notsat", "not sat"] for t in open_)
    assert panel.locator("[data-quick='weakest']").is_disabled()
    panel.locator("[data-quick='notsat']").click()
    assert ticked(page) == open_
    assert panel.locator("[data-quick='notsat']").text_content() == f"Not sat yet ({len(open_)})"
    assert "consider a quiet time" in summary.text_content()
    panel.locator("[data-quick='none']").click()
    two = open_[:2]
    for t in two:
        panel.locator(f"input[data-exam-task='{t}']").check()
    n = sum(items[t] for t in two)
    assert summary.text_content().startswith(f"2 topics · about {round_about(n):,} answers · ")
    assert summary.text_content().endswith(" · the GPU is shared")
    # the area's box ticks the whole area, and says how many
    area = panel.locator("[data-area]").first
    area.locator("[data-area-box]").check()
    inside = area.locator("input[data-exam-task]:not([disabled])")
    assert all(inside.nth(i).is_checked() for i in range(inside.count()))
    # 11k: "0 of 5" read as judged; the heading says both now
    assert area.locator(".excount").text_content() == (
        f"{inside.count()} ticked · 0 of {area.locator('[data-exam-topic]').count()} judged")
    # the control is off by default, and its own box
    ctl = panel.locator("[data-exam-control]")
    if ctl.count():
        assert not ctl.is_checked()
        ctl.check()
        assert " + MMLU control · " in summary.text_content()
    assert page.errors == []


@pytest.mark.dashboard
def test_queue_this_run_sends_exactly_the_ticks_and_the_rows_follow_the_queue(
        live, page, judged_on):
    from service import db
    base = live["base"]
    try:
        panel = open_panel(page, base, FRESH)
        set_name(page, "masein")
        st = statuses(page)
        pick = sorted(t for t, (k, _, off) in st.items() if k == "notsat" and not off)[:3]
        for t in pick:
            panel.locator(f"input[data-exam-task='{t}']").check()
        go = panel.locator("[data-msit-go]")
        assert go.is_enabled()
        go.click()
        page.wait_for_selector("[data-toast='msit']")
        row = next(r for r in db.recent(20) if r["hf_id"] == FRESH and r["suite"] == "judged")
        assert sorted(json.loads(row["tasks"])) == pick          # no control: it was not ticked
        assert row["allow_remote_code"] == 0
        # each ticked topic now says where it is, and cannot be ticked again
        page.wait_for_function("ts => ts.every(t => document.querySelector("
                               "`[data-exam-status='${t}']`).textContent === 'in the queue')",
                               arg=pick)
        assert all(panel.locator(f"input[data-exam-task='{t}']").is_disabled() for t in pick)
        assert ticked(page) == []
        db.update(row["id"], status="running")
        page.evaluate("loadQueue()")
        page.wait_for_function("t => document.querySelector(`[data-exam-status='${t}']`)"
                               ".textContent === 'running…'", arg=pick[0])
        # the MMLU control only when its box is ticked
        ctl = panel.locator("[data-exam-control]")
        if ctl.count():
            free = sorted(t for t, (k, _, off) in statuses(page).items() if not off)[:1]
            panel.locator(f"input[data-exam-task='{free[0]}']").check()
            ctl.check()
            go.click()
            page.wait_for_function("n => document.querySelectorAll(\"[data-toast='msit']\")"
                                   ".length >= 1", arg=1)
            page.wait_for_function(
                f"() => state.queue.some(r => r.hf_id === '{FRESH}' && "
                f"(r.tasks || '').includes('fr_control_mmlu'))")
            row2 = next(r for r in db.recent(20) if r["hf_id"] == FRESH
                        and "fr_control_mmlu" in (r["tasks"] or ""))
            assert sorted(json.loads(row2["tasks"])) == sorted(free + ["fr_control_mmlu"])
        assert page.errors == []
    finally:
        cleanup_rows(FRESH)


@pytest.mark.dashboard
def test_every_way_in_opens_the_panel_on_the_model_page(live, page, judged_on):
    base = live["base"]
    # a Models row opens the model page (12b: rows do not open in place), and
    # its Sit the exam opens the panel
    page.goto(base + "/#tab=leaderboard")
    page.locator(f"table.lb tbody tr[data-lb-row='{MODEL}'] td.num").first.click()
    page.locator(f"[data-sit-open='{MODEL}']").click()
    page.wait_for_selector("[data-panel='msit']")
    assert page.evaluate("state.model") == MODEL
    # the Overview's loop card
    page.goto(base + "/")
    page.locator(f"[data-overview-loop] [data-loop-sit='{MODEL}']").click()
    page.wait_for_selector("[data-panel='msit']")
    assert page.evaluate("state.model") == MODEL
    # the Loop tab, beside "Results for"
    page.goto(base + "/#tab=loop")
    btn = page.locator("[data-loop-sit]")
    btn.wait_for()
    who = btn.get_attribute("data-loop-sit")
    btn.click()
    page.wait_for_selector("[data-panel='msit']")
    assert page.evaluate("state.model") == who
    # ✕ closes it; the hero's button opens it again
    page.locator("[data-msit-close]").click()
    assert page.locator("[data-panel='msit']").count() == 0
    assert page.errors == []


@pytest.mark.dashboard
def test_the_queue_forms_judged_suite_uses_the_same_picker(live, page, judged_on):
    base = live["base"]
    open_submit(page, base)
    choose(page.get_by_label("suite"), "judged")
    picker = page.locator("[data-submit-topics] [data-exam-picker='submit']")
    picker.wait_for()
    assert picker.locator("[data-exam-topic]").count() == 37
    assert picker.locator("[data-area]").count() == 8
    assert picker.locator("[data-quick]").count() == 4
    # a model on the board: its own standing, as on its page
    page.locator("[data-ms='submit'] input").fill(MODEL)
    page.locator("[data-ms='submit'] input").press("ArrowDown")
    page.locator(f"[data-ms-item='{MODEL}']").first.click()
    page.wait_for_function("() => [...document.querySelectorAll(\"[data-exam-picker='submit'] "
                           "[data-status='judged']\")].length > 0")
    assert page.errors == []


# ---------------------------------------------------------------------------
# own model code, on the page
# ---------------------------------------------------------------------------

@pytest.fixture
def upload(live):
    art = live["root"] / "artifacts"
    sha = own_code_upload(art, UPLOAD)
    yield sha
    shutil.rmtree(art / UPLOAD, ignore_errors=True)


def pick_model(page, mid):
    box = page.locator("[data-ms='submit'] input")
    box.fill(mid)
    page.locator(f"[data-ms-item='{mid}']").first.wait_for()
    return page.locator(f"[data-ms-item='{mid}']").first


@pytest.mark.dashboard
def test_own_code_the_server_will_not_run_disables_submit_with_the_reason(live, page, upload):
    base = live["base"]
    open_submit(page, base)
    item = pick_model(page, f"local/{UPLOAD}")
    # said in the search, before it is picked
    assert "ships its own model code — this server does not run it" in item.text_content()
    item.click()
    why = page.locator("[data-why='own-code']")
    page.wait_for_function("() => (document.querySelector(\"[data-why='own-code']\") || {})"
                           ".textContent")
    for words in ("ALLOW_REMOTE_CODE=1", "EVAL_USER", "SERVICE.md § custom model code"):
        assert words in why.text_content()
    assert page.get_by_role("button", name="Submit model").is_disabled()
    assert page.locator("[data-own-code-box]").count() == 0
    shot(page, "11i-own-code-refused-1400-light.png")
    assert page.errors == []


@pytest.mark.dashboard
def test_own_code_the_server_runs_needs_the_box_and_sends_it(live, page, upload, monkeypatch):
    from service import config, db
    monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", True)
    monkeypatch.setattr(config, "EVAL_USER", getpass.getuser())
    base = live["base"]
    page.set_viewport_size({"width": 1400, "height": 900})
    open_submit(page, base, "masein")
    item = pick_model(page, f"local/{UPLOAD}")
    assert "ships its own model code" in item.text_content()
    assert "does not run it" not in item.text_content()
    item.click()
    box = page.locator("[data-own-code-box='submit']")
    box.wait_for()
    label = page.locator("[data-own-code='submit']").text_content()
    assert label.startswith("Run this checkpoint's own model code (modeling_custom.py, sha "
                            + upload[:4] + "…)")
    assert label.endswith(f"as the unprivileged {getpass.getuser()} user")
    assert not box.is_checked()
    submit = page.get_by_role("button", name="Submit model")
    assert submit.is_disabled()
    shot(page, "11i-own-code-box-1400-light.png")
    box.check()
    assert submit.is_enabled()
    submit.click()
    page.wait_for_selector("[data-toast='submit']")
    row = next(r for r in db.recent(20) if r["hf_id"] == f"local/{UPLOAD}")
    assert row["allow_remote_code"] == 1
    db.update(row["id"], status="canceled")
    assert page.errors == []


@pytest.mark.dashboard
def test_a_sha_off_the_allowlist_is_named_on_the_page(live, page, upload, monkeypatch):
    from service import config
    monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", True)
    monkeypatch.setattr(config, "EVAL_USER", getpass.getuser())
    monkeypatch.setattr(config, "REMOTE_CODE_SHAS", {"000000000000"})
    open_submit(page, live["base"])
    pick_model(page, f"local/{UPLOAD}").click()
    page.wait_for_function("() => (document.querySelector(\"[data-why='own-code']\") || {})"
                           ".textContent")
    why = page.locator("[data-why='own-code']").text_content()
    assert f"sha {upload}" in why and f"add {upload} to REMOTE_CODE_SHAS" in why
    assert page.get_by_role("button", name="Submit model").is_disabled()
    assert page.errors == []


@pytest.mark.dashboard
def test_resubmit_on_a_row_that_failed_for_its_own_code_offers_the_box(
        live, page, upload, judged_on, monkeypatch):
    from service import config, db
    monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", True)
    monkeypatch.setattr(config, "EVAL_USER", getpass.getuser())
    mid = f"local/{UPLOAD}"
    sid = db.add(mid, "base", "judged", "masein", "", tasks=["exam_law"])
    db.update(sid, status="failed", error=(
        f"artifact '{UPLOAD}' carries an auto_map, so loading it executes the Python shipped "
        f"in the upload. That is off by default. Resubmit with allow_remote_code=true, on a "
        f"server running with ALLOW_REMOTE_CODE=1 and EVAL_USER set — see SERVICE.md "
        f"§ custom model code."))
    try:
        page.set_viewport_size({"width": 1400, "height": 900})
        page.goto(live["base"] + "/#tab=queue")
        set_name(page, "masein")
        again = page.locator(f"[data-row-resubmit='{sid}']")
        again.wait_for()
        assert again.text_content() == "Resubmit…"
        again.click()
        box = page.locator(f"[data-own-code-box='q{sid}']")
        box.wait_for()
        go = page.locator(f"[data-own-code-go='{sid}']")
        assert go.is_disabled() and not box.is_checked()
        # in a row of its own under the failed one: the page never scrolls sideways
        assert page.locator(f"tr[data-own-code-row='{sid}']").count() == 1
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        shot(page, "11i-resubmit-box-1400-light.png")
        box.check()
        assert go.is_enabled()
        go.click()
        page.wait_for_selector("[data-toast='resubmit']")
        row = next(r for r in db.recent(20) if r["hf_id"] == mid and r["id"] != sid)
        assert row["allow_remote_code"] == 1 and json.loads(row["tasks"]) == ["exam_law"]
        db.update(row["id"], status="canceled")
        # a server that does not run it says why, and offers no Resubmit
        monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", False)
        page.evaluate("delete state.codeInfo['local/custom-arch']")
        page.locator(f"[data-row-resubmit='{sid}']").click()
        why = page.locator(f"[data-own-code-why='{sid}']")
        why.wait_for()
        assert "ALLOW_REMOTE_CODE=1" in why.text_content()
        assert page.locator(f"[data-own-code-go='{sid}']").count() == 0
        assert page.errors == []
    finally:
        db.update(sid, status="canceled")


@pytest.mark.dashboard
def test_the_model_page_panel_asks_about_own_code_too(live, page, judged_on, monkeypatch):
    from service import config, db
    art = live["root"] / "artifacts"
    name = LOCAL.split("/", 1)[1]
    own_code_upload(art, name)
    try:
        panel = open_panel(page, live["base"], LOCAL)
        why = panel.locator("[data-why='msit']")
        page.wait_for_function("() => /ALLOW_REMOTE_CODE/.test(document.querySelector("
                               "\"[data-why='msit']\").textContent)")
        assert panel.locator("[data-msit-go]").is_disabled()
        monkeypatch.setattr(config, "ALLOW_REMOTE_CODE", True)
        monkeypatch.setattr(config, "EVAL_USER", getpass.getuser())
        page.locator("[data-msit-close]").click()
        page.locator(f"[data-sit-open='{LOCAL}']").click()        # asked afresh on open
        box = panel.locator("[data-own-code-box='msit']")
        box.wait_for()
        st = statuses(page)
        t = next(t for t, (k, _, off) in st.items() if not off)
        panel.locator(f"input[data-exam-task='{t}']").check()
        assert panel.locator("[data-msit-go]").is_disabled()
        assert "tick the box to run it" in why.text_content()
        box.check()
        assert panel.locator("[data-msit-go]").is_enabled()
        set_name(page, "masein")
        panel.locator("[data-msit-go]").click()
        page.wait_for_selector("[data-toast='msit']")
        row = next(r for r in db.recent(20) if r["hf_id"] == LOCAL)
        assert row["allow_remote_code"] == 1 and json.loads(row["tasks"]) == [t]
        assert page.errors == []
    finally:
        cleanup_rows(LOCAL)
        shutil.rmtree(art / name, ignore_errors=True)


# ---------------------------------------------------------------------------
# the Suite cell
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
def test_a_37_topic_row_is_one_line_and_its_list_opens_by_area(live, page):
    from service import db
    base = live["base"]
    import urllib.request
    built = json.loads(urllib.request.urlopen(base + "/api/loop").read())["tasks_built"]
    exam = [t for t in built if t.startswith("exam_")]
    sid = db.add(MODEL, "base", "judged", "masein", "", tasks=exam + ["fr_control_mmlu"])
    one = db.add(MODEL, "base", "judged", "masein", "", tasks=["exam_law"])
    for s in (sid, one):
        db.update(s, status="failed", error="stopped by hand")
    try:
        page.set_viewport_size({"width": 1400, "height": 900})
        page.goto(base + "/#tab=queue")
        cell = page.locator(f"tr[data-queue-row='{sid}'] [data-suite-cell='q']")
        cell.wait_for()
        assert cell.locator("summary").text_content() == \
            f"judged · {len(exam)} topics + MMLU control ▸"
        # as tall as the row that sat one topic: #56's stood 650px
        row_h = page.locator(f"tr[data-queue-row='{sid}']").bounding_box()["height"]
        one_h = page.locator(f"tr[data-queue-row='{one}']").bounding_box()["height"]
        assert abs(row_h - one_h) <= 1 and row_h < 80, (row_h, one_h)
        assert page.locator(f"tr[data-queue-row='{one}'] [data-suite-cell='q']") \
            .text_content() == "judged · Law"
        cell.locator("summary").click()
        lines = cell.locator(".suitelist > div")
        assert lines.count() >= 7                           # grouped by area
        assert re.match(r"^[A-Z][^:]+: ", lines.first.text_content())
        shot(page, "11i-suite-cell-open-1400-light.png")
        # the model page's runs say it the same way
        page.goto(base + "/#model=" + MODEL.replace("/", "%2F"))
        page.wait_for_selector(f"tr[data-run='{sid}'] [data-suite-cell='m']")
        assert page.locator(f"tr[data-run='{sid}'] [data-suite-cell='m'] summary") \
            .text_content().startswith(f"judged · {len(exam)} topics")
        assert page.errors == []
    finally:
        for s in (sid, one):
            db.update(s, status="canceled")


# ---------------------------------------------------------------------------
# screenshots
# ---------------------------------------------------------------------------

@pytest.mark.dashboard
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, browser, judged_on, theme):
    for width in (1400, 400):
        ctx = browser.new_context(viewport={"width": width, "height": 1000},
                                  reduced_motion="reduce")
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
            page.wait_for_selector(f"[data-sit-open='{MODEL}']")
            page.evaluate(f"applyTheme('{theme}')")
            shot(page, f"11i-model-hero-{width}-{theme}.png")
            page.locator(f"[data-sit-open='{MODEL}']").click()
            panel = page.locator("[data-panel='msit']")
            panel.locator("[data-exam-picker='msit']").wait_for()
            panel.locator("[data-quick='weakest']").click()
            page.wait_for_timeout(200)
            panel.screenshot(path=SCREENS / f"11i-panel-{width}-{theme}.png")
            if width > 500:
                open_submit(page, live["base"])
                choose(page.get_by_label("suite"), "judged")
                page.locator("[data-exam-picker='submit']").wait_for()
                page.wait_for_timeout(200)
                shot(page, f"11i-queue-form-{width}-{theme}.png")
            assert errors == []
        finally:
            ctx.close()
