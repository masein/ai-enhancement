"""12g.1 on the page: Improve is one pipeline for one model — Weak spots,
Proposals, Training data, Retests — with the Standard benchmarks only as the
watch line under a retest. By topic and Review are gone; their addresses land
here with the model kept. A checkpoint's page says what it was trained from.
The checks are a popover now, and Home's Knowledge exam card names the model
with the most judged topics."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12g1"
MODEL = "fx/good-750m"
CK = "fx/good-750m-tuned-skill"
UPLOADED = "local/nodiag-step400"
PLANTED = []


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def plant(status, topic, task, model=MODEL, **kw):
    from service import db
    pid = db.proposal_create(model, task, topic, "masein",
                             {"n_shown": 3, "diagnose_items": 44, "diagnose_weak": 36})
    db.proposal_update(pid, status=status, spec_text="the missing skill", **kw)
    PLANTED.append(pid)
    time.sleep(0.01)                          # created_at orders them
    return pid


def clear():
    from service import db
    c = db._conn()                            # noqa: SLF001
    try:
        for pid in PLANTED:
            c.execute("DELETE FROM datasets WHERE proposal_id = ?", (pid,))
            c.execute("DELETE FROM proposals WHERE id = ?", (pid,))
        c.commit()
    finally:
        c.close()
    PLANTED.clear()


@pytest.fixture
def planted(live):
    """Four proposals for good-750m — waiting, approved, being written,
    rejected — and a dataset made from the approved one."""
    from service import db
    clear()
    a = plant("proposed", "Economics", "exam_economics")
    b = plant("approved", "Law", "exam_law")
    d = plant("pending", "Sociology", "exam_sociology")
    r = plant("rejected", "Physics & Astronomy", "exam_physics_astronomy",
              reject_reason="names a fact, not a skill")
    did = db.dataset_create(b, "doc", 20, "masein", {})
    db.dataset_update(did, status="ready", provenance=json.dumps(
        {"items": {"kept": 18, "requested": 20,
                   "missing": [{"why": "too short"}, {"why": "not JSON"}]}}))
    yield {"proposed": a, "approved": b, "pending": d, "rejected": r, "dataset": did}
    clear()


def served(page, edit):
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def pipeline(page, base, model=MODEL, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=improve&sub=model&model=" + quote(model, safe=""))
    page.wait_for_selector(f"[data-pipeline='{model}']")
    page.wait_for_function("() => state.rv.loaded")


def stage_items(page, key, attr):
    return [e.get_attribute(attr) for e in page.locator(f"[data-stage='{key}'] [{attr}]").all()]


def weak_expected(page, model, open_tasks):
    """the model's judged topics, weakest first, without the open ones — from the payload"""
    got = page.evaluate("""m => { const x = DATA.models.find(y => y.id === m);
      return DATA.judged.exam.map(t => [t, ((x.judge || {}).tasks || {})[t]]); }""", model)
    rows = []
    for t, v in got:
        if not v or t in open_tasks:
            continue
        s = v.get("score_report")                   # the board's rule (pubScore)
        if s is None and v.get("n_report") is None:
            s = v.get("mean")
        if s is not None:
            rows.append((s, t))
    return [t for _, t in sorted(rows, key=lambda r: r[0])]


# ---------------------------------------------------------------------------
# 1. one model, four stages
# ---------------------------------------------------------------------------

def test_the_pipeline_is_four_stages_for_one_model(live, page, planted):
    pipeline(page, live["base"])
    assert [s.get_attribute("data-stage") for s in page.locator("[data-stage]").all()] == \
        ["weak", "proposals", "data", "retests"]
    assert page.locator("[data-imp-model]").inner_text() == "good-750m ▾"
    # Weak spots: judged topics, weakest first, none with an open proposal
    want = weak_expected(page, MODEL, {"exam_economics", "exam_law", "exam_sociology"})
    # 12g.2: its Everyday groups sit beside them, each labelled (12a.5: eight)
    n = len(want) + 8
    assert page.locator("[data-stage='weak']").get_attribute("data-stage-n") == str(n)
    assert len(stage_items(page, "weak", "data-weak")) == 5            # five, then + n more
    more = page.locator("[data-stage-more='weak']")
    assert more.inner_text() == f"+ {n - 5} more"
    more.click()
    got = stage_items(page, "weak", "data-weak")
    assert [t for t in got if t.startswith("exam_")] == want          # weakest first, within each
    assert len([t for t in got if t.startswith("everyday:")]) == 8
    # Proposals: waiting, approved, being written — newest first, one action each
    p = planted
    assert stage_items(page, "proposals", "data-prop") == \
        [str(p["pending"]), str(p["approved"]), str(p["proposed"])]
    acts = {e.get_attribute("data-prop-act"): e.inner_text()
            for e in page.locator("[data-prop-act]").all()}
    assert acts == {str(p["proposed"]): "Review", str(p["approved"]): "Generate",
                    str(p["pending"]): "Read"}
    assert page.locator(f"[data-rv-status='{p['proposed']}']").inner_text() == "waiting for you"
    # Training data: the dataset, and the flag for a training run
    item = page.locator(f"[data-ds-item='{p['dataset']}']")
    assert item.locator(".si-main").inner_text().startswith("Law · 18 of 20 · 2 missing")
    assert item.locator("[data-ds-flag]").count() == 1
    # Retests: nothing trained from it yet — one line, no box
    assert page.locator("[data-stage='retests'] .stagelist").count() == 0
    assert page.locator("[data-stage-none='retests']").inner_text().startswith("No retests yet")
    # what is no longer open is folded at the end
    assert page.locator("[data-imp-past]").get_attribute("data-imp-past") == "1"
    # Review opens the proposal's card
    page.locator(f"[data-prop-act='{p['proposed']}']").click()
    page.wait_for_selector("#reader[data-ready='1']")
    assert page.locator("#reader").get_attribute("data-key") == f"proposal:{p['proposed']}"
    page.keyboard.press("Escape")
    shot(page, "12g1-pipeline-1400-light.png", full_page=True)
    assert page.errors == []


def test_a_stage_with_nothing_says_one_line_and_draws_no_box(live, page, planted):
    pipeline(page, live["base"], "fx/chance-160m")
    for key, words in (("proposals", "No proposals waiting"), ("data", "No training data yet")):
        assert page.locator(f"[data-stage='{key}'] .stagelist").count() == 0, key
        assert page.locator(f"[data-stage-none='{key}']").inner_text() == words
    assert page.errors == []


def test_propose_opens_the_dialog_on_the_weakest_topic(live, page):
    clear()                                          # Economics has no open proposal
    pipeline(page, live["base"])
    page.locator("[data-stage-more='weak']").click()
    # a topic the dialog would refuse says why, and offers no Propose
    for w in page.locator("[data-weak-why]").all():
        task = w.get_attribute("data-weak-why")
        assert w.inner_text()
        assert page.locator(f"[data-weak-propose='{task}']").count() == 0
    # the button opens on the weakest topic a proposal can be made from
    first = page.locator("[data-weak-propose]").first.get_attribute("data-weak-propose")
    assert first == "exam_economics"                 # the fixture's one topic over the floor
    name = page.evaluate(f"frName({json.dumps(first)})")
    page.locator("[data-imp-propose]").click()
    dlg = page.locator("[data-dialog='propose']")
    dlg.wait_for()
    dlg.locator(f"[data-np-topic={json.dumps(name)}]").wait_for()
    assert dlg.locator(f"[data-np-topic={json.dumps(name)}] input").is_checked()
    page.keyboard.press("Escape")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2. old addresses land here, with the model kept
# ---------------------------------------------------------------------------

def test_old_improve_and_review_addresses_land_on_the_pipeline(live, page):
    pipeline(page, live["base"])                     # the viewer's last model: good-750m
    want = "#tab=improve&sub=model&model=" + quote(MODEL, safe="")
    assert page.evaluate("location.hash") == want
    for old in ("#tab=improve&sub=topics", "#tab=improve&sub=review&view=datasets",
                "#tab=improve&sub=review", "#tab=loop", "#tab=review", "#tab=improve", "#improve"):
        page.goto(live["base"] + "/" + old)
        page.wait_for_selector(f"[data-pipeline='{MODEL}']")
        assert page.evaluate("location.hash") == want, old
    assert [b.get_attribute("data-sub") for b in
            page.locator("[data-subswitch='improve'] [data-sub]").all()] == ["model", "training"]
    assert page.errors == []


def test_the_first_visit_opens_on_the_model_with_the_most_judged_topics(live, browser):
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, reduced_motion="reduce")
    try:
        pg = ctx.new_page()
        pg.goto(live["base"] + "/#tab=improve")
        pg.wait_for_selector("[data-pipeline]")
        most = pg.evaluate("""() => DATA.models.filter(m => !m.duplicateOf).map(m => [m.id,
          DATA.judged.exam.filter(t => { const v = ((m.judge || {}).tasks || {})[t];
            return v && (v.score_report != null || v.mean != null); }).length, m.name])""")
        top = max(n for _, n, _ in most)
        want = sorted((name, mid) for mid, n, name in most if n == top)[0][1]
        assert pg.locator("[data-pipeline]").get_attribute("data-pipeline") == want
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# 3. Retests and the Standard watch
# ---------------------------------------------------------------------------

def watch_numbers(page, base_id, ck_id):
    """"Standard (N) a → b": the average of the benchmarks both have, as Avg of N"""
    got = page.evaluate("""([a, b]) => lbBenchAll().filter(t => (cell(t, a) || {}).v != null
        && (cell(t, b) || {}).v != null).map(t => [t, cell(t, a), cell(t, b), DATA.tasks[t].chance])""",
                        [base_id, ck_id])

    def avg(i):
        vs, v2 = [], 0.0
        for row in got:
            c, chance = row[i], row[3]
            k = 1 / (1 - chance) if chance is not None and 0 < chance < 1 else 1.0
            vs.append(max(0.0, (c["v"] - chance) * k) if k != 1.0 else c["v"])
            v2 += (c["se"] * k) ** 2
        return sum(vs) / len(vs), math.sqrt(v2) / len(vs)
    return len(got), avg(1), avg(2)


def with_ck_cells(change, sig_ok):
    """the checkpoint's Standard cells, edited in what the page is served"""
    def edit(body):
        for t, cells in body["cells"].items():
            if MODEL in cells and CK in cells:
                cells[CK] = change(t, dict(cells[MODEL]))
                if cells[CK] is None:
                    del cells[CK]
        for t, rows in body["sig"].items():
            for r in rows:
                if {r[0], r[1]} == {MODEL, CK}:
                    r[4] = sig_ok(t)
    return edit


@pytest.fixture
def trained(live):
    api(live["base"], "/api/trained-from", {"model": CK, "base": MODEL, "by": "masein"})
    yield


@pytest.mark.parametrize("case", ["held", "noise", "dropped", "untested"])
def test_a_retest_carries_the_standard_watch_and_dropped_only_when_real(live, page, trained, case):
    def change(t, c):
        if case == "untested":
            return None
        if t == "hellaswag" and case == "noise":
            c["v"] -= 0.01
        if t == "hellaswag" and case == "dropped":
            c["v"] -= 0.2
        return c
    served(page, with_ck_cells(change, lambda t: case == "dropped" and t == "hellaswag"))
    pipeline(page, live["base"])
    item = page.locator(f"[data-retest='{CK}']")
    item.wait_for()
    line = item.locator(f"[data-watch-line='{CK}']")
    if case == "untested":
        assert line.inner_text() == "Standard: not tested · Test"
        assert line.get_attribute("data-watch") == "untested"
        return
    n, (a, _), (b, _) = watch_numbers(page, MODEL, CK)
    head = f"Standard ({n}) {100 * a:.1f} → {100 * b:.1f} · "
    if case == "dropped":
        hs = page.evaluate(f"[cell('hellaswag', {json.dumps(MODEL)}).v, "
                           f"cell('hellaswag', {json.dumps(CK)}).v]")
        assert line.inner_text() == head + (f"dropped · hellaswag {100 * hs[0]:.1f} → "
                                            f"{100 * hs[1]:.1f}")
        assert "dropped" in line.get_attribute("class")
        shot(page, "12g1-retest-dropped-1400-light.png")
        line.locator("[data-watch-drop]").click()
        page.wait_for_selector("[data-model-hero]")
        page.wait_for_selector("[data-kind-block='standard']")
        assert page.evaluate("state.model") == CK
    else:
        assert line.inner_text() == head + "no drop"          # inside the noise is no drop
        assert "dropped" not in line.get_attribute("class")
    assert page.errors == []


def test_trained_from_is_set_on_a_checkpoints_page_and_brings_it_to_improve(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + quote(UPLOADED, safe=""))
    page.wait_for_selector("[data-model-hero]")
    line = page.locator("[data-trained-from]")
    assert line.get_attribute("data-trained-from") == "unset"
    assert line.inner_text().startswith("Set what this was trained from to see it in Improve")
    set_name(page, "masein")
    page.locator(f"[data-trained-from-set='{UPLOADED}']").click()
    page.locator("#pop-trained-from input").fill("skewed")
    page.locator("[data-trained-from-pick='fx/skewed-360m']").click()
    page.wait_for_selector("[data-trained-from='fx/skewed-360m']")
    text = page.locator("[data-trained-from]").inner_text()
    assert text.startswith("Trained from skewed-360m · set by masein · see it in Improve")
    shot(page, "12g1-trained-from-1400-light.png")
    page.locator("[data-trained-from-improve]").click()
    page.wait_for_selector("[data-pipeline='fx/skewed-360m']")
    page.locator(f"[data-retest='{UPLOADED}']").wait_for()
    assert page.errors == []


# ---------------------------------------------------------------------------
# 4. the two fixes from the 12b.3 live check
# ---------------------------------------------------------------------------

def test_the_checks_popover_closes_on_escape_a_click_outside_and_a_page_change(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    dot = page.locator("#warnings [data-warn-summary]")
    dot.wait_for()

    def open_it():
        dot.click()
        page.locator("#pop-checks").wait_for()
    open_it()
    shot(page, "12g1-checks-1400-light.png")
    page.keyboard.press("Escape")
    page.wait_for_selector("#pop-checks", state="detached")
    open_it()
    page.mouse.click(700, 600)                                    # outside it
    page.wait_for_selector("#pop-checks", state="detached")
    open_it()
    page.locator("button[data-tab='models']").first.click()
    page.wait_for_selector("[data-lb-card]")
    page.wait_for_selector("#pop-checks", state="detached")
    assert page.errors == []


def test_homes_exam_card_names_the_most_judged_models_weakest_topic(live, page):
    served(page, lambda b: [m.update(judgedAvg=None) for m in b["models"]])
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    card = page.locator("[data-best-weakest]")
    card.wait_for()
    got = page.evaluate("""() => DATA.models.filter(m => !m.duplicateOf).map(m => [m.id, m.name,
      DATA.judged.exam.map(t => [t, ((m.judge || {}).tasks || {})[t]]).filter(([, v]) => v
        && v.score_report != null).map(([t, v]) => [DATA.judged.topics[t], v.score_report])])""")
    top = max(len(ts) for _, _, ts in got)
    mid, name, ts = sorted((x for x in got if len(x[2]) == top), key=lambda x: x[1])[0]
    topic, v = min(ts, key=lambda x: x[1])
    score = f"{v:.2f}".rstrip("0").rstrip(".")
    assert card.locator("[data-best-name='exam']").inner_text() == \
        f"{name} · weakest: {topic} {score} / 4"
    card.get_by_text("Improve it →").click()
    page.wait_for_selector(f"[data-pipeline='{mid}']")
    assert page.errors == []


@pytest.mark.parametrize("width", [400, 1400])
def test_the_screens(live, page, planted, width):
    pipeline(page, live["base"], width=width)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12g1-pipeline-{width}-light.png", full_page=True)
    assert page.errors == []
