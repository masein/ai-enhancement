"""Phase 9b: the first screen and the way around — one line of checks, an
Overview that leads with the board, six tabs, one name, one model on the Loop
board, the same next step for everyone, empty topics folded, buttons that
carry their context, the product's words, and freshness you can see."""

from __future__ import annotations

import copy
import json
import re

import pytest

import report_lm_eval as report
from conftest import fresh, go_tab, make_service, set_name
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


def drop_topic(model: str, task: str) -> None:
    from service import config
    jf = config.OUT_DIR / model.replace("/", "__") / "judge.json"
    j = json.loads(jf.read_text(encoding="utf-8"))
    j["tasks"].pop(task, None)
    jf.write_text(json.dumps(j), encoding="utf-8")


def test_every_check_is_a_short_line_a_severity_and_a_place(payload):
    checks = payload["checks"]
    assert len(checks) == len(payload["warnings"]) >= 2
    assert [c["text"] for c in checks] == payload["warnings"]          # the long text, kept
    for c in checks:
        assert c["severity"] in ("warning", "info")
        assert c["short"] and len(c["short"]) < 90 and "{" not in c["short"]
        # 12g.1: the judge's checks point at the Knowledge exam, where the judge is set up
        assert c["show"]["tab"] in ("overview", "exam", "models", "leaderboard", "provenance")
    prelim = next(c for c in checks if c["key"] == "preliminary")
    assert prelim["show"] == {"tab": "models", "prelim": True}
    assert re.fullmatch(r"\d+ of \d+ models are preliminary", prelim["short"])
    assert any(c["judged"] for c in checks)


def test_the_loop_board_is_one_models_and_says_which(svc):
    client, appmod, _ = svc
    drop_topic("fx/chance-160m", "exam_law")                   # it has one topic fewer
    fresh(appmod)
    j = client.get("/api/loop").json()
    counts = {m["id"]: m["topics"] for m in j["models"]}
    assert j["model"] == j["models"][0]["id"]
    assert counts[j["model"]] == max(counts.values())          # most judged topics by default
    for r in j["topics"]:
        if r["last_judged"]:
            assert r["last_judged"]["model"] == j["model"]     # one column, one model
    j = client.get("/api/loop", params={"model": "fx/chance-160m"}).json()
    assert j["model"] == "fx/chance-160m"
    law = next(r for r in j["topics"] if r["topic"] == "Law")
    assert law["last_judged"] is None                          # not sat, by this model
    assert law["next"]["step"] in ("sit", "review", "generate", "hand")
    econ = next(r for r in j["topics"] if r["topic"] == "Economics")
    assert econ["last_judged"]["model"] == "fx/chance-160m"
    # an unknown model falls back to the default rather than an empty board
    assert client.get("/api/loop", params={"model": "org/nobody"}).json()["model"] == \
        client.get("/api/loop").json()["model"]


def test_the_next_step_is_the_same_for_everyone(svc):
    """It used to turn on a per-browser 'read' flag: physics said Read the
    results and the other four Propose, and two people saw two steps."""
    client, _, _ = svc
    steps = {r["next"]["step"] for r in client.get("/api/loop").json()["topics"]}
    assert "read" not in steps
    assert steps <= {"import", "sit", "propose", "review", "generate", "hand", "unreadable"}
    assert "bench-loop-read" not in report.JS and "loopRead" not in report.JS
    from service import app
    assert list(app.STEPS)[:6] == ["import", "sit", "propose", "review", "generate", "hand"]
    assert app.STEPS["hand"] == "Train"


# ---------------------------------------------------------------------------
# the browser half
# ---------------------------------------------------------------------------

def routed(browser, payload, width=1512, height=900):
    ctx = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce")
    return ctx, Live(ctx, payload, fail=False)


@pytest.mark.dashboard
def test_the_first_screen_is_the_board_not_banners(browser, payload):
    """1,512 × 900: 12b.2's Home — what needs you, what is running, and the
    best in each kind of test — whole on the first screen, no banner above."""
    ctx, s = routed(browser, payload)
    try:
        pg = s.open()
        pg.wait_for_selector("[data-best-by-kind] [data-best]")
        best = pg.locator("[data-best-by-kind]").bounding_box()
        assert best["y"] + best["height"] <= 900, best
        assert pg.locator("#warnings li[data-check]").first.is_hidden()   # the checks: one line
        assert pg.locator("[data-how-to-read]").count() == 0          # 12b: in Help now
        assert s.errors == []
    finally:
        ctx.close()


@pytest.mark.dashboard
def test_the_overview_skips_the_duplicate_and_links_the_preliminary(browser, payload):
    """The hero featured …step945 — the Leaderboard's "duplicate of …-v2" —
    with a sentence calling it preliminary; Top models listed both. 12b.2:
    Home's Standard card is the one place it could, and it names the twin."""
    p = copy.deepcopy(payload)
    ranked = sorted((m for m in p["models"] if m.get("avg") is not None),
                    key=lambda m: -m["avg"])
    top, twin = ranked[0], ranked[1]
    top.update(duplicateOf=twin["id"], duplicateOfName=twin["name"],
               duplicateWhy="every score and item count is identical")
    ctx, s = routed(browser, p)
    try:
        pg = s.open()
        pg.wait_for_selector("[data-best='standard']")
        # the best card names the twin, never the duplicate
        best = pg.locator("[data-best-name='standard']").text_content()
        assert best == twin["name"] and best != top["name"]
        assert "preliminary and carries no overall rank" not in pg.locator("#view").text_content()
        assert s.errors == []
    finally:
        ctx.close()


@pytest.mark.dashboard
def test_home_has_a_way_to_submit(live, page):
    """12b: Test a model, in the header, is the way to submit — a dialog over
    Home. (12b.2: the Overview's loop card is gone; each model's weakest
    topic is first on Improve ▸ By topic, and on Models ▸ Insights.)"""
    base = live["base"]
    page.goto(base + "/")
    page.locator("header [data-test-model]").click()
    page.wait_for_function("document.activeElement === "
                           "document.querySelector('[data-dialog=test] [data-ms=submit] input')")
    assert page.evaluate("location.hash") in ("", "#tab=home")
    assert page.errors == []


@pytest.mark.dashboard
def test_one_name_in_the_header_and_no_box_anywhere_else(live, page):
    base = live["base"]
    page.goto(base + "/#tab=loop")
    page.evaluate("localStorage.removeItem('bench-name'); state.rvName = ''")
    page.reload()
    page.wait_for_selector("[data-who-prompt]")                  # a first visit is asked, up top
    assert "Who are you?" in page.locator("#who").text_content()
    for hash_ in ("#tab=loop", "#topic=law", "#tab=queue", "#tab=exam", "#tab=review"):
        page.goto(base + "/" + hash_)
        page.wait_for_selector("#view > *")
        page.wait_for_timeout(400)
        assert page.locator("#view input[aria-label='your name']").count() == 0, hash_
    # an action that records a name, with none: the header asks — no refusal three panels away
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
    set_name(page, "Omar")
    assert page.locator("#who button[data-who='Omar']").inner_text().strip() == "Omar ▾"
    page.reload()
    page.wait_for_selector("#who button[data-who='Omar']")      # remembered
    assert page.errors == []


@pytest.mark.dashboard
def test_the_page_speaks_the_products_words(live, page):
    base = live["base"]
    for hash_ in ("#tab=exam", "#tab=loop", "#topic=law"):
        page.goto(base + "/" + hash_)
        page.wait_for_selector("#view > *")
        page.wait_for_timeout(800)
        text = page.locator("#view").text_content()
        assert "exam_build.py" not in text and "--root" not in text, hash_
        assert "/app/" not in text and "(fallback)" not in text, hash_
    page.goto(base + "/#tab=exam")
    page.wait_for_selector("[data-panel='rubrics'] tr[data-rubric-row]")
    cells = page.locator("[data-panel='rubrics'] tbody td").all_text_contents()
    assert not any(re.search(r"\b[0-9a-f]{10}\b", c) for c in cells)   # shas: in tooltips only
    assert page.locator("[data-panel='rubrics'] [data-info]").count() == 1
    # the header: no transformers chip; Provenance carries the builds
    assert "transformers" not in page.locator("#metaChips").text_content()
    go_tab(page, "Provenance")
    page.wait_for_selector("[data-builds]")
    assert page.errors == []


@pytest.mark.dashboard
def test_freshness_turns_amber_when_the_polls_stop(browser, payload):
    ctx, s = routed(browser, payload)
    try:
        pg = s.open()
        stamp = pg.locator("[data-stamp]")
        pg.wait_for_selector("[data-stamp][data-fresh='ok'] .dot.ok")
        # 11b: the chip is the bar's LIVE badge — "● LIVE · 12:33"; 11h: the
        # time is the last check that worked, and the data's own time is in
        # its title
        assert re.fullmatch(r"LIVE · \d\d:\d\d", stamp.text_content())
        assert "data last changed" in stamp.get_attribute("title")
        s.fail = True
        pg.wait_for_selector("[data-stamp][data-fresh='stale'] .dot.warn", timeout=30000)
        assert re.fullmatch(r"STALE · \d\d:\d\d", stamp.text_content())
        s.fail = False
        pg.wait_for_selector("[data-stamp][data-fresh='ok']", timeout=40000)
    finally:
        ctx.close()
