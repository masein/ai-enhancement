"""11g: the Reader, in the browser.

Every file the page names opens in the page: from its link, and from a
pasted address; Esc closes it and gives the focus back; a poll never closes
it; Back closes it. And while any of them opens, no response the page
receives carries a hidden (report-half) question's text or qid.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

import exam_build as eb
from conftest import label_domains, report_half_text

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11g"
MODEL = "fx/good-750m"
TOPIC, SLUG = "Economics", "economics"
E2E_MS = 30000
SHORT = "A short note. " * 20


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def shot(page, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, **kw)


def wait_for(fn, page, what, n=200):
    for _ in range(n):
        v = fn()
        if v:
            return v
        page.wait_for_timeout(150)
    raise AssertionError(f"never: {what}")


@pytest.fixture(scope="module")
def dataset(live):
    """One dataset on the live board: twelve documents asked for, spread by
    area, ten kept — one request's two came back too short."""
    from service import llm
    base, root = live["base"], live["root"]
    label_domains(root / "exam", TOPIC)
    was = llm.FakeBatches.responder

    def responder(req):
        if req.custom_id.startswith("gen:") and req.custom_id.rsplit(":", 1)[1] == "1":
            return json.dumps([{"title": "Too brief", "text": SHORT}] * 2)
        return llm.default_responder(req)
    llm.FakeBatches.responder = staticmethod(responder)
    try:
        pid = api(base, "/api/proposals", {"model": MODEL, "topic": TOPIC, "requested_by": "Omar"})["id"]
        import time
        for _ in range(200):
            if api(base, f"/api/proposals/{pid}")["status"] == "proposed":
                break
            time.sleep(0.15)
        api(base, f"/api/proposals/{pid}/approve", {"approver": "Omar"})
        did = api(base, f"/api/proposals/{pid}/generate", {"requester": "Omar", "count": 12})["dataset_id"]
        for _ in range(300):
            if api(base, f"/api/datasets/{did}")["status"] == "ready":
                break
            time.sleep(0.15)
        else:
            raise AssertionError("the dataset never became ready")
        return {"id": did, "pid": pid}
    finally:
        llm.FakeBatches.responder = was


@pytest.fixture(scope="module")
def run_log(live):
    from service import config, db
    sid = db.add(MODEL, "auto", "quick", "omar", "")
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LOGS_DIR / f"service_{sid}_{MODEL.replace('/', '__')}.log"
    path.write_text("\n".join([f"[service] step {i} ok" for i in range(260)]
                              + ["ERROR: CUDA out of memory", "Traceback (most recent call last):"]))
    return {"id": sid, "path": path}


def ready(page, kind):
    page.wait_for_selector(f"#reader[data-kind='{kind}'][data-ready='1']", timeout=E2E_MS)
    return page.locator("#reader")


# the opener of each kind, and what it opens
OPENERS = {
    "dataset": lambda page, base, ctx: (
        page.goto(f"{base}/#topic={SLUG}"),
        page.locator(f"[data-ds-read='{ctx['dataset']['id']}']").click(),
        f"[data-ds-read='{ctx['dataset']['id']}']"),
    "rubric": lambda page, base, ctx: (
        page.goto(f"{base}/#topic={SLUG}"),
        page.locator(f"[data-topic-page] [data-read-rubric='{SLUG}']").click(),
        f"[data-topic-page] [data-read-rubric='{SLUG}']"),
    "criteria": lambda page, base, ctx: (
        page.goto(f"{base}/#topic={SLUG}"),
        page.locator(f"[data-topic-page] [data-read-criteria='{SLUG}']").click(),
        f"[data-topic-page] [data-read-criteria='{SLUG}']"),
    "bank": lambda page, base, ctx: (
        page.goto(f"{base}/#topic={SLUG}"),
        page.locator(f"[data-read-bank='{SLUG}']").click(),
        f"[data-read-bank='{SLUG}']"),
    "log": lambda page, base, ctx: (
        page.goto(f"{base}/#tab=queue"),
        page.locator(f"[data-row-menu='q{ctx['log']['id']}']").click(),
        page.locator(f"[data-pop='q{ctx['log']['id']}'] [data-act='log']").click(),
        f"[data-row-menu='q{ctx['log']['id']}']"),
    "provenance": lambda page, base, ctx: (
        page.goto(f"{base}/#model={MODEL.replace('/', '%2F')}"),
        page.locator(f"[data-how-graded='{MODEL}'] a").click(),
        f"[data-how-graded='{MODEL}'] a"),
}
ADDRESS = {"dataset": lambda c: f"#topic={SLUG}&read=dataset:{c['dataset']['id']}",
           "rubric": lambda c: f"#topic={SLUG}&read=rubric:{SLUG}",
           "criteria": lambda c: f"#tab=exam&read=criteria:{SLUG}",
           "bank": lambda c: f"#topic={SLUG}&read=bank:{SLUG}",
           "log": lambda c: f"#tab=queue&read=log:{c['log']['id']}",
           "provenance": lambda c: f"#tab=review&read=provenance:dataset:{c['dataset']['id']}"}


@pytest.mark.parametrize("kind", list(OPENERS))
def test_every_reader_opens_from_its_link_closes_on_esc_and_survives_a_poll(
        live, page, dataset, run_log, kind):
    ctx = {"dataset": dataset, "log": run_log}
    base = live["base"]
    page.set_viewport_size({"width": 1400, "height": 900})
    *_, opener = OPENERS[kind](page, base, ctx)
    box = ready(page, kind)
    assert "read=" + kind in page.evaluate("location.hash")
    # it is a dialog, and the focus is inside it
    assert page.evaluate("!!document.activeElement.closest('#reader')")
    # a poll: the same sheet, the same body — nothing rebuilt under the reader
    page.evaluate("document.querySelector('#reader .rd-body')._mark = 'kept'")
    page.evaluate("render()")
    page.wait_for_timeout(5500)                              # and a real one
    assert page.evaluate("document.querySelector('#reader .rd-body')._mark") == "kept"
    assert box.get_attribute("data-kind") == kind
    # Esc closes it and gives the focus back to what opened it
    page.keyboard.press("Escape")
    page.wait_for_selector("#reader", state="detached")
    page.wait_for_function("sel => document.activeElement === document.querySelector(sel)", arg=opener)
    assert "read=" not in page.evaluate("location.hash")
    # a pasted address opens it, and Back closes it
    page.goto(base + "/" + ADDRESS[kind](ctx))
    ready(page, kind)
    # ✕ Close closes it too
    page.locator("[data-reader-close]").click()
    page.wait_for_selector("#reader", state="detached")
    assert page.errors == []


@pytest.mark.parametrize("kind", ["dataset", "rubric", "bank", "log"])
def test_back_closes_a_reader_opened_from_a_link(live, page, dataset, run_log, kind):
    ctx = {"dataset": dataset, "log": run_log}
    OPENERS[kind](page, live["base"], ctx)
    ready(page, kind)
    page.go_back()
    page.wait_for_selector("#reader", state="detached")
    assert "read=" not in page.evaluate("location.hash")
    # and Forward opens it again
    page.go_forward()
    ready(page, kind)
    assert page.errors == []


def test_the_dataset_reader_lists_every_document_and_places_the_missing(live, page, dataset):
    base = live["base"]
    did = dataset["id"]
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(f"{base}/#topic={SLUG}&read=dataset:{did}")
    ready(page, "dataset")
    kept = api(base, f"/api/datasets/{did}")["provenance"]["items"]["kept"]
    docs = page.locator("#reader [data-doc]")
    assert docs.count() == kept == 10
    gone = page.locator("#reader [data-missing-doc]")
    assert gone.count() == 2
    assert all("too short (60 words)" in t for t in gone.all_text_contents())
    assert all("Focus: " in t for t in docs.all_text_contents())
    assert page.locator("[data-dataset-counts]").text_content().startswith(
        "requested 12 · kept 10 · missing 2")
    # ← and → move between documents, and the address follows
    assert page.locator("[data-doc-pos]").text_content() == "1 of 10"
    page.locator("[data-doc-next]").click()
    assert page.locator("[data-doc-pos]").text_content() == "2 of 10"
    assert page.evaluate("location.hash").endswith(f"read=dataset:{did}:2")
    page.locator("#reader .reader").press("ArrowRight")
    assert page.locator("[data-doc-pos]").text_content() == "3 of 10"
    page.locator("#reader .reader").press("ArrowLeft")
    assert page.locator("[data-doc-pos]").text_content() == "2 of 10"
    shot(page, "11g-dataset-1400-light.png")
    # a pasted address opens the same document
    page.goto(f"{base}/#topic={SLUG}&read=dataset:{did}:5")
    ready(page, "dataset")
    assert page.locator("[data-doc-pos]").text_content() == "5 of 10"
    # search narrows the list and marks what it found
    titles = page.locator("#reader [data-doc] .rd-t").all_text_contents()
    word = next(t for t in titles if titles.count(t) == 1)
    page.locator("#reader .rd-search").fill(word)
    page.wait_for_function("n => document.querySelectorAll('#reader [data-doc]').length < n", arg=kept)
    assert page.locator("#reader [data-missing-doc]").count() == 0
    assert page.locator("#reader .rd-list mark").count() >= 1
    assert page.errors == []


def test_a_twenty_document_dataset_opens_in_under_300ms(live, page, dataset):
    base = live["base"]
    page.goto(f"{base}/#topic={SLUG}")
    page.wait_for_selector(f"[data-ds-read='{dataset['id']}']")
    page.evaluate("state.readData = {}")
    ms = page.evaluate("""async id => { const t0 = performance.now();
      openReader({ kind: 'dataset', id: String(id) });
      await new Promise(r => { const tick = () => document.querySelector(
        '#reader[data-ready="1"] .rd-doc h3') ? r() : requestAnimationFrame(tick); tick(); });
      return performance.now() - t0; }""", dataset["id"])
    assert ms < 300, ms
    assert page.errors == []


def test_the_criteria_reader_is_the_judges_own_reading(live, page):
    import judge
    base = live["base"]
    page.goto(f"{base}/#tab=exam&read=criteria:arts")
    ready(page, "criteria")
    want = judge.normalise_criteria(json.loads(judge.rubric_path("arts", ".criteria.json").read_text()))
    rows = page.locator("#reader [data-criterion-row]")
    assert rows.count() == len(want["criteria"]) == 20
    assert [r.get_attribute("data-criterion-row") for r in rows.all()] == [c["id"] for c in want["criteria"]]
    assert page.locator("#reader [data-flag]").count() == len(want["flags"]) >= 1
    flag = page.locator("#reader [data-flag]").first
    assert "What it does: " in flag.text_content() and "Examples" in flag.text_content()
    assert page.locator("#reader [data-criteria-top]").count() == 1
    page.locator("#reader .rd-raw summary").click()
    assert json.loads(page.locator("[data-criteria-raw]").text_content())["criteria"][0]["id"] \
        == json.loads(judge.rubric_path("arts", ".criteria.json").read_text())["criteria"][0]["id"]
    assert page.errors == []


def test_a_rubric_with_html_in_it_renders_as_text_and_nothing_runs(live, page):
    root = Path(live["root"]) / "rubrics"
    root.mkdir(parents=True, exist_ok=True)
    (root / "zz_evil.md").write_text(
        "# Evil rubric\n\nA <script>window.__pwned = 1</script> line.\n\n"
        "<img src=x onerror=\"window.__pwned = 2\">\n\n- a [link](javascript:alert(1)) and "
        "a [good one](https://example.org)\n\n| a | b |\n|---|---|\n| <b>1</b> | 2 |\n",
        encoding="utf-8")
    page.goto(live["base"] + "/#tab=exam&read=rubric:zz_evil")
    ready(page, "rubric")
    md = page.locator("#reader [data-md]")
    assert "<script>window.__pwned = 1</script>" in md.text_content()
    assert "<img src=x" in md.text_content()
    assert md.locator("script, img, b").count() == 0
    assert page.evaluate("window.__pwned") is None
    # a javascript: link is only its words; a web link opens safely
    assert md.locator("a[href^='javascript']").count() == 0
    good = md.locator("a[href='https://example.org']")
    assert good.get_attribute("rel") == "noopener noreferrer" and good.get_attribute("target") == "_blank"
    assert page.errors == []


def test_no_reader_lets_a_hidden_question_reach_the_page(live, page, dataset, run_log):
    """Every response the page receives while each reader opens — not only
    the reader's own — carries no hidden question's text and no hidden qid."""
    rows = [r for rs in eb.load_bank(Path(live["root"]) / "exam").values() for r in rs]
    hidden = report_half_text(rows)
    qids = [r["qid"] for r in rows if eb.half_of(r["qid"]) == "report"]
    assert hidden and qids
    bodies = []

    def keep(resp):
        try:
            if resp.request.resource_type in ("fetch", "xhr", "document"):
                bodies.append((resp.url, resp.text()))
        except Exception:
            pass
    page.on("response", keep)
    ctx = {"dataset": dataset, "log": run_log}
    base = live["base"]
    for kind in OPENERS:
        OPENERS[kind](page, base, ctx)
        ready(page, kind)
        page.wait_for_timeout(300)
    for addr in (f"#topic={SLUG}&read=bank:{SLUG}", f"#tab=exam&read=criteria:{SLUG}",
                 f"#topic={SLUG}&read=provenance:dataset:{dataset['id']}"):
        page.goto(base + "/" + addr)
        page.wait_for_selector("#reader[data-ready='1']")
        page.wait_for_timeout(300)
    assert len(bodies) >= 10
    for url, body in bodies:
        leaked = [t[:60] for _, _, t in hidden if t in body]
        assert not leaked, (url, leaked[:3])
        assert not [q for q in qids if q in body], url
    # and the bank reader said how many there are, and showed none
    page.goto(f"{base}/#topic={SLUG}&read=bank:{SLUG}")
    ready(page, "bank")
    n = sum(1 for r in rows if r["topic"] == TOPIC and eb.half_of(r["qid"]) == "report")
    assert page.locator("[data-bank-hidden]").get_attribute("data-bank-hidden") == str(n)
    assert f"{n} hidden questions — never shown, by design" in page.locator("#reader").text_content()
    assert page.errors == []


def test_the_log_reader_follows_a_growing_log_and_stops_when_you_scroll_up(live, page):
    from service import config, db
    sid = db.add(MODEL, "auto", "quick", "omar", "")
    db.update(sid, status="running", progress="1/4 · arc_easy")
    path = config.LOGS_DIR / f"service_{sid}_{MODEL.replace('/', '__')}.log"
    path.write_text("\n".join(f"[service] step {i}" for i in range(300)))
    page.set_viewport_size({"width": 1400, "height": 800})
    page.goto(f"{live['base']}/#tab=queue&read=log:{sid}")
    ready(page, "log")
    box = page.locator("#reader [data-log]")
    at_end = "b => b.scrollTop + b.clientHeight >= b.scrollHeight - 4"
    page.wait_for_function(f"() => ({at_end})(document.querySelector('#reader [data-log]'))")
    assert "following" in page.locator("[data-log-follow]").text_content()
    # the log grows: the new lines arrive, and the box stays at the end
    with path.open("a") as f:
        f.write("\n" + "\n".join(f"[service] later {i}" for i in range(20)))
    page.wait_for_selector("#reader [data-log] .rd-ln >> text=[service] later 19", timeout=10000)
    page.wait_for_function(f"() => ({at_end})(document.querySelector('#reader [data-log]'))")
    # the person scrolls up to read: it stops following
    box.evaluate("b => { b.scrollTop = 0; b.dispatchEvent(new Event('scroll')); }")
    page.wait_for_function("document.querySelector('[data-log-follow]').textContent.startsWith('paused')")
    with path.open("a") as f:
        f.write("\n" + "\n".join(f"[service] even later {i}" for i in range(20)))
    page.wait_for_selector("#reader [data-log] .rd-ln >> text=[service] even later 19", timeout=10000)
    assert box.evaluate("b => b.scrollTop") == 0
    # the lines that failed are in the warning tone
    db.update(sid, status="failed")
    assert page.errors == []


def test_the_log_reader_numbers_searches_and_marks_bad_lines(live, page, run_log):
    page.goto(f"{live['base']}/#tab=queue&read=log:{run_log['id']}")
    ready(page, "log")
    rows = page.locator("#reader [data-log] .rd-ln")
    assert rows.count() == 200
    assert rows.first.get_attribute("data-ln") == "63"                # the last 200 of 262
    assert page.locator("#reader .rd-ln.bad").count() == 2
    page.locator("#reader .rd-search").fill("step 250")
    page.locator("[data-log-next]").click()
    assert page.locator("#reader .rd-ln.cur").get_attribute("data-ln") == "251"
    page.locator("[data-log-earlier]").click()
    page.wait_for_function("document.querySelectorAll('#reader [data-log] .rd-ln').length === 262")
    assert page.errors == []


@pytest.fixture
def phone(browser):
    ctx = browser.new_context(viewport={"width": 400, "height": 860}, reduced_motion="reduce",
                              is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.errors = errors
    yield pg
    ctx.close()


@pytest.mark.parametrize("kind", ["dataset", "criteria", "log"])
def test_at_400px_the_reader_is_the_whole_screen_and_nothing_overflows(live, phone, dataset,
                                                                       run_log, kind):
    ctx = {"dataset": dataset, "log": run_log}
    phone.goto(live["base"] + "/" + ADDRESS[kind](ctx))
    ready(phone, kind)
    box = phone.evaluate("""() => { const r = document.querySelector('#reader .reader').getBoundingClientRect();
      return [r.left, r.width, r.height]; }""")
    assert box[0] == 0 and box[1] == 400 and box[2] >= 860
    over = phone.evaluate("""() => [...document.querySelectorAll('#reader .reader *')]
      .filter(e => { const r = e.getBoundingClientRect(); return r.width && r.right > 401
        && !e.closest('.lb-wrap, .rd-log, pre, table'); })
      .map(e => e.tagName + '.' + e.className).slice(0, 5)""")
    assert over == [], over
    assert phone.evaluate("document.querySelector('#reader .reader').scrollWidth") <= 400
    shot(phone, f"11g-{kind}-400-light.png")
    assert phone.errors == []


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_screenshots_for_the_pr(live, page, dataset, run_log, theme):
    ctx = {"dataset": dataset, "log": run_log}
    page.set_viewport_size({"width": 1400, "height": 900})
    for kind, addr in (("dataset", ADDRESS["dataset"](ctx) + ":3"), ("rubric", ADDRESS["rubric"](ctx)),
                       ("criteria", ADDRESS["criteria"](ctx)), ("bank", ADDRESS["bank"](ctx)),
                       ("log", ADDRESS["log"](ctx)),
                       ("provenance", ADDRESS["provenance"](ctx)),
                       ("provenance-judge", f"#model={MODEL.replace('/', '%2F')}&read=provenance:judge:{MODEL}")):
        page.goto(live["base"] + "/" + addr)
        page.wait_for_selector("#reader[data-ready='1']")
        page.evaluate(f"applyTheme('{theme}')")
        page.wait_for_timeout(250)
        shot(page, f"11g-{kind}-1400-{theme}.png")
    assert page.errors == []


def test_the_tab_underline_sits_under_a_tab_from_more(live, page):
    """Found while reading 11g's screenshots: More ▾ sits in a wrapper of its
    own, and 11f's underline measured it from there — it drew under Overview."""
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=review")
    page.wait_for_selector("#moreBtn[aria-selected='true']")
    page.wait_for_timeout(300)
    ink, btn = page.evaluate("""() => [document.getElementById('tabInk').getBoundingClientRect(),
      document.getElementById('moreBtn').getBoundingClientRect()].map(r => [r.left, r.width])""")
    assert abs(ink[0] - btn[0]) <= 1 and abs(ink[1] - btn[1]) <= 1, (ink, btn)
    assert page.errors == []
