"""12z D on the page: Improve with the judge offline (D1), a served setup's
launch beside what its server reports (D2), the Playground's suggestions
across new chats (D3), and the Runs list's superseded runs, their error once
and a Resubmit that keeps what the run was (D5) — at 1400 and 375 px.
DeviceMark's answers past 200 (D4) are in test_12q_devicemark_model_page_browser,
beside their fixture. Fixtures and route stubs only; question ids only."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

import everyday as ev
from conftest import set_name
from fake_openai import FakeServer

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12z"
WIDTHS = [1400, 375]
DOWN = {"ok": False, "checked": True, "provider": "local", "url": "http://judge.test/v1",
        "why": "the grading model at http://judge.test/v1 did not answer"}


def clip(page, sel, name):
    """a picture of the page where `sel` is: a poll may draw the element again
    while a screenshot of the element itself waits for it to hold still"""
    box = page.evaluate("""s => { const r = document.querySelector(s).getBoundingClientRect();
      return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height }; }""", sel)
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / name, clip=box, full_page=True)


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def start(page, live, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")


# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------

def _judge(page, health, data_local):
    page.route("**/api/judge/health*", lambda r: r.fulfill(json=health))

    def llm(route):
        r = route.fetch()
        body = r.json()
        body["ai_local"] = {**body.get("ai_local", {}), "data": data_local}
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/llm", llm)


@pytest.mark.parametrize("width", WIDTHS)
def test_improve_says_the_judge_is_offline_and_propose_waits(live, page, width):
    _judge(page, DOWN, True)
    start(page, live, width)
    page.goto(live["base"] + "/#tab=improve&sub=model")
    btn = page.locator("[data-imp-propose]")
    btn.wait_for()
    page.wait_for_selector("[data-imp-propose][data-propose-down]")
    assert btn.is_disabled()
    assert btn.get_attribute("title").startswith("The grading model at http://judge.test/v1 did "
                                                 "not answer. The judge and the training-data writer")
    why = page.locator("[data-propose-down-why]")
    assert "judge offline" in why.inner_text() and "start it, then Propose" in why.inner_text()
    # the AI line says it too, beside the judge's name
    assert page.locator("[data-ai-line] [data-judge-offline]").count() == 1
    # every weak spot's Propose waits as well
    weak = page.locator("[data-weak-propose]")
    assert weak.count() == 0 or all(b.is_disabled() for b in weak.all())
    no_sideways(page)
    clip(page, "[data-pipeline]", f"improve-judge-offline-{width}.png")
    assert page.errors == []


def test_propose_stays_when_the_writer_is_not_on_the_server_that_is_down(live, page):
    _judge(page, DOWN, False)
    start(page, live, 1400)
    page.goto(live["base"] + "/#tab=improve&sub=model")
    page.wait_for_selector("[data-imp-propose]")
    page.wait_for_selector("[data-ai-line] [data-judge-offline]")
    assert not page.locator("[data-imp-propose]").is_disabled()
    assert page.locator("[data-propose-down-why]").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stale(live):
    """a setup whose free text says --cpu-moe while its launch is --n-cpu-moe 21"""
    fake = FakeServer()
    sid = api(live, "/api/served", {"name": "D2 stale text", "base_url": fake.base,
                                    "thinking": "off", "how": "llama-server -ngl 99 --cpu-moe",
                                    "flags": "-ngl 99 --n-cpu-moe 21", "by": "masein"})["model"]["id"]
    yield sid
    fake.close()


@pytest.mark.parametrize("width", WIDTHS)
def test_a_served_page_shows_its_launch_and_flags_the_stale_text(live, page, stale, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#model=" + stale.replace("/", "%2F"))
    box = page.locator(f"[data-served-launch='{stale}']")
    box.wait_for()
    assert box.locator(f"[data-served-launch-reg='{stale}']").inner_text() == "-ngl 99 --n-cpu-moe 21"
    assert box.locator(f"[data-served-mismatch='{stale}']").all_inner_texts() == [
        "“How it’s served” says --cpu-moe; the launch flags registered don’t"]
    # it sits under the free text it is about
    assert page.evaluate(f"""() => {{ const how = document.querySelector('[data-served-how="{stale}"]');
        const box = document.querySelector('[data-served-launch="{stale}"]');
        return !!(how.compareDocumentPosition(box) & Node.DOCUMENT_POSITION_FOLLOWING); }}""")
    no_sideways(page)
    clip(page, f"[data-served-head='{stale}']", f"served-launch-{width}.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------

def test_the_playgrounds_suggestions_never_hold_a_hidden_id_over_many_chats(live, page):
    start(page, live, 1400)
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=playground")
    page.wait_for_selector("[data-pg-empty] [data-pg-chip]")
    store = {q["id"] for q in ev._raw_rows(ev.hidden_path())}
    assert store
    seen = set()
    for _ in range(12):                      # a new chat draws again
        ids = page.locator("[data-pg-chip]").evaluate_all(
            "xs => xs.map(x => x.dataset.pgChip.split('|').slice(1).join('|'))")
        seen |= set(ids)
        page.evaluate("pgNew()")
        page.wait_for_selector("[data-pg-empty] [data-pg-chip]")
    assert seen and not seen & store
    assert page.errors == []


# ---------------------------------------------------------------------------
# D5
# ---------------------------------------------------------------------------

def _run(i, **kw):
    return {"id": i, "hf_id": "org/m", "kind": "instruct", "suite": "generative", "status": "done",
            "submitter": "masein", "note": "", "progress": "done", "error": "", "part": "",
            "thinking": 0, "subset": 0, "bbq_all": 0, "pair": "", "tasks": "[]",
            "created_at": 1.7e9 + i, "started_at": None, "finished_at": None, "gpu_seconds": 60.0,
            **kw}


OOM = "CUDA out of memory: the run itself grew past the card"
RUNS = [_run(14, status="done"),
        _run(13, suite="devicemark", part="pilot", thinking=1, hf_id="served/x", status="failed",
             progress="failed", error="the server stopped answering"),
        _run(12, status="failed", progress=OOM, error=OOM),
        # a later run of another part doesn't replace it
        _run(11, suite="devicemark", part="parity", hf_id="served/x", status="done"),
        _run(10, status="failed", progress=OOM, error=OOM)]


@pytest.mark.parametrize("width", WIDTHS)
def test_superseded_runs_say_so_their_error_is_said_once_and_resubmit_keeps_the_run(
        live, page, width):
    posted = []

    def handle(route):
        if route.request.method == "POST":
            posted.append(json.loads(route.request.post_data))
            return route.fulfill(json={"id": 99, "status": "queued"})
        if "/count" in route.request.url:
            return route.fulfill(json={"total": len(RUNS)})
        return route.fulfill(json=RUNS)
    page.route("**/api/submissions**", handle)
    start(page, live, width)
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("tr[data-queue-row='10']")
    # #10 and #12 failed, and #14 — the same model and suite — finished after them
    for rid in ("10", "12"):
        row = page.locator(f"tr[data-queue-row='{rid}']")
        assert row.locator("[data-row-superseded]").inner_text() == "superseded by #14"
        assert row.locator("[data-row-resubmit]").count() == 0
        # the error once: in its own place, not as the progress too
        assert row.inner_text().count("CUDA out of memory") == 1
    # #13's part has no later run: Resubmit, and it asks for the part and mode it ran
    row = page.locator("tr[data-queue-row='13']")
    assert row.locator("[data-row-superseded]").count() == 0
    row.locator("[data-row-resubmit]").click()
    page.wait_for_function("document.querySelector('[data-toast]')")
    assert posted and posted[-1]["part"] == "pilot" and posted[-1]["thinking"] is True
    assert posted[-1]["hf_id"] == "served/x" and posted[-1]["suite"] == "devicemark"
    # the superseding run is followed from the link
    page.locator("tr[data-queue-row='10'] [data-row-superseded]").click()
    page.wait_for_function("state.qMark === 14")
    no_sideways(page)
    clip(page, ".lb-wrap:has([data-queue-table])", f"runs-superseded-{width}.png")
    assert page.errors == []
