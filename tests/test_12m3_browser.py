"""12m.3 on the page, against the fake OpenAI-compatible server answering at
OpenRouter's address (every urllib request in the process goes through
`outside` below; any other outside address fails the module). Nothing calls
OpenRouter, and no model runs.

Test a model ▸ A model from OpenRouter: the list AI models shows, under
each maker; Add pins one and picks it; what the run would cost shows before
Start, and Start is refused with the limit's line when it would pass it. On
Models its row is tagged "via OpenRouter", and the Models picker groups it
under its maker."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from conftest import set_name
from fake_openai import FakeServer
from service import config, db, runner
from test_12m3 import CATALOG, ENDPOINTS, KEY, LUNA, LUNA_V, OR, SID

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12m3"
GEM = "google/gemini-4-flash"
GEM_SID = "served/openrouter-google-gemini-4-flash"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def call(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def orsrv(live):
    """OpenRouter's address answered by the fake; LUNA added through the API
    and an Everyday run of it through the real runner — its answers from the
    fake, marked with the stub judge"""
    import service.app as appmod
    fake = FakeServer()
    fake.key, fake.catalog, fake.endpoints = KEY, [dict(m) for m in CATALOG], list(ENDPOINTS)
    fake.provider = "OpenAI"
    real, tried = urllib.request.urlopen, []

    def urlopen(req, *a, **k):
        if not isinstance(req, urllib.request.Request):
            req = urllib.request.Request(req)
        if req.full_url.startswith(OR + "/"):
            req.full_url = fake.base + req.full_url[len(OR):]
        if urllib.parse.urlsplit(req.full_url).hostname not in ("127.0.0.1", "localhost"):
            tried.append(req.full_url)
            raise urllib.error.URLError(f"a test reached outside: {req.full_url}")
        return real(req, *a, **k)
    names = ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_CONCURRENCY", "JUDGE_MODEL")
    saved = ({k: getattr(config, k) for k in names}, runner.acquire_lock, runner.release_lock)
    urllib.request.urlopen = urlopen
    config.OPENROUTER_API_KEY, config.OPENROUTER_BASE_URL = KEY, OR
    config.OPENROUTER_CONCURRENCY, config.JUDGE_MODEL = 1, "stub"
    runner.acquire_lock, runner.release_lock = (lambda sid: True), (lambda: None)
    try:
        assert call(live, "/api/served/openrouter", {"model": LUNA, "by": "masein"})[0] == 200
        sid = db.add(SID, "instruct", "everyday", "masein", "")
        runner.run_submission(db.get(sid))
        assert db.get(sid)["status"] == "done", db.get(sid)["error"]
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield fake, sid
    finally:
        urllib.request.urlopen = real
        for k, v in saved[0].items():
            setattr(config, k, v)
        runner.acquire_lock, runner.release_lock = saved[1], saved[2]
        fake.close()
    assert tried == []


def open_card(page, live, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.locator("[data-test-model]").click()
    page.locator("[data-openrouter-card] > summary").click()
    page.wait_for_selector(f"[data-or-row='{GEM}']")


def no_sideways(page):
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def test_a_model_from_openrouter_is_added_and_its_cost_shows_before_start(live, page, orsrv):
    db.ai_set("spend_limit", 1000.0, "masein")
    open_card(page, live)
    card = page.locator("[data-openrouter-card]")
    # AI models' list, under each maker; the one added already can be tested
    assert card.locator("[data-or-maker]").all_inner_texts() == \
        ["Anthropic", "Google", "Mistral", "OpenAI"]
    assert card.locator(f"[data-or-test='{SID}']").inner_text() == "Test it"
    assert "$0.30 in · $2.50 out per million" in card.locator(f"[data-or-row='{GEM}']").inner_text()
    card.locator("[data-or-search]").fill("gemini")
    assert card.locator("[data-or-row]").count() == 1
    card.locator(f"[data-or-add='{GEM}']").click()
    msg = card.locator("[data-or-msg]")
    msg.wait_for()
    assert msg.inner_text() == ("Added Gemini 4 Flash: pinned to google/gemini-4-flash-20260901 on "
                                "OpenAI, with no fallbacks. Pick what to test it on above — what it "
                                "would cost shows before Start.")
    # picked for the test, tagged, with what the run would cost — before Start
    assert page.locator("[data-ms='submit'] input").input_value() == GEM_SID
    assert page.locator(f"[data-why='served'] [data-via-openrouter='{GEM_SID}']").inner_text() == \
        "via OpenRouter"
    line = page.locator("[data-or-estimate]:not([data-or-estimate='working'])")
    line.wait_for()
    _, est = call(live, "/api/served/estimate", {"model": GEM_SID, "suite": "everyday"})
    assert est["line"].startswith("about $") and est["n"] > 0
    said = line.inner_text()
    assert said.startswith(f"About {est['line'][len('about '):]} for {est['n']} questions — each "
                           "answer counted at its full length, so it usually costs less · this "
                           "month $"), said
    assert said.endswith(" of the $1,000.00 limit")
    start = page.get_by_role("button", name="Start test")
    assert start.is_enabled()
    shot(page.locator("[data-dialog='test'] .dlg"), "12m3-test-a-model-openrouter-1400.png")
    # Instruction & maths: a seeded MMLU-Pro subset, unless cleared
    page.locator("[data-select='suite']").click()
    page.locator("[role=option][data-value='generative']").click()
    assert page.locator("[data-subset-input]").input_value() == str(config.OPENROUTER_GEN_SUBSET)
    page.wait_for_function("() => { const e = document.querySelector('[data-or-estimate]');"
                           " return e && /^About/.test(e.textContent) && /2,041 questions/"
                           ".test(e.textContent); }")
    assert page.errors == []


def test_start_is_refused_with_the_limits_line_when_it_would_pass_it(live, page, orsrv):
    db.ai_set("spend_limit", 0.5, "masein")
    try:
        open_card(page, live, width=400)
        page.locator(f"[data-or-test='{SID}']").click()
        # its Everyday questions are all answered: nothing to pay for
        page.wait_for_selector("[data-or-estimate='0']")
        assert page.locator("[data-or-estimate]").inner_text().startswith(
            "Under $0.01 for no new questions") or page.locator(
            "[data-or-estimate]").inner_text().startswith("About $0.00 for no new questions")
        # IFEval, MMLU-Pro and MATH-500 at its thinking budget would pass the limit
        page.locator("[data-select='suite']").click()
        page.locator("[role=option][data-value='generative']").click()
        refused = page.locator("[data-or-estimate='refused']")
        refused.wait_for()
        body = {"model": SID, "suite": "generative", "subset": config.OPENROUTER_GEN_SUBSET}
        _, est = call(live, "/api/served/estimate", body)
        assert est["refused"].startswith(f"This run could cost {est['line']}, more than the $")
        assert est["refused"].endswith("of this month's $0.50 AI limit — test fewer questions, "
                                       "or raise the limit on AI models.")
        assert refused.inner_text() == est["refused"]
        start = page.get_by_role("button", name="Start test")
        assert start.is_disabled() and start.get_attribute("title") == est["refused"]
        # and the server refuses it in the same words, nothing queued
        st, j = call(live, "/api/submissions", {"hf_id": SID, "suite": "generative",
                                               "subset": config.OPENROUTER_GEN_SUBSET,
                                               "submitter": "masein"})
        assert st == 409 and j["detail"] == est["refused"] + " Nothing was queued."
        assert no_sideways(page)
        shot(page.locator("[data-dialog='test'] .dlg"), "12m3-start-refused-400.png")
    finally:
        db.ai_set("spend_limit", 20.0, "masein")
    assert page.errors == []


def test_its_row_is_tagged_via_openrouter_and_the_picker_groups_it_under_its_maker(
        live, page, orsrv):
    _, sid = orsrv
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-card] [data-pickers]")
    # Models ▾: its maker's group, the row saying where it runs
    page.locator("#pill-models").click()
    panel = page.locator("#pop-models")
    panel.wait_for()
    groups = panel.locator("[data-model-group]").evaluate_all(
        "es => es.map(e => e.dataset.modelGroup)")
    assert "OpenAI" in groups and "served" not in groups
    rows = panel.locator(".mgroup, .mrow").evaluate_all(
        "es => es.map(e => e.dataset.modelGroup || e.querySelector('input').dataset.modelPick)")
    assert rows[rows.index("OpenAI") + 1] == SID
    assert panel.locator(f"[data-pick-via='{SID}']").inner_text().strip() == "via OpenRouter"
    shot(panel, "12m3-models-picker.png")
    page.keyboard.press("Escape")
    # on Everyday tasks, its row tagged "via OpenRouter", how it runs in the tooltip
    page.locator("[data-models-view='everyday']").click()
    tag = page.locator(f"td.model[data-model='{SID}'] [data-served-tag='{SID}']")
    tag.wait_for()
    assert tag.inner_text() == "via OpenRouter"
    assert tag.get_attribute("title") == f"OpenRouter · {LUNA_V} · on OpenAI · no fallbacks"
    shot(page.locator(f"td.model[data-model='{SID}']").locator("xpath=.."), "12m3-models-row.png")
    # its page: what it is pinned to, checked at every run
    page.goto(live["base"] + "/#model=" + SID.replace("/", "%2F"))
    pin = page.locator(f"[data-served-pin='{SID}']")
    pin.wait_for()
    assert pin.inner_text() == (f"Pinned to {LUNA_V} · on OpenAI · context 400,000, with no "
                                "fallbacks. Every run checks it still is.")
    # its run's row says what it cost on OpenRouter
    assert db.get(sid)["progress"].endswith(" on OpenRouter")
    assert page.errors == []
