"""12f.1 on the page, against a fake OpenAI-compatible server: a served
model is a normal row on Models with a grey "served" tag; its page says how
it is served, what its server reported, the one line about multiple-choice
benchmarks and how it compares with its base, and has no Improve tab;
History shows the pinned details with each run; and Test a model's fourth
way checks a server and saves it, or says in one line that nothing answered."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from conftest import set_name
from fake_openai import FakeServer, nothing_listening
from service import config, db, runner

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12f1"
NAME = "Qwen3.6-35B-A3B k4-LDA (phone build)"
SID = "served/Qwen3.6-35B-A3B-k4-LDA-phone-build"
HOW = "llama.cpp fork teraformer/lda-2026-09-22 @ 91428471f, --cpu-moe, lookahead 1, fusion off"
BASE = "fx/good-750m"
LINE = "Multiple-choice benchmarks need the model loaded here; this one is served elsewhere."
PIN = "Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf · 22.9 GB · context 16,384 · build b6500-91428471f"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def post(live, path, body):
    req = urllib.request.Request(live["base"] + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(live, path):
    with urllib.request.urlopen(live["base"] + path) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def srv(live):
    """registered through the API, and an Everyday run through the real
    runner — its answers from the fake server, marked with the stub judge"""
    import service.app as appmod
    fake = FakeServer()
    post(live, "/api/served", {"name": NAME, "base_url": fake.base, "based_on": BASE,
                               "how": HOW, "thinking": "off", "by": "masein"})
    saved = runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL
    runner.acquire_lock, runner.release_lock = (lambda sid: True), (lambda: None)
    config.JUDGE_MODEL = "stub"
    try:
        sid = db.add(SID, "instruct", "everyday", "masein", "")
        runner.run_submission(db.get(sid))
    finally:
        runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL = saved
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    # and an IFEval score, as lm_eval's local-chat-completions leaves it
    from service import served
    d = config.OUT_DIR / SID.replace("/", "__") / "ifeval_0shot" / "served"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_2026-09-27T10-00-00.000000.json").write_text(json.dumps({
        "results": {"ifeval": {"alias": "ifeval", "prompt_level_strict_acc,none": 0.62}},
        "config": {"model": "local-chat-completions", "model_args": f"pretrained={SID},x=1"},
        "served": served.view(served.get(SID)), "date": 2.0}))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield fake, sid
    fake.close()


def no_sideways(page):
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def test_on_models_a_served_model_is_a_normal_row_with_a_grey_served_tag(live, page, srv):
    page.set_viewport_size({"width": 1400, "height": 900})
    # Instruction & maths: IFEval, MMLU-Pro and MATH-500, which it can sit
    page.goto(live["base"] + "/#tab=models&chip=instruction")
    cell = page.locator(f"td.model[data-model='{SID}']")
    cell.wait_for()
    assert cell.locator(".mname").inner_text() == NAME
    tag = cell.locator(f"[data-served-tag='{SID}']")
    assert tag.inner_text() == "served" and tag.get_attribute("title") == HOW
    # grey: the secondary text colour, as the other quiet badges
    grey = page.evaluate("getComputedStyle(document.querySelector('.badge.ckpt, .badge.served'))"
                         ".color")
    assert tag.evaluate("e => getComputedStyle(e).color") == grey
    shot(cell.locator("xpath=.."), "models-row.png")
    # and on Everyday tasks, the same tag
    page.locator("[data-models-view='everyday']").click()
    tag = page.locator(f"td.model[data-model='{SID}'] [data-served-tag='{SID}']")
    tag.wait_for()
    assert tag.get_attribute("title") == HOW
    assert page.errors == []


def test_its_page_says_how_it_is_served_once_and_has_no_improve(live, page, srv):
    for width in (1400, 400):
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(live["base"] + "/#model=" + SID.replace("/", "%2F"))
        page.wait_for_selector("[data-served-head]")
        assert page.locator(f"[data-served-how='{SID}']").inner_text() == "served " + HOW
        assert page.locator(f"[data-served-pin='{SID}']").inner_text() == \
            f"Its server reported {PIN}. Every run checks it still does."
        # the one line, once; no column of dashes and no list of what it can't sit
        assert page.locator("body").inner_text().count(LINE) == 1
        assert page.locator("[data-avg-verdict]").count() == 0
        # beside its base, never averaged with it
        cmp = page.locator(f"[data-served-compare='{SID}']").inner_text()
        assert cmp.startswith("Compared with good-750m loaded here: Everyday ")
        assert " vs " in cmp
        # no Improve: there are no weights here to train. Chat stays, as on every
        # model page (12d.2), and (12d.3) chats with it through its server
        tabs = page.locator("[data-model-tabs] [role=tab]").all_inner_texts()
        assert tabs == ["Scores", "Answers", "Chat", "History"]
        assert no_sideways(page)
        shot(page, f"model-page-{width}.png", full_page=True)
    page.set_viewport_size({"width": 1400, "height": 900})
    page.locator("[data-mtab='chat']").click()
    page.wait_for_selector(f"[data-model-chat='{SID}']")
    assert page.locator("[data-model-chat-none]").count() == 0          # offered, not refused
    assert page.errors == []


def test_history_shows_the_pinned_details_with_each_run(live, page, srv):
    _, sid = srv
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#model=" + SID.replace("/", "%2F"))
    page.locator("[data-mtab='history']").click()
    line = page.locator(f"[data-run-served='{sid}']")
    line.wait_for()
    assert line.inner_text() == f"served: {PIN} · thinking off · {HOW}"
    prov = page.locator(f"[data-model-prov='{SID}']").inner_text()
    assert "served at" in prov and HOW in prov and "hub id" not in prov
    shot(page.locator(f"tr[data-run='{sid}']"), "history-run.png")
    assert page.errors == []


def open_card(page, live, width=1400):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.locator("[data-test-model]").click()
    page.locator("[data-served-card] > summary").click()
    page.wait_for_selector("[data-srv='name']")


def test_test_a_model_checks_a_server_and_saves_it(live, page, srv):
    fake, _ = srv
    open_card(page, live)
    page.fill("[data-srv='name']", "LDA test build")
    page.fill("[data-srv='base_url']", fake.base)
    page.fill("[data-srv='how']", HOW)
    page.locator("[data-srv-check]").click()
    rep = page.locator("[data-srv-reported]")
    rep.wait_for()
    assert rep.inner_text() == f"The server reports: {PIN}"
    assert "served/LDA-test-build" not in [m["id"] for m in get(live, "/api/served")["models"]]
    page.locator("[data-srv-save]").click()
    page.wait_for_function("() => (document.querySelector('[data-srv-reported]') || {})"
                           ".textContent?.startsWith('Saved')")
    assert "served/LDA-test-build" in [m["id"] for m in get(live, "/api/served")["models"]]
    # picked for the test, with what it can sit and the one line
    assert page.locator("[data-ms='submit'] input").input_value() == "served/LDA-test-build"
    assert page.locator("[data-why='served']").inner_text() == "served " + LINE
    assert page.locator("[data-select='kind']").count() == 0
    page.locator("[data-select='suite']").click()
    for v in ("full", "quick", "control"):
        o = page.locator(f"[role=option][data-value='{v}']")
        assert o.get_attribute("aria-disabled") == "true" and o.get_attribute("title") == LINE
    for v in ("everyday", "generative"):
        assert page.locator(f"[role=option][data-value='{v}']").get_attribute(
            "aria-disabled") is None
    # judged is greyed only when there's no judge, as for any model: not for this
    assert page.locator("[role=option][data-value='judged']").get_attribute("title") != LINE
    page.keyboard.press("Escape")
    # the key field is empty again: it is never shown
    assert page.locator("[data-srv='key']").input_value() == ""
    shot(page.locator("[data-dialog='test'] .dlg"), "test-a-model-served.png")
    assert page.errors == []


def test_nothing_answering_is_one_line_and_nothing_is_saved(live, page, srv):
    open_card(page, live, width=400)
    gone = nothing_listening()
    n = len(get(live, "/api/served")["models"])
    page.fill("[data-srv='name']", "Nowhere")
    page.fill("[data-srv='base_url']", gone)
    page.fill("[data-srv='how']", "not running")
    page.locator("[data-srv-save]").click()
    msg = page.locator("[data-srv-msg]")
    msg.wait_for()
    assert msg.inner_text().startswith(f"Nothing answered at {gone}")
    assert len(get(live, "/api/served")["models"]) == n
    assert no_sideways(page)
    shot(page.locator("[data-served-card]"), "served-nothing-answered-400.png")
    # the refusal it asked for, and nothing else
    assert [e for e in page.errors if "status of 422" not in e] == []
