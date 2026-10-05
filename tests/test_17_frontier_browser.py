"""17, stage 2 on the page: Test a model offers the Frontier benchmarks for a
served model, each one with its runs and budgets and, where it isn't run
Epoch AI's way, how; the thinking box and the benchmarks ticked are what is
queued — the dialog and the server agree (the Mobile suite's box was offered
and never sent). A Hub model sees the option greyed, saying why. The served
model is tests/fake_openai.py; nothing runs."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from fake_openai import FakeServer
from service import db
from test_12i4_browser import api

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase17"
SID = "served/Frontier-box"


@pytest.fixture(scope="module")
def srv(live):
    import service.app as appmod
    fake = FakeServer()
    fake.ctx = 40960
    api(live, "/api/served", {"name": "Frontier box", "base_url": fake.base,
                              "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "llama-server -c 40960",
                              "thinking": "off", "by": "masein"})
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield fake
    fake.close()


def open_test(page, live, mid: str):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + quote(mid, safe=""))
    page.wait_for_selector("[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()


def started(page):
    with page.expect_response(lambda r: r.url.endswith("api/submissions")
                              and r.request.method == "POST") as got:
        page.get_by_role("button", name="Start test").click()
    r = got.value
    return r.status, json.loads(r.request.post_data), r.json()


def test_the_frontier_benchmarks_thinking_on_are_what_is_queued(live, page, srv):
    open_test(page, live, SID)
    page.locator("[role='option'][data-value='frontier']").click()
    opts = page.locator("[data-frontier-opts]")
    opts.wait_for()
    # each benchmark, its runs and budgets, and how it differs from Epoch's
    hle = opts.locator("[data-frontier-task='hle_text_cais']").inner_text()
    assert "Humanity's Last Exam" in hle and "text-only questions" in hle and "graded" in hle
    assert "public set" in opts.locator("[data-frontier-task='arc_agi2_public']").inner_text()
    gpqa = opts.locator("[data-frontier-task='gpqa_diamond_epoch']").inner_text()
    assert "4 runs (Epoch AI: 16) · thinking on 32,768 tokens, off 4,096" in gpqa
    assert "2 attempts (pass@2)" in opts.locator("[data-frontier-task='arc_agi2_public']").inner_text()
    opts.locator("[data-think-switch] input").check()
    opts.locator("[data-frontier-task='arc_agi2_public'] input").uncheck()
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-dialog='test'] .dlg").screenshot(path=SCREENS / "test-a-model-frontier.png")
    st, body, j = started(page)
    assert st == 200, j
    assert body["suite"] == "frontier" and body["thinking"] is True
    assert "arc_agi2_public" not in body["tasks"] and len(body["tasks"]) == 6
    row = db.get(j["id"])
    assert row["thinking"] == 1 and "arc_agi2_public" not in json.loads(row["tasks"])
    assert page.errors == []


def test_the_mobile_suites_thinking_box_is_sent(live, page, srv):
    open_test(page, live, SID)
    page.locator("[role='option'][data-value='mobile']").click()
    page.locator("[data-mab-think] [data-think-switch] input").check()
    st, body, j = started(page)
    assert st == 200, j
    assert body["suite"] == "mobile" and body["thinking"] is True
    assert db.get(j["id"])["thinking"] == 1
    assert page.errors == []


def test_a_hub_model_sees_frontier_greyed_saying_why(live, page, srv):
    open_test(page, live, "fx/good-750m")
    opt = page.locator("[role='option'][data-value='frontier']")
    assert opt.get_attribute("aria-disabled") == "true" or opt.is_disabled()
    assert "running on a server" in (opt.get_attribute("title") or opt.inner_text())
    assert page.errors == []
