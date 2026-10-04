"""12i.4 on the page, for served models: the answer length and ran-out count
beside the base's in the Compare line; a served model with no result listed
with Test and its page open; Test a model's estimate from measured speed; and
the Everyday total with its ran-out note on a line of its own. The served
model is tests/fake_openai.py. A module of its own: publishing a batch (the
other module) changes the Everyday wording, and with it who is compared."""

from __future__ import annotations

import pytest

from fake_openai import FakeServer
from service import config, db, runner
from test_12i4_browser import HOW, IDLE, SID, api, shot

pytestmark = pytest.mark.dashboard


@pytest.fixture(scope="module")
def served_run(live):
    """a served thinking model, run on Everyday through the real runner"""
    import service.app as appmod
    fake = FakeServer()
    n = {"i": 0}

    def reply(body):
        n["i"] += 1
        return "" if n["i"] % 9 == 0 else "The answer."
    fake.reply, fake.reasoning = reply, "Let me think."
    fake.tokens = lambda body: 4096 if n["i"] % 9 == 0 else 990
    for name in ("LDA phone", "Never run"):
        api(live, "/api/served", {"name": name, "base_url": fake.base, "based_on": "fx/good-750m",
                                  "how": HOW, "thinking": "on", "by": "masein"})
    saved = runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL
    runner.acquire_lock, runner.release_lock = (lambda sid: True), (lambda: None)
    config.JUDGE_MODEL = "stub"
    try:
        sid = db.add(SID, "instruct", "everyday", "masein", "")
        runner.run_submission(db.get(sid))
    finally:
        runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL = saved
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield fake
    fake.close()


def test_a_served_models_answer_length_beside_its_bases(live, page, served_run):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + SID.replace("/", "%2F"))
    line = page.locator("[data-answer-length='everyday']")
    line.wait_for()
    txt = line.inner_text()
    # every answer, practice and hidden: "ran out 43 of 388"
    out, of = txt.split("ran out ")[1].split(" (")[0].split(" of ")
    assert txt.endswith(" (every answer, practice and hidden)")              # 12z C2
    assert txt.startswith("median 990 tokens · ran out ") and 0 < int(out) < int(of)
    assert "as the server counted them" in line.get_attribute("title")
    cmp = page.locator(f"[data-served-compare='{SID}']").inner_text()
    assert "Compared with good-750m loaded here: Everyday " in cmp
    assert " · median 990 vs " in cmp and " tokens · ran out " in cmp
    shot(page, "served-answer-length.png")
    assert page.errors == []


def test_a_served_model_with_no_result_is_listed_and_its_page_opens(live, page, served_run):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.locator("[data-not-tested-toggle]").click()
    row = page.locator(f"[data-not-tested-row='{IDLE}']")
    row.wait_for()
    assert row.locator(f"[data-not-tested-test='{IDLE}']").inner_text() == "Test"
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + IDLE.replace("/", "%2F"))
    page.wait_for_selector(f"[data-served-head='{IDLE}']")
    assert page.locator("h1.mtitle").inner_text() == "Never run"
    assert page.errors == []


def test_the_estimate_uses_the_measured_seconds_an_answer(live, page, served_run):
    page.set_viewport_size({"width": 1400, "height": 1000})
    for mid, rough in ((SID, False), (IDLE, True)):
        page.goto("about:blank")
        page.goto(live["base"] + "/#model=" + mid.replace("/", "%2F"))
        # its page first: the header's Test button is there before the page
        # knows the model, and a click then opens the dialog with none filled in
        page.wait_for_selector(f"[data-served-head='{mid}']")
        page.locator("[data-test-model]").click()
        suite = page.locator("[data-dialog='test'] [data-select='suite']")
        suite.wait_for()
        words = suite.locator('.sel-v').inner_text()
        assert words.startswith("Everyday — ") and " min" in words, words
        assert words.endswith(", a rough guess") is rough, words
    assert page.errors == []


def test_the_everyday_total_and_its_ran_out_note_do_not_overlap(live, page, served_run):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(live["base"] + "/#tab=models")
    page.locator("[data-models-view='everyday']").click()
    cell = page.locator(f"tr[data-lb-row='{SID}'] td.evdtotal")
    cell.wait_for()
    total, note = cell.locator("div").first.bounding_box(), cell.locator(".evd-ranout").bounding_box()
    box = cell.bounding_box()
    assert note["y"] >= total["y"] + total["height"] - 1, (total, note)
    assert note["x"] + note["width"] <= box["x"] + box["width"] + 1, (note, box)
    shot(cell.locator("xpath=.."), "everyday-total-ran-out-1280.png")
    assert page.errors == []
