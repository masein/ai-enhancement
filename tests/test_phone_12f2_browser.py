"""12f.2 on the page — 12f.2b: not a view of its own, its old link choosing the
phone builds on Models, and the card on the model page. Before: On phone was the
fourth switch on Models, after Everyday
tasks, only once a phone build is registered. Its card shows what was
measured on the phone as reported — every number "reported by <name>", with
the date and the source — beside what the board measured through the served
model, its base beside that. The model page has the same card as its fourth
kind, and no reported number is anywhere else on the page."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from conftest import set_name
from fake_openai import FakeServer
from service import config, db, phone, runner

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12f2"
NAME = "Qwen3.6-35B-A3B k4-LDA (phone build)"
SID = "served/Qwen3.6-35B-A3B-k4-LDA-phone-build"
HOW = "llama.cpp fork teraformer/lda-2026-09-22 @ 91428471f, --cpu-moe, lookahead 1, fusion off"
BASE = "fx/good-750m"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def post(live, path, body):
    req = urllib.request.Request(live["base"] + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def register(live, fake, phone):
    post(live, "/api/served", {"name": NAME, "base_url": fake.base, "based_on": BASE,
                               "how": HOW, "thinking": "off", "phone": phone, "by": "masein"})


@pytest.fixture(scope="module")
def fake(live):
    s = FakeServer()
    yield s
    s.close()


def sit_everyday() -> None:
    """measured by the board: an Everyday run through the served model"""
    saved = runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL
    runner.acquire_lock, runner.release_lock = (lambda sid: True), (lambda: None)
    config.JUDGE_MODEL = "stub"
    try:
        sid = db.add(SID, "instruct", "everyday", "masein", "")
        runner.run_submission(db.get(sid))
    finally:
        runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL = saved
    assert db.get(sid)["status"] == "done"


def ready(live, fake) -> None:
    """what the card test leaves — the phone build, its Everyday run, Sam's
    report — for a test that reads it: pytest-split can put the two in
    different shards (12k.2's new tests moved the line between them)"""
    register(live, fake, phone=True)
    if not (config.OUT_DIR / SID.replace("/", "__") / "everyday.json").exists():
        sit_everyday()
    if not phone.reports(SID):
        post(live, "/api/phone/reports", {**phone.README, "model": SID, "by": "Sam",
                                          "date": "2026-09-25", "entered_by": "masein"})


def views(page, live):
    page.goto("about:blank")                  # a load, not a hash change: the page's data anew
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-models-switch]")
    return page.locator("[data-models-switch] [role=tab]").all_inner_texts()


def test_on_phone_is_no_view_and_its_old_link_chooses_the_phone_builds(live, page, fake):
    """12f.2b: phone builds are rows on Models; the old On phone link opens
    Models with them chosen"""
    page.set_viewport_size({"width": 1400, "height": 900})
    register(live, fake, phone=True)
    assert views(page, live) == ["Standard", "Knowledge exam", "Everyday tasks"]
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&view=phone")
    page.wait_for_function("location.hash.includes('models=')")
    assert "models=" + SID.replace("/", "%2F") in page.evaluate("location.hash")
    assert page.locator("[data-models-menu]").inner_text() == "Models: 1 ▾"
    assert page.errors == []


def test_the_card_shows_what_was_reported_beside_what_the_board_measured(live, page, fake):
    register(live, fake, phone=True)
    sit_everyday()
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    # 12f.2b: on the model page, its "On the phone · reported" block
    page.goto(live["base"] + "/#model=" + SID.replace("/", "%2F"))
    page.locator("[data-kind-tile='phone'] [data-kind-test='phone']").click()
    card = page.locator(f"[data-phone-card='{SID}']")
    card.wait_for()
    assert card.locator(f"[data-phone-none='{SID}']").inner_text() == \
        "Nothing reported from the phone yet."
    # typed in: the README's numbers, then who measured them and when — Add
    # opened the form
    assert page.locator(f"[data-phone-form='{SID}']").get_attribute("open") == ""
    page.locator(f"[data-phone-readme='{SID}']").click()
    assert page.locator("[data-phone-in='device']").input_value() == "OnePlus 15"
    assert page.locator("[data-phone-in='by']").input_value() == ""
    page.fill("[data-phone-in='by']", "Sam")
    page.fill("[data-phone-in='date']", "2026-09-25")
    page.locator(f"[data-phone-save='{SID}']").click()
    rep = page.locator(f"[data-phone-reported='{SID}'] dl")
    rep.wait_for()
    assert page.locator(f"[data-phone-source='{SID}']").inner_text() == \
        "Reported by Sam, measured 2026-09-25 · from the fork's README (teraformer/lda-2026-09-22)" \
        " · entered by masein"
    fields = dict(zip(rep.locator("dt").all_inner_texts(), rep.locator("dd").all_inner_texts()))
    assert fields == {
        "device": "OnePlus 15 · reported by Sam",
        "chip and RAM": "Snapdragon 8 Elite Gen 5 · 16 GB RAM · reported by Sam",
        "decode": "13.5 tok/s median, 16 best (3 cold repeats, ≤65 °C) · reported by Sam",
        "settings": "experts streamed from flash, lookahead 1, MTP n_max 3, fusion off"
                    " · reported by Sam",
        "MMLU, as reported": "81.98% (all 14,042, measured on Metal) · reported by Sam"}
    # beside it, what this board measured through the served model, and its base
    meas = page.locator(f"[data-phone-measured='{SID}']")
    head = meas.locator("thead th").evaluate_all("xs => xs.map(x => x.textContent)")
    assert head == ["", "served", "good-750m, loaded here"]
    row = meas.locator("[data-phone-row='Everyday tasks'] td").all_inner_texts()
    assert row[0] == "Everyday tasks" and " of 179" in row[1] and " of " in row[2]   # 12a.6
    # the server's own speed is not the phone's, and isn't shown
    assert "tok/s" not in meas.inner_text()
    shot(page.locator("[data-kind-block='phone']"), "on-phone-1400.png")
    page.set_viewport_size({"width": 400, "height": 900})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page, "on-phone-400.png", full_page=True)
    assert page.errors == []


def test_the_model_page_has_it_as_its_fourth_kind_and_nowhere_else(live, page, fake):
    ready(live, fake)
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#model=" + SID.replace("/", "%2F"))
    tile = page.locator("[data-kind-tile='phone']")
    tile.wait_for()
    assert page.locator("[data-kind-tile]").evaluate_all("xs => xs.map(x => x.dataset.kindTile)") \
        == ["standard", "exam", "everyday", "phone"]
    assert tile.locator("[data-kind-value='phone']").inner_text() == "13.5"
    assert "tok/s median on OnePlus 15 · reported by Sam" in tile.inner_text()
    tile.click()
    page.wait_for_selector(f"[data-kind-block='phone'] [data-phone-card='{SID}']")
    # the reported MMLU is in the card, and in no tile or average
    body = page.locator("body").inner_text()
    assert body.count("81.98%") == 1
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.evaluate("scrollTo(0, 0)")
    shot(page, "model-page-phone.png")
    # 12f.2b: on Models it is in the reported group alone — never in the MMLU
    # column the board measured, which a served model can't have
    page.goto(live["base"] + "/#tab=models&chip=knowledge")
    page.wait_for_selector("table[data-lb-table]")
    assert page.locator("[data-rep-cell]").count() == 0      # not a row here: nothing it can have
    # on Everyday tasks, where it is a row, beside its score
    page.goto(live["base"] + "/#tab=models")
    page.locator("[data-models-view='everyday']").click()
    row = page.locator(f"tr[data-lb-row='{SID}']")
    row.wait_for()
    assert row.locator("[data-rep-cell='rep:q:MMLU']").inner_text().startswith("81.98%")
    assert row.locator("[data-rep-cell='rep:median']").inner_text() == "13.5\nreported by Sam"
    assert page.errors == []
