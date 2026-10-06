"""12i.4 on the page: past batches under Build questions, and a batch that
opens with every question it wrote (the hidden half too, to its author) and
where each went; the bank reader filtered by writer and batch; a served
model's answer length and ran-out count, beside its base's; a served model
with no result listed with Test and its page open; Test a model's estimate
from measured speed; and the Everyday total with its ran-out note on a line of
its own. The writer, checker and judge are the fake backend's; the served
model is tests/fake_openai.py."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

from conftest import set_name
from service import config, llm

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12i4"
SID = "served/LDA-phone"
IDLE = "served/Never-run"
HOW = "llama.cpp build, Q4_K_XL, --cpu-moe"


def shot(part, name, **kw):
    """a part's screenshot — taken again when the page's 5-second poll
    replaced the part mid-shot, as test_14_3_browser.steady_shot does (17b:
    the past batches' list failed one CI run "not attached to the DOM")"""
    from playwright.sync_api import Error
    SCREENS.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            part.screenshot(path=SCREENS / name, **kw)
            return
        except Error as e:
            if "not attached" not in str(e) or attempt == 3:
                raise


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def wait(live, draft_id, until, secs=40):
    t0 = time.time()
    while time.time() - t0 < secs:
        d = api(live, f"/api/builder/{draft_id}")
        if until(d):
            return d
        time.sleep(0.3)
    raise AssertionError(f"the batch stayed {d['status']}")


def publish(live, **spec):
    d = api(live, "/api/builder", {"count": 20, "dedup": False, "by": "masein", **spec})
    d = wait(live, d["id"], lambda d: d["status"] != "writing")
    for i, it in enumerate(it for it in d["items"] if not it["auto"]):
        api(live, f"/api/builder/{d['id']}/review",
            {"n": it["n"], "verdict": "reject" if i < 2 else "accept", "reason":
             "too easy" if i < 2 else "", "by": "masein"})
    api(live, f"/api/builder/{d['id']}/rest", {"by": "masein"})
    d = wait(live, d["id"], lambda d: d["status"] == "review")
    for it in d["items"]:
        if not it["auto"] and it["verdict"] is None and (it["flags"] or it["sample"]):
            api(live, f"/api/builder/{d['id']}/review",
                {"n": it["n"], "verdict": "accept", "by": "masein"})
    api(live, f"/api/builder/{d['id']}/publish", {"by": "masein"})
    return api(live, f"/api/builder/{d['id']}")


@pytest.fixture(scope="module")
def roles(live):
    saved = {k: getattr(config, k, None) for k in ("CHECKER_PROVIDER", "CHECKER_MODEL",
                                                    "JUDGE_MODEL", "OPENROUTER_API_KEY")}
    config.CHECKER_PROVIDER, config.CHECKER_MODEL = "fake", "fake-checker"
    config.JUDGE_MODEL, config.OPENROUTER_API_KEY = "stub", ""
    llm.reset()
    yield
    for k, v in saved.items():
        setattr(config, k, v)


@pytest.fixture(scope="module")
def batches(live, roles):
    return {"everyday": publish(live, kind="everyday", group="quick_maths"),
            "knowledge": publish(live, kind="knowledge", topic="Economics",
                                 level="general public", count=12)}


def test_past_batches_and_a_batch_with_every_question_it_wrote(live, page, batches):
    d = batches["everyday"]
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto(live["base"] + "/#tab=build")
    row = page.locator(f"[data-qb-past-row='{d['id']}']")
    row.wait_for()
    assert row.locator(f"[data-qb-past-line='{d['id']}']").inner_text().endswith(" published")
    assert "written by fake fake-exam" in row.inner_text() and "run by masein" in row.inner_text()
    shot(page.locator("[data-qb-past]"), "past-batches.png")
    row.locator("a").click()
    page.wait_for_selector("[data-qb-batch-qs]")
    assert page.evaluate("location.hash") == f"#tab=build&draft={d['id']}"
    assert page.locator("[data-qb-author-view]").inner_text() == \
        "Includes the hidden half — this batch’s author view."
    cards = page.locator("[data-qb-batch-q]")
    assert cards.count() == len(d["items"])
    hidden = [it for it in d["items"] if it["went"]["to"] == "hidden"]
    # which questions went to the hidden half is the publish's own split — a
    # hash of each question's words — and only the tried ten, the flagged and a
    # sample are reviewed. So no question is picked for its half (12f.4 took the
    # first hidden one; then a hidden one accepted, and about one draft in two
    # hundred has none): every card says its own review, whichever half it went
    # to, and every hidden one is here, in this author view
    said = {"accept": "A · accepted by masein", None: "not reviewed"}
    for it in d["items"]:
        if it.get("verdict") in said:
            assert page.locator(f"[data-qb-batch-q='{it['n']}'] [data-qb-review]").inner_text() \
                == said[it.get("verdict")], it["n"]
    for it in hidden:
        assert it["q"]["prompt"].split("\n")[0][:40] in \
            page.locator(f"[data-qb-batch-q='{it['n']}']").inner_text(), it["n"]
    # both kinds are there to see: eight of the tried ten accepted, and the
    # rest of the twenty published unreviewed but for a sample
    assert sum(it.get("verdict") == "accept" for it in d["items"]) >= 8
    assert any(it.get("verdict") is None and it["went"]["to"] != "not published"
               for it in d["items"])
    seen = next(it for it in d["items"] if it.get("verdict") == "accept")
    first = page.locator(f"[data-qb-batch-q='{seen['n']}']")
    assert seen["q"]["prompt"].split("\n")[0][:40] in first.inner_text()
    assert first.locator("[data-qb-checker-ok]").inner_text() in ("checker matched",
                                                                   "checker didn’t match")
    rejected = page.locator("[data-qb-review='reject']")
    assert rejected.count() == 2
    assert "not published: rejected" in rejected.first.locator("xpath=..").inner_text()
    # the list, by where each went
    page.locator("[data-select='where it went']").click()
    page.locator("[role=option][data-value='hidden']").click()
    assert page.locator("[data-qb-batch-q]").count() == len(hidden)
    shot(page, "batch-author-view.png", full_page=True)
    assert page.errors == []


def test_the_bank_reader_finds_a_batchs_practice_questions_and_no_hidden_one(live, page, batches):
    d = batches["knowledge"]
    practice = [it for it in d["items"] if it["went"]["to"] == "practice"]
    hidden = [it for it in d["items"] if it["went"]["to"] == "hidden"]
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + f"/#tab=build&draft={d['id']}")
    page.locator(f"[data-qb-bank-link='{d['id']}']").click()
    page.wait_for_selector("[data-bank-list]")
    assert page.locator("[data-bank-q]").count() == len(practice)
    assert page.locator(f"[data-bank-batch='{d['id']}']").count() == len(practice)
    body = page.locator("[data-bank-list]").inner_text()
    assert not [it for it in hidden if it["q"]["question"][:40] in body]
    # the written-by filter offers the writer
    page.locator("[data-select='written by']").click()
    opts = page.locator("[role=option]").all_inner_texts()
    assert "fake fake-exam (fake/fake-exam)" in opts
    page.keyboard.press("Escape")
    shot(page.locator("[data-bank-list]").locator("xpath=.."), "bank-reader-batch.png")
    assert page.errors == []
