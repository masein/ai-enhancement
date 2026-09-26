"""12i.2 on the page: Build questions, opened from the Knowledge exam's and
the Everyday page's questions — step 1's form (the count free, with its note
under 40; the instructions editable but for their locked output section),
Try 10 reviewed with A, E and R, Make the rest after five, the flagged ones
and a sample reviewed, Publish; and a draft that survives a reload. The
writer, the checker and the judge are the fake backend's."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import set_name
from service import builder, config, llm

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12i2"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


@pytest.fixture(autouse=True)
def roles(live, monkeypatch):
    monkeypatch.setattr(config, "CHECKER_PROVIDER", "fake", raising=False)
    monkeypatch.setattr(config, "CHECKER_MODEL", "fake-checker", raising=False)
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    llm.reset()
    yield


def build_page(page, base, kind="everyday", width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=home")
    set_name(page, "masein")
    page.evaluate(f"openBuilder('{kind}')")
    page.wait_for_selector(f"[data-qb-what='{kind}']")


def fill_count(page, n):
    box = page.locator("[data-qb-count]")
    box.fill(str(n))


def start(page, base, count=20, group="quick_maths", dedup=False):
    build_page(page, base, "everyday")
    page.locator("[data-qb-group]").select_option(group)
    fill_count(page, count)
    if not dedup:
        page.locator("[data-qb-dedup]").uncheck()
    page.locator("[data-qb-try]").click()
    page.wait_for_selector("[data-qb-item]", timeout=20000)


def test_build_questions_opens_from_both_banks(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=benchmarks&sub=exam")
    page.locator("[data-qb-open='knowledge']").click()
    page.wait_for_selector("[data-qb-what='knowledge']")
    assert page.evaluate("location.hash") == "#tab=build"
    assert page.locator("[data-qb-steps]").get_attribute("data-qb-steps") == "1"
    assert page.locator("[data-qb-steps] li").all_inner_texts() == [
        "1 What", "2 Try 10", "3 Make the rest, checked"]
    assert page.locator("[data-qb-topic] option").count() == 37
    subs = page.locator("[data-qb-subtopics]").input_value()
    assert subs.split("\n")[0] and len(subs.split("\n")) >= 2
    shot(page, "12i2-step1-knowledge-1400-light.png", full_page=True)
    page.goto(live["base"] + "/#tab=benchmarks&sub=everyday")
    page.locator("[data-qb-open='everyday']").click()
    page.wait_for_selector("[data-qb-what='everyday']")
    groups = page.locator("[data-qb-group] option").all_inner_texts()
    assert groups[0].startswith("Understanding · ") and groups[-1] == "New group…"
    page.locator("[data-qb-group]").select_option("new")
    page.wait_for_selector("[data-qb-new-label]")
    assert page.errors == []


def test_the_count_is_free_and_under_forty_says_so(live, page):
    build_page(page, live["base"], "knowledge")
    fill_count(page, 25)
    note = page.locator("[data-qb-under]")
    assert note.inner_text() == "fewer than 40 won’t give this topic its own score in Improve"
    fill_count(page, 40)
    assert note.count() == 0
    fill_count(page, 7)
    assert page.locator("[data-qb-try]").inner_text() == "Try 7"
    fill_count(page, 250)
    assert page.locator("[data-qb-try]").inner_text() == "Try 10" and note.count() == 0
    page.wait_for_function("document.querySelector('[data-qb-estimate] .mono').textContent "
                           "!== '…'")
    assert page.locator("[data-qb-estimate]").inner_text() == \
        "Estimated cost: free: every step runs on the local model"
    assert page.locator("[data-qb-no-checker]").count() == 0
    assert page.errors == []


def test_with_no_checker_step_one_says_where_to_choose_one(live, page, monkeypatch):
    monkeypatch.setattr(config, "CHECKER_PROVIDER", "", raising=False)
    llm.reset()
    build_page(page, live["base"], "everyday")
    assert page.locator("[data-qb-no-checker]").inner_text() == (
        "No checker yet: choose one on AI models, or above for this batch. Try 10 runs "
        "without it; Make the rest needs it.")
    page.locator("[data-qb-model='checker']").select_option("local")
    assert page.locator("[data-qb-no-checker]").count() == 0
    assert page.errors == []


def test_the_output_section_is_locked_and_reset_restores_the_default(live, page):
    build_page(page, live["base"], "knowledge")
    page.locator("[data-qb-prompt] summary").click()
    box, locked = page.locator("[data-qb-prompt-text]"), page.locator("[data-qb-locked]")
    default = builder.default_prompt("knowledge")
    assert box.input_value() == default["editable"]
    # the locked section is text on the page, not a field: nothing to type into
    assert locked.inner_text() == default["locked"]
    assert locked.evaluate("e => e.tagName === 'PRE' && !e.isContentEditable")
    assert "## Output" not in box.input_value()
    reset = page.locator("[data-qb-prompt-reset]")
    assert reset.is_disabled()
    box.fill(default["editable"].replace("a curious adult", "a curious teenager"))
    page.locator("[data-qb-count]").fill("12")        # a render: the edit stays
    assert "a curious teenager" in page.locator("[data-qb-prompt-text]").input_value()
    shot(page.locator("[data-qb-what]"), "12i2-step1-instructions-1400-light.png")
    page.locator("[data-qb-prompt-reset]").click()
    assert page.locator("[data-qb-prompt-text]").input_value() == default["editable"]
    assert page.locator("[data-qb-prompt-reset]").is_disabled()
    assert page.errors == []


def test_try_ten_reviews_with_keys_and_the_draft_survives_a_reload(live, page):
    start(page, live["base"])
    did = page.evaluate("state.qb.id")
    assert page.evaluate("location.hash") == f"#tab=build&draft={did}"
    rest = page.locator("[data-qb-rest]")
    assert rest.is_disabled() and rest.get_attribute("title") == \
        "review 5 more of the first ten first"
    shot(page, "12i2-try10-1400-light.png", full_page=True)
    # a reason, then R; then four A's
    page.locator("[data-qb-reason='too easy']").click()
    page.keyboard.press("r")
    page.wait_for_function("state.qb.draft.progress.reviewed_try === 1")
    for k in range(4):
        page.keyboard.press("a")
        page.wait_for_function(f"state.qb.draft.progress.reviewed_try === {k + 2}")
    d = builder.get(did)
    assert [it["verdict"] for it in d["items"][:5]] == ["reject"] + ["accept"] * 4
    assert d["items"][0]["reason"] == "too easy"
    # the draft is on the server: a reload lands where it was
    page.reload()
    page.wait_for_selector("[data-qb-item]")
    assert page.locator("[data-qb-reviewed]").get_attribute("data-qb-reviewed") == "5"
    assert page.locator("[data-qb-rest]").is_enabled()
    assert page.locator("[data-qb-steps]").get_attribute("data-qb-steps") == "2"
    # and it is listed among the drafts on a fresh step 1
    page.locator("[data-qb-new]").click()
    page.wait_for_selector(f"[data-qb-draft='{did}']")
    assert "Everyday · Quick maths" in page.locator(f"[data-qb-draft='{did}']").inner_text()
    assert page.errors == []


def test_make_the_rest_checks_everything_and_publish_makes_a_new_version(live, page, monkeypatch):
    import everyday as ev
    before = ev.version()["hash"]

    def checker(req):
        if req.custom_id.startswith("qbc:") and req.custom_id.endswith(":13"):
            return "about 4 i think"
        return llm.default_responder(req)
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(checker))
    start(page, live["base"], count=20)
    for k in range(10):
        page.keyboard.press("a")
        page.wait_for_function(f"state.qb.draft.progress.reviewed_try === {k + 1}")
    page.locator("[data-qb-rest]").click()
    page.wait_for_function("state.qb.draft.status === 'review' && state.qb.draft.stage === 'rest'",
                           timeout=30000)
    assert page.locator("[data-qb-steps]").get_attribute("data-qb-steps") == "3"
    assert page.locator("[data-qb-progress]").inner_text() == "20 of 20 written · 1 flagged"
    item = page.locator("[data-qb-item]")
    assert item.get_attribute("data-qb-item") == "13"          # the flagged one comes first
    assert item.locator("[data-qb-flag='checker']").inner_text().startswith(
        "the checker's answer failed: ")
    # each check in plain words, not the bank's JSON
    checks = item.locator("[data-qb-checks] li").all_inner_texts()
    assert checks and not any("{" in c for c in checks)
    publish = page.locator("[data-qb-publish]")
    assert publish.is_disabled() and publish.get_attribute("title") == "2 still to review"
    # the two waiting for review are not counted as ready
    assert publish.inner_text() == "Publish 18 questions"
    shot(page, "12i2-rest-flagged-1400-light.png", full_page=True)
    page.keyboard.press("r")                                      # the flagged one
    page.wait_for_function("state.qb.draft.progress.to_review === 1")
    page.keyboard.press("a")                                      # the sample
    page.wait_for_function("state.qb.draft.progress.to_review === 0")
    assert publish.inner_text() == "Publish 19 questions" and publish.is_enabled()
    publish.click()
    done = page.locator("[data-qb-published]")
    done.wait_for()
    after = ev.version()["hash"]
    assert after != before
    assert done.inner_text().startswith(f"Published 19 questions — Everyday, bank version {after}.")
    shot(page, "12i2-published-1400-light.png")
    assert page.errors == []


def test_a_near_duplicate_shows_side_by_side_and_keep_old_drops_it(live, page, monkeypatch):
    import everyday as ev
    bank = next(q for q in ev.load_bank() if len(q["prompt"].split()) > 16)

    def writer(req):
        if req.custom_id.startswith("qbw:") and req.meta.get("start") == 10:
            rows = [json.loads(x) for x in llm.default_responder(req).splitlines()]
            rows[0].update(prompt=bank["prompt"] + " thx", reference="3 pm.",
                           checks=[{"type": "contains_any", "values": ["3 pm", "3pm"]}])
            return "\n".join(json.dumps(x) for x in rows)
        return llm.default_responder(req)
    monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(writer))
    start(page, live["base"], count=20, dedup=True)
    for k in range(5):
        page.keyboard.press("a")
        page.wait_for_function(f"state.qb.draft.progress.reviewed_try === {k + 1}")
    page.locator("[data-qb-rest]").click()
    page.wait_for_function("state.qb.draft.status === 'review' && state.qb.draft.stage === 'rest'",
                           timeout=30000)
    box = page.locator("[data-qb-dup='11']")
    box.wait_for()
    assert box.locator("[data-qb-dup-other]").inner_text() == bank["prompt"]
    assert page.locator("[data-qb-item='11'] [data-qb-flag='dup']").inner_text() == \
        f"looks like {bank['id']}"
    assert box.locator("[data-qb-keep='new']").is_disabled()
    shot(page.locator("[data-qb-draft-open]"), "12i2-duplicate-1400-light.png")
    box.locator("[data-qb-keep='old']").click()
    page.wait_for_function("state.qb.draft.items.find(i => i.n === 11).verdict === 'reject'")
    assert page.errors == []


@pytest.mark.parametrize("width", [400])
def test_the_builder_at_phone_width(live, page, width):
    build_page(page, live["base"], "everyday", width)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12i2-step1-{width}-light.png", full_page=True)
    page.locator("[data-qb-group]").select_option("quick_maths")
    fill_count(page, 12)
    page.locator("[data-qb-dedup]").uncheck()
    page.locator("[data-qb-try]").click()
    page.wait_for_selector("[data-qb-item]", timeout=20000)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12i2-try10-{width}-light.png", full_page=True)
    assert page.errors == []
