"""The live check of 2026-09-22 in the browser (11a).

The More ▾ menu was clipped to the height of the tab strip, so almost nothing
in it could be clicked. It, Theme ▾ and the name menu are one popover
component now: on the body, placed from the button, and alive across a poll.
The Review tab says where the documents will go before anyone presses
Generate, and what happened to the ones that never arrived. And a rubric with
no version marker says "no version", not "v?".
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

from conftest import bar_reveal, label_domains, model_tab, set_name, open_submit


pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase11"
E2E_MS = 30000
WIDTHS = (1280, 400)
# key, the button that owns it, and whether its items are a menu. 12b: More ▾
# and Theme ▾ are gone; the run counter and the name menu are in the header
# at every width, and Menu ▾ holds the places below 720px
MENUS = (("runs", "#runs button[data-runs]", False), ("who", "#who button[data-who]", False))
PLACES_MENU = ("places", "#menuBtn", True)

# every focusable thing in the panel, and whether the point at its centre
# belongs to it — the clipping bug in one question
HIT_TEST = """key => {
  const p = document.querySelector(`[data-pop='${key}']`);
  if (!p) return null;
  const items = [...p.querySelectorAll('[role=menuitem],[role=menuitemradio],input,button,a')];
  return items.map(it => {
    const r = it.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return {text: (it.textContent || it.placeholder || '').trim().slice(0, 24),
            w: r.width, h: r.height,
            mine: !!hit && (hit === it || it.contains(hit))};
  });
}"""


def api(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def open_menu(page, key, sel):
    bar_reveal(page, sel)
    page.locator(sel).click()
    page.wait_for_selector(f"[data-pop='{key}']")
    return page.locator(f"[data-pop='{key}']")


def test_a_menu_is_never_clipped_and_every_item_can_be_clicked(live, page):
    """Finding 1. At both widths, every item in every menu is the thing under
    the point at its centre, and the panel stays 8 px inside the window."""
    base = live["base"]
    SCREENS.mkdir(parents=True, exist_ok=True)
    for width in WIDTHS:
        page.set_viewport_size({"width": width, "height": 820})
        page.goto(base + "/")
        page.wait_for_selector("#who button[data-who]")
        set_name(page, "Omar")
        for key, sel, _menu in MENUS + ((PLACES_MENU,) if width < 720 else ()):
            panel = open_menu(page, key, sel)
            assert page.locator(sel).get_attribute("aria-expanded") == "true"
            assert page.locator(sel).get_attribute("aria-controls") == f"pop-{key}"
            hits = page.evaluate(HIT_TEST, key)
            assert hits, f"{key} at {width}px has no items"
            assert all(h["mine"] for h in hits), (key, width, hits)
            assert all(h["h"] >= 20 for h in hits), (key, width, hits)
            box = panel.bounding_box()
            assert box["x"] >= 7 and box["y"] >= 7, (key, width, box)
            assert box["x"] + box["width"] <= width - 7, (key, width, box)
            assert box["y"] + box["height"] <= 820 - 7, (key, width, box)
            # an outside mousedown closes it — the stamp in the header is text
            page.locator("[data-stamp]").click()
            assert page.locator(f"[data-pop='{key}']").count() == 0
            assert page.locator(sel).get_attribute("aria-expanded") == "false"
        # one at a time: opening the next closes the last
        open_menu(page, "runs", "#runs button[data-runs]")
        open_menu(page, "who", "#who button[data-who]")
        assert page.locator("[data-pop]").count() == 1
        assert page.locator("#runs button[data-runs]").get_attribute("aria-expanded") == "false"
        page.keyboard.press("Escape")
    assert page.errors == []


def test_the_menu_keyboard_path(live, page):
    """The ARIA menu-button pattern, end to end: ↓ opens and focuses, ↑↓ wrap,
    Home and End, Esc closes and gives the button back, Tab closes. 12b: the
    one menu of that kind is Menu ▾, the places below 720px."""
    key, sel, _menu = PLACES_MENU
    base = live["base"]
    page.set_viewport_size({"width": 400, "height": 820})
    page.goto(base + "/")
    # the button is in the page's HTML; its menu is wired once the board has
    # loaded — a key pressed before then opened nothing, now and then
    page.wait_for_selector(sel + "[aria-haspopup]")
    page.locator(sel).focus()
    page.keyboard.press("ArrowDown")
    page.wait_for_selector(f"[data-pop='{key}']")
    items = page.locator(f"[data-pop='{key}'] [role=menuitem], "
                         f"[data-pop='{key}'] [role=menuitemradio]")
    n = items.count()
    assert n >= 3
    texts = items.all_text_contents()
    here = lambda: page.evaluate("document.activeElement.textContent.trim()")   # noqa: E731
    assert here() == texts[0].strip()
    page.keyboard.press("ArrowDown")
    assert here() == texts[1].strip()
    page.keyboard.press("ArrowUp")
    page.keyboard.press("ArrowUp")
    assert here() == texts[-1].strip()                     # ↑ from the first wraps
    page.keyboard.press("Home")
    assert here() == texts[0].strip()
    page.keyboard.press("End")
    assert here() == texts[-1].strip()
    page.keyboard.press("Escape")
    assert page.locator(f"[data-pop='{key}']").count() == 0
    assert page.evaluate("document.activeElement.id") == sel.lstrip("#")
    # Enter opens it too, and Tab closes it
    page.keyboard.press("Enter")
    page.wait_for_selector(f"[data-pop='{key}']")
    page.keyboard.press("Tab")
    assert page.locator(f"[data-pop='{key}']").count() == 0
    assert page.errors == []


def test_the_name_menu_opens_from_the_keyboard_and_saves(live, page):
    base = live["base"]
    page.goto(base + "/")
    page.wait_for_selector("#who button[data-who]")
    set_name(page, "Omar")
    page.locator("#who button[data-who]").focus()
    page.keyboard.press("ArrowDown")
    page.wait_for_selector("[data-pop='who'] input")
    assert page.evaluate("document.activeElement.dataset.whoInput") == "1"
    page.keyboard.press("Escape")
    assert page.locator("[data-pop='who']").count() == 0
    assert page.evaluate("document.activeElement.dataset.who") == "Omar"
    # and it saves from inside the popover
    page.locator("#who button[data-who]").click()
    page.locator("[data-pop='who'] input").fill("Omar Affifi")
    page.locator("[data-pop='who'] [data-who-save]").click()
    page.wait_for_selector("#who button[data-who='Omar Affifi']")
    assert page.locator("[data-pop='who']").count() == 0
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("#who button[data-who]").click()
    page.wait_for_selector("[data-pop='who']")
    page.screenshot(path=SCREENS / "11a-name-menu-1280-light.png")
    assert page.errors == []


def test_two_polls_do_not_close_an_open_menu(live, page):
    """The board redraws every five seconds. An open menu, the item under the
    keyboard, and the page's scroll are all still there afterwards."""
    base = live["base"]
    page.set_viewport_size({"width": 700, "height": 700})         # Menu ▾ holds the places
    page.goto(base + "/#tab=improve")               # 12g.1: the pipeline, where the board was
    page.wait_for_selector("[data-stages]")
    page.locator("#menuBtn").focus()
    page.keyboard.press("ArrowDown")
    page.wait_for_selector("[data-pop='places']")
    page.keyboard.press("ArrowDown")
    was = page.evaluate("document.activeElement.textContent.trim()")
    page.evaluate("window.scrollTo(0, 120)")
    page.wait_for_timeout(11000)                    # two polls, and two renders
    assert page.locator("[data-pop='places']").count() == 1
    assert page.evaluate("document.activeElement.textContent.trim()") == was
    assert page.locator("#menuBtn").get_attribute("aria-expanded") == "true"
    assert page.evaluate("Math.round(window.scrollY)") == 120
    # the panel is still anchored to the button it belongs to
    box = page.locator("[data-pop='places']").bounding_box()
    btn = page.locator("#menuBtn").bounding_box()
    assert abs(box["y"] - (btn["y"] + btn["height"] + 4)) < 2
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2 and 3: where the documents go, and what happened to the missing ones
# ---------------------------------------------------------------------------

SHORT = "A short note. " * 20               # 60 words: under DOC_MIN_WORDS


def _planting_responder():
    from service import llm

    def responder(req):
        if req.custom_id.startswith("gen:") and req.custom_id.rsplit(":", 1)[1] == "1":
            return json.dumps([{"title": "Too brief", "text": SHORT},
                               {"title": "Also brief", "text": SHORT}])
        return llm.default_responder(req)
    return responder


def test_the_review_card_says_where_the_documents_go_and_what_went_missing(
        live, page, monkeypatch):
    base, root = live["base"], live["root"]
    from service import llm
    path, was = label_domains(root / "exam", "Economics")
    try:
        monkeypatch.setattr(llm.FakeBatches, "responder", staticmethod(_planting_responder()))
        pid = api(base, "/api/proposals", {"model": "fx/good-750m", "topic": "Economics",
                                           "requested_by": "Omar"})["id"]
        for _ in range(100):
            if api(base, f"/api/proposals/{pid}")["status"] == "proposed":
                break
            page.wait_for_timeout(200)
        api(base, f"/api/proposals/{pid}/approve", {"approver": "Omar"})
        # 11e: Approve froze the plan; the count in the box, which starts at
        # twenty, takes its first N labels
        frozen = api(base, f"/api/proposals/{pid}/focus?count=20")
        assert frozen["frozen"] and frozen["mode"] == "area"
        plan20 = Counter(frozen["labels"][:20])
        plan = Counter(frozen["labels"][:12])
        assert len(plan) >= 2

        page.set_viewport_size({"width": 1280, "height": 900})
        # 11j: the name first — the card opens in the sheet, over the header
        page.goto(base + "/#tab=review")
        page.wait_for_selector("[data-stages]")      # 12g.1: the pipeline
        set_name(page, "Omar")
        page.goto(base + f"/#tab=review&read=proposal:{pid}")
        page.wait_for_selector("#reader[data-ready='1']", timeout=E2E_MS)
        line = page.locator(f"[data-focus-plan='{pid}'][data-plan-stage='generate']")
        line.wait_for(timeout=E2E_MS)
        # the plan, as chips, before anyone presses Generate
        page.wait_for_selector(f"[data-focus-plan='{pid}'] .chip-static", timeout=E2E_MS)
        text = line.text_content()
        for d, n in plan20.items():
            assert f"{d} {n}" in text
        assert not re.search(r"\bfail", text)
        SCREENS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=SCREENS / "11a-review-focus-plan-1280-light.png", full_page=True)

        # generate: the toast, when the batch finishes, says what came back
        # a different count re-asks, and the chips follow it
        page.locator("#reader input[type=number]").fill("12")
        for d, n in plan.items():
            page.wait_for_function(
                "want => document.querySelector(`[data-focus-plan='%s']`)"
                ".textContent.includes(want)" % pid,
                arg=f"{d} {n}", timeout=E2E_MS)
        page.locator(f"[data-generate='{pid}']").click()
        page.wait_for_selector("[data-toast^='dataset-']", timeout=E2E_MS)
        toast = page.locator("[data-toast^='dataset-']").first.text_content()
        assert "10 of 12 documents · 2 missing — 2 too short (60 words)" in toast
        # and so does its line in the pipeline's Training data (12g.1) — with
        # the reasons in the reader
        page.goto(base + "/#tab=improve&sub=model&model=fx%2Fgood-750m")
        row = page.locator("[data-ds-item]").first
        row.wait_for(timeout=E2E_MS)
        did = row.get_attribute("data-ds-item")
        assert page.locator(f"[data-doc-line='{did}']").text_content() == "10 of 12 · 2 missing"
        row.locator("[data-ds-read]").click()
        page.wait_for_selector("#reader[data-ready='1']", timeout=E2E_MS)
        gone = page.locator("#reader [data-missing-doc]")
        assert gone.count() == 2
        rows = gone.all_text_contents()
        assert all("too short (60 words)" in r for r in rows)
        # each names the area it was meant to cover, so the gap in the plan shows
        assert all(any(d in r for d in plan) for r in rows), rows
        page.screenshot(path=SCREENS / "11a-dataset-missing-1280-light.png", full_page=True)
        assert page.errors == []
    finally:
        path.write_bytes(was)


def test_a_dataset_from_before_this_pr_says_the_reasons_were_not_recorded(live, page):
    """Nothing blank: a dataset generated before the accounting existed says
    that its reasons were not recorded."""
    base = live["base"]
    old = [{"id": 99, "proposal_id": 1, "status": "ready", "fmt": "doc", "count": 20,
            "kept": 18, "requester": "Omar", "error": "", "model": "fx/good-750m",
            "task": "exam_economics", "category": "Economics", "download": None,
            "over_provisional_judge": None,
            "provenance": {"count_requested": 20,
                           "items": {"generated": 18, "dropped": 0, "kept": 18}}}]
    page.route("**/api/datasets", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(old)))
    # 12g.1: the dataset is in its model's pipeline, under Training data
    page.goto(base + "/#tab=improve&sub=model&model=fx%2Fgood-750m")
    page.wait_for_selector("[data-ds-item='99']")
    # 11j: one line on the row; "reasons not recorded (made before 11a)" is
    # in the reader, where the documents are (11g)
    assert page.locator("[data-doc-line='99']").text_content() == "18 of 20 · 2 missing"
    assert page.errors == []


def test_a_rubric_with_no_version_says_so(live, page):
    """Finding 5: the judged header printed "v?" for every author-written
    rubric, because none of them carries a version marker."""
    base = live["base"]
    page.goto(base + "/#model=fx%2Fgood-750m")
    # 12b.2: the judge's line, rubrics and all, is History's How it was graded
    model_tab(page, "history")
    page.wait_for_selector("[data-model-graded]")
    text = page.locator("#view").text_content()
    assert "rubrics no version" in text
    assert "v?" not in text
    assert page.errors == []


# ---------------------------------------------------------------------------
# 4: the search offers what this server can run
# ---------------------------------------------------------------------------

NO_WEIGHTS = "results only — weights not on the server"
DELTA = "local/qwen35-delta-moe-7d560104-step945"
SUGGEST = {"items": [
    {"id": "local/uploaded-step100", "params": 7.6e8, "kind": "base", "on_board": True,
     "judged": 0, "weights": True, "source": "board"},
    {"id": DELTA, "params": 7e9, "kind": "base", "on_board": True, "judged": 8,
     "weights": False, "source": "board"}], "hub_ok": True, "footer": ""}


def test_a_model_whose_weights_are_not_here_is_greyed_and_cannot_be_queued(live, page):
    """Finding 4 on the page: it is listed, because it is on the board, and
    it says why it cannot run — the arrows skip it, and picking it anyway
    turns the submit off with the same sentence the API would answer with."""
    base = live["base"]
    page.route("**/api/models/suggest*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(SUGGEST)))
    open_submit(page, base)
    box = page.locator('[data-ms="submit"] input')
    box.wait_for()
    box.type("qwen35")
    page.wait_for_selector(f'.ms-item[data-ms-item="{DELTA}"]')
    grey = page.locator(f'.ms-item[data-ms-item="{DELTA}"]')
    assert grey.get_attribute("aria-disabled") == "true"
    assert "noweights" in grey.get_attribute("class")
    assert NO_WEIGHTS in grey.text_content()
    ok = page.locator('.ms-item[data-ms-item="local/uploaded-step100"]')
    assert ok.get_attribute("aria-disabled") is None and NO_WEIGHTS not in ok.text_content()
    # the arrows land on the one that can run, and stop there
    box.press("ArrowDown")
    assert page.locator(".ms-item.active").get_attribute("data-ms-item") == "local/uploaded-step100"
    box.press("ArrowDown")
    assert page.locator(".ms-item.active").get_attribute("data-ms-item") == "local/uploaded-step100"
    # picked anyway, with the mouse (aria-disabled is advice, not a lock):
    # the form says so and refuses to queue it
    grey.click(force=True)
    page.wait_for_selector('[data-why="weights"]')
    why = page.locator('[data-why="weights"]').text_content()
    assert why == (f"{DELTA} has results on this board but no weights on this server — upload "
                   f"it first (POST /api/artifacts/qwen35-delta-moe-7d560104-step945) to run "
                   f"it here.")
    assert box.input_value() == DELTA
    submit = page.get_by_role("button", name="Submit model")
    assert submit.is_disabled()
    assert submit.get_attribute("title") == why
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SCREENS / "11a-search-no-weights-1280-light.png")
    assert page.errors == []
