"""12i.0 #9–11, from the runs of 2026-09-26: the Test a model dialog says
"Start test" and one plain line; an Everyday total over fewer questions than
the whole current set is never shown as if beside a full one; and the
Everyday page says when its questions were updated, the hash on hover."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12i0b"
PARTIAL = "fx/skewed-360m"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def served(page, edit):
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def partial(body):
    """one model answered only 169 of the hidden 200: its 31 are new"""
    e = body["everyday"]["models"][PARTIAL]
    e.update(passed=84, total=169, unasked=55)


def test_the_test_dialog_says_start_test_in_one_plain_line(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    page.locator("[data-test-model]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    assert dlg.locator("[data-suite-help]").inner_text() == \
        "Pick a model and what to test. One test runs at a time; results appear on Models."
    assert dlg.get_by_role("button", name="Start test").count() == 1
    assert dlg.get_by_role("button", name="Submit model").count() == 0
    text = dlg.inner_text()
    for word in ("Preflight", "separate runs", "local/<", "full and judged", "Hub"):
        assert word not in text, word
    assert dlg.get_by_label("model id").get_attribute("placeholder") == \
        "search Hugging Face or uploads"
    shot(dlg, "12i0b-test-a-model-1400-light.png")
    assert page.errors == []


def test_a_partial_everyday_total_is_greyed_with_what_is_not_asked_and_its_run(live, page):
    served(page, partial)
    set_name_page(page, live["base"])
    page.goto(live["base"] + "/#tab=benchmarks&sub=everyday")
    head = page.locator(f"[data-evd-count='{PARTIAL}']")
    head.wait_for()
    part = head.locator("[data-evd-partial]")
    assert part.get_attribute("data-evd-partial") == "55"
    # in the column head, the note is a line of its own under the count
    count = part.locator("[data-everyday-count]")
    assert count.inner_text() == "84 of 169"
    assert " ".join(part.locator(".evmissing").inner_text().split()) == "55 not asked yet · Run"
    assert "se" in count.get_attribute("class")
    assert count.get_attribute("title") == ("over the 169 hidden questions it has answered, not "
                                            "all 200 — not comparable with a full count")
    # its group cells are greyed with it
    assert page.locator("[data-evd-g='understanding'] td.evpart").count() == 1
    # the full one beside it is as it was
    full = page.locator("[data-evd-count='fx/good-750m']")
    assert full.locator("[data-evd-partial]").count() == 0
    assert full.inner_text().endswith(" of 200")
    shot(page.locator("[data-everyday-results]"), "12i0b-everyday-partial-1400-light.png")
    # Run ticks that model alone
    part.locator("[data-evd-run]").click()
    dlg = page.locator("[data-dialog='everyday']")
    dlg.wait_for()
    ticked = page.evaluate("""() => [...document.querySelectorAll(
      "[data-dialog='everyday'] [data-evd-pick]")].filter(r => r.querySelector('input').checked)
      .map(r => r.dataset.evdPick)""")
    assert ticked == [PARTIAL]
    assert dlg.locator("[data-dialog-go]").inner_text() == "Queue 1 run"
    page.keyboard.press("Escape")
    # and on Models ▸ Everyday tasks, and the model page
    page.goto(live["base"] + "/#tab=models&view=everyday")
    row = page.locator(f"tr[data-lb-row='{PARTIAL}'] [data-evd-partial='55']")
    row.wait_for()
    assert " ".join(row.inner_text().split()) == "84 of 169 · 55 not asked yet · Run"
    page.goto(live["base"] + "/#model=" + PARTIAL.replace("/", "%2F"))
    block = page.locator(f"[data-everyday-block='{PARTIAL}'] [data-evd-partial='55']")
    block.wait_for()
    assert block.locator(".evcount").inner_text() == "84 of 169"
    assert page.errors == []


def test_home_ranks_a_full_count_above_a_partial_one(live, page):
    """84 of 169 is more passes than 83 of 200, and is still not the best"""
    def edit(body):
        partial(body)
        body["everyday"]["models"]["fx/good-750m"].update(passed=83, total=200, unasked=0)
    served(page, edit)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/")
    card = page.locator("[data-best='everyday']")
    card.wait_for()
    assert card.locator("[data-best-value='everyday']").inner_text() == "83 of 200"
    assert page.errors == []


def test_the_everyday_page_says_when_its_questions_were_updated(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=benchmarks&sub=everyday")
    line = page.locator("[data-everyday-head] [data-evd-version]")
    line.wait_for()
    assert line.inner_text() == "Questions updated 25 Sep"
    tip = line.get_attribute("title")
    assert tip.startswith(f"version {line.get_attribute('data-evd-version')}: the wording and "
                          "the split.")
    assert "History" in tip
    assert page.errors == []


def set_name_page(page, base):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(base + "/#tab=home")
    set_name(page, "masein")
