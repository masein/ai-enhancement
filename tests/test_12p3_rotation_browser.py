"""12p.3 on the page: Build questions ▸ A new hidden set for Everyday — each
group's count, the two rules for the writer and the checker, the cost before
anything starts; the owner's Start and, once every group has its new
questions, the switch with the old questions worth retiring. Nothing is
started or switched here: each dialog is read and cancelled. Fixtures only."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import set_name
from service import config, rotation
from test_12p3_rotation import as_models as _as_models
from test_12p3_rotation import stage_all


def as_models(monkeypatch):
    """the models named as the page would; the fake checker answers, as the unit tests'"""
    monkeypatch.setattr(config, "CHECKER_PROVIDER", "fake", raising=False)
    monkeypatch.setattr(config, "CHECKER_MODEL", "fake-checker", raising=False)
    _as_models(monkeypatch)

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12p3"


def build(page, live, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=build")
    page.wait_for_selector("[data-rot]")


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name)


def test_the_plan_its_rules_and_its_cost_before_start(live, page, monkeypatch):
    as_models(monkeypatch)
    page.goto(live["base"] + "/")
    set_name(page, "masein")
    build(page, live)
    card = page.locator("[data-rot='plan']")
    n = len(rotation.targets())
    assert card.locator("[data-rot-group]").count() == n
    assert card.locator("[data-rot-rules]").get_attribute("data-rot-rules") == "ok"
    rules = card.locator("[data-rot-rule]").all_inner_texts()
    assert rules == ["✓ the writer is not a Qwen model — writer: GLM 5.3 (Z.ai)",
                     "✓ the checker is from another maker than the writer — checker: GPT-6 Luna "
                     "(OpenAI)"]
    est = card.locator("[data-rot-estimate]").inner_text()
    assert est.startswith("Cost: about $") and " questions written, " in est
    # Start says the cost again, and nothing starts until it is pressed
    card.locator("[data-rot-start]").click()
    dlg = page.locator("[data-dialog='rot-start']")
    dlg.wait_for()
    assert "Start writing a new hidden set: about $" in dlg.inner_text()
    shot(dlg.locator(".dlg"), "start-dialog.png")
    dlg.locator("[data-dialog-cancel]").click()
    shot(card, "plan-1400.png")
    assert page.errors == []


def test_someone_else_sees_the_counts_and_no_start(live, page, monkeypatch):
    as_models(monkeypatch)
    page.goto(live["base"] + "/")
    set_name(page, "sam")
    build(page, live)
    card = page.locator("[data-rot='plan']")
    assert card.locator("[data-rot-start]").count() == 0
    assert "The board’s owner writes and switches a hidden set." in card.inner_text()
    assert page.errors == []


def test_ready_to_switch_the_old_worth_retiring_listed_with_why(live, page, monkeypatch):
    as_models(monkeypatch)
    stage_all()
    cands = [{"id": "fx-hidden-honesty-01", "group": "honesty",
              "prompt": "when exactly will my fixture parcel number 1 arrive today",
              "why": ["every model passes it (2 of 2)"]}]
    page.route("**/api/everyday/rotation/candidates*",
               lambda r: r.fulfill(json={"candidates": cands, "warning": "…"}))
    try:
        page.goto(live["base"] + "/")
        set_name(page, "masein")
        build(page, live, width=1400)
        card = page.locator("[data-rot='ready']")
        card.locator("[data-rot-review]").click()
        row = card.locator("[data-rot-cand='fx-hidden-honesty-01']")
        row.wait_for()
        assert row.locator("input").is_checked()                      # ticked: retired
        assert "every model passes it (2 of 2)" in row.inner_text()
        card.locator("[data-rot-go]").click()
        dlg = page.locator("[data-dialog='rot-switch']")
        dlg.wait_for()
        text = dlg.inner_text()
        assert "moves to History, “scored on the retired hidden set”" in text
        assert "1 of the old set retired, the rest made practice." in text
        dlg.locator("[data-dialog-cancel]").click()
        shot(card, "ready-to-switch-1400.png")
        assert page.errors == []
    finally:
        rotation.staged_path().unlink(missing_ok=True)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_at_400px(live, page, monkeypatch, scheme):
    as_models(monkeypatch)
    page.emulate_media(color_scheme=scheme)
    page.goto(live["base"] + "/")
    set_name(page, "masein")
    build(page, live, width=400)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-rot]"), f"plan-400-{scheme}.png")
    assert page.errors == []
