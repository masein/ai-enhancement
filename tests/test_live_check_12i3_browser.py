"""12i.3 on the page: the live check of 12i. AI models from the name menu and
from every "AI: … · change"; the model picker with OpenRouter's whole list —
rows that don't run into each other, the suggested row's reason inside it,
opening by its row, and an order that means something; "Agreement (0–1)";
and Build questions loading with no draft and no checker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import fake_openrouter as fo
from conftest import set_name
from fake_openrouter import FakeOpenRouter
from service import config, llm

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12i3"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


@pytest.fixture
def many(monkeypatch):
    """OpenRouter's list as long as it is: eighty more models, prices all over"""
    extra = [fo._m(f"acme/model-{i}", f"acme/model-{i}-20260101",
                   f"Acme: Model {i} Instruct With A Long Name",
                   f"0.00000{(i * 7) % 9 + 1}", f"0.0000{(i * 3) % 7 + 1}") for i in range(80)]
    monkeypatch.setattr(fo, "MODELS", list(fo.MODELS) + extra)
    fake = FakeOpenRouter.install(monkeypatch)
    (config.BENCH_ROOT / "ai" / "openrouter_models.json").unlink(missing_ok=True)
    return fake


def ai_page(page, base, width=1512, height=900):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(base + "/#tab=ai")
    page.wait_for_selector("[data-ai-jobs]")


def test_ai_models_is_in_the_name_menu(live, page):
    page.set_viewport_size({"width": 1512, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.locator("#who button.who").click()
    item = page.locator("#pop-who [data-menu='ai']")
    assert item.inner_text() == "AI models" and item.is_visible()
    shot(page, "12i3-name-menu-1512-light.png")
    item.click()
    page.wait_for_selector("[data-ai-page]")
    assert page.evaluate("location.hash") == "#tab=ai"
    assert page.errors == []


def test_improve_links_to_ai_models_even_with_no_writer_set_up(live, page, monkeypatch):
    """the line was there only when the training-data writer ran; a job with
    no model is when AI models is needed"""
    monkeypatch.setattr(config, "LLM_PROVIDER", "")
    llm.reset()
    page.set_viewport_size({"width": 1512, "height": 900})
    page.goto(live["base"] + "/#tab=improve")
    line = page.locator("[data-imp-ai]")
    line.wait_for()
    assert line.inner_text().startswith("AI: judge ") and line.inner_text().endswith(" · change")
    line.locator("[data-ai-change]").click()
    page.wait_for_selector("[data-ai-page]")
    # and from the Knowledge exam
    page.goto(live["base"] + "/#tab=benchmarks&sub=exam")
    ex = page.locator("[data-exam-ai]")
    ex.wait_for()
    ex.locator("[data-ai-change]").click()
    page.wait_for_selector("[data-ai-page]")
    assert page.errors == []


def rects(page, sel):
    return page.evaluate(f"""[...document.querySelectorAll({json.dumps(sel)})]
      .map(e => {{ const r = e.getBoundingClientRect(); return [r.top, r.bottom]; }})""")


def test_the_picker_rows_keep_their_room_and_it_opens_by_its_row(live, page, many):
    ai_page(page, live["base"])
    page.wait_for_function("state.ai.models && state.ai.models.length > 80")
    # the last row: the one the panel used to flip away from, to the page's top
    btn = page.locator("[data-ai-change-menu='checker']")
    btn.click()
    panel = page.locator("#pop-ai-checker")
    panel.wait_for()
    b, p = btn.bounding_box(), panel.bounding_box()
    assert p["y"] >= b["y"] + b["height"] and p["y"] - (b["y"] + b["height"]) <= 12, (b, p)
    # every row keeps its height: none starts before the one above it ends
    rows = rects(page, "#pop-ai-checker [data-ai-pick]")
    assert all(nxt[0] >= prev[1] - 0.5 for prev, nxt in zip(rows, rows[1:])), rows
    # the suggested row holds its reason, under its price
    sug = page.locator("#pop-ai-checker [data-ai-pick='openai/gpt-6-luna']")
    assert sug.locator("[data-ai-suggested]").inner_text() == "suggested"
    why = sug.locator("[data-ai-why='checker']")
    price = sug.locator(".aiitem-sub")
    assert why.bounding_box()["y"] >= price.bounding_box()["y"] + price.bounding_box()["height"] - 0.5
    shot(page, "12i3-picker-checker-1512-light.png")
    assert page.errors == []


def test_the_picker_lists_suggested_first_then_by_price(live, page, many):
    ai_page(page, live["base"])
    page.wait_for_function("state.ai.models && state.ai.models.length > 80")
    page.locator("[data-ai-change-menu='judge']").click()
    panel = page.locator("#pop-ai-judge")
    panel.wait_for()
    picks = [b.get_attribute("data-ai-pick") for b in panel.locator("[data-ai-pick]").all()]
    # Local, this job's suggested model, the other jobs' suggested ones
    assert picks[:4] == ["local", "deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3",
                         "openai/gpt-6-luna"]
    assert panel.locator("[data-ai-also]").count() == 2
    prices = page.evaluate("""() => {
      const by = Object.fromEntries(state.ai.models.map(m => [m.id, m.price_in + m.price_out]));
      return [...document.querySelectorAll('#pop-ai-judge [data-ai-pick]')].slice(4)
        .map(b => by[b.dataset.aiPick]); }""")
    assert prices == sorted(prices) and len(prices) == 60
    panel.locator("[data-ai-sort-dear]").click()
    prices = page.evaluate("""() => {
      const by = Object.fromEntries(state.ai.models.map(m => [m.id, m.price_in + m.price_out]));
      return [...document.querySelectorAll('#pop-ai-judge [data-ai-pick]')].slice(4)
        .map(b => by[b.dataset.aiPick]); }""")
    assert prices == sorted(prices, reverse=True)
    assert page.errors == []


def test_the_local_model_and_agreement_in_words(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    rows = [{"key": "cur", "name": "Local (gemma on this server)", "current": True,
             "id": "local/chat", "n": 110, "exact": 0.52, "within1": 0.9, "kappa": 0.64,
             "per_1000": None}]
    page.route("**/api/judge-test/result", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"rows": rows, "person": {"total": 110, "marked": 110, "skipped": 0,
                                                  "next": None},
                         "kappa_min": 0.7, "n_min": 100})))
    ai_page(page, live["base"])
    table = page.locator("[data-jt-result]")
    table.wait_for()
    head = table.locator("thead th").nth(3)
    assert head.text_content() == "Agreement (0–1)"
    assert head.get_attribute("title").startswith("weighted kappa (quadratic) against your marks")
    assert table.locator("tr[data-jt-row='cur'] td").first.inner_text().startswith(
        "Local (gemma on this server)")
    page.locator("[data-ai-change-menu='writer']").click()
    assert page.locator("#pop-ai-writer [data-ai-pick='local'] .aiitem-name").inner_text() \
        .startswith("Local (") and "local the local" not in page.locator("#pop-ai-writer").inner_text()
    assert page.errors == []


def test_build_questions_loads_with_no_draft_and_no_checker(live, page):
    page.set_viewport_size({"width": 1512, "height": 900})
    page.goto(live["base"] + "/#tab=benchmarks&sub=exam")
    page.locator("[data-qb-open='knowledge']").click()
    page.wait_for_selector("[data-qb-what='knowledge']")
    assert page.locator("[data-qb-drafts]").count() == 0
    assert page.locator("[data-qb-no-checker]").is_visible()
    assert "HTTP 500" not in page.locator("#view").inner_text()
    assert page.errors == []
