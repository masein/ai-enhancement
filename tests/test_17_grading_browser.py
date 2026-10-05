"""17, stage 3 on the page: AI models' grading card names each Frontier
grader with its pinned model and its prompt's sha256, shows the dry run —
what waits for each grader, and what Start would cost — and sends nothing
until Start; the Frontier view says beside a graded number who graded it and
with which prompt. At 1400 and 375 px. Answers and their questions invented on
disk; nothing calls OpenRouter."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import frontier as fb
import frontier_graders as fg
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase17"
ROW = "served__grading-box"
SERVED = "served/grading-box"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """SimpleQA answers waiting for their grader, MATH answers the code marks
    wrong, and OpenRouter's list as AI models last fetched it"""
    import service.app as appmod
    from service import ai_models, config, db
    from service import frontier as sf
    saved = config.OPENROUTER_API_KEY
    config.OPENROUTER_API_KEY = "test-key"
    for task, items in (("simpleqa_epoch", [{"id": str(k), "question": f"Fact {k}?",
                                             "answer": f"A{k}", "subject": ""} for k in range(4)]),
                        ("math_l5_epoch", [{"id": f"algebra/{k}", "question": f"P{k}.",
                                            "answer": "2", "subject": "Algebra"}
                                           for k in range(3)])):
        p = fb._cache(config.BENCH_ROOT, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items))
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": "openai/gpt-4.1", "name": "OpenAI: GPT-4.1", "version": "openai/gpt-4.1-2025-04-14",
         "price_in": 2.0, "price_out": 8.0},
        {"id": "google/gemini-2.5-flash", "name": "Google: Gemini 2.5 Flash",
         "version": "google/gemini-2.5-flash-20250617", "price_in": 0.3, "price_out": 2.5}]}))
    db.served_put({"id": SERVED, "name": "grading box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "m.gguf"}, "by": "masein", "at": 0})
    row = Path(config.OUT_DIR) / ROW
    row.mkdir(parents=True, exist_ok=True)
    (row / "model_meta.json").write_text(json.dumps({"model": SERVED}))
    for task, answers in (("simpleqa_epoch", [(str(k), f"A{k}") for k in range(4)]),
                          ("math_l5_epoch", [("algebra/0", "ANSWER: 2"), ("algebra/1", "ANSWER: 3"),
                                             ("algebra/2", "ANSWER: 2.0")])):
        d = sf.task_dir(row, task)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": q, "epoch": 0, "answer": a, "finish": "stop"}) + "\n" for q, a in answers))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    config.OPENROUTER_API_KEY = saved


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


@pytest.mark.parametrize("width", [1400, 375])
def test_the_grading_card_names_each_grader_and_prices_start(live, page, width):
    from service import db
    go(page, live, "tab=ai", "[data-frontier-grading]:not([data-frontier-grading='loading'])",
       width)
    card = page.locator("[data-frontier-grading]")
    sqa = card.locator("[data-frontier-grader='simpleqa']")
    assert "SimpleQA Verified's grader" in sqa.inner_text()
    assert "the owners’: gpt-4.1-2025-04-14" in sqa.inner_text()
    prompt = card.locator("[data-frontier-grader-prompt='simpleqa']").inner_text()
    assert "Google's grader prompt" in prompt and fg.prompt_sha("simpleqa")[:12] in prompt
    # the dry run: four SimpleQA answers, and MATH's one wrong by code (2.0 is
    # the key by code, with math-verify or without: right, never asked)
    est = page.evaluate("state.ai.frg.estimate")
    assert {(r["slot"], r["answers"]) for r in est["rows"]} == {("simpleqa", 4), ("math", 1)}
    cost = page.evaluate("usd(state.ai.frg.estimate.usd)")       # the page's own money format
    assert cost.startswith("$0.01")
    total = card.locator("[data-frontier-total]").inner_text()
    assert total.startswith(f"Start: about {cost} for 5 answers · this month $")
    start = card.locator("[data-frontier-start]")
    assert start.is_enabled() and start.inner_text() == f"Start grading: about {cost}"
    # nothing has been sent
    assert not db.batches_pending()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(card, SCREENS / f"ai-models-grading-{width}.png")
    assert page.errors == []
