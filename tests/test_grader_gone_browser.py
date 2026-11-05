"""A gone grader on the page (masein, 5 Nov): AI models' grading card with
HLE's judge gone (o3-mini, shut down by OpenAI on 23 Oct). Its dry-run row
says it isn't sent and that its answers wait as they are; Start's price is
the other graders' alone. At 1400 and 375 px, light and dark. Answers and questions invented on disk;
nothing calls OpenRouter."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

import frontier as fb
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "grader-gone"
ROW = "served__judge-gone-box"
SERVED = "served/judge-gone-box"
O3 = "openai/o3-mini"
GONE = "openai/o3-mini is no longer on OpenRouter's list — choose the job's model again on AI models"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """four SimpleQA answers and four HLE answers waiting on one row, HLE's
    judge chosen as o3-mini; everything taken away after"""
    import service.app as appmod
    from service import ai_models, config, db
    from service import frontier as sf
    saved = config.OPENROUTER_API_KEY
    config.OPENROUTER_API_KEY = "test-key"
    made = []
    for task, items in (("simpleqa_epoch", [{"id": str(k), "question": f"Fact {k}?",
                                             "answer": f"A{k}", "subject": ""} for k in range(4)]),
                        ("hle_text_cais", [{"id": f"h{k}", "question": f"Hard {k}?",
                                            "answer": f"K{k}", "answer_type": "exactMatch",
                                            "subject": "x"} for k in range(4)])):
        p = fb._cache(config.BENCH_ROOT, task)
        if not p.exists():                  # another module's questions stay as they are
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(items))
            made.append(p)
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    listed = p.read_text() if p.exists() else None
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": "openai/gpt-4.1", "name": "OpenAI: GPT-4.1", "version": "openai/gpt-4.1-2025-04-14",
         "price_in": 2.0, "price_out": 8.0, "reasons": False}]}))
    db.served_put({"id": SERVED, "name": "judge gone box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "m.gguf"}, "by": "masein", "at": 0})
    row = Path(config.OUT_DIR) / ROW
    row.mkdir(parents=True, exist_ok=True)
    (row / "model_meta.json").write_text(json.dumps({"model": SERVED}))
    for task, answers in (("simpleqa_epoch", [(str(k), f"A{k}") for k in range(4)]),
                          ("hle_text_cais", [(f"h{k}", f"I believe K{k}.") for k in range(4)])):
        d = sf.task_dir(row, task)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": q, "epoch": 0, "answer": a, "finish": "stop"}) + "\n" for q, a in answers))
    db.ai_set("grader:hle", {"kind": "openrouter", "id": O3, "version": "openai/o3-mini-2025-01-31",
                             "name": "OpenAI: o3 Mini", "provider": "openai",
                             "provider_name": "OpenAI", "precision": "unknown",
                             "price_in": 1.1, "price_out": 4.4}, "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    db.ai_set("grader:hle", None, "masein")
    shutil.rmtree(row, ignore_errors=True)
    for p in made:
        p.unlink(missing_ok=True)
    if listed is None:
        ai_models._cache_path().unlink(missing_ok=True)
    else:
        ai_models._cache_path().write_text(listed)
    appmod._cache.update(key=None, payload=None, at=0.0)
    config.OPENROUTER_API_KEY = saved


@pytest.mark.parametrize("width,scheme", [(1400, "light"), (1400, "dark"), (375, "light"),
                                          (375, "dark")])
def test_the_card_says_the_gone_judges_answers_wait_and_prices_the_others(live, page, monkeypatch,
                                                                         width, scheme):
    from service import ai_models, db
    gone = lambda pin: GONE if (pin or {}).get("id") == O3 else ""  # noqa: E731
    monkeypatch.setattr(ai_models, "drifted", gone)
    monkeypatch.setattr(ai_models, "provider_gone", gone)
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=ai")
    page.wait_for_selector("[data-frontier-grading]:not([data-frontier-grading='loading'])")
    card = page.locator("[data-frontier-grading]")
    assert card.locator("[data-frontier-grader-gone='hle']").inner_text() == GONE
    # the dry run: HLE's four shown, not sent; SimpleQA's four priced
    est = page.evaluate("state.ai.frg.estimate")
    mine = [r for r in est["rows"] if r["model"] == SERVED]
    assert sorted((r["slot"], r["answers"], r["usd"] is None) for r in mine) == [
        ("hle", 4, True), ("simpleqa", 4, False)]
    row = card.locator(f"[data-frontier-est='hle|{SERVED}']")
    assert row.locator("td").last.inner_text() == "not sent"
    assert row.locator("[data-frontier-not-sent='hle']").inner_text() == (
        f"Not sent: {GONE}. Its 4 answers wait as they are — none marked ungraded or counted "
        "wrong — and Start sends them once a model is chosen for it again")
    assert "hle" not in est["graders"]
    cost = page.evaluate("usd(state.ai.frg.estimate.usd)")
    assert card.locator("[data-frontier-start]").inner_text() == f"Start grading: about {cost}"
    assert not db.batches_pending()                            # nothing sent
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(card, SCREENS / f"grading-judge-gone-{width}-{scheme}.png")
