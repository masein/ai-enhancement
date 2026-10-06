"""17b, part 6 on the page: Start pressed on AI models' grading card — the
button disabled while its request is out, a second press sending nothing;
Stop, then "Carry on" with the reason it waits; and the Frontier view ranking
two models whose MATH Level 5 was checked alike in one pool, though each
cell carries its own "code alone". Answers and questions invented on disk; a
stand-in for OpenRouter in this process."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import frontier as fb
from conftest import set_name
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase17"
WAITING = ("served/press-box", "served__press-box")
CHECKED = {"served/check-a": ["ANSWER: 2", "ANSWER: two", "ANSWER: 3"],
           "served/check-b": ["ANSWER: 2", "ANSWER: 2", "ANSWER: two"]}


def served(sid: str) -> dict:
    return {"id": sid, "name": sid.split("/")[1], "base_url": "", "key": "", "how": "x",
            "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto", "pin": {"file": "m.gguf"},
            "by": "masein", "at": 0}


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """four SimpleQA answers waiting for their grader on one row; MATH Level
    5 on two more, Epoch's check done by the same grader and prompt"""
    import service.app as appmod
    from service import ai_models, config, db
    from service import frontier as sf
    from service import frontier_grade as fgr
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
         "price_in": 2.0, "price_out": 8.0, "reasons": False},
        {"id": "google/gemini-2.5-flash", "name": "Google: Gemini 2.5 Flash",
         "version": "google/gemini-2.5-flash-20250617", "price_in": 0.3, "price_out": 2.5,
         "reasons": True}]}))

    def write(row, task, answers):
        d = sf.task_dir(row, task)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": q, "epoch": 0, "answer": a, "finish": "stop"}) + "\n" for q, a in answers))
        return d
    db.served_put(served(WAITING[0]))
    row = Path(config.OUT_DIR) / WAITING[1]
    row.mkdir(parents=True, exist_ok=True)
    (row / "model_meta.json").write_text(json.dumps({"model": WAITING[0]}))
    write(row, "simpleqa_epoch", [(str(k), f"A{k}") for k in range(4)])
    # MATH: the same check on both rows' answers the code marks wrong — "two"
    # equivalent, 3 not — so each row's own "code alone" differs
    pin = {"id": "google/gemini-2.5-flash", "version": "google/gemini-2.5-flash-20250617",
           "provider": "p", "provider_name": "Prov"}
    who = fgr.grader_record("math", pin, "masein")
    for sid, answers in CHECKED.items():
        db.served_put(served(sid))
        r = Path(config.OUT_DIR) / sid.replace("/", "__")
        r.mkdir(parents=True, exist_ok=True)
        (r / "model_meta.json").write_text(json.dumps({"model": sid}))
        d = write(r, "math_l5_epoch", [(f"algebra/{k}", a) for k, a in enumerate(answers)])
        items = {f"algebra/{k}#0": {"ok": a.endswith("two"), "words": "", "by": who["version"],
                                    "prompt_sha256": who["prompt_sha256"],
                                    "answer_sha256": sf.answer_sha(a), "at": 0}
                 for k, a in enumerate(answers) if a != "ANSWER: 2"}
        (d / sf.GRADES).write_text(json.dumps({"grader": who, "graders": [who], "items": items,
                                               "refused": {}}))
        sf.score_task(r, "math_l5_epoch", db.served_get(sid))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    config.OPENROUTER_API_KEY = saved


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def test_start_pressed_sends_once_and_stop_then_carry_on(live, page, monkeypatch):
    from service import ai_models, db
    from service import frontier_grade as fgr
    monkeypatch.setattr(ai_models, "drifted", lambda pin: "")
    monkeypatch.setattr(ai_models, "pin", lambda mid, job="pin": {
        "kind": "openrouter", "id": mid, "version": mid + "-2025", "name": mid, "provider": "p",
        "provider_name": "Prov", "precision": "unknown", "price_in": 2.0, "price_out": 8.0})
    monkeypatch.setattr("service.config.OPENROUTER_CONCURRENCY", 1)
    gate, asked, presses = threading.Event(), [], []

    def complete(self, row):
        asked.append(row["custom_id"])
        if len(asked) == 1:
            gate.wait(20)                   # the first reply holds until Stop is pressed
        return {"custom_id": row["custom_id"], "text": "A", "error": "", "attempts": 1,
                "finish_reason": "stop"}
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    real = fgr._start

    def slow(by):
        presses.append(by)
        time.sleep(1.0)                     # pinning the graders: the request is out
        return real(by)
    monkeypatch.setattr(fgr, "_start", slow)
    go(page, live, "tab=ai", "[data-frontier-grading]:not([data-frontier-grading='loading'])")
    set_name(page, "masein")
    card = page.locator("[data-frontier-grading]")
    start = card.locator("[data-frontier-start]")
    assert start.inner_text().startswith("Start grading: about ")
    start.click()
    # out: disabled, saying so — and a second press sends nothing
    page.wait_for_function("document.querySelector('[data-frontier-start]').disabled")
    assert start.inner_text() == "Sending…"
    page.evaluate("frontierGradingAct('start')")
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    assert presses == ["masein"]
    # the batch out, its first reply held: Stop, then the card says why it waits
    page.wait_for_selector("[data-frontier-stop]")
    for _ in range(100):
        if asked:
            break
        time.sleep(0.05)
    card.locator("[data-frontier-stop]").click()
    page.wait_for_function("state.ai.frg && state.ai.frg.stopped")
    gate.set()
    page.wait_for_selector("[data-frontier-waits]")
    assert "Stopped by masein: " in card.locator("[data-frontier-waits]").inner_text()
    # 17c: Carry on says what it sends costs, and the dry run what is held
    assert card.locator("[data-frontier-start]").inner_text().startswith("Carry on: about $")
    held = card.locator("[data-frontier-estimate='0']").inner_text()
    assert held.startswith("Nothing new waits for a grader: ") and "held in the batches out" in held
    assert card.locator("[data-frontier-wait]").first.inner_text().endswith(
        " — Carry on sends the rest.")
    assert card.locator("[data-frontier-run]").get_attribute("data-frontier-run") == "held"
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(card, SCREENS / "ai-models-grading-stopped.png")
    n = len(asked)
    time.sleep(0.5)
    assert len(asked) == n < 4              # nothing more went
    card.locator("[data-frontier-start]").click()
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    for _ in range(200):
        if not [b for b in db.batches_pending() if b["kind"] == fgr.KIND]:
            break
        time.sleep(0.1)
    assert sorted(set(asked)) == [f"frgr:{k}#0" for k in range(4)] and len(asked) == 4
    assert page.errors == []


def test_the_frontier_view_ranks_two_models_checked_alike_in_one_pool(live, page):
    # MATH Level 5 is measured here: one of the view's columns by default
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    cells = page.locator("[data-fr-cell='math level 5'] [data-fr-set^='here|'], "
                         "[data-fr-cell='math level 5'][data-fr-set^='here|']")
    page.wait_for_function("document.querySelectorAll(\"[data-fr-cell='math level 5'] "
                           "[data-fr-set^='here|'], [data-fr-cell='math level 5']"
                           "[data-fr-set^='here|']\").length >= 2")
    sets = {cells.nth(i).get_attribute("data-fr-set") for i in range(cells.count())}
    tags = page.locator("[data-fr-cell='math level 5'] .fr-tag").all_inner_texts()
    # each cell says its own code-alone number; both rank in the one setting
    assert any("code alone 33.3" in t for t in tags) and any("code alone 66.7" in t for t in tags)
    assert len(sets) == 1 and "code alone" not in next(iter(sets))
    assert page.locator("[data-fr-cell='math level 5'] [data-fr-lead], "
                        "[data-fr-cell='math level 5'][data-fr-lead]").count() >= 1
    assert page.errors == []
