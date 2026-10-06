"""17g on the page: a benchmark whose two rows are scored by different
graders says so on both rows' cells in the Frontier view, and AI models'
grading card offers the other row's regrade with its price — taken, the dry
run shows it, and Undo takes it back. Answers and grades invented on disk;
nothing is sent."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import frontier as fb
from conftest import set_name

pytestmark = pytest.mark.dashboard
SQA = "simpleqa_epoch"
ROWS = {"served/split-a": "openai/gpt-4.1-2025-04-14",
        "served/split-b": "google/gemini-2.5-flash-20250617"}


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """two rows of SimpleQA, every answer graded — one by GPT-4.1, the other by
    Gemini 2.5 Flash, the grader chosen now"""
    import service.app as appmod
    from service import ai_models, config, db
    from service import frontier as sf
    from service import frontier_grade as fgr
    saved_key, was = config.OPENROUTER_API_KEY, db.ai_get(fgr._setting("simpleqa"))
    config.OPENROUTER_API_KEY = "test-key"
    items = [{"id": str(k), "question": f"Split fact {k}?", "answer": f"A{k}", "subject": ""}
             for k in range(4)]
    p = fb._cache(config.BENCH_ROOT, SQA)
    p.parent.mkdir(parents=True, exist_ok=True)
    have = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(have or items))
    items = have or items
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cached = json.loads(p.read_text()) if p.exists() else {"models": []}
    known = {m["id"] for m in cached.get("models") or []}
    cached["models"] = (cached.get("models") or []) + [m for m in [
        {"id": "openai/gpt-4.1", "name": "OpenAI: GPT-4.1", "version": ROWS["served/split-a"],
         "price_in": 2.0, "price_out": 8.0, "reasons": False},
        {"id": "google/gemini-2.5-flash", "name": "Google: Gemini 2.5 Flash",
         "version": ROWS["served/split-b"], "price_in": 0.3, "price_out": 2.5,
         "reasons": True}] if m["id"] not in known]
    cached["at"] = time.time()
    p.write_text(json.dumps(cached))
    for sid, version in ROWS.items():
        db.served_put({"id": sid, "name": sid.split("/")[1], "base_url": "", "key": "",
                       "how": "x", "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                       "pin": {"file": "m.gguf"}, "by": "masein", "at": 0})
        row = Path(config.OUT_DIR) / sid.replace("/", "__")
        d = sf.task_dir(row, SQA)
        d.mkdir(parents=True, exist_ok=True)
        (row / "model_meta.json").write_text(json.dumps({"model": sid}))
        answers = {it["id"]: f"I think {it['answer']}." for it in items}
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": q, "epoch": 0, "answer": a, "finish": "stop"}) + "\n"
            for q, a in answers.items()))
        who = fgr.grader_record("simpleqa", {"id": version.rsplit("-", 3)[0], "version": version,
                                             "provider": "p", "provider_name": "Prov"}, "masein")
        (d / sf.GRADES).write_text(json.dumps({"grader": who, "graders": [who], "refused": {},
                                               "items": {f"{q}#0": {
                                                   "ok": True, "words": "correct",
                                                   "by": version,
                                                   "prompt_sha256": who["prompt_sha256"],
                                                   "answer_sha256": sf.answer_sha(a), "at": 0}
                                                   for q, a in answers.items()}}))
        sf.score_task(row, SQA, db.served_get(sid))
    db.ai_set(fgr._setting("simpleqa"), {
        "kind": "openrouter", "id": "google/gemini-2.5-flash", "version": ROWS["served/split-b"],
        "name": "Google: Gemini 2.5 Flash", "provider": "p", "provider_name": "Prov",
        "price_in": 0.3, "price_out": 2.5}, "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    # the board as the other tests expect it: no regrade asked, the grader as it was
    fgr.regrade_row("served__split-a", SQA, "masein", undo=True)
    db.ai_set(fgr._setting("simpleqa"), was, "masein")
    config.OPENROUTER_API_KEY = saved_key
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def test_both_rows_say_their_graders_differ_in_the_frontier_view(live, page):
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    page.wait_for_function("document.querySelectorAll('[data-fr-grader-differs]').length >= 2")
    said = {page.locator("[data-fr-grader-differs]").nth(i).get_attribute(
        "data-fr-grader-differs"): page.locator("[data-fr-grader-differs]").nth(i).inner_text()
        for i in range(page.locator("[data-fr-grader-differs]").count())}
    assert set(ROWS) <= set(said), said
    assert said["served/split-a"].startswith(f"graded by {ROWS['served/split-a']}; ")
    assert "split-b by " + ROWS["served/split-b"] in said["served/split-a"]
    assert all("one benchmark, two graders" in w for w in said.values())
    assert page.errors == []


def test_the_card_offers_the_other_rows_regrade_priced_and_undo_takes_it_back(live, page):
    go(page, live, "tab=ai", "[data-frontier-grading]:not([data-frontier-grading='loading'])")
    set_name(page, "masein")
    card = page.locator("[data-frontier-grading]")
    page.wait_for_selector(f"[data-frontier-mismatch='{SQA}']")
    offer = card.locator(f"[data-frontier-offer='{SQA}|served__split-a']")
    text = offer.inner_text()
    assert text.startswith(f"served/split-a: grade it again by {ROWS['served/split-b']}, as the "
                           "others are — 4 answers, about $"), text
    # the row graded by the grader chosen now isn't offered
    assert card.locator(f"[data-frontier-offer='{SQA}|served__split-b']").count() == 0
    offer.locator("button").click()
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    assert "at the next Start" in card.locator(
        f"[data-frontier-offer='{SQA}|served__split-a']").inner_text()
    # the dry run: the row, graded again whole, priced before Start
    est = card.locator("[data-frontier-est='simpleqa|served/split-a']")
    assert est.count() == 1 and "so that the rows compared share" in est.inner_text()
    # Undo: nothing waits for it any more
    card.locator(f"[data-frontier-offer='{SQA}|served__split-a'] button").click()
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    assert card.locator("[data-frontier-est='simpleqa|served/split-a']").count() == 0
    assert page.errors == []
