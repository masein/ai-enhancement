"""17i on the page, part 1, point 6: a Start that costs more than the month
has left is said before anything is sent — what is left, what it costs, and
that it stops part-way — and refused unless started anyway. Answers invented
on disk; the grader is a stand-in, nothing is sent anywhere."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import frontier as fb
from conftest import set_name

pytestmark = pytest.mark.dashboard
SQA = "simpleqa_epoch"
SID = "served/limit-box"
GPT_V = "openai/gpt-4.1-2025-04-14"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """one row of SimpleQA, four answers to grade, GPT-4.1 chosen, and a
    month's limit of a tenth of a cent"""
    import service.app as appmod
    from service import ai_models, config, db, served
    from service import frontier as sf
    from service import frontier_grade as fgr
    saved = config.OPENROUTER_API_KEY
    config.OPENROUTER_API_KEY = "test-key"
    items = [{"id": str(k), "question": f"Limit fact {k}?", "answer": f"A{k}", "subject": ""}
             for k in range(4)]
    p = fb._cache(config.BENCH_ROOT, SQA)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items))
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": "openai/gpt-4.1", "name": "OpenAI: GPT-4.1", "version": GPT_V,
         "price_in": 2.0, "price_out": 8.0, "reasons": False}]}))
    db.served_put({"id": SID, "name": "limit box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "l.gguf"}, "by": "masein", "at": 0})
    served.write_meta(db.served_get(SID))
    row = Path(config.OUT_DIR) / SID.replace("/", "__")
    d = sf.task_dir(row, SQA)
    d.mkdir(parents=True, exist_ok=True)
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": it["id"], "epoch": 0, "answer": f"I think {it['answer']}.", "finish": "stop"})
        + "\n" for it in items))
    db.ai_set(fgr._setting("simpleqa"), {
        "kind": "openrouter", "id": "openai/gpt-4.1", "version": GPT_V, "name": "OpenAI: GPT-4.1",
        "provider": "p", "provider_name": "Prov", "price_in": 2.0, "price_out": 8.0}, "masein")
    limit = db.ai_get("spend_limit", None)
    db.ai_set("spend_limit", 0.001, "masein")
    sent = []

    def complete(self, req):
        sent.append(req["custom_id"])
        return {"custom_id": req["custom_id"], "text": "A", "error": "", "attempts": 1,
                "finish_reason": "stop"}
    real = fgr.GraderChat._complete
    fgr.GraderChat._complete = complete
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield {"dir": d, "sent": sent}
    fgr.GraderChat._complete = real
    db.ai_set("spend_limit", limit if limit is not None else config.AI_MONTHLY_LIMIT_USD,
              "masein")
    config.OPENROUTER_API_KEY = saved


def test_6_more_than_the_month_has_left_is_said_and_refused_unless_started_anyway(
        live, page, board):
    from service import frontier_grade as fgr
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=ai")
    page.wait_for_selector("[data-frontier-grading]:not([data-frontier-grading='loading'])")
    set_name(page, "masein")
    card = page.locator("[data-frontier-grading]")
    short = card.locator("[data-frontier-short]")                 # 511854e: nothing said
    short.wait_for()
    words = short.inner_text()
    assert "has $0.00 left" in words and "stop part-way" in words and "costs about $0.0" in words
    card.locator("[data-frontier-start]").click()
    page.wait_for_selector("[data-toast='frg']")
    assert "Refused. This month's AI limit has $0.00 left" in \
        page.locator("[data-toast='frg']").inner_text()
    assert board["sent"] == [] and fgr.pending() == []            # nothing sent
    anyway = card.locator("[data-frontier-partial]")
    assert anyway.inner_text() == "Start anyway — stops at the limit, $0.00 from now"
    anyway.click()
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    end = time.time() + 15                                        # started, knowing
    while time.time() < end and len(board["sent"]) < 4:
        time.sleep(0.1)
    assert sorted(board["sent"]) == [f"frgr:{k}#0" for k in range(4)]
    # the refused Start's 409 is the browser's own line for it, and expected
    assert [e for e in page.errors if "409 (Conflict)" not in e] == []
