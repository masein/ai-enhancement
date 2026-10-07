"""17j, part 2, point 12: the dry run's cost of a grader's replies, read from
this month's ledger once it has replied — HLE's 900 tokens an answer was an
assumption, and at five times that a Start past the month's limit wasn't
refused. The grader is a stand-in; nothing calls OpenRouter."""

from __future__ import annotations

import json

from service import db
from service import frontier_grade as fgr
from test_17_grading import svc  # noqa: F401 — svc is the fixture
from test_17b_grading import GPT

GPT_V = "openai/gpt-4.1-2025-04-14"


def test_12_the_dry_run_prices_replies_as_the_ledger_says_they_cost(svc):  # noqa: F811
    fgr.save("simpleqa", GPT, "masein")
    est = fgr.estimate()
    g = est["graders"]["simpleqa"]
    assert (g["out_each"], g["out_from"]) == (2, "assumed")
    # this grader's replies to SimpleQA this month, on the ledger: 40 tokens each
    b = fgr.gdir() / "batches" / "or_ledger00001.json"
    b.parent.mkdir(parents=True, exist_ok=True)
    b.write_text(json.dumps({"slot": "simpleqa", "task": "simpleqa_epoch",
                             "pin": {"id": GPT, "version": GPT_V}}))
    for _ in range(3):
        db.spend_add(fgr.JOB, GPT, "Prov", 300, 40, 0.001, "or_ledger00001")
    db.spend_add(fgr.JOB, GPT, "Prov", 300, 999, 0.001, "or_another0001")   # another job's batch
    est2 = fgr.estimate()
    g2 = est2["graders"]["simpleqa"]
    assert (g2["out_each"], g2["out_from"]) == (40, "what its 3 replies this month cost")
    assert g2["tokens_out"] == 40 * g2["answers"]                  # 0abb757: 2 an answer
    assert est2["usd"] > est["usd"]
