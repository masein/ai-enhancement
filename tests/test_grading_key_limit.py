"""The OpenRouter key's own limit, on the grading card (masein, 5 Nov).
SimpleQA's grading stopped at 263 of 983 on twenty 403s in a row and said
only "OpenRouter refused it (HTTP 403)": the board kept the first 400
characters of each error, and OpenRouter's words about the key came after
them. A refusal's plain words are now read from the whole error and kept,
with OpenRouter's own message; at the key's limit the batch says what the key
has used. The card's key warning counted Start's new answers alone, and said
nothing with the key at its limit: it now counts what Carry on sends and the
batches out at OpenRouter's Batch API too, which Carry on no longer counts as
its own. Invented questions and a stand-in for OpenRouter only: nothing is
sent anywhere."""

from __future__ import annotations

import json
import time

import pytest

from service import ai_models, config, db, llm
from service import frontier_grade as fgr
from test_17_grading import svc  # noqa: F401 — svc is the fixture
from test_17b_grading import GPT

URL = "https://openrouter.ai/api/v1/chat/completions"
# a key-limit 403 whose message comes after the 400 characters kept
LATE = json.dumps({"error": {"code": 403,
                             "metadata": {"headers": {f"x-ratelimit-{i}": "v" * 40
                                                      for i in range(12)}},
                             "message": "Key limit exceeded (total limit). Manage it using "
                                        "https://openrouter.ai/settings/keys"}})
LIMIT_WORDS = ("OpenRouter refused the key: it has reached its own spending limit, or the account "
               "is out of credit. That is fixed on OpenRouter’s side: raise the key’s limit or add "
               "credit there")
KEY = {"limit": 250.0, "remaining": 0.0, "usage": 336.15}


def test_a_refusals_words_are_read_from_the_whole_error_and_kept():
    e = llm.LLMError(f"POST {URL}: HTTP 403: {LATE}", status=403)
    assert LATE.index("Key limit") > 400
    # what the board said of it before: the first 400 characters said nothing
    # of the key
    assert ai_models.refusal(403, str(e)[:400])[1] == "OpenRouter refused it (HTTP 403)"
    rec = llm._refused({"custom_id": "frgr:1#0", "text": "", "error": "", "attempts": 1}, e)
    assert (rec["kind"], rec["said"]) == ("limit", LIMIT_WORDS)
    assert rec["error"] == f"POST {URL}: HTTP 403: " + json.dumps({"error": {
        "code": 403, "message": "Key limit exceeded (total limit). Manage it using "
                                "https://openrouter.ai/settings/keys"}})
    # the part kept says so too, for a record written without `said`
    assert llm.said_of({k: v for k, v in rec.items() if k != "said"}) == LIMIT_WORDS
    assert llm.said_of(rec) == LIMIT_WORDS


def test_a_flagged_prompt_that_mentions_credit_is_never_the_keys_limit():
    body = json.dumps({"error": {"code": 403, "message": "openai/gpt-4.1 requires moderation on "
                                 "Azure. Your input was flagged for \"harassment\"",
                                 "metadata": {"reasons": ["harassment"], "provider_name": "Azure",
                                              "flagged_input": "who pays the credit card bill"}}})
    kind, said = ai_models.refusal(403, f"POST {URL}: HTTP 403: {body}")
    assert kind == "answer" and "flagged" in said


def test_a_batch_at_the_keys_limit_says_so_and_the_card_warns_of_what_carry_on_sends(
        svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    monkeypatch.setattr(fgr.GraderChat, "HALT_AFTER", 3)
    monkeypatch.setattr(ai_models, "key_allowance", lambda: dict(KEY))

    def http(method, url, headers, body=None, timeout=60.0):
        req = json.loads(body)
        if req["model"] == GPT:
            raise llm.LLMError(f"POST {url}: HTTP 403: {LATE}", status=403)
        return 200, json.dumps({"choices": [{"message": {"content": "No"},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 1}}).encode()
    monkeypatch.setattr(llm, "_http", http)
    fgr.start("masein")
    sqa = next(p for p in fgr.pending() if p["slot"] == "simpleqa")
    end = time.time() + 15
    while time.time() < end and not (llm.tally(sqa["batch_id"]) or {}).get("halted"):
        time.sleep(0.05)
    page = svc.get("/api/frontier/grading").json()
    line = next(r["progress"] for r in page["running"] if r["slot"] == "simpleqa")
    assert line.endswith("waiting: OpenRouter refused 3 requests in a row — " + LIMIT_WORDS
                         + " (the key has used $336.15 of its $250.00 limit). It waits for "
                         "Start: nothing more is sent until then"), line
    # the card: nothing new for Start, five answers for Carry on — and the key
    # can't pay for them
    h = page["held"]
    assert page["estimate"]["answers"] == 0 and h["answers"] == 5 and h["usd"] > 0
    assert page["key_warning"] == (
        f"About ${h['usd']:,.2f} — what Carry on sends ${h['usd']:,.2f} — is more than the "
        "OpenRouter key may still spend ($0.00: the key has used $336.15 of its $250.00 "
        "limit): raise the key’s limit on OpenRouter first, or what is sent stops where the "
        "key runs out")
    # with room on the key, nothing is said
    monkeypatch.setattr(ai_models, "key_allowance",
                        lambda: {**KEY, "limit": 500.0, "remaining": 163.85})
    assert svc.get("/api/frontier/grading").json()["key_warning"] == ""
    # Carry on with OpenRouter still refusing: the press is kept with what it
    # did, and the batch waits again — said by the wait line, not a toast
    got = svc.post("/api/frontier/grading/start", json={"by": "masein"})
    assert got.status_code == 200 and got.json()["taken"] == ["SimpleQA Verified's grader"]
    lp = got.json()["page"]["last_press"]
    assert (lp["what"], lp["by"], lp["refused"]) == ("Carry on", "masein", False)
    assert lp["words"] == "nothing new to send · took SimpleQA Verified's grader’s batch up again"
    end = time.time() + 15
    while time.time() < end and not (llm.tally(sqa["batch_id"]) or {}).get("halted"):
        time.sleep(0.05)
    assert "refused 3 requests in a row" in (llm.tally(sqa["batch_id"]) or {})["halted"]
    # a press refused: why, kept on the card
    monkeypatch.setattr(ai_models, "over_limit", lambda: "waiting: this month's AI spend has "
                        "reached its $150.00 limit — raise it on AI models, or wait for next month")
    assert svc.post("/api/frontier/grading/start", json={"by": "masein"}).status_code == 409
    lp = svc.get("/api/frontier/grading").json()["last_press"]
    assert lp["refused"] and lp["words"] == (
        "refused — waiting: this month's AI spend has reached its $150.00 limit — raise it on AI "
        "models, or wait for next month. Nothing was sent")


def test_what_is_out_at_openrouters_batch_api_is_no_longer_carry_ons(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ai_models, "key_allowance", lambda: dict(KEY))
    monkeypatch.setattr(llm.OpenRouterBatchAPI, "_ensure_worker", lambda self, bid: False)
    pin = {"kind": "openrouter", "id": "openai/gpt-5.6-sol:batch",
           "version": "openai/gpt-5.6-sol-20260709", "name": "GPT-5.6 Sol (batch)",
           "provider": "openai", "provider_name": "OpenAI", "precision": "unknown",
           "price_in": 2.0, "price_out": 10.0, "batch": True}
    be = fgr._backend(pin)
    ids = [f"frgr:h{k}#0" for k in range(4)]
    bid = be.submit([llm.Request(c, "", "x" * 4000, max_tokens=4096) for c in ids], start=False)
    fgr.gdir().joinpath("batches").mkdir(parents=True, exist_ok=True)
    (fgr.gdir() / "batches" / f"{bid}.json").write_text(json.dumps({
        "slot": "hle", "task": "hle_text_cais", "row": "served__lda-box",
        "model": "served/lda-box", "base": "served/lda-box", "pin": pin,
        "prompt_sha256": "x", "keys": [c.split(":", 1)[1] for c in ids], "answers": {}}))
    db.batch_add(bid, fgr.KIND, 0, len(ids), "openrouter", pin["id"])
    # unsent: Carry on's to send
    h = fgr.held()
    assert (h["answers"], h["out"]["answers"]) == (4, 0)
    # out at OpenRouter: its cost is the batch's, charged as it finishes
    (be.dir / bid / "remote.json").write_text(json.dumps({"id": "batch_1", "ids": ids,
                                                          "sent_at": time.time()}))
    h = fgr.held()
    usd = (4 * 4000 // fgr.CHARS_A_TOKEN * 2.0 + 4 * 900 * 10.0) / 1e6
    assert (h["answers"], h["usd"]) == (0, 0.0)
    assert h["out"] == {"answers": 4, "usd": pytest.approx(usd), "usd_known": True}
    assert fgr._unsent(next(p for p in fgr.pending() if p["batch_id"] == bid)) == 0
    est = fgr.estimate()
    assert fgr.key_warning(est, h).startswith(
        f"About ${est['usd'] + usd:,.2f} — Start’s new answers ${est['usd']:,.2f}, the batches "
        f"out at OpenRouter ${usd:,.2f} — is more than the OpenRouter key may still spend "
        "($0.00: the key has used $336.15 of its $250.00 limit)")


def test_a_batch_that_halts_while_start_is_still_going_keeps_its_halt(svc, monkeypatch):  # noqa: F811
    # 11 Nov (CI on #302): Start sends a batch, then takes up again every halted batch
    # — and took up the one it had just sent, when its worker met the run of refusals
    # first (a busy machine): the halt was lifted, the worker had ended, and the card
    # said "0 of 5 · $0.00 so far" for good. Forced here: each new batch halts before
    # _submit returns
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    monkeypatch.setattr(fgr.GraderChat, "HALT_AFTER", 3)
    monkeypatch.setattr(ai_models, "key_allowance", lambda: dict(KEY))

    def http(method, url, headers, body=None, timeout=60.0):
        if json.loads(body)["model"] == GPT:
            raise llm.LLMError(f"POST {url}: HTTP 403: {LATE}", status=403)
        return 200, json.dumps({"choices": [{"message": {"content": "No"},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 1}}).encode()
    monkeypatch.setattr(llm, "_http", http)
    submit = fgr._submit

    def submit_and_wait(w, pin, by):
        bid = submit(w, pin, by)
        end = time.time() + 15
        while bid and w["slot"] == "simpleqa" and time.time() < end \
                and not (llm.tally(bid) or {}).get("halted"):
            time.sleep(0.02)
        return bid
    monkeypatch.setattr(fgr, "_submit", submit_and_wait)
    got = fgr.start("masein")
    sqa = next(p for p in fgr.pending() if p["slot"] == "simpleqa")
    assert sqa["batch_id"] in {s["batch_id"] for s in got["sent"]}
    assert (llm.tally(sqa["batch_id"]) or {}).get("halted", "").startswith(
        "waiting: OpenRouter refused 3 requests in a row")
    assert "taken" not in got                 # nothing was taken up: it was all new
    # its batches' workers done before the test's folder goes (none left writing)
    end = time.time() + 15
    while time.time() < end and any(fgr._working(p["batch_id"]) for p in fgr.pending()):
        time.sleep(0.05)
