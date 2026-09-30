"""12q.F (the brief's): run #146 (served lookahead + MTP) stopped at 584 of 596.
llama-server answered one item with HTTP 500 — "The model produced output that
does not match the expected peg-native format": its chat parser couldn't read
what the model wrote — and the runner took it for a server that had stopped
answering, waited, and failed the row "at 0 of 0".

Now a 500 on one item is that item's: asked once more; then without the
server's chat parsing — the prompt from POST /apply-template, completed by POST
/completion with the same greedy settings, seed and 4,096 cap, the raw text
scored — marked "raw fallback" with the server's words, and counted on the
row; and if that fails too, no answer (counted wrong, as the protocol counts
one), the errors kept, and the row goes on. A server that really stops
answering still stops the row, and says after how many.

The server is the fake llama-server: every reply is written here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import devicemark as dm
from service import config
from test_12q_devicemark_board import SERVED, _row
from test_12q_devicemark_runs import fake, queue, register, row_dir, svc  # noqa: F401 — fixtures

PEG = "The model produced output that does not match the expected peg-native format"
MATH = [k for b, k in dm.keys_for("full") if b == "math"]
IFEVAL = [k for b, k in dm.keys_for("full") if b == "ifeval"]
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12qf"


def fails_on(*needles: str, status: int = 500, said: str = PEG):
    """the server's own error on the items whose prompt has one of these"""
    def error(body):
        text = (body.get("messages") or [{}])[-1].get("content") or body.get("prompt") or ""
        return (status, said) if any(n in text for n in needles) else None
    return error


def raw(content: str, tokens: int = 12, stop: str = "eos") -> dict:
    return {"content": content, "tokens_predicted": tokens, "tokens_evaluated": 40,
            "stop_type": stop, "timings": {"prompt_n": 40, "predicted_n": tokens,
                                           "predicted_per_second": 30.0}}


def item(mid: str, bench: str, key: str, thinking: bool = False) -> dict:
    return next(s for s in dm.read_items(row_dir(mid, thinking))
                if (s["bench"], s["key"]) == (bench, key))


def log_of(sid: int) -> str:
    [f] = config.LOGS_DIR.glob(f"service_{sid}_*.log")
    return f.read_text()


# ---------------------------------------------------------------------------
# a 500 on one item: asked once more, then without the server's chat parsing
# ---------------------------------------------------------------------------

def test_a_500_on_one_item_is_retried_then_asked_raw_and_the_row_finishes(svc, fake):  # noqa: F811
    mid = register(fake, "Qwen3.6 lookahead MTP")
    k = MATH[3]
    needle = f"Fixture problem {k}."
    fake.chat_error = fails_on(needle)
    fake.completion_reply = lambda body: raw("It is one half: \\boxed{0.5}")
    row = queue(mid)
    assert row["status"] == "done", row["error"]
    # asked, asked once more, then the raw way — and every other item once
    asked = [r for r in fake.requests if needle in r["messages"][-1]["content"]]
    assert len(asked) == 2 and len(fake.requests) == 596 + 1
    [t] = fake.templated
    assert t["messages"] == asked[0]["messages"]
    assert t["chat_template_kwargs"] == {"enable_thinking": False}
    [c] = fake.completions
    assert c == {"prompt": f"<|user|>\n{asked[0]['messages'][-1]['content']}\n<|assistant|>\n",
                 "n_predict": dm.CAP, "temperature": 0.0, "seed": dm.SEED,
                 "cache_prompt": False, "stream": False}
    # scored as any answer, and marked with what the server said
    it = item(mid, "math", k)
    assert it["raw_fallback"] == {"error": f"HTTP 500: {PEG}"}
    assert (it["answered"], it["correct"], it["gen_tokens"], it["capped"]) == (
        True, True, 12, False)
    # counted on the row, in its line and in the log
    out = json.loads((row_dir(mid) / dm.OUT_NAME).read_text())
    assert (out["n"], out["raw_fallback"], out["errors"]) == (596, 1, 0)
    assert " · 1 raw fallback" in row["progress"]
    assert f"[devicemark] math {k}: raw fallback — the server said: HTTP 500: {PEG}" in \
        log_of(row["id"])
    # the next run asks nothing: it is answered
    n = len(fake.requests)
    assert queue(mid)["status"] == "done" and len(fake.requests) == n


def test_the_raw_text_keeps_its_thinking_apart(svc, fake):  # noqa: F811
    """thinking on, the template opens the thinking itself: what the cap cut
    off inside it is thinking, never an answer"""
    mid = register(fake, "Qwen3.6 lookahead MTP")
    closed, cut = MATH[0], MATH[1]
    fake.chat_error = fails_on(f"Fixture problem {closed}.", f"Fixture problem {cut}.")

    def completion(body):
        assert body["prompt"].endswith("<think>\n")
        if f"Fixture problem {cut}." in body["prompt"]:
            return raw("Half of one is \\boxed{0.5}, or is it", tokens=dm.CAP, stop="limit")
        return raw("One half, I think.\n</think>\n\nIt is \\boxed{0.5}", tokens=20)
    fake.completion_reply = completion
    row = queue(mid, thinking=True)
    assert row["status"] == "done", row["error"]
    a = item(mid, "math", closed, thinking=True)
    assert a["answer"] == "It is \\boxed{0.5}" and a["thinking_chars"] == len("One half, I think.")
    assert (a["correct"], a["think_tokens"], a["answer_tokens"]) == (True, 4, 16)
    b = item(mid, "math", cut, thinking=True)
    assert (b["answer"], b["answered"], b["capped"], b["finish"]) == ("", False, True, "length")
    assert json.loads((row_dir(mid, True) / dm.OUT_NAME).read_text())["raw_fallback"] == 2


# ---------------------------------------------------------------------------
# a double failure: no answer, the errors kept, and the row goes on
# ---------------------------------------------------------------------------

def test_a_double_failure_is_no_answer_and_the_row_goes_on(svc, fake):  # noqa: F811
    mid = register(fake, "Qwen3.6 lookahead MTP")
    m, i = MATH[5], IFEVAL[7]
    needles = (f"Fixture problem {m}.", f"fixture note {i} ")
    fake.chat_error = fails_on(*needles)
    fake.raw_error = fails_on(*needles, said="the prompt could not be completed")
    row = queue(mid)
    assert row["status"] == "done", row["error"]
    for bench, key in (("math", m), ("ifeval", i)):
        it = item(mid, bench, key)
        assert it["error"] == {"chat": f"HTTP 500: {PEG}",
                               "fallback": "HTTP 500: the prompt could not be completed"}
        assert (it["answered"], it["correct"], it["text"], it["finish"]) == (
            False, False, "", "error")
        assert "raw_fallback" not in it
    out = json.loads((row_dir(mid) / dm.OUT_NAME).read_text())
    assert (out["n"], out["raw_fallback"], out["errors"]) == (596, 0, 2)
    assert out["benches"]["math"]["answered"] == 99 and out["benches"]["ifeval"]["answered"] == 299
    assert " · 2 no answer after a server error" in row["progress"]
    assert (f"[devicemark] math {m}: no answer — the server said: HTTP 500: {PEG}; without its "
            "chat parsing: HTTP 500: the prompt could not be completed") in log_of(row["id"])
    # recorded: the next run asks nothing
    n = len(fake.requests)
    assert queue(mid)["status"] == "done" and len(fake.requests) == n


def test_a_400_is_the_items_own_too(svc, fake):  # noqa: F811
    mid = register(fake, "Qwen3.6 lookahead MTP")
    k = MATH[2]
    fake.chat_error = fails_on(f"Fixture problem {k}.", status=400,
                               said="the request exceeds the available context size")
    fake.completion_reply = lambda body: raw("\\boxed{0.5}")
    assert queue(mid)["status"] == "done"
    assert item(mid, "math", k)["raw_fallback"]["error"].startswith(
        "HTTP 400: the request exceeds")


# ---------------------------------------------------------------------------
# a server that stops answering still stops the row — and says after how many
# ---------------------------------------------------------------------------

def test_a_server_that_stops_says_after_how_many(svc, fake):  # noqa: F811
    mid = register(fake, "Qwen3.6 lookahead MTP")
    fake.stop_after = fake.answered + 584
    row = queue(mid)
    assert row["status"] == "failed"
    assert row["error"].endswith(
        "the server stopped answering after 584 of 596 (HTTP 503) · the answers it gave are "
        "kept: the next run asks only the other 12")
    assert "0 of 0" not in row["error"]
    assert "[devicemark] the server stopped answering after 584 of 596 (HTTP 503)" in \
        log_of(row["id"])
    # the resubmit asks the missing 12, and the row is whole
    fake.stop_after = None
    n = len(fake.requests)
    row = queue(mid)
    assert row["status"] == "done" and len(fake.requests) - n == 12
    assert json.loads((row_dir(mid) / dm.OUT_NAME).read_text())["n"] == 596


def test_a_server_gone_in_the_middle_of_the_fallback_is_not_an_answer(svc, fake):  # noqa: F811
    """the item's 500, then a server that doesn't answer the raw way either
    (503): nothing is recorded for it, the row stops, and the next run asks it"""
    mid = register(fake, "Qwen3.6 lookahead MTP")
    k = MATH[0]
    needle = f"Fixture problem {k}."
    fake.chat_error = fails_on(needle)
    fake.raw_error = fails_on(needle, status=503, said="Loading model")
    row = queue(mid)
    assert row["status"] == "failed" and "the server stopped answering after " in row["error"]
    assert "(HTTP 503)" in row["error"]
    assert not (row_dir(mid) / dm.OUT_NAME).exists()
    stored = [json.loads(x) for x in (row_dir(mid) / "devicemark_answers.jsonl").read_text()
              .splitlines()]
    assert ("math", k) not in {(r["bench"], r["key"]) for r in stored}


# ---------------------------------------------------------------------------
# where a person reads it
# ---------------------------------------------------------------------------

def test_the_answers_say_which_were_asked_raw_and_which_failed(svc, fake):  # noqa: F811
    mid = register(fake, "Qwen3.6 lookahead MTP")
    a, b = MATH[0], MATH[1]
    fake.chat_error = fails_on(f"Fixture problem {a}.", f"Fixture problem {b}.")
    fake.completion_reply = lambda body: raw("\\boxed{0.5}")
    fake.raw_error = fails_on(f"Fixture problem {b}.", said="no slot")
    assert queue(mid)["status"] == "done"
    got = svc.get("/api/devicemark/answers", params={"model": mid, "bench": "math"}).json()
    by = {x["key"]: x for x in got["items"]}
    assert by[a]["fallback"] == f"HTTP 500: {PEG}" and by[a]["ok"] is True
    assert by[b]["error"] == {"chat": f"HTTP 500: {PEG}", "fallback": "HTTP 500: no slot"}
    assert by[b]["verdict"].startswith("no answer: the server failed on this item")
    assert got["items"][0]["key"] == b                          # no answer first
    runs = svc.get("/api/results").json()["devicemark"][mid]["off"]["row"]
    assert (runs["raw_fallback"], runs["errors"]) == (1, 1)


@pytest.mark.dashboard
def test_the_row_on_the_chart_shows_the_counts(live, page):
    out = Path(live["root"]) / "results" / "full"
    mid = "served/Qwen3.6-k4-LDA-lookahead-MTP"
    _row(out, mid, {"ifeval": 0.8, "mmlu_pro": 0.7, "math": 0.7},
         {**SERVED, "phone": True, "lookahead": True, "name": "Qwen3.6 k4-LDA lookahead MTP"})
    f = out / mid.replace("/", "__") / dm.OUT_NAME
    f.write_text(json.dumps({**json.loads(f.read_text()), "raw_fallback": 1, "errors": 1}))
    try:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=models&chip=ondevice")
        page.wait_for_selector("[data-dm-chart]")
        note = page.locator(f"[data-dm-odd='{mid}']")
        assert note.inner_text().strip() == "1 raw fallback · 1 no answer after a server error"
        tip = json.loads(note.get_attribute("data-tip"))
        assert "/apply-template and /completion" in tip[0] and "counted wrong" in tip[1]
        line = json.loads(page.locator(f"[data-dm-line='{mid}']").get_attribute("data-tip"))
        assert "1 raw fallback · 1 no answer after a server error" in line
        SCREENS.mkdir(parents=True, exist_ok=True)
        page.locator(f"[data-dm-row='{mid}']").screenshot(path=SCREENS / "row-counts.png")
        assert page.errors == []
    finally:
        import shutil
        shutil.rmtree(out / mid.replace("/", "__"), ignore_errors=True)
