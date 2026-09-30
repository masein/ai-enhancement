"""12s: a served model's Everyday, exam, Trust & safety, SimpleQA and
MobileAIBench runs never stop for one question's error.

llama-server answers a question with HTTP 500 when its chat parser can't read
what the model wrote ("The model produced output that does not match the
expected peg-native format"), and the runner took every 5xx for a server
that had stopped answering: it waited on the same question, then failed the
run. 12q.F fixed that for DeviceMark rows; this is the same for the runs that
go through served.answer_task.

- A 500 or a 400 from a server that is answering is the question's own: it is
  asked once more, then without the server's chat parsing (/apply-template,
  then /completion with the same settings), and the raw text is marked as any
  answer is, with a "raw fallback" mark and the server's words.
- If that fails too the question has no answer: counted failed, with both
  errors, and the run goes on. The reason says whose failure it was, never
  "the model wrote nothing".
- More than a few such questions and the server isn't right: the run stops,
  and keeps none of them as answers.
- A server that isn't answering (no connection, 408, 429, 502, 503) is still
  waited for and then stops the run; a model from OpenRouter is asked as before.

The fake llama-server writes every reply: no model runs."""

from __future__ import annotations

import json

import pytest

import everyday as ev
import judge
import simpleqa as sq
import trust_safety as ts
from service import config, db, runner, served
from test_12n2 import STAMP as SQ_STAMP
from test_served_12f1 import SID, fake, model_dir, queue, register, svc  # noqa: F401 — fixtures

PEG = "The model produced output that does not match the expected peg-native format"
SAID = f"HTTP 500: {PEG}"


def _prompt(body: dict) -> str:
    return body["messages"][-1]["content"]


def _fails_on(fake, prompts: set[str], status: int = 500, msg: str = PEG) -> None:  # noqa: F811
    fake.chat_error = lambda body: (status, msg) if _prompt(body) in prompts else None


def _log(sid: int) -> str:
    [f] = config.LOGS_DIR.glob(f"service_{sid}_*.log")
    return f.read_text()


def _item(qid: str) -> dict:
    out = json.loads((model_dir() / "everyday.json").read_text())
    return next(it for it in out["items"] if it["id"] == qid)


# ---------------------------------------------------------------------------
# 1. a 500 on one question: asked again, then raw, and the run finishes
# ---------------------------------------------------------------------------

def test_a_500_on_one_question_is_asked_again_then_raw_and_the_run_finishes(svc, fake):  # noqa: F811
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    bank = ev.load_bank()
    q = bank[7]
    _fails_on(fake, {q["prompt"]})
    fake.completion_reply = lambda body: {"content": "A raw answer.", "tokens_predicted": 3}
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    # asked its own way twice, never more
    assert sum(1 for b in fake.requests if _prompt(b) == q["prompt"]) == 2
    # then the server's own template, with the same message and thinking switch
    [t] = fake.templated
    assert t["messages"] == [{"role": "user", "content": q["prompt"]}]
    assert t["chat_template_kwargs"] == {"enable_thinking": False}
    # and a completion of exactly that prompt, with the run's settings
    [c] = fake.completions
    want = ev.run_settings({})
    assert c["prompt"] == f"<|user|>\n{q['prompt']}\n<|assistant|>\n"
    assert (c["n_predict"], c["temperature"], c["stop"]) == (
        want["max_gen_toks"], want["temperature"], want["until"])
    assert c["cache_prompt"] is False and c["stream"] is False
    # the raw text is the answer, marked like any, and says how it was got
    it = _item(q["id"])
    assert it["answer_text"] == "A raw answer." and it["raw_fallback"] == {"error": SAID}
    assert "server_error" not in it
    [samples] = model_dir().glob("everyday_0shot/served/samples_everyday_*.jsonl")
    recs = [json.loads(x) for x in samples.read_text().splitlines()]
    [rec] = [r for r in recs if r["doc"]["id"] == q["id"]]
    assert rec["raw_fallback"] == {"error": SAID} and rec["tokens"] == 3
    assert not any("raw_fallback" in r or "server_error" in r for r in recs if r is not rec)
    # every question is answered, and the run's line and log count the one
    out = json.loads((model_dir() / "everyday.json").read_text())
    assert len(out["items"]) == len(bank) and out["unasked"] == 0
    assert row["progress"].endswith(" · 1 raw fallback")
    assert (f"[service] everyday question 8 of {len(bank)}: raw fallback — the server said: "
            f"{SAID}") in _log(sid)
    # kept with the answer: a later marking still says so, and the next run asks nothing
    fake.requests.clear()
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done" and fake.requests == []
    assert _item(q["id"])["raw_fallback"] == {"error": SAID}


def test_the_raw_text_keeps_its_thinking_apart(svc, fake):  # noqa: F811
    client, _, _ = svc
    assert register(client, fake, thinking="on").status_code == 200
    a, b = ev.load_bank()[3], ev.load_bank()[4]
    _fails_on(fake, {a["prompt"], b["prompt"]})
    fake.reasoning = "Thinking."
    # the template opened the thinking: one closes it and answers, one never leaves it
    fake.completion_reply = lambda body: (
        {"content": "It asks for a list.\n</think>\n\nThe list.", "tokens_predicted": 9}
        if a["prompt"] in body["prompt"] else {"content": "Still thinking", "tokens_predicted": 2})
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert all(t["chat_template_kwargs"] == {"enable_thinking": True} for t in fake.templated)
    assert all(c["prompt"].endswith("<think>\n") for c in fake.completions)
    ia, ib = _item(a["id"]), _item(b["id"])
    assert ia["answer_text"] == "The list." and ia["had_reasoning"]
    assert ia["reasoning_text"] == "It asks for a list."
    # thinking the budget cut off is never an answer
    assert ib["no_answer"] is True and ib["reason"] == ev.NEVER_FINISHED
    assert ib["raw_fallback"] == {"error": SAID}


# ---------------------------------------------------------------------------
# 2. the fallback fails too: no answer, and the run goes on
# ---------------------------------------------------------------------------

def test_a_double_failure_is_no_answer_and_the_run_goes_on(svc, fake):  # noqa: F811
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    bank = ev.load_bank()
    # one a script check decides, one the judge would: neither is sent to it
    judged = next(q for q in bank if any(c["type"] == "judge" for c in q["checks"]))
    plain = next(q for q in bank if not any(c["type"] == "judge" for c in q["checks"]))
    _fails_on(fake, {judged["prompt"], plain["prompt"]})
    fake.raw_error = lambda body: (500, "the completion failed too")
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    err = {"chat": SAID, "fallback": "HTTP 500: the completion failed too"}
    for q in (judged, plain):
        it = _item(q["id"])
        assert it["pass"] is False and it["answer_text"] == "" and it["server_error"] == err
        assert it["reason"] == judge.server_failed(err) == (
            f"no answer: the server failed on this question twice ({SAID}), and again without "
            f"its chat parsing (HTTP 500: the completion failed too)")
        assert "wrote nothing" not in it["reason"]
    out = json.loads((model_dir() / "everyday.json").read_text())
    assert len(out["items"]) == len(bank) and out["unasked"] == 0 and out["waiting"] == 0
    assert row["progress"].endswith(" · 2 no answer after a server error")
    log = _log(sid)
    assert log.count("no answer — the server said: " + SAID
                     + "; without its chat parsing: HTTP 500: the completion failed too") == 2
    # no answer is an answer on file: the next run doesn't ask it again
    fake.requests.clear()
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done" and fake.requests == []
    assert _item(plain["id"])["server_error"] == err


def test_a_server_without_the_raw_way_round_is_no_answer_too(svc, fake):  # noqa: F811
    # an older llama-server has no /apply-template or /completion and says 404: the
    # fallback failed, so the question has no answer and the run goes on, never "refused"
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    q = ev.load_bank()[11]
    _fails_on(fake, {q["prompt"]})
    fake.raw_error = lambda body: (404, "File Not Found")
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    it = _item(q["id"])
    assert it["server_error"]["chat"] == SAID
    assert it["server_error"]["fallback"].startswith("HTTP 404: ")
    assert "File Not Found" in it["server_error"]["fallback"]
    assert row["progress"].endswith(" · 1 no answer after a server error")


def test_a_400_is_the_questions_own_too(svc, fake):  # noqa: F811
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    q = ev.load_bank()[0]
    _fails_on(fake, {q["prompt"]}, 400, "the request exceeds the available context size")
    fake.completion_reply = lambda body: {"content": "Raw.", "tokens_predicted": 1}
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert _item(q["id"])["raw_fallback"] == {
        "error": "HTTP 400: the request exceeds the available context size"}


# ---------------------------------------------------------------------------
# 3. too many, and the server isn't right
# ---------------------------------------------------------------------------

def test_the_limit_is_three_questions_or_two_percent():
    assert served.item_error_limit(5) == 3 and served.item_error_limit(150) == 3
    assert served.item_error_limit(310) == 7 and served.item_error_limit(1000) == 20


def test_too_many_questions_with_no_answer_stop_the_run_and_none_is_kept(svc, fake, monkeypatch):  # noqa: F811
    client, _, _ = svc
    monkeypatch.setattr(config, "SERVED_CONCURRENCY", 1)     # one at a time: the count is exact
    assert register(client, fake).status_code == 200
    bank = ev.load_bank()
    n = len(bank)
    bad = {q["prompt"] for q in bank[100:]}                 # fine for 100, then every one fails
    _fails_on(fake, bad)
    fake.raw_error = lambda body: (500, "the completion failed too")
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    limit = served.item_error_limit(n)
    assert row["status"] == "failed"
    assert row["error"].startswith(
        f"everyday: the server failed on {limit + 1} questions, asked its own way and without "
        f"its chat parsing ({SAID}): stopped")
    assert row["error"].endswith("· the 100 answered are kept and marked")
    # the 100 before them are kept; none of the failed is an answer
    out = json.loads((model_dir() / "everyday.json").read_text())
    assert len(out["items"]) == 100 and out["unasked"] == n - 100
    assert not any(it.get("server_error") for it in out["items"])
    assert "no answer after a server error" not in (row["progress"] or "")
    # the server put right, the next run asks the rest, those too
    fake.chat_error = None
    fake.requests.clear()
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    assert {_prompt(b) for b in fake.requests} == bad


# ---------------------------------------------------------------------------
# 4. a server that isn't answering is another matter
# ---------------------------------------------------------------------------

def test_a_server_gone_in_the_middle_of_the_fallback_stops_the_run(svc, fake):  # noqa: F811
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    bank = ev.load_bank()
    q = bank[20]
    _fails_on(fake, {q["prompt"]})
    fake.raw_error = lambda body: (503, "loading model")
    sid = queue("everyday")
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed"
    assert row["error"].startswith("everyday: the server stopped answering at ")
    # nothing is recorded for the question: neither an answer nor a no-answer
    out = json.loads((model_dir() / "everyday.json").read_text())
    assert q["id"] not in {it["id"] for it in out["items"]}
    assert not any(it.get("server_error") or it.get("raw_fallback") for it in out["items"])


def test_a_model_from_openrouter_is_asked_as_before(fake):  # noqa: F811
    body = {"model": "x", "messages": [{"role": "user", "content": "hello"}]}
    fake.chat_error = lambda b: (500, PEG)
    url = fake.base + "/chat/completions"
    # a llama-server's 500 is the question's own, in the server's words
    with pytest.raises(served.ItemError, match="HTTP 500: The model produced output"):
        served._post(url, "", body)
    # OpenRouter's is a provider that didn't answer: waited for, as before
    with pytest.raises(served._Retry, match="HTTP 500"):
        served._post(url, "", body, item=False)
    # and a 502 or 503 is never the question's own
    fake.chat_error = lambda b: (503, "loading model")
    with pytest.raises(served._Retry, match="HTTP 503"):
        served._post(url, "", body)


# ---------------------------------------------------------------------------
# the other runs that go the same way
# ---------------------------------------------------------------------------

def test_the_exam_goes_on_and_marks_it_as_the_empty_answer_it_is(svc, fake):  # noqa: F811
    client, _, _ = svc
    assert register(client, fake).status_code == 200
    task = config.judged_tasks()[0]
    items = [json.loads(x) for x in (config.JUDGED_TASKS_DIR / f"{task}.jsonl").read_text()
             .splitlines() if x.strip()]
    _fails_on(fake, {items[2]["prompt"] + "\n\nAnswer:"})
    fake.raw_error = lambda body: (500, "the completion failed too")
    sid = queue("judged", [task])
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert " · 1 no answer after a server error" in row["progress"]
    [samples] = model_dir().glob(f"{task}_0shot/served/samples_{task}_*.jsonl")
    recs = [json.loads(x) for x in samples.read_text().splitlines()]
    assert len(recs) == len(items)
    [rec] = [r for r in recs if r.get("server_error")]
    assert rec["resps"] == [[""]] and judge.answer_parts(rec)["server_error"]["chat"] == SAID
    assert task in json.loads((model_dir() / "judge.json").read_text())["tasks"]


def test_answer_parts_carries_what_the_server_did():
    err = {"chat": SAID, "fallback": "HTTP 500: no"}
    p = judge.answer_parts({"resps": [[""]], "server_error": err})
    assert p["server_error"] == err and p["answer_text"] == "" and not p["no_answer"]
    p = judge.answer_parts({"resps": [["Raw."]], "raw_fallback": {"error": SAID}})
    assert p["raw_fallback"] == {"error": SAID} and "server_error" not in p
    # a run with neither has neither key
    assert not set(judge.SERVER_KEYS) & set(judge.answer_parts({"resps": [["Hi."]]}))


def _sit(d, task: str, stamp: str, docs: list[dict], texts: list[str], err: dict) -> None:
    """a served run's samples for `task`: the last question its server failed on"""
    d = d / f"{task}_0shot" / "served"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"samples_{task}_{stamp}.jsonl", "w", encoding="utf-8") as fh:
        for i, (doc, text) in enumerate(zip(docs, texts)):
            fh.write(json.dumps({"doc_id": i, "doc": doc, "resps": [[text]],
                                 "filtered_resps": [text],
                                 **({"server_error": err} if i == len(docs) - 1 else {})}) + "\n")


def test_trust_and_safety_and_simpleqa_say_the_servers_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    err = {"chat": SAID, "fallback": "HTTP 500: no"}
    d = tmp_path / "served__m"
    dna = ts.load(ts.DNA)[:3]
    _sit(d, ts.DNA, "2026-09-30T10-00-00.000000", dna, ["I can't help with that.", "", ""], err)
    by = {it["id"]: it for it in ts.start(d)["items"]}
    # the one left empty wrote nothing; the one its server failed on says so
    assert by[dna[1]["id"]]["reason"] == ts.NOTHING
    assert by[dna[2]["id"]]["reason"] == judge.server_failed(err)
    assert by[dna[2]["id"]]["unmarked"] is True

    qs = sq.load()[:2]
    docs = [{"id": q["id"], "prompt": q["prompt"]} for q in qs]
    _sit(d, sq.TASK, SQ_STAMP, docs, [f"It is {qs[0]['answer']}.", ""], err)
    by = {it["id"]: it for it in sq.start(d)["items"]}
    assert by[qs[1]["id"]]["reason"] == judge.server_failed(err)
    assert by[qs[1]["id"]]["grade"] == "not_attempted"
