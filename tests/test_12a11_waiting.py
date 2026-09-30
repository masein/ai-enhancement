"""12a.11 (the brief's): after 12a.10's re-mark, served/Qwen3.6-k4-LDA-lookahead
had 28 Summarise answers waiting (11 hidden, 17 practice) and lookahead-MTP 2,
each saying "the judge's reply couldn't be read". They weren't unreadable:
the judge's server stopped answering near the end of the batch (24 requests
timed out, 2 were disconnected, 4 refused), and a failed request was said as
an unreadable reply.

Now each waiting answer says why — the request failed (and how), the reply
was cut off at the token cap (asked again with twice the room), or it
couldn't be read — and one model's failure to land stays that model's: the
verdicts the batch brought stand, its own and every other's. --compare lists
the reasons under its tables. Nothing runs a model."""

from __future__ import annotations

import json

import pytest

import everyday as ev
from conftest import make_service
from service import config, llm
from test_everyday_12a5 import _asked

BANK = {q["id"]: q for q in ev.load_bank()}
SUMM = [q for q in BANK.values() if q["group"] == "summarising" and ev.half(q) == ev.PRACTICE]
A, B = "org__a", "org__b"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    yield client
    client.__exit__(None, None, None)


def remark(names=(A, B)) -> dict[str, list[dict]]:
    """two models' Summarise answers, each marked on an older rubric, sent again"""
    for name in names:
        d = config.OUT_DIR / name
        _asked(d, SUMM)
        ev.write(d, {"model": name.replace("__", "/"), "items": [
            {"id": q["id"], "group": "summarising", "pass": True, "score": 4, "reason": "",
             "rubric": "", "answer_text": q["reference"]} for q in SUMM]})
    ev.remark(config.OUT_DIR, judge=True)
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith(ev.REMARK + ":")]
    return {n: [r for r in sent if f":{n}:" in r["custom_id"]] for n in names}


def replies(sent: list[dict], **odd) -> dict[str, llm.Result]:
    """the stand-in judge's reply to each, but those given"""
    out = {r["custom_id"]: llm.Result(text=ev.stub_reply(r["user"])) for r in sent}
    for qid, res in odd.items():
        out[next(c for c in out if c.endswith(":" + qid))] = res
    return out


def items(name: str) -> dict[str, dict]:
    return {it["id"]: it for it in ev.read(config.OUT_DIR / name)["items"]}


FIRST, SECOND = SUMM[0]["id"], SUMM[1]["id"]
CUT = '{"checklist": [{"fact": "'


def test_one_models_failure_never_fails_the_others(svc, monkeypatch):
    sent = remark()
    real = ev.finish

    def finish(d, rs):
        if d.name == A:
            raise RuntimeError("boom")
        return real(d, rs)
    monkeypatch.setattr(ev, "finish", finish)
    done = ev.finish_remark(config.OUT_DIR, replies(sent[A] + sent[B]))     # never raises
    assert done == [B]
    assert all(it["pass"] is not None for it in items(B).values())
    got = ev.read(config.OUT_DIR / A)
    assert got["waiting"] == len(SUMM) and got["judge_error"].endswith("RuntimeError: boom")
    assert {it["reason"] for it in got["items"]} == {
        "the judge's replies couldn't be taken in: RuntimeError: boom"}


def test_asking_again_failing_leaves_this_models_verdicts_and_says_why(svc, monkeypatch):
    sent = remark((A,))
    monkeypatch.setattr(ev, "_ask_again", lambda *a, **k: (_ for _ in ()).throw(
        OSError("the judge is unreachable")))
    ev.finish_remark(config.OUT_DIR, replies(sent[A], **{FIRST: llm.Result(text="not json")}))
    got = items(A)
    assert got[FIRST]["pass"] is None and got[FIRST]["reason"] == \
        "asking the judge again failed: OSError: the judge is unreachable"
    assert all(it["pass"] is not None for i, it in got.items() if i != FIRST)
    assert ev.read(config.OUT_DIR / A)["waiting"] == 1


def test_a_reply_the_cap_cut_off_is_asked_again_with_twice_the_room(svc):
    sent = remark((A,))
    ev.finish_remark(config.OUT_DIR, replies(sent[A], **{
        FIRST: llm.Result(text=CUT, finish="length"), SECOND: llm.Result(text="not json")}))
    got = items(A)
    assert got[FIRST]["reason"] == ev.ASKED_AGAIN_CUT and got[SECOND]["reason"] == ev.ASKED_AGAIN
    again = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
             if ":retry:" in r["custom_id"]]
    caps = {r["custom_id"].rsplit(":", 1)[1]: r["max_tokens"] for r in again}
    assert caps == {FIRST: 2 * ev.CHECKLIST_TOKENS, SECOND: ev.CHECKLIST_TOKENS}
    assert all(r["schema"] for r in again)
    # cut off again: said so, and waiting
    cid = next(r["custom_id"] for r in again if r["custom_id"].endswith(FIRST))
    ev.finish_remark(config.OUT_DIR, {cid: llm.Result(text=CUT, finish="length")})
    assert items(A)[FIRST]["reason"] == ev.CUT_OFF and items(A)[FIRST]["pass"] is None


def test_a_request_that_failed_says_so_and_is_not_asked_again(svc):
    sent = remark((A,))
    ev.finish_remark(config.OUT_DIR, replies(sent[A], **{
        FIRST: llm.Result(error="HTTP 400: this model's maximum context length is 8192")}))
    assert items(A)[FIRST]["reason"] == ("the judge's request failed: HTTP 400: this model's "
                                         "maximum context length is 8192")
    assert not [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
                if ":retry:" in r["custom_id"]]


def test_compare_counts_waiting_alike_and_says_why(svc):
    sent = remark()
    ev.finish_remark(config.OUT_DIR, replies(sent[A] + sent[B], **{
        FIRST: llm.Result(error="HTTP 500")}))
    # as a failed batch left them: no verdicts, and the stored count zeroed
    d = config.OUT_DIR / B
    out = ev.read(d)
    for it in out["items"]:
        if it["id"] == SECOND:
            it.update({"pass": None, "reason": "not marked: the judge failed — run everyday "
                                               "tasks again"})
    out["waiting"] = 0
    ev.write(d, out)
    table = ev.compare(config.OUT_DIR)
    first = {x.split(" | ")[0].strip("| "): x for x in table.splitlines()
             if x.startswith("| org/")}
    assert first["org/a"].endswith("| 1 |") and first["org/b"].endswith("| 1 |")
    assert "Waiting, by why (every Everyday answer with no verdict yet):" in table
    assert "- org/a: 1 the judge's request failed: HTTP 500" in table
    assert "- org/b: 1 not marked: the judge failed — run everyday tasks again" in table
    assert FIRST not in table and SECOND not in table                  # counted, never named


def test_the_local_backend_says_why_a_reply_ended(tmp_path):
    b = llm.LocalOpenAI.__new__(llm.LocalOpenAI)
    b.dir = tmp_path / "llm_batches" / "local"
    d = b.dir / "local_0123456789ab"
    d.mkdir(parents=True)
    (d / "requests.jsonl").write_text(json.dumps({"custom_id": "x"}) + "\n"
                                      + json.dumps({"custom_id": "y"}) + "\n")
    (d / "results.jsonl").write_text(
        json.dumps({"custom_id": "x", "text": CUT, "error": "", "finish_reason": "length"}) + "\n"
        + json.dumps({"custom_id": "y", "text": "{}", "error": "", "finish_reason": "stop"}) + "\n")
    got = b.fetch("local_0123456789ab")
    assert (got["x"].finish, got["y"].finish) == ("length", "stop")
