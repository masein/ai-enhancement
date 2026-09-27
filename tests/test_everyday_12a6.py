"""12a.6: Summarise, one group again, marked by the judge on a rubric.

On live answers Shorten passed 10 of 111 and Summarise 24 of 57, nearly every
failure the word limit alone: the limit counted the whole reply, some limits
were never asked for, fact lists wanted exact strings and a time written
"1100" read as an invented number. Now "Shorten a message" is merged into
"Summarise" (the fifteen clearest kept, 48 retired), every Summarise answer is
marked by the judge on a rubric scoring 0 to 4 and passing at 3 — one summary,
the key facts by meaning, nothing invented, a length only when the request
states one — and numbers_from_source stays as the one script gate, reading
"11:00", "1100" and "11 am" as the same time. Re-marking every stored answer
needs no model run: --judge sends what waits on the judge, --compare prints
before and after, model by model."""

from __future__ import annotations

import json
import sys

import pytest

import everyday as ev
from conftest import make_service
from service import config, db, llm, llm_poller
from test_everyday_12a5 import _asked

BANK = {q["id"]: q for q in ev.load_bank()}
SUMMARISE = [q for q in BANK.values() if q["group"] == "summarising"]
KEPT = ["everyday-pilot-03"] + [f"everyday-summarising-{n:02d}" for n in
                                (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
RETIRED = [json.loads(x) for x in ev.RETIRED_PATH.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------

def test_shorten_is_merged_into_summarise_and_the_rest_retired():
    assert "shorten" not in ev.groups() and ev.groups()["summarising"] == "Summarise"
    assert len(SUMMARISE) == 60 and all(k in BANK and BANK[k]["group"] == "summarising" for k in KEPT)
    # 48 retired, each with the day and why, and none in the bank
    assert len(RETIRED) == 48 and not {q["id"] for q in RETIRED} & set(BANK)
    assert {q["retired"] for q in RETIRED} == {"2026-09-27"} and all(q["why"] for q in RETIRED)
    assert next(q for q in RETIRED if q["id"] == "everyday-summarising-16")["why"].startswith(
        "a recipe")
    # a new bank version: the wording hash moved, the date with it
    assert ev.version()["date"] == "2026-09-27"


def test_a_question_written_for_shorten_reads_as_summarise(tmp_path):
    f = tmp_path / "bank.jsonl"
    f.write_text(json.dumps({"id": "b1", "group": "shorten", "prompt": 'tldr: "Bins go out on '
                             'Thursday, not Friday, this week."', "reference": "Bins: Thursday.",
                             "checks": [{"type": "contains_any", "values": ["thursday"]},
                                        {"type": "max_words", "n": 8}]}) + "\n",
                 encoding="utf-8")
    [q] = ev.load_bank(f)
    assert q["group"] == "summarising"
    assert [c["type"] for c in q["checks"]] == ["numbers_from_source", "judge"]
    assert "- thursday" in q["checks"][1]["rubric"]


def test_every_summarise_question_has_the_gate_and_the_rubric():
    for q in SUMMARISE:
        gate, j = q["checks"]
        assert gate == {"type": "numbers_from_source"}, q["id"]
        assert (j["type"], j["scale"], j["pass_at"]) == ("judge", 4, 3), q["id"]
        # its reference passes the gate: a good summary gives no number the text doesn't
        assert ev.run_check(gate, q["reference"], q["prompt"])[0] is True, q["id"]
        assert f'"{q["reference"]}"' in j["rubric"]


# ---------------------------------------------------------------------------
# the rubric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,said", [
    ('summarise this in 2 sentences: "…"', "in 2 sentences"),
    ('summarise this email under 60 words pls: "…"', "under 60 words"),
    ('shorten this to one line pls: "…"', "to one line"),
    ('give me 3 bullet points from this: "…"', "3 bullet points"),
    ('tldr pls: "…"', "shorter"),
    ('make this building notice short: "…"', "shorter"),
    ('can u summarise this email thread for me: "… keep it under 50 words …"', ""),
    ('summarise this handover note: "…"', ""),
])
def test_a_length_counts_only_if_the_request_states_one(prompt, said):
    """the words before the pasted text: a limit inside the text is not asked"""
    assert ev.stated_length(prompt) == said


def test_the_rubric_says_what_scores_in_plain_words():
    tldr = ev.judge_check(BANK["everyday-pilot-03"])["rubric"]
    assert tldr.startswith("Score the answer from 0 to 4 as a summary")
    assert "It passes at 3 or more." in tldr
    assert "versions or options to choose from" in tldr and "lead-in" in tldr
    assert "judged by meaning and not by exact words" in tldr
    assert "It should keep all 2 of these:\n- 11:30\n- thursday" in tldr
    assert "Take off 2 if it says anything the text doesn't" in tldr
    assert "The request asks for it shorter" in tldr
    # a request that states no length is never marked on it
    email = next(q for q in SUMMARISE if q["prompt"].startswith("can u summarise this email thread"))
    assert "The request states no length: take nothing off for length." in \
        ev.judge_check(email)["rubric"]
    # a facts check with n: "at least n of these k"
    facts = next(c for c in ev._facts_of(json.loads(next(
        x for x in (ev.REPO / "docs" / "prompts" / "phase-12a5" / "everyday_bank_12a5.jsonl")
        .read_text(encoding="utf-8").splitlines() if email["id"] in x)))[0:1])
    assert facts and "at least 5 of these" in ev.judge_check(email)["rubric"]
    # the page says what the rubric asks, not the whole of it
    assert ev.describe(ev.judge_check(email)) == (
        "the judge, on a rubric (0 to 4, passing at 3): one summary, the key facts, nothing "
        "invented, and the length only if the request asks one")


@pytest.mark.parametrize("reply,want", [
    ('{"score": 4, "reason": "all the facts, one summary"}',
     {"pass": True, "score": 4, "scale": 4, "reason": "4 of 4: all the facts, one summary"}),
    ('{"score": 3, "reason": "a lead-in"}',
     {"pass": True, "score": 3, "scale": 4, "reason": "3 of 4: a lead-in"}),
    ('{"score": "2", "reason": "two options, a fact missing"}',
     {"pass": False, "score": 2, "scale": 4, "reason": "2 of 4: two options, a fact missing"}),
    ('{"score": 0}', {"pass": False, "score": 0, "scale": 4,
                      "reason": "0 of 4: the judge took points off"}),
    ('{"score": 7}', None), ('{"score": true}', None), ("nothing", None),
])
def test_the_judges_score_is_read_against_the_line(reply, want):
    assert ev.parse_verdict(reply, ev.judge_check(BANK["everyday-summarising-02"])) == want


# ---------------------------------------------------------------------------
# the gate: numbers, and times read as times
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src,ans,ok", [
    ("The meeting is at 11:00 in room 4.", "Meeting 1100, room 4.", True),
    ("The meeting is at 11 am in room 4.", "Meeting at 1100 hrs, room 4.", True),
    ("Report at 1100 to room 4.", "Be in room 4 at 11 am.", True),
    ("Report at 1100 to room 4.", "Be in room 4 at 11:00.", True),
    ("The meeting is at 11:00 in room 4.", "Meeting 1400, room 4.", False),
    ("Pay 2,450 by 12 Oct.", "Pay 2450 by 12 Oct at 1100.", False),
    ("Balance is $8,320.75.", "Balance: $8320.75.", True),
])
def test_a_time_written_as_four_digits_is_the_time(src, ans, ok):
    assert ev.run_check({"type": "numbers_from_source"}, ans, src)[0] is ok


# ---------------------------------------------------------------------------
# marking: a verdict is the judge's, on this rubric, or none
# ---------------------------------------------------------------------------

def test_a_script_mark_from_before_is_never_taken_for_the_judges(tmp_path):
    q = BANK["everyday-summarising-01"]
    mdir = tmp_path / "org__m"
    _asked(mdir, [q], answer=lambda q: q["reference"])
    # what 12a.5's checks wrote: failed, the word limit
    ev.write(mdir, {"model": "org/m", "items": [{"id": q["id"], "pass": False,
                                                 "answer_text": q["reference"],
                                                 "reason": "31 words, at most 30"}]})
    [it] = ev.mark(mdir)["items"]
    assert it["pass"] is None and it["reason"] == ev.WAITING
    # the judge's verdict on this rubric is kept while the answer is the same…
    out = ev.mark(mdir, {q["id"]: {"pass": True, "score": 4, "reason": "4 of 4: fine"}})
    ev.write(mdir, out)
    [it] = ev.mark(mdir)["items"]
    assert (it["pass"], it["score"], it["rubric"]) == (True, 4, ev.rubric_key(q))
    # …and asked again when the rubric changes
    before = ev.read(mdir)
    before["items"][0]["rubric"] = "an older rubric"
    ev.write(mdir, before)
    assert ev.mark(mdir)["items"][0]["pass"] is None


# ---------------------------------------------------------------------------
# re-marking every stored answer, with the judge, no model runs
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client
    client.__exit__(None, None, None)


def _two_models(root):
    """two models that answered today's bank, marked as 12a.5 marked them: the
    short summaries failed on the word limit"""
    for name in ("org__a", "org__b"):
        d = root / name
        _asked(d, list(BANK.values()))
        out = ev.mark(d)
        for it in out["items"]:
            if it["group"] == "summarising":
                it["pass"], it["reason"] = False, "70 words, at most 30"
        out["groups"] = {"shorten": {"passed": 1, "total": 13}, "summarising": {"passed": 5,
                                                                                "total": 18}}
        out["passed"], out["total"], out["waiting"] = 120, 200, 0
        ev.write(d, out)


def test_the_re_mark_sends_summarise_to_the_judge_and_compares(svc, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    root = tmp_path / "full"                      # its own results folder: these two models
    _two_models(root)
    monkeypatch.setattr(sys, "argv", ["everyday.py", str(root), "--judge"])
    assert ev.main() == 0
    out = capsys.readouterr().out
    assert "marked 2 model(s)" in out and "sent the judge " in out
    # the stand-in judge marks them at once: nothing waits
    a = ev.read(root / "org__a")
    assert a["waiting"] == 0 and a["passed"] == a["total"] == 179
    assert all(it["score"] == 4 for it in a["items"] if it["group"] == "summarising")
    # the marks from before are kept, once
    before = json.loads((root / ev.BEFORE_NAME).read_text())["models"]
    assert before["org/a"] == {"passed": 120, "total": 200, "waiting": 0,
                               "groups": {"shorten": {"passed": 1, "total": 13},
                                          "summarising": {"passed": 5, "total": 18}}}
    ev.remark(root, judge=True)
    assert json.loads((root / ev.BEFORE_NAME).read_text())["models"] == before
    # before and after, model by model: a table for the PR
    monkeypatch.setattr(sys, "argv", ["everyday.py", str(root), "--compare"])
    assert ev.main() == 0
    table = capsys.readouterr().out.strip().splitlines()
    assert table[0] == ("| model | before | after | Summarise before | Summarise after | "
                        "waiting |")
    assert table[2] == "| org/a | 120 of 200 | 179 of 179 | 6 of 31 | 31 of 31 | 0 |"


def test_with_a_judge_elsewhere_one_batch_goes_and_the_poller_lands_it(svc, monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    root = config.OUT_DIR
    _two_models(root)
    res = ev.remark(root, {"org__a", "org__b"}, judge=True)
    judged = [q for q in BANK.values() if ev.judge_check(q)]
    assert res["sent"] == 2 * len(judged) and res["batch_id"]
    row = next(b for b in db.batches_list(50) if b["batch_id"] == res["batch_id"])
    assert (row["kind"], row["n_items"], row["status"]) == ("everyday_remark", res["sent"],
                                                            "submitted")
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith("everyday-remark:")]
    assert len(sent) == res["sent"]
    assert {r["custom_id"].split(":")[1] for r in sent} == {"org__a", "org__b"}
    assert ev.read(root / "org__a")["waiting"] == len(judged)
    llm_poller.tick()
    for name in ("org__a", "org__b"):
        out = ev.read(root / name)
        assert out["waiting"] == 0 and out["passed"] == out["total"] == 179


def test_a_re_mark_batch_that_fails_says_so_on_each_model(svc, monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    root = config.OUT_DIR
    _two_models(root)
    res = ev.remark(root, {"org__a", "org__b"}, judge=True)
    row = next(b for b in db.batches_list(50) if b["batch_id"] == res["batch_id"])
    llm_poller._mark_failed(row, "the provider refused the batch")
    for name in ("org__a", "org__b"):
        out = ev.read(root / name)
        assert out["waiting"] == 0 and out["judge_error"] == "the provider refused the batch"
        assert {it["reason"] for it in out["items"] if it["pass"] is None} == {
            "not marked: the judge failed — run everyday tasks again"}


def test_the_judge_test_reads_a_rubrics_score(monkeypatch):
    from service import judge_test
    a = {"kind": "everyday", "task": "everyday-summarising-02", "answer": "x"}
    assert judge_test.read_mark(a, '{"score": 3, "reason": "a lead-in"}') == 3
    assert judge_test.read_mark(a, '{"pass": false, "reason": "no"}') == 1


def test_the_re_mark_runs_as_the_server_runs_it(tmp_path):
    """`python scripts/everyday.py …/full --judge` from a shell: the script
    reaches the service's judge and database from scripts/ on its own"""
    import os
    import subprocess
    root = tmp_path / "full"
    _asked(root / "org__a", [BANK["everyday-summarising-02"]], answer=lambda q: q["reference"])
    env = {**os.environ, "JUDGE_MODEL": "stub", "BENCH_ROOT": str(tmp_path), "PYTHONPATH": ""}
    p = subprocess.run([sys.executable, str(ev.REPO / "scripts" / "everyday.py"), str(root),
                        "--judge"], capture_output=True, text=True, cwd=tmp_path, env=env,
                       timeout=120)
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[-1] == "sent the judge 1 answer(s)"
    [it] = ev.read(root / "org__a")["items"]
    assert it["pass"] is True and it["score"] == 4
