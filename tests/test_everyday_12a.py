"""12a: Everyday tasks, off the page — since 12a.3 the bank of 333 questions.

Questions typed the way people type into an assistant on a phone, marked by
their checks on the text after any thinking (#59's split). Each check says
pass or fail and one reason in plain words (the vocabulary itself is
tests/test_everyday_12a2.py's). A model with no chat
template is refused before any GPU is spent. And the pilot is a look, not a
benchmark: nothing that ranks, averages, proposes or generates ever reads it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

import everyday as ev
import make_fixture
import report_lm_eval as report
from conftest import make_service
from service import config, db, llm, llm_poller, runner

MODEL = "fx/one-option-70m"          # a fixture model Everyday tasks have not asked yet


# the TL;DR (03), in one sentence with the time and the day: the stand-in passes it
TLDR = "School closes early at 11:30 on Thursday; buses leave at 11:15."
FULL = '{"name": "Sara Ahmed", "age": 34, "role": "product manager", "city": "Toronto", ' \
       '"joined": "March 2021"}'
BANK = ev.load_bank()
Q = {q["id"]: q for q in BANK}
JUDGED = [q["id"] for q in BANK if any(c["type"] == "judge" for c in q["checks"])]


def test_the_judges_questions_and_its_stand_in():
    """12b.3: English only — the TL;DR's judge reads the rubric the question
    carries; nothing is written for Arabic. 12a.3: eleven questions carry a
    judge check (the TL;DR, round 2's five and round 3's five), each beside
    script checks."""
    assert len(JUDGED) == 11 and "everyday-pilot-03" in JUDGED
    q = Q["everyday-pilot-03"]
    assert q["prompt"].startswith("tldr pls:")
    p = ev.judge_prompt(q, TLDR)
    rubric = next(c["rubric"] for c in q["checks"] if c["type"] == "judge")
    assert rubric in p and q["prompt"] in p and TLDR in p
    assert "arabic" not in p.lower()
    assert ev.parse_verdict(ev.stub_reply(p)) == {
        "pass": True, "reason": "closes 11:30 on Thursday, in two sentences or fewer"}
    assert ev.stub_verdict("School closes early on Thursday.")["pass"] is False
    assert ev.stub_verdict("Thursday: closes at 11:30. Buses at 11:15. Pickup by 11:45.") == {
        "pass": False, "reason": "more than two sentences"}
    assert ev.parse_verdict('{"pass": "yes"}') is None           # a reply it cannot read


# ---------------------------------------------------------------------------
# marking reads the answer after the thinking, and only that
# ---------------------------------------------------------------------------

def write_bank(mdir: Path, answers: dict[str, str], budget: int = 512) -> None:
    """What the harness writes for Everyday tasks: samples and a results file,
    for the questions in `answers` (question id -> answer)."""
    d = mdir / f"{ev.TASK}_0shot" / "fx"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"samples_{ev.TASK}_2026.jsonl", "w", encoding="utf-8") as fh:
        for i, (qid, a) in enumerate(answers.items()):
            fh.write(json.dumps({"doc_id": i, "doc": Q[qid], "resps": [[a]], "filtered_resps": [a]},
                                ensure_ascii=False) + "\n")
    (d / "results_2026.json").write_text(json.dumps({
        "results": {ev.TASK: {"alias": ev.TASK, "bypass,none": 999}},
        "configs": {ev.TASK: {"generation_kwargs": {"until": ["\n\n\n\n"],
                                                   "max_gen_toks": 512}}},
        "config": {"model_args": "pretrained=fx/x",
                   **({"gen_kwargs": f"max_gen_toks={budget}"} if budget != 512 else {})}}),
        encoding="utf-8")


def pilot(*answers) -> dict[str, str]:
    return {f"everyday-pilot-0{i}": a for i, a in enumerate(answers, 1)}


def test_an_unfinished_answer_fails_every_check_with_one_reason(tmp_path):
    mdir = tmp_path / "fx__thinker"
    write_bank(mdir, pilot(*["<think>\nlet me think about february, it has 29 days in a leap "
                             "year, and the json would be {\"name\": \"Sara Ahmed\"}"] * 5),
               budget=2048)
    out = ev.mark(mdir)
    assert [(it["pass"], it["reason"]) for it in out["items"]] == \
        [(False, "never finished answering")] * 5
    assert all(it["no_answer"] and it["had_reasoning"] for it in out["items"])
    # the judge is not asked about an answer that never came
    assert out["waiting"] == 0 and out["passed"] == 0
    assert out["settings"]["override"] == "max_gen_toks=2048"


def test_the_thinking_is_never_marked(tmp_path):
    """#59's split, reused: "29" in the thinking is not "29" in the answer, and
    a mistake the thinking quotes is not a mistake the answer makes."""
    mdir = tmp_path / "fx__thinker"
    write_bank(mdir, pilot(
        "<think>a leap year has 29 days in february</think>\n\nI am not sure.",
        "<think>ok</think>\n\n" + FULL,
        "<think>the notice</think>\n\n" + TLDR,
        "<think>'payed' should be 'paid', 'sended' should be 'sent'</think>\n\nDear Sir, I am "
        "writing to you regarding the invoice which was sent last week and has still not been "
        "paid.",
        "<think>names: Bean There: cozy</think>\n\nBean There\nDaily Grind\nBrew Haven"))
    out = ev.mark(mdir)
    got = {it["id"][-2:]: (it["pass"], it["reason"]) for it in out["items"]}
    assert got["01"] == (False, "never mentions: 29, twenty-nine, twenty nine")   # 12a.4
    assert got["02"][0] is True and got["02"][1].startswith("valid JSON with sara")
    assert got["03"] == (None, "waiting for the judge")
    assert got["04"][0] is True
    assert got["05"] == (True, "3 lines · at most 15 words")
    first = out["items"][0]
    assert first["answer_text"] == "I am not sure."
    assert first["reasoning_text"] == "a leap year has 29 days in february"
    assert first["reasoning_words"] == 8
    # the groups say n of k, k being what this model was asked in each
    assert out["groups"]["understanding"] == {"passed": 0, "total": 1}
    assert out["groups"]["instructions"] == {"passed": 1, "total": 1}
    assert "quick_maths" not in out["groups"]


def test_a_script_check_that_fails_is_never_sent_to_the_judge(tmp_path):
    """A judged question also has script checks; the judge only decides once
    they pass (the TL;DR without 11:30 fails before any judge)."""
    mdir = tmp_path / "fx__x"
    write_bank(mdir, pilot("29", FULL, "School closes early on Thursday.", "x", "a\nb\nc"))
    it = ev.mark(mdir)["items"][2]
    assert it["pass"] is False and it["reason"] == "missing: 11:30"


def test_a_verdict_is_kept_while_the_answer_is_the_same(tmp_path):
    """12a.4: on a run of the whole bank — only this wording's are re-marked"""
    mdir = tmp_path / "fx__x"
    refs = {q["id"]: q["reference"] for q in BANK}

    def tldr(out):
        return next(it for it in out["items"] if it["id"] == "everyday-pilot-03")
    write_bank(mdir, {**refs, **pilot("29", FULL, TLDR, "x", "a\nb\nc")})
    ev.write(mdir, ev.mark(mdir, {"everyday-pilot-03": {"pass": True, "reason": "fine"}}))
    again = ev.mark(mdir)                                   # re-marked from the logs
    assert tldr(again)["pass"] is True and tldr(again)["reason"] == "fine"
    write_bank(mdir, {**refs, **pilot("29", FULL, TLDR + " Friday is normal.", "x", "a\nb\nc")})
    assert tldr(ev.mark(mdir))["pass"] is None              # a new answer: asked again


def test_a_model_that_sat_the_pilot_keeps_its_five_marks(tmp_path):
    """The pilot's answers were logged as "everyday_pilot"; they are read, and
    the model shows the five it was asked. 12a.4: the pilot was an earlier
    bank, so they are an earlier wording's — kept as they were marked, and
    never marked again by today's checks"""
    mdir = tmp_path / "fx__old"
    write_bank(mdir, pilot("29", FULL, TLDR, "x", "a\nb\nc"))
    (mdir / f"{ev.TASK}_0shot").rename(mdir / "everyday_pilot_0shot")
    out = ev.mark(mdir, {"everyday-pilot-03": {"pass": True, "reason": "fine"}})
    assert [it["id"] for it in out["items"]] == [f"everyday-pilot-0{i}" for i in (1, 4, 3, 2, 5)]
    assert out["total"] == 5 and out["passed"] == 4
    assert out["earlier"] is True and out["version"]["hash"] != ev.version()["hash"]
    ev.write(mdir, out)
    again = ev.mark(mdir, {"everyday-pilot-03": {"pass": False, "reason": "changed"}})
    assert again["items"] == out["items"] and again["passed"] == 4


# ---------------------------------------------------------------------------
# running it
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def queue_pilot(client, hf_id=MODEL):
    r = client.post("/api/submissions", json={"hf_id": hf_id, "suite": "everyday"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# every question answered with its reference, but the pilot's email left
# unfixed: 332 of 333 once the judge has agreed with the eleven it marks
DEFAULT_ANSWERS = {**{q["id"]: q["reference"] for q in BANK}, "everyday-pilot-04": "I writing"}


def fake_gpu(monkeypatch, has_template=True, reasoning=False, answers=None):
    """The runner with the harness replaced by one that writes `answers`."""
    monkeypatch.setattr(runner, "preflight", lambda hf_id, kind, **k: {
        "kind": kind, "params": 70_000_000, "vocab": 50304, "batch": 8, "need_gb": 2.0,
        "remote_code": False, "has_template": has_template,
        "archinfo": {"reasoning_template": True} if reasoning else {}})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    seen = []

    def run(sid, cmd, *a, **k):
        seen.append(cmd)
        out = Path(cmd[cmd.index("--output_path") + 1])
        write_bank(out.parent, answers or DEFAULT_ANSWERS,
                   budget=config.EVERYDAY_REASONING_MAX_GEN_TOKS if reasoning else 512)
        return 0
    monkeypatch.setattr(runner, "_run_task", run)
    return seen


def test_a_model_with_no_chat_template_is_refused_at_preflight(svc, monkeypatch):
    client, _, _ = svc
    seen = fake_gpu(monkeypatch, has_template=False)
    sid = queue_pilot(client)
    runner.run_submission(db.get(sid))
    row = db.get(sid)
    assert row["status"] == "failed"
    assert row["error"] == ("This model has no chat template, so it can't be asked questions "
                            "the way a person would.")
    assert seen == []                                         # no GPU second spent


@pytest.mark.parametrize("reasoning", [False, True])
def test_the_bank_is_asked_through_the_chat_template_and_marked_in_the_same_run(
        svc, monkeypatch, reasoning):
    """Whatever kind the board lists: fx/one-option-70m is a base model
    here, and is asked through its template all the same. Its listed kind
    does not change. A reasoning model gets #59's room."""
    client, _, tree = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    mdir = tree["models"][MODEL]["dir"]
    meta_before = (mdir / "model_meta.json").read_text(encoding="utf-8")
    seen = fake_gpu(monkeypatch, reasoning=reasoning)
    sid = queue_pilot(client)
    runner.run_submission(db.get(sid))
    [cmd] = seen
    assert cmd[cmd.index("--tasks") + 1] == "everyday"
    assert "--apply_chat_template" in cmd
    assert cmd[cmd.index("--include_path") + 1] == str(config.EVERYDAY_TASKS_DIR)
    if reasoning:
        # 12a.4: 4,096 for its everyday answers; the exam's 2,048 is the exam's
        assert config.EVERYDAY_REASONING_MAX_GEN_TOKS == 4096 != config.REASONING_MAX_GEN_TOKS
        assert cmd[cmd.index("--gen_kwargs") + 1] == "max_gen_toks=4096"
    else:
        assert "--gen_kwargs" not in cmd
    # the task it ran, written from the bank and the template: the question as
    # typed, no "Answer:" suffix, 512 tokens unless the model thinks
    y = [ln.strip() for ln in (config.EVERYDAY_TASKS_DIR / "everyday.yaml")
         .read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    assert 'doc_to_text: "{{prompt}}"' in y and not any("Answer:" in ln for ln in y)
    assert "max_gen_toks: 512" in y
    assert f"test: {config.EVERYDAY_TASKS_DIR / 'everyday.jsonl'}" in y
    assert len((config.EVERYDAY_TASKS_DIR / "everyday.jsonl").read_text(encoding="utf-8")
               .splitlines()) == 333
    # marked straight after, the judge's question too (the stub is in-process)
    out = json.loads((mdir / "everyday.json").read_text(encoding="utf-8"))
    assert out["model"] == MODEL and out["passed"] == 332 and out["waiting"] == 0
    row = db.get(sid)
    assert row["status"] == "done" and row["progress"] == "Everyday tasks: 332 of 333"
    assert (mdir / "model_meta.json").read_text(encoding="utf-8") == meta_before


def test_run_again_answers_again_and_keeps_the_last_answers(svc, monkeypatch):
    client, _, tree = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    mdir = tree["models"][MODEL]["dir"]
    fake_gpu(monkeypatch)
    runner.run_submission(db.get(queue_pilot(client)))
    seen = fake_gpu(monkeypatch, answers={q["id"]: "" for q in BANK})
    runner.run_submission(db.get(queue_pilot(client)))
    assert len(seen) == 1                                     # answered again, not resumed
    assert json.loads((mdir / "everyday.json").read_text(encoding="utf-8"))["passed"] == 0
    kept = list((config.OUT_DIR.with_name("earlier") / mdir.name).glob("everyday_0shot-*"))
    assert len(kept) == 1                                     # the last answers, kept whole


def test_the_judged_questions_wait_on_the_judge_and_the_row_says_so(svc, monkeypatch):
    client, appmod, tree = svc
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    fake_gpu(monkeypatch)
    sid = queue_pilot(client)
    runner.run_submission(db.get(sid))
    row = next(r for r in client.get("/api/submissions").json() if r["id"] == sid)
    assert row["status"] == "done"
    assert row["progress"] == "Everyday tasks: 321 of 333 · the judge is marking 11"
    assert row["judge"]["n_items"] == 11 and row["judge"]["progress"] == "0/11 done"
    assert row["judge"]["status"] == "submitted"
    # one request per judged question, and nothing else
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith("everyday:")]
    assert sorted(r["custom_id"] for r in sent) == sorted(f"everyday:{sid}:{q}" for q in JUDGED)
    llm_poller.tick()
    mdir = tree["models"][MODEL]["dir"]
    out = json.loads((mdir / "everyday.json").read_text(encoding="utf-8"))
    assert out["passed"] == 332 and out["waiting"] == 0
    tldr = next(it for it in out["items"] if it["id"] == "everyday-pilot-03")
    assert tldr["reason"] == "closes 11:30 on Thursday, in two sentences or fewer"
    row = next(r for r in client.get("/api/submissions").json() if r["id"] == sid)
    assert row["progress"] == "Everyday tasks: 332 of 333"
    assert row["judge"]["status"] == "done"


def test_without_a_judge_the_judged_questions_say_so(svc, monkeypatch):
    client, _, tree = svc
    fake_gpu(monkeypatch)
    sid = queue_pilot(client)
    runner.run_submission(db.get(sid))
    out = json.loads((tree["models"][MODEL]["dir"] / "everyday.json").read_text(encoding="utf-8"))
    waiting = [it for it in out["items"] if it["pass"] is None]
    assert sorted(it["id"] for it in waiting) == sorted(JUDGED)
    assert {it["reason"] for it in waiting} == {"not marked: the judge is not set up on this server"}
    assert db.get(sid)["status"] == "done"


def test_the_suite_is_accepted_and_takes_no_topics(svc):
    client, _, _ = svc
    queue_pilot(client)
    r = client.post("/api/submissions", json={"hf_id": MODEL, "suite": "everyday",
                                              "tasks": ["exam_law"]})
    assert r.status_code == 422
    assert config.tasks_for_suite("everyday") == ["everyday"]
    assert "everyday" not in config.tasks_for_suite("full")


# ---------------------------------------------------------------------------
# a look, not a benchmark
# ---------------------------------------------------------------------------

def test_the_bank_is_on_no_leaderboard_and_in_no_average(tmp_path):
    """The fixture's four Everyday models, against the same tree without
    Everyday tasks: the models, their tasks, cells and averages are identical."""
    with_it = make_fixture.build(tmp_path / "a")
    without = make_fixture.build(tmp_path / "b")
    for m in without["everyday"].values():
        for d in ("everyday_0shot", "everyday_pilot_0shot"):
            if (m["dir"] / d).exists():
                shutil.rmtree(m["dir"] / d)
        (m["dir"] / "everyday.json").unlink()

    def payload(tree):
        runs = report.load_results(tree["out_dir"])
        return report.build_payload(report.merge_runs(runs), "t", source=str(tree["out_dir"]),
                                    everyday=report.load_everyday(tree["out_dir"]))
    a, b = payload(with_it), payload(without)
    for key in ("models", "accTasks", "pplTasks", "required", "tasks", "cells", "sig"):
        assert json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True), key
        # the task's name, not the word (HellaSwag is "about everyday situations")
        assert not re.search(r'"everyday"|everyday[-_]', json.dumps(a[key])), key
    # 12a.4: the two that sat the bank are on its wording; the two that sat
    # only the pilot answered an earlier one, kept beside them
    assert sorted(a["everyday"]["models"]) == ["fx/good-750m", "fx/skewed-360m"]
    assert sorted([*a["everyday"]["models"], *a["everyday"]["earlier"]]) == \
        sorted(with_it["everyday"])
    assert b["everyday"]["models"] == {}


def test_propose_and_the_generator_never_see_the_bank(svc):
    """Over the request bodies actually sent: a proposal and a dataset for a
    model that sat Everyday tasks carry none of its questions or answers."""
    client, _, tree = svc
    pid = client.post("/api/proposals", json={"model": "fx/good-750m", "topic": "Economics",
                                              "requested_by": "masein"}).json()["id"]
    llm_poller.tick()
    client.post(f"/api/proposals/{pid}/approve", json={"approver": "masein"})
    client.post(f"/api/proposals/{pid}/generate", json={"requester": "masein", "count": 4,
                                                        "fmt": "doc"})
    llm_poller.tick()
    sent = llm.FakeBatches("fake-1", config.BENCH_ROOT).recorded()
    kinds = {r["custom_id"].split(":")[0] for r in sent}
    assert {"proposal", "gen"} <= kinds
    body = "\n".join(r["system"] + "\n" + r["user"] for r in sent)
    for q in BANK:
        assert q["prompt"] not in body and q["id"] not in body
        assert q["reference"] not in body
    for ans in make_fixture.EVERYDAY_ANSWERS["fx/good-750m"]:
        assert ans.split("</think>")[-1].strip()[:40] not in body
    assert "everyday" not in body.lower().replace("everyday situations", "")


def test_the_copy_check_knows_the_banks_questions(tmp_path):
    """The gate indexes every question the harness logged, Everyday's
    included: a generated document that repeats one is dropped like any other."""
    from service import contamination as ct
    tree = make_fixture.build(tmp_path)
    ix = ct.index(tree["out_dir"])
    # a question long enough to hold a window the gate matches on: a ten-word
    # one ("hw mch is 300 aed…") is shorter than one
    q = Q["everyday-pilot-01"]["prompt"]
    assert ix.hits(q)
    assert not ix.hits("a sentence about harbours and tides that no question on this board holds")
