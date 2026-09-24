"""12a: the Everyday tasks pilot, off the page.

Five questions typed the way people type into an assistant on a phone,
marked by a script on the text after any thinking (#59's split). Each check
says pass or fail and one reason in plain words. A model with no chat
template is refused before any GPU is spent. And the pilot is a look, not a
benchmark: nothing that ranks, averages, proposes or generates ever reads it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import everyday as ev
import make_fixture
import report_lm_eval as report
from conftest import make_service
from service import config, db, llm, llm_poller, runner

MODEL = "fx/one-option-70m"          # a fixture model the pilot has not asked yet


# ---------------------------------------------------------------------------
# each check: a pass, and each fail reason
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer, ok, why", [
    ("A leap year February has 29 days.", True, "says 29"),
    ("twenty nine days", True, "says twenty nine"),
    ("It has Twenty-Nine.", True, "says twenty-nine"),
    ("February has 28 days.", False, "didn't say 29"),
    ("That rule dates from 1929.", False, "didn't say 29"),       # a number, not a digit run
])
def test_contains(answer, ok, why):
    q = ev.load_pilot()[0]
    assert ev.check_contains(answer, q["check"]) == (ok, why)


FULL = '{"name": "Sara Ahmed", "age": 34, "role": "product manager", "city": "Dubai", ' \
       '"joined": "March 2021"}'


@pytest.mark.parametrize("answer, ok, why", [
    (f"```json\n{FULL}\n```", True, "valid JSON, all five values"),            # a code fence
    (f"```\n{FULL}\n```", True, "valid JSON, all five values"),
    (FULL, True, "valid JSON, all five values"),
    (f"Here's your JSON:\n{FULL}", True, "valid JSON, all five values, with text around it"),
    (f"Sure!\n```json\n{FULL}\n```\nLet me know.", True,
     "valid JSON, all five values, with text around it"),
    # the key names are free, the name may be split, the date may be ISO
    ('{"first": "Sara", "last": "Ahmed", "age": "34", "job": {"title": "Product Manager"}, '
     '"location": "Dubai", "start": "2021-03"}', True, "valid JSON, all five values"),
    ('{"name": "Sara Ahmed", "age": 34, "title": "product manager"}', False,
     "missing: Dubai, March 2021"),
    ("name: Sara Ahmed, age: 34", False, "not valid JSON"),
    ('{"name": "Sara Ahmed", "age": 34,}', False, "not valid JSON"),
])
def test_json(answer, ok, why):
    assert ev.check_json(answer) == (ok, why)


@pytest.mark.parametrize("answer, ok, why", [
    ("Dear Sir, I am writing to you regarding the invoice which was sent last week and "
     "has still not been paid.", True, "all four mistakes fixed"),
    ("Dear Sir, I'm writing to you about the invoice, which was sent last week and is still "
     "not paid.", True, "all four mistakes fixed"),
    ("Dear Sir, I am writing to you regarding the invoice which was sent last week and "
     "still not payed.", False, "still says 'payed'"),
    ("Dear Sir, I writing to you regard the invoice which was sended last week.", False,
     "still says 'I writing', 'sended'"),
    ("Dear Sir, I am writing to you regarding the invoice which was sent last week.", False,
     "doesn't say 'paid'"),
])
def test_fixed(answer, ok, why):
    assert ev.check_fixed(answer) == (ok, why)


@pytest.mark.parametrize("answer, ok, why", [
    ("1. Bean There\n2. Daily Grind\n3. Brew Haven", True, "three names, nothing else"),
    ("- **Bean There**\n- \"Daily Grind\"\n• Brew-Ha", True, "three names, nothing else"),
    ("Bean There\n\nDaily Grind\n\nBrew Haven\n", True, "three names, nothing else"),
    ("Bean There\nDaily Grind\nBrew Haven\nCup of Joy", False, "4 lines, expected 3"),
    ("Here are three names:\n1. Bean There\n2. Daily Grind\n3. Brew Haven", False,
     "4 lines, expected 3"),
    ("1. Bean There\n2. Daily Grind: a place for regulars\n3. Brew Haven", False,
     "line 2 explains the name"),
    ("1. **Bean There** – a cozy spot\n2. Daily Grind\n3. Brew Haven", False,
     "line 1 explains the name"),
    ("The Very Best Coffee In Town\nDaily Grind\nBrew Haven", False, "line 1 is 6 words"),
    ("Bean There", False, "1 line, expected 3"),
])
def test_lines(answer, ok, why):
    assert ev.check_lines(answer, {"n": 3, "max_words": 5}) == (ok, why)


def test_the_judges_one_question_and_its_stand_in():
    q = next(x for x in ev.load_pilot() if x["check"]["type"] == "judge")
    p = ev.judge_prompt(q, "تم نقل الاجتماع إلى يوم الخميس")
    assert "whole sentence" in p.lower() and q["check"]["reference"] in p
    assert ev.parse_verdict(ev.stub_reply(p)) == {
        "pass": True, "reason": "Arabic script, the whole sentence, Thursday"}
    assert ev.stub_verdict("The meeting is moved to Thursday.")["pass"] is False
    assert ev.parse_verdict('{"pass": "yes"}') is None           # a reply it cannot read


def test_the_five_questions_are_the_briefs():
    qs = ev.load_pilot()
    assert [q["id"] for q in qs] == [f"everyday-pilot-0{i}" for i in range(1, 6)]
    assert [q["group"] for q in qs] == ["understanding", "transform", "language", "writing",
                                        "behaviour"]
    assert qs[0]["prompt"] == "hey can u tell me hwo many days is in febuary in a leep yaer"
    assert [q["check"]["type"] for q in qs] == ["contains", "json", "judge", "fixed", "lines"]
    # all readable: no split yet, and the page may show every word
    assert not any("split" in q for q in qs)


# ---------------------------------------------------------------------------
# marking reads the answer after the thinking, and only that
# ---------------------------------------------------------------------------

def write_pilot(mdir: Path, answers: list[str], budget: int = 512) -> None:
    """What the harness writes for the pilot: samples and a results file."""
    d = mdir / f"{ev.TASK}_0shot" / "fx"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "samples_everyday_pilot_2026.jsonl", "w", encoding="utf-8") as fh:
        for i, (q, a) in enumerate(zip(ev.load_pilot(), answers)):
            fh.write(json.dumps({"doc_id": i, "doc": q, "resps": [[a]], "filtered_resps": [a]},
                                ensure_ascii=False) + "\n")
    (d / "results_2026.json").write_text(json.dumps({
        "results": {ev.TASK: {"alias": ev.TASK, "bypass,none": 999}},
        "configs": {ev.TASK: {"generation_kwargs": {"until": ["\n\n\n\n"],
                                                   "max_gen_toks": 512}}},
        "config": {"model_args": "pretrained=fx/x",
                   **({"gen_kwargs": f"max_gen_toks={budget}"} if budget != 512 else {})}}),
        encoding="utf-8")


def test_an_unfinished_answer_fails_every_check_with_one_reason(tmp_path):
    mdir = tmp_path / "fx__thinker"
    write_pilot(mdir, ["<think>\nlet me think about february, it has 29 days in a leap year, "
                       "and the json would be {\"name\": \"Sara Ahmed\"}"] * 5, budget=2048)
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
    write_pilot(mdir, [
        "<think>a leap year has 29 days in february</think>\n\nI am not sure.",
        "<think>ok</think>\n\n" + FULL,
        "<think>thursday</think>\n\nتم نقل الاجتماع إلى يوم الخميس",
        "<think>'payed' should be 'paid', 'sended' should be 'sent'</think>\n\nDear Sir, I am "
        "writing to you regarding the invoice which was sent last week and has still not been "
        "paid.",
        "<think>names: Bean There: cozy</think>\n\nBean There\nDaily Grind\nBrew Haven"])
    out = ev.mark(mdir)
    got = {it["id"][-2:]: (it["pass"], it["reason"]) for it in out["items"]}
    assert got["01"] == (False, "didn't say 29")
    assert got["02"] == (True, "valid JSON, all five values")
    assert got["03"] == (None, "waiting for the judge")
    assert got["04"] == (True, "all four mistakes fixed")
    assert got["05"] == (True, "three names, nothing else")
    first = out["items"][0]
    assert first["answer_text"] == "I am not sure."
    assert first["reasoning_text"] == "a leap year has 29 days in february"
    assert first["reasoning_words"] == 8


def test_a_verdict_is_kept_while_the_answer_is_the_same(tmp_path):
    mdir = tmp_path / "fx__x"
    write_pilot(mdir, ["29", FULL, "تم نقل الاجتماع إلى يوم الخميس", "x", "a\nb\nc"])
    ev.write(mdir, ev.mark(mdir, {"everyday-pilot-03": {"pass": True, "reason": "fine"}}))
    again = ev.mark(mdir)                                   # re-marked from the logs
    assert again["items"][2]["pass"] is True and again["items"][2]["reason"] == "fine"
    write_pilot(mdir, ["29", FULL, "something else", "x", "a\nb\nc"])
    assert ev.mark(mdir)["items"][2]["pass"] is None        # a new answer: asked again


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
        write_pilot(out.parent, answers or ["29", FULL, "تم نقل الاجتماع إلى يوم الخميس",
                                            "I writing", "a\nb\nc"],
                    budget=config.REASONING_MAX_GEN_TOKS if reasoning else 512)
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
def test_the_pilot_asks_through_the_chat_template_and_marks_in_the_same_run(
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
    assert cmd[cmd.index("--tasks") + 1] == "everyday_pilot"
    assert "--apply_chat_template" in cmd
    assert cmd[cmd.index("--include_path") + 1] == str(config.EVERYDAY_TASKS_DIR)
    if reasoning:
        assert cmd[cmd.index("--gen_kwargs") + 1] == f"max_gen_toks={config.REASONING_MAX_GEN_TOKS}"
    else:
        assert "--gen_kwargs" not in cmd
    # the task it ran, written from the pilot and the template: the question as
    # typed, no "Answer:" suffix, 512 tokens unless the model thinks
    y = [ln.strip() for ln in (config.EVERYDAY_TASKS_DIR / "everyday_pilot.yaml")
         .read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    assert 'doc_to_text: "{{prompt}}"' in y and not any("Answer:" in ln for ln in y)
    assert "max_gen_toks: 512" in y
    assert f"test: {config.EVERYDAY_TASKS_DIR / 'everyday_pilot.jsonl'}" in y
    # marked straight after, the judge's question too (the stub is in-process)
    out = json.loads((mdir / "everyday.json").read_text(encoding="utf-8"))
    assert out["model"] == MODEL and out["passed"] == 4 and out["waiting"] == 0
    row = db.get(sid)
    assert row["status"] == "done" and row["progress"] == "Everyday pilot: 4 of 5"
    assert (mdir / "model_meta.json").read_text(encoding="utf-8") == meta_before


def test_run_again_answers_again_and_keeps_the_last_answers(svc, monkeypatch):
    client, _, tree = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    mdir = tree["models"][MODEL]["dir"]
    fake_gpu(monkeypatch)
    runner.run_submission(db.get(queue_pilot(client)))
    seen = fake_gpu(monkeypatch, answers=["28", "{}", "", "", ""])
    runner.run_submission(db.get(queue_pilot(client)))
    assert len(seen) == 1                                     # answered again, not resumed
    assert json.loads((mdir / "everyday.json").read_text(encoding="utf-8"))["passed"] == 0
    kept = list((config.OUT_DIR.with_name("earlier") / mdir.name).glob("everyday_pilot_0shot-*"))
    assert len(kept) == 1                                     # the last answers, kept whole


def test_the_arabic_question_waits_on_the_judge_and_the_row_says_so(svc, monkeypatch):
    client, appmod, tree = svc
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    fake_gpu(monkeypatch)
    sid = queue_pilot(client)
    runner.run_submission(db.get(sid))
    row = next(r for r in client.get("/api/submissions").json() if r["id"] == sid)
    assert row["status"] == "done"
    assert row["progress"] == "Everyday pilot: 3 of 5 · the judge is marking 1"
    assert row["judge"]["n_items"] == 1 and row["judge"]["progress"] == "0/1 done"
    assert row["judge"]["status"] == "submitted"
    # one request, about the Arabic answer, and nothing else
    sent = [r for r in llm.FakeBatches("fake-judge", config.BENCH_ROOT).recorded()
            if r["custom_id"].startswith("everyday:")]
    assert len(sent) == 1 and sent[0]["custom_id"] == f"everyday:{sid}:everyday-pilot-03"
    llm_poller.tick()
    mdir = tree["models"][MODEL]["dir"]
    out = json.loads((mdir / "everyday.json").read_text(encoding="utf-8"))
    assert out["passed"] == 4 and out["waiting"] == 0
    assert out["items"][2]["reason"] == "Arabic script, the whole sentence, Thursday"
    row = next(r for r in client.get("/api/submissions").json() if r["id"] == sid)
    assert row["progress"] == "Everyday pilot: 4 of 5"
    assert row["judge"]["status"] == "done"


def test_without_a_judge_the_arabic_question_says_so(svc, monkeypatch):
    client, _, tree = svc
    fake_gpu(monkeypatch)
    sid = queue_pilot(client)
    runner.run_submission(db.get(sid))
    out = json.loads((tree["models"][MODEL]["dir"] / "everyday.json").read_text(encoding="utf-8"))
    assert out["items"][2]["pass"] is None
    assert out["items"][2]["reason"] == "not marked: the judge is not set up on this server"
    assert db.get(sid)["status"] == "done"


def test_the_suite_is_accepted_and_takes_no_topics(svc):
    client, _, _ = svc
    queue_pilot(client)
    r = client.post("/api/submissions", json={"hf_id": MODEL, "suite": "everyday",
                                              "tasks": ["exam_law"]})
    assert r.status_code == 422
    assert config.tasks_for_suite("everyday") == ["everyday_pilot"]
    assert "everyday_pilot" not in config.tasks_for_suite("full")


# ---------------------------------------------------------------------------
# a look, not a benchmark
# ---------------------------------------------------------------------------

def test_the_pilot_is_on_no_leaderboard_and_in_no_average(tmp_path):
    """The fixture's four pilot models, against the same tree without the
    pilot: the models, their tasks, cells and averages are identical."""
    with_it = make_fixture.build(tmp_path / "a")
    without = make_fixture.build(tmp_path / "b")
    for m in without["everyday"].values():
        shutil.rmtree(m["dir"] / "everyday_pilot_0shot")
        (m["dir"] / "everyday.json").unlink()

    def payload(tree):
        runs = report.load_results(tree["out_dir"])
        return report.build_payload(report.merge_runs(runs), "t", source=str(tree["out_dir"]),
                                    everyday=report.load_everyday(tree["out_dir"]))
    a, b = payload(with_it), payload(without)
    for key in ("models", "accTasks", "pplTasks", "required", "tasks", "cells", "sig"):
        assert json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True), key
        assert "everyday_pilot" not in json.dumps(a[key]), key
    assert sorted(a["everyday"]["models"]) == sorted(with_it["everyday"])
    assert b["everyday"]["models"] == {}


def test_propose_and_the_generator_never_see_the_pilot(svc):
    """Over the request bodies actually sent: a proposal and a dataset for a
    model that sat the pilot carry none of its questions or answers."""
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
    for q in ev.load_pilot():
        assert q["prompt"] not in body and q["id"] not in body
    for ans in make_fixture.EVERYDAY_ANSWERS["fx/good-750m"]:
        assert ans.split("</think>")[-1].strip()[:40] not in body
    assert "everyday" not in body.lower().replace("everyday situations", "")


def test_the_copy_check_knows_the_pilots_questions(tmp_path):
    """The gate indexes every question the harness logged, the pilot's
    included: a generated document that repeats one is dropped like any other."""
    from service import contamination as ct
    tree = make_fixture.build(tmp_path)
    ix = ct.index(tree["out_dir"])
    q = ev.load_pilot()[0]["prompt"]
    assert ix.hits(q)
    assert not ix.hits("a sentence about harbours and tides that no question on this board holds")
