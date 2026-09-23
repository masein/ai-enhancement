"""11l §1: thinking models were being scored on their thinking.

Qwen3-1.7B (run #60) opened <think> on every one of 3,730 answers and closed
it on none: the 256-token budget ran out inside the reasoning, and the judge
graded 3,730 cut-off monologues as answers. Now the reasoning is taken out
before anything is graded or shown; an answer that never left it is no
answer — counted, never scored, in no mean; a topic that is mostly no answer
has no score at all; a reasoning model's judged answers get room; and a
check names any run with unfinished answers. A model that does not reason
scores exactly as it did.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

import judge as jd
from conftest import make_service

MODEL = "fx/good-750m"
TASK = "exam_economics"
TOPIC = "Economics"
UNFINISHED = "<think>\nLet me work out what the question is really asking. First, the"
FINISHED = "<think>\nThe question is about supply. Six units.\n</think>\n\nThe answer is 6."


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def model_dir(tree, mid=MODEL) -> Path:
    return tree["models"][mid]["dir"]


def rewrite_answers(mdir: Path, task: str, make) -> int:
    """Put these answers in the harness's own log, as a reasoning model
    writes them. `make(i)` gives the i-th raw generation."""
    n = 0
    for f in sorted(mdir.glob(f"{task}_*shot/**/samples_*.jsonl")):
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        for i, r in enumerate(recs):
            t = make(n + i)
            r["resps"], r["filtered_resps"] = [[t]], [t]
        n += len(recs)
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                     encoding="utf-8")
    return n


def grade(mdir: Path, task: str = TASK):
    reqs, plan = jd.plan_requests(mdir, "stub", only=[task])
    ident = {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1",
             "family": "stub"}
    out = jd.assemble(plan, jd.stub_results(reqs), ident, "stub", mdir.parent, 0.5, False,
                      record=False)
    return reqs, out["tasks"][task]


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

def test_the_reasoning_comes_out_and_the_answer_stays():
    p = jd.split_reasoning(FINISHED)
    assert p == {"answer_text": "The answer is 6.",
                 "reasoning_text": "The question is about supply. Six units.",
                 "had_reasoning": True, "reasoning_unterminated": False}
    p = jd.split_reasoning(UNFINISHED)
    assert p["had_reasoning"] and p["reasoning_unterminated"] and p["answer_text"] == ""
    # a template that opened the block in the prompt: only the close is written
    p = jd.split_reasoning("Supply meets demand at six.</think>The answer is 6.")
    assert p["answer_text"] == "The answer is 6." and not p["reasoning_unterminated"]
    # no block at all: exactly the text written, whitespace and all
    raw = "  The answer is six.\n"
    assert jd.split_reasoning(raw) == {"answer_text": raw, "reasoning_text": "",
                                       "had_reasoning": False, "reasoning_unterminated": False}


def test_the_wrappers_are_a_setting(monkeypatch):
    monkeypatch.setenv("REASONING_WRAPPERS", "<reasoning>,</reasoning>")
    assert ("<reasoning>", "</reasoning>") in jd.reasoning_wrappers()
    assert ("<think>", "</think>") in jd.reasoning_wrappers()          # the default stays
    p = jd.split_reasoning("<reasoning>hmm</reasoning>Six.")
    assert p["answer_text"] == "Six." and p["had_reasoning"]


def test_no_answer_is_only_ever_a_reasoning_models():
    """An empty generation from a model that does not reason is graded as it
    always was; only one that never left its reasoning block is no answer."""
    rec = lambda t: {"resps": [[t]], "filtered_resps": [t]}           # noqa: E731
    assert jd.answer_parts(rec(""))["no_answer"] is False
    assert jd.answer_parts(rec(UNFINISHED))["no_answer"] is True
    assert jd.answer_parts(rec("<think>done</think>   "))["no_answer"] is True
    assert jd.answer_parts(rec(FINISHED))["no_answer"] is False


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------

def test_an_unterminated_answer_is_not_graded_counted_and_in_no_mean(svc):
    _, _, tree = svc
    mdir = model_dir(tree)
    # every third answer never leaves its reasoning; the rest finish
    n = rewrite_answers(mdir, TASK, lambda i: UNFINISHED if i % 3 == 0 else FINISHED)
    gone = len(range(0, n, 3))
    reqs, t = grade(mdir)
    judged = [r for r in reqs if r.custom_id.startswith("judge:")]
    assert len(judged) == n - gone                         # nothing sent for no answer
    # the judge sees the answer and never the reasoning
    assert all("The answer is 6." in r.user for r in judged)
    assert not any("Six units" in r.user or "<think>" in r.user for r in judged)
    items = t["items"]
    none = [it for it in items if it.get("no_answer")]
    assert len(none) == gone == t["no_answer"]
    assert all(it["score"] is None and not it["graded"] for it in none)
    assert all(it.get("reasoning_unterminated") for it in none)
    scored = [it for it in items if not it.get("no_answer")]
    assert t["mean"] == round(sum(it["score"] for it in scored) / len(scored), 4)
    rep = [it["score"] for it in scored if it["half"] == "report"]
    assert t["score_report"] == round(sum(rep) / len(rep), 4)
    assert sum(t["dist"].values()) == len(scored)             # the distribution too
    # and every criterion mean is over the answered items only
    assert t.get("criteria_n"), "Economics is graded criterion by criterion"
    assert max(t["criteria_n"].values()) <= len(scored)
    assert t["no_score"] if gone * 2 > n else "no_score" not in t
    assert t["answers"]["no_answer"] == gone


def test_a_topic_that_never_answered_reports_a_dash_not_a_number(svc):
    import report_lm_eval as report
    _, _, tree = svc
    mdir = model_dir(tree)
    rewrite_answers(mdir, TASK, lambda i: UNFINISHED)
    reqs, t = grade(mdir)
    assert not [r for r in reqs if r.custom_id.startswith("judge:")]
    assert t["mean"] is None and t["score_report"] is None and t["score_diagnose"] is None
    assert t["no_score"] == "the model never finished answering: these questions were not scored"
    assert t["no_answer"] == t["n"]
    assert report.published_score(t) is None
    gate = report.topic_gate(TASK, t, {"ok": True, "reasons": []}, None)
    assert gate["hard"] and gate["hard"][0]["short"] == "not scored — the model never finished answering"


def test_a_model_that_does_not_reason_scores_exactly_as_before(svc):
    """The refactored aggregates against the formula they replaced, on every
    judged fixture model and topic; and re-checking their files changes
    nothing, byte for byte."""
    _, _, tree = svc
    for mid in ("fx/good-750m", "fx/skewed-360m", "fx/chance-160m"):
        mdir = model_dir(tree, mid)
        before = (mdir / "judge.json").read_bytes()
        j = json.loads(before)
        for task, t in j["tasks"].items():
            items = t["items"]
            assert not any(it.get("no_answer") or it.get("had_reasoning") for it in items)
            rep = [it["score"] for it in items if it["half"] == "report"]
            dia = [it["score"] for it in items if it["half"] == "diagnose"]
            # the formula before 11l, verbatim
            assert t["mean"] == round(sum(it["score"] for it in items) / len(items), 4), task
            assert t["score_report"] == (round(sum(rep) / len(rep), 4) if rep else None)
            assert t["score_diagnose"] == (round(sum(dia) / len(dia), 4) if dia else None)
            c = collections.Counter(str(it["score"]) for it in items)
            assert t["dist"] == {str(k): c.get(str(k), 0) for k in range(5)}
            assert "no_answer" not in t and "no_score" not in t
            # summarise_task on the stored items gives the stored numbers back
            again = jd.summarise_task(task, items, jd.rubric_for(task).criteria)
            for k, v in again.items():
                assert t.get(k) == v, (mid, task, k)
        assert jd.revalidate(mdir) is None
        assert (mdir / "judge.json").read_bytes() == before


# ---------------------------------------------------------------------------
# voiding a run like #60, and the check
# ---------------------------------------------------------------------------

def test_revalidate_voids_a_run_graded_on_its_reasoning_and_says_why(svc):
    client, appmod, tree = svc
    from conftest import fresh
    mdir = model_dir(tree)
    before = json.loads((mdir / "judge.json").read_text(encoding="utf-8"))
    old = {it["doc_hash"]: it["score"] for it in before["tasks"][TASK]["items"]}
    rewrite_answers(mdir, TASK, lambda i: UNFINISHED)
    assert jd.revalidate(mdir) == {TASK: len(old)}
    t = json.loads((mdir / "judge.json").read_text(encoding="utf-8"))["tasks"][TASK]
    assert t["no_score"] and t["no_answer"] == len(old)
    # nothing deleted silently: the judge's grade is kept beside the item
    assert all(it["voided"]["score"] == old[it["doc_hash"]] for it in t["items"])
    assert all(it["score"] is None for it in t["items"])
    # the other topics are untouched
    for task, t0 in before["tasks"].items():
        if task != TASK:
            now = json.loads((mdir / "judge.json").read_text(encoding="utf-8"))["tasks"][task]
            assert now == t0
    # a second pass has nothing to do
    assert jd.revalidate(mdir) is None
    # the board: no score for the topic, and a check that names the model
    fresh(appmod)
    body = client.get("/api/results").json()
    m = next(x for x in body["models"] if x["id"] == MODEL)
    jt = m["judge"]["tasks"][TASK]
    assert jt.get("score_report") is None and jt["no_score"] and jt["no_answer"] == len(old)
    check = next(c for c in body["checks"] if c["key"] == "judge_unfinished")
    assert check["severity"] == "warning"
    assert "good-750m" in check["short"] and "good-750m" in check["text"]
    assert check["show"] == {"model": MODEL}


def test_the_answers_endpoint_shows_the_answer_and_marks_no_answer(svc):
    client, appmod, tree = svc
    from conftest import fresh
    mdir = model_dir(tree)
    rewrite_answers(mdir, TASK, lambda i: UNFINISHED if i % 2 else FINISHED)
    jd.write_judge(mdir, jd.merge_judged(mdir, jd.run_stub(mdir, only=[TASK])))
    fresh(appmod)
    body = client.get("/api/answers", params={"model": MODEL, "topic": TOPIC}).json()
    done = [r for r in body["items"] if not r["no_answer"]]
    none = [r for r in body["items"] if r["no_answer"]]
    assert done and none
    assert all(r["answer"] == "The answer is 6." and r["had_reasoning"] for r in done)
    assert all(r["reasoning"] == "The question is about supply. Six units." for r in done)
    assert all(r["answer"] == "" and r["reasoning_unterminated"] and r["score"] is None
               for r in none)
    assert body["no_answer"] == sum(1 for _ in jd._records(mdir, TASK)) // 2


# ---------------------------------------------------------------------------
# room to answer
# ---------------------------------------------------------------------------

def test_a_reasoning_template_is_recognised(monkeypatch):
    from service import hfmeta
    assert hfmeta.reasoning_template("{% if enable_thinking %}<think>{% endif %}")
    assert hfmeta.reasoning_template("<|assistant|><think>\n")
    assert not hfmeta.reasoning_template("<|im_start|>assistant\n")
    assert not hfmeta.reasoning_template(None)
    monkeypatch.setenv("REASONING_WRAPPERS", "<reasoning>,</reasoning>")
    assert hfmeta.reasoning_template("<reasoning>")
    assert hfmeta._template_id("<think>", "tokenizer_config.json")["reasoning_template"] is True
    assert "reasoning_template" not in hfmeta._template_id("plain", "tokenizer_config.json")


@pytest.mark.parametrize("reasoning", [True, False])
def test_a_reasoning_models_judged_answers_get_room(svc, monkeypatch, reasoning):
    """Only its judged answers, and only when its template thinks: every
    other model keeps the 256 tokens it was always given."""
    import shutil

    from service import config, db, runner
    client, _, tree = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    shutil.rmtree(model_dir(tree) / f"{TASK}_0shot")          # nothing on disk: it must run
    sid = client.post("/api/submissions", json={"hf_id": MODEL, "suite": "judged",
                                                "tasks": [TASK]}).json()["id"]
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {
        "kind": "instruct", "params": 1_700_000_000, "vocab": 151936, "batch": 8,
        "need_gb": 6.0, "remote_code": False,
        "archinfo": {"reasoning_template": True} if reasoning else {}})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    seen = []
    monkeypatch.setattr(runner, "_run_task", lambda sid, cmd, *a, **k: seen.append(cmd) or 1)
    runner.run_submission(db.get(sid))
    cmd = next(c for c in seen if TASK in c)
    if reasoning:
        assert cmd[cmd.index("--gen_kwargs") + 1] == f"max_gen_toks={config.REASONING_MAX_GEN_TOKS}"
    else:
        assert "--gen_kwargs" not in cmd
    assert "--apply_chat_template" in cmd


def test_the_budget_travels_with_the_grades(svc):
    """What the answers were generated with is read from the harness's own
    results file and recorded per topic; How this was graded reads it."""
    client, appmod, tree = svc
    from conftest import fresh
    mdir = model_dir(tree)
    for f in mdir.glob(f"{TASK}_*shot/**/results_*.json"):
        blob = json.loads(f.read_text(encoding="utf-8"))
        blob["configs"][TASK]["generation_kwargs"] = {"until": ["\n\n\n"], "max_gen_toks": 2048}
        blob["config"]["gen_kwargs"] = "max_gen_toks=2048"
        f.write_text(json.dumps(blob), encoding="utf-8")
    jd.write_judge(mdir, jd.merge_judged(mdir, jd.run_stub(mdir, only=[TASK])))
    t = json.loads((mdir / "judge.json").read_text(encoding="utf-8"))["tasks"][TASK]
    assert t["generation"] == {"until": ["\n\n\n"], "max_gen_toks": 2048,
                               "override": "max_gen_toks=2048"}
    fresh(appmod)
    prov = client.get("/api/judge/provenance", params={"model": MODEL}).json()
    assert prov["tasks"][TASK]["generation"]["max_gen_toks"] == 2048


def test_the_deploy_step_says_what_it_voided(svc, monkeypatch, capsys):
    """judge.py --revalidate: the one data step after this merges, and the
    lines masein will read."""
    import sys
    _, _, tree = svc
    mdir = model_dir(tree)
    n = rewrite_answers(mdir, TASK, lambda i: UNFINISHED)
    monkeypatch.setattr(sys, "argv", ["judge.py", str(mdir.parent), "--revalidate"])
    assert jd.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"fx__good-750m: {n} answers never finished — 1 of 1 topics now have no score"
    assert out[-1] == "revalidated: 1 model(s) changed, the rest untouched"
    assert jd.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "revalidated: 0 model(s) changed, the rest untouched"]
