"""14.1: the rest of MobileAIBench's text sets — Dolly, CNN/DailyMail, XSum
and MT-Bench (AlpacaEval left out: CC BY-NC 4.0). Their own samples, pinned
by hash; their prompts; their metrics, ported and checked here on answers
worked out by hand; MT-Bench's second turn after the model's own first, rated
by the board's judge in a step of its own that never holds up a run. Nothing
asks a model, nothing calls OpenRouter: a fake GPU, answers written in, and
the fake judge."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

import mobileaibench as mab
from conftest import make_service
from service import config, contamination, db, llm, reported, runner, served

STAMP = "2026-10-01T10-00-00.000000"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod
    client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

def test_the_manifest_pins_every_file_and_its_rows():
    m = mab.manifest()
    assert m["sampled_by"]["revision"] == "cff7b48f3b1e0c3e2549c2ae211ce14a4ceba8d1"
    assert m["fastchat"]["revision"] == "a37078fc772a3abac082b7af474709d41b5a242b"
    for name, f in m["files"].items():
        raw = (mab.DATA_DIR / f["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == f["sha256"], name
        assert len(raw) == f["bytes"], name
        rows = (list(csv.DictReader(io.StringIO(raw.decode("utf-8")))) if f["file"].endswith(".csv")
                else [x for x in raw.decode("utf-8").splitlines() if x.strip()])
        assert len(rows) == f["n"], name
    # 1,000 each with no judge, MT-Bench's 80, and the 30 that carry GPT-4's reference
    assert [len(mab.load(t)) for t in mab.TASKS] == [1000] * 5
    assert len(mab.mt_bench()) == 80 and len(mab.mt_references()) == 30
    # every source credited, with its licence
    lic = {k: v["licence"] for k, v in m["sources"].items()}
    assert lic["databricks_dolly_15k"] == "CC BY-SA 3.0" and lic["cnn_dailymail"] == "Apache-2.0"
    assert lic["edinburghNLP_xsum"].startswith("not stated") and "Apache" in lic["mt_bench"]


def test_alpacaeval_is_not_on_the_board_and_says_why():
    m = mab.manifest()
    assert "alpaca" not in json.dumps(m["files"]).lower()
    assert "CC BY-NC 4.0" in m["note"] and "AlpacaEval" in mab.__doc__
    assert mab.part_counts("judged") == {"questions": 80, "prompts": 160, "answers": 160,
                                         "judgements": 160}
    assert mab.part_counts("")["answers"] == 5000


def test_the_prompts_are_mobileaibenchs_word_for_word():
    dolly = mab.load(mab.DOLLY)
    # their sample: 1,000 closed-QA instructions, each with its context
    assert {q["category"] for q in dolly} == {"closed_qa"}
    assert all(q["prompt"].startswith("context: ") for q in dolly)
    q = dolly[0]
    assert q["system"] == mab.SYS_QA and q["prompt"].endswith("\nanswer: ")
    assert f"\nquestion: {q['question']}\nanswer: " in q["prompt"]
    cnn = mab.load(mab.CNNDM)[0]
    assert cnn["system"] == "You're a helpful assistant who is good at summarizing articles."
    assert cnn["prompt"].startswith("Create a short summary of the following article: ")
    assert cnn["prompt"].endswith("\n")
    # their escaped quote marks, kept
    assert mab.summary_prompt('He said "no" and it\'s done') == (
        'Create a short summary of the following article: He said \\"no\\" and it\\\'s done\n')
    assert mab.SYSTEM[mab.MTB1] == "" and mab.load(mab.MTB2)[0]["turn"] == 2


# ---------------------------------------------------------------------------
# their metrics, checked by hand
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred, ref, r1, rl", [
    # no word longer than three letters: no stemming
    ("the cat lay on the mat", "the cat sat on the mat", 5 / 6, 5 / 6),
    # doctors -> doctor, running -> run, tests -> test; "ran" is too short to stem
    ("doctor ran the test", "Doctors running tests", 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3),
     2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)),
    # the same words in the reverse order: all of ROUGE-1, a quarter of ROUGE-L
    ("d c b a", "a b c d", 1.0, 0.25),
    # punctuation is a space: "U.S." is two tokens
    ("U.S. GDP grew", "the u s gdp grew 3%", 2 * 1.0 * (4 / 6) / (1.0 + 4 / 6),
     2 * 1.0 * (4 / 6) / (1.0 + 4 / 6)),
    ("", "anything at all", 0.0, 0.0),
])
def test_rouge_1_and_l_as_rouge_score_computes_them(pred, ref, r1, rl):
    assert mab.rouge1(pred, ref) == pytest.approx(r1)
    assert mab.rouge_l(pred, ref) == pytest.approx(rl)


def test_rouge_is_rouge_scores_where_it_is_installed():
    """the image has rouge_score (lm_eval needs it): there, the port is checked
    against the library MobileAIBench calls"""
    rs = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    for q in mab.load(mab.CNNDM)[:20] + mab.load(mab.XSUM)[:20]:
        pred = q["answer"][: len(q["answer"]) // 2] + " running quickly through the tests"
        got = scorer.score(q["answer"], pred)
        assert mab.rouge1(pred, q["answer"]) == pytest.approx(got["rouge1"].fmeasure)
        assert mab.rouge_l(pred, q["answer"]) == pytest.approx(got["rougeL"].fmeasure)


def test_dolly_is_scored_as_hotpotqa_is_and_a_summary_loses_its_label():
    # "Answer:" is dropped (clean_pred_QA); F1 over normalised tokens
    s = mab.score_one(mab.DOLLY, "Answer: The Atari Consumer Division", "Atari's consumer division")
    assert s["em"] == 0 and s["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    assert mab.score_one(mab.DOLLY, "Paris.", "paris")["em"] == 1
    # clean_pred_summurization: "Summary: …" scores the summary
    a = mab.score_one(mab.XSUM, "Summary: the cat sat", "the cat sat")
    assert a["rouge1"] == a["rougeL"] == pytest.approx(1.0)


@pytest.mark.parametrize("text, want", [
    ("The answer is good. Rating: [[8]]", 8.0), ("Rating: [7.5]", 7.5),
    ("I would say 9 out of 10", -1.0), ("[[11]]", 11.0),
    ("<think>it is a 3</think>\nRating: [[6]]", 6.0)])
def test_mt_bench_ratings_are_read_as_fastchat_reads_them(text, want):
    assert mab.parse_rating(text) == want


def test_the_judge_prompts_are_fastchats_with_the_reference_where_it_is_used():
    qs = {q["category"]: q for q in mab.load(mab.MTB1)}
    sys1, u1 = mab.judge_request(qs["writing"], 1, "MY ANSWER")
    assert sys1 == "You are a helpful assistant." and "[The Start of Assistant's Answer]\nMY ANSWER" in u1
    assert "Reference Answer" not in u1
    sys2, u2 = mab.judge_request(qs["math"], 2, "FIRST", "SECOND")
    assert "### Assistant A:\nFIRST" in u2 and "### Assistant A:\nSECOND" in u2
    assert "<|The Start of Reference Answer|>" in u2 and qs["math"]["reference"][1] in u2


# ---------------------------------------------------------------------------
# MT-Bench's second turn carries the first answer
# ---------------------------------------------------------------------------

def sit(model_dir: Path, task: str, answer=lambda q: "an answer of six words here", docs=None):
    """what lm_eval (or a server) logs for a run of `task`"""
    d = model_dir / f"{task}_0shot" / "pretrained__x"
    d.mkdir(parents=True, exist_ok=True)
    qs = docs if docs is not None else mab.load(task)
    with open(d / f"samples_{task}_{STAMP}.jsonl", "w", encoding="utf-8") as fh:
        for i, q in enumerate(qs):
            doc = {"id": q["id"], "prompt": q["prompt"]}
            fh.write(json.dumps({"doc_id": i, "doc": doc, "doc_hash": hashlib.sha256(
                (task + q["id"]).encode()).hexdigest(), "arguments": [[q["prompt"], {}]],
                "resps": [[answer(q)]], "filtered_resps": [answer(q)], "bypass": 999}) + "\n")
    mid = model_dir.name.replace("__", "/", 1)
    (d / f"results_{STAMP}.json").write_text(json.dumps({
        "results": {task: {"alias": task, "bypass,none": 999}}, "group_subtasks": {task: []},
        "n-shot": {task: 0}, "n-samples": {task: {"original": len(qs), "effective": len(qs)}},
        "higher_is_better": {task: {"bypass": True}},
        "config": {"model": "hf", "model_args": f"pretrained={mid},dtype=bfloat16"},
        "chat_template": True, "date": 1790200000.0}), encoding="utf-8")


def test_the_second_turn_carries_the_models_own_first_answer(tmp_path):
    d = tmp_path / "org__m"
    sit(d, mab.MTB1, answer=lambda q: f"<think>hm</think>\nFIRST for {q['id']}")
    docs = mab.turn2_docs(d)
    q = next(x for x in mab.load(mab.MTB2) if x["id"] == docs[0]["id"])
    # a server is asked the conversation so far, thinking taken out
    assert docs[0]["prompt"] == q["prompt"]
    assert docs[0]["history"] == [{"role": "user", "content": q["turns"][0]},
                                  {"role": "assistant", "content": f"FIRST for {q['id']}"}]
    # a model run here: the conversation in its own chat template, as text
    seen = []
    docs = mab.turn2_docs(d, render=lambda msgs: seen.append(msgs) or "RENDERED")
    assert docs[0]["prompt"] == "RENDERED" and docs[0]["rendered"]
    assert [m["role"] for m in seen[0]] == ["user", "assistant", "user"]
    assert seen[0][1]["content"] == f"FIRST for {docs[0]['id']}" and seen[0][2]["content"] == q["prompt"]
    assert len(docs) == 80


def test_a_server_is_asked_the_whole_conversation(monkeypatch):
    bodies = []
    monkeypatch.setattr(served, "_post", lambda url, key, body, item=True: bodies.append(body) or
                        {"choices": [{"message": {"content": "SECOND"}}]})
    rec = {"name": "phone", "base_url": "http://x/v1", "pin": {"file": "f.gguf"}}
    hist = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
    a = served.ask(rec, "Q2", {"system": "", "max_tokens": 10, "temperature": 0, "history": hist})
    assert a == "SECOND"
    assert bodies[0]["messages"] == [*hist, {"role": "user", "content": "Q2"}]
    assert "history" not in bodies[0]


# ---------------------------------------------------------------------------
# a run: answers first, judging second
# ---------------------------------------------------------------------------

def fake_gpu(monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda hf_id, k, **kw: {
        "kind": "instruct", "params": 70_000_000, "vocab": 50304, "batch": 8, "need_gb": 2.0,
        "remote_code": False, "has_template": True, "archinfo": {}})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "_chat_renderer", lambda hf_id, meta: (
        lambda msgs: "|".join(f"{m['role']}:{m['content']}" for m in msgs)))
    seen = []

    def run(sid, cmd, *a, **k):
        seen.append(cmd)
        task = cmd[cmd.index("--tasks") + 1]
        model_dir = Path(cmd[cmd.index("--output_path") + 1]).parent
        inc = Path(cmd[cmd.index("--include_path") + 1])
        docs = [json.loads(x) for x in (inc / f"{task}.jsonl").read_text().splitlines()]
        sit(model_dir, task, docs=docs, answer=lambda q: "a plain answer " * 30)
        return 0
    monkeypatch.setattr(runner, "_run_task", run)
    return seen


def test_a_run_with_the_judge_offline_finishes_awaiting_and_a_later_step_fills_it(svc,
                                                                                  monkeypatch):
    client, appmod = svc
    seen = fake_gpu(monkeypatch)
    import judge as _judge
    monkeypatch.setattr(_judge, "is_stub", lambda: False)
    monkeypatch.setattr(_judge, "blocked", lambda: "")

    def offline(role="llm"):
        raise llm.LocalUnreachable("the grading model at http://judge.test/v1 did not answer")
    monkeypatch.setattr(llm, "client", offline)
    sid = client.post("/api/submissions", json={"hf_id": "org/chat-1b", "suite": "mobile",
                                                "part": "judged", "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert [c[c.index("--tasks") + 1] for c in seen] == [mab.MTB1, mab.MTB2]
    # the first turn through the chat template; the second, the conversation as text
    assert "--apply_chat_template" in seen[0] and "--apply_chat_template" not in seen[1]
    t2 = Path(seen[1][seen[1].index("--include_path") + 1])
    assert t2.name == "org__chat-1b" and "--system_instruction" not in seen[0]
    first = json.loads((t2 / f"{mab.MTB2}.jsonl").read_text().splitlines()[0])
    assert first["prompt"].startswith("user:") and "|assistant:a plain answer" in first["prompt"]
    row = db.get(sid)
    assert row["status"] == "done", row["error"]
    assert "160 of 160 turns awaiting judge" in row["progress"]
    assert "could not be reached" in row["progress"]
    out = mab.read(config.OUT_DIR / "org__chat-1b")["tasks"][mab.MTBENCH]
    assert out["score"] is None and out["awaiting"] == 160 and out["n"] == 160
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    m = next(x for x in data["models"] if x["id"] == "org/chat-1b")
    assert "org/chat-1b" not in (data["cells"].get(mab.MTBENCH) or {})
    assert m["mab"][mab.MTBENCH]["awaiting"] == 160
    # the judge is back: the later step rates every turn, and the column fills
    monkeypatch.setattr(_judge, "is_stub", lambda: True)
    r = client.post("/api/mobileaibench/judge", json={"by": "masein"})
    assert r.status_code == 200 and [x["model"] for x in r.json()["models"]] == ["org__chat-1b"]
    out = mab.read(config.OUT_DIR / "org__chat-1b")["tasks"][mab.MTBENCH]
    assert out["awaiting"] == 0 and out["score"] == pytest.approx(mab.stub_rating(
        "a plain answer " * 30))
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    assert data["cells"][mab.MTBENCH]["org/chat-1b"]["v"] == pytest.approx(out["score"] / 10)
    assert data["tasks"][mab.MTBENCH]["metric"] == "mtbench"
    # nothing waits now: the step says so
    assert client.post("/api/mobileaibench/judge", json={"by": "masein"}).json()["models"] == []


def test_a_judge_that_isnt_set_up_leaves_the_answers_waiting_too(tmp_path, monkeypatch):
    import judge as _judge
    d = tmp_path / "org__m"
    sit(d, mab.MTB1)
    monkeypatch.setattr(_judge, "is_stub", lambda: False)
    monkeypatch.setattr(_judge, "blocked", lambda: "no judge is configured on this server")
    out = mab.start_judge(d)
    assert out["tasks"][mab.MTBENCH]["awaiting"] == 80
    assert out["note"] == "awaiting judge: no judge is configured on this server"


def test_the_poller_writes_ratings_to_the_folder_its_custom_id_names(tmp_path, monkeypatch):
    import judge as _judge
    monkeypatch.setattr(_judge, "identity", lambda: {"id": "judge-x", "provider": "fake",
                                                     "model": "x"})
    monkeypatch.setattr(_judge, "version", lambda ident=None: {"key": "v1"})
    d = tmp_path / "org__m"
    sit(d, mab.MTB1)
    qid = mab.load(mab.MTB1)[0]["id"]
    done = mab.finish(tmp_path, {f"mab:org__m:{mab.MTBENCH}:{qid}:1": llm.Result(text="[[9]]"),
                                 f"mab:org__m:{mab.MTBENCH}:{qid}:2": llm.Result(error="boom")})
    assert done == [d]
    items = {(it["id"], it["turn"]): it for it in mab.read(d)["tasks"][mab.MTBENCH]["items"]}
    assert items[(qid, 1)]["score"] == 9.0 and items[(qid, 1)]["judge"] == "judge-x"
    # another judge's rating is kept, not counted
    monkeypatch.setattr(_judge, "version", lambda ident=None: {"key": "v2"})
    assert mab.mark(d)["tasks"][mab.MTBENCH]["awaiting"] == 80


def test_the_no_judge_part_scores_all_five_and_never_waits(svc, monkeypatch):
    client, appmod = svc
    seen = fake_gpu(monkeypatch)
    sid = client.post("/api/submissions", json={"hf_id": "org/chat-2b", "suite": "mobile",
                                                "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert [c[c.index("--tasks") + 1] for c in seen] == list(mab.TASKS)
    for cmd, t in zip(seen, mab.TASKS):
        assert cmd[cmd.index("--system_instruction") + 1] == mab.SYSTEM[t]
    out = mab.read(config.OUT_DIR / "org__chat-2b")["tasks"]
    assert set(out) == set(mab.TASKS) and out[mab.CNNDM]["rougeL"] is not None
    assert db.get(sid)["status"] == "done"
    assert "CNN/DailyMail ROUGE-L" in db.get(sid)["progress"]
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    m = next(x for x in data["models"] if x["id"] == "org/chat-2b")
    for t, key in ((mab.DOLLY, "f1"), (mab.CNNDM, "rougeL"), (mab.XSUM, "rougeL")):
        assert data["cells"][t]["org/chat-2b"]["v"] == pytest.approx(out[t][key])
    # Standard benchmarks: never in the Avg, never required
    assert not set(mab.TASKS) & set(data["required"]) and m["avg"] is None


def test_the_part_is_checked(svc):
    client, _ = svc
    r = client.post("/api/submissions", json={"hf_id": "org/c", "suite": "mobile", "part": "alpaca",
                                              "kind": "instruct"})
    assert r.status_code == 422 and "judged" in r.json()["detail"]
    assert config.tasks_for_suite("mobile", part="judged") == [mab.MTB1, mab.MTB2]


# ---------------------------------------------------------------------------
# before Start: what each part takes
# ---------------------------------------------------------------------------

def test_each_part_says_its_answers_time_and_the_judges_work(svc, monkeypatch):
    client, _ = svc
    from service import ai_models
    monkeypatch.setattr(ai_models, "choice", lambda job: {
        "kind": "openrouter", "id": "deepseek/deepseek-v4.1-flash", "price_in": 0.2,
        "price_out": 0.8} if job == "judge" else None)
    monkeypatch.setattr(ai_models, "label", lambda job: "DeepSeek V4.1 Flash")
    monkeypatch.setattr(ai_models, "is_local", lambda job: False)
    j = client.get("/api/mobileaibench/estimate", params={"model": "org/new-1b"}).json()["parts"]
    assert j["none"]["answers"] == 5000 and j["none"]["line"] == \
        "5,000 answers, about 83 min, a rough guess"
    jd = j["judged"]
    assert jd["answers"] == 160 and jd["judge"]["judgements"] == 160
    tok = mab.judge_tokens()
    assert jd["judge"]["tokens_in"] == tok["tokens_in"] and jd["judge"]["usd"] == round(
        (0.2 * tok["tokens_in"] + 0.8 * tok["tokens_out"]) / 1e6, 2)
    assert jd["judge"]["line"].startswith("160 judgements · about ")
    # a served model at its own measured pace
    est = mab.estimate(True, 5.2, {"local": True})
    assert est["none"]["line"] == "5,000 answers, about 7.2 h" and \
        est["judged"]["judge"]["line"].endswith("on this server, no charge")


def test_openrouters_estimate_counts_the_judged_part(monkeypatch, tmp_path):
    rec = {"id": "served/openrouter-x", "name": "X", "via": "openrouter", "thinking": "off",
           "base_url": "https://openrouter.ai/api/v1", "how": "OpenRouter", "based_on": "",
           "pin": {"model": "x/y", "price_in": 1.0, "price_out": 2.0, "provider": "p",
                   "version": "x/y-2026"}}
    monkeypatch.setattr(served, "model_dir", lambda r: tmp_path / "none")
    est = served.estimate(rec, "mobile", part="judged")
    assert est["n"] == 160 and est["tokens_out"] == 160 * mab.MTB_MAX_GEN_TOKS


# ---------------------------------------------------------------------------
# never a training target; their paper's numbers, reported
# ---------------------------------------------------------------------------

def test_every_new_item_is_in_the_contamination_index_before_any_run(tmp_path):
    ix = contamination.BenchmarkIndex(tmp_path / "nothing-run").refresh()
    assert ix.n_pinned == 1000 * 5 + 80 + 30
    for t in (mab.DOLLY, mab.CNNDM, mab.XSUM):
        words = contamination.normalize(mab.load(t)[3]["answer"])
        if len(words) >= contamination.NGRAM:
            assert ix.hits(" ".join(words[:contamination.NGRAM])), t
    q = mab.mt_bench()[0]["turns"][0]
    assert ix.hits(q) and ix.source_of(ix.hits(q)[0]) == "benchmark"


def test_their_papers_numbers_come_in_reported_and_never_ranked(svc):
    client, _ = svc
    got = reported.import_paper()
    assert got["status"] == "imported"
    assert reported.import_paper()["status"] == "unchanged"
    v = client.get("/api/reported").json()
    rows = [s for s in v["scores"] if s["source"] == "paper"]
    mt = next(s for s in rows if s["benchmark"] == "MT-Bench (MobileAIBench)"
              and "gemma-2b-it" in s["model"])
    assert mt["value"] == pytest.approx(5.187) and mt["unit"] == "points"
    assert "judge GPT-4" in mt["setting"]
    assert v["sources"]["paper"]["credit"].startswith("as the MobileAIBench paper reports it")
    assert len(rows) == 42


def test_none_of_them_is_a_training_target(tmp_path):
    import diagnose as dx
    d = tmp_path / "org__m"
    for t in (*mab.TASKS, mab.MTB1):
        sit(d, t)
    assert not [t for t in dx.diagnose_model(d)["tasks"] if t.startswith("mab_")]
