"""12n.2 in the service: GPQA Diamond and SimpleQA Verified measured here.

GPQA is gated and its questions are never shown: lm_eval loads it at run time,
its gated failure reads one line, and no endpoint, log or diagnosis carries a
question of it — checked on invented questions in its shape
(tests/fixtures/gpqa). SimpleQA Verified is pinned and committed (MIT), asked
as Trust & safety is, and graded by the judge with the dataset's grader:
correct, incorrect or not attempted; its cell is Epoch AI's number, the share
correct, with not attempted apart. A new suite, "shared"; GPQA's four options
join the full suite; the GGUF table gains GPQA. Neither is in the Avg or
Improve. Nothing asks a model, Hugging Face or OpenRouter: fixtures, the fake
judge and 12m.3's fake OpenRouter only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import gguf_bench as gb
import gguf_data as gd
import report_lm_eval as report
import simpleqa as sq
from conftest import make_service
from service import config, db, llm, llm_poller, runner, served

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gpqa"
INVENTED = [json.loads(x) for x in (FIXTURES / "gpqa_diamond_invented.jsonl").read_text(
    encoding="utf-8").splitlines() if x.strip()]
STAMP = "2026-09-28T10-00-00.000000"
GOOD = "fx/good-750m"


def processed(row: dict, correct_at: int = 2) -> dict:
    """an invented row as lm_eval's process_docs leaves it: four choices, the
    right one at `correct_at`, and its letter"""
    wrong = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    choices = wrong[:correct_at] + [row["Correct Answer"]] + wrong[correct_at:]
    return {**row, **{f"choice{i + 1}": c for i, c in enumerate(choices)},
            "answer": f"({'ABCD'[correct_at]})"}


def prompt_of(doc: dict) -> str:
    return ("What is the correct answer to this question:" + doc["Question"] + "\nChoices:\n"
            + "".join(f"({k}) {doc[f'choice{i + 1}']}\n" for i, k in enumerate("ABCD"))
            + "Let's think step by step: ")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod, tree
    client.__exit__(None, None, None)


def payload(client, appmod) -> dict:
    appmod._cache.update(key=None, payload=None, at=0.0)
    return client.get("/api/results").json()


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

def test_simpleqa_verified_is_pinned_and_credited():
    m = sq.manifest()["sources"]
    s = m["simpleqa_verified"]
    assert s["licence"] == "MIT" and s["revision"] == "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc"
    assert hashlib.sha256((sq.DATA_DIR / sq.CSV_NAME).read_bytes()).hexdigest() == \
        s["files"][sq.CSV_NAME]
    assert hashlib.sha256(sq.GRADER.read_bytes()).hexdigest() == \
        m["grader"]["files"]["grader_template.txt"]
    qs = sq.load()
    assert len(qs) == 1000 == len({q["id"] for q in qs}) and all(q["prompt"] and q["answer"]
                                                               for q in qs)
    # the task holds the questions, never their answers
    d = sq.build_tasks(Path(__import__("tempfile").mkdtemp()))
    rows = [json.loads(x) for x in (d / f"{sq.TASK}.jsonl").read_text().splitlines()]
    assert len(rows) == 1000 and set(rows[0]) == {"id", "prompt"}
    yaml = (d / f"{sq.TASK}.yaml").read_text()
    assert yaml.startswith(f"task: {sq.TASK}\n")
    assert not [k for k in ("__ITEMS_PATH__", "__UNTIL__", "__MAX_GEN_TOKS__", "__DO_SAMPLE__",
                            "__TEMPERATURE__") if k in yaml]


def test_the_suites():
    assert config.tasks_for_suite("shared") == ["gpqa_diamond_cot_zeroshot", "simpleqa_verified"]
    assert "gpqa_diamond_zeroshot" in config.tasks_for_suite("full")
    assert "shared" in config.SUITES and "shared" in served.SUITES
    assert runner.include_args_for("simpleqa_verified") == ["--include_path",
                                                            str(config.SIMPLEQA_TASKS_DIR)]
    # GPQA's forms are the harness's own: no include path of ours names them
    assert "gpqa" in gb.ORDER and gb.BENCHMARKS["gpqa"]["lm_eval"] == "gpqa_diamond_zeroshot"


def test_a_base_model_is_refused_the_shared_suite(svc):
    client, _, _ = svc
    r = client.post("/api/submissions", json={"hf_id": "org/base-1b", "suite": "shared",
                                              "kind": "base"})
    assert r.status_code == 422 and r.json()["detail"] == \
        config.SHARED_INSTRUCT_ONLY + ". Nothing was queued."


# ---------------------------------------------------------------------------
# GPQA is gated: one plain line
# ---------------------------------------------------------------------------

GATED_TAILS = [
    "datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset "
    "on the Hub. You must be authenticated to access it.",
    "huggingface_hub.errors.GatedRepoError: 403 Client Error.\nCannot access gated repo for url "
    "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv.",
]


@pytest.mark.parametrize("tail", GATED_TAILS)
def test_gpqas_gate_reads_one_line(tail):
    assert config.GPQA_GATED == ("GPQA is gated: accept its terms at "
                                 "https://huggingface.co/datasets/Idavidrein/gpqa with this "
                                 "server's HF account")
    assert runner.classify(tail) == config.GPQA_GATED
    # a gated model is still the model's line
    assert runner.classify("GatedRepoError: 401 Client Error for org/private-1b").startswith(
        "the model is gated")


def test_gguf_datas_gate_reads_the_same_line(tmp_path):
    assert gd.GPQA_GATED == config.GPQA_GATED

    def gated(name):
        raise RuntimeError("Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub")
    with pytest.raises(SystemExit) as e:
        gd.build(tmp_path, only=["gpqa"], docs_of=gated)
    assert str(e.value) == config.GPQA_GATED


def test_the_check_says_whether_this_server_may_read_gpqa():
    import check_tasks as ct

    class GatedRepoError(Exception):
        pass

    def no(repo, repo_type):
        raise GatedRepoError(f"Access to dataset {repo} is restricted and you are not in the "
                             "authorized list.")
    assert ct.gpqa_access(no) == "gpqa      " + config.GPQA_GATED
    assert ct.gpqa_access(lambda repo, repo_type: None) == \
        "gpqa      this server's HF account can read GPQA Diamond"


# ---------------------------------------------------------------------------
# GPQA on the GGUF: the binary round-trips
# ---------------------------------------------------------------------------

def test_gpqa_in_the_gguf_table_round_trips_on_invented_questions(tmp_path):
    docs = [processed(r, i % 4) for i, r in enumerate(INVENTED)]
    man = gd.build(tmp_path, only=["gpqa"], docs_of=lambda name: docs)
    e = man["benchmarks"]["gpqa"]
    assert (e["n"], e["of"], e["skipped"]) == (len(docs), len(docs), 0)
    back = gd.read_mc_binary((tmp_path / gb.BENCHMARKS["gpqa"]["data"]).read_bytes())
    assert len(back) == len(docs)
    for i, (t, d) in enumerate(zip(back, docs)):
        assert t["question"] == prompt_of(d).replace("Let's think step by step: ", "Answer:")
        assert t["answers"] == ["(A)", "(B)", "(C)", "(D)"]
        assert t["labels"] == [1 if k == i % 4 else 0 for k in range(4)]
    # four options, 0-shot: as lm_eval's four-options form scores them
    assert e["max_answers"] == 4


# ---------------------------------------------------------------------------
# SimpleQA Verified: three outcomes, and Epoch's number
# ---------------------------------------------------------------------------

def sit_simpleqa(model_dir: Path, replies: dict[str, str]) -> None:
    qs = {q["id"]: q for q in sq.load()}
    d = model_dir / f"{sq.TASK}_0shot" / "pretrained__x"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"samples_{sq.TASK}_{STAMP}.jsonl", "w", encoding="utf-8") as fh:
        for i, (qid, text) in enumerate(replies.items()):
            fh.write(json.dumps({"doc_id": i, "doc": {"id": qid, "prompt": qs[qid]["prompt"]},
                                 "resps": [[text]], "filtered_resps": [text]}) + "\n")
    mid = model_dir.name.replace("__", "/", 1)
    (d / f"results_{STAMP}.json").write_text(json.dumps({
        "results": {sq.TASK: {"alias": sq.TASK, "bypass,none": 999}},
        "group_subtasks": {sq.TASK: []}, "n-shot": {sq.TASK: 0},
        "n-samples": {sq.TASK: {"original": 1000, "effective": len(replies)}},
        "higher_is_better": {sq.TASK: {"bypass": True}},
        "config": {"model": "hf", "model_args": f"pretrained={mid},dtype=bfloat16"},
        "chat_template": True, "date": 1790100000.0}), encoding="utf-8")


def twenty() -> dict[str, str]:
    """6 right, 4 that say they don't know, 2 left empty, 8 wrong"""
    qs = sq.load()[:20]
    out = {}
    for i, q in enumerate(qs):
        out[q["id"]] = (f"It is {q['answer']}." if i < 6 else "I'm not sure." if i < 10
                        else "" if i < 12 else "Tuesday, in Paris, 1850.")
    return out


def test_the_judges_three_outcomes_and_epochs_number(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    d = tmp_path / "org__m"
    sit_simpleqa(d, twenty())
    out = sq.start(d)
    assert out["waiting"] == 0 and len(out["items"]) == 20
    r = out["rates"]
    # Epoch AI's: the share of all the questions answered correctly
    assert (r["correct"]["n"], r["correct"]["of"], r["correct"]["rate"]) == (6, 20, 0.3)
    # not attempted, apart: "not sure" and the two left empty
    assert (r["not_attempted"]["n"], r["incorrect"]["n"]) == (6, 8)
    assert sq.summary(out) == "SimpleQA Verified 30.0% correct · 30% not attempted"
    # the grader is the dataset's: the question, the gold target and the answer, filled in
    q = sq.load()[0]
    p = sq.judge_prompt(q, "It is 7.")
    assert f"Question: {q['prompt']}\nGold target: {q['answer']}\nPredicted answer: It is 7." in p
    assert p.rstrip().endswith('Just return the letters "A", "B", or "C", with no text around it.')
    assert (sq.parse_grade("A"), sq.parse_grade("B"), sq.parse_grade("C")) == (
        "correct", "incorrect", "not_attempted")
    assert sq.parse_grade("no idea") is None
    # graded again with nothing new: the judge is not asked twice
    monkeypatch.setattr(sq, "stub_grade", lambda *a: pytest.fail("asked again"))
    assert sq.start(d)["rates"] == r


def test_one_batch_to_the_judge_and_a_reply_it_cant_read_counts_against_no_one(svc, monkeypatch):
    client, _, _ = svc
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_MODEL", "fake-judge")
    llm.reset()
    d = config.OUT_DIR / "org__m"
    sit_simpleqa(d, twenty())
    sid = client.post("/api/submissions", json={"hf_id": "org/m", "suite": "shared",
                                                "kind": "instruct"}).json()["id"]
    out = sq.start(d, submission=sid)
    assert out["batch_id"] and out["waiting"] == 18 and out["rates"] is None
    row = next(b for b in db.batches_list(50) if b["batch_id"] == out["batch_id"])
    assert (row["kind"], row["n_items"], row["ref_id"]) == ("simpleqa", 18, sid)
    llm_poller.tick()
    got = sq.read(d)
    assert got["waiting"] == 0 and got["rates"]["correct"]["rate"] == 0.3
    assert db.get(sid)["progress"] == "SimpleQA Verified 30.0% correct · 30% not attempted"
    # a reply the judge's reader can't read is its failure: left out, not wrong
    res = {f"simpleqa:{sid}:{sq.load()[i]['id']}": llm.Result(
        text="maybe" if i == 0 else "A") for i in range(6)}
    sq.write(d, {**got, "items": [{**it, "grade": None, "grader": None} if
                                  it["id"] in {sq.load()[i]["id"] for i in range(6)} else it
                                  for it in got["items"]]})
    out = sq.finish(d, res)
    assert out["rates"]["correct"]["of"] == 19 and out["rates"]["correct"]["n"] == 5


def test_the_cell_is_the_share_correct_and_the_row_carries_not_attempted(svc, monkeypatch):
    client, appmod, tree = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    d = tree["models"][GOOD]["dir"]
    before = payload(client, appmod)
    sit_simpleqa(d, twenty())
    sq.start(d)
    after = payload(client, appmod)
    c = after["cells"]["simpleqa_verified"][GOOD]
    assert c["v"] == pytest.approx(0.3)
    m = next(x for x in after["models"] if x["id"] == GOOD)
    assert m["simpleqa"]["not_attempted"] == {"v": 0.3, "n": 6, "of": 20, "se": pytest.approx(
        0.102469, abs=1e-5)}
    # never in the Avg, nor the partial one
    b = next(x for x in before["models"] if x["id"] == GOOD)
    assert (m["avg"], m["avgRaw"], m["partialAvg"]) == (b["avg"], b["avgRaw"], b["partialAvg"])
    assert "simpleqa_verified" not in after["required"]
    # 17: and the Frontier benchmarks as Epoch AI runs them
    assert after["frontierTasks"] == ["gpqa_diamond_zeroshot", "gpqa_diamond_cot_zeroshot",
                                      "simpleqa_verified", "gpqa_diamond_epoch"]
    assert after["shared"]["simpleqa"]["licence"] == "MIT"
    assert "honest, not wrong" in after["shared"]["simpleqa"]["honest"]


# ---------------------------------------------------------------------------
# GPQA never leaves the server — on invented questions in its shape
# ---------------------------------------------------------------------------

def sit_gpqa(model_dir: Path, task: str = "gpqa_diamond_cot_zeroshot", acc: float = 0.5) -> None:
    """what lm_eval logs for a GPQA run: its samples (the question in each
    doc) and its results"""
    d = model_dir / f"{task}_0shot" / "pretrained__x"
    d.mkdir(parents=True, exist_ok=True)
    docs = [processed(r, i % 4) for i, r in enumerate(INVENTED)]
    with open(d / f"samples_{task}_{STAMP}.jsonl", "w", encoding="utf-8") as fh:
        for i, doc in enumerate(docs):
            reply = f"Thinking it through. The answer is {doc['answer']}."
            fh.write(json.dumps({"doc_id": i, "doc": doc, "target": doc["answer"],
                                 "arguments": [[prompt_of(doc), {"until": ["</s>"]}]],
                                 "resps": [[reply]], "filtered_resps": [doc["answer"]],
                                 "exact_match": 1.0}) + "\n")
    mid = model_dir.name.replace("__", "/", 1)
    metric = ("exact_match,flexible-extract" if "cot" in task else "acc,none")
    (d / f"results_{STAMP}.json").write_text(json.dumps({
        "results": {task: {"alias": task, metric: acc,
                           metric.replace(",", "_stderr,"): 0.035}},
        "group_subtasks": {task: []}, "n-shot": {task: 0},
        "n-samples": {task: {"original": 198, "effective": 198}},
        "higher_is_better": {task: {metric.split(",")[0]: True}},
        "config": {"model": "hf", "model_args": f"pretrained={mid},dtype=bfloat16"},
        "chat_template": "cot" in task, "date": 1790100000.0}), encoding="utf-8")


def leaks(text: str) -> list[str]:
    """every invented GPQA question, or option, whose words are in `text`"""
    out = []
    for r in INVENTED:
        for k in ("Question", "Correct Answer", "Incorrect Answer 1"):
            if len(r[k]) > 8 and r[k] in text:
                out.append(f"{r['Record ID']}:{k}")
    return out + (["canary"] if "invented-gpqa-fixture" in text else [])


def test_no_gpqa_question_reaches_an_endpoint_a_log_or_a_diagnosis(svc, monkeypatch):
    client, appmod, tree = svc
    d = tree["models"][GOOD]["dir"]
    sit_gpqa(d)
    sit_gpqa(d, "gpqa_diamond_zeroshot", 0.3)
    # a run's log in which lm_eval printed a prompt of it
    sid = client.post("/api/submissions", json={"hf_id": GOOD, "suite": "shared",
                                                "kind": "instruct"}).json()["id"]
    log = config.LOGS_DIR / f"service_{sid}_{GOOD.replace('/', '__')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    doc = processed(INVENTED[0])
    log.write_text("===== gpqa_diamond_cot_zeroshot =====\n" + prompt_of(doc)
                   + "\nRequesting API: 100%|██| 198/198\n", encoding="utf-8")
    # the diagnosis a person runs writes none of it
    import diagnose as dx
    got = dx.diagnose_model(d)
    assert not [t for t in got["tasks"] if t.startswith("gpqa")]
    (d / "diagnose.json").write_text(json.dumps({**got, "tasks": {**got["tasks"], "gpqa_diamond_zeroshot": {
        "metric": "acc", "n": 6, "examples": {"wrong": [{"q": INVENTED[0]["Question"]}]}}}}))
    p = payload(client, appmod)
    assert p["cells"]["gpqa_diamond_cot_zeroshot"][GOOD]["v"] == 0.5
    assert p["cells"]["gpqa_diamond_zeroshot"][GOOD]["v"] == 0.3
    texts = {"/api/results": json.dumps(p),
             "log": client.get(f"/api/runs/{sid}/log").text,
             "lines": client.get(f"/api/runs/{sid}/lines").text,
             "page": client.get("/").text,
             "submissions": client.get("/api/submissions").text,
             "playground": client.get("/api/playground/practice").text}
    for where, text in texts.items():
        assert leaks(text) == [], where
    lines = client.get(f"/api/runs/{sid}/lines").json()["lines"]
    assert "[line withheld — it quotes a GPQA question, never shown]" in lines
    assert "Requesting API: 100%|██| 198/198" in lines          # the rest of the log is shown


# ---------------------------------------------------------------------------
# kept out of Improve and the Avg
# ---------------------------------------------------------------------------

def test_neither_is_in_the_avg_or_improve(svc, monkeypatch):
    client, appmod, tree = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    before = next(m for m in payload(client, appmod)["models"] if m["id"] == GOOD)
    d = tree["models"][GOOD]["dir"]
    sit_gpqa(d, acc=0.02)
    sit_gpqa(d, "gpqa_diamond_zeroshot", 0.01)
    sit_simpleqa(d, twenty())
    sq.start(d)
    p = payload(client, appmod)
    after = next(m for m in p["models"] if m["id"] == GOOD)
    assert (after["avg"], after["partialAvg"], after["nhave"]) == (
        before["avg"], before["partialAvg"], before["nhave"])
    assert not set(report.FRONTIER_TASKS) & set(p["required"])
    from service import proposals
    weak = proposals.weak_topics(d)
    assert not any(t in json.dumps(weak) for t in ("gpqa", "simpleqa"))
    for t in report.FRONTIER_TASKS:
        r = client.post("/api/proposals", json={"model": GOOD, "topic": t})
        assert r.status_code in (400, 404, 422), (t, r.status_code)


# ---------------------------------------------------------------------------
# the runs: HF, served, and a model from OpenRouter
# ---------------------------------------------------------------------------

def fake_gpu(monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda hf_id, k, **kw: {
        "kind": "instruct", "params": 70_000_000, "vocab": 50304, "batch": 8, "need_gb": 2.0,
        "remote_code": False, "has_template": True, "archinfo": {}})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    monkeypatch.setattr(runner, "gen_backend", lambda: ("hf", "no vLLM here"))
    seen = []

    def run(sid, cmd, *a, **k):
        seen.append(cmd)
        task = cmd[cmd.index("--tasks") + 1]
        model_dir = Path(cmd[cmd.index("--output_path") + 1]).parent
        if task == sq.TASK:
            sit_simpleqa(model_dir, twenty())
        else:
            sit_gpqa(model_dir, task)
        return 0
    monkeypatch.setattr(runner, "_run_task", run)
    return seen


def test_the_shared_suite_asks_gpqa_as_the_generative_three_and_simpleqa_as_trust(
        svc, monkeypatch):
    client, _, _ = svc
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    seen = fake_gpu(monkeypatch)
    sid = client.post("/api/submissions", json={"hf_id": "org/chat-1b", "suite": "shared",
                                                "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert [c[c.index("--tasks") + 1] for c in seen] == ["gpqa_diamond_cot_zeroshot",
                                                         "simpleqa_verified"]
    cot, sqa = seen
    # the chain of thought: through the chat template, with the generative budget
    assert "--apply_chat_template" in cot
    assert cot[cot.index("--gen_kwargs") + 1] == f"max_gen_toks={config.GEN_MAX_GEN_TOKS}"
    assert "--include_path" not in cot or config.SIMPLEQA_TASKS_DIR.name not in " ".join(cot)
    # SimpleQA: its own task, from the pinned file, through the chat template
    assert sqa[sqa.index("--include_path") + 1] == str(config.SIMPLEQA_TASKS_DIR)
    assert "--apply_chat_template" in sqa
    row = db.get(sid)
    assert row["status"] == "done", row
    assert row["progress"] == "SimpleQA Verified 30.0% correct · 30% not attempted"


def test_a_served_model_is_asked_simpleqa_as_typed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SIMPLEQA_TASKS_DIR", sq.build_tasks(tmp_path / "tasks"))
    got = {}

    def answer_task(rec, task, docs, task_out, s, everyday, **k):
        got.update(task=task, n=len(docs), everyday=everyday, s=s,
                   prompt=served.prompt_of(docs[0], everyday))
        return 0
    monkeypatch.setattr(served, "answer_task", answer_task)
    monkeypatch.setattr(runner, "_everyday_settings", lambda meta: {
        "max_gen_toks": 512, "until": ["\n\n\n\n"], "temperature": 0.0})
    rec = {"thinking": "off", "name": "phone", "base_url": "http://x", "pin": {"file": "f.gguf"}}
    status, stopped = runner._ask_served(1, rec, {}, sq.TASK, tmp_path / "out", "2/2 · simpleqa",
                                         tmp_path / "log", False, None)
    assert (status, stopped) == (0, None)
    assert got["task"] == sq.TASK and got["n"] == 1000 and got["everyday"] is True
    assert got["prompt"] == sq.load()[0]["prompt"]


def test_an_openrouter_models_estimate_lists_both_benchmarks(monkeypatch, tmp_path):
    rec = {"id": "served/openrouter-openai-gpt-6-luna", "name": "GPT-6 Luna", "via": "openrouter",
           "thinking": "off", "base_url": "https://openrouter.ai/api/v1", "how": "OpenRouter",
           "based_on": "",
           "pin": {"model": "openai/gpt-6-luna", "price_in": 2.0, "price_out": 8.0,
                   "provider": "openai", "version": "openai/gpt-6-luna-2026-09-01"}}
    monkeypatch.setattr(served, "model_dir", lambda r: tmp_path / "none")
    monkeypatch.setattr(config, "SIMPLEQA_TASKS_DIR", tmp_path / "sq")
    est = served.estimate(rec, "shared")
    # 198 GPQA questions and SimpleQA's 1,000, each answer at its whole budget
    assert est["n"] == 198 + 1000
    budget = runner.gen_thinking({}, {"archinfo": served.archinfo(rec)})["budget"]
    sq_out = served.settings_for(rec, {"archinfo": served.archinfo(rec)}, True)["max_tokens"]
    assert est["tokens_out"] == 198 * budget + 1000 * sq_out
    assert est["tokens_in"] == 198 * served.GEN_PROMPT_TOKENS["gpqa_diamond_cot_zeroshot"] + sum(
        served.tokens_of(q["prompt"]) for q in sq.load())
    assert est["line"].startswith("about $")
