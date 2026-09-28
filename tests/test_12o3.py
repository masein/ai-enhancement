"""12o.3: two more of MobileAIBench's text sets, scored without a judge —
HotpotQA (answer from about ten given passages) and SQL from a question.
Their own 1,000-row samples, pinned; their prompts, word for word; their
metrics, ported and checked here on answers worked out by hand. A new suite,
"mobile": instruct models only, never in the Avg or Improve. Nothing asks a
model: a fake GPU and answers written in."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import mobileaibench as mab
from conftest import make_service
from service import config, db, runner, served

STAMP = "2026-09-28T10-00-00.000000"


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod
    client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

def test_the_manifest_pins_both_files_and_credits_every_licence():
    m = mab.manifest()
    assert m["sampled_by"]["revision"] == "cff7b48f3b1e0c3e2549c2ae211ce14a4ceba8d1"
    assert m["sampled_by"]["licence"] == "Apache-2.0"
    assert m["sources"]["hotpot_qa"]["licence"] == "CC BY-SA 4.0"
    assert m["sources"]["sql_create_context"]["licence"] == "CC BY 4.0"
    for key, f in m["files"].items():
        data = (mab.DATA_DIR / f["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"] and len(data) == f["bytes"], key
    for t in mab.TASKS:
        qs = mab.load(t)
        assert len(qs) == 1000 and len({q["id"] for q in qs}) == 1000
        assert all(q["answer"].strip() for q in qs)


def test_the_prompts_are_mobileaibenchs_word_for_word():
    h = mab.load(mab.HOTPOT)[0]
    assert h["system"] == ("You're a helpful assistant proficient in answering questions based "
                           "on the provided context. Directly provide the final answer without "
                           "any reasoning or justification.")
    assert h["prompt"].startswith("context: ") and h["prompt"].endswith(
        f"\nquestion: {h['question']}\nanswer: ")
    s = mab.load(mab.SQL)[0]
    assert s["system"] == "You're a helpful assistant proficient in crafting SQL queries."
    assert s["prompt"] == (
        f"The following command was used to create the SQL table: {s['context']}\nquestion: "
        f"write a SQL query based on hte provided table for the following inquiry: "
        f"{s['question']}\nanswer: ")


def test_the_tasks_are_built_from_the_pinned_files(tmp_path):
    d = mab.build_tasks(tmp_path / "t")
    for t in mab.TASKS:
        yaml = (d / f"{t}.yaml").read_text()
        assert yaml.startswith(f"task: {t}\n") and str((d / f"{t}.jsonl").resolve()) in yaml
        rows = [json.loads(x) for x in (d / f"{t}.jsonl").read_text().splitlines()]
        assert len(rows) == 1000 and set(rows[0]) == {"id", "prompt"}      # never the answers


# ---------------------------------------------------------------------------
# their metrics, on answers worked out by hand
# ---------------------------------------------------------------------------

def test_hotpotqas_em_f1_and_bleu():
    gold = "Knox County Regional Airport"
    assert mab.normalize_answer("The Knox County Regional Airport!") == \
        "knox county regional airport"
    assert mab.exact_match("the Knox county regional airport.", gold) == 1
    assert mab.exact_match("Knox County", gold) == 0
    assert mab.f1(gold, gold) == 1.0
    # 3 tokens, all in the 4 of the reference: P 1, R 3/4
    assert mab.f1("Knox County airport", gold) == pytest.approx(2 * 0.75 / 1.75)
    assert mab.f1("Sacramento", gold) == 0.0
    # BLEU, as nltk's sentence_bleu reads two strings: character n-grams. "ab"
    # against "abc": unigrams 2/2, bigrams 1/1, no trigram or 4-gram (0.1
    # each, method 1), and the brevity penalty exp(1 − 3/2)
    assert mab.bleu("ab", "abc") == pytest.approx(math.exp(-0.5) * math.sqrt(0.1))
    assert mab.bleu(gold, gold) == pytest.approx(1.0)
    assert mab.bleu("", gold) == 0.0
    # their cleaning: a label before a colon, within 100 characters, is dropped
    assert mab.clean_qa("Answer: Knox County Regional Airport") == gold
    assert mab.score_one(mab.HOTPOT, "Answer: Knox County Regional Airport", gold) == {
        "em": 1, "f1": 1.0, "bleu": 1.0}


def test_sqls_sqlparser_f1_levenshtein_and_exact_match():
    gold = 'SELECT venue FROM table_name_50 WHERE away_team = "essendon"'
    assert mab.parse_sql_to_dict(gold) == [("select", "venue"), ("from", "table_name_50"),
                                           ("where", 'away_team = "essendon"')]
    assert mab.sqlparser_f1(gold, gold) == 1.0
    # the WHERE clause differs in its quotes: two of three clauses each way
    assert mab.sqlparser_f1("SELECT venue FROM table_name_50 WHERE away_team = 'Essendon'",
                            gold) == pytest.approx(2 / 3)
    assert mab.parse_sql_to_dict("SELECT a, b FROM t WHERE x = 1 AND y = 2 LIMIT 5") == [
        ("select", "a"), ("select", "b"), ("from", "t"), ("where", "x = 1"),
        ("where", "AND"), ("where", "y = 2"), ("limit", 5)]
    # Levenshtein.ratio: twice the longest common subsequence over both lengths
    assert mab.levenshtein_ratio("kitten", "sitting") == pytest.approx(8 / 13)
    assert mab.levenshtein_ratio("", "") == 1.0
    # stricter than theirs: equal once case, spacing and quote marks are set aside
    assert mab.sql_exact('SELECT venue FROM t WHERE a = "x";', "select venue  from t where a = 'x'")
    assert not mab.sql_exact("SELECT venue FROM t", "SELECT year FROM t")


@pytest.mark.parametrize("answer, sql, how", [
    ("Here it is:\n```sql\nSELECT venue\nFROM t WHERE a = 1\n```\nDone.",
     "SELECT venue FROM t WHERE a = 1", "code block"),
    ("```\nSELECT name FROM t\n```", "SELECT name FROM t", "code block"),
    ("Sure! The query is:\nSELECT COUNT(*) FROM t WHERE x = 2\nThis counts them.",
     "SELECT COUNT(*) FROM t WHERE x = 2", "SELECT line"),
    ("I can't write SQL for that.", "I can't write SQL for that.", "none"),
])
def test_the_sql_is_taken_from_a_code_block_else_a_select_line(answer, sql, how):
    assert mab.extract_sql(answer) == (sql, how)


# ---------------------------------------------------------------------------
# the suite and a run
# ---------------------------------------------------------------------------

def test_the_suite_is_instruct_only(svc):
    client, _ = svc
    assert config.tasks_for_suite("mobile") == ["mab_hotpotqa", "mab_sql"]
    r = client.post("/api/submissions", json={"hf_id": "org/base-1b", "suite": "mobile",
                                              "kind": "base"})
    assert r.status_code == 422 and r.json()["detail"] == \
        config.MAB_INSTRUCT_ONLY + ". Nothing was queued."
    assert "mobile" in served.SUITES


def sit_mab(model_dir: Path, answer=lambda q: q["answer"]) -> None:
    """what lm_eval logs for a run of both: the prompts and the answers"""
    for t in mab.TASKS:
        d = model_dir / f"{t}_0shot" / "pretrained__x"
        d.mkdir(parents=True, exist_ok=True)
        qs = mab.load(t)
        with open(d / f"samples_{t}_{STAMP}.jsonl", "w", encoding="utf-8") as fh:
            for i, q in enumerate(qs):
                doc = {"id": q["id"], "prompt": q["prompt"]}
                fh.write(json.dumps({"doc_id": i, "doc": doc, "doc_hash": hashlib.sha256(
                    q["id"].encode()).hexdigest(), "arguments": [[q["prompt"], {}]],
                    "resps": [[answer(q)]], "filtered_resps": [answer(q)], "bypass": 999}) + "\n")
        mid = model_dir.name.replace("__", "/", 1)
        (d / f"results_{STAMP}.json").write_text(json.dumps({
            "results": {t: {"alias": t, "bypass,none": 999}}, "group_subtasks": {t: []},
            "n-shot": {t: 0}, "n-samples": {t: {"original": len(qs), "effective": len(qs)}},
            "higher_is_better": {t: {"bypass": True}},
            "config": {"model": "hf", "model_args": f"pretrained={mid},dtype=bfloat16"},
            "chat_template": True, "date": 1790100000.0}), encoding="utf-8")


def fake_gpu(monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda hf_id, k, **kw: {
        "kind": "instruct", "params": 70_000_000, "vocab": 50304, "batch": 8, "need_gb": 2.0,
        "remote_code": False, "has_template": True, "archinfo": {}})
    monkeypatch.setattr(runner, "acquire_lock", lambda sid: True)
    monkeypatch.setattr(runner, "release_lock", lambda: None)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: 10 ** 6)
    seen = []

    def run(sid, cmd, *a, **k):
        seen.append(cmd)
        model_dir = Path(cmd[cmd.index("--output_path") + 1]).parent
        if cmd[cmd.index("--tasks") + 1] == mab.SQL:       # both written once both are asked
            sit_mab(model_dir)
        return 0
    monkeypatch.setattr(runner, "_run_task", run)
    return seen


def test_a_run_asks_both_with_their_system_line_and_scores_them(svc, monkeypatch):
    client, _ = svc
    seen = fake_gpu(monkeypatch)
    sid = client.post("/api/submissions", json={"hf_id": "org/chat-1b", "suite": "mobile",
                                                "kind": "instruct"}).json()["id"]
    runner.run_submission(db.get(sid))
    assert [c[c.index("--tasks") + 1] for c in seen] == ["mab_hotpotqa", "mab_sql"]
    for cmd, t in zip(seen, mab.TASKS):
        assert cmd[cmd.index("--include_path") + 1] == str(config.MAB_TASKS_DIR)
        assert cmd[cmd.index("--system_instruction") + 1] == mab.SYSTEM[t]
        assert "--apply_chat_template" in cmd
    out = mab.read(config.OUT_DIR / "org__chat-1b")
    # the reference itself scores all but what their normaliser empties (an
    # answer of only an article, or punctuation, has no words to match)
    assert out["tasks"]["mab_hotpotqa"]["f1"] > 0.999 and out["tasks"]["mab_sql"]["n"] == 1000
    assert out["tasks"]["mab_sql"]["sqlparser_f1"] > 0.99 and out["tasks"]["mab_sql"]["exact"] == 1
    row = db.get(sid)
    assert row["status"] == "done" and row["progress"].startswith("HotpotQA F1 1.00 · SQL")


def test_a_served_model_is_asked_with_the_system_line(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MAB_TASKS_DIR", mab.build_tasks(tmp_path / "t"))
    got = {}

    def answer_task(rec, task, docs, task_out, s, everyday, **k):
        got.update(task=task, n=len(docs), s=s, everyday=everyday)
        return 0
    monkeypatch.setattr(served, "answer_task", answer_task)
    rec = {"thinking": "off", "name": "phone", "base_url": "http://x", "pin": {"file": "f.gguf"}}
    status, stopped = runner._ask_served(1, rec, {}, mab.SQL, tmp_path / "out", "2/2 · sql",
                                         tmp_path / "log", False, None)
    assert (status, stopped) == (0, None)
    assert got["n"] == 1000 and got["s"]["system"] == mab.SYSTEM[mab.SQL] and got["everyday"]


def test_openrouters_estimate_counts_both(monkeypatch, tmp_path):
    rec = {"id": "served/openrouter-openai-gpt-6-luna", "name": "GPT-6 Luna", "via": "openrouter",
           "thinking": "off", "base_url": "https://openrouter.ai/api/v1", "how": "OpenRouter",
           "based_on": "",
           "pin": {"model": "openai/gpt-6-luna", "price_in": 2.0, "price_out": 8.0,
                   "provider": "openai", "version": "openai/gpt-6-luna-2026-09-01"}}
    monkeypatch.setattr(served, "model_dir", lambda r: tmp_path / "none")
    est = served.estimate(rec, "mobile")
    # both sets' 1,000 prompts as the run sends them, each with its system line
    st = served.settings_for(rec, {"archinfo": served.archinfo(rec)}, True)
    assert est["n"] == 2000 and est["tokens_out"] == 2000 * st["max_tokens"]
    assert est["tokens_in"] == sum(served.tokens_of(mab.SYSTEM[t]) + served.tokens_of(q["prompt"])
                                   for t in mab.TASKS for q in mab.load(t))


# ---------------------------------------------------------------------------
# on the board: a column each, never in the Avg or Improve
# ---------------------------------------------------------------------------

def test_the_cells_are_theirs_and_never_in_the_avg(svc):
    client, appmod = svc
    d = config.OUT_DIR / "fx__good-750m"
    appmod._cache.update(key=None, payload=None, at=0.0)
    before = next(m for m in client.get("/api/results").json()["models"]
                  if m["id"] == "fx/good-750m")
    sit_mab(d, answer=lambda q: "no idea")
    mab.write(d, mab.mark(d))
    appmod._cache.update(key=None, payload=None, at=0.0)
    data = client.get("/api/results").json()
    m = next(x for x in data["models"] if x["id"] == "fx/good-750m")
    got = mab.read(d)["tasks"]
    assert data["cells"]["mab_hotpotqa"]["fx/good-750m"]["v"] == got["mab_hotpotqa"]["f1"]
    assert data["cells"]["mab_sql"]["fx/good-750m"]["v"] == got["mab_sql"]["sqlparser_f1"]
    assert data["tasks"]["mab_sql"]["metric"] == "sqlparser_f1"
    assert m["avg"] == before["avg"] and m["official"] == before["official"]
    assert "mab_hotpotqa" not in data["required"] and data["mabTasks"] == ["mab_hotpotqa",
                                                                          "mab_sql"]
    assert m["mab"]["mab_sql"]["no_sql"] == 1000
    assert data["mab"]["credits"][0]["name"] == "MobileAIBench"


def test_bleu_is_nltks_where_nltk_is_installed():
    """the image has nltk (IFEval's checker needs it): there, the port is
    checked against the library MobileAIBench calls"""
    nltk_bleu = pytest.importorskip("nltk.translate.bleu_score")
    smooth = nltk_bleu.SmoothingFunction().method1
    for pred, gold in (("ab", "abc"), ("Knox County airport", "Knox County Regional Airport"),
                       ("Paris", "paris"), ("the answer is 42", "42"), ("x", "a long answer")):
        want = nltk_bleu.sentence_bleu([mab.normalize_answer(gold)], mab.normalize_answer(pred),
                                       smoothing_function=smooth)
        assert mab.bleu(pred, gold) == pytest.approx(want), (pred, gold)


def test_neither_is_ever_a_training_target(tmp_path):
    """Improve starts from the diagnosis: a run of both leaves it nothing to
    diagnose — no metric it reads, so no example and no target"""
    import diagnose as dx
    d = tmp_path / "org__m"
    sit_mab(d)
    assert not [t for t in dx.diagnose_model(d)["tasks"] if t.startswith("mab_")]
