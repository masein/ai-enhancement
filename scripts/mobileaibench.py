"""12o.3: two more of MobileAIBench's text sets, scored without a judge —
HotpotQA (answer from about ten given passages: on-phone search and
documents) and SQL from a question (sql_create_context: write the SQL for a
plain question, given the table's CREATE statement). 14.1: and the rest of
its text sets — Dolly (answer an instruction, with its context when it has
one), CNN/DailyMail and XSum (summarise an article), and MT-Bench (80
two-turn questions, scored 1–10 by the judge). AlpacaEval is left out: its
instructions and reference outputs are CC BY-NC 4.0.

The data is MobileAIBench's own sample of each
(github.com/SalesforceAIResearch/MobileAIBench, Apache-2.0), and MT-Bench's
judge prompts and GPT-4 reference answers from FastChat (Apache-2.0, the
commit MobileAIBench's submodule pins), all pinned in
eval_tasks/mobileaibench/manifest.json — commit and sha256 — so our numbers
compare with their paper's. HotpotQA is CC BY-SA 4.0 (Yang et al., 2018);
sql-create-context CC BY 4.0 (b-mc2, built from WikiSQL and Spider);
databricks-dolly-15k CC BY-SA 3.0; CNN/DailyMail Apache-2.0 (as its Hugging
Face card gives it); XSum's licence is not stated (BBC articles; Narayan et
al., 2018).

The prompts are theirs, word for word (src/data_processing/data_loader.py,
their typo and their escaped quotes included): a system line, then the
user's turn. Their Dolly sample is 1,000 closed-QA instructions, each with
its context. MT-Bench is FastChat's: no system line, the second turn asked
after the model's own first answer.

The metrics are theirs, ported from src/evaluation/evaluate.py and utils.py:
- HotpotQA and Dolly: exact match, F1 over normalize_answer's tokens, and
  BLEU (nltk's sentence_bleu with smoothing method 1, on the normalised
  strings — which nltk reads character by character). The column is F1;
- SQL: SQLParser F1 (the query's clauses, as sets) and the Levenshtein ratio;
  and, as a stricter reading, the normalised exact match. The column is
  SQLParser F1;
- CNN/DailyMail and XSum: ROUGE-1 and ROUGE-L F-measure, as rouge_score
  computes them with its Porter stemmer (ported: NLTK's stemmer, rouge_score's
  tokenizer and its LCS). The column is ROUGE-L. Word overlap with one
  reference summary: a good summary in other words scores low;
- MT-Bench: each turn rated 1–10 by the board's judge with FastChat's
  single-answer grading prompts (the "math" ones, with GPT-4's reference
  answer, for math, reasoning and coding), read from "[[rating]]" as FastChat
  reads it. The column is the mean of every rated turn, once every answered
  turn is rated; until then it says "awaiting judge".

Answers first, judging second: a run writes its answers and is done. Judging
is a step of its own (start_judge, and `--judge` below) that can run later —
a run never waits on a judge that is offline.

A thinking model's answer is the text after its thinking, as for every
written answer here (judge.answer_parts). Nothing here asks a model: the run
logs the answers and mark() reads them.

    python scripts/mobileaibench.py <results>/<model> …          marks again
    python scripts/mobileaibench.py --judge <results>/<model> …  judges what waits
"""

from __future__ import annotations

import ast
import collections
import csv
import hashlib
import io
import json
import math
import re
import string
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = REPO / "eval_tasks" / "mobileaibench"
MANIFEST = DATA_DIR / "manifest.json"
TEMPLATE = DATA_DIR / "_mab_template_yaml"
OUT_NAME = "mobileaibench.json"
# 14.1: the judge's ratings, by answer, kept apart from the answers they rate
JUDGED_NAME = "mobileaibench_judged.json"
HOTPOT, SQL = "mab_hotpotqa", "mab_sql"
DOLLY, CNNDM, XSUM = "mab_dolly", "mab_cnndm", "mab_xsum"
# MT-Bench is asked in two passes — the second turn after the model's own first
# answer — and is one column
MTB1, MTB2, MTBENCH = "mab_mtbench_t1", "mab_mtbench_t2", "mab_mtbench"
# asked, and scored with no judge
TASKS = (HOTPOT, SQL, DOLLY, CNNDM, XSUM)
# asked, and scored by the judge in a step of its own
JUDGED = (MTB1, MTB2)
ALL = TASKS + JUDGED
# the suite's two parts: no judge (5,000 answers), and judged
PARTS = {"": TASKS, "judged": JUDGED}
PART_WORDS = {"": "no judge", "judged": "judged"}
SOURCE = {HOTPOT: "hotpot_qa", SQL: "sql_create_context", DOLLY: "databricks_dolly_15k",
          CNNDM: "cnn_dailymail", XSUM: "edinburghNLP_xsum", MTB1: "mt_bench", MTB2: "mt_bench",
          MTBENCH: "mt_bench"}
LABEL = {HOTPOT: "HotpotQA", SQL: "SQL", DOLLY: "Dolly", CNNDM: "CNN/DailyMail", XSUM: "XSum",
         MTBENCH: "MT-Bench", MTB1: "MT-Bench, turn 1", MTB2: "MT-Bench, turn 2"}
# src/data_processing/data_loader.py, word for word
SYS_QA = ("You're a helpful assistant proficient in answering questions based on the provided "
          "context. Directly provide the final answer without any reasoning or justification.")
SYS_SQL = "You're a helpful assistant proficient in crafting SQL queries."
SYS_SUM = "You're a helpful assistant who is good at summarizing articles."
# MT-Bench: FastChat asks with no system line of the board's
SYSTEM = {HOTPOT: SYS_QA, SQL: SYS_SQL, DOLLY: SYS_QA, CNNDM: SYS_SUM, XSUM: SYS_SUM,
          MTB1: "", MTB2: ""}
# MT-Bench's answers get FastChat's room (max_new_token 1024) where the
# Everyday settings give 512; a thinking model keeps the Everyday room
MTB_MAX_GEN_TOKS = 1024
# FastChat's: the categories graded against GPT-4's reference answer
NEED_REF_CATS = ("math", "reasoning", "coding")
# the judge's settings, as FastChat's gen_judgment asks GPT-4
JUDGE_MAX_TOKENS = 2048
AWAITING = "awaiting judge"
# what an estimate assumes before a model has answered: an answer's tokens, a
# judgement's tokens (FastChat's GPT-4 explanations run a few sentences)
ANSWER_TOKENS_GUESS = 400
JUDGEMENT_TOKENS = 250
CHARS_PER_TOKEN = 4

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def credits() -> list[dict]:
    """MobileAIBench's sample, FastChat (MT-Bench's judge prompts and
    references), and every dataset under them, for the page"""
    m = manifest()
    return [m["sampled_by"], *([m["fastchat"]] if m.get("fastchat") else []),
            *m["sources"].values()]


def file_of(source: str) -> Path:
    return DATA_DIR / manifest()["files"][source]["file"]


def data_sha(task: str) -> str:
    return hashlib.sha256(file_of(SOURCE[task]).read_bytes()).hexdigest()


def _csv_rows(source: str) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    return list(csv.DictReader(io.StringIO(file_of(source).read_text(encoding="utf-8"))))


def _jsonl(source: str) -> list[dict]:
    return [json.loads(line) for line in file_of(source).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summary_prompt(article: str) -> str:
    """load_summarization_data's user turn — the article's quote marks escaped
    with a backslash, as their loader writes them"""
    return ("Create a short summary of the following article: "
            + article.replace("'", "\\'").replace('"', '\\"') + "\n")


def prompt_of(task: str, row: dict) -> str:
    """the user's turn as their loader builds it (the system line apart)"""
    if task in (HOTPOT, DOLLY):
        ctx = (row.get("context") or "").strip()
        q = row["question"] if task == HOTPOT else row["instruction"]
        return (f"context: {row.get('context')}\n" if ctx else "") + f"question: {q}\nanswer: "
    if task in (CNNDM, XSUM):
        return summary_prompt(row["article"] if task == CNNDM else row["document"])
    return (f"The following command was used to create the SQL table: {row['context']}\n"
            f"question: write a SQL query based on hte provided table for the following "
            f"inquiry: {row['question']}\nanswer: ")


def mt_bench() -> list[dict]:
    """FastChat's 80 questions, as MobileAIBench holds them: {question_id,
    category, turns}"""
    return _jsonl("mt_bench")


def mt_references() -> dict[int, list[str]]:
    """GPT-4's reference answers, by question, for the categories that need one"""
    return {r["question_id"]: r["choices"][0]["turns"] for r in _jsonl("mt_bench_reference")}


def judge_prompts() -> dict[str, dict]:
    """FastChat's judge prompts, by name (single-v1, single-math-v1, …)"""
    return {p["name"]: p for p in _jsonl("mt_bench_judge_prompts")}


def load(task: str) -> list[dict]:
    """{id, prompt, system, answer, …}: each question, as asked"""
    out = []
    if task in (MTB1, MTB2, MTBENCH):
        turn = 2 if task == MTB2 else 1
        refs = mt_references()
        for q in mt_bench():
            out.append({"id": f"mtb-{q['question_id']}", "question_id": q["question_id"],
                        "prompt": q["turns"][turn - 1], "turns": q["turns"], "turn": turn,
                        "category": q["category"], "system": "",
                        "reference": refs.get(q["question_id"]) or []})
        return out
    src = SOURCE[task]
    for i, r in enumerate(_csv_rows(src)):
        if task == HOTPOT:
            qid, ans, extra = f"hotpot-{r['id']}", r["answer"], {"type": r.get("type"),
                                                                 "level": r.get("level"),
                                                                 "question": r["question"]}
        elif task == SQL:
            qid, ans, extra = f"sql-{i:04d}", r["answer"], {"context": r["context"],
                                                           "question": r["question"]}
        elif task == DOLLY:
            qid, ans, extra = f"dolly-{i:04d}", r["response"], {"category": r.get("category"),
                                                               "question": r["instruction"]}
        elif task == CNNDM:
            qid, ans, extra = f"cnndm-{r['id']}", r["highlights"], {}
        else:
            qid, ans, extra = f"xsum-{r['id']}", r["summary"], {}
        out.append({"id": qid, "prompt": prompt_of(task, r), "system": SYSTEM[task],
                    "answer": ans, **extra})
    return out


def _yaml(items: Path, task: str, s: dict, max_gen: int | None = None) -> str:
    return (f"task: {task}\n" + TEMPLATE.read_text(encoding="utf-8")
            .replace("__ITEMS_PATH__", str(items.resolve()))
            .replace("__UNTIL__", json.dumps(s["until"]))
            .replace("__MAX_GEN_TOKS__", str(max_gen or s["max_gen_toks"]))
            .replace("__DO_SAMPLE__", "true" if s["do_sample"] else "false")
            .replace("__TEMPERATURE__", f"{float(s['temperature'])}"))


def build_tasks(dest: Path) -> Path:
    """every task the board asks from a pinned file, under `dest`: the
    prompts (never the answers) and the yaml, with the Everyday settings, as
    simpleqa.build_tasks writes its. MT-Bench's second turn is built for each
    model from its own first answers (build_turn2). Returns the directory for
    --include_path"""
    from everyday import run_settings
    dest.mkdir(parents=True, exist_ok=True)
    s = run_settings(None)
    for task in (*TASKS, MTB1):
        items, stamp, want = dest / f"{task}.jsonl", dest / f"{task}.sha256", data_sha(task)
        if not (items.exists() and stamp.exists()
                and stamp.read_text(encoding="utf-8").strip() == want):
            items.write_text("".join(json.dumps({"id": q["id"], "prompt": q["prompt"]},
                                                ensure_ascii=False) + "\n" for q in load(task)),
                             encoding="utf-8")
            stamp.write_text(want + "\n", encoding="utf-8")
        (dest / f"{task}.yaml").write_text(
            _yaml(items, task, s, MTB_MAX_GEN_TOKS if task == MTB1 else None), encoding="utf-8")
    # a stand-in for the second turn — its questions alone — so that lm_eval
    # finds the task here (check_tasks); a run asks its own (build_turn2)
    stand = dest / f"{MTB2}.jsonl"
    stand.write_text("".join(json.dumps({"id": q["id"], "prompt": q["prompt"]},
                                        ensure_ascii=False) + "\n" for q in load(MTB2)),
                     encoding="utf-8")
    (dest / f"{MTB2}.yaml").write_text(_yaml(stand, MTB2, s, MTB_MAX_GEN_TOKS), encoding="utf-8")
    return dest


def first_answers(model_dir: Path) -> dict[str, str]:
    """the model's answer to each MT-Bench question's first turn, its thinking
    taken out — what its second turn is asked after"""
    import judge as _judge
    out = {}
    for rec in _judge._records(model_dir, MTB1):
        qid = (rec.get("doc") or {}).get("id")
        if qid:
            out[qid] = _judge.answer_parts(rec)["answer_text"]
    return out


def turn2_docs(model_dir: Path, render=None) -> list[dict]:
    """the second turn of every question whose first the model answered: for a
    served model the conversation so far as `history` (asked as messages); for
    one run here, `render(messages)` — the conversation in the model's own chat
    template, as text — is the prompt"""
    firsts = first_answers(model_dir)
    out = []
    for q in load(MTB2):
        if q["id"] not in firsts:
            continue
        history = [{"role": "user", "content": q["turns"][0]},
                   {"role": "assistant", "content": firsts[q["id"]]}]
        if render is None:
            out.append({"id": q["id"], "prompt": q["prompt"], "history": history})
        else:
            out.append({"id": q["id"], "prompt": render(history + [{"role": "user",
                                                                    "content": q["prompt"]}]),
                        "rendered": True})
    return out


def build_turn2(dest: Path, model_dir: Path, render=None) -> Path:
    """MT-Bench's second turn for this model, under `dest` (a folder of the
    run's own: it is this model's conversation). Returns the directory for
    --include_path"""
    from everyday import run_settings
    dest.mkdir(parents=True, exist_ok=True)
    items = dest / f"{MTB2}.jsonl"
    items.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n"
                             for d in turn2_docs(model_dir, render)), encoding="utf-8")
    (dest / f"{MTB2}.yaml").write_text(_yaml(items, MTB2, run_settings(None), MTB_MAX_GEN_TOKS),
                                       encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# their metrics, ported (src/evaluation/evaluate.py and utils.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """utils.normalize_answer: lower case, no punctuation, no articles, one space"""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0


def f1(pred: str, gold: str) -> float:
    """compute_f1's F1: tokens of the normalised answers, counted"""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def _ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def bleu(pred: str, gold: str) -> float:
    """compute_BLEU: nltk's sentence_bleu([gold], pred, smoothing method1) on
    the normalised strings — which nltk takes as sequences of characters.
    The 4-gram geometric mean, clipped counts, the brevity penalty, and an
    epsilon of 0.1 for an n-gram order with no match"""
    ref, hyp = normalize_answer(gold), normalize_answer(pred)
    nums, dens = [], []
    for n in range(1, 5):
        h = collections.Counter(_ngrams(hyp, n))
        r = collections.Counter(_ngrams(ref, n))
        nums.append(sum(min(c, r[g]) for g, c in h.items()))
        dens.append(max(1, sum(h.values())))
    if nums[0] == 0:
        return 0.0
    hyp_len, ref_len = len(hyp), len(ref)
    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    ps = [Fraction(nu, de) if nu else (nu + 0.1) / de for nu, de in zip(nums, dens)]
    return bp * math.exp(math.fsum(0.25 * math.log(p) for p in ps))


def clean_qa(pred: str) -> str:
    """predict.clean_pred_QA (and clean_pred_summurization, the same): a label
    before the first colon (within 100 characters) is dropped — "Answer:
    Paris" is "Paris" """
    if ":" in pred and pred.index(":") < 100:
        pred = pred.split(":", 1)[1].strip()
    return pred


_BLOCK = re.compile(r"```[ \t]*(?:sql|SQL|sqlite|mysql)?[ \t]*\n?(.*?)```", re.S)
_SELECT = re.compile(r"^\s*(SELECT\b.*)$", re.I | re.M)


def extract_sql(pred: str) -> tuple[str, str]:
    """(the SQL, how it was found): from a code block if the answer has one,
    else the first line that starts with SELECT, else the answer as written
    ("none" — it scores what it scores)"""
    m = _BLOCK.search(pred or "")
    if m and m.group(1).strip():
        return " ".join(m.group(1).split()), "code block"
    m = _SELECT.search(pred or "")
    if m:
        return m.group(1).strip(), "SELECT line"
    return (pred or "").strip(), "none"


_SQL_PARTS = {
    "select": re.compile(r"SELECT\s+(.*?)\s+FROM", re.I),
    "from": re.compile(r"FROM\s+(.*?)\s+(WHERE|GROUP BY|ORDER BY|LIMIT|$)", re.I),
    "where": re.compile(r"WHERE\s+(.*?)(GROUP BY|ORDER BY|LIMIT|$)", re.I),
    "group_by": re.compile(r"GROUP BY\s+(.*?)(ORDER BY|LIMIT|$)", re.I),
    "order_by": re.compile(r"ORDER BY\s+(.*?)(LIMIT|$)", re.I),
    "limit": re.compile(r"LIMIT\s+(\d+)", re.I),
}


def parse_sql_to_dict(sql: str) -> list[tuple[str, object]]:
    """utils.parse_sql_to_dict: the query's clauses, as (clause, part)"""
    parsed = {k: [] for k in _SQL_PARTS}
    for key, pattern in _SQL_PARTS.items():
        m = pattern.search(sql)
        if not m:
            continue
        if key == "limit":
            parsed[key] = [int(m.group(1))]
            continue
        values = m.group(1).strip()
        if key in ("select", "from", "group_by", "order_by"):
            parsed[key] = [v.strip() for v in values.split(",")]
        else:
            parts = re.split(r"\s+(AND|OR)\s+", values, flags=re.I)
            parsed[key] = [parts[0].strip()]
            for i in range(1, len(parts), 2):
                parsed[key].append(parts[i].upper())
                parsed[key].append(parts[i + 1].strip())
    return [(k, v) for k, vs in parsed.items() for v in vs]


def _set_f1(a: list, b: list) -> float:
    """calculate_f1_score: over the parts as sets"""
    tp, fp, fn = len(set(a) & set(b)), len(set(a) - set(b)), len(set(b) - set(a))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def sqlparser_f1(pred: str, gold: str) -> float:
    """compute_sqlparser_score: the clauses, lower-cased, as sets"""
    g = [(k, str(v).lower()) for k, v in parse_sql_to_dict(gold)]
    p = [(k, str(v).lower()) for k, v in parse_sql_to_dict(pred)]
    return _set_f1(g, p)


def levenshtein_ratio(a: str, b: str) -> float:
    """Levenshtein.ratio: 1 − the indel distance over both lengths, which is
    twice the longest common subsequence over both lengths"""
    if not a and not b:
        return 1.0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return 2 * prev[-1] / (len(a) + len(b))


def sql_exact(pred: str, gold: str) -> int:
    """a stricter reading than theirs: equal once case, spacing and quote
    marks are set aside"""
    norm = lambda s: " ".join(re.sub(r"[\"'`]", "", s).lower().split()).rstrip(";").strip()  # noqa: E731
    return 1 if norm(pred) == norm(gold) else 0


# 14.1: ROUGE, as rouge_score (Google's) computes it for
# RougeScorer(['rouge1', 'rougeL'], use_stemmer=True).score(reference, summary):
# its tokenizer (lower case; anything not a-z or 0-9 a space; a token longer
# than three letters through NLTK's Porter stemmer, as rouge_score calls it),
# unigram overlap for ROUGE-1 and the longest common subsequence for ROUGE-L;
# each the F-measure of precision (over the summary) and recall (over the
# reference)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_VALID = re.compile(r"^[a-z0-9]+$")
_STEMMER = None


def _stem(w: str) -> str:
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import porter
        _STEMMER = porter.PorterStemmer()
    return _STEMMER.stem(w)


def rouge_tokens(text: str) -> list[str]:
    """rouge_score.tokenize.tokenize(text, PorterStemmer())"""
    toks = _NON_ALNUM.sub(" ", str(text).lower()).split()
    toks = [_stem(t) if len(t) > 3 else t for t in toks]
    return [t for t in toks if _VALID.match(t)]


def _fmeasure(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def rouge1(pred: str, ref: str) -> float:
    """rouge_scorer._score_ngrams(n=1): the F-measure of the unigrams in common"""
    t, p = collections.Counter(rouge_tokens(ref)), collections.Counter(rouge_tokens(pred))
    overlap = sum(min(c, p[g]) for g, c in t.items())
    precision = overlap / max(sum(p.values()), 1)
    recall = overlap / max(sum(t.values()), 1)
    return _fmeasure(precision, recall)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    """rouge_scorer._score_lcs: the F-measure of the longest common
    subsequence over the summary's and the reference's lengths"""
    t, p = rouge_tokens(ref), rouge_tokens(pred)
    if not t or not p:
        return 0.0
    n = _lcs(t, p)
    return _fmeasure(n / len(p), n / len(t))


# ---------------------------------------------------------------------------
# MT-Bench's judge, FastChat's prompts (fastchat/llm_judge/common.py)
# ---------------------------------------------------------------------------

def judge_request(q: dict, turn: int, a1: str, a2: str = "") -> tuple[str, str]:
    """(system, user) for rating one turn: single-v1 (or single-math-v1, with
    GPT-4's reference answer, for math, reasoning and coding); the second turn
    with the multi-turn prompts, the whole conversation in them"""
    math_ = q["category"] in NEED_REF_CATS
    name = ("single-math-v1" if math_ else "single-v1") + ("-multi-turn" if turn == 2 else "")
    p = judge_prompts()[name]
    ref = q.get("reference") or ["", ""]
    kw = {}
    if turn == 1:
        kw = {"question": q["turns"][0], "answer": a1}
        if math_:
            kw["ref_answer_1"] = ref[0]
    else:
        kw = {"question_1": q["turns"][0], "question_2": q["turns"][1], "answer_1": a1,
              "answer_2": a2}
        if math_:
            kw.update(ref_answer_1=ref[0], ref_answer_2=ref[1] if len(ref) > 1 else "")
    return p["system_prompt"], p["prompt_template"].format(**kw)


_RATING = re.compile(r"\[\[(\d+\.?\d*)\]\]")
_RATING_BACKUP = re.compile(r"\[(\d+\.?\d*)\]")


def parse_rating(text: str) -> float:
    """FastChat's run_judge_single: the [[rating]] in the judgement, else a
    [rating]; -1 when there is none (and the turn is left out of the mean, as
    FastChat leaves it). A judge that thinks is read after its thinking"""
    import judge as _judge
    t = _judge.split_reasoning(text or "")["answer_text"] or (text or "")
    m = _RATING.search(t) or _RATING_BACKUP.search(t)
    if not m:
        return -1.0
    try:
        return float(ast.literal_eval(m.group(1)))
    except (ValueError, SyntaxError):
        return -1.0


def stub_rating(answer: str) -> float:
    """the fake judge's rating: steady, and moved by the answer's length"""
    return float(min(10, 1 + len(str(answer).split()) // 20))


def stub_reply(system: str, user: str) -> str:
    m = re.findall(r"### Assistant A:\n(.*?)\n\n(?:### User:|<\|The End)", user, re.S)
    ans = m[-1] if m else (re.search(r"\[The Start of Assistant's Answer\]\n(.*?)\n\[The End",
                                     user, re.S) or [None, ""])[1]
    return f"A steady stub. Rating: [[{stub_rating(ans or '')}]]"


# ---------------------------------------------------------------------------
# a model's answers, marked
# ---------------------------------------------------------------------------

def score_one(task: str, answer: str, gold: str) -> dict:
    if task in (HOTPOT, DOLLY):
        a = clean_qa(answer)
        return {"em": exact_match(a, gold), "f1": round(f1(a, gold), 6),
                "bleu": round(bleu(a, gold), 6)}
    if task in (CNNDM, XSUM):
        # predict.clean_pred_summurization: the same label rule as an answer's
        a = clean_qa(answer)
        return {"rouge1": round(rouge1(a, gold), 6), "rougeL": round(rouge_l(a, gold), 6)}
    sql, how = extract_sql(answer)
    return {"sql": sql, "how": how, "sqlparser": round(sqlparser_f1(sql, gold), 6),
            "levenshtein": round(levenshtein_ratio(sql, gold), 6),
            "exact": sql_exact(sql, gold)}


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 6) if xs else None


def _se(xs: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return round(math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n), 6)


def _summary_of(task: str, items: list[dict]) -> dict:
    if task in (HOTPOT, DOLLY):
        col = [it["f1"] for it in items]
        return {"f1": _mean(col), "se": _se(col), "em": _mean([it["em"] for it in items]),
                "bleu": _mean([it["bleu"] for it in items])}
    if task in (CNNDM, XSUM):
        col = [it["rougeL"] for it in items]
        return {"rougeL": _mean(col), "se": _se(col),
                "rouge1": _mean([it["rouge1"] for it in items])}
    col = [it["sqlparser"] for it in items]
    return {"sqlparser_f1": _mean(col), "se": _se(col),
            "levenshtein": _mean([it["levenshtein"] for it in items]),
            "exact": _mean([it["exact"] for it in items]),
            "no_sql": sum(1 for it in items if it["how"] == "none")}


def read_judged(model_dir: Path) -> dict:
    try:
        got = json.loads((model_dir / JUDGED_NAME).read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def write_judged(model_dir: Path, got: dict) -> None:
    (model_dir / JUDGED_NAME).write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")


def verdict_key(task: str, qid: str, turn: int) -> str:
    return f"{task}:{qid}:{turn}"


def _judge_now() -> dict:
    """the judge the board runs now: {id, version}. A rating counts only with
    the judge that gave it (12i.1): another judge's is kept, not shown"""
    import judge as _judge
    try:
        ident = _judge.identity()
        return {"id": ident["id"], "version": _judge.version(ident)["key"]}
    except Exception:                                   # noqa: BLE001 — no judge set up
        return {"id": "", "version": ""}


def mt_answers(model_dir: Path) -> list[dict]:
    """every MT-Bench turn the model answered: {id, turn, category, answer_text,
    no_answer, first} — `first`, a second turn's own first answer"""
    import judge as _judge
    qs = {q["id"]: q for q in load(MTB1)}
    out = []
    firsts = {}
    for task, turn in ((MTB1, 1), (MTB2, 2)):
        for rec in _judge._records(model_dir, task):
            qid = (rec.get("doc") or {}).get("id")
            if qid not in qs:
                continue
            p = _judge.answer_parts(rec)
            text = p["answer_text"]
            if turn == 1:
                firsts[qid] = text
            out.append({"id": qid, "turn": turn, "category": qs[qid]["category"],
                        "answer_text": "" if p["no_answer"] else text,
                        "no_answer": bool(p["no_answer"])})
    for it in out:
        if it["turn"] == 2:
            it["first"] = firsts.get(it["id"], "")
    return out


def _mt_summary(items: list[dict], judge: dict) -> dict:
    rated = [it for it in items if it.get("score") is not None]
    # FastChat leaves out a judgement with no rating (-1), and only that
    good = [it["score"] for it in rated if it["score"] != -1]
    waiting = sum(1 for it in items if it.get("score") is None)
    turns = {t: _mean([it["score"] for it in rated if it["turn"] == t and it["score"] != -1])
             for t in (1, 2)}
    # the judge whose ratings these are (12i.1: one judge's), else the one now
    by = collections.Counter(it["judge"] for it in rated if it.get("judge")).most_common(1)
    judge = {**judge, "id": by[0][0]} if by else judge
    return {"n": len(items), "of": 2 * len(load(MTB1)), "rated": len(rated),
            "unreadable": len(rated) - len(good), "awaiting": waiting,
            # the column's number only once every answered turn is rated
            "score": _mean(good) if items and not waiting and good else None,
            "se": _se(good) if items and not waiting else None,
            "turn1": turns[1], "turn2": turns[2], "judge": judge}


def mark(model_dir: Path) -> dict | None:
    """every answer on file, scored: {task: {items, the column's number, the
    others}} — None when the model has answered none of them"""
    import judge as _judge
    out = {}
    for task in TASKS:
        recs = _judge._records(model_dir, task)
        if not recs:
            continue
        gold = {q["id"]: q for q in load(task)}
        items = []
        for rec in recs:
            qid = (rec.get("doc") or {}).get("id")
            q = gold.get(qid)
            if not q:
                continue
            parts = _judge.answer_parts(rec)
            ans = "" if parts["no_answer"] else parts["answer_text"]
            items.append({"id": qid, "answer_text": ans, "no_answer": bool(parts["no_answer"]),
                          **score_one(task, ans, q["answer"])})
        if not items:
            continue
        out[task] = {"n": len(items), "of": len(gold), "data": data_sha(task)[:12],
                     **_summary_of(task, items), "items": items}
    mt = mt_answers(model_dir)
    if mt:
        now = _judge_now()
        got = read_judged(model_dir).get("verdicts") or {}
        for it in mt:
            v = got.get(verdict_key(MTBENCH, it["id"], it["turn"]))
            ok = v and (not now["version"] or v.get("version") == now["version"])
            it["score"] = v["score"] if ok else None
            if ok:
                it["judge"] = v.get("judge", "")
        out[MTBENCH] = {**_mt_summary(mt, now), "data": data_sha(MTB1)[:12], "items": mt}
    if not out:
        return None
    return {"marked_at": time.time(), "tasks": out}


def write(model_dir: Path, out: dict) -> Path:
    p = model_dir / OUT_NAME
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return p


def read(model_dir: Path) -> dict | None:
    try:
        return json.loads((model_dir / OUT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summary(out: dict | None) -> str:
    """"HotpotQA F1 0.61 · SQL 0.78 · CNN/DailyMail ROUGE-L 0.21 · MT-Bench 6.3"
    — the run's progress line"""
    t = (out or {}).get("tasks") or {}
    bits = []
    for task, key, words in ((HOTPOT, "f1", "HotpotQA F1"), (SQL, "sqlparser_f1", "SQL"),
                             (DOLLY, "f1", "Dolly F1"), (CNNDM, "rougeL", "CNN/DailyMail ROUGE-L"),
                             (XSUM, "rougeL", "XSum ROUGE-L")):
        if task in t and t[task].get(key) is not None:
            bits.append(f"{words} {t[task][key]:.2f}")
    mt = t.get(MTBENCH)
    if mt:
        bits.append(f"MT-Bench {mt['score']:.2f}" if mt.get("score") is not None
                    else f"MT-Bench: {mt['awaiting']} of {mt['n']} turns {AWAITING}")
    return " · ".join(bits) or "MobileAIBench: nothing marked"


# ---------------------------------------------------------------------------
# judging: a step of its own
# ---------------------------------------------------------------------------

def pending(model_dir: Path, out: dict | None = None) -> list[dict]:
    """the answered MT-Bench turns no rating of the judge's now covers"""
    out = out if out is not None else mark(model_dir)
    mt = ((out or {}).get("tasks") or {}).get(MTBENCH) or {}
    return [it for it in mt.get("items") or [] if it.get("score") is None]


def _requests(model_dir: Path, todo: list[dict]) -> list:
    from service import llm
    qs = {q["id"]: q for q in load(MTB1)}
    reqs = []
    for it in todo:
        q = qs[it["id"]]
        a1 = it.get("first", "") if it["turn"] == 2 else it["answer_text"]
        system, user = judge_request(q, it["turn"], a1, it["answer_text"])
        reqs.append(llm.Request(custom_id=f"mab:{model_dir.name}:{MTBENCH}:{it['id']}:{it['turn']}",
                                system=system, user=user, max_tokens=JUDGE_MAX_TOKENS, json=False,
                                meta={"kind": "mab", "id": it["id"], "turn": it["turn"]}))
    return reqs


def _record(model_dir: Path, ratings: dict[tuple[str, int], float], judge: dict) -> None:
    got = read_judged(model_dir)
    v = got.setdefault("verdicts", {})
    for (qid, turn), score in ratings.items():
        v[verdict_key(MTBENCH, qid, turn)] = {"score": score, "judge": judge.get("id", ""),
                                              "version": judge.get("version", ""),
                                              "at": time.time()}
    write_judged(model_dir, got)


def start_judge(model_dir: Path, submission: int | None = None) -> dict:
    """Judge what waits: every answered turn with no rating of this judge's,
    sent as one batch. A judge that is offline, not set up, or that can't be
    reached leaves them waiting — "awaiting judge" — for a later step; the
    answers are never lost. Returns mobileaibench.json's content, with
    `batch_id` when a batch went out and `note` when it couldn't"""
    import judge as _judge
    from service import db, llm
    out = mark(model_dir)
    if out is None:
        raise RuntimeError("nothing of MobileAIBench's on file for this model")
    todo = pending(model_dir, out)
    if not todo:
        write(model_dir, out)
        return out
    now = _judge_now()
    if _judge.is_stub():
        qs = {q["id"]: q for q in load(MTB1)}
        _record(model_dir, {(it["id"], it["turn"]): parse_rating(stub_reply(*judge_request(
            qs[it["id"]], it["turn"], it.get("first", "") if it["turn"] == 2 else it["answer_text"],
            it["answer_text"]))) for it in todo}, now)
        out = mark(model_dir)
        write(model_dir, out)
        return out
    why = _judge.blocked()
    if why:
        out["note"] = f"{AWAITING}: {why}"
        write(model_dir, out)
        return out
    try:
        backend = llm.client("judge")
        reqs = _requests(model_dir, todo)
        bid = backend.submit(reqs)
    except Exception as e:                          # noqa: BLE001 — the answers are kept
        out["note"] = f"{AWAITING}: the judge could not be reached ({str(e)[:160]})"
        write(model_dir, out)
        return out
    got = read_judged(model_dir)
    got.setdefault("batches", {})[bid] = {"at": time.time(), "n": len(reqs), "judge": now,
                                          "submission": submission}
    write_judged(model_dir, got)
    db.batch_add(bid, "mab", submission or 0, len(reqs), backend.name, backend.model)
    db.batch_progress(bid, f"0/{len(reqs)} done")
    if submission:
        db.update(submission, judge_batch=bid)
    out["batch_id"] = bid
    write(model_dir, out)
    return out


def finish(out_dir: Path, results: dict) -> list[Path]:
    """The poller's half: the judge's ratings in, each model's file out. A
    custom id names its model's folder — a later judging step has no run of
    its own. Returns the folders written"""
    import judge as _judge
    now = _judge_now()
    by_dir: dict[str, dict] = {}
    for cid, res in results.items():
        parts = str(cid).split(":")
        if len(parts) != 5 or parts[0] != "mab" or parts[2] != MTBENCH:
            continue
        _, d, _, qid, turn = parts
        if getattr(res, "error", None):
            continue                          # stays awaiting: a later step asks again
        by_dir.setdefault(d, {})[(qid, int(turn))] = parse_rating(getattr(res, "text", ""))
    done = []
    for d, ratings in by_dir.items():
        model_dir = Path(out_dir) / d
        if not model_dir.is_dir():
            continue
        _record(model_dir, ratings, now if now["id"] else {"id": _judge.identity().get("id", "")})
        out = mark(model_dir)
        if out:
            write(model_dir, out)
            done.append(model_dir)
    return done


def judge_failed(out_dir: Path, batch_id: str, why: str) -> None:
    """the batch failed: the answers it carried wait again, saying why"""
    for model_dir in (p for p in Path(out_dir).iterdir() if p.is_dir()) if Path(out_dir).is_dir() else []:
        got = read_judged(model_dir)
        if batch_id not in (got.get("batches") or {}):
            continue
        out = mark(model_dir)
        if out:
            out["note"] = f"{AWAITING}: the judge's batch failed ({why[:120]})"
            write(model_dir, out)


def awaiting(out_dir: Path) -> list[Path]:
    """every model folder with an answered turn no rating of this judge's covers"""
    return [d for d in sorted(Path(out_dir).iterdir()) if d.is_dir()
            and (d / OUT_NAME).exists() and pending(d)] if Path(out_dir).is_dir() else []


# ---------------------------------------------------------------------------
# what a run takes, before Start
# ---------------------------------------------------------------------------

def part_counts(part: str) -> dict:
    """the questions, answers and (judged) judgements a model's run of a part
    makes: {questions, answers, judgements}"""
    if part == "judged":
        n = len(mt_bench())
        return {"questions": n, "prompts": 2 * n, "answers": 2 * n, "judgements": 2 * n}
    n = sum(len(load(t)) for t in TASKS)
    return {"questions": n, "prompts": n, "answers": n, "judgements": 0}


def judge_tokens(answer_tokens: int | None = None) -> dict:
    """the judge's tokens for one model's MT-Bench, about: every judgement's
    prompt (FastChat's template, the question or the conversation, GPT-4's
    reference where it is used) with each answer `answer_tokens` long (the
    model's median, when it has one), and a judgement of JUDGEMENT_TOKENS"""
    a = int(answer_tokens or ANSWER_TOKENS_GUESS)
    tin = 0
    for q in load(MTB1):
        for turn in (1, 2):
            system, user = judge_request(q, turn, "", "")
            tin += (len(system) + len(user)) // CHARS_PER_TOKEN + a * turn
    n = 2 * len(mt_bench())
    return {"judgements": n, "tokens_in": tin, "tokens_out": n * JUDGEMENT_TOKENS,
            "answer_tokens": a, "guess": answer_tokens is None}


# the page's own guesses before a model has answered here: a served answer
# (SERVED_GUESS_S, the Everyday estimate's) and one generated on this server's GPU
SERVED_GUESS_S = 5.0
HF_GUESS_S = 1.0


def _dur(seconds: float) -> str:
    return f"about {seconds / 3600:.1f} h" if seconds >= 5400 else \
        f"about {max(1, round(seconds / 60))} min"


def estimate(served: bool, secs_each: float | None = None, judge: dict | None = None,
             answer_tokens: int | None = None) -> dict:
    """What each part of a model's run takes, before Start: its answers and
    time (at `secs_each`, the model's own measured seconds an answer, else
    the page's guess), and for the judged part the judge's work — its
    judgements and tokens, and, for a judge with prices (one from
    OpenRouter), what they cost. `judge`: {id, label, price_in, price_out,
    local} (prices per million tokens). {"none": …, "judged": …}"""
    each = secs_each if secs_each and secs_each > 0 else (SERVED_GUESS_S if served else HF_GUESS_S)
    out = {}
    for part in PARTS:
        c = part_counts(part)
        sec = c["answers"] * each
        p = {**c, "seconds": round(sec), "each": round(each, 3),
             "measured": bool(secs_each and secs_each > 0),
             "line": f"{c['answers']:,} answers, {_dur(sec)}"
                     + ("" if secs_each and secs_each > 0 else ", a rough guess")}
        if part == "judged":
            jt = judge_tokens(answer_tokens)
            j = judge or {}
            priced = not j.get("local") and j.get("price_in") is not None \
                and j.get("price_out") is not None
            usd = round((j["price_in"] * jt["tokens_in"] + j["price_out"] * jt["tokens_out"]) / 1e6,
                        2) if priced else None
            p["judge"] = {**jt, "id": j.get("id", ""), "label": j.get("label", ""), "usd": usd,
                          "line": f"{jt['judgements']} judgements · about "
                                  f"{jt['tokens_in'] + jt['tokens_out']:,} judge tokens "
                                  f"({jt['tokens_in']:,} in, {jt['tokens_out']:,} out)"
                                  + (f" · about ${usd:,.2f}" if usd is not None else
                                     " · on this server, no charge" if j.get("local") else "")}
        out[part or "none"] = p
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    judge_now = argv[0] == "--judge"
    dirs = [Path(a) for a in (argv[1:] if judge_now else argv)]
    for d in dirs:
        out = start_judge(d) if judge_now else mark(d)
        if out and not judge_now:
            write(d, out)
        print(f"{d.name}: {summary(out)}" + (f" · judge batch {out['batch_id']}"
                                             if (out or {}).get("batch_id") else "")
              + (f" · {out['note']}" if (out or {}).get("note") else ""))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))                   # service/, for the judge
    raise SystemExit(main(sys.argv[1:]))
