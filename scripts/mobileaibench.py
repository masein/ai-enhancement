"""12o.3: two more of MobileAIBench's text sets, scored without a judge —
HotpotQA (answer from about ten given passages: on-phone search and
documents) and SQL from a question (sql_create_context: write the SQL for a
plain question, given the table's CREATE statement).

The data is MobileAIBench's own 1,000-row sample of each
(github.com/SalesforceAIResearch/MobileAIBench, Apache-2.0), pinned in
eval_tasks/mobileaibench/manifest.json — commit and sha256 — so our numbers
compare with their paper's. HotpotQA is CC BY-SA 4.0 (Yang et al., 2018);
sql-create-context is CC BY 4.0 (b-mc2, built from WikiSQL and Spider).

The prompts are theirs, word for word (src/data_processing/data_loader.py,
their typo included): a system line, then "context: …\\nquestion: …\\nanswer: ".
The metrics are theirs, ported from src/evaluation/evaluate.py and utils.py:
- HotpotQA: exact match, F1 over normalize_answer's tokens, and BLEU (nltk's
  sentence_bleu with smoothing method 1, on the normalised strings — which
  nltk reads character by character). The column is F1;
- SQL: SQLParser F1 (the query's clauses, as sets) and the Levenshtein ratio;
  and, as a stricter reading, the normalised exact match. The column is
  SQLParser F1.

A thinking model's answer is the text after its thinking, as for every
written answer here (judge.answer_parts). Nothing here asks a model: the run
logs the answers and mark() reads them."""

from __future__ import annotations

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
HOTPOT, SQL = "mab_hotpotqa", "mab_sql"
TASKS = (HOTPOT, SQL)
SOURCE = {HOTPOT: "hotpot_qa", SQL: "sql_create_context"}
LABEL = {HOTPOT: "HotpotQA", SQL: "SQL"}
# src/data_processing/data_loader.py, word for word
SYS_QA = ("You're a helpful assistant proficient in answering questions based on the provided "
          "context. Directly provide the final answer without any reasoning or justification.")
SYS_SQL = "You're a helpful assistant proficient in crafting SQL queries."
SYSTEM = {HOTPOT: SYS_QA, SQL: SYS_SQL}

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def credits() -> list[dict]:
    """MobileAIBench's sample, and the two datasets under it, for the page"""
    m = manifest()
    return [m["sampled_by"], *m["sources"].values()]


def data_sha(task: str) -> str:
    return hashlib.sha256((DATA_DIR / manifest()["files"][SOURCE[task]]["file"]).read_bytes()
                          ).hexdigest()


def _rows(task: str) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    f = DATA_DIR / manifest()["files"][SOURCE[task]]["file"]
    return list(csv.DictReader(io.StringIO(f.read_text(encoding="utf-8"))))


def prompt_of(task: str, row: dict) -> str:
    """the user's turn as their loader builds it (the system line apart)"""
    if task == HOTPOT:
        ctx = row.get("context") or ""
        return (f"context: {ctx}\n" if ctx else "") + f"question: {row['question']}\nanswer: "
    return (f"The following command was used to create the SQL table: {row['context']}\n"
            f"question: write a SQL query based on hte provided table for the following "
            f"inquiry: {row['question']}\nanswer: ")


def load(task: str) -> list[dict]:
    """{id, prompt, system, answer, question}: each question, as asked"""
    out = []
    for i, r in enumerate(_rows(task)):
        qid = f"hotpot-{r['id']}" if task == HOTPOT else f"sql-{i:04d}"
        out.append({"id": qid, "prompt": prompt_of(task, r), "system": SYSTEM[task],
                    "answer": r["answer"], "question": r["question"],
                    **({"type": r.get("type"), "level": r.get("level")} if task == HOTPOT
                       else {"context": r["context"]})})
    return out


def build_tasks(dest: Path) -> Path:
    """both tasks under `dest`: the prompts (never the answers) and the yaml,
    with the Everyday settings, as simpleqa.build_tasks writes its. Returns
    the directory for --include_path"""
    from everyday import run_settings
    dest.mkdir(parents=True, exist_ok=True)
    s = run_settings(None)
    for task in TASKS:
        items, stamp, want = dest / f"{task}.jsonl", dest / f"{task}.sha256", data_sha(task)
        if not (items.exists() and stamp.exists()
                and stamp.read_text(encoding="utf-8").strip() == want):
            items.write_text("".join(json.dumps({"id": q["id"], "prompt": q["prompt"]},
                                                ensure_ascii=False) + "\n" for q in load(task)),
                             encoding="utf-8")
            stamp.write_text(want + "\n", encoding="utf-8")
        yaml = (TEMPLATE.read_text(encoding="utf-8")
                .replace("__ITEMS_PATH__", str(items.resolve()))
                .replace("__UNTIL__", json.dumps(s["until"]))
                .replace("__MAX_GEN_TOKS__", str(s["max_gen_toks"]))
                .replace("__DO_SAMPLE__", "true" if s["do_sample"] else "false")
                .replace("__TEMPERATURE__", f"{float(s['temperature'])}"))
        (dest / f"{task}.yaml").write_text(f"task: {task}\n" + yaml, encoding="utf-8")
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
    """predict.clean_pred_QA: a label before the first colon (within 100
    characters) is dropped — "Answer: Paris" is "Paris" """
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


# ---------------------------------------------------------------------------
# a model's answers, marked
# ---------------------------------------------------------------------------

def score_one(task: str, answer: str, gold: str) -> dict:
    if task == HOTPOT:
        a = clean_qa(answer)
        return {"em": exact_match(a, gold), "f1": round(f1(a, gold), 6),
                "bleu": round(bleu(a, gold), 6)}
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


def mark(model_dir: Path) -> dict | None:
    """every answer on file, scored: {task: {items, the column's number, the
    others}} — None when the model has answered neither"""
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
        if task == HOTPOT:
            col = [it["f1"] for it in items]
            summary = {"f1": _mean(col), "se": _se(col),
                       "em": _mean([it["em"] for it in items]),
                       "bleu": _mean([it["bleu"] for it in items])}
        else:
            col = [it["sqlparser"] for it in items]
            summary = {"sqlparser_f1": _mean(col), "se": _se(col),
                       "levenshtein": _mean([it["levenshtein"] for it in items]),
                       "exact": _mean([it["exact"] for it in items]),
                       "no_sql": sum(1 for it in items if it["how"] == "none")}
        out[task] = {"n": len(items), "of": len(gold), "data": data_sha(task)[:12],
                     **summary, "items": items}
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
    """"HotpotQA F1 0.61 · SQL 0.78" — the run's progress line"""
    t = (out or {}).get("tasks") or {}
    bits = []
    if HOTPOT in t and t[HOTPOT].get("f1") is not None:
        bits.append(f"HotpotQA F1 {t[HOTPOT]['f1']:.2f}")
    if SQL in t and t[SQL].get("sqlparser_f1") is not None:
        bits.append(f"SQL {t[SQL]['sqlparser_f1']:.2f}")
    return " · ".join(bits) or "MobileAIBench: nothing marked"
