#!/usr/bin/env python3
"""IFEval, MMLU-Pro and MATH-500 (brief 12h.1): read what a chat model answered.

These three generate text, and an instruct model answers in its own words:
"The answer is (C).", "**C**", "I think it's C because…", a worked solution
ending in \\boxed{\\frac{1}{2}}, sometimes after a <think> block. The harness's
own readers look for one shape each — MMLU-Pro's regex wants "answer is (X)",
and MATH's scorer compares the text between the first and last "$" by string
(lm_eval issue: hendrycks_math scores 0 on chat models) — so this layer reads
the answers again from the samples the harness logged, and scores:

  MMLU-Pro   the letter A–J the answer settles on (below: extract_letter),
             exact match against the key. Beyond lm_eval: bold letters, "I
             think it's C", "Answer: C", a line that opens with "(C)", and the
             one option whose full text the answer names.
  MATH-500   the final answer — the last \\boxed{}, else "the answer is …",
             else the last number — compared with math-verify, so \\frac{1}{2},
             1/2 and 0.5 are one answer. lm_eval compares strings.
  IFEval     the harness's own checker decides; it already reads free text.
             This layer reads its verdicts, on the answer after any thinking
             (lm_eval strips it with think_end_token), and counts the answers
             that never left their thinking.

Every task also counts the answers that ran out of room — a thinking model
still thinking when its budget ended — as 12a.4 does for Everyday tasks.
Those answers fail; the count is shown beside the score.

    python scripts/generative.py results/full            re-score every model
    python scripts/generative.py results/full -m org/x   one model

It writes generative.json beside the model's results; the report reads the
two scores it re-reads from there, and says so in each column's tooltip.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge as _judge  # noqa: E402

OUT_NAME = "generative.json"
# the three, in the order the page shows them
TASKS = {"ifeval": "IFEval", "mmlu_pro": "MMLU-Pro", "hendrycks_math500": "MATH-500"}
LETTERS = "ABCDEFGHIJ"
# what each score is, in the words its column's tooltip uses
SCORERS = {
    "ifeval": "the harness's IFEval checker, on the answer after any thinking: prompt-level "
              "strict accuracy first, instruction-level beside it",
    "mmlu_pro": "exact match on the letter A–J the answer settles on, read by this board "
                "(scripts/generative.py): \"answer is (C)\", \"**C**\", \"I think it's C\", "
                "or the one option it names in full",
    "hendrycks_math500": "exact match on the final answer — the last \\boxed{}, else the "
                         "stated answer, else the last number — compared with math-verify, "
                         "so 1/2 and 0.5 are one answer",
}


# ---------------------------------------------------------------------------
# the answer, after any thinking
# ---------------------------------------------------------------------------

def raw_generation(rec: dict) -> str:
    """What the model wrote, as logged — `resps`, never `filtered_resps`: for
    MMLU-Pro the filtered text is the harness's own regex result, which is
    exactly what is being re-read here"""
    r = rec.get("resps") or rec.get("filtered_resps") or []
    while isinstance(r, list) and r:
        r = r[0]
    return str(r) if isinstance(r, str) else ""


def answer_of(rec: dict) -> dict:
    """{answer_text, had_reasoning, no_answer, raw}: the text after any
    thinking, and whether the model never left it (ran out of room)"""
    raw = raw_generation(rec)
    p = _judge.split_reasoning(raw)
    return {"answer_text": p["answer_text"], "had_reasoning": p["had_reasoning"],
            "no_answer": p["had_reasoning"] and (p["reasoning_unterminated"]
                                                 or not p["answer_text"]),
            "raw": raw}


# ---------------------------------------------------------------------------
# MMLU-Pro: the letter
# ---------------------------------------------------------------------------

_L = r"\(?\**\(?([A-J])\)?\**(?![A-Za-z0-9])"


def _w(words: str) -> str:
    """words in any case; the letter itself stays a capital, so the article
    "a" is never an answer"""
    return "(?i:" + words + ")"


# said outright: "The answer is (C).", "Answer: C", "**Final answer:** C",
# "the correct option is C". The last one said wins: a model that reconsiders
# ends on its answer
_SAID = [re.compile(p) for p in (
    _w(r"answer") + r"\s*" + _w(r"(?:is|:|=)") + r"\s*" + _w(r"(?:option\s*|choice\s*)?")
    + r"\**\s*" + _L,
    _w(r"(?:correct|right|best|final)\s+(?:answer|option|choice)") + r"\s*"
    + _w(r"(?:is|:|=|would be)?") + r"\s*" + _w(r"(?:option\s*|choice\s*)?") + r"\**\s*" + _L,
    _w(r"(?:option|choice)") + r"\s+" + _L + r"\s*" + _w(r"is\s+(?:correct|right|the answer)"),
)]
# chosen in passing: "I think it's C because…", "I'd go with (B)"
_CHOSEN = re.compile(_w(r"(?:\bit'?s|\bit is|\bgo with|\bpick|\bchoose|\bselect)")
                     + r"\s+" + _w(r"(?:option\s+)?") + _L)
# a bold letter on its own: "**C**", "**(C)**", "**C.**"
_BOLD = re.compile(r"\*\*\s*\(?([A-J])\)?[.:)]?\s*\*\*")
# a line that opens with the letter: "(C) 42", "C) Paris", "C. Paris", or just "C"
_LEAD = re.compile(r"^\s*\(?([A-J])(?:[).:]|\s*$)", re.M)


def extract_letter(text: str, options: list[str] | None = None) -> str | None:
    """The letter an answer settles on, or None when it names none"""
    t = (text or "").replace("’", "'")
    n = len(options) if options else 10
    ok = set(LETTERS[:n])
    for pat in _SAID:
        hits = [m.group(1) for m in pat.finditer(t) if m.group(1) in ok]
        if hits:
            return hits[-1]
    for pat in (_CHOSEN, _BOLD):
        hits = [m.group(1) for m in pat.finditer(t) if m.group(1) in ok]
        if hits:
            return hits[-1]
    hits = [m.group(1) for m in _LEAD.finditer(t) if m.group(1) in ok]
    if hits:
        return hits[0]
    # the one option it names in full ("The capital is Canberra."), when only
    # one does and the option is more than a word or number of a letter or two
    if options:
        low = t.lower()
        named = [LETTERS[i] for i, o in enumerate(options)
                 if len(str(o).strip()) >= 4 and re.search(
                     r"(?<![\w])" + re.escape(str(o).strip().lower()) + r"(?![\w])", low)]
        if len(named) == 1:
            return named[0]
    return None


def mmlu_pro_key(doc: dict) -> str | None:
    a = doc.get("answer")
    if isinstance(a, str) and a.strip().upper() in LETTERS:
        return a.strip().upper()
    i = doc.get("answer_index")
    return LETTERS[i] if isinstance(i, int) and 0 <= i < len(LETTERS) else None


# ---------------------------------------------------------------------------
# MATH-500: the final answer, and math-verify
# ---------------------------------------------------------------------------

def last_boxed(text: str) -> str | None:
    """The contents of the last \\boxed{…} (or \\fbox{…}), braces matched"""
    t = text or ""
    i = max(t.rfind("\\boxed"), t.rfind("\\fbox"))
    if i < 0:
        return None
    j = t.find("{", i)
    if j < 0 or t[i:j].strip() not in ("\\boxed", "\\fbox"):
        m = re.match(r"\\boxed\s+([^\s$]+)", t[i:])        # "\boxed 5"
        return m.group(1) if m else None
    depth = 0
    for k in range(j, len(t)):
        if t[k] == "{":
            depth += 1
        elif t[k] == "}":
            depth -= 1
            if depth == 0:
                return t[j + 1:k].strip()
    return None


_STATED = re.compile(r"(?:final answer|the answer|answer)\s*(?:is|:|=)\s*(.+)", re.I)
_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def _tidy(s: str) -> str:
    s = s.strip().strip("*").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    return s.rstrip(".").strip()


def extract_math(text: str) -> str | None:
    """The final answer a solution gives, or None"""
    t = text or ""
    b = last_boxed(t)
    if b:
        return _tidy(b)
    stated = [m.group(1) for m in _STATED.finditer(t)]
    if stated:
        s = stated[-1].split("\n")[0]
        # "the answer is 12 because…" keeps "12"
        s = re.split(r"\s+(?:because|since|as|so)\b", s)[0]
        s = _tidy(s)
        if s:
            return s
    nums = _NUM.findall(t)
    return nums[-1].replace(",", "") if nums else None


def _plain_equal(a: str, b: str) -> bool:
    """The fallback when math-verify is not installed: the same text once
    spaces, $ and \\left/\\right go, or the same number"""
    def norm(x):
        x = re.sub(r"\\left|\\right|\\!|\\,|\\;|\$|\s", "", x or "")
        x = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
        return x.replace("\\dfrac", "\\frac").rstrip(".")
    if norm(a) == norm(b):
        return True
    try:
        from fractions import Fraction
        f = lambda x: Fraction(norm(x).replace("(", "").replace(")", ""))   # noqa: E731
        return f(a) == f(b)
    except (ValueError, ZeroDivisionError):
        return False


def math_equal(pred: str | None, gold: str | None) -> bool:
    """Is the answer the key? math-verify when it is installed (the image and
    the local check both carry it), else a plain comparison"""
    if not pred or not gold:
        return False
    try:
        from math_verify import parse, verify
    except ImportError:
        return _plain_equal(pred, gold)

    def p(x):
        try:
            return parse(f"${x}$", parsing_timeout=None)
        except TypeError:                       # an older math-verify
            return parse(f"${x}$")
    try:
        g, a = p(gold), p(pred)
        try:
            return bool(verify(g, a, timeout_seconds=None))
        except TypeError:
            return bool(verify(g, a))
    except Exception:                           # noqa: BLE001 — unparseable is not equal
        return _plain_equal(pred, gold)


# ---------------------------------------------------------------------------
# one answer, read and scored
# ---------------------------------------------------------------------------

def read(task: str, rec: dict) -> dict:
    """{extracted, key, correct, ran_out, answer_text}: one logged answer,
    read the way this layer reads it"""
    a = answer_of(rec)
    doc = rec.get("doc") or {}
    out = {"answer_text": a["answer_text"], "ran_out": a["no_answer"]}
    if task.startswith("mmlu_pro"):
        key = mmlu_pro_key(doc)
        got = None if a["no_answer"] else extract_letter(a["answer_text"], doc.get("options"))
        out.update(extracted=got, key=key, correct=bool(got and key and got == key))
    elif task.startswith("hendrycks_math"):
        key = doc.get("answer") or (rec.get("target") if isinstance(rec.get("target"), str)
                                     else None)
        got = None if a["no_answer"] else extract_math(a["answer_text"])
        out.update(extracted=got, key=key, correct=math_equal(got, key))
    elif task == "ifeval":
        strict = rec.get("prompt_level_strict_acc")
        inst = rec.get("inst_level_strict_acc")
        out.update(extracted=a["answer_text"],
                   key=", ".join(doc.get("instruction_id_list") or []) or None,
                   correct=bool(strict) and not a["no_answer"],
                   instructions=(list(inst) if isinstance(inst, list) else None))
    else:
        raise ValueError(f"not one of the three: {task}")
    return out


def _se(p: float, n: int) -> float:
    """the standard error of a mean of 0/1 answers, as the harness reports it"""
    return math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else 0.0


def score(task: str, recs: list[dict]) -> dict | None:
    """One task's score from its logged answers"""
    if not recs:
        return None
    rows = [read(task, r) for r in recs]
    n = len(rows)
    right = sum(r["correct"] for r in rows)
    acc = right / n
    out = {"n": n, "correct": right, "acc": acc, "stderr": _se(acc, n),
           "ran_out": sum(r["ran_out"] for r in rows),
           "unreadable": sum(1 for r in rows if not r["ran_out"] and r["extracted"] in (None, "")),
           "scorer": SCORERS[task]}
    if task == "ifeval":
        inst = [x for r in rows for x in (r.get("instructions") or [])]
        if inst:
            out["inst_acc"] = sum(bool(x) for x in inst) / len(inst)
    return out


# ---------------------------------------------------------------------------
# a model
# ---------------------------------------------------------------------------

def mark(model_dir: Path, extra: dict | None = None) -> dict | None:
    """Every one of the three this model answered, scored — and what
    generative.json holds; None when it answered none"""
    tasks = {}
    for task in TASKS:
        s = score(task, _judge._records(model_dir, task))
        if s:
            gen = _judge._generation(model_dir, task) or {}
            tasks[task] = {**s, "settings": gen}
    if not tasks:
        return None
    prev = read_json(model_dir) or {}
    return {"model": prev.get("model") or _model_id(model_dir), "marked_at": time.time(),
            **{k: v for k, v in prev.items() if k in ("thinking", "backend", "subset")},
            **(extra or {}), "tasks": tasks}


def read_json(model_dir: Path) -> dict | None:
    try:
        return json.loads((model_dir / OUT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(model_dir: Path, out: dict) -> Path:
    p = model_dir / OUT_NAME
    p.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return p


def _model_id(model_dir: Path) -> str:
    try:
        return json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))["model"]
    except (OSError, ValueError, KeyError):
        return model_dir.name.replace("__", "/", 1)


def summary(out: dict) -> str:
    """The queue row's words: "IFEval 61.2 · MMLU-Pro 40.1 · MATH-500 22.4" """
    parts = []
    for t, name in TASKS.items():
        s = (out.get("tasks") or {}).get(t)
        if s:
            parts.append(f"{name} {100 * s['acc']:.1f}"
                         + (f" ({s['ran_out']} ran out of room)" if s.get("ran_out") else ""))
    return " · ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("-m", "--model", action="append", default=[])
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    want = {m.replace("/", "__") for m in a.model}
    n = 0
    for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        out = mark(d)
        if out is None:
            continue
        write(d, out)
        n += 1
        print(f"{out['model']}: {summary(out)}")
    print(f"scored {n} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
