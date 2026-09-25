#!/usr/bin/env python3
"""Everyday tasks (briefs 12a, 12a.2, 12a.3): mark what a model answered.

Everyday tasks asks what people type into an assistant on a phone — short,
lowercase, typos, one plain request — and most answers can be marked by a
script, so the score needs no judge. The bank is 333 questions in seven
groups (eval_tasks/everyday/bank.jsonl), all of them readable. It is a
look, not a benchmark: never ranked, never averaged into anything, never on
the Leaderboard, never read by Propose, never sent to a generator. Its
questions are not split into practice and hidden halves yet; 12g.2 does
that, and nothing here reads a `split`.

    python scripts/everyday.py results/full            re-mark every model
    python scripts/everyday.py results/full -m org/x   one model

It reads the generations the harness logged, marks each one on the text
after the reasoning block (judge.answer_parts, #59's split — never the raw
generation), and writes everyday.json beside the model's results. Each
check says pass or fail and one reason in plain words, because the reason
is what the page shows. The few questions with a judge check go to the
judge, with the rubric the question carries, once their script checks
pass; re-marking keeps a verdict while the answer is the same one it read.
English only (2026-09-24).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import judge as _judge  # noqa: E402

# one bank, the harness task "everyday": round 2's 106 questions and the
# pilot's five (12a.2), and round 3's 222 (12a.3), 333 in all. A model that
# sat the pilot logged it as "everyday_pilot", and those answers are still read
TASK = "everyday"
LEGACY_TASKS = ("everyday_pilot",)
BANK_DIR = REPO / "eval_tasks" / "everyday"
BANK_PATH = BANK_DIR / "bank.jsonl"
TEMPLATE_PATH = BANK_DIR / "_everyday_template_yaml"
OUT_NAME = "everyday.json"
# the seven groups, in this order everywhere, as the page shows them
GROUPS = {"understanding": "Understanding", "writing": "Writing",
          "summarising": "Summarising", "transform": "Transform",
          "quick_maths": "Quick maths", "instructions": "Instructions", "honesty": "Honesty"}
# 12a.4: the bank's version is the date its wording last changed and a short
# hash of the question texts. A run's version is the hash of the questions it
# was asked — the harness logs each one — so answers to an earlier wording are
# never marked by today's checks, counted in today's score or compared with
# today's runs. Reword a question, and this date changes with it
# (tests/test_everyday_12a4.py pins the hash beside it).
WORDING_DATE = "2026-09-25"
NEVER_FINISHED = "never finished answering"
WAITING = "waiting for the judge"


def _words(x) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(v, str) and v for v in x)


def _bad_shape(c: dict) -> str:
    """12a.3's two check types, read as the checker reads them: a fact given
    as a string would be read letter by letter, and pass on nearly anything"""
    if c["type"] == "facts":
        if not (isinstance(c["values"], list) and c["values"]
                and all(_words(f) for f in c["values"])):
            return "facts needs values: a list of facts, each a list of ways to say it"
        if not (isinstance(c["n"], int) and 0 < c["n"] <= len(c["values"])):
            return f"facts needs n between 1 and {len(c['values'])}"
    if c["type"] == "first_mention":
        for k in ("right", "wrong"):
            if not _words(c[k]):
                return f"first_mention needs {k}: a list of names"
    return ""


def _bad_check(c) -> str:
    """what is wrong with one check, or '' — an `any` holds checks of its
    own, each read the same way"""
    t = c.get("type") if isinstance(c, dict) else None
    if t not in NEEDS:
        return f"unknown check type {t!r}"
    miss = [k for k in NEEDS[t] if k not in c]
    if miss:
        return f"{t} needs {', '.join(miss)}"
    if t == "any":
        if not (isinstance(c["checks"], list) and c["checks"]):
            return "any needs checks: a list of checks"
        return next((why for why in map(_bad_check, c["checks"]) if why), "")
    return _bad_shape(c)


def load_bank(path: Path = BANK_PATH) -> list[dict]:
    """The bank, checked as it is read: an invalid line, a duplicate id, an
    unknown group or an unknown check type fails here, naming the line —
    never later, on a model's answer."""
    out, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except ValueError as e:
            raise ValueError(f"{path.name} line {n}: not valid JSON ({e})") from None
        if not isinstance(q, dict):
            raise ValueError(f"{path.name} line {n}: not a JSON object")
        for key in ("id", "group", "prompt", "checks"):
            if not q.get(key):
                raise ValueError(f"{path.name} line {n}: no {key}")
        if q["id"] in seen:
            raise ValueError(f"{path.name} line {n}: {q['id']} is there twice")
        seen.add(q["id"])
        if q["group"] not in GROUPS:
            raise ValueError(f"{path.name} line {n}: unknown group {q['group']!r}")
        if not isinstance(q["checks"], list):
            raise ValueError(f"{path.name} line {n}: checks is not a list")
        for c in q["checks"]:
            why = _bad_check(c)
            if why:
                raise ValueError(f"{path.name} line {n}: {why}")
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# the check vocabulary (12a.2, 12a.3, 12a.4). docs/prompts/phase-12a4/checks.py
# is the reference, and replaces 12a.3's: every check below decides exactly as
# it does — the same regexes, the same normalising, the same reasons — and
# tests/test_everyday_12a4.py holds the two to the same verdict on all 382
# probes. What this adds is words: each check says what it looks for in plain
# words (the bank's page), and a failing check says why (the answer's row).
# ---------------------------------------------------------------------------

NUM = re.compile(r'(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?')


def _norm(s: str) -> str:
    return s.replace('’', "'").replace('½', ' 1/2')


_RANGE = re.compile(r'(?<![\d:])(\d{1,2}(?::\d\d)?)\s*[-–—]\s*(\d{1,2}(?::\d\d)?)\s*'
                    r'(am|pm|a\.m\.|p\.m\.)(?!\w)', re.I)


def _ampm(x: str) -> str:
    """12:30pm reads as 12:30 pm, on both sides of a comparison. 12a.3: a
    range says each of its times — "7-11 am", "2–3pm" and "9:30–11:30 am"
    read as "7 am-11 am" — so "7 am" is found in "7–11 am"."""
    x = _RANGE.sub(r'\1 \3-\2 \3', x)
    return re.sub(r'(\d)\s*(am|pm|a\.m\.|p\.m\.)(?!\w)', r'\1 \2', x, flags=re.I)


def _at(text_low: str, v: str) -> list[int]:
    """where `v` starts in lowercased, am/pm-read text, as a whole word"""
    return [m.start() for m in
            re.finditer(r'(?<!\w)' + re.escape(_ampm(v).lower()) + r'(?!\w)', text_low)]


def _has(text: str, v: str, cs: bool = False) -> bool:
    """`v` as a whole word or number, case-insensitive unless asked: "2
    november" is not in "22 november"."""
    text, v = _ampm(text), _ampm(v)
    t, v = (text, v) if cs else (text.lower(), v.lower())
    return re.search(r'(?<!\w)' + re.escape(v) + r'(?!\w)', t) is not None


def _lines(a: str) -> list[str]:
    """non-empty lines, without code fences and one lead-in line ending in
    ':' ("Here you go:")"""
    ls = [ln for ln in a.splitlines() if ln.strip() and not ln.strip().startswith('```')]
    if len(ls) > 1 and ls[0].rstrip().endswith(':'):
        ls = ls[1:]
    return ls


def _body(a: str) -> str:
    return '\n'.join(_lines(a))


def numbers(text: str) -> list[float]:
    """every number in the text; thousands commas are read (1,284.50)"""
    out = []
    for m in NUM.finditer(text):
        try:
            out.append(float(m.group().replace(',', '')))
        except ValueError:
            pass
    return out


def first_json(text: str):
    """the first JSON object or array in the text, fenced or not"""
    t = re.sub(r'```(?:json)?', '', text)
    starts = [k for k, ch in enumerate(t) if ch in '{[']
    for i in starts:
        op = t[i]
        cl = '}' if op == '{' else ']'
        depth = 0
        for j in range(i, len(t)):
            if t[j] == op:
                depth += 1
            elif t[j] == cl:
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(t[i:j + 1])
                        if isinstance(o, (dict, list)) and o:
                            return o
                    except ValueError:
                        pass
                    break
    return None


def _flat(o) -> list[str]:
    if isinstance(o, dict):
        return [x for v in o.values() for x in _flat(v)]
    if isinstance(o, list):
        return [x for v in o for x in _flat(v)]
    return [str(o).lower()]


INVENTED = {
    'phone': r'(?:\+?\d[\d\s-]{6,}\d)',
    'url': r'https?://|www\.|\b\w+\.(?:com|ae|net|org)\b',
    'email': r'\b[\w.]+@[\w.]+\b',
    # 12a.4: "$25" is caught — `\b` before `$` never matched — and € and £,
    # eur, gbp, dollars, euros and pounds too
    'price': r'(?:(?<!\w)(?:aed|dhs?|usd|eur|gbp)\s?\d)|(?:[$€£]\s?\d)|(?:\d\s?'
             r'(?:k|m|mn|million|thousand)?\s?'
             r'(?:aed|dhs|dirhams?|usd|eur|gbp|dollars?|euros?|pounds?)\b)',
    'distance': r'\d+(?:\.\d+)?\s?(?:km|kms|kilomet\w*|metres?|meters?|m\b|mins?\b|minutes?)',
}
def _digits(x: str) -> str:
    return re.sub(r'\D', '', x)


# 12a.4: the usual ways of saying it can't know or can't do something — one
# list for every admits_limit check, where each question used to carry its own
ADMITS = ["can't", "cannot", "can not", "unable", "not able", "no way to", "no way of",
          "no access", "don't have", "do not have", "doesn't have",
          "don't know", "do not know", "not sure which", "i'm not sure", "not possible",
          "isn't possible", "can't see", "don't see", "do not see",
          "not attached", "isn't attached", "wasn't attached", "no attachment", "no photo",
          "no image", "no live", "no real-time", "not aware",
          "came through", "didn't come through", "didn't receive", "haven't received",
          "wasn't received", "nothing came through", "nothing was attached",
          "as an ai", "i'm an ai", "i am an ai", "ai assistant", "language model", "text-based",
          "beyond my", "outside my", "not something i can"]
EMOJI = re.compile('[\U0001F000-\U0001FAFF☀-➿⭐⭕✅❌❤️]')
_ABBR = re.compile(r'\b(?:dr|mr|mrs|ms|st|e\.g|i\.e|etc)\.', re.I)


def run_check(check: dict, answer: str, prompt: str) -> tuple[bool | None, str]:
    """One check on one answer: (True, '') or (False, why). A judge check is
    (None, 'judge'): the script cannot decide it."""
    a = _norm(answer)
    t = check['type']
    cs = check.get('case_sensitive', False)
    if t == 'contains_any':
        ok = any(_has(a, v, cs) for v in check['values'])
        # 12a.4: "never mentions", not "says none of", which read as if the
        # answer was not allowed to say it
        return ok, '' if ok else 'never mentions: ' + ', '.join(check['values'][:3])
    if t == 'contains_all':
        miss = [v for v in check['values'] if not _has(a, v, cs)]
        return not miss, 'missing: ' + ', '.join(miss) if miss else ''
    if t == 'not_contains':
        bad = [v for v in check['values'] if _has(a, v, cs)]
        return not bad, 'still says: ' + ', '.join(bad) if bad else ''
    if t == 'number':
        ok = any(abs(n - check['value']) <= check['tolerance'] for n in numbers(a))
        return ok, '' if ok else f"didn't say {check['value']:g}"
    if t == 'json':
        o = first_json(a)
        if o is None:
            return False, 'not valid JSON'
        vals = _flat(o)
        miss = [alts[0] for alts in check['required_values']
                if not any(x.lower() in v for x in alts for v in vals)]
        return not miss, 'missing: ' + ', '.join(miss) if miss else ''
    if t == 'line_count':
        n = len(_lines(a))
        return n == check['n'], f'{n} lines, expected {check["n"]}'
    if t == 'max_words':
        n = len(a.split())
        return n <= check['n'], f'{n} words, limit {check["n"]}'
    if t == 'in_order':
        # 12a.3: am/pm read as the contains checks read them ("8am" is "8 am")
        pos, low = 0, _ampm(a).lower()
        for v in check['values']:
            alts = v if isinstance(v, list) else [v]
            hits = [h for x in alts for h in _at(low[pos:], x)]
            if not hits:
                return False, 'wrong order'
            pos += min(hits) + 1
        return True, ''
    if t == 'asks_back':
        ok = '?' in a
        return ok, '' if ok else "didn't ask what you meant"
    if t == 'no_invented':
        # 12a.3: what the question itself holds is not invented — repeating
        # the caller's number back names it, it doesn't make one up. Phone,
        # price and distance compare digits; url and email, the text
        src = _digits(prompt)
        for m in re.finditer(INVENTED[check['what']], a, re.I):
            d = _digits(m.group())
            if check['what'] in ('phone', 'price', 'distance') and d and d in src:
                continue
            if check['what'] in ('url', 'email') and m.group().lower() in prompt.lower():
                continue
            return False, f"made up a {check['what']}"
        return True, ''
    if t == 'numbers_from_source':
        src = set(numbers(_norm(prompt)))
        extra = [n for n in numbers(a) if n not in src]
        return not extra, f'invented {extra[0]:g}' if extra else ''
    if t == 'word_count':
        n = len(_body(a).split())
        return n == check['n'], f'{n} words, expected {check["n"]}'
    if t == 'sentence_count':
        b = _ABBR.sub(lambda m: m.group().replace('.', ''), _body(a))
        n = len([x for x in re.split(r'(?<=[.!?])\s+', b.strip()) if re.search(r'\w', x)])
        return n == check['n'], f'{n} sentences, expected {check["n"]}'
    if t == 'no_emoji':
        bad = EMOJI.search(a)
        return not bad, 'used an emoji' if bad else ''
    if t == 'no_digits':
        bad = re.search(r'\d', a)
        return not bad, 'used a number' if bad else ''
    if t == 'facts':
        # 12a.3: at least n of the listed facts, each a list of ways to say it.
        # A good summary keeps most key facts, not every one
        got = [f for f in check['values'] if any(_has(a, v, cs) for v in f)]
        miss = [f[0] for f in check['values'] if f not in got]
        return len(got) >= check['n'], (f"kept {len(got)} of {len(check['values'])} key facts, "
                                         f"needs {check['n']} (missing: {', '.join(miss[:3])})")
    if t == 'first_mention':
        # 12a.3: the right choice is named before any wrong one — "Message
        # Nadia, not Nabil" passes, "Nabil" fails
        low = _ampm(a).lower()

        def first(vals):
            ps = [p for v in vals for p in _at(low, v)]
            return min(ps) if ps else None
        r, w = first(check['right']), first(check['wrong'])
        if r is None:
            return False, 'never names ' + check['right'][0]
        ok = w is None or r < w
        return ok, '' if ok else 'names ' + check['wrong'][0] + ' first'
    if t == 'admits_limit':
        # 12a.4: says it can't know or can't do it, in any of the usual ways
        ok = any(_has(a, v) for v in ADMITS)
        return ok, '' if ok else "never says it can't know or do this"
    if t == 'any':
        # 12a.4: passes when any one of its checks passes ("asks what you
        # meant, or says it can't know")
        res = [run_check(c, answer, prompt) for c in check['checks']]
        if any(r[0] is True for r in res):
            return True, ''
        if any(r[0] is None for r in res):
            return None, 'judge'
        return False, ' and '.join(r[1] for r in res)
    if t == 'judge':
        return None, 'judge'
    raise ValueError(f"unknown check type: {t}")


def describe(check: dict) -> str:
    """What a check looks for, in plain words — the bank's page"""
    t = check['type']
    vals = check.get('values') or []
    q = lambda v: '"' + (' or '.join(v) if isinstance(v, list) else v) + '"'   # noqa: E731
    if t == 'contains_any':
        return 'says ' + ' or '.join(f'"{v}"' for v in vals[:3])
    if t == 'contains_all':
        return 'says ' + ', '.join(f'"{v}"' for v in vals)
    if t == 'not_contains':
        return 'doesn’t say ' + ' or '.join(f'"{v}"' for v in vals)
    if t == 'number':
        return f"says {check['value']:g}"
    if t == 'json':
        return 'valid JSON with ' + ', '.join(alts[0] for alts in check['required_values'])
    if t == 'line_count':
        return f"{check['n']} lines"
    if t == 'max_words':
        return f"at most {check['n']} words"
    if t == 'word_count':
        return f"exactly {check['n']} words"
    if t == 'sentence_count':
        return f"{check['n']} sentence" + ('' if check['n'] == 1 else 's')
    if t == 'in_order':
        return 'in this order: ' + ', then '.join(q(v) for v in vals)
    if t == 'asks_back':
        return 'asks what you meant'
    if t == 'no_invented':
        return f"no invented {check['what']}"
    if t == 'numbers_from_source':
        return 'no number the question doesn’t give'
    if t == 'no_emoji':
        return 'no emoji'
    if t == 'no_digits':
        return 'no numbers'
    if t == 'facts':
        return (f"keeps at least {check['n']} of: "
                + ', '.join(f'"{f[0]}"' for f in check['values']))
    if t == 'first_mention':
        return (f"picks {check['right'][0]}, not "
                + ' or '.join(check['wrong']))
    if t == 'admits_limit':
        return "says it can't know or do this"
    if t == 'any':
        return ', or '.join(describe(c) for c in check['checks'])
    if t == 'judge':
        return 'the judge: ' + check['rubric']
    raise ValueError(f"unknown check type: {t}")


# what each type needs besides its type, for the import to refuse a bad line
NEEDS = {'contains_any': ('values',), 'contains_all': ('values',), 'not_contains': ('values',),
         'number': ('value', 'tolerance'), 'json': ('required_values',), 'line_count': ('n',),
         'max_words': ('n',), 'word_count': ('n',), 'sentence_count': ('n',),
         'in_order': ('values',), 'asks_back': (), 'no_invented': ('what',),
         'numbers_from_source': (), 'no_emoji': (), 'no_digits': (), 'judge': ('rubric',),
         'facts': ('n', 'values'), 'first_mention': ('right', 'wrong'),
         'admits_limit': (), 'any': ('checks',)}


def grade(item: dict, answer: str) -> tuple[bool | None, str]:
    """All of a question's checks: (True, what it did), (False, why not), or
    (None, …) when the script checks pass and the judge decides. A judge is
    never asked about an answer a script check has already failed."""
    res = [(c, *run_check(c, answer, item['prompt'])) for c in item['checks']]
    fails = [why for _, ok, why in res if ok is False]
    if fails:
        return False, '; '.join(fails)
    if any(ok is None for _, ok, _ in res):
        return None, WAITING
    return True, ' · '.join(describe(c) for c, _, _ in res)


# ---------------------------------------------------------------------------
# the one question the judge marks
# ---------------------------------------------------------------------------

# 12b.3: the question carries its rubric; nothing here knows what it asks
JUDGE_PROMPT = """You are checking ONE answer from an assistant against a rubric. Read the question, the rubric and the answer, and decide whether the answer passes the rubric. Reply with one JSON object and nothing else: {{"pass": <true or false>, "reason": <one short sentence for a person, about the answer>}}.

QUESTION
{question}

RUBRIC
{rubric}

THE ANSWER
{answer}"""


def judge_prompt(item: dict, answer: str) -> str:
    rubric = next(c["rubric"] for c in item["checks"] if c["type"] == "judge")
    return JUDGE_PROMPT.format(question=item["prompt"], rubric=rubric,
                               answer=answer.strip() or "(empty)")


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


PILOT_NOTICE = "school will close early at 11:30"


def stub_verdict(answer: str, question: str = PILOT_NOTICE) -> dict:
    """The deterministic stand-in the tests and dry runs use. The pilot's
    TL;DR (the school notice) it checks: at most two sentences, 11:30, and
    Thursday. Any other judged question has passed its script checks before
    it reaches a judge, and the stand-in agrees with them. 12a.3: the notice
    is known by its text — round 3 has seven more "tldr" questions."""
    if PILOT_NOTICE not in (question or ""):
        return {"pass": True, "reason": "the stand-in judge agrees with the script checks"}
    a = (answer or "").strip()
    if not a:
        return {"pass": False, "reason": "no answer"}
    if "11:30" not in a:
        return {"pass": False, "reason": "it doesn't give the 11:30 closing time"}
    if "thursday" not in a.lower():
        return {"pass": False, "reason": "it doesn't say Thursday"}
    if _sentences(a) > 2:
        return {"pass": False, "reason": "more than two sentences"}
    return {"pass": True, "reason": "closes 11:30 on Thursday, in two sentences or fewer"}


def stub_reply(prompt: str) -> str:
    """What the fake judge backend answers to a judge_prompt()."""
    answer = prompt.rsplit("THE ANSWER\n", 1)[-1]
    question = prompt.split("QUESTION\n", 1)[-1].split("\n\nRUBRIC", 1)[0]
    return json.dumps(stub_verdict(answer, question), ensure_ascii=False)


def parse_verdict(text: str) -> dict | None:
    from service import llm
    obj = llm.extract_json(text or "")
    if not isinstance(obj, dict) or not isinstance(obj.get("pass"), bool):
        return None
    reason = " ".join(str(obj.get("reason") or "").split())[:200]
    return {"pass": obj["pass"],
            "reason": reason or ("the judge says it is right" if obj["pass"]
                                 else "the judge says it is wrong")}


# ---------------------------------------------------------------------------
# the version
# ---------------------------------------------------------------------------

def wording_hash(questions) -> str:
    """eight hex digits over each question's id and text, in id order"""
    h = hashlib.sha256()
    for q in sorted(questions, key=lambda q: str(q.get("id"))):
        h.update(f"{q.get('id')}\t{q.get('prompt', '')}\n".encode("utf-8"))
    return h.hexdigest()[:8]


def version() -> dict:
    """the bank's version: {"date": …, "hash": …}"""
    return {"date": WORDING_DATE, "hash": wording_hash(load_bank())}


# ---------------------------------------------------------------------------
# marking
# ---------------------------------------------------------------------------

def records(model_dir: Path) -> dict[str, dict]:
    """The harness's logged samples, by question id — the bank's, or the
    pilot's for a model that sat only the pilot."""
    out = {}
    recs = _judge._records(model_dir, TASK)
    for legacy in LEGACY_TASKS:
        if not recs:
            recs = _judge._records(model_dir, legacy)
    for rec in recs:
        qid = (rec.get("doc") or {}).get("id")
        if qid:
            out[qid] = rec
    return out


def read(model_dir: Path) -> dict | None:
    try:
        return json.loads((model_dir / OUT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_id(model_dir: Path) -> str:
    try:
        return json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))["model"]
    except (OSError, ValueError, KeyError):
        return model_dir.name.replace("__", "/", 1)


def mark(model_dir: Path, verdicts: dict[str, dict] | None = None,
         judge: dict | None = None) -> dict | None:
    """Mark every question of the bank the model answered, and return what
    everyday.json holds — None when the harness logged no answers.
    `verdicts`: {question id: {pass, reason}} from the judge. A verdict
    already on file is kept while the answer it read is unchanged. A model
    that sat only the pilot is marked on the five it was asked."""
    recs = records(model_dir)
    if not recs:
        return None
    prev = read(model_dir) or {}
    # 12a.4: what was this model asked? Answers to an earlier wording are not
    # re-marked by today's checks — the question under them changed. What was
    # marked when they were answered stays, labelled earlier
    now = version()
    asked = wording_hash([rec.get("doc") or {} for rec in recs.values()])
    stamp = {"version": {"hash": asked, "date": now["date"] if asked == now["hash"]
                         else (prev.get("version") or {}).get("date")},
             "earlier": asked != now["hash"]}
    if stamp["earlier"] and prev.get("items"):
        return {**prev, **stamp, "ran_out": _ran_out(prev["items"])}
    before = {it["id"]: it for it in prev.get("items") or []}
    verdicts = verdicts or {}
    items = []
    for q in load_bank():
        rec = recs.get(q["id"])
        if rec is None:
            continue                  # not asked (a model that sat the pilot only)
        it = {"id": q["id"], "group": q["group"],
              "judged": any(c["type"] == "judge" for c in q["checks"])}
        parts = _judge.answer_parts(rec)
        ans = parts["answer_text"]
        it.update(answer_text=ans, had_reasoning=parts["had_reasoning"])
        if parts["had_reasoning"]:
            it.update(reasoning_text=parts["reasoning_text"],
                      reasoning_words=_judge.words(parts["reasoning_text"]))
        if parts["no_answer"]:
            it.update({"pass": False, "reason": NEVER_FINISHED, "no_answer": True})
        elif not ans.strip():
            it.update({"pass": False, "reason": "the model wrote nothing"})
        else:
            ok, why = grade(q, ans)
            if ok is None:
                # the script checks passed; the judge decides the rest
                v = verdicts.get(q["id"])
                old = before.get(q["id"]) or {}
                if v is None and old.get("pass") is not None and old.get("answer_text") == ans:
                    v = {"pass": old["pass"], "reason": old["reason"]}
                if v is None:
                    it.update({"pass": None, "reason": old.get("reason") if (
                        old.get("answer_text") == ans and old.get("pass") is None
                        and old.get("reason")) else WAITING})
                else:
                    it.update({"pass": bool(v["pass"]), "reason": v["reason"]})
            else:
                it.update({"pass": ok, "reason": why})
        items.append(it)
    gen = _judge._generation(model_dir, TASK) or {}
    out = {
        "model": prev.get("model") or _model_id(model_dir),
        "task": TASK,
        "marked_at": time.time(),
        # what the answers were generated with: the chat template always,
        # greedy, and the budget (512, or a reasoning model's 2,048)
        "settings": {"chat_template": True, "greedy": True, **gen},
        "passed": sum(1 for it in items if it["pass"] is True),
        "total": len(items),
        # n of k per group, k being what this model was asked in it
        "groups": {g: {"passed": sum(1 for it in items if it["group"] == g and it["pass"] is True),
                       "total": sum(1 for it in items if it["group"] == g)}
                   for g in GROUPS if any(it["group"] == g for it in items)},
        "waiting": sum(1 for it in items if it["pass"] is None),
        # 12a.4: answers whose thinking used the whole budget; they fail, and
        # the page says how many beside the score
        "ran_out": _ran_out(items),
        **stamp,
        "items": items,
    }
    if judge or prev.get("judge"):
        out["judge"] = judge or prev["judge"]
    return out


def _ran_out(items: list[dict]) -> int:
    return sum(1 for it in items if it.get("no_answer"))


def write(model_dir: Path, out: dict) -> Path:
    p = model_dir / OUT_NAME
    p.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return p


def summary(out: dict) -> str:
    """The queue row's words."""
    line = f"Everyday tasks: {out['passed']} of {out['total']}"
    if out.get("waiting"):
        line += f" · the judge is marking {out['waiting']}"
    return line


# ---------------------------------------------------------------------------
# the service's side: the task the harness runs, and the judge's one request
# ---------------------------------------------------------------------------

def build_task(dest: Path) -> Path:
    """The bank as a harness task under `dest`: the items, and the yaml with
    their absolute path filled in. Returns the directory for --include_path."""
    dest.mkdir(parents=True, exist_ok=True)
    load_bank()                       # a bad bank fails here, before any GPU
    items = dest / f"{TASK}.jsonl"
    shutil.copyfile(BANK_PATH, items)
    yaml = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__ITEMS_PATH__",
                                                             str(items.resolve()))
    (dest / f"{TASK}.yaml").write_text(f"task: {TASK}\n" + yaml, encoding="utf-8")
    return dest


def _pending(out: dict) -> list[dict]:
    """The questions that wait on the judge, with an answer to send it."""
    qs = {q["id"]: q for q in load_bank()}
    return [qs[it["id"]] for it in out["items"] if it["pass"] is None and it["answer_text"]]


def start(model_dir: Path, submission: int | None = None) -> dict:
    """Mark now; send the judge its one question. Called by the runner
    straight after generation, inside the same run. Returns everyday.json's
    content plus `batch_id` when a judge batch went out."""
    from service import config, db, llm
    out = mark(model_dir)
    if out is None:
        raise RuntimeError("the harness logged no answers for everyday tasks")
    todo = [] if out.get("earlier") else _pending(out)
    if not todo:
        write(model_dir, out)
        return out
    answers = {it["id"]: it["answer_text"] for it in out["items"]}
    ident = _judge.identity()
    if config.JUDGE_MODEL == "stub":
        out = mark(model_dir, {q["id"]: stub_verdict(answers[q["id"]], q["prompt"]) for q in todo},
                   judge={"id": ident["id"], "provisional": False})
        write(model_dir, out)
        return out
    why = _judge.blocked()
    if why:
        for it in out["items"]:
            if it["pass"] is None:
                it["reason"] = "not marked: the judge is not set up on this server"
        write(model_dir, out)
        return out
    try:
        backend = llm.client("judge")
        stamp = llm.provisional(backend, "marked")
        reqs = [llm.Request(custom_id=f"everyday:{submission or 0}:{q['id']}", system="",
                            json=True, max_tokens=200,
                            user=judge_prompt(q, answers[q["id"]]),
                            meta={"kind": "everyday", "id": q["id"]})
                for q in todo]
        bid = backend.submit(reqs)
    except Exception as e:                          # noqa: BLE001 — the answers are marked
        for it in out["items"]:
            if it["pass"] is None:
                it["reason"] = "not marked: the judge could not be reached"
        write(model_dir, out)
        out["error"] = str(e)
        return out
    out["judge"] = {"id": ident["id"], "provisional": bool(stamp), "batch_id": bid}
    write(model_dir, out)
    if submission:
        db.update(submission, judge_batch=bid)
        db.batch_add(bid, "everyday", submission, len(reqs), backend.name, backend.model)
        # the queue reads "grading 0/1" from the moment it goes out
        db.batch_progress(bid, f"0/{len(reqs)} done")
    out["batch_id"] = bid
    return out


def finish(model_dir: Path, results: dict) -> dict | None:
    """The poller's half: the judge's replies in, everyday.json out."""
    verdicts = {}
    for cid, res in results.items():
        if not str(cid).startswith("everyday:"):
            continue
        qid = str(cid).rsplit(":", 1)[-1]
        v = None if getattr(res, "error", None) else parse_verdict(getattr(res, "text", ""))
        verdicts[qid] = v or {"pass": None, "reason": "not marked: the judge's reply could "
                                                      "not be read"}
    prev = read(model_dir) or {}
    out = mark(model_dir, {k: v for k, v in verdicts.items() if v["pass"] is not None},
               judge=prev.get("judge"))
    if out is None:
        return None
    for it in out["items"]:
        v = verdicts.get(it["id"])
        if it["pass"] is None and v is not None:
            it["reason"] = v["reason"]
    out["waiting"] = 0
    write(model_dir, out)
    return out


def judge_failed(model_dir: Path, why: str) -> None:
    """The judge's batch failed: the question says so instead of waiting."""
    out = read(model_dir)
    if not out:
        return
    for it in out["items"]:
        if it["pass"] is None:
            it["reason"] = "not marked: the judge failed — run everyday tasks again"
    out["waiting"] = 0
    out["judge_error"] = why[:300]
    write(model_dir, out)


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
        print(f"{out['model']}: {summary(out)}"
              + (" · earlier wording, kept as marked" if out.get("earlier") else ""))
    print(f"marked {n} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
