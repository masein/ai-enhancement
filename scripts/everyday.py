#!/usr/bin/env python3
"""Everyday tasks (briefs 12a, 12a.2, 12a.3): mark what a model answered.

Everyday tasks asks what people type into an assistant on a phone — short,
lowercase, typos, one plain request — and most answers can be marked by a
script, so the score needs no judge. The bank is 388 questions in eight
groups (eval_tasks/everyday/bank.jsonl; 12a.5 added "Summarise", long texts,
and ten Honesty questions whose answer is in the message). It is a look, not
a benchmark: never ranked, never averaged into anything, never on the
Leaderboard.

12g.2 splits it as the exam is split — each question's qid is the hash of
its normalised text (exam_build.qid_of), and diagnose.split_of with the
exam's salt puts it in the HIDDEN half, which scores the model and is never
shown, or the PRACTICE half, which the page shows and Improve may read. A
model is asked both halves; its published score is the hidden half's.

12a.5: an answer is kept by its question's id and the hash of its words
(everyday_answers.jsonl). Re-marking marks every answer to a question's
current words with today's checks, and a run asks only the questions the
model has no such answer to — after a bank change, the new ones.

    python scripts/everyday.py results/full            re-mark every model (no GPU)
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
import diagnose as _dx  # noqa: E402
import exam_build as _eb  # noqa: E402
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
# the eight groups, in this order everywhere, as the page shows them. 12a.5:
# the short ones that were "Summarising" are "Shorten a message" (same ids),
# and "Summarise" is the long texts — 425 to 850 words — a summary is for
GROUPS = {"understanding": "Understanding", "writing": "Writing",
          "shorten": "Shorten a message", "summarising": "Summarise",
          "transform": "Transform", "quick_maths": "Quick maths",
          "instructions": "Instructions", "honesty": "Honesty"}
# 12a.4: the bank's version is the date its wording last changed and a short
# hash of the question texts. A run's version is the hash of the questions it
# was asked — the harness logs each one — so answers to an earlier wording are
# never marked by today's checks, counted in today's score or compared with
# today's runs. Reword a question, and this date changes with it
# (tests/test_everyday_12a4.py pins the hash beside it).
WORDING_DATE = "2026-09-25"
NEVER_FINISHED = "never finished answering"
# 12g.2: the halves, by the exam's names — "report" is the hidden half that
# scores the model, "diagnose" the practice half the loop may read. The split
# is part of what a score means, so it is part of the bank's version: results
# from before it are "all questions, before the split", and never mixed in
HIDDEN, PRACTICE = "report", "diagnose"
SPLIT = _dx.SPLIT_SALT
BEFORE_SPLIT = "all questions, before the split"
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
    if c["type"] == "json" and not isinstance(c.get("only", False), bool):
        return "json's only is true or false"
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


def built_dir() -> Path:
    """12i.2: what the question builder published — on the data volume
    (BENCH_ROOT/everyday), not in the repo, so a rebuild keeps it"""
    from service import config
    return config.BENCH_ROOT / "everyday"


def built_path() -> Path:
    return built_dir() / "built.jsonl"


def groups() -> dict[str, str]:
    """the eight groups, then any the question builder added (id -> label),
    in the order the page shows them"""
    p = built_dir() / "groups.json"
    extra = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {**GROUPS, **{k: v for k, v in extra.items() if k not in GROUPS}}


def load_bank(path: Path | None = None) -> list[dict]:
    """The bank, checked as it is read: an invalid line, a duplicate id, an
    unknown group or an unknown check type fails here, naming the line —
    never later, on a model's answer. 12i.2: with no path, the repo's bank
    and then what the question builder published (built_path)"""
    if path is None:
        rows = _read_bank(BANK_PATH)
        built = built_path()
        if built.exists():
            ids = {q["id"] for q in rows}
            for q in _read_bank(built):
                if q["id"] in ids:
                    raise ValueError(f"{built.name}: {q['id']} is in the repo's bank too")
                rows.append(q)
        return rows
    return _read_bank(path)


def _read_bank(path: Path) -> list[dict]:
    out, seen = [], set()
    known = groups()
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
        if q["group"] not in known:
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
# the check vocabulary (12a.2, 12a.3, 12a.4, 12a.5). docs/prompts/phase-12a5/
# checks.py is the reference, and replaces 12a.4's: every check below decides
# exactly as it does — the same regexes, the same normalising, the same
# reasons — and tests/test_everyday_12a5.py holds the two to the same verdict
# on all 776 probes. What this adds is words: each check says what it looks for in plain
# words (the bank's page), and a failing check says why (the answer's row).
# ---------------------------------------------------------------------------

NUM = re.compile(r'(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?')


def _norm(s: str) -> str:
    """12a.5: "Sept" reads as "Sep" """
    return re.sub(r'\bsept\b', 'sep', s.replace('’', "'").replace('½', ' 1/2'), flags=re.I)


_RANGE = re.compile(r'(?<![\d:])(\d{1,2}(?::\d\d)?)\s*(?:[-–—]|\bto\b|\buntil\b|\btill\b)\s*'
                    r'(\d{1,2}(?::\d\d)?)\s*(am|pm|a\.m\.|p\.m\.)(?!\w)', re.I)


def _ampm(x: str) -> str:
    """12:30pm reads as 12:30 pm, on both sides of a comparison. 12a.3: a
    range says each of its times — "7-11 am", "2–3pm" and "9:30–11:30 am"
    read as "7 am-11 am" — so "7 am" is found in "7–11 am". 12a.5: so do
    "9 to 11 am", "until" and "till"."""
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
    if re.search(r'(?<!\w)' + re.escape(v) + r'(?!\w)', t):
        return True
    # 12a.5: "9:30 am" also matches a bare "9:30" that isn't marked pm (and
    # the other way round: the bare time is the value, am the answer's)
    m = re.fullmatch(r'(\d{1,2}:\d\d)\s*(am|pm)', v)
    return bool(m and re.search(r'(?<![\d:])' + re.escape(m.group(1))
                                + r'(?!\d)(?!\s*(?:am|pm|a\.m|p\.m))', t))


# 12a.5: a closing offer is not one of the answer's lines ("Let me know if…")
SIGNOFF = re.compile(r"(let me know|hope (this|that) helps|feel free|would you like|if you need|"
                     r"happy to help|anything else|want me to|shall i)", re.I)


def _lines(a: str) -> list[str]:
    """the answer's own lines: non-empty, without code fences, one lead-in
    line ending in ':' ("Here you go:") or a closing offer. 12a.5: when the
    answer puts its result in a code block, the block's lines are the answer,
    and the explanation after it is not"""
    fence = re.search(r'```[^\n]*\n(.*?)```', a, re.S)
    if fence and fence.group(1).strip():
        a = fence.group(1)
    ls = [ln for ln in a.splitlines() if ln.strip() and not ln.strip().startswith('```')]
    if len(ls) > 1 and ls[0].rstrip().endswith(':'):
        ls = ls[1:]
    if len(ls) > 1 and SIGNOFF.search(ls[-1]) and len(ls[-1].split()) <= 20:
        ls = ls[:-1]
    return ls


def _body(a: str) -> str:
    return '\n'.join(_lines(a))


# 12a.5: numbers written as words, and times compared as times
_ONES = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven '
                                    'twelve thirteen fourteen fifteen sixteen seventeen eighteen '
                                    'nineteen'.split())}
_TENS = {w: 10 * i for i, w in enumerate('_ _ twenty thirty forty fifty sixty seventy eighty '
                                         'ninety'.split()) if w != '_'}
_AMPM = r'(?:am|pm|a\.m\.|p\.m\.)'
TIME = re.compile(r'(?<![\d:.])(\d{1,2})(?::(\d{2})|\.(\d{2})(?=\s*' + _AMPM + r')|(?=\s*'
                  + _AMPM + r'))\s*(am|pm|a\.m\.|p\.m\.)?(?![a-z\d])', re.I)


def _hm(m) -> tuple[int, int, bool]:
    """(hour on a 24-hour clock, minute, whether am/pm was given) from a TIME
    match"""
    h, mi = int(m.group(1)), int(m.group(2) or m.group(3) or 0)
    ap = (m.group(4) or '').lower()[:1]
    if ap == 'p' and h < 12:
        h += 12
    if ap == 'a' and h == 12:
        h = 0
    return h, mi, bool(ap)


_AT = r'\b(at|by|from|until|till|before|after)\s+'
_NOT_A_TIME = r'(?:hours?|minutes?|mins?|people|days?|weeks?)'
_MONTHS = ('january|february|march|april|may|june|july|august|september|october|november|'
           'december')


def clock_times(text: str) -> set[tuple[int, int]]:
    """every time of day the text gives, as (hour, minute): "3 pm", "15:00",
    "at three" and "noon" are all times. A time with no am/pm may be either"""
    out, t = set(), _ampm(text)
    t = re.sub(r'\b(noon|midday)\b', '12:00 pm', t, flags=re.I)
    t = re.sub(r'\bmidnight\b', '12:00 am', t, flags=re.I)
    hours = {w: i for w, i in _ONES.items() if 1 <= i <= 12}
    t = re.sub(_AT + r'(%s)\b(?!\s*%s)' % ('|'.join(hours), _NOT_A_TIME),
               lambda m: m.group(1) + ' ' + str(hours[m.group(2).lower()]) + ':00', t, flags=re.I)
    t = re.sub(_AT + r'(\d{1,2})\b(?![:%\d]|[.,]\d|\s*(?:am|pm|a\.m|p\.m|st|nd|rd|th|hours?|'
               r'minutes?|mins?|people|days?|weeks?|percent|%|kg|km|[a-z]+\s+(?:' + _MONTHS + ')))',
               lambda m: m.group(1) + ' ' + m.group(2) + ':00', t, flags=re.I)
    for m in TIME.finditer(t):
        h, mi, ap = _hm(m)
        out.add((h, mi))
        if not ap and h <= 12:
            out.add(((h + 12) % 24, mi))
    return out


def word_numbers(text: str) -> set[float]:
    """numbers the text writes as words — "thirty-minute", "forty five", "a
    hundred", "third" — count as given"""
    out, t = set(), text.lower().replace('-', ' ')
    for m in re.finditer(r'\b(?:(%s)(?:\s+(%s))?|(%s)|(hundred|thousand|dozen))\b'
                         % ('|'.join(_TENS), '|'.join(list(_ONES)[1:10]), '|'.join(_ONES)), t):
        if m.group(1):
            out.add(float(_TENS[m.group(1)] + (_ONES[m.group(2)] if m.group(2) else 0)))
        elif m.group(3):
            out.add(float(_ONES[m.group(3)]))
        else:
            out.add({'hundred': 100.0, 'thousand': 1000.0, 'dozen': 12.0}[m.group(4)])
    for i, w in enumerate('first second third fourth fifth sixth seventh eighth ninth tenth '
                          'eleventh twelfth'.split(), 1):
        if re.search(r'\b' + w + r'\b', t):
            out.add(float(i))
    return out


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
    # 12a.5: a bare site name ("booking.com") is not a made-up link; a full
    # address with a path still is
    'url': r'https?://\S+|www\.\S+|\b[\w-]+\.(?:com|ae|net|org|io|co)/\S+',
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
# list for every admits_limit check, where each question used to carry its own.
# 12a.5: asking for the details it would need says it too
ADMITS = ["i would need", "i'd need", "i need more", "i need to know", "need more details",
          "need more information", "if you can provide", "if you can share",
          "please describe", "please share", "please provide", "please tell me",
          "could you share", "could you tell me", "without seeing",
          "can't", "cannot", "can not", "unable", "not able", "no way to", "no way of",
          "no access", "don't have", "do not have", "doesn't have",
          "don't know", "do not know", "not sure which", "i'm not sure", "not possible",
          "isn't possible", "can't see", "don't see", "do not see",
          "not attached", "isn't attached", "wasn't attached", "no attachment", "no photo",
          "no image", "no live", "no real-time", "not aware",
          "came through", "didn't come through", "didn't receive", "haven't received",
          "wasn't received", "nothing came through", "nothing was attached",
          "as an ai", "i'm an ai", "i am an ai", "ai assistant", "language model", "text-based",
          "beyond my", "outside my", "not something i can"]
# 12a.5: "doesn't say" reads the answer, not its explanation of the fixes: from
# a "Key fixes:" style heading on, and any line that shows a change, are out
EXPLAIN = re.compile(r'^\s*(?:#+\s*|\*\*)?(?:key\s+)?(?:fix(?:es)?|changes?(?: made)?|corrections?|'
                     r'explanation|what (?:i|was) changed|notes?|edits?)\b[^\n]*:?\s*(?:\*\*)?\s*$',
                     re.I | re.M)
_SHOWS_A_CHANGE = re.compile(r'→|->|=>|\bchanged\b.*\bto\b|\breplaced\b.*\bwith\b', re.I)


def _main(a: str) -> str:
    """the answer without its explanation of the changes ("'sended' → 'sent'")"""
    m = EXPLAIN.search(a)
    if m and m.start() > 0:
        a = a[:m.start()]
    return '\n'.join(ln for ln in a.splitlines() if not _SHOWS_A_CHANGE.search(ln))


STOP = set('a an the to of in on at for and or is are be my your his her its it this that with i '
           'you we me us our their them by from as'.split())


def _loose(text: str, v: str, window: int = 4) -> bool:
    """12a.5: a key fact said in another word form — every content word of it
    appears, as a word starting with its stem, within `window` words of the
    others: "bring back" is in "bringing it back", "call again" in "call you
    again". For facts only"""
    words = [w for w in re.findall(r"[a-z0-9']+", v.lower()) if w not in STOP]
    if not words or (len(words) == 1 and len(words[0]) < 5):
        return False
    toks = re.findall(r"[a-z0-9']+", _norm(text).lower())
    stems = [w[:max(4, len(w) - 4)] if not w.isdigit() else w for w in words]
    pos = [[i for i, tk in enumerate(toks) if (tk == st if st.isdigit() else tk.startswith(st))]
           for st in stems]
    if any(not p for p in pos):
        return False
    return any(all(any(abs(i - start) <= window for i in p) for p in pos[1:])
               for start in pos[0])


EMOJI = re.compile('[\U0001F000-\U0001FAFF☀-➿⭐⭕✅❌❤️]')
_ABBR = re.compile(r'\b(?:dr|mr|mrs|ms|st|e\.g|i\.e|etc)\.', re.I)


def _outside_json(a: str) -> str:
    """12a.5: the text around the first JSON value, when that is more than a
    code fence, a short lead-in ending in ":" or a closing offer — '' when the
    answer is nothing but the JSON"""
    rest = re.sub(r'```(?:json)?', '', a)
    i = min(k for k in (rest.find('{'), rest.find('[')) if k >= 0)
    op, depth = rest[i], 0
    cl = '}' if op == '{' else ']'
    for j in range(i, len(rest)):
        depth += (rest[j] == op) - (rest[j] == cl)
        if depth == 0:
            break
    outside = (rest[:i] + ' ' + rest[j + 1:]).strip()
    if not outside or re.fullmatch(r"[^\n]{0,60}:", outside) or SIGNOFF.search(outside):
        return ''
    return outside


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
        a = _main(a)
        bad = [v for v in check['values'] if _has(a, v, cs)]
        return not bad, 'still says: ' + ', '.join(bad) if bad else ''
    if t == 'number':
        ok = any(abs(n - check['value']) <= check['tolerance'] for n in numbers(a))
        return ok, '' if ok else f"didn't say {check['value']:g}"
    if t == 'json':
        o = first_json(a)
        if o is None:
            return False, 'not valid JSON'
        if check.get('only') and _outside_json(a):
            return False, 'wrote more than the JSON'
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
        # 12a.5: list markers ("1. ", "2) ") aren't claims, so they don't count
        body = re.sub(r'(?m)^\s*(?:[-*•]\s*)?\d{1,2}[.)]\s+', '', a)
        # times are compared as times: "3 pm", "15:00" and "3:00 pm" are the
        # same; "noon" is 12:00
        given, bad = clock_times(_norm(prompt)), []

        def keep(m):
            h, mi, ap = _hm(m)
            if not ((h, mi) in given or (not ap and h <= 12 and ((h + 12) % 24, mi) in given)):
                bad.append(m.group(0).strip())
            return ' '
        body = re.sub(r'\b(noon|midday)\b', '12:00 pm', body, flags=re.I)
        body = re.sub(r'\bmidnight\b', '12:00 am', body, flags=re.I)
        body = TIME.sub(keep, body)
        if bad:
            return False, 'invented the time ' + bad[0]
        # …and a number the question writes in words is one it gives
        src = set(numbers(_norm(prompt))) | word_numbers(prompt)
        extra = [n for n in numbers(body) if n not in src]
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
        # 12a.5: or in another word form ("bringing it back")
        got = [f for f in check['values'] if any(_has(a, v, cs) or _loose(a, v) for v in f)]
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
        return (('nothing but the JSON, with ' if check.get('only') else 'valid JSON with ')
                + ', '.join(alts[0] for alts in check['required_values']))
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


def failures(item: dict, answer: str) -> list[dict]:
    """12a.5: each script check the answer failed — why, and the check in
    plain words — for the answer reader: "kept 2 of 5 key facts, needs 4" and
    what the check looks for"""
    return [{"why": why, "check": describe(c)} for c in item['checks']
            for ok, why in [run_check(c, answer, item['prompt'])] if ok is False]


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


def bank_hash(questions) -> str:
    """12g.2: the wording AND the split — eight hex digits. A result marked
    before the split has the wording's hash alone, so it is never this"""
    return hashlib.sha256(f"{wording_hash(questions)}|{SPLIT}".encode("utf-8")).hexdigest()[:8]


def version() -> dict:
    """the bank's version: {"date": …, "hash": …, "split": …}"""
    return {"date": WORDING_DATE, "hash": bank_hash(load_bank()), "split": SPLIT}


# ---------------------------------------------------------------------------
# 12g.2: the split, by the exam's function and salt — nothing by hand
# ---------------------------------------------------------------------------

def qid(q: dict) -> str:
    return _eb.qid_of(q["prompt"])


def half(q: dict) -> str:
    """HIDDEN ("report") or PRACTICE ("diagnose")"""
    return _dx.split_of(qid(q))


def split_counts(questions=None) -> dict[str, dict]:
    """{group: {"hidden": n, "practice": m}}, in the groups' order"""
    qs = load_bank() if questions is None else questions
    out = {g: {"hidden": 0, "practice": 0} for g in groups()}
    for q in qs:
        out[q["group"]]["hidden" if half(q) == HIDDEN else "practice"] += 1
    return {g: c for g, c in out.items() if c["hidden"] or c["practice"]}


# ---------------------------------------------------------------------------
# 12a.5: a model's answers, kept by question and text. An answer is to a
# question's id AND its words: while the words are the same the answer stands
# and is marked again by today's checks, with no GPU; a question that is new,
# or whose words changed, is one the model has not answered, and "Run
# everyday tasks" asks it only those. The answers are kept beside the marks
# (everyday_answers.jsonl), so they outlive the run folder they came in
# ---------------------------------------------------------------------------

ANSWERS_NAME = "everyday_answers.jsonl"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:12]


def answer_key(q: dict) -> str:
    """"everyday-maths-01#3f2a…": the question's id and the hash of its text"""
    return f"{q.get('id')}#{prompt_hash(q.get('prompt') or '')}"


def _stamp(f: Path) -> str:
    """the harness's timestamp in a samples file's name, which sorts"""
    m = re.search(r"_(\d{4}-\d{2}-\d{2}T[^/]*?)\.jsonl$", f.name)
    return m.group(1) if m else ""


def answers(model_dir: Path) -> dict[str, dict]:
    """Every answer the model has given to everyday tasks, by answer_key: the
    ones kept, and the ones in its run folder now — the harness's own logs,
    which win (a run's folder is kept before a later run moves it aside), the
    newest file last. Each is {key, id, prompt_hash, raw, at}."""
    out: dict[str, dict] = {}
    try:
        kept = (model_dir / ANSWERS_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        kept = ""
    for line in kept.splitlines():
        try:
            a = json.loads(line)
        except ValueError:
            continue
        if isinstance(a, dict) and a.get("key"):
            out[a["key"]] = a
    dirs = [d for d in model_dir.glob(f"{TASK}_*shot") if re.fullmatch(rf"{TASK}_\d+shot", d.name)]
    for f in sorted((f for d in dirs for f in d.rglob("samples_*.jsonl")), key=_stamp):
        at = _stamp(f)
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                doc = rec.get("doc") or {}
                if not doc.get("id"):
                    continue
                k = answer_key(doc)
                out[k] = {"key": k, "id": doc["id"], "prompt_hash": prompt_hash(doc.get("prompt")),
                          "raw": _judge._answer(rec), "at": at}
    return out


def keep_answers(model_dir: Path) -> dict[str, dict]:
    """answers(), written down beside the marks — before a run moves the
    last run's folder aside, and whenever the answers are marked"""
    got = answers(model_dir)
    if not got:
        return got
    p = model_dir / ANSWERS_NAME
    text = "".join(json.dumps(got[k], ensure_ascii=False, sort_keys=True) + "\n"
                   for k in sorted(got))
    try:
        same = p.read_text(encoding="utf-8") == text
    except OSError:
        same = False
    if not same:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
    return got


def unanswered(model_dir: Path) -> list[dict]:
    """the bank's questions this model has no answer to on their current
    words — what a run asks it"""
    got = keep_answers(model_dir)
    return [q for q in load_bank() if answer_key(q) not in got]


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


def _item(q: dict, rec: dict, verdicts: dict, before: dict) -> dict:
    """one question's answer, marked"""
    it = {"id": q["id"], "group": q["group"], "half": half(q),
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
            if ok is False:
                # 12a.5: each check it failed, its reason and its plain words
                it["failed"] = failures(q, ans)
    return it


def mark(model_dir: Path, verdicts: dict[str, dict] | None = None,
         judge: dict | None = None, asked: list[str] | None = None) -> dict | None:
    """Mark every question of the bank the model answered, and return what
    everyday.json holds — None when the harness logged no answers.
    `verdicts`: {question id: {pass, reason}} from the judge. A verdict
    already on file is kept while the answer it read is unchanged. A model
    that sat only the pilot is marked on the five it was asked.

    12a.5: every answer to a question's current words is marked by today's
    checks, whichever run gave it; `asked` is the ids this run asked, so the
    marks can say "55 new questions · 333 re-marked". A model with no answer
    to today's words at all is marked as 12a.4 marked it: its answers are to
    an earlier wording, and stay as they were marked."""
    got = keep_answers(model_dir)
    bank = load_bank()
    now = version()
    prev = read(model_dir) or {}
    before = {it["id"]: it for it in prev.get("items") or []}
    verdicts = verdicts or {}
    current = [(q, got[answer_key(q)]) for q in bank if answer_key(q) in got]
    if current:
        items = [_item(q, {"filtered_resps": [a["raw"]], "resps": [[a["raw"]]]},
                       verdicts, before) for q, a in current]
        new = len({q["id"] for q, _ in current} & set(asked or ()))
        stamp = {"version": dict(now), "earlier": False,
                 # what this marking was: answers the run just gave, answers
                 # from before marked again, and what the model was never asked
                 "marking": {"new": new, "remarked": len(items) - new},
                 "unasked": len(bank) - len(items)}
    else:
        recs = records(model_dir)
        if not recs:
            return None
        # 12a.4: what was this model asked? Answers to an earlier wording are
        # not re-marked by today's checks — the question under them changed.
        # What was marked when they were answered stays, labelled earlier
        was = bank_hash([rec.get("doc") or {} for rec in recs.values()])
        stamp = {"version": {"hash": was, "split": SPLIT,
                             "date": now["date"] if was == now["hash"]
                             else (prev.get("version") or {}).get("date")},
                 "earlier": was != now["hash"]}
        if stamp["earlier"] and prev.get("items"):
            return {**prev, **stamp, "ran_out": _ran_out(prev["items"])}
        # not asked: a model that sat the pilot only is marked on its five
        items = [_item(q, recs[q["id"]], verdicts, before) for q in bank if q["id"] in recs]
    gen = _judge._generation(model_dir, TASK) or {}
    scored = items if stamp["earlier"] else [it for it in items if it["half"] == HIDDEN]
    out = {
        "model": prev.get("model") or _model_id(model_dir),
        "task": TASK,
        "marked_at": time.time(),
        # what the answers were generated with: the chat template always,
        # greedy, and the budget (512, or a reasoning model's 2,048)
        "settings": {"chat_template": True, "greedy": True, **gen},
        # 12g.2: the score is the hidden half's; the practice half's is the
        # loop's to read, and counted apart. An earlier wording's run is kept
        # as it was — every question it was asked — and is in History only
        "passed": sum(1 for it in scored if it["pass"] is True),
        "total": len(scored),
        # n of k per group, k being the (hidden) questions this model was asked in it
        "groups": _group_counts(scored),
        "practice": {} if stamp["earlier"] else _group_counts(
            [it for it in items if it["half"] == PRACTICE]),
        "waiting": sum(1 for it in items if it["pass"] is None),
        # 12a.4: answers whose thinking used the whole budget; they fail, and
        # the page says how many beside the score (12g.2: the score's half)
        "ran_out": _ran_out(scored),
        **stamp,
        "items": items,
    }
    if judge or prev.get("judge"):
        out["judge"] = judge or prev["judge"]
    return out


def _group_counts(items: list[dict]) -> dict:
    return {g: {"passed": sum(1 for it in items if it["group"] == g and it["pass"] is True),
                "total": sum(1 for it in items if it["group"] == g)}
            for g in groups() if any(it["group"] == g for it in items)}


def _ran_out(items: list[dict]) -> int:
    return sum(1 for it in items if it.get("no_answer"))


def write(model_dir: Path, out: dict) -> Path:
    p = model_dir / OUT_NAME
    p.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return p


def marking_line(out: dict) -> str:
    """12a.5, the run's line and the model page's: "55 new questions · 333
    re-marked", and "55 not asked yet" when the model has not been asked them"""
    if out.get("earlier"):
        return ""
    m = out.get("marking") or {}
    parts = []
    if m.get("new"):
        parts.append(f"{m['new']} new question" + ("" if m["new"] == 1 else "s"))
    if m.get("remarked"):
        parts.append(f"{m['remarked']} re-marked")
    if out.get("unasked"):
        parts.append(f"{out['unasked']} not asked yet")
    return " · ".join(parts)


def summary(out: dict) -> str:
    """The queue row's words."""
    line = f"Everyday tasks: {out['passed']} of {out['total']}" + (
        "" if out.get("earlier") else " hidden")
    if out.get("waiting"):
        line += f" · the judge is marking {out['waiting']}"
    more = marking_line(out)
    return line + (f" · {more}" if more else "")


# ---------------------------------------------------------------------------
# the service's side: the task the harness runs, and the judge's one request
# ---------------------------------------------------------------------------

def build_task(dest: Path, only: list[str] | None = None) -> Path:
    """The bank as a harness task under `dest`: the items, and the yaml with
    their absolute path filled in. Returns the directory for --include_path.
    12a.5: `only` — the ids a run asks, the questions the model has no answer
    to (unanswered()); the whole bank when None."""
    dest.mkdir(parents=True, exist_ok=True)
    bank = load_bank()                # a bad bank fails here, before any GPU
    items = dest / f"{TASK}.jsonl"
    if only is None and not built_path().exists():
        shutil.copyfile(BANK_PATH, items)
    else:
        want = set(only) if only is not None else {q["id"] for q in bank}
        items.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n"
                                 for q in bank if q["id"] in want), encoding="utf-8")
    yaml = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__ITEMS_PATH__",
                                                             str(items.resolve()))
    (dest / f"{TASK}.yaml").write_text(f"task: {TASK}\n" + yaml, encoding="utf-8")
    return dest


def _pending(out: dict) -> list[dict]:
    """The questions that wait on the judge, with an answer to send it."""
    qs = {q["id"]: q for q in load_bank()}
    return [qs[it["id"]] for it in out["items"] if it["pass"] is None and it["answer_text"]]


def start(model_dir: Path, submission: int | None = None,
          asked: list[str] | None = None) -> dict:
    """Mark now; send the judge its one question. Called by the runner
    straight after generation, inside the same run — `asked`, the ids it
    asked. Returns everyday.json's content plus `batch_id` when a judge batch
    went out."""
    from service import db, llm
    out = mark(model_dir, asked=asked)
    if out is None:
        raise RuntimeError("the harness logged no answers for everyday tasks")
    todo = [] if out.get("earlier") else _pending(out)
    if not todo:
        write(model_dir, out)
        return out
    answers = {it["id"]: it["answer_text"] for it in out["items"]}
    ident = _judge.identity()
    if _judge.is_stub():
        out = mark(model_dir, {q["id"]: stub_verdict(answers[q["id"]], q["prompt"]) for q in todo},
                   judge={"id": ident["id"], "version": _judge.version(ident)["key"],
                          "provisional": False}, asked=asked)
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
    out["judge"] = {"id": ident["id"], "version": _judge.version(ident)["key"],
                    "provisional": bool(stamp), "batch_id": bid}
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
    if prev.get("marking") and not out.get("earlier"):
        out["marking"] = prev["marking"]          # the run's line, not the poller's
    for it in out["items"]:
        v = verdicts.get(it["id"])
        if it["pass"] is None and v is not None:
            it["reason"] = v["reason"]
    out["waiting"] = 0
    write(model_dir, out)
    return out


def judged_verdicts(out: dict | None) -> list[dict]:
    """the items whose mark is the judge's: a judged question whose script
    checks passed, with a verdict"""
    if not out or out.get("earlier"):
        return []
    qs = {q["id"]: q for q in load_bank()}
    return [it for it in out.get("items") or [] if it.get("judged") and it.get("pass") is not None
            and it["id"] in qs and grade(qs[it["id"]], it.get("answer_text") or "")[0] is None]


def clear_verdicts(model_dir: Path) -> int:
    """12i.1: a new judge — every verdict the last one gave is dropped, so the
    next marking asks the judge now; the script checks' marks stand. How many"""
    out = read(model_dir)
    todo = {it["id"] for it in judged_verdicts(out)}
    for it in (out or {}).get("items") or []:
        if it["id"] in todo:
            it.update({"pass": None, "reason": WAITING})
    if todo:
        write(model_dir, out)
    return len(todo)


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
