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
    python scripts/everyday.py results/full --judge    12a.6: and send the judge what waits on it
    python scripts/everyday.py results/full --compare  12a.6: before and after, model by model;
                                                       12a.9: and what moved Summarise, change
                                                       by change
    python scripts/everyday.py results/full --judge -q everyday-summarising-07
                                                       12a.8: the judge for one question's answers

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
# the groups, in this order everywhere, as the page shows them. 12a.5: the
# short ones that were "Summarising" were "Shorten a message" (same ids), and
# "Summarise" the long texts — 425 to 850 words — a summary is for. 12a.6:
# one group again, "Summarise": the best fifteen short ones kept, the rest
# retired (retired.jsonl), and every one marked by the judge on a rubric
GROUPS = {"understanding": "Understanding", "writing": "Writing",
          "summarising": "Summarise",
          "transform": "Transform", "quick_maths": "Quick maths",
          "instructions": "Instructions", "honesty": "Honesty"}
# 12a.6: a question written for the old group, anywhere, reads as Summarise
MERGED = {"shorten": "summarising"}
RETIRED_PATH = BANK_DIR / "retired.jsonl"
# 12a.4: the bank's version is the date its wording last changed and a short
# hash of the question texts. A run's version is the hash of the questions it
# was asked — the harness logs each one — so answers to an earlier wording are
# never marked by today's checks, counted in today's score or compared with
# today's runs. Reword a question, and this date changes with it
# (tests/test_everyday_12a4.py pins the hash beside it).
WORDING_DATE = "2026-09-27"
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
    if c["type"] == "judge" and "scale" in c:
        sc, at = c.get("scale"), c.get("pass_at")
        if not (isinstance(sc, int) and isinstance(at, int) and 1 <= at <= sc <= 10):
            return "a judge that scores needs scale and pass_at: whole numbers, pass_at at most scale"
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


# ---------------------------------------------------------------------------
# 12p.1: the hidden half lives on the data volume (BENCH_ROOT/everyday/
# hidden.jsonl), beside what the builder published and the edits — never in
# the repo, which is mirrored in public. What it should be is committed as a
# count and a digest (eval_tasks/everyday/hidden_manifest.json); when the set
# is missing or doesn't match, Everyday runs and scoring stop until it is
# restored, and every page says so. The rest of the board works
# ---------------------------------------------------------------------------

HIDDEN_NAME = "hidden.jsonl"
RESTORE = "sudo docker compose exec -T bench python -m service.hidden_store restore"


class HiddenMissing(RuntimeError):
    """Everyday's hidden set is missing or changed: nothing is scored until
    it is restored"""


def hidden_path() -> Path:
    return built_dir() / HIDDEN_NAME


def _raw_rows(path: Path) -> list[dict]:
    """a bank file's rows as written — no check, no upgrade — for its digest"""
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def hidden_digest(rows: list[dict]) -> str:
    """the hidden set's digest: its rows as written, in id order"""
    lines = sorted(json.dumps(q, ensure_ascii=False, sort_keys=True) for q in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def repo_hidden() -> list[dict]:
    """the hidden half as the repo's bank holds it, each row marked hidden —
    what `hidden_store move` writes to the store"""
    return [{**q, "half": HIDDEN} for q in _raw_rows(BANK_PATH) if half(q) == HIDDEN]


def hidden_manifest() -> dict | None:
    from service import config
    try:
        return json.loads(Path(config.HIDDEN_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def manifest_of(rows: list[dict]) -> dict:
    """what is committed of a hidden set: how many, per group, and its digest"""
    counts: dict[str, int] = {}
    for q in rows:
        counts[q["group"]] = counts.get(q["group"], 0) + 1
    return {"count": len(rows), "groups": counts, "digest": hidden_digest(rows)}


def switched_digests() -> set[str]:
    """12p.3: the hidden sets the owner switched to (service/rotation.py) —
    accepted as the committed one is, until the manifest is committed again"""
    try:
        return {r["new_digest"] for r in json.loads(
            (built_dir() / "switches.json").read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def hidden_status() -> dict:
    """{ok, state, count, why}: "store" (on the data volume, as committed),
    "repo" (still in the repo's bank, not moved yet), "unchecked" (nothing
    committed to check it against), "changed" or "missing" — the last two
    stop Everyday runs and scoring"""
    m, p = hidden_manifest(), hidden_path()
    if p.exists():
        try:
            rows = _raw_rows(p)
        except (OSError, ValueError):
            rows = None
        if rows is not None and (not m or hidden_digest(rows) in (
                {m.get("digest")} | switched_digests())):
            return {"ok": True, "state": "store" if m else "unchecked", "count": len(rows),
                    "why": ""}
        return {"ok": False, "state": "changed", "count": len(rows or []),
                "why": "Everyday's hidden set is missing or changed: restore it · " + RESTORE}
    rows = repo_hidden()
    if not m:
        return {"ok": True, "state": "unchecked", "count": len(rows), "why": ""}
    if rows and hidden_digest(rows) == m.get("digest"):
        return {"ok": True, "state": "repo", "count": len(rows), "why": ""}
    return {"ok": False, "state": "missing", "count": 0,
            "why": "Everyday's hidden set is missing or changed: restore it · " + RESTORE}


def need_hidden() -> None:
    """HiddenMissing, with the banner's words, unless the hidden set is here"""
    st = hidden_status()
    if not st["ok"]:
        raise HiddenMissing(st["why"])


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
        # 12n.1: and as the questions read after every edit made where they are read
        return _edited(_base_bank())
    return _read_bank(path)


def _base_bank() -> list[dict]:
    """the repo's bank and what the question builder published, before any edit.
    12p.1: and the hidden half from the data volume — in the repo's place where
    the repo still holds it, else after its group's questions"""
    rows = _read_bank(BANK_PATH)
    hid = hidden_path()
    if hid.exists():
        store = {q["id"]: q for q in _read_bank(hid)}
        rows = [store.pop(q["id"], q) for q in rows]
        if store:
            order = list(groups())
            rows = sorted(rows + list(store.values()),
                          key=lambda q: order.index(q["group"]) if q["group"] in order
                          else len(order))
    built = built_path()
    if built.exists():
        ids = {q["id"] for q in rows}
        for q in _read_bank(built):
            if q["id"] in ids:
                raise ValueError(f"{built.name}: {q['id']} is in the repo's bank too")
            rows.append(q)
    return rows


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
        out.append(_valid(q, f"{path.name} line {n}", known))
    return out


def _valid(q: dict, where: str, known: dict | None = None) -> dict:
    """one question, checked as the bank is read — or ValueError, naming where"""
    known = groups() if known is None else known
    q["group"] = MERGED.get(q["group"], q["group"])
    if q["group"] == "summarising" and (not _rubric_check(q) or _upgraded(q)):
        q["checks"] = summarise_checks(q)          # 12a.6: marked by the judge's rubric
    if q["group"] not in known:
        raise ValueError(f"{where}: unknown group {q['group']!r}")
    if not isinstance(q["checks"], list):
        raise ValueError(f"{where}: checks is not a list")
    for c in q["checks"]:
        why = _bad_check(c)
        if why:
            raise ValueError(f"{where}: {why}")
    return q


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


# 12a.6: a time written as four digits — "1100", "0930", "1430 hrs". In the
# text it gives that time; in an answer it is the text's time when the text
# gives it (as "11:00", "11 am" or "1100"), and a number like any other when not
HHMM = re.compile(r'(?<![\d:.,$\u20ac\u00a3])([01]\d|2[0-3])([0-5]\d)(?:\s*(?:hrs?|h)\b)?'
                  r'(?!\d|[:.,]\d|\s*%)', re.I)


# 12a.7: a bare hour after these is a time too — "back around 2", "it's now 8"
_AROUND = r"\b(around|about|approx(?:imately)?|roughly|circa|now|since|it's|its)\s+"


def clock_times(text: str, loose: bool = False) -> set[tuple[int, int]]:
    """every time of day the text gives, as (hour, minute): "3 pm", "15:00",
    "at three" and "noon" are all times. A time with no am/pm may be either.
    12a.7, `loose`: and a bare hour after "around", "about", "now"…"""
    out, t = set(), _ampm(text)
    if loose:
        t = re.sub(_AROUND + r'(\d{1,2})\b(?![:%\d]|[.,]\d|\s*(?:am|pm|a\.m|p\.m|st|nd|rd|th|'
                   r'hours?|minutes?|mins?|people|days?|weeks?|percent|%|kg|km|years?|' + _MONTHS
                   + r'))', lambda m: m.group(1) + ' ' + m.group(2) + ':00', t, flags=re.I)
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
    for m in HHMM.finditer(t):
        out.add((int(m.group(1)), int(m.group(2))))
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
    # 12a.7: "half the time", "twice as long", "a couple of days"
    for w, vs in FRACTION_WORDS.items():
        if re.search(r'\b' + w + r's?\b', t):
            out.update(vs)
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


# 12i.0: numbers_from_source's two rules from the long-summary runs
LENGTH_NOTE = re.compile(r'\(?\s*(?:word count|words?)\s*[:=]?\s*\d+\s*\)?|\(\s*\d+\s*words?\s*\)'
                         r'|\b\d+\s*words?\b(?=\s*\)?\s*$)', re.I)
MONTH_DAYS = {"january": 31, "february": 28, "march": 31, "april": 30, "may": 31, "june": 30,
              "july": 31, "august": 31, "september": 30, "october": 31, "november": 30,
              "december": 31}
END_OF_MONTH = re.compile(r'\bend of (' + '|'.join(MONTH_DAYS) + r')\b', re.I)
# 12a.7: numbers_from_source rejected correct answers. An option's, a
# version's or a step's number is its label ("Option 1", "(2)", "3)"); a note
# on the answer's own length is about the answer ("reduced from ~48 to 33
# words", "~30% shorter"). Neither is a fact the text must give
LABEL_NUM = re.compile(r'\b(?:option|version|choice|alternative|draft|variant|take|step|point|'
                       r'part|summary|no\.?)\s*#?\s*\d{1,2}\b|(?<![\w.])\(?\d{1,2}\)(?=\s)', re.I)
OWN_LENGTH = re.compile(
    r'(?:(?:reduced|cut|trimmed|shortened|condensed|down)\s+)?from\s+(?:about\s+|~)?\d+\s*'
    r'(?:words?\s*)?(?:to|→|->|–)\s*(?:about\s+|~)?\d+\s*words?\b'
    r'|(?:about\s+|~)?\d+(?:\.\d+)?\s*%\s*(?:shorter|briefer|fewer\s+words|less\s+text|reduction|'
    r'smaller)\b|(?:shorter|reduced|cut|trimmed|shortened|condensed)\s+by\s+(?:about\s+|~)?'
    r'\d+(?:\.\d+)?\s*%', re.I)
# …and the text's numbers include the ones it writes as words ("half",
# "twice", "a dozen"), and the ones worked out from them — a sum, a
# difference, how long between two times — which pass this gate for the
# judge to mark: right is fine, wrong costs 2 (12a.7's rubric)
FRACTION_WORDS = {"half": (0.5, 50.0), "quarter": (0.25, 25.0), "twice": (2.0,),
                  "double": (2.0,), "triple": (3.0,), "thrice": (3.0,), "couple": (2.0,),
                  "pair": (2.0,), "dozen": (12.0,)}


# a number and what it counts: a currency before it, a % after it, or the
# word after it ("24 chairs") — a sum is only ever of one kind of thing
NUM_UNIT = re.compile(r'([£$€])?\s?(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)'
                      r'(?:\s*(%)|\s*-?\s*([a-z]+))?', re.I)


def _unit(m) -> str:
    w = (m.group(4) or '').lower()
    return m.group(1) or m.group(3) or (w[:-1] if w.endswith('s') and len(w) > 3 else w)


def derived_numbers(text: str) -> dict[str, set[float]]:
    """what the text's numbers make, kind by kind: each pair's sum and
    difference — £900 and £200 make £1,100 and £700; 24 chairs and 16
    chairs, 40 — never a sum of a price and a count"""
    kinds: dict[str, set[float]] = {}
    for m in NUM_UNIT.finditer(text):
        if _unit(m) and not re.match(r'[\w.]', text[m.start() - 1:m.start()] or ' '):
            kinds.setdefault(_unit(m), set()).add(float(m.group(2).replace(',', '')))
    out = {}
    for u, vs in kinds.items():
        vals = sorted(vs)[:100]
        out[u] = ({round(a + b, 2) for i, a in enumerate(vals) for b in vals[i + 1:]}
                  | {round(b - a, 2) for i, a in enumerate(vals) for b in vals[i + 1:]})
    return out


def durations(times: set[tuple[int, int]]) -> set[float]:
    """the time between two of the text's times, in minutes and in hours"""
    out, mins = set(), sorted({h * 60 + m for h, m in times})
    for i, a in enumerate(mins):
        for b in mins[i + 1:]:
            for d in (b - a, 1440 - (b - a)):
                out |= {float(d), round(d / 60, 2)}
    return out


# …and the answer says it worked it out: a sum near "total", "in all", "comes
# to", "+", "="; a difference near "difference", "extra", "more", "left" —
# so a wrong £75 for the text's £70 is not taken for £45 + £30
WORKED = re.compile(r'\btotal|\bin all\b|\baltogether\b|\bcombined\b|\bsum\b|\boverall\b|'
                    r'\bplus\b|\+|=|\bmak(?:es?|ing)\b|\bbring(?:s|ing)?\b|\bcomes? to\b|'
                    r'\badds? up\b|\bdifference\b|\bextra\b|\bmore\b|\bless\b|\bfewer\b|'
                    r'\bleft\b|\bremaining\b|\bbalance\b|\bsav(?:es?|ing)\b|\bover budget\b|'
                    r'\bunder budget\b|\bincrease\b|\bdecrease\b|\bup from\b|\bdown from\b', re.I)
# a number the answer writes as a length of time — "45-minute delay", "2 hours
# late" — not "every 10 minutes", which is how often
DURATION_WORDS = re.compile(r'\bdela(?:y|yed)\b|\blate\b|\blater\b|\bearl(?:y|ier)\b|\bbehind\b|'
                            r'\boverdue\b|\bwait(?:ing|ed)?\b|\btook\b|\btakes?\b|\blast(?:s|ing|ed)?\b|'
                            r'\blong(?:er)?\b|\bbetween\b|\bgap\b|\bin all\b|\btotal\b', re.I)
DURATION = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*(?:minutes?|mins?|hours?|hrs?)\b', re.I)


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
        # 12a.5: list markers ("1. ", "2) ") aren't claims, so they don't count;
        # 12i.0: nor is a note of the answer's own length — "(109 words)",
        # "(Word count: 89)", "… 12 words" at the very end (Qwen3 adds them)
        body = re.sub(r'(?m)^\s*(?:[-*•]\s*)?\d{1,2}[.)]\s+', '', a)
        body = LENGTH_NOTE.sub('', body)
        body = OWN_LENGTH.sub(' ', LABEL_NUM.sub(' ', body))
        # times are compared as times: "3 pm", "15:00" and "3:00 pm" are the
        # same; "noon" is 12:00
        # 12a.7: and a bare hour the text says as one ("back around 2", "now
        # 8") is that hour: "~2 PM", "8:00"
        given, bad = clock_times(_norm(prompt), loose=True), []
        src = set(numbers(_norm(prompt))) | word_numbers(prompt)

        def keep(m):
            h, mi, ap = _hm(m)
            if not ((h, mi) in given or (not ap and h <= 12 and ((h + 12) % 24, mi) in given)):
                bad.append(m.group(0).strip())
            else:
                times.add((h, mi))
            return ' '
        times = set(given)
        body = re.sub(r'\b(noon|midday)\b', '12:00 pm', body, flags=re.I)
        body = re.sub(r'\bmidnight\b', '12:00 am', body, flags=re.I)
        # 12a.6: "1100" for the text's 11:00 is that time, not an invented 1100
        body = HHMM.sub(lambda m: ' ' if (int(m.group(1)), int(m.group(2))) in given
                        else m.group(0), body)
        body = TIME.sub(keep, body)
        if bad:
            return False, 'invented the time ' + bad[0]
        # …and a number the question writes in words is one it gives
        # 12i.0: "end of October" gives that month's last day, so "by October
        # 31" isn't invented
        for mo in END_OF_MONTH.findall(prompt):
            src.add(float(MONTH_DAYS[mo.lower()]))
        # …a sum or a difference of them, and — written as a length of time
        # ("45-minute delay") — the time between two of its times
        made, gaps = derived_numbers(_norm(prompt)), durations(times)
        body = DURATION.sub(lambda m: ' ' if round(float(m.group(1)), 2) in gaps
                            and DURATION_WORDS.search(m.string[max(0, m.start() - 40):m.end() + 40])
                            else m.group(0), body)
        body = NUM_UNIT.sub(lambda m: ' ' if _unit(m) and round(float(m.group(2).replace(
            ',', '')), 2) in made.get(_unit(m), ()) and WORKED.search(
            m.string[max(0, m.start() - 40):m.end() + 40]) else m.group(0), body)
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
        if check.get('checklist'):
            # 12a.9: each key fact checked, and its number or name looked for in code
            return (f"the judge's checklist of the key facts, checked and scored in code (0 to "
                    f"{check['scale']}, passing at {check['pass_at']}): a key fact missing, or "
                    "without its number or name, −1 (2 at most); a fact wrong — who, when or how "
                    "much — or anything invented −2; several versions −1; a length asked and not "
                    "kept −1 — never the style")
        if check.get('findings'):
            # 12a.8: the judge says what is wrong; the score is worked out from it
            return (f"the judge's findings, scored in code (0 to {check['scale']}, passing at "
                    f"{check['pass_at']}): a missing key fact −1 (2 at most), anything invented "
                    "or wrong −2, several versions −1, a length asked and not kept −1 — never "
                    "the style")
        if check.get('scale'):
            # 12a.6: the rubric is the judge's to read; the page says what it asks.
            # 12a.7: what it says, not how it's set out
            return (f"the judge, on a rubric (0 to {check['scale']}, passing at "
                    f"{check['pass_at']}): the key facts, nothing invented or wrong, one version, "
                    "and the length only if the request asks one — never the style")
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


# 12a.6: a rubric that scores — Summarise's, 0 to 4, passing at 3
SCORED_PROMPT = """You are marking ONE answer from an assistant against a rubric that gives it a score. Read the question, the rubric and the answer, work out the score the rubric gives, and reply with one JSON object and nothing else: {{"score": <a whole number from 0 to {scale}>, "reason": <one short sentence for a person, about the answer: what cost it points, naming each key fact it misses, or that it lost none>}}.

QUESTION
{question}

RUBRIC
{rubric}

THE ANSWER
{answer}"""


# 12a.8: Summarise's judge reports what is wrong and gives no score — the
# code works the score out from what it reports (findings_verdict)
FINDINGS_PROMPT = """You are checking ONE answer from an assistant: a summary. Read the question, the rubric and the answer, and report what the rubric asks as one JSON object and nothing else. Give no score: it is worked out from what you report.
{{"missing_facts": [<each key fact from the rubric's list that the answer does not have, quoted exactly as the list gives it>], "invented_or_wrong": [<each thing the answer says that is invented or wrong, quoting the answer's own words exactly>], "several_versions": <true or false>, "length_ok": <"yes", "no" or "not asked">, "note": <one short sentence for a person>}}

QUESTION
{question}

RUBRIC
{rubric}

THE ANSWER
{answer}"""


# 12a.9: 12a.8's findings let a small judge say "all key facts" of a fact with
# the wrong person, or without its number. It fills in a checklist instead —
# each key fact correct, wrong (quoting the answer) or missing — and the code
# checks each one it calls correct for the fact's numbers and names
CHECKLIST_PROMPT = """You are checking ONE answer from an assistant: a summary. Read the question, the rubric and the answer, and fill in the rubric's checklist as one JSON object and nothing else. Give no score: it is worked out from what you report.
{{"checklist": [<one entry for every key fact in the rubric's list, in its order: {{"fact": <the fact, quoted exactly as the list gives it>, "status": <"correct", "wrong" or "missing">, "quote": <for "wrong": the answer's own words that state it>}}>], "invented_or_wrong": [<anything else the answer says that is invented or wrong, quoting the answer's own words exactly>], "several_versions": <true or false>, "length_ok": <"yes", "no" or "not asked">, "note": <one short sentence for a person>}}

QUESTION
{question}

RUBRIC
{rubric}

THE ANSWER
{answer}"""
# the judge's reply cap: a checklist is a line for each fact
JUDGE_TOKENS, CHECKLIST_TOKENS = 300, 800


def judge_check(item: dict) -> dict | None:
    return next((c for c in item.get("checks") or [] if c.get("type") == "judge"), None)


def judge_tokens(item: dict, cap: int = JUDGE_TOKENS) -> int:
    """how long the judge's reply may be: 12a.9's checklist needs more"""
    return CHECKLIST_TOKENS if (judge_check(item) or {}).get("checklist") else cap


def _rubric_check(item: dict | None) -> dict | None:
    """a judge check that scores (12a.6), or None"""
    c = judge_check(item or {})
    return c if c and c.get("scale") else None


def rubric_key(item: dict) -> str:
    """which rubric a verdict was given on: a verdict is kept only while the
    rubric is the one it read"""
    c = judge_check(item) or {}
    return hashlib.sha256(str(c.get("rubric", "")).encode("utf-8")).hexdigest()[:12]


def judge_prompt(item: dict, answer: str) -> str:
    c = judge_check(item)
    if c.get("checklist"):
        return CHECKLIST_PROMPT.format(question=item["prompt"], rubric=c["rubric"],
                                       answer=answer.strip() or "(empty)")
    if c.get("findings"):
        return FINDINGS_PROMPT.format(question=item["prompt"], rubric=c["rubric"],
                                      answer=answer.strip() or "(empty)")
    if c.get("scale"):
        return SCORED_PROMPT.format(scale=c["scale"], question=item["prompt"], rubric=c["rubric"],
                                    answer=answer.strip() or "(empty)")
    return JUDGE_PROMPT.format(question=item["prompt"], rubric=c["rubric"],
                               answer=answer.strip() or "(empty)")


# ---------------------------------------------------------------------------
# 12a.6: Summarise is marked by the judge on a rubric. A word limit the request
# never states was failing good summaries, and so was a lead-in counted into
# it, a fact written another way, a time written as "1100". The script keeps
# one gate — no number the text doesn't give — and the judge marks the rest
# ---------------------------------------------------------------------------

RUBRIC_SCALE, RUBRIC_PASS = 4, 3
_N = (r'(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|'
      r'a hundred)')
_LENGTH = (r'\b(?:(?:in|under|within|below|at most|no more than|max(?:imum)?|less than|fewer than|'
           r'about|around|to)\s+)?' + _N + r'\s+(?:words?|sentences?|lines?|{bullet}|'
           r'points?|paragraphs?)\b|\b(?:one|a|single)[\s-]+(?:line|liner|sentence|paragraph)\b')
# 12a.9: "as 4 bullets" states a length as "in 4 bullet points" does; 12a.8's
# reading, which didn't, is kept for --compare to tell what that moved
STATED_LENGTH = re.compile(_LENGTH.format(bullet=r'bullets?(?:\s*points?)?'), re.I)
STATED_LENGTH_12A8 = re.compile(_LENGTH.format(bullet=r'bullet(?:\s*points?)?'), re.I)
SHORTER = re.compile(r'\b(?:shorten|shorter|short|tl;?\s?dr|condense|brief(?:ly)?)\b', re.I)
# 12a.7: 12a.6's rubric marked style as much as the summary — a lead-in and a
# closing offer a point each, a bulleted summary "not one summary" — and ~50
# of 145 practice answers failed, about two-thirds of them wrongly. Kept to
# know a rubric generated from it (see _upgraded), which the new one replaces
RUBRIC_12A6 = """Score the answer from 0 to {scale} as a summary of the text in the question. Start at {scale} and take points off as below, never going under 0. It passes at {pass_at} or more.
1. One summary. The text itself, or most of it copied, is not a summary: score it 0. Take off 1 if it is not one summary: versions or options to choose from, or the summary wrapped in a lead-in ("Here's a summary:"), a heading, notes about it or a closing offer. Plain formatting of the one summary, such as short lines or bullets, is fine.
2. The key facts, judged by meaning and not by exact words (a date, time or amount written another way is the same fact). It should keep {need}:
{facts}
Take off 1 for each fact it is short of that, and 1 for each fact it gets wrong.
3. It invents nothing. Take off 2 if it says anything the text doesn't: a name, number, date, time, place or detail that isn't there.
4. Length. {length}{reference}"""
# 12a.8: 12a.7's rubric, kept to know a rubric generated from it (_upgraded).
# The judge found the right faults on it but didn't add up its points: several
# versions cost 0 to 2 where the rubric says 1, and a leave request with every
# fact right got 0 of 4
RUBRIC_12A7 = """Score the answer from 0 to {scale} on what it says as a summary of the text in the question: its content, not its style. Start at {scale} and take points off as below, never going under 0. It passes at {pass_at} or more.
1. The text itself, or most of it copied, is not a summary: score it 0.
2. The key facts, judged by meaning and not by exact words (a date, time or amount written another way is the same fact). It should keep {need}:
{facts}
Take off 1 for each fact it is short of that, 2 at most, and name each missing fact in your reason.
3. Take off 2 if it says anything invented or wrong: a wrong number, person, day, time or place, or a detail the text doesn't give. A number worked out correctly from the text (a total, a difference, how long something took) is not invented; one worked out wrongly is wrong ("half the time" is not "50% faster").
4. Take off 1 if it gives several versions instead of one ("Option 1 / Option 2").
5. Style costs nothing: a lead-in ("Here's a concise summary:"), a closing offer ("Let me know if you'd like it shorter"), headings, bullets, bold and emoji take nothing off. A summary in bullets is one summary.
6. Length. {length}{reference}

Two worked examples, on another text: "Team lunch moves from Thursday to Friday, 12:30, at Luigi's. Sam is booking the table; bring £15 cash."
Example 1 scores 4: every fact is there, and the lead-in, the bullets and the offer cost nothing.
    Here's a concise summary:
    - Lunch moves to Friday, 12:30, at Luigi's
    - Sam is booking
    - Bring £15 cash
    Let me know if you'd like it any shorter!
Example 2 scores 3: the same answer without "- Bring £15 cash" misses one fact, the £15 cash."""
# 12a.9: 12a.8's rubric, kept to know a rubric generated from it (_upgraded).
# The judge reported "all key facts" of a fact with the wrong person (the
# school run plan: "She … will pick up Layla") or without its number (the
# overtime cap with no 10 hours)
RUBRIC_12A8 = """Report what is wrong with the answer as a summary of the text in the question: its content, not its style. Give no score: it is worked out from what you report.
1. Missing facts. The key facts, judged by meaning and not by exact words (a date, time or amount written another way is the same fact; "mum", "mom" and "mother" are one). It should keep {need}:
{facts}
List each of these the answer does not have, quoted exactly as above. Nothing else is a missing fact.
2. Invented or wrong. List anything the answer says that is invented or wrong: a wrong number, person, day, time or place, or a detail the text doesn't give, quoting the answer's own words. A number worked out correctly from the text (a total, a difference, how long something took) is not invented; one worked out wrongly is wrong ("half the time" is not "50% faster").
3. Several versions: true if it gives several versions instead of one ("Option 1 / Option 2"). A summary in bullets is one summary.
4. Style is never a finding: a lead-in ("Here's a concise summary:"), a closing offer ("Let me know if you'd like it shorter"), headings, bullets, bold and emoji.
5. Length. {length}{reference}

Two worked examples, on another text: "Team lunch moves from Thursday to Friday, 12:30, at Luigi's. Sam is booking the table; bring £15 cash." Its key facts: Friday; 12:30; Luigi's; £15 cash.
Example 1 has no findings: every fact is there, and the lead-in, the bullets and the offer are style.
    Here's a concise summary:
    - Lunch moves to Friday, 12:30, at Luigi's
    - Sam is booking
    - Bring £15 cash
    Let me know if you'd like it any shorter!
    {{"missing_facts": [], "invented_or_wrong": [], "several_versions": false, "length_ok": "not asked", "note": "every key fact, one summary"}}
Example 2 is the same answer without "- Bring £15 cash": one fact is missing.
    {{"missing_facts": ["£15 cash"], "invented_or_wrong": [], "several_versions": false, "length_ok": "not asked", "note": "misses the £15 cash"}}"""
RUBRIC = """Check the answer as a summary of the text in the question: its content, not its style. Give no score: it is worked out from what you report.
1. The checklist. Check each of the key facts against the text and the answer. It should keep {need}:
{facts}
For each fact, one of:
- "correct": the answer states it as the text does, with the right person, day, time, place and amount. Judged by meaning, not by exact words: a date, time or amount written another way is the same, and "mum", "mom" and "mother" are one.
- "wrong": the answer states it with something wrong — who does it, the day, the time, the place or the amount. Quote the answer's own words. "She picks up Layla" is wrong when the text says you pick her up.
- "missing": the answer doesn't state it, or states it without its number, time, amount or name ("overtime is capped" for "overtime is capped at 10 hours").
2. Invented or wrong. List anything else the answer says that is invented or wrong: a detail the text doesn't give, or a wrong number, person, day, time or place outside the key facts, quoting the answer's own words. A number worked out correctly from the text (a total, a difference, how long something took) is not invented; one worked out wrongly is wrong ("half the time" is not "50% faster").
3. Several versions: true if it gives several versions instead of one ("Option 1 / Option 2"). A summary in bullets is one summary.
4. Style is never a finding: a lead-in ("Here's a concise summary:"), a closing offer ("Let me know if you'd like it shorter"), headings, bullets, bold and emoji.
5. Length. {length}{reference}

Two worked examples, on another text: "Team lunch moves from Thursday to Friday, 12:30, at Luigi's. Sam is booking the table; bring £15 cash." Its key facts: Friday; 12:30; Luigi's; Sam / booking; £15 / cash.
Example 1: every fact is correct, and the lead-in, the bullets and the offer are style.
    Here's a concise summary:
    - Lunch moves to Friday, 12:30, at Luigi's
    - Sam is booking
    - Bring £15 cash
    Let me know if you'd like it any shorter!
    {{"checklist": [{{"fact": "Friday", "status": "correct"}}, {{"fact": "12:30", "status": "correct"}}, {{"fact": "Luigi's", "status": "correct"}}, {{"fact": "Sam / booking", "status": "correct"}}, {{"fact": "£15 / cash", "status": "correct"}}], "invented_or_wrong": [], "several_versions": false, "length_ok": "not asked", "note": "every key fact, one summary"}}
Example 2: the same answer with "- You're booking the table" for "- Sam is booking" and "- Bring cash" for "- Bring £15 cash". The booking is given to the wrong person, and the cash has lost its amount.
    {{"checklist": [{{"fact": "Friday", "status": "correct"}}, {{"fact": "12:30", "status": "correct"}}, {{"fact": "Luigi's", "status": "correct"}}, {{"fact": "Sam / booking", "status": "wrong", "quote": "You're booking the table"}}, {{"fact": "£15 / cash", "status": "missing"}}], "invented_or_wrong": [], "several_versions": false, "length_ok": "not asked", "note": "says you book, not Sam, and no £15"}}"""


def request_of(prompt: str) -> str:
    """what the person asked for: the words before the text they pasted"""
    return re.split(r'["\u201c]', prompt, maxsplit=1)[0]


def stated_length(prompt: str, pattern: re.Pattern = STATED_LENGTH) -> str:
    """the length the request states, in its own words; "shorter" when it
    only asks for it shorter; "" when it states none"""
    req = request_of(prompt)
    m = pattern.search(req)
    if m:
        return m.group(0).strip()
    return "shorter" if SHORTER.search(req) else ""


def _facts_of(q: dict) -> tuple[list[list[str]], int]:
    """the question's key facts, each a list of ways to say it, and how many
    it must keep — from its facts check, or its contains checks; 12a.7: or
    from the rubric it carries, as a rubric generated from them lists them"""
    for c in q.get("checks") or []:
        if c.get("type") == "facts":
            return c["values"], int(c["n"])
    fs = [list(c["values"]) for c in q.get("checks") or [] if c.get("type") == "contains_any"]
    fs += [[v] for c in q.get("checks") or [] if c.get("type") == "contains_all"
           for v in c["values"]]
    if fs:
        return fs, len(fs)
    return _facts_in((_rubric_check(q) or {}).get("rubric", ""))


_KEEP = re.compile(r"It should keep (?:all (\d+) of these|at least (\d+) of these (\d+)):\n"
                   r"((?:- .*\n)+)")


def _facts_in(rubric: str) -> tuple[list[list[str]], int]:
    """the key facts a generated rubric lists ("- 16 oct / oct 16"), and how
    many it asks for"""
    m = _KEEP.search(rubric or "")
    if not m:
        return [], 0
    facts = [[v.strip() for v in line[2:].split(" / ")] for line in m.group(4).splitlines()
             if line.startswith("- ") and not line.startswith("- (none listed")]
    return facts, int(m.group(1) or m.group(2) or len(facts))


def _show(vs: list[str]) -> str:
    """a fact as the rubric lists it: its first two ways of saying it"""
    return " / ".join(list(dict.fromkeys(v.strip() for v in vs))[:2])


def summarise_rubric(q: dict, template: str | None = None) -> str:
    facts, need = _facts_of(q)
    show = _show
    length = stated_length(q["prompt"])
    if template in (None, RUBRIC, RUBRIC_12A8):
        # 12a.8: what the judge reports for it
        says = ('The request states no length: length_ok is "not asked".' if not length else
                'The request asks for it shorter: length_ok is "yes" if the summary itself (not '
                'a lead-in) is clearly shorter than the text it was given, "no" if not.'
                if length == "shorter" else
                f'The request asks for "{length}": length_ok is "yes" if the summary itself (not '
                'a lead-in) keeps to that, "no" if not.')
    elif length == "shorter":
        says = ("The request asks for it shorter: take off 1 if the summary itself is not clearly "
                "shorter than the text it was given.")
    elif length:
        says = (f'The request asks for "{length}". Measure the summary itself, not a lead-in: take '
                "off 1 if it doesn't keep to that.")
    else:
        says = "The request states no length: take nothing off for length."
    return (template or RUBRIC).format(
        scale=RUBRIC_SCALE, pass_at=RUBRIC_PASS,
        # 12a.9: every key fact counts, unless the request limits the length —
        # then a summary may choose, as far as the question's "at least n" allows
        need=(f"all {len(facts)} of these" if need >= len(facts)
              or (template in (None, RUBRIC) and not length)
              else f"at least {need} of these {len(facts)}"),
        facts="\n".join(f"- {show(f)}" for f in facts) or "- (none listed: what the text is about)",
        length=says,
        reference=f'\nA good summary, for reference: "{q["reference"]}"' if q.get("reference") else "")


def summarise_checks(q: dict) -> list[dict]:
    """12a.6: a Summarise question's checks — one script gate, no number the
    text doesn't give, then the judge's rubric, 0 to 4, passing at 3. 12a.8:
    the judge reports findings on it, and the code scores them; 12a.9: a
    checklist of its key facts, which the code checks"""
    facts, need = _facts_of(q)
    return [{"type": "numbers_from_source"},
            {"type": "judge", "scale": RUBRIC_SCALE, "pass_at": RUBRIC_PASS, "checklist": True,
             # the question's own "at least n", which the rubric gives only where the
             # request limits the length; --compare reads it
             **({"at_least": need} if 0 < need < len(facts) else {}),
             "rubric": summarise_rubric(q)}]


def _upgraded(q: dict) -> bool:
    """12a.7: a Summarise question whose rubric is exactly the one 12a.6 — or,
    12a.8, 12a.7, or, 12a.9, 12a.8 — generated from its facts (the repo's, or
    one the question builder published) takes today's; a rubric someone wrote
    or edited stays theirs"""
    c = _rubric_check(q)
    if not c or c.get("checklist"):
        return False
    if c.get("findings"):
        return c.get("rubric") == summarise_rubric(q, RUBRIC_12A8)
    return c.get("rubric") in (summarise_rubric(q, RUBRIC_12A6), summarise_rubric(q, RUBRIC_12A7))


# ---------------------------------------------------------------------------
# 12a.8: the judge's findings, checked and scored in code. The small judge
# spots what is wrong reliably and adds up points badly, so it reports only
# what is wrong; a claim the answer disproves is dropped, and said
# ---------------------------------------------------------------------------

UNREADABLE = "the judge's reply couldn't be read"
ASKED_AGAIN = "asked the judge again: its reply couldn't be read"
LENGTHS = ("yes", "no", "not asked")
# one word for one person, as the rubric says: "mum", "mom" and "mother" are one
SAME_WORD = {"mom": "mum", "mother": "mum", "mam": "mum", "mummy": "mum", "mommy": "mum",
             "father": "dad", "daddy": "dad", "children": "kids", "child": "kid"}
_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december")
MONTHS = {**{m: m[:3] for m in _MONTH_NAMES}, **{m[:3]: m[:3] for m in _MONTH_NAMES}, "sept": "sep"}
KEY_STOP = {"a", "an", "the", "of", "to", "and", "or", "in", "on", "at", "for", "by", "is", "are",
            "be", "with", "from", "your", "you", "my", "her", "his", "their", "it", "its", "that",
            "this", "will", "was", "has", "have", "up", "am", "pm"}
# 12a.9: a number in words is that number ("twelve" is 12), as the number rule reads it
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split()) if i >= 2}
NUMBER_WORDS.update(thirty="30", forty="40", fifty="50", sixty="60")
_TIME = re.compile(r'(?<![\d,.$£€])(\d{1,2})[:.]([0-5]\d)\s*(a\.?m\.?|p\.?m\.?)?(?![\d])', re.I)
_HOUR = re.compile(r'(?<![\d:.,$£€])(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?![a-z])', re.I)
_MONEY = re.compile(r'(?<![\w:.,])[$£€]?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w:])')


def _clock(h: str, m: str, ap: str | None) -> str:
    """7:00 PM, 19:00 and 7 pm are one time: "7:00", with "7" beside it"""
    hh = int(h) % 12 or 12
    return f"{hh}:{m} {hh}" if m == "00" else f"{hh}:{m}"


def key_words(text: str) -> set[str]:
    """12a.8: the words that carry a fact — its numbers, names, times and
    nouns — as one form each: case, 7:00 PM = 7 pm = 7, 2,450.00 = 2450,
    mum = mom = mother, 3rd = 3, november = nov, a plural's s dropped"""
    t = str(text or "").lower().replace("\u2019", "'")
    t = _TIME.sub(lambda m: " " + _clock(m.group(1), m.group(2), m.group(3)) + " ", t)
    t = _HOUR.sub(lambda m: " " + _clock(m.group(1), "00", m.group(2)) + " ", t)
    t = _MONEY.sub(lambda m: " " + m.group(1).replace(",", "") + (
        "" if not m.group(2) or set(m.group(2)) == {"0"} else "." + m.group(2)) + " ", t)
    out = set()
    for w in re.findall(r"\d+:\d{2}|\d+(?:st|nd|rd|th)(?![a-z])|\d+(?:\.\d+)?|[a-z]+", t):
        w = re.sub(r"^(\d+)(?:st|nd|rd|th)$", r"\1", w)
        w = NUMBER_WORDS.get(w, w)
        w = SAME_WORD.get(w, w)
        # 12a.9: a month's name or its short form — not "Mariam" or "novel"
        w = MONTHS.get(w, w)
        if w.isalpha() and len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in KEY_STOP:
            out.add(w)
    return out


def _need(way: str) -> set[str]:
    """a way of saying a fact, as the key words it needs: "6 pm" needs the
    hour, 6, which "at 6", "6pm" and "18:00" all give"""
    k = key_words(way)
    return {w for w in k if not (w.endswith(":00") and w[:-3] in k)}


def names(text: str) -> set[str]:
    """12a.9: the text's names, as key words — each word it writes with a
    capital, somewhere a sentence doesn't start, and never without: Mariam,
    Hamilton, Copper Kettle, November. "Your mum" isn't one"""
    caps, lower, text = set(), set(), str(text or "")
    for m in re.finditer(r"[A-Za-z]+", text):
        w, before = m.group(0), text[:m.start()].rstrip(" \t\"'\u201c\u2018(*-\u2022")
        if w[0].islower():
            lower.add(w.lower())
        elif before and before[-1] not in ".!?\n" and len(w) > 1:
            caps.add(w.lower())
    return key_words(" ".join(sorted(caps - lower)))


def anchors(way: str, named: set[str]) -> set[str]:
    """12a.9: what a way of saying a fact can't be right without — its
    numbers, times and amounts, and its names"""
    return {w for w in _need(way) if any(ch.isdigit() for ch in w) or w in named}


def _ways(fact: list[str], named: set[str] | None) -> list[str]:
    """the ways of saying a fact that count: when some carry a number or a
    name, only those — "overtime" is not "10 hours / overtime" without its 10"""
    held = [v for v in fact if anchors(v, named or set())]
    return held or list(fact)


def has_fact(answer: str, fact: list[str], named: set[str] | None = None) -> bool:
    """any way of saying the fact whose key words are all in the answer. 12a.9:
    of those with a number or a name (`named`, the text's), when it has one"""
    have = key_words(answer)
    return any(k and k <= have for k in (_need(v) for v in _ways(fact, named)))


def unsaid(answer: str, fact: list[str], named: set[str]) -> str:
    """12a.9: the number or name a fact needs and the answer doesn't give, in
    the fact's words ("10 hours") — "" when it gives one, or has none to give"""
    held = {}
    for v in fact:                  # "9:30 / 9.30" is one time: said once
        held.setdefault(frozenset(anchors(v, named)), v.strip())
    held.pop(frozenset(), None)
    have = key_words(answer)
    if not held or any(k <= have for k in held):
        return ""
    return " or ".join(f"“{v}”" for v in held.values())


def _quoted(claim: str, answer: str) -> bool:
    """a claim that quotes the answer: its words there, in order, whatever the
    case, spacing or quote marks"""
    norm = lambda x: " ".join(re.sub(r"[\u2018\u2019\u201c\u201d\"'`*]", "", str(x)).lower().split())  # noqa: E731
    # 12a.9: a quote cut short with "…" is its pieces, in order
    parts = [norm(p).strip(" .,;:!?-–—") for p in re.split(r"\.\.\.|\u2026", str(claim))]
    parts, text, at = [p for p in parts if p], norm(answer), 0
    for p in parts:
        at = text.find(p, at)
        if at < 0:
            return False
        at += len(p)
    return bool(parts)


COPY_WINDOW, COPY_SHARE = 6, 0.6
# style is never a finding: a claim that is only a lead-in or a closing offer
LEAD_IN = re.compile(r"^(?:here(?:'s| is| are)|sure|certainly|okay|ok)\b.{0,60}$|^[^.!?]{0,60}:$",
                     re.I)


def _style(claim: str) -> bool:
    c = " ".join(str(claim).replace("\u2019", "'").split()).strip(" \"'")
    return bool(c) and len(c.split()) <= 14 and bool(LEAD_IN.match(c) or SIGNOFF.match(c))


def copied(q: dict, answer: str) -> bool:
    """12a.8: the text itself given back, or most of it — no summary, and 0 in
    code, as 12a.7's rubric scored it: most of the text's six-word runs are in
    the answer, and the answer is most of the text's length"""
    prompt = q.get("prompt", "")
    req = request_of(prompt)
    # the quoted text; with no quotes, what follows the request's colon
    src = prompt[len(req):] if len(req) < len(prompt) else prompt.split(":", 1)[-1]
    words = lambda x: re.findall(r"[a-z0-9]+(?:[.:,'][a-z0-9]+)*", str(x).lower())  # noqa: E731
    sw, aw = words(src), words(answer)
    if len(sw) < COPY_WINDOW or len(aw) < COPY_SHARE * len(sw):
        return False
    have = {tuple(aw[i:i + COPY_WINDOW]) for i in range(len(aw) - COPY_WINDOW + 1)}
    runs = [tuple(sw[i:i + COPY_WINDOW]) for i in range(len(sw) - COPY_WINDOW + 1)]
    return sum(r in have for r in runs) >= COPY_SHARE * len(runs)


def _fact_of(claim: str, facts: list[list[str]]) -> int | None:
    """which listed fact a claimed missing fact quotes. 12a.9: the one it
    quotes exactly first — "1 october" is in "31 october" — then the words"""
    c = " ".join(str(claim).lower().split()).strip(" .\"'")
    ways = [[" ".join(v.lower().split()) for v in f] + [_show(f).lower()] for f in facts]
    for i, ws in enumerate(ways):
        if c in ws:
            return i
    inside = lambda a, b: bool(a) and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b)  # noqa: E731
    for i, ws in enumerate(ways):
        if any(inside(w, c) or inside(c, w) for w in ws):
            return i
    return None


def _findings(obj) -> dict | None:
    """the judge's findings, or None when they can't be read as findings"""
    if not isinstance(obj, dict) or not {"missing_facts", "invented_or_wrong"} <= set(obj):
        return None
    miss, wrong = obj.get("missing_facts"), obj.get("invented_or_wrong")
    if not isinstance(miss, list) or not isinstance(wrong, list):
        return None
    rest = _versions_length(obj)
    if rest is None:
        return None
    return {"missing_facts": [str(x) for x in miss if str(x).strip()],
            "invented_or_wrong": [str(x) for x in wrong if str(x).strip()], **rest}


def _versions_length(obj: dict) -> dict | None:
    """several versions, the length and the note, as the judge reports them"""
    sev = obj.get("several_versions", False)
    if isinstance(sev, str) and sev.strip().lower() in ("true", "false"):
        sev = sev.strip().lower() == "true"
    length = str(obj.get("length_ok", "not asked")).strip().lower()
    if not isinstance(sev, bool) or length not in LENGTHS:
        return None
    return {"several_versions": sev, "length_ok": length,
            "note": " ".join(str(obj.get("note") or "").split())[:200]}


def _invented(claims: list[str], answer: str, dropped: list[dict]) -> list[str]:
    """the invented or wrong things the judge quotes from the answer; a claim
    that is style, or doesn't quote it, is dropped, and said"""
    wrong = []
    for claim in claims:
        if _style(claim):
            dropped.append({"kind": "style", "claim": claim,
                            "text": f"the judge counted “{claim}” as invented or wrong; style is "
                                    "never a finding"})
        elif _quoted(claim, answer):
            wrong.append(claim)
        else:
            dropped.append({"kind": "invented", "claim": claim,
                            "text": f"the judge said “{claim}” is invented or wrong; the answer "
                                    "doesn’t say that"})
    return wrong


def findings_verdict(obj, q: dict, answer: str) -> dict | None:
    """12a.8: the score from the judge's findings — 4, less 1 a missing fact
    (2 at most), 2 for anything invented or wrong, 1 for several versions and
    1 for a length asked and not met; passing at 3. A missing fact the answer
    has, or an invented thing it doesn't say, is dropped, and said"""
    f = _findings(obj)
    if f is None:
        return None
    c = judge_check(q) or {}
    scale, pass_at = int(c.get("scale") or RUBRIC_SCALE), int(c.get("pass_at") or RUBRIC_PASS)
    facts, need = _facts_in(c.get("rubric", ""))
    named = names(q.get("prompt", ""))
    missing, dropped, seen = [], [], set()
    for claim in f["missing_facts"]:
        i = _fact_of(claim, facts)
        if i is None:
            dropped.append({"kind": "missing", "claim": claim,
                            "text": f"the judge said it missed “{claim}”, which isn’t one of the "
                                    "key facts"})
        elif i in seen:
            continue
        elif has_fact(answer, facts[i], named):
            seen.add(i)
            dropped.append({"kind": "missing", "claim": claim,
                            "text": f"the judge said it missed “{claim}”; the answer has it"})
        else:
            seen.add(i)
            missing.append(_show(facts[i]))
    wrong = _invented(f["invented_or_wrong"], answer, dropped)
    short = max(0, len(missing) - (len(facts) - need)) if facts else len(missing)
    asked = bool(stated_length(q.get("prompt", "")))
    too_long = asked and f["length_ok"] == "no"
    kept = {"missing_facts": missing, "invented_or_wrong": wrong,
            "several_versions": f["several_versions"], "length_ok": f["length_ok"]}
    if copied(q, answer):
        return {"pass": False, "score": 0, "scale": scale,
                "reason": f"0 of {scale}: the text given back, not a summary", "findings": kept,
                **({"dropped": dropped} if dropped else {}), "judge_raw": obj}
    score = max(0, scale - min(2, short) - (2 if wrong else 0) - (1 if f["several_versions"] else 0)
                - (1 if too_long else 0))
    parts = []
    if missing:
        parts.append("missing: " + "; ".join(missing)
                     + (f" (−{min(2, short)})" if short else " (it may leave that out)"))
    if wrong:
        parts.append("invented or wrong: " + "; ".join(f"“{w}”" for w in wrong) + " (−2)")
    if f["several_versions"]:
        parts.append("several versions (−1)")
    if too_long:
        parts.append("not the length asked (−1)")
    return {"pass": score >= pass_at, "score": score, "scale": scale,
            "reason": f"{score} of {scale}: " + (" · ".join(parts) or "all key facts, one version"),
            "findings": kept,
            **({"dropped": dropped} if dropped else {}),
            # for the audit: what the judge said, as it said it
            "judge_raw": obj}


# ---------------------------------------------------------------------------
# 12a.9: the judge's checklist, checked and scored in code — each key fact
# correct, wrong or missing; one it calls correct without the fact's number or
# name is missing, one it calls missing that the answer has is correct, and a
# "wrong" that doesn't quote the answer is decided by the code; each said
# ---------------------------------------------------------------------------

STATUSES = ("correct", "wrong", "missing")


def _checklist(obj) -> dict | None:
    """the judge's checklist, or None when it can't be read as one"""
    if not isinstance(obj, dict) or not isinstance(obj.get("checklist"), list):
        return None
    rows = []
    for r in obj["checklist"]:
        st = str(r.get("status") or "").strip().lower() if isinstance(r, dict) else ""
        if st not in STATUSES:
            return None
        rows.append({"fact": " ".join(str(r.get("fact") or "").split()), "status": st,
                     "quote": " ".join(str(r.get("quote") or "").split())})
    other = obj.get("invented_or_wrong", [])
    rest = _versions_length(obj)
    if not isinstance(other, list) or rest is None:
        return None
    return {"checklist": rows, "invented_or_wrong": [str(x) for x in other if str(x).strip()],
            **rest}


def checklist_verdict(obj, q: dict, answer: str, rules: dict | None = None) -> dict | None:
    """12a.9: the score from the judge's checklist — 4, less 1 a missing fact
    (2 at most), 2 for a fact wrong or anything invented, 1 for several
    versions and 1 for a length asked and not met; passing at 3. Every key
    fact counts unless the request limits the length (the rubric says which),
    and the code has the last word on each, as said above. `rules` — the facts,
    how many it needs and whether a length was asked — marks it as another
    round would have (--compare)"""
    f = _checklist(obj)
    if f is None:
        return None
    c = judge_check(q) or {}
    scale, pass_at = int(c.get("scale") or RUBRIC_SCALE), int(c.get("pass_at") or RUBRIC_PASS)
    facts, need = _facts_in(c.get("rubric", ""))
    asked = bool(stated_length(q.get("prompt", "")))
    if rules:
        facts, need, asked = rules["facts"], rules["need"], rules["asked"]
    named = names(q.get("prompt", ""))
    said, dropped = {}, []
    for row in f["checklist"]:
        i = _fact_of(row["fact"], facts)
        if i is None:
            dropped.append({"kind": "unknown", "claim": row["fact"],
                            "text": f"the judge checked “{row['fact']}”, which isn’t one of the "
                                    "key facts"})
            continue
        if i in said:
            continue
        show, st, by = _show(facts[i]), row["status"], "judge"
        if st == "wrong" and not (row["quote"] and _quoted(row["quote"], answer)):
            dropped.append({"kind": "wrong", "claim": row["quote"] or show,
                            "text": f"the judge said “{show}” is wrong, quoting "
                                    f"“{row['quote']}”; the answer doesn’t say that"
                                    if row["quote"] else
                                    f"the judge said “{show}” is wrong, quoting nothing"})
            continue                                # the code decides it, below
        if st == "missing" and has_fact(answer, facts[i], named):
            dropped.append({"kind": "missing", "claim": row["fact"],
                            "text": f"the judge said it missed “{row['fact']}”; the answer has it"})
            st, by = "correct", "code"
        elif st == "correct" and unsaid(answer, facts[i], named):
            dropped.append({"kind": "correct", "claim": row["fact"],
                            "text": f"the judge said it has “{show}”; the answer doesn’t say "
                                    f"{unsaid(answer, facts[i], named)}"})
            st, by = "missing", "code"
        said[i] = {"fact": show, "status": st, "by": by,
                   **({"quote": row["quote"]} if st == "wrong" else {})}
    for i, fact in enumerate(facts):
        if i not in said:
            ok = has_fact(answer, fact, named)
            said[i] = {"fact": _show(fact), "status": "correct" if ok else "missing", "by": "code"}
            if not ok:
                dropped.append({"kind": "unchecked", "claim": _show(fact),
                                "text": f"the judge didn’t check “{_show(fact)}”; the answer "
                                        "hasn’t got it"})
    rows = [said[i] for i in range(len(facts))]
    missing = [r["fact"] for r in rows if r["status"] == "missing"]
    wrong = [r for r in rows if r["status"] == "wrong"]
    other = _invented(f["invented_or_wrong"], answer, dropped)
    too_long = asked and f["length_ok"] == "no"
    kept = {"checklist": rows, "invented_or_wrong": other,
            "several_versions": f["several_versions"], "length_ok": f["length_ok"]}
    extra = {**({"dropped": dropped} if dropped else {}), "judge_raw": obj}
    if copied(q, answer):
        return {"pass": False, "score": 0, "scale": scale,
                "reason": f"0 of {scale}: the text given back, not a summary", "findings": kept,
                **extra}
    short = max(0, len(missing) - (len(facts) - need))
    score = max(0, scale - min(2, short) - (2 if wrong or other else 0)
                - (1 if f["several_versions"] else 0) - (1 if too_long else 0))
    parts = []
    if missing:
        parts.append("missing: " + "; ".join(missing)
                     + (f" (−{min(2, short)})" if short else " (it may leave that out)"))
    if wrong:
        parts.append("wrong: " + "; ".join(f"{r['fact']} (“{r['quote']}”)" for r in wrong))
    if other:
        parts.append("invented or wrong: " + "; ".join(f"“{w}”" for w in other))
    if wrong or other:
        parts[-1] += " (−2)"
    if f["several_versions"]:
        parts.append("several versions (−1)")
    if too_long:
        parts.append("not the length asked (−1)")
    return {"pass": score >= pass_at, "score": score, "scale": scale,
            "reason": f"{score} of {scale}: " + (" · ".join(parts) or "all key facts, one version"),
            "findings": kept, **extra}


def stub_checklist(rubric: str, answer: str, question: str = "") -> dict:
    """the stand-in judge's checklist: each listed fact correct when the
    answer has it, missing when not; several versions where it numbers its
    options"""
    facts, _ = _facts_in(rubric)
    named = names(question)
    return {"checklist": [{"fact": _show(f), "status": "correct" if has_fact(answer, f, named)
                           else "missing"} for f in facts],
            "invented_or_wrong": [],
            "several_versions": bool(re.search(r"\b(?:option|version)\s*[1-9]\b", answer or "",
                                               re.I)),
            "length_ok": "not asked", "note": "the stand-in judge's checklist"}


def stub_findings(rubric: str, answer: str) -> dict:
    """the stand-in judge's findings: each listed fact the answer hasn't got,
    and several versions where it numbers its options"""
    facts, _ = _facts_in(rubric)
    return {"missing_facts": [_show(f) for f in facts if not has_fact(answer, f)],
            "invented_or_wrong": [],
            "several_versions": bool(re.search(r"\b(?:option|version)\s*[1-9]\b", answer or "",
                                               re.I)),
            "length_ok": "not asked", "note": "the stand-in judge's findings"}


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


# 12p.2: the pilot's TL;DR is a hidden question, so the stand-in judge knows
# the tests' invented one (tests/fixtures/everyday_hidden_invented.jsonl) by its
# text, never the server's
PILOT_NOTICE = "the fixture pool will shut at 10:45"


def stub_verdict(answer: str, question: str = PILOT_NOTICE) -> dict:
    """The deterministic stand-in the tests and dry runs use. The tests'
    pilot TL;DR (an invented pool notice) it checks: at most two sentences,
    10:45, and Wednesday. Any other judged question has passed its script
    checks before it reaches a judge, and the stand-in agrees with them."""
    if PILOT_NOTICE not in (question or ""):
        return {"pass": True, "reason": "the stand-in judge agrees with the script checks"}
    a = (answer or "").strip()
    if not a:
        return {"pass": False, "reason": "no answer"}
    if "10:45" not in a:
        return {"pass": False, "reason": "it doesn't give the 10:45 closing time"}
    if "wednesday" not in a.lower():
        return {"pass": False, "reason": "it doesn't say Wednesday"}
    if _sentences(a) > 2:
        return {"pass": False, "reason": "more than two sentences"}
    return {"pass": True, "reason": "shuts 10:45 on Wednesday, in two sentences or fewer"}


def stub_reply(prompt: str) -> str:
    """What the fake judge backend answers to a judge_prompt()."""
    answer = prompt.rsplit("THE ANSWER\n", 1)[-1]
    question = prompt.split("QUESTION\n", 1)[-1].split("\n\nRUBRIC", 1)[0]
    rubric = prompt.split("\n\nRUBRIC\n", 1)[-1].rsplit("\n\nTHE ANSWER\n", 1)[0]
    if prompt.startswith(CHECKLIST_PROMPT.split("{{", 1)[0]):
        return json.dumps(stub_checklist(rubric, answer, question), ensure_ascii=False)
    if prompt.startswith(FINDINGS_PROMPT.split("{{", 1)[0]):
        return json.dumps(stub_findings(rubric, answer), ensure_ascii=False)
    return json.dumps(stub_verdict(answer, question), ensure_ascii=False)


def parse_verdict(text: str, check: dict | None = None, q: dict | None = None,
                  answer: str = "") -> dict | None:
    """the judge's reply: {pass, reason} — and, on a rubric that scores
    (12a.6), its score, the pass being the score at or over the line. 12a.8:
    on a rubric of findings, the score the code works out from them (q, the
    question, and the answer they are about); 12a.9, from a checklist"""
    from service import llm
    obj = llm.extract_json(text or "")
    if not isinstance(obj, dict):
        return None
    if (check or {}).get("checklist"):
        return checklist_verdict(obj, {**(q or {}), "checks": [check]}, answer)
    if (check or {}).get("findings"):
        return findings_verdict(obj, {**(q or {}), "checks": [check]}, answer)
    reason = " ".join(str(obj.get("reason") or "").split())[:200]
    scale = int((check or {}).get("scale") or 0)
    if scale:
        sc = obj.get("score")
        if isinstance(sc, str) and sc.strip().isdigit():
            sc = int(sc.strip())
        if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not 0 <= sc <= scale:
            # a pass or fail with no score (the stand-in judge) reads as the top or 1
            if not isinstance(obj.get("pass"), bool):
                return None
            sc = scale if obj["pass"] else 1
        sc = int(round(sc))
        ok = sc >= int(check.get("pass_at") or scale)
        return {"pass": ok, "score": sc, "scale": scale,
                "reason": f"{sc} of {scale}: " + (reason or ("nothing taken off" if sc == scale
                                                             else "the judge took points off"))}
    if not isinstance(obj.get("pass"), bool):
        return None
    return {"pass": obj["pass"],
            "reason": reason or ("the judge says it is right" if obj["pass"]
                                 else "the judge says it is wrong")}


def stub_for(q: dict, answer: str) -> dict:
    """the stand-in judge's verdict on a question, scored when its rubric scores"""
    c = judge_check(q) or {}
    if c.get("checklist"):
        return checklist_verdict(stub_checklist(c["rubric"], answer, q.get("prompt", "")), q,
                                 answer)
    if c.get("findings"):
        return findings_verdict(stub_findings(c["rubric"], answer), q, answer)
    v = stub_verdict(answer, q["prompt"])
    return parse_verdict(json.dumps(v), judge_check(q)) or v


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
    """the bank's version: {"date": …, "hash": …, "split": …}. 12n.1: every
    edit that leaves the bank different is a new one, dated the day of the
    last edit; an edit undone is the version before it again"""
    h, d, log = bank_hash(load_bank()), _edit_digest(), edit_log()
    return {"date": max(WORDING_DATE, _day(log[-1]["at"])) if log else WORDING_DATE,
            "hash": hashlib.sha256(f"{h}|{d}".encode("utf-8")).hexdigest()[:8] if d else h,
            "split": SPLIT}


# ---------------------------------------------------------------------------
# 12g.2: the split, by the exam's function and salt — nothing by hand
# ---------------------------------------------------------------------------

def qid(q: dict) -> str:
    return _eb.qid_of(q["prompt"])


def half(q: dict) -> str:
    """HIDDEN ("report") or PRACTICE ("diagnose"). 12n.1: an edited question
    keeps the half it was in, whatever its words hash to now"""
    h = q.get("half")
    return h if h in (HIDDEN, PRACTICE) else _dx.split_of(qid(q))


def split_counts(questions=None) -> dict[str, dict]:
    """{group: {"hidden": n, "practice": m}}, in the groups' order"""
    qs = load_bank() if questions is None else questions
    out = {g: {"hidden": 0, "practice": 0} for g in groups()}
    for q in qs:
        out[q["group"]]["hidden" if half(q) == HIDDEN else "practice"] += 1
    return {g: c for g, c in out.items() if c["hidden"] or c["practice"]}


# ---------------------------------------------------------------------------
# 12n.1: a question edited where it is read. Every save is one line of
# BENCH_ROOT/everyday/edits.jsonl — beside what the question builder
# published, never the repo's bank, so a later image keeps it — and the bank
# is read through them: an edited question as it reads now, a retired one
# gone (and in BENCH_ROOT/everyday/retired.jsonl with its reason). Each
# keeps the half it was in: a practice question has been seen and is never
# made hidden; a hidden one made practice is revealed on purpose. Every save
# is a new version of the bank, and can be undone
# ---------------------------------------------------------------------------

EDITS_NAME = "edits.jsonl"
EDITED = ("prompt", "reference", "checks", "group", "half")
PRACTICE_STAYS = ("A practice question has been seen, so it can't be made hidden. Retire it, "
                  "and write a new one")


def edits_path() -> Path:
    return built_dir() / EDITS_NAME


def retired_here_path() -> Path:
    return built_dir() / "retired.jsonl"


def _day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t))


def edit_log() -> list[dict]:
    """every save, oldest first: {n, id, action, at, by, why, what, before, after, undoes?}"""
    try:
        text = edits_path().read_text(encoding="utf-8")
    except OSError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _now_rows(log: list[dict] | None = None) -> dict[str, dict | None]:
    """each edited question as it reads now; None when it is retired"""
    return {r["id"]: r["after"] for r in (edit_log() if log is None else log)}


def _edited(rows: list[dict]) -> list[dict]:
    now = _now_rows()
    if not now:
        return rows
    out = []
    for q in rows:
        if q["id"] not in now:
            out.append(q)
        elif now[q["id"]] is not None:
            out.append(_valid(dict(now[q["id"]]), f"{EDITS_NAME}: {q['id']}"))
    return out


def _edit_digest() -> str:
    """what the edits leave different from the bank as it was published —
    none once every edit is undone"""
    now = _now_rows()
    if not now:
        return ""
    base = {q["id"]: q for q in _base_bank()}
    diff = {}
    for i, a in now.items():
        b = base.get(i)
        if b is None:
            continue
        if a is None:
            diff[i] = None
            continue
        d = {k: a.get(k) for k in EDITED if k != "half" and a.get(k) != b.get(k)}
        if a.get("half") and a["half"] != half(b):
            d["half"] = a["half"]
        if d:
            diff[i] = d
    return hashlib.sha256(json.dumps(diff, sort_keys=True).encode("utf-8")).hexdigest()[:8] \
        if diff else ""


def question(qid_: str) -> dict | None:
    """one question of the bank as it reads now, by its id"""
    return next((q for q in load_bank() if q["id"] == qid_), None)


def apply_changes(q: dict, changes: dict) -> dict:
    """the question with these changes, checked as the bank is read — or
    ValueError in one line. `rubric` is its judge check's rubric; `half` may
    go from hidden to practice, never back"""
    new = {k: v for k, v in q.items() if k != "edited"}
    was = new["half"] = half(q)
    for k, v in (changes or {}).items():
        if k == "prompt":
            if not str(v or "").strip():
                raise ValueError("The question can't be empty")
            new["prompt"] = str(v).strip()
        elif k == "reference":
            new["reference"] = str(v or "").strip()
        elif k == "checks":
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except ValueError as e:
                    raise ValueError(f"The checks aren't valid JSON: {e}") from None
            if not isinstance(v, list) or not v:
                raise ValueError("The checks are a list of at least one check")
            for c in v:
                why = _bad_check(c)
                if why:
                    raise ValueError(f"A check: {why}")
            new["checks"] = v
        elif k == "rubric":
            c = judge_check(new)
            if not c:
                raise ValueError("This question has no judge: edit its checks instead")
            if not str(v or "").strip():
                raise ValueError("A rubric can't be empty")
            new["checks"] = [{**x, "rubric": str(v).strip()} if x is c else x
                             for x in new["checks"]]
        elif k == "group":
            if v not in groups():
                raise ValueError(f"There is no group {v!r}")
            new["group"] = v
        elif k == "half":
            if v not in (HIDDEN, PRACTICE):
                raise ValueError("A question is hidden or practice")
            if v == HIDDEN and was == PRACTICE:
                raise ValueError(PRACTICE_STAYS)
            new["half"] = v
        else:
            raise ValueError(f"{k} isn't something an edit changes")
    return _valid(new, "the edit")


def _changed(a: dict, b: dict) -> list[str]:
    return [k for k in EDITED if a.get(k) != b.get(k)]


def _write_retired(log: list[dict]) -> None:
    """the questions retired here, each with its reason — as 12a.6's retired.jsonl"""
    last = {r["id"]: r for r in log}
    rows = [{"retired": _day(r["at"]), "why": r["why"], "by": r["by"], **r["before"]}
            for r in last.values() if r["after"] is None]
    p = retired_here_path()
    if rows or p.exists():
        p.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n"
                             for x in rows), encoding="utf-8")


def _save(qid_: str, action: str, before: dict, after: dict | None, why: str, by: str,
          what: list[str], undoes: int | None = None) -> dict:
    log = edit_log()
    n = (log[-1]["n"] + 1) if log else 1
    at = time.time()
    b = {k: v for k, v in before.items() if k != "edited"}
    b["half"] = half(before)
    if after is not None:
        after = {**{k: v for k, v in after.items() if k != "edited"},
                 "edited": {"n": n, "action": action, "by": by, "at": at, "why": why}}
    rec = {"n": n, "id": qid_, "action": action, "at": at, "by": by, "why": why, "what": what,
           "before": b, "after": after, **({"undoes": undoes} if undoes else {})}
    built_dir().mkdir(parents=True, exist_ok=True)
    with open(edits_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    _write_retired(log + [rec])
    return rec


def _who_why(why: str, by: str) -> tuple[str, str]:
    why, by = str(why or "").strip()[:300], str(by or "").strip()[:80]
    if not by:
        raise ValueError("Who: every edit keeps a name")
    if not why:
        raise ValueError("Why: every edit keeps its reason")
    return why, by


def edit(qid_: str, changes: dict, why: str, by: str) -> dict:
    """save an edit: the record of it — or KeyError (no such question),
    ValueError (why not, in one line)"""
    q = question(qid_)
    if q is None:
        raise KeyError(qid_)
    why, by = _who_why(why, by)
    new = apply_changes(q, changes)
    what = _changed(new, {**q, "half": half(q)})
    if not what:
        raise ValueError("Nothing changed")
    return _save(qid_, "edit", q, new, why, by, what)


def retire(qid_: str, why: str, by: str) -> dict:
    q = question(qid_)
    if q is None:
        raise KeyError(qid_)
    why, by = _who_why(why, by)
    return _save(qid_, "retire", q, None, why, by, ["retired"])


def undo(qid_: str, by: str) -> dict:
    """the question as it read before its last edit not undone yet"""
    by = str(by or "").strip()[:80]
    if not by:
        raise ValueError("Who: every edit keeps a name")
    log = [r for r in edit_log() if r["id"] == qid_]
    if not log:
        raise KeyError(qid_)
    undone = {r["undoes"] for r in log if r.get("undoes")}
    last = next((r for r in reversed(log) if r["action"] != "undo" and r["n"] not in undone), None)
    if last is None:
        raise ValueError("Nothing left to undo")
    now = log[-1]["after"]
    return _save(qid_, "undo", now if now is not None else last["before"], last["before"],
                 f"undo #{last['n']}: {last['why']}", by, last["what"], undoes=last["n"])


def history(qid_: str) -> list[dict]:
    """one question's saves, newest first: who, when, why, what — no text"""
    log = [r for r in edit_log() if r["id"] == qid_]
    undone = {r["undoes"] for r in log if r.get("undoes")}
    return [{k: r.get(k) for k in ("n", "action", "at", "by", "why", "what", "undoes")}
            | {"undone": r["n"] in undone} for r in reversed(log)]


def impact(qid_: str, changes: dict, results: Path) -> dict:
    """What saving these changes would do to the answers kept, before it is
    saved: how many are marked against the question, which marks would
    flip, which go to the judge, and — the words changed — which answers no
    longer count until the question is asked again. No model runs"""
    q = question(qid_)
    if q is None:
        raise KeyError(qid_)
    new = apply_changes(q, changes)
    reworded = new["prompt"] != q["prompt"]
    out = {"answers": 0, "same": 0, "flips": [], "judge": [], "unasked": []}
    for d in sorted(p for p in results.iterdir() if p.is_dir()) if results.is_dir() else []:
        prev = read(d)
        if not prev or prev.get("earlier"):
            continue
        a = answers(d).get(answer_key(q))
        if not a:
            continue
        name = prev.get("model") or _model_id(d)
        out["answers"] += 1
        if reworded:
            out["unasked"].append(name)
            continue
        it = next((x for x in prev.get("items") or [] if x["id"] == qid_), None)
        now = _item(new, {"filtered_resps": [a["raw"]], "resps": [[a["raw"]]]}, {},
                    {qid_: it} if it else {}, verdict_memory(d))
        was = (it or {}).get("pass")
        if now["pass"] is None:
            out["judge"].append(name)
        elif now["pass"] != was:
            out["flips"].append({"model": name, "from": was, "to": now["pass"]})
        else:
            out["same"] += 1
    out["line"] = impact_line(out)
    return out


def impact_line(i: dict) -> str:
    """"3 would flip: Qwen3-1.7B ✗→✓, … · 2 go to the judge" """
    mark = {True: "✓", False: "✗", None: "…"}
    if not i["answers"]:
        return "No model has answered it yet: nothing is marked again"
    if i["unasked"]:
        n = len(i["unasked"])
        return (f"The words change: {n} answer{'s' if n > 1 else ''} to the old words stop "
                "counting — each model reads it as changed, not re-asked yet, until it is asked "
                "again")
    parts = []
    if i["flips"]:
        parts.append(f"{len(i['flips'])} would flip: " + ", ".join(
            f"{f['model'].split('/')[-1]} {mark[f['from']]}→{mark[f['to']]}" for f in i["flips"]))
    if i["judge"]:
        parts.append(f"{len(i['judge'])} go{'es' if len(i['judge']) == 1 else ''} to the judge")
    if i["same"]:
        parts.append(f"{i['same']} keep{'s' if i['same'] == 1 else ''} its mark"
                     if i["same"] == 1 else f"{i['same']} keep their marks")
    return " · ".join(parts) or "Nothing changes"


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


# what a judge's verdict carries beside its pass and reason: the score; 12a.8:
# the findings it was scored from, the claims dropped, and the judge's own reply
VERDICT_EXTRA = ("score", "findings", "dropped", "judge_raw")


def _item(q: dict, rec: dict, verdicts: dict, before: dict, memory: dict | None = None) -> dict:
    """one question's answer, marked. 12n.1: `memory`, the judge's verdicts
    kept by question, rubric and answer — an edit undone gets its marks back"""
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
            # 12a.6: an earlier verdict is kept only if the judge gave it, on this
            # rubric, to this answer — never a script check's mark from before
            if v is None and old.get("pass") is not None and old.get("answer_text") == ans \
                    and old.get("rubric") == rubric_key(q):
                v = {"pass": old["pass"], "reason": old["reason"],
                     **{k: old[k] for k in VERDICT_EXTRA if k in old}}
            if v is None and memory:
                v = memory.get(_memory_key(q["id"], rubric_key(q), ans))
            if v is None:
                it.update({"pass": None, "reason": old.get("reason") if (
                    old.get("answer_text") == ans and old.get("pass") is None
                    and old.get("reason")) else WAITING})
            else:
                it.update({"pass": bool(v["pass"]), "reason": v["reason"], "rubric": rubric_key(q),
                           **{k: v[k] for k in VERDICT_EXTRA if v.get(k) is not None}})
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
    need_hidden()                 # 12p.1: nothing is scored without the hidden set
    got = keep_answers(model_dir)
    bank = load_bank()
    now = version()
    prev = read(model_dir) or {}
    before = {it["id"]: it for it in prev.get("items") or []}
    verdicts = verdicts or {}
    memory = verdict_memory(model_dir)
    current = [(q, got[answer_key(q)]) for q in bank if answer_key(q) in got]
    if current:
        items = [_item(q, {"filtered_resps": [a["raw"]], "resps": [[a["raw"]]]},
                       verdicts, before, memory) for q, a in current]
        new = len({q["id"] for q, _ in current} & set(asked or ()))
        # 12n.1: a question reworded since this model answered it — its answer
        # was to other words: not counted until it is asked again
        was = {a.get("id") for a in got.values()}
        stamp = {"version": dict(now), "earlier": False,
                 # what this marking was: answers the run just gave, answers
                 # from before marked again, and what the model was never asked
                 "marking": {"new": new, "remarked": len(items) - new},
                 "unasked": len(bank) - len(items),
                 "changed": [q["id"] for q in bank if answer_key(q) not in got and q["id"] in was]}
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
        items = [_item(q, recs[q["id"]], verdicts, before, memory) for q in bank if q["id"] in recs]
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
    remember_verdicts(model_dir, out)
    return p


# 12n.1: the judge's verdicts, kept by question, rubric and answer — so a
# rubric changed and changed back, or a question edited and the edit undone,
# has its marks back without asking the judge again. A new judge (12i.1)
# forgets them all
VERDICTS_NAME = "everyday_verdicts.json"


def _memory_key(qid_: str, rubric: str, answer: str) -> str:
    return f"{qid_}|{rubric}|{prompt_hash(answer)}"


def verdict_memory(model_dir: Path) -> dict:
    try:
        return json.loads((model_dir / VERDICTS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def remember_verdicts(model_dir: Path, out: dict) -> None:
    mem = verdict_memory(model_dir)
    more = {_memory_key(it["id"], it["rubric"], it.get("answer_text") or ""):
            {"pass": it["pass"], "reason": it.get("reason") or "",
             **{k: it[k] for k in VERDICT_EXTRA if it.get(k) is not None}}
            for it in (out or {}).get("items") or []
            if it.get("rubric") and it.get("pass") is not None}
    if more and any(mem.get(k) != v for k, v in more.items()):
        mem.update(more)
        (model_dir / VERDICTS_NAME).write_text(json.dumps(mem, sort_keys=True), encoding="utf-8")


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

# 12d.1: how an Everyday answer is generated. Everything below is the ONE
# place these are said: build_task writes them into the task, the runner reads
# the reasoning budget from here, and the Playground's defaults are this dict —
# "what you see is what was scored"
MAX_GEN_TOKS = 512
UNTIL = ["\n\n\n\n"]


def run_settings(archinfo: dict | None) -> dict:
    """The settings an Everyday answer is generated with, for a model whose
    archinfo this is (None: the task's defaults). `thinking` is None: the run
    passes no enable_thinking, so the chat template's own default applies;
    `thinking_on` says what that default is, for a page to show."""
    from service import config
    a = archinfo or {}
    reasoning = bool(a.get("reasoning_template"))
    on = bool(a.get("thinking_default_on") if a.get("thinking") == "switch"
              else a.get("thinking") == "always" or reasoning)
    return {"chat_template": True, "system": "", "thinking": None, "thinking_on": on,
            "can_think": reasoning or a.get("thinking") in ("switch", "always"),
            "max_gen_toks": config.EVERYDAY_REASONING_MAX_GEN_TOKS if reasoning else MAX_GEN_TOKS,
            "do_sample": False, "temperature": 0.0, "until": list(UNTIL)}


def build_task(dest: Path, only: list[str] | None = None) -> Path:
    """The bank as a harness task under `dest`: the items, and the yaml with
    their absolute path filled in. Returns the directory for --include_path.
    12a.5: `only` — the ids a run asks, the questions the model has no answer
    to (unanswered()); the whole bank when None."""
    dest.mkdir(parents=True, exist_ok=True)
    bank = load_bank()                # a bad bank fails here, before any GPU
    items = dest / f"{TASK}.jsonl"
    if only is None and not built_path().exists() and not hidden_path().exists():
        shutil.copyfile(BANK_PATH, items)
    else:
        want = set(only) if only is not None else {q["id"] for q in bank}
        items.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n"
                                 for q in bank if q["id"] in want), encoding="utf-8")
    s = run_settings(None)            # 12d.1: the settings the Playground shows too
    yaml = (TEMPLATE_PATH.read_text(encoding="utf-8")
            .replace("__ITEMS_PATH__", str(items.resolve()))
            .replace("__UNTIL__", json.dumps(s["until"]))
            .replace("__MAX_GEN_TOKS__", str(s["max_gen_toks"]))
            .replace("__DO_SAMPLE__", "true" if s["do_sample"] else "false")
            .replace("__TEMPERATURE__", f"{float(s['temperature'])}"))
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
        out = mark(model_dir, {q["id"]: stub_for(q, answers[q["id"]]) for q in todo},
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
                            json=True, max_tokens=judge_tokens(q),
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
    """The poller's half: the judge's replies in, everyday.json out. 12a.8: a
    reply that can't be read is asked once more; after that it is "the judge's
    reply couldn't be read" — neither a pass nor a fail, and waiting"""
    verdicts, again = {}, []
    qs = {q["id"]: q for q in load_bank()}
    prev = read(model_dir) or {}
    answers = {it["id"]: it.get("answer_text") or "" for it in prev.get("items") or []}
    for cid, res in results.items():
        parts = str(cid).split(":")
        if parts[0] != "everyday":
            continue
        qid, retried = parts[-1], len(parts) > 2 and parts[1] == "retry"
        q = qs.get(qid) or {}
        v = None if getattr(res, "error", None) else parse_verdict(
            getattr(res, "text", ""), judge_check(q), q, answers.get(qid, ""))
        if v is None and not getattr(res, "error", None) and not retried and q \
                and answers.get(qid):
            again.append(q)
            continue
        verdicts[qid] = v or {"pass": None, "reason": UNREADABLE}
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
        elif it["pass"] is None and any(q["id"] == it["id"] for q in again):
            it["reason"] = ASKED_AGAIN
    out["waiting"] = sum(1 for it in out["items"] if it["pass"] is None)
    write(model_dir, out)
    if again:
        _ask_again(model_dir, again, answers)
    return out


def _ask_again(model_dir: Path, qs: list[dict], answers: dict) -> str | None:
    """12a.8: the judge asked once more for the replies it couldn't be read on,
    as a re-mark batch the poller lands as it lands any other"""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from service import db, llm
    if _judge.is_stub() or _judge.blocked():
        return None
    backend = llm.client("judge")
    reqs = [llm.Request(custom_id=f"{REMARK}:{model_dir.name}:retry:{q['id']}", system="",
                        json=True, max_tokens=judge_tokens(q),
                        user=judge_prompt(q, answers[q["id"]]),
                        meta={"kind": "everyday", "id": q["id"]}) for q in qs]
    bid = backend.submit(reqs)
    db.batch_add(bid, "everyday_remark", 0, len(reqs), backend.name, backend.model)
    db.batch_progress(bid, f"0/{len(reqs)} done")
    return bid


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
    # 12n.1: and the verdicts kept for an undo are the last judge's too
    (model_dir / VERDICTS_NAME).unlink(missing_ok=True)
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


# ---------------------------------------------------------------------------
# 12a.6: re-marking with the judge — no model runs. Before the first one, each
# model's marks are kept (BEFORE_NAME), so the change reads model by model
# ---------------------------------------------------------------------------

# 12a.7: each round of re-marking keeps its own — 12a.6's is the marks from
# before 12a.6, 12a.7's the marks from before 12a.7; 12a.8's before is 12a.7's
# marks, the judge adding points up; 12a.9's, 12a.8's marks, on findings
BEFORE_NAME = "everyday_before_12a9.json"
REMARK = "everyday-remark"
# 12a.9: the facts a question gained this round; --compare counts what they
# moved with "every fact where no length is set"
GAINED_12A9 = {"everyday-summarising-long-01": ["9:30 / 9.30"]}


def _counts(out: dict | None) -> dict:
    if not out:
        return {}
    return {"passed": out.get("passed"), "total": out.get("total"),
            "waiting": out.get("waiting") or 0, "groups": out.get("groups") or {}}


def remark(results: Path, want: set[str] | None = None, judge: bool = False,
           only: set[str] | None = None, snapshot: bool = True) -> dict:
    """Mark every model again; with `judge`, send each answer waiting on the
    judge — one batch for all of them, which the poller finishes. Returns
    {"models": {id: summary}, "batch_id", "sent"}. The marks before the first
    of these are kept in BEFORE_NAME, and never written over. 12n.1: `only`,
    the questions an edit changed — the only ones sent to the judge — and no
    snapshot for an edit's re-mark"""
    before_f = results / BEFORE_NAME
    before = json.loads(before_f.read_text(encoding="utf-8")) if before_f.exists() else None
    snap, lines, todo, dirs, passes = {}, {}, [], {}, {}
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        prev = read(d)
        out = mark(d)
        if out is None:
            continue
        if prev:
            snap[prev.get("model") or out["model"]] = _counts(prev)
            # 12a.9: and each Summarise answer's pass, both halves, for --compare's causes
            passes[prev.get("model") or out["model"]] = {
                it["id"]: it.get("pass") for it in prev.get("items") or []
                if it.get("group") == "summarising"}
        write(d, out)
        dirs[d.name] = d
        lines[out["model"]] = summary(out) + (
            " · earlier wording, kept as marked" if out.get("earlier") else "")
        if judge and not out.get("earlier"):
            answers = {it["id"]: it["answer_text"] for it in out["items"]}
            todo += [(d.name, q, answers[q["id"]]) for q in _pending(out)
                     if only is None or q["id"] in only]
    if before is None and snap and snapshot:
        before_f.write_text(json.dumps({"at": time.time(), "version": version(), "models": snap,
                                        "summarise": passes}, indent=1), encoding="utf-8")
    res = {"models": lines, "batch_id": None, "sent": 0}
    if not todo:
        return res
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))        # run as a script: the service's judge and database
    from service import db, llm
    ident = _judge.identity()
    if _judge.is_stub():
        by = {}
        for name, q, ans in todo:
            by.setdefault(name, {})[q["id"]] = stub_for(q, ans)
        for name, vs in by.items():
            out = mark(dirs[name], vs, judge={"id": ident["id"], "version": _judge.version(ident)["key"],
                                              "provisional": False})
            write(dirs[name], out)
            lines[out["model"]] = summary(out)
        res["sent"] = len(todo)
        return res
    why = _judge.blocked()
    if why:
        raise RuntimeError(f"the judge isn't set up on this server: {why}")
    backend = llm.client("judge")
    stamp = llm.provisional(backend, "marked")
    reqs = [llm.Request(custom_id=f"{REMARK}:{name}:{q['id']}", system="", json=True,
                        max_tokens=judge_tokens(q), user=judge_prompt(q, ans),
                        meta={"kind": "everyday", "id": q["id"]}) for name, q, ans in todo]
    bid = backend.submit(reqs)
    for name in {n for n, _, _ in todo}:
        out = read(dirs[name])
        out["judge"] = {"id": ident["id"], "version": _judge.version(ident)["key"],
                        "provisional": bool(stamp), "batch_id": bid}
        write(dirs[name], out)
    db.batch_add(bid, "everyday_remark", 0, len(reqs), backend.name, backend.model)
    db.batch_progress(bid, f"0/{len(reqs)} done")
    res.update(batch_id=bid, sent=len(reqs))
    return res


def finish_remark(results_dir: Path, results: dict) -> list[str]:
    """the poller's half of a re-mark: each model's verdicts into its marks"""
    by: dict[str, dict] = {}
    for cid, r in results.items():
        parts = str(cid).split(":")
        if len(parts) == 3 and parts[0] == REMARK:
            by.setdefault(parts[1], {})[f"everyday:0:{parts[2]}"] = r
        elif len(parts) == 4 and parts[0] == REMARK and parts[2] == "retry":
            # 12a.8: the second asking of a reply that couldn't be read
            by.setdefault(parts[1], {})[f"everyday:retry:{parts[3]}"] = r
    done = []
    for name, rs in by.items():
        d = results_dir / name
        if d.is_dir() and finish(d, rs) is not None:
            done.append(name)
    return done


def compare(results: Path) -> str:
    """before and after, model by model — a table for a PR"""
    f = results / BEFORE_NAME
    if not f.exists():
        return "no marks from before the re-mark: run with --judge first"
    was = json.loads(f.read_text(encoding="utf-8"))["models"]
    rows = ["| model | before | after | Summarise before | Summarise after | waiting |",
            "|---|---|---|---|---|---|"]
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        now = read(d)
        if not now or now.get("earlier"):
            continue
        b = was.get(now["model"]) or {}
        bg = b.get("groups") or {}
        old_sum = [bg[g] for g in ("shorten", "summarising") if g in bg]
        sb = (f"{sum(x['passed'] for x in old_sum)} of {sum(x['total'] for x in old_sum)}"
              if old_sum else "—")
        ng = (now.get("groups") or {}).get("summarising")
        rows.append(f"| {now['model']} | "
                    + (f"{b['passed']} of {b['total']}" if b else "—") + " | "
                    + f"{now['passed']} of {now['total']} | {sb} | "
                    + (f"{ng['passed']} of {ng['total']}" if ng else "—")
                    + f" | {now.get('waiting') or 0} |")
    causes = moved(results)
    return "\n".join(rows) + ("\n\n" + causes if causes else "")


def _steps(q: dict, it: dict) -> tuple | None:
    """12a.9: an answer's pass after each of this round's changes in turn, on
    the judge's one stored checklist — the checklist alone (12a.8's facts,
    "at least n" and lengths), then the lengths "as 4 bullets" states, then
    every fact counting where no length is set, with the facts a question
    gained: its mark now. None while it waits on the judge"""
    raw, c = it.get("judge_raw"), judge_check(q) or {}
    if not (c.get("checklist") and isinstance(raw, dict) and "checklist" in raw
            and isinstance(it.get("pass"), bool)):
        return None
    facts, _ = _facts_in(c["rubric"])
    old = [f for f in facts if _show(f) not in GAINED_12A9.get(q["id"], ())]
    need = min(int(c.get("at_least") or len(old)), len(old))
    ans = it.get("answer_text") or ""
    was = [checklist_verdict(raw, q, ans, {"facts": old, "need": need,
                                          "asked": bool(stated_length(q["prompt"], pattern))})
           for pattern in (STATED_LENGTH_12A8, STATED_LENGTH)]
    return None if None in was else (was[0]["pass"], was[1]["pass"], it["pass"])


def _reaches(q: dict) -> tuple[bool, bool]:
    """whether 12a.9's reading of lengths, and its every-fact rule, reach a question"""
    c = judge_check(q) or {}
    length = bool(stated_length(q["prompt"])) != bool(stated_length(q["prompt"],
                                                                     STATED_LENGTH_12A8))
    every = not stated_length(q["prompt"]) and bool(c.get("at_least") or GAINED_12A9.get(q["id"]))
    return length, every


def moved(results: Path) -> str:
    """12a.9: what moved Summarise, change by change — each column adds one of
    this round's changes to the column before it. Both halves; a hidden
    question is counted, never named"""
    f = results / BEFORE_NAME
    was = json.loads(f.read_text(encoding="utf-8")).get("summarise") if f.exists() else None
    if not was:
        return ""
    bank = {q["id"]: q for q in load_bank() if q["group"] == "summarising"}
    rows = ["Summarise, what moved it: each column adds one change to the column before",
            "",
            "| model | half | before | the checklist | + lengths “as 4 bullets” asks | "
            "+ every fact where no length is set (now) | waiting |",
            "|---|---|---|---|---|---|---|"]
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        now = read(d)
        if not now or now.get("earlier"):
            continue
        before = was.get(now["model"]) or {}
        for name, h in (("hidden", HIDDEN), ("practice", PRACTICE)):
            n, waiting, cols = 0, 0, [0, 0, 0, 0]
            for it in now["items"]:
                q = bank.get(it["id"])
                if q is None or half(q) != h or it["id"] not in before:
                    continue
                steps = _steps(q, it)
                if steps is None:
                    waiting += 1
                    continue
                n += 1
                for i, ok in enumerate((before[it["id"]], *steps)):
                    cols[i] += ok is True
            if n or waiting:
                rows.append(f"| {now['model']} | {name} | {cols[0]} of {n} | " + " | ".join(
                    f"{c}" + (f" ({c - p:+d})".replace("-", "−") if c != p else "")
                    for p, c in zip(cols, cols[1:])) + f" | {waiting} |")
    reach = {q["id"]: (_reaches(q), half(q)) for q in bank.values()}

    def which(k: int) -> str:
        seen = [i for i, (r, h) in reach.items() if r[k] and h == PRACTICE]
        hid = sum(1 for r, h in reach.values() if r[k] and h == HIDDEN)
        return (", ".join(i.replace("everyday-summarising-", "") for i in sorted(seen))
                or "none") + f" in the practice half; {hid} hidden"
    rows += ["", f"Lengths “as 4 bullets” asks reach: {which(0)}.",
             f"Every fact where no length is set reaches: {which(1)}."]
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("-m", "--model", action="append", default=[])
    ap.add_argument("--judge", action="store_true",
                    help="12a.6: also send the answers waiting on the judge (no model runs)")
    ap.add_argument("--compare", action="store_true",
                    help="12a.6: before and after the re-mark, model by model; 12a.9: and "
                         "what moved Summarise, change by change")
    ap.add_argument("-q", "--question", action="append", default=[],
                    help="12a.8: with --judge, send the judge only this question's answers "
                         "(a rubric or reference changed); every model is still marked")
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    unknown = sorted(set(a.question) - {q["id"] for q in load_bank()})
    if unknown:
        print(f"no such question: {', '.join(unknown)}", file=sys.stderr)
        return 2
    if a.compare:
        print(compare(a.results))
        return 0
    want = {m.replace("/", "__") for m in a.model}
    # 12a.8: one question's answers — no new "before" for --compare
    res = remark(a.results, want, judge=a.judge, only=set(a.question) or None,
                 snapshot=not a.question)
    for model, line in res["models"].items():
        print(f"{model}: {line}")
    print(f"marked {len(res['models'])} model(s)")
    if a.judge:
        print(f"sent the judge {res['sent']} answer(s)" + (
            f" · batch {res['batch_id']}: the service marks them as it lands; then "
            "--compare" if res["batch_id"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
