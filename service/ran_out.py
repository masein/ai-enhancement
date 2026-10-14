"""21b: a report of the answers our models ran out of room on — where the
loop happens, what it looks like, and whether the model already had the right
answer before it did (eval_tasks/reports/ran-out-of-room.json, kind "ran_out").

For insight only, never for training: these are benchmark questions and our
models' answers to them, and the page says so above every question. Nothing
here is offered to a training, generation or Improve path, and the page has no
export: one question is read at a time.

The hidden half: a report file that says `show_hidden: true` (masein's
decision of 14 Oct, for this report alone) lists every answer that ran out,
the hidden half's too. Any other report file is blind to it, as the rest of
the board is — the reader, the question generators, the diagnosis, the logs,
the question browser and the scorecard. The question text of a benchmark the
board never shows (fb.unlisted: GPQA Diamond, as its authors ask; HLE, as its
authors ask; MATH, its competitions' disputed copyright) stays withheld here
too, with its right answer, and any line of the thinking that quotes it.

Every number comes from the stored answers, worked out in a thread of their
own and kept until a file they read changes. Nothing here starts a run, an
import or a grading batch."""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from . import config

WARNING = ("For insight only — not for training. These are benchmark questions and our models' "
           "answers to them. Don't use them, or anything written from them, to train or "
           "fine-tune a model, or to make training data: the model would have seen the test, "
           "and its scores on these benchmarks would mean nothing. Use them to understand where "
           "and how the loop happens.")
WITHHELD_LINE = "[a line quoting the question, withheld]"
HOW_READ = ("Counted only where the thinking states it clearly: \"the answer is (E)\", the right "
            "option's letter with its text, or the exact final answer on MATH Level 5 and OTIS. "
            "The token is where that first came, as a share of the answer's characters times its "
            "output tokens. Not read on HLE, SimpleQA and ARC-AGI-2.")
LIST_Q = 200                    # a question's first characters in the list: two lines


def _scripts() -> None:
    p = str(Path(__file__).resolve().parent.parent / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# the halves, and what is withheld
# ---------------------------------------------------------------------------

def shows(rep: dict, task: str, qid: str) -> bool:
    """whether this report shows a question: the shown half always, the hidden
    half only where its file says show_hidden: true"""
    from . import reports
    return bool(rep.get("show_hidden") is True) or reports.shown_half(task, qid)


def withheld(task: str) -> str:
    """'' when the benchmark's questions may be shown; else why not"""
    _scripts()
    import frontier as fb
    return fb.unlisted(task)


def _quote_check(item: dict | None):
    """a function: does a line quote this question (its text, its options)?"""
    _scripts()
    import export_safe as es
    if not item:
        return lambda line: False
    texts = [str(item.get("question") or "")] + [str(o) for o in item.get("options") or []]
    texts += [str(item.get(k) or "") for k in ("right",) if item.get(k)]
    texts += [str(x) for x in item.get("wrong") or []]
    q = es.Questions([t for t in texts if t])
    return q.quotes


def scrub(text: str, item: dict | None) -> str:
    """the thinking, each line that quotes the question withheld"""
    check = _quote_check(item)
    return "\n".join(WITHHELD_LINE if line.strip() and check(line) else line
                     for line in text.split("\n"))


# ---------------------------------------------------------------------------
# what an answer says: its loop, and the right answer before the limit
# ---------------------------------------------------------------------------

def waits(text: str) -> int:
    return len(re.findall(r"\bWait\b", text))


def top_line(text: str) -> tuple[str, int]:
    """its most repeated line (eight characters or more), and how often"""
    lines = [x.strip() for x in text.splitlines() if len(x.strip()) >= 8]
    if not lines:
        return "", 0
    line, n = Counter(lines).most_common(1)[0]
    return line, n


def repeated(text: str, at_least: int = 3) -> list[str]:
    """the lines it wrote at least `at_least` times — marked when it is opened"""
    c = Counter(x.strip() for x in text.splitlines() if len(x.strip()) >= 8)
    return [x for x, n in c.most_common(40) if n >= at_least]


_LETTER = r"\(?\**\s*([A-J])\s*\**\)?(?![A-Za-z0-9])"
_SAYS = re.compile(r"(?i)\b(?:the\s+)?(?:final\s+|correct\s+|right\s+)?answer\s*"
                   r"(?:is|would be|should be|must be|:|=)\s*(?:option\s+)?" + _LETTER)


def _norm_math(s: str) -> str:
    s = re.sub(r"\\(?:left|right|displaystyle|!|,|;|:)", "", str(s))
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    return re.sub(r"[\s$]", "", s).rstrip(".")


def right_of(task: str, item: dict) -> dict:
    """what the right answer is, as the scorer has it: a letter and its
    option's text (GPQA's in its seeded order), or the final answer — read
    from the question, never by building its prompt"""
    _scripts()
    import frontier as fb
    if task == "gpqa_diamond_epoch":
        opts = fb.order_of(task, item)
        return {"letter": fb.LETTERS[opts.index(item["right"])], "text": str(item["right"])}
    if task == "mmlupro_tiger":
        letter = str(item.get("answer") or "")
        opts = item.get("options") or []
        i = fb.MMLU_PRO_LETTERS.find(letter) if letter else -1
        return {"letter": letter, "text": str(opts[i]) if 0 <= i < len(opts) else ""}
    return {"answer": str(item.get("answer") or "")}


def right_words(task: str, item: dict) -> str:
    """the right answer in words: "(E) 7 bits", or the final answer"""
    r = right_of(task, item)
    return f"({r['letter']}) {r['text']}" if "letter" in r else r["answer"]


def first_right(task: str, item: dict | None, text: str) -> int | None:
    """the first place the thinking clearly states the right answer: "the
    answer is (E)", the right option's letter with its text, or the exact
    final answer on MATH and OTIS. None when it never does — or on a
    benchmark this isn't read on"""
    if not item or not text:
        return None
    if task in ("mmlupro_tiger", "gpqa_diamond_epoch"):
        r = right_of(task, item)
        hits = [m.start() for m in _SAYS.finditer(text) if m.group(1) == r["letter"]]
        opt = re.sub(r"\s+", " ", r["text"]).strip()[:40]
        if len(opt) >= 3:
            pat = (r"\(?\b" + re.escape(r["letter"]) + r"\)?\s*[.:)\-–]?\s*"
                   + r"\s+".join(re.escape(w) for w in opt.split()))
            hits += [m.start() for m in re.finditer(pat, text, re.I)]
        return min(hits) if hits else None
    if task in ("math_l5_epoch", "otis_aime_epoch"):
        want = _norm_math(right_of(task, item)["answer"])
        if not want:
            return None
        hits = []
        for m in re.finditer(r"\\boxed\{", text):
            depth, j = 1, m.end()
            while j < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            if _norm_math(text[m.end():j - 1]) == want:
                hits.append(m.start())
        for m in re.finditer(r"(?i)\b(?:final\s+)?answer\s*(?:is|:|=)\s*\$?([^\n$]{1,60})", text):
            got = _norm_math(re.split(r"(?<=\S)[.,;](?:\s|$)", m.group(1).strip())[0])
            if got == want:
                hits.append(m.start())
        return min(hits) if hits else None
    return None


# ---------------------------------------------------------------------------
# the answers that ran out, a model's — kept until a file they read changes
# ---------------------------------------------------------------------------

def _runs(f: Path | None) -> dict[tuple[str, int], dict]:
    """a samples file's runs, by (question, run): ok, ran out, read, text"""
    out: dict = {}
    if f is None:
        return out
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        qid = str((row.get("doc") or {}).get("id") or row.get("doc_id"))
        resps = row.get("resps") or []
        for k, run in enumerate(row.get("frontier") or []):
            e = int(run.get("epoch", k) or 0)
            out[(qid, e)] = {**run, "answer": (resps[k][0] if k < len(resps) and resps[k] else "")
                             or "", "read": run.get("read")}
    return out


def _items(task: str) -> dict[str, dict]:
    _scripts()
    import frontier as fb
    return {str(it["id"]): it for it in (fb.cached(task, config.BENCH_ROOT) or [])}


def _subject(task: str, item: dict | None) -> str:
    if not item:
        return ""
    return str(item.get("category") or item.get("subject") or "")


def rows_of(rep: dict, label: str, task: str) -> dict:
    """the answers `label` ran out of room on in one benchmark, each beside
    the same question's run of the model it is compared with: {rows, answers,
    hidden}. Kept until a samples file it read changes"""
    from . import reports
    by = {o["label"]: o for o in rep["ours"]}
    me = samples(by[label]["row"], task)
    other = samples(by[rep["against"]]["row"], task) if rep.get("against") in by else None
    if me is None:
        return {"rows": [], "answers": 0, "hidden": 0}
    key = "|".join(["ranout", task, str(bool(rep.get("show_hidden") is True)),
                    *(reports._key(f, "") if f else "-" for f in (me, other))])
    got = reports._cache_get(key)
    if got is not None:
        return got
    _scripts()
    import frontier as fb
    items = _items(task)
    mine, theirs = _runs(me), _runs(other)
    gone = withheld(task)
    rows, hidden = [], 0
    for (qid, e), r in sorted(mine.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if not r.get("ran_out"):
            continue
        if not shows(rep, task, qid):
            hidden += 1
            continue
        it = items.get(qid)
        text = r["answer"]
        line, n = top_line(text)
        at = first_right(task, it, text)
        tok = r.get("tokens") if isinstance(r.get("tokens"), int) else None
        o = theirs.get((qid, e)) or theirs.get((qid, 0))
        right = None if gone or not it else right_words(task, it)
        rows.append({
            "task": task, "id": qid, "epoch": e, "subject": _subject(task, it),
            "half": "shown" if reports.shown_half(task, qid) else "hidden",
            "q": None if gone or not it else str(it.get("question") or "")[:LIST_Q],
            "withheld": gone, "right": right if right is None else str(right)[:LIST_Q],
            "chars": len(text), "tokens": tok, "waits": waits(text),
            "line": scrub(line, it)[:200] if gone else line[:200], "line_n": n,
            "had": at is not None if task in ("mmlupro_tiger", "gpqa_diamond_epoch",
                                              "math_l5_epoch", "otis_aime_epoch") else None,
            "at_token": int(round(tok * at / len(text))) if at is not None and tok and text
            else None,
            "against": None if o is None else {
                "ok": o.get("ok"), "ran_out": bool(o.get("ran_out")),
                "chars": len(o.get("answer") or ""), "verdict": fb.verdict(task, o)}})
    got = {"rows": rows, "answers": len(mine), "hidden": hidden}
    reports._cache_put(key, got)
    return got


def samples(mid: str, task: str) -> Path | None:
    from . import reports
    return reports.samples_file(reports.row_dir(mid), task)


def summary(rep: dict, label: str, per: dict[str, dict]) -> list[dict]:
    """the top of the report, a benchmark a line: how many ran out and their
    share; the other model on the same questions; had the right answer before
    the limit, and the median token it first came at; the loops' length"""
    out = []
    for b in rep["benchmarks"]:
        got = per.get(b["task"]) or {"rows": [], "answers": 0, "hidden": 0}
        rows = got["rows"]
        n = len(rows) + got["hidden"]
        vs = [r["against"] for r in rows if r.get("against")]
        had = [r for r in rows if r.get("had")]
        read = [r for r in rows if r.get("had") is not None]
        subjects = Counter(r["subject"] for r in rows if r["subject"])
        out.append({
            "key": b["key"], "task": b["task"], "label": b["label"], "n": n,
            "answers": got["answers"], "share": n / got["answers"] if got["answers"] else None,
            "hidden": got["hidden"], "listed": len(rows),
            "against": {"right": sum(1 for x in vs if x["ok"] is True and not x["ran_out"]),
                        "wrong": sum(1 for x in vs if x["ok"] is False and not x["ran_out"]),
                        "ran_out": sum(1 for x in vs if x["ran_out"]),
                        "of": len(vs)},
            "had": len(had) if read else None, "read": len(read),
            "had_token": int(statistics.median([r["at_token"] for r in had
                                                if r.get("at_token") is not None]))
            if any(r.get("at_token") is not None for r in had) else None,
            "waits": statistics.median([r["waits"] for r in rows]) if rows else None,
            "line_n": statistics.median([r["line_n"] for r in rows]) if rows else None,
            "subjects": [{"subject": s, "n": k} for s, k in subjects.most_common()]
            if b["task"] in ("mmlupro_tiger", "hle_text_cais") else []})
    return out


def report(rep: dict, compute: bool = False) -> dict:
    """the report: each picked model's summary and its list; worked out in a
    thread of its own unless `compute`"""
    from . import reports
    picks = [o["label"] for o in rep["ours"] if o.get("pick")]
    if not compute:
        jobs = [(lbl, b["task"]) for lbl in picks for b in rep["benchmarks"]]
        todo = [j for j in jobs if not _cached(rep, *j)]
        work = reports.ensure_jobs([lambda j=j: rows_of(rep, *j) for j in todo]) if todo else {}
    else:
        work = {}
    models = {}
    for lbl in picks:
        per = {b["task"]: (rows_of(rep, lbl, b["task"]) if compute else _cached(rep, lbl, b["task"]))
               for b in rep["benchmarks"]}
        models[lbl] = {"summary": summary(rep, lbl, {k: v for k, v in per.items() if v}),
                       "rows": [r for v in per.values() if v for r in v["rows"]],
                       "pending": [b["label"] for b in rep["benchmarks"] if not per[b["task"]]]}
    return {"id": rep["id"], "kind": "ran_out", "title": rep.get("title") or rep["id"],
            "what": rep.get("what") or "", "warning": WARNING, "howRead": HOW_READ,
            "insightOnly": True, "showHidden": rep.get("show_hidden") is True,
            "decided": rep.get("decided") or "", "picks": picks, "against": rep.get("against"),
            "benchmarks": rep["benchmarks"], "models": models, "working": work}


def _cached(rep: dict, label: str, task: str) -> dict | None:
    from . import reports
    by = {o["label"]: o for o in rep["ours"]}
    me = samples(by[label]["row"], task)
    if me is None:
        return {"rows": [], "answers": 0, "hidden": 0}
    other = samples(by[rep["against"]]["row"], task) if rep.get("against") in by else None
    key = "|".join(["ranout", task, str(bool(rep.get("show_hidden") is True)),
                    *(reports._key(f, "") if f else "-" for f in (me, other))])
    return reports._cache_get(key)


class Refused(PermissionError):
    """a question this report doesn't show"""


def one(rep: dict, label: str, task: str, qid: str, epoch: int) -> dict:
    """one answer, opened: the question and its right answer (withheld where
    the board withholds them), the other model's answer, and this one's
    thinking in full — its repeated lines, and where the right answer first
    came, marked"""
    _scripts()
    import frontier as fb
    by = {o["label"]: o for o in rep["ours"]}
    if label not in by or task not in {b["task"] for b in rep["benchmarks"]}:
        raise KeyError("no such model or benchmark in this report")
    if not shows(rep, task, qid):
        raise Refused(f"question {qid} is in the hidden half: this report doesn't show it")
    r = _runs(samples(by[label]["row"], task)).get((str(qid), int(epoch)))
    if r is None or not r.get("ran_out"):
        raise KeyError(f"{label} didn't run out of room on question {qid}, run {epoch}")
    it = _items(task).get(str(qid))
    gone = withheld(task)
    o = None
    if rep.get("against") in by:
        theirs = _runs(samples(by[rep["against"]]["row"], task))
        o = theirs.get((str(qid), int(epoch))) or theirs.get((str(qid), 0))
    text = r["answer"]
    at = first_right(task, it, text)
    shown = fb.shown(task, it) if it and not gone else None
    body = scrub(text, it) if gone else text
    right = None if gone or not it else right_of(task, it)
    return {
        "task": task, "id": str(qid), "epoch": int(epoch), "label": label, "warning": WARNING,
        "withheld": gone, "subject": _subject(task, it),
        "question": None if not shown else {"q": shown["q"], "options": shown.get("options") or [],
                                            "context": shown.get("context")},
        "right": right, "chars": len(text), "tokens": r.get("tokens"), "waits": waits(text),
        "thinking": body, "repeated": [scrub(x, it) if gone else x for x in repeated(text)],
        "first_right": at if not gone else None,
        "first_right_token": (int(round(r["tokens"] * at / len(text)))
                              if at is not None and isinstance(r.get("tokens"), int) else None),
        "had": at is not None,
        "against": None if o is None else {
            "label": rep["against"], "ok": o.get("ok"), "ran_out": bool(o.get("ran_out")),
            "chars": len(o.get("answer") or ""), "verdict": fb.verdict(task, o),
            "answer": (scrub(fb.visible(o.get("answer") or ""), it) if gone
                       else fb.visible(o.get("answer") or ""))[:4000]}}
