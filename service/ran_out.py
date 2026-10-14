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

Every number comes from the stored answers, worked out by the stats worker
(21c, service/report_stats.py) and kept: a page reads what is kept, and opens
one answer at the place in its file the worker noted. Nothing here starts a
run, an import or a grading batch."""

from __future__ import annotations

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

def _run(row: dict, k: int, run: dict) -> dict:
    resps = row.get("resps") or []
    return {**run, "answer": (resps[k][0] if k < len(resps) and resps[k] else "") or "",
            "read": run.get("read")}


def _qid(row: dict) -> str:
    return str((row.get("doc") or {}).get("id") or row.get("doc_id"))


def _items(task: str) -> dict[str, dict]:
    _scripts()
    import frontier as fb
    return {str(it["id"]): it for it in (fb.cached(task, config.BENCH_ROOT) or [])}


def _subject(task: str, item: dict | None) -> str:
    if not item:
        return ""
    return str(item.get("category") or item.get("subject") or "")


def _slot(rep: dict, label: str, task: str) -> str:
    # the flag is part of the slot: a list kept for a report that shows the
    # hidden half is never read for one that doesn't
    return "|".join(["ranout", str(rep.get("id")), "all" if rep.get("show_hidden") is True
                     else "shown", label, task])


def _files(rep: dict, label: str, task: str) -> tuple[Path | None, Path | None]:
    by = {o["label"]: o for o in rep["ours"]}
    me = samples(by[label]["row"], task)
    other = samples(by[rep["against"]]["row"], task) if rep.get("against") in by else None
    return me, other


def _key(me: Path | None, other: Path | None, task: str) -> str:
    """the files a list reads: the two models' answers and the questions"""
    from . import report_stats as rs
    from . import reports
    return "|".join([rs.identity(me), rs.identity(other),
                     rs.identity(reports.items_file(task))])


def jobs(rep: dict) -> list:
    """21c: a ran-out report's lists for the stats worker, a model's benchmark each"""
    from . import report_stats as rs
    out = []
    for lbl in [o["label"] for o in rep["ours"] if o.get("pick")]:
        for b in rep["benchmarks"]:
            me, other = _files(rep, lbl, b["task"])
            if me is not None:
                out.append(rs.Job(_slot(rep, lbl, b["task"]), _key(me, other, b["task"]),
                                  f"ran out, {lbl} · {b['label']}",
                                  lambda lbl=lbl, t=b["task"]: work_out(rep, lbl, t)))
    return out


def rows_of(rep: dict, label: str, task: str) -> dict:
    """the answers `label` ran out of room on in one benchmark, each beside
    the same question's run of the model it is compared with: {rows, answers,
    hidden} — worked out now unless kept for these very files"""
    from . import report_stats as rs
    me, other = _files(rep, label, task)
    if me is None:
        return {"rows": [], "answers": 0, "hidden": 0}
    key, slot = _key(me, other, task), _slot(rep, label, task)
    e = rs.kept(slot)
    if e and e.get("key") == key and "value" in e:
        return _shown(rep, e["value"])
    got = work_out(rep, label, task)
    rs.keep(slot, key, got)
    return _shown(rep, got)


def _shown(rep: dict, got: dict) -> dict:
    """belt and braces: a report without the flag never lists the hidden half"""
    if rep.get("show_hidden") is True:
        return got
    return {**got, "rows": [r for r in got["rows"] if r.get("half") != "hidden"]}


def work_out(rep: dict, label: str, task: str) -> dict:
    """rows_of, worked out: each samples file read a row at a time — the
    answer's text kept only while its line is read — and where each answer's
    line starts, so one is opened without reading the file again"""
    from . import reports
    _scripts()
    import frontier as fb
    me, other = _files(rep, label, task)
    if me is None:
        return {"rows": [], "answers": 0, "hidden": 0}
    items = _items(task)
    gone = withheld(task)
    seen: dict[tuple[str, int], object] = {}
    for off, row in reports.jsonl(me):
        qid = _qid(row)
        for k, run in enumerate(row.get("frontier") or []):
            e = int(run.get("epoch", k) or 0)
            if not run.get("ran_out"):
                seen[(qid, e)] = None
                continue
            if not shows(rep, task, qid):
                seen[(qid, e)] = "hidden"
                continue
            r = _run(row, k, run)
            it = items.get(qid)
            text = r["answer"]
            line, n = top_line(text)
            at = first_right(task, it, text)
            tok = r.get("tokens") if isinstance(r.get("tokens"), int) else None
            right = None if gone or not it else right_words(task, it)
            seen[(qid, e)] = {
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
                "against": None, "_at": [off, k]}
    rows = [v for v in seen.values() if isinstance(v, dict)]
    want = {r["id"] for r in rows}
    theirs: dict[tuple[str, int], dict] = {}
    if other is not None and want:
        for off, row in reports.jsonl(other):
            qid = _qid(row)
            if qid not in want:
                continue
            for k, run in enumerate(row.get("frontier") or []):
                o = _run(row, k, run)
                theirs[(qid, int(run.get("epoch", k) or 0))] = {
                    "ok": o.get("ok"), "ran_out": bool(o.get("ran_out")),
                    "chars": len(o.get("answer") or ""), "verdict": fb.verdict(task, o),
                    "_at": [off, k]}
    for r in rows:
        r["against"] = theirs.get((r["id"], r["epoch"])) or theirs.get((r["id"], 0))
    rows.sort(key=lambda r: (r["id"], r["epoch"]))
    return {"rows": rows, "answers": len(seen),
            "hidden": sum(1 for v in seen.values() if v == "hidden")}


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


def _public(r: dict) -> dict:
    """a row as the page gets it: without where its answers sit in their files"""
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    if isinstance(out.get("against"), dict):
        out["against"] = {k: v for k, v in out["against"].items() if not k.startswith("_")}
    return out


def report(rep: dict, compute: bool = False) -> dict:
    """the report: each picked model's summary and its list — worked out now
    with `compute`, else as the stats worker kept them"""
    from . import report_stats as rs
    from . import reports
    picks = [o["label"] for o in rep["ours"] if o.get("pick")]
    models, ats = {}, []
    for lbl in picks:
        per = {}
        for b in rep["benchmarks"]:
            if compute:
                rows_of(rep, lbl, b["task"])
            per[b["task"]] = _cached(rep, lbl, b["task"])
            e = rs.kept(_slot(rep, lbl, b["task"]))
            if e and "value" in e:
                ats.append(e["at"])
        models[lbl] = {"summary": summary(rep, lbl, {k: v for k, v in per.items() if v}),
                       "rows": [_public(r) for v in per.values() if v for r in v["rows"]],
                       "pending": [b["label"] for b in rep["benchmarks"] if not per[b["task"]]]}
    return {"id": rep["id"], "kind": "ran_out", "title": rep.get("title") or rep["id"],
            "what": rep.get("what") or "", "warning": WARNING, "howRead": HOW_READ,
            "insightOnly": True, "showHidden": rep.get("show_hidden") is True,
            "decided": rep.get("decided") or "", "picks": picks, "against": rep.get("against"),
            "benchmarks": rep["benchmarks"], "models": models,
            "working": {} if compute else rs.working(jobs(rep)),
            "statsAt": reports.stats_at(ats)}


def _cached(rep: dict, label: str, task: str) -> dict | None:
    """the list as kept — shown until the new one is in when the answers
    changed since; None when never worked out"""
    from . import report_stats as rs
    me, _ = _files(rep, label, task)
    if me is None:
        return {"rows": [], "answers": 0, "hidden": 0}
    e = rs.kept(_slot(rep, label, task))
    return _shown(rep, e["value"]) if e and "value" in e else None


def _find(f: Path | None, qid: str, epoch: int, at: list | None) -> dict | None:
    """one run of a question: at the place the worker noted when the file is
    the one it read, else looked for, the lines holding the question's id
    alone parsed"""
    from . import reports
    if f is None:
        return None
    if at:
        row = reports.row_at(f, int(at[0]))
        if row is not None and _qid(row) == str(qid):
            runs = row.get("frontier") or []
            k = int(at[1])
            if k < len(runs):
                return _run(row, k, runs[k])
    for _, row in reports.jsonl(f, f'"{qid}"'.encode()):
        if _qid(row) != str(qid):
            continue
        for k, run in enumerate(row.get("frontier") or []):
            if int(run.get("epoch", k) or 0) == int(epoch):
                return _run(row, k, run)
    return None


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
    from . import report_stats as rs
    me, other = _files(rep, label, task)
    # 21c: where the worker found it, when these are the files it read
    e = rs.kept(_slot(rep, label, task))
    fresh = e and e.get("key") == _key(me, other, task) and "value" in e
    hit = next((x for x in e["value"]["rows"] if x["id"] == str(qid)
                and x["epoch"] == int(epoch)), None) if fresh else None
    r = _find(me, str(qid), int(epoch), (hit or {}).get("_at"))
    if r is None or not r.get("ran_out"):
        raise KeyError(f"{label} didn't run out of room on question {qid}, run {epoch}")
    it = _items(task).get(str(qid))
    gone = withheld(task)
    o = None
    if rep.get("against") in by:
        o = _find(other, str(qid), int(epoch), ((hit or {}).get("against") or {}).get("_at"))
        if o is None and int(epoch):
            o = _find(other, str(qid), 0, None)
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
