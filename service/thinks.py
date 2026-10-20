"""23d: how it thinks — when a build first states the right answer, how much
it thinks after that, and how often it had the right answer and then gave a
wrong one (docs/prompts/phase-23d-how-it-thinks.md).

For each row and each of GPQA Diamond, MMLU-Pro, MATH Level 5 and OTIS Mock
AIME (the four the ran-out report's first_right reads), from the row's
newest scored samples and this machine's question cache:

- where the thinking first states the right answer (ran_out.first_right, on
  the thinking alone — think_split's blocks, the tags left out), as a share
  of the thinking's characters;
- the tokens after it: the answer's output tokens × the characters after that
  point (the rest of the thinking and the reply) over all its characters —
  the token stats' character-share rule;
- right answers that didn't run out: the share stating it in the thinking;
  of those, the median position, the median tokens before and after it, and
  the checking share (tokens after it over all its tokens), averaged;
- wrong answers that didn't run out: the share whose thinking had stated it —
  "had it, then lost it";
- answers that ran out: the share that stated it before the limit, read on
  the whole text as the ran-out report reads it (the same number);
- 23e: and the answers behind those two — had it, then lost it, and ran out
  with it stated — each listed (its question, run, where it first stated
  the answer, what it answered in the end) in the shown half only, the
  hidden half counted (reports.shown_half, the reader's rule);
- second thoughts: SECOND's words at the start of a sentence in the
  thinking, each and in all, per 1,000 thinking tokens (the token stats'
  estimate), right answers and wrong ones apart.

Totals over both halves; no question, answer or excerpt is kept. A question
not in this machine's cache, and a run not graded, are left out and counted.
Worked out by the stats worker (service/report_stats.py) and kept; a page
reads what is kept, never computing. Nothing here starts a run or a paid
call."""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

TASKS = ("gpqa_diamond_epoch", "mmlupro_tiger", "math_l5_epoch", "otis_aime_epoch")
WAY = "thinks-2"                # 23e: and the answers behind two of its numbers
# "second thoughts": these words at the start of a sentence — named on the page
SECOND = ("Wait", "Alternatively", "Hmm", "Actually")
# a sentence's start: the text's or a line's, or after . ! or ? — then any
# spaces and markup before the word ("**Wait**", "- Hmm"), the word whole and
# as written (capitalised: mid-sentence it is lower case)
_SECOND = re.compile(r"(?:^|(?<=[.!?]))[ \t]*[*_>\"'(\-]*[ \t]*(" + "|".join(SECOND) + r")\b",
                     re.M)


def _scripts() -> None:
    p = str(Path(__file__).resolve().parent.parent / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


def second_thoughts(text: str) -> dict[str, int]:
    """each word of SECOND at a sentence's start in `text`, counted"""
    out = dict.fromkeys(SECOND, 0)
    for m in _SECOND.finditer(text or ""):
        out[m.group(1)] += 1
    return out


def split(text: str) -> tuple[str, str]:
    """(the thinking, the reply) — every thinking block joined, the tags left
    out, as the token stats count them"""
    _scripts()
    from frontier_graders import think_split
    outside, thoughts, _ = think_split(text or "")
    return "".join(thoughts), outside


def one(task: str, item: dict | None, text: str, tokens) -> dict:
    """one answer: where its thinking first states the right answer (a share
    of the thinking's characters), the tokens before and after that point,
    its thinking tokens, and its second thoughts"""
    from . import ran_out
    thinking, reply = split(text)
    th, an = len(thinking), len(reply)
    n = tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None
    at = ran_out.first_right(task, item, thinking) if thinking else None
    out = {"stated": at is not None, "think_tokens": round(n * th / (th + an)) if n is not None
           and th + an else (0 if n is not None else None),
           "second": second_thoughts(thinking)}
    if at is not None:
        after = (th - at + an) / (th + an)
        out.update(pos=at / th, after_share=after,
                   after=round(n * after) if n is not None else None,
                   before=n - round(n * after) if n is not None else None)
    return out


def _med(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def _per_k(rows: list[dict]) -> dict:
    """second thoughts per 1,000 thinking tokens, each word and in all"""
    toks = sum(r["think_tokens"] for r in rows if isinstance(r.get("think_tokens"), int))
    counts = {w: sum(r["second"][w] for r in rows if isinstance(r.get("think_tokens"), int))
              for w in SECOND}
    return {"think_tokens": toks, "answers": len(rows), "counts": counts,
            **({"per_1k": {w: 1000 * c / toks for w, c in counts.items()},
                "all_per_1k": 1000 * sum(counts.values()) / toks} if toks else {})}


def measures(f: Path, task: str) -> dict:
    """a row's benchmark, every answer of its samples file read once"""
    from . import ran_out, reports
    items = ran_out._items(task)
    right, wrong, cut = [], [], []
    answers = ungraded = 0
    # 23e: the answers behind "had it, then lost it" and "ran out with it
    # stated" — listed in the shown half, counted in the hidden
    lists: dict[str, dict] = {k: {"items": [], "hidden": 0} for k in ("lost", "ran_out")}

    def listed(which: str, qid: str, epoch: int, run: dict, text: str) -> None:
        if not reports.shown_half(task, qid):
            lists[which]["hidden"] += 1
            return
        thinking, _ = split(text)
        at = ran_out.first_right(task, it, thinking) if thinking else None
        lists[which]["items"].append({
            "id": qid, "epoch": epoch, "ran_out": bool(run.get("ran_out")),
            "pos": at / len(thinking) if at is not None else None,
            "read": None if run.get("read") is None else str(run["read"])[:60]})
    missing: set[str] = set()
    missing_runs = 0
    for _, row in reports.jsonl(f):
        qid = ran_out._qid(row)
        resps = row.get("resps") or []
        it = items.get(qid)
        for k, run in enumerate(row.get("frontier") or []):
            if not isinstance(run, dict):
                continue
            answers += 1
            if it is None:
                missing.add(qid)
                missing_runs += 1
                continue
            text = (resps[k][0] if k < len(resps) and resps[k] else "") or ""
            e = int(run.get("epoch", k) or 0)
            if run.get("ran_out"):
                # as the ran-out report reads it: the whole text
                cut.append(ran_out.first_right(task, it, text) is not None)
                if cut[-1]:
                    listed("ran_out", qid, e, run, text)
                continue
            ok = run.get("ok")
            if ok is None or run.get("ungraded"):
                ungraded += 1
                continue
            (right if ok is True else wrong).append(one(task, it, text, run.get("tokens")))
            if ok is not True and wrong[-1]["stated"]:
                listed("lost", qid, e, run, text)
    st = [r for r in right if r["stated"]]
    lost = [r for r in wrong if r["stated"]]
    return {
        "task": task, "answers": answers, "ungraded": ungraded,
        "no_item": len(missing), "no_item_answers": missing_runs,
        "right": {"n": len(right), "stated": len(st),
                  "stated_share": len(st) / len(right) if right else None,
                  "pos_median": _med([r["pos"] for r in st]),
                  "after_median": _med([r["after"] for r in st]),
                  "before_median": _med([r["before"] for r in st]),
                  "checking": statistics.fmean([r["after_share"] for r in st]) if st else None},
        "wrong": {"n": len(wrong), "lost": len(lost),
                  "lost_share": len(lost) / len(wrong) if wrong else None},
        "ran_out": {"n": len(cut), "stated": sum(cut),
                    "stated_share": sum(cut) / len(cut) if cut else None},
        "second": {"right": _per_k(right), "wrong": _per_k(wrong)},
        "lists": lists}


# ---------------------------------------------------------------------------
# the worker's jobs, and what is kept
# ---------------------------------------------------------------------------

def _slot(d: Path, task: str) -> str:
    return f"thinks|{d}|{task}"


def _key(f: Path, task: str) -> str:
    """the files it reads: the answers and the questions"""
    from . import report_stats as rs
    from . import reports
    return f"{WAY}|{rs.identity(f)}|{rs.identity(reports.items_file(task))}"


def jobs(dirs: list[Path] | None = None) -> list:
    from . import beyond, reports
    from . import report_stats as rs
    out = []
    for d in beyond.row_dirs() if dirs is None else dirs:
        for t in TASKS:
            f = reports.samples_file(d, t)
            if f is not None:
                out.append(rs.Job(_slot(d, t), _key(f, t), f"how it thinks, {d.name} · {t}",
                                  lambda f=f, t=t: measures(f, t)))
    return out


def kept(d: Path, task: str, compute: bool = False) -> dict | None:
    """a row's benchmark's measures as kept (worked out now with `compute`):
    None when never worked out; `fresh` False when its files changed since"""
    from . import report_stats as rs
    from . import reports
    f = reports.samples_file(d, task)
    if f is None:
        return None
    key = _key(f, task)
    e = rs.kept(_slot(d, task))
    if compute and (not e or e.get("key") != key or "value" not in e):
        e = rs.keep(_slot(d, task), key, measures(f, task))
    if not e or "value" not in e:
        return None
    return {**e["value"], "fresh": e.get("key") == key, "at": e.get("at")}


def of_row(d: Path, compute: bool = False) -> dict:
    """{task: measures} for a row, the four benchmarks it has kept"""
    return {t: x for t in TASKS if (x := kept(d, t, compute))}
