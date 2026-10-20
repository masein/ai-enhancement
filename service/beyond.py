"""23b: beyond the score — how reliable a build is, how well calibrated, how
honest and how it thinks (docs/prompts/phase-23b-beyond-the-score.md). Four
measures from the answers and grades the board keeps, each row's benchmark
from its newest scored samples (`frontier[]` a run, with its `grade`), as 23
reads them:

- reliability (GPQA Diamond ×4, OTIS Mock AIME ×8): the share of questions
  right every time, never, and sometimes;
- calibration (HLE): CAIS's calibration error, ported below, over the
  confidence its judge reads from each answer; ten bins of stated confidence;
  the mean stated confidence beside the share right; how many replies stated
  none;
- SimpleQA Verified: right, wrong, not attempted and ran out; how often it
  tries, how often it is right when it does, and the SimpleQA paper's F;
- loops (every benchmark): the share of answers fb.ends_in_loop flags, by the
  board's loop rule, of all answers and of those that ran out of room.

Totals over both halves: no question is listed or quoted. Worked out by the
stats worker (service/report_stats.py) for every row with scored Frontier
answers, and kept; a page reads what is kept, never computing. Nothing here
starts a run, a grading or a paid call."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from . import config

GPQA, OTIS, HLE, SQA = ("gpqa_diamond_epoch", "otis_aime_epoch", "hle_text_cais",
                        "simpleqa_epoch")
RELIABLE = (GPQA, OTIS)
WAY = "beyond-1"
FEW = 20                    # a confidence bin with fewer answers is drawn faint
# HLE_SYSTEM asks each answer for a "Confidence: N%" line
_STATED = re.compile(r"(?im)^\W*confidence\W*:\s*\d{1,3}(?:\.\d+)?\s*%")


def _scripts() -> None:
    p = str(Path(__file__).resolve().parent.parent / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# CAIS's calibration error, as Scale's HLE board shows it — ported from
# https://github.com/centerforaisafety/hle/blob/a59048963171c5b9cfe89fbcb97ddde39b540897/hle_eval/run_judge_results.py
# (calib_err, L97–127, and its call in dump_metrics, L151), numpy taken out,
# its quirks kept:
# - the answers sorted by confidence and cut into bins of beta (100), the
#   last bin taking the rest;
# - the loop runs over every bin but the last (range(len(bins) - 1)): the
#   last bin is never counted, and with 100 to 199 answers the error is 0;
# - fewer than beta answers make no bins, and bins[-1] raises IndexError;
# - dump_metrics divides the judge's confidence by 100 and reports
#   100 * round(err, 2) — numpy's round, rint(x·100)/100, as Python's round.
# One difference, said: numpy's default argsort isn't stable, so where
# answers share a confidence CAIS's order inside the tie is its sort's (which
# varies with numpy and the machine); this one keeps them in question order,
# so the same answers always give the same number
# ---------------------------------------------------------------------------

def calib_err(confidence: list[float], correct: list[bool], p: str = "2",
              beta: int = 100) -> float:
    # beta is target bin size
    idxs = sorted(range(len(confidence)), key=lambda i: confidence[i])
    confidence = [confidence[i] for i in idxs]
    correct = [correct[i] for i in idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0.0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = abs(sum(bin_confidence) / num_examples_in_bin
                             - sum(1.0 if x else 0.0 for x in bin_correct) / num_examples_in_bin)

            if p == "2":
                cerr += num_examples_in_bin / total_examples * difference ** 2
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ("infty", "infinity", "max"):
                cerr = max(cerr, difference)
            else:
                raise AssertionError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = math.sqrt(cerr)

    return cerr


def calibration_error(confidence_pct: list[float], correct: list[bool]) -> float | None:
    """dump_metrics's number: 100 * round(calib_err(confidence / 100, correct,
    p='2', beta=100), 2) — None for fewer than 100 answers, where CAIS's
    raises"""
    if len(confidence_pct) < 100:
        return None
    err = calib_err([c / 100 for c in confidence_pct], correct, p="2", beta=100)
    return 100 * (round(err * 100) / 100)


# ---------------------------------------------------------------------------
# the measures, a benchmark's at a time
# ---------------------------------------------------------------------------

def _graded(run: dict) -> bool:
    return run.get("ok") is not None and not run.get("ungraded")


def reliability(rows: list[dict]) -> dict:
    """each question's runs right every time, never, sometimes; a question
    with a run not graded (or its grader gave none) left out, and counted"""
    always = never = sometimes = left = 0
    k_max = 0
    for row in rows:
        runs = row["runs"]
        if not runs or not all(_graded(r) for r in runs):
            left += 1
            continue
        k, r = len(runs), sum(1 for x in runs if x.get("ok") is True)
        k_max = max(k_max, k)
        if r == k:
            always += 1
        elif r == 0:
            never += 1
        else:
            sometimes += 1
    n = always + never + sometimes
    return {"n": n, "left_out": left, "runs": k_max, "always": always, "never": never,
            "sometimes": sometimes,
            **({"always_share": always / n, "never_share": never / n,
                "sometimes_share": sometimes / n} if n else {})}


def calibration(rows: list[dict]) -> dict:
    """HLE: CAIS's calibration error over the graded answers with a
    confidence from the judge; the ten bins; the mean stated confidence and
    the share right; how many replies stated no confidence"""
    _scripts()
    import frontier as fb
    conf, ok = [], []
    out = waiting = no_judge = unstated = 0
    for row in rows:
        for run, text in zip(row["runs"], row["texts"]):
            if run.get("ran_out"):
                out += 1                                    # never graded
                continue
            if not _graded(run):
                waiting += 1
                continue
            c = (run.get("grade") or {}).get("confidence")
            if not isinstance(c, (int, float)) or isinstance(c, bool):
                no_judge += 1                               # the judge read none
                continue
            conf.append(float(c))
            ok.append(run.get("ok") is True)
            if not _STATED.search(fb.visible(text)):
                unstated += 1
    n = len(conf)
    # ten bins of stated confidence: 0–9, 10–19, …, 90–100
    per: list[list[bool]] = [[] for _ in range(10)]
    for k, c in zip(ok, conf):
        per[max(0, min(9, int(c // 10)))].append(k)
    bins = [{"lo": 10 * b, "hi": 10 * b + 9 if b < 9 else 100, "n": len(v),
             "right": sum(v) / len(v) if v else None} for b, v in enumerate(per)]
    return {"n": n, "ran_out": out, "waiting": waiting, "no_judge_confidence": no_judge,
            "unstated": unstated, "error": calibration_error(conf, ok),
            "mean_confidence": sum(conf) / n if n else None,
            "right": sum(ok) / n if n else None, "bins": bins}


_LETTER = {"correct": "A", "incorrect": "B", "not attempted": "C"}


def simpleqa(rows: list[dict]) -> dict:
    """right (A), wrong (B), not attempted (C) by its grader; ran out of room
    a fourth group, never graded; tries, right when it tries, and F"""
    a = b = c = out = waiting = 0
    for row in rows:
        for run in row["runs"]:
            if run.get("ran_out"):
                out += 1
                continue
            g = run.get("grade") or {}
            letter = g.get("grade") or _LETTER.get(str(g.get("words") or ""))
            if not _graded(run) or letter not in ("A", "B", "C"):
                waiting += 1
                continue
            a += letter == "A"
            b += letter == "B"
            c += letter == "C"
    total = a + b + c + out
    right = a / total if total else None
    tries = (a + b) / total if total else None
    when = a / (a + b) if a + b else None
    f = (2 * right * when / (right + when)) if right and when else (0.0 if total else None)
    return {"right": a, "wrong": b, "not_attempted": c, "ran_out": out, "waiting": waiting,
            "total": total, "tries": tries, "right_share": right, "right_when_tries": when,
            "f": f}


def loops(rows: list[dict]) -> dict:
    """the answers ending in a loop by the board's rule (fb.ends_in_loop), of
    all and of those that ran out of room"""
    _scripts()
    import frontier as fb
    n = looped = out = out_looped = 0
    for row in rows:
        for run, text in zip(row["runs"], row["texts"]):
            n += 1
            lp = fb.ends_in_loop(text)
            looped += lp
            if run.get("ran_out"):
                out += 1
                out_looped += lp
    return {"answers": n, "looped": looped, "ran_out": out, "ran_out_looped": out_looped,
            "share": looped / n if n else None,
            "share_of_ran_out": out_looped / out if out else None}


def rows_of(f: Path) -> list[dict]:
    """a samples file, a question a row: its runs and their texts"""
    from . import reports
    out = []
    for _, row in reports.jsonl(f):
        runs = [r for r in row.get("frontier") or [] if isinstance(r, dict)]
        resps = row.get("resps") or []
        texts = [(resps[k][0] if k < len(resps) and resps[k] else "") or ""
                 for k in range(len(runs))]
        out.append({"runs": runs, "texts": texts})
    return out


def measures(f: Path, task: str) -> dict:
    """a row's benchmark, every measure that applies to it"""
    rows = rows_of(f)
    out = {"task": task, "loops": loops(rows)}
    if task in RELIABLE:
        out["reliability"] = reliability(rows)
    if task == HLE:
        out["calibration"] = calibration(rows)
    if task == SQA:
        out["simpleqa"] = simpleqa(rows)
    return out


# ---------------------------------------------------------------------------
# the worker's jobs, and what is kept
# ---------------------------------------------------------------------------

def _slot(d: Path, task: str) -> str:
    return f"beyond|{d}|{task}"


def _key(f: Path) -> str:
    from . import report_stats as rs
    return f"{WAY}|{rs.identity(f)}"


def row_dirs() -> list[Path]:
    """every row of the board holding scored Frontier answers"""
    _scripts()
    import frontier as fb
    root = Path(config.OUT_DIR)
    if not root.is_dir():
        return []
    return sorted({p.parents[2] for t in fb.TASKS
                   for p in root.glob(f"*/{t}_0shot/frontier/samples_{t}_*.jsonl")})


def jobs(dirs: list[Path] | None = None) -> list:
    _scripts()
    import frontier as fb

    from . import report_stats as rs
    from . import reports
    out = []
    for d in row_dirs() if dirs is None else dirs:
        for t in fb.TASKS:
            f = reports.samples_file(d, t)
            if f is not None:
                out.append(rs.Job(_slot(d, t), _key(f), f"beyond the score, {d.name} · "
                                  f"{fb.BENCH[t]['label']}", lambda f=f, t=t: measures(f, t)))
    return out


def kept(d: Path, task: str, compute: bool = False) -> dict | None:
    """a row's benchmark's measures as kept (worked out now with `compute`):
    None when never worked out; `fresh` False when its answers changed since"""
    from . import report_stats as rs
    from . import reports
    f = reports.samples_file(d, task)
    if f is None:
        return None
    e = rs.kept(_slot(d, task))
    if compute and (not e or e.get("key") != _key(f) or "value" not in e):
        e = rs.keep(_slot(d, task), _key(f), measures(f, task))
    if not e or "value" not in e:
        return None
    return {**e["value"], "fresh": e.get("key") == _key(f), "at": e.get("at")}


def of_row(mid: str, compute: bool = False) -> dict:
    """everything kept for one board row: {task: measures}, and how far the
    worker is with it"""
    _scripts()
    import frontier as fb

    from . import report_stats as rs
    from . import reports
    d = reports.row_dir(mid)
    got, ats = {}, []
    for t in fb.TASKS:
        x = kept(d, t, compute)
        if x:
            got[t] = x
            ats.append(x["at"])
    has = [t for t in fb.TASKS if reports.samples_file(d, t) is not None]
    return {"model": mid, "benches": got, "has": has,
            "labels": {t: fb.BENCH[t]["label"] for t in fb.TASKS},
            "working": {} if compute or not has else rs.working(jobs([d])),
            "statsAt": reports.stats_at(ats)}
