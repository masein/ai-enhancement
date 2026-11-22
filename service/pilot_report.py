"""A pilot's report (eval_tasks/reports/<id>.json, kind "pilot") — benchmarks tried on
rented boxes outside the board, small fixed subsets, one try per item, to decide which
are worth a full run: the verdicts, the scores, each test's clean runs and cost, the
full-run estimate, the blind audit, and the boxes, time and money.

Internal: no row of the board's tables, nothing imported. Every number is the pilot's
own summary, a file under $BENCH_ROOT/pilot/ named by the report's "data" (written by
the pilot's board_summary.py from its results), read when the page asks, never worked
out here. Scores, counts and dollars only: no problem text, no box addresses. Nothing
here starts a run, an import or a grading batch."""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config

KIND = "pilot"
SCHEMA = 1


def path(rep: dict) -> Path | None:
    """the summary's file: under $BENCH_ROOT/pilot/, never elsewhere"""
    rel = str(rep.get("data") or "").strip()
    if not rel:
        return None
    base = (config.BENCH_ROOT / "pilot").resolve()
    p = (config.BENCH_ROOT / rel).resolve()
    return p if p.is_relative_to(base) and p != base else None


def _read(rep: dict) -> tuple[dict | None, float | None]:
    p = path(rep)
    if p is None:
        return None, None
    try:
        got = json.loads(p.read_text(encoding="utf-8"))
        at = p.stat().st_mtime
    except (OSError, ValueError):
        return None, None
    if not isinstance(got, dict) or got.get("schema") != SCHEMA:
        return None, None
    return got, at


def _kept(s: dict) -> tuple[int, int]:
    v = [x for x in s.get("verdicts") or [] if isinstance(x, dict)]
    return sum(x.get("verdict") == "keep" for x in v), len(v)


def listing_line(rep: dict) -> dict:
    """the card's lines: whose, how many tests kept, and when the summary was put here"""
    from .reports import when
    s, at = _read(rep)
    if s is None:
        return {"line": "nothing imported yet", "changed": "", "changed_at": None}
    models = [m for m in s.get("models") or [] if isinstance(m, dict)]
    ours = [m.get("label") or m.get("key") for m in models if m.get("ours")]
    theirs = [m.get("label") or m.get("key") for m in models if not m.get("ours")]
    kept, of = _kept(s)
    who = "step2388 " + " and ".join(ours) if ours else ""
    if theirs:
        who += (" against " if who else "") + ", ".join(theirs[:-1]) \
            + (" and " if len(theirs) > 1 else "") + theirs[-1]
    return {"line": f"{who} · {kept} of {of} tests kept" if who else f"{kept} of {of} tests kept",
            "changed": when(at) if at else "", "changed_at": at}


def report(rep: dict) -> dict:
    """the report's payload: the pilot's summary as kept, or none yet"""
    from .reports import when
    s, at = _read(rep)
    return {"id": rep["id"], "kind": KIND, "title": rep.get("title") or rep["id"],
            "what": rep.get("what") or "", "internal": True, "summary": s,
            "sources": {"ours": {"at": at, "words": f"the pilot's summary, put here {when(at)}"
                                 if at else "nothing imported yet"}},
            "read_at": time.time()}
