"""25: instruction following, thinking on and off
(docs/prompts/phase-25-instruction-following-on-rented-boxes.md).

One row a build of the report's: IFEval strict and loose and IFBench loose —
the prompt-level shares their authors report, each prompt right when every
instruction of it was followed — with thinking on (the build's · thinking
row) and off (its plain row: plan I's second step), each with its standard
error over prompts. Above them one sentence from 23's question-by-question
verdicts, as the stats worker kept them: the latest checkpoint against the
report's reference, on each benchmark, thinking on and off. A request never
computes. DeviceMark's IFEval is another protocol and never in these
columns."""

from __future__ import annotations

COLUMNS = (("ifeval_google", "strict"), ("ifeval_google", "loose"), ("ifbench_test", "loose"))
THINKING = " · thinking"


def off_row(row: str) -> str | None:
    """the build's thinking-off row beside its thinking row"""
    return row.removesuffix(THINKING) if row.endswith(THINKING) else None


def cell(cells: dict, task: str, row: str | None, how: str) -> dict | None:
    """{v, se} of a row's prompt-level `how` on a benchmark, as scored"""
    if not row:
        return None
    f = (((cells.get(task) or {}).get(row) or {}).get("follow") or {}).get(f"prompt_{how}") or {}
    return {"v": f["v"], "se": f.get("se")} if isinstance(f.get("v"), (int, float)) else None


def said(x: dict | None, has: bool) -> str:
    """a kept pair in words: higher or lower by how many points, or can't tell"""
    from .training import WORDS
    if not has:
        return "not measured yet"
    if not x or not x.get("n"):
        return "not worked out yet"
    w = WORDS.get(x.get("verdict") or "")
    if w and x.get("d") is not None:
        return f"{w}, by {abs(100 * x['d']):.1f} points"
    return "can't tell"


def sentence(rep: dict, rows: list[dict], cells: dict, compute: bool = False) -> str:
    from . import paired, training
    prep = training._paired(rep["id"])
    # 27e: the baseline the viewer picked, else the report's reference
    ref = rep.get("baseline") or rep.get("reference")
    have = {r["label"]: r for r in rows}
    out = []
    for b in rep["benchmarks"]:
        t = b["task"]
        if t not in {c[0] for c in COLUMNS}:
            continue
        how = next(h for x, h in COLUMNS if x == t)
        # the latest checkpoint with a score on it, thinking on
        a = next((c["label"] for c in reversed(training.checkpoints(rep))
                  if cell(cells, t, (have.get(c["label"]) or {}).get("row"), how)), None)
        if not a or ref not in have or prep is None or a not in paired.labels(prep) \
                or ref not in paired.labels(prep):
            continue
        # 27e: worked out only where both builds' answers are stored (a baseline
        # that hasn't run it has none)
        def stored(off: bool) -> bool:
            return all(paired.samples(prep, x, t, off) is not None for x in (a, ref))
        has_on = bool(cell(cells, t, have[ref]["row"], how))
        on = said(paired.kept(prep, a, ref, t, compute) if has_on and stored(False) else None,
                  has_on)
        has_off = all(cell(cells, t, off_row(have[x]["row"]), how) for x in (a, ref))
        off = said(paired.kept(prep, a, ref, t, compute, off=True) if has_off and stored(True)
                   else None, has_off)
        out.append(f"{a} against {ref} on {b['label']}: {on}; with thinking off, {off}.")
    return " ".join(out)


def of_report(rep: dict, rows: list[dict], cells: dict, compute: bool = False) -> dict | None:
    """the section's data — None when the report asks neither benchmark"""
    benches = {b["task"]: b for b in rep["benchmarks"]}
    cols = [(t, h) for t, h in COLUMNS if t in benches]
    if not cols:
        return None
    return {"columns": [{"key": f"{t}:{h}", "task": t, "how": h,
                         "label": f"{benches[t]['label']} {h}"} for t, h in cols],
            "rows": [{"label": r["label"], "colour": r.get("colour"), "row": r["row"],
                      "rowOff": off_row(r["row"]),
                      "cells": {f"{t}:{h}": {"on": cell(cells, t, r["row"], h),
                                             "off": cell(cells, t, off_row(r["row"]), h)}
                                for t, h in cols}} for r in rows],
            "sentence": sentence(rep, rows, cells, compute)}
