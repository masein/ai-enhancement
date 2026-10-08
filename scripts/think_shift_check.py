#!/usr/bin/env python3
"""18c point 12: before the deploy — which stored Frontier scores the new
thinking split (18b: every thinking block counts, wherever it sits; one never
closed, wherever it opened, ran out) changes the next time each is scored (a
re-import, a grading batch landing, a grader switched at Start).

Read-only: it reads each row's answers, grades and stored score, scores them
again the new way in memory, and prints counts — per row and benchmark, never
an answer's text. Nothing is written. Run with the new code against the board's
files, before the board is deployed (docs/AGENT-RUNS.md § A, step 1):

    sudo docker compose build
    sudo docker compose run --rm --no-deps bench python scripts/think_shift_check.py

One line a row and benchmark whose reading changes:

    served__x__thinking · GPQA Diamond: 3 of 792 answers read differently (2 now ran out,
      1 right → wrong) · thinking in 790 → 792 · score 71.2% → 70.8%

and a last line: how many stored scores change."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def stored_runs(d: Path, task: str) -> dict:
    """{(question, epoch): run} as the last scoring kept them (samples_*.jsonl:
    each question's runs, without their text)"""
    out: dict = {}
    for p in sorted(d.glob(f"samples_{task}_*.jsonl"))[-1:]:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            qid = str((row.get("doc") or {}).get("id") or row.get("doc_id"))
            for k, r in enumerate(row.get("frontier") or []):
                out[(qid, int(r.get("epoch", k)))] = r
    return out


def check_task(row: Path, task: str) -> dict | None:
    """what changes for one row's benchmark; None when it has no stored score"""
    import frontier as fb
    from service import frontier as sf
    d = sf.task_dir(row, task)
    before = sf.stored_score(d, task)
    if before is None:
        return None
    old = stored_runs(d, task)
    m = sf.marks(row, task)                     # reads; writes nothing
    if m["missing"]:
        return {"skipped": "not every question answered"}
    runs = m["runs"]
    flat = [(q, r) for q, rs in runs.items() for r in rs]
    out = {"answers": len(flat), "ran_out": 0, "right_wrong": 0, "wrong_right": 0,
           "read": 0, "thinking_before": 0, "thinking_now": 0}
    for q, r in flat:
        o = old.get((str(q), int(r.get("epoch") or 0)))
        if o is None:
            continue
        diff = False
        if bool(r["ran_out"]) and not o.get("ran_out"):
            out["ran_out"] += 1
            diff = True
        if bool(o.get("ok")) and r["ok"] is False:
            out["right_wrong"] += 1
            diff = True
        elif not o.get("ok") and r["ok"] is True:
            out["wrong_right"] += 1
            diff = True
        out["read"] += diff
    answers = [r["answer"] for _, r in flat]
    out["thinking_now"] = sum(1 for a in answers if sf.thought(a))
    detail = sf._read_json(sorted(d.glob("results_*.json"))[-1]) if list(
        d.glob("results_*.json")) else {}
    held = (detail.get("frontier") or {}).get("thinking_held")
    out["thinking_before"] = held if isinstance(held, int) else None
    # the score the new way, as score_task would make it — in memory
    spec = fb.BENCH[task]
    o = sf.outcome(spec, [r for _, r in flat], d)
    if o["state"] in ("waiting", "no score"):
        out.update(score_before=before[0], score_now=None, state=o["state"])
        return out
    page = sf._share({q: [1.0 if r["ok"] else 0.0 for r in rs] for q, rs in runs.items()},
                     task, m["items"])
    out.update(score_before=before[0], score_now=page["score"], version=before[1])
    return out


def line(row: str, label: str, c: dict) -> str:
    from service import frontier as sf
    words = [f"{row} · {label}: {c['read']:,} of {c['answers']:,} answers read differently"]
    parts = [f"{c['ran_out']:,} now ran out" if c["ran_out"] else "",
             f"{c['right_wrong']:,} right → wrong" if c["right_wrong"] else "",
             f"{c['wrong_right']:,} wrong → right" if c["wrong_right"] else ""]
    parts = [x for x in parts if x]
    if parts:
        words[0] += f" ({', '.join(parts)})"
    if c.get("thinking_before") is not None and c["thinking_before"] != c["thinking_now"]:
        words.append(f"thinking in {c['thinking_before']:,} → {c['thinking_now']:,}")
    if c.get("state"):
        words.append(f"its score waits ({c['state']})")
    else:
        words.append(f"score {sf.score_words(c['score_before'])} → "
                     f"{sf.score_words(c['score_now'])}")
    return " · ".join(words)


def main(argv: list[str] | None = None) -> int:
    import frontier as fb
    from service import config
    from service import frontier as sf
    root = Path(config.OUT_DIR)
    seen = changed = 0
    for task, spec in fb.BENCH.items():
        for d in sorted(root.glob(f"*/{task}_0shot/{sf.SUB}")) if root.is_dir() else []:
            row = d.parent.parent
            try:
                c = check_task(row, task)
            except Exception as e:                      # noqa: BLE001 — one row's, said
                print(f"{row.name} · {spec['label']}: couldn't be read ({type(e).__name__})")
                continue
            if c is None:
                continue
            if c.get("skipped"):
                print(f"{row.name} · {spec['label']}: {c['skipped']}")
                continue
            seen += 1
            moved = (c.get("score_now") is not None and c.get("score_before") is not None
                     and abs(c["score_now"] - c["score_before"]) > 1e-9)
            if c["read"] or moved or c.get("state"):
                print(line(row.name, spec["label"], c))
            changed += moved
    print(f"{changed:,} of {seen:,} stored Frontier scores change the next time they are scored "
          f"({fb.VERSION}); nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
