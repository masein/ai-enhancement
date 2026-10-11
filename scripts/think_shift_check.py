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

18d point 6: one line for every row and benchmark with a stored score — what
it holds and what scoring will write (score_task's own reading: the code's
score while answers wait for Epoch's model check, none while they wait for a
grader, a refusal when more than a quarter of an off row's answers think),
how many answers read differently, an off row's thinking share, and how many
answers wait for Epoch's model check, paid at the next Start:

    served__x__thinking · MATH Level 5: stored 50.0% → 25.0% · 12 answers wait for
      Epoch's model check, paid at the next Start

and a last line: how many stored scores change, of how many (close to the
count of results files the deploy step saved). A folder with no Frontier
results exits 1, naming it."""

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
    """what changes for one row's benchmark; None when it has no stored
    score. 18d point 6: the score as scoring will write it (service/frontier
    .score_plan, score_task's own reading) — the code's while answers wait for
    Epoch's model check, none while they wait for a grader, a refusal when
    too many of an off row's answers think"""
    from service import frontier as sf
    d = sf.task_dir(row, task)
    before = sf.stored_score(d, task)
    if before is None:
        return None
    old = stored_runs(d, task)
    m = sf.marks(row, task)                     # reads; writes nothing
    if m["missing"]:
        return {"skipped": "not every question answered", "score_before": before[0]}
    plan = sf.score_plan(row, task, m)          # reads; writes nothing
    runs = m["runs"]
    flat = [(q, r) for q, rs in runs.items() for r in rs]
    out = {"answers": len(flat), "ran_out": 0, "right_wrong": 0, "wrong_right": 0,
           "read": 0, "look": plan["look"]}
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
    results = sorted(d.glob("results_*.json"))
    fr = (sf._read_json(results[-1]) if results else {}).get("frontier") or {}
    # 18d point 6: an off row's answers that think, before and now — 0 before
    # when its results said none (only a count above 0 was kept)
    if plan["thinking"] == "off":
        held = fr.get("thinking_held")
        out["thinking_before"] = held if isinstance(held, int) else 0
        out["thinking_now"] = sum(1 for _, r in flat if sf.thought(r["answer"]))
    out.update(score_before=before[0],
               score_now=plan["page"]["score"] if plan["page"] else None, version=before[1],
               state=plan["state"], why=plan["why"])
    return out


def moved(c: dict) -> bool:
    """the score the page shows changes: another number, or none where one was"""
    if c.get("skipped"):
        return False
    if c["state"] in ("refused", "waiting", "no score"):
        return c.get("score_before") is not None
    return c.get("score_now") is not None and c.get("score_before") is not None \
        and abs(c["score_now"] - c["score_before"]) > 1e-9


def line(row: str, label: str, c: dict) -> str:
    """18d point 6: every row with a stored score — what it holds and what
    scoring will write, whether it moves or not"""
    from service import frontier as sf
    words = [f"{row} · {label}: stored {sf.score_words(c['score_before'])}"]
    if c.get("skipped"):
        return words[0] + f" · {c['skipped']}: not scored again yet"
    if c["state"] == "refused":
        words.append(f"would be refused — {c['why']}")
    elif c["state"] in ("waiting", "no score"):
        words.append(f"no score written: {c['why']}")
    elif moved(c):
        words[0] += f" → {sf.score_words(c['score_now'])}"
    else:
        words.append("stays")
    if c["read"]:
        parts = [f"{c['ran_out']:,} now ran out" if c["ran_out"] else "",
                 f"{c['right_wrong']:,} right → wrong" if c["right_wrong"] else "",
                 f"{c['wrong_right']:,} wrong → right" if c["wrong_right"] else ""]
        parts = [x for x in parts if x]
        words.append(f"{c['read']:,} of {c['answers']:,} answers read differently"
                     + (f" ({', '.join(parts)})" if parts else ""))
    if "thinking_before" in c and c["thinking_before"] != c["thinking_now"]:
        words.append(f"thinking in {c['thinking_before']:,} → {c['thinking_now']:,} of "
                     f"{c['answers']:,}")
    if c.get("look"):
        words.append(f"{c['look']:,} answers wait for Epoch's model check, paid at the next "
                     "Start")
    return " · ".join(words)


def main(argv: list[str] | None = None) -> int:
    import frontier as fb
    from service import config
    from service import frontier as sf
    root = Path(config.OUT_DIR)
    # 18d point 6: a folder that isn't there, or holds no Frontier score, is
    # a check of nothing — said, and not a pass
    found = sorted(root.glob(f"*/*_0shot/{sf.SUB}")) if root.is_dir() else []
    if not found:
        print(f"{root}: no Frontier results here — is this the board's results folder "
              "(OUT_ROOT)? Nothing was checked")
        return 1
    seen = changed = look = 0
    for task, spec in fb.BENCH.items():
        for d in sorted(root.glob(f"*/{task}_0shot/{sf.SUB}")):
            row = d.parent.parent
            try:
                c = check_task(row, task)
            except Exception as e:                      # noqa: BLE001 — one row's, said
                print(f"{row.name} · {spec['label']}: couldn't be read ({type(e).__name__})")
                continue
            if c is None:
                continue
            seen += 1
            changed += moved(c)
            look += c.get("look") or 0
            print(line(row.name, spec["label"], c))
    print(f"{changed:,} of {seen:,} stored Frontier scores change the next time they are scored "
          f"({fb.VERSION}); {look:,} answers wait for Epoch's model check, paid at the next "
          "Start; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
