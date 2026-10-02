#!/usr/bin/env python3
"""15.7: every DeviceMark row on the board scored again by today's reading of
a reply (devicemark.SCORING) from the answers it saved, and the table of what
changed. Nothing is asked of a model. Inside the container, after the deploy:

    sudo docker compose exec -T bench python scripts/rescore_devicemark.py --dry-run
    sudo docker compose exec -T bench python scripts/rescore_devicemark.py

A row whose answers can't be told apart from their thinking (Gemma 4 with the
thinking on, run before 15.7: its markers were dropped) keeps its numbers and
is marked provisional, with why: it needs a new run.

For each row it also says where the cap fell — how many replies it cut, and
how many of those inside the thinking — and, for a served row, how many of
the replies cut with an answer had their thinking split off by the server
(answers cut short, not thinking).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

TESTS = (("ifeval", "IFEval"), ("mmlu_pro", "MMLU-Pro"), ("math", "MATH"))


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def line(nums: dict | None) -> str:
    if not nums:
        return "—"
    return f"{pct(nums.get('composite'))} ({', '.join(pct(nums.get(k)) for k, _ in TESTS)})"


def served_cuts(row_dir: Path) -> str:
    """a served row's replies cut with an answer: how many the server had split
    the thinking off (answers cut short), and how many it hadn't"""
    import devicemark as dm
    cut = [r for r in dm.read_items(row_dir)
           if r.get("capped") and "answer" in r and (r.get("answer") or "").strip()]
    if not cut:
        return ""
    split = sum(1 for r in cut if r.get("thinking_chars"))
    return (f" · {len(cut)} cut with an answer: {split} after thinking the server split off"
            + (f", {len(cut) - split} with none split off (read them)" if len(cut) - split else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="the table only: nothing written")
    ap.add_argument("--row", action="append", help="one row's folder (repeatable); every row "
                                                   "when none is given")
    a = ap.parse_args(argv)
    import devicemark as dm
    from service import config
    from service import devicemark as sdm
    names = set(a.row or [])
    dirs = [d for d in sorted(config.OUT_DIR.iterdir()) if d.is_dir()
            and (d / dm.OUT_NAME).exists() and (not names or d.name in names)]
    print(f"scoring {dm.SCORING}: {dm.SCORING_WORDS[dm.SCORING]}"
          + (" — dry run, nothing written" if a.dry_run else ""))
    print("row · thinking | before: composite (IFEval, MMLU-Pro, MATH) | after | status")
    changed = 0
    for d in dirs:
        r = sdm.rescore(d, write=not a.dry_run)
        if r["status"] == "no row":
            continue
        moved = r.get("after") and r["after"] != r["before"]
        changed += bool(moved)
        print(f"{r['model'] or d.name} · thinking {'on' if r['thinking'] else 'off'} | "
              f"{line(r['before'])} | {line(r.get('after'))} | {r['status']}"
              + (f": {r['why']}" if r.get("why") else "") + ("" if moved or not r.get("after")
                                                             else " (no change)"))
        caps = r.get("caps") or {}
        if any((c or {}).get("capped") for c in caps.values()):
            print("    capped: " + ", ".join(
                f"{label} {caps[k]['capped']} ({caps[k]['capped_in_thinking']} inside the thinking)"
                for k, label in TESTS if caps.get(k)) + served_cuts(d))
    print(f"{len(dirs)} rows · {changed} changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
