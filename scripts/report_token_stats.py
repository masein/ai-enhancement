#!/usr/bin/env python3
"""21: a report's thinking-token stats as the board works them out
(service/reports.py, from each row's scored samples), one JSON line a row and
benchmark — and, given --against, each line set beside the same row and
benchmark of another file in that shape (masein's ~/tokens.py output):
"same", or the fields that differ. Read-only; it works the stats out where
none are kept yet, and keeps them as the page does.

    sudo docker compose exec -T bench python scripts/report_token_stats.py step-checkpoints --against - < ~/tokens-2026-10-14.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="the report's id: eval_tasks/reports/<id>.json")
    ap.add_argument("--against", default="", help="a JSON-lines file of rows to set them "
                    "beside (- for standard input)")
    a = ap.parse_args(argv)
    from service import reports
    rep = reports.files().get(a.report)
    if rep is None:
        print(f"no report {a.report!r} in {reports.REPORTS_DIR}")
        return 2
    other = {}
    if a.against:
        text = sys.stdin.read() if a.against == "-" else Path(a.against).read_text()
        for line in text.splitlines():
            if line.strip():
                r = json.loads(line)
                other[(r.get("row"), r.get("task"))] = r
    same = differ = 0
    for row in reports.our_rows(rep, {}):
        d = Path(row["dir"])
        for b in rep["benchmarks"]:
            got = reports.token_stats(d, b["task"])
            if got is None:
                continue
            mine = {"row": d.name, "task": b["task"], **got}
            if not a.against:
                print(json.dumps(mine))
                continue
            theirs = other.get((d.name, b["task"]))
            if theirs is None:
                print(f"{d.name} · {b['task']}: not in the other file")
                continue
            diff = {k: (mine.get(k), v) for k, v in theirs.items() if mine.get(k) != v}
            if diff:
                differ += 1
                print(f"{d.name} · {b['task']}: differs — " + ", ".join(
                    f"{k} {m} here, {t} there" for k, (m, t) in diff.items()))
            else:
                same += 1
                print(f"{d.name} · {b['task']}: same")
    if a.against:
        print(f"{same} the same, {differ} different")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
