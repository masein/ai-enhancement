#!/usr/bin/env python3
"""21b: the top of a ran-out report, as text — per model and benchmark, how
many answers ran out of room (and the board's own count beside it), how the
compared model did on the same questions, and how often the right answer was
clearly stated before the limit. Counts only: no question, no answer. It works
the numbers out where none are kept yet, and keeps them as the page does.

    sudo docker compose exec -T bench python scripts/report_ran_out.py ran-out-of-room
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="the report's id: eval_tasks/reports/<id>.json")
    a = ap.parse_args(argv)
    from service import reports
    rep = reports.files().get(a.report)
    if rep is None or rep.get("kind") != "ran_out":
        print(f"no ran-out report {a.report!r} in {reports.REPORTS_DIR}")
        return 2
    got = reports.report(a.report, compute=True)
    vs = got["against"]
    for label, m in got["models"].items():
        print(f"{label} (the board's count in brackets) — {vs} on the same questions; the right "
              "answer stated before the limit")
        for x in m["summary"]:
            if not x["n"]:
                continue
            v = x["against"]
            had = ("not read" if x["had"] is None else f"{x['had']} of {x['read']}"
                   + (f", first at token {x['had_token']:,} (median)" if x["had_token"] else ""))
            print(f"  {x['label']}: {x['n']:,} ran out ({x['board']}) of {x['answers']:,}, "
                  f"{100 * (x['share'] or 0):.1f}% · {vs}: {v['right']} right, {v['wrong']} wrong, "
                  f"{v['ran_out']} also ran out · had it: {had}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
