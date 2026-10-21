#!/usr/bin/env python3
"""21: a report's model-card cells, entered through the board's own card
entry (service/reported.card_add) — each with its card's link, the card's
date, its setting, and who entered it. The file (eval_tasks/reports/<id>.json)
names the card each number came from; check each against its card first.

A cell already entered (the same model, benchmark, value and link) is left as
it is: running it again changes nothing. Run inside the board's container:

    sudo docker compose exec -T bench python scripts/enter_report_cards.py step-checkpoints --by masein

Nothing here asks anyone outside; it never imports, runs or grades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def cells(rep: dict) -> list[dict]:
    """the file's card cells, each as the card entry takes it"""
    out = []
    for t in rep.get("theirs") or []:
        for b in rep.get("benchmarks") or []:
            c = t.get(b["key"])
            if isinstance(c, dict) and c.get("from") == "card":
                out.append({"label": t["label"], "model": c.get("model") or t["label"],
                            "maker": c.get("maker") or t.get("maker") or "",
                            "benchmark": c.get("benchmark") or b["label"],
                            "value": str(c.get("value")), "setting": c.get("setting") or "",
                            "url": c.get("url") or "", "date": c.get("date") or "",
                            # 25c: whose page it is on; a base model's kept for the report
                            "published_by": c.get("published_by") or t.get("published_by") or "",
                            "report_only": t.get("base_only") is True})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="the report's id: eval_tasks/reports/<id>.json")
    ap.add_argument("--by", required=True, help="who entered them, as the board records it")
    a = ap.parse_args(argv)
    from service import db, reported, reports
    db.init()
    rep = reports.files().get(a.report)
    if rep is None:
        print(f"no report {a.report!r} in {reports.REPORTS_DIR}")
        return 2
    have = {(r.get("key"), r.get("benchmark"), round(float(r.get("value") or 0), 6), r.get("url"))
            for r in db.reported_scores() if r.get("source") == "card"}
    entered = kept = 0
    for c in cells(rep):
        k = (reported.key(c["model"], c["maker"]), c["benchmark"], round(float(c["value"]) / 100, 6),
             c["url"])
        if k in have:
            kept += 1
            continue
        try:
            reported.card_add(c, a.by)
        except ValueError as e:
            print(f"{c['label']} · {c['benchmark']}: not entered — {e}")
            return 1
        entered += 1
        print(f"{c['label']} · {c['benchmark']}: {c['value']} from {c['url']} ({c['date']})")
    print(f"{entered} card number{'s' if entered != 1 else ''} entered by {a.by}, {kept} already "
          "there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
