"""12a.6, run once (2026-09-27): "Shorten a message" merged into "Summarise".

The fifteen short questions kept are the clearest: one short notice or
message, one plain summary. The other 48 are retired — the recipe (shortening
steps can drop a step), the two-topic AC message, and round 3's 46 longer
messages, each with many details and more than one fair summary. They leave
the bank for eval_tasks/everyday/retired.jsonl, with the date and why; answers
to them stay on disk and are no longer marked.

Every Summarise question is then marked by the judge on a rubric, 0 to 4,
passing at 3 (everyday.summarise_checks): one gate, no number the text doesn't
give, and the rubric built from the question's own key facts, its reference,
and a length only when the request states one.

    python docs/prompts/phase-12a6/merge_summarise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import everyday as ev  # noqa: E402

KEEP = ["everyday-pilot-03"] + [f"everyday-summarising-{n:02d}" for n in
                                (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
WHY = {"everyday-summarising-16": "a recipe: shortening its steps can drop a step, so more "
                                  "than one summary is fair",
       "everyday-summarising-06": "two topics in one message (the AC and the chiller fee)"}
ROUND3 = "round 3's longer message, with many details: more than one summary is fair"


def main() -> int:
    rows = [json.loads(line) for line in ev.BANK_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    kept, retired = [], []
    for q in rows:
        if q["group"] == "shorten" and q["id"] not in KEEP:
            retired.append({"retired": "2026-09-27", "why": WHY.get(q["id"], ROUND3), **q})
            continue
        if q["group"] in ("shorten", "summarising"):
            q = {**q, "group": "summarising", "checks": ev.summarise_checks(q)}
        kept.append(q)
    assert len([q for q in rows if q["group"] == "shorten"]) - len(retired) == len(KEEP)
    ev.BANK_PATH.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in kept),
                            encoding="utf-8")
    ev.RETIRED_PATH.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in retired),
                               encoding="utf-8")
    print(f"bank: {len(kept)} questions · retired: {len(retired)} · Summarise: "
          f"{sum(1 for q in kept if q['group'] == 'summarising')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
