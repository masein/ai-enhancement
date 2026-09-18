#!/usr/bin/env python3
"""Calibrate the judge against a person, or nothing judged counts.

    python scripts/judge_calibrate.py export results/full --out cal.csv [-m model …] [--n 100]
    #   → a person fills in the human_score column (0–4) with the rubric beside each row
    python scripts/judge_calibrate.py import results/full cal.csv

export samples judged answers stratified by category and by the judge's
score, writes them to a CSV with the rubric attached and the judge's score
HIDDEN — the grader must not see it. import re-joins the rows to judge.json
by id, computes Cohen's κ per category and overall, and writes
results/full/judge_calibration.json, which the dashboard shows next to every
judged number. Below κ 0.60 the suite is preliminary: shown, never ranked,
never in an average. That is the same mechanism a model missing required
tasks gets, applied to the judge.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import rubric_for  # noqa: E402

KAPPA_MIN = 0.60
CALIBRATION_FILE = "judge_calibration.json"
FIELDS = ["id", "model", "task", "category", "prompt", "reference", "answer", "answer_words",
          "rubric", "human_score"]


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    """Unweighted Cohen's κ between two raters over the same items."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def _judged_rows(results: Path, models: set[str]) -> list[dict]:
    rows = []
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        if models and d.name not in models:
            continue
        jf = d / "judge.json"
        if not jf.exists():
            continue
        j = json.loads(jf.read_text(encoding="utf-8"))
        if j.get("skipped"):
            continue
        answers = _answers(d)
        for task, t in j.get("tasks", {}).items():
            for it in t.get("items", []):
                key = (task, it.get("doc_hash"))
                a = answers.get(key)
                if not a:
                    continue
                rows.append({"id": f"{d.name}|{task}|{it['doc_hash']}", "model": d.name,
                             "task": task, "category": it.get("category") or task[3:],
                             "prompt": a["prompt"], "reference": a["reference"],
                             "answer": a["answer"], "answer_words": it["answer_words"],
                             "judge_score": it["score"]})
    return rows


def _answers(model_dir: Path) -> dict[tuple, dict]:
    from judge import _answer, _records
    from exam_build import ALL_TASKS
    out = {}
    for task in ALL_TASKS:
        for rec in _records(model_dir, task):
            doc = rec.get("doc") or {}
            out[(task, rec.get("doc_hash"))] = {"prompt": doc.get("prompt", ""),
                                                "reference": doc.get("reference", ""),
                                                "answer": _answer(rec)}
    return out


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over (category, judge score) strata so every score level
    and every category is represented, whatever the judge's distribution."""
    strata: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        strata[(r["category"], r["judge_score"])].append(r)
    rng = random.Random(seed)
    for v in strata.values():
        v.sort(key=lambda r: r["id"])
        rng.shuffle(v)
    keys = sorted(strata)
    out: list[dict] = []
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out


def export(results: Path, out: Path, models: list[str], n: int, seed: int) -> int:
    rows = _judged_rows(results, {m.replace("/", "__") for m in models})
    picked = sample(rows, n, seed)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in picked:
            w.writerow({**{k: r[k] for k in FIELDS if k in r},
                        "rubric": rubric_for(r["task"])[0], "human_score": ""})
    return len(picked)


def import_csv(results: Path, csv_path: Path) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        human = {r["id"]: r for r in csv.DictReader(fh)}
    judged = {r["id"]: r for r in _judged_rows(results, set())}
    pairs: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    skipped = 0
    for rid, r in human.items():
        hs = (r.get("human_score") or "").strip()
        j = judged.get(rid)
        if not hs.isdigit() or j is None or not 0 <= int(hs) <= 4:
            skipped += 1
            continue
        pairs[j["category"]].append((j["judge_score"], int(hs)))
    per_cat = {c: {"n": len(v), "kappa": cohen_kappa([x for x, _ in v], [y for _, y in v]),
                   "agreement": round(sum(1 for x, y in v if x == y) / len(v), 4)}
               for c, v in sorted(pairs.items())}
    allp = [p for v in pairs.values() for p in v]
    overall = cohen_kappa([x for x, _ in allp], [y for _, y in allp])
    judge_meta = {}
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        jf = d / "judge.json"
        if jf.exists():
            judge_meta = json.loads(jf.read_text(encoding="utf-8")).get("judge", {})
            break
    out = {"kappa": overall, "n": len(allp), "kappa_min": KAPPA_MIN,
           "calibrated": overall is not None and overall >= KAPPA_MIN,
           "per_category": per_cat, "rows_skipped": skipped,
           "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
           "judge": {k: judge_meta.get(k) for k in ("id", "provider", "model", "weights_sha256",
                                                    "prompt_sha256", "rubrics", "stub")},
           "note": (f"below kappa {KAPPA_MIN:.2f} the judged suite is preliminary: shown, never "
                    f"ranked, never in an average")}
    (results / CALIBRATION_FILE).write_text(json.dumps(out, indent=2, sort_keys=True),
                                            encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("results", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("-m", "--model", action="append", default=[])
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--seed", type=int, default=1234)
    i = sub.add_parser("import")
    i.add_argument("results", type=Path)
    i.add_argument("csv", type=Path)
    a = ap.parse_args()
    if a.cmd == "export":
        n = export(a.results, a.out, a.model, a.n, a.seed)
        print(f"wrote {n} rows to {a.out} — fill human_score (0–4) with the rubric column "
              f"beside you, then: judge_calibrate.py import {a.results} {a.out}")
        return 0 if n else 1
    out = import_csv(a.results, a.csv)
    k = out["kappa"]
    print(f"Cohen's kappa overall: {k if k is not None else '—'} over {out['n']} rows "
          f"({out['rows_skipped']} skipped)")
    for c, v in out["per_category"].items():
        print(f"  {c:26} kappa {v['kappa']!s:>7}  agreement {v['agreement']:.0%}  n={v['n']}")
    print("calibrated — judged scores may enter the judged average" if out["calibrated"]
          else f"PRELIMINARY — below {KAPPA_MIN}: shown, never ranked, never averaged")
    print(f"wrote {a.results / CALIBRATION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
