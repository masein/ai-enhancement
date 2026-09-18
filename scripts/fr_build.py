#!/usr/bin/env python3
"""Build the free-response task directory the harness runs from.

    python scripts/fr_build.py results/full --out $BENCH_ROOT/eval_tasks/fr

Two kinds of item land in --out:

  fr_<category>.jsonl     copied from eval_tasks/fr/ in this repo — human-
                          authored (ten SEED items each until someone writes
                          the rest; see eval_tasks/fr/AUTHORING.md)
  fr_control_mmlu.jsonl   BUILT here from the DIAGNOSE half of MMLU already on
                          disk: the question only, no options, graded against
                          the gold option's text. About ten items per human
                          category, stratified, chosen by sorted doc_hash so
                          the set is the same on every machine.

plus one lm_eval yaml per task with the items file's absolute path filled in
(lm_eval resolves data_files relative to nothing useful; make_ppl_task.py
does the same), and a manifest with every file's sha256.

Why the control set is diagnose-only, and why that is tested: these items
are shown to a judge and, through judge.json, to people. The report half is
never shown to anything.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import categories as _categories  # noqa: E402
import diagnose as dx  # noqa: E402

SEED_DIR = REPO / "eval_tasks" / "fr"
CATEGORIES = ["instruction_following", "factual_accuracy", "reasoning", "cultural"]
FR_TASKS = [f"fr_{c}" for c in CATEGORIES]
CONTROL_TASK = "fr_control_mmlu"
ALL_TASKS = FR_TASKS + [CONTROL_TASK]
CONTROL_PER_CATEGORY = 10
CONTROL_SUFFIX = " Answer in one or two sentences, without listing options."


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mmlu_docs(results_root: Path) -> dict[str, dict]:
    """Every distinct MMLU document on disk, by doc_hash — from whichever
    model's per-item log has them; the documents are the same for all."""
    out: dict[str, dict] = {}
    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for task_dir in sorted(model_dir.glob("mmlu_*shot")):
            if not re.fullmatch(r"mmlu_\d+shot", task_dir.name):
                continue                   # not mmlu_perm: rotated options are not the question
            for f in dx.newest_per_subtask(sorted(task_dir.rglob("samples_mmlu_*.jsonl"))):
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        dh, doc = rec.get("doc_hash"), rec.get("doc") or {}
                        if dh and dh not in out and doc.get("question") and \
                                isinstance(doc.get("choices"), list):
                            out[dh] = doc
        if out:
            break                          # one model's log is the whole benchmark
    return out


def control_items(docs: dict[str, dict], per_category: int = CONTROL_PER_CATEGORY) -> list[dict]:
    """Diagnose-half MMLU questions, stratified by human category, deterministic."""
    by_cat: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for dh in sorted(docs):
        if dx.split_of(dh) != "diagnose":                  # the rule
            continue
        doc = docs[dh]
        cat = _categories.categorize(doc.get("subject", "")) or _categories.OTHER
        by_cat[cat].append((dh, doc))
    items = []
    for cat in _categories.category_order():
        for dh, doc in by_cat.get(cat, [])[:per_category]:
            try:
                gold = str(doc["choices"][int(doc["answer"])])
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            items.append({"id": f"{CONTROL_TASK}-{dh[:12]}", "category": cat,
                          "subject": doc.get("subject", ""),
                          "prompt": str(doc["question"]).strip() + CONTROL_SUFFIX,
                          "reference": gold, "mmlu_doc_hash": dh})
    return items


def build(results_root: Path, out: Path, per_category: int = CONTROL_PER_CATEGORY,
          seed_dir: Path = SEED_DIR) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    template = (seed_dir / "_fr_template_yaml").read_text(encoding="utf-8")
    manifest = {"tasks": {}, "rubrics": {}}
    for task in FR_TASKS:
        src = seed_dir / f"{task}.jsonl"
        dst = out / f"{task}.jsonl"
        shutil.copyfile(src, dst)
        n = sum(1 for ln in dst.read_text(encoding="utf-8").splitlines() if ln.strip())
        seeds = sum(1 for ln in dst.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and json.loads(ln).get("seed"))
        manifest["tasks"][task] = {"items": n, "seed_items": seeds, "sha256": sha256_file(dst),
                                   "authored": True}
    items = control_items(mmlu_docs(results_root), per_category)
    dst = out / f"{CONTROL_TASK}.jsonl"
    dst.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items),
                   encoding="utf-8")
    manifest["tasks"][CONTROL_TASK] = {
        "items": len(items), "seed_items": 0, "sha256": sha256_file(dst), "authored": False,
        "per_category": collections.Counter(it["category"] for it in items),
        "split": "diagnose", "split_salt": dx.SPLIT_SALT}
    for task in ALL_TASKS:
        yaml = template.replace("__ITEMS_PATH__", str((out / f"{task}.jsonl").resolve()))
        (out / f"{task}.yaml").write_text(f"task: {task}\n" + yaml, encoding="utf-8")
    for r in sorted((seed_dir / "rubrics").glob("*.md")):
        manifest["rubrics"][r.stem] = sha256_file(r)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                       encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="the results/full tree (source of the control set)")
    ap.add_argument("--out", type=Path, required=True,
                    help="where the harness will find the tasks: $BENCH_ROOT/eval_tasks/fr")
    ap.add_argument("--per-category", type=int, default=CONTROL_PER_CATEGORY)
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    m = build(a.results, a.out, a.per_category)
    for t, v in m["tasks"].items():
        print(f"{t:26} {v['items']:4} items" + (f"  ({v['seed_items']} SEED — see AUTHORING.md)"
                                                 if v["seed_items"] else "  (built, diagnose half)"))
    if not m["tasks"][CONTROL_TASK]["items"]:
        print("\nno MMLU per-item log under the results tree: the control set is empty",
              file=sys.stderr)
        return 1
    print(f"\nwrote {a.out} — run with suite=judged, or by hand with "
          f"--include_path {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
