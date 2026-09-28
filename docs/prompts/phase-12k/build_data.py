"""12k.2, run once (2026-09-28): Trust & safety's data, from the pinned sources.

    python docs/prompts/phase-12k/build_data.py <downloads>

<downloads> holds, fetched at the revisions in SOURCES below:
    dna.csv                         LibrAI/do-not-answer   data_en.csv
    xstest.csv                      Paul/XSTest            xstest_prompts.csv
    bbq/<Category>.jsonl            nyu-mll/BBQ            data/*.jsonl (11 files)
    bbq/additional_metadata.csv     nyu-mll/BBQ            analysis_scripts/additional_metadata.csv

It writes eval_tasks/trust_safety/: the prompts and questions the runs ask,
and manifest.json — each source's URL, revision, licence and the sha256 of
every file read, and the sha256 and count of every file written. A changed
source is a new version (tests/test_trust_safety_12k2.py holds them).

- Do-Not-Answer: its 939 prompts, each with its risk area and harm types.
  The six models' replies and their labels stay out; the authors' six action
  categories are in the judge's rubric (scripts/trust_safety.py).
- XSTest: its 450 prompts, 250 safe (over-refusal) and 200 unsafe.
- BBQ: the ambiguous-context questions only (29,246 of 58,492), where the
  right answer is "unknown"; each carries where its stereotyped answer is
  (the authors' target_loc), for the bias score. A seeded 3,000 is the
  default run; all of them is the second choice.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "eval_tasks" / "trust_safety"
SEED = "bbq-ambig-3000"
N_SUBSET = 3000
CATEGORIES = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance",
              "Race_ethnicity", "Race_x_SES", "Race_x_gender", "Religion", "SES",
              "Sexual_orientation"]
SOURCES = {
    "do_not_answer": {
        "name": "Do-Not-Answer", "licence": "Apache-2.0",
        "url": "https://huggingface.co/datasets/LibrAI/do-not-answer",
        "revision": "74e74f2e4507ef256fe536f78a776f4a1ff67955",
        "files": {"dna.csv": "data_en.csv"},
        "cite": "Wang et al., Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs (2023)"},
    "xstest": {
        "name": "XSTest", "licence": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/Paul/XSTest",
        "revision": "f600c994b256f12867dfa5b3eb3d545a3e62f8b5",
        "files": {"xstest.csv": "xstest_prompts.csv"},
        "cite": "Röttger et al., XSTest: A Test Suite for Identifying Exaggerated Safety "
                "Behaviours in Large Language Models (NAACL 2024)"},
    "bbq": {
        "name": "BBQ", "licence": "CC BY 4.0",
        "url": "https://github.com/nyu-mll/BBQ",
        "revision": "bea11bd97d79217245b5871acd247b9d6eb24598",
        "files": {**{f"bbq/{c}.jsonl": f"data/{c}.jsonl" for c in CATEGORIES},
                  "bbq/additional_metadata.csv": "analysis_scripts/additional_metadata.csv"},
        "cite": "Parrish et al., BBQ: A Hand-Built Bias Benchmark for Question Answering "
                "(Findings of ACL 2022)"},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_jsonl(p: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    if p.suffix == ".gz":
        # mtime 0: the same rows are the same bytes, so the sha256 pins them
        with open(p, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(text.encode("utf-8"))
    else:
        p.write_text(text, encoding="utf-8")


def do_not_answer(src: Path) -> list[dict]:
    return [{"id": f"dna-{int(r['id']):03d}", "risk_area": r["risk_area"],
             "types_of_harm": r["types_of_harm"], "specific_harms": r["specific_harms"],
             "prompt": r["question"].strip()}
            for r in csv.DictReader(open(src / "dna.csv", encoding="utf-8"))]


def xstest(src: Path) -> list[dict]:
    return [{"id": f"xstest-{int(r['id']):03d}", "label": r["label"], "type": r["type"],
             "focus": r["focus"], "note": r["note"], "prompt": r["prompt"].strip()}
            for r in csv.DictReader(open(src / "xstest.csv", encoding="utf-8"))]


def bbq(src: Path) -> list[dict]:
    target = {}
    for r in csv.DictReader(open(src / "bbq" / "additional_metadata.csv", encoding="utf-8")):
        loc = (r.get("target_loc") or "").strip()
        target[(r["category"], int(r["example_id"]))] = int(float(loc)) if loc not in ("", "NA", "nan") else None
    out = []
    for c in CATEGORIES:
        for line in open(src / "bbq" / f"{c}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["context_condition"] != "ambig":
                continue
            info = r["answer_info"]
            unknown = next(i for i in range(3) if info[f"ans{i}"][1] == "unknown")
            assert r["label"] == unknown, (c, r["example_id"])       # ambiguous: "unknown" is right
            out.append({"id": f"bbq-{c}-{r['example_id']}", "category": c,
                        "polarity": r["question_polarity"], "context": r["context"],
                        "question": r["question"], "choices": [r["ans0"], r["ans1"], r["ans2"]],
                        "label": r["label"], "target": target.get((c, r["example_id"]))})
    return out


def main(argv: list[str]) -> int:
    src = Path(argv[0]) if argv else None
    if not src or not src.is_dir():
        print(__doc__)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    dna, xs, bq = do_not_answer(src), xstest(src), bbq(src)
    sub = sorted(random.Random(SEED).sample(bq, N_SUBSET), key=lambda r: r["id"])
    written = {"do_not_answer.jsonl": dna, "xstest.jsonl": xs,
               "bbq_ambig.jsonl.gz": bq, "bbq_ambig_3000.jsonl": sub}
    for name, rows in written.items():
        write_jsonl(OUT / name, rows)
    manifest = {"built": "2026-09-28", "sources": {
        k: {**{x: v for x, v in s.items() if x != "files"},
            "files": {up: sha(src / local) for local, up in s["files"].items()}}
        for k, s in SOURCES.items()},
        "files": {name: {"sha256": sha(OUT / name), "n": len(rows)} for name, rows in written.items()},
        "bbq_subset": {"seed": SEED, "n": N_SUBSET, "of": len(bq)}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    for name, rows in written.items():
        print(f"{name}: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
