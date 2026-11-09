#!/usr/bin/env python3
"""Scan a folder about to be published for internal terms.

    python3 scan_terms.py <folder> [--private terms.json] [--allow allow.json]

Each file is read as a whole, a CSV cell by cell and a JSONL file field by field. Our
own text must come back empty: any hit there is printed with where it is, and the scan
exits 1. Text that came from a benchmark or a model (a prompt, a model's answer, a
question) is ordinary text, so its hits are counted by term, with examples of each, and
don't fail the scan: read them.

--private names a JSON file of terms that can't live in a public repository (people's
names, addresses, box names, an earlier file's size and hash): {"name": [regex, ...]}.
--allow lets exact hits through, each by its file, row (a CSV row's first cell), column,
term, the text the hit must begin, and how many; they print as "allowed: N", and an
entry that matches nothing prints too."""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TERMS = {
    "step or checkpoint": r"(?i)\bstep[-_ ]?\d+|\bsteps?\b|checkpoint",
    "Q2": r"(?<![A-Za-z0-9])Q2(?![A-Za-z0-9])", "Q4": r"(?<![A-Za-z0-9])Q4(?![A-Za-z0-9])",
    "BF16": r"(?i)\bbf16\b|bfloat16",
    # lower case: the internal names are; "Kimi K3" is a model's
    "k3, k6, k3k6": r"\bk3k6\w*|\bk[346]\b|\bk[346]-\w+",
    "lda": r"(?i)\blda\b|\blda-\w+|-lda\b",
    "LOOKAHEAD": r"LOOKAHEAD\w*",
    # an environment variable's name; a file's quantisation label (Q4_K_M, PQ2_0) isn't one
    "environment variables": r"\b(?!(?:I?Q\d|PQ\d|TQ\d|PTQ\d)\w*)[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",
    "evalboard": r"(?i)evalboard", "board": r"(?i)\bboard\b",
    "IP addresses": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "home paths": r"/home/[a-z]+|/Users/[a-z]+|/workspace/|/data-\d+",
    "a person's handle": r"\bmasein\b(?!-model)",
    "MobileAIBench": r"(?i)mobileaibench|hotpotqa|xsum",
    "Artificial Analysis": r"(?i)artificial analysis",
    "token or secret": r"\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9-]{20,}|\bghp_[A-Za-z0-9]{20,}|"
                       r"(?i:\b(?:api[_-]?key|secret|password|bearer)\b)\s*[:=]",
}
# fields of text from a benchmark or a model: read, not failed
TEXT = {"prompt", "final_answer", "answer", "text"}


def opt(args: list[str], name: str) -> str | None:
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return None


def pieces(f: Path):
    """(where, field, text): a CSV's cells, a JSONL file's fields, any other file whole"""
    t = f.read_text(encoding="utf-8", errors="replace")
    if f.suffix == ".csv":
        rows = list(csv.reader(io.StringIO(t)))
        for r in rows:
            for i, c in enumerate(r):
                yield r[0], rows[0][i] if i < len(rows[0]) else "", c
    elif f.suffix == ".jsonl":
        for n, line in enumerate(t.splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec.get("question_id") or rec.get("question") or n)
            for k, v in rec.items():
                yield qid, k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    else:
        yield "", "", t


def main(argv: list[str]) -> int:
    args = list(argv)
    private = opt(args, "--private")
    allow_file = opt(args, "--allow")
    root = Path(args[0])
    terms = dict(TERMS)
    if private:
        for k, pats in json.loads(Path(private).read_text(encoding="utf-8")).items():
            if pats and not k.startswith("_"):
                terms[k] = "|".join(f"(?i:{p[4:]})" if p.startswith("(?i)") else f"(?:{p})"
                                    for p in pats)
    pats = {k: re.compile(v) for k, v in terms.items()}
    allow = json.loads(Path(allow_file).read_text(encoding="utf-8")) if allow_file else []
    left = [e["count"] for e in allow]

    def allowed(fname, row, col, term, text, at):
        for i, e in enumerate(allow):
            if left[i] and (e["file"], e["row"], e["column"], e["term"]) == (fname, row, col, term) \
                    and text.startswith(e["label"], at):
                left[i] -= 1
                return e["label"]
        return None
    ours, theirs, eg, ok = defaultdict(Counter), defaultdict(Counter), defaultdict(list), Counter()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for f in files:
        rel = str(f.relative_to(root))
        for row, col, text in pieces(f):
            for term, pat in pats.items():
                for m in pat.finditer(text):
                    bit = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
                    if f.suffix == ".jsonl" and col in TEXT:
                        theirs[term][(rel, col)] += 1
                        if len(eg[(term, rel, col)]) < 3:
                            eg[(term, rel, col)].append((row, m.group(0), bit))
                        continue
                    lab = allowed(f.name, row, col, term, text, m.start())
                    if lab:
                        ok[(f.name, row, col, lab)] += 1
                        continue
                    ours[term][(rel, row, col, m.group(0))] += 1
                    eg.setdefault((term, rel, row, col, m.group(0)), [bit])
    print(f"scanned {len(files)} files under {root}")
    print("our own text:")
    for term in terms:
        n = sum(ours[term].values())
        print(f"  {term}: {n}")
        for (rel, row, col, s), k in ours[term].most_common(12):
            print(f"     {k} × {s!r} in {rel}" + (f", row {row!r}, column {col!r}" if row or col else "")
                  + f" · …{eg[(term, rel, row, col, s)][0]}…")
    print(f"  allowed: {sum(ok.values())}")
    for (fn, row, col, lab), k in sorted(ok.items()):
        print(f"     {k} × {lab!r} in {fn}, row {row!r}, column {col!r}")
    gone = [e for e, n in zip(allow, left) if n]
    if gone:
        print(f"  allow-list entries with no hit: {len(gone)}")
        for e in gone:
            print(f"     {e['label']!r} in {e['file']}, row {e['row']!r}, column {e['column']!r}")
    print("text from benchmarks and models (read, not failed):")
    if not any(theirs.values()):
        print("  none")
    for term in terms:
        for (rel, col), k in sorted(theirs[term].items()):
            print(f"  {term}: {k} in {rel}, field {col}")
            for qid, s, bit in eg[(term, rel, col)]:
                print(f"     e.g. {qid}: {s!r} · …{bit}…")
    bad = sum(sum(c.values()) for c in ours.values())
    print("CLEAN: our own text has no hit" if not bad else f"NOT CLEAN: {bad} hits in our own text")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
