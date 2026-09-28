"""12m.2, run once (2026-09-28): the tests' Epoch AI fixture, trimmed from
their benchmark_data.zip (CC BY 4.0 — "Data: Epoch AI, CC BY 4.0").

    python docs/prompts/phase-12m/trim_epoch.py <unzipped benchmark_data>

It writes tests/fixtures/epoch/: benchmark_metadata.csv and model_metadata.csv
(only the rows the others need), and a few rows each of GPQA diamond (their
own runs: OpenAI's twelve newest models with every reasoning effort, two of
Anthropic's and of Google's, and Qwen3-1.7B), MMLU and Aider polyglot (from
other sources, one in percent), and a superseded FrontierMath set. Every
value is theirs, as downloaded; the zip itself is not kept.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests" / "fixtures" / "epoch"
KEEP = {"GPQA diamond", "MMLU", "Aider polyglot", "FrontierMath-2025-02-28-Private"}


def rows(p: Path) -> tuple[list[str], list[dict]]:
    with open(p, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def write(name: str, fields: list[str], rs: list[dict]) -> None:
    with open(OUT / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rs)


def main(src: Path) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bf, bench = rows(src / "benchmark_metadata.csv")
    mf, meta = rows(src / "model_metadata.csv")
    meta_of = {r["model_version"]: r for r in meta if r["model_version"]}
    _, gpqa = rows(src / "gpqa_diamond.csv")
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in gpqa:
        m = meta_of.get(r["Model version"], {})
        groups.setdefault((m.get("organization", ""), m.get("model_group", "")), []).append(r)
    newest = lambda org, n: sorted(  # noqa: E731
        (g for g in groups if g[0] == org), key=lambda g: max(meta_of[r["Model version"]]["date"]
                                                             for r in groups[g]), reverse=True)[:n]
    want = newest("OpenAI", 12) + newest("Anthropic", 2) + newest("Google DeepMind", 2)
    gp = [r for g in want for r in groups[g]]
    gp += [r for r in gpqa if r["Model version"] in ("qwen3-1.7b", "qwen3-1.7b_none")]
    gf, _ = rows(src / "gpqa_diamond.csv")
    write("gpqa_diamond.csv", gf, gp)
    used = {r["Model version"] for r in gp}
    for name, n in (("mmlu_external.csv", 5), ("aider_polyglot_external.csv", 4),
                    ("frontiermath.csv", 3)):
        f, rs = rows(src / name)
        rs = rs[:n]
        write(name, f, rs)
        used |= {r["Model version"] for r in rs}
    write("benchmark_metadata.csv", bf, [r for r in bench if r["benchmark"] in KEEP])
    write("model_metadata.csv", mf, [meta_of[v] for v in sorted(used) if v in meta_of])
    for p in sorted(OUT.glob("*.csv")):
        print(p.name, sum(1 for _ in open(p, encoding="utf-8")) - 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
