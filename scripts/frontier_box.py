#!/usr/bin/env python3
"""17e: one pasted line a box — G5's two plans (docs/REMOTE-RUNS.md), each
box's commands run one after the other, so a box needs no second visit.

    python scripts/frontier_box.py A3 --as served/<build> \\
        --gguf hf://<you>/evalboard-private/<file>.gguf \\
        --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz

Each step is scripts/remote_gguf.py with the box's benchmarks, thinking
setting, shard and slots, its own --out (/workspace/A3-1, /workspace/A3-2, …)
and the build's flags; the GGUF and the tarball are fetched once, into the
folder every step shares (/workspace/files). A step that stops doesn't stop
the next, and the last lines say how each ended. Paste the same line again to
carry on: each step asks only what it hasn't answered, and a step already
whole only makes its bundle again.

--list A (or B) prints a plan: each box's steps and its hours at the pilot's
paces (one RTX 5090, the phone build, 6 Oct).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frontier as fb  # noqa: E402

BASED_ON = "Qwen/Qwen3.6-35B-A3B"
FLAGS = "-ctk q8_0 -ctv q8_0 --flash-attn on"

# the pilot's seconds an answer, wall-clock on the box at 8 slots (ARC-AGI-2
# at 5); SimpleQA with thinking on wasn't asked (its MMLU-Pro was stopped by
# hand first), so its pace is a guess
PACE = {
    ("on", "gpqa_diamond_epoch"): 23.2, ("on", "hle_text_cais"): 47.5,
    ("on", "mmlupro_tiger"): 5.3, ("on", "math_l5_epoch"): 35.3,
    ("on", "otis_aime_epoch"): 104.3, ("on", "arc_agi2_public"): 109.4,
    ("on", "simpleqa_epoch"): 3.6,
    ("off", "gpqa_diamond_epoch"): 4.3, ("off", "hle_text_cais"): 5.0,
    ("off", "math_l5_epoch"): 5.2, ("off", "mmlupro_tiger"): 1.9,
    ("off", "otis_aime_epoch"): 9.1, ("off", "simpleqa_epoch"): 0.6,
}
GUESSED = {("on", "simpleqa_epoch")}

# a benchmark's answers: its questions × its runs. ARC-AGI-2's questions are
# its test grids, 167 in its 120 tasks at the pinned revision, each asked twice
ANSWERS = {t: (167 if t == "arc_agi2_public" else b["n"]) * b["epochs"]
           for t, b in fb.BENCH.items()}

GPQA, OTIS, MATH, HLE = "gpqa_diamond_epoch", "otis_aime_epoch", "math_l5_epoch", "hle_text_cais"
SQA, MMLU, ARC = "simpleqa_epoch", "mmlupro_tiger", "arc_agi2_public"

# a step: (thinking, its benchmarks, its shard, its slots). ARC-AGI-2's prompts
# need 98,304 tokens a slot: 5 slots, and a step of its own
PLANS: dict[str, dict[str, list[tuple[str, tuple[str, ...], str, int]]]] = {
    # about 6 boxes a build, the longest about 18 hours
    "A": {
        "A1": [("on", (HLE,), "1/2", 8), ("off", (HLE,), "", 8)],
        "A2": [("on", (HLE,), "2/2", 8), ("off", (OTIS, MATH), "", 8)],
        "A3": [("on", (MMLU,), "", 8)],
        "A4": [("on", (MATH, SQA), "", 8), ("off", (GPQA, SQA), "", 8)],
        "A5": [("on", (ARC,), "", 5), ("on", (GPQA,), "", 8)],
        "A6": [("on", (OTIS,), "", 8), ("off", (MMLU,), "", 8)],
    },
    # 10 boxes a build, the longest about 10 hours
    "B": {
        "B1": [("on", (HLE,), "1/3", 8), ("off", (OTIS,), "", 8)],
        "B2": [("on", (HLE,), "2/3", 8)],
        "B3": [("on", (HLE,), "3/3", 8)],
        "B4": [("on", (MMLU,), "1/2", 8), ("on", (SQA,), "", 8)],
        "B5": [("on", (MMLU,), "2/2", 8), ("off", (GPQA, SQA), "", 8)],
        "B6": [("on", (MATH,), "1/2", 8), ("off", (MMLU,), "1/2", 8)],
        "B7": [("on", (MATH,), "2/2", 8), ("off", (MMLU,), "2/2", 8)],
        "B8": [("on", (ARC,), "", 5)],
        "B9": [("on", (OTIS,), "", 8)],
        "B10": [("on", (GPQA,), "", 8), ("off", (HLE, MATH), "", 8)],
    },
}


def box_of(name: str) -> list[tuple[str, tuple[str, ...], str, int]]:
    plan = PLANS.get(name[:1].upper()) or {}
    steps = plan.get(name.upper())
    if not steps:
        raise SystemExit(f"{name}: no such box — "
                         + "; ".join(f"plan {p}: {', '.join(b)}" for p, b in PLANS.items()))
    return steps


def hours(step: tuple[str, tuple[str, ...], str, int]) -> float:
    """a step's hours at the pilot's paces"""
    th, tasks, shard, _ = step
    of = fb.parse_shard(shard)[1] if shard else 1
    return sum(ANSWERS[t] * PACE[(th, t)] for t in tasks) / of / 3600


def words(step: tuple[str, tuple[str, ...], str, int]) -> str:
    th, tasks, shard, slots = step
    return (f"thinking {th} · {', '.join(fb.BENCH[t]['label'] for t in tasks)}"
            + (f" · shard {shard}" if shard else "") + f" · {slots} slots · about "
            f"{hours(step):.1f} h" + (" (a guess)" if any((th, t) in GUESSED for t in tasks)
                                      else ""))


def argv_of(box: str, k: int, step: tuple[str, tuple[str, ...], str, int],
            a: argparse.Namespace) -> list[str]:
    """remote_gguf.py's arguments for step k (1-based) of `box`"""
    th, tasks, shard, slots = step
    out = [sys.executable, str(HERE / "remote_gguf.py"), "--as", a.served_as, "--gguf", a.gguf,
           "--server", a.server, "--based-on", a.based_on, "--thinking", th, "--slots",
           str(slots), "--flags", a.flags]
    if a.env:
        out += ["--env", a.env]
    for t in tasks:
        out += ["--only", t]
    if shard:
        out += ["--shard", shard]
    return out + ["--out", str(Path(a.root) / f"{box}-{k}")]


def listing(plan: str) -> list[str]:
    boxes = PLANS[plan]
    lines = []
    for box, steps in boxes.items():
        lines.append(f"{box} · about {sum(hours(s) for s in steps):.1f} h")
        lines += [f"  {k}. {words(s)}" for k, s in enumerate(steps, 1)]
    total = sum(hours(s) for steps in boxes.values() for s in steps)
    lines.append(f"{len(boxes)} boxes a build · {total:.1f} box-hours a build · the longest "
                 f"{max(sum(hours(s) for s in st) for st in boxes.values()):.1f} h")
    return lines


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("box", nargs="?", help="the box: A1 … A6, B1 … B10 (G5)")
    ap.add_argument("--list", choices=sorted(PLANS), help="print a plan and stop")
    ap.add_argument("--as", dest="served_as", default="", help="served/<name>, the build's")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--server", default="")
    ap.add_argument("--based-on", default=BASED_ON)
    ap.add_argument("--flags", default=FLAGS, help=f"llama-server's flags (default: {FLAGS})")
    ap.add_argument("--env", default="", help="the build's llama-server environment, if any")
    ap.add_argument("--root", default="/workspace", help="where each step's --out goes")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(listing(a.list)))
        return 0
    if not a.box:
        ap.error("give the box (A1 … B10), or --list A")
    missing = [n for n, v in (("--as", a.served_as), ("--gguf", a.gguf),
                              ("--server", a.server)) if not v]
    if missing:
        ap.error(f"{', '.join(missing)}: the build's, as G2 gives them")
    box = a.box.upper()
    steps = box_of(box)
    print(f"{box}: {len(steps)} step{'s' if len(steps) > 1 else ''}, about "
          f"{sum(hours(s) for s in steps):.1f} h at the pilot's paces", flush=True)
    ends = []
    for k, step in enumerate(steps, 1):
        cmd = argv_of(box, k, step, a)
        print(f"\n{box}, step {k} of {len(steps)}: {words(step)}\n"
              + " ".join(shlex.quote(x) for x in cmd[1:]), flush=True)
        code = subprocess.run(cmd).returncode
        ends.append((k, step, code))
    print()
    for k, step, code in ends:
        out = Path(a.root) / f"{box}-{k}"
        print(f"{box}, step {k}: " + ("whole — its bundle is in " + str(out) if code == 0 else
                                       f"not whole (exit {code}) — its lines above say why; "
                                       "paste the same line again to carry on"), flush=True)
    return 0 if all(code == 0 for _, _, code in ends) else 1


if __name__ == "__main__":
    raise SystemExit(main())
