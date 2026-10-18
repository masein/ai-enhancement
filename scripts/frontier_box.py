#!/usr/bin/env python3
"""17e: one pasted line a box — G5's two plans (docs/REMOTE-RUNS.md), each
box's commands run one after the other, so a box needs no second visit.

    python scripts/frontier_box.py A3 --as served/<build> \\
        --gguf hf://<you>/evalboard-private/<file>.gguf \\
        --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz

Each step is scripts/remote_gguf.py with the box's benchmarks, thinking
setting, shard and slots, the build's flags, and its own --out; the GGUF and
the tarball are fetched once, into the folder every step shares
(/workspace/files). A step that stops doesn't stop the next, and the last
lines say how each ended. Paste the same line again to carry on: each step
asks only what it hasn't answered, and a step already whole only makes its
bundle again (the import tells its answers are the same: nothing changes).

17f:
- each build's steps go in a folder of the build's own
  (/workspace/<build>/A3-1, …), so the same box can run the other build
  after, and frontier_fetch.py finds both;
- the box's label goes with each step (remote_gguf.py --label A3): into its
  bundle's setup.json and its progress file;
- a step planned at 8 slots runs 7 where the card holds no more
  (--min-slots 7) — 17g: HLE with thinking on, at 86,016 tokens a slot, runs
  8 on a 5090, by the pilot's measured slope;
- one box a build (A3, B12) asks the parity questions first.

--list A (or B) prints a plan: each box's steps and its hours, at the pilot's
paces (one RTX 5090, the phone build, 6 Oct) under the token limits of 6 Oct.

20: plan L — step4000 against Q4 and BF16, the lda runtime, thinking on only,
no box over about 6 hours on a 5090 at those paces. Its boxes run the
runtime of PLAN_RUNTIME (--runtime lda-2610: no one types its values), from
the lda branch's tarball as any build's:

    python scripts/frontier_box.py L3 --as served/step4000-lda-2610 \\
        --gguf hf://<you>/evalboard-private/step4000.gguf \\
        --server hf://<you>/evalboard-private/llama-server-lda-a4a76a7.tar.gz

--server-dir, in place of --server, is the fallback: a build directory the
team put on the box (docs/STEP4000-RUN.md).

22: Ornith-1.5-35B-A3B (Q4_K_M), a third-party model, on plan L's boxes with
a runtime of its own — card-ornith: upstream llama.cpp b11555 and its card's
settings, never the lda runtime. Its model names its runtime (fb.RUNTIMES's
"models") and what it is based on; its file is pinned (PLAN_PINS):

    python scripts/frontier_box.py L13 --as served/Ornith-1.5-35B-A3B-Q4_K_M \\
        --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@<commit>/Ornith-1.5-35B-Q4_K_M.gguf \\
        --server hf://<you>/evalboard-private@<commit>/llama-server-b11555.tar.gz \\
        --runtime card-ornith

--pilot, on the first box before the other 20 are rented: 20 GPQA Diamond
and 5 MMLU-Pro questions with the runtime, a line each, the plan's hours at
their pace, and last whether the thinking and the answer came back apart
(docs/ORNITH-RUN.md).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frontier as fb  # noqa: E402

BASED_ON = "Qwen/Qwen3.6-35B-A3B"
FLAGS = "-ctk q8_0 -ctv q8_0 --flash-attn on"

GPQA, OTIS, MATH, HLE = "gpqa_diamond_epoch", "otis_aime_epoch", "math_l5_epoch", "hle_text_cais"
SQA, MMLU, ARC = "simpleqa_epoch", "mmlupro_tiger", "arc_agi2_public"

# the pilot's seconds an answer, wall-clock on the box at 8 slots (ARC-AGI-2
# at 5); SimpleQA with thinking on wasn't asked (its MMLU-Pro was stopped by
# hand first), so its pace is a guess
PACE = {
    ("on", GPQA): 23.2, ("on", HLE): 47.5, ("on", MMLU): 5.3, ("on", MATH): 35.3,
    ("on", OTIS): 104.3, ("on", ARC): 109.4, ("on", SQA): 3.6,
    ("off", GPQA): 4.3, ("off", HLE): 5.0, ("off", MATH): 5.2, ("off", MMLU): 1.9,
    ("off", OTIS): 9.1, ("off", SQA): 0.6,
}
GUESSED = {("on", SQA)}
PILOT_SLOTS = {ARC: 5}                  # the slots each pace was measured at; 8 otherwise

# 17f: what ran out of room on the pilot — (answers that ran out, of, the mean
# answer's tokens, the limit then). The hours assume each that ran out runs
# to the new limit, at the same tokens a second: the most it could take
RAN_OUT = {
    ("on", GPQA): (2, 20, 7237, 32768), ("on", HLE): (17, 54, 20801, 32768),
    ("on", OTIS): (8, 16, 39601, 65536), ("on", ARC): (3, 10, 34288, 65536),
    ("off", GPQA): (4, 20, 2177, 4096), ("off", HLE): (20, 54, 2468, 4096),
    ("off", OTIS): (8, 16, 4923, 8192), ("off", MMLU): (11, 301, 898, 4096),
}

# the pilot's card and file, for the slots a step's context leaves room for:
# its header's KV cache a token, a slot's recurrent state, the file, the card
PILOT = {"per_token": 10_880, "recurrent": 68_059_136, "file": 22_854_339_808,
         "card_mib": 32_607}

# a benchmark's answers: its questions × its runs. ARC-AGI-2's questions are
# its test grids, 167 in its 120 tasks at the pinned revision, each asked twice
ANSWERS = {t: (167 if t == ARC else b["n"]) * b["epochs"] for t, b in fb.BENCH.items()}
PARITY_STEP = ("parity", (), "", 8)
# 22: a runtime's pilot (fb.PILOT's questions), on the first box of a plan
# before the others are rented — at the plan's 8 slots
PILOT_STEP = ("pilot", tuple(fb.PILOT["n"]), "", 8)
# 22: a 5090 on vast.ai, an hour, for the pilot's sum
BOX_PRICE = 0.50
RUNS = 3                    # 17g: a step's runs on its own (service/frontier.WRITE_OFF_RUNS)

# a step: (thinking, its benchmarks, its shard, its slots). ARC-AGI-2's prompts
# need 114,688 tokens a slot: 5 slots, and a step of its own
Step = tuple[str, tuple[str, ...], str, int]
PLANS: dict[str, dict[str, list[Step]]] = {
    # about 9 boxes a build, the longest under 18 hours
    "A": {
        "A1": [("on", (HLE,), "1/4", 8), ("off", (GPQA, OTIS), "", 8)],
        "A2": [("on", (HLE,), "2/4", 8), ("off", (MATH, SQA), "", 8)],
        "A3": [PARITY_STEP, ("on", (HLE,), "3/4", 8)],
        "A4": [("on", (HLE,), "4/4", 8)],
        "A5": [("on", (MMLU,), "", 8)],
        "A6": [("on", (MATH, SQA), "", 8)],
        "A7": [("on", (OTIS,), "", 8), ("off", (MMLU,), "1/2", 8)],
        "A8": [("on", (ARC,), "", 5), ("off", (MMLU,), "2/2", 8)],
        "A9": [("on", (GPQA,), "", 8), ("off", (HLE,), "", 8)],
    },
    # 15 boxes a build, the longest about 10 hours
    "B": {
        **{f"B{i}": [("on", (HLE,), f"{i}/6", 8)] for i in range(1, 7)},
        "B7": [("on", (MMLU,), "1/2", 8), ("on", (SQA,), "", 8)],
        "B8": [("on", (MMLU,), "2/2", 8), ("off", (SQA,), "", 8)],
        "B9": [("on", (MATH,), "1/2", 8), ("off", (MMLU,), "1/2", 8)],
        "B10": [("on", (MATH,), "2/2", 8), ("off", (MMLU,), "2/2", 8)],
        "B11": [("on", (OTIS,), "1/2", 8), ("off", (GPQA, MATH), "", 8)],
        "B12": [PARITY_STEP, ("on", (OTIS,), "2/2", 8), ("off", (OTIS,), "", 8)],
        "B13": [("on", (ARC,), "1/2", 5), ("off", (HLE,), "1/2", 8)],
        "B14": [("on", (ARC,), "2/2", 5), ("off", (HLE,), "2/2", 8)],
        "B15": [("on", (GPQA,), "", 8)],
    },
    # 20: step4000 against Q4 and BF16 — thinking on only, every box under
    # about 6 hours on a 5090 at the pilot's paces: HLE in ninths, MMLU-Pro in
    # thirds, MATH Level 5 with GPQA Diamond in quarters, OTIS in thirds (the
    # first with SimpleQA), ARC-AGI-2 in halves at 5 slots. No parity step:
    # the server is out of reach, and asks with another runtime
    "L": {
        **{f"L{i}": [("on", (HLE,), f"{i}/9", 8)] for i in range(1, 10)},
        **{f"L{9 + i}": [("on", (MMLU,), f"{i}/3", 8)] for i in range(1, 4)},
        **{f"L{12 + i}": [("on", (MATH, GPQA), f"{i}/4", 8)] for i in range(1, 5)},
        "L17": [("on", (OTIS,), "1/3", 8), ("on", (SQA,), "", 8)],
        "L18": [("on", (OTIS,), "2/3", 8)],
        "L19": [("on", (OTIS,), "3/3", 8)],
        "L20": [("on", (ARC,), "1/2", 5)],
        "L21": [("on", (ARC,), "2/2", 5)],
    },
}
# 20: the runtime a plan's boxes run, unless the line names another
PLAN_RUNTIME = {"L": "lda-2610"}
# 20: the cards a plan's build runs on — the lda tarball is built for sm_120
# alone (the RTX 5090 and the RTX PRO 6000): a box line on any other card is
# refused before anything is fetched
PLAN_GPUS = {"L": ("RTX 5090", "RTX PRO 6000")}
# 20: the files a plan's boxes run, by name, and the sha256 each must have:
# the lda tarball (teraformer/lda-2026-09-22 at a4a76a751, build 9954, sm_120,
# CUDA 12.8, no web page), step4000's GGUF (13,592,481,632 bytes), and the
# Q4 and BF16 files the earlier rows ran — unsloth/Qwen3.6-35B-A3B-MTP-GGUF at
# 5bc3e238d916, as Hugging Face publishes them (BF16 in two parts: its split
# identity f391f702c4ad97282bf9a9babb494f2639322ff7d17ae28f54237fe00e715ca2)
PLAN_PINS = {"L": {
    "llama-server-lda-a4a76a7.tar.gz":
        "5b538290644d67139c6c996fdaa3e04cce82fd0b343cce66a4ff234648afb618",
    "step4000.gguf": "630c408a3836f82afa1f79629b06d9390d3f736f6140791baa42e180a640565e",
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf":
        "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1",
    "Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf":
        "a172dfea1fe8bb29132ea2f250c74cdc7d619d518d3f072d3816e31a69ce44a2",
    "Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf":
        "b90b4d07e1be4967aec932e2501e7476ec5f44de772c7eb082d6e9c3ac105161",
    # 22: Ornith-1.5-35B-A3B's Q4_K_M, 21,713,463,040 bytes, as Hugging Face
    # publishes it in ornith-ai/Ornith-1.5-35B-A3B-GGUF at 12393612fd4f
    "Ornith-1.5-35B-Q4_K_M.gguf":
        "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"}}


def gpu_refused(box: str, info: dict | None = None) -> str:
    """20: '' when this box's every card is one its plan's build runs on;
    else why not, in one line"""
    want = PLAN_GPUS.get(box[:1].upper())
    if not want:
        return ""
    if info is None:
        import remote_bundle as rb
        info = rb.gpu_info()
    names = info.get("names") or ([info["name"]] if info.get("name") else [])
    cards = " or an ".join(want)
    if not names:
        return (f"{box}: no GPU could be read here (nvidia-smi) — plan {box[:1].upper()} runs "
                f"on an {cards} only. Nothing was fetched")
    bad = sorted({n for n in names if not any(w in n for w in want)})
    if bad:
        return (f"{box}: this box has {', '.join(bad)} — plan {box[:1].upper()} runs on an "
                f"{cards} only (its builds are for sm_120 alone). Nothing was fetched")
    return ""


def pins_of(a: argparse.Namespace, box: str) -> list[str]:
    """20: --pin NAME=SHA256 for each file the line gives that its plan pins"""
    import remote_gguf as rg
    pins = PLAN_PINS.get(box[:1].upper()) or {}
    # a GGUF in parts: each part's, as the box fetches and hashes each
    given = [*(rg.split_parts(a.gguf) if a.gguf else []),
             *([] if getattr(a, "server_dir", "") else [a.server])]
    return [f"{n}={pins[n]}" for n in sorted({str(x).rsplit("/", 1)[-1] for x in given if x})
            if n in pins]


def box_of(name: str) -> list[Step]:
    plan = PLANS.get(name[:1].upper()) or {}
    steps = plan.get(name.upper())
    if not steps:
        raise SystemExit(f"{name}: no such box — "
                         + "; ".join(f"plan {p}: {', '.join(b)}" for p, b in PLANS.items()))
    return steps


def ordered(tasks: tuple[str, ...]) -> list[str]:
    """a step's benchmarks in the order the box asks them: the suite's"""
    return [t for t in fb.TASKS if t in tasks]


def limit_factor(th: str, t: str) -> float:
    """how much longer a benchmark takes under 6 Oct's limit, at the most"""
    k, n, mean, old = RAN_OUT.get((th, t), (0, 1, 1, 0))
    new = fb.BENCH[t]["budget"][th]
    return (mean + k / n * max(0, new - old)) / mean


def fits(ctx: int) -> int:
    """the slots of `ctx` tokens the pilot's card holds, as remote_gguf.py
    checks them: its file, CUDA's and the buffers, and the room kept spare"""
    import remote_gguf as rg
    fixed = max(0, rg.ABOVE_FILE - rg.ABOVE_FILE_SLOTS * PILOT["recurrent"])     # 17g
    room = PILOT["card_mib"] * 1024 ** 2 - PILOT["file"] - fixed - rg.ROOM
    return int(room // ((PILOT["per_token"] + rg.CTX_BUFFERS) * ctx + PILOT["recurrent"]))


def slots_run(step: Step) -> int:
    """the slots a step runs on a 5090: as planned, or fewer where they don't fit"""
    th, tasks, _, slots = step
    if th == "parity":
        return slots
    return min(slots, fits(fb.slot_context(list(tasks), th == "on")))


def hours(step: Step) -> float:
    """a step's hours at the pilot's paces, under the limits of 6 Oct, at the
    slots it runs"""
    th, tasks, shard, _ = step
    if th == "parity":
        return 2 * fb.PARITY["n"] * PACE[("off", MMLU)] / 3600
    of = fb.parse_shard(shard)[1] if shard else 1
    run = slots_run(step)
    return sum(ANSWERS[t] * PACE[(th, t)] * limit_factor(th, t) * PILOT_SLOTS.get(t, 8) / run
               for t in tasks) / of / 3600


def words(step: Step) -> str:
    th, tasks, shard, slots = step
    if th == "parity":
        return (f"the parity questions ({fb.PARITY['n']} of MMLU-Pro, each twice, thinking off) "
                f"· about {hours(step):.1f} h")
    run = slots_run(step)
    return (f"thinking {th} · {', '.join(fb.BENCH[t]['label'] for t in ordered(tasks))}"
            + (f" · shard {shard}" if shard else "")
            + f" · {run} slots" + (f" (of {slots}: no more fit a 5090)" if run < slots else "")
            + f" · about {hours(step):.1f} h"
            + (" (a guess)" if any((th, t) in GUESSED for t in tasks) else ""))


def step_name(box: str, k: int, step: Step) -> str:
    return f"{box}-parity" if step[0] == "parity" else f"{box}-{k}"


def folder(a: argparse.Namespace, box: str, k: int, step: Step) -> Path:
    """17f: the build's own folder — another build on the same box goes
    beside it, never into it"""
    build = a.served_as.split("/", 1)[-1]
    return Path(a.root) / build / step_name(box, k, step)


def pilot_folder(a: argparse.Namespace) -> Path:
    """22: the pilot's folder, in the build's own — named with no "-", so the
    fetch takes it for no step of a plan"""
    return Path(a.root) / a.served_as.split("/", 1)[-1] / "pilot"


def steps_of(box: str) -> list[tuple[str, Step]]:
    """18b: each step folder a box's line makes, with its step"""
    plan = PLANS.get(box[:1].upper()) or {}
    return [(step_name(box.upper(), k, st), st)
            for k, st in enumerate(plan.get(box.upper()) or [], 1)]


def planned(box: str) -> list[str]:
    """17g: every step folder a box's line makes for a build, in order — the
    fetch calls a box safe to destroy only when each is whole and home"""
    plan = PLANS.get(box[:1].upper()) or {}
    return [step_name(box.upper(), k, st) for k, st in enumerate(plan.get(box.upper()) or [], 1)]


def state_of(out: Path) -> dict:
    """a step's progress file (remote_gguf.py), {} when it has none"""
    try:
        got = json.loads((out / "progress.json").read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def whole(out: Path, step: Step, ask_written_off: bool = False) -> bool:
    """17g: a step already whole — its file there — isn't run again by a paste
    (parity has no resume: its 1,000 answers were asked again). 17h: with
    --ask-written-off, one holding answers written off is (its progress says
    how many; one from an image before 17h doesn't, and is run again)"""
    st = state_of(out)
    if st.get("state") != "whole":
        return False
    if ask_written_off and step[0] != "parity" and st.get("written_off", 1):
        return False
    if step[0] == "parity":
        return (out / "parity.jsonl").exists()
    return bool((st.get("bundle") or {}).get("name")) and (out / st["bundle"]["name"]).exists()


def argv_of(box: str, k: int, step: Step, a: argparse.Namespace) -> list[str]:
    """remote_gguf.py's arguments for step k (1-based) of `box`"""
    th, tasks, shard, slots = step
    if getattr(a, "slots", 0):                      # 17g: the line's own --slots, a cap
        slots = min(slots, a.slots)
    out = [sys.executable, str(HERE / "remote_gguf.py"), "--as", a.served_as, "--gguf", a.gguf,
           *(["--server-dir", a.server_dir] if getattr(a, "server_dir", "") else
             ["--server", a.server]),
           "--based-on", based_on_of(a), "--label", box,
           "--files", str(Path(a.root) / "files"), "--flags", flags_of(a, box)]
    rt = runtime_of(a, box)
    if rt:
        out += ["--runtime", rt]
    for pin in pins_of(a, box):
        out += ["--pin", pin]
    if getattr(a, "slots_fit", False):
        out += ["--slots-fit"]
    if getattr(a, "max_slots", 0) and th not in ("parity", "pilot"):   # 18e point 21
        out += ["--max-slots", str(a.max_slots)]
    if a.env:
        out += ["--env", a.env]
    if a.ask_written_off and th != "pilot":
        out += ["--ask-written-off"]
    if th == "parity":
        return out + ["--thinking", "off", "--slots", str(slots), "--parity",
                      "--out", str(folder(a, box, k, step))]
    if th == "pilot":
        # 22: at the plan's slots, so its pace is the plan's
        return out + ["--thinking", "on", "--slots", str(slots), "--min-slots",
                      str(max(1, slots - 1)), "--pilot", "--out", str(pilot_folder(a))]
    out += ["--thinking", th, "--slots", str(slots)]
    if slots > 1 and ARC not in tasks:
        out += ["--min-slots", str(slots - 1)]
    for t in ordered(tasks):
        out += ["--only", t]
    if shard:
        out += ["--shard", shard]
    return out + ["--out", str(folder(a, box, k, step))]


def runtime_of(a: argparse.Namespace, box: str) -> str:
    """20: the runtime a box's steps run: the line's, else its plan's. 22:
    else the one its model runs on alone (Ornith-1.5's card-ornith), before
    the plan's"""
    return (getattr(a, "runtime", "") or fb.runtime_of_model(getattr(a, "served_as", ""))
            or PLAN_RUNTIME.get(box[:1].upper(), ""))


def based_on_of(a: argparse.Namespace) -> str:
    """22: what the model is based on: the line's, else what its runtime
    names for it, else BASED_ON"""
    rt = fb.RUNTIMES.get(fb.runtime_of_model(getattr(a, "served_as", ""))) or {}
    return (getattr(a, "based_on", "") or (rt.get("models") or {}).get(a.served_as)
            or BASED_ON)


def flags_of(a: argparse.Namespace, box: str) -> str:
    """the box's llama-server flags: the line's; else ours (FLAGS), or none
    beside a runtime, which sets its own"""
    if getattr(a, "flags", None) is not None:
        return a.flags
    return "" if runtime_of(a, box) else FLAGS


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
    ap.add_argument("box", nargs="?", help="the box: A1 … A9, B1 … B15 (G5), L1 … L21")
    ap.add_argument("then", nargs="*", help="18e point 1: more labels, run one after another "
                                            "on this box (L1 L2): each later one's line is "
                                            "written \"queued\" at the start, so the fetch "
                                            "never reads the box done between them")
    ap.add_argument("--list", choices=sorted(PLANS), help="print a plan and stop")
    ap.add_argument("--as", dest="served_as", default="", help="served/<name>, the build's")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--server", default="")
    ap.add_argument("--server-dir", default="",
                    help="20: a llama.cpp build directory on the box, in place of --server")
    ap.add_argument("--runtime", default="", choices=["", *sorted(fb.RUNTIMES)],
                    help="20: a named runtime (plan L's boxes run "
                         f"{PLAN_RUNTIME.get('L')} without it; 22: a model a runtime names, "
                         "that runtime)")
    ap.add_argument("--based-on", default="",
                    help=f"what the model is based on (default: what its runtime names for "
                         f"it, else {BASED_ON})")
    ap.add_argument("--flags", default=None,
                    help=f"llama-server's flags (default: {FLAGS}; none beside a runtime)")
    ap.add_argument("--env", default="", help="the build's llama-server environment, if any")
    ap.add_argument("--root", default="/workspace", help="where the build's folder goes")
    ap.add_argument("--ask-written-off", action="store_true",
                    help="17f: each step asks again what earlier runs wrote off as no answer")
    ap.add_argument("--slots", type=int, default=0,
                    help="17g: at most this many slots a step (what a refusal advises)")
    ap.add_argument("--max-slots", type=int, default=0,
                    help="18e point 21: each step runs more slots than planned, up to this "
                         "many, where the box's memory check confirms they fit (25 on an RTX "
                         "PRO 6000 with BF16)")
    ap.add_argument("--pilot", action="store_true",
                    help="22: a runtime's pilot on this box before the others are rented: "
                         f"{' and '.join(str(n) for n in fb.PILOT['n'].values())} questions of "
                         "GPQA Diamond and MMLU-Pro, thinking on, a line each and last whether "
                         "the thinking and the answer came back apart (into /workspace/<build>/"
                         "pilot, which the fetch never reads)")
    ap.add_argument("--slots-fit", action="store_true",
                    help="17g: the slots given fit this card — the memory check's estimate is "
                         "printed, and not held to")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(listing(a.list)))
        return 0
    if not a.box:
        ap.error("give the box (A1 … B15), or --list A")
    missing = [n for n, v in (("--as", a.served_as), ("--gguf", a.gguf),
                              ("--server", a.server or a.server_dir)) if not v]
    if missing:
        ap.error(f"{', '.join(missing)}: the build's, as G2 gives them"
                 + (" (or --server-dir)" if "--server" in missing else ""))
    if a.server and a.server_dir:
        ap.error("--server or --server-dir, not both")
    boxes = [x.upper() for x in [a.box, *(a.then or [])]]
    for box in boxes:
        box_of(box)                     # a label of no plan: refused before anything
    if a.pilot and len(boxes) > 1:
        ap.error("--pilot: one box's label (the box it runs on)")
    # 22: a model a runtime names runs on it alone, and the runtime its own
    # models alone — said before anything is fetched
    for box in dict.fromkeys(boxes):
        why = fb.runtime_model_refused(runtime_of(a, box), a.served_as)
        if why:
            raise SystemExit(f"{box}: {why}. Nothing was fetched")
    # 20: the card first — the plan's build runs on its cards alone
    for box in dict.fromkeys(boxes):
        why = gpu_refused(box)
        if why:
            raise SystemExit(why)
    if a.pilot:
        return run_pilot(a, boxes[0])
    # 18e point 1: every label after the first is queued from the start
    for box in boxes[1:]:
        line_state(a, box, "queued")
    codes = [run_box(a, box) for box in boxes]
    return 0 if all(c == 0 for c in codes) else 1


def run_pilot(a: argparse.Namespace, box: str) -> int:
    """22: a runtime's pilot on this box — remote_gguf.py --pilot, then what
    its pace makes of the plan, and its verdict again as the last line"""
    rt = runtime_of(a, box)
    if not rt:
        raise SystemExit("--pilot: a runtime's pilot, and this line runs none (--runtime)")
    print(f"{box}: the pilot — " + " and ".join(
        f"{fb.PILOT['n'][t]} {fb.BENCH[t]['label']}" for t in ordered(PILOT_STEP[1]))
        + " questions, thinking on, each asked once, before the other boxes are rented",
        flush=True)
    print(fb.runtime_words(rt), flush=True)
    cmd = argv_of(box, 0, PILOT_STEP, a)
    print(" ".join(shlex.quote(x) for x in cmd[1:]), flush=True)
    summary = pilot_folder(a) / "pilot.json"
    summary.unlink(missing_ok=True)             # only this pilot's is read after
    code = subprocess.run(cmd).returncode
    try:
        got = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        got = {}
    if not isinstance(got, dict):
        got = {}
    if not got.get("verdict"):
        print(f"\n{box}: the pilot didn't finish (exit {code}) — its lines above say why; paste "
              "the same line again to ask it again", flush=True)
        return code or 1
    print()
    for line in plan_at_pace(box[:1].upper(), got.get("paces") or {}):
        print(line, flush=True)
    print(got["verdict"], flush=True)
    return code


def plan_at_pace(plan: str, paces: dict) -> list[str]:
    """22: the plan's hours at the pilot's GPQA Diamond pace, against the pace
    the plan assumes for it (the 6 Oct pilot's, under today's limits) — a
    rough guide from 20 answers"""
    p = paces.get(GPQA) or {}
    if not p.get("pace_s"):
        return [f"plan {plan}: no GPQA Diamond pace from the pilot — its hours stay the plan's"]
    assumed = PACE[("on", GPQA)] * limit_factor("on", GPQA)
    ratio = p["pace_s"] / assumed
    boxes = {b: sum(hours(s) for s in st) * ratio for b, st in PLANS[plan].items()}
    total = sum(boxes.values())
    out = [f"GPQA Diamond here: about {p['pace_s']:.1f} s of the box's time an answer, "
           f"{p['at_once']} at a time, where plan {plan} assumes {assumed:.1f} (×{ratio:.2f})"]
    m = paces.get(MMLU) or {}
    if m.get("pace_s"):
        want = PACE[("on", MMLU)] * limit_factor("on", MMLU)
        out.append(f"MMLU-Pro here: about {m['pace_s']:.1f} s of the box's time an answer, "
                   f"{m['at_once']} at a time, where plan {plan} assumes {want:.1f} at 8 "
                   f"(×{m['pace_s'] / want:.2f}, from {m.get('answers', 5)} answers: for "
                   "information)")
    out.append(f"plan {plan} at GPQA Diamond's pace here: about {max(boxes.values()):.1f} h the "
               f"longest box, {total:.0f} box-hours, ${total * BOX_PRICE:,.0f} at "
               f"${BOX_PRICE:.2f} an hour — from {p['answers']} answers, a rough guide")
    return out


def line_state(a: argparse.Namespace, box: str, state: str) -> None:
    """18c point 17: a box line's own state, for the fetch — a step another
    box imported counts as done here only once this line has ended. 18e: and
    "queued" for a label waiting its turn on the box"""
    steps = box_of(box)
    if not steps:
        return
    mark = folder(a, box, 1, steps[0]).parent / f"{box}.line.json"
    try:
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text(json.dumps({"box": box, "state": state, "at": time.time()}))
    except OSError:
        pass


def run_box(a: argparse.Namespace, box: str) -> int:
    """one label's steps, one after the other: 0 when every one is whole"""
    steps = box_of(box)
    print(f"{box}: {len(steps)} step{'s' if len(steps) > 1 else ''}, about "
          f"{sum(hours(s) for s in steps):.1f} h at the pilot's paces", flush=True)
    if runtime_of(a, box):
        print(fb.runtime_words(runtime_of(a, box)), flush=True)
    line_state(a, box, "running")
    ends = []
    for k, step in enumerate(steps, 1):
        out = folder(a, box, k, step)
        if whole(out, step, a.ask_written_off):
            print(f"\n{box}, step {k} of {len(steps)}: whole already — not run again", flush=True)
            ends.append((k, step, 0, ""))
            continue
        cmd = argv_of(box, k, step, a)
        print(f"\n{box}, step {k} of {len(steps)}: {words(step)}\n"
              + " ".join(shlex.quote(x) for x in cmd[1:]), flush=True)
        began = time.time()
        code = subprocess.run(cmd).returncode
        # 17g: a step left short after asking (a question the server failed on
        # with a 5xx is kept for the next run, written off after three) runs
        # again by itself, up to RUNS — a one-benchmark step had no bundle
        # until the line was pasted a third time
        for again in range(2, RUNS + 1):
            st = state_of(out)
            # 17h: only a step this run left short, after asking — never one
            # whose progress is an earlier run's (a refusal at start-up, run
            # three times over a stale `incomplete`)
            if code == 0 or step[0] == "parity" or not st.get("incomplete") \
                    or st.get("state") != "stopped" or float(st.get("at") or 0) < began - 1:
                break
            left = ", ".join(f"{fb.BENCH[t]['label'] if t in fb.BENCH else t} "
                             f"{x.get('answers', 0):,} of {x.get('of', 0):,}"
                             for t, x in st["incomplete"].items())
            print(f"\n{box}, step {k}: not whole ({left}) — asking what is left again, run "
                  f"{again} of {RUNS}", flush=True)
            code = subprocess.run(cmd).returncode
        ends.append((k, step, code, state_of(out).get("why") or ""))
    line_state(a, box, "ended")
    print()
    for k, step, code, why in ends:
        out = folder(a, box, k, step)
        what = "its parity file is" if step[0] == "parity" else "its bundle is"
        print(f"{box}, step {k}: " + (
            f"whole — {what} in {out}" if code == 0 else
            f"not whole (exit {code})" + (f": {why}" if why else " — its lines above say why")
            + (" — paste the same line again to ask the parity questions again"
               if step[0] == "parity" else " — paste the same line again to carry on")),
            flush=True)
    return 0 if all(code == 0 for _, _, code, _ in ends) else 1

if __name__ == "__main__":
    raise SystemExit(main())
