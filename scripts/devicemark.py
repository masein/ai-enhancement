#!/usr/bin/env python3
"""12q: DeviceMark's protocol (devicemark.github.io), run here — the battery
devicemark-replica-v1, its prompts, its scorers and each row's numbers.

The battery (eval_tasks/devicemark/battery-v1.json, ids only):
  IFEval    DeviceMark's own 300 keys — the same 300 in all 12 of their raw
            files (huggingface.co/datasets/devicemark/results, raw/), whose
            301st line is each run's summary record;
  MMLU-Pro  14 a category × 14 categories of the test split, and
  MATH-500  100 across its 7 subjects in proportion (largest remainder):
            our draw of their design, seeded, from source_ids.json — their
            keys ("mmlu-biology-0") don't name dataset items. The page says
            so: IFEval, DeviceMark's items; MMLU-Pro and MATH, our draw.
The questions are read on the server from the pinned datasets (load_items),
never committed.

The protocol: 0-shot, the chat template, one user message (prompts.json);
greedy — temperature 0, seed 0; a cap of 4,096 generated tokens, thinking
included; thinking off unless the run asks for it, a row of its own. No
answer within the cap is wrong and stays in the denominator: `acc` is the
headline, `acc_answered` beside it.

The scorers: IFEval by its official checkers (scripts/ifeval_official, lm_eval
v0.4.12's), the mean of prompt- and instruction-level, strict and loose, with
random, langdetect and PYTHONHASHSEED seeded (items 1122 and 1129 draw a
letter at random otherwise); MMLU-Pro, the letter in the last \\boxed{}, else
one of a short list of unambiguous phrasings, which is recorded; MATH, the
last \\boxed{}, equal as maths (math-verify on sympy: the board's MATH-500
check, generative.math_equal).

A row: the composite (the mean of the three) with an item bootstrap, each
bench with Wilson's interval, answered % and median tokens, accuracy against
budget (their time_frontier), and the setup. Ranks: above another only when
its interval is wholly above the other's; overlapping is a tie, "=".

    python scripts/devicemark.py battery                 draw it again and compare
    python scripts/devicemark.py battery --write         …and write it
    python scripts/devicemark.py mark results/full -m served/x   score one row again
    python scripts/devicemark.py ifeval-score IN OUT     (the scorer's child)
    python scripts/devicemark.py snapshot                what the committed snapshot holds
    python scripts/devicemark.py snapshot --fetch        12q.B: DeviceMark's board.json again
                                                         (by hand: nothing fetches it on a page)
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

VERSION = "devicemark-replica-v1"
DATA_DIR = REPO / "eval_tasks" / "devicemark"
SOURCE_IDS = DATA_DIR / "source_ids.json"
BATTERY_PATH = DATA_DIR / "battery-v1.json"
PROMPTS_PATH = DATA_DIR / "prompts.json"
# 12q.B: DeviceMark's board as published, their numbers never changed
SNAPSHOT_PATH = DATA_DIR / "board-snapshot.json"
BOARD_URL = "https://devicemark.github.io/data/leaderboard/board.json"
BENCHES = ("ifeval", "mmlu_pro", "math")
LABEL = {"ifeval": "IFEval", "mmlu_pro": "MMLU-Pro", "math": "MATH"}
# the tasks a Hugging Face model sits them as, through lm_eval (build_tasks)
TASK = {"ifeval": "dm_ifeval", "mmlu_pro": "dm_mmlu_pro", "math": "dm_math"}
BENCH_OF = {t: b for b, t in TASK.items()}
DATASETS = {"ifeval": ("google/IFEval", "train"), "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "test"),
            "math": ("HuggingFaceH4/MATH-500", "test")}
CAP = 4096
SEED = 0
BUDGETS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
PER_CATEGORY, MATH_N = 14, 100
PARTS = {"pilot": {"ifeval": 10, "mmlu_pro": 10, "math": 10},
         "parity": {"ifeval": 20, "mmlu_pro": 20, "math": 10}}
PARITY_NEED = 48
BOOT_N, BOOT_SEED = 2000, 0
Z = 1.959963984540054
LETTERS = "ABCDEFGHIJ"
WHOSE = "IFEval: DeviceMark's items; MMLU-Pro and MATH: our draw of the same design"
SERVER_SPEED_LABEL = "server: RTX 5090 + CPU experts"
OUT_NAME = "devicemark.json"
ITEMS_NAME = "devicemark_items.jsonl"
PILOT_NAME = "devicemark_pilot.json"
PARITY_NAME = "devicemark_parity.json"
SPEED_NAME = "devicemark_speed.json"
DEVICE_NAME = "devicemark_device.json"


# ---------------------------------------------------------------------------
# the battery: ids only, drawn from source_ids.json with fixed seeds
# ---------------------------------------------------------------------------

def _rng(*parts) -> random.Random:
    """a generator for one draw, seeded by its name: a string seed is hashed
    with sha512, so it is the same on every machine and every PYTHONHASHSEED"""
    return random.Random(":".join((VERSION, *map(str, parts))))


def largest_remainder(counts: dict[str, int], n: int) -> dict[str, int]:
    """n shared in proportion to counts: each its floor, then one more to the
    largest remainders (exactly, as fractions) — a tie to the larger count,
    then the name"""
    total = sum(counts.values())
    raw = {k: Fraction(n * v, total) for k, v in counts.items()}
    take = {k: math.floor(x) for k, x in raw.items()}
    for k in sorted(counts, key=lambda k: (-(raw[k] - take[k]), -counts[k], k))[
            :n - sum(take.values())]:
        take[k] += 1
    return take


def draw_mmlu(by_category: dict[str, list[int]], per: int = PER_CATEGORY) -> dict[str, list[int]]:
    return {c: sorted(_rng("mmlu_pro", c).sample(sorted(ids), per))
            for c, ids in sorted(by_category.items())}


def draw_math(by_subject: dict[str, list[str]], n: int = MATH_N) -> dict[str, list[str]]:
    take = largest_remainder({s: len(v) for s, v in by_subject.items()}, n)
    return {s: sorted(_rng("math", s).sample(sorted(v), take[s]))
            for s, v in sorted(by_subject.items())}


def flat(bat: dict, bench: str) -> list[str]:
    """one bench's keys, as strings, in the battery's order"""
    v = bat[bench]
    return [str(k) for k in (v if isinstance(v, list) else [x for ks in v.values() for x in ks])]


def draw_part(bat: dict, name: str) -> dict[str, list[str]]:
    """the pilot's or the parity check's fixed items, from the battery"""
    out = {}
    for b, n in PARTS[name].items():
        keys = flat(bat, b)
        got = set(_rng(name, b).sample(keys, n))
        out[b] = [k for k in keys if k in got]
    return out


def make_battery(src: dict) -> dict:
    """the battery, drawn from the ids"""
    bat = {"version": VERSION,
           "what": "DeviceMark's protocol, replicated: ids only (the questions are read on the "
                   "server from the pinned datasets). " + WHOSE + ".",
           "sources": {"ifeval": {"dataset": "google/IFEval", "license": "Apache-2.0",
                                  "items": "DeviceMark's 300 keys: the key field of "
                                           "huggingface.co/datasets/devicemark/results "
                                           "raw/full_*_ifeval.jsonl — the same 300 in all 12 "
                                           "files, whose 301st line is a summary record"},
                       "mmlu_pro": {"dataset": "TIGER-Lab/MMLU-Pro", "split": "test",
                                    "license": "MIT",
                                    "items": f"{PER_CATEGORY} a category × 14, seeded: our draw"},
                       "math": {"dataset": "HuggingFaceH4/MATH-500", "split": "test",
                                "license": "MIT",
                                "items": f"{MATH_N} across the 7 subjects in MATH-500's "
                                         "proportions (largest remainder, a tie to the larger "
                                         "subject), seeded: our draw"}},
           "revisions": src["revisions"],
           "seed": f"random.Random('{VERSION}:<bench>:<category or subject>')",
           "ifeval": list(src["ifeval_devicemark_keys"]),
           "mmlu_pro": draw_mmlu(src["mmlu_pro"]),
           "math": draw_math(src["math500"])}
    bat["pilot"] = draw_part(bat, "pilot")
    bat["parity"] = draw_part(bat, "parity")
    return bat


@functools.lru_cache(maxsize=1)
def battery() -> dict:
    return json.loads(BATTERY_PATH.read_text(encoding="utf-8"))


def keys_for(part: str = "full", bat: dict | None = None) -> list[tuple[str, str]]:
    """(bench, key) of every item a run of this part asks, in order"""
    bat = bat or battery()
    if part in PARTS:
        return [(b, k) for b in BENCHES for k in bat[part][b]]
    return [(b, k) for b in BENCHES for k in flat(bat, b)]


# ---------------------------------------------------------------------------
# the prompts: one file, every row the same
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def prompts() -> dict:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


_FIELD = re.compile(r"\{(prompt|question|options|problem|letter|option)\}")


def _fill(template: str, **values) -> str:
    """the template's {fields} filled in one pass: a filled value is never
    read again, and \\boxed{X} and \\boxed{} are not fields"""
    return _FIELD.sub(lambda m: str(values[m.group(1)]), template)


def prompt_for(item: dict) -> str:
    p, b = prompts(), item["bench"]
    if b == "ifeval":
        return _fill(p["ifeval"], prompt=item["prompt"])
    if b == "mmlu_pro":
        opts = "\n".join(_fill(p["mmlu_pro_option"], letter=LETTERS[i], option=o)
                         for i, o in enumerate(item["options"]))
        return _fill(p["mmlu_pro"], question=item["question"], options=opts)
    return _fill(p["math"], problem=item["problem"])


def load_items(cache: Path, bat: dict | None = None) -> dict[tuple[str, str], dict]:
    """every battery item's question and key, from the pinned datasets — on
    the server, where they are fetched; kept in `cache` (beside BENCH_ROOT,
    never in the repo) once read"""
    bat = bat or battery()
    if cache.exists():
        rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x]
        if rows and rows[0].get("version") == VERSION:
            return {(r["bench"], r["key"]): r for r in rows[1:]}
    from datasets import load_dataset
    rev = bat["revisions"]
    want = {b: set(flat(bat, b)) for b in BENCHES}
    out: dict[tuple[str, str], dict] = {}
    name, split = DATASETS["ifeval"]
    for r in load_dataset(name, split=split, revision=rev[name]):
        if str(r["key"]) in want["ifeval"]:
            out[("ifeval", str(r["key"]))] = {
                "bench": "ifeval", "key": str(r["key"]), "prompt": r["prompt"],
                "instruction_id_list": list(r["instruction_id_list"]),
                "kwargs": [dict(k) for k in r["kwargs"]]}
    name, split = DATASETS["mmlu_pro"]
    for r in load_dataset(name, split=split, revision=rev[name]):
        if str(r["question_id"]) in want["mmlu_pro"]:
            opts = [o for o in r["options"] if o is not None]
            out[("mmlu_pro", str(r["question_id"]))] = {
                "bench": "mmlu_pro", "key": str(r["question_id"]), "question": r["question"],
                "options": opts, "answer": r["answer"], "category": r["category"]}
    name, split = DATASETS["math"]
    for r in load_dataset(name, split=split, revision=rev[name]):
        if r["unique_id"] in want["math"]:
            out[("math", r["unique_id"])] = {
                "bench": "math", "key": r["unique_id"], "problem": r["problem"],
                "answer": r["answer"], "subject": r["subject"]}
    missing = [f"{b}:{k}" for b in BENCHES for k in want[b] if (b, k) not in out]
    if missing:
        raise ValueError(f"the pinned datasets lack {len(missing)} battery item(s): "
                         f"{', '.join(sorted(missing)[:5])}")
    for item in out.values():
        item["text"] = prompt_for(item)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text(json.dumps({"version": VERSION}) + "\n" + "".join(
        json.dumps(out[k], ensure_ascii=False) + "\n" for k in keys_for("full", bat)),
        encoding="utf-8")
    tmp.replace(cache)
    return out


# ---------------------------------------------------------------------------
# reading an answer
# ---------------------------------------------------------------------------

def split_thinking(text: str) -> tuple[str, str]:
    """(thinking, answer): the answer is what follows the last </think>; a
    reply that opened its thinking and never closed it has no answer"""
    t = text or ""
    if "</think>" in t:
        head, _, tail = t.rpartition("</think>")
        return head.replace("<think>", "", 1).strip(), tail.strip()
    if t.lstrip().startswith("<think>"):
        return t.lstrip()[len("<think>"):].strip(), ""
    return "", t.strip()


def last_boxed(text: str) -> str | None:
    """the last \\boxed{…}, braces matched (the board's, generative.last_boxed)"""
    import generative
    return generative.last_boxed(text or "")


_WRAP = re.compile(r"\\(?:text|textbf|mathrm|mathbf|mbox|textrm)\s*\{([^{}]*)\}")
_BOX_LETTER = re.compile(r"\(?([A-Ja-j])\)?(?:\s*[.:)]\s*.*)?", re.S)
# the tested phrasings an MMLU-Pro answer may give instead of a box, each
# unambiguous: a capital A–J after them, and not the start of a word
FALLBACKS = (
    ("answer is", re.compile(r"(?i:\b(?:the\s+)?(?:final\s+|correct\s+)?answer\s+is)\s*:?\s*"
                             r"\**\s*\(?([A-J])\)?(?![A-Za-z])")),
    ("Answer:", re.compile(r"(?:^|\n)[\s*#]*(?i:(?:final\s+)?answer)\s*\**\s*:\s*\**\s*"
                           r"\(?([A-J])\)?(?![A-Za-z])")),
)


def box_letter(content: str) -> str | None:
    """the one letter a box holds: C, \\text{C}, (C), C. or "C. the option" """
    c = content or ""
    for _ in range(3):
        c = _WRAP.sub(r"\1", c)
    c = c.strip().strip("$").strip()
    m = _BOX_LETTER.fullmatch(c)
    return m.group(1).upper() if m else None


def mmlu_letter(answer: str) -> tuple[str | None, str]:
    """(the letter, how it was read): the last box's; with no box, the last of
    the tested phrasings. A letter in prose is never read"""
    b = last_boxed(answer)
    if b is not None:
        got = box_letter(b)
        return (got, "boxed") if got else (None, "")
    for how, rx in FALLBACKS:
        hits = rx.findall(answer or "")
        if hits:
            return hits[-1], how
    return None, ""


def math_answer(answer: str) -> str | None:
    """the last box's contents, tidied as the board tidies them; no box, no
    answer"""
    import generative
    b = last_boxed(answer)
    return generative._tidy(b) if b else None


def math_equal(pred: str | None, gold: str | None) -> bool:
    import generative
    return generative.math_equal(pred, gold)


# ---------------------------------------------------------------------------
# IFEval: the official checkers, seeded, in a child with PYTHONHASHSEED fixed
# ---------------------------------------------------------------------------

def _ifeval_here(rows: list[dict]) -> dict[str, dict]:
    """each row {key, instruction_id_list, kwargs, prompt, response}: the four
    verdicts, `random` and langdetect seeded before each item, so an item
    whose kwargs leave a choice to chance (1122's letter "#", 1129's "!")
    makes the same one every time"""
    import langdetect

    from ifeval_official import utils as U
    out = {}
    for r in rows:
        inp = U.InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"],
                             prompt=r["prompt"], kwargs=r["kwargs"])
        got = {}
        for name, fn in (("strict", U.test_instruction_following_strict),
                         ("loose", U.test_instruction_following_loose)):
            random.seed(SEED)
            langdetect.DetectorFactory.seed = SEED
            o = fn(inp, r["response"])
            got[name] = {"prompt": bool(o.follow_all_instructions),
                         "inst": [bool(x) for x in o.follow_instruction_list]}
        out[str(r["key"])] = got
    return out


def ifeval_verdicts(rows: list[dict]) -> dict[str, dict]:
    """_ifeval_here in a child with PYTHONHASHSEED=0: the same answers score
    the same in any process, any day"""
    if not rows:
        return {}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "PYTHONHASHSEED": str(SEED)}
        env["PYTHONPATH"] = os.pathsep.join(
            [str(HERE)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        p = subprocess.run([sys.executable, str(HERE / "devicemark.py"), "ifeval-score",
                            str(src), str(dst)], env=env, capture_output=True, text=True,
                           timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"the IFEval checker failed: {(p.stderr or p.stdout)[-600:]}")
        return json.loads(dst.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# one item, scored
# ---------------------------------------------------------------------------

def score_items(records: list[dict], items: dict[tuple[str, str], dict]) -> list[dict]:
    """each answer record {bench, key, text or answer/thinking, gen_tokens,
    capped, …}, scored: answered, the parsed answer, correct — and for IFEval
    the four verdicts. The record's own fields are kept"""
    out, ife = [], []
    for r in records:
        item = items[(r["bench"], str(r["key"]))]
        if "answer" in r:
            answer = r["answer"] or ""
        else:
            _, answer = split_thinking(r.get("text") or "")
        s = {**r, "key": str(r["key"])}
        if r["bench"] == "mmlu_pro":
            got, how = (None, "") if not answer else mmlu_letter(answer)
            s.update(parsed=got, how=how, gold=item["answer"], answered=got is not None,
                     correct=got is not None and got == item["answer"])
        elif r["bench"] == "math":
            got = math_answer(answer) if answer else None
            s.update(parsed=got, how="boxed" if got else "", gold=item["answer"],
                     answered=got is not None, correct=math_equal(got, item["answer"]))
        else:
            s.update(answered=bool(answer.strip()) and not r.get("capped"))
            ife.append({"key": item["key"], "instruction_id_list": item["instruction_id_list"],
                        "kwargs": item["kwargs"], "prompt": item["prompt"], "response": answer})
        out.append(s)
    verdicts = ifeval_verdicts(ife)
    for s in out:
        if s["bench"] == "ifeval":
            v = verdicts[s["key"]]
            s.update(ifeval=v, correct=v["strict"]["prompt"])
    return out


# ---------------------------------------------------------------------------
# a row's numbers
# ---------------------------------------------------------------------------

def wilson(p: float, n: int) -> list[float]:
    """Wilson's 95% interval for a proportion p of n"""
    if n <= 0:
        return [0.0, 0.0]
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def ifeval_mean4(scored: list[dict]) -> float:
    """the mean of prompt-level strict, instruction-level strict, prompt-level
    loose and instruction-level loose — instruction level over every
    instruction of the items given"""
    if not scored:
        return 0.0
    parts = []
    for mode in ("strict", "loose"):
        parts.append(sum(s["ifeval"][mode]["prompt"] for s in scored) / len(scored))
        inst = [x for s in scored for x in s["ifeval"][mode]["inst"]]
        parts.append(sum(inst) / len(inst) if inst else 0.0)
    return sum(parts) / 4


def bench_score(bench: str, scored: list[dict]) -> float:
    if bench == "ifeval":
        return ifeval_mean4(scored)
    return sum(bool(s["correct"]) for s in scored) / len(scored) if scored else 0.0


def composite_ci(by_bench: dict[str, list[dict]], n: int = BOOT_N,
                 seed: int = BOOT_SEED) -> list[float]:
    """the composite's 95% interval: items resampled with replacement within
    each bench, each bench scored again, the three averaged, n times"""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(sum(bench_score(b, [rng.choice(v) for _ in v]) for b, v in
                        by_bench.items()) / len(by_bench))
    vals.sort()
    return [round(vals[int(0.025 * (n - 1))], 4), round(vals[int(math.ceil(0.975 * (n - 1)))], 4)]


def time_frontier(scored: list[dict], budgets=BUDGETS) -> dict:
    """what the score would be at each cap, from each answer's own length: an
    item counts at budget b when it is right and was written within b tokens.
    As theirs, pooled over the MMLU-Pro and MATH items; each bench beside"""
    def at(items):
        return [round(sum(bool(s["correct"]) and (s.get("gen_tokens") or 0) <= b
                          for s in items) / len(items), 4) if items else None for b in budgets]
    pooled = [s for s in scored if s["bench"] in ("mmlu_pro", "math")]
    return {"b": list(budgets), "acc": at(pooled),
            "by_bench": {b: at([s for s in scored if s["bench"] == b])
                         for b in ("mmlu_pro", "math")}}


def _median(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def summarize(scored: list[dict], setup: dict, part: str = "full") -> dict:
    """a row: the composite and its interval, the three benches with theirs,
    answered and median tokens per bench and overall, accuracy against budget,
    and the setup it ran on"""
    by = {b: [s for s in scored if s["bench"] == b] for b in BENCHES}
    benches = {}
    for b, items in by.items():
        n = len(items)
        acc = bench_score(b, items)
        answered = [s for s in items if s["answered"]]
        b_out = {"n": n, "acc": round(acc, 4), "ci": wilson(acc, n),
                 "answered": len(answered),
                 "answered_pct": round(len(answered) / n, 4) if n else None,
                 "capped": sum(bool(s.get("capped")) for s in items),
                 "median_tokens": _median([s.get("gen_tokens") for s in items])}
        if b == "ifeval":
            b_out["parts"] = {f"{lvl}_{mode}": round(
                (sum(s["ifeval"][mode]["prompt"] for s in items) / n if lvl == "prompt" else
                 (lambda xs: sum(xs) / len(xs) if xs else 0.0)(
                     [x for s in items for x in s["ifeval"][mode]["inst"]])), 4)
                for lvl in ("prompt", "inst") for mode in ("strict", "loose")} if n else {}
        else:
            b_out["acc_answered"] = (round(sum(bool(s["correct"]) for s in answered)
                                           / len(answered), 4) if answered else None)
            b_out["fallbacks"] = {h: sum(s.get("how") == h for s in items)
                                  for h in sorted({s.get("how") for s in items} - {"", None})}
        benches[b] = b_out
    full = all(by[b] for b in BENCHES)
    comp = round(sum(benches[b]["acc"] for b in BENCHES) / 3, 4) if full else None
    return {
        "version": VERSION, "part": part, "at": time.time(), "setup": setup,
        "composite": {"value": comp, "ci": composite_ci(by) if full else None},
        "benches": benches,
        "answered_pct": round(sum(b["answered"] for b in benches.values())
                              / max(1, len(scored)), 4),
        "median_tokens": _median([s.get("gen_tokens") for s in scored]),
        "time_frontier": time_frontier(scored),
        # 12q.F: the items the server failed on — answered without its chat
        # parsing (scored as any), and those with no answer after that too
        "raw_fallback": sum(bool(s.get("raw_fallback")) for s in scored),
        "errors": sum(bool(s.get("error")) for s in scored),
        "n": len(scored), "whose": WHOSE}


def ranks(rows: list[dict]) -> list[str]:
    """each row's rank by composite, theirs: 1 + the rows strictly better —
    whose interval's lower end is above this one's upper end. A rank more
    than one row shares is a tie, "=3" """
    got = []
    for r in rows:
        lo_hi = r["ci"]
        got.append(1 + sum(1 for o in rows if o is not r and o["ci"][0] > lo_hi[1]))
    return [f"={g}" if got.count(g) > 1 else str(g) for g in got]


# ---------------------------------------------------------------------------
# MTP parity: the same items, with and without MTP, greedy both ways
# ---------------------------------------------------------------------------

def same_answer(a: dict, b: dict) -> bool:
    if a["bench"] == "ifeval":
        return a["ifeval"] == b["ifeval"]
    if a["bench"] == "math":
        return (a["parsed"] == b["parsed"]
                or (a["parsed"] is not None and math_equal(a["parsed"], b["parsed"])))
    return a["parsed"] == b["parsed"]


def parity(mtp: list[dict], plain: list[dict]) -> dict:
    """how many outputs are the same token for token (the same text: the one
    model, the one tokenizer), and how many answers are the same; at least
    PARITY_NEED of the 50 alike, and the setup without MTP takes its quality
    from the MTP row"""
    theirs = {(s["bench"], s["key"]): s for s in plain}
    rows = []
    for s in mtp:
        o = theirs.get((s["bench"], s["key"]))
        if o is None:
            continue
        rows.append({"bench": s["bench"], "key": s["key"],
                     "identical": (s.get("text") or "") == (o.get("text") or ""),
                     "same_answer": same_answer(s, o)})
    same = sum(r["same_answer"] for r in rows)
    return {"n": len(rows), "identical": sum(r["identical"] for r in rows), "same_answer": same,
            "need": PARITY_NEED, "passes": len(rows) == sum(PARTS["parity"].values())
            and same >= PARITY_NEED, "items": rows}


def inherit(own: dict | None, mtp_row: dict | None, report: dict | None) -> dict | None:
    """a setup without MTP's quality: its own full run when it has one (the
    switch that runs it anyway); else the MTP row's, when the parity check
    passed on the same thinking mode — marked so"""
    if own and own.get("part") == "full":
        return own
    if not (mtp_row and report and report.get("passes")):
        return own
    if bool(report.get("thinking")) != bool((mtp_row.get("setup") or {}).get("thinking")):
        return own
    return {**mtp_row, "inherited": {"from": report["mtp"], "parity": report["same_answer"],
                                     "of": report["n"],
                                     "line": f"quality from MTP run, parity "
                                             f"{report['same_answer']}/{report['n']}"}}


# ---------------------------------------------------------------------------
# a Hugging Face model: three lm_eval tasks, answered on hf
# ---------------------------------------------------------------------------

def build_tasks(dest: Path, items: dict[tuple[str, str], dict], part: str = "full") -> Path:
    """the three tasks lm_eval asks a Hugging Face model: each item's prompt as
    the one user message (the chat template applied), greedy, the cap of
    4,096 generated tokens, nothing to stop on but the end of the turn"""
    dest.mkdir(parents=True, exist_ok=True)
    for b in BENCHES:
        rows = [items[(bb, k)] for bb, k in keys_for(part) if bb == b]
        (dest / f"{TASK[b]}.jsonl").write_text("".join(
            json.dumps({"bench": b, "key": r["key"], "text": r["text"]}, ensure_ascii=False)
            + "\n" for r in rows), encoding="utf-8")
        (dest / f"{TASK[b]}.yaml").write_text(
            f"task: {TASK[b]}\n"
            "dataset_path: json\n"
            f"dataset_kwargs:\n  data_files:\n    test: {dest / (TASK[b] + '.jsonl')}\n"
            "test_split: test\n"
            "output_type: generate_until\n"
            "doc_to_text: '{{text}}'\n"
            "doc_to_target: ''\n"
            "generation_kwargs:\n"
            "  until: []\n"
            "  do_sample: false\n"
            "  temperature: 0.0\n"
            f"  max_gen_toks: {CAP}\n"
            "metric_list:\n  - metric: bypass\n"
            "metadata:\n  version: 1.0\n", encoding="utf-8")
    return dest


def stand_in_items(part: str = "full") -> dict[tuple[str, str], dict]:
    """each battery key with "x" for its question: enough for build_tasks when
    only the tasks' names are asked about (scripts/check_tasks.py)"""
    return {(b, k): {"bench": b, "key": k, "text": "x"} for b, k in keys_for(part)}


def task_keys(tasks_dir: Path, task: str) -> set[str]:
    """the battery items a built task holds (build_tasks' file)"""
    try:
        text = (tasks_dir / f"{task}.jsonl").read_text(encoding="utf-8")
    except OSError:
        return set()
    return {str(json.loads(x)["key"]) for x in text.splitlines() if x.strip()}


def answered_keys(task_out: Path, task: str) -> set[str]:
    """the battery items a run's saved answers to `task` cover: its newest
    samples file, the one records_from_samples scores. lm_eval writes it
    whole, once the task is answered; a line cut short is no answer"""
    files = sorted(task_out.rglob(f"samples_{task}_*.jsonl"))
    got: set[str] = set()
    for line in files[-1].read_text(encoding="utf-8").splitlines() if files else []:
        try:
            key = (json.loads(line).get("doc") or {}).get("key")
        except ValueError:
            continue
        if key is not None:
            got.add(str(key))
    return got


def records_from_samples(row_dir: Path, token_count=None) -> list[dict]:
    """each answer a Hugging Face run logged, as a record: its length counted
    again from the text with the model's tokenizer (`token_count`), and
    capped when that reaches the cap"""
    out = []
    for b in BENCHES:
        files = sorted((row_dir / f"{TASK[b]}_0shot").rglob(f"samples_{TASK[b]}_*.jsonl"))
        if not files:
            continue
        for line in files[-1].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            text = (s.get("resps") or [[""]])[0][0] or ""
            n = token_count(text) if token_count else None
            out.append({"bench": b, "key": str(s["doc"]["key"]), "text": text,
                        "gen_tokens": n, "capped": bool(n is not None and n >= CAP - 2),
                        "tokens_from": "the text, counted again with the model's tokenizer"})
    return out


# ---------------------------------------------------------------------------
# on disk: a row's answers and its numbers
# ---------------------------------------------------------------------------

def read_items(row_dir: Path, name: str = ITEMS_NAME) -> list[dict]:
    f = row_dir / name
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def mark(row_dir: Path, items: dict[tuple[str, str], dict], setup: dict,
         records: list[dict] | None = None, part: str = "full") -> dict:
    """a row scored from its answers: the scored answers beside it
    (devicemark_items.jsonl, the question browser's), its numbers in
    devicemark.json (or the pilot's own file)"""
    records = records if records is not None else read_items(row_dir)
    order = {k: i for i, k in enumerate(keys_for(part))}
    records = sorted((r for r in records if (r["bench"], str(r["key"])) in order),
                     key=lambda r: order[(r["bench"], str(r["key"]))])
    scored = score_items(records, items)
    row = summarize(scored, setup, part)
    row_dir.mkdir(parents=True, exist_ok=True)
    if part == "full":
        (row_dir / ITEMS_NAME).write_text("".join(json.dumps(s, ensure_ascii=False) + "\n"
                                                  for s in scored), encoding="utf-8")
    (row_dir / (OUT_NAME if part == "full" else PILOT_NAME)).write_text(
        json.dumps(row if part == "full" else {**row, "items": scored}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    return row


def rows(out_dir: Path, launch=None) -> list[dict]:
    """every row on the board that sat the battery: its numbers, its speeds
    (the server's and a device's), a setup without MTP's quality from its
    MTP partner when the parity check allows — and the ranks. 12z A1: a
    served setup's lookahead and MTP are what its launch says now — `launch`,
    model id -> {lookahead, mtp} (its registration, service side), else the
    flags recorded with the row — never words in its description"""
    got = {}
    for d in sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.is_dir() else []:
        f = d / OUT_NAME
        row = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        rep = d / PARITY_NAME
        report = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else None
        if row is None and not (report and report.get("plain") == _model_of(d)):
            continue
        got[d.name] = {"dir": d.name, "row": row, "report": report}
    out = []
    for name, g in got.items():
        row, report = g["row"], g["report"]
        if report and report.get("plain") == _model_of(out_dir / name):
            mtp = got.get(report["mtp"].replace("/", "__") + ("__thinking" if report.get(
                "thinking") else ""), {}).get("row")
            row = inherit(row, mtp, report)
        if not row or row.get("composite", {}).get("value") is None:
            continue
        mid = _model_of(out_dir / name)
        su = row.get("setup") or {}
        if su.get("runtime") == "llama-server":
            # its registration's launch; with none, the flags recorded with the
            # row where they say anything, else what the row recorded
            lk = launch(mid) if launch else None
            if lk is None:
                lk = launch_of(su.get("server_flags") or "")
                if not lk["said"]:
                    lk = {"lookahead": su.get("lookahead"), "mtp": su.get("mtp")}
            row = {**row, "setup": {**su, "lookahead": bool(lk.get("lookahead")),
                                    "mtp": bool(lk.get("mtp"))}}
        base = out_dir / name.removesuffix("__thinking")
        speed = _json(base / SPEED_NAME)
        device = _json(base / DEVICE_NAME)
        thinks = name.endswith("__thinking")
        out.append({"id": mid + (" · thinking" if thinks else ""), "model": mid,
                    "thinking": thinks, "row": row,
                    "label": setup_label(row.get("setup") or {}, thinks, mid),
                    "server_tok_s": speed.get("decode_tok_s") if speed else None,
                    "server_label": SERVER_SPEED_LABEL if speed else None,
                    "device": device or None})
    rk = ranks([{"ci": r["row"]["composite"]["ci"]} for r in out])
    for r, k in zip(out, rk):
        r["rank"] = k
    for r in out:
        r["retention"] = retention(r, out)
    return out


def retention(r: dict, rows_: list[dict]) -> dict | None:
    """12q.B: a phone build's scores over the original's, per bench — on the
    same battery (so the same items), the same thinking mode, neither with
    lookahead — and only when both have run"""
    su = r["row"].get("setup") or {}
    if not su.get("phone") or su.get("lookahead") or su.get("runtime") != "llama-server":
        return None

    def orig(o):
        so = o["row"].get("setup") or {}
        return (so.get("runtime") == "llama-server" and not so.get("phone")
                and not so.get("lookahead") and bool(so.get("thinking")) == bool(su.get("thinking"))
                and o["row"].get("version") == r["row"].get("version"))
    cands = sorted((o for o in rows_ if o is not r and orig(o)),
                   key=lambda o: (bool(o["row"].get("inherited")),
                                  bool((o["row"].get("setup") or {}).get("mtp")), o["id"]))
    if not cands:
        return None
    o = cands[0]
    got = {}
    for b in BENCHES:
        mine, theirs = r["row"]["benches"][b]["acc"], o["row"]["benches"][b]["acc"]
        got[b] = round(mine / theirs, 4) if theirs else None
    return {"of": o["id"], **got}


# ---------------------------------------------------------------------------
# 12q.B: DeviceMark's own rows, from the committed snapshot
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def external_rows(snap: dict | None = None) -> list[dict]:
    """their rows, as the page reads a row — every number theirs, as
    published, read-only; where each came from said on it"""
    snap = snap or snapshot()
    out = []
    for r in snap["rows"]:
        q = r["quality"]
        kind = ("cloud" if r.get("format") == "api" else
                "system" if r.get("format") == "system" else "device")
        n = (q.get("mmlu_n") or 0) + (q.get("math_n") or 0)
        answered = (q.get("mmlu_answered") or 0) + (q.get("math_answered") or 0)
        out.append({
            "id": r["artifact_id"], "external": True, "kind": kind,
            "name": f"{r['model']} ({r['quant']})" if kind == "device" else r["model"],
            "model": r["model"],
            # what theirs is, beside ours — 12z A3: DeviceMark scores quality on
            # a Mac (the bundle on Apple's engine, or LiteRT) and takes only the
            # speed, and a short word-for-word check, from the phone: "int8,
            # scored on a Mac; speed on iPhone 17 Pro"
            "label": (f"{QUANTS.get(r['quant'], r['quant'])}, scored on {SCORED_ON}; "
                      f"speed on {snap.get('device') or 'the device'}" if kind == "device"
                      else r["model"]),
            "vendor": r.get("vendor"), "params_b": r.get("params_b"),
            "composite": {"value": r["composite"]["value"], "ci": r["composite"]["ci"]},
            "benches": {
                "ifeval": {"acc": q.get("ifeval_mean4"), "ci": q.get("ifeval_ci"),
                           "n": q.get("ifeval_n")},
                "mmlu_pro": {"acc": q.get("mmlu_acc"), "ci": q.get("mmlu_ci"),
                             "n": q.get("mmlu_n"), "answered": q.get("mmlu_answered"),
                             "acc_answered": q.get("mmlu_acc_answered")},
                "math": {"acc": q.get("math_acc"), "ci": q.get("math_ci"), "n": q.get("math_n"),
                         "answered": q.get("math_answered"),
                         "acc_answered": q.get("math_acc_answered")}},
            "answered_pct": round(answered / n, 4) if n else None,
            "answered_of": "MMLU-Pro and MATH",
            "median_tokens": q.get("gen_tokens_median"),
            "time_frontier": q.get("time_frontier"),
            "cap": q.get("cap_tokens"), "battery": q.get("battery_version"),
            "device": ({"tok_s": r["iphone_tok_s"], "device": snap.get("device"),
                        "source": "DeviceMark"} if r.get("iphone_tok_s") else None),
            "retention": r.get("retention"), "note": (snap.get("notes") or {}).get(
                r["artifact_id"]),
            "credit": snap["credit"]})
    return out


# their open models, and the Hugging Face repo each is published on. Our run
# of one on hf (bf16, our battery) sits beside their row — in the table and in
# their point's hover — and is never a point of its own at their device's
# speed, which is their quantized build's
THEIR_OPEN = {
    "lfm2.5-1.2b__int8hu__aimodel": "LiquidAI/LFM2.5-1.2B-Instruct",
    "granite-4.0-h-1b__int8hu__aimodel": "ibm-granite/granite-4.0-h-1b",
    "qwen3.5-0.8b__int8hu__aimodel": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b__int8hu__aimodel": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b__int8hu__aimodel": "Qwen/Qwen3.5-4B",
    "gemma-4-e2b__int4__litertlm": "google/gemma-4-E2B-it",
    "nemotron-4b__int8hu__aimodel": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "nanbeige-3b__int8hu__aimodel": "Nanbeige/Nanbeige4.1-3B",
    "youtu-2b__int8__aimodel": "tencent/Youtu-LLM-2B",
}
# the calibration: how close our measurement comes to theirs, on the models
# whose raw files match DeviceMark's board (12q.D: Nemotron-3-Nano-4B can't
# run here — CANT_RUN_PATH)
CALIBRATION = ("Qwen/Qwen3.5-4B", "Nanbeige/Nanbeige4.1-3B", "tencent/Youtu-LLM-2B")
CANT_RUN_PATH = REPO / "service" / "cant_run_here.json"


def cant_run_here() -> dict[str, str]:
    """12q.D: hf id -> why this server can't run it (service/cant_run_here.json)"""
    try:
        return {k: str(v.get("why") or "") for k, v in json.loads(
            CANT_RUN_PATH.read_text(encoding="utf-8")).items() if isinstance(v, dict)}
    except (OSError, ValueError):
        return {}


DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
# 12z A3: where DeviceMark scores a device row's quality (its methodology page)
SCORED_ON = "a Mac"
QUANTS = {"int8hu": "int8"}


def _beside(r: dict) -> dict:
    """a row of ours as it sits beside theirs: its numbers and how it ran"""
    row, su = r["row"], r["row"].get("setup") or {}
    return {"id": r["id"], "model": r["model"], "thinking": r["thinking"],
            "label": f"{DTYPES.get(su.get('dtype'), su.get('dtype') or 'hf')}, our battery"
                     + (", thinking" if r["thinking"] else ""),
            "calibration": r["model"] in CALIBRATION,
            **{k: row.get(k) for k in ("composite", "benches", "answered_pct", "median_tokens",
                                       "time_frontier")}}


# 12q.E: two runs of one model whose median answers differ by more than this
# ran in different modes — one reasoned, the other didn't — and their scores
# aren't a calibration point (Youtu-LLM-2B: ours 370 tokens, theirs 2,893)
MODE_RATIO = 2.0


def modes_differ(ours: dict, theirs: dict) -> str:
    """'' unless the two runs' median tokens differ by more than MODE_RATIO;
    then the sentence that says both"""
    a, b = ours.get("median_tokens"), theirs.get("median_tokens")
    if not a or not b or max(a, b) <= MODE_RATIO * min(a, b):
        return ""

    def pct(v):
        return "—" if v is None else f"{round(100 * v)}%"
    return (f"the modes differ: ours answered {pct(ours.get('answered_pct'))} at a median of "
            f"{round(a):,} tokens, theirs {pct(theirs.get('answered_pct'))} at {round(b):,} — "
            "not a calibration point")


def intervals_overlap(a: dict | None, b: dict | None) -> bool | None:
    """whether two composites' 95% intervals overlap; None where one has none"""
    ca, cb = (a or {}).get("ci"), (b or {}).get("ci")
    if not ca or not cb:
        return None
    return ca[0] <= cb[1] and cb[0] <= ca[1]


def pair(ours: list[dict], theirs: list[dict]) -> None:
    """each of our hf runs of one of their open models, beside their row
    (`ours` on it, thinking off first) and out of our own rows (`paired`)"""
    by_hf = {THEIR_OPEN[t["id"]]: t for t in theirs if t["id"] in THEIR_OPEN}
    # 12q.D: a model of theirs this server can't run says so on their row
    cant = cant_run_here()
    for hf, t in by_hf.items():
        if cant.get(hf):
            t["cant_run"] = cant[hf]
    for r in sorted(ours, key=lambda r: r["thinking"]):
        t = by_hf.get(r["model"])
        if t and str((r["row"].get("setup") or {}).get("runtime", "")).startswith("hf"):
            r["paired"] = t["id"]
            o = _beside(r)
            # 12q.E: a pair in different modes is said so, and isn't calibration;
            # any other says whether the two intervals overlap
            o["mode_differs"] = modes_differ(o, t)
            o["within"] = None if o["mode_differs"] else intervals_overlap(
                o.get("composite"), t.get("composite"))
            o["calibration"] = o["calibration"] and not o["mode_differs"]
            t.setdefault("ours", []).append(o)


# 12z A1: a setup's lookahead and MTP, from its launch flags and environment
# — `--spec-type draft-mtp`, `LLAMA_MOE_ROUTE_MODE=lookahead` — read as flags
# and VAR=value, never as words: "routing local (no lookahead)" is not
# lookahead, and "MTP-GGUF" in a file's name is not MTP
_ENV_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.:/+-]+)")
ROUTE_MODE, ROUTE_LOOKAHEAD = "LLAMA_MOE_ROUTE_MODE", "LLAMA_MOE_ROUTE_LOOKAHEAD"


def launch_of(*texts: str) -> dict:
    """{"lookahead", "mtp", "said"}: what these launch flags and environment
    say — `said` when they say either at all"""
    text = " ".join(t for t in texts if t)
    env = dict(m.groups() for m in _ENV_TOKEN.finditer(text))
    toks = [t for t in re.split(r"[\s,;()]+", text) if t]
    spec = None
    for i, t in enumerate(toks):
        if t.startswith("--spec-type="):
            spec = t.split("=", 1)[1]
        elif t == "--spec-type" and i + 1 < len(toks):
            spec = toks[i + 1]
    mode = env.get(ROUTE_MODE, "").lower()
    return {"lookahead": mode == "lookahead" and env.get(ROUTE_LOOKAHEAD, "1") != "0",
            "mtp": bool(spec) and "mtp" in spec.lower(),
            "said": spec is not None or ROUTE_MODE in env}


# the build a served setup is: "k4-LDA" in a phone build's name or file, an
# original's "k=8"
_BUILD = re.compile(r"\bk\d+-[A-Za-z]+\b")
_TOP_K = re.compile(r"\bk\s*[-=]?\s*(\d+)\b")


def setup_label(su: dict, thinking: bool, model: str = "") -> str:
    """12q.B2: a served setup as the chart names it — "phone build (k4-LDA) ·
    MTP · thinking off"; any other row, its model"""
    if su.get("runtime") != "llama-server":
        return model + (" · thinking" if thinking else "")
    where = f"{su.get('name') or ''} {su.get('file') or ''} {model}"
    b = _BUILD.search(where)
    k = None if b else _TOP_K.search(where)
    tag = b.group(0) if b else f"k={k.group(1)}" if k else ""
    return " · ".join(x for x in (
        ("phone build" if su.get("phone") else "original") + (f" ({tag})" if tag else ""),
        "MTP" if su.get("mtp") else "", "lookahead" if su.get("lookahead") else "",
        f"thinking {'on' if thinking else 'off'}") if x)


def board(out_dir: Path, launch=None) -> dict:
    """ours and theirs, ranked together by their rule — our runs of their
    models beside their rows, never ranked apart. 12q.B2: the cloud APIs
    aren't ranked (their "☁" rows), as DeviceMark's board doesn't"""
    ours, theirs = rows(out_dir, launch), external_rows()
    pair(ours, theirs)
    solo = [r for r in ours if not r.get("paired")]
    ranked = [t for t in theirs if t["kind"] != "cloud"]
    rk = ranks([{"ci": r["row"]["composite"]["ci"]} for r in solo]
               + [{"ci": r["composite"]["ci"]} for r in ranked])
    for r in ours + theirs:
        r["rank_all"] = None
    for r, k in zip(solo + ranked, rk):
        r["rank_all"] = k
    s = snapshot()
    return {"rows": ours, "external": {"rows": theirs, **{k: s.get(k) for k in (
        "source", "last_modified", "fetched", "credit", "licence", "device")}}}


# ---------------------------------------------------------------------------
# 12q.C: a model's DeviceMark runs, as its page reads them
# ---------------------------------------------------------------------------

PARITY_ANSWERS = "devicemark_parity_{}_answers.jsonl"
BENCH_KEYS = ("acc", "ci", "n", "answered", "capped", "median_tokens")


def _benches(row: dict) -> dict:
    return {b: {k: ((row.get("benches") or {}).get(b) or {}).get(k) for k in BENCH_KEYS}
            for b in BENCHES}


def _said(bench: str, rec: dict | None):
    """what an answer says, read without its gold: MMLU-Pro's letter, MATH's
    boxed answer; IFEval has none to read"""
    if not rec or bench == "ifeval":
        return None
    ans = rec.get("answer")
    if ans is None:
        _, ans = split_thinking(rec.get("text") or "")
    if not ans:
        return None
    return mmlu_letter(ans)[0] if bench == "mmlu_pro" else math_answer(ans)


def _parity_card(rep: dict, out_dir: Path, suffix: str) -> dict:
    """the parity check, and each pair that differs with what each setup
    answered"""
    def answers(model: str, tag: str) -> dict:
        d = out_dir / (str(model).replace("/", "__") + suffix)
        return {(r["bench"], str(r["key"])): r
                for r in read_items(d, PARITY_ANSWERS.format(tag))}
    a, b = answers(rep.get("mtp"), "mtp"), answers(rep.get("plain"), "plain")
    differ = []
    for it in rep.get("items") or []:
        if it.get("identical") and it.get("same_answer"):
            continue
        k = (it["bench"], str(it["key"]))
        differ.append({"bench": it["bench"], "key": str(it["key"]),
                       "identical": bool(it.get("identical")),
                       "same_answer": bool(it.get("same_answer")),
                       "mtp": _said(it["bench"], a.get(k)), "plain": _said(it["bench"], b.get(k)),
                       "mtp_tokens": (a.get(k) or {}).get("gen_tokens"),
                       "plain_tokens": (b.get(k) or {}).get("gen_tokens")})
    return {**{k: rep.get(k) for k in ("n", "identical", "same_answer", "need", "passes", "mtp",
                                       "plain", "thinking", "at")}, "differ": differ}


def model_runs(out_dir: Path, launch=None) -> dict[str, dict]:
    """each model with a DeviceMark run, by thinking mode ("off", "on"): its
    full row as the board ranks it — and its row on the On-device chart, its
    own or DeviceMark's beside it — then its pilot, its parity check and its
    speed test. The model page's DeviceMark cards, and what "Open results"
    opens"""
    if not out_dir.is_dir():
        return {}
    b = board(out_dir, launch)
    full = {r["id"]: r for r in b["rows"]}
    ext = {t["id"]: t for t in b["external"]["rows"]}
    out: dict[str, dict] = {}
    for d in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        mode = "on" if d.name.endswith("__thinking") else "off"
        model = _model_of(d)
        rid = model + (" · thinking" if mode == "on" else "")
        r = full.get(rid)
        pilot, rep = _json(d / PILOT_NAME), _json(d / PARITY_NAME)
        speed = _json(d / SPEED_NAME) if mode == "off" else None
        if not (r or pilot or rep or speed):
            continue
        card: dict = {"id": rid, "row": None, "pilot": None, "parity": None, "speed": None}
        if r:
            row = r["row"]
            card["row"] = {
                "composite": row.get("composite"), "benches": _benches(row),
                "answered_pct": row.get("answered_pct"), "median_tokens": row.get("median_tokens"),
                "n": row.get("n"), "rank": r.get("rank_all"), "label": r.get("label"),
                "device": r.get("device"), "server_tok_s": r.get("server_tok_s"),
                "server_label": r.get("server_label"), "inherited": row.get("inherited"),
                "paired": r.get("paired"), "chart_id": r.get("paired") or r["id"],
                "items": (d / ITEMS_NAME).exists(), "at": row.get("at"),
                "raw_fallback": row.get("raw_fallback") or 0, "errors": row.get("errors") or 0,
                "version": row.get("version")}
            # 12z A5: DeviceMark's own row beside ours, as the chart's table has it:
            # their number, and whether the two are a calibration point
            t = ext.get(r.get("paired") or "")
            if t:
                o = next((x for x in t.get("ours") or [] if x.get("id") == rid), {})
                card["row"]["theirs"] = {
                    "id": t["id"], "name": t["name"], "label": t.get("label"),
                    "composite": t.get("composite"), "mode_differs": o.get("mode_differs") or "",
                    "within": o.get("within"), "calibration": bool(o.get("calibration"))}
        if pilot:
            cc = pilot.get("cap_check") or {}
            card["pilot"] = {"composite": pilot.get("composite"), "benches": _benches(pilot),
                             "answered_pct": pilot.get("answered_pct"),
                             "median_tokens": pilot.get("median_tokens"),
                             "n": pilot.get("n") or len(pilot.get("items") or []),
                             "cap_check": {"ok": cc.get("ok"), "line": cc.get("line")},
                             "at": pilot.get("at")}
        if rep:
            card["parity"] = _parity_card(rep, out_dir, "__thinking" if mode == "on" else "")
        if speed:
            card["speed"] = {k: speed.get(k) for k in ("decode_tok_s", "trials", "prompt_tokens",
                                                       "decode_tokens", "label", "at")}
        out.setdefault(model, {})[mode] = card
    return out


def fetch_snapshot() -> dict:
    """DeviceMark's board.json again — by hand, never from a page; the notes
    we keep beside their rows stay"""
    import urllib.request
    with urllib.request.urlopen(BOARD_URL, timeout=60) as r:
        rows_ = json.loads(r.read())
        modified = r.headers.get("last-modified")
    old = snapshot() if SNAPSHOT_PATH.exists() else {}
    snap = {**old, "source": BOARD_URL, "last_modified": modified,
            "fetched": time.strftime("%Y-%m-%d"), "rows": rows_}
    SNAPSHOT_PATH.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    snapshot.cache_clear()
    return snap


def _json(f: Path) -> dict | None:
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_of(d: Path) -> str:
    return d.name.removesuffix("__thinking").replace("__", "/", 1)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("battery", help="draw the battery again from source_ids.json")
    b.add_argument("--write", action="store_true")
    s = sub.add_parser("ifeval-score")
    s.add_argument("src", type=Path)
    s.add_argument("dst", type=Path)
    sn = sub.add_parser("snapshot", help="12q.B: DeviceMark's board, as committed")
    sn.add_argument("--fetch", action="store_true", help="fetch board.json again and write it")
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        snap = fetch_snapshot() if a.fetch else snapshot()
        print(f"{len(snap['rows'])} rows from {snap['source']} (last modified "
              f"{snap.get('last_modified')}, fetched {snap.get('fetched')}) · {snap['credit']}")
        for r in external_rows(snap):
            print(f"  {r['id']:40s} {r['composite']['value']:.4f} {r['composite']['ci']}")
        return 0
    if a.cmd == "ifeval-score":
        rows_in = json.loads(a.src.read_text(encoding="utf-8"))
        a.dst.write_text(json.dumps(_ifeval_here(rows_in)), encoding="utf-8")
        return 0
    bat = make_battery(json.loads(SOURCE_IDS.read_text(encoding="utf-8")))
    if a.write:
        BATTERY_PATH.write_text(json.dumps(bat, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {BATTERY_PATH.relative_to(REPO)}: " + ", ".join(
            f"{LABEL[k]} {len(flat(bat, k))}" for k in BENCHES))
        return 0
    same = BATTERY_PATH.exists() and json.loads(BATTERY_PATH.read_text(encoding="utf-8")) == bat
    print("the committed battery is the draw" if same else "the committed battery differs")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
