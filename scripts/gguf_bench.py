"""12f.3: the benchmarks measured on a GGUF file by llama.cpp's llama-perplexity.

One table, so a later benchmark (a Persian multiple-choice set, or MMLU-Pro as
multiple choice) is a new row and its converter, not new code. The standard
library only: the worker on the host imports it.

Everything here is read from the fork's own source, tools/perplexity/
perplexity.cpp and common/arg.cpp (teraformer/lda-2026-09-22, 91428471f):

- **The modes.** `--hellaswag` reads six lines a task (context, the right
  ending's index, four endings); `--winogrande` reads CSV rows
  (index,sentence,choice 1,choice 2,answer 1|2); `--multiple-choice` reads a
  binary file of tasks (see gguf_data.py). The text formats go in with `-f`,
  the binary with `-bf`.
- **How many.** `--hellaswag-tasks` is 400 unless given, so a full run always
  says how many; `--winogrande-tasks` and `--multiple-choice-tasks` are 0 (all)
  unless given. Fewer than all is a random choice with llama-perplexity's own
  fixed seed (std::mt19937 rng(1)): the same subset every time.
- **What it prints.** A line a task — the number done, then the running
  accuracy — and at the end "Final Winogrande score(N tasks): X +/- Y" or
  "Final result: X +/- Y" with "Random chance: …". HellaSwag prints no final
  line: its score is the last task line's. Percentages throughout.
- **How it scores.** Each answer's mean log-probability a token after the
  question (or context) and a space; the best is its pick. No examples before
  the question: 0-shot, and not lm_eval's prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

GROUP = "Measured on the GGUF · llama.cpp, 0-shot"
TIP = ("Scored by llama.cpp's llama-perplexity on the quantised file. Not comparable with the "
       "lm_eval columns to its left: different prompts and no examples.")

# key -> label, llama-perplexity's mode, the dataset file under gguf_data/, the
# lm_eval task (and split) whose questions it holds, and how many that is
BENCHMARKS = {
    "mmlu": {"label": "MMLU", "mode": "multiple-choice", "data": "mmlu-test.bin",
             "lm_eval": "mmlu", "split": "test", "n": 14042},
    "hellaswag": {"label": "HellaSwag", "mode": "hellaswag", "data": "hellaswag-validation.txt",
                  "lm_eval": "hellaswag", "split": "validation", "n": 10042},
    "winogrande": {"label": "Winogrande", "mode": "winogrande",
                   "data": "winogrande-validation.csv", "lm_eval": "winogrande",
                   "split": "validation", "n": 1267},
    "arc_challenge": {"label": "ARC-C", "mode": "multiple-choice", "data": "arc-challenge-test.bin",
                      "lm_eval": "arc_challenge", "split": "test", "n": 1172},
    "arc_easy": {"label": "ARC-E", "mode": "multiple-choice", "data": "arc-easy-test.bin",
                 "lm_eval": "arc_easy", "split": "test", "n": 2376},
    "truthfulqa": {"label": "TruthfulQA", "mode": "multiple-choice",
                   "data": "truthfulqa-mc1-validation.bin", "lm_eval": "truthfulqa_mc1",
                   "split": "validation", "n": 817,
                   "note": "MC1, one right answer: llama-perplexity scores no MC2"},
}
ORDER = list(BENCHMARKS)
MODES = {
    "hellaswag": {"flag": "--hellaswag", "tasks": "--hellaswag-tasks", "file": "-f",
                  "chance": 0.25},
    "winogrande": {"flag": "--winogrande", "tasks": "--winogrande-tasks", "file": "-f",
                   "chance": 0.5},
    "multiple-choice": {"flag": "--multiple-choice", "tasks": "--multiple-choice-tasks",
                        "file": "-bf", "chance": None},
}
DEFAULT_FLAGS = ["-ngl", "99", "--cpu-moe"]     # as the servers run: it fits beside the judge

# 12f.3 addendum: a GGUF is measured in setups — environment variables and
# extra flags passed to llama-perplexity (lookahead routing, say). "As built"
# is always one: nothing added. A setup's id is its settings' hash, so
# changed settings are another setup, and results never mix them
AS_BUILT = {"id": "as-built", "name": "as built", "env": {}, "flags": []}
MTP_LINE = ("No MTP setups: llama-perplexity only scores the choices, so there is nothing for "
            "MTP to draft.")


def setup_id(env: dict, flags: list[str]) -> str:
    if not env and not flags:
        return AS_BUILT["id"]
    canon = json.dumps({"env": dict(sorted(env.items())), "flags": list(flags)})
    return "s" + hashlib.sha256(canon.encode()).hexdigest()[:8]


def command(bench: str, binary: str, model: str, flags: list[str], data: str,
            n: int) -> list[str]:
    """one llama-perplexity run: `n` tasks — the whole file's count for a full
    run (HellaSwag's default is 400), fewer for a subset"""
    m = MODES[BENCHMARKS[bench]["mode"]]
    return [binary, "-m", model, *flags, m["flag"], m["file"], data, m["tasks"], str(int(n))]


_TASK = {
    "hellaswag": re.compile(r"^(\d+)\t([\d.]+)%\t\[([\d.]+)%, ([\d.]+)%\]\s*$"),
    "winogrande": re.compile(r"^(\d+)\t([\d.]+)\t\s*-?[\d.]+\s+-?[\d.]+\s+\d\s+\d\s*$"),
    "multiple-choice": re.compile(r"^(\d+)\t([\d.]+)\s*$"),
}
_FINAL = {
    "winogrande": re.compile(r"Final Winogrande score\((\d+) tasks\): ([\d.]+) \+/- ([\d.]+)"),
    "multiple-choice": re.compile(r"Final result: ([\d.]+) \+/- ([\d.]+)"),
}
_CHANCE = re.compile(r"Random chance: ([\d.]+) \+/- ([\d.]+)")
_TOTAL = [re.compile(r"selecting (\d+) (?:random|randomized|the first)"),
          re.compile(r"there are (\d+) tasks in prompt"),
          re.compile(r"loaded (\d+) tasks from prompt")]
_ERROR = re.compile(r"(unable to load model|failed to create context|does not fit in the context "
                    r"window|no tasks|failed to read task|not a multiple of 6|found \d+ malformed "
                    r"tasks|failed to open file|error: .+)", re.I)


def parse(mode: str, text: str) -> dict:
    """what llama-perplexity has printed so far: {done, total, acc, se, final,
    chance, error}. acc, se and chance are fractions (it prints percentages)"""
    done, acc, total = 0, None, None
    for line in text.splitlines():
        m = _TASK[mode].match(line)
        if m:
            done, acc = int(m.group(1)), float(m.group(2)) / 100
            continue
        for p in _TOTAL:
            t = p.search(line)
            if t:
                # "selecting N" follows "loaded M": the chosen count wins
                total = int(t.group(1)) if total is None or "selecting" in line else total
                break
    out = {"done": done, "total": total, "acc": acc, "se": None, "final": False,
           "chance": MODES[mode]["chance"], "error": ""}
    f = _FINAL.get(mode)
    fm = None
    for fm in f.finditer(text) if f else []:
        pass
    if fm:
        if mode == "winogrande":
            out.update(done=int(fm.group(1)), acc=float(fm.group(2)) / 100,
                       se=float(fm.group(3)) / 100)
        else:
            out.update(acc=float(fm.group(1)) / 100, se=float(fm.group(2)) / 100)
        out["final"] = True
    c = _CHANCE.search(text)
    if c:
        out["chance"] = float(c.group(1)) / 100
    if out["se"] is None and acc is not None and done > 1:
        # HellaSwag prints no final line: the error as the others compute it
        p = out["acc"]
        out["se"] = math.sqrt(max(p * (1 - p), 0.0) / (done - 1))
    # HellaSwag prints no final line, and the others none under 100 tasks: every
    # task done is the score
    if total and done == total and acc is not None:
        out["final"] = True
    e = _ERROR.search(text)
    if e and not out["final"]:
        out["error"] = e.group(1).strip()
    return out
