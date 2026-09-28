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
  the question: 0-shot.
- **12f.5: MMLU is asked as lm_eval's mmlu asks it** — the subject's line,
  the question, the four options lettered, "Answer:" — and scored on the
  letter. Scoring each option's text after the bare question (cloze) put a
  35B at 42.5 where its author reports about 82. ARC, HellaSwag, Winogrande
  and TruthfulQA stay as they are: lm_eval scores those on the option's text
  too.
- **12f.5: -np and -c.** `--multiple-choice` puts each of a task's answers
  in a sequence of its own and refuses a task with more answers than
  `-np|--parallel` ("task N requires a higher -np|--parallel value (at least
  5)": ARC has five-answer questions, TruthfulQA up to thirteen). See
  `mc_flags`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct

GROUP = "Measured on the GGUF · llama.cpp, 0-shot"
TIP = ("Scored by llama.cpp's llama-perplexity on the quantised file. Not comparable with the "
       "lm_eval columns to its left: different prompts and no examples.")

# key -> label, llama-perplexity's mode, the dataset file under gguf_data/, the
# lm_eval task (and split) whose questions it holds, and how many that is
BENCHMARKS = {
    # 12f.5: lettered, as lm_eval asks it; a result on the cloze file is History's
    "mmlu": {"label": "MMLU", "mode": "multiple-choice", "data": "mmlu-test.bin",
             "lm_eval": "mmlu", "split": "test", "n": 14042, "format": "lettered",
             "earlier": "cloze, not comparable"},
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
# 12f.5: a job left "running" by a worker that stopped, as the worker that
# starts after it says, and the board once the worker has moved on
RESTARTED = "The GGUF worker restarted during this run."


def setup_id(env: dict, flags: list[str]) -> str:
    if not env and not flags:
        return AS_BUILT["id"]
    canon = json.dumps({"env": dict(sorted(env.items())), "flags": list(flags)})
    return "s" + hashlib.sha256(canon.encode()).hexdigest()[:8]


def command(bench: str, binary: str, model: str, flags: list[str], data: str,
            n: int, shape: dict | None = None) -> list[str]:
    """one llama-perplexity run: `n` tasks — the whole file's count for a full
    run (HellaSwag's default is 400), fewer for a subset. A multiple-choice
    file's -np and -c come from its shape, after the model's and the setup's
    flags, so they are the ones it runs with"""
    m = MODES[BENCHMARKS[bench]["mode"]]
    size = mc_flags(shape) if shape and m["flag"] == "--multiple-choice" else []
    return [binary, "-m", model, *flags, *size, m["flag"], m["file"], data, m["tasks"],
            str(int(n))]


# ---------------------------------------------------------------------------
# 12f.5: the multiple-choice file, read as multiple_choice_score reads it
# ---------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJ"


def read_mc(data: bytes) -> list[dict]:
    """the file back, as multiple_choice_score deserialises it — and each
    task read again from its own offset, as a subset run seeks to it"""
    def string(at):
        (n,) = struct.unpack_from("<I", data, at)
        return data[at + 4:at + 4 + n].decode("utf-8"), at + 4 + n

    def block(at):
        (n,) = struct.unpack_from("<I", data, at)
        if n > 100:
            raise ValueError("more than 100 answers: llama-perplexity refuses it")
        at += 4
        answers = []
        for _ in range(n):
            a, at = string(at)
            answers.append(a)
        labels = list(struct.unpack_from(f"<{n}i", data, at))
        return answers, labels, at + 4 * n

    try:
        (n_task,) = struct.unpack_from("<I", data, 0)
        offsets = struct.unpack_from(f"<{n_task}I", data, 4)
        out, at = [], 4 + 4 * n_task
        for i in range(n_task):
            if offsets[i] != at:
                raise ValueError(f"task {i} starts at {at}, its offset says {offsets[i]}")
            q, at = string(at)
            a1, l1, at = block(at)
            a2, l2, at = block(at)
            out.append({"question": q, "answers": a1, "labels": l1, "mc2": a2})
    except struct.error:
        raise ValueError("it ends in the middle of a task") from None
    if at != len(data):
        raise ValueError(f"{len(data) - at} bytes after the last task")
    return out


def task_tokens(question: str, answers: list[str]) -> int:
    """at least the tokens multiple_choice_prepare_one_task makes of a task,
    for any tokenizer: each answer is tokenized as question + " " + answer,
    with a BOS; the tokens every answer shares (the question) count once,
    the rest once an answer. A token is at least one byte of UTF-8, and a
    few more a sequence cover the BOS, a SentencePiece leading space and a
    merge where the question meets the answer"""
    return len(question.encode("utf-8")) + 2 + sum(len(a.encode("utf-8")) + 3 for a in answers)


def mc_shape(data: bytes) -> dict:
    """what -np and -c are sized from: the most answers any task has, and
    the biggest task's tokens (task_tokens); and whether its answers are
    letters (lm_eval's lettered MMLU) or each option's text"""
    tasks = read_mc(data)
    lettered = bool(tasks) and all(t["answers"] == list(LETTERS[:len(t["answers"])])
                                   for t in tasks)
    return {"max_answers": max((len(t["answers"]) for t in tasks), default=0),
            "max_task_tokens": max((task_tokens(t["question"], t["answers"]) for t in tasks),
                                   default=0),
            "format": "lettered" if lettered else "text"}


def mc_flags(shape: dict) -> list[str]:
    """-np and -c for a multiple-choice file, as perplexity.cpp uses them:
    - main() makes n_parallel max(4, -np), sets the KV cache unified (one
      buffer every sequence shares, not n_ctx / n_seq each) and n_ctx
      n_parallel × -c; n_batch is min(n_batch, n_ctx), and decode_helper
      feeds a batch through in n_batch pieces, so it limits nothing;
    - multiple_choice_score gives each of a task's answers a sequence, at
      most n_parallel at once: fewer than the task's answers is "requires a
      higher -np|--parallel value". It packs tasks while they fit in n_ctx:
      a task bigger than n_ctx alone "does not fit in the context window".
    So -np is the most answers a task has (at least 4, its own floor), and
    -np × -c holds the biggest task: -c is its tokens (task_tokens, a bound
    for any tokenizer) over -np, rounded up to 256, and never under 512,
    llama-perplexity's own default — what ran before."""
    n_par = max(4, int(shape.get("max_answers") or 0))
    ctx = -(-int(shape.get("max_task_tokens") or 0) // n_par)
    ctx = max(512, -(-ctx // 256) * 256)
    return ["-np", str(n_par), "-c", str(ctx)]


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
                    r"tasks|failed to open file|task \d+ requires a higher -np\|--parallel value "
                    r"\(at least \d+\)|error: .+)", re.I)


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
