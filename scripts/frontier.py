"""17: the Frontier benchmarks, asked the way their owners and Epoch AI ask
them — the definitions, the prompts and the code scorers. No service import:
the board (service/frontier.py) and a rented box (scripts/remote_gguf.py) read
the same file, so a question is asked the same way on both.

Each benchmark's protocol, from its source (fetched 5 Oct 2026):
- GPQA Diamond — Epoch AI's v1.0.6 (20 Feb 2026): zero-shot, the four choices
  shuffled once per question (seeded here, so a run repeats), and Inspect's
  choice() parse: the last line that is "ANSWER: X". Epoch averages 16 runs;
  the board 4.
- OTIS Mock AIME 2024–2025 — Epoch's prompt; scored by code on the last
  "ANSWER: X", then (stage 3) a model check as a second look on what the code
  marks wrong or can't read: Epoch extracts the answer with a model that
  doesn't see the key (Sept 2025), whose prompt it hasn't published. Epoch
  averages 16 runs; the board 8.
- MATH Level 5 — Epoch's prompt and its own answer extraction (scorer.py); by
  code (math-verify) here, then (stage 3) Epoch's equivalence check by a model
  as a second look on what the code marks wrong. Epoch averages 8 runs; the
  board 1.
- Humanity's Last Exam — CAIS's protocol on its text-only questions: the
  system prompt CAIS's repository sends today, graded (stage 3) by CAIS's
  judge prompt and model. Epoch imports HLE; it doesn't run it.
- SimpleQA Verified — Epoch's: the question and its "single best guess" line
  (27 Aug 2026), graded (stage 3) by Google's grader prompt; the share right.
- MMLU-Pro — TIGER-Lab's API protocol: 5-shot chain of thought from the
  question's own category, "The answer is (X)", its three-regex extraction —
  every one of the 12,032 test questions. Epoch doesn't run MMLU-Pro. No random
  guess for an answer nothing can be read from: wrong, and counted.
- ARC-AGI-2 — ARC Prize's harness on the public evaluation set: its prompt,
  its grid parser, two attempts, a task the share of its test grids solved.

Every run uses the model card's sampling (PRESETS) where Epoch uses each API's
default temperature, a seed a question and run, and the budgets decided on 5
Oct (thinking on 32,768 or 65,536 tokens, off 4,096 or 8,192). An answer that
runs out of room is wrong, and counted.

The questions are never in the repo, the runner image or a bundle: they are
fetched where the run is, with the person's token for a gated set, and kept on
that machine's disk only (SimpleQA Verified, MIT, is in the repo already).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import tarfile
import urllib.request
from pathlib import Path

VERSION = "frontier-1"
LETTERS = "ABCD"
HERE = Path(__file__).resolve().parent

# Epoch AI's GPQA Diamond template, v1.0.6 (20 Feb 2026), word for word
GPQA_TEMPLATE = ("Answer the following multiple choice question. The last line of your response "
                 "should be of the following format: 'ANSWER: LETTER' (without quotes) where "
                 "LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n\n"
                 "A) {A}\nB) {B}\nC) {C}\nD) {D}")
# Epoch AI's OTIS Mock AIME prompt (its code), the line break after "from" included
OTIS_TEMPLATE = ("Please solve this AIME problem step by step. The answer is an integer ranging "
                 "from\n000 to 999, inclusive.\n\n{prompt}\n\nRemember to show your work clearly "
                 "and end with 'ANSWER: X' where X is your final numerical answer.")
# Epoch AI's MATH Level 5 prompt (its solver.py)
MATH_TEMPLATE = ('Solve the following math problem step by step. The last line of your response '
                 'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
                 'answer to the problem.\n\n{prompt}\n\nRemember to put your answer on its own '
                 'line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is '
                 'the answer to the problem, and you do not need to use a \\boxed command.')
# Epoch AI's anti-abstention line (27 Aug 2026), after the question. Epoch
# doesn't say what separates the two: a blank line here
SIMPLEQA_GUESS = ("This is a guessing game, not a test. There's no penalty for being wrong, and "
                  "an uncertain guess is far more useful to me than \"I don't know\". Give your "
                  "single best guess.")
# CAIS's system prompt for every HLE question (hle_eval, since 6 Jun 2025)
HLE_SYSTEM = ("Your response should be in the following format:\nExplanation: {your explanation "
              "for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your "
              "confidence score between 0% and 100% for your answer}")
# TIGER-Lab's MMLU-Pro (evaluate_from_api.py): the head, and each example
MMLU_PRO_HEAD = ("The following are multiple choice questions (with answers) about {}. Think "
                 "step by step and then output the answer in the format of \"The answer is (X)\" "
                 "at the end.\n\n")
MMLU_PRO_LETTERS = "ABCDEFGHIJ"
# ARC Prize's harness (arc-agi-benchmarking, prompts/system_prompt.txt), sent as
# the user's message
ARC_TEMPLATE = ("You are participating in a puzzle solving competition. You are an expert at "
                "solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. "
                "Your goal is to identify the pattern or transformation in the training examples "
                "that maps the input to the output, then apply that pattern to the test input to "
                "give a final output.\n\nRespond in the format of the training output examples\n\n"
                "--Training Examples--\n{training_examples}\n--End of Training Examples--\n\n"
                "--Test Input--\n{test_input}\n--End of Test Input--\n\nYour response:")
MATH_CONFIGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus")

ON_LONG, OFF_LONG = 65536, 8192          # OTIS, MATH Level 5, ARC-AGI-2
ON, OFF = 32768, 4096                    # GPQA, HLE, MMLU-Pro, SimpleQA Verified

BENCH: dict[str, dict] = {
    "gpqa_diamond_epoch": {
        "short": "gpqa", "label": "GPQA Diamond", "group": "Science",
        # the names others report it under (the Frontier view joins on these)
        "reported_as": ["gpqa diamond"],
        "protocol": "Epoch AI's (v1.0.6): zero-shot, ANSWER: LETTER, Inspect's choice() parse",
        "protocol_version": "epoch-gpqa-1.0.6",
        "source": {"hf": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                   "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6", "gated": True,
                   "licence": "CC BY 4.0"},
        "n": 198, "epochs": 4, "epoch_runs": 16, "budget": {"on": ON, "off": OFF},
        # the prompt's room in a slot's context, above the budget
        "room": 2048, "scorer": "choice", "unlisted": True},
    "otis_aime_epoch": {
        "short": "otis", "label": "OTIS Mock AIME 2024–2025", "column": "OTIS Mock AIME 2024-2025",
        "group": "Maths", "reported_as": ["otis mock aime 2024-2025", "otis mock aime"],
        "protocol": "Epoch AI's: its prompt, the last ANSWER: X, exact match",
        "protocol_version": "epoch-otis-2025-09",
        "source": {"hf": "EpochAI/otis-mock-aime-24-25", "split": "train",
                   "revision": "3072536d76ff88f487f65148c5245165b5d8e627", "gated": True,
                   "licence": "Apache-2.0"},
        "n": 45, "epochs": 8, "epoch_runs": 16, "budget": {"on": ON_LONG, "off": OFF_LONG},
        "room": 2048, "scorer": "integer", "look": "extract"},
    "math_l5_epoch": {
        "short": "math-l5", "label": "MATH Level 5", "column": "MATH level 5", "group": "Maths",
        "reported_as": ["math level 5"],
        "protocol": "Epoch AI's: its prompt and answer extraction, equivalence by code",
        "protocol_version": "epoch-math-l5-2025",
        "source": {"hf": "EleutherAI/hendrycks_math", "configs": list(MATH_CONFIGS),
                   "split": "test", "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
                   "gated": False, "licence": "MIT"},
        "n": 1324, "epochs": 1, "epoch_runs": 8, "budget": {"on": ON_LONG, "off": OFF_LONG},
        "room": 2048, "scorer": "math", "look": "equivalent",
        # 17: its problems are competitions' (AMC, AIME), whose copyright is in
        # dispute: its Hugging Face home was taken down, and Epoch withholds its logs
        "unlisted": ("MATH's problems aren't shown here: they are competitions' problems "
                     "whose copyright is disputed (their first home on Hugging Face was taken "
                     "down, and Epoch AI withholds its MATH logs)")},
    "hle_text_cais": {
        "short": "hle", "label": "Humanity's Last Exam", "group": "Knowledge & reasoning",
        "reported_as": ["humanity's last exam", "hle", "humanity's last exam (text only)",
                        "hle (text only)"],
        "note": "text-only questions",
        "protocol": "CAIS's: its system prompt, graded by its judge prompt",
        "protocol_version": "cais-hle-2025-06",
        "source": {"hf": "cais/hle", "split": "test",
                   "revision": "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29", "gated": True,
                   "licence": "MIT"},
        "n": 2158, "epochs": 1, "budget": {"on": ON, "off": OFF}, "room": 4096,
        "scorer": "graded", "system": HLE_SYSTEM,
        "grader": {"who": "CAIS's judge", "prompt": "hle_judge", "model": "openai/o3-mini",
                   "version": "o3-mini-2025-01-31"},
        "unlisted": ("Humanity's Last Exam's questions are never shown: its authors ask that "
                     "it not be shared, re-uploaded or distributed")},
    "simpleqa_epoch": {
        "short": "simpleqa", "label": "SimpleQA Verified", "group": "Knowledge",
        "reported_as": ["simpleqa verified"],
        "protocol": "Epoch AI's: the question and its single-best-guess line, the share "
                    "graded correct",
        "protocol_version": "epoch-simpleqa-verified-2026-08-27",
        "source": {"hf": "google/simpleqa-verified", "file": "simpleqa_verified.csv",
                   "split": "eval", "revision": "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",
                   "sha256": "b5db21155444763543fe31b67e7cf28ce2bb225742a5b889421c5f182e2f92f5",
                   "gated": False, "licence": "MIT"},
        "n": 1000, "epochs": 1, "epoch_runs": 1, "budget": {"on": ON, "off": OFF},
        "room": 1024, "scorer": "graded",
        "grader": {"who": "Google's grader", "prompt": "simpleqa_google",
                   "model": "openai/gpt-4.1", "version": "gpt-4.1-2025-04-14"}},
    "mmlupro_tiger": {
        "short": "mmlu-pro", "label": "MMLU-Pro", "group": "Knowledge", "reported_as": ["mmlu-pro", "mmlu pro"],
        # 17b: what differs from TIGER-Lab's own script, in a few words
        "note": "TIGER-Lab's 5-shot prompt; the card's sampling, not temperature 0; no "
                "random guess when unread",
        "protocol": "TIGER-Lab's (evaluate_from_api.py): 5-shot chain of thought from the "
                    "question's category, \"The answer is (X)\", its three-regex extraction",
        "protocol_version": "tiger-mmlu-pro-api-f418b116", "shots": 5,
        "source": {"hf": "TIGER-Lab/MMLU-Pro", "split": "test",
                   "revision": "527feea0afed1de15a8c115abf7be4c912123315", "gated": False,
                   "licence": "MIT"},
        "n": 12032, "epochs": 1, "budget": {"on": ON, "off": OFF}, "room": 4096,
        "scorer": "mmlu_pro"},
    "arc_agi2_public": {
        "short": "arc-agi-2", "label": "ARC-AGI-2", "group": "Puzzles", "reported_as": ["arc-agi-2", "arc agi 2"],
        "note": "public set",
        "protocol": "ARC Prize's harness: its prompt and grid parser, two attempts, a task the "
                    "share of its test grids solved",
        "protocol_version": "arcprize-benchmarking-9e2828fb",
        "source": {"github": "arcprize/ARC-AGI-2", "path": "data/evaluation",
                   "revision": "f3283f727488ad98fe575ea6a5ac981e4a188e49", "gated": False,
                   "licence": "Apache-2.0"},
        "n": 120, "epochs": 2, "aggregate": "pass@2", "budget": {"on": ON_LONG, "off": OFF_LONG},
        # a task's training pairs, 30 × 30 grids at most, are long prompts
        "room": 32768, "scorer": "grid"},
}
TASKS = list(BENCH)

# the sampling a model's own card recommends, by thinking setting: the nearest
# a local model has to "each API's default temperature", which Epoch uses
PRESETS = {
    # Qwen3.6-35B-A3B's card
    "qwen3.6": {"on": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
                "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
    # Gemma's published defaults (temperature 1.0, top_k 64, top_p 0.95)
    "gemma-4": {"on": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
                "off": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}},
}
_FAMILY = [("qwen3.6", re.compile(r"qwen[\s_-]*3[._-]?6", re.I)),
           ("gemma-4", re.compile(r"gemma[\s_-]*4", re.I))]


def family(*names: str) -> str:
    """a model's sampling preset by its names (its id, file, what it is based
    on) — '' when none is known: the server's own defaults then"""
    text = " ".join(n for n in names if n)
    return next((k for k, rx in _FAMILY if rx.search(text)), "")


def sampling(fam: str, on: bool) -> dict:
    return dict((PRESETS.get(fam) or {}).get("on" if on else "off") or {})


def slot_context(tasks: list[str], on: bool) -> int:
    """the context one llama-server slot needs for the longest of `tasks`:
    its budget and its prompt's room"""
    return max(BENCH[t]["budget"]["on" if on else "off"] + BENCH[t]["room"] for t in tasks)


def seed_of(task: str, qid: str, epoch: int) -> int:
    """the same seed for the same question and run, on every machine"""
    return int(hashlib.sha256(f"{task}:{qid}:{epoch}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# the questions: fetched where the run is, kept on that disk only
# ---------------------------------------------------------------------------

def data_dir(root: Path) -> Path:
    return Path(root) / "frontier" / "data"


def _cache(root: Path, task: str) -> Path:
    s = BENCH[task]["source"]
    return data_dir(root) / f"{task}-{s['revision'][:12]}.json"


def _hf_load(task: str, config: str | None = None, split: str | None = None):
    from datasets import load_dataset
    s = BENCH[task]["source"]
    return load_dataset(s["hf"], config if config is not None else s.get("config"),
                        split=split or s["split"], revision=s["revision"],
                        token=os.environ.get("HF_TOKEN") or None)


def _hf_file(task: str) -> Path:
    from huggingface_hub import hf_hub_download
    s = BENCH[task]["source"]
    return Path(hf_hub_download(s["hf"], s["file"], repo_type="dataset", revision=s["revision"],
                                token=os.environ.get("HF_TOKEN") or None))


def last_boxed_only_string(string: str) -> str | None:
    """the last \\boxed{…} (or \\fbox{…}) of a solution, as Epoch's scorer.py
    (Hendrycks's) takes it"""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, right, depth = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            depth += 1
        if string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return None if right is None else string[idx:right + 1]


def remove_boxed(s: str | None) -> str | None:
    if s is None:
        return None
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    if s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{"):-1]
    return s


def _simpleqa_rows() -> list[dict]:
    """SimpleQA Verified: the repo's pinned copy where it is (the board), else
    the same file from Hugging Face at the same revision (a rented box) — the
    one the repo's manifest hashes either way"""
    spec = BENCH["simpleqa_epoch"]["source"]
    here = HERE.parent / "eval_tasks" / "simpleqa" / spec["file"]
    path = here if here.exists() else _hf_file("simpleqa_epoch")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError(f"{spec['file']} isn't the pinned file (sha256 "
                         f"{hashlib.sha256(raw).hexdigest()[:16]}…)")
    return [{"id": str(r["original_index"]), "question": r["problem"].strip(),
             "answer": r["answer"].strip(), "subject": r.get("topic") or ""}
            for r in csv.DictReader(io.StringIO(raw.decode("utf-8")))]


def _arc_rows() -> list[dict]:
    """ARC-AGI-2's public evaluation set at its commit, from GitHub: each test
    grid of each task a question, its task the group"""
    s = BENCH["arc_agi2_public"]["source"]
    url = f"https://codeload.github.com/{s['github']}/tar.gz/{s['revision']}"
    with urllib.request.urlopen(url, timeout=300) as r:
        raw = r.read()
    out = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile()
                       and f"/{s['path']}/" in m.name and m.name.endswith(".json"))
        for name in names:
            task = json.loads(tar.extractfile(name).read())
            tid = Path(name).stem
            for i, t in enumerate(task["test"]):
                out.append({"id": f"{tid}#{i}", "group": tid, "train": task["train"],
                            "test_input": t["input"], "answer": t["output"]})
    return out


def _count(task: str, got) -> int:
    """the questions a benchmark holds — ARC-AGI-2 counted by its tasks"""
    items = got["items"] if isinstance(got, dict) else got
    if BENCH[task].get("aggregate") == "pass@2":
        return len({group_of(task, it) for it in items})
    return len(items)


def _expected_n(task: str) -> int | None:
    return BENCH[task]["n"]


# the questions a benchmark must hold — tests/conftest.py lets invented sets
# through, and the tests of the count put this back
expected = _expected_n


def check_count(task: str, got, where: str) -> None:
    """17b: the benchmark holds the questions it should (HLE's text-only filter,
    MMLU-Pro's 12,032, ARC-AGI-2's 120 tasks), or nothing is asked. 17c: on
    every load — a copy on disk from before the check is counted too"""
    n, want = _count(task, got), expected(task)
    if want is not None and n != want:
        raise ValueError(f"{BENCH[task]['label']}: {n:,} questions {where}, and this board "
                         f"expects {want:,} — nothing is asked until they agree")


def load_failed(task: str, e: Exception) -> str:
    """17c: why a benchmark's questions couldn't be had, in the error's own
    words — a count that doesn't agree says so — and a gated benchmark's terms
    named only when access was what was refused"""
    label = BENCH[task]["label"]
    said = str(e).strip() or repr(e)
    if not said.startswith(label):
        said = f"{label}: its questions could not be fetched: {said}"
    if BENCH[task]["source"].get("gated") and re.search(
            r"\b40[13]\b|gated|unauthori[sz]ed|forbidden|access to|restricted", said, re.I):
        said += (" — it is gated: accept its terms on Hugging Face with the account whose "
                 "token this machine has")
    return said


def _fetch(task: str):
    """the dataset at its pinned revision, its count checked"""
    got = _fetch_source(task)
    check_count(task, got, f"came from {source_name(task)} at "
                           f"{BENCH[task]['source']['revision'][:12]}")
    return got


def _fetch_source(task: str):
    """the dataset at its pinned revision, with HF_TOKEN for a gated one — a
    list of questions, or {items, extra} (MMLU-Pro's examples)"""
    if task == "gpqa_diamond_epoch":
        return [{"id": str(r["Record ID"]), "question": r["Question"].strip(),
                 "right": r["Correct Answer"].strip(),
                 "wrong": [r[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)]}
                for r in _hf_load(task)]
    if task == "otis_aime_epoch":
        return [{"id": str(r["id"]), "question": r["input"], "answer": str(r["target"]).strip()}
                for r in _hf_load(task)]
    if task == "math_l5_epoch":
        out = []
        for cfg in MATH_CONFIGS:
            for i, r in enumerate(_hf_load(task, cfg)):
                if r["level"] == "Level 5":
                    out.append({"id": f"{cfg}/{i}", "question": r["problem"],
                                "answer": remove_boxed(last_boxed_only_string(r["solution"])),
                                "subject": r["type"]})
        return out
    if task == "hle_text_cais":
        # the text-only questions: no image
        return [{"id": str(r["id"]), "question": r["question"], "answer": r["answer"],
                 "answer_type": r["answer_type"], "subject": r.get("category") or ""}
                for r in _hf_load(task) if not r.get("image")]
    if task == "simpleqa_epoch":
        return _simpleqa_rows()
    if task == "mmlupro_tiger":
        def opts(r):
            return [o for o in r["options"] if o != "N/A"]
        items = sorted(({"id": str(r["question_id"]), "question": r["question"],
                         "options": opts(r), "answer": r["answer"], "category": r["category"]}
                        for r in _hf_load(task)), key=lambda x: (x["category"], int(x["id"])))
        shots: dict[str, list] = {}
        for r in _hf_load(task, split="validation"):
            shots.setdefault(r["category"], []).append(
                {"question": r["question"], "options": opts(r), "cot_content": r["cot_content"]})
        return {"items": items, "extra": {"shots": shots}}
    if task == "arc_agi2_public":
        return _arc_rows()
    raise KeyError(task)


def _read_cache(p: Path):
    got = json.loads(p.read_text(encoding="utf-8"))
    return got if isinstance(got, dict) else {"items": got, "extra": {}}


def load(task: str, root: Path) -> list[dict]:
    """the task's questions, in its order — read once, then from this
    machine's own copy"""
    p = _cache(root, task)
    if p.exists():
        got = _read_cache(p)
        check_count(task, got, f"are in this machine's copy ({p}: remove it, and the next "
                               "run fetches the pinned revision again)")
        return got["items"]
    got = _fetch(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return got["items"] if isinstance(got, dict) else got


def extra(task: str, root: Path) -> dict:
    """what a task keeps beside its questions — MMLU-Pro's 5-shot examples"""
    load(task, root)
    return _read_cache(_cache(root, task))["extra"]


def shard_of(items: list, shard: tuple[int, int] | None) -> list:
    """every n-th question from the i-th (1-based), the same split everywhere"""
    if not shard:
        return list(items)
    i, n = shard
    return [x for k, x in enumerate(items) if k % n == i - 1]


def parse_shard(text: str) -> tuple[int, int] | None:
    text = (text or "").strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if not m:
        raise ValueError(f"{text!r} isn't i/n")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 2 or not 1 <= i <= n:
        raise ValueError(f"{i}/{n} isn't a shard: 1 ≤ i ≤ n, and n at least 2")
    return i, n


# ---------------------------------------------------------------------------
# a question asked, and its answer read
# ---------------------------------------------------------------------------

def order_of(task: str, item: dict) -> list[str]:
    """GPQA: the four choices in their shuffled order — once per question,
    seeded, as Epoch shuffles once per question (unseeded)"""
    rng = random.Random(seed_of(task, item["id"], 0))
    opts = [item["right"], *item["wrong"]]
    rng.shuffle(opts)
    return opts


def _mmlu_pro_example(question: str, options: list[str], cot_content: str = "") -> str:
    """TIGER-Lab's format_example (evaluate_from_api.py), as it is"""
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(MMLU_PRO_LETTERS[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def _arc_prompt(item: dict) -> str:
    """ARC Prize's prompt_manager: each training pair numbered from 0, grids as
    json.dumps writes them"""
    ex = ""
    for i, pair in enumerate(item["train"]):
        ex += f"--Example {i}-- \n\n INPUT: \n\n"
        ex += json.dumps(pair["input"]) + "\n\n"
        ex += "OUTPUT: \n\n"
        ex += json.dumps(pair["output"]) + "\n\n"
    return (ARC_TEMPLATE.replace("{training_examples}", ex)
            .replace("{test_input}", json.dumps(item["test_input"])))


_SHOTS: dict = {}
_ROOT = Path(os.environ.get("BENCH_ROOT", os.getcwd()))


def shots_for(task: str, category: str) -> list[dict]:
    """MMLU-Pro's five examples of a category, from its validation split, in
    this machine's copy (set_root)"""
    key = (task, str(_ROOT))
    if key not in _SHOTS:
        _SHOTS[key] = extra(task, _ROOT).get("shots") or {}
    return _SHOTS[key].get(category) or []


def set_root(root: Path) -> None:
    """where this machine keeps the datasets (the board's BENCH_ROOT)"""
    global _ROOT
    _ROOT = Path(root)


def system_of(task: str) -> str | None:
    """the system message a benchmark sends: HLE's; none for the others"""
    return BENCH[task].get("system")


def prompt(task: str, item: dict) -> tuple[str, dict]:
    """(the message, what scoring needs: the right letter, or the answer)"""
    if task == "gpqa_diamond_epoch":
        opts = order_of(task, item)
        text = GPQA_TEMPLATE.format(question=item["question"], **dict(zip(LETTERS, opts)))
        return text, {"key": LETTERS[opts.index(item["right"])]}
    if task == "otis_aime_epoch":
        return OTIS_TEMPLATE.replace("{prompt}", item["question"]), {"answer": item["answer"]}
    if task == "math_l5_epoch":
        return MATH_TEMPLATE.replace("{prompt}", item["question"]), {"answer": item["answer"]}
    if task == "hle_text_cais":
        return item["question"], {"answer": item["answer"]}
    if task == "simpleqa_epoch":
        return f"{item['question']}\n\n{SIMPLEQA_GUESS}", {"answer": item["answer"]}
    if task == "mmlupro_tiger":
        head = MMLU_PRO_HEAD.format(item["category"])
        for ex in shots_for(task, item["category"]):
            head += _mmlu_pro_example(ex["question"], ex["options"], ex["cot_content"])
        return (head + _mmlu_pro_example(item["question"], item["options"]),
                {"key": item["answer"], "letters": MMLU_PRO_LETTERS[:len(item["options"])]})
    if task == "arc_agi2_public":
        return _arc_prompt(item), {"answer": item["answer"]}
    raise KeyError(task)


# 17d: the leading spaces taken whole (possessive): a reply of 20,000 newlines
# and no </think> took 2 s a call, the pattern backtracking into them
_THINK = re.compile(r"(?s)^\s*+(?:<think>)?.*?</think>")
# Inspect's multiple-choice parse (inspect_ai solver/_multiple_choice.py,
# parse_answers, as of 5 Oct 2026), the single-answer case: wrappers off
# ($B$, **B**, (B)), then the last line that is "ANSWER: X", else the last
# "ANSWER: X" anywhere; one letter of the choices, or nothing
_UNWRAP = ((re.compile(r"\x24\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\x24"), r"\1"),
           (re.compile(r"\*\*\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\*\*"), r"\1"),
           (re.compile(r"(?i)(^|[:,])(\s*)\(([A-Za-z\d])\)(?=\s*(?:,|\.|\n|\Z))"), r"\1\2\3"))
_ANSWER_LINE = re.compile(r"(?im)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)")
_ANSWER_ANY = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")


def visible(text: str) -> str:
    """the reply without its thinking"""
    text = text or ""
    if "</think>" not in text:
        return text.strip()
    return _THINK.sub("", text, count=1).strip()


def ran_out(text: str, finish: str | None) -> bool:
    """the reply stopped at its budget: cut, or its thinking never closed"""
    t = text or ""
    return finish == "length" or ("<think>" in t[:200] and "</think>" not in t)


def read_choice(text: str, letters: str = LETTERS) -> str | None:
    """the letter a reply answers with, as Inspect's choice() reads it — None
    when it gives none, or more than one"""
    t = visible(text)
    for rx, to in _UNWRAP:
        t = rx.sub(to, t)
    found = _ANSWER_LINE.findall(t) or _ANSWER_ANY.findall(t)
    if not found:
        return None
    got = [x for x in found[-1].strip().rstrip(".").upper().split(",") if x]
    return got[0] if len(got) == 1 and got[0] in letters else None


# OTIS: the last "ANSWER: X", an integer from 0 to 999 (Epoch's extractor reads
# what the model gave as its final answer; a line of code reads the same, mostly)
# 17b: and nothing after the digits but a brace, a dollar, bold or a full stop —
# "3.5", "3/4", "2^{10}", "12 or 13" are no integer, for the second look to read
# 17c: the last "ANSWER:" only — an answer the model replaced ("ANSWER: 42 …
# ANSWER: 43 (mod 1000)") is never read from the line before — and an integer
# that ends its line, at most six digits (and its thousands' commas), spaces
# and tabs only between, each run of them taken whole (possessive): a long run
# of digits or spaces can't hang or raise
_OTIS_AT = re.compile(r"(?i)ANSWER[ \t]*:")
_OTIS_INT = re.compile(r"[ \t]*+\**+[ \t]*+\$?[ \t]*+(?:\\boxed\{)?[ \t]*+"
                       r"(-?\d{1,6}+(?:,\d{3}){0,2}+)(?![\d,])"
                       r"[ \t]*+\}?[ \t]*+\$?[ \t]*+\**+[ \t]*+\.?[ \t]*+(?=\r?\n|\Z)")


def read_integer(text: str) -> str | None:
    """OTIS's answer: the integer of the reply's last "ANSWER:", or None"""
    vis = visible(text)
    at = None
    for at in _OTIS_AT.finditer(vis):
        pass
    if at is None:
        return None
    m = _OTIS_INT.match(vis, at.end())
    if not m:
        return None
    try:
        return str(int(m.group(1).replace(",", "")))
    except ValueError:
        return None


def _math_helper(text: str) -> str | None:
    """Epoch's _extract_answer_helper: what follows the first ANSWER, past a
    colon and spaces"""
    try:
        start = text.index("ANSWER") + 6
    except ValueError:
        return None
    while start < len(text) and (text[start] == ":" or text[start].isspace()):
        start += 1
    got = text[start:].strip()
    return got or None


def read_math(completion: str) -> str | None:
    """Epoch's extract_answer (MATH Level 5's scorer.py), in its order: inside
    \\text{ANSWER…}, a bold line with ANSWER, else the first line with ANSWER"""
    completion = visible(completion)
    if "\\text{ANSWER" in completion:
        try:
            start = completion.index("\\text{ANSWER")
            end = completion.index("}", start)
            inside = _math_helper(completion[start:end + 1][:-1])
            if inside:
                return inside
            after = completion[end + 1:].strip()
            if "\n" in after:
                after = after[:after.index("\n")]
            return after or None
        except ValueError:
            pass
    lines = completion.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("**") and line.endswith("**") and "ANSWER" in line:
            got = _math_helper(line[2:-2].strip())
            if got:
                return got
    for line in lines:
        if "ANSWER" in line:
            got = _math_helper(line)
            if got:
                return got
    return None


def math_equal(a: str | None, b: str | None) -> bool:
    """equivalence by code: the board's math-verify comparison (generative.py)"""
    import sys
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import generative
    return generative.math_equal(a, b)


def read_mmlu_pro(text: str, letters: str = MMLU_PRO_LETTERS) -> str | None:
    """TIGER-Lab's extract_answer → extract_again → extract_final, after its
    response.replace('**', ''), as they are (no random guess)"""
    text = visible(text).replace("**", "")
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if not m:
        m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if not m:
        m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
        return m.group(0) if m else None
    return m.group(1)


def _backscan(text: str):
    """ARC Prize's backscan_json_parser: the last bracketed JSON, a non-empty
    list of lists"""
    last, close = -1, None
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("]", "}"):
            last, close = i, text[i]
            break
    if last == -1:
        return None
    opening, count, start = ("[" if close == "]" else "{"), 1, -1
    for i in range(last - 1, -1, -1):
        if text[i] == close:
            count += 1
        elif text[i] == opening:
            count -= 1
            if count == 0:
                start = i
                break
    if start == -1:
        return None
    try:
        got = json.loads(text[start:last + 1])
    except json.JSONDecodeError:
        return None
    return got if isinstance(got, list) and got and all(isinstance(r, list) for r in got) else None


def read_grid(text: str):
    """ARC Prize's parse: a \\boxed{…} grid, else the last JSON list of lists"""
    text = visible(text)
    m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if m:
        try:
            got = json.loads(m.group(1).strip())
            if isinstance(got, list) and all(isinstance(r, list) for r in got):
                return got
        except json.JSONDecodeError:
            pass
    return _backscan(text)


_HLE_ANSWER = re.compile(r"(?im)^\s*(?:exact\s+)?answer\s*:\s*(.+)$")


def score(task: str, answer: str, finish: str | None, need: dict) -> dict:
    """{ok, read, ran_out} for one answer — ok None for a graded benchmark's
    (its grader decides)"""
    out = ran_out(answer, finish)
    kind = BENCH[task]["scorer"]
    if kind == "choice":
        got = None if out else read_choice(answer)
        return {"ok": got == need["key"], "read": got, "ran_out": out}
    if kind == "integer":
        got = None if out else read_integer(answer)
        try:
            ok = got is not None and int(got) == int(str(need["answer"]).strip())
        except ValueError:
            ok = False
        return {"ok": ok, "read": got, "ran_out": out}
    if kind == "math":
        got = None if out else read_math(answer)
        return {"ok": got is not None and math_equal(got, need["answer"]), "read": got,
                "ran_out": out}
    if kind == "mmlu_pro":
        got = None if out else read_mmlu_pro(answer)
        return {"ok": got is not None and got == need["key"], "read": got, "ran_out": out}
    if kind == "grid":
        got = None if out else read_grid(answer)
        return {"ok": got is not None and got == need["answer"],
                "read": None if got is None else json.dumps(got), "ran_out": out}
    if kind == "graded":
        vis = visible(answer)
        m = _HLE_ANSWER.findall(vis) if task == "hle_text_cais" else []
        got = (m[-1].strip() if m else vis.strip()[-300:]) if not out else None
        return {"ok": None, "read": got or None, "ran_out": out}
    raise KeyError(task)


# 17b: the parity check — does a box answer as the server does? The same
# MMLU-Pro questions (public, scored by code), thinking off and greedy, asked
# with each side's own launch: the box's flags, slots and KV cache, the
# server's. 17d: decided on accuracy, not on letters — on the pilot, letter
# agreement over 50 questions measured run-to-run noise (two runs on one box
# agreed on 43 of 50), not the setup. "The same": the paired difference in
# right answers (the box's minus the server's, question by question) with its
# 90% interval inside ±MARGIN points — the two one-sided tests of equivalence,
# at 5% each, its margin stated here before any run. Letter agreement and
# identical replies are reported beside it for information, with the box's
# agreement with a second run of itself
PARITY = {"task": "mmlupro_tiger", "n": 500, "seed": "frontier-parity-1", "max_tokens": 2048,
          "sampling": {"temperature": 0.0, "top_k": 1, "seed": 0},
          "margin": 0.05, "z": 1.645}


def parity_items(root: Path, n: int | None = None) -> list[dict]:
    """the parity check's questions: the same n of MMLU-Pro on every machine"""
    items = load(PARITY["task"], root)
    pick = sorted(random.Random(PARITY["seed"]).sample(range(len(items)),
                                                       min(n or PARITY["n"], len(items))))
    return [items[i] for i in pick]


def _agree(a: dict, b: dict, key_a: str = "answer", key_b: str = "answer") -> tuple[int, int, int]:
    """(both read a letter, the same letter, identical replies) over a's and b's ids"""
    ids = sorted(set(a) & set(b))
    la = {i: read_mmlu_pro(a[i].get(key_a) or "") for i in ids}
    lb = {i: read_mmlu_pro(b[i].get(key_b) or "") for i in ids}
    both = [i for i in ids if la[i] and lb[i]]
    return (len(both), sum(1 for i in both if la[i] == lb[i]),
            sum(1 for i in ids if (a[i].get(key_a) or "") and visible(a[i].get(key_a) or "")
                == visible(b[i].get(key_b) or "")))


def parity_compare(server: list[dict], box: list[dict]) -> dict:
    """the two sides' answers to the parity questions, compared on accuracy:
    {n, right_server, right_box, diff, lo, hi, same, letters, identical,
    self, problems, ok, words} — `diff` the box's share right minus the
    server's on the same questions, [lo, hi] its 90% paired interval"""
    problems = []
    for side, rows in (("the server's", server), ("the box's", box)):
        ids = [str(r.get("id")) for r in rows]
        twice = sorted({i for i in ids if ids.count(i) > 1})
        if twice:
            problems.append(f"{side} file holds question{'s' if len(twice) > 1 else ''} "
                            f"{', '.join(twice[:5])} more than once")
    a = {str(r["id"]): r for r in server}
    b = {str(r["id"]): r for r in box}
    if set(a) != set(b):
        problems.append(f"the two files hold other questions ({len(set(a) ^ set(b))} on one side "
                        "only): ask both with the same --n")
    ids = sorted(set(a) & set(b))
    n = len(ids)
    # 17c: two sides that read no letter aren't the same answer — each side
    # reads one from at least half its answers, or there is nothing to compare
    read = [sum(1 for i in ids if read_mmlu_pro(x[i].get("answer") or "")) for x in (a, b)]
    if not n or min(read) < n / 2:
        problems.append(f"a letter was read from {read[0]:,} of the server's answers and "
                        f"{read[1]:,} of the box's, of {n:,}: too few to compare (each side "
                        "needs half)")

    def right(x: dict, key: str = "answer") -> int:
        return int(bool(read_mmlu_pro(x.get(key) or "") and
                        read_mmlu_pro(x.get(key) or "") == x.get("key")))
    d = [right(b[i]) - right(a[i]) for i in ids]
    rs, rb = sum(right(a[i]) for i in ids), sum(right(b[i]) for i in ids)
    margin, z = PARITY["margin"], PARITY["z"]
    if n:
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        se = (var / n) ** 0.5
        lo, hi = mean - z * se, mean + z * se
    else:
        mean = lo = hi = 0.0
    same = bool(n) and lo >= -margin and hi <= margin
    both, letters, identical = _agree(a, b)
    own = {i: x for i, x in b.items() if x.get("answer2") is not None}
    self_agree = _agree(own, own, "answer", "answer2") if own else None
    pt = lambda x: f"{100 * x:+.1f}"                         # noqa: E731
    head = (f"the box answers {100 * rb / max(1, n):.1f}% right and the server "
            f"{100 * rs / max(1, n):.1f}% on the same {n:,} questions — a difference of "
            f"{pt(mean)} points, 90% interval {pt(lo)} to {pt(hi)}, "
            f"{'inside' if same else 'not inside'} ±{100 * margin:.0f} points")
    info = (f"For information: the same letter on {letters:,} of the {both:,} read on both "
            f"sides, {identical:,} identical replies"
            + (f"; the box against a second run of itself: the same letter on {self_agree[1]:,} "
               f"of {self_agree[0]:,}, {self_agree[2]:,} identical" if self_agree else ""))
    ok = not problems and same
    return {"n": n, "right_server": rs, "right_box": rb, "diff": mean, "lo": lo, "hi": hi,
            "same": same, "letters": letters, "both": both, "identical": identical,
            "self": self_agree, "problems": problems, "ok": ok,
            "words": ("The same: " if ok else "Not the same: ")
            + "; ".join([*problems, head] if problems else [head]) + ". " + info + "."}


def group_of(task: str, item: dict) -> str:
    """what a question's score is averaged within first: ARC-AGI-2's task (a
    task of two test grids scores half for one); every other question alone"""
    return str(item.get("group") or item["id"])


def summary(per_question: dict[str, list[float]], task: str = "",
            groups: dict[str, str] | None = None) -> dict:
    """the share right over every run of every question, and its standard
    error over questions (the runs of one question move together). ARC-AGI-2
    (pass@2): a test grid is solved when either attempt is, a task scores the
    share of its grids solved, and the score is the tasks' mean"""
    if task and BENCH[task].get("aggregate") == "pass@2":
        solved = {q: 1.0 if any(v) else 0.0 for q, v in per_question.items() if v}
        by: dict[str, list[float]] = {}
        for q, x in solved.items():
            by.setdefault((groups or {}).get(q, q), []).append(x)
        per_question = {g: [sum(v) / len(v)] for g, v in by.items()}
    means = [sum(v) / len(v) for v in per_question.values() if v]
    n = len(means)
    if not n:
        return {"score": None, "se": None, "n": 0}
    mu = sum(means) / n
    var = sum((x - mu) ** 2 for x in means) / (n - 1) if n > 1 else 0.0
    return {"score": mu, "se": (var / n) ** 0.5, "n": n}


def doc_of(task: str, item: dict) -> dict:
    """what a samples line keeps of a question: its id (and ARC's task) —
    never its text, which stays in this machine's copy of the dataset"""
    return {"id": item["id"], **({"group": item["group"]} if item.get("group") else {})}


def target_of(task: str, need: dict):
    return need.get("key", need.get("answer"))


def source_name(task: str) -> str:
    s = BENCH[task]["source"]
    return s.get("hf") or s.get("github") or ""


def cached(task: str, root: Path) -> list[dict] | None:
    """the task's questions if this machine has them — never fetched: a page
    asking must not reach the network"""
    p = _cache(root, task)
    try:
        return _read_cache(p)["items"] if p.exists() else None
    except (OSError, ValueError, KeyError):
        return None


def unlisted(task: str) -> str:
    """'' when the task's questions may be shown; else why not"""
    u = BENCH.get(task, {}).get("unlisted")
    return (u if isinstance(u, str) else
            f"{BENCH[task]['label']}'s questions are never shown, as its authors ask") if u else ""


def verdict(task: str, run: dict) -> str:
    """one run's verdict in words: what was read, or why nothing was, and the
    grader's or the model check's word when there is one"""
    g = run.get("grade") or {}
    if run.get("ran_out"):
        return "ran out of room before its answer"
    bits = [f"read as {str(run['read'])[:80]}" if run.get("read") is not None
            else "no answer read"]
    if g:
        bits.append(f"{g.get('by') or 'the grader'}: {g.get('words') or ('right' if g.get('ok') else 'wrong')}")
    return " · ".join(bits)


def shown(task: str, item: dict) -> dict:
    """what the question viewer shows of a question: {q, options, reference,
    subject, context} — never for an unlisted benchmark"""
    if unlisted(task):
        raise PermissionError(unlisted(task))
    if task == "mmlupro_tiger":
        return {"q": item["question"], "options": item["options"],
                "reference": item["answer"], "subject": item["category"]}
    if task == "arc_agi2_public":
        tid, i = item["id"].split("#")
        return {"q": f"Task {tid}, test grid {int(i) + 1}: the test input "
                     f"{json.dumps(item['test_input'])}", "options": [],
                "reference": json.dumps(item["answer"]), "subject": "",
                "context": "\n".join(f"{json.dumps(p['input'])} → {json.dumps(p['output'])}"
                                     for p in item["train"])}
    return {"q": str(item.get("question") or ""), "options": [],
            "reference": item.get("answer"), "subject": item.get("subject") or ""}
