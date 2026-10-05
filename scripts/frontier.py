"""17: the Frontier benchmarks, asked the way their owners and Epoch AI ask
them — the definitions, the prompts and the code scorers. No service import:
the board (service/frontier.py) and a rented box (scripts/remote_gguf.py) read
the same file, so a question is asked the same way on both.

Stage 1 holds GPQA Diamond; stage 2 adds the others.

GPQA Diamond, as Epoch AI runs it (epoch.ai/benchmarks/gpqa-diamond, v1.0.6,
20 Feb 2026): zero-shot, the four choices shuffled once per question, the
prompt below word for word, and Inspect's choice() parse — the last line that
is "ANSWER: X"; a reply without one scores 0. Epoch averages 16 runs; the board
4, with the error over questions (decided 5 Oct). Our shuffle is seeded, so a
run can be repeated; Epoch's is not.

The questions are never in the repo, the runner image or a bundle: they are
fetched where the run is, with the person's token (GPQA is gated, and its
authors ask that its questions never appear in plain text online), and kept
on that machine's disk only.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

VERSION = "frontier-1"
LETTERS = "ABCD"

# Epoch AI's GPQA Diamond template, v1.0.6 (20 Feb 2026), word for word
GPQA_TEMPLATE = ("Answer the following multiple choice question. The last line of your response "
                 "should be of the following format: 'ANSWER: LETTER' (without quotes) where "
                 "LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n\n"
                 "A) {A}\nB) {B}\nC) {C}\nD) {D}")

BENCH: dict[str, dict] = {
    "gpqa_diamond_epoch": {
        "label": "GPQA Diamond", "group": "Science",
        # the names others report it under (the Frontier view joins on these)
        "reported_as": ["gpqa diamond"],
        "protocol": "Epoch AI's (v1.0.6): zero-shot, ANSWER: LETTER, Inspect's choice() parse",
        "protocol_version": "epoch-gpqa-1.0.6",
        "source": {"hf": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                   "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6", "gated": True,
                   "licence": "CC BY 4.0"},
        "n": 198, "epochs": 4, "budget": {"on": 32768, "off": 4096},
        # the prompt's room in a slot's context, above the budget
        "room": 2048, "scorer": "choice", "unlisted": True},
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


def _fetch(task: str) -> list[dict]:
    """the dataset at its pinned revision, with HF_TOKEN for a gated one"""
    from datasets import load_dataset
    s = BENCH[task]["source"]
    ds = load_dataset(s["hf"], s.get("config"), split=s["split"], revision=s["revision"],
                      token=os.environ.get("HF_TOKEN") or None)
    if task == "gpqa_diamond_epoch":
        return [{"id": str(r["Record ID"]), "question": r["Question"].strip(),
                 "right": r["Correct Answer"].strip(),
                 "wrong": [r[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)]}
                for r in ds]
    raise KeyError(task)


def load(task: str, root: Path) -> list[dict]:
    """the task's questions, in the dataset's order — read once, then from this
    machine's own copy"""
    p = _cache(root, task)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    rows = _fetch(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return rows


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


def prompt(task: str, item: dict) -> tuple[str, dict]:
    """(the message, what scoring needs: the right letter)"""
    if task == "gpqa_diamond_epoch":
        opts = order_of(task, item)
        text = GPQA_TEMPLATE.format(question=item["question"], **dict(zip(LETTERS, opts)))
        return text, {"key": LETTERS[opts.index(item["right"])]}
    raise KeyError(task)


_THINK = re.compile(r"(?s)^\s*(?:<think>)?.*?</think>")
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
    return _THINK.sub("", text or "", count=1).strip()


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


def score(task: str, answer: str, finish: str | None, need: dict) -> dict:
    """{ok, read, ran_out} for one answer"""
    out = ran_out(answer, finish)
    if task == "gpqa_diamond_epoch":
        got = None if out else read_choice(answer)
        return {"ok": got == need["key"], "read": got, "ran_out": out}
    raise KeyError(task)


def summary(per_question: dict[str, list[float]]) -> dict:
    """the share right over every run of every question, and its standard
    error over questions (the runs of one question move together)"""
    means = [sum(v) / len(v) for v in per_question.values() if v]
    n = len(means)
    if not n:
        return {"score": None, "se": None, "n": 0}
    mu = sum(means) / n
    var = sum((x - mu) ** 2 for x in means) / (n - 1) if n > 1 else 0.0
    return {"score": mu, "se": (var / n) ** 0.5, "n": n}
