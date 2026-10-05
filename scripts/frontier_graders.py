"""17, stage 3: the Frontier benchmarks' graders — each one's prompt, word
for word (scripts/grader_prompts/), how it is filled in and how its reply is
read. No service import: service/frontier_grade.py sends them, on the server
only, after masein's Start.

- SimpleQA Verified: Google's grader prompt from its own starter code (the
  Kaggle notebook it publishes with the dataset), and its reading — the first
  A, B or C, else the words, else NOT_ATTEMPTED; graded by gpt-4.1-2025-04-14
  in Google's runs. The share graded CORRECT is the score (Epoch's).
- Humanity's Last Exam: CAIS's judge prompt (hle_eval/run_judge_results.py,
  its typos and |\\%| as they are), judged by o3-mini-2025-01-31 in CAIS's
  runs. CAIS asks for structured output; its fields are read from the reply's
  text here ("correct: yes"). A reply nothing can be read from is wrong, as a
  question CAIS's judge fails on counts against the accuracy.
- MATH Level 5: Epoch AI's equivalence prompt (its scorer.py), expression 1
  the key and 2 the answer as Epoch's code extracted it; "Yes" is right. Epoch
  named gemini-1.5-flash-002, which is retired. Asked only of the answers the
  code marks wrong (an answer the code can't read, Epoch scores wrong without
  asking).
- OTIS Mock AIME: Epoch extracts the final answer with a model that doesn't
  see the key, then matches it exactly; its prompt isn't published, so the
  one here is ours (otis_extract.txt). Asked only of the answers the code marks
  wrong or can't read.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "grader_prompts"

GRADERS: dict[str, dict] = {
    "simpleqa": {
        "label": "SimpleQA Verified's grader", "task": "simpleqa_epoch",
        "prompt": "simpleqa_google.txt", "prompt_words": "Google's grader prompt",
        "suggested": "openai/gpt-4.1", "owners": "gpt-4.1-2025-04-14",
        "why": "the model Google grades SimpleQA Verified with",
        "does": "grades every answer CORRECT, INCORRECT or NOT_ATTEMPTED",
        "max_tokens": 16, "out_tokens": 2},
    "hle": {
        "label": "Humanity's Last Exam's judge", "task": "hle_text_cais",
        "prompt": "hle_judge.txt", "prompt_words": "CAIS's judge prompt",
        "suggested": "openai/o3-mini", "owners": "o3-mini-2025-01-31",
        "why": "the judge CAIS grades Humanity's Last Exam with",
        "does": "judges every answer against the key, yes or no",
        # o3-mini thinks before it answers: its reasoning tokens are billed
        "max_tokens": 4096, "out_tokens": 900},
    "math": {
        "label": "MATH Level 5's equivalence check", "task": "math_l5_epoch",
        "prompt": "math_equivalence.txt", "prompt_words": "Epoch AI's equivalence prompt",
        "suggested": "google/gemini-2.5-flash", "owners": "gemini-1.5-flash-002 (retired)",
        "why": "Epoch's check, gemini-1.5-flash-002, is retired: the same family's",
        "does": "a second look at the answers the code marks wrong: equivalent to the key?",
        "max_tokens": 16, "out_tokens": 2, "look": True},
    "otis": {
        "label": "OTIS Mock AIME's extractor", "task": "otis_aime_epoch",
        "prompt": "otis_extract.txt", "prompt_words": "our extraction prompt (Epoch's isn't "
                                                       "published)",
        "suggested": "google/gemini-2.5-flash", "owners": "not published",
        "why": "a small model that reads an answer well; Epoch's isn't named",
        "does": "a second look at the answers the code marks wrong or can't read: the final "
                "answer, read without the key",
        "max_tokens": 32, "out_tokens": 4, "look": True},
}
SLOT_OF = {g["task"]: s for s, g in GRADERS.items()}


def prompt_text(slot: str) -> str:
    return (PROMPTS / GRADERS[slot]["prompt"]).read_text(encoding="utf-8")


def prompt_sha(slot: str) -> str:
    """the grader's prompt, pinned: shown beside the scores it produced"""
    return hashlib.sha256(prompt_text(slot).encode("utf-8")).hexdigest()


def render(slot: str, item: dict, run: dict) -> str:
    """the grader's message for one answer: `item` the question as the dataset
    has it, `run` the answer as service/frontier.marks scored it"""
    t = prompt_text(slot)
    answer = _visible(run.get("answer") or "")
    if slot == "simpleqa":
        return (t.replace("{question}", item["question"]).replace("{target}", item["answer"])
                .replace("{predicted_answer}", answer))
    if slot == "hle":
        return (t.replace("{question}", item["question"]).replace("{response}", answer)
                .replace("{correct_answer}", item["answer"]))
    if slot == "math":
        return t % {"expression1": item["answer"], "expression2": run.get("read") or ""}
    if slot == "otis":
        return t.replace("{response}", answer)
    raise KeyError(slot)


def _visible(text: str) -> str:
    return re.sub(r"(?s)^\s*(?:<think>)?.*?</think>", "", text or "", count=1).strip()


_SQA_WORDS = {"A": "correct", "B": "incorrect", "C": "not attempted"}


def read(slot: str, text: str, item: dict) -> dict:
    """the grader's reply, as its owners read it: {ok, words, said} — `said`
    what it answered, short"""
    text = text or ""
    said = text.strip()[:200]
    if slot == "simpleqa":
        # Google's reading: the first A, B or C; else the words; else C
        m = re.search(r"(A|B|C)", text)
        if m:
            g = m.group(0)
        elif "CORRECT" in text.upper():
            g = "A"
        elif "INCORRECT" in text.upper():
            g = "B"
        else:
            g = "C"
        return {"ok": g == "A", "words": _SQA_WORDS[g], "said": said, "grade": g}
    if slot == "hle":
        m = re.search(r"(?im)^\W*correct\W*:\W*(yes|no)\b", text)
        conf = re.search(r"(?im)^\W*confidence\W*:\W*(\d{1,3})", text)
        ext = re.search(r"(?im)^\W*extracted_final_answer\W*:\s*(.+)$", text)
        if not m:
            return {"ok": False, "words": "the judge's reply couldn't be read: wrong",
                    "said": said}
        ok = m.group(1).lower() == "yes"
        return {"ok": ok, "words": ("the judge: correct" if ok else "the judge: incorrect")
                + (f" · read as {ext.group(1).strip()[:60]}" if ext else ""),
                "said": said, **({"confidence": int(conf.group(1))} if conf else {})}
    if slot == "math":
        ok = text.strip().lower() == "yes"
        return {"ok": ok, "words": "equivalent" if ok else "not equivalent", "said": said}
    if slot == "otis":
        m = re.fullmatch(r"\s*\**\s*(-?\d{1,4})\s*\**\.?\s*", text)
        if not m:
            return {"ok": False, "words": "no final answer read", "said": said}
        got = int(m.group(1))
        ok = got == int(str(item["answer"]).strip())
        return {"ok": ok, "words": f"read as {got}", "said": said, "read": str(got)}
    raise KeyError(slot)
