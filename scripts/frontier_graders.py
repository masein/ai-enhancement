"""17, stage 3: the Frontier benchmarks' graders — each one's prompt, word
for word (scripts/grader_prompts/), how it is filled in and how its reply is
read. No service import: service/frontier_grade.py sends them, on the server
only, after masein's Start.

- SimpleQA Verified: Google's grader prompt from its own starter code (the
  Kaggle notebook it publishes with the dataset); graded by gpt-4.1-2025-04-14
  in Google's runs. The share graded CORRECT is the score (Epoch's). 17c: a
  grade only in its exact form — a lone letter, a lone CORRECT, INCORRECT or
  NOT_ATTEMPTED, or the prompt's own "B: INCORRECT", with only punctuation
  or markup around it; anything else is no grade, asked again. Google's own
  reading (the first A, B or C anywhere) read "NOT_ATTEMPTED" and "As an
  AI…" as A, and 17b's whole words "not correct" as correct.
- Humanity's Last Exam: CAIS's judge prompt (hle_eval/run_judge_results.py,
  its typos and |\\%| as they are), judged by o3-mini-2025-01-31 in CAIS's
  runs, at its default reasoning effort and a 4,096-token cap. CAIS asks for
  structured output; its fields are read from the reply here, as text
  ("correct: yes") or JSON ("correct": "yes"), the last one given, its value
  alone ("correct: yes/no" isn't one). 17b: a reply nothing can be read from
  isn't a grade: asked again — 17c: its words never shown (the judge may
  quote HLE's question, which is gated).
- MATH Level 5: Epoch AI's equivalence prompt (its scorer.py), expression 1
  the key and 2 the answer as Epoch's code extracted it; "Yes" is right (17c:
  yes or no alone, punctuation or markup around it; anything else, asked
  again — it was a final "not equivalent"). Epoch
  named gemini-1.5-flash-002, which is retired. Asked only of the answers the
  code marks wrong (an answer the code can't read, Epoch scores wrong without
  asking).
- OTIS Mock AIME: Epoch extracts the final answer with a model that doesn't
  see the key, then matches it exactly; its prompt isn't published, so the
  one here is ours (otis_extract.txt). Asked only of the answers the code marks
  wrong or can't read. An integer is read; NONE is no final answer (wrong);
  anything else isn't a grade.

17b: each grader's reasoning is set, never left to the model's default
(`reasoning`, OpenRouter's switch, which a model that doesn't reason
ignores), and its cap sized for it — reasoning tokens count against the cap.
The owners' graders as their owners ran them: gpt-4.1 doesn't reason;
o3-mini at medium, its default, which CAIS kept; Epoch's MATH check,
gemini-1.5-flash-002, didn't reason, so neither does its suggested successor.
A chosen model that reasons is given REASONING_ROOM more, in case it can't be
switched off. An empty reply, or one cut at the cap, is never a grade.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "grader_prompts"
REASONING_ROOM = 2048       # 17b: a model that reasons when told not to still answers
OFF = {"enabled": False}

GRADERS: dict[str, dict] = {
    "simpleqa": {
        "label": "SimpleQA Verified's grader", "task": "simpleqa_epoch",
        "prompt": "simpleqa_google.txt", "prompt_words": "Google's grader prompt",
        "suggested": "openai/gpt-4.1", "owners": "gpt-4.1-2025-04-14",
        "why": "the model Google grades SimpleQA Verified with",
        "does": "grades every answer CORRECT, INCORRECT or NOT_ATTEMPTED",
        "reasoning": OFF, "reasoning_words": "reasoning off, as gpt-4.1 answers",
        "max_tokens": 16, "out_tokens": 2},
    "hle": {
        "label": "Humanity's Last Exam's judge", "task": "hle_text_cais",
        "prompt": "hle_judge.txt", "prompt_words": "CAIS's judge prompt",
        "suggested": "openai/o3-mini", "owners": "o3-mini-2025-01-31",
        "why": "the judge CAIS grades Humanity's Last Exam with",
        "does": "judges every answer against the key, yes or no",
        # o3-mini thinks before it answers: its reasoning tokens are billed, and
        # count against CAIS's own cap (max_completion_tokens=4096)
        "reasoning": {"effort": "medium"},
        "reasoning_words": "reasoning at medium effort, o3-mini's default, as CAIS ran it",
        "max_tokens": 4096, "out_tokens": 900},
    "math": {
        "label": "MATH Level 5's equivalence check", "task": "math_l5_epoch",
        "prompt": "math_equivalence.txt", "prompt_words": "Epoch AI's equivalence prompt",
        "suggested": "google/gemini-2.5-flash", "owners": "gemini-1.5-flash-002 (retired)",
        "why": "Epoch's check, gemini-1.5-flash-002, is retired: the same family's",
        "does": "a second look at the answers the code marks wrong: equivalent to the key?",
        "reasoning": OFF, "reasoning_words": "reasoning off, as gemini-1.5-flash-002 answered",
        "max_tokens": 16, "out_tokens": 2, "look": True},
    "otis": {
        "label": "OTIS Mock AIME's extractor", "task": "otis_aime_epoch",
        "prompt": "otis_extract.txt", "prompt_words": "our extraction prompt (Epoch's isn't "
                                                       "published)",
        "suggested": "google/gemini-2.5-flash", "owners": "not published",
        "why": "a small model that reads an answer well; Epoch's isn't named",
        "does": "a second look at the answers the code marks wrong or can't read: the final "
                "answer, read without the key",
        "reasoning": OFF, "reasoning_words": "reasoning off: it copies a number out",
        "max_tokens": 32, "out_tokens": 4, "look": True},
}
SLOT_OF = {g["task"]: s for s, g in GRADERS.items()}


def prompt_text(slot: str) -> str:
    return (PROMPTS / GRADERS[slot]["prompt"]).read_text(encoding="utf-8")


def prompt_sha(slot: str) -> str:
    """the grader's prompt, pinned: shown beside the scores it produced"""
    return hashlib.sha256(prompt_text(slot).encode("utf-8")).hexdigest()


def ask(slot: str, reasons: bool | None = None) -> dict:
    """17b: what is sent with each answer, and what the dry run counts —
    {max_tokens, reasoning, out_tokens}. `reasons`: whether OpenRouter lists
    the chosen model as one that reasons"""
    g = GRADERS[slot]
    cap = g["max_tokens"]
    if reasons and g["reasoning"] == OFF:
        cap += REASONING_ROOM
    return {"max_tokens": cap, "reasoning": dict(g["reasoning"]), "out_tokens": g["out_tokens"]}


def _fill(t: str, values: dict) -> str:
    """17b: every placeholder filled in one pass — an answer holding
    "{correct_answer}" stays as it is, never the key"""
    return re.sub(r"\{(" + "|".join(map(re.escape, values)) + r")\}",
                  lambda m: values[m.group(1)], t)


def render(slot: str, item: dict, run: dict) -> str:
    """the grader's message for one answer: `item` the question as the dataset
    has it, `run` the answer as service/frontier.marks scored it"""
    t = prompt_text(slot)
    answer = _visible(run.get("answer") or "")
    if slot == "simpleqa":
        return _fill(t, {"question": item["question"], "target": item["answer"],
                         "predicted_answer": answer})
    if slot == "hle":
        return _fill(t, {"question": item["question"], "response": answer,
                         "correct_answer": item["answer"]})
    if slot == "math":
        return t % {"expression1": item["answer"], "expression2": run.get("read") or ""}
    if slot == "otis":
        return _fill(t, {"response": answer})
    raise KeyError(slot)


def _visible(text: str) -> str:
    return re.sub(r"(?s)^\s*(?:<think>)?.*?</think>", "", text or "", count=1).strip()


_SQA_WORDS = {"A": "correct", "B": "incorrect", "C": "not attempted"}


_SQA_OF = {"correct": "A", "incorrect": "B", "not_attempted": "C"}


def _last(pattern: str, text: str):
    ms = list(re.finditer(pattern, text))
    return ms[-1] if ms else None


# 17c: a grade only in its exact expected form — a lone letter, word or yes/no,
# with only punctuation or markup around it ("**B**", "(A)", "Yes.")
_WRAP = re.compile(r"^[\s*_`\"'#>(\[]+|[\s*_`\"'.!:)\]]+$")


def lone(text: str) -> str:
    """the reply without the punctuation and markup around it"""
    return _WRAP.sub("", text or "")


def _json_fields(text: str) -> dict | None:
    """the reply's JSON object, when the whole reply is one (a code fence
    around it allowed) — 17e: never an object quoted in a reply of text, which
    beat the verdict on its own line"""
    import json
    t = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, re.S | re.I)
    t = m.group(1).strip() if m else t
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        got = json.loads(t)
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def unread(text: str, show: bool = True) -> dict:
    """17b: a reply that isn't a grade — never kept as one; the next Start
    asks again. 17c: its words without the reply for a benchmark that is
    never shown (the judge may quote its question)"""
    return {"ok": None, "unread": "the grader's reply isn't a grade"
            + ((f": “{text.strip()[:80]}”" if text.strip() else ": it was empty") if show
               else ("" if text.strip() else ": it was empty")),
            "said": text.strip()[:200] if show else ""}


def read(slot: str, text: str, item: dict) -> dict:
    """the grader's reply, as its owners read it: {ok, words, said} — `said`
    what it answered, short — or 17b's unread(): {ok: None, unread}"""
    text = text or ""
    shown = slot not in ("hle", "math")            # 17c: never a gated question's words
    said = text.strip()[:200] if shown else ""
    if not text.strip():
        return unread(text, shown)
    if slot == "simpleqa":
        # 17c: a lone letter (as Google's reading takes it), a lone grade word,
        # or the prompt's own "B: INCORRECT" — never a word out of a sentence
        core = lone(text)
        m = re.fullmatch(r"([ABC])(?:\s*[:.)\-]\s*((?i:CORRECT|INCORRECT|NOT[ _]ATTEMPTED)))?",
                         core)
        w = re.fullmatch(r"(?i)(CORRECT|INCORRECT|NOT[ _]ATTEMPTED)", core)
        if m:
            g = m.group(1)
            if m.group(2) and _SQA_OF[re.sub(r"[ _]", "_", m.group(2).lower())] != g:
                return unread(text)
        elif w:
            g = _SQA_OF[re.sub(r"[ _]", "_", w.group(1).lower())]
        else:
            return unread(text)
        return {"ok": g == "A", "words": _SQA_WORDS[g], "said": said, "grade": g}
    if slot == "hle":
        # CAIS's fields: as JSON (its structured output), else each at the start
        # of its own line — 17d: the last such "correct:" line, its value
        # alone; never a "correct: no" inside the judge's reasoning, and a
        # carriage return isn't part of the value
        t = text.replace("\r", "")
        fields = _json_fields(t)
        if fields is not None and str(fields.get("correct", "")).strip().lower() in ("yes", "no"):
            ok = str(fields["correct"]).strip().lower() == "yes"
            ext = str(fields.get("extracted_final_answer") or "").strip()
            conf = re.fullmatch(r"\s*(\d{1,3})\s*%?\s*", str(fields.get("confidence", "")))
            return {"ok": ok, "words": ("the judge: correct" if ok else "the judge: incorrect")
                    + (f" · read as {ext[:60]}" if ext else ""), "said": said,
                    **({"confidence": int(conf.group(1))} if conf else {})}
        w = r"[\s*_`\"'#>-]*"
        m = _last(rf"(?im)^{w}correct{w}:{w}(yes|no)[\s*_`\"'.]*$", t)
        conf = _last(rf"(?im)^{w}confidence{w}:{w}(\d{{1,3}})\s*%?[\s*_`\"'.]*$", t)
        ext = _last(rf"(?im)^{w}extracted_final_answer{w}:\s*(.+)$", t)
        if not m:
            return unread(text, shown)
        ok = m.group(1).lower() == "yes"
        return {"ok": ok, "words": ("the judge: correct" if ok else "the judge: incorrect")
                + (f" · read as {ext.group(1).strip()[:60]}" if ext else ""),
                "said": said, **({"confidence": int(conf.group(1))} if conf else {})}
    if slot == "math":
        # 17c: yes or no, alone — "Yes." and "**Yes**" are yes; anything else
        # is no grade, where it was a final "not equivalent"
        core = lone(text).lower()
        if core not in ("yes", "no"):
            return unread(text, shown)
        ok = core == "yes"
        return {"ok": ok, "words": "equivalent" if ok else "not equivalent", "said": said}
    if slot == "otis":
        if re.fullmatch(r"\W*NONE\W*", text, re.I):
            return {"ok": False, "words": "no final answer", "said": said}
        m = re.fullmatch(r"\s*\**\s*(-?\d{1,4})\s*\**\.?\s*", text)
        if not m:
            return unread(text)
        got = int(m.group(1))
        ok = got == int(str(item["answer"]).strip())
        return {"ok": ok, "words": f"read as {got}", "said": said, "read": str(got)}
    raise KeyError(slot)
