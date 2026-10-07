"""17i: what checking 17h found (docs/prompts/phase-17i-seventh-review-hle-
reader-export-fetch.md). Points 3 to 6 are the grading rules test's
(tests/test_17h_grading_rules.py); each other point has its test here, and
each fails on 511854e."""

from __future__ import annotations

import json

import pytest

import frontier_graders as fg


# ---------------------------------------------------------------------------
# part 1, point 1: HLE — each field compared with the prompt's own text for
# that field, whole and from its start; an example is an object every one of
# whose fields matches; a reply that ends cut is no grade
# ---------------------------------------------------------------------------

def _o(**k) -> str:
    return json.dumps({"extracted_final_answer": "42", "reasoning": "it matches",
                       "correct": "yes", "confidence": 90, **k})


NO_ANSWER = "There is no exact, final answer to extract from the response."

POINT_1 = [
    # real verdicts whose reasoning echoes the prompt: thrown away on 511854e,
    # each a try paid three times
    ("no exact answer", _o(reasoning=NO_ANSWER, correct="no"), False),
    ("echoes the extracted line",
     _o(reasoning="The final exact answer extracted from the response is 42, which matches "
                  "the correct_answer."), True),
    ("echoes the differences line",
     _o(reasoning="There are meaningful differences between [correct_answer] and the "
                  "extracted_final_answer.", correct="no"), False),
    ("echoes the margin line",
     _o(reasoning="Within a small margin of error for numerical problems."), True),
    ("None, no exact answer",
     _o(reasoning="No exact, final answer to extract from the response",
        extracted_final_answer="None", correct="no"), False),
    # a quoted "yes" before the real "no": no, never the quote's yes
    ("a quoted yes, then the real no",
     'The response ends with {"correct": "yes"}.\n' + _o(reasoning=NO_ANSWER, correct="no"),
     False),
    # an example alone is no grade
    ("a <…> example alone",
     '{"extracted_final_answer": "<the extracted answer>", "reasoning": "<your reasoning>", '
     '"correct": "yes", "confidence": "<0-100>"}', None),
    ("a <…> example, its confidence the prompt's 100",
     '{"extracted_final_answer": "<answer>", "reasoning": "<reasoning>", "correct": "yes", '
     '"confidence": 100}', None),
    ("a … example", '{"extracted_final_answer": "...", "reasoning": "...", "correct": "yes", '
                    '"confidence": "..."}', None),
    ("a real verdict, its confidence 100", _o(confidence=100), True),
    # a reply that ends cut is no grade, whatever came before
    ("a real yes, then an object cut", _o() + '\n{"extracted_final_answer": "42", "reas', None),
    ("cut before correct, after an inline quote",
     'extracted_final_answer: 42\nreasoning: the response printed {"correct": "yes"} and then it',
     None),
]

# point 2: what a person reads at a glance — unread on 511854e and 0adb522
POINT_2 = [
    ("raw LaTeX backslashes",
     '{"extracted_final_answer": "\\alpha = 2", "reasoning": "\\alpha matches", '
     '"correct": "yes", "confidence": 90}', True),
    ("a trailing comma",
     '{"extracted_final_answer": "42", "reasoning": "ok", "correct": "no", "confidence": 80,}',
     False),
    ("single quotes",
     "{'extracted_final_answer': '42', 'reasoning': 'ok', 'correct': 'yes', 'confidence': 85}",
     True),
]


@pytest.mark.parametrize("name,reply,ok", POINT_1 + POINT_2, ids=[c[0] for c in POINT_1 + POINT_2])
def test_1_2_hle_reads_each_field_against_its_own_prompt_text(name, reply, ok):
    assert fg.read("hle", reply, {"id": "h1", "answer": "42"})["ok"] is ok, name
