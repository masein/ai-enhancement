"""The permutation control: MMLU with the answer options rotated by doc_id mod 4.

Every position-skewed model on the board is 360M or smaller. That is either a
property of small models or an artefact of how we pose MMLU — five shots, the
answer letter after "Answer:", and a real answer key that is not uniform over
A–D. One run settles it: ask the same questions with the options cycled so the
correct answer visits every slot exactly as often, and see whether the score
moves. If it does, the skew is ours and the fix is the prompt; if it holds, the
ceiling the Diagnose page shows is the model's real ceiling.

Cyclic rotation by doc_id, not a random shuffle: deterministic without a seed,
identical for every model, and the correct answer's slot is (answer - doc_id)
mod 4, which is uniform over any run of four consecutive items whatever the
original key looks like. The option TEXT is untouched — this changes where the
right answer sits, and nothing else.
"""

from __future__ import annotations

N_OPTIONS = 4


def rotate(doc: dict, idx: int) -> dict:
    """One item with its options cycled left by idx mod 4. Returns a new dict;
    `perm_shift` and `orig_answer` ride along so a per-item log can be undone."""
    k = idx % N_OPTIONS
    choices = list(doc["choices"])
    answer = int(doc["answer"])
    return {**doc,
            "choices": choices[k:] + choices[:k],
            "answer": (answer - k) % N_OPTIONS,
            "perm_shift": k,
            "orig_answer": answer}


def process_docs(dataset):
    """lm_eval's process_docs hook: a datasets.Dataset in, one out. Applied to
    the test split and the few-shot dev split alike, so the examples in the
    prompt are rotated too and every prompt is deterministic."""
    return dataset.map(rotate, with_indices=True)
