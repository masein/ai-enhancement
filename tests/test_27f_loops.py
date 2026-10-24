"""27f: the loop rule on the part where an answer ended, alone — a finished
answer's reply, the thinking of one that ran out inside it, the reply of one
that ran out after its thinking closed (or has none). A reply drafted in the
thinking, checked and written out once is no loop: on step8804's IFEval and
IFBench, 81 of the 82 answers the whole text flagged were that. Invented
texts; nothing runs."""

from __future__ import annotations

import frontier as fb
from frontier_graders import think_split
from service import beyond

# a reply of about 900 characters, with no passage repeating inside it
REPLY = " ".join(f"Sentence {w} says something new about part {w}." for w in
                 ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                  "eighteen", "nineteen", "twenty"))
STUCK = "Wait, let me re-check the invented option against the question once more.\n" * 40


def drafted() -> str:
    """the reply written out twice in the thinking (a draft, then its check),
    then once as the reply"""
    return (f"<think>\nPlan the answer.\nDraft:\n{REPLY}\nCheck every constraint:\n{REPLY}\n"
            f"All good.\n</think>\n\n{REPLY}")


def test_a_reply_drafted_in_the_thinking_is_no_loop():
    t = drafted()
    assert fb.ends_in_loop(t)                    # the whole text: three copies of the end
    assert not fb.ended_in_loop(t, False)        # the reply alone: once
    assert not fb.ends_in_loop(REPLY)


def test_a_thinking_loop_that_ran_out_still_is_one():
    t = "<think>\nSome steps.\n" + STUCK               # never closed: cut inside its thinking
    assert think_split(t)[2]
    assert fb.ended_in_loop(t, True)
    # and a tail of nothing but newlines, where the thinking ran out
    assert fb.ended_in_loop("<think>\nSome steps." + "\n" * 1500, True)


def test_a_reply_that_loops_is_one_finished_or_cut():
    looping = REPLY + "\n" + STUCK
    assert fb.ended_in_loop(f"<think>\nShort.\n</think>\n\n{looping}", False)
    # ran out in its reply, its thinking closed: the reply is read
    assert fb.ended_in_loop(f"<think>\nShort.\n</think>\n\n{looping}", True)
    # no thinking at all (a thinking-off row): the whole text is its reply, its
    # trailing newlines kept as written
    assert fb.ended_in_loop(looping, True) and fb.ended_in_loop(looping, False)
    assert fb.ended_in_loop("Some words." + "\n" * 1500, True)
    assert think_split("Some words.\n\n\n", strip=False)[0] == "Some words.\n\n\n"
    assert think_split("Some words.\n\n\n")[0] == "Some words."          # unchanged by default


def test_a_finished_answer_is_never_read_in_its_thinking():
    # a loop in the thinking that closed, then a clean reply: finished, no loop
    t = f"<think>\n{STUCK}</think>\n\n{REPLY}"
    assert not fb.ended_in_loop(t, False)


def test_the_kept_measure_reads_each_answer_s_part():
    texts = [drafted(), "<think>\nSome steps.\n" + STUCK, f"<think>\nShort.\n</think>\n\n{REPLY}",
             REPLY + "\n" + STUCK]
    outs = [False, True, False, True]
    rows = [{"runs": [{"ran_out": o}], "texts": [t]} for t, o in zip(texts, outs)]
    got = beyond.loops(rows)
    # the draft isn't one; the thinking cut in a loop and the looping reply are
    assert (got["answers"], got["looped"], got["ran_out"], got["ran_out_looped"]) == (4, 2, 2, 2)
    assert got["share"] == 0.5 and got["share_of_ran_out"] == 1.0


def test_the_kept_measures_are_worked_out_again():
    assert beyond.WAY == "beyond-3"
