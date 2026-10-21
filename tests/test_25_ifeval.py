"""25: IFEval on rented boxes, checked by its official checkers (the ones
DeviceMark's IFEval uses, scripts/ifeval_official) on the reply alone: five
of its kinds of instruction each followed by one reply and broken by
another, the four numbers as the checkers give them, the thinking never
counting, and an answer that ran out of room following nothing. Invented
prompts, the instructions and their arguments real; no model runs."""

from __future__ import annotations

import pytest

import frontier as fb
import instruction_following as inf

TASK = "ifeval_google"
END = "Is there anything else I can help with?"
CASES = [
    ("punctuation:no_comma", {}, "No commas here at all.", "Yes, there is one."),
    ("detectable_format:number_bullet_lists", {"num_bullets": 3},
     "* one\n* two\n* three", "* one\n* two"),
    ("startend:end_checker", {"end_phrase": END}, f"Done. {END}", "Done."),
    ("change_case:english_lowercase", {},
     "this whole reply is written in english and in lowercase letters only.",
     "This Whole Reply Is Written In English With Capitals."),
    ("length_constraints:number_words", {"num_words": 10, "relation": "at least"},
     "one two three four five six seven eight nine ten eleven twelve", "three words only"),
]


def item(k: int, ids: list[str], kws: list[dict]) -> dict:
    return {"id": str(1000 + k), "question": f"Invented prompt {k}.", "instructions": ids,
            "kwargs": kws}


def official(it: dict, reply: str) -> dict:
    """IFEval's checkers as lm_eval calls them"""
    from ifeval_official import utils as U
    inp = U.InputExample(key=0, instruction_id_list=it["instructions"], prompt=it["question"],
                         kwargs=it["kwargs"])
    return {"strict": U.test_instruction_following_strict(inp, reply).follow_instruction_list,
            "loose": U.test_instruction_following_loose(inp, reply).follow_instruction_list}


def score(it: dict, reply: str, finish: str = "stop") -> dict:
    text, need = fb.prompt(TASK, it)
    assert text == it["question"] and fb.system_of(TASK) is None   # as it is, no system prompt
    return fb.score(TASK, reply, finish, need)


def test_five_kinds_followed_and_broken_and_the_four_numbers():
    follows = []
    for k, (iid, kw, good, bad) in enumerate(CASES):
        it = item(k, [iid], [kw])
        for reply, want in ((good, True), (bad, False)):
            sc = score(it, reply)
            assert sc["ok"] is want, (iid, reply)
            assert {h: sc["follow"][h] for h in ("strict", "loose")} == official(it, reply)
            follows.append(sc["follow"])
    # loose sets a last line aside: the phrase ends the reply but for its sign-off
    it = item(7, ["startend:end_checker"], [{"end_phrase": END}])
    sc = score(it, f"Done. {END}\nThanks!")
    assert sc["follow"]["strict"] == [False] and sc["follow"]["loose"] == [True]
    assert sc["ok"] is False                                  # IFEval's measure is strict
    follows.append(sc["follow"])
    # two instructions, one followed: no prompt, half an instruction-level
    it = item(8, ["punctuation:no_comma", "change_case:english_lowercase"], [{}, {}])
    sc = score(it, "all of this is written in lowercase, though with a comma in it.")
    assert sc["follow"]["strict"] == [False, True] and sc["ok"] is False
    follows.append(sc["follow"])
    n = inf.numbers(follows)
    assert (n["prompts"], n["instructions"]) == (12, 13)
    assert n["prompt_strict"]["v"] == pytest.approx(5 / 12)
    assert n["prompt_loose"]["v"] == pytest.approx(6 / 12)
    assert n["inst_strict"]["v"] == pytest.approx(6 / 13)
    assert n["inst_loose"]["v"] == pytest.approx(7 / 13)
    assert n["prompt_strict"]["se"] == pytest.approx(((5 / 12) * (7 / 12) * 12 / 11 / 12) ** 0.5)


def test_the_thinking_never_counts_and_an_answer_cut_off_follows_nothing():
    it = item(0, ["punctuation:no_comma"], [{}])
    plain = score(it, "No commas here at all.")
    # commas in the thinking, none in the reply: followed
    thought = score(it, "<think>\nWell, first, I will, carefully, avoid commas.\n</think>\n\n"
                        "No commas here at all.")
    assert thought["follow"] == plain["follow"] and thought["ok"] is True
    # and the reverse: a clean thinking never rescues a reply with a comma
    assert score(it, "<think>\nno commas\n</think>\n\nYes, one.")["ok"] is False
    # ran out of room: every instruction failed, the reply never checked
    for ans, fin in (("No commas here at all.", "length"), ("<think>\nstill thinking", "stop")):
        sc = score(it, ans, fin)
        assert sc["ran_out"] and sc["ok"] is False and sc["follow"]["loose"] == [False]
    # an empty reply follows nothing, as the checkers say
    assert score(it, "<think>\nall said here\n</think>\n\n")["ok"] is False


def test_an_unset_argument_draws_the_same_every_time():
    """IFEval's checkers draw an argument the dataset leaves out at random:
    seeded, so a reply checks the same each time — and the process's own
    random numbers are left as they were"""
    import random
    it = item(0, ["detectable_format:number_bullet_lists"], [{}])     # num_bullets unset
    reply = "* a\n* b"
    random.seed(1)
    before = random.random()
    random.seed(1)
    inf._KEPT.clear()
    first = score(it, reply)["follow"]
    assert random.random() == before                          # the process's own, as it was
    for _ in range(3):
        inf._KEPT.clear()                                     # worked out again each time
        assert score(it, reply)["follow"] == first
