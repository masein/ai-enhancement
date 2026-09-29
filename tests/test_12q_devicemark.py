"""12q (the brief's): DeviceMark's protocol, here — the battery, the prompts,
the parsers, MATH's equality, IFEval's official checkers seeded, the cap, the
composite and its bootstrap, the ranks with ties, and MTP parity. No model
runs and nothing is fetched: the answers are written here."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import devicemark as dm

HERE = Path(__file__).resolve().parent
RAW_KEYS = HERE / "fixtures" / "devicemark" / "ifeval_keys_raw.txt"
SRC = json.loads(dm.SOURCE_IDS.read_text(encoding="utf-8"))
BAT = dm.battery()


# ---------------------------------------------------------------------------
# the battery
# ---------------------------------------------------------------------------

def test_ifeval_is_devicemarks_300_keys():
    raw = [x for x in RAW_KEYS.read_text().splitlines() if x and not x.startswith("#")]
    assert len(raw) == 300 and len(set(raw)) == 300
    assert set(BAT["ifeval"]) == set(raw) and len(BAT["ifeval"]) == 300
    # in google/IFEval's order, which is the source file's
    assert BAT["ifeval"] == SRC["ifeval_devicemark_keys"]
    assert "1122" in BAT["ifeval"] and "1129" in BAT["ifeval"]


def test_mmlu_pro_is_14_a_category_of_14():
    assert sorted(BAT["mmlu_pro"]) == sorted(SRC["mmlu_pro"]) and len(BAT["mmlu_pro"]) == 14
    for cat, ids in BAT["mmlu_pro"].items():
        assert len(ids) == 14 == len(set(ids)) and set(ids) <= set(SRC["mmlu_pro"][cat]), cat
    assert len(dm.flat(BAT, "mmlu_pro")) == 196


def test_math_is_100_in_math500s_proportions():
    sizes = {s: len(v) for s, v in SRC["math500"].items()}
    assert sum(sizes.values()) == 500
    got = {s: len(v) for s, v in BAT["math"].items()}
    assert got == dm.largest_remainder(sizes, 100) == {
        "Algebra": 25, "Intermediate Algebra": 20, "Prealgebra": 16, "Number Theory": 12,
        "Precalculus": 11, "Geometry": 8, "Counting & Probability": 8}
    for s, ids in BAT["math"].items():
        assert set(ids) <= set(SRC["math500"][s])
    assert len(dm.flat(BAT, "math")) == 100


def test_largest_remainder_is_exact_and_breaks_ties_by_size():
    # 19.4, 16.4 and 12.4 tie exactly (floats wouldn't): the largest subject wins
    assert dm.largest_remainder({"a": 97, "b": 82, "c": 62, "d": 259}, 100) == {
        "a": 20, "b": 16, "c": 12, "d": 52}
    assert sum(dm.largest_remainder({"x": 1, "y": 1, "z": 1}, 2).values()) == 2


def test_the_same_seed_gives_the_same_ids():
    once, again = dm.make_battery(SRC), dm.make_battery(json.loads(json.dumps(SRC)))
    assert once == again
    assert json.loads(dm.BATTERY_PATH.read_text(encoding="utf-8")) == once   # the committed one
    # another category's order in the file changes nothing
    shuffled = {**SRC, "mmlu_pro": dict(reversed(list(SRC["mmlu_pro"].items())))}
    assert dm.make_battery(shuffled)["mmlu_pro"] == once["mmlu_pro"]
    assert dm.main(["battery"]) == 0


def test_every_row_records_the_battery_version_and_the_parts_are_fixed():
    assert BAT["version"] == dm.VERSION == "devicemark-replica-v1"
    assert {b: len(v) for b, v in BAT["pilot"].items()} == dm.PARTS["pilot"]
    assert {b: len(v) for b, v in BAT["parity"].items()} == {"ifeval": 20, "mmlu_pro": 20,
                                                             "math": 10}
    for part in ("pilot", "parity"):
        for b, ks in BAT[part].items():
            assert set(ks) <= set(dm.flat(BAT, b))
    assert len(dm.keys_for("full")) == 596 and len(dm.keys_for("pilot")) == 30
    assert not any("question" in json.dumps(v) for k, v in BAT.items() if k in dm.BENCHES)


# ---------------------------------------------------------------------------
# the prompts
# ---------------------------------------------------------------------------

def test_the_prompts_are_built_as_the_file_says():
    p = json.loads(dm.PROMPTS_PATH.read_text(encoding="utf-8"))
    assert p["mmlu_pro"].endswith("Think step by step, then give your final answer as "
                                  "\\boxed{X}, where X is the letter of the correct option.")
    assert p["math"].endswith("Solve it step by step, then put your final answer in \\boxed{}.")
    assert dm.prompt_for({"bench": "ifeval", "prompt": "Write a {question} poem."}) == \
        "Write a {question} poem."                             # a filled value isn't read again
    mm = dm.prompt_for({"bench": "mmlu_pro", "question": "Which is a gas?",
                        "options": ["Iron", "Neon", "Salt"]})
    assert mm == ("Which is a gas?\n\nA. Iron\nB. Neon\nC. Salt\n\nThink step by step, then give "
                  "your final answer as \\boxed{X}, where X is the letter of the correct option.")
    assert dm.prompt_for({"bench": "math", "problem": "What is $1+1$?"}) == (
        "What is $1+1$?\n\nSolve it step by step, then put your final answer in \\boxed{}.")
    ten = dm.prompt_for({"bench": "mmlu_pro", "question": "q", "options": list("abcdefghij")})
    assert "\nJ. j\n" in ten


# ---------------------------------------------------------------------------
# the parsers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("so it is \\boxed{C}", ("C", "boxed")),
    ("\\boxed{\\text{C}}", ("C", "boxed")),
    ("\\boxed{\\textbf{(D)}}", ("D", "boxed")),
    ("\\boxed{C. Neon}", ("C", "boxed")),
    ("first \\boxed{A}, then on reflection \\boxed{B}", ("B", "boxed")),     # the last one wins
    ("\\boxed{Carbon}", (None, "")),                                         # not a letter
    ("The answer is (C).", ("C", "answer is")),                              # the tested fallbacks
    ("so the final answer is: D", ("D", "answer is")),
    ("Answer: E", ("E", "Answer:")),
    ("**Answer:** F", ("F", "Answer:")),
    ("I think C is right, as B is too big", (None, "")),                     # a letter in prose
    ("the answer is a trick question", (None, "")),
    ("", (None, "")),
])
def test_mmlu_pro_reads_the_last_box_or_a_tested_phrasing(text, want):
    assert dm.mmlu_letter(text) == want


def test_math_reads_the_last_box_with_nested_braces():
    assert dm.math_answer("so \\boxed{\\frac{1}{\\sqrt{2}}}.") == "\\frac{1}{\\sqrt{2}}"
    assert dm.math_answer("\\boxed{1} then \\boxed{\\dfrac{3}{4}}") == "\\dfrac{3}{4}"
    assert dm.math_answer("the answer is 7") is None                # no box, no answer
    assert dm.math_answer("\\boxed{\\frac{1}{2") is None            # cut off inside the box


@pytest.mark.parametrize("pred,gold,same", [
    ("1/2", "\\frac{1}{2}", True), ("0.5", "\\frac{1}{2}", True), ("0.5", "1/2", True),
    ("\\sqrt{8}", "2\\sqrt{2}", True), ("(-\\infty, 3]", "(-\\infty,3]", True),
    ("[1,2)", "[1,2]", False), ("3", "4", False), ("\\frac{1}{3}", "0.33", False),
])
def test_math_equality_is_symbolic(pred, gold, same):
    assert dm.math_equal(pred, gold) is same


def test_thinking_is_split_off_and_unfinished_thinking_has_no_answer():
    assert dm.split_thinking("<think>\nhmm\n</think>\n\n\\boxed{B}") == ("hmm", "\\boxed{B}")
    assert dm.split_thinking("<think>\nstill going") == ("still going", "")
    assert dm.split_thinking("plain \\boxed{A}") == ("", "plain \\boxed{A}")


# ---------------------------------------------------------------------------
# IFEval: the official checkers, seeded
# ---------------------------------------------------------------------------

# the two items whose kwargs leave the letter to chance ("#" and "!" aren't
# letters), with their instructions and kwargs as google/IFEval has them —
# the prompts are ours
ODD = [
    {"key": "1122", "instruction_id_list": ["change_case:english_lowercase",
                                            "keywords:letter_frequency"],
     "kwargs": [{}, {"let_relation": "at least", "letter": "#", "let_frequency": 4}],
     "prompt": "Write a short note about a fixture, all lowercase.",
     "response": "a short note about a fixture: aaaa eeee tttt, all of it lowercase."},
    {"key": "1129", "instruction_id_list": ["keywords:letter_frequency",
                                            "combination:repeat_prompt"],
     "kwargs": [{"let_relation": "at least", "letter": "!", "let_frequency": 6},
                {"prompt_to_repeat": "Write a fixture pitch."}],
     "prompt": "Write a fixture pitch. Then pitch it.",
     "response": "Write a fixture pitch. It's great!!!!!! eeeeee ssssss"},
]


def test_items_1122_and_1129_would_draw_a_letter_at_random():
    from ifeval_official import instructions_registry as R
    letters = set()
    for seed in range(8):
        random.seed(seed)
        ins = R.INSTRUCTION_DICT["keywords:letter_frequency"]("keywords:letter_frequency")
        ins.build_description(let_relation="at least", letter="#", let_frequency=4)
        letters.add(ins.get_instruction_args()["letter"])
    assert len(letters) > 1                                        # why the scorer seeds it


def test_ifeval_scores_the_same_twice_1122_and_1129_included():
    a, b = dm.ifeval_verdicts(ODD), dm.ifeval_verdicts(json.loads(json.dumps(ODD)))
    assert a == b and set(a) == {"1122", "1129"}
    for v in a.values():
        assert set(v) == {"strict", "loose"} and len(v["strict"]["inst"]) == 2


def test_the_checker_is_imported_without_fetching():
    """punkt is fetched when a sentence is first counted, not on import"""
    import importlib
    import subprocess
    import sys
    code = ("import nltk; nltk.download = lambda *a, **k: (_ for _ in ()).throw("
            "AssertionError('fetched on import'))\n"
            "import ifeval_official.utils as U; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(dm.HERE))
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stderr[-400:]
    assert importlib.util.find_spec("ifeval_official") is not None


def test_ifeval_is_the_mean_of_four():
    s = [{"ifeval": {"strict": {"prompt": True, "inst": [True, True]},
                     "loose": {"prompt": True, "inst": [True, True]}}},
         {"ifeval": {"strict": {"prompt": False, "inst": [True, False, False]},
                     "loose": {"prompt": True, "inst": [True, True, True]}}}]
    # prompt strict 1/2, inst strict 3/5, prompt loose 2/2, inst loose 5/5
    assert dm.ifeval_mean4(s) == pytest.approx((0.5 + 0.6 + 1 + 1) / 4)


# ---------------------------------------------------------------------------
# the cap, the composite, the ranks
# ---------------------------------------------------------------------------

def _mmlu(key, parsed, gold, tokens=500, capped=False):
    return {"bench": "mmlu_pro", "key": key, "parsed": parsed, "gold": gold,
            "answered": parsed is not None, "correct": parsed is not None and parsed == gold,
            "gen_tokens": tokens, "capped": capped}


def test_no_answer_within_the_cap_is_wrong_and_stays_in():
    items = {("mmlu_pro", str(i)): {"bench": "mmlu_pro", "key": str(i), "answer": "A"}
             for i in range(4)}
    recs = [{"bench": "mmlu_pro", "key": "0", "answer": "\\boxed{A}", "gen_tokens": 300},
            {"bench": "mmlu_pro", "key": "1", "answer": "\\boxed{A}", "gen_tokens": 900},
            {"bench": "mmlu_pro", "key": "2", "answer": "\\boxed{C}", "gen_tokens": 700},
            # cut off by the cap, never boxed: no answer
            {"bench": "mmlu_pro", "key": "3", "answer": "Let me think about option A",
             "gen_tokens": 4096, "capped": True}]
    scored = dm.score_items(recs, items)
    assert [s["answered"] for s in scored] == [True, True, True, False]
    assert scored[3]["correct"] is False and scored[3]["parsed"] is None
    b = dm.summarize(scored, {})["benches"]["mmlu_pro"]
    assert (b["acc"], b["acc_answered"], b["n"], b["answered"], b["capped"]) == (
        0.5, round(2 / 3, 4), 4, 3, 1)


def test_wilson_matches_devicemarks_own_interval():
    # LFM2.5-1.2B's MMLU-Pro on their board: 81 of 196, [0.3466, 0.4832]
    assert dm.wilson(81 / 196, 196) == [0.3466, 0.4832]


def _row():
    rng = random.Random(7)
    s = []
    for i in range(30):
        ok = rng.random() < 0.7
        s.append({"bench": "ifeval", "key": str(i), "answered": True, "gen_tokens": 200 + i,
                  "correct": ok, "ifeval": {"strict": {"prompt": ok, "inst": [ok, True]},
                                            "loose": {"prompt": True, "inst": [True, True]}}})
    for i in range(20):
        s.append(_mmlu(str(i), "A" if rng.random() < 0.5 else "B", "A", tokens=100 * (i + 1)))
    for i in range(10):
        ok = rng.random() < 0.6
        s.append({"bench": "math", "key": str(i), "parsed": "1", "answered": True, "correct": ok,
                  "gen_tokens": 250 * (i + 1)})
    return s


def test_the_composite_and_its_bootstrap_are_deterministic():
    s = _row()
    a, b = dm.summarize(s, {"model": "x"}), dm.summarize(list(s), {"model": "x"})
    assert a["composite"] == b["composite"]
    c = a["composite"]
    assert c["value"] == round(sum(a["benches"][x]["acc"] for x in dm.BENCHES) / 3, 4)
    assert c["ci"][0] < c["value"] < c["ci"][1]
    assert dm.composite_ci({x: [r for r in s if r["bench"] == x] for x in dm.BENCHES},
                           n=200, seed=1) != c["ci"]               # another seed, another interval


def test_accuracy_against_budget_is_from_each_answers_length():
    tf = dm.time_frontier(_row())
    assert tf["b"] == list(dm.BUDGETS)
    pooled = [r for r in _row() if r["bench"] != "ifeval"]
    at = {b: sum(r["correct"] and r["gen_tokens"] <= b for r in pooled) / len(pooled)
          for b in dm.BUDGETS}
    assert tf["acc"] == [round(at[b], 4) for b in dm.BUDGETS]
    assert tf["acc"] == sorted(tf["acc"])                              # never falls with room


# DeviceMark's board as published (board.json, 14 Sep 2026; CC-BY-4.0): each
# row's composite interval, and the rank their rule gives it
THEIRS = [
    ("gemini-flash", [0.8963, 0.9447], "=1"), ("gemini-pro", [0.8776, 0.9307], "=1"),
    ("lfm2.5-1.2b", [0.6458, 0.7215], "=3"), ("qwen3.5-2b", [0.5778, 0.666], "=3"),
    ("nemotron-4b", [0.5782, 0.6511], "=3"), ("apple-fm", [0.5155, 0.6009], "4"),
    # Qwen3.5-2B's and Nemotron's lower ends are above these three's upper ends
    ("gemma-4-e2b", [0.4881, 0.5724], "=6"), ("granite-4.0-h-1b", [0.4839, 0.566], "=6"),
    ("youtu-2b", [0.4785, 0.5655], "=6"), ("qwen3.5-0.8b", [0.3779, 0.462], "10"),
    ("nanbeige-3b", [0.2748, 0.3558], "=11"), ("qwen3.5-4b", [0.224, 0.2995], "=11"),
]


def test_the_ranks_are_their_rule_on_their_own_rows():
    got = dm.ranks([{"ci": ci} for _, ci, _ in THEIRS])
    assert got == [want for _, _, want in THEIRS]
    # a row wholly above another is ranked above it; overlapping is a tie
    assert dm.ranks([{"ci": [0.5, 0.6]}, {"ci": [0.3, 0.4]}]) == ["1", "2"]
    assert dm.ranks([{"ci": [0.5, 0.6]}, {"ci": [0.55, 0.7]}]) == ["=1", "=1"]


# ---------------------------------------------------------------------------
# MTP parity
# ---------------------------------------------------------------------------

def _parity_pair(differ: int):
    mtp, plain = [], []
    for b, n in dm.PARTS["parity"].items():
        for i in range(n):
            base = ({"bench": b, "key": str(i), "text": f"t{i}", "parsed": "A",
                     "ifeval": {"strict": {"prompt": True, "inst": [True]},
                                "loose": {"prompt": True, "inst": [True]}}})
            mtp.append(dict(base))
            plain.append(dict(base))
    for s in plain[:differ]:
        s["text"] = "another output"
        s["ifeval"] = {"strict": {"prompt": False, "inst": [False]},
                       "loose": {"prompt": False, "inst": [False]}}
    return mtp, plain


@pytest.mark.parametrize("differ,passes", [(0, True), (2, True), (3, False)])
def test_parity_48_of_50_passes_and_47_does_not(differ, passes):
    rep = dm.parity(*_parity_pair(differ))
    assert (rep["n"], rep["same_answer"], rep["identical"], rep["passes"]) == (
        50, 50 - differ, 50 - differ, passes)


def test_a_setup_without_mtp_takes_its_quality_from_the_mtp_row():
    mtp_row = {"part": "full", "composite": {"value": 0.6, "ci": [0.55, 0.65]},
               "setup": {"thinking": False}}
    rep = {**dm.parity(*_parity_pair(2)), "mtp": "served/x-MTP", "plain": "served/x",
           "thinking": False}
    got = dm.inherit(None, mtp_row, rep)
    assert got["inherited"]["line"] == "quality from MTP run, parity 48/50"
    # 47 of 50: it doesn't
    low = {**rep, **dm.parity(*_parity_pair(3))}
    assert dm.inherit(None, mtp_row, low) is None
    # its own full run — the switch that runs it anyway — is its own
    own = {"part": "full", "composite": {"value": 0.58, "ci": [0.5, 0.66]}}
    assert dm.inherit(own, mtp_row, rep) is own
    # never across thinking modes
    assert dm.inherit(None, mtp_row, {**rep, "thinking": True}) is None
