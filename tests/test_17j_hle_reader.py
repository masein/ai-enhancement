"""17j, part 1: the HLE reader, by its cases (docs/prompts/phase-17j-eighth-
review-hle-cases-runs-page.md). tests/fixtures/hle_reader_cases.json is the
reviewers' file: 134 replies an HLE judge could send, each with what a careful
person reads ('no grade': the reply must not be graded). Every case reads as
wanted, and the file stays in the suite. Invented text only: no benchmark
question is in it."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import frontier_graders as fg

CASES = json.loads((Path(__file__).resolve().parent / "fixtures" / "hle_reader_cases.json")
                   .read_text(encoding="utf-8"))
READ = {True: "yes", False: "no", None: "no grade"}


@pytest.mark.parametrize("case", CASES["cases"], ids=[c["id"] for c in CASES["cases"]])
def test_every_case_reads_as_a_careful_person_reads_it(case):
    got = READ[fg.read("hle", case["reply"], {"id": "h1", "answer": "42"})["ok"]]
    assert got in case["want"], f"{case['id']} ({case['what']}): read {got!r}"


def test_the_cases_0abb757_got_wrong_are_the_files_own():
    """the file's two lists are its cases that 0abb757 read wrong — each is a
    case here, and reads as wanted now (the test above)"""
    ids = {c["id"] for c in CASES["cases"]}
    # the reviewers' 134, and 18b's eight (points 20 and 26) beside them
    assert len([c for c in CASES["cases"] if not c.get("added_in")]) == 134
    assert len([c for c in CASES["cases"] if c.get("added_in") == "18b"]) == 8
    for name in ("wrong_grade_on_0abb757", "unread_on_0abb757"):
        assert set(CASES[name]) <= ids, name
    assert len(CASES["wrong_grade_on_0abb757"]) == 9 and len(CASES["unread_on_0abb757"]) == 14


def test_a_strict_json_reply_is_read_right_whatever_its_strings_hold():
    """the schema's own replies: whatever the answer and the reasoning hold —
    braces, quotes, backslashes, newlines, another object's text — the
    verdict is the object's own"""
    rnd = random.Random(17)
    bits = ["{", "}", '"', "'", "\\", "\n", "correct: yes", '{"correct": "yes"}',
            "\\left\\{", "x | x > 0", "[response]", "...", "<why>", "None", "yes", "no", "42"]
    for _ in range(2000):
        words = lambda: "".join(rnd.choice(bits) + rnd.choice(["", " ", "a"])  # noqa: E731
                                for _ in range(rnd.randint(0, 6)))
        verdict = rnd.choice(["yes", "no"])
        reply = json.dumps({"extracted_final_answer": words() or "x", "reasoning": words() or "y",
                            "correct": verdict, "confidence": rnd.randint(0, 99)},
                           indent=rnd.choice([None, 2]))
        assert READ[fg.read("hle", reply, {"id": "h1", "answer": "42"})["ok"]] == verdict, reply


def test_7_a_whole_reply_that_ends_at_the_cap_is_read(tmp_path):
    """point 7: a reply that reached its cap is read all the same — a whole
    object that ended exactly at the cap was a paid try; one the cap cut off
    is no grade, and says the cap"""
    from service import frontier as sf
    from service import frontier_grade as fgr
    from service import llm
    d = tmp_path / "hle"
    d.mkdir()
    (d / sf.ANSWERS).write_text(json.dumps({"id": "h1", "epoch": 0, "answer": "41"}) + "\n" +
                                json.dumps({"id": "h2", "epoch": 0, "answer": "42"}) + "\n")
    meta = {"answers": {"h1#0": sf.answer_sha("41"), "h2#0": sf.answer_sha("42")},
            "max_tokens": 4096, "by": "masein", "prompt_sha256": fg.prompt_sha("hle")}
    whole = json.dumps({"extracted_final_answer": "41", "reasoning": "differs", "correct": "no",
                        "confidence": 90})
    g: dict = {}
    fgr._apply(g, d, "hle", {"id": "openai/o3-mini", "version": "o3-mini-2025-01-31"}, meta,
               {"h1": {"id": "h1", "answer": "42"}, "h2": {"id": "h2", "answer": "42"}},
               {"frgr:h1#0": llm.Result(text=whole, finish="length"),
                "frgr:h2#0": llm.Result(text=whole[:40], finish="length")})
    assert g["items"]["h1#0"]["ok"] is False                     # 0abb757: a paid try
    assert g["refused"]["h2#0"]["words"] == "the reply was cut at its cap of 4096 tokens"
