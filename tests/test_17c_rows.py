"""17c, part 5 of the second review round — the lookahead and MTP rows, which
this run doesn't fill. Each test fails on 0a81fc2:
- 31: a correct MTP bundle on the board's own MTP record, its flags in both
  "Launch flags" and "How it's served", imports (each counted once);
- 32: speculative decoding under its other spellings — -hfd, --hf-repo-draft,
  LLAMA_ARG_* in the environment — and what the box's slots say, never import
  onto a record without it;
- 33: the registered setup from its own fields first: a quoted value reads,
  lookahead said only in words is refused in words, and a sentence's full stop
  after "…LOOKAHEAD=1." doesn't refuse a correct box.
Nothing runs."""

from __future__ import annotations

import import_frontier as imf

PLAIN = {"id": "served/plain", "flags": "-ctk q8_0", "env": "", "how": "llama-server"}
BOX = {"flags": ["-ctk", "q8_0", "--flash-attn", "on"], "env": {}}


def test_31_an_mtp_record_with_its_flags_said_twice_takes_its_own_box():
    rec = {"id": "served/mtp", "flags": "--spec-type draft-mtp --spec-draft-n-max 3", "env": "",
           "how": "llama-server --spec-type draft-mtp --spec-draft-n-max 3", "speculative": True}
    box = {"flags": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], "env": {}}
    assert imf.launch_differs(rec, box) == []
    assert imf.record_launch(rec)["spec"] == ["--spec-draft-n-max=3", "--spec-type=draft-mtp"]


def test_32_speculative_decoding_under_any_spelling_is_seen():
    for box in ({**BOX, "flags": ["-hfd", "org/draft-GGUF"]},
                {**BOX, "flags": ["--hf-repo-draft", "org/draft-GGUF"]},
                {**BOX, "env": {"LLAMA_ARG_HF_REPO_DRAFT": "org/draft-GGUF"}},
                {**BOX, "env": {"LLAMA_ARG_DRAFT_MAX": "8"}}):
        got = imf.launch_differs(PLAIN, box)
        assert got and got[0].startswith("speculative decoding: the box ran with "), box
    # what its slots say, though no flag said it
    got = imf.launch_differs(PLAIN, {**BOX, "speculative": True})
    assert got == ["speculative decoding: the box's slots said they draft tokens; served/plain "
                   "is registered without it"]
    assert imf.launch_differs(PLAIN, {**BOX, "speculative": False}) == []


def test_33_the_registered_setup_from_its_fields_quoted_values_and_words():
    lookahead = {"LLAMA_MOE_ROUTE_MODE": "lookahead", "LLAMA_MOE_ROUTE_LOOKAHEAD": "1"}
    quoted = {**PLAIN, "env": 'LLAMA_MOE_ROUTE_MODE="lookahead" LLAMA_MOE_ROUTE_LOOKAHEAD=1'}
    assert imf.record_launch(quoted)["env"] == lookahead
    assert imf.launch_differs(quoted, BOX)[0].startswith("routing: the box ran with none")
    assert imf.launch_differs(quoted, {**BOX, "env": lookahead}) == []
    # lookahead in words only: nothing to compare with, said
    words = {**PLAIN, "how": "llama-server, routing with lookahead"}
    assert "says lookahead in words" in imf.launch_differs(words, BOX)[0]
    assert imf.launch_differs({**PLAIN, "how": "routing local (no lookahead)"}, BOX) == []
    # the field, not a sentence's full stop in "How it's served"
    told = {**PLAIN, "env": "LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1",
            "how": "lookahead routing, started with LLAMA_MOE_ROUTE_LOOKAHEAD=1."}
    assert imf.launch_differs(told, {**BOX, "env": lookahead}) == []
    # "How it's served" still read when the fields say nothing
    assert imf.record_launch({**PLAIN, "how": "with LLAMA_MOE_ROUTE_MODE=lookahead."})["env"] \
        == {"LLAMA_MOE_ROUTE_MODE": "lookahead"}
