"""29: the Ask assistant tells NVIDIA Nemotron 3.5 Bolt E4B's two builds
measured here apart — "bolt q4" is the Q4_K_M, "Bolt Q8_0" the Q8_0, "the
measured bolt" asked back between them — while "bolt" stays NVIDIA's published
numbers, as the bank has it; both builds sit with the open models, tied to the
published entry, and its follow-ups put them side by side. 28's world with
Bolt's two builds measured here; invented numbers, no model runs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import assistant_world as aw
from conftest import make_service
from service import assistant, db, reports, telemetry
from test_21c_page import results
from test_28_assistant import BANK, fake

ROOT = Path(__file__).resolve().parent.parent
BQ4 = "served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M"
BQ8 = "served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0"
PUB = "reported/nvidia/nvidia-nemotron-3.5-bolt-e4b"
Q4 = "served/Qwen3.6-35B-A3B-Q4-lda-2610"


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("a29")
    client, appmod, _ = make_service(root, mp)
    saved = telemetry.NAMES
    aw.world()
    # Bolt's two builds, measured here thinking on, registered by their names
    for mid, name, v in ((BQ4, "NVIDIA Nemotron 3.5 Bolt E4B · Q4_K_M", 0.52),
                         (BQ8, "NVIDIA Nemotron 3.5 Bolt E4B · Q8_0", 0.55)):
        db.served_put({"id": mid, "name": name, "base_url": "", "key": "", "how": "own box",
                       "based_on": "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", "thinking": "on",
                       "runtime": "card-bolt", "pin": {}, "by": "masein", "at": 0})
        results(reports.row_dir(mid + " · thinking"), mid, "gpqa_diamond_epoch", v, 0.02, 792, 0)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield {"cat": assistant.catalog()}
    aw.unwind(saved)
    client.__exit__(None, None, None)
    mp.undo()


def opened(name: str, cat: dict) -> dict:
    return assistant.resolve({"action": "open_page", "page": "model_page", "model": name}, cat)


@pytest.mark.parametrize("name, want", [
    ("bolt q4", BQ4), ("Bolt Q4_K_M", BQ4), ("the q4 bolt", BQ4), ("bolt's q4_k_m", BQ4),
    ("nemotron bolt q4_k_m", BQ4), ("NVIDIA Nemotron 3.5 Bolt E4B · Q4_K_M", BQ4),
    ("bolt q8", BQ8), ("Bolt Q8_0", BQ8), ("q8 bolt", BQ8), ("bolt's q8_0", BQ8),
    ("NVIDIA Nemotron 3.5 Bolt E4B · Q8_0", BQ8),
    # the published numbers, as the bank has them
    ("bolt", PUB), ("Bolt", PUB), ("nemotron bolt", PUB), ("the nemotron one", PUB),
    # and our own Q4 is still ours
    ("q4", Q4), ("the q4", Q4)])
def test_each_build_has_its_own_names_and_bolt_is_still_the_published(board, name, want):
    got = opened(name, board["cat"])
    assert got == {"action": "open_page", "page": "model_page", "model": want}, (name, got)


@pytest.mark.parametrize("name", ["the measured bolt", "bolt measured here", "bolt on the v100"])
def test_the_measured_bolt_is_asked_back_between_its_builds(board, name):
    got = opened(name, board["cat"])
    assert got["action"] == "clarify" and set(got["candidates"]) == {BQ4, BQ8}, got


def test_a_comparison_of_the_two_builds_and_the_banks_bolt(board):
    cat = board["cat"]
    got = assistant.resolve({"action": "compare", "models": ["bolt q4", "bolt q8"],
                             "benchmarks": ["gpqa"]}, cat)
    assert got["models"] == [BQ4, BQ8]
    # every bank question that names Bolt resolves as it did: the published entry
    for b in [b for b in BANK if re.search(r"\bbolt\b", b["q"], re.I)]:
        a = assistant.resolve(assistant.as_named(b["want"], cat), cat, b["q"])
        assert assistant.same(a, b["want"], cat, b.get("prev")), (b["id"], a)
        got = assistant.ask(b["q"], "masein", b.get("prev"), cat=cat, model=fake(b["want"], cat))
        assert assistant.same(got["action"], b["want"], cat, b.get("prev")), (b["id"], got)


def test_the_builds_are_open_models_tied_to_the_published_entry(board):
    cat = board["cat"]
    m = {x["id"]: x for x in cat["models"]}
    assert m[BQ4]["group"] == m[BQ8]["group"] == "open" and not m[BQ4]["published"]
    assert m[PUB]["builds"] == [BQ4, BQ8] and m[PUB]["published"]
    assert (m[BQ4]["build"], m[BQ4]["siblings"], m[BQ4]["of"]) == ("Q4_K_M", [BQ8], PUB)
    assert (m[BQ8]["build"], m[BQ8]["siblings"], m[BQ8]["of"]) == ("Q8_0", [BQ4], PUB)
    # two models, never one: each its own rows
    assert m[BQ4]["rows"] == [BQ4 + " · thinking"] and m[BQ8]["rows"] == [BQ8 + " · thinking"]


def test_the_assistants_outside_models_are_the_pages():
    page = (ROOT / "scripts" / "report_lm_eval.py").read_text(encoding="utf-8")
    js = re.search(r"const MT_OUTSIDE_RE = /(.*)/i;", page).group(1)
    assert assistant.OUTSIDE_RE.pattern == js


def test_the_follow_ups_put_the_builds_side_by_side(board):
    cat = board["cat"]
    # the published Bolt beside Gemma: first, the same with its builds measured here
    a = assistant.resolve({"action": "compare", "models": ["bolt", "gemma 4 e4b"],
                           "benchmarks": ["ifbench"]}, cat)
    first = assistant.suggestions(a, cat)[0]
    assert first["label"] == "and its builds measured here"
    assert first["action"]["models"] == [PUB, BQ4, BQ8, "reported/google/gemma-4-e4b"]
    assert first["action"]["benchmarks"] == ["ifbench"]
    # one build's number: beside the other build
    got = assistant.suggestions({"action": "compare", "models": [BQ4], "benchmarks": ["gpqa"]},
                                cat)
    beside = next(s for s in got if s["label"] == "beside Q8_0")
    assert beside["action"] == {"action": "compare", "models": [BQ4, BQ8],
                                "benchmarks": ["gpqa"]}
    # both builds shown already: nothing more to add
    got = assistant.suggestions({"action": "compare", "models": [PUB, BQ4, BQ8],
                                 "benchmarks": ["gpqa"]}, cat)
    assert not any("builds measured here" in s["label"] for s in got)
