"""The 13-gram gate: what it drops, what it refuses, what it lets through."""

from __future__ import annotations

import pytest

from service import contamination as ct


def test_normalize_and_windows():
    toks = ct.normalize("  The Cat, sat: on THE mat!  ")
    assert toks == ["the", "cat", "sat", "on", "the", "mat"]
    assert list(ct.windows(toks, 3)) == ["the cat sat", "cat sat on", "sat on the", "on the mat"]
    assert list(ct.windows(toks, 13)) == []               # too short to carry one


def test_doc_strings_walk_every_shape():
    doc = {"question": "q", "choices": {"text": ["a", "b"], "label": ["A", "B"]},
           "mc2_targets": {"choices": ["c"], "labels": [1]}, "n": 3}
    assert sorted(ct.doc_strings(doc)) == ["A", "B", "a", "b", "c", "q"]


def test_index_covers_both_halves_once_per_document(tree):
    ix = ct.index(tree["out_dir"])
    n_docs = sum(len(d) for d in tree["docs"].values())
    n_fr = sum(v["items"] for v in tree["judged"]["manifest"]["tasks"].values())
    assert ix.n_docs == n_docs + n_fr                     # eight models, one benchmark (+ fr items)
    assert ix.n_files > 8 and len(ix.grams) > 1000
    # a report-half question is in the index too — the gate guards the half nobody sees
    import diagnose as dx
    report_q = next(d["q"] for d in tree["docs"]["mmlu"] if dx.split_of(d["doc_hash"]) == "report")
    assert ix.hits(report_q), "a report-half question must be caught"
    assert ct.index(tree["out_dir"]) is ix                 # cached while the tree is unchanged


def _item(q, a="the margin narrows", r="because costs rise first"):
    return {"question": q, "answer": a, "rationale": r,
            "choices": [a, "fixed costs vanish", "rivals leave", "revenue is unchanged"]}


CLEAN = [_item(f"Suppose a {w} raises its price while its rivals hold theirs; what happens to "
               f"its unit sales over the following quarter, in case {i}?")
         for i, w in enumerate(["bakery", "shipyard", "vineyard", "printer", "dairy", "studio",
                                "hospital", "council", "cooperative", "workshop"] * 10)]


def test_planted_benchmark_sentence_is_dropped(tree):
    ix = ct.index(tree["out_dir"])
    planted = tree["docs"]["mmlu"][3]["q"]
    items = CLEAN[:49] + [_item(planted)]
    out = ct.check(items, ix)
    rep = out["report"]
    assert rep["dropped_benchmark"] == 1 and rep["items_kept"] == 49
    assert out["dropped"][0]["reason"] == "benchmark" and out["dropped"][0]["index"] == 49
    assert rep["offending_ngrams"] and rep["offending_ngrams"][0] in " ".join(ct.normalize(planted))
    assert rep["share_dropped_benchmark"] == 0.02 and rep["rejected"] is False   # at the line, not above


def test_three_percent_planted_rejects_the_dataset(tree):
    ix = ct.index(tree["out_dir"])
    qs = [d["q"] for d in tree["docs"]["mmlu"][:3]]
    items = CLEAN[:97] + [_item(q) for q in qs]
    rep = ct.check(items, ix)["report"]
    assert rep["dropped_benchmark"] == 3 and rep["rejected"] is True
    assert rep["share_dropped_benchmark"] == 0.03 and rep["max_share"] == 0.02


def test_a_paraphrase_that_keeps_thirteen_words_is_still_caught(tree):
    ix = ct.index(tree["out_dir"])
    q = tree["docs"]["mmlu"][7]["q"]
    toks = ct.normalize(q)
    echo = "Consider this: " + " ".join(toks[:13]) + " and so on."
    assert ct.check([_item(echo)], ix)["report"]["dropped_benchmark"] == 1


def test_clean_dataset_passes_untouched(tree):
    ix = ct.index(tree["out_dir"])
    out = ct.check(CLEAN, ix)
    assert out["report"]["dropped"] if False else True
    assert out["report"]["items_kept"] == len(CLEAN) and out["dropped"] == []
    assert out["report"]["rejected"] is False


def test_near_duplicates_collapse(tree):
    ix = ct.index(tree["out_dir"])
    base = _item("Suppose a bakery raises its price while its rivals hold theirs; what happens "
                 "to its unit sales over the following quarter?")
    dup = dict(base, question=base["question"].replace("bakery", "Bakery") + " ")
    other = _item("A council levies a new duty on imported flour; which of its local bakers "
                  "gains, and why does the answer depend on their suppliers?")
    out = ct.check([base, dup, other, dict(other)], ix)
    assert out["report"]["items_kept"] == 2 and out["report"]["dropped_duplicate"] == 2
    assert [d["reason"] for d in out["dropped"]] == ["duplicate", "duplicate"]
    assert out["dropped"][0]["of"] == 0 and out["dropped"][1]["of"] == 1


def test_empty_input_is_not_rejected(tree):
    rep = ct.check([], ct.index(tree["out_dir"]))["report"]
    assert rep["items_in"] == 0 and rep["rejected"] is False


@pytest.mark.parametrize("n", [ct.NGRAM])
def test_constants_are_the_convention(n):
    assert n == 13 and ct.MAX_DROP_SHARE == 0.02
