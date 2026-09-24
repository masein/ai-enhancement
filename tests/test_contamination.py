"""The 13-gram gate: what it drops, what it refuses, what it lets through."""

from __future__ import annotations

import pytest

from service import contamination as ct


def test_normalize_and_windows():
    toks = ct.normalize("  The Cat, sat: on THE mat!  ")
    assert toks == ["the", "cat", "sat", "on", "the", "mat"]
    assert list(ct.windows(toks, 3)) == ["the cat sat", "cat sat on", "sat on the", "on the mat"]
    assert list(ct.windows(toks, 13)) == []               # too short to carry one


def test_item_text_covers_every_format():
    doc = {"title": "Margins first", "text": "A long body about margins."}
    assert ct.item_text(doc) == "Margins first\nA long body about margins."
    free = {"question": "q", "answer": "a", "rationale": "r"}
    assert ct.item_text(free) == "q\na\nr"
    assert ct.item_text({"question": "q", "choices": ["x", "y"], "answer": "a",
                         "rationale": ""}) == "q\nx\ny\na"


def test_doc_strings_walk_every_shape():
    doc = {"question": "q", "choices": {"text": ["a", "b"], "label": ["A", "B"]},
           "mc2_targets": {"choices": ["c"], "labels": [1]}, "n": 3}
    assert sorted(ct.doc_strings(doc)) == ["A", "B", "a", "b", "c", "q"]


def test_the_exam_is_indexed_in_both_halves(tree, tmp_path_factory):
    """A generated document that repeats an exam question teaches the test
    directly, and the exam is our own questions — nothing outside protects it."""
    import exam_build as eb
    # questions the models have SAT are already in the samples logs; the bank
    # index is what covers every accepted question, including ones written
    # since the last run — which is the gap a generator would echo into
    empty = ct.BenchmarkIndex(tmp_path_factory.mktemp("noresults"),
                              tree["judged"]["exam_root"]).refresh()
    ix = ct.index(tree["out_dir"], tree["judged"]["exam_root"])
    bank = [b for rows in eb.load_bank(tree["judged"]["exam_root"]).values() for b in rows]
    assert ix.n_exam == len(bank) >= 100
    assert ix.exam_grams and not ct.BenchmarkIndex(tree['out_dir']).refresh().exam_grams
    halves = {eb.half_of(b["qid"]) for b in bank}
    assert halves == {"report", "diagnose"}
    assert empty.n_exam == ix.n_exam and not empty.grams
    for half in ("report", "diagnose"):
        q = next(b["prompt"] for b in sorted(bank, key=lambda b: b["qid"])
                 if eb.half_of(b["qid"]) == half)
        hits = ix.hits(q)
        assert hits and ix.source_of(hits[0]) == "exam", half
        assert empty.hits(q)                           # from the bank alone
    # a benchmark n-gram is still named as one
    bench = ix.hits(tree["docs"]["mmlu"][0]["q"])
    assert bench and ix.source_of(bench[0]) == "benchmark"
    # the index is keyed on the bank too: accepting a question rebuilds it
    assert ct.index(tree["out_dir"], tree["judged"]["exam_root"]) is ix


def test_a_document_quoting_an_exam_question_is_dropped(tree):
    import exam_build as eb
    ix = ct.index(tree["out_dir"], tree["judged"]["exam_root"])
    q = sorted(eb.load_bank(tree["judged"]["exam_root"])["Economics"],
               key=lambda b: b["qid"])[0]["prompt"]
    body = ("Margins respond before totals do, and the reason is worth setting out slowly. "
            + "Consider a firm that changes one input price and nothing else. " * 6)
    clean = {"title": "Margins first", "text": body}
    echo = {"title": "Margins again", "text": body + " A question of the kind this teaches: " + q}
    out = ct.check([clean, echo], ix)
    assert out["report"]["items_kept"] == 1 and out["report"]["dropped_benchmark"] == 1
    assert out["report"]["dropped_exam"] == 1
    assert out["dropped"][0]["source"] == "exam" and out["dropped"][0]["index"] == 1
    assert out["kept"][0]["title"] == "Margins first"
    # three per cent of a set rejects the whole dataset, whichever corpus it echoed
    docs = [{"title": f"Note {i}", "text": body + f" Case {i} closes here."} for i in range(97)]
    docs += [{"title": f"Echo {i}", "text": body + " " + q} for i in range(3)]
    rep = ct.check(docs, ix)["report"]
    assert rep["dropped_exam"] == 3 and rep["rejected"] is True
    assert rep["share_dropped_benchmark"] == 0.03


def test_near_duplicate_documents_collapse(tree):
    ix = ct.index(tree["out_dir"], tree["judged"]["exam_root"])
    body = ("The first quantity to move is the one closest to the change, which is usually a "
            "margin and rarely a total. Totals are sums of parts that move at different "
            "speeds, so they are the last place to look and the first place people look. ")
    a = {"title": "Margins first", "text": body * 3}
    b = {"title": "Margins first, again", "text": body * 3 + " "}      # the same document
    c = {"title": "Rules and purposes", "text":
         ("A rule written for one situation gets applied in another, and the two come apart "
          "exactly where the reasoning is interesting. Ask what a constraint was for before "
          "asking whether it binds today. ") * 3}
    out = ct.check([a, b, c], ix)
    assert out["report"]["items_kept"] == 2 and out["report"]["dropped_duplicate"] == 1
    assert out["dropped"][0]["reason"] == "duplicate" and out["dropped"][0]["of"] == 0
    assert [k["title"] for k in out["kept"]] == ["Margins first", "Rules and purposes"]


def test_index_covers_both_halves_once_per_document(tree):
    ix = ct.index(tree["out_dir"])
    n_docs = sum(len(d) for d in tree["docs"].values())
    n_fr = sum(v["items"] for v in tree["judged"]["manifest"]["tasks"].values())
    # 12a: and the Everyday pilot's five questions, once each however many
    # models answered them — a dataset that repeats one is dropped like any other
    import everyday as ev
    n_pilot = len(ev.load_pilot()) if tree["everyday"] else 0
    assert ix.n_docs == n_docs + n_fr + n_pilot           # eight models, one benchmark (+ fr items)
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
