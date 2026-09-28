"""12o.5: the duplicate check's cosine for the local model, chosen on a
labelled set from the Everyday PRACTICE half — sixty questions reworded by
hand, sixty pairs of different questions from one group — as the highest
that still flags 95% of the rewordings, with both rates reported. And the
builder names the question a new one duplicates: the same words over a
cosine, the closest cosine over the first.

Here a word-count stand-in takes the model's place. Where the model is in
the image — deploy step 3 — the last two tests run it for real: the set's
threshold, and the builder catching a rewording at it."""

from __future__ import annotations

import json

import pytest

import everyday as ev
from fake_openrouter import embedding
from service import builder, config, dup_threshold, embed_local
from test_12o1_embeddings import Local
from test_builder_12i2 import _near_copy_writer, create, rest, review_all, svc  # noqa: F401

REAL = pytest.mark.skipif(not embed_local.available(),
                          reason="the embedding model isn't in this image: deploy step 3 runs this")


def rows():
    return [json.loads(x) for x in dup_threshold.PAIRS.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def test_the_labelled_set_is_the_practice_half_only():
    bank = {q["id"]: q for q in ev.load_bank()}
    same = [r for r in rows() if r["kind"] == "reworded"]
    diff = [r for r in rows() if r["kind"] == "different"]
    assert len(same) == len(diff) == 60 and len(rows()) == 120
    for r in rows():
        ids = [r["id"]] if r["kind"] == "reworded" else r["ids"]
        assert all(ev.half(bank[i]) == ev.PRACTICE for i in ids), ids
        assert all(bank[i]["group"] == r["group"] for i in ids)
    # every group, and no pair twice
    assert {r["group"] for r in same} == {r["group"] for r in diff} == set(ev.GROUPS)
    assert len({r["id"] for r in same}) == 60
    assert len({tuple(sorted(r["ids"])) for r in diff}) == 60
    assert all(r["ids"][0] != r["ids"][1] for r in diff)
    # a rewording shares no 13 words in a row with its question, so only the
    # embeddings can catch it — what the cosine is for
    for r in same:
        assert r["text"] != bank[r["id"]]["prompt"]
        assert not builder._grams(r["text"]) & builder._grams(bank[r["id"]]["prompt"]), r["id"]
    # no hidden question's words anywhere in it
    hidden = [q["prompt"] for q in bank.values() if ev.half(q) == ev.HIDDEN]
    text = dup_threshold.PAIRS.read_text(encoding="utf-8")
    assert not [h for h in hidden if h in text]


@pytest.mark.parametrize("same, different, want", [
    # 20 rewordings, 0.80 … 0.99: 95% is 19, the 19th highest is 0.81
    ([round(0.80 + i / 100, 2) for i in range(20)], [0.70, 0.805, 0.83, 0.90],
     {"cosine": 0.81, "reworded": 20, "caught": 19, "recall": 0.95, "different": 4,
      "false_flags": 2, "false_rate": 0.5}),
    # rounded down: 0.8675 → 0.86, which catches the same 19 and flags 0.861 too
    ([0.9] * 18 + [0.8675, 0.5], [0.861, 0.5],
     {"cosine": 0.86, "reworded": 20, "caught": 19, "recall": 0.95, "different": 2,
      "false_flags": 1, "false_rate": 0.5}),
    # 0.29 × 100 is 28.999… in floating point: still 0.29
    ([0.29] * 20, [], {"cosine": 0.29, "reworded": 20, "caught": 20, "recall": 1.0,
                       "different": 0, "false_flags": 0, "false_rate": None}),
    ([], [0.5], {"cosine": None}),
])
def test_the_cosine_is_the_highest_that_flags_95_percent(same, different, want):
    assert dup_threshold.choose(same, different) == want


def test_it_runs_on_the_set_and_the_builder_uses_it(svc, monkeypatch, capsys):  # noqa: F811
    local = Local(monkeypatch)
    dup_threshold.main()
    out = capsys.readouterr().out
    got = json.loads((config.BENCH_ROOT / "builder" / "dup_threshold.json").read_text())
    assert got["model"] == embed_local.IDENT and builder.dup_cosine() == got["cosine"]
    assert got["reworded"] == got["different"] == 60 and got["recall"] >= 0.95
    assert got["gone"] == 0 and got["pairs_sha256"] == dup_threshold.labelled()["sha256"]
    assert out.startswith(f"{embed_local.MODEL} at cosine {got['cosine']} flags "
                          f"{got['caught']} of 60 reworded questions")
    assert f"and {got['false_flags']} of 60 different ones" in out
    # the practice half only was embedded, and no question is printed
    practice = {q["prompt"] for q in ev.load_bank() if ev.half(q) == ev.PRACTICE}
    assert set(local.asked) - practice == {r["text"] for r in rows() if r["kind"] == "reworded"}
    assert not [q for q in ev.load_bank() if q["prompt"][:30] in out]
    # it no longer needs OpenRouter's vectors: there are none here
    assert not (config.BENCH_ROOT / "builder" / "embeddings.json").exists()


def test_a_pair_that_left_the_practice_half_is_counted_out(svc, tmp_path):  # noqa: F811
    hidden = next(q for q in ev.load_bank() if ev.half(q) == ev.HIDDEN)
    f = tmp_path / "pairs.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows()[:3] + [
        {"kind": "reworded", "group": hidden["group"], "id": hidden["id"], "text": "x"},
        {"kind": "different", "group": "writing", "ids": ["everyday-writing-03", "retired-1"]},
    ]), encoding="utf-8")
    got = dup_threshold.labelled(f)
    assert got["gone"] == 2 and len(got["same"]) == 3
    out = dup_threshold.check(embed=lambda ts: [embedding(t) for t in ts], write=False, path=f)
    assert out["gone"] == 2 and out["reworded"] == 3


def _draft(monkeypatch, others, vecs):
    """one new question against `others`, with vectors given"""
    monkeypatch.setattr(builder, "_others", lambda d: others)
    monkeypatch.setattr(builder, "can_embed", lambda: True)
    monkeypatch.setattr(builder, "_embeddings", lambda texts: vecs)
    monkeypatch.setattr(builder, "dup_cosine", lambda: 0.9)
    d = {"kind": "everyday", "spec": {}, "items": [
        {"n": 1, "auto": False, "verdict": None, "flags": [], "q": {"prompt": NEW}}]}
    builder._dedup(d)
    return d["items"][0]["flags"]


NEW = ("could you tell me what time the pharmacy on the corner of main street opens on a "
       "sunday morning please")


def test_the_same_words_win_over_a_cosine_found_first(monkeypatch):
    others = [{"src": "bank", "id": "a", "label": "a", "text": "something else entirely"},
              {"src": "bank", "id": "b", "label": "b", "text": NEW + " thanks"}]
    [f] = _draft(monkeypatch, others, [[1.0, 0.0], [0.99, 0.141], [0.5, 0.866]])
    assert f["text"] == "looks like b" and f["how"] == "13 words in a row"


def test_of_the_cosines_over_the_cut_the_closest_is_named(monkeypatch):
    others = [{"src": "bank", "id": k, "label": k, "text": t} for k, t in
              (("a", "one"), ("b", "two"), ("c", "three"))]
    [f] = _draft(monkeypatch, others, [[1.0, 0.0], [0.91, 0.415], [0.98, 0.199], [0.2, 0.98]])
    assert f["text"] == "looks like b" and f["how"] == "cosine 0.98"
    assert _draft(monkeypatch, others, [[1.0, 0.0], [0.5, 0.866], [0.6, 0.8], [0, 1.0]]) == []


@REAL
def test_the_model_in_this_image_meets_the_set():
    """deploy step 3: the real model on the labelled set"""
    out = dup_threshold.check(write=False)
    assert out["reworded"] == out["different"] == 60 and out["recall"] >= 0.95
    assert 0.5 < out["cosine"] < 0.99


@REAL
def test_the_builder_catches_a_rewording_at_that_cosine(svc, monkeypatch):  # noqa: F811
    """deploy step 3: the set's cosine written as the server's check writes it,
    then a rewording from the set — no 13 words in common — written by the
    fake writer, and flagged against its question"""
    client, _, _ = svc
    out = dup_threshold.check(write=True)
    assert builder.dup_cosine() == out["cosine"]
    bank = {q["id"]: q for q in ev.load_bank()}
    same = [r for r in rows() if r["kind"] == "reworded" and r["group"] != "summarising"]
    vecs = embed_local.embed([bank[r["id"]]["prompt"] for r in same] + [r["text"] for r in same])
    cos = [dup_threshold._cos(vecs[i], vecs[len(same) + i]) for i in range(len(same))]
    pick = same[max(range(len(same)), key=cos.__getitem__)]
    _near_copy_writer(monkeypatch, pick["text"])
    d = create(client, kind="everyday", group="quick_maths", count=20)
    d = rest(client, review_all(client, d))
    assert d["dedup_how"] == "13-gram and embeddings on this server"
    dup = next(f for it in d["items"] if it["n"] == 11 for f in it["flags"] if f["kind"] == "dup")
    assert dup["how"].startswith("cosine ") and float(dup["how"].split()[1]) >= out["cosine"]
    assert dup["text"] == f"looks like {pick['id']}"
