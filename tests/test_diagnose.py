"""scripts/diagnose.py: the split, the buckets, the guards — and the rule."""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

import diagnose as dx
import make_fixture


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

def test_split_is_stable_and_salted():
    assert dx.SPLIT_SALT == "evalboard-split-v1"
    # Literal expectations on purpose. Changing the salt or the hashing
    # re-splits every benchmark and invalidates every score already published
    # against the old halves — this is where that change is meant to fail.
    assert dx.split_of("0" * 64) == "diagnose"
    assert dx.split_of("a3f1") == "report"
    assert dx.split_of("deadbeef") == "report"
    assert dx.split_of(hashlib.sha256(b"fixture").hexdigest()) == "diagnose"


def test_split_is_about_half():
    hs = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(20000)]
    share = sum(dx.split_of(h) == "diagnose" for h in hs) / len(hs)
    assert 0.48 < share < 0.52


def test_split_is_model_independent(tree, diag):
    """Every model answered the same documents, so the leaderboard half is the
    same set of items for all of them — that is what keeps scores comparable."""
    for task in make_fixture.FULL:
        n_rep = {d["tasks"][task]["n_report"] for d in diag.values() if task in d["tasks"]}
        assert len(n_rep) == 1, task
    for task, docs in tree["docs"].items():
        halves = collections.Counter(dx.split_of(d["doc_hash"]) for d in docs)
        assert halves["report"] and halves["diagnose"], task
        assert 0.35 <= halves["diagnose"] / len(docs) <= 0.65, task
        for mid, d in diag.items():
            if task in d["tasks"]:
                assert d["tasks"][task]["n_report"] == halves["report"], (mid, task)
                assert d["tasks"][task]["n_diagnose"] == halves["diagnose"], (mid, task)


# ---------------------------------------------------------------------------
# buckets and the arithmetic under them
# ---------------------------------------------------------------------------

def test_bucket_every_branch():
    assert dx.bucket([0.9, 0.1], 0, True) == "right"
    assert dx.bucket([1.0], 0, False) == "wrong"                          # one option
    assert dx.bucket([0.27, 0.25, 0.24, 0.24], 1, False) == "at_chance"   # lift 1.08
    assert dx.bucket([0.40, 0.35, 0.15, 0.10], 1, False) == "near_miss"   # 2nd, gap .05
    assert dx.bucket([0.60, 0.13, 0.14, 0.13], 1, False) == "confident_wrong"
    assert dx.bucket([0.40, 0.28, 0.16, 0.16], 1, False) == "wrong"       # 2nd, gap .12
    # confident, but the correct option is AT chance: not the actionable kind
    assert dx.bucket([0.55, 0.25, 0.10, 0.10], 1, False) == "wrong"
    # no known correct index: nothing finer than wrong can be claimed
    assert dx.bucket([0.60, 0.20, 0.10, 0.10], None, False) == "wrong"
    assert dx.bucket([0.60, 0.20, 0.10, 0.10], 7, False) == "wrong"
    # two options: 'confident' starts at lift 1.6, and a near miss is
    # unreachable (a gap under 0.10 means a top under 0.55, which is at chance)
    assert dx.bucket([0.80, 0.20], 1, False) == "confident_wrong"
    assert dx.bucket([0.75, 0.25], 1, False) == "wrong"
    assert dx.bucket([0.54, 0.46], 1, False) == "at_chance"


def test_conf_lift_scales_with_option_count():
    assert dx.conf_lift(2) == pytest.approx(1.6)
    assert dx.conf_lift(3) == pytest.approx(2.0)
    assert dx.conf_lift(4) == 2.0 and dx.conf_lift(10) == 2.0


def test_softmax_and_float_parsing_survive_the_harness_strings():
    assert dx._f("-1.5") == -1.5 and dx._f("-inf") == float("-inf")
    assert dx._f("nope") == float("-inf") and dx._f(None) == float("-inf")
    assert dx.softmax([-1.0, float("-inf")]) == [1.0, 0.0]
    assert dx.softmax([float("-inf")] * 3) == [pytest.approx(1 / 3)] * 3
    assert sum(dx.softmax([-1.2, -0.3, -4.0])) == pytest.approx(1.0)


def test_target_index_reads_every_form_the_tasks_use():
    assert dx.target_index({"target": 2}, 4) == 2
    assert dx.target_index({"target": "2"}, 4) == 2
    assert dx.target_index({"target": " B "}, 4) == 1
    assert dx.target_index({"target": True}, 4) is None
    assert dx.target_index({"target": 9}, 4) is None
    assert dx.target_index({"target": "E"}, 4) is None
    assert dx.target_index({"target": " the rest.", "doc": {"answer": 1}}, 2) == 1
    assert dx.target_index({"target": " the rest.", "doc": {"answer": "1"}}, 2) is None


def test_norm_lengths_reads_every_shape_the_tasks_use():
    assert dx.norm_lengths({"choices": ["a", "bcd"]}, 2) == [1, 3]
    assert dx.norm_lengths({"endings": ["", "é"]}, 2) == [1, 2]
    assert dx.norm_lengths({"choices": {"text": ["ab", "c"], "label": ["A", "B"]}}, 2) == [2, 1]
    assert dx.norm_lengths({"choices": ["a", "b", "c"]}, 2) is None
    assert dx.norm_lengths({"sol1": "a", "sol2": "b"}, 2) is None


def test_primary_metric_prefers_acc_norm():
    assert dx.primary_metric({"acc": 0.0, "acc_norm": 1.0}) == ("acc_norm", 1.0)
    assert dx.primary_metric({"acc": 1.0}) == ("acc", 1.0)
    assert dx.primary_metric({"exact_match": 0}) == ("exact_match", 0.0)
    assert dx.primary_metric({"acc": "1"}) is None


def _rec(correct: int, pick: int, n: int = 4, top: float = 0.6, doc: dict | None = None) -> dict:
    probs = [(1 - top) / (n - 1)] * n
    probs[pick] = top
    doc = doc or {"question": f"q{correct}{pick}", "choices": ["a", "bb", "ccc", "dddd"][:n],
                  "subject": "s"}
    return {"doc": doc, "target": correct, "doc_hash": hashlib.sha256(
        json.dumps(doc, sort_keys=True).encode()).hexdigest(),
        "filtered_resps": [[f"{p:.4f}", str(i == pick)] for i, p in enumerate(
            [__import__('math').log(x) for x in probs])],
        "acc": float(pick == correct), "metrics": ["acc"], "filter": "none"}


def _write(tmp_path: Path, recs: list[dict], name="samples_t_2026-01-01T00-00-00.000000.jsonl"):
    f = tmp_path / name
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return f


def test_tvd_and_ceiling_on_a_known_distribution(tmp_path):
    """Gold uniform over four slots, every pick on A or B: TVD is 0.5 and the
    most such a model could score is 0.5. The page shows 1 - TVD as the ceiling;
    both numbers are computed here from the same picks and gold."""
    recs = []
    for i in range(200):
        d = {"question": f"q{i}", "choices": ["a", "bb", "ccc", "dddd"], "subject": "s"}
        recs.append(_rec(i % 4, i % 2, doc=d))
    out = dx.diagnose_task([_write(tmp_path, recs)])
    a = out["answers"]
    assert a["picks"] == {"0": 100, "1": 100} and a["gold"] == {str(i): 50 for i in range(4)}
    assert a["pick_skew"] == pytest.approx(0.5) and a["position_biased"] is True
    picks = [a["picks"].get(str(i), 0) / 200 for i in range(4)]
    gold = [a["gold"][str(i)] / 200 for i in range(4)]
    ceiling = sum(min(p, g) for p, g in zip(picks, gold))
    assert ceiling == pytest.approx(1 - a["pick_skew"])
    assert out["score_all"] == pytest.approx(0.5)             # it hit the ceiling exactly
    assert a["degenerate"] is False and a["top_share"] == 0.5


def test_tvd_and_ceiling_agree_on_every_fixture_task(diag):
    checked = 0
    for mid, d in diag.items():
        for task, v in d["tasks"].items():
            a = v.get("answers") or {}
            if "pick_skew" not in a:
                continue
            n = a["n_options"]
            tp = sum(a["picks"].values())
            tg = sum(a["gold"].values())
            picks = [a["picks"].get(str(i), 0) / tp for i in range(n)]
            gold = [a["gold"].get(str(i), 0) / tg for i in range(n)]
            tvd = 0.5 * sum(abs(p - g) for p, g in zip(picks, gold))
            assert a["pick_skew"] == pytest.approx(tvd, abs=1e-4), (mid, task)
            assert sum(min(p, g) for p, g in zip(picks, gold)) == pytest.approx(1 - tvd)
            assert a["position_biased"] == (tvd >= dx.POSITION_SKEW), (mid, task)
            assert a["degenerate"] == (a["top_share"] >= dx.DEGENERATE_SHARE), (mid, task)
            checked += 1
    assert checked >= 20


# ---------------------------------------------------------------------------
# the guards: tasks the analysis does not fit
# ---------------------------------------------------------------------------

def test_multi_true_task_reports_no_statistics(diag):
    for mid, d in diag.items():
        v = d["tasks"]["truthfulqa_mc2"]
        assert "multi-true" in v["answers"]["unsupported"], mid
        assert set(v["answers"]) == {"unsupported"}
        assert set(v["buckets"]) <= {"right", "wrong"}
        assert v["examples"] == {}
        assert v["score_report"] is not None            # the score itself stands


def test_task_without_a_single_answer_index_is_unsupported(diag):
    for mid, d in diag.items():
        v = d["tasks"]["winogrande"]
        assert v["answers"]["unsupported"].startswith("only 0% of items expose"), mid
        assert v["examples"] == {} and set(v["buckets"]) <= {"right", "wrong"}


def test_varying_option_count_is_unsupported(tmp_path):
    recs = [_rec(0, 1, n=4) for _ in range(10)] + [_rec(0, 1, n=2) for _ in range(10)]
    out = dx.diagnose_task([_write(tmp_path, recs)])
    assert "number of options varies" in out["answers"]["unsupported"]


def test_approximate_buckets_when_lengths_are_hidden(diag):
    # piqa is scored acc_norm but its doc has sol1/sol2, not a choices list
    for d in diag.values():
        assert d["tasks"]["piqa"]["approx_buckets"] is True
        assert d["tasks"]["hellaswag"]["approx_buckets"] is False


# ---------------------------------------------------------------------------
# the findings land where they were planted, and nowhere else
# ---------------------------------------------------------------------------

def _flags(v: dict) -> set[str]:
    a = v.get("answers") or {}
    nb = v["n"] or 1
    out = {k for k in ("degenerate", "position_biased", "length_biased") if a.get(k)}
    if v["buckets"].get("confident_wrong", 0) / nb >= dx.CONFIDENT_SHARE:
        out.add("confident_wrong")
    return out


def test_findings_land_on_the_model_they_were_planted_in(diag):
    mm = {mid: d["tasks"]["mmlu"] for mid, d in diag.items()}
    a = mm["fx/one-option-70m"]["answers"]
    assert a["degenerate"] and a["top_choice"] == 0 and a["top_share"] >= 0.8
    a = mm["fx/skewed-360m"]["answers"]
    assert a["position_biased"] and not a["degenerate"] and not a["length_biased"]
    assert a["pick_skew"] >= dx.POSITION_SKEW
    a = mm["fx/short-pick-410m"]["answers"]
    assert a["length_biased"] and a["short_pick_rate"] >= 1.6 * a["short_pick_baseline"]
    assert not a["position_biased"]
    assert "confident_wrong" in _flags(mm["fx/below-135m-it"])
    assert mm["fx/below-135m-it"]["score_report"] < 0.25
    for clean in ("fx/chance-160m", "fx/good-750m", "fx/miscount-1b"):
        assert _flags(mm[clean]) == set(), clean
    chance = mm["fx/chance-160m"]
    assert chance["buckets"]["at_chance"] / chance["n"] >= 0.40
    good = mm["fx/good-750m"]
    assert good["score_report"] > 0.5 and good["buckets"]["near_miss"] > 0
    assert good["buckets"].get("at_chance", 0) / good["n"] < 0.40
    weakest = min(good["groups"], key=lambda g: good["groups"][g]["score_report"])
    assert weakest == "econometrics"
    assert set(good["groups"]) == set(make_fixture.MMLU_SUBJECTS)
    for g in good["groups"].values():
        assert set(g) == {"n", "n_report", "score_report", "buckets"}


def test_item_count_disagreement_shows_in_the_log_count(diag, tree):
    mc = tree["miscount"]
    assert diag[mc["model"]]["tasks"][mc["task"]]["n"] == mc["logged"] != mc["declared"]


def test_stale_samples_file_is_ignored(diag, tree):
    st = tree["stale"]
    assert diag[st["model"]]["tasks"][st["task"]]["n"] == st["n"]
    files = [Path("samples_hellaswag_2026-08-18T09-00-00.000000.jsonl"),
             Path("samples_hellaswag_2026-08-19T10-02-11.000000.jsonl"),
             Path("samples_mmlu_anatomy_2026-08-19T10-02-11.000000.jsonl"),
             Path("odd.jsonl")]
    keep = dx.newest_per_subtask(files)
    assert files[0] not in keep and files[1] in keep and files[2] in keep and files[3] in keep


def test_spread_examples_is_round_robin_across_groups():
    pools = {("wrong", "a"): [f"a{i}" for i in range(5)],
             ("wrong", "b"): [f"b{i}" for i in range(5)],
             ("wrong", "c"): [f"c{i}" for i in range(5)],
             ("near_miss", "a"): ["n0"],
             ("at_chance", "z"): []}
    out = dx.spread_examples(pools)
    assert out["wrong"] == ["a0", "b0", "c0", "a1", "b1", "c1", "a2", "b2"]
    assert out["near_miss"] == ["n0"] and "at_chance" not in out
    assert dx.spread_examples({("wrong", "a"): ["x", "y"], ("wrong", "b"): ["p"]})["wrong"] == [
        "x", "p", "y"]
    assert dx.spread_examples({}) == {}


# ---------------------------------------------------------------------------
# THE RULE: no report-half item is ever shown
# ---------------------------------------------------------------------------

def test_no_report_half_item_ever_appears_in_examples(tree, diag):
    """Every example in every diagnose.json maps back to a document whose hash
    splits to `diagnose`. The mapping is by question text, which the fixture
    keeps unique per task so the lookup cannot be ambiguous."""
    lookup: dict[str, dict[str, str]] = {}
    for task, docs in tree["docs"].items():
        # winogrande has no question text the diagnosis could print (and no
        # examples either — it is unsupported); every other task must have a
        # distinct question per item or the lookup below could be ambiguous
        with_q = [d for d in docs if d["q"]]
        lookup[task] = {d["q"]: d["doc_hash"] for d in with_q}
        assert len(lookup[task]) == len(with_q), f"{task}: questions are not unique"
        assert with_q or task == "winogrande", task
    seen = 0
    for mid, d in diag.items():
        for task, v in d["tasks"].items():
            for bucket, items in v["examples"].items():
                assert bucket != "right"
                for e in items:
                    dh = lookup[task].get(e["q"])
                    assert dh is not None, f"{mid}/{task}: example not a known item: {e['q']!r}"
                    assert dx.split_of(dh) == "diagnose", f"{mid}/{task}: REPORT item shown"
                    seen += 1
    assert seen > 100          # not vacuous: the fixture has plenty of failures to show


def test_diagnose_json_carries_no_item_hashes(diag):
    """The file is served to browsers wholesale. It must describe items, never
    identify them — a doc_hash in it would let a reader rebuild the split."""
    hex64 = re.compile(r"\b[0-9a-f]{64}\b")
    for mid, d in diag.items():
        assert not hex64.search(json.dumps(d)), mid


def test_example_questions_are_the_diagnose_half_and_wrong(tree, diag):
    # the examples are a cross-section of the diagnosis half's failures: every
    # one is a wrong answer, and the correct answer shown is the key's, not the pick
    docs = {d["q"]: d for d in tree["docs"]["mmlu"]}
    v = diag["fx/good-750m"]["tasks"]["mmlu"]
    groups = collections.Counter()
    for items in v["examples"].values():
        for e in items:
            assert e["chose"] != e["answer"]
            assert docs[e["q"]]["group"] == e["group"]
            groups[e["group"]] += 1
    assert len(groups) >= 3, "examples should be spread across subjects, not one"


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------

def test_cli_writes_every_model_and_prints_the_findings(tree, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["diagnose.py", str(tree["out_dir"]), "--out", str(tmp_path)])
    assert dx.main() == 0
    out = capsys.readouterr().out
    assert "FINDINGS" in out
    for label, who in (("answers one option", "fx__one-option-70m/mmlu"),
                       ("answer positions skewed", "fx__skewed-360m/mmlu"),
                       ("picks by option length", "fx__short-pick-410m/mmlu"),
                       ("confidently wrong on most items", "fx__below-135m-it/mmlu")):
        assert label in out and who in out, label
    assert "fx__good-750m/mmlu" not in out.split("FINDINGS")[1]
    for mid, m in tree["models"].items():
        assert (tmp_path / m["safe"] / "diagnose.json").exists(), mid
    # -m narrows to one model; a bad tree is a usage error
    monkeypatch.setattr(sys, "argv", ["diagnose.py", str(tree["out_dir"]), "-m", "fx/good-750m",
                                      "--out", str(tmp_path / "one"), "-q"])
    assert dx.main() == 0
    assert [p.name for p in (tmp_path / "one").iterdir()] == ["fx__good-750m"]
    monkeypatch.setattr(sys, "argv", ["diagnose.py", str(tmp_path / "nowhere")])
    assert dx.main() == 2
