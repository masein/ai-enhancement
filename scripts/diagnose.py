#!/usr/bin/env python3
"""Per-item diagnosis from lm-evaluation-harness --log_samples output.

    python scripts/diagnose.py results/full            # all models
    python scripts/diagnose.py results/full -m local/my-ckpt

Writes results/full/<model>/diagnose.json, which the dashboard reads. Nothing
here re-runs an evaluation: --log_samples is already on, so every per-item
outcome is on disk and this is a read of files we already produce.

WHY THIS EXISTS, AND THE ONE RULE IT ENFORCES
---------------------------------------------
The point of per-item diagnosis is to decide what to train next. That creates a
direct route to training on the test set: look at the items a model failed,
generate data for those items, retrain, score higher, learn nothing. The score
moves and the model does not.

So every item is assigned, by a hash of its content, to one of two halves:

  report    the half the leaderboard number comes from. Never shown, never
            exported, never visible to anything that generates training data.
  diagnose  the half this tool explains, and the only half any future generator
            may see.

The assignment is by `doc_hash`, which the harness already writes and which is
identical for every model and every run, so the split is stable without any
bookkeeping and costs nothing to apply retroactively.

What that buys: a model retrained on data derived from its `diagnose` failures
still has an uncontaminated `report` score. If the training taught a skill,
`report` moves with `diagnose`. If it taught the test, `diagnose` climbs and
`report` sits still — and that divergence is itself the alarm.

Halving the item count raises standard error by √2 (on MMLU, ~0.4 → ~0.6
points). That is the price of a number that still means something after we
start optimising against it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))    # scripts/, for categories.py
import categories as _categories  # noqa: E402

# Changing this re-splits every benchmark, which invalidates comparisons against
# every score already published. It is a constant, not a setting.
SPLIT_SALT = "evalboard-split-v1"

# Bucket thresholds, anchored on CHANCE rather than on absolute probability.
#
# The anchor matters. Measured across this board, median top-probability runs
# from 0.30 (SmolLM2-135M) to 0.65 (pythia-160m), so a fixed cutoff like "p>0.5
# is confident" would call one model confident on most items and another almost
# never — describing the models relative to each other instead of describing
# each model's own behaviour, which is the opposite of what a diagnostic wants.
#
# So everything is expressed as LIFT over chance: lift = p * n_options, where
# 1.0 is exactly uniform and n is certainty. That is model-independent, and it
# works for 2-option tasks (Winogrande, PIQA) and 4-option ones alike.
CHANCE_LIFT   = 1.30     # top option barely above uniform ⇒ no information
NEAR_MISS_GAP = 0.10     # correct ranked 2nd, within this much probability
MAX_EXAMPLES  = 8        # per bucket per task, diagnose half only
# A group or category with fewer leaderboard-half items than this is noise: 13
# items carry about ±13 points, and a ranking of such groups ranks the dice.
# Written into diagnose.json so the page and this file agree on the floor.
MIN_GROUP_N   = 30

# A model that answers the same letter for nearly every question scores about
# chance and looks exactly like an ignorant model on the leaderboard. It is not
# the same thing: the output distribution is broken, and no amount of training
# data for the "weak subject" will fix it. Two checkpoints on this board do this
# today (ptop50 = 1.000, pcorr50 = 0.000), which is why it earns a flag.
DEGENERATE_SHARE = 0.80  # one option chosen for this fraction of items
# Total variation distance between what the model picks and what the answer key
# contains. 0 is a model whose answers are distributed like the truth; SmolLM2-360M
# measures 0.44 on MMLU, putting 92% of its picks on the first two of four options.
POSITION_SKEW = 0.20
# a model whose wrong answers are mostly *confident* wrong answers, which is a
# different thing from not knowing and rarely wants the same response
CONFIDENT_SHARE = 0.35


def conf_lift(n: int) -> float:
    """Where 'confident' starts, per option count. Certainty is lift n, so a
    fixed 2.0 is unreachable on a 2-option task; scale it down there."""
    return min(2.0, 1.0 + (n - 1) * 0.6)


def split_of(doc_hash: str) -> str:
    """'report' or 'diagnose' — deterministic, model-independent, stable."""
    h = hashlib.sha256((SPLIT_SALT + doc_hash).encode()).hexdigest()
    return "diagnose" if int(h[:8], 16) & 1 else "report"


def _f(x) -> float:
    """The harness writes logprobs as strings; some are 'inf'/'-inf'."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("-inf")


def softmax(xs: list[float]) -> list[float]:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return [1.0 / len(xs)] * len(xs)
    m = max(finite)
    e = [math.exp(x - m) if math.isfinite(x) else 0.0 for x in xs]
    s = sum(e) or 1.0
    return [v / s for v in e]


def bucket(probs: list[float], correct: int, right: bool) -> str:
    """Four ways to be wrong, which are not the same thing and do not have the
    same fix.

      at_chance        no information either way — more data might help
      confident_wrong  actively prefers a wrong answer AND rates the right one
                       below chance. Something false was learned, or the prompt
                       format is fighting the model; more of the same data will
                       not fix either
      near_miss        the right answer is second and close — a nudge may work
      wrong            wrong without a clean story

    Note what confident_wrong requires: not just a confident pick, but the
    correct option scoring *below* uniform. A model can be confident and wrong
    by accident; being confident while actively disfavouring the truth is the
    signal worth acting on. gemma-3-270m-it does exactly this on the median MMLU
    item (p_correct 0.170 against a 0.250 baseline) while its base twin sits at
    chance — which is the chat-template pathology, not a knowledge gap.
    """
    if right:
        return "right"
    n = len(probs)
    if n < 2:
        return "wrong"
    top = max(probs)
    lift_top = top * n
    if lift_top < CHANCE_LIFT:
        return "at_chance"
    if correct is not None and 0 <= correct < n:
        order = sorted(range(n), key=lambda i: -probs[i])
        if order.index(correct) == 1 and (top - probs[correct]) <= NEAR_MISS_GAP:
            return "near_miss"
        if lift_top >= conf_lift(n) and probs[correct] * n < 1.0:
            return "confident_wrong"
    return "wrong"


def norm_lengths(doc: dict, n: int) -> list[float] | None:
    """Byte lengths of the choices, for acc_norm tasks. Returns None when the
    doc does not expose choices in a usable shape — in which case the buckets
    are computed on raw logprobs and the caller says so."""
    for key in ("choices", "endings", "options"):
        v = doc.get(key)
        if isinstance(v, list) and len(v) == n and all(isinstance(s, str) for s in v):
            return [max(1, len(s.encode("utf-8"))) for s in v]
    if isinstance(doc.get("choices"), dict):          # arc: {text: [...], label: [...]}
        v = doc["choices"].get("text")
        if isinstance(v, list) and len(v) == n:
            return [max(1, len(str(s).encode("utf-8"))) for s in v]
    return None


def primary_metric(rec: dict) -> tuple[str, float] | None:
    """acc_norm before acc, matching the dashboard's own precedence, so the
    diagnosis grades the same answer the leaderboard reports."""
    for name in ("acc_norm", "acc", "exact_match", "em", "f1"):
        if name in rec and isinstance(rec[name], (int, float)):
            return name, float(rec[name])
    return None


def target_index(rec: dict, n: int) -> int | None:
    t = rec.get("target")
    if isinstance(t, bool):
        return None
    if isinstance(t, int) and 0 <= t < n:
        return t
    if isinstance(t, str):
        if re.fullmatch(r"\d+", t.strip()) and 0 <= int(t) < n:
            return int(t)
        # letter targets ("A"/"B"…), used by some task configs
        s = t.strip().upper()
        if len(s) == 1 and "A" <= s <= "Z" and (ord(s) - 65) < n:
            return ord(s) - 65
    d = rec.get("doc") or {}
    for key in ("answer", "label", "gold"):
        v = d.get(key)
        if isinstance(v, int) and 0 <= v < n:
            return v
    return None


def newest_per_subtask(files: list[Path]) -> list[Path]:
    """One file per subtask: a re-run leaves the older timestamped file in place
    and counting both would double every item in that subtask."""
    keep: dict[str, Path] = {}
    for f in files:
        m = re.match(r"samples_(.+?)_\d{4}-\d{2}-\d{2}T", f.name)
        key = m.group(1) if m else f.name
        if key not in keep or f.name > keep[key].name:     # timestamp sorts
            keep[key] = f
    return sorted(keep.values())


def spread_examples(by_bucket_group: dict) -> dict:
    """MAX_EXAMPLES per bucket, taken round-robin across groups.

    Collecting in file order gives every example the same subject — whichever
    one sorts first — and a list of eight abstract_algebra questions reads as a
    finding about abstract_algebra when it is nothing of the kind. Rotating
    across groups makes the sample say what it is: a cross-section.
    """
    pools: dict[str, list[list]] = {}
    for (b, _group), items in sorted(by_bucket_group.items()):
        if items:
            pools.setdefault(b, []).append(list(items))
    out: dict[str, list] = {}
    for b, groups in pools.items():
        picked, i = [], 0
        while len(picked) < MAX_EXAMPLES and any(groups):
            g = groups[i % len(groups)]
            if g:
                picked.append(g.pop(0))
            i += 1
            if i > MAX_EXAMPLES * (len(groups) + 1):     # belt and braces
                break
        if picked:
            out[b] = picked
    return out


def rollup_categories(groups: dict) -> tuple[dict, list[str]]:
    """Roll the per-group tallies (MMLU's 57 subjects) up into the categories a
    person thinks in, from scripts/categories.yaml. Same fields as `groups`,
    plus the list of subjects each category holds so the page can expand it.

    A group the mapping does not know goes into OTHER AND into the returned
    `unmapped` list. Dropping it would hide a gap in the mapping; folding it in
    silently would misfile it. Listing it is what lets someone fix the file.
    Returns ({}, []) when nothing maps at all — the groups are then not MMLU
    subjects and a categories block would be one row called General & Multidisciplinary.
    """
    cats: dict[str, dict] = {}
    unmapped: list[str] = []
    mapped = 0
    for name, g in sorted(groups.items()):
        cat = _categories.categorize(name)
        if cat is None:
            unmapped.append(name)
            cat = _categories.OTHER
        else:
            mapped += 1
        c = cats.setdefault(cat, {"n": 0, "n_report": 0, "hit_report": 0.0,
                                  "n_diagnose": 0, "hit_diagnose": 0.0,
                                  "buckets": collections.Counter(), "groups": []})
        c["n"] += g["n"]
        c["n_report"] += g["n_report"]
        c["hit_report"] += g["hit_report"]
        c["n_diagnose"] += g["n_diagnose"]
        c["hit_diagnose"] += g["hit_diagnose"]
        c["buckets"].update(g["buckets"])
        c["groups"].append(name)
    if not mapped:
        return {}, []
    out = {}
    for cat in _categories.category_order():
        c = cats.get(cat)
        if not c:
            continue
        out[cat] = {"n": c["n"], "n_report": c["n_report"], "n_diagnose": c["n_diagnose"],
                    "score_report": (round(c["hit_report"] / c["n_report"], 6)
                                     if c["n_report"] else None),
                    "score_diagnose": (round(c["hit_diagnose"] / c["n_diagnose"], 6)
                                       if c["n_diagnose"] else None),
                    "buckets": dict(c["buckets"]), "groups": c["groups"]}
    return out, unmapped


def diagnose_task(files: list[Path]) -> dict | None:
    agg = {
        "n": 0, "n_report": 0, "n_diagnose": 0,
        "hit_report": 0.0, "hit_diagnose": 0.0, "hit_all": 0.0,
        "buckets": collections.Counter(), "groups": {},
        "examples": collections.defaultdict(list),
        "metric": None, "approx_buckets": False,
        # task-level shape of the answers, which is where the failures that are
        # invisible on a leaderboard live
        "picks": collections.Counter(),   # which option index the model chose
        "gold": collections.Counter(),    # which index was correct
        "n_len": 0, "short_pick": 0,      # did it pick the shortest option?
        "opts": collections.Counter(),
        "n_scored": 0, "n_target_ok": 0, "multi_true": 0,
    }
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pm = primary_metric(rec)
                if pm is None:
                    continue
                metric, value = pm
                agg["metric"] = agg["metric"] or metric
                dh = rec.get("doc_hash") or rec.get("prompt_hash")
                if not dh:
                    continue
                half = split_of(str(dh))
                right = value >= 0.5
                doc = rec.get("doc") or {}
                group = doc.get("subject") or doc.get("category") or "—"

                resps = rec.get("filtered_resps") or rec.get("resps") or []
                lps: list[float] = []
                for r in resps:
                    if isinstance(r, list) and r and isinstance(r[0], list):
                        r = r[0]                       # resps nests one deeper
                    lps.append(_f(r[0] if isinstance(r, list) else r))
                b = "right" if right else "wrong"
                probs: list[float] = []
                if len(lps) >= 2:
                    if metric == "acc_norm":
                        L = norm_lengths(doc, len(lps))
                        if L:
                            lps = [x / l for x, l in zip(lps, L)]
                        else:
                            agg["approx_buckets"] = True
                    probs = softmax(lps)
                    ci = target_index(rec, len(lps))
                    agg["n_scored"] += 1
                    if ci is not None:
                        agg["n_target_ok"] += 1
                    # multi-true tasks (TruthfulQA mc2) score the total mass on
                    # ALL correct options — there is no single right index, so
                    # every statistic below is meaningless for them
                    for k in ("mc2_targets", "mc1_targets"):
                        tv = doc.get(k)
                        if isinstance(tv, dict) and isinstance(tv.get("labels"), list):
                            if sum(1 for x in tv["labels"] if x) > 1:
                                agg["multi_true"] += 1
                            break
                    b = bucket(probs, ci, right)
                    pick = max(range(len(probs)), key=lambda i: probs[i])
                    agg["opts"][len(probs)] += 1
                    agg["picks"][pick] += 1
                    if ci is not None:
                        agg["gold"][ci] += 1
                    # length bias: a small model's raw loglikelihood is heavily
                    # influenced by how many tokens an option has, so "always
                    # picks the shortest" is a real and common failure that
                    # looks like knowledge on a score sheet
                    Lb = norm_lengths(doc, len(probs))
                    if Lb:
                        agg["n_len"] += 1
                        if pick == min(range(len(Lb)), key=lambda i: Lb[i]):
                            agg["short_pick"] += 1

                agg["n"] += 1
                agg["hit_all"] += 1.0 if right else 0.0
                agg["n_" + half] += 1
                agg["hit_" + half] += 1.0 if right else 0.0
                agg["buckets"][b] += 1
                g = agg["groups"].setdefault(group, {
                    "n": 0, "n_report": 0, "hit_report": 0.0,
                    "n_diagnose": 0, "hit_diagnose": 0.0,
                    "buckets": collections.Counter()})
                g["n"] += 1
                g["buckets"][b] += 1
                # both halves per group: the before/after comparison of a
                # retrained model needs the half the training saw AND the half
                # it never did, per category
                if half == "report":
                    g["n_report"] += 1
                    g["hit_report"] += 1.0 if right else 0.0
                else:
                    g["n_diagnose"] += 1
                    g["hit_diagnose"] += 1.0 if right else 0.0

                # examples come from the DIAGNOSE half only. This is the whole
                # safety property: a report item is never shown to a human and
                # never reaches a generator, so nothing downstream can be
                # derived from it.
                # Keep a few candidates per group and thin them at the end.
                # Taking the first MAX_EXAMPLES in file order means every
                # example comes from whichever subject sorts first, which makes
                # the list look like a subject finding when it is not one.
                if (half == "diagnose" and b != "right"
                        and len(agg["examples"][(b, group)]) < MAX_EXAMPLES):
                    ci = target_index(rec, len(lps) or 4)
                    ch = doc.get("choices")
                    if isinstance(ch, dict):
                        ch = ch.get("text")
                    pick = max(range(len(probs)), key=lambda i: probs[i]) if probs else None
                    agg["examples"][(b, group)].append({
                        "group": group,
                        "q": str(doc.get("question") or doc.get("query")
                                 or doc.get("ctx") or doc.get("goal") or "")[:240],
                        "chose": (str(ch[pick])[:90]
                                  if isinstance(ch, list) and pick is not None
                                  and pick < len(ch) else None),
                        "answer": (str(ch[ci])[:90]
                                   if isinstance(ch, list) and ci is not None
                                   and ci < len(ch) else None),
                        "p": round(max(probs), 3) if probs else None,
                    })
    if not agg["n"]:
        return None
    out = {
        "metric": agg["metric"],
        "n": agg["n"], "n_report": agg["n_report"], "n_diagnose": agg["n_diagnose"],
        "score_all": round(agg["hit_all"] / agg["n"], 6),
        "score_report": (round(agg["hit_report"] / agg["n_report"], 6)
                         if agg["n_report"] else None),
        "score_diagnose": (round(agg["hit_diagnose"] / agg["n_diagnose"], 6)
                           if agg["n_diagnose"] else None),
        "buckets": dict(agg["buckets"]),
        "approx_buckets": agg["approx_buckets"],
        "examples": spread_examples(agg["examples"]),
        "groups": {},
    }
    # ---- how the answers are shaped, not just how many were right ----------
    #
    # Everything below assumes ONE correct option per item and a stable option
    # count. TruthfulQA mc2 satisfies neither: several options are true and the
    # metric is the mass on all of them, so "the correct index" does not exist
    # and every statistic derived from it is noise. Reporting a finding that
    # fires on every model — including the strongest — is worse than reporting
    # none, so say the analysis does not apply and stop.
    npick = sum(agg["picks"].values())
    ok_share = (agg["n_target_ok"] / agg["n_scored"]) if agg["n_scored"] else 0.0
    modal_share = ((agg["opts"].most_common(1)[0][1] / sum(agg["opts"].values()))
                   if agg["opts"] else 0.0)
    why = None
    if agg["multi_true"]:
        why = "multi-true task: several options are correct, so there is no single answer index"
    elif agg["n_scored"] and ok_share < 0.90:
        why = f"only {ok_share:.0%} of items expose a single answer index"
    elif agg["opts"] and modal_share < 0.80:
        why = "the number of options varies across items"
    if why:
        out["answers"] = {"unsupported": why}
        # buckets computed against a nonexistent correct index are not evidence
        out["buckets"] = {"right": agg["buckets"].get("right", 0),
                          "wrong": sum(v for k, v in agg["buckets"].items()
                                       if k != "right")}
        out["examples"] = {}
        npick = 0
    if npick:
        nopt = (agg["opts"].most_common(1)[0][0]) if agg["opts"] else 0
        top_i, top_n = agg["picks"].most_common(1)[0]
        share = top_n / npick
        out["answers"] = {
            "n_options": nopt,
            "picks": {str(k): v for k, v in sorted(agg["picks"].items())},
            "gold": {str(k): v for k, v in sorted(agg["gold"].items())},
            "top_choice": top_i,
            "top_share": round(share, 4),
            # answering the same letter almost every time: chance-level score,
            # broken model. Not the same finding as "does not know the subject",
            # and not fixable with more subject data.
            "degenerate": bool(share >= DEGENERATE_SHARE),
        }
        # How far the answers are from the answer key's own distribution, as
        # total variation distance. This is the general form of the degeneracy
        # check and it catches what the single-option version cannot: a model
        # spreading 92% of its picks over A and B while the gold answer is
        # uniform over four is not ignorant, it cannot reach half the slots —
        # and its ceiling is about 0.47 x chance no matter what it knows.
        gtot = sum(agg["gold"].values())
        if gtot and nopt:
            tv = 0.5 * sum(abs(agg["picks"].get(i, 0) / npick
                               - agg["gold"].get(i, 0) / gtot)
                           for i in range(nopt))
            out["answers"]["pick_skew"] = round(tv, 4)
            out["answers"]["position_biased"] = bool(tv >= POSITION_SKEW)
        if agg["n_len"]:
            rate = agg["short_pick"] / agg["n_len"]
            base = 1.0 / nopt if nopt else None
            out["answers"]["short_pick_rate"] = round(rate, 4)
            out["answers"]["short_pick_baseline"] = round(base, 4) if base else None
            # picking by option length rather than content — looks like
            # knowledge on a score sheet, is not
            out["answers"]["length_biased"] = bool(base and rate >= base * 1.6)
    for name, g in sorted(agg["groups"].items()):
        out["groups"][name] = {
            "n": g["n"], "n_report": g["n_report"], "n_diagnose": g["n_diagnose"],
            "score_report": (round(g["hit_report"] / g["n_report"], 6)
                             if g["n_report"] else None),
            "score_diagnose": (round(g["hit_diagnose"] / g["n_diagnose"], 6)
                               if g["n_diagnose"] else None),
            "buckets": dict(g["buckets"]),
        }
    if len(out["groups"]) < 2:        # a single "—" group carries no information
        out["groups"] = {}
    if out["groups"]:
        cats, unmapped = rollup_categories(agg["groups"])
        if cats:
            out["categories"], out["unmapped"] = cats, unmapped
    return out


def diagnose_model(model_dir: Path) -> dict:
    tasks: dict[str, dict] = {}
    for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        files = newest_per_subtask(sorted(task_dir.rglob("samples_*.jsonl")))
        if not files:
            continue
        task = re.sub(r"_\d+shot$", "", task_dir.name)
        d = diagnose_task(files)
        if d:
            tasks[task] = d
    return {"split_salt": SPLIT_SALT, "tasks": tasks,
            "thresholds": {"chance_lift": CHANCE_LIFT,
                           "near_miss_gap": NEAR_MISS_GAP,
                           "degenerate_share": DEGENERATE_SHARE,
                           "min_group_n": MIN_GROUP_N,
                           "note": "lift = probability x n_options; 1.0 is chance"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="the results/full tree")
    ap.add_argument("-m", "--model", action="append", default=[],
                    help="only this model id (repeatable); default every model")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write diagnose.json under this directory instead of "
                         "beside the results (use when the results tree is not "
                         "yours to write)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    want = {m.replace("/", "__") for m in a.model}
    n, denied, found = 0, [], {}
    unmapped: dict[str, set] = {}
    for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        out = diagnose_model(d)
        if not out["tasks"]:
            continue
        dest = (a.out / d.name) if a.out else d
        try:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "diagnose.json").write_text(json.dumps(out), encoding="utf-8")
        except OSError as e:
            # the service writes results as root, so a results tree is often not
            # writable by the person running this. One unwritable model must not
            # abandon the other twenty-nine.
            denied.append((d.name, e.strerror or str(e)))
            continue
        n += 1
        # findings, collected for the summary — a tool that detects a broken
        # answer distribution should say so without being asked a second time
        for t, v in sorted(out["tasks"].items()):
            ans = v.get("answers") or {}
            if ans.get("unsupported"):
                continue
            for label, on in (("answers one option", ans.get("degenerate")),
                              ("answer positions skewed", ans.get("position_biased")),
                              ("picks by option length", ans.get("length_biased"))):
                if on:
                    found.setdefault(label, []).append(f"{d.name}/{t}")
            nb = v["n"] or 1
            if v["buckets"].get("confident_wrong", 0) / nb >= CONFIDENT_SHARE:
                found.setdefault("confidently wrong on most items", []).append(
                    f"{d.name}/{t}")
            for g in v.get("unmapped", []):
                unmapped.setdefault(g, set()).add(t)
        if not a.quiet:
            bits = []
            for t, v in sorted(out["tasks"].items()):
                sr = v["score_report"]
                bits.append(f"{t}={sr:.3f}" if sr is not None else f"{t}=—")
            print(f"{d.name:46} {' '.join(bits)}")
    if not a.quiet:
        print(f"\nwrote diagnose.json for {n} model(s) "
              f"· split salt {SPLIT_SALT!r}")
        if found:
            print("\nFINDINGS — failures a score cannot show:")
            for label in sorted(found):
                who = found[label]
                print(f"\n  {label}  ({len(who)})")
                for w in who[:12]:
                    print(f"      {w}")
                if len(who) > 12:
                    print(f"      … and {len(who) - 12} more")
            print("\n  None of these are fixed by more training data for the "
                  "subject:\n  they are properties of the output distribution, "
                  "not of what the model knows.")
        if unmapped:
            # a mapping gap is a fact about scripts/categories.yaml, not about
            # any model — say it once, board-wide, where it will be read
            print(f"\n{len(unmapped)} group(s) are not in scripts/categories.yaml and "
                  f"were rolled into {_categories.OTHER!r}:")
            for g in sorted(unmapped):
                print(f"      {g}  ({', '.join(sorted(unmapped[g]))})")
    if denied:
        print(f"\ncould not write {len(denied)} model(s) — the results tree is "
              f"owned by whoever ran the eval:", file=sys.stderr)
        for name, why in denied[:8]:
            print(f"  {name}: {why}", file=sys.stderr)
        if len(denied) > 8:
            print(f"  … and {len(denied) - 8} more", file=sys.stderr)
        print("\nrun it the way the service does (inside the container), or send "
              "the output elsewhere with --out:\n"
              "  sudo docker compose exec -T bench python3 scripts/diagnose.py results/full\n"
              "  python3 scripts/diagnose.py results/full --out ~/diagnose",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
