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
                    "buckets": collections.Counter()})
                g["n"] += 1
                g["buckets"][b] += 1
                if half == "report":
                    g["n_report"] += 1
                    g["hit_report"] += 1.0 if right else 0.0

                # examples come from the DIAGNOSE half only. This is the whole
                # safety property: a report item is never shown to a human and
                # never reaches a generator, so nothing downstream can be
                # derived from it.
                if (half == "diagnose" and b not in ("right",)
                        and len(agg["examples"][b]) < MAX_EXAMPLES):
                    ci = target_index(rec, len(lps) or 4)
                    ch = doc.get("choices")
                    if isinstance(ch, dict):
                        ch = ch.get("text")
                    pick = max(range(len(probs)), key=lambda i: probs[i]) if probs else None
                    agg["examples"][b].append({
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
        "examples": {k: v for k, v in agg["examples"].items() if v},
        "groups": {},
    }
    # ---- how the answers are shaped, not just how many were right ----------
    npick = sum(agg["picks"].values())
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
            "n": g["n"], "n_report": g["n_report"],
            "score_report": (round(g["hit_report"] / g["n_report"], 6)
                             if g["n_report"] else None),
            "buckets": dict(g["buckets"]),
        }
    if len(out["groups"]) < 2:        # a single "—" group carries no information
        out["groups"] = {}
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
    n, denied = 0, []
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
        if not a.quiet:
            bits = []
            for t, v in sorted(out["tasks"].items()):
                sr = v["score_report"]
                bits.append(f"{t}={sr:.3f}" if sr is not None else f"{t}=—")
            print(f"{d.name:46} {' '.join(bits)}")
    if not a.quiet:
        print(f"\nwrote diagnose.json for {n} model(s) "
              f"· split salt {SPLIT_SALT!r}")
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
