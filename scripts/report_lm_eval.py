#!/usr/bin/env python3
"""
Turn lm-evaluation-harness output into one self-contained interactive HTML report.

    python scripts/report_lm_eval.py results/full -o artifacts/benchmark_report.html \
        --csv artifacts/benchmark.csv

`results/` is whatever you passed to `lm_eval --output_path`. The script walks it,
finds every results JSON (run_benchmarks.sh writes one per model+task), merges them
per model, and builds a dashboard:

  Overview      the headline: best model, biggest statistically-real gap, warnings
  Leaderboard   sortable table — every model x every task, with stderr
  Tasks         one panel per benchmark, models ranked, chance line drawn
  Scaling       score vs parameter count on a log axis — THE plot for a model ladder
  Perplexity    bits-per-byte on pinned corpora (lower is better, tokenizer-neutral)
  Significance  pairwise z-test matrix — which gaps are real, which are noise
  Runs          full provenance + every metric + CSV/JSON export

Design decisions worth knowing:

  * SINGLE FILE, NO NETWORK. The data is embedded as JSON; charts are hand-drawn
    SVG; no CDN, no build step. It renders identically over `python -m http.server`
    on a Tailscale IP, from an email attachment, or off a USB stick in a demo room.
    (A Next.js + backend version of this is deliberate overkill until the data is
    live — a static batch of results wants a static artifact you can archive next
    to the numbers it argues for.)
  * The z-test gates every "X beats Y" claim. Bars persuade; the matrix decides.
  * Perplexities never share an axis with accuracies (unbounded, lower-is-better),
    and never enter the z-test (not proportions).
  * Parameter counts come from the harness config when present, else are parsed
    from the model name (pythia-410m -> 410e6) — the provenance table says which.

Deliberately dependency-free (stdlib only) so it runs anywhere your harness runs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import os
import re
from pathlib import Path

import categories as _categories   # scripts/categories.yaml: MMLU subjects -> categories

# Timestamps render in TZ (e.g. TZ=Asia/Qatar) when set — otherwise whatever
# this process's local time is. Inside a container that defaults to UTC, which
# reads three hours wrong in Doha; docker-compose sets TZ for exactly that.
def _tzinfo():
    name = os.environ.get("TZ")
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return None


_TZ = _tzinfo()

# ---------------------------------------------------------------------------
# parsing lm-eval output
# ---------------------------------------------------------------------------

# lm-eval writes metrics as "<metric>,<filter>" e.g. "acc,none", "acc_norm,none",
# with a matching "<metric>_stderr,<filter>". We parse generically rather than
# hard-coding metric names, so new tasks and metrics work without a code change.
_METRIC_RE = re.compile(r"^(?P<metric>[a-zA-Z0-9_@\-]+?)(?P<stderr>_stderr)?,(?P<filter>.+)$")


_META_CACHE: dict = {}


def _beside(source: Path, name: str) -> dict | None:
    """Read a JSON file that sits at the MODEL's directory level. Results files
    live one or two levels below it, so try both.

    The cache is keyed on (path, mtime, size) and an ABSENT file is never
    cached. That matters because the service imports this module once and lives
    for weeks: caching "no diagnose.json here" by path alone means a model whose
    diagnosis is written after the service started never shows one until someone
    restarts the container, and the same goes for a diagnosis that gets
    regenerated. A stat per candidate per rebuild is nothing next to that.
    """
    if len(_META_CACHE) > 4096:           # one entry per rewrite; keep it bounded
        _META_CACHE.clear()
    for up in (1, 2):
        if len(source.parents) <= up:
            continue
        cand = source.parents[up] / name
        try:
            st = cand.stat()
        except OSError:
            continue                      # not here (yet) — nothing to remember
        key = (str(cand), st.st_mtime_ns, st.st_size)
        if key not in _META_CACHE:
            try:
                _META_CACHE[key] = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                _META_CACHE[key] = None
        if _META_CACHE[key] is not None:
            return _META_CACHE[key]
    return None


def _beside_mtime(source: Path, name: str) -> float | None:
    """When the file _beside read was last written, for a record that has no
    time of its own to show."""
    for up in (1, 2):
        if len(source.parents) > up:
            try:
                return (source.parents[up] / name).stat().st_mtime
            except OSError:
                continue
    return None


def _model_meta(source: Path) -> dict | None:
    """Written by the service at preflight: architecture, param count, template."""
    return _beside(source, "model_meta.json")


# Per-item diagnosis, written by scripts/diagnose.py from the harness's
# --log_samples output. Absent until that has been run, which is why every
# consumer treats it as optional.
#
# It is trimmed here rather than there. diagnose.json is the archive — it keeps
# 8 examples per bucket per task with 240 characters of question — and this page
# is a single file that has to load over a tailnet. Four examples is enough to
# see the pattern; the file on disk still has the rest.
_DIAG_EXAMPLES = 4
_DIAG_Q = 180


def _clip(v, n: int):
    if not v:
        return None
    v = str(v)
    return v if len(v) <= n else v[:n - 1].rstrip() + "\u2026"


def _trim_diag(d: dict | None) -> dict | None:
    if not isinstance(d, dict) or not d.get("tasks"):
        return None
    out = {"split_salt": d.get("split_salt"), "tasks": {}}
    for task, v in d["tasks"].items():
        if not isinstance(v, dict):
            continue
        t = {k: v.get(k) for k in
             ("metric", "n", "n_report", "n_diagnose", "score_all", "score_report",
              "score_diagnose", "buckets", "approx_buckets", "groups", "answers",
              "categories", "unmapped")
             if v.get(k) is not None}
        ex = {}
        for bucket, items in (v.get("examples") or {}).items():
            keep = []
            for e in items[:_DIAG_EXAMPLES]:
                keep.append({
                    "group": e.get("group"),
                    "q": _clip(e.get("q"), _DIAG_Q),
                    "chose": _clip(e.get("chose"), 70),
                    "answer": _clip(e.get("answer"), 70),
                    "p": e.get("p"),
                })
            if keep:
                ex[bucket] = keep
        if ex:
            t["examples"] = ex
        out["tasks"][task] = t
    return out


# ---------------------------------------------------------------------------
# Proposal gating: when a "propose a skill spec" button may be live.
#
# The single most important piece of UX in the generation phase. Four of the
# five things a diagnosis can find are properties of how we POSE a task or of
# the output distribution, and more subject data moves the score without
# teaching the model anything. A generator must not be offered for any of
# them. The same function gates the API, so the button and the server cannot
# disagree, and the reason is text the page shows as written.
# ---------------------------------------------------------------------------
PROPOSE_MIN_N = 30            # diagnose.MIN_GROUP_N — DIAGNOSE.md's noise floor
_FORMAT_FLAGS = (("degenerate", "one option only"), ("position_biased", "answer positions"),
                 ("length_biased", "option length"))
_CONFIDENT_SHARE = 0.35       # diagnose.CONFIDENT_SHARE; the page's chip uses the same line


def proposal_gate(task: str, t: dict, cell: dict | None, chance: float | None) -> dict:
    """{ok, why, categories: {name: {ok, why}}} for one model x task, from the
    trimmed diagnosis and the reported cell."""
    a = t.get("answers") or {}
    why = None
    if a.get("unsupported"):
        why = "per-item analysis does not apply to this task — " + str(a["unsupported"])
    else:
        # a format finding first: a position-skewed model is usually at chance
        # too, and the skew is the cause — the reason should name the cause
        flags = [label for key, label in _FORMAT_FLAGS if a.get(key)]
        nb = t.get("n") or 1
        if (t.get("buckets") or {}).get("confident_wrong", 0) / nb >= _CONFIDENT_SHARE:
            flags.append("confidently wrong")
        v = se = None
        if cell and cell.get("se"):
            v, se = cell["v"], cell["se"]
        elif t.get("score_report") is not None and t.get("n_report"):
            v = t["score_report"]
            se = math.sqrt(max(v * (1 - v), 1e-9) / t["n_report"])
        if flags:
            why = ("a format failure, not a knowledge gap: " + ", ".join(flags)
                   + " — see the finding above. Fix the posing; data will not")
        elif chance and v is not None and v - 1.96 * se <= chance:
            why = (f"the score ({100 * v:.1f}% ±{100 * se:.1f}) has not cleared chance "
                   f"({100 * chance:.0f}%) — the breakdown describes how the model guesses, "
                   f"not what it knows, so no category is evidence of a gap")
    cats = {}
    for name, g in (t.get("categories") or {}).items():
        if why:
            cats[name] = {"ok": False, "why": why}
        elif (g.get("n_report") or 0) < PROPOSE_MIN_N:
            cats[name] = {"ok": False, "why": f"{g.get('n_report') or 0} leaderboard-half items "
                          f"— under the {PROPOSE_MIN_N}-item noise floor, so this score is not "
                          f"evidence of a gap"}
        else:
            cats[name] = {"ok": True, "why": None}
    return {"ok": why is None, "why": why, "categories": cats}


# ---------------------------------------------------------------------------
# Judged free response (scripts/judge.py, scripts/judge_calibrate.py).
#
# The judge's scores are a benchmark only once a person has graded a sample
# and agreement (Cohen's kappa) clears KAPPA_MIN. Below that the numbers are
# shown, marked preliminary, and never ranked or averaged — the same mechanism
# a model missing required tasks gets. Above it, category scores may enter a
# SEPARATE judged average; they never touch the multiple-choice average.
# ---------------------------------------------------------------------------
KAPPA_MIN = 0.60
# the exam's spine is the topic list in scripts/categories.yaml; one task each
EXAM_TASKS = [f"exam_{_categories.topic_slug(c)}" for c in _categories.category_order()]
EXAM_TOPICS = {f"exam_{_categories.topic_slug(c)}": c for c in _categories.category_order()}
FR_CONTROL = "fr_control_mmlu"


def _trim_judge(j: dict | None) -> dict | None:
    """The per-item lists stay on disk (the calibration tool and the proposal
    step read them); the page gets the summaries. A file from the phase-5
    local judge has no provider: it is labelled `local`, kept, and shown as a
    different series from API-judged scores rather than merged."""
    if not isinstance(j, dict) or "judge" not in j:
        return None
    jj = dict(j["judge"])
    if not jj.get("provider"):
        jj["provider"] = "stub" if jj.get("stub") else "local"
        jj["model"] = jj.get("id", "")
    out = {"judge": {k: jj.get(k) for k in
                     ("id", "provider", "model", "family", "stub", "weights_sha256",
                      "prompt_sha256", "prompt_version", "rubrics", "greedy", "batch_id",
                      "single_provider_loop",
                      # a local judge (service/llm.py::local_mark): what it was
                      # and why nothing it graded counts
                      "provisional", "provisional_reason", "base_url", "served_model",
                      "weights",
                      # a rubric whose author has not signed it off yet
                      "rubric_status", "rubrics_draft")},
           "skipped": j.get("skipped"), "correct_at": j.get("correct_at"),
           "canary": ({k: v for k, v in (j.get("canary") or {}).items()
                       if k in ("n", "graded", "mad_vs_human", "mad_vs_previous", "threshold",
                                "drifted")} if j.get("canary") else None),
           "preliminaryReasons": list(j.get("preliminary_reasons") or []), "tasks": {}}
    for task, t in (j.get("tasks") or {}).items():
        if not isinstance(t, dict):
            continue
        # the flagged qids stay on disk: the page has no use for them, and a
        # qid list is the one thing this file must not hand out casually
        if isinstance(t.get("flags"), dict):
            t = {**t, "flags": {fid: {k: v for k, v in f.items() if k != "qids"}
                                for fid, f in t["flags"].items()}}
        out["tasks"][task] = {k: t.get(k) for k in
                              ("n", "mean", "max", "dist", "dist_report", "dist_diagnose",
                               "score_vs_length", "control",
                               "n_report", "score_report", "n_diagnose", "score_diagnose",
                               # what the model actually wrote: the gate reads it
                               "answers", "ungraded",
                               # a task graded criterion by criterion: how each
                               # one did, the flags that override them all, and
                               # the breakdown by whichever metadata fields the
                               # topic carries. No per-item lists.
                               "criteria_mean", "criteria_n", "criteria_labels",
                               "criteria_conditional", "flags", "breakdowns",
                               "breakdowns_constant", "unparseable",
                               # 11l: answers that never left a reasoning
                               # block, why a topic has no score, and what
                               # the answers were generated with
                               "no_answer", "no_score", "generation",
                               # when this topic's grades landed — its own
                               # time, not the file's last merge
                               "judged_at")
                              if t.get(k) is not None}
    return out


def _judged_now(row: dict, fingerprints: dict[str, str] | None) -> dict | None:
    """The trimmed judge.json of one model with only the topics that count
    now, and the rest as `history` (scripts/judge.py::split_by_bank)."""
    from judge import split_by_bank
    now, hist = split_by_bank(row.get("judge"), fingerprints, row.get("judge_mtime"))
    out = _trim_judge(now)
    if out is not None:
        out["history"] = hist
    return out


def published_score(t: dict) -> float | None:
    """The per-topic number the board shows: the REPORT half's mean. The
    diagnose half is what a proposal may be built from and never the score.
    Older judge.json files (no halves recorded) fall back to the overall mean."""
    if t.get("score_report") is not None:
        return t["score_report"]
    return t.get("mean") if "n_report" not in t else None


def judged_state(trimmed: dict | None, cal: dict | None, current_id: str | None) -> dict:
    """Whether this model's judged numbers may be ranked: {ok, reasons,
    current}. Preliminary when the judge is uncalibrated, calibrated as a
    different judge, not the judge this server runs now, or when its canary
    moved. The single-provider caveat is a stamp, not a reason."""
    reasons = []
    if not trimmed:
        return {"ok": False, "reasons": ["not judged"], "current": False}
    jid = (trimmed.get("judge") or {}).get("id")
    current = bool(current_id) and jid == current_id
    if current_id and not current:
        reasons.append(f"judged by {jid}, not the judge this server runs now ({current_id}) — "
                       f"a different series, not comparable")
    if not cal:
        reasons.append("the judge has not been calibrated against a person")
    elif not cal.get("calibrated"):
        reasons.append(f"Cohen's kappa {cal.get('kappa')} is below {KAPPA_MIN}")
    elif cal.get("judge_id") and jid and cal["judge_id"] != jid:
        reasons.append(f"the calibration on file is for {cal['judge_id']}, not {jid}")
    reasons.extend(trimmed.get("preliminaryReasons") or [])
    prov = provisional_reason(trimmed)
    if prov and prov not in reasons:
        reasons.append(prov)
    if trimmed.get("skipped"):
        reasons.append(trimmed["skipped"])
    return {"ok": not reasons, "reasons": reasons, "current": current}


def provisional_reason(trimmed: dict | None) -> str | None:
    """Why a local judge's scores never count, or None for any other judge.
    Not a flag anyone sets: scripts/judge.py stamps every file a local judge
    writes, and there is no way to write one without it."""
    j = (trimmed or {}).get("judge") or {}
    if not j.get("provisional"):
        return None
    return j.get("provisional_reason") or "graded by a local model — not a pinned benchmark"


# A model that wrote nothing on a topic has not revealed a gap in it.
EMPTY_SHARE = 0.5          # empty or near-empty answers, as a share of the topic
DEGENERATE_DISTINCT = 2    # distinct answers, at or below which it wrote one thing


def topic_gate(task: str, t: dict, state: dict | None, caution: str | None,
               head: dict | None = None) -> dict:
    """Whether a skill spec may be proposed from one exam topic, as two kinds
    of reason, both always evaluated:

    - `soft`: about the JUDGE — not calibrated against a person, provisional
      (a local model), a canary that moved, a different judge than the one
      this server runs. Overridable in the service, with a warning, and the
      mark travels with everything made from it. `soft_extra` holds the two
      caveats that are not gates today (single-provider loop, draft rubric):
      they join `soft` only where the override is on.
    - `hard`: about the DATA — too few report-half questions to have a score,
      a model that wrote nothing usable, output that collapsed. Never
      overridable: no warning makes noise evidence.

    `ok`, `why` and `short` are the gate exactly as it was before the split:
    the preliminary suite first, then the data. The service's
    ALLOW_PRELIMINARY_OVERRIDE=0 uses them unchanged. `caution` is MMLU's
    finding for the same category — context, never a gate."""
    head = head or {}
    soft = list((state or {}).get("reasons") or ([] if (state or {}).get("ok")
                                                  else ["the judged suite is preliminary"]))
    soft_extra = []
    if head.get("single_provider_loop"):
        soft_extra.append("single-provider loop: the same provider wrote, sat or graded more "
                          "than one step of this loop")
    if task in (head.get("rubrics_draft") or []):
        soft_extra.append("the rubric for this topic is a draft its author has not signed off")
    hard = []
    n_rep = t.get("n_report") or 0
    a = t.get("answers") or {}
    # 11l: an answer that never left its reasoning block is not an answer —
    # not a blank one, not a repeated one. The checks below read the rest
    n = (a.get("n") or 0) - (a.get("no_answer") or 0)
    blank = (a.get("empty", 0) + a.get("short", 0))
    if t.get("no_score"):
        hard.append({"why": f"{t['no_score']} — {t.get('no_answer')} of {t.get('n')} answers "
                            f"stopped inside the model's reasoning. Sit it again: a reasoning "
                            f"model now gets room to answer",
                     "short": "not scored — the model never finished answering"})
    if n_rep < PROPOSE_MIN_N:
        hard.append({"why": f"{n_rep} hidden questions in this topic — under the "
                            f"{PROPOSE_MIN_N} a topic needs, so its score is noise. Write more on "
                            f"the Exam tab",
                     "short": f"under the {PROPOSE_MIN_N}-question floor — write more on the "
                              f"Exam tab"})
    if n and blank / n >= EMPTY_SHARE:
        hard.append({"why": f"the model wrote nothing usable on {blank} of {n} answers here — "
                            f"that is a generation failure, not a topic gap; multiple choice "
                            f"is the instrument for this model",
                     "short": f"the model wrote nothing usable on {blank} of {n} answers"})
    elif n >= 8 and (a.get("distinct") or n) <= DEGENERATE_DISTINCT:
        hard.append({"why": f"the model gave the same answer on nearly every question here "
                            f"({a.get('distinct')} distinct answers in {n}) — the output has "
                            f"collapsed, and no data for this topic fixes that",
                     "short": "the model gave the same answer on nearly every question"})
    if soft:
        why = ("the judged suite is preliminary, so no topic score is evidence yet: "
               + "; ".join(soft))
        short = "the judged suite is preliminary"
    elif hard:
        why, short = hard[0]["why"], hard[0]["short"]
    else:
        why = short = None
    # the caution is decision-relevant only where a decision is possible; on a
    # row that is already refused it would be one more line of noise
    return {"ok": why is None, "why": why, "short": short,
            "caution": caution if why is None else None,
            "soft": soft, "soft_extra": soft_extra, "hard": hard,
            "provisional": bool(head.get("provisional"))}


def judged_avg(trimmed: dict | None, tainted: list[str] | None = None) -> float | None:
    """Mean published (report-half) score over the exam topics this model sat,
    on the 0–4 scale. None when fewer than three topics were judged. A topic
    whose diagnostics trained this model is left out: its score is shown, it
    is not a ranking claim. A local judge's scores are in no average at all."""
    if not trimmed or trimmed.get("skipped") or provisional_reason(trimmed):
        return None
    skip = set(tainted or ())
    vals = [published_score(trimmed["tasks"][t]) for t in EXAM_TASKS
            if t in trimmed["tasks"] and t not in skip]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if len(vals) >= 3 else None


# ---------------------------------------------------------------------------
# Close the loop: what did the training teach?
#
# A tainted model trained on data derived from a task's DIAGNOSIS half. Its
# parent did not. The leaderboard half was never touched by that data, so it
# is the honest test: if the training taught the skill, both halves move
# together; if it taught the test, the half the generator saw moves and the
# other sits still — and that divergence is the alarm this whole design exists
# to raise. The sentence is derived from the four numbers and their errors and
# from nothing else.
# ---------------------------------------------------------------------------
_Z = 1.96


def _half(t: dict, half: str) -> dict | None:
    v, n = t.get(f"score_{half}"), t.get(f"n_{half}")
    if v is None or not n:
        return None
    return {"v": v, "n": n, "se": math.sqrt(max(v * (1 - v), 1e-9) / n)}


def taint_verdict(task: str, d_rep: float, se_rep: float, d_dia: float, se_dia: float,
                  scale: str = "pct") -> tuple[str, str, float | None]:
    """(verdict, sentence, ratio). Verdicts: skill | test | none | mixed.
    `scale` is how a delta reads: accuracy in percentage points, or a rubric
    mean out of four."""
    sig_rep, sig_dia = abs(d_rep) > _Z * se_rep, abs(d_dia) > _Z * se_dia
    z_rep = abs(d_rep) / se_rep if se_rep else 0.0
    z_dia = abs(d_dia) / se_dia if se_dia else 0.0
    ratio = (d_dia / d_rep) if abs(d_rep) > 1e-9 else None
    pts = ((lambda d: f"{100 * d:+.1f} points") if scale == "pct"
           else (lambda d: f"{d:+.2f} of 4"))
    if sig_dia and d_dia > 0 and not sig_rep:
        tail = (f"The diagnosis-half gain is {abs(ratio):.0f}× the leaderboard-half change."
                if ratio is not None and abs(ratio) >= 1.5
                else "The leaderboard half did not move at all.")
        return ("test",
                f"The training taught the test. On {task}, the half the generator's spec was "
                f"derived from rose {pts(d_dia)} ({z_dia:.1f} standard errors) while the half "
                f"it never saw moved {pts(d_rep)} ({z_rep:.1f} SE), within noise. {tail} The "
                f"score moved; the model did not learn the subject.", ratio)
    if sig_rep and sig_dia and (d_rep > 0) == (d_dia > 0):
        return ("skill",
                f"The training taught the skill. On {task} both halves moved together: the "
                f"leaderboard half {pts(d_rep)} ({z_rep:.1f} SE), the diagnosis half "
                f"{pts(d_dia)} ({z_dia:.1f} SE). The half the training never saw moved too, "
                f"which is what learning the subject looks like.", ratio)
    if not sig_rep and not sig_dia:
        return ("none",
                f"The training changed nothing measurable on {task}: leaderboard half "
                f"{pts(d_rep)} (±{100 * _Z * se_rep:.1f}), diagnosis half {pts(d_dia)} "
                f"(±{100 * _Z * se_dia:.1f}), both within noise.", ratio)
    return ("mixed",
            f"An unusual pattern on {task}: the report half moved {pts(d_rep)} "
            f"({z_rep:.1f} SE) and the diagnosis half {pts(d_dia)} ({z_dia:.1f} SE). Neither "
            f"reading fits; check that parent and child share a template and a harness build "
            f"before reading anything into it.", ratio)


def _rubric_half(t: dict, half: str) -> dict | None:
    """{v, n, se} for one half of a judged topic. The standard error is the
    ordinary one for a mean of bounded scores, read off the per-half score
    distribution the judge writes."""
    dist = (t or {}).get(f"dist_{half}") or {}
    n = sum(dist.values())
    if not n:
        return None
    mean = sum(int(k) * v for k, v in dist.items()) / n
    var = sum(v * (int(k) - mean) ** 2 for k, v in dist.items()) / n
    return {"v": round(mean, 4), "n": n, "se": round(math.sqrt(var / n), 4)}


def taint_exam_compare(task: str, after: dict, before: dict, parent: str,
                       judge_id: str | None = None) -> dict | None:
    """The same before/after for an exam topic: the published REPORT half
    against the DIAGNOSE half the generator's spec came from, on the rubric's
    0–4 scale. Both sides must come from the same judge, or the comparison is
    two different instruments and says so."""
    ta = ((after or {}).get("tasks") or {}).get(task)
    tb = ((before or {}).get("tasks") or {}).get(task)
    if not ta or not tb:
        return None
    ja = ((after or {}).get("judge") or {}).get("id")
    jb = ((before or {}).get("judge") or {}).get("id")
    if ja and jb and ja != jb:
        return {"parent": parent, "scale": "rubric",
                "missing": f"the parent was judged by {jb} and this model by {ja} — two "
                           f"different instruments, so there is no before to compare"}
    a_rep, a_dia = _rubric_half(ta, "report"), _rubric_half(ta, "diagnose")
    b_rep, b_dia = _rubric_half(tb, "report"), _rubric_half(tb, "diagnose")
    if not all((a_rep, a_dia, b_rep, b_dia)):
        return None
    d_rep, d_dia = a_rep["v"] - b_rep["v"], a_dia["v"] - b_dia["v"]
    se_rep = math.sqrt(a_rep["se"] ** 2 + b_rep["se"] ** 2)
    se_dia = math.sqrt(a_dia["se"] ** 2 + b_dia["se"] ** 2)
    verdict, text, ratio = taint_verdict(EXAM_TOPICS.get(task, task), d_rep, se_rep,
                                         d_dia, se_dia, scale="rubric")
    return {"parent": parent, "scale": "rubric", "judge": ja or jb,
            "before": {"report": b_rep, "diagnose": b_dia},
            "after": {"report": a_rep, "diagnose": a_dia},
            "dReport": round(d_rep, 6), "dDiagnose": round(d_dia, 6),
            "seReport": round(se_rep, 6), "seDiagnose": round(se_dia, 6),
            "verdict": verdict, "ratio": (round(ratio, 3) if ratio is not None else None),
            "text": text, "categories": {}}


def taint_compare(task: str, after: dict, before: dict, parent: str) -> dict | None:
    """The per-task, per-half, per-category comparison of a tainted model
    against its parent, from the two trimmed diagnoses."""
    ta, tb = (after.get("tasks") or {}).get(task), (before.get("tasks") or {}).get(task)
    if not ta or not tb:
        return None
    a_rep, a_dia, b_rep, b_dia = _half(ta, "report"), _half(ta, "diagnose"), \
        _half(tb, "report"), _half(tb, "diagnose")
    if not all((a_rep, a_dia, b_rep, b_dia)):
        return None
    d_rep, d_dia = a_rep["v"] - b_rep["v"], a_dia["v"] - b_dia["v"]
    se_rep = math.sqrt(a_rep["se"] ** 2 + b_rep["se"] ** 2)
    se_dia = math.sqrt(a_dia["se"] ** 2 + b_dia["se"] ** 2)
    verdict, text, ratio = taint_verdict(task, d_rep, se_rep, d_dia, se_dia)
    cats = {}
    for name, ca in (ta.get("categories") or {}).items():
        cb = (tb.get("categories") or {}).get(name)
        if not cb or ca.get("score_report") is None or cb.get("score_report") is None:
            continue
        cats[name] = {
            "before": {k: cb.get(k) for k in ("score_report", "score_diagnose", "n_report",
                                              "n_diagnose")},
            "after": {k: ca.get(k) for k in ("score_report", "score_diagnose", "n_report",
                                             "n_diagnose")},
            "dReport": round(ca["score_report"] - cb["score_report"], 6),
            "dDiagnose": (round(ca["score_diagnose"] - cb["score_diagnose"], 6)
                          if ca.get("score_diagnose") is not None
                          and cb.get("score_diagnose") is not None else None),
        }
    return {"parent": parent,
            "before": {"report": b_rep, "diagnose": b_dia},
            "after": {"report": a_rep, "diagnose": a_dia},
            "dReport": round(d_rep, 6), "dDiagnose": round(d_dia, 6),
            "seReport": round(se_rep, 6), "seDiagnose": round(se_dia, 6),
            "verdict": verdict, "ratio": (round(ratio, 3) if ratio is not None else None),
            "text": text, "categories": cats, "scale": "pct"}


def load_results(path: Path) -> list[dict]:
    """Find and parse every lm-eval results file under `path`."""
    files = sorted(path.rglob("results*.json")) if path.is_dir() else [path]
    runs = []
    for f in files:
        try:
            blob = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "results" not in blob:
            continue
        runs.append(parse_run(blob, f))
    return runs


def parse_run(blob: dict, source: Path) -> dict:
    cfg = blob.get("config", {}) or {}
    model_args = cfg.get("model_args") or ""
    if isinstance(model_args, dict):
        model_args = ",".join(f"{k}={v}" for k, v in model_args.items())

    # the model id lives inside model_args as pretrained=<id>
    m = re.search(r"pretrained=([^,\s]+)", str(model_args))
    model = m.group(1) if m else (cfg.get("model_name") or source.parent.name)
    # uploaded artifacts are evaluated by absolute path; normalize back to the
    # local/<name> id they were submitted under so every view joins on one key
    la = re.search(r"/artifacts/([^/,\s]+)/?$", model)
    if la:
        model = "local/" + la.group(1)

    tasks: dict[str, dict] = {}
    for task, metrics in (blob.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        entry: dict = {"alias": (metrics.get("alias") or task).strip()}
        for key, val in metrics.items():
            mm = _METRIC_RE.match(key)
            if not mm or not isinstance(val, (int, float)):
                continue
            # A NaN/Infinity score is not a score (a diverged model's perplexity
            # comes back as Infinity) — and it would poison JSON serialization
            # downstream: the API layer rightly refuses non-finite floats.
            if not math.isfinite(float(val)):
                continue
            name = mm.group("metric")
            # generate_until tasks that a judge scores afterwards carry the
            # harness's `bypass` sentinel (999), not a score; nothing here may
            # read it as one
            if name == "bypass":
                continue
            slot = "stderr" if mm.group("stderr") else "value"
            # keep the first filter seen per metric, but prefer flexible-extract for
            # generative tasks (gsm8k reports strict-match AND flexible-extract; the
            # flexible one is the number people mean)
            bucket = entry.setdefault(name, {})
            filt = mm.group("filter")
            prev = bucket.get(f"_filt_{slot}")
            if prev is None or (filt == "flexible-extract" and prev != "flexible-extract"):
                bucket[slot] = float(val)
                bucket[f"_filt_{slot}"] = filt
        tasks[task] = entry

    n_samples = {k: (v.get("effective") if isinstance(v, dict) else v)
                 for k, v in (blob.get("n-samples") or {}).items()}
    n_shot = dict(blob.get("n-shot") or {})
    # The harness tells us which tasks are children of a group (MMLU's 57 subjects,
    # its four category roll-ups, etc). Use it rather than pattern-matching names.
    group_subtasks = blob.get("group_subtasks") or {}
    subtasks: set[str] = set()
    for parent, kids in group_subtasks.items():
        for k in kids or []:
            if k != parent:
                subtasks.add(k)
    # n-samples and n-shot are recorded per LEAF task only (checked against a
    # real 0.4.12 results file). A group such as mmlu therefore had no item or
    # shot count of its own, and the biggest task on the board showed "—" for
    # both. Items are the sum over the group's leaves; shots are the one value
    # the leaves agree on — and nothing when they disagree, because a mixed
    # shot count is a warning, not a number.
    for g in group_subtasks:
        lv = _leaves(g, group_subtasks)
        if lv == [g]:
            continue
        if g not in n_samples and all(isinstance(n_samples.get(t), int) for t in lv):
            n_samples[g] = sum(n_samples[t] for t in lv)
        if g not in n_shot:
            shots = {n_shot.get(t) for t in lv}
            if len(shots) == 1 and None not in shots:
                n_shot[g] = shots.pop()
    # The harness tells us the direction of every metric. Use it rather than guessing
    # from the name: perplexity and bits-per-byte are LOWER-is-better and are not
    # proportions, so they must not share a chart with accuracies, and the
    # two-proportion z-test does not apply to them at all.
    hib: dict[str, dict[str, bool]] = {}
    for task, metrics in (blob.get("higher_is_better") or {}).items():
        if isinstance(metrics, dict):
            hib[task] = {k: bool(v) for k, v in metrics.items() if v is not None}
    return {
        "higher_is_better": hib,
        "subtasks": subtasks,
        "source": str(source),
        "model": model,
        "model_args": str(model_args),
        "backend": cfg.get("model"),
        "dtype": _extract(model_args, "dtype"),
        "batch_size": cfg.get("batch_size"),
        "device": cfg.get("device"),
        "limit": cfg.get("limit"),
        "seed": cfg.get("random_seed"),
        "fewshot_seed": cfg.get("fewshot_seed"),
        "chat_template": bool(blob.get("chat_template") or cfg.get("apply_chat_template")),
        "num_params": _to_float(cfg.get("model_num_parameters")),
        "n_shot": n_shot,
        "n_samples": n_samples,
        "git_hash": blob.get("git_hash"),
        "date": _norm_date(blob.get("date")),
        "transformers_version": blob.get("transformers_version"),
        "eval_seconds": _to_float(blob.get("total_evaluation_time_seconds")),
        "archinfo": _model_meta(source),
        "diag": _beside(source, "diagnose.json"),
        # rubric scores from scripts/judge.py, same place and pattern as the diagnosis
        "judge": _beside(source, "judge.json"),
        "judge_mtime": _beside_mtime(source, "judge.json"),
        "tasks": tasks,
    }


def _leaves(task: str, group_subtasks: dict, seen: tuple = ()) -> list[str]:
    """The leaf tasks under a group, through any depth of sub-groups."""
    kids = [k for k in (group_subtasks.get(task) or []) if k != task and k not in seen]
    if not kids:
        return [task]
    return [x for k in kids for x in _leaves(k, group_subtasks, seen + (task,))]


def _extract(args: str, key: str):
    m = re.search(rf"{key}=([^,\s]+)", str(args))
    return m.group(1) if m else None


def _norm_date(v):
    """lm-eval writes `date` as a unix float; normalize every form to an ISO string
    so sorting, slicing and display never meet a bare number."""
    if not v:
        return None
    try:
        return _dt.datetime.fromtimestamp(float(v), _TZ).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return str(v)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def primary_metric(entry: dict) -> tuple[str, float, float] | None:
    """Pick one headline metric per task, preferring the conventional one.

    acc_norm before acc for HellaSwag-style tasks, exact_match for GSM8K, pass@1 for
    code, bits_per_byte for perplexity corpora. The choice is recorded in the output
    so nobody has to guess which number they are looking at.
    """
    for name in ("acc_norm", "acc", "exact_match", "pass@1", "f1", "em",
                 "bits_per_byte", "byte_perplexity", "word_perplexity"):
        d = entry.get(name)
        if isinstance(d, dict) and "value" in d:
            return name, d["value"], d.get("stderr", 0.0)
    for name, d in entry.items():
        if isinstance(d, dict) and "value" in d:
            return name, d["value"], d.get("stderr", 0.0)
    return None


def significant(a: float, sa: float, b: float, sb: float, z: float = 1.96) -> tuple[bool, float]:
    """
    Two-proportion comparison. Returns (is_significant, z_score).

    diff / sqrt(se_a^2 + se_b^2) — the standard test for a difference between two
    independent estimates. |z| > 1.96 is the usual 95% threshold.

    Caveat worth knowing and stating: when both models were evaluated on the SAME
    items (which they were), a *paired* test is more sensitive — this one is
    conservative, so it will occasionally call a real difference insignificant. It
    will not do the opposite, which is the direction that matters.
    """
    se = math.sqrt(sa * sa + sb * sb)
    if se <= 0:
        return (a != b), float("inf") if a != b else 0.0
    zz = (a - b) / se
    return abs(zz) > z, zz


# ---------------------------------------------------------------------------
# payload — everything the dashboard needs, as one JSON blob
# ---------------------------------------------------------------------------

# Canonical display order for the tasks run_benchmarks.sh runs; anything else the
# harness produced is appended alphabetically. Chance level rides along: 25% on
# 4-option tasks, 50% on the 2-option ones — the single most misread thing on a
# small-model chart. truthfulqa_mc2 has no clean chance level (multi-true, weighted),
# and gsm8k's is 0.
_CANON = ["mmlu", "mmlu_perm", "hellaswag", "arc_challenge", "arc_easy",
          "winogrande", "piqa", "truthfulqa_mc2", "gsm8k"]
_CHANCE = {"mmlu": 0.25, "mmlu_perm": 0.25, "hellaswag": 0.25, "arc_challenge": 0.25,
           "arc_easy": 0.25, "winogrande": 0.5, "piqa": 0.5, "gsm8k": 0.0}

# Controls: tasks run to test how we POSE a benchmark, not what a model knows.
# They are shown wherever the task they control for is shown, and they never
# enter an average — official or partial — however REQUIRED_TASKS is set,
# because a control that could move a rank would stop being a control.
CONTROL_TASKS = {"mmlu_perm"}

# ---------------------------------------------------------------------------
# What each task is, AS WE RUN IT.
#
# Public descriptions describe a benchmark as the field uses it. That is not the
# thing on this page: our shot count, our metric choice and our template policy
# are part of what the number means, and they are what a friend reading this
# board needs in order to not misread it. So these are written for our protocol,
# not copied from anywhere.
#
# `domain` is the SAME vocabulary the radar's CATS folding uses — one taxonomy
# driving the chips and the axes, so the two cannot drift apart.
# ---------------------------------------------------------------------------
_TASK_META = {
    "mmlu": ("knowledge",
             "Four-choice exam questions across 57 academic and professional "
             "subjects. 5-shot, acc. Breadth of recall rather than depth of "
             "reasoning — and the task where a sub-1B model most often sits at "
             "chance, so read it against the chance line, not against 0%."),
    "mmlu_perm": ("knowledge (control)",
                  "MMLU re-posed: a fixed subset of subjects with the answer options "
                  "rotated by item index, so the correct answer visits every slot "
                  "equally. Same 5-shot, same acc. A control, not a leaderboard task: "
                  "it tells you whether a position-skewed model's MMLU score is about "
                  "the format or about the knowledge, and it never enters an average."),
    "hellaswag": ("commonsense",
                  "Four-choice sentence completion about everyday situations. "
                  "5-shot, acc_norm — length-normalized on purpose, because raw "
                  "accuracy here rewards picking the longest ending."),
    "arc_challenge": ("reasoning",
                      "Four-choice grade-school science questions, the split that "
                      "defeated retrieval baselines. 5-shot, acc_norm."),
    "arc_easy": ("reasoning",
                 "Four-choice grade-school science questions, the easier split. "
                 "5-shot, acc_norm. Usually the first task a small model clears "
                 "chance on, which makes it a useful early signal in a run."),
    "winogrande": ("commonsense",
                   "Two-choice pronoun resolution needing commonsense to bind the "
                   "referent. 5-shot, acc. Chance is 50%, so a score near 50 means "
                   "nothing at all — the most misread number on this board."),
    "piqa": ("commonsense",
             "Two-choice questions about physical commonsense — which of two "
             "procedures actually works. 0-shot, acc_norm. Chance is 50%."),
    "truthfulqa_mc2": ("truthfulness",
                       "Multiple choice on questions where a common human "
                       "misconception is the tempting answer. 0-shot, mc2 — "
                       "several options can be true, so there is no clean chance "
                       "level and none is drawn."),
    "gsm8k": ("math",
              "Grade-school word problems needing several arithmetic steps, "
              "scored by exact match on the final answer. 5-shot, generative. "
              "Reported here but deliberately excluded from the overall average: "
              "it sits near 0% below ~1B and only adds noise to a mean."),
}

# ---------------------------------------------------------------------------
# Frontier reference — where the ceiling is, for orientation only.
#
# Our scores are NOT comparable to published leaderboards: different n-shot
# conventions, different harness, sometimes a different metric. At our model
# scale that does not matter for orientation — a few points of protocol against
# a forty-point gap does not change what anyone concludes — but it would matter
# if one of our models ever came close, so every one of these carries its source
# and the date we read it, and the UI renders it as a reference line rather than
# a sortable column.
#
# Source data is CC-BY (Epoch AI), which we may use with attribution and do.
# Fill the rest in from https://epoch.ai/benchmarks — do NOT write a number here
# from memory: these move, and a remembered one goes stale silently.
# ---------------------------------------------------------------------------
_FRONTIER_SRC = "Epoch AI (CC-BY)"
_FRONTIER = {
    "mmlu": {"v": 0.88, "asof": "2026-09-15", "src": _FRONTIER_SRC},
}


def _task_facts(task: str, cells: dict, sig: dict) -> dict:
    """The per-task metadata the UI hangs its chips and reference lines off."""
    chance = _CHANCE.get(task)
    domain, desc = _TASK_META.get(task, (None, None))
    if desc is None and task.startswith("ppl_"):
        domain, desc = "language modelling", (
            "Perplexity on a pinned corpus slice — the same tokens for every "
            "model, so it measures modelling quality directly. Lower is better, "
            "there is no chance level, and unlike the multiple-choice tasks it "
            "keeps separating models that all sit at chance elsewhere.")
    rows = sig.get(task, [])
    return {
        "desc": desc,
        "domain": domain,
        # a task has a chance level precisely BECAUSE it is multiple-choice, so
        # the option count is derivable rather than a hand-kept flag
        "options": (round(1 / chance) if chance and 0 < chance < 1 else None),
        "nmodels": len(cells.get(task, {})),
        # does this task separate ANY pair of the models we actually have? A task
        # where every pair is within noise is telling us nothing, and right now it
        # looks exactly like one that works.
        "discriminates": (any(r[4] for r in rows) if rows else None),
        "frontier": _FRONTIER.get(task),
        "control": task in CONTROL_TASKS,
    }

# ---------------------------------------------------------------------------
# The protocol: which tasks an OFFICIAL average requires.
#
# Averaging whatever tasks a model happened to finish is not a ranking — a
# 2-task quick run scored 59% outranks an 8-task model at 45% while measuring
# something else entirely. So the average is computed over one fixed list, a
# model that is missing any of it is 'preliminary' (per-task results only, no
# overall rank), and completion is shown next to every official number.
#
# gsm8k is deliberately NOT in the list: it is generative and sits at ~0% for
# everything under ~1B, so it contributes noise rather than signal to a mean.
# It is still run, still reported, still charted. REQUIRED_TASKS overrides.
# ---------------------------------------------------------------------------
_REQUIRED_DEFAULT = ["mmlu", "hellaswag", "arc_challenge", "arc_easy",
                     "winogrande", "piqa", "truthfulqa_mc2"]


def required_tasks(acc_tasks: list[str]) -> tuple[list[str], list[str]]:
    """(required, absent) — the protocol list intersected with what this results
    tree actually contains, plus the protocol tasks nobody has run. Intersecting
    keeps a report built from a narrow tree honest instead of calling every
    model preliminary against tasks that were never attempted; `absent` drives a
    warning so a narrow list is never mistaken for the full protocol."""
    env = os.environ.get("REQUIRED_TASKS", "").strip()
    want = ([t.strip() for t in env.split(",") if t.strip()] if env
            else list(_REQUIRED_DEFAULT))
    want = [t for t in want if t not in CONTROL_TASKS]      # never, whatever the env says
    return [t for t in want if t in acc_tasks], [t for t in want if t not in acc_tasks]


def avg_se(tasks: list[str], cells: dict[str, dict], scaled: bool) -> float | None:
    """The standard error of a mean over `tasks`, from each task's own: the
    tasks are separate item sets, so the variances add and the mean divides
    by k. Scaled above chance, each error scales with 1/(1 − chance) as the
    score does. None when any task carries no error (a perplexity, a smoke
    run): an average of those is not z-testable, and the page says so."""
    ses = []
    for t in tasks:
        se = (cells.get(t) or {}).get("se")
        if se is None:
            return None
        c = _CHANCE.get(t)
        ses.append(se / (1 - c) if scaled and c is not None and 0 < c < 1 else se)
    return math.sqrt(sum(x * x for x in ses)) / len(ses) if ses else None


def above_chance(task: str, v: float) -> float:
    """Accuracy rescaled so 0 = chance and 1 = perfect. Raw accuracy is not
    comparable across tasks with different guess rates: 50% on a 2-option task
    (Winogrande, PIQA) is nothing, 50% on a 4-option one is real. Averaging raw
    numbers silently rewards whoever ran the easier-to-guess tasks."""
    c = _CHANCE.get(task)
    if c is None or not (0 < c < 1):
        return v
    return max(0.0, (v - c) / (1 - c))

# Proportion metrics: the only ones the two-proportion z-test is valid for.
PROPORTION = {"acc", "acc_norm", "exact_match", "pass@1", "f1", "em", "rubric_pass"}

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)([mb])(?![a-z0-9])", re.I)


def params_from_name(model_id: str) -> float | None:
    """pythia-410m -> 410e6, Qwen3-0.6B -> 6e8. Last size-looking token wins."""
    hits = _PARAM_RE.findall(model_id.split("/")[-1].lower())
    if not hits:
        return None
    v, unit = hits[-1]
    return float(v) * (1e6 if unit == "m" else 1e9)


def merge_runs(runs: list[dict]) -> dict[str, dict]:
    """Union per-task result files into one record per model.

    run_benchmarks.sh makes one lm_eval invocation per (model, task), so one model's
    results arrive as several JSONs — "last file wins" would show one task per model.
    A re-run of the same task still wins by date order.
    """
    by_model: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: str(r.get("date") or "")):
        m = by_model.get(r["model"])
        if m is None:
            by_model[r["model"]] = r
            continue
        m["tasks"].update(r["tasks"])
        m["n_shot"].update(r["n_shot"])
        m["n_samples"].update(r["n_samples"])
        m["subtasks"] |= r["subtasks"]
        m["higher_is_better"].update(r["higher_is_better"])
        m["eval_seconds"] = (m["eval_seconds"] or 0) + (r["eval_seconds"] or 0)
        m["date"] = r["date"]
        m["chat_template"] = m["chat_template"] or r["chat_template"]
        m["limit"] = m["limit"] or r["limit"]
        m["num_params"] = m["num_params"] or r["num_params"]
        m["archinfo"] = m.get("archinfo") or r.get("archinfo")
        m["diag"] = m.get("diag") or r.get("diag")
        m["judge"] = m.get("judge") or r.get("judge")
        m["judge_mtime"] = m.get("judge_mtime") or r.get("judge_mtime")
    return by_model


def build_payload(by_model: dict[str, dict], title: str, source: str,
                  taint: dict[str, list[str]] | None = None,
                  calibration: dict | None = None,
                  parents: dict[str, str] | None = None,
                  judge_identity: dict | None = None,
                  fingerprints: dict[str, str] | None = None) -> dict:
    """`taint`: model id -> tasks whose diagnostics its training data was
    derived from (the service computes it from the run/dataset join). A
    tainted task is treated exactly like a missing required task: shown per
    task, excluded from the official average, the model unranked.

    `fingerprints`: task -> the question set the exam holds now
    (exam_build.current_fingerprints). A judged topic graded on any other
    set — or before sets were fingerprinted — is left out of every number
    here and handed to the model page as `judge.history`. None when the exam
    directory is not known (a frozen report built from a results tree
    alone): then nothing is filtered."""
    models = list(by_model)
    taint = taint or {}
    parents = parents or {}      # tainted model id -> the model its training run started from
    current_judge = (judge_identity or {}).get("id") or None
    cal = calibration if isinstance(calibration, dict) and calibration.get("kappa") is not None \
        else None
    if cal is not None:
        cal = {**{k: cal.get(k) for k in ("kappa", "n", "calibrated", "kappa_min", "per_category")},
               "judge_id": (cal.get("judge") or {}).get("id")}

    # display names: short unless two orgs publish the same repo name
    # (google/gemma-3-270m vs unsloth/gemma-3-270m must not collapse into one row)
    shorts: dict[str, list[str]] = {}
    for m in models:
        shorts.setdefault(m.split("/")[-1], []).append(m)
    display = {m: (m if len(shorts[m.split("/")[-1]]) > 1 else m.split("/")[-1])
               for m in models}

    # headline metric per (task, model)
    cells: dict[str, dict[str, dict]] = {}
    metric_used: dict[str, str] = {}
    all_tasks: list[str] = []
    child_tasks: set[str] = set()
    for r in by_model.values():
        child_tasks |= r.get("subtasks", set())
    for mid, run in by_model.items():
        for task, entry in run["tasks"].items():
            pm = primary_metric(entry)
            if pm is None:
                continue
            name, v, s = pm
            cells.setdefault(task, {})[mid] = {
                "v": v, "se": s or 0.0,
                "shots": run["n_shot"].get(task),
                "n": run["n_samples"].get(task),
            }
            metric_used.setdefault(task, name)
            if task not in all_tasks:
                all_tasks.append(task)

    # headline tasks: groups + standalones (children live in the Runs tab), canonical
    # order first. A task missing for some models still shows — the dashboard renders
    # the gap honestly instead of hiding the task.
    headline = [t for t in all_tasks if t not in child_tasks]
    headline.sort(key=lambda t: (_CANON.index(t) if t in _CANON else 99, t))

    def is_lower_better(t: str) -> bool:
        met = metric_used.get(t, "")
        return any(r["higher_is_better"].get(t, {}).get(met) is False
                   for r in by_model.values())

    # judged free-response tasks never have a harness metric (bypass) and so
    # never a cell; the guard keeps a future metric from putting them here
    headline = [t for t in headline if not t.startswith(("fr_", "exam_"))]
    acc_tasks = [t for t in headline
                 if metric_used.get(t) in PROPORTION and not is_lower_better(t)]
    ppl_tasks = [t for t in headline if t not in acc_tasks]

    # official vs preliminary: the average is over the REQUIRED list or it does
    # not exist. A model missing any required task gets no overall number.
    required, req_absent = required_tasks(acc_tasks)
    model_rows = []
    for mid, r in by_model.items():
        # every task whose diagnostics this model's training data came from —
        # a multiple-choice benchmark, an exam topic, or both. The two are
        # treated the same way (shown, never ranked) in their own averages.
        tainted = sorted(set(taint.get(mid, ())))
        tainted_acc = [t for t in tainted if t in acc_tasks]
        have = [cells[t][mid]["v"] for t in acc_tasks
                if mid in cells.get(t, {}) and t not in CONTROL_TASKS and t not in tainted_acc]
        got_req = [t for t in required if mid in cells.get(t, {}) and t not in tainted_acc]
        missing = [t for t in required if mid not in cells.get(t, {})]
        official = bool(required) and not missing and not (set(tainted_acc) & set(required))
        params = r["num_params"] or params_from_name(mid)
        judge = _judged_now(r, fingerprints)
        jstate = judged_state(judge, cal, current_judge) if judge else None
        diag = _trim_diag(r.get("diag"))
        compare = {}
        for t in tainted:
            if not t.startswith("exam_"):
                continue
            pid = parents.get(mid)
            pjudge = _judged_now(by_model[pid], fingerprints) if pid in by_model else None
            cmp = taint_exam_compare(t, judge, pjudge, pid) if judge and pjudge else None
            if cmp:
                compare[t] = cmp
            elif not pid:
                compare[t] = {"parent": None, "scale": "rubric", "missing":
                              "the training run recorded no parent (bench.init(parent=…) or "
                              "base_model in its config), so there is no before to compare"}
            elif pid not in by_model:
                compare[t] = {"parent": pid, "scale": "rubric", "missing":
                              f"the parent {pid} is not on this board, so there is no before to "
                              f"compare — evaluate it with suite=judged"}
            else:
                compare[t] = {"parent": pid, "scale": "rubric", "missing":
                              f"{'this model' if not judge else 'the parent'} has no judged "
                              f"answers on file for this topic — submit it with suite=judged"}
        for t in tainted_acc:
            pid = parents.get(mid)
            pdiag = _trim_diag(by_model[pid].get("diag")) if pid in by_model else None
            cmp = taint_compare(t, diag, pdiag, pid) if diag and pdiag else None
            if cmp:
                compare[t] = cmp
            elif not pid:
                compare[t] = {"parent": None, "missing":
                              "the training run recorded no parent (bench.init(parent=…) or "
                              "base_model in its config), so there is no before to compare"}
            elif pid not in by_model:
                compare[t] = {"parent": pid, "missing":
                              f"the parent {pid} is not on this board, so there is no before to "
                              f"compare — evaluate it"}
            else:
                compare[t] = {"parent": pid, "missing":
                              f"{'this model' if not diag else 'the parent'} has no diagnosis "
                              f"on file for {t} — run scripts/diagnose.py"}
        if diag:
            for task, t in diag["tasks"].items():
                if t.get("categories"):
                    # MMLU's own gate stays, but it no longer offers anything:
                    # its findings are the caution beside an exam topic
                    t["propose"] = proposal_gate(task, t, cells.get(task, {}).get(mid),
                                                 _CHANCE.get(task))
        if judge:
            mmlu_cats = (((diag or {}).get("tasks") or {}).get("mmlu") or {}).get("propose") or {}
            for task, t in judge["tasks"].items():
                if not task.startswith("exam_"):
                    continue
                topic = EXAM_TOPICS.get(task)
                mc = (mmlu_cats.get("categories") or {}).get(topic) or {}
                t["propose"] = topic_gate(task, t, jstate, mc.get("why"), judge.get("judge"))
        model_rows.append({
            "id": mid, "name": display[mid],
            "family": re.split(r"[^a-z0-9]", mid.split("/")[-1].lower())[0],
            # uploaded checkpoints are experiment points, not reference models —
            # the dashboard separates the two so sweeps don't drown the ladder
            "source": "artifact" if mid.startswith("local/") else "hub",
            "kind": "instruct" if r["chat_template"] else "base",
            "params": params,
            "paramsSrc": ("config" if r["num_params"] else
                          "name" if params is not None else None),
            "backend": r["backend"], "dtype": r["dtype"],
            "batch": r["batch_size"], "chat": r["chat_template"],
            "seed": r["seed"], "limit": r["limit"],
            "minutes": round((r["eval_seconds"] or 0) / 60, 1),
            "hash": r["git_hash"], "date": r["date"],
            "archinfo": r.get("archinfo"),
            "diag": diag,
            # tasks whose diagnostics this model's training data was derived
            # from: shown, badged, and out of the average
            "tainted": tainted,
            # rubric scores from the pinned judge, summarised; the separate
            # judged average exists only when every category was judged
            "judge": judge,
            "judgedAvg": judged_avg(judge, tainted),
            # may these judged numbers be ranked? uncalibrated, a different
            # judge, or a moved canary all say no, in words
            "judgeState": jstate,
            # per tainted task: both halves before and after training, and
            # the sentence derived from them
            "taintCompare": compare or None,
            "kindReason": (r.get("archinfo") or {}).get("kind_reason"),
            # official numbers only: normalized (the ranking key) and raw (the
            # number you quote), both over the required list, both None when the
            # model has not completed it
            "avg": (sum(above_chance(t, cells[t][mid]["v"]) for t in required)
                    / len(required)) if official else None,
            "avgRaw": (sum(cells[t][mid]["v"] for t in required)
                       / len(required)) if official else None,
            # and their standard errors, carried through the mean so two
            # averages can be z-tested like any two scores (11c's frontier)
            "avgSe": avg_se(required, {t: cells[t][mid] for t in required}, True)
            if official else None,
            "avgRawSe": avg_se(required, {t: cells[t][mid] for t in required}, False)
            if official else None,
            # the same two over whatever it DID run — a diagnostic, never a rank
            "partialAvg": (sum(have) / len(have)) if have else None,
            "official": official,
            "nreq": len(required),
            "nhave": len(got_req),
            "missing": missing,
            "navg": len(have),
        })

    # pairwise significance per accuracy task, model ids in payload order
    sig: dict[str, list] = {}
    for t in acc_tasks:
        rows = []
        present = [m for m in models if m in cells.get(t, {})]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                ca, cb = cells[t][a], cells[t][b]
                ok, z = significant(ca["v"], ca["se"], cb["v"], cb["se"])
                rows.append([a, b, round(ca["v"] - cb["v"], 6),
                             (round(z, 3) if math.isfinite(z) else None), bool(ok)])
        sig[t] = rows

    # every metric incl. sub-tasks — the full table and the CSV export
    extra = []
    for mid, r in by_model.items():
        for task in sorted(r["tasks"]):
            for name, d in sorted(r["tasks"][task].items()):
                if not isinstance(d, dict) or "value" not in d:
                    continue
                extra.append([display[mid], task, name, round(d["value"], 6),
                              (round(d["stderr"], 6) if d.get("stderr") else None),
                              r["n_shot"].get(task), r["n_samples"].get(task)])
            # derived: cross-entropy loss in nats/byte (= bits_per_byte * ln 2) —
            # the same quantity as a training loss, per byte instead of per token,
            # so it is comparable across tokenizers and against training curves
            bpb = r["tasks"][task].get("bits_per_byte")
            if isinstance(bpb, dict) and "value" in bpb:
                extra.append([display[mid], task, "cross_entropy_nats_per_byte",
                              round(bpb["value"] * math.log(2), 6), None,
                              r["n_shot"].get(task), r["n_samples"].get(task)])

    # THE SAME RUN, SUBMITTED TWICE. The test is the ranked tasks, because
    # that is what a rank is computed from, and it is "the same answers", not
    # "the same floats": the live pair that sat at #1 and #2 for weeks is
    # bit-identical on six of seven required tasks and differs on the seventh
    # by ONE item in 10,042 — a re-run of the same weights, not a second
    # model. Two different models agreeing within one item on every ranked
    # task, mmlu's 14,042 included, does not happen.
    #
    # Both rows stay on the board — deleting someone's submission is not the
    # page's call — but only one is ranked, and the other says which it
    # duplicates. The keeper is the more complete run: same answers, more
    # tasks finished.
    DUP_MIN_TASKS = 3
    dup_tasks = required or acc_tasks

    def same_run(a: str, b: str) -> str | None:
        """'' when the rows are bit-identical, a sentence when they agree to
        within one item, None when they are two different runs."""
        worst = ""
        for t in dup_tasks:
            ca, cb = cells[t][a], cells[t][b]
            if ca.get("n") != cb.get("n"):
                return None                      # a different number of questions
            d = abs(ca["v"] - cb["v"])
            if d == 0:
                continue
            n = ca.get("n") or 0
            if not n or d > 1.0 / n + 1e-12:     # more than one item apart
                return None
            worst = (f"{t} differs by one item in {n}"
                     if not worst else f"{worst}, and {t} by one in {n}")
        return worst

    ranked_rows = [r for r in model_rows
                   if len(dup_tasks) >= DUP_MIN_TASKS
                   and all(r["id"] in cells.get(t, {}) for t in dup_tasks)]
    n_cells = {r["id"]: sum(1 for t in cells if r["id"] in cells[t]) for r in model_rows}
    # most tasks finished first; then the earlier run, then the id, so the
    # keeper is the same on every rebuild
    ranked_rows.sort(key=lambda r: (-n_cells[r["id"]], str(r.get("date") or ""), r["id"]))
    for i, keep in enumerate(ranked_rows):
        if keep.get("duplicateOf"):
            continue
        for other in ranked_rows[i + 1:]:
            if other.get("duplicateOf"):
                continue
            why = same_run(keep["id"], other["id"])
            if why is None:
                continue
            other["duplicateOf"] = keep["id"]
            other["duplicateOfName"] = keep["name"]
            other["duplicateWhy"] = why or "every score and item count is identical"

    # Two rows that are NOT the same run and still print the same numbers:
    # every ranked score within a tenth of a point. Not merged — that is a
    # claim about identity and these are two runs — but named on the row and
    # in a warning, because "why are these both here?" is a question the
    # board should answer rather than leave to a diff.
    NEAR = 1e-3
    for i, keep in enumerate(ranked_rows):
        for other in ranked_rows[i + 1:]:
            if other.get("duplicateOf") or keep.get("duplicateOf") \
                    or other.get("nearDuplicateOf"):
                continue
            gaps, counts = [], []
            for t in dup_tasks:
                ca, cb = cells[t][keep["id"]], cells[t][other["id"]]
                if ca.get("n") != cb.get("n"):
                    # a different number of questions: not the same run, and
                    # the most useful thing to say about the pair
                    counts.append(f"{t}: {ca.get('n')} items against {cb.get('n')}")
                    continue
                gaps.append((abs(ca["v"] - cb["v"]),
                             f"{t}: {ca['v']:.6f} against {cb['v']:.6f}"))
            if gaps and max(g[0] for g in gaps) <= NEAR:
                other["nearDuplicateOf"] = keep["id"]
                other["nearDuplicateOfName"] = keep["name"]
                other["nearDuplicateWhy"] = counts[0] if counts else max(gaps)[1]

    # provenance warnings — they gate every claim below them. Each is also a
    # check: a key, a one-line summary, a severity, and where on the page the
    # rows it concerns are ("Show me"); the long text stays behind a disclosure
    warnings: list[str] = []
    checks: list[dict] = []

    def warn(key: str, severity: str, show: dict, short: str, text: str) -> None:
        warnings.append(text)
        checks.append({"key": key, "severity": severity, "show": show, "short": short,
                       "text": text, "judged": key.startswith("judge_")})
    shots_seen: dict[str, set] = {}
    for r in by_model.values():
        for t in headline:
            if t in r["n_shot"]:
                shots_seen.setdefault(t, set()).add(r["n_shot"][t])
    mism = [t for t, s in shots_seen.items() if len(s) > 1]
    if mism:
        warn('fewshot', 'warning', {'tab': 'leaderboard'}, 'Few-shot count differs between models',
            f"Few-shot count differs between models on: {', '.join(mism)}. Those "
            f"columns are not comparable — re-run with the same --num_fewshot.")
    if len({r["chat_template"] for r in by_model.values()}) > 1:
        applied = [display[m] for m, r in by_model.items() if r["chat_template"]]
        warn('chat_mixed', 'info', {'tab': 'models', 'kind': 'instruct'}, 'Chat template on some models, not others',
            "Chat template applied to some models but not others (applied to: "
            + ", ".join(applied) + "). Correct if and only if those are the instruct "
            "models — it moves scores by tens of points, so check the list.")
    if any(r["limit"] for r in by_model.values()):
        warn('limit', 'warning', {'tab': 'provenance'}, 'A run used --limit: smoke data, not a reported number',
            "At least one run used --limit, so it did not see the full task. Fine "
            "for a smoke test, not for a reported number.")
    # a template that EXISTS but differs between models applied the same way is
    # also a comparability break — the hash is what has to match, not the yes/no
    shas = {(r.get("archinfo") or {}).get("tmpl_sha")
            for m, r in by_model.items() if r["chat_template"]}
    shas.discard(None)
    if len(shas) > 1:
        warn('templates', 'info', {'tab': 'provenance'}, 'Models evaluated with different chat templates',
            f"The models evaluated WITH a chat template used {len(shas)} different "
            f"templates ({', '.join(sorted(shas))}). Prompt format differs, so "
            f"those scores answer slightly different questions.")
    unconf = [display[m] for m, r in by_model.items()
              if (r.get("archinfo") or {}).get("kind_unconfirmed")]
    if unconf:
        warn('kind_unconfirmed', 'warning', {'tab': 'models', 'kind': 'instruct'}, 'Chat template applied on detection alone',
            "Chat template applied on detection alone to: " + ", ".join(unconf)
            + ". Those repos ship a template but their names do not say "
            "instruct, so nothing corroborates the choice — if any of them is a "
            "pretrained checkpoint, resubmit it with kind=base.")
    if req_absent:
        warn('required_narrow', 'info', {'tab': 'leaderboard'}, 'The required-task list is narrower than the protocol',
            "This report's required-task list is narrower than the protocol: "
            + ", ".join(req_absent) + " were not run by anyone here, so 'official' "
            "means complete within this report, not complete under the full "
            "protocol.")
    alarms = [m for m in model_rows
              if any(c.get("verdict") == "test" for c in (m["taintCompare"] or {}).values())]
    if alarms:
        warn('taught_test', 'warning', {'tab': 'models', 'tainted': True}, f"{len(alarms)} trained model{'s' if len(alarms) > 1 else ''} learned the test, not the skill",
            f"{len(alarms)} model{'s' if len(alarms) > 1 else ''} "
            f"({', '.join(m['name'] for m in alarms[:4])}) moved on the diagnosis half and not "
            f"on the leaderboard half after training on derived data — the training taught the "
            f"test, not the skill. This is the alarm the split exists to raise; see the model's "
            f"page.")
    tainted_rows = [m for m in model_rows if m["tainted"]]
    if tainted_rows:
        warn('tainted', 'info', {'tab': 'models', 'tainted': True}, f"{len(tainted_rows)} model{'s' if len(tainted_rows) > 1 else ''} trained on data derived from diagnostics",
            f"{len(tainted_rows)} model{'s' if len(tainted_rows) > 1 else ''} "
            f"({', '.join(m['name'] for m in tainted_rows[:4])}"
            f"{', …' if len(tainted_rows) > 4 else ''}) trained on data derived from "
            f"benchmark diagnostics. The affected task is shown per model, badged, and "
            f"excluded from that model's official average — it is not a ranking claim.")
    dups = [m for m in model_rows if m.get("duplicateOf")]
    if dups:
        warn('duplicates', 'info', {'tab': 'leaderboard'}, f"{len(dups)} duplicate row{'s' if len(dups) > 1 else ''}: the same run submitted twice",
            f"{len(dups)} row{'s' if len(dups) > 1 else ''} "
            f"({', '.join(m['name'] for m in dups[:4])}{', …' if len(dups) > 4 else ''}) "
            f"give the same answers as another row on every ranked task — the same run "
            f"submitted twice ("
            + "; ".join(f"{m['name']}: {m['duplicateWhy']}" for m in dups[:2])
            + "). Both are shown; the less complete one is not ranked and says which row it "
              "duplicates. Nothing has been deleted.")
    near = [m for m in model_rows if m.get("nearDuplicateOf")]
    if near:
        warn('near_duplicates', 'info', {'tab': 'leaderboard'}, f"{len(near)} row{'s' if len(near) > 1 else ''} print the same scores as another, and are not the same run",
            f"{len(near)} row{'s' if len(near) > 1 else ''} print the same score as another "
            f"row on every ranked task and are NOT the same run: "
            + "; ".join(f"{m['name']} vs {m['nearDuplicateOfName']} differ on "
                        f"{m['nearDuplicateWhy']}" for m in near[:3])
            + (", …" if len(near) > 3 else "")
            + ". Both are ranked, because identical is the test for a duplicate and these "
              "are not identical — but two runs this close are worth a look.")
    n_prelim = sum(1 for m in model_rows if not m["official"])
    if n_prelim and required:
        warn('preliminary', 'info', {'tab': 'models', 'prelim': True}, f"{n_prelim} of {len(model_rows)} models are preliminary",
            f"{n_prelim} of {len(model_rows)} models are preliminary (they have not "
            f"finished all {len(required)} required tasks) and carry no overall "
            f"average or rank. Their per-task numbers are shown everywhere and are "
            f"valid on their own — resubmit with suite=full to make them official.")
    hashes = sorted({r["git_hash"] for r in by_model.values() if r["git_hash"]})
    if len(hashes) > 1:
        warn('harness_builds', 'warning', {'tab': 'provenance'}, 'Results come from several harness builds',
            f"Results come from {len(hashes)} different harness builds "
            f"({', '.join(hashes)}). A benchmark whose code changed is a different "
            f"benchmark — treat cross-build comparisons with suspicion.")

    # the topics judged on the question set they hold now — a topic sat only
    # on a retired set is history on the model page, not a column here
    judged_tasks = sorted({t for m in model_rows
                           for t in ((m.get("judge") or {}).get("tasks") or {})})
    judge_meta = next(((m["judge"] or {}).get("judge") for m in model_rows if m.get("judge")), None)
    if judged_tasks and cal is None:
        warn('judge_uncalibrated', 'warning', {'tab': 'loop'}, 'The judge is not calibrated against a person',
            "Judged free-response scores are on file but the judge has not been calibrated "
            "against a person (scripts/judge_calibrate.py). They are shown as preliminary and "
            "enter no average and no rank.")
    elif cal and not cal.get("calibrated"):
        warn('judge_kappa', 'warning', {'tab': 'loop'}, "The judge's agreement with a person is below the line",
            f"The judge's agreement with a human grader is Cohen's kappa {cal['kappa']} over "
            f"{cal.get('n')} answers, below the {KAPPA_MIN} line. Judged scores are preliminary: "
            f"shown, never ranked, never averaged.")
    if judge_meta and judge_meta.get("stub"):
        warn('judge_stub', 'warning', {'tab': 'loop'}, 'Judged scores come from the stub grader',
            "Judged scores on this board come from the STUB grader (a word-overlap stand-in "
            "used for plumbing tests). They are not judgements of anything.")
    drifted = [m["name"] for m in model_rows if m.get("judge")
               and (m["judge"].get("canary") or {}).get("drifted")]
    if drifted:
        warn('judge_canary', 'warning', {'tab': 'loop'}, "The judge's canary moved",
            f"The judge's canary moved on {len(drifted)} run{'s' if len(drifted) > 1 else ''} "
            f"({', '.join(drifted[:4])}): the same thirty scripts were graded differently from "
            f"the previous run. Those judged scores are preliminary — a vendor may have changed "
            f"the model behind the id.")
    # 11l: answers that never left a reasoning block. Run #60 was found by a
    # person reading one answer; this class of failure is a check now
    unfinished = []
    for m in model_rows:
        ts = ((m.get("judge") or {}).get("tasks") or {}).values()
        gone = sum((t.get("no_answer") or 0) for t in ts if isinstance(t, dict))
        if gone:
            total = sum((t.get("n") or 0) for t in ts if isinstance(t, dict))
            unfinished.append((m, gone, total))
    if unfinished:
        unfinished.sort(key=lambda x: -x[1])
        named = [f"{m['name']} ({gone:,} of {total:,})" for m, gone, total in unfinished]
        warn('judge_unfinished', 'warning', {'model': unfinished[0][0]["id"]},
            f"Answers that never finished: {named[0]}"
            + (f" and {len(named) - 1} more" if len(named) > 1 else ""),
            f"{'; '.join(named[:4])}{', …' if len(named) > 4 else ''}: answers that stopped "
            f"inside the model's reasoning before it answered. They are not scored — counted as "
            f"no answer and left out of every mean — and a topic where most answers never "
            f"finished has no score at all. Sit the exam again: a reasoning model now gets "
            f"room to answer.")
    local_judged = [m["name"] for m in model_rows if provisional_reason(m.get("judge"))]
    if local_judged:
        warn('judge_local', 'info', {'tab': 'loop'}, f"{len(local_judged)} model{'s' if len(local_judged) > 1 else ''} graded by a local judge: provisional",
            f"Judged scores for {len(local_judged)} model{'s' if len(local_judged) > 1 else ''} "
            f"({', '.join(local_judged[:4])}{', …' if len(local_judged) > 4 else ''}) were "
            f"graded by a local model — not a pinned benchmark. They are provisional: shown "
            f"greyed on the model page, never ranked, never in any average.")
    if any((m.get("judge") or {}).get("judge", {}).get("single_provider_loop") for m in model_rows):
        warn('judge_single_provider', 'info', {'tab': 'loop'}, 'Single-provider loop',
            "Single-provider loop: the judge shares a provider with the exam writer or the "
            "generator (ALLOW_SINGLE_PROVIDER_LOOP). Every judged score carries that caveat; "
            "self-preference in LLM judges is documented and large.")
    other_judges = sorted({(m["judge"]["judge"] or {}).get("id") for m in model_rows
                           if m.get("judge")} - {current_judge, None})
    if current_judge and other_judges:
        warn('judge_other', 'warning', {'tab': 'loop'}, 'Some judged scores come from a different judge',
            f"Some judged scores come from a different judge ({', '.join(other_judges)}) than "
            f"the one this server runs now ({current_judge}). They are shown as their own "
            f"series and never ranked against the current judge's — resubmit with suite=judged "
            f"to re-grade.")

    dates = sorted(str(r["date"]) for r in by_model.values() if r["date"])
    return {
        "title": title,
        "generated": _dt.datetime.now(_TZ).strftime("%Y-%m-%d %H:%M %Z").strip(),
        "source": source,
        "models": model_rows,
        "accTasks": acc_tasks,
        "pplTasks": ppl_tasks,
        "required": required,          # the protocol list an official average needs
        "reqAbsent": req_absent,
        "tasks": {t: {"metric": metric_used.get(t, ""),
                      "lower": is_lower_better(t),
                      "chance": _CHANCE.get(t),
                      **_task_facts(t, cells, sig)} for t in headline},
        "cells": cells,
        "sig": sig,
        # the judged suite: which tasks anyone ran, the judge, and whether a
        # person has agreed with it enough for the numbers to count
        "judged": {"tasks": judged_tasks, "exam": EXAM_TASKS, "topics": EXAM_TOPICS,
                   "control": FR_CONTROL, "judge": judge_meta, "kappaMin": KAPPA_MIN,
                   # the judge this server runs now; files from other judges are their own series
                   "current": judge_identity or None,
                   "calibration": cal},
        "extra": extra,
        "warnings": warnings,
        "checks": checks,
        "meta": {
            "hashes": hashes,
            "dates": [dates[0][:16] if dates else None,
                      dates[-1][:16] if dates else None],
            "hours": round(sum(r["eval_seconds"] or 0
                               for r in by_model.values()) / 3600, 2),
            "transformers": next((r["transformers_version"]
                                  for r in by_model.values()
                                  if r["transformers_version"]), None),
            "anyLimit": any(r["limit"] for r in by_model.values()),
            # whether per-item diagnosis exists for anyone. When it does, a model
            # without it says so instead of silently dropping the section — an
            # absent diagnosis is a fact about the run, not a reason to hide it.
            "anyDiag": any(r.get("diag") for r in by_model.values()),
            # the category order the page lays MMLU out in (scripts/categories.yaml)
            "categories": _categories.category_order(),
            # the 37 topics in their 8 areas (scripts/areas.yaml): the Leaderboard's
            # MMLU-by-area and judged-by-area columns, and the model page's groups
            "areas": _categories.areas(),
            "diagSalt": next((r["diag"].get("split_salt")
                              for r in by_model.values() if r.get("diag")), None),
        },
    }


# ---------------------------------------------------------------------------
# the page. One file: tokens -> CSS -> JS -> template.
# Palette: the dataviz reference palette (validated for CVD + contrast in both
# modes; see docs). Charts are drawn client-side from the embedded JSON.
# ---------------------------------------------------------------------------

CSS = r"""
:root { color-scheme: light; }
.viz-root {
  /* phase 9d: one visual system. Spacing 4/8/12/16/24/32, type 12/14/16/20/28,
     radii 6/10, one border colour (--border) and one surface (--surface-1) */
  --sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:16px; --sp-5:24px; --sp-6:32px;
  --fs-1:12px; --fs-2:14px; --fs-3:16px; --fs-4:20px; --fs-5:28px;
  --r-1:6px; --r-2:10px;
  --bar-h:57px;                  /* the sticky bar and its hairline */
  /* 11f: one motion system. Hover and press; popovers, expanding and the tab
     underline; toasts and view changes. The opened row takes 200ms and its
     chevron 150ms. All of it is 0 under reduced motion. */
  --dur-1:120ms; --dur-2:180ms; --dur-3:240ms; --dur-row:200ms; --dur-chev:150ms;
  --ease:cubic-bezier(.2, .8, .2, 1);
  /* phase 11b: prose in the system sans, data and labels in the system mono.
     No web font: the page has to work on a server with no internet. */
  --font-sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  /* 11b: a pale blue-grey page, white cards, one saturated accent. Contrast
     against the page (#F6F8FC): ink 15.0:1, secondary 8.8:1, muted 5.1:1. */
  --surface-1:#ffffff; --plane:#f6f8fc; --text-primary:#14213d; --text-secondary:#3d4b66;
  --muted:#5a6b85; --grid:#e4e9f2; --axis:#c9d2e3; --border:#e4e9f2;
  --good:#0ca30c; --critical:#d03b3b; --warning:#fab219; --success-text:#006300;
  /* the warning and danger tones as TEXT: amber on white is 1.8:1, so text
     that must be read gets a darker amber; 4.5:1 or better on every surface */
  --warning-text:#8a5b00; --critical-text:#c23030;
  /* the accent is also the link colour: 5.9:1 on white, 5.5:1 on the page */
  --accent:#2f54eb; --accent-soft:rgba(47,84,235,0.08);
  /* live: the text tone is read, the dot tone is never text (2.6:1) */
  --live-text:#0b7a5a; --live-dot:#12b886;
  --bar-bg:rgba(246,248,252,0.86);
  /* the rank tint, five steps of the accent mixed into the card surface. Text
     on the strongest step is ink at 11.1:1 — never the accent, which is 4.1:1 */
  --heat-1:color-mix(in srgb, var(--accent) 3.5%, var(--surface-1));
  --heat-2:color-mix(in srgb, var(--accent) 7%, var(--surface-1));
  --heat-3:color-mix(in srgb, var(--accent) 11.5%, var(--surface-1));
  --heat-4:color-mix(in srgb, var(--accent) 17%, var(--surface-1));
  --heat-5:color-mix(in srgb, var(--accent) 24%, var(--surface-1));
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --success-text:#0ca30c; --accent:#7c9bff; --accent-soft:rgba(124,155,255,0.16);
    --live-text:#2fd39a; --live-dot:#12b886; --bar-bg:rgba(13,13,13,0.80);
    --heat-1:color-mix(in srgb, var(--accent) 6%, var(--surface-1));
    --heat-2:color-mix(in srgb, var(--accent) 11%, var(--surface-1));
    --heat-3:color-mix(in srgb, var(--accent) 17%, var(--surface-1));
    --heat-4:color-mix(in srgb, var(--accent) 25%, var(--surface-1));
    --heat-5:color-mix(in srgb, var(--accent) 34%, var(--surface-1));
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --warning-text:#fab219; --critical-text:#ef6b6b;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --success-text:#0ca30c; --accent:#7c9bff; --accent-soft:rgba(124,155,255,0.16);
  --live-text:#2fd39a; --live-dot:#12b886; --bar-bg:rgba(13,13,13,0.80);
  --heat-1:color-mix(in srgb, var(--accent) 6%, var(--surface-1));
  --heat-2:color-mix(in srgb, var(--accent) 11%, var(--surface-1));
  --heat-3:color-mix(in srgb, var(--accent) 17%, var(--surface-1));
  --heat-4:color-mix(in srgb, var(--accent) 25%, var(--surface-1));
  --heat-5:color-mix(in srgb, var(--accent) 34%, var(--surface-1));
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --warning-text:#fab219; --critical-text:#ef6b6b;
}
/* dim: slate blue-grey, softer than the near-black dark theme (the GitHub /
   wandb "dimmed" look). Same dark series palette — re-validated against both
   dim surfaces: all six checks pass. */
:root[data-theme="dim"] .viz-root {
  color-scheme: dark;
  --surface-1:#1c2333; --plane:#141a26; --text-primary:#e6edf3; --text-secondary:#b6c2d1;
  --muted:#8b98a8; --grid:#2b3546; --axis:#3a465a; --border:rgba(230,237,243,0.11);
  --success-text:#3fb950; --accent:#7c9bff; --accent-soft:rgba(124,155,255,0.16);
  --live-text:#2fd39a; --live-dot:#12b886; --bar-bg:rgba(20,26,38,0.82);
  --heat-1:color-mix(in srgb, var(--accent) 6%, var(--surface-1));
  --heat-2:color-mix(in srgb, var(--accent) 11%, var(--surface-1));
  --heat-3:color-mix(in srgb, var(--accent) 17%, var(--surface-1));
  --heat-4:color-mix(in srgb, var(--accent) 25%, var(--surface-1));
  --heat-5:color-mix(in srgb, var(--accent) 34%, var(--surface-1));
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --warning-text:#fab219; --critical-text:#ef6b6b;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--plane); color:var(--text-primary);
  font-family:var(--font-sans); font-size:var(--fs-2); line-height:1.5; }
.wrap { max-width:1320px; margin:0 auto; padding:0 22px 70px; }
/* 11k: whatever a card holds, the page itself never slides sideways. clip,
   not auto: it makes no scroller, so a sticky header still sticks to the
   page and a popover still lives on the body */
#view, .wrap, .card { overflow-x:clip; }
@media (max-width:720px) { .wrap { padding:0 16px 70px; } }
/* 11b: a heavy, tight title; the section title after its mono index */
h1 { font-size:var(--fs-5); font-weight:800; margin:0; letter-spacing:-0.025em; line-height:1.15; }
h2 { font-size:var(--fs-4); font-weight:700; margin:0 0 3px; letter-spacing:-0.01em; }
/* prose stays in the sans face and stops at a readable measure */
.sub, .note, .warn, p.small { max-width:72ch; }
.sub { color:var(--text-secondary); font-size:var(--fs-2); margin:2px 0 0; }
/* ---- 11b: the sticky bar ------------------------------------------------
   56px, translucent with a blur, a hairline under it. Title and status badge
   on the left, the tabs in the middle, the checks pill, the name and the
   theme on the right. At 720px the title shortens and the tabs drop to a
   second row inside the same sticky block. */
.bar { position:sticky; top:0; z-index:50; background:var(--bar-bg);
  backdrop-filter:blur(8px) saturate(1.4); -webkit-backdrop-filter:blur(8px) saturate(1.4);
  border-bottom:1px solid var(--border); }
.bar-in { position:relative; max-width:1320px; margin:0 auto; padding:0 22px;
  height:56px; display:flex; align-items:center; gap:14px; }
.bar-title { font-size:var(--fs-3); font-weight:800; letter-spacing:-0.02em;
  white-space:nowrap; }
.bar-title .t-short { display:none; }
.bar-right { margin-left:auto; display:flex; gap:8px; align-items:center; }
.livebadge { display:inline-flex; align-items:center; gap:6px; font-family:var(--font-mono);
  font-size:var(--fs-1); font-weight:600; text-transform:uppercase; letter-spacing:.04em;
  color:var(--live-text); border:1px solid var(--live-dot); border-radius:var(--r-1);
  padding:2px 8px; white-space:nowrap; }
.livebadge[hidden] { display:none; }
.livebadge[data-fresh="stale"], .livebadge[data-fresh="judge-offline"] {
  color:var(--warning-text); border-color:var(--warning); text-transform:none;
  letter-spacing:0; }
.livebadge .dot { animation:livepulse 2.4s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) { .livebadge .dot { animation:none; } }
/* 11f: under reduced motion every duration is 0 — nothing moves */
@media (prefers-reduced-motion: reduce) {
  .viz-root { --dur-1:0ms; --dur-2:0ms; --dur-3:0ms; --dur-row:0ms; --dur-chev:0ms; } }
/* buttons, chips, pills and tabs: their colours ease; a press gives a little */
button, .btn, .pill, .chip-btn { transition:background-color var(--dur-1) var(--ease),
  border-color var(--dur-1) var(--ease), color var(--dur-1) var(--ease),
  transform var(--dur-1) var(--ease); }
button:active:not(:disabled):not([aria-disabled="true"]) { transform:scale(.98); }
/* the tab underline: one element, moved by a transform */
.tabs { position:relative; }
.tab-ink { position:absolute; left:0; bottom:0; width:1px; height:2px; background:var(--accent);
  transform-origin:0 0; pointer-events:none;
  transition:transform var(--dur-2) var(--ease), opacity var(--dur-2) var(--ease); }
.tab-ink.still { transition:none; }
/* popovers come 4px from the side away from their button, and leave in half the time */
.pop { transition:opacity var(--dur-2) var(--ease), transform var(--dur-2) var(--ease); }
.pop.pop-in { opacity:0; transform:translateY(4px); }
.pop.flip.pop-in { transform:translateY(-4px); }
.pop.pop-out { opacity:0; transform:translateY(4px); pointer-events:none;
  transition-duration:calc(var(--dur-2) / 2); }
.pop.flip.pop-out { transform:translateY(-4px); }
/* 11g: the Reader — a sheet from the right, min(760px, 92vw); the whole
   screen below 720px. It slides in and fades with the 11f tokens */
body.reading { overflow:hidden; }
.reader-wrap { position:fixed; inset:0; z-index:90; }
.reader-scrim { position:absolute; inset:0; background:rgba(10,16,30,.28); opacity:0;
  transition:opacity var(--dur-3) var(--ease); }
.reader { position:absolute; top:0; right:0; bottom:0; width:min(760px, 92vw);
  display:flex; flex-direction:column; background:var(--surface-1);
  box-shadow:-12px 0 40px rgba(0,0,0,.18); border-left:1px solid var(--border);
  transform:translateX(24px); opacity:0;
  transition:transform var(--dur-3) var(--ease), opacity var(--dur-3) var(--ease); }
.reader-wrap.open .reader { transform:none; opacity:1; }
.reader-wrap.open .reader-scrim { opacity:1; }
.reader:focus { outline:none; }
@media (max-width:720px) { .reader { width:100vw; border-left:0; } }
.rd-head { display:flex; gap:12px; align-items:flex-start; justify-content:space-between;
  padding:14px 18px 12px; border-bottom:1px solid var(--border); }
.rd-titles { min-width:0; }
.rd-title { margin:0; font-size:var(--fs-4); overflow-wrap:anywhere; }
.rd-src { margin:4px 0 0; color:var(--text-secondary); font-size:var(--fs-2); overflow-wrap:anywhere; }
.rd-src .badge { margin-left:8px; }
.rd-top { display:flex; align-items:flex-start; gap:6px; flex:none; flex-wrap:wrap;
  justify-content:flex-end; }
.rd-acts { display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
.rd-acts > button, .rd-acts > a.btn { height:32px; min-height:32px; padding:0 10px; border-radius:6px;
  display:inline-flex; align-items:center; text-decoration:none; font-size:var(--fs-1); }
.rd-close { height:32px; }
@media (max-width:720px) { .rd-head { flex-direction:column; } .rd-top { align-items:flex-start;
  flex-direction:row-reverse; flex-wrap:wrap; } }
.rd-body { flex:1; min-height:0; overflow:auto; padding:14px 18px 28px; overscroll-behavior:contain; }
.rd-body:focus { outline:none; }
.rd-summary { margin-bottom:12px; }
.rd-summary p { margin:0 0 6px; }
.rd-search { width:100%; box-sizing:border-box; }
.rd-filters { margin:0 0 10px; }
.rd-filters .rd-search { flex:1; min-width:160px; width:auto; }
/* the dataset: the list on the left, the document on the right */
.rd-split { display:grid; grid-template-columns:minmax(0, 240px) minmax(0, 1fr); gap:16px; }
@media (max-width:720px) { .rd-split { grid-template-columns:minmax(0, 1fr); }
  .rd-list { max-height:220px; } }
.rd-side { display:flex; flex-direction:column; gap:8px; min-width:0; }
.rd-list { list-style:none; margin:0; padding:0; overflow:auto; max-height:calc(100vh - 260px);
  border:1px solid var(--border); border-radius:var(--r-1); }
.rd-item { display:grid; grid-template-columns:auto minmax(0, 1fr); gap:2px 8px; padding:7px 9px;
  border-bottom:1px solid var(--grid); cursor:pointer; font-size:var(--fs-2); }
.rd-item:last-child { border-bottom:0; }
.rd-item:hover, .rd-item:focus-visible { background:var(--plane); outline:none; }
.rd-item.on { background:var(--accent-soft); box-shadow:inset 3px 0 0 var(--accent); }
.rd-item .rd-n { grid-row:span 2; color:var(--muted); font-size:var(--fs-1); padding-top:2px; }
.rd-item .rd-t { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.rd-item .rd-m { font-size:var(--fs-1); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.rd-item.gone { cursor:default; color:var(--muted); background:none; }
.rd-item.gone .rd-t { font-style:italic; }
.rd-doc { min-width:0; }
.rd-doc-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.rd-doc-head h3 { margin:0; flex:1 1 260px; font-size:var(--fs-4); }
.rd-doc-head button { height:32px; min-height:32px; width:36px; padding:0; }
.rd-prose { font-family:var(--font-sans); font-size:var(--fs-3); line-height:1.6; max-width:68ch;
  color:var(--text-primary); }
.rd-prose p { margin:0 0 .9em; }
.rd-qa .eyebrow { margin-top:12px; }
mark { background:color-mix(in srgb, var(--warning) 35%, transparent); color:inherit;
  border-radius:2px; padding:0 1px; }
/* markdown, as text: headings, lists, tables, emphasis and code */
.md { font-size:var(--fs-2); line-height:1.6; max-width:78ch; }
.md .md-h { margin:1.1em 0 .4em; line-height:1.3; }
.md h3.md-h { font-size:var(--fs-4); } .md h4.md-h { font-size:var(--fs-3); }
.md h5.md-h, .md h6.md-h { font-size:var(--fs-2); }
.md code { font-family:var(--font-mono); font-size:.92em; background:var(--plane);
  padding:1px 4px; border-radius:4px; }
.md-pre { font-family:var(--font-mono); font-size:var(--fs-1); background:var(--plane);
  border:1px solid var(--border); border-radius:var(--r-1); padding:10px 12px; overflow:auto;
  white-space:pre-wrap; overflow-wrap:anywhere; }
.md blockquote { margin:0 0 1em; padding-left:12px; border-left:3px solid var(--border);
  color:var(--text-secondary); }
.md-table td, .md-table th { white-space:normal; height:auto; }
.rd-toc { border:1px solid var(--border); border-radius:var(--r-1); padding:8px 12px; margin-bottom:12px; }
.rd-toc > summary { cursor:pointer; }
.rd-kv-in { margin:0; font-size:var(--fs-2); }
.rd-toc ul { list-style:none; margin:4px 0 0; padding:0; font-size:var(--fs-1); }
.rd-toc li { margin:2px 0; }
.rd-kv { display:grid; grid-template-columns:auto minmax(0, 1fr); gap:4px 12px; margin:0 0 12px;
  font-size:var(--fs-2); }
.rd-kv dt { font-family:var(--font-mono); font-size:var(--fs-1); color:var(--muted); }
.rd-kv dd { margin:0; overflow-wrap:anywhere; }
.rd-flag { border:1px solid var(--border); border-radius:var(--r-1); padding:8px 12px; margin:8px 0; }
.rd-flag p { margin:0 0 4px; }
.rd-raw { margin-top:14px; }
/* the bank: one card a question */
.rd-q { border-bottom:1px solid var(--grid); padding:10px 0; }
.rd-q-meta { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:4px; }
/* the log: mono, numbered, the bad lines in the warning tone */
.rd-log { font-family:var(--font-mono); font-size:var(--fs-1); line-height:1.5; border:1px solid var(--border);
  border-radius:var(--r-1); background:var(--plane); max-height:calc(100vh - 230px); overflow:auto;
  padding:6px 0; }
.rd-ln { display:grid; grid-template-columns:4.5em minmax(0, 1fr); gap:10px; padding:0 10px;
  white-space:pre; }
.rd-log.wrap .rd-ln { white-space:pre-wrap; overflow-wrap:anywhere; }
.rd-ln.bad { background:color-mix(in srgb, var(--warning) 14%, transparent); }
.rd-ln.cur { outline:2px solid var(--accent); outline-offset:-2px; }
.rd-no { color:var(--muted); text-align:right; user-select:none; }
/* provenance: a tree */
.rd-tree { font-size:var(--fs-2); }
.rd-node > summary { cursor:pointer; padding:3px 0; }
.rd-kids { padding-left:16px; border-left:1px solid var(--grid); margin-left:4px; }
.rd-leaf { display:grid; grid-template-columns:minmax(120px, auto) minmax(0, 1fr); gap:10px;
  padding:2px 0; }
.rd-k { font-family:var(--font-mono); font-size:var(--fs-1); color:var(--muted); }
.rd-leaf > span:last-child { overflow-wrap:anywhere; }
.rd-hash { display:inline-flex; gap:6px; align-items:center; }
.rd-hash button { padding:0 4px; min-height:0; font-size:var(--fs-1); }
a.dllink { text-decoration:none; }
.actcell > a.dllink { font-size:var(--fs-1); }
/* 11f: Select and Combobox — 36px, radius 6, the border, the accent ring */
button.sel { display:inline-flex; align-items:center; justify-content:space-between; gap:10px;
  height:36px; min-height:36px; border-radius:6px; border:1px solid var(--border);
  background:var(--surface-1); padding:0 10px 0 12px; font-size:var(--fs-2); max-width:100%;
  text-align:left; white-space:nowrap; }
button.sel .sel-v { overflow:hidden; text-overflow:ellipsis; }
button.sel .sel-c { color:var(--muted); font-size:var(--fs-1); }
button.sel[aria-expanded="true"] { border-color:var(--accent); }
input.cbox { font:inherit; font-size:var(--fs-2); height:36px; box-sizing:border-box;
  border:1px solid var(--border); border-radius:6px; padding:0 12px; min-width:220px;
  background:var(--surface-1); color:var(--text-primary); }
input.cbox:focus { outline:2px solid var(--accent); outline-offset:1px; border-color:var(--accent); }
.listbox [role=option] { display:flex; align-items:center; gap:8px; padding:7px 10px;
  border-radius:var(--r-1); cursor:pointer; font-size:var(--fs-2); color:var(--text-primary); }
.listbox [role=option]:hover, .listbox [role=option]:focus, .listbox [role=option].active {
  background:var(--accent-soft); outline:none; }
.listbox [role=option][aria-selected="true"] { font-weight:600; }
.listbox [role=option][aria-selected="true"]::before { content:"✓"; color:var(--accent); width:12px; }
.listbox [role=option][aria-selected="false"]::before { content:""; width:12px; }
.listbox [role=option][aria-disabled="true"] { color:var(--muted); cursor:not-allowed; }
.listbox [role=option].has-sub { align-items:flex-start; }
.listbox .opt-t { display:flex; flex-direction:column; gap:2px; min-width:0; }
.listbox .opt-sub { font-weight:400; font-size:var(--fs-1); color:var(--text-secondary);
  white-space:normal; max-width:420px; }
.cblist { min-width:320px; max-width:min(440px, calc(100vw - 16px)); }
.cbgroup + .cbgroup { margin-top:4px; }
.cbhead { font-family:var(--font-mono); font-size:var(--fs-1); text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); padding:6px 10px 2px; }
.cb-t { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cb-r { font-family:var(--font-mono); font-size:var(--fs-1); color:var(--text-secondary); }
.cb-b { width:36px; height:4px; background:var(--grid); border-radius:2px; overflow:hidden; flex:none; }
.cb-b > span { display:block; height:100%; background:var(--accent); opacity:.75; }
.cbnone { padding:8px 10px; }
/* 11f: one action cell for every table with actions */
.actcell { display:flex; justify-content:flex-end; align-items:center; gap:8px; white-space:nowrap; }
.actcell > button, .actcell > a.btn, .actcell .confirm > button { height:32px; min-height:32px;
  border-radius:6px; padding:0 12px; font-size:var(--fs-2); box-sizing:border-box; }
.actcell > a.btn { display:inline-flex; align-items:center; text-decoration:none; }
a.btn.ghost { background:none; border:1px solid var(--border); color:var(--accent); }
a.btn.ghost:hover { background:var(--accent-soft); border-color:var(--accent); }
.actcell .confirm { display:inline-flex; align-items:center; gap:8px; }
.actcell > button.rowmenu { width:32px; padding:0; font-size:var(--fs-3); line-height:1;
  color:var(--text-secondary); }
button.ghost { background:none; border:1px solid var(--border); color:var(--accent); }
button.ghost:hover { background:var(--accent-soft); border-color:var(--accent); }
td.rowacts { width:1%; text-align:right; }
table[data-queue-table] td.rowacts { min-width:170px; }
td.nowrap, .nowrap { white-space:nowrap; }
/* a long failure: two lines, the rest a click away */
.clamp { display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; overflow:hidden;
  overflow-wrap:anywhere; }
.clamp.open { display:block; -webkit-line-clamp:unset; white-space:pre-wrap; }
button.clampbtn { padding:0; min-height:0; border:0; background:none; font-size:var(--fs-1);
  color:var(--accent); }
/* a toast rises 8px as it comes, and fades as it goes */
.toast { animation:toastIn var(--dur-3) var(--ease);
  transition:opacity var(--dur-3) var(--ease), transform var(--dur-3) var(--ease); }
.toast.out { opacity:0; transform:translateY(8px); pointer-events:none; }
@keyframes toastIn { from { opacity:0; transform:translateY(8px); } }
/* a new view — after a navigation, never a poll — fades in and rises 6px */
#view.view-enter { animation:viewIn var(--dur-3) var(--ease); }
@keyframes viewIn { from { opacity:0; transform:translateY(6px); } }
/* a value a poll changed: a soft wash that fades over 1.2s */
.changed { animation:changed 1.2s var(--ease); }
@keyframes changed { from { background-color:var(--accent-soft); } }
@media (prefers-reduced-motion: reduce) { .changed, .toast, #view.view-enter { animation:none; } }
/* a theme change: the colours cross-fade for 250ms, then the class goes */
html.theme-fade, html.theme-fade *, html.theme-fade *::before, html.theme-fade *::after {
  transition:background-color 250ms var(--ease), color 250ms var(--ease),
    border-color 250ms var(--ease) !important; }
@keyframes livepulse { 0%,100% { opacity:1; } 50% { opacity:.45; } }
/* the checks: the pill lives in the bar, its list opens just under the bar */
.bar-checks details.checks { position:static; margin:0; }
.bar-checks > details > summary { display:inline-flex; align-items:center; gap:6px;
  white-space:nowrap; font-family:var(--font-sans); font-size:var(--fs-1); font-weight:600;
  border:1px solid var(--border); border-radius:var(--r-1); padding:var(--sp-1) var(--sp-3);
  min-height:32px; box-sizing:border-box; background:var(--surface-1);
  color:var(--text-primary); cursor:pointer; list-style:none; }
.bar-checks > details > summary:hover { background:var(--plane); }
.bar-checks > details > summary::-webkit-details-marker { display:none; }
.bar-checks .checklist { position:absolute; left:0; right:0; top:100%;
  background:var(--surface-1); border-bottom:1px solid var(--border);
  box-shadow:0 10px 28px rgba(0,0,0,.10); padding:10px 22px; margin:0;
  max-height:min(60vh,520px); overflow:auto; z-index:59; }
/* the pill's panel: what kind of checks, then one line each */
.bar-checks .checklist li.checks-judged { list-style:none; color:var(--text-secondary);
  margin:0 0 6px; }
@media (max-width:720px) {
  .bar-in { height:auto; padding:8px 16px; flex-wrap:wrap; row-gap:4px; }
  .bar-title .t-full { display:none; }
  .bar-title .t-short { display:inline; }
  .bar-in .tabs { order:3; flex-basis:100%; margin:0; }
}
/* 11e: on a phone the bar is two rows, at most 96px — the title, LIVE and
   one ⋯ menu holding the checks, the name and the theme; then the tabs */
.bar-more { display:none; }
@media (max-width:600px) {
  .bar-in { padding:6px 16px 0; row-gap:2px; }
  .bar-more { display:inline-flex; align-items:center; justify-content:center; margin-left:auto;
    min-height:32px; min-width:40px; padding:0 10px; font-size:var(--fs-3); line-height:1; }
  .bar-right { display:none; position:absolute; right:16px; top:44px; z-index:60;
    flex-direction:column; align-items:stretch; gap:8px; min-width:220px; max-width:calc(100vw - 32px);
    background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2);
    padding:10px; box-shadow:0 10px 28px rgba(0,0,0,.16); }
  .bar[data-more="open"] .bar-right { display:flex; }
  .bar-right > * { width:100%; }
  .bar-right button, .bar-right summary { width:100%; justify-content:flex-start; text-align:left; }
  .bar-checks .checklist { position:static; box-shadow:none; border:0; padding:6px 0 0 18px;
    max-height:50vh; }
  .tabs button { padding:8px 10px; }
}
/* the hero, on Overview only */
.pagehero { margin:18px 0 0; display:grid; grid-template-columns:minmax(0, 1fr) auto;
  grid-template-areas:"eb acts" "sub acts" "chips acts"; column-gap:24px; align-items:center; }
.pagehero > #heroEyebrow { grid-area:eb; margin:0; }
.pagehero > #pageSub { grid-area:sub; margin:4px 0 0; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.pagehero > h1 { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0);
  white-space:nowrap; }
.hero-acts { grid-area:acts; display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.hero-acts:empty { display:none; }
.pagehero[hidden] { display:none; }
.meta-chips { grid-area:chips; display:flex; gap:14px; flex-wrap:wrap; margin-top:6px; }
.meta-chips a.chip { border:0; background:none; padding:0; font-size:var(--fs-2); color:var(--accent); }
.meta-chips a.chip:hover { text-decoration:underline; }
@media (max-width:720px) { .pagehero { grid-template-columns:minmax(0, 1fr);
  grid-template-areas:"eb" "sub" "chips" "acts"; }
  .pagehero > #pageSub { white-space:normal; } .hero-acts { justify-content:flex-start; margin-top:10px; } }
/* back to top: a mono pill, after two screens of scrolling */
.totop { position:fixed; right:18px; bottom:18px; z-index:45; font-family:var(--font-sans);
  font-size:var(--fs-1); border-radius:999px; padding:6px 13px; background:var(--surface-1);
  box-shadow:0 6px 18px rgba(0,0,0,.12); }
.totop[hidden] { display:none; }
@media print { .totop { display:none; } }
.chip { display:inline-flex; align-items:center; gap:6px; font-size:var(--fs-1);
  font-family:var(--font-sans);
  color:var(--text-secondary); background:var(--surface-1); border:1px solid var(--border);
  border-radius:999px; padding:3px 10px; }
.chip .mono { font-size:var(--fs-1); }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2);
  padding:var(--sp-5); margin:var(--sp-4) 0; }
@media (max-width:720px) { .card { padding:var(--sp-4); } }
button, .btn { font:inherit; font-size:var(--fs-2); color:var(--text-primary); background:var(--surface-1);
  border:1px solid var(--border); border-radius:var(--r-1); padding:var(--sp-1) var(--sp-3);
  min-height:32px; cursor:pointer; }
button:hover, .btn:hover { background:var(--plane); }
input[type=search] { font:inherit; font-size:var(--fs-2); color:var(--text-primary);
  background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-1);
  padding:6px 11px; width:220px; }
input[type=search]:focus { outline:2px solid var(--accent-soft); border-color:var(--accent); }
.filters { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:14px 0 4px; }
.seg { display:inline-flex; border:1px solid var(--border); border-radius:var(--r-1); overflow:hidden; }
.seg button { border:0; border-radius:0; background:var(--surface-1); padding:6px 12px; }
.seg button + button { border-left:1px solid var(--border); }
.seg button[aria-pressed="true"] { background:var(--accent-soft); color:var(--text-primary); font-weight:600; }
.count-note { font-size:var(--fs-1); color:var(--muted); }
.tabs { display:flex; gap:2px; margin:0; overflow-x:auto; scrollbar-width:none;
  align-self:stretch; align-items:stretch; flex:1 1 auto; min-width:0; }
.tabs::-webkit-scrollbar { display:none; }
.tabs .morewrap { display:flex; align-items:stretch; }
.tabs button { border:0; background:none; border-radius:0; padding:8px 13px;
  color:var(--text-secondary); white-space:nowrap; }
.tabs button:hover { color:var(--text-primary); }
.tabs button[aria-selected="true"] { color:var(--text-primary); font-weight:600; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }
.tile { background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2);
  padding:13px 16px; }
.tile .label { color:var(--text-secondary); font-size:var(--fs-1); }
.tile .value { font-size:var(--fs-4); font-weight:600; letter-spacing:-0.02em; }
.tile .note { font-size:var(--fs-1); color:var(--muted); margin-top:2px; }
.hero-row { display:grid; grid-template-columns:minmax(260px,1.2fr) 2fr; gap:12px; }
@media (max-width:800px){ .hero-row { grid-template-columns:1fr; } }
.hero { font-size:var(--fs-5); font-weight:700; letter-spacing:-0.02em; line-height:1.15; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
/* 11b: one table component. Mono uppercase headers, one-line rows of 44px,
   hairline separators, numbers right-aligned in mono with tabular figures. */
th { text-align:left; font-family:var(--font-mono); font-size:var(--fs-1);
  text-transform:uppercase; letter-spacing:.03em;
  color:var(--muted); font-weight:600; padding:var(--sp-2) var(--sp-3); border-bottom:1px solid var(--border);
  white-space:nowrap; }
th .unit { display:block; font-weight:400; text-transform:none; letter-spacing:0; }
td { padding:10px var(--sp-3); height:44px; box-sizing:border-box;
  border-bottom:1px solid var(--border); font-size:var(--fs-2); }
td.num, th.num { text-align:right; }
td.num { font-family:var(--font-mono); font-variant-numeric:tabular-nums; }
tbody tr:hover { background:var(--accent-soft); }
/* an opened or selected row carries a 3px accent bar on its left */
tbody tr.open > td:first-child, tbody tr[aria-selected="true"] > td:first-child {
  box-shadow:inset 3px 0 0 var(--accent); }
tbody tr.open { background:var(--accent-soft); }
/* ---- 11e: the plan box on a proposal card -------------------------------- */
.planbox { border:1px solid var(--border); border-radius:var(--r-2); padding:10px 14px;
  margin:10px 0; background:var(--plane); display:flex; flex-direction:column; gap:4px; }
.planbox .plan { margin:0; color:var(--text-primary); }
.planbox label.spread { display:inline-flex; align-items:center; gap:6px; }
/* ---- 11d: Overview ------------------------------------------------------- */
.statline { font-size:var(--fs-2); color:var(--muted); margin:6px 0 2px; font-family:var(--font-sans); }
/* 11e: the hero's "Submit a model" and the stats line under it: 12px */
.pagehero:not([hidden]) + #view > .statline[data-statline] { margin-top:12px; }
.pagehero:not([hidden]) { margin-bottom:0; }
#view > .statline:first-child { margin-top:-2px; }
.hlgrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:14px;
  margin-top:10px; }
.hcard { border:1px solid var(--border); border-radius:var(--r-2); padding:14px 16px;
  display:flex; flex-direction:column; gap:6px; background:var(--surface-1); }
.hcard-v { font-family:var(--font-sans); font-size:var(--fs-5); font-weight:700;
  font-variant-numeric:tabular-nums; letter-spacing:-0.02em; line-height:1.15;
  color:var(--text-primary); overflow-wrap:anywhere; }
.hcard-v { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.hcard-name { font-size:var(--fs-2); font-weight:600; color:var(--text-primary);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
.hcard-verdict { margin:0; font-size:var(--fs-2); color:var(--muted); }
.hcard-link { margin-top:auto; padding-top:4px; font-size:var(--fs-2); font-family:var(--font-sans);
  color:var(--accent); text-decoration:none; }
.hcard-link:hover, .hcard-link:focus-visible { text-decoration:underline; }
.hlgrid { align-items:stretch; }
table.mini-lb td.model { position:static; box-shadow:inset 3px 0 0 var(--fam, var(--axis));
  padding-left:12px; background:none; }
/* 11h: Top models — the hover is the whole row */
table.mini-lb tbody tr:hover > td { background-color:var(--accent-soft); }
table.mini-lb tbody tr:hover > td[data-lead] { background-image:linear-gradient(var(--accent-soft),
  var(--accent-soft)); }
/* ---- 11d: the model page -------------------------------------------------- */
.mhero .mhead { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-top:4px; }
.mhero h1.mtitle { font-size:var(--fs-5); font-weight:800; letter-spacing:-0.025em; margin:0; }
.mhero .eyebrow { margin:0; }
.mprose { color:var(--muted); font-size:var(--fs-2); max-width:72ch; margin:14px 0 0; }
nav.modelnav { position:sticky; top:var(--bar-h); z-index:6; display:flex; gap:6px; flex-wrap:wrap;
  padding:8px 0; margin:var(--sp-4) 0 0; background:var(--plane); }
.navchip { font-family:var(--font-sans); font-size:var(--fs-1); text-decoration:none;
  border:1px solid var(--border); border-radius:999px; padding:3px 11px; color:var(--text-secondary);
  background:var(--surface-1); }
.navchip::before { content:attr(data-ix); color:var(--accent); margin-right:6px; font-weight:600;
  font-family:var(--font-mono); }
.navchip[aria-current="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.navchip[aria-current="true"]::before { color:#fff; }
[id^="sec-"] { scroll-margin-top:calc(var(--bar-h) + 52px); }
tr.arearow td { background:var(--plane); height:auto; padding-top:8px; padding-bottom:6px; }
table[data-judged-topics] td.num { white-space:nowrap; }
/* the row's action is a small text button: it goes to the topic page */
table[data-judged-topics] a.propose, table[data-judged-topics] button.propose {
  border:0; background:none; padding:0; min-height:0;
  font-family:var(--font-sans); font-size:var(--fs-1); font-weight:600; color:var(--accent);
  text-decoration:none; }
table[data-judged-topics] a.propose:hover,
table[data-judged-topics] button.propose:hover { text-decoration:underline; }
td.lencell { font-family:var(--font-mono); color:var(--text-primary); }
td.lencell.few { color:var(--muted); }
details.lenfold > summary { cursor:pointer; font-size:var(--fs-1); color:var(--accent); margin:4px 0; }
/* ---- 11c: the Leaderboard ------------------------------------------------ */
.lbbar { display:flex; flex-wrap:wrap; align-items:center; gap:8px 16px; margin:10px 0 2px; }
.lbbar .chips, .lbbar .pills { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.lbbar .pills { margin-left:auto; }
.lbbar .propwhy { flex-basis:100%; }
.chip-btn, .pill { font-family:var(--font-sans); font-size:var(--fs-1); border-radius:999px;
  padding:4px 12px; min-height:28px; border:1px solid var(--border); background:var(--surface-1);
  color:var(--text-secondary); white-space:nowrap; }
.chip-btn:hover, .pill:hover { color:var(--text-primary); border-color:var(--axis); }
.chip-btn.on { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
.chip-btn[aria-disabled="true"], .chip-btn[aria-disabled="true"]:hover { opacity:.5;
  cursor:not-allowed; color:var(--text-secondary); border-color:var(--border); background:none; }
.lbbar .chipnote { flex-basis:100%; margin:0; font-size:var(--fs-1); color:var(--text-secondary); }
.lbbar .chipnote[hidden] { display:none; }
.pill.on { border-color:var(--accent); color:var(--text-primary); }
.pill[aria-expanded="true"] { border-color:var(--accent); }
/* 11f: a quiet group row over one line of names. The group is muted, with a
   hairline bracket across its columns: a line, and a short tick at each end */
table.lb thead tr.grp th { font-size:var(--fs-1); color:var(--muted); font-weight:500;
  border-bottom:0; padding:0 6px 7px; letter-spacing:.08em; text-align:center; height:24px;
  box-sizing:border-box; }
table.lb thead tr.grp th.nogrp { color:transparent; }
table.lb thead tr.grp th:not(.nogrp) {
  background:
    linear-gradient(var(--axis), var(--axis)) left 8px bottom 3px / calc(100% - 16px) 1px no-repeat,
    linear-gradient(var(--axis), var(--axis)) left 8px bottom 0 / 1px 4px no-repeat,
    linear-gradient(var(--axis), var(--axis)) right 8px bottom 0 / 1px 4px no-repeat,
    var(--surface-1); }
table.lb th .dir { color:var(--accent); }
/* the setup is the name's tooltip: the name says so on hover and focus */
table.lb thead th[data-tip] { cursor:help; }
table.lb thead th.sortable { cursor:pointer; }
table.lb thead th[data-tip]:hover .hname, table.lb thead th[data-tip]:focus-visible .hname {
  text-decoration:underline dotted; text-underline-offset:3px; }
/* the cells: plain, and the leaders bold (their wash is --heat-3, inline) */
table.lb td.tcell { font-weight:400; }
table.lb td.tcell.lead b { font-weight:700; }
table.lb td.tcell:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.lbcap { font-family:var(--font-sans); font-size:var(--fs-1); color:var(--muted); margin:8px 0 0; }
/* one line per row */
table.lb td { white-space:nowrap; }
table.lb td.tcell b { font-family:var(--font-mono); }
table.lb td.tcell .se { font-family:var(--font-mono); display:inline; }
table.lb td.model { position:sticky; left:32px; background:var(--surface-1); z-index:1;
  padding-left:12px; max-width:280px; }
table.lb td.model .badge { margin-left:0; padding:0 4px; }
table.lb .mcell { display:flex; align-items:center; gap:4px; max-width:260px; overflow:hidden;
  white-space:nowrap; }
table.lb .mcell .mname { flex:0 3 auto; min-width:40px; max-width:none; overflow:hidden;
  text-overflow:ellipsis; }
/* a long badge ("duplicate of <name>") gives way too, after the name: every
   child stays inside the cell. The short ones (base, prelim) keep their word */
table.lb .mcell .badge { flex:none; white-space:nowrap; }
table.lb .mcell .badge[data-duplicate], table.lb .mcell .badge[data-near-duplicate] {
  flex:0 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; }
table.lb .mcell .duptoggle { flex:none; padding:0 2px; min-height:0; }
.hfade { position:relative; min-width:0; }
.hfade::after { content:""; position:absolute; top:0; right:0; bottom:0; width:32px; z-index:6;
  pointer-events:none; opacity:0; transition:opacity .15s;
  background:linear-gradient(to right, transparent, var(--surface-1)); }
.hfade[data-more="1"]::after { opacity:1; }
.hfade .scrollhint { display:none; position:absolute; top:0; right:0; z-index:7;
  pointer-events:none; font-family:var(--font-sans); font-size:var(--fs-1);
  height:24px; line-height:24px; color:var(--text-secondary); background:var(--surface-1);
  padding:0 2px 0 8px; box-shadow:-10px 0 8px var(--surface-1); }
.hfade[data-more="1"] .scrollhint { display:block; }
/* a phone: the rank and the model together take at most 45% of the scroller,
   the name ellipsises, and of the badges only "prelim" stays — the scores
   are what the table is for */
/* the Leaderboard's scroller is a query container: the opened panel and, on
   a phone, the pinned columns are sized by what is visible, not the table */
.lb-wrap.stick[data-hkeep="lb"] { container-type:inline-size; }
@media (max-width:600px) {
  table.lb td.model { padding-left:6px; padding-right:4px; }
  table.lb .mcell { max-width:calc(45cqi - 42px); }
  table.lb .mcell .badge:not(.prelim), table.lb .mcell .duptoggle { display:none; }
  /* the active parameters stay in the tooltip: Params is one short number */
  table.lb td .act { display:none; }
  table.lb .mcell .badge.prelim { flex:0 1 auto; min-width:0; overflow:hidden;
    text-overflow:ellipsis; }
}
/* the opened row's blocks: each at least 300px (or the row), none painting
   into the next — the Leaderboard's one-line rule stops at the detail */
table.lb tr.detail .dgrid { grid-template-columns:repeat(auto-fit, minmax(min(100%, 300px), 1fr)); }
table.lb tr.detail .dblock { min-width:0; }
table.lb tr.detail td td, table.lb tr.detail td th { white-space:normal; }
table.lb tr.detail table.mini { width:100%; }
table.lb tbody td.model { box-shadow:inset 3px 0 0 var(--fam, var(--axis)); }
table.lb td.model .mname { display:inline-block; max-width:190px; overflow:hidden;
  text-overflow:ellipsis; vertical-align:bottom; }
table.lb th.model { position:sticky; left:32px; z-index:3; }
/* the rank and the model stay put while the scores scroll sideways */
table.lb .rank { white-space:nowrap; color:var(--muted); position:sticky; left:0; z-index:2;
  text-align:left;
  background:var(--surface-1); width:32px; min-width:32px; max-width:32px;
  padding-left:4px; padding-right:4px; }
table.lb tbody tr.open td.rank { background:color-mix(in srgb, var(--accent) 8%, var(--surface-1)); }
table.lb .rank .disclose { border:0; background:none; min-height:0; padding:0 3px 0 0;
  color:var(--accent); font-size:var(--fs-1); }
/* 11f: one chevron, turned a quarter when the row is open */
table.lb .disclose .chev { display:inline-block; transition:transform var(--dur-chev) var(--ease); }
table.lb .disclose[aria-expanded="true"] .chev { transform:rotate(90deg); }
table.lb .disclose.pre[aria-expanded="true"] .chev { transform:none; }
/* it should be obvious a row opens: the wash, the pointer, the chevron in ink */
table.lb tbody tr[data-lb-row] { cursor:pointer; }
table.lb tbody tr[data-lb-row]:hover { background:var(--accent-soft); }
table.lb tbody tr[data-lb-row]:hover .disclose { color:var(--text-primary); }
/* the pinned cells are opaque (they slide over the scores): the same wash,
   laid over their own surface, so the whole row reads as one */
table.lb tbody tr[data-lb-row]:hover > td.rank, table.lb tbody tr[data-lb-row]:hover > td.model,
table.lb tbody tr.open > td.rank, table.lb tbody tr.open > td.model {
  background:linear-gradient(var(--accent-soft), var(--accent-soft)), var(--surface-1); }
/* the opened panel: the row's own continuation — no hairline between them,
   the row's 3px bar down its left edge, a faint wash, 16px in, rounded below */
table.lb tbody tr.open > td { border-bottom-color:transparent; }
table.lb tbody tr.detail { cursor:auto; background:none; }
table.lb tbody tr.detail:hover { background:none; }
table.lb tbody tr.detail > td { white-space:normal; padding:0; height:auto; }
.dwrap { display:grid; grid-template-rows:1fr; }
.dwrap > .dinner { min-height:0; overflow:hidden; }
.dwrap .dpad { position:sticky; left:0; max-width:100cqi; box-sizing:border-box;
  padding:16px; margin:0 0 10px;
  background:color-mix(in srgb, var(--accent) 5%, var(--surface-1));
  box-shadow:inset 3px 0 0 var(--accent); border-radius:0 0 var(--r-2) var(--r-2); }
.dwrap.anim { transition:grid-template-rows var(--dur-row) var(--ease); }
.dwrap.anim .dpad { transition:opacity var(--dur-row) var(--ease), transform var(--dur-row) var(--ease); }
.dwrap.closed { grid-template-rows:0fr; }
.dwrap.closed .dpad { opacity:0; transform:translateY(-4px); }
.dwrap .dclose { position:absolute; top:8px; right:10px; min-height:28px; padding:2px 8px;
  font-size:var(--fs-1); z-index:1; }
.dwrap .dgrid { padding-right:72px; }
/* the Tasks list: one line each */
.tlist { display:flex; flex-direction:column; }
.tl { display:grid; grid-template-columns:minmax(0, 1fr) auto 56px auto; gap:10px; align-items:center;
  min-height:26px; border-bottom:1px solid var(--grid); font-size:var(--fs-2); white-space:nowrap; }
.tl:last-child { border-bottom:0; }
.tl .tl-n { overflow:hidden; text-overflow:ellipsis; }
.tl .tl-v { font-family:var(--font-mono); text-align:right; }
.tl .tl-b { height:4px; background:var(--grid); border-radius:2px; overflow:hidden; }
.tl .tl-f { display:block; height:100%; background:var(--accent); opacity:.7; }
.tl .tl-r { font-size:var(--fs-1); }
.tl .se { color:var(--muted); font-size:var(--fs-1); }
table.lb tbody tr.open td.model { background:color-mix(in srgb, var(--accent) 8%, var(--surface-1)); }
/* a tinted cell's text is ink, never the accent — its error too, in the
   secondary ink, which holds 4.5:1 on the strongest step in every theme */
table.lb td[data-lead] { color:var(--text-primary); }
table.lb td[data-lead] .se { color:var(--text-secondary); }
.dgrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:14px 22px; }
.dblock .eyebrow { margin-bottom:6px; }
table.mini { width:auto; }
table.mini td, table.mini th { height:auto; padding:3px 10px 3px 0; border-bottom:1px solid var(--border); }
.minibars { display:flex; flex-direction:column; gap:4px; }
.minibar { display:grid; grid-template-columns:minmax(0, 150px) 1fr 40px; gap:8px; align-items:center;
  font-size:var(--fs-1); }
.minibar .mb-t { height:8px; background:var(--grid); border-radius:4px; overflow:hidden; }
.minibar .mb-f { display:block; height:100%; background:var(--accent); }
.minibar .mb-v { text-align:right; }
.jgroup { margin:4px 0 6px; }
.tchip.grey { color:var(--text-secondary); background:var(--plane); border-color:var(--border); }
a.tchip { text-decoration:none; }
.dlinks { display:flex; flex-direction:column; gap:6px; align-items:flex-start; }
.dlinks button.quiet { padding:0; border:0; min-height:0; background:none; color:var(--accent);
  font:inherit; text-align:left; cursor:pointer; }
.dlinks button.quiet:hover:not(:disabled) { background:none; text-decoration:underline; }
.colmenu-list .colgroup { margin:6px 0 8px; display:flex; flex-direction:column; gap:2px; }
.colmenu-list.pop { min-width:260px; padding:10px 12px; gap:4px; }
.colmenu-list .colgroup { max-height:260px; overflow:auto; }
.tintsw { margin-top:8px; border-top:1px solid var(--border); padding-top:8px; }
.modelsmenu.pop { min-width:280px; padding:10px 12px; gap:6px; }
.mlist { display:flex; flex-direction:column; gap:2px; max-height:280px; overflow:auto; }
.mlist .mrow { display:flex; align-items:center; gap:4px; white-space:nowrap; }
.mlist [role=menuitem] { border:0; background:none; text-align:left; padding:5px 8px; }
/* 11k: inside a list, nothing is bordered or underlined; the chosen row is
   tinted and ticked, and the keyboard's row carries the ring */
.moremenu [role=menuitem], .moremenu [role=menuitemradio], .mlist [role=menuitem] {
  border:0; border-bottom:0; text-decoration:none; }
.moremenu [role=menuitem][aria-current="true"], .mlist [role=menuitem][aria-current="true"],
.moremenu [role=menuitemradio][aria-checked="true"] { background:var(--accent-soft); }
.moremenu [role=menuitem][aria-current="true"]::after,
.mlist [role=menuitem][aria-current="true"]::after,
.moremenu [role=menuitemradio][aria-checked="true"]::after {
  content:"✓"; color:var(--accent); margin-left:auto; padding-left:8px; }
.mlist [role=menuitem] { display:flex; align-items:center; }
.famdot { width:8px; height:8px; border-radius:50%; display:inline-block; flex:none; }
details.howto { margin-top:14px; border-top:1px solid var(--border); padding-top:10px; }
details.howto > summary { cursor:pointer; font-family:var(--font-sans); font-size:var(--fs-1);
  color:var(--text-secondary); list-style:none; }
details.howto > summary::-webkit-details-marker { display:none; }
details.howto .about { margin-top:12px; }
/* Insights */
.igrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 520px), 1fr));
  gap:22px 30px; }
/* 11e: a chart is never drawn narrower than its viewBox, so its 12px text is
   12px on the screen; on a phone it scrolls sideways in its own box */
.chartscroll { overflow-x:auto; }
.chartscroll > svg { display:block; max-width:680px; }
/* Weakest topics on a phone: HTML rows, not a shrunken SVG */
.wrows { display:none; }
.wrow { display:grid; grid-template-columns:minmax(0, 1fr) 34% auto; gap:8px; align-items:center;
  min-height:28px; color:var(--text-primary); text-decoration:none; font-size:var(--fs-2); }
.wrow:hover { background:var(--plane); }
.wrow .wname { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--text-secondary); }
.wrow .wtrack { height:10px; border-radius:2px; background:var(--grid); overflow:hidden; }
.wrow .wfill { height:100%; background:var(--accent); opacity:.75; }
.wrow .wfill.hatch { background:repeating-linear-gradient(45deg, var(--axis) 0 2px, var(--grid) 2px 6px);
  opacity:1; }
.wrow .wv { font-family:var(--font-mono); font-size:var(--fs-1); }
@media (max-width:600px) {
  .ibox .weakest-svg { display:none; }
  .wrows { display:block; }
}
.ibox { min-width:0; }
.ihead .eyebrow { margin-bottom:2px; }
svg .fpt { cursor:pointer; }
svg .fpt:focus { outline:none; stroke:var(--accent); stroke-width:3; }
svg .wbar:focus rect { stroke:var(--accent); stroke-width:2; }
.rchips { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:6px 0; }
.mchip { display:inline-flex; align-items:center; gap:6px; font-family:var(--font-sans);
  font-size:var(--fs-1); border:1px solid var(--border); border-radius:999px; padding:2px 4px 2px 10px; }
details.astable > summary { cursor:pointer; font-size:var(--fs-1); color:var(--accent); }
@media (max-width:720px) { .igrid { grid-template-columns:1fr; } .lbbar .pills { margin-left:0; } }
@media (max-width:520px) { .igrid { grid-template-columns:minmax(0, 1fr); } }
/* a table that scrolls sideways can pin its first column (11c uses it) */
td.pin, th.pin { position:sticky; left:0; background:var(--surface-1); z-index:1; }
/* the Leaderboard carries a dozen numeric columns: it keeps the 44px row and
   takes its breathing room from the gutters instead (11c narrows the cells
   themselves, to one line each) */
table.lb td, table.lb th { padding-left:6px; padding-right:6px; }
tbody tr.open td.pin, tbody tr:hover td.pin {
  background:color-mix(in srgb, var(--accent) 8%, var(--surface-1)); }
tr:last-child td { border-bottom:none; }
.lb-wrap { overflow-x:auto; }
th.sortable { cursor:pointer; user-select:none; }
th.sortable:hover { color:var(--text-primary); }
th .dir { font-size:9px; }
/* typeahead dropdown under the metric query bar */
.ac { position:relative; max-width:620px; }
.ac-list { position:absolute; top:calc(100% + 4px); left:0; right:0; z-index:30;
  background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2);
  box-shadow:0 10px 28px rgba(0,0,0,.14); max-height:288px; overflow-y:auto; }
.ac-item { display:flex; gap:10px; align-items:center; padding:7px 11px;
  cursor:pointer; font-size:var(--fs-2); }
.ac-item[aria-selected=true], .ac-item:hover { background:var(--accent-soft); }
.ac-item .ac-name { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:var(--fs-1); }
.ac-item mark { background:none; color:var(--accent); font-weight:650; padding:0; }
.ac-tag { font-size:10px; color:var(--muted); border:1px solid var(--border);
  border-radius:999px; padding:1px 7px; flex:none; }
.toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:10px 0 4px; }
/* a select is as wide as its longest option — an author's criterion label can
   be longer than a phone is wide */
select { max-width:100%; }
.toolbar input, .toolbar select { font:inherit; font-size:var(--fs-1); color:var(--text-primary);
  background:var(--plane); border:1px solid var(--border); border-radius:var(--r-1);
  padding:5px 9px; }
.toolbar input:focus, .toolbar select:focus { outline:2px solid var(--accent-soft);
  border-color:var(--accent); }
.lb td.model, .lb th.model { position:sticky; left:0; background:var(--surface-1); z-index:1; }
.lb tr:hover td { background:var(--plane); }
.lb .se { color:var(--muted); font-size:var(--fs-1); }
.best { font-weight:650; }
.best::after { content:"\2009\25CF"; color:var(--accent); font-size:8px; vertical-align:2px; }
.badge { display:inline-block; font-size:var(--fs-1); font-family:var(--font-mono);
  border:1px solid var(--border);
  border-radius:var(--r-1); padding:0 5px; margin-left:6px; color:var(--text-secondary);
  vertical-align:1px; }
/* three tones and no others (9d): "instruct" is information, so neutral —
   the accent tint it wore was 3.9:1 in the dark theme */
.badge.instruct { color:var(--text-primary); font-weight:600; }
.badge.ckpt { border-style:dashed; color:var(--text-secondary); }
.badge.prelim { color:var(--warning-text); border-color:var(--warning); }
.tiebest { font-weight:650; }
.tiebest::after { content:"\2009\2248"; color:var(--muted); font-size:9px; vertical-align:1px; }
.mono { font-family:var(--font-mono); font-size:var(--fs-1);
  color:var(--text-secondary); }
/* 11b: the eyebrow — mono, uppercase, letter-spaced, in the accent */
.eyebrow { font-family:var(--font-mono); font-size:var(--fs-1); font-weight:600;
  text-transform:uppercase; letter-spacing:.16em; color:var(--accent); }
h2[data-ix]::before { content:attr(data-ix); font-family:var(--font-mono);
  font-size:var(--fs-1); font-weight:600; color:var(--accent); letter-spacing:.06em;
  vertical-align:2px; margin-right:8px; }
/* the section index that runs 01, 02, … down each tab */
.sechead { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin:0 0 2px; }
.sechead .ix { font-family:var(--font-mono); font-size:var(--fs-1); font-weight:600;
  color:var(--accent); letter-spacing:.06em; }
.sechead .acts { margin-left:auto; display:flex; gap:8px; align-items:center; }
/* the one-line status above a big table */
.statusline { font-family:var(--font-sans); font-size:var(--fs-1); color:var(--muted);
  margin:8px 0 6px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 10px; }
.legend span { display:inline-flex; align-items:center; gap:6px; font-size:var(--fs-1);
  color:var(--text-secondary); }
.key { width:11px; height:11px; border-radius:3px; display:inline-block; }
.key.line { width:14px; height:0; border-top:3px solid; border-radius:2px; }
.panels { display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:12px; }
@media (max-width:520px){ .panels { grid-template-columns:1fr; } }
.panel { background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2);
  padding:14px 16px 8px; }
.panel h3 { font-size:var(--fs-2); font-weight:600; margin:0; }
.panel .pmeta { font-size:var(--fs-1); color:var(--muted); margin:1px 0 8px; }
/* chip row under a task heading: what it measures, how many options, coverage,
   whether it separates anything, where the ceiling is. Every chip carries the
   long version in its title, so the row stays short. */
/* "about these benchmarks" */

.about { padding:0; }
.about-toggle { font:inherit; font-size:var(--fs-2); font-weight:600; width:100%; text-align:left;
  background:none; border:0; color:var(--text-primary); padding:12px 16px; cursor:pointer;
  border-radius:var(--r-2); }
.about-toggle:hover { color:var(--accent); }
.about-body { padding:0 16px 14px; display:grid;
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px 22px; }
.about-item { border-top:1px solid var(--border); padding-top:10px; }
.about-head { display:flex; align-items:center; gap:5px; flex-wrap:wrap; margin-bottom:4px; }
.about-name { font-weight:600; font-size:var(--fs-2); }
.about-desc { margin:0; font-size:var(--fs-1); line-height:1.5; color:var(--text-secondary); }
/* model detail page */
.backlink { display:inline-block; font-size:var(--fs-1); color:var(--accent);
  text-decoration:none; margin:10px 2px; }
.backlink:hover { text-decoration:underline; }
.mhead { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.mtitle { margin:0; font-size:var(--fs-4); letter-spacing:-0.01em; }
.rankbadge { font-size:var(--fs-1); font-weight:650; color:var(--accent);
  background:var(--accent-soft); border-radius:var(--r-1); padding:1px 7px; }
.mlink { color:inherit; text-decoration:none; border-bottom:1px dotted var(--border); }
.mlink:hover { color:var(--accent); border-bottom-color:var(--accent); }
.mtbl td, .mtbl th { vertical-align:middle; }
.mtbl .tname { font-weight:550; cursor:help; }
.mtbl .sbar { display:block; }
.mtbl td.dimmed { color:var(--muted); }
.mtbl tr.domrow td { font-size:var(--fs-1); font-weight:650; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); padding-top:14px; border-bottom:none; }
.mtbl tr.flagrow td { font-size:var(--fs-1); color:var(--muted); padding-top:0; border-bottom:none; }
.provlist { display:grid; grid-template-columns:minmax(120px,max-content) 1fr;
  gap:4px 16px; margin:8px 0 0; font-size:var(--fs-1); }
.provlist dt { color:var(--muted); }
.provlist dd { margin:0; overflow-wrap:anywhere; }
/* per-item diagnosis */
.dxlead { margin:4px 0 0; padding:9px 12px; border-radius:var(--r-2); font-size:var(--fs-2);
  line-height:1.55; background:var(--accent-soft); color:var(--text-primary); }
.dxlead + .dxlead { margin-top:7px; }
.dxlead b { font-weight:650; }
.dxlead.calm { background:var(--plane); color:var(--text-secondary); }
.dx { border:1px solid var(--border); border-radius:var(--r-2); margin-top:8px;
  background:var(--surface-1); }
.dx > summary { cursor:pointer; padding:9px 12px; display:flex; align-items:center;
  gap:10px; flex-wrap:wrap; font-size:var(--fs-2); list-style:none; }
.dx > summary::-webkit-details-marker { display:none; }
.dx > summary::before { content:'▸'; color:var(--muted); font-size:var(--fs-1); width:9px; }
.dx[open] > summary::before { content:'▾'; }
.dx[open] > summary { border-bottom:1px solid var(--border); }
.dx > summary:hover { color:var(--accent); }
.dx .dxname { font-weight:600; }
.dx .dxbody { padding:10px 12px 14px; }
.dxflag { font-size:var(--fs-1); font-weight:650; letter-spacing:.03em; border-radius:var(--r-1);
  padding:1px 6px; background:color-mix(in srgb, var(--s2) 16%, transparent);
  color:var(--s2); border:1px solid color-mix(in srgb, var(--s2) 45%, transparent); }
.dxbar { display:flex; height:12px; border-radius:var(--r-1); overflow:hidden;
  background:var(--plane); min-width:120px; }
.dxbar > span { display:block; height:100%; }
.dxkey { display:flex; flex-wrap:wrap; gap:4px 16px; margin:9px 0 0; font-size:var(--fs-1); }
.dxkey > span { display:flex; align-items:baseline; gap:6px; cursor:help; }
.dxkey i { display:inline-block; width:10px; height:10px; border-radius:3px;
  border:1px solid var(--border); flex:none; transform:translateY(1px); }
.dxkey b { font-weight:600; font-variant-numeric:tabular-nums; }
.dxsub { width:100%; border-collapse:collapse; font-size:var(--fs-1); margin-top:6px; }
.dxsub td, .dxsub th { padding:3px 8px 3px 0; border-bottom:1px solid var(--border);
  text-align:left; }
.dxsub th { font-size:var(--fs-1); font-weight:600; color:var(--muted); }
.dxsub td.num, .dxsub th.num { text-align:right; font-variant-numeric:tabular-nums; }
/* categories first, subjects on expand; a dim row is under the noise floor */
.dxcat { border-bottom:1px solid var(--border); }
.dxcat > summary { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  padding:5px 0; cursor:pointer; font-size:var(--fs-1); list-style:none; }
.dxcat > summary::-webkit-details-marker { display:none; }
.dxcat > summary::before { content:"\25B8"; color:var(--muted); font-size:10px; width:10px; }
.dxcat[open] > summary::before { content:"\25BE"; }
.dxcat .dxcname { font-weight:600; min-width:15ch; }
.dxcat .num { font-variant-numeric:tabular-nums; min-width:6ch; text-align:right; }
.dxcat.dim > summary { color:var(--muted); }
.dxcat.dim .dxcname { font-weight:500; }
.dxcat .dxsub { margin:2px 0 8px 20px; width:auto; min-width:60%; max-width:calc(100% - 20px); }
/* a subject is one long word (high_school_government_and_politics): let it
   break rather than push the table past a phone's width */
.dxsub td:first-child { overflow-wrap:anywhere; }
.dxperm { margin:14px 0 4px; }
.dxperm .dxsub { width:auto; min-width:60%; }
.lb td.dim { color:var(--muted); }
.badge.taint { color:var(--s2); border-color:var(--s2); }
.jbar { display:flex; height:10px; border-radius:var(--r-1); overflow:hidden; background:var(--plane); min-width:110px; }
.jbar > span { display:block; height:100%; }
.jd { width:100%; border-collapse:collapse; font-size:var(--fs-1); margin-top:6px; }
.jd td, .jd th { padding:4px 8px 4px 0; border-bottom:1px solid var(--border); text-align:left; }
.jd th { font-size:var(--fs-1); font-weight:600; color:var(--muted); }
.jd td.num, .jd th.num { text-align:right; font-variant-numeric:tabular-nums; }
.jd tr.dim td { color:var(--muted); }
.jd tr.dim .jbar { opacity:.4; }
.lb th.judged { color:var(--text-secondary); font-style:italic; }
/* the one button that can spend money and make training data: never colour alone */
.propose { font:inherit; font-size:var(--fs-1); padding:2px 9px; border-radius:var(--r-1);
  border:1px solid var(--accent); background:var(--accent-soft); color:var(--accent);
  cursor:pointer; margin-left:auto; }
.propose:disabled { border-color:var(--border); background:var(--plane);
  color:var(--muted); cursor:not-allowed; }
.propwhy { flex-basis:100%; font-size:var(--fs-1); color:var(--muted); margin:0 0 4px 20px; }
/* 11i: sit the exam from the model page — one grouped picker, and the
   Suite cell that stays one line */
.mcards { display:flex; gap:14px; align-items:stretch; }
.mcards > .hlgrid { flex:1; min-width:0; }
.msit-cta { display:flex; flex-direction:column; justify-content:center; align-items:flex-start;
  gap:6px; flex:0 0 auto; }
@media (max-width: 900px) { .mcards { flex-direction:column; } }
.msit-head { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.msit-head h2 { margin:0; }
.expick { margin-top:var(--sp-3); }
.exquick { gap:8px; flex-wrap:wrap; }
.exareas { display:grid; grid-template-columns:repeat(auto-fill, minmax(270px, 1fr));
  gap:14px 24px; margin-top:12px; }
.exarea { min-width:0; }
.exhead { display:flex; justify-content:space-between; align-items:baseline; gap:8px;
  padding-bottom:4px; margin-bottom:4px; border-bottom:1px solid var(--border);
  font-weight:600; font-size:var(--fs-1); }
.exhead label { cursor:pointer; }
.excount { font-weight:400; color:var(--muted); font-variant-numeric:tabular-nums; }
.extopic { display:grid; grid-template-columns:auto minmax(0, 1fr) auto; column-gap:8px;
  align-items:baseline; padding:3px 0; font-size:var(--fs-1); cursor:pointer; }
.extopic.off { cursor:default; color:var(--muted); }
.extopic .exname { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.extopic .exst { color:var(--muted); white-space:nowrap; font-variant-numeric:tabular-nums; }
.extopic[data-status="judged"] .exst { color:var(--text-primary); }
.extopic .exprov { color:var(--muted); }
.extopic .exnote { grid-column:2 / -1; font-size:12px; color:var(--muted); }
.excontrol, .owncode { display:flex; gap:8px; align-items:baseline; margin-top:12px;
  font-size:var(--fs-1); }
.owncode code { font-size:12px; }
.exsum { margin:12px 0 0; font-size:var(--fs-1); color:var(--muted);
  font-variant-numeric:tabular-nums; }
.msit-go { margin-top:12px; }
.suitecell > summary { cursor:pointer; list-style:none; white-space:nowrap; }
.suitecell > summary::-webkit-details-marker { display:none; }
.suitelist { margin-top:4px; white-space:normal; min-width:260px; max-width:420px; }
.owncode-row { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; }
.owncode-row .owncode { margin-top:0; flex-basis:100%; }
tr.owncode-tr > td { background:var(--accent-soft); padding:10px 12px 12px; }
.suitelist > div { margin-top:2px; }
.suitelist b { font-weight:600; }
.rv { border:1px solid var(--border); border-radius:var(--r-2); padding:12px 14px; margin:10px 0; }
.rv h3 { margin:0 0 4px; font-size:var(--fs-3); }
.rv textarea { width:100%; box-sizing:border-box; min-height:70px; font:inherit;
  font-size:var(--fs-2); color:var(--text-primary); background:var(--surface-1);
  border:1px solid var(--border); border-radius:var(--r-1); padding:8px 10px; margin:6px 0; }
.rv .frm input { min-width:120px; }
.rv .ex { border-left:2px solid var(--border); padding:2px 0 2px 10px; margin:0 0 8px;
  font-size:var(--fs-1); }
.rv .ex .kv { color:var(--muted); font-size:var(--fs-1); }
.kvs { display:flex; flex-wrap:wrap; gap:4px 18px; font-size:var(--fs-1); margin:4px 0; }
.kvs b { font-weight:600; }
.provlist.small dd { font-size:var(--fs-1); }
.dxex { margin-top:10px; font-size:var(--fs-1); }
.dxex > summary { cursor:pointer; color:var(--accent); font-size:var(--fs-1); }
.dxex ul { list-style:none; padding:0; margin:8px 0 0; }
.dxex li { border-left:2px solid var(--border); padding:2px 0 2px 10px; margin:0 0 9px; }
.dxex .q { display:block; }
.dxex .kv { color:var(--muted); font-size:var(--fs-1); }
.dxex .kv b { color:var(--text-secondary); font-weight:550; }
.dxh { font-size:var(--fs-1); font-weight:650; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); margin:16px 0 2px; }
.domhead { font-size:var(--fs-1); font-weight:650; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); margin:18px 2px 8px; }
.domhead .se { text-transform:none; letter-spacing:0; font-weight:400; }
.tchips { display:flex; flex-wrap:wrap; gap:5px; margin:5px 0 2px; }
.tchip { font-size:var(--fs-1); line-height:1.5; border:1px solid var(--border);
  border-radius:var(--r-1); padding:0 6px; color:var(--text-secondary); cursor:help;
  white-space:nowrap; }
.tchip.dom { background:var(--accent-soft); border-color:var(--accent-soft);
  color:var(--accent); }
.tchip.flat { color:var(--warning-text); border-color:var(--warning); }
.tchip.front { border-style:dashed; }
/* a two-state toggle in a .ctrl row — pressed state is not colour alone */
.tgl { font:inherit; font-size:var(--fs-1); cursor:pointer; border-radius:var(--r-1);
  padding:4px 10px; border:1px solid var(--border); background:var(--plane);
  color:var(--text-secondary); }
.tgl.on { background:var(--accent-soft); border-color:var(--accent);
  color:var(--accent); font-weight:600; }
.tgl:disabled { opacity:.45; cursor:not-allowed; }
.tv { display:none; margin-top:12px; } .tv.open { display:block; }
.small { font-size:var(--fs-1); color:var(--text-secondary); }
.up { color:var(--success-text); } .down { color:var(--critical-text); }
.note { border-left:2px solid var(--axis); padding:6px 0 6px 12px; margin:12px 0;
  color:var(--text-secondary); font-size:var(--fs-2); }
.warn { border-left:2px solid var(--warning); padding:6px 0 6px 12px; margin:10px 0;
  color:var(--text-secondary); font-size:var(--fs-2); }
.warn b, .note b { color:var(--text-primary); }
/* a band the page cannot lose: a demo report must say what it is wherever
   it is opened, so it is markup at the top of the document, not a toast */
.pagebanner { border:1px solid var(--warning); border-left-width:4px; border-radius:var(--r-1);
  padding:10px 14px; margin:0 0 14px; font-size:var(--fs-2); font-weight:600;
  color:var(--text-primary); background:color-mix(in srgb, var(--warning) 10%, transparent); }
.pagebanner a { font-weight:500; }
/* the board's checks, folded where they are not about the work in front of
   you. Folded, never dismissed: they are findings */
.warnfold { margin:8px 0; }
.warnfold > summary { cursor:pointer; font-size:var(--fs-1); color:var(--text-secondary);
  padding:4px 0; }
.warnfold[open] > summary { color:var(--text-primary); }
.lb td.model { white-space:nowrap; }
.lb td.model .mname { display:inline-block; max-width:22ch; overflow:hidden;
  text-overflow:ellipsis; vertical-align:bottom; }
.st { display:inline-block; font-size:var(--fs-1); border-radius:999px; padding:2px 9px;
  border:1px solid var(--border); white-space:nowrap; }
.st-done { color:var(--success-text); border-color:var(--success-text); }
.st-failed { color:var(--critical-text); border-color:var(--critical); }
.st-active { color:var(--accent); border-color:var(--accent); }
.st-muted { color:var(--muted); }
@keyframes pulse { 0%,100%{opacity:.35} 50%{opacity:1} }
.st-active::before { content:'●'; margin-right:5px; animation:pulse 1.4s infinite; }
.frm { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:10px; }
.frm > .badge { align-self:center; margin-left:0; }
.frm input, .frm select { font:inherit; font-size:var(--fs-2); color:var(--text-primary);
  background:var(--plane); border:1px solid var(--border); border-radius:var(--r-1);
  padding:6px 10px; }
.frm input:focus, .frm select:focus { outline:2px solid var(--accent-soft);
  border-color:var(--accent); }
/* minmax(0,1fr) + min-width:0: a grid track's default minimum is its content's
   min-content width, so one unbreakable config value (a 900-char JSON string)
   used to push the whole right column to 3500px and scroll the page sideways */
.tr-grid { display:grid; grid-template-columns:minmax(250px,320px) minmax(0,1fr);
  gap:12px; align-items:start; margin-top:12px; }
.tr-grid > * { min-width:0; }
@media (max-width:900px){ .tr-grid { grid-template-columns:1fr; } }
.runlist { max-height:72vh; overflow-y:auto; }
.runrow { display:flex; gap:8px; padding:7px 9px; border-bottom:1px solid var(--grid);
  cursor:pointer; align-items:flex-start; font-size:var(--fs-2); border-radius:var(--r-1); }
.runrow:hover { background:var(--plane); }
.runrow.sel { background:var(--accent-soft); }
.rchip { width:10px; height:10px; border-radius:3px; border:1px solid var(--border);
  flex:none; margin-top:5px; }
.rbody { flex:1; min-width:0; }
.rtop { display:flex; align-items:center; gap:8px; }
.rname { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.rmeta { font-size:var(--fs-1); color:var(--muted); margin-top:1px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.ctrl { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:2px 0 10px; }
.ctrl input[type=range] { width:140px; accent-color:var(--accent); }
.ctrl input[type=search] { font:inherit; font-size:var(--fs-1); color:var(--text-primary);
  background:var(--plane); border:1px solid var(--border); border-radius:var(--r-1);
  padding:5px 9px; width:190px; }
.msec { margin-bottom:10px; }
.msec-h { background:none; border:0; padding:4px 0 6px; font-size:var(--fs-1); font-weight:600;
  color:var(--text-secondary); cursor:pointer; display:flex; gap:6px; align-items:center;
  box-shadow:none; }
.msec-h:hover { background:none; color:var(--text-primary); }
.msec-h::before { content:'▾'; font-size:var(--fs-1); color:var(--muted); }
.msec.closed .msec-h::before { content:'▸'; }
.msec-h .count-note { font-weight:400; }
.diffrow td { background:var(--accent-soft); }
td.cfgv { white-space:pre-wrap; overflow-wrap:anywhere; max-width:440px; }
/* an answer is long; two lines is enough to recognise it, and the rest is one
   click away. -webkit-line-clamp is the only cross-browser way to do this */
.clamp2 { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
          overflow:hidden; }
.radar-grid { display:grid; grid-template-columns:minmax(300px,540px) minmax(0,1fr);
  gap:16px; align-items:start; margin-top:6px; }
@media (max-width:900px){ .radar-grid { grid-template-columns:1fr; } }
.radar .hit { cursor:pointer; }
.radar-tbl th, .radar-tbl td { font-size:var(--fs-1); }
.xbtn { border:0; background:none; padding:0 2px; font-size:var(--fs-3); line-height:1;
  color:var(--muted); cursor:pointer; }
.xbtn:hover { color:var(--critical-text); background:none; }
.lb td input[type=checkbox] { accent-color:var(--accent); margin:0; vertical-align:middle;
  cursor:pointer; }
mark { background:var(--accent-soft); color:var(--text-primary); border-radius:3px;
  padding:0 1px; }
pre.mono { background:var(--plane); border:1px solid var(--border); border-radius:var(--r-1);
  padding:10px 12px; margin:8px 0 0; }
.mx { border-collapse:separate; border-spacing:2px; font-variant-numeric:tabular-nums;
  width:auto; }
.mx th { border:0; font-size:var(--fs-1); padding:3px 6px; text-transform:none; letter-spacing:0; }
.mx th.rowh { text-align:right; max-width:170px; overflow:hidden; text-overflow:ellipsis; }
.mx th.colh span { writing-mode:vertical-rl; transform:rotate(180deg); max-height:120px;
  overflow:hidden; text-overflow:ellipsis; display:inline-block; }
.mx td { border:0; border-radius:var(--r-1); background:var(--plane); width:34px; height:30px;
  text-align:center; font-size:var(--fs-2); cursor:default; }
.mx td.self { background:none; }
.mx td:hover { outline:2px solid var(--accent-soft); }
/* 11b: one tooltip, inverted so it reads in every theme */
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--text-primary); color:var(--surface-1); border-radius:var(--r-1);
  padding:7px 10px; font-family:var(--font-sans); font-size:var(--fs-1);
  box-shadow:0 4px 16px rgba(0,0,0,.22); z-index:70; max-width:280px; }
#tip .v { font-size:var(--fs-1); font-weight:700; color:var(--surface-1); }
#tip .l { color:var(--surface-1); opacity:.85; }
#tip .k { display:inline-block; width:10px; border-top:3px solid; border-radius:2px;
  margin-right:6px; vertical-align:3px; }
svg text { font-family:system-ui,-apple-system,sans-serif; }
.hit { fill:transparent; }
.bar { transition:opacity .12s; }
.dimmed .bar:not(.hot) { opacity:0.3; }
.dimmed text.blab:not(.hot) { opacity:0.35; }
a { color:var(--accent); }
footer { margin-top:28px; font-size:var(--fs-1); color:var(--muted);
  font-family:var(--font-sans); }
/* ---- phase 9a: buttons that say what they are ---- */
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
button.primary:hover { filter:brightness(1.07); background:var(--accent); }
button.secondary { background:var(--surface-1); border-color:var(--accent); color:var(--accent); }
button.quiet { background:none; border-color:transparent; color:var(--accent); }
button.quiet:hover { background:var(--accent-soft); }
button:disabled, button:disabled:hover { opacity:.5; cursor:not-allowed; filter:none;
  background:var(--plane); border-color:var(--border); color:var(--text-secondary); }
.dot-warn::before { content:''; display:inline-block; width:8px; height:8px; border-radius:50%;
  background:var(--warning); margin-right:6px; vertical-align:1px; }
.badge.over { color:var(--warning-text); border-color:var(--warning); }
/* ---- the pager ---- */
.pager { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin:10px 0 2px; }
/* 11h: a phone — the chips one row that scrolls, the pills behind Filters ▾,
   and the pager on one line */
.lbbar.narrow { display:block; }
.lbbar.narrow .chiprow { display:flex; gap:8px; align-items:center; }
.lbbar.narrow .chips { flex:1; min-width:0; flex-wrap:nowrap; overflow-x:auto; scrollbar-width:none; }
.lbbar.narrow .chips::-webkit-scrollbar { display:none; }
.lbbar.narrow .chips .chip-btn { flex:none; }
.lbbar.narrow .chiprow > .pill { flex:none; }
.fsheet { position:fixed; left:0; right:0; bottom:0; z-index:55; max-height:70vh; overflow:auto;
  background:var(--surface-1); border-top:1px solid var(--border); border-radius:var(--r-2) var(--r-2) 0 0;
  box-shadow:0 -12px 32px rgba(0,0,0,.18); padding:12px 16px 20px; }
.fsheet-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.fsheet .pills { display:flex; flex-direction:column; align-items:stretch; gap:8px; margin:0; }
.fsheet .pills .pill { width:100%; text-align:left; justify-content:space-between; }
@media (max-width:720px) {
  .pager { flex-wrap:nowrap; gap:4px; overflow-x:auto; scrollbar-width:none; }
  .pager button { padding:3px 6px; }
  .pager .count-note { white-space:nowrap; }
}
.pager select { font:inherit; font-size:var(--fs-1); color:var(--text-primary);
  background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-1); padding:3px 6px; }
.pager button { padding:3px 9px; font-size:var(--fs-1); }
.pager .pgnum[aria-current="page"] { background:var(--accent); border-color:var(--accent);
  color:#fff; font-weight:600; }
/* ---- model search ---- */
.ms { position:relative; flex:2; min-width:240px; }
.ms input { width:100%; box-sizing:border-box; }
.ms-list { position:absolute; top:calc(100% + 4px); left:0; right:0; z-index:40; margin:0;
  padding:4px 0; list-style:none; background:var(--surface-1); border:1px solid var(--border);
  border-radius:var(--r-2); box-shadow:0 10px 28px rgba(0,0,0,.16); max-height:320px; overflow-y:auto; }
.ms-list[hidden] { display:none; }
.ms-item { display:flex; flex-direction:column; gap:1px; padding:6px 11px; cursor:pointer;
  font-size:var(--fs-2); }
.ms-item.active, .ms-item:hover { background:var(--accent-soft); }
.ms-item.over { opacity:.6; }
/* on the board, but its weights are not here: listed, greyed, and it says so */
.ms-item.noweights { color:var(--muted); }
.ms-item.noweights .mono { text-decoration:line-through; }
.ms-sep { padding:6px 11px 2px; font-size:var(--fs-1); font-weight:600; color:var(--muted);
  text-transform:uppercase; letter-spacing:.05em; }
.ms-foot { padding:6px 11px; font-size:var(--fs-1); color:var(--warning-text); border-top:1px solid var(--grid); }
/* ---- the build bar ---- */
.buildbar { position:sticky; top:0; z-index:60; display:flex; gap:10px; align-items:center;
  flex-wrap:wrap; padding:9px 22px; background:var(--accent-soft); border-bottom:1px solid var(--accent);
  font-size:var(--fs-2); backdrop-filter:blur(6px); }
/* ---- dialog ---- */
.dlg-back { position:fixed; inset:0; z-index:70; background:rgba(0,0,0,.38);
  display:flex; align-items:center; justify-content:center; padding:16px; }
.dlg { background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2);
  max-width:540px; width:100%; padding:20px 22px; box-shadow:0 18px 50px rgba(0,0,0,.3); }
.dlg h2 { font-size:var(--fs-3); margin-bottom:6px; }
.dlg ul { margin:6px 0 10px; padding-left:20px; font-size:var(--fs-2); }
.dlg-field { display:flex; flex-direction:column; gap:3px; margin:10px 0; }
.dlg-field input { font:inherit; font-size:var(--fs-2); color:var(--text-primary); background:var(--plane);
  border:1px solid var(--border); border-radius:var(--r-1); padding:6px 10px; }
.dlg-check { display:flex; gap:8px; align-items:center; font-weight:600; margin:10px 0; }
/* a hidden row is hidden, whatever its class says */
.dlg [hidden] { display:none; }
.dlg-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:14px; }
/* ---- 11j: the Review tab — four views, one-line rows, a card in the sheet ---- */
.rvbar { display:flex; align-items:flex-start; justify-content:space-between; gap:12px;
  flex-wrap:wrap; }
.rvbar h2 { margin:0; }
.rvbar .sub { margin:2px 0 0; }
.rvviews { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
table.rvlist td { vertical-align:middle; }
table.rvlist tr.clickrow { cursor:pointer; }
table.rvlist tr.clickrow:hover > td { background:var(--accent-soft); }
.chip-static { display:inline-block; font-family:var(--font-sans); font-size:var(--fs-1);
  border:1px solid var(--border); border-radius:999px; padding:2px 10px; margin:0 6px 6px 0;
  color:var(--muted); }
.focuschips { display:flex; flex-wrap:wrap; align-items:baseline; gap:0 4px; margin:6px 0 2px; }
.rd-skill > summary { cursor:pointer; font-weight:600; font-size:var(--fs-2); }
.rd-skill > summary::-webkit-details-marker { display:none; }
.rd-skill > .spec { margin:6px 0 0; font-size:var(--fs-2); line-height:1.5; }
.rd-spec { width:100%; box-sizing:border-box; min-height:150px; font:inherit;
  font-size:var(--fs-3); line-height:1.45; color:var(--text-primary); background:var(--plane);
  border:1px solid var(--border); border-radius:var(--r-1); padding:10px 12px; margin:4px 0; }
.reader .spec { font-size:var(--fs-3); line-height:1.5; margin:4px 0 10px; }
.rd-acts-row { margin:10px 0 4px; }
.rd-answers { margin:10px 0; }
.rd-answers > summary { cursor:pointer; font-weight:600; font-size:var(--fs-2); }
.rd-answers > summary::-webkit-details-marker { display:none; }
.rd-alist { margin:8px 0 0; padding-left:22px; display:flex; flex-direction:column; gap:12px; }
.rd-alist li { padding-bottom:10px; border-bottom:1px solid var(--border); }
.rd-alist p { margin:0 0 4px; font-size:var(--fs-2); }
.rd-alist .rd-q { color:var(--text-primary); }
.rd-alist .rd-a { color:var(--muted); margin:0 0 4px; font-size:var(--fs-2); }
.rd-alist .rd-atext { display:inline; }
.rd-alist .rd-atext.clamp3 { display:-webkit-box; }
.rd-alist .rd-more { padding:2px 0; font-size:12px; display:block; }
.rd-det { margin-top:14px; }
.rd-det > summary { cursor:pointer; font-size:var(--fs-1); color:var(--muted); }
.npdlg { max-width:640px; }
.nptopics { max-height:46vh; overflow:auto; margin-top:4px; padding-right:4px;
  grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); }
.nptopics { row-gap:10px; }
.nptopics .extopic { cursor:pointer; }
tr.dsfail > td { padding-top:0; border-top:0; }
tr.dsfail .warn { margin:0 0 4px; }
tr.dsfail details > summary { cursor:pointer; list-style:none; color:var(--text-secondary); }
tr.dsfail details > summary::-webkit-details-marker { display:none; }
tr.dsfail details[open] > summary { color:var(--text-primary); }
.dswhy { margin:6px 0 4px; padding-left:22px; }
.dswhy li { margin:2px 0; }
.nptopics .exnote { padding-bottom:2px; }
.nptopics .extopic.off { cursor:default; }
/* ---- the answers, as cards: nothing sideways ---- */
.anslist { display:flex; flex-direction:column; gap:10px; margin-top:8px; min-width:0; }
.anscard { display:grid; grid-template-columns:minmax(0,1fr) 140px; gap:8px 16px;
  border:1px solid var(--border); border-radius:var(--r-2); padding:12px 14px; background:var(--surface-1); }
.anscard .ansmeta { font-size:var(--fs-1); color:var(--text-secondary); margin-bottom:4px; }
.anscard .qa { display:grid; grid-template-columns:52px minmax(0,1fr); gap:5px 10px; }
.anscard .lbl { font-size:var(--fs-1); font-weight:600; color:var(--muted); text-transform:uppercase;
  letter-spacing:.05em; padding-top:2px; }
.anscard .txt { overflow-wrap:anywhere; font-size:var(--fs-2); min-width:0; }
.clamp3 { display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
.anscard .side { text-align:right; }
.anscard .score { font-size:var(--fs-4); font-weight:650; letter-spacing:-.02em; line-height:1.1; }
.anscard .side .badge { margin:4px 0 0 4px; }
.critrow { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
/* where a button took you: a card or row marked for a few seconds */
.landed { outline:2px solid var(--accent); outline-offset:2px; border-radius:var(--r-1); }
.critcells { display:flex; gap:2px; flex-wrap:wrap; min-width:0; }
.critcell { width:14px; height:14px; border-radius:3px; flex:none; box-shadow:inset 0 0 0 1px var(--border); }
.critcell.na { background:none; border:1px dashed var(--axis); box-sizing:border-box; }
.badge.new { color:var(--text-primary); font-weight:600; }
@media (max-width:800px) {
  .anscard { grid-template-columns:minmax(0,1fr); }
  .anscard .side { text-align:left; }
}
/* ---- phase 9b: the header, the tabs, the checks ---- */
.topright { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
button.who { font-weight:600; font-family:var(--font-sans); font-size:var(--fs-1);
  white-space:nowrap; }
#themeBtn { white-space:nowrap; font-family:var(--font-sans); font-size:var(--fs-1);
  font-weight:600; }
.bar-right button, .bar-right summary { min-height:32px; box-sizing:border-box; }
button.who.ask { border-color:var(--warning);
  outline:2px solid color-mix(in srgb, var(--warning) 35%, transparent); }
.who-edit { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.who-edit input { font:inherit; font-size:var(--fs-2); color:var(--text-primary); background:var(--surface-1);
  border:1px solid var(--border); border-radius:var(--r-1); padding:5px 9px; width:150px; }
.who-edit.ask input { border-color:var(--warning); outline:2px solid color-mix(in srgb, var(--warning) 35%, transparent); }
.morewrap { position:relative; display:inline-block; }
.moremenu { min-width:200px;
  background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2); padding:4px;
  box-shadow:0 10px 28px rgba(0,0,0,.16); display:flex; flex-direction:column; }
.moremenu[hidden] { display:none; }
/* a popover panel lives on the body, so no scroller can clip it — #tabs
   scrolls sideways on a phone, and that clipped More ▾ to 37 px */
.pop { position:fixed; z-index:100; overflow:auto; overscroll-behavior:contain; }
.pop.whopop { min-width:260px; gap:8px; padding:12px; }
.pop.whopop p { margin:0; color:var(--text-secondary); }
.moremenu [role=menuitem] { border:0; background:none; text-align:left; border-radius:var(--r-1);
  padding:7px 10px; color:var(--text-primary); font-size:var(--fs-2); text-decoration:none; }
.moremenu [role=menuitem]:hover, .moremenu [role=menuitem]:focus { background:var(--accent-soft); }
.moremenu [role=menuitem][aria-current] { font-weight:600; }
.moremenu [role=menuitem][hidden] { display:none; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px;
  vertical-align:1px; flex:none; }
.dot.ok { background:var(--good); }
.dot.warn { background:var(--warning); }
/* 11e: a warning dot and a warning badge are not a warning paragraph — the
   paragraph's padding, margin and left rule made the checks pill 40px tall
   and the "provisional" badge a 30px box */
.dot.warn { padding:0; margin:0 7px 0 0; border:0; max-width:none; }
.dot.info { background:var(--axis); }
details.checks { margin:10px 0 0; font-size:var(--fs-2); }
details.checks > summary { cursor:pointer; list-style:none; display:inline-flex; align-items:center;
  color:var(--text-secondary); padding:3px 0; }
details.checks > summary::-webkit-details-marker { display:none; }
/* 11m: the three pills in the bar are one control in one rule. The checks
   pill lost its side padding to the details.checks rule above — the same
   weight, and later — and its text touched its border. This one outweighs
   both, so the three cannot drift apart again */
.bar .barpill { padding:4px 12px; min-height:32px; box-sizing:border-box;
  font-family:var(--font-sans); font-size:var(--fs-1); font-weight:600; white-space:nowrap;
  border:1px solid var(--border); border-radius:var(--r-1); }
details.checks > summary .showhide::after { content:'\00a0— show'; color:var(--accent); }
details.checks[open] > summary .showhide::after { content:'\00a0— hide'; }
.checklist { list-style:none; margin:6px 0 0; padding:0; border:1px solid var(--border);
  border-radius:var(--r-2); background:var(--surface-1); }
.checklist .check { display:flex; flex-wrap:wrap; gap:4px 10px; align-items:baseline;
  padding:8px 12px; border-bottom:1px solid var(--grid); }
.checklist .check:last-child { border-bottom:0; }
.check-short { flex:1; min-width:240px; }
.check-more { flex-basis:100%; }
.check-more > summary { cursor:pointer; color:var(--muted); }
.infowrap { position:relative; display:inline-block; margin-left:6px; vertical-align:1px; }
button.info { border:0; background:none; padding:0 3px; font-size:var(--fs-2); color:var(--muted);
  cursor:pointer; line-height:1; }
button.info:hover, button.info[aria-expanded="true"] { color:var(--accent); background:none; }
.infopop { position:absolute; left:0; top:calc(100% + 6px); z-index:35; width:min(420px, 80vw);
  background:var(--surface-1); border:1px solid var(--border); border-radius:var(--r-2); padding:10px 12px;
  font-size:var(--fs-1); font-weight:400; color:var(--text-secondary); line-height:1.5;
  box-shadow:0 10px 28px rgba(0,0,0,.16); white-space:normal; }
.infopop[hidden] { display:none; }
/* ---- phase 9c: every action answers, every table fits ---- */
.toasts { position:fixed; right:18px; bottom:18px; z-index:80; display:flex; flex-direction:column;
  gap:8px; align-items:flex-end; max-width:min(440px, calc(100vw - 36px)); }
.toast { display:flex; gap:10px; align-items:center; background:var(--text-primary);
  color:var(--plane); border-radius:var(--r-2); padding:10px 12px 10px 14px; font-size:var(--fs-2);
  box-shadow:0 10px 28px rgba(0,0,0,.25); }
.toast a, .toast button.quiet { color:var(--plane); font-weight:600; text-decoration:underline; }
.toast button.quiet:hover { background:rgba(255,255,255,.12); }
.toast .xbtn { color:var(--plane); opacity:.7; }
button.danger { background:var(--critical); border-color:var(--critical); color:#fff; font-weight:600; }
td.rowacts { white-space:nowrap; }
.frm.fields { align-items:flex-end; }
.fld { display:flex; flex-direction:column; gap:3px; min-width:0; }
.fld-label { font-size:var(--fs-1); font-weight:600; color:var(--text-secondary); }
.fld-text { font-size:var(--fs-2); padding:5px 0; }
/* the Leaderboard: the model column stays put, and (11c) a score and its
   error share one line — the unit is in the header */
table.lb th.model, table.lb td.model { position:sticky; left:32px; z-index:1;
  background:var(--surface-1); }
table.lb td.num .se { display:inline; font-size:var(--fs-1); }
table.lb thead tr.names th { white-space:nowrap; vertical-align:bottom; }
tr.duprow td { background:var(--plane); }
tr.duprow td.model { background:var(--plane); padding-left:22px; }
button.duptoggle { display:inline; padding:0 4px; font-size:var(--fs-1); }
details.colmenu { position:relative; display:inline-block; }
details.colmenu > summary { list-style:none; cursor:pointer; }
details.colmenu > summary::-webkit-details-marker { display:none; }
.colmenu-list { position:absolute; z-index:40; top:calc(100% + 4px); left:0; min-width:220px;
  display:flex; flex-direction:column; gap:3px; padding:10px 12px; background:var(--surface-1);
  border:1px solid var(--border); border-radius:var(--r-2); box-shadow:0 10px 28px rgba(0,0,0,.16); }
/* Provenance: long ids wrap, the model column stays */
table.prov th, table.prov td { white-space:normal; font-size:var(--fs-1); padding:5px 6px; }
table.prov th { text-transform:none; letter-spacing:0; }
table.prov td .mono, table.prov td { overflow-wrap:anywhere; }
table.prov th.model, table.prov td.model, table.prov th:first-child { position:sticky; left:0;
  background:var(--surface-1); z-index:1; box-shadow:1px 0 0 var(--grid); }
/* the model page: caveats on one line */
.caveats { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:10px 0; }
.caveats .badge { margin:0; }
details.caveat-why { display:inline-block; }
details.caveat-why > summary { cursor:pointer; color:var(--accent); font-size:var(--fs-1); }
details.caveat-why[open] { display:block; flex-basis:100%; }
/* ---- phase 9d: one visual system ---- */
:where(button, a, input, select, textarea, summary, [tabindex]):focus-visible {
  outline:2px solid var(--accent); outline-offset:2px; }
/* the ring a pointer never asked for (11k) */
[data-noring]:focus, [data-noring]:focus-visible { outline:none; }
button.secondary { background:var(--surface-1); }
.badge.warn { padding:0 5px; margin:0 0 0 6px; border-width:1px; border-style:solid;
  max-width:none; font-size:var(--fs-1); }
.badge.warn, .badge.taint, .badge.prelim, .badge.over {
  color:var(--warning-text); border-color:color-mix(in srgb, var(--warning) 70%, transparent);
  background:color-mix(in srgb, var(--warning) 10%, transparent); }
.badge.danger { color:var(--critical-text); border-color:color-mix(in srgb, var(--critical) 70%, transparent);
  background:color-mix(in srgb, var(--critical) 10%, transparent); }
/* the paged tables: the header stays in view while the page scrolls */
.lb-wrap.stick { overflow:visible; }
.lb-wrap.stick thead tr:first-child th { position:sticky; top:var(--bar-h); z-index:3;
  background:var(--surface-1); box-shadow:0 1px 0 var(--border); }
.lb-wrap.stick table.lb thead tr:first-child th.model { z-index:4; }
/* the Leaderboard has two header rows: the group row is a fixed height, and
   the names row sticks just under it — both under the page's own bar */
.lb-wrap.stick table.lb thead tr.grp th { height:24px; box-shadow:none; }
.lb-wrap.stick table.lb thead tr.grp + tr.names th { position:sticky; top:calc(var(--bar-h) + 24px);
  z-index:3; background:var(--surface-1); box-shadow:0 1px 0 var(--border); }
.lb-wrap.stick table.lb thead tr.grp + tr.names th.model { z-index:4; }
/* 11h: wider than its card, at any width: the same as narrow below */
.lb-wrap.stick.hscroll { overflow-x:auto; }
.lb-wrap.stick.hscroll thead tr th,
.lb-wrap.stick.hscroll thead tr:first-child th,
.lb-wrap.stick.hscroll table.lb thead tr:first-child th,
.lb-wrap.stick.hscroll table.lb thead tr.grp + tr.names th { position:static; top:auto; }
.lb-wrap.stick.hscroll table.lb thead tr.names th.rank,
.lb-wrap.stick.hscroll table.lb thead tr.names th.model { position:sticky; top:auto; }
/* narrow: the table scrolls sideways in its own box, so a header cannot also
   stick to the page — only the rank and the model stay put, sideways */
@media (max-width:900px) { .lb-wrap.stick { overflow-x:auto; }
  .lb-wrap.stick thead tr th,
  .lb-wrap.stick thead tr:first-child th,
  .lb-wrap.stick table.lb thead tr:first-child th,
  .lb-wrap.stick table.lb thead tr.grp + tr.names th { position:static; top:auto; }
  .lb-wrap.stick table.lb thead tr.names th.rank,
  .lb-wrap.stick table.lb thead tr.names th.model { position:sticky; top:auto; }
  /* the table scrolls sideways here anyway: a name on one line keeps the
     header two lines tall instead of five */
  table.lb thead tr:not(.grp) th.sortable:not(.model) { white-space:nowrap; } }
.empty { display:flex; flex-direction:column; align-items:flex-start; gap:var(--sp-2);
  padding:var(--sp-4); margin:var(--sp-2) 0; border:1px dashed var(--border);
  border-radius:var(--r-2); color:var(--text-secondary); }
.empty p { margin:0; }
.skeleton { display:flex; flex-direction:column; gap:var(--sp-3); padding:var(--sp-3) 0; }
.sk-row { height:14px; border-radius:var(--r-1);
  background:linear-gradient(90deg, var(--plane) 0%, var(--border) 50%, var(--plane) 100%);
  background-size:200% 100%; animation:sk 2.4s ease-in-out infinite; }
@keyframes sk { from { background-position:200% 0; } to { background-position:-200% 0; } }
@media (prefers-reduced-motion: reduce) { .sk-row { animation:none; } }
@media print { .filters, .tabs, button { display:none !important; }
  .view { display:block !important; } body { background:#fff; } }
"""

JS = r"""
'use strict';
// Two lives, one page: embedded JSON = the frozen single-file report;
// an empty (null) data slot = served by service/app.py, which makes this the
// LIVE dashboard — same charts, data fetched, plus a Queue tab.
let DATA = JSON.parse(document.getElementById('data').textContent || 'null');
const LIVE = DATA === null;
const state = {
  q: '', kind: 'all', tab: 'overview',
  model: null,                         // open model detail page, by id (hash-routed)
  read: null,                          // 11g: the open reader, { kind, id, n } (hash-routed)
  src: 'all',                          // All | Models | Checkpoints — a filter, nothing hidden by default
  panelOpen: {},                       // per-task "show all bars" toggles
  sort: { key: 'avg', dir: -1 },
  runsQ: '',                           // the Evals tab query string
  runsSort: { idx: 0, dir: 1 },        // column sort for the metric query table
  provSort: { key: 'date', dir: -1 },  // run-provenance table: newest eval first
  ceSort: { key: 'task', dir: 1 },     // cross-entropy table
  queue: [], qmsg: '',
  qQ: '', qStatus: 'all', qSort: { key: 'id', dir: -1 },   // queue filter/sort
  trSel: [], trColors: {}, trSmooth: 0, trLog: false,      // Training tab
  trRuns: [], trSeries: {}, trFetching: false,
  trQ: '', trStatus: 'all', trOrder: 'updated',            // runs-list filter/sort
  trMetricQ: '', trSecClosed: {}, trSecSig: '',            // metric panels filter / sections
  trXAxis: 'step',                     // benchmark-join chart: step | tokens | compute
  cmpSel: [], cmpColors: {},                               // radar: compared models (≤CMP_MAX)
  accScale: 'raw',                     // task panels: 'raw' | 'chance' (diverging)
  cmpEvicted: '',                      // last model the compare FIFO dropped
  radarNorm: 'chance', radarAxes: 'tasks',                 // radar scaling / axis mode
  avgMode: 'chance',                   // official average: above-chance | raw
  lbHeat: false,                       // leaderboard cells: plain | heat-shaded
  lbAbout: false,                      // "about these benchmarks" panel open
  lbView: 'tasks',                     // leaderboard columns: 'tasks' | 'cats' (MMLU by category)
  rv: { llm: null, proposals: [], datasets: [], loaded: false, msg: '',
        // 11j: which of the four views, and the small per-proposal choices
        view: '', focus: {}, watch: new Set(), answers: {}, answersOpen: 0,
        spread: {}, count: {}, fmt: {}, rejecting: {}, landed: null,
        detailsOpen: {}, specOpen: {}, landedDs: null },                    // Review tab
  ex: { status: null, candidates: [], loaded: false, msg: '', topic: '' },   // Exam tab
  topic: null,                         // open topic page, by slug (hash-routed)
  loop: { rows: null, blocked: '', msg: '', loaded: false, q: '' },          // Loop tab
  judgeHealth: null,                   // {ok, url, why}: is the grading model answering
  mine: new Set((() => { try { return JSON.parse(localStorage.getItem('bench-mine') || '[]'); }
                         catch (e) { return []; } })()),   // queue rows this browser queued
  qMark: null,                         // the queue row to mark, when one of mine moved
  loopSit: { model: '', kind: 'auto', tasks: null, msg: '', busy: false },   // sit-the-exam form
  ans: { model: '', rows: null, loading: false, topic: '', open: {}, seen: {}, fresh: {},
         acuity: 'all', flag: 'all', score: 'all', crit: 'all', sort: 'score' }, // Answers panel
  mdl: { q: '', kind: 'all', src: 'all', family: 'all', judgedOnly: false,
         taintedOnly: false, prelimOnly: false, sort: { key: 'avg', dir: -1 } },  // Models tab
  rvName: '',                          // the name approvals are recorded under (remembered)
  sub: { hf_id: '', kind: 'auto', suite: 'full', submitter: '', note: '', tasks: null },  // Submit form
  // in-place refreshers registered by the mounted tab, so the 5s poll updates
  // data WITHOUT rebuilding the DOM — a full render() mid-keystroke would steal
  // focus from filter inputs and kill slider drags
  trRedraw: null, queueRedraw: null,
};

// ---------- tiny DOM helper: everything dynamic goes through textContent ----------
const NATIVE = new Set(['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA', 'SUMMARY', 'LABEL',
                        'DETAILS', 'OPTION']);
function el(tag, attrs = {}, ...kids) {
  const svg = tag.startsWith('svg:');
  const e = svg ? document.createElementNS('http://www.w3.org/2000/svg', tag.slice(4))
                : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  // phase 9d: a click handler on something that is not a control (a sortable
  // column header, a row) is reachable from the keyboard too — Tab to it,
  // Enter or Space to act
  if (!svg && attrs.onclick && !NATIVE.has(e.tagName) && !e.hasAttribute('tabindex')) {
    e.setAttribute('tabindex', '0');
    if (!/^T[HDR]$/.test(e.tagName) && !e.hasAttribute('role')) e.setAttribute('role', 'button');
    e.addEventListener('keydown', ev => {
      if (ev.target === e && (ev.key === 'Enter' || ev.key === ' ')) { ev.preventDefault(); e.click(); }
    });
  }
  for (const k of kids.flat(2)) if (k !== null && k !== undefined)
    e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
const pct  = (v, d = 1) => v == null ? '—' : (100 * v).toFixed(d) + '%';
// "qwen35-d…-step945": the start and the end of a long name, never just the start
const midTrunc = (s, n) => s.length <= n ? s
  : s.slice(0, Math.ceil((n - 1) * 0.45)) + '…' + s.slice(s.length - Math.floor((n - 1) * 0.55));
const num  = (v, d = 3) => v == null ? '—'
  : (+v).toFixed(d).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
const P    = v => v == null ? '—' : v >= 995e6 ? (v / 1e9).toFixed(v % 1e9 ? 1 : 0) + 'B'
                                  : Math.round(v / 1e6) + 'M';
const cell = (t, m) => (DATA.cells[t] || {})[m];
// natural string order: checkpoint names sort by their numbers ("s300" < "s1000"),
// which plain lexicographic gets wrong the moment step counts cross a digit boundary
const natCmp = (a, b) =>
  String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
// a <select> whose change handler gets the picked value — toolbar shorthand
// 11f: a short fixed list is a Select — the old native <select> is gone
function mkSel(label, opts, cur, onpick) { return Select(label, opts, cur, onpick); }

// ---------------------------------------------------------------------------
// 11f: two dropdowns, both on the shared popover. After this there is no
// native <select> in the view.
//
// Select, for short fixed lists: a button that shows its value and a chevron
// and opens a listbox — ↑ ↓, Home, End, typeahead, Enter, Esc. 36px, radius 6.
// Combobox, for long lists (topics, models): a search field that opens a
// grouped list as you type — the ARIA combobox pattern, aria-activedescendant.
// Both answer the code around them the way the select did: .value, and a
// 'change' event.
// ---------------------------------------------------------------------------
function listKeys(e, box, pick) {
  const opts = [...box.querySelectorAll('[role=option]:not([aria-disabled=true])')];
  if (!opts.length) return;
  const i = opts.indexOf(document.activeElement);
  const go = j => { e.preventDefault(); opts[(j + opts.length) % opts.length].focus(); };
  if (e.key === 'ArrowDown') go(i + 1);
  else if (e.key === 'ArrowUp') go(i < 0 ? opts.length - 1 : i - 1);
  else if (e.key === 'Home') go(0);
  else if (e.key === 'End') go(opts.length - 1);
  else if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault(); if (i >= 0) pick(opts[i].dataset.value); }
  else if (e.key === 'Tab') popClose();
  else if (e.key.length === 1 && /\S/.test(e.key)) {
    // typeahead: the letters typed in the last half second
    box._ta = (Date.now() - (box._taAt || 0) < 500 ? box._ta || '' : '') + e.key.toLowerCase();
    box._taAt = Date.now();
    const hit = opts.find(o => o.textContent.trim().toLowerCase().startsWith(box._ta));
    if (hit) { e.preventDefault(); hit.focus(); }
  }
}

function Select(label, opts, cur, onpick, attrs = {}) {
  const { key: k, ...rest } = attrs;
  const key = 'sel-' + (k || label).replace(/[^a-z0-9]+/gi, '-');
  let list = opts, value = String(cur ?? (opts[0] || [''])[0]);
  const labelOf = v => (list.find(o => String(o[0]) === v) || [, v])[1];
  const shown = el('span', { class: 'sel-v' });
  const btn = el('button', { class: 'sel', type: 'button', 'aria-label': label, 'data-select': label,
    ...rest }, shown, el('span', { class: 'sel-c', 'aria-hidden': 'true', text: '▾' }));
  const show = () => { btn.dataset.value = value; shown.textContent = labelOf(value); };
  Object.defineProperty(btn, 'value', { get: () => value, set: v => { value = String(v); show(); } });
  btn.setOptions = o => { list = o; show(); };
  show();
  const pick = v => {
    const o = list.find(x => String(x[0]) === v);
    if (!o || (o[2] || {}).disabled) return;
    const moved = v !== value;
    value = v; show(); popClose(true);
    if (moved) { btn.dispatchEvent(new Event('change', { bubbles: true })); if (onpick) onpick(v); }
  };
  popover(btn, () => {
    // an option may carry one plain line under its name (`sub`): what it gets you
    const box = el('div', { class: 'moremenu listbox', role: 'listbox', id: 'pop-' + key,
      'aria-label': label }, list.map(([v, l, o = {}], i) => el('div', { role: 'option',
        id: `${key}-o${i}`, tabindex: '-1', 'data-value': String(v), text: o.sub ? null : l,
        title: o.title || null, class: o.sub ? 'has-sub' : null,
        'aria-selected': String(String(v) === value), 'aria-disabled': o.disabled ? 'true' : null,
        onclick: () => pick(String(v)) }, o.sub ? el('span', { class: 'opt-t' },
          el('span', { class: 'opt-l', text: l }),
          el('span', { class: 'opt-sub', 'data-opt-sub': String(v), text: o.sub })) : null)));
    box.addEventListener('keydown', e => listKeys(e, box, pick));
    return box;
  }, { key, menu: false, focus: '[role=option][aria-selected=true], [role=option]' });
  btn.setAttribute('aria-haspopup', 'listbox');
  return btn;
}

// groups: [{ label, options: [{ value, text, right, bar, mark, search }] }]
function Combobox(label, groups, cur, onpick, attrs = {}) {
  const { key: k, placeholder, ...rest } = attrs;
  const key = 'cb-' + (k || label).replace(/[^a-z0-9]+/gi, '-');
  const all = groups.flatMap(g => g.options);
  const textOf = v => (all.find(o => o.value === v) || {}).text || '';
  const input = el('input', { type: 'text', class: 'cbox', role: 'combobox', autocomplete: 'off',
    'aria-label': label, 'aria-autocomplete': 'list', 'aria-expanded': 'false',
    'aria-controls': 'pop-' + key, 'data-combobox': label, 'data-value': cur || '',
    'data-keep': key, placeholder: placeholder || 'search…', value: textOf(cur), ...rest });
  input.dataset.popAnchor = key;              // a render finds the open list again
  const panel = () => (POP.key === key ? POP.panel : null);
  const matches = q => {
    q = q.trim().toLowerCase();
    // a field showing the chosen value is not a search: everything is offered
    if (q === textOf(input.dataset.value).toLowerCase()) q = '';
    return groups.map(g => ({ ...g, options: g.options.filter(o => !q
      || (o.search || o.text).toLowerCase().includes(q)) })).filter(g => g.options.length);
  };
  const setActive = o => {
    const p = panel(); if (!p) return;
    p.querySelectorAll('[role=option].active').forEach(x => x.classList.remove('active'));
    if (!o) { input.removeAttribute('aria-activedescendant'); return; }
    o.classList.add('active');
    input.setAttribute('aria-activedescendant', o.id);
    o.scrollIntoView({ block: 'nearest' });
  };
  const pick = v => {
    input.dataset.value = v;
    input.value = textOf(v);
    close();
    input.dispatchEvent(new Event('change', { bubbles: true }));
    if (onpick) onpick(v);
  };
  const build = () => {
    const found = matches(input.value);
    let i = 0;
    const box = el('div', { class: 'moremenu listbox cblist', role: 'listbox', id: 'pop-' + key,
      'aria-label': label,
      // the field keeps the focus: a press on the list must not take it away
      onmousedown: e => e.preventDefault() },
      found.length ? found.map(g => el('div', { role: 'group', class: 'cbgroup',
          'aria-label': g.label || null },
        g.label ? el('div', { class: 'cbhead', 'aria-hidden': 'true', text: g.label }) : '',
        g.options.map(o => el('div', { role: 'option', id: `${key}-o${i++}`, 'data-value': o.value,
            'aria-selected': String(o.value === input.dataset.value),
            onclick: () => pick(o.value) },
          el('span', { class: 'cb-t', text: o.text }),
          o.mark ? el('span', { class: 'badge warn', text: o.mark }) : '',
          o.right ? el('span', { class: 'cb-r', text: o.right }) : '',
          o.bar != null ? el('span', { class: 'cb-b' }, el('span', {
            style: `width:${(100 * Math.max(0, Math.min(1, o.bar))).toFixed(0)}%` })) : ''))))
      : el('div', { class: 'cbnone small se', text: 'nothing matches' }));
    return box;
  };
  const open = () => {
    if (panel()) { panel().replaceChildren(...build().childNodes); }
    else { popOpen(key, input, build(), { menu: false }); input.focus(); }
    input.setAttribute('aria-expanded', 'true');
    const p = panel();
    setActive(p && (p.querySelector('[role=option][aria-selected=true]')
      || p.querySelector('[role=option]')));
  };
  function close() {
    if (panel()) popClose();
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
  }
  input.addEventListener('input', open);
  input.addEventListener('click', () => { if (!panel()) { input.select(); open(); } });
  input.addEventListener('keydown', e => {
    const p = panel();
    const opts = p ? [...p.querySelectorAll('[role=option]')] : [];
    const i = opts.findIndex(o => o.classList.contains('active'));
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!p) { open(); return; }
      if (!opts.length) return;
      setActive(opts[(i + (e.key === 'ArrowDown' ? 1 : -1) + opts.length) % opts.length]);
    } else if (e.key === 'Home' && p && opts.length) { e.preventDefault(); setActive(opts[0]); }
    else if (e.key === 'End' && p && opts.length) { e.preventDefault(); setActive(opts[opts.length - 1]); }
    else if (e.key === 'Enter') {
      if (p && i >= 0) { e.preventDefault(); pick(opts[i].dataset.value); }
    } else if (e.key === 'Escape') {
      if (p) { e.preventDefault(); e.stopPropagation(); close();
        input.value = textOf(input.dataset.value); }
    } else if (e.key === 'Tab') close();
  });
  input.addEventListener('blur', () => setTimeout(() => {
    // a render replaced this field: the new one owns the list now
    if (!input.isConnected) return;
    if (document.activeElement !== input && !(panel() && panel().contains(document.activeElement))) {
      close(); input.value = textOf(input.dataset.value); }
  }, 120));
  return input;
}

// a model list, as the Combobox shows it: the id, and on the right its
// parameters and how many topics it has been judged on (like the phase-9 search)
function modelGroups(rows) {
  return [{ label: '', options: rows.map(r => {
    const m = DATA.models.find(x => x.id === r.id) || {};
    const k = r.judged ?? Object.keys((m.judge || {}).tasks || {})
      .filter(t => t.startsWith('exam_')).length;
    return { value: r.id, text: r.id, search: `${r.id} ${m.name || ''}`,
             right: r.right || [m.params ? P(m.params) : null, k ? `${k} judged` : null]
               .filter(Boolean).join(' · ') };
  }) }];
}

// the topic list, as the Combobox shows it: grouped by the 8 areas, weakest
// first within each, the score in mono on the right with a tiny bar, and
// "provisional" where the judge's scores are
function topicGroups(topics, scoreOf, provisional) {
  const areas = Object.entries(DATA.meta.areas || {});
  const inArea = new Set();
  const opt = t => { const v = scoreOf ? scoreOf(t) : null;
    return { value: t, text: frName(t), search: frName(t) + ' ' + t,
             right: v != null ? `${(+v).toFixed(2)} / 4` : '', bar: v != null ? v / 4 : null,
             mark: v != null && provisional ? 'provisional' : '' }; };
  const byScore = (a, b) => ((scoreOf && scoreOf(a)) ?? 9) - ((scoreOf && scoreOf(b)) ?? 9)
    || frName(a).localeCompare(frName(b));
  const out = areas.map(([a, ts]) => {
    const here = topics.filter(t => ts.includes(frName(t)));
    here.forEach(t => inArea.add(t));
    return { label: a, options: here.sort(byScore).map(opt) };
  }).filter(g => g.options.length);
  const rest = topics.filter(t => !inArea.has(t));
  if (rest.length) out.push({ label: 'Other', options: rest.sort(byScore).map(opt) });
  return out;
}
// The option count rides on the name as a superscript — mmlu⁴, piqa². It is
// derived from the chance level rather than hand-kept (a task HAS a chance level
// because it is multiple-choice), and it is the fact that stops someone reading
// 52% on a two-option task as a result.
const SUP = { 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const taskLabel = t => {
  const info = DATA.tasks[t] || {};
  return t + (info.options ? (SUP[info.options] || `^${info.options}`) : '')
           + (info.metric ? ` (${info.metric})` : '');
};

// The chip row under a task heading: what it measures, how many options, how
// many of our models have run it, whether it separates them, and where the
// ceiling is. Each chip carries the long version in its title.
function taskChips(t) {
  const i = DATA.tasks[t] || {};
  const out = [];
  if (i.domain) out.push(el('span', { class: 'tchip dom', text: i.domain,
    title: i.desc || '' }));
  if (i.options) out.push(el('span', { class: 'tchip', text: i.options + '-choice',
    title: `multiple choice with ${i.options} options — chance is `
         + `${Math.round(100 / i.options)}%` }));
  if (i.nmodels) out.push(el('span', { class: 'tchip', text: i.nmodels + ' models',
    title: 'models on this board with a score for this task' }));
  // a task where no pair of models differs by more than their combined error is
  // measuring nothing here, and today it looks identical to one that works
  if (i.discriminates === false) out.push(el('span', { class: 'tchip flat',
    text: 'no separation',
    title: 'no pair of models on this board differs by more than their combined '
         + 'standard error — this task is not distinguishing our models, whatever '
         + 'the ranking suggests' }));
  if (i.frontier) out.push(el('span', { class: 'tchip front',
    text: 'frontier ' + pct(i.frontier.v),
    title: `best published score ${pct(i.frontier.v)} — ${i.frontier.src}, as of `
         + `${i.frontier.asof}. A different protocol from ours (n-shot, harness), `
         + `so read it as where the ceiling is, not as a like-for-like gap.` }));
  return out.length ? el('div', { class: 'tchips' }, ...out) : '';
}

// Source is a FILTER like Base/Instruct, never a default hide: everything a
// friend uploads stays in every comparison. Checkpoints are marked instead —
// hollow bars in the panels, a dashed ckpt badge in the tables — and ranking
// does the tidying on its own, since a chance-level checkpoint sorts to the
// tail and long panels open on their best 12.
function visible() {
  const q = state.q.toLowerCase();
  return DATA.models.filter(m =>
    (state.src === 'all' || m.source === state.src) &&
    (state.kind === 'all' || m.kind === state.kind) &&
    (!q || m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q) ||
     m.family.includes(q)));
}
const anyCk = () => DATA.models.some(m => m.source === 'artifact');
// the one number allowed to rank models, or null. Preliminary models have no
// average at all — not a smaller one — so every ranking view drops them.
const officialAvg = m => state.avgMode === 'raw' ? m.avgRaw : m.avg;

// when this model was last evaluated — a judged run counts. SmolLM2-360M was
// judged on 09-21 and the Models tab said 09-20
function lastEval(m) {
  const judged = Math.max(0, ...Object.values((m.judge || {}).tasks || {})
    .map(t => t.judged_at || 0));
  const ran = m.date ? Date.parse(String(m.date)) / 1000 : 0;
  const t = Math.max(judged, isNaN(ran) ? 0 : ran);
  return t ? new Date(t * 1000).toISOString().slice(0, 16) : (m.date || null);
}

const ord = n => { const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
                   return n + (s[(v - 20) % 10] || s[v] || s[0]); };

// Rank over the WHOLE board, never the filtered view: "#1 of 7" that changes
// because someone typed in the search box is not a rank. Preliminary models are
// absent rather than last — they have no number, not a worse one. Memoized per
// render because the avg mode is the only thing that can reorder it.
let _rankMap = null, _rankKey = '';
function rankOf(m) {
  const key = state.avgMode + ':' + DATA.models.length;
  if (_rankKey !== key) {
    const ranked = DATA.models.filter(x => officialAvg(x) != null && !x.duplicateOf)
                              .sort((a, b) => officialAvg(b) - officialAvg(a));
    _rankMap = new Map(ranked.map((x, i) => [x.id, { n: i + 1, of: ranked.length }]));
    _rankKey = key;
  }
  return _rankMap.get(m.id) || null;
}

// One sentence about a model, entirely derived — nothing here is typed by hand,
// so it cannot go stale or disagree with the table above it. The last clause is
// the one that matters at our model scale: a board full of numbers hides the
// fact that several of them are indistinguishable from guessing.
function modelSentence(m) {
  const out = [];
  out.push(`${m.name} is ${m.source === 'artifact' ? 'an uploaded checkpoint'
                                                   : 'a Hub model'}`
    + `, evaluated as ${m.kind === 'instruct' ? 'instruct-tuned (chat template applied)'
                                              : 'base (no chat template)'}`
    + (m.params ? `, ${P(m.params)} parameters` : '') + '.');
  const r = rankOf(m), a = officialAvg(m);
  const tn = (m.tainted || []).map(tName).join(' and ');
  if (r && a != null) {
    // the date belongs beside the rank, not two sentences later: a rank
    // without one is a claim about now, and this run may be months old
    out.push(`It averages ${pct(a)} `
      + `${state.avgMode === 'raw' ? 'raw' : 'above chance'} over the ${m.nreq} `
      + `required tasks, ranking ${ord(r.n)} of ${r.of} ranked models here`
      + (m.date ? ` — evaluated ${String(m.date).slice(0, 10)}.` : '.'));
    // the multiple-choice average can be intact while an exam topic is not
    if (tn) out.push(`Its training consumed a dataset derived from ${tn} diagnostics, so that `
      + 'score is shown and never ranked.');
  } else if (m.duplicateOf)
    out.push(`It gave the same answers as ${m.duplicateOfName} on every ranked task — the `
      + `same run submitted twice — so it is ranked once, as ${m.duplicateOfName}.`);
  else if (tn && !(m.missing || []).length)
    out.push(`Its training consumed a dataset derived from ${tn} diagnostics, so ${tn} is `
      + 'excluded from its averages and never ranked; the score itself stands and is shown.');
  else
    out.push(`It has ${m.nhave} of ${m.nreq} required tasks, so it is preliminary `
      + `and carries no overall rank`
      + ((m.missing || []).length ? ` — still missing ${m.missing.join(', ')}.` : '.')
      + ((m.tainted || []).length ? ` ${m.tainted.map(tName).join(', ')} is excluded as well: `
         + 'its training data was derived from that task\'s diagnostics.' : ''));
  // "best at" is only a claim on a task that separates anybody. Leading a task
  // where no pair of models differs by more than their combined error is an
  // artifact of the sort order, and it reads as praise — the weakest model on
  // the board was being credited with winning the one task we flag as flat.
  const bestAt = DATA.accTasks.filter(t => {
    if ((DATA.tasks[t] || {}).discriminates === false) return false;
    const mv = (cell(t, m.id) || {}).v;
    if (mv == null) return false;
    const vs = DATA.models.map(x => (cell(t, x.id) || {}).v).filter(v => v != null);
    return vs.length > 1 && mv === Math.max(...vs);
  });
  // capped: a sentence naming eight tasks is a list wearing punctuation
  const few = (xs, n = 3) => xs.slice(0, n).join(', ')
    + (xs.length > n ? ` and ${xs.length - n} more` : '');
  if (bestAt.length) out.push(`Best on the board at ${few(bestAt)}.`);
  // not "scored low" — not distinguishable from guessing, which is a different
  // and much more useful claim
  const atChance = DATA.accTasks.filter(t => {
    const i = DATA.tasks[t] || {}, c = cell(t, m.id);
    return c && i.chance > 0 && (c.v - 1.96 * (c.se || 0)) <= i.chance;
  });
  if (atChance.length)
    out.push(`Not statistically above chance on ${few(atChance)}.`);
  if (lastEval(m)) out.push(`Last evaluated ${String(lastEval(m)).slice(0, 10)}.`);
  return out.join(' ');
}
// the same run submitted twice: both rows stay, one rank. Naming the row it
// duplicates is the whole point — "duplicate" alone would send someone hunting
const dupBadge = m => m.duplicateOf
  ? el('span', { class: 'badge', 'data-duplicate': m.duplicateOf,
      title: `the same answers as ${m.duplicateOfName} on every ranked task — `
        + `${m.duplicateWhy || 'identical'}. The same run submitted twice: shown, not `
        + 'ranked, nothing deleted.',
      text: `duplicate of ${m.duplicateOfName}` })
  // not merged, and the page says why rather than leaving two rows that look
  // the same sitting next to each other with no explanation
  : m.nearDuplicateOf
  ? el('span', { class: 'badge', 'data-near-duplicate': m.nearDuplicateOf,
      title: `prints the same score as ${m.nearDuplicateOfName} on every ranked task, but is `
        + `not the same run: ${m.nearDuplicateWhy}. Both are ranked.`,
      text: `same scores as ${m.nearDuplicateOfName}` })
  : null;
const prelimBadge = m => m.official ? null
  : el('span', { class: 'badge prelim',
      title: `preliminary — ${m.nhave}/${m.nreq} required tasks`
        + (m.missing && m.missing.length ? `\nmissing: ${m.missing.join(', ')}` : '')
        + '\nPer-task scores are valid; there is no overall average until the '
        + 'required suite completes.',
      text: `prelim ${m.nhave}/${m.nreq}` });
// trained on data derived from a benchmark's diagnostics: the task stays on
// the page and leaves the average. Not a punishment — the only honest way to
// keep a board where some models have been tuned against it.
// an exam task reads as its topic wherever a person sees it
const tName = t => ((DATA.judged || {}).topics || {})[t] || t;
const taintBadge = m => (m.tainted || []).length
  ? el('span', { class: 'badge taint',
      title: `trained on the practice questions of ${m.tainted.map(tName).join(', ')}`
        + ' — that score is shown for this model and never ranked',
      text: 'trained on ' + m.tainted.map(tName).join(', ') + ' practice data' }) : null;
// one warning badge per row: trained-on-it outranks preliminary, and the
// other one rides in its tooltip rather than as a second amber chip
const warnBadge = m => {
  const t = taintBadge(m), p = prelimBadge(m);
  if (t && p) t.title += `\n\nalso ${p.textContent}: ${p.title}`;
  return t || p;
};
const ckBadge = m => m.source === 'artifact'
  ? el('span', { class: 'badge ckpt', title: 'uploaded checkpoint (local artifact)',
                 text: 'ckpt' }) : null;

// ---------- tooltip: one element, filled with textContent, follows pointer ----------
const tip = document.getElementById('tip');
function fillTip(target) {
  let rows;
  try { rows = JSON.parse(target.getAttribute('data-tip')); } catch { return false; }
  if (!Array.isArray(rows) || !rows.length) return false;
  tip.replaceChildren();
  const key = target.getAttribute('data-tipkey');
  const head = el('div', { class: 'v' });
  if (key) head.append(el('span', { class: 'k', style: `border-color:${key}` }));
  head.append(document.createTextNode(String(rows[0])));
  tip.append(head);
  for (const r of rows.slice(1)) tip.append(el('div', { class: 'l', text: String(r) }));
  return true;
}
// 11k: it flips to the other side when it would run off, and then stays 8px
// inside the window whatever happens — a tooltip cut off at the right edge
// is a tooltip nobody can read
function placeTip(x, y) {
  const p = 14, m = 8, w = tip.offsetWidth, h = tip.offsetHeight;
  let left = x + p, top = y + p;
  if (left + w > innerWidth - m) left = x - w - p;
  if (top + h > innerHeight - m) top = y - h - p;
  left = Math.min(Math.max(m, left), Math.max(m, innerWidth - w - m));
  top = Math.min(Math.max(m, top), Math.max(m, innerHeight - h - m));
  tip.style.left = left + 'px'; tip.style.top = top + 'px';
}
// what the tooltip is currently describing, so a screen reader is told the
// same thing the pointer is shown (11b)
let _tipFor = null;
function tipFor(t) {
  if (_tipFor === t) return;
  if (_tipFor && _tipFor.getAttribute('aria-describedby') === 'tip')
    _tipFor.removeAttribute('aria-describedby');
  _tipFor = t;
  if (t) t.setAttribute('aria-describedby', 'tip');
}
function hideTip() { tip.style.opacity = 0; tipFor(null); }
document.addEventListener('pointermove', e => {
  const t = e.target.closest('[data-tip]');
  if (!t || !fillTip(t)) { hideTip(); return; }
  tip.style.opacity = 1; tipFor(t); placeTip(e.clientX, e.clientY);
});
document.addEventListener('focusin', e => {
  const t = e.target.closest('[data-tip]');
  if (!t || !fillTip(t)) return;
  const r = t.getBoundingClientRect();
  tip.style.opacity = 1; tipFor(t); placeTip(r.right, r.bottom);
});
document.addEventListener('focusout', hideTip);

// hover-sync: pointing at a model anywhere highlights it everywhere in the view
document.addEventListener('pointerover', e => {
  const t = e.target.closest('[data-model]');
  const view = document.getElementById('view');
  view.classList.toggle('dimmed', !!t);
  view.querySelectorAll('.hot').forEach(n => n.classList.remove('hot'));
  if (t) view.querySelectorAll(`[data-model="${CSS.escape(t.getAttribute('data-model'))}"]`)
             .forEach(n => n.classList.add('hot'));
});

// ---------- shared SVG pieces ----------
function niceTicks(hi, want = 4) {
  const raw = hi / want, mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || mag;
  const out = []; for (let v = 0; v <= hi + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}

// One panel: horizontal bars, one hue (identity is the label, not the color —
// eleven models cannot share eight distinguishable hues), whiskers = ±1 stderr,
// chance line where the task has one.
function barPanel(task, models, opts) {
  const info = DATA.tasks[task] || {};
  const lower = !!opts.lower;
  const rows = models.map(m => ({ m, c: cell(task, m.id) })).filter(r => r.c)
                     .sort((a, b) => lower ? a.c.v - b.c.v : b.c.v - a.c.v);
  const shots = [...new Set(rows.map(r => r.c.shots).filter(s => s != null))];
  const ns    = [...new Set(rows.map(r => r.c.n).filter(n => n != null))];
  const hasChance = info.chance != null && info.chance > 0 && !lower;
  // "above chance" is a statistical claim, not a pixel one: the bar must clear
  // the chance line by more than 1.96 standard errors to count
  const noisy = c => hasChance && (c.v - 1.96 * (c.se || 0)) <= info.chance;
  const above = hasChance ? rows.filter(r => !noisy(r.c)).length : 0;
  // missing models: name them, and say WHY when the answer is "quick suite"
  const missing = models.filter(m => !cell(task, m.id));
  const missTip = missing.length
    ? 'missing: ' + missing.slice(0, 12).map(m => m.name).join(', ')
      + (missing.length > 12 ? ` +${missing.length - 12} more` : '')
      + (missing.every(m => m.source === 'artifact')
         ? ' — checkpoints ran the quick suite; resubmit with suite=full to fill this panel'
         : '')
    : null;
  const panel = el('div', { class: 'panel' },
    el('h3', { text: taskLabel(task) }),
    taskChips(task),
    el('div', { class: 'pmeta', title: missTip, text:
      (lower ? 'lower is better' : 'higher is better')
      + (shots.length ? ` · ${shots.length > 1 ? 'MIXED n-shot!' : shots[0] + '-shot'}` : '')
      + (ns.length === 1 ? ` · ${ns[0]} items` : '')
      + (hasChance && rows.length > 1
         ? ` · ${above ? above + ' of ' + rows.length + ' clearly above chance'
                       : 'none clearly above chance'}` : '')
      + (missing.length ? ` · ${missing.length} model(s) missing${
           missing.every(m => m.source === 'artifact') ? ' (quick-suite)' : ''}` : '') }));
  if (!rows.length) { panel.append(el('p', { class: 'small', text: 'no data' })); return panel; }

  // long lists collapse to the best 12 — a wall of forty chance-level bars is
  // scrolling, not reading; the button below the axis brings the rest back
  const CAPN = 12;
  const capped = !state.panelOpen[task] && rows.length > CAPN + 2;
  const shown = capped ? rows.slice(0, CAPN) : rows;

  // "vs chance" is a DIVERGING bar centred on the chance line instead of on
  // zero — the form for "above/below a baseline". It is the honest fix for
  // models bunched at the floor: at 26%, 28% and 31% on a 0–100% axis they are
  // three bars of the same length, while against chance they are 1%, 4% and 8%
  // of the available headroom and the difference is finally visible. (A logit
  // axis would separate them too, but bar length has to stay proportional to
  // its baseline — a non-linear axis on a bar chart encodes a lie.)
  const div = hasChance && state.accScale === 'chance';
  const dv  = c => (c.v - info.chance) / (1 - info.chance);
  const dse = c => (c.se || 0) / (1 - info.chance);
  const front = info.frontier;

  const W = 460, LBL = 150, PAD = 56, BH = 15, GAP = 7;
  const TOP = (hasChance || (front && div)) ? 20 : 8;   // headroom for a rule label
  const plotW = W - LBL - PAD;
  const H = shown.length * (BH + GAP) + TOP + 16;
  let X, base, ticks, tickFmt;
  if (div) {
    // the axis runs from the worst model to perfect, so the frontier reference
    // always fits on it — which is why the frontier line is drawn in this mode
    // and only chipped in the raw one
    const fd = front ? dv({ v: front.v, se: 0 }) : 0;
    const dlo = Math.min(0, ...shown.map(r => dv(r.c) - dse(r.c))) * 1.08;
    const dhi = Math.max(0.05, fd, ...shown.map(r => dv(r.c) + dse(r.c))) * 1.05;
    X = v => LBL + plotW * (Math.max(dlo, Math.min(v, dhi)) - dlo) / (dhi - dlo);
    base = X(0);
    ticks = [];
    for (let v = Math.ceil(dlo / 0.25) * 0.25; v <= dhi + 1e-9; v += 0.25)
      ticks.push(+v.toFixed(4));
    tickFmt = v => (v > 0 ? '+' : '') + Math.round(100 * v) + '%';
  } else {
    const maxv = Math.max(...shown.map(r => r.c.v + (r.c.se || 0)), info.chance || 0);
    // scale to the data, not to a fixed floor — gsm8k at 2% must not be squashed
    // into an axis drawn for 25%-chance tasks
    const hi = lower ? maxv * 1.15
                     : Math.min(1, Math.max(maxv * 1.15, (info.chance || 0) * 1.25, 0.05));
    X = v => LBL + plotW * Math.max(0, Math.min(v, hi)) / hi;
    base = LBL;
    ticks = niceTicks(hi, 4);
    tickFmt = v => lower ? num(v, 2) : Math.round(100 * v) + '%';
  }
  const fmt = lower ? (v => num(v, 3)) : (v => pct(v));
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
                              role: 'img', 'aria-label': task });
  for (const t of ticks) {
    svg.append(el('svg:line', { x1: X(t), y1: TOP - 4, x2: X(t), y2: H - 18,
      stroke: 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('svg:text', { x: X(t), y: H - 5, 'font-size': 10,
      fill: 'var(--muted)', 'text-anchor': 'middle', text: tickFmt(t) }));
  }
  // the chance rule: in raw mode it sits where chance falls on the axis, in
  // diverging mode it IS the axis origin
  if (hasChance) {
    const cx = div ? base : X(info.chance);
    svg.append(el('svg:line', { x1: cx, y1: 14, x2: cx, y2: H - 18,
      stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
    svg.append(el('svg:text', { x: cx + 4, y: 10, 'font-size': 9.5,
      fill: 'var(--muted)',
      text: 'chance ' + Math.round(100 * info.chance) + '%' }));
  }
  if (front && div) {
    const fx = X(dv({ v: front.v, se: 0 }));
    svg.append(el('svg:line', { x1: fx, y1: 14, x2: fx, y2: H - 18,
      stroke: 'var(--axis)', 'stroke-width': 1, 'stroke-dasharray': '1 3' }));
    svg.append(el('svg:text', { x: fx - 4, y: 10, 'font-size': 9.5,
      fill: 'var(--muted)', 'text-anchor': 'end',
      text: 'frontier ' + Math.round(100 * front.v) + '%' }));
  }
  // rounded at the data end, square at the baseline — and mirrored when a bar
  // points left, which only happens below chance
  const barPath = (x0, x1, yy, h) => {
    const dir = x1 >= x0 ? 1 : -1;
    const w = Math.max(2, Math.abs(x1 - x0)), r = Math.min(4, w), tip = x0 + dir * w;
    return dir > 0
      ? `M${x0},${yy} H${tip - r} q${r},0 ${r},${r} V${yy + h - r} q0,${r} -${r},${r} H${x0} Z`
      : `M${x0},${yy} H${tip + r} q${-r},0 ${-r},${r} V${yy + h - r} q0,${r} ${r},${r} H${x0} Z`;
  };
  let y = TOP;
  for (const { m, c } of shown) {
    const dim = noisy(c);                  // within noise of chance: still there, but quiet
    const isCk = m.source === 'artifact';  // hollow = uploaded checkpoint, same hue
    const vx = div ? X(dv(c)) : X(c.v);
    // below chance is the red arm of the diverging pair, and it is not a
    // curiosity: an instruct template applied to a base model puts real models
    // under the line, so the chart has to be able to say so
    const below = div && dv(c) < 0;
    const hue = below ? 'var(--s8)' : 'var(--s1)';
    // 18, not 22: at 11.5px a 22-character id runs past the panel's left edge and
    // the first letters are simply cut off. The full id is on hover and in the
    // row tooltip, so the gutter is the constraint, not the information.
    // cut in the MIDDLE: "qwen35-delta-moe-…" named both checkpoints of one
    // run identically; their difference is at the end of the name
    const name = midTrunc(m.name, 18);
    svg.append(el('svg:text', { x: LBL - 8, y: y + BH * 0.75, 'font-size': 11.5,
      fill: dim ? 'var(--muted)' : 'var(--text-secondary)', 'text-anchor': 'end', class: 'blab',
      'data-model': m.id, 'data-full-name': m.name, text: name },
      el('svg:title', { text: m.name === m.id ? m.id : `${m.name}\n${m.id}` })));
    svg.append(el('svg:path', { class: 'bar', 'data-model': m.id,
      d: barPath(base, vx, y, BH),
      opacity: dim ? 0.45 : null,
      'fill-opacity': isCk ? 0.28 : null,
      stroke: isCk ? hue : null,
      'stroke-width': isCk ? 1.2 : null,
      fill: hue }));
    if (c.se > 0 && !lower) {
      const lo = div ? X(dv(c) - dse(c)) : X(Math.max(0, c.v - c.se));
      const hx = div ? X(dv(c) + dse(c)) : X(c.v + c.se);
      const cy = y + BH / 2;
      svg.append(el('svg:line', { x1: lo, y1: cy, x2: hx, y2: cy,
        stroke: 'var(--text-primary)', 'stroke-width': 1.4, opacity: 0.55 }));
      for (const xx of [lo, hx])
        svg.append(el('svg:line', { x1: xx, y1: cy - 3.5, x2: xx, y2: cy + 3.5,
          stroke: 'var(--text-primary)', 'stroke-width': 1.4, opacity: 0.55 }));
    }
    // the label follows the bar's OWN transform (it floated off the end of every
    // diverging bar otherwise) and flips to the outside when the bar points left
    // a left-pointing bar leaves its row empty to the RIGHT of the origin, so the
    // number goes there. Putting it outside the bar tip instead walks it straight
    // into the model-name gutter, which is where it collided.
    const labX = div
      ? (below ? Math.max(base, X(dv(c) + dse(c))) + 6 : X(dv(c) + dse(c)) + 6)
      : X(c.v + (lower ? 0 : c.se || 0)) + 6;
    svg.append(el('svg:text', { x: labX, y: y + BH * 0.75,
      'font-size': 11, fill: dim ? 'var(--muted)' : 'var(--text-primary)',
      class: 'blab', 'data-model': m.id,
      // always the real score — the axis may be relative, the number never is
      text: fmt(c.v) }));
    const tipRows = [
      fmt(c.v) + (c.se ? ` ± ${lower ? num(c.se, 3) : (100 * c.se).toFixed(1) + ' pts'}` : ''),
      `${task} — ${c.shots != null ? c.shots + '-shot, ' : ''}${c.n != null ? c.n + ' items' : ''}`,
      m.id + (m.params ? ` · ${P(m.params)} params` : '')];
    if (dim) tipRows.splice(1, 0, '≈ chance — not statistically above it');
    if (isCk) tipRows.push('uploaded checkpoint (local artifact)');
    if (lower && info.metric === 'bits_per_byte')
      tipRows.splice(1, 0, `cross-entropy ${num(c.v * Math.LN2, 3)} nats/byte`);
    svg.append(el('svg:rect', { class: 'hit', x: 0, y: y - GAP / 2, width: W,
      height: BH + GAP, tabindex: 0, 'data-model': m.id,
      'data-tip': JSON.stringify(tipRows) }));
    y += BH + GAP;
  }
  panel.append(svg);
  if (rows.length > CAPN + 2)
    panel.append(el('button', { style: 'margin:2px 0 8px;padding:3px 10px;font-size:12px',
      text: capped ? `show all ${rows.length}` : `show best ${CAPN}`,
      onclick: () => { state.panelOpen[task] = !state.panelOpen[task]; render(); } }));
  return panel;
}

// ---------- views ----------
// One score on a full 0–100% track, with the two reference marks that make it
// readable: where guessing sits, and where the published ceiling is. Perplexity
// scales to its own data instead, and carries neither mark.
function scoreBar(t, c) {
  const i = DATA.tasks[t] || {}, W = 200, H = 14;
  const hi = i.lower ? Math.max(c.v * 1.25, 0.1) : 1;
  const X = v => Math.max(0, Math.min(W, (v / hi) * W));
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H,
    class: 'sbar', role: 'img',
    'aria-label': `${t}: ${i.lower ? num(c.v, 3) : pct(c.v)}` });
  svg.append(el('svg:rect', { x: 0, y: H / 2 - 3, width: W, height: 6, rx: 3,
    fill: 'var(--grid)' }));
  svg.append(el('svg:rect', { x: 0, y: H / 2 - 3, width: Math.max(2, X(c.v)),
    height: 6, rx: 3, fill: 'var(--s1)' }));
  if (!i.lower && i.chance > 0)
    svg.append(el('svg:line', { x1: X(i.chance), y1: 1, x2: X(i.chance), y2: H - 1,
      stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '2 2' },
      el('svg:title', { text: `chance ${pct(i.chance)}` })));
  if (!i.lower && i.frontier)
    svg.append(el('svg:line', { x1: X(i.frontier.v), y1: 1, x2: X(i.frontier.v), y2: H - 1,
      stroke: 'var(--axis)', 'stroke-width': 1.5 },
      el('svg:title', { text: `frontier ${pct(i.frontier.v)} — ${i.frontier.src}, `
        + `as of ${i.frontier.asof}; a different protocol from ours` })));
  return svg;
}

// ---------- per-item diagnosis ----------
// Reads what scripts/diagnose.py wrote from the harness's --log_samples output.
// Two things make this section worth acting on:
//
//   * Every item is assigned by a hash of its content to a leaderboard half or
//     a diagnosis half, and ONLY the diagnosis half appears here. So a model
//     retrained on anything read below still has an uncontaminated number — and
//     if the two halves ever diverge, the divergence is itself the alarm.
//
//   * The headline is the CAUSE, not the item list. The failures that a score
//     cannot show are mostly not knowledge gaps: a model that puts 92% of its
//     answers on two of four slots is not ignorant of the subject, it cannot
//     reach the other two slots, and generating subject data for it is wasted
//     work. Leading with the item list invites exactly that mistake.

const BUCKETS = [
  ['right',           'right',             'var(--s1)'],
  ['near_miss',       'near miss',         'color-mix(in srgb, var(--s1) 60%, var(--plane))'],
  ['wrong',           'wrong',             'color-mix(in srgb, var(--s1) 32%, var(--plane))'],
  ['at_chance',       'no information',    'var(--axis)'],
  ['confident_wrong', 'confidently wrong', 'var(--s2)'],
];
// One accent hue ramped by lightness for "how far from the truth", a neutral
// for "no signal", and a second hue for the one category that is different in
// kind rather than degree. Blue/neutral/orange is the safest CVD pairing there
// is, and every segment also carries its count in the key below it.
const BUCKET_WHY = {
  right: 'scored correct by the harness',
  near_miss: 'the correct option ranked second and within 0.10 probability of the '
    + 'top one — close enough that a nudge might move it',
  wrong: 'wrong, with no cleaner story than that',
  at_chance: 'the top option is barely above uniform: no opinion either way. This is '
    + 'the honest-ignorance case, and the one where more training data is the right answer',
  confident_wrong: 'backs a wrong option confidently AND rates the correct one below '
    + 'chance. Something false was learned, or the prompt format is fighting the model — '
    + 'more data on the subject fixes neither',
};
const slotName = i => i < 26 ? String.fromCharCode(65 + i) : '#' + (i + 1);
const andList = xs => xs.length < 2 ? (xs[0] || '')
  : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];

// Where the answers land, versus where the answer key says they should.
function pickStats(a) {
  if (!a || a.unsupported || !a.picks) return null;
  const n = a.n_options || Object.keys(a.picks).length;
  if (!n) return null;
  const tp = Object.values(a.picks).reduce((x, y) => x + y, 0);
  const tg = Object.values(a.gold || {}).reduce((x, y) => x + y, 0);
  if (!tp) return null;
  const p = [], g = [];
  for (let i = 0; i < n; i++) {
    p.push((a.picks[i] || 0) / tp);
    g.push(tg ? (a.gold[i] || 0) / tg : 1 / n);
  }
  // The highest accuracy ANY model with this pick distribution could reach:
  // sum over slots of min(picked share, correct share). That is exactly
  // 1 - total variation distance, and it is a property of where the answers go,
  // not of what the model knows — which is the whole point of showing it.
  const ceiling = p.reduce((s, v, i) => s + Math.min(v, g[i]), 0);
  const order = p.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]);
  return { n, p, g, ceiling, order, tp, tg };
}

// A score that has not cleared chance is not a measurement of the subject, and
// that fact outranks everything else this section could say about the task. It
// uses the REPORTED score and its harness stderr where one exists, so it tells
// the same story as the flag in the Results table rather than a second one.
function atChanceLine(t, d, mid) {
  const ch = (DATA.tasks[t] || {}).chance;
  if (!(ch > 0)) return null;
  const c = cell(t, mid);
  let v, se, what;
  if (c && c.se) { v = c.v; se = c.se; what = 'The reported score'; }
  else if (d.score_report != null && d.n_report) {
    v = d.score_report; what = 'Its leaderboard half';
    se = Math.sqrt(Math.max(v * (1 - v), 1e-9) / d.n_report);
  } else return null;
  if (v - 1.96 * se > ch) return null;
  return { key: 'atchance', calm: true, text:
    `${what}, ${pct(v)} ±${(100 * se).toFixed(1)}, is not distinguishable from the `
    + `${pct(ch)} you get by guessing. So the breakdown below describes how this model `
    + `guesses on ${t}, not what it knows about it — and no number here is evidence `
    + 'about the subject until the score clears chance.' };
}

// The cause sentences for one task, most conclusive first. Every number in them
// is read off the diagnosis, so none of this can drift from the data.
function diagCauses(t, d, mid) {
  const a = d.answers || {}, out = [];
  if (a.unsupported)
    return [atChanceLine(t, d, mid), { key: 'unsupported', calm: true, text:
      'Per-item analysis does not apply to this task — ' + a.unsupported + '. The '
      + 'score is valid; the breakdown below is limited to right and wrong, and no '
      + 'finding is claimed either way.' }].filter(Boolean);
  const s = pickStats(a), nb = d.n || 1, b = d.buckets || {};
  const cw = (b.confident_wrong || 0) / nb, ac = (b.at_chance || 0) / nb;
  const sr = d.score_report;
  if (a.degenerate)
    out.push({ key: 'degenerate', flag: 'one option only', text:
      `Answers ${slotName(a.top_choice)} on ${pct(a.top_share, 0)} of items. That is `
      + 'not a weak subject — the output distribution has collapsed, and a model that '
      + `never picks anything else scores about chance whatever ${t} data it is given.` });
  else if (a.position_biased && s) {
    // how few slots hold 80% of the picks, and how much of the key is there
    let k = 0, acc = 0;
    while (k < s.n && acc < 0.80) acc += s.order[k++][0];
    const slots = s.order.slice(0, k).map(([, i]) => i).sort((x, y) => x - y);
    const goldThere = slots.reduce((x, i) => x + s.g[i], 0);
    const tight = sr != null && sr >= s.ceiling - 0.05;
    out.push({ key: 'position', flag: 'answer positions', text:
      `${pct(acc, 0)} of its answers land on ${andList(slots.map(slotName))}, where only `
      + `${pct(goldThere, 0)} of the correct answers are. A model answering this way `
      + `cannot score above ${pct(s.ceiling)} on ${t} however much it knows`
      + (tight ? ` — and it scored ${pct(sr)}, so the answer distribution is what is `
                 + 'capping this, not the subject.'
               : `; it scored ${pct(sr)}, so the distribution explains part of the gap `
                 + 'and the subject the rest.') });
  }
  if (a.length_biased)
    out.push({ key: 'length', flag: 'option length', text:
      `Picks the shortest option on ${pct(a.short_pick_rate, 0)} of items, where chance `
      + `is ${pct(a.short_pick_baseline, 0)}. On a likelihood-scored task a short option `
      + 'is cheap to say, so this is a choice about option length rather than content.' });
  if (cw >= 0.35)
    out.push({ key: 'confident', flag: 'confidently wrong', text:
      `Wrong with conviction on ${pct(cw, 0)} of items: it backs another option and `
      + 'rates the correct one below chance. That is a learned falsehood or a prompt '
      + 'format fighting the model, and more data on the subject fixes neither.' });
  if (ac >= 0.40)
    out.push({ key: 'chance', calm: true, text:
      `No opinion at all on ${pct(ac, 0)} of items — the options are near-`
      + `indistinguishable to it. This is the one pattern here that more ${t} training `
      + 'data is the right answer to.' });
  const chance = atChanceLine(t, d, mid);
  if (!out.length && !chance)
    out.push({ key: 'clean', calm: true, text:
      'Nothing anomalous in how the answers are distributed: the picks track the answer '
      + 'key, length is not driving them, and the wrong answers are not confident ones. '
      + 'What is left looks like ordinary subject gaps — see the breakdown below.' });
  if (chance) out.unshift(chance);
  return out;
}

// stacked composition of the buckets; `mini` is the version that rides in a
// <summary> row, which carries no key of its own and so gets titles instead
function dxBar(b, n, mini) {
  const bar = el('div', { class: 'dxbar',
    style: mini ? 'width:120px;height:8px;flex:none' : null });
  for (const [k, label, fill] of BUCKETS) {
    const v = b[k] || 0;
    if (!v) continue;
    bar.append(el('span', { style: `width:${(100 * v / n).toFixed(2)}%;background:${fill}`,
      title: `${label}: ${v} of ${n} (${pct(v / n, 0)})` }));
  }
  return bar;
}

function dxKey(b, n) {
  const row = el('div', { class: 'dxkey' });
  for (const [k, label, fill] of BUCKETS) {
    const v = b[k] || 0;
    if (!v) continue;
    row.append(el('span', { title: label + ' — ' + (BUCKET_WHY[k] || '') },
      el('i', { style: 'background:' + fill }),
      el('span', {}, el('b', { text: String(v) }), ' ', label,
        el('span', { class: 'se', text: ' ' + pct(v / n, 0) }))));
  }
  return row;
}

// picked-vs-correct per answer slot. Zero-anchored, same scale for both series,
// because the entire claim is "these two shapes do not match".
function dxSlots(s) {
  const RH = 22, W = 360, X0 = 26, X1 = 252, H = s.n * RH + 6;
  const hi = Math.max(0.01, ...s.p, ...s.g);
  const wide = v => Math.max(v > 0 ? 1.5 : 0, (v / hi) * (X1 - X0));
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
    style: `max-width:${W}px;height:auto`, role: 'img',
    'aria-label': 'share of answers picked versus share of correct answers, per option slot' });
  for (let i = 0; i < s.n; i++) {
    const y = i * RH + 3;
    svg.append(el('svg:text', { x: 0, y: y + 12, 'font-size': 11,
      fill: 'var(--text-secondary)', text: slotName(i) }));
    svg.append(el('svg:rect', { x: X0, y: y, width: wide(s.p[i]), height: 7, rx: 2,
      fill: 'var(--s1)' },
      el('svg:title', { text: `picked ${slotName(i)} on ${pct(s.p[i], 1)} of items` })));
    svg.append(el('svg:rect', { x: X0, y: y + 9, width: wide(s.g[i]), height: 7, rx: 2,
      fill: 'var(--axis)' },
      el('svg:title', { text: `${slotName(i)} is the correct answer on ${pct(s.g[i], 1)} `
        + 'of items' })));
    svg.append(el('svg:text', { x: X1 + 6, y: y + 13, 'font-size': 10.5,
      fill: 'var(--muted)', text: pct(s.p[i], 0) + ' / ' + pct(s.g[i], 0) }));
  }
  return el('div', {},
    el('div', { class: 'dxkey', style: 'margin:2px 0 4px' },
      el('span', { title: 'the share of items on which the model chose this slot' },
        el('i', { style: 'background:var(--s1)' }), 'picked'),
      el('span', { title: "the share of items on which this slot is the answer key's choice" },
        el('i', { style: 'background:var(--axis)' }), 'correct')),
    svg);
}

function dxExamples(ex) {
  const wrap = el('details', { class: 'dxex' },
    el('summary', { text: 'Show the items it got wrong (practice half only)' }));
  for (const [k, label] of BUCKETS.map(x => [x[0], x[1]])) {
    const items = ex[k];
    if (!items || k === 'right') continue;
    wrap.append(el('div', { class: 'dxh', text: label }));
    wrap.append(el('ul', {}, items.map(e => el('li', {},
      el('span', { class: 'q', text: e.q || '(no question text in the log)' }),
      el('span', { class: 'kv' },
        e.group ? e.group + ' · ' : '',
        e.chose ? [el('b', { text: 'chose' }), ' ' + e.chose + ' · '] : '',
        e.answer ? [el('b', { text: 'answer' }), ' ' + e.answer] : '',
        e.p != null ? ' · p ' + num(e.p, 2) : '')))));
  }
  return wrap;
}

// DIAGNOSE.md's noise floor: a category or subject with fewer leaderboard-half
// items than this carries about ±13 points and is shown greyed, never ranked.
// Mirrors MIN_GROUP_N in scripts/diagnose.py (also in every file's thresholds).
const CAT_MIN_N = 30;

// Categories first, subjects on expand. MMLU's 57 subjects are not the
// categories a person thinks in; "weak in economics" is a sentence someone can
// act on, and it is what the generation phase will be asked about. Weakest
// first, item counts on every row, and the same rule as the subject table: a
// task that has not cleared chance is describing how the model guesses.
// The action that starts the generation pipeline, on the exam topic it is
// about. Disabled WITH the reason on the row: a preliminary judged suite, a
// topic under the noise floor, or a model that wrote nothing must say so and
// never quietly offer data instead.
// 11k: one entry point, and it is the dialog — from the model page, the Loop
// board or the topic page. Until now this link sent the person to the topic
// page, which has had no Propose of its own since 11j moved it into the
// dialog: they landed on a page with nothing to do.
const taskOfTopic = topic => {
  const e = Object.entries(((DATA.judged || {}).topics) || {}).find(([, v]) => v === topic);
  return e ? e[0] : null;
};
const RV_OPEN = ['proposed', 'pending', 'approved'];
function openProposalFor(mid, topic) {
  const task = taskOfTopic(topic);
  return (state.rv.proposals || []).find(p => p.model === mid && RV_OPEN.includes(p.status)
    && (p.task === task || p.category === topic)) || null;
}
// the proposals are the Review tab's, and these rows are not on it: ask once
function rvNeeded() {
  if (LIVE && !state.rv.loaded && netReady()) loadReview();
}
function proposeBtn(mid, topic, gate) {
  rvNeeded();
  const open = openProposalFor(mid, topic);
  if (open) return el('a', { class: 'propose', 'data-review-link': String(open.id),
    href: '#tab=review&read=proposal:' + open.id, text: 'Review it →',
    title: `proposal #${open.id} is ${rvStatusWords(open.status).toLowerCase()}`,
    onclick: e => { if (e.metaKey || e.ctrlKey || e.shiftKey) return;
      e.preventDefault();
      openReader({ kind: 'proposal', id: String(open.id) },
        `[data-review-link="${open.id}"]`); } });
  return el('button', { class: 'propose', 'data-propose-link': topic,
    'data-propose-model': mid, text: 'Propose →',
    onclick: () => npDialog({ model: mid, topic, stay: true,
      returnTo: `[data-propose-link="${CSS.escape(topic)}"]` }) });
}

// the payload's copy of the gate, in words for a row that links to the topic
// page: a data reason refuses; a judge reason is asked about there
function proposeWhy(g) {
  if (!g) return '';
  if ((g.hard || []).length) return el('div', { class: 'propwhy', title: g.why,
    text: 'no proposal: ' + g.hard[0].short });
  if ((g.soft || []).length) return el('div', { class: 'propwhy', title: g.why,
    text: 'the judged suite is preliminary — the topic page says what proposing would mean' });
  return '';
}

function dxCategories(mid, t, v, atChance) {
  const cats = Object.entries(v.categories || {}).filter(([, g]) => g.score_report != null)
    .sort((x, y) => x[1].score_report - y[1].score_report);
  if (!cats.length) return null;
  const wrap = el('div', {}, el('div', { class: 'dxh', text: 'Weakest categories first' }));
  wrap.append(el('p', { class: 'small', style: 'margin:2px 0 6px', text: atChance
    ? 'The score has not cleared chance, so these rows say how the model guesses per '
      + 'category, not what it knows about it. No row below is a subject claim.'
    : `Leaderboard-half items per row. Under ${CAT_MIN_N} is greyed: that few items carry `
      + 'about ±13 points, and ranking them ranks the dice. Open a category for its subjects.' }));
  for (const [name, g] of cats) {
    const dim = g.n_report < CAT_MIN_N;
    const subs = (g.groups || []).map(s => [s, (v.groups || {})[s]])
      .filter(([, x]) => x && x.score_report != null)
      .sort((x, y) => x[1].score_report - y[1].score_report);
    const mm = DATA.models.find(x => x.id === mid);
    const cmp = ((((mm || {}).taintCompare || {})[t] || {}).categories || {})[name];
    const det = el('details', { class: 'dxcat' + (dim ? ' dim' : ''), 'data-cat': name },
      el('summary', {},
        el('span', { class: 'dxcname', text: name }),
        el('span', { class: 'num', text: pct(g.score_report) }),
        el('span', { class: 'se', text: `${g.n_report} items`
          + (dim ? ` · under ${CAT_MIN_N}, noise` : '') }),
        cmp ? el('span', { class: 'se taintdelta', title: 'change against the parent model, '
          + 'on the hidden half (never in the training data) and the practice half',
          text: `vs parent: lb ${dpts(cmp.dReport)} · dx ${cmp.dDiagnose == null ? '—' : dpts(cmp.dDiagnose)}` }) : '',
        dxBar(g.buckets || {}, g.n || 1, true)));
    if (subs.length)
      det.append(el('table', { class: 'dxsub' },
        el('thead', {}, el('tr', {},
          el('th', { text: 'subject' }), el('th', { class: 'num', text: 'score' }),
          el('th', { class: 'num', text: 'items' }), el('th', { text: 'composition' }))),
        el('tbody', {}, subs.map(([s, x]) => el('tr', {
            class: x.n_report < CAT_MIN_N ? 'dim' : null },
          el('td', { text: s }),
          el('td', { class: 'num', text: pct(x.score_report) }),
          el('td', { class: 'num se', text: String(x.n_report) }),
          el('td', {}, dxBar(x.buckets || {}, x.n || 1, true)))))));
    wrap.append(det);
  }
  if (v.unmapped && v.unmapped.length)
    wrap.append(el('p', { class: 'warn' }, el('b', { text: 'Mapping gap: ' }),
      `${v.unmapped.length} group${v.unmapped.length > 1 ? 's' : ''} not in `
      + `scripts/categories.yaml rolled into "General & Multidisciplinary": ${v.unmapped.join(', ')}. `
      + 'Add them to the file and re-run diagnose.py.'));
  return wrap;
}

// The one experiment (DIAGNOSE.md). mmlu_perm re-poses a fixed subset of MMLU
// with the options rotated so the correct answer visits every slot equally —
// the same knowledge asked in a way a position-skewed model can reach. Both
// scores, both errors and the gap in standard errors are on the page; the
// sentence is derived from them and from nothing else.
function permControl(m) {
  const a = cell('mmlu', m.id), b = cell('mmlu_perm', m.id);
  if (!a || !b) return null;
  const ch = (DATA.tasks.mmlu || {}).chance || 0.25;
  const clears = c => c.v - 1.96 * (c.se || 0) > ch;
  const ca = clears(a), cb = clears(b);
  const se = Math.sqrt((a.se || 0) ** 2 + (b.se || 0) ** 2);
  const d = b.v - a.v, z = se ? d / se : null;
  const zs = z == null ? '' : ` — ${Math.abs(z).toFixed(1)} standard errors`;
  let text, calm = true;
  if (cb && !ca) {
    text = `The format was hiding measurable knowledge: with the options rotated the model `
      + `clears chance (${pct(b.v)}) where the standard posing does not (${pct(a.v)})${zs}. `
      + 'The fix is the prompt format, not training data.';
    calm = false;
  } else if (!ca && !cb) {
    text = `The knowledge is not there to hide: rotating the options leaves it at chance `
      + `(${pct(b.v)} against ${pct(a.v)}${zs}). Whatever the answer distribution looks `
      + 'like, the ceiling on its page is not what is capping this model.';
  } else if (ca && cb) {
    text = `Both posings clear chance (${pct(a.v)} standard, ${pct(b.v)} rotated${zs}). `
      + 'The MMLU number is measuring knowledge, not slot preference.';
  } else {
    text = `The standard posing clears chance (${pct(a.v)}) and the rotated one does not `
      + `(${pct(b.v)})${zs}. That is the unexpected direction: check the two runs share a `
      + 'template and a harness build before reading anything into it.';
    calm = false;
  }
  const row = (label, c, ok) => el('tr', {},
    el('td', { text: label }),
    el('td', { class: 'num', text: pct(c.v) }),
    el('td', { class: 'num se', text: c.se ? `±${(100 * c.se).toFixed(1)}` : '—' }),
    el('td', { class: 'num se', text: c.n != null ? String(c.n) : '—' }),
    el('td', { text: ok ? 'clears chance' : 'at chance' }));
  return el('div', { class: 'dxperm' },
    el('div', { class: 'dxh', text: 'The permutation control' }),
    el('table', { class: 'dxsub' },
      el('thead', {}, el('tr', {},
        el('th', { text: 'posing' }), el('th', { class: 'num', text: 'score' }),
        el('th', { class: 'num', text: 'stderr' }), el('th', { class: 'num', text: 'items' }),
        el('th', { text: `vs ${pct(ch)} chance` }))),
      el('tbody', {}, row('mmlu — options as published', a, ca),
                      row('mmlu_perm — options rotated by item', b, cb))),
    el('p', { class: 'dxlead' + (calm ? ' calm' : ''), text: text }),
    el('p', { class: 'small', text: `Difference ${d >= 0 ? '+' : ''}${(100 * d).toFixed(1)} `
      + `points${zs}. mmlu_perm covers a fixed subset of subjects, so read the direction and `
      + 'the chance line, not the decimals.' }));
}

function vDiagnose(m) {
  if (!DATA.meta.anyDiag) return null;
  const d = m.diag;
  if (!d || !Object.keys(d.tasks || {}).length)
    return el('div', { class: 'card' },
      el('h2', { text: 'Diagnose' }),
      note('No per-item diagnosis on file for this model. It appears once '
        + 'scripts/diagnose.py has read this run’s --log_samples output; nothing '
        + 'needs re-evaluating, the per-item outcomes are already on disk.'));

  // payload order first, so the section reads in the same order as Results
  const order = [...DATA.accTasks, ...DATA.pplTasks].filter(t => d.tasks[t]);
  for (const t of Object.keys(d.tasks)) if (!order.includes(t)) order.push(t);

  const causes = {}, flagged = [];
  for (const t of order) {
    causes[t] = diagCauses(t, d.tasks[t], m.id);
    if (causes[t].some(c => !c.calm)) flagged.push(t);
  }

  const card = el('div', { class: 'card' },
    el('h2', { text: 'Diagnose' }),
    el('p', { class: 'sub', text: 'The second opinion, free: why the multiple-choice score is what it is, read off the per-item '
      + 'log. The finding is the cause, not the list of missed questions — most of what '
      + 'a score hides is not a knowledge gap, and the two want opposite responses.' }),
    el('p', { class: 'note', text: 'Every item is split, by a hash of its content, into a '
      + 'hidden half (the score) and a practice half. The split is the same for every model '
      + 'and every run. Only the practice half is shown here. A model retrained on '
      + 'anything below therefore still has an honest leaderboard number — and if the two '
      + 'halves start to diverge, that divergence is the alarm.' }));

  // findings first: the reason someone opened this page
  const lead = flagged.flatMap(t => causes[t].filter(c => !c.calm)
    .map(c => el('p', { class: 'dxlead' }, el('b', { text: taskLabel(t) }), ' — ' + c.text)));
  if (lead.length) {
    card.append(el('div', { class: 'dxh', text: `${lead.length} finding`
      + (lead.length > 1 ? 's' : '') + ' a score cannot show' }));
    card.append(...lead.slice(0, 5));
    if (lead.length > 5)
      card.append(el('p', { class: 'small',
        text: `…and ${lead.length - 5} more, under the benchmarks below.` }));
  } else {
    card.append(el('p', { class: 'dxlead calm', text: 'No distribution-level failure on '
      + 'any task: the answers are shaped like the answer key, and what this model gets '
      + 'wrong it gets wrong for ordinary reasons. Per-task detail below.' }));
  }

  const perm = permControl(m);
  if (perm) card.append(perm);

  card.append(el('div', { class: 'dxh', text: 'By benchmark' }));
  for (const t of order) {
    const v = d.tasks[t], b = v.buckets || {}, n = v.n || 1;
    const s = pickStats(v.answers);
    const flags = causes[t].filter(c => !c.calm);
    const det = el('details', { class: 'dx' },
      el('summary', {},
        el('span', { class: 'dxname', text: taskLabel(t) }),
        el('span', { class: 'se', text: v.score_report != null
          ? pct(v.score_report) + ' on the leaderboard half' : '—' }),
        dxBar(b, n, true),
        ...flags.map(c => el('span', { class: 'dxflag', text: c.flag || c.key }))));
    const body = el('div', { class: 'dxbody' });
    for (const c of causes[t])
      body.append(el('p', { class: 'dxlead calm', text: c.text }));
    body.append(el('p', { class: 'small', style: 'margin:10px 0 0' },
      `${n} items in the log · hidden half ${v.n_report} `
      + `(${pct(v.score_report)}) · practice half ${v.n_diagnose} `
      + `(${pct(v.score_diagnose)})`));
    const c0 = cell(t, m.id);
    if (c0 && c0.n != null && Math.abs(c0.n - n) > Math.max(2, 0.02 * c0.n))
      body.append(el('p', { class: 'warn', text: `The reported score covers ${c0.n} items, `
        + `the per-item log ${n}. Everything below is computed from the log, so if this `
        + 'gap is not one you can account for, the two are not describing the same run.' }));
    if (v.approx_buckets)
      body.append(el('p', { class: 'small', text: 'This task is scored by length-'
        + 'normalised likelihood, but its options are not exposed in a shape this tool can '
        + 'measure, so the buckets are computed on raw likelihoods and are approximate.' }));
    body.append(el('div', { class: 'dxh', text: 'How the items went' }));
    body.append(dxBar(b, n, false), dxKey(b, n));
    if (s) {
      body.append(el('div', { class: 'dxh', text: 'Where the answers went' }));
      body.append(dxSlots(s));
      body.append(el('p', { class: 'small', style: 'margin:6px 0 0',
        text: `Ceiling for this answer distribution: ${pct(s.ceiling)} — the most any `
          + 'model placing its answers this way could score here.' }));
    }
    const groups = Object.entries(v.groups || {})
      .filter(([, g]) => g.score_report != null)
      .sort((x, y) => x[1].score_report - y[1].score_report);
    if (v.categories && Object.keys(v.categories).length)
      body.append(dxCategories(m.id, t, v, causes[t].some(c => c.key === 'atchance')));
    else if (groups.length > 1) {
      body.append(el('div', { class: 'dxh', text: 'Weakest groups first' }));
      body.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'dxsub' },
        el('thead', {}, el('tr', {},
          el('th', { text: 'group' }), el('th', { class: 'num', text: 'score' }),
          el('th', { class: 'num', text: 'items' }), el('th', { text: 'composition' }))),
        el('tbody', {}, groups.slice(0, 12).map(([name, g]) => el('tr', {},
          el('td', { text: name }),
          el('td', { class: 'num', text: pct(g.score_report) }),
          el('td', { class: 'num se', text: String(g.n_report) }),
          el('td', {}, dxBar(g.buckets || {}, g.n || 1, true))))))));
      if (groups.length > 12)
        body.append(el('p', { class: 'small',
          text: `${groups.length - 12} more groups scored at or above these.` }));
    }
    if (v.examples && Object.keys(v.examples).length) body.append(dxExamples(v.examples));
    det.append(body);
    card.append(det);
  }
  if (DATA.meta.diagSalt)
    card.append(el('p', { class: 'small', style: 'margin-top:14px',
      text: 'Split salt ' + DATA.meta.diagSalt + '. Changing it re-splits every benchmark '
        + 'and invalidates every score already published against the old halves.' }));
  return card;
}

// ---------- judged free response ----------
// Rubric scores 0–4 from a local, pinned judge (scripts/judge.py), one card
// per model. Three things the card has to make impossible to miss: whether
// a person has agreed with the judge (kappa), whether longer answers scored
// higher (length bias), and the control result — of the MMLU items this
// model got wrong as multiple choice, how many did it answer correctly when
// asked openly. That last number is what phase 4 depends on.
const SCORE_FILL = ['var(--s8)', 'color-mix(in srgb, var(--s8) 45%, var(--plane))', 'var(--axis)',
                    'color-mix(in srgb, var(--s1) 55%, var(--plane))', 'var(--s1)'];
const frName = t => t === DATA.judged.control ? 'MMLU control (open-ended)'
  : (DATA.judged.topics || {})[t] || t.replace(/^(fr_|exam_)/, '').replace(/_/g, ' ');
// the published topic score is the REPORT half; older files carry only a mean
const pubScore = v => v.score_report != null ? v.score_report : (v.n_report == null ? v.mean : null);

const upFirst = s => s ? s[0].toUpperCase() + s.slice(1) : s;

function jBar(dist, n) {
  const bar = el('div', { class: 'jbar' });
  for (let s = 0; s <= 4; s++) {
    const v = (dist || {})[String(s)] || 0;
    if (v) bar.append(el('span', { style: `width:${(100 * v / n).toFixed(2)}%;background:${SCORE_FILL[s]}`,
      title: `score ${s}: ${v} of ${n}` }));
  }
  return bar;
}

function controlSentence(c) {
  const tot = { n: 0, mc_wrong: 0, knew: 0, didnt: 0, unjoined: 0 };
  for (const v of Object.values(c || {})) for (const k of Object.keys(tot)) tot[k] += v[k] || 0;
  if (!tot.mc_wrong) return { text: `Every one of the ${tot.n} control items this model saw was `
    + 'right as multiple choice, so the control has nothing to compare.', lead: null, tot };
  const share = tot.knew / tot.mc_wrong;
  const lead = share >= 0.5 ? 'Knew it, couldn\'t pick it' : 'Didn\'t know it either way';
  const text = share >= 0.5
    ? `: of the ${tot.mc_wrong} control items this model got wrong as multiple choice, it `
      + `answered ${tot.knew} (${pct(share, 0)}) correctly when asked openly (rubric ≥ 3). The `
      + 'knowledge was there under the format.'
    : `: of the ${tot.mc_wrong} control items wrong as multiple choice, ${tot.didnt} were wrong `
      + `open-ended too; only ${tot.knew} (${pct(share, 0)}) were answered correctly when asked `
      + 'openly. Missing knowledge is the problem here, not the posing.';
  return { lead, text, tot };
}

// Grades on question sets the exam no longer holds: what this model scored on
// the retired law, and when. Shown here and nowhere else — not on the Loop
// board, in an average, a Leaderboard column or a proposal — because a number
// on other questions is not a Law score, however its task was named.
function vEarlier(m) {
  const h = ((m.judge || {}).history || []).filter(e => e && e.task);
  if (!h.length) return null;
  const day = t => t ? new Date(t * 1000).toISOString().slice(0, 10) : 'date not recorded';
  return el('div', { class: 'card', 'data-earlier': String(h.length) },
    el('h2', { text: 'Earlier exams (retired question sets)' }),
    el('p', { class: 'sub', text: 'Judged on questions the exam no longer holds, or before the '
      + 'exam recorded which questions a grade was on. Kept as history: none of these counts '
      + 'toward anything on this board.' }),
    el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
      el('thead', {}, el('tr', {}, el('th', { text: 'topic' }),
        el('th', { class: 'num', text: 'score (hidden questions)' }),
        el('th', { class: 'num', text: 'items' }), el('th', { text: 'judged' }),
        el('th', { text: 'judge' }))),
      el('tbody', {}, h.map(e => el('tr', { class: 'dim', 'data-earlier-task': e.task },
        el('td', { text: e.topic || e.task }),
        el('td', { class: 'num', text: e.score_report != null ? `${num(e.score_report, 2)} / 4`
          : e.mean != null ? `${num(e.mean, 2)} / 4 (all items)` : '—' }),
        el('td', { class: 'num se', text: String(e.n_report != null ? e.n_report : (e.n || '—')) }),
        el('td', { class: 'se', text: day(e.judged_at) }),
        el('td', { class: 'se', text: (e.judge_id || '—') + (e.provisional ? ' · provisional' : '') })))))));
}

function vJudged(m) {
  const J = DATA.judged;
  if (!J || !J.tasks.length) return null;
  const cal = J.calibration;
  const st = m.judgeState || { ok: false, reasons: ['not judged'], current: false };
  const ok = st.ok;
  // a local judge: never ranked, never averaged, and greyed so it cannot be
  // read as a number that counts. The canary below still says whether the
  // weights behind the served id changed between runs.
  const prov = !!(m.judge && m.judge.judge && m.judge.judge.provisional);
  const card = el('div', { class: 'card' },
    el('h2', { text: 'Judged free response — the exam' }),
    // 11h: say what it is, then how it works behind a click
    el('p', { class: 'sub', text: 'Open questions per topic, answered in writing and graded 0–4 '
      + 'against a written rubric.' }),
    el('details', { class: 'howto small', 'data-how-judged': '1' },
      el('summary', { text: 'How this works ▸' }),
      el('p', { class: 'small', text: (prov ? 'The judge is a local model whose id cannot be '
        + 'pinned. ' : 'The judge is an API model pinned to a dated id. ')
        + 'It grades single answers, never pairs. Length is in the rubric and reported below. '
        + 'Thirty fixed scripts are graded again every run, so a change to the judge would show.' })),
    // 11g: the judge runs and the judge file behind these numbers, in the reader
    LIVE && m.judge ? el('p', { class: 'small', 'data-how-graded': m.id },
      readLink({ kind: 'provenance', id: 'judge:' + m.id }, 'How this was graded ▸')) : '');
  // up to five paragraphs of caveats stood above the first number; now one
  // line of badges, and the words behind "why?"
  const caveats = [], why = el('div', { class: 'caveat-text' });
  const caveat = (label, para, attrs = {}) => {
    caveats.push(el('span', { class: 'badge taint', ...attrs, text: label }));
    why.append(para);
  };
  const standing = el('p', { class: ok ? 'note' : 'warn', 'data-judge-state': ok ? 'counts' : 'prelim' },
    el('b', { text: ok ? 'Counts. ' : 'Preliminary. ' }),
    cal ? `Agreement with a person: ${cal.kappa} over ${cal.n} answers `
        + `(the bar is ${J.kappaMin}).` + ' '
        : 'No calibration on file for this judge yet. ',
    ok ? 'Topic scores may enter the separate judged average; they never enter the '
       + 'multiple-choice average.'
       : st.reasons.length ? 'Shown, never ranked, never averaged: ' + st.reasons.join('; ') + '.'
       : 'Shown, never ranked, never averaged.');
  if (ok) card.append(standing);
  else caveat(cal ? 'below the agreement bar' : 'not checked by a person', standing,
    { 'data-caveat': 'calibration' });
  const j = m.judge;
  if (!j) { card.append(note('Not judged yet. Sit the exam from the Queue, with the judged '
    + 'suite.')); return card; }
  if (j.skipped) { card.append(note(j.skipped + '. A judge scores its own family higher; the '
    + 'cell stays empty rather than flattering.')); return card; }
  if (!Object.keys(j.tasks || {}).length) { card.append(note('Not judged on the current exam: '
    + 'every judged run on file was on questions the exam no longer holds (Earlier exams, below). '
    + 'Sit the exam to score it on the current questions.')); return card; }
  if (j.judge.stub) caveat('stub grader', el('p', { class: 'warn', text: 'Graded by the STUB grader — a '
    + 'word-overlap stand-in for plumbing tests. Not a judgement of anything.' }), { 'data-caveat': 'stub' });
  if (prov) caveat('provisional', el('p', { class: 'warn', 'data-provisional': 'judge' },
    el('b', { text: 'Provisional. ' }),
    `${upFirst(j.judge.provisional_reason || 'graded by a local model — not a pinned benchmark')}: `
    + `${j.judge.served_model || j.judge.model} at ${j.judge.base_url || 'a local server'}`
    + (j.judge.weights ? ` (weights ${j.judge.weights})` : '')
    + '. A local server\'s model id is whatever was typed at launch, so these scores are shown '
    + 'greyed, never ranked and never in any average.'), { 'data-caveat': 'provisional' });
  // a rubric is an instrument: one its author has not signed off yet grades,
  // but it does not settle anything, and deleting DRAFT from its heading
  // changes its sha — which is the point, a different rubric is a different
  // instrument and before/after across the change do not compare
  if (j.judge.rubric_status === 'draft') caveat('draft rubric', el('p', { class: 'warn',
    'data-rubric': 'draft' }, el('b', { text: 'Draft rubric. ' }),
    `${(j.judge.rubrics_draft || []).map(frName).join(', ') || 'a topic'} is graded against a `
    + 'rubric its author has not signed off yet, so its scores are a reading, not a result. '
    + 'Sign-off is recorded by removing DRAFT from the rubric\'s heading, which changes its '
    + 'sha: scores from before and after do not compare.'), { 'data-caveat': 'draft' });
  if (j.judge.single_provider_loop) caveat('single provider', el('p', { class: 'warn',
    text: 'Single-provider loop: the judge shares a provider with the exam writer or the '
    + 'generator. Every score here carries that caveat — a judge scores its own family higher.' }),
    { 'data-caveat': 'single-provider' });
  const cn = j.canary;
  const canaryPara = cn && el('p', { class: cn.drifted ? 'warn' : 'small', 'data-canary': cn.drifted ? 'drifted' : 'steady' },
    el('b', { text: cn.drifted ? 'Canary moved. ' : 'Canary steady. ' }),
    `${cn.graded} of ${cn.n} fixed scripts re-graded: mean absolute deviation `
    + `${cn.mad_vs_human} from the human marks` + (cn.mad_vs_previous != null
      ? `, ${cn.mad_vs_previous} from the previous run (limit ${cn.threshold})` : ', first run for this judge')
    + (cn.drifted ? ' — the judge is not the judge it was; these scores are preliminary.' : '.'));
  if (cn && cn.drifted) caveat('canary moved', canaryPara, { 'data-caveat': 'canary' });
  // the whole list of caveats is one line; their words are one click away
  if (caveats.length) card.append(el('div', { class: 'caveats', 'data-caveats': String(caveats.length) },
    ...caveats, el('details', { class: 'caveat-why' }, el('summary', { text: 'why?' }), why)));
  if (cn && !cn.drifted) card.append(canaryPara);

  // per topic, weakest first, on the REPORT half — the diagnose half is never the score
  const cats = J.exam.filter(t => j.tasks[t] && pubScore(j.tasks[t]) != null)
    .sort((a, b) => pubScore(j.tasks[a]) - pubScore(j.tasks[b]));
  // 11l: a topic whose answers never finished has no score — it stays in the
  // table with a dash and says why, rather than vanishing from it
  const voids = J.exam.filter(t => j.tasks[t] && pubScore(j.tasks[t]) == null
    && j.tasks[t].no_score);
  if (voids.length) card.append(el('p', { class: 'warn', 'data-not-scored': String(voids.length) },
    el('b', { text: `${voids.length} topic${voids.length === 1 ? ' was' : 's were'} not scored. ` }),
    upFirst(voids.length && j.tasks[voids[0]].no_score) + '. Sit the exam again: a reasoning '
      + 'model now gets room to answer.'));
  if (cats.length || voids.length) {
    card.append(el('div', { class: 'dxh', text: 'By topic (0–4), weakest first within each area — '
      + 'hidden questions' + (prov ? ' · demo only, not ranked' : '') }));
    // what is true of every row is said once, above the table: a soft gate
    // (the judged suite is preliminary) used to repeat on each one
    const soft = cats.some(t => { const g = j.tasks[t].propose;
      return g && !(g.hard || []).length && (g.soft || []).length; });
    if (LIVE && soft) card.append(el('p', { class: 'propwhy', 'data-prelim-note': '1',
      text: 'The judged suite is preliminary — the topic page says what proposing would mean.' }));
    const areaOf = t => (Object.entries(DATA.meta.areas || {})
      .find(([, ts]) => ts.includes(frName(t))) || ['Other'])[0];
    const areas = [...Object.keys(DATA.meta.areas || {}), 'Other'];
    const row = t => { const v = j.tasks[t];
      const k = cal && cal.per_category && cal.per_category[frName(t)];
      const nr = v.n_report != null ? v.n_report : v.n;
      const g = v.propose;
      const tainted = (m.tainted || []).includes(t);
      return el('tr', { class: prov || v.no_score || nr < CAT_MIN_N ? 'dim' : null,
          'data-topic': frName(t) },
        el('td', {}, frName(t), tainted ? el('span', { class: 'badge taint',
          title: 'this model trained on this topic\'s practice data — the score is shown, and '
            + 'it is not a ranking',
          text: 'trained on it' }) : ''),
        el('td', { class: 'num', 'data-topic-score': frName(t),
            text: pubScore(v) == null ? '—' : `${num(pubScore(v), 2)} / 4` },
          v.no_answer ? el('div', { class: 'se', 'data-no-answer': String(v.no_answer),
            text: `${v.n} answers, ${v.no_answer} no answer` }) : ''),
        el('td', { class: 'num se', text: k ? String(k.kappa) : '—',
          title: k ? `${k.n} human-graded answers in this topic` : 'not calibrated per topic' }),
        // a topic with no score stands on no hidden question: a dash, not "0 ·
        // under 30", which reads as a bank still being written (11l)
        el('td', { class: 'num se', text: v.no_score ? '—'
          : String(nr) + (nr < CAT_MIN_N ? ' · under ' + CAT_MIN_N : '') }),
        el('td', {}, v.no_score ? el('span', { class: 'small se', text: upFirst(v.no_score) })
          : jBar(v.dist, v.n)),
        LIVE ? el('td', {}, g ? proposeBtn(m.id, frName(t), g) : '',
          g && (g.hard || []).length ? proposeWhy(g) : '',
          g && g.caution ? el('div', { class: 'propwhy', text: 'caution — MMLU for this '
            + 'category: ' + g.caution }) : '') : ''); };
    const body = [];
    for (const area of areas) {
      const ts = cats.filter(t => areaOf(t) === area);    // already weakest first
      const vs = voids.filter(t => areaOf(t) === area);
      if (!ts.length && !vs.length) continue;
      const of = ((DATA.meta.areas || {})[area] || []).length;
      // an area mean only for a judge whose scores count — never while provisional
      const mean = !prov && ok && ts.length * 2 >= of
        ? ts.filter(t => !(m.tainted || []).includes(t)).map(t => pubScore(j.tasks[t])) : null;
      body.push(el('tr', { class: 'arearow', 'data-area-row': area },
        el('td', { colspan: LIVE ? 6 : 5 }, el('span', { class: 'eyebrow', text: area }),
          el('span', { class: 'se', text: ` · ${ts.length} of ${of || ts.length} topics judged`
            + (vs.length ? ` · ${vs.length} not scored` : '') }),
          mean && mean.length ? el('span', { class: 'se', 'data-area-mean': area,
            text: ` · mean ${num(mean.reduce((x, y) => x + y, 0) / mean.length, 2)} / 4` }) : '')),
        ...ts.map(row), ...vs.map(row));
    }
    card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-judged-topics': '1' },
      el('thead', {}, el('tr', {}, el('th', { text: 'topic' }), el('th', { class: 'num', text: 'score' }),
        el('th', { class: 'num', text: 'agreement' }),
        el('th', { class: 'num', text: 'hidden questions' }),
        el('th', { text: 'score distribution 0 → 4 (all items)' }),
        LIVE ? el('th', { text: 'what to do' }) : '')),
      el('tbody', {}, body))));
    if (m.judgedAvg != null)
      card.append(el('p', { class: 'small', text: `Judged average ${num(m.judgedAvg, 2)} / 4 over `
        + `${cats.filter(t => !(m.tainted || []).includes(t)).length} topics, hidden questions`
        + ((m.tainted || []).some(t => cats.includes(t))
           ? `, excluding ${(m.tainted || []).filter(t => cats.includes(t)).map(frName).join(', ')} `
             + '— this model trained on data derived from that topic' : '')
        + (ok ? '.' : ' — preliminary until the judge is calibrated.') }));
    card.append(el('p', { class: 'small', text: `A topic with under ${CAT_MIN_N} hidden questions `
      + 'is greyed. Its bank is still being written, on the Exam tab.' }));
    // score vs length: one row per topic, a column per length (11d) — the
    // row-per-bucket table was 148 rows for 37 topics
    const LEN = [['≤20 words', '≤ 20'], ['21–50', '21–50'], ['51–120', '51–120'], ['>120', '> 120']];
    const withLen = cats.filter(t => (j.tasks[t].score_vs_length || []).length);
    if (withLen.length) {
      const drift = withLen.map(t => { const b = j.tasks[t].score_vs_length || [];
        return b.length > 1 ? b[b.length - 1].mean - b[0].mean : 0; });
      const worst = Math.max(...drift);
      card.append(el('div', { class: 'dxh', text: 'Score against answer length' }));
      card.append(el('p', { class: 'small', text: worst >= 1
        ? `Longer answers score up to ${num(worst, 1)} points higher in at least one category — `
          + 'length may be driving the judge. Check the rubric\'s length clause before believing '
          + 'the category scores.'
        : 'No category rewards length by a point or more; the length clause is holding.' }));
      // a neutral grey by the mean on the absolute 0–4 scale: these are
      // provisional scores, so nothing here is a rank tint
      const cellOf = b => {
        if (!b) return el('td', { class: 'num se', text: '—' });
        const few = b.n < 5;
        return el('td', { class: 'num lencell' + (few ? ' few' : ''),
          style: few ? null : `background:color-mix(in srgb, var(--text-primary) `
            + `${(3 + 11 * Math.max(0, Math.min(4, b.mean)) / 4).toFixed(1)}%, var(--surface-1))`,
          title: few ? `${b.n} answers — too few to read` : `${b.n} answers` },
          el('b', { text: num(b.mean, 2) }), el('span', { class: 'se', text: ` · ${b.n}` }),
          few ? el('span', { class: 'se', text: ' few' }) : '');
      };
      const table = el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-length-table': '1' },
        el('thead', {}, el('tr', {}, el('th', { text: 'topic' }),
          LEN.map(([, h]) => el('th', { class: 'num' }, h, el('span', { class: 'unit',
            text: 'words · mean · items' }))))),
        el('tbody', {}, withLen.map(t => { const bs = j.tasks[t].score_vs_length || [];
          return el('tr', { 'data-length-row': frName(t) }, el('td', { text: frName(t) }),
            LEN.map(([k]) => cellOf(bs.find(b => b.bucket === k)))); }))));
      card.append(withLen.length > 10
        ? el('details', { class: 'lenfold', 'data-length-fold': '1' },
            el('summary', { text: `Show the table (${withLen.length} topics)` }), table)
        : table);
    }
    // 11m: the per-topic block — by criterion, its flags and its by-field
    // tables, under one topic picker — is gone from the model page, at
    // masein's request. Each answer card keeps its own criteria strip.
    // the answers themselves, for whichever topic is picked — the same panel
    // the topic page shows, because "see the answers" is the step between a
    // score and knowing what to do about it
    // a topic with no score keeps its answers readable — that is how a
    // person checks what the model did write (11l)
    if (LIVE) card.append(modelAnswers(m, [...cats, ...voids]));
  }
  // the control
  const ctl = j.tasks[J.control];
  if (ctl && ctl.control) {
    const s = controlSentence(ctl.control);
    card.append(el('div', { class: 'dxh', text: 'The control: MMLU asked openly' }));
    card.append(el('p', { class: 'dxlead' + (s.lead && s.lead.startsWith('Knew') ? '' : ' calm') },
      s.lead ? el('b', { text: s.lead }) : '', s.text));
    card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
      el('thead', {}, el('tr', {}, el('th', { text: 'category' }), el('th', { class: 'num', text: 'items' }),
        el('th', { class: 'num', text: 'wrong as MC' }), el('th', { class: 'num', text: 'knew it' }),
        el('th', { class: 'num', text: "didn't know" }))),
      el('tbody', {}, Object.entries(ctl.control).sort((a, b) => b[1].mc_wrong - a[1].mc_wrong)
        .map(([c, v]) => el('tr', {}, el('td', { text: c }),
          el('td', { class: 'num se', text: String(v.n) }),
          el('td', { class: 'num', text: String(v.mc_wrong) }),
          el('td', { class: 'num', text: String(v.knew) }),
          el('td', { class: 'num', text: String(v.didnt) })))))));
    card.append(el('p', { class: 'small', text: `MMLU practice-half items only, asked without `
      + `their options and graded against the gold option's text; "knew it" is a rubric score of `
      + `${j.correct_at} or more.` + (s.tot.unjoined ? ` ${s.tot.unjoined} items had no MMLU run to `
      + 'join to.' : '') }));
  }
  card.append(el('p', { class: 'small', style: 'margin-top:12px' },
    `Judge ${j.judge.id} (${j.judge.provider})`
    + (j.judge.batch_id ? ` · batch ${String(j.judge.batch_id).slice(0, 18)}` : '')
    + (j.judge.weights_sha256 ? ` · weights ${String(j.judge.weights_sha256).slice(0, 12)}` : '')
    + ` · prompt v${j.judge.prompt_version} ${String(j.judge.prompt_sha256 || '').slice(0, 12)} · rubrics `
    + [...new Set(Object.values(j.judge.rubrics || {}).map(r => rubricVersion(r.version)))]
        .join(', ')
    + (st.current ? '' : ` · not the judge this server runs now${J.current ? ` (${J.current.id})` : ''}`)));
  return card;
}

// ---------- close the loop: what did the training teach? ----------
// For a tainted model with its parent on the board: both halves, before and
// after, side by side with their errors, and the sentence the numbers derive.
// "Taught the test" is rendered as a warning — it is the alarm.
const VERDICT = { skill: 'The training taught the skill', test: 'The training taught the test',
                  none: 'Nothing measurable changed', mixed: 'An unusual pattern' };
const dpts = d => (d >= 0 ? '+' : '') + (100 * d).toFixed(1);

function vTaint(m) {
  const tc = m.taintCompare;
  if (!tc || !Object.keys(tc).length) return null;
  const card = el('div', { class: 'card' },
    el('h2', { text: 'What the training taught' }),
    el('p', { class: 'sub', text: 'This model trained on the practice half of an exam topic '
      + 'or a benchmark. The hidden half was never in that data, so '
      + 'it is the honest test: if the training taught the skill, both halves move together; '
      + 'if it taught the test, only the half the generator\'s spec came from moves.' }));
  // the trail backwards: which run, which dataset, which proposal. Every step
  // of the loop that produced this model is one click from here
  const tt = m.taintTrail;
  if (tt) card.append(el('p', { class: 'small', 'data-taint-trail': '1' },
    'From ', el('a', { href: '#tab=training', text: `training run #${tt.run_id}`,
      onclick: e => { e.preventDefault(); state.trSel = [tt.run_id];
        navigate({ tab: 'training', model: null, topic: null }); } }),
    ', which trained on ',
    tt.datasets.map((d, i) => el('span', {}, i ? ', ' : '',
      el('a', { href: `api/datasets/${d}`, target: '_blank', rel: 'noopener',
        text: `dataset #${d}` }))),
    tt.proposals.length ? el('span', {}, ' from ',
      tt.proposals.map((pid, i) => el('span', {}, i ? ', ' : '',
        el('a', { href: '#tab=review', text: `proposal #${pid}`,
          onclick: e => { e.preventDefault(); state.rv.loaded = false;
            navigate({ tab: 'review', model: null, topic: null }); } }), ' ',
        overBadge((tt.over_provisional_judge || {})[String(pid)])))) : '',
    '.'));
  for (const [t, c] of Object.entries(tc)) {
    const rubric = c.scale === 'rubric';
    card.append(el('div', { class: 'dxh', text: rubric ? `${tName(t)} — the exam`
      : `${taskLabel(t)} — multiple choice` }));
    if (c.missing) { card.append(note(c.missing)); continue; }
    const parent = DATA.models.find(x => x.id === c.parent);
    const pm = parent ? parent.name : c.parent;
    const val = v => rubric ? `${num(v, 2)} / 4` : pct(v);
    const err = v => rubric ? ` ±${num(v, 2)}` : ` ±${(100 * v).toFixed(1)}`;
    const dlt = d => rubric ? `${d >= 0 ? '+' : ''}${num(d, 2)}` : dpts(d);
    const half = (label, b, a, d, se) => el('tr', {},
      el('td', { text: label }),
      el('td', { class: 'num', text: val(b.v) }, el('span', { class: 'se', text: err(b.se) }),
        el('span', { class: 'se', text: ` · ${b.n}` })),
      el('td', { class: 'num', text: val(a.v) }, el('span', { class: 'se', text: err(a.se) }),
        el('span', { class: 'se', text: ` · ${a.n}` })),
      el('td', { class: 'num', text: dlt(d) }),
      el('td', { class: 'num se', text: `${(Math.abs(d) / se).toFixed(1)} SE` }));
    card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
      el('thead', {}, el('tr', {}, el('th', { text: 'half' }),
        el('th', { class: 'num', text: `before — ${pm}` }), el('th', { class: 'num', text: `after — ${m.name}` }),
        el('th', { class: 'num', text: rubric ? 'Δ rubric' : 'Δ points' }),
        el('th', { class: 'num', text: 'moved by' }))),
      el('tbody', {},
        half(rubric ? 'hidden questions (the published score, never in the training data)'
                    : 'hidden half (never in the training data)',
             c.before.report, c.after.report, c.dReport, c.seReport),
        half('practice half (the missing skill came from here)', c.before.diagnose, c.after.diagnose, c.dDiagnose, c.seDiagnose)))));
    if (rubric && c.judge) card.append(el('p', { class: 'small',
      text: `Both sides graded by ${c.judge}; a comparison across two judges is two `
        + 'instruments, and the card says so instead of drawing it.' }));
    const warn = c.verdict === 'test' || c.verdict === 'mixed';
    card.append(el('p', { class: warn ? 'warn' : 'dxlead calm', 'data-verdict': c.verdict },
      el('b', { text: VERDICT[c.verdict] + '. ' }), c.text.replace(/^The training taught the (skill|test)\. |^Nothing measurable changed\. |^An unusual pattern[^:]*: /, '')));
    if (c.ratio != null && c.verdict === 'test')
      card.append(el('p', { class: 'small', text: 'Ratio of the two changes (practice / '
        + `hidden): ${num(c.ratio, 1)}.` }));
    const cats = Object.entries(c.categories || {})
      .sort((x, y) => (y[1].dDiagnose ?? 0) - (x[1].dDiagnose ?? 0));
    if (cats.length) {
      card.append(el('div', { class: 'dxh', text: 'By category — the half we never touched' }));
      card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
        el('thead', {}, el('tr', {}, el('th', { text: 'category' }),
          el('th', { class: 'num', text: 'hidden half before → after' }), el('th', { class: 'num', text: 'Δ' }),
          el('th', { class: 'num', text: 'practice half before → after' }), el('th', { class: 'num', text: 'Δ' }),
          el('th', { class: 'num', text: 'items (lb half)' }))),
        el('tbody', {}, cats.map(([name, v]) => el('tr', {
            class: (v.after.n_report || 0) < CAT_MIN_N ? 'dim' : null },
          el('td', { text: name }),
          el('td', { class: 'num', text: `${pct(v.before.score_report)} → ${pct(v.after.score_report)}` }),
          el('td', { class: 'num', text: dpts(v.dReport) }),
          el('td', { class: 'num', text: v.dDiagnose == null ? '—'
            : `${pct(v.before.score_diagnose)} → ${pct(v.after.score_diagnose)}` }),
          el('td', { class: 'num', text: v.dDiagnose == null ? '—' : dpts(v.dDiagnose) }),
          el('td', { class: 'num se', text: String(v.after.n_report) })))))));
      card.append(el('p', { class: 'small', text: `Categories under ${CAT_MIN_N} leaderboard-half `
        + 'items are greyed: a delta there is noise. "We were weak in economics" becomes '
        + '"economics moved by Δ on the half we never touched" — read that column.' }));
    }
  }
  return card;
}

function vModel() {
  const m = DATA.models.find(x => x.id === state.model);
  if (!m) return [note('No such model.')];
  const a = m.archinfo || {}, r = rankOf(m), avg = officialAvg(m), comp = computeOf(m);
  const back = el('a', { class: 'backlink', href: '#tab=' + state.tab, onclick: backTo(state.tab),
    text: '← Back to ' + (TABS.find(([id]) => id === state.tab) || [, 'the board'])[1] });

  // 11d: the hero — an eyebrow, the name, the id, three highlight cards, and
  // the prose underneath for anyone who wants it in words
  const ranked = DATA.models.filter(x => officialAvg(x) != null && !x.duplicateOf)
    .sort((x, y) => officialAvg(y) - officialAvg(x));
  let avgVerdict;
  if (avg == null) avgVerdict = `Preliminary — ${m.nhave} of ${m.nreq} required tasks, so no `
    + 'average and no rank.';
  else {
    const i = ranked.findIndex(x => x.id === m.id);
    const other = i > 0 ? ranked[i - 1] : ranked[1];
    if (!other) avgVerdict = 'The only ranked model on the board.';
    else {
      const d = avg - officialAvg(other), sa = officialSe(m), sb = officialSe(other);
      const words = i > 0 ? `${(100 * -d).toFixed(1)} points behind ${other.name}`
                          : `Leads ${other.name} by ${(100 * d).toFixed(1)} points`;
      avgVerdict = `${words[0].toUpperCase()}${words.slice(1)} — ` + (sa == null || sb == null
        ? 'no standard error to test it.'
        : Math.abs(d) / Math.sqrt(sa * sa + sb * sb || 1e-12) > 1.96 ? 'a real gap.' : 'within noise.');
    }
  }
  const eyebrow = ['model', m.source === 'artifact' ? 'checkpoint' : m.kind,
    r ? `#${r.n} of ${r.of}` : 'preliminary'].join(' · ');
  const head = el('div', { class: 'card mhero', 'data-model-hero': '1' },
    el('p', { class: 'eyebrow', 'data-model-eyebrow': '1', text: eyebrow }),
    el('div', { class: 'mhead' }, el('h1', { class: 'mtitle', text: m.name }),
      warnBadge(m) || '', dupBadge(m) || ''),
    el('p', { class: 'sub mono mid', 'data-model-id': m.id, text: m.id }),
    el('div', { class: 'mcards' }, el('div', { class: 'hlgrid' },
      hlCard('params', 'Parameters', m.params ? P(m.params) : 'Unknown',
        (a.active_params ? `${P(a.active_params)} active · ${a.experts} experts, `
          + `${a.experts_per_tok} per token (${a.active_src}).`
          : m.paramsSrc ? `From the ${m.paramsSrc === 'config' ? 'harness config' : 'model name'}.`
          : 'Not in the config or the name.')
        // the compute estimate, when it is known — not a card of its own
        + (comp ? ` Training compute ${flop(comp.c)} FLOP (6ND, run ${comp.run.name}).` : '')),
      hlCard('avg', state.avgMode === 'raw' ? 'Average · raw' : 'Average above chance',
        avg != null ? `${(100 * avg).toFixed(1)}` + (officialSe(m) != null
          ? ` ±${(100 * officialSe(m)).toFixed(1)}` : '') : '—', avgVerdict),
      hlCard('tasks', 'Tasks', `${m.nhave}/${m.nreq}`,
        (m.missing || []).length ? `Missing ${m.missing.join(', ')}.`
          : 'All required tasks' + (m.date ? `, evaluated ${String(m.date).slice(0, 10)}.` : '.'))),
      sitCta(m)),
    el('p', { class: 'mprose', text: modelSentence(m) }));

  // results, grouped by domain the way the task panels are
  const rows = [];
  for (const [dom, ts] of domainGroups([...DATA.accTasks, ...DATA.pplTasks])) {
    const have = ts.filter(t => cell(t, m.id));
    if (!have.length) continue;
    rows.push(el('tr', { class: 'domrow' },
      el('td', { colspan: 5, text: dom })));
    for (const t of have) {
      const c = cell(t, m.id), i = DATA.tasks[t] || {};
      const atChance = i.chance > 0 && (c.v - 1.96 * (c.se || 0)) <= i.chance;
      rows.push(el('tr', {},
        el('td', {}, el('span', { title: i.desc || '', class: 'tname',
          text: taskLabel(t) })),
        el('td', { class: 'num' + (atChance ? ' dimmed' : ''),
          text: i.lower ? num(c.v, 3) : pct(c.v) },
          c.se ? el('span', { class: 'se', text: ` ±${(100 * c.se).toFixed(1)}` }) : ''),
        el('td', {}, scoreBar(t, c)),
        el('td', { class: 'num se', text: c.shots != null ? c.shots + '-shot' : '—' }),
        el('td', { class: 'num se', text: c.n != null ? String(c.n) : '—' })));
      if (atChance) rows.push(el('tr', { class: 'flagrow' }, el('td', { colspan: 5,
        text: '↑ within 1.96 standard errors of chance — not distinguishable '
            + 'from guessing' })));
      if ((m.tainted || []).includes(t)) rows.push(el('tr', { class: 'flagrow' },
        el('td', { colspan: 5, text: '↑ this model trained on data derived from this '
          + 'task\'s diagnostics — shown, excluded from its official average' })));
    }
  }
  const results = el('div', { class: 'card' },
    el('h2', { text: 'Results' }),
    el('p', { class: 'sub', text: 'Dashed mark is chance; the solid mark, where one '
      + 'exists, is the best published score — a different protocol from ours, shown '
      + 'for orientation rather than comparison.' }),
    el('div', { class: 'lb-wrap' }, el('table', { class: 'lb mtbl' },
      el('thead', {}, el('tr', {}, el('th', { text: 'Benchmark' }),
        el('th', { class: 'num', text: 'Score' }), el('th', { text: '' }),
        el('th', { class: 'num', text: 'Shots' }), el('th', { class: 'num', text: 'Items' }))),
      el('tbody', {}, rows))));

  const prov = [
    ['architecture', a.arch], ['shape', a.hidden ? `hidden ${a.hidden} · layers ${a.layers}`
      + ` · heads ${a.heads} · ctx ${a.ctx}` : null],
    ['vocab', a.vocab], ['backend', m.backend], ['dtype', m.dtype],
    ['weights stored as', a.stored_dtype], ['batch size', m.batch],
    ['chat template', m.chat ? `applied${a.tmpl_sha ? ' · ' + a.tmpl_sha : ''}` : 'none'],
    ['kind decided by', m.kindReason], ['seed', m.seed], ['limit', m.limit],
    ['model code', (a.code_sha || []).join(', ')],
    ['eval wall clock', m.minutes != null ? m.minutes + ' min' : null],
    ['last evaluated', lastEval(m)],
  ].filter(([, v]) => v != null && v !== '' && v !== false);
  const provCard = el('div', { class: 'card' },
    el('h2', { text: 'Provenance' }),
    el('p', { class: 'sub', text: 'What produced these numbers. Two runs whose '
      + 'template id or harness differ are not comparable, whatever the scores say.' }),
    el('dl', { class: 'provlist' }, prov.flatMap(([k, v]) =>
      [el('dt', { text: k }), el('dd', { text: String(v) })])));

  // The exam leads: it is the instrument the loop steers by. The
  // multiple-choice results and the per-item diagnosis follow as the free
  // second opinion — same GPU, no API call, and a different kind of evidence.
  const judged = vJudged(m), taint = vTaint(m), diag = vDiagnose(m), earlier = vEarlier(m);
  const sections = [['judged', 'Judged', judged], ['earlier', 'Earlier exams', earlier],
                    ['results', 'Results', results],
                    ['diagnose', 'Diagnose', diag], ['taint', 'Training data', taint],
                    ['provenance', 'Provenance', provCard],
                    ['runs', 'Runs', LIVE ? vModelRuns(m) : null]];
  for (const [id, , node] of sections) if (node) node.id = 'sec-' + id;
  return [back, head, modelSitPanel(m), modelNav(sections), ...sections.map(([, , n]) => n),
          ].filter(Boolean);
}

// 11i: the model page is where a person is thinking about this model, so the
// exam starts here — beside the three cards, with how far it has got
function sitCta(m) {
  if (!LIVE) return '';
  const J = DATA.judged || {};
  const n = Object.entries((m.judge || {}).tasks || {})
    .filter(([t, v]) => (J.exam || []).includes(t) && pubScore(v) != null).length;
  return el('div', { class: 'msit-cta' },
    el('button', { class: 'primary', 'data-sit-open': m.id, text: 'Sit the exam',
      'aria-expanded': String(state.msit.open && state.msit.model === m.id),
      onclick: () => openSit(m.id) }),
    el('span', { class: 'small se', 'data-sit-progress': String(n),
      text: `${n} of ${(J.exam || []).length} topics judged` }));
}

// The model page is long, and since the judged section arrived the
// interesting part is halfway down it. A sub-nav that sticks is the cheapest
// fix: anchors, not routes, so Back still leaves the page the way it came.
function modelNav(sections) {
  const have = sections.filter(([, , node]) => node);
  return el('nav', { class: 'modelnav', 'data-model-nav': '1', 'aria-label': 'sections' },
    have.map(([id, label], i) =>
      el('a', { href: '#sec-' + id, class: 'navchip', 'data-nav': id,
        'data-ix': String(i + 1).padStart(2, '0'), text: label,
        onclick: e => { e.preventDefault();
          const t = document.getElementById('sec-' + id);
          if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); } })));
}

// The chip of the section in view is lit as you scroll. One observer for the
// page; a render re-points it at the new section nodes.
let _secObs = null;
function watchSections() {
  if (_secObs) { _secObs.disconnect(); _secObs = null; }
  const nav = document.querySelector('[data-model-nav]');
  if (!nav || !('IntersectionObserver' in window)) return;
  const secs = [...document.querySelectorAll('#view [id^="sec-"]')];
  const seen = new Map();
  const light = () => {
    // the topmost section with any part inside the band under the bar
    const vis = secs.filter(x => seen.get(x.id));
    const cur = vis.length ? vis[0].id.slice(4) : null;
    for (const a of nav.querySelectorAll('a[data-nav]'))
      a.setAttribute('aria-current', a.dataset.nav === cur ? 'true' : 'false');
  };
  _secObs = new IntersectionObserver(es => {
    for (const e of es) seen.set(e.target.id, e.isIntersecting);
    light();
  }, { rootMargin: '-110px 0px -55% 0px' });
  secs.forEach(x => _secObs.observe(x));
}

// Every submission of this model, newest first: which suite, what came of it,
// and the log. "Why is this preliminary?" is answered here rather than in
// someone's memory of the queue.
function vModelRuns(m) {
  const rows = (state.queue || []).filter(r => r.hf_id === m.id);
  if (!state.queue.length && netReady()) loadQueue();
  const card = el('div', { class: 'card' }, el('h2', { text: 'Runs of this model' }),
    el('p', { class: 'sub', text: 'Every time this model was submitted, and what came of '
      + 'it — the answer to "why is this one preliminary".' }));
  if (!rows.length) {
    card.append(note('No submission of this model is in the queue\u2019s recent history. It '
      + 'may have been evaluated before this service kept one, or by hand.'));
    return card;
  }
  card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-runs-table': '1' },
    el('thead', {}, el('tr', {}, el('th', { text: '#' }), el('th', { text: 'submitted' }),
      el('th', { text: 'suite' }), el('th', { text: 'status' }), el('th', { text: 'what happened' }),
      el('th', { text: '' }))),
    el('tbody', {}, rows.map(r => el('tr', { 'data-run': String(r.id) },
      el('td', { class: 'num se', text: '#' + r.id }),
      el('td', { class: 'small se', text: r.created_at ? rel(r.created_at) + ' ago' : '—' }),
      el('td', { class: 'small' }, suiteCell(r, 'm')),
      el('td', {}, (() => { const st = runStage(r);
        return el('span', { class: stClass(st.cls), 'data-stage': st.key, text: st.text }); })()),
      el('td', { class: 'small se', text: r.error || r.progress || '' }),
      el('td', {}, el('a', { href: `api/runs/${r.id}/log`, target: '_blank', rel: 'noopener',
        class: 'small', text: 'log' }))))))));
  return card;
}

function vOverview(ms) {
  const frag = [];
  // only official models can be ranked — a preliminary model has no average —
  // and a duplicate is the same run twice: it is not a second entry anywhere
  // on this page (the hero once featured the duplicate, with a sentence
  // calling it preliminary, and Top models listed the pair as #1 and #2)
  const ranked = ms.filter(m => officialAvg(m) != null && !m.duplicateOf)
                   .sort((a, b) => officialAvg(b) - officialAvg(a));
  const prelim = ms.filter(m => !m.official && !m.duplicateOf);
  const dup = new Set(ms.filter(m => m.duplicateOf).map(m => m.id));
  let pairs = 0, real = 0, big = null;
  for (const t of DATA.accTasks) for (const [a, b, diff, z, ok] of (DATA.sig[t] || [])) {
    if (!ms.find(m => m.id === a) || !ms.find(m => m.id === b)) continue;
    if (dup.has(a) || dup.has(b)) continue;
    pairs++; if (ok) real++;
    if (ok && (!big || Math.abs(diff) > Math.abs(big.diff))) big = { t, a, b, diff, z };
  }
  // 11d: the four stat tiles are one mono line under the hero
  const hours = DATA.meta.hours >= 1 ? `${DATA.meta.hours} h` : `${Math.round(DATA.meta.hours * 60)} min`;
  const dates = DATA.meta.dates[0]
    ? `${String(DATA.meta.dates[0]).slice(0, 10)} → ${String(DATA.meta.dates[1]).slice(0, 10)}` : null;
  frag.push(el('p', { class: 'statline', 'data-statline': '1' }, [
    `${ms.length} model${ms.length === 1 ? '' : 's'}`
      + (ms.length !== DATA.models.length ? ` of ${DATA.models.length}` : ''),
    `${DATA.accTasks.length + DATA.pplTasks.length} tasks`,
    pairs ? `${real.toLocaleString()} of ${pairs.toLocaleString()} gaps are real` : null,
    `${hours} of evaluation`, dates].filter(Boolean).join(' · ')));
  // 01 Highlights: a value and a sentence that says what it means, each
  // derived from the data it sits on
  frag.push(highlightsCard(ranked));
  // 02 Top models, in the table component, with the biggest real gap under it
  if (ranked.length) {
    let gap = '';
    if (big) {
      const an = DATA.models.find(m => m.id === big.a), bn = DATA.models.find(m => m.id === big.b);
      const [win, lose] = big.diff > 0 ? [an, bn] : [bn, an];
      gap = el('p', { class: 'statline', 'data-biggest-gap': '1',
        text: `Biggest real gap · ${big.t}: ${win.name} over ${lose.name} by `
          + `${(100 * Math.abs(big.diff)).toFixed(1)} points (z = ${Math.abs(big.z).toFixed(1)}) · `
          + `${pairs - real} of ${pairs} pairs are inside the noise` });
    }
    frag.push(el('div', { class: 'card', 'data-top-models': '1' },
      el('div', { class: 'sechead' }, el('h2', { text: 'Top models' }),
        el('span', { class: 'acts' }, el('a', { href: '#tab=leaderboard', 'data-see-leaderboard': '1',
          text: 'See the leaderboard →', onclick: e => { e.preventDefault();
            navigate({ tab: 'leaderboard', model: null, topic: null }); } }))),
      el('div', { class: 'lb-wrap' }, lbMini(ranked.slice(0, 5))), gap));
  } else {
    frag.push(el('div', { class: 'card' },
      el('h2', { text: 'No official result yet' }),
      el('p', { class: 'sub', text: DATA.required.length
        ? `Nothing here has completed all ${DATA.required.length} required tasks `
          + `(${DATA.required.join(', ')}), so there is no overall ranking to show — `
          + `only per-task numbers, which are on the Leaderboard and Tasks tabs. `
          + `Submit with suite=full to produce an official result.`
        : 'No accuracy tasks in this results tree.' })));
  }
  if (prelim.length)
    frag.push(el('p', { class: 'small', style: 'margin:10px 2px 0', 'data-prelim-count':
      String(prelim.length) }, `${prelim.length} preliminary model${prelim.length > 1 ? 's' : ''}`
      + ' — per-task results only, not in the ranking above. ',
      el('a', { href: '#tab=models', 'data-show-prelim': '1', text: 'See them',
        onclick: e => { e.preventDefault(); showMe({ tab: 'models', prelim: true }); } })));
  // 03 The loop
  if (LIVE) frag.push(overviewLoop());
  frag.push(el('details', { class: 'card', 'data-how-to-read': '1' },
    el('summary', { style: 'cursor:pointer' }, el('h2', { style: 'display:inline',
      text: 'How to read these numbers' })),
    note('Chance is not zero. 4-option tasks (MMLU, ARC, HellaSwag) sit at 25% for a model that knows nothing; 2-option tasks (Winogrande, PIQA) sit at 50%. A "50%" that looks respectable may be a coin flip.'),
    note('GSM8K near zero is a finding, not a failure — sub-billion models mostly cannot do written arithmetic. TruthfulQA is famous for NOT improving with scale.'),
    note('Perplexity (bits per byte) is the scale-sensitive metric here: it separates models that multiple-choice tasks cannot tell apart, and it works on base models with no prompt format at all. Multiply by ln 2 for cross-entropy loss in nats/byte — the Perplexity & Loss tab does it for you. Lower is better.'),
    note('Whiskers are ±1 standard error. If two whiskers overlap, do not call a winner — every pairwise z-test verdict rides along in the JSON export (the "sig" field) when you need the arbiter.')));
  return frag;
}
// ---------------------------------------------------------------------------
// 11d: the highlight cards. Each has a value and ONE sentence that says what
// it means, derived from the numbers under it — the best model's lead is the
// z-test's verdict, not an adjective. Cards 2–4 are the loop's, so only a
// live page has them.
// ---------------------------------------------------------------------------
// 11e: the value is the number only, on one line; what it is about — a model,
// a topic — is its own 16px line, cut with an ellipsis, whole in the tooltip
const hlCard = (key, eyebrow, value, verdict, link, name, full) => el('div', { class: 'hcard',
    'data-hl': key },
  el('div', { class: 'eyebrow', text: eyebrow }),
  el('div', { class: 'hcard-v', 'data-hl-value': key, text: value }),
  name ? el('div', { class: 'hcard-name', 'data-hl-name': key, title: full || name, text: name }) : '',
  el('p', { class: 'hcard-verdict', 'data-verdict': key, text: verdict }),
  link || '');
const hlLink = (text, go) => el('a', { href: '#', class: 'hcard-link', text,
  onclick: e => { e.preventDefault(); go(); } });

// the model the loop is on: the one judged last
function loopModel() {
  const lastJ = m => Math.max(0, ...Object.entries((m.judge || {}).tasks || {})
    .filter(([t]) => t.startsWith('exam_')).map(([, v]) => v.judged_at || 0));
  const judged = DATA.models.filter(m => !m.duplicateOf && Object.keys((m.judge || {}).tasks || {})
    .some(t => t.startsWith('exam_')));
  return judged.length ? [...judged].sort((a, b) => lastJ(b) - lastJ(a))[0] : null;
}
const whyProvisional = m => {
  const jj = ((m.judge || {}).judge) || {};
  if (jj.provisional || jj.provider === 'local') return 'provisional, local judge';
  if (!judgedOkM(m)) return 'provisional, the judge is not calibrated';
  return 'calibrated judge';
};

function hlBest(ranked) {
  const toLb = hlLink('See the leaderboard →', () => navigate({ tab: 'leaderboard', model: null, topic: null }));
  const top = ranked[0], next = ranked[1];
  if (!top) return hlCard('best', 'Best model', '—', `Nothing has completed all `
    + `${DATA.required.length} required tasks yet, so nothing is ranked.`, toLb);
  const av = officialAvg(top);
  let verdict;
  if (!next) verdict = 'The only ranked model on the board.';
  else {
    const d = av - officialAvg(next), sa = officialSe(top), sb = officialSe(next);
    const lead = `Leads ${next.name} by ${(100 * d).toFixed(1)} points — `;
    if (sa == null || sb == null) verdict = lead + 'no standard error to test it.';
    else {
      const z = d / Math.sqrt(sa * sa + sb * sb || 1e-12);
      verdict = lead + (Math.abs(z) > 1.96 ? `a real gap (z = ${z.toFixed(1)}).` : 'within noise.');
    }
  }
  return hlCard('best', 'Best model', (100 * av).toFixed(1), verdict, toLb, top.name, top.id);
}

function hlWeakest() {
  const m = loopModel();
  const toLoop = hlLink('Open the Loop →', () => navigate({ tab: 'loop', model: null, topic: null }));
  if (!m) return hlCard('weakest', 'Weakest topic', '—',
    'No model has been judged yet — Loop ▸ Sit the exam.', toLoop);
  const exam = (DATA.judged || {}).exam || [];
  const xs = Object.entries(m.judge.tasks).filter(([t]) => t.startsWith('exam_'))
    .map(([t, v]) => ({ t, v: pubScore(v) })).filter(x => x.v != null).sort((a, b) => a.v - b.v);
  if (!xs.length) return hlCard('weakest', 'Weakest topic', '—',
    `${m.name} has no score on the hidden questions yet.`, toLoop);
  const w = xs[0];
  // one model's own topic, so it is not a ranking — and no area mean
  return hlCard('weakest', 'Weakest topic', `${(+w.v).toFixed(2)} / 4`,
    `${m.name}, ${xs.length} of ${exam.length} topics judged — ${whyProvisional(m)}.`,
    hlLink('Open the topic →', () => navigate({ topic: w.t.replace(/^exam_/, ''), model: null })),
    frName(w.t));
}

function hlLoop() {
  const exam = new Set((DATA.judged || {}).exam || []);
  const done = new Set(DATA.models.flatMap(m => Object.entries((m.judge || {}).tasks || {})
    .filter(([t, v]) => exam.has(t) && pubScore(v) != null).map(([t]) => t)));
  const m = loopModel();
  const at = m ? Math.max(0, ...Object.values(m.judge.tasks).map(v => v.judged_at || 0)) : 0;
  return hlCard('loop', 'The loop', `${done.size} / ${exam.size}`,
    m ? `Last judged ${m.name}` + (at ? ` ${rel(at)} ago.` : '.')
      : 'Nothing judged yet — sit a model on a topic from the Loop.',
    hlLink('Open the Loop →', () => navigate({ tab: 'loop', model: null, topic: null })),
    'topics judged');
}

function hlJudge() {
  const m = loopModel();
  const toProv = hlLink('Provenance →', () => navigate({ tab: 'provenance', model: null, topic: null }));
  const cn = m && (m.judge || {}).canary;
  const cal = (DATA.judged || {}).calibration;
  const calWords = cal && cal.calibrated ? `Agreement with a person: ${cal.kappa}.`
    : 'No person has checked the judge yet.';
  if (!cn) return hlCard('judge', 'Judge steadiness', '—',
    (m ? 'No canary on file for this judge. ' : 'No judged run yet. ') + calWords, toProv);
  const r2 = x => x == null ? '—' : (+x).toFixed(2);
  return hlCard('judge', 'Judge steadiness', `${cn.graded} / ${cn.n}`,
    `${cn.n} fixed scripts re-graded: ${r2(cn.mad_vs_human)} from the human marks`
      + (cn.mad_vs_previous != null ? `, ${r2(cn.mad_vs_previous)} from the last run `
         + `(limit ${cn.threshold})` : ', the first run for this judge') + `. ${calWords}`, toProv,
    cn.drifted ? 'moved' : 'steady');
}

function highlightsCard(ranked) {
  return el('div', { class: 'card', 'data-highlights': '1' },
    el('h2', { text: 'Highlights' }),
    el('div', { class: 'hlgrid' }, hlBest(ranked), LIVE ? [hlWeakest(), hlLoop(), hlJudge()] : ''));
}

// The loop on the first tab: how many topics have been judged, each judged
// model's weakest topic one click away, and the last judged run and when. The
// newest work on the server had no presence on the page people open first.
function overviewLoop() {
  const exam = new Set((DATA.judged || {}).exam || []);
  const rows = DATA.models.filter(m => !m.duplicateOf && m.judge).map(m => {
    const ts = Object.entries(m.judge.tasks || {}).filter(([t, v]) => exam.has(t)
      && pubScore(v) != null);
    if (!ts.length) return null;
    const [wt, wv] = ts.reduce((a, b) => pubScore(b[1]) < pubScore(a[1]) ? b : a);
    const at = Math.max(0, ...ts.map(([, v]) => v.judged_at || 0));
    return { m, n: ts.length, wt, wv, at, topics: ts.map(([t]) => t) };
  }).filter(Boolean).sort((a, b) => b.n - a.n || b.at - a.at);
  const topics = new Set(rows.flatMap(r => r.topics));
  const card = el('div', { class: 'card', 'data-overview-loop': '1' },
    el('h2', { text: 'The loop' }));
  if (!rows.length) {
    card.append(el('p', { class: 'small' }, 'No model has sat the exam yet. ',
      el('a', { href: '#tab=loop', text: 'Open the Loop →',
        onclick: e => { e.preventDefault(); navigate({ tab: 'loop', topic: null, model: null }); } })));
    return card;
  }
  const last = rows.reduce((a, b) => b.at > a.at ? b : a);
  card.append(el('p', { class: 'sub' }, `${topics.size} topic${topics.size > 1 ? 's' : ''} `
    + `judged across ${rows.length} model${rows.length > 1 ? 's' : ''}.`
    + (last.at ? ` Last judged: ${last.m.name}, ${rel(last.at)} ago.` : ''), ' ',
    el('a', { href: '#tab=loop', text: 'Open the Loop →',
      onclick: e => { e.preventDefault(); navigate({ tab: 'loop', topic: null, model: null }); } })));
  card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
    el('thead', {}, el('tr', {}, el('th', { text: 'model' }), el('th', { class: 'num', text: 'topics' }),
      el('th', { text: 'weakest topic' }), el('th', { class: 'num', text: 'score' }), el('th', { text: '' }))),
    el('tbody', {}, rows.slice(0, 6).map(r => el('tr', { 'data-loop-model': r.m.id },
      el('td', {}, el('a', { href: '#model=' + encodeURIComponent(r.m.id), text: r.m.name,
        onclick: e => { e.preventDefault(); navigate({ model: r.m.id, topic: null }); } })),
      el('td', { class: 'num', text: String(r.n) }),
      el('td', { text: frName(r.wt) }),
      el('td', { class: 'num', text: `${num(pubScore(r.wv), 2)} / 4` }),
      el('td', {}, el('button', { class: 'quiet', 'data-weakest-topic': r.wt,
        text: `${frName(r.wt)} →`, onclick: () => {
          state.ans.model = r.m.id; state.ans.rows = null;
          state.after = { scroll: '[data-panel="answers"]' };
          navigate({ topic: r.wt.replace(/^exam_/, ''), model: null }); } }),
        // 11i: and the exam, on the model's page
        el('button', { class: 'quiet', 'data-loop-sit': r.m.id, text: 'Sit the exam',
          onclick: () => openSit(r.m.id) }))))))));
  return card;
}

// ---------------------------------------------------------------------------
// Routing. The whole report is one file with no server, so the address bar is
// the only place a view can live — and putting it there is what makes Back work.
// A dashboard whose Back button leaves the page instead of returning to the list
// you came from is the single most reported annoyance in apps shaped like this.
// ---------------------------------------------------------------------------
const hashFor = () => (state.model ? 'model=' + encodeURIComponent(state.model)
                                  : state.topic ? 'topic=' + encodeURIComponent(state.topic)
                                  : 'tab=' + state.tab
                                    + (state.tab === 'leaderboard' && lbHash() ? '&' + lbHash() : '')
                                    // 11j: which Review view, so a link opens it
                                    + (state.tab === 'review' && state.rv.view
                                       ? '&view=' + state.rv.view : ''))
  // 11g: an open reader rides along, so a pasted link opens it too
  + (state.read ? '&read=' + encRead(state.read) : '');

function routeFromHash() {
  // 11g: the reader's part first — it can follow any page
  const [rest, rd] = splitRead(location.hash);
  state.read = rd;
  const h = decodeURIComponent(rest);
  const m = /^model=(.+)$/.exec(h);
  if (m && DATA.models.some(x => x.id === m[1])) { state.model = m[1]; state.topic = null; return; }
  state.model = null;
  // a topic page is the loop for one topic — deep-linkable, because it is the
  // page a person is sent to when someone says "look at law"
  const tp = /^topic=(.+)$/.exec(h);
  if (tp && LIVE && topicOfSlug(tp[1])) { state.topic = tp[1]; state.tab = 'loop'; return; }
  state.topic = null;
  // "tab=leaderboard&chip=knowledge&open=…": the Leaderboard's view rides
  // along, so a pasted link reproduces it (11c). Old hashes have no "&".
  const t = /^tab=([^&]+)(?:&(.*))?$/.exec(rest);
  if (!t) return;
  const tab = decodeURIComponent(t[1]);
  const want = TAB_ALIASES[tab] || tab;
  if (TABS.some(([id]) => id === want)) state.tab = want;
  if (state.tab === 'leaderboard') lbFromHash(t[2] || '');
  // 11j: "tab=review&view=datasets"
  if (state.tab === 'review') {
    const v = /(?:^|&)view=([^&]+)/.exec(t[2] || '');
    state.rv.view = v && RV_VIEWS.some(([k]) => k === v[1]) ? v[1] : '';
  }
}

// slug ↔ topic, from the same map the payload carries (exam_law ↔ law)
function topicOfSlug(slug) {
  const task = 'exam_' + slug;
  return ((DATA.judged || {}).topics || {})[task] || null;
}
function slugOfTopic(topic) {
  const e = Object.entries(((DATA.judged || {}).topics) || {}).find(([, v]) => v === topic);
  return e ? e[0].replace(/^exam_/, '') : '';
}

// every navigation goes through here, so history and state cannot disagree
function navigate(patch) {
  // entering the Queue starts it on page 1: the newest rows, where a new
  // one — even a failure — is
  if (patch.tab === 'queue' && (state.tab !== 'queue' || state.model || state.topic)
      && state.pg.queue) state.pg.queue.page = 1;
  const from = location.hash;
  Object.assign(state, patch);
  const want = hashFor();
  // push history, then paint. Painting here rather than leaving it to the
  // hashchange handler is deliberate: that handler ignores a hash which already
  // agrees with state (it is our own write echoing back), so relying on it to
  // render meant a tab click updated the URL and nothing else.
  // 11f: the entry being left keeps where it was scrolled to, for Back; the
  // new one remembers where it came from, for "← Back to …"
  if (location.hash.slice(1) !== want) {
    clearTimeout(_saveT); saveScroll();
    history.pushState({ from, y: 0 }, '', '#' + want);
  }
  _navigated = true;
  render();
}

window.addEventListener('hashchange', () => {
  // a hash that already matches state is our own write echoing back; anything
  // else is the user pressing Back or Forward, and we adopt it — at the
  // scroll they left that view at (11f)
  if (location.hash.slice(1) === hashFor()) return;
  routeFromHash();
  _restore = { y: (history.state || {}).y || 0, until: Date.now() + 3000 };
  _navigated = true;
  render();
});

// ---- 11f: a new page starts at the top; Back returns to where you were -----
// The browser cannot restore a view this page rebuilds from state, so the
// page does it: every entry keeps its scrollY in history.state (saved as the
// person scrolls, and on the way out), and a view that changes by
// navigation starts at 0 unless a button aimed it at a section.
try { history.scrollRestoration = 'manual'; } catch (e) { /* an old browser */ }
const viewKey = () => state.model ? 'model:' + state.model
  : state.topic ? 'topic:' + state.topic : 'tab:' + state.tab;
let _lastView = null, _restore = null, _navigated = false, _saveT = null;
function saveScroll() {
  try { history.replaceState({ ...(history.state || {}), y: Math.round(scrollY) }, ''); }
  catch (e) { /* a sandboxed frame */ }
}
// a save belongs to the entry that was scrolled: one still waiting when Back
// or Forward lands must not write this entry's scroll into the other one
window.addEventListener('scroll', () => {
  clearTimeout(_saveT);
  const at = location.hash;
  _saveT = setTimeout(() => { if (location.hash === at) saveScroll(); }, 150);
}, { passive: true });
window.addEventListener('popstate', () => clearTimeout(_saveT));
// the person scrolling or typing ends a restore that is still waiting for
// the page to grow tall enough
for (const ev of ['wheel', 'touchstart', 'keydown'])
  window.addEventListener(ev, () => { _restore = null; }, { passive: true, capture: true });

// a plain link to a view on this page ("#model=…", "#topic=…") is a
// navigation like any other: it pushes an entry that knows where it came
// from and saves the scroll of the one it leaves. A link with a job of its
// own (it called preventDefault) keeps it
document.addEventListener('click', e => {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  const a = e.target.closest && e.target.closest('a[href^="#"]');
  if (!a || a.target || a.hasAttribute('download')) return;
  const h = a.getAttribute('href');
  if (h.length < 2 || h === location.hash) return;
  e.preventDefault();
  const from = location.hash;
  clearTimeout(_saveT); saveScroll();
  history.pushState({ from, y: 0 }, '', h);
  routeFromHash();
  _navigated = true;
  render();
});

// "← Back to …" is Back when the entry before this one is the view it names
function backTo(tab) {
  return e => {
    const from = (history.state || {}).from;
    if (from == null || /[#&](model|topic)=/.test(from)) return;
    if (new URLSearchParams(from.replace(/^#/, '')).get('tab') !== tab) return;
    e.preventDefault();
    history.back();
  };
}

// FLOP with a readable exponent — 6ND spans twenty orders of magnitude across a
// board that holds both a 14M probe and a 750M checkpoint
const flop = v => {
  if (!(v > 0) || !isFinite(v)) return null;
  const e = Math.floor(Math.log10(v));
  return `${(v / Math.pow(10, e)).toFixed(1)}e${e}`;
};

// Training compute for a model, or null. C ≈ 6ND is the standard dense estimate;
// for a sparse model the N that matters is the ACTIVE parameter count, not the
// total, so prefer it and say which was used. D only exists for checkpoints that
// came from a tracked training run — for a Hub model we simply do not know, and
// (per epoch.ai, whose job this is) neither does anyone else for most of them.
function computeOf(m) {
  const a = m.archinfo || {};
  const N = a.active_params || m.params;
  if (!N) return null;
  const run = (state.trRuns || []).find(r =>
    r.hf_prefix && (m.id === r.hf_prefix || m.id.startsWith(r.hf_prefix)));
  if (!run || !(run.tokens > 0)) return null;
  return { c: 6 * N * run.tokens, N, D: run.tokens, run,
           nsrc: a.active_params ? `active (${a.active_src})` : 'total' };
}

const tile = (label, value, notetext, tip) => el('div', { class: 'tile', title: tip || null },
  el('div', { class: 'label', text: label }),
  el('div', { class: 'value', text: value }),
  notetext ? el('div', { class: 'note', text: notetext }) : null);
const note = t => el('p', { class: 'note', text: t });
// an empty state is a sentence and the action that fills it
function empty(text, label, go, attrs = {}) {
  return el('div', { class: 'empty', 'data-empty': '1', ...attrs },
    el('p', { text }),
    label ? el('button', { class: 'secondary', 'data-empty-action': '1', text: label,
      onclick: go }) : '');
}
// while something loads: the shape of what is coming, not the word "Loading…"
function skeleton(rows = 4, attrs = {}) {
  return el('div', { class: 'skeleton', role: 'status', 'aria-busy': 'true',
      'aria-label': 'loading', ...attrs },
    Array.from({ length: rows }, (_, i) => el('div', { class: 'sk-row',
      style: `width:${[92, 78, 86, 64, 88, 72][i % 6]}%` })));
}

// ---------------------------------------------------------------------------
// A button that calls the API answers in three ways, always: it says it is
// working while the request is out, it says what happened when it lands, and
// when it is refused it says what the SERVER said. The rebuild button used to
// answer "refused: 500" in small grey text under itself, which a person read
// as "nothing happens" — and it was a missing file in the image.
// ---------------------------------------------------------------------------

const ACT = {};
const actState = slot => ACT[slot] || (ACT[slot] = { busy: false, ok: '', err: '' });

async function post(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
    body: JSON.stringify(body || {}) }).catch(() => null);
  if (!r) throw new Error('the server is unreachable — it may be restarting');
  checkBuild(r);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail
    : (j.detail ? JSON.stringify(j.detail) : `the server answered HTTP ${r.status}`));
  return j;
}

// run() returns the sentence to show on success, or throws with the server's
// own words. Both are rendered by actNote(slot) wherever it is placed.
function actButton(slot, label, run, attrs = {}) {
  const a = actState(slot);
  return el('button', Object.assign({}, attrs, {
    'data-action': slot,
    disabled: a.busy ? '' : (attrs.disabled === undefined ? null : attrs.disabled),
    text: a.busy ? 'working…' : label,
    onclick: async () => {
      a.busy = true; a.ok = ''; a.err = ''; render();
      try {
        const out = await run();
        // {toast, …}: the answer is a toast, and nothing is left under the button
        if (out && typeof out === 'object') { toast(out.toast, out); a.ok = out.line || ''; }
        else a.ok = out || 'done.';
      }
      catch (e) { a.err = String((e && e.message) || e); }
      finally { a.busy = false; render(); }
    } }));
}

function actNote(slot) {
  const a = actState(slot);
  if (a.busy) return el('p', { class: 'small', 'data-action-busy': slot,
    text: 'Working — the request is out.' });
  if (a.err) return el('p', { class: 'warn', 'data-action-error': slot },
    el('b', { text: 'Refused. ' }), a.err);
  if (a.ok) return el('p', { class: 'note', 'data-action-ok': slot, text: a.ok });
  return '';
}

// ---------------------------------------------------------------------------
// A toast: every action that changes something says so — bottom-right, four
// seconds, a polite live region, and a link to where the thing went. Errors
// stay inline, next to the control that caused them. A toast that carries a
// link stays eight seconds and one with a button fifteen, and neither goes
// while the pointer is on it or focus is in it: four seconds was gone before
// anyone reached "see the queue".
// ---------------------------------------------------------------------------
const TOAST_MS = 4000, TOAST_LINK_MS = 8000, TOAST_ACTION_MS = 15000;

function toast(text, opts = {}) {
  let box = document.getElementById('toasts');
  if (!box) {
    box = el('div', { id: 'toasts', class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.append(box);
  }
  const t = el('div', { class: 'toast', 'data-toast': opts.key || '1' },
    el('span', { class: 'toast-text', text }),
    opts.go ? el('a', { href: opts.href || '#', 'data-toast-link': '1', text: opts.link || 'open',
      onclick: e => { e.preventDefault(); drop(); opts.go(); } }) : '',
    opts.action ? el('button', { class: 'quiet', 'data-toast-action': '1', text: opts.action.label,
      onclick: async () => { drop(); await opts.action.run(); } }) : '',
    el('button', { class: 'xbtn', 'aria-label': 'dismiss', text: '×', onclick: () => drop() }));
  // 11f: it fades out; what fades has no key, so nothing finds a gone toast
  function drop() {
    if (!t.isConnected || t.classList.contains('out')) return;
    t.removeAttribute('data-toast');
    t.classList.add('out');
    setTimeout(() => t.remove(), motionOff() ? 0 : 240);
  }
  box.append(t);
  // the clock runs only while nobody is on it: hover or focus stops it, and
  // leaving gives the full time back rather than whatever was left
  const ms = opts.ms || (opts.action ? TOAST_ACTION_MS : opts.go ? TOAST_LINK_MS : TOAST_MS);
  let timer = null;
  const arm = () => { clearTimeout(timer); timer = setTimeout(drop, ms); };
  const hold = () => { clearTimeout(timer); timer = null; };
  const busy = () => t.matches(':hover') || t.contains(document.activeElement);
  t.addEventListener('mouseenter', hold);
  t.addEventListener('focusin', hold);
  t.addEventListener('mouseleave', () => { if (!busy()) arm(); });
  t.addEventListener('focusout', () => setTimeout(() => { if (!busy()) arm(); }, 0));
  arm();
  return t;
}
const goQueue = () => navigate({ tab: 'queue', topic: null, model: null });
const goReview = () => { state.rv.loaded = false; navigate({ tab: 'review', topic: null, model: null }); };

// ---------------------------------------------------------------------------
// ⓘ beside a card's title: the long explanation, one click away, instead of a
// paragraph above every card pushing the work down the page.
// ---------------------------------------------------------------------------
let _tipN = 0;
function infoTip(text) {
  const id = 'info-' + (++_tipN);
  const pop = el('span', { class: 'infopop', id, role: 'note', hidden: '', text });
  const b = el('button', { class: 'info', 'aria-label': 'what this is', 'aria-expanded': 'false',
    'aria-controls': id, 'data-info': '1', text: 'ⓘ',
    onclick: e => { e.stopPropagation(); pop.hidden = !pop.hidden;
                    b.setAttribute('aria-expanded', String(!pop.hidden)); } });
  return el('span', { class: 'infowrap' }, b, pop);
}

// ---------------------------------------------------------------------------
// One pager for every long table: "1–25 of 49" · size · ‹ Prev · 1 2 … 7 ·
// Next ›. The page lives in state, so a 5-second poll never throws anyone back
// to page 1; a change of filter or sort does (the caller passes a signature of
// them), because page 3 of a different list is a different page. The size is
// remembered per table in this browser — a convenience: 25 without storage.
// ---------------------------------------------------------------------------
const PAGE_SIZES = [10, 25, 50, 100];
state.pg = {};

function pgState(key, dflt) {
  let p = state.pg[key];
  if (!p) {
    let size = dflt;
    try {
      const v = +localStorage.getItem('bench-pagesize-' + key);
      if (PAGE_SIZES.includes(v) || v === dflt) size = v;
    } catch (e) { /* storage unavailable: the default */ }
    p = state.pg[key] = { page: 1, size, dflt, sig: undefined };
  }
  return p;
}

function pageNumbers(cur, n) {
  if (n <= 7) return Array.from({ length: n }, (_, i) => i + 1);
  if (cur <= 4) return [1, 2, 3, 4, 5, '…', n];
  if (cur >= n - 3) return [1, '…', n - 4, n - 3, n - 2, n - 1, n];
  return [1, '…', cur - 1, cur, cur + 1, '…', n];
}

// rows -> { rows: this page's slice, pager: the control (or '' when one page
// of the smallest size holds everything). `live`: one pager node for the
// life of the page, updated in place (livePager) — for a table a poll
// redraws while someone is clicking it
// what the table is sorted by, in the words of its own header
function lbSortLabel(cols) {
  const c = (cols || []).find(x => x.key === state.sort.key);
  return `${c ? (c.label || c.key) : state.sort.key} ${state.sort.dir > 0 ? '▲' : '▼'}`;
}

// 11b: one mono line above a big table — what is on screen, out of what, and
// how it is sorted. "Showing 1–25 of 32 models · 15 ranked · sorted by Avg ▼".
function statusLine(pg, noun, extra) {
  const bits = [`Showing ${pg.from}–${pg.to} of ${pg.total} ${noun}`];
  for (const x of [].concat(extra || [])) if (x) bits.push(x);
  const line = el('p', { class: 'statusline', 'data-statusline': noun }, bits.join(' · '));
  if (LIVE && DATA) line.append(' · ', el('span', { class: 'dot ok' }), `live ${checkedAt()}`);
  return line;
}

function paged(key, rows, sig, redraw, dflt = 25, live = false) {
  const p = pgState(key, dflt);
  if (p.sig !== sig) { p.sig = sig; p.page = 1; }
  const pages = Math.max(1, Math.ceil(rows.length / p.size));
  if (p.page > pages) p.page = pages;
  const start = (p.page - 1) * p.size;
  const shown = rows.slice(start, start + p.size);
  return { rows: shown, total: rows.length,
           from: rows.length ? start + 1 : 0, to: start + shown.length,
           pager: (live ? livePager : pager)(key, rows.length, redraw) };
}

// The queue's pager was rebuilt on every refresh while a run was live, and a
// click could land on a button that had just been replaced — the tab bar's
// bug (#21) again. This one is built once per key and only its text, state
// and number buttons change; one delegated handler reads the current page
// when it is clicked, so no listener holds a stale one.
const _livePagers = {};
function livePager(key, total, redraw) {
  const p = state.pg[key];
  let nav = _livePagers[key];
  if (!nav) {
    nav = _livePagers[key] = el('nav', { class: 'pager', 'data-pager': key, 'aria-label': 'pages' });
    nav._note = el('span', { class: 'count-note' });
    nav._size = Select('rows per page', [], '', null, { key: 'size-' + key });
    nav._prev = el('button', { class: 'quiet', text: '‹ Prev', 'data-page-prev': '1' });
    nav._nums = el('span', { class: 'pgnums', style: 'display:contents' });
    nav._next = el('button', { class: 'quiet', text: 'Next ›', 'data-page-next': '1' });
    nav.append(nav._note, nav._size, nav._prev, nav._nums, nav._next);
    nav.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b || b.disabled || !nav.contains(b)) return;
      const q = state.pg[key], pages = Math.max(1, Math.ceil(nav._total / q.size));
      const n = b.dataset.pagePrev ? q.page - 1 : b.dataset.pageNext ? q.page + 1 : +b.dataset.page;
      if (!n) return;
      q.page = Math.min(Math.max(1, n), pages);
      (nav._redraw || render)();
    });
    nav._size.addEventListener('change', e => {
      const q = state.pg[key];
      q.size = +e.target.value; q.page = 1;
      try { localStorage.setItem('bench-pagesize-' + key, e.target.value); } catch (x) { /* private */ }
      (nav._redraw || render)();
    });
  }
  nav._total = total; nav._redraw = redraw;
  nav.style.display = total <= PAGE_SIZES[0] ? 'none' : '';
  const pages = Math.max(1, Math.ceil(total / p.size));
  const from = total ? (p.page - 1) * p.size + 1 : 0, to = Math.min(total, p.page * p.size);
  nav._note.dataset.pageRange = `${from}-${to}`;
  nav._note.textContent = `${from}–${to} of ${total}`;
  const sizes = [...new Set([...PAGE_SIZES, p.dflt])].sort((a, b) => a - b).map(String);
  nav._size.setOptions(sizes.map(n => [n, `${n} per page`]));
  nav._size.value = String(p.size);
  nav._prev.disabled = p.page <= 1;
  nav._next.disabled = p.page >= pages;
  // the number buttons: reused by position, added or dropped only when the
  // count of them changes
  const want = pageNumbers(p.page, pages);
  const have = [...nav._nums.children];
  want.forEach((n, i) => {
    let node = have[i];
    const isGap = n === '…';
    if (!node || (node.tagName === 'SPAN') !== isGap) {
      const fresh = isGap ? el('span', { class: 'se', text: '…' })
                          : el('button', { class: 'pgnum' });
      if (node) node.replaceWith(fresh); else nav._nums.append(fresh);
      node = fresh;
    }
    if (!isGap) {
      node.textContent = String(n);
      node.dataset.page = String(n);
      if (n === p.page) node.setAttribute('aria-current', 'page');
      else node.removeAttribute('aria-current');
    }
  });
  for (const extra of have.slice(want.length)) extra.remove();
  return nav;
}

function pager(key, total, redraw) {
  const p = state.pg[key];
  if (total <= PAGE_SIZES[0]) return '';
  const pages = Math.max(1, Math.ceil(total / p.size));
  const go = n => { p.page = Math.min(Math.max(1, n), pages); (redraw || render)(); };
  const from = total ? (p.page - 1) * p.size + 1 : 0;
  const to = Math.min(total, p.page * p.size);
  return el('nav', { class: 'pager', 'data-pager': key, 'aria-label': 'pages' },
    el('span', { class: 'count-note', 'data-page-range': `${from}-${to}`,
      text: `${from}–${to} of ${total}` }),
    mkSel('rows per page', [...new Set([...PAGE_SIZES, p.dflt])].sort((a, b) => a - b)
      .map(n => [String(n), `${n} per page`]), String(p.size),
      v => { p.size = +v; p.page = 1;
        try { localStorage.setItem('bench-pagesize-' + key, v); } catch (e) { /* private mode */ }
        (redraw || render)(); }),
    el('button', { class: 'quiet', text: '‹ Prev', disabled: p.page <= 1 ? '' : null,
      'data-page-prev': '1', onclick: () => go(p.page - 1) }),
    ...pageNumbers(p.page, pages).map(n => n === '…'
      ? el('span', { class: 'se', text: '…' })
      : el('button', { class: 'pgnum', text: String(n), 'data-page': String(n),
          'aria-current': n === p.page ? 'page' : null, onclick: () => go(n) })),
    el('button', { class: 'quiet', text: 'Next ›', disabled: p.page >= pages ? '' : null,
      'data-page-next': '1', onclick: () => go(p.page + 1) }));
}

// ---------------------------------------------------------------------------
// Model search in a model-id box: after two characters (200 ms debounce) a
// list opens under it — the models this board knows first, then the Hub's,
// fetched by the service. ARIA combobox: ↑/↓ move, Enter picks, Esc closes,
// Tab accepts the highlighted one. Its state lives in state.ms, so a poll that
// rebuilds the view gives back the text, the caret (data-keep) and the list.
// ---------------------------------------------------------------------------
state.ms = {};

function msState(key) {
  return state.ms[key] || (state.ms[key] = { items: [], open: false, active: -1, footer: '',
                                             seq: 0, timer: null, q: '', onPick: null });
}

function msQuery(key, q) {
  const s = msState(key);
  s.q = q;
  clearTimeout(s.timer);
  if (q.trim().length < 2) { s.open = false; s.items = []; s.active = -1; msPaint(key); return; }
  s.timer = setTimeout(async () => {
    const seq = ++s.seq;
    try {
      const j = await api('api/models/suggest?q=' + encodeURIComponent(q.trim()));
      if (seq !== s.seq) return;                  // a newer keystroke already asked
      s.items = j.items || []; s.footer = j.footer || '';
      for (const it of s.items) {
        if (it.weights === false) state.noWeights.add(it.id);
        else if (it.weights === true) state.noWeights.delete(it.id);
      }
      s.open = true; s.active = -1;
    } catch (e) {
      if (seq !== s.seq) return;
      s.items = []; s.footer = 'search unavailable — type the id'; s.open = true;
    }
    msPaint(key);
  }, 200);
}

function msPick(key, i) {
  const s = msState(key);
  const it = s.items[i];
  if (!it) return;
  s.open = false; s.active = -1;
  const input = document.querySelector(`[data-ms="${key}"] input`);
  if (input) input.value = it.id;
  if (s.onPick) s.onPick(it);
  msPaint(key);
}

function msKey(key, e) {
  const s = msState(key);
  const n = s.items.length;
  // the arrows skip a model that cannot run here; it is still listed, and it
  // still says why
  const step = (from, d) => {
    let i = from;
    while (true) {
      const next = i + d;
      // nothing further this server can run: the selection stays where it was
      if (next < 0 || next > s.items.length - 1) return from;
      i = next;
      if (!msNoWeights(s.items[i])) return i;
    }
  };
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (!s.open && n) s.open = true;
    s.active = step(s.active, 1);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    s.active = step(s.active, -1);
  } else if (e.key === 'Enter') {
    if (s.open && s.active >= 0) { e.preventDefault(); msPick(key, s.active); }
    return;
  } else if (e.key === 'Escape') {
    if (s.open) { e.preventDefault(); s.open = false; s.active = -1; }
  } else if (e.key === 'Tab') {
    if (s.open && s.active >= 0) msPick(key, s.active);   // and focus moves on as usual
    return;
  } else return;
  msPaint(key);
}

// a local/ id whose results are on this board but whose weights are not on
// this server: #53 picked one from the search and the run failed at start
const msNoWeights = it => it.weights === false;
const NO_WEIGHTS = 'results only — weights not on the server';
// every id the search has said that about, so a form can refuse before the
// API does — in the API's own words
state.noWeights = new Set();
const noWeightsWhy = id =>
  `${id} has results on this board but no weights on this server — upload it first `
  + `(POST /api/artifacts/${String(id).replace(/^local\//, '')}) to run it here.`;
const cannotRun = id => state.noWeights.has(String(id || '').trim());

function msItem(key, it, i, active) {
  const bits = [];
  if (it.params) bits.push(P(it.params));
  if (it.kind) bits.push(it.kind_guessed ? it.kind + ' (from its name)' : it.kind);
  if (it.on_board) bits.push('on the board');
  if (it.judged) bits.push(`judged on ${it.judged} topic${it.judged > 1 ? 's' : ''}`);
  if (it.queued) bits.push('in the queue');
  if (it.artifact) bits.push('uploaded checkpoint');
  if (it.over_cap) bits.push('over the size cap');
  // 11i: said in the list, before it is picked
  if (it.own_code) bits.push(it.own_code.runs ? 'ships its own model code'
    : 'ships its own model code — this server does not run it');
  if (msNoWeights(it)) bits.push(NO_WEIGHTS);
  return el('li', { id: `ms-${key}-${i}`, role: 'option', class: 'ms-item'
      + (i === active ? ' active' : '') + (it.over_cap ? ' over' : '')
      + (msNoWeights(it) ? ' noweights' : ''),
    'aria-selected': String(i === active),
    'aria-disabled': msNoWeights(it) ? 'true' : null, 'data-ms-item': it.id,
    onmousedown: e => { e.preventDefault(); msPick(key, i); } },
    el('span', { class: 'mono', text: it.id }),
    bits.length ? el('span', { class: 'se', text: bits.join(' · ') }) : '');
}

// the list and the input's ARIA, in place: a search landing must not rebuild
// the page around the person typing
function msPaint(key) {
  const s = msState(key);
  for (const box of document.querySelectorAll(`[data-ms="${key}"]`)) {
    const input = box.querySelector('input');
    const list = box.querySelector('[role="listbox"]');
    const show = s.open && (s.items.length > 0 || !!s.footer);
    input.setAttribute('aria-expanded', String(show));
    if (show && s.active >= 0) input.setAttribute('aria-activedescendant', `ms-${key}-${s.active}`);
    else input.removeAttribute('aria-activedescendant');
    list.hidden = !show;
    const kids = [];
    let hubShown = false;
    s.items.forEach((it, i) => {
      if (it.source === 'hub' && !hubShown) {
        hubShown = true;
        kids.push(el('li', { role: 'presentation', class: 'ms-sep', text: 'Hugging Face Hub' }));
      }
      kids.push(msItem(key, it, i, s.active));
    });
    if (s.open && !s.items.length && !s.footer)
      kids.push(el('li', { role: 'presentation', class: 'ms-sep', text: 'no matches' }));
    if (s.footer) kids.push(el('li', { role: 'presentation', class: 'ms-foot',
      'data-ms-footer': '1', text: s.footer }));
    list.replaceChildren(...kids);
    const act = list.querySelector('.ms-item.active');
    if (act && act.scrollIntoView) act.scrollIntoView({ block: 'nearest' });
  }
}

function modelBox(key, value, onInput, onPick, attrs = {}) {
  const s = msState(key);
  s.onPick = onPick;
  const input = el('input', Object.assign({ type: 'text', role: 'combobox',
    'aria-autocomplete': 'list', 'aria-controls': `ms-list-${key}`, 'aria-expanded': 'false',
    autocomplete: 'off', spellcheck: 'false', 'data-keep': 'ms-' + key, value }, attrs, {
    oninput: e => { onInput(e.target.value); msQuery(key, e.target.value); },
    onkeydown: e => msKey(key, e),
    onblur: () => setTimeout(() => {
      // a poll re-render moves focus to the new copy of this box; that is not
      // the person leaving it
      const a = document.activeElement;
      if (a && a.dataset && a.dataset.keep === 'ms-' + key) return;
      if (s.open) { s.open = false; msPaint(key); }
    }, 150) }));
  const box = el('div', { class: 'ms', 'data-ms': key, style: attrs.style ? null : null },
    input, el('ul', { id: `ms-list-${key}`, role: 'listbox', class: 'ms-list', hidden: '',
      'aria-label': 'matching models' }));
  // painted once it is in the document
  queueMicrotask(() => msPaint(key));
  return box;
}

// ---------------------------------------------------------------------------
// A page left open across a deploy runs the old code forever: it never
// reloads its own JavaScript. Every /api/* answer carries the build that
// served it; when it differs from the one this page was built as, a bar says
// so, and after a minute the page reloads itself — never while someone is
// typing or a dialog is open. The hash survives a reload.
// ---------------------------------------------------------------------------
const BUILD = (document.querySelector('meta[name="evalboard-build"]') || {}).content || '';
let _newBuild = '';

function checkBuild(r) {
  const b = r && r.headers ? r.headers.get('X-Evalboard-Build') : '';
  if (!BUILD || !b || b === BUILD || _newBuild) return;
  _newBuild = b;
  buildBar();
}

function typingNow() {
  const a = document.activeElement;
  if (!a || a === document.body) return false;
  if (a.isContentEditable || a.tagName === 'TEXTAREA' || a.tagName === 'SELECT') return true;
  return a.tagName === 'INPUT'
    && !['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color'].includes(a.type);
}
const dialogOpen = () => !!document.querySelector('[role="dialog"]');

function buildBar() {
  const wait = window.__evalboardReloadMs || 60000;
  const due = Date.now() + wait;
  const secs = el('span', { class: 'se' });
  const bar = el('div', { class: 'buildbar', role: 'status', 'data-build-bar': _newBuild },
    el('b', { text: 'The dashboard was updated. ' }), 'Reload to get the new version. ',
    el('button', { class: 'primary', text: 'Reload', 'data-reload': '1',
      onclick: () => location.reload() }), ' ', secs);
  document.body.prepend(bar);
  const tick = setInterval(() => {
    const left = Math.max(0, Math.ceil((due - Date.now()) / 1000));
    const held = typingNow() || dialogOpen();
    secs.textContent = held ? 'It reloads by itself once you stop typing.'
      : left ? `Reloading by itself in ${left}s.` : 'Reloading…';
    if (!left && !held) { clearInterval(tick); location.reload(); }
  }, 500);
}

// ---------------------------------------------------------------------------
// Propose over a provisional judge: an in-page dialog, not window.confirm. The
// reasons in words, what the mark means, the name it is recorded under, and a
// box to tick before "Propose anyway" can be pressed. Esc cancels; focus stays
// inside and goes back to the button that opened it.
// ---------------------------------------------------------------------------
function lbMini(rows) {
  // 11f: the Leaderboard's rule — the leaders of the whole board (the best
  // average and every one the z-test cannot tell from it) are bold and
  // tinted, the rest plain, and the ± is the cell's tooltip
  const pool = DATA.models.filter(m => officialAvg(m) != null && !m.duplicateOf)
    .sort((a, b) => officialAvg(b) - officialAvg(a));
  const best = pool[0];
  const lead = m => !!best && (m.id === best.id || tiedWithBest({ key: 'avg' }, m,
    { id: best.id, v: officialAvg(best) }, null));
  const tb = el('tbody', {}, rows.map(m => { const r = rankOf(m), se = officialSe(m);
    const v = (100 * officialAvg(m)).toFixed(1), on = pool.length > 1 && lead(m);
    return el('tr', { 'data-top-row': m.id },
      el('td', { class: 'num mono se', text: r ? String(r.n) : '—' }),
      el('td', { class: 'model', 'data-model': m.id, style: `--fam:${famColor(m)}` },
        el('a', { href: '#model=' + encodeURIComponent(m.id), class: 'mlink', text: m.name }),
        ckBadge(m) || (m.kind === 'instruct'
          ? el('span', { class: 'badge instruct', text: 'instruct' }) : '')),
      el('td', { class: 'num', text: P(m.params) }),
      el('td', { class: 'num tcell' + (on ? ' lead' : ''), 'data-lead': on ? '1' : null,
          style: on ? 'background:var(--heat-3)' : null, tabindex: '0',
          'data-tip': JSON.stringify([v + (se != null ? ` ± ${(100 * se).toFixed(1)}` : ''),
            `${m.name} · Avg`, ...(on ? [m.id === best.id ? 'best on the board'
              : 'within the noise of the best'] : [])]) },
        on ? el('b', { text: v }) : v)); }));
  return el('table', { class: 'lb mini-lb tinted' },
    el('thead', {}, el('tr', { class: 'names' },
      el('th', { class: 'num', text: '#' }), el('th', { text: 'Model' }),
      el('th', { class: 'num', text: 'Params' }),
      el('th', { class: 'num', 'data-tip': JSON.stringify([`Avg — mean of the required tasks, % `
        + (state.avgMode === 'raw' ? 'raw accuracy' : 'above chance')]), tabindex: '0',
        text: 'Avg' }))), tb);
}

// ---------- capability profile: the radar ----------
// The Eval-Gauntlet-style view: one axis per benchmark (or per category), one
// polygon per model. Two honesty rules baked in: axes are scaled *above
// chance* by default — 25% on a 4-way task is 0, a perfect score is 100% —
// because raw accuracy makes a 2-way task look twice as strong as a 4-way one;
// and the overlapping-polygon form caps at three models (the all-pairs ladder),
// with colors that follow each model rather than its position in the list.
const CATS = [
  ['knowledge',    ['mmlu']],
  ['commonsense',  ['hellaswag', 'piqa', 'winogrande']],
  ['reasoning',    ['arc_challenge', 'arc_easy']],
  ['math',         ['gsm8k']],
  ['truthfulness', ['truthfulqa_mc2']],
];
function radarAxes() {
  if (state.radarAxes === 'tasks') return DATA.accTasks.map(t => ({ key: t, label: t, tasks: [t] }));
  const used = new Set(), out = [];
  for (const [name, ts] of CATS) {
    const have = ts.filter(t => DATA.accTasks.includes(t));
    if (have.length) { out.push({ key: name, label: name, tasks: have }); have.forEach(t => used.add(t)); }
  }
  for (const t of DATA.accTasks) if (!used.has(t)) out.push({ key: t, label: t, tasks: [t] });
  return out;
}
function normScore(t, v) {
  const c = (DATA.tasks[t] || {}).chance;
  if (state.avgMode === 'raw' || !(c > 0)) return v;
  return Math.max(0, Math.min(1, (v - c) / (1 - c)));
}
// How many models the capability profile will hold at once. The cap exists for
// the chart, not the code: past about five overlapping polygons a radar stops
// being readable. Five is the ceiling the eight-slot palette and the eye agree
// on. The eviction below used to be silent, which read as a bug — it is now
// labelled on the card and the evicted row is named.
const CMP_MAX = 5;

// the compared set: explicit ticks, else the top few by average
// what the radar draws: the models someone ticked, and none until they do —
// five pre-ticked boxes read as a choice the page had made for you
function cmpEffective(ms) {
  return state.cmpSel.filter(id => ms.some(m => m.id === id));
}
function cmpToggle(id, ms) {
  if (!state.cmpSel.length) {        // first tick: materialize the default so it edits intuitively
    state.cmpSel = cmpEffective(ms);
    state.cmpColors = {};
    state.cmpSel.forEach((x, i) => { state.cmpColors[x] = i; });
  }
  const i = state.cmpSel.indexOf(id);
  if (i >= 0) { state.cmpSel.splice(i, 1); delete state.cmpColors[id]; }
  else {
    if (state.cmpSel.length >= CMP_MAX) {  // FIFO: the newest tick always lands
      const old = state.cmpSel.shift(); delete state.cmpColors[old];
      const gone = DATA.models.find(x => x.id === old);
      state.cmpEvicted = gone ? gone.name : old;   // said out loud on the card
    } else state.cmpEvicted = '';
    const used = new Set(Object.values(state.cmpColors));
    let slot = 0; while (used.has(slot)) slot++;
    state.cmpColors[id] = slot;      // color follows the model while it is compared
    state.cmpSel.push(id);
  }
  render();
}
// The benchmark descriptions, where someone reading the table can actually find
// them. A tooltip alone is not an answer for a friend who has never seen the
// board — this is collapsed by default so it costs nothing, and open it once and
// every column has a sentence.
function aboutBenchmarks(tasks, inline = false) {
  const body = el('div', { class: 'about-body' }, tasks.map(t => {
    const i = DATA.tasks[t] || {};
    return el('div', { class: 'about-item' },
      el('div', { class: 'about-head' },
        el('span', { class: 'about-name', text: taskLabel(t) }),
        i.domain ? el('span', { class: 'tchip dom', text: i.domain }) : '',
        i.options ? el('span', { class: 'tchip', text: i.options + '-choice' }) : '',
        i.chance > 0 ? el('span', { class: 'tchip', text: 'chance ' + pct(i.chance) }) : '',
        i.frontier ? el('span', { class: 'tchip front',
          text: 'frontier ' + pct(i.frontier.v),
          title: `${i.frontier.src}, as of ${i.frontier.asof} — a different protocol `
               + `from ours` }) : '',
        i.discriminates === false ? el('span', { class: 'tchip flat', text: 'no separation',
          title: 'no pair of models here differs by more than their combined error' }) : ''),
      el('p', { class: 'about-desc', text: i.desc || 'No description recorded for this task.' }));
  }));
  // 11c: inside "How to read this table", which is itself the disclosure
  if (inline) return el('div', { class: 'about', 'data-about': '1' },
    el('div', { class: 'eyebrow', text: `About these benchmarks (${tasks.length})` }), body);
  return el('div', { class: 'card about' },
    el('button', { class: 'about-toggle', 'aria-expanded': String(state.lbAbout),
      onclick: () => { state.lbAbout = !state.lbAbout; render(); },
      text: (state.lbAbout ? '▾' : '▸')
          + ` About these benchmarks (${tasks.length})` }),
    state.lbAbout ? body : '');
}

// ---------------------------------------------------------------------------
// The Models tab. Thirty-three models, nineteen of them checkpoints, and until
// now the only ways to one were the Overview top five, a leaderboard row, or
// knowing its hash. The two filter rows that used to float above the tabs live
// here, where their scope is visible: they filter THIS list.
// ---------------------------------------------------------------------------

const MCOLS = [
  { key: 'name', label: 'model' },
  { key: 'kind', label: 'kind' },
  { key: 'family', label: 'family' },
  { key: 'params', label: 'params', num: true, defDir: -1 },
  { key: 'avg', label: 'official average', num: true, defDir: -1 },
  { key: 'judged', label: 'judged topics', num: true, defDir: -1 },
  { key: 'date', label: 'last evaluated', defDir: -1 },
];

function mdlValue(m, key) {
  if (key === 'avg') return officialAvg(m);
  if (key === 'date') return lastEval(m);
  if (key === 'judged') return m.judge ? Object.keys(m.judge.tasks || {}).length : 0;
  if (key === 'params') return m.params || 0;
  return m[key];
}

function mdlVisible() {
  const f = state.mdl;
  const q = (f.q || '').toLowerCase();
  let ms = DATA.models.filter(m =>
    (f.src === 'all' || m.source === f.src) &&
    (f.kind === 'all' || m.kind === f.kind) &&
    (f.family === 'all' || m.family === f.family) &&
    (!f.judgedOnly || (m.judge && Object.keys(m.judge.tasks || {}).length)) &&
    (!f.taintedOnly || (m.tainted || []).length) &&
    (!f.prelimOnly || officialAvg(m) == null) &&
    (!q || m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q)
        || (m.family || '').includes(q)));
  const c = MCOLS.find(x => x.key === f.sort.key) || MCOLS[0];
  return ms.sort((a, b) => {
    const va = mdlValue(a, c.key), vb = mdlValue(b, c.key);
    if (va == null && vb == null) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return f.sort.dir * (c.num ? va - vb : natCmp(va, vb));
  });
}

function vModels() {
  const f = state.mdl;
  const families = [...new Set(DATA.models.map(m => m.family).filter(Boolean))].sort();
  const ms = mdlVisible();
  const nCk = DATA.models.filter(m => m.source === 'artifact').length;
  const count = pred => DATA.models.filter(pred).length;
  // 11c: the same "Label: Value ▾" pills as the Leaderboard, on the shared
  // popover. The three yes/no filters combine, so they are one pill of boxes
  const shows = [['judged', 'has a judged run', 'judgedOnly'], ['tainted', 'tainted', 'taintedOnly'],
                 ['preliminary', 'preliminary', 'prelimOnly']];
  const on = shows.filter(([, , k]) => f[k]);
  const showBtn = el('button', { class: 'pill' + (on.length ? ' on' : ''), id: 'pill-mshow',
    'data-show-filters': on.map(([v]) => v).join(' '),
    text: `Show: ${on.length ? on.map(([, t]) => t).join(', ') : 'all'} ▾` });
  const showPill = popover(showBtn, () => el('div', { class: 'moremenu colmenu-list', id: 'pop-mshow',
      'aria-label': 'show only' },
    shows.map(([v, t, k]) => el('label', { class: 'small' },
      el('input', { type: 'checkbox', 'data-filter': v, checked: f[k] ? '' : null,
        onchange: e => { f[k] = e.target.checked; render(); } }), ' ' + t))),
    { key: 'mshow', menu: false, rebuild: true });
  const head = el('div', { class: 'card' },
    el('h2', { text: 'Models' }),
    el('p', { class: 'sub', text: 'Every model on this board, with what is known about it. '
      + 'The filters are this table\'s — they narrow the list below and nothing else. The '
      + 'radar and its model chips belong to the Leaderboard, where they are.' }),
    el('div', { class: 'toolbar lbbar' },
      el('input', { type: 'search', id: 'mq', value: f.q, style: 'flex:1;min-width:180px',
        placeholder: 'name, id or family…', 'aria-label': 'filter models',
        oninput: e => { f.q = e.target.value; render(); } }),
      el('div', { class: 'pills' },
        pillMenu('mkind', 'Kind', [['all', `All (${DATA.models.length})`],
          ['base', `base (${count(m => m.kind === 'base')})`],
          ['instruct', `instruct (${count(m => m.kind === 'instruct')})`]],
          f.kind, v => { f.kind = v; render(); }),
        nCk ? pillMenu('msrc', 'Source', [['all', 'All'], ['hub', 'Hub models'],
          ['artifact', `checkpoints (${nCk})`]], f.src, v => { f.src = v; render(); }) : '',
        pillMenu('mfamily', 'Family', [['all', 'any'], ...families.map(x => [x, x])],
          f.family, v => { f.family = v; render(); }),
        showPill),
      el('span', { class: 'count-note', 'data-model-count': String(ms.length),
        text: `${ms.length} of ${DATA.models.length} models` })));
  const th = c => el('th', { class: c.num ? 'num' : null,
    style: c.key ? 'cursor:pointer' : null, 'data-sort': c.key || '',
    onclick: c.key ? () => {
      f.sort = f.sort.key === c.key ? { key: c.key, dir: -f.sort.dir }
                                    : { key: c.key, dir: c.defDir || 1 };
      render();
    } : null },
    c.label + (f.sort.key === c.key ? (f.sort.dir > 0 ? ' ▲' : ' ▼') : ''));
  const pg = paged('models', ms, JSON.stringify([f.q, f.kind, f.src, f.family, f.judgedOnly,
                                                 f.taintedOnly, f.prelimOnly, f.sort]));
  const table = el('div', { class: 'card' },
    statusLine(pg, 'models', [`sorted by ${(MCOLS.find(c => c.key === f.sort.key) || {}).label
      || f.sort.key} ${f.sort.dir > 0 ? '▲' : '▼'}`]),
    pg.pager,
    el('div', { class: 'lb-wrap stick' }, el('table', { class: 'jd', 'data-models-table': '1' },
      el('thead', {}, el('tr', {}, MCOLS.map(th), el('th', { text: 'flags' }))),
      el('tbody', {}, pg.rows.map(m => {
        const topics = m.judge ? Object.keys(m.judge.tasks || {})
          .filter(t => t !== (DATA.judged || {}).control) : [];
        // no compare tick here: the radar is the Leaderboard's, and two tables
        // sharing one selection is what broke the Leaderboard's own ticks
        return el('tr', { 'data-model-row': m.id },
          el('td', {}, el('a', { href: '#model=' + encodeURIComponent(m.id), text: m.name,
            onclick: e => { e.preventDefault(); navigate({ model: m.id, topic: null }); } }),
            dupBadge(m) || '',
            el('div', { class: 'se mono', text: m.id })),
          el('td', {}, el('span', { class: 'badge' + (m.kind === 'instruct' ? ' instruct' : ''),
            text: m.kind }), m.source === 'artifact' ? el('span', { class: 'badge',
              text: 'ckpt' }) : ''),
          el('td', { class: 'small', text: m.family || '—' }),
          el('td', { class: 'num se', text: P(m.params) }),
          el('td', { class: 'num' }, officialAvg(m) != null ? pct(officialAvg(m), 1)
            : el('span', { class: 'se', title: (m.kindReason || '') + ' ' +
                ((m.judgeState || {}).reasons || []).join('; '),
                text: `preliminary ${m.nhave ?? 0}/${m.nreq ?? 0}` })),
          el('td', { class: 'num' }, topics.length
            ? el('span', { title: topics.map(t => `${frName(t)} ${num(pubScore(m.judge.tasks[t]), 2)}`)
                  .join(' · ') },
                String(topics.length),
                (m.judgeState && m.judgeState.ok) ? '' : el('span', { class: 'badge taint',
                  title: ((m.judgeState || {}).reasons || []).join('; '), text: 'not ranked' }))
            : el('span', { class: 'se', text: '—' })),
          el('td', { class: 'small se', text: lastEval(m) ? String(lastEval(m)).slice(0, 10) : '—' }),
          el('td', {}, (m.tainted || []).length ? el('span', { class: 'badge taint',
            title: 'trained on data derived from ' + m.tainted.map(frName).join(', '),
            text: 'tainted' }) : '',
            m.provisional ? el('span', { class: 'badge taint', text: 'provisional' }) : ''));
      })))));
  if (!ms.length) table.append(empty('No model matches these filters.', 'Clear the filters',
    () => { Object.assign(state.mdl, { q: '', kind: 'all', src: 'all', family: 'all',
      judgedOnly: false, taintedOnly: false, prelimOnly: false }); render(); }));
  return [head, table];
}

// ===========================================================================
// The Leaderboard (11c). One toolbar row: topic-group chips on the left and
// "Label: Value ▾" filter pills on the right, every one of them on 11a's
// popover so a poll never closes it. Two header rows, one-line cells tinted
// by their rank on the whole board, rows that open in place, and Insights
// under the table. The view lives in the hash, so a pasted link reproduces it.
// ===========================================================================

const LB_CHIPS = [
  ['all', 'All tasks'], ['knowledge', 'Knowledge'], ['commonsense', 'Commonsense'],
  ['reasoning', 'Reasoning'], ['math', 'Math'], ['truthfulness', 'Truthfulness'],
  ['judged', 'Judged topics']];
const LB_GROUP = { knowledge: 'Knowledge', commonsense: 'Commonsense', reasoning: 'Reasoning',
  math: 'Math', truthfulness: 'Truthfulness' };
const LB_KINDS = [['all', 'All'], ['base', 'base'], ['instruct', 'instruct'],
  ['checkpoint', 'checkpoint']];
const LB_SIZES = [['all', 'All'], ['s', '< 200M'], ['m', '200M–1B'], ['l', '1–3B'],
  ['xl', '> 3B']];
const LB_STATUS = [['all', 'All'], ['ranked', 'ranked'], ['preliminary', 'preliminary'],
  ['tainted', 'tainted']];
const LB_DEFAULTS = { chip: 'all', kind: 'all', size: 'all', status: 'all' };

// everything the Leaderboard remembers lives here, not in the DOM: a render
// every five seconds rebuilds the DOM, and an opened row must survive it
function lbS() {
  if (!state.lb) {
    let tint = true, howto = false;
    try {
      tint = localStorage.getItem('bench-lb-tint') !== 'off';
      howto = localStorage.getItem('bench-lb-howto') === 'open';
      state.lbSe = localStorage.getItem('bench-lb-se') === 'on';
    } catch (e) { /* private mode: the defaults */ }
    state.lb = { ...LB_DEFAULTS, open: [], models: null, tint, howto, shown: {},
                 focus: null, weak: null, radarSrc: 'tasks' };
  }
  return state.lb;
}

// the part of the hash after "tab=leaderboard": only what differs from the
// defaults, so a plain link stays plain
function lbHash() {
  const L = lbS(), out = [];
  for (const k of Object.keys(LB_DEFAULTS))
    if (L[k] !== LB_DEFAULTS[k]) out.push(`${k}=${encodeURIComponent(L[k])}`);
  if (L.open.length) out.push('open=' + L.open.map(encodeURIComponent).join(','));
  return out.join('&');
}
function lbFromHash(rest) {
  const L = lbS(), p = new URLSearchParams(rest || '');
  L.chip = LB_CHIPS.some(([v]) => v === p.get('chip')) ? p.get('chip') : 'all';
  if (L.chip === 'judged' && !judgedCalibrated()) L.chip = 'all';
  L.kind = LB_KINDS.some(([v]) => v === p.get('kind')) ? p.get('kind') : 'all';
  L.size = LB_SIZES.some(([v]) => v === p.get('size')) ? p.get('size') : 'all';
  L.status = LB_STATUS.some(([v]) => v === p.get('status')) ? p.get('status') : 'all';
  L.open = (p.get('open') || '').split(',').filter(id => DATA.models.some(m => m.id === id));
}
// a change to the view: the address bar follows without a history entry per
// click, and the page paints
function lbSet(patch) {
  Object.assign(lbS(), patch);
  const want = hashFor();
  if (location.hash.slice(1) !== want) history.replaceState(history.state, '', '#' + want);
  render();
}

// the family a model belongs to: the Hub organisation, or a local run's name
// without its step. Coloured by the series palette in name order; past eight,
// grey. Never colour alone: the family is in the tooltip, the opened row and
// the Models popover.
function famOf(m) {
  if (m.id.startsWith('local/')) return m.id.slice(6).replace(/-step\d+$/, '');
  return m.id.includes('/') ? m.id.split('/')[0] : m.family || m.id;
}
let _famKey = null, _famMap = null;
function famColor(m) {
  if (_famKey !== DATA.models.length) {
    const fams = [...new Set(DATA.models.map(famOf))].sort(natCmp);
    _famMap = new Map(fams.map((f, i) => [f, i < 8 ? `var(--s${i + 1})` : 'var(--axis)']));
    _famKey = DATA.models.length;
  }
  return _famMap.get(famOf(m)) || 'var(--axis)';
}

const officialSe = m => state.avgMode === 'raw' ? m.avgRawSe : m.avgSe;
// "2026-09-22 15:17 +04" -> "15:17": the clock, never the zone (11e)
const refreshedAt = () => (String((DATA || {}).generated || '').match(/\b\d{1,2}:\d{2}\b/)
  || [''])[0];
const judgedCalibrated = () => !!(DATA.judged && DATA.judged.calibration
  && DATA.judged.calibration.calibrated && DATA.models.some(m => m.judgeState && m.judgeState.ok));
const judgedOkM = m => !!(m.judgeState && m.judgeState.ok);
// why the judged chip, column set and radar source are off, in words
function judgedOffWhy() {
  const cal = (DATA.judged || {}).calibration;
  const local = DATA.models.some(m => m.judge && ((m.judgeState || {}).reasons || [])
    .some(r => /local|provisional/i.test(r)));
  return 'judged columns appear once a person has agreed with the judge — today: not '
    + 'calibrated' + (cal ? '' : ' (no calibration on file)') + (local ? ', local judge' : '');
}

// a diagnosis made with the old categories (before the 37-topic exam) names
// topics this board no longer has: its areas would all be empty (11e)
function diagStale(m) {
  const cats = mmluCats(m);
  const now = new Set(DATA.meta.categories || []);
  return !!cats && Object.keys(cats).some(k => !now.has(k));
}
function staleSentence(ms) {
  const stale = ms.filter(diagStale);
  if (!stale.length) return '';
  const n = Math.max(...stale.map(m => Object.keys(mmluCats(m)).length));
  return stale.length === 1 && ms.length === 1
    ? `MMLU by area needs a fresh diagnosis — this one was made with the old ${n} categories.`
    : `MMLU by area needs a fresh diagnosis — ${stale.length === ms.filter(mmluCats).length
        ? 'the diagnoses on this board were' : `${stale.length} of them were`} made with the `
      + `old ${n} categories.`;
}

// MMLU by area: the leaderboard-half items of every MMLU subject mapped to an
// area's topics, pooled — a mean weighted by item count
// 11h: one scale for an area's number, wherever it is — the column, the
// opened row's bars and the radar all follow the Scale pill (the table said
// 44.7 raw and the row 26 above chance, for the same number)
const areaScaled = v => state.avgMode === 'raw' ? v : Math.max(0, (v - 0.25) / 0.75);
const scaleWords = () => state.avgMode === 'raw' ? 'raw accuracy' : 'above chance';
function areaMmlu(m, area) {
  const cats = mmluCats(m);
  if (!cats || diagStale(m)) return null;
  let s = 0, n = 0;
  for (const t of (DATA.meta.areas || {})[area] || []) {
    const g = cats[t];
    if (g && g.score_report != null && g.n_report) { s += g.score_report * g.n_report; n += g.n_report; }
  }
  return n ? { v: s / n, n, se: Math.sqrt(Math.max(s / n * (1 - s / n), 1e-9) / n) } : null;
}
// a judged area mean, by judged_avg()'s rules: only a calibrated judge, no
// tainted topic, and at least half the area's topics judged
function areaJudged(m, area) {
  const topics = (DATA.meta.areas || {})[area] || [];
  const tasks = topics.map(t => 'exam_' + slugOfTopic(t)).filter(t => t !== 'exam_');
  const out = { v: null, k: 0, n: tasks.length };
  if (!judgedOkM(m)) return out;
  const vals = [];
  for (const t of tasks) {
    const j = ((m.judge || {}).tasks || {})[t];
    if (!j || (m.tainted || []).includes(t)) continue;
    const v = pubScore(j);
    if (v != null) vals.push(v);
  }
  out.k = vals.length;
  if (vals.length && vals.length * 2 >= tasks.length)
    out.v = vals.reduce((a, b) => a + b, 0) / vals.length;
  return out;
}

// the columns a chip shows, with the group each one sits under
function lbColumns(ms) {
  const L = lbS();
  if (L.chip === 'judged' && !judgedCalibrated()) L.chip = 'all';
  const areas = Object.keys(DATA.meta.areas || {});
  const groupOf = t => {
    for (const [g, ts] of CATS) if (ts.includes(t)) return LB_GROUP[g];
    return DATA.pplTasks.includes(t) ? 'Perplexity' : 'Other tasks';
  };
  const shot = t => {
    const s = [...new Set(ms.map(m => (cell(t, m.id) || {}).shots).filter(x => x != null))];
    return s.length > 1 ? 'mixed!' : s.length ? s[0] + '-shot' : '';
  };
  const task = t => {
    const lower = DATA.pplTasks.includes(t);
    return { key: t, label: t, short: LB_SHORT[t] || t, num: true, task: t, lower,
             group: groupOf(t), shot: shot(t),
             unit: lower ? (DATA.tasks[t] || {}).metric : [shot(t), '%'].filter(Boolean).join(' · ') };
  };
  const judgedCols = () => (DATA.judged && DATA.judged.exam || [])
    .filter(t => (DATA.judged.tasks || []).includes(t)).map(t => ({
      key: 'j:' + t, label: frName(t), num: true, judged: t, optional: true,
      group: 'Judged · ' + (Object.entries(DATA.meta.areas || {})
        .find(([, ts]) => ts.includes(frName(t))) || ['topics'])[0],
      unit: 'agreement ' + ((((DATA.judged.calibration || {}).per_category || {})[frName(t)] || {}).kappa
                   ?? (DATA.judged.calibration || {}).kappa) + ' · 0–4' }));
  const cats = lbCategoryCols(ms).map(c => ({ ...c, label: c.cat, optional: true,
    group: 'MMLU by topic', unit: 'hidden questions · %' }));
  // 11f: the judged average is a column only while some model on the page
  // has one — a column of dashes says nothing
  const javg = ms.some(m => m.judgedAvg != null)
    ? [{ key: 'javg', num: true, judged: 'avg', group: 'Judged', short: 'Judged',
         label: 'Judged avg', unit: judgedCalibrated()
           ? `agreement ${DATA.judged.calibration.kappa} · 0–4` : 'rubric 0–4' }] : [];
  const lead = [
    { key: 'rank', label: '#', group: '', nosort: true },
    { key: 'name', label: 'Model', group: '' },
    { key: 'params', label: 'Params', num: true, group: '' },
    { key: 'avg', label: 'Avg', num: true, group: '',
      unit: state.avgMode === 'raw' ? 'raw · %' : 'above chance · %' }];
  const tail = [{ key: 'date', label: 'Updated', group: '' }];
  // an area with no number for any model here is not a column (11e)
  const areasHere = areas.filter(a => ms.some(m => areaMmlu(m, a)));
  let mid;
  if (L.chip === 'all') {
    // today's view: the harness tasks, six by default, the rest one tick away
    const order = [...CATS.map(([g]) => LB_GROUP[g]), 'Other tasks', 'Perplexity'];
    mid = [...DATA.accTasks.map(task), ...DATA.pplTasks.map(task)]
      .map(c => ({ ...c, optional: true }))
      .sort((a, b) => order.indexOf(a.group) - order.indexOf(b.group));
    if (judgedCalibrated()) mid.push(...judgedCols());
    mid.push(...cats, ...javg);
  } else if (L.chip === 'knowledge') {
    mid = [...(CATS.find(([g]) => g === 'knowledge')[1]).filter(t => DATA.accTasks.includes(t))
             .map(task),
           ...areasHere.map(a => ({ key: 'area:' + a, label: a, num: true, area: a,
             group: 'MMLU by area', unit: 'hidden questions · %' })),
           ...cats];
  } else if (L.chip === 'judged') {
    mid = judgedCalibrated()
      ? [...areas.map(a => ({ key: 'jarea:' + a, label: a, num: true, jarea: a,
            group: 'Judged by area', unit: 'mean · 0–4' })), ...javg, ...judgedCols()]
      : [...javg];
  } else {
    mid = (CATS.find(([g]) => g === L.chip) || [null, []])[1]
      .filter(t => DATA.accTasks.includes(t)).map(task);
  }
  return [...lead, ...mid, ...tail];
}

// 11f: one word per column name; the long ones are the tooltip's
const LB_SHORT = { arc_challenge: 'ARC-C', arc_easy: 'ARC-E', truthfulqa_mc2: 'TruthfulQA' };

// A column's setup, in words — its tooltip, and its accessible name. The
// header shows only the name; this is where the n-shot, the unit and the
// scale went (11f).
function lbColTip(c) {
  const scale = state.avgMode === 'raw' ? 'raw accuracy' : 'above chance';
  if (c.key === 'rank') return ['# — rank among the ranked models on this board'];
  if (c.key === 'name') return ['Model — sort by name'];
  if (c.key === 'params') return ['Params — parameter count, from the harness config or the name'];
  if (c.key === 'date') return ['Updated — when the model was last evaluated'];
  if (c.key === 'avg') return [`Avg — mean of the required tasks, % ${scale}`,
    'the Scale pill switches it'];
  if (c.task) {
    const info = DATA.tasks[c.task] || {};
    return [c.lower ? `${c.task} — ${c.unit}, lower is better`
                    : `${c.task} — ${c.shot || 'n-shot unknown'}, % accuracy`,
      ...(info.control ? ['CONTROL — never in Avg'] : []),
      ...[info.domain, info.desc].filter(Boolean)];
  }
  if (c.area) return [`${c.area} — MMLU, % ${scaleWords()}`, 'the hidden questions of each topic',
    'pooled over ' + ((DATA.meta.areas || {})[c.area] || []).join(', ')];
  if (c.cat) return [`MMLU ${c.cat} — hidden questions, %`];
  if (c.jarea) return [`${c.jarea} — judged mean, rubric 0–4`, 'never part of Avg'];
  if (c.judged === 'avg') return ['Judged — mean rubric score 0–4 over the judged topics',
    'from the calibrated judge — never part of Avg'];
  if (c.judged) return [`${c.label} — rubric 0–4, ${c.unit}`, 'never part of Avg'];
  return [c.label];
}

// which optional columns show, per chip: All tasks keeps its remembered six
function lbShownFor(cols) {
  const L = lbS();
  const opt = cols.filter(c => c.optional);
  if (L.chip === 'all') return lbShownTasks(opt);
  const want = L.shown[L.chip];
  // a chip's own group columns show; the long per-topic lists wait to be asked for
  return new Set(Array.isArray(want) ? want.filter(k => opt.some(c => c.key === k)) : []);
}
function lbSaveShown(next) {
  const L = lbS();
  if (L.chip === 'all') {
    state.lbShown = next;
    try { localStorage.setItem('bench-lb-shown', JSON.stringify(next)); } catch (e) { /* private */ }
  } else L.shown[L.chip] = next;
  render();
}

// the filters, for the rows: kind, size and status, and the Models popover's
// own selection
function lbFilter(ms) {
  const L = lbS();
  const sizeOk = m => {
    if (L.size === 'all') return true;
    const p = m.params;
    if (p == null) return false;
    return L.size === 's' ? p < 2e8 : L.size === 'm' ? p >= 2e8 && p < 1e9
      : L.size === 'l' ? p >= 1e9 && p < 3e9 : p >= 3e9;
  };
  return ms.filter(m =>
    (L.kind === 'all' || (L.kind === 'checkpoint' ? m.source === 'artifact' : m.kind === L.kind))
    && sizeOk(m)
    && (L.status === 'all'
        || (L.status === 'ranked' && officialAvg(m) != null)
        || (L.status === 'preliminary' && officialAvg(m) == null)
        || (L.status === 'tainted' && (m.tainted || []).length))
    && (!L.models || L.models.includes(m.id)));
}

// A "Label: Value ▾" pill with a single-choice menu on the shared popover
function pillMenu(key, label, opts, cur, pick, attrs = {}) {
  const now = (opts.find(([v]) => v === cur) || opts[0])[1];
  const btn = el('button', { class: 'pill' + (cur !== opts[0][0] ? ' on' : ''), id: 'pill-' + key,
    'data-pill': key, 'data-value': cur, text: `${label}: ${now} ▾`, ...attrs });
  return popover(btn, () => el('div', { class: 'moremenu', id: 'pop-' + key,
      'aria-label': label.toLowerCase() },
    opts.map(([v, t]) => el('button', { role: 'menuitemradio', 'data-choice': v,
      'aria-checked': String(v === cur), text: t,
      onclick: () => { popClose(true); pick(v); } }))), { key });
}

// 11f: each column's LEADERS — its best score, and every score the z-test
// cannot tell from it — over the whole board, so a filter never changes them.
// They are bold, and (with Tint on) tinted; every other cell is plain. This
// replaced five rank steps, which painted the top of a 14-model board one slab.
function lbLeaders(cols, val) {
  const out = {};
  for (const c of cols) {
    if (!c.num || c.key === 'params') continue;
    // provisional scores are never tinted: a judged column is on the board
    // only once the judge is calibrated, and a model whose judge is not ok
    // has no judged cell to tint
    const pool = DATA.models.filter(m => !m.duplicateOf
      && (c.key !== 'avg' || officialAvg(m) != null)
      && (!c.judged && !c.jarea || judgedOkM(m)))
      .map(m => ({ id: m.id, v: val(m, c) })).filter(x => x.v != null);
    if (pool.length < 2) continue;
    pool.sort((a, b) => c.lower ? a.v - b.v : b.v - a.v);
    const best = pool[0];
    // a lower-is-better column has no standard error: within 1% (floor
    // 0.005) of the best is inside the noise, as the Perplexity tab says
    const band = Math.max(0.005, 0.01 * Math.abs(best.v));
    const lead = new Set(pool.filter(x => x.id === best.id || (c.lower ? x.v <= best.v + band
      : tiedWithBest(c, DATA.models.find(m => m.id === x.id), best, val))).map(x => x.id));
    out[c.key] = { best, lead };
  }
  return out;
}

// tied with the best by the same z-test the rest of the board uses: a task's
// pairwise table, or two averages and their standard errors
function tiedWithBest(c, m, best, val) {
  if (!best || best.id === m.id) return false;
  if (c.task && !c.lower) {
    const row = (DATA.sig[c.task] || []).find(([a, b]) =>
      (a === best.id && b === m.id) || (a === m.id && b === best.id));
    return !!row && !row[4];
  }
  const pair = (sa, sb, va, vb) => sa != null && sb != null
    && Math.abs(va - vb) / Math.sqrt(sa * sa + sb * sb || 1e-12) <= 1.96;
  if (c.key === 'avg') {
    const b = DATA.models.find(x => x.id === best.id);
    return pair(officialSe(b), officialSe(m), officialAvg(b), officialAvg(m));
  }
  if (c.area) {
    const b = DATA.models.find(x => x.id === best.id);
    const x = areaMmlu(b, c.area), y = areaMmlu(m, c.area);
    return !!(x && y) && pair(x.se, y.se, x.v, y.v);
  }
  return false;
}

function vLeaderboard(ms) {
  const L = lbS();
  const cols = lbColumns(ms);
  const shown = lbShownFor(cols);
  const opt = cols.filter(c => c.optional);
  const nHidden = opt.length - opt.filter(c => shown.has(c.key)).length;
  const visCols = cols.filter(c => !c.optional || shown.has(c.key));
  const jval = (m, c) => c.judged === 'avg' ? m.judgedAvg
    : !judgedOkM(m) ? null
    : (m.tainted || []).includes(c.judged) ? null      // shown on the page, never ranked here
    : (((m.judge || {}).tasks || {})[c.judged] ? pubScore(m.judge.tasks[c.judged]) : null);
  const val = (m, c) => c.key === 'avg' ? officialAvg(m)
    : c.key === 'params' ? m.params
    : c.key === 'name' ? m.name
    : c.key === 'date' ? lastEval(m)
    : c.judged ? jval(m, c)
    : c.jarea ? areaJudged(m, c.jarea).v
    : c.area ? (areaMmlu(m, c.area) || {}).v
    : c.cat ? ((mmluCats(m) || {})[c.cat] || {}).score_report
    : c.task ? (cell(c.task, m.id) || {}).v : null;
  const rowsIn = lbFilter(ms);
  const sortCol = cols.find(c => c.key === state.sort.key) || cols.find(c => c.key === 'avg');
  const sorted = [...rowsIn].sort((a, b) => {
    const va = val(a, sortCol), vb = val(b, sortCol);
    if (va == null && vb == null) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return typeof va === 'string' ? state.sort.dir * natCmp(va, vb) : state.sort.dir * (va - vb);
  });
  // ranked rows first, whatever the sort: a preliminary model's per-task
  // numbers are valid, and it is still not on the ladder
  const ordered = [...sorted.filter(m => officialAvg(m) != null),
                   ...sorted.filter(m => officialAvg(m) == null)];
  const dupsOf = {};
  for (const m of ordered)
    if (m.duplicateOf && ordered.some(x => x.id === m.duplicateOf))
      (dupsOf[m.duplicateOf] = dupsOf[m.duplicateOf] || []).push(m);
  const lbAll = ordered.filter(m => !(m.duplicateOf && dupsOf[m.duplicateOf]));
  const lbPg = paged('leaderboard', lbAll, JSON.stringify([state.sort, state.q, state.kind,
    state.src, state.avgMode, L.chip, L.kind, L.size, L.status, L.models]));
  const rows = lbPg.rows.flatMap(m => [m, ...((state.lbDupOpen || {})[m.id] ? dupsOf[m.id] || [] : [])]);
  // the leaders are bold whatever the Tint switch says; Tint only adds the wash
  const leaders = lbLeaders(visCols, val);

  // ---- header (11f): one line of one-word names. The setup — n-shot, unit,
  // scale — is the name's tooltip, not three more lines; the group row is
  // quiet, and only on All tasks, where there is more than one group
  const groups = [];
  for (const c of visCols) {
    const g = c.group || '';
    if (groups.length && groups[groups.length - 1].g === g) groups[groups.length - 1].n++;
    else groups.push({ g, n: 1 });
  }
  const thead = el('thead', {},
    L.chip === 'all' ? el('tr', { class: 'grp' }, groups.map(({ g, n }) => el('th', {
      colspan: String(n), class: g ? 'grp' : 'grp nogrp', scope: 'colgroup', text: g }))) : '',
    el('tr', { class: 'names' }, visCols.map(c => {
      const tipRows = lbColTip(c);
      if (c.nosort) return el('th', { class: 'rank pin0', scope: 'col', 'data-col': c.key,
        'data-tip': JSON.stringify(tipRows), text: c.label });
      return el('th', { 'data-col': c.key, 'data-task': c.task || null, 'data-area': c.area || null,
        'data-jarea': c.jarea || null,
        class: (c.num ? 'num ' : '') + 'sortable' + (c.key === 'name' ? ' model pin' : '')
          + (c.judged || c.jarea ? ' judged' : ''),
        scope: 'col', 'data-tip': JSON.stringify(tipRows),
        'aria-label': tipRows.join(' — '),
        'aria-sort': state.sort.key === c.key ? (state.sort.dir > 0 ? 'ascending' : 'descending') : 'none',
        onclick: () => { state.sort = { key: c.key,
          dir: state.sort.key === c.key ? -state.sort.dir : (c.key === 'name' ? 1 : c.lower ? 1 : -1) };
          render(); } },
        el('span', { class: 'hname', text: c.short || c.label }),
        state.sort.key === c.key ? el('span', { class: 'dir', text: state.sort.dir > 0 ? ' ▲' : ' ▼' }) : '');
    })));

  // ---- a cell (11f): the number only. A leader — the column's best, or
  // inside its noise — is bold, and tinted with Tint on; nothing else is.
  // The ± is the cell's tooltip (hover or focus), in the opened row, and on
  // every cell with "Show ± errors"
  const one = (c, m, v, se, fmt, extra = {}) => {
    const t = leaders[c.key];
    const lead = !!(t && v != null && t.lead.has(m.id));
    const txt = v == null ? '—' : fmt(v);
    const { title, ...rest } = extra;
    return el('td', { class: 'num tcell' + (lead ? ' lead' : ''),
        style: lead && L.tint ? 'background:var(--heat-3)' : null,
        'data-lead': lead ? '1' : null, tabindex: v != null ? '0' : null,
        'data-watch': `lb|${m.id}|${c.key}`,
        'data-tip': v == null ? null : JSON.stringify([txt + (se != null ? ` ± ${se}` : ''),
          `${m.name} · ${c.short || c.label}`,
          ...(lead ? [t.best.id === m.id ? 'best in the column' : 'within the noise of the best']
            : []), ...(title ? [title] : [])]),
        ...rest },
      lead ? el('b', { text: txt }) : txt,
      // a thin space: the error belongs to the number
      se != null && state.lbSe ? el('span', { class: 'se', text: `\u2009±${se}` }) : '');
  };
  const pctn = v => (100 * v).toFixed(1);
  const ncols = visCols.length;
  const tbody = el('tbody', {});
  rows.forEach((m, i) => {
    const open = L.open.includes(m.id);
    const did = 'lbd-' + i;
    const tr = el('tr', { class: (m.duplicateOf && dupsOf[m.duplicateOf] ? 'duprow' : '')
        + (open ? ' open' : ''), 'data-lb-row': m.id,
      onclick: e => {
        // links, buttons, checkboxes and badges with a job of their own keep it
        if (e.target.closest('a, button, input, select, label, .badge[title]')) return;
        lbToggle(m.id);
      },
      // Esc on an opened row closes it, and the chevron has the focus back
      onkeydown: e => { if (e.key === 'Escape' && open) { e.preventDefault(); lbToggle(m.id, true); } } },
      visCols.map(c => {
        if (c.key === 'rank') {
          const r = rankOf(m);
          return el('td', { class: 'rank pin0 num' },
            // 11f: one chevron that turns; it starts unturned when this
            // render is the one that opens the row, so the turn is seen
            el('button', { class: 'disclose' + (open && state.lbAnim === m.id ? ' pre' : ''),
              'aria-expanded': String(open),
              'aria-controls': did, 'data-open-row': m.id,
              'aria-label': (open ? 'close ' : 'open ') + m.name,
              onclick: () => lbToggle(m.id) }, el('span', { class: 'chev', text: '▸' })),
            el('span', { class: 'mono', text: r ? String(r.n) : '—',
              title: r ? `rank ${r.n} of ${r.of} ranked models on this board`
                       : 'preliminary — not ranked' }));
        }
        // 11e: one clipped line — the name ellipsises, the badges and the
        // duplicate toggle follow inside the cell's own width, and nothing
        // paints into Params
        if (c.key === 'name') return el('td', { class: 'model pin', 'data-model': m.id,
            style: `--fam:${famColor(m)}`,
            title: modelSentence(m) + `\n\nfamily: ${famOf(m)}\n` + m.id },
          el('div', { class: 'mcell' },
            el('a', { class: 'mname mlink', text: m.name, href: '#model=' + encodeURIComponent(m.id) }),
            ckBadge(m) || (m.kind === 'instruct'
              ? el('span', { class: 'badge instruct', text: 'instruct' })
              : el('span', { class: 'badge', text: 'base' })),
            warnBadge(m) || '', dupBadge(m) || '',
            dupsOf[m.id] ? dupToggle(m, dupsOf[m.id]) : ''));
        if (c.key === 'params') {
          const a = m.archinfo || {};
          return el('td', { class: 'num', title: (m.paramsSrc ? 'from ' + (m.paramsSrc === 'config'
              ? 'harness config' : 'model name') : '') + (a.active_params
              ? `\n${P(a.active_params)} active · ${a.experts} experts, ${a.experts_per_tok} per token`
              : '') },
            P(m.params), a.active_params
              ? el('span', { class: 'act', text: ` · ${P(a.active_params)} act` }) : '');
        }
        if (c.key === 'date') {
          const d = String(lastEval(m) || '');
          const [y, mo, da] = d.slice(0, 10).split('-');
          // 11f: "15 Sep"; the year only when it is not this one
          const short = da ? `${+da} ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
            'Sep', 'Oct', 'Nov', 'Dec'][+mo - 1]}`
            + (+y !== new Date().getFullYear() ? ` ${y}` : '') : '—';
          return el('td', { class: 'num small nowrap', 'data-date': d.slice(0, 10),
            title: d.replace('T', ' '), text: short });
        }
        if (c.key === 'avg') {
          const a = officialAvg(m);
          if (a == null) return el('td', { class: 'num se',
            title: (m.missing || []).length ? 'missing: ' + m.missing.join(', ') : '',
            text: `— ${m.nhave}/${m.nreq}` });
          const se = officialSe(m);
          return one(c, m, a, se != null ? (100 * se).toFixed(1) : null, pctn, {
            title: `mean over the ${m.nreq} required tasks, `
              + (state.avgMode === 'raw' ? 'raw accuracy' : 'scaled so chance = 0') });
        }
        if (c.judged) {
          const v = jval(m, c);
          if (v == null) return el('td', { class: 'num se', text: '—',
            title: c.judged !== 'avg' && (m.tainted || []).includes(c.judged)
              ? 'trained on data derived from this topic — shown on the model page, not ranked'
              : m.judgeState ? (m.judgeState.reasons || []).join('; ') || 'not judged' : 'not judged' });
          const prelim = !judgedOkM(m);
          const td = prelim ? el('td', { class: 'num dim', text: num(v, 2) })
            : one(c, m, v, null, x => num(x, 2));
          if (c.judged === 'avg') td.dataset.judgedAvg = prelim ? 'preliminary' : 'counts';
          return td;
        }
        if (c.jarea) {
          const r = areaJudged(m, c.jarea);
          if (r.v == null) return el('td', { class: 'num se', text: '—', 'data-jarea-cell': c.jarea,
            title: `${r.k} of ${r.n} topics judged` + (judgedOkM(m) ? '' : ' — not ranked') });
          return one(c, m, r.v, null, x => num(x, 2), { title: `${r.k} of ${r.n} topics judged`,
            'data-jarea-cell': c.jarea });
        }
        if (c.area) {
          const r = areaMmlu(m, c.area);
          if (!r) return el('td', { class: 'num se', text: '—' });
          const k = state.avgMode === 'raw' ? 1 : 1 / 0.75;
          return one(c, m, areaScaled(r.v), (100 * r.se * k).toFixed(1), pctn,
            { title: `${r.n} questions · % ${scaleWords()}`, 'data-area-cell': c.area });
        }
        if (c.cat) {
          const g = (mmluCats(m) || {})[c.cat];
          if (!g || g.score_report == null) return el('td', { class: 'num se', text: '—' });
          if (g.n_report < CAT_MIN_N) return el('td', { class: 'num dim',
            title: `${g.n_report} items — under ${CAT_MIN_N}, treat as noise`,
            text: pctn(g.score_report) });
          return one(c, m, g.score_report, null, pctn, { title: `${g.n_report} leaderboard-half items` });
        }
        const cc = cell(c.task, m.id);
        if (!cc) return el('td', { class: 'num se', text: '—' });
        return one(c, m, cc.v, cc.se && !c.lower ? (100 * cc.se).toFixed(1) : null,
          c.lower ? x => num(x, 3) : pctn);
      }));
    tbody.append(tr);
    if (open) tbody.append(lbDetailRow(m, did, ncols, dupsOf[m.id]));
  });

  const table = el('table', { class: 'lb' + (L.tint ? ' tinted' : ''), 'data-lb-table': '1' },
    thead, tbody);
  return [el('div', { class: 'card', 'data-lb-card': '1' },
      el('h2', { text: 'Leaderboard' }),
      lbToolbar(ms, cols, shown, nHidden),
      L.chip === 'knowledge' && staleSentence(ms)
        ? el('p', { class: 'warn', 'data-stale-diag': '1', text: staleSentence(ms) }) : '',
      statusLine(lbPg, 'models', [
        `${lbAll.filter(m => officialAvg(m) != null).length} ranked`,
        `sorted by ${lbSortLabel(cols)}`,
        L.chip !== 'all' ? (LB_CHIPS.find(([v]) => v === L.chip) || [])[1] : null]),
      lbPg.pager,
      hfade('lb', el('div', { class: 'lb-wrap stick' + (state.lbWide ? ' hscroll' : ''),
        'data-hkeep': 'lb' }, table)),
      el('p', { class: 'lbcap', 'data-lb-caption': '1', text: 'Bold = best in the column or '
        + 'within its noise · hover a score for its ± error · hover a column name for its setup' }),
      lbHowTo(ms)),
    insightsCard(ms)];
}

// A box that scrolls sideways, with a fade on its right edge and a small
// "scroll →" while there is more to the right (11e)
function hfade(key, scroller) {
  return el('div', { class: 'hfade', 'data-hfade': key }, scroller,
    el('span', { class: 'scrollhint', 'aria-hidden': 'true', text: 'scroll →' }));
}
function hfadeUpdate(root) {
  // 11k: the Queue's table painted across its card at 1,512px — .stick left
  // the scroller with overflow:visible, so only the Leaderboard clipped. Now
  // every sticky table does, and gives up its page-sticky header while it
  // scrolls, exactly as the Leaderboard's does
  (root || document).querySelectorAll('.lb-wrap.stick').forEach(sc => {
    const t = sc.querySelector('table');
    if (!t) return;
    const wide = t.offsetWidth > sc.clientWidth + 1;
    sc.classList.toggle('hscroll', wide);
    if (sc.dataset.hkeep === 'lb') state.lbWide = wide;
  });
  (root || document).querySelectorAll('[data-hfade]').forEach(box => {
    const sc = box.firstElementChild;
    if (!sc) return;
    // 11h: a table wider than its card scrolls in its own box at every
    // width — the Knowledge chip pushed the whole page sideways at 1,512px.
    // Only then does its header give up sticking to the page
    const upd = () => {
      const ov = getComputedStyle(sc).overflowX;
      box.dataset.more = (ov === 'auto' || ov === 'scroll')
        && sc.scrollLeft + sc.clientWidth < sc.scrollWidth - 2 ? '1' : '0';
    };
    if (!sc._hfade) { sc._hfade = true; sc.addEventListener('scroll', upd, { passive: true }); }
    upd();
  });
}
window.addEventListener('resize', () => hfadeUpdate());

// "1 duplicate ▸": on the row, and in the opened row where a phone can reach it
function dupToggle(m, dups) {
  const open = !!(state.lbDupOpen || {})[m.id];
  return el('button', { class: 'quiet duptoggle', 'data-dup-toggle': m.id,
    'aria-expanded': String(open),
    text: `${dups.length} duplicate ${open ? '▾' : '▸'}`,
    onclick: e => { e.preventDefault(); state.lbDupOpen = state.lbDupOpen || {};
      state.lbDupOpen[m.id] = !open; render(); } });
}

// 11f: a row opens and closes as one motion — the height from 0fr to 1fr,
// the content fading in and rising 4px — only when a person toggles it. A
// row already open (a poll, a sort, a pasted link) is drawn open, still.
function motionOff() { return matchMedia('(prefers-reduced-motion: reduce)').matches; }

function lbToggle(id, refocus) {
  const L = lbS();
  const back = () => { if (refocus) state.after = {
    focus: `button[data-open-row="${CSS.escape(id)}"]` }; };
  if (!L.open.includes(id)) {
    state.lbAnim = motionOff() ? null : id;
    lbSet({ open: [...L.open, id] });
    return;
  }
  const wrap = document.querySelector(`[data-dwrap="${CSS.escape(id)}"]`);
  // the state closes now, so a poll in the middle does not open it again
  L.open = L.open.filter(x => x !== id);
  if (!wrap || motionOff()) { back(); lbSet({}); return; }
  const btn = document.querySelector(`button[data-open-row="${CSS.escape(id)}"]`);
  if (btn) btn.setAttribute('aria-expanded', 'false');
  wrap.classList.add('anim');
  requestAnimationFrame(() => wrap.classList.add('closed'));
  let done = false;
  const fin = () => { if (done) return; done = true; back(); lbSet({}); };
  wrap.addEventListener('transitionend', e => {
    if (e.target === wrap && e.propertyName === 'grid-template-rows') fin(); });
  setTimeout(fin, 400);        // a transition that never ends still ends
}

// the opened row: the row's own continuation — its accent bar down the left,
// a faint wash, ✕ Close at the top right — in a wrapper that can move
function lbDetailRow(m, did, ncols, dups) {
  const anim = state.lbAnim === m.id;
  const wrap = el('div', { class: 'dwrap' + (anim ? ' anim closed' : ''), 'data-dwrap': m.id },
    el('div', { class: 'dinner' }, el('div', { class: 'dpad' },
      el('button', { class: 'quiet dclose', 'data-detail-close': m.id,
        'aria-label': 'close ' + m.name, text: '✕ Close', onclick: () => lbToggle(m.id, true) }),
      lbDetail(m, dups))));
  if (anim) {
    state.lbAnim = null;                 // consumed: the next render draws it still
    requestAnimationFrame(() => requestAnimationFrame(() => {
      wrap.classList.remove('closed');
      const b = document.querySelector(`button.disclose.pre[data-open-row="${CSS.escape(m.id)}"]`);
      if (b) b.classList.remove('pre');
    }));
    wrap.addEventListener('transitionend', e => {
      if (e.target !== wrap || e.propertyName !== 'grid-template-rows') return;
      wrap.classList.remove('anim');
      // the panel ends below the screen: just enough scroll to show it
      const tr = wrap.closest('tr');
      if (tr) tr.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
  }
  return el('tr', { class: 'detail', id: did, 'data-lb-detail': m.id,
      onkeydown: e => { if (e.key === 'Escape' && !e.target.closest('[data-pop]')) {
        e.preventDefault(); e.stopPropagation(); lbToggle(m.id, true); } } },
    el('td', { colspan: String(ncols) }, wrap));
}

// ---- the toolbar: one row ---------------------------------------------------
function lbToolbar(ms, cols, shown, nHidden) {
  const L = lbS();
  const calOk = judgedCalibrated();
  const chips = el('div', { class: 'chips', role: 'group', 'aria-label': 'task groups' },
    LB_CHIPS.map(([v, t]) => {
      const off = v === 'judged' && !calOk;
      // 11e: an unavailable chip still takes the click (aria-disabled, not
      // disabled) — the click says why, in one line under the chips; the
      // tooltip and the screen reader say it without one
      return el('button', { class: 'chip-btn' + (L.chip === v ? ' on' : ''), 'data-chip': v,
        'aria-pressed': String(L.chip === v), 'aria-disabled': off ? 'true' : null,
        'aria-describedby': off ? 'why-judged-chip' : null,
        title: off ? judgedOffWhy() : null, text: t,
        onclick: () => { if (!off) lbSet({ chip: v });
          else { state.lbChipWhy = !state.lbChipWhy; render(); } } });
    }));
  const pills = el('div', { class: 'pills' },
    pillMenu('kind', 'Kind', LB_KINDS, L.kind, v => lbSet({ kind: v })),
    pillMenu('size', 'Size', LB_SIZES, L.size, v => lbSet({ size: v })),
    pillMenu('status', 'Status', LB_STATUS, L.status, v => lbSet({ status: v })),
    lbColumnsPill(cols, shown, nHidden),
    lbModelsPill(ms),
    pillMenu('scale', 'Scale', [['chance', 'above chance'], ['raw', 'raw accuracy']],
      state.avgMode, v => { state.avgMode = v; render(); }));
  const note = calOk ? '' : el('p', { class: 'chipnote', id: 'why-judged-chip', role: 'note',
    'data-why': 'judged-chip', hidden: state.lbChipWhy ? null : '', text: judgedOffWhy() });
  // 11h: on a phone the six pills are one "Filters ▾" beside the chips, and
  // open as a sheet from the bottom; the chips are one row that scrolls
  if (narrowLb()) {
    const set = [L.kind !== LB_DEFAULTS.kind, L.size !== LB_DEFAULTS.size,
      L.status !== LB_DEFAULTS.status, !!L.models, state.avgMode === 'raw'].filter(Boolean).length;
    const open = !!state.lbFilters;
    const toggle = el('button', { class: 'pill' + (set ? ' on' : ''), 'data-filters': String(set),
      'aria-expanded': String(open), 'aria-controls': 'filter-sheet',
      text: `Filters${set ? ` · ${set}` : ''} ▾`,
      onclick: () => { state.lbFilters = !open; render(); } });
    return el('div', { class: 'lbbar narrow' },
      el('div', { class: 'chiprow' }, chips, toggle), note,
      open ? el('div', { class: 'fsheet', id: 'filter-sheet', role: 'dialog', 'aria-label': 'filters',
          'data-filter-sheet': '1',
          onkeydown: e => { if (e.key === 'Escape' && !POP.panel) { state.lbFilters = false; render(); } } },
        el('div', { class: 'fsheet-head' }, el('b', { text: 'Filters' }),
          el('button', { class: 'ghost', text: 'Done', 'data-filters-done': '1',
            onclick: () => { state.lbFilters = false; render(); } })),
        pills) : '');
  }
  // the note sits right under the chips, where the click was
  return el('div', { class: 'lbbar' }, chips, note, pills);
}
const narrowLb = () => matchMedia('(max-width:720px)').matches;
// crossing 720px changes what the toolbar is made of: draw it again
matchMedia('(max-width:720px)').addEventListener('change', () => {
  state.lbFilters = false; if (DATA) render(); });

// Columns · 4 hidden ▾ — the current chip's columns, Show all, and the tint
function lbColumnsPill(cols, shown, nHidden) {
  const L = lbS();
  const opt = cols.filter(c => c.optional);
  const btn = el('button', { class: 'pill' + (nHidden ? ' on' : ''), id: 'pill-columns',
    'data-columns-menu': '1', 'data-hidden-tasks': nHidden ? String(nHidden) : null,
    text: `Columns${nHidden ? ` · ${nHidden} hidden` : ''} ▾` });
  return popover(btn, () => {
    const tagOf = c => c.judged ? 'judged' : c.cat ? 'cats' : 'tasks';
    const heads = { tasks: 'Task columns', judged: 'Judged topics (rubric 0–4)',
                    cats: 'MMLU by topic' };
    const tags = ['tasks', 'judged', 'cats'].filter(t => opt.some(c => tagOf(c) === t));
    return el('div', { class: 'moremenu colmenu-list', id: 'pop-columns', 'aria-label': 'columns' },
      opt.length ? tags.map(tag => {
        const cs = opt.filter(c => tagOf(c) === tag), keys = cs.map(c => c.key);
        // what is shown NOW, read when a control is used: the panel outlives
        // the render that built it (11e)
        const now = () => lbShownFor(cols);
        return el('div', { class: 'colgroup', 'data-column-group': tag },
          el('div', { class: 'small se' }, heads[tag] + ' ',
            el('button', { class: 'quiet', text: 'all', 'data-column-group-all': tag,
              onclick: () => lbSaveShown([...new Set([...now(), ...keys])]) }),
            el('button', { class: 'quiet', text: 'none', 'data-column-group-none': tag,
              onclick: () => lbSaveShown([...now()].filter(k => !keys.includes(k))) })),
          cs.map(c => el('label', { class: 'small' },
            el('input', { type: 'checkbox', 'data-column': c.key, checked: shown.has(c.key) ? '' : null,
              onchange: e => lbSaveShown(e.target.checked ? [...now(), c.key]
                                                          : [...now()].filter(k => k !== c.key)) }),
            ' ' + c.label)));
      }) : el('p', { class: 'small', text: 'Every column of this group is shown.' }),
      el('div', { class: 'frm' },
        opt.length ? el('button', { class: 'quiet', 'data-show-all': '1', text: 'Show all',
          onclick: () => lbSaveShown(opt.map(c => c.key)) }) : '',
        L.chip === 'all' ? el('button', { class: 'quiet', text: 'the default six', onclick: () => {
          state.lbShown = null;
          try { localStorage.removeItem('bench-lb-shown'); } catch (e) { /* private */ }
          render(); } }) : ''),
      el('label', { class: 'small tintsw' },
        el('input', { type: 'checkbox', 'data-tint': '1', checked: L.tint ? '' : null,
          onchange: e => {
            try { localStorage.setItem('bench-lb-tint', e.target.checked ? 'on' : 'off'); }
            catch (x) { /* private */ }
            lbSet({ tint: e.target.checked }); } }),
        ' Tint the leaders'),
      // 11f: the ± is a hover away by default; this puts it on every cell
      el('label', { class: 'small tintsw' },
        el('input', { type: 'checkbox', 'data-show-se': '1', checked: state.lbSe ? '' : null,
          onchange: e => {
            try { localStorage.setItem('bench-lb-se', e.target.checked ? 'on' : 'off'); }
            catch (x) { /* private */ }
            state.lbSe = e.target.checked; render(); } }),
        ' Show ± errors'));
  }, { key: 'columns', menu: false, rebuild: true });
}

// Models ▾ — a search, a checklist with each family's colour, and Apply
function lbModelsPill(ms) {
  const L = lbS();
  const n = L.models ? L.models.length : ms.length;
  const btn = el('button', { class: 'pill' + (L.models ? ' on' : ''), id: 'pill-models',
    'data-models-menu': '1', text: `Models${L.models ? ` · ${n} of ${ms.length}` : ''} ▾` });
  return popover(btn, () => {
    const pick = new Set(L.models || ms.map(m => m.id));
    const list = el('div', { class: 'mlist' });
    const foot = el('p', { class: 'small se', 'data-models-foot': '1' });
    const say = () => { foot.textContent = pick.size === ms.length ? 'All models shown'
      : `${pick.size} of ${ms.length} shown`; };
    const fill = q => list.replaceChildren(...ms.filter(m => !q
        || (m.name + ' ' + m.id + ' ' + famOf(m)).toLowerCase().includes(q.toLowerCase()))
      .map(m => el('label', { class: 'small mrow' },
        el('input', { type: 'checkbox', 'data-model-pick': m.id, checked: pick.has(m.id) ? '' : null,
          onchange: e => { if (e.target.checked) pick.add(m.id); else pick.delete(m.id); say(); } }),
        el('span', { class: 'famdot', style: `background:${famColor(m)}`, title: famOf(m) }),
        ' ' + m.name, el('span', { class: 'se', text: ' ' + famOf(m) }))));
    fill('');
    say();
    return el('div', { class: 'moremenu modelsmenu', id: 'pop-models', 'aria-label': 'models' },
      el('input', { type: 'search', placeholder: 'search models…', 'aria-label': 'search models',
        'data-keep': 'lbmodels', oninput: e => fill(e.target.value) }),
      el('div', { class: 'frm' },
        el('button', { class: 'quiet', text: 'Select all', onclick: () => {
          ms.forEach(m => pick.add(m.id)); fill(''); say(); } }),
        el('button', { class: 'quiet', text: 'Clear', onclick: () => {
          pick.clear(); fill(''); say(); } })),
      list, foot,
      el('button', { class: 'primary', 'data-models-apply': '1', text: 'Apply', onclick: () => {
        popClose(true);
        lbSet({ models: pick.size === ms.length ? null : [...pick] }); } }));
  }, { key: 'models', menu: false });
}

// ---- the paragraph that used to sit above the table -------------------------
function lbHowTo(ms) {
  const L = lbS();
  const nOff = ms.filter(m => m.official).length;
  return el('details', { class: 'howto', 'data-how-to-read-lb': '1', open: L.howto ? '' : null,
      ontoggle: e => {
        L.howto = e.target.open;
        try { localStorage.setItem('bench-lb-howto', L.howto ? 'open' : 'closed'); }
        catch (x) { /* private */ } } },
    el('summary', { text: 'How to read this table ▾' }),
    el('p', { class: 'sub', text: 'Click a column to sort, and a row to open it in place. Each '
      + 'cell is a score and its standard error, on one line; the unit is in the header. '
      + 'Perplexity columns are lower-is-better, excluded from Avg, and carry no standard error. '
      + '● marks a column\'s best value and ≈ marks values the z-test cannot tell from it. '
      + 'The tint is each cell\'s rank in its column, over the whole board, in five steps — '
      + 'filtering never changes a colour. '
      + `Avg exists only for models that completed all ${DATA.required.length} required tasks `
      + `(${DATA.required.join(', ')}): ${nOff} of ${ms.length} here. Anything short of that is `
      + 'preliminary — its per-task scores are valid and shown, it just has no overall number.' }),
    aboutBenchmarks([...DATA.accTasks, ...DATA.pplTasks], true));
}

// ---- a row opened in place -------------------------------------------------
function lbDetail(m, dups) {
  const block = (eyebrow, ...kids) => el('div', { class: 'dblock' },
    el('div', { class: 'eyebrow', text: eyebrow }), ...kids);
  const out = [];
  // TASKS (11f): a compact list, one line each — the name, the score, a thin
  // bar on the column's scale, the rank; the n-shot and the ± muted
  out.push(block('Tasks', el('div', { class: 'tlist', role: 'list' },
    [...DATA.accTasks, ...DATA.pplTasks].map(t => {
      const c = cell(t, m.id);
      const name = el('span', { class: 'tl-n', title: t }, LB_SHORT[t] || t,
        c && c.shots != null ? el('span', { class: 'se', text: ` ${c.shots}-shot` }) : '');
      if (!c) return el('div', { class: 'tl', role: 'listitem', 'data-task-line': t }, name,
        el('span', { class: 'tl-v se', text: '—' }), el('span'), el('span'));
      const lower = DATA.pplTasks.includes(t);
      const pool = DATA.models.map(x => ({ id: x.id, c: cell(t, x.id) })).filter(x => x.c)
        .sort((a, b) => lower ? a.c.v - b.c.v : b.c.v - a.c.v);
      const r = pool.findIndex(x => x.id === m.id) + 1;
      const best = pool[0];
      const row = best && best.id !== m.id && !lower ? (DATA.sig[t] || []).find(([a, b]) =>
        (a === best.id && b === m.id) || (a === m.id && b === best.id)) : null;
      // the column's scale: 0–100% for accuracy; best to worst on the board
      // for a lower-is-better task
      const lo = lower ? pool[pool.length - 1].c.v : 0, hi = lower ? pool[0].c.v : 1;
      const fill = lower ? (lo === hi ? 1 : (lo - c.v) / (lo - hi)) : c.v;
      return el('div', { class: 'tl', role: 'listitem', 'data-task-line': t }, name,
        el('span', { class: 'tl-v' }, el('b', { text: lower ? num(c.v, 3) : (100 * c.v).toFixed(1) }),
          c.se && !lower ? el('span', { class: 'se', text: ` ±${(100 * c.se).toFixed(1)}` }) : ''),
        el('span', { class: 'tl-b' }, el('span', { class: 'tl-f',
          style: `width:${(100 * Math.max(0, Math.min(1, fill))).toFixed(1)}%` })),
        el('span', { class: 'tl-r se', text: `#${r}/${pool.length}`
          + (best && best.id === m.id ? ' · best' : row && !row[4] ? ' · tied with best' : '') }));
    }))));
  // MMLU BY AREA: eight mini bars, on the Scale pill's scale (11h)
  const areas = Object.keys(DATA.meta.areas || {}).filter(a => areaMmlu(m, a));
  if (mmluCats(m) && diagStale(m)) {
    out.push(block('MMLU by area', el('p', { class: 'small', 'data-stale-diag': m.id,
      text: staleSentence([m]) })));
  } else if (mmluCats(m) && areas.length) {
    out.push(block('MMLU by area', el('p', { class: 'small se', 'data-bars-scale': state.avgMode,
        text: `% ${scaleWords()} — the Scale pill switches it` }),
      el('div', { class: 'minibars' }, areas.map(a => {
      const r = areaMmlu(m, a);
      const v = r ? areaScaled(r.v) : null;
      return el('div', { class: 'minibar', tabindex: '0', 'data-area-bar': a,
          'data-tip': JSON.stringify([a, r ? `${(100 * v).toFixed(1)}% ${scaleWords()} · ${r.n} questions`
            : 'no questions', 'topics: ' + ((DATA.meta.areas || {})[a] || []).join(', ')]) },
        el('span', { class: 'mb-l', text: a }),
        el('span', { class: 'mb-t' }, el('span', { class: 'mb-f',
          style: `width:${v == null ? 0 : (100 * v).toFixed(1)}%` })),
        el('span', { class: 'mb-v mono', text: v == null ? '—' : (100 * v).toFixed(1) }));
    }))));
  }
  // JUDGED TOPICS: report half only, by area, weakest first
  const jt = Object.entries((m.judge || {}).tasks || {}).filter(([t]) => t.startsWith('exam_'));
  if (jt.length) {
    const ok = judgedOkM(m);
    const byArea = Object.entries(DATA.meta.areas || {}).map(([a, ts]) => [a,
      jt.filter(([t]) => ts.includes(frName(t)))
        .map(([t, v]) => ({ t, v: pubScore(v) })).filter(x => x.v != null)
        .sort((x, y) => x.v - y.v)]).filter(([, xs]) => xs.length);
    out.push(block('Judged topics',
      ok ? '' : el('p', { class: 'small', 'data-provisional-line': '1',
        text: 'provisional — ' + ((m.judgeState || {}).reasons || ['judge not calibrated'])[0]
          + ' · not ranked' }),
      byArea.map(([a, xs]) => el('div', { class: 'jgroup' },
        el('span', { class: 'se', text: a + (ok ? '' : '') }),
        el('div', { class: 'tchips' }, xs.map(x => el('a', {
          class: 'tchip' + (ok ? '' : ' grey'), href: '#topic=' + x.t.replace(/^exam_/, ''),
          text: `${frName(x.t)} ${num(x.v, 2)}` })))))));
  }
  // LINKS
  const canRun = LIVE && (m.source === 'artifact' || !m.id.startsWith('local/'));
  out.push(block('Links', dups && dups.length ? el('p', { class: 'small', 'data-dup-line': m.id },
      `Same run as ${dups.map(d => d.name).join(', ')} `, dupToggle(m, dups)) : '',
    el('p', { class: 'small', 'data-family': famOf(m) },
      el('span', { class: 'famdot', style: `background:${famColor(m)}` }),
      ` family: ${famOf(m)} · ${m.kind}` + (m.source === 'artifact' ? ' · uploaded checkpoint' : '')),
    el('div', { class: 'dlinks' },
    el('a', { href: '#model=' + encodeURIComponent(m.id), text: 'Open model page →' }),
    el('a', { href: '#tab=provenance', text: 'Provenance →' }),
    el('button', { class: 'quiet', 'data-add-radar': m.id,
      text: state.cmpSel.includes(m.id) ? 'On the radar' : 'Add to radar',
      disabled: state.cmpSel.includes(m.id) || state.cmpSel.length >= CMP_MAX ? '' : null,
      title: !state.cmpSel.includes(m.id) && state.cmpSel.length >= CMP_MAX
        ? `the radar holds ${CMP_MAX} — remove one first` : null,
      onclick: () => cmpToggle(m.id, DATA.models) }),
    canRun ? el('button', { class: 'quiet', 'data-run-exam': m.id, text: 'Run exam',
      onclick: () => openSit(m.id) }) : '')));
  return el('div', { class: 'dgrid' }, out);
}

// ===========================================================================
// Insights: three charts under the table, each with its table beside it for
// anyone who does not read charts, every point reachable by keyboard.
// ===========================================================================

function insightsCard(ms) {
  return el('div', { class: 'card', 'data-insights': '1' },
    el('h2', { text: 'Insights' }),
    el('p', { class: 'sub', text: 'What the table says at a glance: which models are the best '
      + 'for their size, where one model is weakest, and how a handful of models compare shape '
      + 'for shape. Every judged number here is a score on the hidden questions.' }),
    el('div', { class: 'igrid', style: 'margin-top:var(--sp-4)' },
      frontierChart(ms), weakestChart(), radarBlock(ms)));
}

const logx = (v, lo, hi, a, b) => a + (Math.log10(v) - Math.log10(lo))
  / Math.max(1e-9, Math.log10(hi) - Math.log10(lo)) * (b - a);

// A model is on the frontier when no model of the SAME SIZE OR SMALLER beats
// it by a gap the z-test calls real — the same test as the rest of the board,
// so a lead inside the noise never draws the line (11e: two 750M models ten
// points apart are not both on it). Without standard errors a lead cannot be
// called real, and does not count.
function frontierOf(pts) {
  const real = (q, p) => q.se != null && p.se != null
    && (q.y - p.y) / Math.sqrt(q.se * q.se + p.se * p.se || 1e-12) > 1.96;
  return pts.filter(p => !pts.some(q => q !== p && q.x <= p.x && q.y > p.y && real(q, p)));
}

function frontierChart(ms) {
  const L = lbS();
  const ranked = DATA.models.filter(m => officialAvg(m) != null && m.params && !m.duplicateOf);
  const prelim = DATA.models.filter(m => officialAvg(m) == null && !m.duplicateOf).length;
  const pts = ranked.map(m => ({ m, x: m.params, y: officialAvg(m), se: officialSe(m) }));
  const head = el('div', { class: 'ihead' },
    el('div', { class: 'eyebrow', text: 'Score against size' }),
    el('p', { class: 'small', text: `Ranked models only, parameters on a log scale. The dashed `
      + `line is the frontier: models no model of the same size or smaller beats by more than the noise.`
      + (prelim ? ` ${prelim} preliminary not shown.` : '') }));
  if (pts.length < 2) return el('div', { class: 'ibox', 'data-frontier': '1' }, head,
    el('p', { class: 'small', text: 'Two ranked models are needed to draw this.' }));
  const front = new Set(frontierOf(pts).map(p => p.m.id));
  const W = 520, H = 290, x0 = 46, x1 = W - 16, y0 = H - 34, y1 = 14;
  const lo = Math.min(...pts.map(p => p.x)) / 1.4, hi = Math.max(...pts.map(p => p.x)) * 1.4;
  const ymax = Math.min(1, Math.max(...pts.map(p => p.y + (p.se || 0))) * 1.1 || 1);
  const X = v => logx(v, lo, hi, x0, x1), Y = v => y0 - (v / ymax) * (y0 - y1);
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'frontier',
    style: `min-width:${W}px`,
    role: 'img', 'aria-label': 'average score against parameter count',
    onclick: e => { if (!e.target.closest('.fpt') && L.focus) lbSet({ focus: null }); } });
  for (let k = 0; k <= 4; k++) {
    const v = ymax * k / 4;
    svg.append(el('svg:line', { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: 'var(--grid)' }),
      el('svg:text', { x: x0 - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 12,
        fill: 'var(--muted)', text: Math.round(100 * v) + '%' }));
  }
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++)
    for (const f of [1, 3]) {
      const v = f * Math.pow(10, e);
      if (v < lo || v > hi) continue;
      svg.append(el('svg:text', { x: X(v), y: H - 12, 'text-anchor': 'middle', 'font-size': 12,
        fill: 'var(--muted)', text: P(v) }));
    }
  const sel = L.focus && pts.find(p => p.m.id === L.focus);
  let caption = '';
  if (sel) {
    const worse = pts.filter(p => p.x > sel.x && p.y < sel.y);
    svg.append(el('svg:rect', { x: X(sel.x), y: Y(sel.y), width: Math.max(0, x1 - X(sel.x)),
      height: Math.max(0, y0 - Y(sel.y)), fill: 'var(--accent-soft)', 'data-shade': '1' }));
    caption = !worse.length ? `No model here is bigger and scores lower than ${sel.m.name}.`
      : `${worse.length} model${worse.length === 1 ? '' : 's'} here `
        + `${worse.length === 1 ? 'is' : 'are'} bigger and score${worse.length === 1 ? 's' : ''} `
        + `lower than ${sel.m.name}.`;
  }
  // the line runs through the frontier by size, and at any one size through
  // the best point only: a tie inside the noise is marked, not jumped between
  const bySize = new Map();
  for (const p of pts.filter(p => front.has(p.m.id)))
    if (!bySize.has(p.x) || bySize.get(p.x).y < p.y) bySize.set(p.x, p);
  const fp = [...bySize.values()].sort((a, b) => a.x - b.x);
  if (fp.length > 1) {
    svg.append(el('svg:path', { d: 'M' + fp.map(p => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`)
      .join('L'), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.5,
      'stroke-dasharray': '5 4', 'data-frontier-line': '1' }));
    const last = fp[fp.length - 1];
    svg.append(el('svg:text', { x: Math.min(X(last.x) + 6, x1 - 52), y: Y(last.y) - 8,
      'font-size': 12, fill: 'var(--accent)', text: 'frontier' }));
  }
  for (const p of pts) {
    const on = front.has(p.m.id), hot = sel && sel.m.id === p.m.id;
    svg.append(el('svg:circle', { class: 'fpt' + (hot ? ' hot' : ''), cx: X(p.x), cy: Y(p.y),
      r: hot ? 7 : 5, fill: famColor(p.m), stroke: on ? 'var(--text-primary)' : 'var(--surface-1)',
      'stroke-width': on ? 1.8 : 1.2, tabindex: 0, 'data-point': p.m.id,
      'data-on-frontier': on ? '1' : null,
      'data-tip': JSON.stringify([p.m.name, `${P(p.x)} parameters`,
        `Avg ${(100 * p.y).toFixed(1)}%` + (p.se != null ? ` ± ${(100 * p.se).toFixed(1)}` : ''),
        on ? 'on the frontier' : 'a model its size or smaller beats it by more than the noise']),
      onclick: () => lbSet({ focus: hot ? null : p.m.id }),
      onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault(); lbSet({ focus: hot ? null : p.m.id }); } } }));
  }
  const table = el('details', { class: 'astable' }, el('summary', { text: 'Show as table' }),
    el('table', { class: 'mini' }, el('thead', {}, el('tr', {},
        el('th', { text: 'model' }), el('th', { class: 'num', text: 'params' }),
        el('th', { class: 'num', text: 'avg' }), el('th', { text: 'frontier' }))),
      el('tbody', {}, [...pts].sort((a, b) => a.x - b.x).map(p => el('tr', {},
        el('td', { text: p.m.name }), el('td', { class: 'num', text: P(p.x) }),
        el('td', { class: 'num', text: (100 * p.y).toFixed(1)
          + (p.se != null ? ` ±${(100 * p.se).toFixed(1)}` : '') }),
        el('td', { text: front.has(p.m.id) ? 'on it' : '—' }))))));
  return el('div', { class: 'ibox', 'data-frontier': '1' }, head,
    hfade('frontier', el('div', { class: 'chartscroll', 'data-hkeep': 'frontier' }, svg)),
    el('p', { class: 'small', 'data-frontier-caption': '1', text: caption
      || 'Click a point to see which models are bigger and score lower.' }), table);
}

// One judged model's own topics, weakest first, on the 0–4 scale. It orders
// one model's topics; it does not rank models.
function weakestChart() {
  const L = lbS();
  const judged = DATA.models.filter(m => Object.keys((m.judge || {}).tasks || {})
    .some(t => t.startsWith('exam_')));
  const head = el('div', { class: 'ihead' }, el('div', { class: 'eyebrow', text: 'Weakest topics' }));
  if (!judged.length) return el('div', { class: 'ibox', 'data-weakest': '1' }, head,
    el('p', { class: 'small', 'data-weakest-empty': '1',
      text: 'No model has been judged yet — Loop ▸ Sit the exam' }));
  // the model the loop is on: the one judged last, unless someone picked another
  const lastJ = m => Math.max(0, ...Object.values((m.judge || {}).tasks || {}).map(t => t.judged_at || 0));
  const m = judged.find(x => x.id === L.weak) || [...judged].sort((a, b) => lastJ(b) - lastJ(a))[0];
  const ok = judgedOkM(m);
  const all = Object.entries(m.judge.tasks).filter(([t]) => t.startsWith('exam_'))
    .map(([t, v]) => ({ t, v: pubScore(v) })).filter(x => x.v != null).sort((a, b) => a.v - b.v);
  // the twelve weakest, and the rest one click away: 37 bars is a wall
  const xs = L.weakAll ? all : all.slice(0, 12);
  const btn = el('button', { class: 'pill', id: 'pill-weak', 'data-weak-model': m.id,
    text: `${m.name} · Change model… ▾` });
  const change = popover(btn, () => {
    const list = el('div', { class: 'mlist' });
    const fill = q => list.replaceChildren(...judged.filter(x => !q
        || x.name.toLowerCase().includes(q.toLowerCase()))
      .map(x => el('button', { role: 'menuitem', text: x.name, 'data-weak-pick': x.id,
        // 11k: the one it is showing is ticked, not bordered
        'aria-current': x.id === m.id ? 'true' : null,
        onclick: () => { popClose(true); lbSet({ weak: x.id }); } })));
    fill('');
    return el('div', { class: 'moremenu', id: 'pop-weak', 'aria-label': 'judged models' },
      el('input', { type: 'search', placeholder: 'judged models…', 'aria-label': 'search judged models',
        oninput: e => fill(e.target.value) }), list);
  }, { key: 'weak', menu: false });
  const rowH = 24, W = 520, H = xs.length * rowH + 8, x0 = 230, x1 = W - 44;
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'weakest',
    style: `min-width:${W}px`,
    role: 'img', 'aria-label': `${m.name}'s judged topics, weakest first` });
  if (!ok) svg.append(el('svg:defs', {}, el('svg:pattern', { id: 'hatch', width: 6, height: 6,
    patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' },
    el('svg:rect', { width: 6, height: 6, fill: 'var(--grid)' }),
    el('svg:line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: 'var(--axis)', 'stroke-width': 2 }))));
  xs.forEach((x, i) => {
    const y = 4 + i * rowH, w = (x.v / 4) * (x1 - x0);
    const a = el('svg:a', { href: '#topic=' + x.t.replace(/^exam_/, ''), class: 'wbar',
      'data-weak-topic': x.t,
      'data-tip': JSON.stringify([frName(x.t), `${num(x.v, 2)} / 4 · hidden questions`
        + (ok ? '' : ' · provisional'), 'Open the topic →']) },
      el('svg:text', { x: x0 - 8, y: y + 16, 'text-anchor': 'end', 'font-size': 12,
        fill: 'var(--text-secondary)', text: frName(x.t) }),
      el('svg:rect', { x: x0, y: y + 4, width: Math.max(1, w), height: rowH - 10, rx: 2,
        fill: ok ? 'var(--accent)' : 'url(#hatch)', 'fill-opacity': ok ? 0.75 : 1 }),
      el('svg:text', { x: x0 + w + 6, y: y + 16, 'font-size': 12, fill: 'var(--text-primary)',
        class: 'mono', text: num(x.v, 2) }));
    svg.append(a);
  });
  return el('div', { class: 'ibox', 'data-weakest': '1' },
    head, el('div', { class: 'frm' }, change,
      ok ? '' : el('span', { class: 'badge warn', 'data-provisional': 'weakest', text: 'provisional',
        title: ((m.judgeState || {}).reasons || []).join('; ') })),
    el('div', { class: 'weakest-svg' },
      hfade('weakest', el('div', { class: 'chartscroll', 'data-hkeep': 'weakest' }, svg))),
    el('div', { class: 'wrows', 'data-weak-rows': m.id }, xs.map(x => el('a', { class: 'wrow',
        href: '#topic=' + x.t.replace(/^exam_/, ''), 'data-weak-row': x.t,
        title: `${frName(x.t)} · ${num(x.v, 2)} / 4 · hidden questions` + (ok ? '' : ' · demo only') },
      el('span', { class: 'wname', text: frName(x.t) }),
      el('span', { class: 'wtrack' }, el('span', { class: 'wfill' + (ok ? '' : ' hatch'),
        style: `width:${Math.max(1, 100 * x.v / 4).toFixed(1)}%` })),
      el('span', { class: 'wv', text: num(x.v, 2) })))),
    all.length > 12 ? el('button', { class: 'quiet', 'data-weak-all': '1',
      text: L.weakAll ? 'Show the twelve weakest' : `Show all ${all.length} topics`,
      onclick: () => lbSet({ weakAll: !L.weakAll }) }) : '',
    el('p', { class: 'small', text: `${m.name}'s own topics, weakest first, on the hidden `
      + 'questions (0–4). This orders one model\'s topics; it does not rank models.'
      + (ok ? '' : ' Demo only: the judge is not checked by a person, so nothing here is ranked.') }),
    el('details', { class: 'astable' }, el('summary', { text: 'Show as table' }),
      el('table', { class: 'mini' }, el('tbody', {}, all.map(x => el('tr', {},
        el('td', {}, el('a', { href: '#topic=' + x.t.replace(/^exam_/, ''), text: frName(x.t) })),
        el('td', { class: 'num', text: num(x.v, 2) + ' / 4' })))))));
}

// The radar: up to five models as chips, and three sources. Judged by area is
// an average, so it waits for a calibrated judge.
function radarSource() {
  const L = lbS();
  const areas = Object.keys(DATA.meta.areas || {});
  if (L.radarSrc === 'areas') return areas.map(a => ({ key: a, label: a,
    get: m => { const r = areaMmlu(m, a); return r ? { n: areaScaled(r.v),
      rows: [`${(100 * areaScaled(r.v)).toFixed(1)}% ${scaleWords()} · ${r.n} questions`] } : null; } }));
  if (L.radarSrc === 'judged' && judgedCalibrated()) return areas.map(a => ({ key: a, label: a,
    get: m => { const r = areaJudged(m, a); return r.v == null ? null
      : { n: r.v / 4, rows: [`${num(r.v, 2)} / 4 · ${r.k} of ${r.n} topics`] }; } }));
  return (DATA.required || DATA.accTasks).filter(t => DATA.accTasks.includes(t)).map(t => ({
    key: t, label: t, get: m => { const c = cell(t, m.id);
      return c ? { n: normScore(t, c.v), rows: [`${(100 * c.v).toFixed(1)}% raw`
        + (c.se ? ` ± ${(100 * c.se).toFixed(1)}` : '')] } : null; } }));
}

function radarBlock(ms) {
  const L = lbS();
  const ids = state.cmpSel.filter(id => DATA.models.some(m => m.id === id));
  const axes = radarSource();
  const calOk = judgedCalibrated();
  const add = el('button', { class: 'pill', id: 'pill-radar-add', 'data-radar-add': '1',
    disabled: ids.length >= CMP_MAX ? '' : null,
    title: ids.length >= CMP_MAX ? `the radar holds ${CMP_MAX} — remove one first` : null,
    text: 'Add a model… ▾' });
  const addPop = popover(add, () => {
    const list = el('div', { class: 'mlist' });
    const fill = q => list.replaceChildren(...DATA.models.filter(m => !ids.includes(m.id)
        && (!q || m.name.toLowerCase().includes(q.toLowerCase())))
      .map(m => el('button', { role: 'menuitem', 'data-radar-pick': m.id, text: m.name,
        onclick: () => { popClose(true); cmpToggle(m.id, DATA.models); } })));
    fill('');
    return el('div', { class: 'moremenu', id: 'pop-radar-add', 'aria-label': 'add a model' },
      el('input', { type: 'search', placeholder: 'models…', 'aria-label': 'search models',
        oninput: e => fill(e.target.value) }), list);
  }, { key: 'radar-add', menu: false });
  const chips = el('div', { class: 'rchips', 'data-radar-chips': '1' },
    ids.map(id => {
      const m = DATA.models.find(x => x.id === id);
      const color = trColor(state.cmpColors[id] ?? ids.indexOf(id));
      return el('span', { class: 'mchip', 'data-radar-chip': id },
        el('span', { class: 'key', style: `background:${color}` }), m.name,
        el('button', { class: 'xbtn', 'aria-label': 'remove ' + m.name, text: '×',
          onclick: () => cmpToggle(id, DATA.models) }));
    }), addPop);
  const src = el('div', { class: 'seg', role: 'group', 'aria-label': 'radar source' },
    [['tasks', 'Tasks'], ['areas', 'MMLU by area'], ['judged', 'Judged by area']].map(([v, t]) => {
      const off = v === 'judged' && !calOk;
      return el('button', { 'aria-pressed': String(L.radarSrc === v), 'data-radar-src': v, text: t,
        disabled: off ? '' : null, title: off ? judgedOffWhy() : null,
        onclick: () => { if (!off) lbSet({ radarSrc: v }); } });
    }));
  const head = el('div', { class: 'ihead' }, el('div', { class: 'eyebrow', text: 'Compare shapes' }));
  const body = !ids.length || axes.length < 3
    ? el('p', { class: 'small', 'data-radar-prompt': '1', text: axes.length < 3
        ? 'Too few axes to draw a shape for this source.'
        : `Add up to ${CMP_MAX} models to draw their shapes — one axis per `
          + (L.radarSrc === 'tasks' ? 'task' : 'area') + ', one shape per model.' })
    : radarSvg(axes, ids.map(id => ({ m: DATA.models.find(x => x.id === id),
        color: trColor(state.cmpColors[id] ?? ids.indexOf(id)) })));
  return el('div', { class: 'ibox', 'data-radar': '1' }, head, chips, src, body,
    calOk ? '' : el('p', { class: 'small se', text: 'Judged by area: ' + judgedOffWhy() + '.' }));
}

function radarSvg(axes, series) {
  const W = 520, H = 340, cx = 260, cy = 172, R = 104, N = axes.length;
  const ang = i => -Math.PI / 2 + 2 * Math.PI * i / N;
  const pt = (i, r) => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'radar',
    style: `min-width:${W}px`, role: 'img', 'aria-label': 'model shapes' });
  for (const f of [0.25, 0.5, 0.75, 1]) {
    svg.append(el('svg:polygon', {
      points: axes.map((_, i) => pt(i, f * R).map(v => v.toFixed(1)).join(',')).join(' '),
      fill: 'none', stroke: f === 1 ? 'var(--axis)' : 'var(--grid)', 'stroke-width': 1 }));
  }
  axes.forEach((ax, i) => {
    const [x, y] = pt(i, R);
    svg.append(el('svg:line', { x1: cx, y1: cy, x2: x, y2: y, stroke: 'var(--grid)' }));
    const [lx, ly] = pt(i, R + 14), c = Math.cos(ang(i));
    svg.append(el('svg:text', { x: lx, y: ly + 4, 'font-size': 12, fill: 'var(--text-secondary)',
      'text-anchor': c > 0.15 ? 'start' : c < -0.15 ? 'end' : 'middle', text: ax.label }));
  });
  const hits = [];
  for (const s of series) {
    const vals = axes.map(ax => ax.get(s.m));
    const pts = vals.map((v, i) => v ? pt(i, v.n * R) : null);
    const have = pts.filter(Boolean);
    if (have.length >= 2) svg.append(el('svg:path', {
      d: 'M' + have.map(p => p.map(v => v.toFixed(1)).join(',')).join('L') + 'Z',
      fill: s.color, 'fill-opacity': 0.12, stroke: s.color, 'stroke-width': 2,
      'pointer-events': 'none' }));
    vals.forEach((v, i) => {
      if (!v) return;
      const [x, y] = pts[i];
      svg.append(el('svg:circle', { cx: x, cy: y, r: 3.5, fill: s.color,
        stroke: 'var(--surface-1)', 'stroke-width': 1.5, 'pointer-events': 'none' }));
      hits.push(el('svg:circle', { class: 'hit', cx: x, cy: y, r: 8, tabindex: 0,
        'data-tip': JSON.stringify([s.m.name, `${axes[i].label}: ${(100 * v.n).toFixed(0)}`,
          ...v.rows]), 'data-tipkey': s.color }));
    });
  }
  svg.append(...hits);
  const table = el('details', { class: 'astable' }, el('summary', { text: 'Show as table' }),
    el('table', { class: 'mini' },
      el('thead', {}, el('tr', {}, el('th', { text: 'axis' }),
        series.map(s => el('th', { class: 'num', title: s.m.id, text: midTrunc(s.m.name, 16) })))),
      el('tbody', {}, axes.map(ax => el('tr', {}, el('td', { text: ax.label }),
        series.map(s => { const v = ax.get(s.m);
          return el('td', { class: 'num', text: v ? (100 * v.n).toFixed(0) : '—' }); }))))));
  return el('div', {}, hfade('radar', el('div', { class: 'chartscroll', 'data-hkeep': 'radar' }, svg)),
    table);
}

document.addEventListener('keydown', e => {
  // Esc clears the frontier's highlight — after any open popover has had it
  if (e.key === 'Escape' && !e.defaultPrevented && state.lb && state.lb.focus) lbSet({ focus: null });
});


// The optional "MMLU by category" view: one column per category from
// scripts/categories.yaml, one row per model that has a diagnosis, so a trained
// checkpoint's profile can be set against the reference models. Every number
// is a LEADERBOARD-half score read from diagnose.json — the same half the MMLU
// column comes from — so nothing here is a diagnosis-half claim.
const mmluCats = m => ((((m.diag || {}).tasks || {}).mmlu || {}).categories) || null;

// MMLU by category as optional Leaderboard columns — one per topic MMLU has
// subjects for, from the models' diagnoses
function lbCategoryCols(ms) {
  const have = ms.filter(mmluCats);
  return (DATA.meta.categories || []).filter(c => have.some(m => mmluCats(m)[c]))
    .map(c => ({ key: 'cat:' + c, label: 'MMLU ' + c, num: true, cat: c, group: 'cats' }));
}

// which task columns show: remembered per browser; by default the required
// tasks, then the rest, six at most — harness tasks only: the judged topics
// and the MMLU categories are thirty-six and two dozen, and start hidden
function lbShownTasks(taskCols) {
  const keys = taskCols.map(c => c.key);
  let want = null;
  try { want = JSON.parse(localStorage.getItem('bench-lb-shown') || 'null'); } catch (e) { /* none */ }
  if (state.lbShown) want = state.lbShown;
  if (!Array.isArray(want)) {
    const plain = taskCols.filter(c => c.task).map(c => c.key);
    const req = (DATA.required || []).filter(t => plain.includes(t));
    want = [...req, ...plain.filter(k => !req.includes(k))].slice(0, 6);
  }
  return new Set(want.filter(k => keys.includes(k)));
}

// tasks under their domain, payload order preserved, undomained ones last
function domainGroups(tasks) {
  const by = new Map();
  for (const t of tasks) {
    const d = (DATA.tasks[t] || {}).domain || 'other';
    if (!by.has(d)) by.set(d, []);
    by.get(d).push(t);
  }
  return [...by.entries()].sort((a, b) =>
    (a[0] === 'other') - (b[0] === 'other'));
}

function vTasks(ms) {
  if (!DATA.accTasks.length) return [note('No accuracy tasks found.')];
  const scaleBtn = (v, label, tip) => el('button', {
    class: 'tgl' + (state.accScale === v ? ' on' : ''), title: tip, text: label,
    'aria-pressed': String(state.accScale === v),
    onclick: () => { state.accScale = v; render(); } });
  return [
    el('p', { class: 'sub', style: 'margin:10px 2px', text:
      'One panel per benchmark, models ranked. Bars share one hue on purpose — the label is the identity; '
      + 'pointing at any model highlights it in every panel. Dashed line = chance.'
      + (anyCk() ? ' Hollow bars are uploaded checkpoints.' : '') }),
    el('div', { class: 'ctrl', style: 'margin:0 2px 10px' },
      el('span', { class: 'small', text: 'scale' }),
      scaleBtn('raw', 'raw score',
               'accuracy as the harness reports it, 0% to the best score on the board'),
      scaleBtn('chance', 'vs chance',
               'share of the headroom above chance: 0% = guessing, 100% = perfect. '
             + 'Bars diverge from the chance line, so a model BELOW chance points '
             + 'left in red. This is the scale on which models bunched at the floor '
             + 'become distinguishable, and the one the frontier reference fits on.'),
      el('span', { class: 'small', style: 'margin-left:auto',
        text: state.accScale === 'chance'
          ? '0% = chance · 100% = perfect'
          : 'tasks with no chance level are unchanged by this toggle' })),
    // grouped under their domain — the same vocabulary the radar folds on, so a
    // reader learns one taxonomy rather than two
    ...domainGroups(DATA.accTasks).map(([dom, ts]) => el('div', {},
      el('h3', { class: 'domhead' }, dom,
        el('span', { class: 'se', text: ` · ${ts.length} task${ts.length > 1 ? 's' : ''}` })),
      el('div', { class: 'panels' }, ts.map(t => barPanel(t, ms, { lower: false }))))),
    tableTwin('tasks-table', ms, DATA.accTasks, false)];
}

function vPpl(ms) {
  if (!DATA.pplTasks.length) return [note('No perplexity tasks found. Create pinned corpus slices with scripts/make_ppl_task.py — they are the eval that separates models multiple-choice cannot.')];
  const CE_NOTE = 'cross-entropy = bits_per_byte × ln 2, in nats per byte — the same quantity '
    + 'as a training loss, just per byte instead of per token (token counts are not comparable '
    + 'across tokenizers; bytes are).';
  // the CE table: bits/byte and its loss form side by side, per task × model
  const ceData = [];
  for (const t of DATA.pplTasks) {
    const info = DATA.tasks[t] || {};
    for (const m of ms) {
      const c = cell(t, m.id);
      if (!c) continue;
      const isBpb = info.metric === 'bits_per_byte';
      ceData.push({ task: t, model: m.name, mid: m.id,
        bpb: isBpb ? c.v : null, ce: isBpb ? c.v * Math.LN2 : null,
        other: !isBpb ? c.v : null, om: info.metric, docs: c.n });
    }
  }
  const CE_COLS = [
    { key: 'task',  label: 'task' },
    { key: 'model', label: 'model' },
    { key: 'bpb',   label: 'bits / byte', num: true },
    { key: 'ce',    label: 'CE loss (nats / byte)', num: true },
    { key: 'other', label: 'other metric', num: true },
    { key: 'docs',  label: 'docs', num: true },
  ];
  const { key: ceKey, dir: ceDir } = state.ceSort;
  const ceCol = CE_COLS.find(c => c.key === ceKey) || CE_COLS[0];
  const ceRows = [...ceData].sort((A, B) => {
    const va = A[ceCol.key], vb = B[ceCol.key];
    if (va == null && vb == null) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    const base = ceCol.num ? ceDir * (va - vb) : ceDir * natCmp(va, vb);
    // sorting by task keeps each group internally best-first
    return base || (ceCol.key === 'task' ? (A.bpb ?? 1e9) - (B.bpb ?? 1e9) : 0);
  }).map(r => {
    // the CE column used to carry class 'best' on EVERY row, so the ● that means
    // "best in column" appeared beside every value. Mark the actual leader per
    // task, and only when it is clear of the tie band.
    const peers = ceData.filter(x => x.task === r.task && x.ce != null).map(x => x.ce);
    const lead = peers.length > 1 ? Math.min(...peers) : null;
    const band = lead == null ? 0 : Math.max(0.005, 0.01 * Math.abs(lead));
    const within = lead != null && r.ce != null && r.ce <= lead + band;
    const tied = lead == null ? 0 : peers.filter(v => v <= lead + band).length;
    return el('tr', {},
      el('td', { text: r.task }),
      el('td', { 'data-model': r.mid, text: r.model }),
      el('td', { class: 'num', text: r.bpb != null ? num(r.bpb, 4) : '—' }),
      el('td', { class: 'num' + (within ? (tied > 1 ? ' tiebest' : ' best') : ''),
        title: within && tied > 1 ? 'tied for lowest on this corpus — no standard error' : '',
        text: r.ce != null ? num(r.ce, 4) : '—' }),
      el('td', { class: 'num', text: r.other != null ? `${num(r.other, 4)} (${r.om})` : '—' }),
      el('td', { class: 'num', text: r.docs != null ? String(r.docs) : '—' }));
  });
  return [
    el('p', { class: 'sub', style: 'margin:10px 2px', text:
      'Rolling-loglikelihood language modelling over pinned corpus samples. LOWER is better '
      + 'everywhere here. Quote bits_per_byte across model families; ' + CE_NOTE + ' '
      + 'The harness reports NO standard error for these, so the dashboard will not '
      + 'call a small lead a win: ≈ marks values within 1% (floor 0.005) of the best, '
      + 'which is a stated placeholder until per-shard bootstrap intervals exist. Two '
      + 'models 0.001 apart are tied, not ranked.'
      + (anyCk() ? ' Hollow bars are uploaded checkpoints.' : '') }),
    el('div', { class: 'panels' }, DATA.pplTasks.map(t => barPanel(t, ms, { lower: true }))),
    el('div', { class: 'card' },
      el('h2', { text: 'Cross-entropy loss' }),
      el('p', { class: 'sub', text: CE_NOTE + ' Compare it directly against the loss curves '
        + 'in your training logs to see where a checkpoint sits.' }),
      el('table', {},
        el('thead', {}, el('tr', {}, CE_COLS.map(c => el('th', {
          class: (c.num ? 'num ' : '') + 'sortable',
          onclick: () => { state.ceSort = { key: c.key,
            dir: state.ceSort.key === c.key ? -state.ceSort.dir : 1 }; render(); },
          'aria-sort': ceKey === c.key ? (ceDir > 0 ? 'ascending' : 'descending') : 'none' },
          c.label + ' ', ceKey === c.key
            ? el('span', { class: 'dir', text: ceDir > 0 ? '▲' : '▼' }) : '')))),
        el('tbody', {}, ceRows))),
    tableTwin('ppl-table', ms, DATA.pplTasks, true)];
}

function tableTwin(id, ms, tasks, lower) {
  const rows = [];
  for (const t of tasks) for (const m of ms) {
    const c = cell(t, m.id); if (!c) continue;
    rows.push(el('tr', {},
      el('td', { text: t }), el('td', { text: DATA.tasks[t].metric }),
      el('td', { text: m.name }),
      el('td', { class: 'num', text: lower ? num(c.v, 4) : pct(c.v, 2) }),
      el('td', { class: 'num', text: c.se ? (lower ? num(c.se, 4) : pct(c.se, 2)) : '—' }),
      el('td', { class: 'num', text: c.shots != null ? String(c.shots) : '—' }),
      el('td', { class: 'num', text: c.n != null ? String(c.n) : '—' })));
  }
  const tv = el('div', { class: 'tv', id },
    el('table', {}, el('thead', {}, el('tr', {},
      ['task', 'metric', 'model', 'score', 'stderr', 'n-shot', 'items'].map((h, i) =>
        el('th', { class: i >= 3 ? 'num' : '', text: h })))), el('tbody', {}, rows)));
  const btn = el('button', { style: 'margin-top:10px', onclick: () => {
    tv.classList.toggle('open');
    btn.textContent = tv.classList.contains('open') ? 'Hide data table' : 'Show data table';
  }, text: 'Show data table' });
  return el('div', {}, btn, tv);
}

function vRuns(ms) {
  const frag = [];
  // the board checks are already above every tab, in full on the board tabs
  // and folded elsewhere. A second copy here was the same finding twice.
  // provenance was rendered in results-directory scan order — effectively random,
  // and worst exactly when it matters most (a burst of checkpoint evals). Sortable
  // now, defaulting to last run first: the eval you just finished is row one.
  const PROV_COLS = [
    { key: 'name',   label: 'model' },
    { key: 'id',     label: 'hf id' },
    { key: 'arch',   label: 'architecture', get: m => (m.archinfo || {}).arch },
    { key: 'shape',  label: 'shape', num: true, get: m =>
        m.archinfo && m.archinfo.hidden != null
          ? m.archinfo.hidden * 1e4 + (m.archinfo.layers || 0) : null },
    { key: 'vocab',  label: 'vocab', num: true, get: m => (m.archinfo || {}).vocab },
    { key: 'backend', label: 'backend' },
    { key: 'dtype',  label: 'dtype' },
    { key: 'batch',  label: 'batch', num: true },
    { key: 'chat',   label: 'template applied', num: true, get: m => m.chat ? 1 : 0 },
    { key: 'tmpl',   label: 'template id', get: m => (m.archinfo || {}).tmpl_sha },
    { key: 'code',   label: 'model code', get: m => ((m.archinfo || {}).code_sha || []).join(' ') },
    { key: 'seed',   label: 'seed', num: true },
    { key: 'limit',  label: 'limit', num: true,
      get: m => m.limit == null ? Infinity : m.limit },   // 'full' sorts as largest
    { key: 'paramsSrc', label: 'params from' },
    { key: 'stored',  label: 'weights dtype', get: m => (m.archinfo || {}).stored_dtype },
    { key: 'minutes', label: 'wall clock', num: true },
    { key: 'hash',   label: 'harness' },
    { key: 'date',   label: 'last run', defDir: -1 },
  ];
  const pv = (m, c) => c.get ? c.get(m) : m[c.key];
  const provCol = PROV_COLS.find(c => c.key === state.provSort.key) || PROV_COLS[14];
  const provAll = [...ms].sort((a, b) => {
    const va = pv(a, provCol), vb = pv(b, provCol);
    if (va === vb) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return state.provSort.dir * (provCol.num ? va - vb : natCmp(va, vb));
  });
  const provPg = paged('provenance', provAll, JSON.stringify([state.provSort, state.q,
                                                              state.kind, state.src]));
  const provRows = provPg.rows;
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'Run provenance' }),
    el('div', { class: 'kvs', 'data-builds': '1' },
      DATA.meta.hashes.length ? el('span', {}, el('b', { text: 'harness ' }),
        el('span', { class: 'mono', text: DATA.meta.hashes.join(', ') })) : '',
      DATA.meta.transformers ? el('span', {}, el('b', { text: 'transformers ' }),
        DATA.meta.transformers) : ''),
    el('p', { class: 'sub', text: 'Every field here can change a score. Publish this table with the numbers, or the numbers are hearsay. Sorted newest-eval-first — click any column to re-sort.' }),
    provPg.pager,
    el('div', { class: 'lb-wrap stick' }, el('table', { class: 'prov', 'data-prov-table': '1' },
      el('thead', {}, el('tr', {}, PROV_COLS.map(c => el('th', {
        class: (c.num ? 'num ' : '') + 'sortable',
        onclick: () => { state.provSort = { key: c.key,
          dir: state.provSort.key === c.key ? -state.provSort.dir
             : (c.defDir || (c.num ? -1 : 1)) }; render(); },
        'aria-sort': state.provSort.key === c.key
          ? (state.provSort.dir > 0 ? 'ascending' : 'descending') : 'none' },
        c.label + ' ', state.provSort.key === c.key
          ? el('span', { class: 'dir', text: state.provSort.dir > 0 ? '▲' : '▼' }) : '')))),
      el('tbody', {}, provRows.map(m => el('tr', {},
        el('td', { class: 'model', 'data-model': m.id }, m.name, ckBadge(m) || ''),
        el('td', {}, el('span', { class: 'mono', text: m.id })),
        el('td', { text: (m.archinfo && m.archinfo.arch) || '—' }),
        el('td', { class: 'num', title: m.archinfo ? `heads ${m.archinfo.heads ?? '—'} · ctx ${m.archinfo.ctx ?? '—'}` : '',
          text: m.archinfo && m.archinfo.hidden ? `${m.archinfo.hidden}×${m.archinfo.layers ?? '?'}` : '—' }),
        el('td', { class: 'num', text: (m.archinfo && m.archinfo.vocab)
          ? m.archinfo.vocab.toLocaleString() : '—' }),
        el('td', { text: m.backend || '—' }),
        el('td', { text: m.dtype || '—' }),
        el('td', { text: m.batch == null ? '—' : String(m.batch) }),
        el('td', { title: m.kindReason || '' },
          m.chat ? 'yes' : 'no',
          (m.archinfo || {}).kind_unconfirmed
            ? el('span', { class: 'badge prelim', title: 'detection only — the '
                + 'repo name does not corroborate it', text: '?' }) : ''),
        el('td', {}, el('span', { class: 'mono',
          title: ((m.archinfo || {}).tmpl_src ? 'from ' + m.archinfo.tmpl_src : 'no chat template in the repo')
            + (m.kindReason ? `\npolicy: ${m.kindReason}` : ''),
          text: (m.archinfo || {}).tmpl_sha || '—' })),
        // which Python produced this score, when the checkpoint brought its own
        el('td', {}, ((m.archinfo || {}).code_sha || []).length
          ? el('span', { class: 'mono', title: m.archinfo.code_sha.join('\n'),
              text: `custom ×${m.archinfo.code_sha.length}` })
          : el('span', { class: 'se', text: 'library' })),
        el('td', { class: 'num', text: m.seed == null ? '—' : String(m.seed) }),
        el('td', { text: m.limit == null ? 'full' : String(m.limit) }),
        el('td', { text: m.paramsSrc || '—' }),
        el('td', { title: (m.archinfo || {}).params_src
            ? 'parameter count from ' + m.archinfo.params_src : '',
          text: (m.archinfo || {}).stored_dtype || '—' }),
        el('td', { class: 'num', text: m.minutes + ' min' }),
        el('td', {}, el('span', { class: 'mono', text: m.hash || '—' })),
        el('td', {}, el('span', { class: 'mono',
          text: String(m.date || '—').slice(0, 16).replace('T', ' ') })))))))));
  // ---- query every metric — a search bar instead of a wall of rows -----------
  // Client-side on purpose: the whole corpus is a few thousand rows, so a search
  // service would be infrastructure guarding data a browser filters in under a
  // millisecond. The query language is the useful part.
  const names = new Set(ms.map(m => m.name));
  // each row joins its model's shape, so queries can mix scores with architecture:
  // "gptneox layers>=24 task:hellaswag" or "vocab>100000 metric:acc value>0.3"
  const byName = new Map(DATA.models.map(m => [m.name, m]));
  const all = DATA.extra.filter(r => names.has(r[0])).map(r => {
    const m = byName.get(r[0]) || {};
    const a = m.archinfo || {};
    return { r,
      hay: (r[0] + ' ' + r[1] + ' ' + r[2] + ' ' + (a.arch || '') + ' '
            + (m.kind || '')).toLowerCase(),
      words: null,   // built lazily for fuzzy matching
      nums: { value: r[3], stderr: r[4], shots: r[5], items: r[6],
              vocab: a.vocab, hidden: a.hidden, layers: a.layers,
              heads: a.heads, ctx: a.ctx, params: m.params } };
  });
  for (const row of all) row.words = row.hay.split(/[^a-z0-9_]+/).filter(Boolean);
  // one edit (insert/delete/substitute) of tolerance — the typo-forgiveness that
  // makes search feel like a search engine instead of a grep
  function within1(a, b) {
    if (a === b) return true;
    const la = a.length, lb = b.length;
    if (Math.abs(la - lb) > 1) return false;
    let i = 0, j = 0, edits = 0;
    while (i < la && j < lb) {
      if (a[i] === b[j]) { i++; j++; continue; }
      if (++edits > 1) return false;
      // adjacent transposition ("shaeps" -> "shapes") counts as ONE edit —
      // it is the most common human typo
      if (la === lb && a[i] === b[j + 1] && a[i + 1] === b[j]) { i += 2; j += 2; continue; }
      if (la > lb) i++; else if (lb > la) j++; else { i++; j++; }
    }
    return edits + (la - i) + (lb - j) <= 1;
  }
  const NUMF = 'value|stderr|shots|items|vocab|hidden|layers|heads|ctx|params';
  function buildPred(q) {
    const toks = [...q.trim().toLowerCase().matchAll(/"([^"]+)"|(\S+)/g)]
      .map(m => m[1] != null ? { s: m[1], quoted: true } : { s: m[2], quoted: false });
    const lits = [];          // positive literals, for highlighting
    const preds = toks.map(({ s, quoted }) => {
      let m;
      if (!quoted && (m = s.match(new RegExp('^(' + NUMF + ')(>=|<=|>|<|=)(-?\\d*\\.?\\d+)$')))) {
        const f = m[1], op = m[2], x = +m[3];
        return row => { const v = row.nums[f];
          return v != null && (op === '>' ? v > x : op === '<' ? v < x :
                 op === '>=' ? v >= x : op === '<=' ? v <= x : Math.abs(v - x) < 1e-9); };
      }
      if (!quoted && (m = s.match(/^(model|task|metric|arch):(.+)$/))) {
        const idx = { model: 0, task: 1, metric: 2 }[m[1]];
        const alts = m[2].split('|').filter(Boolean);
        alts.forEach(a => lits.push(a));
        if (m[1] === 'arch')
          return row => alts.some(a => row.hay.includes(a));
        return row => alts.some(a => String(row.r[idx]).toLowerCase().includes(a));
      }
      if (!quoted && s.length > 1 && s.startsWith('-')) {
        const neg = s.slice(1);
        return row => !row.hay.includes(neg);
      }
      const alts = quoted ? [s] : s.split('|').filter(Boolean);
      alts.forEach(a => lits.push(a));
      return row => alts.some(a =>
        row.hay.includes(a) ||
        (!quoted && a.length >= 5 && row.words.some(w => within1(w, a))));
    });
    return { pred: row => preds.every(p => p(row)), lits };
  }
  // highlight positive literals inside a cell, DOM-built (never innerHTML)
  function hcell(text, lits, cls) {
    const td = el('td', cls ? { class: cls } : {});
    const lower = String(text).toLowerCase();
    let cuts = [];
    for (const l of lits) {
      let at = 0;
      while (l && (at = lower.indexOf(l, at)) !== -1) { cuts.push([at, at + l.length]); at += l.length; }
    }
    if (!cuts.length) { td.textContent = String(text); return td; }
    cuts.sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const c of cuts) {
      const last = merged[merged.length - 1];
      if (last && c[0] <= last[1]) last[1] = Math.max(last[1], c[1]);
      else merged.push(c);
    }
    let pos = 0;
    for (const [a, b] of merged) {
      if (a > pos) td.append(String(text).slice(pos, a));
      td.append(el('mark', { text: String(text).slice(a, b) }));
      pos = b;
    }
    if (pos < String(text).length) td.append(String(text).slice(pos));
    return td;
  }
  const CAP = 500;
  const countEl = el('p', { class: 'small', style: 'margin:8px 0 6px' });
  const tbody = el('tbody');
  let current = all;
  function requery() {
    const { pred, lits } = buildPred(state.runsQ);
    let rows = all.filter(pred);
    const { idx, dir } = state.runsSort;
    rows.sort((A, B) => {
      const va = A.r[idx], vb = B.r[idx];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return typeof va === 'number' && typeof vb === 'number'
        ? dir * (va - vb) : dir * natCmp(va, vb);
    });
    current = rows.map(x => x.r);
    countEl.textContent = `${rows.length} of ${all.length} rows`
      + (rows.length > CAP ? ` — showing the first ${CAP}; refine the query` : '');
    tbody.replaceChildren(...rows.slice(0, CAP).map(({ r }) => el('tr', {},
      hcell(r[0], lits), hcell(r[1], lits), hcell(r[2], lits),
      el('td', { class: 'num', text: num(r[3], 4) }),
      el('td', { class: 'num', text: r[4] == null ? '—' : num(r[4], 4) }),
      el('td', { class: 'num', text: r[5] == null ? '—' : String(r[5]) }),
      el('td', { class: 'num', text: r[6] == null ? '—' : String(r[6]) }))));
  }
  // requery() updates the table in place instead of re-rendering the tab, so the
  // input keeps focus and caret across every keystroke.
  const qIn = el('input', { type: 'search', value: state.runsQ,
    style: 'width:100%', role: 'combobox', 'aria-autocomplete': 'list',
    'aria-expanded': 'false', autocomplete: 'off',
    placeholder: 'pythia|gemma task:mmlu_ value>0.3 · "acc_norm" stderr<0.01 · gptneox layers>=24',
    oninput: e => { state.runsQ = e.target.value; requery(); refreshAc(); } });
  // ---- typeahead: complete the token under the caret --------------------------
  // Suggestions come from the data itself (model / task / metric / architecture
  // names) plus the query language (field prefixes, numeric fields), so nobody
  // has to remember what a task is called or how a filter is spelled.
  const AC_VOCAB = {
    model:  [...new Set(all.map(x => x.r[0]))],
    task:   [...new Set(all.map(x => x.r[1]))],
    metric: [...new Set(all.map(x => x.r[2]))],
    arch:   [...new Set(DATA.models.map(m => (m.archinfo || {}).arch).filter(Boolean))],
  };
  function suggestFor(rawTok) {
    let neg = '', tok = rawTok;
    if (tok.startsWith('-') && tok.length > 1) { neg = '-'; tok = tok.slice(1); }
    if (!tok || tok.startsWith('"')) return [];
    const low = tok.toLowerCase();
    const out = [];
    const fm = low.match(/^(model|task|metric|arch):(.*)$/);
    if (fm) {                       // completing a field's value
      const part = fm[2], starts = [], subs = [];
      for (const v of AC_VOCAB[fm[1]] || []) {
        const vl = v.toLowerCase();
        if (part && vl.startsWith(part)) starts.push(v);
        else if (!part || vl.includes(part)) subs.push(v);
      }
      for (const v of [...starts, ...subs])
        out.push({ ins: neg + fm[1] + ':' + v, show: fm[1] + ':' + v, tag: fm[1] });
      return out.slice(0, 8);
    }
    for (const f of ['model:', 'task:', 'metric:', 'arch:'])
      if (f.startsWith(low) && f !== low)
        out.push({ ins: neg + f, show: f, tag: 'field', keep: true });
    if (low.length >= 2)
      for (const f of NUMF.split('|'))
        if (f.startsWith(low) && f !== low)
          out.push({ ins: neg + f + '>', show: f + '> …', tag: 'number', keep: true });
    const buckets = [[], [], []];   // prefix > substring > one-typo fuzzy
    for (const [tag, pool] of Object.entries(AC_VOCAB))
      for (const v of pool) {
        const vl = v.toLowerCase();
        if (vl === low) continue;
        if (vl.startsWith(low)) buckets[0].push({ ins: neg + v, show: v, tag });
        else if (vl.includes(low)) buckets[1].push({ ins: neg + v, show: v, tag });
        else if (low.length >= 4 &&
                 vl.split(/[^a-z0-9_]+/).some(w => within1(w, low)))
          buckets[2].push({ ins: neg + v, show: v, tag });
      }
    out.push(...buckets[0], ...buckets[1], ...buckets[2]);
    return out.slice(0, 8);
  }
  function tokenAt() {              // the whitespace-delimited token under the caret
    const v = qIn.value, c = qIn.selectionStart == null ? v.length : qIn.selectionStart;
    let s = c; while (s > 0 && !/\s/.test(v[s - 1])) s--;
    let e = c; while (e < v.length && !/\s/.test(v[e])) e++;
    return { s, e, tok: v.slice(s, e) };
  }
  let acItems = [], acIdx = -1;
  const acList = el('div', { class: 'ac-list', role: 'listbox',
                             style: 'display:none' });
  function closeAc() {
    acItems = []; acIdx = -1;
    acList.style.display = 'none'; acList.replaceChildren();
    qIn.setAttribute('aria-expanded', 'false');
  }
  function acLabel(show, rawTok) {  // bold the part the user actually typed
    const wrap = el('span', { class: 'ac-name' });
    const t = rawTok.replace(/^-/, '').toLowerCase();
    const at = t ? show.toLowerCase().indexOf(t) : -1;
    if (at < 0) { wrap.textContent = show; return wrap; }
    wrap.append(show.slice(0, at), el('mark', { text: show.slice(at, at + t.length) }),
                show.slice(at + t.length));
    return wrap;
  }
  function refreshAc() {
    const { tok } = tokenAt();
    acItems = suggestFor(tok);
    if (!acItems.length) { closeAc(); return; }
    acIdx = 0;
    acList.replaceChildren(...acItems.map((s, i) => el('div', {
      class: 'ac-item', role: 'option', 'aria-selected': String(i === acIdx),
      // mousedown (not click) + preventDefault: accept BEFORE the input can blur
      onmousedown: ev => { ev.preventDefault(); accept(s); } },
      acLabel(s.show, tok), el('span', { class: 'ac-tag', text: s.tag }))));
    acList.style.display = '';
    qIn.setAttribute('aria-expanded', 'true');
  }
  function accept(sug) {
    const { s, e } = tokenAt();
    const v = qIn.value, tail = sug.keep ? '' : ' ';
    qIn.value = v.slice(0, s) + sug.ins + tail + v.slice(e);
    const pos = s + sug.ins.length + tail.length;
    qIn.setSelectionRange(pos, pos);
    state.runsQ = qIn.value;
    requery();
    refreshAc();                    // field prefixes keep suggesting their values
  }
  qIn.addEventListener('keydown', ev => {
    const open = acItems.length > 0;
    if (!open) { if (ev.key === 'ArrowDown') refreshAc(); return; }
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      acIdx = (acIdx + (ev.key === 'ArrowDown' ? 1 : -1) + acItems.length) % acItems.length;
      [...acList.children].forEach((n, i) =>
        n.setAttribute('aria-selected', String(i === acIdx)));
      const active = acList.children[acIdx];
      if (active && active.scrollIntoView) active.scrollIntoView({ block: 'nearest' });
    } else if (ev.key === 'Enter' || ev.key === 'Tab') {
      ev.preventDefault(); accept(acItems[acIdx] || acItems[0]);
    } else if (ev.key === 'Escape') { closeAc(); }
  });
  qIn.addEventListener('focus', refreshAc);
  qIn.addEventListener('blur', () => setTimeout(closeAc, 120));
  const acWrap = el('div', { class: 'ac' }, qIn, acList);
  const EXAMPLES = ['pythia task:mmlu_ metric:acc value>0.3', 'metric:cross_entropy',
    'gemma|qwen vocab>100000', 'layers>=24 task:hellaswag', '"acc_norm" stderr<0.01'];
  const exChips = el('div', { class: 'chips', style: 'margin:8px 0 0' }, EXAMPLES.map(q =>
    el('button', { class: 'st st-muted', style: 'cursor:pointer', text: q,
      onclick: () => { state.runsQ = q; qIn.value = q; requery(); } })));
  const thead = el('thead', {}, el('tr', {},
    ['model', 'task', 'metric', 'value', 'stderr', 'n-shot', 'items'].map((h, i) =>
      el('th', { class: (i >= 3 ? 'num ' : '') + 'sortable',
        onclick: () => { state.runsSort = { idx: i,
          dir: state.runsSort.idx === i ? -state.runsSort.dir : (i >= 3 ? -1 : 1) };
          requery(); },
        text: h }))));
  requery();
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'Query every metric' }),
    el('p', { class: 'sub', text:
      'Every number from every run — MMLU’s 57 subjects, acc and acc_norm side by side, '
      + 'derived cross-entropy, all of it — with each model\u2019s shape joined in, so '
      + 'architecture is queryable. Terms AND; a|b is OR; "quotes" match exactly; -word '
      + 'excludes; typos within one edit still hit. Fields: model:, task:, metric:, arch:. '
      + 'Numerics: value, stderr, shots, items, vocab, hidden, layers, heads, ctx, params '
      + 'with > < >= <= =. Suggestions appear as you type — ↑↓ to pick, Enter or Tab to '
      + 'insert, Esc to dismiss. Click a column to sort; the CSV button exports exactly '
      + 'what the query matches.' }),
    acWrap, exChips, countEl,
    el('div', { class: 'lb-wrap', style: 'max-height:480px;overflow-y:auto' },
      el('table', { class: 'lb' }, thead, tbody)),
    el('button', { style: 'margin-top:10px', text: 'Download filtered CSV', onclick: () => {
      const esc = v => v == null ? '' : /[",\n]/.test(String(v))
        ? '"' + String(v).replace(/"/g, '""') + '"' : String(v);
      download('benchmark_query.csv', 'text/csv',
        [['model', 'task', 'metric', 'value', 'stderr', 'n_shot', 'n_samples',
          ...PROV_COLS].join(','),
         ...current.map(r => [...r, ...provOf(r[0])].map(esc).join(','))].join('\n'));
    }})));
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'Export' }),
    el('p', { class: 'sub', text: 'The CSV is the flat every-metric table; the JSON is everything this page renders.' }),
    el('div', { style: 'display:flex;gap:8px;margin-top:8px' },
      el('button', { onclick: exportCsv, text: 'Download CSV' }),
      el('button', { onclick: exportJson, text: 'Download JSON' }))));
  return frag;
}

// ---------- live mode: training-run tracking (the wandb-shaped tab) ----------
const trColor = slot => `var(--s${(slot % 8) + 1})`;
const rel = ts => {
  const s = Math.max(0, Date.now() / 1000 - ts);
  return s < 90 ? `${Math.round(s)}s` : s < 5400 ? `${Math.round(s / 60)}m`
       : s < 129600 ? `${Math.round(s / 3600)}h` : `${Math.round(s / 86400)}d`;
};
const absT = ts => ts ? new Date(ts * 1000).toLocaleString() : '—';   // viewer's zone
const fmtv = v => !isFinite(v) ? '—' : Math.abs(v) >= 100 ? Math.round(v).toLocaleString()
              : Math.abs(v) >= 1 ? v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
              : v.toPrecision(3);
const fmtCount = v => v == null ? '—' : v < 1e3 ? String(Math.round(v))
              : v < 1e6 ? (v / 1e3).toFixed(v < 1e4 ? 1 : 0) + 'K'
              : v < 1e9 ? (v / 1e6).toFixed(v < 1e7 ? 1 : 0) + 'M'
              // T matters now that this formats a token axis: modern runs are
              // measured in trillions, and "11000.00B" is not a number anyone reads
              : v < 1e12 ? (v / 1e9).toFixed(2) + 'B'
              : (v / 1e12).toFixed(2) + 'T';
const ema = (pts, a) => {
  if (!a) return pts;
  let s = null;
  return pts.map(([x, y]) => [x, s = (s === null ? y : a * s + (1 - a) * y)]);
};

// One metric panel: lines = selected runs, crosshair readout, checkpoint markers.
function lineChart(title, seriesList, o = {}) {
  const W = 460, H = 235, L = 56, R = 14, T = 14, B = 28;
  const smoothed = seriesList.map(s => ({ ...s, spts: ema(s.pts, o.smooth || 0) }));
  const allPts = smoothed.flatMap(s => s.pts);
  if (!allPts.length) return null;
  let ys = allPts.map(p => p[1]);
  const logY = o.logY && ys.every(y => y > 0);
  if (logY) ys = ys.map(Math.log10);
  // x can be a step (linear, the default) or a quantity that spans orders of
  // magnitude — training tokens, or 6ND compute. Both are opt-in so every
  // existing metric chart keeps exactly the code path it had.
  const logX = !!o.logX && allPts.every(p => p[0] > 0);
  const tx = x => logX ? Math.log10(x) : x;
  const xfmt = o.xfmt || (v => Math.round(v).toLocaleString());
  const xlabel = o.xlabel || 'step';
  const xmin = Math.min(...allPts.map(p => tx(p[0]))),
        xmax = Math.max(...allPts.map(p => tx(p[0])));
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (ymin === ymax) { ymin -= 0.5; ymax += 0.5; }
  const pad = (ymax - ymin) * 0.06;
  ymin -= pad; ymax += pad;
  const X = x => L + (W - L - R) * (tx(x) - xmin) / ((xmax - xmin) || 1);
  const Y = v => { const t = logY ? Math.log10(Math.max(v, 1e-12)) : v;
                   return T + (H - T - B) * (1 - (t - ymin) / (ymax - ymin)); };
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img',
                              'aria-label': title });
  // y gridlines: 4 loose ticks (log mode: powers of ten inside the range)
  const yticks = [];
  if (logY) {
    for (let e = Math.ceil(ymin); e <= Math.floor(ymax); e++) yticks.push(e);
    if (!yticks.length) yticks.push(ymin, ymax);
  } else {
    const step = (ymax - ymin) / 3.5, mag = Math.pow(10, Math.floor(Math.log10(step || 1)));
    const st = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step) || mag;
    for (let v = Math.ceil(ymin / st) * st; v <= ymax; v += st) yticks.push(v);
  }
  for (const t of yticks) {
    const yy = T + (H - T - B) * (1 - (t - ymin) / (ymax - ymin));
    svg.append(el('svg:line', { x1: L, y1: yy, x2: W - R, y2: yy,
      stroke: 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('svg:text', { x: L - 7, y: yy + 3.5, 'font-size': 10,
      fill: 'var(--muted)', 'text-anchor': 'end',
      text: logY ? fmtv(Math.pow(10, t)) : fmtv(t) }));
  }
  for (const xv of [xmin, (xmin + xmax) / 2, xmax]) {
    // xv is in transformed space; place from it directly and label the real value
    svg.append(el('svg:text', {
      x: L + (W - L - R) * (xv - xmin) / ((xmax - xmin) || 1), y: H - 8,
      'font-size': 10, fill: 'var(--muted)', 'text-anchor': 'middle',
      text: xfmt(logX ? Math.pow(10, xv) : xv) }));
  }
  // checkpoint markers first (under the data)
  for (const s of smoothed) for (const ev of (s.events || [])) {
    if (tx(ev.step) < xmin || tx(ev.step) > xmax) continue;
    svg.append(el('svg:line', { x1: X(ev.step), y1: T, x2: X(ev.step), y2: H - B,
      stroke: s.color, 'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: 0.5 }));
  }
  for (const s of smoothed) {
    if (o.smooth > 0 && s.pts.length > 1)
      svg.append(el('svg:polyline', { points: s.pts.map(p => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' '),
        fill: 'none', stroke: s.color, 'stroke-width': 1.2, opacity: 0.25 }));
    svg.append(el('svg:polyline', { points: s.spts.map(p => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' '),
      fill: 'none', stroke: s.color, 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    const last = s.spts[s.spts.length - 1];
    svg.append(el('svg:circle', { cx: X(last[0]), cy: Y(last[1]), r: 4,
      fill: s.color, stroke: 'var(--surface-1)', 'stroke-width': 2 }));
  }
  // crosshair: nearest-step readout across every series, via the shared tooltip
  const hair = el('svg:line', { x1: -9, y1: T, x2: -9, y2: H - B,
    stroke: 'var(--axis)', 'stroke-width': 1 });
  svg.append(hair);
  const steps = [...new Set(allPts.map(p => p[0]))].sort((a, b) => a - b);
  const overlay = el('svg:rect', { class: 'hit', x: L, y: 0, width: W - L - R, height: H,
    onpointermove: e => {
      const box = svg.getBoundingClientRect();
      // pointer position is in transformed space; bring it back before matching
      // against the real x values so the readout names a real step/token/FLOP
      const fxT = xmin + ((e.clientX - box.left) / box.width * W - L) / (W - L - R) * (xmax - xmin);
      const fx = logX ? Math.pow(10, fxT) : fxT;
      let lo = 0, hi = steps.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; steps[mid] < fx ? lo = mid + 1 : hi = mid; }
      const stp = (lo > 0 && fx - steps[lo - 1] < steps[lo] - fx) ? steps[lo - 1] : steps[lo];
      hair.setAttribute('x1', X(stp)); hair.setAttribute('x2', X(stp));
      const rows = [`${xlabel} ${xfmt(stp)} — ${title}`];
      for (const s of smoothed) {
        let best = null;
        for (const p of s.spts) if (best === null || Math.abs(p[0] - stp) < Math.abs(best[0] - stp)) best = p;
        if (best && Math.abs(tx(best[0]) - tx(stp)) <= (xmax - xmin) * 0.05 + (logX ? 0 : 1))
          rows.push(`${s.label}: ${fmtv(best[1])}`);
        const ev = (s.events || []).find(ev => ev.step === stp);
        if (ev) rows.push(`⚑ checkpoint: ${ev.detail}`);
      }
      e.currentTarget.setAttribute('data-tip', JSON.stringify(rows));
    },
    onpointerleave: e => { hair.setAttribute('x1', -9); hair.setAttribute('x2', -9);
                           e.currentTarget.removeAttribute('data-tip'); } });
  svg.append(overlay);
  return el('div', { class: 'panel' }, el('h3', { text: title }), svg);
}

async function loadTraining(force = false) {
  if (!LIVE || state.trFetching) return;
  state.trFetching = true;
  try {
    state.trRuns = await api('api/truns');
    // an empty right pane until someone picks a run was the first thing the
    // tab showed; open on the most recent one instead, once
    let picked = false;
    if (!state.trAutoPicked && !state.trSel.length && state.trRuns.length) {
      state.trAutoPicked = picked = true;
      const latest = [...state.trRuns].sort((a, b) =>
        (b.updated_at || b.created_at || 0) - (a.updated_at || a.created_at || 0))[0];
      state.trSel = [latest.id]; state.trColors = { [latest.id]: 0 };
    }
    await Promise.all(state.trSel.map(async id => {
      const row = state.trRuns.find(r => r.id === id);
      if (force || !state.trSeries[id] || (row && row.status === 'running'))
        state.trSeries[id] = await api(`api/truns/${id}`);
    }));
    // a run picked here changes the right pane, which only a full render builds
    if (state.tab === 'training') (picked ? render : (state.trRedraw || render))();
  } catch (e) { /* next poll retries */ }
  finally { state.trFetching = false; }
}

function toggleRun(id) {
  const i = state.trSel.indexOf(id);
  if (i >= 0) {
    state.trSel.splice(i, 1);
    delete state.trColors[id];
  } else {
    if (state.trSel.length >= 8) { state.qmsg = ''; return; }   // palette cap
    const used = new Set(Object.values(state.trColors));
    let slot = 0; while (used.has(slot)) slot++;
    state.trColors[id] = slot;                 // color follows the run while selected
    state.trSel.push(id);
  }
  render();
  loadTraining();
}

const METRIC_ORDER = ['loss', 'lr', 'grad_norm', 'tokens_per_s', 'gpu_mem_gb', 'gpu_util'];

function vTraining() {
  if (!state.trRuns.length && LIVE && netReady()) loadTraining();
  const frag = [];
  // ---- left: the runs list --------------------------------------------------
  // two lines per run: name + status on top, the numbers underneath. The old
  // single line gave the name whatever width was left — in a 250px column
  // that was zero, and rows showed a status chip and nothing to attach it to.
  // "idle": still 'running' (finish() never came) but silent for longer than
  // this run's own rhythm allows — 5× its usual gap between updates, never
  // less than IDLE_MIN_S. A run logging every 20 min is not idle at 25; a run
  // logging every 5 s that went quiet for an hour is. Display-only: the row
  // says how long, the tooltip says why, and the next log() clears it.
  const IDLE_MIN_S = 30 * 60;
  const idleAfter = r => Math.max(IDLE_MIN_S, 5 * (r.cadence_s || 0));
  const runRow = r => {
    const sel = state.trSel.includes(r.id);
    const silence = Date.now() / 1000 - (r.updated_at || 0);
    const idle = r.status === 'running' && silence > idleAfter(r);
    const meta = [
      r.last_step != null ? `step ${r.last_step.toLocaleString()}` : null,
      r.last_loss != null ? `loss ${fmtv(r.last_loss)}` : null,
      r.tokens != null ? `${fmtCount(r.tokens)} tok` : null,
      r.submitter || null,
      r.updated_at ? `${rel(r.updated_at)} ago` : null,
    ].filter(Boolean).join(' · ');
    return el('div', { class: 'runrow' + (sel ? ' sel' : ''), onclick: () => toggleRun(r.id),
      role: 'button', tabindex: 0,
      title: `${r.name}\nproject: ${r.project} · started ${rel(r.created_at)} ago`
        + (r.updated_at ? ` · last update ${new Date(r.updated_at * 1000).toLocaleString()}` : '')
        + (idle ? `\nno update for ${rel(r.updated_at)}`
             + (r.cadence_s ? ` — this run normally reports every ${rel(Date.now() / 1000 - r.cadence_s)}` : '')
             + `. Still "running" because run.finish() was never called (crash or Ctrl-C?); `
             + 'logging again clears this.' : '') },
      el('span', { class: 'rchip', style: sel ? `background:${trColor(state.trColors[r.id])}` : '' }),
      el('div', { class: 'rbody' },
        el('div', { class: 'rtop' },
          el('span', { class: 'rname', text: r.name }),
          el('span', { class: idle ? 'st st-muted' : r.status === 'running' ? 'st st-active'
                            : r.status === 'failed' ? 'st st-failed' : 'st st-done',
                       text: idle ? `idle ${rel(r.updated_at)}` : r.status })),
        el('div', { class: 'rmeta', text: meta })));
  };
  // filter + order: with three runs this is furniture; with forty (three friends
  // × a dozen sweeps each) it is the difference between a list and a haystack
  const TR_ORDERS = [['updated', 'recently updated'], ['created', 'newest'],
    ['name', 'name'], ['loss', 'best loss'], ['tokens', 'most tokens'],
    ['steps', 'most steps']];
  function trVisible() {
    const q = state.trQ.trim().toLowerCase();
    const nul = v => v == null ? 1 : 0;               // nulls always sink
    const cmp = {
      updated: (a, b) => (b.updated_at || 0) - (a.updated_at || 0),
      created: (a, b) => (b.created_at || 0) - (a.created_at || 0),
      name:    (a, b) => natCmp(a.name, b.name),
      loss:    (a, b) => nul(a.last_loss) - nul(b.last_loss)
                         || (a.last_loss || 0) - (b.last_loss || 0),
      tokens:  (a, b) => nul(a.tokens) - nul(b.tokens)
                         || (b.tokens || 0) - (a.tokens || 0),
      steps:   (a, b) => nul(a.last_step) - nul(b.last_step)
                         || (b.last_step || 0) - (a.last_step || 0),
    }[state.trOrder] || (() => 0);
    return state.trRuns.filter(r =>
      (state.trStatus === 'all' || r.status === state.trStatus) &&
      (!q || `${r.name} ${r.project || ''} ${r.submitter || ''}`
               .toLowerCase().includes(q)))
      .sort(cmp);
  }
  const listWrap = el('div', { class: 'runlist' });
  const listCount = el('span', { class: 'count-note' });
  const trToolbar = el('div', { class: 'toolbar' },
    el('input', { type: 'search', value: state.trQ, style: 'flex:1;min-width:120px',
      placeholder: 'name, project, person…', 'aria-label': 'filter runs',
      oninput: e => { state.trQ = e.target.value; rebuildRows(); } }),
    mkSel('status filter',
      [['all', 'status: all'], ['running', 'running'], ['finished', 'finished'],
       ['failed', 'failed']],
      state.trStatus, v => { state.trStatus = v; rebuildRows(); }),
    mkSel('sort runs', TR_ORDERS.map(([v, l]) => [v, '↕ ' + l]),
      state.trOrder, v => { state.trOrder = v; rebuildRows(); }),
    listCount);
  function rebuildRows() {
    trToolbar.style.display = state.trRuns.length ? '' : 'none';
    if (!state.trRuns.length) {
      listWrap.replaceChildren(
        el('p', { class: 'small', text: 'No runs yet. From training code:' },
          el('pre', { class: 'mono', style: 'white-space:pre-wrap', text:
'run = bench.init("run7", config={"lr": 3e-4})\n'
+ 'run.log({"loss": loss}, step=step)\n'
+ 'run.log_checkpoint(step, model_id)\n'
+ 'run.finish()' })));
      return;
    }
    const rs = trVisible();
    listCount.textContent = `${rs.length} of ${state.trRuns.length}`;
    const pg = paged('training', rs, JSON.stringify([state.trQ, state.trStatus, state.trOrder]),
                     rebuildRows, 20);
    listWrap.replaceChildren(...(rs.length ? [pg.pager, ...pg.rows.map(runRow)]
      : [el('p', { class: 'small', text: 'No run matches the filter.' })]));
  }
  rebuildRows();
  const listCard = el('div', { class: 'card', style: 'margin:0' },
    el('h2', { text: 'Training runs' }),
    el('p', { class: 'sub', text: 'Click to overlay up to 8 runs. Streamed by your '
      + 'training code via bench.init()/run.log() — see API.md.' }),
    trToolbar, listWrap);
  // ---- right: controls + charts ----------------------------------------------
  // getSel() re-derives the selection fresh on every call: the poll swaps
  // state.trSeries entries, and a captured array would pin the stale objects.
  const getSel = () => state.trSel.map(id => ({
    row: state.trRuns.find(r => r.id === id),
    det: state.trSeries[id], slot: state.trColors[id],
  })).filter(x => x.row);
  const right = [];
  let redraw = null, extraWrap = null, buildExtras = null;
  if (!getSel().length) {
    right.push(note('Select a run on the left — charts, config and its benchmark scores appear here.'));
  } else {
    // one panel per metric name, union across selected runs. A run that logs
    // per-block MoE stats has dozens of metrics, so panels group into
    // collapsible sections by their "prefix/" (the wandb convention) and a
    // filter box narrows them.
    const metricCount = el('span', { class: 'count-note' });
    function buildPanels() {
      const selRuns = getSel();
      // the series are still in flight: say so where the charts will be, not in
      // a muted note beside the controls — a run with 100+ per-block metrics
      // takes a moment and a blank pane reads as broken
      if (selRuns.some(x => !x.det))
        return [el('div', { class: 'card' }, el('p', { class: 'small', text:
          `Loading metrics for ${selRuns.filter(x => !x.det).length} run(s)…` }))];
      const all = [...new Set(selRuns.flatMap(x => Object.keys(x.det?.metrics || {})))];
      // a hundred SVGs built in one synchronous pass locks the tab; with that
      // many metrics, open the first section and leave the rest one click away
      const sig = state.trSel.join(',');
      if (sig !== state.trSecSig) {
        state.trSecSig = sig;
        if (all.length > 24) {
          const keys = [...new Set(all.map(n => n.includes('/') ? n.split('/')[0] : ''))];
          state.trSecClosed = {};
          keys.slice(1).forEach(k => { state.trSecClosed[k] = true; });
        }
      }
      const q = state.trMetricQ.trim().toLowerCase();
      const names = all.filter(n => !q || n.toLowerCase().includes(q));
      names.sort((a, b) => {
        const ia = METRIC_ORDER.indexOf(a), ib = METRIC_ORDER.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
      });
      metricCount.textContent = !all.length ? ''
        : q ? `${names.length} of ${all.length} metrics` : `${all.length} metrics`;
      const panelOf = nm => lineChart(nm,
        selRuns.filter(x => x.det?.metrics?.[nm]?.length).map(x => ({
          label: x.row.name, color: trColor(x.slot),
          pts: x.det.metrics[nm], events: x.det.events || [],
        })), { smooth: state.trSmooth, logY: state.trLog });
      const groups = new Map();
      for (const nm of names) {
        const i = nm.indexOf('/');
        const key = i > 0 ? nm.slice(0, i) : '';
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(nm);
      }
      if (groups.size <= 1)
        return [el('div', { class: 'panels' }, names.map(panelOf).filter(Boolean))];
      return [...groups.entries()].map(([key, nms]) => {
        const closed = !!state.trSecClosed[key];
        return el('div', { class: 'msec' + (closed ? ' closed' : '') },
          el('button', { class: 'msec-h', 'aria-expanded': String(!closed),
            onclick: () => { state.trSecClosed[key] = !closed; redraw(); } },
            key || 'metrics', el('span', { class: 'count-note', text: String(nms.length) })),
          closed ? '' : el('div', { class: 'panels' }, nms.map(panelOf).filter(Boolean)));
      });
    }
    // redraw() swaps ONLY the panels — a full render() would replace the slider
    // element mid-drag and kill the gesture after one notch
    const panelsWrap = el('div', {}, buildPanels());
    const smoothVal = el('span', { class: 'count-note',
      text: state.trSmooth ? state.trSmooth.toFixed(2) : 'off' });
    const loadNote = el('span', { class: 'count-note', text:
      getSel().some(x => !x.det) ? 'loading series…' : '' });
    redraw = () => {
      panelsWrap.replaceChildren(...buildPanels());
      smoothVal.textContent = state.trSmooth ? state.trSmooth.toFixed(2) : 'off';
      loadNote.textContent = getSel().some(x => !x.det) ? 'loading series…' : '';
    };
    const logBtn = el('button', { 'aria-pressed': String(state.trLog),
      text: state.trLog ? 'log y: on' : 'log y: off',
      onclick: e => { state.trLog = !state.trLog;
        e.target.textContent = state.trLog ? 'log y: on' : 'log y: off';
        e.target.setAttribute('aria-pressed', String(state.trLog)); redraw(); } });
    const mfilter = el('input', { type: 'search', value: state.trMetricQ,
      placeholder: 'filter metrics…', 'aria-label': 'filter metrics',
      oninput: e => { state.trMetricQ = e.target.value; redraw(); } });
    const ctrl = el('div', { class: 'ctrl' },
      el('span', { class: 'small', text: 'smoothing' }),
      el('input', { type: 'range', min: 0, max: 0.95, step: 0.05, value: state.trSmooth,
        oninput: e => { state.trSmooth = +e.target.value; redraw(); } }),
      smoothVal, logBtn, mfilter, metricCount, loadNote);
    right.push(ctrl, panelsWrap);
    // benchmark-join + config cards, rebuilt in place as events and scores arrive
    buildExtras = () => {
      const selRuns = getSel();
      const out = [];
      // ---- benchmark join: single run selected → scores vs step ---------------
      if (selRuns.length === 1 && selRuns[0].det) {
        const evs = (selRuns[0].det.events || []).filter(e => e.kind === 'checkpoint');
        // Same data, three x-axes. Step is what the trainer counts; tokens is what
        // the model actually saw; compute (6ND) is the only one of the three that
        // compares runs of DIFFERENT model sizes, which is the whole reason to
        // want it. All three come from this run's own metrics — we never guess a
        // token count, so a run that does not log `tokens` simply cannot offer
        // the last two axes and says so.
        const XA = state.trXAxis || 'step';
        const tokSeries = ((selRuns[0].det.metrics || {}).tokens) || [];
        const tokensAt = s => {          // last logged token count at or before s
          let best = null;
          for (const [st, v] of tokSeries) if (st <= s && (!best || st > best[0])) best = [st, v];
          return best ? best[1] : null;
        };
        const xOf = ev => {
          if (XA === 'step') return ev.step;
          const d = tokensAt(ev.step);
          if (!(d > 0)) return null;
          if (XA === 'tokens') return d;
          const mm = DATA.models.find(x => x.id === ev.detail), aa = (mm || {}).archinfo || {};
          const N = aa.active_params || (mm || {}).params;   // sparse: ACTIVE params
          return N ? 6 * N * d : null;
        };
        const series = DATA.accTasks.map((t, i) => ({
          label: t, color: trColor(i),
          pts: evs.map(ev => { const x = xOf(ev), c = cell(t, ev.detail);
                               return (x != null && c) ? [x, c.v] : null; })
                  .filter(Boolean).sort((p, q) => p[0] - q[0]),
        })).filter(s => s.pts.length);
        const haveTokens = tokSeries.length > 0;
        const axisBtn = (v, label, tip) => el('button', {
          class: 'tgl' + (XA === v ? ' on' : ''), text: label,
          title: tip, disabled: (v !== 'step' && !haveTokens) ? '' : null,
          'aria-pressed': String(XA === v),
          onclick: () => { state.trXAxis = v; render(); } });
        const axisCtrl = el('div', { class: 'ctrl', style: 'margin:2px 0 8px' },
          el('span', { class: 'small', text: 'x-axis' }),
          axisBtn('step', 'step', 'optimizer steps, as the trainer counts them'),
          axisBtn('tokens', 'tokens',
            haveTokens ? 'training tokens seen, log scale'
                       : 'this run does not log a `tokens` metric'),
          axisBtn('compute', 'compute (6ND)',
            haveTokens ? '6 x parameters x tokens, in FLOP, log scale — the axis that '
                       + 'compares runs of different model sizes. Uses ACTIVE parameters '
                       + 'for a sparse model.'
                       : 'this run does not log a `tokens` metric'),
          haveTokens ? '' : el('span', { class: 'small',
            text: 'log a `tokens` metric to unlock the other two axes' }));
        const pendingEvs = evs.filter(ev => !DATA.accTasks.some(t => cell(t, ev.detail)));
        // honesty line: "finished" on a run means TRAINING finished — say where the
        // benchmarks are, so a green chip with an empty chart is never confusing
        const stById = new Map(state.queue.map(q => [q.hf_id, q.status]));
        const inQueue = pendingEvs.filter(ev =>
          ['queued', 'preflight', 'waiting_lock', 'waiting_gpu', 'running']
            .includes(stById.get(ev.detail))).length;
        const failedEvs = pendingEvs.filter(ev => stById.get(ev.detail) === 'failed').length;
        let statusLine = `Training ${selRuns[0].row.status} · benchmarks: `
          + `${evs.length - pendingEvs.length}/${evs.length} done`;
        if (inQueue) statusLine += `, ${inQueue} in the eval queue`;
        if (failedEvs) statusLine += `, ${failedEvs} failed (see the Queue tab)`;
        if (series.length || evs.length) {
          const XTITLE = { step: 'benchmark score vs step',
                           tokens: 'benchmark score vs training tokens',
                           compute: 'benchmark score vs training compute (6ND)' };
          const panel = series.length
            ? lineChart(XTITLE[XA], series, {
                logX: XA !== 'step',
                xlabel: XA === 'compute' ? 'compute' : XA === 'tokens' ? 'tokens' : 'step',
                xfmt: XA === 'compute' ? (v => flop(v) + ' FLOP')
                    : XA === 'tokens' ? fmtCount : undefined })
            : note('Checkpoints are queued — scores appear here when evaluation finishes.');
          out.push(el('div', { class: 'card' },
            el('h2', { text: 'Benchmarks along this run' }),
            el('p', { class: 'sub', text: statusLine + '. Every checkpoint this run submitted, '
              + 'joined to its scores on the leaderboard — capability against training '
              + 'effort, next to the loss.' }),
            axisCtrl,
            panel));
        }
      }
      // ---- config: table for one run, diff for several -------------------------
      const cfgs = selRuns.map(x => { try { return JSON.parse(x.row.config || '{}'); }
                                      catch { return {}; } });
      const keys = [...new Set(cfgs.flatMap(c => Object.keys(c)))].sort();
      if (keys.length) {
        const diffRows = keys.map(k => {
          const vals = cfgs.map(c => c[k] === undefined ? '—' : JSON.stringify(c[k]));
          const differ = new Set(vals).size > 1;
          return el('tr', { class: differ && selRuns.length > 1 ? 'diffrow' : '' },
            el('td', {}, el('span', { class: 'mono', text: k })),
            vals.map(v => el('td', { class: 'num cfgv', text: v })));
        });
        out.push(el('div', { class: 'card' },
          el('h2', { text: selRuns.length > 1 ? 'Config diff' : 'Config' }),
          selRuns.length > 1 ? el('p', { class: 'sub', text: 'Highlighted rows differ between the selected runs — usually the whole explanation of why their curves differ.' }) : '',
          el('div', { class: 'lb-wrap' }, el('table', {},
            el('thead', {}, el('tr', {}, el('th', { text: 'key' }),
              selRuns.map(x => el('th', { class: 'num', text: x.row.name })))),
            el('tbody', {}, diffRows)))));
      }
      return out;
    };
    extraWrap = el('div', {}, buildExtras());
    right.push(extraWrap);
  }
  // the 5s poll calls this instead of render(): fresh points, statuses and rows
  // flow in without rebuilding the DOM under the user's cursor
  state.trRedraw = () => {
    rebuildRows();
    if (redraw) redraw();
    if (extraWrap && buildExtras) extraWrap.replaceChildren(...buildExtras());
  };
  frag.push(el('div', { class: 'tr-grid' }, listCard, el('div', {}, right)));
  return frag;
}

// ===========================================================================
// 11g: the Reader. Every file the page names opens here, in the page — a
// sheet from the right. It has an address (read=<kind>:<id>[:<n>]), so a
// pasted link opens it and Back closes it, and it lives in state like an
// opened row, so a poll never closes it. What it shows is always text: a
// file is escaped, its markdown goes through a renderer that allows no HTML,
// and nothing in it is ever run. No hidden (report-half) question reaches it:
// the bank reader asks for the practice half and a count, and the log
// reader's lines come with any line that quotes one withheld.
// ===========================================================================
const READ_KINDS = ['dataset', 'rubric', 'criteria', 'bank', 'log', 'provenance',
                    'proposal'];

function readStr(r) {
  return r ? [r.kind, r.id, r.n].filter(x => x != null && x !== '').join(':') : '';
}
function parseRead(s) {
  const p = String(s || '').split(':');
  if (!READ_KINDS.includes(p[0]) || !p[1]) return null;
  if (p[0] === 'provenance')
    return p[2] ? { kind: 'provenance', id: p[1] + ':' + p.slice(2).join(':') } : null;
  return { kind: p[0], id: p[1], n: p[2] ? (+p[2] || null) : null };
}
const encRead = r => encodeURIComponent(readStr(r)).replace(/%3A/gi, ':');
// the read= part of a hash, and the hash without it
function splitRead(h) {
  const raw = String(h || '').replace(/^#/, '');
  const m = /(?:^|&)read=([^&]*)/.exec(raw);
  return [raw.replace(/(?:^|&)read=[^&]*/, '').replace(/^&/, ''),
          m ? parseRead(decodeURIComponent(m[1])) : null];
}

// Open it from a link or a button. `from` is a selector for what opened it:
// focus goes back there when it closes.
function openReader(r, from) {
  state.readFrom = from || null;
  if (readStr(r) === readStr(state.read)) return;
  navigate({ read: r });
}
function closeReader() {
  if (!state.read) return;
  // the entry before this one is the same page without the reader: that is
  // Back, so Forward opens it again
  const from = (history.state || {}).from;
  if (from != null && !splitRead(from)[1] && splitRead(from)[0] === splitRead(location.hash)[0]) {
    history.back();
    return;
  }
  state.read = null;
  history.replaceState(history.state, '', '#' + hashFor());
  render();
}
// a read link: an <a> with the address as its href (open it in a new tab and
// it opens there too), which opens the reader in place
function readLink(r, text, attrs = {}) {
  const key = readStr(r);
  return el('a', { href: '#' + splitRead(location.hash)[0] + '&read=' + encRead(r),
    'data-read-open': key, text, ...attrs,
    onclick: e => { if (e.metaKey || e.ctrlKey || e.shiftKey) return;
      e.preventDefault(); openReader(r, `[data-read-open="${CSS.escape(key)}"]`); } });
}
function readButton(r, text, attrs = {}) {
  const key = readStr(r);
  return el('button', { class: 'ghost', 'data-read-open': key, text, ...attrs,
    onclick: () => openReader(r, `[data-read-open="${CSS.escape(key)}"]`) });
}

// ---- data -------------------------------------------------------------------
state.readData = {};
function readKey(r) { return r.kind + ':' + r.id; }
async function readFetch(r, extra) {
  const k = readKey(r);
  const put = patch => { state.readData[k] = { ...(state.readData[k] || {}), ...patch }; };
  put({ loading: true, error: '' });
  try {
    let data;
    if (r.kind === 'dataset') {
      const q = (extra && extra.q != null) ? extra.q : ((state.readData[k] || {}).q || '');
      const [head, page] = await Promise.all([api(`api/datasets/${r.id}`),
        api(`api/datasets/${r.id}/items?offset=0&limit=50&q=${encodeURIComponent(q)}`)]);
      data = { head, page };
      put({ q });
    } else if (r.kind === 'rubric' || r.kind === 'criteria') {
      data = await api(`api/exam/rubrics/${encodeURIComponent(r.id)}/read?kind=${r.kind}`);
    } else if (r.kind === 'bank') {
      const topic = topicOfSlug(r.id) || r.id;
      data = await api(`api/exam/bank?topic=${encodeURIComponent(topic)}&half=diagnose`);
    } else if (r.kind === 'log') {
      const tail = (state.readData[k] || {}).tail || 200;
      data = await api(`api/runs/${encodeURIComponent(r.id)}/lines?tail=${tail}`);
      put({ tail });
    } else if (r.kind === 'proposal') {
      // 11j: a proposal opens in the same sheet, as one short card
      data = await api(`api/proposals/${r.id}`);
    } else if (r.kind === 'provenance') {
      const [what, id] = [r.id.split(':')[0], r.id.split(':').slice(1).join(':')];
      data = what === 'dataset' ? { kind: 'dataset', rec: await api(`api/datasets/${id}`) }
        : { kind: 'judge', rec: await api(`api/judge/provenance?model=${encodeURIComponent(id)}`) };
    }
    put({ loading: false, data, v: ((state.readData[k] || {}).v || 0) + 1 });
  } catch (e) {
    put({ loading: false, error: e.message || String(e), v: ((state.readData[k] || {}).v || 0) + 1 });
  }
  renderReader();
}

// ---- the sheet ----------------------------------------------------------------
function readerShell() {
  const title = el('h2', { id: 'readerTitle', class: 'rd-title' });
  const src = el('p', { class: 'rd-src' });
  const acts = el('div', { class: 'rd-acts' });
  const body = el('div', { class: 'rd-body', tabindex: '-1' });
  const close = el('button', { class: 'ghost rd-close', 'data-reader-close': '1',
    'aria-label': 'close the reader', text: '✕ Close', onclick: () => closeReader() });
  const aside = el('aside', { class: 'reader', role: 'dialog', 'aria-modal': 'true',
      'aria-labelledby': 'readerTitle', tabindex: '-1' },
    el('header', { class: 'rd-head' }, el('div', { class: 'rd-titles' }, title, src),
      el('div', { class: 'rd-top' }, acts, close)),
    body);
  const wrap = el('div', { id: 'reader', class: 'reader-wrap' },
    el('div', { class: 'reader-scrim', onclick: () => closeReader() }), aside);
  // focus stays inside while it is open; Esc closes it, unless a menu of
  // its own is open — that closes first
  aside.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !POP.panel) {
      e.preventDefault(); e.stopPropagation(); closeReader(); return;
    }
    if (e.key !== 'Tab') return;
    const f = [...aside.querySelectorAll('a[href],button:not([disabled]),input,textarea,summary,'
      + '[tabindex]:not([tabindex="-1"])')].filter(x => x.offsetParent || x === document.activeElement);
    if (!f.length) return;
    const i = f.indexOf(document.activeElement);
    if (e.shiftKey && (i <= 0)) { e.preventDefault(); f[f.length - 1].focus(); }
    else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
  });
  Object.assign(wrap, { _title: title, _src: src, _acts: acts, _body: body, _aside: aside });
  return wrap;
}

function renderReader() {
  let wrap = document.getElementById('reader');
  const r = state.read;
  if (!r) {
    if (wrap && !wrap._closing) {
      wrap._closing = true;
      wrap.classList.remove('open');
      if (wrap._stop) wrap._stop();
      document.body.classList.remove('reading');
      setTimeout(() => wrap.remove(), motionOff() ? 0 : 240);
      const back = state.readFrom && document.querySelector(state.readFrom);
      state.readFrom = null;
      if (back) back.focus();
    }
    return;
  }
  if (!wrap || wrap._closing) {
    if (wrap) wrap.remove();
    wrap = readerShell();
    document.body.append(wrap);
    document.body.classList.add('reading');
    requestAnimationFrame(() => requestAnimationFrame(() => wrap.classList.add('open')));
    if (motionOff()) wrap.classList.add('open');
    wrap._aside.focus();
  }
  const k = readKey(r);
  if (wrap.dataset.key !== k) {
    if (wrap._stop) wrap._stop();
    wrap._stop = null;
    wrap.dataset.key = k;
    wrap.dataset.kind = r.kind;
    wrap._v = -1;
    if (!state.readData[k] || (!state.readData[k].data && !state.readData[k].loading)) readFetch(r);
  }
  const got = state.readData[k] || {};
  if (wrap._v === (got.v || 0) && !got.loading) { if (wrap._update) wrap._update(r); return; }
  if (got.loading && !got.data) {
    wrap._title.textContent = 'Opening…';
    wrap._body.replaceChildren(skeleton(5, { 'data-loading': 'reader' }));
    wrap._v = got.v || 0;
    return;
  }
  wrap._v = got.v || 0;
  wrap._update = null;
  if (got.error && !got.data) {
    wrap._title.textContent = 'This file could not be opened';
    wrap._src.textContent = '';
    wrap._acts.replaceChildren();
    wrap._body.replaceChildren(el('p', { class: 'warn', 'data-reader-error': '1', text: got.error }));
    return;
  }
  const build = { dataset: readDataset, rubric: readRubric, criteria: readCriteria, bank: readBank,
                  log: readLog, provenance: readProvenance, proposal: readProposal }[r.kind];
  wrap._aside.onkeydown = null;
  build(wrap, r, got.data, got);
  wrap.dataset.ready = '1';
}

// the header's three actions: the file's text to the clipboard, the file to
// disk, and the raw file in a new tab
function readActs(wrap, { copy, download, name, raw }) {
  const blobUrl = t => URL.createObjectURL(new Blob([t], { type: 'text/plain' }));
  wrap._acts.replaceChildren(
    copy ? el('button', { class: 'ghost', 'data-reader-copy': '1', text: 'Copy',
      onclick: () => copyText(typeof copy === 'function' ? copy() : copy, 'the text') }) : '',
    download ? el('a', { class: 'btn ghost', 'data-reader-download': '1', text: 'Download',
      href: typeof download === 'string' ? download : blobUrl(download.text), download: name || '' })
      : '',
    raw ? el('a', { class: 'btn ghost', 'data-reader-raw': '1', href: raw, target: '_blank',
      rel: 'noopener', text: 'Open raw ↗' }) : '');
}

// marks: the matches of a search, as text nodes and <mark>s — never HTML
function marked(text, q) {
  const t = String(text || '');
  const n = (q || '').trim();
  if (!n) return [t];
  const out = [], low = t.toLowerCase(), ql = n.toLowerCase();
  let i = 0, j;
  while ((j = low.indexOf(ql, i)) >= 0) {
    if (j > i) out.push(t.slice(i, j));
    out.push(el('mark', { text: t.slice(j, j + n.length) }));
    i = j + n.length;
  }
  if (i < t.length) out.push(t.slice(i));
  return out;
}
const shortSha = s => s ? String(s).slice(0, 4) + '…' : '—';
const paras = (text, q) => String(text || '').split(/\n\s*\n/).filter(p => p.trim())
  .map(p => el('p', {}, ...marked(p.trim(), q)));

// ---- 1. dataset documents -------------------------------------------------
function readDataset(wrap, r, data, got) {
  const { head, page } = data;
  const pv = head.provenance || {};
  const it = pv.items || {};
  const docs = page.entries.filter(e => e.type === 'doc');
  const q = got.q || '';
  const n = Math.min(Math.max(1, r.n || (docs[0] || {}).n || 1), page.kept || 1);
  wrap._title.textContent = `Dataset #${head.id} · ${head.category || '—'}`;
  wrap._src.replaceChildren(`${head.model || '—'} · ${page.kept} document${page.kept === 1 ? '' : 's'}`
    + (page.fmt === 'free' ? ' · question and answer' : ''),
    head.over_provisional_judge || pv.provisional
      ? el('span', { class: 'badge warn', 'data-demo-only': '1', text: 'Demo only',
          title: [pv.provisional_reason, head.over_provisional_judge
            ? 'proposed from a judge no person has checked yet' : ''].filter(Boolean).join(' · ') })
      : '');
  readActs(wrap, { copy: () => { const d = docs.find(x => x.n === (state.read || {}).n) || docs[0] || {};
      return d.title ? `${d.title}\n\n${d.text || [d.question, d.answer, d.rationale].join('\n\n')}` : ''; },
    download: head.download ? head.download.replace(/^\//, '') : null,
    name: `dataset-${head.id}.jsonl`,
    raw: head.download ? head.download.replace(/^\//, '') : null });
  const missing = page.missing;
  const summary = el('div', { class: 'rd-summary', 'data-dataset-summary': String(head.id) },
    el('p', { class: 'small', 'data-dataset-counts': '1',
      text: `requested ${page.requested} · kept ${page.kept} · missing ${missing}`
        + (pv.focus_mode ? ` · spread ${pv.focus_mode === 'concept' ? 'by concept'
          : pv.focus_mode === 'area' ? 'by area' : 'not spread'}` : '') }),
    // 11k: it was an empty grey box — 11j's textarea took its class name,
    // and a dataset whose provenance predates the frozen spec had nothing to
    // show. The proposal's own words are the fallback
    el('details', { class: 'rd-skill', 'data-dataset-spec': String(head.id),
        open: (state.rv.specOpen || {})[head.id] ? '' : null,
        ontoggle: e => { state.rv.specOpen = { ...(state.rv.specOpen || {}),
          [head.id]: e.target.open }; } },
      el('summary', { text: 'The missing skill ▸' }),
      el('p', { class: 'spec', text: pv.approved_spec || head.spec_text
        || 'no spec recorded' })),
    el('div', { class: 'frm' },
      readLink({ kind: 'provenance', id: 'dataset:' + head.id }, 'Provenance ▸',
        { class: 'small', 'data-dataset-provenance': String(head.id) })));
  const search = el('input', { type: 'search', class: 'rd-search', placeholder: 'search the documents',
    'aria-label': 'search the documents', value: q, 'data-keep': 'reader-search',
    oninput: e => { clearTimeout(search._t);
      const v = e.target.value;
      search._t = setTimeout(() => readFetch(r, { q: v }), 200); } });
  const list = el('ol', { class: 'rd-list', 'aria-label': 'documents', 'data-doc-list': '1' },
    page.entries.map(e => e.type === 'doc'
      ? el('li', { class: 'rd-item' + (e.n === n ? ' on' : ''), 'data-doc': String(e.n),
          'aria-current': e.n === n ? 'true' : null, tabindex: '0',
          onclick: () => go(e.n),
          onkeydown: ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(e.n); } } },
        el('span', { class: 'rd-n mono', text: String(e.n) }),
        el('span', { class: 'rd-t' }, ...marked(e.title || '(no title)', q)),
        el('span', { class: 'rd-m se' }, e.focus ? `Focus: ${e.focus} · ` : '', `${e.words} words`))
      : el('li', { class: 'rd-item gone', 'data-missing-doc': '1' },
          el('span', { class: 'rd-n mono', text: '—' }),
          el('span', { class: 'rd-t', text: 'missing' + (e.focus ? ` · ${e.focus}` : '') }),
          el('span', { class: 'rd-m se', text: e.why || 'no reason recorded' }))),
    page.total > page.entries.length ? el('li', { class: 'small se', text:
      `showing ${page.entries.length} of ${page.total}` }) : '');
  const pane = el('article', { class: 'rd-doc', 'aria-live': 'polite' });
  const prev = el('button', { class: 'ghost', 'data-doc-prev': '1', 'aria-label': 'previous document',
    text: '←', onclick: () => step(-1) });
  const next = el('button', { class: 'ghost', 'data-doc-next': '1', 'aria-label': 'next document',
    text: '→', onclick: () => step(1) });
  function show(k) {
    const d = docs.find(x => x.n === k);
    list.querySelectorAll('.rd-item.on').forEach(x => { x.classList.remove('on');
      x.removeAttribute('aria-current'); });
    const li = list.querySelector(`[data-doc="${k}"]`);
    if (li) { li.classList.add('on'); li.setAttribute('aria-current', 'true');
      li.scrollIntoView({ block: 'nearest' }); }
    const i = docs.findIndex(x => x.n === k);
    prev.disabled = i <= 0; next.disabled = i < 0 || i >= docs.length - 1;
    if (!d) { pane.replaceChildren(el('p', { class: 'se', text: q ? 'No document matches.'
      : 'No documents in this dataset.' })); return; }
    pane.dataset.docOpen = String(k);
    pane.replaceChildren(
      el('div', { class: 'rd-doc-head' },
        el('h3', {}, ...marked(d.title || '(no title)', q)),
        el('span', { class: 'se mono', 'data-doc-pos': '1',
          text: `${i + 1} of ${docs.length}` }), prev, next),
      el('p', { class: 'small se' }, d.focus ? `Focus: ${d.focus} · ` : '', `${d.words} words`),
      page.fmt === 'free'
        ? el('div', { class: 'rd-qa' },
            el('div', { class: 'eyebrow', text: 'Question' }), ...paras(d.question, q),
            el('div', { class: 'eyebrow', text: 'Answer' }), ...paras(d.answer, q),
            d.rationale ? el('div', { class: 'eyebrow', text: 'Rationale' }) : '',
            ...paras(d.rationale, q))
        : el('div', { class: 'rd-prose' }, ...paras(d.text, q)));
  }
  function go(k) {
    state.read = { ...state.read, n: k };
    history.replaceState(history.state, '', '#' + hashFor());
    show(k);
  }
  function step(d) {
    const i = docs.findIndex(x => x.n === (state.read.n || n));
    const t = docs[i + d];
    if (t) go(t.n);
  }
  wrap._keys = e => {
    if (e.target.closest('input, textarea')) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
  };
  wrap._aside.onkeydown = e => wrap._keys(e);
  wrap._body.replaceChildren(summary,
    el('div', { class: 'rd-split' },
      el('div', { class: 'rd-side' }, search, list), pane));
  wrap._update = rr => { if (rr.n && rr.n !== +pane.dataset.docOpen) show(rr.n); };
  show(docs.some(x => x.n === n) ? n : (docs[0] || {}).n);
  if (document.activeElement === document.body && q) search.focus();
}

// ---- 2. rubric: its markdown, safely ------------------------------------------
function mdInline(s) {
  const out = [];
  const re = /(`[^`]+`)|\[([^\]]+)\]\(([^)\s]+)\)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;
  let i = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > i) out.push(s.slice(i, m.index));
    if (m[1]) out.push(el('code', { text: m[1].slice(1, -1) }));
    else if (m[2]) {
      const url = m[3];
      out.push(/^(https?:|mailto:|#)/i.test(url)
        ? el('a', { href: url, target: url.startsWith('#') ? null : '_blank', rel: 'noopener noreferrer',
                    text: m[2] })
        : m[2]);
    } else if (m[4]) out.push(el('strong', { text: m[4].slice(2, -2) }));
    else if (m[5]) out.push(el('em', { text: m[5].slice(1, -1) }));
    i = m.index + m[0].length;
  }
  if (i < s.length) out.push(s.slice(i));
  return out;
}
function mdRender(text) {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [], toc = [];
  let i = 0;
  const flushPara = buf => { if (buf.length) out.push(el('p', {}, ...mdInline(buf.join(' ')))); };
  let para = [];
  while (i < lines.length) {
    const ln = lines[i];
    if (/^\s*```/.test(ln)) {
      flushPara(para); para = [];
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      i++;
      out.push(el('pre', { class: 'md-pre' }, el('code', { text: code.join('\n') })));
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(ln);
    if (h) {
      flushPara(para); para = [];
      const id = 'md-' + toc.length;
      toc.push({ id, level: h[1].length, text: h[2].replace(/[*_`]/g, '') });
      out.push(el('h' + Math.min(6, h[1].length + 2), { id, class: 'md-h' }, ...mdInline(h[2])));
      i++;
      continue;
    }
    if (/^\s*\|/.test(ln) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara(para); para = [];
      const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const head = cells(ln);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd md-table' },
        el('thead', {}, el('tr', {}, head.map(c => el('th', {}, ...mdInline(c))))),
        el('tbody', {}, rows.map(r => el('tr', {}, r.map(c => el('td', {}, ...mdInline(c)))))))));
      continue;
    }
    const li = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(ln);
    if (li) {
      flushPara(para); para = [];
      const ordered = /\d/.test(li[2]);
      const items = [];
      while (i < lines.length) {
        const x = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]);
        if (!x) {
          // a wrapped line belongs to the item above it
          if (/^\s{2,}\S/.test(lines[i]) && items.length) {
            items[items.length - 1].text += ' ' + lines[i].trim(); i++; continue; }
          break;
        }
        items.push({ depth: Math.min(2, Math.floor(x[1].length / 2)), text: x[3] });
        i++;
      }
      out.push(el(ordered ? 'ol' : 'ul', { class: 'md-list' }, items.map(it =>
        el('li', { style: it.depth ? `margin-left:${it.depth * 18}px` : null }, ...mdInline(it.text)))));
      continue;
    }
    if (/^\s*>/.test(ln)) {
      flushPara(para); para = [];
      const q = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ''));
      out.push(el('blockquote', {}, ...mdInline(q.join(' '))));
      continue;
    }
    if (!ln.trim()) { flushPara(para); para = []; i++; continue; }
    if (/^\s*(---+|\*\*\*+)\s*$/.test(ln)) { flushPara(para); para = []; out.push(el('hr')); i++; continue; }
    para.push(ln.trim());
    i++;
  }
  flushPara(para);
  return { nodes: out, toc };
}

function readRubric(wrap, r, d) {
  wrap._title.textContent = d.file;
  wrap._src.replaceChildren(el('span', { class: 'mono', text: d.file }),
    ` · ${d.version ? 'version ' + d.version : 'no version'} · sha `,
    el('span', { class: 'mono', title: d.sha256, text: shortSha(d.sha256) }),
    d.status === 'draft' ? el('span', { class: 'badge taint', text: 'DRAFT' }) : '');
  readActs(wrap, { copy: d.text, download: `api/exam/rubrics/${d.name}`, name: d.file,
                   raw: `api/exam/rubrics/${d.name}` });
  const { nodes, toc } = mdRender(d.text);
  // the contents: the file's top two heading levels, folded when long
  const top = Math.min(...toc.map(t => t.level), 9);
  const tocs = toc.filter(t => t.level <= top + 1);
  const links = el('ul', {}, tocs.map(t => el('li', { style: `margin-left:${(t.level - top) * 12}px` },
    el('a', { href: '#', text: t.text, onclick: e => { e.preventDefault();
      const x = wrap._body.querySelector('#' + t.id); if (x) x.scrollIntoView({ block: 'start' }); } }))));
  wrap._body.replaceChildren(
    tocs.length > 2 ? el('details', { class: 'rd-toc', 'data-rubric-toc': '1',
        open: tocs.length <= 12 ? '' : null, 'aria-label': 'contents' },
      el('summary', { class: 'eyebrow', text: `Contents (${tocs.length})` }), links) : '',
    el('div', { class: 'md', 'data-md': d.name }, nodes));
}

// ---- 3. criteria, as the judge reads them --------------------------------------
function readCriteria(wrap, r, d) {
  wrap._title.textContent = d.file;
  wrap._src.replaceChildren(el('span', { class: 'mono', text: d.file }),
    ` · ${d.criteria.length} criteria · sha `,
    el('span', { class: 'mono', title: d.sha256, text: shortSha(d.sha256) }));
  readActs(wrap, { copy: d.raw, download: `api/exam/rubrics/${d.name}?kind=criteria`,
                   name: d.file, raw: `api/exam/rubrics/${d.name}?kind=criteria` });
  const val = v => v == null ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v);
  // a field that is itself a set of fields (score anchors, a scale) reads
  // as its lines, not as JSON
  const cell = v => v && typeof v === 'object' && !Array.isArray(v)
    ? el('dl', { class: 'rd-kv rd-kv-in' }, Object.entries(v).flatMap(([k, x]) =>
        [el('dt', { text: k }), el('dd', { text: val(x) })]))
    : Array.isArray(v) ? el('ul', {}, v.map(x => el('li', { text: val(x) })))
    : val(v);
  wrap._body.replaceChildren(
    Object.keys(d.top).length ? el('dl', { class: 'rd-kv', 'data-criteria-top': '1' },
      Object.entries(d.top).flatMap(([k, v]) => [el('dt', { text: k }), el('dd', {}, cell(v))])) : '',
    el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-criteria-read': d.name },
      el('thead', {}, el('tr', {}, el('th', { class: 'num', text: '#' }), el('th', { text: 'criterion' }),
        el('th', { class: 'num', text: 'weight' }), el('th', { text: 'when it applies' }),
        el('th', { text: 'what counts as a pass' }))),
      el('tbody', {}, d.criteria.map((c, i) => el('tr', { 'data-criterion-row': c.id || String(i) },
        el('td', { class: 'num', text: String(i + 1) }),
        el('td', {}, el('b', { text: c.name || c.id }), el('div', { class: 'se mono', text: c.id || '' })),
        el('td', { class: 'num', text: c.weight != null ? String(c.weight) : c.max != null ? `max ${c.max}` : '1' }),
        el('td', { text: c.conditional ? 'only when the question calls for it' : 'always' }),
        el('td', { text: c.definition || c.pass || '—' })))))),
    d.flags.length ? el('div', { 'data-criteria-flags': '1' },
      el('div', { class: 'dxh', text: d.flags.length === 1 ? 'The flag' : `Flags (${d.flags.length})` }),
      d.flags.map(f => el('div', { class: 'rd-flag', 'data-flag': f.id },
        el('p', {}, el('b', { text: f.name || f.label || f.id }), ' ',
          el('span', { class: 'se mono', text: f.id })),
        el('p', { class: 'small' }, el('b', { text: 'What it does: ' }), f.effect_words || f.effect || '—'),
        f.condition ? el('p', { class: 'small' }, el('b', { text: 'When: ' }), f.condition) : '',
        (f.examples || []).length ? el('div', {}, el('div', { class: 'eyebrow', text: 'Examples' }),
          el('ul', {}, f.examples.map(x => el('li', { text: val(x) })))) : '',
        (f.not_critical || []).length ? el('div', {},
          el('div', { class: 'eyebrow', text: 'Not critical' }),
          el('ul', {}, f.not_critical.map(x => el('li', { text: val(x) })))) : ''))) : '',
    d.principles.length ? el('div', {}, el('div', { class: 'dxh', text: 'How to judge' }),
      el('ul', {}, d.principles.map(p => el('li', { text: p })))) : '',
    el('details', { class: 'rd-raw' }, el('summary', { text: 'Show raw JSON' }),
      el('pre', { class: 'md-pre', 'data-criteria-raw': '1', text: d.raw })));
}

// ---- 4. the practice half of a topic's questions ------------------------------
function readBank(wrap, r, d) {
  const topic = d.topic;
  wrap._title.textContent = `${topic} · practice questions`;
  wrap._src.textContent = `${d.questions.length} practice questions · ${d.report_count} hidden `
    + 'questions — never shown, by design';
  readActs(wrap, { copy: () => d.questions.map(q => q.prompt).join('\n\n'),
    download: { text: JSON.stringify(d.questions, null, 2) }, name: `${r.id}-practice.json`,
    raw: `api/exam/bank?topic=${encodeURIComponent(topic)}&half=diagnose` });
  const f = state.readBankF = state.readBankF || {};
  const uniq = k => [...new Set(d.questions.map(q => q[k]).filter(x => x != null && x !== ''))]
    .map(String).sort((a, b) => natCmp(a, b));
  const pick = (label, k) => Select(label, [['', `${label}: all`], ...uniq(k).map(v => [v, v])],
    f[k] || '', v => { f[k] = v; draw(); }, { key: 'rd-' + k });
  const search = el('input', { type: 'search', class: 'rd-search', placeholder: 'search the questions',
    'aria-label': 'search the questions', value: f.q || '', 'data-keep': 'reader-bank-q',
    oninput: e => { f.q = e.target.value; draw(); } });
  const list = el('div', { class: 'rd-bank', 'data-bank-list': topic });
  const count = el('p', { class: 'small se', 'data-bank-count': '1' });
  let shown = 50;
  function draw() {
    const q = (f.q || '').trim().toLowerCase();
    const rows = d.questions.filter(x => (!f.difficulty || String(x.difficulty) === f.difficulty)
      && (!f.domain || String(x.domain) === f.domain) && (!f.style || String(x.style) === f.style)
      && (!q || String(x.prompt).toLowerCase().includes(q)));
    count.textContent = `${rows.length} of ${d.questions.length} practice questions`;
    list.replaceChildren(...rows.slice(0, shown).map(x => el('div', { class: 'rd-q', 'data-bank-q': x.qid },
      el('div', { class: 'rd-q-meta small se' },
        x.difficulty != null ? el('span', { class: 'chip', text: `difficulty ${x.difficulty}` }) : '',
        x.domain ? el('span', { class: 'chip', text: x.domain }) : '',
        x.style ? el('span', { class: 'chip', text: x.style }) : '',
        x.written_by ? el('span', { text: `written by ${x.written_by}` }) : ''),
      el('div', { class: 'rd-prose' }, ...paras(x.prompt, f.q)),
      x.reference ? el('details', { class: 'small' }, el('summary', { text: 'Reference answer ▸' }),
        el('div', { class: 'rd-prose' }, ...paras(x.reference))) : '')),
      rows.length > shown ? el('button', { class: 'ghost', text: `Show ${Math.min(50, rows.length - shown)} more`,
        onclick: () => { shown += 50; draw(); } }) : '');
  }
  wrap._body.replaceChildren(
    el('p', { class: 'note', 'data-bank-hidden': String(d.report_count) },
      `${d.report_count} hidden questions — never shown, by design. `,
      el('span', { class: 'se', text: 'They score the model; nothing is ever trained on them.' })),
    el('div', { class: 'frm rd-filters' }, search, pick('difficulty', 'difficulty'),
      pick('domain', 'domain'), pick('style', 'style')),
    count, list);
  draw();
}

// ---- 5. a run's log ---------------------------------------------------------------
function readLog(wrap, r, d, got) {
  const k = readKey(r);
  wrap._title.textContent = `Run #${d.id} · log`;
  wrap._src.textContent = `${d.model} · ${d.status}` + (d.total ? ` · ${d.total.toLocaleString()} lines` : '');
  readActs(wrap, { copy: () => d.lines.join('\n'), download: `api/runs/${d.id}/log?tail=2000`,
    name: `run-${d.id}.log`, raw: `api/runs/${d.id}/log?tail=2000` });
  const opt = state.readLogOpt = state.readLogOpt || { wrap: true };
  const box = el('div', { class: 'rd-log' + (opt.wrap ? ' wrap' : ''), 'data-log': String(d.id),
    tabindex: '0', 'aria-label': 'log lines' });
  const bad = /ERROR|Traceback|failed/i;
  let hits = [], at = -1;
  const search = el('input', { type: 'search', class: 'rd-search', placeholder: 'search the log',
    'aria-label': 'search the log', value: opt.q || '', 'data-keep': 'reader-log-q',
    oninput: e => { opt.q = e.target.value; paint(); } });
  const nextHit = el('button', { class: 'ghost', 'data-log-next': '1', text: 'next ↓', onclick: () => {
    if (!hits.length) return;
    at = (at + 1) % hits.length;
    const row = box.querySelector(`[data-ln="${hits[at]}"]`);
    if (row) { opt.follow = false; row.scrollIntoView({ block: 'center' });
      box.querySelectorAll('.cur').forEach(x => x.classList.remove('cur')); row.classList.add('cur'); } } });
  const follow = el('span', { class: 'small se', 'data-log-follow': '1' });
  const earlier = el('button', { class: 'ghost', 'data-log-earlier': '1', text: 'Load earlier lines',
    disabled: d.first <= 1 || (got.tail || 200) >= 2000 ? '' : null,
    onclick: () => { state.readData[k].tail = Math.min(2000, (got.tail || 200) + 500); readFetch(r); } });
  const wrapBox = el('label', { class: 'small' }, el('input', { type: 'checkbox', 'data-log-wrap': '1',
    checked: opt.wrap ? '' : null, onchange: e => { opt.wrap = e.target.checked;
      box.classList.toggle('wrap', opt.wrap); } }), ' wrap lines');
  // long logs are drawn a chunk a frame, so 2,000 lines never freeze the page
  function paint() {
    const q = (opt.q || '').trim();
    hits = []; at = -1;
    box.replaceChildren();
    let i = 0;
    const chunk = () => {
      const frag = document.createDocumentFragment();
      for (const end = Math.min(d.lines.length, i + 400); i < end; i++) {
        const t = d.lines[i], ln = d.first + i;
        if (q && t.toLowerCase().includes(q.toLowerCase())) hits.push(ln);
        frag.append(el('div', { class: 'rd-ln' + (bad.test(t) ? ' bad' : ''), 'data-ln': String(ln) },
          el('span', { class: 'rd-no', text: String(ln) }), el('span', { class: 'rd-lt' }, ...marked(t, q))));
      }
      box.append(frag);
      if (i < d.lines.length) requestAnimationFrame(chunk);
      else { nextHit.disabled = !hits.length; if (opt.follow) box.scrollTop = box.scrollHeight; }
    };
    chunk();
  }
  // follow: while the run is going, the tail is fetched again and the box
  // stays at the end — until the person scrolls up to read
  if (opt.follow == null || wrap._followFor !== k) { opt.follow = true; wrap._followFor = k; }
  box.addEventListener('scroll', () => {
    const end = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    if (!end && opt.follow && !box._auto) opt.follow = false;
    if (end && !opt.follow) opt.follow = true;
    follow.textContent = d.active ? (opt.follow ? '● following' : 'paused — scroll to the end to follow') : '';
  }, { passive: true });
  follow.textContent = d.active ? (opt.follow ? '● following' : 'paused — scroll to the end to follow') : '';
  wrap._body.replaceChildren(
    el('div', { class: 'frm rd-filters' }, search, nextHit, wrapBox, earlier, follow),
    d.withheld ? el('p', { class: 'small se', text: `${d.withheld} line${d.withheld === 1 ? '' : 's'} `
      + 'withheld: they quote a hidden question.' }) : '',
    d.note ? el('p', { class: 'small se', text: d.note }) : '', box);
  paint();
  if (wrap._stop) wrap._stop();
  if (d.active) {
    const t = setInterval(async () => {
      if (!state.read || readKey(state.read) !== k) { clearInterval(t); return; }
      try {
        const fresh = await api(`api/runs/${d.id}/lines?tail=${got.tail || 200}`);
        const had = d.first + d.lines.length;
        const add = fresh.lines.slice(Math.max(0, had - fresh.first));
        if (fresh.first + fresh.lines.length > had && add.length) {
          const frag = document.createDocumentFragment();
          add.forEach((t2, j) => frag.append(el('div', { class: 'rd-ln' + (bad.test(t2) ? ' bad' : ''),
              'data-ln': String(had + j) },
            el('span', { class: 'rd-no', text: String(had + j) }), el('span', { class: 'rd-lt', text: t2 }))));
          d.lines = d.lines.concat(add);
          box.append(frag);
          if (opt.follow) { box._auto = true; box.scrollTop = box.scrollHeight;
            requestAnimationFrame(() => { box._auto = false; }); }
        }
        d.status = fresh.status;
        if (!fresh.active) { d.active = false; clearInterval(t);
          follow.textContent = ''; wrap._src.textContent = `${d.model} · ${d.status}`; }
      } catch (e) { /* the next tick tries again */ }
    }, 2000);
    wrap._stop = () => clearInterval(t);
  }
}

// ---- 6. provenance: a tree -------------------------------------------------------
function provTree(v, key, depth) {
  const isHash = s => typeof s === 'string' && /^[0-9a-f]{16,}$/i.test(s);
  const isTime = (k, x) => typeof x === 'number' && x > 1e9 && x < 4e9
    && /(_at|^at|created|finished|approved|proposed|requested|generated|judged)$/i.test(String(k));
  const leaf = x => x == null ? el('span', { class: 'se', text: '—' })
    : isHash(x) ? el('span', { class: 'rd-hash' }, el('span', { class: 'mono', title: x, text: shortSha(x) }),
        el('button', { class: 'quiet', 'aria-label': 'copy ' + key, text: 'copy',
          onclick: () => copyText(x, 'the hash') }))
    : isTime(key, x) ? el('span', { title: String(x), text: new Date(x * 1000).toLocaleString() })
    : typeof x === 'boolean' ? el('span', { class: 'mono', text: x ? 'yes' : 'no' })
    : el('span', { class: typeof x === 'number' ? 'mono' : '', text: String(x) });
  if (v && typeof v === 'object') {
    const entries = Array.isArray(v) ? v.map((x, i) => [String(i + 1), x]) : Object.entries(v);
    return el('details', { class: 'rd-node', open: depth < 1 ? '' : null },
      el('summary', {}, el('span', { class: 'rd-k', text: key }),
        el('span', { class: 'se small', text: Array.isArray(v) ? ` ${v.length} item${v.length === 1 ? '' : 's'}`
          : ` ${entries.length} field${entries.length === 1 ? '' : 's'}` })),
      el('div', { class: 'rd-kids' }, entries.length ? entries.map(([k, x]) => x && typeof x === 'object'
        ? provTree(x, k, depth + 1)
        : el('div', { class: 'rd-leaf' }, el('span', { class: 'rd-k', text: k }), leaf(x)))
        : el('span', { class: 'se', text: 'empty' })));
  }
  return el('div', { class: 'rd-leaf' }, el('span', { class: 'rd-k', text: key }), leaf(v));
}
function readProvenance(wrap, r, d) {
  const rec = d.rec;
  if (d.kind === 'dataset') {
    const pv = rec.provenance || {};
    wrap._title.textContent = `Dataset #${rec.id} · provenance`;
    wrap._src.textContent = `${rec.category || '—'} · ${rec.model || '—'} · ${rec.status}`;
    readActs(wrap, { copy: JSON.stringify(pv, null, 2), download: { text: JSON.stringify(pv, null, 2) },
      name: `dataset-${rec.id}-provenance.json`, raw: `api/datasets/${rec.id}` });
    wrap._body.replaceChildren(
      el('div', { class: 'frm', 'data-prov-marks': '1' },
        pv.provisional ? el('span', { class: 'badge warn', 'data-demo-only': '1', text: 'Demo only',
          title: pv.provisional_reason || '' }) : '',
        pv.proposed_over_provisional_judge ? el('span', { class: 'badge warn', 'data-override': '1',
          text: 'proposed past an unchecked judge',
          title: JSON.stringify(pv.proposed_over_provisional_judge) }) : '',
        readLink({ kind: 'dataset', id: String(rec.id) }, 'Read the documents ▸', { class: 'small' })),
      el('div', { class: 'rd-tree', 'data-prov-tree': 'dataset' }, provTree(pv, 'provenance', 0)));
  } else {
    wrap._title.textContent = `How ${rec.model} was graded`;
    wrap._src.textContent = `${rec.runs.length} judge run${rec.runs.length === 1 ? '' : 's'} · `
      + `${Object.keys(rec.tasks).length} topics`;
    readActs(wrap, { copy: JSON.stringify(rec, null, 2), download: { text: JSON.stringify(rec, null, 2) },
      name: 'judge-provenance.json', raw: `api/judge/provenance?model=${encodeURIComponent(rec.model)}` });
    const pre = ((rec.judge_file || {}).preliminary_reasons || []);
    wrap._body.replaceChildren(
      el('div', { class: 'frm', 'data-prov-marks': '1' }, pre.length
        ? el('span', { class: 'badge warn', 'data-demo-only': '1', text: 'Demo only',
            title: pre.join(' · ') }) : ''),
      answerBudgetLine(rec.tasks),
      el('div', { class: 'rd-tree', 'data-prov-tree': 'judge' },
        provTree(rec.runs, 'judge runs', 0), provTree(rec.judge_file, 'the judge', 0),
        provTree(rec.tasks, 'topics', 1)));
  }
}

// 11l: what the answers were generated with, said once where the grading is
// explained — it changes what a score means. A reasoning model gets room to
// answer; an answer that still never finished is counted, not scored.
function answerBudget(g) {
  if (!g) return null;
  const over = /max_gen_toks\s*=\s*(\d+)/.exec(typeof g.override === 'string' ? g.override
    : JSON.stringify(g.override || ''));
  return over ? +over[1] : (g.max_gen_toks != null ? +g.max_gen_toks : null);
}
function answerBudgetLine(tasks) {
  const ts = Object.values(tasks || {});
  const budgets = [...new Set(ts.map(t => answerBudget(t.generation)).filter(x => x != null))]
    .sort((a, b) => a - b);
  const gone = ts.reduce((a, t) => a + (t.no_answer || 0), 0);
  const raised = budgets.some(b => b > 256);
  if (!budgets.length && !gone) return '';
  return el('p', { class: 'small', 'data-answer-budget': budgets.join(',') },
    budgets.length ? `Answers were given up to ${budgets.map(b => b.toLocaleString('en'))
      .join(' or ')} tokens` + (raised ? ' — more than the usual 256, because this model '
        + 'reasons before it answers.' : '.') : '',
    gone ? el('span', { class: 'warn', 'data-no-answer-total': String(gone),
      text: ` ${gone.toLocaleString('en')} answers never finished: they were not scored.` }) : '');
}

// ---------- live mode: submit + queue (only reachable when served by the API) ----------
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const ACTIVE_STATUS = new Set(['preflight', 'waiting_gpu', 'waiting_lock', 'running']);
const stClass = s => s === 'done' ? 'st st-done' : s === 'failed' ? 'st st-failed'
                   : ACTIVE_STATUS.has(s) ? 'st st-active' : 'st st-muted';

// A long failure is two lines, and "details ▸" for the rest (11f): it used
// to push a Queue row to five lines. Open or not lives in state, for polls.
function clampText(text, key) {
  state.clampOpen = state.clampOpen || {};
  const open = !!state.clampOpen[key];
  const body = el('div', { class: 'down clamp' + (open ? ' open' : ''), text });
  const btn = el('button', { class: 'quiet clampbtn', 'data-clamp': key,
    'aria-expanded': String(open), text: open ? 'less ▾' : 'details ▸',
    onclick: () => { state.clampOpen[key] = !open; (state.queueRedraw || render)(); } });
  return el('div', { 'data-clamped': key }, body, text.length > 90 || text.includes('\n') ? btn : '');
}

// What a row can do, by what it is. Queued: Cancel. Running: Log and Cancel,
// which asks first. Failed or canceled: the reason is in the row, and
// Resubmit is one click — same model, suite and topics. Done: Open results —
// the topic page for one topic, the model page otherwise.
async function queueCancel(r) {
  try {
    const j = await post(`api/submissions/${r.id}/cancel`);
    toast(j.status === 'canceling' ? `Stopping #${r.id} — the task in flight ends first`
                                   : `Canceled #${r.id}`, { key: 'cancel' });
  } catch (e) { state.qmsg = `#${r.id}: ${e.message}`; }
  state.qConfirm = null;
  await loadQueue(); (state.queueRedraw || render)();
}

async function queueResubmit(r, regrade = false, extra = {}) {
  let tasks = [];
  try { tasks = JSON.parse(r.tasks || '[]'); } catch (e) { /* older row */ }
  state.qRc = null;
  try {
    const j = await post('api/submissions', { hf_id: r.hf_id, kind: r.kind || 'auto',
      suite: r.suite, note: r.note || '', submitter: whoName() || r.submitter || '',
      ...(tasks.length ? { tasks } : {}), ...extra });
    rememberQueued(j.id);
    markQueueRow(j.id);
    toast(j.note ? `#${j.id}: ${j.note}`
      : regrade ? `Grading #${r.id}'s answers again as #${j.id} — they are on disk, so no GPU`
      : `Queued #${j.id} again — ${r.hf_id}`, { key: regrade ? 'regrade' : 'resubmit' });
  } catch (e) { state.qmsg = `#${r.id}: ${e.message}`; }
  await loadQueue(); (state.queueRedraw || render)();
}

// the runner's own words when preflight met an auto_map without leave to run it
const ownCodeFailure = r => /auto_map|own model code/.test(r.error || '');
function ownCodeResubmit(r) {
  const redraw = () => (state.queueRedraw || render)();
  const info = codeInfo(r.hf_id, redraw);
  const keep = el('button', { class: 'ghost', 'data-own-code-keep': String(r.id), text: 'Keep it',
    onclick: () => { state.qRc = null; redraw(); } });
  if (info === null) return el('div', { class: 'owncode-row' },
    el('span', { class: 'small se', text: 'Checking the checkpoint…' }), keep);
  const why = ownCodeWhy(info, true);
  if (why || !info || !info.own_code) return el('div', { class: 'owncode-row' },
    el('span', { class: 'small', 'data-own-code-why': String(r.id),
      text: why || 'This checkpoint no longer ships its own model code.' }),
    why ? '' : el('button', { class: 'ghost', text: 'Resubmit', onclick: () => queueResubmit(r) }),
    keep);
  const go = el('button', { class: 'primary', 'data-own-code-go': String(r.id), text: 'Resubmit',
    disabled: state.qRcAllow ? null : '',
    onclick: () => queueResubmit(r, false, { allow_remote_code: true }) });
  return el('div', { class: 'owncode-row' },
    el('span', { class: 'small', text: `#${r.id} stopped because this checkpoint ships its `
      + 'own model code. Resubmit it with leave to run that code:' }),
    ownCodeBox(info, state.qRcAllow, v => { state.qRcAllow = v; go.disabled = !v; }, 'q' + r.id),
    go, keep);
}

function queueOpen(r) {
  let tasks = [];
  try { tasks = JSON.parse(r.tasks || '[]'); } catch (e) { /* older row */ }
  const exam = tasks.filter(t => t.startsWith('exam_'));
  if (r.suite === 'judged' && exam.length === 1) {
    state.ans.model = r.hf_id; state.ans.rows = null;
    state.after = { scroll: '[data-panel="answers"]' };
    return navigate({ topic: exam[0].replace(/^exam_/, ''), model: null });
  }
  if (DATA.models.some(m => m.id === r.hf_id)) return navigate({ model: r.hf_id, topic: null });
  toast(`${r.hf_id} is not on the board yet — its results land on the next refresh`);
}

// ---------------------------------------------------------------------------
// 11f: one action cell, for every table with actions. The row's next step is
// its one visible button — a ghost, except a step that needs a person, which
// is filled — and everything else is in a ⋯ menu on the shared popover.
// Right-aligned, 8px apart, every button 32px tall with radius 6.
// ---------------------------------------------------------------------------
function actCell(key, main, items, attrs = {}) {
  const more = (items || []).filter(Boolean);
  const menu = more.length ? popover(el('button', { class: 'ghost rowmenu', 'data-row-menu': key,
      'aria-label': 'more actions', title: 'more actions', text: '⋯' }),
    () => el('div', { class: 'moremenu', id: 'pop-' + key, 'aria-label': 'row actions' },
      more.map(it => it.href
        ? el('a', { role: 'menuitem', class: 'menulink', href: it.href, 'data-act': it.act || null,
            target: it.blank ? '_blank' : null, rel: it.blank ? 'noopener' : null, text: it.label,
            onclick: () => popClose() })
        : el('button', { role: 'menuitem', 'data-act': it.act || null, text: it.label,
            onclick: () => { popClose(); it.run(); } }))),
    { key, placement: 'bottom-end' }) : '';
  return el('div', { class: 'actcell', ...attrs }, main || '', menu);
}

// a small copy, with a fallback where the clipboard API is not allowed
async function copyText(t, what) {
  try { await navigator.clipboard.writeText(t); }
  catch (e) {
    const ta = el('textarea', { style: 'position:fixed;opacity:0' });
    ta.value = t; document.body.append(ta); ta.select();
    try { document.execCommand('copy'); } catch (x) { /* nothing else to try */ }
    ta.remove();
  }
  toast(`Copied ${what || t}`, { key: 'copy' });
}

// 11k: the stage, not the table's column. #58 said "done" for several
// minutes while the judge was still grading 240 of 570 answers, with an
// empty action cell: it looked as if nothing was happening.
const stillGrading = r => r.suite === 'judged' && r.judge && r.judge.status !== 'done'
  && !r.judge_failed;
function runStage(r) {
  if (r.status === 'done' && stillGrading(r))
    return { key: 'grading', cls: 'running', text: 'grading ' + judgeCount(r.judge) };
  if (r.status === 'running') {
    const m = /^(\d+)\s*\/\s*(\d+)/.exec(r.progress || '');
    return { key: 'running', cls: 'running', text: m ? `running ${m[1]}/${m[2]}` : 'running' };
  }
  return { key: r.status, cls: r.status, text: r.status };
}

function queueActions(r) {
  const id = String(r.id);
  // 11g: the log opens in the reader; the raw text is one item further down
  const log = { label: 'Log', act: 'log', run: () => openReader({ kind: 'log', id: id },
    `[data-row-menu="q${id}"]`) };
  const raw = { label: 'Open raw log ↗', act: 'log-raw', href: `api/runs/${r.id}/log`, blank: true };
  const resubmit = { label: 'Resubmit', act: 'resubmit', run: () => queueResubmit(r) };
  const copy = { label: 'Copy id', act: 'copy-id', run: () => copyText(id, '#' + id) };
  const page = DATA.models.some(m => m.id === r.hf_id)
    ? { label: 'Open model page', act: 'model', run: () => navigate({ model: r.hf_id, topic: null }) }
    : null;
  const ghost = (attr, text, run, extra = {}) => el('button', { class: 'ghost', [attr]: id, text,
    onclick: run, ...extra });
  const cell = (main, ...menu) => actCell('q' + id, main, [log, raw, ...menu, copy, page]);
  if (r.status === 'queued')
    return cell(ghost('data-row-cancel', 'Cancel', () => queueCancel(r)));
  if (ACTIVE_STATUS.has(r.status)) {
    // Cancel on a running job asks first, in its own place: the question and
    // its answer replace the button until one is chosen
    if (state.qConfirm === r.id)
      return actCell('q' + id, el('span', { class: 'confirm' },
        el('span', { class: 'small', text: 'Stop this run?' }),
        el('button', { class: 'danger', 'data-row-stop': id, text: 'Stop it',
          onclick: () => queueCancel(r) }),
        el('button', { class: 'ghost', text: 'Keep it',
          onclick: () => { state.qConfirm = null; (state.queueRedraw || render)(); } })), []);
    return cell(ghost('data-row-cancel', 'Cancel',
      () => { state.qConfirm = r.id; (state.queueRedraw || render)(); }));
  }
  if (r.status === 'canceling') return cell(el('span', { class: 'small se', text: 'stopping…' }));
  // only the grading failed: the answers are on disk, and a retry re-grades
  // them — 10b's resume answers nothing again, so it costs no GPU. It is the
  // one filled button: it needs a person
  if (r.judge_failed)
    return cell(el('button', { class: 'primary', 'data-row-regrade': id, text: 'Retry grading',
      title: 'queue this run again: the answers it wrote are kept and graded again — no GPU',
      onclick: () => queueResubmit(r, true) }), resubmit);
  // 11i: it failed because the checkpoint ships its own model code. A plain
  // Resubmit would fail the same way; this one asks first, with the box
  if (r.status === 'failed' && ownCodeFailure(r)) {
    return cell(ghost('data-row-resubmit', 'Resubmit…', () => {
      state.qRc = state.qRc === r.id ? null : r.id; state.qRcAllow = false;
      delete state.codeInfo[r.hf_id];
      (state.queueRedraw || render)(); },
      { title: 'this checkpoint ships its own model code — resubmitting asks first',
        'aria-expanded': String(state.qRc === r.id) }));
  }
  if (r.status === 'failed' || r.status === 'canceled')
    return cell(ghost('data-row-resubmit', 'Resubmit', () => queueResubmit(r),
      { title: `the same model, suite${r.suite === 'judged' ? ' and topics' : ''}, queued again` }));
  if (r.status === 'done') {
    // never an empty cell: while the judge works, the row says so
    if (stillGrading(r))
      return cell(el('span', { class: 'badge', 'data-grading': id,
        title: 'the answers are in; the judge is grading them',
        text: `Grading… ${judgeCount(r.judge)}` }), resubmit);
    return cell(ghost('data-row-open', 'Open results', () => queueOpen(r)), resubmit);
  }
  return cell('');
}

function vQueue() {
  // the judged suite's availability (and its reason when it has none) comes
  // from the same endpoint the Loop tab reads
  if (!state.loop.loaded && netReady()) loadLoop();
  // the form's values live in state: picking a model re-renders (to set its
  // kind), and a re-render must not wipe the suite and note already chosen
  const sf = state.sub;
  const f = {
    hf_id: modelBox('submit', sf.hf_id, v => { sf.hf_id = v; },
      it => { sf.hf_id = it.id; if (it.kind) sf.kind = it.kind; sf.allow = false;
              delete state.codeInfo[it.id]; render(); },
      { 'aria-label': 'model id',
        placeholder: 'search: org/model on the Hub, or local/<name> for an uploaded artifact' }),
    kind: Select('kind', [['auto', 'kind: auto-detect'], ['base', 'kind: base'],
      ['instruct', 'kind: instruct']], sf.kind || 'auto', v => { sf.kind = v; },
      { key: 'submit-kind' }),
    // judged is offered even when it cannot run: an option that is simply
    // absent tells a person nothing, and 'why is there no judged suite?' was
    // the first question asked of this page
    // 11m: masein read the drop-down as `judged` and nothing else, and took
    // `full` for removed. Each option says what it gets you.
    suite: Select('suite', [
      ['full', 'full — every task', { sub: 'The benchmark tasks: the model\'s average and its '
        + 'place on the leaderboard.' }],
      ['quick', 'quick — three tasks, minutes', { sub: 'hellaswag, arc_easy and perplexity: a '
        + 'first look. A full run later adds only the tasks still missing.' }],
      ['control', 'control — MMLU, options rotated', { sub: 'The position-bias experiment: '
        + 'MMLU with the options moved round. About a fifth of a full MMLU.' }],
      ['judged', 'judged — the written exam' + (state.loop.blocked ? ' (unavailable)' : ''),
        { disabled: !!state.loop.blocked, title: state.loop.blocked || '',
          sub: 'The exam topics, answered in writing and graded by the judge: the model\'s '
            + 'judged score per topic.' }]],
      sf.suite || 'full', v => { sf.suite = v; render(); }, { key: 'submit-suite' }),
    note: el('input', { type: 'text', placeholder: 'note (optional)', style: 'flex:1;min-width:140px',
      'aria-label': 'note', 'data-keep': 'submit-note', value: sf.note,
      oninput: e => { sf.note = e.target.value; } }),
  };
  // judged: 11i's grouped picker, the one the model page uses — every exam
  // topic ticked to begin with, the MMLU control off. The ticks are sent as
  // they are: the whole exam is 37 names, not an empty list
  const built = (state.loop.built || []);
  const J = DATA.judged || {};
  if (sf.tasks == null && built.length) sf.tasks = built.filter(t => t !== J.control);
  const topicBoxes = sf.suite === 'judged' && built.length ? el('div', { 'data-submit-topics': '1' },
    el('p', { class: 'small', text: 'topics in this run:' }),
    examPicker(sf, sf.hf_id.trim(), 'submit', () => gateSubmit())) : '';
  const judgedOff = () => (sf.suite === 'judged' && judgeDown()) || cannotRun(sf.hf_id);
  // 11i: a picked checkpoint that ships its own model code says so here,
  // before Submit — the box to allow it, or why this server will not
  const info = codeInfo(sf.hf_id);
  const ownWhy = el('span', { class: 'propwhy', 'data-why': 'own-code' });
  const btn = el('button', { class: 'primary', text: 'Submit model', onclick: async () => {
    const body = { hf_id: sf.hf_id.trim(), kind: sf.kind, suite: sf.suite,
                   submitter: whoName(), note: sf.note };
    if (sf.suite === 'judged') {
      body.tasks = [...(sf.tasks || []), ...(sf.control && J.control ? [J.control] : [])];
      if (!body.tasks.length) { state.qmsg = 'pick at least one topic'; render(); return; }
    }
    if (!body.hf_id) { state.qmsg = 'enter a Hugging Face model id first'; render(); return; }
    if (cannotRun(body.hf_id)) { state.qmsg = noWeightsWhy(body.hf_id); render(); return; }
    if (judgedOff()) { state.qmsg = judgeWhy(); render(); return; }
    if (/^local\/[^/]+$/.test(body.hf_id)) {
      const code = ownCodeWhy(await loadCodeInfo(body.hf_id), !!sf.allow);
      if (code) { state.qmsg = code; render(); return; }
      if (sf.allow && (state.codeInfo[body.hf_id] || {}).own_code) body.allow_remote_code = true;
    }
    // while it is in flight the button says so and cannot be pressed twice
    btn.disabled = true;
    const said = btn.textContent;
    btn.textContent = 'Queueing…';
    try {
      const r = await fetch('api/submissions', { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
        body: JSON.stringify(body) });
      const j = await r.json().catch(() => ({}));
      if (r.ok) {
        state.qmsg = '';
        rememberQueued(j.id);
        // the form goes back to where it started: an empty model box and the
        // ticks it opens with, so the next submission is not the last one's
        sf.hf_id = ''; sf.note = ''; sf.allow = false;
        sf.tasks = null; sf.control = false;
        toast(j.note ? `#${j.id}: ${j.note} —` : `Run #${j.id} queued —`,
              { key: 'submit', go: () => followRun(j.id), link: 'follow it →' });
      } else {
        // refused: every field stays as it was, with the server's words
        state.qmsg = 'Refused. ' + (typeof j.detail === 'string' ? j.detail
          : j.detail ? JSON.stringify(j.detail) : `the server answered HTTP ${r.status}`);
      }
    } catch (e) { state.qmsg = 'Refused. The server is unreachable — it may be restarting.'; }
    btn.disabled = false; btn.textContent = said;
    await loadQueue(); render();
  }});
  const qrow = r => el('tr', { 'data-queue-row': String(r.id),
      class: state.qMark === r.id ? 'landed' : null },
    el('td', { class: 'num', text: '#' + r.id }),
    el('td', { class: 'small', style: 'white-space:nowrap',
      title: `submitted ${absT(r.created_at)}`
        + (r.started_at ? `\nstarted ${absT(r.started_at)}` : '')
        + (r.finished_at ? `\nfinished ${absT(r.finished_at)}` : ''),
      text: r.created_at ? rel(r.created_at) + ' ago' : '—' }),
    el('td', { title: (r.note || '') + (r.arch && r.arch.length > 2 ? (() => {
        try { const a = JSON.parse(r.arch);
              return `\n${a.arch || ''} · hidden ${a.hidden ?? '—'} · layers ${a.layers ?? '—'} · vocab ${a.vocab ?? '—'}`; }
        catch { return ''; } })() : '') }, r.hf_id,
      el('span', { class: 'badge' + (r.kind === 'instruct' ? ' instruct' : ''), text: r.kind })),
    // a judged row says what it sat, in one line: the list is behind ▸ (11i)
    el('td', { class: 'small' }, suiteCell(r, 'q')),
    el('td', { text: r.submitter || '—' }),
    el('td', { 'data-watch': `q|${r.id}|status` }, (() => { const st = runStage(r);
      return el('span', { class: stClass(st.cls), 'data-stage': st.key, text: st.text }); })()),
    el('td', { class: 'small', text: r.progress || '', 'data-watch': `q|${r.id}|progress` },
      // the GPU half finishing is not the job finishing: the judge batch is
      // still out, and the row says how far it is
      r.judge && !r.judge_failed ? el('div', { class: 'se', 'data-judge-progress': judgeCount(r.judge),
        title: `judge batch ${r.judge.batch_id}`,
        text: r.judge.status === 'done' ? `judged ${r.judge.n_items} answers`
          : `judging ${judgeCount(r.judge)}` }) : '',
      // written for whoever reads the queue, not for whoever wrote the
      // runner: one plain line, and the container hostnames behind details
      r.judge_failed ? el('div', { 'data-judge-failed': '1' },
        el('div', { class: 'down', text: 'The grading model isn\'t running. Start it, then '
          + 'Retry grading — the answers are kept.' }),
        el('details', { class: 'small' }, el('summary', { text: 'details' }),
          el('div', { class: 'se', style: 'white-space:pre-wrap;overflow-wrap:anywhere',
            text: [r.error, r.judge && r.judge.error].filter(Boolean).join('\n') })))
        : r.error ? clampText(r.error, 'q' + r.id) : ''),
    el('td', { class: 'num nowrap', text: r.gpu_seconds
      ? Math.max(1, Math.round(r.gpu_seconds / 60)) + '\u00a0min' : '—' }),
    el('td', { class: 'rowacts' }, queueActions(r)));
  // ---- queue filter + sort: a long shared queue needs "my jobs, failures first" ----
  const QCOLS = [
    { key: 'id',          label: '#', num: true, defDir: -1 },
    { key: 'created_at',  label: 'submitted', num: true, defDir: -1 },
    { key: 'hf_id',       label: 'model' },
    { key: 'suite',       label: 'suite' },
    { key: 'submitter',   label: 'by' },
    { key: 'status',      label: 'status' },
    { key: null,          label: 'progress' },
    { key: 'gpu_seconds', label: 'gpu', num: true, defDir: -1 },
    { key: null,          label: '' },
  ];
  function qVisible() {
    const q = state.qQ.trim().toLowerCase();
    const rows = state.queue.filter(r =>
      (state.qStatus === 'all'
        || (state.qStatus === 'active' ? ACTIVE_STATUS.has(r.status)
                                       : r.status === state.qStatus)) &&
      (!q || `#${r.id} ${r.hf_id} ${r.submitter || ''} ${r.note || ''} ${r.status} `
               .toLowerCase().includes(q)));
    const c = QCOLS.find(x => x.key === state.qSort.key) || QCOLS[0];
    return rows.sort((a, b) => {
      const va = a[c.key], vb = b[c.key];
      if (va == null && vb == null) return 0;
      if (va == null) return 1; if (vb == null) return -1;
      return state.qSort.dir * (c.num ? va - vb : natCmp(va, vb));
    });
  }
  const qCount = el('span', { class: 'count-note' });
  const qToolbar = el('div', { class: 'toolbar', style: 'margin-top:2px' },
    el('input', { type: 'search', value: state.qQ, style: 'flex:1;min-width:160px',
      placeholder: 'filter: model, person, note…', 'aria-label': 'filter queue',
      oninput: e => { state.qQ = e.target.value; rebuildQueue(); } }),
    mkSel('status filter',
      [['all', 'status: all'], ['active', 'active'], ['queued', 'queued'],
       ['done', 'done'], ['failed', 'failed'], ['canceled', 'canceled']],
      state.qStatus, v => { state.qStatus = v; rebuildQueue(); }),
    qCount);
  const qThead = el('thead');
  const qTbody = el('tbody');
  const qPager = el('div');
  const qTableWrap = el('div', { class: 'lb-wrap stick' }, el('table', { 'data-queue-table': '1' },
    qThead, qTbody));
  const qEmpty = empty('Nothing in the queue yet. Submit a model above — it runs here, one '
    + 'at a time.', 'Submit a model', () => {
      const i = document.querySelector('[data-ms="submit"] input'); if (i) i.focus(); });
  // in place, same reason as everywhere: the 5s poll must never eat a keystroke
  function rebuildQueue() {
    const any = state.queue.length > 0;
    qToolbar.style.display = any ? '' : 'none';
    qTableWrap.style.display = any ? '' : 'none';
    qEmpty.style.display = any ? 'none' : '';
    if (!any) return;
    qThead.replaceChildren(el('tr', {}, QCOLS.map(c => c.key
      ? el('th', { class: (c.num ? 'num ' : '') + 'sortable',
          onclick: () => { state.qSort = { key: c.key,
            dir: state.qSort.key === c.key ? -state.qSort.dir
               : (c.defDir || 1) }; rebuildQueue(); },
          'aria-sort': state.qSort.key === c.key
            ? (state.qSort.dir > 0 ? 'ascending' : 'descending') : 'none' },
          c.label + ' ', state.qSort.key === c.key
            ? el('span', { class: 'dir', text: state.qSort.dir > 0 ? '▲' : '▼' }) : '')
      : el('th', { text: c.label }))));
    const rs = qVisible();
    qCount.textContent = `${rs.length} of ${state.queue.length}`;
    const pg = paged('queue', rs, JSON.stringify([state.qQ, state.qStatus, state.qSort]),
                     rebuildQueue, 25, true);
    if (pg.pager.parentNode !== qPager) qPager.replaceChildren(pg.pager);
    const was = snapWatch(qTbody);
    // 11i: a row whose Resubmit asks about its own model code opens a row
    // of its own under it — the box is too long for the action cell
    qTbody.replaceChildren(...pg.rows.flatMap(r => state.qRc === r.id && ownCodeFailure(r)
      ? [qrow(r), el('tr', { class: 'owncode-tr', 'data-own-code-row': String(r.id) },
          el('td', { colspan: String(QCOLS.length) }, ownCodeResubmit(r)))]
      : [qrow(r)]));
    markChanged(qTbody, was);
    if (qTbody.isConnected) popReanchor();     // an open ⋯ hangs from the new button
  }
  state.queueRedraw = rebuildQueue;
  rebuildQueue();
  function gateSubmit() {
    const code = info ? ownCodeWhy(info, !!sf.allow) : '';
    const w = judgedOff() ? (cannotRun(sf.hf_id) ? noWeightsWhy(sf.hf_id.trim()) : judgeWhy())
      : code || (sf.suite === 'judged' && built.length && !(sf.tasks || []).length
                 && !sf.control ? 'Tick at least one topic.' : '');
    btn.disabled = !!w;
    btn.title = w;
    ownWhy.textContent = judgedOff() ? '' : code;
    ownWhy.hidden = !ownWhy.textContent;
  }
  gateSubmit();
  return [
    el('div', { class: 'card' },
      el('h2', { text: 'Submit a model' }),
      el('p', { class: 'sub', text:
        'Any public (or server-accessible) Hugging Face model up to the size cap. Preflight '
        + 'checks the repo before any GPU is spent, and one run goes at a time. Results land '
        + 'on this leaderboard automatically.' }),
      el('p', { class: 'small', 'data-suite-help': '1' },
        el('b', { text: 'full' }), ' and ', el('b', { text: 'judged' }),
        ' are separate runs, not one inside the other: a model needs both to have an average '
        + 'and a judged score. Resubmitting is free — each run does only the tasks still '
        + 'missing, which is also how a quick run becomes a full one.'),
      el('div', { class: 'frm' }, f.hf_id, f.kind, f.suite, f.note, btn,
        cannotRun(sf.hf_id)
          ? el('span', { class: 'propwhy', 'data-why': 'weights',
                         text: noWeightsWhy(sf.hf_id.trim()) })
          : sf.suite === 'judged' && judgeDown()
          ? el('span', { class: 'propwhy', 'data-why': 'submit', text: judgeWhy() }) : '',
        ownWhy),
      ownCodeBox(info, sf.allow, v => { sf.allow = v; gateSubmit(); }, 'submit'),
      topicBoxes,
      state.qmsg ? el('p', { class: 'warn', 'data-qmsg': '1', style: 'margin-top:8px',
        text: state.qmsg }) : ''),
    el('div', { class: 'card' },
      el('h2', { text: 'Queue' }),
      qToolbar, qPager, qTableWrap, qEmpty)];
}

// ---------- Review: the human in the loop ----------
// Proposals wait here for a person. Everything DIAGNOSE.md says that person
// needs is on the card: the spec, the evidence counts, the category's score and
// ceiling, the item counts, the model's own findings for the task, and up to
// eight diagnosis-half items the LLM saw, labelled as such. Approve (edited or
// not) under a typed name, or reject with a reason. Only an approved spec can
// be sent to the generator, and only the spec text goes.
let _rvInFlight = false;
async function loadReview() {
  if (_rvInFlight) return;
  _rvInFlight = true;
  try {
    const [llm, props, ds] = await Promise.all([
      api('api/llm'), api('api/proposals'), api('api/datasets')]);
    const changed = !state.rv.loaded || JSON.stringify([llm, props, ds])
      !== JSON.stringify([state.rv.llm, state.rv.proposals, state.rv.datasets]);
    // a batch that finished while this page was open says what it holds and
    // what never arrived — the same line the card carries
    const was = new Map((state.rv.datasets || []).map(d => [d.id, d.status]));
    for (const d of ds) {
      if (d.status === 'pending') continue;
      // one this browser asked for, or one that was running when we looked
      if (!state.rv.watch.has(d.id) && !(state.rv.loaded && was.get(d.id) === 'pending')) continue;
      state.rv.watch.delete(d.id);
      toast(`Dataset #${d.id}: ${docLine(d)}`, { key: `dataset-${d.id}`, ms: 12000 });
    }
    Object.assign(state.rv, { llm, proposals: props, datasets: ds, loaded: true });
    // 11j: a card open in the sheet follows the record it shows
    const open = state.read && state.read.kind === 'proposal' ? state.read : null;
    if (changed && open) {
      const now = props.find(x => String(x.id) === String(open.id));
      const had = (state.readData[readKey(open)] || {}).data;
      if (now && JSON.stringify(now) !== JSON.stringify(had)) readFetch(open);
    }
    if (changed && state.tab === 'review' && !state.model) render();
    // 11k: a Propose row on another page waits for this list to know whether
    // a proposal is already open
    else if (changed && (state.model || state.topic)) render();
    if (state.read && state.read.kind === 'proposal') renderReader();
  } catch (e) { /* server briefly away */ }
  finally { _rvInFlight = false; }
}

async function rvPost(path, body) {
  if ('approver' in body || 'requester' in body) {
    if (!whoName()) { state.rv.msg = askName(); render(); return; }
  }
  const r = await fetch(path, { method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
    body: JSON.stringify(body) }).catch(() => null);
  const j = r ? await r.json().catch(() => ({})) : {};
  state.rv.msg = r && r.ok ? '' : 'Refused. ' + (j.detail || (r ? `the server answered HTTP `
    + r.status : 'the server is unreachable — it may be restarting'));
  const pid = (/proposals\/(\d+)\//.exec(path) || [])[1];
  // what this browser asked for, so the toast lands when the batch finishes
  if (r && r.ok && j.dataset_id) state.rv.watch.add(j.dataset_id);
  // 11m: generating is the end of this card's work. The sheet closes and the
  // Datasets view opens on the row it just made, being written — leaving the
  // card open to say "#9 Waiting for the AI" left the person where they were
  const made = r && r.ok && /generate$/.test(path) ? j.dataset_id : null;
  if (made) {
    state.rv.view = 'datasets';
    markDataset(made);
    await loadReview();
    navigate({ tab: 'review', model: null, topic: null, read: null });
  } else if (r && r.ok && state.read && state.read.kind === 'proposal') {
    // 11j: the card in the sheet is one of these records — reload it too
    readFetch(state.read);
  }
  if (r && r.ok) toast(/approve$/.test(path) ? `Approved the missing skill of proposal #${pid}`
    : /reject$/.test(path) ? 'Rejected' : made
      ? `Dataset #${made} is being written — it lands here when the AI answers` : 'Done',
    { key: 'review' });
  if (made) return;
  await loadReview(); render();
}

// the dataset a click just asked for, marked while someone finds it
function markDataset(did) {
  state.rv.landedDs = did;
  setTimeout(() => { if (state.rv.landedDs === did) { state.rv.landedDs = null;
    if (state.tab === 'review') render(); } }, LANDED_MS);
}

// The name every decision is recorded under. It is remembered for this
// browser and restored at boot, not per tab: typing it on the Loop board and
// finding it gone on the topic page is the same person, one name.
function rememberedName() {
  if (state.rvName) return state.rvName;
  try { state.rvName = localStorage.getItem('bench-name') || ''; } catch (e) { /* private mode */ }
  return state.rvName;
}

// One name, in the header: "masein ▾", click to change. Every action reads
// it; the five per-panel boxes are gone (and Submit's was empty while the
// others were filled). A first visit with no name gets a small prompt up
// there, rather than a refusal after the first click.
const whoName = () => rememberedName().trim();

function setWho(v) {
  state.rvName = v;
  try { localStorage.setItem('bench-name', v); } catch (e) { /* private mode */ }
}

// an action that records a name, when there is none: open the header box
// and say so there, instead of a refusal three panels away
function askName() {
  state.whoAsk = true;
  renderWho(true);
  const btn = document.querySelector('#who [data-pop-anchor="who"]');
  if (btn && POP.key !== 'who') btn.click();
  const i = document.querySelector('#pop-who input, #who input');
  if (i) { i.focus(); i.select(); }
  return 'your name is recorded on this — type it at the top of the page first';
}

function renderWho(force = false) {
  const box = document.getElementById('who');
  if (!box || !LIVE) return;
  // a results refresh redraws the header: never under someone's caret
  if (!force && box.contains(document.activeElement)
      && document.activeElement.tagName === 'INPUT') return;
  const name = whoName();
  // one control in the bar, whatever the state: a button that opens the same
  // panel. It is never clipped, it survives a poll, and Esc gives it back.
  const btn = el('button', { class: 'who barpill' + (name ? '' : ' ask'), 'data-who': name,
    'data-who-prompt': name ? null : '1',
    title: name
      ? 'the name recorded on anything you start, approve or import here — click to change'
      : 'the name recorded on anything you start, approve or import here',
    'aria-label': name ? `your name: ${name} — change` : 'who are you?',
    text: name ? `${name} ▾` : 'Who are you? ▾' });
  box.replaceChildren(popover(btn, () => {
    const input = el('input', { type: 'text', value: name, 'aria-label': 'your name',
      'data-keep': 'whoname',
      placeholder: 'your name', 'data-who-input': '1', autocomplete: 'name',
      onkeydown: e => { if (e.key === 'Enter') { e.preventDefault(); save(); } } });
    const save = () => {
      const v = input.value.trim();
      if (!v) { input.focus(); return; }
      setWho(v); state.whoAsk = false; popClose(); renderWho(true);
    };
    return el('div', { class: 'moremenu whopop' + (state.whoAsk ? ' ask' : ''),
        id: 'pop-who', 'aria-label': 'your name' },
      el('p', { class: 'small', text: 'Recorded on anything you start, approve or import '
        + 'here — this tailnet has no login, so the name you type is the record.' }),
      el('div', { class: 'frm' }, input,
        el('button', { class: 'primary', 'data-who-save': '1', text: 'Save', onclick: save })));
  }, { key: 'who', menu: false, placement: 'bottom-end' }));
}

// kept for the call sites that used to place a box: they place nothing now
function rvNameInput() { return ''; }

// "18 of 20 documents · 2 missing — 1 too short (87 words), 1 reply not JSON".
// Every document asked for is accounted for: held, or missing with its reason.
// A dataset made before 11a recorded no reasons, and says that rather than
// leaving the gap unexplained.
function docLine(d) {
  const pv = d.provenance || {}, it = pv.items || {};
  const kept = it.kept ?? d.kept ?? 0;
  const req = it.requested ?? pv.count_requested ?? d.count ?? kept;
  const missing = Array.isArray(it.missing) ? it.missing : null;
  const gap = missing ? missing.length : Math.max(0, req - kept);
  let line = `${kept} of ${req} documents`;
  if (!gap) return line;
  line += ` · ${gap} missing — `;
  if (!missing) return line + 'reasons not recorded (made before 11a)';
  const by = new Map();
  for (const m of missing) by.set(m.why, (by.get(m.why) || 0) + 1);
  return line + [...by].map(([why, n]) => `${n} ${why}`).join(', ');
}

// the daily limit is per provider (a paid API's, or LOCAL_DAILY_ITEM_CAP):
// null is no limit, and the judge's spend is on the judge's line
const underCap = u => u.daily_cap == null || (u.usage_today || 0) < u.daily_cap;
function usageLine(llm) {
  const rows = (llm.usage || []).length ? llm.usage
    : [{ provider: llm.provider, items: llm.usage_today, cap: llm.daily_cap, roles: ['generator'] }];
  return el('span', { 'data-usage': '1' }, el('b', { text: 'today ' }),
    rows.map(u => `${u.provider}${u.roles && u.roles.length ? ` (${u.roles.join(', ')})` : ''}: `
      + (u.cap == null ? `${u.items} items, no daily limit`
         : `${u.items} of ${u.cap} items` + (u.items >= u.cap ? ' — cap reached until tomorrow' : '')))
      .join(' · '));
}

// ---------------------------------------------------------------------------
// 11j: the Review tab, rebuilt. masein: "If I want to start a new review, or
// want to review the pending ones, the approved ones and the generated
// datasets — I think the UX is bad." Four views with their counts, one
// compact line per row, + New proposal at the top, and a proposal that opens
// as a short card in the reader's sheet — not a 1,300 px block on the page.
// ---------------------------------------------------------------------------
const RV_VIEWS = [['review', 'To review'], ['ready', 'Ready to generate'],
                  ['datasets', 'Datasets'], ['history', 'History']];

function rvLists() {
  const props = state.rv.proposals || [];
  const is = (p, ...s) => s.includes(p.status);
  return { review: props.filter(p => is(p, 'proposed', 'pending')),
           ready: props.filter(p => is(p, 'approved')),
           datasets: state.rv.datasets || [],
           history: props.filter(p => is(p, 'rejected', 'failed')) };
}
// it opens on what is waiting, and on the datasets when nothing is
function rvView() {
  if (RV_VIEWS.some(([k]) => k === state.rv.view)) return state.rv.view;
  return rvLists().review.length ? 'review' : 'datasets';
}

// One badge, not four warning boxes. Its tooltip gives the reasons in plain
// words: the server's own sentences are the record, and they live in Details.
const DEMO_WORDS = [
  [/stub/i, 'the grader is a stand-in, not a judge'],
  [/calibrat|kappa|agreement|checked by a person|no person/i,
   'no person has checked the judge yet'],
  [/provider|more than one step/i, 'the same AI did more than one step'],
  [/local|provisional|pinned/i, 'the judge is a small local AI'],
];
const demoWords = r => (DEMO_WORDS.find(([re]) => re.test(String(r))) || [, String(r)])[1];
function demoReasons(p) {
  const ev = p.evidence || {};
  const raw = [...(((p.override || p.over_provisional_judge) || {}).reasons || []),
               ...(ev.provisional ? [ev.provisional_reason || 'a local judge'] : [])];
  const out = [];
  for (const r of raw) if (!out.includes(demoWords(r))) out.push(demoWords(r));
  return out;
}
function demoBadge(p, key) {
  const why = demoReasons(p);
  if (!why.length) return '';
  return el('span', { class: 'badge taint', 'data-demo-only': String(key ?? p.id),
    title: 'for demos and trials, not for results — ' + why.join('; '), text: 'Demo only' });
}
function dsDemoBadge(d) {
  const pv = d.provenance || {};
  return demoBadge({ id: d.id, override: pv.proposed_over_provisional_judge
      || d.over_provisional_judge,
    evidence: { provisional: pv.provisional, provisional_reason: pv.provisional_reason } },
    'ds' + d.id);
}

const rvStatusWords = s => s === 'proposed' ? 'To review' : s === 'pending' ? 'Waiting for the AI'
  : s === 'approved' ? 'Approved' : s === 'rejected' ? 'Rejected' : s === 'failed' ? 'Failed' : s;
const rvStatusClass = s => s === 'proposed' ? 'queued' : s === 'pending' ? 'running'
  : s === 'approved' ? 'done' : s;
const rvStatusChip = p => el('span', { class: stClass(rvStatusClass(p.status)),
  'data-rv-status': String(p.id), text: rvStatusWords(p.status) });
const modelName = id => (DATA.models.find(x => x.id === id) || {}).name || id || '—';

// "20 of 20", or "18 of 20 · 2 missing" — the reasons are in the reader
function dsDocs(d) {
  const it = (d.provenance || {}).items || {};
  const kept = it.kept ?? d.kept ?? 0;
  const req = it.requested ?? (d.provenance || {}).count_requested ?? d.count ?? kept;
  const gap = Array.isArray(it.missing) ? it.missing.length : Math.max(0, req - kept);
  // "0 of 26 · 26 missing" says one thing twice: none kept is the count alone
  return `${kept} of ${req}` + (gap && kept ? ` · ${gap} missing` : '');
}

const GAP_TIP = 'Pass this to your training run. The run records the dataset, and the '
  + 'checkpoints it makes are marked as trained on this topic\'s practice data.';
function useInTraining(d) {
  return el('button', { class: 'ghost', 'data-ds-flag': String(d.id), title: GAP_TIP,
    text: 'Use in training ⧉', onclick: () => copyText(`--gap-dataset ${d.id}`) });
}

// failed, or turned back by the copy check: either way nothing to read
const dsFailed = d => d.status === 'failed' || d.status === 'rejected' || (d.status !== 'ready' && !!d.error);

// one row for a dataset, here and on the topic page
function dsRow(d) {
  return el('tr', { 'data-ds-row': String(d.id),
      class: state.rv.landedDs === d.id ? 'landed' : null },
    el('td', { class: 'num se', text: '#' + d.id }),
    el('td', { text: d.category || '—' }),
    el('td', { class: 'small', text: modelName(d.model) }),
    // 11m: a count belongs here even when it failed — "0 of 26" — and the
    // failure is the status beside the actions
    el('td', { class: 'small num nowrap', 'data-doc-line': String(d.id),
      text: d.status === 'ready' || dsFailed(d) ? dsDocs(d) : rvStatusWords(d.status) }),
    el('td', { class: 'small se', text: d.requester || '—' }),
    el('td', { class: 'small se num nowrap',
      text: d.created_at ? rel(d.created_at) + ' ago' : '—' }),
    el('td', {}, dsDemoBadge(d)),
    el('td', { class: 'rowacts' }, actCell('ds-' + d.id,
      d.status === 'ready'
        ? [readButton({ kind: 'dataset', id: String(d.id) }, 'Read', { 'data-ds-read': String(d.id) }),
           useInTraining(d)]
        : dsFailed(d)
          ? el('span', { class: 'badge danger', 'data-ds-failed': String(d.id),
              text: d.status === 'rejected' ? 'Rejected' : 'Failed' })
          : el('span', { class: 'small se', text: 'being written' }),
      [{ label: 'Provenance', act: 'provenance',
         run: () => openReader({ kind: 'provenance', id: 'dataset:' + d.id }) },
       d.download ? { label: 'Download', act: 'download',
                      href: `api/datasets/${d.id}/items.jsonl` } : null,
       { label: 'Copy the training flag', act: 'copy-flag',
         run: () => copyText(`--gap-dataset ${d.id}`) },
       { label: 'Copy dataset id', act: 'copy-id', run: () => copyText(String(d.id), '#' + d.id) }])));
}

// 11m: what a failed dataset says. #9 said "the generator returned no
// parseable items", while the reason for each of its 26 missing items was on
// record in its provenance. The line says what happened; Details lists them.
function dsWhy(d) {
  const pv = d.provenance || {}, it = pv.items || {};
  const miss = Array.isArray(it.missing) ? it.missing : [];
  const kept = it.kept ?? 0;
  const req = it.requested ?? pv.count_requested ?? d.count ?? 0;
  const head = `${kept} of ${req} kept — `;
  if (d.status === 'rejected') return head + (d.error || 'the copy check turned it back') + '.';
  // made before question-and-answer items had a register of their own: the
  // generator was told to write prose, and did
  if (d.fmt === 'free' && !/question-and-answer items/.test(pv.audience || '') && miss.length
      && miss.every(m => m.why === 'no question or no answer'))
    return head + 'asked for question-and-answer items, but the generator was given the prose '
      + 'register.';
  if (miss.length) {
    const by = new Map();
    for (const m of miss) by.set(m.why, (by.get(m.why) || 0) + 1);
    return head + [...by].map(([w, n]) => `${n} ${w}`).join(', ') + '.';
  }
  return head + String(d.error || 'nothing came back that could be read').replace(/\.$/, '') + '.';
}

// the row, and for a failed dataset the line under it that explains
function dsRows(d) {
  if (!dsFailed(d)) return [dsRow(d)];
  const miss = Array.isArray(((d.provenance || {}).items || {}).missing)
    ? d.provenance.items.missing : [];
  const open = !!(state.rv.dsWhyOpen || {})[d.id];
  const pg = paged('ds-why-' + d.id, miss, `${d.id}:${miss.length}`, render, 10);
  return [dsRow(d), el('tr', { class: 'dsfail', 'data-ds-why': String(d.id) },
    el('td', { colspan: '8' },
      el('p', { class: 'warn', 'data-ds-why-line': String(d.id), text: dsWhy(d) }),
      miss.length ? el('details', { class: 'small', 'data-ds-details': String(d.id),
          open: open ? '' : null,
          ontoggle: e => { state.rv.dsWhyOpen = { ...(state.rv.dsWhyOpen || {}),
            [d.id]: e.target.open }; } },
        el('summary', { text: `Details — ${miss.length} item${miss.length === 1 ? '' : 's'}, `
          + 'one reason each ▸' }),
        el('ol', { class: 'dswhy', start: String(pg.from || 1) }, pg.rows.map(m =>
          el('li', { 'data-ds-missing': String(m.request ?? '') },
            `request ${m.request ?? '—'} · ${m.focus || 'no area'} · ${m.why}`))),
        pg.pager || '') : ''))];
}

// a proposal row opens the card in the sheet — the whole row, and the link in
// it for the keyboard
function rvRow(p) {
  const key = readStr({ kind: 'proposal', id: String(p.id) });
  return el('tr', { 'data-rv-row': String(p.id),
      class: 'clickrow' + (state.rv.landed === p.id ? ' landed' : ''),
      onclick: e => { if (e.target.closest('a, button')) return;
        openReader({ kind: 'proposal', id: String(p.id) }, `[data-read-open="${CSS.escape(key)}"]`); } },
    el('td', {}, readLink({ kind: 'proposal', id: String(p.id) }, p.category || p.task,
      { 'data-rv-open': String(p.id) })),
    el('td', { class: 'small', text: modelName(p.model) }),
    el('td', { class: 'small' }, rvStatusChip(p)),
    el('td', { class: 'small se', text: p.requested_by || '—' }),
    el('td', { class: 'small se num nowrap',
      text: p.created_at ? rel(p.created_at) + ' ago' : '—' }),
    el('td', {}, demoBadge(p)));
}

function rvTable(cols, rows, emptyNode) {
  if (!rows.length) return emptyNode;
  return el('div', { class: 'lb-wrap' }, el('table', { class: 'jd rvlist' },
    el('thead', {}, el('tr', {}, cols.map(c => el('th',
      { class: /^(when|documents)$/.test(c) ? 'num' : null, text: c })))),
    el('tbody', {}, rows)));
}

// the focus plan as chips: the first 8 areas, then "+12 more"
function focusChips(p, n, box, done) {
  const key = `${p.id}:${p.status}:${n}`;
  const paint = f => {
    const labels = (f.labels || []).slice(0, n);
    box.dataset.focusMode = f.mode || 'off';
    if (!labels.length) {
      box.replaceChildren(el('span', { class: 'small se', text: f.legacy
        ? 'Approved before plans were shown.'
        : f.reason ? 'Not spread — ' + f.reason + '.' : 'Not spread over areas.' }));
      if (done) done(f);
      return;
    }
    const counts = new Map();
    for (const l of labels) counts.set(l, (counts.get(l) || 0) + 1);
    const all = [...counts];
    box.replaceChildren(
      ...all.slice(0, 8).map(([l, k]) => el('span', { class: 'chip-static', text: `${l} ${k}` })),
      all.length > 8 ? el('span', { class: 'small se', 'data-focus-more': String(all.length - 8),
        title: all.slice(8).map(([l, k]) => `${l} ${k}`).join(' · '),
        text: `+${all.length - 8} more` }) : '');
    if (done) done(f);
  };
  if (state.rv.focus[key]) { paint(state.rv.focus[key]); return; }
  box.replaceChildren(el('span', { class: 'small se', text: 'loading…' }));
  api(`api/proposals/${p.id}/focus?count=${n}`)
    .then(f => { state.rv.focus[key] = f; paint(f); })
    .catch(() => box.replaceChildren(el('span', { class: 'small se',
      text: 'the plan could not be loaded' })));
}

// ---- the practice answers the AI read, for the reviewer ---------------------
state.rv.answers = {};
async function loadRvAnswers(pid) {
  if (state.rv.answers[pid] && !state.rv.answers[pid].error) return;
  state.rv.answers[pid] = { loading: true };
  try { state.rv.answers[pid] = await api(`api/proposals/${pid}/answers`); }
  catch (e) { state.rv.answers[pid] = { error: (e && e.message) || String(e) }; }
  renderReader();
}

function rvAnswersBlock(p, ev, redraw) {
  const open = state.rv.answersOpen === p.id;
  const det = el('details', { class: 'rd-answers', 'data-answers-read': String(p.id),
    open: open ? '' : null },
    el('summary', { text: `The answers it read (${ev.n_shown ?? '—'}) ▸`,
      onclick: () => { state.rv.answersOpen = open ? 0 : p.id;
        if (!open) loadRvAnswers(p.id); } }));
  if (!open) return det;
  det.append(el('p', { class: 'small se', 'data-answers-note': '1',
    text: 'The AI that wrote the missing skill saw only the judge\'s comments, with the '
      + 'question wording taken out, so no exam wording can reach the training data. You see '
      + 'the questions so you can check its reading. Practice questions only.' }));
  const a = state.rv.answers[p.id];
  if (!a || a.loading) { det.append(skeleton(3, { 'data-loading': 'answers-read' })); return det; }
  if (a.error) { det.append(el('p', { class: 'warn', text: a.error })); return det; }
  // 11k: 33 answers in full was one 14,000px scroll. Ten at a time, each
  // answer clamped to three lines until it is asked for — the question and
  // the judge's comment stay whole
  const items = a.items || [];
  const pg = paged('rv-answers', items, `${p.id}:${items.length}`, redraw || render, 10);
  det.append(el('p', { class: 'small se', 'data-answers-count': String(a.n),
    text: `${a.n} practice answers` + (a.gone ? `, ${a.gone} no longer on file` : '')
      + (items.length > pg.rows.length ? ` · ${pg.from}–${pg.to} below` : '') + '.' }));
  det.append(el('ol', { class: 'rd-alist', start: String(pg.from || 1) }, pg.rows.map(it => {
    const shown = !!(state.rv.answersShown || {})[it.qid];
    const text = it.answer || '(the model wrote nothing)';
    const long = text.length > 200;
    return el('li', { 'data-answer-qid': String(it.qid) },
      el('p', { class: 'rd-q' }, el('b', { text: 'The question. ' }), it.question || '—'),
      el('div', { class: 'rd-a' }, el('b', { text: 'Its answer. ' }),
        el('div', { class: shown || !long ? 'rd-atext' : 'rd-atext clamp3',
          'data-answer-text': String(it.qid), text }),
        long ? el('button', { class: 'quiet rd-more', 'data-answer-toggle': String(it.qid),
          'aria-expanded': String(shown), text: shown ? 'Show less' : 'Show the whole answer',
          onclick: () => { state.rv.answersShown = { ...(state.rv.answersShown || {}),
            [it.qid]: !shown }; (redraw || render)(); } }) : ''),
      el('p', { class: 'small' }, el('b', { text: `Scored ${it.score} of 4. ` }), it.comment));
  })));
  if (pg.pager) det.append(pg.pager);
  return det;
}

// ---- the card in the sheet --------------------------------------------------
function readProposal(wrap, r, p) {
  const paint = () => {
    const ev = p.evidence || {};
    const llm = state.rv.llm || {};
    const llmOk = !!llm.configured && underCap(llm);
    wrap._title.textContent = `${p.category || p.task} · ${modelName(p.model)}`;
    wrap._src.replaceChildren(el('span', { class: 'se', text: `#${p.id}` }), ' ',
      rvStatusChip(p), ' ', demoBadge(p), ' ',
      el('span', { class: 'se', text: (p.requested_by ? `asked by ${p.requested_by}` : '')
        + (p.created_at ? ` · ${rel(p.created_at)} ago` : '') }));
    readActs(wrap, { copy: () => p.edited_text || p.spec_text || '' });
    const body = [];
    if (p.status === 'pending')
      body.push(el('p', { class: 'small', text: 'Waiting for the AI to answer. That takes '
        + 'minutes to hours; this page checks by itself.' }));
    if (p.error) body.push(el('p', { class: 'warn', text: p.error }));
    // 1. what is missing, editable while it is to review
    const ta = el('textarea', { class: 'rd-spec', 'data-spec-edit': String(p.id),
      'aria-label': 'the missing skill' });
    ta.value = p.spec_text || '';
    if (p.spec_text || p.edited_text) {
      body.push(el('div', { class: 'dxh', text: 'What\'s missing' }));
      body.push(p.status === 'proposed' ? ta
        : el('p', { class: 'spec', 'data-spec': String(p.id),
            text: p.edited_text || p.spec_text }));
      if (p.edited_text && p.status !== 'proposed')
        body.push(el('p', { class: 'small se', text: 'Approved as edited.' }));
    }
    // 2. why, in one line — this is the count that used to be blank
    body.push(el('div', { class: 'dxh', text: 'Why' }));
    body.push(el('p', { class: 'small', 'data-why-line': String(p.id),
      text: `${ev.diagnose_weak ?? '—'} of ${ev.diagnose_items ?? '—'} practice answers scored `
        + `below 3 of 4 · ${p.category} score `
        + (ev.topic_score_report != null ? `${num(ev.topic_score_report, 2)} / 4` : '—')
        + ` (hidden questions)` }));
    // 3. the answers it read
    body.push(rvAnswersBlock(p, ev, paint));
    // 4. where the documents go
    const chips = el('div', { class: 'focuschips', 'data-focus-plan': String(p.id),
      'data-plan-stage': p.status === 'proposed' ? 'decide' : 'generate' });
    const spreadBox = el('input', { type: 'checkbox', 'data-spread': String(p.id),
      checked: state.rv.spread[p.id] !== false ? '' : null,
      onchange: e => { state.rv.spread[p.id] = e.target.checked; } });
    const spreadLabel = el('label', { class: 'small spread' }, spreadBox,
      ' Spread the documents over these');
    const count = el('input', { type: 'number', value: String(state.rv.count[p.id] || 20),
      min: '1', max: '1000', style: 'width:80px', 'aria-label': 'how many documents',
      oninput: e => { const n = +e.target.value;
        if (n >= 1 && n <= 1000) { state.rv.count[p.id] = n; focusChips(p, n, chips); } } });
    if (p.status === 'proposed' || p.status === 'approved') {
      body.push(el('div', { class: 'dxh', text: 'Documents will cover' }));
      body.push(chips);
      if (p.status === 'proposed') body.push(spreadLabel);
      focusChips(p, p.status === 'proposed' ? 20 : (state.rv.count[p.id] || 20), chips, f => {
        const none = !(f.labels || []).length;
        spreadBox.disabled = none;
        spreadLabel.classList.toggle('dim', none);
      });
    }
    // 5. what this status can do
    if (p.status === 'proposed') {
      const reason = el('input', { type: 'text', placeholder: 'why reject it?',
        style: 'flex:1;min-width:160px', 'aria-label': 'reject reason',
        'data-reject-reason': String(p.id), 'data-keep': 'reject-' + p.id });
      body.push(el('div', { class: 'frm rd-acts-row' },
        el('button', { class: 'primary', 'data-approve': String(p.id), text: 'Approve',
          onclick: () => rvPost(`api/proposals/${p.id}/approve`, { approver: whoName(),
            edited_text: ta.value,
            spread: state.rv.spread[p.id] !== false && !spreadBox.disabled }) }),
        el('button', { class: 'ghost', 'data-reject-open': String(p.id), text: 'Reject…',
          onclick: () => { state.rv.rejecting = { ...(state.rv.rejecting || {}), [p.id]: true };
            paint();
            const i = wrap._body.querySelector(`[data-reject-reason="${p.id}"]`);
            if (i) i.focus(); } })));
      if ((state.rv.rejecting || {})[p.id]) {
        const go = el('button', { class: 'danger', 'data-reject': String(p.id), text: 'Reject',
          disabled: '', onclick: () => rvPost(`api/proposals/${p.id}/reject`,
            { approver: whoName(), reason: reason.value }) });
        reason.addEventListener('input', () => { go.disabled = !reason.value.trim(); });
        body.push(el('div', { class: 'frm', 'data-reject-form': String(p.id) }, reason, go,
          el('button', { class: 'ghost', text: 'Cancel',
            onclick: () => { state.rv.rejecting[p.id] = false; paint(); } })));
      }
    }
    if (p.status === 'approved') {
      const fmt = Select('format', [['doc', 'Documents'], ['free', 'Q&A (for comparison)']],
        state.rv.fmt[p.id] || 'doc', v => { state.rv.fmt[p.id] = v; },
        { key: 'rv-fmt-' + p.id });
      body.push(el('div', { class: 'frm rd-acts-row' }, count, fmt,
        el('button', { class: 'primary', 'data-generate': String(p.id), text: 'Generate',
          disabled: llmOk ? null : '',
          title: llmOk ? '' : (llm.reason || 'no AI is set up here to write documents'),
          onclick: () => rvPost(`api/proposals/${p.id}/generate`, { requester: whoName(),
            count: +count.value, fmt: state.rv.fmt[p.id] || 'doc' }) }),
        llmOk ? '' : el('span', { class: 'propwhy', 'data-why': 'generate',
          text: llm.reason || 'no AI is set up here to write documents' })));
    }
    if (p.status === 'rejected' && p.reject_reason)
      body.push(el('p', { class: 'small', text: 'Rejected: ' + p.reject_reason }));
    // 11m: a refusal is said here, not only on the tab behind the sheet
    if (state.rv.msg)
      body.push(el('p', { class: 'warn', 'data-rv-msg': String(p.id), text: state.rv.msg }));
    // 6. the datasets made from it
    if ((p.datasets || []).length) {
      body.push(el('div', { class: 'dxh', text: 'Datasets from this proposal' }));
      body.push(el('p', { class: 'small', 'data-rv-datasets': String(p.id) },
        p.datasets.flatMap((d, i) => [i ? ' · ' : '',
          d.status === 'ready' ? readLink({ kind: 'dataset', id: String(d.id) }, '#' + d.id)
            : el('span', { class: 'se', text: `#${d.id} ${rvStatusWords(d.status)}` })])));
    }
    // 7. everything a record needs and a reviewer does not
    const usage = state.rv.llm || {};
    const rows = [
      ['asked by', (p.requested_by || '—') + (p.created_at ? ` · ${absT(p.created_at)}` : '')],
      [p.status === 'rejected' ? 'rejected by' : 'approved by',
        p.approver ? p.approver + (p.decided_at ? ` · ${absT(p.decided_at)}` : '') : '—'],
      ['proposed by', p.proposer || '—'],
      ['prompt fingerprint', p.prompt_sha ? String(p.prompt_sha).slice(0, 12) : '—'],
      ['judge', (p.evidence || {}).judge_id || '—'],
      ['the answers behind it', `${ev.n_shown ?? '—'} of ${ev.diagnose_weak ?? '—'} practice `
        + `answers that scored below 3 of 4, of ${ev.diagnose_items ?? '—'} in the topic`],
      ['the copy check', 'Every document a generator writes is checked against every exam and '
        + 'benchmark question, both halves: no document may copy 13 words in a row from one.'],
      ['today', (usage.usage || []).map(u => `${u.provider}: ` + (u.cap == null
        ? `${u.items} AI requests, no daily limit` : `${u.items} of ${u.cap} AI requests`))
        .join(' · ') || '—'],
      ['the full reasons', [...(((p.override || {}).reasons) || []),
        ...(ev.provisional ? [`${ev.provisional_reason}: ${ev.served_model} at ${ev.base_url}`]
          : [])].join('; ') || 'none recorded'],
      ['task', p.task],
    ];
    // 11m: a repaint keeps what the reader opened — the state says whether
    // it is open, never the DOM that is about to be replaced
    body.push(el('details', { class: 'rd-det', 'data-rv-details': String(p.id),
        open: (state.rv.detailsOpen || {})[p.id] ? '' : null,
        ontoggle: e => { state.rv.detailsOpen = { ...(state.rv.detailsOpen || {}),
          [p.id]: e.target.open }; } },
      el('summary', { text: 'Details ▸' }),
      el('dl', { class: 'provlist small' }, rows.flatMap(([k, v]) =>
        [el('dt', { text: k }), el('dd', { text: String(v) })]))));
    wrap._body.replaceChildren(...body);
  };
  wrap._update = () => paint();
  paint();
}

// ---- + New proposal ---------------------------------------------------------
// One dialog, from the tab itself or from a topic page, with the model and
// topic filled in. It asks the server's own gate about every topic, so a
// blocked one says why instead of failing after the click.
function npTopics(mid) {
  const J = DATA.judged || {};
  const m = DATA.models.find(x => x.id === mid);
  const open = (state.rv.proposals || []).filter(p =>
    ['proposed', 'pending', 'approved'].includes(p.status) && p.model === mid);
  return (J.exam || []).map(task => {
    const jt = m && m.judge && (m.judge.tasks || {})[task];
    const score = jt ? pubScore(jt) : null;
    const g = (jt || {}).propose;
    const has = open.find(p => p.task === task);
    let why = '';
    if (!jt || score == null) why = 'not sat yet';
    else if (has) why = `proposal #${has.id} is open`;
    else if ((jt.n_report ?? jt.n ?? 0) < CAT_MIN_N)
      why = `under ${CAT_MIN_N} hidden questions — its score is noise`;
    else if (g && (g.hard || []).length) why = g.hard[0].short || g.why;
    return { task, topic: frName(task), score, why, soft: (g && g.soft) || [],
             overridable: !!(g && g.overridable) };
  });
}

function npDialog(pre = {}) {
  const st = { model: pre.model || '', topic: pre.topic || '', ack: false };
  const back = el('div', { class: 'dlg-back', 'data-dialog': 'propose' });
  const err = el('p', { class: 'warn', hidden: '', 'data-dialog-error': '1' });
  const name = el('input', { type: 'text', value: rememberedName(), 'aria-label': 'your name',
    placeholder: 'your name (recorded)' });
  const ack = el('input', { type: 'checkbox', id: 'dlg-ack', 'data-dialog-ack': '1' });
  const ackRow = el('label', { class: 'dlg-check', for: 'dlg-ack' }, ack,
    ' I understand these grades are not evidence');
  const demoNote = el('p', { class: 'small', 'data-dialog-demo': '1' });
  const go = el('button', { class: 'primary', 'data-dialog-go': '1', text: 'Propose' });
  const cancel = el('button', { 'data-dialog-cancel': '1', text: 'Cancel' });
  const topicBox = el('div', { class: 'exareas nptopics', 'data-np-topics': '1' });
  const judged = DATA.models.filter(m => m.judge && Object.keys(m.judge.tasks || {})
    .some(t => t.startsWith('exam_')));
  const sync = () => {
    const t = st.model ? npTopics(st.model).find(x => x.topic === st.topic) : null;
    const soft = t && !t.why && t.soft.length ? t.soft : [];
    demoNote.replaceChildren(soft.length
      ? el('span', { 'data-dialog-reasons': '1' }, el('b', { text: 'Demo only. ' }),
        'These grades are not evidence yet — '
        + demoReasons({ override: { reasons: soft } }).join('; ')
        + '. Anything made from this proposal carries the same mark: the missing skill, the '
        + 'dataset, and any model trained on it. For demos and trials, not for results.')
      : '');
    ackRow.hidden = !soft.length;
    go.disabled = !(st.model && st.topic && name.value.trim() && (!soft.length || ack.checked));
  };
  const paintTopics = () => {
    const rows = npTopics(st.model);
    const byName = Object.fromEntries(rows.map(x => [x.topic, x]));
    if (st.topic && (!byName[st.topic] || byName[st.topic].why)) st.topic = '';
    const areas = Object.entries(DATA.meta.areas || {}).map(([a, names]) =>
      [a, names.map(n => byName[n]).filter(Boolean)
        .sort((x, y) => (x.score ?? 9) - (y.score ?? 9) || x.topic.localeCompare(y.topic))])
      .filter(([, xs]) => xs.length);
    topicBox.replaceChildren(...areas.map(([a, xs]) => el('div', { class: 'exarea',
        role: 'group', 'aria-label': a },
      el('div', { class: 'exhead' }, el('span', { text: a })),
      xs.map(x => el('label', { class: 'extopic' + (x.why ? ' off' : ''),
          'data-np-topic': x.topic, 'data-np-why': x.why || null },
        el('input', { type: 'radio', name: 'np-topic', disabled: x.why ? '' : null,
          checked: st.topic === x.topic ? '' : null,
          onchange: () => { st.topic = x.topic; sync(); } }),
        el('span', { class: 'exname', title: x.topic, text: x.topic }),
        el('span', { class: 'exst', text: x.score != null ? `${num(x.score, 2)} / 4` : '' }),
        // the name keeps its line; the reason goes under it
        x.why ? el('span', { class: 'exnote', text: x.why }) : '')))));
    // a topic filled in from a topic page is in view, not scrolled past
    const on = topicBox.querySelector('input:checked');
    if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest' });
    sync();
  };
  const models = Combobox('model', modelGroups(judged.map(m => ({ id: m.id }))), st.model,
    v => { st.model = v; paintTopics(); }, { key: 'np-model', placeholder: 'find a model' });
  const box = el('div', { class: 'dlg npdlg', role: 'dialog', 'aria-modal': 'true',
      'aria-labelledby': 'dlg-title' },
    el('h2', { id: 'dlg-title', text: 'New proposal' }),
    el('p', { class: 'small', text: 'The AI reads what the judge wrote about this model\'s '
      + 'practice answers on the topic, and names the skill that is missing. It never sees a '
      + 'question.' }),
    el('label', { class: 'dlg-field' }, el('span', { class: 'small', text: 'model' }), models),
    el('div', { class: 'dlg-field' }, el('span', { class: 'small', text: 'topic' }), topicBox),
    demoNote, ackRow,
    el('label', { class: 'dlg-field' }, el('span', { class: 'small', text: 'your name' }), name),
    err,
    el('div', { class: 'dlg-actions' }, cancel, go));
  back.append(box);
  const close = () => {
    back.remove();
    document.removeEventListener('keydown', onKey, true);
    const again = pre.returnTo && document.querySelector(pre.returnTo);
    if (again) again.focus();
  };
  const onKey = e => {
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab' || POP.panel) return;
    const f = [...box.querySelectorAll('input:not([disabled]), button:not([disabled]), textarea')]
      .filter(x => x.offsetParent);
    if (!f.length) return;
    const i = f.indexOf(document.activeElement);
    if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
    else if (!e.shiftKey && (i === f.length - 1 || i < 0)) { e.preventDefault(); f[0].focus(); }
  };
  name.addEventListener('input', () => { setWho(name.value.trim()); renderWho(); sync(); });
  ack.addEventListener('change', sync);
  cancel.addEventListener('click', close);
  back.addEventListener('mousedown', e => { if (e.target === back) close(); });
  go.addEventListener('click', async () => {
    const t = npTopics(st.model).find(x => x.topic === st.topic) || {};
    go.disabled = true; go.textContent = 'Proposing…'; err.hidden = true;
    try {
      const j = await post('api/proposals', { model: st.model, topic: st.topic,
        requested_by: name.value.trim(),
        ...(t.soft && t.soft.length ? { override_preliminary: true } : {}) });
      close();
      state.rv.loaded = false;
      toast(`Proposal #${j.id} requested — ${st.topic}`
        + (t.soft && t.soft.length ? ' · over a provisional judge' : ''),
        { key: 'propose', go: goReview, link: 'Review' });
      await loadReview();
      // from a topic page, the person stays on it; from the tab, the new
      // proposal is in To review
      if (pre.stay) { state.loop.loaded = false; loadLoop(); render(); return; }
      state.rv.view = 'review';
      if (state.tab === 'review') render(); else navigate({ tab: 'review', topic: null, model: null });
    } catch (e) {
      err.hidden = false;
      err.replaceChildren(el('b', { text: 'Refused. ' }), String((e && e.message) || e));
      go.textContent = 'Propose'; sync();
    }
  });
  document.addEventListener('keydown', onKey, true);
  document.body.append(back);
  if (st.model) paintTopics(); else sync();
  (st.model && st.topic ? (name.value.trim() ? go : name) : models).focus();
}

// ---- the tab ----------------------------------------------------------------
function vReview() {
  if (!state.rv.loaded && netReady()) loadReview();
  rememberedName();
  const llm = state.rv.llm || {};
  const L = rvLists();
  const view = rvView();
  const howto = el('details', { class: 'howto small', 'data-how-review': '1' },
    el('summary', { text: 'How this works ▸' }),
    el('p', { class: 'small', text: 'A proposal reads what the judge wrote about one model\'s '
      + 'practice answers on a weak topic — never the questions — and names the skill that is '
      + 'missing. You approve, edit or reject that sentence, and only the approved words reach '
      + 'the AI that writes documents. Every document is then checked against every exam and '
      + 'benchmark question: none may copy 13 words in a row. Every decision is recorded under '
      + 'your name.' }),
    el('p', { class: 'small' }, el('b', { text: 'the AI ' }), llm.configured
      ? `${llm.provider}/${llm.model || '—'}` : 'not set up here', ' · ',
      llm.configured ? usageLine(llm) : '',
      llm.datasets_quota_bytes ? ` · storage ${(llm.datasets_bytes / 1e6).toFixed(1)} MB of `
        + `${(llm.datasets_quota_bytes / 1e9).toFixed(0)} GB` : ''));
  const head = el('div', { class: 'card', 'data-review-head': '1' },
    el('div', { class: 'rvbar' },
      el('div', {}, el('h2', { text: 'Review' }),
        el('p', { class: 'sub', text: 'What the AI says each model is missing, and the data '
          + 'made from it.' })),
      el('button', { class: 'primary', 'data-new-proposal': '1', text: '+ New proposal',
        onclick: () => npDialog({ returnTo: '[data-new-proposal]' }) })),
    howto,
    el('div', { class: 'rvviews', role: 'tablist', 'aria-label': 'review views' },
      RV_VIEWS.map(([k, label]) => el('button', { class: 'chip-btn' + (k === view ? ' on' : ''),
        role: 'tab', 'data-rv-view': k, 'aria-selected': String(k === view),
        text: `${label} (${L[k].length})`,
        onclick: () => { state.rv.view = k; navigate({ tab: 'review' }); } }))),
    !llm.configured && llm.reason ? el('p', { class: 'warn', text: llm.reason }) : '',
    state.rv.msg ? el('p', { class: 'small', text: state.rv.msg }) : '');
  const PCOLS = ['topic', 'model', 'status', 'asked by', 'when', ''];
  const body = el('div', { class: 'card', 'data-rv-list': view });
  if (view === 'datasets') {
    body.append(el('h2', { text: 'Datasets' }),
      el('p', { class: 'sub', text: 'What the AI wrote, after the copy check. Read one here, '
        + 'or hand it to a training run.' }),
      rvTable(['#', 'topic', 'model', 'documents', 'made by', 'when', '', ''],
        L.datasets.flatMap(dsRows),
        empty('No datasets yet. Approve a proposal, then generate from it.')));
  } else {
    const sub = { review: 'Proposals waiting for a person. Open one to read it.',
      ready: 'Approved. Generate the documents when you are ready.',
      history: 'Rejected and failed proposals.' }[view];
    body.append(el('h2', { text: RV_VIEWS.find(([k]) => k === view)[1] }),
      el('p', { class: 'sub', text: sub }),
      rvTable(PCOLS, L[view].map(rvRow),
        empty(view === 'review' ? 'Nothing waiting. Start one with + New proposal.'
          : view === 'ready' ? 'Nothing approved yet. Approve one in To review.'
          : 'Nothing rejected or failed.',
          view === 'review' ? '+ New proposal' : '',
          view === 'review' ? () => npDialog({ returnTo: '[data-new-proposal]' }) : null)));
  }
  return [head, body];
}

// The judged card's answers section on a model page: pick one of this model's
// judged topics and read the diagnosis-half answers. Everything below the
// picker is the topic page's own panel, unchanged — one implementation of the
// rule that the report half is never a row.
function modelAnswers(m, cats) {
  const a = state.ans;
  const topics = cats.map(t => frName(t));
  if (!topics.length) return '';
  const topic = topics.includes(a.topic) ? a.topic : topics[0];
  const wrap = el('div', { 'data-panel': 'model-answers' },
    el('div', { class: 'dxh', text: 'The answers, topic by topic' }),
    el('div', { class: 'frm' },
      Combobox('answers topic', topicGroups(topics, t => {
          const x = Object.entries((m.judge || {}).tasks || {}).find(([k]) => frName(k) === t);
          return x ? pubScore(x[1]) : null; }, !judgedOkM(m)), topic,
        v => { state.ans.topic = v; state.ans.rows = null; render(); },
        { key: 'answers-topic', placeholder: 'find a topic' }),
      el('a', { class: 'small', href: '#topic=' + slugOfTopic(topic),
        text: 'this topic\u2019s page in the loop',
        onclick: e => { e.preventDefault(); navigate({ topic: slugOfTopic(topic), model: null }); } })));
  if (netReady() && !a.loading && (!a.rows || a.model !== m.id || a.topic !== topic))
    loadAnswers(m.id, topic);
  const j = a.rows;
  if (!j || a.model !== m.id || a.topic !== topic) {
    wrap.append(skeleton(4, { 'data-loading': 'answers' }));
    return wrap;
  }
  const rep = j.report_half || {};
  wrap.append(el('p', { class: 'note', 'data-report-half': '1' },
    el('b', { text: 'The hidden questions. ' }),
    // 11l: a topic that was not scored says so, instead of "mean — / 4"
    j.no_score ? `${rep.n ?? 0} questions, not scored: ${j.no_score}`
      : `${rep.n ?? 0} questions, mean ${num(rep.mean, 2)} / 4`
        + (j.no_answer ? ` · ${j.no_answer} answers never finished, left out` : '')
    + (Object.entries(rep.flags || {}).filter(([, n]) => n).length
       ? ' · ' + Object.entries(rep.flags).filter(([, n]) => n)
           .map(([fid, n]) => `${n} ${fid.replace(/_/g, ' ')}`).join(', ') : '')
    + '. That is the published score, and this line is all of it you will see here.'));
  const rows = ansVisible(j);
  wrap.append(ansFilters(j));
  wrap.append(el('p', { class: 'small', 'data-answer-count': String(rows.length),
    text: `${rows.length} of ${j.n_diagnose} practice answers` }));
  const pg = paged('answers', rows,
    JSON.stringify([a.model, a.topic, a.acuity, a.flag, a.score, a.crit, a.sort]));
  wrap.append(pg.pager);
  wrap.append(el('div', { class: 'anslist', 'data-answers-table': '1' },
    pg.rows.map(it => ansCard(it, j))));
  if (pg.pager) wrap.append(pager('answers', rows.length));
  return wrap;
}

// "judging 40/130", not "judging 130 answers": the provider reports how many
// of the batch are back, the poller records it, and this is where a person
// waiting on a judged run finds out whether to wait or come back later.
function judgeCount(j) {
  const m = /^(\d+)\s*\/\s*(\d+)/.exec(j.progress || '');
  if (m) return `${m[1]}/${m[2]}` + (/failed/.test(j.progress) ? ' · some failed' : '');
  return `${j.n_items} answers`;
}

// ---------------------------------------------------------------------------
// The Loop tab: one row per topic, and for each the single next step. Nothing
// here computes a gate — the server says whether a topic may be sat or
// proposed from, in the same words the API refuses with, and this table shows
// that answer. The person should never have to know which tab is next.
// ---------------------------------------------------------------------------

let _loopInFlight = false;
let _loopSig = '';
async function loadLoop() {
  if (_loopInFlight) return;      // a 5 s poll must not stack requests
  _loopInFlight = true;
  try {
    const j = await api('api/loop?model=' + encodeURIComponent(state.loop.model || ''));
    // re-render only when the board actually moved: rebuilding the view every
    // five seconds detaches whatever the person is reaching for
    const sig = JSON.stringify([j.topics, j.judged_blocked, j.tasks_built, j.model, j.models,
                                (j.judge_health || {}).ok, j.pace]);
    const changed = sig !== _loopSig || !state.loop.loaded || state.loop.failed;
    _loopSig = sig;
    Object.assign(state.loop, { rows: j.topics, blocked: j.judged_blocked,
                                built: j.tasks_built || [], floor: j.floor, loaded: true,
                                model: j.model || '', models: j.models || [], failed: '',
                                items: j.built_items || {}, pace: j.pace || null });
    if (j.judge_health) setJudgeHealth(j.judge_health);
    // the Queue tab reads it too: whether the judged suite is on, and its topics
    if (changed && (state.tab === 'loop' || state.tab === 'queue' || state.topic)
        && !state.model) render();
    else if (changed && state.model && state.msitRedraw) state.msitRedraw();
  } catch (e) {
    // netFail already put the banner up and set the backoff; the board itself
    // must also stop saying "Loading…" forever, which is what it did
    state.loop.failed = String((e && e.message) || e);
    if ((state.tab === 'loop' || state.topic) && !state.model) render();
  } finally { _loopInFlight = false; }
}

// Is the grading model answering? Asked of the service (which asks the judge
// at most every thirty seconds) on every poll, on every tab: a judged run
// queued against a judge that is down spends GPU on answers nobody grades.
let _judgeSig = null;
function setJudgeHealth(h) {
  if (!h || typeof h.ok !== 'boolean') return;        // an answer that says nothing
  const sig = JSON.stringify([h.ok, h.why]);
  state.judgeHealth = h;
  if (sig === _judgeSig) return;
  _judgeSig = sig;
  renderFresh();
  if ((state.tab === 'loop' || state.tab === 'queue' || state.topic) && !state.model) render();
  else if (state.model && state.msitRedraw) state.msitRedraw();
}
async function loadJudgeHealth() {
  try { setJudgeHealth(await api('api/judge/health')); } catch (e) { /* the banner says it */ }
}
const judgeDown = () => !!(state.judgeHealth && state.judgeHealth.ok === false);
const judgeWhy = () => { const w = (state.judgeHealth || {}).why || 'the grading model is not answering';
  return w[0].toUpperCase() + w.slice(1) + '. Start it, then queue the run — nothing is queued '
    + 'until it answers.'; };
function judgeChip() {
  if (!judgeDown()) return '';
  return el('span', { class: 'badge danger', 'data-judge-offline': '1', title: judgeWhy(),
    text: 'judge offline' });
}
function judgeOfflineLine() {
  if (!judgeDown()) return '';
  return el('p', { class: 'warn', 'data-judge-offline-why': '1' }, judgeChip(), ' ', judgeWhy());
}

// the same words the network banner uses, inside the card that has no data:
// what failed, what it said, and that it will try again on its own
function loopFailure() {
  if (!state.loop.failed) return '';
  const secs = Math.max(1, Math.round((NET.nextAt - Date.now()) / 1000));
  return el('p', { class: 'warn', 'data-loop-failed': '1' },
    el('b', { text: 'Not reaching the service. ' }),
    `api/loop — ${state.loop.failed}. `
    + (NET.nextAt > Date.now() ? `Retrying in ${secs}s` : 'Retrying')
    + `, backing off to ${NET.MAX / 1000}s. `
    + (state.loop.rows ? 'The board below is from the last good load.'
                       : 'Nothing has loaded yet.'));
}

function loopRowOf(slug) {
  return (state.loop.rows || []).find(r => r.slug === slug) || null;
}

// where a step goes, with its context: the import panel open on this topic
// with the file picker focused; the sit panel with this topic ticked and the
// caret in the model box; the answers; the proposal on Review.
function loopGo(r, step) {
  if (step === 'import') {
    const imp = eximpState();
    if (imp.topic !== r.topic) { imp.topic = r.topic; imp.preview = null; }
    state.after = { scroll: '[data-panel="import"]', focus: '[data-panel="import"] input[type=file]' };
    return navigate({ tab: 'exam', topic: null, model: null });
  }
  if (step === 'review' || step === 'generate') {
    // 11j: to the proposal itself, open in the sheet — not the top of a tab
    // where the reader had to find it among every other proposal
    state.rv.loaded = false;
    const pid = r.proposal && r.proposal.id;
    state.rv.view = step === 'generate' ? 'ready' : 'review';
    return navigate({ tab: 'review', topic: null, model: null,
                      read: pid ? { kind: 'proposal', id: String(pid) } : null });
  }
  if (step === 'sit') {
    state.loopSit.tasks = [r.task]; state.loopSit.page = r.task;
    state.after = { scroll: '[data-panel="sit"]', focus: '[data-ms="sit"] input' };
  } else if (step === 'read') {
    state.after = { scroll: '[data-panel="answers"]' };
  } else if (step === 'hand') {
    state.after = { scroll: '[data-panel="output"]' };
  }
  if (state.loop.model) { state.ans.model = state.loop.model; state.ans.rows = null; }
  return navigate({ topic: r.slug, model: null });
}

// the next step is the server's, and the same for everyone who looks: it
// used to turn on a per-browser "read" flag, so two people saw two steps
const loopStep = r => ({ ...r.next, short: r.next.why });

// the board's button for the next step. Proposing is not done from here: it
// opens the topic page, where the answers are and the one Propose control is
function loopBtn(r) {
  const st = loopStep(r);
  const b = el('button', { 'data-step': st.step, disabled: st.ok ? null : '',
    class: st.ok ? (st.label.endsWith('…') ? 'secondary dot-warn' : 'primary') : null,
    title: st.ok ? '' : st.why, text: st.label,
    onclick: () => loopGo(r, st.step) });
  // 11f: the one action cell — the next step, and the rest behind ⋯
  const items = [
    { label: 'Open the topic', act: 'topic', run: () => loopGo(r, 'topic') },
    r.last_judged ? { label: 'Read the results', act: 'read', run: () => loopGo(r, 'read') } : null,
    r.proposal ? { label: `Open proposal #${r.proposal.id}`, act: 'proposal',
                   run: () => loopGo(r, 'review') } : null];
  return el('div', {}, actCell('loop-' + r.slug, b, items),
    st.ok ? '' : el('div', { class: 'propwhy', 'data-why': st.step,
      title: st.why, text: st.short || st.why || '' }));
}

// Which model the topic page proposes for: the one whose answers are open,
// else the last judged. The gate is the service's, per model.
function topicProposeModel(r) {
  const models = ansJudgedModels(r.task);
  const a = state.ans;
  if (a.model && models.includes(a.model)) return a.model;
  const last = (r.last_judged || {}).model;
  return last && models.includes(last) ? last : models[0] || '';
}

// Propose, on the topic page — the only place it starts. ok: a primary
// button. Every reason about the judge and none about the data: "Propose…",
// a secondary button with an amber dot that asks first. Any reason about the
// data: disabled, and the reason beside it in words.
function proposeControl(r) {
  const model = topicProposeModel(r);
  const gate = (r.propose_by_model || {})[model]
    || (model && model === (r.last_judged || {}).model ? r.propose : null);
  if (!model || !gate) return '';
  const slot = 'propose:' + r.slug;
  const attrs = { 'data-propose': r.slug, 'data-propose-model': model };
  const done = j => `Proposal #${j.id} requested — ${r.topic}`;
  rvNeeded();
  const open = openProposalFor(model, r.topic);
  if (open)
    return el('div', {}, el('a', { ...attrs, class: 'propose', 'data-gate': 'open',
      'data-review-link': String(open.id), href: '#tab=review&read=proposal:' + open.id,
      text: 'Review it →', title: `proposal #${open.id} is `
        + rvStatusWords(open.status).toLowerCase(),
      onclick: e => { if (e.metaKey || e.ctrlKey || e.shiftKey) return;
        e.preventDefault();
        openReader({ kind: 'proposal', id: String(open.id) },
          `[data-review-link="${open.id}"]`); } }), actNote(slot));
  // 11k: one dialog, here too — it opens over this page and does not leave it
  if (gate.ok)
    return el('div', {}, el('button', { ...attrs, class: 'primary', 'data-gate': 'ok',
      text: 'Propose', title: `ask the AI what skill ${model} is missing on ${r.topic}`,
      onclick: () => npDialog({ model, topic: r.topic, stay: true,
        returnTo: `[data-propose="${r.slug}"]` }) }), actNote(slot));
  if (gate.overridable)
    return el('div', {}, el('button', { ...attrs, class: 'secondary dot-warn',
      'data-gate': 'overridable', text: 'Propose…', title: gate.why,
      // 11j: the same dialog the Review tab opens, with these two filled in
      onclick: () => npDialog({ model, topic: r.topic, stay: true,
        returnTo: `[data-propose="${r.slug}"]` }) }),
      el('div', { class: 'propwhy', 'data-why': 'propose', title: gate.why,
        text: 'the judge is provisional — Propose… says what that means' }), actNote(slot));
  return el('div', {}, el('button', { ...attrs, class: 'primary', disabled: '', 'data-gate': 'hard',
      title: gate.why, text: 'Propose' }),
    el('div', { class: 'propwhy', 'data-why': 'propose', title: gate.why,
      text: gate.short || gate.why || '' }), actNote(slot));
}

// a proposal made over a provisional judge, wherever it shows up: the mark
// and, in its tooltip, why the grades were not evidence
function overBadge(over) {
  if (!over) return '';
  return el('span', { class: 'badge over', 'data-over-provisional': '1',
    title: `proposed by ${over.by} over a provisional judge: ${(over.reasons || []).join('; ')}`,
    text: 'over a provisional judge' });
}

// a row carries one warning badge at most: its own (the model trained on this
// topic). What is true of every score on the board — a provisional judge, a
// single provider, draft rubrics — is said once, above it (loopCaveats)
function judgedBadges(last) {
  if (!last || !last.tainted) return [];
  return [el('span', { class: 'badge taint',
    title: 'this model trained on data derived from this topic', text: 'trained on it' })];
}

function loopCaveats(rows) {
  const lasts = rows.map(r => r.last_judged).filter(Boolean);
  if (!lasts.length) return '';
  const bits = [];
  const prov = lasts.find(l => l.provisional);
  if (prov) bits.push(el('span', { class: 'badge taint', title: prov.provisional_reason,
    text: 'provisional judge' }));
  if (lasts.some(l => l.single_provider_loop)) bits.push(el('span', { class: 'badge taint',
    title: 'the same provider wrote, sat or graded more than one step of this loop',
    text: 'single provider' }));
  const draft = rows.filter(r => r.last_judged && r.last_judged.draft_rubric).map(r => r.topic);
  if (draft.length) bits.push(el('span', { class: 'badge taint',
    title: 'rubrics their author has not signed off yet', text: `draft rubric: ${draft.join(', ')}` }));
  if (!bits.length) return '';
  return el('p', { class: 'caveats', 'data-loop-caveats': '1' },
    el('span', { class: 'small', text: 'Every score below: ' }), ...bits);
}

function vLoop() {
  if (!state.loop.loaded && netReady()) loadLoop();
  const rows = state.loop.rows || [];
  const models = state.loop.models || [];
  const head = el('div', { class: 'card' },
    el('h2', {}, 'The loop, by topic', infoTip('Write the exam, sit it, read what the judge made '
      + 'of the answers, propose the skill that is missing, approve it, generate data, hand it '
      + 'to training. One row per topic, and the one thing to do next. Every refusal below is '
      + 'the API\'s own, in its words — this table asks, it does not decide.')),
    el('div', { class: 'frm' },
      el('label', { class: 'small', for: 'loopModel' }, el('b', { text: 'Results for ' })),
      models.length ? Combobox('results for', modelGroups(models.map(m => ({ id: m.id,
          judged: m.topics }))), state.loop.model,
        v => { state.loop.model = v; state.loop.loaded = false; _loopSig = null; loadLoop(); },
        { key: 'loop-model', placeholder: 'find a model' })
        : el('span', { class: 'se', text: 'no model has sat the exam yet' }),
      // 11i: the exam for this model starts on its page
      state.loop.model && DATA.models.some(x => x.id === state.loop.model)
        ? el('button', { class: 'quiet', 'data-loop-sit': state.loop.model, text: 'Sit the exam ▸',
            title: 'choose its topics on the model\'s page', onclick: () => openSit(state.loop.model) })
        : '',
      el('a', { class: 'small', href: 'guide#the-loop', target: '_blank', rel: 'noopener',
        text: 'what the loop is and whose job each step is' })),
    state.loop.blocked ? el('p', { class: 'warn', 'data-loop-blocked': '1' },
      el('b', { text: 'No topic can be sat right now. ' }), state.loop.blocked) : '',
    judgeOfflineLine(),
    state.loop.msg ? el('p', { class: 'small', 'data-loop-msg': '1', text: state.loop.msg }) : '',
    loopFailure(), loopCaveats(rows));
  const sel = head.querySelector('[aria-label="results for"]');
  if (sel) sel.id = 'loopModel';
  if (!state.loop.loaded)
    return [head, el('div', { class: 'card' }, el('p', { class: 'small',
      text: state.loop.failed ? 'No board to show yet — see above.' : '' }),
      state.loop.failed ? '' : skeleton(8, { 'data-loading': 'loop' }))];
  // topics with questions first; the ones without fold into one row that
  // says so and opens — ten empty rows were two thirds of the board. Of the
  // rest, the weakest for the model in "Results for" first — the one the loop
  // is for — then the ones it has not sat, by name; a search narrows both
  const q = (state.loop.q || '').trim().toLowerCase();
  const hit = r => !q || r.topic.toLowerCase().includes(q);
  const score = r => r.last_judged && r.last_judged.score_report != null
    ? r.last_judged.score_report : null;
  const full = rows.filter(r => (r.error || r.bank.accepted) && hit(r)).sort((a, b) => {
    const sa = score(a), sb = score(b);
    if (sa != null && sb != null && sa !== sb) return sa - sb;
    if ((sa == null) !== (sb == null)) return sa == null ? 1 : -1;
    return natCmp(a.topic, b.topic);
  });
  const empty = rows.filter(r => !r.error && !r.bank.accepted && hit(r));
  const pg = paged('loop', full, JSON.stringify([q, state.loop.model]), render, 25);
  const who = (models.find(m => m.id === state.loop.model) || {}).name || state.loop.model;
  const row = r => {
    const last = r.last_judged;
    if (r.error) return el('tr', { 'data-loop-row': r.slug, 'data-row-error': '1' },
      el('td', {}, r.topic),
      el('td', { class: 'warn', colspan: '6' }, r.error));
    return el('tr', { 'data-loop-row': r.slug },
      el('td', {}, el('a', { href: '#topic=' + r.slug, text: r.topic,
        onclick: e => { e.preventDefault(); loopGo(r, 'topic'); } })),
      el('td', { class: 'small' }, r.bank.accepted
        ? `${r.bank.accepted} — ${r.bank.report} report / ${r.bank.diagnose} diagnose`
        : 'no questions yet',
        r.bank.accepted && r.bank.under_floor
          ? el('div', { class: 'se', 'data-under-floor': '1',
              text: `under ${r.bank.floor} hidden questions` }) : '',
        r.bank.awaiting ? el('div', { class: 'se', text: `${r.bank.awaiting} awaiting curation` }) : ''),
      el('td', { class: 'small' },
        el('a', { href: '#tab=exam', 'data-fallback': r.rubric.fallback ? '1' : null,
          title: (r.rubric.fallback ? `this topic has no rubric of its own: the shared `
            + `${r.rubric.name}.md grades it. ` : '') + 'The rubric and criteria panel on the Exam tab.',
          text: r.rubric.fallback ? 'shared rubric' : `${r.rubric.name}.md`,
          onclick: e => { e.preventDefault(); navigate({ tab: 'exam', topic: null }); } }),
        r.rubric.status === 'draft' ? el('span', { class: 'badge taint', text: 'DRAFT' }) : '',
        r.rubric.scoring === 'criteria'
          ? el('div', { class: 'se', text: `${r.rubric.criteria_count} criteria` }) : ''),
      el('td', { class: 'small', 'data-loop-score': last ? last.model : '' }, last
        ? el('span', {}, `${num(last.score_report, 2)} / 4 `,
            el('span', { class: 'se', text: `(${last.n_report ?? 0} hidden) ` }),
            ...judgedBadges(last), ' ',
            el('a', { href: '#topic=' + r.slug, 'data-read': r.slug, text: 'read the results',
              onclick: e => { e.preventDefault(); loopGo(r, 'read'); } }))
        : r.bank.accepted
          ? el('span', { 'data-not-sat': '1' }, el('span', { class: 'se', text: 'Not sat — ' }),
              el('a', { href: '#topic=' + r.slug, text: 'Sit the exam',
                onclick: e => { e.preventDefault(); loopGo(r, 'sit'); } }))
          : el('span', { class: 'se', text: '—' })),
      el('td', { class: 'small' }, r.proposal
        ? el('a', { href: '#tab=review', text: `#${r.proposal.id} `
              + rvStatusWords(r.proposal.status).toLowerCase(),
            onclick: e => { e.preventDefault(); loopGo(r, 'review'); } })
        : el('span', { class: 'se', text: '—' })),
      el('td', { class: 'small' }, r.datasets.length
        ? r.datasets.map(d => el('div', { class: 'se', text: `#${d.id} ${d.status}` }))
        : el('span', { class: 'se', text: '—' })),
      el('td', { class: 'rowacts' }, loopBtn(r)));
  };
  const fold = empty.length ? el('tr', { 'data-empty-topics': String(empty.length) },
    el('td', { colspan: '6', class: 'small' },
      el('b', { text: `${empty.length} topic${empty.length > 1 ? 's' : ''} without questions: ` }),
      empty.map(r => r.topic).join(', '), ' ',
      el('button', { class: 'quiet', 'data-show-empty': '1',
        text: state.loop.showEmpty ? 'hide them' : 'show them',
        onclick: () => { state.loop.showEmpty = !state.loop.showEmpty; render(); } })),
    el('td', {}, el('button', { class: 'primary', 'data-step': 'import', text: 'Import a bank',
      onclick: () => { eximpState().topic = ''; loopGo({ topic: '' }, 'import'); } }))) : '';
  const table = el('div', { class: 'card' },
    el('div', { class: 'lb-wrap stick' }, el('table', { class: 'jd', 'data-loop-table': '1' },
      el('thead', {}, el('tr', {},
        el('th', { text: 'topic' }), el('th', { text: 'bank (report / diagnose)' }),
        el('th', { text: 'rubric' }), el('th', { text: who ? `results — ${who}` : 'results' }),
        el('th', { text: 'open proposal' }), el('th', { text: 'datasets' }),
        el('th', { text: 'next step' }))),
      el('tbody', {}, pg.rows.map(row), fold, state.loop.showEmpty ? empty.map(row) : []))),
    pg.pager);
  const search = el('div', { class: 'toolbar' },
    el('input', { type: 'search', placeholder: 'find a topic', 'aria-label': 'find a topic',
      'data-keep': 'loop-q', value: state.loop.q || '', style: 'flex:1;min-width:160px',
      oninput: e => { state.loop.q = e.target.value; render(); } }),
    el('span', { class: 'count-note', 'data-loop-count': '1',
      text: `${full.length} topic${full.length === 1 ? '' : 's'}`
        + (q ? ` match “${state.loop.q.trim()}”` : ' with questions')
        + (full.some(r => score(r) != null) ? ' · weakest first' : '') }));
  table.prepend(search);
  return [head, table];
}

// ---------------------------------------------------------------------------
// One topic, in loop order: where it stands, sit it, read the answers, propose,
// and what came out. The panels are the steps; the Review tab keeps its job as
// the cross-topic queue for whoever approves.
// ---------------------------------------------------------------------------

function vTopic() {
  if (!state.loop.loaded && netReady()) loadLoop();
  if (!state.queue.length && netReady()) loadQueue();
  const r = loopRowOf(state.topic);
  const back = el('p', { class: 'small' },
    el('a', { href: '#tab=loop', text: '← every topic',
      onclick: e => { e.preventDefault(); navigate({ topic: null, tab: 'loop' }); } }));
  if (!r) return [el('div', { class: 'card' }, back,
    el('h2', { text: state.topic }), el('p', { class: 'small',
      text: state.loop.loaded ? 'No such topic.' : '' }),
      state.loop.loaded ? '' : skeleton(5, { 'data-loading': 'topic' }))];
  const last = r.last_judged;
  const st = loopStep(r);
  const headCard = el('div', { class: 'card', 'data-topic-page': r.slug }, back,
    el('h2', { text: r.topic }), judgeOfflineLine(),
    el('div', { class: 'kvs' },
      el('span', {}, el('b', { text: 'questions ' }), r.bank.accepted
        ? [`${r.bank.accepted} (${r.bank.report} hidden / ${r.bank.diagnose} practice) `,
           readLink({ kind: 'bank', id: r.slug }, 'Read the practice questions',
             { 'data-read-bank': r.slug })] : 'none yet'),
      el('span', {}, el('b', { text: 'rubric ' }),
        readLink({ kind: 'rubric', id: r.rubric.name }, `${r.rubric.name}.md`,
          { 'data-read-rubric': r.rubric.name }),
        ` ${rubricVersion(r.rubric.version)}` + (r.rubric.status === 'draft' ? ' · DRAFT' : ''),
        r.rubric.scoring === 'criteria' ? [' · ', readLink({ kind: 'criteria', id: r.rubric.name },
          `${r.rubric.criteria_count} criteria`, { 'data-read-criteria': r.rubric.name })] : ''),
      el('span', {}, el('b', { text: 'next ' }), st.label)),
    r.bank.under_floor && r.bank.accepted ? el('p', { class: 'warn', text:
      `${r.bank.report} hidden questions — under the ${r.bank.floor} a topic needs before `
      + 'anything is proposed from it. Import or write more on the Exam tab.' }) : '',
    el('div', { class: 'frm' },
      r.next.step === 'propose' ? proposeControl(r) : loopBtn(r)),
    judgingLine(r),
    state.loop.msg ? el('p', { class: 'small', 'data-loop-msg': '1', text: state.loop.msg }) : '');
  return [headCard, loopSitPanel(r), loopAnswersPanel(r), loopOutputPanel(r)];
}

// a judged run of this topic that is still out: the queue row's own progress,
// on the page where the person is waiting for it
function judgingLine(r) {
  const mine = (state.queue || []).filter(q => q.suite === 'judged' && q.judge
    && q.judge.status !== 'done'
    && (() => { let t = []; try { t = JSON.parse(q.tasks || '[]'); } catch (e) { /* older row */ }
                return !t.length || t.includes(r.task); })());
  if (!mine.length) return '';
  return el('p', { class: 'small', 'data-judging': '1' },
    mine.map(q => el('span', {}, `${q.hf_id}: judging ${judgeCount(q.judge)} `,
      el('a', { href: '#tab=queue', text: 'in the queue',
        onclick: e => { e.preventDefault(); navigate({ tab: 'queue', topic: null }); } }), ' ')));
}

// The topic boxes a judged run is narrowed with, on the topic page's Sit panel
// and the Queue tab's judged suite. Thirty-six don't fit a glance: a filter,
// All and None over what the filter shows, and one line saying what the ticks
// cost — "12 of 36 topics · about 1,200 answers · about 25 min". The minutes
// come from the last judged runs (the service's pace), never a constant.
const roundAbout = n => n < 100 ? n : n < 1000 ? Math.round(n / 10) * 10
  : Math.round(n / 100) * 100;
function durationWords(sec) {
  const min = Math.max(1, Math.round(sec / 60));
  if (min < 90) return `about ${min} min`;
  const h = Math.floor(min / 60), m = Math.round((min % 60) / 10) * 10;
  return `about ${h} h` + (m ? ` ${m} min` : '');
}
function costLine(tasks, built) {
  const exam = built.filter(t => t.startsWith('exam_'));
  const ticked = tasks.filter(t => built.includes(t));
  const n = ticked.reduce((a, t) => a + ((state.loop.items || {})[t] || 0), 0);
  const pace = state.loop.pace;
  return `${ticked.filter(t => t.startsWith('exam_')).length} of ${exam.length} topics`
    + (n ? ` · about ${roundAbout(n).toLocaleString('en')} answers` : '')
    + (!n ? '' : pace && pace.sec_per_answer ? ` · ${durationWords(n * pace.sec_per_answer)}`
       : ' · no judged run yet to time it by');
}
function topicPicker(st, built, key) {
  const q = (st.pickQ || '').trim().toLowerCase();
  const shown = built.filter(t => !q || frName(t).toLowerCase().includes(q));
  const count = el('span', { class: 'count-note', 'data-pick-count': key,
    text: costLine(st.tasks || [], built) });
  const set = next => { st.tasks = next; count.textContent = costLine(next, built);
    for (const b of box.querySelectorAll('input[type=checkbox]'))
      b.checked = next.includes(b.dataset.task); };
  const box = el('div', { class: 'frm topicpick', style: 'flex-wrap:wrap', 'data-topic-picks': key },
    shown.map(task => el('label', { class: 'small', style: 'margin-right:10px' },
      el('input', { type: 'checkbox', 'data-task': task,
        [key === 'sit' ? 'data-sit-task' : 'data-submit-task']: task,
        checked: (st.tasks || []).includes(task) ? '' : null,
        onchange: e => set(e.target.checked ? [...(st.tasks || []), task]
                                            : (st.tasks || []).filter(t => t !== task)) }),
      ' ' + frName(task))),
    shown.length ? '' : el('span', { class: 'small se', text: 'no topic matches' }));
  return el('div', { 'data-picker': key },
    el('div', { class: 'toolbar' },
      el('input', { type: 'search', placeholder: 'find a topic', 'aria-label': 'find a topic',
        'data-keep': key + '-pick-q', value: st.pickQ || '', style: 'flex:1;min-width:140px',
        oninput: e => { st.pickQ = e.target.value; render(); } }),
      el('button', { class: 'quiet', 'data-pick-all': key, text: q ? 'All shown' : 'All',
        onclick: () => set([...new Set([...(st.tasks || []), ...shown])]) }),
      el('button', { class: 'quiet', 'data-pick-none': key, text: q ? 'None shown' : 'None',
        onclick: () => set((st.tasks || []).filter(t => !shown.includes(t))) }),
      count),
    box);
}

function loopSitPanel(r) {
  const s = state.loopSit;
  // a topic page ticks its own topic, exactly: the ticks belong to the page
  // they were made on, and law's must not ride along onto physics'
  if (s.tasks == null || s.page !== r.task) { s.tasks = [r.task]; s.page = r.task; }
  const built = state.loop.built || [];
  const pick = topicPicker(s, built, 'sit');
  const go = async () => {
    const body = { hf_id: s.model.trim(), kind: s.kind, suite: 'judged',
                   submitter: whoName(), tasks: s.tasks };
    if (!body.hf_id) { s.msg = 'a model id first — org/name, or local/<name>'; return render(); }
    if (!body.tasks.length) { s.msg = 'pick at least one topic'; return render(); }
    s.busy = true; render();
    const res = await fetch('api/submissions', { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
      body: JSON.stringify(body) }).catch(() => null);
    const j = res ? await res.json().catch(() => ({})) : {};
    s.busy = false;
    if (res && res.ok) {
      s.msg = '';
      rememberQueued(j.id);
      toast(`Queued #${j.id} — ${(j.tasks || []).map(frName).join(', ')}`,
            { key: 'sit', go: goQueue, link: 'see the queue' });
      loadQueue();
    } else s.msg = 'refused: ' + (j.detail || (res ? res.status : 'server unreachable'));
    render();
  };
  return el('div', { class: 'card', 'data-panel': 'sit' },
    el('h2', { text: 'Sit the exam' }),
    el('p', { class: 'sub', text: 'One model, this topic — the same judged suite, narrowed to '
      + 'the tasks you tick. The answers are graded by the pinned judge when the run finishes; '
      + 'only the judge sees the hidden questions.' }),
    state.loop.blocked ? el('p', { class: 'warn', 'data-sit-blocked': '1',
      text: state.loop.blocked }) : '',
    el('div', { class: 'frm' },
      modelBox('sit', s.model, v => { s.model = v; },
        it => { s.model = it.id; if (it.kind) s.kind = it.kind; render(); },
        { 'aria-label': 'model id', placeholder: 'search: org/model, or local/<name>' }),
      Select('kind', [['auto', 'kind: auto-detect'], ['base', 'kind: base'],
        ['instruct', 'kind: instruct']], s.kind || 'auto', v => { s.kind = v; }, { key: 'sit-kind' }),
      rvNameInput(),
      el('button', { 'data-sit': '1',
        disabled: (state.loop.blocked || s.busy || judgeDown() || cannotRun(s.model))
          ? '' : null,
        title: cannotRun(s.model) ? noWeightsWhy(s.model.trim())
          : judgeDown() ? judgeWhy() : '',
        text: s.busy ? 'queueing…' : 'Queue this run', onclick: go }),
      // disabled says why, beside it, not only on hover
      cannotRun(s.model)
        ? el('span', { class: 'propwhy', 'data-why': 'weights',
                       text: noWeightsWhy(s.model.trim()) })
        : judgeDown() ? el('span', { class: 'propwhy', 'data-why': 'sit', text: judgeWhy() }) : ''),
    el('p', { class: 'small', text: 'topics in this run:' }), pick,
    s.msg ? el('p', { class: 'small', 'data-sit-msg': '1', text: s.msg }) : '',
    el('p', { class: 'small' }, 'The queue is on ',
      el('a', { href: '#tab=queue', text: 'Queue',
        onclick: e => { e.preventDefault(); navigate({ tab: 'queue', topic: null }); } }),
      ' — a judged row says which topics it sat and how far the judge batch is.'));
}

// ---------------------------------------------------------------------------
// 11i: sit the exam from the model page. masein: "it should be from the model
// page, so I can click to start on all or some of the topics for a model."
// One picker — the 37 topics under their 8 areas, each with where THIS model
// stands on it — for the model page's panel and the Queue form's judged
// suite. The topic page keeps its one-topic panel.
// ---------------------------------------------------------------------------
// "22 Sep", whatever the browser's locale would make of September
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const exDay = t => { if (!t) return ''; const d = new Date(t * 1000);
  return `${d.getDate()} ${MONTHS[d.getMonth()]}`; };

// topic -> 'queued' | 'running' for this model's judged rows still in flight:
// the topics a row sat, or every built one for a row that sat the whole exam.
// A row whose answers are in and whose grades are not is still running.
function examInFlight(mid) {
  const out = {};
  const built = state.loop.built || [];
  for (const r of state.queue || []) {
    if (!mid || r.hf_id !== mid || r.suite !== 'judged') continue;
    const grading = r.status === 'done' && r.judge && r.judge.status !== 'done' && !r.judge_failed;
    const st = r.status === 'running' || grading ? 'running'
      : r.status === 'queued' || ACTIVE_STATUS.has(r.status) ? 'queued' : null;
    if (!st) continue;
    let ts = []; try { ts = JSON.parse(r.tasks || '[]'); } catch (e) { /* older row */ }
    for (const t of ts.length ? ts : built) if (out[t] !== 'running') out[t] = st;
  }
  return out;
}

// where one model stands on each topic: what its row says, and whether it
// can be ticked. From the model's judge.json (the current questions), its
// history (the retired ones, after 10b) and the queue.
function examStatus(mid) {
  const m = DATA.models.find(x => x.id === mid);
  const fly = examInFlight(mid);
  const built = state.loop.built || [];
  const prov = m ? !judgedOkM(m) : false;
  return t => {
    if (!built.includes(t)) return { k: 'nobank', text: 'no questions yet', off: true };
    if (fly[t]) return { k: fly[t], text: fly[t] === 'running' ? 'running…' : 'in the queue',
                         off: true };
    const v = m && m.judge && (m.judge.tasks || {})[t];
    if (v) {
      // sat on the current questions: its score is the hidden half's, and a
      // topic with no hidden questions yet was graded on practice ones only
      const sc = pubScore(v), day = v.judged_at ? ` · ${exDay(v.judged_at)}` : '';
      return sc != null ? { k: 'judged', score: sc, prov, text: `${num(sc, 2)} / 4` + day }
        : { k: 'judged', score: null, prov: false, text: 'no hidden questions yet' + day };
    }
    if (m && ((m.judge || {}).history || []).some(e => e && e.task === t))
      return { k: 'older', text: 'on older questions' };
    return { k: 'notsat', text: 'not sat' };
  };
}

// "12 topics · about 1,200 answers · about 25 min of GPU, then grading · the
// GPU is shared". A topic already judged on the same questions costs no GPU:
// 10b's resume keeps its answers and only grades them again.
function examSummary(ticked, control, statusOf) {
  const J = DATA.judged || {}, items = state.loop.items || {}, pace = state.loop.pace;
  if (!ticked.length && !control) return 'Tick the topics this model should sit.';
  const regrade = ticked.filter(t => statusOf(t).k === 'judged');
  const fresh = [...ticked.filter(t => !regrade.includes(t)), ...(control ? [J.control] : [])];
  const n = fresh.reduce((a, t) => a + (items[t] || 0), 0);
  const bits = [(ticked.length === 1 ? '1 topic' : `${ticked.length} topics`)
    + (control ? ' + MMLU control' : '')];
  if (regrade.length) bits.push(`${regrade.length} re-grade only`);
  if (n) {
    bits.push(`about ${roundAbout(n).toLocaleString('en')} answers`);
    bits.push(pace && pace.sec_per_answer
      ? `${durationWords(n * pace.sec_per_answer)} of GPU, then grading`
      : 'no judged run yet to time the GPU by');
    bits.push('the GPU is shared');
  } else bits.push('no GPU, grading only');
  return bits.join(' · ') + (ticked.length > 10 ? ' — consider a quiet time' : '');
}

// st: { tasks: [ticked exam tasks], control: bool }. Everything after the
// first paint changes in place, so a tick never rebuilds the panel around it.
function examPicker(st, mid, key, onChange) {
  const J = DATA.judged || {};
  const statusOf = examStatus(mid);
  const byName = Object.fromEntries(Object.entries(J.topics || {}).map(([t, n]) => [n, t]));
  const areas = Object.entries(DATA.meta.areas || {}).map(([a, names]) =>
    [a, names.map(n => byName[n]).filter(Boolean)]).filter(([, ts]) => ts.length);
  const placed = new Set(areas.flatMap(([, ts]) => ts));
  const rest = (J.exam || []).filter(t => !placed.has(t));
  if (rest.length) areas.push(['Other', rest]);
  const all = areas.flatMap(([, ts]) => ts);
  const can = t => !statusOf(t).off;
  const judged = all.filter(t => can(t) && statusOf(t).score != null)
    .sort((a, b) => statusOf(a).score - statusOf(b).score);
  const sets = {
    notsat: all.filter(t => can(t) && ['notsat', 'older'].includes(statusOf(t).k)),
    all: all.filter(can),
    weakest: judged.slice(0, 5),
    none: [],
  };
  const boxes = new Map(), notes = new Map(), groups = [];
  const summary = el('p', { class: 'exsum', 'data-exam-summary': key, 'aria-live': 'polite' });
  const set = (next, quiet) => {
    st.tasks = all.filter(t => next.includes(t) && can(t));
    for (const [t, b] of boxes) b.checked = st.tasks.includes(t);
    for (const { box, count, ts } of groups) {
      const on = ts.filter(t => st.tasks.includes(t)).length, open = ts.filter(can).length;
      box.checked = open > 0 && on === open;
      box.indeterminate = on > 0 && on < open;
      box.disabled = !open;
      // 11k: "0 of 5" read as "0 of 5 judged"; it means ticked. Say both
      count.textContent = `${on} ticked · ${ts.filter(t => statusOf(t).k === 'judged').length}`
        + ` of ${ts.length} judged`;
    }
    for (const [t, n] of notes) n.hidden = !st.tasks.includes(t);
    summary.textContent = examSummary(st.tasks, !!st.control, statusOf);
    if (!quiet && onChange) onChange();
  };
  const quick = (k, label, extra = {}) => el('button', { class: 'chip-btn', type: 'button',
    'data-quick': k, text: label, onclick: () => set(sets[k]), ...extra });
  const body = areas.map(([a, ts]) => {
    const box = el('input', { type: 'checkbox', 'data-area-box': a,
      onchange: e => set(e.target.checked ? [...st.tasks, ...ts.filter(can)]
                                          : st.tasks.filter(t => !ts.includes(t))) });
    const count = el('span', { class: 'excount' });
    groups.push({ box, count, ts });
    return el('div', { class: 'exarea', role: 'group', 'aria-label': a, 'data-area': a },
      el('div', { class: 'exhead' }, el('label', {}, box, ' ' + a), count),
      ts.map(t => {
        const s = statusOf(t);
        const b = el('input', { type: 'checkbox', 'data-exam-task': t,
          disabled: s.off ? '' : null,
          onchange: e => set(e.target.checked ? [...st.tasks, t] : st.tasks.filter(x => x !== t)) });
        boxes.set(t, b);
        const note = s.k === 'judged' ? el('span', { class: 'exnote', 'data-regrade': t,
          hidden: '', text: 'same questions — re-grade only, no GPU' }) : '';
        if (note) notes.set(t, note);
        return el('label', { class: 'extopic' + (s.off ? ' off' : ''), 'data-exam-topic': t,
            'data-status': s.k },
          b, el('span', { class: 'exname', title: frName(t), text: frName(t) }),
          el('span', { class: 'exst', 'data-exam-status': t, text: s.text },
            s.prov ? el('span', { class: 'exprov', text: ' · provisional' }) : ''),
          note);
      }));
  });
  // the control is its own question, off unless someone asks for it
  const cs = J.control && (state.loop.built || []).includes(J.control) ? statusOf(J.control) : null;
  if (!cs || cs.off) st.control = false;
  const control = cs ? el('label', { class: 'excontrol', 'data-exam-control-row': '1' },
    el('input', { type: 'checkbox', 'data-exam-control': '1', checked: st.control ? '' : null,
      disabled: cs.off ? '' : null,
      onchange: e => { st.control = e.target.checked; set(st.tasks); } }),
    el('span', {}, el('b', { text: 'MMLU control (open-ended)' }),
      ' — the MMLU questions this model got wrong as multiple choice, asked openly: was the '
      + 'knowledge there under the format?' + (cs.off ? ` (${cs.text})` : ''))) : '';
  const box = el('div', { class: 'expick', 'data-exam-picker': key },
    el('div', { class: 'toolbar exquick' },
      el('span', { class: 'small se', text: 'Quick picks' }),
      quick('notsat', `Not sat yet (${sets.notsat.length})`,
        sets.notsat.length ? {} : { disabled: '' }),
      quick('all', `All ${sets.all.length}`, sets.all.length ? {} : { disabled: '' }),
      quick('weakest', 'Weakest 5', judged.length ? { title: 'this model\'s five lowest judged '
        + 'topics' } : { disabled: '', title: 'this model has no judged topics yet' }),
      quick('none', 'None')),
    el('div', { class: 'exareas' }, body),
    control, summary);
  set(st.tasks || [], true);
  return box;
}

// ---------------------------------------------------------------------------
// 11i: a checkpoint that ships its own model code (an auto_map in its
// config.json) says so BEFORE it is queued. #56 learned it at start, after the
// wait, and its Resubmit had no way to ask. The server reads the file; the
// page offers an unticked box when the server may run it, and says why not
// when it may not — in the words the API refuses with.
// ---------------------------------------------------------------------------
state.codeInfo = {};
async function loadCodeInfo(mid) {
  const c = state.codeInfo[mid];
  if (c && c !== 'loading') return c;
  state.codeInfo[mid] = 'loading';
  try {
    const j = await api('api/models/code?id=' + encodeURIComponent(mid));
    state.codeInfo[mid] = j;
    if (j.weights === false) state.noWeights.add(mid);
    else if (j.weights === true) state.noWeights.delete(mid);
  } catch (e) {
    // the API still refuses in its own words at Queue; the page stops asking
    state.codeInfo[mid] = { own_code: false, unknown: true };
  }
  return state.codeInfo[mid];
}
// false: nothing to ask (a Hub id never runs its own code here); null: being
// asked; else the server's answer
function codeInfo(mid, redraw) {
  mid = String(mid || '').trim();
  if (!LIVE || !/^local\/[^/]+$/.test(mid)) return false;
  const c = state.codeInfo[mid];
  if (c && c !== 'loading') return c;
  if (!c) loadCodeInfo(mid).then(() => (redraw || render)());
  return null;
}
function ownCodeBox(info, checked, onToggle, key) {
  if (!info || !info.own_code || info.why) return '';
  const f = info.files || [];
  return el('label', { class: 'owncode', 'data-own-code': key },
    el('input', { type: 'checkbox', 'data-own-code-box': key, checked: checked ? '' : null,
      onchange: e => onToggle(e.target.checked) }),
    el('span', {}, 'Run this checkpoint\'s own model code (',
      el('code', { text: f.map(x => x.file).join(', ') }), ', sha ',
      el('code', { title: f.map(x => `${x.file} ${x.sha}`).join('\n'),
        text: f.map(x => x.sha.slice(0, 4) + '…').join(', ') }),
      ') — as the unprivileged ', el('code', { text: info.user || 'benchjob' }), ' user'));
}
// why this run cannot be queued yet, or '' — the reason goes beside the button
function ownCodeWhy(info, allowed) {
  if (info === null) return 'Checking the checkpoint…';
  if (!info || !info.own_code) return '';
  if (info.why) return info.why;
  return allowed ? '' : 'This checkpoint ships its own model code — tick the box to run it.';
}

// ---------------------------------------------------------------------------
// The model page's panel. It opens from the hero's Sit the exam, from an
// opened Leaderboard row's Run exam, and from the Loop — always here, and
// scrolled to.
// ---------------------------------------------------------------------------
state.msit = { model: null, open: false, tasks: [], control: false, allow: false,
               busy: false, msg: '' };
state.msitRedraw = null;

function openSit(mid) {
  const s = state.msit;
  if (s.model !== mid) Object.assign(s, { model: mid, tasks: [], control: false, allow: false,
                                          msg: '' });
  s.open = true;
  delete state.codeInfo[mid];              // asked afresh each time it opens
  if (!state.loop.loaded && netReady()) loadLoop();
  state.after = { scroll: '[data-panel="msit"]' };
  if (state.model !== mid || state.topic) navigate({ model: mid, topic: null });
  else render();
}

function sitWhy(m, s, info) {
  if (!state.loop.loaded) return 'Loading the exam…';
  if (state.loop.blocked) return state.loop.blocked;
  if (judgeDown()) return judgeWhy();
  const code = ownCodeWhy(info, s.allow);
  if (code === 'Checking the checkpoint…') return code;
  if (cannotRun(m.id)) return noWeightsWhy(m.id);
  if (code) return code;
  if (!s.tasks.length && !s.control) return 'Tick at least one topic.';
  return '';
}

async function sitGo(m, s) {
  const J = DATA.judged || {};
  const tasks = [...s.tasks, ...(s.control && J.control ? [J.control] : [])];
  const body = { hf_id: m.id, kind: ['base', 'instruct'].includes(m.kind) ? m.kind : 'auto',
                 suite: 'judged', submitter: whoName(), tasks,
                 ...(s.allow ? { allow_remote_code: true } : {}) };
  s.busy = true; s.msg = '';
  if (state.msitRedraw) state.msitRedraw();
  try {
    const j = await post('api/submissions', body);
    rememberQueued(j.id);
    const n = s.tasks.length;
    toast(j.note ? `#${j.id}: ${j.note}` : `Queued #${j.id} — ${m.name} · `
      + (n === 1 ? frName(s.tasks[0]) : `${n} topics`) + (s.control ? ' + MMLU control' : ''),
      { key: 'msit', go: goQueue, link: 'see the queue' });
    Object.assign(s, { tasks: [], control: false, allow: false });
    s.busy = false;
    await loadQueue();
  } catch (e) { s.msg = 'Refused: ' + e.message; }
  s.busy = false;
  if (state.msitRedraw) state.msitRedraw(); else render();
}

function modelSitPanel(m) {
  const s = state.msit;
  if (!LIVE || !s.open || s.model !== m.id) { state.msitRedraw = null; return null; }
  if (!state.loop.loaded && netReady()) loadLoop();
  const card = el('div', { class: 'card', 'data-panel': 'msit' });
  const paint = () => {
    const info = codeInfo(m.id, () => state.msitRedraw && state.msitRedraw());
    const why = el('span', { class: 'propwhy', 'data-why': 'msit' });
    const btn = el('button', { class: 'primary', 'data-msit-go': m.id,
      text: s.busy ? 'Queueing…' : 'Queue this run', onclick: () => sitGo(m, s) });
    const gate = () => {
      const w = sitWhy(m, s, info);
      btn.disabled = !!w || s.busy;
      why.textContent = w;
      why.hidden = !w;
    };
    card.replaceChildren(
      el('div', { class: 'msit-head' },
        el('h2', { text: 'Sit the exam' }),
        el('button', { class: 'ghost', 'data-msit-close': '1', text: '✕ Close',
          onclick: () => { s.open = false; render(); } })),
      el('p', { class: 'sub', text: `${m.name}, on the topics you tick. The judge grades the `
        + 'answers when the run finishes; only the judge sees the hidden questions.' }),
      state.loop.loaded ? examPicker(s, m.id, 'msit', gate)
        : skeleton(4, { 'data-loading': 'msit' }),
      ownCodeBox(info, s.allow, v => { s.allow = v; gate(); }, 'msit'),
      el('div', { class: 'frm msit-go' }, btn, why),
      s.msg ? el('p', { class: 'small warn', 'data-msit-msg': '1', text: s.msg }) : '');
    gate();
  };
  state.msitRedraw = () => { if (card.isConnected) paint(); };
  paint();
  return card;
}

// the Suite cell: one line whatever the run sat, and the list behind ▸,
// grouped by area. #56's row listed 37 names and stood 650px tall.
state.suiteOpen = new Set();
function suiteCell(r, key) {
  let ts = [];
  try { ts = JSON.parse(r.tasks || '[]'); } catch (e) { /* older row */ }
  const J = DATA.judged || {};
  if (r.suite !== 'judged') return el('span', { 'data-suite-cell': key, text: r.suite });
  if (!ts.length) return el('span', { 'data-suite-cell': key, text: 'judged · the whole exam' });
  const ex = ts.filter(t => t !== J.control), ctl = ts.includes(J.control);
  const label = 'judged · ' + (ex.length === 1 ? frName(ex[0]) : ex.length
    ? `${ex.length} topics` : '') + (ctl ? (ex.length ? ' + ' : '') + 'MMLU control' : '');
  if (ex.length <= 1) return el('span', { 'data-suite-cell': key, text: label });
  const byName = Object.fromEntries(Object.entries(J.topics || {}).map(([t, n]) => [n, t]));
  const lines = Object.entries(DATA.meta.areas || {}).map(([a, names]) =>
    [a, names.map(n => byName[n]).filter(t => ex.includes(t))]).filter(([, xs]) => xs.length);
  const placed = new Set(lines.flatMap(([, xs]) => xs));
  const other = ex.filter(t => !placed.has(t));
  if (other.length) lines.push(['Other', other]);
  const id = key + ':' + r.id;
  return el('details', { class: 'suitecell', 'data-suite-cell': key,
      open: state.suiteOpen.has(id) ? '' : null,
      ontoggle: e => { if (e.target.open) state.suiteOpen.add(id); else state.suiteOpen.delete(id); } },
    el('summary', { text: label + ' ▸' }),
    el('div', { class: 'suitelist' }, lines.map(([a, xs]) => el('div', {},
      el('b', { text: a + ': ' }), xs.map(frName).join(', ')))));
}

// ---------------------------------------------------------------------------
// The answers, for the diagnosis half only. The report half is one aggregate
// line above the table and nothing else: no row, no qid, no answer text — an
// answer quotes its question often enough that showing one shows the other.
// The endpoint enforces the same rule; this is the second lock on the door.
// ---------------------------------------------------------------------------

async function loadAnswers(model, topic) {
  const a = state.ans;
  a.loading = true; a.model = model; a.topic = topic;
  try {
    const j = await api(`api/answers?model=${encodeURIComponent(model)}`
                        + `&topic=${encodeURIComponent(topic)}`);
    Object.assign(a, { rows: j, loading: false });
  } catch (e) { Object.assign(a, { rows: null, loading: false }); }
  if (state.topic || state.model) render();
}

function ansJudgedModels(task) {
  return DATA.models.filter(m => m.judge && (m.judge.tasks || {})[task])
    .map(m => m.id).sort(natCmp);
}

function ansFilters(j) {
  const a = state.ans;
  const acuities = [...new Set(j.items.map(it => (it.meta || {}).acuity).filter(Boolean))];
  const sel = (label, cur, opts, on) => mkSel(label, opts, cur, on);
  return el('div', { class: 'toolbar' },
    acuities.length ? sel('acuity filter', a.acuity,
      [['all', 'acuity: all'], ...acuities.map(v => [v, v])],
      v => { a.acuity = v; render(); }) : '',
    j.flags.length ? sel('flag filter', a.flag,
      [['all', 'flags: any'], ['none', 'no flag raised'],
       ...j.flags.map(f => [f.id, 'flagged: ' + f.label])],
      v => { a.flag = v; render(); }) : '',
    sel('score filter', a.score,
      [['all', 'score: any'], ['weak', 'scored under 2'], ['0', '0'], ['1', '1'], ['2', '2'],
       ['3', '3'], ['4', '4'], ['ungraded', 'the judge\'s reply was unreadable']],
      v => { a.score = v; render(); }),
    (j.criteria || []).length ? sel('criterion filter', a.crit,
      [['all', 'criteria: any'], ['any', 'any criterion below 0.5'],
       ...(j.criteria || []).map(c => [c.id, `${c.label} below 0.5`])],
      v => { a.crit = v; render(); }) : '',
    sel('sort', a.sort, [['score', 'weakest first'], ['best', 'best first'],
                         ['difficulty', 'by difficulty']],
      v => { a.sort = v; render(); }));
}

function ansVisible(j) {
  const a = state.ans;
  let rows = j.items.filter(it => {
    if (a.acuity !== 'all' && (it.meta || {}).acuity !== a.acuity) return false;
    if (a.flag === 'none' && Object.values(it.flags || {}).some(Boolean)) return false;
    if (a.flag !== 'all' && a.flag !== 'none' && !(it.flags || {})[a.flag]) return false;
    const cv = it.criteria || {};
    if (a.crit === 'any' && !Object.values(cv).some(v => v != null && v < 0.5)) return false;
    if (a.crit !== 'all' && a.crit !== 'any' && !(cv[a.crit] != null && cv[a.crit] < 0.5))
      return false;
    if (a.score === 'ungraded') return !it.graded;
    if (a.score === 'weak') return it.graded && it.score < 2;
    if (a.score !== 'all') return it.graded && String(it.score) === a.score;
    return true;
  });
  if (a.sort === 'best') rows = rows.slice().sort((x, y) => (y.score ?? -1) - (x.score ?? -1));
  else if (a.sort === 'difficulty') rows = rows.slice().sort((x, y) =>
    ((x.meta || {}).difficulty ?? 99) - ((y.meta || {}).difficulty ?? 99));
  return rows;
}

// The criteria as one row of 14 px cells — at most 23, about 350 px — on a
// one-hue sequential scale from the theme's accent (colour-blind safe: it is
// lightness that carries the value). The cells are the overview; the three
// weakest, named in words beside them, are what people read. Each cell says
// its criterion and value to the mouse and to a screen reader.
const critColor = v => `color-mix(in srgb, var(--accent) ${Math.round(10 + 90 * v)}%, var(--plane))`;

function ansCells(it, criteria) {
  return el('span', { class: 'critcells' }, criteria.map(c => {
    const v = (it.criteria || {})[c.id];
    const words = `${c.label}: ${v == null ? 'not applicable' : num(v, 2)}`;
    return el('span', { class: 'critcell' + (v == null ? ' na' : ''), role: 'img',
      'data-criterion': c.id, title: words, 'aria-label': words,
      style: v == null ? null : `background:${critColor(v)}` });
  }));
}

function ansWeakest(it, criteria, n = 3) {
  return criteria.map(c => ({ c, v: (it.criteria || {})[c.id] }))
    .filter(x => x.v != null && x.v < 1)
    .sort((a, b) => a.v - b.v || natCmp(a.c.label, b.c.label)).slice(0, n);
}

// One answer: what was asked, what the model wrote, what the judge wrote, the
// criteria; the score large on the right with the flags under it. Nothing
// sideways — the judge's note is the most useful text here and it wraps.
const wordsIn = t => (String(t || '').match(/\S+/g) || []).length;

function ansCard(it, j) {
  const meta = it.meta || {};
  const flags = (j.flags || []).filter(f => (it.flags || {})[f.id]);
  const open = state.ans.open[it.qid];
  const crit = j.criteria || [];
  const weak = ansWeakest(it, crit);
  const head = [meta.acuity, meta.difficulty != null ? 'difficulty ' + meta.difficulty : null,
                meta.jurisdiction_required != null
                  ? (meta.jurisdiction_required ? 'jurisdiction required' : '') : null,
                meta.intent].filter(Boolean).join(' · ');
  const long = (it.answer || '').length > 220;
  // 11l: a reasoning model's answer is what came after its reasoning; the
  // reasoning is one click away. One that never left it is no answer at all
  const why = open => el('details', { class: 'small', 'data-reasoning': it.qid,
      open: open ? '' : null },
    el('summary', { text: `The model's reasoning (${wordsIn(it.reasoning)} words)`
      + (it.reasoning_unterminated ? ' — it never finished' : '') + ' ▸' }),
    el('div', { class: 'se', style: 'white-space:pre-wrap;overflow-wrap:anywhere',
      text: it.reasoning || '' }));
  return el('article', { class: 'anscard' + (it.no_answer ? ' noanswer' : ''),
      'data-answer': it.qid, 'data-half': 'diagnose',
      'data-no-answer': it.no_answer ? '1' : null },
    el('div', { class: 'main', style: 'min-width:0' },
      head ? el('div', { class: 'ansmeta', text: head }) : '',
      el('div', { class: 'qa' },
        el('span', { class: 'lbl', text: 'Q' }),
        el('div', { class: 'txt', text: it.prompt || '(question not on disk)' }),
        el('span', { class: 'lbl', text: 'A' }),
        el('div', { class: 'txt' },
          it.no_answer
            ? el('div', { class: 'warn', 'data-answer-text': '1',
                text: 'No answer: the model was still reasoning when it ran out of room. '
                  + 'This question was not scored.' })
            : el('div', { class: open ? '' : 'clamp3', 'data-answer-text': '1',
                text: it.answer || '(the model wrote nothing)' }),
          long && !it.no_answer ? el('button', { class: 'quiet', style: 'padding:2px 0;font-size:12px',
            'data-answer-toggle': '1', 'aria-expanded': String(!!open),
            text: open ? 'Show less' : 'Show the whole answer',
            onclick: () => { state.ans.open[it.qid] = !open; render(); } }) : '',
          it.had_reasoning ? why(it.no_answer) : ''),
        el('span', { class: 'lbl', text: 'Judge' }),
        el('div', { class: 'txt', 'data-judge-note': '1', text: it.justification || '—' }),
        crit.length ? el('span', { class: 'lbl', text: 'Criteria' }) : '',
        crit.length ? el('div', { class: 'critrow' }, ansCells(it, crit),
          el('span', { class: 'small se', 'data-weakest': '1', text: weak.length
            ? 'weakest: ' + weak.map(x => `${x.c.label} ${(+x.v).toFixed(1)}`).join(' · ')
            : 'every criterion met' })) : '')),
    el('div', { class: 'side' },
      el('div', { class: 'score', text: it.graded ? `${it.score} / 4` : '—' }),
      it.graded ? '' : el('div', { class: 'se', text: it.no_answer ? 'no answer — not scored'
        : 'the judge\'s reply was unreadable' }),
      el('div', {}, flags.map(f => el('span', { class: 'badge danger', title: f.effect_words,
        'data-flag': f.id, text: f.label })))));
}

function loopAnswersPanel(r) {
  const a = state.ans;
  const models = ansJudgedModels(r.task);
  const card = el('div', { class: 'card', 'data-panel': 'answers' },
    el('h2', { text: 'The answers' }),
    el('p', { class: 'sub', text: 'What the model wrote on this topic\'s practice questions, '
      + 'and what the judge made of each answer. The hidden questions are the score: they '
      + 'appear below as one line and never as a row — not their questions, not '
      + 'its answers, not its qids.' }));
  if (!models.length) {
    card.append(empty('No model has been judged on this topic yet.', 'Sit the exam', () => {
      state.after = { scroll: '[data-panel="sit"]', focus: '[data-ms="sit"] input' }; render(); }));
    return card;
  }
  const want = a.model && models.includes(a.model) ? a.model
             : (r.last_judged || {}).model && models.includes((r.last_judged || {}).model)
               ? r.last_judged.model : models[0];
  if (netReady() && (!a.rows || a.model !== want || a.topic !== r.topic) && !a.loading) {
    loadAnswers(want, r.topic);
  }
  // a model judged while this page was open: said once, beside the select,
  // until the select is used — a list that grows silently is a list nobody
  // notices has grown
  const seen = a.seen[r.task];
  if (seen) for (const m of models) if (!seen.has(m)) (a.fresh[r.task] = a.fresh[r.task] || []).push(m);
  a.seen[r.task] = new Set(models);
  const fresh = (a.fresh[r.task] || []).filter(m => models.includes(m));
  const scoreOf = m => { const mm = DATA.models.find(x => x.id === m);
    const t = mm && mm.judge && mm.judge.tasks[r.task];
    return t && pubScore(t) != null ? ` — ${num(pubScore(t), 2)} / 4` : ''; };
  card.append(el('div', { class: 'frm' },
    el('span', { 'data-answers-model': '1' },
      Combobox('model', modelGroups(models.map(id => ({ id, right: scoreOf(id).replace(/^ — /, '') }))),
        want, v => { state.ans.model = v; state.ans.rows = null; a.fresh[r.task] = []; render(); },
        { key: 'answers-model', placeholder: 'find a model' })),
    fresh.length ? el('span', { class: 'badge new', 'data-new-model': fresh.join(','),
      text: 'new: ' + fresh.join(', ') }) : '',
    el('a', { class: 'small', href: '#model=' + encodeURIComponent(want),
      text: 'this model\'s page',
      onclick: e => { e.preventDefault(); navigate({ model: want, topic: null }); } })));
  const j = a.rows;
  if (!j || a.topic !== r.topic) {
    card.append(skeleton(4, { 'data-loading': 'answers' }));
    return card;
  }
  const rep = j.report_half || {};
  card.append(el('p', { class: 'note', 'data-report-half': '1' },
    el('b', { text: 'The hidden questions. ' }),
    // 11l: a topic that was not scored says so, instead of "mean — / 4"
    j.no_score ? `${rep.n ?? 0} questions, not scored: ${j.no_score}`
      : `${rep.n ?? 0} questions, mean ${num(rep.mean, 2)} / 4`
        + (j.no_answer ? ` · ${j.no_answer} answers never finished, left out` : '')
    // the flags OF THIS HALF: the whole-bank count under this heading would
    // be a claim about the published score that is not true of it
    + (Object.entries(rep.flags || {}).filter(([, n]) => n).length
       ? ' · ' + Object.entries(rep.flags).filter(([, n]) => n)
           .map(([fid, n]) => `${n} ${fid.replace(/_/g, ' ')}`).join(', ') : '')
    + '. That is the published score, and this line is all of it you will see '
    + 'here.'
    + (Object.entries(rep.flags_whole_bank || {}).some(([fid, n]) => n > (rep.flags || {})[fid])
       ? ' Across both halves: ' + Object.entries(rep.flags_whole_bank)
           .filter(([, n]) => n).map(([fid, n]) => `${n} ${fid.replace(/_/g, ' ')}`).join(', ')
           + '.' : '')));
  const rows = ansVisible(j);
  card.append(ansFilters(j));
  card.append(el('p', { class: 'small', 'data-answer-count': String(rows.length),
    text: `${rows.length} of ${j.n_diagnose} practice answers` }));
  const pg = paged('answers', rows,
    JSON.stringify([a.model, a.topic, a.acuity, a.flag, a.score, a.crit, a.sort]));
  card.append(pg.pager);
  card.append(el('div', { class: 'anslist', 'data-answers-table': '1' },
    pg.rows.map(it => ansCard(it, j))));
  if (pg.pager) card.append(pager('answers', rows.length));        // and again under the list
  if (!rows.length) card.append(empty('No answer matches these filters.', 'Clear the filters',
    () => { Object.assign(state.ans, { acuity: 'all', flag: 'all', score: 'all', crit: 'all' });
            render(); }));
  return card;
}

// What came out of the loop for this topic: the proposal being reviewed, and
// the dataset a person hands to training.
function loopOutputPanel(r) {
  const card = el('div', { class: 'card', 'data-panel': 'output' },
    el('h2', { text: 'Propose, approve, generate' }),
    el('p', { class: 'sub', text: 'A proposal reads the judge\'s written assessments of the '
      + 'answers above — never the questions — and says what skill is missing. A person '
      + 'approves that sentence on the Review tab, and only the approved text reaches a '
      + 'generator.' }));
  if (r.proposal) {
    card.append(el('p', { class: 'small', 'data-proposal': String(r.proposal.id) },
      `Proposal #${r.proposal.id} for ${r.proposal.model} is ${r.proposal.status}`
      + (r.proposal.requested_by ? `, requested by ${r.proposal.requested_by}` : ''), ' ',
      overBadge(r.proposal.override), '. ',
      el('a', { href: '#tab=review', text: 'Review it',
        onclick: e => { e.preventDefault(); loopGo(r, 'review'); } })));
  } else {
    card.append(el('p', { class: 'small' }, 'No open proposal. '
      + (r.next.step === 'propose' ? 'Propose from the top of this page.' : '')));
  }
  const ready = (r.datasets || []).filter(d => d.status === 'ready');
  if (ready.length) {
    card.append(el('div', { class: 'dxh', text: 'Datasets from this topic' }));
    // 11j: the same one-line row as the Review tab's Datasets view
    card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd rvlist',
      'data-datasets-table': '1' },
      el('thead', {}, el('tr', {}, el('th', { text: '#' }), el('th', { text: 'topic' }),
        el('th', { text: 'model' }), el('th', { class: 'num', text: 'documents' }),
        el('th', { text: 'made by' }), el('th', { class: 'num', text: 'when' }),
        el('th', { text: '' }), el('th', { text: '' }))),
      el('tbody', {}, ready.map(d => dsRow({ ...d,
        provenance: d.provenance || { items: { kept: d.kept ?? d.count, requested: d.count,
          ...(Array.isArray(d.missing) ? { missing: d.missing } : {}) } },
        category: d.category || r.topic, model: d.model || (r.last_judged || {}).model,
        status: d.status, download: `api/datasets/${d.id}/items.jsonl` }))))));
  } else if (r.datasets.length) {
    card.append(note('A dataset for this topic is being generated — it appears here when the '
      + 'batch completes and the contamination gate has run.'));
  }
  return card;
}

// ---------- Exam: write the instrument ----------
// An LLM drafts candidate questions per topic (scripts/exam_build.py draft); a
// person accepts, edits or rejects each one here, under a name; accepted
// questions land in the bank, split by qid into a report half (the published
// score) and a diagnose half (what a proposal may read). Nothing an LLM wrote
// reaches the bank unread, and no report-half question is ever shown again.
async function loadExam() {
  try {
    const q = state.ex.topic ? '?topic=' + encodeURIComponent(state.ex.topic) : '';
    const [status, cands] = await Promise.all([
      api('api/exam'), api('api/exam/candidates' + q)]);
    const changed = !state.ex.loaded || JSON.stringify([status.summary, cands])
      !== JSON.stringify([(state.ex.status || {}).summary, state.ex.candidates]);
    Object.assign(state.ex, { status, candidates: cands, loaded: true });
    if (changed && state.tab === 'exam' && !state.model) render();
  } catch (e) { /* server briefly away */ }
}

async function exPost(path, body) {
  if ('approver' in (body || {}) && !whoName()) { state.ex.msg = askName(); render(); return; }
  const r = await fetch(path, { method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
    body: JSON.stringify(body || {}) }).catch(() => null);
  const j = r ? await r.json().catch(() => ({})) : {};
  state.ex.msg = r && r.ok ? '' : 'refused: ' + (j.detail || (r ? r.status : 'server unreachable'));
  if (r && r.ok) toast(j.qid ? `Accepted into ${j.topic} — ${j.half} half`
                             + ((j.build || {}).built ? ', and it can be sat now' : '')
                           : j.status === 'rejected' ? 'Rejected' : 'Done', { key: 'curate' });
  state.ex.loaded = false;
  await loadExam(); render();
}

function exCandidate(c) {
  const prompt = el('textarea', { 'aria-label': 'question' });
  prompt.value = c.prompt;
  const ref = el('textarea', { 'aria-label': 'reference answer', style: 'min-height:50px' });
  ref.value = c.reference;
  const reason = el('input', { type: 'text', placeholder: 'reason (for reject)',
    style: 'flex:1;min-width:160px', 'aria-label': 'reject reason' });
  return el('div', { class: 'rv', 'data-candidate': c.cid },
    el('div', { class: 'mhead' }, el('h3', { text: c.topic }),
      el('span', { class: 'small', text: `drafted by ${c.drafted_by || '—'} · batch ${(c.batch_id || '').slice(0, 14)}`
        + (c.provisional ? ` · provisional: ${c.provisional_reason}` : '') })),
    c.notes ? el('p', { class: 'small', text: 'drafter\'s note: ' + c.notes }) : '',
    el('div', { class: 'dxh', text: 'Question' }), prompt,
    el('div', { class: 'dxh', text: 'Reference (what a full-marks answer must contain)' }), ref,
    el('p', { class: 'small', text: 'Edit freely — the text you accept is what gets hashed, and '
      + 'the hash decides which half it lands in. You will not see it again if it lands in '
      + 'the hidden half.' }),
    el('div', { class: 'frm' },
      el('button', { class: 'primary', text: 'Accept into the bank', onclick: () => exPost(
        `api/exam/candidates/${c.cid}/accept`, { approver: whoName(), prompt: prompt.value,
                                                 reference: ref.value }) }),
      reason,
      el('button', { text: 'Reject', onclick: () => exPost(
        `api/exam/candidates/${c.cid}/reject`, { approver: whoName(), reason: reason.value }) })));
}

function vExam() {
  if (!state.ex.loaded && netReady()) loadExam();
  rememberedName();
  const st = state.ex.status || {};
  const sum = st.summary || {};
  const topics = Object.keys(sum);
  const total = topics.reduce((a, t) => a + (sum[t].accepted || 0), 0);
  const pending = topics.reduce((a, t) => a + (sum[t].pending || 0), 0);
  const head = el('div', { class: 'card' },
    el('h2', {}, 'Exam', infoTip('The instrument. One question bank across the topics; a person '
      + 'imports a bank, or the AI drafts candidates and a person accepts, edits or rejects each '
      + 'one here. Every accepted question is split by the hash of its text into a hidden half — '
      + 'the published per-topic score — and a diagnose half — the only half a proposal may '
      + 'read. That split is what lets the loop train on what the exam finds and still have an '
      + 'honest number.')),
    el('div', { class: 'kvs' },
      el('span', {}, el('b', { text: 'exam writer ' }), st.configured
        ? `${st.provider}/${st.model || '—'}` : 'not configured'),
      el('span', {}, el('b', { text: 'bank ' }), `${total} questions across ${topics.filter(t => sum[t].accepted).length} of ${topics.length} topics`),
      el('span', {}, el('b', { text: 'awaiting curation ' }), String(pending)),
      el('span', {}, el('b', { text: 'tasks built ' }), String((st.tasks_built || []).length))),
    !st.configured && st.reason ? el('p', { class: 'warn', text: st.reason }) : '',
    // an empty bank is not a page bug, but the page is where someone finds
    // out about it, so it says which of the three ways in they want
    total ? '' : el('p', { class: 'warn', 'data-bank': 'empty' },
      el('b', { text: 'No questions yet. ' }),
      'Import a person\'s bank below, or ask whoever runs the server to have the exam writer '
      + 'draft candidates — they appear under "Awaiting curation" within a poll.'),
    // an import or an accepted question makes itself sittable; this shows only
    // when the bank got ahead of the harness anyway (a judged run was sitting)
    st.tasks_stale ? el('div', { class: 'frm', style: 'margin-top:8px' },
      actButton('exbuild', 'Make new questions sittable', async () => {
        const j = await post('api/exam/build');
        const ts = Object.entries(j.tasks || {});
        state.ex.loaded = false; loadExam();
        return { key: 'build', toast: `New questions can be sat now — ${ts.length} `
          + `topic${ts.length === 1 ? '' : 's'} rebuilt.` };
      }, { class: 'primary', title: 'the bank has questions a judged run would not ask yet' }),
      el('span', { class: 'small se', text: 'the bank has questions a judged run would not ask yet' }))
      : '',
    actNote('exbuild'),
    state.ex.msg ? el('p', { class: 'small', text: state.ex.msg }) : '');
  const cands = state.ex.candidates;
  const cur = el('div', { class: 'card' },
    el('h2', { text: 'Awaiting curation' + (state.ex.topic ? ` — ${state.ex.topic}` : '') }),
    el('p', { class: 'sub', text: 'Read each against the rubric: does it ask for understanding, is '
      + 'the reference the substance rather than a wording, is it answerable in five sentences, '
      + 'is it new? Accept, edit and accept, or reject with a reason. Click a topic in the table '
      + 'above to show only its questions.' }),
    cands == null ? skeleton(3, { 'data-loading': 'candidates' })
      : cands.length ? cands.slice(0, 40).map(exCandidate)
      : empty('Nothing waiting' + (state.ex.topic ? ' in this topic.' : '.')
          + ' Drafted questions appear here to accept or reject; a person\'s bank is imported whole above.'),
    cands && cands.length > 40
      ? el('p', { class: 'small', text: `${cands.length - 40} more after these.` }) : '');
  return [head, exImport(), exRubrics(), cur];
}

// ---------------------------------------------------------------------------
// A person delivers a bank from here: a file, a topic, their name. Two steps,
// always — a preview that writes nothing, then a commit. The preview shows a
// diagnose-half question in full and a report-half one as its qid and
// metadata: he wrote them, and the page still does not echo them back.
// ---------------------------------------------------------------------------

function eximpState() {
  return state.eximp || (state.eximp = { topic: '', source: '', author: '', file: '',
                                         name: '', fileName: '', preview: null, msg: '',
                                         busy: false });
}

function exImport() {
  const s = eximpState();
  // the topic list is categories.yaml's, the same spine the bank uses
  const topicSel = Combobox('topic', topicGroups(Object.values((DATA.judged || {}).topics || {})),
    s.topic || '', v => { s.topic = v; s.preview = null; render(); },
    { key: 'import-topic', placeholder: 'topic…' });
  const fileIn = el('input', { type: 'file', accept: '.json,application/json',
    'aria-label': 'questions file', 'data-keep': 'import-file', onchange: async e => {
      const f = e.target.files[0]; if (!f) return;
      s.file = await f.text(); s.name = `${f.name} — ${(f.size / 1024).toFixed(0)} KB`;
      s.fileName = f.name;
      // the file's own name, never the topic's: "economics" in the source
      // column of the economics bank says nothing about where it came from
      s.source = s.source || f.name.replace(/\.json$/, '');
      // a new file is a new attempt: the last answer does not apply to it
      s.preview = null; actState('eximport').ok = actState('eximport').err = '';
      render(); } });
  // the source is the file's own name, shown as text; "change" opens a box.
  // Last night's two wrong sources came from typing the topic into it
  const srcIn = s.editSource
    ? el('input', { type: 'text', value: s.source, 'aria-label': 'source', 'data-keep': 'import-source',
        oninput: e => { s.source = e.target.value; } })
    : el('span', { class: 'fld-text', 'data-source': s.source || '' },
        s.source || el('span', { class: 'se', text: 'the file\'s name, once chosen' }), ' ',
        s.source ? el('button', { class: 'quiet', 'data-source-change': '1', text: 'change',
          onclick: () => { s.editSource = true; render(); } }) : '');
  // WHO WROTE THEM, which is not usually who is sitting here: the record has
  // to carry the author, or "these are Dr. Hossein's questions" lives only in
  // somebody's memory of the afternoon
  const authorIn = el('input', { type: 'text', placeholder: 'e.g. Dr. Hossein',
    value: s.author, 'aria-label': 'written by', 'data-keep': 'import-author',
    oninput: e => { s.author = e.target.value; } });
  // labels above the fields: a placeholder is not a label, and it was cut off
  const fld = (label, control) => el('label', { class: 'fld' },
    el('span', { class: 'fld-label', text: label }), control);
  // both buttons answer in the same place, so a new attempt cannot leave the
  // last one's success sitting above its refusal — that read as a partial
  // import when a wrapped file was rejected after a good one
  const ready = () => {
    if (!s.file || !s.topic) throw new Error('a file and a topic, please');
    if (!s.author.trim()) throw new Error('who wrote these questions? that name goes on '
      + 'every one of them');
    if (!whoName()) throw new Error(askName());
    return { topic: s.topic, approver: s.author, imported_by: whoName(),
             source: s.source, filename: s.fileName, text: s.file };
  };
  const p = s.preview;
  return el('div', { class: 'card', 'data-panel': 'import' },
    el('h2', {}, 'Import a bank', infoTip('A JSON array of questions written by a person — '
      + 'or an object holding one — read exactly as the command-line import reads it. The '
      + 'author\'s name goes on every question and the source says which file they came '
      + 'from; your own name (top right) is recorded as the person who imported them.')),
    el('p', { class: 'sub', text: 'Choose the file and the topic, and say who wrote it. '
      + 'Preview first: nothing is written until you commit.' }),
    el('div', { class: 'frm fields' }, fld('Questions file', fileIn), fld('Topic', topicSel),
      fld('Written by', authorIn), fld('Source', srcIn)),
    el('div', { class: 'frm' },
      actButton('eximport', 'Preview', async () => {
        s.preview = null;                       // never a stale table under a new answer
        const j = await post('api/exam/import/preview', ready());
        s.preview = j;
        return `Read ${j.imported + (j.updated || 0) + j.skipped + j.invalid} questions`
          + (j.wrapper ? ` from "${j.wrapper}" in the file` : ' from the file')
          + `: ${j.imported} new`
          + (j.updated ? `, ${j.updated} whose metadata this revises` : '')
          + (j.skipped ? `, ${j.skipped} already in the bank unchanged` : '')
          + (j.invalid ? `, ${j.invalid} unusable` : '')
          + '. Nothing is written yet.';
      }),
      p ? ((p.imported || p.updated)
        ? actButton('eximport', `Import ${p.imported + (p.updated || 0)} questions`,
            async () => {
              const body = ready();
              s.preview = null;
              const j = await post('api/exam/import', body);
              s.file = ''; s.name = ''; s.source = ''; s.editSource = false;
              state.ex.loaded = false; loadExam();
              state.loop.loaded = false;
              const b = j.build || {};
              return { key: 'import', toast: `Imported ${j.imported} question${j.imported === 1 ? '' : 's'}`
                  + (j.updated ? `, revised ${j.updated}` : '') + ` — ${j.topic}`
                  + (b.built ? '. They can be sat now.' : '.'),
                go: () => { const r = loopRowOf(slugOfTopic(j.topic)); if (r) loopGo(r, 'topic'); },
                link: 'the topic page',
                ...(b.built ? {} : { action: { label: 'Make these questions sittable',
                  run: async () => { await post('api/exam/build'); state.ex.loaded = false;
                                     loadExam(); toast('They can be sat now.'); } } }),
                line: `report ${j.report} / diagnose ${j.diagnose}`
                  + (j.skipped ? ` · ${j.skipped} already in the bank, unchanged` : '')
                  + (b.built ? '' : ` · ${b.why || 'not yet sittable'}`) };
            }, { 'data-commit': 'import', class: 'primary' })
        : el('button', { disabled: '', 'data-commit': 'import', text: 'Nothing new to import',
            title: 'every question in this file is already in the bank, unchanged' })) : ''),
    // the file that is loaded, which is not a result and does not replace one
    s.name ? el('p', { class: 'small se', 'data-import-file': '1',
                       text: `file: ${s.name}` }) : '',
    actNote('eximport'),
    p ? exImportPreview(p) : '');
}

function exImportPreview(p) {
  const rows = (p.items || []).slice(0, 12);
  return el('div', {},
    el('div', { class: 'kvs', 'data-preview': 'provenance' },
      el('span', {}, el('b', { text: 'written by ' }), p.approver || '—'),
      el('span', {}, el('b', { text: 'source ' }), p.source || '—'),
      p.wrapper ? el('span', {}, el('b', { text: 'read from ' }), `"${p.wrapper}" in the file`) : '',
      el('span', {}, el('b', { text: 'imported by ' }), whoName() || '—')),
    el('div', { class: 'kvs', 'data-preview': 'counts' },
      el('span', {}, el('b', { text: 'would import ' }), String(p.imported)),
      // a question already in the bank whose metadata this file revises: the
      // prompt is the identity, so it keeps its qid and its half
      p.updated ? el('span', {}, el('b', { text: 'would update ' }), String(p.updated)) : '',
      el('span', {}, el('b', { text: 'already in the bank, unchanged ' }), String(p.skipped)),
      el('span', {}, el('b', { text: 'unusable ' }), String(p.invalid)),
      el('span', {}, el('b', { text: 'split ' }), `report ${p.report} / diagnose ${p.diagnose}`)),
    Object.keys(p.acuity || {}).length ? el('p', { class: 'small', text: 'acuity — '
      + Object.entries(p.acuity).map(([k, v]) => `${k} ${v}`).join(', ') }) : '',
    Object.keys(p.intent || {}).length ? el('p', { class: 'small', text: 'intent — '
      + Object.entries(p.intent).map(([k, v]) => `${k} ${v}`).join(', ') }) : '',
    p.invalid ? el('p', { class: 'warn' }, el('b', { text: 'Unusable: ' }),
      (p.invalid_items || []).map(x => `#${x.index + 1}${x.id != null ? ` (id ${x.id})` : ''}`)
        .join(', ') + ' — no prompt, or shorter than 15 characters. They are skipped.') : '',
    el('p', { class: 'small', text: `${p.report} of these land in the hidden half and are not `
      + 'shown below, here or anywhere else — that is the split, and it applies to a bank you '
      + 'wrote yourself.' }),
    el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
      el('thead', {}, el('tr', {}, el('th', { text: 'half' }), el('th', { text: 'question' }),
        el('th', { text: 'reference (its metadata)' }))),
      el('tbody', {}, rows.map(it => el('tr', { 'data-half': it.half },
        el('td', {}, el('span', { class: 'badge' + (it.half === 'report' ? '' : ' instruct'),
                                  text: it.half })),
        el('td', {}, it.prompt || el('span', { class: 'se',
          text: `withheld · ${it.qid.slice(0, 12)}` }),
          it.change === 'updated' ? el('span', { class: 'badge', text: 'revised' }) : ''),
        el('td', { class: 'se', text: it.reference })))))),
    (p.items || []).length > 12 ? el('p', { class: 'small',
      text: `${p.items.length - 12} more not shown.` }) : '');
}

// ---------------------------------------------------------------------------
// The rubric and the criteria file that grade each topic: what the judge
// would use right now, and how to replace them. Committing changes a sha
// that is recorded in every judge.json, so the page says what that costs.
// ---------------------------------------------------------------------------

// A rubric heading need not carry "(version N)" — the author's own do not,
// and the sha is the identity. "v?" read like an error.
const rubricVersion = v => (!v || v === '?') ? 'no version' : 'v' + v;

// what a topic's bank holds, beside the files that would grade it
// 11m: and opens it. masein came to the Exam tab to read the questions, the
// obvious place, and found the count as plain text; the reader was only on
// the topic page. Only the practice half opens — the hidden half never does
function bankCell(b, topic) {
  if (!b || !b.accepted) return el('span', { class: 'warn', 'data-bank': '0',
    text: 'no questions yet' });
  const slug = slugOfTopic(topic);
  const read = (text, attrs) => slug ? readLink({ kind: 'bank', id: slug }, text, attrs) : text;
  return el('span', { 'data-bank': String(b.accepted) },
    read(String(b.accepted), { 'data-read-bank-count': slug,
      title: 'read the practice questions' }), ' — ',
    el('span', { class: 'se', text: `${b.report} hidden / ${b.diagnose} practice` }),
    slug ? [' · ', read('read the practice half', { 'data-read-bank': slug })] : '',
    b.pending ? el('span', { class: 'se', text: ` · ${b.pending} awaiting curation` }) : '');
}

function exRubrics() {
  const st = state.exrub || (state.exrub = { rows: null, open: '', kind: 'rubric',
                                             content: '', name: '', preview: null, msg: '' });
  if (st.rows == null && netReady()) loadRubrics();
  const rows = st.rows || [];
  return el('div', { class: 'card', 'data-panel': 'rubrics' },
    el('h2', {}, 'Rubrics and criteria', infoTip('What the judge grades each topic with right '
      + 'now. A topic without a rubric of its own uses the shared one. A criteria file turns '
      + 'the 0–4 into a fold of per-criterion scores. Changing either file changes its sha256, '
      + 'which is recorded in every judge.json: scores from before and after are not '
      + 'comparable, so re-sit the topic after a change.')),
    st.rows == null ? '' : el('div', { class: 'toolbar' },
      el('input', { type: 'search', placeholder: 'find a topic or file', 'aria-label': 'find a rubric',
        'data-keep': 'rubric-q', value: st.q || '', style: 'flex:1;min-width:160px',
        oninput: e => { st.q = e.target.value; render(); } })),
    st.rows == null ? skeleton(5, { 'data-loading': 'rubrics' })
      : el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
        el('thead', {}, el('tr', {}, el('th', { text: 'topic' }), el('th', { text: 'bank' }),
          el('th', { text: 'rubric' }),
          el('th', { text: 'criteria' }), el('th', { text: 'files' }))),
        el('tbody', {}, rubRows(rows, st).map(r => r.fold ? r.fold : r.error
          ? el('tr', { 'data-rubric-row': r.topic, 'data-rubric-error': '1' },
              el('td', {}, r.topic),
              el('td', { class: 'warn', colspan: '4' }, r.error))
          : el('tr', { 'data-rubric-row': r.topic,
              class: state.ex.topic === r.topic ? 'domrow' : null },
          el('td', {}, el('a', { href: '#', text: r.topic, 'data-filter-topic': r.topic,
            title: 'show only this topic\'s questions awaiting curation',
            onclick: e => { e.preventDefault();
              state.ex.topic = state.ex.topic === r.topic ? '' : r.topic; state.ex.loaded = false;
              // drop the list with the filter: another topic's questions under
              // this topic's heading, until the fetch lands, is a lie
              state.ex.candidates = null; render(); } })),
          // having a rubric is not having questions: three topics shipped
          // with both files and an empty bank, and looked ready
          el('td', { class: 'small' }, bankCell((st.banks || {})[r.topic], r.topic)),
          el('td', { title: `sha256 ${r.sha256}` }, r.fallback
            ? el('span', { 'data-fallback': '1',
                title: `this topic has no rubric of its own; the shared ${r.name}.md grades it`
                  + ` — sha256 ${r.sha256}`, text: 'shared rubric ' })
            : `${r.name}.md `,
            r.fallback ? '' : r.version === '?'
              ? el('span', { class: 'se', 'data-no-version': '1',
                  title: 'the author\'s own heading carries no version; the sha is the identity',
                  text: rubricVersion(r.version) + ' ' })
              : `v${r.version} `,
            r.status === 'draft' ? el('span', { class: 'badge taint', text: 'DRAFT' }) : ''),
          el('td', {}, r.scoring === 'criteria'
            ? el('span', { title: `sha256 ${r.criteria_sha256}` }, `${r.criteria_count} criteria `,
                r.criteria_status === 'draft'
                  ? el('span', { class: 'badge taint', text: 'DRAFT' }) : '')
            : el('span', { class: 'se', text: 'one overall score' })),
          el('td', {},
            // 11m: the questions first, beside the files that grade them
            ((st.banks || {})[r.topic] || {}).accepted && slugOfTopic(r.topic)
              ? [readLink({ kind: 'bank', id: slugOfTopic(r.topic) }, 'questions',
                  { 'data-read-questions': slugOfTopic(r.topic) }), ' · '] : '',
            // 11g: read them in the page; the download stays beside each
            readLink({ kind: 'rubric', id: r.name }, 'rubric', { 'data-read-rubric': r.name }),
            el('a', { class: 'dllink', href: `api/exam/rubrics/${r.name}`, download: `${r.name}.md`,
              'aria-label': `download ${r.name}.md`, title: 'download', text: ' ↓' }),
            ' · ',
            r.scoring === 'criteria'
              ? [readLink({ kind: 'criteria', id: r.name }, 'criteria', { 'data-read-criteria': r.name }),
                 el('a', { class: 'dllink', href: `api/exam/rubrics/${r.name}?kind=criteria`,
                   download: `${r.name}.criteria.json`, 'aria-label': `download ${r.name}.criteria.json`,
                   title: 'download', text: ' ↓' })]
              : el('span', { class: 'se', text: '—' }),
            ' · ',
            el('a', { href: '#', text: 'replace', onclick: e => { e.preventDefault();
              st.open = st.open === r.name ? '' : r.name; st.name = r.name;
              st.preview = null; st.content = ''; render(); } }))))))),
    st.rows == null ? '' : st.pager || '',
    st.open ? exRubricUpload(st) : '',
    (st.changes || []).length ? el('p', { class: 'small', text: 'last change: '
      + st.changes.map(c => `${c.name}.${c.kind === 'criteria' ? 'criteria.json' : 'md'} by `
        + `${c.approver} ${rel(c.changed_at)} ago`).slice(0, 3).join(' · ') }) : '');
}

// topics with questions first; the rest fold into one row that opens. The
// same shared rubric used to be printed ten times over for empty topics.
function rubRows(rows, st) {
  const has = r => ((st.banks || {})[r.topic] || {}).accepted;
  // thirty-six topics: a search over the topic and its file's name, and the
  // shared pager
  const q = (st.q || '').trim().toLowerCase();
  const hit = r => !q || r.topic.toLowerCase().includes(q) || (r.name || '').toLowerCase().includes(q);
  const pg = paged('rubrics', rows.filter(r => (r.error || has(r)) && hit(r)), q, render, 25);
  st.pager = pg.pager;
  const full = pg.rows, empty = rows.filter(r => !r.error && !has(r) && hit(r));
  if (!empty.length) return full;
  const fold = el('tr', { 'data-empty-topics': String(empty.length) },
    el('td', { colspan: '5', class: 'small' },
      el('b', { text: `${empty.length} topic${empty.length > 1 ? 's' : ''} without questions: ` }),
      empty.map(r => r.topic).join(', '), ' ',
      el('button', { class: 'quiet', 'data-show-empty': '1',
        text: st.showEmpty ? 'hide them' : 'show them',
        onclick: () => { st.showEmpty = !st.showEmpty; render(); } })));
  return [...full, { fold }, ...(st.showEmpty ? empty : [])];
}

function exRubricUpload(st) {
  const slot = 'exrubric';
  const body = () => {
    if (!st.content) throw new Error('choose a file first');
    if (!whoName()) throw new Error(askName());
    return { name: st.name, kind: st.kind, content: st.content,
             approver: whoName(), note: st.note || '' };
  };
  const p = st.preview;
  return el('div', { style: 'margin-top:10px', 'data-upload': st.name },
    el('div', { class: 'frm' },
      Select('which file', [['rubric', `${st.name}.md (prose)`],
        ['criteria', `${st.name}.criteria.json`]], st.kind || 'rubric',
        v => { st.kind = v; st.preview = null; render(); }, { key: 'upload-' + st.name }),
      el('input', { type: 'file', accept: '.md,.json,text/markdown,application/json',
        'aria-label': 'new file', onchange: async e => {
          const f = e.target.files[0]; if (!f) return;
          st.content = await f.text(); st.preview = null;
          st.file = `${f.name} — ${(f.size / 1024).toFixed(1)} KB`;
          actState(slot).ok = actState(slot).err = '';   // a new file, a new answer
          render(); } }),
      el('input', { type: 'text', placeholder: 'note (why)', 'aria-label': 'note',
        'data-keep': 'rubric-note', oninput: e => { st.note = e.target.value; } }),
      rvNameInput(),
      actButton(slot, 'Check it', async () => {
        st.preview = null;
        const j = await post('api/exam/rubrics/preview', body());
        st.preview = j;
        return j.ok
          ? (j.changed ? 'Valid, and different from the file in use — read the diff below '
                       + 'before replacing it.'
                       : 'Valid, and identical to the file in use. Nothing to replace.')
          : `${j.problems.length} problem${j.problems.length > 1 ? 's' : ''} — `
            + 'the file cannot be saved until they are fixed. Listed below.';
      }),
      p && p.ok ? actButton(slot, 'Replace the file', async () => {
        const send = body();
        st.preview = null;
        const j = await post('api/exam/rubrics', send);
        st.content = ''; st.file = ''; st.rows = null; loadRubrics();
        return `Written to ${j.written}. Its sha is recorded in every judge.json from now `
          + 'on — re-run suite=judged for this topic.';
      }, { 'data-commit': 'rubric' }) : ''),
    st.file ? el('p', { class: 'small se', 'data-rubric-file': '1', text: `file: ${st.file}` }) : '',
    actNote(slot),
    p ? el('div', {},
      p.problems.length ? el('div', { class: 'warn', 'data-problems': String(p.problems.length) },
        el('b', { text: `${p.problems.length} problem${p.problems.length > 1 ? 's' : ''}: ` }),
        p.problems.join('; ')) : el('p', { class: 'note', text: 'Valid.' }),
      p.ok && p.changed ? el('p', { class: 'warn', 'data-sha-warning': '1' },
        el('b', { text: 'This changes the file\'s sha256. ' }), p.warning) : '',
      p.ok && !p.changed ? el('p', { class: 'small', text: 'Identical to the file in use.' }) : '',
      el('pre', { class: 'mono', style: 'max-height:260px;overflow:auto;font-size:12px',
                  text: (p.diff || []).join('\n') || '(no difference)' })) : '');
}

async function loadRubrics() {
  const st = state.exrub;
  try {
    const j = await api('api/exam/rubrics');
    Object.assign(st, { rows: j.topics, store: j.store, note: j.note, changes: j.changes,
                        banks: j.banks || {} });
    if (state.tab === 'exam' && !state.model) render();
  } catch (e) { /* netFail said so */ }
}

function download(name, mime, text) {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type: mime })), download: name });
  document.body.append(a); a.click(); a.remove();
}
// The flat CSV joins each row's model-level provenance, so an exported number
// carries the conditions that produced it: a value without its template policy,
// dtype, seed and harness build is not reproducible and should not be quoted.
const PROV_COLS = ['result_class', 'required_done', 'template_applied', 'template_id',
                   'template_policy', 'model_code', 'active_params', 'dtype',
                   'backend', 'batch_size', 'seed', 'limit', 'harness_git',
                   'transformers', 'eval_finished'];
function provOf(name) {
  const m = DATA.models.find(x => x.name === name) || {};
  const a = m.archinfo || {};
  return [m.official ? 'official' : 'preliminary',
          m.nreq ? `${m.nhave}/${m.nreq}` : '', m.chat ? 'yes' : 'no',
          a.tmpl_sha || '', m.kindReason || '', (a.code_sha || []).join(' ') || 'library',
          a.active_params || '', m.dtype || '', m.backend || '',
          m.batch == null ? '' : m.batch, m.seed == null ? '' : m.seed,
          m.limit == null ? 'full' : m.limit, m.hash || '',
          DATA.meta.transformers || '', m.date || ''];
}
function exportCsv() {
  const esc = v => v == null ? '' : /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : String(v);
  const lines = [['model', 'task', 'metric', 'value', 'stderr', 'n_shot', 'n_samples',
                  ...PROV_COLS].join(',')];
  for (const r of DATA.extra) lines.push([...r, ...provOf(r[0])].map(esc).join(','));
  download('benchmark.csv', 'text/csv', lines.join('\n'));
}
function exportJson() { download('benchmark.json', 'application/json', JSON.stringify(DATA, null, 1)); }

// ---------- shell ----------
// Tab order is how often each is opened, and every id is the label's own
// slug — a tab called Evals whose hash said `runs` and whose heading said
// "Run provenance" was three names for one thing.
const TABS = [
  ['overview', 'Overview', vOverview],
  // the loop is what this server is for, so it sits where the eye lands
  ...(LIVE ? [['loop', 'Loop', vLoop]] : []),
  ['models', 'Models', vModels],
  ['leaderboard', 'Leaderboard', vLeaderboard],
  ...(LIVE ? [['queue', 'Queue', vQueue]] : []),
  // one click further, under More: Exam and Review are steps of the loop and
  // are reached from it; the rest are for the people who go looking
  ...(LIVE ? [['exam', 'Exam', vExam, 'more'],
              ['review', 'Review', vReview, 'more'],
              ['training', 'Training', vTraining, 'more']] : []),
  ['tasks', 'Tasks', vTasks, 'more'],
  ['perplexity', 'Perplexity & Loss', vPpl, 'more'],
  ['provenance', 'Provenance', vRuns, 'more'],
];

// Old hashes keep working: a link someone pasted into a message last month
// should still land, and silently landing on Overview instead is the worst
// of the three possible behaviours.
const TAB_ALIASES = {
  runs: 'provenance', evals: 'provenance', submit: 'queue', 'submit-queue': 'queue',
  models_tab: 'models', ppl: 'perplexity', 'perplexity-loss': 'perplexity',
};
// 11b: the cards of a tab are its sections, numbered 01, 02, … in the order
// they are read. The index is drawn from the DOM rather than written into
// twenty view functions, so a card that moves takes its place in the count.
function numberSections() {
  const view = document.getElementById('view');
  if (!view) return;
  let n = 0;
  for (const card of view.querySelectorAll(':scope > .card')) {
    const h2 = card.querySelector(':scope > h2, :scope > .sechead > h2');
    if (!h2 || !h2.textContent.trim()) continue;
    // the index is drawn, not written into the heading: a section's name is
    // its own, and a screen reader reads "Top models", not "02 Top models"
    h2.dataset.ix = String(++n).padStart(2, '0');
  }
}

function render() {
  // full rebuild: drop the in-place refreshers so a poll can never touch the
  // DOM of a tab that just got torn down — the mounted tab re-registers its own
  state.trRedraw = state.queueRedraw = null;
  renderWarnings();                 // one line, in the bar, on every tab
  const ms = visible();
  renderTabs();
  // 11b: the hero belongs to Overview. Every other tab starts straight at its
  // first numbered section.
  const hero = document.getElementById('pagehero');
  if (hero) hero.hidden = !!(state.model || state.topic || state.tab !== 'overview');
  const view = document.getElementById('view');
  view.classList.remove('dimmed');
  // a poll rebuilds the view every few seconds. Whatever the person is typing
  // in — and where their caret is — comes back afterwards, or the field is
  // unusable on a live page: this is the same bug as the tab bar's, one layer
  // down, and the cure is the same one (never lose what the DOM was holding).
  // 11f: did this render change the view (a tab, a model, a topic)? Only a
  // navigation moves the scroll or plays the entrance; a poll never does
  const vk = viewKey(), changed = _lastView != null && vk !== _lastView;
  const nav = _navigated && changed;
  _lastView = vk; _navigated = false;
  const aimed = !!(state.after && state.after.scroll);
  // a value a poll changed is marked for a moment: what the view said before
  const was = changed ? null : snapWatch(view);
  const live = document.activeElement;
  const keep = live && live.dataset && live.dataset.keep && view.contains(live)
    ? { key: live.dataset.keep, value: live.value,
        start: live.selectionStart, end: live.selectionEnd } : null;
  const hkeep = [...view.querySelectorAll('[data-hkeep]')]
    .map(e => [e.dataset.hkeep, e.scrollLeft]).filter(([, x]) => x > 0);
  if (state.model) view.replaceChildren(...vModel());
  else if (state.topic) view.replaceChildren(...vTopic());
  else view.replaceChildren(...TABS.find(([id]) => id === state.tab)[2](ms));
  if (keep) {
    const again = view.querySelector(`[data-keep="${keep.key}"]`);
    if (again) {
      if (again.type !== 'file') again.value = keep.value;   // a file input's is not settable
      again.focus();
      try { again.setSelectionRange(keep.start, keep.end); } catch (e) { /* not a text field */ }
    }
  }
  // a button that carries its context: scroll to what it was about, and put
  // the caret where the next keystroke goes
  if (state.after) {
    const a = state.after;
    state.after = null;
    if (a.scroll) settleOn(a.scroll);
    requestAnimationFrame(() => {
      const t = a.scroll && document.querySelector(a.scroll);
      if (t) t.scrollIntoView({ block: 'start' });
      const f = a.focus && document.querySelector(a.focus);
      if (f) f.focus();
    });
  } else if (_settle) {
    requestAnimationFrame(settleAgain);
  }
  hfadeUpdate(view);                     // decides which boxes scroll…
  for (const [k, x] of hkeep) {          // …so the old sideways scroll can come back
    const e = view.querySelector(`[data-hkeep="${k}"]`);
    if (e) e.scrollLeft = x;
  }
  hfadeUpdate(view);
  if (was) markChanged(view, was);
  if (nav) {
    view.classList.remove('view-enter');
    void view.offsetWidth;                     // restart the entrance
    view.classList.add('view-enter');
    view.addEventListener('animationend', function done(ev) {
      if (ev.target !== view) return;
      view.classList.remove('view-enter'); view.removeEventListener('animationend', done); });
  }
  if (_restore) {
    const r = _restore;
    scrollTo(0, r.y);
    if (Math.abs(scrollY - r.y) < 2 || Date.now() > r.until) _restore = null;
  } else if (changed && !aimed) scrollTo(0, 0);
  numberSections();
  if (state.model) watchSections();
  // an open popover keeps its panel, its scroll and its focus across a render;
  // only its button is a new node
  popReanchor();
  renderReader();
}

// 11f: a value a poll changed — a score, a status, a count — is washed for a
// moment. What the view said before, keyed by data-watch, against what it says now
const snapWatch = root => new Map([...root.querySelectorAll('[data-watch]')]
  .map(e => [e.dataset.watch, e.textContent]));
function markChanged(root, was) {
  if (!was || !was.size) return;
  for (const e of root.querySelectorAll('[data-watch]')) {
    const before = was.get(e.dataset.watch);
    if (before == null || before === e.textContent) continue;
    e.classList.add('changed');
    e.addEventListener('animationend', () => e.classList.remove('changed'), { once: true });
  }
}

// how long a row or card a button took you to stays marked
const LANDED_MS = 6000;

// A button that scrolls to a panel usually lands before the panel's data
// does: the page is short, the scroll stops early, and when the answers
// arrive and the page grows the browser puts the old offset back — with 37
// topics, "Read the results" from low on the board landed past the answers.
// So for a few seconds after, every render puts the panel back where the
// button put it, until the person scrolls or types for themselves.
let _settle = null;
const SETTLE_MS = 4000;

function settleOn(sel) {
  _settle = { sel, until: Date.now() + SETTLE_MS };
  const stop = () => { _settle = null; };
  for (const ev of ['wheel', 'touchstart', 'keydown', 'mousedown'])
    window.addEventListener(ev, stop, { once: true, passive: true, capture: true });
}

function settleAgain() {
  if (!_settle || Date.now() > _settle.until) { _settle = null; return; }
  const t = document.querySelector(_settle.sel);
  if (!t) return;
  const top = t.getBoundingClientRect().top;
  if (top < -2 || top > innerHeight / 2) t.scrollIntoView({ block: 'start' });
}

// The tab bar is the one thing on the page that must survive a render. A
// poll, a finished fetch and a click all call render(); replacing the buttons
// each time hands whoever is mid-click a node that is no longer in the
// document, and the click goes nowhere. Build them once, then only move the
// selection. Six tabs; the rest under More ▾, a real menu.
let _tabsBuilt = false;

// ---------------------------------------------------------------------------
// One popover, for every menu on the page.
//
// The More ▾ menu lived inside #tabs, which scrolls sideways on a phone
// (overflow-x:auto), so the panel was clipped to the height of the tab strip:
// 37 px, with almost nothing in it reachable. A panel that has to escape its
// scroller belongs on document.body, placed from its button's rect. So this
// is the component for More ▾, Theme ▾ and the name menu — and for the
// Leaderboard's popovers later.
//
// It survives a render because the panel is NOT inside the view: render()
// rebuilds #view and the header, and the panel keeps its DOM, its scroll and
// its focus. Only the button is found again, by data-pop-anchor.
// ---------------------------------------------------------------------------

const POP = { key: null, panel: null, anchor: null, opts: null, build: null };
const POP_EDGE = 8;             // never closer than this to the window's edge

function popItems() {
  return POP.panel ? [...POP.panel.querySelectorAll(
    '[role=menuitem]:not([hidden]):not([disabled]),[role=menuitemradio]:not([hidden])')] : [];
}

function popPlace() {
  const { panel, anchor, opts } = POP;
  if (!panel || !anchor || !anchor.isConnected) return;
  const r = anchor.getBoundingClientRect();
  // its button has scrolled out of the window: there is nothing to hang from
  if (r.bottom < 0 || r.top > innerHeight) { popClose(); return; }
  panel.style.maxHeight = '';
  // never wider than the window it has to sit 8px inside
  panel.style.maxWidth = `${Math.max(160, innerWidth - 2 * POP_EDGE)}px`;
  const pr = panel.getBoundingClientRect();
  const below = innerHeight - r.bottom - 4, above = r.top - 4;
  const flip = pr.height > below && above > below;
  const room = Math.max(120, (flip ? above : below) - POP_EDGE);
  const top = flip ? Math.max(POP_EDGE, r.top - 4 - Math.min(pr.height, room)) : r.bottom + 4;
  let left = (opts || {}).placement === 'bottom-end' ? r.right - pr.width : r.left;
  left = Math.min(Math.max(POP_EDGE, left), Math.max(POP_EDGE, innerWidth - POP_EDGE - pr.width));
  panel.classList.toggle('flip', flip);        // it comes in from the side away from its button
  panel.style.top = `${Math.min(top, innerHeight - POP_EDGE - Math.min(pr.height, room))}px`;
  panel.style.left = `${left}px`;
  panel.style.maxHeight = `${room}px`;
}

// 11k: a mouse click left a thick ring on the checks pill, because closing
// a popover gives focus back to its button and Chrome calls that focus
// visible. The ring is for people who are moving by keyboard: we remember
// which it was, and mark the button so the ring stays off for a pointer.
let _byPointer = false;
addEventListener('pointerdown', () => { _byPointer = true; }, true);
addEventListener('keydown', () => { _byPointer = false; }, true);
function refocus(el) {
  if (!el || !el.isConnected) return;
  if (_byPointer) {
    el.dataset.noring = '1';
    const off = () => { delete el.dataset.noring;
      el.removeEventListener('blur', off); el.removeEventListener('keydown', off); };
    el.addEventListener('blur', off); el.addEventListener('keydown', off);
  }
  el.focus();
}

function popClose(backToButton = false) {
  const { panel, anchor } = POP;
  if (!panel) return;
  // 11f: it leaves in half the time it came. What fades is a ghost — no
  // key, no ids, inert — so nothing can find or click a closed panel
  if (motionOff()) panel.remove();
  else {
    panel.removeAttribute('data-pop');
    for (const e of [panel, ...panel.querySelectorAll('[id]')]) e.removeAttribute('id');
    for (const a of ['role', 'aria-label']) panel.removeAttribute(a);
    panel.setAttribute('aria-hidden', 'true');
    panel.inert = true;
    panel.classList.add('pop-out');
    setTimeout(() => panel.remove(), 150);
  }
  if (anchor && anchor.isConnected) anchor.setAttribute('aria-expanded', 'false');
  POP.key = POP.panel = POP.anchor = POP.opts = POP.build = null;
  if (backToButton) refocus(anchor);
}

function popOpen(key, anchor, panel, opts = {}) {
  const again = POP.key === key;
  popClose();
  if (again) return;                       // a second click on the button closes it
  POP.key = key; POP.panel = panel; POP.anchor = anchor; POP.opts = opts;
  panel.dataset.pop = key;
  panel.classList.add('pop');
  if (!motionOff()) panel.classList.add('pop-in');
  if (opts.menu !== false) panel.setAttribute('role', 'menu');
  document.body.append(panel);
  anchor.setAttribute('aria-expanded', 'true');
  popPlace();
  requestAnimationFrame(() => requestAnimationFrame(() => panel.classList.remove('pop-in')));
  const first = (opts.focus && panel.querySelector(opts.focus))
    || panel.querySelector('[aria-current],[aria-checked=true]') || popItems()[0]
    || panel.querySelector('input,select,textarea,button');
  if (first) first.focus();
}

// a button that owns a popover: the ARIA menu-button pattern, and the key
// the panel is found again by after a render
function popover(btn, build, opts = {}) {
  const key = opts.key || btn.id || 'pop';
  btn.setAttribute('aria-haspopup', opts.menu === false ? 'dialog' : 'menu');
  btn.setAttribute('aria-expanded', String(POP.key === key));
  btn.setAttribute('aria-controls', 'pop-' + key);
  btn.dataset.popAnchor = key;
  btn._popBuild = build;
  const open = () => { popOpen(key, btn, build(), opts); if (POP.key === key) POP.build = build; };
  btn.addEventListener('click', e => { e.preventDefault(); open(); });
  btn.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
      if (POP.key === key) return;
      e.preventDefault();
      // …and this keystroke opened the menu: it does not also move inside it
      e.stopPropagation();
      open();
    }
  });
  return btn;
}

// after every render: the button is a new node, the panel is not
function popReanchor() {
  if (!POP.key) return;
  const a = document.querySelector(`[data-pop-anchor="${POP.key}"]`);
  if (!a) { popClose(); return; }          // its button is gone: so is the menu
  POP.anchor = a;
  if (a._popBuild) POP.build = a._popBuild;
  a.setAttribute('aria-expanded', 'true');
  // a panel whose choices apply at once (Columns) shows what is true now.
  // 11e: when it holds the same controls, their state is patched IN PLACE, so
  // the control that has focus is still the same node and keeps it; only a
  // different set of controls (another chip's) is built again
  if ((POP.opts || {}).rebuild && POP.build) {
    const fresh = POP.build();
    const sig = root => [...root.querySelectorAll('input, button')].map(e => e.tagName + ':'
      + (e.dataset.column || e.dataset.filter || e.dataset.tint || e.dataset.showAll
         || e.dataset.columnGroupAll || e.dataset.columnGroupNone || e.textContent)).join('|');
    if (sig(POP.panel) === sig(fresh)) {
      const now = [...fresh.querySelectorAll('input')];
      [...POP.panel.querySelectorAll('input')].forEach((e, i) => {
        if (e.type === 'checkbox') e.checked = now[i].checked; });
    } else {
      const f = document.activeElement;
      const key = f && POP.panel.contains(f) && (f.dataset.column || f.dataset.filter);
      POP.panel.replaceChildren(...fresh.childNodes);
      const again = key && POP.panel.querySelector(
        `[data-column="${CSS.escape(key)}"], [data-filter="${CSS.escape(key)}"]`);
      if (again) again.focus();
    }
  }
  popPlace();
}

document.addEventListener('mousedown', e => {
  if (!POP.panel) return;
  if (POP.panel.contains(e.target) || (POP.anchor && POP.anchor.contains(e.target))) return;
  popClose();
});
document.addEventListener('keydown', e => {
  if (!POP.panel) return;
  const list = popItems();
  const i = list.indexOf(document.activeElement);
  if (e.key === 'Escape') { e.preventDefault(); popClose(true); return; }
  if (e.key === 'Tab' && (POP.opts || {}).menu !== false) { popClose(); return; }
  if (!list.length || !POP.panel.contains(document.activeElement)) return;
  if (e.key === 'ArrowDown') { e.preventDefault(); list[(i + 1) % list.length].focus(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); list[(i - 1 + list.length) % list.length].focus(); }
  else if (e.key === 'Home') { e.preventDefault(); list[0].focus(); }
  else if (e.key === 'End') { e.preventDefault(); list[list.length - 1].focus(); }
});
// the page scrolling moves the button, so the panel follows; the panel's own
// scroll does not — measured at full height mid-scroll, it would flip above
addEventListener('scroll', e => {
  if (POP.panel && e.target instanceof Node && POP.panel.contains(e.target)) return;
  popPlace(); }, true);
addEventListener('resize', () => popPlace());

// ---------------------------------------------------------------------------
// Back to top (11b): a mono pill after two screens of scrolling. It moves
// focus to the bar, so the keyboard lands where the eye does.
// ---------------------------------------------------------------------------
function toTopWatch() {
  const btn = document.getElementById('toTop');
  if (!btn) return;
  btn.hidden = scrollY < innerHeight * 2;
}
addEventListener('scroll', toTopWatch, { passive: true });
addEventListener('resize', toTopWatch);
document.addEventListener('click', e => {
  if (!e.target.closest('#toTop')) return;
  scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
    ? 'auto' : 'smooth' });
  const bar = document.getElementById('bar');
  const first = bar && bar.querySelector('button, a, [tabindex]');
  if (first) first.focus({ preventScroll: true });
});

function moreMenu(open) {                  // kept for the call sites that toggle it
  const btn = document.getElementById('moreBtn');
  if (!btn) return;
  if (!open) popClose();
  else if (POP.key !== 'more') btn.click();
}

function renderTabs() {
  const tabs = document.getElementById('tabs');
  const main = TABS.filter(t => !t[3]), more = TABS.filter(t => t[3]);
  const go = id => navigate({ tab: id, model: null, topic: null });
  if (!_tabsBuilt) {
    _tabsBuilt = true;
    const moreBtn = el('button', { role: 'tab', id: 'moreBtn', 'data-tab': 'more' });
    // the panel is built on open and lives on the body (popover): inside the
    // tab strip, which scrolls sideways, it was clipped to 37 px
    popover(moreBtn, () => el('div', { class: 'moremenu', id: 'pop-more',
        'aria-label': 'more tabs' },
      more.map(([id, label]) => el('button', { role: 'menuitem', 'data-tab': id,
        'aria-current': (state.model ? '' : state.tab) === id ? 'page' : null,
        text: label, onclick: () => { popClose(); go(id); } }))),
      { key: 'more' });
    tabs.replaceChildren(...main.map(([id, label]) =>
      el('button', { role: 'tab', 'data-tab': id, onclick: () => go(id), text: label })),
      el('div', { class: 'morewrap' }, moreBtn),
      // 11f: one underline, which slides from the old tab to the new one
      el('span', { class: 'tab-ink still', id: 'tabInk', 'aria-hidden': 'true' }));
    window.addEventListener('resize', () => placeInk(true));
  }
  const sel = state.model ? '' : state.tab;
  for (const b of tabs.querySelectorAll('button[role=tab][data-tab]'))
    if (b.id !== 'moreBtn') b.setAttribute('aria-selected', String(b.dataset.tab === sel));
  const inMore = more.find(t => t[0] === sel);
  const moreBtn = document.getElementById('moreBtn');
  moreBtn.textContent = (inMore ? inMore[1] : 'More') + ' ▾';
  moreBtn.setAttribute('aria-selected', String(!!inMore));
  placeInk();
}

function placeInk(still) {
  const ink = document.getElementById('tabInk');
  const b = document.querySelector('#tabs [role=tab][aria-selected=true]');
  if (!ink) return;
  if (!b) { ink.style.opacity = '0'; return; }
  if (still) ink.classList.add('still');
  ink.style.opacity = '1';
  // measured against the strip itself: More ▾ sits inside a wrapper of its own
  const tabs = document.getElementById('tabs');
  const x = b.getBoundingClientRect().left - tabs.getBoundingClientRect().left + tabs.scrollLeft;
  ink.style.transform = `translateX(${Math.round(x)}px) scaleX(${b.offsetWidth})`;
  // the first placement (and a resize) jumps; every move after that slides
  if (ink.classList.contains('still'))
    requestAnimationFrame(() => requestAnimationFrame(() => ink.classList.remove('still')));
}

// The board's checks: one line on every tab — "6 checks · 3 about the
// judged suite" — never dismissed, never hidden, never a screenful of
// paragraphs above the thing the tab is named after. Open, each is one line
// with a severity dot, a "Show me" to the rows it concerns, and its full
// text behind a disclosure.
let _warnSig = null;

function showMe(show) {
  // 11l: a check about one model opens that model's page
  if (show.model) return navigate({ model: show.model, topic: null });
  if (show.tab === 'models')
    Object.assign(state.mdl, { kind: show.kind || 'all', taintedOnly: !!show.tainted,
                               prelimOnly: !!show.prelim, judgedOnly: false, q: '' });
  navigate({ tab: show.tab, model: null, topic: null });
}

function renderWarnings() {
  const box = document.getElementById('warnings');
  if (!box || !DATA) return;
  const cs = DATA.checks || (DATA.warnings || []).map(w => ({ key: '', severity: 'warning',
    short: w.split(/\. |—/)[0], text: w, show: null,
    judged: /judge|judged|rubric|criteria|canary|calibrat/i.test(w) }));
  // rebuilding this on every poll would snap shut a fold someone just opened,
  // and hand Playwright (and a mouse) a node that vanishes mid-click
  const sig = JSON.stringify(cs.map(c => c.text));
  if (sig === _warnSig) return;
  _warnSig = sig;
  if (!cs.length) { box.replaceChildren(); return; }
  // what kind, not just how many: after a judged run most of them are about
  // the judge, and "5 checks" says nothing about whether to open it
  const judged = cs.filter(c => c.judged).length;
  const worst = cs.some(c => c.severity === 'warning') ? 'warn' : 'info';
  const fold = el('details', { class: 'checks', 'data-warnings': 'collapsed',
      open: state.checksOpen ? '' : null,
      ontoggle: e => { state.checksOpen = e.target.open; } },
    // 11e: the pill says the count and nothing else — it is a button like its
    // neighbours; what kind of checks they are is the first line of the panel
    el('summary', { class: 'barpill', 'data-warn-summary': String(cs.length) },
      el('span', { class: 'dot ' + worst }),
      `${cs.length} check${cs.length > 1 ? 's' : ''} ▾`),
    el('ul', { class: 'checklist' },
      judged ? el('li', { class: 'small checks-judged', 'data-checks-judged': String(judged),
        text: `${judged} of ${cs.length} ${cs.length > 1 ? 'are' : 'is'} about the judged suite` }) : '',
      cs.map(c => el('li', { class: 'check warnrow',
        'data-check': c.key, 'data-severity': c.severity },
      el('span', { class: 'dot ' + (c.severity === 'warning' ? 'warn' : 'info'),
        title: c.severity === 'warning' ? 'warning' : 'for information' }),
      el('span', { class: 'check-short', text: c.short }),
      c.show ? el('a', { href: '#', class: 'small', 'data-show-me': c.key, text: 'Show me',
        onclick: e => { e.preventDefault(); showMe(c.show); } }) : '',
      el('details', { class: 'check-more' },
        el('summary', { class: 'small', text: 'why' }),
        el('p', { class: 'warn', text: c.text }))))));
  box.replaceChildren(fold);
}

// "live · refreshed 12:33" with a dot: green while the polls land, amber and
// "last update 3 min ago — retrying" once they stop. A page that silently
// shows ten-minute-old numbers looks exactly like one that is current.
function renderFresh() {
  const chip = document.querySelector('[data-stamp]');
  if (!chip || !DATA) return;
  // a static report is not live and says nothing about being live: no badge
  if (!LIVE) { chip.hidden = true; return; }
  chip.hidden = false;
  // 11h: the time is the last check that worked, so it moves with every
  // poll — the data's own time ("17:29" at 17:45) looked frozen. Checks that
  // fail for more than two intervals turn it amber: STALE, at the last good one
  const stale = NET.fails > 0 && NET.lastOk && Date.now() - NET.lastOk > 2 * POLL_MS;
  // the live dot covers the judge too: a board that refreshes while nothing
  // can grade is not all green
  const judge = judgeDown();
  chip.dataset.fresh = stale ? 'stale' : judge ? 'judge-offline' : 'ok';
  const at = checkedAt();
  const ago = NET.lastOk ? Math.max(0, Math.round((Date.now() - NET.lastOk) / 1000)) : null;
  chip.title = (judge ? judgeWhy() + '\n' : '') + `data last changed ${refreshedAt() || '—'}`
    + (ago != null ? ` · checked ${ago < 60 ? ago + ' s' : Math.round(ago / 60) + ' min'} ago` : '');
  chip.replaceChildren(el('span', { class: 'dot ' + (stale || judge ? 'warn' : 'ok') }),
    stale ? `STALE · ${at}` : judge ? `LIVE · ${at} · judge offline` : `LIVE · ${at}`);
}
// the last check that worked, as the clock on the wall reads it
const checkedAt = () => NET.lastOk ? new Date(NET.lastOk).toTimeString().slice(0, 5) : refreshedAt();
const POLL_MS = 5000;

// static shell bits (rendered whenever a payload arrives)
function renderStatic() {
  if (LIVE) document.getElementById('pageSub').textContent =
    'Submit models, follow training runs and compare the results. It updates as work finishes.';
  renderWarnings();
  // the model filters moved into the Models tab, where what they filter is on
  // screen beneath them; each one carries its own count there
  const eyebrow = document.getElementById('heroEyebrow');
  const topics = ((DATA.judged || {}).exam || []).length;
  if (eyebrow) eyebrow.textContent = LIVE
    ? `team benchmark${topics ? ` · ${topics} topics` : ''} · ${DATA.models.length} models`
    : `report · ${DATA.models.length} models · ${DATA.tasks.length} tasks`;
  document.getElementById('metaChips').replaceChildren(
    LIVE ? '' : el('span', { class: 'chip', text: `generated ${DATA.generated}` }),
    LIVE ? el('a', { class: 'chip', href: 'guide', target: '_blank', rel: 'noopener',
                     style: 'text-decoration:none', text: '📖 guide for new users' }) : '',
    LIVE ? el('a', { class: 'chip', href: 'guide#the-loop', target: '_blank', rel: 'noopener',
                     style: 'text-decoration:none', text: 'the loop, explained' }) : '',
    // the harness build and the transformers version are provenance: they
    // live on that tab now, not on every tab's first line
    DATA.meta.anyLimit ? el('span', { class: 'chip', text: '⚠ smoke data (--limit)' }) : '');
  // the page's own action, where the page introduces itself (11b)
  const acts = document.getElementById('heroActs');
  if (acts) acts.replaceChildren(LIVE
    ? el('button', { class: 'primary', 'data-submit-model': '1', text: 'Submit a model',
        onclick: () => { state.after = { focus: '[data-ms="submit"] input' };
          navigate({ tab: 'queue', model: null, topic: null }); } })
    : '');
  renderFresh();
  renderWho();
}

function barMoreInit() {
  const bar = document.getElementById('bar'), btn = document.getElementById('barMore');
  if (!bar || !btn || btn._init) return;
  btn._init = true;
  const set = open => { bar.dataset.more = open ? 'open' : '';
    btn.setAttribute('aria-expanded', String(open)); };
  btn.addEventListener('click', e => { e.stopPropagation(); set(bar.dataset.more !== 'open'); });
  document.addEventListener('click', e => {
    // a popover opened from inside the panel (the name, the theme) is part of it
    if (bar.dataset.more === 'open' && !e.target.closest('#barRight, #barMore, [data-pop]'))
      set(false); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && bar.dataset.more === 'open') { set(false); btn.focus(); } });
}

function initData(d) {
  DATA = d;
  barMoreInit();
  renderStatic();
  // adopt the address bar before the first paint, so a shared #model= link opens
  // that model rather than the overview
  routeFromHash();
  render();
}

// ---------------------------------------------------------------------------
// One place where the page talks to the service. Every loader goes through it
// so that a failure is (a) visible and (b) not a hammer: the renderers call
// load*() whenever their data is missing, and a failed load leaves it missing,
// which turned a blocked API into 27 requests in a few seconds. Backoff lives
// here, and the renderers ask netReady() before firing.
// ---------------------------------------------------------------------------
const NET = { fails: 0, nextAt: 0, last: null, lastOk: 0, MIN: 1000, MAX: 30000 };
const netReady = () => Date.now() >= NET.nextAt;

function netRender() {
  const box = document.getElementById('netstatus');
  if (!box) return;
  // one failure is a blip (a deploy, a sleeping laptop); from the second on,
  // say what is wrong rather than showing "Loading results…" forever
  if (NET.fails < 2) { box.replaceChildren(); return; }
  const secs = Math.round(NET.last.wait / 1000);
  box.replaceChildren(el('div', { class: 'warn', 'data-net': 'down' },
    el('b', { text: 'Not reaching the service. ' }),
    `${NET.last.path} — ${NET.last.err}. ${NET.fails} attempts; retrying in ${secs}s and `
    + `backing off to ${NET.MAX / 1000}s. Anything shown below is from the last good load.`));
  const view = document.getElementById('view');
  if (view && !DATA) view.replaceChildren(el('p', { class: 'small', style: 'margin:20px 4px',
    text: `Cannot reach ${NET.last.path} (${NET.last.err}). Still trying.` }));
}

function netOk() {
  const had = NET.fails;
  NET.fails = 0; NET.nextAt = 0; NET.last = null; NET.lastOk = Date.now();
  if (had) netRender();
  renderFresh();
}

function netFail(path, err) {
  NET.fails++;
  const wait = Math.min(NET.MAX, NET.MIN * Math.pow(2, NET.fails - 1));
  NET.nextAt = Date.now() + wait;
  NET.last = { path, err: String((err && err.message) || err), wait };
  netRender();
  renderFresh();
}

async function api(path, opts) {
  try {
    const r = await fetch(path, opts);
    checkBuild(r);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    netOk();
    return j;
  } catch (e) { netFail(path, e); throw e; }
}

// a results fetch the page owes itself: at boot, and whenever a run lands.
// Cleared only by a fetch that worked — a failed one stays owed and the poll
// retries it. "Retry only while there is no data" left a page that had any
// data showing it for as long as the tab stayed open.
// One fetch at a time; one asked for while another is out stays owed, since
// the one in flight may have left before the judge's file landed.
let RESULTS_DUE = true, RESULTS_ASKED = 0, RESULTS_BUSY = false;
async function refreshResults() {
  RESULTS_DUE = true;
  const ask = ++RESULTS_ASKED;
  if (RESULTS_BUSY) return;
  RESULTS_BUSY = true;
  try {
    initData(await api('api/results'));
    if (ask === RESULTS_ASKED) RESULTS_DUE = false;
  } catch (e) { /* netFail said so; the poll retries while RESULTS_DUE */ }
  finally { RESULTS_BUSY = false; }
}

// the rows this browser queued: when one of them changes status the Queue
// goes back to page 1 and marks it — coming back to page 2 of the queue hid
// a new row, even a failure
function rememberQueued(id) {
  if (!id) return;
  state.mine.add(id);
  try { localStorage.setItem('bench-mine', JSON.stringify([...state.mine].slice(-50))); }
  catch (e) { /* private mode: this tab only */ }
}
// the confirmation's link: the row it made, marked and in view
function followRun(id) {
  markQueueRow(id);
  state.after = { scroll: `tr[data-queue-row="${id}"]` };
  if (state.tab !== 'queue' || state.model || state.topic)
    navigate({ tab: 'queue', model: null, topic: null });
  else render();
}

function markQueueRow(id) {
  state.qMark = id;
  if (state.pg.queue) state.pg.queue.page = 1;
  setTimeout(() => { if (state.qMark === id) { state.qMark = null;
    if (state.tab === 'queue') (state.queueRedraw || render)(); } }, LANDED_MS);
}

async function loadQueue() {
  try {
    const rows = await api('api/submissions?limit=100');
    const prev = new Map(state.queue.map(r => [r.id, r]));
    const moved = rows.find(r => state.mine.has(r.id) && prev.has(r.id)
      && prev.get(r.id).status !== r.status);
    if (moved) markQueueRow(moved.id);
    // new scores exist when a run's answers land (status -> done) AND, for a
    // judged run, again when its judge batch lands minutes later: the row is
    // already 'done' by then, so only the judge's own status says so
    const judgeDone = r => ((r || {}).judge || {}).status === 'done';
    // a row this page has never seen that is already finished counts too: a
    // fast run, or a laptop waking up, can go from queued to judged between
    // two polls. The first load is not "new" — the boot fetch covered it.
    const first = !state.queueLoaded;
    const justFinished = rows.some(r => {
      const p = prev.get(r.id);
      if (!p) return !first && (r.status === 'done' || judgeDone(r));
      return (r.status === 'done' && p.status !== 'done') || (judgeDone(r) && !judgeDone(p));
    });
    state.queueLoaded = true;
    const changed = rows.length !== state.queue.length || rows.some(r => {
      const p = prev.get(r.id);
      return !p || p.status !== r.status || p.progress !== r.progress
        || JSON.stringify(p.judge || null) !== JSON.stringify(r.judge || null);
    });
    state.queue = rows;
    if (justFinished) await refreshResults();       // new scores -> re-render everything
    else if (changed && state.tab === 'queue') (state.queueRedraw || render)();
    // 11i: the model page's exam panel shows each topic's place in the queue
    else if (changed && state.model && state.msitRedraw) state.msitRedraw();
  } catch (e) { /* netFail said so, and set how long to wait */ }
}

const THEMES = ['auto', 'light', 'dark', 'dim'];
// 11f: the colours cross-fade for 250ms on a person's change, never at load
function themeFade() {
  if (motionOff()) return;
  const r = document.documentElement;
  r.classList.add('theme-fade');
  clearTimeout(themeFade._t);
  themeFade._t = setTimeout(() => r.classList.remove('theme-fade'), 260);
}
function applyTheme(t) {
  if (t === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', t);
  document.getElementById('themeBtn').textContent = 'Theme \u25be';
  document.getElementById('themeBtn').title =
    `theme: ${t} — choose auto, light, dark or dim; remembered in this browser`;
  for (const it of document.querySelectorAll('#themeMenu [role=menuitemradio]'))
    it.setAttribute('aria-checked', String(it.dataset.theme === t));
  try { localStorage.setItem('bench-theme', t); } catch (e) { /* private mode etc. */ }
}
// "Theme ▾" promised a menu and cycled on click. A menu: Auto, Light, Dark,
// Dim, the current one ticked — on the shared popover, so it is never clipped
(() => {
  const btn = document.getElementById('themeBtn');
  const labels = { auto: 'Auto (follow the system)', light: 'Light', dark: 'Dark', dim: 'Dim' };
  popover(btn, () => el('div', { class: 'moremenu themes', id: 'pop-theme',
      'aria-label': 'theme' },
    THEMES.map(t => el('button', { role: 'menuitemradio', 'data-theme': t,
      'aria-checked': String(THEMES[themeIdx] === t), text: labels[t],
      onclick: () => { themeIdx = THEMES.indexOf(t); themeFade(); applyTheme(t); popClose(true); } }))),
    { key: 'theme', placement: 'bottom-end' });
})();
let themeIdx = 0;
try {   // remembered per browser — the dashboard is a page people leave open
  const saved = localStorage.getItem('bench-theme');
  if (THEMES.includes(saved)) themeIdx = THEMES.indexOf(saved);
} catch (e) { /* storage unavailable: stay on auto */ }
// always applied, even on 'auto': the button's title names the current theme,
// and a button whose tooltip is only right after the first click is a lie
applyTheme(THEMES[themeIdx]);

// boot: embedded data renders immediately; live mode fetches then polls
if (LIVE) {
  document.getElementById('view').replaceChildren(
    skeleton(6, { 'data-loading': 'results' }));
  refreshResults().then(() => {
    if (DATA && !DATA.models.length) { state.tab = 'queue'; render(); }
  });
  loadQueue();
  loadJudgeHealth();
  setInterval(renderFresh, 15000);           // "3 min ago" has to count
  setInterval(() => {
    if (!netReady()) return;                 // still inside the backoff window
    // a results fetch that failed is still owed: the first load (the board
    // would stay empty), or the one a finished run asked for (the board
    // would keep the numbers from before it for as long as the tab stayed open)
    if (RESULTS_DUE) refreshResults();
    loadQueue();
    loadJudgeHealth();
    if (state.tab === 'training') loadTraining();
    if (state.tab === 'review') loadReview();
    if (state.tab === 'exam') loadExam();
    // the Loop board and a topic page: without this nothing ever re-fetched
    // /api/loop, so a board whose first load failed stayed empty for as long
    // as the tab was open — which is exactly what happened on the live tree
    if (state.tab === 'loop' || state.topic || (state.model && state.msit.open)) loadLoop();
  }, POLL_MS);
} else {
  initData(DATA);
}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>__CSS__</style></head>
<body class="viz-root"><div id="tip" role="status"></div>
<header class="bar" id="bar">
  <div class="bar-in">
    <span class="bar-title"><span class="t-full">__TITLE__</span><span class="t-short">Benchmark</span></span>
    <span class="livebadge" id="liveBadge" data-stamp="1" hidden></span>
    <div class="tabs" role="tablist" id="tabs"></div>
    <button class="bar-more" id="barMore" aria-expanded="false" aria-controls="barRight"
      aria-label="checks, name and theme" title="checks, name and theme">&#8943;</button>
    <div class="bar-right" id="barRight">
      <div id="warnings" class="bar-checks"></div>
      <div id="who"></div>
      <button id="themeBtn" class="barpill" title="cycle auto / light / dark / dim — remembered in this browser">Theme &#9662;</button>
    </div>
  </div>
</header>
<div class="wrap">
__BANNER__
  <div id="netstatus"></div>
  <div class="pagehero" id="pagehero">
    <p class="eyebrow" id="heroEyebrow"></p>
    <h1>__TITLE__</h1>
    <p class="sub" id="pageSub">lm-evaluation-harness results, one self-contained
    file — data embedded, charts drawn locally, nothing fetched.</p>
    <div class="meta-chips" id="metaChips"></div>
    <div class="hero-acts" id="heroActs"></div>
  </div>
  <div id="view"></div>
  <button class="totop" id="toTop" hidden>&#8593; Top</button>
  <footer>Every score carries its standard error · differences are z-tested before
  they are called wins · provenance is in the Provenance tab · scores are only comparable to
  published numbers when n-shot, prompt template and metric all match.</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>__JS__SLOT__</script>
</body></html>"""


def banner_html(text: str, link: str = "", link_text: str = "") -> str:
    """A band across the top of a report that cannot be dismissed. It exists
    for the demo: a page of numbers that look like a leaderboard, and are not
    one, must say so wherever it is opened and however far it is scrolled
    from — so it is in the markup, not in a toast."""
    if not text:
        return ""
    tail = (f' <a href="{html.escape(link)}">{html.escape(link_text or link)}</a>'
            if link else "")
    return f'  <div class="pagebanner" role="note">{html.escape(text)}{tail}</div>'


def build_report(runs: list[dict], out_path: Path, title: str,
                 calibration: dict | None = None, taint: dict | None = None,
                 parents: dict | None = None, judge_identity: dict | None = None,
                 banner: str = "", banner_link: tuple[str, str] = ("", ""),
                 fingerprints: dict | None = None) -> Path:
    if not runs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"<h1>No lm-eval results found.</h1><p>{html.escape(banner)}</p>",
                            encoding="utf-8")
        return out_path
    payload = build_payload(merge_runs(runs), title, source="", calibration=calibration,
                            taint=taint, parents=parents, judge_identity=judge_identity,
                            fingerprints=fingerprints)
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = (TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__BANNER__", banner_html(banner, *banner_link))
            .replace("__CSS__", CSS)
            .replace("__DATA__", blob)
            .replace("__JS__SLOT__", JS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path,
                    help="directory passed to lm_eval --output_path (searched recursively)")
    ap.add_argument("-o", "--out", type=Path, default=Path("artifacts/benchmark_report.html"))
    ap.add_argument("--title", default="Model benchmark report")
    ap.add_argument("--csv", type=Path, help="also write a flat CSV of every metric")
    ap.add_argument("--exam-tasks", type=Path, default=None,
                    help="the built exam (manifest.json): judged results on any other question "
                         "set are history. Default: $BENCH_ROOT/exam/tasks, else "
                         "<results>/../../exam/tasks")
    args = ap.parse_args()

    runs = load_results(args.results)
    if not runs:
        print(f"no lm-eval results found under {args.results}")
        return 1

    print(f"found {len(runs)} result file(s):")
    merged = merge_runs(runs)
    for mid, r in merged.items():
        pm = {t: primary_metric(e) for t, e in r["tasks"].items()}
        summary = "  ".join(f"{t}={100 * v[1]:.1f}%" for t, v in sorted(pm.items())
                            if v and len(t) < 20 and v[0] in PROPORTION)
        print(f"  {mid:<45} {summary[:90]}")

    # the judge's calibration lives at the root of the results tree
    cal_path = args.results / "judge_calibration.json"
    cal = None
    if cal_path.exists():
        try:
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cal = None
    fps = None
    for cand in ([args.exam_tasks] if args.exam_tasks else
                 [Path(os.environ["BENCH_ROOT"]) / "exam" / "tasks"] * bool(os.environ.get("BENCH_ROOT"))
                 + [args.results.resolve().parent.parent / "exam" / "tasks"]):
        if (cand / "manifest.json").is_file():
            import exam_build
            fps = exam_build.current_fingerprints(cand)
            print(f"judged results count only on the exam built in {cand}")
            break
    else:
        print("no built exam found (--exam-tasks): judged results are shown whatever question "
              "set they were graded on")
    out = build_report(runs, args.out, args.title, calibration=cal, fingerprints=fps)
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.1f} KB)")

    if args.csv:
        # provenance rides along with every row: a value whose template policy,
        # dtype, seed and harness build are unknown cannot be reproduced
        lines = ["model,task,metric,value,stderr,n_shot,n_samples,template_applied,"
                 "template_id,template_policy,dtype,backend,batch_size,seed,limit,"
                 "harness_git,transformers,eval_finished"]
        for r in runs:
            ai = r.get("archinfo") or {}
            for task, entry in r["tasks"].items():
                for name, d in entry.items():
                    if not isinstance(d, dict) or "value" not in d:
                        continue
                    lines.append(",".join(str(x) for x in [
                        r["model"], task, name, d["value"], d.get("stderr", ""),
                        r["n_shot"].get(task, ""), r["n_samples"].get(task, ""),
                        r["chat_template"], ai.get("tmpl_sha") or "",
                        (ai.get("kind_reason") or "").replace(",", ";"),
                        r["dtype"] or "", r["backend"] or "",
                        r["batch_size"] or "", r["seed"] if r["seed"] is not None else "",
                        "full" if r["limit"] is None else r["limit"],
                        r["git_hash"] or "", r["transformers_version"] or "",
                        r["date"] or ""]))
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {args.csv}  ({len(lines) - 1} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
