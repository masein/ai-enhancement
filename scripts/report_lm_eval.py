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
                               # when this topic's grades landed — its own
                               # time, not the file's last merge
                               "judged_at")
                              if t.get(k) is not None}
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


def topic_gate(task: str, t: dict, state: dict | None, caution: str | None) -> dict:
    """{ok, why, caution} for proposing a skill spec from one exam topic.
    Broadest reason first: if the judged suite cannot be believed, no topic
    score is evidence of anything; then whether this topic has enough
    report-half questions to have a score at all; then whether the model
    wrote enough for the judge to have assessed anything. `caution` is
    MMLU's finding for the same category — context, never a gate."""
    why = short = None
    if not state or not state.get("ok"):
        rs = "; ".join((state or {}).get("reasons") or ["the judged suite is preliminary"])
        why = f"the judged suite is preliminary, so no topic score is evidence yet: {rs}"
        short = "the judged suite is preliminary"
    else:
        n_rep = t.get("n_report") or 0
        a = t.get("answers") or {}
        n = a.get("n") or 0
        blank = (a.get("empty", 0) + a.get("short", 0))
        if n_rep < PROPOSE_MIN_N:
            why = (f"{n_rep} report-half questions in this topic — under the {PROPOSE_MIN_N} "
                   f"floor, so the score is noise. Write more questions on the Exam tab")
            short = f"under the {PROPOSE_MIN_N}-question floor — write more on the Exam tab"
        elif n and blank / n >= EMPTY_SHARE:
            why = (f"the model wrote nothing usable on {blank} of {n} answers here — that "
                   f"is a generation failure, not a topic gap; multiple choice is the "
                   f"instrument for this model")
            short = f"the model wrote nothing usable on {blank} of {n} answers"
        elif n >= 8 and (a.get("distinct") or n) <= DEGENERATE_DISTINCT:
            why = (f"the model gave the same answer on nearly every question here "
                   f"({a.get('distinct')} distinct answers in {n}) — the output has "
                   f"collapsed, and no data for this topic fixes that")
            short = "the model gave the same answer on nearly every question"
    # the caution is decision-relevant only where a decision is possible; on a
    # row that is already refused it would be one more line of noise
    return {"ok": why is None, "why": why, "short": short,
            "caution": caution if why is None else None}


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
    return by_model


def build_payload(by_model: dict[str, dict], title: str, source: str,
                  taint: dict[str, list[str]] | None = None,
                  calibration: dict | None = None,
                  parents: dict[str, str] | None = None,
                  judge_identity: dict | None = None) -> dict:
    """`taint`: model id -> tasks whose diagnostics its training data was
    derived from (the service computes it from the run/dataset join). A
    tainted task is treated exactly like a missing required task: shown per
    task, excluded from the official average, the model unranked."""
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
        judge = _trim_judge(r.get("judge"))
        jstate = judged_state(judge, cal, current_judge) if judge else None
        diag = _trim_diag(r.get("diag"))
        compare = {}
        for t in tainted:
            if not t.startswith("exam_"):
                continue
            pid = parents.get(mid)
            pjudge = _trim_judge(by_model[pid].get("judge")) if pid in by_model else None
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
                t["propose"] = topic_gate(task, t, jstate, mc.get("why"))
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

    # provenance warnings — they gate every claim below them
    warnings: list[str] = []
    shots_seen: dict[str, set] = {}
    for r in by_model.values():
        for t in headline:
            if t in r["n_shot"]:
                shots_seen.setdefault(t, set()).add(r["n_shot"][t])
    mism = [t for t, s in shots_seen.items() if len(s) > 1]
    if mism:
        warnings.append(
            f"Few-shot count differs between models on: {', '.join(mism)}. Those "
            f"columns are not comparable — re-run with the same --num_fewshot.")
    if len({r["chat_template"] for r in by_model.values()}) > 1:
        applied = [display[m] for m, r in by_model.items() if r["chat_template"]]
        warnings.append(
            "Chat template applied to some models but not others (applied to: "
            + ", ".join(applied) + "). Correct if and only if those are the instruct "
            "models — it moves scores by tens of points, so check the list.")
    if any(r["limit"] for r in by_model.values()):
        warnings.append(
            "At least one run used --limit, so it did not see the full task. Fine "
            "for a smoke test, not for a reported number.")
    # a template that EXISTS but differs between models applied the same way is
    # also a comparability break — the hash is what has to match, not the yes/no
    shas = {(r.get("archinfo") or {}).get("tmpl_sha")
            for m, r in by_model.items() if r["chat_template"]}
    shas.discard(None)
    if len(shas) > 1:
        warnings.append(
            f"The models evaluated WITH a chat template used {len(shas)} different "
            f"templates ({', '.join(sorted(shas))}). Prompt format differs, so "
            f"those scores answer slightly different questions.")
    unconf = [display[m] for m, r in by_model.items()
              if (r.get("archinfo") or {}).get("kind_unconfirmed")]
    if unconf:
        warnings.append(
            "Chat template applied on detection alone to: " + ", ".join(unconf)
            + ". Those repos ship a template but their names do not say "
            "instruct, so nothing corroborates the choice — if any of them is a "
            "pretrained checkpoint, resubmit it with kind=base.")
    if req_absent:
        warnings.append(
            "This report's required-task list is narrower than the protocol: "
            + ", ".join(req_absent) + " were not run by anyone here, so 'official' "
            "means complete within this report, not complete under the full "
            "protocol.")
    alarms = [m for m in model_rows
              if any(c.get("verdict") == "test" for c in (m["taintCompare"] or {}).values())]
    if alarms:
        warnings.append(
            f"{len(alarms)} model{'s' if len(alarms) > 1 else ''} "
            f"({', '.join(m['name'] for m in alarms[:4])}) moved on the diagnosis half and not "
            f"on the leaderboard half after training on derived data — the training taught the "
            f"test, not the skill. This is the alarm the split exists to raise; see the model's "
            f"page.")
    tainted_rows = [m for m in model_rows if m["tainted"]]
    if tainted_rows:
        warnings.append(
            f"{len(tainted_rows)} model{'s' if len(tainted_rows) > 1 else ''} "
            f"({', '.join(m['name'] for m in tainted_rows[:4])}"
            f"{', …' if len(tainted_rows) > 4 else ''}) trained on data derived from "
            f"benchmark diagnostics. The affected task is shown per model, badged, and "
            f"excluded from that model's official average — it is not a ranking claim.")
    dups = [m for m in model_rows if m.get("duplicateOf")]
    if dups:
        warnings.append(
            f"{len(dups)} row{'s' if len(dups) > 1 else ''} "
            f"({', '.join(m['name'] for m in dups[:4])}{', …' if len(dups) > 4 else ''}) "
            f"give the same answers as another row on every ranked task — the same run "
            f"submitted twice ("
            + "; ".join(f"{m['name']}: {m['duplicateWhy']}" for m in dups[:2])
            + "). Both are shown; the less complete one is not ranked and says which row it "
              "duplicates. Nothing has been deleted.")
    near = [m for m in model_rows if m.get("nearDuplicateOf")]
    if near:
        warnings.append(
            f"{len(near)} row{'s' if len(near) > 1 else ''} print the same score as another "
            f"row on every ranked task and are NOT the same run: "
            + "; ".join(f"{m['name']} vs {m['nearDuplicateOfName']} differ on "
                        f"{m['nearDuplicateWhy']}" for m in near[:3])
            + (", …" if len(near) > 3 else "")
            + ". Both are ranked, because identical is the test for a duplicate and these "
              "are not identical — but two runs this close are worth a look.")
    n_prelim = sum(1 for m in model_rows if not m["official"])
    if n_prelim and required:
        warnings.append(
            f"{n_prelim} of {len(model_rows)} models are preliminary (they have not "
            f"finished all {len(required)} required tasks) and carry no overall "
            f"average or rank. Their per-task numbers are shown everywhere and are "
            f"valid on their own — resubmit with suite=full to make them official.")
    hashes = sorted({r["git_hash"] for r in by_model.values() if r["git_hash"]})
    if len(hashes) > 1:
        warnings.append(
            f"Results come from {len(hashes)} different harness builds "
            f"({', '.join(hashes)}). A benchmark whose code changed is a different "
            f"benchmark — treat cross-build comparisons with suspicion.")

    judged_tasks = sorted({t for r in by_model.values()
                           for t in ((r.get("judge") or {}).get("tasks") or {})})
    judge_meta = next(((m["judge"] or {}).get("judge") for m in model_rows if m.get("judge")), None)
    if judged_tasks and cal is None:
        warnings.append(
            "Judged free-response scores are on file but the judge has not been calibrated "
            "against a person (scripts/judge_calibrate.py). They are shown as preliminary and "
            "enter no average and no rank.")
    elif cal and not cal.get("calibrated"):
        warnings.append(
            f"The judge's agreement with a human grader is Cohen's kappa {cal['kappa']} over "
            f"{cal.get('n')} answers, below the {KAPPA_MIN} line. Judged scores are preliminary: "
            f"shown, never ranked, never averaged.")
    if judge_meta and judge_meta.get("stub"):
        warnings.append(
            "Judged scores on this board come from the STUB grader (a word-overlap stand-in "
            "used for plumbing tests). They are not judgements of anything.")
    drifted = [m["name"] for m in model_rows if m.get("judge")
               and (m["judge"].get("canary") or {}).get("drifted")]
    if drifted:
        warnings.append(
            f"The judge's canary moved on {len(drifted)} run{'s' if len(drifted) > 1 else ''} "
            f"({', '.join(drifted[:4])}): the same thirty scripts were graded differently from "
            f"the previous run. Those judged scores are preliminary — a vendor may have changed "
            f"the model behind the id.")
    local_judged = [m["name"] for m in model_rows if provisional_reason(m.get("judge"))]
    if local_judged:
        warnings.append(
            f"Judged scores for {len(local_judged)} model{'s' if len(local_judged) > 1 else ''} "
            f"({', '.join(local_judged[:4])}{', …' if len(local_judged) > 4 else ''}) were "
            f"graded by a local model — not a pinned benchmark. They are provisional: shown "
            f"greyed on the model page, never ranked, never in any average.")
    if any((m.get("judge") or {}).get("judge", {}).get("single_provider_loop") for m in model_rows):
        warnings.append(
            "Single-provider loop: the judge shares a provider with the exam writer or the "
            "generator (ALLOW_SINGLE_PROVIDER_LOOP). Every judged score carries that caveat; "
            "self-preference in LLM judges is documented and large.")
    other_judges = sorted({(m["judge"]["judge"] or {}).get("id") for m in model_rows
                           if m.get("judge")} - {current_judge, None})
    if current_judge and other_judges:
        warnings.append(
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
  --surface-1:#fcfcfb; --plane:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --good:#0ca30c; --critical:#d03b3b; --warning:#fab219; --success-text:#006300;
  --accent:#2a78d6; --accent-soft:rgba(42,120,214,0.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --success-text:#0ca30c; --accent:#3987e5; --accent-soft:rgba(57,135,229,0.16);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --success-text:#0ca30c; --accent:#3987e5; --accent-soft:rgba(57,135,229,0.16);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
/* dim: slate blue-grey, softer than the near-black dark theme (the GitHub /
   wandb "dimmed" look). Same dark series palette — re-validated against both
   dim surfaces: all six checks pass. */
:root[data-theme="dim"] .viz-root {
  color-scheme: dark;
  --surface-1:#1c2333; --plane:#141a26; --text-primary:#e6edf3; --text-secondary:#b6c2d1;
  --muted:#8b98a8; --grid:#2b3546; --axis:#3a465a; --border:rgba(230,237,243,0.11);
  --success-text:#3fb950; --accent:#58a6ff; --accent-soft:rgba(88,166,255,0.16);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--plane); color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px; line-height:1.5; }
.wrap { max-width:1320px; margin:0 auto; padding:26px 22px 70px; }
h1 { font-size:21px; font-weight:650; margin:0; letter-spacing:-0.01em; }
h2 { font-size:15px; font-weight:600; margin:0 0 3px; }
.sub { color:var(--text-secondary); font-size:13px; margin:2px 0 0; }
.topbar { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.meta-chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.chip { display:inline-flex; align-items:center; gap:6px; font-size:12px;
  color:var(--text-secondary); background:var(--surface-1); border:1px solid var(--border);
  border-radius:999px; padding:3px 10px; }
.chip .mono { font-size:11px; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:18px 20px; margin:14px 0; }
button, .btn { font:inherit; font-size:13px; color:var(--text-primary); background:var(--surface-1);
  border:1px solid var(--border); border-radius:8px; padding:6px 11px; cursor:pointer; }
button:hover, .btn:hover { background:var(--plane); }
input[type=search] { font:inherit; font-size:13px; color:var(--text-primary);
  background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:6px 11px; width:220px; }
input[type=search]:focus { outline:2px solid var(--accent-soft); border-color:var(--accent); }
.filters { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:14px 0 4px; }
.seg { display:inline-flex; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.seg button { border:0; border-radius:0; background:var(--surface-1); padding:6px 12px; }
.seg button + button { border-left:1px solid var(--border); }
.seg button[aria-pressed="true"] { background:var(--accent-soft); color:var(--text-primary); font-weight:600; }
.count-note { font-size:12px; color:var(--muted); }
.tabs { display:flex; gap:2px; border-bottom:1px solid var(--axis); margin:10px 0 0;
  overflow-x:auto; }
.tabs button { border:0; background:none; border-radius:8px 8px 0 0; padding:8px 13px;
  color:var(--text-secondary); white-space:nowrap; }
.tabs button:hover { background:var(--surface-1); }
.tabs button[aria-selected="true"] { color:var(--text-primary); font-weight:600;
  box-shadow:inset 0 -2px 0 var(--accent); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }
.tile { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:13px 16px; }
.tile .label { color:var(--text-secondary); font-size:12px; }
.tile .value { font-size:24px; font-weight:600; letter-spacing:-0.02em; }
.tile .note { font-size:11.5px; color:var(--muted); margin-top:2px; }
.hero-row { display:grid; grid-template-columns:minmax(260px,1.2fr) 2fr; gap:12px; }
@media (max-width:800px){ .hero-row { grid-template-columns:1fr; } }
.hero { font-size:46px; font-weight:650; letter-spacing:-0.03em; line-height:1.05; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--muted); font-weight:600; padding:6px 9px; border-bottom:1px solid var(--axis);
  white-space:nowrap; }
td { padding:6px 9px; border-bottom:1px solid var(--grid); font-size:13px; }
td.num, th.num { text-align:right; }
tr:last-child td { border-bottom:none; }
.lb-wrap { overflow-x:auto; }
th.sortable { cursor:pointer; user-select:none; }
th.sortable:hover { color:var(--text-primary); }
th .dir { font-size:9px; }
/* typeahead dropdown under the metric query bar */
.ac { position:relative; max-width:620px; }
.ac-list { position:absolute; top:calc(100% + 4px); left:0; right:0; z-index:30;
  background:var(--surface-1); border:1px solid var(--border); border-radius:10px;
  box-shadow:0 10px 28px rgba(0,0,0,.14); max-height:288px; overflow-y:auto; }
.ac-item { display:flex; gap:10px; align-items:center; padding:7px 11px;
  cursor:pointer; font-size:13px; }
.ac-item[aria-selected=true], .ac-item:hover { background:var(--accent-soft); }
.ac-item .ac-name { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:12.5px; }
.ac-item mark { background:none; color:var(--accent); font-weight:650; padding:0; }
.ac-tag { font-size:10px; color:var(--muted); border:1px solid var(--border);
  border-radius:999px; padding:1px 7px; flex:none; }
.toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:10px 0 4px; }
.toolbar input, .toolbar select { font:inherit; font-size:12.5px; color:var(--text-primary);
  background:var(--plane); border:1px solid var(--border); border-radius:8px;
  padding:5px 9px; }
.toolbar input:focus, .toolbar select:focus { outline:2px solid var(--accent-soft);
  border-color:var(--accent); }
.lb td.model, .lb th.model { position:sticky; left:0; background:var(--surface-1); z-index:1; }
.lb tr:hover td { background:var(--plane); }
.lb .se { color:var(--muted); font-size:11px; }
.best { font-weight:650; }
.best::after { content:"\2009\25CF"; color:var(--accent); font-size:8px; vertical-align:2px; }
.badge { display:inline-block; font-size:10.5px; border:1px solid var(--border);
  border-radius:5px; padding:0 5px; margin-left:6px; color:var(--text-secondary);
  vertical-align:1px; }
.badge.instruct { color:var(--accent); border-color:var(--accent-soft);
  background:var(--accent-soft); }
.badge.ckpt { border-style:dashed; color:var(--text-secondary); }
.badge.prelim { color:var(--warning); border-color:var(--warning); }
.tiebest { font-weight:650; }
.tiebest::after { content:"\2009\2248"; color:var(--muted); font-size:9px; vertical-align:1px; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
  color:var(--text-secondary); }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 10px; }
.legend span { display:inline-flex; align-items:center; gap:6px; font-size:12px;
  color:var(--text-secondary); }
.key { width:11px; height:11px; border-radius:3px; display:inline-block; }
.key.line { width:14px; height:0; border-top:3px solid; border-radius:2px; }
.panels { display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:12px; }
@media (max-width:520px){ .panels { grid-template-columns:1fr; } }
.panel { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:14px 16px 8px; }
.panel h3 { font-size:13.5px; font-weight:600; margin:0; }
.panel .pmeta { font-size:11.5px; color:var(--muted); margin:1px 0 8px; }
/* chip row under a task heading: what it measures, how many options, coverage,
   whether it separates anything, where the ceiling is. Every chip carries the
   long version in its title, so the row stays short. */
/* heat legend + "about these benchmarks" */
.hl { display:inline-flex; align-items:center; gap:10px; flex-wrap:wrap; font-size:11.5px;
  color:var(--text-secondary); }
.hl-item { display:inline-flex; align-items:center; gap:4px; }
.hl-sw { width:16px; height:11px; border-radius:3px; border:1px solid var(--border);
  display:inline-block; }
.lb th.hasinfo { cursor:help; }
.lb th.hasinfo::after { content:"\2009\24D8"; font-size:9px; color:var(--muted);
  vertical-align:1px; }
.about { padding:0; }
.about-toggle { font:inherit; font-size:13px; font-weight:600; width:100%; text-align:left;
  background:none; border:0; color:var(--text-primary); padding:12px 16px; cursor:pointer;
  border-radius:12px; }
.about-toggle:hover { color:var(--accent); }
.about-body { padding:0 16px 14px; display:grid;
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px 22px; }
.about-item { border-top:1px solid var(--border); padding-top:10px; }
.about-head { display:flex; align-items:center; gap:5px; flex-wrap:wrap; margin-bottom:4px; }
.about-name { font-weight:600; font-size:13px; }
.about-desc { margin:0; font-size:12.5px; line-height:1.5; color:var(--text-secondary); }
/* model detail page */
.backlink { display:inline-block; font-size:12.5px; color:var(--accent);
  text-decoration:none; margin:10px 2px; }
.backlink:hover { text-decoration:underline; }
.mhead { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.mtitle { margin:0; font-size:24px; letter-spacing:-0.01em; }
.rankbadge { font-size:12px; font-weight:650; color:var(--accent);
  background:var(--accent-soft); border-radius:6px; padding:1px 7px; }
.mlink { color:inherit; text-decoration:none; border-bottom:1px dotted var(--border); }
.mlink:hover { color:var(--accent); border-bottom-color:var(--accent); }
.mtbl td, .mtbl th { vertical-align:middle; }
.mtbl .tname { font-weight:550; cursor:help; }
.mtbl .sbar { display:block; }
.mtbl td.dimmed { color:var(--muted); }
.mtbl tr.domrow td { font-size:11px; font-weight:650; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); padding-top:14px; border-bottom:none; }
.mtbl tr.flagrow td { font-size:11.5px; color:var(--muted); padding-top:0; border-bottom:none; }
.provlist { display:grid; grid-template-columns:minmax(120px,max-content) 1fr;
  gap:4px 16px; margin:8px 0 0; font-size:12.5px; }
.provlist dt { color:var(--muted); }
.provlist dd { margin:0; overflow-wrap:anywhere; }
/* per-item diagnosis */
.dxlead { margin:4px 0 0; padding:9px 12px; border-radius:10px; font-size:13px;
  line-height:1.55; background:var(--accent-soft); color:var(--text-primary); }
.dxlead + .dxlead { margin-top:7px; }
.dxlead b { font-weight:650; }
.dxlead.calm { background:var(--plane); color:var(--text-secondary); }
.dx { border:1px solid var(--border); border-radius:10px; margin-top:8px;
  background:var(--surface-1); }
.dx > summary { cursor:pointer; padding:9px 12px; display:flex; align-items:center;
  gap:10px; flex-wrap:wrap; font-size:13px; list-style:none; }
.dx > summary::-webkit-details-marker { display:none; }
.dx > summary::before { content:'▸'; color:var(--muted); font-size:11px; width:9px; }
.dx[open] > summary::before { content:'▾'; }
.dx[open] > summary { border-bottom:1px solid var(--border); }
.dx > summary:hover { color:var(--accent); }
.dx .dxname { font-weight:600; }
.dx .dxbody { padding:10px 12px 14px; }
.dxflag { font-size:10.5px; font-weight:650; letter-spacing:.03em; border-radius:5px;
  padding:1px 6px; background:color-mix(in srgb, var(--s2) 16%, transparent);
  color:var(--s2); border:1px solid color-mix(in srgb, var(--s2) 45%, transparent); }
.dxbar { display:flex; height:12px; border-radius:6px; overflow:hidden;
  background:var(--plane); min-width:120px; }
.dxbar > span { display:block; height:100%; }
.dxkey { display:flex; flex-wrap:wrap; gap:4px 16px; margin:9px 0 0; font-size:12px; }
.dxkey > span { display:flex; align-items:baseline; gap:6px; cursor:help; }
.dxkey i { display:inline-block; width:10px; height:10px; border-radius:3px;
  border:1px solid var(--border); flex:none; transform:translateY(1px); }
.dxkey b { font-weight:600; font-variant-numeric:tabular-nums; }
.dxsub { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:6px; }
.dxsub td, .dxsub th { padding:3px 8px 3px 0; border-bottom:1px solid var(--border);
  text-align:left; }
.dxsub th { font-size:11px; font-weight:600; color:var(--muted); }
.dxsub td.num, .dxsub th.num { text-align:right; font-variant-numeric:tabular-nums; }
/* categories first, subjects on expand; a dim row is under the noise floor */
.dxcat { border-bottom:1px solid var(--border); }
.dxcat > summary { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  padding:5px 0; cursor:pointer; font-size:12.5px; list-style:none; }
.dxcat > summary::-webkit-details-marker { display:none; }
.dxcat > summary::before { content:"\25B8"; color:var(--muted); font-size:10px; width:10px; }
.dxcat[open] > summary::before { content:"\25BE"; }
.dxcat .dxcname { font-weight:600; min-width:15ch; }
.dxcat .num { font-variant-numeric:tabular-nums; min-width:6ch; text-align:right; }
.dxcat.dim > summary { color:var(--muted); }
.dxcat.dim .dxcname { font-weight:500; }
.dxcat .dxsub { margin:2px 0 8px 20px; width:auto; min-width:60%; }
.dxperm { margin:14px 0 4px; }
.dxperm .dxsub { width:auto; min-width:60%; }
.lb td.dim { color:var(--muted); }
.badge.taint { color:var(--s2); border-color:var(--s2); }
.jbar { display:flex; height:10px; border-radius:5px; overflow:hidden; background:var(--plane); min-width:110px; }
.jbar > span { display:block; height:100%; }
.jd { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:6px; }
.jd td, .jd th { padding:4px 8px 4px 0; border-bottom:1px solid var(--border); text-align:left; }
.jd th { font-size:11px; font-weight:600; color:var(--muted); }
.jd td.num, .jd th.num { text-align:right; font-variant-numeric:tabular-nums; }
.jd tr.dim td { color:var(--muted); }
.jd tr.dim .jbar { opacity:.4; }
.lb th.judged { color:var(--text-secondary); font-style:italic; }
/* the one button that can spend money and make training data: never colour alone */
.propose { font:inherit; font-size:11.5px; padding:2px 9px; border-radius:6px;
  border:1px solid var(--accent); background:var(--accent-soft); color:var(--accent);
  cursor:pointer; margin-left:auto; }
.propose:disabled { border-color:var(--border); background:var(--plane);
  color:var(--muted); cursor:not-allowed; }
.propwhy { flex-basis:100%; font-size:11.5px; color:var(--muted); margin:0 0 4px 20px; }
.rv { border:1px solid var(--border); border-radius:10px; padding:12px 14px; margin:10px 0; }
.rv h3 { margin:0 0 4px; font-size:15px; }
.rv textarea { width:100%; box-sizing:border-box; min-height:70px; font:inherit;
  font-size:13px; color:var(--text-primary); background:var(--surface-1);
  border:1px solid var(--border); border-radius:8px; padding:8px 10px; margin:6px 0; }
.rv .frm input { min-width:120px; }
.rv .ex { border-left:2px solid var(--border); padding:2px 0 2px 10px; margin:0 0 8px;
  font-size:12.5px; }
.rv .ex .kv { color:var(--muted); font-size:11.5px; }
.kvs { display:flex; flex-wrap:wrap; gap:4px 18px; font-size:12.5px; margin:4px 0; }
.kvs b { font-weight:600; }
.provlist.small dd { font-size:12px; }
.dxex { margin-top:10px; font-size:12.5px; }
.dxex > summary { cursor:pointer; color:var(--accent); font-size:12px; }
.dxex ul { list-style:none; padding:0; margin:8px 0 0; }
.dxex li { border-left:2px solid var(--border); padding:2px 0 2px 10px; margin:0 0 9px; }
.dxex .q { display:block; }
.dxex .kv { color:var(--muted); font-size:11.5px; }
.dxex .kv b { color:var(--text-secondary); font-weight:550; }
.dxh { font-size:11px; font-weight:650; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); margin:16px 0 2px; }
.domhead { font-size:12px; font-weight:650; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); margin:18px 2px 8px; }
.domhead .se { text-transform:none; letter-spacing:0; font-weight:400; }
.tchips { display:flex; flex-wrap:wrap; gap:5px; margin:5px 0 2px; }
.tchip { font-size:10.5px; line-height:1.5; border:1px solid var(--border);
  border-radius:5px; padding:0 6px; color:var(--text-secondary); cursor:help;
  white-space:nowrap; }
.tchip.dom { background:var(--accent-soft); border-color:var(--accent-soft);
  color:var(--accent); }
.tchip.flat { color:var(--warning); border-color:var(--warning); }
.tchip.front { border-style:dashed; }
/* a two-state toggle in a .ctrl row — pressed state is not colour alone */
.tgl { font:inherit; font-size:12px; cursor:pointer; border-radius:8px;
  padding:4px 10px; border:1px solid var(--border); background:var(--plane);
  color:var(--text-secondary); }
.tgl.on { background:var(--accent-soft); border-color:var(--accent);
  color:var(--accent); font-weight:600; }
.tgl:disabled { opacity:.45; cursor:not-allowed; }
.tv { display:none; margin-top:12px; } .tv.open { display:block; }
.small { font-size:12px; color:var(--text-secondary); }
.up { color:var(--success-text); } .down { color:var(--critical); }
.note { border-left:2px solid var(--axis); padding:6px 0 6px 12px; margin:12px 0;
  color:var(--text-secondary); font-size:13px; }
.warn { border-left:2px solid var(--warning); padding:6px 0 6px 12px; margin:10px 0;
  color:var(--text-secondary); font-size:13px; }
.warn b, .note b { color:var(--text-primary); }
/* a band the page cannot lose: a demo report must say what it is wherever
   it is opened, so it is markup at the top of the document, not a toast */
.pagebanner { border:1px solid var(--warning); border-left-width:4px; border-radius:6px;
  padding:10px 14px; margin:0 0 14px; font-size:13.5px; font-weight:600;
  color:var(--text-primary); background:color-mix(in srgb, var(--warning) 10%, transparent); }
.pagebanner a { font-weight:500; }
/* the board's checks, folded where they are not about the work in front of
   you. Folded, never dismissed: they are findings */
.warnfold { margin:8px 0; }
.warnfold > summary { cursor:pointer; font-size:12.5px; color:var(--text-secondary);
  padding:4px 0; }
.warnfold[open] > summary { color:var(--text-primary); }
.lb td.model { white-space:nowrap; }
.lb td.model .mname { display:inline-block; max-width:22ch; overflow:hidden;
  text-overflow:ellipsis; vertical-align:bottom; }
.st { display:inline-block; font-size:11px; border-radius:999px; padding:2px 9px;
  border:1px solid var(--border); white-space:nowrap; }
.st-done { color:var(--success-text); border-color:var(--success-text); }
.st-failed { color:var(--critical); border-color:var(--critical); }
.st-active { color:var(--accent); border-color:var(--accent); }
.st-muted { color:var(--muted); }
@keyframes pulse { 0%,100%{opacity:.35} 50%{opacity:1} }
.st-active::before { content:'●'; margin-right:5px; animation:pulse 1.4s infinite; }
.frm { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:10px; }
.frm input, .frm select { font:inherit; font-size:13px; color:var(--text-primary);
  background:var(--plane); border:1px solid var(--border); border-radius:8px;
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
  cursor:pointer; align-items:flex-start; font-size:13px; border-radius:6px; }
.runrow:hover { background:var(--plane); }
.runrow.sel { background:var(--accent-soft); }
.rchip { width:10px; height:10px; border-radius:3px; border:1px solid var(--border);
  flex:none; margin-top:5px; }
.rbody { flex:1; min-width:0; }
.rtop { display:flex; align-items:center; gap:8px; }
.rname { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.rmeta { font-size:11.5px; color:var(--muted); margin-top:1px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.ctrl { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:2px 0 10px; }
.ctrl input[type=range] { width:140px; accent-color:var(--accent); }
.ctrl input[type=search] { font:inherit; font-size:12.5px; color:var(--text-primary);
  background:var(--plane); border:1px solid var(--border); border-radius:8px;
  padding:5px 9px; width:190px; }
.msec { margin-bottom:10px; }
.msec-h { background:none; border:0; padding:4px 0 6px; font-size:12.5px; font-weight:600;
  color:var(--text-secondary); cursor:pointer; display:flex; gap:6px; align-items:center;
  box-shadow:none; }
.msec-h:hover { background:none; color:var(--text-primary); }
.msec-h::before { content:'▾'; font-size:11px; color:var(--muted); }
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
.radar-tbl th, .radar-tbl td { font-size:12.5px; }
.xbtn { border:0; background:none; padding:0 2px; font-size:15px; line-height:1;
  color:var(--muted); cursor:pointer; }
.xbtn:hover { color:var(--critical); background:none; }
.lb td input[type=checkbox] { accent-color:var(--accent); margin:0; vertical-align:middle;
  cursor:pointer; }
mark { background:var(--accent-soft); color:var(--text-primary); border-radius:3px;
  padding:0 1px; }
pre.mono { background:var(--plane); border:1px solid var(--border); border-radius:8px;
  padding:10px 12px; margin:8px 0 0; }
.mx { border-collapse:separate; border-spacing:2px; font-variant-numeric:tabular-nums;
  width:auto; }
.mx th { border:0; font-size:10.5px; padding:3px 6px; text-transform:none; letter-spacing:0; }
.mx th.rowh { text-align:right; max-width:170px; overflow:hidden; text-overflow:ellipsis; }
.mx th.colh span { writing-mode:vertical-rl; transform:rotate(180deg); max-height:120px;
  overflow:hidden; text-overflow:ellipsis; display:inline-block; }
.mx td { border:0; border-radius:5px; background:var(--plane); width:34px; height:30px;
  text-align:center; font-size:13px; cursor:default; }
.mx td.self { background:none; }
.mx td:hover { outline:2px solid var(--accent-soft); }
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:7px 10px; font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,.16); z-index:50;
  max-width:320px; }
#tip .v { font-size:14px; font-weight:650; color:var(--text-primary); }
#tip .l { color:var(--text-secondary); }
#tip .k { display:inline-block; width:10px; border-top:3px solid; border-radius:2px;
  margin-right:6px; vertical-align:3px; }
svg text { font-family:system-ui,-apple-system,sans-serif; }
.hit { fill:transparent; }
.bar { transition:opacity .12s; }
.dimmed .bar:not(.hot) { opacity:0.3; }
.dimmed text.blab:not(.hot) { opacity:0.35; }
a { color:var(--accent); }
footer { margin-top:28px; font-size:12px; color:var(--muted); }
@media print { .filters, .tabs, button { display:none !important; }
  .view { display:block !important; } body { background:#fff; } }
"""

JS = r"""
'use strict';
// Two lives, one page: embedded JSON = the frozen single-file report;
// an empty (null) data slot = served by service/app.py, which makes this the
// LIVE dashboard — same charts, data fetched, plus a Submit & Queue tab.
let DATA = JSON.parse(document.getElementById('data').textContent || 'null');
const LIVE = DATA === null;
const state = {
  q: '', kind: 'all', tab: 'overview',
  model: null,                         // open model detail page, by id (hash-routed)
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
        topic: '', just: {}, justOpen: '' },                               // Review tab
  ex: { status: null, candidates: [], loaded: false, msg: '', topic: '' },   // Exam tab
  topic: null,                         // open topic page, by slug (hash-routed)
  loop: { rows: null, blocked: '', msg: '', loaded: false },                 // Loop tab
  loopSit: { model: '', kind: 'auto', tasks: null, msg: '', busy: false },   // sit-the-exam form
  ans: { model: '', rows: null, loading: false, topic: '', open: {},
         acuity: 'all', flag: 'all', score: 'all', sort: 'score' },          // Answers panel
  loopRead: {},                        // topics whose answers this browser has opened
  mdl: { q: '', kind: 'all', src: 'all', family: 'all', judgedOnly: false,
         taintedOnly: false, sort: { key: 'avg', dir: -1 } },              // Models tab
  rvName: '',                          // the name approvals are recorded under (remembered)
  // in-place refreshers registered by the mounted tab, so the 5s poll updates
  // data WITHOUT rebuilding the DOM — a full render() mid-keystroke would steal
  // focus from filter inputs and kill slider drags
  trRedraw: null, queueRedraw: null,
};

// ---------- tiny DOM helper: everything dynamic goes through textContent ----------
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
  for (const k of kids.flat(2)) if (k !== null && k !== undefined)
    e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
const pct  = (v, d = 1) => v == null ? '—' : (100 * v).toFixed(d) + '%';
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
function mkSel(label, opts, cur, onpick) {
  const s = el('select', { 'aria-label': label },
    opts.map(([v, l]) => el('option', { value: v, text: l,
                                        selected: cur === v ? '' : null })));
  s.addEventListener('change', e => onpick(e.target.value));
  return s;
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
  } else if (tn && !(m.missing || []).length)
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
  if (m.date) out.push(`Last evaluated ${String(m.date).slice(0, 10)}.`);
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
      title: `trained on data derived from the diagnosis half of ${m.tainted.map(tName).join(', ')}`
        + ' — that score is shown per model and never ranked',
      text: 'trained on ' + m.tainted.map(tName).join(', ') + ' diagnostics' }) : null;
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
function placeTip(x, y) {
  const p = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let left = x + p, top = y + p;
  if (left + w > innerWidth) left = x - w - p;
  if (top + h > innerHeight) top = y - h - p;
  tip.style.left = left + 'px'; tip.style.top = top + 'px';
}
document.addEventListener('pointermove', e => {
  const t = e.target.closest('[data-tip]');
  if (!t || !fillTip(t)) { tip.style.opacity = 0; return; }
  tip.style.opacity = 1; placeTip(e.clientX, e.clientY);
});
document.addEventListener('focusin', e => {
  const t = e.target.closest('[data-tip]');
  if (!t || !fillTip(t)) return;
  const r = t.getBoundingClientRect();
  tip.style.opacity = 1; placeTip(r.right, r.bottom);
});
document.addEventListener('focusout', () => { tip.style.opacity = 0; });

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
    const name = m.name.length > 18 ? m.name.slice(0, 17) + '…' : m.name;
    svg.append(el('svg:text', { x: LBL - 8, y: y + BH * 0.75, 'font-size': 11.5,
      fill: dim ? 'var(--muted)' : 'var(--text-secondary)', 'text-anchor': 'end', class: 'blab',
      'data-model': m.id, text: name },
      el('svg:title', { text: m.id })));
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
    el('summary', { text: 'Show the items it got wrong (diagnosis half only)' }));
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
function proposeBtn(mid, topic, gate) {
  const b = el('button', { class: 'propose', text: 'Propose a skill spec',
    disabled: gate.ok ? null : '', title: gate.ok
      ? 'ask the configured LLM what skill is missing here, from the judge\'s written '
        + 'assessments of diagnosis-half answers only — a person reviews the answer '
        + 'before anything is generated'
      : gate.why,
    onclick: async e => {
      e.preventDefault();
      b.disabled = true;
      const r = await fetch('api/proposals', { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
        body: JSON.stringify({ model: mid, topic: topic,
                               requested_by: state.rvName }) }).catch(() => null);
      const j = r ? await r.json().catch(() => ({})) : {};
      state.rv.msg = r && r.ok ? `proposal #${j.id} submitted for ${mid} · ${topic} — `
          + 'the LLM answer lands here when the batch completes'
        : 'refused: ' + (j.detail || (r ? r.status : 'server unreachable'));
      state.rv.loaded = false;
      navigate({ tab: 'review', model: null });
    } });
  return b;
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
          + 'leaderboard half (never in the training data) and diagnosis half',
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
      + `scripts/categories.yaml rolled into "other": ${v.unmapped.join(', ')}. `
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
    el('p', { class: 'note', text: 'Every item is assigned by a hash of its own content '
      + 'to a leaderboard half or a diagnosis half, identically for every model and every '
      + 'run. Only the diagnosis half is shown or exported here. A model retrained on '
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
      `${n} items in the log · leaderboard half ${v.n_report} `
      + `(${pct(v.score_report)}) · diagnosis half ${v.n_diagnose} `
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
    el('p', { class: 'sub', text: 'Open questions per topic, answered in writing, graded 0–4 '
      + (prov ? 'against a written rubric by a local model whose id cannot be pinned — '
              : 'against a written rubric by an API judge pinned to a dated model id — ')
      + 'single answers, never pairwise. Length is in the rubric and reported below; a '
      + 'thirty-script canary is re-graded every run so a change to the model behind the id '
      + 'would show.' }));
  card.append(el('p', { class: ok ? 'note' : 'warn' },
    el('b', { text: ok ? 'Counts. ' : 'Preliminary. ' }),
    cal ? `Cohen's κ ${cal.kappa} against a human grader over ${cal.n} answers `
        + `(line at ${J.kappaMin})` + (cal.judge_id ? `, for judge ${cal.judge_id}` : '') + '. '
        : 'No calibration on file for this judge (scripts/judge_calibrate.py). ',
    ok ? 'Topic scores may enter the separate judged average; they never enter the '
       + 'multiple-choice average.'
       : st.reasons.length ? 'Shown, never ranked, never averaged: ' + st.reasons.join('; ') + '.'
       : 'Shown, never ranked, never averaged.'));
  const j = m.judge;
  if (!j) { card.append(note('Not judged: this model has no judge.json on file. Submit it with '
    + 'suite=judged, or run scripts/judge.py over its fr_* answers.')); return card; }
  if (j.skipped) { card.append(note(j.skipped + '. A judge scores its own family higher; the '
    + 'cell stays empty rather than flattering.')); return card; }
  if (j.judge.stub) card.append(el('p', { class: 'warn', text: 'Graded by the STUB grader — a '
    + 'word-overlap stand-in for plumbing tests. Not a judgement of anything.' }));
  if (prov) card.append(el('p', { class: 'warn', 'data-provisional': 'judge' },
    el('b', { text: 'Provisional. ' }),
    `${upFirst(j.judge.provisional_reason || 'graded by a local model — not a pinned benchmark')}: `
    + `${j.judge.served_model || j.judge.model} at ${j.judge.base_url || 'a local server'}`
    + (j.judge.weights ? ` (weights ${j.judge.weights})` : '')
    + '. A local server\'s model id is whatever was typed at launch, so these scores are shown '
    + 'greyed, never ranked and never in any average.'));
  // a rubric is an instrument: one its author has not signed off yet grades,
  // but it does not settle anything, and deleting DRAFT from its heading
  // changes its sha — which is the point, a different rubric is a different
  // instrument and before/after across the change do not compare
  if (j.judge.rubric_status === 'draft') card.append(el('p', { class: 'warn',
    'data-rubric': 'draft' }, el('b', { text: 'Draft rubric. ' }),
    `${(j.judge.rubrics_draft || []).map(frName).join(', ') || 'a topic'} is graded against a `
    + 'rubric its author has not signed off yet, so its scores are a reading, not a result. '
    + 'Sign-off is recorded by removing DRAFT from the rubric\'s heading, which changes its '
    + 'sha: scores from before and after do not compare.'));
  if (j.judge.single_provider_loop) card.append(el('p', { class: 'warn', text: 'Single-provider '
    + 'loop: the judge shares a provider with the exam writer or the generator. Every score here '
    + 'carries that caveat — a judge scores its own family higher.' }));
  const cn = j.canary;
  if (cn) card.append(el('p', { class: cn.drifted ? 'warn' : 'small', 'data-canary': cn.drifted ? 'drifted' : 'steady' },
    el('b', { text: cn.drifted ? 'Canary moved. ' : 'Canary steady. ' }),
    `${cn.graded} of ${cn.n} fixed scripts re-graded: mean absolute deviation `
    + `${cn.mad_vs_human} from the human marks` + (cn.mad_vs_previous != null
      ? `, ${cn.mad_vs_previous} from the previous run (limit ${cn.threshold})` : ', first run for this judge')
    + (cn.drifted ? ' — the judge is not the judge it was; these scores are preliminary.' : '.')));

  // per topic, weakest first, on the REPORT half — the diagnose half is never the score
  const cats = J.exam.filter(t => j.tasks[t] && pubScore(j.tasks[t]) != null)
    .sort((a, b) => pubScore(j.tasks[a]) - pubScore(j.tasks[b]));
  if (cats.length) {
    card.append(el('div', { class: 'dxh', text: 'By topic (0–4), weakest first — report half'
      + (prov ? ' · provisional, not ranked' : '') }));
    card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
      el('thead', {}, el('tr', {}, el('th', { text: 'topic' }), el('th', { class: 'num', text: 'score' }),
        el('th', { class: 'num', text: 'κ' }), el('th', { class: 'num', text: 'items (report half)' }),
        el('th', { text: 'score distribution 0 → 4 (all items)' }),
        LIVE ? el('th', { text: 'what to do' }) : '')),
      el('tbody', {}, cats.map(t => { const v = j.tasks[t];
        const k = cal && cal.per_category && cal.per_category[frName(t)];
        const nr = v.n_report != null ? v.n_report : v.n;
        const g = v.propose;
        const tainted = (m.tainted || []).includes(t);
        return el('tr', { class: prov || nr < CAT_MIN_N ? 'dim' : null, 'data-topic': frName(t) },
          el('td', {}, frName(t), tainted ? el('span', { class: 'badge taint',
            title: 'this model trained on data derived from this topic\'s diagnosis half — the '
              + 'score is shown and is not a ranking claim',
            text: 'trained on it' }) : ''),
          el('td', { class: 'num', text: `${num(pubScore(v), 2)} / 4` }),
          el('td', { class: 'num se', text: k ? String(k.kappa) : '—',
            title: k ? `${k.n} human-graded answers in this topic` : 'not calibrated per topic' }),
          el('td', { class: 'num se', text: String(nr) + (nr < CAT_MIN_N ? ' · under ' + CAT_MIN_N : '') }),
          el('td', {}, jBar(v.dist, v.n)),
          LIVE ? el('td', {}, g ? proposeBtn(m.id, frName(t), g) : '',
            g && !g.ok ? el('div', { class: 'propwhy', title: g.why,
              text: 'no proposal: ' + (g.short || g.why) }) : '',
            g && g.caution ? el('div', { class: 'propwhy', text: 'caution — MMLU for this '
              + 'category: ' + g.caution }) : '') : ''); })))));
    if (m.judgedAvg != null)
      card.append(el('p', { class: 'small', text: `Judged average ${num(m.judgedAvg, 2)} / 4 over `
        + `${cats.filter(t => !(m.tainted || []).includes(t)).length} topics, report half`
        + ((m.tainted || []).some(t => cats.includes(t))
           ? `, excluding ${(m.tainted || []).filter(t => cats.includes(t)).map(frName).join(', ')} `
             + '— this model trained on data derived from that topic' : '')
        + (ok ? '.' : ' — preliminary until the judge is calibrated.') }));
    card.append(el('p', { class: 'small', text: `Topics under ${CAT_MIN_N} report-half questions `
      + 'are greyed: the exam bank is still being written (Exam tab). The diagnose half of each '
      + 'topic is what a proposal may read; it is never the score.' }));
    // score vs length
    const rows = [];
    for (const t of cats) for (const b of j.tasks[t].score_vs_length || [])
      rows.push([frName(t), b.bucket, b.n, b.mean]);
    if (rows.length) {
      const drift = cats.map(t => { const b = j.tasks[t].score_vs_length || [];
        return b.length > 1 ? b[b.length - 1].mean - b[0].mean : 0; });
      const worst = Math.max(...drift);
      card.append(el('div', { class: 'dxh', text: 'Score against answer length' }));
      card.append(el('p', { class: 'small', text: worst >= 1
        ? `Longer answers score up to ${num(worst, 1)} points higher in at least one category — `
          + 'length may be driving the judge. Check the rubric\'s length clause before believing '
          + 'the category scores.'
        : 'No category rewards length by a point or more; the length clause is holding.' }));
      card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
        el('thead', {}, el('tr', {}, el('th', { text: 'category' }), el('th', { text: 'answer length' }),
          el('th', { class: 'num', text: 'items' }), el('th', { class: 'num', text: 'mean score' }))),
        el('tbody', {}, rows.map(([c, b, n, s]) => el('tr', {}, el('td', { text: c }),
          el('td', { text: b }), el('td', { class: 'num se', text: String(n) }),
          el('td', { class: 'num', text: num(s, 2) })))))));
    }
    // a topic graded criterion by criterion: what it was weak AT, the
    // critical failures in words, and the acuity the failures fell on
    for (const t of cats) {
      const v = j.tasks[t];
      if (!v.criteria_mean) continue;
      const labels = v.criteria_labels || {};
      const ids = Object.keys(v.criteria_mean)
        .sort((a, b) => (v.criteria_mean[a] ?? 2) - (v.criteria_mean[b] ?? 2));
      card.append(el('div', { class: 'dxh', 'data-criteria': frName(t),
        text: `${frName(t)} — by criterion (0–1), weakest first · ${v.n} answers, `
          + `${v.unparseable || 0} unreadable` }));
      card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd',
        'data-criteria-table': frName(t) },
        el('thead', {}, el('tr', {}, el('th', { text: 'criterion' }),
          el('th', { class: 'num', text: 'mean' }), el('th', { class: 'num', text: 'answers' }),
          el('th', { text: '0 → 1' }))),
        el('tbody', {}, ids.map(id => {
          const m = v.criteria_mean[id], n = (v.criteria_n || {})[id] || 0;
          // only the file says a criterion may be skipped; a low count on any
          // other one means replies that could not be read, counted above
          const cond = (v.criteria_conditional || []).includes(id);
          return el('tr', { class: n ? null : 'dim', 'data-criterion': id },
            el('td', {}, labels[id] || id,
              cond ? el('span', { class: 'se', text: ` · conditional, ${n} of ${v.n}` }) : ''),
            // exactly 0 is a finding, and weakest-first puts it at the top:
            // "0" beside "0.65" read as an empty cell, so this column keeps
            // its decimals rather than trimming them away
            el('td', { class: 'num', text: m == null ? '—' : (+m).toFixed(2) }),
            el('td', { class: 'num se', text: String(n) }),
            // zero-anchored, one hue: the mean beside it is the encoding
            el('td', {}, el('div', { class: 'dxbar', style: 'width:180px;height:8px' },
              m == null ? '' : el('span', { 'data-bar': (+m).toFixed(2),
                // a hairline at zero: an empty track and a missing value look
                // the same, and one of them is a result
                style: `width:${Math.max(0.8, 100 * m).toFixed(1)}%;`
                  + 'background:var(--s1)' }))));
        })))));
      const flags = v.flags || {};
      const fids = Object.keys(flags);
      if (!fids.length) card.append(el('p', { class: 'note', 'data-flags': '0' },
        'This topic\'s criteria file names no flag: the score is the fold of the criteria '
        + 'above and nothing overrides it.'));
      // one line per flag, in words: what it is, how often it fired, and what
      // it did to the score — the effect is the file's, applied in code
      fids.forEach(fid => {
        const f = flags[fid];
        card.append(el('p', { class: f.n ? 'warn' : 'note', 'data-flag': fid,
          'data-flag-n': String(f.n), 'data-flag-topic': frName(t) },
          el('b', { text: `${f.label || fid}${f.n ? '. ' : ' — none. '}` }),
          f.n
            ? `${f.n} of ${v.n} answers (${pct(f.share, 0)}) were flagged, and each one `
              + `${f.effect_words || 'changes the score'}. That is the rule in the criteria `
              + 'file, applied here and not by the judge.'
            : `None of the ${v.n} answers was flagged. When it is true it `
              + `${f.effect_words || 'changes the score'}, whatever the criteria said.`));
      });
      // one table per metadata field that actually splits this topic —
      // acuity for a medical bank, difficulty and domain for a technical one.
      // A field every item shares is a sentence, not a table.
      const constant = v.breakdowns_constant || {};
      if (Object.keys(constant).length) card.append(el('p', { class: 'small',
        'data-constant-fields': Object.keys(constant).join(','),
        text: Object.entries(constant)
          .map(([f, val]) => `${f} is ${val} on every item`).join('; ')
          + ' — no table for that.' }));
      const order = ['acuity', 'difficulty', 'jurisdiction_required', 'intent'];
      const fields = Object.keys(v.breakdowns || {}).sort(
        (a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
      fields.forEach(field => {
        const cells = v.breakdowns[field];
        card.append(el('div', { class: 'dxh', text: `By ${field}` }));
        if (field === 'difficulty') card.append(el('p', { class: 'small',
          'data-difficulty-note': '1', text: 'The author\'s own level for each question, 1 '
            + 'easiest: does the model do well only on basic questions, or can it reason about '
            + 'the ambiguous and high-risk ones too?' }));
        card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd',
          'data-breakdown-table': field, 'data-breakdown-topic': frName(t) },
          el('thead', {}, el('tr', {}, el('th', { text: field }),
            el('th', { class: 'num', text: 'answers' }), el('th', { class: 'num', text: 'mean' }),
            ...fids.map(fid => el('th', { class: 'num',
              text: flags[fid].label || fid })))),
          el('tbody', {}, Object.entries(cells).map(([k, b]) =>
            el('tr', { 'data-value': k }, el('td', { text: k }),
              el('td', { class: 'num se', text: String(b.n) }),
              el('td', { class: 'num', text: `${num(b.mean, 2)} / 4` }),
              ...fids.map(fid => el('td', { class: 'num',
                text: String((b.flags || {})[fid] || 0) }))))))));
      });
    }
    // the answers themselves, for whichever topic is picked — the same panel
    // the topic page shows, because "see the answers" is the step between a
    // score and knowing what to do about it
    if (LIVE) card.append(modelAnswers(m, cats));
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
    card.append(el('p', { class: 'small', text: `Diagnosis-half MMLU items only, asked without `
      + `their options and graded against the gold option's text; "knew it" is a rubric score of `
      + `${j.correct_at} or more.` + (s.tot.unjoined ? ` ${s.tot.unjoined} items had no MMLU run to `
      + 'join to.' : '') }));
  }
  card.append(el('p', { class: 'small', style: 'margin-top:12px' },
    `Judge ${j.judge.id} (${j.judge.provider})`
    + (j.judge.batch_id ? ` · batch ${String(j.judge.batch_id).slice(0, 18)}` : '')
    + (j.judge.weights_sha256 ? ` · weights ${String(j.judge.weights_sha256).slice(0, 12)}` : '')
    + ` · prompt v${j.judge.prompt_version} ${String(j.judge.prompt_sha256 || '').slice(0, 12)} · rubrics `
    + [...new Set(Object.values(j.judge.rubrics || {}).map(r => `v${r.version}`))].join(', ')
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
    el('p', { class: 'sub', text: 'This model trained on data derived from the diagnosis half '
      + 'of an exam topic or a benchmark. The report half was never touched by that data, so '
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
            navigate({ tab: 'review', model: null, topic: null }); } })))) : '',
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
        half(rubric ? 'report half (the published score, never in the training data)'
                    : 'leaderboard half (never in the training data)',
             c.before.report, c.after.report, c.dReport, c.seReport),
        half('diagnosis half (the spec came from here)', c.before.diagnose, c.after.diagnose, c.dDiagnose, c.seDiagnose)))));
    if (rubric && c.judge) card.append(el('p', { class: 'small',
      text: `Both sides graded by ${c.judge}; a comparison across two judges is two `
        + 'instruments, and the card says so instead of drawing it.' }));
    const warn = c.verdict === 'test' || c.verdict === 'mixed';
    card.append(el('p', { class: warn ? 'warn' : 'dxlead calm', 'data-verdict': c.verdict },
      el('b', { text: VERDICT[c.verdict] + '. ' }), c.text.replace(/^The training taught the (skill|test)\. |^Nothing measurable changed\. |^An unusual pattern[^:]*: /, '')));
    if (c.ratio != null && c.verdict === 'test')
      card.append(el('p', { class: 'small', text: 'Ratio of the two deltas (diagnosis / '
        + `report): ${num(c.ratio, 1)}.` }));
    const cats = Object.entries(c.categories || {})
      .sort((x, y) => (y[1].dDiagnose ?? 0) - (x[1].dDiagnose ?? 0));
    if (cats.length) {
      card.append(el('div', { class: 'dxh', text: 'By category — the half we never touched' }));
      card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
        el('thead', {}, el('tr', {}, el('th', { text: 'category' }),
          el('th', { class: 'num', text: 'leaderboard half before → after' }), el('th', { class: 'num', text: 'Δ' }),
          el('th', { class: 'num', text: 'diagnosis half before → after' }), el('th', { class: 'num', text: 'Δ' }),
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
  const back = el('a', { class: 'backlink', href: '#tab=' + state.tab,
    text: '← Back to ' + (TABS.find(([id]) => id === state.tab) || [, 'the board'])[1] });

  const head = el('div', { class: 'card' },
    el('div', { class: 'mhead' },
      el('h2', { class: 'mtitle', text: m.name }),
      ckBadge(m) || el('span', { class: 'badge' + (m.kind === 'instruct' ? ' instruct' : ''),
        text: m.kind }),
      prelimBadge(m) || '', taintBadge(m) || '',
      // the badge carries its date too: #7 from August is not #7 today
      r ? el('span', { class: 'rankbadge', 'data-rank': String(r.n),
        title: `${r.n} of ${r.of} ranked models`
          + (m.date ? `, from the evaluation of ${String(m.date).slice(0, 10)}` : ''),
        text: `#${r.n}` + (m.date ? ` · evaluated ${String(m.date).slice(0, 10)}` : '') }) : ''),
    el('p', { class: 'sub mono', text: m.id }),
    el('p', { class: 'small', style: 'margin-top:8px', text: modelSentence(m) }),
    el('div', { class: 'tiles', style: 'margin-top:14px' },
      tile('Parameters', m.params ? P(m.params) : 'Unknown',
        a.active_params ? `${P(a.active_params)} active · ${a.experts} experts, `
                        + `${a.experts_per_tok}/token (${a.active_src})`
                        : (m.paramsSrc ? 'from ' + m.paramsSrc : null)),
      tile('Training compute', comp ? flop(comp.c) + ' FLOP' : 'Unknown',
        comp ? `6ND · N ${P(comp.N)} ${comp.nsrc} · D ${fmtCount(comp.D)} tokens `
             + `(run ${comp.run.name})`
             : 'no tracked training run supplies a token count for this model'),
      tile(state.avgMode === 'raw' ? 'Average (raw)' : 'Average (above chance)',
        avg != null ? pct(avg) : '—',
        r ? `${ord(r.n)} of ${r.of} ranked` : `preliminary · ${m.nhave}/${m.nreq} required`),
      tile('Tasks', `${m.nhave}/${m.nreq}`,
        (m.missing || []).length ? 'missing ' + m.missing.join(', ') : 'all required tasks')));

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
    ['last evaluated', m.date],
  ].filter(([, v]) => v != null && v !== '' && v !== false);
  const provCard = el('div', { class: 'card' },
    el('h2', { text: 'Provenance' }),
    el('p', { class: 'sub', text: 'What produced these numbers. Two runs whose '
      + 'template id or harness differ are not comparable, whatever the scores say.' }),
    el('dl', { class: 'provlist' }, prov.flatMap(([k, v]) =>
      [el('dt', { text: k }), el('dd', { class: 'mono', text: String(v) })])));

  // The exam leads: it is the instrument the loop steers by. The
  // multiple-choice results and the per-item diagnosis follow as the free
  // second opinion — same GPU, no API call, and a different kind of evidence.
  const judged = vJudged(m), taint = vTaint(m), diag = vDiagnose(m);
  const sections = [['judged', 'Judged', judged], ['results', 'Results', results],
                    ['diagnose', 'Diagnose', diag], ['taint', 'Training data', taint],
                    ['provenance', 'Provenance', provCard],
                    ['runs', 'Runs', LIVE ? vModelRuns(m) : null]];
  for (const [id, , node] of sections) if (node) node.id = 'sec-' + id;
  return [back, head, modelNav(sections), ...sections.map(([, , n]) => n),
          ].filter(Boolean);
}

// The model page is long, and since the judged section arrived the
// interesting part is halfway down it. A sub-nav that sticks is the cheapest
// fix: anchors, not routes, so Back still leaves the page the way it came.
function modelNav(sections) {
  return el('div', { class: 'card modelnav', 'data-model-nav': '1',
    style: 'position:sticky;top:0;z-index:5;padding:8px 14px' },
    el('div', { class: 'frm', style: 'gap:14px' },
      sections.filter(([, , node]) => node).map(([id, label]) =>
        el('a', { href: '#sec-' + id, class: 'small', 'data-nav': id, text: label,
          onclick: e => { e.preventDefault();
            const t = document.getElementById('sec-' + id);
            if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); } }))));
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
      el('td', { class: 'small' }, r.suite,
        (() => { let t = []; try { t = JSON.parse(r.tasks || '[]'); } catch (e) { /* older row */ }
          return t.length ? el('div', { class: 'se', text: t.map(frName).join(', ') }) : ''; })()),
      el('td', {}, el('span', { class: stClass(r.status), text: r.status })),
      el('td', { class: 'small se', text: r.error || r.progress || '' }),
      el('td', {}, el('a', { href: `api/runs/${r.id}/log`, target: '_blank', rel: 'noopener',
        class: 'small', text: 'log' }))))))));
  return card;
}

function vOverview(ms) {
  const frag = [];
  // only official models can be ranked — a preliminary model has no average
  const ranked = ms.filter(m => officialAvg(m) != null)
                   .sort((a, b) => officialAvg(b) - officialAvg(a));
  const prelim = ms.filter(m => !m.official);
  let pairs = 0, real = 0, big = null;
  for (const t of DATA.accTasks) for (const [a, b, diff, z, ok] of (DATA.sig[t] || [])) {
    if (!ms.find(m => m.id === a) || !ms.find(m => m.id === b)) continue;
    pairs++; if (ok) real++;
    if (ok && (!big || Math.abs(diff) > Math.abs(big.diff))) big = { t, a, b, diff, z };
  }
  if (ranked.length) {
    const top = ranked[0];
    frag.push(el('div', { class: 'hero-row' },
      el('div', { class: 'card' },
        el('p', { class: 'sub', text: `Best official average — all ${top.nreq} required `
          + `tasks, ${state.avgMode === 'raw' ? 'raw accuracy' : 'scaled above chance'}` }),
        el('div', { class: 'hero', text: pct(officialAvg(top)) }),
        el('p', { class: 'sub', text: top.id
          + (top.params ? ` · ${P(top.params)} params` : '') }),
        // every clause derived from the payload, so it cannot drift from the
        // table below it — including the one nobody writes down by choice
        el('p', { class: 'small', style: 'margin-top:8px',
                  text: modelSentence(top) })),
      el('div', { class: 'card' },
        el('h2', { text: 'Top models' }),
        lbMini(ranked.slice(0, 5)))));
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
    frag.push(el('p', { class: 'small', style: 'margin:10px 2px 0', text:
      `${prelim.length} preliminary model${prelim.length > 1 ? 's' : ''} `
      + `(${prelim.map(m => `${m.name} ${m.nhave}/${m.nreq}`).join(', ')}) `
      + `— per-task results only, excluded from the ranking above.` }));
  frag.push(el('div', { class: 'tiles' },
    tile('Models compared', String(ms.length),
         ms.length !== DATA.models.length ? `of ${DATA.models.length} (filtered)` : null),
    tile('Tasks', String(DATA.accTasks.length + DATA.pplTasks.length),
         `${DATA.accTasks.length} accuracy · ${DATA.pplTasks.length} perplexity`),
    tile('Real differences', pairs ? `${real} / ${pairs}` : '—',
         'pairwise comparisons that clear |z| > 1.96'),
    tile('Eval wall-clock', DATA.meta.hours >= 1 ? DATA.meta.hours + ' h'
                            : Math.round(DATA.meta.hours * 60) + ' min',
         DATA.meta.dates[0] ? (DATA.meta.dates[0] + ' → ' + DATA.meta.dates[1])
           .replaceAll('T', ' ') : null)));
  if (big) {
    const an = DATA.models.find(m => m.id === big.a), bn = DATA.models.find(m => m.id === big.b);
    const [win, lose] = big.diff > 0 ? [an, bn] : [bn, an];
    frag.push(el('div', { class: 'card' },
      el('h2', { text: 'Biggest statistically real gap' }),
      el('p', { class: 'sub', text:
        `${big.t}: ${win.name} beats ${lose.name} by ${(100 * Math.abs(big.diff)).toFixed(1)} points (z = ${Math.abs(big.z).toFixed(1)}). `
        + `${pairs - real} of ${pairs} pairwise comparisons are inside the noise — treat overlapping whiskers as ties.` })));
  }
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'How to read these numbers' }),
    note('Chance is not zero. 4-option tasks (MMLU, ARC, HellaSwag) sit at 25% for a model that knows nothing; 2-option tasks (Winogrande, PIQA) sit at 50%. A "50%" that looks respectable may be a coin flip.'),
    note('GSM8K near zero is a finding, not a failure — sub-billion models mostly cannot do written arithmetic. TruthfulQA is famous for NOT improving with scale.'),
    note('Perplexity (bits per byte) is the scale-sensitive metric here: it separates models that multiple-choice tasks cannot tell apart, and it works on base models with no prompt format at all. Multiply by ln 2 for cross-entropy loss in nats/byte — the Perplexity & Loss tab does it for you. Lower is better.'),
    note('Whiskers are ±1 standard error. If two whiskers overlap, do not call a winner — every pairwise z-test verdict rides along in the JSON export (the "sig" field) when you need the arbiter.')));
  return frag;
}
// ---------------------------------------------------------------------------
// Routing. The whole report is one file with no server, so the address bar is
// the only place a view can live — and putting it there is what makes Back work.
// A dashboard whose Back button leaves the page instead of returning to the list
// you came from is the single most reported annoyance in apps shaped like this.
// ---------------------------------------------------------------------------
const hashFor = () => state.model ? 'model=' + encodeURIComponent(state.model)
                                  : state.topic ? 'topic=' + encodeURIComponent(state.topic)
                                  : 'tab=' + state.tab;

function routeFromHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ''));
  const m = /^model=(.+)$/.exec(h);
  if (m && DATA.models.some(x => x.id === m[1])) { state.model = m[1]; state.topic = null; return; }
  state.model = null;
  // a topic page is the loop for one topic — deep-linkable, because it is the
  // page a person is sent to when someone says "look at law"
  const tp = /^topic=(.+)$/.exec(h);
  if (tp && LIVE && topicOfSlug(tp[1])) { state.topic = tp[1]; state.tab = 'loop'; return; }
  state.topic = null;
  const t = /^tab=(.+)$/.exec(h);
  if (!t) return;
  const want = TAB_ALIASES[t[1]] || t[1];
  if (TABS.some(([id]) => id === want)) state.tab = want;
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
  Object.assign(state, patch);
  const want = hashFor();
  // push history, then paint. Painting here rather than leaving it to the
  // hashchange handler is deliberate: that handler ignores a hash which already
  // agrees with state (it is our own write echoing back), so relying on it to
  // render meant a tab click updated the URL and nothing else.
  if (location.hash.slice(1) !== want) location.hash = want;
  render();
}

window.addEventListener('hashchange', () => {
  // a hash that already matches state is our own write echoing back; anything
  // else is the user pressing Back or Forward, and we adopt it
  if (location.hash.slice(1) === hashFor()) return;
  routeFromHash();
  render();
});

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

const tile = (label, value, notetext) => el('div', { class: 'tile' },
  el('div', { class: 'label', text: label }),
  el('div', { class: 'value', text: value }),
  notetext ? el('div', { class: 'note', text: notetext }) : null);
const note = t => el('p', { class: 'note', text: t });

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
      try { a.ok = (await run()) || 'done.'; }
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

function lbMini(rows) {
  const tb = el('tbody', {}, rows.map((m, i) => el('tr', {},
    el('td', { text: String(i + 1) }),
    el('td', { 'data-model': m.id }, m.name,
      ckBadge(m) || (m.kind === 'instruct'
        ? el('span', { class: 'badge instruct', text: 'instruct' }) : '')),
    el('td', { class: 'num', text: P(m.params) }),
    el('td', { class: 'num best', text: pct(officialAvg(m)) }))));
  return el('table', {},
    el('thead', {}, el('tr', {},
      el('th', { text: '#' }), el('th', { text: 'model' }),
      el('th', { class: 'num', text: 'params' }), el('th', { class: 'num', text: 'avg' }))), tb);
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
  if (state.radarNorm === 'raw' || !(c > 0)) return v;
  return Math.max(0, Math.min(1, (v - c) / (1 - c)));
}
// How many models the capability profile will hold at once. The cap exists for
// the chart, not the code: past about five overlapping polygons a radar stops
// being readable. Five is the ceiling the eight-slot palette and the eye agree
// on. The eviction below used to be silent, which read as a bug — it is now
// labelled on the card and the evicted row is named.
const CMP_MAX = 5;

// the compared set: explicit ticks, else the top few by average
function cmpEffective(ms) {
  if (state.cmpSel.length) return state.cmpSel.filter(id => ms.some(m => m.id === id));
  // prefer official models; fall back to partial averages so a report with no
  // official result still shows a profile rather than an empty card
  return [...ms].filter(m => m.partialAvg != null)
    .sort((a, b) => (officialAvg(b) ?? -1) - (officialAvg(a) ?? -1)
                 || b.partialAvg - a.partialAvg)
    .slice(0, CMP_MAX).map(m => m.id);
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
function radarCard(ms) {
  const axes = radarAxes();
  if (axes.length < 3) return null;
  const ids = cmpEffective(ms);
  const series = ids.map((id, i) => {
    const m = DATA.models.find(x => x.id === id);
    const slot = state.cmpSel.length ? state.cmpColors[id] : i;
    const vals = {};
    for (const ax of axes) {
      const parts = ax.tasks.map(t => ({ t, c: cell(t, id) })).filter(p => p.c);
      vals[ax.key] = parts.length
        ? { n: parts.reduce((s, p) => s + normScore(p.t, p.c.v), 0) / parts.length, parts }
        : null;
    }
    return { id, m, color: trColor(slot), vals };
  }).filter(s => s.m);
  const W = 540, H = 400, cx = 270, cy = 205, R = 140, N = axes.length;   // side margins fit long task names
  const ang = i => -Math.PI / 2 + 2 * Math.PI * i / N;
  const pt = (i, r) => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'radar',
    role: 'img', 'aria-label': 'capability profile' });
  for (const f of [0.25, 0.5, 0.75, 1]) {
    svg.append(el('svg:polygon', {
      points: axes.map((_, i) => pt(i, f * R).map(v => v.toFixed(1)).join(',')).join(' '),
      fill: 'none', stroke: f === 1 ? 'var(--axis)' : 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('svg:text', { x: cx + 4, y: cy - f * R + 3.5, 'font-size': 9.5,
      fill: 'var(--muted)', text: Math.round(f * 100) + '%' }));
  }
  axes.forEach((ax, i) => {
    const [x, y] = pt(i, R);
    svg.append(el('svg:line', { x1: cx, y1: cy, x2: x, y2: y, stroke: 'var(--grid)', 'stroke-width': 1 }));
    const [lx, ly] = pt(i, R + 16), c = Math.cos(ang(i));
    svg.append(el('svg:text', { x: lx, y: ly + 4, 'font-size': 11, fill: 'var(--text-secondary)',
      'text-anchor': c > 0.15 ? 'start' : c < -0.15 ? 'end' : 'middle', text: ax.label }));
  });
  const unit = state.radarNorm === 'chance' ? 'above chance' : 'accuracy';
  // marks are pointer-transparent and every hit target is appended LAST: a later
  // model's translucent fill must never swallow an earlier model's hover
  const hits = [];
  for (const s of series) {
    const pts = axes.map((ax, i) => s.vals[ax.key] ? pt(i, s.vals[ax.key].n * R) : null);
    const have = pts.filter(Boolean);
    if (have.length >= 2)
      svg.append(el('svg:path', { d: 'M' + have.map(p => p.map(v => v.toFixed(1)).join(',')).join('L') + 'Z',
        fill: s.color, 'fill-opacity': 0.12, stroke: s.color, 'stroke-width': 2,
        'stroke-linejoin': 'round', 'pointer-events': 'none' }));
    axes.forEach((ax, i) => {
      const v = s.vals[ax.key]; if (!v) return;
      const [x, y] = pts[i];
      svg.append(el('svg:circle', { cx: x, cy: y, r: 3.5, fill: s.color,
        stroke: 'var(--surface-1)', 'stroke-width': 1.5, 'pointer-events': 'none' }));
      const rows = [s.m.name, `${ax.label}: ${pct(v.n)} ${unit}`,
        ...v.parts.map(p => `${p.t}: ${pct(p.c.v)} raw` + (p.c.se ? ` ± ${(100 * p.c.se).toFixed(1)}` : ''))];
      hits.push(el('svg:circle', { class: 'hit', cx: x, cy: y, r: 8, tabindex: 0,
        'data-tip': JSON.stringify(rows), 'data-tipkey': s.color }));
    });
  }
  svg.append(...hits);
  const seg = (label, opts, cur, pick) => el('div', { class: 'seg', role: 'group', 'aria-label': label },
    opts.map(([v, l]) => el('button', { 'aria-pressed': String(cur === v), text: l,
      onclick: () => pick(v) })));
  const legend = el('div', { class: 'legend', style: 'margin:6px 0 0' }, series.map(s => {
    const missing = axes.filter(ax => !s.vals[ax.key]).map(ax => ax.label);
    return el('span', { title: missing.length ? 'not evaluated on: ' + missing.join(', ') : s.id },
      el('span', { class: 'key', style: `background:${s.color}` }), s.m.name,
      missing.length ? el('span', { class: 'se', text: ` (${missing.length} axis missing)` }) : '',
      el('button', { class: 'xbtn', 'aria-label': 'remove ' + s.m.name, text: '×',
        onclick: () => cmpToggle(s.id, ms) }));
  }));
  const table = el('table', { class: 'radar-tbl' },
    // truncated with the full id on hover: an untruncated checkpoint name here
    // sets the column's min-content width and shoves every other model out of
    // the card. The legend above already carries the names in full.
    el('thead', {}, el('tr', {}, el('th', { text: 'axis' }),
      series.map(s => el('th', { class: 'num', title: s.id,
        text: s.m.name.length > 16 ? s.m.name.slice(0, 15) + '…' : s.m.name })))),
    el('tbody', {}, axes.map(ax => el('tr', {}, el('td', { text: ax.label }),
      series.map(s => el('td', { class: 'num', text: s.vals[ax.key] ? pct(s.vals[ax.key].n) : '—' }))))));
  return el('div', { class: 'card' },
    el('h2', { text: 'Capability profile' }),
    el('p', { class: 'sub', text:
      `One axis per benchmark, one shape per model — tick up to ${CMP_MAX} in the table below. `
      + 'Axes are scaled ABOVE CHANCE by default: 25% on a '
      + '4-way task is 0, perfect is 100%, so a 4-way and a 2-way task are comparable; switch to '
      + 'raw accuracy to quote the number itself. Perplexity tasks are excluded (different scale). '
      + 'Read the shape here and the numbers below — a radar’s area exaggerates differences and '
      + 'its shape depends on axis order.' }),
    el('div', { class: 'ctrl', style: 'margin-top:8px' },
      seg('scale', [['chance', 'above chance'], ['raw', 'raw accuracy']], state.radarNorm,
          v => { state.radarNorm = v; render(); }),
      seg('axes', [['tasks', 'tasks'], ['categories', 'categories']], state.radarAxes,
          v => { state.radarAxes = v; render(); }),
      // the slot count, said out loud. A full set silently dropping a model on
      // the next tick is the single most confusing thing this card used to do.
      el('span', { class: 'small', style: 'margin-left:auto',
        text: `comparing ${ids.length} of ${CMP_MAX} slots`
            + (ids.length >= CMP_MAX ? ' — full, the next tick replaces the oldest' : '')
            + (state.cmpEvicted ? ` · dropped ${state.cmpEvicted}` : '') })),
    el('div', { class: 'radar-grid' },
      el('div', {}, svg, legend),
      el('div', { class: 'lb-wrap' }, table)));
}

// Cell shading. Polarity where a baseline exists, magnitude where it does not:
// a task WITH a chance level gets a diverging fill centred on it (blue above,
// red below, nothing at the line), because "which side of chance" is the fact
// that decides whether a number means anything on this board. A task without
// one — perplexity, gsm8k — gets a single-hue sequential fill across its own
// column range.
//
// Not red-to-green, which was the obvious thing to copy: that scheme fails the
// palette checks (the yellow sits outside the lightness band and hits 1.63:1
// against the surface) and green/yellow separate by only ΔE 10 under
// protanopia, which is roughly one man in twelve. The pair used here is the one
// already validated for this palette. Intensity tops out well below opaque and
// the number stays in a text token, so every cell is readable without colour.
function heatBg(c, v, rng) {
  if (v == null) return null;
  const i = DATA.tasks[c.task] || {};
  let hue, k;
  if (!c.lower && i.chance > 0) {
    const d = (v - i.chance) / (1 - i.chance);
    if (Math.abs(d) < 0.005) return null;            // sitting on the line
    const span = d > 0 ? 1 : (i.chance / (1 - i.chance)) || 1;
    hue = d > 0 ? 'var(--s1)' : 'var(--s8)';
    k = Math.abs(d) / span;
  } else {
    if (!rng || rng.hi === rng.lo) return null;
    const t = (v - rng.lo) / (rng.hi - rng.lo);
    k = c.lower ? 1 - t : t;                          // perplexity: lower is better
    hue = 'var(--s1)';
  }
  const a = (8 + 52 * Math.max(0, Math.min(1, k))).toFixed(1);
  return `color-mix(in srgb, ${hue} ${a}%, transparent)`;
}

// The benchmark descriptions, where someone reading the table can actually find
// them. A tooltip alone is not an answer for a friend who has never seen the
// board — this is collapsed by default so it costs nothing, and open it once and
// every column has a sentence.
function aboutBenchmarks(tasks) {
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
  const seg = (label, cur, opts, on) => el('div', { class: 'seg', role: 'group',
    'aria-label': label }, opts.map(([v, t, n]) =>
    el('button', { 'aria-pressed': String(cur === v), 'data-filter': `${label}:${v}`,
      text: n == null ? t : `${t} (${n})`, onclick: () => { on(v); render(); } })));
  const nCk = DATA.models.filter(m => m.source === 'artifact').length;
  const head = el('div', { class: 'card' },
    el('h2', { text: 'Models' }),
    el('p', { class: 'sub', text: 'Every model on this board, with what is known about it. '
      + 'The filters are this table\'s — they narrow the list below and nothing else. The '
      + 'radar and its compare ticks belong to the Leaderboard, where they are.' }),
    el('div', { class: 'toolbar' },
      el('input', { type: 'search', id: 'mq', value: f.q, style: 'flex:1;min-width:180px',
        placeholder: 'name, id or family…', 'aria-label': 'filter models',
        oninput: e => { f.q = e.target.value; render(); } }),
      seg('kind', f.kind, [['all', 'All', DATA.models.length],
        ['base', 'Base', DATA.models.filter(m => m.kind === 'base').length],
        ['instruct', 'Instruct', DATA.models.filter(m => m.kind === 'instruct').length]],
        v => { f.kind = v; }),
      nCk ? seg('source', f.src, [['all', 'All'], ['hub', 'Models'],
        ['artifact', 'Checkpoints', nCk]], v => { f.src = v; }) : '',
      mkSel('family filter', [['all', 'family: any'], ...families.map(x => [x, x])],
        f.family, v => { f.family = v; render(); }),
      el('label', { class: 'small' },
        el('input', { type: 'checkbox', 'data-filter': 'judged',
          checked: f.judgedOnly ? '' : null,
          onchange: e => { f.judgedOnly = e.target.checked; render(); } }), ' has a judged run'),
      el('label', { class: 'small' },
        el('input', { type: 'checkbox', 'data-filter': 'tainted',
          checked: f.taintedOnly ? '' : null,
          onchange: e => { f.taintedOnly = e.target.checked; render(); } }), ' tainted'),
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
  const table = el('div', { class: 'card' },
    el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-models-table': '1' },
      el('thead', {}, el('tr', {}, MCOLS.map(th), el('th', { text: 'flags' }))),
      el('tbody', {}, ms.map(m => {
        const topics = m.judge ? Object.keys(m.judge.tasks || {})
          .filter(t => t !== (DATA.judged || {}).control) : [];
        // no compare tick here: the radar is the Leaderboard's, and two tables
        // sharing one selection is what broke the Leaderboard's own ticks
        return el('tr', { 'data-model-row': m.id },
          el('td', {}, el('a', { href: '#model=' + encodeURIComponent(m.id), text: m.name,
            onclick: e => { e.preventDefault(); navigate({ model: m.id, topic: null }); } }),
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
          el('td', { class: 'small se', text: m.date ? String(m.date).slice(0, 10) : '—' }),
          el('td', {}, (m.tainted || []).length ? el('span', { class: 'badge taint',
            title: 'trained on data derived from ' + m.tainted.map(frName).join(', '),
            text: 'tainted' }) : '',
            m.provisional ? el('span', { class: 'badge taint', text: 'provisional' }) : ''));
      })))));
  if (!ms.length) table.append(note('No model matches these filters.'));
  return [head, table];
}

function vLeaderboard(ms) {
  const cmpSet = new Set(cmpEffective(ms));
  const cols = [
    { key: 'cmp',    label: '', nosort: true },
    { key: 'name',   label: 'Model',  num: false },
    { key: 'params', label: 'Params', num: true },
    { key: 'avg',    label: 'Avg',    num: true },
    ...DATA.accTasks.map(t => ({ key: t, label: t, num: true, task: t })),
    ...DATA.pplTasks.map(t => ({ key: t, label: t, num: true, task: t, lower: true })),
    // judged columns exist on the board only once a person has agreed with the
    // judge (kappa over the line); the kappa rides in the header
    ...(DATA.judged && DATA.judged.calibration && DATA.judged.calibration.calibrated
        && DATA.models.some(m => m.judgeState && m.judgeState.ok)
      ? [...DATA.judged.exam.filter(t => DATA.judged.tasks.includes(t)).map(t => ({
          key: 'j:' + t, label: frName(t) + ' κ'
            + (((DATA.judged.calibration.per_category || {})[frName(t)] || {}).kappa
               ?? DATA.judged.calibration.kappa), num: true, judged: t })),
         { key: 'javg', label: `Judged avg κ${DATA.judged.calibration.kappa}`, num: true, judged: 'avg' }]
      : []),
    { key: 'date', label: 'Last eval', num: false },   // when its newest task ran
  ];
  const jval = (m, c) => !(m.judgeState && m.judgeState.ok) ? null : c.judged === 'avg' ? m.judgedAvg
    : (m.tainted || []).includes(c.judged) ? null      // shown on the page, never ranked here
    : (((m.judge || {}).tasks || {})[c.judged] ? pubScore(m.judge.tasks[c.judged]) : null);
  const val = (m, c) => c.key === 'avg' ? officialAvg(m)
                      : c.judged ? jval(m, c)
                      : c.task ? (cell(c.task, m.id) || {}).v : m[c.key];
  const rows = [...ms].sort((a, b) => {
    const c = cols.find(c => c.key === state.sort.key) || cols.find(c => c.key === 'avg');
    const va = val(a, c), vb = val(b, c);
    if (va == null && vb == null) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return typeof va === 'string' ? state.sort.dir * natCmp(va, vb)
                                  : state.sort.dir * (va - vb);
  });
  // best per column (max for accuracy/avg, min for perplexity). Perplexity has
  // NO standard error from the harness, so a 0.001 lead is not a win: values
  // within a tie band of the leader are all marked tied (≈) instead. The band
  // is a stated placeholder until bootstrap CIs exist — see the tab text.
  const pplBand = lead => Math.max(0.005, 0.01 * Math.abs(lead));
  const best = {}, tiedCount = {}, rng = {};
  for (const c of cols) {
    if (!c.num || c.key === 'params') continue;
    const vs = ms.map(m => val(m, c)).filter(v => v != null);
    if (vs.length > 1) {
      best[c.key] = c.lower ? Math.min(...vs) : Math.max(...vs);
      tiedCount[c.key] = c.lower
        ? vs.filter(v => v <= best[c.key] + pplBand(best[c.key])).length : 1;
      // column range drives the sequential fill for tasks with no chance level
      if (c.task) rng[c.key] = { lo: Math.min(...vs), hi: Math.max(...vs) };
    }
  }
  const shotOf = t => {
    const s = [...new Set(ms.map(m => (cell(t, m.id) || {}).shots).filter(x => x != null))];
    return s.length > 1 ? 'mixed!' : s.length ? s[0] + '-shot' : '';
  };
  const thead = el('thead', {},
    el('tr', {}, cols.map(c => c.nosort
      ? el('th', { title: 'tick to compare in the capability profile above', text: '' })
      : el('th', {
      class: (c.num ? 'num ' : '') + 'sortable' + (c.key === 'name' ? ' model' : '')
           + (c.judged ? ' judged' : '')
           + (c.task && (DATA.tasks[c.task] || {}).desc ? ' hasinfo' : ''),
      // the description on the column itself; the full list is in the panel
      // below the table, because a tooltip is not documentation
      title: c.judged ? 'rubric score 0–4 from the calibrated judge — a separate judged average, '
          + 'never part of Avg' : c.task ? [(DATA.tasks[c.task] || {}).control ? 'CONTROL — never in Avg' : null,
                       (DATA.tasks[c.task] || {}).domain,
                       (DATA.tasks[c.task] || {}).desc].filter(Boolean).join(' — ') : null,
      onclick: () => { state.sort = { key: c.key,
        dir: state.sort.key === c.key ? -state.sort.dir : (c.key === 'name' ? 1 : c.lower ? 1 : -1) };
        render(); },
      'aria-sort': state.sort.key === c.key ? (state.sort.dir > 0 ? 'ascending' : 'descending') : 'none' },
      c.label + ' ', state.sort.key === c.key
        ? el('span', { class: 'dir', text: state.sort.dir > 0 ? '▲' : '▼' }) : ''))),
    el('tr', {}, cols.map(c => el('th', {
      class: (c.num ? 'num' : '') + (c.key === 'name' ? ' model' : ''),
      text: c.judged ? 'rubric 0–4' : c.task ? (c.lower ? DATA.tasks[c.task].metric : shotOf(c.task)) : '' }))));
  const tbody = el('tbody', {}, rows.map(m => el('tr', {},
    cols.map(c => {
      if (c.key === 'cmp') return el('td', {}, el('input', { type: 'checkbox',
        'aria-label': 'compare ' + m.name, checked: cmpSet.has(m.id) ? '' : null,
        onchange: () => cmpToggle(m.id, ms) }));
      if (c.key === 'name') return el('td', { class: 'model', 'data-model': m.id,
        title: modelSentence(m) + '\n\n' + m.id + (m.archinfo && m.archinfo.hidden
          ? `\n${m.archinfo.arch || ''} · hidden ${m.archinfo.hidden} · layers ${m.archinfo.layers} · vocab ${m.archinfo.vocab}` : '') },
        // the name truncates, the badges never do: a long checkpoint id used to
        // set the column's min-content width and push every task off-screen
        // a real link, not a click handler: middle-click, copy-link-address and
        // the Back button all work for free because the view lives in the URL
        el('a', { class: 'mname mlink', text: m.name,
                  href: '#model=' + encodeURIComponent(m.id) }),
        ckBadge(m) || (m.kind === 'instruct'
          ? el('span', { class: 'badge instruct', text: 'instruct' })
          : el('span', { class: 'badge', text: 'base' })),
        prelimBadge(m) || '', taintBadge(m) || '', dupBadge(m) || '');
      if (c.key === 'params') {
        const a = m.archinfo || {};
        // a sparse model loads every expert but routes each token through a few:
        // total drives VRAM, active drives the fair comparison, so show both
        return el('td', { class: 'num',
          title: (m.paramsSrc ? 'from ' + (m.paramsSrc === 'config' ? 'harness config' : 'model name') : '')
            + (a.active_params ? `\n${a.experts} experts, ${a.experts_per_tok} per token`
                 + `\nactive ${P(a.active_params)} of ${P(m.params)} (${a.active_src})` : '') },
          P(m.params),
          a.active_params ? el('span', { class: 'se', text: ` ${P(a.active_params)} act` }) : '');
      }
      if (c.key === 'date') return el('td', { class: 'small', style: 'white-space:nowrap',
        text: String(m.date || '—').slice(0, 16).replace('T', ' ') });
      if (c.key === 'avg') {
        const a = officialAvg(m);
        if (a == null) return el('td', { class: 'num' },
          el('span', { class: 'se',
            title: (m.missing || []).length ? 'missing: ' + m.missing.join(', ') : '',
            text: `— ${m.nhave}/${m.nreq}` }));
        const r = rankOf(m);
        return el('td', { class: 'num' + (a === best.avg ? ' best' : ''),
          title: `mean over the ${m.nreq} required tasks, `
            + (state.avgMode === 'raw' ? 'raw accuracy' : 'scaled so chance = 0')
            + (r ? `\nrank ${r.n} of ${r.of} ranked models on this board (not of the `
                 + `filtered view)` : '') },
          pct(a),
          // the rank is over the whole board, so it does not move when you filter
          r ? el('span', { class: 'se', text: ` #${r.n}/${r.of}` })
            : el('span', { class: 'se', text: ` ${m.nreq}/${m.nreq}` }));
      }
      if (c.judged) {
        const v = jval(m, c);
        if (v == null) return el('td', { class: 'num se', text: '—',
          title: (m.tainted || []).includes(c.judged)
            ? 'trained on data derived from this topic — shown on the model page, not ranked'
            : m.judgeState ? m.judgeState.reasons.join('; ') : 'not judged' });
        return el('td', { class: 'num' + (v === best[c.key] ? ' best' : ''), text: num(v, 2) },
          el('span', { class: 'se', text: ' /4' }));
      }
      const cc = cell(c.task, m.id);
      if (!cc) return el('td', { class: 'num', text: '—' });
      const lead = best[c.key];
      const within = lead != null && (c.lower
        ? cc.v <= lead + pplBand(lead) : cc.v === lead);
      const mark = !within ? '' : (c.lower && tiedCount[c.key] > 1) ? ' tiebest' : ' best';
      return el('td', { class: 'num' + mark,
        style: state.lbHeat ? `background:${heatBg(c, cc.v, rng[c.key]) || 'none'}` : null,
        title: mark === ' tiebest'
          ? 'tied for best — perplexity carries no standard error here, so a lead '
            + 'this small is not a difference' : '' },
        c.lower ? num(cc.v, 3) : pct(cc.v),
        cc.se && !c.lower ? el('span', { class: 'se', text: ` ±${(100 * cc.se).toFixed(1)}` }) : '');
    }))));
  const nOff = ms.filter(m => m.official).length;
  // the table first: it is what the tab is named after and what most visits
  // want. The radar follows, drawn from whatever the compare ticks hold
  return [el('div', { class: 'card' },
    el('h2', { text: 'Leaderboard' }),
    el('p', { class: 'sub', text: 'Click a column to sort. Accuracy cells are score ± stderr; '
      + 'perplexity columns are lower-is-better, excluded from Avg, and carry no standard '
      + 'error — so ● marks a best value and ≈ marks values too close to call. '
      + `Avg exists only for models that completed all ${DATA.required.length} required `
      + `tasks (${DATA.required.join(', ')}): ${nOff} of ${ms.length} here. `
      + 'Anything short of that is preliminary — its per-task scores are valid and shown, '
      + 'it just has no overall number.' }),
    el('div', { class: 'ctrl', style: 'margin:8px 0 2px' },
      el('span', { class: 'small', text: 'Avg scale' }),
      el('div', { class: 'seg', role: 'group', 'aria-label': 'average scale' },
        [['chance', 'above chance'], ['raw', 'raw accuracy']].map(([v, l]) =>
          el('button', { 'aria-pressed': String(state.avgMode === v), text: l,
            onclick: () => { state.avgMode = v; render(); } }))),
      el('span', { class: 'count-note', text: state.avgMode === 'raw'
        ? 'raw: the number you quote, but a 2-option task starts at 50%'
        : 'chance = 0, perfect = 100% — comparable across tasks with different guess rates' })),
    el('div', { class: 'ctrl', style: 'margin:2px 0 6px' },
      el('span', { class: 'small', text: 'Cells' }),
      el('div', { class: 'seg', role: 'group', 'aria-label': 'cell shading' },
        [[false, 'numbers'], [true, 'heat']].map(([v, l]) =>
          el('button', { 'aria-pressed': String(state.lbHeat === v), text: l,
            onclick: () => { state.lbHeat = v; render(); } }))),
      state.lbHeat ? heatLegend() : el('span', { class: 'count-note',
        text: 'shade every score by how far it is from chance — useful once the '
            + 'table is taller than the screen' })),
    lbViewCtrl(ms),
    state.lbView === 'cats' ? lbCategoryTable(ms)
      : el('div', { class: 'lb-wrap' }, el('table', { class: 'lb' }, thead, tbody))),
    radarCard(ms) || '',
    aboutBenchmarks([...DATA.accTasks, ...DATA.pplTasks])];
}

// The optional "MMLU by category" view: one column per category from
// scripts/categories.yaml, one row per model that has a diagnosis, so a trained
// checkpoint's profile can be set against the reference models. Every number
// is a LEADERBOARD-half score read from diagnose.json — the same half the MMLU
// column comes from — so nothing here is a diagnosis-half claim.
const mmluCats = m => ((((m.diag || {}).tasks || {}).mmlu || {}).categories) || null;

function lbViewCtrl(ms) {
  const n = ms.filter(mmluCats).length;
  return el('div', { class: 'ctrl', style: 'margin:2px 0 6px' },
    el('span', { class: 'small', text: 'Columns' }),
    el('div', { class: 'seg', role: 'group', 'aria-label': 'leaderboard columns' },
      [['tasks', 'tasks'], ['cats', 'MMLU by category']].map(([v, l]) =>
        el('button', { 'aria-pressed': String(state.lbView === v), text: l,
          disabled: (v === 'cats' && !n) ? '' : null,
          title: v === 'cats' && !n ? 'no model here has an MMLU diagnosis on file' : null,
          onclick: () => { state.lbView = v; render(); } }))),
    el('span', { class: 'count-note', text: state.lbView === 'cats'
      ? `leaderboard-half MMLU score per category, from each model's diagnosis (${n} of `
        + `${ms.length} have one) — categories under ${CAT_MIN_N} items are greyed`
      : 'switch to see MMLU broken down by category for every diagnosed model' }));
}

function lbCategoryTable(ms) {
  const have = ms.filter(mmluCats);
  const cats = (DATA.meta.categories || []).filter(c => have.some(m => mmluCats(m)[c]));
  const cols = [{ key: 'name', label: 'Model' }, { key: 'mmlu', label: 'mmlu', task: true },
                ...cats.map(c => ({ key: 'cat:' + c, label: c, cat: c }))];
  const val = (m, c) => c.key === 'name' ? m.name
    : c.task ? (cell('mmlu', m.id) || {}).v
    : ((mmluCats(m)[c.cat] || {}).score_report);
  const sortKey = cols.some(c => c.key === state.sort.key) ? state.sort.key : 'mmlu';
  const dir = sortKey === state.sort.key ? state.sort.dir : -1;
  const rows = [...have].sort((a, b) => {
    const c = cols.find(c => c.key === sortKey);
    const va = val(a, c), vb = val(b, c);
    if (va == null && vb == null) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return typeof va === 'string' ? dir * natCmp(va, vb) : dir * (va - vb);
  });
  const best = {};
  for (const c of cols) if (c.key !== 'name') {
    const vs = have.map(m => val(m, c)).filter(v => v != null);
    if (vs.length > 1) best[c.key] = Math.max(...vs);
  }
  const th = c => el('th', {
    class: (c.key === 'name' ? 'model ' : 'num ') + 'sortable',
    'aria-sort': sortKey === c.key ? (dir > 0 ? 'ascending' : 'descending') : 'none',
    onclick: () => { state.sort = { key: c.key,
      dir: state.sort.key === c.key ? -state.sort.dir : (c.key === 'name' ? 1 : -1) };
      render(); } },
    c.label + ' ', sortKey === c.key ? el('span', { class: 'dir', text: dir > 0 ? '▲' : '▼' }) : '');
  const table = el('table', { class: 'lb lbcats' },
    el('thead', {}, el('tr', {}, cols.map(th))),
    el('tbody', {}, rows.map(m => el('tr', {}, cols.map(c => {
      if (c.key === 'name') return el('td', { class: 'model', 'data-model': m.id },
        el('a', { class: 'mname mlink', text: m.name, href: '#model=' + encodeURIComponent(m.id) }),
        prelimBadge(m) || '');
      if (c.task) {
        const cc = cell('mmlu', m.id);
        return el('td', { class: 'num' + (cc && cc.v === best.mmlu ? ' best' : ''),
          style: state.lbHeat && cc ? `background:${heatBg({ task: 'mmlu' }, cc.v) || 'none'}` : null },
          cc ? pct(cc.v) : '—',
          cc && cc.se ? el('span', { class: 'se', text: ` ±${(100 * cc.se).toFixed(1)}` }) : '');
      }
      const g = mmluCats(m)[c.cat];
      if (!g || g.score_report == null) return el('td', { class: 'num', text: '—' });
      const dim = g.n_report < CAT_MIN_N;
      return el('td', { class: 'num' + (dim ? ' dim' : '')
          + (!dim && g.score_report === best[c.key] ? ' best' : ''),
        style: state.lbHeat && !dim ? `background:${heatBg({ task: 'mmlu' }, g.score_report) || 'none'}` : null,
        title: `${g.n_report} leaderboard-half items` + (dim ? ` — under ${CAT_MIN_N}, treat as noise` : '')
          + ` · subjects: ${(g.groups || []).join(', ')}` },
        pct(g.score_report), el('span', { class: 'se', text: ` ${g.n_report}` }));
    })))));
  const skipped = ms.length - have.length;
  return el('div', {},
    el('div', { class: 'lb-wrap' }, table),
    el('p', { class: 'small', style: 'margin:8px 0 0', text:
      'Each cell: leaderboard-half accuracy on that category, then its item count. '
      + (skipped ? `${skipped} model${skipped > 1 ? 's' : ''} without a diagnosis on file `
                 + '(run scripts/diagnose.py) are not shown in this view.' : '') }));
}

// the key for the shading — colour alone is never the only encoding here (the
// number is in every cell), but an unexplained colour is still a puzzle
function heatLegend() {
  const chip = (bg, label) => el('span', { class: 'hl-item' },
    el('span', { class: 'hl-sw', style: `background:${bg}` }), label);
  return el('span', { class: 'hl' },
    chip('color-mix(in srgb, var(--s8) 52%, transparent)', 'below chance'),
    chip('none', 'at chance'),
    chip('color-mix(in srgb, var(--s1) 22%, transparent)', 'above'),
    chip('color-mix(in srgb, var(--s1) 60%, transparent)', 'far above'),
    el('span', { class: 'small', text: '· perplexity shades by column range' }));
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
  const provRows = [...ms].sort((a, b) => {
    const va = pv(a, provCol), vb = pv(b, provCol);
    if (va === vb) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return state.provSort.dir * (provCol.num ? va - vb : natCmp(va, vb));
  });
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'Run provenance' }),
    el('p', { class: 'sub', text: 'Every field here can change a score. Publish this table with the numbers, or the numbers are hearsay. Sorted newest-eval-first — click any column to re-sort.' }),
    el('div', { class: 'lb-wrap' }, el('table', {},
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
        el('td', { 'data-model': m.id }, m.name, ckBadge(m) || ''),
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
    await Promise.all(state.trSel.map(async id => {
      const row = state.trRuns.find(r => r.id === id);
      if (force || !state.trSeries[id] || (row && row.status === 'running'))
        state.trSeries[id] = await api(`api/truns/${id}`);
    }));
    if (state.tab === 'training') (state.trRedraw || render)();
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
    listWrap.replaceChildren(...(rs.length ? rs.map(runRow)
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
        if (failedEvs) statusLine += `, ${failedEvs} failed (see Submit & Queue)`;
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

// ---------- live mode: submit + queue (only reachable when served by the API) ----------
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const ACTIVE_STATUS = new Set(['preflight', 'waiting_gpu', 'waiting_lock', 'running']);
const stClass = s => s === 'done' ? 'st st-done' : s === 'failed' ? 'st st-failed'
                   : ACTIVE_STATUS.has(s) ? 'st st-active' : 'st st-muted';

function vQueue() {
  // the judged suite's availability (and its reason when it has none) comes
  // from the same endpoint the Loop tab reads
  if (!state.loop.loaded && netReady()) loadLoop();
  const f = {
    hf_id: el('input', { type: 'text', style: 'flex:2;min-width:260px',
      placeholder: 'org/model on the Hub, or local/<name> for an uploaded artifact' }),
    kind: el('select', {}, ['auto', 'base', 'instruct'].map(v =>
      el('option', { value: v, text: v === 'auto' ? 'kind: auto-detect' : 'kind: ' + v }))),
    // judged is offered even when it cannot run: an option that is simply
    // absent tells a person nothing, and 'why is there no judged suite?' was
    // the first question asked of this page
    suite: el('select', {},
      el('option', { value: 'full', text: 'full — all tasks, comparable' }),
      el('option', { value: 'quick', text: 'quick — hellaswag + arc_easy + ppl, minutes' }),
      el('option', { value: 'control', text: 'control — mmlu_perm only: MMLU with the '
        + 'options rotated (the position-bias experiment), ~a fifth of a full MMLU' }),
      el('option', { value: 'judged', disabled: state.loop.blocked ? '' : null,
        title: state.loop.blocked || '',
        text: 'judged — the written exam, graded by the judge'
          + (state.loop.blocked ? ' (unavailable)' : '') })),
    submitter: el('input', { type: 'text', placeholder: 'your name', style: 'width:130px' }),
    note: el('input', { type: 'text', placeholder: 'note (optional)', style: 'flex:1;min-width:140px' }),
  };
  const btn = el('button', { text: 'Submit model', onclick: async () => {
    const body = { hf_id: f.hf_id.value.trim(), kind: f.kind.value, suite: f.suite.value,
                   submitter: f.submitter.value, note: f.note.value };
    if (!body.hf_id) { state.qmsg = 'enter a Hugging Face model id first'; render(); return; }
    btn.disabled = true;
    try {
      const r = await fetch('api/submissions', { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
        body: JSON.stringify(body) });
      const j = await r.json().catch(() => ({}));
      state.qmsg = r.ok ? `#${j.id}: ${j.note || 'queued'}`
                        : 'rejected: ' + (typeof j.detail === 'string' ? j.detail : r.status);
    } catch (e) { state.qmsg = 'submit failed — server unreachable?'; }
    await loadQueue(); render();
  }});
  const qrow = r => el('tr', {},
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
    el('td', { class: 'small' }, r.suite,
      // a judged row says what it sat: one topic is now the usual unit
      (() => { let t = []; try { t = JSON.parse(r.tasks || '[]'); } catch (e) { /* older row */ }
        return t.length ? el('div', { class: 'se', 'data-row-tasks': '1',
          text: t.map(frName).join(', ') }) : ''; })()),
    el('td', { text: r.submitter || '—' }),
    el('td', {}, el('span', { class: stClass(r.status), text: r.status })),
    el('td', { class: 'small', text: r.progress || '' },
      // the GPU half finishing is not the job finishing: the judge batch is
      // still out, and the row says how far it is
      r.judge ? el('div', { class: 'se', 'data-judge-progress': judgeCount(r.judge),
        title: `judge batch ${r.judge.batch_id}`,
        text: r.judge.status === 'done' ? `judged ${r.judge.n_items} answers`
          : `judging ${judgeCount(r.judge)}` }) : '',
      r.error ? el('div', { class: 'down', text: r.error }) : ''),
    el('td', { class: 'num', text: r.gpu_seconds ? Math.round(r.gpu_seconds / 60) + ' min' : '—' }),
    el('td', {},
      el('a', { href: `api/runs/${r.id}/log`, target: '_blank', rel: 'noopener', text: 'log' }),
      r.status === 'queued' ? el('button', { style: 'margin-left:8px;padding:2px 8px;font-size:12px',
        text: 'cancel', onclick: async () => {
          await fetch(`api/submissions/${r.id}/cancel`, { method: 'POST' }).catch(() => {});
          await loadQueue(); render();
        }}) : ''));
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
  const qTableWrap = el('div', { class: 'lb-wrap' }, el('table', {}, qThead, qTbody));
  const qEmpty = el('p', { class: 'small', text: 'Nothing submitted yet.' });
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
    qTbody.replaceChildren(...rs.map(qrow));
  }
  state.queueRedraw = rebuildQueue;
  rebuildQueue();
  return [
    el('div', { class: 'card' },
      el('h2', { text: 'Submit a model' }),
      el('p', { class: 'sub', text:
        'Any public (or server-accessible) Hugging Face model up to the size cap. Preflight '
        + 'checks the repo before any GPU is spent; one run at a time, per-task resume — '
        + 'resubmitting a finished model costs nothing, and a quick run upgrades to full by '
        + 'running only the missing tasks. Results land on this leaderboard automatically.' }),
      el('div', { class: 'frm' }, f.hf_id, f.kind, f.suite, f.submitter, f.note, btn),
      state.qmsg ? el('p', { class: 'small', style: 'margin-top:8px', text: state.qmsg }) : ''),
    el('div', { class: 'card' },
      el('h2', { text: 'Queue' }),
      qToolbar, qTableWrap, qEmpty)];
}

// ---------- Review: the human in the loop ----------
// Proposals wait here for a person. Everything DIAGNOSE.md says that person
// needs is on the card: the spec, the evidence counts, the category's score and
// ceiling, the item counts, the model's own findings for the task, and up to
// eight diagnosis-half items the LLM saw, labelled as such. Approve (edited or
// not) under a typed name, or reject with a reason. Only an approved spec can
// be sent to the generator, and only the spec text goes.
async function loadReview() {
  try {
    const [llm, props, ds] = await Promise.all([
      api('api/llm'), api('api/proposals'), api('api/datasets')]);
    const changed = !state.rv.loaded || JSON.stringify([llm, props, ds])
      !== JSON.stringify([state.rv.llm, state.rv.proposals, state.rv.datasets]);
    Object.assign(state.rv, { llm, proposals: props, datasets: ds, loaded: true });
    if (changed && state.tab === 'review' && !state.model) render();
  } catch (e) { /* server briefly away */ }
}

async function rvPost(path, body) {
  const r = await fetch(path, { method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
    body: JSON.stringify(body) }).catch(() => null);
  const j = r ? await r.json().catch(() => ({})) : {};
  state.rv.msg = r && r.ok ? '' : 'refused: ' + (j.detail || (r ? r.status : 'server unreachable'));
  await loadReview(); render();
}

// The name every decision is recorded under. It is remembered for this
// browser and restored at boot, not per tab: typing it on the Loop board and
// finding it gone on the topic page is the same person, one name.
function rememberedName() {
  if (state.rvName) return state.rvName;
  try { state.rvName = localStorage.getItem('bench-name') || ''; } catch (e) { /* private mode */ }
  return state.rvName;
}

let _nameKey = 0;
function rvNameInput() {
  // data-keep: render() puts focus and caret back on this node after a poll
  // rebuilds the view, so a 5-second refresh cannot eat what is being typed
  const key = 'name-' + (++_nameKey);
  return el('input', { type: 'text', placeholder: 'your name (recorded)',
    value: rememberedName(), 'aria-label': 'your name', 'data-keep': key,
    oninput: e => { state.rvName = e.target.value;
      try { localStorage.setItem('bench-name', state.rvName); } catch (err) { /* private mode */ }
      // every other copy of the field on this page follows along
      for (const o of document.querySelectorAll('input[data-keep^="name-"]'))
        if (o !== e.target) o.value = state.rvName;
    } });
}

function rvFindings(p) {
  const ev = p.evidence || {};
  const m = DATA.models.find(x => x.id === p.model);
  const t = m && m.judge && m.judge.tasks[p.task];
  const out = [el('div', { class: 'kvs' },
    el('span', {}, el('b', { text: p.category + ' ' }),
      ev.topic_score_report != null ? `${num(ev.topic_score_report, 2)} / 4` : '—',
      el('span', { class: 'se', text: ` on ${ev.topic_n_report ?? '—'} report-half questions` })),
    el('span', {}, el('b', { text: 'diagnosis half ' }),
      ev.topic_score_diagnose != null ? `${num(ev.topic_score_diagnose, 2)} / 4` : '—',
      el('span', { class: 'se', text: ` on ${ev.topic_n_diagnose ?? '—'} answers · `
        + `${ev.diagnose_weak ?? '—'} fell short` })),
    el('span', {}, el('b', { text: 'judge ' }), ev.judge_id || '—'))];
  const caution = (t && t.propose && t.propose.caution) || ev.mmlu_caution;
  if (caution)
    out.push(el('p', { class: 'small' }, el('b', { text: 'MMLU for this category: ' }), caution));
  if (m && m.judgeState && !m.judgeState.ok)
    out.push(el('p', { class: 'warn', text: 'This model\'s judged numbers are preliminary: '
      + m.judgeState.reasons.join('; ') + '. Read the spec, but do not treat the topic score '
      + 'as established.' }));
  if (!t)
    out.push(el('p', { class: 'small', text: 'The judged run this was proposed from is no longer '
      + 'on the board (re-judged, or the model was removed).' }));
  return el('div', {}, out);
}

function rvExamples(ev) {
  const ex = ev.examples || [];
  if (!ex.length) return '';
  const det = el('details', {}, el('summary', { class: 'small', style: 'cursor:pointer',
    text: `Show ${ex.length} of the ${ev.n_shown} judge assessments the LLM saw — diagnosis `
      + 'half only, question text removed' }));
  det.append(el('div', { style: 'margin-top:8px' }, ex.map(e => el('div', { class: 'ex' },
    el('div', { text: e.justification }),
    el('div', { class: 'kv' }, el('b', { text: `scored ${e.score} of 4` }),
      e.answer_words != null ? ` · the model wrote ${e.answer_words} words` : '',
      e.qid ? ` · ${String(e.qid).slice(0, 10)}` : '')))));
  return det;
}

// ---------- the Review tab starts from a topic ----------
// Pick the topic the exam says is weakest, see where every model stands on it
// and what the judge wrote about the diagnosis-half answers that fell short,
// then propose. The order here is the loop's order.
async function loadJust(mid, topic) {
  const key = mid + '|' + topic;
  if (state.rv.just[key]) return;
  state.rv.just[key] = { loading: true, items: [] };
  try {
    const r = await fetch(`api/judge/justifications?model=${encodeURIComponent(mid)}`
      + `&topic=${encodeURIComponent(topic)}`);
    state.rv.just[key] = await r.json();
  } catch (e) { state.rv.just[key] = { items: [], error: 'could not load' }; }
  if (state.tab === 'review' && !state.model) render();
}

function rvJust(mid, topic) {
  const key = mid + '|' + topic;
  const j = state.rv.just[key];
  const det = el('details', { class: 'dxex', open: state.rv.justOpen === key ? '' : null },
    el('summary', { text: 'What the judge wrote about the answers that fell short '
      + '(diagnosis half only, question text removed)',
      onclick: () => { state.rv.justOpen = state.rv.justOpen === key ? '' : key;
                       loadJust(mid, topic); } }));
  if (state.rv.justOpen !== key) return det;
  if (!j || j.loading) { det.append(el('p', { class: 'small', text: 'loading…' })); return det; }
  if (j.error || !(j.items || []).length) {
    det.append(el('p', { class: 'small', text: j.error
      || 'The judge wrote no assessment of a diagnosis-half answer that fell short here.' }));
    return det;
  }
  det.append(el('p', { class: 'small', text: `${j.counts.diagnose_weak} of `
    + `${j.counts.diagnose_items} diagnosis-half answers scored below 3 of 4; the first `
    + `${j.items.length} are shown. This is exactly what a proposal would be built from.` }));
  det.append(el('ul', {}, j.items.map(it => el('li', {},
    el('span', { class: 'q', text: it.justification }),
    el('span', { class: 'kv' }, el('b', { text: `scored ${it.score} of 4` }),
      it.answer_words != null ? ` · the model wrote ${it.answer_words} words` : '')))));
  return det;
}

function rvTopicPicker() {
  const J = DATA.judged || {};
  const rows = [];
  for (const task of (J.exam || [])) {
    const on = DATA.models
      .filter(m => m.judge && m.judge.tasks[task] && pubScore(m.judge.tasks[task]) != null)
      .map(m => ({ m, v: pubScore(m.judge.tasks[task]), t: m.judge.tasks[task] }))
      .sort((a, b) => a.v - b.v);
    if (on.length) rows.push({ task, topic: frName(task), on, weakest: on[0] });
  }
  rows.sort((a, b) => a.weakest.v - b.weakest.v);
  if (!rows.length)
    return el('div', { class: 'card' }, el('h2', { text: 'Pick a topic' }),
      note('No model has sat the exam yet. Write the bank on the Exam tab, then submit a '
        + 'model with suite=judged.'));
  const card = el('div', { class: 'card' }, el('h2', { text: 'Pick a topic' }),
    el('p', { class: 'sub', text: 'The exam is the instrument: it says, per topic, how good '
      + 'each model is and why. Weakest first, on the report half — the diagnosis half is '
      + 'what a proposal may read and is never the score.' }),
    el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
      el('thead', {}, el('tr', {}, el('th', { text: 'topic' }),
        el('th', { class: 'num', text: 'weakest model' }), el('th', { class: 'num', text: 'its score' }),
        el('th', { class: 'num', text: 'models judged' }),
        el('th', { class: 'num', text: 'report-half questions' }), el('th', { text: '' }))),
      el('tbody', {}, rows.map(r => {
        const nr = r.weakest.t.n_report ?? r.weakest.t.n;
        return el('tr', { class: (state.rv.topic === r.topic ? 'domrow' : null)
                            + (nr < CAT_MIN_N ? ' dim' : ''), 'data-pick': r.topic },
          el('td', { text: r.topic }),
          el('td', { class: 'num' }, el('a', { class: 'mlink', text: r.weakest.m.name,
            href: '#model=' + encodeURIComponent(r.weakest.m.id) })),
          el('td', { class: 'num', text: `${num(r.weakest.v, 2)} / 4` }),
          el('td', { class: 'num se', text: String(r.on.length) }),
          el('td', { class: 'num se', text: String(nr) + (nr < CAT_MIN_N ? ' · noise' : '') }),
          el('td', {}, el('button', { class: 'tgl' + (state.rv.topic === r.topic ? ' on' : ''),
            text: state.rv.topic === r.topic ? 'chosen' : 'choose',
            onclick: () => { state.rv.topic = state.rv.topic === r.topic ? '' : r.topic;
                             state.rv.justOpen = ''; render(); } })));
      })))));
  return card;
}

function rvTopicDetail(llmOk) {
  const topic = state.rv.topic;
  if (!topic) return '';
  const task = 'exam_' + topic.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  const on = DATA.models
    .filter(m => m.judge && m.judge.tasks[task] && pubScore(m.judge.tasks[task]) != null)
    .map(m => ({ m, v: pubScore(m.judge.tasks[task]), t: m.judge.tasks[task] }))
    .sort((a, b) => a.v - b.v);
  const card = el('div', { class: 'card', 'data-topic-detail': topic },
    el('h2', { text: topic + ' — across the board' }),
    el('p', { class: 'sub', text: 'Every model that has sat this topic, weakest first. Read '
      + 'the judge before proposing: if the answers are empty or the suite is preliminary, '
      + 'there is no topic gap to fix here.' }));
  for (const { m, v, t } of on) {
    const g = t.propose;
    const nr = t.n_report ?? t.n;
    const row = el('div', { class: 'rv', 'data-topic-model': m.id },
      el('div', { class: 'mhead' },
        el('h3', {}, el('a', { class: 'mlink', text: m.name,
          href: '#model=' + encodeURIComponent(m.id) })),
        el('span', { class: 'small', text: `${num(v, 2)} / 4 on ${nr} report-half questions `
          + `· ${num(t.score_diagnose, 2)} / 4 on ${t.n_diagnose} diagnosis-half`
          + ((m.tainted || []).includes(task) ? ' · trained on this topic' : '') })),
      el('div', { class: 'frm' }, g ? proposeBtn(m.id, topic, g) : '',
        g && !g.ok ? el('span', { class: 'propwhy', title: g.why,
          text: 'no proposal: ' + (g.short || g.why) }) : '',
        g && g.caution ? el('span', { class: 'propwhy', text: 'caution — MMLU: ' + g.caution }) : ''),
      rvJust(m.id, topic));
    card.append(row);
  }
  return card;
}

function rvProposal(p, llmOk) {
  const ev = p.evidence || {};
  const head = el('div', { class: 'mhead' },
    el('h3', { text: `#${p.id} · ${p.model} · ${p.task} · ${p.category}` }),
    el('span', { class: stClass(p.status === 'proposed' ? 'queued' : p.status === 'pending'
      ? 'running' : p.status === 'approved' ? 'done' : p.status), text: p.status }));
  const meta = el('p', { class: 'small', text:
    `requested ${p.created_at ? rel(p.created_at) + ' ago' : ''}`
    + (p.requested_by ? ` by ${p.requested_by}` : '')
    + (p.proposer ? ` · proposed by ${p.proposer}` : '')
    + (p.approver ? ` · ${p.status} by ${p.approver}` : '')
    + (p.prompt_sha ? ` · prompt ${p.prompt_sha.slice(0, 12)}` : '') });
  const card = el('div', { class: 'rv', 'data-proposal': p.id }, head, meta);
  if (p.status === 'pending')
    card.append(el('p', { class: 'small', text: 'Waiting for the LLM batch to complete '
      + `(batch ${p.batch_id}). Batches take minutes to hours; this page polls.` }));
  if (p.error) card.append(el('p', { class: 'warn', text: p.error }));
  if (ev.provisional) card.append(el('p', { class: 'warn', 'data-provisional': 'proposal' },
    el('b', { text: 'Provisional. ' }), `${upFirst(ev.provisional_reason)}: ${ev.served_model} at `
    + `${ev.base_url}` + (ev.weights ? ` (weights ${ev.weights})` : '') + '.'));
  if (p.spec_text) {
    card.append(el('div', { class: 'dxh', text: 'Proposed skill spec (the LLM\'s words)' }));
    card.append(el('p', { class: 'spec', text: p.spec_text }));
    card.append(el('div', { class: 'kvs' },
      ev.share_explained != null ? el('span', {}, el('b', { text: 'explains ' }),
        pct(ev.share_explained, 0) + ' of the failures shown') : '',
      el('span', {}, el('b', { text: 'failures ' }),
        `${ev.diagnose_wrong ?? '—'} of ${ev.diagnose_items ?? '—'} diagnosis-half items in `
        + `${p.category}; ${ev.n_shown ?? '—'} shown to the LLM`)));
    if ((ev.patterns || []).length)
      card.append(el('ul', { class: 'small', style: 'margin:4px 0 6px' },
        ev.patterns.map(x => el('li', { text: x }))));
  }
  if (p.edited_text) {
    card.append(el('div', { class: 'dxh', text: 'Approved as edited' }));
    card.append(el('p', { class: 'spec', text: p.edited_text }));
  }
  card.append(el('div', { class: 'dxh', text: 'What the diagnosis says about this task' }));
  card.append(rvFindings(p));
  card.append(rvExamples(ev));
  if (p.status === 'proposed') {
    const ta = el('textarea', { 'aria-label': 'skill spec to approve' });
    ta.value = p.spec_text;
    const reason = el('input', { type: 'text', placeholder: 'reason (for reject)',
      style: 'flex:1;min-width:160px', 'aria-label': 'reject reason' });
    const name = rvNameInput();
    card.append(el('div', { class: 'dxh', text: 'Decide' }),
      el('p', { class: 'small', text: 'Edit the spec if it names the wrong skill or leaks a '
        + 'question. Whatever text is in the box is what the generator will receive — and '
        + 'the only thing it receives.' }),
      ta,
      el('div', { class: 'frm' }, name,
        el('button', { text: 'Approve this spec', onclick: () => rvPost(
          `api/proposals/${p.id}/approve`, { approver: name.value, edited_text: ta.value }) }),
        reason,
        el('button', { text: 'Reject', onclick: () => rvPost(
          `api/proposals/${p.id}/reject`, { approver: name.value, reason: reason.value }) })));
  }
  if (p.status === 'approved') {
    const count = el('input', { type: 'number', value: '20', min: '1', max: '1000',
      style: 'width:80px', 'aria-label': 'item count' });
    const fmt = mkSel('format', [['doc', 'documents — prose that teaches the skill'],
                                 ['free', 'question and answer (comparison only)']],
      'doc', () => {});
    const name = rvNameInput();
    const usage = state.rv.llm || {};
    card.append(el('div', { class: 'dxh', text: 'Generate' }),
      el('p', { class: 'small', text: 'The generator receives the approved spec above, the '
        + 'topic, the count, the format and a style constraint. No benchmark item and no exam '
        + 'question, in any form. Documents are the default: prose a person could learn from, '
        + 'because question-and-answer pairs shaped like the exam are the most direct route to '
        + 'teaching the test there is. Every document then passes the 13-gram contamination '
        + 'gate against both halves of every benchmark AND every exam question. Today: '
        + `${usage.usage_today ?? '—'} of ${usage.daily_cap ?? '—'} batch items used.` }),
      el('div', { class: 'frm' }, count, fmt, name,
        el('button', { text: 'Generate data', disabled: llmOk ? null : '',
          title: llmOk ? '' : (usage.reason || 'LLM not configured'),
          onclick: () => rvPost(`api/proposals/${p.id}/generate`,
            { requester: name.value, count: +count.value, fmt: fmt.value }) })));
    if ((p.datasets || []).length)
      card.append(el('p', { class: 'small', text: 'datasets: ' + p.datasets.map(d =>
        `#${d.id} ${d.status}${d.error ? ' (' + d.error + ')' : ''}`).join(' · ') }));
  }
  if (p.status === 'rejected' && p.reject_reason)
    card.append(el('p', { class: 'small', text: 'reason: ' + p.reject_reason }));
  return card;
}

function rvDataset(d) {
  const pv = d.provenance || {};
  const g = pv.gate || {};
  const det = el('details', { class: 'rv', 'data-dataset': d.id },
    el('summary', { style: 'cursor:pointer' },
      el('b', { text: `dataset #${d.id}` }), ` · ${d.model || '—'} · ${d.task || '—'} · `
      + `${d.category || '—'} · ${d.fmt} · `,
      el('span', { class: stClass(d.status === 'ready' ? 'done' : d.status === 'pending'
        ? 'running' : 'failed'), text: d.status }),
      pv.items ? ` · ${pv.items.kept} kept of ${pv.items.generated} generated` : '',
      pv.provisional ? ' · provisional' : '',
      d.download ? [' · ', el('a', { href: d.download.replace(/^\//, ''), text: 'items.jsonl' })] : ''));
  if (d.error) det.append(el('p', { class: 'warn', text: d.error }));
  if (pv.provisional) det.append(el('p', { class: 'warn', 'data-provisional': 'dataset' },
    el('b', { text: 'Provisional. ' }), upFirst(pv.provisional_reason) + ': '
    + Object.entries(pv.local_models || {}).map(([role, s]) => `${role} ${s.served_model} at `
      + `${s.base_url}` + (s.weights ? ` (weights ${s.weights})` : '')).join('; ') + '.'));
  if (g.items_in != null)
    det.append(el('p', { class: 'small', text: `Contamination gate: ${g.dropped_benchmark} of `
      + `${g.items_in} items shared a ${g.ngram}-gram with something we evaluate on `
      + `(${pct(g.share_dropped_benchmark)}, line at ${pct(g.max_share, 0)}`
      + (g.dropped_exam ? `; ${g.dropped_exam} of them with an EXAM question` : '') + '); '
      + `${g.dropped_duplicate} near-duplicates collapsed; checked against ${g.benchmark_docs} `
      + `benchmark documents and ${g.exam_questions ?? 0} exam questions, both halves of each.`
      + (g.offending_ngrams && g.offending_ngrams.length ? ' First offending n-gram: “'
        + g.offending_ngrams[0] + '”.' : '') }));
  const rows = [];
  const walk = (o, pre) => { for (const [k, v] of Object.entries(o || {}))
    if (v && typeof v === 'object' && !Array.isArray(v)) walk(v, pre + k + '.');
    else rows.push([pre + k, Array.isArray(v) ? v.join(', ')
      : pre === 'timestamps.' && v ? absT(v) : String(v)]); };
  walk(pv, '');
  if (rows.length) det.append(el('div', { class: 'dxh', text: 'Provenance, in full' }),
    el('dl', { class: 'provlist small' }, rows.flatMap(([k, v]) =>
      [el('dt', { text: k }), el('dd', { class: 'mono', text: v })])));
  return det;
}

function vReview() {
  if (!state.rv.loaded && netReady()) { loadReview(); }
  rememberedName();
  const llm = state.rv.llm || {};
  const llmOk = !!llm.configured && (llm.usage_today || 0) < (llm.daily_cap || 0);
  const llmCard = el('div', { class: 'card' },
    el('h2', { text: 'Review' }),
    el('p', { class: 'sub', text: 'The human in the loop. An LLM reads the judge\'s written '
      + 'assessments of one model\'s DIAGNOSIS-half answers on one weak exam topic and '
      + 'proposes the skill that is missing; you approve, edit or reject that sentence; only '
      + 'the approved text reaches a generator, which never sees an exam question. Every '
      + 'decision is recorded under the name you type — the tailnet is the auth boundary, so '
      + 'that name is the record.' }),
    el('div', { class: 'kvs' },
      el('span', {}, el('b', { text: 'LLM ' }), llm.configured
        ? `${llm.provider}/${llm.model || '—'}` : 'not configured'),
      llm.configured ? el('span', {}, el('b', { text: 'today ' }),
        `${llm.usage_today} of ${llm.daily_cap} batch items` + ((llm.usage_today || 0) >= (llm.daily_cap || 0)
          ? ' — cap reached, generation paused until tomorrow' : '')) : '',
      llm.datasets_quota_bytes ? el('span', {}, el('b', { text: 'dataset storage ' }),
        `${(llm.datasets_bytes / 1e6).toFixed(1)} MB of ${(llm.datasets_quota_bytes / 1e9).toFixed(0)} GB`) : ''),
    !llm.configured && llm.reason ? el('p', { class: 'warn', text: llm.reason }) : '',
    // the name every decision on this page — and every propose click — is
    // recorded under. Remembered in this browser; there is no login to read it from
    el('div', { class: 'frm', style: 'margin-top:8px' }, rvNameInput(),
      el('span', { class: 'small', text: 'recorded on proposals you request and specs you '
        + 'approve or reject' })),
    state.rv.msg ? el('p', { class: 'small', text: state.rv.msg }) : '');
  const props = state.rv.proposals || [];
  const waiting = props.filter(p => p.status === 'pending' || p.status === 'proposed');
  const approved = props.filter(p => p.status === 'approved');
  const closed = props.filter(p => p.status === 'rejected' || p.status === 'failed');
  const sec = (title, sub, list, empty) => el('div', { class: 'card' },
    el('h2', { text: title }), el('p', { class: 'sub', text: sub }),
    list.length ? list.map(p => rvProposal(p, llmOk)) : el('p', { class: 'small', text: empty }));
  return [llmCard, rvTopicPicker(), rvTopicDetail(llmOk),
    sec('Awaiting review', 'Read the judged numbers first: if the suite is preliminary or the '
      + 'model wrote nothing on this topic, the button that made this proposal should have '
      + 'been disabled — reject it.', waiting,
      'Nothing waiting. A proposal starts from a topic row in a model\'s Judged section.'),
    sec('Approved — ready to generate', 'The spec below is exactly what the generator '
      + 'receives.', approved, 'No approved specs.'),
    el('div', { class: 'card' }, el('h2', { text: 'Datasets' }),
      el('p', { class: 'sub', text: 'Generated, gated, and recorded. A training run that '
        + 'consumes one registers it (datasets=[id] on the run), and every checkpoint of '
        + 'that run then carries the taint badge and loses the task from its average.' }),
      (state.rv.datasets || []).length ? (state.rv.datasets || []).map(rvDataset)
        : el('p', { class: 'small', text: 'No datasets yet.' })),
    closed.length ? el('details', { class: 'card' },
      el('summary', { style: 'cursor:pointer', text: `${closed.length} rejected or failed` }),
      closed.map(p => rvProposal(p, llmOk))) : ''];
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
      mkSel('answers topic', topics.map(t => [t, t]), topic,
        v => { state.ans.topic = v; state.ans.rows = null; render(); }),
      el('a', { class: 'small', href: '#topic=' + slugOfTopic(topic),
        text: 'this topic\u2019s page in the loop',
        onclick: e => { e.preventDefault(); navigate({ topic: slugOfTopic(topic), model: null }); } })));
  if (netReady() && !a.loading && (!a.rows || a.model !== m.id || a.topic !== topic))
    loadAnswers(m.id, topic);
  const j = a.rows;
  if (!j || a.model !== m.id || a.topic !== topic) {
    wrap.append(el('p', { class: 'small', text: 'Loading…' }));
    return wrap;
  }
  const rep = j.report_half || {};
  wrap.append(el('p', { class: 'note', 'data-report-half': '1' },
    el('b', { text: 'The report half. ' }),
    `${rep.n ?? 0} questions, mean ${num(rep.mean, 2)} / 4`
    + (Object.entries(rep.flags || {}).filter(([, n]) => n).length
       ? ' · ' + Object.entries(rep.flags).filter(([, n]) => n)
           .map(([fid, n]) => `${n} ${fid.replace(/_/g, ' ')}`).join(', ') : '')
    + '. That is the published score, and this line is all of it you will see here.'));
  const rows = ansVisible(j);
  wrap.append(ansFilters(j));
  wrap.append(el('p', { class: 'small', 'data-answer-count': String(rows.length),
    text: `${rows.length} of ${j.n_diagnose} diagnosis-half answers` }));
  wrap.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-answers-table': '1' },
    el('thead', {}, el('tr', {}, el('th', { text: 'acuity' }),
      el('th', { text: 'question and answer' }), el('th', { class: 'num', text: 'score' }),
      el('th', { text: `criteria (${(j.criteria || []).length})` }),
      el('th', { text: 'what the judge wrote' }))),
    el('tbody', {}, rows.map(it => ansRow(it, j))))));
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
    const j = await api('api/loop');
    // re-render only when the board actually moved: rebuilding the view every
    // five seconds detaches whatever the person is reaching for
    const sig = JSON.stringify([j.topics, j.judged_blocked, j.tasks_built]);
    const changed = sig !== _loopSig || !state.loop.loaded || state.loop.failed;
    _loopSig = sig;
    Object.assign(state.loop, { rows: j.topics, blocked: j.judged_blocked,
                                built: j.tasks_built || [], floor: j.floor, loaded: true,
                                failed: '' });
    if (changed && (state.tab === 'loop' || state.topic) && !state.model) render();
  } catch (e) {
    // netFail already put the banner up and set the backoff; the board itself
    // must also stop saying "Loading…" forever, which is what it did
    state.loop.failed = String((e && e.message) || e);
    if ((state.tab === 'loop' || state.topic) && !state.model) render();
  } finally { _loopInFlight = false; }
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

// where a step goes. The button is the same object on the board and on the
// topic page; only 'read' and 'sit' behave differently depending on which of
// the two you are already looking at.
function loopGo(r, step) {
  if (step === 'import') return navigate({ tab: 'exam', topic: null, model: null });
  if (step === 'review' || step === 'generate')
    { state.rv.topic = r.topic; state.rv.loaded = false; return navigate({ tab: 'review', topic: null, model: null }); }
  if (step === 'read') markRead(r.topic);
  return navigate({ topic: r.slug, model: null });
}

let _readRestored = false;
function loopReadRestore() {
  if (_readRestored) return;
  _readRestored = true;
  try { state.loopRead = JSON.parse(localStorage.getItem('bench-loop-read') || '{}') || {}; }
  catch (e) { /* private mode, or someone else's JSON */ }
}

function markRead(topic) {
  state.loopRead[topic] = true;
  try { localStorage.setItem('bench-loop-read', JSON.stringify(state.loopRead)); } catch (e) { /* private mode */ }
}

// 'Read the results' becomes 'Propose' once this browser has opened the
// answers for the topic: the step after reading is only honest after reading.
function loopStep(r) {
  if (r.next.step === 'read' && state.loopRead[r.topic] && r.propose)
    return { step: 'propose', label: 'Propose', ok: r.propose.ok,
             why: r.propose.why || '', short: r.propose.short || '' };
  return { ...r.next, short: r.next.why };
}

function loopBtn(r) {
  const st = loopStep(r);
  const slot = 'loop:' + r.slug;
  const b = st.step === 'propose'
    ? actButton(slot, st.label, () => loopPropose(r),
        { 'data-step': st.step, disabled: st.ok ? null : '', title: st.ok ? '' : st.why })
    : el('button', { 'data-step': st.step, disabled: st.ok ? null : '',
        title: st.ok ? '' : st.why, text: st.label,
        onclick: () => loopGo(r, st.step) });
  return el('div', {}, b,
    st.ok ? '' : el('div', { class: 'propwhy', 'data-why': st.step,
      title: st.why, text: st.short || st.why || '' }),
    st.step === 'propose' ? actNote(slot) : '');
}

async function loopPropose(r) {
  const last = r.last_judged || {};
  if (!last.model) throw new Error('no judged run on this topic to propose from');
  if (!state.rvName.trim()) throw new Error('your name is recorded on a proposal — type it first');
  const j = await post('api/proposals',
    { model: last.model, topic: r.topic, requested_by: state.rvName });
  state.loop.loaded = false; state.rv.loaded = false;
  loadLoop();
  return `Proposal #${j.id} submitted for ${last.model} on ${r.topic}. `
    + 'It lands on the Review tab when the batch completes.';
}

function judgedBadges(last) {
  if (!last) return [];
  const out = [];
  if (last.provisional) out.push(el('span', { class: 'badge taint', title: last.provisional_reason,
    text: 'provisional' }));
  if (last.draft_rubric) out.push(el('span', { class: 'badge taint', text: 'draft rubric' }));
  if (last.single_provider_loop) out.push(el('span', { class: 'badge taint',
    title: 'the same provider wrote, sat or graded more than one step of this loop',
    text: 'single provider' }));
  if (last.tainted) out.push(el('span', { class: 'badge taint',
    title: 'this model trained on data derived from this topic', text: 'trained on it' }));
  return out;
}

function vLoop() {
  if (!state.loop.loaded && netReady()) loadLoop();
  loopReadRestore();
  const rows = state.loop.rows || [];
  const head = el('div', { class: 'card' },
    el('h2', { text: 'The loop, by topic' }),
    el('p', { class: 'sub', text: 'Write the exam, sit it, read what the judge made of the '
      + 'answers, propose the skill that is missing, approve it, generate data, hand it to '
      + 'training. One row per topic, and the one thing to do next. Every refusal below is '
      + 'the API\'s own, in its words — this table asks, it does not decide.' }),
    state.loop.blocked ? el('p', { class: 'warn', 'data-loop-blocked': '1' },
      el('b', { text: 'No topic can be sat right now. ' }), state.loop.blocked) : '',
    state.loop.msg ? el('p', { class: 'small', 'data-loop-msg': '1', text: state.loop.msg }) : '',
    loopFailure(),
    el('div', { class: 'frm' }, rvNameInput(),
      el('span', { class: 'small', text: 'recorded on anything you start from here' })),
    el('p', { class: 'small' }, 'New here? ',
      el('a', { href: 'guide#the-loop', target: '_blank', rel: 'noopener',
                text: 'what the loop is and whose job each step is' }), '.'));
  if (!state.loop.loaded)
    return [head, el('div', { class: 'card' }, el('p', { class: 'small',
      text: state.loop.failed ? 'No board to show yet — see above.' : 'Loading…' }))];
  const table = el('div', { class: 'card' },
    el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-loop-table': '1' },
      el('thead', {}, el('tr', {},
        el('th', { text: 'topic' }), el('th', { text: 'bank (report / diagnose)' }),
        el('th', { text: 'rubric' }), el('th', { text: 'last judged' }),
        el('th', { text: 'open proposal' }), el('th', { text: 'datasets' }),
        el('th', { text: 'next step' }))),
      el('tbody', {}, rows.map(r => {
        const last = r.last_judged;
        if (r.error) return el('tr', { 'data-loop-row': r.slug, 'data-row-error': '1' },
          el('td', {}, r.topic),
          el('td', { class: 'warn', colspan: '6' }, r.error));
        return el('tr', { 'data-loop-row': r.slug },
          el('td', {}, el('a', { href: '#topic=' + r.slug, text: r.topic,
            onclick: e => { e.preventDefault(); navigate({ topic: r.slug, model: null }); } })),
          el('td', { class: 'small' }, r.bank.accepted
            ? `${r.bank.accepted} — ${r.bank.report} report / ${r.bank.diagnose} diagnose`
            : 'empty',
            r.bank.accepted && r.bank.under_floor
              ? el('div', { class: 'se', 'data-under-floor': '1',
                  text: `under ${r.bank.floor} report-half questions` }) : '',
            r.bank.awaiting ? el('div', { class: 'se', text: `${r.bank.awaiting} awaiting curation` }) : ''),
          el('td', { class: 'small' },
            el('a', { href: '#tab=exam', title: 'the rubric and criteria panel on the Exam tab',
              text: `${r.rubric.name}.md`,
              onclick: e => { e.preventDefault(); navigate({ tab: 'exam', topic: null }); } }),
            r.rubric.fallback ? el('span', { class: 'se', 'data-fallback': '1',
              title: 'this topic has no rubric of its own', text: ' (fallback)' }) : '',
            r.rubric.status === 'draft' ? el('span', { class: 'badge taint', text: 'DRAFT' }) : '',
            el('div', { class: 'se', text: r.rubric.scoring === 'criteria'
              ? `${r.rubric.criteria_count} criteria` : 'single score' })),
          el('td', { class: 'small' }, last
            ? el('span', {}, el('a', { href: '#model=' + encodeURIComponent(last.model),
                text: last.model, onclick: e => { e.preventDefault();
                  navigate({ model: last.model, topic: null }); } }),
                ` · ${num(last.score_report, 2)} / 4 `,
                el('span', { class: 'se', text: `(${last.n_report ?? 0} report-half)` }),
                ...judgedBadges(last))
            : el('span', { class: 'se', text: 'not judged yet' })),
          el('td', { class: 'small' }, r.proposal
            ? el('a', { href: '#tab=review', text: `#${r.proposal.id} ${r.proposal.status}`,
                onclick: e => { e.preventDefault(); state.rv.topic = r.topic;
                  navigate({ tab: 'review', topic: null }); } })
            : el('span', { class: 'se', text: '—' })),
          el('td', { class: 'small' }, r.datasets.length
            ? r.datasets.map(d => el('div', { class: 'se', text: `#${d.id} ${d.status}` }))
            : el('span', { class: 'se', text: '—' })),
          el('td', {}, loopBtn(r)));
      })))));
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
  loopReadRestore();
  const r = loopRowOf(state.topic);
  const back = el('p', { class: 'small' },
    el('a', { href: '#tab=loop', text: '← every topic',
      onclick: e => { e.preventDefault(); navigate({ topic: null, tab: 'loop' }); } }));
  if (!r) return [el('div', { class: 'card' }, back,
    el('h2', { text: state.topic }), el('p', { class: 'small',
      text: state.loop.loaded ? 'No such topic.' : 'Loading…' }))];
  const last = r.last_judged;
  const st = loopStep(r);
  const headCard = el('div', { class: 'card', 'data-topic-page': r.slug }, back,
    el('h2', { text: r.topic }),
    el('div', { class: 'kvs' },
      el('span', {}, el('b', { text: 'bank ' }), r.bank.accepted
        ? `${r.bank.accepted} (${r.bank.report} report / ${r.bank.diagnose} diagnose)` : 'empty'),
      el('span', {}, el('b', { text: 'rubric ' }),
        `${r.rubric.name}.md ${rubricVersion(r.rubric.version)}`
        + (r.rubric.status === 'draft' ? ' · DRAFT' : '')
        + (r.rubric.scoring === 'criteria' ? ` · ${r.rubric.criteria_count} criteria` : '')),
      el('span', {}, el('b', { text: 'next ' }), st.label)),
    r.bank.under_floor && r.bank.accepted ? el('p', { class: 'warn', text:
      `${r.bank.report} report-half questions — under the ${r.bank.floor} this topic needs `
      + 'before anything may be proposed from it. Import or write more on the Exam tab.' }) : '',
    el('div', { class: 'frm' }, loopBtn(r), rvNameInput()),
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

function loopSitPanel(r) {
  const s = state.loopSit;
  if (s.tasks == null) s.tasks = [r.task];
  const built = state.loop.built || [];
  const pick = el('div', { class: 'frm', style: 'flex-wrap:wrap' }, built.map(task => {
    const id = 'sit-' + task;
    const box = el('input', { type: 'checkbox', id, 'data-sit-task': task,
      checked: s.tasks.includes(task) ? '' : null,
      onchange: e => { s.tasks = e.target.checked ? [...s.tasks, task]
                                                  : s.tasks.filter(t => t !== task); } });
    return el('label', { class: 'small', for: id, style: 'margin-right:10px' }, box,
      ' ' + frName(task));
  }));
  const go = async () => {
    const body = { hf_id: s.model.trim(), kind: s.kind, suite: 'judged',
                   submitter: state.rvName, tasks: s.tasks };
    if (!body.hf_id) { s.msg = 'a model id first — org/name, or local/<name>'; return render(); }
    if (!body.tasks.length) { s.msg = 'pick at least one topic'; return render(); }
    s.busy = true; render();
    const res = await fetch('api/submissions', { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
      body: JSON.stringify(body) }).catch(() => null);
    const j = res ? await res.json().catch(() => ({})) : {};
    s.busy = false;
    s.msg = res && res.ok
      ? `#${j.id} queued — ${(j.tasks || []).map(frName).join(', ')}. The queue shows it, and `
        + 'the judge batch after it.'
      : 'refused: ' + (j.detail || (res ? res.status : 'server unreachable'));
    render();
  };
  return el('div', { class: 'card', 'data-panel': 'sit' },
    el('h2', { text: 'Sit the exam' }),
    el('p', { class: 'sub', text: 'One model, this topic — the same judged suite, narrowed to '
      + 'the tasks you tick. The answers are graded by the pinned judge when the run finishes; '
      + 'nothing here touches the report half except the judge.' }),
    state.loop.blocked ? el('p', { class: 'warn', 'data-sit-blocked': '1',
      text: state.loop.blocked }) : '',
    el('div', { class: 'frm' },
      el('input', { type: 'text', 'aria-label': 'model id', value: s.model,
        placeholder: 'org/model, or local/<name>', style: 'flex:2;min-width:240px',
        'data-keep': 'sit-model', oninput: e => { s.model = e.target.value; } }),
      el('select', { 'aria-label': 'kind', onchange: e => { s.kind = e.target.value; } },
        ['auto', 'base', 'instruct'].map(v => el('option', { value: v,
          selected: s.kind === v ? '' : null, text: v === 'auto' ? 'kind: auto-detect' : 'kind: ' + v }))),
      rvNameInput(),
      el('button', { 'data-sit': '1', disabled: (state.loop.blocked || s.busy) ? '' : null,
        text: s.busy ? 'queueing…' : 'Queue this run', onclick: go })),
    el('p', { class: 'small', text: 'topics in this run:' }), pick,
    s.msg ? el('p', { class: 'small', 'data-sit-msg': '1', text: s.msg }) : '',
    el('p', { class: 'small' }, 'The queue is on ',
      el('a', { href: '#tab=queue', text: 'Submit & Queue',
        onclick: e => { e.preventDefault(); navigate({ tab: 'queue', topic: null }); } }),
      ' — a judged row says which topics it sat and how far the judge batch is.'));
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

// the per-criterion strip: bars, because fifteen numbers in a row is a wall.
// Each bar carries its criterion and value as a title, for the mouse and for
// anything reading the DOM.
function ansStrip(it, criteria) {
  return el('div', { class: 'frm', style: 'gap:2px;flex-wrap:nowrap;align-items:flex-end' },
    criteria.map(c => {
      const v = (it.criteria || {})[c.id];
      return el('span', { class: 'dxbar', 'data-criterion': c.id,
        title: `${c.label}: ${v == null ? 'not applicable' : num(v, 2)}`,
        style: 'width:8px;height:22px;display:inline-flex;align-items:flex-end' },
        v == null ? '' : el('span', { style: `height:${Math.max(8, 100 * v).toFixed(0)}%;`
          + 'width:100%;display:block;background:var(--s1);border-radius:1px' }));
    }));
}

function ansRow(it, j) {
  const meta = it.meta || {};
  const flags = (j.flags || []).filter(f => (it.flags || {})[f.id]);
  const open = state.ans.open[it.qid];
  return el('tr', { 'data-answer': it.qid, 'data-half': 'diagnose' },
    el('td', { class: 'small', style: 'white-space:nowrap' }, meta.acuity || '—',
      meta.difficulty != null ? el('div', { class: 'se', text: 'difficulty ' + meta.difficulty }) : ''),
    el('td', { style: 'min-width:340px' },
      el('div', { class: 'small', text: it.prompt || '(question not on disk)' }),
      el('div', { class: open ? '' : 'clamp2', style: 'margin-top:4px',
                  'data-answer-text': '1' },
        el('span', { class: 'se', text: it.answer || '(the model wrote nothing)' })),
      el('a', { href: '#', class: 'small', text: open ? 'less' : 'more',
        onclick: e => { e.preventDefault();
          state.ans.open[it.qid] = !open; render(); } })),
    el('td', { class: 'num' }, it.graded ? `${it.score} / 4`
      : el('span', { class: 'se', text: 'unreadable' }),
      flags.length ? el('div', {}, flags.map(f => el('span', { class: 'badge taint',
        title: f.effect_words, 'data-flag': f.id, text: f.label }))) : ''),
    el('td', {}, ansStrip(it, j.criteria || [])),
    el('td', { class: 'small se', text: it.justification || '—' }));
}

function loopAnswersPanel(r) {
  const a = state.ans;
  const models = ansJudgedModels(r.task);
  const card = el('div', { class: 'card', 'data-panel': 'answers' },
    el('h2', { text: 'The answers' }),
    el('p', { class: 'sub', text: 'What the model actually wrote on this topic\'s DIAGNOSIS '
      + 'half, and what the judge made of each answer. The report half is the published '
      + 'score: it appears below as one line and never as a row — not its questions, not '
      + 'its answers, not its qids.' }));
  if (!models.length) {
    card.append(note('No model has been judged on this topic yet. Sit the exam above.'));
    return card;
  }
  const want = a.model && models.includes(a.model) ? a.model
             : (r.last_judged || {}).model && models.includes((r.last_judged || {}).model)
               ? r.last_judged.model : models[0];
  if (netReady() && (!a.rows || a.model !== want || a.topic !== r.topic) && !a.loading) {
    loadAnswers(want, r.topic);
    markRead(r.topic);
  }
  card.append(el('div', { class: 'frm' },
    mkSel('model', models.map(m => [m, m]), want,
      v => { state.ans.model = v; state.ans.rows = null; render(); }),
    el('a', { class: 'small', href: '#model=' + encodeURIComponent(want),
      text: 'this model\'s page',
      onclick: e => { e.preventDefault(); navigate({ model: want, topic: null }); } })));
  const j = a.rows;
  if (!j || a.topic !== r.topic) {
    card.append(el('p', { class: 'small', text: 'Loading…' }));
    return card;
  }
  const rep = j.report_half || {};
  card.append(el('p', { class: 'note', 'data-report-half': '1' },
    el('b', { text: 'The report half. ' }),
    `${rep.n ?? 0} questions, mean ${num(rep.mean, 2)} / 4`
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
    text: `${rows.length} of ${j.n_diagnose} diagnosis-half answers` }));
  card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-answers-table': '1' },
    el('thead', {}, el('tr', {}, el('th', { text: 'acuity' }),
      el('th', { text: 'question and answer' }), el('th', { class: 'num', text: 'score' }),
      el('th', { text: `criteria (${(j.criteria || []).length})` }),
      el('th', { text: 'what the judge wrote' }))),
    el('tbody', {}, rows.map(it => ansRow(it, j))))));
  if (!rows.length) card.append(note('No answer matches these filters.'));
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
      + (r.proposal.requested_by ? `, requested by ${r.proposal.requested_by}` : ''), '. ',
      el('a', { href: '#tab=review', text: 'Review it',
        onclick: e => { e.preventDefault(); state.rv.topic = r.topic;
          navigate({ tab: 'review', topic: null }); } })));
  } else {
    card.append(el('p', { class: 'small' }, 'No open proposal. ', loopBtn(r)));
  }
  const ready = (r.datasets || []).filter(d => d.status === 'ready');
  if (ready.length) {
    card.append(el('div', { class: 'dxh', text: 'Datasets from this topic' }));
    card.append(el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-datasets-table': '1' },
      el('thead', {}, el('tr', {}, el('th', { text: 'dataset' }), el('th', { class: 'num', text: 'documents' }),
        el('th', { text: 'hand to training' }))),
      el('tbody', {}, ready.map(d => el('tr', { 'data-dataset': String(d.id) },
        el('td', {}, el('a', { href: `api/datasets/${d.id}`, target: '_blank', rel: 'noopener',
          text: `#${d.id}` }),
          el('div', { class: 'se', text: `${d.kept ?? d.count} kept of ${d.count}` })),
        el('td', { class: 'num', text: String(d.kept ?? d.count) }),
        el('td', {}, el('code', { class: 'mono', text: `--gap-dataset ${d.id}` }),
          el('div', { class: 'se', text: 'the training run that consumes it registers it, and '
            + 'its checkpoints carry the taint badge on this topic' }))))))));
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
  const r = await fetch(path, { method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
    body: JSON.stringify(body || {}) }).catch(() => null);
  const j = r ? await r.json().catch(() => ({})) : {};
  state.ex.msg = r && r.ok ? (j.qid ? `accepted → ${j.half} half (${j.topic})`
    : j.status === 'rejected' ? 'rejected' : j.tasks ? 'tasks rebuilt: '
      + Object.entries(j.tasks).map(([t, v]) => `${t.replace(/^exam_/, '')} ${v.items}`).join(', ') : 'ok')
    : 'refused: ' + (j.detail || (r ? r.status : 'server unreachable'));
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
  const name = rvNameInput();
  return el('div', { class: 'rv', 'data-candidate': c.cid },
    el('div', { class: 'mhead' }, el('h3', { text: c.topic }),
      el('span', { class: 'small', text: `drafted by ${c.drafted_by || '—'} · batch ${(c.batch_id || '').slice(0, 14)}`
        + (c.provisional ? ` · provisional: ${c.provisional_reason}` : '') })),
    c.notes ? el('p', { class: 'small', text: 'drafter\'s note: ' + c.notes }) : '',
    el('div', { class: 'dxh', text: 'Question' }), prompt,
    el('div', { class: 'dxh', text: 'Reference (what a full-marks answer must contain)' }), ref,
    el('p', { class: 'small', text: 'Edit freely — the text you accept is what gets hashed, and '
      + 'the hash decides which half it lands in. You will not see it again if it lands in '
      + 'the report half.' }),
    el('div', { class: 'frm' }, name,
      el('button', { text: 'Accept into the bank', onclick: () => exPost(
        `api/exam/candidates/${c.cid}/accept`, { approver: name.value, prompt: prompt.value,
                                                 reference: ref.value }) }),
      reason,
      el('button', { text: 'Reject', onclick: () => exPost(
        `api/exam/candidates/${c.cid}/reject`, { approver: name.value, reason: reason.value }) })));
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
    el('h2', { text: 'Exam' }),
    el('p', { class: 'sub', text: 'The instrument. One question bank across the topics in '
      + 'scripts/categories.yaml; an LLM drafts candidates, a person accepts, edits or rejects '
      + 'each one here. Every accepted question is split by the hash of its text into a report '
      + 'half — the published per-topic score — and a diagnose half — the only half a proposal '
      + 'may read. That split is what lets the loop train on what the exam finds and still '
      + 'have an honest number.' }),
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
      'Nothing has been migrated, imported or drafted into this bank. Run ',
      el('code', { text: 'exam_build.py migrate' }), ' for the 40 seed items, ',
      el('code', { text: 'import' }), ' for a human-written bank, or ',
      el('code', { text: 'draft' }), ' for LLM candidates — the commands are below.'),
    el('p', { class: 'small' }, 'A person\'s bank, imported whole: ',
      el('code', { style: 'overflow-wrap:anywhere',
                   text: st.import_command || 'scripts/exam_build.py import' }),
      '. The approver is recorded on every question, exactly as accepting one here is.'),
    el('p', { class: 'small' }, 'LLM candidates to curate: ',
      el('code', { style: 'overflow-wrap:anywhere',
                   text: st.draft_command || 'scripts/exam_build.py draft' }),
      ' on the server; candidates appear below within a poll.'),
    el('div', { class: 'frm', style: 'margin-top:8px' }, rvNameInput(),
      actButton('exbuild', 'Rebuild the harness tasks from the bank', async () => {
        const j = await post('api/exam/build');
        const ts = Object.entries(j.tasks || {});
        state.ex.loaded = false; loadExam();
        return `Built ${ts.length} task${ts.length === 1 ? '' : 's'}: `
          + ts.map(([t, v]) => `${frName(t)} (${v.items})`).join(', ')
          + `. They are in ${j.tasks_dir}; a suite=judged run uses them now.`;
      }, { title: 'writes $EXAM_DIR/tasks from the accepted questions + the MMLU control '
                  + 'set; no GPU' })),
    actNote('exbuild'),
    state.ex.msg ? el('p', { class: 'small', text: state.ex.msg }) : '');
  const table = el('div', { class: 'card' }, el('h2', { text: 'By topic' }),
    el('p', { class: 'sub', text: `Target ${st.target_per_topic || 60} accepted questions per topic. `
      + 'Under 30 in the report half the published score is noise and the page greys it.' }),
    el('div', { class: 'lb-wrap' }, el('table', { class: 'jd', 'data-topics-table': '1' },
      el('thead', {}, el('tr', {}, el('th', { text: 'topic' }), el('th', { class: 'num', text: 'accepted' }),
        el('th', { class: 'num', text: 'report half' }), el('th', { class: 'num', text: 'diagnose half' }),
        el('th', { class: 'num', text: 'awaiting curation' }), el('th', { text: 'toward target' }))),
      el('tbody', {}, topics.map(t => { const s = sum[t];
        const bar = el('div', { class: 'dxbar', style: 'width:140px;height:8px' },
          el('span', { style: `width:${Math.min(100, 100 * s.accepted / (s.target || 60)).toFixed(1)}%;background:var(--s1)` }));
        return el('tr', { class: s.report < CAT_MIN_N ? 'dim' : null },
          el('td', {}, el('a', { href: '#', text: t, onclick: e => { e.preventDefault();
            state.ex.topic = state.ex.topic === t ? '' : t; state.ex.loaded = false;
            // drop the list with the filter: showing another topic's questions
            // under this topic's heading, until the fetch lands, is a lie
            state.ex.candidates = null; render(); } })),
          el('td', { class: 'num', text: String(s.accepted) }),
          el('td', { class: 'num', text: String(s.report) }),
          el('td', { class: 'num', text: String(s.diagnose) }),
          el('td', { class: 'num', text: String(s.pending) }),
          el('td', {}, bar)); })))));
  const cands = state.ex.candidates;
  const cur = el('div', { class: 'card' },
    el('h2', { text: 'Awaiting curation' + (state.ex.topic ? ` — ${state.ex.topic}` : '') }),
    el('p', { class: 'sub', text: 'Read each against the rubric: does it ask for understanding, is '
      + 'the reference the substance rather than a wording, is it answerable in five sentences, '
      + 'is it new? Accept, edit and accept, or reject with a reason. Click a topic above to filter.' }),
    cands == null ? el('p', { class: 'small', 'data-loading': 'candidates', text: 'Loading…' })
      : cands.length ? cands.slice(0, 40).map(exCandidate)
      : el('p', { class: 'small', text: 'Nothing waiting' + (state.ex.topic ? ' in this topic.' : '.') }),
    cands && cands.length > 40
      ? el('p', { class: 'small', text: `${cands.length - 40} more after these.` }) : '');
  return [head, exImport(), exRubrics(), table, cur];
}

// ---------------------------------------------------------------------------
// A person delivers a bank from here: a file, a topic, their name. Two steps,
// always — a preview that writes nothing, then a commit. The preview shows a
// diagnose-half question in full and a report-half one as its qid and
// metadata: he wrote them, and the page still does not echo them back.
// ---------------------------------------------------------------------------

function exImport() {
  const s = state.eximp || (state.eximp = { topic: '', source: '', author: '', file: '',
                                            name: '', fileName: '', preview: null, msg: '',
                                            busy: false });
  const topicSel = el('select', { 'aria-label': 'topic', onchange: e => {
    s.topic = e.target.value; s.preview = null; render(); } },
    el('option', { value: '', text: 'topic…' }),
    // the topic list is categories.yaml's, the same spine the bank uses
    ...Object.values((DATA.judged || {}).topics || {}).map(t =>
      el('option', { value: t, text: t, selected: s.topic === t ? '' : null })));
  const fileIn = el('input', { type: 'file', accept: '.json,application/json',
    'aria-label': 'questions file', onchange: async e => {
      const f = e.target.files[0]; if (!f) return;
      s.file = await f.text(); s.name = `${f.name} — ${(f.size / 1024).toFixed(0)} KB`;
      s.fileName = f.name;
      // the file's own name, never the topic's: "economics" in the source
      // column of the economics bank says nothing about where it came from
      s.source = s.source || f.name.replace(/\.json$/, '');
      // a new file is a new attempt: the last answer does not apply to it
      s.preview = null; actState('eximport').ok = actState('eximport').err = '';
      render(); } });
  const srcIn = el('input', { type: 'text', placeholder: 'source (the file, e.g. law_v2)',
    value: s.source, 'aria-label': 'source', 'data-keep': 'import-source',
    oninput: e => { s.source = e.target.value; } });
  // WHO WROTE THEM, which is not usually who is sitting here: the record has
  // to carry the author, or "these are Dr. Hossein's questions" lives only in
  // somebody's memory of the afternoon
  const authorIn = el('input', { type: 'text', placeholder: 'written by (the author)',
    value: s.author, 'aria-label': 'written by', 'data-keep': 'import-author',
    oninput: e => { s.author = e.target.value; } });
  const nameIn = rvNameInput();
  // both buttons answer in the same place, so a new attempt cannot leave the
  // last one's success sitting above its refusal — that read as a partial
  // import when a wrapped file was rejected after a good one
  const ready = () => {
    if (!s.file || !s.topic) throw new Error('a file and a topic, please');
    if (!s.author.trim()) throw new Error('who wrote these questions? that name goes on '
      + 'every one of them');
    if (!state.rvName.trim()) throw new Error('your own name is recorded with the import');
    return { topic: s.topic, approver: s.author, imported_by: state.rvName,
             source: s.source, filename: s.fileName, text: s.file };
  };
  const p = s.preview;
  return el('div', { class: 'card', 'data-panel': 'import' },
    el('h2', { text: 'Import a bank' }),
    el('p', { class: 'sub', text: 'A JSON array of questions written by a person — or an '
      + 'object holding one — read exactly as scripts/exam_build.py import reads it. The '
      + 'author\'s name goes on every question and the source says which file they came '
      + 'from; your own name is recorded as the person who imported them. Preview first: '
      + 'nothing is written until you commit.' }),
    el('div', { class: 'frm' }, fileIn, topicSel, authorIn, srcIn, nameIn,
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
      p ? actButton('eximport', `Import ${p.imported + (p.updated || 0)} questions`,
            async () => {
              const body = ready();
              s.preview = null;
              const j = await post('api/exam/import', body);
              s.file = ''; s.name = '';
              state.ex.loaded = false; loadExam();
              return `Imported ${j.imported}`
                + (j.updated ? `, revised ${j.updated} already in the bank` : '')
                + (j.skipped ? `, skipped ${j.skipped} unchanged` : '')
                + ` — report ${j.report} / diagnose ${j.diagnose}. They are in the bank; `
                + 'rebuild the harness tasks to sit them.';
            }, { 'data-commit': 'import',
                 disabled: (p.imported || p.updated) ? null : '' }) : ''),
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
      el('span', {}, el('b', { text: 'imported by ' }), state.rvName || '—')),
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
    el('p', { class: 'small', text: `${p.report} of these land in the report half and are not `
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
function bankCell(b) {
  if (!b || !b.accepted) return el('span', { class: 'warn', 'data-bank': '0',
    text: 'no questions yet' });
  return el('span', { 'data-bank': String(b.accepted) },
    `${b.accepted} — `,
    el('span', { class: 'se', text: `${b.report} report / ${b.diagnose} diagnose` }),
    b.pending ? el('span', { class: 'se', text: ` · ${b.pending} awaiting curation` }) : '');
}

function exRubrics() {
  const st = state.exrub || (state.exrub = { rows: null, open: '', kind: 'rubric',
                                             content: '', name: '', preview: null, msg: '' });
  if (st.rows == null && netReady()) loadRubrics();
  const rows = st.rows || [];
  return el('div', { class: 'card', 'data-panel': 'rubrics' },
    el('h2', { text: 'Rubrics and criteria' }),
    el('p', { class: 'sub', text: 'What the judge grades each topic with right now. A topic '
      + 'without a rubric of its own uses rubrics/exam.md. A criteria file turns the 0–4 into '
      + 'a fold of per-criterion scores. Changing either file changes its sha256, which is '
      + 'recorded in every judge.json: scores from before and after are not comparable.' }),
    st.store ? el('p', { class: 'small', text: `Uploads are written to ${st.store}. ${st.note || ''}` }) : '',
    st.rows == null ? el('p', { class: 'small', text: 'Loading…' })
      : el('div', { class: 'lb-wrap' }, el('table', { class: 'jd' },
        el('thead', {}, el('tr', {}, el('th', { text: 'topic' }), el('th', { text: 'bank' }),
          el('th', { text: 'rubric' }),
          el('th', { text: 'criteria' }), el('th', { text: 'files' }))),
        el('tbody', {}, rows.map(r => r.error
          ? el('tr', { 'data-rubric-row': r.topic, 'data-rubric-error': '1' },
              el('td', {}, r.topic),
              el('td', { class: 'warn', colspan: '4' }, r.error))
          : el('tr', { 'data-rubric-row': r.topic },
          el('td', {}, r.topic),
          // having a rubric is not having questions: three topics shipped
          // with both files and an empty bank, and looked ready
          el('td', { class: 'small' }, bankCell((st.banks || {})[r.topic])),
          el('td', {}, `${r.name}.md `,
            r.version === '?'
              ? el('span', { class: 'se', 'data-no-version': '1',
                  title: 'the author\'s own heading carries no version; the sha is the identity',
                  text: rubricVersion(r.version) + ' ' })
              : `v${r.version} `,
            r.fallback ? el('span', { class: 'se', 'data-fallback': '1',
              title: 'this topic has no rubric of its own; exam.md grades it',
              text: '(fallback) ' }) : '',
            el('span', { class: 'se', text: r.sha256.slice(0, 10) }),
            r.status === 'draft' ? el('span', { class: 'badge taint', text: 'DRAFT' }) : ''),
          el('td', {}, r.scoring === 'criteria'
            ? el('span', {}, `${r.criteria_count} criteria `,
                el('span', { class: 'se', text: r.criteria_sha256.slice(0, 10) }),
                r.criteria_status === 'draft'
                  ? el('span', { class: 'badge taint', text: 'DRAFT' }) : '')
            : el('span', { class: 'se', text: 'single score' })),
          el('td', {},
            el('a', { href: `api/exam/rubrics/${r.name}`, text: 'rubric', download: `${r.name}.md` }),
            ' · ',
            r.scoring === 'criteria'
              ? el('a', { href: `api/exam/rubrics/${r.name}?kind=criteria`, text: 'criteria',
                          download: `${r.name}.criteria.json` })
              : el('span', { class: 'se', text: '—' }),
            ' · ',
            el('a', { href: '#', text: 'replace', onclick: e => { e.preventDefault();
              st.open = st.open === r.name ? '' : r.name; st.name = r.name;
              st.preview = null; st.content = ''; render(); } }))))))),
    st.open ? exRubricUpload(st) : '',
    (st.changes || []).length ? el('p', { class: 'small', text: 'last change: '
      + st.changes.map(c => `${c.name}.${c.kind === 'criteria' ? 'criteria.json' : 'md'} by `
        + `${c.approver} ${rel(c.changed_at)} ago`).slice(0, 3).join(' · ') }) : '');
}

function exRubricUpload(st) {
  const slot = 'exrubric';
  const body = () => {
    if (!st.content) throw new Error('choose a file first');
    if (!state.rvName.trim()) throw new Error('your name is recorded with the change');
    return { name: st.name, kind: st.kind, content: st.content,
             approver: state.rvName, note: st.note || '' };
  };
  const p = st.preview;
  return el('div', { style: 'margin-top:10px', 'data-upload': st.name },
    el('div', { class: 'frm' },
      el('select', { 'aria-label': 'which file', onchange: e => {
        st.kind = e.target.value; st.preview = null; render(); } },
        el('option', { value: 'rubric', text: `${st.name}.md (prose)` }),
        el('option', { value: 'criteria', text: `${st.name}.criteria.json` })),
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
      el('pre', { class: 'mono', style: 'max-height:260px;overflow:auto;font-size:11.5px',
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
  ...(LIVE ? [['queue', 'Submit & Queue', vQueue],
              ['exam', 'Exam', vExam],
              ['review', 'Review', vReview],
              ['training', 'Training', vTraining]] : []),
  ['tasks', 'Tasks', vTasks],
  ['perplexity', 'Perplexity & Loss', vPpl],
  ['provenance', 'Provenance', vRuns],
];

// Old hashes keep working: a link someone pasted into a message last month
// should still land, and silently landing on Overview instead is the worst
// of the three possible behaviours.
const TAB_ALIASES = {
  runs: 'provenance', evals: 'provenance', submit: 'queue', 'submit-queue': 'queue',
  models_tab: 'models', ppl: 'perplexity', 'perplexity-loss': 'perplexity',
};
function render() {
  // full rebuild: drop the in-place refreshers so a poll can never touch the
  // DOM of a tab that just got torn down — the mounted tab re-registers its own
  state.trRedraw = state.queueRedraw = null;
  renderWarnings();                 // full on the board, folded elsewhere
  const ms = visible();
  renderTabs();
  const view = document.getElementById('view');
  view.classList.remove('dimmed');
  // a poll rebuilds the view every few seconds. Whatever the person is typing
  // in — and where their caret is — comes back afterwards, or the field is
  // unusable on a live page: this is the same bug as the tab bar's, one layer
  // down, and the cure is the same one (never lose what the DOM was holding).
  const live = document.activeElement;
  const keep = live && live.dataset && live.dataset.keep && view.contains(live)
    ? { key: live.dataset.keep, value: live.value,
        start: live.selectionStart, end: live.selectionEnd } : null;
  if (state.model) view.replaceChildren(...vModel());
  else if (state.topic) view.replaceChildren(...vTopic());
  else view.replaceChildren(...TABS.find(([id]) => id === state.tab)[2](ms));
  if (keep) {
    const again = view.querySelector(`[data-keep="${keep.key}"]`);
    if (again) {
      again.value = keep.value;
      again.focus();
      try { again.setSelectionRange(keep.start, keep.end); } catch (e) { /* not a text field */ }
    }
  }
}

// The tab bar is the one thing on the page that must survive a render. A
// poll, a finished fetch and a click all call render(); replacing the buttons
// each time hands whoever is mid-click a node that is no longer in the
// document, and the click goes nowhere. Build them once, then only move the
// selection.
function renderTabs() {
  const tabs = document.getElementById('tabs');
  if (tabs.children.length !== TABS.length) {
    tabs.replaceChildren(...TABS.map(([id, label]) =>
      el('button', { role: 'tab', onclick: () => navigate({ tab: id, model: null }),
                     text: label })));
  }
  const sel = state.model ? '' : state.tab;
  [...tabs.children].forEach((b, i) =>
    b.setAttribute('aria-selected', String(TABS[i][0] === sel)));
}

// The board-level checks belong to the board. They are findings, not
// notifications — never dismissed, never hidden — but on the Exam and Review
// tabs they are about something else entirely and push the work down the
// page, so there they collapse to one line that opens.
const WARN_TABS = ['overview', 'leaderboard'];

let _warnSig = null;
function renderWarnings() {
  const box = document.getElementById('warnings');
  if (!box || !DATA) return;
  const ws = DATA.warnings || [];
  const mode = (!state.model && WARN_TABS.includes(state.tab)) ? 'full' : 'fold';
  // rebuilding this on every poll would snap shut a fold someone just opened,
  // and hand Playwright (and a mouse) a node that vanishes mid-click
  const sig = mode + '' + ws.join('');
  if (sig === _warnSig) return;
  _warnSig = sig;
  if (!ws.length) { box.replaceChildren(); return; }
  const full = ws.map(w => el('div', { class: 'warn' }, el('b', { text: 'Check: ' }), w));
  if (mode === 'full') {
    box.replaceChildren(...full);
    return;
  }
  // what kind, not just how many: after a judged run most of them are about
  // the judge, and "5 checks" says nothing about whether to open it
  const judged = ws.filter(w => /judge|judged|rubric|criteria|canary|calibrat/i.test(w)).length;
  box.replaceChildren(el('details', { class: 'warnfold', 'data-warnings': 'collapsed' },
    el('summary', { 'data-warn-summary': String(ws.length),
      text: `${ws.length} check${ws.length > 1 ? 's' : ''} on the leaderboard`
        + (judged ? ` · ${judged} about the judged suite` : '') }),
    ...full));
}

// static shell bits (rendered whenever a payload arrives)
function renderStatic() {
  if (LIVE) document.getElementById('pageSub').textContent =
    'Live team benchmark: submit models, track training runs, compare results — '
    + 'updates as work finishes.';
  renderWarnings();
  // the model filters moved into the Models tab, where what they filter is on
  // screen beneath them; each one carries its own count there
  document.getElementById('metaChips').replaceChildren(
    el('span', { class: 'chip', 'data-stamp': '1',
      text: LIVE ? `live · refreshed ${DATA.generated}` : `generated ${DATA.generated}` }),
    LIVE ? el('a', { class: 'chip', href: 'guide', target: '_blank', rel: 'noopener',
                     style: 'text-decoration:none', text: '📖 guide for new users' }) : '',
    // a demo has its own page, of its own tree; this is the only thread
    // between them, and it appears only once a run has left one behind
    LIVE && DATA.demo ? el('a', { class: 'chip', href: DATA.demo.href, 'data-demo': 'link',
                                  style: 'text-decoration:none',
                                  title: 'a demo run\'s own page — provisional, not the board',
                                  text: `demo run from ${absT(DATA.demo.at).split(',')[0]}` }) : '',
    DATA.meta.hashes.length ? el('span', { class: 'chip' }, 'harness ',
      el('span', { class: 'mono', text: DATA.meta.hashes.join(', ') })) : '',
    DATA.meta.transformers ? el('span', { class: 'chip', text: `transformers ${DATA.meta.transformers}` }) : '',
    DATA.meta.anyLimit ? el('span', { class: 'chip', text: '⚠ smoke data (--limit)' }) : '');
}

function initData(d) {
  DATA = d;
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
const NET = { fails: 0, nextAt: 0, last: null, MIN: 1000, MAX: 30000 };
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
  NET.fails = 0; NET.nextAt = 0; NET.last = null;
  if (had) netRender();
}

function netFail(path, err) {
  NET.fails++;
  const wait = Math.min(NET.MAX, NET.MIN * Math.pow(2, NET.fails - 1));
  NET.nextAt = Date.now() + wait;
  NET.last = { path, err: String((err && err.message) || err), wait };
  netRender();
}

async function api(path, opts) {
  try {
    const r = await fetch(path, opts);
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

async function loadQueue() {
  try {
    const rows = await api('api/submissions?limit=100');
    const prev = new Map(state.queue.map(r => [r.id, r]));
    // new scores exist when a run's answers land (status -> done) AND, for a
    // judged run, again when its judge batch lands minutes later: the row is
    // already 'done' by then, so only the judge's own status says so
    const judgeDone = r => ((r || {}).judge || {}).status === 'done';
    const justFinished = rows.some(r => {
      const p = prev.get(r.id);
      return p && ((r.status === 'done' && p.status !== 'done')
                   || (judgeDone(r) && !judgeDone(p)));
    });
    const changed = rows.length !== state.queue.length || rows.some(r => {
      const p = prev.get(r.id);
      return !p || p.status !== r.status || p.progress !== r.progress
        || JSON.stringify(p.judge || null) !== JSON.stringify(r.judge || null);
    });
    state.queue = rows;
    if (justFinished) await refreshResults();       // new scores -> re-render everything
    else if (changed && state.tab === 'queue') (state.queueRedraw || render)();
  } catch (e) { /* netFail said so, and set how long to wait */ }
}

const THEMES = ['auto', 'light', 'dark', 'dim'];
function applyTheme(t) {
  if (t === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', t);
  document.getElementById('themeBtn').textContent = 'Theme \u25be';
  document.getElementById('themeBtn').title =
    `theme: ${t} — click to cycle auto / light / dark / dim, remembered in this browser`;
  try { localStorage.setItem('bench-theme', t); } catch (e) { /* private mode etc. */ }
}
let themeIdx = 0;
try {   // remembered per browser — the dashboard is a page people leave open
  const saved = localStorage.getItem('bench-theme');
  if (THEMES.includes(saved)) themeIdx = THEMES.indexOf(saved);
} catch (e) { /* storage unavailable: stay on auto */ }
// always applied, even on 'auto': the button's title names the current theme,
// and a button whose tooltip is only right after the first click is a lie
applyTheme(THEMES[themeIdx]);
document.getElementById('themeBtn').addEventListener('click', () => {
  themeIdx = (themeIdx + 1) % THEMES.length;
  applyTheme(THEMES[themeIdx]);
});

// boot: embedded data renders immediately; live mode fetches then polls
if (LIVE) {
  document.getElementById('view').replaceChildren(
    el('p', { class: 'small', style: 'margin:20px 4px', text: 'Loading results…' }));
  refreshResults().then(() => {
    if (DATA && !DATA.models.length) { state.tab = 'queue'; render(); }
  });
  loadQueue();
  setInterval(() => {
    if (!netReady()) return;                 // still inside the backoff window
    // a results fetch that failed is still owed: the first load (the board
    // would stay empty), or the one a finished run asked for (the board
    // would keep the numbers from before it for as long as the tab stayed open)
    if (RESULTS_DUE) refreshResults();
    loadQueue();
    if (state.tab === 'training') loadTraining();
    if (state.tab === 'review') loadReview();
    if (state.tab === 'exam') loadExam();
    // the Loop board and a topic page: without this nothing ever re-fetched
    // /api/loop, so a board whose first load failed stayed empty for as long
    // as the tab was open — which is exactly what happened on the live tree
    if (state.tab === 'loop' || state.topic) loadLoop();
  }, 5000);
} else {
  initData(DATA);
}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>__CSS__</style></head>
<body class="viz-root"><div id="tip" role="status"></div><div class="wrap">
__BANNER__
  <div class="topbar">
    <div>
      <h1>__TITLE__</h1>
      <p class="sub" id="pageSub">lm-evaluation-harness results, one self-contained
      file — data embedded, charts drawn locally, nothing fetched.</p>
      <div class="meta-chips" id="metaChips"></div>
    </div>
    <button id="themeBtn" title="cycle auto / light / dark / dim — remembered in this browser">Theme &#9662;</button>
  </div>
  <div id="netstatus"></div>
  <div id="warnings"></div>
  <div class="tabs" role="tablist" id="tabs"></div>
  <div id="view"></div>
  <footer>Every score carries its standard error; differences are z-tested before
  they are called wins; provenance is in the Provenance tab. Scores are only comparable to
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
                 banner: str = "", banner_link: tuple[str, str] = ("", "")) -> Path:
    if not runs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"<h1>No lm-eval results found.</h1><p>{html.escape(banner)}</p>",
                            encoding="utf-8")
        return out_path
    payload = build_payload(merge_runs(runs), title, source="", calibration=calibration,
                            taint=taint, parents=parents, judge_identity=judge_identity)
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
    out = build_report(runs, args.out, args.title, calibration=cal)
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
