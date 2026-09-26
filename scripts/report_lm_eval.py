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
                      "rubric_status", "rubrics_draft",
                      # 12i.1: the judge version — what decides whether today's
                      # views show these scores — and the provider it was pinned to
                      "version", "pin")},
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


def judged_current(trimmed: dict | None, current_id: str | None,
                   current_version: str | None) -> bool:
    """12i.1: were these judged scores marked by the judge version this server
    runs now? A version is the model, the provider it is pinned to and the
    judge's prompts; a file from before versions is the current one's while
    its judge id is. With no judge on the server (a frozen report) nothing
    is set aside"""
    j = (trimmed or {}).get("judge") or {}
    if not current_id:
        return True
    v = (j.get("version") or {}).get("key")
    if current_version and v:
        return v == current_version
    return j.get("id") == current_id


def judged_by(trimmed: dict | None) -> str:
    """the judge that marked a file, in words: "DeepSeek V4.1 Flash", "local/chat" """
    j = (trimmed or {}).get("judge") or {}
    return (j.get("version") or {}).get("label") or j.get("id") or j.get("model") or "an earlier judge"


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
        if cal.get("by"):
            # 12i.1: the judge test's — a weighted kappa over enough answers
            reasons.append(f"weighted kappa {cal.get('kappa')} on {cal.get('n')} answers against "
                           f"{cal['by']} — {cal.get('kappa_min')} on {cal.get('n_min')} is needed")
        else:
            reasons.append(f"Cohen's kappa {cal.get('kappa')} is below {KAPPA_MIN}")
    elif cal.get("judge_id") and jid and cal["judge_id"] != jid:
        reasons.append(f"the calibration on file is for {cal['judge_id']}, not {jid}")
    elif cal.get("judge_version") and ((trimmed.get("judge") or {}).get("version") or {}) \
            .get("key") not in (None, cal["judge_version"]):
        reasons.append("the calibration on file is for another version of this judge (its "
                       "provider or prompts changed)")
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
                            f"Benchmarks ▸ Knowledge exam",
                     "short": f"under the {PROPOSE_MIN_N}-question floor — write more on "
                              f"Benchmarks ▸ Knowledge exam"})
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


# 12a: Everyday tasks are a look, not a benchmark. Their answers are
# written by the same harness into the same tree, and a task of them here
# would be a column, a count toward "tasks", a date — so it is never read as a
# run. The page reads everyday.json instead (load_everyday). The bank's task
# is "everyday" (12a.2); a model that sat the pilot logged "everyday_pilot"
NOT_A_BENCHMARK = ("everyday",)


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
        if blob["results"] and all(str(t).startswith(NOT_A_BENCHMARK) for t in blob["results"]):
            continue
        runs.append(parse_run(blob, f))
    return runs


# what the page needs of each marked answer; everything else stays on disk
_EVERYDAY_ITEM = ("id", "group", "pass", "reason", "answer_text", "had_reasoning",
                  "reasoning_text", "reasoning_words", "no_answer", "failed")


def _evd_label(q: dict) -> str:
    """a question's short name in the answer panel's title: its skill, as a
    heading ("heavy typos" → "Heavy typos")"""
    s = str(q.get("label") or q.get("skill") or q["id"])
    return s[:1].upper() + s[1:]


def load_everyday(out_dir: Path | None) -> dict | None:
    """12a.3: the Everyday bank — 12a.5: 388 questions in eight groups, split by
    12g.2 into a hidden half that scores and a practice half that is shown —
    and every model's marks, from
    the everyday.json beside its results. Kept apart from the models' rows,
    so nothing that ranks or averages can reach it.

    12a.4: only answers to this wording (its version, everyday.version()) are
    in `models`. Answers to an earlier wording — everything marked before the
    version existed, and the pilot's five — are in `earlier`, as their count
    and date alone: never in a score, a table or a comparison."""
    try:
        import everyday as ev                    # scripts/, beside this file
        qs = ev.load_bank()
    except (ImportError, OSError, ValueError):
        return None
    group_of = {q["id"]: q["group"] for q in qs}
    now = ev.version()
    # 12g.2: the split — the hidden half scores and is never shown; the page
    # gets the practice half's questions and answers, and the hidden counts
    half_of = {q["id"]: ev.half(q) for q in qs}
    counts = ev.split_counts(qs)
    models, earlier = {}, {}
    for f in sorted(Path(out_dir).glob("*/everyday.json")) if out_dir and Path(out_dir).is_dir() \
            else []:
        try:
            e = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(e, dict) or not isinstance(e.get("items"), list) or not e.get("model"):
            continue
        stamp = e.get("version") or {}
        ver = stamp.get("hash") or ""
        if ver != now["hash"]:
            # as it was marked then: its questions may be gone or reworded —
            # or it was marked on all of them, before the split (12g.2)
            earlier[e["model"]] = {
                "passed": e.get("passed", sum(1 for it in e["items"] if it.get("pass") is True)),
                "total": e.get("total", len(e["items"])), "marked_at": e.get("marked_at"),
                "hash": ver, "label": ev.BEFORE_SPLIT if not stamp.get("split")
                else "an earlier wording"}
            continue
        # a question the bank no longer holds is not shown (and not counted)
        items = [{**{k: it[k] for k in _EVERYDAY_ITEM if k in it}, "group": group_of[it["id"]]}
                 for it in e["items"] if it.get("id") in group_of]
        hidden = [it for it in items if half_of[it["id"]] == ev.HIDDEN]
        practice = [it for it in items if half_of[it["id"]] == ev.PRACTICE]

        def by_group(xs):
            return {g: {"passed": sum(1 for it in xs if it["group"] == g and it.get("pass") is True),
                        "total": sum(1 for it in xs if it["group"] == g)}
                    for g in ev.groups() if any(it["group"] == g for it in xs)}
        models[e["model"]] = {
            # the published score: the hidden half's
            "passed": sum(1 for it in hidden if it.get("pass") is True), "total": len(hidden),
            "waiting": sum(1 for it in hidden if it.get("pass") is None),
            "marked_at": e.get("marked_at"), "settings": e.get("settings") or {},
            "provisional": bool((e.get("judge") or {}).get("provisional")),
            "version": ver,
            # 12a.4: answers whose thinking used the whole budget
            "ran_out": sum(1 for it in hidden if it.get("no_answer")),
            # 12a.5: "55 new questions · 333 re-marked", and the questions it
            # has not been asked yet
            "marking": ev.marking_line(e), "unasked": int(e.get("unasked") or 0),
            "groups": by_group(hidden),
            # the practice half: its counts, and its answers — the only ones shown
            "practice": by_group(practice), "items": practice,
        }
    shown = [q for q in qs if half_of[q["id"]] == ev.PRACTICE]
    names = ev.groups()
    return {"groups": [[k, v] for k, v in names.items()],
            # 12g.2: the practice half only; the hidden half is its counts
            "questions": [{"id": q["id"], "n": i, "group": q["group"],
                           "groupLabel": names[q["group"]],
                           "label": _evd_label(q), "skill": q.get("skill") or "",
                           "prompt": q["prompt"], "reference": q.get("reference") or "",
                           "checks": [ev.describe(c) for c in q["checks"]],
                           "judged": any(c["type"] == "judge" for c in q["checks"])}
                          for i, q in enumerate(shown, 1)],
            "hidden": {g: c["hidden"] for g, c in counts.items()},
            "practice": {g: c["practice"] for g, c in counts.items()},
            "version": now, "models": models, "earlier": earlier}


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
    # 12h.1: a thinking-on run of a model that can turn it off is a row of its
    # own, never averaged with the thinking-off one
    if re.search(r"enable_thinking=True", str(model_args)):
        model += " · thinking"

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

    # 12h.1: MMLU-Pro and MATH-500 as this board reads the answers
    # (scripts/generative.py) — the harness's readers find one shape each
    gen = _beside(source, "generative.json")
    for t, g in ((gen or {}).get("tasks") or {}).items():
        if t in ("mmlu_pro", "hendrycks_math500") and t in tasks and "acc" in g:
            tasks[t]["exact_match"] = {"value": float(g["acc"]),
                                       "stderr": float(g.get("stderr") or 0.0),
                                       "_filt_value": "board"}

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
        "generative": gen,
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
    for name in ("acc_norm", "acc", "exact_match", "prompt_level_strict_acc", "pass@1", "f1",
                 "em", "bits_per_byte", "byte_perplexity", "word_perplexity"):
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
          "winogrande", "piqa", "truthfulqa_mc2", "gsm8k",
          "ifeval", "mmlu_pro", "hendrycks_math500"]
_CHANCE = {"mmlu": 0.25, "mmlu_perm": 0.25, "hellaswag": 0.25, "arc_challenge": 0.25,
           "arc_easy": 0.25, "winogrande": 0.5, "piqa": 0.5, "gsm8k": 0.0,
           "mmlu_pro": 0.1, "hendrycks_math500": 0.0}
# 12h.1: the three that generate text, for instruct models only; never in Avg
GEN_TASKS = ("ifeval", "mmlu_pro", "hendrycks_math500")

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
    # 12h.1: asked through the chat template, scored on what the model writes;
    # instruct models only, and never in the overall average
    "ifeval": ("instruction & maths",
               "Instructions to follow to the letter — no commas, three bullet points, "
               "under 100 words — over 541 prompts, asked through the chat template. "
               "Prompt-level strict accuracy: an answer counts only when it keeps every "
               "instruction; instruction-level, beside it, counts each one. Instruct models "
               "only, and never in the overall average."),
    "mmlu_pro": ("instruction & maths",
                 "MMLU made harder: 12,032 questions with ten options each, answered with "
                 "the reasoning written out (5-shot chain of thought) and scored on the "
                 "letter the answer settles on. Hours per model where the other tasks take "
                 "minutes. Instruct models only, and never in the overall average."),
    "hendrycks_math500": ("instruction & maths",
                          "500 competition maths problems (MATH-500), answered in the "
                          "model's own words and scored on the final answer, compared as "
                          "maths: 1/2 and 0.5 are one answer. Instruct models only, and "
                          "never in the overall average."),
}

# 12h.1: what each model's makers publish, from their model cards (their own
# setups: shots, templates and thinking differ from ours). Not a target — a
# neighbourhood: a score of ours more than 15 points below it points at a
# scoring bug, and the cell says so for masein to look into
_PUBLISHED = {
    "Qwen/Qwen3.5-2B": {"ifeval": (61.2, "no thinking"), "mmlu_pro": (55.3, "no thinking")},
    "LiquidAI/LFM2.5-1.2B-Instruct": {"ifeval": (86.2, ""), "mmlu_pro": (44.4, "")},
    "ibm-granite/granite-4.0-h-1b": {"ifeval": (78.5, "average"),
                                     "mmlu_pro": (32.9, "5-shot CoT")},
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": {"ifeval": (82.8, "prompt, no thinking"),
                                              "hendrycks_math500": (95.4, "thinking")},
    "tencent/Youtu-LLM-2B": {"ifeval": (81.2, ""), "mmlu_pro": (61.6, ""),
                             "hendrycks_math500": (93.7, "")},
    "google/gemma-4-E2B-it": {"mmlu_pro": (60.0, "")},
}
FAR_BELOW = 15.0            # points under the published score that flag a cell


def _thinking_modes() -> dict[str, str]:
    """model id -> switch | always | never, for the catalogue's models
    (service/catalog.py), read from its file: this script also runs where the
    service package is not importable"""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "service" / "catalog.py"
    try:
        spec = importlib.util.spec_from_file_location("_catalog", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except (OSError, ImportError, AttributeError):
        return {}
    return {m["id"]: m["thinking"] for m in mod.MODELS}


def far_below(model: str, task: str, v: float | None) -> dict | None:
    """{published, note, gap} when our score is more than FAR_BELOW points
    under what the model's makers publish, else None"""
    pub = (_PUBLISHED.get(model.removesuffix(" · thinking")) or {}).get(task)
    if not pub or v is None:
        return None
    gap = pub[0] - 100 * v
    return {"published": pub[0], "note": pub[1], "gap": round(gap, 1)} if gap > FAR_BELOW \
        else None

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
PROPORTION = {"acc", "acc_norm", "exact_match", "pass@1", "f1", "em", "rubric_pass",
              "prompt_level_strict_acc"}

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
        m["generative"] = m.get("generative") or r.get("generative")
    return by_model


def build_payload(by_model: dict[str, dict], title: str, source: str,
                  taint: dict[str, list[str]] | None = None,
                  calibration: dict | None = None,
                  parents: dict[str, str] | None = None,
                  judge_identity: dict | None = None,
                  fingerprints: dict[str, str] | None = None,
                  everyday: dict | None = None) -> dict:
    """`everyday`: the pilot's questions and marks (load_everyday), carried
    beside the models and never inside them.

    `taint`: model id -> tasks whose diagnostics its training data was
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
    current_version = (judge_identity or {}).get("version") or None
    cal = calibration if isinstance(calibration, dict) and calibration.get("kappa") is not None \
        else None
    if cal is not None:
        cal = {**{k: cal.get(k) for k in ("kappa", "n", "calibrated", "kappa_min", "per_category",
                                          "n_min", "by", "method")},
               "judge_id": (cal.get("judge") or {}).get("id"),
               "judge_version": (cal.get("judge") or {}).get("version")}

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
        # 12h.1: the three generative tasks are instruct-only and never in any
        # average, the diagnostic one included
        have = [cells[t][mid]["v"] for t in acc_tasks
                if mid in cells.get(t, {}) and t not in CONTROL_TASKS and t not in tainted_acc
                and t not in GEN_TASKS]
        got_req = [t for t in required if mid in cells.get(t, {}) and t not in tainted_acc]
        missing = [t for t in required if mid not in cells.get(t, {})]
        official = bool(required) and not missing and not (set(tainted_acc) & set(required))
        params = r["num_params"] or params_from_name(mid)
        judge = _judged_now(r, fingerprints)
        # 12i.1: only the current judge version's scores are in today's views;
        # an earlier one's go to the model's History, "judged by <model>"
        judged_earlier = None
        if judge and not judged_current(judge, current_judge, current_version):
            judged_earlier = {"by": judged_by(judge), "id": (judge.get("judge") or {}).get("id"),
                              "avg": judged_avg(judge, tainted),
                              "topics": len(judge.get("tasks") or {}),
                              "at": r.get("judge_mtime")}
            judge = None
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
        gen = r.get("generative") or {}
        model_rows.append({
            # 12h.1: how its IFEval, MMLU-Pro and MATH-500 were asked and read —
            # thinking, backend, a subset, and per task the answers that ran out
            # of room, the unreadable ones and the scorer; and any score far
            # below what its makers publish
            "gen": ({"thinking": gen.get("thinking"), "backend": gen.get("backend"),
                     "fellBack": gen.get("fell_back"), "subset": gen.get("subset"),
                     "tasks": {t: {k: g.get(k) for k in ("ran_out", "unreadable", "inst_acc",
                                                         "n", "scorer")}
                               for t, g in (gen.get("tasks") or {}).items()},
                     "far": {t: f for t in GEN_TASKS
                             if (f := far_below(mid, t, (cells.get(t) or {}).get(mid, {})
                                                .get("v")))}}
                    if gen or any(mid in cells.get(t, {}) for t in GEN_TASKS) else None),
            "thinkingRow": mid.endswith(" · thinking"),
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
            "judgedEarlier": judged_earlier,
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
    # 12b.3: a check is a PROBLEM — something happened that a person should act
    # on (a duplicate row, answers that never finished, a judge that drifted)
    # — or a KNOWN LIMIT, a standing condition of the setup that stays true for
    # weeks (a judge nobody has calibrated, one provider, preliminary models).
    # The status dot counts problems only: a dot that is always amber stops
    # meaning anything, and then nobody sees the duplicate row
    LIMITS = {"chat_templates", "required_narrow", "tainted", "preliminary", "harness_builds",
              "judge_uncalibrated", "judge_kappa", "judge_local", "judge_single_provider"}

    def warn(key: str, severity: str, show: dict, short: str, text: str) -> None:
        warnings.append(text)
        checks.append({"key": key, "severity": severity, "show": show, "short": short,
                       "text": text, "judged": key.startswith("judge_"),
                       "limit": key in LIMITS})
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
    # 12b.3: "on some models, not others" and "different templates" were two
    # checks about one thing; they are one, saying whichever is true
    chat_says: list[str] = []
    if len({r["chat_template"] for r in by_model.values()}) > 1:
        applied = [display[m] for m, r in by_model.items() if r["chat_template"]]
        chat_says.append("Chat template applied to some models but not others (applied to: "
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
        chat_says.append(f"The models evaluated WITH a chat template used {len(shas)} different "
            f"templates ({', '.join(sorted(shas))}). Prompt format differs, so "
            f"those scores answer slightly different questions.")
    if chat_says:
        warn('chat_templates', 'info', {'tab': 'models', 'kind': 'instruct'},
             'Chat templates differ between models', " ".join(chat_says))
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
        warn('judge_uncalibrated', 'warning', {'tab': 'exam'}, 'The judge is not calibrated against a person',
            "Judged free-response scores are on file but the judge has not been calibrated "
            "against a person (scripts/judge_calibrate.py). They are shown as preliminary and "
            "enter no average and no rank.")
    elif cal and not cal.get("calibrated"):
        warn('judge_kappa', 'warning', {'tab': 'exam'}, "The judge's agreement with a person is below the line",
            f"The judge's agreement with a human grader is Cohen's kappa {cal['kappa']} over "
            f"{cal.get('n')} answers, below the {KAPPA_MIN} line. Judged scores are preliminary: "
            f"shown, never ranked, never averaged.")
    if judge_meta and judge_meta.get("stub"):
        warn('judge_stub', 'warning', {'tab': 'exam'}, 'Judged scores come from the stub grader',
            "Judged scores on this board come from the STUB grader (a word-overlap stand-in "
            "used for plumbing tests). They are not judgements of anything.")
    drifted = [m["name"] for m in model_rows if m.get("judge")
               and (m["judge"].get("canary") or {}).get("drifted")]
    if drifted:
        warn('judge_canary', 'warning', {'tab': 'exam'}, "The judge's canary moved",
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
        warn('judge_unfinished', 'warning', {'model': unfinished[0][0]["id"], 'kind': 'exam'},
            f"Answers that never finished: {named[0]}"
            + (f" and {len(named) - 1} more" if len(named) > 1 else ""),
            f"{'; '.join(named[:4])}{', …' if len(named) > 4 else ''}: answers that stopped "
            f"inside the model's reasoning before it answered. They are not scored — counted as "
            f"no answer and left out of every mean — and a topic where most answers never "
            f"finished has no score at all. Sit the exam again: a reasoning model now gets "
            f"room to answer.")
    local_judged = [m["name"] for m in model_rows if provisional_reason(m.get("judge"))]
    if local_judged:
        warn('judge_local', 'info', {'tab': 'exam'}, f"{len(local_judged)} model{'s' if len(local_judged) > 1 else ''} graded by a local judge: provisional",
            f"Judged scores for {len(local_judged)} model{'s' if len(local_judged) > 1 else ''} "
            f"({', '.join(local_judged[:4])}{', …' if len(local_judged) > 4 else ''}) were "
            f"graded by a local model — not a pinned benchmark. They are provisional: shown "
            f"greyed on the model page, never ranked, never in any average.")
    if any((m.get("judge") or {}).get("judge", {}).get("single_provider_loop") for m in model_rows):
        warn('judge_single_provider', 'info', {'tab': 'exam'}, 'Single-provider loop',
            "Single-provider loop: the judge shares a provider with the exam writer or the "
            "generator (ALLOW_SINGLE_PROVIDER_LOOP). Every judged score carries that caveat; "
            "self-preference in LLM judges is documented and large.")
    earlier = [m for m in model_rows if m.get("judgedEarlier")]
    if current_judge and earlier:
        by = sorted({m["judgedEarlier"]["by"] for m in earlier})
        warn('judge_other', 'warning', {'tab': 'exam'}, 'Some judged scores come from an earlier judge',
            f"{len(earlier)} model{'s' if len(earlier) != 1 else ''}' Knowledge exam scores were "
            f"judged by {', '.join(by)}, not the judge this server runs now. A score compares "
            f"only with the same judge's, so they are in each model's History until they are "
            f"judged again (AI models ▸ Re-judge).")

    dates = sorted(str(r["date"]) for r in by_model.values() if r["date"])
    return {
        "title": title,
        "generated": _dt.datetime.now(_TZ).strftime("%Y-%m-%d %H:%M %Z").strip(),
        "source": source,
        "models": model_rows,
        "accTasks": acc_tasks,
        "pplTasks": ppl_tasks,
        "required": required,          # the protocol list an official average needs
        # 12h.1: the three generative tasks, and what their makers publish
        "genTasks": list(GEN_TASKS),
        "thinkingModes": _thinking_modes(),
        "published": {m: {t: {"v": v, "note": note} for t, (v, note) in ts.items()}
                      for m, ts in _PUBLISHED.items()},
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
        # 12a: the Everyday pilot — its own key, read by its own two views
        "everyday": everyday,
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
/* 12g.1: the pipeline — four stages, left to right; one column on a phone */
.imphead .imp-title { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:0; }
.imphead .imp-model { font-size:var(--fs-2); font-weight:600; }
.stages { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:16px; }
@media (max-width:1100px) { .stages { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
@media (max-width:640px) { .stages { grid-template-columns:1fr; } }
.stage { min-width:0; }
.stage .stage-h { font-size:var(--fs-1); text-transform:uppercase; letter-spacing:.06em;
  color:var(--text-secondary); margin:0 0 8px; font-weight:600; }
.stagelist { list-style:none; margin:0; padding:0; border:1px solid var(--border);
  border-radius:var(--r-2); }
.stageitem { padding:8px 10px; border-top:1px solid var(--border); display:grid;
  grid-template-columns:minmax(0, 1fr) auto; gap:2px 8px; align-items:center; }
.stageitem:first-child { border-top:0; }
.stageitem .si-main { min-width:0; overflow-wrap:anywhere; }
.stageitem .si-sub { grid-column:1; font-size:var(--fs-1); color:var(--text-secondary); }
.stageitem .si-act { grid-column:2; grid-row:1 / span 2; }
.stageitem.landed { background:var(--heat-1); }
.stagemore { padding:6px 10px; border-top:1px solid var(--border); }
.stage-none { margin:0; }
.watch { margin-top:2px; }
.watch.dropped, .watch.dropped a { color:var(--warning-text); }
.imp-past { margin-top:14px; }
/* 12g.2: what a weak spot is — an exam topic or an Everyday group — and a
   group whose hidden half is under the line, greyed */
.imp-kind { display:inline-block; font-size:11px; letter-spacing:.04em; text-transform:uppercase;
  color:var(--text-secondary); border:1px solid var(--border); border-radius:999px;
  padding:0 6px; margin-right:6px; vertical-align:1px; }
.stageitem.greyed { color:var(--text-secondary); }
.stageitem.greyed .imp-kind { opacity:.7; }
.evsplit { margin:4px 0 6px; }
.evd-read { margin:6px 0 0; padding-left:20px; }
.evd-read li { margin:4px 0; }
/* the checks: a popover from the status dot, as the run counter's */
.checkspop { width:min(460px, calc(100vw - 32px)); max-height:min(60vh, 520px); overflow:auto;
  padding:10px 14px; }
.checkspop .checklist { margin:0; padding-left:18px; }
.checkspop .checklist li.checks-judged { list-style:none; color:var(--text-secondary);
  margin:0 0 6px; }
.checkspop .known-limits { list-style:none; margin:8px 0 4px; }
.checkspop .known-limits > details > summary { cursor:pointer; list-style:none;
  color:var(--text-secondary); }
.checkspop .known-limits > details > summary::-webkit-details-marker { display:none; }
.checkspop .known-limits .checklist { margin:6px 0 0; border:1px solid var(--border);
  border-radius:var(--r-2); padding:6px 6px 6px 22px; }
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
/* 12b.3: the known limits, folded at the end of the panel — a list inside
   it, not a second panel: the rule above made every .checklist in the bar a
   floating panel, and opened, this one floated out under the first */
.bar-checks .known-limits { list-style:none; margin:8px 0 4px; }
.bar-checks .known-limits > details > summary { cursor:pointer; list-style:none;
  color:var(--text-secondary); }
.bar-checks .known-limits > details > summary::-webkit-details-marker { display:none; }
.bar-checks .known-limits .checklist { position:static; box-shadow:none; max-height:none;
  overflow:visible; padding:0; margin:6px 0 0; border:1px solid var(--border);
  border-radius:var(--r-2); z-index:auto; }
/* 12b: one line at every width. Below 720px the four places are one
   Menu ▾ on the left; the right side keeps the run counter, Test a model,
   the status dot and the name, each at its shortest */
.menubtn { display:none; }
#testAct .t-short, #runs .t-short, button.who .t-short { display:none; }
@media (max-width:720px) {
  .bar-in { padding:0 16px; gap:8px; }
  .bar-title .t-full { display:none; }
  .bar-title .t-short { display:inline; }
  .bar-in .tabs { display:none; }
  .menubtn { display:inline-flex; align-items:center; flex:none; }
  .bar-right { gap:6px; }
  #testAct .t-full, #runs .t-full { display:none; }
  #testAct .t-short { display:inline; }
  button.who { max-width:96px; overflow:hidden; text-overflow:ellipsis; }
  .bar-checks .checklist { padding:10px 16px; }
}
@media (max-width:480px) {
  .bar-title { display:none; }
  .livebadge { padding:2px 5px; gap:4px; }
  .livebadge .t-full, #runs .t-idle { display:none; }
  .bar-in { gap:5px; }
  .bar-right { gap:3px; }
  .bar .barpill { padding:4px 6px; }
  .bar .runpill, .bar .statusdot { gap:4px; min-width:0; }
  #testAct button { padding:4px 8px; }
  button.who .t-full { display:none; }
  button.who .t-short { display:inline; }
}
/* 12b: a static report's date and a smoke run's warning, above the view */
.meta-chips { display:flex; gap:14px; flex-wrap:wrap; margin-top:12px; }
.meta-chips:empty { display:none; }
.meta-chips a.chip { border:0; background:none; padding:0; font-size:var(--fs-2); color:var(--accent); }
.meta-chips a.chip:hover { text-decoration:underline; }
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
/* 12h.1: Instruction & maths' two options, under the form */
/* 12i.1: AI models and the judge test */
table.aijobs { width:100%; border-collapse:collapse; margin-top:10px; }
table.aijobs th, table.aijobs td { text-align:left; padding:8px 10px 8px 0; vertical-align:top;
  border-bottom:1px solid var(--border); }
table.aijobs th { font-size:var(--fs-1); color:var(--text-secondary); font-weight:600; }
.aiprice { white-space:nowrap; }
.aimenu.pop { min-width:320px; max-width:min(92vw, 460px); padding:10px 12px; gap:6px; }
.aimenu .ailist { display:flex; flex-direction:column; gap:2px; max-height:min(60vh, 460px);
  overflow:auto; }
.aiitem { display:flex; flex-direction:column; align-items:flex-start; text-align:left; gap:2px;
  padding:6px 8px; border-radius:6px; }
.aiitem:hover, .aiitem:focus-visible { background:var(--accent-soft); }
.aiitem-sub { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.aiwhy { margin:0 8px 6px; }
.warntext { color:var(--warning-text); }
.airejudge .frm { display:flex; gap:8px; }
.jtcands { margin-top:10px; }
.jtpick { display:flex; flex-wrap:wrap; gap:6px 16px; margin:6px 0; }
.jtpick label { display:inline-flex; align-items:center; gap:6px; }
table.jtresult { width:100%; border-collapse:collapse; margin-top:12px; }
table.jtresult th, table.jtresult td { text-align:left; padding:8px 10px 8px 0;
  border-bottom:1px solid var(--border); }
table.jtresult th { font-size:var(--fs-1); color:var(--text-secondary); font-weight:600; }
table.jtresult tr.best td:first-child { font-weight:600; }
.jtmark .jtcrit { margin:4px 0 10px; padding-left:20px; }
.jtmark .jtrubric { white-space:pre-wrap; font-size:var(--fs-2); margin-bottom:10px; }
.jtmark .jtkeys { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 6px; }
.jtmark .jtkeys .chip-btn { min-width:44px; min-height:40px; font-family:var(--font-mono); }
.jtref { margin:0 0 10px; }
/* 12i.2: Build questions */
.qbopen { float:right; margin-left:12px; }
.qbsteps { display:flex; flex-wrap:wrap; gap:6px 18px; list-style:none; padding:0; margin:10px 0 0; }
.qbsteps li { color:var(--text-secondary); font-size:var(--fs-2); }
.qbsteps li.on { color:var(--text-primary); font-weight:600; }
.qbsteps li.done { color:var(--text-secondary); text-decoration:line-through; }
.qbprompt { margin:10px 0; }
.qbprompt textarea { width:100%; box-sizing:border-box; }
.qblocked { white-space:pre-wrap; background:var(--plane); border:1px solid var(--border);
  border-radius:6px; padding:8px 10px; max-height:14em; overflow:auto; color:var(--text-secondary); }
.qbdrafts { margin:6px 0 0; padding-left:18px; }
.qbitem { border-top:1px solid var(--border); margin-top:10px; padding-top:10px; }
.qbflags { margin:6px 0; padding-left:18px; }
.qbdup { display:grid; grid-template-columns:1fr 1fr; gap:8px 14px; margin:8px 0;
  padding:8px 10px; border:1px solid var(--border); border-radius:6px; }
.qbdup .frm { grid-column:1 / -1; display:flex; gap:8px; flex-wrap:wrap; }
.qbreasons, .qbacts { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 4px; }
.qbacts .chip-btn { min-height:40px; }
.qbedit textarea { width:100%; box-sizing:border-box; }
.qbaside { margin-top:10px; }
@media (max-width: 640px) { .qbdup { grid-template-columns:1fr; } .qbopen { float:none; margin:0 0 8px; } }
.genopts { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; margin-top:8px; }
.genopts label.spread { display:inline-flex; align-items:center; gap:6px; }
/* ---- 11d: Overview ------------------------------------------------------- */
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
/* 12a.4: a card's badge, one short line under the heading, with the room
   around it the section's caveat has beside its heading */
.hcard-badge { margin:2px 0; line-height:1.2; }
.hcard-badge .badge { margin-left:0; white-space:nowrap; }
/* 12a.4: "3 answers ran out of room", its own line under a score */
.evd-ranout { display:block; font-weight:400; }
.ktile .evd-ranout, .evscore .evd-ranout { display:inline; }
.hcard-link { margin-top:auto; padding-top:4px; font-size:var(--fs-2); font-family:var(--font-sans);
  color:var(--accent); text-decoration:none; }
.hcard-link:hover, .hcard-link:focus-visible { text-decoration:underline; }
.hlgrid { align-items:stretch; }
/* ---- 11d: the model page -------------------------------------------------- */
/* 12b: the header — name, facts, Test this model; a tile per kind of test */
.mtop { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap; }
.mtop-l { min-width:0; }
.mfacts { margin:4px 0 0; font-size:var(--fs-2); color:var(--muted); }
.ktiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px;
  margin-top:var(--sp-4); }
.ktile { display:flex; flex-direction:column; align-items:flex-start; gap:4px; text-align:left;
  border:1px solid var(--border); border-radius:var(--r-2); padding:12px 14px;
  background:var(--surface-1); color:var(--text-primary); font:inherit; min-width:0; }
button.ktile { cursor:pointer; }
button.ktile:hover { border-color:var(--axis); background:var(--accent-soft); }
.ktile-k { margin:0; }
/* the hcard's number: sans, tabular, one line (11h) */
.ktile-v { font-family:var(--font-sans); font-size:var(--fs-5); font-weight:700;
  font-variant-numeric:tabular-nums; line-height:1.1; }
.ktile-sub { font-size:var(--fs-1); color:var(--muted); display:flex; gap:6px; align-items:center;
  flex-wrap:wrap; }
.ktile.none { background:var(--plane); }
.ktile-none { font-size:var(--fs-2); color:var(--muted); }
button.ktest { padding:0 4px; min-height:0; font-size:var(--fs-2); }
/* the tabs: Scores · Answers · Improve · History */
.mtabs { display:flex; gap:4px; margin:var(--sp-4) 0 0; border-bottom:1px solid var(--border);
  overflow-x:auto; }
.mtab { background:none; border:0; border-bottom:2px solid transparent; border-radius:0;
  padding:8px 12px; font:inherit; font-size:var(--fs-2); color:var(--text-secondary); cursor:pointer;
  white-space:nowrap; }
.mtab:hover { color:var(--text-primary); }
.mtab[aria-selected="true"] { color:var(--text-primary); border-bottom-color:var(--accent); font-weight:600; }
/* a kind's block on Scores: its name and number, the rest behind the fold */
details.kblock > summary { display:flex; align-items:baseline; gap:12px; cursor:pointer;
  list-style:none; }
details.kblock > summary::-webkit-details-marker { display:none; }
details.kblock > summary::before { content:'▸'; color:var(--muted); }
details.kblock[open] > summary::before { content:'▾'; }
.kblock-k { font-size:var(--fs-4); font-weight:700; }
.kblock-v { font-family:var(--font-sans); font-size:var(--fs-3); font-weight:600;
  font-variant-numeric:tabular-nums; color:var(--text-secondary); }
.kpart { margin-top:var(--sp-5); }
.kpart > h2 { font-size:var(--fs-3); }
details.kfold { margin-top:var(--sp-4); }
details.kfold > summary { cursor:pointer; color:var(--accent); font-size:var(--fs-2);
  list-style:none; }
details.kfold > summary::-webkit-details-marker { display:none; }
/* the pilot's block says its name and count once, in the summary; its badge
   is the tile's, in the header (a caveat once per page) */
.kblock .evhead > h2, .kblock .evhead .evcount { display:none; }
.kacts { margin-top:var(--sp-3); }
.evrow-static { cursor:default; }
/* Home: Needs you */
ul.needs { margin:8px 0 0; padding-left:18px; display:flex; flex-direction:column; gap:6px; }
ul.needs a { font-size:var(--fs-2); }
.mhero .mhead { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-top:4px; }
.mhero h1.mtitle { font-size:var(--fs-5); font-weight:800; letter-spacing:-0.025em; margin:0; }
.mhero .eyebrow { margin:0; }
.mprose { color:var(--muted); font-size:var(--fs-2); max-width:72ch; margin:14px 0 0; }
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
/* 12h.1: a generative cell's note under its number — ran out of room, a
   subset, far below published — in words, small, never a second number line */
table.lb td .cellnote { display:block; font-family:var(--font-sans); font-size:10px;
  font-weight:400; line-height:1.2; color:var(--muted); white-space:nowrap; }
table.lb td .cellnote.warn { color:var(--warning-text); }
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
.msit-cta { display:flex; flex-direction:column; justify-content:center; align-items:flex-start;
  gap:6px; flex:0 0 auto; }
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
/* 12h.2: Benchmarks ▾, Models ▾ and Filters ▾ sit together beside the chips;
   on a phone they stack under them, on a row of their own */
.lbbar .pickers { display:flex; gap:8px; align-items:center; flex:none; }
.lbbar .chipdiv { flex:none; width:1px; align-self:stretch; margin:2px 4px; background:var(--border); }
.lbbar .savedchip { flex:none; display:inline-flex; align-items:center; }
.lbbar .chip-more { border:0; background:none; cursor:pointer; padding:0 6px; font-size:var(--fs-1);
  color:var(--text-secondary); }
.lbbar .chip-more:hover, .lbbar .chip-more[aria-expanded="true"] { color:var(--text-primary); }
.lbbar .viewform { display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; margin:8px 0 2px; }
.lbbar .viewform input { min-width:0; width:220px; max-width:100%; }
.customline { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:baseline;
  gap:4px 12px; margin:10px 0 0; font-size:var(--fs-1); }
.customline .cl-what { color:var(--text-secondary); min-width:0; overflow-wrap:anywhere; }
.customline .cl-what b { color:var(--text-primary); }
.customline .cl-acts { display:flex; gap:10px; align-items:baseline; flex:none; }
.benchmenu.pop { min-width:260px; padding:10px 12px; gap:6px; }
.benchmenu .benchlist { max-height:min(60vh, 520px); overflow:auto; }
.benchmenu .colgroup { display:flex; flex-direction:column; gap:2px; margin:6px 0 8px; }
.benchmenu .colgroup label { display:flex; align-items:center; gap:4px; white-space:nowrap; }
/* 12i.0: a benchmark nothing has run yet, listed and greyed */
.benchmenu .colgroup label.notrun { color:var(--text-secondary); }
.modelsmenu .mgroup { margin:6px 0 2px; text-transform:none; }
@media (max-width:720px) {
  .lbbar.narrow .chiprow { flex-wrap:wrap; row-gap:8px; }
  .lbbar.narrow .chips { flex-basis:100%; }
  .lbbar .pickers { flex-wrap:wrap; }
}
.fsheet { position:fixed; left:0; right:0; bottom:0; z-index:55; max-height:70vh; overflow:auto;
  background:var(--surface-1); border-top:1px solid var(--border); border-radius:var(--r-2) var(--r-2) 0 0;
  box-shadow:0 -12px 32px rgba(0,0,0,.18); padding:12px 16px 20px; }
.fsheet-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
/* 12b: Filters ▾ at every width — a panel under the chips on a wide screen,
   the sheet from the bottom on a phone */
@media (min-width:721px) {
  .fsheet { position:static; max-height:none; border:1px solid var(--border);
    border-radius:var(--r-2); box-shadow:none; margin:8px 0 4px; padding:12px 14px; }
  .fsheet .pills { flex-direction:row; flex-wrap:wrap; align-items:center; }
  .fsheet .pills .pill { width:auto; }
}
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
/* ---- 12a: the Everyday pilot — a block on the model page, and #everyday ---- */
.evhead h2 { margin:0; }
.evscore { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
.evcount { font-family:var(--font-mono); font-size:var(--fs-3); font-weight:650;
  font-variant-numeric:tabular-nums; color:var(--text-primary); }
.evrows { margin-top:12px; border-top:1px solid var(--border); }
.evitem { border-bottom:1px solid var(--border); }
.evrow { box-sizing:border-box; display:grid; width:100%; cursor:pointer; margin:0;
  grid-template-columns:minmax(96px, 140px) 28px minmax(0, 1fr); align-items:center; gap:10px;
  min-height:40px; padding:6px 4px; font:inherit; font-size:var(--fs-2); text-align:left;
  color:var(--text-primary); background:none; border:0; border-radius:0; }
.evrow:hover { background:var(--accent-soft); }
.evrow:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.evgroup { font-weight:600; }
.evmark { font-weight:700; text-align:center; }
.evmark.ok, .evcell.ok { color:var(--success-text); }
.evmark.no, .evcell.no { color:var(--critical-text); }
.evmark.wait, .evcell.wait { color:var(--muted); }
.evreason { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-secondary); }
.evans-wrap { padding:4px 4px 14px; }
.evq-label { margin:10px 0 4px; font-size:var(--fs-1); font-weight:600; color:var(--text-secondary); }
.evq { margin:0; padding:8px 12px; border-left:3px solid var(--border); background:var(--plane);
  border-radius:0 var(--r-1) var(--r-1) 0; font-size:var(--fs-2); white-space:pre-wrap;
  overflow-wrap:anywhere; }
.evans { white-space:pre-wrap; overflow-wrap:anywhere; font-size:var(--fs-2); line-height:1.55;
  color:var(--text-primary); }
.evans.none { color:var(--text-secondary); }
.evthink { margin-top:10px; font-size:var(--fs-1); }
.evthink > summary { cursor:pointer; color:var(--text-secondary); list-style:none; }
.evthink > summary::-webkit-details-marker { display:none; }
.evthink[open] > summary { color:var(--text-primary); }
.evthink-t { margin-top:6px; padding:8px 12px; white-space:pre-wrap; overflow-wrap:anywhere;
  color:var(--text-secondary); background:var(--plane); border-radius:var(--r-1); }
.evtable { border-collapse:collapse; width:100%; }
.evtable th, .evtable td { border-bottom:1px solid var(--border); padding:8px 10px;
  vertical-align:top; }
.evtable thead th { vertical-align:bottom; }
.evq-th { min-width:220px; width:40%; }
.evm-th { text-align:center; min-width:96px; font-family:var(--font-sans); text-transform:none;
  letter-spacing:0; font-size:var(--fs-2); color:var(--text-primary); }
.evm-th a, .evm-th > span:first-child { display:block; font-weight:600; }
.evm-count .evsep { display:none; }
.evm-count .evmissing { display:block; white-space:nowrap; }
.evpart .evcell { color:var(--text-secondary); font-weight:500; }
.evm-count { display:block; font-family:var(--font-mono); font-size:var(--fs-1);
  color:var(--text-secondary); font-weight:400; margin-top:2px; }
.evq-cell { text-align:left; font-family:var(--font-sans); text-transform:none; letter-spacing:0;
  font-weight:400; white-space:normal; }
.evq-short { display:block; font-weight:650; font-size:var(--fs-2); color:var(--text-primary); }
.evq-group { display:block; font-size:var(--fs-1); color:var(--text-secondary); }
.evq-full { display:block; margin-top:4px; font-size:var(--fs-1); color:var(--text-secondary);
  max-width:360px; overflow-wrap:anywhere; }
.evcell-td { text-align:center; vertical-align:middle !important; }
/* 12a.2: a cell is a group's n of k, a number: the mono figures */
.evcell { min-width:40px; min-height:32px; font-family:var(--font-mono); font-size:var(--fs-2);
  font-weight:600; font-variant-numeric:tabular-nums; padding:4px 8px; white-space:nowrap;
  background:transparent; border:1px solid transparent; border-radius:var(--r-1);
  cursor:pointer; }
.evcell:hover { border-color:var(--border); background:var(--plane); }
.evnav { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
/* 12a.2: a group's row (its name and n of k), and a group's answers — one row
   per question: the mark, the question, the reason */
.evgrow { grid-template-columns:minmax(120px, 1fr) auto; }
.evgcount { font-family:var(--font-mono); font-variant-numeric:tabular-nums; font-weight:600;
  color:var(--text-primary); }
.evglist { padding:0 0 8px 12px; }
.evqrow { grid-template-columns:28px minmax(0, 1.4fr) minmax(0, 1fr); min-height:36px;
  border-bottom:1px solid var(--border); }
.evprompt { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.evcell.on { border-color:var(--accent); background:var(--accent-soft); }
.evpanel { margin-top:var(--sp-4); border-top:1px solid var(--border); padding-top:var(--sp-3); }
.evpanel h3 { margin:0; font-size:var(--fs-3); }
.evbank-g { border-bottom:1px solid var(--border); padding:8px 0; }
.evbank-g > summary { cursor:pointer; }
ol.evbank { margin:8px 0 4px; padding-left:22px; display:flex; flex-direction:column; gap:10px; }
ol.evbank li p { margin:2px 0; }
.evverdict { display:flex; gap:8px; align-items:baseline; margin:0 0 8px; font-size:var(--fs-2);
  font-weight:600; }
.evfail-check { display:block; margin-top:2px; font-weight:400; font-size:var(--fs-1);
  color:var(--text-secondary); }
/* on a phone the question column gives the marks room: its label and group
   stay, the full question wraps under them */
@media (max-width:600px) {
  .evq-th { min-width:150px; width:auto; }
  .evq-full { max-width:180px; }
  .evtable th, .evtable td { padding:8px 6px; }
  .evm-th { min-width:72px; }
}
.evpick { display:flex; flex-direction:column; gap:2px; margin:10px 0; max-height:320px;
  overflow:auto; }
.evpick-row { display:flex; align-items:center; gap:8px; min-height:32px; font-size:var(--fs-2); }
.evpick-name { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
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
/* ---- 12b: the header's right side ---- */
.runpill { display:inline-flex; align-items:center; gap:6px; }
.runpill .num { font-family:var(--font-mono); }
.dot.idle { background:var(--axis); }
.dot.pulse { animation:livepulse 1.6s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) { .dot.pulse { animation:none; } }
.statusdot { gap:6px; min-width:32px; justify-content:center; }
.statusdot .statusn { font-family:var(--font-mono); }
#testAct button { min-height:32px; white-space:nowrap; font-size:var(--fs-1); padding:4px 14px; }
.runspop { width:min(460px, calc(100vw - 24px)); }
.runslist { display:flex; flex-direction:column; gap:2px; }
.runline { display:grid; grid-template-columns:auto minmax(0, 1fr) auto; gap:4px 10px;
  align-items:baseline; padding:6px 4px; border-radius:var(--r-1); font-size:var(--fs-2); }
.runline .runname { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.runline .runprog { grid-column:1 / -1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.runslist.full .runline { grid-template-columns:auto minmax(0, 1fr) auto minmax(0, 2fr); }
.runslist.full .runline .runprog { grid-column:auto; }
.runslist .runhead { margin:8px 4px 2px; }
.runs-all { display:block; margin:8px 4px 2px; font-weight:600; }
.whopop .menusect { display:flex; flex-direction:column; gap:6px; margin:10px 0 6px;
  padding-top:10px; border-top:1px solid var(--border); }
.whopop .themeopts { display:flex; flex-wrap:wrap; gap:6px; }
.whopop .themeopts [aria-checked="true"] { background:var(--accent-soft); border-color:var(--accent); }
.whopop .menulink { display:block; width:100%; text-align:left; padding:7px 8px; border:0;
  background:none; border-radius:var(--r-1); font-size:var(--fs-2); color:var(--text-primary); }
.whopop .menulink:hover, .whopop .menulink:focus { background:var(--accent-soft); }
.placemenu .subitem { padding-left:22px !important; font-size:var(--fs-1); }
/* the switch at the top of Improve, Benchmarks and Models */
.subswitch { display:flex; gap:8px; flex-wrap:wrap; margin:18px 0 0; }
.card .subswitch { margin:10px 0 2px; }
.subswitch + .card { margin-top:14px; }
/* Test a model: the Submit form, in a dialog */
.dlg.testdlg { max-width:880px; position:relative; max-height:calc(100vh - 32px); overflow:auto;
  padding:14px 16px; }
.dlg.testdlg > .card { border:0; box-shadow:none; margin:0; padding:6px 4px; }
.dlg-x { position:absolute; right:12px; top:12px; z-index:1; }
/* Models: a row opens the model page; the untested sit under one line */
table.lb tr.clickrow { cursor:pointer; }
table.lb tr.clickrow:hover td { background:var(--accent-soft); }
table.lb tr.nottested td { background:var(--plane); }
table.lb tr.nottested-row td { font-size:var(--fs-2); padding-left:24px; }
table.lb td.evmark { font-weight:700; }
/* a table with no rank column (Knowledge exam, Everyday tasks) pins its
   model column at the edge */
table.lb.norank th.model, table.lb.norank td.model { left:0; }
.helplinks { margin:8px 0 0; padding-left:20px; font-size:var(--fs-2); line-height:1.9; }
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
  everyday: false,                     // 12a: the Everyday pilot page, #everyday (hash-routed)
  read: null,                          // 11g: the open reader, { kind, id, n } (hash-routed)
  src: 'all',                          // All | Models | Checkpoints — a filter, nothing hidden by default
  panelOpen: {},                       // per-task "show all bars" toggles
  sort: { key: 'avg', dir: -1 },
  runsQ: '',                           // the Evals tab query string
  runsSort: { idx: 0, dir: 1 },        // column sort for the metric query table
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
  ai: {},                        // 12i.1: the AI models page and the judge test
  qb: {},                        // 12i.2: the question builder
  rv: { llm: null, proposals: [], datasets: [], loaded: false, msg: '',
        // 11j: which of the four views, and the small per-proposal choices
        view: '', focus: {}, watch: new Set(), answers: {}, answersOpen: 0,
        spread: {}, count: {}, fmt: {}, rejecting: {}, landed: null,
        detailsOpen: {}, specOpen: {}, landedDs: null },                    // Review tab
  ex: { status: null, candidates: [], loaded: false, msg: '', topic: '' },   // Exam tab
  // 12b: the model page — its tab, which kind blocks are open, the Answers kind
  mtab: null, mblk: {}, mans: null, mansGroup: '',
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
// 12g.2: an Everyday group's task, "everyday:instructions", in words
const evdTaskName = t => 'Everyday · ' + ((((DATA.everyday || {}).groups) || [])
  .find(([k]) => k === String(t).slice(9)) || [null, String(t).slice(9)])[1];
const tName = t => String(t).startsWith('everyday:') ? evdTaskName(t)
  : ((DATA.judged || {}).topics || {})[t] || t;
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
    href: '#tab=improve&sub=review&read=proposal:' + open.id, text: 'Review it →',
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
  : String(t).startsWith('everyday:') ? evdTaskName(t)
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

function vJudged(m, more = []) {
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
    el('div', { class: 'sechead' }, el('h2', { text: 'Judged free response — the exam' }),
      ok ? judgeChecked() : ''),
    // 11h: say what it is, then how it works behind a click
    el('p', { class: 'sub', text: 'Open questions per topic, answered in writing and graded 0–4 '
      + 'against a written rubric.' }),
    el('details', { class: 'howto small', 'data-how-judged': '1' },
      el('summary', { text: 'How this works ▸' }),
      el('p', { class: 'small', text: (prov ? 'The judge is a local model whose id cannot be '
        + 'pinned. ' : 'The judge is an API model pinned to a dated id. ')
        + 'It grades single answers, never pairs. Length is in the rubric and reported under '
        + 'More detail. Thirty fixed scripts are graded again every run, so a change to the '
        + 'judge would show. The judge’s runs and file are in History.' })));
  // 12b: score against answer length and the earlier exams, folded
  const detail = el('details', { class: 'kfold', 'data-more-detail': 'judged' },
    el('summary', { text: 'More detail ▸' }));
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
  if (!j) { card.append(note('Not judged yet. Sit the exam from this page, or from Test a '
    + 'model.')); return card; }
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
  // 12b.3: steady, the main view says so in two words; the deviations are
  // How this works' — system words, for whoever opens it
  if (cn && !cn.drifted) {
    card.append(el('p', { class: 'small', 'data-canary': 'steady' },
      el('b', { text: 'Judge steady.' })));
    canaryPara.removeAttribute('data-canary');
    canaryPara.querySelector('b').textContent = 'The canary: ';
    const how = card.querySelector('[data-how-judged]');
    if (how) how.append(canaryPara);
  }

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
      + 'is greyed. Its bank is still being written, on Benchmarks ▸ Knowledge exam.' }));
    // score vs length: one row per topic, a column per length (11d) — the
    // row-per-bucket table was 148 rows for 37 topics
    const LEN = [['≤20 words', '≤ 20'], ['21–50', '21–50'], ['51–120', '51–120'], ['>120', '> 120']];
    const withLen = cats.filter(t => (j.tasks[t].score_vs_length || []).length);
    if (withLen.length) {
      const drift = withLen.map(t => { const b = j.tasks[t].score_vs_length || [];
        return b.length > 1 ? b[b.length - 1].mean - b[0].mean : 0; });
      const worst = Math.max(...drift);
      detail.append(el('div', { class: 'dxh', text: 'Score against answer length' }));
      detail.append(el('p', { class: 'small', text: worst >= 1
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
      detail.append(withLen.length > 10
        ? el('details', { class: 'lenfold', 'data-length-fold': '1' },
            el('summary', { text: `Show the table (${withLen.length} topics)` }), table)
        : table);
    }
    // 11m: the per-topic block — by criterion, its flags and its by-field
    // tables, under one topic picker — is gone from the model page, at
    // masein's request. Each answer card keeps its own criteria strip.
    // 12b: the answers themselves are the model page's Answers tab
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
  // 12b: the judge's ids are History's (How it was graded)
  for (const x of more) if (x) { x.classList.remove('card'); detail.append(x); }
  if (detail.children.length > 1) card.append(detail);
  return card;
}
// the judge behind a model's exam scores, ids and all — History's, not Scores'
function judgeIdLine(m) {
  const J = DATA.judged || {}, j = m.judge || {}, jj = j.judge || {};
  const st = m.judgeState || {};
  return `Judge ${jj.id} (${jj.provider})`
    + (jj.batch_id ? ` · batch ${String(jj.batch_id).slice(0, 18)}` : '')
    + (jj.weights_sha256 ? ` · weights ${String(jj.weights_sha256).slice(0, 12)}` : '')
    + ` · prompt v${jj.prompt_version} ${String(jj.prompt_sha256 || '').slice(0, 12)} · rubrics `
    + [...new Set(Object.values(jj.rubrics || {}).map(r => rubricVersion(r.version)))]
        .join(', ')
    + (st.current ? '' : ' · not the judge this server runs now'
      + (J.current && J.current.id ? ` (${J.current.id})` : ''));
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
    'From ', el('a', { href: '#tab=improve&sub=training', text: `training run #${tt.run_id}`,
      onclick: e => { e.preventDefault(); state.trSel = [tt.run_id];
        navigate({ tab: 'training', model: null, topic: null }); } }),
    ', which trained on ',
    tt.datasets.map((d, i) => el('span', {}, i ? ', ' : '',
      el('a', { href: `api/datasets/${d}`, target: '_blank', rel: 'noopener',
        text: `dataset #${d}` }))),
    tt.proposals.length ? el('span', {}, ' from ',
      tt.proposals.map((pid, i) => el('span', {}, i ? ', ' : '',
        el('a', { href: '#tab=improve&sub=model', text: `proposal #${pid}`,
          onclick: e => { e.preventDefault(); openImprove(m.trainedFrom
            ? m.trainedFrom.base : null); } }), ' ',
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
  const back = el('a', { class: 'backlink', href: '#' + viewHash(state.tab), onclick: backTo(state.tab),
    text: '← Back to ' + (TABS.find(([id]) => id === state.tab) || [, 'the board'])[1] });
  // 12b: a header — the name, one line of facts, Test this model, and one
  // tile per kind of test — then four tabs, remembered per viewer. Improve
  // is a tab only when the Review lists hold something of this model's
  rvNeeded();
  const kinds = modelKinds(m);
  const tabs = modelTabs(m);
  const cur = modelTab(tabs.map(([id]) => id));
  const strip = el('div', { class: 'mtabs', role: 'tablist', 'aria-label': 'this model',
      'data-model-tabs': '1',
      onkeydown: e => {
        const ids = tabs.map(([id]) => id), k = ids.indexOf(cur);
        const to = e.key === 'ArrowRight' ? ids[(k + 1) % ids.length]
          : e.key === 'ArrowLeft' ? ids[(k - 1 + ids.length) % ids.length]
          : e.key === 'Home' ? ids[0] : e.key === 'End' ? ids[ids.length - 1] : null;
        if (!to) return;
        e.preventDefault();
        state.after = { focus: `[data-mtab="${to}"]` };
        setModelTab(to);
      } },
    tabs.map(([id, label]) => el('button', { role: 'tab', class: 'mtab', 'data-mtab': id,
      id: 'mtab-' + id, 'aria-selected': String(id === cur), 'aria-controls': 'mpanel',
      tabindex: id === cur ? '0' : '-1', text: label, onclick: () => setModelTab(id) })));
  const body = cur === 'answers' ? modelAnswersTab(m, kinds)
    : cur === 'improve' ? modelImproveTab(m)
    : cur === 'history' ? modelHistoryTab(m)
    : modelScoresTab(m, kinds);
  return [back, modelHead(m, kinds), modelSitPanel(m), strip,
    el('div', { id: 'mpanel', role: 'tabpanel', 'aria-labelledby': 'mtab-' + cur,
      'data-mtab-panel': cur }, body)].filter(Boolean);
}

// the kinds of test, named the same and in the same order everywhere (12b).
// A kind the board has no test for is not shown at all; one this model has
// not taken is a tile that says so, with the button that fills it
function modelKinds(m) {
  const J = DATA.judged || {}, E = evd(), e = evdOf(m.id);
  const jt = Object.entries((m.judge || {}).tasks || {}).filter(([t]) => t.startsWith('exam_'));
  const ran = m.date ? Date.parse(String(m.date)) / 1000 : 0;
  return [
    { kind: 'standard', label: 'Standard',
      taken: [...DATA.accTasks, ...DATA.pplTasks].some(t => cell(t, m.id)), at: ran || 0 },
    (J.exam || []).length ? { kind: 'exam', label: 'Knowledge exam',
      taken: jt.length > 0 || ((m.judge || {}).history || []).length > 0,
      at: Math.max(0, ...jt.map(([, v]) => v.judged_at || 0)) } : null,
    (E.questions || []).length ? { kind: 'everyday', label: 'Everyday tasks', taken: !!e,
      at: e ? e.marked_at || 0 : 0 } : null,
  ].filter(Boolean);
}
// a kind's one number, and the line under it
function kindValue(m, kind) {
  if (kind === 'standard') {
    const avg = officialAvg(m), r = rankOf(m);
    return avg != null
      ? [(100 * avg).toFixed(1), `${state.avgMode === 'raw' ? 'raw' : 'above chance'} · `
        + `${m.nhave} of ${m.nreq} tasks` + (r ? ` · #${r.n} of ${r.of}` : '')]
      : ['—', `preliminary · ${m.nhave} of ${m.nreq} tasks`];
  }
  if (kind === 'exam') {
    const J = DATA.judged || {};
    const n = Object.entries((m.judge || {}).tasks || {})
      .filter(([t, v]) => (J.exam || []).includes(t) && pubScore(v) != null).length;
    const N = (J.exam || []).length;
    if (m.judgedAvg != null)
      return [num(m.judgedAvg, 2), `out of 4 · ${n} of ${N} topics` + (judgedOkM(m) ? '' : ' · not ranked')];
    // 12b.3: no average — a provisional score never enters one — but not a
    // dash either: the topics judged, and the weakest, which a topic score is
    const w = weakestTopic(m);
    return [`${n} of ${N}`, 'topics judged'
      + (w ? ` · weakest: ${frName(w.task)} ${num(w.v, 2)} / 4` : '')];
  }
  const e = evdOf(m.id);
  // 12i.0: a partial count says so under it
  return [e ? evdCount(e) : '—', e && evdMissing(e) ? `${evdMissing(e)} not asked yet` : ''];
}
// a model's weakest judged topic on the current exam: { task, v }, or null
function weakestTopic(m) {
  const J = DATA.judged || {};
  const xs = Object.entries((m.judge || {}).tasks || {})
    .filter(([t, v]) => (J.exam || []).includes(t) && pubScore(v) != null)
    .map(([t, v]) => ({ task: t, v: pubScore(v) })).sort((a, b) => a.v - b.v);
  return xs[0] || null;
}
// the exam's one badge on a tile or a card: its scores are not evidence yet
const provBadge = (why, attrs = {}) => el('span', { class: 'badge prelim', 'data-provisional-badge': '1',
  title: why, text: 'provisional', ...attrs });

// "15.3 points behind good-750m-tuned-test — a real gap"
function avgVerdictOf(m) {
  const avg = officialAvg(m);
  if (avg == null) return `Preliminary — ${m.nhave} of ${m.nreq} required tasks, so no `
    + 'average and no rank.';
  const ranked = DATA.models.filter(x => officialAvg(x) != null && !x.duplicateOf)
    .sort((x, y) => officialAvg(y) - officialAvg(x));
  const i = ranked.findIndex(x => x.id === m.id);
  const other = i > 0 ? ranked[i - 1] : ranked[1];
  if (!other) return 'The only ranked model on the board.';
  const d = avg - officialAvg(other), sa = officialSe(m), sb = officialSe(other);
  const words = i > 0 ? `${(100 * -d).toFixed(1)} points behind ${other.name}`
                      : `Leads ${other.name} by ${(100 * d).toFixed(1)} points`;
  return `${words[0].toUpperCase()}${words.slice(1)} — ` + (sa == null || sb == null
    ? 'no standard error to test it.'
    : Math.abs(d) / Math.sqrt(sa * sa + sb * sb || 1e-12) > 1.96 ? 'a real gap.' : 'within noise.');
}

function modelHead(m, kinds) {
  const facts = [m.params ? P(m.params) : null, m.source === 'artifact' ? 'checkpoint' : m.kind,
    famOf(m)].filter(Boolean).join(' · ');
  return el('div', { class: 'card mhero', 'data-model-hero': '1' },
    el('div', { class: 'mtop' },
      el('div', { class: 'mtop-l' },
        el('div', { class: 'mhead' }, el('h1', { class: 'mtitle', text: m.name }),
          warnBadge(m) || '', dupBadge(m) || ''),
        el('p', { class: 'mfacts', 'data-model-facts': '1', text: facts }),
        trainedFromLine(m))),
      // 12b.3: the page's one main action is the header's, which reads Test
      // this model here — two filled buttons side by side was one too many
    el('div', { class: 'ktiles', 'data-kind-tiles': '1' }, kinds.map(k => kindTile(m, k))));
}
// 12g.1: what a checkpoint was trained from — set once by a person, from the
// models on the board, or recorded by its training run. Improve's Retests pair
// it with that model; a checkpoint without one is not in Improve
const isCheckpoint = m => m.source === 'artifact' || !!m.trainedFrom
  || (m.tainted || []).length > 0;
function trainedFromLine(m) {
  if (!isCheckpoint(m)) return '';
  const tf = m.trainedFrom;
  const base = tf && DATA.models.find(x => x.id === tf.base);
  const pick = LIVE ? popover(el('button', { class: 'quiet', 'data-trained-from-set': m.id,
      text: tf ? 'change' : 'Set it ▾' }),
    () => {
      const list = el('div', { class: 'mlist' });
      const fill = q => list.replaceChildren(...DATA.models.filter(x => x.id !== m.id
          && !x.duplicateOf && (!q || (x.name + ' ' + x.id).toLowerCase().includes(q.toLowerCase())))
        .map(x => el('button', { role: 'menuitem', 'data-trained-from-pick': x.id,
          onclick: async () => {
            if (!whoName()) { popClose(); askName(); return; }
            try {
              await post('api/trained-from', { model: m.id, base: x.id, by: whoName() });
              popClose(true);
              toast(`${m.name}: trained from ${x.name}`, { key: 'trained-from' });
              await refreshResults();
            } catch (e) { toast('Refused. ' + String((e && e.message) || e), { key: 'trained-from' }); }
          } }, x.name, el('span', { class: 'se', text: ' ' + x.id }))));
      fill('');
      return el('div', { class: 'moremenu modelsmenu', id: 'pop-trained-from',
          'aria-label': 'trained from' },
        el('input', { type: 'search', placeholder: 'the model it was trained from…',
          'aria-label': 'search models', 'data-keep': 'trainedfrom',
          oninput: e => fill(e.target.value) }), list);
    }, { key: 'trained-from', menu: false }) : '';
  if (!tf) return el('p', { class: 'small se', 'data-trained-from': 'unset' },
    'Set what this was trained from to see it in Improve', pick ? [' · ', pick] : '');
  return el('p', { class: 'small', 'data-trained-from': tf.base },
    'Trained from ', base ? el('a', { href: '#model=' + encodeURIComponent(base.id),
      text: base.name }) : el('span', { class: 'mono', text: tf.base }),
    el('span', { class: 'se', text: tf.source === 'person' ? ` · set by ${tf.by}`
      : ` · recorded by training run #${tf.run}` }),
    base ? [' · ', el('a', { href: '#', 'data-trained-from-improve': base.id,
      text: 'see it in Improve', onclick: e => { e.preventDefault(); openImprove(base.id); } })]
      : el('span', { class: 'se', text: ' · not on the board, so not in Improve' }),
    pick ? [' · ', pick] : '');
}
function kindTile(m, k) {
  const attrs = { class: 'ktile' + (k.taken ? '' : ' none'), 'data-kind-tile': k.kind };
  if (!k.taken) {
    const msg = k.kind === 'everyday' ? (state.evdMsg || {})[m.id] : '';
    return el('div', { ...attrs, ...(k.kind === 'everyday' ? { 'data-everyday-none': m.id } : {}) },
      el('span', { class: 'eyebrow ktile-k', text: k.label }),
      el('span', { class: 'ktile-none' }, 'Not tested',
        LIVE ? [el('span', { class: 'se', text: ' · ' }), kindTest(m, k.kind)] : ''),
      // 12a.4: it answered an earlier wording, which is in its History
      k.kind === 'everyday' && evdEarlier(m.id) ? el('span', { class: 'small se',
        'data-evd-earlier-note': m.id, text: (evdEarlier(m.id).label === 'all questions, before '
          + 'the split' ? 'scored on all questions, before the split' : 'answered an earlier '
          + 'wording') + ' · in History' }) : '',
      msg ? el('span', { class: 'warn small', 'data-everyday-refused': m.id, text: msg }) : '');
  }
  const [v, sub] = kindValue(m, k.kind);
  return el('button', { ...attrs, title: k.kind === 'standard' ? avgVerdictOf(m) : null,
      onclick: () => showKind(m, k.kind) },
    el('span', { class: 'eyebrow ktile-k', text: k.label }),
    el('span', { class: 'ktile-v', 'data-kind-value': k.kind, text: v }),
    el('span', { class: 'ktile-sub' }, sub,
      k.kind === 'everyday' ? evdBadge(evdOf(m.id).provisional) : '',
      // the provisional badge, once, on the tile it is about
      k.kind === 'exam' && !judgedOkM(m) ? provBadge(whyProvisional(m)) : ''),
    k.kind === 'everyday' ? evdRanOut(evdOf(m.id)) : '');
}
// the button that fills an empty tile: the form for Standard, the exam's own
// topic picker, and the pilot's one-click queue
function kindTest(m, kind) {
  if (kind === 'exam') return el('button', { class: 'quiet ktest', 'data-sit-open': m.id,
    'aria-expanded': String(state.msit.open && state.msit.model === m.id), text: 'Test',
    onclick: () => openSit(m.id) });
  if (kind === 'everyday') return evdTestBtn(m, 'quiet ktest');
  return el('button', { class: 'quiet ktest', 'data-kind-test': 'standard', text: 'Test',
    onclick: () => { state.sub.suite = 'full'; openTest(m.id); } });
}
// a tile opens its block, on Scores
function showKind(m, kind) {
  (state.mblk[m.id] = state.mblk[m.id] || {})[kind] = true;
  state.after = { scroll: `[data-kind-block="${kind}"]` };
  setModelTab('scores');
}

const MODEL_TABS = { scores: 'Scores', answers: 'Answers', improve: 'Improve', history: 'History' };
function modelTabs(m) {
  return ['scores', 'answers', ...(modelImproveRows(m).n ? ['improve'] : []), 'history']
    .map(id => [id, MODEL_TABS[id]]);
}
// remembered per viewer; a tab this model has not got (Improve) falls back
// to Scores without forgetting the choice
function modelTab(have) {
  let t = state.mtab;
  if (!t) try { t = localStorage.getItem('bench-model-tab'); } catch (e) { /* private */ }
  return have.includes(t) ? t : 'scores';
}
function setModelTab(t) {
  state.mtab = t;
  try { localStorage.setItem('bench-model-tab', t); } catch (e) { /* private */ }
  render();
}

// ---- Scores: one block per kind the model has taken, the newest open ----
function modelScoresTab(m, kinds) {
  const taken = kinds.filter(k => k.taken);
  if (!taken.length) return [el('div', { class: 'card', 'data-scores-none': '1' },
    el('p', { class: 'small', text: 'No scores yet: this model has not taken a test.' }))];
  const newest = taken.reduce((a, b) => (b.at || 0) > (a.at || 0) ? b : a).kind;
  const mine = state.mblk[m.id] || {};
  return taken.map(k => {
    const open = k.kind in mine ? mine[k.kind] : k.kind === newest;
    const [v, sub] = kindValue(m, k.kind);
    // a closed block is built when it opens: the judged tables are the
    // page's heaviest, and most visits read one kind
    return el('details', { class: 'card kblock', 'data-kind-block': k.kind, open: open ? '' : null,
        ontoggle: e => {
          const now = e.target.open, blk = state.mblk[m.id] = state.mblk[m.id] || {};
          if (blk[k.kind] === now || (!(k.kind in blk) && now === open)) return;
          blk[k.kind] = now;
          if (now) { state.after = { focus: `[data-kind-block="${k.kind}"] > summary` }; render(); }
        } },
      el('summary', { class: 'kblock-sum' }, el('span', { class: 'kblock-k', text: k.label }),
        el('span', { class: 'kblock-v', text: v }),
        // 12b.3: without an average, the header says what the tile says
        k.kind === 'exam' && m.judgedAvg == null
          ? el('span', { class: 'kblock-sub small se', text: sub }) : ''),
      ...(open ? kindParts(m, k.kind) : []));
  });
}
// the sections a block holds: each keeps its code, not its card
function kindParts(m, kind) {
  const part = (node, id) => {
    if (!node) return null;
    node.classList.remove('card'); node.classList.add('kpart');
    if (id) node.id = 'sec-' + id;
    return node;
  };
  if (kind === 'standard') {
    const diag = part(vDiagnose(m), 'diagnose');
    return [el('p', { class: 'mprose', text: modelSentence(m) }),
      el('p', { class: 'small', 'data-avg-verdict': '1', text: avgVerdictOf(m)
        + ((m.missing || []).length ? ` Missing ${m.missing.join(', ')}.` : '') }),
      // 12h.1: the mode is part of the result
      m.gen ? el('p', { class: 'small', 'data-gen-mode': m.id,
        text: 'IFEval, MMLU-Pro and MATH-500: ' + genMode(m) + (() => {
          const gt = (m.gen || {}).tasks || {};
          const out = genTasks().filter(t => (gt[t] || {}).ran_out)
            .map(t => `${gt[t].ran_out} on ${LB_SHORT[t] || t}`);
          return out.length ? ` · answers that ran out of room: ${out.join(', ')}` : '';
        })() + '.' }) : '',
      part(resultsPart(m), 'results'),
      // item analysis of the benchmark, not the improvement loop (12b §7)
      diag ? el('details', { class: 'kfold', 'data-cant-show': '1' },
        el('summary', { text: 'What the score can’t show ▸' }), diag) : ''].filter(Boolean);
  }
  if (kind === 'exam') {
    const earlier = vEarlier(m), judged = part(vJudged(m, [earlier]), 'judged');
    // a card that stopped early (not judged on the current exam) still keeps
    // the earlier exams it is about
    if (judged && earlier && !judged.contains(earlier)) {
      earlier.classList.remove('card');
      judged.append(el('details', { class: 'kfold', 'data-more-detail': 'judged' },
        el('summary', { text: 'More detail ▸' }), earlier));
    }
    return [LIVE ? el('div', { class: 'kacts' }, sitCta(m)) : '', judged].filter(Boolean);
  }
  return [part(vEverydayBlock(m), 'everyday')].filter(Boolean);
}
// Results: every task this model has, grouped by domain
function resultsPart(m) {
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
  return el('div', { class: 'card' },
    el('h2', { text: 'Results' }),
    el('p', { class: 'sub', text: 'Dashed mark is chance; the solid mark, where one '
      + 'exists, is the best published score — a different protocol from ours, shown '
      + 'for orientation rather than comparison.' }),
    el('div', { class: 'lb-wrap' }, el('table', { class: 'lb mtbl' },
      el('thead', {}, el('tr', {}, el('th', { text: 'Benchmark' }),
        el('th', { class: 'num', text: 'Score' }), el('th', { text: '' }),
        el('th', { class: 'num', text: 'Shots' }), el('th', { class: 'num', text: 'Items' }))),
      el('tbody', {}, rows))));
}

// ---- Answers: what the model wrote, by kind, then topic or group ----
function modelAnswersTab(m, kinds) {
  const J = DATA.judged || {};
  const tasks = (m.judge || {}).tasks || {};
  // weakest first; a topic with no score keeps its answers readable (11l)
  const cats = Object.keys(tasks).filter(t => (J.exam || []).includes(t))
    .sort((a, b) => (pubScore(tasks[a]) ?? 9) - (pubScore(tasks[b]) ?? 9));
  const have = [cats.length ? ['exam', 'Knowledge exam'] : null,
    evdOf(m.id) ? ['everyday', 'Everyday tasks'] : null].filter(Boolean);
  const card = el('div', { class: 'card', 'data-model-answers': m.id },
    el('h2', { text: 'Answers' }),
    el('p', { class: 'sub', text: 'What the model wrote, on the questions anyone may read. '
      + 'The hidden questions stay hidden: their score is all you see of them.' }));
  if (!have.length) {
    card.append(el('p', { class: 'small', 'data-answers-none': '1', text: 'No written answers '
      + 'yet: this model has not sat the Knowledge exam or the Everyday tasks.' }));
    return [card];
  }
  const kind = have.some(([k]) => k === state.mans) ? state.mans : have[0][0];
  if (have.length > 1) card.append(el('div', { class: 'chiprow', 'data-answers-kinds': '1' },
    have.map(([k, label]) => el('button', { class: 'chip-btn' + (k === kind ? ' on' : ''),
      'data-answers-kind': k, 'aria-pressed': String(k === kind), text: label,
      onclick: () => { state.mans = k; render(); } }))));
  if (kind === 'exam') card.append(LIVE ? modelAnswers(m, cats)
    : el('p', { class: 'small', text: 'The answers are read from the live board; this report '
      + 'does not carry them.' }));
  else card.append(evdAnswersList(m));
  return [card];
}
// 12a.2: a model's everyday answers, one group at a time — the first it was
// asked opens; 333 answers at once is not a page anyone reads
function evdAnswersList(m) {
  const e = evdOf(m.id);
  const groups = evdGroups().filter(([g]) => ((e && e.groups) || {})[g]);
  const g = groups.some(([k]) => k === state.mansGroup) ? state.mansGroup
    : (groups[0] || [''])[0];
  return el('div', { 'data-panel': 'everyday-answers' },
    el('div', { class: 'chiprow' }, groups.map(([k, label]) =>
      el('button', { class: 'chip-btn' + (k === g ? ' on' : ''), 'data-answers-group': k,
        'aria-pressed': String(k === g), text: `${label} · ${evdGroupCount(e, k)}`,
        onclick: () => { state.mansGroup = k; render(); } }))),
    evdQs(g).filter(q => evdItem(e, q.id)).map(q => {
      const it = evdItem(e, q.id), mk = evdMark(it);
      return el('div', { class: 'evitem open', 'data-answers-q': q.id },
        el('div', { class: 'evrow evrow-static' },
          el('span', { class: 'evgroup', text: q.groupLabel }),
          el('span', { class: 'evmark ' + mk.cls, 'aria-label': mk.words, text: mk.t }),
          el('span', { class: 'evreason', text: it ? it.reason : 'not asked' })),
        evdAnswer(q, it));
    }));
}

// ---- Improve: this model's proposals and datasets, the Review lists ----
function modelImproveRows(m) {
  const props = (state.rv.proposals || []).filter(p => p.model === m.id);
  const ds = (state.rv.datasets || []).filter(d => d.model === m.id);
  return { props, ds, n: props.length + ds.length };
}
function modelImproveTab(m) {
  // 12g.2: the same four stages Improve shows, for this model
  return [impStagesCard(m)];
}
// ---- History: the runs, what produced the numbers, how they were graded ----
function modelHistoryTab(m) {
  return [LIVE ? vModelRuns(m) : null, judgedEarlierCard(m), evdEarlierCard(m), provRecord(m),
    gradedCard(m), vTaint(m)].filter(Boolean);
}
// 12i.1: Knowledge exam scores an earlier judge marked — a score compares only
// with the same judge's, so they are kept here, said once, and nowhere else
function judgedEarlierCard(m) {
  const x = m.judgedEarlier;
  if (!x) return null;
  const when = x.at ? new Date(x.at * 1000).toISOString().slice(0, 10) : '';
  return el('div', { class: 'card', 'data-judged-earlier': m.id },
    el('div', { class: 'sechead' }, el('h2', { text: 'Knowledge exam' }),
      el('span', { class: 'badge', 'data-judged-by': x.by, text: 'judged by ' + x.by })),
    el('p', { class: 'sub', text: (x.avg != null ? `${x.avg.toFixed(2)} of 4 on average, over `
        : 'Over ') + `${x.topics} topic${x.topics === 1 ? '' : 's'}${when ? ', judged ' + when : ''}. `
      + 'A score compares only with scores from the same judge, and this server runs another one '
      + 'now, so these are in no table and no comparison until they are judged again.' }));
}
// 12a.4: its everyday answers to an earlier wording — kept, and said once, as
// what they were; never in a score or beside this wording's answers
function evdEarlierCard(m) {
  const x = evdEarlier(m.id);
  if (!x) return null;
  const when = x.marked_at ? new Date(x.marked_at * 1000).toISOString().slice(0, 10) : '';
  // 12g.2: marked on every question, before the bank was split — kept, and
  // never beside a hidden-half score
  const split = x.label === 'all questions, before the split';
  return el('div', { class: 'card', 'data-evd-earlier': m.id },
    el('div', { class: 'sechead' }, el('h2', { text: 'Everyday tasks' }),
      el('span', { class: 'badge', 'data-earlier-badge': '1',
        text: split ? x.label : 'earlier wording' })),
    el('p', { class: 'sub', 'data-evd-earlier-count': `${x.passed} of ${x.total}`,
      text: `${x.passed} of ${x.total}${when ? ', marked ' + when : ''}. ` + (split
        ? 'These were marked on all the questions, before the bank was split into a hidden half '
          + 'that scores and a practice half that is shown, so they are in no score and no '
          + 'comparison. Run everyday tasks to be scored on the hidden half.'
        : 'The questions have been reworded since, so these answers are in no score and no '
          + 'comparison. Run everyday tasks to answer this wording.') }));
}
// an everyday run that answered another wording than the bank's current one
const evdEarlierRun = r => r.suite === 'everyday' && r.status === 'done'
  && (r.bank_version || '') !== ((evd().version || {}).hash || '');
// the model's row of the old Run provenance table, and its hero's small print
function provRecord(m) {
  const a = m.archinfo || {}, comp = computeOf(m);
  const params = m.params ? P(m.params)
    + (a.active_params ? ` · ${P(a.active_params)} active, ${a.experts} experts, `
      + `${a.experts_per_tok} per token (${a.active_src})` : '')
    + (m.paramsSrc ? ` · from the ${m.paramsSrc === 'config' ? 'harness config' : 'model name'}`
      : '') : null;
  const prov = [
    ['hub id', m.id], ['parameters', params],
    ['training compute', comp ? `${flop(comp.c)} FLOP (6ND, run ${comp.run.name})` : null],
    ['architecture', a.arch], ['shape', a.hidden ? `hidden ${a.hidden} · layers ${a.layers}`
      + ` · heads ${a.heads} · ctx ${a.ctx}` : null],
    ['vocab', a.vocab], ['backend', m.backend], ['dtype', m.dtype],
    ['weights stored as', a.stored_dtype], ['batch size', m.batch],
    ['chat template', m.chat ? `applied${a.tmpl_sha ? ' · ' + a.tmpl_sha : ''}` : 'none'],
    ['kind decided by', m.kindReason], ['seed', m.seed],
    ['limit', m.limit == null ? 'full' : m.limit],
    ['model code', (a.code_sha || []).join(', ')],
    ['harness', m.hash],
    ['eval wall clock', m.minutes != null ? m.minutes + ' min' : null],
    ['last evaluated', lastEval(m)],
  ].filter(([, v]) => v != null && v !== '' && v !== false);
  return el('div', { class: 'card', 'data-model-prov': m.id },
    el('h2', { text: 'Run provenance' }),
    el('p', { class: 'sub', text: 'What produced these numbers. Two runs whose '
      + 'template id or harness differ are not comparable, whatever the scores say.' }),
    el('dl', { class: 'provlist' }, prov.flatMap(([k, v]) =>
      [el('dt', { text: k }), el('dd', { text: String(v) })])));
}
// how each kind was graded, in a line each
function gradedCard(m) {
  const rows = [];
  if ([...DATA.accTasks, ...DATA.pplTasks].some(t => cell(t, m.id)))
    rows.push(['Standard', 'lm-evaluation-harness'
      + (m.hash ? ` ${m.hash}` : '') + ': each task’s own metric, at the shots its row says.']);
  const j = m.judge;
  if (j && j.judge) rows.push(['Knowledge exam', judgeIdLine(m),
    LIVE ? el('span', { 'data-how-graded': m.id },
      readLink({ kind: 'provenance', id: 'judge:' + m.id }, 'How this was graded ▸')) : '']);
  const e = evdOf(m.id);
  if (e) rows.push(['Everyday tasks', 'Four answers are checked by a script; the TL;DR is '
    + 'marked by the judge, against its rubric'
    + (e.provisional ? ', and its marks are not evidence yet.' : '.')]);
  if (!rows.length) return null;
  return el('div', { class: 'card', 'data-model-graded': m.id },
    el('h2', { text: 'How it was graded' }),
    el('dl', { class: 'provlist' }, rows.flatMap(([k, v, x]) =>
      [el('dt', { text: k }), el('dd', {}, v, x ? [' ', x] : '')])));
}

// 11i: the model page is where a person is thinking about this model, so the
// exam starts here — beside the three cards, with how far it has got
function sitCta(m) {
  if (!LIVE) return '';
  const J = DATA.judged || {};
  const n = Object.entries((m.judge || {}).tasks || {})
    .filter(([t, v]) => (J.exam || []).includes(t) && pubScore(v) != null).length;
  return el('div', { class: 'msit-cta' },
    el('button', { class: 'secondary', 'data-sit-open': m.id, text: 'Sit the exam',
      'aria-expanded': String(state.msit.open && state.msit.model === m.id),
      onclick: () => openSit(m.id) }),
    el('span', { class: 'small se', 'data-sit-progress': String(n),
      text: `${n} of ${(J.exam || []).length} topics judged` }));
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
      el('td', { class: 'small se' }, r.error || r.progress || '',
        evdEarlierRun(r) ? el('span', { class: 'badge', 'data-earlier-run': String(r.id),
          text: 'earlier wording' }) : ''),
      el('td', {}, el('a', { href: `api/runs/${r.id}/log`, target: '_blank', rel: 'noopener',
        class: 'small', text: 'log' }))))))));
  return card;
}

function vOverview(ms) {
  // 12b: three blocks, in this order — what needs a person, what is running,
  // and the best in each kind of test. No hero paragraph and no stats line
  if (LIVE && !NET.fails) { rvNeeded(); if (!state.trLoaded) loadTruns(); }
  return [needsYou(), LIVE ? runningNow() : null, bestByKind(ms)].filter(Boolean);
}
const onHome = () => state.tab === 'overview' && !state.model && !state.topic;
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

// one line per thing waiting, each a link to where it is dealt with
function needsYou() {
  const lines = [];
  const add = (key, n, text, go) => { if (n) lines.push(el('li', { 'data-needs': key },
    el('a', { href: '#', text, onclick: e => { e.preventDefault(); go(); } }))); };
  if (LIVE && state.rv.loaded) {
    const L = rvLists();
    add('proposals', L.review.length,
      plural(L.review.length, 'proposal waiting for review', 'proposals waiting for review'),
      () => openImprove((L.review[0] || {}).model));
    if (state.trLoaded) {
      const used = new Set((state.trRuns || []).flatMap(r => r.datasets || []));
      // 12b.3: a Demo only dataset is for trying the loop, not for training on —
      // training happens off the board, and it would wait here for ever
      const idle = (state.rv.datasets || []).filter(d => d.status === 'ready'
        && !used.has(d.id) && !dsDemoOnly(d));
      add('datasets', idle.length, plural(idle.length, 'dataset made but not used in training',
        'datasets made but not used in training'),
        () => openImprove((idle[0] || {}).model));
    }
  }
  if (LIVE) {
    const since = Date.now() / 1000 - 7 * 86400;
    const failed = (state.queue || []).filter(r => r.status === 'failed'
      && (r.finished_at || r.created_at || 0) >= since);
    add('failed', failed.length, plural(failed.length, 'run failed in the last seven days',
      'runs failed in the last seven days'), () => navigate({ tab: 'queue', model: null, topic: null }));
  }
  // 12b.3: the problems, not the known limits — those stay true for weeks
  const checks = DATA.checks ? boardProblems(DATA.checks).length : (DATA.warnings || []).length;
  add('checks', checks, plural(checks, 'check is not green', 'checks are not green'), openChecks);
  return el('div', { class: 'card', 'data-needs-you': String(lines.length) },
    el('h2', { text: 'Needs you' }),
    lines.length ? el('ul', { class: 'needs' }, lines)
      : el('p', { class: 'small', 'data-needs-none': '1', text: 'Nothing needs you.' }));
}
// the status dot's list, opened from Home's line about it
function openChecks() {
  const b = document.querySelector('#warnings [data-pop-anchor="checks"]');
  if (b && POP.key !== 'checks') b.click();
}

// the run counter's list, full width
function runningNow() {
  const { running, queued } = runsNow();
  const card = el('div', { class: 'card', 'data-running-now': String(running.length + queued.length) },
    el('h2', { text: 'Running now' }));
  if (!running.length && !queued.length) card.append(el('p', { class: 'small', 'data-running-none': '1' },
    'Nothing running · ', el('a', { href: '#', 'data-running-test': '1', text: 'Test a model',
      onclick: e => { e.preventDefault(); openTest(); } })));
  else card.append(el('div', { class: 'runsfull' }, runsList()));
  return card;
}

// one card per kind of test that has data: the best model's number, its name,
// and one link to the kind on Models. A kind with no data has no card
function bestByKind(ms) {
  // comparing models side by side is 12h; this link goes to the table
  const go = v => hlLink('See all models →', () => openModelsView(v));
  // 12a.4: a card's badge is a line of its own under the heading, not a
  // second line of it
  const card = (key, eyebrow, value, m, badge) => el('div', { class: 'hcard', 'data-best': key },
    el('div', { class: 'eyebrow', text: eyebrow }),
    badge ? el('div', { class: 'hcard-badge', 'data-best-badge': key }, badge) : '',
    el('div', { class: 'hcard-v', 'data-best-value': key, text: value }),
    el('div', { class: 'hcard-name', 'data-best-name': key, title: m.id, text: m.name }),
    go(key));
  const cards = [];
  const std = ms.filter(m => officialAvg(m) != null && !m.duplicateOf)
    .sort((a, b) => officialAvg(b) - officialAvg(a))[0];
  if (std) cards.push(card('standard', 'Standard · '
    + (state.avgMode === 'raw' ? 'raw accuracy' : 'above chance'),
    (100 * officialAvg(std)).toFixed(1), std));
  const exam = ms.filter(m => m.judgedAvg != null && !m.duplicateOf)
    .sort((a, b) => b.judgedAvg - a.judgedAvg)[0];
  if (exam) cards.push(card('exam', 'Knowledge exam', `${num(exam.judgedAvg, 2)} / 4`, exam));
  // 12b.3: no model has a judged average (none counts yet) but models have
  // judged topics. 12g.1: the weakest topic OF THE MODEL WITH THE MOST judged
  // topics — the weakest across the board was a 31M model's 0 / 4, which a
  // model that scores zero everywhere says about every topic
  let weak = null;
  if (!exam) {
    const most = ms.filter(x => !x.duplicateOf && judgedTopics(x).length)
      .sort((a, b) => judgedTopics(b).length - judgedTopics(a).length || natCmp(a.name, b.name))[0];
    const w = most && weakestTopic(most);
    if (w) weak = { ...w, m: most };
  }
  if (weak) cards.push(el('div', { class: 'hcard', 'data-best': 'exam', 'data-best-weakest': '1' },
    el('div', { class: 'eyebrow', text: 'Knowledge exam · weakest topic' }),
    el('div', { class: 'hcard-v', 'data-best-value': 'exam', text: `${num(weak.v, 2)} / 4` }),
    el('div', { class: 'hcard-name', 'data-best-name': 'exam', title: weak.m.id,
      text: `${weak.m.name} · weakest: ${frName(weak.task)} ${num(weak.v, 2)} / 4` }),
    LIVE ? hlLink('Improve it →', () => openImprove(weak.m.id)) : go('exam')));
  const E = evd();
  // 12i.0: counts over the whole set first — a partial one is never ranked beside them
  const ev = Object.entries(E.models || {}).filter(([id]) => ms.some(m => m.id === id))
    .sort(([a, x], [b, y]) => (!!evdMissing(x) - !!evdMissing(y)) || (y.passed - x.passed)
      || evdName(a).localeCompare(evdName(b)))[0];
  if (ev) cards.push(card('everyday', 'Everyday tasks', evdCount(ev[1]),
    DATA.models.find(m => m.id === ev[0]), evdBadge(ev[1].provisional)));
  // the provisional-judge caveat, once, in the block's header
  const caveat = (exam && !(judgedCalibrated() && judgedOkM(exam))) || (weak && !judgedOkM(weak.m))
    ? el('span', { class: 'badge prelim', 'data-best-caveat': '1',
        title: judgedCalibrated() ? whyProvisional(exam || weak.m) : judgedOffWhy(),
        text: 'Knowledge exam: provisional judge' }) : exam ? judgeChecked() : '';
  return el('div', { class: 'card', 'data-best-by-kind': String(cards.length) },
    el('div', { class: 'sechead' }, el('h2', { text: 'Best in each kind of test' }), caveat),
    cards.length ? el('div', { class: 'hlgrid' }, cards)
      : el('p', { class: 'small', text: 'No model has a score yet.' }));
}
// Models, on one of its three views
function openModelsView(v) {
  const L = lbS();
  try { localStorage.setItem('bench-models-view', v); } catch (e) { /* private */ }
  Object.assign(L, { view: v, chip: v === 'exam' ? 'judged' : (L.stdChip || 'all') });
  navigate({ tab: 'leaderboard', model: null, topic: null });
}
// the training runs' list alone — Home asks which datasets no run has used
async function loadTruns() {
  if (!LIVE || state.trFetching || !netReady()) return;
  state.trFetching = true;
  try {
    const runs = await api('api/truns');
    const changed = !state.trLoaded || JSON.stringify(runs) !== JSON.stringify(state.trRuns);
    state.trRuns = runs; state.trLoaded = true;
    if (changed && onHome()) render();
  } catch (e) { /* next poll retries */ }
  finally { state.trFetching = false; }
}

// "30 fixed scripts re-graded …" — Home's old Judge steadiness card, as a
// line under the checks behind the status dot (12b §6)
function judgeSteadiness() {
  const m = loopModel();
  const cn = m && (m.judge || {}).canary;
  if (!cn) return null;
  const cal = (DATA.judged || {}).calibration;
  const r2 = x => x == null ? '—' : (+x).toFixed(2);
  return `Judge steadiness: ${cn.drifted ? 'moved' : 'steady'} — ${cn.n} fixed scripts `
    + `re-graded: ${r2(cn.mad_vs_human)} from the human marks`
    + (cn.mad_vs_previous != null ? `, ${r2(cn.mad_vs_previous)} from the last run `
       + `(limit ${cn.threshold})` : ', the first run for this judge') + '. '
    + (cal && cal.calibrated ? `Agreement with a person: ${cal.kappa}.`
       : 'No person has checked the judge yet.');
}
// a card's one link (the hcard component, 11d): Home's Best in each kind
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

// ===========================================================================
// 12a: Everyday tasks — what people type into an assistant on a phone,
// marked by checks (scripts/everyday.py) on the text after any thinking.
// 12a.2: one bank in seven groups, the pilot's five in it; 12a.3: round 3
// grew it to 333; 12a.5 to 388 in eight groups. Still a look, not a benchmark: every question readable,
// never ranked, never averaged into anything, read by nothing that proposes
// or generates. One badge wherever it is shown: not ranked (12a.4: one short
// line; the wording's version is on the Everyday tab, not in the badge).
// 12a.4: only answers to the bank's current wording are shown or compared;
// a model's answers to an earlier wording are in its History
// ===========================================================================
// the five instruct models on the board; Run everyday tasks ticks them
const EVD_DEFAULTS = ['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-0.6B', 'HuggingFaceTB/SmolLM2-360M-Instruct',
                      'HuggingFaceTB/SmolLM2-135M-Instruct', 'google/gemma-3-270m-it'];
const evd = () => DATA.everyday || { groups: [], questions: [], models: {} };
// 12g.2: the split — how many questions are hidden (they score, and are never
// shown) and practice (shown), in a group or in all
const evdHidden = g => g ? ((evd().hidden || {})[g] || 0)
  : Object.values(evd().hidden || {}).reduce((a, b) => a + b, 0);
const evdPractice = g => g ? ((evd().practice || {})[g] || 0)
  : Object.values(evd().practice || {}).reduce((a, b) => a + b, 0);
// every question a run asks: both halves (the page shows the practice half)
const evdAll = () => evdHidden() + evdPractice() || (evd().questions || []).length;
// the groups, in their order, and a group's questions — "eight groups"
const evdGroups = () => evd().groups || [];
const evdGroupsWord = () => ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten'][evdGroups().length] || String(evdGroups().length);
const evdQs = g => (evd().questions || []).filter(q => q.group === g);
// "12 of 16": what this model passed of what it was asked in the group
function evdGroupCount(e, g) {
  const x = ((e && e.groups) || {})[g];
  return x ? `${x.passed} of ${x.total}` : null;
}
const evdOf = id => (evd().models || {})[id] || null;
// 12a.4: a model's answers to an earlier wording — a count and a date, never
// a score here
const evdEarlier = id => (evd().earlier || {})[id] || null;
// "3 answers ran out of room", beside a score, only when there are any: the
// thinking used the whole budget before the answer; those answers fail
function evdRanOut(e) {
  const n = (e && e.ran_out) || 0;
  return n ? el('span', { class: 'small se evd-ranout', 'data-evd-ran-out': String(n),
    title: 'the model was still thinking when it reached its answer budget; these answers '
      + 'count as failed', text: `${n} answer${n === 1 ? '' : 's'} ran out of room` }) : '';
}
const evdName = id => (DATA.models.find(x => x.id === id) || {}).name || String(id).split('/').pop();
const evdCount = e => `${e.passed} of ${e.total}`;
// 12i.0: a model that hasn't answered the whole current hidden set is scored
// over fewer questions — "84 of 169" — so it is never shown as if beside a
// full count: greyed, with how many are not asked yet, and its Run
function evdMissing(e) {
  if (!e || (e.total || 0) >= evdHidden()) return 0;
  return e.unasked > 0 ? e.unasked : evdHidden() - (e.total || 0);
}
function evdTotal(e, id, attrs = {}) {
  const miss = evdMissing(e);
  const count = el('span', { 'data-everyday-count': evdCount(e), ...attrs,
    class: [attrs.class, miss ? 'se' : ''].filter(Boolean).join(' ') || null,
    title: miss ? `over the ${e.total} hidden questions it has answered, not all `
      + `${evdHidden()} — not comparable with a full count` : null, text: evdCount(e) });
  if (!miss) return count;
  // in a narrow column head the note takes its own line (.evm-count .evmissing)
  return el('span', { class: 'evpartial', 'data-evd-partial': String(miss) }, count,
    el('span', { class: 'small se evsep', text: ' · ' }),
    el('span', { class: 'evmissing' },
      el('span', { class: 'small se', text: `${miss} not asked yet` }),
      LIVE && id ? [el('span', { class: 'small se', text: ' · ' }),
        el('a', { href: '#', class: 'small', 'data-evd-run': id, text: 'Run',
          onclick: ev => { ev.preventDefault(); ev.stopPropagation();
            evdDialog({ only: id, returnTo: `[data-evd-run="${CSS.escape(id)}"]` }); } })] : ''));
}
// "2026-09-25" -> "25 Sep"
const evdDay = d => { const [y, mo, da] = String(d || '').split('-').map(Number);
  return y ? `${da} ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
    'Nov', 'Dec'][mo - 1]}` : String(d || ''); };
function evdBadge(provisional) {
  return el('span', { class: 'badge prelim', 'data-pilot-badge': '1',
    title: `${evdHidden()} hidden questions score it and ${evdPractice()} practice ones are shown: `
      + 'a look at what the models say, not a ranking. Never ranked, never averaged into anything.'
      + (provisional
        ? ' Some answers were marked by a judge whose marks are not evidence yet.' : ''),
    text: 'not ranked' + (provisional ? ' · provisional judge' : '') });
}
// ✓, ✗, or a question still with the judge
function evdMark(it) {
  if (!it) return { t: '·', cls: 'wait', words: 'not asked' };
  if (it.pass === true) return { t: '✓', cls: 'ok', words: 'passed' };
  if (it.pass === false) return { t: '✗', cls: 'no', words: 'failed' };
  return { t: '…', cls: 'wait', words: 'not marked yet' };
}
const evdItem = (e, qid) => ((e && e.items) || []).find(x => x.id === qid) || null;

// the question, the answer (with its mark and reason, in the side panel),
// and the thinking folded away
function evdAnswer(q, it, attrs = {}, verdict = null) {
  return el('div', { class: 'evans-wrap', ...attrs },
    el('p', { class: 'evq-label', text: 'Question' }),
    el('blockquote', { class: 'evq', 'data-evd-question': q.id, text: q.prompt }),
    el('p', { class: 'evq-label', text: 'Answer' }),
    verdict || '',
    // 12a.2: a model that sat the pilot's five was never asked the rest
    !it ? el('p', { class: 'evans none', 'data-evd-answer': q.id, 'data-not-asked': '1',
          text: 'Not asked: this model\u2019s run did not include this question.' })
    : it.no_answer
      ? el('p', { class: 'evans none', 'data-evd-answer': q.id, 'data-no-answer': '1',
          text: 'No answer: the model was still thinking when it ran out of room.' })
      : el('div', { class: 'evans', 'data-evd-answer': q.id,
          text: (it && it.answer_text) || '(the model wrote nothing)' }),
    it && it.had_reasoning ? el('details', { class: 'evthink', 'data-evd-thinking': q.id },
      el('summary', { text: `thinking ▸ ${(it.reasoning_words || 0).toLocaleString('en')} words` }),
      el('div', { class: 'evthink-t', text: it.reasoning_text || '' })) : '');
}

// queue everyday tasks for these models, one run each; returns [{id, ok, sid, why}]
async function evdQueue(ids) {
  const out = [];
  for (const id of ids) {
    try {
      const r = await fetch('api/submissions', { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
        body: JSON.stringify({ hf_id: id, kind: 'auto', suite: 'everyday',
          submitter: whoName(), note: 'everyday tasks' }) });
      const j = await r.json().catch(() => ({}));
      if (r.ok) { rememberQueued(j.id); out.push({ id, ok: true, sid: j.id, note: j.note }); }
      else out.push({ id, ok: false, why: typeof j.detail === 'string' ? j.detail
        : `the server answered HTTP ${r.status}` });
    } catch (e) { out.push({ id, ok: false, why: 'the server is unreachable' }); }
  }
  loadQueue();
  return out;
}

// a model's answers in one group, one row per question — the mark, the
// question and the reason — so a group reads from top to bottom. A row opens
// the whole answer in the side panel
function evdGroupList(id, g) {
  const e = evdOf(id);
  const qs = evdQs(g).filter(q => evdItem(e, q.id));
  const pr = ((e && e.practice) || {})[g];
  return el('div', { class: 'evglist', 'data-evd-answers': `${id}|${g}` },
    // 12g.2: the score above is the hidden half's; these are the practice half
    el('p', { class: 'small se evsplit', 'data-evd-split-note': g },
      `Practice questions${pr ? ` · ${pr.passed} of ${pr.total} passed` : ''}. The `
      + `${evdHidden(g)} hidden ones score it and are not shown.`),
    ...qs.map(q => {
    const it = evdItem(e, q.id), mk = evdMark(it);
    const r = { kind: 'everyday', id, n: q.n }, key = readStr(r);
    return el('button', { class: 'evrow evqrow', 'data-evd-row': q.id, 'data-read-open': key,
        title: 'read the whole answer',
        onclick: () => openReader(r, `[data-read-open="${CSS.escape(key)}"]`) },
      el('span', { class: 'evmark ' + mk.cls, 'data-evd-mark': mk.cls, 'aria-label': mk.words,
        text: mk.t }),
      el('span', { class: 'evprompt', text: q.prompt }),
      el('span', { class: 'evreason', text: it.reason }));
  }));
}

// ---- on the model page -------------------------------------------------------
function vEverydayBlock(m) {
  const qs = evd().questions || [];
  if (!qs.length) return null;
  const e = evdOf(m.id);
  // 12b: an untested model is its Everyday tile's "Not tested · Test"
  if (!e) return null;
  // 12a.2: the groups, n of k each; a group opens to its answers
  const open = state.evdOpen || (state.evdOpen = {});
  const rows = evdGroups().filter(([g]) => (e.groups || {})[g]).map(([g, label]) => {
    const key = m.id + '|' + g, on = !!open[key];
    return el('div', { class: 'evitem' + (on ? ' open' : '') },
      el('button', { class: 'evrow evgrow', 'data-evd-group': g, 'aria-expanded': String(on),
          onclick: () => { open[key] = !on; render(); } },
        el('span', { class: 'evgroup', text: label }),
        el('span', { class: 'evgcount', 'data-evd-group-count': g, text: evdGroupCount(e, g) })),
      on ? evdGroupList(m.id, g) : '');
  });
  // numbered with the page's other sections (the h2 sits in a .sechead)
  return el('div', { class: 'card', 'data-everyday-block': m.id },
    el('div', { class: 'sechead evhead' },
      el('h2', {}, 'Everyday tasks', evdBadge(e.provisional)),
      el('div', { class: 'acts evscore' },
        evdTotal(e, m.id, { class: 'evcount' }),
        e.waiting ? el('span', { class: 'small se', text: `${e.waiting} with the judge` }) : '',
        evdRanOut(e),
        el('a', { href: '#tab=benchmarks&sub=everyday', 'data-everyday-compare': '1',
          text: 'Compare models →',
          onclick: ev => { ev.preventDefault();
            navigate({ tab: 'everyday', model: null, topic: null }); } }))),
    // 12a.5: what the last marking was, in one line — "55 new questions · 333
    // re-marked", and the questions this model has not been asked yet
    e.marking ? el('p', { class: 'small se', 'data-evd-marking': m.id, text: e.marking }) : '',
    el('p', { class: 'sub', text: `Questions typed the way people type on a phone, in `
      + `${evdGroupsWord()} groups, each marked by its checks. The score is the ${evdHidden()} hidden questions; `
      + `open a group to read the answers to its practice ones.` }),
    el('div', { class: 'evrows' }, rows));
}

// the pilot's one-click queue, for one model
function evdTestBtn(m, cls = 'ghost') {
  return el('button', { class: cls, 'data-everyday-test': m.id, text: 'Test',
    onclick: async ev => {
      const b = ev.currentTarget; b.disabled = true; b.textContent = 'Queueing…';
      const [r] = await evdQueue([m.id]);
      state.evdMsg = { ...(state.evdMsg || {}), [m.id]: r.ok ? '' : 'Refused. ' + r.why };
      if (r.ok) toast(r.note ? `#${r.sid}: ${r.note} —` : `Run #${r.sid} queued —`,
        { key: 'submit', go: () => followRun(r.sid), link: 'follow it →' });
      render(); } });
}

// ---- #everyday: the models side by side ---------------------------------------
function vEverydayPage() {
  const E = evd(), qs = E.questions || [], groups = evdGroups();
  const ids = Object.keys(E.models || {}).sort((a, b) => evdName(a).localeCompare(evdName(b)));
  const prov = ids.some(id => E.models[id].provisional);
  const run = LIVE ? el('button', { class: 'primary', 'data-everyday-run': '1',
    text: 'Run everyday tasks', onclick: () => evdDialog({ returnTo: '[data-everyday-run]' }) }) : '';
  const head = el('div', { class: 'card', 'data-everyday-head': '1' },
    el('div', { class: 'rvbar' },
      el('div', {}, el('h2', {}, 'Everyday tasks', evdBadge(prov)),
        el('p', { class: 'sub', text: `${evdHidden() + evdPractice()} questions people type `
          + 'into an assistant on a phone — lowercase, typos, one plain request — in '
          + `${evdGroupsWord()} groups. Each model is scored on the ${evdHidden()} hidden ones, which are never `
          + `shown; the ${evdPractice()} practice ones are here, with its answers. Click a `
          + 'group\u2019s count to read them.' }),
        // 12a.4: the wording these answers are to — 12g.2: and the split;
        // earlier ones are in each model's History
        // 12i.0: in words; the hash, and what it covers, on hover
        E.version ? el('p', { class: 'small se', 'data-evd-version': E.version.hash,
          title: `version ${E.version.hash}: the wording and the split. Answers from before it `
            + 'are in each model\u2019s History, and in no score here.',
          text: `Questions updated ${evdDay(E.version.date)}` }) : ''),
      run));
  if (!ids.length) {
    return [head, el('div', { class: 'card' }, empty('No model has taken everyday tasks yet.',
      LIVE ? 'Run everyday tasks' : '', () => evdDialog({ returnTo: '[data-empty-action]' }),
      { 'data-everyday-empty': '1' })), evdBankCard()];
  }
  // 12a.2: models across the top, the groups down the side, n of k
  const sel = state.evdCell;
  const cell = (id, g) => {
    const n = evdGroupCount(E.models[id], g);
    if (!n) return el('td', { class: 'evcell-td se', title: 'not asked', text: '—' });
    const on = !!(sel && sel.id === id && sel.g === g);
    // 12i.0: over fewer questions than the others: greyed, as its total is
    return el('td', { class: 'evcell-td' + (evdMissing(E.models[id]) ? ' evpart' : '') },
      el('button', { class: 'evcell' + (on ? ' on' : ''),
      'data-evd-cell': `${id}|${g}`, 'aria-pressed': String(on),
      'aria-label': `${evdName(id)}, ${(groups.find(x => x[0] === g) || [, g])[1]}: ${n}`,
      text: n, onclick: () => { state.evdCell = on ? null : { id, g }; render(); } }));
  };
  const table = el('table', { class: 'evtable', 'data-everyday-table': '1' },
    el('thead', {}, el('tr', {}, el('th', { class: 'evq-th', text: 'Group' }),
      ids.map(id => el('th', { class: 'evm-th', 'data-evd-model': id },
        DATA.models.some(x => x.id === id)
          ? el('a', { href: '#model=' + encodeURIComponent(id), text: evdName(id),
              onclick: ev => { ev.preventDefault(); navigate({ model: id, topic: null }); } })
          : el('span', { text: evdName(id) }),
        el('span', { class: 'evm-count', 'data-evd-count': id }, evdTotal(E.models[id], id)),
        evdRanOut(E.models[id]))))),
    el('tbody', {}, groups.map(([g, label]) => el('tr', { 'data-evd-g': g },
      el('th', { scope: 'row', class: 'evq-cell' },
        el('span', { class: 'evq-short', text: label }),
        el('span', { class: 'evq-group', 'data-evd-group-split': g,
          text: `${evdHidden(g)} hidden · ${evdPractice(g)} practice` })),
      ids.map(id => cell(id, g))))));
  const panel = sel && (((E.models[sel.id] || {}).groups) || {})[sel.g]
    ? el('div', { class: 'evpanel', 'data-evd-panel': `${sel.id}|${sel.g}` },
        el('div', { class: 'sechead' },
          el('h3', { text: `${evdName(sel.id)} · ${(groups.find(x => x[0] === sel.g) || [, sel.g])[1]}` }),
          el('span', { class: 'evgcount', text: evdGroupCount(E.models[sel.id], sel.g) }),
          el('button', { class: 'quiet', 'data-evd-panel-close': '1', text: '✕',
            'aria-label': 'close', onclick: () => { state.evdCell = null; render(); } })),
        evdGroupList(sel.id, sel.g))
    : '';
  return [head, el('div', { class: 'card', 'data-everyday-results': '1' },
    el('div', { class: 'lb-wrap', 'data-hkeep': 'everyday' }, table), panel), evdBankCard()];
}

// the bank itself, by group: every question readable, with its checks in
// plain words — what it takes to pass
function evdBankCard() {
  const E = evd();
  // its heading sits in a div, as the page head's does: the page's three
  // cards are not steps, and this one alone took a section number
  return el('div', { class: 'card', 'data-everyday-bank': '1' },
    // 12i.2: new questions for a group, or a new group
    LIVE ? el('div', { class: 'qbopen' }, el('button', { class: 'secondary', 'data-qb-open': 'everyday',
      text: 'Build questions', onclick: () => openBuilder('everyday') })) : '',
    el('div', {}, el('h2', { text: 'The practice questions' }),
      el('p', { class: 'sub', 'data-evd-bank-split': `${evdHidden()}|${evdPractice()}`,
        text: `${(E.questions || []).length} practice questions in ${evdGroupsWord()} groups; `
        + `${evdHidden()} more are hidden — they score the models, and are never shown. `
        + 'An answer passes when every one of its checks does; where the judge is one of them, '
        + 'it only decides once the others have passed.' })),
    evdGroups().map(([g, label]) => {
      const qs = evdQs(g);
      return el('details', { class: 'evbank-g', 'data-evd-bank-group': g },
        el('summary', {}, el('span', { class: 'evgroup', text: label }),
          el('span', { class: 'small se', text: ` · ${qs.length} practice · `
            + `${evdHidden(g)} hidden` })),
        el('ol', { class: 'evbank' }, qs.map(q => el('li', { 'data-evd-bank-q': q.id },
          el('p', { class: 'evq', text: q.prompt }),
          el('p', { class: 'small', 'data-evd-checks': q.id, text: 'Passes if it: '
            + q.checks.join(' · ') }),
          q.reference ? el('p', { class: 'small se', text: 'A good answer: ' + q.reference }) : ''))));
    }));
}

// the side panel: the question, the answer and its reason; ↑↓ move along the
// questions, ←→ along the models
function readEveryday(wrap, r) {
  const E = evd(), qs = E.questions || [];
  const ids = Object.keys(E.models || {}).sort((a, b) => evdName(a).localeCompare(evdName(b)));
  const paint = rr => {
    const q = qs.find(x => x.n === rr.n) || qs[0];
    const e = E.models[rr.id] || null, it = evdItem(e, q && q.id), mk = evdMark(it);
    if (!q) return;
    wrap._title.textContent = `${evdName(rr.id)} · ${q.label}`;
    wrap._src.textContent = `${q.groupLabel} · question ${q.n} of ${qs.length}`;
    wrap._acts.replaceChildren();
    const i = ids.indexOf(rr.id);
    const go = (id, n) => openReader({ kind: 'everyday', id, n }, state.readFrom);
    const step = (label, attr, ok, fn) => el('button', { class: 'ghost', [attr]: '1', text: label,
      disabled: ok ? null : '', onclick: fn });
    // the step pressed is rebuilt with the rest: focus goes back to it, or to
    // the panel, so the keys (and Esc) keep working
    const was = document.activeElement;
    const had = wrap._aside.contains(was) || was === document.body;
    const stepAttr = was && was.attributes
      ? [...was.attributes].map(a => a.name).find(n => n.startsWith('data-evd-')) : null;
    wrap._body.replaceChildren(
      el('div', { class: 'evnav', 'data-evd-nav': '1' },
        step('↑ Question', 'data-evd-prev-q', q.n > 1, () => go(rr.id, q.n - 1)),
        step('↓ Question', 'data-evd-next-q', q.n < qs.length, () => go(rr.id, q.n + 1)),
        step('← Model', 'data-evd-prev-m', i > 0, () => go(ids[i - 1], q.n)),
        step('→ Model', 'data-evd-next-m', i >= 0 && i < ids.length - 1, () => go(ids[i + 1], q.n))),
      evdAnswer(q, it, { 'data-evd-read': `${rr.id}|${q.id}` },
        // a question it was not asked has nothing to mark: the answer says so
        !it ? null
        // 12a.5: a failed answer lists each check it failed — why, and what
        // the check looks for, in the question list's words
        : it.pass === false && (it.failed || []).length
          ? el('div', { class: 'evverdicts', 'data-evd-verdict': mk.cls },
              it.failed.map(f => el('p', { class: 'evverdict ' + mk.cls, 'data-evd-fail': '1' },
                el('span', { class: 'evmark ' + mk.cls, text: mk.t }),
                el('span', {}, el('span', { 'data-evd-fail-why': '1', text: f.why }),
                  el('span', { class: 'evfail-check', 'data-evd-fail-check': '1',
                    text: 'The check: ' + f.check })))))
          : el('p', { class: 'evverdict ' + mk.cls, 'data-evd-verdict': mk.cls },
              el('span', { class: 'evmark ' + mk.cls, text: mk.t }),
              el('span', { text: it.reason }))));
    if (had) {
      const again = stepAttr && wrap._body.querySelector(`[${stepAttr}]:not([disabled])`);
      (again || wrap._aside).focus();
    }
    wrap._aside.onkeydown = ev => {
      if (ev.target.closest('input, textarea, summary')) return;
      const k = { ArrowUp: [rr.id, q.n - 1], ArrowDown: [rr.id, q.n + 1],
                  ArrowLeft: [ids[i - 1], q.n], ArrowRight: [ids[i + 1], q.n] }[ev.key];
      if (!k || !k[0] || k[1] < 1 || k[1] > qs.length) return;
      ev.preventDefault();
      go(k[0], k[1]);
    };
  };
  paint(r);
  wrap._update = paint;             // a poll or a step repaints in place
}

// ---- Run everyday tasks ----------------------------------------------------------
function evdDialog(pre = {}) {
  const board = DATA.models.filter(m => m.kind === 'instruct').map(m => m.id);
  const ids = [...EVD_DEFAULTS, ...board.filter(id => !EVD_DEFAULTS.includes(id))
    .sort((a, b) => evdName(a).localeCompare(evdName(b)))];
  // 12i.0: a model's own "Run" ticks that model alone
  if (pre.only && !ids.includes(pre.only)) ids.push(pre.only);
  // 12a.5: and the ones with questions not asked yet — a run asks them only those
  const pick = Object.fromEntries(ids.map(id => [id, pre.only ? id === pre.only
    : EVD_DEFAULTS.includes(id) && (!evdOf(id) || evdOf(id).unasked > 0)]));
  const back = el('div', { class: 'dlg-back', 'data-dialog': 'everyday' });
  const err = el('div', { class: 'warn', hidden: '', 'data-dialog-error': '1' });
  const go = el('button', { class: 'primary', 'data-dialog-go': '1' });
  const cancel = el('button', { 'data-dialog-cancel': '1', text: 'Cancel' });
  const sync = () => {
    const n = ids.filter(id => pick[id]).length;
    go.textContent = `Queue ${n} run${n === 1 ? '' : 's'}`;
    go.disabled = !n;
  };
  const nBank = evdAll();
  const list = el('div', { class: 'evpick', 'data-everyday-pick': '1' }, ids.map(id =>
    el('label', { class: 'evpick-row', 'data-evd-pick': id },
      el('input', { type: 'checkbox', checked: pick[id] ? '' : null,
        onchange: ev => { pick[id] = ev.target.checked; sync(); } }),
      el('span', { class: 'evpick-name', title: id, text: evdName(id) }),
      // 12a.5: a run asks only what the model has no answer to on today's words
      evdOf(id) ? el('span', { class: 'small se', 'data-evd-done': id,
          text: evdOf(id).unasked > 0 ? `${evdOf(id).unasked} not asked yet · asks only those`
            : 'all answered · marks them again' })
        // 12a.4: its answers are to an earlier wording (the pilot's five, or
        // a run before the questions were reworded)
        : evdEarlier(id) ? el('span', { class: 'small se', 'data-evd-done': id,
          text: `earlier wording · run all ${nBank}` }) : '')));
  const box = el('div', { class: 'dlg', role: 'dialog', 'aria-modal': 'true',
      'aria-labelledby': 'dlg-title' },
    el('h2', { id: 'dlg-title', text: 'Run everyday tasks' }),
    el('p', { class: 'small', text: `One run per model. It asks the questions the model has no `
      + `answer to on today's words — all ${evdAll()} the first time — through its chat `
      + `template, then marks every answer; the ${evdHidden()} hidden ones make its score. `
      + 'A few minutes each.' }),
    list, err, el('div', { class: 'dlg-actions' }, cancel, go));
  back.append(box);
  const close = () => {
    back.remove();
    document.removeEventListener('keydown', onKey, true);
    const again = pre.returnTo && document.querySelector(pre.returnTo);
    if (again) again.focus();
  };
  const onKey = e => {
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab') return;
    const f = [...box.querySelectorAll('input:not([disabled]), button:not([disabled])')]
      .filter(x => x.offsetParent);
    if (!f.length) return;
    const i = f.indexOf(document.activeElement);
    if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
    else if (!e.shiftKey && (i === f.length - 1 || i < 0)) { e.preventDefault(); f[0].focus(); }
  };
  cancel.onclick = close;
  back.addEventListener('click', e => { if (e.target === back) close(); });
  go.onclick = async () => {
    const want = ids.filter(id => pick[id]);
    go.disabled = true; go.textContent = 'Queueing…';
    const got = await evdQueue(want);
    const bad = got.filter(x => !x.ok), ok = got.filter(x => x.ok);
    for (const x of ok) pick[x.id] = false;            // what went in is not asked twice
    if (!bad.length) {
      close();
      toast(`${ok.length} run${ok.length === 1 ? '' : 's'} queued ·`,
        { key: 'submit', go: () => followRun(ok[0].sid), link: `follow ${ok.length === 1 ? 'it' : 'them'} →` });
      render();
      return;
    }
    // refused: what went in says so, what did not stays ticked with the reason
    err.hidden = false;
    err.replaceChildren(...(ok.length ? [el('p', { text: `${ok.length} queued.` })] : []),
      ...bad.map(x => el('p', { 'data-evd-refused': x.id, text: `${evdName(x.id)}: ${x.why}` })));
    for (const row of list.querySelectorAll('[data-evd-pick]'))
      row.querySelector('input').checked = !!pick[row.dataset.evdPick];
    sync();
  };
  document.addEventListener('keydown', onKey, true);
  document.body.append(back);
  sync();
  go.focus();
}

// ---------------------------------------------------------------------------
// Routing. The whole report is one file with no server, so the address bar is
// the only place a view can live — and putting it there is what makes Back work.
// A dashboard whose Back button leaves the page instead of returning to the list
// you came from is the single most reported annoyance in apps shaped like this.
// ---------------------------------------------------------------------------
const hashFor = () => (state.model ? 'model=' + encodeURIComponent(state.model)
                                  : state.topic ? 'topic=' + encodeURIComponent(state.topic)
                                  : viewHash(state.tab))
  // 11g: an open reader rides along, so a pasted link opens it too
  + (state.read ? '&read=' + encRead(state.read) : '');
// 12b: "tab=improve&sub=review&view=datasets", "tab=models&view=exam"
function viewHash(v) {
  const place = placeOf(v);
  if (v === 'leaderboard') return 'tab=models' + (lbHash() ? '&' + lbHash() : '');
  // 12i.1: the judge test's marking is a view of its own, so Back leaves it
  if (v === 'ai') return 'tab=ai' + (state.ai.mark ? '&sub=mark' : '');
  // 12i.2: the draft open, so a reload lands on it
  if (v === 'build') return 'tab=build' + (state.qb.id ? '&draft=' + state.qb.id : '');
  if (place === 'improve' || place === 'benchmarks')
    return `tab=${place}&sub=${SUB_SLUG[v]}`
      // 12g.1: which model Improve is on, so a link opens it
      + (v === 'pipeline' && state.imp.model ? '&model=' + encodeURIComponent(state.imp.model)
        : '');
  return 'tab=' + (PAGE_SLUG[v] || v);
}
// where an address lands: its view, and the state it carries. Old names and
// new ones both, so a bookmark from before 12b opens its new home (§8)
function viewOfHash(name, params) {
  const p = new URLSearchParams(params || '');
  const has = v => TABS.some(t => t[0] === v);
  const n = TAB_ALIASES[name] || name;
  // 12g.1: By topic and Review are the pipeline now — their addresses land on it
  const sub = { model: 'pipeline', topics: 'pipeline', review: 'pipeline', training: 'training',
                standard: 'tasks', exam: 'exam', everyday: 'everyday' }[p.get('sub')];
  let v = { home: 'overview', overview: 'overview', models: 'leaderboard',
            leaderboard: 'leaderboard', perplexity: 'leaderboard', loop: 'pipeline',
            review: 'pipeline', training: 'training', runs: 'queue', queue: 'queue',
            submit: 'queue', data: 'provenance', provenance: 'provenance', help: 'help',
            tasks: 'tasks', exam: 'exam', everyday: 'everyday', ai: 'ai', build: 'build' }[n];
  if (n === 'ai') state.ai.mark = p.get('sub') === 'mark';
  if (n === 'build') {
    const id = p.get('draft') || null;
    if (id !== state.qb.id) Object.assign(state.qb, { id, draft: null, at: 0, editing: null });
  }
  if (n === 'improve') v = sub || 'pipeline';
  if (n === 'benchmarks') v = ['tasks', 'exam', 'everyday'].includes(sub) ? sub : benchSub();
  if (!v || !has(v)) return null;
  if (v === 'leaderboard') {
    lbFromHash(params || '');
    // Perplexity & Loss is Models ▸ Standard ▸ Language modelling now
    if (n === 'perplexity') Object.assign(lbS(), { view: 'standard', chip: 'lm' });
  }
  // the model Improve is on, from the address; an old address keeps the
  // viewer's last one
  if (v === 'pipeline' && p.get('model')) state.imp.model = p.get('model');
  // Queue ▸ Submit a model is the Test a model dialog now
  if (n === 'submit') state.testOpen = true;
  return v;
}

function routeFromHash() {
  // 11g: the reader's part first — it can follow any page
  const [rest, rd] = splitRead(location.hash);
  state.read = rd;
  const h = decodeURIComponent(rest);
  const m = /^model=(.+)$/.exec(h);
  if (m && DATA.models.some(x => x.id === m[1])) { state.model = m[1]; state.topic = null; return; }
  state.model = null;
  // a topic page is the loop for one topic — deep-linkable, because it is the
  // page a person is sent to when someone says "look at law". 12b: it sits
  // under Benchmarks ▸ Knowledge exam, and its back link says so
  const tp = /^topic=(.+)$/.exec(h);
  if (tp && LIVE && topicOfSlug(tp[1])) { state.topic = tp[1]; state.tab = 'exam'; return; }
  state.topic = null;
  // 12a's pilot page is Benchmarks ▸ Everyday tasks now. 12b.3: a bare place
  // name — #home, #models, #benchmarks — is what people type, so it lands too
  const t = h === 'everyday' ? ['', 'everyday', '']
    : PLACE_WORDS.includes(h) ? ['', h, '']
    : /^tab=([^&]+)(?:&(.*))?$/.exec(rest);
  const v = t && viewOfHash(decodeURIComponent(t[1]), t[2] || '');
  // 12b.3: an empty address, or one the router does not know, is Home, and
  // the address bar says so — the bare address is the one the team was sent
  state.tab = v || 'overview';
  // an old address is shown as its new one; history keeps the entry
  const want = hashFor();
  if (location.hash.slice(1) !== want) history.replaceState(history.state, '', '#' + want);
}

const PLACE_WORDS = ['home', 'models', 'improve', 'benchmarks', 'runs', 'data', 'help'];

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
  if (!DATA) return;                  // routed by initData when the scores arrive
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
    // 12b: the address names a place and its part, not the view id
    const was = new URLSearchParams(from.replace(/^#/, '')), want = new URLSearchParams(viewHash(tab));
    if (was.get('tab') !== want.get('tab') || was.get('sub') !== want.get('sub')) return;
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
  // the column the table really sorts by: a key this view has no column for
  // (Avg, on the Knowledge exam) falls back as the sort does (12b)
  const cs = cols || [];
  const c = cs.find(x => x.key === state.sort.key) || cs.find(x => x.key === 'avg')
    || cs.find(x => x.key === 'cavg') || cs.find(x => x.key === 'javg')
    || cs.find(x => x.key === 'name');
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
  // 12h.1: the three that generate text — instruct models only, never in Avg
  ['instruction',  ['ifeval', 'mmlu_pro', 'hendrycks_math500']],
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

// ===========================================================================
// The Leaderboard (11c), and since 12b the Models place: topic-group chips,
// and the filters in Filters ▾, every one of them on 11a's popover so a poll
// never closes it. Two header rows, one-line cells tinted by their rank on
// the whole board, a row that opens the model page, and Insights under the
// table. The view lives in the hash, so a pasted link reproduces it. The
// Models tab's list merged into it; its facts are optional columns.
// ===========================================================================

// 12b: Standard's groups, and Language modelling (the old Perplexity & Loss
// page). Judged topics is not a chip any more: it is Knowledge exam, on the
// switch above the chips
const LB_CHIPS = [
  ['all', 'All tasks'], ['knowledge', 'Knowledge'], ['commonsense', 'Commonsense'],
  ['reasoning', 'Reasoning'], ['math', 'Math'], ['truthfulness', 'Truthfulness'],
  ['instruction', 'Instruction & maths'], ['lm', 'Language modelling']];
// the four kinds of test, named the same and in the same order everywhere.
// A kind with no data yet is not offered (On phone arrives in 12f)
const MODELS_VIEWS = { standard: 'Standard', exam: 'Knowledge exam', everyday: 'Everyday tasks' };
function modelsViews() {
  const out = ['standard'];
  if (DATA.models.some(m => Object.keys((m.judge || {}).tasks || {}).some(t => t.startsWith('exam_'))))
    out.push('exam');
  if (Object.keys(evd().models || {}).length) out.push('everyday');
  return out;
}
// Models opens on Standard, and remembers the viewer's last choice
function modelsView() {
  let v = 'standard';
  try { v = localStorage.getItem('bench-models-view') || v; } catch (e) { /* private */ }
  return modelsViews().includes(v) ? v : 'standard';
}
function setModelsView(v) {
  const L = lbS();
  try { localStorage.setItem('bench-models-view', v); } catch (e) { /* private */ }
  lbSet({ view: v, chip: v === 'exam' ? 'judged' : (L.stdChip || 'all') });
}
// the model facts the Models tab carried: columns, off until asked for
const FACT_KEYS = ['family', 'kind', 'date', 'flags'];
function lbFactsShown() {
  try { return JSON.parse(localStorage.getItem('bench-lb-facts') || '[]'); }
  catch (e) { return []; }
}
const LB_GROUP = { knowledge: 'Knowledge', commonsense: 'Commonsense', reasoning: 'Reasoning',
  math: 'Math', truthfulness: 'Truthfulness', instruction: 'Instruction & maths' };
// 12h.1: IFEval, MMLU-Pro and MATH-500 — asked through the chat template and
// scored on what the model writes; instruct models only, never in Avg
const genTasks = () => DATA.genTasks || [];
const isGen = t => genTasks().includes(t);
// how a model thinks — the catalogue's word for the nine, else what its last
// preflight read from its chat template: switch, always or never
const thinkingModeOf = id => (DATA.thinkingModes || {})[id]
  || (((DATA.models.find(x => x.id === id) || {}).archinfo) || {}).thinking || null;
// how a model's generative answers were asked: "thinking off · on vllm"
function genMode(m) {
  const g = m.gen || {}, th = g.thinking || {};
  const on = m.thinkingRow || th.on || th.mode === 'always';
  return [(on ? 'thinking on' : 'thinking off') + (th.mode === 'always'
      ? ' (it cannot turn thinking off)' : ''),
    g.backend ? 'on ' + g.backend + (g.fellBack ? ` (${g.fellBack})` : '') : '',
    g.subset ? `MMLU-Pro: a seeded subset of ${g.subset.n.toLocaleString('en')} of `
      + `${g.subset.of.toLocaleString('en')}, not comparable to published numbers` : '']
    .filter(Boolean).join(' · ');
}
// a column's words: who scores it, how it is asked, what its makers publish
function genTip(t) {
  const scorer = (DATA.models.map(m => ((m.gen || {}).tasks || {})[t]).find(x => x && x.scorer)
    || {}).scorer;
  const pub = Object.entries(DATA.published || {}).filter(([, ts]) => ts[t])
    .map(([mid, ts]) => `${mid.split('/').pop()} ${ts[t].v}${ts[t].note ? ` (${ts[t].note})` : ''}`);
  return [scorer ? 'scored by ' + scorer : '',
    'thinking off unless the row says thinking; a thinking run is a row of its own',
    pub.length ? 'published by the makers, in their own setups: ' + pub.join(' · ') : '',
    'instruct models only, and never part of the board\'s Avg — choose it under '
      + 'Benchmarks to average it with others'].filter(Boolean);
}
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
    state.lb = { ...LB_DEFAULTS, open: [], models: null, cols: null, tint, howto, shown: {},
                 focus: null, weak: null, radarSrc: 'tasks', view: 'standard', stdChip: 'all' };
  }
  return state.lb;
}

// the part of the hash after "tab=leaderboard": only what differs from the
// defaults, so a plain link stays plain
function lbHash() {
  const L = lbS(), out = [];
  if (L.view && L.view !== 'standard') out.push('view=' + L.view);
  for (const k of Object.keys(LB_DEFAULTS)) {
    if (k === 'chip' && L.view !== 'standard') continue;
    if (L[k] !== LB_DEFAULTS[k]) out.push(`${k}=${encodeURIComponent(L[k])}`);
  }
  // 12h.2: a table someone built — its benchmarks and its models — so the
  // address alone opens it: "cols=ifeval,mmlu_pro,math500&models=…"
  if (L.view === 'standard' && L.cols)
    out.push('cols=' + L.cols.map(t => LB_ALIAS[t] || t).join(','));
  if (L.models) out.push('models=' + L.models.map(encodeURIComponent).join(','));
  return out.join('&');
}
function lbFromHash(rest) {
  const L = lbS(), p = new URLSearchParams(rest || '');
  // an old link's "chip=judged" is the Knowledge exam view now (12b)
  const view = p.get('chip') === 'judged' ? 'exam' : p.get('view');
  L.view = Object.keys(MODELS_VIEWS).includes(view) ? view : 'standard';
  L.stdChip = LB_CHIPS.some(([v]) => v === p.get('chip')) ? p.get('chip') : 'all';
  L.chip = L.view === 'exam' ? 'judged' : L.stdChip;
  L.kind = LB_KINDS.some(([v]) => v === p.get('kind')) ? p.get('kind') : 'all';
  L.size = LB_SIZES.some(([v]) => v === p.get('size')) ? p.get('size') : 'all';
  L.status = LB_STATUS.some(([v]) => v === p.get('status')) ? p.get('status') : 'all';
  // 12h.2: the chosen benchmarks and models; a name this board does not know
  // is dropped, and a list with nothing left is the default
  const list = k => (p.get(k) || '').split(',').map(x => x.trim()).filter(Boolean);
  const unalias = Object.fromEntries(Object.entries(LB_ALIAS).map(([k, v]) => [v, k]));
  L.cols = L.view === 'standard' ? lbKnownCols(list('cols').map(t => unalias[t] || t)) : null;
  const ids = new Set(((DATA || {}).models || []).map(m => m.id));
  const ms = list('models').filter(id => !DATA || ids.has(id));
  L.models = ms.length ? [...new Set(ms)] : null;
}
// ---- 12h.2: a table you build ----------------------------------------------
// Benchmarks ▾ offers every Standard benchmark, in its chip groups: the
// harness's scored tasks and the three that generate. Not the Everyday tasks
// or the Knowledge exam (provisional or judged, never averaged with these),
// not Language modelling (a perplexity is not a percentage), and not a
// control (a control that could move a rank would stop being one).
const LB_ALIAS = { hendrycks_math500: 'math500' };
function lbBenchGroups() {
  const ok = t => DATA.accTasks.includes(t) && !(DATA.tasks[t] || {}).control;
  const used = new Set(), out = [];
  for (const [g, ts] of CATS) {
    const have = ts.filter(ok);
    have.forEach(t => used.add(t));
    if (have.length) out.push([g, LB_GROUP[g], have]);
  }
  const other = DATA.accTasks.filter(t => ok(t) && !used.has(t));
  if (other.length) out.push(['other', 'Other tasks', other]);
  return out;
}
const lbBenchAll = () => lbBenchGroups().flatMap(([, , ts]) => ts);
// 12i.0: the picker lists every benchmark the board knows, in its group; one
// nothing has run yet is there too, greyed — IFEval, MMLU-Pro and MATH-500 were
// missing until something had run them
function lbBenchPicker() {
  const have = new Set(lbBenchAll());
  return CATS.map(([g, ts]) => [g, LB_GROUP[g], ts.map(t => [t, have.has(t)])])
    .concat(lbBenchGroups().filter(([g]) => g === 'other')
      .map(([g, name, ts]) => [g, name, ts.map(t => [t, true])]));
}
// the chosen benchmarks in the checklist's order, the unknown ones dropped;
// none left is the default (the chip's own). Before the scores arrive the
// names are kept as given, and checked on the first paint
function lbKnownCols(ts) {
  if (!DATA) return ts.length ? [...new Set(ts)] : null;
  const want = new Set(ts);
  const out = lbBenchAll().filter(t => want.has(t));
  return out.length ? out : null;
}
// 12i.0: a benchmark by its own name, wherever it is named in words — the
// column headers keep LB_SHORT's short forms
const BENCH_NAMES = { mmlu: 'MMLU', hellaswag: 'HellaSwag', piqa: 'PIQA', winogrande: 'WinoGrande',
  arc_challenge: 'ARC-Challenge', arc_easy: 'ARC-Easy', gsm8k: 'GSM8K',
  truthfulqa_mc2: 'TruthfulQA', ifeval: 'IFEval', mmlu_pro: 'MMLU-Pro',
  hendrycks_math500: 'MATH-500' };
const benchName = t => BENCH_NAMES[t] || LB_SHORT[t] || t;
// the mean of the chosen benchmarks, on the Scale pill's scale; each error
// scales as its score does, and the errors add as variances (separate item
// sets), so the mean's is √Σse² / k. Null when a benchmark is missing; no ±
// when any benchmark has none
function customAvg(m, ts) {
  if (!ts || !ts.length) return null;
  const raw = state.avgMode === 'raw';
  let sum = 0, v2 = 0, noSe = false;
  for (const t of ts) {
    const cc = cell(t, m.id);
    if (!cc || cc.v == null) return null;
    const c = (DATA.tasks[t] || {}).chance;
    const scaled = !raw && c != null && c > 0 && c < 1;
    const k = scaled ? 1 / (1 - c) : 1;
    sum += scaled ? Math.max(0, (cc.v - c) * k) : cc.v;
    if (cc.se == null) noSe = true; else v2 += (cc.se * k) ** 2;
  }
  return { v: sum / ts.length, se: noSe ? null : Math.sqrt(v2) / ts.length };
}
// the rank on the chosen average — over the whole board, as the Avg's rank
// is: every model with all the chosen benchmarks. The Models picker never
// changes it, as no filter changes a rank
let _crank = { key: '', map: null };
function customRankOf(m, ts) {
  const key = [state.avgMode, ts.join(','), DATA.models.length, DATA.generated].join('|');
  if (_crank.key !== key) {
    const pool = DATA.models.filter(x => !x.duplicateOf)
      .map(x => ({ id: x.id, a: customAvg(x, ts) })).filter(x => x.a)
      .sort((a, b) => b.a.v - a.a.v);
    _crank = { key, map: new Map(pool.map((x, i) => [x.id, { n: i + 1, of: pool.length }])) };
  }
  return _crank.map.get(m.id) || null;
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
// 12i.1: what taking "provisional" off rests on, where judged scores rank
function judgeChecked() {
  const cal = (DATA.judged || {}).calibration;
  if (!judgedCalibrated() || !cal.by) return '';
  return el('span', { class: 'badge', 'data-judge-checked': String(cal.n),
    title: cal.method || null,
    text: `judge checked against ${cal.by} on ${cal.n} answers · κ ${cal.kappa}` });
}
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
  // 12b: the Models tab's facts — family, kind, last evaluated, flags —
  // are columns under Filters ▾, off by default
  const tail = [
    { key: 'family', label: 'Family', group: 'Model facts', fact: true, optional: true },
    { key: 'kind', label: 'Kind', group: 'Model facts', fact: true, optional: true },
    { key: 'date', label: 'Last evaluated', short: 'Updated', group: 'Model facts', fact: true,
      optional: true },
    { key: 'flags', label: 'Flags', group: 'Model facts', fact: true, optional: true }];
  // an area with no number for any model here is not a column (11e)
  const areasHere = areas.filter(a => ms.some(m => areaMmlu(m, a)));
  // 12h.2: the chosen benchmarks, and their own average: "Avg of 3"
  if (L.view === 'standard' && L.cols) {
    if (!L.cols.length) return [...lead.slice(1, 3), ...tail];
    // 12i.0: it says what it is — a reader who averaged the columns by hand
    // got another number and took this one for wrong
    lead[3] = { key: 'cavg', label: state.avgMode === 'raw' ? 'Avg, raw' : 'Avg above chance',
      num: true, group: '', unit: state.avgMode === 'raw' ? 'raw · %' : 'above chance · %' };
    return [...lead, ...L.cols.map(task), ...tail];
  }
  let mid;
  if (L.chip === 'all') {
    // today's view: the harness tasks, six by default, the rest one tick away
    const order = [...CATS.map(([g]) => LB_GROUP[g]), 'Other tasks', 'Perplexity'];
    mid = [...DATA.accTasks.map(task), ...DATA.pplTasks.map(task)]
      .map(c => ({ ...c, optional: true }))
      .sort((a, b) => order.indexOf(a.group) - order.indexOf(b.group));
    // 12b: the judged columns are Knowledge exam's, on the switch
    mid.push(...cats);
  } else if (L.chip === 'knowledge') {
    mid = [...(CATS.find(([g]) => g === 'knowledge')[1]).filter(t => DATA.accTasks.includes(t))
             .map(task),
           ...areasHere.map(a => ({ key: 'area:' + a, label: a, num: true, area: a,
             group: 'MMLU by area', unit: 'hidden questions · %' })),
           ...cats];
  } else if (L.chip === 'judged') {
    // 12b: the Knowledge exam view stands on its own numbers — the Standard
    // rank and average are Standard's, and a model with no exam result is
    // not a row here
    lead.splice(0, lead.length, ...lead.filter(c => c.key !== 'rank' && c.key !== 'avg'));
    mid = judgedCalibrated()
      ? [...javg, ...areas.map(a => ({ key: 'jarea:' + a, label: a, num: true, jarea: a,
            group: 'Judged by area', unit: 'mean · 0–4' })), ...judgedCols()]
      : [...javg];
  } else {
    mid = (CATS.find(([g]) => g === L.chip) || [null, []])[1]
      .filter(t => DATA.accTasks.includes(t)).map(task);
    // 12h.1: Instruction & maths stands on its own three numbers — the
    // Standard rank and average are Standard's, and none of these is in them
    if (L.chip === 'instruction')
      lead.splice(0, lead.length, ...lead.filter(c => c.key !== 'rank' && c.key !== 'avg'));
  }
  return [...lead, ...mid, ...tail];
}

// 11f: one word per column name; the long ones are the tooltip's
const LB_SHORT = { arc_challenge: 'ARC-C', arc_easy: 'ARC-E', truthfulqa_mc2: 'TruthfulQA',
  ifeval: 'IFEval', mmlu_pro: 'MMLU-Pro', hendrycks_math500: 'MATH-500' };

// A column's setup, in words — its tooltip, and its accessible name. The
// header shows only the name; this is where the n-shot, the unit and the
// scale went (11f).
function lbColTip(c) {
  const scale = state.avgMode === 'raw' ? 'raw accuracy' : 'above chance';
  if (c.key === 'rank' && (lbS().cols || lbS().models)) return ['# — this table\u2019s rows, '
    + 'in the order they are sorted'];
  if (c.key === 'rank') return ['# — rank among the ranked models on this board'];
  if (c.key === 'cavg') return [state.avgMode === 'raw'
    ? 'the mean of the chosen benchmarks\u2019 raw scores; the Scale pill switches it'
    : '0 = guessing, 100 = perfect, so a 25% guess on a 4-option test counts as 0'];
  if (c.key === 'name') return ['Model — sort by name'];
  if (c.key === 'params') return ['Params — parameter count, from the harness config or the name'];
  if (c.key === 'date') return ['Updated — when the model was last evaluated'];
  if (c.key === 'family') return ['Family — the Hub organisation, or a local run\'s name'];
  if (c.key === 'kind') return ['Kind — base, instruct or an uploaded checkpoint'];
  if (c.key === 'flags') return ['Flags — trained on diagnostics, or graded by a provisional judge'];
  if (c.key === 'avg') return [`Avg — mean of the required tasks, % ${scale}`,
    'the Scale pill switches it'];
  if (c.task) {
    const info = DATA.tasks[c.task] || {};
    return [c.lower ? `${c.task} — ${c.unit}, lower is better`
                    : `${c.task} — ${c.shot || 'n-shot unknown'}, % accuracy`,
      ...(info.control ? ['CONTROL — never in Avg'] : []),
      ...[info.domain, info.desc].filter(Boolean),
      ...(isGen(c.task) ? genTip(c.task) : [])];
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
  const opt = cols.filter(c => c.optional && !c.fact);
  // the facts are one choice across every chip
  const facts = lbFactsShown().filter(k => FACT_KEYS.includes(k));
  if (L.chip === 'all') return new Set([...lbShownTasks(opt), ...facts]);
  const want = L.shown[L.chip];
  // a chip's own group columns show; the long per-topic lists wait to be asked for
  return new Set([...(Array.isArray(want) ? want.filter(k => opt.some(c => c.key === k)) : []),
                  ...facts]);
}
function lbSaveShown(next) {
  const L = lbS();
  try { localStorage.setItem('bench-lb-facts', JSON.stringify(next.filter(k => FACT_KEYS.includes(k)))); }
  catch (e) { /* private */ }
  next = next.filter(k => !FACT_KEYS.includes(k));
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
  if (c.key === 'cavg') {
    const ts = lbS().cols;
    const x = customAvg(DATA.models.find(y => y.id === best.id), ts), y = customAvg(m, ts);
    return !!(x && y) && pair(x.se, y.se, x.v, y.v);
  }
  if (c.area) {
    const b = DATA.models.find(x => x.id === best.id);
    const x = areaMmlu(b, c.area), y = areaMmlu(m, c.area);
    return !!(x && y) && pair(x.se, y.se, x.v, y.v);
  }
  return false;
}

// 12b: Models — the Leaderboard and the Models tab, one table. A switch for
// the kind of test; a row opens the model page; models with nothing in this
// view sit under one line at the bottom, not in rows of dashes
function modelsHead(badge) {
  const L = lbS(), views = modelsViews();
  return [el('h2', {}, 'Models', badge || ''),
    views.length > 1 ? el('nav', { class: 'subswitch', 'data-models-switch': '1', role: 'tablist',
        'aria-label': 'kind of test' },
      views.map(v => el('button', { class: 'chip-btn' + (L.view === v ? ' on' : ''), role: 'tab',
        'data-models-view': v, 'aria-selected': String(L.view === v), text: MODELS_VIEWS[v],
        onclick: () => setModelsView(v) }))) : ''];
}
// "Not tested on this (12) ▸": one collapsed line, each model with its Test
function notTestedRows(none, ncols, suite, whyNot = null) {
  if (!none.length) return [];
  const open = !!state.lbNotTested;
  const head = el('tr', { class: 'nottested', 'data-not-tested': String(none.length) },
    el('td', { colspan: String(ncols) }, el('button', { class: 'quiet', 'data-not-tested-toggle': '1',
      'aria-expanded': String(open), onclick: () => { state.lbNotTested = !open; render(); },
      text: `Not tested on this (${none.length}) ${open ? '▾' : '▸'}` })));
  if (!open) return [head];
  return [head, ...none.map(m => {
    // a string says why there is no Test ("instruct only"); {text, suite}
    // says what is missing (12h.2), and Test asks for it
    const why = whyNot ? whyNot(m) : null;
    const said = typeof why === 'string';
    const note = said ? why : why && why.text;
    const test = said ? null : why ? why.suite : suite;
    return el('tr', { class: 'nottested-row', 'data-not-tested-row': m.id },
      el('td', { colspan: String(ncols) },
        el('a', { href: '#model=' + encodeURIComponent(m.id), text: m.name }),
        note ? el('span', { class: 'se', [said ? 'data-instruct-only' : 'data-missing']: m.id,
          text: ' · ' + note }) : '',
        LIVE && test ? [el('span', { class: 'se', text: ' · ' }), el('a', { href: '#',
          'data-not-tested-test': m.id, text: 'Test', onclick: e => { e.preventDefault();
            state.sub.suite = test; openTest(m.id); } })] : ''));
  })];
}
// Everyday tasks: each model's row — n of k per group, and the total (12a.2)
function lbEveryday(ms) {
  const groups = evdGroups();
  const rows = lbFilter(ms);
  const have = rows.filter(m => evdOf(m.id)).sort((a, b) => natCmp(a.name, b.name));
  const none = rows.filter(m => !evdOf(m.id));
  const prov = have.some(m => evdOf(m.id).provisional);
  const ncols = groups.length + 2;
  const table = el('table', { class: 'lb norank', 'data-lb-table': '1', 'data-lb-everyday': '1' },
    el('thead', {}, el('tr', { class: 'names' },
      el('th', { class: 'model pin', scope: 'col', text: 'Model' }),
      groups.map(([g, label]) => el('th', { class: 'num', scope: 'col', 'data-evd-col': g,
        title: `${evdQs(g).length} questions`, text: label })),
      el('th', { class: 'num', scope: 'col', text: 'Total' }))),
    el('tbody', {}, have.map(m => {
      const e = evdOf(m.id);
      return el('tr', { class: 'clickrow', 'data-lb-row': m.id,
          onclick: ev => { if (ev.target.closest('a, button')) return;
            navigate({ model: m.id, topic: null }); } },
        el('td', { class: 'model pin', 'data-model': m.id },
          el('a', { class: 'mname mlink', href: '#model=' + encodeURIComponent(m.id), text: m.name })),
        groups.map(([g, label]) => {
          const n = evdGroupCount(e, g);
          return el('td', { class: 'num' + (n && !evdMissing(e) ? '' : ' se'), 'data-evd-g': g,
            title: n ? label : 'not asked', text: n || '—' });
        }),
        el('td', { class: 'num' }, evdTotal(e, m.id), evdRanOut(e)));
    }), notTestedRows(none, ncols, 'everyday')));
  return [el('div', { class: 'card', 'data-lb-card': '1' },
    ...modelsHead(evdBadge(prov)),
    lbToolbar(ms, lbColumns(ms), new Set(), 0),
    hfade('lb', el('div', { class: 'lb-wrap', 'data-hkeep': 'lb' }, table)),
    el('p', { class: 'lbcap', text: `${evdAll()} questions in ${evdGroupsWord()} groups, typed the way `
      + `people type on a phone; each count is the ${evdHidden()} hidden ones. Open a model for `
      + 'its answers to the practice ones; Benchmarks ▸ Everyday tasks has those questions.' }))];
}

// 12h.1: one generative cell — its number, and under it, only when there is
// something to say: the answers that ran out of room, IFEval's instruction-
// level score, a subset, a score far below what its makers publish
function genCell(c, m, cc, one, pctn) {
  const g = ((m.gen || {}).tasks || {})[c.task] || {}, far = ((m.gen || {}).far || {})[c.task];
  const sub = c.task === 'mmlu_pro' && (m.gen || {}).subset;
  const td = one(c, m, cc.v, cc.se ? (100 * cc.se).toFixed(1) : null, pctn, {
    title: [genMode(m), g.inst_acc != null ? `instruction-level ${pctn(g.inst_acc)}` : '',
      g.ran_out ? `${g.ran_out} answer${g.ran_out === 1 ? '' : 's'} ran out of room` : '',
      far ? `far below published (${far.published}${far.note ? ', ' + far.note : ''}): `
        + 'check extraction' : ''].filter(Boolean).join(' · '),
    'data-gen-cell': c.task });
  const note = (text, attrs, warn) => el('span', { class: 'cellnote' + (warn ? ' warn' : ''),
    ...attrs, text });
  if (g.inst_acc != null) td.append(note(`${pctn(g.inst_acc)} by instruction`,
    { 'data-inst-level': c.task }));
  if (g.ran_out) td.append(note(`${g.ran_out} ran out of room`, { 'data-ran-out': String(g.ran_out) }));
  if (sub) td.append(note('subset', { 'data-subset': c.task }));
  if (far) td.append(note('far below published, check extraction', { 'data-far-below': c.task },
    true));
  return td;
}

function vLeaderboard(ms) {
  const L = lbS();
  if (!modelsViews().includes(L.view)) { L.view = 'standard'; L.chip = L.stdChip || 'all'; }
  if (L.view === 'everyday') return lbEveryday(ms);
  // the exam's scores are not ranked until a person has agreed with the judge:
  // one line says so, instead of an empty table
  if (L.view === 'exam' && !judgedCalibrated())
    return [el('div', { class: 'card', 'data-lb-card': '1' }, ...modelsHead(),
      el('p', { class: 'note', 'data-exam-off': '1', text: 'Knowledge exam scores are shown on '
        + 'each model page and are not ranked here yet: ' + judgedOffWhy() + '.' }))];
  // Language modelling: the old Perplexity & Loss page, its charts under it
  if (L.view === 'standard' && L.chip === 'lm' && !L.cols)
    return [el('div', { class: 'card', 'data-lb-card': '1' }, ...modelsHead(),
      lbToolbar(ms, lbColumns(ms), new Set(), 0)), ...vPpl(lbFilter(ms))];
  // 12h.2: every benchmark unticked — a sentence, not a table of names
  if (L.view === 'standard' && L.cols && !L.cols.length)
    return [el('div', { class: 'card', 'data-lb-card': '1' }, ...modelsHead(),
      lbToolbar(ms, lbColumns(ms), new Set(), 0), lbCustomLine(0, 0),
      el('p', { class: 'note', 'data-no-bench': '1', text: 'No benchmark chosen: tick one under '
        + 'Benchmarks ▾, or choose a group.' }))];
  const cols = lbColumns(ms);
  const shown = lbShownFor(cols);
  const opt = cols.filter(c => c.optional);
  const nHidden = opt.length - opt.filter(c => shown.has(c.key)).length;
  const visCols = cols.filter(c => !c.optional || shown.has(c.key));
  const jval = (m, c) => c.judged === 'avg' ? m.judgedAvg
    : !judgedOkM(m) ? null
    : (m.tainted || []).includes(c.judged) ? null      // shown on the page, never ranked here
    : (((m.judge || {}).tasks || {})[c.judged] ? pubScore(m.judge.tasks[c.judged]) : null);
  const custom = L.view === 'standard' && !!L.cols;
  // 12i.0: a table someone built numbers its rows 1, 2, 3, not by board ranks
  const built = L.view === 'standard' && (!!L.cols || !!L.models);
  const val = (m, c) => c.key === 'avg' ? officialAvg(m)
    : c.key === 'cavg' ? (customAvg(m, L.cols) || {}).v
    : c.key === 'params' ? m.params
    : c.key === 'name' ? m.name
    : c.key === 'date' ? lastEval(m)
    : c.key === 'family' ? famOf(m)
    : c.key === 'kind' ? (m.source === 'artifact' ? 'checkpoint' : m.kind)
    : c.key === 'flags' ? [(m.tainted || []).length ? 'tainted' : '', m.provisional
        ? 'provisional' : ''].filter(Boolean).join(' ') || null
    : c.judged ? jval(m, c)
    : c.jarea ? areaJudged(m, c.jarea).v
    : c.area ? (areaMmlu(m, c.area) || {}).v
    : c.cat ? ((mmluCats(m) || {})[c.cat] || {}).score_report
    : c.task ? (cell(c.task, m.id) || {}).v : null;
  const rowsIn = lbFilter(ms);
  const sortCol = cols.find(c => c.key === state.sort.key) || cols.find(c => c.key === 'avg')
    || cols.find(c => c.key === 'cavg') || cols.find(c => c.key === 'javg')
    || cols.find(c => c.key === 'name');
  const sorted = [...rowsIn].sort((a, b) => {
    const va = val(a, sortCol), vb = val(b, sortCol);
    if (va == null && vb == null) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return typeof va === 'string' ? state.sort.dir * natCmp(va, vb) : state.sort.dir * (va - vb);
  });
  // ranked rows first, whatever the sort: a preliminary model's per-task
  // numbers are valid, and it is still not on the ladder
  const ordered = L.chip === 'judged' || custom ? sorted
    : [...sorted.filter(m => officialAvg(m) != null), ...sorted.filter(m => officialAvg(m) == null)];
  const dupsOf = {};
  for (const m of ordered)
    if (m.duplicateOf && ordered.some(x => x.id === m.duplicateOf))
      (dupsOf[m.duplicateOf] = dupsOf[m.duplicateOf] || []).push(m);
  // 12b: a model with nothing in this view's columns is not a row of dashes
  const dataCols = visCols.filter(c => !['rank', 'name', 'params'].includes(c.key) && !c.fact);
  // …but a model a judge that does not count (a local one) has graded WAS
  // tested: its row stays, its cells blank with the reason on hover
  const judgedAny = m => Object.keys((m.judge || {}).tasks || {}).some(t => t.startsWith('exam_'));
  // 12h.2: with benchmarks chosen, a model is a row only with every one of
  // them — one missing any is not averaged, and says what it is missing
  const testedIn = m => custom ? L.cols.every(t => (cell(t, m.id) || {}).v != null)
    : dataCols.some(c => val(m, c) != null) || (L.view === 'exam' && judgedAny(m));
  const notTested = ordered.filter(m => !testedIn(m) && !(m.duplicateOf && dupsOf[m.duplicateOf]));
  const lbAll = ordered.filter(m => !(m.duplicateOf && dupsOf[m.duplicateOf]) && testedIn(m));
  const lbPg = paged('leaderboard', lbAll, JSON.stringify([state.sort, state.q, state.kind,
    state.src, state.avgMode, L.view, L.chip, L.kind, L.size, L.status, L.models, L.cols]));
  const rows = lbPg.rows.flatMap(m => [m, ...((state.lbDupOpen || {})[m.id] ? dupsOf[m.id] || [] : [])]);
  // the leaders are bold whatever the Tint switch says; Tint only adds the wash
  const leaders = lbLeaders(visCols, val);

  // the chosen average stands in for Avg, arrow and all (12h.2)
  const sortedBy = c => state.sort.key === c.key || (custom && c.key === 'cavg' && sortCol === c);
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
        'aria-sort': sortedBy(c) ? (state.sort.dir > 0 ? 'ascending' : 'descending') : 'none',
        onclick: () => { state.sort = { key: c.key,
          dir: state.sort.key === c.key ? -state.sort.dir : (c.key === 'name' ? 1 : c.lower ? 1 : -1) };
          render(); } },
        el('span', { class: 'hname', text: c.short || c.label }),
        sortedBy(c) ? el('span', { class: 'dir', text: state.sort.dir > 0 ? ' ▲' : ' ▼' }) : '');
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
  rows.forEach(m => {
    // 12b: a row opens the model page — the detail lives there, not in a
    // row that expands
    const tr = el('tr', { class: 'clickrow' + (m.duplicateOf && dupsOf[m.duplicateOf] ? ' duprow' : ''),
      'data-lb-row': m.id,
      onclick: e => {
        // links, buttons, checkboxes and badges with a job of their own keep it
        if (e.target.closest('a, button, input, select, label, .badge[title]')) return;
        navigate({ model: m.id, topic: null });
      } },
      visCols.map(c => {
        if (c.key === 'rank') {
          if (built) {
            // an opened duplicate sits under its row and takes no number
            const at = lbPg.rows.indexOf(m);
            const n = at < 0 ? '' : String(lbPg.from + at);
            return el('td', { class: 'rank pin0 num' },
              el('span', { class: 'mono', 'data-row-n': n, text: n,
                title: n ? `row ${n} of ${lbAll.length}, in the order the table is sorted`
                  : 'a duplicate of the row above' }));
          }
          const r = rankOf(m);
          return el('td', { class: 'rank pin0 num' },
            el('span', { class: 'mono', text: r ? String(r.n) : '—',
              title: r ? `rank ${r.n} of ${r.of} ranked models on this board`
                       : 'preliminary — not ranked' }));
        }
        if (c.key === 'family') return el('td', { class: 'small', 'data-fact': 'family',
          text: famOf(m) });
        if (c.key === 'kind') return el('td', { class: 'small', 'data-fact': 'kind',
          text: m.source === 'artifact' ? 'checkpoint' : m.kind || '—' });
        if (c.key === 'flags') return el('td', { class: 'small', 'data-fact': 'flags' },
          (m.tainted || []).length ? el('span', { class: 'badge taint', text: 'tainted',
            title: 'trained on data derived from ' + m.tainted.join(', ') }) : '',
          m.provisional ? el('span', { class: 'badge taint', text: 'provisional' }) : '');
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
            // 12h.1: a thinking row, or a model that cannot stop thinking
            m.thinkingRow || ((m.gen || {}).thinking || {}).mode === 'always'
              ? el('span', { class: 'badge instruct', 'data-thinking-badge': m.id,
                  title: genMode(m), text: 'thinking' }) : '',
            // a thinking row has only these three: Standard's "preliminary" is not its
            m.thinkingRow ? '' : warnBadge(m) || '', dupBadge(m) || '',
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
        if (c.key === 'cavg') {
          const a = customAvg(m, L.cols);
          return one(c, m, a.v, a.se != null ? (100 * a.se).toFixed(1) : null, pctn, {
            title: `mean over ${L.cols.map(benchName).join(', ')}, `
              + (state.avgMode === 'raw' ? 'raw accuracy' : 'scaled so chance = 0'),
            'data-cavg': m.id });
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
        if (!cc) return el('td', { class: 'num se', text: '—',
          title: isGen(c.task) && m.kind === 'base'
            ? 'instruct only: asked through the chat template' : null });
        if (isGen(c.task)) return genCell(c, m, cc, one, pctn);
        return one(c, m, cc.v, cc.se && !c.lower ? (100 * cc.se).toFixed(1) : null,
          c.lower ? x => num(x, 3) : pctn);
      }));
    tbody.append(tr);
  });
  // 12h.1: under Instruction & maths, a base model is not "not tested" but
  // "instruct only", said once here, never as an empty cell
  const genView = !custom && dataCols.length > 0 && dataCols.every(c => c.task && isGen(c.task));
  // 12h.2: "Qwen3.5-0.8B · no MATH-500 · Test" — Test asks for what is missing
  const missing = m => {
    const miss = L.cols.filter(t => (cell(t, m.id) || {}).v == null);
    const text = 'no ' + miss.map(benchName).join(', ');
    const harness = miss.filter(t => !isGen(t));
    if (m.thinkingRow && harness.length)
      return { text: text + ' · a thinking row has only IFEval, MMLU-Pro and MATH-500' };
    if (!harness.length && m.kind === 'base') return { text: text + ' · instruct only' };
    return { text, suite: harness.length ? 'full' : 'generative' };
  };
  tbody.append(...notTestedRows(notTested, ncols,
    genView ? 'generative' : L.view === 'exam' ? 'judged' : 'full',
    custom ? missing : genView ? m => (m.kind === 'base' ? 'instruct only' : null) : null));
  // what Copy as CSV copies: these rows, these columns, as shown (12h.2)
  state.lbTable = { cols: visCols, rows: lbAll, val, custom };

  const table = el('table', { class: 'lb' + (L.tint ? ' tinted' : '')
      + (visCols.some(c => c.key === 'rank') ? '' : ' norank'), 'data-lb-table': '1' },
    thead, tbody);
  return [el('div', { class: 'card', 'data-lb-card': '1' },
      ...modelsHead(L.view === 'exam' ? judgeChecked() : ''),
      lbToolbar(ms, cols, shown, nHidden),
      L.chip === 'knowledge' && staleSentence(ms) && !custom
        ? el('p', { class: 'warn', 'data-stale-diag': '1', text: staleSentence(ms) }) : '',
      // 12i.0: a chosen model with no score in these columns is not "tested"
      lbCustomLine(lbAll.length, lbAll.filter(m => (L.cols || DATA.accTasks)
        .some(t => (cell(t, m.id) || {}).v != null)).length),
      statusLine(lbPg, 'models', [
        L.chip === 'judged' || custom ? null
          : `${lbAll.filter(m => officialAvg(m) != null).length} ranked`,
        `sorted by ${lbSortLabel(cols)}`,
        L.chip !== 'all' && !custom ? (LB_CHIPS.find(([v]) => v === L.chip) || [])[1] : null]),
      lbPg.pager,
      // the Models tab's empty state, kept: a sentence and the way back
      custom && !lbAll.length && rowsIn.length ? empty(`No model here has all ${L.cols.length} `
        + 'of these — each one is under the line with what it is missing.', 'Reset',
        () => lbSet({ cols: null, models: null }))
        : '',
      // 12i.0: Clear in Models ▾ applies at once, and leaves this
      L.models && !L.models.length ? empty('No model chosen: tick one under Models ▾.',
        'All ranked', () => lbSet({ models: null }), { 'data-no-models': '1' })
      : !rowsIn.length ? empty('No model matches these filters.', 'Clear the filters',
        () => lbSet({ kind: 'all', size: 'all', status: 'all', models: null }))
        : hfade('lb', el('div', { class: 'lb-wrap stick' + (state.lbWide ? ' hscroll' : ''),
          'data-hkeep': 'lb' }, table)),
      el('p', { class: 'lbcap', 'data-lb-caption': '1', text: 'Bold = best in the column or '
        + 'within its noise · hover a score for its ± error · hover a column name for its setup · '
        + 'click a row for the model' }),
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
      // 12h.2: a group chip fills the Benchmarks checklist with its own; while
      // benchmarks are chosen, no group is the one shown
      const on = L.chip === v && !(L.view === 'standard' && L.cols);
      return el('button', { class: 'chip-btn' + (on ? ' on' : ''), 'data-chip': v,
        'aria-pressed': String(on), 'aria-disabled': off ? 'true' : null,
        'aria-describedby': off ? 'why-judged-chip' : null,
        title: off ? judgedOffWhy() : null, text: t,
        onclick: () => { if (!off) lbSet({ chip: v, cols: null });
          else { state.lbChipWhy = !state.lbChipWhy; render(); } } });
    }),
    // 12h.2: the team's saved views, after a small divider
    ...(LIVE && (state.views || []).length ? [el('span', { class: 'chipdiv', 'aria-hidden': 'true' }),
      ...state.views.map(viewChip)] : []));
  const std = L.view === 'standard';
  const pills = el('div', { class: 'pills' },
    pillMenu('kind', 'Kind', LB_KINDS, L.kind, v => lbSet({ kind: v })),
    pillMenu('size', 'Size', LB_SIZES, L.size, v => lbSet({ size: v })),
    pillMenu('status', 'Status', LB_STATUS, L.status, v => lbSet({ status: v })),
    lbColumnsPill(cols, shown, nHidden),
    std ? '' : lbModelsPill(ms),
    pillMenu('scale', 'Scale', [['chance', 'above chance'], ['raw', 'raw accuracy']],
      state.avgMode, v => { state.avgMode = v; render(); }));
  const note = calOk ? '' : el('p', { class: 'chipnote', id: 'why-judged-chip', role: 'note',
    'data-why': 'judged-chip', hidden: state.lbChipWhy ? null : '', text: judgedOffWhy() });
  // 11h: on a phone the six pills are one "Filters ▾" beside the chips, and
  // open as a sheet from the bottom; the chips are one row that scrolls.
  // 12b: at every width — Kind, Size, Status, Columns, Models and Scale all
  // sit in Filters ▾. Standard has its chips; the other kinds have none
  const set = [L.kind !== LB_DEFAULTS.kind, L.size !== LB_DEFAULTS.size,
    L.status !== LB_DEFAULTS.status, !std && !!L.models, state.avgMode === 'raw']
    .filter(Boolean).length;
  const open = !!state.lbFilters;
  const toggle = el('button', { class: 'pill' + (set ? ' on' : ''), 'data-filters': String(set),
    'aria-expanded': String(open), 'aria-controls': 'filter-sheet',
    text: `Filters${set ? ` · ${set}` : ''} ▾`,
    onclick: () => { state.lbFilters = !open; render(); } });
  // the benchmarks in view now, for the checklist: read when it is used, as
  // the panel outlives the render that built it
  state.lbBenchNow = L.cols || cols.filter(c => c.task && !c.lower && !(DATA.tasks[c.task] || {})
    .control && (!c.optional || shown.has(c.key))).map(c => c.task);
  return el('div', { class: 'lbbar narrow' },
    el('div', { class: 'chiprow' }, std ? chips : '',
      el('div', { class: 'pickers', 'data-pickers': '1' },
        std ? lbBenchPill() : '', std ? lbModelsPill(ms) : '', toggle)),
    note, std ? lbViewForm() : '',
    open ? el('div', { class: 'fsheet', id: 'filter-sheet', role: 'dialog', 'aria-label': 'filters',
        'data-filter-sheet': '1',
        onkeydown: e => { if (e.key === 'Escape' && !POP.panel) { state.lbFilters = false; render(); } } },
      el('div', { class: 'fsheet-head' }, el('b', { text: 'Filters' }),
        el('button', { class: 'ghost', text: 'Done', 'data-filters-done': '1',
          onclick: () => { state.lbFilters = false; render(); } })),
      pills) : '');
}
// crossing 720px turns the panel into a sheet: close it
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
    const tagOf = c => c.fact ? 'facts' : c.judged ? 'judged' : c.cat ? 'cats' : 'tasks';
    const heads = { tasks: 'Task columns', judged: 'Judged topics (rubric 0–4)',
                    cats: 'MMLU by topic', facts: 'Model facts' };
    const tags = ['tasks', 'judged', 'cats', 'facts'].filter(t => opt.some(c => tagOf(c) === t));
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

// Models ▾ — a search and a checklist with each family's colour
// 12h.2: "Models: 6 ▾", beside Filters on Standard — grouped as the board
// groups them (instruct, base, checkpoints); "All ranked" is today's default.
// 12i.0: a tick applies at once, as Benchmarks ▾ does — there is no Apply
function lbModelsPill(ms) {
  const L = lbS();
  const btn = el('button', { class: 'pill' + (L.models ? ' on' : ''), id: 'pill-models',
    'data-models-menu': '1', text: `Models: ${L.models ? L.models.length : 'all'} ▾` });
  return popover(btn, () => {
    const pick = new Set(L.models || ms.map(m => m.id));
    const list = el('div', { class: 'mlist' });
    const foot = el('p', { class: 'small se', 'data-models-foot': '1' });
    const say = () => { foot.textContent = pick.size === ms.length ? 'All models shown'
      : `${pick.size} of ${ms.length} shown`; };
    const apply = () => lbSet({ models: pick.size === ms.length ? null : [...pick] });
    const groupOf = m => m.source === 'artifact' ? 'checkpoints'
      : m.kind === 'instruct' ? 'instruct' : 'base';
    const row = m => el('label', { class: 'small mrow' },
      el('input', { type: 'checkbox', 'data-model-pick': m.id, checked: pick.has(m.id) ? '' : null,
        onchange: e => { if (e.target.checked) pick.add(m.id); else pick.delete(m.id);
          say(); apply(); } }),
      el('span', { class: 'famdot', style: `background:${famColor(m)}`, title: famOf(m) }),
      ' ' + m.name, el('span', { class: 'se', text: ' ' + famOf(m) }));
    const fill = q => {
      const hit = ms.filter(m => !q
        || (m.name + ' ' + m.id + ' ' + famOf(m)).toLowerCase().includes(q.toLowerCase()));
      list.replaceChildren(...['instruct', 'base', 'checkpoints'].flatMap(g => {
        const gs = hit.filter(m => groupOf(m) === g);
        return gs.length ? [el('div', { class: 'small se mgroup', 'data-model-group': g, text: g }),
          ...gs.map(row)] : [];
      }));
    };
    fill(state.lbModelsQ || '');
    say();
    return el('div', { class: 'moremenu modelsmenu', id: 'pop-models', 'aria-label': 'models' },
      el('div', { class: 'frm' },
        el('button', { class: 'quiet', text: 'All ranked', 'data-models-default': '1',
          title: 'every model, the ranked ones first — the default', onclick: () => {
            popClose(true); lbSet({ models: null }); } }),
        el('button', { class: 'quiet', text: 'Clear', 'data-models-clear': '1', onclick: () => {
          pick.clear(); say(); apply(); } })),
      el('input', { type: 'search', placeholder: 'search models…', 'aria-label': 'search models',
        'data-keep': 'lbmodels', value: state.lbModelsQ || '',
        oninput: e => { state.lbModelsQ = e.target.value; fill(e.target.value); } }),
      list, foot);
  }, { key: 'models', menu: false, rebuild: true, inCard: true });
}

// 12h.2: "Benchmarks: 3 ▾" — every Standard benchmark in its chip groups,
// with a search. A tick applies at once; the average follows
function lbBenchPill() {
  const L = lbS();
  const n = (state.lbBenchNow || []).length;
  const btn = el('button', { class: 'pill' + (L.cols ? ' on' : ''), id: 'pill-benchmarks',
    'data-benchmarks-menu': String(n), text: `Benchmarks: ${n} ▾` });
  return popover(btn, () => {
    const now = new Set(state.lbBenchNow || []);
    const set = (t, on) => {
      const next = new Set(state.lbBenchNow || []);
      if (on) next.add(t); else next.delete(t);
      lbSet({ cols: lbBenchAll().filter(x => next.has(x)) });
    };
    const list = el('div', { class: 'benchlist' });
    const fill = () => {
      const q = (state.lbBenchQ || '').trim().toLowerCase();
      // a benchmark's own names: "math" finds MATH-500, not its whole group
      const hit = ([t]) => !q || (t + ' ' + benchName(t)).toLowerCase().includes(q);
      const groups = lbBenchPicker().map(([g, name, ts]) => [g, name, ts.filter(hit)])
        .filter(([, , ts]) => ts.length);
      // 12i.0: each by its own name alone; one not run yet is greyed and says so
      list.replaceChildren(...(groups.length ? groups.map(([g, name, ts]) =>
        el('div', { class: 'colgroup', 'data-bench-group': g },
          el('div', { class: 'small se', text: name }),
          ts.map(([t, ran]) => el('label', { class: 'small' + (ran ? '' : ' notrun'),
              'data-bench-row': t },
            el('input', { type: 'checkbox', 'data-bench': t, checked: now.has(t) ? '' : null,
              disabled: ran ? null : '', onchange: e => set(t, e.target.checked) }),
            ' ' + benchName(t),
            ran ? '' : el('span', { class: 'se', 'data-not-run': t, text: ' · not run yet' })))))
        : [el('p', { class: 'small se', text: 'No benchmark matches.' })]));
    };
    fill();
    return el('div', { class: 'moremenu benchmenu', id: 'pop-benchmarks', 'aria-label': 'benchmarks' },
      el('input', { type: 'search', placeholder: 'search benchmarks…', 'aria-label': 'search benchmarks',
        'data-keep': 'lbbench', value: state.lbBenchQ || '',
        oninput: e => { state.lbBenchQ = e.target.value; fill(); } }),
      list,
      el('div', { class: 'frm' },
        el('button', { class: 'quiet', text: 'Clear', 'data-bench-clear': '1',
          onclick: () => lbSet({ cols: [] }) })),
      el('p', { class: 'small se', 'data-bench-foot': '1', text: 'The average is over the ticked '
        + 'ones only. Everyday tasks and the Knowledge exam keep their own tables.' }));
  }, { key: 'benchmarks', menu: false, rebuild: true, inCard: true });
}

// 12h.2: one line above a table someone built — what is shown, Save view,
// Reset, and ⋯ Copy as CSV. Nothing when the table is today's
function lbCustomLine(nRows, nTested = nRows) {
  const L = lbS();
  // 12i.0: models chosen that have no scores here say so: "3 chosen · 2 tested"
  const chosen = L.models ? L.models.length : null;
  const count = chosen != null && nTested < chosen ? `${chosen} chosen · ${nTested} tested`
    : `${nRows} model${nRows === 1 ? '' : 's'}`;
  if (L.view !== 'standard' || (!L.cols && !L.models)) return '';
  const what = L.cols ? (L.cols.length ? L.cols.map(benchName).join(', ') : 'no benchmarks')
    : (LB_CHIPS.find(([v]) => v === L.chip) || [null, 'All tasks'])[1];
  const more = popover(el('button', { class: 'quiet cl-more', id: 'pill-custom-more',
      'data-custom-more': '1', 'aria-label': 'more: copy as CSV', text: '⋯' }),
    () => el('div', { class: 'moremenu', id: 'pop-custom-more', 'aria-label': 'more' },
      el('button', { role: 'menuitem', 'data-copy-csv': '1', text: 'Copy as CSV',
        onclick: () => { popClose(true);
          const n = ((state.lbTable || {}).rows || []).length;
          copyText(lbCsv(), `${n} row${n === 1 ? '' : 's'} as CSV`); } })),
    { key: 'custom-more' });
  return el('div', { class: 'customline', 'data-custom-line': '1' },
    el('span', { class: 'cl-what', 'data-custom-what': '1' }, el('b', { text: 'Custom' }),
      ` · ${what} · ${count}`),
    el('span', { class: 'cl-acts' },
      LIVE ? el('button', { class: 'quiet', 'data-save-view': '1', text: 'Save view',
        'aria-expanded': String(state.lbForm === 'save'),
        onclick: () => { state.lbForm = state.lbForm === 'save' ? null : 'save';
          state.lbViewName = ''; state.after = { focus: '[data-view-name]' }; render(); } }) : '',
      el('button', { class: 'quiet', 'data-custom-reset': '1', text: 'Reset',
        onclick: () => { state.lbForm = null; lbSet({ cols: null, models: null }); } }),
      L.cols && !L.cols.length ? '' : more));
}

// a saved view is the state it names: its group, its benchmarks, its models
const sameList = (a, b) => (!a && !b) || (!!a && !!b && a.length === b.length
  && [...a].sort().join('\n') === [...b].sort().join('\n'));
function lbIsView(v) {
  const L = lbS(), sp = v.spec || {};
  return L.view === 'standard' && sameList(L.cols, lbKnownCols(sp.cols || []))
    && sameList(L.models, sp.models || null) && (!!L.cols || L.chip === (sp.chip || 'all'));
}
const ownView = v => !!whoName() && whoName().toLowerCase() === String(v.saved_by).toLowerCase();
// a saved view's chip; its owner has ⋯ beside it, to rename or delete it
function viewChip(v) {
  const on = lbIsView(v);
  const chip = el('button', { class: 'chip-btn saved' + (on ? ' on' : ''),
    'data-saved-view': String(v.id), 'aria-pressed': String(on), title: `saved by ${v.saved_by}`,
    text: v.name, onclick: () => { const sp = v.spec || {};
      lbSet({ chip: sp.chip || 'all', cols: lbKnownCols(sp.cols || []),
              models: sp.models && sp.models.length ? sp.models : null }); } });
  if (!ownView(v)) return chip;
  return el('span', { class: 'savedchip' }, chip,
    popover(el('button', { class: 'chip-more', 'data-view-menu': String(v.id),
        'aria-label': `${v.name}: rename or delete`, text: '⋯' }),
      () => el('div', { class: 'moremenu', id: 'pop-view-' + v.id, 'aria-label': v.name },
        el('button', { role: 'menuitem', 'data-view-rename': String(v.id), text: 'Rename…',
          onclick: () => { popClose(true); state.lbForm = 'rename:' + v.id;
            state.lbViewName = v.name; state.after = { focus: '[data-view-name]' }; render(); } }),
        el('button', { role: 'menuitem', 'data-view-delete': String(v.id), text: 'Delete…',
          onclick: () => { popClose(true); state.lbForm = 'delete:' + v.id; render(); } })),
      { key: 'view-' + v.id }));
}
// the one form under the chips: save this view, rename one, or delete one
function lbViewForm() {
  const f = state.lbForm;
  if (!LIVE || !f) return '';
  const L = lbS();
  const close = () => { state.lbForm = null; render(); };
  const cancel = el('button', { class: 'quiet', text: 'Cancel', onclick: close });
  const by = () => { if (!whoName()) throw new Error(askName()); return whoName(); };
  // the name typed so far lives in state: the click on Save redraws the form
  const nameBox = (label, key) => [el('label', { class: 'small', for: 'view-name', text: label }),
    el('input', { id: 'view-name', 'data-view-name': '1', 'data-keep': key, maxlength: '60',
      placeholder: 'Phone shortlist', value: state.lbViewName || '',
      oninput: e => { state.lbViewName = e.target.value; },
      onkeydown: e => { if (e.key === 'Enter') { e.preventDefault();
        const b = e.target.closest('.viewform').querySelector('[data-action]'); if (b) b.click(); } } })];
  if (f === 'save') {
    if (!L.cols && !L.models) { state.lbForm = null; return ''; }
    return el('div', { class: 'viewform', 'data-view-form': 'save' },
      ...nameBox('Name this view', 'viewname'),
      actButton('view-save', 'Save for the team', async () => {
        const name = state.lbViewName || '';
        const v = await sendView('api/views', 'POST', { name, by: by(),
          spec: { chip: L.chip, cols: L.cols, models: L.models } });
        state.lbForm = null;
        await loadViews();
        return { toast: `Saved “${v.name}” for the team — it is a chip after the groups` };
      }), cancel, actNote('view-save'));
  }
  const [what, id] = f.split(':');
  const v = (state.views || []).find(x => String(x.id) === id);
  if (!v) { state.lbForm = null; return ''; }
  if (what === 'rename')
    return el('div', { class: 'viewform', 'data-view-form': 'rename' },
      ...nameBox(`Rename “${v.name}”`, 'viewname:' + v.id),
      actButton('view-rename', 'Rename', async () => {
        const name = state.lbViewName || '';
        const out = await sendView('api/views/' + v.id, 'PATCH', { name, by: by() });
        state.lbForm = null;
        await loadViews();
        return { toast: `Renamed to “${out.name}”` };
      }), cancel, actNote('view-rename'));
  return el('div', { class: 'viewform', 'data-view-form': 'delete' },
    el('span', { class: 'small', text: `Delete “${v.name}” for the whole team?` }),
    actButton('view-delete', 'Delete', async () => {
      await sendView('api/views/' + v.id, 'DELETE', { by: by() });
      state.lbForm = null;
      await loadViews();
      return { toast: `Deleted “${v.name}”` };
    }, { class: 'danger' }), el('button', { class: 'quiet', text: 'Keep it', onclick: close }),
    actNote('view-delete'));
}
// post(), for the verbs a view also needs
async function sendView(path, method, body) {
  if (method === 'POST') return post(path, body);
  const r = await fetch(path, { method, headers: { 'Content-Type': 'application/json',
    'X-Token': TOKEN }, body: JSON.stringify(body) }).catch(() => null);
  if (!r) throw new Error('the server is unreachable — it may be restarting');
  checkBuild(r);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail
    : `the server answered HTTP ${r.status}`);
  return j;
}
// the team's saved views: once the scores are in, after a change, and every
// half minute on Models. A nicety, not the board: a failed fetch leaves the
// chips as they were and is not one of the failures the network line counts
let VIEWS_AT = 0;
async function loadViews() {
  if (!LIVE || !netReady()) return;
  VIEWS_AT = Date.now();
  try {
    const r = await fetch('api/views');
    if (!r.ok) return;
    const j = await r.json();
    const was = JSON.stringify(state.views || []);
    state.views = j.views || [];
    if (JSON.stringify(state.views) !== was && DATA) render();
  } catch (e) { /* the chips stay as they were */ }
}

// 12h.2: the table as shown — its rows and columns, the average and each
// ± error beside its number, in the table's own units
function lbCsv() {
  const t = state.lbTable;
  if (!t) return '';
  const q = x => { const s = String(x ?? ''); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  const pctn = v => (100 * v).toFixed(1);
  const hasSe = c => ['avg', 'cavg'].includes(c.key) || (!!c.task && !c.lower) || !!c.area;
  const se = (m, c) => {
    if (c.key === 'cavg') { const a = customAvg(m, lbS().cols); return a && a.se != null ? pctn(a.se) : ''; }
    if (c.key === 'avg') { const x = officialSe(m); return officialAvg(m) != null && x != null ? pctn(x) : ''; }
    if (c.area) { const r = areaMmlu(m, c.area);
      return r ? (100 * r.se * (state.avgMode === 'raw' ? 1 : 1 / 0.75)).toFixed(1) : ''; }
    // as the cell shows it: no ± where the harness gave none
    const cc = cell(c.task, m.id);
    return cc && cc.se ? pctn(cc.se) : '';
  };
  const text = (m, c) => {
    if (c.key === 'rank') { const r = t.custom ? customRankOf(m, lbS().cols) : rankOf(m); return r ? r.n : ''; }
    if (c.key === 'params') return m.params == null ? '' : P(m.params);
    if (c.key === 'date') return String(lastEval(m) || '').slice(0, 10);
    const v = t.val(m, c);
    if (v == null) return '';
    if (typeof v === 'string') return v;
    if (c.judged || c.jarea) return num(v, 2);
    if (c.lower) return num(v, 3);
    if (c.area) return pctn(areaScaled(v));
    return pctn(v);
  };
  const head = t.cols.flatMap(c => {
    const name = c.short || c.label;
    return hasSe(c) ? [name, name + ' ±'] : [name];
  });
  const lines = [head.map(q).join(',')];
  for (const m of t.rows)
    lines.push(t.cols.flatMap(c => hasSe(c) ? [text(m, c), se(m, c)] : [text(m, c)]).map(q).join(','));
  return lines.join('\n') + '\n';
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
    el('p', { class: 'sub', text: 'Click a column to sort, and a row for the model\'s page. Each '
      + 'cell is a score and its standard error, on one line; the unit is in the header. '
      + 'Perplexity columns are lower-is-better, excluded from Avg, and carry no standard error. '
      + '● marks a column\'s best value and ≈ marks values the z-test cannot tell from it. '
      + 'The tint is each cell\'s rank in its column, over the whole board, in five steps — '
      + 'filtering never changes a colour. '
      + `Avg exists only for models that completed all ${DATA.required.length} required tasks `
      + `(${DATA.required.join(', ')}): ${nOff} of ${ms.length} here. Anything short of that is `
      + 'preliminary — its per-task scores are valid and shown, it just has no overall number. '
      + 'What each benchmark measures is under Benchmarks ▸ Standard.' }));
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
      text: 'No model has been judged yet — sit the exam from a model\'s page' }));
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
  // 12b: each model's run provenance is its page's History tab. What is true
  // of the whole board — the harness and the library that ran it — stays here
  frag.push(el('div', { class: 'card', 'data-builds-card': '1' },
    el('h2', { text: 'Run provenance' }),
    el('div', { class: 'kvs', 'data-builds': '1' },
      DATA.meta.hashes.length ? el('span', {}, el('b', { text: 'harness ' }),
        el('span', { class: 'mono', text: DATA.meta.hashes.join(', ') })) : '',
      DATA.meta.transformers ? el('span', {}, el('b', { text: 'transformers ' }),
        DATA.meta.transformers) : ''),
    el('p', { class: 'sub', text: 'Each model\u2019s run provenance — architecture, dtype, '
      + 'template, seed, harness — is on its page, under History. Every field there can '
      + 'change a score.' })));
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
    state.trLoaded = true;
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
        if (failedEvs) statusLine += `, ${failedEvs} failed (see All runs)`;
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
                    'proposal', 'everyday'];

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
    } else if (r.kind === 'everyday') {
      // 12a: already on the page — the answers are in the payload
      data = { model: r.id };
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
                  log: readLog, provenance: readProvenance, proposal: readProposal,
                  everyday: readEveryday }[r.kind];
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
    + (page.fmt === 'free' ? ' · question and answer' : page.fmt === 'chat'
      ? ' · chat examples, each passed its own checks' : ''),
    head.over_provisional_judge || pv.provisional
      ? el('span', { class: 'badge warn', 'data-demo-only': '1', text: 'Demo only',
          title: [pv.provisional_reason, head.over_provisional_judge
            ? 'proposed from a judge no person has checked yet' : ''].filter(Boolean).join(' · ') })
      : '');
  readActs(wrap, { copy: () => { const d = docs.find(x => x.n === (state.read || {}).n) || docs[0] || {};
      return d.user ? `User: ${d.user}\n\nAssistant: ${d.assistant}`
        : d.title ? `${d.title}\n\n${d.text || [d.question, d.answer, d.rationale].join('\n\n')}` : ''; },
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
      // 12g.2: a chat example — the request, the reply, and the checks it passed
      page.fmt === 'chat'
        ? el('div', { class: 'rd-qa', 'data-chat-example': String(k) },
            el('div', { class: 'eyebrow', text: 'Request' }), ...paras(d.user, q),
            el('div', { class: 'eyebrow', text: 'Reply' }), ...paras(d.assistant, q),
            el('div', { class: 'eyebrow', text: 'Passed its checks' }),
            el('p', { class: 'small', 'data-chat-checks': String(k),
              text: (d.checks || []).join(' · ') || '—' }))
      : page.fmt === 'free'
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
  // 12a: a pilot run opens on its answers — on the model page, or the pilot's
  if (r.suite === 'everyday') {
    if (DATA.models.some(m => m.id === r.hf_id)) {
      (state.mblk[r.hf_id] = state.mblk[r.hf_id] || {}).everyday = true;
      state.mtab = 'scores';
      state.after = { scroll: '[data-kind-block="everyday"], [data-kind-tile="everyday"]' };
      return navigate({ model: r.hf_id, topic: null });
    }
    return navigate({ tab: 'everyday', model: null, topic: null });
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
// 12a: and a pilot run, whose TL;DR waits on the judge
const stillGrading = r => (r.suite === 'judged' || r.suite === 'everyday') && r.judge
  && r.judge.status !== 'done' && r.judge.status !== 'failed' && !r.judge_failed;
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

// 12b: the form opens in the Test a model dialog, the list is All runs —
// the same code, in two containers
function vQueue(part = { form: true, list: true }) {
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
        placeholder: 'search Hugging Face or uploads' }),
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
            + 'judged score per topic.' }],
      // 12a: the pilot. 12c replaces this drop-down with cards
      ['everyday', `Everyday tasks — ${evdAll() || 111} questions, a few minutes`],
      // 12h.1: instruct models only; MMLU-Pro alone is hours
      ['generative', 'Instruction & maths — IFEval, MMLU-Pro, MATH-500, hours',
        { sub: 'Asked through the chat template and scored on what the model writes. '
          + 'Instruct models only; never in the average.' }]],
      sf.suite || 'full', v => { sf.suite = v; render(); }, { key: 'submit-suite' }),
    note: el('input', { type: 'text', placeholder: 'note (optional)', style: 'flex:1;min-width:140px',
      'aria-label': 'note', 'data-keep': 'submit-note', value: sf.note,
      oninput: e => { sf.note = e.target.value; } }),
  };
  // 12h.1: Instruction & maths asks two more things, and only then — whether
  // the model thinks first (a model with a switch only; off by default, and a
  // row of its own), and a seeded MMLU-Pro subset (off: the full 12,032 is the
  // only run comparable to published numbers)
  const canThink = thinkingModeOf(sf.hf_id.trim()) === 'switch';
  if (!canThink) sf.thinking = false;
  const genOpts = sf.suite === 'generative' ? el('div', { class: 'genopts', 'data-gen-opts': '1' },
    canThink ? el('label', { class: 'spread', 'data-think-switch': '1' },
      el('input', { type: 'checkbox', checked: sf.thinking ? '' : null,
        onchange: e => { sf.thinking = e.target.checked; } }),
      ' Think before answering', el('span', { class: 'small se',
        text: ' — off by default; its scores are a row of their own' })) : '',
    el('label', { class: 'spread small' }, 'MMLU-Pro subset ',
      el('input', { type: 'number', min: '0', max: '12031', step: '100', placeholder: 'all',
        'aria-label': 'MMLU-Pro subset', 'data-subset-input': '1', value: sf.subset || '',
        style: 'width:7em', oninput: e => { sf.subset = parseInt(e.target.value, 10) || 0; } }),
      el('span', { class: 'se', text: ' items, seeded — empty runs all 12,032' }))) : '';
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
  // before Start test — the box to allow it, or why this server will not
  const info = codeInfo(sf.hf_id);
  const ownWhy = el('span', { class: 'propwhy', 'data-why': 'own-code' });
  const btn = el('button', { class: 'primary', text: 'Start test', onclick: async () => {
    const body = { hf_id: sf.hf_id.trim(), kind: sf.kind, suite: sf.suite,
                   submitter: whoName(), note: sf.note };
    if (sf.suite === 'generative') {
      if (sf.thinking) body.thinking = true;
      if (sf.subset) body.subset = sf.subset;
    }
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
        state.testOpen = false;                       // 12b: the dialog's job is done
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
      r.judge && !r.judge_failed && r.suite !== 'everyday' ? el('div', { class: 'se',
        'data-judge-progress': judgeCount(r.judge),
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
  const qEmpty = empty('Nothing has run yet. Test a model — it runs here, one '
    + 'at a time.', 'Test a model', () => {
      openTest(); });
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
  if (part.list) {
    state.queueRedraw = rebuildQueue;
    rebuildQueue();
  }
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
    part.form ? el('div', { class: 'card', 'data-submit-form': '1' },
      el('h2', { id: 'test-title', text: 'Test a model' }),
      // 12i.0: one line, no system words
      el('p', { class: 'sub', 'data-suite-help': '1', text: 'Pick a model and what to test. '
        + 'One test runs at a time; results appear on Models.' }),
      el('div', { class: 'frm' }, f.hf_id, f.kind, f.suite, f.note, btn,
        cannotRun(sf.hf_id)
          ? el('span', { class: 'propwhy', 'data-why': 'weights',
                         text: noWeightsWhy(sf.hf_id.trim()) })
          : sf.suite === 'judged' && judgeDown()
          ? el('span', { class: 'propwhy', 'data-why': 'submit', text: judgeWhy() }) : '',
        ownWhy),
      ownCodeBox(info, sf.allow, v => { sf.allow = v; gateSubmit(); }, 'submit'),
      topicBoxes, genOpts,
      state.qmsg ? el('p', { class: 'warn', 'data-qmsg': '1', style: 'margin-top:8px',
        text: state.qmsg }) : '') : null,
    part.list ? el('div', { class: 'card', 'data-all-runs': '1' },
      el('h2', { text: 'All runs' }),
      qToolbar, qPager, qTableWrap, qEmpty) : null].filter(Boolean);
}

// ---------------------------------------------------------------------------
// 12b: All runs — today's Queue table, unchanged, with its filters. Not in
// the header's places: reached from the run counter and every "follow it →".
// ---------------------------------------------------------------------------
function vAllRuns() {
  return vQueue({ list: true });
}

// Test a model: the one filled button on every page. It opens today's Submit
// form in a dialog, unchanged — 12c replaces its insides with four cards.
// From a model page it opens with that model filled in.
function openTest(prefill) {
  if (!LIVE) return;
  const id = prefill || state.model;
  if (id) {
    const m = DATA.models.find(x => x.id === id);
    // 12h.1: a thinking row is its model, asked to think
    const think = / · thinking$/.test(id);
    Object.assign(state.sub, { hf_id: id.replace(/ · thinking$/, ''), allow: false,
      ...(m && m.kind && m.kind !== 'checkpoint' ? { kind: m.kind } : {}),
      ...(think ? { suite: 'generative', thinking: true } : {}) });
  }
  state.testOpen = true;
  state.qmsg = '';
  state.after = { focus: '[data-dialog="test"] [data-ms="submit"] input' };
  render();
}
function closeTest() {
  if (!state.testOpen) return;
  state.testOpen = false;
  state.after = { focus: '[data-test-model]' };
  render();
}
function testDialog() {
  if (!state.testOpen || !LIVE) return [];
  const [form] = vQueue({ form: true });
  // the backdrop closes it on a click, and is not itself a control
  const back = el('div', { class: 'dlg-back', 'data-dialog': 'test' },
    el('div', { class: 'dlg testdlg', role: 'dialog', 'aria-modal': 'true',
        'aria-labelledby': 'test-title' },
      el('button', { class: 'ghost dlg-x', 'data-dialog-close': '1', 'aria-label': 'close',
        text: '✕ Close', onclick: closeTest }),
      form));
  back.addEventListener('click', e => { if (e.target === back) closeTest(); });
  return [back];
}
// Esc closes it — unless a list of its own is open, or the reader is
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && state.testOpen && !POP.panel && !state.read) {
    e.preventDefault(); closeTest(); }
});
function renderTestAct() {
  const box = document.getElementById('testAct');
  if (!box || !LIVE) return;
  if (!box.firstChild)
    box.append(el('button', { class: 'primary', 'data-test-model': '1',
      onclick: () => openTest() }, el('span', { class: 't-full' }),
      el('span', { class: 't-short', text: 'Test' })));
  // 12b.3: on a model page it is that model's — the dialog opens with it
  // filled in (openTest takes state.model) — and it says so
  const words = state.model ? 'Test this model' : 'Test a model';
  const btn = box.firstChild;
  btn.title = words;
  if (state.model) btn.dataset.testThis = state.model;
  else delete btn.dataset.testThis;
  btn.querySelector('.t-full').textContent = words;
}

// ● n running: a pulsing dot while anything runs, "Runs" when nothing does.
// It opens the running and queued runs with their progress, then the last
// five that finished, each linking to its model; All runs → at the bottom.
const RUNNING_ST = new Set(['preflight', 'waiting_gpu', 'waiting_lock', 'running', 'canceling']);
function runsNow() {
  const q = state.queue || [];
  const running = q.filter(r => RUNNING_ST.has(r.status) || (r.status === 'done' && stillGrading(r)));
  const queued = q.filter(r => r.status === 'queued');
  const done = q.filter(r => !running.includes(r) && !queued.includes(r))
    .sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0)).slice(0, 5);
  return { running, queued, done };
}
function runLine(r, attrs = {}) {
  const st = runStage(r);
  const onBoard = DATA.models.some(m => m.id === r.hf_id);
  const name = (DATA.models.find(m => m.id === r.hf_id) || {}).name || r.hf_id.split('/').pop();
  return el('div', { class: 'runline', 'data-run-line': String(r.id), ...attrs },
    el('span', { class: stClass(st.cls), text: st.text }),
    onBoard ? el('a', { href: '#model=' + encodeURIComponent(r.hf_id), class: 'runname', text: name,
        onclick: e => { e.preventDefault(); popClose(); navigate({ model: r.hf_id, topic: null }); } })
      : el('span', { class: 'runname', title: r.hf_id, text: name }),
    el('span', { class: 'small se runwhat', text: r.suite === 'everyday' ? 'everyday tasks'
      : r.suite }),
    el('span', { class: 'small se runprog', title: r.progress || '',
      text: RUNNING_ST.has(r.status) || r.status === 'done' ? (r.progress || '')
        : r.status === 'queued' ? 'waiting its turn' : (r.error || '') }));
}
function runsList(full = false) {
  const { running, queued, done } = runsNow();
  const all = el('a', { href: '#tab=runs', class: 'runs-all', 'data-all-runs-link': '1',
    text: 'All runs →', onclick: e => { e.preventDefault(); popClose();
      navigate({ tab: 'queue', model: null, topic: null }); } });
  return el('div', { class: full ? 'runslist full' : 'runslist', 'data-runs-list': '1' },
    running.length || queued.length
      ? [...running, ...queued].map(r => runLine(r))
      : el('p', { class: 'small', 'data-nothing-running': '1', text: 'Nothing running.' }),
    done.length ? [el('p', { class: 'small se runhead', text: 'Finished' }),
      ...done.map(r => runLine(r, { 'data-run-done': '1' }))] : '',
    full ? '' : all);
}
let _runsSig = null;
function renderRuns() {
  const box = document.getElementById('runs');
  if (!box || !LIVE) return;
  const { running } = runsNow();
  const n = running.length;
  const sig = String(n);
  if (_runsSig === sig && box.firstChild) return;
  _runsSig = sig;
  const btn = el('button', { class: 'barpill runpill' + (n ? ' on' : ''), 'data-runs': String(n),
    'aria-label': n ? `${n} running — runs` : 'runs', title: n ? `${n} running` : 'no run is going',
  }, el('span', { class: 'dot ' + (n ? 'ok pulse' : 'idle') }),
    n ? [el('span', { class: 'num', text: String(n) }), el('span', { class: 't-full', text: ' running' })]
      : el('span', { class: 't-idle', text: 'Runs' }));
  box.replaceChildren(popover(btn, () => el('div', { class: 'moremenu runspop', id: 'pop-runs',
      'aria-label': 'runs' }, runsList()), { key: 'runs', menu: false, placement: 'bottom-end',
    rebuild: true }));
}

// ---------------------------------------------------------------------------
// 12b: Help — the guides and How to read these numbers, from the old hero
// and Overview. Settings and reference: reached from the name menu.
// ---------------------------------------------------------------------------
function vHelp() {
  return [
    LIVE ? el('div', { class: 'card', 'data-help-guides': '1' },
      el('h2', { text: 'Help' }),
      el('p', { class: 'sub', text: 'What this board is for and how to use it.' }),
      el('ul', { class: 'helplinks' },
        el('li', {}, el('a', { href: 'guide', target: '_blank', rel: 'noopener',
          'data-help-guide': '1', text: '📖 The guide for new users ↗' })),
        el('li', {}, el('a', { href: 'guide#the-loop', target: '_blank', rel: 'noopener',
          'data-help-loop': '1', text: 'The loop, explained ↗' })))) : '',
    el('div', { class: 'card', 'data-how-to-read': '1' },
      el('h2', { text: 'How to read these numbers' }),
      note('Chance is not zero. 4-option tasks (MMLU, ARC, HellaSwag) sit at 25% for a model that knows nothing; 2-option tasks (Winogrande, PIQA) sit at 50%. A "50%" that looks respectable may be a coin flip.'),
      note('GSM8K near zero is a finding, not a failure — sub-billion models mostly cannot do written arithmetic. TruthfulQA is famous for NOT improving with scale.'),
      note('Perplexity (bits per byte) is the scale-sensitive metric here: it separates models that multiple-choice tasks cannot tell apart, and it works on base models with no prompt format at all. Multiply by ln 2 for cross-entropy loss in nats/byte — Models ▸ Standard ▸ Language modelling does it for you. Lower is better.'),
      note('Whiskers are ±1 standard error. If two whiskers overlap, do not call a winner — every pairwise z-test verdict rides along in the JSON export (the "sig" field) when you need the arbiter.'))]
    .filter(Boolean);
}

// Benchmarks ▸ Standard: today's Tasks page, with About these benchmarks
// under it (from the Leaderboard's How to read this table)
function vStandardBench(ms) {
  return [...vTasks(ms), aboutBenchmarks([...DATA.accTasks, ...DATA.pplTasks])];
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
    if (changed && (state.tab === 'pipeline' || onHome()) && !state.model) render();
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
    markDataset(made);
    await loadReview();
    // 12g.1: to the pipeline of the proposal's model, where it lands in Training data
    const pr = (state.rv.proposals || []).find(x => String(x.id) === String(pid));
    if (pr) state.imp.model = pr.model;
    navigate({ tab: 'pipeline', model: null, topic: null, read: null });
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
    if (state.tab === 'pipeline') render(); } }, LANDED_MS);
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
  if (!box) return;
  // a results refresh redraws the header: never under someone's caret
  if (!force && box.contains(document.activeElement)
      && document.activeElement.tagName === 'INPUT') return;
  const name = LIVE ? whoName() : '';
  const sig = `${LIVE}|${name}|${state.whoAsk}`;
  if (!force && box._sig === sig && box.firstChild) return;
  box._sig = sig;
  // one control in the bar, whatever the state: a button that opens the same
  // panel. It is never clipped, it survives a poll, and Esc gives it back.
  // 12b: and it holds what is settings and reference, not a place — the name
  // at the top, then Theme, Data & sources and Help
  const btn = el('button', { class: 'who barpill' + (LIVE && !name ? ' ask' : ''),
    'data-who': name, 'data-who-prompt': LIVE && !name ? '1' : null,
    title: LIVE ? 'your name, the theme, data & sources, and help'
      : 'the theme, data & sources, and help',
    'aria-label': LIVE ? (name ? `${name}: name, theme, data and help` : 'who are you?')
      : 'settings and help',
  }, el('span', { class: 't-full', text: LIVE ? (name ? `${name} ▾` : 'Who are you? ▾') : 'Settings ▾' }),
    // at a phone's width, the initial: "Wh…" said nothing
    el('span', { class: 't-short', 'aria-hidden': 'true',
      text: LIVE ? (name ? name[0].toUpperCase() : '?') + ' ▾' : 'Settings ▾' }));
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
    const labels = { auto: 'Auto', light: 'Light', dark: 'Dark', dim: 'Dim' };
    const go = tab => { popClose(); navigate({ tab, model: null, topic: null }); };
    return el('div', { class: 'moremenu whopop' + (state.whoAsk ? ' ask' : ''),
        id: 'pop-who', 'aria-label': LIVE ? 'your name, theme, data and help' : 'settings' },
      LIVE ? [el('p', { class: 'small', text: 'Your name is recorded on anything you start, '
          + 'approve or import here — this tailnet has no login, so the name you type is the '
          + 'record.' }),
        el('div', { class: 'frm' }, input,
          el('button', { class: 'primary', 'data-who-save': '1', text: 'Save', onclick: save }))]
        : '',
      el('div', { class: 'menusect', role: 'group', 'aria-label': 'theme' },
        el('span', { class: 'small se', text: 'Theme' }),
        el('div', { class: 'themeopts' }, THEMES.map(t => el('button', {
          role: 'menuitemradio', 'data-theme': t, class: 'chip-btn',
          'aria-checked': String(THEMES[themeIdx] === t), text: labels[t],
          onclick: () => { themeIdx = THEMES.indexOf(t); themeFade(); applyTheme(t); } })))),
      // 12i.1: which AI model does each job, and the judge test
      LIVE ? el('button', { role: 'menuitem', class: 'menulink', 'data-menu': 'ai',
        text: 'AI models', onclick: () => go('ai') }) : '',
      el('button', { role: 'menuitem', class: 'menulink', 'data-menu': 'data',
        text: 'Data & sources', onclick: () => go('provenance') }),
      el('button', { role: 'menuitem', class: 'menulink', 'data-menu': 'help',
        text: 'Help', onclick: () => go('help') }));
    // the name first: the checked theme would otherwise take the focus
  }, { key: 'who', menu: false, placement: 'bottom-end', focus: '[data-who-input]' }));
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

// 12i.0: one line about the AI, in plain words. 12i.1: each job's model —
// "AI: judge DeepSeek V4.1 Flash · writer GLM 5.3 · change". `writer` is the
// job that writes here: training data on Improve, questions on the exam
function aiLine(llm, writer = 'data') {
  const ai = llm.ai || {};
  return el('span', { 'data-ai-line': writer },
    `AI: judge ${ai.judge || 'none'} · writer ${ai[writer] || 'none'}`,
    llm.ai_waiting ? ` · ${llm.ai_waiting}` : '',
    LIVE ? [' · ', el('a', { href: '#tab=ai', 'data-ai-change': '1', text: 'change',
      onclick: e => { e.preventDefault(); navigate({ tab: 'ai', model: null, topic: null }); } })]
      : '');
}

// the daily limit is per provider (a paid API's, or LOCAL_DAILY_ITEM_CAP):
// null is no limit, and the judge's spend is on the judge's line
const underCap = u => u.daily_cap == null || (u.usage_today || 0) < u.daily_cap;

// ---------------------------------------------------------------------------
// 11j: a proposal opens as a short card in the reader's sheet. 12g.1: the
// Review tab that listed them is Improve's pipeline now; these are its rows'
// words and the lists Home counts
// ---------------------------------------------------------------------------
function rvLists() {
  const props = state.rv.proposals || [];
  const is = (p, ...s) => s.includes(p.status);
  return { review: props.filter(p => is(p, 'proposed', 'pending')),
           ready: props.filter(p => is(p, 'approved')),
           datasets: state.rv.datasets || [],
           history: props.filter(p => is(p, 'rejected', 'failed')) };
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
// a dataset made over a provisional judge, or past one: Demo only
function dsAsProposal(d) {
  const pv = d.provenance || {};
  return { id: d.id, override: pv.proposed_over_provisional_judge || d.over_provisional_judge,
    evidence: { provisional: pv.provisional, provisional_reason: pv.provisional_reason } };
}
const dsDemoOnly = d => demoReasons(dsAsProposal(d)).length > 0;
function dsDemoBadge(d) {
  return demoBadge(dsAsProposal(d), 'ds' + d.id);
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

// a failed dataset's line that explains, and every reason one click away:
// under its row on the model page, under its item in Improve (12g.1)
function dsWhyBlock(d) {
  const miss = Array.isArray(((d.provenance || {}).items || {}).missing)
    ? d.provenance.items.missing : [];
  const open = !!(state.rv.dsWhyOpen || {})[d.id];
  const pg = paged('ds-why-' + d.id, miss, `${d.id}:${miss.length}`, render, 10);
  return [el('p', { class: 'warn', 'data-ds-why-line': String(d.id), text: dsWhy(d) }),
      miss.length ? el('details', { class: 'small', 'data-ds-details': String(d.id),
          open: open ? '' : null,
          ontoggle: e => { state.rv.dsWhyOpen = { ...(state.rv.dsWhyOpen || {}),
            [d.id]: e.target.open }; } },
        el('summary', { text: `Details — ${miss.length} item${miss.length === 1 ? '' : 's'}, `
          + 'one reason each ▸' }),
        el('ol', { class: 'dswhy', start: String(pg.from || 1) }, pg.rows.map(m =>
          el('li', { 'data-ds-missing': String(m.request ?? '') },
            `request ${m.request ?? '—'} · ${m.focus || 'no area'} · ${m.why}`))),
        pg.pager || '') : ''];
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
// 12g.2: what an Everyday proposal's AI read — the group's PRACTICE requests
// the model failed, each with its skill and why; never a hidden one
function evdReadBlock(p, ev) {
  const failed = ev.failed || [];
  return el('details', { class: 'small', 'data-evd-read-block': String(p.id) },
    el('summary', { text: `The practice requests it read (${failed.length}) ▸` }),
    el('ol', { class: 'evd-read' }, failed.map(f => el('li', { 'data-evd-read-q': f.id },
      el('span', { class: 'se', text: (f.skill || '—') + ' · ' }), f.prompt,
      el('div', { class: 'se', text: 'why it failed: ' + (f.reason || '—') })))));
}
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
    // 12g.2: an Everyday group's proposal read its failed PRACTICE requests
    const evdP = ev.kind === 'everyday';
    const unit = evdP ? 'examples' : 'documents';
    body.push(el('div', { class: 'dxh', text: 'Why' }));
    body.push(el('p', { class: 'small', 'data-why-line': String(p.id),
      text: evdP ? `${ev.practice_failed ?? '—'} of ${ev.practice_total ?? '—'} practice requests `
          + `failed · ${p.category} ` + (ev.hidden ? `${ev.hidden.passed} of ${ev.hidden.total}`
            : '—') + ' (hidden questions)'
        : `${ev.diagnose_weak ?? '—'} of ${ev.diagnose_items ?? '—'} practice answers scored `
        + `below 3 of 4 · ${p.category} score `
        + (ev.topic_score_report != null ? `${num(ev.topic_score_report, 2)} / 4` : '—')
        + ` (hidden questions)` }));
    // 3. the answers it read
    body.push(evdP ? evdReadBlock(p, ev) : rvAnswersBlock(p, ev, paint));
    // 4. where the documents go
    const chips = el('div', { class: 'focuschips', 'data-focus-plan': String(p.id),
      'data-plan-stage': p.status === 'proposed' ? 'decide' : 'generate' });
    const spreadBox = el('input', { type: 'checkbox', 'data-spread': String(p.id),
      checked: state.rv.spread[p.id] !== false ? '' : null,
      onchange: e => { state.rv.spread[p.id] = e.target.checked; } });
    const spreadLabel = el('label', { class: 'small spread' }, spreadBox,
      ` Spread the ${unit} over these`);
    const count = el('input', { type: 'number', value: String(state.rv.count[p.id] || 20),
      min: '1', max: '1000', style: 'width:80px', 'aria-label': `how many ${unit}`,
      oninput: e => { const n = +e.target.value;
        if (n >= 1 && n <= 1000) { state.rv.count[p.id] = n; focusChips(p, n, chips); } } });
    if (p.status === 'proposed' || p.status === 'approved') {
      body.push(el('div', { class: 'dxh', text: evdP ? 'Examples will cover, one skill a batch'
        : 'Documents will cover' }));
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
      // 12g.2: chat examples are an Everyday group's; documents an exam topic's
      const fmts = evdP ? [['chat', 'Chat examples']]
        : [['doc', 'Documents'], ['free', 'Q&A (for comparison)']];
      const fmtNow = fmts.some(([v]) => v === state.rv.fmt[p.id]) ? state.rv.fmt[p.id] : fmts[0][0];
      const fmt = Select('format', fmts, fmtNow, v => { state.rv.fmt[p.id] = v; },
        { key: 'rv-fmt-' + p.id });
      body.push(el('div', { class: 'frm rd-acts-row' }, count, fmt,
        el('button', { class: 'primary', 'data-generate': String(p.id), text: 'Generate',
          disabled: llmOk ? null : '',
          title: llmOk ? '' : (llm.reason || 'no AI is set up here to write documents'),
          onclick: () => rvPost(`api/proposals/${p.id}/generate`, { requester: whoName(),
            count: +count.value, fmt: fmts.some(([v]) => v === state.rv.fmt[p.id])
              ? state.rv.fmt[p.id] : fmts[0][0] }) }),
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
        { key: 'propose', go: () => openImprove(st.model), link: 'See it' });
      await loadReview();
      // from a topic page, the person stays on it; anywhere else, the new
      // proposal is in its model's Proposals (12g.1)
      if (pre.stay) { state.loop.loaded = false; loadLoop(); render(); return; }
      state.imp.model = st.model;
      if (state.tab === 'pipeline' && !state.model) render();
      else navigate({ tab: 'pipeline', topic: null, model: null });
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

// ---------------------------------------------------------------------------
// 12g.1: Improve is one pipeline for one model. Weak spots → Proposals →
// Training data → Retests, left to right: a count and five one-line items
// each, every item with one action. The exam is what it trains toward; the
// Standard benchmarks are never a target, and appear here once — the watch
// line under a retest, before → after.
// ---------------------------------------------------------------------------
state.imp = { model: null, more: {} };
const IMP_KEY = 'bench-improve-model';
const IMP_SHOW = 5;
// a model's judged topics on the current exam: the tasks with a score
const judgedTopics = m => ((DATA.judged || {}).exam || []).filter(t => {
  const jt = ((m.judge || {}).tasks || {})[t];
  return !!jt && pubScore(jt) != null;
});
// what Improve can work on: every model with a judged topic, or a proposal or
// dataset of its own — the most judged topics first
function impModels() {
  const own = new Set([...(state.rv.proposals || []), ...(state.rv.datasets || [])]
    .map(x => x.model));
  return DATA.models.filter(m => !m.duplicateOf && (judgedTopics(m).length || own.has(m.id)))
    .sort((a, b) => judgedTopics(b).length - judgedTopics(a).length || natCmp(a.name, b.name));
}
// the model shown: the address's, else the viewer's last, else the one with
// the most judged topics
function impModel() {
  const ms = impModels();
  const ok = id => !!id && ms.some(m => m.id === id);
  let id = state.imp.model;
  if (!ok(id)) {
    let saved = null;
    try { saved = localStorage.getItem(IMP_KEY); } catch (e) { /* private */ }
    id = ok(saved) ? saved : (ms[0] || {}).id || null;
  }
  state.imp.model = id;
  return DATA.models.find(m => m.id === id) || null;
}
function setImpModel(id) {
  state.imp.more = {};
  state.imp.model = id;
  try { localStorage.setItem(IMP_KEY, id); } catch (e) { /* private */ }
  navigate({ tab: 'pipeline', model: null, topic: null });
}
// Improve, opened on one model — from Home, a dataset, a proposal
function openImprove(mid) {
  if (mid) {
    state.imp.model = mid;
    try { localStorage.setItem(IMP_KEY, mid); } catch (e) { /* private */ }
  }
  state.rv.loaded = false;
  navigate({ tab: 'pipeline', model: null, topic: null });
}
const OPEN_PROPOSAL = ['proposed', 'pending', 'approved'];
// Weak spots: judged topics, weakest first, with no open proposal
// — each with what stops a proposal, when something does (the dialog's own
// reasons: too few hidden questions, no practice answer to read)
function impWeak(m) {
  const open = new Set((state.rv.proposals || []).filter(p => p.model === m.id
    && OPEN_PROPOSAL.includes(p.status)).map(p => p.task));
  const why = new Map(npTopics(m.id).map(x => [x.task, x.why]));
  return judgedTopics(m).filter(t => !open.has(t))
    .map(t => ({ task: t, v: pubScore(m.judge.tasks[t]), why: why.get(t) || '' }))
    .sort((a, b) => a.v - b.v);
}
const newestFirst = (a, b) => (b.created_at || 0) - (a.created_at || 0);
const impProposals = m => (state.rv.proposals || [])
  .filter(p => p.model === m.id && OPEN_PROPOSAL.includes(p.status)).sort(newestFirst);
const impDatasets = m => (state.rv.datasets || []).filter(d => d.model === m.id).sort(newestFirst);
// Retests: the checkpoints whose page says they were trained from this model
const impRetests = m => DATA.models.filter(c => !c.duplicateOf
  && (c.trainedFrom || {}).base === m.id);

// 12g.1: the Standard watch — the checkpoint against what it was trained from,
// over the benchmarks BOTH were tested on, averaged as 12h.2's "Avg of N" is.
// "dropped" only where the board's z-test calls a difference real: the
// averages', or any one benchmark's
function standardWatch(base, ck) {
  const has = (m, t) => { const c = cell(t, m.id); return !!c && c.v != null; };
  const all = lbBenchAll();
  if (!all.some(t => has(ck, t))) return { state: 'untested', who: ck };
  const both = all.filter(t => has(base, t) && has(ck, t));
  if (!both.length) return { state: 'untested', who: base };
  const a = customAvg(base, both), b = customAvg(ck, both);
  const real = (va, sa, vb, sb) => sa != null && sb != null
    && Math.abs(vb - va) / Math.sqrt(sa * sa + sb * sb || 1e-12) > 1.96;
  const avgDrop = b.v < a.v && real(a.v, a.se, b.v, b.se);
  const drops = both.map(t => {
    const ca = cell(t, base.id), cb = cell(t, ck.id);
    const row = (DATA.sig[t] || []).find(([x, y]) => (x === base.id && y === ck.id)
      || (x === ck.id && y === base.id));
    const sig = row ? !!row[4] : real(ca.v, ca.se, cb.v, cb.se);
    return { t, a: ca.v, b: cb.v, real: sig && cb.v < ca.v };
  }).filter(x => x.real).sort((x, y) => (y.a - y.b) - (x.a - x.b));
  return { state: avgDrop || drops.length ? 'dropped' : 'held', n: both.length, a, b, drops };
}
// to a model page's block of one kind of test
function goKind(mid, kind) {
  (state.mblk[mid] = state.mblk[mid] || {})[kind] = true;
  state.after = { scroll: `[data-kind-block="${kind}"]` };
  state.mtab = 'scores';
  try { localStorage.setItem('bench-model-tab', 'scores'); } catch (e) { /* private */ }
  navigate({ model: mid, topic: null });
}
function watchLine(base, ck) {
  const w = standardWatch(base, ck);
  const pc = v => (100 * v).toFixed(1);
  const mono = t => el('span', { class: 'mono', text: t });
  if (w.state === 'untested')
    return el('div', { class: 'small se watch', 'data-watch-line': ck.id, 'data-watch': 'untested' },
      w.who === ck ? 'Standard: not tested' : `Standard: ${base.name} not tested`,
      LIVE ? [' · ', el('a', { href: '#', 'data-watch-test': w.who.id, text: 'Test',
        onclick: e => { e.preventDefault(); state.sub.suite = 'full'; openTest(w.who.id); } })] : '');
  const drop = w.drops[0];
  return el('div', { class: 'small watch' + (w.state === 'dropped' ? ' dropped' : ''),
      'data-watch-line': ck.id, 'data-watch': w.state,
      title: `the average of the ${w.n} Standard benchmarks both were tested on, `
        + (state.avgMode === 'raw' ? 'raw accuracy' : 'above chance')
        + ' — "dropped" only where the z-test calls it real' },
    `Standard (${w.n}) `, mono(pc(w.a.v)), ' → ', mono(pc(w.b.v)), ' · ',
    w.state === 'dropped'
      ? el('a', { href: '#', class: 'watch-drop', 'data-watch-drop': ck.id,
          onclick: e => { e.preventDefault(); goKind(ck.id, 'standard'); } },
          'dropped', drop ? [' · ' + benchName(drop.t) + ' ', mono(pc(drop.a)), ' → ',
            mono(pc(drop.b))] : '')
      : 'no drop');
}

// one stage: a count, five items, "+ n more" opening the rest in place — or,
// with nothing in it, one line and no box
function impStage(key, title, items, none) {
  const more = !!state.imp.more[key];
  const shown = more ? items : items.slice(0, IMP_SHOW);
  return el('section', { class: 'stage', 'data-stage': key, 'data-stage-n': String(items.length) },
    el('h3', { class: 'stage-h' }, title + ' ', el('span', { class: 'mono se',
      text: `(${items.length})` })),
    items.length ? el('ul', { class: 'stagelist' }, shown,
      items.length > IMP_SHOW ? el('li', { class: 'stagemore' }, el('button', { class: 'quiet',
        'data-stage-more': key, 'aria-expanded': String(more),
        text: more ? 'show fewer' : `+ ${items.length - IMP_SHOW} more`,
        onclick: () => { state.imp.more[key] = !more; render(); } })) : '')
      : el('p', { class: 'small se stage-none', 'data-stage-none': key, text: none }));
}
const impItem = (attrs, main, sub, action) => el('li', { class: 'stageitem', ...attrs },
  el('div', { class: 'si-main' }, main), sub ? el('div', { class: 'si-sub' }, sub) : '',
  action ? el('div', { class: 'si-act' }, action) : '');
const PROP_WORDS = { proposed: 'waiting for you', pending: 'the AI is writing it',
  approved: 'approved — ready to generate' };

function vPipeline() {
  if (!state.rv.loaded && netReady()) loadReview();
  rememberedName();
  const m = impModel();
  // the address names the model it opened on — a first visit picks it here
  const want = hashFor();
  if (m && location.hash.slice(1) !== want) history.replaceState(history.state, '', '#' + want);
  const llm = state.rv.llm || {};
  // 12i.1: after a new judge, the answers on file wait for it — not "no exam"
  const earlier = DATA.models.some(x => x.judgedEarlier);
  if (!m) return [el('div', { class: 'card', 'data-pipeline': 'none' },
    el('h2', { text: 'Improve' }),
    earlier ? empty('Nothing to improve yet: the exam answers on file were judged by another '
      + 'judge, and a score compares only with its own judge’s.', 'Re-judge them',
      () => navigate({ tab: 'ai', model: null, topic: null }))
      : empty('Nothing to improve yet: no model has sat the Knowledge exam.', 'Sit the exam',
        () => navigate({ tab: 'exam', model: null, topic: null })))];
  const weak = impWeak(m), props = impProposals(m), ds = impDatasets(m), rts = impRetests(m);
  // the one filled button opens on the weakest topic a proposal can be made from
  const first = weak.find(w => !w.why);
  const picker = popover(el('button', { class: 'pill imp-model', id: 'pill-imp-model',
      'data-imp-model': m.id, text: m.name + ' ▾' }),
    () => el('div', { class: 'moremenu', id: 'pop-imp-model', 'aria-label': 'model to improve' },
      impModels().map(x => el('button', { role: 'menuitemradio', 'data-imp-pick': x.id,
        'aria-checked': String(x.id === m.id), onclick: () => { popClose(true); setImpModel(x.id); } },
        x.name, el('span', { class: 'se', text: ` · ${judgedTopics(x).length} judged topics` })))),
    { key: 'imp-model' });
  const head = el('div', { class: 'card imphead', 'data-pipeline': m.id },
    el('div', { class: 'rvbar' },
      el('h2', { class: 'imp-title' }, 'Improving: ', picker),
      el('button', { class: 'primary', 'data-imp-propose': '1', text: 'Propose',
        title: first ? `opens on ${frName(first.task)}, the weakest topic a proposal can be `
          + 'made from' : 'no weak spot a proposal can be made from yet',
        onclick: () => npDialog({ model: m.id, topic: first ? frName(first.task) : '',
          returnTo: '[data-imp-propose]' }) })),
    el('p', { class: 'sub', text: 'What the Knowledge exam and Everyday tasks say this model is '
      + 'missing, the data made for it, and what training changed. The Standard benchmarks are only watched here, '
      + 'never trained toward.' }),
    !llm.configured && llm.reason ? el('p', { class: 'warn', text: llm.reason }) : '',
    llm.configured ? el('p', { class: 'small se', 'data-imp-ai': '1' }, aiLine(llm)) : '',
    state.rv.msg ? el('p', { class: 'small', text: state.rv.msg }) : '');

  return [head, impStagesCard(m)];
}

// the four stages for one model — Improve's pipeline, and the model page's Improve tab
function impStagesCard(m) {
  const weak = impWeak(m), props = impProposals(m), ds = impDatasets(m), rts = impRetests(m);
  // 12g.2: the Everyday groups beside the exam topics, each labelled — the
  // weakest first within each, as fractions of their own scale; a group
  // under the line is one greyed line at the end, with no Propose
  const evw = impEvdWeak(m);
  const tag = k => el('span', { class: 'imp-kind', 'data-kind': k, text: k === 'exam' ? 'Exam'
    : 'Everyday' });
  const mixed = [...weak.map(w => ({ ...w, kind: 'exam', frac: w.v / 4 })),
                 ...evw.open.map(w => ({ ...w, kind: 'everyday' }))]
    .sort((a, b) => a.frac - b.frac);
  const weakItems = [...mixed.map(w => w.kind === 'exam'
    ? impItem({ 'data-weak': w.task, 'data-weak-kind': 'exam' },
      [tag('exam'), ' ', frName(w.task), ' ',
       el('span', { class: 'mono', text: `${num(w.v, 2)}/4` })],
      w.why ? el('span', { 'data-weak-why': w.task, text: w.why }) : '',
      w.why ? '' : el('button', { class: 'ghost', 'data-weak-propose': w.task, text: 'Propose',
        onclick: () => npDialog({ model: m.id, topic: frName(w.task),
          returnTo: `[data-weak-propose="${CSS.escape(w.task)}"]` }) }))
    : impItem({ 'data-weak': w.task, 'data-weak-kind': 'everyday' },
      [tag('everyday'), ' ', w.label],
      // the score on the line under the name: "8 of 24 hidden questions"
      [el('span', { class: 'mono', 'data-weak-score': w.task,
        text: `${w.x.passed} of ${w.x.total}` }), ' hidden questions',
       w.why ? el('span', { 'data-weak-why': w.task, text: ' · ' + w.why }) : ''],
      w.why ? '' : el('button', { class: 'ghost', 'data-weak-propose': w.task, text: 'Propose',
        onclick: e => impProposeEveryday(m, w.g, e.currentTarget) }))),
    ...evw.under.map(w => impItem({ 'data-weak': w.task, 'data-weak-kind': 'everyday',
        'data-weak-need': String(w.need), class: 'stageitem greyed' },
      [tag('everyday'), ` ${w.label} · needs ${w.need} more hidden question`
        + `${w.need === 1 ? '' : 's'} to improve on`], '', ''))];
  const propItems = props.map(p => impItem({ 'data-prop': String(p.id) },
    [tag(String(p.task).startsWith('everyday:') ? 'everyday' : 'exam'), ' ',
     p.category || frName(p.task), demoBadge(p)],
    el('span', { class: 'se', 'data-rv-status': String(p.id), text: PROP_WORDS[p.status] }),
    readButton({ kind: 'proposal', id: String(p.id) },
      p.status === 'approved' ? 'Generate' : p.status === 'proposed' ? 'Review' : 'Read',
      { 'data-prop-act': String(p.id) })));
  const dataItems = ds.map(d => impItem({ 'data-ds-item': String(d.id),
      class: 'stageitem' + (state.rv.landedDs === d.id ? ' landed' : '') },
    // a ready dataset's name opens it in the reader; its one action hands it on
    [d.status === 'ready' ? readLink({ kind: 'dataset', id: String(d.id) }, d.category || '—',
        { 'data-ds-read': String(d.id) }) : d.category || '—', ' · ',
     el('span', { 'data-doc-line': String(d.id),
      text: d.status === 'ready' || dsFailed(d) ? dsDocs(d) : 'being written' }), dsDemoBadge(d)],
    // failed: the count, the word, and why — every reason one click away (11m)
    dsFailed(d) ? [el('span', { class: 'badge danger', 'data-ds-failed': String(d.id),
      text: d.status === 'rejected' ? 'Rejected' : 'Failed' }), ...dsWhyBlock(d)] : '',
    // the one action, and the old row's ⋯: its provenance, the file, its id
    d.status === 'ready' ? actCell('ds-' + d.id, useInTraining(d), [
        { label: 'Provenance', act: 'provenance',
          run: () => openReader({ kind: 'provenance', id: 'dataset:' + d.id }) },
        d.download ? { label: 'Download', act: 'download',
                       href: `api/datasets/${d.id}/items.jsonl` } : null,
        { label: 'Copy dataset id', act: 'copy-id', run: () => copyText(String(d.id), '#' + d.id) }])
      : dsFailed(d) ? readButton({ kind: 'dataset', id: String(d.id) }, 'Why',
          { 'data-ds-read': String(d.id) }) : ''));
  const retestItems = rts.map(c => {
    const topics = (c.tainted || []).filter(t => t.startsWith('exam_'));
    // 12g.2: an Everyday group it trained on, before → after on the hidden half
    const groups = (c.tainted || []).filter(t => t.startsWith('everyday:'));
    const score = (x, t) => { const jt = ((x.judge || {}).tasks || {})[t];
      return jt && pubScore(jt) != null ? num(pubScore(jt), 2) : 'not sat'; };
    return impItem({ 'data-retest': c.id },
      [el('a', { href: '#model=' + encodeURIComponent(c.id), text: c.name }), ckBadge(c) || ''],
      [...topics.map(t => el('div', { class: 'small', 'data-retest-topic': t },
          frName(t) + ' ', el('span', { class: 'mono', text: `${score(m, t)} → ${score(c, t)}` }))),
       ...groups.map(t => evdRetestLine(m, c, t.slice(9))),
       ...(topics.length || groups.length ? []
         : [el('div', { class: 'small se', text: 'trained on no dataset made here' })]),
       watchLine(m, c)],
      el('button', { class: 'ghost', 'data-retest-compare': c.id, text: 'Compare',
        title: 'its page: what the training taught, against ' + m.name,
        onclick: () => { state.mtab = 'history';
          try { localStorage.setItem('bench-model-tab', 'history'); } catch (e) { /* private */ }
          navigate({ model: c.id, topic: null }); } }));
  });
  if (!state.rv.loaded) return el('div', { class: 'card', 'data-pipeline-stages': m.id },
    skeleton(4, { 'data-loading': 'pipeline' }));
  const stages = el('div', { class: 'stages', 'data-stages': '1' },
    impStage('weak', 'Weak spots', weakItems, judgedTopics(m).length || evdOf(m.id)
      ? 'Every judged topic and group has a proposal' : 'No judged topics yet'),
    impStage('proposals', 'Proposals', propItems, 'No proposals waiting'),
    impStage('data', 'Training data', dataItems, 'No training data yet'),
    impStage('retests', 'Retests', retestItems, 'No retests yet — a checkpoint shows here once '
      + 'its page says it was trained from ' + m.name));
  // what is no longer open: rejected and failed, folded
  const past = (state.rv.proposals || []).filter(p => p.model === m.id
    && ['rejected', 'failed'].includes(p.status)).sort(newestFirst);
  const pastFold = past.length ? el('details', { class: 'small imp-past', 'data-imp-past':
      String(past.length), open: state.imp.pastOpen ? '' : null,
      ontoggle: e => { state.imp.pastOpen = e.target.open; } },
    el('summary', { text: `Rejected or failed (${past.length}) ▸` }),
    el('ul', {}, past.map(p => el('li', {}, readLink({ kind: 'proposal', id: String(p.id) },
      p.category || frName(p.task)), el('span', { class: 'se', text: ' · '
        + rvStatusWords(p.status).toLowerCase() }))))) : '';
  return el('div', { class: 'card', 'data-pipeline-stages': m.id }, stages, pastFold);
}

// 12g.2: a model's Everyday groups for Weak spots — the hidden half's score,
// on this version of the questions; a group whose hidden half is under the
// line (EVERYDAY_MIN_HIDDEN) is `under`, and says how many more it needs
function impEvdWeak(m) {
  const e = evdOf(m.id);
  if (!e) return { open: [], under: [] };
  const min = evd().minHidden || 20;
  const open = new Set(impProposals(m).map(p => p.task));
  const items = evdGroups().filter(([g]) => (e.groups || {})[g] && !open.has('everyday:' + g))
    .map(([g, label]) => {
      const x = e.groups[g], pr = (e.practice || {})[g];
      return { task: 'everyday:' + g, g, label, x, frac: x.total ? x.passed / x.total : 1,
        need: Math.max(0, min - evdHidden(g)),
        why: pr && pr.total && pr.passed === pr.total ? 'no practice question failed' : '' };
    });
  return { open: items.filter(w => !w.need), under: items.filter(w => w.need) };
}
// Propose, for an Everyday group: no dialog to choose in — the group is chosen
async function impProposeEveryday(m, g, btn) {
  if (!whoName()) { askName(); return; }
  if (btn) { btn.disabled = true; btn.textContent = 'Proposing…'; }
  try {
    const j = await post('api/proposals', { model: m.id, everyday: g, requested_by: whoName() });
    toast(`Proposal #${j.id} requested — ${evdTaskName('everyday:' + g)}`, { key: 'propose' });
    state.rv.loaded = false;
    await loadReview();
  } catch (e) {
    toast('Refused. ' + String((e && e.message) || e), { key: 'propose' });
  }
  render();
}
// "Instructions 6 of 22 → 13 of 22": both on this version of the questions,
// or a line that says to retest (12g.2)
function evdRetestLine(m, c, g) {
  const name = evdTaskName('everyday:' + g).replace(/^Everyday · /, '');
  const a = ((evdOf(m.id) || {}).groups || {})[g], b = ((evdOf(c.id) || {}).groups || {})[g];
  const test = who => LIVE ? [' · ', el('a', { href: '#', 'data-evd-retest-test': who.id,
    text: 'Test', onclick: e => { e.preventDefault(); evdQueue([who.id]); } })] : '';
  if (a && b) return el('div', { class: 'small', 'data-retest-group': g },
    name + ' ', el('span', { class: 'mono', text: `${a.passed} of ${a.total} → ${b.passed} of `
      + `${b.total}` }));
  // one of them answered another version: no before → after across it
  const stale = !b ? (evdEarlier(c.id) ? c : null) : (evdEarlier(m.id) ? m : null);
  return el('div', { class: 'small se', 'data-retest-group': g,
      'data-retest-stale': stale ? stale.id : null },
    `${name}: ` + (stale ? 'retest on the current questions' : 'not tested'),
    test(stale || (!b ? c : m)));
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
    if (changed && (state.tab === 'queue' || state.topic) && !state.model) render();
    else if (changed && state.model && state.msitRedraw) state.msitRedraw();
  } catch (e) {
    // netFail already put the banner up and set the backoff; the board itself
    // must also stop saying "Loading…" forever, which is what it did
    state.loop.failed = String((e && e.message) || e);
    if (state.topic && !state.model) render();
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
  if ((state.tab === 'queue' || state.topic) && !state.model) render();
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
    if (r.proposal && r.proposal.model) state.imp.model = r.proposal.model;
    return navigate({ tab: 'pipeline', topic: null, model: null,
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
      'data-review-link': String(open.id), href: '#tab=improve&sub=review&read=proposal:' + open.id,
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
    el('a', { href: '#tab=benchmarks&sub=exam', 'data-topic-back': '1', text: '← Knowledge exam',
      onclick: e => { e.preventDefault(); navigate({ topic: null, tab: 'exam' }); } }));
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
      + 'anything is proposed from it. Import or write more on Benchmarks ▸ Knowledge exam.' })
      : '',
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
      el('a', { href: '#tab=runs', text: 'in All runs',
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
    el('p', { class: 'small' }, 'Every run is on ',
      el('a', { href: '#tab=runs', text: 'All runs',
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
  if (r.suite === 'everyday') return el('span', { 'data-suite-cell': key, text: 'everyday tasks' });
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
      + 'approves that sentence on Improve, and only the approved text reaches a '
      + 'generator.' }));
  if (r.proposal) {
    card.append(el('p', { class: 'small', 'data-proposal': String(r.proposal.id) },
      `Proposal #${r.proposal.id} for ${r.proposal.model} is ${r.proposal.status}`
      + (r.proposal.requested_by ? `, requested by ${r.proposal.requested_by}` : ''), ' ',
      overBadge(r.proposal.override), '. ',
      el('a', { href: '#tab=improve&sub=review', text: 'Review it',
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
  if (!state.rv.loaded && netReady()) loadReview();
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
    // 12i.1: the judge and the question writer, in words, and where to change them
    state.rv.llm ? el('p', { class: 'small se', 'data-exam-ai': '1' },
      aiLine(state.rv.llm, 'writer')) : '',
    el('div', { class: 'kvs' },
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
    // 12i.2: or have them written, checked and reviewed here
    LIVE ? el('div', { class: 'qbopen' }, el('button', { class: 'secondary', 'data-qb-open': 'knowledge',
      text: 'Build questions', onclick: () => openBuilder('knowledge') })) : '',
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
              state.ex.candidates = null; render(); } }),
            // 12g.1: the topic's own page — the By topic board that listed
            // them is Improve's pipeline now, so they are reached from here
            slugOfTopic(r.topic) ? [' ', el('a', { class: 'small', href: '#topic='
              + slugOfTopic(r.topic), 'data-topic-link': r.topic, text: 'page ▸' })] : ''),
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

// ===========================================================================
// 12i.1: AI models — which model does each job (the judge, the question
// writer, the training-data writer, the checker), what it costs, and the
// judge test that picks the judge. Under the name menu: settings, not a place
// ===========================================================================
const AI_DEFAULT_CANDIDATES = ['deepseek/deepseek-v4.1-flash', 'openai/gpt-6-luna',
  'z-ai/glm-5.3-flash', 'local'];
async function loadAi() {
  if (state.ai.asked) return;
  state.ai.asked = true;
  try { state.ai.page = await api('api/ai'); } catch (e) { state.ai.msg = e.message; }
  if (state.ai.page && state.ai.page.has_key && !state.ai.models) {
    try { state.ai.models = (await api('api/ai/models')).models; } catch (e) { /* the page stands */ }
  }
  state.ai.asked = false;
  if (state.tab === 'ai') render();
}
async function loadJudgeTest() {
  if (state.ai.jtAsked) return;
  state.ai.jtAsked = true;
  try {
    const [jt, res] = await Promise.all([api('api/judge-test'), api('api/judge-test/result')]);
    state.ai.jt = jt; state.ai.res = res;
  } catch (e) { state.ai.msg = e.message; }
  state.ai.jtAsked = false;
  if (state.tab === 'ai') render();
}
// "$0.14" — dollars per million tokens, or a month's spend
const usd = (v, d = 2) => v == null ? '—' : '$' + Number(v).toLocaleString('en',
  { minimumFractionDigits: d, maximumFractionDigits: Math.max(d, 4) });
const aiName = c => !c ? '—' : c.kind === 'local' ? `Local (${(state.ai.page || {}).local
  ? state.ai.page.local.name : 'the local model'} on this server)` : String(c.name || c.id)
  .split(': ').pop();
function aiPrice(j) {
  const c = j.chosen;
  if (c && c.kind === 'openrouter')
    return `${usd(c.price_in)} in · ${usd(c.price_out)} out`;
  if ((c && c.kind === 'local') || j.provider === 'local') return 'free';
  return '—';
}

// change ▾ — Local, then the suggested model, then OpenRouter's text models
function aiChange(j) {
  const A = state.ai;
  const btn = el('button', { class: 'quiet', 'data-ai-change-menu': j.job, text: 'change ▾',
    'aria-label': `change the ${j.label.toLowerCase()}'s model` });
  return popover(btn, () => {
    const list = el('div', { class: 'ailist' });
    const pick = async id => {
      if (!whoName()) { popClose(); askName(); return; }
      popClose(true);
      try {
        const r = await post(`api/ai/jobs/${j.job}`, { model: id, by: whoName() });
        A.page = r.page;
        if (r.rejudge && r.rejudge.n) A.confirm = r.rejudge;
        toast(`${j.label}: ${aiName(r.saved)}`, { key: 'ai' });
      } catch (e) { toast('Refused. ' + e.message, { key: 'ai' }); }
      render();
    };
    const item = (id, name, sub, attrs = {}) => el('button', { role: 'menuitem',
        class: 'aiitem', 'data-ai-pick': id, onclick: () => pick(id), ...attrs },
      el('span', { class: 'aiitem-name', text: name }), sub);
    const fill = q => {
      q = (q || '').trim().toLowerCase();
      const ms = (A.models || []).filter(m => !q || (m.id + ' ' + m.name).toLowerCase()
        .includes(q));
      const sug = ms.find(m => m.id === j.suggested);
      const rest = ms.filter(m => m !== sug).slice(0, 60);
      const row = (m, extra) => item(m.id, m.name.split(': ').pop(),
        el('span', { class: 'small se aiitem-sub' }, extra || '',
          el('span', { class: 'mono', text: `${usd(m.price_in)} in · ${usd(m.price_out)} out` }),
          m.context ? ` · ${Math.round(m.context / 1000).toLocaleString('en')}k context` : ''));
      list.replaceChildren(
        !q || 'local'.includes(q) ? item('local', `Local (${A.page.local.name} on this server)`,
          el('span', { class: 'small se aiitem-sub', text: 'free · no key needed' })) : '',
        sug ? row(sug, el('span', { class: 'badge', 'data-ai-suggested': sug.id,
          text: 'suggested' })) : '',
        sug ? el('p', { class: 'small se aiwhy', 'data-ai-why': j.job, text: j.why }) : '',
        ...rest.map(m => row(m)),
        !A.page.has_key ? el('p', { class: 'small se', text: 'OpenRouter’s models show once '
          + 'the server has its key.' }) : !ms.length && q ? el('p', { class: 'small se',
          text: 'No model matches.' }) : '');
    };
    fill(A.q || '');
    return el('div', { class: 'moremenu aimenu', id: 'pop-ai-' + j.job, 'aria-label': 'models' },
      A.page.has_key ? el('input', { type: 'search', placeholder: 'search models…',
        'aria-label': 'search models', 'data-keep': 'aiq', value: A.q || '',
        oninput: e => { A.q = e.target.value; fill(A.q); } }) : '',
      list);
  }, { key: 'ai-' + j.job, menu: false });
}

function aiJobsTable(P) {
  const rows = P.jobs.map(j => el('tr', { 'data-ai-job': j.job },
    el('td', {}, el('b', { text: j.label }), el('div', { class: 'small se', text: j.does })),
    el('td', {}, el('span', { 'data-ai-now': j.job, text: j.chosen ? aiName(j.chosen) : j.now }),
      j.chosen && j.chosen.kind === 'openrouter' ? el('div', { class: 'small se',
        'data-ai-provider': j.job, text: `on ${j.chosen.provider_name || j.chosen.provider}`
          + (j.chosen.precision && j.chosen.precision !== 'unknown' ? ` · ${j.chosen.precision}` : '')
          + ` · ${j.chosen.version}` }) : '',
      j.blocked ? el('div', { class: 'small warntext', 'data-ai-blocked': j.job, text: j.blocked })
        : ''),
    el('td', { class: 'mono small aiprice', 'data-ai-price': j.job, text: aiPrice(j) }),
    el('td', { class: 'aiact' }, LIVE ? aiChange(j) : '')));
  return el('table', { class: 'aijobs', 'data-ai-jobs': '1' },
    el('thead', {}, el('tr', {}, el('th', { text: 'Job' }), el('th', { text: 'Model' }),
      el('th', { text: 'Price per million tokens' }), el('th', {}))),
    el('tbody', {}, rows));
}

function aiSpendLine(P) {
  const A = state.ai, s = P.spend;
  const edit = A.editLimit;
  const input = el('input', { type: 'number', min: '0', step: '1', value: String(s.limit),
    'aria-label': 'monthly limit in dollars', 'data-ai-limit-input': '1', style: 'width:6em' });
  return el('div', { class: 'aispend', 'data-ai-spend': String(s.month) },
    el('p', { class: 'small' }, 'This month: ', el('span', { class: 'mono', text: usd(s.month) }),
      ' of ', el('span', { class: 'mono', 'data-ai-limit': String(s.limit), text: usd(s.limit) }),
      ' · ', edit ? el('span', {}, input, ' ', el('button', { class: 'quiet', text: 'Save',
        'data-ai-limit-save': '1', onclick: async () => {
          if (!whoName()) { askName(); return; }
          try { A.page = await post('api/ai/limit', { usd: Number(input.value), by: whoName() });
            A.editLimit = false; } catch (e) { toast('Refused. ' + e.message, { key: 'ai' }); }
          render(); } }))
        : el('button', { class: 'quiet', 'data-ai-limit-edit': '1', text: 'change the limit',
          onclick: () => { A.editLimit = true; render(); } })),
    s.waiting ? el('p', { class: 'warn', 'data-ai-waiting': '1', text: s.waiting }) : '');
}

// changing the judge asks first: re-judge what another judge marked, or later
function aiRejudgeBox() {
  const A = state.ai, c = A.confirm;
  return el('div', { class: 'note airejudge', 'data-ai-rejudge': String(c.n) },
    el('p', {}, `Re-judge the ${c.n.toLocaleString('en')} answer${c.n === 1 ? '' : 's'} on file `
      + `with the new judge? About ${usd(c.usd)}.`),
    el('p', { class: 'small se', text: 'It uses no GPU. Until then, the scores the last judge '
      + 'gave are in each model’s History, not in today’s tables.' }),
    el('div', { class: 'frm' },
      el('button', { class: 'primary', 'data-ai-rejudge-yes': '1', text: 'Re-judge them',
        onclick: async () => {
          try {
            const r = await post('api/ai/rejudge', { by: whoName() });
            toast(`${r.queued.length} run${r.queued.length === 1 ? '' : 's'} queued to re-judge `
              + `${r.n.toLocaleString('en')} answers`, { key: 'ai' });
            A.confirm = null;
          } catch (e) { toast('Refused. ' + e.message, { key: 'ai' }); }
          render(); } }),
      el('button', { class: 'quiet', 'data-ai-rejudge-later': '1', text: 'Later',
        onclick: () => { A.confirm = null; render(); } })));
}

function vAiModels() {
  const A = state.ai;
  if (!A.page && netReady()) loadAi();
  if (A.mark) return vJudgeMark();
  if (!A.page) return [el('div', { class: 'card' }, skeleton(4, { 'data-loading': 'ai' }))];
  const P = A.page;
  return [el('div', { class: 'card', 'data-ai-page': '1' },
      el('h2', { text: 'AI models' }),
      el('p', { class: 'sub', text: 'Which AI model does each job. A model chosen here is '
        + 'pinned — its dated version and the provider running it — so its marks cannot change '
        + 'quietly. Before one is chosen, a job keeps the model the server was set up with.' }),
      P.has_key ? '' : el('p', { class: 'note', 'data-ai-no-key': '1', text: 'OpenRouter has no '
        + 'key on this server, so only the local model is offered. Add OPENROUTER_API_KEY to '
        + 'the server’s .env to choose others.' }),
      aiSpendLine(P),
      ...(P.warnings || []).map(w => el('p', { class: 'warn', 'data-ai-warning': w.job,
        text: w.text })),
      A.confirm ? aiRejudgeBox() : '',
      aiJobsTable(P)),
    judgeTestCard()];
}

// ---- the judge test ----------------------------------------------------------
const jtPct = v => v == null ? '—' : `${Math.round(100 * v)}%`;
function judgeTestCard() {
  const A = state.ai;
  if (!A.jt && netReady()) loadJudgeTest();
  const jt = A.jt, res = A.res;
  const pr = jt ? jt.progress : null;
  const started = pr && (pr.marked + pr.skipped) > 0;
  const done = pr && pr.next == null;
  if (!A.pick) A.pick = new Set(AI_DEFAULT_CANDIDATES.filter(id => id === 'local'
    || (A.models || []).some(m => m.id === id)));
  const cands = [...AI_DEFAULT_CANDIDATES.filter(id => id === 'local'
      || (A.models || []).some(m => m.id === id)),
    ...(A.models || []).map(m => m.id).filter(id => !AI_DEFAULT_CANDIDATES.includes(id))
      .filter(id => A.pick.has(id))];
  const nameOf = id => id === 'local' ? `Local (${A.page.local.name})`
    : ((A.models || []).find(m => m.id === id) || { name: id }).name.split(': ').pop();
  const est = el('span', { class: 'small se', 'data-jt-estimate': '1' });
  const estimate = async () => {
    const ids = [...A.pick];
    if (!ids.length) { est.textContent = ''; return; }
    try {
      const e = await api('api/judge-test/estimate?models=' + ids.map(encodeURIComponent).join(','));
      est.textContent = `about ${usd(e.usd)} for ${e.n} answers`;
    } catch (x) { est.textContent = ''; }
  };
  if (jt && started) setTimeout(estimate, 0);
  const cal = ((DATA.judged || {}).calibration) || null;
  return el('div', { class: 'card', 'data-judge-test': '1' },
    el('div', { class: 'rvbar' },
      el('div', {}, el('h2', { text: 'Judge test' }),
        el('p', { class: 'sub', text: 'How to pick the judge: you mark answers already on file, '
          + 'up to four candidate judges mark the same ones, and the table says which agrees '
          + 'with you. The judge you run now loses "provisional" at a weighted kappa of '
          + `${jt ? jt.kappa_min : 0.7} on ${jt ? jt.n_min : 100} answers or more.` })),
      jt && jt.answers.length ? el('button', { class: 'primary', 'data-jt-mark': '1',
        text: done ? 'Look at your marks' : started ? 'Continue marking' : 'Start marking',
        onclick: () => { A.mark = true; A.at = done ? 0 : pr.next; navigate({ tab: 'ai' }); } })
        : ''),
    !jt ? skeleton(2) : !jt.answers.length ? el('p', { class: 'note', 'data-jt-empty': '1',
      text: 'No judged answers are on file yet. Sit the Knowledge exam with a model first.' })
      : el('p', { class: 'small', 'data-jt-progress': `${pr.marked}|${pr.total}` },
        `You’ve marked ${pr.marked} of ${pr.total}` + (pr.skipped ? ` · ${pr.skipped} skipped`
          : '') + '.'),
    cal && cal.method ? el('p', { class: 'small', 'data-jt-calibration': '1',
      text: cal.calibrated ? `The judge now: checked against ${cal.by} on ${cal.n} answers · `
        + `κ ${cal.kappa}` : `The judge now: κ ${cal.kappa} on ${cal.n} answers — `
        + `${cal.kappa_min} on ${cal.n_min} takes "provisional" off.` }) : '',
    jt && started ? el('div', { class: 'jtcands', 'data-jt-candidates': '1' },
      el('p', { class: 'small', text: 'Candidates — tick up to four; each marks the same '
        + 'answers with the board’s judge prompts:' }),
      el('div', { class: 'jtpick' }, cands.map(id => el('label', { class: 'small',
          'data-jt-cand': id },
        el('input', { type: 'checkbox', checked: A.pick.has(id) ? '' : null,
          disabled: !A.pick.has(id) && A.pick.size >= 4 ? '' : null,
          onchange: e => { if (e.target.checked) A.pick.add(id); else A.pick.delete(id);
            render(); } }), ' ' + nameOf(id)))),
      el('div', { class: 'frm' },
        el('button', { class: 'secondary', 'data-jt-run': '1', text: 'Run the candidates',
          disabled: A.pick.size ? null : '', onclick: async () => {
            if (!whoName()) { askName(); return; }
            try {
              const r = await post('api/judge-test/run', { models: [...A.pick], by: whoName() });
              toast(`${r.runs.length} candidate${r.runs.length === 1 ? '' : 's'} marking — the `
                + 'table fills in as each finishes', { key: 'ai' });
            } catch (e) { toast('Refused. ' + e.message, { key: 'ai' }); }
          } }), est)) : '',
    res && res.rows.length ? judgeTestTable(res) : '');
}

function judgeTestTable(res) {
  const A = state.ai;
  return el('table', { class: 'jtresult', 'data-jt-result': '1' },
    el('thead', {}, el('tr', {}, ['Judge', 'Same mark as you', 'Within 1 point',
      'Agreement (weighted κ)', 'Cost per 1,000 answers', ''].map(h => el('th', { text: h })))),
    el('tbody', {}, res.rows.map(r => el('tr', { 'data-jt-row': r.key,
        class: r.best ? 'best' : null },
      el('td', {}, r.name, r.current ? el('span', { class: 'small se', text: ' · the judge now' })
        : '', r.best ? el('span', { class: 'badge', 'data-jt-best': '1', text: 'best' }) : '',
        r.provider ? el('div', { class: 'small se', text: 'on ' + r.provider }) : ''),
      el('td', { class: 'mono num', 'data-jt-exact': String(r.exact), text: jtPct(r.exact) }),
      el('td', { class: 'mono num', text: jtPct(r.within1) }),
      el('td', { class: 'mono num', 'data-jt-kappa': String(r.kappa),
        text: r.kappa == null ? '—' : `${r.kappa.toFixed(2)} · ${r.n}` }),
      el('td', { class: 'mono num', text: r.per_1000 == null ? '—' : usd(r.per_1000) }),
      el('td', {}, r.current || !LIVE ? '' : el('button', { class: r.best ? 'primary' : 'quiet',
        'data-jt-use': r.key, text: 'Use this judge', onclick: async () => {
          if (!whoName()) { askName(); return; }
          try {
            const x = await post('api/judge-test/use', { key: r.key, by: whoName() });
            if (x.rejudge && x.rejudge.n) A.confirm = x.rejudge;
            A.page = null; A.res = null; A.jt = null;
            toast(`The judge is ${r.name} now`, { key: 'ai' });
          } catch (e) { toast('Refused. ' + e.message, { key: 'ai' }); }
          render(); } }))))));
}

// one answer at a time: the question, what a full answer contains, the answer;
// keys 0–4 (or P and F), S to skip. The judges' marks are never shown here
function vJudgeMark() {
  const A = state.ai, jt = A.jt;
  if (!jt) { if (netReady()) loadJudgeTest(); return [el('div', { class: 'card' }, skeleton(4))]; }
  const list = jt.answers;
  // opened from the address (a reload): where the marking stopped
  if (A.at == null) A.at = jt.progress.next == null ? 0 : jt.progress.next;
  const i = Math.max(0, Math.min(list.length - 1, A.at));
  const a = list[i];
  const mine = jt.marks[a.key];
  const give = async v => {
    if (!whoName()) { askName(); return; }
    try {
      jt.progress = await post('api/judge-test/mark', { key: a.key, mark: v, by: whoName() });
      jt.marks[a.key] = v === 'P' ? 4 : v === 'F' ? 0 : v;
      A.at = i + 1 < list.length ? i + 1 : i;
      if (i + 1 >= list.length) { A.mark = false; A.res = null; loadJudgeTest(); }
    } catch (e) { toast('Refused. ' + e.message, { key: 'ai' }); }
    render();
  };
  A.give = give;
  const pf = a.scale === 'pf';
  const scale = pf ? [['P', 'Pass', 4], ['F', 'Fail', 0]]
    : [0, 1, 2, 3, 4].map(n => [n, String(n), n]);
  const pr = jt.progress;
  return [el('div', { class: 'card jtmark', 'data-jt-answer': a.key, 'data-jt-scale': a.scale },
    el('div', { class: 'rvbar' },
      el('a', { href: '#tab=ai', 'data-jt-back': '1', text: '← AI models', onclick: e => {
        e.preventDefault(); A.mark = false; A.res = null; loadJudgeTest(); navigate({ tab: 'ai' }); } }),
      el('span', { class: 'small mono', 'data-jt-at': `${i + 1}|${list.length}`,
        text: `${i + 1} of ${list.length} · ${pr.marked} marked` })),
    el('p', { class: 'small se', text: `${a.kind_label} · ${a.topic}` }),
    el('p', { class: 'evq-label', text: 'Question' }),
    el('blockquote', { class: 'evq', text: a.question }),
    el('p', { class: 'evq-label', text: pf ? 'It passes if' : 'A full answer' }),
    a.criteria.length ? el('ul', { class: 'jtcrit', 'data-jt-criteria': '1' },
      a.criteria.map(c => el('li', { text: c })),
      ...a.flags.map(f => el('li', { class: 'se', text: 'flag: ' + f })))
      : el('div', { class: 'jtrubric', 'data-jt-rubric': '1', text: a.rubric }),
    a.reference ? el('details', { class: 'jtref' }, el('summary', { text: 'the reference answer' }),
      el('div', { class: 'small', text: a.reference })) : '',
    el('p', { class: 'evq-label', text: 'Answer' }),
    el('div', { class: 'evans', 'data-jt-text': '1', text: a.answer }),
    el('div', { class: 'jtkeys', role: 'group', 'aria-label': 'your mark' },
      ...scale.map(([key, label, val]) => el('button', { class: 'chip-btn' + (mine === val ? ' on' : ''),
        'data-jt-give': String(key), 'aria-pressed': String(mine === val),
        'aria-keyshortcuts': String(key), text: pf ? `${label} (${key})` : label,
        onclick: () => give(key) })),
      el('button', { class: 'quiet', 'data-jt-give': 'S', 'aria-keyshortcuts': 'S',
        text: 'Skip (S)', onclick: () => give(null) })),
    el('p', { class: 'small se', text: pf ? 'Keys: P, F, S to skip; ← and → move.'
      : 'Keys: 0 to 4, S to skip; ← and → move. Your marks save as you give them.' }))];
}
addEventListener('keydown', e => {
  const A = state.ai;
  if (!A || !A.mark || state.tab !== 'ai' || e.target.closest('input, textarea, select')
      || e.metaKey || e.ctrlKey || e.altKey || !A.give) return;
  const k = e.key.toUpperCase();
  const a = ((A.jt || {}).answers || [])[A.at || 0];
  if (!a) return;
  if (a.scale === 'pf' ? (k === 'P' || k === 'F') : /^[0-4]$/.test(k)) {
    e.preventDefault(); A.give(a.scale === 'pf' ? k : Number(k));
  } else if (k === 'S') { e.preventDefault(); A.give(null); }
  else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    A.at = Math.max(0, Math.min(A.jt.answers.length - 1, (A.at || 0) + (e.key === 'ArrowRight' ? 1 : -1)));
    render();
  }
});

// ===========================================================================
// 12i.2: Build questions — new Knowledge exam or Everyday questions in three
// steps: What, Try 10, Make the rest (checked). Opened from the exam's and the
// Everyday page's questions; a draft lives on the server, so it survives a
// reload, and the address names it
// ===========================================================================
const QB_FORM_KEY = 'qb-form';
function qbForm() {
  const Q = state.qb;
  if (!Q.form) {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(QB_FORM_KEY) || 'null'); } catch (e) { /* private */ }
    Q.form = Object.assign({ kind: 'knowledge', topic: '', subtopics: null, level: 'general public',
      group: '', newLabel: '', newAbout: '', count: 60, writer: '', checker: '', dedup: true,
      prompt: { knowledge: null, everyday: null } }, saved || {});
  }
  if (Q.kind && Q.kind !== Q.form.kind) { Q.form.kind = Q.kind; }
  Q.kind = null;
  return Q.form;
}
function qbSave() {
  try { localStorage.setItem(QB_FORM_KEY, JSON.stringify(state.qb.form)); } catch (e) { /* private */ }
}
async function loadQb(force) {
  const Q = state.qb;
  if (Q.asked || (Q.page && !force)) return;
  Q.asked = true;
  try {
    Q.page = await api('api/builder');
    if (Q.page.has_key && !Q.models) {
      try { Q.models = (await api('api/ai/models')).models; } catch (e) { /* the page stands */ }
    }
  } catch (e) { Q.msg = e.message; }
  Q.asked = false;
  if (state.tab === 'build') render();
}
async function loadQbDraft(id) {
  const Q = state.qb;
  if (Q.draftAsked) return;
  Q.draftAsked = true;
  try { Q.draft = await api('api/builder/' + encodeURIComponent(id)); Q.msg = ''; }
  catch (e) { Q.msg = e.message; Q.draft = null; }
  Q.draftAsked = false;
  if (state.tab === 'build') render();
  qbPoll();
}
// while the writer or the checker is at work, the page asks again
function qbPoll() {
  const Q = state.qb, d = Q.draft;
  clearTimeout(Q.timer);
  if (!d || !['writing', 'checking'].includes(d.status)) return;
  Q.timer = setTimeout(() => {
    if (state.tab === 'build' && Q.id === d.id) loadQbDraft(d.id);
  }, 1500);
}
function openBuilder(kind) {
  Object.assign(state.qb, { id: null, draft: null, kind, at: 0, editing: null });
  navigate({ tab: 'build', model: null, topic: null });
}
function qbOpen(id) {
  Object.assign(state.qb, { id, draft: null, at: 0, editing: null });
  navigate({ tab: 'build', model: null, topic: null });
}
const qbWhat = d => d.kind === 'knowledge' ? `${d.spec.topic} · ${d.spec.level}`
  : `Everyday · ${d.spec.group_label || d.spec.group}`;

function vBuild() {
  const Q = state.qb;
  if (!Q.page && netReady()) loadQb();
  if (Q.id && (!Q.draft || Q.draft.id !== Q.id) && netReady()) loadQbDraft(Q.id);
  const d = Q.id && Q.draft && Q.draft.id === Q.id ? Q.draft : null;
  const step = !d ? 1 : d.stage === 'try' ? 2 : 3;
  const steps = el('ol', { class: 'qbsteps', 'data-qb-steps': String(step) },
    ['What', `Try ${Q.page ? Q.page.try_n : 10}`, 'Make the rest, checked'].map((t, i) =>
      el('li', { class: i + 1 === step ? 'on' : i + 1 < step ? 'done' : '',
        'aria-current': i + 1 === step ? 'step' : null, text: `${i + 1} ${t}` })));
  const head = el('div', { class: 'card', 'data-qb': '1' },
    el('div', { class: 'rvbar' }, el('h2', { text: 'Build questions' }),
      d ? el('a', { href: '#tab=build', 'data-qb-new': '1', text: 'Start another',
        onclick: e => { e.preventDefault(); Q.id = null; Q.draft = null; navigate({ tab: 'build' }); } })
        : ''),
    el('p', { class: 'sub', text: 'An AI model writes new questions, another answers them blind to '
      + 'check them, and you review the ones it flags. Published questions join the bank as a new '
      + 'version and split into practice and hidden halves, as the bank’s always do.' }),
    steps, Q.msg ? el('p', { class: 'warn', text: Q.msg }) : '');
  if (!Q.page) return [head, el('div', { class: 'card' }, skeleton(4, { 'data-loading': 'qb' }))];
  if (Q.id && !d) return [head, el('div', { class: 'card' }, skeleton(4, { 'data-loading': 'qb-draft' }))];
  if (!d) return [head, qbStepOne(), qbDrafts()];
  return [head, qbDraftCard(d)];
}

function qbDrafts() {
  const ds = (state.qb.page.drafts || []).filter(x => !x.published);
  if (!ds.length) return '';
  return el('div', { class: 'card', 'data-qb-drafts': String(ds.length) },
    el('h2', { text: 'Drafts' }),
    el('ul', { class: 'qbdrafts' }, ds.map(x => el('li', { 'data-qb-draft': x.id },
      el('a', { href: '#tab=build&draft=' + x.id, text: qbWhat(x),
        onclick: e => { e.preventDefault(); qbOpen(x.id); } }),
      el('span', { class: 'small se', text: ` · ${x.progress.line} · ${x.status}` }
      )))));
}

// ---- step 1: what ------------------------------------------------------------
function qbModelSelect(job, F) {
  const Q = state.qb, P = Q.page, now = P[job];
  const opts = [['', `as on AI models: ${now.label}`], ['local', 'Local (the model on this server)'],
    ...(Q.models || []).map(m => [m.id, m.name.split(': ').pop()])];
  return el('select', { 'aria-label': job === 'writer' ? 'question writer' : 'checker',
      'data-qb-model': job, onchange: e => { F[job] = e.target.value; qbSave(); render(); } },
    opts.map(([v, t]) => el('option', { value: v, selected: F[job] === v ? '' : null, text: t })));
}
function qbStepOne() {
  const Q = state.qb, P = Q.page, F = qbForm();
  const kn = F.kind === 'knowledge';
  const topics = P.topics || [];
  if (kn && !F.topic && topics.length) F.topic = topics[0].name;
  if (!kn && !F.group && (P.groups || []).length) F.group = P.groups[0].id;
  const topic = topics.find(t => t.name === F.topic);
  const subs = F.subtopics && F.subtopicsFor === F.topic ? F.subtopics
    : (topic ? topic.subtopics.join('\n') : '');
  const fld = (label, control, note) => el('label', { class: 'fld' },
    el('span', { class: 'fld-label', text: label }), control, note || '');
  const count = Number(F.count) || 0;
  const under = count > 0 && count < P.suggest_min;
  const prompt = P.prompts[F.kind];
  const edited = F.prompt[F.kind];
  const est = el('p', { class: 'small', 'data-qb-estimate': '1' },
    'Estimated cost: ', el('span', { class: 'mono', text: Q.est ? Q.est.line : '…' }));
  qbEstimate(F);
  const nTry = Math.min(P.try_n, count || P.try_n);
  return el('div', { class: 'card', 'data-qb-what': F.kind },
    el('h2', { text: 'What' }),
    el('div', { class: 'subswitch', role: 'radiogroup', 'aria-label': 'kind of question' },
      [['knowledge', 'Knowledge exam'], ['everyday', 'Everyday tasks']].map(([k, t]) =>
        el('button', { class: 'chip-btn' + (F.kind === k ? ' on' : ''), role: 'radio',
          'aria-checked': String(F.kind === k), 'data-qb-kind': k, text: t,
          onclick: () => { F.kind = k; Q.est = null; qbSave(); render(); } }))),
    el('div', { class: 'frm fields' },
      kn ? fld('Topic', el('select', { 'aria-label': 'topic', 'data-qb-topic': '1',
          onchange: e => { F.topic = e.target.value; F.subtopics = null; qbSave(); render(); } },
        topics.map(t => el('option', { value: t.name, selected: t.name === F.topic ? '' : null,
          text: `${t.name} · ${t.bank} in the bank` }))))
        : fld('Group', el('select', { 'aria-label': 'group', 'data-qb-group': '1',
          onchange: e => { F.group = e.target.value; Q.est = null; qbSave(); render(); } },
        [...(P.groups || []).map(g => el('option', { value: g.id, selected: g.id === F.group ? '' : null,
          text: `${g.label} · ${g.bank} in the bank` })),
         el('option', { value: 'new', selected: F.group === 'new' ? '' : null, text: 'New group…' })])),
      !kn && F.group === 'new' ? fld('Its name', el('input', { type: 'text', value: F.newLabel,
        'data-keep': 'qb-new-label', 'data-qb-new-label': '1', placeholder: 'e.g. Travel plans',
        oninput: e => { F.newLabel = e.target.value; qbSave(); } })) : '',
      !kn && F.group === 'new' ? fld('What it tests, in one line', el('input', { type: 'text',
        value: F.newAbout, 'data-keep': 'qb-new-about', 'data-qb-new-about': '1',
        oninput: e => { F.newAbout = e.target.value; qbSave(); } })) : '',
      kn ? fld('Level', el('div', { role: 'radiogroup', 'aria-label': 'level', class: 'frm' },
        P.levels.map(l => el('label', { class: 'small' }, el('input', { type: 'radio', name: 'qb-level',
          value: l, 'data-qb-level': l, checked: F.level === l ? '' : null,
          onchange: () => { F.level = l; qbSave(); } }), ' ' + l)))) : '',
      fld('How many', el('input', { type: 'number', min: '1', step: '1', value: String(F.count),
        'data-keep': 'qb-count', 'data-qb-count': '1', style: 'width:7em',
        oninput: e => { F.count = e.target.value; Q.est = null; qbSave(); render(); } }),
        under ? el('span', { class: 'small se', 'data-qb-under': '1', text: `fewer than `
          + `${P.suggest_min} won’t give this ${kn ? 'topic' : 'group'} its own score in Improve` })
          : ''),
      fld('Question writer', qbModelSelect('writer', F)),
      fld('Checker', qbModelSelect('checker', F))),
    kn ? fld('Subtopics — one a line, edit freely', el('textarea', { rows: '4',
      'data-keep': 'qb-subtopics', 'data-qb-subtopics': '1', text: subs,
      oninput: e => { F.subtopics = e.target.value; F.subtopicsFor = F.topic; qbSave(); } })) : '',
    el('label', { class: 'small' }, el('input', { type: 'checkbox', 'data-qb-dedup': '1',
      checked: F.dedup ? '' : null, onchange: e => { F.dedup = e.target.checked; qbSave(); } }),
      ' Check for duplicates', el('span', { class: 'se', text: ` — against this bank, earlier `
        + `batches and each other (${P.dedup_how})` })),
    el('details', { class: 'qbprompt', 'data-qb-prompt': '1', open: Q.promptOpen ? '' : null,
        ontoggle: e => { Q.promptOpen = e.target.open; } },
      el('summary', { text: 'Edit the writing instructions ▸' }),
      el('p', { class: 'small se', text: `From ${prompt.path}. Your edits apply to this batch, `
        + 'and the instructions used are saved with it.' }),
      el('textarea', { rows: '16', class: 'mono small', 'aria-label': 'writing instructions',
        'data-keep': 'qb-prompt-' + F.kind, 'data-qb-prompt-text': '1',
        text: edited == null ? prompt.editable : edited,
        oninput: e => { F.prompt[F.kind] = e.target.value === prompt.editable ? null : e.target.value;
          qbSave(); } }),
      el('p', { class: 'small', text: 'The output section is locked, so the result can always be read:' }),
      el('pre', { class: 'qblocked small', 'data-qb-locked': '1', text: prompt.locked }),
      el('button', { class: 'quiet', 'data-qb-prompt-reset': '1', text: 'Reset to default',
        disabled: edited == null ? '' : null,
        onclick: () => { F.prompt[F.kind] = null; qbSave(); render(); } })),
    est,
    el('div', { class: 'frm' },
      el('button', { class: 'primary', 'data-qb-try': '1', text: `Try ${nTry}`,
        disabled: count >= 1 ? null : '', onclick: () => qbCreate(F, subs) })),
    P.writer_blocked ? el('p', { class: 'warn', text: 'The question writer can’t run: '
      + P.writer_blocked }) : '',
    // the checker has no .env of its own: it is chosen on AI models, or here
    P.checker_blocked && !F.checker ? el('p', { class: 'small se', 'data-qb-no-checker': '1',
      text: 'No checker yet: choose one on AI models, or above for this batch. Try 10 runs '
        + 'without it; Make the rest needs it.' }) : '');
}
let qbEstTimer = null;
function qbEstimate(F) {
  const Q = state.qb;
  const want = JSON.stringify([F.kind, F.count, F.group, F.writer, F.checker]);
  if (Q.estFor === want) return;
  Q.estFor = want;
  clearTimeout(qbEstTimer);
  qbEstTimer = setTimeout(async () => {
    try {
      Q.est = await post('api/builder/estimate', { kind: F.kind, count: Number(F.count) || 0,
        group: F.kind === 'everyday' ? F.group : '', writer: F.writer, checker: F.checker });
    } catch (e) { Q.est = { line: '—' }; }
    const n = document.querySelector('[data-qb-estimate] .mono');
    if (n) n.textContent = Q.est.line;
  }, 250);
}
async function qbCreate(F, subs) {
  const Q = state.qb;
  if (!whoName()) { askName(); return; }
  const kn = F.kind === 'knowledge';
  try {
    const d = await post('api/builder', {
      kind: F.kind, count: Number(F.count) || 0, dedup: !!F.dedup, by: whoName(),
      writer: F.writer, checker: F.checker, prompt: F.prompt[F.kind] || '',
      ...(kn ? { topic: F.topic, level: F.level,
                 subtopics: String(subs || '').split('\n').map(x => x.trim()).filter(Boolean) }
             : { group: F.group, new_label: F.newLabel, new_about: F.newAbout }) });
    Q.draft = d; Q.id = d.id; Q.at = 0;
    Q.page = null;                        // the drafts list has a new one
    navigate({ tab: 'build' });
    qbPoll();
  } catch (e) { toast('Refused. ' + e.message, { key: 'qb' }); }
}

// ---- steps 2 and 3: a draft --------------------------------------------------
function qbReviewable(d) {
  // step 2: the first ten; step 3: the flagged, then the sample
  if (d.stage === 'try') return d.items.filter(it => !it.auto);
  const need = d.items.filter(it => !it.auto && (it.flags.length || it.sample));
  return [...need.filter(it => it.flags.length), ...need.filter(it => !it.flags.length)];
}
function qbDraftCard(d) {
  const Q = state.qb, pr = d.progress;
  const busy = ['writing', 'checking'].includes(d.status);
  const bits = [];
  bits.push(el('div', { class: 'rvbar' },
    el('div', {}, el('h2', { text: d.stage === 'try' ? `Try ${Math.min(Q.page.try_n,
      d.spec.count)}` : 'Make the rest, checked' }),
      el('p', { class: 'small se', 'data-qb-what-line': '1', text: `${qbWhat(d)} · ${d.spec.count} `
        + `questions · writer ${d.writer.label} · checker ${d.checker.label}` })),
    busy ? el('button', { class: 'quiet', 'data-qb-cancel': '1', text: 'Cancel',
      onclick: () => qbStep(d, 'cancel') }) : ''));
  bits.push(el('p', { class: 'small', 'data-qb-progress': `${pr.written}|${pr.count}|${pr.flagged}`,
    text: (busy ? (d.status === 'writing' ? 'Writing… ' : 'Checking… ') : '') + pr.line
      + (d.stage === 'rest' && d.status === 'checking' ? ` · ${pr.checked} checked` : '') }));
  if (d.status === 'cancelled' || d.status === 'failed')
    bits.push(el('p', { class: 'warn', 'data-qb-stopped': d.status,
      text: d.status === 'cancelled' ? 'Cancelled. What was written is kept here.'
        : `Stopped: ${d.error}` }),
      el('div', { class: 'frm' }, el('button', { class: 'secondary', 'data-qb-resume': '1',
        text: 'Resume', onclick: () => qbStep(d, 'resume') })));
  if (d.published) {
    const p = d.published;
    bits.push(el('p', { class: 'note', 'data-qb-published': String(p.n), text: `Published `
      + `${p.added} question${p.added === 1 ? '' : 's'} — ` + (p.kind === 'knowledge'
        ? `${p.topic}, bank version ${p.version || 'rebuilt'}` : `Everyday, bank version ${p.version}`)
      + '. They split into practice and hidden halves as the bank’s always do.' }),
      el('div', { class: 'frm' }, el('button', { class: 'secondary', 'data-qb-see': '1',
        text: 'See the bank', onclick: () => navigate({ tab: p.kind === 'knowledge' ? 'exam' : 'everyday',
          model: null, topic: null }) })));
    return el('div', { class: 'card', 'data-qb-draft-open': d.id }, bits);
  }
  const aside = d.items.filter(it => it.auto);
  const list = busy && d.stage === 'rest' ? [] : qbReviewable(d);
  if (!busy || d.stage === 'try') {
    if (list.length) {
      Q.at = Math.max(0, Math.min(list.length - 1, Q.at || 0));
      bits.push(qbItemCard(d, list[Q.at], Q.at, list.length));
    } else if (d.stage === 'rest' && d.status === 'review') {
      bits.push(el('p', { class: 'small', text: 'Nothing is flagged, and the sample is reviewed.' }));
    }
  }
  if (aside.length)
    bits.push(el('details', { class: 'qbaside', 'data-qb-aside': String(aside.length) },
      el('summary', { text: `Set aside before review (${aside.length})` }),
      el('ul', { class: 'small' }, aside.map(it => el('li', {},
        el('span', { class: 'mono', text: `#${it.n} ` }), qbText(d, it.q).slice(0, 90),
        el('span', { class: 'se', text: ` — ${it.auto}` }))))));
  if (d.stage === 'try') {
    const live = d.items.filter(it => !it.auto).length;
    bits.push(el('div', { class: 'frm' },
      el('button', { class: 'primary', 'data-qb-rest': '1', text: 'Make the rest',
        disabled: d.can_rest ? '' : null, title: d.can_rest || null,
        onclick: () => qbStep(d, 'rest') }),
      el('span', { class: 'small se', 'data-qb-reviewed': String(pr.reviewed_try),
        text: `${pr.reviewed_try} of ${live} reviewed` + (d.can_rest ? ` · ${d.can_rest}` : '') })));
  } else if (d.status === 'review') {
    const n = pr.publishable;
    bits.push(el('p', { class: 'small', 'data-qb-left': String(pr.to_review),
      text: `${pr.to_review} to review · ${n} ready to publish` }),
      el('div', { class: 'frm' }, el('button', { class: 'primary', 'data-qb-publish': '1',
        text: `Publish ${n} question${n === 1 ? '' : 's'}`, disabled: d.can_publish ? '' : null,
        title: d.can_publish || null, onclick: () => qbStep(d, 'publish') })));
  }
  return el('div', { class: 'card', 'data-qb-draft-open': d.id }, bits);
}
const qbText = (d, q) => d.kind === 'knowledge' ? q.question : q.prompt;
function qbItemCard(d, it, i, n) {
  const Q = state.qb, q = it.q, kn = d.kind === 'knowledge';
  const editing = Q.editing === it.n;
  const dup = it.flags.find(f => f.kind === 'dup' && !f.resolved);
  const verdictWord = { accept: 'accepted', edit: 'edited', reject: 'rejected' }[it.verdict];
  const nav = el('div', { class: 'rvbar' },
    el('span', { class: 'small mono', 'data-qb-at': `${i + 1}|${n}`,
      text: `${i + 1} of ${n} · #${it.n}` + (it.sample && !it.flags.length ? ' · sampled at random' : '') }),
    el('span', { class: 'frm' },
      el('button', { class: 'quiet', 'aria-label': 'previous', text: '←', disabled: i ? null : '',
        onclick: () => { Q.at = i - 1; Q.editing = null; render(); } }),
      el('button', { class: 'quiet', 'aria-label': 'next', text: '→', disabled: i + 1 < n ? null : '',
        onclick: () => { Q.at = i + 1; Q.editing = null; render(); } })));
  const flags = it.flags.length ? el('ul', { class: 'qbflags' }, it.flags.map(f =>
    el('li', { class: 'warntext', 'data-qb-flag': f.kind, text: f.text }))) : '';
  const body = editing ? qbEditor(d, it) : el('div', {},
    el('p', { class: 'evq-label', text: 'Question' }),
    el('blockquote', { class: 'evq', 'data-qb-q': '1', text: qbText(d, q) }),
    el('p', { class: 'evq-label', text: 'Reference answer' }),
    el('div', { class: 'small', 'data-qb-ref': '1', text: q.reference }),
    el('p', { class: 'evq-label', text: kn ? 'A full answer' : 'Checks' }),
    el('ul', { class: 'small', 'data-qb-checks': '1' }, (kn ? q.criteria : it.checks_words || [])
      .map(c => el('li', { text: c }))),
    it.answer != null ? el('details', { class: 'small', 'data-qb-answer': '1' },
      el('summary', { text: 'the checker’s own answer' + (it.mark != null ? ` · the judge’s mark ${it.mark}/4` : '') }),
      el('div', { text: it.answer || '(none)' })) : '',
    q.notes ? el('p', { class: 'small se', text: 'The writer’s note: ' + q.notes }) : '');
  const dupBox = dup ? el('div', { class: 'qbdup', 'data-qb-dup': String(it.n) },
    el('div', {}, el('b', { class: 'small', text: 'This one' }), el('p', { class: 'small', text: qbText(d, q) })),
    el('div', {}, el('b', { class: 'small', text: dup.other.label }),
      el('p', { class: 'small', 'data-qb-dup-other': '1', text: dup.other.text })),
    el('div', { class: 'frm' }, ['new', 'old', 'both'].map(k => el('button', {
      class: k === 'old' ? 'secondary' : 'quiet', 'data-qb-keep': k, text: `keep ${k}`,
      disabled: k === 'new' && dup.other.src !== 'batch' ? '' : null,
      title: k === 'new' && dup.other.src !== 'batch' ? 'the other one is already in the bank' : null,
      onclick: () => qbDup(d, it, k) })))) : '';
  const reasons = el('div', { class: 'qbreasons', role: 'group', 'aria-label': 'reason' },
    Q.page.reasons.map(r => el('button', { class: 'chip-btn' + (Q.reason === r ? ' on' : ''),
      'data-qb-reason': r, 'aria-pressed': String(Q.reason === r), text: r,
      onclick: () => { Q.reason = Q.reason === r ? '' : r; render(); } })));
  const acts = editing ? '' : el('div', { class: 'frm qbacts' },
    el('button', { class: 'chip-btn', 'data-qb-verdict': 'accept', 'aria-keyshortcuts': 'A',
      text: 'Accept (A)', onclick: () => qbGive(d, it, 'accept') }),
    el('button', { class: 'chip-btn', 'data-qb-verdict': 'edit', 'aria-keyshortcuts': 'E',
      text: 'Edit (E)', onclick: () => { Q.editing = it.n; render(); } }),
    el('button', { class: 'chip-btn', 'data-qb-verdict': 'reject', 'aria-keyshortcuts': 'R',
      text: 'Reject (R)', onclick: () => qbGive(d, it, 'reject') }));
  return el('div', { class: 'qbitem', 'data-qb-item': String(it.n),
      'data-qb-verdict-now': it.verdict || '' },
    nav, flags, dupBox, body,
    verdictWord ? el('p', { class: 'small se', 'data-qb-done': it.verdict,
      text: verdictWord + (it.reason ? ` · ${it.reason}` : '') }) : '',
    editing ? '' : reasons, acts,
    editing ? '' : el('p', { class: 'small se', text: 'Keys: A, E, R; ← and → move. A reason is optional.' }));
}
function qbEditor(d, it) {
  const Q = state.qb, q = it.q, kn = d.kind === 'knowledge';
  const f = { text: qbText(d, q), reference: q.reference,
              extra: kn ? q.criteria.join('\n') : JSON.stringify(q.checks, null, 1) };
  const ta = (key, label, v, rows) => el('label', { class: 'fld' },
    el('span', { class: 'fld-label', text: label }),
    el('textarea', { rows: String(rows), 'data-qb-edit': key, 'data-keep': 'qb-edit-' + key,
      text: v, oninput: e => { f[key] = e.target.value; } }));
  return el('div', { class: 'qbedit' },
    ta('text', 'Question', f.text, 3), ta('reference', 'Reference answer', f.reference, 3),
    ta('extra', kn ? 'A full answer — one point a line' : 'Checks (JSON)', f.extra, 5),
    el('div', { class: 'frm' },
      el('button', { class: 'primary', 'data-qb-save': '1', text: 'Save the edit', onclick: () => {
        let edited;
        try {
          edited = kn ? { question: f.text, reference: f.reference,
                          criteria: f.extra.split('\n').map(x => x.trim()).filter(Boolean) }
            : { prompt: f.text, reference: f.reference, checks: JSON.parse(f.extra) };
        } catch (e) { toast('The checks are not valid JSON.', { key: 'qb' }); return; }
        qbGive(d, it, 'edit', edited);
      } }),
      el('button', { class: 'quiet', text: 'Cancel', onclick: () => { Q.editing = null; render(); } })));
}
async function qbGive(d, it, verdict, edited) {
  const Q = state.qb;
  if (!whoName()) { askName(); return; }
  try {
    Q.draft = await post(`api/builder/${d.id}/review`, { n: it.n, verdict, reason: Q.reason || '',
      edited: edited || {}, by: whoName() });
    Q.reason = ''; Q.editing = null;
    const list = qbReviewable(Q.draft);
    // on to the next one still to review, else stay
    const next = list.findIndex((x, j) => j > Q.at && !x.verdict);
    Q.at = next >= 0 ? next : Q.at;
  } catch (e) { toast('Refused. ' + e.message, { key: 'qb' }); }
  render();
}
async function qbDup(d, it, keep) {
  const Q = state.qb;
  if (!whoName()) { askName(); return; }
  try { Q.draft = await post(`api/builder/${d.id}/duplicate`, { n: it.n, keep, by: whoName() }); }
  catch (e) { toast('Refused. ' + e.message, { key: 'qb' }); }
  render();
}
async function qbStep(d, step) {
  const Q = state.qb;
  if (!whoName()) { askName(); return; }
  try {
    const r = await post(`api/builder/${d.id}/${step}`, { by: whoName() });
    if (step === 'publish') {
      Q.draft = r.draft;
      toast(`Published ${r.published.added} questions — a new bank version`, { key: 'qb' });
      if (LIVE) refreshResults();
    } else Q.draft = r;
    Q.at = 0;
  } catch (e) { toast('Refused. ' + e.message, { key: 'qb' }); }
  render();
  qbPoll();
}
addEventListener('keydown', e => {
  const Q = state.qb;
  if (!Q || state.tab !== 'build' || !Q.draft || Q.editing != null
      || e.target.closest('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey) return;
  const d = Q.draft;
  if (['writing', 'checking'].includes(d.status) && d.stage === 'rest') return;
  const list = qbReviewable(d);
  const it = list[Q.at || 0];
  if (!it) return;
  const k = e.key.toUpperCase();
  if (k === 'A') { e.preventDefault(); qbGive(d, it, 'accept'); }
  else if (k === 'R') { e.preventDefault(); qbGive(d, it, 'reject'); }
  else if (k === 'E') { e.preventDefault(); Q.editing = it.n; render(); }
  else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    Q.at = Math.max(0, Math.min(list.length - 1, (Q.at || 0) + (e.key === 'ArrowRight' ? 1 : -1)));
    render();
  }
});

// ---------- shell ----------
// 12b: five places, organised around what people come to do. A table, a
// reader or a dialog keeps its code and changes container: the views below
// are the old tabs under their old ids, so every call that opens one still
// does, and the places group them in the header.
const TABS = [
  ['overview', 'Home', vOverview],
  ['leaderboard', 'Models', vLeaderboard],
  ...(LIVE ? [['pipeline', 'By model', vPipeline],
              ['training', 'Training runs', vTraining]] : []),
  ['tasks', 'Standard', vStandardBench],
  ...(LIVE ? [['exam', 'Knowledge exam', vExam]] : []),
  ['everyday', 'Everyday tasks', vEverydayPage],
  // pages, not places: reached from the header's right side
  ...(LIVE ? [['queue', 'All runs', vAllRuns]] : []),
  ['provenance', 'Data & sources', vRuns],
  ['help', 'Help', vHelp],
  // 12i.1: settings, under the name menu
  ...(LIVE ? [['ai', 'AI models', vAiModels]] : []),
  // 12i.2: reached from the Knowledge exam's and the Everyday page's questions
  ...(LIVE ? [['build', 'Build questions', vBuild]] : []),
];
// Playground joins in 12d, between Models and Improve
const PLACES = [['home', 'Home', ['overview']], ['models', 'Models', ['leaderboard']],
  ['improve', 'Improve', ['pipeline', 'training']],
  ['benchmarks', 'Benchmarks', ['tasks', 'exam', 'everyday']]]
  .map(([id, label, views]) => [id, label, views.filter(v => TABS.some(t => t[0] === v))])
  .filter(p => p[2].length);
const placeOf = v => (PLACES.find(p => p[2].includes(v)) || [])[0] || null;
const viewLabel = v => (TABS.find(t => t[0] === v) || [, v])[1];
// the address of each view: its place, and which part of it
const SUB_SLUG = { pipeline: 'model', training: 'training',
                   tasks: 'standard', exam: 'exam', everyday: 'everyday' };
const PAGE_SLUG = { overview: 'home', leaderboard: 'models', queue: 'runs',
                    provenance: 'data', help: 'help', ai: 'ai', build: 'build' };

// Old hashes keep working (§8): a link someone pasted into a message last
// month still lands, and the address bar then shows where it lives now.
const TAB_ALIASES = {
  evals: 'provenance', 'submit-queue': 'submit', models_tab: 'models', ppl: 'perplexity',
  'perplexity-loss': 'perplexity',
};
// the Benchmarks switch remembers the viewer's last choice
function benchSub() {
  let v = 'tasks';
  try { v = localStorage.getItem('bench-benchmarks-sub') || v; } catch (e) { /* private */ }
  return ['tasks', 'exam', 'everyday'].includes(v) && TABS.some(t => t[0] === v) ? v : 'tasks';
}
function goPlace(pid) {
  const p = PLACES.find(x => x[0] === pid);
  if (!p) return;
  if (pid === 'models') lbS().view = modelsView();
  navigate({ tab: pid === 'benchmarks' ? benchSub() : p[2][0], model: null, topic: null });
}
// the switch at the top of Improve and Benchmarks
function subSwitch(place) {
  const p = PLACES.find(x => x[0] === place);
  return el('nav', { class: 'subswitch', 'data-subswitch': place, role: 'tablist',
      'aria-label': p[1] },
    p[2].map(v => el('button', { class: 'chip-btn' + (v === state.tab ? ' on' : ''), role: 'tab',
      'data-sub': SUB_SLUG[v], 'aria-selected': String(v === state.tab), text: viewLabel(v),
      onclick: () => {
        if (place === 'benchmarks')
          try { localStorage.setItem('bench-benchmarks-sub', v); } catch (e) { /* private */ }
        navigate({ tab: v, model: null, topic: null }); } })));
}
// 11b: the cards of a tab are its sections, numbered 01, 02, … in the order
// they are read. The index is drawn from the DOM rather than written into
// twenty view functions, so a card that moves takes its place in the count.
function numberSections() {
  const view = document.getElementById('view');
  if (!view) return;
  // 12b.3: Home's three blocks are not steps in a sequence: no numbers there
  if (!state.model && !state.topic && state.tab === 'overview') return;
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
  // 12b.3: before the scores arrive there is a header to draw, and nothing else
  if (!DATA) { renderShell(); return; }
  // full rebuild: drop the in-place refreshers so a poll can never touch the
  // DOM of a tab that just got torn down — the mounted tab re-registers its own
  state.trRedraw = state.queueRedraw = null;
  renderWarnings();                 // one line, in the bar, on every tab
  const ms = visible();
  renderTabs();
  const view = document.getElementById('view');
  view.classList.remove('dimmed');
  // a poll rebuilds the view every few seconds. Whatever the person is typing
  // in — and where their caret is — comes back afterwards, or the field is
  // unusable on a live page: this is the same bug as the tab bar's, one layer
  // down, and the cure is the same one (never lose what the DOM was holding).
  // 11f: did this render change the view (a tab, a model, a topic)? Only a
  // navigation moves the scroll or plays the entrance; a poll never does
  const vk = viewKey(), changed = _lastView != null && vk !== _lastView;
  // 12g.1: the checks and the run counter hang from the header, which a page
  // change does not rebuild — so it closes them
  if (changed && ['checks', 'runs'].includes(POP.key)) popClose();
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
  // 12b: the Test a model dialog is part of the view, so a poll redraws it
  // with everything else and keeps what is being typed into it
  const dlg = testDialog();
  if (state.model) view.replaceChildren(...vModel(), ...dlg);
  else if (state.topic) view.replaceChildren(...vTopic(), ...dlg);
  else {
    if (!TABS.some(([id]) => id === state.tab)) state.tab = 'overview';
    const place = placeOf(state.tab);
    view.replaceChildren(
      ...(place === 'improve' || place === 'benchmarks' ? [subSwitch(place)] : []),
      ...TABS.find(([id]) => id === state.tab)[2](ms), ...dlg);
  }
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
  // never wider than the window it has to sit 8px inside — 12i.0: or the
  // page card its button is on, for a picker that must not run past it
  const card = (opts || {}).inCard && anchor.closest('.card');
  const cr = card ? card.getBoundingClientRect() : null;
  const lo = Math.max(POP_EDGE, cr ? cr.left + 4 : 0);
  const hi = Math.min(innerWidth - POP_EDGE, cr ? cr.right - 4 : innerWidth);
  panel.style.maxWidth = `${Math.max(160, hi - lo)}px`;
  const pr = panel.getBoundingClientRect();
  const below = innerHeight - r.bottom - 4, above = r.top - 4;
  const flip = pr.height > below && above > below;
  const room = Math.max(120, (flip ? above : below) - POP_EDGE);
  const top = flip ? Math.max(POP_EDGE, r.top - 4 - Math.min(pr.height, room)) : r.bottom + 4;
  let left = (opts || {}).placement === 'bottom-end' ? r.right - pr.width : r.left;
  left = Math.min(Math.max(lo, left), Math.max(lo, hi - pr.width));
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
      + (e.dataset.column || e.dataset.bench || e.dataset.filter || e.dataset.tint || e.dataset.showAll
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
  // the first control you can see: Menu ▾ is there only on a phone (12b)
  const first = bar && [...bar.querySelectorAll('button, a, summary, [tabindex]')]
    .find(e => e.offsetParent !== null);
  if (first) first.focus({ preventScroll: true });
});

// 12b: the header holds four places and nothing that opens a list of more.
// Below 720px the four collapse into Menu ▾, on the left.
function renderTabs() {
  const tabs = document.getElementById('tabs');
  if (!_tabsBuilt) {
    _tabsBuilt = true;
    tabs.replaceChildren(...PLACES.map(([id, label]) =>
      el('button', { role: 'tab', 'data-tab': id, onclick: () => goPlace(id), text: label })),
      // 11f: one underline, which slides from the old tab to the new one
      el('span', { class: 'tab-ink still', id: 'tabInk', 'aria-hidden': 'true' }));
    window.addEventListener('resize', () => placeInk(true));
    const menu = document.getElementById('menuBtn');
    if (menu) popover(menu, () => el('div', { class: 'moremenu placemenu', id: 'pop-places',
        'aria-label': 'places' },
      PLACES.flatMap(([id, label, views]) => [
        el('button', { role: 'menuitem', 'data-place': id, class: 'placeitem',
          'aria-current': placeOf(state.model || state.topic ? '' : state.tab) === id
            ? 'page' : null,
          text: label, onclick: () => { popClose(); goPlace(id); } }),
        ...(views.length > 1 ? views.map(v => el('button', { role: 'menuitem',
          class: 'subitem', 'data-place-sub': v, text: viewLabel(v),
          'aria-current': !state.model && !state.topic && state.tab === v ? 'page' : null,
          onclick: () => { popClose(); navigate({ tab: v, model: null, topic: null }); } }))
          : [])])), { key: 'places' });
  }
  const sel = state.model ? '' : placeOf(state.tab) || '';
  for (const b of tabs.querySelectorAll('button[role=tab][data-tab]'))
    b.setAttribute('aria-selected', String(b.dataset.tab === sel));
  placeInk();
  renderRuns();
  renderTestAct();
}

function placeInk(still) {
  const ink = document.getElementById('tabInk');
  const b = document.querySelector('#tabs [role=tab][aria-selected=true]');
  if (!ink) return;
  if (!b) { ink.style.opacity = '0'; return; }
  if (still) ink.classList.add('still');
  ink.style.opacity = '1';
  // measured against the strip itself, wherever the bar has put it
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
  // 11l: a check about one model opens that model's page — on the block its
  // check is about (12b.2 folds all but the newest)
  if (show.model) {
    if (show.kind) {
      (state.mblk[show.model] = state.mblk[show.model] || {})[show.kind] = true;
      state.mtab = 'scores';
      state.after = { scroll: `[data-kind-block="${show.kind}"]` };
    }
    return navigate({ model: show.model, topic: null });
  }
  // 12b: the Models tab's list is the Models table: its filters say the same
  if (show.tab === 'models') {
    Object.assign(lbS(), { view: 'standard', chip: 'all', stdChip: 'all', models: null,
      kind: show.kind || 'all', size: 'all',
      status: show.tainted ? 'tainted' : show.prelim ? 'preliminary' : 'all' });
    return navigate({ tab: 'leaderboard', model: null, topic: null });
  }
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
  const steady = judgeSteadiness();
  const sig = JSON.stringify([cs.map(c => c.text), steady]);
  if (sig === _warnSig) return;
  _warnSig = sig;
  // 12b.3: the dot counts problems — something to act on. The known limits,
  // standing conditions of the setup, are one folded line under them: an
  // always-amber dot is a dot nobody reads
  const problems = boardProblems(cs), limits = cs.filter(c => c.limit);
  // what kind, not just how many: after a judged run most of them are about
  // the judge, and "5 checks" says nothing about whether to open it
  const judged = problems.filter(c => c.judged).length;
  // 12b: a status dot, not a pill of words — green and nothing else when every
  // check passes, amber with the count when any does not
  const n = problems.length;
  const row = c => el('li', { class: 'check warnrow',
      'data-check': c.key, 'data-severity': c.severity, 'data-limit': c.limit ? '1' : null },
    el('span', { class: 'dot ' + (c.severity === 'warning' && !c.limit ? 'warn' : 'info'),
      title: c.limit ? 'a known limit of the setup' : c.severity === 'warning' ? 'warning'
        : 'for information' }),
    el('span', { class: 'check-short', text: c.short }),
    c.show ? el('a', { href: '#', class: 'small', 'data-show-me': c.key, text: 'Show me',
      onclick: e => { e.preventDefault(); showMe(c.show); } }) : '',
    el('details', { class: 'check-more' },
      el('summary', { class: 'small', text: 'why' }),
      el('p', { class: 'warn', text: c.text })));
  // 12g.1: the 11a popover, as the run counter is — it closes on Escape, a
  // click outside and a page change. The <details> it replaces stayed open
  // through all three
  const btn = el('button', { class: 'barpill statusdot' + (n ? ' warn' : ' ok'),
      'data-warn-summary': String(n),
      'aria-label': n ? `${n} problem${n > 1 ? 's' : ''} to look at` : 'no problems',
      title: n ? `${n} problem${n > 1 ? 's' : ''} to look at` : 'no problems'
        + (limits.length ? ` · ${limits.length} known limit${limits.length > 1 ? 's' : ''}` : '') },
    el('span', { class: 'dot ' + (n ? 'warn' : 'ok') }),
    n ? el('span', { class: 'statusn', text: String(n) }) : '');
  const list = () => el('div', { class: 'moremenu checkspop', id: 'pop-checks',
      'aria-label': 'checks', 'data-warnings': 'open' },
    el('ul', { class: 'checklist' },
      !n ? el('li', { class: 'small', 'data-checks-none': '1', text: 'No problems.' }) : '',
      judged ? el('li', { class: 'small checks-judged', 'data-checks-judged': String(judged),
        text: `${judged} of ${n} ${n > 1 ? 'are' : 'is'} about the judged suite` }) : '',
      problems.map(row),
      limits.length ? el('li', { class: 'known-limits', 'data-known-limits': String(limits.length) },
        el('details', { open: state.limitsOpen ? '' : null,
            ontoggle: e => { state.limitsOpen = e.target.open; } },
          el('summary', { class: 'small', text: `Known limits (${limits.length}) ▸` }),
          el('ul', { class: 'checklist' }, limits.map(row)))) : '',
      steady ? el('li', { class: 'small se', 'data-judge-steady': '1', text: steady }) : ''));
  box.replaceChildren(popover(btn, list, { key: 'checks', menu: false, placement: 'bottom-end',
    rebuild: true }));
  if (POP.key === 'checks') popReanchor();
}

// the checks that are problems — something happened that a person should act
// on — not the known limits of the setup (12b.3). A payload from before 12b.3
// has no `limit`, and every check in it counts
const boardProblems = cs => cs.filter(c => !c.limit);

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
  // 12b: on a phone the badge is the dot and the time — the words go first
  chip.replaceChildren(el('span', { class: 'dot ' + (stale || judge ? 'warn' : 'ok') }),
    el('span', { class: 't-full', text: stale ? 'STALE · ' : 'LIVE · ' }), at,
    judge ? el('span', { class: 't-full', text: ' · judge offline' }) : '');
}
// the last check that worked, as the clock on the wall reads it
const checkedAt = () => NET.lastOk ? new Date(NET.lastOk).toTimeString().slice(0, 5) : refreshedAt();
const POLL_MS = 5000;

// static shell bits (rendered whenever a payload arrives)
function renderStatic() {
  renderWarnings();
  // 12b: Home has no hero. What it said that a reader still needs — when a
  // static report was made, and that a run was a smoke run — sits above the
  // view on every page, and only when there is something to say
  document.getElementById('metaChips').replaceChildren(
    LIVE ? '' : el('span', { class: 'chip', text: `generated ${DATA.generated}` }),
    // 12b: the guides and "How to read these numbers" are in Help
    // the harness build and the transformers version are provenance: they
    // live on that tab now, not on every tab's first line
    DATA.meta.anyLimit ? el('span', { class: 'chip', text: '⚠ smoke data (--limit)' }) : '');
  renderFresh();
  renderWho();
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
    let d;
    try { d = await api('api/results'); }
    catch (e) { return; }              // netFail said so; the poll retries while RESULTS_DUE
    // 12b.3: an error drawing the board is said, not swallowed — a page with
    // its title and footer and nothing else, and no word why, is how the
    // live check found the bare address
    try { initData(d); }
    catch (e) { console.error(e); drawFailed(e); }
    if (ask === RESULTS_ASKED) RESULTS_DUE = false;
  } finally { RESULTS_BUSY = false; }
}
function drawFailed(e) {
  const view = document.getElementById('view');
  if (view) view.replaceChildren(el('div', { class: 'card', 'data-draw-failed': '1' },
    el('p', { class: 'warn' }, el('b', { text: 'The board could not be drawn. ' }),
      `${(e && e.message) || e}. Reload the page; if it happens again, the browser's `
      + 'console has the details.')));
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
    // 12b: the run counter is on every page, and says so as runs move
    if (changed) { renderRuns(); if (POP.key === 'runs') popReanchor(); }
    if (justFinished) await refreshResults();       // new scores -> re-render everything
    else if (changed && state.tab === 'queue') (state.queueRedraw || render)();
    else if (changed && onHome()) render();
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
  // 12b: Theme is in the name menu; its choices say which one is on
  for (const it of document.querySelectorAll('#pop-who [role=menuitemradio][data-theme]'))
    it.setAttribute('aria-checked', String(it.dataset.theme === t));
  try { localStorage.setItem('bench-theme', t); } catch (e) { /* private mode etc. */ }
}
let themeIdx = 0;
try {   // remembered per browser — the dashboard is a page people leave open
  const saved = localStorage.getItem('bench-theme');
  if (THEMES.includes(saved)) themeIdx = THEMES.indexOf(saved);
} catch (e) { /* storage unavailable: stay on auto */ }
applyTheme(THEMES[themeIdx]);

// 12b.3: the header — the four places, the run counter, Test a model and the
// name — needs no scores, so it is drawn before they arrive. The scores are
// the board's biggest answer; on the tailnet the first one took a minute, and
// a page with nothing on it for that long reads as broken
function renderShell() {
  renderTabs();
  renderWho();
}

// boot: embedded data renders immediately; live mode fetches then polls
if (LIVE) {
  renderShell();
  document.getElementById('view').replaceChildren(
    skeleton(6, { 'data-loading': 'results' }));
  refreshResults().then(() => {
    if (DATA && !DATA.models.length) { state.tab = 'queue'; render(); }
    if (DATA) loadViews();
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
    // 12b.2: Home's lists, and a model's Improve tab — only while the service
    // answers: a board that cannot reach it does not ask for four more things
    const answering = DATA && !NET.fails;
    if (answering && onHome()) { loadReview(); loadTruns(); }
    if (state.tab === 'pipeline' || (answering && state.model && state.mtab === 'improve')) loadReview();
    if (state.tab === 'exam') loadExam();
    // 12h.2: a view someone else saved reaches this page within half a minute
    if (answering && state.tab === 'leaderboard' && Date.now() - VIEWS_AT > 30000) loadViews();
    // the Loop board and a topic page: without this nothing ever re-fetched
    // /api/loop, so a board whose first load failed stayed empty for as long
    // as the tab was open — which is exactly what happened on the live tree
    if (state.topic || (state.model && state.msit.open)) loadLoop();
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
    <button class="menubtn barpill" id="menuBtn" type="button" aria-label="places">Menu &#9662;</button>
    <span class="bar-title"><span class="t-full">__TITLE__</span><span class="t-short">Benchmark</span></span>
    <span class="livebadge" id="liveBadge" data-stamp="1" hidden></span>
    <div class="tabs" role="tablist" id="tabs"></div>
    <div class="bar-right" id="barRight">
      <div id="runs"></div>
      <div id="testAct"></div>
      <div id="warnings" class="bar-checks"></div>
      <div id="who"></div>
    </div>
  </div>
</header>
<div class="wrap">
__BANNER__
  <div id="netstatus"></div>
  <div class="meta-chips" id="metaChips"></div>
  <div id="view"></div>
  <button class="totop" id="toTop" hidden>&#8593; Top</button>
  <footer>Every score carries its standard error · differences are z-tested before
  they are called wins · provenance is in Data &amp; sources · scores are only comparable to
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
                 fingerprints: dict | None = None, everyday: dict | None = None) -> Path:
    if not runs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"<h1>No lm-eval results found.</h1><p>{html.escape(banner)}</p>",
                            encoding="utf-8")
        return out_path
    payload = build_payload(merge_runs(runs), title, source="", calibration=calibration,
                            taint=taint, parents=parents, judge_identity=judge_identity,
                            fingerprints=fingerprints, everyday=everyday)
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
    out = build_report(runs, args.out, args.title, calibration=cal, fingerprints=fps,
                       everyday=load_everyday(args.results if args.results.is_dir() else None))
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
