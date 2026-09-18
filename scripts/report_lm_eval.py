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

    n_samples = blob.get("n-samples") or {}
    # The harness tells us which tasks are children of a group (MMLU's 57 subjects,
    # its four category roll-ups, etc). Use it rather than pattern-matching names.
    subtasks: set[str] = set()
    for parent, kids in (blob.get("group_subtasks") or {}).items():
        for k in kids or []:
            if k != parent:
                subtasks.add(k)
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
        "n_shot": blob.get("n-shot") or {},
        "n_samples": {k: (v.get("effective") if isinstance(v, dict) else v)
                      for k, v in n_samples.items()},
        "git_hash": blob.get("git_hash"),
        "date": _norm_date(blob.get("date")),
        "transformers_version": blob.get("transformers_version"),
        "eval_seconds": _to_float(blob.get("total_evaluation_time_seconds")),
        "archinfo": _model_meta(source),
        "diag": _beside(source, "diagnose.json"),
        "tasks": tasks,
    }


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
    return by_model


def build_payload(by_model: dict[str, dict], title: str, source: str) -> dict:
    models = list(by_model)

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

    acc_tasks = [t for t in headline
                 if metric_used.get(t) in PROPORTION and not is_lower_better(t)]
    ppl_tasks = [t for t in headline if t not in acc_tasks]

    # official vs preliminary: the average is over the REQUIRED list or it does
    # not exist. A model missing any required task gets no overall number.
    required, req_absent = required_tasks(acc_tasks)
    model_rows = []
    for mid, r in by_model.items():
        have = [cells[t][mid]["v"] for t in acc_tasks
                if mid in cells.get(t, {}) and t not in CONTROL_TASKS]
        got_req = [t for t in required if mid in cells.get(t, {})]
        missing = [t for t in required if t not in got_req]
        official = bool(required) and not missing
        params = r["num_params"] or params_from_name(mid)
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
            "diag": _trim_diag(r.get("diag")),
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
    const ranked = DATA.models.filter(x => officialAvg(x) != null)
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
  if (r && a != null)
    out.push(`It averages ${pct(a)} `
      + `${state.avgMode === 'raw' ? 'raw' : 'above chance'} over the ${m.nreq} `
      + `required tasks, ranking ${ord(r.n)} of ${r.of} ranked models here.`);
  else
    out.push(`It has ${m.nhave} of ${m.nreq} required tasks, so it is preliminary `
      + `and carries no overall rank`
      + ((m.missing || []).length ? ` — still missing ${m.missing.join(', ')}.` : '.'));
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
const prelimBadge = m => m.official ? null
  : el('span', { class: 'badge prelim',
      title: `preliminary — ${m.nhave}/${m.nreq} required tasks`
        + (m.missing && m.missing.length ? `\nmissing: ${m.missing.join(', ')}` : '')
        + '\nPer-task scores are valid; there is no overall average until the '
        + 'required suite completes.',
      text: `prelim ${m.nhave}/${m.nreq}` });
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
function dxCategories(t, v, atChance) {
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
    const det = el('details', { class: 'dxcat' + (dim ? ' dim' : ''), 'data-cat': name },
      el('summary', {},
        el('span', { class: 'dxcname', text: name }),
        el('span', { class: 'num', text: pct(g.score_report) }),
        el('span', { class: 'se', text: `${g.n_report} items`
          + (dim ? ` · under ${CAT_MIN_N}, noise` : '') }),
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
    el('p', { class: 'sub', text: 'Why the score is what it is, read off the per-item '
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
      body.append(dxCategories(t, v, causes[t].some(c => c.key === 'atchance')));
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
      prelimBadge(m) || '',
      r ? el('span', { class: 'rankbadge', title: `${r.n} of ${r.of} ranked models`,
        text: `#${r.n}` }) : ''),
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

  // vDiagnose is null when no model on this board has a per-item diagnosis
  return [back, head, results, vDiagnose(m), provCard].filter(Boolean);
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
                                  : 'tab=' + state.tab;

function routeFromHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ''));
  const m = /^model=(.+)$/.exec(h);
  if (m && DATA.models.some(x => x.id === m[1])) { state.model = m[1]; return; }
  state.model = null;
  const t = /^tab=(.+)$/.exec(h);
  if (t && TABS.some(([id]) => id === t[1])) state.tab = t[1];
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

function vLeaderboard(ms) {
  const cmpSet = new Set(cmpEffective(ms));
  const cols = [
    { key: 'cmp',    label: '', nosort: true },
    { key: 'name',   label: 'Model',  num: false },
    { key: 'params', label: 'Params', num: true },
    { key: 'avg',    label: 'Avg',    num: true },
    ...DATA.accTasks.map(t => ({ key: t, label: t, num: true, task: t })),
    ...DATA.pplTasks.map(t => ({ key: t, label: t, num: true, task: t, lower: true })),
    { key: 'date', label: 'Last eval', num: false },   // when its newest task ran
  ];
  const val = (m, c) => c.key === 'avg' ? officialAvg(m)
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
           + (c.task && (DATA.tasks[c.task] || {}).desc ? ' hasinfo' : ''),
      // the description on the column itself; the full list is in the panel
      // below the table, because a tooltip is not documentation
      title: c.task ? [(DATA.tasks[c.task] || {}).control ? 'CONTROL — never in Avg' : null,
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
      text: c.task ? (c.lower ? DATA.tasks[c.task].metric : shotOf(c.task)) : '' }))));
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
        prelimBadge(m) || '');
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
  return [radarCard(ms) || '', el('div', { class: 'card' },
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
  if (DATA.warnings.length)
    frag.push(el('div', { class: 'card' }, el('h2', { text: 'Warnings' }),
      DATA.warnings.map(w => el('div', { class: 'warn', text: w }))));
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
    state.trRuns = await (await fetch('api/truns')).json();
    await Promise.all(state.trSel.map(async id => {
      const row = state.trRuns.find(r => r.id === id);
      if (force || !state.trSeries[id] || (row && row.status === 'running'))
        state.trSeries[id] = await (await fetch(`api/truns/${id}`)).json();
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
  if (!state.trRuns.length && LIVE) loadTraining();
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
  const f = {
    hf_id: el('input', { type: 'text', style: 'flex:2;min-width:260px',
      placeholder: 'org/model on the Hub, or local/<name> for an uploaded artifact' }),
    kind: el('select', {}, ['auto', 'base', 'instruct'].map(v =>
      el('option', { value: v, text: v === 'auto' ? 'kind: auto-detect' : 'kind: ' + v }))),
    suite: el('select', {},
      el('option', { value: 'full', text: 'full — all tasks, comparable' }),
      el('option', { value: 'quick', text: 'quick — hellaswag + arc_easy + ppl, minutes' }),
      el('option', { value: 'control', text: 'control — mmlu_perm only: MMLU with the '
        + 'options rotated (the position-bias experiment), ~a fifth of a full MMLU' })),
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
    el('td', { text: r.suite }),
    el('td', { text: r.submitter || '—' }),
    el('td', {}, el('span', { class: stClass(r.status), text: r.status })),
    el('td', { class: 'small', text: r.progress || '' },
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
const TABS = [
  ['overview', 'Overview', vOverview],
  ...(LIVE ? [['training', 'Training', vTraining],
              ['queue', 'Submit & Queue', vQueue]] : []),
  ['leaderboard', 'Leaderboard', vLeaderboard],
  ['tasks', 'Tasks', vTasks],
  ['perplexity', 'Perplexity & Loss', vPpl],
  ['runs', 'Evals', vRuns],
];
function render() {
  // full rebuild: drop the in-place refreshers so a poll can never touch the
  // DOM of a tab that just got torn down — the mounted tab re-registers its own
  state.trRedraw = state.queueRedraw = null;
  const ms = visible();
  document.getElementById('countNote').textContent =
    `${ms.length} of ${DATA.models.length} models shown`;
  const tabs = document.getElementById('tabs');
  tabs.replaceChildren(...TABS.map(([id, label]) =>
    el('button', { role: 'tab',
      'aria-selected': String(!state.model && state.tab === id),
      onclick: () => navigate({ tab: id, model: null }), text: label })));
  const view = document.getElementById('view');
  view.classList.remove('dimmed');
  if (state.model) { view.replaceChildren(...vModel()); return; }
  const fn = TABS.find(([id]) => id === state.tab)[2];
  view.replaceChildren(...fn(ms));
}

// static shell bits (rendered whenever a payload arrives)
function renderStatic() {
  if (LIVE) document.getElementById('pageSub').textContent =
    'Live team benchmark: submit models, track training runs, compare results — '
    + 'updates as work finishes.';
  document.getElementById('warnings').replaceChildren(
    ...DATA.warnings.map(w => el('div', { class: 'warn' }, el('b', { text: 'Check: ' }), w)));
  const nCk = DATA.models.filter(m => m.source === 'artifact').length;
  const srcSeg = document.getElementById('srcSeg');
  srcSeg.style.display = nCk ? '' : 'none';   // no artifacts -> no third filter
  // every filter says how many it holds, not just the one that happened to have
  // a count. A filter that might return nothing should say so before it is clicked.
  const setCount = (seg, attr, val, label, n) => {
    const b = seg.querySelector(`[${attr}="${val}"]`);
    if (b) b.textContent = `${label} (${n})`;
  };
  const kindSeg = document.getElementById('kindSeg');
  setCount(kindSeg, 'data-kind', 'all', 'All', DATA.models.length);
  setCount(kindSeg, 'data-kind', 'base', 'Base',
           DATA.models.filter(m => m.kind === 'base').length);
  setCount(kindSeg, 'data-kind', 'instruct', 'Instruct',
           DATA.models.filter(m => m.kind === 'instruct').length);
  setCount(srcSeg, 'data-src', 'all', 'All', DATA.models.length);
  setCount(srcSeg, 'data-src', 'hub', 'Models',
           DATA.models.filter(m => m.source === 'hub').length);
  setCount(srcSeg, 'data-src', 'artifact', 'Checkpoints', nCk);
  document.getElementById('metaChips').replaceChildren(
    el('span', { class: 'chip', text: `generated ${DATA.generated}` }),
    LIVE ? el('span', { class: 'chip', text: 'live — updates as runs finish' }) : '',
    LIVE ? el('a', { class: 'chip', href: 'guide', target: '_blank', rel: 'noopener',
                     style: 'text-decoration:none', text: '📖 guide for new users' }) : '',
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

async function refreshResults() {
  try { initData(await (await fetch('api/results')).json()); } catch (e) { /* next poll retries */ }
}

async function loadQueue() {
  try {
    const rows = await (await fetch('api/submissions?limit=100')).json();
    const prev = new Map(state.queue.map(r => [r.id, r.status]));
    const justFinished = rows.some(r => r.status === 'done' && prev.get(r.id)
                                        && prev.get(r.id) !== 'done');
    const changed = rows.length !== state.queue.length || rows.some(r => {
      const p = state.queue.find(o => o.id === r.id);
      return !p || p.status !== r.status || p.progress !== r.progress;
    });
    state.queue = rows;
    if (justFinished) await refreshResults();       // new scores -> re-render everything
    else if (changed && state.tab === 'queue') (state.queueRedraw || render)();
  } catch (e) { /* server briefly away; keep polling */ }
}

document.getElementById('q').addEventListener('input', e => { state.q = e.target.value; render(); });
document.getElementById('kindSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  state.kind = b.getAttribute('data-kind');
  for (const x of e.currentTarget.querySelectorAll('button'))
    x.setAttribute('aria-pressed', String(x === b));
  render();
});
document.getElementById('srcSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  state.src = b.getAttribute('data-src');
  for (const x of e.currentTarget.querySelectorAll('button'))
    x.setAttribute('aria-pressed', String(x === b));
  render();
});
const THEMES = ['auto', 'light', 'dark', 'dim'];
function applyTheme(t) {
  if (t === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', t);
  document.getElementById('themeBtn').textContent = 'Theme: ' + t;
  try { localStorage.setItem('bench-theme', t); } catch (e) { /* private mode etc. */ }
}
let themeIdx = 0;
try {   // remembered per browser — the dashboard is a page people leave open
  const saved = localStorage.getItem('bench-theme');
  if (THEMES.includes(saved)) { themeIdx = THEMES.indexOf(saved); applyTheme(saved); }
} catch (e) { /* storage unavailable: stay on auto */ }
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
  setInterval(() => { loadQueue(); if (state.tab === 'training') loadTraining(); }, 5000);
} else {
  initData(DATA);
}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>__CSS__</style></head>
<body class="viz-root"><div id="tip" role="status"></div><div class="wrap">
  <div class="topbar">
    <div>
      <h1>__TITLE__</h1>
      <p class="sub" id="pageSub">lm-evaluation-harness results, one self-contained
      file — data embedded, charts drawn locally, nothing fetched.</p>
      <div class="meta-chips" id="metaChips"></div>
    </div>
    <button id="themeBtn" title="cycle auto / light / dark / dim — remembered in this browser">Theme: auto</button>
  </div>
  <div id="warnings"></div>
  <div class="filters">
    <input type="search" id="q" placeholder="Filter models&hellip;" aria-label="filter models">
    <div class="seg" role="group" aria-label="model kind" id="kindSeg">
      <button data-kind="all" aria-pressed="true">All</button>
      <button data-kind="base" aria-pressed="false">Base</button>
      <button data-kind="instruct" aria-pressed="false">Instruct</button>
    </div>
    <div class="seg" role="group" aria-label="model source" id="srcSeg" style="display:none">
      <button data-src="all" aria-pressed="true">All</button>
      <button data-src="hub" aria-pressed="false">Models</button>
      <button data-src="artifact" aria-pressed="false">Checkpoints</button>
    </div>
    <span class="count-note" id="countNote"></span>
  </div>
  <div class="tabs" role="tablist" id="tabs"></div>
  <div id="view"></div>
  <footer>Every score carries its standard error; differences are z-tested before
  they are called wins; provenance is in the Runs tab. Scores are only comparable to
  published numbers when n-shot, prompt template and metric all match.</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>__JS__SLOT__</script>
</body></html>"""


def build_report(runs: list[dict], out_path: Path, title: str) -> Path:
    if not runs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("<h1>No lm-eval results found.</h1>", encoding="utf-8")
        return out_path
    payload = build_payload(merge_runs(runs), title, source="")
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = (TEMPLATE
            .replace("__TITLE__", html.escape(title))
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

    out = build_report(runs, args.out, args.title)
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
