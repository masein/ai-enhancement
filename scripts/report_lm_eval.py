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


def _model_meta(source: Path) -> dict | None:
    """model_meta.json sits at the model's directory level (the service writes
    it at preflight); results files are one or two levels below it."""
    for up in (1, 2):
        if len(source.parents) <= up:
            continue
        cand = source.parents[up] / "model_meta.json"
        key = str(cand)
        if key not in _META_CACHE:
            try:
                _META_CACHE[key] = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                _META_CACHE[key] = None
        if _META_CACHE[key] is not None:
            return _META_CACHE[key]
    return None


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
_CANON = ["mmlu", "hellaswag", "arc_challenge", "arc_easy",
          "winogrande", "piqa", "truthfulqa_mc2", "gsm8k"]
_CHANCE = {"mmlu": 0.25, "hellaswag": 0.25, "arc_challenge": 0.25, "arc_easy": 0.25,
           "winogrande": 0.5, "piqa": 0.5, "gsm8k": 0.0}

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
        have = [cells[t][mid]["v"] for t in acc_tasks if mid in cells.get(t, {})]
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
                      "chance": _CHANCE.get(t)} for t in headline},
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
.tv { display:none; margin-top:12px; } .tv.open { display:block; }
.small { font-size:12px; color:var(--text-secondary); }
.up { color:var(--success-text); } .down { color:var(--critical); }
.note { border-left:2px solid var(--axis); padding:6px 0 6px 12px; margin:12px 0;
  color:var(--text-secondary); font-size:13px; }
.warn { border-left:2px solid var(--warning); padding:6px 0 6px 12px; margin:10px 0;
  color:var(--text-secondary); font-size:13px; }
.warn b, .note b { color:var(--text-primary); }
.lb td.model { white-space:nowrap; }
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
  trMetricQ: '', trSecClosed: {},                          // metric panels filter / sections
  cmpSel: [], cmpColors: {},                               // radar: compared models (≤3)
  radarNorm: 'chance', radarAxes: 'tasks',                 // radar scaling / axis mode
  avgMode: 'chance',                   // official average: above-chance | raw
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
const taskLabel = t => {
  const info = DATA.tasks[t] || {};
  return t + (info.metric ? ` (${info.metric})` : '');
};

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

  const W = 460, LBL = 150, PAD = 56, BH = 15, GAP = 7;
  const TOP = hasChance ? 20 : 8;          // reserve headroom for the chance label
  const plotW = W - LBL - PAD;
  const H = shown.length * (BH + GAP) + TOP + 16;
  const maxv = Math.max(...shown.map(r => r.c.v + (r.c.se || 0)), info.chance || 0);
  // scale to the data, not to a fixed floor — gsm8k at 2% must not be squashed
  // into an axis drawn for 25%-chance tasks
  const hi = lower ? maxv * 1.15
                   : Math.min(1, Math.max(maxv * 1.15, (info.chance || 0) * 1.25, 0.05));
  const X = v => LBL + plotW * Math.max(0, Math.min(v, hi)) / hi;
  const fmt = lower ? (v => num(v, 3)) : (v => pct(v));
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%',
                              role: 'img', 'aria-label': task });
  for (const t of niceTicks(hi, 4)) {
    svg.append(el('svg:line', { x1: X(t), y1: TOP - 4, x2: X(t), y2: H - 18,
      stroke: 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('svg:text', { x: X(t), y: H - 5, 'font-size': 10,
      fill: 'var(--muted)', 'text-anchor': 'middle',
      text: lower ? num(t, 2) : Math.round(100 * t) + '%' }));
  }
  if (hasChance) {
    svg.append(el('svg:line', { x1: X(info.chance), y1: 14, x2: X(info.chance), y2: H - 18,
      stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
    svg.append(el('svg:text', { x: X(info.chance) + 4, y: 10, 'font-size': 9.5,
      fill: 'var(--muted)', text: 'chance ' + Math.round(100 * info.chance) + '%' }));
  }
  let y = TOP;
  for (const { m, c } of shown) {
    const dim = noisy(c);                  // within noise of chance: still there, but quiet
    const isCk = m.source === 'artifact';  // hollow = uploaded checkpoint, same hue
    const w = Math.max(2, X(c.v) - LBL), r = Math.min(4, w);
    const name = m.name.length > 22 ? m.name.slice(0, 21) + '…' : m.name;
    svg.append(el('svg:text', { x: LBL - 8, y: y + BH * 0.75, 'font-size': 11.5,
      fill: dim ? 'var(--muted)' : 'var(--text-secondary)', 'text-anchor': 'end', class: 'blab',
      'data-model': m.id, text: name }));
    svg.append(el('svg:path', { class: 'bar', 'data-model': m.id,
      d: `M${LBL},${y} H${LBL + w - r} q${r},0 ${r},${r} V${y + BH - r} q0,${r} -${r},${r} H${LBL} Z`,
      opacity: dim ? 0.45 : null,
      'fill-opacity': isCk ? 0.28 : null,
      stroke: isCk ? 'var(--s1)' : null,
      'stroke-width': isCk ? 1.2 : null,
      fill: 'var(--s1)' }));
    if (c.se > 0 && !lower) {
      const lo = X(Math.max(0, c.v - c.se)), hx = X(c.v + c.se), cy = y + BH / 2;
      svg.append(el('svg:line', { x1: lo, y1: cy, x2: hx, y2: cy,
        stroke: 'var(--text-primary)', 'stroke-width': 1.4, opacity: 0.55 }));
      for (const xx of [lo, hx])
        svg.append(el('svg:line', { x1: xx, y1: cy - 3.5, x2: xx, y2: cy + 3.5,
          stroke: 'var(--text-primary)', 'stroke-width': 1.4, opacity: 0.55 }));
    }
    svg.append(el('svg:text', { x: X(c.v + (lower ? 0 : c.se || 0)) + 6, y: y + BH * 0.75,
      'font-size': 11, fill: dim ? 'var(--muted)' : 'var(--text-primary)',
      class: 'blab', 'data-model': m.id,
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
          + (top.params ? ` · ${P(top.params)} params` : '') })),
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
// the compared set: explicit ticks, else the top three by average
function cmpEffective(ms) {
  if (state.cmpSel.length) return state.cmpSel.filter(id => ms.some(m => m.id === id));
  // prefer official models; fall back to partial averages so a report with no
  // official result still shows a profile rather than an empty card
  return [...ms].filter(m => m.partialAvg != null)
    .sort((a, b) => (officialAvg(b) ?? -1) - (officialAvg(a) ?? -1)
                 || b.partialAvg - a.partialAvg)
    .slice(0, 3).map(m => m.id);
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
    if (state.cmpSel.length >= 3) {  // FIFO: the newest tick always lands
      const old = state.cmpSel.shift(); delete state.cmpColors[old];
    }
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
    el('thead', {}, el('tr', {}, el('th', { text: 'axis' }),
      series.map(s => el('th', { class: 'num', text: s.m.name })))),
    el('tbody', {}, axes.map(ax => el('tr', {}, el('td', { text: ax.label }),
      series.map(s => el('td', { class: 'num', text: s.vals[ax.key] ? pct(s.vals[ax.key].n) : '—' }))))));
  return el('div', { class: 'card' },
    el('h2', { text: 'Capability profile' }),
    el('p', { class: 'sub', text:
      'One axis per benchmark, one shape per model — tick up to three in the table below '
      + '(a fourth tick replaces the oldest). Axes are scaled ABOVE CHANCE by default: 25% on a '
      + '4-way task is 0, perfect is 100%, so a 4-way and a 2-way task are comparable; switch to '
      + 'raw accuracy to quote the number itself. Perplexity tasks are excluded (different scale). '
      + 'Read the shape here and the numbers below — a radar’s area exaggerates differences and '
      + 'its shape depends on axis order.' }),
    el('div', { class: 'ctrl', style: 'margin-top:8px' },
      seg('scale', [['chance', 'above chance'], ['raw', 'raw accuracy']], state.radarNorm,
          v => { state.radarNorm = v; render(); }),
      seg('axes', [['tasks', 'tasks'], ['categories', 'categories']], state.radarAxes,
          v => { state.radarAxes = v; render(); })),
    el('div', { class: 'radar-grid' },
      el('div', {}, svg, legend),
      el('div', { class: 'lb-wrap' }, table)));
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
  const best = {}, tiedCount = {};
  for (const c of cols) {
    if (!c.num || c.key === 'params') continue;
    const vs = ms.map(m => val(m, c)).filter(v => v != null);
    if (vs.length > 1) {
      best[c.key] = c.lower ? Math.min(...vs) : Math.max(...vs);
      tiedCount[c.key] = c.lower
        ? vs.filter(v => v <= best[c.key] + pplBand(best[c.key])).length : 1;
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
      class: (c.num ? 'num ' : '') + 'sortable' + (c.key === 'name' ? ' model' : ''),
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
        title: m.id + (m.archinfo && m.archinfo.hidden
          ? `\n${m.archinfo.arch || ''} · hidden ${m.archinfo.hidden} · layers ${m.archinfo.layers} · vocab ${m.archinfo.vocab}` : '') },
        m.name, ckBadge(m) || (m.kind === 'instruct'
          ? el('span', { class: 'badge instruct', text: 'instruct' })
          : el('span', { class: 'badge', text: 'base' })),
        prelimBadge(m) || '');
      if (c.key === 'params') return el('td', { class: 'num',
        title: m.paramsSrc ? 'from ' + (m.paramsSrc === 'config' ? 'harness config' : 'model name') : '',
        text: P(m.params) });
      if (c.key === 'date') return el('td', { class: 'small', style: 'white-space:nowrap',
        text: String(m.date || '—').slice(0, 16).replace('T', ' ') });
      if (c.key === 'avg') {
        const a = officialAvg(m);
        if (a == null) return el('td', { class: 'num' },
          el('span', { class: 'se',
            title: (m.missing || []).length ? 'missing: ' + m.missing.join(', ') : '',
            text: `— ${m.nhave}/${m.nreq}` }));
        return el('td', { class: 'num' + (a === best.avg ? ' best' : ''),
          title: `mean over the ${m.nreq} required tasks, `
            + (state.avgMode === 'raw' ? 'raw accuracy' : 'scaled so chance = 0') },
          pct(a), el('span', { class: 'se', text: ` ${m.nreq}/${m.nreq}` }));
      }
      const cc = cell(c.task, m.id);
      if (!cc) return el('td', { class: 'num', text: '—' });
      const lead = best[c.key];
      const within = lead != null && (c.lower
        ? cc.v <= lead + pplBand(lead) : cc.v === lead);
      const mark = !within ? '' : (c.lower && tiedCount[c.key] > 1) ? ' tiebest' : ' best';
      return el('td', { class: 'num' + mark,
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
    el('div', { class: 'lb-wrap' }, el('table', { class: 'lb' }, thead, tbody)))];
}

function vTasks(ms) {
  if (!DATA.accTasks.length) return [note('No accuracy tasks found.')];
  return [
    el('p', { class: 'sub', style: 'margin:10px 2px', text:
      'One panel per benchmark, models ranked. Bars share one hue on purpose — the label is the identity; '
      + 'pointing at any model highlights it in every panel. Dashed line = chance.'
      + (anyCk() ? ' Hollow bars are uploaded checkpoints.' : '') }),
    el('div', { class: 'panels' }, DATA.accTasks.map(t => barPanel(t, ms, { lower: false }))),
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
              : (v / 1e9).toFixed(2) + 'B';
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
  const xmin = Math.min(...allPts.map(p => p[0])), xmax = Math.max(...allPts.map(p => p[0]));
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (ymin === ymax) { ymin -= 0.5; ymax += 0.5; }
  const pad = (ymax - ymin) * 0.06;
  ymin -= pad; ymax += pad;
  const X = x => L + (W - L - R) * (x - xmin) / ((xmax - xmin) || 1);
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
    svg.append(el('svg:text', { x: X(xv), y: H - 8, 'font-size': 10,
      fill: 'var(--muted)', 'text-anchor': 'middle', text: Math.round(xv).toLocaleString() }));
  }
  // checkpoint markers first (under the data)
  for (const s of smoothed) for (const ev of (s.events || [])) {
    if (ev.step < xmin || ev.step > xmax) continue;
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
      const fx = xmin + ((e.clientX - box.left) / box.width * W - L) / (W - L - R) * (xmax - xmin);
      let lo = 0, hi = steps.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; steps[mid] < fx ? lo = mid + 1 : hi = mid; }
      const stp = (lo > 0 && fx - steps[lo - 1] < steps[lo] - fx) ? steps[lo - 1] : steps[lo];
      hair.setAttribute('x1', X(stp)); hair.setAttribute('x2', X(stp));
      const rows = [`step ${stp.toLocaleString()} — ${title}`];
      for (const s of smoothed) {
        let best = null;
        for (const p of s.spts) if (best === null || Math.abs(p[0] - stp) < Math.abs(best[0] - stp)) best = p;
        if (best && Math.abs(best[0] - stp) <= (xmax - xmin) * 0.05 + 1)
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
      const all = [...new Set(selRuns.flatMap(x => Object.keys(x.det?.metrics || {})))];
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
        const series = DATA.accTasks.map((t, i) => ({
          label: t, color: trColor(i),
          pts: evs.map(ev => cell(t, ev.detail) ? [ev.step, cell(t, ev.detail).v] : null)
                  .filter(Boolean),
        })).filter(s => s.pts.length);
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
          const panel = series.length
            ? lineChart('benchmark score vs step', series, {})
            : note('Checkpoints are queued — scores appear here when evaluation finishes.');
          out.push(el('div', { class: 'card' },
            el('h2', { text: 'Benchmarks along this run' }),
            el('p', { class: 'sub', text: statusLine + '. Every checkpoint this run submitted, '
              + 'joined to its scores on the leaderboard — capability versus training step, '
              + 'next to the loss.' }),
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
      el('option', { value: 'quick', text: 'quick — hellaswag + arc_easy + ppl, minutes' })),
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
                   'template_policy', 'dtype', 'backend', 'batch_size', 'seed',
                   'limit', 'harness_git', 'transformers', 'eval_finished'];
function provOf(name) {
  const m = DATA.models.find(x => x.name === name) || {};
  const a = m.archinfo || {};
  return [m.official ? 'official' : 'preliminary',
          m.nreq ? `${m.nhave}/${m.nreq}` : '', m.chat ? 'yes' : 'no',
          a.tmpl_sha || '', m.kindReason || '', m.dtype || '', m.backend || '',
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
    el('button', { role: 'tab', 'aria-selected': String(state.tab === id),
      onclick: () => { state.tab = id; render(); }, text: label })));
  const view = document.getElementById('view');
  view.classList.remove('dimmed');
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
  const ckBtn = srcSeg.querySelector('[data-src="artifact"]');
  if (ckBtn) ckBtn.textContent = `Checkpoints (${nCk})`;
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
