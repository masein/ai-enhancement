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
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# parsing lm-eval output
# ---------------------------------------------------------------------------

# lm-eval writes metrics as "<metric>,<filter>" e.g. "acc,none", "acc_norm,none",
# with a matching "<metric>_stderr,<filter>". We parse generically rather than
# hard-coding metric names, so new tasks and metrics work without a code change.
_METRIC_RE = re.compile(r"^(?P<metric>[a-zA-Z0-9_@\-]+?)(?P<stderr>_stderr)?,(?P<filter>.+)$")


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

    tasks: dict[str, dict] = {}
    for task, metrics in (blob.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        entry: dict = {"alias": (metrics.get("alias") or task).strip()}
        for key, val in metrics.items():
            mm = _METRIC_RE.match(key)
            if not mm or not isinstance(val, (int, float)):
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
        return _dt.datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M:%S")
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

    # per-model average over the accuracy tasks it has (leaderboard sort key; the
    # dashboard shows n/total so a partial model's average is visibly partial)
    model_rows = []
    for mid, r in by_model.items():
        have = [cells[t][mid]["v"] for t in acc_tasks if mid in cells.get(t, {})]
        params = r["num_params"] or params_from_name(mid)
        model_rows.append({
            "id": mid, "name": display[mid],
            "family": re.split(r"[^a-z0-9]", mid.split("/")[-1].lower())[0],
            "kind": "instruct" if r["chat_template"] else "base",
            "params": params,
            "paramsSrc": ("config" if r["num_params"] else
                          "name" if params is not None else None),
            "backend": r["backend"], "dtype": r["dtype"],
            "batch": r["batch_size"], "chat": r["chat_template"],
            "seed": r["seed"], "limit": r["limit"],
            "minutes": round((r["eval_seconds"] or 0) / 60, 1),
            "hash": r["git_hash"], "date": r["date"],
            "avg": (sum(have) / len(have)) if have else None,
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
    hashes = sorted({r["git_hash"] for r in by_model.values() if r["git_hash"]})
    if len(hashes) > 1:
        warnings.append(
            f"Results come from {len(hashes)} different harness builds "
            f"({', '.join(hashes)}). A benchmark whose code changed is a different "
            f"benchmark — treat cross-build comparisons with suspicion.")

    dates = sorted(str(r["date"]) for r in by_model.values() if r["date"])
    return {
        "title": title,
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "models": model_rows,
        "accTasks": acc_tasks,
        "pplTasks": ppl_tasks,
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
* { box-sizing:border-box; }
body { margin:0; background:var(--plane); color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px; line-height:1.5; }
.wrap { max-width:1180px; margin:0 auto; padding:26px 22px 70px; }
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
.lb th.sortable { cursor:pointer; user-select:none; }
.lb th.sortable:hover { color:var(--text-primary); }
.lb th .dir { font-size:9px; }
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
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
  color:var(--text-secondary); }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 10px; }
.legend span { display:inline-flex; align-items:center; gap:6px; font-size:12px;
  color:var(--text-secondary); }
.key { width:11px; height:11px; border-radius:3px; display:inline-block; }
.key.line { width:14px; height:0; border-top:3px solid; border-radius:2px; }
.tchip { display:inline-flex; align-items:center; gap:7px; font-size:12.5px;
  border:1px solid var(--border); border-radius:999px; padding:4px 12px;
  background:var(--surface-1); color:var(--text-secondary); cursor:pointer; }
.tchip[aria-pressed="true"] { color:var(--text-primary); border-color:var(--axis);
  background:var(--accent-soft); }
.tchip i { width:12px; border-top:3px solid; border-radius:2px; display:inline-block; }
.tchip.off i { border-color:var(--muted) !important; }
.chips { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 12px; }
.panels { display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:12px; }
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
const DATA = JSON.parse(document.getElementById('data').textContent);
const SLOTS = 8;                       // fixed categorical palette; never cycled
const state = {
  q: '', kind: 'all', tab: 'overview',
  sort: { key: 'avg', dir: -1 },
  scal: null,                          // Set of task names shown on the scaling chart
  sigTask: (DATA.accTasks[0] || null),
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
const slotColor = i => `var(--s${(i % SLOTS) + 1})`;
// Fixed task -> color-slot assignment, informative-defaults first: the four tasks
// selected by default on the Scaling chart occupy slots 1-4, the palette's
// validated opening chain. The mapping never changes with filtering or toggling —
// color follows the task, not its current row number.
const SLOT_ORDER = (() => {
  const pref = ['hellaswag', 'arc_easy', 'piqa', 'winogrande']
    .filter(t => DATA.accTasks.includes(t));
  return [...pref, ...DATA.accTasks.filter(t => !pref.includes(t))];
})();
const slotOf = t => SLOT_ORDER.indexOf(t);
const cell = (t, m) => (DATA.cells[t] || {})[m];
const taskLabel = t => {
  const info = DATA.tasks[t] || {};
  return t + (info.metric ? ` (${info.metric})` : '');
};

function visible() {
  const q = state.q.toLowerCase();
  return DATA.models.filter(m =>
    (state.kind === 'all' || m.kind === state.kind) &&
    (!q || m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q) ||
     m.family.includes(q)));
}

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
  const panel = el('div', { class: 'panel' },
    el('h3', { text: taskLabel(task) }),
    el('div', { class: 'pmeta', text:
      (lower ? 'lower is better' : 'higher is better')
      + (shots.length ? ` · ${shots.length > 1 ? 'MIXED n-shot!' : shots[0] + '-shot'}` : '')
      + (ns.length === 1 ? ` · ${ns[0]} items` : '')
      + (rows.length < models.length ? ` · ${models.length - rows.length} model(s) missing` : '') }));
  if (!rows.length) { panel.append(el('p', { class: 'small', text: 'no data' })); return panel; }

  const W = 460, LBL = 150, PAD = 56, BH = 15, GAP = 7;
  const hasChance = info.chance != null && info.chance > 0 && !lower;
  const TOP = hasChance ? 20 : 8;          // reserve headroom for the chance label
  const plotW = W - LBL - PAD;
  const H = rows.length * (BH + GAP) + TOP + 16;
  const maxv = Math.max(...rows.map(r => r.c.v + (r.c.se || 0)), info.chance || 0);
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
  for (const { m, c } of rows) {
    const w = Math.max(2, X(c.v) - LBL), r = Math.min(4, w);
    const name = m.name.length > 22 ? m.name.slice(0, 21) + '…' : m.name;
    svg.append(el('svg:text', { x: LBL - 8, y: y + BH * 0.75, 'font-size': 11.5,
      fill: 'var(--text-secondary)', 'text-anchor': 'end', class: 'blab',
      'data-model': m.id, text: name }));
    svg.append(el('svg:path', { class: 'bar', 'data-model': m.id,
      d: `M${LBL},${y} H${LBL + w - r} q${r},0 ${r},${r} V${y + BH - r} q0,${r} -${r},${r} H${LBL} Z`,
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
      'font-size': 11, fill: 'var(--text-primary)', class: 'blab', 'data-model': m.id,
      text: fmt(c.v) }));
    svg.append(el('svg:rect', { class: 'hit', x: 0, y: y - GAP / 2, width: W,
      height: BH + GAP, tabindex: 0, 'data-model': m.id,
      'data-tip': JSON.stringify([
        fmt(c.v) + (c.se ? ` ± ${lower ? num(c.se, 3) : (100 * c.se).toFixed(1) + ' pts'}` : ''),
        `${task} — ${c.shots != null ? c.shots + '-shot, ' : ''}${c.n != null ? c.n + ' items' : ''}`,
        m.id + (m.params ? ` · ${P(m.params)} params` : '')]) }));
    y += BH + GAP;
  }
  panel.append(svg);
  return panel;
}

// ---------- views ----------
function vOverview(ms) {
  const frag = [];
  const ranked = ms.filter(m => m.avg != null).sort((a, b) => b.avg - a.avg);
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
        el('p', { class: 'sub', text: 'Best average across ' +
          (top.navg === DATA.accTasks.length ? `all ${top.navg}` : `${top.navg} of ${DATA.accTasks.length}`)
          + ' accuracy tasks' }),
        el('div', { class: 'hero', text: pct(top.avg) }),
        el('p', { class: 'sub', text: top.id
          + (top.params ? ` · ${P(top.params)} params` : '') })),
      el('div', { class: 'card' },
        el('h2', { text: 'Top models' }),
        lbMini(ranked.slice(0, 5)))));
  }
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
        + `The Significance tab has the full matrix — ${pairs - real} of ${pairs} comparisons here are inside the noise.` })));
  }
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'How to read these numbers' }),
    note('Chance is not zero. 4-option tasks (MMLU, ARC, HellaSwag) sit at 25% for a model that knows nothing; 2-option tasks (Winogrande, PIQA) sit at 50%. A "50%" that looks respectable may be a coin flip.'),
    note('GSM8K near zero is a finding, not a failure — sub-billion models mostly cannot do written arithmetic. TruthfulQA is famous for NOT improving with scale.'),
    note('Perplexity (bits per byte) is the scale-sensitive metric here: it separates models that multiple-choice tasks cannot tell apart, and it works on base models with no prompt format at all. Lower is better.'),
    note('Whiskers are ±1 standard error. If two whiskers overlap, do not call a winner — check the Significance tab; the z-test is the arbiter.')));
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
      m.kind === 'instruct' ? el('span', { class: 'badge instruct', text: 'instruct' }) : ''),
    el('td', { class: 'num', text: P(m.params) }),
    el('td', { class: 'num best', text: pct(m.avg) }))));
  return el('table', {},
    el('thead', {}, el('tr', {},
      el('th', { text: '#' }), el('th', { text: 'model' }),
      el('th', { class: 'num', text: 'params' }), el('th', { class: 'num', text: 'avg' }))), tb);
}

function vLeaderboard(ms) {
  const cols = [
    { key: 'name',   label: 'Model',  num: false },
    { key: 'params', label: 'Params', num: true },
    { key: 'avg',    label: 'Avg',    num: true },
    ...DATA.accTasks.map(t => ({ key: t, label: t, num: true, task: t })),
    ...DATA.pplTasks.map(t => ({ key: t, label: t, num: true, task: t, lower: true })),
  ];
  const val = (m, c) => c.task ? (cell(c.task, m.id) || {}).v : m[c.key];
  const rows = [...ms].sort((a, b) => {
    const c = cols.find(c => c.key === state.sort.key) || cols[2];
    const va = val(a, c), vb = val(b, c);
    if (va == null && vb == null) return 0;
    if (va == null) return 1; if (vb == null) return -1;
    return typeof va === 'string' ? state.sort.dir * va.localeCompare(vb)
                                  : state.sort.dir * (va - vb);
  });
  // best per column (max for accuracy/avg, min for perplexity)
  const best = {};
  for (const c of cols) {
    if (!c.num || c.key === 'params') continue;
    const vs = ms.map(m => val(m, c)).filter(v => v != null);
    if (vs.length > 1) best[c.key] = c.lower ? Math.min(...vs) : Math.max(...vs);
  }
  const shotOf = t => {
    const s = [...new Set(ms.map(m => (cell(t, m.id) || {}).shots).filter(x => x != null))];
    return s.length > 1 ? 'mixed!' : s.length ? s[0] + '-shot' : '';
  };
  const thead = el('thead', {},
    el('tr', {}, cols.map(c => el('th', {
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
      if (c.key === 'name') return el('td', { class: 'model', 'data-model': m.id, title: m.id },
        m.name, m.kind === 'instruct' ? el('span', { class: 'badge instruct', text: 'instruct' })
                                      : el('span', { class: 'badge', text: 'base' }));
      if (c.key === 'params') return el('td', { class: 'num',
        title: m.paramsSrc ? 'from ' + (m.paramsSrc === 'config' ? 'harness config' : 'model name') : '',
        text: P(m.params) });
      if (c.key === 'avg') return el('td', {
        class: 'num' + (m.avg != null && m.avg === best.avg ? ' best' : '') },
        pct(m.avg), m.navg < DATA.accTasks.length
          ? el('span', { class: 'se', text: ` ${m.navg}/${DATA.accTasks.length}` }) : '');
      const cc = cell(c.task, m.id);
      if (!cc) return el('td', { class: 'num', text: '—' });
      const isBest = best[c.key] != null && cc.v === best[c.key];
      return el('td', { class: 'num' + (isBest ? ' best' : '') },
        c.lower ? num(cc.v, 3) : pct(cc.v),
        cc.se && !c.lower ? el('span', { class: 'se', text: ` ±${(100 * cc.se).toFixed(1)}` }) : '');
    }))));
  return [el('div', { class: 'card' },
    el('h2', { text: 'Leaderboard' }),
    el('p', { class: 'sub', text: 'Click a column to sort. Accuracy cells are score ± stderr; '
      + 'perplexity columns are lower-is-better and excluded from Avg. '
      + 'Avg is the mean over the accuracy tasks a model actually ran (count shown when partial). '
      + '● marks the best value per column.' }),
    el('div', { class: 'lb-wrap' }, el('table', { class: 'lb' }, thead, tbody)))];
}

function vTasks(ms) {
  if (!DATA.accTasks.length) return [note('No accuracy tasks found.')];
  return [
    el('p', { class: 'sub', style: 'margin:10px 2px', text:
      'One panel per benchmark, models ranked. Bars share one hue on purpose — the label is the identity; '
      + 'pointing at any model highlights it in every panel. Dashed line = chance.' }),
    el('div', { class: 'panels' }, DATA.accTasks.map(t => barPanel(t, ms, { lower: false }))),
    tableTwin('tasks-table', ms, DATA.accTasks, false)];
}

function vPpl(ms) {
  if (!DATA.pplTasks.length) return [note('No perplexity tasks found. Create pinned corpus slices with scripts/make_ppl_task.py — they are the eval that separates models multiple-choice cannot.')];
  return [
    el('p', { class: 'sub', style: 'margin:10px 2px', text:
      'Rolling-loglikelihood perplexity over pinned corpus samples. Quote bits_per_byte when comparing '
      + 'across tokenizers — per-token perplexity is not comparable between model families. LOWER is better. '
      + 'No standard error is reported for these, so they are excluded from the significance tests.' }),
    el('div', { class: 'panels' }, DATA.pplTasks.map(t => barPanel(t, ms, { lower: true }))),
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

function vScaling(ms) {
  const withP = ms.filter(m => m.params != null);
  const noP = ms.filter(m => m.params == null);
  if (!state.scal) state.scal = new Set(SLOT_ORDER.slice(0, 4));
  const chips = el('div', { class: 'chips' }, DATA.accTasks.map(t => {
    const overflow = slotOf(t) >= SLOTS;
    const on = state.scal.has(t) && !overflow;
    return el('button', { class: 'tchip' + (on ? '' : ' off'),
      'aria-pressed': String(on), disabled: overflow ? '' : null,
      title: overflow ? 'more than 8 series cannot be told apart — see the Tasks panels' : '',
      onclick: () => { state.scal.has(t) ? state.scal.delete(t) : state.scal.add(t); render(); } },
      el('i', { style: `border-color:${slotColor(slotOf(t))}` }), t);
  }));
  const sel = DATA.accTasks.filter(t => state.scal.has(t) && slotOf(t) < SLOTS);
  const frag = [el('div', { class: 'card' },
    el('h2', { text: 'Score vs parameter count' }),
    el('p', { class: 'sub', text:
      'The scaling view: log-x parameter axis, one line per task. Lines connect base and instruct '
      + 'models alike in size order — use the Base filter above for a clean pretraining ladder '
      + '(the Pythia models are trained on identical data in identical order, so that curve is a '
      + 'genuine controlled experiment). Dashed lines mark chance.' }),
    chips)];
  if (withP.length < 2 || !sel.length) {
    frag[0].append(note(sel.length ? 'Need at least two models with known parameter counts.'
                                   : 'Select at least one task above.'));
    return frag;
  }
  const W = 780, H = 400, L = 52, R = 130, T = 16, B = 40;
  const pw = W - L - R, ph = H - T - B;
  const lo = Math.min(...withP.map(m => m.params)) * 0.8;
  const hi = Math.max(...withP.map(m => m.params)) * 1.25;
  const X = v => L + pw * (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  const Y = v => T + ph * (1 - v);
  const svg = el('svg:svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img',
                              'aria-label': 'score versus parameter count' });
  for (const yv of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(el('svg:line', { x1: L, y1: Y(yv), x2: L + pw, y2: Y(yv),
      stroke: 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('svg:text', { x: L - 8, y: Y(yv) + 4, 'font-size': 10.5,
      fill: 'var(--muted)', 'text-anchor': 'end', text: Math.round(yv * 100) + '%' }));
  }
  const xt = [];
  for (const base of [1e6, 1e7, 1e8, 1e9, 1e10]) for (const m of [1, 3])
    if (base * m >= lo && base * m <= hi) xt.push(base * m);
  for (const xv of xt) {
    svg.append(el('svg:line', { x1: X(xv), y1: T, x2: X(xv), y2: T + ph,
      stroke: 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('svg:text', { x: X(xv), y: H - B + 16, 'font-size': 10.5,
      fill: 'var(--muted)', 'text-anchor': 'middle', text: P(xv) }));
  }
  svg.append(el('svg:text', { x: L + pw / 2, y: H - 4, 'font-size': 11,
    fill: 'var(--text-secondary)', 'text-anchor': 'middle', text: 'parameters (log scale)' }));
  // chance lines for the selected tasks (deduped by level)
  const chances = [...new Set(sel.map(t => DATA.tasks[t].chance).filter(c => c != null && c > 0))];
  for (const c of chances) {
    svg.append(el('svg:line', { x1: L, y1: Y(c), x2: L + pw, y2: Y(c),
      stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '4 4' }));
    svg.append(el('svg:text', { x: L + pw + 6, y: Y(c) + 3.5, 'font-size': 10,
      fill: 'var(--muted)', text: `chance ${Math.round(c * 100)}%` }));
  }
  for (const t of sel) {
    const col = slotColor(slotOf(t));
    const pts = withP.map(m => ({ m, c: cell(t, m.id) })).filter(p => p.c)
                     .sort((a, b) => a.m.params - b.m.params);
    if (!pts.length) continue;
    svg.append(el('svg:path', { fill: 'none', stroke: col, 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      d: pts.map((p, j) => `${j ? 'L' : 'M'}${X(p.m.params).toFixed(1)},${Y(p.c.v).toFixed(1)}`).join(' ') }));
    for (const p of pts) {
      svg.append(el('svg:circle', { cx: X(p.m.params), cy: Y(p.c.v), r: 4.5, fill: col,
        stroke: 'var(--surface-1)', 'stroke-width': 2, 'data-model': p.m.id, class: 'bar' }));
      svg.append(el('svg:circle', { class: 'hit', cx: X(p.m.params), cy: Y(p.c.v), r: 12,
        tabindex: 0, 'data-model': p.m.id, 'data-tipkey': col,
        'data-tip': JSON.stringify([
          pct(p.c.v) + (p.c.se ? ` ± ${(100 * p.c.se).toFixed(1)} pts` : ''),
          t, `${p.m.id} · ${P(p.m.params)} params${p.m.kind === 'instruct' ? ' · instruct' : ''}`]) }));
    }
  }
  frag[0].append(svg);
  if (noP.length) frag[0].append(el('p', { class: 'small', text:
    'Not plotted (unknown parameter count): ' + noP.map(m => m.name).join(', ') }));
  return frag;
}

function vSig(ms) {
  if (!DATA.accTasks.length || ms.length < 2)
    return [note('Need at least two models and one accuracy task.')];
  if (!DATA.accTasks.includes(state.sigTask)) state.sigTask = DATA.accTasks[0];
  const chips = el('div', { class: 'chips' }, DATA.accTasks.map(t =>
    el('button', { class: 'tchip', 'aria-pressed': String(t === state.sigTask),
      onclick: () => { state.sigTask = t; render(); } }, t)));
  const t = state.sigTask;
  const present = ms.filter(m => cell(t, m.id)).sort((a, b) => cell(t, b.id).v - cell(t, a.id).v);
  const pair = {};
  for (const [a, b, diff, z, ok] of (DATA.sig[t] || [])) {
    pair[a + '|' + b] = { diff, z, ok }; pair[b + '|' + a] = { diff: -diff, z: z == null ? null : -z, ok };
  }
  const mx = el('table', { class: 'mx' });
  mx.append(el('tr', {}, el('th'), present.map(m =>
    el('th', { class: 'colh' }, el('span', { text: m.name })))));
  let nsig = 0, npair = 0;
  for (const a of present) {
    const tr = el('tr', {}, el('th', { class: 'rowh', title: a.id, text: a.name }));
    for (const b of present) {
      if (a.id === b.id) { tr.append(el('td', { class: 'self' })); continue; }
      const p = pair[a.id + '|' + b.id];
      if (!p) { tr.append(el('td', { text: '—' })); continue; }
      npair++;
      const glyph = p.ok ? (p.diff > 0 ? '▲' : '▼') : '·';
      if (p.ok) nsig++;
      tr.append(el('td', { class: p.ok ? (p.diff > 0 ? 'up' : 'down') : 'small',
        tabindex: 0,
        'data-tip': JSON.stringify([
          `${p.diff > 0 ? '+' : ''}${(100 * p.diff).toFixed(1)} pts` +
          (p.z != null ? ` (z = ${p.z > 0 ? '+' : ''}${p.z.toFixed(2)})` : ''),
          `${a.name}: ${pct(cell(t, a.id).v)}  vs  ${b.name}: ${pct(cell(t, b.id).v)}`,
          p.ok ? 'significant at 95% — the row model really is ' + (p.diff > 0 ? 'better' : 'worse')
               : 'inside the noise — cannot call a winner from this run']),
        text: glyph }));
    }
    mx.append(tr);
  }
  return [el('div', { class: 'card' },
    el('h2', { text: 'Is the difference real?' }),
    el('p', { class: 'sub', text: `Two-proportion z-test on ${t}: `
      + `${nsig / 2} of ${npair / 2} pairs are significant at 95%. `
      + '▲ row model significantly better than column · ▼ significantly worse · '
      + '· inside the noise. Rows and columns are sorted by score.' }),
    chips,
    el('div', { style: 'overflow-x:auto' }, mx),
    el('p', { class: 'sub', style: 'margin-top:10px', text:
      'z = (p₁−p₂) / √(se₁²+se₂²), significant at |z| > 1.96. Conservative: both models saw the same '
      + 'items, so a paired test would be more sensitive — this one will not call a non-difference '
      + 'significant, which is the direction that matters. Proportion metrics only; perplexity is '
      + 'not a proportion and never enters this table.' }))];
}

function vRuns(ms) {
  const frag = [];
  if (DATA.warnings.length)
    frag.push(el('div', { class: 'card' }, el('h2', { text: 'Warnings' }),
      DATA.warnings.map(w => el('div', { class: 'warn', text: w }))));
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'Run provenance' }),
    el('p', { class: 'sub', text: 'Every field here can change a score. Publish this table with the numbers, or the numbers are hearsay.' }),
    el('div', { class: 'lb-wrap' }, el('table', {},
      el('thead', {}, el('tr', {}, ['model', 'hf id', 'backend', 'dtype', 'batch',
        'chat template', 'seed', 'limit', 'params from', 'wall clock', 'harness', 'last run']
        .map(h => el('th', { text: h })))),
      el('tbody', {}, ms.map(m => el('tr', {},
        el('td', { 'data-model': m.id, text: m.name }),
        el('td', {}, el('span', { class: 'mono', text: m.id })),
        el('td', { text: m.backend || '—' }),
        el('td', { text: m.dtype || '—' }),
        el('td', { text: m.batch == null ? '—' : String(m.batch) }),
        el('td', { text: m.chat ? 'yes' : 'no' }),
        el('td', { class: 'num', text: m.seed == null ? '—' : String(m.seed) }),
        el('td', { text: m.limit == null ? 'full' : String(m.limit) }),
        el('td', { text: m.paramsSrc || '—' }),
        el('td', { class: 'num', text: m.minutes + ' min' }),
        el('td', {}, el('span', { class: 'mono', text: m.hash || '—' })),
        el('td', {}, el('span', { class: 'mono',
          text: String(m.date || '—').slice(0, 16).replace('T', ' ') })))))))));
  const names = new Set(ms.map(m => m.name));
  const rows = DATA.extra.filter(r => names.has(r[0]));
  const tv = el('div', { class: 'tv' }, el('table', {},
    el('thead', {}, el('tr', {}, ['model', 'task', 'metric', 'value', 'stderr', 'n-shot', 'items']
      .map((h, i) => el('th', { class: i >= 3 ? 'num' : '', text: h })))),
    el('tbody', {}, rows.map(r => el('tr', {},
      el('td', { text: r[0] }), el('td', { text: r[1] }), el('td', { text: r[2] }),
      el('td', { class: 'num', text: num(r[3], 4) }),
      el('td', { class: 'num', text: r[4] == null ? '—' : num(r[4], 4) }),
      el('td', { class: 'num', text: r[5] == null ? '—' : String(r[5]) }),
      el('td', { class: 'num', text: r[6] == null ? '—' : String(r[6]) }))))));
  const btn = el('button', { onclick: () => {
    tv.classList.toggle('open');
    btn.textContent = tv.classList.contains('open')
      ? 'Hide all metrics' : `Show every metric (${rows.length} rows, incl. MMLU subjects)`;
  }, text: `Show every metric (${rows.length} rows, incl. MMLU subjects)` });
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'Every metric, including sub-tasks' }),
    el('p', { class: 'sub', text: 'Both acc and acc_norm appear wherever the harness reported both — if they disagree, the benchmark is partly measuring option length; say which one you quote.' }),
    btn, tv));
  frag.push(el('div', { class: 'card' },
    el('h2', { text: 'Export' }),
    el('p', { class: 'sub', text: 'The CSV is the flat every-metric table; the JSON is everything this page renders.' }),
    el('div', { style: 'display:flex;gap:8px;margin-top:8px' },
      el('button', { onclick: exportCsv, text: 'Download CSV' }),
      el('button', { onclick: exportJson, text: 'Download JSON' }))));
  return frag;
}

function download(name, mime, text) {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type: mime })), download: name });
  document.body.append(a); a.click(); a.remove();
}
function exportCsv() {
  const esc = v => v == null ? '' : /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : String(v);
  const lines = ['model,task,metric,value,stderr,n_shot,n_samples'];
  for (const r of DATA.extra) lines.push(r.map(esc).join(','));
  download('benchmark.csv', 'text/csv', lines.join('\n'));
}
function exportJson() { download('benchmark.json', 'application/json', JSON.stringify(DATA, null, 1)); }

// ---------- shell ----------
const TABS = [
  ['overview', 'Overview', vOverview],
  ['leaderboard', 'Leaderboard', vLeaderboard],
  ['tasks', 'Tasks', vTasks],
  ['scaling', 'Scaling', vScaling],
  ['perplexity', 'Perplexity', vPpl],
  ['significance', 'Significance', vSig],
  ['runs', 'Runs', vRuns],
];
function render() {
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

// static bits
document.getElementById('warnings').replaceChildren(
  ...DATA.warnings.map(w => el('div', { class: 'warn' }, el('b', { text: 'Check: ' }), w)));
document.getElementById('metaChips').replaceChildren(
  el('span', { class: 'chip', text: `generated ${DATA.generated}` }),
  DATA.meta.hashes.length ? el('span', { class: 'chip' }, 'harness ',
    el('span', { class: 'mono', text: DATA.meta.hashes.join(', ') })) : '',
  DATA.meta.transformers ? el('span', { class: 'chip', text: `transformers ${DATA.meta.transformers}` }) : '',
  DATA.meta.anyLimit ? el('span', { class: 'chip', text: '⚠ smoke data (--limit)' }) : '');
document.getElementById('q').addEventListener('input', e => { state.q = e.target.value; render(); });
document.getElementById('kindSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  state.kind = b.getAttribute('data-kind');
  for (const x of e.currentTarget.querySelectorAll('button'))
    x.setAttribute('aria-pressed', String(x === b));
  render();
});
const THEMES = ['auto', 'light', 'dark'];
let themeIdx = 0;
document.getElementById('themeBtn').addEventListener('click', e => {
  themeIdx = (themeIdx + 1) % 3;
  const t = THEMES[themeIdx];
  if (t === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', t);
  e.target.textContent = 'Theme: ' + t;
});
render();
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>__CSS__</style></head>
<body class="viz-root"><div id="tip" role="status"></div><div class="wrap">
  <div class="topbar">
    <div>
      <h1>__TITLE__</h1>
      <p class="sub">lm-evaluation-harness results, one self-contained file — data
      embedded, charts drawn locally, nothing fetched.</p>
      <div class="meta-chips" id="metaChips"></div>
    </div>
    <button id="themeBtn" title="cycle auto / light / dark">Theme: auto</button>
  </div>
  <div id="warnings"></div>
  <div class="filters">
    <input type="search" id="q" placeholder="Filter models&hellip;" aria-label="filter models">
    <div class="seg" role="group" aria-label="model kind" id="kindSeg">
      <button data-kind="all" aria-pressed="true">All</button>
      <button data-kind="base" aria-pressed="false">Base</button>
      <button data-kind="instruct" aria-pressed="false">Instruct</button>
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
        lines = ["model,task,metric,value,stderr,n_shot,n_samples,chat_template,git_hash"]
        for r in runs:
            for task, entry in r["tasks"].items():
                for name, d in entry.items():
                    if not isinstance(d, dict) or "value" not in d:
                        continue
                    lines.append(",".join(str(x) for x in [
                        r["model"], task, name, d["value"], d.get("stderr", ""),
                        r["n_shot"].get(task, ""), r["n_samples"].get(task, ""),
                        r["chat_template"], r["git_hash"] or ""]))
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {args.csv}  ({len(lines) - 1} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
