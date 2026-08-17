#!/usr/bin/env python3
"""
Turn lm-evaluation-harness output into one self-contained HTML report.

    python scripts/report_lm_eval.py results/ -o artifacts/benchmark_report.html

`results/` is whatever you passed to `lm_eval --output_path`. The script walks it,
finds every results JSON, and builds a comparison across models.

What it produces that the harness's own stdout does not:

  * every score with its standard error, side by side across models
  * a PAIRWISE SIGNIFICANCE TABLE — is model A actually better than model B, or is
    the gap inside the noise? This is the question people will ask you, and the
    answer is arithmetic, not opinion.
  * the full provenance for every run: n-shot, n samples, batch size, dtype, chat
    template applied or not, harness git hash, eval wall-clock. Two rows that
    differ in any of these are not comparable, and the report says so.
  * a table view under every chart, because a chart persuades and a table lets
    someone check.

Deliberately dependency-free (stdlib only) so it runs anywhere your harness runs.
"""

from __future__ import annotations

import argparse
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
            entry.setdefault(name, {})[slot] = float(val)
        tasks[task] = entry

    n_samples = blob.get("n-samples") or {}
    # The harness tells us which tasks are children of a group (MMLU's 57 subjects,
    # its four category roll-ups, etc). Use it rather than pattern-matching names:
    # the headline chart shows groups and standalone tasks, and the children go in
    # the full table where they belong.
    subtasks: set[str] = set()
    for parent, kids in (blob.get("group_subtasks") or {}).items():
        for k in kids or []:
            if k != parent:
                subtasks.add(k)
    # The harness tells us the direction of every metric. Use it rather than guessing
    # from the name: perplexity and bits-per-byte are LOWER-is-better and are not
    # proportions, so they must not go in the same chart as accuracies, and the
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
        "n_shot": blob.get("n-shot") or {},
        "n_samples": {k: (v.get("effective") if isinstance(v, dict) else v)
                      for k, v in n_samples.items()},
        "git_hash": blob.get("git_hash"),
        "date": blob.get("date"),
        "transformers_version": blob.get("transformers_version"),
        "eval_seconds": _to_float(blob.get("total_evaluation_time_seconds")),
        "tasks": tasks,
    }


def _extract(args: str, key: str):
    m = re.search(rf"{key}=([^,\s]+)", str(args))
    return m.group(1) if m else None


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def primary_metric(entry: dict) -> tuple[str, float, float] | None:
    """Pick one headline metric per task, preferring the conventional one.

    acc_norm before acc for HellaSwag-style tasks, exact_match for GSM8K, pass@1 for
    code. The choice is recorded in the output so nobody has to guess which number
    they are looking at — and both are always in the full table.
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
# rendering
# ---------------------------------------------------------------------------

SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"]

CSS = """
:root { color-scheme: light; }
.viz-root {
  --surface-1:#fcfcfb; --plane:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --good:#0ca30c; --critical:#d03b3b; --warning:#fab219; --success-text:#006300;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --success-text:#0ca30c;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --success-text:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--plane); color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px; line-height:1.5; }
.wrap { max-width:1120px; margin:0 auto; padding:30px 20px 70px; }
h1 { font-size:22px; font-weight:600; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:15px; font-weight:600; margin:0 0 3px; }
.sub { color:var(--text-secondary); font-size:13px; margin:0; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:20px; margin:16px 0; }
button { font:inherit; color:var(--text-primary); background:var(--surface-1);
  border:1px solid var(--border); border-radius:8px; padding:6px 10px; cursor:pointer; }
button:hover { background:var(--plane); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.tile { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:14px 16px; }
.tile .label { color:var(--text-secondary); font-size:12px; }
.tile .value { font-size:25px; font-weight:600; letter-spacing:-0.02em; }
.hero { font-size:50px; font-weight:600; letter-spacing:-0.03em; line-height:1.05; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--muted); font-weight:600; padding:6px 10px; border-bottom:1px solid var(--axis); }
td { padding:7px 10px; border-bottom:1px solid var(--grid); font-size:13px; }
td.num, th.num { text-align:right; }
tr:last-child td { border-bottom:none; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
  color:var(--text-secondary); }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin:4px 0 12px; }
.legend span { display:inline-flex; align-items:center; gap:6px; font-size:12px;
  color:var(--text-secondary); }
.key { width:11px; height:11px; border-radius:3px; display:inline-block; }
.tv { display:none; margin-top:12px; } .tv.open { display:block; }
.small { font-size:12px; color:var(--text-secondary); }
.up { color:var(--success-text); } .down { color:var(--critical); }
.note { border-left:2px solid var(--axis); padding:7px 0 7px 12px; margin:14px 0;
  color:var(--text-secondary); font-size:13px; }
.warn { border-left:2px solid var(--warning); padding:7px 0 7px 12px; margin:14px 0;
  color:var(--text-secondary); font-size:13px; }
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:6px 9px; font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,.14); z-index:50;
  max-width:300px; }
svg text { font-family:system-ui,-apple-system,sans-serif; }
.hit { fill:transparent; }
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{
  const t=e.target.closest('[data-tip]');
  if(!t){tip.style.opacity=0;return;}
  tip.innerHTML=t.getAttribute('data-tip'); tip.style.opacity=1;
  const p=14,w=tip.offsetWidth,h=tip.offsetHeight;
  let x=e.clientX+p,y=e.clientY+p;
  if(x+w>innerWidth)x=e.clientX-w-p;
  if(y+h>innerHeight)y=e.clientY-h-p;
  tip.style.left=x+'px'; tip.style.top=y+'px';
});
function toggleTable(id,btn){const el=document.getElementById(id);el.classList.toggle('open');
  btn.textContent=el.classList.contains('open')?'Hide data table':'Show data table';}
function setTheme(){const c=document.documentElement.getAttribute('data-theme');
  document.documentElement.setAttribute('data-theme',c==='dark'?'light':'dark');}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(v, digits=1) -> str:
    return f"{100 * v:.{digits}f}%" if v is not None else "—"


def fmt(v, digits=3) -> str:
    """Plain number, for metrics that are not proportions (perplexity, bits/byte)."""
    if v is None:
        return "—"
    return f"{v:,.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:,.0f}"


def short(model: str) -> str:
    return model.split("/")[-1]


def nice_ticks(hi: float, n: int = 5) -> list[float]:
    raw = hi / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag)
    out, v = [], 0.0
    while v <= hi + step * 0.5:
        out.append(round(v, 10))
        v += step
    return out


def svg_grouped_bars(tasks: list[str], models: list[str], data: dict,
                     width: int = 660, row_h: int = 22, gap: int = 5,
                     group_gap: int = 18, label_w: int = 150,
                     as_pct: bool = True) -> str:
    """
    Grouped horizontal bars: one group per task, one bar per model.

    Error bars are drawn from the standard error, because a bar chart of accuracies
    without them invites exactly the comparison the numbers cannot support.
    """
    rows = [t for t in tasks if any((t, m) in data for m in models)]
    if not rows:
        return "<p class='small'>no data</p>"
    hi = max(v for (t, m), (v, _s) in data.items() if t in rows)
    hi = min(1.0, max(0.25, hi * 1.18)) if as_pct else hi * 1.18
    show = (lambda v: pct(v)) if as_pct else (lambda v: fmt(v, 3))
    tick = (lambda t: f"{100 * t:.0f}%") if as_pct else (lambda t: fmt(t, 2))
    plot_w = width - label_w - 60
    height = sum(len([m for m in models if (t, m) in data]) * (row_h + gap) + group_gap
                 for t in rows) + 26

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" aria-label="benchmark scores by model">']
    for t in nice_ticks(hi, 4):
        x = label_w + plot_w * t / hi
        out.append(f'<line x1="{x:.1f}" y1="6" x2="{x:.1f}" y2="{height - 20}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 6}" font-size="10" fill="var(--muted)" '
                   f'text-anchor="middle">{tick(t)}</text>')

    y = 6
    for t in rows:
        present = [m for m in models if (t, m) in data]
        mid = y + (len(present) * (row_h + gap)) / 2 - gap / 2
        out.append(f'<text x="{label_w - 10}" y="{mid + 4:.1f}" font-size="12" '
                   f'font-weight="600" fill="var(--text-primary)" text-anchor="end">'
                   f'{esc(t)}</text>')
        for m in present:
            v, s = data[(t, m)]
            col = f"var({SERIES[models.index(m) % 8]})"
            w = max(2.0, plot_w * v / hi)
            r = min(4, w)
            tip = (f"<b>{esc(short(m))}</b><br>{esc(t)}<br>{show(v)}"
                   + (f" &plusmn; {show(s)}" if s else ""))
            out.append(
                f'<path d="M{label_w},{y} H{label_w + w - r:.1f} q{r},0 {r},{r} '
                f'V{y + row_h - r} q0,{r} -{r},{r} H{label_w} Z" fill="{col}" '
                f'data-tip="{tip}"/>')
            # error bar: +/- 1 stderr, drawn on the surface so it reads as a whisker
            if s > 0:
                lo = label_w + plot_w * max(0, v - s) / hi
                hi_x = label_w + plot_w * min(hi, v + s) / hi
                cy = y + row_h / 2
                out.append(f'<line x1="{lo:.1f}" y1="{cy}" x2="{hi_x:.1f}" y2="{cy}" '
                           f'stroke="var(--text-primary)" stroke-width="1.5" opacity="0.55"/>')
                for xx in (lo, hi_x):
                    out.append(f'<line x1="{xx:.1f}" y1="{cy - 4}" x2="{xx:.1f}" '
                               f'y2="{cy + 4}" stroke="var(--text-primary)" '
                               f'stroke-width="1.5" opacity="0.55"/>')
            out.append(f'<rect class="hit" x="{label_w}" y="{y - 2}" width="{plot_w}" '
                       f'height="{row_h + 4}" data-tip="{tip}"/>')
            out.append(f'<text x="{label_w + max(w, plot_w * (v + s) / hi) + 8:.1f}" '
                       f'y="{y + row_h * 0.72:.1f}" font-size="11.5" '
                       f'fill="var(--text-primary)">{show(v)}</text>')
            y += row_h + gap
        y += group_gap
    out.append(f'<line x1="{label_w}" y1="6" x2="{label_w}" y2="{height - 20}" '
               f'stroke="var(--axis)" stroke-width="1"/>')
    out.append("</svg>")
    return "".join(out)


def table(headers, rows, num_cols=None) -> str:
    num_cols = num_cols or set()
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{esc(h)}</th>'
                 for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>'
                         for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def table_view(tid, headers, rows, num_cols=None) -> str:
    return (f'<button onclick="toggleTable(\'{tid}\',this)">Show data table</button>'
            f'<div class="tv" id="{tid}">{table(headers, rows, num_cols)}</div>')


def build_report(runs: list[dict], out_path: Path, title: str) -> Path:
    if not runs:
        out_path.write_text("<h1>No lm-eval results found.</h1>", encoding="utf-8")
        return out_path

    # one row per model; if a model was run more than once, the later file wins
    by_model: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: str(r.get("date") or "")):
        by_model[r["model"]] = r
    models = list(by_model)

    # collect the headline metric per (task, model)
    data: dict[tuple[str, str], tuple[float, float]] = {}
    metric_used: dict[str, str] = {}
    all_tasks: list[str] = []
    for model, run in by_model.items():
        for task, entry in run["tasks"].items():
            # skip MMLU's 57 leaf subjects and other sub-tasks; keep groups + standalones
            pm = primary_metric(entry)
            if pm is None:
                continue
            name, v, s = pm
            data[(task, model)] = (v, s)
            metric_used.setdefault(task, name)
            if task not in all_tasks:
                all_tasks.append(task)

    # headline tasks = the ones every model has, sorted by name; subtasks go to the table
    child_tasks: set[str] = set()
    for r in by_model.values():
        child_tasks |= r.get("subtasks", set())
    common = [t for t in all_tasks
              if t not in child_tasks
              and sum((t, m) in data for m in models) == len(models)]
    headline = sorted(common, key=lambda t: (len(t), t))[:12] or sorted(all_tasks)[:12]

    # Split by metric direction. Accuracies are 0-1 proportions where higher is
    # better; perplexity and bits-per-byte are unbounded and lower is better. Putting
    # them on one axis would be nonsense, and the z-test only applies to the first kind.
    PROPORTION = {"acc", "acc_norm", "exact_match", "pass@1", "f1", "em", "rubric_pass"}
    acc_tasks, ppl_tasks = [], []
    for t in headline:
        met = metric_used.get(t, "")
        lower_better = any(r["higher_is_better"].get(t, {}).get(met) is False
                           for r in by_model.values())
        (acc_tasks if (met in PROPORTION and not lower_better) else ppl_tasks).append(t)

    parts: list[str] = []

    # ---- provenance warnings, first, because they invalidate everything below ----
    warns = []
    shots = {}
    for m, r in by_model.items():
        for t in headline:
            if t in r["n_shot"]:
                shots.setdefault(t, set()).add(r["n_shot"][t])
    mismatched = [t for t, s in shots.items() if len(s) > 1]
    if mismatched:
        warns.append(f"<b>Few-shot count differs between models</b> on: "
                     f"{', '.join(esc(t) for t in mismatched)}. Those columns are not "
                     f"comparable — re-run with the same <span class='mono'>--num_fewshot</span>.")
    templates = {r["chat_template"] for r in by_model.values()}
    if len(templates) > 1:
        applied = [short(m) for m, r in by_model.items() if r["chat_template"]]
        warns.append(f"<b>Chat template applied to some models but not others</b> "
                     f"(applied to: {', '.join(esc(a) for a in applied)}). This alone moves "
                     f"scores by tens of points on instruct models. Fix before comparing.")
    limits = {r["limit"] for r in by_model.values()}
    if any(l for l in limits):
        warns.append("<b>At least one run used <span class='mono'>--limit</span></b>, so it "
                     "did not see the full task. Fine for a smoke test, not for a reported number.")
    hashes = {r["git_hash"] for r in by_model.values() if r["git_hash"]}
    if len(hashes) > 1:
        warns.append(f"<b>Results came from {len(hashes)} different harness versions.</b> "
                     f"A benchmark whose code changed is a different benchmark.")
    for w in warns:
        parts.append(f'<div class="warn">{w}</div>')

    # ---- hero + tiles ----
    best_task = (acc_tasks or ppl_tasks or [None])[0]
    if best_task:
        ranked = sorted(((data[(best_task, m)][0], m) for m in models if (best_task, m) in data),
                        reverse=True)
        top_v, top_m = ranked[0]
        parts.append(
            '<div class="card">'
            f'<p class="sub">Best on <b>{esc(best_task)}</b> '
            f'({esc(metric_used.get(best_task, ""))})</p>'
            f'<div class="hero">{pct(top_v)}</div>'
            f'<p class="sub">{esc(top_m)}</p></div>')

    total_time = sum(r["eval_seconds"] or 0 for r in by_model.values())
    parts.append(
        '<div class="tiles">'
        f'<div class="tile"><div class="label">Models compared</div>'
        f'<div class="value">{len(models)}</div></div>'
        f'<div class="tile"><div class="label">Tasks</div>'
        f'<div class="value">{len(all_tasks)}</div></div>'
        f'<div class="tile"><div class="label">Total eval time</div>'
        f'<div class="value">{total_time / 3600:.1f}<span style="font-size:14px"> h</span>'
        f'</div></div>'
        f'<div class="tile"><div class="label">Harness build</div>'
        f'<div class="value" style="font-size:16px">'
        f'<span class="mono">{esc((list(hashes) or ["—"])[0])}</span></div></div>'
        '</div>')

    # ---- the chart ----
    legend = "".join(
        f'<span><i class="key" style="background:var({SERIES[i % 8]})"></i>'
        f'{esc(short(m))}</span>' for i, m in enumerate(models[:8]))
    def tv_for(tasks, as_pct):
        f = (lambda v: pct(v, 2)) if as_pct else (lambda v: fmt(v, 4))
        return [[esc(t), esc(metric_used.get(t, "")), esc(short(m)),
                 f(data[(t, m)][0]), (f(data[(t, m)][1]) if data[(t, m)][1] else "—"),
                 by_model[m]["n_shot"].get(t, "—"),
                 by_model[m]["n_samples"].get(t, "—")]
                for t in tasks for m in models if (t, m) in data]

    COLS = ["task", "metric", "model", "score", "stderr", "n-shot", "n samples"]
    if acc_tasks:
        parts.append(
            '<div class="card"><h2>Accuracy by task &mdash; higher is better</h2>'
            '<p class="sub">Whiskers are &plusmn;1 standard error. Two bars whose whiskers '
            'overlap are not distinguishable at this sample size &mdash; check the '
            'significance table below before claiming one model is better. Chance level '
            'depends on the number of options: 25% on a 4-way task, <b>50% on a 2-way '
            'task like Winogrande or PIQA</b>.</p>'
            f'<div class="legend">{legend}</div>'
            + svg_grouped_bars(acc_tasks, models, data)
            + table_view("tv-scores", COLS, tv_for(acc_tasks, True), {3, 4, 5, 6})
            + '</div>')

    if ppl_tasks:
        parts.append(
            '<div class="card"><h2>Perplexity / bits-per-byte &mdash; LOWER is better</h2>'
            '<p class="sub">Separate chart on purpose: these are unbounded and '
            'lower-is-better, so they share no axis with the accuracies above. '
            '<b>Quote bits_per_byte</b> when comparing models with different tokenizers '
            '&mdash; per-token perplexity is on a different scale for each tokenizer and '
            'is not comparable across model families. The harness reports no standard '
            'error for these, so the significance test below excludes them.</p>'
            f'<div class="legend">{legend}</div>'
            + svg_grouped_bars(ppl_tasks, models, data, as_pct=False)
            + table_view("tv-ppl", COLS, tv_for(ppl_tasks, False), {3, 4, 5, 6})
            + '</div>')

    # ---- pairwise significance ----
    if len(models) > 1 and acc_tasks:
        sig_rows = []
        for t in acc_tasks:
            present = [m for m in models if (t, m) in data]
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    va, sa = data[(t, a)]
                    vb, sb = data[(t, b)]
                    ok, z = significant(va, sa, vb, sb)
                    diff = va - vb
                    winner = a if diff > 0 else b
                    if ok:
                        verdict = (f'<span class="up">&#9679; {esc(short(winner))} wins</span>')
                    else:
                        verdict = ('<span class="small">&#9675; inside the noise — '
                                   'not distinguishable</span>')
                    sig_rows.append([esc(t), esc(short(a)), esc(short(b)),
                                     f"{100 * diff:+.1f} pts",
                                     f"{z:+.2f}" if math.isfinite(z) else "—", verdict])
        n_noise = sum(1 for r in sig_rows if "noise" in r[5])
        noisy = [f"{r[1]} vs {r[2]} on {r[0]}" for r in sig_rows if "noise" in r[5]]
        head = (f"<b>{n_noise} of {len(sig_rows)}</b> pairwise comparisons are inside the "
                f"noise at 95%" + (f" &mdash; {'; '.join(noisy[:4])}"
                                   + ("&hellip;" if len(noisy) > 4 else "") if noisy else ""))
        parts.append(
            '<div class="card"><h2>Is the difference real?</h2>'
            f'<p class="sub">{head}. These are the pairs where you cannot claim a winner '
            'from this run, no matter what the bar chart looks like.</p>'
            '<p class="sub" style="margin-top:8px">Two-proportion z-test: '
            '<span class="mono">z = (p&#8321;&minus;p&#8322;) / &radic;(se&#8321;&sup2;+se&#8322;&sup2;)</span>, '
            'significant at |z| &gt; 1.96. Conservative &mdash; both models saw the same '
            'items, so a paired test would be more sensitive. It will not call a '
            'non-difference significant, which is the direction that matters. '
            'Proportion metrics only &mdash; perplexity is not a proportion.</p>'
            + table_view("tv-sig", ["task", "model A", "model B", "difference", "z", "verdict"],
                         sig_rows, {3, 4})
            + '</div>')

    # ---- full metric table, every metric, every task ----
    full_rows = []
    for m in models:
        run = by_model[m]
        for task in sorted(run["tasks"]):
            entry = run["tasks"][task]
            for name, d in sorted(entry.items()):
                if not isinstance(d, dict) or "value" not in d:
                    continue
                full_rows.append([
                    esc(short(m)), esc(task), esc(name),
                    f'{d["value"]:.4f}',
                    f'{d.get("stderr", 0.0):.4f}' if d.get("stderr") else "—",
                    run["n_shot"].get(task, "—"), run["n_samples"].get(task, "—")])
    parts.append(
        '<div class="card"><h2>Every metric, including sub-tasks</h2>'
        '<p class="sub">MMLU\'s 57 subjects and any other sub-tasks live here. Both '
        '<span class="mono">acc</span> and <span class="mono">acc_norm</span> are shown '
        'wherever the harness reported both — if they disagree, the benchmark is partly '
        'measuring option length, and you should say which one you are quoting.</p>'
        + table_view("tv-full", ["model", "task", "metric", "value", "stderr",
                                 "n-shot", "n samples"], full_rows, {3, 4, 5, 6})
        + '</div>')

    # ---- provenance ----
    prov = [[esc(short(m)), f'<span class="mono">{esc(m)}</span>',
             esc(r["backend"] or "—"), esc(r["dtype"] or "—"),
             esc(r["batch_size"] or "—"),
             "yes" if r["chat_template"] else "no",
             esc(r["seed"] if r["seed"] is not None else "—"),
             esc(r["limit"] or "full"),
             f'{(r["eval_seconds"] or 0) / 60:.1f} min',
             f'<span class="mono">{esc(r["git_hash"] or "—")}</span>']
            for m, r in by_model.items()]
    parts.append(
        '<div class="card"><h2>Run provenance</h2>'
        '<p class="sub">Every field here can change a score. Publish this table with the '
        'numbers, or the numbers are hearsay.</p>'
        + table(["model", "hf id", "backend", "dtype", "batch", "chat template",
                 "seed", "limit", "wall clock", "harness"], prov, {6, 8})
        + '</div>')

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body class="viz-root"><div id="tip"></div><div class="wrap">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">
    <div><h1>{esc(title)}</h1>
    <p class="sub">Generated from lm-evaluation-harness output. Self-contained: no
    network, no build step. {len(models)} model{"s" if len(models) != 1 else ""},
    {len(all_tasks)} task{"s" if len(all_tasks) != 1 else ""}.</p></div>
    <button onclick="setTheme()">Toggle theme</button>
  </div>
  {"".join(parts)}
  <p class="small" style="margin-top:30px">Every score carries its standard error;
  every chart has a table view; differences are tested before they are called wins.
  Scores are only comparable to published numbers when n-shot, prompt template and
  metric all match — check the provenance table before quoting anything.</p>
</div><script>{JS}</script></body></html>"""
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
    for r in runs:
        pm = {t: primary_metric(e) for t, e in r["tasks"].items()}
        summary = "  ".join(f"{t}={100 * v[1]:.1f}%" for t, v in sorted(pm.items())
                            if v and len(t) < 20)
        print(f"  {r['model']:<45} {summary[:90]}")

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
