"""
The dashboard: registry + metrics -> one self-contained HTML file.

This is the "automation and dashboard" half of your job, and the deliverable that
makes the rest of the pipeline legible to people who will never read the code. It
takes runs/registry.jsonl plus each run's metrics.jsonl and emits a single HTML
file with no external assets, no build step and no server — which means it can be
attached to a PR, emailed, or dropped in CI artifacts and it still works in five
years.

Design rules followed here (they are not decoration — each one prevents a specific
way dashboards mislead people):

  * Runs from different eval suites are NEVER shown in the same table. Different
    suite hash = different exam. The filter row switches between them.
  * Every chart has a table view. Charts persuade; tables let someone check.
  * Every accuracy carries n and stderr, so nobody reads a 1-point move as a win.
  * Deltas are labelled with an arrow AND a word, never colour alone.
  * One number is a headline, but its components are on the same screen.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

from .registry import DEFAULT_PATH, load_all
from .tracking import read_metrics

# Validated categorical palette (light / dark steps of the same eight hues).
# Verified with the dataviz validator: all checks pass on the adjacent pairlist in
# both modes. Slot order is the colourblind-safety mechanism — do not reorder, and
# never generate a 9th hue; fold the tail into "other" instead.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

CSS = """
:root { color-scheme: light; }
.viz-root {
  --surface-1:#fcfcfb; --plane:#f9f9f7; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,0.10); --good:#0ca30c; --critical:#d03b3b; --success-text:#006300;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#ffffff;
    --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
    --border:rgba(255,255,255,0.10); --success-text:#0ca30c;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --text-primary:#ffffff;
  --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,0.10); --success-text:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--plane); color:var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size:14px;
  line-height:1.5; }
.wrap { max-width:1120px; margin:0 auto; padding:32px 20px 72px; }
h1 { font-size:22px; font-weight:600; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:15px; font-weight:600; margin:0 0 2px; }
.sub { color:var(--text-secondary); font-size:13px; margin:0; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:20px; margin:16px 0; }
.filters { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:20px 0 4px; }
.filters label { color:var(--text-secondary); font-size:12px; text-transform:uppercase;
  letter-spacing:0.06em; }
button, select { font:inherit; color:var(--text-primary); background:var(--surface-1);
  border:1px solid var(--border); border-radius:8px; padding:6px 10px; cursor:pointer; }
button:hover, select:hover { background:var(--plane); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.tile { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:14px 16px; }
.tile .label { color:var(--text-secondary); font-size:12px; }
.tile .value { font-size:26px; font-weight:600; letter-spacing:-0.02em; }
.hero { font-size:52px; font-weight:600; letter-spacing:-0.03em; line-height:1.05; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--muted); font-weight:600; padding:6px 10px; border-bottom:1px solid var(--axis); }
td { padding:7px 10px; border-bottom:1px solid var(--grid); font-size:13px; }
td.num, th.num { text-align:right; }
tr:last-child td { border-bottom:none; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px;
  color:var(--text-secondary); }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin:2px 0 10px; }
.legend span { display:inline-flex; align-items:center; gap:6px; font-size:12px;
  color:var(--text-secondary); }
.key { width:14px; height:2px; border-radius:1px; display:inline-block; }
.keydot { width:9px; height:9px; border-radius:50%; display:inline-block; }
.tv { display:none; margin-top:12px; }
.tv.open { display:block; }
.small { font-size:12px; color:var(--text-secondary); }
.up { color:var(--success-text); } .down { color:var(--critical); }
.note { border-left:2px solid var(--axis); padding:6px 0 6px 12px; margin:14px 0;
  color:var(--text-secondary); font-size:13px; }
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:6px 9px; font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,.14); z-index:50;
  max-width:280px; }
svg text { font-family: system-ui, -apple-system, sans-serif; }
.hit { fill:transparent; }
"""

JS = """
const tip = document.getElementById('tip');
document.addEventListener('mousemove', e => {
  const t = e.target.closest('[data-tip]');
  if (!t) { tip.style.opacity = 0; return; }
  tip.innerHTML = t.getAttribute('data-tip');
  tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth) x = e.clientX - w - pad;
  if (y + h > innerHeight) y = e.clientY - h - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
});
function toggleTable(id, btn) {
  const el = document.getElementById(id);
  el.classList.toggle('open');
  btn.textContent = el.classList.contains('open') ? 'Hide data table' : 'Show data table';
}
function setTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
}
function selectSuite(sel) {
  document.querySelectorAll('[data-suite-view]').forEach(v => {
    v.style.display = (v.getAttribute('data-suite-view') === sel.value) ? '' : 'none';
  });
}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def fmt(n, digits: int = 2) -> str:
    if n is None:
        return "-"
    if isinstance(n, float):
        if abs(n) >= 1e6:
            return f"{n:,.0f}"
        return f"{n:,.{digits}f}".rstrip("0").rstrip(".") if digits else f"{n:,.0f}"
    return f"{n:,}"


def human(n) -> str:
    if n is None:
        return "-"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round axis ticks to clean numbers. Ugly axes are a tell that nobody checked."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------

def svg_hbars(rows: list[tuple[str, float, str]], vmax: float | None = None,
              width: int = 640, bar: int = 20, gap: int = 14,
              label_w: int = 220, unit: str = "", ref: tuple[float, str] | None = None) -> str:
    """
    Horizontal bars, one series (so no legend — the heading names it).

    Specs from the design method: <=24px thick bars, 4px rounded data-end and square
    at the baseline, value labelled at the tip, hairline baseline, recessive grid.
    """
    if not rows:
        return "<p class='small'>no data</p>"
    vmax = vmax or max(v for _, v, _ in rows) or 1
    plot_w = width - label_w - 70
    height = len(rows) * (bar + gap) + 28
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" aria-label="bar chart">']
    for t in nice_ticks(0, vmax, 4):
        x = label_w + plot_w * t / vmax
        out.append(f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{height - 20}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 6}" font-size="10" fill="var(--muted)" '
                   f'text-anchor="middle">{fmt(t, 0) if t == int(t) else fmt(t)}</text>')
    for i, (label, value, tipbody) in enumerate(rows):
        y = 8 + i * (bar + gap)
        w = max(2.0, plot_w * value / vmax)
        out.append(
            f'<text x="{label_w - 10}" y="{y + bar * 0.72:.1f}" font-size="12" '
            f'fill="var(--text-secondary)" text-anchor="end">{esc(label)}</text>')
        # 4px rounded data-end, square at the baseline: a path, not a rounded rect.
        r = min(4, w)
        out.append(
            f'<path d="M{label_w},{y} H{label_w + w - r:.1f} q{r},0 {r},{r} '
            f'V{y + bar - r} q0,{r} -{r},{r} H{label_w} Z" fill="var(--s1)" '
            f'data-tip="{esc(tipbody)}"/>')
        out.append(f'<rect class="hit" x="{label_w}" y="{y - 4}" width="{plot_w}" '
                   f'height="{bar + 8}" data-tip="{esc(tipbody)}"/>')
        out.append(f'<text x="{label_w + w + 8:.1f}" y="{y + bar * 0.72:.1f}" font-size="12" '
                   f'fill="var(--text-primary)" font-weight="600">{fmt(value)}{unit}</text>')
    if ref is not None:
        rv, rlabel = ref
        rx = label_w + plot_w * rv / vmax
        out.append(f'<line x1="{rx:.1f}" y1="4" x2="{rx:.1f}" y2="{height - 22}" '
                   f'stroke="var(--text-secondary)" stroke-width="1"/>')
        out.append(f'<text x="{rx + 4:.1f}" y="{height - 24}" font-size="10" '
                   f'fill="var(--text-secondary)">{esc(rlabel)}</text>')
    out.append(f'<line x1="{label_w}" y1="8" x2="{label_w}" y2="{height - 20}" '
               f'stroke="var(--axis)" stroke-width="1"/>')
    out.append("</svg>")
    return "".join(out)


def svg_lines(series: list[dict], width: int = 660, height: int = 260,
              x_label: str = "step", y_label: str = "loss") -> str:
    """Multi-series line chart. 2px lines, >=8px end markers with a 2px surface ring,
    selective end labels, hairline grid, per-point hit targets for the tooltip."""
    series = [s for s in series if len(s["points"]) >= 2][:8]
    if not series:
        return "<p class='small'>no curves logged</p>"
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1
    pad = (y1 - y0) * 0.1
    y0, y1 = max(0, y0 - pad), y1 + pad
    L, R, T, B = 52, 108, 24, 30
    pw, ph = width - L - R, height - T - B

    def px(x): return L + pw * (x - x0) / max(1e-9, x1 - x0)
    def py(y): return T + ph * (1 - (y - y0) / max(1e-9, y1 - y0))

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" aria-label="line chart">']
    for t in nice_ticks(y0, y1, 4):
        if t < y0 - 1e-9:
            continue
        out.append(f'<line x1="{L}" y1="{py(t):.1f}" x2="{L + pw}" y2="{py(t):.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{py(t) + 3.5:.1f}" font-size="10" '
                   f'fill="var(--muted)" text-anchor="end">{fmt(t)}</text>')
    for t in nice_ticks(x0, x1, 4):
        if t < x0 - 1e-9 or t > x1 + 1e-9:
            continue
        out.append(f'<text x="{px(t):.1f}" y="{height - 10}" font-size="10" '
                   f'fill="var(--muted)" text-anchor="middle">{fmt(t, 0)}</text>')
    out.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" '
               f'stroke="var(--axis)" stroke-width="1"/>')
    out.append(f'<text x="{L - 8}" y="10" font-size="10" fill="var(--muted)" '
               f'text-anchor="end">{esc(y_label)}</text>')
    out.append(f'<text x="{L + pw}" y="{height - 10}" font-size="10" fill="var(--muted)" '
               f'text-anchor="end">{esc(x_label)}</text>')

    # End labels are placed only where they do not collide. When lines converge,
    # nudging labels apart detaches them from their line and reads as noise, so the
    # rule is: label the ones that fit, let the legend and tooltip carry the rest.
    placed: list[float] = []
    for i, s in enumerate(series):
        col = f"var(--s{i % 8 + 1})"
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in s["points"])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in s["points"]:
            out.append(f'<circle class="hit" cx="{px(x):.1f}" cy="{py(y):.1f}" r="11" '
                       f'data-tip="<b>{esc(s["name"])}</b><br>{esc(x_label)} {fmt(x, 0)}'
                       f'<br>{esc(y_label)} {fmt(y, 4)}"/>')
        lx, ly = s["points"][-1]
        out.append(f'<circle cx="{px(lx):.1f}" cy="{py(ly):.1f}" r="4.5" fill="{col}" '
                   f'stroke="var(--surface-1)" stroke-width="2"/>')
        yy = py(ly)
        if all(abs(yy - q) >= 13 for q in placed):
            placed.append(yy)
            out.append(f'<text x="{px(lx) + 10:.1f}" y="{yy + 3.5:.1f}" font-size="11" '
                       f'fill="var(--text-secondary)">{esc(s["name"][:14])}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(names: list[str], dot: bool = False) -> str:
    if len(names) < 2:
        return ""   # one series needs no legend; the heading names it
    parts = []
    for i, n in enumerate(names[:8]):
        cls = "keydot" if dot else "key"
        parts.append(f'<span><i class="{cls}" style="background:var(--s{i % 8 + 1})"></i>'
                     f'{esc(n)}</span>')
    return f'<div class="legend">{"".join(parts)}</div>'


def table(headers: list[str], rows: list[list], num_cols: set[int] | None = None) -> str:
    num_cols = num_cols or set()
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{esc(h)}</th>'
                 for i, h in enumerate(headers))
    trs = []
    for r in rows:
        tds = "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>'
                      for i, c in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def table_view(tid: str, headers: list[str], rows: list[list], num_cols=None) -> str:
    """Every chart gets one of these. A chart persuades; a table lets someone check."""
    return (f'<button onclick="toggleTable(\'{tid}\', this)">Show data table</button>'
            f'<div class="tv" id="{tid}">{table(headers, rows, num_cols)}</div>')


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def _suite_view(suite: str, rows: list[dict], runs_dir: Path) -> str:
    rows = sorted(rows, key=lambda r: r.get("points") or 0, reverse=True)
    best = rows[0]
    parts: list[str] = []

    # ---- hero + tiles -------------------------------------------------
    total_tokens = sum(r.get("train_tokens") or 0 for r in rows)
    total_wall = sum(r.get("wall_clock_s") or 0 for r in rows)
    parts.append(
        '<div class="card"><div style="display:flex;gap:28px;flex-wrap:wrap;'
        'align-items:flex-end;justify-content:space-between">'
        f'<div><p class="sub">Best on suite <span class="mono">{esc(suite)}</span></p>'
        f'<div class="hero">{fmt(best.get("points"))}<span style="font-size:20px;'
        f'color:var(--text-secondary)"> / 100 points</span></div>'
        f'<p class="sub">{esc(best["name"])} &middot; {esc(best.get("kind"))}'
        f'{" &middot; from " + esc(best["parent"]) if best.get("parent") else ""}</p></div>'
        '</div></div>')
    # A zero here means "not recorded", not "no tokens" — post-training stages count
    # examples and iterations, not tokens. Printing 0 would be a lie of omission.
    tok_note = ('' if total_tokens
                else '<div class="small">not recorded for post-training stages</div>')
    parts.append(
        '<div class="tiles">'
        f'<div class="tile"><div class="label">Runs on this suite</div>'
        f'<div class="value">{len(rows)}</div></div>'
        f'<div class="tile"><div class="label">Tokens trained (total)</div>'
        f'<div class="value">{human(total_tokens) if total_tokens else "&mdash;"}</div>'
        f'{tok_note}</div>'
        f'<div class="tile"><div class="label">Compute (wall clock)</div>'
        f'<div class="value">{total_wall / 60:.1f}<span style="font-size:14px"> min</span></div></div>'
        f'<div class="tile"><div class="label">Largest model</div>'
        f'<div class="value">{human(max((r.get("params_total") or 0) for r in rows))}</div></div>'
        '</div>')

    # ---- points bars ---------------------------------------------------
    bar_rows, tv_rows = [], []
    for r in rows:
        comp = " &middot; ".join(f"{k} {v['points']:.1f}" for k, v in (r.get("breakdown") or {}).items())
        bar_rows.append((r["name"], r.get("points") or 0.0,
                         f"<b>{r['name']}</b><br>{r.get('kind')} &middot; "
                         f"{human(r.get('params_total'))} params<br>{comp or 'no breakdown'}"))
        tv_rows.append([r["name"], r.get("kind"), f"{r.get('points'):.2f}",
                        f'<span class="mono">{esc(r.get("config_hash"))}</span>', comp or "-"])
    parts.append(
        '<div class="card"><h2>Composite points by run</h2>'
        '<p class="sub">Weighted composite, 0-100. Component weights are published in '
        'evaluate.POINTS_WEIGHTS &mdash; a composite whose weights are not visible is a '
        'way of hiding regressions.</p>'
        + svg_hbars(bar_rows, vmax=100)
        + table_view(f"tv-points-{suite}", ["run", "kind", "points", "config", "components"],
                     tv_rows, {2})
        + '</div>')

    # ---- loss curves ---------------------------------------------------
    # One chart PER LOGGED METRIC (small multiples), never several objectives on one
    # y-axis. A pretraining loss and an RL reward share no scale and no meaning; putting
    # them on one axis invents a comparison that does not exist.
    by_metric: dict[str, list[dict]] = {}
    curve_tv = []
    for r in rows[:12]:
        ms = read_metrics(runs_dir / r["name"])
        for key in ("eval/val_loss", "sft/val_loss", "kd/loss", "reward/mean",
                    "eval/arith_exact"):
            pts = [(m.get("_step", i), m[key]) for i, m in enumerate(ms) if key in m]
            if len(pts) >= 2:
                by_metric.setdefault(key, []).append({"name": r["name"], "points": pts})
                curve_tv.append([r["name"], key, len(pts), fmt(pts[0][1], 4), fmt(pts[-1][1], 4)])
    if by_metric:
        charts = []
        for key, ser in by_metric.items():
            charts.append(
                f'<h3 style="font-size:13px;margin:14px 0 2px">{esc(key)}'
                f'<span class="small"> &middot; {len(ser)} run{"s" if len(ser) != 1 else ""}</span></h3>'
                + legend([x["name"] for x in ser])
                + svg_lines(ser, y_label=key.split("/")[-1]))
        parts.append(
            '<div class="card"><h2>Curves, one chart per logged metric</h2>'
            '<p class="sub">Against optimizer step. Deliberately not overlaid: a '
            'pretraining loss and an RL reward share no scale, so putting them on one '
            'axis would invent a comparison that does not exist.</p>'
            + "".join(charts)
            + table_view(f"tv-curves-{suite}", ["run", "metric logged", "points", "first", "last"],
                         curve_tv, {2, 3, 4})
            + '</div>')

    # ---- MoE utilisation ----------------------------------------------
    moe_run = next((r for r in rows if any(k.startswith("moe/expert_") for k in r.get("metrics", {}))), None)
    if moe_run is None:
        for r in rows:
            ms = read_metrics(runs_dir / r["name"])
            if ms and any(k.startswith("moe/expert_") for k in ms[-1]):
                moe_run = r
                break
    if moe_run is not None:
        ms = read_metrics(runs_dir / moe_run["name"])
        last = next((m for m in reversed(ms) if any(k.startswith("moe/expert_") for k in m)), {})
        fr = sorted((int(k.split("_")[1]), v) for k, v in last.items()
                    if k.startswith("moe/expert_") and k.endswith("_frac"))
        if fr:
            ideal = 1.0 / len(fr)
            parts.append(
                '<div class="card"><h2>MoE expert utilisation &mdash; '
                f'{esc(moe_run["name"])}</h2>'
                f'<p class="sub">Share of routed tokens per expert. Balanced would be '
                f'{ideal:.3f} each. A collapsing router shows up here long before it shows '
                f'up in the loss.</p>'
                + svg_hbars([(f"expert {i}", v,
                              f"expert {i}<br>{v:.4f} of routed tokens<br>"
                              f"balanced would be {ideal:.4f}") for i, v in fr],
                            vmax=max(max(v for _, v in fr), ideal * 1.5), label_w=90,
                            ref=(ideal, "balanced"))
                + table_view(f"tv-moe-{suite}", ["expert", "share of tokens", "vs balanced"],
                             [[f"expert {i}", f"{v:.4f}", f"{v / ideal:.2f}x"] for i, v in fr],
                             {1, 2})
                + '</div>')

    # ---- lineage deltas -----------------------------------------------
    by_name = {r["name"]: r for r in rows}
    delta_rows = []
    for r in rows:
        p = by_name.get(r.get("parent") or "")
        if not p:
            continue
        shared = sorted(set(r.get("metrics", {})) & set(p.get("metrics", {})))
        for k in shared:
            a, b = p["metrics"][k], r["metrics"][k]
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            if k.endswith("stderr") or "params" in k or "wall" in k or "tokens" in k:
                continue
            d = b - a
            lower_better = "loss" in k or "ppl" in k
            good = (d < 0) if lower_better else (d > 0)
            if abs(d) < 1e-9:
                cue = '<span class="small">&#8212; no change</span>'
            else:
                cue = (f'<span class="{"up" if good else "down"}">'
                       f'{"&#9650;" if d > 0 else "&#9660;"} {abs(d):.4f} '
                       f'{"better" if good else "worse"}</span>')
            delta_rows.append([f'<span class="mono">{esc(k)}</span>', f"{a:.4f}", f"{b:.4f}", cue,
                               f'{esc(p["name"])} &rarr; {esc(r["name"])}'])
    if delta_rows:
        parts.append(
            '<div class="card"><h2>Stage-over-stage deltas</h2>'
            '<p class="sub">Each row compares a run against its parent checkpoint. Direction '
            'is labelled in words as well as colour and arrow &mdash; never colour alone.</p>'
            + table(["metric", "parent", "child", "change", "lineage"], delta_rows, {1, 2})
            + '</div>')

    # ---- leaderboard ---------------------------------------------------
    lb = []
    for i, r in enumerate(rows, 1):
        acc = r.get("metrics", {}).get("arith_exact/exact_match")
        se = r.get("metrics", {}).get("arith_exact/stderr")
        acc_s = "-" if acc is None else (f"{acc:.3f} &plusmn; {se:.3f}" if se else f"{acc:.3f}")
        lb.append([str(i), esc(r["name"]), esc(r.get("kind")),
                   f'{r.get("points"):.2f}', acc_s,
                   human(r.get("params_total")), human(r.get("params_active")),
                   esc(r.get("data") or "-"), esc(r.get("parent") or "-"),
                   f'<span class="mono">{esc(r.get("git_sha") or "-")}</span>'])
    parts.append(
        '<div class="card"><h2>Leaderboard</h2>'
        '<p class="sub">Every row is one run on this suite. Accuracy carries its standard '
        'error; params are reported total AND active-per-token, which are the same number '
        'only for dense models.</p>'
        + table(["#", "run", "kind", "points", "arith exact match", "params", "active",
                 "data", "parent", "code"], lb, {0, 3})
        + '</div>')
    return "".join(parts)


def build_dashboard(registry_path: str | Path = DEFAULT_PATH,
                    runs_dir: str | Path = "runs",
                    out_path: str | Path = "artifacts/dashboard.html",
                    title: str = "aienh — model leaderboard") -> Path:
    runs_dir = Path(runs_dir)
    rows = [r for r in load_all(registry_path) if r.get("points") is not None]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out.write_text("<h1>No scored runs yet — run the pipeline first.</h1>", encoding="utf-8")
        return out

    suites: dict[str, list[dict]] = {}
    for r in rows:
        suites.setdefault(r.get("suite_hash") or "unknown", []).append(r)
    order = sorted(suites, key=lambda s: -len(suites[s]))

    opts = "".join(
        f'<option value="{esc(s)}">{esc(s)} — {len(suites[s])} run'
        f'{"s" if len(suites[s]) != 1 else ""}</option>' for s in order)
    views = "".join(
        f'<div data-suite-view="{esc(s)}" style="{"" if s == order[0] else "display:none"}">'
        f'{_suite_view(s, suites[s], runs_dir)}</div>' for s in order)

    unscored = len(load_all(registry_path)) - len(rows)
    note = ""
    if len(order) > 1:
        note = ('<div class="note"><b>Multiple eval suites present.</b> A suite hash covers the '
                'task list, the item seed and the prompt template. Runs scored under different '
                'hashes sat different exams and are not comparable, so they are never mixed into '
                'one table here — use the selector above.</div>')

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body class="viz-root"><div id="tip"></div><div class="wrap">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">
    <div><h1>{esc(title)}</h1>
    <p class="sub">Generated from <span class="mono">runs/registry.jsonl</span>. Self-contained:
    no network, no build step. {len(rows)} scored run{"s" if len(rows) != 1 else ""}{f", {unscored} unscored" if unscored > 0 else ""}.</p></div>
    <button onclick="setTheme()">Toggle theme</button>
  </div>
  <div class="filters"><label for="suite">Eval suite</label>
    <select id="suite" onchange="selectSuite(this)">{opts}</select></div>
  {note}
  {views}
  <p class="small" style="margin-top:32px">Charts follow one convention throughout: single-series
  charts carry no legend, every chart has a table view, accuracies carry n and standard error, and
  deltas are labelled in words as well as colour.</p>
</div><script>{JS}</script></body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":  # python -m aienh.dashboard
    p = build_dashboard()
    print(f"wrote {p} ({p.stat().st_size / 1024:.1f} KB)")
