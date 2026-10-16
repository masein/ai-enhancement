"""21c: a report's page as it is — the page masein publishes, served by the
board, beside 21's view in the board's own style.

The page (eval_tasks/reports/pages/<file>) goes in as it was published and is
never edited: it is written for a "dashboard" runtime, and gets one of ours
(service/static/reports/dash.js) that gives it exactly what it uses, the
theme tokens it reads (theme.css, light and dark) and d3 v7 kept in the repo
(ISC licence). Nothing on it is fetched from another host: a Content-Security-
Policy says so to the browser too.

Its four datasets are built live from the same tables 21's view reads, every
number rounded as 21's view rounds it (JavaScript's toFixed), so both show the
same numbers cell for cell:
- ours: a row per model and benchmark, with its status line;
- others: every other model the report file names, with its source in words;
- tokens: the token stats as the stats worker kept them;
- loops: the loop examples as kept, the shown half only, with the note the
  report file gives.
A header line of ours says when each source last changed."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from . import reports

STATIC = Path(__file__).resolve().parent / "static" / "reports"
ASSETS = {"dash.js": "text/javascript; charset=utf-8", "theme.css": "text/css; charset=utf-8",
          "d3.v7.9.0.min.js": "text/javascript; charset=utf-8"}
DATASETS = ("ours", "others", "tokens", "loops")
WORKING = "Working these out, back in a minute."
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; "
       "base-uri 'self'; form-action 'none'; frame-ancestors 'self'")


class Working(Exception):
    """a dataset of stats none of which is worked out yet"""


fixed = reports.fixed            # JavaScript's toFixed, as 21's view rounds


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def page_of(rep: dict) -> dict | None:
    """the page a report file names: {file, params, datasets}"""
    p = rep.get("page")
    if not isinstance(p, dict) or not p.get("file"):
        return None
    f = (reports.REPORTS_DIR / str(p["file"])).resolve()
    if reports.REPORTS_DIR.resolve() not in f.parents or not f.is_file():
        return None
    return {"file": f, "params": p.get("params") or {},
            "datasets": [d for d in p.get("datasets") or DATASETS if d in DATASETS]}


# ---------------------------------------------------------------------------
# the datasets
# ---------------------------------------------------------------------------

def status(spec: dict, row: dict) -> str:
    """a row's status as the page says it: "Complete · 21 of 21 chunks",
    "Running · 20 of 21 chunks in"; a reference row, the file's own words"""
    if spec.get("page_status"):
        return str(spec["page_status"])
    cells = row["cells"].values()
    have = sum(1 for c in cells if c.get("state") == "score")
    waits = any(c.get("state") == "waits" for c in cells)
    ch = row.get("chunks")
    if ch:
        done, n = ch
        if done < n:
            return f"Running · {done} of {n} chunks in"
        if have == len(row["cells"]):
            return f"Complete · {n} of {n} chunks"
        if waits:
            return f"Grading · {n} of {n} chunks in"
        return f"{have} of {len(row['cells'])} benchmarks in · {n} of {n} chunks"
    return row["status"]


def ours(rep: dict, rows: list[dict]) -> list[dict]:
    out = []
    for i, (spec, r) in enumerate(zip(rep["ours"], rows)):
        st = status(spec, r)
        for b in rep["benchmarks"]:
            c = r["cells"].get(b["key"]) or {}
            on = c.get("state") == "score" and c.get("v") is not None
            n = c.get("answers") if on and _int(c.get("answers")) and c["answers"] > 0 else None
            ro = (c.get("ranOut") or 0) if n else None
            out.append({"model": r["label"], "order": i, "status": st, "bench": b["key"],
                        "benchmark": b["label"],
                        "score": fixed(100 * c["v"]) if on else None,
                        "se": fixed(100 * c["se"]) if on and c.get("se") is not None else None,
                        "ran_out_n": ro, "answers": n,
                        "ran_out_pct": fixed(100 * ro / n) if n else None})
    return out


source = reports.source_words      # where a model's numbers come from, in words


def others(rep: dict, theirs: list[dict]) -> list[dict]:
    """every other model, its numbers in percent: Epoch AI's own runs with
    their error, HLE marked where it came from ("other": Epoch's collection,
    "lab": a card), a card's number on its own"""
    benches = [b for b in rep["benchmarks"] if b.get("compare")]
    ext = {k for t in rep.get("theirs") or [] for k, v in t.items()
           if isinstance(v, dict) and v.get("from") == "epoch_external"}
    own = [b for b in benches if b.get("epoch") and b["key"] not in ext]
    src = [b for b in benches if b["key"] in ext]
    rest = [b for b in benches if b not in own and b not in src]
    out = []
    for t in theirs:
        cells = t["cells"]
        row = {"model": t["label"], "maker": t.get("maker") or "", "size": t.get("size") or ""}

        def v(k):
            x = (cells.get(k) or {}).get("v")
            return fixed(100 * x) if x is not None else None
        for b in own:
            c = cells.get(b["key"]) or {}
            row[b["key"]] = v(b["key"])
            # Epoch AI's own run gives its error, none where it scored 100%
            # (as 0); a card gives none
            row[b["key"] + "_se"] = (None if c.get("v") is None or c.get("src") == "card"
                                     else fixed(100 * (c.get("se") or 0)))
        for b in src:
            c = cells.get(b["key"]) or {}
            row[b["key"]] = v(b["key"])
            row[b["key"] + "_src"] = ("" if c.get("v") is None else "other"
                                      if c.get("src") == "epoch_external" else "lab"
                                      if c.get("src") == "card" else "")
        for b in rest:
            row[b["key"]] = v(b["key"])
        row["source"] = t.get("source") or source(benches, cells)
        out.append(row)
    return out


STATES = {"score": "all answers in, scored", "waits": "all answers in, waiting for grading"}
TOKEN_FIELDS = ("think_mean", "think_med", "tok_mean", "tok_med", "tok_p90", "tok_max", "tok_sum",
                "think_share", "ran_out", "right_n", "right_mean", "wrong_n", "wrong_mean")


def tokens(rep: dict, rows: list[dict], stats: dict) -> list[dict]:
    """the token stats as kept, a row a model's benchmark"""
    out = []
    for i, r in enumerate(rows):
        for j, b in enumerate(rep["benchmarks"]):
            s = (stats.get(r["label"]) or {}).get(b["task"])
            if not s:
                continue
            state = (r["cells"].get(b["key"]) or {}).get("state")
            # answers neither right, wrong nor cut off: a grader's (or Epoch's
            # model check's) verdict still to come
            ungraded = ((s.get("n_tok") or 0) - (s.get("right_n") or 0) - (s.get("wrong_n") or 0)
                        - (s.get("ran_out") or 0)) > 0
            out.append({"model": r["label"], "order": i, "bench": b["key"],
                        "benchmark": b["label"], "bench_order": j,
                        "state": STATES["waits"] if ungraded or state == "waits"
                        else STATES.get(state, "all answers in, not scored yet"),
                        "answers": s.get("n"), **{k: s.get(k) for k in TOKEN_FIELDS}})
    return sorted(out, key=lambda x: (x["model"], x["bench_order"]))


def loops(rep: dict, examples: list[dict]) -> list[dict]:
    """the loop examples as kept — each from the shown half, its note the
    report file's"""
    lp = rep.get("loops") or {}
    label = next((b["label"] for b in rep["benchmarks"] if b["task"] == lp.get("task")),
                 lp.get("task") or "")
    out = []
    for k, x in enumerate(examples):
        if x.get("refused") or x.get("pending") or not x.get("first") or not x.get("item"):
            continue
        f, it, a = x["first"], x["item"], x.get("against") or {}
        out.append({"id": x["id"], "subject": it.get("subject") or "",
                    "question": it.get("question") or "", "right": it.get("right") or "",
                    "right_text": it.get("right_text") or "",
                    "q4_verdict": a.get("verdict") or "no stored answer",
                    # 21d.1: "thought for N characters": the thinking alone
                    "q4_chars": a.get("thought", a.get("chars")) or 0,
                    "s_chars": f.get("thought", f["chars"]), "waits": f["waits"],
                    "top_line": f["line"], "top_count": f["line_n"], "excerpt": f["excerpt"],
                    "what": x.get("note") or "", "order": k + 1, "benchmark": label,
                    "s_verdict": f["verdict"]})
    return out


def _report(rid: str) -> dict:
    rep = reports.files().get(rid)
    if rep is None or rep.get("kind") == "ran_out" or not page_of(rep):
        raise KeyError(f"no report page {rid!r}")
    return rep


def dataset(rid: str, did: str, payload: dict | None = None) -> list[dict]:
    """one of the page's datasets, live. Working when it is stats none of
    which is worked out yet (the worker has been woken)"""
    rep = _report(rid)
    if did not in page_of(rep)["datasets"]:
        raise KeyError(f"no dataset {did!r} on {rid!r}'s page")
    if did == "others":
        return others(rep, reports.their_rows(rep))
    if payload is None:
        from . import app
        payload = app.results_payload()
    rows = reports.our_rows(rep, payload.get("cells") or {})
    if did == "ours":
        return ours(rep, rows)
    if did == "tokens":
        stats, ats = reports.kept_stats(rep, rows)
        got = tokens(rep, rows, stats)
        if not got and reports.stats_working(rid, rep):
            raise Working(WORKING)
        return got
    examples = reports.loops(rep, rows, compute=False)
    got = loops(rep, examples)
    if not got and any(x.get("pending") for x in examples) and reports.stats_working(rid, rep):
        raise Working(WORKING)
    return got


def info(rid: str, payload: dict | None = None) -> dict:
    """the shell's header line: when each source last changed, and the stats"""
    rep = _report(rid)
    if payload is None:
        from . import app
        payload = app.results_payload()
    rows = reports.our_rows(rep, payload.get("cells") or {})
    theirs = reports.their_rows(rep)
    src = reports._sources(rep, rows, theirs)
    _, ats = reports.kept_stats(rep, rows)
    work = reports.stats_working(rid, rep)
    at = reports.stats_at(ats)
    if not ats:
        stats_words = "working these out, back in a minute" if work else "none stored yet"
    elif work:
        stats_words = (f"{at['words']}; the newest answers being worked out "
                       f"({work.get('done', 0)} of {work.get('todo', 0)})")
    else:
        stats_words = at["words"]
    cards = src["cards"]["at"]
    return {"id": rid, "title": rep.get("title") or rid,
            "board": f"#tab=reports&report={rid}",
            "sources": [
                {"key": "ours", "at": src["ours"]["at"], "words": "Our scores: " + src["ours"]["words"]},
                {"key": "epoch", "at": src["epoch"]["at"], "words": "Epoch AI: " + src["epoch"]["words"]},
                {"key": "cards", "at": cards, "words": "Model cards: " + (
                    f"last entered {reports.day(cards)}" if cards else "none entered yet")},
                {"key": "stats", "at": at["at"],
                 "words": "Thinking tokens and loop examples: " + stats_words}],
            "credit": {"words": src["epoch"]["credit"], "url": src["epoch"]["url"]},
            "working": work}


# ---------------------------------------------------------------------------
# the shell
# ---------------------------------------------------------------------------

def asset(name: str) -> tuple[Path, str] | None:
    if name not in ASSETS:
        return None
    p = STATIC / name
    return (p, ASSETS[name]) if p.is_file() else None


def _v(name: str) -> str:
    return hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:10]


SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script>try { var t = localStorage.getItem('bench-theme');
  if (t && t !== 'auto') document.documentElement.setAttribute('data-theme', t); } catch (e) {}</script>
<link rel="stylesheet" href="../static/theme.css?v=__THEME_V__">
<script src="../static/d3.v7.9.0.min.js"></script>
<script src="../static/dash.js?v=__DASH_V__" data-api="../../api/reports/__RID__/datasets"
  data-sets="__SETS__" data-params="__PARAMS__"></script>
</head>
<body>
<header class="dash-shell" data-dash-shell>
  <a class="dash-shell-back" href="../../__BOARD__">&larr; Reports on the board</a>
  <p class="dash-shell-fresh" data-dash-fresh>Reading when each source last changed&hellip;</p>
</header>
<main id="dash-root">
__PAGE__
</main>
</body>
</html>
"""


def shell(rid: str) -> str | None:
    """the page inside our shell — None when the report has no page"""
    rep = reports.files().get(rid)
    pg = page_of(rep) if rep and rep.get("kind") != "ran_out" else None
    if pg is None:
        return None
    page = pg["file"].read_text(encoding="utf-8")
    m = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
    title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else (rep.get("title") or rid)
    esc = lambda x: html.escape(x, quote=True)                  # noqa: E731
    return (SHELL.replace("__TITLE__", esc(title))
            .replace("__THEME_V__", _v("theme.css")).replace("__DASH_V__", _v("dash.js"))
            .replace("__RID__", esc(quote(rid)))
            .replace("__SETS__", esc(json.dumps(pg["datasets"])))
            .replace("__PARAMS__", esc(json.dumps(pg["params"])))
            .replace("__BOARD__", esc(f"#tab=reports&report={rid}"))
            .replace("__PAGE__", page))
