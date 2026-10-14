"""21: Reports — a place on the board for reports built from its own tables.

A report is a file in eval_tasks/reports/ (reported.REPORTS_DIR): which of our
rows it shows, which benchmarks, and which other models — each by the exact
Epoch AI version to read, HLE from Epoch's collection by its version, and the
cells a model card gives, entered through the board's own card entry
(scripts/enter_report_cards.py) with their links. A later report is a new
file there; the place lists every one.

Every number on a report's page comes from the board's tables, read when the
page asks:
- ours, as the board shows them (results_payload's cells: the score, its
  standard error, the answers and how many ran out of room); a benchmark that
  isn't finished is a dash, never an older number;
- Epoch AI's, from the daily import (12m.2), only at the versions the file
  names;
- a card's, from its entry, with the card's link;
- the thinking tokens and the loop examples, worked out here from the stored
  answers, kept until a file they read changes (the next import or scoring).

Never an Artificial Analysis number: a report is shareable. Never the hidden
half: a loop example from it is refused. Nothing here starts a run, an import
or a grading batch."""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import statistics
import sys
import threading
from collections import Counter
from pathlib import Path

from . import config, db, reported

REPORTS_DIR = reported.REPORTS_DIR
OWN_RUN = "Epoch AI's own run"
NOTES = [
    "Protocols. Our scores follow Epoch AI's protocols for GPQA Diamond, OTIS Mock AIME, MATH "
    "Level 5 and SimpleQA Verified. HLE is the text-only set, judged by o3-mini with CAIS's "
    "prompt. SimpleQA is graded by gpt-4.1 with Google's prompt. MATH Level 5 and OTIS include "
    "Epoch's model check where it has run.",
    "HLE marked † was collected by Epoch AI from another source, usually the lab; it is not "
    "Epoch's own run. HLE is without tools everywhere, and labs don't all say which question set "
    "they used.",
    "\"Level\" means the gap between the two scores is no more than two standard errors, ours "
    "and theirs combined. Epoch AI gives an error for its own runs. Lab and model-card figures "
    "come without one and use each lab's own setup, so a small gap to them says little.",
    "ARC-AGI-2 is left out of the comparison. Published numbers are on ARC Prize's held-back "
    "set; ours is the public set.",
    "MATH Level 5: Epoch AI hasn't run it on the newest models, so that column holds older "
    "models only.",
    "Reasoning settings: each model is shown at the setting the file names, normally its "
    "highest. gpt-oss-20b is at medium, which scored higher than its high setting.",
]


def _scripts() -> None:
    p = str(Path(__file__).resolve().parent.parent / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------

def files() -> dict[str, dict]:
    """every report file, by its id — one that can't be read is left out"""
    out = {}
    for f in sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.is_dir() else []:
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rep, dict) and rep.get("id") and isinstance(rep.get("ours"), list):
            out[str(rep["id"])] = rep
    return out


def row_dir(mid: str) -> Path:
    """a board row's folder: "served/x · thinking" is served__x__thinking"""
    base = mid.removesuffix(" · thinking")
    return config.OUT_DIR / (base.replace("/", "__") + ("__thinking" if base != mid else ""))


def _local(ts: float) -> _dt.datetime:
    _scripts()
    import report_lm_eval as rpt
    return _dt.datetime.fromtimestamp(ts, rpt._TZ) if rpt._TZ else _dt.datetime.fromtimestamp(ts)


def when(ts: float | None) -> str:
    """"14 Oct 11:20" """
    if not ts:
        return ""
    t = _local(ts)
    return f"{t.day} {t.strftime('%b')} {t:%H:%M}"


def day(ts: float | None) -> str:
    """"14 Oct" """
    if not ts:
        return ""
    t = _local(ts)
    return f"{t.day} {t.strftime('%b')}"


# ---------------------------------------------------------------------------
# ours: what the board shows, and each row's status
# ---------------------------------------------------------------------------

def _registry(d: Path) -> dict:
    try:
        reg = json.loads((d / "frontier_imports.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return reg if isinstance(reg, dict) else {}


def _boxes(reg: dict) -> list[str]:
    """the box labels a row's imports came from ("L13")"""
    out = []
    for e in (reg.get("tasks") or {}).values():
        if isinstance(e, dict):
            out += [str(b) for b in e.get("boxes") or [] if b] + ([str(e["box"])]
                                                                  if e.get("box") else [])
    for sh in (reg.get("shards") or {}).values():
        for e in ((sh or {}).get("have") or {}).values():
            if isinstance(e, dict) and e.get("box"):
                out.append(str(e["box"]))
    return out


def chunks(reg: dict) -> tuple[int, int] | None:
    """(boxes in, boxes planned) of the box plan a row's imports came from —
    a box is in when every step of it is: each benchmark it asks, and its
    shard, on the row (held, or merged whole); None when the row's boxes name
    no plan"""
    _scripts()
    import frontier_box as fbx
    letters = Counter(b[:1].upper() for b in _boxes(reg)
                      if b[:1].upper() in fbx.PLANS and b.upper() in fbx.PLANS[b[:1].upper()])
    if not letters:
        return None
    plan = fbx.PLANS[letters.most_common(1)[0][0]]
    tasks, shards = reg.get("tasks") or {}, reg.get("shards") or {}

    def part_in(t: str, shard: str) -> bool:
        if not shard:
            return t in tasks
        i, n = (int(x) for x in shard.split("/"))
        if isinstance(tasks.get(t), dict) and tasks[t].get("shards") == n:
            return True
        sh = shards.get(t) or {}
        return sh.get("n") == n and str(i) in (sh.get("have") or {})
    boxes = [[st for st in box if st[0] != "parity"] for box in plan.values()]
    done = sum(1 for box in boxes if all(part_in(t, shard) for _, ts, shard, _ in box
                                         for t in ts))
    return done, len(boxes)


def _answers_whole(d: Path, task: str) -> bool:
    """the row holds this benchmark's answers (not held in shards)"""
    from . import frontier as sf
    return (sf.task_dir(d, task) / sf.ANSWERS).is_file()


def our_rows(rep: dict, cells: dict) -> list[dict]:
    """each of our rows as the board shows it: per benchmark the score, its
    standard error, the answers and how many ran out of room — or a dash and
    why — and the row's status line"""
    _scripts()
    import frontier as fb
    out = []
    for o in rep["ours"]:
        mid, d = o["row"], row_dir(o["row"])
        reg = _registry(d)
        per, waits = {}, []
        for b in rep["benchmarks"]:
            t = b["task"]
            c = (cells.get(t) or {}).get(mid)
            if c and not c.get("shardsHeld"):
                per[b["key"]] = {"v": c.get("v"), "se": c.get("se"), "answers": c.get("answers"),
                                 "ranOut": c.get("ranOut"), "how": c.get("how") or "",
                                 "final": c.get("final") is not False, "state": "score"}
                continue
            if _answers_whole(d, t) and fb.BENCH.get(t, {}).get("grader"):
                waits.append(b["label"])
                per[b["key"]] = {"v": None, "state": "waits", "why": "waits for grading"}
            elif c and c.get("shardsHeld"):
                per[b["key"]] = {"v": None, "state": "held", "why": c["shardsHeld"]}
            else:
                per[b["key"]] = {"v": None, "state": "none", "why": "not in yet"}
        ch = chunks(reg)
        parts = []
        if ch and ch[0] < ch[1]:
            parts.append(f"{ch[0]} of {ch[1]} chunks in")
        if waits:
            parts.append((waits[0] if len(waits) == 1 else
                          ", ".join(waits[:-1]) + " and " + waits[-1])
                         + (" waits" if len(waits) == 1 else " wait") + " for grading")
        have = sum(1 for x in per.values() if x["state"] == "score")
        if not parts:
            parts.append("Complete" if have == len(per) else "Not run yet" if not have
                         else f"{have} of {len(per)} benchmarks in")
        out.append({"id": "ours:" + o["label"], "label": o["label"], "row": mid, "ours": True,
                    "colour": o.get("colour"), "pick": bool(o.get("pick")),
                    "note": o.get("note") or "", "cells": per, "status": " · ".join(parts),
                    "chunks": list(ch) if ch else None, "dir": str(d)})
    return out


def ours_changed(rows: list[dict]) -> float | None:
    """when our numbers last changed: the newest import or scoring of our rows"""
    best = None
    for r in rows:
        d = Path(r["dir"])
        cands = []
        try:
            cands.append((d / "frontier_imports.json").stat().st_mtime)
        except OSError:
            pass
        for f in d.glob("*_0shot/frontier/results_*.json"):
            try:
                cands.append(float(json.loads(f.read_text(encoding="utf-8")).get("date") or 0))
            except (OSError, ValueError, TypeError):
                pass
        if cands:
            best = max(best or 0, *cands)
    return best


# ---------------------------------------------------------------------------
# theirs: Epoch AI at the versions named, HLE from Epoch's collection, cards
# ---------------------------------------------------------------------------

def _is_hle(r: dict) -> bool:
    name = str(r.get("benchmark") or "").lower()
    return "hle" in name or "humanity" in name or str(r.get("file") or "").startswith("hle")


def their_rows(rep: dict, scores: list[dict] | None = None) -> list[dict]:
    """each other model the file names, every number from the board's tables:
    Epoch AI's own runs at the version named, HLE from Epoch's collection by
    its version (†), a card's cell from its entry, with its link"""
    scores = db.reported_scores() if scores is None else scores
    own: dict[tuple[str, str], dict] = {}
    ext: dict[str, list[dict]] = {}
    cards: list[dict] = []
    for r in scores:
        if r.get("source") == "epoch" and r.get("version"):
            if r.get("setting") == OWN_RUN:
                own[(str(r["version"]), str(r.get("benchmark") or "").lower())] = r
            else:
                ext.setdefault(str(r["version"]), []).append(r)
        elif r.get("source") == "card":
            cards.append(r)
    out = []
    for t in rep.get("theirs") or []:
        per = {}
        for b in rep["benchmarks"]:
            spec = t.get(b["key"])
            cell = None
            if isinstance(spec, dict) and spec.get("from") == "card":
                k = reported.key(spec.get("model") or "", spec.get("maker") or "")
                hit = [c for c in cards if reported.canon(c.get("key") or "") == k
                       and c.get("benchmark") == spec.get("benchmark")
                       and c.get("url") == spec.get("url")]
                if hit:
                    c = max(hit, key=lambda x: x.get("id") or 0)
                    cell = {"v": c["value"], "se": None, "src": "card", "url": c["url"],
                            "card": spec.get("card") or "", "setting": c.get("setting") or "",
                            "date": c.get("date") or "", "by": c.get("by") or "",
                            "entered": c.get("id")}
                else:
                    cell = {"v": None, "src": "card", "missing": True, "url": spec.get("url"),
                            "card": spec.get("card") or "",
                            "why": "not entered yet (scripts/enter_report_cards.py)"}
            elif isinstance(spec, dict) and spec.get("from") == "epoch_external":
                hit = [r for r in ext.get(str(spec.get("version")), []) if _is_hle(r)]
                if hit:
                    r = hit[-1]
                    cell = {"v": r["value"], "se": None, "src": "epoch_external", "dagger": True,
                            "setting": r.get("setting") or "", "url": r.get("url") or ""}
            elif b.get("epoch") and t.get("epoch_version"):
                r = own.get((str(t["epoch_version"]), str(b["epoch"]).lower()))
                if r:
                    cell = {"v": r["value"], "se": r.get("se"), "src": "epoch",
                            "setting": OWN_RUN}
            per[b["key"]] = cell or {"v": None}
        out.append({"id": "theirs:" + t["label"], "label": t["label"], "ours": False,
                    "maker": t.get("maker") or "", "size": t.get("size"),
                    "group": t.get("group") or "", "version": t.get("epoch_version"),
                    "cells": per})
    return out


def card_entries(rep: dict, scores: list[dict] | None = None) -> list[dict]:
    """the file's card cells and their entries: what was entered, when, by whom"""
    rows = their_rows(rep, scores)
    out = []
    for t, r in zip(rep.get("theirs") or [], rows):
        for b in rep["benchmarks"]:
            spec = t.get(b["key"])
            if isinstance(spec, dict) and spec.get("from") == "card":
                c = r["cells"][b["key"]]
                out.append({"model": t["label"], "benchmark": b["label"], "value": spec.get("value"),
                            "card": spec.get("card"), "url": spec.get("url"),
                            "date": spec.get("date"), "entered": bool(c.get("entered")),
                            "by": c.get("by") or ""})
    return out


# ---------------------------------------------------------------------------
# where we stand: the sums
# ---------------------------------------------------------------------------

def band(se_a: float | None, se_b: float | None) -> float:
    """"level": a gap no more than 2·√(se_ours² + se_theirs²) — a number
    without a standard error (a card's) counts it as 0"""
    return 2 * math.sqrt((se_a or 0) ** 2 + (se_b or 0) ** 2)


def verdict(ours: dict, theirs: dict) -> str | None:
    """"higher", "lower" or "level" — ours against theirs; None without both"""
    if ours.get("v") is None or theirs.get("v") is None:
        return None
    gap = ours["v"] - theirs["v"]
    if abs(gap) <= band(ours.get("se"), theirs.get("se")) + 1e-12:
        return "level"
    return "higher" if gap > 0 else "lower"


def standing(pick: dict, models: list[dict], benches: list[dict]) -> dict:
    """where one of ours stands against every other model: the summary line,
    a tile a compared benchmark, and the head-to-head rows"""
    compared = [b for b in benches if b.get("compare")]
    others = [m for m in models if m["id"] != pick["id"]]
    count = Counter()
    tiles, rows = [], []
    for b in compared:
        mine = pick["cells"].get(b["key"]) or {}
        dots = []
        for m in others:
            c = m["cells"].get(b["key"]) or {}
            v = verdict(mine, c)
            if c.get("v") is None:
                continue
            dots.append({"label": m["label"], "v": c["v"], "se": c.get("se"), "state": v,
                         "ours": m["ours"], "dagger": bool(c.get("dagger")),
                         "card": c.get("src") == "card"})
            if v:
                count[v] += 1
        tiles.append({"key": b["key"], "label": b["label"], "v": mine.get("v"),
                      "se": mine.get("se"), "dots": sorted(dots, key=lambda x: -x["v"])})
    for m in others:
        diffs, hi, lo = {}, 0, 0
        for b in compared:
            mine, c = pick["cells"].get(b["key"]) or {}, m["cells"].get(b["key"]) or {}
            v = verdict(mine, c)
            if v is None:
                diffs[b["key"]] = None
                continue
            diffs[b["key"]] = {"d": mine["v"] - c["v"], "state": v, "ours": mine["v"],
                               "theirs": c["v"], "band": band(mine.get("se"), c.get("se"))}
            hi += v == "higher"
            lo += v == "lower"
        rows.append({"label": m["label"], "ours": m["ours"], "diffs": diffs, "higher": hi,
                     "lower": lo})
    n = sum(count.values())
    line = (f"{pick['label']} is higher than {count['higher']} and lower than {count['lower']} "
            f"of the {n} scores it can be set against, and level with {count['level']}.")
    return {"pick": pick["label"], "line": line, "higher": count["higher"],
            "lower": count["lower"], "level": count["level"], "n": n, "tiles": tiles,
            "rows": rows,
            "reference": {b["key"]: (pick["cells"].get(b["key"]) or {}).get("v")
                          for b in compared}}


# ---------------------------------------------------------------------------
# thinking tokens, from the stored answers
# ---------------------------------------------------------------------------

_tok_lock = threading.Lock()
_tok_mem: dict[str, dict] = {}
_work = {"thread": None, "todo": 0, "done": 0}


def _cache_file() -> Path:
    return Path(config.BENCH_ROOT) / "reports" / "token-stats.json"


def _cache_get(key: str) -> dict | None:
    with _tok_lock:
        if not _tok_mem:
            try:
                _tok_mem.update(json.loads(_cache_file().read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
            _tok_mem.setdefault("_", {})
        return _tok_mem.get(key)


def _cache_put(key: str, value: dict) -> None:
    with _tok_lock:
        _tok_mem[key] = value
        f = _cache_file()
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".part")
            tmp.write_text(json.dumps(_tok_mem), encoding="utf-8")
            tmp.replace(f)
        except OSError:
            pass


def samples_file(d: Path, task: str) -> Path | None:
    from . import frontier as sf
    got = sorted(sf.task_dir(d, task).glob(f"samples_{task}_*.jsonl"))
    return got[-1] if got else None


def _key(f: Path, what: str) -> str:
    st = f.stat()
    return f"{what}|{f}|{st.st_mtime_ns}|{st.st_size}"


def p90(xs: list[int]) -> int:
    """the 90th percentile, nearest rank: the smallest value with 90% of them
    at or below it"""
    s = sorted(xs)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def think_share(text: str) -> float:
    """the thinking part's share of an answer's characters — every thinking
    block, wherever it sits"""
    _scripts()
    from frontier_graders import think_split
    if not text:
        return 0.0
    return sum(len(x) for x in think_split(text)[1]) / len(text)


def stats_of(runs: list[tuple[int | None, bool, bool | None, str]]) -> dict:
    """one row's benchmark: runs of (output tokens, ran out, right, answer).
    Output tokens are llama-server's own count; thinking tokens that count
    times the thinking part's share of the answer's characters. Right and
    wrong leave out the answers that ran out of room"""
    toks = [t for t, _, _, _ in runs if isinstance(t, int) and not isinstance(t, bool)]
    shares = [think_share(a) for t, _, _, a in runs
              if isinstance(t, int) and not isinstance(t, bool)]
    thinks = [t * s for t, s in zip(toks, shares)]
    right = [t for t, out, ok, _ in runs if isinstance(t, int) and not out and ok is True]
    wrong = [t for t, out, ok, _ in runs if isinstance(t, int) and not out and ok is False]

    def r(x):
        return int(round(x)) if x is not None else None
    return {"n": len(runs), "n_tok": len(toks), "ran_out": sum(1 for _, o, _, _ in runs if o),
            "tok_mean": r(statistics.fmean(toks)) if toks else None,
            "tok_med": r(statistics.median(toks)) if toks else None,
            "tok_p90": p90(toks) if toks else None, "tok_max": max(toks) if toks else None,
            "tok_sum": sum(toks),
            "think_mean": r(statistics.fmean(thinks)) if thinks else None,
            "think_med": r(statistics.median(thinks)) if thinks else None,
            "think_share": round(statistics.fmean(shares), 3) if shares else None,
            "right_n": len(right), "right_mean": r(statistics.fmean(right)) if right else None,
            "wrong_n": len(wrong), "wrong_mean": r(statistics.fmean(wrong)) if wrong else None}


def token_stats(d: Path, task: str, compute: bool = True) -> dict | None:
    """a row's benchmark's token stats from its scored samples, kept until
    that file changes (the next import or scoring); None when it has none,
    or (compute False) when they aren't worked out yet"""
    f = samples_file(d, task)
    if f is None:
        return None
    key = _key(f, "tokens")
    got = _cache_get(key)
    if got is not None or not compute:
        return got
    runs = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        resps = row.get("resps") or []
        for k, run in enumerate(row.get("frontier") or []):
            text = (resps[k][0] if k < len(resps) and resps[k] else "") or ""
            runs.append((run.get("tokens"), bool(run.get("ran_out")), run.get("ok"), text))
    got = {"src": "samples", **stats_of(runs)}
    _cache_put(key, got)
    return got


def token_line(rep: dict, rows: list[dict], stats: dict) -> str:
    """"step4000 thinks X× as long as Q4 on GPQA Diamond and Y× on MMLU-Pro;
    over the N benchmarks both finished, it wrote A million output tokens to
    Q4's B million." — '' until both have numbers"""
    first = rows[0] if rows else None
    ref = next((r for r in rows if r["label"] == rep.get("reference")), None)
    if not first or not ref or first is ref:
        return ""
    names = {b["task"]: b["label"] for b in rep["benchmarks"]}

    def times(task: str) -> str | None:
        a, b = (stats.get(first["label"]) or {}).get(task), (stats.get(ref["label"]) or {}).get(task)
        if not a or not b or not a.get("think_mean") or not b.get("think_mean"):
            return None
        return f"{a['think_mean'] / b['think_mean']:.1f}×"
    g, m = times("gpqa_diamond_epoch"), times("mmlupro_tiger")
    both = [t for t in names if (stats.get(first["label"]) or {}).get(t)
            and (stats.get(ref["label"]) or {}).get(t)]
    if not g or not m or not both:
        return ""
    a = sum(stats[first["label"]][t]["tok_sum"] for t in both) / 1e6
    b = sum(stats[ref["label"]][t]["tok_sum"] for t in both) / 1e6
    short = ref["label"].split(" (")[0]
    return (f"{first['label']} thinks {g} as long as {short} on {names['gpqa_diamond_epoch']} and "
            f"{m} on {names['mmlupro_tiger']}; over the {len(both)} benchmarks both finished, it "
            f"wrote {a:.1f} million output tokens to {short}'s {b:.1f} million.")


# ---------------------------------------------------------------------------
# the loop examples: the shown half only
# ---------------------------------------------------------------------------

class Hidden(ValueError):
    """a loop example from the hidden half — never shown"""


def shown_half(task: str, qid: str) -> bool:
    """the reader's rule (service/questions.half_of): the diagnose half is shown"""
    from . import questions
    return questions.half_of(task, str(qid), {}) == "diagnose"


def _run_of(f: Path, qid: str) -> dict | None:
    """a question's first run in a samples file: its answer, how it was read"""
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if f'"{qid}"' not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if str((row.get("doc") or {}).get("id")) != str(qid):
            continue
        runs, resps = row.get("frontier") or [], row.get("resps") or []
        if not runs:
            return None
        return {**runs[0], "answer": (resps[0][0] if resps and resps[0] else "") or "",
                "read": (row.get("filtered_resps") or [None])[0]}
    return None


def loop_example(task: str, qid: str, first: Path | None, against: Path | None) -> dict:
    """one example: the other model's length and how its answer was read;
    ours's length, its "Wait"s, its most repeated line, and its thinking
    shortly before it ran out"""
    if not shown_half(task, qid):
        raise Hidden(f"question {qid} is in the hidden half: the reader never shows it")
    _scripts()
    import frontier as fb
    out: dict = {"id": str(qid)}
    if against is not None:
        a = _run_of(against, qid)
        if a:
            out["against"] = {"chars": len(a["answer"]), "verdict": fb.verdict(task, a),
                              "ok": a.get("ok")}
    if first is not None:
        r = _run_of(first, qid)
        if r:
            think = fb.thinking_of(r["answer"])
            lines = [x.strip() for x in think.splitlines() if len(x.strip()) >= 8]
            top = Counter(lines).most_common(1)
            end = think[-750:-150] if len(think) > 900 else think[-600:]
            # from a line's start, or a word's
            cut = end.find("\n", 0, 200)
            end = end[cut + 1:] if cut >= 0 else end[end.find(" ") + 1:] if " " in end[:40] \
                else end
            out["first"] = {"chars": len(r["answer"]), "verdict": fb.verdict(task, r),
                            "ran_out": bool(r.get("ran_out")), "ok": r.get("ok"),
                            "waits": len(re.findall(r"\bWait\b", r["answer"])),
                            "line": top[0][0][:200] if top else "",
                            "line_n": top[0][1] if top else 0, "excerpt": end.strip()}
    return out


def loops(rep: dict, rows: list[dict], compute: bool = True) -> list[dict]:
    """the file's loop examples, worked out from the stored answers"""
    lp = rep.get("loops") or {}
    task = lp.get("task") or "mmlupro_tiger"
    by = {r["label"]: Path(r["dir"]) for r in rows}
    first = samples_file(by[lp["first"]], task) if lp.get("first") in by else None
    against = samples_file(by[lp["against"]], task) if lp.get("against") in by else None
    out = []
    for ex in lp.get("examples") or []:
        base = {"id": str(ex["id"]), "note": ex.get("note") or ""}
        if not shown_half(task, ex["id"]):
            out.append({**base, "refused": f"question {ex['id']} is in the hidden half: the "
                                           "reader never shows it"})
            continue
        key = "|".join(["loop", str(ex["id"]),
                        *(_key(f, "") if f else "-" for f in (first, against))])
        got = _cache_get(key)
        if got is None and compute:
            got = loop_example(task, ex["id"], first, against)
            _cache_put(key, got)
        out.append({**base, **(got or {"pending": True})})
    return out


def _work_on(jobs: list) -> None:
    for fn in jobs:
        try:
            fn()
        except Exception as e:                          # noqa: BLE001 — one job's, said
            print(f"[reports] a stat couldn't be worked out: {type(e).__name__}: {e}")
        _work["done"] += 1


def ensure(rep: dict, rows: list[dict]) -> dict:
    """the token stats and loop examples not worked out yet, in one thread
    of their own; {todo, done} while it runs"""
    jobs = []
    for r in rows:
        d = Path(r["dir"])
        for b in rep["benchmarks"]:
            f = samples_file(d, b["task"])
            if f is not None and _cache_get(_key(f, "tokens")) is None:
                jobs.append(lambda d=d, t=b["task"]: token_stats(d, t))
    if any(x.get("pending") for x in loops(rep, rows, compute=False)):
        jobs.append(lambda: loops(rep, rows))
    th = _work["thread"]
    if jobs and not (th and th.is_alive()):
        _work.update(todo=len(jobs), done=0)
        _work["thread"] = threading.Thread(target=_work_on, args=(jobs,), name="report-stats",
                                           daemon=True)
        _work["thread"].start()
    th = _work["thread"]
    return {"todo": _work["todo"], "done": _work["done"]} if th and th.is_alive() else {}


# ---------------------------------------------------------------------------
# the report, and the list
# ---------------------------------------------------------------------------

def _sources(rep: dict, rows: list[dict], theirs: list[dict]) -> dict:
    ep = db.reported_import_last("epoch")
    entered = [db_row_time(c.get("entered")) for t in theirs for c in t["cells"].values()
               if c.get("entered")]
    ours_at = ours_changed(rows)
    return {"ours": {"at": ours_at, "words": f"as of the last import, {when(ours_at)}"
                     if ours_at else "nothing imported yet"},
            "epoch": {"at": ep["at"] if ep else None,
                      "words": f"imported {day(ep['at'])}" if ep else "not imported yet",
                      "credit": reported.SOURCES["epoch"]["credit"],
                      "url": reported.SOURCES["epoch"]["url"]},
            "cards": {"at": max(entered) if entered else None}}


def db_row_time(rid) -> float | None:
    """when a card's entry was made"""
    from contextlib import closing
    with closing(db._conn()) as c:
        row = c.execute("SELECT created_at FROM reported_scores WHERE id=?", (rid,)).fetchone()
    return float(row[0]) if row else None


def report(rid: str, payload: dict | None = None, compute: bool = False) -> dict | None:
    """one report, every number from the board's tables. compute: work the
    token stats and loop examples out now (the tests); the page has them
    worked out in a thread of their own"""
    rep = files().get(rid)
    if rep is None:
        return None
    if payload is None:
        from . import app
        payload = app.results_payload()
    rows = our_rows(rep, payload.get("cells") or {})
    theirs = their_rows(rep)
    work = {} if compute else ensure(rep, rows)
    stats = {r["label"]: {b["task"]: token_stats(Path(r["dir"]), b["task"], compute=compute)
                          for b in rep["benchmarks"]} for r in rows}
    stats = {k: {t: s for t, s in v.items() if s} for k, v in stats.items()}
    for t in theirs:
        for c in t["cells"].values():
            if c.get("entered"):
                c["entered_at"] = day(db_row_time(c["entered"]))
    models = rows + theirs
    picks = [r for r in rows if r["pick"]]
    return {
        "id": rid, "title": rep.get("title") or rid, "what": rep.get("what") or "",
        "benchmarks": rep["benchmarks"], "ours": rows, "theirs": theirs,
        "standing": {p["label"]: standing(p, models, rep["benchmarks"]) for p in picks},
        "picks": [p["label"] for p in picks],
        "tokens": stats, "tokenLine": token_line(rep, rows, stats),
        "tokenLimits": rep.get("token_limits") or {},
        "loops": {"task": (rep.get("loops") or {}).get("task"),
                  "first": (rep.get("loops") or {}).get("first"),
                  "against": (rep.get("loops") or {}).get("against"),
                  "examples": loops(rep, rows, compute=compute)},
        "working": work, "sources": _sources(rep, rows, theirs), "notes": NOTES,
        "reference": rep.get("reference")}


def listing(payload: dict | None = None) -> list[dict]:
    """every report as a card: its title, what it is, the models in it, and
    when its numbers last changed"""
    if payload is None:
        from . import app
        payload = app.results_payload()
    out = []
    for rid, rep in files().items():
        rows = our_rows(rep, payload.get("cells") or {})
        src = _sources(rep, rows, their_rows(rep))
        last = max([x for x in (src["ours"]["at"], src["epoch"]["at"], src["cards"]["at"])
                    if x] or [0]) or None
        out.append({"id": rid, "title": rep.get("title") or rid, "what": rep.get("what") or "",
                    "ours": [r["label"] for r in rows],
                    "others": len(rep.get("theirs") or []),
                    "changed": when(last) if last else "", "changed_at": last})
    return out
