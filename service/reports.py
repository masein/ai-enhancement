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
- the thinking tokens and the loop examples, worked out from the stored
  answers by the stats worker (21c, service/report_stats.py) after each
  import or scoring, and kept: a page reads what is kept, never computing.

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
    # 23b: beyond the score, as the brief words them
    "Always right / never right. GPQA Diamond is asked 4 times a question and OTIS Mock AIME 8 "
    "times, each time with a different seed. A question the build gets right every time is one "
    "it knows; one it gets right only sometimes is a coin toss. Two builds with the same score "
    "can split very differently.",
    "Calibration error is CAIS's measure for HLE, the one Scale's HLE leaderboard shows: answers "
    "are sorted by the confidence the model states, cut into groups of 100, and in each group the "
    "stated confidence is compared with the share actually right. 0 means \"80% sure\" is right "
    "80% of the time; most models score far above 0 on HLE, because they state high confidence on "
    "questions they get wrong. The confidence is the one the judge reads from the answer; where "
    "the answer states none, the judge counts 100%.",
    "SimpleQA: tries and F. \"Not attempted\" means the model said it didn't know. Right when it "
    "tries is the share right among the answers it gave. F is the harmonic mean of the two "
    "scores, the SimpleQA paper's single number. Our prompt asks the model to always guess "
    "(Epoch AI's line), so most tries are expected.",
    "Ended in a loop uses the board's rule: the last 200 characters before the end repeat at "
    "least three times in the last 20,000. Other papers count loops differently, so compare "
    "these numbers only with each other.",
]


def _scripts() -> None:
    p = str(Path(__file__).resolve().parent.parent / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------

_files_memo: dict = {"stamp": None, "out": {}}


def files() -> dict[str, dict]:
    """every report file, by its id — one that can't be read is left out.
    21d: read again only when one of them changed (Models reads them on every
    poll)"""
    fs = sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.is_dir() else []
    stamp = []
    for f in fs:
        try:
            st = f.stat()
        except OSError:
            continue
        stamp.append((str(f), st.st_mtime_ns, st.st_size))
    if stamp == _files_memo["stamp"]:
        return {k: json.loads(json.dumps(v)) for k, v in _files_memo["out"].items()}
    out, paired = {}, {}
    for f in fs:
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rep, dict) and rep.get("id") and rep.get("kind") == "paired":
            paired[str(rep["id"])] = rep
        elif isinstance(rep, dict) and rep.get("id") and isinstance(rep.get("ours"), list):
            out[str(rep["id"])] = rep
    # 23: a paired report's builds are another report's rows, read from it
    from . import paired as pd
    for rid, rep in paired.items():
        out[rid] = pd.resolve(rep, out)
    _files_memo.update(stamp=stamp, out=out)
    return {k: json.loads(json.dumps(v)) for k, v in out.items()}


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
            parts.append("Complete" if have == len(per) else "Not scored yet" if not have
                         else f"{have} of {len(per)} benchmarks in")
        out.append({"id": "ours:" + o["label"], "label": o["label"], "row": mid, "ours": True,
                    "colour": o.get("colour"), "pick": bool(o.get("pick")),
                    # 22: on Models ▸ Latest, as a published row marked "glance" is
                    "glance": o.get("glance") is True,
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
                            "entered": c.get("id"),
                            # 21d: what the file says of the number ("medium effort")
                            "note": spec.get("note") or ""}
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
                    "cells": per,
                    # 21d: where its numbers come from, in words; and whether
                    # Models ▸ Latest sets it beside ours
                    "source": source_words([b for b in rep["benchmarks"] if b.get("compare")],
                                           per),
                    "glance": t.get("glance") is True})
    return out


def _card_word(card: str) -> str:
    """"DeepSeek-V4-Pro model card" -> "DeepSeek"; "Qwen3.5-35B-A3B model card" -> "Qwen" """
    m = re.match(r"[A-Za-z]+", card or "")
    return m.group(0) if m else (card or "model")


def source_words(benches: list[dict], cells: dict) -> str:
    """where a model's numbers come from, in words: "Epoch AI", "Epoch AI;
    HLE, MMLU-Pro: DeepSeek card", "Qwen's model card" when every one is a
    card's — and the file's note on a card's number, in brackets after its
    card: MMLU-Pro: Qwen card (medium effort)"""
    cards: dict[str, list[str]] = {}
    notes: dict[str, list[str]] = {}
    epoch = False
    for b in benches:
        c = cells.get(b["key"]) or {}
        if c.get("v") is None:
            continue
        if c.get("src") == "card":
            w = _card_word(c.get("card") or "")
            cards.setdefault(w, []).append(b["label"])
            if c.get("note") and c["note"] not in notes.get(w, []):
                notes.setdefault(w, []).append(c["note"])
        else:
            epoch = True
    if not cards:
        return "Epoch AI"
    said = {w: f"{w} card" + (f" ({', '.join(notes[w])})" if notes.get(w) else "")
            for w in cards}
    if not epoch and len(cards) == 1:
        w = next(iter(cards))
        return f"{w}'s model card" + (f" ({', '.join(notes[w])})" if notes.get(w) else "")
    line = "; ".join(f"{', '.join(labels)}: {said[w]}" for w, labels in cards.items())
    return f"Epoch AI; {line}" if epoch else line


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

def fixed(x: float, digits: int = 1) -> float:
    """x as JavaScript's toFixed rounds it, the way the pages show it: the
    exact binary value, a tie away from zero (Python's round takes a tie to
    even)"""
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def points(x: float | None) -> float | None:
    """a share in points, as shown: 0.74213 -> 74.2"""
    return None if x is None else fixed(100 * x)


def band(se_a: float | None, se_b: float | None) -> float:
    """"level": a gap no more than 2·√(se_ours² + se_theirs²) — a number
    without a standard error (a card's) counts it as 0"""
    return 2 * math.sqrt((se_a or 0) ** 2 + (se_b or 0) ** 2)


def verdict(ours: dict, theirs: dict) -> str | None:
    """"higher", "lower" or "level" — ours against theirs; None without both.
    21c: worked out from the numbers as shown, in points, as the published
    page works it out — the gap is the difference of the two shown scores"""
    if ours.get("v") is None or theirs.get("v") is None:
        return None
    a, t = points(ours["v"]), points(theirs["v"])
    if abs(a - t) <= 2 * math.sqrt(points(ours.get("se") or 0) ** 2
                                   + points(theirs.get("se") or 0) ** 2):
        return "level"
    return "higher" if a > t else "lower"


def gap(ours: dict, theirs: dict) -> float:
    """ours minus theirs, as a share, from the shown numbers to a tenth of a
    point (JavaScript's Math.round, as the published page)"""
    return math.floor((points(ours["v"]) - points(theirs["v"])) * 10 + 0.5) / 10 / 100


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
            diffs[b["key"]] = {"d": gap(mine, c), "state": v, "ours": mine["v"],
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
# thinking tokens, from the stored answers — 21c: worked out by the stats
# worker (service/report_stats.py) and kept; a page reads what is kept
# ---------------------------------------------------------------------------

def samples_file(d: Path, task: str) -> Path | None:
    from . import frontier as sf
    got = sorted(sf.task_dir(d, task).glob(f"samples_{task}_*.jsonl"))
    return got[-1] if got else None


def jsonl(f: Path, needle: bytes = b""):
    """a JSON-lines file a row at a time, with the byte where each starts —
    read as it goes, never whole (21c: a benchmark's answers run to hundreds
    of megabytes); `needle`: only the lines holding it are parsed"""
    at = 0
    with open(f, "rb") as fh:
        for raw in fh:
            start, at = at, at + len(raw)
            if needle and needle not in raw:
                continue
            try:
                row = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(row, dict):
                yield start, row


def row_at(f: Path, offset: int) -> dict | None:
    """the row starting at `offset`"""
    try:
        with open(f, "rb") as fh:
            fh.seek(offset)
            row = json.loads(fh.readline().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) else None


# 21d: the token stats are ~/tokens.py's, field by field (masein's script,
# tests/fixtures/tokens_py/tokens.py): 21c's differed from it four ways —
# think_share was each answer's share averaged, not the thinking characters
# over all characters; an answer's share counted its <think> tags; a median
# was the two middle values' mean and p90 the nearest rank, not the value at
# int(p·n) of the sorted list; and an answer's thinking tokens were averaged
# before rounding, not rounded one by one. Where a row's benchmark has no
# scored samples yet, its answers are read, as tokens.py reads them

def quantile(v: list[int], p: float) -> int | None:
    """tokens.py's q(): the value at int(p·n) of the sorted values (the upper
    middle one for a median of an even count)"""
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def token_stats_from(recs: list[tuple[str, object, object, object]]) -> dict:
    """tokens.py's stats of (answer, output tokens, finish, right) a run each:
    an answer ran out when it stopped at the limit or its thinking never
    closed; its thinking tokens are its output tokens times its thinking's
    share of its characters (the tags left out), rounded; right and wrong
    leave out the answers that ran out"""
    _scripts()
    from frontier_graders import think_split
    tok, think, right, wrong, out_n = [], [], [], [], 0
    tc = ac = 0
    for text, n, fin, ok in recs:
        outside, thoughts, unclosed = think_split(text or "")
        th, an = sum(len(x) for x in thoughts), len(outside)
        cut = fin == "length" or unclosed
        out_n += cut
        if not isinstance(n, int):
            continue
        tc += th
        ac += an
        tok.append(n)
        think.append(round(n * th / (th + an)) if th + an else 0)
        if not cut and ok is True:
            right.append(n)
        elif not cut and ok is False:
            wrong.append(n)

    def mean(v):
        return round(statistics.mean(v)) if v else None
    return {"n": len(recs), "n_tok": len(tok), "ran_out": out_n, "tok_mean": mean(tok),
            "tok_med": quantile(tok, .5), "tok_p90": quantile(tok, .9),
            "tok_max": max(tok) if tok else None, "tok_sum": sum(tok), "think_mean": mean(think),
            "think_med": quantile(think, .5),
            "think_share": round(tc / (tc + ac), 3) if tc + ac else None,
            "right_n": len(right), "right_mean": mean(right),
            "wrong_n": len(wrong), "wrong_mean": mean(wrong)}


def token_stats_of(f: Path) -> dict:
    """a samples file's token stats, read a row at a time; an answers file's
    (a benchmark not scored yet) when that is what `f` is"""
    from . import frontier as sf
    recs = []
    if f.name == sf.ANSWERS:
        for _, r in jsonl(f):
            if not r.get("error"):
                recs.append((r.get("answer") or "", r.get("tokens"), r.get("finish"), None))
        return {"src": "answers", **token_stats_from(recs)}
    for _, row in jsonl(f):
        for resp, r in zip(row.get("resps") or [], row.get("frontier") or []):
            recs.append(((resp[0] if resp else "") or "", r.get("tokens"), r.get("finish"),
                         r.get("ok")))
    return {"src": "samples", **token_stats_from(recs)}


def stats_file(d: Path, task: str) -> Path | None:
    """what a row's benchmark's token stats read: its scored samples, else
    (all its answers in, not scored yet) its answers"""
    from . import frontier as sf
    f = samples_file(d, task)
    if f is not None:
        return f
    a = sf.task_dir(d, task) / sf.ANSWERS
    return a if a.is_file() else None


def _tok_slot(d: Path, task: str) -> str:
    return f"tokens|{d}|{task}"


# 21d: the way of counting is part of what a kept stat was worked out from:
# stats kept under 21c's counting are stale, and worked out again
TOKENS_WAY = "tokens.py"


def _tok_key(f: Path) -> str:
    from . import report_stats as rs
    return f"{TOKENS_WAY}|{rs.identity(f)}"


def token_kept(d: Path, task: str) -> dict | None:
    """{stats, at, fresh} as kept — `fresh` False when the answers changed
    since (shown until the new ones are in); None when never worked out"""
    from . import report_stats as rs
    f = stats_file(d, task)
    if f is None:
        return None
    e = rs.kept(_tok_slot(d, task))
    if not e or "value" not in e:
        return None
    return {"stats": e["value"], "at": e["at"], "fresh": e["key"] == _tok_key(f)}


def token_stats(d: Path, task: str, compute: bool = True) -> dict | None:
    """a row's benchmark's token stats from its stored answers: compute, work
    them out now unless kept for these very answers (the worker, the tests,
    scripts/report_token_stats.py); else what is kept. None when the row has
    no stored answers, or (compute False) nothing is kept"""
    from . import report_stats as rs
    f = stats_file(d, task)
    if f is None:
        return None
    if not compute:
        k = token_kept(d, task)
        return k["stats"] if k else None
    key, slot = _tok_key(f), _tok_slot(d, task)
    e = rs.kept(slot)
    if e and e.get("key") == key and "value" in e:
        return e["value"]
    got = token_stats_of(f)
    rs.keep(slot, key, got)
    return got


def token_line(rep: dict, rows: list[dict], stats: dict) -> str:
    """"step4000 thinks X× as long as Q4 on GPQA Diamond and Y× on MMLU-Pro;
    over the N benchmarks both finished, it wrote A million output tokens to
    Q4's B million." — '' until both have numbers"""
    ref = next((r for r in rows if r["label"] == rep.get("reference")), None)
    # 23b: the first row with thinking stats on both — step8804, first, once
    # it has them; step4000 until then
    first = next((r for r in rows if r is not ref and all(
        ((stats.get(r["label"]) or {}).get(t) or {}).get("think_mean")
        for t in ("gpqa_diamond_epoch", "mmlupro_tiger"))), None)
    if not first or not ref:
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


def _first_run(row: dict) -> dict | None:
    runs, resps = row.get("frontier") or [], row.get("resps") or []
    if not runs:
        return None
    return {**runs[0], "answer": (resps[0][0] if resps and resps[0] else "") or "",
            "read": (row.get("filtered_resps") or [None])[0]}


def _runs_by_id(f: Path | None, ids: list[str]) -> dict[str, dict]:
    """each question's first run in a samples file, for these questions: its
    answer, how it was read — the file read once"""
    out: dict[str, dict] = {}
    if f is None or not ids:
        return out
    want = set(ids)
    for _, row in jsonl(f):
        qid = str((row.get("doc") or {}).get("id"))
        if qid in want and qid not in out:
            r = _first_run(row)
            if r is not None:
                out[qid] = r
    return out


def _run_of(f: Path, qid: str) -> dict | None:
    """a question's first run in a samples file"""
    for _, row in jsonl(f, f'"{qid}"'.encode()):
        if str((row.get("doc") or {}).get("id")) == str(qid):
            return _first_run(row)
    return None


def _example(task: str, qid: str, r: dict | None, a: dict | None,
             item: dict | None = None) -> dict:
    """one example from its two runs: the other model's length and how its
    answer was read; ours's length, its "Wait"s, its most repeated line, and
    its thinking shortly before it ran out — and (21c) the question as the
    board shows it, for the full page"""
    _scripts()
    import frontier as fb
    out: dict = {"id": str(qid)}
    # 21d.1: how long each thought — its thinking alone, as the page's "thought
    # for N characters" says (Q4 on 486: 6,485, of 7,247 whole); its whole
    # output beside it
    if a:
        out["against"] = {"chars": len(a["answer"]), "thought": len(fb.thinking_of(a["answer"])),
                          "verdict": fb.verdict(task, a), "ok": a.get("ok")}
    if r:
        think = fb.thinking_of(r["answer"])
        lines = [x.strip() for x in think.splitlines() if len(x.strip()) >= 8]
        top = Counter(lines).most_common(1)
        end = think[-750:-150] if len(think) > 900 else think[-600:]
        # from a line's start, or a word's
        cut = end.find("\n", 0, 200)
        end = end[cut + 1:] if cut >= 0 else end[end.find(" ") + 1:] if " " in end[:40] \
            else end
        out["first"] = {"chars": len(r["answer"]), "thought": len(think),
                        "verdict": fb.verdict(task, r),
                        "ran_out": bool(r.get("ran_out")), "ok": r.get("ok"),
                        "waits": len(re.findall(r"\bWait\b", r["answer"])),
                        "line": top[0][0][:200] if top else "",
                        "line_n": top[0][1] if top else 0, "excerpt": end.strip()}
    if item:
        from . import ran_out
        gone = fb.unlisted(task)
        right = {} if gone else ran_out.right_of(task, item)
        out["item"] = {"subject": ran_out._subject(task, item),
                       "question": gone or str(item.get("question") or ""),
                       "right": right.get("letter") or right.get("answer") or "",
                       "right_text": right.get("text") or "", "withheld": gone}
        if gone and out.get("first"):
            out["first"]["excerpt"] = ran_out.scrub(out["first"]["excerpt"], item)
            out["first"]["line"] = ran_out.scrub(out["first"]["line"], item)
    return out


def loop_example(task: str, qid: str, first: Path | None, against: Path | None) -> dict:
    """one example, worked out from the two samples files"""
    if not shown_half(task, qid):
        raise Hidden(f"question {qid} is in the hidden half: the reader never shows it")
    return _example(task, qid, _run_of(first, qid) if first is not None else None,
                    _run_of(against, qid) if against is not None else None)


def _loop_files(rep: dict) -> tuple[str, Path | None, Path | None]:
    lp = rep.get("loops") or {}
    task = lp.get("task") or "mmlupro_tiger"
    by = {o["label"]: row_dir(o["row"]) for o in rep["ours"]}
    first = samples_file(by[lp["first"]], task) if lp.get("first") in by else None
    against = samples_file(by[lp["against"]], task) if lp.get("against") in by else None
    return task, first, against


def items_file(task: str) -> Path:
    """where this machine keeps a benchmark's questions"""
    _scripts()
    import frontier as fb
    return fb._cache(config.BENCH_ROOT, task)


def _loops_key(rep: dict) -> str:
    """the files the loop examples read: the two models' answers and the
    questions"""
    from . import report_stats as rs
    task, first, against = _loop_files(rep)
    ids = [str(ex["id"]) for ex in (rep.get("loops") or {}).get("examples") or []]
    # 21d.1: "thought" — examples kept before they counted the thinking alone
    # are worked out again
    return "|".join(["thought", task, rs.identity(first), rs.identity(against),
                     rs.identity(items_file(task)), ",".join(ids)])


def loops_of(rep: dict) -> list[dict]:
    """the file's loop examples, worked out: each samples file read once, the
    hidden half refused"""
    _scripts()
    import frontier as fb
    task, first, against = _loop_files(rep)
    ids = [str(ex["id"]) for ex in (rep.get("loops") or {}).get("examples") or []]
    shown = [i for i in ids if shown_half(task, i)]
    mine, theirs = _runs_by_id(first, shown), _runs_by_id(against, shown)
    items = {str(it["id"]): it for it in (fb.cached(task, config.BENCH_ROOT) or [])
             if str(it.get("id")) in set(shown)} if shown else {}
    return [_example(task, i, mine.get(i), theirs.get(i), items.get(i)) if i in shown
            else {"id": i, "refused": True} for i in ids]


def loops(rep: dict, rows: list[dict] | None = None, compute: bool = True) -> list[dict]:
    """the file's loop examples: worked out now (compute), else as kept — a
    question from the hidden half refused either way"""
    from . import report_stats as rs
    lp = rep.get("loops") or {}
    task = lp.get("task") or "mmlupro_tiger"
    slot = "loops|" + str(rep.get("id"))
    got = None
    if compute:
        key = _loops_key(rep)
        e = rs.kept(slot)
        got = e["value"] if e and e.get("key") == key and "value" in e else None
        if got is None:
            got = loops_of(rep)
            rs.keep(slot, key, got)
    else:
        e = rs.kept(slot)
        got = e["value"] if e and "value" in e else None
    by = {str(x.get("id")): x for x in got or []}
    out = []
    for ex in lp.get("examples") or []:
        base = {"id": str(ex["id"]), "note": ex.get("note") or ""}
        if not shown_half(task, ex["id"]):
            out.append({**base, "refused": f"question {ex['id']} is in the hidden half: the "
                                           "reader never shows it"})
            continue
        x = by.get(str(ex["id"]))
        out.append({**base, **x} if x and not x.get("refused") else {**base, "pending": True})
    return out


def jobs(rid: str, rep: dict) -> list:
    """21c: a scorecard's stats for the worker — each row's benchmark's token
    stats, and the loop examples"""
    from . import report_stats as rs
    out = []
    for o in rep["ours"]:
        d = row_dir(o["row"])
        for b in rep["benchmarks"]:
            f = stats_file(d, b["task"])
            if f is not None:
                out.append(rs.Job(_tok_slot(d, b["task"]), _tok_key(f),
                                  f"thinking tokens, {o['label']} · {b['label']}",
                                  lambda f=f: token_stats_of(f)))
    # the loop examples, once either model has answers stored
    if (rep.get("loops") or {}).get("examples") and any(_loop_files(rep)[1:]):
        out.append(rs.Job("loops|" + rid, _loops_key(rep), f"loop examples, {rid}",
                          lambda: loops_of(rep)))
    # 23b: and its rows' measures beyond the score
    from . import beyond
    out += beyond.jobs([row_dir(o["row"]) for o in rep["ours"]])
    return out


# ---------------------------------------------------------------------------
# the report, and the list
# ---------------------------------------------------------------------------

def _sources(rep: dict, rows: list[dict], theirs: list[dict]) -> dict:
    ep = db.reported_import_last("epoch")
    times = entered_times([c.get("entered") for t in theirs for c in t["cells"].values()
                           if c.get("entered")])
    entered = list(times.values())
    ours_at = ours_changed(rows)
    return {"ours": {"at": ours_at, "words": f"as of the last import, {when(ours_at)}"
                     if ours_at else "nothing imported yet", "when": when(ours_at)},
            "epoch": {"at": ep["at"] if ep else None,
                      "words": f"imported {day(ep['at'])}" if ep else "not imported yet",
                      "credit": reported.SOURCES["epoch"]["credit"],
                      "url": reported.SOURCES["epoch"]["url"]},
            "cards": {"at": max(entered) if entered else None}}


def db_row_time(rid) -> float | None:
    """when a card's entry was made"""
    return entered_times([rid]).get(rid)


def entered_times(ids: list) -> dict:
    """21d: when each card entry was made, in one read"""
    from contextlib import closing
    ids = [i for i in ids if i is not None]
    if not ids:
        return {}
    with closing(db._conn()) as c:
        rows = c.execute("SELECT id, created_at FROM reported_scores WHERE id IN (%s)"
                         % ",".join("?" * len(ids)), ids).fetchall()
    return {r[0]: float(r[1]) for r in rows if r[1] is not None}


def bench_meta(benches: list[dict]) -> list[dict]:
    """21d: each benchmark with how many times each of its questions is asked
    and what a question is called: a score line says 198 questions × 4, never
    792 answers"""
    _scripts()
    import frontier as fb
    return [{**b, "epochs": int((fb.BENCH.get(b["task"]) or {}).get("epochs") or 1),
             "unit": "tasks" if b["task"] == "arc_agi2_public" else "questions"} for b in benches]


def report(rid: str, payload: dict | None = None, compute: bool = False, a: str = "",
           b: str = "") -> dict | None:
    """one report, every number from the board's tables. compute: work the
    token stats and loop examples out now (the tests, the scripts); a page
    reads what the stats worker kept (21c), never computing. 23: a paired
    report for the builds a and b (its default pair when not given)"""
    rep = files().get(rid)
    if rep is None:
        return None
    if payload is None:
        from . import app
        payload = app.results_payload()
    rows = our_rows(rep, payload.get("cells") or {})
    if rep.get("kind") == "paired":
        # 23: two builds, question by question (service/paired.py)
        from . import paired as pd
        got = pd.report(rep, rows, a, b, compute=compute)
        src = ours_changed(rows)
        return {**got, "sources": {"ours": {"at": src, "words": f"as of the last import, "
                                                                 f"{when(src)}"
                                            if src else "nothing imported yet"}}}
    if rep.get("kind") == "ran_out":
        # 21b: the answers our models ran out of room on (service/ran_out.py),
        # each benchmark's count beside the board's own
        from . import ran_out
        got = ran_out.report(rep, compute=compute)
        board = {r["label"]: {b["key"]: r["cells"][b["key"]].get("ranOut")
                              for b in rep["benchmarks"]} for r in rows}
        for lbl, m in got["models"].items():
            for x in m["summary"]:
                x["board"] = board.get(lbl, {}).get(x["key"])
        src = ours_changed(rows)
        return {**got, "benchmarks": bench_meta(got["benchmarks"]),
                "sources": {"ours": {"at": src, "words": f"as of the last import, "
                                     f"{when(src)}" if src else "nothing imported yet"}}}
    theirs = their_rows(rep)
    if compute:
        for r in rows:
            for b in rep["benchmarks"]:
                token_stats(Path(r["dir"]), b["task"])
        loops(rep, rows)
    stats, kept_at = kept_stats(rep, rows)
    times = entered_times([c.get("entered") for t in theirs for c in t["cells"].values()
                           if c.get("entered")])
    for t in theirs:
        for c in t["cells"].values():
            if c.get("entered"):
                c["entered_at"] = day(times.get(c["entered"]))
    models = rows + theirs
    picks = [r for r in rows if r["pick"]]
    return {
        "id": rid, "title": rep.get("title") or rid, "what": rep.get("what") or "",
        "benchmarks": bench_meta(rep["benchmarks"]), "ours": rows, "theirs": theirs,
        # 21d: Models ▸ Latest — its benchmarks (the models are the rows,
        # ours and the others, marked "glance" — 22: ours too, where it was
        # every picked row)
        "glance": {"benchmarks": [k for k in rep.get("glance_benchmarks") or []
                                  if any(b["key"] == k for b in rep["benchmarks"])]},
        "standing": {p["label"]: standing(p, models, rep["benchmarks"]) for p in picks},
        "picks": [p["label"] for p in picks],
        "tokens": stats, "tokenLine": token_line(rep, rows, stats),
        # 21d: what each token column means, as the published page says it
        "tokenDefs": page_defs(rep),
        "tokenLimits": rep.get("token_limits") or {},
        "loops": {"task": (rep.get("loops") or {}).get("task"),
                  "first": (rep.get("loops") or {}).get("first"),
                  "against": (rep.get("loops") or {}).get("against"),
                  "examples": loops(rep, rows, compute=False)},
        "working": {} if compute else stats_working(rid, rep),
        "statsAt": stats_at(kept_at),
        "sources": _sources(rep, rows, theirs), "notes": NOTES,
        "reference": rep.get("reference"),
        # 21b: where a model's ran-out answers are listed, report by report
        "ranOut": ran_out_links(),
        # 23b: beyond the score — each of our rows' measures as kept
        "beyond": beyond_of(rows, rep["benchmarks"], compute),
        # 23: the report comparing two of these rows question by question
        "paired": next((x for x, r in files().items() if r.get("kind") == "paired"
                        and (r.get("builds") or {}).get("from") == rid), None),
        # 21c: the page as masein publishes it, served here
        "page": page_url(rid, rep)}


def beyond_of(rows: list[dict], benches: list[dict], compute: bool = False) -> dict:
    """23b: {label: {benchmark key: measures}} as the stats worker kept them
    (service/beyond.py); worked out now with `compute`"""
    from . import beyond
    out: dict = {}
    for r in rows:
        d = Path(r["dir"])
        for b in benches:
            x = beyond.kept(d, b["task"], compute)
            if x:
                out.setdefault(r["label"], {})[b["key"]] = {
                    k: v for k, v in x.items() if k in ("loops", "reliability", "calibration",
                                                        "simpleqa", "fresh", "at")}
    return out


def page_defs(rep: dict) -> list[list[str]]:
    """21d: the published page's definitions under its token table — [[term,
    definition], …] from its <dl class="sc-defs">, word for word; [] without
    a page or the list"""
    import html as _html

    from . import report_page
    pg = report_page.page_of(rep)
    if not pg:
        return []
    text = pg["file"].read_text(encoding="utf-8")
    m = re.search(r'<dl class="sc-defs">(.*?)</dl>', text, re.S)
    if not m:
        return []
    plain = lambda x: _html.unescape(re.sub(r"<[^>]+>", "", x)).strip()      # noqa: E731
    return [[plain(dt), plain(dd)] for dt, dd in
            re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", m.group(1), re.S)]


def page_url(rid: str, rep: dict) -> str:
    """21c: where a report's full page is, relative to the board — '' when it
    has none"""
    from urllib.parse import quote

    from . import report_page
    return f"reports/{quote(rid)}/page" if report_page.page_of(rep) else ""


def kept_stats(rep: dict, rows: list[dict]) -> tuple[dict, list[float]]:
    """21c: the token stats as kept, a row's benchmark each — {label: {task:
    stats}} — and when each was worked out (the loop examples' too)"""
    from . import report_stats as rs
    stats: dict = {}
    ats: list[float] = []
    for r in rows:
        for b in rep["benchmarks"]:
            k = token_kept(Path(r["dir"]), b["task"])
            if k:
                stats.setdefault(r["label"], {})[b["task"]] = k["stats"]
                ats.append(k["at"])
        stats.setdefault(r["label"], {})
    e = rs.kept("loops|" + str(rep.get("id")))
    if e and "value" in e:
        ats.append(e["at"])
    return stats, ats


def stats_working(rid: str, rep: dict) -> dict:
    """21c: how far the stats worker is with this report's — {} when every
    one is worked out for the answers stored now"""
    from . import report_stats as rs
    return rs.working(jobs(rid, rep))


def stats_at(ats: list[float]) -> dict:
    """when the kept stats were worked out: the newest, in words"""
    if not ats:
        return {"at": None, "words": ""}
    return {"at": max(ats), "words": f"worked out {when(max(ats))}"}


def glance_meta() -> dict | None:
    """21d: what Models ▸ Latest reads — the first report naming its glance
    benchmarks: its id and title, our rows it marks "glance" (board model
    ids; 22: where it was every picked row) and those benchmarks' tasks. None
    when no report names any"""
    for rid, rep in files().items():
        keys = rep.get("glance_benchmarks") or []
        if rep.get("kind") == "ran_out" or not keys:
            continue
        task = {b["key"]: b["task"] for b in rep["benchmarks"]}
        return {"report": rid, "title": rep.get("title") or rid,
                "rows": [o["row"] for o in rep["ours"] if o.get("glance") is True],
                "tasks": [task[k] for k in keys if k in task]}
    return None


def ran_out_links() -> list[dict]:
    """21b: each ran-out report and the models it lists — the scorecard's
    ran-out chart and loop examples link to it"""
    return [{"id": rid, "models": [o["label"] for o in rep["ours"] if o.get("pick")]}
            for rid, rep in files().items() if rep.get("kind") == "ran_out"]


def listing(payload: dict | None = None) -> list[dict]:
    """every report as a card: its title, what it is, the models in it, and
    when its numbers last changed. 21d: never the board's whole payload — a
    card needs the rows' names and folders, not their numbers (2.3 s on the
    server whenever that payload was being built again)"""
    out = []
    for rid, rep in files().items():
        rows = our_rows(rep, (payload or {}).get("cells") or {})
        if rep.get("kind") == "ran_out":
            last = ours_changed(rows)
            out.append({"id": rid, "title": rep.get("title") or rid, "what": rep.get("what") or "",
                        "ours": [r["label"] for r in rows if r["pick"]], "others": 0,
                        "against": rep.get("against"), "insight": True,
                        "changed": when(last) if last else "", "changed_at": last})
            continue
        if rep.get("kind") == "paired":
            # 23: every build of the file, compared with itself only
            last = ours_changed(rows)
            out.append({"id": rid, "title": rep.get("title") or rid, "what": rep.get("what") or "",
                        "ours": [r["label"] for r in rows], "others": 0, "paired": True,
                        "changed": when(last) if last else "", "changed_at": last})
            continue
        src = _sources(rep, rows, their_rows(rep))
        last = max([x for x in (src["ours"]["at"], src["epoch"]["at"], src["cards"]["at"])
                    if x] or [0]) or None
        out.append({"id": rid, "title": rep.get("title") or rid, "what": rep.get("what") or "",
                    "ours": [r["label"] for r in rows],
                    "others": len(rep.get("theirs") or []),
                    "changed": when(last) if last else "", "changed_at": last,
                    "page": page_url(rid, rep)})
    return out
