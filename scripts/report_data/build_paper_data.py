#!/usr/bin/env python3
"""paper-data/ for the report's public data repository, built from the paper itself.

    python3 build_paper_data.py <out folder> <paper.md> [--figures <dir>] [--speeds <csv>] [--trial]

Every table in the paper becomes table-<its number>.csv (the untitled table under Key
numbers, key-numbers.csv) and every figure with data figure-<its number>.csv. The numbers
come from the paper's own tables and captions, and each file takes its name from the
paper's own numbering, so a round that renumbers tables, or changes a value in a table,
only needs this run again.

Some figures need what the paper doesn't print: Epoch AI's release dates (the timeline),
which models a chart names or groups (the GPQA Diamond lanes and the 21-row grid), the
on-device chart's points, and the writing speeds of the DeviceMark-against-speed chart.
They read them from --figures (default: round13-figures beside this paper) and, for the
speeds, from --speeds (default: figures-r19/figure-1.csv beside this paper, which the
figure script writes). Every score taken from there is checked against the paper's
tables: a score the paper doesn't print the same way stops the build. Figures are told
apart by phrases of their captions, not by their numbers, so a round that renumbers the
figures needs no change here; a figure this script doesn't know stops it.

Cells keep the paper's values as printed, with thousands separators dropped from
numbers. A "76.1 ± 2.3" cell becomes a value and an SE column, "[79.7, 86.0]" a low and a
high, "15.4 ± 2.1 [16.5%]" also a share, and "655 / 130" under a "right / wrong" heading
two columns. Numbers we didn't measure are marked: a Source column where rows differ,
the column heading where columns do. Every number written is checked against the row it
came from. index.csv lists every file with the caption it comes from.

Any TBD left in the paper (from its title on) stops the build and writes nothing.
--trial lists them and goes on; a TBD is never changed into anything else, and the run
lists each file that carries one."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EPOCH = "Epoch AI Benchmarking Hub, CC BY 4.0"
PIPETTE = ("Pipette (Liquid AI), Galaxy S26 Ultra leaderboard, data as of 5 Nov 2026. Copyright 2026 "
           "Liquid AI, Inc. Pipette Data License v1.0, https://pipette.liquid.ai/data-license. "
           "Provided as-is, without warranties. Values reproduced unmodified.")
SPEED_HEAD = ["model", "mode", "devicemark_composite", "ci95_low", "ci95_high", "score_source", "speed_tok_s",
              "speed_source", "speed_credit"]
OURS = "measured by Teraformer"
DEVICEMARK = "DeviceMark (devicemark.github.io), CC BY 4.0, published"
MAKER = "published by the model's maker or NVIDIA, as the row says"


# ---------------------------------------------------------------------------
# reading the paper
# ---------------------------------------------------------------------------

def published_text(paper: str) -> list[str]:
    """the paper from its title on: the notes before it are for the authors"""
    t = paper.split("\n")
    a = next(i for i, x in enumerate(t) if x.startswith("## Abstract"))
    return t[max(i for i in range(a) if t[i].startswith("# ")):]


def tbds(lines: list[str]) -> list[str]:
    return [f"{x.strip()[:150]}" for x in lines if "TBD" in x]


def raw_cells(row: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]


def clean(c: str) -> str:
    c = c.strip()
    for a, b in (("**", ""), ("\\[", "["), ("\\]", "]"), ("\\_", "_"), ("\\|", "|"),
                 ("&#91;", "[")):
        c = c.replace(a, b)
    return re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"\1", c).strip()


def italic(c: str) -> bool:
    """a cell set in italics as a whole: not measured by us"""
    c = c.strip()
    return bool(re.fullmatch(r"\*[^*].*\*", c)) and not c.startswith("**")


def tables(lines: list[str]) -> list[dict]:
    """each table with its caption: {name, part, caption, head, rows (raw cells)}"""
    out, cap, name, part, i = [], "", None, 0, 0
    while i < len(lines):
        m = re.match(r"\*\*(Table [A-Z]?\d+[a-z]?)\.\*\* ?(.*)", lines[i])
        if m:
            name, cap, part = m.group(1), m.group(2), 0
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [raw_cells(r) for r in lines[i:j]]
            part += 1
            out.append({"name": name or "Key numbers", "part": part if name else 1,
                        "caption": cap, "head": [clean(c) for c in rows[0]], "rows": rows[2:]})
            i = j
            continue
        if lines[i].startswith("#"):
            name, cap, part = None, "", 0
        i += 1
    return out


def figures(lines: list[str]) -> list[tuple[str, str]]:
    out = []
    for x in lines:
        m = re.search(r"embedded content: Figure (\d+)\. (.*?)\\?\]?$", x)
        if m:
            out.append((m.group(1), clean(m.group(2))))
    return out


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

NUM = r"[+−-]?[\d,]*\.?\d+"
PM = re.compile(rf"^({NUM}) ± ({NUM})(?: [\[(]({NUM})%[\])])?$")
RANGE = re.compile(rf"^({NUM}) \[({NUM}), ({NUM})\]$")
PAIR = re.compile(rf"^({NUM}) / ({NUM})$")
GAP = re.compile(rf"^({NUM}) \(({NUM}) to ({NUM})\)$")
VERDICT = re.compile(rf"^(level|higher|lower), ({NUM}) \(({NUM}) to ({NUM})\)((?: [†‡])*)$")


def num(x: str) -> str:
    """a number as printed, without its thousands separators and with a plain minus"""
    return x.replace(",", "").replace("−", "-")


def is_num(x: str) -> bool:
    return bool(re.fullmatch(NUM, x.strip()))


def split_columns(head: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    outh, cols = [], []
    for k, h in enumerate(head):
        col = [r[k] if k < len(r) else "" for r in rows]
        pm = [PM.match(c) for c in col]
        rng = [RANGE.match(c) for c in col]
        pair = [PAIR.match(c) for c in col]
        two = re.match(r"^(.*?)(\w[\w ]*) / (\w[\w ]*)$", h)
        if any(pm):
            outh += [h, f"{h}, SE"]
            cols += [[num(m.group(1)) if m else c for m, c in zip(pm, col)],
                     [num(m.group(2)) if m else "" for m in pm]]
            if any(m and m.group(3) for m in pm):
                outh.append(f"{h}, share that fit (%)")
                cols.append([num(m.group(3)) if m and m.group(3) else "" for m in pm])
        elif any(rng):
            h = re.sub(r" \[95% range\]$", "", h)
            outh += [h, f"{h}, 95% low", f"{h}, 95% high"]
            cols += [[num(m.group(1)) if m else c for m, c in zip(rng, col)],
                     [num(m.group(2)) if m else "" for m in rng],
                     [num(m.group(3)) if m else "" for m in rng]]
        elif two and all(m or not c or c == "–" for m, c in zip(pair, col)) and any(pair):
            outh += [f"{two.group(1)}{two.group(2)}", f"{two.group(1)}{two.group(3)}"]
            cols += [[num(m.group(1)) if m else c for m, c in zip(pair, col)],
                     [num(m.group(2)) if m else "" for m in pair]]
        else:
            outh.append(h)
            cols.append([num(c) if is_num(c) else c for c in col])
    return outh, [list(r) for r in zip(*cols)]


def credit(caption: str) -> str:
    """whose numbers a table's italic cells are, from its caption"""
    if "DeviceMark's published" in caption:
        return DEVICEMARK
    if "as published" in caption:
        return MAKER
    if "Epoch" in caption:
        return EPOCH
    return "not measured by us (italic in the report)"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

EPOCH_LIST = ["Model (setting)", "Epoch score", "Verdict", "Gap (95% range)"]
MODES = re.compile(r"^(.*), thinking (on / off|off / on|on|off)$")


def epoch_list(t: dict) -> tuple[list[str], list[list[str]]]:
    """Tables E1 and F1 to F3: one model a row, its † and ‡ apart, the gap as numbers"""
    h = ["Model (setting)", "Below high effort (†)", "Score (%)", "SE", "Verdict",
         "Within a quarter of a combined SE of the line (‡)", "Gap, Masein model minus model",
         "Gap, 95% low", "Gap, 95% high", "Source"]
    out = []
    for raw in t["rows"]:
        label, score, verdict, gap = (clean(c) for c in raw)
        m, g = PM.match(score), GAP.match(gap)
        gaps = [num(x) for x in g.groups()] if g else [gap, "", ""]
        out.append([label.replace(" †", ""), "yes" if "†" in label else "",
                    num(m.group(1)) if m else num(score) if is_num(score) else score,
                    num(m.group(2)) if m else "", verdict.replace(" ‡", ""),
                    "yes" if "‡" in verdict else "", *gaps,
                    EPOCH if italic(raw[1]) else OURS])
    return h, out


def verdict_grid(t: dict, cols: list[int]) -> tuple[list[str], list[list[str]]]:
    """a table whose cells read "level, −2.1 (−8.8 to +4.5) †": each as five columns"""
    h = []
    for k, x in enumerate(t["head"]):
        h += ([f"{x}, verdict", f"{x}, gap", f"{x}, 95% low", f"{x}, 95% high", f"{x}, marks"]
              if k in cols else [x])
    out = []
    for raw in t["rows"]:
        r = []
        for k, c in enumerate(raw):
            c = clean(c)
            m = VERDICT.match(c)
            r += ([m.group(1), num(m.group(2)), num(m.group(3)), num(m.group(4)),
                   m.group(5).strip()] if m else [c, "", "", "", ""]) if k in cols else [c]
        out.append(r)
    return h + ["Source"], [r + ["the other model: " + EPOCH + "; the Masein model: " + OURS]
                            for r in out]


def key_numbers(t: dict) -> tuple[list[str], list[list[str]]]:
    """one row per benchmark per mode, never two scores in a cell; a row that isn't a
    score (the phone speeds) stays as printed"""
    head = t["head"]
    me = head[1]
    h = [head[0], "mode", me, f"{me}, SE", f"{me}, 95% low", f"{me}, 95% high",
         head[2] + " (numbers we didn't measure: Epoch AI's runs, CC BY 4.0, or makers' "
         "published numbers, as the cell says)"]
    out = []
    for raw in t["rows"]:
        measure, value, compare = (clean(c) for c in raw)
        m = MODES.match(measure)
        modes = ["thinking " + x for x in m.group(2).split(" / ")] if m else [""]
        scores = value.split(" / ")
        parsed = [PM.match(s) or RANGE.match(s) for s in scores]
        if not m or len(scores) != len(modes) or not all(parsed):
            out.append([m.group(1) if m else measure, modes[0] if len(modes) == 1 else "",
                        value, "", "", "", compare])
            continue
        about = "thinking off" if compare.startswith("Thinking off") else "thinking on"
        for mode, s in zip(modes, scores):
            pm, rng = PM.match(s), RANGE.match(s)
            out.append([m.group(1), mode, num((pm or rng).group(1)), num(pm.group(2)) if pm else "",
                        num(rng.group(2)) if rng else "", num(rng.group(3)) if rng else "",
                        compare if mode == about or len(modes) == 1 else ""])
    return h, out


def table_file(t: dict) -> tuple[str, list[str], list[list[str]]]:
    name = t["name"]
    stem = "key-numbers" if name == "Key numbers" else name.lower().replace(" ", "-")
    if t["part"] > 1:
        stem += f"-part-{t['part']}"
    head, raw = t["head"], t["rows"]
    if name == "Key numbers":
        return stem, *key_numbers(t)
    if head == EPOCH_LIST:
        return stem, *epoch_list(t)
    vcols = [k for k in range(len(head)) if any(VERDICT.match(clean(r[k])) for r in raw if k < len(r))]
    if vcols:
        return stem, *verdict_grid(t, vcols)
    rows = [[clean(c) for c in r] for r in raw]
    ital = [[italic(c) for c in r] for r in raw]
    h, out = split_columns(head, rows)
    who = credit(t["caption"])
    by_col = [k for k in range(1, len(head)) if any(r[k] for r in ital if k < len(r))
              and all(r[k] or not clean(rr[k]) or clean(rr[k]) == "–"
                      for r, rr in zip(ital, raw) if k < len(r))]
    by_row = [any(r[1:]) or r[0] for r in ital]
    pub = re.search(r"\*Published: ([^*]+?)\.?\*", t["caption"])
    if by_col and all(k in by_col for r in ital for k, x in enumerate(r) if x):
        # columns that are all someone else's (Table 9): say so in their headings
        names = [head[k] for k in by_col]
        h = [x + f" [{who}]" if any(x == n or x.startswith(n + ", ") for n in names) else x
             for x in h]
    elif any(by_row):
        h.append("Whose numbers" if any(x.startswith("Source") for x in h) else "Source")
        for r, theirs in zip(out, by_row):
            r.append(who if theirs else OURS)
    elif pub:
        h.append("Source")
        for r in out:
            r.append(OURS if r[0].startswith("Masein") else "published: " + clean(pub.group(1)))
    return stem, h, out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def model_part(label: str) -> str:
    """a table label's model, without its setting and marks"""
    x = label.replace(" †", "").replace(" ‡", "")
    if "(base model)" in x:
        return x[:x.index(" (base model)")]
    return re.sub(r" \([^()]*\)$", "", x)


class Paper:
    def __init__(self, tabs: list[dict]):
        self.tabs = tabs

    def find(self, what: str, test) -> dict:
        got = [t for t in self.tabs if test(t)]
        if len(got) != 1:
            raise SystemExit(f"the paper has {len(got)} tables that look like {what}, not one")
        return got[0]

    def epoch(self, bench: str) -> dict:
        return self.find(f"Epoch's {bench} list", lambda t: t["head"] == EPOCH_LIST
                         and t["caption"].startswith(bench + ","))

    def masein(self, bench: str) -> tuple[str, str]:
        """the Masein model's thinking-on score and SE as the scores table prints them"""
        t = self.find("the scores in both modes", lambda t: "Thinking on (%)" in t["head"])
        k = t["head"].index("Thinking on (%)")
        r = next(r for r in t["rows"] if clean(r[0]).startswith(bench))
        m = PM.match(clean(r[k]))
        return num(m.group(1)), num(m.group(2))


def epoch_rows(t: dict) -> list[dict]:
    out = []
    for raw in t["rows"]:
        if not italic(raw[1]):
            continue
        label, score, verdict, gap = (clean(c) for c in raw)
        m = PM.match(score)
        out.append({"label": label.replace(" †", ""), "dagger": "†" in label, "score": score,
                    "v": num(m.group(1)) if m else num(score), "se": num(m.group(2)) if m else "",
                    "verdict": verdict.replace(" ‡", ""), "ddagger": "‡" in verdict, "gap": gap})
    return out


def dates_for(rows: list[dict], data: list[dict], dec: int) -> list[str]:
    """Epoch's release date for each table row: the data row printed the same way (score,
    SE and gap), and where several are, the one of the same model"""
    pm = lambda x: f"{x:+.1f}".replace("-", "−")
    by = defaultdict(list)
    for e in data:
        by[(f"{e['value']:.{dec}f} ± {e['se']:.{dec}f}",
            f"{pm(e['gap'])} ({pm(e['lo'])} to {pm(e['hi'])})")].append(e)
    out = []
    for r in rows:
        c = by.get((r["score"], r["gap"]), [])
        if len(c) > 1:
            c = [e for e in c if norm(e["model"]) == norm(model_part(r["label"]))] or c
        if len({e["released"] for e in c}) != 1:
            raise SystemExit(f"the timeline: can't tell which Epoch row {r['label']} is ({len(c)} found)")
        out.append(c[0]["released"] or "")
    return out


def fig_release(p: Paper, cap: str, src: Path) -> tuple[list[str], list[list[str]]]:
    rec = json.loads((src.figures / "epoch-recompute.json").read_text(encoding="utf-8"))["benchmarks"]
    at = re.search(r"plotted at (\d{1,2} \w+ \d{4})", cap)
    day = datetime.strptime(at.group(1), "%d %b %Y").strftime("%Y-%m-%d")
    out = []
    for bench, key, dec in (("GPQA Diamond", "gpqa", 1), ("MATH Level 5", "math", 2)):
        v, se = p.masein(bench)
        out.append([bench, "Masein model, thinking on", day, v, se, "",
                    OURS + " (plotted at the date of our runs)"])
        rows = epoch_rows(p.epoch(bench))
        for r, d in zip(rows, dates_for(rows, rec[key]["rows"], dec)):
            if d:
                out.append([bench, r["label"], d, r["v"], r["se"], "yes" if r["dagger"] else "",
                            EPOCH])
    return ["Benchmark", "Model (setting)", "Released", "Score (%)", "SE",
            "Below high effort (†)", "Source"], out


def fig_reliability(p: Paper, cap: str, src: Path):
    got = re.findall(r"(\d[\d,]*) ([A-Z][\w ]+?) questions \((\d+) runs each: (\d+), (\d+) and (\d+)",
                     cap)
    if not got:
        raise SystemExit("Figure 'right on every run': no counts in its caption")
    return (["Benchmark", "Questions", "Runs per question", "Right on every run",
             "Right on some runs", "Right on none"],
            [[b, num(n), r, a, s, z] for n, b, r, a, s, z in got])


def fig_thinking(p: Paper, cap: str, src: Path):
    t = p.find("how the thinking goes", lambda t: any("Mean thinking tokens" in h for h in t["head"]))
    i, j = (next(k for k, h in enumerate(t["head"]) if h.startswith(s))
            for s in ("Finished answers", "Mean thinking tokens"))
    out = []
    for raw in t["rows"]:
        a, b = PAIR.match(clean(raw[i])), PAIR.match(clean(raw[j]))
        out.append([clean(raw[0]), num(a.group(1)), num(a.group(2)), num(b.group(1)),
                    num(b.group(2))])
    return ["Benchmark", "Finished answers, right", "Finished answers, wrong",
            "Mean thinking tokens, right", "Mean thinking tokens, wrong"], out


def fig_ranout(p: Paper, cap: str, src: Path):
    t = p.find("ran out and looped", lambda t: t["head"][:2] == ["Benchmark", "Ran out of room"])
    names = re.search(r"thinking on, on (.*?);", cap).group(1)
    names = [x.strip() for x in re.split(r", | and ", names)]
    out = []
    for raw in t["rows"]:
        b = clean(raw[0])
        if b in names:
            k, n = re.match(r"([\d,]+) of ([\d,]+)", clean(raw[1])).groups()
            out.append([b, num(n), num(k), f"{100 * int(num(k)) / int(num(n)):.1f}"])
    if len(out) != len(names):
        raise SystemExit(f"Figure 'out of room': {len(out)} of {len(names)} benchmarks found")
    shares = {r[0]: r[3] for r in out}
    for share, b in re.findall(r"([\d.]+)% of ([A-Z][\w-]*(?: [A-Z][\w-]*)*)", cap):
        if shares.get(b, share) != share:
            raise SystemExit(f"Figure 'out of room': its caption says {share}% for {b}, "
                             f"the table gives {shares[b]}%")
    out.sort(key=lambda r: -float(r[3]))
    return ["Benchmark", "Answers", "Ran out of room", "Share that ran out (%)"], out


def fig_gpqa(p: Paper, cap: str, src: Path):
    extra = json.loads((src.figures / "figure-extras.json").read_text(encoding="utf-8"))
    named = {norm(x) for x in extra["figure-6 labelled"]}
    v, se = p.masein("GPQA Diamond")
    out = [["Masein model, thinking on", v, se, "", "", "", "yes", OURS]]
    for r in epoch_rows(p.epoch("GPQA Diamond")):
        out.append([r["label"], r["v"], r["se"], r["verdict"], "yes" if r["ddagger"] else "",
                    "yes" if r["dagger"] else "",
                    "yes" if norm(model_part(r["label"])) in named else "", EPOCH])
    if sum(1 for r in out if r[6]) != len(named) + 1:
        raise SystemExit("the GPQA Diamond lanes: not every labelled model is in the GPQA Diamond list")
    return ["Model (setting)", "Score (%)", "SE", "Verdict (Masein model … the model)",
            "Within a quarter of a combined SE of the line (‡)", "Below high effort (†)",
            "Labelled on the chart", "Source"], out


def fig_counts(p: Paper, cap: str, src: Path):
    out = []
    for bench in ("GPQA Diamond", "OTIS Mock AIME", "MATH Level 5"):
        rows = epoch_rows(p.epoch(bench))
        c = {k: sum(1 for r in rows if r["verdict"] == k)
             for k in ("higher than", "level with", "lower than")}
        out.append([bench, str(len(rows)), str(c["higher than"]), str(c["level with"]),
                    str(c["lower than"])])
    return ["Benchmark", "Models", "Masein model higher than", "Level with",
            "Masein model lower than"], out


def fig_grid(p: Paper, cap: str, src: Path):
    grid = list(csv.reader(open(src.figures / "figure-8.csv", encoding="utf-8")))[1:]
    benches = [("GPQA Diamond", 2), ("OTIS Mock AIME", 5), ("MATH Level 5", 8)]
    lists = {b: epoch_rows(p.epoch(b)) for b, _ in benches}
    dec = {"GPQA Diamond": 1, "OTIS Mock AIME": 1, "MATH Level 5": 2}
    out = []
    for g in grid:
        row = [g[0], g[1]]
        for b, k in benches:
            if g[0] == "Subject":
                row += [*p.masein(b), "thinking on", ""]
                continue
            if not g[k]:
                row += ["", "", "", "not run by Epoch"]
                continue
            want = f"{float(g[k]):.{dec[b]}f}"
            hit = [r for r in lists[b] if r["v"] == want and norm(model_part(r["label"])).startswith(norm(g[1]))]
            if len(hit) != 1:
                raise SystemExit(f"the 21-row grid: {g[1]} on {b} at {want} is {len(hit)} rows of the paper's list")
            r = hit[0]
            setting = re.search(r"\(([^()]*)\)$", r["label"])
            row += [r["v"], r["se"], setting.group(1) if setting else "",
                    r["verdict"] + (" ‡" if r["ddagger"] else "") + (" †" if r["dagger"] else "")]
        out.append(row + [g[14], OURS if g[0] == "Subject" else EPOCH])
    h = ["Group", "Model"]
    for b, _ in benches:
        h += [b, f"{b}, SE", f"{b}, setting", f"{b}, verdict (Masein model … the model)"]
    return h + ["Parameters, total / active (model cards; Table F4)", "Source"], out


def fig_size(p: Paper, cap: str, src: Path):
    t = p.find("on-device models", lambda t: len(t["head"]) > 1
               and t["head"][1] == "File size and memory it runs at")
    pts = list(csv.DictReader(open(src.figures / "figure-9.csv", encoding="utf-8")))
    names = {"Bolt": "Nemotron 3.5 Bolt E4B (Bolt)", "Qwen3.5-4B, measured by NVIDIA":
             "Qwen3.5-4B, measured by NVIDIA"}
    cols = {h: t["head"].index(h) for h in ("GPQA Diamond", "MMLU-Pro", "IFEval", "IFBench")}
    out = [["Masein model", "16.0", *p.masein("GPQA Diamond"), *p.masein("MMLU-Pro"),
            *p.masein("IFEval"), *p.masein("IFBench"), "GPQA Diamond; IFBench", OURS]]
    for x in pts:
        nm = names.get(x["Model"], x["Model"])
        raw = next((r for r in t["rows"] if clean(r[0]).startswith(nm)), None)
        if raw is None:
            raise SystemExit(f"the on-device chart: {x['Model']} is not a row of Table G1")
        size = x["Size (GB)"]
        shown = {"1.0": "1 GB", "2.0": "2.0 GB"}.get(size, f"{float(size):g} GB")
        if shown not in clean(raw[1]) and x["Model"] not in names:
            raise SystemExit(f"the on-device chart: {x['Model']}'s {shown} is not in Table G1")
        row = [x["Model"], size]
        for h in ("GPQA Diamond", "MMLU-Pro", "IFEval", "IFBench"):
            v = x[h]
            if v:
                cell = clean(raw[cols[h]])
                printed = next((s for s in re.findall(NUM, cell) if float(s) == float(v)), None)
                if printed is None:
                    raise SystemExit(f"the on-device chart: {x['Model']}'s {h} {v} is not in Table G1 ({cell})")
                v = printed
            row += [v, ""]
        out.append(row + [x["Plotted"], MAKER])
    return ["Model", "File or deployed size (GB)", "GPQA Diamond", "GPQA Diamond, SE", "MMLU-Pro",
            "MMLU-Pro, SE", "IFEval", "IFEval, SE", "IFBench", "IFBench, SE",
            "Plotted in", "Source"], out


def fig_devicemark(p: Paper, cap: str, src: Path):
    t = p.find("DeviceMark's rows", lambda t: any(h.startswith("Composite") for h in t["head"]))
    k = next(i for i, h in enumerate(t["head"]) if h.startswith("Composite"))
    out = []
    for raw in t["rows"]:
        m = RANGE.match(clean(raw[k]))
        out.append([clean(raw[0]), clean(raw[1]), num(m.group(1)), num(m.group(2)), num(m.group(3)),
                    DEVICEMARK if italic(raw[k]) else OURS + " (replica)"])
    return ["Model", "Run by", "Composite", "95% low", "95% high", "Source"], out


def fig_speed(p: Paper, cap: str, src):
    """DeviceMark composite against writing speed on a phone: each score must be Table 12's
    as printed; a Pipette speed must carry Pipette's notice, and the Masein model's must be
    the one the key numbers print"""
    f = src.speeds
    if not f.exists():
        raise SystemExit(f"the speed chart: {f} is missing (the figure script writes it)")
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    if not rows or list(rows[0].keys()) != SPEED_HEAD:
        raise SystemExit(f"the speed chart: {f} doesn't have the columns {', '.join(SPEED_HEAD)}")
    t = p.find("DeviceMark's rows", lambda t: any(h.startswith("Composite") for h in t["head"]))
    k = next(i for i, h in enumerate(t["head"]) if h.startswith("Composite"))
    by = {clean(r[0]): r for r in t["rows"]}
    ours = [r for x in p.tabs for r in x["rows"] if clean(r[0]).startswith("Writing speed on a phone")]
    out = []
    for r in rows:
        name = r["model"] + (f", {r['mode']}" if r["mode"] else "")
        raw = by.get(name)
        if raw is None:
            raise SystemExit(f"the speed chart: {name} is not a row of {t['name']}")
        m = RANGE.match(clean(raw[k]))
        if (num(m.group(1)), num(m.group(2)), num(m.group(3))) != (r["devicemark_composite"], r["ci95_low"],
                                                                   r["ci95_high"]):
            raise SystemExit(f"the speed chart: {name} is plotted at {r['devicemark_composite']} "
                             f"[{r['ci95_low']}, {r['ci95_high']}]; {t['name']} prints {clean(raw[k])}")
        if italic(raw[k]):
            if r["speed_tok_s"]:
                raise SystemExit(f"the speed chart: {name} is a published row and has no speed")
            out.append([name, num(m.group(1)), num(m.group(2)), num(m.group(3)), DEVICEMARK, "", "", ""])
            continue
        if not r["speed_tok_s"] or not is_num(r["speed_tok_s"]):
            raise SystemExit(f"the speed chart: {name} has no speed")
        if r["speed_source"].startswith("Pipette"):
            if r["speed_credit"] != PIPETTE:
                raise SystemExit(f"the speed chart: {name}'s speed is Pipette's and must carry its notice")
        elif r["speed_credit"] != OURS:
            raise SystemExit(f"the speed chart: {name}'s speed is not Pipette's, so it is {OURS}")
        if name.startswith("Masein model") and not any(r["speed_tok_s"] in numbers_in(" ".join(x)) for x in ours):
            raise SystemExit(f"the speed chart: the Masein model's {r['speed_tok_s']} tokens/s is not in the "
                             "paper's writing-speed row")
        out.append([name, num(m.group(1)), num(m.group(2)), num(m.group(3)), OURS, r["speed_tok_s"],
                    r["speed_source"], r["speed_credit"]])
    return ["Model", "DeviceMark composite", "95% low", "95% high", "Score: whose", "Writing speed (tokens/s)",
            "Speed: how it was measured", "Speed: whose"], out


def is_speed_chart(cap: str) -> bool:
    return "writing speed" in cap.lower() and "DeviceMark" in cap


FIGURES = [  # phrases of the caption (any one will do), and what builds the figure's data
    ((is_speed_chart,), fig_speed),
    (("by release date",), fig_release),
    (("How one question becomes a score",), None),
    (("right on every run, on some runs and on none",), fig_reliability),
    (("Mean thinking tokens per finished answer",), fig_thinking),
    (("share of answers that ran out of room at the final limit",), fig_ranout),
    (("Each dot is one model's best-scoring Epoch run", "in the lane of its verdict"), fig_gpqa),
    (("The bars count the models", "Each bar counts the models"), fig_counts),
    (("one model per row",), fig_grid),
    (("Score against file size",), fig_size),
    (("DeviceMark composite with its 95% range",), fig_devicemark),
]


def builder(cap: str):
    for keys, fn in FIGURES:
        if any(k(cap) if callable(k) else k in cap for k in keys):
            return fn
    return "?"


class Sources:
    def __init__(self, figures: Path, speeds: Path):
        self.figures, self.speeds = figures, speeds


# ---------------------------------------------------------------------------
# checks and output
# ---------------------------------------------------------------------------

def numbers_in(text: str) -> set[str]:
    return {num(x) for x in re.findall(NUM, clean(text))}


def check_table(stem: str, t: dict, rows: list[list[str]]) -> None:
    """every number written is printed in the row it came from (key numbers: in the table,
    whose rows are split by mode)"""
    whole = numbers_in(" ".join(c for raw in t["rows"] for c in raw))
    for raw, r in (zip(t["rows"], rows) if stem != "key-numbers" else ((None, r) for r in rows)):
        have = numbers_in(" ".join(raw)) if raw else whole
        for c in r:
            if is_num(c) and num(c) not in have:
                raise SystemExit(f"{stem}: {c} is not printed in its row ({raw[0][:60]})")


def first_sentence(cap: str) -> str:
    m = re.match(r"^(.*?[.?!])(?:\s|$)", clean(cap))
    return m.group(1) if m else clean(cap)


def write(path: Path, head: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        w.writerows(rows)


def main(argv: list[str]) -> int:
    trial = "--trial" in argv
    args = [a for a in argv if a != "--trial"]
    opt = {}
    for flag in ("--figures", "--speeds"):
        if flag in args:
            i = args.index(flag)
            opt[flag] = Path(args[i + 1])
            del args[i:i + 2]
    out, paper = Path(args[0]), Path(args[1])
    src = Sources(opt.get("--figures", paper.parent / "round13-figures"),
                  opt.get("--speeds", paper.parent / "figures-r19" / "figure-1.csv"))
    lines = published_text(paper.read_text(encoding="utf-8"))
    left = tbds(lines)
    if left:
        print(f"{len(left)} TBD left in the paper:", *left, sep="\n  ")
        if not trial:
            print("stopped: nothing written. Fill every TBD in the paper, then build again")
            return 1
    tabs = tables(lines)
    built, index = [], []
    for t in tabs:
        stem, h, rows = table_file(t)
        check_table(stem, t, rows)
        built.append((stem, h, rows))
        index.append([f"{stem}.csv", t["name"] + (", its second part" if t["part"] > 1 else ""),
                      first_sentence(t["caption"]) if t["caption"] else "The report's key numbers."])
    p = Paper(tabs)
    for n, cap in figures(lines):
        fn = builder(cap)
        if fn == "?":
            raise SystemExit(f"Figure {n} is new to this script: {cap[:90]}")
        if fn is None:
            print(f"figure-{n}: a diagram, no data")
            continue
        built.append((f"figure-{n}", *fn(p, cap, src)))
        index.append([f"figure-{n}.csv", f"Figure {n}", first_sentence(cap)])
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.csv"):
        f.unlink()
    for stem, h, rows in built:
        write(out / f"{stem}.csv", h, rows)
    write(out / "index.csv", ["File", "In the report", "What it holds (the caption's first sentence)"],
          index)
    print(f"{len(built)} files in {out}: " + ", ".join(s for s, _, _ in built))
    tbd = [s for s, h, rows in built if any("TBD" in c for r in [h, *rows] for c in r)]
    if tbd:
        print("files that carry a TBD from the paper: " + ", ".join(tbd))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
