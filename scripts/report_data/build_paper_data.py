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
on-device chart's points, and the throughputs of the DeviceMark-against-throughput chart.
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
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EPOCH = "Epoch AI Benchmarking Hub, CC BY 4.0"
SPEED_HEAD = ["model", "mode", "devicemark_composite", "ci95_low", "ci95_high", "score_source", "speed_tok_s",
              "speed_source", "speed_credit"]
OURS = "measured by Teraformer"
DEVICEMARK = "DeviceMark (devicemark.github.io), CC BY 4.0, published"
MAKER = "published by the model's maker or NVIDIA, as the row says"


# r37 (masein, 16 Nov): LaTeX numbers the figures in print order, as it does the tables: a
# figure prints as its caption's place among the md's figures, whatever its md id (the files the
# figure script draws keep their ids). What goes out names each figure as printed
FIG_REF = re.compile(r"\b(Figures?) (\d+(?:(?:, and |, | and )\d+)*)\b(?! (?:to|–) ?\d)")
FIG_RANGE = re.compile(r"\bFigures? \d+ ?(?:to|–) ?\d+")


class FigNumbers:
    """the md's figure ids as LaTeX prints them: their order in the paper"""

    def __init__(self, caps: list[tuple[str, str]]):
        self.order = {n: str(k) for k, (n, _) in enumerate(caps, 1)}

    def number(self, n: str) -> str:
        if n not in self.order:
            raise SystemExit(f"the text names Figure {n}, which the paper doesn't have")
        return self.order[n]

    def text(self, s: str) -> str:
        """a cell or a caption with its figure references as printed — a range ("Figures 2 to 5")
        would need its own reading in print order, so it stops the build"""
        if "Figure" not in s:
            return s
        r = FIG_RANGE.search(s)
        if r:
            raise SystemExit(f"a range of figures, {r.group(0)!r}: in print order a range of md ids "
                             "isn't a range, so write the figures out")
        return FIG_REF.sub(lambda m: m.group(1) + " " + re.sub(r"\d+", lambda k: self.number(k.group(0)),
                                                              m.group(2)), s)


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


def tables(lines: list[str]) -> list[dict]:
    """each table with its caption: {name, part, caption, head, rows (raw cells)}"""
    out, cap, name, part, i, section = [], "", None, 0, 0, ""
    while i < len(lines):
        if lines[i].startswith("## "):
            section = lines[i][3:].strip()
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
                        "caption": cap, "head": [clean(c) for c in rows[0]], "rows": rows[2:],
                        "section": section})
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


# r29 (masein, 16 Nov): LaTeX numbers the tables, and the files go by the numbers it prints.
# Section 3 is Kevin's own file there: its five tables are the md's "Table P1" to "P5" in the main
# text, and print as Tables 1 to 5. The md's own Section 3 stands in for his file, and its tables
# print nowhere. The md's main-text tables come after, in order: its Tables 4 to 12 print as 6 to
# 14. Appendix tables keep their letters. A paper whose text names no "Table P" keeps its numbers
KEVIN_SECTION = "3. "
REF_LIST = re.compile(r"\b(Tables?) ((?:P?\d+|[A-Z]\d+[a-z]?)(?:(?:, and |, | and | to |–)(?:P?\d+|[A-Z]\d+[a-z]?))*)")


class Numbers:
    """the md's table numbers as LaTeX prints them"""

    def __init__(self, lines: list[str], tabs: list[dict], his: list[dict] | None = None):
        """`his`: Kevin's file's tables in its order (kevin_tables), or None when not given"""
        outside = [x for x in lines if not x.startswith("|")]
        text = "\n".join(outside)
        kevin_refs = [int(n) for m in REF_LIST.finditer(text) for n in re.findall(r"P(\d+)", m.group(2))]
        self.named = max(kevin_refs) if kevin_refs else 0          # P1 to P<named> in the text
        in3 = [x for x in tabs if x["section"].startswith(KEVIN_SECTION)]
        self.stand_in = sorted({x["name"] for x in in3 if re.fullmatch(r"Table P?\d+", x["name"])})
        # his tables print 1 to n in his file's order; a P the text names is the one its label is
        self.his, self.order = his, {}
        if self.named and his is not None:
            labels = [x["label"] for x in his]
            for k in range(1, self.named + 1):
                lab = KEVIN_LABELS.get(f"P{k}")
                if lab not in labels:
                    raise SystemExit(f"Kevin's file has no {lab or f'label for P{k}'} (the text's "
                                     f"Table P{k}): nothing written")
                self.order[f"P{k}"] = str(labels.index(lab) + 1)
        self.kevin = len(his) if self.named and his is not None else self.named
        self.shift, self.first = 0, 1
        if not self.kevin:
            self.stand_in = []
            return
        main = []
        for x in tabs:
            m = re.fullmatch(r"Table (\d+)", x["name"])
            if m and not x["section"].startswith(KEVIN_SECTION) and int(m.group(1)) not in main:
                main.append(int(m.group(1)))
        self.first = main[0] if main else 1
        if main != list(range(self.first, self.first + len(main))):
            raise SystemExit(f"the md's main-text tables are numbered {main}: LaTeX numbers them in "
                             f"order after Kevin's {self.kevin}, so they must run with no gap")
        self.shift = self.kevin - (self.first - 1)

    def skipped(self, t: dict) -> bool:
        """a table of the md's Section 3 when it stands in for Kevin's file: printed nowhere"""
        return bool(self.kevin) and t["name"] in self.stand_in and t["section"].startswith(KEVIN_SECTION)

    def number(self, n: str) -> str:
        if n.startswith("P"):
            return self.order.get(n, n[1:]) if self.kevin else n
        if not n.isdigit() or not self.kevin:
            return n
        if int(n) < self.first:
            raise SystemExit(f"the text names Table {n}, which stands in Section 3 for Kevin's file "
                             "and prints nowhere")
        return str(int(n) + self.shift)

    def name(self, name: str) -> str:
        m = re.fullmatch(r"Table (P?\d+)", name)
        return f"Table {self.number(m.group(1))}" if m else name

    def text(self, s: str) -> str:
        """a cell or a caption with its table references as printed"""
        if not self.kevin or "Table" not in s:
            return s
        return REF_LIST.sub(lambda m: m.group(1) + " " + re.sub(
            r"P?\d+(?![a-z\d])|[A-Z]\d+[a-z]?", lambda k: self.number(k.group(0)), m.group(2)), s)


# r29 (masein, 16 Nov): Kevin's five tables, read straight from his own file (on-the-phone.tex,
# downloaded from Overleaf, refreshed on release day), never from the md. The md's main text
# names them P1 to P5, each by its label as overleaf_build.py links them; LaTeX numbers them in
# the order his file has them. A label missing, or a row that doesn't parse, stops the build
KEVIN_LABELS = {"P1": "tab:phone-modes", "P2": "tab:phone-memory", "P3": "tab:context-instant",
                "P4": "tab:context-medium", "P5": "tab:context-high"}
LATEX_NUM = re.compile(r"[\d,]*\.?\d+")


def braced(s: str, i: int) -> tuple[str, int]:
    """the text of the {…} group that opens at s[i], and the index after it"""
    if s[i] != "{":
        raise ValueError(f"no group at {s[i:i + 30]!r}")
    depth, j = 0, i
    while j < len(s):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"a group never closed: {s[i:i + 40]!r}")


def latex_words(s: str) -> str:
    """a cell or a caption as it prints: dashes, spaces, %, °C, a bold or italic group's words"""
    s = re.sub(r"\\(?:textbf|textit|emph|mbox)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:ref\*?|label)\{[^{}]*\}", "", s)
    for a, b in (("---", "—"), ("--", "–"), ("~", " "), ("\\%", "%"), ("$^\\circ$C", "°C"),
                 ("\\,", " "), ("``", "“"), ("''", "”")):
        s = s.replace(a, b)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    return " ".join(s.replace("{", "").replace("}", "").split())


def latex_cells(row: str) -> list[str]:
    """a tabular row's cells, a \\multicolumn{n}{…}{text} as n cells of its text"""
    out = []
    for c in re.split(r"(?<!\\)&", row):
        m = re.match(r"\s*\\multicolumn\{(\d+)\}\{[^}]*\}", c)
        if m:
            text, _ = braced(c, m.end())
            out += [latex_words(text)] * int(m.group(1))
        else:
            out.append(latex_words(c))
    return out


def kevin_tables(path: Path) -> list[dict]:
    """each labelled table in Kevin's file, in its order: {label, caption, head, rows}"""
    src = re.sub(r"(?<!\\)%.*", "", path.read_text(encoding="utf-8"))
    out = []
    for blk in re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", src, re.S):
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        cap = blk.find("\\caption{")
        tab = re.search(r"\\begin\{tabular\}", blk)
        if not (lab and cap >= 0 and tab):
            raise SystemExit(f"Kevin's file: a table without its label, caption or tabular "
                             f"({(lab.group(1) if lab else blk.strip()[:60])!r})")
        caption = latex_words(braced(blk, cap + len("\\caption"))[0])
        spec, k = braced(blk, tab.end())
        cols = re.findall(r"[lcr]", spec)
        body = blk[k:blk.index("\\end{tabular}")]
        head_part, _, rows_part = body.partition("\\midrule")
        if not rows_part:
            raise SystemExit(f"Kevin's file: {lab.group(1)} has no \\midrule under its header")

        def lines(part):
            part = re.sub(r"\\(?:toprule|midrule|bottomrule)|\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", part)
            return [x.strip() for x in re.split(r"\\\\", part) if x.strip()]
        names = [""] * len(cols)
        for raw in lines(head_part):
            h = latex_cells(raw)
            if len(h) != len(cols):
                raise SystemExit(f"Kevin's file: {lab.group(1)}'s header has {len(h)} cells for "
                                 f"{len(cols)} columns")
            for j, c in enumerate(h):
                if c and not names[j]:
                    names[j] = c
                elif c:                                    # a unit under its name, a name under its group
                    names[j] = f"{names[j]} {c}" if c.startswith("(") else f"{names[j]}, {c}"
        rows, above = [], []
        for x in lines(rows_part):
            cells = latex_cells(x)
            if len(cells) != len(cols):
                raise SystemExit(f"Kevin's file: a row of {lab.group(1)} has {len(cells)} cells for "
                                 f"{len(cols)} columns: {x[:80]!r}")
            for j, (c, kind) in enumerate(zip(cells, cols)):
                if kind == "l" and not c and above and all(not y for y in cells[:j]):
                    cells[j] = above[j]                    # the device, said once a group
                if kind == "r" and not (LATEX_NUM.fullmatch(c) or c == "–"):
                    raise SystemExit(f"Kevin's file: {lab.group(1)}'s {names[j]!r} has {c!r}, not a "
                                     f"number: {x[:80]!r}")
            rows.append(cells)
            above = cells
        out.append({"label": lab.group(1), "caption": caption, "head": names, "rows": rows})
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
    """whose a table's other numbers are, from its caption"""
    if "DeviceMark's published" in caption or "run by DeviceMark" in caption:
        return DEVICEMARK
    if "as published" in caption:
        return MAKER
    if "Epoch" in caption:
        return EPOCH
    return "not measured by us"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

EPOCH_LIST = ["Model (setting)", "Epoch score", "Verdict", "Gap (95% range)"]
# a key number's modes: "thinking on / off" (to r22) or the app's, "High / Medium / Instant" (r23)
MODES = re.compile(r"^(.*), (thinking (?:on / off|off / on|on|off)|(?:High|Medium|Instant)(?: / (?:High|Medium|Instant))*)$")


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
                    OURS if masein_row(raw[0]) else EPOCH])
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
        modes = ([("thinking " + x if m.group(2).startswith("thinking") else x)
                  for x in m.group(2).removeprefix("thinking ").split(" / ")] if m else [""])
        scores = value.split(" / ")
        parsed = [PM.match(s) or RANGE.match(s) for s in scores]
        if not m or len(scores) != len(modes) or not all(parsed):
            out.append([m.group(1) if m else measure, modes[0] if len(modes) == 1 else "",
                        value, "", "", "", compare])
            continue
        # the comparison is about the mode it starts with ("High: …", "Thinking off: …"), else
        # the first (thinking on, or High)
        about = next((x for x in modes if compare.lower().startswith(x.lower() + ":")
                      or compare.lower().startswith(x.lower() + " ")),
                     "thinking on" if "thinking on" in modes else modes[0])
        for mode, s in zip(modes, scores):
            pm, rng = PM.match(s), RANGE.match(s)
            out.append([m.group(1), mode, num((pm or rng).group(1)), num(pm.group(2)) if pm else "",
                        num(rng.group(2)) if rng else "", num(rng.group(3)) if rng else "",
                        compare if mode == about or len(modes) == 1 else ""])
    return h, out


def masein_row(label: str) -> bool:
    return clean(label).startswith("Masein model")


def whose(t: dict) -> tuple[list[int], list[bool]]:
    """r22 (masein): the columns, or else the rows, of a table that are someone else's numbers,
    from the paper's words — r18 took italics out of every table. A "Run by" column says it
    row by row ("us (replica)" or "DeviceMark", Table 12); a caption that says the other
    models' columns are not measured by us (Table 9) makes every column but the Masein
    model's theirs; one that says the other rows' numbers are as published, none measured by
    us (Table G1), every row but the Masein model's"""
    head, cap = t["head"], " ".join(t["caption"].split())
    if "Run by" in head:
        k = head.index("Run by")
        return [], [not clean(r[k]).startswith("us") for r in t["rows"]]
    if re.search(r"columns? (?:are|is) [^.]*not measured by us", cap):
        return [k for k in range(1, len(head)) if not head[k].startswith("Masein model")], []
    if re.search(r"none measured by us|not measured by us", cap) and head[0].startswith("Model"):
        return [], [not masein_row(r[0]) for r in t["rows"]]
    return [], []


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
    h, out = split_columns(head, rows)
    who = credit(t["caption"])
    by_col, by_row = whose(t)
    pub = re.search(r"\*Published: ([^*]+?)\.?\*", t["caption"])
    if by_col:
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
    def __init__(self, tabs: list[dict], lines: list[str] | None = None):
        self.tabs, self.lines = tabs, lines or []

    def find(self, what: str, test) -> dict:
        got = [t for t in self.tabs if test(t)]
        if len(got) != 1:
            raise SystemExit(f"the paper has {len(got)} tables that look like {what}, not one")
        return got[0]

    def epoch(self, bench: str) -> dict:
        return self.find(f"Epoch's {bench} list", lambda t: t["head"] == EPOCH_LIST
                         and t["caption"].startswith(bench + ","))

    def top(self) -> str:
        """the mode the Masein model's scores beside Epoch's are in: High (r23's Table 5) or,
        before, thinking on"""
        return "High" if any("High (%)" in t["head"] for t in self.tabs) else "thinking on"

    def masein(self, bench: str, mode: str | None = None) -> tuple[str, str]:
        """the Masein model's High (thinking-on) score and SE as the scores table prints them — r29:
        or another mode's, by its column"""
        col = f"{mode} (%)" if mode else "High (%)" if self.top() == "High" else "Thinking on (%)"
        t = self.find("the scores in its modes", lambda t: col in t["head"])
        k = t["head"].index(col)
        r = next(r for r in t["rows"] if clean(r[0]).startswith(bench))
        m = PM.match(clean(r[k]))
        return num(m.group(1)), num(m.group(2))


def epoch_rows(t: dict) -> list[dict]:
    """Epoch's rows of an Epoch list: every row but the Masein model's own (r22, masein: r18
    took italics out of every table)"""
    out = []
    for raw in t["rows"]:
        if masein_row(raw[0]):
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
        out.append([bench, f"Masein model, {p.top()}", day, v, se, "",
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
    # r31 (16 Nov): "in High mode (thinking on), on …" as well as "thinking on, on …"
    names = re.search(r"thinking on\)?, on (.*?);", cap).group(1)
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
    out = [[f"Masein model, {p.top()}", v, se, "", "", "", "yes", OURS]]
    # r29 (16 Nov, masein): Medium and Instant are reference lines beside High's, with no verdict —
    # the scores table's (the md's Table 5, printed 7) and Section 4.4's
    modes = [m for m in ("Medium", "Instant") if p.top() == "High"]
    for m in modes:
        mv, mse = p.masein("GPQA Diamond", m)
        out.append([f"Masein model, {m} (a reference line, no verdict)", mv, mse, "", "", "", "yes",
                    f"{OURS} (Table 5, Section 4.4)"])
    for r in epoch_rows(p.epoch("GPQA Diamond")):
        out.append([r["label"], r["v"], r["se"], r["verdict"], "yes" if r["ddagger"] else "",
                    "yes" if r["dagger"] else "",
                    "yes" if norm(model_part(r["label"])) in named else "", EPOCH])
    if sum(1 for r in out if r[6]) != len(named) + 1 + len(modes):
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
                row += [*p.masein(b), p.top(), ""]
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


def run_by_us(t: dict, raw: list[str]) -> bool:
    """r22 (masein, 14 Nov): Table 12 says who ran each row in words, in its "Run by" column
    ("us (replica)" or "DeviceMark"); since r18 no table sets anything in italics"""
    k = next((i for i, h in enumerate(t["head"]) if clean(h) == "Run by"), None)
    if k is None:
        raise SystemExit(f"{t['name']} has no \"Run by\" column to say whose each row is")
    return clean(raw[k]).startswith("us")


def fig_devicemark(p: Paper, cap: str, src: Path):
    t = p.find("DeviceMark's rows", lambda t: any(h.startswith("Composite") for h in t["head"]))
    k = next(i for i, h in enumerate(t["head"]) if h.startswith("Composite"))
    out = []
    for raw in t["rows"]:
        m = RANGE.match(clean(raw[k]))
        out.append([clean(raw[0]), clean(raw[1]), num(m.group(1)), num(m.group(2)), num(m.group(3)),
                    OURS + " (replica)" if run_by_us(t, raw) else DEVICEMARK])
    return ["Model", "Run by", "Composite", "95% low", "95% high", "Source"], out


# r22–r25 (12 and 14 Nov): Figure 1 plots the Masein model's modes, each at the short-chat
# median the key numbers print for it on PHONE — r25: the iPhone 17 Pro Max, the phone family
# of the other models' speeds: Instant in its own row, "Token generation throughput on a phone
# (…), Instant mode"; another mode in a throughput row's words, as "Medium mode, preliminary:
# 12.6 and 9.7 in short chats", its numbers in the order the row names the phones. A mode with
# no speed yet (High) is kept unplotted
THROUGHPUT = "Token generation throughput on a phone"
APP_MODES = ("Instant", "Medium", "High")
PHONE = "iPhone 17 Pro Max"
PHONES = re.compile(r"(OnePlus 15|iPhone 17 Pro Max)(?: \([^)]*\))?: ")
SHORT_CHATS = re.compile(re.escape(PHONE) + r"(?: \([^)]*\))?: (\d+(?:\.\d+)?)(?: \([^)]*\))? in short chats")
# r25 (masein): the other models' speeds are DeviceMark's published decode tokens/s on an iPhone
# 17 Pro (CC BY 4.0), each the one on DeviceMark's board as the repo keeps it
BOARD = Path(__file__).resolve().parents[2] / "eval_tasks" / "devicemark" / "board-snapshot.json"


def masein_speed(p: Paper, mode: str) -> tuple[str, bool]:
    """(the iPhone 17 Pro Max's short-chat median the key numbers print for the mode, and whether
    they call it preliminary)"""
    rows = [r for x in p.tabs for r in x["rows"] if clean(r[0]).startswith(THROUGHPUT)]
    own = [r for r in rows if clean(r[0]).endswith(f"{mode} mode")]
    if len(own) > 1:
        raise SystemExit(f"the speed chart: the key numbers have {len(own)} rows \"{THROUGHPUT} (…), "
                         f"{mode} mode\"; the Masein model's throughput is read from one")
    if own:
        m = SHORT_CHATS.search(clean(own[0][1]))
        if m is None:
            raise SystemExit(f"the speed chart: the key numbers' throughput row for {mode} mode gives no "
                             f"{PHONE} median in short chats")
        return num(m.group(1)), "preliminary" in clean(own[0][0]).lower()
    said = [(r, m) for r in rows for m in re.finditer(
        rf"{mode} mode(, preliminary)?: (\d+(?:\.\d+)?(?: and \d+(?:\.\d+)?)*)(?:(?!\. [A-Z])[^;|])*? "
        r"in short chats", clean(r[1]))]
    # r29 (16 Nov): "In short chats, Medium mode: 12.6 and 9.2; High mode, thinking included: 10.2 and
    # 8.5" — one sentence, a clause a mode, whatever each clause says of its mode
    for r in rows:
        for s in re.finditer(r"In short chats, (.*?)(?:\.(?=\s|$)|$)", clean(r[1])):
            for clause in s.group(1).split("; "):
                m = re.fullmatch(rf"{mode} mode(, [^:]*preliminary[^:]*|, [^:]+)?: "
                                 r"(\d+(?:\.\d+)?(?: and \d+(?:\.\d+)?)*)", clause.strip())
                if m:
                    said.append((r, m))
    if len(said) != 1:
        raise SystemExit(f"the speed chart: the key numbers give {mode} mode's short-chat throughput "
                         f"{len(said)} times; the Masein model's is read from one")
    r, m = said[0]
    phones = [x.group(1) for x in PHONES.finditer(clean(r[1]))]
    values = m.group(2).split(" and ")
    if len(values) != len(phones) or PHONE not in phones:
        raise SystemExit(f"the speed chart: the key numbers give {mode} mode {len(values)} short-chat "
                         f"numbers for {len(phones)} phones; which is the {PHONE}'s can't be told")
    return num(values[phones.index(PHONE)]), "preliminary" in (m.group(1) or "")


# r29 (masein, 16 Nov): Table 12's row at the app's limits is H1 re-cut at 2,048 tokens in total
# (docs/whitepaper-data/h1-recut). The build traces it to the re-cut's own result, recut-result.json,
# whose run records lie beside it: the row's name, composite and range, three subscores and
# answered share must be the result's, and the records the ones it names
APP_ROW = "at the app's limits"
APP_COLUMNS = (("Composite", "composite"), ("IFEval", "ifeval"), ("MMLU-Pro", "mmlu_pro"), ("MATH", "math"),
               ("Answered", "answered"))


def check_app_limits(p: Paper, path: Path | None) -> None:
    dm = [x for x in p.tabs if any(h.startswith("Composite") for h in x["head"])]
    rows = [r for x in dm for r in x["rows"] if clean(r[0]).endswith(APP_ROW)]
    if not rows:
        return
    t = p.find("DeviceMark's rows", lambda x: x in dm)
    name = clean(rows[0][0])
    if len(rows) > 1:
        raise SystemExit(f"{t['name']} has {len(rows)} rows at the app's limits; the re-cut gives one")
    if path is None:
        raise SystemExit(f"{t['name']} has {name}: give its re-cut's result (recut-result.json) with "
                         "--app-limits")
    res = json.loads(path.read_text(encoding="utf-8"))
    if res.get("row") != name:
        raise SystemExit(f"{path.name} is the result for {res.get('row')!r}, not {name!r}")
    for what, rec in (res.get("records") or {}).items():
        f = path.parent / rec["file"]
        data = f.read_bytes() if f.exists() else b""
        if hashlib.sha256(data).hexdigest() != rec["sha256"] or data.count(b"\n") != rec["n"]:
            raise SystemExit(f"{path.name}: its {what} records ({rec['file']}) are not the ones it names")
    if not res.get("records"):
        raise SystemExit(f"{path.name} names no run records")
    a = res["after"]
    c = a["composite"]
    want = {"composite": f"{100 * c['value']:.1f} [{100 * c['ci'][0]:.1f}, {100 * c['ci'][1]:.1f}]",
            "answered": f"{100 * a['answered_pct']:.0f}%",
            **{b: f"{100 * a['benches'][b]['acc']:.1f}" for b in ("ifeval", "mmlu_pro", "math")}}
    for head, key in APP_COLUMNS:
        k = next((i for i, h in enumerate(t["head"]) if h.startswith(head)), None)
        if k is None:
            raise SystemExit(f"{t['name']} has no {head} column to check {name} against")
        if clean(rows[0][k]) != want[key]:
            raise SystemExit(f"{t['name']}'s {name}: {t['head'][k]} prints {clean(rows[0][k])}; the re-cut "
                             f"({path.name}) gives {want[key]}")


# r31 (masein, 16 Nov): the unnumbered figure under the abstract — GPQA Diamond, the Masein model's
# three modes among three of Epoch's models — as the figure script writes it, figure-abstract.csv
# beside figure-1.csv. Each row is traced to the scores table (the md's Table 5, printed 7) or
# Epoch's GPQA Diamond list (Table F1): its score and SE as printed there, its 95% range ±1.96 SE of
# those, the rows highest first
ABSTRACT = "figure-abstract.csv"
ABSTRACT_HEAD = ["Model (setting)", "GPQA Diamond (%)", "SE", "95% low", "95% high", "In the report", "Source"]


def fig_abstract(p: Paper, f: Path) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(open(f, encoding="utf-8")))
    if not rows or rows[0] != ABSTRACT_HEAD:
        raise SystemExit(f"the figure under the abstract: {f.name} doesn't have the columns "
                         f"{', '.join(ABSTRACT_HEAD)}")
    epoch = {r["label"]: r for r in epoch_rows(p.epoch("GPQA Diamond"))}
    out, last = [], None
    for r in rows[1:]:
        name, v, se, lo, hi, where, who = r
        m = re.fullmatch(r"Masein model, (High|Medium|Instant)", name)
        if m:
            want, table, kind = p.masein("GPQA Diamond", m.group(1)), "Table 5", OURS
        elif name in epoch:
            want, table, kind = (epoch[name]["v"], epoch[name]["se"]), "Table F1", EPOCH
        else:
            raise SystemExit(f"the figure under the abstract: {name} is neither a mode in the scores table "
                             "nor a row of Table F1")
        if (v, se) != want:
            raise SystemExit(f"the figure under the abstract: {name} is drawn at {v} ± {se}; {table} prints "
                             f"{want[0]} ± {want[1]}")
        span = (f"{float(v) - 1.96 * float(se):.1f}", f"{float(v) + 1.96 * float(se):.1f}")
        if (lo, hi) != span:
            raise SystemExit(f"the figure under the abstract: {name}'s bar is {lo} to {hi}, not ±1.96 SE of "
                             f"what {table} prints ({span[0]} to {span[1]})")
        if where != table or not who.startswith(kind.split(",")[0]):
            raise SystemExit(f"the figure under the abstract: {name} says {where!r}, {who!r}; it is {table}'s, "
                             f"{kind}")
        if last is not None and float(v) > last:
            raise SystemExit(f"the figure under the abstract: {name} is out of order (highest first)")
        last = float(v)
        out.append([name, v, se, lo, hi, table, kind])
    return ABSTRACT_HEAD, out


def check_phone_speeds(p: Paper, his: list[dict]) -> None:
    """r29: each mode's iPhone short-chat speed the key numbers print is the decode median of
    Kevin's short-chat table (tab:phone-modes) — the number Figure 1 plots and credits to it"""
    t = next((x for x in his if x["label"] == KEVIN_LABELS["P1"]), None)
    col = next((k for k, h in enumerate(t["head"]) if h.startswith("Decode")), None) if t else None
    if t is None or col is None or "Mode" not in t["head"]:
        raise SystemExit(f"Kevin's file: {KEVIN_LABELS['P1']} has no Device, Mode and Decode columns")
    mode_col = t["head"].index("Mode")
    for mode in APP_MODES:
        try:
            want, _ = masein_speed(p, mode)
        except SystemExit:
            continue                                # a mode the key numbers give no speed for
        got = [r[col] for r in t["rows"] if r[0] == PHONE and r[mode_col] == mode]
        if len(got) != 1 or num(got[0]) != want:
            raise SystemExit(f"the key numbers print {want} tokens/s for {mode} mode on the {PHONE}; "
                             f"Kevin's short-chat table has {got or 'no row'}")


def board_speed(model: str) -> float:
    """the model's decode tokens/s on DeviceMark's board (the repo's copy, eval_tasks/devicemark)"""
    if not BOARD.exists():
        raise SystemExit(f"the speed chart: DeviceMark's board ({BOARD}) is missing")
    got = [r["iphone_tok_s"] for r in json.loads(BOARD.read_text(encoding="utf-8"))["rows"]
           if r["model"].lower() == model.lower() and r.get("iphone_tok_s") is not None]
    if len(got) != 1:
        raise SystemExit(f"the speed chart: {model} is {len(got)} times on DeviceMark's board with an "
                         "iPhone speed, not once")
    return float(got[0])


def fig_speed(p: Paper, cap: str, src):
    """DeviceMark composite against token generation throughput on a phone: each score must be Table 12's
    as printed; another model's speed must be DeviceMark's, as its board gives it and credited,
    and each of the Masein model's must be the iPhone 17 Pro Max's short-chat median the key
    numbers print for its mode"""
    f = src.speeds
    if not f.exists():
        raise SystemExit(f"the speed chart: {f} is missing (the figure script writes it)")
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    if not rows or list(rows[0].keys()) != SPEED_HEAD:
        raise SystemExit(f"the speed chart: {f} doesn't have the columns {', '.join(SPEED_HEAD)}")
    t = p.find("DeviceMark's rows", lambda t: any(h.startswith("Composite") for h in t["head"]))
    k = next(i for i, h in enumerate(t["head"]) if h.startswith("Composite"))
    by = {clean(r[0]): r for r in t["rows"]}
    out = []
    for r in rows:
        name = r["model"] + (f", {r['mode']}" if r["mode"] else "")
        if name.startswith("Masein model") and not (r["speed_tok_s"] or r["speed_source"] or r["speed_credit"]):
            # r23 (masein, 14 Nov): a mode not plotted yet (High, until the app's own number) isn't
            # checked against the paper or written: its score goes in when it is drawn
            continue
        raw = by.get(name)
        if raw is None:
            raise SystemExit(f"the speed chart: {name} is not a row of {t['name']}")
        m = RANGE.match(clean(raw[k]))
        if (num(m.group(1)), num(m.group(2)), num(m.group(3))) != (r["devicemark_composite"], r["ci95_low"],
                                                                   r["ci95_high"]):
            raise SystemExit(f"the speed chart: {name} is plotted at {r['devicemark_composite']} "
                             f"[{r['ci95_low']}, {r['ci95_high']}]; {t['name']} prints {clean(raw[k])}")
        if not run_by_us(t, raw):
            if r["speed_tok_s"]:
                raise SystemExit(f"the speed chart: {name} is a published row and has no speed")
            out.append([name, num(m.group(1)), num(m.group(2)), num(m.group(3)), DEVICEMARK, "", "", ""])
            continue
        mine = name.startswith("Masein model")
        if not r["speed_tok_s"] or not is_num(r["speed_tok_s"]):
            raise SystemExit(f"the speed chart: {name} has no speed")
        if not mine:
            if not (r["speed_source"].startswith("DeviceMark") and r["speed_credit"] == DEVICEMARK):
                raise SystemExit(f"the speed chart: {name}'s speed must be DeviceMark's, credited "
                                 f"\"{DEVICEMARK}\"")
            if float(r["speed_tok_s"]) != board_speed(r["model"]):
                raise SystemExit(f"the speed chart: {name} is plotted at {r['speed_tok_s']} tokens/s; "
                                 f"DeviceMark's board gives {board_speed(r['model']):g}")
        elif r["speed_credit"] != OURS:
            raise SystemExit(f"the speed chart: {name}'s speed is the app's, so it is {OURS}")
        if mine:
            if PHONE not in r["speed_source"]:
                raise SystemExit(f"the speed chart: {name}'s speed_source must name the phone, {PHONE}")
            modes = [x for x in APP_MODES if f"{x} mode" in r["speed_source"]]
            if len(modes) != 1:
                raise SystemExit(f"the speed chart: {name}'s speed_source must name its mode, one of "
                                 f"{', '.join(x + ' mode' for x in APP_MODES)}")
            want, early = masein_speed(p, modes[0])
            if num(r["speed_tok_s"]) != want:
                raise SystemExit(f"the speed chart: {name} is plotted at {r['speed_tok_s']} tokens/s; the key "
                                 f"numbers print {want} for {modes[0]} mode on the {PHONE} in short chats")
            if early and "preliminary" not in r["speed_source"]:
                raise SystemExit(f"the speed chart: the key numbers call {modes[0]} mode's throughput "
                                 f"preliminary, and {name}'s speed_source must say so")
            if not early and "preliminary" in r["speed_source"]:
                raise SystemExit(f"the speed chart: {name}'s speed_source calls it preliminary, and the "
                                 f"key numbers don't: {modes[0]} mode's throughput is final")
        out.append([name, num(m.group(1)), num(m.group(2)), num(m.group(3)), OURS, r["speed_tok_s"],
                    r["speed_source"], r["speed_credit"]])
    return ["Model", "DeviceMark composite", "95% low", "95% high", "Score: whose", "Token generation throughput (tokens/s)",
            "Speed: how it was measured", "Speed: whose"], out


# r37 (masein, 16 Nov): three figures, md ids 12 to 14. Each is built from its own sources here,
# its caption (and for Figure 12 Section 2.1's text) held to them, and the figure script's CSV
# for it — the numbers it drew, beside figure-1.csv as the abstract figure's is — held to the same

def drawn(src: Path | Sources, md_id: str, head: list[str]) -> list[list[str]]:
    """the figure script's CSV of md Figure `md_id`, beside figure-1.csv: its rows under `head`"""
    f = src.speeds.parent / f"figure-{md_id}.csv"
    if not f.exists():
        raise SystemExit(f"Figure {md_id}: no figure-{md_id}.csv beside {src.speeds.name} "
                         f"({src.speeds.parent}) — the figure script's numbers, which this build checks")
    rows = list(csv.reader(open(f, encoding="utf-8")))
    if not rows or rows[0] != head:
        raise SystemExit(f"Figure {md_id}: {f.name} doesn't have the columns {', '.join(head)}")
    return rows[1:]


def half_up(x: float, places: int = 0) -> str:
    """as the paper rounds: 62.5 is 63"""
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


# Figure 12: the routed-expert weights a token reads, from the evaluated file's tensor sizes
TENSORS = "step2388-tensors.tsv"
EXPERT_MODES = [("Instant", 3), ("Medium and High", 6), ("Base model, Qwen3.6-35B-A3B", 8)]
EXPERTS_HEAD = ["Mode", "Experts per layer", "Routed-expert weights read per token (bytes)",
                "Routed-expert weights read per token (MB)", "Less than the base model (%)", "Source"]


def expert_reads(tsv: Path) -> dict:
    """from the file's header as read (name, type, shape, elements, bytes a tensor): one routed
    expert's bytes in every layer, and what a token reads besides — every other tensor, one
    mode's own copies (.k3 or .k6, the same size), one row of token_embd for the whole table"""
    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
    size = {r["name"]: int(r["bytes"]) for r in rows}
    shape = {r["name"]: r["shape (ggml order, ne0 first)"] for r in rows}
    exps = [n for n in size if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", n)]
    layers = {n.split(".")[1] for n in exps}
    counts = {int(shape[n].split("×")[-1]) for n in exps}
    if len(exps) != 3 * len(layers) or len(counts) != 1:
        raise SystemExit(f"{tsv.name}: the routed experts aren't 3 tensors a layer of one expert count")
    (n_exp,) = counts
    routed = sum(size[n] for n in exps)
    if routed % n_exp:
        raise SystemExit(f"{tsv.name}: the routed experts' {routed:,} bytes don't split into {n_exp}")
    k6 = sum(v for n, v in size.items() if ".k6." in n or n.endswith(".k6"))
    k3 = sum(v for n, v in size.items() if ".k3." in n or n.endswith(".k3"))
    if k3 != k6:
        raise SystemExit(f"{tsv.name}: the two modes' own copies differ ({k3:,} and {k6:,} bytes)")
    vocab = int(shape["token_embd.weight"].split("×")[-1])
    row = size["token_embd.weight"] // vocab
    rest = sum(size.values()) - routed - k3 - size["token_embd.weight"] + row
    return {"per_expert": routed // n_exp, "experts": n_exp, "layers": len(layers), "rest": rest}


def fig_experts(p: Paper, cap: str, src: Sources):
    r = expert_reads(src.figures / TENSORS)
    base = 8 * r["per_expert"]
    out, said = [], {}
    for mode, k in EXPERT_MODES:
        b = k * r["per_expert"]
        mb, less = half_up(b / 1e6), "" if k == 8 else half_up(100 * (1 - b / base))
        said[k] = (mb, less)
        out.append([mode, str(k), str(b), mb, less, "the evaluated file's tensor sizes, Section 2.1"])
    gb = half_up(r["rest"] / 1e9, 2)
    # the text (Section 2.1), the caption and the figure say the same
    text = " ".join(p.lines)
    for want in (f"{said[3][0]} or {said[6][0]} MB", f"{said[8][0]} MB", f"about {gb} GB"):
        if want not in text:
            raise SystemExit(f"Figure 12: Section 2.1 doesn't say {want!r}, which the file's tensor sizes give")
    for want in (f"{said[3][0]} MB", f"{said[3][1]}% less than the {said[8][0]} MB", f"{said[6][0]} MB",
                 f"{said[6][1]}% less", f"{gb} GB"):
        if want not in cap:
            raise SystemExit(f"Figure 12: its caption doesn't say {want!r}, which the file's tensor sizes give")
    other = sorted(set(re.findall(r"(\d+) MB", cap)) - {said[k][0] for k in (3, 6, 8)})
    if other:
        raise SystemExit(f"Figure 12: its caption names {', '.join(other)} MB, which the file's tensor sizes don't give")
    got = [x[:5] for x in drawn(src, "12", EXPERTS_HEAD)]
    if got != [x[:5] for x in out]:
        raise SystemExit(f"Figure 12: the figure script drew {got}; the file's tensor sizes give "
                         f"{[x[:5] for x in out]}")
    out.append(["Everything else a token reads, in every mode (one row of the embedding table)", "",
                str(r["rest"]), "", "", f"the evaluated file's tensor sizes, Section 2.1 (about {gb} GB)"])
    return EXPERTS_HEAD, out


# Figure 13: what thinking adds — the three modes on the five benchmarks run in all three, from
# right answers whose shares round to Table 5's printed scores; High over the better of the
# two thinking-off modes from the unrounded shares, printed to one decimal
MODES_FILE = "modes-scores.json"
THREE_MODES = ("High", "Medium", "Instant")
MODES_HEAD = ["Benchmark", "Answers", "High, right", "Medium, right", "Instant, right", "High (%)", "Medium (%)",
              "Instant (%)", "High over the best thinking-off (points)", "Source"]


def fig_modes(p: Paper, cap: str, src: Sources):
    data = json.loads((src.figures / MODES_FILE).read_text(encoding="utf-8"))["benchmarks"]
    t = p.find("the scores in its modes", lambda t: all(f"{m} (%)" in t["head"] for m in THREE_MODES))
    in_all = [clean(r[0]) for r in t["rows"] if not clean(r[0]).startswith("DeviceMark")
              and all(PM.match(clean(r[t["head"].index(f"{m} (%)")])) for m in THREE_MODES)]
    if sorted(b["name"] for b in data) != sorted(in_all):
        raise SystemExit(f"Figure 13: {MODES_FILE} has {', '.join(b['name'] for b in data)}; the scores table "
                         f"runs {', '.join(in_all)} in all three modes")
    out, gains = [], {}
    for b in data:
        share = {m: 100 * b[m] / b["answers"] for m in THREE_MODES}
        for m in THREE_MODES:
            printed = p.masein(b["name"], m)[0]
            if f"{share[m]:.1f}" != printed:
                raise SystemExit(f"Figure 13: {b['name']}, {m}: {b[m]} of {b['answers']} is {share[m]:.2f}%; "
                                 f"the scores table prints {printed}")
        gains[b["name"]] = share["High"] - max(share["Medium"], share["Instant"])
        out.append([b["name"], str(b["answers"]), *(str(b[m]) for m in THREE_MODES), *(f"{share[m]:.1f}" for m in THREE_MODES),
                    f"{gains[b['name']]:+.1f}", f"{OURS}, Table 5"])
    # the caption's numbers: instruction following, then the range of the other three
    best = {b["name"]: max(b["Medium"], b["Instant"]) / b["answers"] * 100 for b in data}
    high = {b["name"]: b["High"] / b["answers"] * 100 for b in data}
    follow = [n for n in ("IFBench", "IFEval") if n in gains]
    for name in follow:
        want = f"{high[name]:.1f} against {best[name]:.1f}"
        if want not in cap:
            raise SystemExit(f"Figure 13: its caption doesn't say {want!r} for {name}")
    rest = [g for n, g in gains.items() if n not in follow]
    lo, hi = f"{min(rest):.1f}", f"{max(rest):.1f}"
    want = f"adds {lo} points" if lo == hi else f"adds {lo} to {hi} points"
    if want not in cap:
        raise SystemExit(f"Figure 13: its caption doesn't say {want!r}, the other three's gains")
    got = drawn(src, "13", MODES_HEAD)
    if [x[:9] for x in got] != [x[:9] for x in out]:
        raise SystemExit(f"Figure 13: the figure script drew {[x[:9] for x in got]}; Table 5's counts give "
                         f"{[x[:9] for x in out]}")
    return MODES_HEAD, out


# Figure 14: the hosted models' output list prices (Table 11's), lowest first, under the names the
# figure gives them; the Masein model's row first, as the table has it
PRICE_NAMES = {"deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash", "openai/gpt-5-mini": "GPT-5 mini",
               "openai/o4-mini": "o4-mini", "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
               "openai/gpt-6.1-sol": "GPT-6.1 Sol", "google/gemini-4-argon": "Gemini 4 Argon",
               "anthropic/claude-opus-5.5": "Claude Opus 5.5", "openai/gpt-6-astra": "GPT-6 Astra"}
PRICES_HEAD = ["Model", "OpenRouter id", "Output ($ per million tokens)", "Source"]
NO_CHARGE = "no per-token charge"


def fig_prices(p: Paper, cap: str, src: Sources):
    t = p.find("the list prices", lambda t: t["head"][:1] == ["Model (OpenRouter id)"]
               and "Output ($ per million tokens)" in t["head"])
    k = t["head"].index("Output ($ per million tokens)")
    rows = [(clean(r[0]), clean(r[k])) for r in t["rows"]]
    mine = [r for r in rows if r[1] == NO_CHARGE]
    hosted = [r for r in rows if r[1] != NO_CHARGE]
    if len(mine) != 1 or not mine[0][0].startswith("Masein model"):
        raise SystemExit("Figure 14: the price table has no one Masein model row with no per-token charge")
    unnamed = [i for i, _ in hosted if i not in PRICE_NAMES]
    if unnamed:
        raise SystemExit(f"Figure 14: no name for {', '.join(unnamed)} (PRICE_NAMES)")
    read = {i: "1 Nov 2026" if i in ("openai/gpt-6.1-sol", "openai/gpt-6-astra") else "29 Oct 2026" for i, _ in hosted}
    for d in set(read.values()):
        if d.replace(" 2026", "") not in t["caption"]:
            raise SystemExit(f"Figure 14: the price table's caption doesn't say {d}, when its prices were read")
    hosted = sorted(hosted, key=lambda r: float(r[1]))          # lowest first; ties as the table has them
    out = [["Masein model, on the phone", "", NO_CHARGE, "Table 11"]]
    out += [[PRICE_NAMES[i], i, v, f"OpenRouter model page, read {read[i]} (Table 11)"] for i, v in hosted]
    lo, hi = float(hosted[0][1]), float(hosted[-1][1])
    for want in (f"${lo:.2f}", f"${hi:g}", NO_CHARGE):
        if want not in cap:
            raise SystemExit(f"Figure 14: its caption doesn't say {want!r}, as the price table gives it")
    got = drawn(src, "14", PRICES_HEAD)
    if [x[:3] for x in got] != [x[:3] for x in out]:
        raise SystemExit(f"Figure 14: the figure script drew {[x[:3] for x in got]}; the price table gives "
                         f"{[x[:3] for x in out]}")
    return PRICES_HEAD, out


def is_speed_chart(cap: str) -> bool:
    # r35: "…84.4 on the DeviceMark replica at 11.4 tokens per second on an iPhone 17 Pro Max…"
    return any(k in cap.lower() for k in ("token generation throughput", "tokens per second")) and "DeviceMark" in cap


FIGURES = [  # phrases of the caption (any one will do), and what builds the figure's data
    ((is_speed_chart,), fig_speed),
    (("by release date",), fig_release),
    (("How one question becomes a score",), None),
    (("right on every run, on some runs and on none",), fig_reliability),
    (("Mean thinking tokens per finished answer",), fig_thinking),
    (("share of answers that ran out of room at the final limit",), fig_ranout),
    (("Each dot is one model's best-scoring Epoch run", "in the lane of its verdict"), fig_gpqa),
    (("The bars count the models", "Each bar counts the models"), fig_counts),
    (("one model per row", "Each cell is the model's best-scoring Epoch run"), fig_grid),   # r35: the second
    (("Score against file size", "Score against deployed file size"), fig_size),   # r34: "deployed"
    (("DeviceMark composite with its 95% range",), fig_devicemark),
    # r37: md ids 12 to 14
    (("of routed-expert weights from the file",), fig_experts),
    (("High minus the better of Medium and Instant",), fig_modes),
    (("output list prices",), fig_prices),
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
    for flag in ("--figures", "--speeds", "--phone-tex", "--app-limits"):
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
    his = kevin_tables(opt["--phone-tex"]) if "--phone-tex" in opt else None
    nums = Numbers(lines, tabs, his)
    if nums.stand_in:
        print(f"Section 3 is Kevin's own file in LaTeX: the md's {', '.join(nums.stand_in)} there "
              "stand in for it and print nowhere, so they are not built")
    if nums.named and his is None:
        print(f"Kevin's Tables P1 to P{nums.named} (printed 1 to {nums.named}) are read from his own "
              f"file, and none was given: no table-1 to table-{nums.named}" + ("" if trial else "."))
        if not trial:
            print("stopped: nothing written. Give his file (on-the-phone.tex, from Overleaf) with "
                  "--phone-tex, or leave his tables out on purpose with --trial")
            return 1
    tabs = [x for x in tabs if not nums.skipped(x)]
    built, index = [], []
    # his tables, as his file prints them, numbered 1 to n in its order
    for i, x in enumerate(his if nums.named and his else []):
        built.append((f"table-{i + 1}", x["head"], x["rows"]))
        index.append([f"table-{i + 1}.csv", f"Table {i + 1}", first_sentence(x["caption"])])
    for t in tabs:
        printed = {**t, "name": nums.name(t["name"])}
        stem, h, rows = table_file(printed)
        check_table(stem, t, rows)
        built.append((stem, h, rows))
        index.append([f"{stem}.csv", printed["name"] + (", its second part" if t["part"] > 1 else ""),
                      first_sentence(t["caption"]) if t["caption"] else "The report's key numbers."])
    p = Paper(tabs, lines)
    if nums.named and his:
        check_phone_speeds(p, his)
    check_app_limits(p, opt.get("--app-limits"))
    caps = figures(lines)
    figs = FigNumbers(caps)
    for n, cap in caps:
        fn = builder(cap)
        if fn == "?":
            raise SystemExit(f"Figure {n} is new to this script: {cap[:90]}")
        pn = figs.number(n)
        if fn is None:
            print(f"figure-{pn}: a diagram, no data" + (f" (the md's Figure {n})" if pn != n else ""))
            continue
        built.append((f"figure-{pn}", *fn(p, cap, src)))
        index.append([f"figure-{pn}.csv", f"Figure {pn}", first_sentence(cap)])
    # r31: the figure under the abstract, unnumbered, when the figure script's file is there
    absf = src.speeds.parent / ABSTRACT
    if absf.exists():
        built.append(("figure-abstract", *fig_abstract(p, absf)))
        index.append([ABSTRACT, "The figure under the abstract", "GPQA Diamond: the Masein model's three modes "
                      "among three of Epoch AI's models, each score with its 95% range."])
    else:
        print(f"no {ABSTRACT} beside {src.speeds.name} ({src.speeds.parent}): the figure under the abstract "
              "is not built")
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.csv"):
        f.unlink()
    # r29: every table reference in what goes out, as LaTeX prints it — r37: and every figure's
    def printed_refs(s: str) -> str:
        return figs.text(nums.text(s))
    for stem, h, rows in built:
        write(out / f"{stem}.csv", [printed_refs(x) for x in h], [[printed_refs(x) for x in r] for r in rows])
    # the name is the printed one already; the caption's sentence is the md's
    write(out / "index.csv", ["File", "In the report", "What it holds (the caption's first sentence)"],
          [[f, name, printed_refs(said)] for f, name, said in index])
    print(f"{len(built)} files in {out}: " + ", ".join(s for s, _, _ in built))
    tbd = [s for s, h, rows in built if any("TBD" in c for r in [h, *rows] for c in r)]
    if tbd:
        print("files that carry a TBD from the paper: " + ", ".join(tbd))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
