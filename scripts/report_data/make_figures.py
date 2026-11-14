#!/usr/bin/env python3
"""The report's eleven figures in the house style of round 19 (11 Nov 2026).

    python3 make_figures.py <data dir> <fonts dir> <out dir>

<data dir> holds epoch-recompute.json (Epoch AI's lists with every verdict), figure-8.csv
(the 21-row grid) and figure-10.csv (DeviceMark's 27 rows). <fonts dir> holds
texgyreheros-regular.otf and texgyreheros-bold.otf; their TrueType versions are written to
<fonts dir>/ttf/ and those are the ones embedded. <out dir> gets figure-N.pdf,
figure-N.png (220 dpi) and figure-N.csv for N = 1 to 11, and contact-sheet.png.

Figure 1's speeds are read from <out dir>/figure-1.csv when it is there (speed_tok_s and
speed_source, by model and mode), so that new timings need one rebuild; without it, the
other models' come from DeviceMark's published board as the repo keeps it
(eval_tasks/devicemark/board-snapshot.json: decode tokens/s on an iPhone 17 Pro) and the
Masein model's from the values below.

The style: TeX Gyre Heros at 8.5 pt (panel titles 9.5 pt bold), 6.5 in wide, the paper's
text width; one colour, one meaning, in every figure; text in ink or muted grey only;
legends in one row above the plot; filled marks are measured by us and hollow ones
published by someone else; squares are runs below high effort (†) and rings verdicts
within a quarter of a combined standard error of the line (‡); error bars only where the
paper gives one or its scores give one (score ± 1.96 × SE). Every figure is checked before it is saved: no label overlaps
another label, a point, a line or an error bar, nothing leaves the page, no text is under
8.5 pt, and the numbers the captions state are the numbers drawn."""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

WIDTH = 6.5                                    # in: the paper's text width
BLUE, ORANGE, GREEN, MUSTARD = "#3D75D5", "#E07040", "#52AA7C", "#E4A338"
LEVEL, CONTEXT = "#C9CDD3", "#9A9A9A"
INK, MUTED, GRID, AXIS = "#222222", "#666666", "#EBEBEB", "#9A9A9A"
SMALL, TITLE = 8.5, 9.5
EBAR = dict(elinewidth=0.8, capsize=2, capthick=0.8)


def tint(colour: str, share: float) -> str:
    """the colour mixed with white: share 1 is the colour, 0 is white"""
    c = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - x) * share):02X}" for x in c)


BLUE25, ORANGE25, BLUE45 = tint(BLUE, 0.25), tint(ORANGE, 0.25), tint(BLUE, 0.45)
EPOCH = "Epoch AI Benchmarking Hub, CC BY 4.0, imported 7 Nov 2026"
DEVICEMARK = "DeviceMark, devicemark.github.io, CC BY 4.0"
OURS = "measured by Teraformer"
# r25 (14 Nov): the other models' speeds are DeviceMark's (CC BY 4.0), as the data build credits them
DM_SPEED_CREDIT = "DeviceMark (devicemark.github.io), CC BY 4.0, published"

# ---------------------------------------------------------------------------
# the numbers the figures plot that are not in the data files: each as the paper prints it
# ---------------------------------------------------------------------------
RELIABILITY = [("GPQA Diamond", 198, 4, (141, 44, 13)), ("OTIS Mock AIME", 45, 8, (32, 8, 5))]
THINKING = [  # benchmark, finished answers right / wrong, thinking tokens right / wrong (Table 7)
    ("GPQA Diamond", 655, 130, 10249801, 3249334),
    ("MMLU-Pro", 9983, 2049, 24042649, 7024913),
    ("OTIS Mock AIME", 302, 36, 9231738, 2535585),
    ("MATH Level 5", 1304, 14, 18696721, 480579)]
RAN_OUT = [("OTIS Mock AIME", 22, 360), ("IFBench", 15, 300), ("IFEval", 7, 541), ("GPQA Diamond", 7, 792),
           ("MATH Level 5", 6, 1324), ("MMLU-Pro", 0, 12032)]
ON_DEVICE = [  # Table G1: name, size GB, GPQA Diamond, MMLU-Pro, IFEval, IFBench, source
    ("Edge0-35B", 19.6, 79.8, 81.0, None, 57.9, "Edge0"),
    ("Edge0-8B", 4.55, 70.7, 70.1, None, 53.9, "Edge0"),
    ("Qwen3.5-9B", 5.68, 81.7, 82.5, 91.5, 64.5, "Qwen"),
    ("Qwen3.5-4B", 2.74, 76.2, 79.1, 89.8, 59.2, "Qwen"),
    ("Ternary Bonsai 2 27B", 7.21, None, None, 91.31, 74.00, "PrismML"),
    ("Ternary Bonsai 2 9B", 2.0, 69.9, None, 90.0, 57.0, "PrismML"),
    ("Ternary Bonsai 2 4B", 0.95, 63.8, None, 88.9, 56.3, "PrismML"),
    ("Ternary Bonsai 2 2B", 0.43, 39.6, None, 85.8, 48.0, "PrismML"),
    ("Bonsai 27B (1-bit)", 3.9, None, None, 79.1, 52.4, "PrismML"),
    ("Gemma 4 E4B", 2.3, 58.6, 69.4, 96.7, 44.0, "Google"),
    ("Gemma 4 E2B", 0.8, 43.4, 60.0, 94.6, 38.0, "Google"),
    ("LFM3-1.7B", 1.0, None, None, 96.66, None, "Liquid AI"),
    ("Bolt", 6.18, None, None, None, 64.00, "NVIDIA")]
NV_QWEN4B = ("Qwen3.5-4B, measured by NVIDIA", 2.74, 66.67)
MASEIN_ON_DEVICE = {"GPQA Diamond": "82.7 ± 2.2", "MMLU-Pro": "83.0 ± 0.3", "IFEval": "90.6 ± 1.3",
                    "IFBench": "63.7 ± 2.8"}
MASEIN_SIZE = 16.0
PARAMS_MASEIN = "34.8B / 2.3B or 2.7B"            # Table F4

# Figure 1: each other model in its higher-scoring mode on our DeviceMark replica (Table 12), the
# Masein model in its modes, and the speed each writes at on a phone. Label (Table 12's row name,
# which figure-1.csv keeps), its DeviceMark row (dm_rows), its name on DeviceMark's board.
# r25 (14 Nov, masein): the eight models we ran on the replica, each in its higher-scoring mode,
# at DeviceMark's published decode speed on an iPhone 17 Pro (CC BY 4.0)
SPEED_ROWS = [("Masein model, Instant", "Masein model, Instant", None),
              ("Masein model, Medium", "Masein model, Medium", None),
              ("Masein model, High", None, None),
              ("Qwen3.5-4B, thinking off", "Qwen3.5-4B, thinking off", "Qwen3.5-4B"),
              ("Gemma 4 E2B, thinking on", "gemma-4-E2B-it, thinking on", "Gemma 4 E2B"),
              ("Youtu-LLM-2B, thinking off", "Youtu-LLM-2B, thinking off", "Youtu-LLM-2B"),
              ("LFM2.5-1.2B", "LFM2.5-1.2B-Instruct, thinking off", "LFM2.5-1.2B"),
              ("Qwen3.5-2B, thinking off", "Qwen3.5-2B, thinking off", "Qwen3.5-2B"),
              ("granite-4.0-h-1b", "granite-4.0-h-1b, thinking off", "Granite-4.0-H-1B"),
              ("Nanbeige4.1-3B", "Nanbeige4.1-3B, thinking off", "Nanbeige4.1-3B"),
              ("Qwen3.5-0.8B, thinking off", "Qwen3.5-0.8B, thinking off", "Qwen3.5-0.8B")]
SNAPSHOT = Path(__file__).resolve().parents[2] / "eval_tasks" / "devicemark" / "board-snapshot.json"
RUNTIME_WORDS = {"coreai": "Core AI", "litertlm": "LiteRT-LM"}
# r25 (14 Nov, masein): the Masein model on the iPhone 17 Pro Max, the app's short-chat medians (Table 1)
MASEIN_SPEED = ("11.4", "Teraformer's app on an iPhone 17 Pro Max (12 GB), Instant mode, short chats, median, "
                        "on the 16.0 GB file (Table 1); measured by the app's developers")
MEDIUM_SPEED = ("9.7", "Teraformer's app on an iPhone 17 Pro Max (12 GB), Medium mode, short chats, median, "
                       "preliminary (Table 1); measured by the app's developers")
OWN_SPEEDS = {"Masein model, Instant": MASEIN_SPEED, "Masein model, Medium": MEDIUM_SPEED}
# High at the app's limits (12 Nov), not Table 12's High row (the replica's 4,096-token cap): kept
# in figure-1.csv with no speed and not drawn until the app's own short-chat number comes (not
# the shell's, which runs faster than the app); its score goes in the paper when it is drawn
OWN_SCORES = {"Masein model, High": ("78.7", "75.4", "82.0")}
# a point drawn under another name than its Table 12 row's
SHOWN = {"Masein model, Medium": "Masein model, Medium (preliminary)"}
# a Masein point whose label goes up, at this height (Figure 1's y axis)
MINE_UP = {"Masein model, Medium": 98.2}
# r23 (14 Nov): figure-10.csv (round 13's DeviceMark rows) under Table 12's names, with Medium's
# replica row of 12 Nov (M4; the replica asks greedily with no presence penalty, so it stands
# for Medium at 0.75 too): 28 rows
DM_RENAMED = {"Masein model, thinking off": "Masein model, Instant", "Masein model, thinking on": "Masein model, High"}
DM_ADDED = [{"Model": "Masein model, Medium", "Kind": "Masein", "Composite": "84.6", "95% low": "81.6",
             "95% high": "87.3", "Verdict vs Masein off": "", "Verdict vs Masein on": "",
             "Source": "measured by Teraformer"}]
CLOUD = ["Gemini Flash (cloud)", "Gemini Pro (cloud)"]

# Figure 7 (GPQA Diamond lanes): the models the text names
NAMED_GPQA = ["o1", "o3-mini", "o4-mini", "Qwen 3.6 35B-A3B", "GPT-5 mini", "Claude Haiku 4.5", "Gemini 4 Argon",
              "GPT-6 Astra", "Claude Sonnet 5.5", "DeepSeek V4.1 Flash 1017"]
ALIAS = {"Qwen 3.6 35B-A3B": "Qwen3.6-35B-A3B (base model)"}
# figure-8.csv's names for Epoch's (the writer's labels in Tables F1 to F3)
CSV_NAME = {"Grok-3 mini": "Grok 3 mini", "DeepSeek-R1 (May 2025)": "DeepSeek-R1-0528",
            "Qwen3-235B-A22B-Thinking (Jul 2025)": "Qwen3-235B-A22B-Thinking-2507",
            "Qwen3-30B-A3B-Thinking (Jul 2025)": "Qwen3-30B-A3B-Thinking-2507", "Qwen3.5 397B-A17B": "Qwen3.5-397B-A17B",
            "glm-4.7-flash": "GLM-4.7-Flash", "DeepSeek-V3 (Mar 2025)": "DeepSeek-V3-0324",
            "Gemini 2.0 Flash (Feb 2025)": "Gemini 2.0 Flash", "Claude 3.5 Sonnet (October 2024)": "Claude 3.5 Sonnet (Oct 2024)",
            "Gemini 2.5 Flash (Apr 2025)": "Gemini 2.5 Flash (Apr preview)", "Tulu 3 (Tülu 3) 70B": "Tülu 3 70B"}
DEVICEMARK_NAMES = {
    "gemma-4-E2B-it, thinking on": "Gemma 4 E2B, thinking on", "gemma-4-E2B-it, thinking off": "Gemma 4 E2B, thinking off",
    "LFM2.5-1.2B-Instruct, thinking off": "LFM2.5-1.2B", "granite-4.0-h-1b, thinking off": "granite-4.0-h-1b",
    "Nanbeige4.1-3B, thinking off": "Nanbeige4.1-3B", "Qwen3.5-0.8B": "Qwen3.5-0.8B (int8)", "Qwen3.5-2B": "Qwen3.5-2B (int8)",
    "LFM2.5-1.2B": "LFM2.5-1.2B (int8)", "Granite-4.0-H-1B": "Granite-4.0-H-1B (int8)", "Qwen3.5-4B": "Qwen3.5-4B (int8)",
    "Youtu-LLM-2B": "Youtu-LLM-2B (int8)", "Gemma 4 E2B": "Gemma 4 E2B (int4)", "Nemotron-3-Nano-4B": "Nemotron-3-Nano-4B (int8)",
    "Nanbeige4.1-3B": "Nanbeige4.1-3B (int8)"}


def pm(x: float) -> str:
    return f"{x:+.1f}".replace("-", "−")


def d(s: str) -> date:
    y, mo, da = (int(x) for x in s[:10].split("-"))
    return date(y, mo, da)


def name_of(r: dict) -> str:
    return ALIAS.get(r["model"], r["model"])


def setting_words(s: str) -> str:
    return {"none": "thinking off", "setting not stated": ""}.get(s, s)


def label_of(r: dict) -> str:
    s = setting_words(r["setting"])
    return name_of(r) + (f", {s}" if s and "base model" in name_of(r) else (f" ({s})" if s else "")) + (" †" if r["dagger"] else "")


# ---------------------------------------------------------------------------
# style and checks
# ---------------------------------------------------------------------------

def truetype(otf: Path, ttf: Path) -> Path:
    """the OTF's CFF outlines as TrueType quadratics (fontTools' otf2ttf recipe with cu2qu,
    error at most 1 font unit in 1,000): with pdf.fonttype 42 the PDF then embeds a
    TrueType font in its TrueType slot, never CFF outlines there. Names, metrics and
    advance widths are the font's own; written once, and again only when the OTF is newer"""
    if ttf.exists() and ttf.stat().st_mtime >= otf.stat().st_mtime:
        return ttf
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    f = TTFont(otf, recalcTimestamp=False)
    order, gs = f.getGlyphOrder(), f.getGlyphSet()
    f["loca"] = newTable("loca")
    f["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for g in order:
        pen = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[g] = pen.glyph()
    del f["CFF "]
    if "VORG" in f:
        del f["VORG"]
    glyf.compile(f)
    for g, gl in glyf.glyphs.items():
        if hasattr(gl, "xMin"):
            f["hmtx"][g] = (f["hmtx"][g][0], gl.xMin)
    f["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones, maxp.maxTwilightPoints, maxp.maxStorage, maxp.maxFunctionDefs = 1, 0, 0, 0
    maxp.maxInstructionDefs, maxp.maxStackElements, maxp.maxSizeOfInstructions = 0, 0, 0
    maxp.maxComponentElements = max(len(getattr(gl, "components", [])) for gl in glyf.glyphs.values())
    maxp.compile(f)
    post = f["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    f["head"].glyphDataFormat = 0
    f.sfntVersion = "\000\001\000\000"
    ttf.parent.mkdir(parents=True, exist_ok=True)
    f.save(ttf)
    return ttf


def setup(fonts: Path) -> None:
    for f in ("texgyreheros-regular", "texgyreheros-bold"):
        font_manager.fontManager.addfont(str(truetype(fonts / f"{f}.otf", fonts / "ttf" / f"{f}.ttf")))
    plt.rcParams.update({
        "font.family": "TeX Gyre Heros", "font.size": SMALL, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
        "axes.titlesize": TITLE, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.titlepad": 4, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": MUTED,
        "axes.labelsize": SMALL, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": SMALL, "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.6, "legend.borderaxespad": 0, "legend.borderpad": 0,
        "savefig.dpi": 220, "figure.dpi": 220, "lines.solid_capstyle": "butt"})


def figure(height: float):
    return plt.figure(figsize=(WIDTH, height))


def axes_in(fig, left: float, bottom: float, width: float, height: float):
    """an axes placed in inches from the figure's bottom-left corner"""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def tidy(ax, grid: str = "both") -> None:
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def legend_row(fig, handles, labels, top: float = 0.06, left: float = 0.0, spacing: float = 1.6):
    """one row of legend above everything, no frame, `top` in from the figure's top edge"""
    w, h = fig.get_size_inches()
    return fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left / w, 1 - top / h), ncol=len(labels),
                      frameon=False, borderaxespad=0, columnspacing=spacing)


def mark(colour, hollow=False, marker="o", size=5.0, line=None, err=False):
    """a legend handle: a point (filled or hollow), with an error bar or as a line"""
    if line:
        return Line2D([], [], color=colour, lw=1.2, ls=line)
    return Line2D([], [], marker=marker, ls="", ms=size, mec=colour, mew=0.9,
                  mfc="white" if hollow else colour)


class Checker:
    """what a figure draws, read back from it: labels and the points, lines and bars they must
    not cover"""

    def __init__(self, name: str):
        self.name, self.problems, self.skip = name, [], set()

    def allow(self, *pairs):
        self.skip |= {frozenset(p) for p in pairs}

    def run(self, fig) -> list[str]:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        W, H = fig.bbox.width, fig.bbox.height
        texts = [t for t in fig.findobj(matplotlib.text.Text)
                 if t.get_visible() and t.get_text().strip() and t.get_figure() is fig]
        boxes = []
        for t in texts:
            b = t.get_window_extent(r)
            if b.width <= 0:
                continue
            if t.get_fontsize() < SMALL - 1e-6:
                self.problems.append(f"under {SMALL} pt: {t.get_text()!r} ({t.get_fontsize()} pt)")
            if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5:
                self.problems.append(f"off the page: {t.get_text()!r}")
            pad, cut = 0.6 * fig.dpi / 72, 0.12 * b.height      # touching counts; the line's leading doesn't
            boxes.append((t, matplotlib.transforms.Bbox.from_extents(b.x0 - pad, b.y0 + cut, b.x1 + pad, b.y1 - cut)))
        for (a, ba), (b, bb) in combinations(boxes, 2):
            if ba.overlaps(bb) and not self._skip(a, b):
                self.problems.append(f"labels overlap: {a.get_text()!r} and {b.get_text()!r}")
        marks, segs = self._marks_and_lines(fig)
        for t, b in boxes:
            if t.get_gid() == "inside":
                continue
            for (x, y, rad, owner) in marks:
                cx, cy = min(max(x, b.x0), b.x1), min(max(y, b.y0), b.y1)
                if (cx - x) ** 2 + (cy - y) ** 2 < rad ** 2:
                    self.problems.append(f"label on a point: {t.get_text()!r} at ({x:.0f}, {y:.0f})")
                    break
            for (p, q, owner) in segs:
                if owner and t.get_gid() and owner == "leader:" + t.get_gid():
                    continue
                if seg_hits_box(p, q, b):
                    self.problems.append(f"label on a line: {t.get_text()!r} ({owner or 'line'})")
                    break
        return self.problems

    def _skip(self, a, b):
        return frozenset((a.get_text(), b.get_text())) in self.skip

    @staticmethod
    def _marks_and_lines(fig):
        marks, segs = [], []
        px = fig.dpi / 72
        for ax in fig.axes:
            for c in ax.collections:
                if isinstance(c, PathCollection) and len(c.get_offsets()):
                    offs = c.get_offset_transform().transform(c.get_offsets())
                    sizes = c.get_sizes()
                    for i, (x, y) in enumerate(offs):
                        s = sizes[i % len(sizes)] if len(sizes) else 20
                        marks.append((x, y, (math.sqrt(s) / 2 + 0.5) * px, c.get_gid()))
                elif isinstance(c, LineCollection):
                    for s in c.get_segments():
                        pts = c.get_transform().transform(s)
                        segs += [(pts[i], pts[i + 1], c.get_gid()) for i in range(len(pts) - 1)]
            for ln in ax.get_lines():
                if not ln.get_visible() or ln.get_gid() == "no-check":
                    continue
                xy = ln.get_transform().transform(ln.get_xydata())
                if ln.get_marker() not in (None, "None", "", " ") and len(xy):
                    rad = (ln.get_markersize() / 2 + 0.5) * px
                    marks += [(x, y, rad, ln.get_gid()) for x, y in xy]
                if ln.get_linestyle() not in ("None", "", " ") and len(xy) > 1:
                    segs += [(xy[i], xy[i + 1], ln.get_gid()) for i in range(len(xy) - 1)]
        return marks, segs


def seg_hits_box(p, q, b) -> bool:
    """does the segment p-q cross the box (Liang-Barsky clipping)"""
    x0, y0 = p
    dx, dy = q[0] - x0, q[1] - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - b.x0), (dx, b.x1 - x0), (-dy, y0 - b.y0), (dy, b.y1 - y0)):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def save(fig, out: Path, n: int, head: list[str], rows: list[list], check: Checker, problems_out: dict):
    w, _ = fig.get_size_inches()
    assert abs(w - WIDTH) < 1e-9, w
    problems_out[n] = check.run(fig)
    fig.savefig(out / f"figure-{n}.pdf", metadata={"CreationDate": None, "Producer": "matplotlib"})
    fig.savefig(out / f"figure-{n}.png", dpi=220, metadata={"Software": None})
    plt.close(fig)
    with open(out / f"figure-{n}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(head)
        wr.writerows([["" if c is None else c for c in r] for r in rows])


def text(ax, x, y, s, gid=None, **kw):
    kw.setdefault("fontsize", SMALL)
    kw.setdefault("color", INK)
    t = ax.text(x, y, s, **kw)
    if gid:
        t.set_gid(gid)
    return t


def leader(ax, xy0, xy1, gid, **kw):
    ln, = ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], color=kw.get("color", MUTED), lw=0.5, zorder=2,
                  transform=kw.get("transform", ax.transData))
    ln.set_gid("leader:" + gid)
    return ln


# ---------------------------------------------------------------------------
# Figure 1: DeviceMark composite against token generation throughput on a phone
# ---------------------------------------------------------------------------

def speeds(out: Path) -> dict:
    """label -> (speed, source); the file, when it is there, wins over DeviceMark's board"""
    got = {}
    f = out / "figure-1.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r.get("speed_tok_s"):
                name = r["model"] + (f", {r['mode']}" if r["mode"] else "")
                # r22: a speed under a name no point has would be dropped without a word
                if name not in {label for label, _, _ in SPEED_ROWS}:
                    raise SystemExit(f"figure-1.csv: {name} has a speed, but no point of Figure 1 is {name}")
                got[name] = (r["speed_tok_s"], r["speed_source"])
    for label, _, dmname in SPEED_ROWS:
        if label not in got:
            mine = OWN_SPEEDS.get(label)
            if dmname is not None:
                got[label] = dm_speed(dmname)
            elif mine:
                got[label] = mine
    return got


def dm_speed(name: str) -> tuple[str, str]:
    """(speed, source): the model's decode tokens/s on DeviceMark's board as the repo keeps it"""
    from email.utils import parsedate_to_datetime
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    got = [r for r in snap["rows"] if r["model"] == name and r.get("iphone_tok_s") is not None]
    assert len(got) == 1, (name, "is not once on DeviceMark's board with an iPhone speed")
    r, day = got[0], parsedate_to_datetime(snap["last_modified"])
    return (f"{r['iphone_tok_s']:g}",
            f"DeviceMark leaderboard, {snap['device']}: {r['model']} {r['quant']} on "
            f"{RUNTIME_WORDS.get(r['native_runtime'], r['native_runtime'])}, decode tokens/s, warm, a 128-token "
            f"prompt and 256 decoded tokens, as published (data as of {day.day} {day:%b %Y})")


def dm_rows(data: Path) -> list[dict]:
    """DeviceMark's rows as Table 12 names them (r23), Medium's added"""
    rows = list(csv.DictReader(open(data / "figure-10.csv", encoding="utf-8")))
    for r in rows:
        r["Model"] = DM_RENAMED.get(r["Model"], r["Model"])
    assert not {r["Model"] for r in rows} & {r["Model"] for r in DM_ADDED}
    return rows + [dict(r) for r in DM_ADDED]


def fig1(data: Path, out: Path, P: dict):
    dm = {r["Model"]: r for r in dm_rows(data)}
    sp = speeds(out)
    pts = []
    for label, key, pip in SPEED_ROWS:
        if key is None:                       # r22: a Masein mode figure-10.csv doesn't have
            v, lo, hi = (float(x) for x in OWN_SCORES[label])
        else:
            r = dm[key]
            base = key.split(",")[0]
            others = [o for k, o in dm.items() if k.split(",")[0] == base and k != key and o["Kind"] != "published"]
            assert base == "Masein model" or all(float(o["Composite"]) < float(r["Composite"]) for o in others), \
                (key, "not its higher-scoring mode")
            v, lo, hi = float(r["Composite"]), float(r["95% low"]), float(r["95% high"])
        # r22: a row with no speed yet (High) is kept in figure-1.csv and not drawn
        xs, how = sp.get(label, ("", ""))
        pts.append((label, xs, v, lo, hi, how, OURS + (" (replica)" if pip else "")))
    clouds = [(c, float(dm[c]["Composite"])) for c in CLOUD]
    fig = figure(3.9)
    ax = axes_in(fig, 0.78, 0.72, 5.62, 2.84)
    ax.set_xlim(0, 80)
    ax.set_ylim(25, 100)                          # r25: Nanbeige4.1-3B's range reaches 30.3
    tidy(ax)
    for (c, v), va, dy in zip(clouds, ("bottom", "top"), (2.5, -2.5)):
        ax.axhline(v, color=GREEN, lw=1.1, ls=(0, (4, 2.5)), zorder=1)
        t = text(ax, 79.6, v, f"{c}: {v:.1f}, published by DeviceMark", ha="right", va=va,
                 transform=ax.transData)
        t.set_position((79.6, v))
        t.set_transform(ax.transData + matplotlib.transforms.ScaledTranslation(0, dy / 72, fig.dpi_scale_trans))
    # r25: (where from: "lo"/"hi" a bar's end, else the point; offset in points; alignment), or
    # ("at", x, y): the label's left end at those data values, a leader from the point
    place = {"Qwen3.5-4B, thinking off": ("lo", 0, -3, "center", "top"),
             "Gemma 4 E2B, thinking on": (5, 0, "left", "center"),
             "Youtu-LLM-2B, thinking off": (-5, 0, "right", "center"),
             "LFM2.5-1.2B": (5, 0, "left", "center"),
             "Qwen3.5-2B, thinking off": ("at", 32.5, 74.0),
             "granite-4.0-h-1b": (5, 0, "left", "center"),
             "Nanbeige4.1-3B": (5, 0, "left", "center"),
             "Qwen3.5-0.8B, thinking off": (-5, 0, "right", "center")}
    for label, xs, y, lo, hi, _, _ in pts:
        if not xs:
            continue
        x = float(xs)
        me = label.startswith("Masein")
        col = BLUE if me else ORANGE
        # r22: the other models' bars and points are drawn over the Masein model's points, so a
        # bar that runs behind one (Qwen3.5-4B's, beside Instant) still shows
        e = ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=col, zorder=3 if me else 5, **EBAR)
        for part in e.lines[1] + tuple(e.lines[2]):
            part.set_gid("pt:" + label)
        sc = ax.scatter([x], [y], s=62 if me else 30, color=col, edgecolor="white", lw=0.8, zorder=4 if me else 6)
        sc.set_gid("pt:" + label)
        words = f"{SHOWN.get(label, label)}: {y:.1f} at {xs} tokens/s"
        if me and label in MINE_UP:
            # r22: Medium sits left of Instant, Qwen3.5-4B's and Instant's bars between them and
            # the free space on the right: its label goes above the cloud lines, a leader up from
            # its bar
            ytext = MINE_UP[label]
            leader(ax, (x, hi + 0.6), (x, ytext - 1.2), "pt:" + label)
            t = ax.annotate(words, (x, ytext), xytext=(-2, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=SMALL, fontweight="bold", color=INK)
        elif me:
            t = ax.annotate(words, (x, y), xytext=(8, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=SMALL, fontweight="bold", color=INK)
        elif place[label][0] == "at":
            _, lx, ly = place[label]
            leader(ax, (x + 0.7, y + 0.8), (lx - 0.4, ly - 0.6), "pt:" + label)
            t = ax.annotate(label, (lx, ly), ha="left", va="center", fontsize=SMALL, color=INK)
        else:
            spec = place[label]
            at = (x, {"lo": lo, "hi": hi}[spec[0]]) if spec[0] in ("lo", "hi") else (x, y)
            dx, dy, ha, va = spec[1:] if spec[0] in ("lo", "hi") else spec
            t = ax.annotate(label, at, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=SMALL, color=INK)
        t.set_gid("pt:" + label)
    ax.set_xlabel("Token generation throughput on a phone (tokens per second)")
    ax.set_ylabel("DeviceMark composite (%)\nmean of MMLU-Pro, MATH and IFEval")
    fig.text(0.78 / WIDTH + 5.62 / WIDTH / 2, 0.08 / 3.9, "Speeds come from different phones and setups; see caption.",
             ha="center", va="bottom", fontsize=SMALL, color=MUTED)
    legend_row(fig, [mark(BLUE, size=6.5), mark(ORANGE, size=5), mark(GREEN, line=(0, (4, 2.5)))],
               ["Masein model (95% range)", "other models, scored by us (95% range)", "cloud model"], left=0.78)
    rows = [[lab.split(", ")[0], (lab.split(", ")[1] if ", " in lab else ""), f"{y:.1f}", f"{lo:.1f}", f"{hi:.1f}",
             src_s, x, how, DM_SPEED_CREDIT if how.startswith("DeviceMark") else OURS if x else ""]
            for lab, x, y, lo, hi, how, src_s in pts]
    rows += [[c, "", f"{v:.1f}", dm[c]["95% low"], dm[c]["95% high"], DEVICEMARK, "", "", ""] for c, v in clouds]
    head = ["model", "mode", "devicemark_composite", "ci95_low", "ci95_high", "score_source", "speed_tok_s",
            "speed_source", "speed_credit"]
    return fig, head, rows, Checker("1")


# ---------------------------------------------------------------------------
# Figure 2: Epoch's runs by release date
# ---------------------------------------------------------------------------

def fig2(E: dict, P: dict):
    B, MZ = E["benchmarks"], E["masein"]
    o1 = next(r for r in B["gpqa"]["rows"] if r["model"] == "o1")
    # the caption's facts
    g = sorted([r for r in B["gpqa"]["rows"] if r["released"]], key=lambda r: r["released"])
    first = next(r for r in g if r["value"] >= MZ["gpqa"]["v"])
    assert (first["model"], first["released"]) == ("Gemini 2.5 Pro (Mar 2025)", "2025-03-25"), first["model"]
    m = [r for r in B["math"]["rows"] if r["released"]]
    best = max(m, key=lambda r: r["value"])
    assert not [r for r in m if r["value"] >= MZ["math"]["v"]] and len(m) == B["math"]["models"] == 94
    assert (best["model"], best["released"], f"{best['value']:.1f}") == ("GPT-5", "2025-08-07", "98.1")
    assert max(r["released"] for r in m).startswith("2025-10")
    assert (o1["released"][:7], o1["verdict"], f"{o1['value']:.1f}") == ("2024-12", "level with", "76.8")
    fig = figure(5.25)
    t0, t1 = date(2023, 1, 1), date(2026, 12, 31)
    X = lambda dt: (dt - t0).days
    rows_out = []
    axs = [axes_in(fig, 0.55, 3.0, 5.85, 1.78), axes_in(fig, 0.55, 0.42, 5.85, 1.78)]
    for ax, (b, title) in zip(axs, (("gpqa", "GPQA Diamond"), ("math", "MATH Level 5"))):
        ax.set_title(title)
        ax.set_xlim(0, X(t1))
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ticks = [date(y, m, 1) for y in (2023, 2024, 2025, 2026) for m in (1, 7)]
        ax.set_xticks([X(t) for t in ticks])
        ax.set_xticklabels([f"{'Jan' if t.month == 1 else 'Jul'} {t.year}" for t in ticks])
        tidy(ax)
        mz = MZ[b]
        rows = sorted([r for r in B[b]["rows"] if r["released"]], key=lambda r: (r["released"], -r["value"]))
        if b == "math":
            last = max(d(r["released"]) for r in rows)
            ax.axvspan(X(last) + 12, X(t1), color="#F2F2F2", zorder=0, lw=0)
            t = text(ax, (X(last) + 12 + X(t1)) / 2, 6, "not run by Epoch", ha="center", va="bottom", color=MUTED)
        best, path = None, []
        for r in rows:
            if best is None or r["value"] > best:
                if best is not None:
                    path.append((X(d(r["released"])), best))
                best = r["value"]
                path.append((X(d(r["released"])), best))
        path.append((X(t1), best))
        ax.plot(*zip(*path), color=INK, lw=0.9, zorder=3, solid_joinstyle="miter")
        circ = [r for r in rows if not r["dagger"] and r is not o1]
        sq = [r for r in rows if r["dagger"]]
        ax.scatter([X(d(r["released"])) for r in circ], [r["value"] for r in circ], s=13, facecolor="white",
                   edgecolor=CONTEXT, lw=0.7, zorder=2)
        ax.scatter([X(d(r["released"])) for r in sq], [r["value"] for r in sq], s=13, marker="s", facecolor="white",
                   edgecolor=CONTEXT, lw=0.7, zorder=2)
        ax.axhline(mz["v"], color=BLUE, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
        mx = X(date(2026, 11, 7))
        ax.errorbar([mx], [mz["v"]], yerr=1.96 * mz["se"], fmt="none", ecolor=BLUE, zorder=4, **EBAR)
        ax.scatter([mx], [mz["v"]], s=34, color=BLUE, edgecolor="white", lw=0.7, zorder=5)
        # the Masein model's line, labelled at its left end
        t = text(ax, X(date(2023, 1, 20)), mz["v"], f"Masein model: {mz['v']:.1f}", fontweight="bold",
                 va="bottom" if b == "gpqa" else "top")
        t.set_transform(ax.transData + matplotlib.transforms.ScaledTranslation(
            0, (2 if b == "gpqa" else -2.5) / 72, fig.dpi_scale_trans))
        if b == "gpqa":
            ox = X(d(o1["released"]))
            ax.errorbar([ox], [o1["value"]], yerr=1.96 * o1["se"], fmt="none", ecolor=ORANGE, zorder=5, **EBAR)
            s = ax.scatter([ox], [o1["value"]], s=26, facecolor="white", edgecolor=ORANGE, lw=1.1, zorder=6)
            s.set_gid("o1")
            lx, ly = X(date(2023, 3, 1)), 70.5
            t1_ = text(ax, lx, ly, f"o1 (Dec 2024) {o1['value']:.1f}: level,\ngap {pm(o1['gap'])} "
                                   f"({pm(o1['lo'])} to {pm(o1['hi'])})", va="center", linespacing=1.25, gid="o1")
            bb = t1_.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
            leader(ax, (bb.x1 + 10, ly + 2.5), (ox - 16, o1["value"]), "o1")
        for r in rows:
            rows_out.append([title, label_of(r), r["released"], f"{r['value']:.2f}", f"{r['se']:.2f}",
                             "yes" if r["dagger"] else "",
                             "labelled; bar: 95% range, score ± 1.96 × SE" if r is o1 else "", EPOCH])
        rows_out.append([title, "Masein model, thinking on", "2026-11-07", f"{mz['v']:.2f}", f"{mz['se']:.2f}", "",
                         "plotted at the date of our runs; bar: 95% range, score ± 1.96 × SE", OURS])
    axs[0].tick_params(labelbottom=True)
    fig.text(0.1 / WIDTH, (0.42 + 3.0 + 1.78) / 2 / 5.25, "Score (%)", rotation=90, ha="left", va="center",
             color=MUTED, fontsize=SMALL)
    legend_row(fig, [mark(CONTEXT, hollow=True, size=4.2), mark(CONTEXT, hollow=True, marker="s", size=4.2),
                     Line2D([], [], color=INK, lw=0.9), mark(BLUE, size=5.5)],
               ["Epoch AI run", "below high effort (†)", "best so far", "Masein model, measured here (95% range)"],
               left=0.02, spacing=1.1)
    ck = Checker("2")
    head = ["Benchmark", "Model (setting)", "Released", "Score (%)", "SE", "Below high effort (†)", "Note", "Source"]
    return fig, head, rows_out, ck


# ---------------------------------------------------------------------------
# Figure 3: how one question becomes a score (a diagram)
# ---------------------------------------------------------------------------

def fig3(P: dict):
    H = 136.0
    fig = figure(H / 72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH * 72)
    ax.set_ylim(H, 0)
    ax.axis("off")
    bw, bh, gap = 85.6, 34.0, 10.0
    xs = [i * (bw + gap) for i in range(5)]
    top, low = 4.0, 92.0

    def box(x, y, l1, l2, me=False):
        ax.add_patch(FancyBboxPatch((x + 0.6, y), bw - 1.2, bh, boxstyle="round,pad=0,rounding_size=4",
                                    fc=BLUE25 if me else "white", ec=BLUE if me else AXIS, lw=1.1 if me else 0.7))
        text(ax, x + bw / 2, y + (bh / 2 if not l2 else 12.5), l1, ha="center", va="center", fontweight="bold",
             gid="inside")
        if l2:
            text(ax, x + bw / 2, y + 24, l2, ha="center", va="center", color=MUTED, gid="inside")

    def arrow(p, q, dashed=False):
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.3", color=MUTED, lw=0.8,
                                              ls=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0.5,
                                              mutation_scale=8))
    steps = [("Question", ""), ("Masein model", "thinking on"), ("Reply split", "from thinking"),
             ("Grader", "reads the reply"), ("Right or wrong", "per answer")]
    for i, (a, b_) in enumerate(steps):
        box(xs[i], top, a, b_, me=(i == 1))
    for i in range(4):
        arrow((xs[i] + bw, top + bh / 2), (xs[i + 1], top + bh / 2))
    text(ax, xs[3] + bw / 2, top + bh + 9, "code, the benchmark's", ha="center", va="center", color=MUTED)
    text(ax, xs[3] + bw / 2, top + bh + 19.5, "own checker or a judge", ha="center", va="center", color=MUTED)
    box(xs[1], low, "Ran out of room", "token limit hit")
    arrow((xs[1] + bw / 2, top + bh), (xs[1] + bw / 2, low), dashed=True)
    box(xs[4], low, "Score", "mean of runs ± SE")
    arrow((xs[4] + bw / 2, top + bh), (xs[4] + bw / 2, low))
    arrow((xs[1] + bw, low + bh / 2), (xs[4], low + bh / 2), dashed=True)
    text(ax, (xs[1] + bw + xs[4]) / 2, low + bh / 2 - 7, "counted wrong, never graded", ha="center", va="center",
         color=MUTED)
    rows = [[a + (f" ({b_})" if b_ else ""), nxt] for (a, b_), nxt in
            zip(steps, ["Masein model", "Reply split, or Ran out of room at the token limit", "Grader", "Right or wrong",
                        "Score"])]
    rows += [["Ran out of room (token limit hit)", "Score: counted wrong, never graded"],
             ["Score (mean of runs ± SE)", ""]]
    return fig, ["Box", "Leads to"], rows, Checker("3")


# ---------------------------------------------------------------------------
# Figure 4: right on every run, some runs or none
# ---------------------------------------------------------------------------

def fig4(P: dict):
    fig = figure(1.9)
    ax = axes_in(fig, 1.45, 0.08, 4.95, 1.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(1.75, -0.55)
    ax.axis("off")
    cols = (BLUE, BLUE45, "white")
    rows = []
    for i, (name, nq, runs, parts) in enumerate(RELIABILITY):
        n = sum(parts)
        assert n == nq, name
        text(ax, -29.2, i - 0.08, name, fontweight="bold", va="bottom", gid="inside")
        text(ax, -29.2, i + 0.04, f"{nq} questions × {runs} runs", color=MUTED, va="top", gid="inside")
        x = 0
        for k, (v, c) in enumerate(zip(parts, cols)):
            w = 100 * v / n
            ax.barh(i, w, left=x, height=0.62, color=c, ec=AXIS if c == "white" else "white", lw=0.6, zorder=2)
            lab = f"{v} ({100 * v / n:.1f}%)"
            bold = (i == 0 and k == 1)
            if w > 14:
                text(ax, x + w / 2, i, lab, ha="center", va="center", color="white" if c == BLUE else INK,
                     fontweight="bold" if bold else "normal", gid="inside")
            else:
                ax.plot([x + w / 2, x + w / 2], [i + 0.31, i + 0.42], color=MUTED, lw=0.5)
                text(ax, min(x + w / 2, 100 - 6.5), i + 0.45, lab, ha="center", va="top",
                     fontweight="bold" if bold else "normal")
            x += w
        rows.append([name, nq, runs, *parts, *(f"{100 * v / n:.1f}" for v in parts), OURS])
    legend_row(fig, [Rectangle((0, 0), 1, 1, fc=BLUE), Rectangle((0, 0), 1, 1, fc=BLUE45),
                     Rectangle((0, 0), 1, 1, fc="white", ec=AXIS, lw=0.6)],
               ["Right on every run", "Right on some runs", "Never right"])
    head = ["Benchmark", "Questions", "Runs each", "Right on every run", "Right on some runs", "Never right",
            "Every run (%)", "Some runs (%)", "Never (%)", "Source"]
    return fig, head, rows, Checker("4")


# ---------------------------------------------------------------------------
# Figure 5: thinking tokens in right and wrong answers
# ---------------------------------------------------------------------------

def fig5(P: dict):
    fig = figure(3.05)
    ax = axes_in(fig, 1.55, 0.52, 4.2, 2.2)
    mx = 75000
    ax.set_xlim(0, mx)
    n = len(THINKING)
    ax.set_ylim(n - 0.35, -0.45)
    ax.set_xticks(range(0, mx + 1, 15000))
    ax.set_xticklabels(["0" if v == 0 else f"{v // 1000}k" for v in range(0, mx + 1, 15000)])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    tidy(ax, "x")
    rows = []
    for i, (name, nr, nw, tr, twr) in enumerate(THINKING):
        mr, mw = tr / nr, twr / nw
        for k, (m, fc) in enumerate(((mr, BLUE), (mw, BLUE25))):
            y = i - 0.17 + k * 0.36
            ax.barh(y, m, height=0.32, color=fc, ec=BLUE, lw=0.6, zorder=2)
            text(ax, m + 900, y, f"{m:,.0f}", va="center", gid="inside")
        text(ax, -mx * 0.36, i - 0.06, name, fontweight="bold", va="bottom", gid="inside")
        text(ax, -mx * 0.36, i + 0.04, f"{nr:,} right, {nw:,} wrong", va="top", color=MUTED, gid="inside")
        text(ax, mx * 1.155, i + 0.02, f"×{mw / mr:.1f}", ha="right", va="center", fontweight="bold", gid="inside")
        rows.append([name, nr, nw, tr, twr, f"{mr:.0f}", f"{mw:.0f}", f"{mw / mr:.1f}", OURS + " (Table 7)"])
    text(ax, mx * 1.155, -0.62, "wrong ÷ right", ha="right", va="bottom", color=MUTED, gid="inside")
    ax.set_xlabel("Mean thinking tokens per finished answer")
    legend_row(fig, [Rectangle((0, 0), 1, 1, fc=BLUE, ec=BLUE, lw=0.6), Rectangle((0, 0), 1, 1, fc=BLUE25, ec=BLUE, lw=0.6)],
               ["right answers", "wrong answers"])
    head = ["Benchmark", "Finished right", "Finished wrong", "Thinking tokens, right", "Thinking tokens, wrong",
            "Mean, right", "Mean, wrong", "Wrong ÷ right", "Source"]
    return fig, head, rows, Checker("5")


# ---------------------------------------------------------------------------
# Figure 6: out of room at the final limit
# ---------------------------------------------------------------------------

def fig6(P: dict):
    fig = figure(2.45)
    ax = axes_in(fig, 1.2, 0.5, 3.75, 1.88)
    mxs = 8.0
    ax.set_xlim(0, mxs)
    n = len(RAN_OUT)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in RAN_OUT], color=INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 9, 2))
    ax.set_xticklabels([f"{v}%" for v in range(0, 9, 2)])
    tidy(ax, "x")
    rows = []
    for i, (name, k, tot) in enumerate(RAN_OUT):
        share = 100 * k / tot
        ax.barh(i, mxs, height=0.62, color="#F3F3F3", lw=0, zorder=1)
        ax.barh(i, share, height=0.62, color=BLUE if i < 2 else BLUE45, lw=0, zorder=2)
        text(ax, mxs * 1.02, i, f"{share:.1f}% ({k} of {tot:,})", va="center", gid="inside", clip_on=False)
        rows.append([name, k, tot, f"{share:.1f}", OURS + " (Tables 5 and 8)"])
    ax.set_xlabel("Share of answers out of room at the final limit (%)")
    assert f"{100 * 22 / 360:.1f}" == "6.1" and f"{100 * 15 / 300:.1f}" == "5.0"
    assert max(100 * k / t for _, k, t in RAN_OUT[2:]) <= 1.3 + 1e-9
    return fig, ["Benchmark", "Ran out of room", "Answers", "Share (%)", "Source"], rows, Checker("6")


# ---------------------------------------------------------------------------
# Figure 7: GPQA Diamond in three verdict lanes
# ---------------------------------------------------------------------------

def swarm(xs: list[float], r: float) -> dict:
    """y offsets from the lane's centre, in points: each mark as near the centre as it goes
    without touching one already placed"""
    placed, out = [], {}
    for i in sorted(range(len(xs)), key=lambda i: xs[i]):
        x = xs[i]
        for k in range(0, 80):
            hit = None
            for off in ((0,) if k == 0 else (k * 0.9, -k * 0.9)):
                if all((x - px) ** 2 + (off - py) ** 2 >= (2 * r + 0.6) ** 2 for px, py in placed if abs(x - px) < 2 * r + 1):
                    hit = off
                    break
            if hit is not None:
                break
        placed.append((x, hit))
        out[i] = hit
    return out


VERDICT_COLOUR = {"higher than": BLUE, "level with": LEVEL, "lower than": ORANGE}


def fig7(E: dict, P: dict):
    rows = E["benchmarks"]["gpqa"]["rows"]
    mz = E["masein"]["gpqa"]
    level = [r for r in rows if r["verdict"] == "level with"]
    assert (len(level), len(rows)) == (53, 226)
    assert sum(r["dagger"] for r in rows) == 6 and sum(r["ddagger"] for r in rows) == 13
    ax_w = 6.5 - 0.1 - 0.1
    ptx = ax_w * 72 / 100                         # points per score point
    R = 2.25
    lanes = [("lower than", f"Masein model lower than ({sum(r['verdict'] == 'lower than' for r in rows)})", 66),
             ("level with", f"level with ({len(level)})", 70),
             ("higher than", f"higher than ({sum(r['verdict'] == 'higher than' for r in rows)})", 58)]
    lay, y = [], 0.0
    for key, lab, minh in lanes:
        lr = [r for r in rows if r["verdict"] == key]
        offs = swarm([r["value"] * ptx for r in lr], R)
        h = max(minh, 2 * (max(abs(o) for o in offs.values()) + R) + 22)
        lay.append((key, lab, lr, offs, y, h))
        y += h + 5
    total = y - 5
    head_room, foot = 34.0, 40.0
    H = head_room + total + foot
    fig = figure(H / 72)
    ax = axes_in(fig, 0.1, foot / 72, ax_w, total / 72)
    ax.set_xlim(0, 100)
    ax.set_ylim(total, 0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("GPQA Diamond score (%)")
    named = {}
    for key, lab, lr, offs, y0, h in lay:
        ax.add_patch(Rectangle((0, y0), 100, h, fc="white", ec=GRID, lw=0.6, zorder=0))
        text(ax, 0.8, y0 + 4, lab, fontweight="bold", va="top", gid="inside")
        cy = y0 + 16 + (h - 16) / 2
        col = VERDICT_COLOUR[key]
        for i, r in enumerate(lr):
            x, py = r["value"], cy + offs[i]
            sc = ax.scatter([x], [py], s=(2 * R) ** 2 * 0.95, marker="s" if r["dagger"] else "o", facecolor="white",
                            edgecolor=col, lw=0.9, zorder=3)
            sc.set_gid("pt:" + r["model"])
            if r["ddagger"]:
                ring = ax.scatter([x], [py], s=(3.5 * R) ** 2, marker="o", facecolor="none", edgecolor=INK, lw=0.45,
                                  zorder=3)
                ring.set_gid("pt:" + r["model"])
            if r["model"] in NAMED_GPQA:
                named[r["model"]] = (x, py, r, key, y0, h)
    mx = mz["v"]
    ln, = ax.plot([mx, mx], [-4, total], color=BLUE, lw=1.2, zorder=4, clip_on=False)
    ln.set_gid("masein-line")
    lab = text(ax, mx, -6, f"Masein model {mz['v']:.1f} ± {mz['se']:.1f} (measured here)", ha="center", va="bottom",
               fontweight="bold", clip_on=False)
    # labels in the empty part of each lane, with short leaders
    PLACE = {"Gemini 4 Argon": (80, 0), "GPT-6 Astra": (80, 1), "Claude Sonnet 5.5": (80, 2),
             "DeepSeek V4.1 Flash 1017": (80, 3),
             "o4-mini": (73.5, 0), "o3-mini": (73.5, 1), "o1": (73.5, 2), "Qwen 3.6 35B-A3B": (73.5, 3),
             "GPT-5 mini": (84.0, 2), "Claude Haiku 4.5": (84.0, 0)}
    order = {}
    for m, (x, py, r, key, y0, h) in named.items():
        order.setdefault((key, PLACE[m][0]), []).append((py, m))
    for grp in order.values():
        ks = sorted(PLACE[m][1] for _, m in grp)
        for (py_, m), k in zip(sorted(grp), ks):
            PLACE[m] = (PLACE[m][0], k)
    for m, (x, py, r, key, y0, h) in named.items():
        at, k = PLACE[m]
        lab_s = label_of(r) + (" ‡" if r["ddagger"] else "")
        ty = y0 + 22 + k * 11.5
        gid = "pt:" + r["model"]
        if key != "higher than":
            text(ax, at, ty, lab_s, ha="right", va="center", gid=gid)
            leader(ax, (at + 0.6, ty), (x - 0.9, py), gid)
        else:
            two = lab_s.replace(" (", "\n(", 1) if len(lab_s) > 22 else lab_s
            text(ax, at, ty + (5 if "\n" in two else 0), two, ha="left", va="center", linespacing=1.15, gid=gid)
            leader(ax, (at - 0.6, ty), (x + 0.9, py), gid)
    legend_row(fig, [mark(MUTED, hollow=True, size=4.6), mark(MUTED, hollow=True, marker="s", size=4.4),
                     Line2D([], [], marker="o", ls="", ms=7.9, mfc="none", mec=INK, mew=0.45)],
               ["an Epoch AI model, its best-scoring run", "below high effort (†)", "borderline verdict (‡)"],
               left=0.1)
    out = [["Masein model (measured here)", "", f"{mz['v']:.2f}", f"{mz['se']:.2f}", "", "", "", "yes", "", OURS]]
    for r in sorted(rows, key=lambda r: -r["value"]):
        out.append([name_of(r), setting_words(r["setting"]), f"{r['value']:.2f}", f"{r['se']:.2f}", r["verdict"],
                    "‡" if r["ddagger"] else "", "†" if r["dagger"] else "", "yes" if r["model"] in NAMED_GPQA else "",
                    r["released"], EPOCH])
    head = ["Model", "Setting", "Score (%)", "SE", "Verdict (Masein model …)", "‡", "† below high effort",
            "Named on the chart", "Released", "Source"]
    return fig, head, out, Checker("7")


# ---------------------------------------------------------------------------
# Figure 8: verdict counts; Figure 9: the 21-row grid
# ---------------------------------------------------------------------------

def fig8(E: dict, P: dict):
    B = E["benchmarks"]
    data = [("GPQA Diamond", "gpqa"), ("OTIS Mock AIME", "otis"), ("MATH Level 5", "math")]
    fig = figure(2.35)
    ax = axes_in(fig, 1.2, 0.5, 4.75, 1.52)
    rows = []
    for i, (name, b) in enumerate(data):
        t = B[b]["tally"]
        hi, lv, lo = t.get("higher than", 0), t.get("level with", 0), t.get("lower than", 0)
        tot = B[b]["models"]
        assert hi + lv + lo == tot
        x = 0
        for nn, col, txc in ((hi, BLUE, "white"), (lv, LEVEL, INK), (lo, ORANGE, "white")):
            if nn == 0:
                continue
            w = 100 * nn / tot
            ax.barh(i, w - 0.35, left=x, height=0.58, color=col, zorder=2, lw=0)
            text(ax, x + w / 2, i, str(nn), ha="center", va="center", color=txc, fontweight="bold", gid="inside")
            x += w
        text(ax, 101.5, i, f"of {tot}", va="center", color=MUTED, gid="inside", clip_on=False)
        rows.append([name, tot, hi, lv, lo, EPOCH + "; the Masein model " + OURS])
    assert [r[2] + r[3] for r in rows] == [170, 165, 94]
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in data], color=INK)
    ax.set_ylim(2.55, -0.55)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of the models Epoch AI has run, Masein model with thinking on")
    tidy(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    legend_row(fig, [Rectangle((0, 0), 1, 1, fc=c) for c in (BLUE, LEVEL, ORANGE)],
               ["Masein model higher", "level with", "Masein model lower"], left=1.2)
    return fig, ["Benchmark", "Models", "Masein model higher than", "Level with", "Masein model lower than",
                 "Source"], rows, Checker("8")


GLYPH_MARK = {"higher": "^", "level": "o", "lower": "v"}
GRID_FILL = {"higher": BLUE25, "level": LEVEL, "lower": ORANGE25}


def glyph(ax, x, y, kind, size=5.2, colour=INK):
    """▲ ● ▼ as marks (the font has no such characters); x, y in the axes' data"""
    ln, = ax.plot([x], [y], marker=GLYPH_MARK[kind], ms=size * (0.82 if kind == "level" else 1.0), color=colour,
                  mew=0, ls="")
    ln.set_gid("glyph")
    return ln


def fig9(E: dict, data: Path, P: dict):
    B = E["benchmarks"]
    grid = list(csv.reader(open(data / "figure-8.csv", encoding="utf-8")))[1:]
    byname = {b: {CSV_NAME.get(r["model"], r["model"]): r for r in B[b]["rows"]} for b in ("gpqa", "otis", "math")}
    rh, ghead = 14.5, 15.0
    groups = []
    for r in grid[1:]:
        if not groups or groups[-1] != r[0]:
            groups.append(r[0])
    H = 22 + 26 + 18 + len(grid[1:]) * rh + len(groups) * ghead + 4
    fig = figure(H / 72)
    ax = fig.add_axes([0, 0, 1, 1])
    W = WIDTH * 72
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    cols, cw = [176.0, 240.0, 304.0], 60.0
    px = 418.0
    y = 2.0
    # legend: shading with its mark
    lx = 0.0
    for kind, lab in (("higher", "Masein model higher"), ("level", "level with"), ("lower", "Masein model lower")):
        ax.add_patch(Rectangle((lx, y + 1.5), 16, 10, fc=GRID_FILL[kind], lw=0))
        glyph(ax, lx + 8, y + 6.5, kind, size=4.6)
        t = text(ax, lx + 21, y + 6.5, lab, va="center", gid="inside")
        lx = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).x1 + 16
    text(ax, lx, y + 6.5, "–  not run by Epoch", va="center", color=MUTED, gid="inside")
    y += 22
    for c, (l1, l2) in zip(cols, (("GPQA", "Diamond"), ("OTIS Mock", "AIME"), ("MATH", "Level 5"))):
        text(ax, c + cw / 2, y + 5, l1, ha="center", va="center", fontweight="bold", gid="inside")
        text(ax, c + cw / 2, y + 16, l2, ha="center", va="center", fontweight="bold", gid="inside")
    text(ax, px, y + 5, "Parameters", ha="center", va="center", fontweight="bold", gid="inside")
    text(ax, px, y + 16, "total / active", ha="center", va="center", fontweight="bold", gid="inside")
    y += 26
    me = grid[0]
    text(ax, 0, y + 7, "Masein model (measured here)", va="center", fontweight="bold", gid="inside")
    for c, k in zip(cols, (2, 5, 8)):
        text(ax, c + cw / 2, y + 7, f"{float(me[k]):.1f}", ha="center", va="center", fontweight="bold", gid="inside")
    assert me[14] == PARAMS_MASEIN, me[14]
    text(ax, px, y + 7, PARAMS_MASEIN, ha="center", va="center", fontweight="bold", gid="inside")
    ax.plot([0, W], [y + 16, y + 16], color=INK, lw=1.0, gid="no-check")
    y += 18
    PAR = {"DeepSeek V4.1 Flash 1017": "552B / 8–16B"}
    group, out, ddag = None, [], 0
    for r in grid[1:]:
        if r[0] != group:
            group = r[0]
            text(ax, 0, y + ghead / 2 + 1, group, va="center", color=MUTED, gid="inside")
            y += ghead
        cells = [(r[2], r[4], r[11]), (r[5], r[7], r[12]), (r[8], r[10], r[13])]
        settings = {s.replace(" †", "") for v, s, _ in cells if v}
        uniform = len(settings) == 1
        s0 = setting_words(next(iter(settings))) if uniform else None
        shown = "Qwen3.6-35B-A3B (base model)" if r[1] == "Qwen 3.6 35B-A3B" else r[1]
        lab = shown + (f" ({s0})" if uniform and s0 else "")
        if uniform and any(s.endswith("†") for v, s, _ in cells if v):
            lab += " †"
        text(ax, 0, y + rh / 2, lab, va="center", gid="inside")
        line = [r[0], lab]
        for c, (v, s, vd), b in zip(cols, cells, ("gpqa", "otis", "math")):
            kind = vd.split()[0] if vd and vd != "not run" else "not run"
            if v:
                full = byname[b][r[1]]["value"]
                cell = f"{full:.1f}" + (" †" if (not uniform and s.endswith("†")) else "") + (" ‡" if "‡" in vd else "")
                ddag += "‡" in vd
                ax.add_patch(Rectangle((c, y + 1), cw, rh - 2, fc=GRID_FILL[kind], lw=0))
                t = text(ax, c + cw / 2 - 4, y + rh / 2, cell, ha="center", va="center")
                bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
                glyph(ax, bb.x1 + 5.5, y + rh / 2, kind)
                line += [f"{full:.1f}", s, vd]
            else:
                text(ax, c + cw / 2, y + rh / 2, "–", ha="center", va="center", color=MUTED, gid="inside")
                line += ["", "", "not run"]
        par = PAR.get(r[1], r[14])
        text(ax, px, y + rh / 2, par, ha="center", va="center", color=MUTED, gid="inside")
        out.append(line + [par, EPOCH])
        y += rh
    assert len(grid[1:]) == 21 and ddag == 4, (len(grid[1:]), ddag)
    head = ["Group", "Model", "GPQA Diamond", "GPQA setting", "GPQA verdict", "OTIS Mock AIME", "OTIS setting",
            "OTIS verdict", "MATH Level 5", "MATH setting", "MATH verdict",
            "Parameters, total / active (model cards; the Masein model from its file's header)", "Source"]
    out.insert(0, ["Subject", "Masein model (measured here)", me[2], "thinking on", "", me[5], "thinking on", "",
                   me[8], "thinking on", "", PARAMS_MASEIN, OURS])
    return fig, head, out, Checker("9")


# ---------------------------------------------------------------------------
# Figure 10: on-device scores against size, and the scorecard
# ---------------------------------------------------------------------------

def fig10(E: dict, P: dict):
    MZ = E["masein"]
    H = 6.55
    fig = figure(H)
    s0, s1 = math.log2(0.35), math.log2(40)
    X = lambda s: math.log2(s)
    POS = {"GPQA": {"Ternary Bonsai 2 2B": (5.5, -3.5, "left"), "Gemma 4 E2B": (5.5, 7, "left"),
                    "Ternary Bonsai 2 4B": (5.5, 0, "left"), "Ternary Bonsai 2 9B": (-5.5, 0, "right"),
                    "Gemma 4 E4B": (5.5, -5.5, "left"), "Qwen3.5-4B": (-5.5, 0, "right"), "Edge0-8B": (5.5, -3, "left"),
                    "Qwen3.5-9B": (-5.5, 4, "right"), "Edge0-35B": (0, -9, "center")},
           "IFBench": {"Ternary Bonsai 2 2B": (5.5, 0, "left"), "Gemma 4 E2B": (5.5, 0, "left"),
                       "Ternary Bonsai 2 4B": ("x", -1.40, 15, "left"), "Ternary Bonsai 2 9B": (-5.5, -9.6, "right"),
                       "Gemma 4 E4B": (5.5, 0, "left"), "Qwen3.5-4B": (0, 9, "center"),
                       "Qwen3.5-4B, measured by NVIDIA": (-5.5, 4, "right"), "Bonsai 27B (1-bit)": (5.5, -7, "left"),
                       "Edge0-8B": (5.5, 8.5, "left"), "Qwen3.5-9B": (0, 9, "center"), "Bolt": (5.5, -3, "left"),
                       "Edge0-35B": (-5.5, -9.6, "right"), "Ternary Bonsai 2 27B": (-5.5, 0, "right")}}
    axs = [axes_in(fig, 0.5, 4.72, 5.9, 1.25), axes_in(fig, 0.5, 3.02, 5.9, 1.25)]
    rows_out = []
    for ax, (panel, k, lo, hi, me_v, me_se) in zip(axs, (("GPQA", 2, 30, 90, MZ["gpqa"]["v"], MZ["gpqa"]["se"]),
                                                         ("IFBench", 5, 35, 75, 63.7, 2.8))):
        ax.set_title("GPQA Diamond" if panel == "GPQA" else "IFBench")
        ax.set_xlim(s0, s1)
        ax.set_ylim(lo, hi)
        ax.set_yticks(range(lo, hi + 1, 10 if panel == "GPQA" else 10))
        ax.set_xticks([X(s) for s in (0.5, 1, 2, 4, 8, 16, 32)])
        ax.set_xticklabels([f"{s:g}" for s in (0.5, 1, 2, 4, 8, 16, 32)] if panel == "IFBench" else [])
        tidy(ax)
        pts = [(g[0], g[1], g[k]) for g in ON_DEVICE if g[k] is not None] + ([NV_QWEN4B] if panel == "IFBench" else [])
        for nm, s, v in pts:
            sc = ax.scatter([X(s)], [v], s=20, facecolor="white", edgecolor=ORANGE, lw=1.0, zorder=3)
            sc.set_gid("pt:" + nm + panel)
            lab = f"{v:.1f} {nm}" if v != 66.67 else f"66.7 {nm}"
            spec = POS[panel][nm]
            if spec[0] == "x":                       # set apart, with a leader
                _, xd, dy, ha = spec
                ppu = 1.25 * 72 / (hi - lo)
                t = text(ax, xd, v + dy / ppu, lab, ha=ha, va="center", gid="pt:" + nm + panel)
                bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
                leader(ax, (bb.x1 - 0.25, bb.y0), (X(s) - 0.03, v + 0.9), "pt:" + nm + panel)
            else:
                dx, dy, ha = spec
                t = ax.annotate(lab, (X(s), v), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                                fontsize=SMALL, color=INK)
                t.set_gid("pt:" + nm + panel)
            rows_out.append([nm, s, panel, f"{v:g}" if v != 66.67 else "66.67", "published by " + {
                "Edge0": "Edge0", "Qwen": "Qwen (Alibaba)", "PrismML": "PrismML", "Google": "Google",
                "Liquid AI": "Liquid AI", "NVIDIA": "NVIDIA"}.get(next((g[6] for g in ON_DEVICE if g[0] == nm), "NVIDIA"))])
        ax.errorbar([X(MASEIN_SIZE)], [me_v], yerr=1.96 * me_se, fmt="none", ecolor=BLUE, zorder=4, **EBAR)
        sc = ax.scatter([X(MASEIN_SIZE)], [me_v], s=40, color=BLUE, edgecolor="white", lw=0.7, zorder=5)
        sc.set_gid("me" + panel)
        t = ax.annotate(f"{me_v:.1f} Masein model", (X(MASEIN_SIZE), me_v + 1.96 * me_se), xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=SMALL, color=INK)
        t.set_gid("me" + panel)
        rows_out.append(["Masein model", MASEIN_SIZE, panel, f"{me_v:.1f}",
                         OURS + f"; bar: 95% range, {me_v - 1.96 * me_se:.1f} to {me_v + 1.96 * me_se:.1f}"])
    axs[1].set_xlabel("File or deployed size, GB (log scale)")
    fig.text(0.04 / WIDTH, (3.02 + 4.72 + 1.25) / 2 / H, "Score (%)", rotation=90, ha="left", va="center",
             color=MUTED, fontsize=SMALL)
    # the scorecard
    tab = fig.add_axes([0, 0, 1, 2.55 / H])
    th = 2.55 * 72
    tab.set_xlim(0, WIDTH * 72)
    tab.set_ylim(th, 0)
    tab.axis("off")
    cx = [0, 150, 222, 290, 352, 410, 468]
    heads = ["Scores (%)", "Size", "GPQA Diamond", "MMLU-Pro", "IFEval", "IFBench", "Source"]
    al = ["left", "right", "center", "center", "center", "center", "right"]
    y = 4.0
    for i, hd in enumerate(heads):
        text(tab, cx[i], y + 6, hd, ha=al[i], va="center", fontweight="bold", gid="inside")
    y += 13
    rh = 11.6
    tab.add_patch(Rectangle((-2, y), WIDTH * 72 + 4, rh, fc=BLUE25, lw=0, clip_on=False))
    me = ["Masein model", "16.0 GB", *MASEIN_ON_DEVICE.values(), "ours"]
    for i, v in enumerate(me):
        text(tab, cx[i], y + rh / 2, v, ha=al[i], va="center", fontweight="bold" if i != 6 else "normal", gid="inside")
    y += rh
    for nm, s, g, mm, ie, ib, src in ON_DEVICE:
        size = {1.0: "about 1 GB", 2.0: "2.0 GB"}.get(s, f"{s:g} GB")
        cells = [nm if nm != "Bolt" else "Nemotron 3.5 Bolt E4B", size, g, mm, ie, ib, src]
        for i, v in enumerate(cells):
            if i in (2, 3, 4, 5):
                vv = "–" if v is None else (f"{v:.2f}" if v in (96.66, 64.00, 91.31, 74.00) else f"{v:.1f}")
                text(tab, cx[i], y + rh / 2, vv, ha="center", va="center", color=MUTED, gid="inside")
            else:
                text(tab, cx[i], y + rh / 2, v, ha=al[i], va="center", color=INK if i == 0 else MUTED, gid="inside")
        tab.plot([-2, WIDTH * 72 + 2], [y + rh, y + rh], color=GRID, lw=0.6, gid="no-check", clip_on=False)
        y += rh
    legend_row(fig, [mark(BLUE, size=5.5), mark(ORANGE, hollow=True, size=4.6)],
               ["Masein model, measured by us (95% range)", "published, each under its publisher's own harness"],
               left=0.5)
    above_gpqa = [g for g in ON_DEVICE if g[2] is not None and g[2] > 82.7]
    above_ifb = sorted([g[5] for g in ON_DEVICE if g[5] is not None and g[5] > 63.7] + [NV_QWEN4B[2]])
    assert not above_gpqa and above_ifb == [64.0, 64.5, 66.67, 74.0], above_ifb   # + Bolt's 16-bit 63.79, not plotted
    head = ["Model", "Size (GB)", "Panel", "Score (%)", "Source"]
    return fig, head, rows_out, Checker("10")


# ---------------------------------------------------------------------------
# Figure 11: DeviceMark, replica on top, published below
# ---------------------------------------------------------------------------

def fig11(data: Path, P: dict):
    rows = dm_rows(data)
    rep = sorted([r for r in rows if r["Kind"] in ("Masein", "replica")], key=lambda r: -float(r["Composite"]))
    pub = sorted([r for r in rows if r["Kind"] == "published"], key=lambda r: -float(r["Composite"]))
    assert len(rows) == 28
    off = next(r for r in rows if r["Model"] == "Masein model, Instant")
    assert off["Composite"] == "84.4" and sum(float(r["Composite"]) > 84.4 for r in pub) == 2
    rh, gh = 13.2, 16.0
    H = 26 + gh * 2 + rh * (len(rep) + len(pub)) + 6 + 34
    fig = figure(H / 72)
    left = 1.95
    plot_h = (H - 26 - 34) / 72
    ax = axes_in(fig, left, 34 / 72, 6.5 - left - 0.42, plot_h)
    total = plot_h * 72
    ax.set_xlim(0, 100)
    ax.set_ylim(total, 0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("DeviceMark composite (%)")
    tidy(ax, "x")
    y, out = 0.0, []
    for head, grp in (("Measured by us (replica)", rep), ("DeviceMark's published rows", pub)):
        text(ax, 0.02, y + gh / 2 + 2, head, ha="left", va="center", color=MUTED, gid="inside",
             transform=matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData))
        y += gh
        for r in grp:
            name = DEVICEMARK_NAMES.get(r["Model"], r["Model"])
            me = name.startswith("Masein")
            cloud = r["Model"] in CLOUD
            col = BLUE if me else (GREEN if cloud else ORANGE)
            v, lo, hi = float(r["Composite"]), float(r["95% low"]), float(r["95% high"])
            cy = y + rh / 2
            gid = "pt:" + name
            text(ax, -1.6, cy, name, ha="right", va="center", fontweight="bold" if me else "normal", gid=gid,
                 clip_on=False)
            e = ax.errorbar([v], [cy], xerr=[[v - lo], [hi - v]], fmt="none", ecolor=col, zorder=3, **EBAR)
            for part in e.lines[1] + tuple(e.lines[2]):
                part.set_gid(gid)
            hollow = r["Kind"] == "published"
            sc = ax.scatter([v], [cy], s=36 if me else 22, facecolor="white" if hollow else col,
                            edgecolor=col if hollow else "white", lw=1.0 if hollow else 0.6, zorder=4)
            sc.set_gid(gid)
            text(ax, hi + 1.6, cy, f"{v:.1f}", va="center", color=INK if me else MUTED,
                 fontweight="bold" if me else "normal", gid=gid)
            out.append([name, r["Kind"], r["Composite"], r["95% low"], r["95% high"],
                        r.get("Verdict vs Masein off", ""), r["Source"]])
            y += rh
        y += 6
    legend_row(fig, [mark(BLUE, size=5.5), mark(ORANGE, size=4.8), mark(ORANGE, hollow=True, size=4.6),
                     mark(GREEN, hollow=True, size=4.6)],
               ["Masein model", "replica, measured by us", "published by DeviceMark", "cloud model, published"])
    head = ["Model", "Kind", "Composite", "95% low", "95% high", "Verdict vs the Masein model, Instant", "Source"]
    return fig, head, out, Checker("11")


# ---------------------------------------------------------------------------

def contact_sheet(out: Path, n: int = 11) -> None:
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(out / f"figure-{i}.png").convert("RGB") for i in range(1, n + 1)]
    colw, gap, pad, labh = 1430, 60, 60, 70
    font = ImageFont.truetype(str(FONTS / "texgyreheros-bold.otf"), 40)
    cols = [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9]]
    hts = [sum(ims[i].height + labh + gap for i in c) for c in cols]
    sheet = Image.new("RGB", (pad * 2 + colw * 2 + gap, pad * 2 + max(hts)), "white")
    dr = ImageDraw.Draw(sheet)
    for ci, c in enumerate(cols):
        x, y = pad + ci * (colw + gap), pad
        for i in c:
            dr.text((x, y), f"Figure {i + 1}", fill=INK, font=font)
            y += labh
            sheet.paste(ims[i], (x, y))
            dr.rectangle([x - 1, y - 1, x + ims[i].width, y + ims[i].height], outline="#DDDDDD")
            y += ims[i].height + gap
    sheet.save(out / "contact-sheet.png", optimize=True)


FONTS = Path(".")


def main(argv: list[str]) -> int:
    global FONTS
    data, FONTS, out = (Path(a) for a in argv[:3])
    out.mkdir(parents=True, exist_ok=True)
    setup(FONTS)
    E = json.loads((data / "epoch-recompute.json").read_text(encoding="utf-8"))
    problems: dict = {}
    made = [(1, fig1(data, out, problems)), (2, fig2(E, problems)), (3, fig3(problems)), (4, fig4(problems)),
            (5, fig5(problems)), (6, fig6(problems)), (7, fig7(E, problems)), (8, fig8(E, problems)),
            (9, fig9(E, data, problems)), (10, fig10(E, problems)), (11, fig11(data, problems))]
    for n, (fig, head, rows, ck) in made:
        save(fig, out, n, head, rows, ck, problems)
    contact_sheet(out)
    bad = {n: p for n, p in problems.items() if p}
    for n in sorted(problems):
        print(f"figure-{n}: " + ("ok" if not problems[n] else f"{len(problems[n])} problems"))
        for p in problems[n]:
            print("   ", p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
