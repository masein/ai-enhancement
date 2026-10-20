"""23e: our checkpoints of one training run as one line through training,
and the benchmarks near the ceiling
(docs/prompts/phase-23e-across-training-and-examples.md).

A report's `ours` entries with a `step` are checkpoints of the same run; its
`training` names the label of the axis and the rows drawn as reference lines.
Between two consecutive checkpoints scored on a benchmark, the segment
carries 23's question-by-question verdict for that pair, read from what the
stats worker kept (service/paired.py) — none where it hasn't worked the pair
out. A request never computes.

Near the ceiling: a benchmark where the reference row scores CEILING or more
— a few questions decide a difference there. Every number is the board's;
nothing from outside."""

from __future__ import annotations

CEILING = 0.90
CEILING_WORDS = ("Most models score above 90% here, so a difference of a point or two says "
                 "little about which is better; it often comes down to a handful of questions.")
WORDS = {"A higher": "higher", "B higher": "lower"}           # paired_stats' verdicts, said


def near_ceiling(rep: dict, rows: list[dict]) -> dict[str, bool]:
    """{benchmark key: whether the reference row scores CEILING or more}"""
    ref = next((r for r in rows if r["label"] == rep.get("reference")), None)
    out = {}
    for b in rep["benchmarks"]:
        v = ((ref or {}).get("cells") or {}).get(b["key"], {}).get("v")
        out[b["key"]] = isinstance(v, (int, float)) and v >= CEILING
    return out


def checkpoints(rep: dict) -> list[dict]:
    """the checkpoints, in order of their step"""
    got = [{"label": o["label"], "step": o["step"]} for o in rep["ours"]
           if isinstance(o.get("step"), int) and not isinstance(o.get("step"), bool)]
    return sorted(got, key=lambda x: x["step"])


def _paired(rep_id: str) -> dict | None:
    from . import reports
    return next((r for r in reports.files().values() if r.get("kind") == "paired"
                 and (r.get("builds") or {}).get("from") == rep_id), None)


def said(verdict: str | None) -> str:
    return WORDS.get(verdict or "", "can't tell")


def segment(prep: dict | None, later: str, earlier: str, task: str, compute: bool = False
            ) -> dict | None:
    """23's verdict for later against earlier on one benchmark, as kept: its
    word and the difference in points — None where it isn't worked out"""
    from . import paired
    if prep is None or later not in paired.labels(prep) or earlier not in paired.labels(prep):
        return None
    if paired.samples(prep, later, task) is None or paired.samples(prep, earlier, task) is None:
        return None
    x = paired.kept(prep, later, earlier, task, compute)
    if not x or not x.get("n"):
        return None
    return {"verdict": said(x.get("verdict")), "d": 100 * x["d"] if x.get("d") is not None
            else None, "fresh": x.get("fresh", True)}


def _names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def of_report(rep: dict, rows: list[dict], compute: bool = False) -> dict | None:
    """the section's data: the checkpoints with their steps, the reference
    rows, each benchmark's segments with their verdicts, the sentence and
    the line against the reference, and the benchmarks near the ceiling.
    None when the report has fewer than two checkpoints"""
    cps = checkpoints(rep)
    if len(cps) < 2:
        return None
    t = rep.get("training") if isinstance(rep.get("training"), dict) else {}
    have = {r["label"]: r for r in rows}
    refs = [x for x in t.get("references") or [] if x in have]
    prep = _paired(rep["id"])
    ceil = near_ceiling(rep, rows)
    segs: dict[str, list[dict]] = {}
    for b in rep["benchmarks"]:
        scored = [c for c in cps if ((have.get(c["label"]) or {}).get("cells") or {})
                  .get(b["key"], {}).get("v") is not None]
        segs[b["key"]] = []
        for e, lt in zip(scored, scored[1:]):
            s = segment(prep, lt["label"], e["label"], b["task"], compute)
            segs[b["key"]].append({"a": e["label"], "b": lt["label"], **(s or {})})
    # the sentence: the last two checkpoints with a score anywhere
    some = [c for c in cps if any(v.get("v") is not None for v in
                                  ((have.get(c["label"]) or {}).get("cells") or {}).values())]
    text = against = ""
    if len(some) >= 2:
        a, b = some[-2]["label"], some[-1]["label"]
        groups: dict[str, list[str]] = {"higher": [], "can't tell": [], "lower": []}
        for bm in rep["benchmarks"]:
            s = next((x for x in segs[bm["key"]] if x["a"] == a and x["b"] == b
                      and x.get("verdict")), None)
            if s:
                groups[s["verdict"]].append(bm["label"] + (
                    " (near the ceiling)" if s["verdict"] != "can't tell" and ceil[bm["key"]]
                    else ""))
        parts = [f"{w} on {_names(ns)}" for w, ns in groups.items() if ns]
        text = f"From {a} to {b}: " + "; ".join(parts) + "." if parts else ""
        if refs:
            ref = refs[0]
            vs = [(bm, segment(prep, b, ref, bm["task"], compute)) for bm in rep["benchmarks"]]
            vs = [(bm, s) for bm, s in vs if s]
            low = [bm for bm, s in vs if s["verdict"] == "lower"]
            if vs:
                short = ref.split(" (")[0]
                cl = [bm["label"] for bm in low if ceil[bm["key"]]]
                against = (f"Against {short}, {b} is still lower on {len(low)} of {len(vs)}"
                           + (f" ({_names(cl)} near the ceiling)" if cl else "") + "."
                           if low else f"Against {short}, {b} is lower on none of {len(vs)}.")
    return {"label": t.get("label") or "Training steps", "checkpoints": cps,
            "references": refs, "segments": segs, "sentence": text, "against": against,
            "paired": prep["id"] if prep else None}
