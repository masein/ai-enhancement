"""25c: On-device models — of the models that can run on a phone, which are
best, and where are ours (docs/prompts/phase-25c-on-device-models-report.md).

The report file (eval_tasks/reports/on-device.json, `kind: "on_device"`)
names the rows and where each number comes from; every number is read from
the board's own tables:
- ours, measured here: the rows the file names (the step checkpoints and the
  Q4 baseline), and every phone build with a score on its benchmarks or a
  test on our testers' phones; their scores as the board has them
  (`reports.our_rows`), their file's size as the box measured it
  (`reports.build_file`), a phone speed as someone reported it by hand
  (service/phone.py) and, 26, as our testers' phones measured it
  (service/telemetry.py) — the two never mixed;
- theirs, published: each model's numbers as entered through the board's
  card entry (`reports.their_rows`, scripts/enter_report_cards.py), and its
  facts as the file gives them, each with its link — size on disk, memory it
  needs, phone speed with the device named, what else it published.

Retention is the fair comparison: each source ran its own harness, but each
ran its model beside its own full-precision base on the same harness — so a
row's score over its base's, from the same source (ours over our BF16)."""

from __future__ import annotations

from . import phone, reports, telemetry

KIND = "on_device"
THINKING = " · thinking"


def _phone_rows(rep: dict, cells: dict, measured: dict) -> list[dict]:
    """our phone builds with a score on the report's benchmarks — the row
    with thinking on when it has one, else the plain one — or, 26, a test on
    our testers' phones"""
    have = {o["row"] for o in rep["ours"]}
    out = []
    for b in phone.builds():
        for mid in (b["id"] + THINKING, b["id"]):
            if mid in have:
                break
            if any((cells.get(x["task"]) or {}).get(mid) for x in rep["benchmarks"]) \
                    or mid == b["id"] and measured.get(b["id"]):
                out.append({"label": f"{b['name']} (phone build)", "row": mid, "colour": None,
                            "phone_build": b["id"]})
                break
    return out


def _testers_of(spec: dict, measured: dict) -> dict | None:
    """26: a phone build as our testers' phones measured it — per phone the
    medians of decode and engine memory, the phone named; None for a row that
    is no phone build"""
    if not spec.get("phone_build"):
        return None
    got = [p for p in measured.get(spec["phone_build"]) or [] if p.get("decode") is not None]
    if not got:
        return {"none": "no tests from our testers' phones yet"}
    return {"phones": [{"phone": p["phone"]["name"], "decode": p["decode"],
                        "engine_mem_mb": p["engine_mem_mb"], "n": p["n"],
                        "first": p["first"], "last": p["last"]} for p in got]}


def _gb(n_bytes) -> float | None:
    return n_bytes / 1e9 if isinstance(n_bytes, (int, float)) and n_bytes > 0 else None


def _ratio(v, base) -> float | None:
    return v / base if isinstance(v, (int, float)) and isinstance(base, (int, float)) and base else None


def _phone_of_ours(spec: dict) -> dict:
    """a phone speed of ours — only as someone typed it in from the phone,
    and said so; the board never takes its server's speed for a phone's"""
    if not spec.get("phone_build"):
        return {"none": "not measured on a phone yet"}
    got = [r for r in phone.reports(spec["phone_build"]) if r.get("decode_median")]
    if not got:
        return {"none": "none reported by hand yet"}
    r = got[0]                                            # newest first
    return {"tok_s": r["decode_median"], "device": r.get("device") or "",
            "words": f"{r['decode_median']:g} tok/s decode, median",
            "by_hand": r.get("by") or r.get("entered_by") or "", "date": r.get("date") or ""}


def report(rep: dict, cells: dict) -> dict:
    """the report's payload: its rows with their facts, scores, sources and
    retention, the benchmarks and the notes"""
    measured = telemetry.by_row() if rep.get("phone_builds") else {}
    rep = {**rep, "ours": [*rep["ours"],
                           *(_phone_rows(rep, cells, measured) if rep.get("phone_builds")
                             else [])]}
    benches = rep["benchmarks"]
    rows = reports.our_rows(rep, cells)
    base_ours = next((r for r in rows if r["label"] == rep.get("ours_base")), None)
    out = []
    for spec, r in zip(rep["ours"], rows):
        if spec.get("base_only"):
            continue
        gb = _gb(r.get("size"))
        sc = {b["key"]: {**r["cells"][b["key"]], "src": "here"} for b in benches}
        base = (base_ours or {}).get("cells") or {}
        out.append({
            "label": r["label"], "ours": True, "colour": r.get("colour"), "row": r["row"],
            **({"step": r["step"]} if r.get("step") is not None else {}),
            "source": "measured here", "phoneBuild": bool(spec.get("phone_build")),
            "base": "Qwen3.6-35B-A3B", "baseLabel": f"our {rep.get('ours_base')}, measured here",
            "disk": {"gb": gb, "words": f"{gb:.1f} GB", "how": "its GGUF, as the box measured it"}
            if gb else None,
            "memory": {"gb": gb, "words": f"{gb:.1f} GB",
                       "how": "the whole file, loaded on the server; not measured on a phone yet"}
            if gb else None,
            "phone": _phone_of_ours(spec), "testers": _testers_of(spec, measured), "cells": sc,
            "retention": {} if r is base_ours else {
                b["key"]: _ratio(sc[b["key"]].get("v"), (base.get(b["key"]) or {}).get("v"))
                for b in benches if _ratio(sc[b["key"]].get("v"),
                                           (base.get(b["key"]) or {}).get("v")) is not None}})
    theirs = {t["label"]: t for t in reports.their_rows(rep)}
    from . import reported
    board, al = reported.board_ids(), reported.aliases()
    for spec in rep.get("theirs") or []:
        if spec.get("base_only"):
            continue
        t = theirs[spec["label"]]
        # its page on Models: a board model when it is one, else its reported page
        card = next((c for b in benches if isinstance(c := spec.get(b["key"]), dict)
                     and c.get("from") == "card"), {})
        k = reported.canon(reported.key(card.get("model") or spec["label"],
                                        card.get("maker") or spec.get("maker") or ""))
        mid = reported.resolve(k, board, al)
        page = mid if mid in board else "reported/" + k
        by = spec.get("published_by") or t.get("maker") or ""
        sc = {b["key"]: {**t["cells"][b["key"]], "src": "published", "by_org": by}
              for b in benches}
        base = (theirs.get(spec.get("base") or "") or {}).get("cells") or {}
        f = spec.get("facts") or {}
        out.append({
            "label": spec["label"], "ours": False, "colour": None, "maker": t.get("maker"),
            "page": page,
            "source": f"published by {by}", "publishedBy": by, "link": spec.get("link") or "",
            "base": f.get("base") or "", "baseLabel": spec.get("base") or "",
            "kind": f.get("kind") or "", "disk": f.get("disk"), "memory": f.get("memory"),
            "phone": f.get("phone") or {"none": "no phone numbers published"},
            "speed": f.get("speed") or [], "also": f.get("also") or [],
            "paper": f.get("paper") or "", "cells": sc,
            "retention": {b["key"]: _ratio(sc[b["key"]].get("v"), (base.get(b["key"]) or {}).get("v"))
                          for b in benches
                          if _ratio(sc[b["key"]].get("v"), (base.get(b["key"]) or {}).get("v"))
                          is not None}})
    return {"id": rep["id"], "kind": KIND, "title": rep.get("title") or rep["id"],
            "what": rep.get("what") or "", "benchmarks": reports.bench_meta(benches),
            "rows": out, "notes": rep.get("notes") or []}


def listing_counts(rep: dict) -> tuple[list[str], int]:
    """the Reports list's card: our rows shown and the others' count"""
    ours = [o["label"] for o in rep["ours"] if not o.get("base_only")]
    return ours, sum(1 for t in rep.get("theirs") or [] if not t.get("base_only"))
