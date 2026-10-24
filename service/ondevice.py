"""25c: On-device models — of the models that can run on a phone, which are
best, and where are ours (docs/prompts/phase-25c-on-device-models-report.md).

The report file (eval_tasks/reports/on-device.json, `kind: "on_device"`)
names the rows and where each number comes from; every number is read from
the board's own tables:
- ours, measured here: the rows the file names (the step checkpoints and the
  Q4 and Q2 baselines), and every phone build with a score on its benchmarks or a
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
row's score over its base's, from the same source (ours over our BF16). 27e:
on a benchmark our BF16 hasn't run, the file's `ours_stand_in` (our Q4)
stands in for it, said on every row it serves, until BF16's own comes in."""

from __future__ import annotations

from . import phone, reports, telemetry

KIND = "on_device"
THINKING = " · thinking"


def _phone_rows(rep: dict, cells: dict, measured: dict, cps: list[dict] | None = None
                ) -> list[dict]:
    """our phone builds with a score on the report's benchmarks — the row
    with thinking on when it has one, else the plain one — or, 26, a test on
    our testers' phones (27c: by its checkpoint too)"""
    tested = {r for e in cps or [] if not e["variant"] and e["phones"] for r in e["rows"]}
    have = {o["row"] for o in rep["ours"]}
    out = []
    for b in phone.builds():
        for mid in (b["id"] + THINKING, b["id"]):
            if mid in have:
                break
            if any((cells.get(x["task"]) or {}).get(mid) for x in rep["benchmarks"]) \
                    or mid == b["id"] and (measured.get(b["id"]) or b["id"] in tested):
                out.append({"label": f"{b['name']} (phone build)", "row": mid, "colour": None,
                            "phone_build": b["id"]})
                break
    return out


def _testers_of(spec: dict, measured: dict, cps: list[dict] | None = None) -> dict | None:
    """26: a phone build as our testers' phones measured it — per phone the
    medians of decode and engine memory, the phone named; None for a row with
    none. 27c: and any row of ours whose checkpoint an app model is (names.json's
    `checkpoint`), with what is known of the file: "the same file (sha256
    checked)", or the app's file not yet checked to be this one — a variant
    (model-003) never stands for its checkpoint here"""
    def phones(got):
        return [{"phone": p["phone"]["name"], "decode": p["decode"],
                 "engine_mem_mb": p["engine_mem_mb"], "n": p["n"],
                 "first": p["first"], "last": p["last"]} for p in got]
    got = [p for p in measured.get(spec.get("phone_build") or "") or []
           if p.get("decode") is not None]
    if got:
        return {"phones": phones(got), "line": "the same file (sha256 checked)"}
    base = str(spec.get("row") or "").removesuffix(THINKING)
    for e in cps or []:
        if not e["variant"] and base in e["rows"]:
            got = [p for p in e["phones"] if p.get("decode") is not None]
            if got:
                return {"phones": phones(got), "line": telemetry.checked_line(e, base),
                        "model": e["model"]["name"]}
    if not spec.get("phone_build"):
        return None
    return {"none": "no tests from our testers' phones yet"}


def _gb(n_bytes) -> float | None:
    return n_bytes / 1e9 if isinstance(n_bytes, (int, float)) and n_bytes > 0 else None


def _ratio(v, base) -> float | None:
    return v / base if isinstance(v, (int, float)) and isinstance(base, (int, float)) and base else None


def _has(cells: dict, key: str) -> bool:
    return isinstance((cells.get(key) or {}).get("v"), (int, float))


def stand_in(rep: dict, rows: list[dict], base_ours: dict | None) -> dict | None:
    """27e: the build standing in for our BF16 on the benchmarks BF16 hasn't
    run — the file's `ours_stand_in`, on those of its benchmarks it has a
    score on — with what it keeps of BF16 on the others both have run; None
    where BF16 has its own on every one"""
    s = rep.get("ours_stand_in") or {}
    row = next((r for r in rows if r["label"] == s.get("label")), None)
    base = (base_ours or {}).get("cells") or {}
    keys = [k for k in s.get("benchmarks") or []
            if row and not _has(base, k) and _has(row["cells"], k)]
    if not keys:
        return None
    keeps = [x for b in rep["benchmarks"] if b["key"] not in keys
             if (x := _ratio((row["cells"].get(b["key"]) or {}).get("v"),
                             (base.get(b["key"]) or {}).get("v"))) is not None]
    short, of = row["label"].split(" (")[0], rep.get("ours_base")
    lo, hi = (f"{100 * min(keeps):.1f}", f"{100 * max(keeps):.1f}") if keeps else ("", "")
    where = f"{short} keeps {lo if lo == hi else lo + '–' + hi}% of {of} elsewhere" if keeps else ""
    return {"label": row["label"], "short": short, "for": of, "benchmarks": keys,
            "keeps": [min(keeps), max(keeps)] if keeps else None,
            "of": f"our {short}, standing in for {of}", "elsewhere": where,
            "line": f"of our {short}, standing in for {of}" + (f" ({where})" if where else ""),
            "cells": row["cells"]}


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
    cps = telemetry.by_checkpoint() if rep.get("phone_builds") else []
    rep = {**rep, "ours": [*rep["ours"],
                           *(_phone_rows(rep, cells, measured, cps) if rep.get("phone_builds")
                             else [])]}
    benches = rep["benchmarks"]
    rows = reports.our_rows(rep, cells)
    base_ours = next((r for r in rows if r["label"] == rep.get("ours_base")), None)
    si = stand_in(rep, rows, base_ours)
    out = []
    for spec, r in zip(rep["ours"], rows):
        if spec.get("base_only"):
            continue
        gb = _gb(r.get("size"))
        sc = {b["key"]: {**r["cells"][b["key"]], "src": "here"} for b in benches}
        # 27e: BF16's where it has one, else the stand-in's (never the stand-in's own row)
        base = {**{k: si["cells"].get(k) for k in (si or {}).get("benchmarks") or []
                   if si["label"] != r["label"]},
                **{k: c for k, c in ((base_ours or {}).get("cells") or {}).items() if _has({k: c}, k)}}
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
            # 27e: its line under the name ("the original, squeezed to 2 bits — …")
            **({"what": spec["line"]} if spec.get("line") else {}),
            "phone": _phone_of_ours(spec), "testers": _testers_of(spec, measured, cps), "cells": sc,
            "retention": {} if r is base_ours else {
                b["key"]: _ratio(sc[b["key"]].get("v"), (base.get(b["key"]) or {}).get("v"))
                for b in benches if _ratio(sc[b["key"]].get("v"),
                                           (base.get(b["key"]) or {}).get("v")) is not None},
            # 27e: which of those are of the stand-in, and the benchmarks a named
            # build of ours hasn't run ("–" in What each keeps, "not run yet")
            "keepOf": {k: "stand_in" for k in (si or {}).get("benchmarks") or []
                       if r is not base_ours and si["label"] != r["label"] and _has(sc, k)},
            "notRun": [] if spec.get("phone_build") or r is base_ours
            else [b["key"] for b in benches if not _has(sc, b["key"])]})
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
            "rows": out, "notes": rep.get("notes") or [],
            "standIn": {k: v for k, v in si.items() if k != "cells"} if si else None}


def listing_counts(rep: dict) -> tuple[list[str], int]:
    """the Reports list's card: our rows shown and the others' count"""
    ours = [o["label"] for o in rep["ours"] if not o.get("base_only")]
    return ours, sum(1 for t in rep.get("theirs") or [] if not t.get("base_only"))
