"""12p.3: a new hidden set for Everyday — built now, switched to when the
owner says (after the demo: new hidden questions change every score).

1. The plan: each group's new questions to write (as many as its hidden set
   holds now, and 40% more for what review sets aside), the models that write
   and check them — a writer that is not a Qwen model, a checker from another
   maker than the writer — and the cost of all of it, shown before anything
   starts.
2. Start: one Build questions draft a group, "for the hidden set" — the
   board's owner's alone, reviewed as any draft is. Publishing one stages its
   questions (BENCH_ROOT/everyday/hidden_next.jsonl): they score nothing and
   are shown nowhere until the switch.
3. The switch, once every group has its new questions: the new set becomes the
   hidden set; the old one becomes practice questions (it has been public),
   but for any the owner retires — the weak or flagged, listed with why;
   each model's Everyday score goes to History as "scored on the retired
   hidden set". Logged, backed up before and after. The committed manifest and
   the CI guard's fingerprints are made again then (HANDOFF)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from . import ai_models, builder, config, db

SPARE = 1.4
RETIRED_LABEL = "scored on the retired hidden set"
NOT_QWEN = "the writer is not a Qwen model"
OTHER_MAKER = "the checker is from another maker than the writer"


def _ev():
    builder._scripts()
    import everyday as ev
    return ev


def staged_path() -> Path:
    return _ev().built_dir() / "hidden_next.jsonl"


def switches_path() -> Path:
    return _ev().built_dir() / "switches.json"


def staged() -> list[dict]:
    p = staged_path()
    return _ev()._raw_rows(p) if p.exists() else []


def switches() -> list[dict]:
    try:
        return json.loads(switches_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


# ---------------------------------------------------------------------------
# the rules and the plan
# ---------------------------------------------------------------------------

def _maker(w: dict) -> str:
    wid = str(w.get("id") or "")
    if w.get("kind") == ai_models.LOCAL or wid.startswith(("local/", ai_models.LOCAL)):
        return "the local model"
    return ai_models.maker(wid, w.get("label") or "")


def rules(writer: dict, checker: dict) -> list[dict]:
    """the two rules for a hidden set's writer and checker, each said"""
    wtxt = f"{writer.get('label', '')} {writer.get('id', '')}".lower()
    mw, mc = _maker(writer), _maker(checker)
    return [{"rule": NOT_QWEN, "ok": "qwen" not in wtxt,
             "line": f"writer: {writer.get('label') or 'none'} ({mw})"},
            {"rule": OTHER_MAKER, "ok": bool(mw) and mw != mc,
             "line": f"checker: {checker.get('label') or 'none'} ({mc})"}]


def rules_ok(writer: dict, checker: dict) -> str:
    """'' when both hold; else which does not"""
    bad = [r["rule"] for r in rules(writer, checker) if not r["ok"]]
    return ("for the hidden set, " + " and ".join(bad).replace("the writer is not",
            "the writer must not be").replace("the checker is from", "the checker must be from")
            if bad else "")


def targets() -> dict[str, int]:
    """each group's hidden questions now: the new set replaces them one for one"""
    ev = _ev()
    out: dict[str, int] = {}
    for q in ev._raw_rows(ev.hidden_path()) if ev.hidden_path().exists() else []:
        out[q["group"]] = out.get(q["group"], 0) + 1
    return {g: out[g] for g in ev.groups() if g in out}


def hidden_drafts() -> list[dict]:
    return [d for d in db.qb_list() if (d.get("spec") or {}).get("target") == "hidden"
            and d.get("status") != "cancelled"]


def plan() -> dict:
    """what a new hidden set takes: per group, the models and their rules, and
    the cost — before anything starts"""
    ev = _ev()
    w, c = builder.who("writer"), builder.who("checker")
    have = {}
    for q in staged():
        have[q["group"]] = have.get(q["group"], 0) + 1
    running = {(d.get("spec") or {}).get("group") for d in hidden_drafts()
               if not d.get("published")}
    groups, parts = [], {"writer": 0.0, "checker": 0.0, "judge": 0.0}
    usd, unknown = 0.0, False
    labels = ev.groups()
    for g, n in targets().items():
        write = math.ceil(n * SPARE)
        e = builder.estimate("everyday", write, w, c, g)
        groups.append({"group": g, "label": labels.get(g, g), "target": n,
                       "staged": have.get(g, 0), "write": write, "running": g in running,
                       "estimate": e})
        if e["usd"] is None:
            unknown = True
        else:
            usd += e["usd"]
            for k, v in e["parts"].items():
                parts[k] += v or 0.0
    written = sum(x["write"] for x in groups)
    kept = sum(x["target"] for x in groups)
    line = ("cost not known: no price for a model here" if unknown else
            "free: every step runs on the local model" if usd == 0 else
            f"about ${usd:,.2f} for {written} questions written, {kept} kept")
    why = rules_ok(w, c) or builder.blocked("writer") or builder.blocked("checker") or (
        "a hidden set's drafts are under way" if running else "")
    todo = [x for x in groups if x["staged"] < x["target"]]
    return {"groups": groups, "rules": rules(w, c), "writer": w, "checker": c,
            "judge": builder.who("judge"),
            "estimate": {"usd": None if unknown else round(usd, 2), "line": line,
                         "parts": {k: round(v, 2) for k, v in parts.items()}},
            "can_start": not why and bool(todo), "why": why or ("" if todo else
                                                               "every group has its questions"),
            "can_switch": bool(groups) and not todo and not running,
            "switches": switches()}


def start(by: str) -> list[str]:
    """one draft a group still short of its new questions, for the hidden set"""
    if not config.is_owner(by):
        raise PermissionError("Only the board's owner writes a new hidden set")
    p = plan()
    if not p["can_start"]:
        raise ValueError(p["why"])
    made = []
    for g in p["groups"]:
        if g["staged"] >= g["target"]:
            continue
        d = builder.create({"kind": "everyday", "group": g["group"], "target": "hidden",
                            "count": math.ceil((g["target"] - g["staged"]) * SPARE),
                            "dedup": True}, by)
        made.append(d["id"])
    return made


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

def candidates() -> list[dict]:
    """the old hidden questions worth retiring rather than making practice —
    each with why: every model on the board passes it, or fails it (it tells
    them apart no more), or it was edited since it was written"""
    ev = _ev()
    rows = ev._raw_rows(ev.hidden_path()) if ev.hidden_path().exists() else []
    now = ev.version()["hash"]
    seen: dict[str, list[bool]] = {}
    for f in sorted(config.OUT_DIR.glob("*/everyday.json")) if config.OUT_DIR.is_dir() else []:
        e = ev.read(f.parent) or {}
        if (e.get("version") or {}).get("hash") != now:
            continue
        for it in e.get("items") or []:
            if it.get("pass") is not None:
                seen.setdefault(it["id"], []).append(bool(it["pass"]))
    edited = {x.get("id") for x in ev.edit_log()}
    out = []
    for q in rows:
        why = []
        marks = seen.get(q["id"]) or []
        if len(marks) >= 2 and all(marks):
            why.append(f"every model passes it ({len(marks)} of {len(marks)})")
        elif len(marks) >= 2 and not any(marks):
            why.append(f"every model fails it (0 of {len(marks)})")
        if q["id"] in edited:
            why.append("edited since it was written")
        if why:
            out.append({"id": q["id"], "group": q["group"], "prompt": q["prompt"],
                        "why": why})
    return out


def switch(by: str, retire: list[str], confirm: bool = False) -> dict:
    """the new set in, the old one out — to practice, or retired"""
    from . import hidden_store
    ev = _ev()
    if not config.is_owner(by):
        raise PermissionError("Only the board's owner switches the hidden set")
    p = plan()
    if not p["can_switch"]:
        raise ValueError("every group needs its new questions first: " + ", ".join(
            f"{g['label']} {g['staged']} of {g['target']}" for g in p["groups"]
            if g["staged"] < g["target"]))
    if confirm is not True:
        raise ValueError("switching moves every model's Everyday score to History: confirm it")
    old = ev._raw_rows(ev.hidden_path())
    new = staged()
    unknown = sorted(set(retire) - {q["id"] for q in old})
    if unknown:
        raise ValueError(f"not in the hidden set: {', '.join(unknown)}")
    before = ev.version()["hash"]
    hidden_store.backup()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    keep = [{**q, "half": ev.PRACTICE, "moved_from_hidden": stamp} for q in old
            if q["id"] not in set(retire)]
    # the old set: kept beside, never overwritten
    ev.hidden_path().rename(ev.hidden_path().with_name(f"hidden.before-{stamp}.jsonl"))
    tmp = ev.hidden_path().with_suffix(".part")
    tmp.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in new),
                   encoding="utf-8")
    tmp.replace(ev.hidden_path())
    with open(ev.built_path(), "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in keep)
    staged_path().rename(staged_path().with_name(f"hidden_next.switched-{stamp}.jsonl"))
    rec = {"at": time.time(), "by": by.strip()[:80], "old_digest": ev.hidden_digest(old),
           "new_digest": ev.hidden_digest(new), "old_version": before,
           "new_version": ev.version()["hash"], "count": len(new),
           "to_practice": len(keep), "retired": sorted(retire)}
    switches_path().write_text(json.dumps(switches() + [rec], indent=1), encoding="utf-8")
    db.hidden_audit_add(rec["by"], "Everyday's hidden set switched: the old one to practice "
                        f"({len(keep)}), retired ({len(retire)})", len(old))
    hidden_store.backup()
    return rec
