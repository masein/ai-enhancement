"""16.1: a model's size as a person enters it — its total parameters, and the
active ones where they differ (a mixture of experts: "35B · 3B active").

The board takes a size, in order, from what a person entered here, the
harness's count, a GGUF file's header, the board model it is based on, and —
for a Hub model alone — its name (report_lm_eval.size_of). A served model's
or a GGUF file's name is never read: it is the person's words, suggested in
the form and confirmed there.

On the server, for rows that have none (the deploy step of 16.1):

    sudo docker compose exec -T bench python -m service.sizes list
    sudo docker compose exec -T bench python -m service.sizes set \\
        --based-on Qwen/Qwen3.6-35B-A3B --total 35B --active 3B --by masein
"""

from __future__ import annotations

import argparse
import re
import sys

from . import db

_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmbt]?)\s*$", re.I)
_UNIT = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse(text) -> float | None:
    """"35B" -> 35e9, "410M" -> 410e6, 3e9 -> 3e9; '' -> None"""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        v = float(text)
    else:
        m = _SIZE.match(str(text))
        if not m:
            raise ValueError(f"{text!r} isn't a size: write it as 35B, 3B or 410M")
        v = float(m.group(1)) * _UNIT[m.group(2).lower()]
    if v <= 0 or v > 1e13:
        raise ValueError(f"{text!r} isn't a size a model has")
    return v


def check(total, active) -> tuple[float, float | None]:
    """(total, active) as numbers: a total, and an active size no larger"""
    t, a = parse(total), parse(active)
    if t is None:
        raise ValueError("A size: its total parameters, like 35B")
    if a is not None and a > t:
        raise ValueError("Its active parameters can't be more than its total")
    return t, (a if a is not None and a < t else None)


def set_size(model: str, total, active, by: str) -> dict:
    by = (by or "").strip()[:80]
    if not by:
        raise ValueError("Who entered it: a name")
    t, a = check(total, active)
    db.size_set(model, t, a, by)
    return {"model": model, "total": t, "active": a, "by": by}


def _words(v: float | None) -> str:
    if not v:
        return "—"
    return f"{v / 1e9:g}B" if v >= 1e9 else f"{v / 1e6:g}M"


def _rows() -> list[dict]:
    """the board's served and GGUF models: their id, name, base and size"""
    out = [{"id": r["id"], "name": r["name"], "based_on": r.get("based_on") or "",
            "kind": "served"} for r in db.served_all()]
    out += [{"id": r["id"], "name": r["name"], "based_on": r.get("based_on") or "",
             "kind": "gguf"} for r in db.gguf_all()]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="16.1: models' sizes, as people enter them")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the served and GGUF models, and the sizes entered for them")
    s = sub.add_parser("set", help="enter a size: for one model, or every one based on one")
    who = s.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="a board model's id")
    who.add_argument("--based-on", help="every served and GGUF model based on this one")
    s.add_argument("--total", required=True, help="its total parameters: 35B")
    s.add_argument("--active", default="", help="its active ones, where they differ: 3B")
    s.add_argument("--by", required=True, help="who entered it")
    s.add_argument("--dry-run", action="store_true", help="say what it would set, set nothing")
    a = ap.parse_args(argv)
    entered = db.sizes_all()
    if a.cmd == "list":
        for r in _rows():
            e = entered.get(r["id"]) or {}
            print(f"{r['id']:<48} {r['kind']:<6} based on {r['based_on'] or '—':<28} "
                  f"{_words(e.get('total'))}" + (f" · {_words(e['active'])} active"
                                                 if e.get("active") else "")
                  + (f" (entered by {e['by']})" if e else " (no size entered)"))
        return 0
    ids = [a.model] if a.model else [r["id"] for r in _rows() if r["based_on"] == a.based_on]
    if not ids:
        print(f"No served or GGUF model is based on {a.based_on}", file=sys.stderr)
        return 1
    try:
        t, act = check(a.total, a.active)
        for mid in ids:
            if not a.dry_run:
                set_size(mid, t, act, a.by)
            print(f"{mid}: {_words(t)}" + (f" · {_words(act)} active" if act else "")
                  + (" (dry run: nothing set)" if a.dry_run else f", entered by {a.by}"))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"{len(ids)} model(s)" + (" would be set" if a.dry_run else " set"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
