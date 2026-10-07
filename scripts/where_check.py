#!/usr/bin/env python3
"""17i: the Runs rows whose "where it ran" isn't what their own records say —
and, with --fix, each set to what they say. 0adb522's start-up fill read
where a run ran from its note: a note typed to read like an import's made a
run on this server "rented", and a GPU whose name holds a bracket ("Tesla V100
(16 GB)") was cut. 511854e's fill touched only blank rows, so those stayed.

What a row's records say, in order:
- its import's record (the Frontier registry, DeviceMark's remote_imports.json):
  the GPUs and the box it ran on — for a row blank or cut (its brackets
  unbalanced: "rented GPU · Tesla V100 (16 GB");
- else its log's first line, when an import wrote it ("===== [12] imported …
  run on a rented GPU (Tesla V100 (16 GB)) ====="): a failed import, never
  recorded as imported — for a row cut;
- else nothing was imported: a row reading "rented GPU …" ran on this server
  (''), its note typed to read like an import's.

Read-only unless --fix. It uses only what the board's image has had since 17g,
so it runs against the live board before a deploy, from the server's checkout:

    sudo docker compose exec -T bench python - < scripts/where_check.py
    sudo docker compose exec -T bench python - --fix < scripts/where_check.py

The board makes the same fix itself at start-up (service/db.py)."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

_HEADER = re.compile(r"^===== \[(?P<sid>\d+)\] imported \S+ \(sha256 [0-9a-f]+\) by .*?"
                     r"run on a rented GPUs? \((?P<gpu>.*)\)(?P<rest>.*?) =====$")
_BOX = re.compile(r" · box (\S+)")


def _paths() -> None:
    here = Path.cwd() / "scripts"
    if here.is_dir() and str(here) not in sys.path:
        sys.path.insert(0, str(here))


def records(out_dir: Path) -> dict[int, str]:
    """each imported run's where, from its row's record of its imports"""
    import import_frontier as imf
    out: dict[int, str] = {}
    for name in (imf.REGISTRY, "remote_imports.json"):
        for f in sorted(out_dir.glob(f"*/{name}")) if out_dir.is_dir() else []:
            try:
                reg = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for x in reg.get("imports") or [] if isinstance(reg, dict) else []:
                sid = x.get("sid") if isinstance(x, dict) else None
                if not isinstance(sid, int):
                    continue
                gpus = [str(g) for g in x.get("gpu_names") or [x.get("gpu") or ""]]
                box = x.get("box") if isinstance(x.get("box"), str) else ""
                out[sid] = imf.where_words(gpus, [box] if box else [])
    return out


def from_log(sid: int, hf_id: str, note: str, logs_dir: Path) -> str | None:
    """a failed import's where, from the first line its import wrote — None
    when its log isn't an import's"""
    import import_frontier as imf
    p = logs_dir / f"service_{sid}_{(hf_id or '').replace('/', '__')}.log"
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            first = next((x.rstrip("\n") for x in fh if x.strip()), "")
    except OSError:
        return None
    m = _HEADER.match(first)
    if not m or int(m["sid"]) != sid:
        return None
    box = _BOX.search(note or "")
    return imf.where_words([m["gpu"]], [box[1]] if box else [])


def wrong(c: sqlite3.Connection, out_dir: Path, logs_dir: Path) -> list[dict]:
    """[{id, model, now, should, why}] — every row whose where isn't its records'"""
    recs = records(out_dir)
    out = []
    for sid, hf_id, where, note in c.execute(
            "SELECT id, hf_id, COALESCE(where_ran, ''), COALESCE(note, '') FROM submissions"):
        whole = where.count("(") == where.count(")")   # 0adb522 cut "Tesla V100 (16 GB"
        if sid in recs:
            if where and whole:
                continue                                # the import's own words, whole
            want, why = recs[sid], "its import's record"
        elif not where:
            continue                                    # no import: this server, as it says
        else:
            got = from_log(sid, hf_id, note, logs_dir)
            if got is None:
                if not where.startswith("rented GPU"):
                    continue
                want, why = "", "nothing was imported: it ran on this server"
            elif whole:
                continue                                # a failed import's own words, whole
            else:
                want, why = got, "its import's log"
        if where != want:
            out.append({"id": sid, "model": hf_id, "now": where, "should": want, "why": why})
    return out


def fix(c: sqlite3.Connection, out_dir: Path, logs_dir: Path) -> list[dict]:
    got = wrong(c, out_dir, logs_dir)
    for x in got:
        c.execute("UPDATE submissions SET where_ran=? WHERE id=?", (x["should"], x["id"]))
    return got


def main(argv: list[str] | None = None) -> int:
    _paths()
    from service import config
    argv = sys.argv[1:] if argv is None else argv
    c = sqlite3.connect(config.DB_PATH)
    try:
        got = (fix if "--fix" in argv else wrong)(c, Path(config.OUT_DIR),
                                                  Path(config.LOGS_DIR))
        c.commit()
    finally:
        c.close()
    for x in got:
        print(f"#{x['id']} {x['model']}: {x['now'] or 'this server'!r} → "
              f"{x['should'] or 'this server'!r} ({x['why']})")
    print(f"{len(got)} row(s) {'set to' if '--fix' in argv else 'not as'} their records say"
          + ("" if got or "--fix" in argv else " — none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
