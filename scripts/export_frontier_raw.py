#!/usr/bin/env python3
"""17f: a Frontier run's raw record for the Hugging Face raw-run export, as
DeviceMark's rows have theirs (scripts/export_devicemark_raw.py) — the server's
runs and the runs imported from rented boxes alike, each saying where it ran.

    sudo docker compose exec -T -e SCRUB_HOSTS="$(hostname)" bench \\
        python scripts/export_frontier_raw.py --run <run id>
    sudo docker compose exec -T -e SCRUB_HOSTS="$(hostname)" bench \\
        python scripts/export_frontier_raw.py --all

One folder a run, under $BENCH_ROOT/raw-export/private/ (or public/):
- setup.json: the model, where it ran ("this server", or "rented GPU · RTX 5090
  · box A3"), the box's times, sessions, image and GGUF, and each benchmark's
  setup (protocol, revision, budget, sampling, launch);
- scores.json: each benchmark's score as the board scored it, its grader and
  where;
- items.jsonl: one line an answer — its benchmark, question id, run, right or
  wrong, ran out of room, tokens, the grade. The answer's text only for the
  benchmarks whose questions may be shown (MMLU-Pro, SimpleQA Verified,
  ARC-AGI-2): GPQA Diamond, OTIS Mock AIME and Humanity's Last Exam are gated
  and ask not to be redistributed, and MATH's problems are withheld by Epoch;
  an answer quotes its question, so theirs stay on the server;
- log.txt: the run's log;
- README.md.

Private by default for a build served here (a Qwen3.6 build, served/…), as
DeviceMark's; --public or --private decides instead. Every file is scrubbed
as DeviceMark's are (keys, home paths, host names, private addresses), and a
box's address is never in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import export_devicemark_raw as dmx  # noqa: E402
import frontier as fb  # noqa: E402
import import_frontier as imf  # noqa: E402

# the benchmarks whose answers may leave the server: their questions may be shown
SHOWN = ("mmlupro_tiger", "simpleqa_epoch", "arc_agi2_public")


def run_of(sid: int) -> dict:
    from service import db
    r = db.get(sid)
    if not r:
        raise SystemExit(f"#{sid}: no such run")
    if r.get("suite") != "frontier":
        raise SystemExit(f"#{sid} is a {r.get('suite')} run: this exports Frontier runs "
                         "(export_devicemark_raw.py exports DeviceMark's)")
    return r


def row_of(r: dict) -> Path:
    from service import config
    safe = r["hf_id"].replace("/", "__")
    return config.OUT_DIR / (safe + ("__thinking" if r.get("thinking") else ""))


def export_run(sid: int, out: Path, public: bool | None = None) -> Path:
    from service import config
    from service import frontier as sf
    r = run_of(sid)
    row = row_of(r)
    tasks = [t for t in fb.TASKS if t in set(json.loads(r.get("tasks") or "[]") or fb.TASKS)]
    rented = imf.rented_of(sid) or {}
    where = r.get("where_ran") or "this server"
    per_setup, scores, items = {}, {}, []
    for t in tasks:
        d = sf.task_dir(row, t)
        try:
            per_setup[t] = json.loads((d / sf.SETUP).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        res = sorted(d.glob("results_*.json"))
        if res:
            blob = json.loads(res[-1].read_text(encoding="utf-8"))
            scores[t] = {**(blob.get("results") or {}).get(t, {}),
                         "frontier": blob.get("frontier") or {}}
        m = sf.marks(row, t)
        for it in m["items"]:
            for x in m["runs"].get(it["id"], []):
                g = x.get("grade") or {}
                items.append({"task": t, "id": it["id"], "epoch": x["epoch"], "ok": x["ok"],
                              "ran_out": x["ran_out"], "finish": x.get("finish"),
                              "tokens": x.get("tokens"), "unanswered": bool(x.get("unanswered")),
                              **({"grade": {"ok": g.get("ok"), "by": g.get("by")}} if g else {}),
                              **({"answer": x["answer"]} if t in SHOWN else {})})
    setup = {"run": sid, "model": r["hf_id"], "thinking": bool(r.get("thinking")),
             "where": where, "tasks": per_setup,
             **({"box": {k: rented.get(k) for k in (
                 "box", "image", "gpu", "gguf_sha256", "bundle", "sha256", "started_at",
                 "finished_at", "hours", "sessions", "restarts", "shard")}} if rented else {})}
    meta = {"based_on": "", "runtime": "llama-server" if r["hf_id"].startswith("served/") else ""}
    pub = dmx.public_by_default(row, meta) if public is None else public
    dest = out / ("public" if pub else "private") / f"{row.name}-frontier-run-{sid}"
    dest.mkdir(parents=True, exist_ok=True)
    log = config.LOGS_DIR / f"service_{sid}_{r['hf_id'].replace('/', '__')}.log"
    (dest / "setup.json").write_text(json.dumps(dmx.scrub_obj(setup), indent=1), encoding="utf-8")
    (dest / "scores.json").write_text(json.dumps(dmx.scrub_obj(scores), indent=1),
                                      encoding="utf-8")
    (dest / "items.jsonl").write_text("".join(json.dumps(dmx.scrub_obj(x), ensure_ascii=False)
                                              + "\n" for x in items), encoding="utf-8")
    (dest / "log.txt").write_text(dmx.scrub(log.read_text(encoding="utf-8", errors="replace"))
                                  if log.exists() else "", encoding="utf-8")
    withheld = [fb.BENCH[t]["label"] for t in tasks if t not in SHOWN]
    (dest / "README.md").write_text(
        f"# {r['hf_id']} · the Frontier benchmarks · run #{sid}\n\n"
        f"Where it ran: {where}.\n\n"
        + "".join(f"- {fb.BENCH[t]['label']}: {fb.BENCH[t]['protocol']} "
                  f"({fb.BENCH[t]['protocol_version']})\n" for t in tasks)
        + (f"\nThe answers' text is withheld for {', '.join(withheld)}: their questions are "
           "gated or withheld from redistribution, and an answer quotes its question.\n"
           if withheld else "")
        + "\nScrubbed before writing: " + "; ".join(dmx.WHAT_THE_SCRUB_REMOVES) + ".\n",
        encoding="utf-8")
    return dest


def runs(rented_only: bool = False) -> list[int]:
    from service import db
    return sorted(r["id"] for r in db.recent(5000) if r.get("suite") == "frontier"
                  and r.get("status") == "done" and (r.get("where_ran") or not rented_only))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--run", type=int, help="a Frontier run's id")
    pick.add_argument("--all", action="store_true", help="every Frontier run that finished")
    ap.add_argument("--rented", action="store_true", help="with --all: the imported ones only")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the folders go (default $BENCH_ROOT/raw-export)")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="public, whatever the default")
    vis.add_argument("--private", action="store_true", help="private, whatever the default")
    a = ap.parse_args(argv)
    from service import config, db
    db.init()
    out = a.out or (config.BENCH_ROOT / "raw-export")
    public = True if a.public else False if a.private else None
    ids = [a.run] if a.run else runs(a.rented)
    for sid in ids:
        print(export_run(sid, out, public))
    print(f"{len(ids)} run(s) under {out} — upload: docs/REMOTE-RUNS.md § Publishing the raw runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
