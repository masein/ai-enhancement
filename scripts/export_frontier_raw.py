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
- items.jsonl: one line an answer this run brought (17g: an import's shard
  its own questions, a run whose answers were replaced since none, said in
  the README) — its benchmark, question id, run, right or
  wrong, ran out of room, tokens, the grade. The answer's text only for the
  benchmarks whose questions may be shown (MMLU-Pro, SimpleQA Verified,
  ARC-AGI-2): GPQA Diamond, OTIS Mock AIME and Humanity's Last Exam are gated
  and ask not to be redistributed, and MATH's problems are withheld by Epoch;
  an answer quotes its question, so theirs stay on the server;
- log.txt: the run's log, each line quoting a hidden or gated question
  withheld (17g), as the board's log view withholds it;
- README.md.

17h: written by a list of what may go out, field by field (scripts/
export_safe.py) — the launch's flags and environment by name, the log only
the runner's own lines, never one quoting a gated question (and no log when
the questions can't be loaded to check it) — then scrubbed as DeviceMark's
are. Public only for a model whose page marks its weights public (or an
import's --public-weights), and only after a typed yes to the list the
export prints; --private keeps every run private. A box's address is never
in it.
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
import export_safe as es  # noqa: E402
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


# 17h: what each file may hold, field by field — anything else is left out
# (export_safe.pick). The launch's flags and environment by name, never whole
SETUP_TASK = {"version": es.TEXT, "task": es.NAME, "protocol": es.TEXT,
              "protocol_version": es.TEXT,
              "source": {"hf": es.NAME, "config": es.NAME, "split": es.NAME,
                         "revision": es.NAME, "gated": bool, "licence": es.TEXT},
              "epochs": int, "thinking": es.TEXT, "budget": int, "family": es.TEXT,
              "sampling": {"*": float}, "where": es.TEXT, "file": es.NAME,
              "server": {"file": es.NAME, "size": int, "build": es.TEXT},
              "launch_setup": {"spec": [es.NAME], "drafts": bool},
              "shard": {"i": int, "n": int}}
BOX = {"box": es.NAME, "image": es.NAME, "gpu": es.TEXT, "gguf_sha256": es.NAME,
       "bundle": es.NAME, "sha256": es.NAME, "started_at": es.TEXT, "finished_at": es.TEXT,
       "hours": float, "sessions": int, "restarts": int, "shard": [int]}
GRADER = {"label": es.TEXT, "model": es.NAME, "version": es.NAME, "provider": es.TEXT,
          "prompt_words": es.TEXT, "prompt_sha256": es.NAME, "n": int}
SCORE = {"alias": es.NAME, "acc,none": float, "acc_stderr,none": float,
         "acc_code,none": float, "acc_code_stderr,none": float,
         "frontier": {"version": es.TEXT, "protocol": es.TEXT, "epochs": int, "questions": int,
                      "answers": int, "ran_out": int, "unread": int, "errors": int,
                      "thinking_held": int, "ungraded": int, "unanswered": int,
                      "unanswered_ids": [es.NAME], "budget": int, "sampling": {"*": float},
                      "family": es.TEXT, "where": es.TEXT, "thinking": es.TEXT,
                      "scored_by": es.TEXT, "code": {"score": float, "se": float},
                      "grader": {**GRADER, "topup": GRADER}, "graders": [GRADER],
                      "final": bool, "look": {"done": int, "waiting": int}, "runs": [int]}}


def task_setup(t: dict) -> tuple[dict, int]:
    """a task's setup as it may go out, and how many of its launch's flags and
    variables were left out"""
    out = es.pick(t, SETUP_TASK) or {}
    launch = t.get("launch") if isinstance(t.get("launch"), dict) else {}
    ls = t.get("launch_setup") if isinstance(t.get("launch_setup"), dict) else {}
    fl, l1 = es.flags(launch.get("flags"))
    ev, l2 = es.env(launch.get("env"))
    ev2, l3 = es.env(ls.get("env") or {})
    an, l4 = es.flags(ls.get("answers") or [])
    out["launch"] = {"flags": fl, "env": {**ev2, **ev}}
    if an:
        out.setdefault("launch_setup", {})["answers"] = an
    return out, l1 + l2 + l3 + l4


def known_public(r: dict) -> bool:
    """17h: a model the board was told is public (its page, or the import's
    --public-weights) — never by its name (teamacct/bonsai-2-27b read public),
    never by a flag of the export's, and never an in-house build (a Qwen3.6
    one), marked or not"""
    from service import served
    model = r["hf_id"]
    based = (served.get(model) or {}).get("based_on") or "" if served.is_served(model) else ""
    if dmx.IN_HOUSE.search(model) or dmx.IN_HOUSE.search(based):
        return False
    return es.public(model)


def brought(r: dict, t: str, rented: dict, reg: dict) -> tuple[set | None, str]:
    """17g: the answers of `t` this run brought, by (question, run) — None for
    every one the row holds — and, when none, why. 17h: this server's runs by
    the run each answer names (two runs on one row exported the same); one
    answered before answers named their run, the row's latest run of it"""
    from service import config, db
    from service import frontier as sf
    x = (reg.get("tasks") or {}).get(t) or {}
    came = {x.get("sha256")} | {b.get("sha256") for b in x.get("shard_bundles") or []}
    if not rented:
        if x:
            return set(), f"its answers were replaced by an import ({x.get('bundle')})"
        got = sf.read_answers(sf.task_dir(row_of(r), t) / sf.ANSWERS)
        mine = {k for k, a in got.items() if a.get("run") == r["id"]}
        unnamed = {k for k, a in got.items() if not a.get("run")}
        if unnamed:
            latest = max((s["id"] for s in db.recent(5000) if s.get("suite") == "frontier"
                          and s.get("hf_id") == r["hf_id"] and not s.get("where_ran")
                          and bool(s.get("thinking")) == bool(r.get("thinking"))
                          and t in set(json.loads(s.get("tasks") or "[]") or fb.TASKS)),
                         default=r["id"])
            if latest == r["id"]:
                mine |= unnamed
        return mine, ("" if mine else "its answers on the row are another run's")
    if rented.get("sha256") not in came:
        return set(), ("replaced since by another import, or its shards aren't all in yet")
    sh = rented.get("shard")
    if not sh:
        return None, ""
    ids = {it["id"] for it in fb.shard_of(fb.load(t, config.BENCH_ROOT), tuple(sh))}
    return {(q, e) for q in ids for e in range(fb.BENCH[t]["epochs"])}, ""


def questions(tasks: list[str]) -> es.Questions | None:
    """17h: the gated benchmarks' questions, every one — a log line quoting
    any six words of one in a row is left out. None when one couldn't be
    loaded: no log at all, rather than one unchecked"""
    from service import config
    texts = []
    for t in tasks:
        if t in SHOWN:
            continue
        try:
            items = fb.load(t, config.BENCH_ROOT)
        except Exception:                               # noqa: BLE001 — fail closed
            return None
        texts += [str(it[k]) for it in items for k in ("question", "problem", "prompt")
                  if it.get(k)]
    return es.Questions(texts)


def export_run(sid: int, out: Path, public: bool | None = None,
               publish: bool = True) -> Path:
    """a run's folder. Public only for a model the board knows as public, and
    only with `publish` (the typed yes); --private keeps it private"""
    from service import config
    from service import frontier as sf
    r = run_of(sid)
    row = row_of(r)
    tasks = [t for t in fb.TASKS if t in set(json.loads(r.get("tasks") or "[]") or fb.TASKS)]
    rented = imf.rented_of(sid) or {}
    reg = imf.registry(row)
    where = r.get("where_ran") or "this server"
    per_setup, scores, items, notes, left_launch = {}, {}, [], {}, 0
    for t in tasks:
        d = sf.task_dir(row, t)
        try:
            raw = json.loads((d / sf.SETUP).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        per_setup[t], n = task_setup(raw)
        left_launch += n
        res = sorted(d.glob("results_*.json"))
        if res:
            blob = json.loads(res[-1].read_text(encoding="utf-8"))
            scores[t] = es.pick({**(blob.get("results") or {}).get(t, {}),
                                 "frontier": blob.get("frontier") or {}}, SCORE) or {}
        keys, why = brought(r, t, rented, reg)
        if why:
            notes[t] = why
        m = sf.marks(row, t)
        for it in m["items"]:
            for x in m["runs"].get(it["id"], []):
                if keys is not None and (it["id"], x["epoch"]) not in keys:
                    continue
                g = x.get("grade") or {}
                items.append({"task": t, "id": str(it["id"]), "epoch": x["epoch"],
                              "ok": x["ok"], "ran_out": x["ran_out"],
                              **({"finish": x["finish"]} if x.get("finish") in (
                                  "stop", "length") else {}),
                              **({"tokens": x["tokens"]} if isinstance(x.get("tokens"), int)
                                 else {}),
                              "unanswered": bool(x.get("unanswered")),
                              **({"grade": {"ok": g.get("ok"), "by": g.get("by")}} if g else {}),
                              **({"answer": x["answer"]} if t in SHOWN else {})})
    setup = {"run": sid, "model": r["hf_id"], "thinking": bool(r.get("thinking")),
             "where": where, "tasks": per_setup,
             **({"box": es.pick(rented, BOX)} if rented else {})}
    pub = known_public(r) and public is not False and publish
    dest = out / ("public" if pub else "private") / f"{row.name}-frontier-run-{sid}"
    dest.mkdir(parents=True, exist_ok=True)
    log = config.LOGS_DIR / f"service_{sid}_{r['hf_id'].replace('/', '__')}.log"
    lines, left = es.log_lines(log.read_text(encoding="utf-8", errors="replace")
                               if log.exists() else "", questions(tasks))
    (dest / "setup.json").write_text(es.dumps(dmx.scrub_obj(setup)), encoding="utf-8")
    (dest / "scores.json").write_text(es.dumps(dmx.scrub_obj(scores)), encoding="utf-8")
    (dest / "items.jsonl").write_text("".join(json.dumps(dmx.scrub_obj(x), ensure_ascii=False)
                                              + "\n" for x in items), encoding="utf-8")
    (dest / "log.txt").write_text(dmx.scrub("\n".join(lines) + ("\n" if lines else "")),
                                  encoding="utf-8")
    withheld = [fb.BENCH[t]["label"] for t in tasks if t not in SHOWN]
    (dest / "README.md").write_text(
        f"# {r['hf_id']} · the Frontier benchmarks · run #{sid}\n\n"
        f"Where it ran: {where}.\n\n"
        + "".join(f"- {fb.BENCH[t]['label']}: {fb.BENCH[t]['protocol']} "
                  f"({fb.BENCH[t]['protocol_version']})\n" for t in tasks)
        + (f"\nThe answers' text is withheld for {', '.join(withheld)}: their questions are "
           "gated or withheld from redistribution, and an answer quotes its question.\n"
           if withheld else "")
        + "".join(f"\n{fb.BENCH[t]['label']}: no answers from this run — {w}.\n"
                  for t, w in notes.items())
        + "\nWritten by a list of what may go out, field by field; everything else left out"
        + (f" ({left_launch} launch flags or variables)" if left_launch else "") + ". "
        + "The log holds only the runner's own lines"
        + (": " + "; ".join(f"{n} left out ({w})" for w, n in left.items()) if left else "")
        + ". Then scrubbed: " + "; ".join(dmx.WHAT_THE_SCRUB_REMOVES) + ".\n",
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
    ap.add_argument("--private", action="store_true",
                    help="every run private, even a model marked public")
    ap.add_argument("--public", action="store_true",
                    help="17h: no effect — a model is public when its page marks its weights "
                         "public, and the export asks before it writes anything there")
    a = ap.parse_args(argv)
    from service import config, db
    db.init()
    out = a.out or (config.BENCH_ROOT / "raw-export")
    ids = [a.run] if a.run else runs(a.rented)
    if a.public:
        print("--public: no effect — a model is public when its page marks its weights public")
    # 17h: what would go to public/, listed, and a typed yes before any of it
    would = [] if a.private else [db.get(sid)["hf_id"] for sid in ids
                                  if known_public(run_of(sid))]
    publish = es.confirm(would)
    if would and not publish:
        print("not published: those runs go to private/")
    for sid in ids:
        print(export_run(sid, out, False if a.private else None, publish=publish))
    print(f"{len(ids)} run(s) under {out} — upload: docs/REMOTE-RUNS.md § Publishing the raw runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
