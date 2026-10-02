#!/usr/bin/env python3
"""15.2: a bundle from a rented GPU (scripts/remote_run.py), into the board.
Run it inside the container, where the board's results are:

    sudo docker compose exec -T bench python scripts/import_remote.py \\
        /home/masein/benchmarks/bundles/<bundle>.tar.gz --by masein
    sudo docker compose exec -T bench python scripts/import_remote.py --battery

The checks. It refuses the bundle, saying which check failed, when any of
these differ from this server's:
- the battery's hashes: its ids, its prompt templates, every item as built;
- the protocol's version (and its cap and seed);
- the pinned libraries: torch and its CUDA, transformers, lm_eval, the fast
  kernels.
A bundle whose answers don't cover a task it names is refused too.

The merge. Each task the bundle holds replaces that task's answers here for
the same model and mode — the earlier ones kept under results/earlier/ — and
the tasks it doesn't hold stay as they are: Qwen3.5-4B's IFEval from a rented
GPU joins the MMLU-Pro and MATH answers #167 left on the server.

The row is then scored by the board's own scoring (service/devicemark.mark_hf)
from the answers on disk, and its setup says which tasks ran on a rented GPU
(<its name>); the model page shows that line. The Runs list gets an entry for
the import, its log the bundle's.

A bundle already imported — by its sha256 — changes nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import remote_bundle as rb  # noqa: E402

REFUSED = 2
BATTERY_WORDS = {"battery": "ids (battery-v1.json)", "prompts": "prompt templates (prompts.json)",
                 "items": "items (each question, key and prompt)"}


def server_battery() -> dict:
    import devicemark as dm
    from service import config
    return dm.battery_hashes(dm.load_items(config.DM_ITEMS))


def _bundle_answers(b: dict, row: str, task: str) -> dict[str, str]:
    """a task's answers as the bundle holds them: its newest samples file"""
    names = sorted(n for n in b["files"] if n.startswith(f"results/{row}/{task}_0shot/")
                   and Path(n).name.startswith(f"samples_{task}_"))
    out: dict[str, str] = {}
    for line in b["files"][names[-1]].decode("utf-8").splitlines() if names else []:
        try:
            s = json.loads(line)
        except ValueError:
            continue
        key = (s.get("doc") or {}).get("key")
        if key is not None:
            out[str(key)] = ((s.get("resps") or [[""]])[0] or [""])[0] or ""
    return out


def checks(b: dict) -> list[str]:
    """every check the bundle fails, in words; [] when it may be imported"""
    import devicemark as dm
    bundle, setup = b["bundle"], b["setup"]
    if bundle.get("format") != rb.FORMAT:
        return [f"its format: the bundle's {bundle.get('format')}, this board reads {rb.FORMAT}"]
    if bundle.get("suite") not in rb.SUITES:
        return [f"its suite: {bundle.get('suite')!r}, which this board doesn't import"]
    out = []
    if (setup.get("protocol"), setup.get("cap"), setup.get("seed")) != (dm.VERSION, dm.CAP,
                                                                         dm.SEED):
        out.append(f"the protocol: the bundle's {setup.get('protocol')} (cap {setup.get('cap')}, "
                   f"seed {setup.get('seed')}), this server's {dm.VERSION} (cap {dm.CAP}, "
                   f"seed {dm.SEED})")
    theirs, ours = setup.get("battery") or {}, server_battery()
    for k, words in BATTERY_WORDS.items():
        if theirs.get(k) != ours[k]:
            out.append(f"the battery's {words}: the bundle's hash {str(theirs.get(k))[:16]}, "
                       f"this server's {ours[k][:16]}")
    out += [f"a pinned library — {d}" for d in
            rb.pin_differences(setup.get("libraries") or {}, rb.library_versions())]
    model, row = bundle.get("model") or "", bundle.get("row") or ""
    safe = model.replace("/", "__")
    if not model or row not in (safe, safe + "__thinking") \
            or row.endswith("__thinking") and not bundle.get("thinking"):
        out.append(f"its row {row!r} isn't {model!r}'s, thinking "
                   f"{'on' if bundle.get('thinking') else 'off'}")
        return out
    for t in bundle.get("tasks") or {}:
        if t not in rb.SUITES[bundle["suite"]]["tasks"]:
            out.append(f"{t}: not a task of {bundle['suite']}")
            continue
        want = {k for bb, k in dm.keys_for("full") if dm.TASK[bb] == t}
        got = set(_bundle_answers(b, row, t))
        if not want <= got:
            out.append(f"{t}: {len(want & got)} of {len(want)} items answered")
    if not bundle.get("tasks"):
        out.append("it holds no task answered whole")
    return out


def registry(row: Path) -> dict:
    from service import devicemark as sdm
    try:
        return json.loads((row / sdm.REMOTE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"imports": [], "tasks": {}}


def _write_registry(row: Path, reg: dict) -> None:
    from service import devicemark as sdm
    row.mkdir(parents=True, exist_ok=True)
    tmp = row / (sdm.REMOTE_NAME + ".part")
    tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    tmp.replace(row / sdm.REMOTE_NAME)


def import_bundle(path: Path, by: str, say=print) -> int:
    import devicemark as dm
    from service import config, db
    from service import devicemark as sdm
    try:
        b = rb.read(path)
    except (ValueError, OSError) as e:
        say(f"refused: {e}")
        return REFUSED
    bundle, setup = b["bundle"], b["setup"]
    model, thinking = bundle.get("model") or "", bool(bundle.get("thinking"))
    row_name = bundle.get("row") or ""
    say(f"{path.name} · sha256 {b['sha256'][:16]} · {model} · thinking "
        f"{'on' if thinking else 'off'} · {', '.join(bundle.get('tasks') or {}) or 'no task'}")
    row = config.OUT_DIR / row_name if row_name else None
    reg = registry(row) if row else {"imports": [], "tasks": {}}
    was = next((x for x in reg.get("imports") or [] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, as Runs #{was['sid']} on {was['at']}: nothing changed")
        return 0
    bad = checks(b)
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    libs = setup.get("libraries") or {}
    gpu = (setup.get("gpu") or {}).get("name") or "a GPU"
    say(f"as this server's: the battery ({setup['battery']['items'][:16]}), the protocol "
        f"({dm.VERSION}), torch {libs.get('torch')} (CUDA {libs.get('torch_cuda')}), "
        f"transformers {libs.get('transformers')}, lm_eval {libs.get('lm_eval')}, the fast "
        f"kernels")

    # the merge: each task the bundle holds replaces this server's, kept aside
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    tasks = list(bundle["tasks"])
    base = config.OUT_DIR / model.replace("/", "__")
    lines = []
    for t in tasks:
        dest = row / f"{t}_0shot"
        if dest.exists():
            aside = config.OUT_DIR.with_name("earlier") / row_name / f"{t}_0shot-before-import-{stamp}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(aside))
            lines.append(f"{t}: the answers here before are kept at {aside}")
    for name, data in sorted(b["files"].items()):
        if not name.startswith("results/"):
            continue
        rel = Path(name).relative_to("results")
        if rel.parts[0] not in (row_name, base.name):
            continue
        target = config.OUT_DIR / rel
        if rel.name == "model_meta.json" and len(rel.parts) == 2:
            if target.exists():
                continue                      # the server's own record of the model stands
        elif not (rel.parts[0] == row_name and len(rel.parts) > 2
                  and rel.parts[1] in {f"{t}_0shot" for t in tasks}):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for t in tasks:
        n = bundle["tasks"][t]
        lines.append(f"{t}: {n} answers from a rented GPU ({gpu})")
        reg.setdefault("tasks", {})[t] = {
            "gpu": gpu, "sha256": b["sha256"], "bundle": path.name, "at": stamp,
            "samples": [str(p.relative_to(row)) for p in
                        sorted((row / f"{t}_0shot").rglob(f"samples_{t}_*.jsonl"))]}
    _write_registry(row, reg)

    # the row, by the board's own scoring, from what is on disk now
    have = [t for t in rb.SUITES[bundle["suite"]]["tasks"]
            if (row / f"{t}_0shot").is_dir() and rb.task_answers(row / f"{t}_0shot", t)]
    missing = [t for t in rb.SUITES[bundle["suite"]]["tasks"] if t not in have]
    status, error = "done", ""
    if missing:
        line = f"answers imported; the row is scored once {', '.join(missing)} is here too"
    else:
        try:
            line = sdm.mark_hf(0, {"hf_id": model}, {"revision": setup.get("revision")}, row,
                               {"on": thinking, "mode": setup.get("thinking_mode")})
        except Exception as e:                      # noqa: BLE001 — the answers are on disk
            status, error, line = "failed", f"scoring: {e}", f"answers imported; scoring: {e}"
    where = sdm.where_of(row).get("where") or f"run on a rented GPU ({gpu})"

    # the Runs list: an entry for the import, with the bundle's log
    sid = db.add(model, "instruct", bundle["suite"], by,
                 f"imported from a rented GPU ({gpu}) · {path.name}", thinking=thinking,
                 part="full", tasks=tasks if len(tasks) < len(rb.SUITES[bundle["suite"]]["tasks"])
                 else None, status=status)
    db.update(sid, finished_at=time.time(), progress=line, error=error)
    log = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"===== [{sid}] imported {path.name} (sha256 {b['sha256'][:16]}) by {by}: "
                   f"{', '.join(tasks)} {where} =====\n" + b["log"]
                   + "".join(f"\n[import] {x}" for x in lines) + f"\n[import] {line}\n",
                   encoding="utf-8")
    reg.setdefault("imports", []).append({"sha256": b["sha256"], "sid": sid, "tasks": tasks,
                                          "gpu": gpu, "by": by, "at": stamp, "bundle": path.name})
    _write_registry(row, reg)
    for x in lines:
        say(x)
    say(f"the row {model}{' · thinking' if row_name.endswith('__thinking') else ''}: {line} · "
        f"{where}")
    say(f"Runs #{sid}, its log the bundle's")
    return 0 if status == "done" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", type=Path)
    ap.add_argument("--by", default="", help="your name, for the Runs list")
    ap.add_argument("--battery", action="store_true",
                    help="this server's battery hashes, for remote_run.py --battery")
    a = ap.parse_args(argv)
    from service import db
    db.init()
    if a.battery:
        h = server_battery()
        print(f"items {h['items']}\nbattery {h['battery']}\nprompts {h['prompts']}")
        return 0
    if not a.bundle:
        ap.error("a bundle, or --battery")
    if not a.by.strip():
        ap.error("--by: your name, for the Runs list")
    return import_bundle(a.bundle, a.by.strip()[:80])


if __name__ == "__main__":
    raise SystemExit(main())
