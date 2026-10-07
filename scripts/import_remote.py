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

17: a GGUF's Frontier bundle (scripts/remote_gguf.py) is imported by
scripts/import_frontier.py, onto the served model's row: it is refused unless
its GGUF is the file registered for that model (--file-sha256 for a model
whose file isn't registered here: the sha256 of the file its server serves).

    sudo docker compose exec -T bench python scripts/import_remote.py \
        /home/masein/benchmarks/bundles/frontier-<…>.tar.gz --by masein

15.5: shards. A bundle from `remote_run.py --shard i/n` holds shard i of n of
each task (devicemark.shard_of: every n-th item from the i-th) and is checked
for exactly those items. It waits under results/shards/<row>/<task>/<i>-of-<n>/
until all n shards of the task are in — the import says which are missing —
and then the shards are merged into the task's answers, in the battery's
order, as one run would have written them, and the row is scored as above.
The shards come in any order; a shard imported twice changes nothing; a newer
bundle of the same shard replaces it, the earlier one kept under
results/earlier/. Every shard of a task has the same n.
"""

from __future__ import annotations

import argparse
import json
import re
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
    if bundle.get("suite") != "devicemark":
        return [f"its suite: {bundle.get('suite')!r}, which this board doesn't import"]
    try:
        shard = rb.shard_of_bundle(bundle)
    except ValueError as e:
        return [str(e)]
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
    # 15.6: and this server's items are the repo's (items-v1.sha256.json)
    from service import config
    if config.DM_ITEMS_SHA256 and ours["items"] != config.DM_ITEMS_SHA256:
        out.append(f"this server's battery: its items hash to {ours['items'][:16]}, and the repo "
                   f"expects {config.DM_ITEMS_SHA256[:16]}")
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
        # 15.5: a shard's are its share of the task's items, and no others
        want = set(dm.shard_of([k for bb, k in dm.keys_for("full") if dm.TASK[bb] == t], shard))
        got = set(_bundle_answers(b, row, t))
        of = f" (shard {shard[0]} of {shard[1]})" if shard else ""
        if not want <= got:
            out.append(f"{t}: {len(want & got)} of {len(want)} items answered{of}")
        elif shard and got - want:
            out.append(f"{t}: {len(got - want)} answers to items outside shard {shard[0]} of "
                       f"{shard[1]}")
    if not bundle.get("tasks"):
        out.append("it holds no task answered whole")
    return out


def registry(row: Path) -> dict:
    from service import devicemark as sdm
    try:
        return json.loads((row / sdm.REMOTE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"imports": [], "tasks": {}}


def shards_dir(row_name: str) -> Path:
    """15.5: where a task's shards wait until every one of them is in"""
    from service import config
    return config.OUT_DIR.with_name("shards") / row_name


def _and(xs: list) -> str:
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _shards(js: list[int], n: int) -> str:
    """"shard 2 of 3", "shards 1 and 2 of 3" """
    return f"shard{'s' if len(js) > 1 else ''} {_and(js)} of {n}"


def shard_conflicts(reg: dict, tasks: list[str], shard: tuple[int, int]) -> list[str]:
    """15.5: a task whose shards here, still waiting for the rest, are of
    another n than this bundle's"""
    i, n = shard
    out = []
    for t in tasks:
        sh = (reg.get("shards") or {}).get(t) or {}
        m, have = sh.get("n"), sorted(int(j) for j in sh.get("have") or {})
        if m and m != n and len(have) < m:
            out.append(f"{t}: {_shards(have, m)} {'is' if len(have) == 1 else 'are'} here, "
                       f"waiting for the rest, and this bundle is shard {i} of {n} — every "
                       f"shard of a task is one of the same n")
    return out


def merge_shards(row: Path, t: str, n: int, stamp: str) -> int:
    """15.5: a task's n shards as one run's answers: their samples in one
    file, in the battery's order, each line's doc_id its place in the whole
    task (as lm_eval numbers a whole run's) — the row's task folder"""
    import devicemark as dm
    every = [k for bb, k in dm.keys_for("full") if dm.TASK[bb] == t]
    place = {k: p for p, k in enumerate(every)}
    got: dict[str, dict] = {}
    sub = None
    for j in range(1, n + 1):
        slot = shards_dir(row.name) / t / f"{j}-of-{n}"
        newest = rb.samples_files(slot, t)[-1]
        sub = sub or newest.parent.relative_to(slot)
        for line in newest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                s["doc_id"] = place[str(s["doc"]["key"])]
                got[str(s["doc"]["key"])] = s
    if set(got) != set(every):
        raise ValueError(f"{t}: the shards hold {len(got)} of its {len(every)} items")
    # 15.7: the form the replies were saved in, the same in every shard
    forms = {dm.reply_form(shards_dir(row.name) / t / f"{j}-of-{n}") for j in range(1, n + 1)}
    if len(forms) > 1:
        raise ValueError(f"{t}: its shards' replies were saved in different forms ({forms})")
    dest = row / f"{t}_0shot" / sub
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"samples_{t}_{stamp}.jsonl").write_text(
        "".join(json.dumps(got[k], ensure_ascii=False) + "\n" for k in every), encoding="utf-8")
    form = forms.pop()
    if form:
        (row / f"{t}_0shot" / dm.FORM_NAME).write_text(json.dumps({"form": form}),
                                                       encoding="utf-8")
    return len(every)


def _write_registry(row: Path, reg: dict) -> None:
    from service import devicemark as sdm
    row.mkdir(parents=True, exist_ok=True)
    tmp = row / (sdm.REMOTE_NAME + ".part")
    tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    tmp.replace(row / sdm.REMOTE_NAME)


def import_bundle(path: Path, by: str, say=print, file_sha: str = "", register: str = "",
                  aside: bool = False, public_weights: bool = False) -> int:
    try:
        b = rb.read(path)
    except (ValueError, OSError) as e:
        say(f"refused: {e}")
        return REFUSED
    # 17: a GGUF's Frontier run, onto a served model's row
    if b["bundle"].get("suite") == "frontier":
        import import_frontier
        return import_frontier.import_bundle(b, path, by, say, file_sha, register, aside,
                                             public_weights=public_weights)
    if file_sha or register or public_weights:
        say("refused — --file-sha256 and --register are for a GGUF's Frontier bundle")
        return REFUSED
    return import_devicemark(b, path, by, say)


def import_devicemark(b: dict, path: Path, by: str, say=print) -> int:
    import devicemark as dm
    from service import config, db
    from service import devicemark as sdm
    bundle, setup = b["bundle"], b["setup"]
    model, thinking = bundle.get("model") or "", bool(bundle.get("thinking"))
    row_name = bundle.get("row") or ""
    try:
        shard = rb.shard_of_bundle(bundle)
    except ValueError:
        shard = None                                  # refused by checks, below
    say(f"{path.name} · sha256 {b['sha256'][:16]} · {model} · thinking "
        f"{'on' if thinking else 'off'} · {', '.join(bundle.get('tasks') or {}) or 'no task'}"
        + (f" · shard {shard[0]} of {shard[1]}" if shard else ""))
    row = config.OUT_DIR / row_name if row_name else None
    reg = registry(row) if row else {"imports": [], "tasks": {}}
    was = next((x for x in reg.get("imports") or [] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, as Runs #{was['sid']} on {was['at']}: nothing changed")
        return 0
    bad = checks(b)
    if not bad and shard:
        bad = shard_conflicts(reg, [t for t in rb.SUITES[bundle["suite"]]["tasks"]
                                    if t in bundle["tasks"]], shard)
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

    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    tasks = [t for t in rb.SUITES[bundle["suite"]]["tasks"] if t in bundle["tasks"]]
    base = config.OUT_DIR / model.replace("/", "__")
    earlier = config.OUT_DIR.with_name("earlier") / row_name
    lines: list[str] = []
    # 15.5: a shard waits for the task's others; the tasks every shard of which
    # is in now go into the row, merged — a whole bundle's tasks go in as they are
    waiting: dict[str, list[int]] = {}
    into_row = [] if shard else tasks

    def set_aside(t: str) -> None:
        dest = row / f"{t}_0shot"
        if dest.exists():
            aside = earlier / f"{t}_0shot-before-import-{stamp}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(aside))
            lines.append(f"{t}: the answers here before are kept at {aside}")

    if shard:
        i, n = shard
        for t in tasks:
            sh = (reg.setdefault("shards", {}).get(t) or {})
            if sh.get("n") and sh["n"] != n:          # a whole set of another n: kept aside
                for j in sorted(sh.get("have") or {}, key=int):
                    old = shards_dir(row_name) / t / f"{j}-of-{sh['n']}"
                    if old.exists():
                        aside = earlier / f"{t}_0shot-shard-{j}-of-{sh['n']}-before-import-{stamp}"
                        aside.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(old), str(aside))
                sh = {}
            sh = {"n": n, "have": dict(sh.get("have") or {})}
            slot = shards_dir(row_name) / t / f"{i}-of-{n}"
            if slot.exists():
                aside = earlier / f"{t}_0shot-shard-{i}-of-{n}-before-import-{stamp}"
                aside.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(slot), str(aside))
                lines.append(f"{t}: shard {i} of {n} as it was here before is kept at {aside}")
            prefix = f"results/{row_name}/{t}_0shot/"
            for name, data in sorted(b["files"].items()):
                if name.startswith(prefix):
                    target = slot / name[len(prefix):]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            sh["have"][str(i)] = {"gpu": gpu, "sha256": b["sha256"], "bundle": path.name,
                                  "at": stamp, "answers": bundle["tasks"][t]}
            reg["shards"][t] = sh
            lines.append(f"{t}: shard {i} of {n}, {bundle['tasks'][t]} answers from a rented "
                         f"GPU ({gpu})")
            # 15.7: a shard here saved in another form than this one (cut, before
            # 15.7, under a new run's whole replies) is set aside: its new run
            # replaces it, and the task waits for it
            form = dm.reply_form(slot)
            for j in sorted((x for x in sh["have"] if x != str(i)), key=int):
                other = shards_dir(row_name) / t / f"{j}-of-{n}"
                if dm.reply_form(other) == form:
                    continue
                aside = earlier / f"{t}_0shot-shard-{j}-of-{n}-other-form-{stamp}"
                aside.parent.mkdir(parents=True, exist_ok=True)
                if other.exists():
                    shutil.move(str(other), str(aside))
                del sh["have"][j]
                lines.append(f"{t}: shard {j} of {n} here was saved in another form; it's kept "
                             f"at {aside}, and its new run replaces it")
            missing = [j for j in range(1, n + 1) if str(j) not in sh["have"]]
            if missing:
                waiting[t] = missing
                lines.append(f"{t}: {_shards(missing, n)} missing — {t} is scored once "
                             f"{'it is' if len(missing) == 1 else 'they are'} imported")
                continue
            set_aside(t)
            count = merge_shards(row, t, n, stamp)
            parts = [sh["have"][str(j)] for j in range(1, n + 1)]
            gpus = sorted({x["gpu"] for x in parts})
            reg.setdefault("tasks", {})[t] = {
                "gpu": _and(gpus), "gpus": gpus, "shards": n, "sha256": b["sha256"],
                "bundle": path.name, "at": stamp,
                "shard_bundles": [{"shard": j, **{k: x[k] for k in ("gpu", "sha256", "bundle")}}
                                  for j, x in enumerate(parts, 1)],
                "samples": [str(q.relative_to(row)) for q in
                            sorted((row / f"{t}_0shot").rglob(f"samples_{t}_*.jsonl"))]}
            lines.append(f"{t}: every shard is in ({n} of {n}) · {count} answers, merged in the "
                         f"battery's order")
            into_row.append(t)
    else:
        for t in tasks:
            set_aside(t)
    # the answers' files, and the model's record where the server has none
    for name, data in sorted(b["files"].items()) if into_row else []:
        if not name.startswith("results/"):
            continue
        rel = Path(name).relative_to("results")
        if rel.parts[0] not in (row_name, base.name):
            continue
        target = config.OUT_DIR / rel
        if rel.name == "model_meta.json" and len(rel.parts) == 2:
            if target.exists():
                continue                      # the server's own record of the model stands
        elif shard or not (rel.parts[0] == row_name and len(rel.parts) > 2
                           and rel.parts[1] in {f"{t}_0shot" for t in tasks}):
            continue                          # a shard's answers went in merged, above
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for t in [] if shard else tasks:
        n_ = bundle["tasks"][t]
        lines.append(f"{t}: {n_} answers from a rented GPU ({gpu})")
        reg.setdefault("tasks", {})[t] = {
            "gpu": gpu, "sha256": b["sha256"], "bundle": path.name, "at": stamp,
            "samples": [str(q.relative_to(row)) for q in
                        sorted((row / f"{t}_0shot").rglob(f"samples_{t}_*.jsonl"))]}
    _write_registry(row, reg)

    # the row, by the board's own scoring, from what is on disk now
    have = [t for t in rb.SUITES[bundle["suite"]]["tasks"]
            if (row / f"{t}_0shot").is_dir() and rb.task_answers(row / f"{t}_0shot", t)]
    missing = [t for t in rb.SUITES[bundle["suite"]]["tasks"] if t not in have]
    status, error = "done", ""
    if missing:
        by_gap: dict[tuple, list[str]] = {}
        for t, js in waiting.items():
            by_gap.setdefault(tuple(js), []).append(t)
        waits = [f"{_and(ts)} {'waits' if len(ts) == 1 else 'wait'} for "
                 f"{_shards(list(js), shard[1])}" for js, ts in by_gap.items()]
        rest = [t for t in missing if t not in waiting]
        line = "; ".join(
            ([f"shard {shard[0]} of {shard[1]} imported"] if shard else ["answers imported"])
            + waits + ([f"the row is scored once {', '.join(rest)} is here too"] if rest else []))
    else:
        try:
            line = sdm.mark_hf(0, {"hf_id": model}, {"revision": setup.get("revision")}, row,
                               {"on": thinking, "mode": setup.get("thinking_mode")})
        except Exception as e:                      # noqa: BLE001 — the answers are on disk
            status, error, line = "failed", f"scoring: {e}", f"answers imported; scoring: {e}"
    where = sdm.where_of(row).get("where") or f"run on a rented GPU ({gpu})"

    # the Runs list: an entry for the import, with the bundle's log
    of = f" · shard {shard[0]} of {shard[1]}" if shard else ""
    sid = db.add(model, "instruct", bundle["suite"], by,
                 f"imported from a rented GPU ({gpu}){of} · {path.name}", thinking=thinking,
                 part="full", tasks=tasks if len(tasks) < len(rb.SUITES[bundle["suite"]]["tasks"])
                 else None, status=status)
    # 17g: where it ran, as a Frontier import says it (it read "this server")
    import import_frontier
    db.update(sid, finished_at=time.time(), progress=line, error=error,
              where_ran=import_frontier.where_words(import_frontier.gpu_names(setup) or [gpu],
                                                    []))
    log = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"===== [{sid}] imported {path.name} (sha256 {b['sha256'][:16]}) by {by}: "
                   f"{', '.join(tasks)}{of} {where} =====\n" + b["log"]
                   + "".join(f"\n[import] {x}" for x in lines) + f"\n[import] {line}\n",
                   encoding="utf-8")
    reg.setdefault("imports", []).append({"sha256": b["sha256"], "sid": sid, "tasks": tasks,
                                          "gpu": gpu, "by": by, "at": stamp, "bundle": path.name,
                                          **({"shard": list(shard)} if shard else {})})
    _write_registry(row, reg)
    for x in lines:
        say(x)
    say(f"the row {model}{' · thinking' if row_name.endswith('__thinking') else ''}: {line}"
        + (f" · {where}" if into_row else ""))
    say(f"Runs #{sid}, its log the bundle's")
    return 0 if status == "done" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", type=Path)
    ap.add_argument("--by", default="", help="your name, for the Runs list")
    ap.add_argument("--battery", action="store_true",
                    help="this server's battery hashes, for remote_run.py --battery")
    ap.add_argument("--file-sha256", default="",
                    help="17: a Frontier bundle of a served model whose file isn't registered "
                         "here: the sha256 of the file its server serves (kept with --by)")
    ap.add_argument("--register", default="",
                    help="17: a Frontier bundle of a model this board doesn't serve (run on "
                         "rented GPUs only): its name here; the bundle's file is pinned")
    ap.add_argument("--public-weights", action="store_true",
                    help="17g: with --register, a model whose weights are public (G6's "
                         "calibration): its raw runs may be exported public. Nothing else ever "
                         "is")
    ap.add_argument("--served", action="store_true",
                    help="17h: the models this board serves, as a JSON list (for "
                         "frontier_fetch.py's by-hand lines)")
    ap.add_argument("--boxes", type=Path, default=None,
                    help="17f: what frontier_fetch.py read from the rented boxes, for Runs' "
                         "list (their labels and progress, never an address)")
    ap.add_argument("--set-aside-shards", action="store_true",
                    help="17c: a Frontier shard whose task's shards here were made with another "
                         "setup (a rebuilt tarball): those are set aside, and this one starts "
                         "the task's shards again")
    a = ap.parse_args(argv)
    from service import db
    db.init()
    if a.battery:
        from service import config
        h = server_battery()
        print(f"items {h['items']}\nbattery {h['battery']}\nprompts {h['prompts']}")
        if config.DM_ITEMS_SHA256:                  # 15.6: and whether they're the repo's
            print("the repo's items " + config.DM_ITEMS_SHA256 + (
                " (the same)" if h["items"] == config.DM_ITEMS_SHA256 else " (they differ)"))
        return 0
    if a.served:
        print(json.dumps(sorted(r["id"] for r in db.served_all())))
        return 0
    if a.boxes:
        import import_frontier
        try:
            n = import_frontier.store_boxes(json.loads(a.boxes.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            print(f"refused — {a.boxes.name} couldn't be read: {e}")
            return REFUSED
        print(f"{n} step{'s' if n != 1 else ''} on rented boxes, as the fetch read them")
        return 0
    if not a.bundle:
        ap.error("a bundle, or --battery")
    if not a.by.strip():
        ap.error("--by: your name, for the Runs list")
    sha = a.file_sha256.strip().lower()
    if sha and not re.fullmatch(r"[0-9a-f]{64}", sha):
        ap.error("--file-sha256: 64 hex digits")
    if a.public_weights and not a.register.strip():
        ap.error("--public-weights: with --register, when the model is registered")
    return import_bundle(a.bundle, a.by.strip()[:80], file_sha=sha, register=a.register.strip(),
                         aside=a.set_aside_shards, public_weights=a.public_weights)


if __name__ == "__main__":
    raise SystemExit(main())
