"""17: a GGUF's Frontier bundle from a rented GPU (scripts/remote_gguf.py),
into the board — scripts/import_remote.py hands it here by its suite.

The checks. It refuses the bundle, saying which check failed, when:
- the served model it names (--as on the box) isn't registered here, or is a
  model from OpenRouter;
- its GGUF isn't the file registered for that model: the bundle's sha256
  against the one the board hashed when the file was registered on the
  model's page — or, for a model whose file isn't registered here, the
  sha256 the person importing gives (--file-sha256, kept on the model with
  their name);
- a task's protocol, its dataset's revision or its runs aren't this board's;
- the questions weren't asked as the board asks this model: each task's
  budget, the model card's sampling and the thinking switch, as
  service/frontier.py would set them for it here;
- its answers don't cover its questions (its shard's, for a shard), or
  answer questions outside them.

The merge. A whole bundle's task replaces the task's answers on the row (the
model's, or its "· thinking" row) — the earlier ones kept under
results/earlier/. A shard waits under results/shards/<row>/<task>/<i>-of-<n>/
until every shard of the task is in; then they are merged, question by
question. Then the task is scored by code, as on the board (code-scored
benchmarks are scored at import), and its results say it ran on a rented GPU.

A bundle already imported — by its sha256 — changes nothing.
"""

from __future__ import annotations

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
import frontier as fb  # noqa: E402
import remote_bundle as rb  # noqa: E402

REFUSED = 2
SUITE = "frontier"
REGISTRY = "frontier_imports.json"


def registered_sha(rec: dict) -> str:
    """the sha256 of the file registered for a served model: the board's own
    hash of it, else the one an import was given"""
    return ((rec.get("gguf_pin") or {}).get("sha256")
            or (rec.get("file_sha256") or {}).get("sha256") or "")


def _answers(b: dict, row: str, task: str) -> dict[tuple[str, int], dict]:
    from service import frontier as sf
    name = f"results/{row}/{task}_0shot/{sf.SUB}/{sf.ANSWERS}"
    out: dict[tuple[str, int], dict] = {}
    for line in b["files"].get(name, b"").decode("utf-8").splitlines():
        try:
            r = json.loads(line)
            out[(str(r["id"]), int(r["epoch"]))] = r
        except (ValueError, KeyError, TypeError):
            continue
    return out


def checks(b: dict, rec: dict | None, file_sha: str = "") -> list[str]:
    """every check the bundle fails, in words; [] when it may be imported"""
    from service import config, served
    from service import frontier as sf
    bundle, setup = b["bundle"], b["setup"]
    want_fmt = rb.SUITES[SUITE]["format"]
    if bundle.get("format") != want_fmt:
        return [f"its format: the bundle's {bundle.get('format')}, this board reads {want_fmt} "
                "for a GGUF's Frontier run"]
    try:
        shard = rb.shard_of_bundle(bundle)
    except ValueError as e:
        return [str(e)]
    model = bundle.get("model") or ""
    if not rec:
        return [f"{model or 'its model'} isn't registered on this board: add it under Add a "
                "model ▸ Running on a server, and register its file on its page"]
    if served.is_openrouter(rec):
        return [f"{model} is a model from OpenRouter: a GGUF's answers can't go on its row"]
    out = []
    on = bool(bundle.get("thinking"))
    safe = model.replace("/", "__")
    if bundle.get("row") != (safe + "__thinking" if on else safe):
        out.append(f"its row {bundle.get('row')!r} isn't {model}'s, thinking "
                   f"{'on' if on else 'off'}")
    have = registered_sha(rec)
    theirs = (setup.get("gguf") or {}).get("sha256") or ""
    if file_sha and have and file_sha != have:
        out.append(f"--file-sha256 {file_sha[:16]}… isn't the file registered for {model} "
                   f"({have[:16]}…)")
    have = have or file_sha
    if not have:
        out.append(f"{model} has no file registered with its sha256: register its file on its "
                   "page (the board hashes it), or give --file-sha256 with the sha256 of the "
                   "file its server serves")
    elif theirs != have or bundle.get("gguf_sha256") != theirs:
        out.append(f"its GGUF isn't the file registered for {model}: the bundle's "
                   f"{(setup.get('gguf') or {}).get('name') or 'file'} has sha256 "
                   f"{theirs[:16]}…, the registered file {have[:16]}…")
    if setup.get("protocol") != fb.VERSION:
        out.append(f"the protocol: the bundle's {setup.get('protocol')}, this board's "
                   f"{fb.VERSION}")
    tasks = bundle.get("tasks") or {}
    if not tasks:
        out.append("it holds no task answered whole")
    for t in tasks:
        if t not in fb.BENCH:
            out.append(f"{t}: not a Frontier benchmark on this board")
            continue
        spec, got = fb.BENCH[t], (setup.get("tasks") or {}).get(t) or {}
        if (got.get("protocol_version"), got.get("revision"), got.get("epochs")) != (
                spec["protocol_version"], spec["source"]["revision"], spec["epochs"]):
            out.append(f"{t}: asked as {got.get('protocol_version')} on revision "
                       f"{str(got.get('revision'))[:12]}, {got.get('epochs')} runs; this board "
                       f"asks {spec['protocol_version']} on {spec['source']['revision'][:12]}, "
                       f"{spec['epochs']} runs")
            continue
        s = sf.settings(rec, t, on)
        mine = {k: v for k, v in s.items() if k not in ("max_tokens", "chat_template_kwargs")}
        if got.get("budget") != s["max_tokens"]:
            out.append(f"{t}: a budget of {got.get('budget')} tokens; this board gives "
                       f"{s['max_tokens']}")
        if (got.get("sampling") or {}) != mine:
            out.append(f"{t}: asked with {got.get('family') or 'the server’s defaults'}' "
                       f"sampling {got.get('sampling')}; this board asks {model} with "
                       f"{sf.family_of(rec) or 'the server’s defaults'}' {mine} — set "
                       f"--based-on on the box to what {model} is based on here "
                       f"({rec.get('based_on') or 'nothing yet'})")
        if got.get("switch") != s.get("chat_template_kwargs"):
            out.append(f"{t}: its thinking switch {got.get('switch')}; this board says "
                       f"{s.get('chat_template_kwargs')}")
        try:
            items = fb.shard_of(fb.load(t, config.BENCH_ROOT), shard)
        except Exception as e:                          # noqa: BLE001 — said in one line
            out.append(f"{t}: this server can't read its questions to check the answers ({e})")
            continue
        want = {(it["id"], e) for it in items for e in range(spec["epochs"])}
        ans = set(_answers(b, bundle.get("row") or "", t))
        of = f" (shard {shard[0]} of {shard[1]})" if shard else ""
        if not want <= ans:
            out.append(f"{t}: {len(want & ans)} of {len(want)} answers{of}")
        elif ans - want:
            out.append(f"{t}: {len(ans - want)} answers to questions outside its "
                       f"{'shard' if shard else 'task'}{of}")
    return out


def registry(row: Path) -> dict:
    try:
        return json.loads((row / REGISTRY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"imports": [], "tasks": {}, "shards": {}}


def _write_registry(row: Path, reg: dict) -> None:
    row.mkdir(parents=True, exist_ok=True)
    tmp = row / (REGISTRY + ".part")
    tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    tmp.replace(row / REGISTRY)


def where_of(row: Path, task: str) -> str:
    """"run on a rented GPU (NVIDIA GeForce RTX 5090)" for a task whose answers
    a bundle brought — '' for one answered on the board"""
    x = (registry(row).get("tasks") or {}).get(task)
    if not x:
        return ""
    gpus = x.get("gpus") or [x.get("gpu") or "a GPU"]
    return (f"run on a rented GPU ({gpus[0]})" if len(gpus) == 1
            else f"run on rented GPUs ({', '.join(gpus)})")


def import_bundle(b: dict, path: Path, by: str, say=print, file_sha: str = "") -> int:
    from service import config, db, served
    from service import frontier as sf
    bundle, setup = b["bundle"], b["setup"]
    model, on = bundle.get("model") or "", bool(bundle.get("thinking"))
    row_name = bundle.get("row") or ""
    try:
        shard = rb.shard_of_bundle(bundle)
    except ValueError:
        shard = None
    gg = setup.get("gguf") or {}
    say(f"{path.name} · sha256 {b['sha256'][:16]} · {model} · thinking "
        f"{'on' if on else 'off'} · {', '.join(bundle.get('tasks') or {}) or 'no task'}"
        + (f" · shard {shard[0]} of {shard[1]}" if shard else "")
        + f" · {gg.get('name') or 'its GGUF'} ({str(gg.get('sha256'))[:16]})")
    row = config.OUT_DIR / row_name if row_name and "/" not in row_name else None
    reg = registry(row) if row else {"imports": []}
    was = next((x for x in reg.get("imports") or [] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, as Runs #{was['sid']} on {was['at']}: nothing changed")
        return 0
    rec = served.get(model) if served.is_served(model) else None
    bad = checks(b, rec, file_sha)
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    if file_sha and not registered_sha(rec):
        rec["file_sha256"] = {"sha256": file_sha, "by": by, "at": time.time(),
                              "name": gg.get("name") or ""}
        db.served_put(rec)
        say(f"{model}'s file is now registered by its sha256 {file_sha[:16]}…, as {by} gave it")
    srv = setup.get("server") or {}
    gpu = (setup.get("gpu") or {}).get("name") or "a GPU"
    say(f"as this board asks it: {gg.get('name')} (the registered file), llama.cpp "
        f"{srv.get('build') or '?'} ({srv.get('commit') or '?'}), {srv.get('slots')} slots, "
        f"on {gpu}")

    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    earlier = config.OUT_DIR.with_name("earlier") / row_name
    shards = config.OUT_DIR.with_name("shards") / row_name
    tasks = [t for t in fb.TASKS if t in (bundle.get("tasks") or {})]
    lines, scored = [], []

    def set_aside(t: str) -> None:
        dest = sf.task_dir(row, t)
        if dest.exists():
            aside = earlier / f"{t}_0shot-frontier-before-import-{stamp}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(aside))
            lines.append(f"{t}: the answers here before are kept at {aside}")

    def put(t: str, answers: dict, task_setup: bytes) -> None:
        d = sf.task_dir(row, t)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(
            json.dumps(answers[k], ensure_ascii=False) + "\n" for k in sorted(answers)),
            encoding="utf-8")
        (d / sf.SETUP).write_bytes(task_setup)

    for t in tasks:
        prefix = f"results/{row_name}/{t}_0shot/{sf.SUB}/"
        ans = _answers(b, row_name, t)
        tsetup = b["files"].get(prefix + sf.SETUP, b"{}")
        entry = {"gpu": gpu, "sha256": b["sha256"], "bundle": path.name, "at": stamp,
                 "by": by, "gguf_sha256": gg.get("sha256"), "llama_cpp": srv.get("build"),
                 "answers": len(ans)}
        if shard:
            i, n = shard
            sh = (reg.setdefault("shards", {}).get(t) or {})
            if sh.get("n") and sh["n"] != n:
                old = shards / t
                if old.exists():
                    aside = earlier / f"{t}-shards-of-{sh['n']}-before-import-{stamp}"
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old), str(aside))
                    lines.append(f"{t}: shards of {sh['n']} waiting here are kept at {aside}")
                sh = {}
            sh = {"n": n, "have": dict(sh.get("have") or {})}
            slot = shards / t / f"{i}-of-{n}"
            slot.mkdir(parents=True, exist_ok=True)
            (slot / sf.ANSWERS).write_bytes(b["files"][prefix + sf.ANSWERS])
            (slot / sf.SETUP).write_bytes(tsetup)
            sh["have"][str(i)] = entry
            reg["shards"][t] = sh
            lines.append(f"{t}: shard {i} of {n}, {len(ans):,} answers from a rented GPU ({gpu})")
            missing = [j for j in range(1, n + 1) if str(j) not in sh["have"]]
            if missing:
                lines.append(f"{t}: shard{'s' if len(missing) > 1 else ''} "
                             f"{', '.join(map(str, missing))} of {n} missing — {t} is scored once "
                             f"{'it is' if len(missing) == 1 else 'they are'} imported")
                continue
            merged: dict = {}
            for j in range(1, n + 1):
                merged.update(sf.read_answers(shards / t / f"{j}-of-{n}" / sf.ANSWERS))
            set_aside(t)
            put(t, merged, tsetup)
            parts = [sh["have"][str(j)] for j in range(1, n + 1)]
            reg.setdefault("tasks", {})[t] = {
                **entry, "gpus": sorted({x["gpu"] for x in parts}), "shards": n,
                "shard_bundles": [{"shard": j, **{k: x[k] for k in ("gpu", "sha256", "bundle")}}
                                  for j, x in enumerate(parts, 1)], "answers": len(merged)}
            lines.append(f"{t}: every shard is in ({n} of {n}) · {len(merged):,} answers")
        else:
            set_aside(t)
            put(t, ans, tsetup)
            reg.setdefault("tasks", {})[t] = {**entry, "gpus": [gpu]}
            lines.append(f"{t}: {len(ans):,} answers from a rented GPU ({gpu})")
        scored.append(t)
    # the model's record where this board has none (a thinking row's)
    for name, data in b["files"].items():
        rel = Path(name)
        if rel.parts[0] == "results" and len(rel.parts) == 3 and rel.name == "model_meta.json" \
                and rel.parts[1] == row_name and not (row / "model_meta.json").exists():
            row.mkdir(parents=True, exist_ok=True)
            (row / "model_meta.json").write_bytes(data)
    reg.setdefault("imports", [])
    _write_registry(row, reg)

    # each task whose answers are all in, scored by code, as the board scores it
    status, error, words = "done", "", []
    for t in scored:
        try:
            sc = sf.score_task(row, t, rec)
        except Exception as e:                          # noqa: BLE001 — the answers are on disk
            status, error = "failed", f"scoring {t}: {e}"
            words.append(f"{fb.BENCH[t]['label']}: answers imported; scoring failed: {e}")
            continue
        if sc:
            _mark_where(row, t, f"{WHERE_WORDS} ({', '.join(reg['tasks'][t]['gpus'])})")
            words.append(sf.words(t, sc))
    line = " · ".join(words) or "; ".join(x for x in lines if "missing" in x) or "imported"
    of = f" · shard {shard[0]} of {shard[1]}" if shard else ""
    sid = db.add(model, "instruct", SUITE, by,
                 f"imported from a rented GPU ({gpu}){of} · {path.name}", thinking=on,
                 tasks=tasks if len(tasks) < len(fb.TASKS) else None, status=status)
    db.update(sid, finished_at=time.time(), progress=line, error=error)
    log = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"===== [{sid}] imported {path.name} (sha256 {b['sha256'][:16]}) by {by}: "
                   f"{', '.join(tasks)}{of}, run on a rented GPU ({gpu}) =====\n" + b["log"]
                   + "".join(f"\n[import] {x}" for x in lines) + f"\n[import] {line}\n",
                   encoding="utf-8")
    reg["imports"].append({"sha256": b["sha256"], "sid": sid, "tasks": tasks, "gpu": gpu,
                           "by": by, "at": stamp, "bundle": path.name,
                           **({"shard": list(shard)} if shard else {})})
    _write_registry(row, reg)
    for x in lines:
        say(x)
    say(f"the row {model}{' · thinking' if on else ''}: {line}")
    say(f"Runs #{sid}, its log the bundle's")
    return 0 if status == "done" else 1


WHERE_WORDS = "run on a rented GPU"


def _mark_where(row: Path, task: str, where: str) -> None:
    """the task's results say where it ran (the score's own detail)"""
    from service import frontier as sf
    for f in sf.task_dir(row, task).glob("results_*.json"):
        blob = json.loads(f.read_text(encoding="utf-8"))
        blob.setdefault("frontier", {})["where"] = where
        f.write_text(json.dumps(blob, indent=1), encoding="utf-8")
