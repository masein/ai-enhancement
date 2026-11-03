#!/usr/bin/env python3
"""31.2: DeviceMark's battery on a rented box's own llama-server — a served
model asked as the board asks one (service/devicemark.py), its answers brought
home in a bundle and marked again by the import.

On the box, remote_gguf.py --devicemark runs it: the server started and
checked as for any step (a runtime of expert paths: its settings check on the
step's path first), the battery's 596 items asked through the board's own
runner, then the bundle

    devicemark-served-<served id>-thinking-<on|off>.tar.gz

holding the answers as they came (devicemark_answers.jsonl), the box's own
marking of them (for information), its setup (the file, the server, the
runtime and its check, the battery's hashes and what each request sent) and
its log. A step left short asks only what it hasn't answered when its line is
pasted again, and has no bundle until every item is answered.

On the server, import_remote.py takes it as it takes a Frontier bundle (with
--register and --file-sha256 for a model run on rented GPUs only). It is
refused unless its GGUF is the file registered for the model, its launch and
runtime the registered ones (and the path's settings check passed), its
battery the board's (ids, prompts and items, by sha256) and each request what
the board sends that model (service/devicemark.settings): on an expert path
greedy, temperature 0 and no presence penalty — "DeviceMark's protocol:
greedy, no penalties" — or, for a served id asked at the file's own settings
(frontier.FILE_SAMPLING), the path and the seed alone. The answers are then
marked here, by the board's own scoring, onto the model's DeviceMark row.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402

for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import remote_bundle as rb  # noqa: E402

SUITE = "devicemark-served"
TASK = "devicemark"                     # the bundle's one task, and the runner's suite
REFUSED = 2
# a slot: the longest prompt of the battery and DeviceMark's cap (4,096 tokens,
# thinking included), with room to spare
CONTEXT = 8192
REGISTRY_TASK = "devicemark"


def _dm():
    import devicemark
    return devicemark


def items_path() -> Path:
    """the battery's items on this machine: beside its BENCH_ROOT (on a box,
    the step's own), where devicemark.load_items keeps them"""
    from service import config
    if os.environ.get("DM_ITEMS"):
        return Path(os.environ["DM_ITEMS"])
    return Path(config.BENCH_ROOT) / "devicemark" / "items-v1.jsonl"


def prepare(say) -> dict:
    """the battery's items, read from the pinned datasets before the GPU is
    spent, and held to the repo's hash — the runner reads them from here"""
    from service import config
    d = _dm()
    config.DM_ITEMS = items_path()
    try:
        items = d.load_items(config.DM_ITEMS)
    except Exception as e:                              # noqa: BLE001 — said in one line
        raise SystemExit(rb.scrub(f"DeviceMark: its items couldn't be read from the pinned "
                                  f"datasets ({e}). Nothing was asked")) from None
    why = d.items_differ(items, config.DM_ITEMS_SHA256)
    if why:
        raise SystemExit(f"DeviceMark: {why}. Nothing was asked")
    say(f"DeviceMark: its {len(d.keys_for('full')):,} items ({d.VERSION}), items sha256 "
        f"{d.battery_hashes(items)['items'][:16]}… as the repo's, each asked once, a cap of "
        f"{d.CAP:,} tokens")
    return items


def row_of(served_as: str, thinking: bool) -> str:
    safe = served_as.replace("/", "__")
    return safe + "__thinking" if thinking else safe


def state(served_as: str, thinking: bool) -> tuple[int, int]:
    """(answered, of) for the battery on this box, under this run's settings"""
    from service import config, served
    from service import devicemark as sdm
    d = _dm()
    keys = d.keys_for("full")
    rec = served.get(served_as) if served.is_served(served_as) else None
    store = config.OUT_DIR / row_of(served_as, thinking) / sdm.ANSWERS_NAME
    if not rec or not store.exists():
        return 0, len(keys)
    s = sdm.settings(thinking, rec=rec)
    return len(sdm.kept(rec, keys, s, store)), len(keys)


def make_bundle(out: Path, served_as: str, thinking: bool, run_state: dict, gguf: dict,
                server: dict, rec: dict) -> tuple[Path | None, dict, dict]:
    """the bundle of the battery, once every item is answered: its path (None
    until then), {task: answers} in it and {task: {answers, of}} not finished"""
    import remote_gguf as rg
    import card_check as cc
    from service import config
    from service import devicemark as sdm
    d = _dm()
    row = config.OUT_DIR / row_of(served_as, thinking)
    n, of = state(served_as, thinking)
    if n < of:
        return None, {}, {TASK: {"answers": n, "of": of}}
    s = sdm.settings(thinking, rec=rec)
    keys = d.keys_for("full")
    got = sdm.kept(rec, keys, s, row / sdm.ANSWERS_NAME)
    answers = "".join(json.dumps(got[k], ensure_ascii=False) + "\n" for k in keys)
    files: dict[str, bytes] = {f"results/{row.name}/{sdm.ANSWERS_NAME}": answers.encode("utf-8")}
    for name in (d.OUT_NAME, d.ITEMS_NAME):
        if (row / name).exists():
            files[f"results/{row.name}/{name}"] = (row / name).read_bytes()
    base = config.OUT_DIR / row_of(served_as, False)
    for r in {row, base}:
        if (r / "model_meta.json").exists():
            files[f"results/{r.name}/model_meta.json"] = (r / "model_meta.json").read_bytes()
    safe = served_as.replace("/", "__")
    logs = sorted(config.LOGS_DIR.glob(f"service_*_{safe}.log"),
                  key=lambda p: int(p.name.split("_")[1]))
    log = "".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
    own = out / rg.OWN_LOG
    log += ("\n===== remote_gguf =====\n" + own.read_text(encoding="utf-8")) if own.exists() else ""
    log += "\n===== llama-server, as it started =====\n" + rg.startup_log(out / rg.SERVER_LOG)
    samples = cc.read_samples(out / rg.GPU_FILE)
    if samples:
        files[rg.GPU_FILE] = (out / rg.GPU_FILE).read_bytes()
    use = cc.summarise(samples) if samples else {}
    price = rg.price_of(run_state)
    spent = {}
    if use.get("seconds"):
        spent["server_s"] = round(use["seconds"], 1)
        if use.get("wh") is not None:
            spent["energy_wh"] = round(use["wh"], 3)
        if price is not None:
            spent["cost_usd"] = round(price * spent["server_s"] / 3600, 4)
    try:
        marked = json.loads((row / d.OUT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        marked = {}
    items = d.load_items(items_path())
    off = sum(1 for r in got.values() if r.get("error"))
    setup = {"suite": SUITE, "protocol": d.VERSION, "served_as": served_as,
             **({"runtime": rec["runtime"]} if rec.get("runtime") else {}),
             "based_on": rec.get("based_on") or "", "thinking": "on" if thinking else "off",
             "gguf": gguf, "server": server, "gpu": rb.gpu_info(),
             "where": rg.WHERE,
             "runner": "service/runner.py → service/devicemark.py, called directly",
             "devicemark": {"battery": d.battery_hashes(items), "version": d.VERSION,
                            "cap": d.CAP, "seed": d.SEED, "request": s,
                            "sampling": sdm.sampling_of(s) if "expert_path" in s else "greedy",
                            "protocol_words": (sdm.PROTOCOL_WORDS[sdm.sampling_of(s)]
                                               if "expert_path" in s else ""),
                            "answers": len(got), "written_off": off,
                            # the box's own marking, for information: the import marks again
                            "box_composite": (marked.get("composite") or {}).get("value"),
                            "row_setup": marked.get("setup") or {}},
             "tasks": {TASK: {"answers": len(got), **spent}},
             "board_commit": rg.board_commit(), "sessions": run_state.get("sessions", 1),
             "box": run_state.get("label") or None, "image": (rg.board_commit() or "")[:7] or None,
             "started_at": run_state.get("started_at"),
             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             **({"card_checks": run_state["card_checks"]} if run_state.get("card_checks") else {}),
             **({"gpu_use": use} if use else {}),
             **({"price": {"per_hour": price, "by": "the box line (--price)"}}
                if price is not None else {})}
    files["run.log"] = rb.scrub(log).encode("utf-8")
    files["setup.json"] = rb.scrub(json.dumps(setup, indent=1, sort_keys=True)).encode("utf-8")
    bundle = {"format": rb.SUITES[SUITE]["format"], "suite": SUITE, "model": served_as,
              "thinking": thinking, "row": row.name, "gguf_sha256": gguf["sha256"],
              "tasks": {TASK: len(got)},
              "answers_sha256": hashlib.sha256(answers.encode("utf-8")).hexdigest(),
              "files": {n_: hashlib.sha256(b).hexdigest() for n_, b in files.items()}}
    files["bundle.json"] = json.dumps(bundle, indent=1, sort_keys=True).encode("utf-8")
    path = rb.write(out / rb.bundle_name(SUITE, served_as, thinking), files)
    return path, {TASK: {"answers": len(got), "of": of}}, {}


# ---------------------------------------------------------------------------
# the import, on the server
# ---------------------------------------------------------------------------

# what the import reads of setup.json's devicemark record, and the type of each
_DM_TYPES = {"battery": {"battery": str, "prompts": str, "items": str}, "version": str,
             "cap": int, "seed": int, "request": dict, "sampling": str, "protocol_words": str,
             "answers": int, "row_setup": dict}


def setup_problems(setup: dict) -> list[str]:
    import import_frontier as imf
    out = imf.setup_problems(setup)
    if out:
        return out
    if not isinstance(setup.get("devicemark"), dict):
        return ["its setup.json has no devicemark record"]
    imf._types(setup["devicemark"], _DM_TYPES, "devicemark", out)
    return out


def _records(b: dict, row_name: str) -> tuple[list[dict], str]:
    """the answer lines, and what is wrong with them ('' when nothing)"""
    from service import devicemark as sdm
    raw = b["files"].get(f"results/{row_name}/{sdm.ANSWERS_NAME}")
    if raw is None:
        return [], f"it holds no {sdm.ANSWERS_NAME}"
    out = []
    for k, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            return [], f"line {k} of its answers isn't JSON"
        if not (isinstance(r, dict) and isinstance(r.get("bench"), str)
                and isinstance(r.get("key"), (str, int)) and isinstance(r.get("text"), str)
                and isinstance(r.get("answer", ""), str)):
            return [], f"line {k} of its answers isn't an answer (bench, key and text)"
        out.append({**r, "key": str(r["key"])})
    return out, ""


def checks(b: dict, rec: dict | None, file_sha: str, battery: dict) -> list[str]:
    """every check the bundle fails, in words; [] when it may be imported"""
    import import_frontier as imf
    from service import devicemark as sdm
    d = _dm()
    bundle, setup = b["bundle"], b["setup"]
    bad = setup_problems(setup)
    if bad:
        return bad
    want_fmt = rb.SUITES[SUITE]["format"]
    if bundle.get("format") != want_fmt:
        return [f"its format: the bundle's {bundle.get('format')}, this board reads {want_fmt} "
                "for a served model's DeviceMark run"]
    if bundle.get("shard"):
        return ["a DeviceMark run on a box is whole: no shard"]
    out, stop = imf.model_problems(b, rec, file_sha)
    if stop:
        return out
    on = bool(bundle.get("thinking"))
    dm_ = setup["devicemark"]
    if setup.get("protocol") != d.VERSION or dm_.get("version") != d.VERSION:
        out.append(f"the battery: the bundle's {setup.get('protocol')}, this board's {d.VERSION}")
    from import_remote import BATTERY_WORDS
    theirs = dm_.get("battery") or {}
    for k, words in BATTERY_WORDS.items():
        if theirs.get(k) != battery.get(k):
            out.append(f"its battery's {words} aren't the board's: the bundle's sha256 "
                       f"{str(theirs.get(k))[:16]}…, this board's {str(battery.get(k))[:16]}…")
    if (dm_.get("cap"), dm_.get("seed")) != (d.CAP, d.SEED):
        out.append(f"asked with a cap of {dm_.get('cap')} and seed {dm_.get('seed')}; "
                   f"DeviceMark's are {d.CAP} and {d.SEED}")
    want = sdm.settings(on, rec=rec)
    if dm_.get("request") != want:
        out.append(f"each request sent {json.dumps(dm_.get('request'), sort_keys=True)}; this "
                   f"board asks {bundle.get('model')} with {json.dumps(want, sort_keys=True)}")
    records, why = _records(b, bundle.get("row") or "")
    if why:
        out.append(why)
        return out
    keys = [(bk, str(k)) for bk, k in d.keys_for("full")]
    got = [(r["bench"], r["key"]) for r in records]
    if len(set(got)) != len(got):
        out.append(f"{len(got) - len(set(got)):,} items answered twice")
    if set(got) != set(keys):
        miss, extra = set(keys) - set(got), set(got) - set(keys)
        out.append(f"{len(set(keys) & set(got)):,} of the battery's {len(keys):,} items answered"
                   + (f", {len(extra):,} not in it" if extra else "")
                   + (f" (none of {', '.join(f'{x}:{y}' for x, y in sorted(miss)[:3])}…)"
                      if miss else ""))
    if "expert_path" in want and any(r.get("raw_fallback") for r in records):
        out.append("answers asked without the chat endpoint: an expert path is asked through "
                   "it alone")
    return out


def import_bundle(b: dict, path: Path, by: str, say=print, file_sha: str = "",
                  register: str = "", public_weights: bool = False) -> int:
    """a served model's DeviceMark run from a rented box, onto its row"""
    import import_frontier as imf
    import import_remote as ir
    from service import config, db, served
    from service import devicemark as sdm
    d = _dm()
    bad = imf.bundle_problems(b.get("bundle"))
    if not bad and isinstance(b.get("setup"), dict):
        bad = setup_problems(b["setup"])
    elif not bad:
        bad = ["its setup.json isn't a JSON object"]
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    bundle, setup = b["bundle"], b["setup"]
    model, on = bundle.get("model") or "", bool(bundle.get("thinking"))
    row_name = bundle.get("row") or ""
    gg = setup.get("gguf") or {}
    dm_ = setup["devicemark"]
    say(f"{path.name} · sha256 {b['sha256'][:16]} · {model} · thinking {'on' if on else 'off'} · "
        f"DeviceMark ({dm_.get('answers')} answers) · {gg.get('name') or 'its GGUF'} "
        f"({str(gg.get('sha256'))[:16]})")
    if not row_name or "/" in row_name or row_name.startswith("."):
        say(f"refused — its row {row_name!r} isn't a row's name")
        return REFUSED
    row = config.OUT_DIR / row_name
    reg = ir.registry(row)
    was = next((x for x in reg.get("imports") or [] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, as Runs #{was['sid']} on {was['at']}: nothing changed")
        return 0
    rec = served.get(model) if served.is_served(model) else None
    if register:
        why = ("" if not rec else f"{model} is registered here already: import without "
               "--register") or imf.register_refused(model, file_sha)
        if why:
            say(f"refused — {why}")
            return REFUSED
    try:
        items = d.load_items(config.DM_ITEMS)
        battery = d.battery_hashes(items)
    except Exception as e:                              # noqa: BLE001 — said in one line
        say(f"refused — this server can't read DeviceMark's items to check the answers ({e})")
        return REFUSED
    bad = checks(b, rec or (imf._would_be(b, file_sha) if register else None), file_sha, battery)
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    if register:
        rec = imf.registered_here(b, register, by, public_weights)
        say(f"{model} registered as {rec['name']}: a model run on rented GPUs only, its file "
            f"{(gg.get('name') or '')} pinned by the sha256 you gave, {file_sha[:16]}…")
    if file_sha and not imf.registered_sha(rec):
        rec["file_sha256"] = {"sha256": file_sha, "by": by, "at": time.time(),
                              "name": gg.get("name") or "",
                              **({"source": str(gg["source"])[:300]} if gg.get("source") else {}),
                              **({"parts": gg["parts"]} if gg.get("parts") else {})}
        db.served_put(rec)
        say(f"{model}'s file is now registered by its sha256 {file_sha[:16]}…, as {by} gave it")
    srv = setup.get("server") or {}
    gpu = (setup.get("gpu") or {}).get("name") or "a GPU"
    s = sdm.settings(on, rec=rec)
    say(f"as this board asks it: {gg.get('name')} (the registered file), llama.cpp "
        f"{srv.get('build') or '?'} ({srv.get('commit') or '?'}), {srv.get('slots')} slots, "
        f"on {gpu} · each request {json.dumps({k: v for k, v in s.items() if k != 'max_tokens'})}"
        + (f" — {sdm.PROTOCOL_WORDS[sdm.sampling_of(s)]}" if "expert_path" in s else ""))

    # the row's answers before this bundle, set aside where another run left them
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    records, _ = _records(b, row_name)
    lines = []
    earlier = [n for n in (sdm.ANSWERS_NAME, d.OUT_NAME, d.ITEMS_NAME) if (row / n).exists()]
    if earlier:
        keep = config.OUT_DIR.with_name("earlier") / f"{row_name}-{stamp}"
        keep.mkdir(parents=True, exist_ok=True)
        for n in earlier:
            (row / n).replace(keep / n)
        lines.append(f"the row's earlier DeviceMark files set aside in {keep.name}")
    row.mkdir(parents=True, exist_ok=True)
    (row / sdm.ANSWERS_NAME).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                for r in records), encoding="utf-8")
    imf.write_meta(row, rec, on)
    props = (dm_.get("row_setup") or {}).get("props")
    where = f"run on a rented GPU ({gpu})"
    row_setup = {**sdm.setup_of(rec, on, props if isinstance(props, dict) else {}, s),
                 "where": where, "bundle": path.name, "box": setup.get("box") or None}
    status, error = "done", ""
    try:
        marked = d.mark(row, items, row_setup, records, "full", reading={"on": on})
        line = sdm.summary_line(marked)
    except Exception as e:                              # noqa: BLE001 — the answers are on disk
        status, error, line = "failed", f"scoring: {e}", f"answers imported; scoring: {e}"
    off = sum(1 for r in records if r.get("error"))
    if off:
        lines.append(f"{off:,} items the server failed on, counted wrong as the protocol counts "
                     "an item with no answer")
    box_value = dm_.get("box_composite")
    if status == "done" and isinstance(box_value, (int, float)) and not isinstance(
            box_value, bool):
        here = (marked.get("composite") or {}).get("value")
        lines.append(f"marked here: {here} — the box marked {box_value}"
                     + ("" if here == box_value else " (this board's scoring stands)"))
    reg.setdefault("tasks", {})[REGISTRY_TASK] = {
        "gpu": gpu, "sha256": b["sha256"], "bundle": path.name, "at": stamp,
        "samples": [sdm.ANSWERS_NAME], "box": setup.get("box") or "",
        **imf.spent_of(setup, TASK, len(records))}
    sid = db.add(model, "instruct", TASK, by, f"imported from a rented GPU ({gpu})"
                 + (f" · box {setup['box']}" if setup.get("box") else "") + f" · {path.name}",
                 thinking=on, part="full", status=status)
    db.update(sid, finished_at=time.time(), progress=line, error=error,
              where_ran=imf.where_words(imf.gpu_names(setup) or [gpu], []))
    log = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"===== [{sid}] imported {path.name} (sha256 {b['sha256'][:16]}): DeviceMark "
                   f"{where} =====\n" + b["log"]
                   + "".join(f"\n[import] {x}" for x in lines) + f"\n[import] {line}\n",
                   encoding="utf-8")
    reg.setdefault("imports", []).append({"sha256": b["sha256"], "sid": sid, "tasks": [TASK],
                                          "gpu": gpu, "by": by, "at": stamp,
                                          "bundle": path.name})
    ir._write_registry(row, reg)
    for x in lines:
        say(x)
    say(f"the row {model}{' · thinking' if on else ''}: {line} · {where}")
    say(f"Runs #{sid}, its log the bundle's")
    return 0 if status == "done" else 1
