"""17: a GGUF's Frontier bundle from a rented GPU (scripts/remote_gguf.py),
into the board — scripts/import_remote.py hands it here by its suite. A
bundle is untrusted (17b): every check is made before anything on the board
changes, and a task's answers are scored in a folder of their own before they
take the row's place.

The checks. It refuses the bundle, saying which check failed, when:
- the served model it names (--as on the box) isn't registered here, or is a
  model from OpenRouter;
- its GGUF isn't the file registered for that model: the bundle's sha256
  against the one the board hashed when the file was registered on the
  model's page — or, for a model whose file isn't registered here, the
  sha256 the person importing gives (--file-sha256, kept on the model with
  their name);
- 17b: its launch isn't the one registered for that model. Compared: the
  routing environment (every LLAMA_MOE_* variable: lookahead or not) and
  speculative decoding (--spec-type, --draft-*, a draft model; and a server
  registered as drafting tokens). Free to differ, as they change where the
  model runs and not what it computes: memory and offload (-ngl, --cpu-moe,
  --n-cpu-moe, -ot), context (-c), slots (-np), the KV cache's type (-ctk,
  -ctv), flash attention, threads, batch sizes, GGML_CPU_* — the parity check
  (scripts/frontier_parity.py) is what says the box answers as the server;
- a task's protocol, its dataset's revision or its runs aren't this board's;
- the questions weren't asked as the board asks this model: each task's
  budget, the model card's sampling and the thinking switch, as
  service/frontier.py would set them for it here;
- 17b: an answer line isn't one (each field's type and range), or is a line
  the server failed on; a thinking bundle none of whose answers thought, or a
  thinking-off one whose answers did;
- its answers don't cover its questions (its shard's, for a shard), or
  answer questions outside them;
- 17b: a shard made with another build, other flags or another environment
  than the shards of its task already here.

The merge. A whole bundle's task replaces the task's answers on the row (the
model's, or its "· thinking" row) — the earlier ones kept under
results/earlier/. A shard waits under results/shards/<row>/<task>/<i>-of-<n>/
until every shard of the task is in; then they are merged, question by
question. Each task is scored by code in results/staging/ first, as on the
board (code-scored benchmarks are scored at import); only a task scored there
takes the row's place, and its results say it ran on a rented GPU. The row's
model record is the board's own (never the bundle's).

A bundle already imported — by its sha256 — changes nothing.

A model the board doesn't serve — the calibration's Gemma 4 26B A4B, run on
rented GPUs only — is registered by its first import (--register "<name>",
with --file-sha256: the sha256 masein checked, never the bundle's word for
it): a served entry with no server here, its file pinned, its build and
launch as the box recorded them. Its id may not hold "__" and no row of it
may exist yet.
"""

from __future__ import annotations

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
import frontier as fb  # noqa: E402
import remote_bundle as rb  # noqa: E402

REFUSED = 2
SUITE = "frontier"
REGISTRY = "frontier_imports.json"
_ID = re.compile(r"^served/[A-Za-z0-9._-]+$")
_ENV_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.:/+-]+)")
ROUTING_ENV = "LLAMA_MOE_"
_SPEC = re.compile(r"^(--spec-[a-z-]+|--draft[a-z-]*|-md|--model-draft|-ngld|--gpu-layers-draft"
                   r"|-cd|--ctx-size-draft|-devd|--device-draft)$")


def registered_sha(rec: dict) -> str:
    """the sha256 of the file registered for a served model: the board's own
    hash of it, else the one an import was given"""
    return ((rec.get("gguf_pin") or {}).get("sha256")
            or (rec.get("file_sha256") or {}).get("sha256") or "")


# ---------------------------------------------------------------------------
# the launch: what a box ran against what the board registered
# ---------------------------------------------------------------------------

def _spec_flags(tokens: list[str]) -> list[str]:
    """speculative decoding's flags among `tokens`, each with its value"""
    out = []
    for i, t in enumerate(tokens):
        name, _, val = t.partition("=")
        if not _SPEC.match(name):
            continue
        if not val and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            val = tokens[i + 1]
        out.append(f"{name}={val}" if val else name)
    return sorted(out)


def record_launch(rec: dict) -> dict:
    """the registered setup's routing environment and speculative flags — its
    launch flags and environment as typed, "How it's served", and the GGUF
    setup it serves the same file as (served.launch's sources)"""
    from service import db
    texts = [rec.get("flags") or "", rec.get("env") or "", rec.get("how") or ""]
    sa = rec.get("same_as") or {}
    if sa.get("gguf"):
        g = db.gguf_get(sa["gguf"]) or {}
        su = next((x for x in g.get("setups") or [] if x.get("id") == sa.get("setup")), None)
        if su:
            texts += [" ".join(f"{k}={v}" for k, v in (su.get("env") or {}).items()),
                      " ".join(su.get("flags") or [])]
    text = " ".join(t for t in texts if t)
    env = {k: v for k, v in _ENV_TOKEN.findall(text) if k.startswith(ROUTING_ENV)}
    toks = [t for t in re.split(r"[\s,;()]+", text) if t]
    return {"env": env, "spec": _spec_flags(toks), "drafts": rec.get("speculative") is True}


def box_launch(server: dict) -> dict:
    env = {k: str(v) for k, v in (server.get("env") or {}).items() if k.startswith(ROUTING_ENV)}
    return {"env": env, "spec": _spec_flags([str(x) for x in server.get("flags") or []])}


def launch_differs(rec: dict, server: dict) -> list[str]:
    """each way the box's launch differs from the registered one, in words"""
    want, got = record_launch(rec), box_launch(server)
    out = []
    if want["env"] != got["env"]:
        w = " ".join(f"{k}={v}" for k, v in sorted(want["env"].items())) or "none"
        g = " ".join(f"{k}={v}" for k, v in sorted(got["env"].items())) or "none"
        out.append(f"routing: the box ran with {g}; {rec['id']} is registered with {w}")
    if want["spec"] != got["spec"] or (want["drafts"] and not got["spec"]):
        w = " ".join(want["spec"]) or ("drafting tokens (its server says so)" if want["drafts"]
                                       else "none")
        g = " ".join(got["spec"]) or "none"
        out.append(f"speculative decoding: the box ran with {g}; {rec['id']} is registered "
                   f"with {w}")
    return out


def shard_setup(server: dict) -> dict:
    """17b: what every shard of a task must share"""
    return {"binary_sha256": server.get("binary_sha256"), "commit": server.get("commit"),
            "build": server.get("build"), "flags": list(server.get("flags") or []),
            "env": dict(server.get("env") or {})}


# ---------------------------------------------------------------------------
# the answers: read, and each line checked
# ---------------------------------------------------------------------------

def _lines(b: dict, row: str, task: str) -> list:
    from service import frontier as sf
    name = f"results/{row}/{task}_0shot/{sf.SUB}/{sf.ANSWERS}"
    out = []
    for line in b["files"].get(name, b"").decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            out.append(None)
    return out


def bad_line(r, epochs: int) -> str:
    """'' when a bundle's answer line is one; else what is wrong with it"""
    if not isinstance(r, dict):
        return "isn't a JSON object"
    if r.get("error"):
        return "is a question the server failed on, not an answer"
    if not isinstance(r.get("id"), str) or not r["id"]:
        return "has no question id"
    if not isinstance(r.get("epoch"), int) or isinstance(r.get("epoch"), bool) \
            or not 0 <= r["epoch"] < epochs:
        return f"has a run {r.get('epoch')!r}, not one of 0 to {epochs - 1}"
    if not isinstance(r.get("answer"), str):
        return "has an answer that isn't text"
    if r.get("finish") is not None and not isinstance(r["finish"], str):
        return "has a finish reason that isn't text"
    for k in ("tokens", "seed"):
        if r.get(k) is not None and (not isinstance(r[k], int) or isinstance(r[k], bool)
                                     or r[k] < 0):
            return f"has a {k} that isn't a count"
    if r.get("fallback") is not None and not isinstance(r["fallback"], dict):
        return "has a fallback that isn't a record"
    return ""


def _answers(b: dict, row: str, task: str) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for r in _lines(b, row, task):
        if isinstance(r, dict) and not bad_line(r, 10 ** 6):
            out[(r["id"], r["epoch"])] = r
    return out


def registered_here(b: dict, name: str, by: str) -> dict:
    """a served entry for a model run only on rented GPUs, from its bundle"""
    from service import db, served
    setup = b["setup"]
    gg, srv = setup.get("gguf") or {}, setup.get("server") or {}
    rec = {"id": b["bundle"]["model"], "name": name.strip()[:80], "base_url": "", "key": "",
           "based_on": setup.get("based_on") or "", "thinking": "auto", "phone": False,
           "how": (f"{gg.get('name')} on rented GPUs, llama.cpp {srv.get('build') or '?'} "
                   f"({srv.get('commit') or '?'}): " + " ".join(srv.get("argv") or [])),
           "gguf_path": "", "gguf_flags": "", "gguf_setups": [],
           "pin": {"model": gg.get("name"), "file": gg.get("name"), "size": gg.get("size"),
                   "ctx": srv.get("n_ctx"), "build": srv.get("build")},
           "file_sha256": {"sha256": gg.get("sha256"), "by": by, "at": time.time(),
                           "name": gg.get("name") or "",
                           **({"parts": gg["parts"]} if gg.get("parts") else {})},
           "rented_only": True, "answered": [], "by": by, "at": time.time(),
           "flags": " ".join(srv.get("flags") or []),
           "env": " ".join(f"{k}={v}" for k, v in (srv.get("env") or {}).items())}
    db.served_put(rec)
    served.write_meta(rec)
    return rec


def register_refused(model: str, file_sha: str) -> str:
    """17b: '' when a bundle's model may be registered by its import; else why
    not — the id from the bundle is never trusted to name a row"""
    from service import config
    if not file_sha:
        return ("--register needs --file-sha256: the sha256 of the file, as you checked it — "
                "never the bundle's own word for it")
    if not _ID.match(model) or "__" in model:
        return f"{model!r} can't be a new model's id: served/<name>, with no \"__\" in it"
    safe = model.replace("/", "__")
    have = [r for r in (safe, safe + "__thinking") if (config.OUT_DIR / r).exists()]
    if have:
        return f"{model} has a row here already ({', '.join(have)}): import without --register"
    return ""


# 17c: what the import reads of setup.json, and the type each must be (a list
# as gpu.name swapped the row in, then failed with a traceback and no Runs entry)
_SETUP_TYPES = {
    "protocol": str, "based_on": str, "where": str,
    "gguf": {"name": str, "sha256": str, "size": int, "source": str},
    "server": {"build": (int, str), "commit": str, "version": str, "slots": int,
               "flags": [str], "env": dict, "argv": [str], "binary_sha256": str,
               "tarball_sha256": str},
    "gpu": {"name": str},
    "tasks": {"*": {"protocol_version": str, "revision": str, "epochs": int, "budget": int,
                    "family": str}},
}


def _types(v, want, path: str, out: list[str]) -> None:
    if v is None:
        return
    if isinstance(want, dict):
        if not isinstance(v, dict):
            out.append(f"setup.json's {path} isn't a record")
            return
        for k, w in want.items():
            if k == "*":
                for kk, vv in v.items():
                    _types(vv, w, f"{path}.{kk}", out)
            else:
                _types(v.get(k), w, f"{path}.{k}", out)
        return
    if isinstance(want, list):
        if not isinstance(v, list) or not all(isinstance(x, want[0]) for x in v):
            out.append(f"setup.json's {path} isn't a list of {want[0].__name__}")
        return
    ok = isinstance(v, want) and not (isinstance(v, bool) and bool not in (
        want if isinstance(want, tuple) else (want,)))
    if not ok:
        out.append(f"setup.json's {path} is {type(v).__name__}, not "
                   + (" or ".join(t.__name__ for t in want) if isinstance(want, tuple)
                      else want.__name__))


def setup_problems(setup) -> list[str]:
    """17c: each field the import reads of setup.json, the type it must be"""
    if not isinstance(setup, dict):
        return ["its setup.json isn't a JSON object"]
    out: list[str] = []
    for k, w in _SETUP_TYPES.items():
        _types(setup.get(k), w, k, out)
    env = (setup.get("server") or {}).get("env") if isinstance(setup.get("server"), dict) else None
    if isinstance(env, dict) and not all(isinstance(x, str) for x in env.values()):
        out.append("setup.json's server.env holds a value that isn't text")
    return out


def checks(b: dict, rec: dict | None, file_sha: str = "") -> list[str]:
    """every check the bundle fails, in words; [] when it may be imported"""
    from service import config, served
    from service import frontier as sf
    bundle, setup = b["bundle"], b["setup"]
    if not isinstance(bundle, dict):
        return ["its bundle.json isn't a JSON object"]
    bad = setup_problems(setup)
    if bad:
        return bad
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
    # 17b: the launch registered for this model, not another setup of its file
    for d in launch_differs(rec, setup.get("server") or {}):
        out.append(f"its launch isn't the one registered for {model} — {d}. Run the box with "
                   f"that setup's own variables and flags (docs/REMOTE-RUNS.md G2)")
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
        # 17b: each line an answer, and none a question the server failed on
        lines = _lines(b, bundle.get("row") or "", t)
        bad = [(k, bad_line(r, spec["epochs"])) for k, r in enumerate(lines, 1)]
        bad = [(k, w) for k, w in bad if w]
        if bad:
            k, w = bad[0]
            out.append(f"{t}: {len(bad):,} of its {len(lines):,} answer lines aren't answers — "
                       f"line {k} {w}")
            continue
        why = sf.thinking_refused(t, "on" if on else "off", [r["answer"] for r in lines])
        if why:
            out.append(why)
        try:
            items = fb.shard_of(fb.load(t, config.BENCH_ROOT), shard)
        except Exception as e:                          # noqa: BLE001 — said in one line
            out.append(f"{t}: this server can't read its questions to check the answers ({e})")
            continue
        want = {(it["id"], e) for it in items for e in range(spec["epochs"])}
        ans = {(r["id"], r["epoch"]) for r in lines}
        of = f" (shard {shard[0]} of {shard[1]})" if shard else ""
        if len(ans) != len(lines):
            out.append(f"{t}: {len(lines) - len(ans):,} answers given twice{of}")
        if not want <= ans:
            out.append(f"{t}: {len(want & ans)} of {len(want)} answers{of}")
        elif ans - want:
            out.append(f"{t}: {len(ans - want)} answers to questions outside its "
                       f"{'shard' if shard else 'task'}{of}")
    return out


def shard_conflicts(reg: dict, b: dict, tasks: list[str], shard: tuple[int, int]) -> list[str]:
    """17b: a task whose shards here were made with another build, other flags
    or another environment than this one — never merged into one row"""
    mine = shard_setup(b["setup"].get("server") or {})
    out = []
    for t in tasks:
        for j, x in sorted(((reg.get("shards") or {}).get(t) or {}).get("have", {}).items()):
            if j == str(shard[0]) or not x.get("setup") or x["setup"] == mine:
                continue
            diff = [k for k in mine if mine[k] != x["setup"].get(k)]
            out.append(f"{t}: shard {j} here was made with another setup ({', '.join(diff)}: "
                       f"{', '.join(repr(x['setup'].get(k)) for k in diff)} there, "
                       f"{', '.join(repr(mine[k]) for k in diff)} here) — every shard of a task "
                       "is run the same way: import with --set-aside-shards to set the shards "
                       "here aside and start its shards again with this one")
    return out


def _would_be(b: dict, file_sha: str) -> dict:
    """what --register would keep, for the checks before it is kept — its file
    the sha256 masein gave"""
    setup = b["setup"]
    gg = setup.get("gguf") or {}
    srv = setup.get("server") or {}
    return {"id": b["bundle"].get("model") or "", "name": "", "based_on":
            setup.get("based_on") or "", "pin": {"file": gg.get("name"), "model": gg.get("name")},
            "file_sha256": {"sha256": file_sha},
            "flags": " ".join(srv.get("flags") or []),
            "env": " ".join(f"{k}={v}" for k, v in (srv.get("env") or {}).items())}


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


def write_meta(row: Path, rec: dict, thinking: bool) -> None:
    """17b: the row's model record — the board's own, never the bundle's"""
    from service import served
    if not thinking:
        served.write_meta(rec)
        return
    row.mkdir(parents=True, exist_ok=True)
    (row / "model_meta.json").write_text(json.dumps(
        {"model": rec["id"] + " · thinking", "base_model": rec["id"], "kind": "instruct",
         "params": None, "kind_reason": "served elsewhere: asked through its server's own chat "
                                        "template", **served.archinfo(rec)}), encoding="utf-8")


def import_bundle(b: dict, path: Path, by: str, say=print, file_sha: str = "",
                  register: str = "", aside: bool = False) -> int:
    """17c: `aside` (--set-aside-shards) sets aside the shards waiting here of
    each task this bundle holds a shard of, when they were made with another
    setup: this bundle's shard starts the task's shards again"""
    from service import config, db, served
    from service import frontier as sf
    # 17c: its setup.json read only once every field is the type it must be
    bad = (setup_problems(b["setup"]) if isinstance(b.get("bundle"), dict)
           else ["its bundle.json isn't a JSON object"])
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
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
    if not row_name or "/" in row_name or row_name.startswith("."):
        say(f"refused — its row {row_name!r} isn't a row's name")
        return REFUSED
    row = config.OUT_DIR / row_name
    reg = registry(row)
    was = next((x for x in reg.get("imports") or [] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, as Runs #{was['sid']} on {was['at']}: nothing changed")
        return 0
    rec = served.get(model) if served.is_served(model) else None
    if register:
        why = ("" if not rec else f"{model} is registered here already: import without "
               "--register") or register_refused(model, file_sha)
        if why:
            say(f"refused — {why}")
            return REFUSED
    bad = checks(b, rec or (_would_be(b, file_sha) if register else None), file_sha)
    tasks = [t for t in fb.TASKS if t in (bundle.get("tasks") or {})]
    set_aside_for: set[str] = set()
    if not bad and shard:
        clash = shard_conflicts(reg, b, tasks, shard)
        if clash and aside:
            set_aside_for = {x.split(":", 1)[0] for x in clash}
            for x in clash:
                say(f"set aside — {x}")
        else:
            bad = clash
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    if register:
        rec = registered_here(b, register, by)
        say(f"{model} registered as {rec['name']}: a model run on rented GPUs only, its file "
            f"{(gg.get('name') or '')} pinned by the sha256 you gave, {file_sha[:16]}…")
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
    shards = config.OUT_DIR.with_name("shards") / row_name
    staging = config.OUT_DIR.with_name("staging") / f"{row_name}-{stamp}" / row_name
    lines, ready, todo_shards, aside_shards = [], {}, {}, []

    def stage(t: str, answers: dict, task_setup: bytes) -> None:
        d = sf.task_dir(staging, t)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(
            json.dumps(answers[k], ensure_ascii=False) + "\n" for k in sorted(answers)),
            encoding="utf-8")
        (d / sf.SETUP).write_bytes(task_setup)

    # 17c: everything staged and scored first — the row, the shards waiting
    # and the registry change only once every task has scored, in one go
    for t in tasks:
        prefix = f"results/{row_name}/{t}_0shot/{sf.SUB}/"
        ans = _answers(b, row_name, t)
        tsetup = b["files"].get(prefix + sf.SETUP, b"{}")
        entry = {"gpu": gpu, "sha256": b["sha256"], "bundle": path.name, "at": stamp,
                 "by": by, "gguf_sha256": gg.get("sha256"), "llama_cpp": srv.get("build"),
                 "answers": len(ans), "setup": shard_setup(srv)}
        if shard:
            i, n = shard
            sh = (reg.get("shards") or {}).get(t) or {}
            if (sh.get("n") and sh["n"] != n) or (aside and t in set_aside_for):
                aside_shards.append((t, sh.get("n")))
                sh = {}
            sh = {"n": n, "have": {**dict(sh.get("have") or {}), str(i): entry}}
            todo_shards[t] = (sh, b["files"][prefix + sf.ANSWERS], tsetup)
            lines.append(f"{t}: shard {i} of {n}, {len(ans):,} answers from a rented GPU ({gpu})")
            missing = [j for j in range(1, n + 1) if str(j) not in sh["have"]]
            if missing:
                lines.append(f"{t}: shard{'s' if len(missing) > 1 else ''} "
                             f"{', '.join(map(str, missing))} of {n} missing — {t} is scored once "
                             f"{'it is' if len(missing) == 1 else 'they are'} imported")
                continue
            merged: dict = {}
            for j in range(1, n + 1):
                merged.update(ans if j == i else
                              sf.read_answers(shards / t / f"{j}-of-{n}" / sf.ANSWERS))
            stage(t, merged, tsetup)
            parts = [sh["have"][str(j)] for j in range(1, n + 1)]
            ready[t] = {**entry, "gpus": sorted({x["gpu"] for x in parts}), "shards": n,
                        "shard_bundles": [{"shard": j, **{k: x[k] for k in ("gpu", "sha256",
                                                                            "bundle")}}
                                          for j, x in enumerate(parts, 1)],
                        "answers": len(merged)}
            lines.append(f"{t}: every shard is in ({n} of {n}) · {len(merged):,} answers")
        else:
            stage(t, ans, tsetup)
            ready[t] = {**entry, "gpus": [gpu]}
            lines.append(f"{t}: {len(ans):,} answers from a rented GPU ({gpu})")

    status, error, words, scored = "done", "", [], {}
    for t in ready:
        try:
            sc = sf.score_task(staging, t, rec)
        except Exception as e:                          # noqa: BLE001 — the row is untouched
            status, error = "failed", f"scoring {t}: {e}"
            words.append(f"{fb.BENCH[t]['label']}: not imported — scoring it failed ({e})")
            continue
        if sc and sc.get("refused"):
            status, error = "failed", sc["refused"]
            words.append(f"{sc['refused']} — not imported")
            continue
        scored[t] = sc
    if status == "done":
        # every task scored: the shards and the row take them now, together
        for t, n_was in aside_shards:
            old = shards / t
            if old.exists():
                where = (config.OUT_DIR.with_name("earlier") / row_name
                         / f"{t}-shards{f'-of-{n_was}' if n_was else ''}-before-import-{stamp}")
                where.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(where))
                lines.append(f"{t}: the shards waiting here were set aside, kept at {where}")
            (reg.get("shards") or {}).pop(t, None)
        for t, (sh, raw, tsetup) in todo_shards.items():
            slot = shards / t / f"{shard[0]}-of-{shard[1]}"
            slot.mkdir(parents=True, exist_ok=True)
            (slot / sf.ANSWERS).write_bytes(raw)
            (slot / sf.SETUP).write_bytes(tsetup)
            reg.setdefault("shards", {})[t] = sh
        for t, entry in ready.items():
            was = sf.set_aside(row, t, "before-import")
            if was:
                lines.append(f"{t}: the answers here before are kept at {was}")
            dest = sf.task_dir(row, t)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sf.task_dir(staging, t)), str(dest))
            reg.setdefault("tasks", {})[t] = entry
            if scored.get(t):
                _mark_where(row, t, f"{WHERE_WORDS} ({', '.join(entry['gpus'])})")
                words.append(sf.words(t, scored[t]))
        if ready:
            write_meta(row, rec, on)
    else:
        words.append("nothing was imported; the row is as it was, and the bundle can be "
                     "imported again")
    shutil.rmtree(staging.parent, ignore_errors=True)
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
    if status == "done":
        # 17c: recorded as imported only when it was
        reg.setdefault("imports", []).append(
            {"sha256": b["sha256"], "sid": sid, "tasks": tasks, "gpu": gpu, "by": by,
             "at": stamp, "bundle": path.name, **({"shard": list(shard)} if shard else {})})
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
