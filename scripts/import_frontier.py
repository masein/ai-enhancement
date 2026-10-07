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
import math
import re
import shutil
import sys
import time
from collections import Counter
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
# 17c: a value quoted or bare — a bare one without the full stop that ends a
# sentence ("…LOOKAHEAD=1.")
_ENV_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]*)=(?:\"([^\"]*)\"|'([^']*)'|([A-Za-z0-9_.:/+-]+))")
ROUTING_ENV = "LLAMA_MOE_"
# 17c: speculative decoding under every spelling llama-server takes — a draft
# model fetched from Hugging Face (-hfd, --hf-repo-draft, …) too
_SPEC = re.compile(r"^(--spec-[a-z-]+|--draft[a-z-]*|-md|--model-draft|-ngld|--gpu-layers-draft"
                   r"|-cd|--ctx-size-draft|-devd|--device-draft|-hfd|-hfrd|--hf-repo-draft"
                   r"|-hffd|--hf-file-draft|-mdu|--model-draft-url|-ctkd|--cache-type-k-draft"
                   r"|-ctvd|--cache-type-v-draft)$")
# and through the environment: llama-server reads LLAMA_ARG_<FLAG> as its flag
_SPEC_ENV = re.compile(r"^LLAMA_ARG_\w*(DRAFT|SPEC)\w*$")
# lookahead said in words only, with no routing variable to compare
_LOOKAHEAD_WORDS = re.compile(r"(?i)\blookahead\b")
# 17d: lookahead said not to be used isn't lookahead said in words
_NO_LOOKAHEAD = re.compile(r"(?i)\b(?:no|non|without)[\s-]+lookahead\b"
                           r"|\blookahead[\s:=-]*(?:off|disabled|none|false|0)\b")


def _env_pairs(text: str) -> dict[str, str]:
    out = {}
    for m in _ENV_TOKEN.finditer(text or ""):
        v = next(g for g in m.groups()[1:] if g is not None)
        out[m.group(1)] = v if m.group(2) is not None or m.group(3) is not None \
            else v.rstrip(".,;:")
    return out


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
    return sorted(set(out))                 # 17c: a flag said twice is one flag


def record_launch(rec: dict) -> dict:
    """the registered setup's routing environment and speculative flags — 17c:
    from its own fields first (its launch flags and environment, and the GGUF
    setup it serves the same file as), "How it's served" only when they hold
    none; a flag said in both counted once; quoted values read; lookahead said
    in words only, with no variable, reported as such (`words`)"""
    from service import db
    fields = [rec.get("flags") or "", rec.get("env") or ""]
    sa = rec.get("same_as") or {}
    if sa.get("gguf"):
        g = db.gguf_get(sa["gguf"]) or {}
        su = next((x for x in g.get("setups") or [] if x.get("id") == sa.get("setup")), None)
        if su:
            fields += [" ".join(f"{k}={v}" for k, v in (su.get("env") or {}).items()),
                       " ".join(su.get("flags") or [])]
    how = rec.get("how") or ""

    def read(text: str) -> tuple[dict, list[str]]:
        pairs = _env_pairs(text)
        env = {k: v for k, v in pairs.items() if k.startswith(ROUTING_ENV)}
        # 17d: a flag's value without a sentence's full stop ("--spec-type mtp.")
        toks = [t.rstrip(".") for t in re.split(r"[\s,;()]+", text) if t.rstrip(".")]
        spec = _spec_flags(toks) + sorted(f"{k}={v}" for k, v in pairs.items()
                                          if _SPEC_ENV.match(k))
        return env, sorted(set(spec))
    env, spec = read(" ".join(t for t in fields if t))
    if not env and not spec:
        env, spec = read(how)
    words = bool(not env and _LOOKAHEAD_WORDS.search(
        _NO_LOOKAHEAD.sub("", f"{how} {rec.get('name') or ''}")))
    return {"env": env, "spec": spec, "drafts": rec.get("speculative") is True,
            "words": words}


# 17d: the flags that change what a server answers — the board's own resume
# asks again when one of them changes (an import lets the box's differ: the
# parity check covers them)
_ANSWERS = re.compile(r"^(-ctk|--cache-type-k|-ctv|--cache-type-v|--chat-template|"
                      r"--chat-template-file|--chat-template-kwargs|--reasoning-budget|"
                      r"--reasoning-format|--override-kv|--rope-scaling|--rope-scale|"
                      r"--rope-freq-base|--rope-freq-scale|--yarn-[a-z-]+|--no-jinja|"
                      r"--jinja)$")


def answer_flags(rec: dict) -> list[str]:
    """the registered launch's flags that change its answers, each with its
    value, from its own fields (its launch flags and its GGUF setup's)"""
    from service import db
    texts = [rec.get("flags") or ""]
    sa = rec.get("same_as") or {}
    if sa.get("gguf"):
        g = db.gguf_get(sa["gguf"]) or {}
        su = next((x for x in g.get("setups") or [] if x.get("id") == sa.get("setup")), None)
        if su:
            texts.append(" ".join(su.get("flags") or []))
    toks = [t for t in re.split(r"\s+", " ".join(texts)) if t]
    out = []
    for i, t in enumerate(toks):
        name, _, val = t.partition("=")
        if not _ANSWERS.match(name):
            continue
        if not val and i + 1 < len(toks) and (not toks[i + 1].startswith("-")
                                              or re.fullmatch(r"-\d+", toks[i + 1])):
            val = toks[i + 1]
        out.append(f"{name}={val}" if val else name)
    return sorted(set(out))


def box_launch(server: dict) -> dict:
    env = {k: str(v) for k, v in (server.get("env") or {}).items() if k.startswith(ROUTING_ENV)}
    spec = _spec_flags([str(x) for x in server.get("flags") or []]) + sorted(
        f"{k}={v}" for k, v in (server.get("env") or {}).items() if _SPEC_ENV.match(k))
    return {"env": env, "spec": sorted(set(spec)),
            # 17c: what its slots said of speculation, when the box asked
            "slots_draft": server.get("speculative")}


def launch_differs(rec: dict, server: dict) -> list[str]:
    """each way the box's launch differs from the registered one, in words"""
    want, got = record_launch(rec), box_launch(server)
    out = []
    if want.get("words"):
        out.append(f"routing: {rec['id']}'s record says lookahead in words, with no routing "
                   "variable to compare a box with — give its launch's environment on its page "
                   "(Launch env), then import again")
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
    # 17c: and what the box's own slots said, against what is registered
    drafting = bool(want["spec"] or want["drafts"])
    if got.get("slots_draft") is not None and bool(got["slots_draft"]) != drafting:
        out.append(f"speculative decoding: the box's slots said they "
                   f"{'draft' if got['slots_draft'] else 'don’t draft'} tokens; {rec['id']} is "
                   f"registered {'drafting' if drafting else 'without it'}")
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


def registered_here(b: dict, name: str, by: str, public_weights: bool = False) -> dict:
    """a served entry for a model run only on rented GPUs, from its bundle —
    17g: with public weights when the import said so (G6's calibration), the
    one way a served model's raw runs may be exported public"""
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
    if public_weights:
        db.public_set(rec["id"], True, by)          # 17h: the board's mark, shown and cleared on its page
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
    "gpu": {"name": str, "count": int, "names": [str]},
    "tasks": {"*": {"protocol_version": str, "revision": str, "epochs": int, "budget": int,
                    "family": str}},
    # 17g: what the run's own view and the where read — checked before
    # anything moves (a box given as a number was a crash after the answers
    # had moved)
    "box": str, "sessions": int, "started_at": str, "finished_at": str, "image": str,
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


# 17d: and bundle.json's own fields (a wrong type gave a traceback)
_BUNDLE_TYPES = {"format": int, "model": str, "row": str, "thinking": bool, "tasks": dict,
                 "gguf_sha256": str, "shard": dict, "files": dict}


def bundle_problems(bundle) -> list[str]:
    """each field the import reads of bundle.json, the type it must be"""
    if not isinstance(bundle, dict):
        return ["its bundle.json isn't a JSON object"]
    out: list[str] = []
    for k, want in _BUNDLE_TYPES.items():
        v = bundle.get(k)
        if v is not None and (not isinstance(v, want)
                              or (isinstance(v, bool) and want is not bool)):
            out.append(f"bundle.json's {k} is {type(v).__name__}, not {want.__name__}")
    return out


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


def split_problem(gg: dict, file_sha: str) -> str:
    """17i: a GGUF in parts — '' when its parts make the split identity the
    bundle gives (the import recorded the parts unchecked) and --file-sha256
    isn't one part's own sha256 (refused with a line that didn't say so)"""
    parts = gg.get("parts") if isinstance(gg, dict) else None
    if not parts:
        return ""
    pairs = [(str(x.get("name") or ""), str(x.get("sha256") or "")) for x in parts
             if isinstance(x, dict)]
    ident = rb.split_sha(pairs)
    if len(pairs) != len(parts) or ident != gg.get("sha256"):
        return (f"its GGUF's {len(parts)} parts don't make the split identity the bundle gives "
                f"({str(gg.get('sha256'))[:16]}…, its parts make {ident[:16]}…): the bundle "
                "isn't as the box wrote it")
    hit = next((n for n, h in pairs if h == file_sha), None)
    if hit:
        return (f"--file-sha256 {file_sha[:16]}… is the sha256 of one part, {hit}: a GGUF in "
                f"{len(pairs)} parts is registered by its split identity, {ident} — the sha256 "
                "of its parts' names and sha256s (frontier_fetch.py prints it, each part under "
                "it)")
    return ""


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
    split = split_problem(setup.get("gguf") or {}, file_sha)
    if split:
        return [split]
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
        # 17d: a shard's off answers that thought are counted, never refused
        # here — the 1% rule is the whole benchmark's, once its shards merge
        # (score_task); a thinking shard with none is refused at once
        answers = [r["answer"] for r in lines]
        why = sf.thinking_refused(t, "on" if on else "off", answers) if not shard or on else ""
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


def shard_conflicts(reg: dict, b: dict, tasks: list[str], shard: tuple[int, int],
                    shards: Path | None = None) -> list[str]:
    """17b: a task whose shards here were made with another build, other flags
    or another environment than this one — never merged into one row. 17g:
    nor asked another way — each task's own setup (its token limit, its
    protocol, its sampling: what `against` compares a row's answers by) read
    from each shard waiting here, beside the bundle's"""
    from service import frontier as sf
    mine = shard_setup(b["setup"].get("server") or {})
    out = []
    again = ("every shard of a task is run the same way: import with --set-aside-shards to set "
             "the shards here aside and start its shards again with this one")
    for t in tasks:
        sh = (reg.get("shards") or {}).get(t) or {}
        try:
            here = json.loads(b["files"].get(
                f"results/{b['bundle'].get('row')}/{t}_0shot/{sf.SUB}/{sf.SETUP}") or b"{}")
        except ValueError:
            here = {}
        for j, x in sorted((sh.get("have") or {}).items()):
            if j == str(shard[0]):
                continue
            if x.get("setup") and x["setup"] != mine:
                diff = [k for k in mine if mine[k] != x["setup"].get(k)]
                out.append(f"{t}: shard {j} here was made with another setup ({', '.join(diff)}: "
                           f"{', '.join(repr(x['setup'].get(k)) for k in diff)} there, "
                           f"{', '.join(repr(mine[k]) for k in diff)} here) — {again}")
                continue
            # 17g: shards of another split are set aside anyway ("resetting")
            if shards is None or not isinstance(here, dict) or sh.get("n") != shard[1]:
                continue
            there = sf._read_json(shards / t / f"{j}-of-{shard[1]}" / sf.SETUP)
            diff = [k for k in sf.setup_differs(there, here) if k != "shard"]
            if diff:
                out.append(f"{t}: shard {j} here was asked another way ("
                           + "; ".join(f"{k}: {_said(there.get(k))} there, {_said(here.get(k))} "
                                       "here" for k in diff)
                           + f") — {again}")
    return out


def _said(v) -> str:
    """a setup's value in a refusal: 16,384 rather than 16384"""
    return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else repr(v)


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
                  register: str = "", aside: bool = False, public_weights: bool = False) -> int:
    """17c: `aside` (--set-aside-shards) sets aside the shards waiting here of
    each task this bundle holds a shard of, when they were made with another
    setup: this bundle's shard starts the task's shards again"""
    from service import config, db, served
    from service import frontier as sf
    # 17c: its setup.json read only once every field is the type it must be
    bad = bundle_problems(b.get("bundle")) or setup_problems(b["setup"])
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
        clash = shard_conflicts(reg, b, tasks, shard,
                                config.OUT_DIR.with_name("shards") / row_name)
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
        rec = registered_here(b, register, by, public_weights)
        say(f"{model} registered as {rec['name']}: a model run on rented GPUs only, its file "
            f"{(gg.get('name') or '')} pinned by the sha256 you gave, {file_sha[:16]}…")
    if file_sha and not registered_sha(rec):
        # 17i: and its parts, checked against the identity (checks())
        rec["file_sha256"] = {"sha256": file_sha, "by": by, "at": time.time(),
                              "name": gg.get("name") or "",
                              **({"parts": gg["parts"]} if gg.get("parts") else {})}
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
    late: dict = {}                 # 17i: a task the row holds, its record lost to a kill

    def stage(t: str, answers: dict, task_setup: bytes, keep: Path | None = None) -> None:
        d = sf.task_dir(staging, t)
        d.mkdir(parents=True, exist_ok=True)
        (d / sf.ANSWERS).write_text("".join(
            json.dumps(answers[k], ensure_ascii=False) + "\n" for k in sorted(answers)),
            encoding="utf-8")
        (d / sf.SETUP).write_bytes(task_setup)
        # 17f: the grades of answers that didn't change go with them (each
        # grade names the answer it graded: a changed one is graded again).
        # 17g: whenever the setup is the same, however many changed — only
        # those whose text is the same as the one graded
        if keep is not None:
            keep_grades(keep, d, answers)

    def against(t: str, new: dict, where: Path, task_setup: bytes, what: str) -> str:
        """17f: 'same', 'more' or 'other' — told by the answers, not the
        tarball's bytes (a step's bundle made again is the same answers in a
        new tarball), and said. Another setup is 'other', whatever the answers"""
        have = sf.read_answers(where / sf.ANSWERS)
        try:
            old_setup = json.loads((where / sf.SETUP).read_text(encoding="utf-8"))
            new_setup = json.loads(task_setup or b"{}")
        except (OSError, ValueError):
            old_setup = new_setup = {}
        other = [k for k in sf.setup_differs(old_setup, new_setup) if k != "shard"]
        how = "setup" if other else compare_answers(have, new)
        if how == "same":
            lines.append(f"{t}: the same {len(new):,} answers as {what} — nothing changed, its "
                         "grades kept")
        elif how == "more":
            kept = sum(1 for k in have if k in new and not have[k].get("unanswered"))
            more = len(new) - kept
            lines.append(f"{t}: {more:,} answer{'s' if more != 1 else ''} more than {what} "
                         f"holds, the grades of the {kept:,} unchanged kept")
        elif how == "flags":
            n = sum(1 for k, r in have.items() if _flags(new.get(k) or {}) != _flags(r))
            lines.append(f"{t}: the same {len(new):,} answers as {what}, {n:,} with another "
                         "finish or token count — taken from this bundle, every grade kept")
        elif how == "other" and have:
            # 17g: some answers differ — the rest keep their grades
            changed = sum(1 for k, r in have.items()
                          if (new.get(k) or {}).get("answer") != r.get("answer"))
            lines.append(f"{t}: {changed:,} of {len(have):,} answers differ from {what} — "
                         "those are graded again, the unchanged keep their grades")
        return how

    # 17c: everything staged and scored first — the row, the shards waiting
    # and the registry change only once every task has scored, in one go
    for t in tasks:
        prefix = f"results/{row_name}/{t}_0shot/{sf.SUB}/"
        ans = _answers(b, row_name, t)
        tsetup = b["files"].get(prefix + sf.SETUP, b"{}")
        # 17e: the answers that ran out of room, and how many of them end in a
        # loop — information only (the cell shows the share that ran out)
        cut = [r.get("answer") or "" for r in ans.values()
               if fb.ran_out(r.get("answer") or "", r.get("finish"))]
        if cut:
            loops = sum(1 for x in cut if fb.ends_in_loop(x))
            lines.append(f"{t}: {len(cut):,} of these {len(ans):,} answers ran out of room, "
                         f"{loops:,} of them ending in a loop (the last passage repeating) — "
                         "for information")
        # 17f: and the questions the box wrote off, the server having failed on them
        off = sorted({str(r.get("id")) for r in ans.values() if r.get("unanswered")})
        if off:
            n_off = sum(1 for r in ans.values() if r.get("unanswered"))
            lines.append(f"{t}: {n_off:,} written off as no answer, counted wrong — the server "
                         f"failed on {', '.join(off[:5])}{' …' if len(off) > 5 else ''} "
                         "(remote_gguf.py --ask-written-off asks them again)")
        entry = {"gpu": gpu, "gpu_names": gpu_names(setup), "sha256": b["sha256"],
                 "bundle": path.name, "at": stamp,
                 "by": by, "gguf_sha256": gg.get("sha256"), "llama_cpp": srv.get("build"),
                 "answers": len(ans), "setup": shard_setup(srv),
                 # 17f: G5's box it ran on, as its line said (never its address)
                 "box": setup.get("box") or ""}
        if shard:
            i, n = shard
            sh = (reg.get("shards") or {}).get(t) or {}
            resetting = bool((sh.get("n") and sh["n"] != n) or (aside and t in set_aside_for))
            if resetting:
                aside_shards.append((t, sh.get("n")))
                sh = {}
            sh = {"n": n, "have": {**dict(sh.get("have") or {}), str(i): entry}}
            # 17f: the shard here already, the same answers: nothing to do —
            # unless its shards are being set aside, or split another way.
            # 17i: or the registry doesn't list it — an import killed after
            # its slot was written and before the registry (a deploy during
            # a round): every retry said "nothing changed", and the shard was
            # never recorded, the benchmark never scored
            listed = str(i) in dict(((reg.get("shards") or {}).get(t) or {}).get("have") or {})
            if not resetting and against(t, ans, shards / t / f"{i}-of-{n}", tsetup,
                                         f"shard {i} of {n} here") == "same":
                if listed:
                    continue
                lines.append(f"{t}: shard {i} of {n} was here but not recorded — an import "
                             "stopped part-way: recorded now")
            todo_shards[t] = (sh, b["files"][prefix + sf.ANSWERS], tsetup)
            lines.append(f"{t}: shard {i} of {n}, {len(ans):,} answers from a rented GPU ({gpu})")
            held = 0 if on else sum(1 for r in ans.values() if sf.thought(r.get("answer") or ""))
            if held:
                # 17d: counted here, judged on the whole once its shards merge
                lines.append(f"{t}: {held} of this shard's {len(ans):,} answers hold thinking, "
                             f"though it was off — the 1% rule is the whole benchmark's, once "
                             "its shards merge")
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
            parts = [sh["have"][str(j)] for j in range(1, n + 1)]
            # 17g: each box's cards, a box of two named twice
            cards: Counter = Counter()
            for x in parts:
                cards |= Counter(x.get("gpu_names") or [x["gpu"]])
            whole = {**entry, "gpus": sorted(cards.elements()), "shards": n,
                     "shard_bundles": [{"shard": j, **{k: x.get(k) for k in (
                                            "gpu", "sha256", "bundle", "box")}}
                                       for j, x in enumerate(parts, 1)],
                     "boxes": sorted({x.get("box") for x in parts if x.get("box")}),
                     "answers": len(merged)}
            how = against(t, merged, sf.task_dir(row, t), tsetup, "the row")
            if how == "same":
                if t not in (reg.get("tasks") or {}):
                    # 17i: the row took them before the import was killed: its
                    # record, as the import would have written it
                    late[t] = whole
                continue
            # 17g: one changed answer in a remade shard kept no grade of either
            stage(t, merged, tsetup, keep=sf.task_dir(row, t) if how != "setup" else None)
            ready[t] = whole
            lines.append(f"{t}: every shard is in ({n} of {n}) · {len(merged):,} answers")
        else:
            how = against(t, ans, sf.task_dir(row, t), tsetup, "the row")
            if how == "same":
                continue
            stage(t, ans, tsetup, keep=sf.task_dir(row, t) if how != "setup" else None)
            ready[t] = {**entry, "gpus": gpu_names(setup),
                        "boxes": [entry["box"]] if entry["box"] else []}
            lines.append(f"{t}: {len(ans):,} answers from a rented GPU ({gpu})")

    if not ready and not todo_shards and not aside_shards:
        # 17f: every task's answers are here already: a bundle made again
        # changes nothing, and adds no run
        shutil.rmtree(staging.parent, ignore_errors=True)
        for x in lines:
            say(x)
        say(f"imported already: the same answers as the row {model}"
            f"{' · thinking' if on else ''} holds — nothing changed")
        return 0
    status, error, words, scored = "done", "", [], {}
    # 17g: where it ran, in words, before anything moves
    box = setup.get("box") or ""
    here = where_words(gpu_names(setup), [box] if box else [])
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
            # 17d: the bundles those shards came from aren't imported any more:
            # their record goes, so the same command brings each back
            was = ((reg.get("shards") or {}).get(t) or {}).get("have") or {}
            gone = {x.get("sha256") for x in was.values()} - {b["sha256"]}
            if gone:
                reg["imports"] = [x for x in reg.get("imports") or []
                                  if x.get("sha256") not in gone]
                names = sorted({x.get("bundle") or "" for x in was.values()
                                if x.get("sha256") in gone})
                lines.append(f"{t}: the shards set aside came from {', '.join(names)} — import "
                             f"{'it' if len(names) == 1 else 'them'} again once this task's "
                             "shards are run one way")
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
        for t, entry in late.items():
            reg.setdefault("tasks", {})[t] = entry
        for t, entry in ready.items():
            was = sf.set_aside(row, t, "before-import")
            if was:
                lines.append(f"{t}: the answers here before are kept at {was}")
            dest = sf.task_dir(row, t)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sf.task_dir(staging, t)), str(dest))
            reg.setdefault("tasks", {})[t] = entry
            if scored.get(t):
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
                 f"imported from a rented GPU ({gpu}){of}{f' · box {box}' if box else ''} · "
                 f"{path.name}", thinking=on,
                 tasks=tasks if len(tasks) < len(fb.TASKS) else None, status=status)
    # 17f: the row says where it ran, and when on the box — its import's time
    # is when it was made here (created_at)
    ran = {k: _epoch(setup.get(k)) for k in ("started_at", "finished_at")}
    db.update(sid, finished_at=ran["finished_at"] or time.time(), progress=line, error=error,
              where_ran=here, **({"started_at": ran["started_at"]} if ran["started_at"] else {}))
    log = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"===== [{sid}] imported {path.name} (sha256 {b['sha256'][:16]}) by {by}: "
                   f"{', '.join(tasks)}{of}, run on a rented GPU ({gpu}) =====\n" + b["log"]
                   + "".join(f"\n[import] {x}" for x in lines) + f"\n[import] {line}\n",
                   encoding="utf-8")
    if status == "done":
        # 17c: recorded as imported only when it was. 17f: with what the run's
        # own view shows — the box, its times and sessions, the image, the file
        reg.setdefault("imports", []).append(
            {"sha256": b["sha256"], "sid": sid, "tasks": tasks, "gpu": gpu, "by": by,
             "at": stamp, "bundle": path.name, **({"shard": list(shard)} if shard else {}),
             "box": box, "where": here, "image": setup.get("image") or "",
             "started_at": setup.get("started_at"), "finished_at": setup.get("finished_at"),
             "sessions": setup.get("sessions"), "gguf_sha256": gg.get("sha256"),
             "imported_at": time.time()})
        _write_registry(row, reg)
        # 17f: each score says where it ran — every box its shards came from —
        # and the Runs rows it came from
        for t in ready:
            if scored.get(t):
                _mark_where(row, t, *where_and_runs(row, t))
        for t in late:
            _mark_where(row, t, *where_and_runs(row, t))
        for t in todo_shards:
            _mark_held(row, t)
    for x in lines:
        say(x)
    say(f"the row {model}{' · thinking' if on else ''}: {line}")
    say(f"Runs #{sid}, its log the bundle's")
    return 0 if status == "done" else 1


def keep_grades(src: Path, dst: Path, answers: dict) -> int:
    """17g: the grades (and no-grades, and those kept aside) of the answers
    whose text is the one graded, from `src` to `dst` — every other dropped,
    graded again. How many grades went with their answers"""
    from service import frontier as sf
    g = sf.read_grades(src)
    if not g:
        return 0
    now = {sf.gkey(r.get("id"), r.get("epoch")): sf.answer_sha(r.get("answer") or "")
           for r in answers.values() if isinstance(r, dict)}
    kept = 0

    def walk(x):
        nonlocal kept
        if isinstance(x, list):
            return [walk(v) for v in x]
        if not isinstance(x, dict):
            return x
        out = {}
        for k, v in x.items():
            if k in ("items", "refused") and isinstance(v, dict) and all(
                    isinstance(e, dict) and "answer_sha256" in e for e in v.values()):
                out[k] = {q: e for q, e in v.items() if now.get(q) == e["answer_sha256"]}
                if k == "items" and x is g:
                    kept += len(out[k])
            else:
                out[k] = walk(v)
        return out
    g = walk(g)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / sf.GRADES).write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding="utf-8")
    return kept


def compare_answers(have: dict, new: dict) -> str:
    """17f: a bundle's answers for a task against those here — 'same' (the
    same answers), 'more' (those here, unchanged, and more; or one written
    off here, answered now), 'flags' (17h: the same answers, another finish
    or token count), or 'other' (a different run: set aside)"""
    if not have:
        return "other"
    for k, r in have.items():
        n = new.get(k)
        if n is None:
            return "other"
        if (n.get("answer"), bool(n.get("unanswered"))) != (r.get("answer"),
                                                            bool(r.get("unanswered"))) \
                and not r.get("unanswered"):
            return "other"
    changed = any((new[k].get("answer"), bool(new[k].get("unanswered")))
                  != (r.get("answer"), bool(r.get("unanswered"))) for k, r in have.items())
    if len(new) != len(have) or changed:
        return "more"
    # 17h: the same answers with another finish or token count (one that ran
    # out of room said so, and the row kept the old flags): taken, the grades
    # kept — each grade names its answer's text, which is the same
    if any(_flags(new[k]) != _flags(r) for k, r in have.items()):
        return "flags"
    return "same"


def _flags(r: dict) -> tuple:
    return (r.get("finish"), r.get("tokens"))


def _mark_where(row: Path, task: str, where: str, runs: list[int] | None = None) -> None:
    """the task's results say where it ran (the score's own detail) — 17f: and
    the Runs rows it came from"""
    from service import frontier as sf
    for f in sf.task_dir(row, task).glob("results_*.json"):
        blob = json.loads(f.read_text(encoding="utf-8"))
        blob.setdefault("frontier", {})["where"] = where
        if runs:
            blob["frontier"]["runs"] = runs
        f.write_text(json.dumps(blob, indent=1), encoding="utf-8")


def short_gpu(name) -> str:
    """'NVIDIA GeForce RTX 5090' → 'RTX 5090' — 17g: '' for none, or an unknown"""
    n = re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", str(name or "").strip())
    return "" if n.lower() in ("", "a gpu", "none", "unknown") else n


def gpu_names(setup: dict) -> list[str]:
    """17g: every card a box ran on — the bundle's list when it has one (the
    image after 308fcf3), else its first card's, once"""
    g = setup.get("gpu") if isinstance(setup.get("gpu"), dict) else {}
    names = [str(x) for x in g.get("names") or [] if x]
    return names or ([str(g["name"])] if g.get("name") else [])


def where_words(gpus: list[str], boxes: list[str]) -> str:
    """17f: where a run ran, one wording everywhere — 'this server' for the
    board's own (said by the page), 'rented GPU · RTX 5090 · box A3' for an
    import, 'boxes A1, A2' when its shards came from several. 17g: the one
    implementation (the page and Runs read it), 'rented GPU' alone for a GPU
    it doesn't know, and every card of a box: '2 × RTX 5090'"""
    count: dict[str, int] = {}
    for x in gpus or []:                      # one entry a card
        n = short_gpu(x)
        if n:
            count[n] = count.get(n, 0) + 1
    g = ", ".join(f"{k} × {n}" if k > 1 else n for n, k in sorted(count.items()))
    b = sorted({str(x) for x in boxes or [] if x})
    return "rented GPU" + (f" · {g}" if g else "") + (
        f" · box{'es' if len(b) > 1 else ''} {', '.join(b)}" if b else "")


def where_and_runs(row: Path, task: str) -> tuple[str, list[int]]:
    """17g: where a task's answers on the row ran, and the Runs rows they came
    from — from the row's own record of its imports, so a score made again
    (grading) says it as the import did"""
    reg = registry(row)
    x = (reg.get("tasks") or {}).get(task)
    if not isinstance(x, dict):
        return "", []
    shas = {s.get("sha256") for s in x.get("shard_bundles") or []} | {x.get("sha256")}
    runs = sorted({i["sid"] for i in reg.get("imports") or []
                   if i.get("sha256") in shas and isinstance(i.get("sid"), int)})
    return where_words(x.get("gpus") or [x.get("gpu") or ""], x.get("boxes") or []), runs


def shards_held(row: Path, task: str) -> str:
    """17i: a benchmark's shards imported and held, waiting for the rest — in
    words, for its score; '' when none wait. A shard imported after a whole
    run was held with no sign on the board beyond its Runs line"""
    sh = (registry(row).get("shards") or {}).get(task) or {}
    have, n = dict(sh.get("have") or {}), sh.get("n")
    if not have or not isinstance(n, int) or len(have) >= n:
        return ""
    got = sorted(int(k) for k in have if str(k).isdigit())
    return (f"shard{'s' if len(got) > 1 else ''} {', '.join(map(str, got))} of {n} imported and "
            f"held until the other {n - len(got)} {'is' if n - len(got) == 1 else 'are'} in — "
            "this score is the answers here before")


def _mark_held(row: Path, task: str) -> None:
    """17i: the task's results say which of its shards wait"""
    from service import frontier as sf
    words = shards_held(row, task)
    for f in sf.task_dir(row, task).glob("results_*.json"):
        blob = json.loads(f.read_text(encoding="utf-8"))
        fr = blob.setdefault("frontier", {})
        if words:
            fr["shards_held"] = words
        else:
            fr.pop("shards_held", None)
        f.write_text(json.dumps(blob, indent=1), encoding="utf-8")


def _epoch(stamp) -> float | None:
    """'2026-10-06T08:12:03Z' → seconds"""
    try:
        import calendar
        return float(calendar.timegm(time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


# 17f: what the rented boxes are doing, as frontier_fetch.py last read them —
# each step's label, model, what it asks, its line, its times, never an address
BOX_FIELDS = {"label": str, "model": str, "step": str, "thinking": str, "tasks": list,
              "shard": str, "parity": bool, "state": str, "line": str, "started_at": str,
              "sessions": int, "at": (int, float), "seen_at": (int, float), "reachable": bool,
              "safe": bool, "why": str, "box_id": str}
QUIET_S = 45 * 60                       # a box not heard from for this long says so
# 17g: a step done (its box safe to destroy) leaves the list this long after
# the fetch last saw it — thirty days on it still read "done, safe to destroy"
DONE_KEEP_S = 6 * 3600
# 17h: a step not reached for this long leaves the list (a box destroyed
# mid-run stayed listed for ever)
GONE_KEEP_S = 24 * 3600


def boxes_path() -> Path:
    from service import config
    return Path(config.BENCH_ROOT) / "frontier" / "boxes.json"


def _fine(v, t) -> bool:
    """17g: a value of its field's type — a number finite (a box's `at: NaN`
    broke the list's endpoint for every box), a list of text only"""
    if t is list:
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if not isinstance(v, t) or (isinstance(v, bool) and bool not in (
            t if isinstance(t, tuple) else (t,))):
        return False
    return not isinstance(v, float) or math.isfinite(v)


CAP = 300                               # 17i: a string kept on the list, at most


def box_row(r) -> dict | None:
    """17i: one step as the list keeps it — only BOX_FIELDS, each of its type,
    strings cut to CAP (a 2 MB line was stored and served on every poll), a
    list's items too; None unless it has its label, model and step. The one
    guard, on write and on read: a stored row with a number for its step or a
    list for its label was a 500 for every box, and failed the next post"""
    if not isinstance(r, dict):
        return None
    x = {}
    for k, t in BOX_FIELDS.items():
        v = r.get(k)
        if k not in r or not _fine(v, t):
            continue
        x[k] = v[:CAP] if isinstance(v, str) else [i[:CAP] for i in v[:50]] \
            if isinstance(v, list) else v
    return x if x.get("label") and x.get("model") and x.get("step") else None


def store_boxes(records) -> int:
    """the fetch's last reading of each step, kept on the board: only the
    fields above, each of its type — anything else is dropped. 17g: a step
    the board had that this reading hasn't is kept as not reached (a box
    destroyed before it was done vanished), unless it was done — a box safe
    to destroy and gone was destroyed. 17h: only for the boxes this fetch
    asked — the others' steps stay as they were (fetching one box marked every
    other "not reached"). 17i: a step gone from a box the fetch reached is
    gone, at once; the fetch's reading is a list whose first element may say
    which boxes it asked and reached (a board before 17i passes it by), or
    17h's {"steps", "asked"}; a row the board kept is guarded as a new one"""
    asked = reached = every = None
    if isinstance(records, dict):
        asked = {x for x in records.get("asked") or [] if isinstance(x, str)}
        records = records.get("steps")
    if isinstance(records, list) and records and isinstance(records[0], dict) \
            and "asked" in records[0] and "label" not in records[0]:
        head, records = records[0], records[1:]
        asked = {x for x in head.get("asked") or [] if isinstance(x, str)}
        reached = {x for x in head.get("reached") or [] if isinstance(x, str)}
        every = _num(head.get("every"))
    keep = [x for x in (box_row(r) for r in records if isinstance(records, list)) if x] \
        if isinstance(records, list) else []
    now = {(x["label"], x["model"], x["step"]) for x in keep}
    try:
        was = json.loads(boxes_path().read_text(encoding="utf-8")).get("boxes") or []
    except (OSError, ValueError, AttributeError):
        was = []
    for x in (box_row(r) for r in (was if isinstance(was, list) else [])):
        if not x or (x["label"], x["model"], x["step"]) in now:
            continue
        mine = asked is None or not x.get("box_id") or x.get("box_id") in asked
        if not mine:
            keep.append(x)                          # another fetch's box: as it was
        elif reached is not None and x.get("box_id") in reached:
            continue                                # 17i: gone from a box reached: gone
        elif not x.get("safe"):
            keep.append({**x, "reachable": False})
    p = boxes_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    # 17i: and how often the fetch reads them (--every), so the list says when
    # the next reading is due
    tmp.write_text(json.dumps({"posted_at": time.time(), "boxes": keep,
                               **({"every": every} if every and every > 0 else {})}),
                   encoding="utf-8")
    tmp.replace(p)
    return len(keep)


_LEFT = re.compile(r"(?:(?<![\d.])(?P<h>\d{1,6}(?:\.\d{1,6})?) h|(?<![\d.])(?P<m>\d{1,7}) min) left")
_NOF = re.compile(r"(?<![\d,])(?P<n>\d[\d,]{0,15}) of (?P<of>\d[\d,]{0,15})(?![\d,])")


def read_boxes(now: float | None = None) -> dict:
    """17f: the boxes for Runs — each step with what it asks, n of N, when it
    should finish (its benchmark's time left, and its box's later steps at
    the plan's hours), and when it was last heard from. 17i: each row read
    as it is written (box_row) and on its own — one that can't be read is
    left out, never the list; and a row leaves by when it was last seen,
    whatever it says of itself (a box taken off the fetch's command line read
    as live a month on)"""
    now = time.time() if now is None else now
    try:
        got = json.loads(boxes_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"boxes": [], "posted_at": None}
    try:
        import frontier_box as fbx
    except ImportError:                                   # the plan beside the script
        fbx = None
    out = []
    for raw in got.get("boxes") or [] if isinstance(got, dict) else []:
        try:
            r = box_row(raw)
            if r is None:
                continue
            row = _box_read(r, now, fbx)
        except Exception:                               # noqa: BLE001 — that row, never the list
            continue
        if row is not None:
            out.append(row)
    posted = got.get("posted_at") if isinstance(got, dict) else None
    posted = posted if _num(posted) is not None else None
    every = _num(got.get("every")) if isinstance(got, dict) else None
    return {"boxes": out, "posted_at": posted, "every": every,
            "next_at": posted + every if posted and every else None,
            "runs": rented_runs(out, now, posted)}


def _box_read(r: dict, now: float, fbx) -> dict | None:
    seen = _num(r.get("seen_at"))
    if not seen:
        return None                                     # 17i: never seen: not on the list
    if r.get("safe") and now - seen > DONE_KEEP_S:
        return None
    if now - seen > GONE_KEEP_S:
        return None                                     # 17i: whatever "reachable" says
    line = r.get("line") or ""
    # 17h: a line's numbers read only when they are numbers — "v 1.2.3 h
    # left" or "pages , of , done" broke the list for every box. 17i: and
    # never more digits than a number has (three long runs of them did)
    m, nof = _LEFT.search(line), _NOF.search(line)
    left = _num(m["h"]) * 3600 if m and m["h"] and _num(m["h"]) is not None else \
        int(m["m"]) * 60 if m and m["m"] else None
    later = 0.0
    if fbx is not None and left is not None:
        try:
            steps = fbx.box_of(r["label"])
            k = int(r["step"].rsplit("-", 1)[1]) if not r["step"].endswith("parity") else 0
            later = sum(fbx.hours(s) for s in steps[k:] if s[0] != "parity") * 3600
        except (SystemExit, ValueError, IndexError, KeyError, TypeError):
            later = 0.0
    # 17g: quiet only while it should be writing — a step whole or stopped
    # writes no more, and read "not heard from for 600 min" on a box
    # still working
    at = _num(r.get("at"))
    quiet = (now - at) if at and r.get("state") in ("asking", "starting") else None

    def count(x: str | None) -> int | None:
        try:
            return int(x.replace(",", "")) if x else None
        except ValueError:
            return None
    # 17i: two times, each named — when this benchmark finishes, and when the
    # box does, its later steps included ("Expected finish" was the box's,
    # shown on a step's row)
    task_finish = ((at or now) + left) if left is not None else None
    return {**r, "at": at, "seen_at": seen,
            "n": count(nof["n"]) if nof else None, "of": count(nof["of"]) if nof else None,
            "finish": task_finish + later if task_finish is not None else None,
            "task_finish": task_finish,
            "box_finish": task_finish + later if task_finish is not None else None,
            "heard": seen,
            "quiet_min": round(quiet / 60) if quiet and quiet > QUIET_S else None}


# 17i: a step's state in plain words, as Runs says it
STATE_WORDS = {"whole": "Done", "asking": "Running", "starting": "Loading the model",
               "stopped": "Stopped"}


def _ago(seconds: float) -> str:
    m = max(1, round(seconds / 60))
    return f"{m} min" if m < 90 else f"{round(m / 60)} h"


def step_status(b: dict, now: float) -> tuple[str, str]:
    """17i: (its key, its words) — running, loading, done, stopped (and why),
    quiet (asking, and no word for 45 minutes: stopped by hand, likely) or
    unreached (no contact since the fetch last reached it)"""
    if b.get("reachable") is False:
        return "unreached", f"No contact for {_ago(now - (b.get('heard') or now))}"
    st = b.get("state") or "starting"
    if st == "asking" and b.get("quiet_min"):
        return "quiet", f"Stopped? No word for {_ago(b['quiet_min'] * 60)}"
    if st == "stopped":
        return "stopped", "Stopped" + (f": {b['why']}" if b.get("why") else "")
    key = {"whole": "done", "asking": "running", "starting": "loading"}.get(st, "loading")
    return key, STATE_WORDS.get(st, "Loading the model")


def _imported(b: dict, regs: dict) -> bool:
    """17i: a step whose bundle the row's registry shows imported — its Runs
    row is the import's, never a second one"""
    model, think = b.get("model") or "", b.get("thinking")
    row = model.replace("/", "__") + ("__thinking" if think == "on" else "")
    if row not in regs:
        from service import config
        try:
            regs[row] = json.loads((config.OUT_DIR / row / REGISTRY).read_text(
                encoding="utf-8"))
        except (OSError, ValueError, AttributeError):
            regs[row] = {}
    tasks = set(b.get("tasks") or [])
    want = [int(x) for x in str(b.get("shard") or "").split("/")] \
        if re.fullmatch(r"\d+/\d+", str(b.get("shard") or "")) else None
    for x in (regs[row] or {}).get("imports") or [] if isinstance(regs[row], dict) else []:
        if not isinstance(x, dict) or x.get("box") != b.get("label"):
            continue
        if tasks and not tasks <= set(x.get("tasks") or []):
            continue
        if (x.get("shard") or None) != want:
            continue
        return True
    return False


def rented_runs(boxes: list[dict], now: float, posted: float | None) -> list[dict]:
    """17i: the rented runs as Runs shows them — a row a model, benchmark and
    thinking setting, its boxes merged (Humanity's Last Exam on four boxes is
    one row, "414 of 2,158 · 4 boxes", opening to a line a box), the steps
    whose bundles are imported left to their import's own row. Its status:
    Done when every box is; else Running while any box asks; else Loading
    the model while one loads; else No contact when the only others are
    unreached; else Stopped and why. A box stopped or unreached among
    running ones is said on the row, and the benchmark's finish waits on
    it (unknown) rather than leaving it out"""
    regs: dict = {}
    groups: dict = {}
    for b in boxes:
        if b.get("parity"):
            key = (b.get("model"), "parity", "off")
        else:
            key = (b.get("model"), ",".join(b.get("tasks") or []) or b.get("step"),
                   b.get("thinking") or "")
        try:
            if _imported(b, regs):
                continue
        except Exception:                           # noqa: BLE001 — shown, never the list
            pass
        groups.setdefault(key, []).append(b)
    out = []
    for (model, tasks, thinking), steps in groups.items():
        steps = sorted(steps, key=lambda b: (str(b.get("label")), str(b.get("step"))))
        st = [step_status(b, now) for b in steps]
        keys = {k for k, _ in st}
        if keys == {"done"}:
            status, words = "done", "Done"
        elif "running" in keys:
            status, words = "running", "Running"
        elif "loading" in keys:
            status, words = "loading", "Loading the model"
        elif keys <= {"unreached", "done"}:
            status, words = "unreached", next(w for k, w in st if k == "unreached")
        else:
            status, words = "stopped", next(w for k, w in st if k in ("stopped", "quiet"))
        counted = [b for b in steps if b.get("n") is not None and b.get("of")]
        n = sum(b["n"] for b in counted)
        of = sum(b["of"] for b in counted)
        # the benchmark's finish: the last of its boxes', known only while
        # every box not done is running with one
        open_ = [(b, k) for b, (k, _) in zip(steps, st) if k != "done"]
        finish = max((b.get("task_finish") or 0) for b, _ in open_) if open_ and all(
            k == "running" and b.get("task_finish") for b, k in open_) else None
        odd = [f"{b.get('label')} · {b.get('step')}: {w}" for b, (k, w) in zip(steps, st)
               if k not in ("running", "done", "loading") and status in ("running", "loading")]
        # heard: when the box last spoke — for a step gone quiet, its last
        # word (the fetch reached the box just now, and read nothing new)
        said = [(b.get("at") if k == "quiet" else b.get("heard")) or 0
                for b, (k, _) in zip(steps, st)]
        heard = min(said) or None
        behind = bool(posted and heard and heard < posted - 60) or any(
            k in ("unreached", "quiet") for k, _ in st)
        labels = sorted({str(b.get("label")) for b in steps if b.get("label")})
        started = sorted(str(b.get("started_at")) for b in steps if b.get("started_at"))
        out.append({
            "id": f"rented:{model}|{tasks}|{thinking}", "hf_id": model, "tasks": tasks,
            "parity": tasks == "parity", "thinking": thinking, "status": status,
            "status_words": words, "n": n if counted else None, "of": of if counted else None,
            "boxes_n": len(steps), "finish": finish, "attention": odd,
            "heard": heard, "behind": behind, "started_at": started[0] if started else None,
            "where": "rented GPU · " + (f"box {labels[0]}" if len(labels) == 1 else
                                        f"boxes {', '.join(labels)}"),
            "boxes": [{"label": b.get("label"), "step": b.get("step"), "status": k,
                       "words": w, "n": b.get("n"), "of": b.get("of"), "shard": b.get("shard"),
                       "task_finish": b.get("task_finish") if k == "running" else None,
                       "box_finish": b.get("box_finish") if k == "running" else None,
                       "heard": h or None,
                       "behind": bool(posted and b.get("heard") and b["heard"] < posted - 60)
                       or k in ("unreached", "quiet")}
                      for b, (k, w), h in zip(steps, st, said)]})
    order = {"running": 0, "loading": 1, "unreached": 2, "stopped": 3, "done": 4}
    return sorted(out, key=lambda r: (order.get(r["status"], 9), r["hf_id"] or "", r["tasks"]))


def _num(v) -> float | None:
    """a finite number from the boxes' file, else None"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rented_of(sid: int) -> dict | None:
    """17f: what an imported run's own view shows — its import's record, found
    by its Runs id in the row it went to"""
    from service import config
    root = config.OUT_DIR
    for p in sorted(root.glob(f"*/{REGISTRY}")) if root.is_dir() else []:
        try:
            reg = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hit = next((x for x in reg.get("imports") or [] if x.get("sid") == sid), None)
        if hit:
            ran = [_epoch(hit.get(k)) for k in ("started_at", "finished_at")]
            return {**hit, "hours": round((ran[1] - ran[0]) / 3600, 2) if all(ran) else None,
                    "restarts": max(0, int(hit.get("sessions") or 1) - 1)}
    return None
