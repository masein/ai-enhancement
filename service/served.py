"""12f.1: served models — a model something else serves over an
OpenAI-compatible endpoint (masein's llama-server, say), which the board tests
but never starts, stops or restarts.

- **Registered with what its server reports.** `GET /v1/models`, and `/props`
  and `/health` when they answer: the model file and its size, the context
  size and the build are pinned with it. Every run compares them first; a
  different file stops the run.
- **Asked as a local run asks.** Everyday tasks and the Knowledge exam are sent
  one question a chat message, with the settings a local run uses (the
  runner's _everyday_settings — everyday.run_settings — and _exam_settings),
  and the answers are written in the files lm_eval writes, so the marking, the
  judge and the page read them unchanged. The generative Standard tasks go
  through lm_eval's own `local-chat-completions`. Log-likelihood tasks can't be
  asked: a chat endpoint returns no log-probabilities.
- **A server that stops answering** is retried SERVED_RETRY_S, then the run
  stops, keeping what is finished: "the server stopped answering at 140 of 200".
- **The key** stays in the service's database: no endpoint returns it, no log
  or page shows it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, db

PREFIX = "served/"
BACKEND = "local-chat-completions"           # lm_eval's, for the generative tasks
LOGLIK_LINE = ("Multiple-choice benchmarks need the model loaded here; this one is served "
               "elsewhere.")
CHANGED_LINE = ("The server now serves a different file than the one registered. Register it "
                "again if that's intended.")
KEPT_FOR_NEXT = " · the answers it gave are kept: the next run asks only the rest"
SUITES = ("everyday", "judged", "generative")
THINKING = {"on": "on", "off": "off", "auto": "the model decides"}
PINNED = ("file", "size", "ctx", "build")
TS_FMT = "%Y-%m-%dT%H-%M-%S.000000"


class ServerStopped(Exception):
    """the server stopped answering (or refused) at `done` of `total`"""
    def __init__(self, done: int, total: int, why: str = "", refused: str = ""):
        super().__init__(refused or f"the server stopped answering at {done} of {total}")
        self.done, self.total, self.why, self.task = done, total, why, ""


class _Retry(Exception):
    """worth asking again: no answer, a timeout, a 5xx, 408 or 429"""


def is_served(model_id: str) -> bool:
    return (model_id or "").startswith(PREFIX)


def is_phone(rec: dict | None) -> bool:
    """12f.2: a phone build — its registration's box, or "phone" in how it's
    served"""
    return bool(rec) and (bool(rec.get("phone"))
                          or bool(re.search(r"\bphone\b", rec.get("how") or "", re.I)))


# ---------------------------------------------------------------------------
# talking to the server
# ---------------------------------------------------------------------------

def _http(method: str, url: str, key: str = "", body: dict | None = None,
          timeout: float = 10.0) -> tuple[int, bytes]:
    h = {"content-type": "application/json"}
    if key:
        h["authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def root_of(base: str) -> str:
    """http://host:8090/v1 -> http://host:8090, where llama-server keeps
    /props and /health"""
    return re.sub(r"/v1/?$", "", base.rstrip("/"))


def probe(base: str, key: str = "") -> dict:
    """what the server reports: {model, file, size, ctx, build, answered} — or
    ValueError, in one line, when nothing answers"""
    base = (base or "").strip().rstrip("/")
    if not re.match(r"^https?://[^/\s]+", base):
        raise ValueError("The address is the server's OpenAI-compatible base URL, like "
                         "http://host.docker.internal:8090/v1")
    try:
        st, raw = _http("GET", base + "/models", key)
    except Exception as e:                                  # noqa: BLE001 — said in one line
        raise ValueError(f"Nothing answered at {base}: {getattr(e, 'reason', e)}") from None
    if st == 401:
        raise ValueError(f"{base} refused the key (HTTP 401)")
    if st != 200:
        raise ValueError(f"Nothing usable answered at {base}/models (HTTP {st})")
    try:
        data = json.loads(raw).get("data") or []
    except (ValueError, AttributeError):
        raise ValueError(f"{base}/models did not return an OpenAI-style list of models") from None
    if not data:
        raise ValueError(f"{base} serves no model")
    first = data[0]
    meta = first.get("meta") or {}
    out = {"model": str(first.get("id") or ""), "file": "", "size": meta.get("size"),
           "ctx": meta.get("n_ctx_train"), "build": "", "answered": ["models"]}
    root = root_of(base)
    try:
        st, raw = _http("GET", root + "/props", key, timeout=5)
        if st == 200:
            props = json.loads(raw)
            out["answered"].append("props")
            out["file"] = str(props.get("model_path") or "").rsplit("/", 1)[-1]
            out["ctx"] = ((props.get("default_generation_settings") or {}).get("n_ctx")
                          or props.get("n_ctx") or out["ctx"])
            out["build"] = str(props.get("build_info") or props.get("build") or "")
    except Exception:                                       # noqa: BLE001 — optional
        pass
    try:
        st, _ = _http("GET", root + "/health", key, timeout=5)
        if st == 200:
            out["answered"].append("health")
    except Exception:                                       # noqa: BLE001 — optional
        pass
    out["file"] = out["file"] or out["model"].rsplit("/", 1)[-1]
    return out


def pin_of(p: dict) -> dict:
    return {k: p.get(k) for k in ("model", *PINNED)}


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def slug(name: str) -> str:
    return PREFIX + re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")[:80]


def public(rec: dict | None) -> dict | None:
    """a served model as any endpoint may show it: never its key"""
    if not rec:
        return None
    return {**{k: v for k, v in rec.items() if k != "key"}, "has_key": bool(rec.get("key"))}


def get(model_id: str) -> dict | None:
    return db.served_get(model_id)


def all_public() -> list[dict]:
    return [public(r) for r in db.served_all()]


def check(f: dict) -> dict:
    """Check before saving: what the server at this address reports"""
    old = get(slug(f.get("name") or "")) or {}
    key = (f.get("key") or "").strip() or old.get("key", "")
    return probe(f.get("base_url") or "", key)


def register(f: dict, by: str) -> dict:
    """check the server, pin what it reports, and keep it — or say why not"""
    name = (f.get("name") or "").strip()
    how = (f.get("how") or "").strip()
    thinking = f.get("thinking") or "auto"
    if not slug(name)[len(PREFIX):]:
        raise ValueError("A name: it is shown everywhere")
    if not how:
        raise ValueError("How it's served: the build and its flags. It is the record of what "
                         "was tested")
    if thinking not in THINKING:
        raise ValueError("Thinking is on, off, or the model decides")
    mid = slug(name)
    old = get(mid) or {}
    key = (f.get("key") or "").strip() or old.get("key", "")
    base = (f.get("base_url") or "").strip().rstrip("/")
    p = probe(base, key)
    rec = {"id": mid, "name": name, "base_url": base, "key": key,
           "based_on": (f.get("based_on") or "").strip(), "how": how, "thinking": thinking,
           "phone": bool(f.get("phone")),
           # 12f.3: its GGUF file on the server, for llama-perplexity's benchmarks
           "gguf_path": (f.get("gguf_path") or "").strip(),
           "gguf_flags": (f.get("gguf_flags") or "").strip(),
           "pin": pin_of(p), "answered": p["answered"], "by": by, "at": time.time()}
    if rec["gguf_path"] and old.get("gguf_path") == rec["gguf_path"] and old.get("gguf_pin"):
        rec["gguf_pin"] = old["gguf_pin"]
    db.served_put(rec)
    write_meta(rec)
    return public(rec)


def model_dir(rec: dict) -> Path:
    return config.OUT_DIR / rec["id"].replace("/", "__")


def view(rec: dict) -> dict:
    """what History and the page show: how it was served and what was pinned"""
    return {"name": rec["name"], "base_url": rec["base_url"], "how": rec["how"],
            "based_on": rec["based_on"], "thinking": rec["thinking"], "pin": rec["pin"],
            "phone": is_phone(rec),
            "gguf_path": rec.get("gguf_path") or "",
            # 12i.4: measured by its last run — Test a model's time estimate
            "speed": rec.get("speed")}


def record_speed(model_id: str, secs_each: float, n: int) -> None:
    """a run's measured seconds per answer, kept with the model (a speed
    over fewer than ten answers says too little to keep)"""
    rec = get(model_id)
    if not rec or n < 10 or secs_each <= 0:
        return
    rec["speed"] = {"secs_each": round(secs_each, 3), "n": n, "at": time.time()}
    db.served_put(rec)
    write_meta(rec)


def archinfo(rec: dict) -> dict:
    """its shape, as far as the board knows it: its template is the server's,
    and whether it thinks is what it was registered with"""
    return {"ctx": rec["pin"].get("ctx"), "tmpl_sha": "served",
            "reasoning_template": rec["thinking"] != "off",
            "thinking": "never" if rec["thinking"] == "off" else "always",
            "served": view(rec)}


def write_meta(rec: dict) -> None:
    """its model_meta.json, as a run writes one: what the page reads"""
    d = model_dir(rec)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps(
        {"model": rec["id"], "kind": "instruct", "params": None,
         "kind_reason": "served elsewhere: asked through its server's own chat template",
         **archinfo(rec)}), encoding="utf-8")


def check_pin(rec: dict) -> str:
    """'' when the server serves what was registered, else why not — in one line"""
    try:
        now = pin_of(probe(rec["base_url"], rec.get("key", "")))
    except ValueError as e:
        return str(e)
    return CHANGED_LINE if any(now.get(k) != rec["pin"].get(k) for k in PINNED) else ""


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------

def preflight(sub: dict) -> dict:
    """the runner's preflight for a served model: no weights to look at — its
    registration, the suites it can sit, and its server's pin"""
    from .hfmeta import PreflightError
    rec = get(sub["hf_id"])
    if not rec:
        raise PreflightError(f"{sub['hf_id']} is not registered: add it under Test a model ▸ "
                             f"A model served elsewhere")
    if sub["suite"] not in SUITES:
        raise PreflightError(LOGLIK_LINE)
    why = check_pin(rec)
    if why:
        raise PreflightError(why)
    write_meta(rec)
    return {"kind": "instruct", "params": None, "vocab": None, "batch": 1, "need_gb": 0.0,
            "has_template": True, "remote_code": False, "archinfo": archinfo(rec),
            "kind_reason": "served elsewhere", "served": rec}


def settings_for(rec: dict, meta: dict, everyday: bool) -> dict:
    """how a question is asked: what a local run of the same model is given —
    the runner's _everyday_settings (everyday.run_settings, the function the
    Playground reads too) or _exam_settings. Thinking on or off is asked of
    the server's template; "the model decides" asks nothing"""
    from . import runner
    s = runner._everyday_settings(meta) if everyday else runner._exam_settings(meta)
    out = {"max_tokens": s["max_gen_toks"], "stop": list(s["until"])[:4],
           "temperature": float(s["temperature"])}
    if s.get("system"):
        out["system"] = s["system"]
    if rec["thinking"] in ("on", "off"):
        out["chat_template_kwargs"] = {"enable_thinking": rec["thinking"] == "on"}
    return out


def prompt_of(doc: dict, everyday: bool) -> str:
    """the text a local run sends: the task's doc_to_text"""
    return doc["prompt"] if everyday else f"{doc['prompt']}\n\nAnswer:"


def _doc_hash(doc: dict) -> str:
    """lm_eval's own: the doc as it dumps it (indent 2), hashed"""
    return hashlib.sha256(json.dumps(doc, indent=2, default=str, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


class Answer(str):
    """a reply's text, with how many tokens the server says it generated
    (12i.4: an answer's length, thinking included, as the server counts it)"""
    tokens: int | None = None


def ask(rec: dict, text: str, s: dict) -> str:
    """one chat message, one reply — thinking the server split off put back in
    its tags, as a local run's text has it, so the board reads it the same"""
    system = [{"role": "system", "content": s["system"]}] if s.get("system") else []
    body = {"model": rec["pin"].get("model") or rec["name"],
            "messages": [*system, {"role": "user", "content": text}],
            **{k: v for k, v in s.items() if k != "system"}}
    try:
        st, raw = _http("POST", rec["base_url"] + "/chat/completions", rec.get("key", ""), body,
                        timeout=config.SERVED_TIMEOUT_S)
    except Exception as e:                                  # noqa: BLE001 — no answer: retried
        raise _Retry(str(getattr(e, "reason", e))) from None
    if st in (408, 429) or st >= 500:
        raise _Retry(f"HTTP {st}")
    if st != 200:
        raise ValueError(f"HTTP {st}: {raw[:160].decode('utf-8', 'replace')}")
    reply = json.loads(raw)
    msg = (reply.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content") or ""
    think = msg.get("reasoning_content") or ""
    out = Answer(f"<think>\n{think}\n</think>\n\n{text}" if think else text)
    used = (reply.get("usage") or {}).get("completion_tokens")
    out.tokens = int(used) if isinstance(used, (int, float)) and used >= 0 else None
    return out


def _ask_patiently(rec: dict, text: str, s: dict) -> str:
    """a server that doesn't answer is asked again for SERVED_RETRY_S"""
    t0 = time.time()
    while True:
        try:
            return ask(rec, text, s)
        except _Retry as e:
            if time.time() - t0 >= config.SERVED_RETRY_S:
                raise ServerStopped(0, 0, str(e)) from None
            time.sleep(min(5.0, max(0.05, config.SERVED_RETRY_S / 24)))


def answer_task(rec: dict, task: str, docs: list[dict], task_out: Path, s: dict,
                everyday: bool, on_progress=None, canceled=lambda: False) -> int:
    """ask every doc and write lm_eval's files: 0, or -15 when canceled — or
    ServerStopped after writing the answers before the first it didn't get.
    `on_progress(done, total, seconds an answer)`"""
    total = len(docs)
    answers: dict[int, str] = {}
    lock = threading.Lock()
    halt: list[ServerStopped] = []
    t0 = time.time()

    def one(i: int) -> None:
        if halt or canceled():
            return
        try:
            a = _ask_patiently(rec, prompt_of(docs[i], everyday), s)
        except ServerStopped as e:
            with lock:
                halt.append(e)
            return
        except ValueError as e:                 # a 4xx: this question, refused — said, not retried
            with lock:
                halt.append(ServerStopped(0, 0, str(e), refused=(
                    f"the server refused question {i + 1} of {total}: {e}")))
            return
        with lock:
            answers[i] = a
            done = len(answers)
        if on_progress:
            on_progress(done, total, (time.time() - t0) / done)

    with ThreadPoolExecutor(max_workers=max(1, config.SERVED_CONCURRENCY)) as pool:
        list(pool.map(one, range(total)))
    if answers:
        record_speed(rec["id"], (time.time() - t0) / len(answers), len(answers))
    # what is finished is kept, in order: a stop keeps the answers before the first gap
    gap = next((i for i in range(total) if i not in answers), total)
    kept = {i: answers[i] for i in sorted(answers) if i < gap}
    if kept or not halt:
        _write(rec, task, docs, kept, task_out, s, total, everyday)
    if canceled():
        return -15
    if halt:
        e = halt[0]
        if not str(e).startswith("the server refused"):
            e = ServerStopped(len(kept), total, e.why)
        e.done, e.total = len(kept), total
        raise e
    return 0


def _write(rec: dict, task: str, docs: list[dict], answers: dict[int, str], task_out: Path,
           s: dict, total: int, everyday: bool) -> None:
    """results_<TS>.json and samples_<task>_<TS>.jsonl, as lm_eval writes them;
    each answer records how it was asked"""
    out = task_out / "served"
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime(TS_FMT, time.gmtime())
    how = {**view(rec), "settings": s}
    gen = {"until": s["stop"], "max_gen_toks": s["max_tokens"], "do_sample": False,
           "temperature": s["temperature"]}
    with open(out / f"samples_{task}_{ts}.jsonl", "w", encoding="utf-8") as fh:
        for i in sorted(answers):
            doc, raw = docs[i], answers[i]
            text = prompt_of(doc, everyday)
            fh.write(json.dumps({
                "doc_id": i, "doc": doc, "target": doc.get("reference", ""),
                "arguments": {"gen_args_0": {"arg_0": text, "arg_1": gen}},
                "resps": [[raw]], "filtered_resps": [raw], "filter": "none",
                # 12i.4: its length as the server counted it, thinking included
                "tokens": getattr(raw, "tokens", None),
                "metrics": ["bypass"], "bypass": 999, "doc_hash": _doc_hash(doc),
                "prompt_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "target_hash": hashlib.sha256(str(doc.get("reference", "")).encode("utf-8"))
                .hexdigest(),
                "served": how}, ensure_ascii=False) + "\n")
    (out / f"results_{ts}.json").write_text(json.dumps({
        "results": {task: {"alias": task, "bypass,none": 999, "bypass_stderr,none": "N/A"}},
        "group_subtasks": {task: []},
        "configs": {task: {"task": task, "output_type": "generate_until",
                           "generation_kwargs": gen, "num_fewshot": 0}},
        "versions": {task: 1.0}, "n-shot": {task: 0},
        "higher_is_better": {task: {"bypass": True}},
        "n-samples": {task: {"original": total, "effective": len(answers)}},
        "config": {"model": "served", "model_args": f"pretrained={rec['id']}", "batch_size": 1,
                   "batch_sizes": [], "device": None, "limit": None,
                   "random_seed": config.SEED, "gen_kwargs": {"max_gen_toks": s["max_tokens"]}},
        "chat_template": "the server's own", "date": time.time(),
        "served": {**how, **({"partial": {"answered": len(answers), "of": total}}
                             if len(answers) < total else {})}},
        indent=1), encoding="utf-8")


def lm_eval_model_args(rec: dict) -> str:
    """lm_eval's local-chat-completions, for the generative Standard tasks.
    Its retries wait 1, 1, 2, 4, 8 s and then 10 s each: as many as cover
    SERVED_RETRY_S. The key goes in the child's OPENAI_API_KEY (job_env),
    never on the command line"""
    retries = 5 + max(0, math.ceil((config.SERVED_RETRY_S - 16) / 10))
    return (f"model={rec['pin'].get('model') or rec['name']},"
            f"base_url={rec['base_url']}/chat/completions,"
            f"num_concurrent={max(1, config.SERVED_CONCURRENCY)},max_retries={retries},"
            f"timeout={config.SERVED_TIMEOUT_S}")


def job_env(env: dict, rec: dict) -> dict:
    """the lm_eval child's environment: the served model's key, and no other"""
    env = dict(env)
    env.pop("OPENAI_API_KEY", None)
    if rec.get("key"):
        env["OPENAI_API_KEY"] = rec["key"]
    return env


def cache_path(rec: dict, task: str) -> Path:
    """lm_eval's cache of the answers it has, per file served: a run the server
    stopped asks only the rest, and a different file starts again"""
    pin = hashlib.sha256(json.dumps(rec["pin"], sort_keys=True).encode()).hexdigest()[:12]
    d = model_dir(rec) / "served_cache" / pin
    d.mkdir(parents=True, exist_ok=True)
    return d / task


def adopt_lm_eval_results(task_out: Path, rec: dict) -> None:
    """lm_eval names a local-chat-completions run by the served model's name:
    the results say which board row it is (pretrained=served/…), and how it
    was served"""
    for f in task_out.rglob("results*.json"):
        try:
            blob = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if blob.get("served"):
            continue
        cfg = blob.setdefault("config", {})
        args = cfg.get("model_args") or ""
        if isinstance(args, dict):
            args = ",".join(f"{k}={v}" for k, v in args.items())
        cfg["model_args"] = f"pretrained={rec['id']}," + str(args)
        blob["chat_template"] = blob.get("chat_template") or "the server's own"
        blob["served"] = {**view(rec), "settings": {"backend": BACKEND,
                                                    "concurrency": config.SERVED_CONCURRENCY,
                                                    "gen_kwargs": cfg.get("gen_kwargs")}}
        f.write_text(json.dumps(blob, indent=1), encoding="utf-8")
