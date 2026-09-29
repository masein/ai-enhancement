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

12m.3: **a model from OpenRouter** is a served entry too — picked from AI
models' live list, asked at OpenRouter's address with AI models' key (never
kept with it), and pinned as the judge is: its dated version and its first
provider, sent with no fallbacks and checked at every run's start. Money:
`estimate` says what a run would cost before Start; a run that would pass the
month's AI limit doesn't start; each answer's cost, thinking included, is
counted into the month's AI spend as it lands (`Meter`), shown in the run's
progress, and the run stops before a question could pass the limit, keeping
what is finished. lm_eval's generative three reach it through `Relay`, which
holds each request against the same limit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import ai_models, config, db

PREFIX = "served/"
BACKEND = "local-chat-completions"           # lm_eval's, for the generative tasks
LOGLIK_LINE = ("Multiple-choice benchmarks need the model loaded here; this one is served "
               "elsewhere.")
CHANGED_LINE = ("The server now serves a different file than the one registered. Register it "
                "again if that's intended.")
KEPT_FOR_NEXT = " · the answers it gave are kept: the next run asks only the rest"
SUITES = ("everyday", "judged", "generative", "safety", "shared", "mobile")
THINKING = {"on": "on", "off": "off", "auto": "the model decides"}
PINNED = ("file", "size", "ctx", "build")
TS_FMT = "%Y-%m-%dT%H-%M-%S.000000"
# 12m.3: a model from OpenRouter — how its entry says so, and the job its
# runs' spend is counted under on AI models
OPENROUTER = "openrouter"
SPEND_JOB = "tests"


class ServerStopped(Exception):
    """the server stopped answering (or refused) at `done` of `total`"""
    def __init__(self, done: int, total: int, why: str = "", refused: str = ""):
        super().__init__(refused or f"the server stopped answering at {done} of {total}")
        self.done, self.total, self.why, self.task = done, total, why, ""


class LimitReached(ServerStopped):
    """12m.3: the next question, at its most, would pass this month's AI
    limit — the run stops before asking it, keeping what is finished"""
    def __init__(self, done: int, total: int, meter: "Meter"):
        super().__init__(done, total, "the monthly AI limit")
        self.meter = meter

    def __str__(self) -> str:
        return (f"stopped at {self.done} of {self.total}: the next question could pass this "
                f"month's {usd(ai_models.limit())} AI limit · {usd(self.meter.spent)} spent on "
                f"this run")


class _Retry(Exception):
    """worth asking again: no answer, a timeout, a 5xx, 408 or 429"""


def is_served(model_id: str) -> bool:
    return (model_id or "").startswith(PREFIX)


def is_openrouter(rec: dict | None) -> bool:
    """12m.3: a model from OpenRouter, not one a server of ours serves"""
    return bool(rec) and rec.get("via") == OPENROUTER


def _endpoint(rec: dict) -> tuple[str, str]:
    """(base URL, key) its questions go to. A model from OpenRouter's are AI
    models' — OPENROUTER_BASE_URL and OPENROUTER_API_KEY, read when asked and
    never kept with it"""
    if is_openrouter(rec):
        return config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY
    return rec["base_url"], rec.get("key", "")


def _pinned_route(rec: dict) -> dict:
    """what every request to a model from OpenRouter carries, as the judge's
    do: its pinned provider with no fallbacks, and the usage (with the cost)
    in the reply"""
    from . import ai_models
    return {"provider": ai_models.provider_prefs(rec["pin"]["provider"]),
            "usage": {"include": True}}


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


def _not_openrouter(base: str) -> None:
    """12m.3: OpenRouter's models are added as models from OpenRouter, where
    their cost is counted — never typed in as a server of ours"""
    host = re.sub(r"^https?://", "", (base or "").strip().lower()).split("/")[0]
    if host == "openrouter.ai" or host.endswith(".openrouter.ai") or (
            base or "").strip().rstrip("/") == config.OPENROUTER_BASE_URL:
        raise ValueError("That is OpenRouter: add its models under Test a model ▸ A model from "
                         "OpenRouter, where what they cost is counted")


def check(f: dict) -> dict:
    """Check before saving: what the server at this address reports"""
    _not_openrouter(f.get("base_url") or "")
    old = get(slug(f.get("name") or "")) or {}
    key = (f.get("key") or "").strip() or old.get("key", "")
    return probe(f.get("base_url") or "", key)


def _gguf_setups(text) -> list[dict]:
    from . import gguf
    return gguf.parse_setups(text)


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
    _not_openrouter(base)
    p = probe(base, key)
    rec = {"id": mid, "name": name, "base_url": base, "key": key,
           "based_on": (f.get("based_on") or "").strip(), "how": how, "thinking": thinking,
           "phone": bool(f.get("phone")),
           # 12f.3: its GGUF file on the server, for llama-perplexity's benchmarks
           "gguf_path": (f.get("gguf_path") or "").strip(),
           "gguf_flags": (f.get("gguf_flags") or "").strip(),
           # 12f.3 addendum: the setups its GGUF is measured in
           "gguf_setups": _gguf_setups(f.get("gguf_setups")),
           "pin": pin_of(p), "answered": p["answered"], "by": by, "at": time.time()}
    if rec["gguf_path"] and old.get("gguf_path") == rec["gguf_path"] and old.get("gguf_pin"):
        rec["gguf_pin"] = old["gguf_pin"]
    # 12o.1: registered again, it keeps the file someone said it serves
    if old.get("same_as"):
        rec["same_as"] = old["same_as"]
    db.served_put(rec)
    write_meta(rec)
    return public(rec)


def same_as_set(served_id: str, gguf_id: str, setup: str = "as-built") -> dict | None:
    """12o.1: "Same file as" — the GGUF entry, and the setup of it, a served
    entry serves: the board joins them without guessing. gguf_id "" guesses
    again (by file, then routing); "none" is never the same file. Kept on the
    served entry; the GGUF entry's own form sets the same thing from its side"""
    from . import gguf
    rec = get(served_id)
    if not rec:
        raise ValueError(f"no served model {served_id}")
    if not gguf_id:
        rec.pop("same_as", None)
    elif gguf_id == "none":
        rec["same_as"] = {"none": True}
    else:
        g = db.gguf_get(gguf_id)
        if not g:
            raise ValueError(f"no GGUF entry {gguf_id}")
        ids = [gguf.gb.AS_BUILT["id"]] + [x["id"] for x in g.get("setups") or []]
        if setup not in ids:
            raise ValueError(f"{g['name']} has no setup {setup}: its setups are "
                             + ", ".join(ids))
        if rec.get("gguf_path"):
            raise ValueError(f"{rec['name']} has a GGUF file of its own")
        rec["same_as"] = {"gguf": gguf_id, "setup": setup}
    db.served_put(rec)
    write_meta(rec)
    return rec.get("same_as")


def same_as_clear(gguf_id: str, setup: str) -> list[str]:
    """the GGUF entry's side: no served entry is this setup of it now"""
    out = []
    for rec in db.served_all():
        ln = rec.get("same_as") or {}
        if ln.get("gguf") == gguf_id and (ln.get("setup") or "as-built") == setup:
            rec.pop("same_as")
            db.served_put(rec)
            write_meta(rec)
            out.append(rec["id"])
    return out


def slug_openrouter(model_id: str) -> str:
    """"openai/gpt-6-luna" -> "served/openrouter-openai-gpt-6-luna" """
    return PREFIX + "openrouter-" + re.sub(r"[^a-z0-9._-]+", "-", (model_id or "").lower()
                                          ).strip("-.")[:80]


def register_openrouter(model_id: str, by: str) -> dict:
    """12m.3: a model from OpenRouter's list, kept as a served entry: its
    address is OpenRouter's and its key AI models' (neither asked here), and
    it is pinned as the judge is (ai_models.pin) — the dated version and the
    first provider, with that provider's prices. Whether it thinks is what
    OpenRouter says; not said, "the model decides". Added again, it is pinned
    again, to what OpenRouter lists now"""
    p = ai_models.pin((model_id or "").strip())
    m = ai_models.model(p["id"]) or {}
    name = p["name"].split(": ", 1)[-1]
    thinking = "off" if m.get("reasons") is False else "auto"
    rec = {"id": slug_openrouter(p["id"]), "name": name, "via": OPENROUTER,
           "base_url": config.OPENROUTER_BASE_URL, "key": "", "based_on": "",
           "maker": ai_models.maker(p["id"], p["name"]),
           "how": (f"OpenRouter · {p['version']} · on {p['provider_name']}"
                   + (f" ({p['precision']})" if p["precision"] != "unknown" else "")
                   + " · no fallbacks"),
           "thinking": thinking, "phone": False, "gguf_path": "", "gguf_flags": "",
           "gguf_setups": [],
           "pin": {"model": p["id"], "version": p["version"], "provider": p["provider"],
                   "provider_name": p["provider_name"], "precision": p["precision"],
                   "price_in": p["price_in"], "price_out": p["price_out"],
                   "ctx": m.get("context")},
           "answered": [], "by": by, "at": time.time()}
    old = get(rec["id"])
    if old and old.get("speed"):
        rec["speed"] = old["speed"]
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
            "gguf_setups": rec.get("gguf_setups") or [],
            # 12o.1: the GGUF entry, and its setup, someone said it serves
            "same_as": rec.get("same_as"),
            # 12i.4: measured by its last run — Test a model's time estimate
            "speed": rec.get("speed"),
            # 12m.3: a model from OpenRouter, its maker, and the MMLU-Pro
            # subset it sits unless a person clears it
            **({"via": OPENROUTER, "maker": rec.get("maker") or "",
                "subset": config.OPENROUTER_GEN_SUBSET} if is_openrouter(rec) else {})}


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
    if is_openrouter(rec):
        return check_openrouter(rec)
    try:
        now = pin_of(probe(rec["base_url"], rec.get("key", "")))
    except ValueError as e:
        return str(e)
    return CHANGED_LINE if any(now.get(k) != rec["pin"].get(k) for k in PINNED) else ""


AGAIN = "add it again under Test a model ▸ A model from OpenRouter if that's intended"


def check_openrouter(rec: dict) -> str:
    """12m.3: '' while OpenRouter still means what was pinned — the id on its
    list (asked afresh), at the dated version it was added with, its pinned
    provider listed and up — else why not, in one line. There is no key here
    to ask with: said too"""
    pin = rec["pin"]
    if not ai_models.has_key():
        return "OpenRouter has no key on this server (OPENROUTER_API_KEY), so it can't be asked"
    listed = ai_models.models(refresh=True)
    if not listed:
        return "OpenRouter didn't answer with its list of models: try again later"
    now = next((m for m in listed if m["id"] == pin["model"]), None)
    if now is None:
        return f"{pin['model']} is no longer on OpenRouter's list"
    if now["version"] != pin["version"]:
        return (f"{pin['model']} now points to {now['version']}, not the {pin['version']} it was "
                f"added with — {AGAIN}")
    provs = ai_models.providers(pin["model"])
    if provs is None:
        return f"OpenRouter didn't say which providers run {pin['model']} now: try again later"
    if not any(p["up"] and (p["tag"] or p["name"]) == pin["provider"] for p in provs):
        return (f"{pin['provider_name']} doesn't run {pin['model']} on OpenRouter now, and it is "
                f"pinned there with no fallbacks — {AGAIN}")
    return ""


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
    est = None
    if is_openrouter(rec):
        # 12m.3: a run that would pass this month's AI limit doesn't start
        est = estimate(rec, sub["suite"], json.loads(sub.get("tasks") or "[]"),
                       int(sub.get("subset") or 0))
        why = over_limit_line(est)
        if why:
            raise PreflightError(why)
    write_meta(rec)
    return {"kind": "instruct", "params": None, "vocab": None, "batch": 1, "need_gb": 0.0,
            "has_template": True, "remote_code": False, "archinfo": archinfo(rec),
            "kind_reason": "served elsewhere", "served": rec, "estimate": est}


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
    # 12m.3: OpenRouter has no chat template to switch: a model there thinks
    # as it does, and its budget is the thinking one when it does
    if rec["thinking"] in ("on", "off") and not is_openrouter(rec):
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
    # 12f.3 addendum: MTP's drafts, as llama-server's timings report them
    draft: dict | None = None
    # 12m.3: the reply's usage as reported (its cost, from OpenRouter), and
    # the provider that answered
    usage: dict | None = None
    provider: str = ""


def ask(rec: dict, text: str, s: dict) -> str:
    """one chat message, one reply — thinking the server split off put back in
    its tags, as a local run's text has it, so the board reads it the same"""
    system = [{"role": "system", "content": s["system"]}] if s.get("system") else []
    body = {"model": rec["pin"].get("model") or rec["name"],
            "messages": [*system, {"role": "user", "content": text}],
            **{k: v for k, v in s.items() if k != "system"}}
    if is_openrouter(rec):
        body.update(_pinned_route(rec))
    base, key = _endpoint(rec)
    try:
        st, raw = _http("POST", base + "/chat/completions", key, body,
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
    # llama-server says reasoning_content; OpenRouter says reasoning (12m.3)
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    out = Answer(f"<think>\n{think}\n</think>\n\n{text}" if think else text)
    used = (reply.get("usage") or {}).get("completion_tokens")
    out.tokens = int(used) if isinstance(used, (int, float)) and used >= 0 else None
    out.usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else None
    out.provider = str(reply.get("provider") or "")
    t = reply.get("timings") or {}
    if isinstance(t.get("draft_n"), (int, float)) and t["draft_n"] > 0:
        out.draft = {"n": int(t["draft_n"]), "accepted": int(t.get("draft_n_accepted") or 0)}
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


def concurrency(rec: dict) -> int:
    """questions at a time: llama-server's slots (SERVED_CONCURRENCY), or
    (12m.3) OpenRouter's, as the AI jobs ask it"""
    return max(1, config.OPENROUTER_CONCURRENCY if is_openrouter(rec)
               else config.SERVED_CONCURRENCY)


def answer_task(rec: dict, task: str, docs: list[dict], task_out: Path, s: dict,
                everyday: bool, on_progress=None, canceled=lambda: False,
                meter: "Meter | None" = None) -> int:
    """ask every doc and write lm_eval's files: 0, or -15 when canceled — or
    ServerStopped after writing the answers before the first it didn't get.
    `on_progress(done, total, seconds an answer)`. 12m.3: with a `meter`,
    each question is held against the month's AI limit before it is asked
    and counted when answered; a question that could pass the limit stops
    the run with LimitReached, and every answer already paid for is kept"""
    total = len(docs)
    answers: dict[int, str] = {}
    lock = threading.Lock()
    halt: list[ServerStopped] = []
    t0 = time.time()

    def one(i: int) -> None:
        if halt or canceled():
            return
        text = prompt_of(docs[i], everyday)
        held = meter.worst([s.get("system") or "", text], s["max_tokens"]) if meter else 0.0
        if meter and not meter.room(held):
            with lock:
                halt.append(LimitReached(0, total, meter))
            return
        try:
            a = _ask_patiently(rec, text, s)
        except ServerStopped as e:
            if meter:
                meter.release(held)
            with lock:
                halt.append(e)
            return
        except ValueError as e:                 # a 4xx: this question, refused — said, not retried
            if meter:
                meter.release(held)
            with lock:
                halt.append(ServerStopped(0, 0, str(e), refused=(
                    f"the server refused question {i + 1} of {total}: {e}")))
            return
        if meter:
            meter.count(held, a.usage, a.provider)
        with lock:
            answers[i] = a
            done = len(answers)
        if on_progress:
            on_progress(done, total, (time.time() - t0) / done)

    with ThreadPoolExecutor(max_workers=concurrency(rec)) as pool:
        list(pool.map(one, range(total)))
    if answers:
        record_speed(rec["id"], (time.time() - t0) / len(answers), len(answers))
    # what is finished is kept, in order: a stop keeps the answers before the
    # first gap. 12m.3: at the AI limit, every answer — each is paid for
    at_limit = any(isinstance(e, LimitReached) for e in halt)
    gap = total if at_limit else next((i for i in range(total) if i not in answers), total)
    kept = {i: answers[i] for i in sorted(answers) if i < gap}
    if kept or not halt:
        _write(rec, task, docs, kept, task_out, s, total, everyday)
    if canceled():
        return -15
    if halt:
        e = next((x for x in halt if isinstance(x, LimitReached)), halt[0])
        if not isinstance(e, LimitReached) and not str(e).startswith("the server refused"):
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
                # 12f.3 addendum: MTP's drafted and accepted tokens, when reported
                "draft": getattr(raw, "draft", None),
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


def lm_eval_model_args(rec: dict, relay: "Relay | None" = None) -> str:
    """lm_eval's local-chat-completions, for the generative Standard tasks.
    Its retries wait 1, 1, 2, 4, 8 s and then 10 s each: as many as cover
    SERVED_RETRY_S. The key goes in the child's OPENAI_API_KEY (job_env),
    never on the command line. 12m.3: a model from OpenRouter is asked
    through its run's `relay`, which holds the key"""
    retries = 5 + max(0, math.ceil((config.SERVED_RETRY_S - 16) / 10))
    return (f"model={rec['pin'].get('model') or rec['name']},"
            f"base_url={relay.base if relay else rec['base_url']}/chat/completions,"
            f"num_concurrent={concurrency(rec)},max_retries={retries},"
            f"timeout={config.SERVED_TIMEOUT_S}")


def job_env(env: dict, rec: dict) -> dict:
    """the lm_eval child's environment: the served model's key, and no other.
    12m.3: a model from OpenRouter's child gets no key at all — its relay
    adds it — and not OpenRouter's either"""
    env = dict(env)
    env.pop("OPENAI_API_KEY", None)
    if is_openrouter(rec):
        env.pop("OPENROUTER_API_KEY", None)
    elif rec.get("key"):
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


# ---------------------------------------------------------------------------
# 12m.3: a model from OpenRouter — what a run would cost, and what it costs
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4
# IFEval, MMLU-Pro and MATH-500 come with lm_eval — its datasets, fetched on a
# first run, not on this server before — so their size is said here: how many
# questions, and about how many tokens each prompt is as lm_eval poses it
# (MMLU-Pro with its five worked examples). As rough as the rule above
# 12n.2: and GPQA Diamond's 198, gated — its question, four options and the
# "think step by step" line
GEN_ITEMS = {"ifeval": 541, "hendrycks_math500": 500, "gpqa_diamond_cot_zeroshot": 198}
GEN_PROMPT_TOKENS = {"ifeval": 90, "mmlu_pro": 2300, "hendrycks_math500": 110,
                     "gpqa_diamond_cot_zeroshot": 330}


def tokens_of(text: str) -> int:
    """a text's length in tokens, by the rough rule: one for every four characters"""
    return -(-len(text or "") // CHARS_PER_TOKEN)


def usd(v: float) -> str:
    """"$0.40", "$1,240.00" """
    return f"${v:,.2f}"


def about(v: float) -> str:
    """"about $0.40" — an estimate says it is one"""
    return "under $0.01" if 0 < v < 0.005 else f"about {usd(v)}"


def _run_tasks(rec: dict, suite: str, tasks: list[str] | None) -> list[str]:
    """the tasks a run of `suite` asks: the chosen ones, less those answered
    already (the run skips them too)"""
    from . import runner
    todo = config.tasks_for_suite(suite)
    chosen = [t for t in (tasks or []) if t in todo]
    return [t for t in chosen or todo if suite == "everyday" or not runner._task_done(
        model_dir(rec) / f"{t}_{config.NFEWSHOT.get(t, 0)}shot", t)]


def estimate(rec: dict, suite: str, tasks: list[str] | None = None, subset: int = 0) -> dict:
    """What a run of `suite` would cost, about. The rule, said once:

    - each question's prompt as the run sends it, at one token for every four
      characters (tokens_of; the chat template's few tokens left out);
    - each answer at its whole budget: the max tokens the settings function
      gives it — settings_for (everyday.run_settings, or the exam's), which
      gives a model that thinks the thinking budgets — and for IFEval,
      MMLU-Pro and MATH-500 gen_thinking's budget. Their prompts are
      lm_eval's, not on this server, so each is GEN_PROMPT_TOKENS long;
    - at the pinned provider's prices, per million tokens.

    Only what the run would ask: the Everyday questions it has no answer to,
    the tasks not answered already. An answer usually stops well short of
    its budget, so a run usually costs less than this.
    {n, tokens_in, tokens_out, usd, line: "about $0.40"}"""
    from . import runner
    runner._scripts()
    meta = {"archinfo": archinfo(rec)}
    n = tin = tout = 0
    for task in _run_tasks(rec, suite, tasks):
        # 12n.2: GPQA's chain of thought is asked as the generative three are
        if suite == "generative" or task == config.GPQA_COT:
            k = (min(subset, sum(config.MMLU_PRO_SUBJECTS.values())) if task == "mmlu_pro"
                 and subset > 0 else sum(config.MMLU_PRO_SUBJECTS.values())
                 if task == "mmlu_pro" else GEN_ITEMS.get(task, 0))
            n += k
            tin += k * GEN_PROMPT_TOKENS.get(task, 0)
            tout += k * runner.gen_thinking({}, meta)["budget"]
            continue
        everyday = suite in ("everyday", "safety", "shared", "mobile")
        if suite == "everyday":
            import everyday as _ev
            docs = _ev.unanswered(model_dir(rec))
        elif suite == "safety":
            import trust_safety as _ts
            docs = _ts.load(task)
        elif task == config.SIMPLEQA_TASK:
            import simpleqa as _sq                  # 12n.2: its questions, as the run sends them
            docs = [{"id": q["id"], "prompt": q["prompt"]} for q in _sq.load()]
        elif task in config.MAB_TASKS:
            import mobileaibench as _mab            # 12o.3: its prompts, and its system line
            docs = [{"id": q["id"], "prompt": q["prompt"]} for q in _mab.load(task)]
            tin += len(docs) * tokens_of(_mab.SYSTEM[task])
        else:
            path = config.JUDGED_TASKS_DIR / f"{task}.jsonl"
            docs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
                    if x.strip()] if path.exists() else []
        st = settings_for(rec, meta, everyday)
        n += len(docs)
        tin += sum(tokens_of(st.get("system") or "") + tokens_of(prompt_of(d, everyday))
                   for d in docs)
        tout += len(docs) * st["max_tokens"]
    cost = ai_models.cost(rec["pin"], tin, tout)
    return {"n": n, "tokens_in": tin, "tokens_out": tout, "usd": round(cost, 2),
            "line": about(cost)}


def over_limit_line(est: dict) -> str:
    """'' when a run of about `est` fits in what is left of this month's AI
    limit; else why it doesn't start, in one line naming both"""
    cap, month = ai_models.limit(), db.spend_this_month()
    left = max(0.0, cap - month)
    if not est["n"] or (est["usd"] <= left and left > 0):
        return ""
    return (f"This run could cost {est['line']}, more than the {usd(left)} left of this month's "
            f"{usd(cap)} AI limit — test fewer questions, or raise the limit on AI models.")


def reply_cost(pin: dict, usage: dict) -> tuple[int, int, float]:
    """(tokens in, tokens out, dollars) of one reply: its cost as OpenRouter
    reports it (usage.cost), else its tokens at the pinned prices. Tokens out
    count the thinking: OpenRouter counts reasoning tokens in
    completion_tokens (completion_tokens_details.reasoning_tokens says how
    many), and a reply that reports more reasoning tokens than completion
    tokens has them added — thinking is paid for either way"""
    tin = int(usage.get("prompt_tokens") or 0)
    comp = int(usage.get("completion_tokens") or 0)
    think = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    tout = comp if think <= comp else comp + think
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or cost < 0:
        cost = ai_models.cost(pin, tin, tout)
    return tin, tout, float(cost)


class Meter:
    """One run's spend on OpenRouter. Each answer's cost goes into the
    month's AI spend (the ai_spend ledger the judge's goes into, under the
    job "tests", this run's id as its batch) as it lands, and into the run's
    running total. Before each question, room() holds back what it could
    cost at most (worst) and says no when that would pass the monthly limit
    — from then on it says no to every question, and the run stops"""

    def __init__(self, rec: dict, sid: int, estimate_usd: float = 0.0):
        self.rec, self.sid, self.estimate = rec, sid, float(estimate_usd or 0.0)
        self.batch = f"run-{sid}"
        self.spent = db.spend_of_batch(self.batch)      # a run the service restarted
        self.held = 0.0
        self.stopped = False
        self._lock = threading.Lock()

    def worst(self, texts: list[str], max_tokens: int) -> float:
        """what one question could cost at most: its prompt at twice the
        rough rule (tokens_of) and its whole answer budget"""
        return ai_models.cost(self.rec["pin"], 2 * sum(tokens_of(t) for t in texts),
                              int(max_tokens or 0))

    def room(self, worst: float) -> bool:
        with self._lock:
            if self.stopped or db.spend_this_month() + self.held + worst > ai_models.limit():
                self.stopped = True
                return False
            self.held += worst
            return True

    def release(self, worst: float) -> None:
        """a question that got no answer: what was held for it is let go"""
        with self._lock:
            self.held = max(0.0, self.held - worst)

    def count(self, worst: float, usage: dict | None, provider: str = "") -> float:
        """an answer's cost, counted: as reported, or — with no usage at all —
        the most it could have cost"""
        pin = self.rec["pin"]
        tin, tout, cost = reply_cost(pin, usage) if usage else (0, 0, worst)
        with self._lock:
            db.spend_add(SPEND_JOB, pin["model"], provider or pin.get("provider_name") or "",
                         tin, tout, cost, self.batch)
            self.held = max(0.0, self.held - worst)
            self.spent += cost
        return cost

    def line(self) -> str:
        """"$0.12 of about $0.40 so far · limit $5.00, $4.60 left" """
        cap = ai_models.limit()
        left = max(0.0, cap - db.spend_this_month())
        return (f"{usd(self.spent)} of {about(self.estimate)} so far · limit {usd(cap)}, "
                f"{usd(left)} left")


class Relay:
    """lm_eval's local-chat-completions asks a model from OpenRouter the
    generative three through this: a relay on 127.0.0.1, under a path no one
    else knows, for the length of one run. Each request is held against the
    month's AI limit before it goes (Meter.room), goes to OpenRouter with the
    key and the pinned provider with no fallbacks, and its cost is counted
    when the answer comes back. At the limit it answers 402 and sends nothing
    on, and the runner stops lm_eval at its next look. The lm_eval child never
    sees the key"""

    def __init__(self, rec: dict, meter: Meter):
        self.rec, self.meter = rec, meter
        self.nonce = secrets.token_hex(12)
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):                      # nothing of it in any log
                pass

            def do_POST(self):
                code, body = relay.handle(self.path, self.rfile.read(
                    int(self.headers.get("content-length") or 0)))
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{self.nonce}/v1"

    def __enter__(self) -> "Relay":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, path: str, raw: bytes) -> tuple[int, dict]:
        if path != f"/{self.nonce}/v1/chat/completions":
            return 404, {"error": {"message": "not here"}}
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return 400, {"error": {"message": "not JSON"}}
        texts = [str(m.get("content") or "") for m in body.get("messages") or []]
        held = self.meter.worst(texts, int(body.get("max_tokens") or config.GEN_MAX_GEN_TOKS))
        if not self.meter.room(held):
            return 402, {"error": {"message": "stopped at this month's AI limit"}}
        out = {**body, "model": self.rec["pin"]["model"], **_pinned_route(self.rec)}
        base, key = _endpoint(self.rec)
        try:
            st, got = _http("POST", base + "/chat/completions", key, out,
                            timeout=config.SERVED_TIMEOUT_S)
            reply = json.loads(got) if st == 200 else None
        except Exception as e:                              # noqa: BLE001 — lm_eval retries
            self.meter.release(held)
            return 502, {"error": {"message": str(getattr(e, "reason", e))[:200]}}
        if not isinstance(reply, dict) or not reply.get("choices"):
            self.meter.release(held)
            try:
                said = json.loads(got)
            except ValueError:
                said = {"error": {"message": got[:200].decode("utf-8", "replace")}}
            return (st if st != 200 else 502), said
        self.meter.count(held, reply.get("usage") if isinstance(reply.get("usage"), dict)
                         else None, str(reply.get("provider") or ""))
        for c in reply["choices"]:                          # a reply that ran out thinking
            msg = c.get("message") or {}
            if msg.get("content") is None:
                msg["content"] = ""
        return 200, reply

