"""12i.1: which AI model does each job — the AI models page.

Four jobs, one model each: the judge, the question writer, the training-data
writer and the checker. Each is either the local model on this server, or a
model on OpenRouter, pinned so it can't change quietly:

- the exact id OpenRouter lists, and its dated version (its canonical slug) —
  a request is refused when the id has since been moved to another version;
- the provider: the first one OpenRouter lists for the model, sent with
  `allow_fallbacks: false`, because another provider can run another
  precision, and that shifts marks.

Before a job has a model here, it keeps the one its environment names
(LLM_*, EXAM_*, JUDGE_*), as before 12i.1. OpenRouter's key lives in the
server's environment (OPENROUTER_API_KEY), never in the page or the repo; with
none, only Local is offered, and nothing here calls OpenRouter.

AI spend is counted per request, and at the monthly limit AI jobs wait with a
plain message — they never fall back to another model.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from . import config, db

# the jobs, in the page's order: what each does today, the suggested model and
# one line why, and the environment identity it replaces (llm.ROLES)
JOBS = {
    "judge": {"label": "Judge", "role": "judge",
              "does": "marks Knowledge exam answers 0–4, and the few Everyday questions with a "
                      "judge check",
              "suggested": "deepseek/deepseek-v4.1-flash",
              "why": "cheap and strong as a judge when given a reference answer"},
    "writer": {"label": "Question writer", "role": "exam",
               "does": "writes new questions for the Knowledge exam and Everyday tasks",
               "suggested": "z-ai/glm-5.3",
               "why": "open weights, and strong at writing to a format"},
    "data": {"label": "Training-data writer", "role": "llm",
             "does": "writes Improve's documents and chat examples",
             "suggested": "z-ai/glm-5.3",
             "why": "open weights: its output can become training data"},
    "checker": {"label": "Checker", "role": "checker",
                "does": "answers new questions without seeing the reference, to test them",
                "suggested": "openai/gpt-6-luna",
                "why": "another family from the writer, so it doesn't share its blind spots"},
}
ROLE_JOB = {j["role"]: k for k, j in JOBS.items()}
# 16.5: what each does while the Knowledge exam is switched off
DOES_NO_EXAM = {"judge": "marks the Everyday questions with a judge check, 0–4",
                "writer": "writes new questions for Everyday tasks"}


def does(job: str) -> str:
    """a job's line, in what the board runs now"""
    if not config.KNOWLEDGE_EXAM and job in DOES_NO_EXAM:
        return DOES_NO_EXAM[job]
    return JOBS[job]["does"]
LOCAL = "local"
# the model families whose terms restrict training on their output: easy to
# edit. Open-weight families (GLM, DeepSeek, Qwen) avoid the question
RESTRICTED_TRAINING = {"openai", "google", "anthropic"}
# a family by one name, whichever org spelling it comes under (OpenRouter's
# or the Hub's)
FAMILY_ALIASES = {"z-ai": "glm", "zai-org": "glm", "thudm": "glm", "deepseek-ai": "deepseek",
                  "qwen": "qwen", "meta-llama": "llama", "mistralai": "mistral",
                  "google": "google", "openai": "openai", "anthropic": "anthropic",
                  "moonshotai": "kimi", "x-ai": "grok", "microsoft": "microsoft",
                  "nvidia": "nvidia", "ibm-granite": "granite", "liquidai": "liquid",
                  "huggingfacetb": "smollm", "tencent": "tencent"}
FAMILY_NAMES = {"glm": "GLM", "deepseek": "DeepSeek", "qwen": "Qwen", "llama": "Llama",
                "mistral": "Mistral", "google": "Google", "openai": "OpenAI",
                "anthropic": "Anthropic", "kimi": "Kimi", "grok": "Grok", "smollm": "SmolLM"}
CACHE_S = 24 * 3600                      # the model list is fetched once a day
# 16c: a refresh that failed is not tried again on every call for this long —
# drifted() asks for the list before each request of a batch and on every poll
RETRY_S = 300
_LIST_FAILED = {"at": 0.0, "asking": False}
_LIST_LOCK = threading.Lock()
_DATED = re.compile(r"-(20\d{6})$")


def has_key() -> bool:
    return bool(config.OPENROUTER_API_KEY)


def family(model_id: str) -> str:
    """"z-ai/glm-5.3" -> "glm"; "Qwen/Qwen3-1.7B" -> "qwen" """
    org = (model_id or "").split("/")[0].lower()
    return FAMILY_ALIASES.get(org, org)


def family_name(fam: str) -> str:
    return FAMILY_NAMES.get(fam, fam[:1].upper() + fam[1:])


# ---------------------------------------------------------------------------
# OpenRouter's lists: the models (cached a day) and a model's providers
# ---------------------------------------------------------------------------

def _cache_path() -> Path:
    return config.BENCH_ROOT / "ai" / "openrouter_models.json"


def _get(path: str) -> dict:
    from . import llm
    _, raw = llm._http("GET", config.OPENROUTER_BASE_URL + path, _headers(), timeout=20)
    return json.loads(raw)


def _headers() -> dict:
    h = {"content-type": "application/json", "x-title": "evalboard"}
    if config.OPENROUTER_API_KEY:
        h["authorization"] = f"Bearer {config.OPENROUTER_API_KEY}"
    return h


def per_million(price) -> float | None:
    """OpenRouter's per-token price (a string) as $ per million tokens"""
    try:
        v = float(price)
    except (TypeError, ValueError):
        return None
    return round(v * 1e6, 4) if v >= 0 else None


def _fetch_models() -> list[dict]:
    """OpenRouter's list, asked now, kept on disk — or LLMError and friends"""
    data = _get("/models").get("data") or []
    out = []
    for m in data:
        arch = m.get("architecture") or {}
        if "text" not in (arch.get("output_modalities") or ["text"]) \
                or "text" not in (arch.get("input_modalities") or ["text"]):
            continue
        pin, pout = (per_million((m.get("pricing") or {}).get(k)) for k in ("prompt", "completion"))
        if pin is None or pout is None or str(m.get("id", "")).endswith(":free"):
            continue
        params = m.get("supported_parameters")
        out.append({"id": m["id"], "name": m.get("name") or m["id"],
                    "version": m.get("canonical_slug") or m["id"],
                    "price_in": pin, "price_out": pout,
                    "context": m.get("context_length"),
                    # 12m.3: whether it thinks before it answers, when OpenRouter says
                    "reasons": ("reasoning" in params) if isinstance(params, list) else None,
                    # 17f: whether it takes a JSON schema (OpenAI's structured outputs)
                    "structured": (("structured_outputs" in params)
                                   if isinstance(params, list) else None)})
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": time.time(), "models": out}), encoding="utf-8")
    return out


def _refresh_models() -> list[dict] | None:
    """one refresh; a failure is remembered for RETRY_S (16c)"""
    try:
        out = _fetch_models()
    except BaseException:                           # noqa: BLE001 — the day-old list stands
        with _LIST_LOCK:
            _LIST_FAILED["at"] = time.time()
        return None
    with _LIST_LOCK:
        _LIST_FAILED["at"] = 0.0
    return out


def _refresh_in_background() -> None:
    def go():
        try:
            _refresh_models()
        finally:
            with _LIST_LOCK:
                _LIST_FAILED["asking"] = False
    with _LIST_LOCK:
        if _LIST_FAILED.get("asking") or time.time() - _LIST_FAILED["at"] < RETRY_S:
            return
        _LIST_FAILED["asking"] = True
    threading.Thread(target=go, name="openrouter-models", daemon=True).start()


def models(refresh: bool = False, wait: bool = True) -> list[dict]:
    """OpenRouter's text models — {id, name, version, price_in, price_out,
    context} — fetched live and kept a day. [] with no key: nothing is asked.
    16c: a day-old list is returned as it is and asked again in the
    background — a page, a batch's requests and its status polls never wait
    on OpenRouter for it — and a background refresh that failed isn't tried
    again for RETRY_S. Only with no list at all, and `wait`, is it asked here
    (a person choosing a model)"""
    if not has_key():
        return []
    try:
        cached = json.loads(_cache_path().read_text(encoding="utf-8"))
        if not isinstance(cached.get("models"), list):
            cached = None
    except (OSError, ValueError, AttributeError):
        cached = None
    if refresh:
        got = _refresh_models()
        return got if got is not None else (cached or {}).get("models") or []
    if cached and time.time() - cached.get("at", 0) < CACHE_S:
        return cached["models"]
    if cached or not wait:
        _refresh_in_background()
        return (cached or {}).get("models") or []
    # a person choosing a model, with no list at all: asked now, whatever
    # failed a moment ago — they are waiting for this answer
    return _refresh_models() or []


def model(model_id: str, wait: bool = True) -> dict | None:
    return next((m for m in models(wait=wait) if m["id"] == model_id), None)


def providers(model_id: str) -> list[dict] | None:
    """Every provider OpenRouter lists for the model, in its order — {name,
    tag, precision, price_in, price_out, up} — or None when it can't be asked"""
    try:
        eps = (_get(f"/models/{model_id}/endpoints").get("data") or {}).get("endpoints") or []
    except Exception:                               # noqa: BLE001
        return None
    out = []
    for e in eps:
        pr = e.get("pricing") or {}
        out.append({"name": e.get("provider_name") or "", "tag": e.get("tag") or "",
                    "precision": e.get("quantization") or "unknown",
                    "price_in": per_million(pr.get("prompt")),
                    "price_out": per_million(pr.get("completion")),
                    "up": e.get("status", 0) in (0, None)})
    return out


# 12p.1: every request to OpenRouter excludes the providers that may store or
# train on prompts — our prompts are the test. Set here, on each request
# (`provider.data_collection: "deny"`); the account-wide switch at
# openrouter.ai/settings/privacy is the same rule, and worth turning on too
DATA_COLLECTION = "deny"


def provider_prefs(provider: str | None = None) -> dict:
    """what every request to OpenRouter carries: the pinned provider with no
    fallbacks, and none that may train on the prompt"""
    return {**({"order": [provider], "allow_fallbacks": False} if provider else {}),
            "data_collection": DATA_COLLECTION}


PROBE = "ping"
PROBE_TIMEOUT_S = 30
# 16c: never the key or a header in anything said back
_SECRET = re.compile(r"(?i)bearer\s+\S+|sk-or-[\w-]+|authorization\S*")


def _error_of(text: str) -> tuple[str, dict]:
    """OpenRouter's own message and metadata from an error body — {"error":
    {"code", "message", "metadata"}} — wherever in `text` the body starts"""
    text = str(text or "")
    i = text.find("{")
    try:
        e = json.loads(text[i:]).get("error") if i >= 0 else None
    except (ValueError, AttributeError):
        e = None
    if not isinstance(e, dict):
        return "", {}
    meta = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
    return _SECRET.sub("…", str(e.get("message") or "")).strip()[:200], meta


def refusal(status: int | None, text: str) -> tuple[str, str]:
    """(kind, words): what OpenRouter said when it didn't answer, in plain
    words — from its status and its error body (16c). Kinds: limit (the key's
    own limit, or the account's credit), data (no provider takes a prompt it
    may not store — only when OpenRouter says so), key, rate, unreached,
    down, refused"""
    msg, meta = _error_of(text)
    low = f"{msg} {json.dumps(meta)}".lower()
    if status is None:
        if "no response within" in str(text) or "timed out" in str(text).lower():
            return "unreached", "OpenRouter didn’t answer in time"
        return "unreached", "OpenRouter couldn’t be reached"
    # 17g: a key's spend cap, the account's credit — never the answer's
    if status == 402 or "key limit" in low or "openrouter_key_limit" in low \
            or "insufficient credits" in low or re.search(r"spend(ing)? (cap|limit)", low) \
            or "credit" in low:
        return "limit", ("OpenRouter refused the key: it has reached its own spending limit, or "
                         "the account is out of credit. That is fixed on OpenRouter’s side: "
                         "raise the key’s limit or add credit there")
    if "data policy" in low or "data_collection" in low or "data collection" in low:
        return "data", "no provider running it takes a prompt it may not store or train on"
    if status == 401:
        return "key", ("OpenRouter refused the key (HTTP 401): it has been disabled or "
                       "replaced. The server’s OPENROUTER_API_KEY needs a look")
    if status == 429:
        return "rate", "OpenRouter is limiting this key’s requests (HTTP 429)"
    if status in (408, 504):
        return "unreached", "OpenRouter didn’t answer in time"
    # 17g: the provider's own failure, passed on as a 400 — never the answer's
    if status >= 500 or "provider returned error" in low:
        return "down", f"OpenRouter or the provider failed (HTTP {status})"
    return "refused", f"OpenRouter refused it (HTTP {status})" + (f": {msg}" if msg else "")


def probe(model_id: str, prov: dict, job: str = "pin") -> tuple[bool, str, str]:
    """12p.1: the provider takes a request that may not be stored or trained
    on — one token, asked when a model is pinned. 16c: (ok, kind, words), the
    words what OpenRouter said when it didn't (refusal). 17b: paid, so its
    cost counts against the month's limit, as `job`'s"""
    from . import llm
    body = {"model": model_id, "max_tokens": 1,
            "messages": [{"role": "user", "content": PROBE}],
            "provider": provider_prefs(prov.get("tag") or prov.get("name")),
            "usage": {"include": True}}
    try:
        status, raw = llm._http("POST", config.OPENROUTER_BASE_URL + "/chat/completions",
                                _headers(), json.dumps(body).encode(), timeout=PROBE_TIMEOUT_S)
    except llm.LLMError as e:
        return (False, *refusal(e.status, str(e)))
    except Exception as e:                          # noqa: BLE001 — not reached: said so
        return (False, *refusal(None, repr(e)))
    if status == 200:
        _probe_spent(job, model_id, prov, raw)
        return True, "", ""
    return (False, *refusal(status, raw.decode("utf-8", "replace") if isinstance(raw, bytes)
                            else str(raw)))


def _probe_spent(job: str, model_id: str, prov: dict, raw) -> None:
    """17b: a probe's cost in the month's spend — OpenRouter's figure, else
    its tokens (a short chat's, where it gives none) at the provider's price"""
    from . import db
    try:
        u = json.loads(raw).get("usage") or {}
    except (ValueError, TypeError, AttributeError):
        u = {}
    tin, tout = int(u.get("prompt_tokens") or 8), int(u.get("completion_tokens") or 1)
    usd = u.get("cost")
    usd = float(usd) if isinstance(usd, (int, float)) else cost(prov, tin, tout)
    db.spend_add(job, model_id, prov.get("name") or "", tin, tout, usd, "probe")


def first_provider(model_id: str) -> dict | None:
    """The first provider OpenRouter lists for the model — the one pinned:
    {name, tag, precision, price_in, price_out}. Never one that is down"""
    return next(({k: v for k, v in p.items() if k != "up"} for p in providers(model_id) or []
                 if p["up"]), None)


# 12m.3: who makes a model, by its id's organisation — a picker groups the
# models tested through OpenRouter under these
MAKERS = {"openai": "OpenAI", "google": "Google", "anthropic": "Anthropic", "x-ai": "xAI",
          "meta-llama": "Meta", "mistralai": "Mistral", "deepseek": "DeepSeek", "qwen": "Qwen",
          "z-ai": "Z.ai", "moonshotai": "Moonshot AI", "microsoft": "Microsoft",
          "nvidia": "NVIDIA", "amazon": "Amazon", "cohere": "Cohere", "minimax": "MiniMax"}


def maker(model_id: str, name: str = "") -> str:
    """"openai/gpt-6-luna" -> "OpenAI"; an organisation not listed is named
    as OpenRouter names it ("Acme: Painter" -> "Acme")"""
    org = (model_id or "").split("/")[0].lower()
    if org in MAKERS:
        return MAKERS[org]
    return name.split(": ", 1)[0] if ": " in (name or "") else org[:1].upper() + org[1:]


# ---------------------------------------------------------------------------
# each job's model
# ---------------------------------------------------------------------------

def choice(job: str) -> dict | None:
    """what the AI models page chose for a job — {kind: local} or {kind:
    openrouter, id, version, name, provider, …} — or None: the environment's"""
    return db.ai_get("job:" + job)


def local_model() -> str:
    """the id the local server serves, as the environment names it for any
    job already on it (LOCAL_MODEL overrides)"""
    import os
    for pv, mv in (("JUDGE_PROVIDER", "JUDGE_MODEL"), ("LLM_PROVIDER", "LLM_MODEL"),
                   ("EXAM_PROVIDER", "EXAM_MODEL")):
        if getattr(config, pv, "") == LOCAL and getattr(config, mv, ""):
            return os.environ.get("LOCAL_MODEL", "") or getattr(config, mv)
    return os.environ.get("LOCAL_MODEL", "")


def effective(role: str) -> tuple[str, str, str] | None:
    """(provider, model, key) the AI models page chose for the role's job, or
    None: the environment's"""
    job = ROLE_JOB.get(role)
    c = choice(job) if job else None
    if not c:
        return None
    if c.get("kind") == "openrouter":
        return ("openrouter", c["id"], config.OPENROUTER_API_KEY)
    return (LOCAL, local_model(), "")


def _local_in_use() -> bool:
    """a job runs on the local model — only then is its server asked its name"""
    if any(getattr(config, pv, "") == LOCAL for pv in
           ("JUDGE_PROVIDER", "LLM_PROVIDER", "EXAM_PROVIDER", "CHECKER_PROVIDER")):
        return True
    return any((choice(j) or {}).get("kind") == LOCAL for j in JOBS)


def local_name(ask: bool = False) -> str:
    """"gemma" — the local model by its weights, when the server says them.
    12i.3: with `ask`, a page that shows the name asks the server itself when
    no client has yet; a hot path (a judge's identity, its health probe) never
    does — it names what is already known"""
    from . import llm
    served = {}
    for v in llm._SERVED.values():
        served.update(v)
    if ask and not any(served.values()) and _local_in_use():
        served = llm.served_weights()
    root = next((w for w in served.values() if w), "")
    name = (root.rstrip("/").split("/")[-1] or "").split("-")[0].lower()
    return name or "the model"


def local_label(ask: bool = False) -> str:
    """"Local (gemma on this server)" — the local model wherever it is named"""
    return f"Local ({local_name(ask)} on this server)"


def _env(job: str) -> tuple[str, str]:
    """(provider, model) the environment set the job up with — the judge's is
    judge.py's (JUDGE_MODEL=stub is its overlap judge), the others an LLM role's"""
    if job == "judge":
        import judge as _judge
        ident = _judge.identity()
        return ident["provider"], ident["model"]
    from . import llm
    p, m, _ = llm.identity(JOBS[job]["role"])
    return p, m


def label(job: str) -> str:
    """the job's model in words: "DeepSeek V4.1 Flash", "Local (gemma on this server)" """
    c = choice(job)
    if c and c.get("kind") == "openrouter":
        return (c.get("name") or c["id"]).split(": ", 1)[-1]
    # before the environment's: a judge chosen Local names itself through
    # this (judge.identity), and must not ask back (12i.3)
    if c and c.get("kind") == LOCAL:
        return local_label(ask=True)
    p, m = _env(job)
    if p == LOCAL:
        return local_label(ask=True)
    return f"{p} {m}".strip() if p else "none"


def is_local(job: str) -> bool:
    """12z D1: the job runs on the local server — the one the judge's health
    probe asks, so when that is down this job can't be done either"""
    c = choice(job)
    if c:
        return c.get("kind") == LOCAL
    return _env(job)[0] == LOCAL


def pin(model_id: str, job: str = "pin") -> dict:
    """An OpenRouter model, pinned: the id, its dated version and its first
    provider, with that provider's prices — or ValueError, in one line. The
    judge is pinned by this, and (12m.3) a model tested through OpenRouter.
    17b: each provider asked is a paid token, counted as `job`'s"""
    if not has_key():
        raise ValueError("OpenRouter has no key on this server (OPENROUTER_API_KEY)")
    m = model(model_id)
    if not m:
        raise ValueError(f"{model_id} is not one of OpenRouter's text models")
    up = [{k: v for k, v in p.items() if k != "up"} for p in providers(model_id) or [] if p["up"]]
    if not up:
        raise ValueError(f"OpenRouter lists no provider running {model_id} now")
    # 12p.1: the first that takes a prompt it may not store or train on. 16c:
    # what OpenRouter said when one didn't — a refusal that isn't the
    # provider's own (the key's limit, OpenRouter not answering) is the same
    # for every provider, so the next isn't asked
    prov, said = None, []
    for p in up:
        ok, kind, words = probe(m["id"], p, job)
        if ok:
            prov = p
            break
        said.append((kind, words))
        if kind not in ("data", "refused", "down"):
            break
    if not prov:
        other = next(((k, w) for k, w in said if k != "data"), None)
        if other is None:
            raise ValueError(f"every provider running {model_id} may store or train on prompts: "
                             "choose another model")
        raise ValueError(f"{model_id} couldn’t be pinned: {other[1]}")
    return {"kind": "openrouter", "id": m["id"], "version": m["version"],
            "name": m["name"], "provider": prov["tag"] or prov["name"],
            "provider_name": prov["name"], "precision": prov["precision"],
            "price_in": prov["price_in"] if prov["price_in"] is not None else m["price_in"],
            "price_out": prov["price_out"] if prov["price_out"] is not None
            else m["price_out"]}


def save(job: str, model_id: str, by: str) -> dict:
    """Pin a job to a model: LOCAL, or an OpenRouter id — saved with its dated
    version and its first provider, never an alias alone"""
    if job not in JOBS:
        raise ValueError(f"no such job: {job}")
    if model_id == LOCAL:
        value = {"kind": LOCAL}
    else:
        if not has_key():
            raise ValueError("OpenRouter has no key on this server (OPENROUTER_API_KEY), "
                             "so only Local can be chosen")
        value = pin(model_id, job)
    db.ai_set("job:" + job, value, by)
    return value


def drifted(c: dict) -> str:
    """'' while a pinned model still means what was saved; else why not"""
    if not c or c.get("kind") != "openrouter":
        return ""
    # 16c: asked before each request of a batch and on every poll — never
    # waits on OpenRouter: with no list yet, there is nothing to say
    listed = models(wait=False)
    if not listed:
        return ""
    now = next((m for m in listed if m["id"] == c["id"]), None)
    if now is None:
        return (f"{c['id']} is no longer on OpenRouter's list — choose the job's model again "
                "on AI models")
    if now["version"] != c.get("version"):
        return (f"{c['id']} now points to {now['version']}, not the {c.get('version')} this job "
                "was pinned to — choose it again on AI models")
    return ""


# ---------------------------------------------------------------------------
# spend
# ---------------------------------------------------------------------------

def limit() -> float:
    return float(db.ai_get("spend_limit", config.AI_MONTHLY_LIMIT_USD))


def over_limit() -> str:
    """'' under the monthly limit; at it, the plain message AI jobs wait with"""
    spent, cap = db.spend_this_month(), limit()
    if spent < cap:
        return ""
    return (f"waiting: this month's AI spend has reached its ${cap:,.2f} limit — raise it on "
            f"AI models, or wait for next month")


# ---------------------------------------------------------------------------
# 16c: what the OpenRouter key may still spend — its own limit, set on
# OpenRouter, apart from this board's monthly one. GET /api/v1/key: {data:
# {limit, limit_remaining, usage, …}}, null for a key with no limit. Asked in
# the background and kept a minute: a page shows the last answer and never
# waits on OpenRouter for it
# ---------------------------------------------------------------------------

KEY_TTL_S = 60
_KEY = {"got": None, "at": 0.0, "asking": False, "why": ""}
_KEY_LOCK = threading.Lock()


def _ask_key() -> None:
    from . import llm
    try:
        _, raw = llm._http("GET", config.OPENROUTER_BASE_URL + "/key", _headers(), timeout=10)
        d = json.loads(raw).get("data") or {}
        got = {"limit": d.get("limit"), "remaining": d.get("limit_remaining"),
               "usage": d.get("usage"), "reset": d.get("limit_reset")}
        with _KEY_LOCK:
            _KEY.update(got=got, why="")
    except llm.LLMError as e:
        with _KEY_LOCK:
            _KEY["why"] = refusal(e.status, str(e))[1]
    except BaseException as e:                      # noqa: BLE001 — a background line, said
        with _KEY_LOCK:
            _KEY["why"] = refusal(None, repr(e))[1]
    finally:
        with _KEY_LOCK:
            _KEY.update(at=time.time(), asking=False)


def key_allowance() -> dict | None:
    """{limit, remaining, usage, at} as OpenRouter last said — remaining None
    for a key with no limit of its own — or {why} when it couldn't be asked;
    None before its first answer. Never waits: a stale answer is asked again
    in the background"""
    if not has_key():
        return None
    with _KEY_LOCK:
        stale = time.time() - _KEY["at"] >= KEY_TTL_S
        if stale and not _KEY["asking"]:
            _KEY["asking"] = True
            threading.Thread(target=_ask_key, name="openrouter-key", daemon=True).start()
        if _KEY["got"] is None:
            return {"why": _KEY["why"]} if _KEY["why"] else None
        return {**_KEY["got"], "at": _KEY["at"], **({"why": _KEY["why"]} if _KEY["why"] else {})}


def more_than_key(usd: float | None) -> str:
    """'' unless `usd` is more than the OpenRouter key may still spend; else
    the warning, in one line"""
    k = key_allowance() or {}
    left = k.get("remaining")
    if usd is None or left is None or float(usd) <= float(left):
        return ""
    return (f"About ${float(usd):,.2f} is more than the OpenRouter key may still spend "
            f"(${float(left):,.2f}): raise the key’s limit on OpenRouter first, or what is "
            "sent stops where the key runs out")


# ---------------------------------------------------------------------------
# 12i.2: embeddings, for the question builder's duplicate check
# ---------------------------------------------------------------------------

def embed(texts: list[str], job: str = "writer") -> list[list[float]] | None:
    """One vector per text from OPENROUTER_EMBED_MODEL, or None — no key, the
    month's limit reached, or OpenRouter refused: the caller keeps its 13-gram
    check alone. Counted against the month like any other call"""
    if not texts or not has_key() or over_limit():
        return None
    from . import llm
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        chunk = texts[i:i + 64]
        try:
            status, raw = llm._http("POST", config.OPENROUTER_BASE_URL + "/embeddings", _headers(),
                                    json.dumps({"model": config.OPENROUTER_EMBED_MODEL,
                                                "input": chunk,
                                                "provider": provider_prefs()}).encode(),
                                    timeout=60)
            got = json.loads(raw)
        except Exception:                           # noqa: BLE001 — the 13-gram check stands
            return None
        rows = sorted(got.get("data") or [], key=lambda r: r.get("index", 0))
        if status != 200 or len(rows) != len(chunk):
            return None
        out.extend(r["embedding"] for r in rows)
        usage = got.get("usage") or {}
        usd = usage.get("cost")
        db.spend_add(job, config.OPENROUTER_EMBED_MODEL, "openrouter",
                     int(usage.get("prompt_tokens") or 0), 0, float(usd or 0.0))
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def cost(c: dict, tokens_in: int, tokens_out: int) -> float:
    return ((c.get("price_in") or 0) * tokens_in + (c.get("price_out") or 0) * tokens_out) / 1e6


# ---------------------------------------------------------------------------
# warnings — each one line, and only when it holds
# ---------------------------------------------------------------------------

def job_family(job: str) -> str:
    c = choice(job)
    if c and c.get("kind") == "openrouter":
        return family(c["id"])
    from . import llm
    p, m = _env(job)
    if (c and c.get("kind") == LOCAL) or p == LOCAL:
        served = {}
        for v in llm._SERVED.values():
            served.update(v)
        root = next((w for w in served.values() if w), "")
        return family(root) if root else ""
    return family(m) if "/" in (m or "") else ""


def warnings(improving: list[str] | None = None) -> list[dict]:
    """[{job, text}] — the judge sharing a family with another job; a
    writer sharing one with a model being improved; a training-data writer
    whose terms restrict training on its output"""
    fam = {j: job_family(j) for j in JOBS}
    out = []
    for other in ("writer", "data", "checker"):
        if fam["judge"] and fam["judge"] == fam[other]:
            out.append({"job": "judge", "text": f"The judge and the {JOBS[other]['label'].lower()} "
                        f"are both {family_name(fam['judge'])}: a judge tends to favour its own "
                        "family's style."})
    if fam["checker"] and fam["checker"] == fam["writer"]:
        out.append({"job": "checker", "text": "The checker and the question writer are both "
                    f"{family_name(fam['checker'])}: a checker shares its writer's blind spots."})
    for mid in improving or []:
        for j in ("writer", "data"):
            if fam[j] and fam[j] == family(mid):
                out.append({"job": j, "text": f"The {JOBS[j]['label'].lower()} is "
                            f"{family_name(fam[j])}, like {mid.split('/')[-1]}, which is being "
                            "improved: its data carries its own family's habits."})
    if fam["data"] in RESTRICTED_TRAINING:
        out.append({"job": "data", "text": "Check the terms — its output becomes training data. "
                    "Open-weight models (GLM, DeepSeek, Qwen) avoid this."})
    return out
