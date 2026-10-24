"""28: a chat assistant on the board, for the team only
(docs/prompts/phase-28-chat-assistant.md, docs/ASSISTANT.md).

The model picks, the board does. For every message the model chooses one
action from a fixed list and fills in its fields, in JSON forced into that
shape by the provider's JSON-schema mode. The board resolves every name
against its own data — refusing what it doesn't know — and runs the action on
its own numbers: a page opened by the board's own `#tab=…` link, a small table
of 27's cells, a chart from the board's chart code, the testers' phones, the
rented boxes, a help text the board already shows. The model never writes a
number into a reply.

Read-only by construction: every action is a read or a navigation; nothing
here starts, stops, queues, grades, spends or changes anything.

Off unless ASSISTANT_USERS names someone; only those names see it, and every
question, its action and its thumbs go to BENCH_ROOT/assistant/log.jsonl."""

from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
import threading
import time
import uuid
from pathlib import Path

from . import config

REPO = Path(__file__).resolve().parent.parent
NAMES_FILE = REPO / "eval_tasks" / "assistant" / "names.json"
BANK_FILE = REPO / "tests" / "fixtures" / "assistant" / "queries.jsonl"
DEFAULT_PROVIDER = "served:served/Qwen3.6-35B-A3B-Q4-original-k-8"
MAX_TOKENS = 600
TIMEOUT_S = 60.0

ACTIONS = ("open_page", "compare", "best_on", "chart", "phone", "boxes", "explain", "clarify",
           "cant")
PAGES = ("models", "model", "report", "runs", "benchmarks", "playground", "improve", "home")
VIEWS = ("latest", "standard", "frontier", "mobile", "everyday", "exam")
# a Models tab's groups (Group ▾), by view
GROUPS = {"standard": ("knowledge", "commonsense", "reasoning", "math", "trust", "instruction",
                       "lm"),
          "mobile": ("devicemark", "mobileaibench", "mmlu")}
ONLY = ("ours", "open", "closed")
THINKING = ("default", "on", "off", "both")
WHEN = ("any", "today", "7d", "30d", "month", "since_checkpoint", "since_model", "range")
MEASURES = ("score", "keeps", "ran_out", "loops")
PHONE_MEASURES = ("decode", "prefill", "ttft", "memory", "temperature", "all")
AMONG = ("all", "ours", "open", "closed", "phone")
CHARTS = ("bars", "checkpoints", "size", "memory")
WHICH = ("all", "rented", "server")
ABOUT = ("model", "benchmark", "report", "phone", "other")
SIDES = ("a_better", "b_better")
# what the assistant can't do, and where the button is
CANT = {
    "start_run": ("I can't start or queue runs.", "Test a model, at the top right, starts one.",
                  "#tab=runs"),
    "stop_run": ("I can't stop runs.", "Runs lists each run with its Stop.", "#tab=runs"),
    "delete": ("I can't delete anything.", "A model's page has what can be removed, under "
               "History.", ""),
    "settings": ("I can't change the board's settings.", "AI models is where the judge and the "
                 "other jobs are set.", "#tab=ai"),
    "spend": ("I can't spend money or rent boxes.", "Rented runs start from Runs, by masein.",
              "#tab=runs"),
    "grade": ("I can't grade or judge answers.", "Runs has each run's grading.", "#tab=runs"),
    "upload": ("I can't add or upload models.", "Add a model, at the top right, does.", ""),
    "outside": ("That's outside what I can help with.", "I answer questions about what the "
                "board holds and open its pages.", ""),
}
# a benchmark's key on each Models tab's columns (Columns ▾, "&cols=")
FRONTIER_COL = {"gpqa": "gpqa", "hle": "hle", "otis": "otis", "math": "math-l5",
                "mmlupro": "mmlu-pro", "simpleqa": "simpleqa", "arc": "arc-agi-2",
                "ifeval": "ifeval", "ifbench": "ifbench"}
# a published benchmark's name, as each source names it, to our key (MT_BENCH's)
PUBLISHED_AS = (("gpqa", r"^gpqa"), ("hle", r"^hle\b|humanity.s last exam"),
                ("mmlupro", r"^mmlu.?pro"), ("simpleqa", r"^simpleqa"),
                ("otis", r"otis|mock aime"), ("math", r"^math level 5"),
                ("ifeval", r"^ifeval\b"), ("ifbench", r"^ifbench\b"),
                ("arc", r"^arc-?agi-?2"))
# the page's mtGroup: an outside model served here is open, not ours; a
# published one open by its name, else closed
OUTSIDE_RE = re.compile(r"^(ornith|bonsai|gemma|llama|deepseek|gpt-oss|mistral|phi|granite|"
                        r"smollm|pythia|qwen[23](?:\.5)?-)", re.I)
OPEN_RE = re.compile(r"/|gemma|gpt-oss|llama|qwen|qwq|deepseek|mistral|mixtral|\bphi-|stablelm|"
                     r"tinyllama|bonsai|edge0|ornith|\bling\b|granite|smollm|pythia|\bopt-|lfm|"
                     r"youtu|nanbeige|olmo|kimi|\bglm|minimax|nemotron", re.I)
THINK = " · thinking"


# ---------------------------------------------------------------------------
# who sees it
# ---------------------------------------------------------------------------

def users() -> list[str]:
    return [x.strip() for x in str(config.ASSISTANT_USERS or "").split(",") if x.strip()]


def on() -> bool:
    """the assistant exists on the board: ASSISTANT_USERS names someone"""
    return bool(users())


def allowed(name: str | None) -> bool:
    n = (name or "").strip().casefold()
    return bool(n) and n in {u.casefold() for u in users()}


# ---------------------------------------------------------------------------
# the names file, and what the board holds — the model's whole world
# ---------------------------------------------------------------------------

_memo: dict = {"names": None, "stamp": None}


def names() -> dict:
    try:
        st = NAMES_FILE.stat().st_mtime
    except OSError:
        return {}
    if _memo["stamp"] != st:
        _memo.update(names=json.loads(NAMES_FILE.read_text(encoding="utf-8")), stamp=st)
    return _memo["names"]


def norm(s: str) -> str:
    """a name as matched: case, spaces, hyphens, dots and brackets don't count"""
    return re.sub(r"[^a-z0-9=]+", "", str(s or "").lower())


def entity_of(row: str) -> str:
    return str(row).removesuffix(THINK)


def _rep_files() -> dict:
    from . import reports
    return reports.files()


def catalog(payload: dict | None = None) -> dict:
    """what the model may name: the board's models (one a model, its thinking
    row and plain row together), the published ones, the benchmarks, reports,
    phones and the phone app's models, and the topics it can explain"""
    from . import reported
    if payload is None:
        from . import app
        payload = app.results_payload()
    nm = names()
    alias_of = nm.get("models") or {}
    models: dict[str, dict] = {}
    for m in payload.get("models") or []:
        mid = m["id"]
        e = entity_of(mid)
        x = models.setdefault(e, {"id": e, "name": str(m.get("name") or e).removesuffix(THINK),
                                  "rows": [], "published": False, "aliases": [],
                                  "added": None, "tested": None})
        x["rows"].append(mid)
        # the first row's added, the last row's tested
        for k, at, better in (("added", m.get("addedAt"), min), ("tested", m.get("testedAt"), max)):
            if isinstance(at, (int, float)):
                x[k] = at if x[k] is None else better(x[k], at)
    rv = reported.view([m["id"] for m in payload.get("models") or []])
    for m in rv.get("models") or []:
        if m.get("measured"):
            continue                                       # the board's own row already
        models.setdefault(m["id"], {"id": m["id"], "name": m["name"], "rows": [m["id"]],
                                    "published": True, "maker": m.get("maker") or "",
                                    "aliases": [], "added": None, "tested": None})
    # a report's own labels are names people use: "step8804", "Q4 (UD-Q4_K_XL)"
    sc = _rep_files().get("step-checkpoints") or {}
    for o in sc.get("ours") or []:
        e = entity_of(o["row"])
        if e in models and o["label"] not in models[e]["aliases"]:
            models[e]["aliases"].append(o["label"])
    by_name = {norm(x["name"]): x for x in models.values()}
    for key, al in alias_of.items():
        x = models.get(entity_of(key)) or by_name.get(norm(key))
        if x:
            x["aliases"] += [a for a in al if a not in x["aliases"]]
    phone_ours = _phone_ours(models)
    od = _rep_files().get("on-device") or {}
    od_pub = {norm(t["label"]) for t in od.get("theirs") or [] if not t.get("base_only")}
    for x in models.values():
        x["group"] = _group(x)
        x["phone"] = x["id"] in phone_ours or (x["published"] and norm(x["name"]) in od_pub)
    benches = []
    std = nm.get("standard") or {}
    for b in sc.get("benchmarks") or []:
        benches.append({"key": b["key"], "task": b["task"], "label": b["label"],
                        "frontier": True})
    for k, label in std.items():
        benches.append({"key": k, "task": k, "label": label, "frontier": False})
    balias = nm.get("benchmarks") or {}
    for b in benches:
        b["aliases"] = balias.get(b["key"], [])
    reps = [{"id": rid, "title": r.get("title") or rid} for rid, r in sorted(
        _rep_files().items(), key=lambda kv: (kv[1].get("order", 99), kv[0]))]
    phones, apps = _phones()
    return {"models": sorted(models.values(), key=lambda x: (x["group"] != "ours",
                                                             x["published"], x["name"].lower())),
            "benchmarks": benches, "reports": reps, "phones": phones, "apps": apps,
            "topics": topics(payload), "payload": payload, "reported": rv,
            "today": _dt.date.today().isoformat()}


def _group(x: dict) -> str:
    if x["published"]:
        return "open" if OPEN_RE.search(x["name"]) or re.search(r"\bopen\b", x.get("maker") or "",
                                                                  re.I) else "closed"
    if re.match(r"^(local|served|gguf)/", x["id"]) and not OUTSIDE_RE.search(x["name"]):
        return "ours"
    return "open"


def _phone_ours(models: dict) -> set[str]:
    """our builds that have run on a phone: each app model with a test, by its
    checkpoint's rows (step4000-denseq8's are step4000's), and the phone
    builds the app runs (names.json's `checkpoint` naming the build's row)"""
    from . import telemetry
    out = set()
    try:
        cps = telemetry.by_checkpoint()
    except Exception:                                       # noqa: BLE001 — no telemetry yet
        cps = []
    for e in cps:
        if not e.get("phones"):
            continue
        for r in e.get("rows") or []:
            out.add(entity_of(r))
        cp = str((e.get("model") or {}).get("checkpoint") or "")
        if cp.startswith("served/"):
            out.add(entity_of(cp))
    return {x for x in out if x in models}


def _phones() -> tuple[list[dict], list[dict]]:
    """the testers' phones ({name, maker, code}) and the app's models, from
    the testers' report — names only"""
    from . import telemetry
    phones, apps = {}, {}
    try:
        ts, _ = telemetry.tests()
    except Exception:                                       # noqa: BLE001
        ts = []
    for t in ts:
        p = t.get("phone") or {}
        if p.get("id"):
            phones.setdefault(p["id"], {"id": p["id"], "name": p.get("name") or p["id"],
                                        "maker": p.get("maker") or "", "code": p.get("code") or ""})
        m = t.get("model") or {}
        if m.get("id"):
            apps.setdefault(m["id"], {"id": m["id"], "name": m.get("name") or m["id"],
                                      "checkpoint": m.get("checkpoint") or ""})
    for mid, m in (telemetry.names().get("models") or {}).items():
        apps.setdefault(mid, {"id": mid, "name": m.get("name") or mid,
                              "checkpoint": m.get("checkpoint") or ""})
    return sorted(phones.values(), key=lambda p: p["name"]), sorted(apps.values(),
                                                                    key=lambda m: m["name"])


def topics(payload: dict | None = None) -> list[dict]:
    """what `explain` can say: the board's help texts, each from where the
    board keeps it — a report's notes, a benchmark's own description — or,
    for a text the page keeps (`page:`), filled in by the page itself"""
    from . import paired, reports
    notes = [*reports.NOTES, *paired.NOTES]
    out = []
    for key, t in (names().get("topics") or {}).items():
        src = t.get("from") or ""
        text = ""
        if src.startswith("notes:"):
            head = src.split(":", 1)[1]
            text = next((n for n in notes if n.startswith(head)), "")
        out.append({"key": key, "title": t.get("title") or key, "text": text,
                    "page": src.split(":", 1)[1] if src.startswith("page:") else ""})
    tasks = (payload or {}).get("tasks") or {}
    fb = (payload or {}).get("frontierBench") or {}
    for b in (_rep_files().get("step-checkpoints") or {}).get("benchmarks") or []:
        f = fb.get(b["task"]) or {}
        text = " ".join(x for x in (f.get("protocol"), f.get("note")) if x)
        out.append({"key": "bench:" + b["key"], "title": b["label"],
                    "text": (f"{b['label']}: {text}" if text else ""), "page": ""})
    for k, label in (names().get("standard") or {}).items():
        out.append({"key": "bench:" + k, "title": label,
                    "text": (tasks.get(k) or {}).get("desc") or "", "page": ""})
    return out


# ---------------------------------------------------------------------------
# the schema the model fills, and its prompt
# ---------------------------------------------------------------------------

def schema(cat: dict) -> dict:
    """one flat object: the action first, then every field any action takes,
    each an enum of what the board holds where it can be — the ids can't be
    misspelt"""
    mids = [m["id"] for m in cat["models"]]
    bkeys = [b["key"] for b in cat["benchmarks"]]
    groups = sorted({g for gs in GROUPS.values() for g in gs})
    s = lambda enum: {"type": "string", "enum": list(enum)}       # noqa: E731
    arr = lambda enum, n: {"type": "array", "items": s(enum), "maxItems": n}   # noqa: E731
    props = {
        "action": s(ACTIONS), "page": s(PAGES), "view": s(VIEWS), "group": s(groups),
        "show": s(("table", "chart")), "only": s(ONLY), "q": {"type": "string"},
        "models": arr(mids, 8), "benchmarks": arr(bkeys, 9),
        "sort": s(bkeys + ["avg", "name"]), "sort_dir": s(("desc", "asc")),
        "thinking": s(THINKING), "when": s(WHEN),
        "from": {"type": "string"}, "to": {"type": "string"}, "since": s(mids),
        "date": s(("tested", "added")), "phone_builds": {"type": "boolean"},
        "report": s([r["id"] for r in cat["reports"]]), "a": s(mids), "b": s(mids),
        "side": s(SIDES), "question": {"type": "integer"},
        "base": s(mids), "phone": {"type": "string"},
        "app": s([a["name"] for a in cat["apps"]] or [""]),
        "measure": s(MEASURES + PHONE_MEASURES), "among": s(AMONG), "n": {"type": "integer"},
        "chart": s(CHARTS), "which": s(WHICH), "topic": s([t["key"] for t in cat["topics"]]),
        "about": s(ABOUT), "term": {"type": "string"}, "kind": s(tuple(CANT)),
    }
    return {"type": "object", "properties": props, "required": ["action"],
            "additionalProperties": False}


RULES = """You turn one question about the team's model benchmark board into ONE action, as JSON.
You never answer the question yourself and never write a number: the board runs the action and shows its own numbers.

Actions and their fields:
- open_page: open a page of the board.
  page=models: a Models tab. view (latest, standard, frontier, mobile, everyday, exam); group (Standard: knowledge, commonsense, reasoning, math, trust, instruction, lm; Mobile: devicemark, mobileaibench, mmlu); show=chart for its chart; only (ours, open, closed); q: a search's words; benchmarks: the columns to show; sort: a benchmark key (or avg, name) with sort_dir (desc is best first); thinking (on, off, both; "hide the thinking-off rows" is on); when (today, 7d, 30d, month, since_checkpoint, since_model with since, range with from and to as YYYY-MM-DD) and date (tested, or added); models: the models to show alone; phone_builds=true: only the phone builds.
  page=model: models[0]'s page; benchmarks[0] opens its answers on that benchmark; models[1] beside it (vs).
  page=report: report. two-builds reader (report=paired): a, b, benchmarks[0], side (a_better: questions a got right and b wrong; b_better: b right and a wrong), question (its number). step-checkpoints: base (the baseline: Q2 or Q4). ran-out-of-room: models[0] and benchmarks[0]. testers: phone and app open a test's panel (the newest).
  page=benchmarks: the catalogue; benchmarks[0] opens its questions. page=runs, playground, improve, home.
- compare: models (1 to 8) on benchmarks (none: the usual ones). measure: score (default), keeps (what it keeps of its own base), ran_out, loops. One model and one benchmark is "what's X's score".
- best_on: benchmarks[0]; among (all, ours, open, closed, phone: the models that run on a phone); n (default 3).
- chart: chart (bars: models on one benchmark; checkpoints: a line over our training checkpoints; size: score against file size; memory: score against the memory each needs, the phone models); benchmarks[0]; only; models.
- phone: what our testers' phones measured. app: the phone app's model; phone: the phone's name as typed (empty: every phone); measure (decode, prefill, ttft, memory, temperature, all).
- boxes: what is running. which (all, rented, server); models[0] for one model's box.
- explain: topic (a benchmark is bench:<key>).
- clarify: when the question could mean two or more things, or names something not in the lists. about (model, benchmark, report, phone, other); term: the words that are unclear.
- cant: anything that would change the board (start, queue, stop, delete, change a setting, spend, grade, upload) or isn't about the board. kind (start_run, stop_run, delete, settings, spend, grade, upload, outside).

Rules:
- Take ids, keys and topics from the lists below, exactly. Names people type are matched to them: "8804" is step8804, "q4" is Q4 (UD-Q4_K_XL).
- Leave out a field you don't need. Never guess a field the question doesn't ask for.
- A follow-up ("and on HLE?", "and Q2?") repeats the previous action with that change.
- Two questions in one: answer the first.
"""


def prompt(cat: dict) -> str:
    """the system prompt: the rules, then the board's lists — the same for
    every message while the board is unchanged, so a server's prompt cache
    keeps it; today's date and the question come after"""
    ms = "\n".join(
        f"{m['id']} | {m['name']}" + (f" | also: {', '.join(m['aliases'])}" if m['aliases'] else "")
        + f" | {m['group']}" + (" | phone" if m["phone"] else "")
        + (" | published" if m["published"] else "") for m in cat["models"])
    bs = "\n".join(f"{b['key']} | {b['label']}" + (f" | also: {', '.join(b['aliases'])}"
                                                    if b["aliases"] else "")
                   for b in cat["benchmarks"])
    rs = "\n".join(f"{r['id']} | {r['title']}" for r in cat["reports"])
    ps = "\n".join(f"{p['name']}" + (f" | {p['maker']}" if p["maker"] else "")
                   for p in cat["phones"])
    ap = "\n".join(f"{a['name']}" + (f" | checkpoint {a['checkpoint']}" if a["checkpoint"] else "")
                   for a in cat["apps"])
    ts = "\n".join(f"{t['key']} | {t['title']}" for t in cat["topics"])
    ex = "\n".join(f"Q: {q}\nA: {json.dumps(a, ensure_ascii=False)}"
                   for q, a in names().get("examples") or [])
    return (RULES + f"\nModels (id | name | other names | group):\n{ms}\n\nBenchmarks (key | name):\n"
            f"{bs}\n\nReports (id | title):\n{rs}\n\nOur testers' phones:\n{ps}\n\n"
            f"The phone app's models:\n{ap}\n\nTopics:\n{ts}\n\nExamples:\n{ex}\n")


def user_message(q: str, prev: dict | None, today: str) -> str:
    out = []
    if prev and prev.get("q"):
        out.append(f"Previous question: {prev['q']}")
        if prev.get("action"):
            out.append("Its action: " + json.dumps(prev["action"], ensure_ascii=False))
    out.append(f"Today is {today}.")
    out.append(f"Question: {q}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the model: a served one of ours, a llama-server, or OpenRouter
# ---------------------------------------------------------------------------

class Down(Exception):
    """the model didn't answer"""


def provider() -> str:
    return (config.ASSISTANT_PROVIDER or "").strip() or DEFAULT_PROVIDER


def _endpoint(prov: str) -> tuple[str, str, str, dict]:
    """(base URL, key, model name, extra request fields) for a provider:
    served:<board id>, llama:<base URL>, openrouter:<model>"""
    from . import db, served
    kind, _, what = prov.partition(":")
    if kind == "served":
        rec = db.served_get(what)
        if not rec:
            raise Down(f"{what} isn't a served model here")
        base, key = served._endpoint(rec)
        extra = served._pinned_route(rec) if served.is_openrouter(rec) else {}
        return base, key, (rec.get("pin") or {}).get("model") or rec["name"], extra
    if kind == "llama":
        return what.rstrip("/"), "", "assistant", {}
    if kind == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise Down("no OpenRouter key on the server")
        return config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY, what, {
            "provider": {"require_parameters": True}, "usage": {"include": True}}
    raise Down(f"unknown provider {prov!r}")


def call(system: str, user: str, sch: dict, prov: str | None = None) -> dict:
    """one request: {raw, ms, usage}. Thinking off, temperature 0, 600 tokens,
    the reply forced into the schema"""
    from . import served
    prov = prov or provider()
    base, key, model, extra = _endpoint(prov)
    body = {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "action", "schema": sch}},
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True, **extra}
    if prov.startswith("openrouter:"):
        body.pop("chat_template_kwargs")
        body.pop("cache_prompt")
        body["reasoning"] = {"enabled": False}
    t0 = time.time()
    try:
        st, raw = served._http("POST", base.rstrip("/") + "/chat/completions", key, body,
                               timeout=TIMEOUT_S)
    except Exception as e:                                  # noqa: BLE001 — nothing answered
        raise Down(str(getattr(e, "reason", e))) from None
    ms = round(1000 * (time.time() - t0))
    if st >= 400:
        raise Down(f"HTTP {st}: {raw[:200].decode('utf-8', 'replace')}")
    try:
        got = json.loads(raw)
        text = got["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        raise Down("an answer that isn't the server's JSON") from None
    return {"raw": text, "ms": ms, "usage": got.get("usage") or {}}


def busy() -> bool:
    """a board run is using the served model the assistant asks"""
    prov = provider()
    if not prov.startswith("served:"):
        return False
    mid = prov.split(":", 1)[1]
    from . import db
    try:
        rows = db.recent(50)
    except Exception:                                       # noqa: BLE001
        return False
    run = ("preflight", "waiting_gpu", "waiting_lock", "running")
    return any(entity_of(r.get("hf_id") or "") == mid and r.get("status") in run for r in rows)


def up() -> bool:
    """the model answers at all — /health for a llama-server; OpenRouter is up"""
    from . import served
    prov = provider()
    if prov.startswith("openrouter:"):
        return bool(config.OPENROUTER_API_KEY)
    try:
        base, key, _, _ = _endpoint(prov)
        st, _ = served._http("GET", served.root_of(base) + "/health", key, timeout=3.0)
        return st < 500
    except Exception:                                       # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# what the model returned, checked against the board
# ---------------------------------------------------------------------------

def parse(raw: str) -> dict | None:
    """the model's JSON — the first object in what it wrote — or None"""
    s = (raw or "").strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        got = json.loads(m.group(0))
    except ValueError:
        return None
    return got if isinstance(got, dict) and got.get("action") in ACTIONS else None


def _match(term: str, items: list[tuple[str, list[str]]]) -> list[str]:
    """the ids a name matches, loosely: exact, then contained, then close"""
    t = norm(term)
    if not t:
        return []
    exact = [i for i, ns in items if any(norm(n) == t for n in ns)]
    if exact:
        return exact
    part = [i for i, ns in items if any(t in norm(n) for n in ns if len(t) >= 3)]
    if part:
        return part
    close = []
    for i, ns in items:
        r = max((difflib.SequenceMatcher(None, t, norm(n)).ratio() for n in ns), default=0)
        if r >= 0.84:
            close.append((r, i))
    return [i for _, i in sorted(close, reverse=True)]


def _model_items(cat: dict) -> list[tuple[str, list[str]]]:
    return [(m["id"], [m["id"], m["name"], *m["aliases"], *m["rows"]]) for m in cat["models"]]


def model_id(x, cat: dict) -> tuple[str | None, list[str]]:
    """(the model's id, or None and its nearest candidates)"""
    ids = {m["id"] for m in cat["models"]}
    if x in ids:
        return x, []
    if isinstance(x, str) and entity_of(x) in ids:
        return entity_of(x), []
    got = _match(str(x or ""), _model_items(cat))
    return (got[0], []) if len(got) == 1 else (None, got[:5] or nearest(str(x), cat))


def nearest(term: str, cat: dict, about: str = "model") -> list[str]:
    """a few candidates for a name that matched nothing: the closest by name"""
    items = {"model": _model_items(cat),
             "benchmark": [(b["key"], [b["key"], b["label"], *b["aliases"]])
                           for b in cat["benchmarks"]],
             "report": [(r["id"], [r["id"], r["title"]]) for r in cat["reports"]],
             "phone": [(p["name"], [p["name"], p["maker"], p["code"]]) for p in cat["phones"]]
             }.get(about) or []
    got = _match(term, items)
    if got:
        return got[:6]
    t = norm(term)
    scored = sorted(((max((difflib.SequenceMatcher(None, t, norm(n)).ratio() for n in ns),
                          default=0), i) for i, ns in items), reverse=True)
    return [i for r, i in scored[:4] if r > 0.3]


def bench_key(x, cat: dict) -> str | None:
    keys = {b["key"] for b in cat["benchmarks"]}
    if x in keys:
        return x
    got = _match(str(x or ""), [(b["key"], [b["key"], b["label"], *b["aliases"]])
                                for b in cat["benchmarks"]])
    return got[0] if len(got) == 1 else None


STOP = {"the", "one", "show", "open", "me", "what", "whats", "how", "is", "its", "it", "and",
        "that", "this", "about", "model", "models", "page", "compare", "with", "other", "two",
        "score", "doing", "fast", "phone", "on", "of", "for", "which", "hard", "report"}


def resolve(a: dict, cat: dict, q: str = "") -> dict:
    """the action as the board will run it: every name its id, every field the
    action doesn't take dropped, defaults left out — or a clarify naming what
    it couldn't place"""
    act = a.get("action")
    out: dict = {"action": act}
    if act not in ACTIONS:
        return {"action": "cant", "kind": "outside"}
    keep = FIELDS[act]
    for k in keep:
        if k in a and a[k] not in (None, "", [], False):
            out[k] = a[k]
    # models: ids, every one
    for k in ("models",):
        if k in out:
            got = []
            for x in out[k] if isinstance(out[k], list) else [out[k]]:
                mid, near = model_id(x, cat)
                if mid is None:
                    return {"action": "clarify", "about": "model", "term": str(x),
                            "candidates": near}
                if mid not in got:
                    got.append(mid)
            out[k] = got
    for k in ("since", "a", "b", "base"):
        if k in out:
            mid, near = model_id(out[k], cat)
            if mid is None:
                return {"action": "clarify", "about": "model", "term": str(out[k]),
                        "candidates": near}
            out[k] = mid
    if "benchmarks" in out:
        got = []
        for x in out["benchmarks"] if isinstance(out["benchmarks"], list) else [out["benchmarks"]]:
            k = bench_key(x, cat)
            if k is None:
                return {"action": "clarify", "about": "benchmark", "term": str(x),
                        "candidates": nearest(str(x), cat, "benchmark")}
            if k not in got:
                got.append(k)
        out["benchmarks"] = got
    if "sort" in out and out["sort"] not in ("avg", "name"):
        k = bench_key(out["sort"], cat)
        if k:
            out["sort"] = k
        else:
            out.pop("sort")
    if act == "clarify":
        about = out.get("about") or "model"
        out["candidates"] = nearest(out.get("term") or "", cat, about) if out.get("term") else []
        # no term, or one that matched nothing: a name in the question itself
        for w in [] if out["candidates"] else re.findall(r"[a-z0-9.\-]{3,}", q.lower()):
            if w not in STOP:
                out["candidates"] = nearest(w, cat, about)
                if out["candidates"]:
                    break
    return defaults(out)


# the fields each action takes
FIELDS = {
    "open_page": ("page", "view", "group", "show", "only", "q", "models", "benchmarks", "sort",
                  "sort_dir", "thinking", "when", "from", "to", "since", "date", "phone_builds",
                  "report", "a", "b", "side", "question", "base", "phone", "app"),
    "compare": ("models", "benchmarks", "measure", "thinking"),
    "best_on": ("benchmarks", "among", "n", "thinking"),
    "chart": ("chart", "benchmarks", "only", "models"),
    "phone": ("app", "phone", "measure"),
    "boxes": ("which", "models"),
    "explain": ("topic",),
    "clarify": ("about", "term"),
    "cant": ("kind",),
}
DEFAULTS = {"measure": {"compare": "score"}, "among": {"best_on": "all"}, "n": {"best_on": 3},
            "which": {"boxes": "all"}, "show": {"open_page": "table"},
            "thinking": {"open_page": "default", "compare": "default", "best_on": "default"},
            "sort_dir": {"open_page": "desc"}, "date": {"open_page": "tested"},
            "when": {"open_page": "any"}, "about": {"clarify": "model"}}


def defaults(a: dict) -> dict:
    """the action with its defaults left out, so two that mean the same are equal"""
    act = a.get("action")
    out = {k: v for k, v in a.items()
           if not (k in DEFAULTS and DEFAULTS[k].get(act) == v)}
    if act == "phone" and out.get("measure") == "decode":
        out.pop("measure")
    if out.get("action") == "open_page" and "sort" not in out:
        out.pop("sort_dir", None)
    return out


def same(got: dict, want: dict, cat: dict | None = None) -> bool:
    """an evaluation's pass: the same action with the same fields, lists as
    sets (a clarify's candidates are the board's, not the model's); a phone
    as the phones it matches ("the oneplus" is "OnePlus"), a search's words
    and an app's name as matched"""
    def key(a):
        out = {}
        for k, v in defaults(a).items():
            if k == "candidates":
                continue
            if k == "phone" and cat is not None:
                v = sorted(p["id"] for p in cat["phones"] if _phone_ok(p, v)) or norm(v)
            elif k in ("q", "app"):
                v = norm(v)
            out[k] = sorted(v) if isinstance(v, list) else v
        return out
    g, w = key(got), key(want)
    if w.get("action") == "clarify":
        return g.get("action") == "clarify" and g.get("about", "model") == w.get("about", "model")
    return g == w


# ---------------------------------------------------------------------------
# running the action: the board's own pages, numbers and texts
# ---------------------------------------------------------------------------

def _q(s: str) -> str:
    """as the page's encodeURIComponent writes it"""
    from urllib.parse import quote
    return quote(str(s), safe="-_.!~*'()")


def _model(cat: dict, mid: str) -> dict:
    return next(m for m in cat["models"] if m["id"] == mid)


def _label(cat: dict, mid: str) -> str:
    """a model's name as the report and the chat say it: the report's label
    where it has one ("step8804"), else its name"""
    m = _model(cat, mid)
    return m["aliases"][0] if m["aliases"] and not m["published"] else m["name"]


def _rep_label(mid: str, rid: str = "step-checkpoints") -> str | None:
    """its label in a report's file (the two-builds report reads the step
    checkpoints')"""
    rep = _rep_files().get(rid) or {}
    return next((o["label"] for o in rep.get("ours") or [] if entity_of(o["row"]) == mid), None)


def _bench(cat: dict, key: str) -> dict:
    return next(b for b in cat["benchmarks"] if b["key"] == key)


def _date(ts) -> str:
    return _dt.date.fromtimestamp(ts).isoformat() if isinstance(ts, (int, float)) else ""


def when_range(a: dict, cat: dict) -> str:
    """When ▾ as the link says it: 7d, 30d, or "from..to" (today, this month,
    since the newest checkpoint or a model was added)"""
    w = a.get("when") or "any"
    today = _dt.date.fromisoformat(cat["today"])
    if w in ("7d", "30d"):
        return w
    if w == "today":
        return f"{today}..{today}"
    if w == "month":
        return f"{today.replace(day=1)}..{today}"
    if w == "since_checkpoint":
        steps = [m for m in cat["models"] if re.search(r"step\d+", m["id"]) and m.get("added")]
        if steps:
            return f"{_date(max(m['added'] for m in steps))}..{today}"
        return ""
    if w == "since_model" and a.get("since"):
        at = _model(cat, a["since"]).get("added")
        return f"{_date(at)}..{today}" if at else ""
    if w == "range" and a.get("from"):
        f, t = str(a["from"]), str(a.get("to") or today)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", f) and re.match(r"^\d{4}-\d{2}-\d{2}$", t):
            return f"{min(f, t)}..{max(f, t)}"
    return ""


def models_link(a: dict, cat: dict) -> tuple[str, list[str]]:
    """a Models tab's link, and what it says it set"""
    view = a.get("view") or ""
    p, said = [], []
    if view:
        p.append("view=" + view)
        said.append(view.capitalize() if view != "exam" else "Knowledge exam")
    if a.get("group") and view in GROUPS and a["group"] in GROUPS[view]:
        p.append("group=" + a["group"])
        said.append(a["group"])
    if a.get("show") == "chart":
        p.append("show=chart")
        said.append("as a chart")
    ws = when_range(a, cat)
    if ws:
        p.append("tested=" + ws)
        said.append(("added " if a.get("date") == "added" else "tested ") + ws.replace("..", " to "))
    elif (a.get("when") or "any") != "any":
        said.append("any date (that one isn't known)")
    if a.get("date") == "added" and ws:
        p.append("date=added")
    cols = a.get("benchmarks") or []
    if cols:
        v = view or "latest"
        if v in ("latest", "frontier", "standard"):
            keys = [FRONTIER_COL.get(k, k) if v == "frontier" else
                    (_bench(cat, k)["task"] if v == "standard" else k) for k in cols]
            if v != view:
                p.append("view=" + v)
            p.append("cols=" + ",".join(_q(k) for k in keys))
            said.append("columns " + ", ".join(_bench(cat, k)["label"] for k in cols))
    ids = []
    if a.get("phone_builds"):
        ids = [r for m in cat["models"] if m["id"] in _phone_builds(cat) for r in m["rows"]]
        said.append("the phone builds")
    for mid in a.get("models") or []:
        ids += [r for r in _model(cat, mid)["rows"] if r not in ids]
        said.append(_label(cat, mid))
    if ids:
        p.append("models=" + ",".join(_q(i) for i in ids))
    if a.get("only"):
        p.append("only=" + a["only"])
        said.append(a["only"] + " only")
    if a.get("q"):
        p.append("q=" + _q(a["q"]))
        said.append(f"search “{a['q']}”")
    if a.get("sort"):
        k = a["sort"]
        col = FRONTIER_COL.get(k, k) if view == "frontier" else (
            _bench(cat, k)["task"] if view == "standard" and k not in ("avg", "name") else k)
        p.append("sort=" + _q(col) + (":asc" if a.get("sort_dir") == "asc" else ""))
        nm = _bench(cat, k)["label"] if k not in ("avg", "name") else k
        said.append(f"sorted by {nm}" + (", lowest first" if a.get("sort_dir") == "asc" else ""))
    if a.get("thinking") and a["thinking"] != "default":
        p.append("thinking=" + a["thinking"])
        said.append("thinking " + a["thinking"])
    return "#tab=models" + ("&" + "&".join(p) if p else ""), said


def _phone_builds(cat: dict) -> set[str]:
    from . import served
    return {m["id"] for m in cat["models"] if any(
        served.is_phone(_rec(r)) for r in m["rows"])}


def _rec(row: str) -> dict | None:
    from . import db
    return db.served_get(entity_of(row))


def open_page(a: dict, cat: dict) -> dict:
    page = a.get("page") or "models"
    if page == "models":
        link, said = models_link(a, cat)
        return {"kind": "open", "link": link,
                "say": "Opened Models" + (": " + ", ".join(said) if said else "") + "."}
    if page == "model":
        ms = a.get("models") or []
        if not ms:
            return clarify({"about": "model", "term": "", "candidates": []}, cat)
        mid = ms[0]
        row = pick_row(cat, mid, [b["task"] for b in cat["benchmarks"]], "default")
        link = "#model=" + _q(row)
        said = f"Opened {_label(cat, mid)}’s page"
        if a.get("benchmarks"):
            b = _bench(cat, a["benchmarks"][0])
            link += "&answers=" + _q(b["task"])
            said += f", its answers on {b['label']}"
            if len(ms) > 1:
                link += "&vs=" + _q(pick_row(cat, ms[1], [b["task"]], "default"))
                said += f" beside {_label(cat, ms[1])}"
        return {"kind": "open", "link": link, "say": said + "."}
    if page == "report":
        return _open_report(a, cat)
    if page == "benchmarks":
        if a.get("benchmarks"):
            b = _bench(cat, a["benchmarks"][0])
            return {"kind": "open", "link": "#tab=benchmarks&q=" + _q(b["task"]),
                    "say": f"Opened {b['label']}’s questions."}
        return {"kind": "open", "link": "#tab=benchmarks", "say": "Opened Benchmarks."}
    slug = {"runs": "runs", "playground": "playground", "improve": "improve", "home": "home"}[page]
    return {"kind": "open", "link": "#tab=" + slug, "say": f"Opened {page.capitalize()}."}


def _open_report(a: dict, cat: dict) -> dict:
    rid = a.get("report")
    if not rid:
        return clarify({"about": "report", "term": "", "candidates": []}, cat)
    title = next((r["title"] for r in cat["reports"] if r["id"] == rid), rid)
    link, said = "#tab=reports&report=" + _q(rid), [title]
    if rid == "paired":
        for k in ("a", "b"):
            if a.get(k):
                lab = _rep_label(a[k])
                if lab:
                    link += f"&{k}=" + _q(lab)
        if a.get("a") and a.get("b"):
            said.append(f"{_label(cat, a['a'])} against {_label(cat, a['b'])}")
        if a.get("benchmarks"):
            b = _bench(cat, a["benchmarks"][0])
            side = "b" if a.get("side") == "b_better" else "a"
            link += f"&bench={_q(b['key'])}&side={side}&q={int(a.get('question') or 1)}"
            win = a.get("b") if side == "b" else a.get("a")
            lose = a.get("a") if side == "b" else a.get("b")
            said.append(f"the reader on {b['label']}" + (
                f", where {_label(cat, win)} is right and {_label(cat, lose)} wrong"
                if a.get("side") and win and lose else ""))
    elif rid == "step-checkpoints" and a.get("base"):
        lab = _rep_label(a["base"])
        if lab:
            link += "&base=" + _q(lab)
            said.append(f"baseline {lab.split(' (')[0]}")
    elif rid == "ran-out-of-room" and (a.get("models") or a.get("benchmarks")):
        if a.get("models"):
            lab = _rep_label(a["models"][0], "ran-out-of-room") or _label(cat, a["models"][0])
            link += "&model=" + _q(lab)
            said.append(lab)
        if a.get("benchmarks"):
            link += "&bench=" + _q(a["benchmarks"][0])
            said.append(_bench(cat, a["benchmarks"][0])["label"])
    elif rid == "testers" and (a.get("phone") or a.get("app")):
        t = _newest_test(cat, a.get("app"), a.get("phone"))
        if t:
            link += "&test=" + _q(t["test_id"])
            said.append(f"the newest test of {t['model']['name']} on {t['phone']['name']}")
        else:
            said.append("no test of that yet")
    return {"kind": "open", "link": link, "say": "Opened " + ", ".join(said) + "."}


def _newest_test(cat: dict, app: str | None, phone: str | None) -> dict | None:
    from . import telemetry
    ts, _ = telemetry.tests()
    ok = [t for t in ts if _app_ok(t.get("model") or {}, app) and _phone_ok(t.get("phone") or {},
                                                                          phone)]
    return max(ok, key=lambda t: t.get("received") or "", default=None)


def _app_ok(m: dict, app: str | None) -> bool:
    if not app:
        return True
    a = norm(app)
    return a in (norm(m.get("name")), norm(m.get("id"))) or (
        a and a == norm(str(m.get("checkpoint") or "").split("/")[-1]))


def _phone_ok(p: dict, phone: str | None) -> bool:
    """a phone as typed: its name, maker or code — "the oneplus" is every
    OnePlus"""
    if not phone:
        return True
    t = norm(re.sub(r"^(the|my|our)\s+", "", phone.strip(), flags=re.I))
    return any(t and (t in norm(x) or norm(x) in t and len(norm(x)) > 3)
               for x in (p.get("name"), p.get("maker"), p.get("code")) if x)


def pick_row(cat: dict, mid: str, tasks: list[str], thinking: str) -> str:
    """a model's row for these benchmarks: thinking on or off when asked, else
    the one with the most scores — a tie goes to thinking on (27c's rule)"""
    rows = _model(cat, mid)["rows"]
    on = [r for r in rows if r.endswith(THINK)]
    off = [r for r in rows if not r.endswith(THINK)]
    if thinking == "on" and on:
        return on[0]
    if thinking == "off" and off:
        return off[0]
    cells = cat["payload"].get("cells") or {}

    def n(r):
        return sum(1 for t in tasks if _num(((cells.get(t) or {}).get(r) or {}).get("v")))
    return max(rows, key=lambda r: (n(r), r.endswith(THINK)))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pct(v) -> str:
    return f"{100 * v:.1f}"


def _published(cat: dict, mid: str, key: str) -> dict | None:
    """a published number of a model on one of our benchmarks: Epoch AI's own
    run first, then a model card, then the others"""
    rank = {"epoch": 0, "card": 1, "epoch_external": 2, "aa": 3}
    got = []
    for s in cat["reported"].get("scores") or []:
        if s["model"] != mid:
            continue
        name = str(s.get("benchmark") or "").lower()
        if any(k == key and re.search(rx, name) for k, rx in PUBLISHED_AS):
            got.append(s)
    if not got:
        return None
    s = min(got, key=lambda x: rank.get(x["source"], 9))
    v = s["value"] / 100 if s.get("unit") == "points" or s["value"] > 1.5 else s["value"]
    by = s.get("published_by") or {"epoch": "Epoch AI", "epoch_external": "Epoch AI",
                                    "aa": "Artificial Analysis"}.get(s["source"], "")
    return {"v": v, "se": None, "published": by or "its card", "url": s.get("url") or "",
            "setting": s.get("setting") or ""}


def value(cat: dict, mid: str, key: str, thinking: str = "default",
          measure: str = "score") -> dict | None:
    """one cell: {text, v, se, published?, url, row} — or None"""
    b = _bench(cat, key)
    m = _model(cat, mid)
    if measure == "keeps":
        return _keeps(cat, mid, key)
    if measure == "loops":
        return _loops(cat, mid, key)
    if m["published"]:
        p = _published(cat, mid, key)
        return {**p, "text": _pct(p["v"])} if p else None
    row = pick_row(cat, mid, [b["task"]], thinking)
    c = ((cat["payload"].get("cells") or {}).get(b["task"]) or {}).get(row) or {}
    if measure == "ran_out":
        n, out = c.get("answers"), c.get("ranOut")
        if not _num(n) or not _num(out):
            return None
        return {"v": out / n if n else None, "text": f"{out:,} of {n:,}", "row": row}
    if not _num(c.get("v")):
        p = _published(cat, mid, key)          # a board model reported elsewhere too
        return {**p, "text": _pct(p["v"])} if p else None
    se = c.get("se")
    return {"v": c["v"], "se": se if _num(se) else None, "row": row,
            "text": _pct(c["v"]) + (f" ± {_pct(se)}" if _num(se) else "")}


def _keeps(cat: dict, mid: str, key: str) -> dict | None:
    """what it keeps of its own base, as On-device works it out"""
    d = _report(cat, "on-device")
    if not d:
        return None
    m = _model(cat, mid)
    for r in d.get("rows") or []:
        if (r.get("ours") and entity_of(r.get("row") or "") == mid) or (
                not r.get("ours") and norm(r["label"]) == norm(m["name"])):
            k = (r.get("retention") or {}).get(key)
            if k is None:
                return None
            si = d.get("standIn") or {}
            of = si.get("of") if (r.get("keepOf") or {}).get(key) == "stand_in" else (
                r.get("baseLabel") or "its base")
            return {"v": k, "text": f"{100 * k:.1f}% of {of}"}
    return None


def _loops(cat: dict, mid: str, key: str) -> dict | None:
    d = _report(cat, "step-checkpoints")
    lab = _rep_label(mid)
    if not d or not lab:
        return None
    lp = (((d.get("beyond") or {}).get(lab) or {}).get(key) or {}).get("loops") or {}
    if not lp.get("answers"):
        return None
    return {"v": lp.get("share"), "text": f"{100 * (lp.get('share') or 0):.1f}% "
            f"({lp.get('looped', 0):,} of {lp['answers']:,})"}


_reps: dict = {}
_reps_lock = threading.Lock()


def _report(cat: dict, rid: str) -> dict | None:
    """a report's payload, as its page reads it, once a message"""
    from . import reports
    k = (id(cat), rid)
    with _reps_lock:
        if k not in _reps:
            try:
                _reps.clear() if len(_reps) > 8 else None
                _reps[k] = reports.report(rid, cat["payload"])
            except Exception:                               # noqa: BLE001 — that report alone
                _reps[k] = None
        return _reps[k]


def _cell(x: dict | None) -> dict:
    """a value as the chat's table draws it (27's cell)"""
    if not x:
        return {"v": None, "text": "–", "why": "not in yet"}
    out = {"v": x.get("v"), "text": x["text"]}
    if x.get("published"):
        out.update(pub=x["published"], url=x.get("url") or "", tip=x.get("setting") or "")
    return out


def compare(a: dict, cat: dict) -> dict:
    ms = a.get("models") or []
    if not ms:
        return clarify({"about": "model", "term": "", "candidates": []}, cat)
    keys = a.get("benchmarks") or _usual(cat)
    meas, th = a.get("measure") or "score", a.get("thinking") or "default"
    rows = [{"id": mid, "name": _label(cat, mid), "group": _model(cat, mid)["group"],
             "published": _model(cat, mid)["published"],
             "cells": {k: _cell(value(cat, mid, k, th, meas)) for k in keys}} for mid in ms]
    cols = [{"key": k, "label": _bench(cat, k)["label"]} for k in keys]
    what = {"score": "", "keeps": "what each keeps of its own base",
            "ran_out": "answers that ran out of room", "loops": "answers that ended in a loop"}
    link, _ = models_link({"view": "frontier" if all(_bench(cat, k)["frontier"] for k in keys)
                           else "standard", "models": ms, "benchmarks": keys}, cat)
    if meas in ("keeps",):
        link = "#tab=reports&report=on-device"
    if len(ms) == 1 and len(keys) == 1:
        c = rows[0]["cells"][keys[0]]
        say = (f"{rows[0]['name']} on {cols[0]['label']}" + (f", {what[meas]}" if what[meas]
                                                             else "") + ":")
        if c["v"] is None and c["text"] == "–":
            say = f"{rows[0]['name']} has no {what[meas] or 'score'} on {cols[0]['label']} yet."
    else:
        say = (", ".join(r["name"] for r in rows) + " on " + ", ".join(c["label"] for c in cols)
               + (f": {what[meas]}" if what[meas] else "") + ".")
    if any(r["published"] for r in rows) and any(not r["published"] for r in rows):
        say += " Published numbers are shown as published, never ranked with ours."
    return {"kind": "table", "say": say, "cols": cols, "rows": rows, "link": link,
            "measure": meas}


def _usual(cat: dict) -> list[str]:
    """the benchmarks Models ▸ Latest shows first"""
    g = (_rep_files().get("step-checkpoints") or {}).get("glance_benchmarks") or []
    return [k for k in g if any(b["key"] == k for b in cat["benchmarks"])] or ["gpqa"]


def best_on(a: dict, cat: dict) -> dict:
    keys = a.get("benchmarks") or []
    if not keys:
        return clarify({"about": "benchmark", "term": "", "candidates": []}, cat)
    k, among, n = keys[0], a.get("among") or "all", max(1, min(10, int(a.get("n") or 3)))
    th = a.get("thinking") or "default"
    pool = [m for m in cat["models"] if among == "all" or (
        m["phone"] if among == "phone" else m["group"] == among)]
    here, pub = [], []
    for m in pool:
        x = value(cat, m["id"], k, th)
        if x and _num(x.get("v")):
            (pub if x.get("published") else here).append((x["v"], m["id"], x))
    here.sort(key=lambda t: -t[0])
    pub.sort(key=lambda t: -t[0])
    b = _bench(cat, k)
    rows = [{"id": mid, "name": _label(cat, mid), "rank": i + 1, "cells": {k: _cell(x)}}
            for i, (_, mid, x) in enumerate(here[:n])]
    prow = [{"id": mid, "name": _label(cat, mid), "cells": {k: _cell(x)}}
            for _, mid, x in pub[:n]]
    who = {"all": "", "ours": " of ours", "open": " among open models",
           "closed": " among closed models", "phone": " among the models that run on a phone"}
    say = (f"Best on {b['label']}{who[among]}, measured here: {rows[0]['name']}."
           if rows else f"No model{who[among]} is measured here on {b['label']}.")
    if prow:
        say += " Published numbers are listed apart: they come from each maker's own runs."
    view = "frontier" if b["frontier"] else "standard"
    link, _ = models_link({"view": view, "only": among if among in ONLY else None,
                           "sort": k}, cat)
    return {"kind": "table", "say": say, "cols": [{"key": k, "label": b["label"]}], "rows": rows,
            "published": prow, "link": link, "measure": "score"}


def chart(a: dict, cat: dict) -> dict:
    kind = a.get("chart") or "bars"
    keys = a.get("benchmarks") or []
    k = keys[0] if keys else _usual(cat)[0]
    b = _bench(cat, k)
    if kind == "checkpoints":
        return {"kind": "chart", "chart": kind, "report": "step-checkpoints", "bench": k,
                "say": f"{b['label']} across training, our checkpoints by step.",
                "link": "#tab=reports&report=step-checkpoints"}
    if kind == "size":
        return {"kind": "chart", "chart": kind, "report": "step-checkpoints", "bench": k,
                "say": f"{b['label']} against each build's file size.",
                "link": "#tab=reports&report=step-checkpoints"}
    if kind == "memory":
        return {"kind": "chart", "chart": kind, "report": "on-device", "bench": k,
                "say": f"{b['label']} against the memory each needs.",
                "link": "#tab=reports&report=on-device"}
    ms = a.get("models") or [m["id"] for m in cat["models"] if (
        not a.get("only") or m["group"] == a["only"])]
    pts = []
    for mid in ms:
        x = value(cat, mid, k)
        if x and _num(x.get("v")):
            pts.append({"id": mid, "name": _label(cat, mid), "v": x["v"], "se": x.get("se"),
                        "published": x.get("published") or "", "text": x["text"],
                        "ours": _model(cat, mid)["group"] == "ours"})
    pts.sort(key=lambda p: -p["v"])
    link, _ = models_link({"view": "frontier" if b["frontier"] else "standard", "show": "chart",
                           "only": a.get("only")}, cat)
    who = f", {a['only']} only" if a.get("only") else ""
    return {"kind": "chart", "chart": "bars", "bench": k, "points": pts[:20], "link": link,
            "say": f"{b['label']}{who}: " + (f"{len(pts)} models" if pts else "no scores yet")
            + (", the 20 highest shown." if len(pts) > 20 else ".")}


PHONE_COLS = (("decode", "decode", "tok/s"), ("prefill", "prefill_warm", "tok/s"),
              ("ttft", "ttft_ms", "ms"), ("memory", "engine_mem_mb", "MB"),
              ("temperature", "temp_rise", "°C"))


def phone(a: dict, cat: dict) -> dict:
    d = _report(cat, "testers") or {}
    meas = a.get("measure") or "decode"
    cols = [c for c in PHONE_COLS if meas == "all" or c[0] == meas]
    rows = []
    for p in d.get("perPhone") or []:
        if not _app_ok(p.get("model") or {}, a.get("app")) or not _phone_ok(p.get("phone") or {},
                                                                            a.get("phone")):
            continue
        cells = {}
        for key, f, unit in cols:
            v = p.get(f)
            cells[key] = {"v": v, "text": "–" if v is None else (
                f"{v:,.0f} {unit}" if unit in ("MB", "ms") else f"{v:.1f} {unit}"),
                **({"why": "not measured in its tests"} if v is None else {})}
        rows.append({"id": f"{p['phone']['id']}|{p['model']['id']}",
                     "name": f"{p['model']['name']} on {p['phone']['name']}",
                     "cells": cells, "tests": p.get("tests") or p.get("n")})
    first = cols[0][0]
    rows.sort(key=lambda r: -(r["cells"][first]["v"] or -1))
    head = {"decode": "decode speed", "prefill": "prefill speed", "ttft": "time to first token",
            "memory": "engine memory", "temperature": "temperature rise", "all": "measures"}[meas]
    what = " ".join(x for x in (a.get("app") or "", "on " + a["phone"] if a.get("phone") else "")
                    if x)
    say = (f"Our testers' phones, {head}" + (f" — {what}" if what else "") + ": "
           + (f"{len(rows)} phone and model pair{'s' if len(rows) != 1 else ''}, the median of "
              "each pair's tests." if rows else "no tests of that yet."))
    return {"kind": "table", "say": say, "rows": rows,
            "cols": [{"key": c[0], "label": head if meas != "all" else c[0]} for c in cols],
            "link": "#tab=reports&report=testers"}


def boxes(a: dict, cat: dict) -> dict:
    """what is running: the rented boxes (as Runs reads them) and the
    server's own runs"""
    which = a.get("which") or "all"
    ms = a.get("models") or []
    out = []
    if which in ("all", "rented"):
        from . import reports
        reports._scripts()
        import import_frontier as imf
        got = imf.read_boxes()
        for r in got.get("runs") or []:
            if r.get("status") not in ("running", "loading"):
                continue
            if ms and entity_of(r.get("hf_id") or "") not in ms:
                continue
            tasks = [t for t in str(r.get("tasks") or "").split(",") if t]
            labels = [next((b["label"] for b in cat["benchmarks"] if b["task"] == t), t)
                      for t in tasks]
            out.append({"where": "rented", "name": _name_of(cat, r.get("hf_id") or ""),
                        "what": ", ".join(labels) + (f" · {r['boxes_n']} boxes"
                                                     if (r.get("boxes_n") or 0) > 1 else ""),
                        "progress": f"{r['n']:,} of {r['of']:,}" if _num(r.get("n"))
                        and _num(r.get("of")) else (r.get("status_words") or ""),
                        "eta": _when(r.get("finish"))})
    if which in ("all", "server"):
        from . import db
        run = ("preflight", "waiting_gpu", "waiting_lock", "running")
        for r in db.recent(100):
            if r.get("status") not in run:
                continue
            if ms and entity_of(r.get("hf_id") or "") not in ms:
                continue
            out.append({"where": "server", "name": _name_of(cat, r.get("hf_id") or ""),
                        "what": r.get("suite") or "", "progress": r.get("progress") or "",
                        "eta": ""})
    where = {"all": "", "rented": " on rented boxes", "server": " on the server"}[which]
    if ms:
        where += " for " + ", ".join(_label(cat, m) for m in ms)
    say = (f"{len(out)} run{'s' if len(out) != 1 else ''} going{where}." if out
           else f"Nothing is running{where}.")
    return {"kind": "list", "say": say, "items": out, "link": "#tab=runs"}


def _name_of(cat: dict, hf_id: str) -> str:
    e = entity_of(hf_id)
    m = next((x for x in cat["models"] if x["id"] == e), None)
    return _label(cat, e) if m else hf_id


def _when(ts) -> str:
    if not _num(ts):
        return ""
    return _dt.datetime.fromtimestamp(ts).strftime("%-d %b %H:%M")


def explain(a: dict, cat: dict) -> dict:
    t = next((x for x in cat["topics"] if x["key"] == a.get("topic")), None)
    if not t:
        return clarify({"about": "other", "term": a.get("topic") or "", "candidates": []}, cat)
    link = ""
    if t["key"].startswith("bench:"):
        k = t["key"].split(":", 1)[1]
        link = "#tab=benchmarks&q=" + _q(_bench(cat, k)["task"]) if any(
            b["key"] == k for b in cat["benchmarks"]) else ""
    elif not t["page"]:
        link = "#tab=reports&report=step-checkpoints"
    return {"kind": "text", "say": t["title"] + ".", "title": t["title"], "text": t["text"],
            "page": t["page"], "link": link}


def clarify(a: dict, cat: dict) -> dict:
    about = a.get("about") or "model"
    cands = a.get("candidates")
    if cands is None:
        cands = nearest(a.get("term") or "", cat, about)
    if not cands:
        cands = {"report": [r["id"] for r in cat["reports"]],
                 "phone": [p["name"] for p in cat["phones"]]}.get(about, [])
    label = {"model": lambda x: _label(cat, x) if any(m["id"] == x for m in cat["models"]) else x,
             "benchmark": lambda x: _bench(cat, x)["label"] if any(
                 b["key"] == x for b in cat["benchmarks"]) else x,
             "report": lambda x: next((r["title"] for r in cat["reports"] if r["id"] == x), x)
             }.get(about, lambda x: x)
    q = {"model": "Which model do you mean?", "benchmark": "Which benchmark do you mean?",
         "report": "Which report?", "phone": "Which phone?"}.get(about, "Which do you mean?")
    return {"kind": "clarify", "say": q, "choices": [{"label": label(c), "value": c}
                                                    for c in cands[:6]]}


def cant(a: dict, cat: dict) -> dict:
    said, where, link = CANT.get(a.get("kind") or "outside", CANT["outside"])
    return {"kind": "text", "say": said, "text": where, "link": link}


RUN = {"open_page": open_page, "compare": compare, "best_on": best_on, "chart": chart,
       "phone": phone, "boxes": boxes, "explain": explain, "clarify": clarify, "cant": cant}


def run(a: dict, cat: dict) -> dict:
    """the reply: what the board shows for the action, every number its own"""
    try:
        return RUN[a["action"]](a, cat)
    except StopIteration:
        return clarify({"about": "other", "term": "", "candidates": []}, cat)


# ---------------------------------------------------------------------------
# a message, start to end; the log
# ---------------------------------------------------------------------------

def log_path() -> Path:
    return Path(config.BENCH_ROOT) / "assistant" / "log.jsonl"


_log_lock = threading.Lock()


def log(entry: dict) -> None:
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.time(), **entry}, ensure_ascii=False) + "\n")


def ask(q: str, who: str, prev: dict | None = None, prov: str | None = None,
        cat: dict | None = None, model=None) -> dict:
    """one message: the model's action, resolved and run. `model` replaces the
    provider (a test's fake: (system, user, schema) -> raw text)"""
    q = (q or "").strip()[:500]
    cat = cat or catalog()
    sys_p, user = prompt(cat), user_message(q, prev, cat["today"])
    mid = uuid.uuid4().hex[:12]
    t0 = time.time()
    raw, usage, err = "", {}, ""
    try:
        if model is not None:
            raw = model(sys_p, user, schema(cat))
        else:
            got = call(sys_p, user, schema(cat), prov)
            raw, usage = got["raw"], got["usage"]
    except Down as e:
        log({"id": mid, "who": who, "q": q, "error": f"down: {e}", "provider": prov or provider()})
        return {"id": mid, "down": True, "reply": {"kind": "down",
                "say": "The assistant's model isn't answering right now."}}
    ms = round(1000 * (time.time() - t0))
    a = parse(raw)
    if a is None:
        err = "not an action"
        action = {"action": "clarify", "about": "other", "term": q}
        reply = {"kind": "text", "say": "I didn't get that — could you say it another way?",
                 "text": "", "link": ""}
    else:
        action = resolve(a, cat, q)
        reply = run(action, cat)
    log({"id": mid, "who": who, "q": q, "prev": prev or None, "raw": raw[:2000],
         "action": action, "kind": reply.get("kind"), "ms": ms, "error": err,
         "provider": "fake" if model is not None else (prov or provider()),
         "usage": usage or None})
    return {"id": mid, "action": action, "reply": reply, "ms": ms}


def thumbs(mid: str, who: str, up_: bool, note: str = "") -> None:
    log({"id": str(mid)[:40], "who": who, "thumbs": "up" if up_ else "down",
         "note": (note or "").strip()[:300]})


def examples(n: int = 3) -> list[str]:
    """three questions for an empty panel, from the test bank, different each
    time and each from another scenario"""
    import random
    try:
        bank = [json.loads(x) for x in BANK_FILE.read_text(encoding="utf-8").splitlines()
                if x.strip()]
    except (OSError, ValueError):
        return []
    good = [b for b in bank if b.get("scenario") not in ("refused", "messy", "ambiguous")
            and not b.get("prev")]
    random.shuffle(good)
    out, seen = [], set()
    for b in good:
        if b["scenario"] not in seen:
            out.append(b["q"])
            seen.add(b["scenario"])
        if len(out) == n:
            break
    return out
