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
# open_page's pages — 28b: "models_table" (the Models tabs) and "model_page" (one
# model), so the two can't be mixed up
PAGES = ("models_table", "model_page", "report", "runs", "benchmarks", "playground", "improve",
         "home")
OTHER_PAGES = ("runs", "benchmarks", "playground", "improve", "reports", "home")
# 28d: the top bar's places, by name; their other names are names.json's
PLACE_NAMES = {"home": "Home", "models": "Models", "runs": "Runs", "benchmarks": "Benchmarks",
               "playground": "Playground", "improve": "Improve", "reports": "Reports"}
VIEWS = ("latest", "standard", "frontier", "mobile", "everyday", "exam")
# a Models tab's groups (Group ▾), by view, with the names the page gives them
GROUPS = {"standard": ("knowledge", "commonsense", "reasoning", "math", "trust", "instruction",
                       "lm"),
          "mobile": ("devicemark", "mobileaibench", "mmlu")}
GROUP_NAMES = {"knowledge": "Knowledge", "commonsense": "Commonsense", "reasoning": "Reasoning",
               "math": "Math", "trust": "Trust & safety", "instruction": "Instruction & maths",
               "lm": "Language modelling", "devicemark": "DeviceMark",
               "mobileaibench": "MobileAIBench", "mmlu": "Mobile-MMLU"}
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
# a row's settings after its model's id: " · thinking", " · lookahead 1"
SETTINGS_RE = re.compile(r"(?: · (?:thinking|lookahead \d+))+$")


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
    """the model a row is a setting of (28d: thinking on or off, lookahead 1
    — one model, never asked between)"""
    return str(row).split(" · ")[0]


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
        x = models.setdefault(e, {"id": e, "name": SETTINGS_RE.sub("", str(m.get("name") or e)),
                                  "rows": [], "published": False, "aliases": [],
                                  "added": None, "tested": None})
        x["rows"].append(mid)
        if mid == e and m.get("name"):                     # the plain row's name, as written
            x["name"] = str(m["name"])
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
    groups = nm.get("benchmark_groups") or {}
    for b in benches:
        b["aliases"] = balias.get(b["key"], [])
        # 28f: a Frontier group's name means every benchmark in it ("the
        # maths one" is OTIS and MATH Level 5: asked back)
        b["groups"] = [x for g, v in groups.items() if b["key"] in v.get("benchmarks", [])
                       for x in (g, *v.get("also", []))]
    ral = nm.get("reports") or {}
    titles = nm.get("report_titles") or {}       # 28f: none that names a model
    reps = [{"id": rid, "title": titles.get(rid) or r.get("title") or rid,
             "aliases": list(ral.get(rid, []))}
            for rid, r in sorted(_rep_files().items(),
                                 key=lambda kv: (kv[1].get("order", 99), kv[0]))]
    phones, apps = _phones()
    # a phone's other names (names.json's phones, by its model code); an app
    # model's, the board's names of the model it is ("the phone build", "8804")
    for ph in phones:
        ph["aliases"] = list((nm.get("phones") or {}).get(ph["code"], []))
    for ap in apps:
        cp = str(ap.get("checkpoint") or "")
        e = models.get(entity_of(cp)) or next(
            (x for x in models.values() if not x["published"] and cp and cp in x["aliases"]), None)
        ap["aliases"] = [a for a in (e["aliases"] if e else []) if norm(a) != norm(ap["name"])]
        ap["variant"] = norm(ap["name"]) != norm(cp.split("/")[-1]) and not cp.startswith(
            "served/")
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
    """the reply: {"do": an action, "then": a second action or nothing} — the
    root an object, as every provider takes it. One branch an action (for
    open_page one a page), each strict: its own fields alone, the closed lists
    as enums. Every name — a model, benchmark, report, phone or app — is a
    string as the person wrote it: the board matches it, and asks back when
    several match"""
    e = lambda xs: {"type": "string", "enum": list(xs)}            # noqa: E731
    name = {"type": "string"}
    names_ = {"type": "array", "items": name, "minItems": 1, "maxItems": 9}

    def branch(action: str, props: dict, required=(), page: str | None = None) -> dict:
        head = {"action": {"type": "string", "const": action}}
        if page:
            head["page"] = {"type": "string", "const": page}
        return {"type": "object", "properties": {**head, **props},
                "required": ["action", *(["page"] if page else []), *required],
                "additionalProperties": False}
    actions = {"anyOf": [
        branch("open_page", {"view": e(VIEWS), "group": name, "show": e(("table", "chart")),
                             "only": e(ONLY), "search": name, "columns": names_,
                             "models": names_, "phone_builds": {"type": "boolean"},
                             "sort": name, "sort_dir": e(("desc", "asc")),
                             "thinking": e(THINKING), "when": e(WHEN),
                             "from": name, "to": name, "since": name,
                             "date": e(("tested", "added"))}, page="models_table"),
        branch("open_page", {"model": name, "benchmark": name, "beside": name,
                             "thinking": e(("on", "off"))}, ["model"], page="model_page"),
        branch("open_page", {"report": name, "a": name, "b": name, "benchmark": name,
                             "side": e(SIDES), "base": name, "model": name, "phone": name,
                             "app": name}, ["report"], page="report"),
        branch("open_page", {"page": name, "benchmark": name}, ["page"]),
        branch("compare", {"models": names_, "benchmarks": names_, "measure": e(MEASURES),
                           "thinking": e(THINKING)}, ["models"]),
        branch("best_on", {"benchmark": name, "among": e(AMONG),
                           "n": {"type": "integer", "minimum": 1, "maximum": 10},
                           "thinking": e(THINKING)}, ["benchmark"]),
        branch("chart", {"chart": e(CHARTS), "benchmark": name, "only": e(ONLY),
                         "models": names_}, ["chart", "benchmark"]),
        branch("phone", {"app": name, "phone": name, "measure": e(PHONE_MEASURES)}),
        branch("boxes", {"which": e(WHICH), "model": name}),
        branch("explain", {"topic": e([t["key"] for t in cat["topics"]])}, ["topic"]),
        branch("clarify", {"about": e(ABOUT), "term": name}, ["about"]),
        branch("cant", {"kind": e(tuple(CANT))}, ["kind"]),
    ]}
    return {"type": "object", "properties": {"do": actions, "then": actions},
            "required": ["do"], "additionalProperties": False}


RULES = """You turn one message about the team's model benchmark board into actions, as JSON: {"do": the action} — or, when the message asks two things, {"do": the first, "then": the second}.
You never answer it yourself and never write a number: the board runs the action and shows its own numbers.

Actions and their fields:
- open_page, page=models_table: a Models tab (the tables of every model). view (latest, standard, frontier, mobile, everyday, exam) only when the message names one; group: a group of the tab, as the message names it (Standard's Knowledge, Commonsense, Reasoning, Math, Trust & safety, Instruction & maths, Language modelling; Mobile's DeviceMark, MobileAIBench, Mobile-MMLU); show=chart for its chart; only (ours, open, closed); search: words to search for; columns: the benchmarks to show as columns; sort: a benchmark, avg or name, with sort_dir (desc is best first, asc lowest first); thinking (on, off, both; "hide the thinking-off rows" is on); when (today, 7d, 30d, month, since_checkpoint, since_model with since, range with from and to as YYYY-MM-DD) and date (tested, or added); models: the models to show alone; phone_builds=true: only the phone builds.
- open_page, page=model_page: one model's own page, when the message asks for its page or its answers. model; benchmark opens its answers on that benchmark; beside: another model's answers beside them; thinking (on, off) when the message says it.
- open_page, page=report: report. For the two-builds reader: a, b, benchmark, side (a_better: questions a got right and b wrong; b_better: b right and a wrong). For the step checkpoints: base, the baseline (Q2 or Q4). For the ran-out report: model and benchmark — both when the message names them. For the testers' report: phone, and app — the model the message names — open the newest test of that model on that phone.
- open_page, page: any other place on the board, as the message names it (Runs, Benchmarks — "the catalogue" —, Playground, Improve, Reports, Home). benchmark: its card on Benchmarks ("the GPQA questions").
- compare: models (1 to 8) on benchmarks (leave them out for the usual ones). measure: score (default), keeps (what it keeps of its own base), ran_out (how many answers ran out of room, how often a model runs out of room: compare, and the ran-out report only when the message asks for the report), loops. One model and one benchmark ("X's gpqa", "what did X get on Y") is compare, not its page. thinking when the message says thinking on or off.
- best_on: benchmark; among (all, ours, open, closed, phone: the models that run on a phone); n (default 3).
- chart: chart (bars: models on one benchmark; checkpoints: a line over our training checkpoints; size: score against file size; memory: score against the memory each needs); benchmark; only (ours, open or closed, when the message says so — "the phone models" isn't ours); models.
- phone: what our testers' phones measured. app: the phone app's model; phone: a phone the message names — leave it out for every phone ("on phones", "which phone"); measure: decode for "how fast", speed or tok/s (the default); prefill, ttft (time to first token), memory, temperature, all (everything, or what the testers measured).
- boxes: what is running — runs, rented boxes, the server, how long a run has left, when it finishes or ends. which (all, rented, server); model for one model's run.
- explain: topic, from the topics list (a benchmark's is bench:…). "What's X", "what does X mean", "explain X" for a word the board uses is explain — also when the message names a report: then open the report as the second action.
- clarify: only when the message names nothing at all ("it", "the two", "that one") and there is no previous action. about (model, benchmark, report, phone, other).
- cant: anything that would change the board (start, queue, stop, delete, change a setting, spend, grade, upload) or isn't about the board. kind (start_run, stop_run, delete, settings, spend, grade, upload, outside).

Rules:
- Write every name — model, benchmark, report, group, place, phone, app — in the message's own words ("step one", "8804", "the q4", "the arc one", "the report about phones", "the galaxy"), never a name from the lists below that the message doesn't use. The board matches names and asks back when several match: never choose between them yourself, and never clarify for a name.
- Leave out every field the message doesn't ask for.
- A follow-up: you get the previous action. Keep it, and change only what the new message changes ("and on HLE?" changes the benchmarks; "and q2?" the models).
- Two requests in one message: "do" the first and "then" the second, each a whole action. One request: "do" alone.
"""


def prompt(cat: dict) -> str:
    """the system prompt: the rules, then the board's lists — the same for
    every message while the board is unchanged, so a server's prompt cache
    keeps it; today's date, the previous action and the message come after"""
    ms = "\n".join(
        f"{m['name']}" + (f" | also: {', '.join(m['aliases'])}" if m['aliases'] else "")
        + f" | {m['group']}" + (" | phone" if m["phone"] else "")
        + (" | published" if m["published"] else "") for m in cat["models"])
    bs = "\n".join(b["label"] + (f" | also: {', '.join(x for x in b['aliases'] if x != b['label'])}"
                                  if b["aliases"] else "")
                   + (f" | group: {b['groups'][0]}" if b.get("groups") else "")
                   for b in cat["benchmarks"])
    rs = "\n".join(f"{r['title']}" + (f" | also: {', '.join(r['aliases'])}" if r.get("aliases")
                                     else "") for r in cat["reports"])
    ps = "\n".join(f"{p['name']}" + (f" | {p['maker']}" if p["maker"] else "")
                   + (f" | also: {', '.join(p['aliases'])}" if p.get("aliases") else "")
                   for p in cat["phones"])
    ap = "\n".join(f"{a['name']}" + (f" | also: {', '.join(a['aliases'])}" if a.get("aliases")
                                     else "") for a in cat["apps"])
    pl = "\n".join(PLACE_NAMES[p] + (f" | also: {', '.join(x)}" if x else "")
                   for p in OTHER_PAGES for x in [(names().get("places") or {}).get(p) or []])
    ts = "\n".join(f"{t['key']} | {t['title']}" for t in cat["topics"])
    ex = "\n".join(f"Message: {q}\nReply: {json.dumps(a, ensure_ascii=False)}"
                   for q, a in names().get("examples") or [])
    return (RULES + "\nWhat the board holds — to know what there is; a name is still written in "
            f"the message's words:\n\nModels (name | other names | group):\n{ms}\n\nBenchmarks "
            f"(name | other names | group):\n{bs}\n\nReports (title | other names):\n{rs}\n\nPlaces "
            f"(name | other names):\n{pl}\n\nOur testers' phones (name | maker | other names):\n"
            f"{ps}\n\nThe phone app's models:\n{ap}\n\nTopics:\n{ts}\n\nExamples:\n{ex}\n")


def as_named(a: dict | None, cat: dict) -> dict | None:
    """an action as the model writes it: each model, benchmark, report and
    group by its name, not the board's id — the previous action a follow-up
    keeps"""
    if not a:
        return None
    out = {k: v for k, v in a.items() if k not in ("candidates", "then")}
    mids = {m["id"] for m in cat["models"]}
    keys = {b["key"]: b["label"] for b in cat["benchmarks"]}
    one = {"model": lambda x: _label(cat, x) if x in mids else x,
           "bench": lambda x: keys.get(x, x)}
    for k, kind in (("model", "model"), ("beside", "model"), ("a", "model"), ("b", "model"),
                    ("base", "model"), ("since", "model"), ("benchmark", "bench"),
                    ("sort", "bench"), ("models", "model"), ("benchmarks", "bench"),
                    ("columns", "bench")):
        if isinstance(out.get(k), str):
            out[k] = one[kind](out[k])
        elif isinstance(out.get(k), list):
            out[k] = [one[kind](x) for x in out[k]]
    if out.get("report"):
        out["report"] = next((r["title"] for r in cat["reports"] if r["id"] == out["report"]),
                             out["report"])
    if out.get("group"):
        out["group"] = GROUP_NAMES.get(out["group"], out["group"])
    return out


def user_message(q: str, prev: dict | None, today: str, cat: dict | None = None) -> str:
    out = []
    if prev and prev.get("q"):
        out.append(f"Previous message: {prev['q']}")
        if prev.get("action"):
            a = as_named(prev["action"], cat) if cat else prev["action"]
            out.append("Its action (keep it; change only what the new message changes): "
                       + json.dumps(a, ensure_ascii=False))
    out.append(f"Today is {today}.")
    out.append(f"Message: {q}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the model: a served one of ours, a llama-server, or OpenRouter
# ---------------------------------------------------------------------------

class Down(Exception):
    """the model didn't answer"""


class Unfit(Down):
    """28e: the model can't take what the assistant needs (a JSON schema)"""


# 28e: what the assistant sends OpenRouter that a model may not take; with
# require_parameters, one a model doesn't take finds no endpoint at all (404)
OR_OPTIONAL = ("temperature", "max_tokens", "reasoning")
_or_params: dict = {}


def openrouter_params(model: str, fresh: bool = False) -> set[str] | None:
    """the request parameters OpenRouter's models API says a model takes
    (`supported_parameters`), read once a run (an hour on the board) — None
    when it can't say"""
    hit = _or_params.get(model)
    if hit and not fresh and time.time() - hit[0] < 3600:
        return hit[1]
    from . import ai_models
    try:
        data = ai_models._get("/models").get("data") or []
    except Exception:                                       # noqa: BLE001 — sent as it is
        return hit[1] if hit else None
    m = next((x for x in data if model in (x.get("id"), x.get("canonical_slug"))), None)
    ps = m.get("supported_parameters") if m else None
    got = set(ps) if isinstance(ps, list) else None
    _or_params[model] = (time.time(), got)
    return got


def openrouter_fit(model: str, fresh: bool = False) -> list[str]:
    """the optional parameters left out for this model — or Unfit when it
    can't take a JSON schema (response_format)"""
    ps = openrouter_params(model, fresh)
    if ps is None:
        return []
    if "response_format" not in ps:
        raise Unfit(f"{model} doesn't take response_format on OpenRouter, so its answers can't "
                    "be held to the assistant's schema — choose another model")
    return [k for k in OR_OPTIONAL if k not in ps]


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
        for k in openrouter_fit(model):                # 28e: only what the model takes
            body.pop(k, None)
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
    """the model's action — {"do": …, "then": …}'s first, the second kept in
    it as "then" — or None"""
    s = (raw or "").strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        got = json.loads(m.group(0))
    except ValueError:
        return None
    if isinstance(got, dict) and isinstance(got.get("do"), dict):
        first, second = dict(got["do"]), got.get("then")
        if isinstance(second, dict) and second.get("action") in ACTIONS:
            first["then"] = second
        got = first
    return got if isinstance(got, dict) and got.get("action") in ACTIONS else None


STOP = {"the", "a", "an", "one", "show", "open", "me", "my", "our", "what", "whats", "how", "is",
        "its", "it", "and", "that", "this", "about", "model", "models", "page", "compare",
        "with", "other", "two", "score", "doing", "fast", "on", "of", "for", "which", "hard",
        "report", "build", "please", "pls", "phone", "run", "box", "then"}


def _words(term: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9.\-]+", str(term or "").lower())
            if w not in STOP and len(norm(w)) >= 2]


def _within(t: str, name: str) -> bool:
    """28e: a name contained in another from one of its words' starts —
    never from inside a word, nor running on into part of a later one ("then"
    is not in "the newest checkpoint"; "8804" is in "step8804", "step" too:
    letters and digits are words of their own)"""
    toks = re.findall(r"[a-z]+|[0-9]+|=+", str(name).lower())
    s, pos, words = "".join(toks), 0, []
    for x in toks:
        words.append((pos, pos + len(x)))
        pos += len(x)
    ends = {e for _, e in words}
    return any(s.startswith(t, b) and (b + len(t) <= e or b + len(t) in ends) for b, e in words)


def _match(term: str, items: list[tuple[str, list[str]]]) -> list[str]:
    """the ids a name matches, loosely: exact, then contained, then close,
    then its words in one name, the first starting it ("step one" is every
    step… model; "gemma 3 270m it" is gemma-3-270m-it). 28e: one that matches
    nothing with an "s" stuck on ("q4s", "step4000s") is tried without it"""
    got = _match1(term, items)
    if not got and re.search(r"[a-z0-9]s$", str(term or "").strip().lower()):
        got = _match1(str(term).strip()[:-1], items)
    return got


def _match1(term: str, items: list[tuple[str, list[str]]]) -> list[str]:
    t = norm(term)
    if not t:
        return []
    exact = [i for i, ns in items if any(norm(n) == t for n in ns)]
    if exact:
        return exact
    part = [i for i, ns in items if len(t) >= 3 and any(_within(t, n) for n in ns)]
    if part:
        return part
    close = []
    for i, ns in items:
        r = max((difflib.SequenceMatcher(None, t, norm(n)).ratio() for n in ns), default=0)
        if r >= 0.84:
            close.append((r, i))
    if close:
        return [i for _, i in sorted(close, reverse=True)]
    ws = [norm(w) for w in _words(term)]
    if ws and ws != [t]:
        return [i for i, ns in items if any(norm(n).startswith(ws[0])
                                            and all(w in norm(n) for w in ws[1:]) for n in ns)]
    return []


def _model_items(cat: dict) -> list[tuple[str, list[str]]]:
    return [(m["id"], [m["id"], m["name"], *m["aliases"], *m["rows"]]) for m in cat["models"]]


def model_id(x, cat: dict) -> tuple[str | None, list[str]]:
    """(the model's id, or None and the candidates: the models that match —
    none when nothing does)"""
    ids = {m["id"] for m in cat["models"]}
    if x in ids:
        return x, []
    if isinstance(x, str) and entity_of(x) in ids:
        return entity_of(x), []
    got = _match(_bare(x), _model_items(cat))
    return (got[0], []) if len(got) == 1 else (None, got[:6])


def _bench_items(cat: dict) -> list[tuple[str, list[str]]]:
    """a benchmark by its names — 28d: not its key, so "arc" is every ARC;
    28f: and its Frontier group's, so "the instruction-following one" is IFEval
    and IFBench"""
    return [(b["key"], [b["label"], *b["aliases"], *b.get("groups", [])])
            for b in cat["benchmarks"]]


def nearest(term: str, cat: dict, about: str = "model") -> list[str]:
    """a few candidates for a name: the ones that match it (28d: never the
    closest of those that don't)"""
    items = {"model": _model_items(cat),
             "benchmark": _bench_items(cat),
             "report": [(r["id"], [r["id"], r["title"], *r.get("aliases", [])])
                        for r in cat["reports"]],
             "phone": [(p["name"], [p["name"], p["maker"], p["code"], *p.get("aliases", [])])
                       for p in cat["phones"]]}.get(about) or []
    return _match(_bare(term), items)[:6]


def bench_key(x, cat: dict) -> str | None:
    got, _ = bench_of(x, cat)
    return got


def bench_of(x, cat: dict) -> tuple[str | None, list[str]]:
    """the benchmark a name means, or None and the candidates ("the arc one"
    is ARC-AGI-2, ARC-Challenge and ARC-Easy: asked back)"""
    got = _match(_bare(x), _bench_items(cat))
    return (got[0], []) if len(got) == 1 else (None, got[:6])


def app_name(x, cat: dict) -> tuple[str | None, list[str]]:
    """the phone app's model a name means: its own name, its checkpoint, or
    the board's names of the model it is ("the phone build")"""
    if not x:
        return None, []
    items = [(a["name"], [a["name"], a["id"], *a.get("aliases", [])]) for a in cat["apps"]]
    got = _match(str(x), items)
    if len(got) > 1:     # "step4000": the app model named so; "4000": its own, not a variant
        exact = [g for g in got if norm(g) == norm(x)]
        own = [g for g in got if not next(a for a in cat["apps"] if a["name"] == g)["variant"]]
        got = exact or (own if len(own) == 1 else got)
    return (got[0], []) if len(got) == 1 else (None, got[:6])


def report_id(x, cat: dict) -> tuple[str | None, list[str]]:
    """the report a name means — its id, title or another name ("the report
    about phones" is the testers' and the On-device report: asked back)"""
    items = [(r["id"], [r["id"], r["title"], *r.get("aliases", [])]) for r in cat["reports"]]
    if any(x == i for i, _ in items):
        return x, []
    got = _match(_bare(x), items)
    return (got[0], []) if len(got) == 1 else (None, got[:6])


def place_of(x) -> str | None:
    """a place on the top bar a name means: its name or another one
    ("the benchmarks catalogue", "the GPQA questions" are Benchmarks) — the
    longest name in the words wins"""
    t = f" {' '.join(re.findall(r'[a-z0-9]+', str(x or '').lower()))} "
    best = (0, None)
    for pid, label in PLACE_NAMES.items():
        for n in (pid, label, *((names().get("places") or {}).get(pid) or [])):
            w = " ".join(re.findall(r"[a-z0-9]+", n.lower()))
            if w and f" {w} " in t and len(w) > best[0]:
                best = (len(w), pid)
    return best[1]


def group_of(x, view: str | None) -> tuple[str | None, str | None]:
    """(a Models group a name means, the tab it is on): "mobile-mmlu" is
    Mobile's Mobile-MMLU, "maths" Standard's Math. Among the groups of the tab
    asked for; a group's name exactly or closely first, inside a longer one
    only when nothing else matches (28f: "Maths" was "Instruction & maths",
    where the Frontier group's name sits inside it — b016)"""
    views = [view] if view in GROUPS else list(GROUPS)
    items = [(g, [g, GROUP_NAMES[g]]) for v in views for g in GROUPS[v]]
    t = norm(_bare(x))
    got = []
    for u in (t, t[:-1] if t.endswith("s") else None):
        got = got or [g for g, ns in items if u and any(norm(n) == u for n in ns)]
    if not got:
        r = {g: max(difflib.SequenceMatcher(None, t, norm(n)).ratio() for n in ns)
             for g, ns in items}
        best = max(r.values(), default=0)
        got = [g for g, v in r.items() if v >= 0.84 and v == best]
    got = got or _match(_bare(x), items)
    if len(got) != 1:
        return None, None
    return got[0], next(v for v in GROUPS if got[0] in GROUPS[v])


def as_group(x, cat: dict) -> str | None:
    """28e: a Models group written where a benchmark goes ("DeviceMark",
    "Mobile-MMLU") — the group, unless a benchmark has that very name ("math"
    is MATH Level 5)"""
    t = norm(_bare(x))
    if not t or any(t == norm(n) for _, ns in _bench_items(cat) for n in ns):
        return None
    return next((g for g in GROUP_NAMES if t in (norm(g), norm(GROUP_NAMES[g]))), None)


def _to_group(out: dict, cat: dict) -> dict:
    """28e: groups are recognised before benchmarks — a group in a benchmark's
    place is that group on its Models tab: "mobile devicemark chart" is
    Mobile's DeviceMark as a chart, a column or a sort by it the group"""
    names = [(k, x) for k in ("benchmark", "benchmarks", "columns", "sort")
             for x in (out.get(k) if isinstance(out.get(k), list) else [out.get(k)]) if x]
    g = next((as_group(x, cat) for _, x in names if as_group(x, cat)), None)
    if not g:
        return out
    view = next(v for v in GROUPS if g in GROUPS[v])
    if out.get("page") == "models_table":
        for k in ("columns", "sort"):
            if isinstance(out.get(k), list):
                out[k] = [x for x in out[k] if not as_group(x, cat)] or None
            elif out.get(k) and as_group(out[k], cat):
                out[k] = None
        return {k: v for k, v in {**out, "group": GROUP_NAMES[g], "view": out.get("view") or view
                                  }.items() if v is not None}
    # another action: the group's tab, with the models it named, as a chart if it was one
    return {"action": "open_page", "page": "models_table", "view": view, "group": GROUP_NAMES[g],
            **({"show": "chart"} if out["action"] == "chart" else {}),
            **({"models": out["models"]} if out.get("models") else {})}


# 28d: what is never a name: a date, "none" or "every…", a page's name
NOT_A_NAME = re.compile(r"^(?:\d{4}-\d{2}-\d{2}.*|today|yesterday|none|null|n/?a|nothing|any|all|"
                        r"every|everything|every ?(?:phone|model|benchmark)s?|all (?:the )?"
                        r"(?:phones|models|benchmarks)|(?:the )?(?:models_table|model_page|report|"
                        r"page|tab|home|runs|benchmarks|playground|improve|reports|models))$", re.I)


def _a_name(x) -> bool:
    return isinstance(x, str) and bool(x.strip()) and not NOT_A_NAME.match(x.strip())


# 28f: a model referred to without its name
UNNAMED = re.compile(r"\b(?:they|them|it|the two(?![- ]builds?\b)|the other one)\b", re.I)


def _names_a_model(q: str, cat: dict) -> bool:
    """a model named in the message: a run of its words that matches one —
    by an other name when all its words are ordinary ("the phone build"), so
    "the phone" alone names none"""
    items = _model_items(cat)
    for run in _runs(q):
        if not _match(run, items):
            continue
        if any(w not in STOP for w in _words(run)) or any(
                norm(_bare(n)) == norm(run) for _, ns in items for n in ns):
            return True
    return False


# the words a run of the person's words neither starts nor ends with
RUN_STOP = {"the", "a", "an", "show", "open", "me", "my", "our", "what", "whats", "how", "is",
            "its", "it", "and", "that", "this", "about", "with", "on", "of", "for", "which",
            "please", "pls", "vs", "versus", "to", "in", "at", "by", "did", "does", "do", "get",
            "got", "give", "compare", "against", "one", "s", "was", "were", "has", "have", "go",
            "chart", "find", "same", "then", "also", "only", "just", "i", "you", "we", "or"}


def _runs(said: str, n: int = 4) -> list[str]:
    """the runs of up to n of the person's words, longest first"""
    ws = re.findall(r"[a-z0-9][a-z0-9.\-+_]*", said.lower())
    out = []
    for k in range(min(n, len(ws)), 0, -1):
        for i in range(len(ws) - k + 1):
            if ws[i] not in RUN_STOP and ws[i + k - 1] not in RUN_STOP:
                out.append(" ".join(ws[i:i + k]))
    return list(dict.fromkeys(out))


def _as_written(x, said: str, match, drop: bool = True) -> str | None:
    """28d: a name as the person wrote it. One the messages hold is theirs.
    One they don't is the model's own choice: kept when some of the person's
    words mean that one alone ("Q4 (UD-Q4_K_XL)" for "the q4"); the person's
    words instead when they mean it among others ("Qwen3.6-35B-A3B" for "the
    qwen3.6 model", "testers" for "the report about phones": asked back); and
    when nothing they wrote means it, left out as if not written (a model or
    an app) or kept (a report, benchmark or phone can be what the message
    asks for without its name: "what step8804 got wrong that Q4 got right")"""
    if not isinstance(x, str) or norm(x) in norm(said):
        return x
    target = match(x)
    if len(target) != 1:
        return x                      # not one thing: matched, and asked back, as written
    among = ""
    for run in _runs(said):
        got = match(run)
        if target[0] in got:
            if len(got) == 1:
                return x
            among = among or run      # the longest
    return among or (None if drop else x)


def _bare(x) -> str:
    """a name without "the", "my" or "our" before it"""
    return re.sub(r"^(the|my|our)\s+", "", str(x or "").strip(), flags=re.I)


# the words that name a Models tab, and thinking: a view or a thinking setting
# the message (or the one before it) doesn't name is the board's to leave out
VIEW_WORDS = re.compile(r"\b(latest|standard|frontier|mobile|everyday|exam|devicemark|"
                        r"mobileaibench|mobile-mmlu)\b", re.I)
THINK_WORDS = re.compile(r"\b(think\w*|reasoning)\b", re.I)


def phones_of(term, cat: dict) -> list[str]:
    """the phones a name means: name, model code or another name — or, when
    it names none of them, its maker's: "the oneplus" is every OnePlus, but
    "the oneplus 15" is that one (28f: it was both, the maker being in it)"""
    t = norm(_bare(term))
    if t and any(t == norm(p.get("maker")) for p in cat["phones"]):
        return [p["id"] for p in cat["phones"] if norm(p.get("maker")) == t]
    named = [p["id"] for p in cat["phones"] if _phone_ok(p, term, maker=False)]
    return named or [p["id"] for p in cat["phones"] if _phone_ok(p, term)]


# the fields each action takes; for open_page, each page's
PAGE_FIELDS = {
    "models_table": ("view", "group", "show", "only", "search", "columns", "models",
                     "phone_builds", "sort", "sort_dir", "thinking", "when", "from", "to",
                     "since", "date"),
    "model_page": ("model", "benchmark", "beside", "thinking"),
    "report": ("report", "a", "b", "benchmark", "side", "base", "model", "phone", "app"),
}
FIELDS = {
    "compare": ("models", "benchmarks", "measure", "thinking"),
    "best_on": ("benchmark", "among", "n", "thinking"),
    "chart": ("chart", "benchmark", "only", "models"),
    "phone": ("app", "phone", "measure"),
    "boxes": ("which", "model"),
    "explain": ("topic",),
    "clarify": ("about", "term"),
    "cant": ("kind",),
}
# the fields each report's link takes
REPORT_FIELDS = {"paired": ("a", "b", "benchmark", "side"), "step-checkpoints": ("base",),
                 "ran-out-of-room": ("model", "benchmark"), "testers": ("phone", "app")}
DEFAULTS = {"measure": {"compare": "score", "phone": "decode"}, "among": {"best_on": "all"},
            "n": {"best_on": 3}, "which": {"boxes": "all"}, "show": {"open_page": "table"},
            "thinking": {"open_page": "default", "compare": "default", "best_on": "default"},
            "sort_dir": {"open_page": "desc"}, "date": {"open_page": "tested"},
            "when": {"open_page": "any"}, "about": {"clarify": "model"}}


def _ask_back(about: str, term: str, cands: list[str], keep: dict) -> dict:
    """several match: ask which. 28d: none does — no clarify without choices:
    "I couldn't find it", and a search for it"""
    if not cands:
        return {"action": "not_found", "about": about, "term": str(term),
                **({"then": keep["then"]} if keep.get("then") else {})}
    return {"action": "clarify", "about": about, "term": str(term), "candidates": cands,
            **({"then": keep["then"]} if keep.get("then") else {})}


NAME_FIELDS = ("model", "beside", "a", "b", "base", "since", "models", "benchmark",
               "benchmarks", "columns", "report", "phone", "app", "sort")


def resolve(a: dict, cat: dict, q: str = "", prev_q: str = "",
            prev_action: dict | None = None) -> dict:
    """the action as the board will run it: every model, benchmark and report
    its id, every app its name, a place on the top bar its page, the fields
    the action (or its page) doesn't take dropped, a name the person didn't
    write taken as they wrote it, a view or thinking setting no message named
    left out, defaults left out — or a clarify naming what it couldn't place,
    its candidates the board's, or "not found" when nothing matches. A second
    action ("then") resolved the same way"""
    act = a.get("action")
    if act not in ACTIONS:
        return {"action": "cant", "kind": "outside"}
    then = resolve({k: v for k, v in a["then"].items() if k != "then"}, cat, q, prev_q,
                   prev_action) if (isinstance(a.get("then"), dict)
                                    and a["then"].get("action") in ACTIONS) else None
    page = a.get("page") if act == "open_page" else None
    if act == "open_page" and page not in PAGE_FIELDS:
        # 28d: another place, by its name as written ("the catalogue")
        place = place_of(page) or place_of(q)
        page = "models_table" if place == "models" else place
        if page is None:
            return _ask_back("page", str(a.get("page") or q), [], {"then": then})
    keep = PAGE_FIELDS.get(page) or (("benchmark",) if page else FIELDS[act])
    out: dict = {"action": act, **({"page": page} if page else {})}
    for k in keep:
        if k in a and a[k] not in (None, "", [], False):
            out[k] = a[k]
    if page == "benchmarks" and "benchmark" not in out:
        for r in _runs(str(a.get("page") or "")):      # "the GPQA questions": its card
            if bench_of(r, cat)[0]:
                out["benchmark"] = r
                break
    # 28d: a date, "none" or a page's name is never a model, benchmark,
    # report, phone or app: left out, as if not written
    for k in NAME_FIELDS:
        if k == "sort" and out.get(k) in ("avg", "name"):
            continue
        if isinstance(out.get(k), list):
            out[k] = [x for x in out[k] if _a_name(x)]
        if k in out and not (out[k] if isinstance(out[k], list) else _a_name(out[k])):
            out.pop(k)
    if act == "chart" and out.get("chart", "bars") != "bars":
        out.pop("only", None)                     # a report's chart: its own models
        out.pop("models", None)
    if q:
        some = _written(out, cat, " ".join((q, prev_q, json.dumps(
            as_named(prev_action, cat) or {}, ensure_ascii=False))))
        if some:
            return _ask_back("model", ", ".join(map(str, a.get("models") or [])), some,
                             {"then": then})
    if out.get("when") == "since_model" and "since" not in out:
        out.pop("when")
    out = _to_group(out, cat)
    act, page = out["action"], out.get("page")
    keep = PAGE_FIELDS.get(page) or (("benchmark",) if page else FIELDS[act])
    # 28c: a tab or a thinking setting neither this message nor the one before
    # names is the board's to leave out — "only ours" stays on the tab you're on
    said = f"{q} {prev_q}"
    if "group" in out:
        g, v = group_of(out["group"], out.get("view"))
        out.pop("group")
        if g:
            out.update(group=g, view=out.get("view") or v)
    if "view" in out and not VIEW_WORDS.search(said) and not (
            "group" in out and re.search(re.escape(GROUP_NAMES[out["group"]]).replace(
                r"\-", "[- ]?"), said, re.I)):
        out.pop("view")
        out.pop("group", None)
    if "thinking" in out and not THINK_WORDS.search(said):
        out.pop("thinking")
    if then:
        out["then"] = then
    if "report" in out:
        rid, cands = report_id(out["report"], cat)
        if rid is None:
            return _ask_back("report", out["report"], cands, out)
        out["report"] = rid
        takes = REPORT_FIELDS.get(rid, ())
        # 28f: written in a field it doesn't take, read as the one it is —
        # the testers' "model" is the app's ("open step8804's newest test on
        # the oneplus"); a benchmark in "app" is the benchmark ("HLE")
        if "model" in out and "model" not in takes and "app" in takes and "app" not in out:
            out["app"] = out.pop("model")
        if "app" in out and "app" not in takes and "benchmark" in takes and (
                "benchmark" not in out) and nearest(str(out["app"]), cat, "benchmark"):
            out["benchmark"] = out.pop("app")
        # 28d: the fields that report doesn't use left out ("app" on the ran-out report)
        for k in set(PAGE_FIELDS["report"]) - {"report", *takes}:
            out.pop(k, None)
    # models: each name the board's id; several match, or none: ask back
    if "models" in out:
        got = []
        for x in out["models"] if isinstance(out["models"], list) else [out["models"]]:
            mid, cands = model_id(x, cat)
            if mid is None:
                return _ask_back("model", x, cands, out)
            if mid not in got:
                got.append(mid)
        out["models"] = got
    for k in ("model", "beside", "a", "b", "base", "since"):
        if k in out:
            mid, cands = model_id(out[k], cat)
            if mid is None:
                return _ask_back("model", out[k], cands, out)
            out[k] = mid
    if out.get("beside") and out["beside"] == out.get("model"):
        out.pop("beside")                         # 28e: itself beside itself is its page
    # 28d: a phone that is no phone but the app's model ("step8804") is the app
    if "phone" in out and not phones_of(out["phone"], cat) and "app" not in out and (
            "app" in keep) and app_name(out["phone"], cat)[0]:
        out["app"] = out.pop("phone")
    if "app" in out:
        got, cands = app_name(out["app"], cat)
        if got is None:
            return _ask_back("model", out["app"], cands, out)
        out["app"] = got
    if "phone" in out and not phones_of(out["phone"], cat):
        return _ask_back("phone", out["phone"], nearest(str(out["phone"]), cat, "phone"), out)
    for k in ("benchmarks", "columns"):
        if k in out:
            got = []
            for x in out[k] if isinstance(out[k], list) else [out[k]]:
                key, cands = bench_of(x, cat)
                if key is None:
                    return _ask_back("benchmark", x, cands, out)
                if key not in got:
                    got.append(key)
            out[k] = got
    if "benchmark" in out:
        key, cands = bench_of(out["benchmark"], cat)
        if key is None:
            return _ask_back("benchmark", out["benchmark"], cands, out)
        out["benchmark"] = key
    if "sort" in out and out["sort"] not in ("avg", "name"):
        key, cands = bench_of(out["sort"], cat)
        if key is None:
            return _ask_back("benchmark", out["sort"], cands, out)
        out["sort"] = key
    # a name the action needs that nobody wrote: asked for
    need = {"compare": ("models", "model"), "best_on": ("benchmark", "benchmark"),
            "model_page": ("model", "model"), "report": ("report", "report")}.get(page or act)
    if need and need[0] not in out:
        act, out = "clarify", {"action": "clarify", "about": need[1],
                               **({"then": out["then"]} if out.get("then") else {})}
    if act == "clarify":
        about = out.get("about") or "model"
        out["candidates"] = nearest(out.get("term") or "", cat, about) if out.get("term") else []
        # no term, or one that matched nothing: a name in the message itself
        # (28e: never one of the ordinary words — "then" isn't a model)
        for w in [] if out["candidates"] else [w for w in _words(q) if w not in RUN_STOP]:
            out["candidates"] = nearest(w, cat, about)
            if out["candidates"]:
                break
    out = defaults(out)
    # 28f: "they", "it", "the two", "the other one" with no earlier turn and
    # no model named in the message: which model is asked, whatever the
    # action — the model answered anyway (a ranking, a guess). Not a refusal,
    # nor a word explained ("had it, then lost it")
    if (not prev_q and not prev_action and out["action"] not in ("cant", "explain", "clarify",
                                                                 "not_found")
            and UNNAMED.search(q) and not _names_a_model(q, cat)):
        return {"action": "clarify", "candidates": []}
    t = out.get("then")
    if isinstance(t, dict):
        first = {k: v for k, v in out.items() if k != "then"}
        # 28e: a follow-up whose first action is the turn before's again
        # ("and as a chart" after "best on gpqa"): only the new one
        if prev_action and _same(first, {k: v for k, v in prev_action.items()
                                          if k not in ("then", "candidates")}, cat):
            return t
        # "what's X in the report": the explanation first, the report second
        if first.get("page") == "report" and t.get("action") == "explain":
            return {**t, "then": first}
    return out


def _written(out: dict, cat: dict, said: str) -> list[str] | None:
    """each name in the action as the person wrote it (_as_written). 28f: a
    list of models some of which go and some stay is never shown as what's
    left ("compare the two checkpoints" kept only step8804, whose other names
    say "checkpoint"): the models it named, kept and left out, to ask back
    between — returned"""
    ids = {m["id"] for m in cat["models"]}
    items = _model_items(cat)

    def model(t):
        return [t] if t in ids else [entity_of(t)] if entity_of(t) in ids else _match(
            _bare(t), items)
    reps = [(r["id"], [r["id"], r["title"], *r.get("aliases", [])]) for r in cat["reports"]]
    apps = [(x["name"], [x["name"], x["id"], *x.get("aliases", [])]) for x in cat["apps"]]
    match = {"model": model, "bench": lambda t: _match(_bare(t), _bench_items(cat)),
             "report": lambda t: _match(_bare(t), reps),
             "app": lambda t: _match(str(t), apps),
             "phone": lambda t: phones_of(t, cat)}
    for k, kind in (("models", "model"), ("model", "model"), ("beside", "model"), ("a", "model"),
                    ("b", "model"), ("base", "model"), ("since", "model"),
                    ("benchmarks", "bench"), ("columns", "bench"), ("benchmark", "bench"),
                    ("sort", "bench"), ("report", "report"), ("app", "app"), ("phone", "phone")):
        if k not in out or (k == "sort" and out[k] in ("avg", "name")):
            continue
        drop = kind in ("model", "app")
        if isinstance(out[k], list):
            ys = [_as_written(x, said, match[kind], drop) for x in out[k]]
            if kind == "model" and None in ys and any(ys):
                return list(dict.fromkeys(i for x in out[k] for i in match[kind](x)))
            # several the person's same words mean ("the three bonsai 2 models"):
            # the model spelt out what they wrote, kept
            twice = {y for y in ys if y and ys.count(y) > 1 and y not in out[k]}
            out[k] = [x if y in twice else y for x, y in zip(out[k], ys) if y]
            out[k] = list(dict.fromkeys(out[k]))
            if not out[k]:
                out.pop(k)
        else:
            y = _as_written(out[k], said, match[kind], drop)
            if y:
                out[k] = y
            else:
                out.pop(k)
    return None


def defaults(a: dict) -> dict:
    """the action with its defaults left out, so two that mean the same are equal"""
    act = a.get("action")
    out = {k: v for k, v in a.items() if not (k in DEFAULTS and DEFAULTS[k].get(act) == v)}
    if act == "open_page" and "sort" not in out:
        out.pop("sort_dir", None)
    return out


def same(got: dict, want: dict, cat: dict | None = None, prev: dict | None = None) -> bool:
    """an evaluation's pass, by what the person sees (28d): the same action
    with the same fields after the board's defaults and matching, lists as
    sets; a phone as the phones it matches, a search's words as matched; a
    clarify by what it asks about (its candidates are the board's); a second
    request offered (`then`) when the bank's has one. And what looks different
    but shows the same:
    - boxes: every run (which=all) shows the rented ones and the server's;
    - keeps: a base the board shows anyway beside the models (BF16 by Q2);
    - a refusal is a refusal: its kind changes only the words;
    - a follow-up's new models or benchmarks ("and on HLE?", "same for
      step1000") alone, or beside what was shown (`prev`, the bank's);
    - a Models tab a follow-up doesn't name is the tab it is on;
    - one model on one benchmark: its page opened on that benchmark states
      the number, as compare does.
    `want` may name others that are as good (`also`)"""
    for w in [want, *(want.get("also") or [])]:
        if _same(got, {k: v for k, v in w.items() if k != "also"}, cat, prev):
            return True
    return False


def _bases() -> set[str]:
    """the models the On-device report shows as a base (BF16) or a stand-in
    for one (Q4 on IFEval and IFBench)"""
    od = _rep_files().get("on-device") or {}
    labs = {od.get("ours_base"), (od.get("ours_stand_in") or {}).get("label")}
    return {entity_of(o["row"]) for o in od.get("ours") or []
            if o.get("label") in labs or o.get("base_only")}


def _same(got: dict, want: dict, cat: dict | None, prev: dict | None = None) -> bool:
    def key(a):
        out = {}
        for k, v in defaults(a).items():
            if k in ("candidates", "then", "term"):
                continue
            if k == "phone" and cat is not None:
                v = sorted(phones_of(v, cat)) or norm(v)
            elif k in ("search", "app"):
                v = norm(v)
            out[k] = sorted(v) if isinstance(v, list) else v
        return out
    if want.get("then"):                       # the second request, as a whole action
        if not isinstance(got.get("then"), dict) or not _same(got["then"], want["then"], cat):
            return False
    g, w = key(got), key(want)
    if w.get("action") == "clarify":                # "not found" is its no-choices clarify
        return g.get("action") in ("clarify", "not_found") and g.get("about", "model") == w.get(
            "about", "model")
    if w["action"] == g["action"] == "cant":
        return True
    if w["action"] == g["action"] == "boxes" and "which" not in g:
        w.pop("which", None)
    if w["action"] == "compare" and g.get("page") == "model_page" and "beside" not in g and (
            len(w.get("models") or []) == 1 and len(w.get("benchmarks") or []) == 1
            and not w.get("measure")):
        g = {"action": "compare", "models": [g.get("model")],
             **({"benchmarks": [g["benchmark"]]} if g.get("benchmark") else {}),
             **({"thinking": g["thinking"]} if g.get("thinking") else {})}
    if w.get("measure") == "keeps" and g.get("measure") == "keeps" and "models" in g:
        bases = _bases() - set(w.get("models") or [])
        g["models"] = [m for m in g["models"] if m not in bases]
    pa = (prev or {}).get("action") or {}
    for k in ("models", "benchmarks", "columns"):
        p = set(pa.get(k) or [])
        if p and k in g and k in w and set(w[k]) != p and set(w[k]) <= set(g[k]) <= set(w[k]) | p:
            g[k] = w[k]
    if pa.get("page") == "models_table" and pa.get("view"):
        for x in (g, w):
            if x.get("page") == "models_table":
                x.setdefault("view", pa["view"])
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
    cols = a.get("columns") or []
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
    if a.get("search"):
        p.append("q=" + _q(a["search"]))
        said.append(f"search “{a['search']}”")
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
    page = a.get("page") or "models_table"
    if page == "models_table":
        link, said = models_link(a, cat)
        return {"kind": "open", "link": link,
                "say": "Opened Models" + (": " + ", ".join(said) if said else "") + "."}
    if page == "model_page":
        mid = a.get("model")
        if not mid:
            return clarify({"about": "model", "term": "", "candidates": []}, cat)
        b = _bench(cat, a["benchmark"]) if a.get("benchmark") else None
        th = a.get("thinking") or "default"
        row = pick_row(cat, mid, [b["task"]] if b else [x["task"] for x in cat["benchmarks"]], th)
        link = "#model=" + _q(row)
        said = f"Opened {_label(cat, mid)}’s page"
        if b:
            link += "&answers=" + _q(b["task"])
            said += f", its answers on {b['label']}"
            if a.get("beside"):
                link += "&vs=" + _q(pick_row(cat, a["beside"], [b["task"]], "default"))
                said += f" beside {_label(cat, a['beside'])}"
            else:                       # 28d: one model, one benchmark — its number
                x = value(cat, mid, b["key"], th)
                said += f": {_stated(cat, mid, x)}" if x else "; it has no score on it yet"
        return {"kind": "open", "link": link, "say": said + "."}
    if page == "report":
        return _open_report(a, cat)
    if page == "benchmarks":
        if a.get("benchmark"):
            b = _bench(cat, a["benchmark"])
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
        if a.get("benchmark"):
            b = _bench(cat, a["benchmark"])
            side = "b" if a.get("side") == "b_better" else "a"
            link += f"&bench={_q(b['key'])}&side={side}&q=1"
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
    elif rid == "ran-out-of-room" and (a.get("model") or a.get("benchmark")):
        if a.get("model"):
            lab = _rep_label(a["model"], "ran-out-of-room") or _label(cat, a["model"])
            link += "&model=" + _q(lab)
            said.append(lab)
        if a.get("benchmark"):
            link += "&bench=" + _q(a["benchmark"])
            said.append(_bench(cat, a["benchmark"])["label"])
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
    ids = set(phones_of(phone, cat)) if phone else None
    ok = [t for t in ts if _app_ok(t.get("model") or {}, app)
          and (ids is None or (t.get("phone") or {}).get("id") in ids)]
    return max(ok, key=lambda t: t.get("received") or "", default=None)


def _app_ok(m: dict, app: str | None) -> bool:
    """an app model, by the name the board resolved (app_name)"""
    return not app or norm(app) in (norm(m.get("name")), norm(m.get("id")))


def _phone_ok(p: dict, phone: str | None, maker: bool = True) -> bool:
    """a phone as typed: its name, code or another name, or (`maker`) its
    maker's — "the oneplus" is every OnePlus"""
    if not phone:
        return True
    t = norm(re.sub(r"^(the|my|our)\s+", "", str(phone).strip(), flags=re.I))
    al = p.get("aliases") or (names().get("phones") or {}).get(p.get("code") or "", [])
    return any(t and (t in norm(x) or norm(x) in t and len(norm(x)) > 3)
               for x in (p.get("name"), p.get("maker") if maker else None, p.get("code"), *al)
               if x)


def _thinks(row: str) -> bool:
    return THINK in str(row)


def pick_row(cat: dict, mid: str, tasks: list[str], thinking: str) -> str:
    """a model's row for these benchmarks: thinking on or off when asked, else
    the one with the most scores — a tie goes to thinking on (27c's rule), then
    to the row with the fewest settings (28d: "· lookahead 1" is a setting of
    the same model)"""
    rows = _model(cat, mid)["rows"]
    if thinking in ("on", "off") and any(_thinks(r) == (thinking == "on") for r in rows):
        rows = [r for r in rows if _thinks(r) == (thinking == "on")]
    cells = cat["payload"].get("cells") or {}

    def n(r):
        return sum(1 for t in tasks if _num(((cells.get(t) or {}).get(r) or {}).get("v")))
    return max(rows, key=lambda r: (n(r), _thinks(r), -r.count(" · ")))


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


def _stated(cat: dict, mid: str, x: dict) -> str:
    """a number as a reply says it: with its ± and the thinking setting it was
    measured with, or who published it"""
    if x.get("published"):
        by = x["published"]
        return x["text"] + (f", as published on {by}" if by.startswith("its ")
                            else f", as published by {by}")
    row = x.get("row") or ""
    if row and (_thinks(row) or len(_model(cat, mid)["rows"]) > 1):
        return x["text"] + ", thinking " + ("on" if _thinks(row) else "off")
    return x["text"]


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
                           else "standard", "models": ms, "columns": keys}, cat)
    if meas in ("keeps",):
        link = "#tab=reports&report=on-device"
    if len(ms) == 1 and len(keys) == 1:
        # 28d: one model on one benchmark — the reply says its number
        x = value(cat, ms[0], keys[0], th, meas)
        say = (f"{rows[0]['name']} on {cols[0]['label']}" + (f", {what[meas]}" if what[meas]
                                                             else "") + f": {_stated(cat, ms[0], x)}."
               if x else f"{rows[0]['name']} has no {what[meas] or 'score'} on "
               f"{cols[0]['label']} yet.")
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
    if not a.get("benchmark"):
        return clarify({"about": "benchmark", "term": "", "candidates": []}, cat)
    k, among, n = a["benchmark"], a.get("among") or "all", max(1, min(10, int(a.get("n") or 3)))
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
    k = a.get("benchmark") or _usual(cat)[0]
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
    ids = set(phones_of(a["phone"], cat)) if a.get("phone") else None
    for p in d.get("perPhone") or []:
        if not _app_ok(p.get("model") or {}, a.get("app")) or (
                ids is not None and (p.get("phone") or {}).get("id") not in ids):
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
    ms = [a["model"]] if a.get("model") else []
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


def not_found(a: dict, cat: dict) -> dict:
    """28d: a name nothing on the board matches — said, and a search for it
    where that kind of name is"""
    term = _bare(a.get("term") or "").strip() or "that"
    about = a.get("about") or "model"
    link, where = {"model": ("#tab=models&q=" + _q(term), "Search Models for it ▸"),
                   "benchmark": ("#tab=benchmarks&q=" + _q(term), "Search Benchmarks for it ▸"),
                   "report": ("#tab=reports", "See the reports ▸"),
                   "phone": ("#tab=reports&report=testers", "See our testers' phones ▸"),
                   }.get(about, ("", ""))
    return {"kind": "text", "say": f"I couldn't find “{term}”.", "text": "", "link": link,
            "linkText": where}


RUN = {"open_page": open_page, "compare": compare, "best_on": best_on, "chart": chart,
       "phone": phone, "boxes": boxes, "explain": explain, "clarify": clarify, "cant": cant,
       "not_found": not_found}


def then_label(r: dict) -> str:
    """a second request's button: "Open On-device models", "Q4 on GPQA Diamond" """
    say = str(r.get("say") or "").strip().rstrip(".:")
    return "Open " + say[len("Opened "):] if say.startswith("Opened ") else say


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
    sys_p, user = prompt(cat), user_message(q, prev, cat["today"], cat)
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
        action = resolve(a, cat, q, (prev or {}).get("q") or "", (prev or {}).get("action"))
        reply = run({k: v for k, v in action.items() if k != "then"}, cat)
        if isinstance(action.get("then"), dict):
            # the second request: worked out now, shown when its button is pressed
            second = run(action["then"], cat)
            reply["then"] = {"label": then_label(second), "action": action["then"],
                             "reply": second}
    log({"id": mid, "who": who, "q": q, "prev": prev or None, "raw": raw[:2000],
         "action": action, "kind": reply.get("kind"), "ms": ms, "error": err,
         "provider": "fake" if model is not None else (prov or provider()),
         "usage": usage or None})
    return {"id": mid, "action": action, "reply": reply, "ms": ms}


def thumbs(mid: str, who: str, up_: bool, note: str = "") -> None:
    log({"id": str(mid)[:40], "who": who, "thumbs": "up" if up_ else "down",
         "note": (note or "").strip()[:300]})


def thumbs_down() -> list[dict]:
    """the questions a thumbs down was last given to, with the turn before,
    the board's action and what the person wanted — the next bank's questions
    (28f)"""
    try:
        lines = [json.loads(x) for x in log_path().read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    except (OSError, ValueError):
        return []
    asked = {x["id"]: x for x in lines if x.get("id") and "q" in x}
    last = {x["id"]: x for x in lines if x.get("id") and x.get("thumbs")}
    return [{"q": asked[i]["q"], "prev": asked[i].get("prev"), "action": asked[i].get("action"),
             "kind": asked[i].get("kind"), "note": t.get("note") or "", "who": t.get("who"),
             "at": t.get("at")}
            for i, t in last.items() if t["thumbs"] == "down" and i in asked]


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
