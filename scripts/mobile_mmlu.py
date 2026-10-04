"""14.3: Mobile-MMLU-Pro (MBZUAI's Mobile-MMLU, DMLR 2026) — four-option
questions about everyday phone topics, in 80 fields such as first aid,
travel planning, cooking and tech help — with our own answer key. 14.4: and
the full Mobile-MMLU beside it.

**The two sets.** Mobile-MMLU-Pro (9,497 questions) is CC BY-ND 4.0: its
scores may be used, and nothing derived from it is published. It stays the
default: its scores are the ones that may leave the board. The full
Mobile-MMLU (16,186 questions, every Pro question among them) is CC BY-NC-ND
4.0. 14.3 left it out for that; on 4 Oct 2026 masein decided to add it for
internal research evaluation only, labelled "Non-commercial" wherever the set
or its scores appear (eval_tasks/mobile_mmlu/manifest.json records the
decision). Neither file is ever committed — the mirror is public, and our key
is a derivative. The deploy's data step (scripts/fetch_data.py) fetches each
from the revision its manifest pins, checks every file's sha256, and keeps
them in the server's data folder (MMP_DIR, MMF_DIR); the key lives beside
Pro's, in key/. Tests use invented rows with the same columns.

**Why our own key.** The authors hold the answers back and score through
their portal. Ours comes from strong models of different makers agreeing
(service/mmp_key.py asks them; `decide` is the rule):
- the first two labellers answer every question on their own: one letter,
  and whether the right answer depends on a current app version, price or
  date ("depends on now");
- both say it depends on now: the question is dropped (time-sensitive);
- they agree: their letter is the key;
- they differ: a third labeller, from a third maker, answers. A letter two of
  the three chose is the key; three different letters drop it (split) — the
  authors dropped what three models disagreed on, too.
Each question's labels, decision and reason are kept (key/key.json).

**How a model is scored.** 0-shot multiple choice, as the board asks MMLU:
the field's line, the question, the options lettered, "Answer:".
- An HF model (lm_eval) and a GGUF (llama-perplexity) pick the letter with
  the highest log-likelihood. No chat template, as the paper ran
  lm-evaluation-harness.
- A served or OpenRouter model answers with a letter, asked the authors' own
  way (their generate_answers.py prompt, word for word).
Every pick is kept (mobile_mmlu_pro.json), and scored against the key as it
stands: accuracy on the kept questions, with n, overall and in the paper's 9
categories. The key is "provisional" until three models from the paper's
Table 2, run as the paper ran them, land within 3 points of its numbers.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = REPO / "eval_tasks" / "mobile_mmlu_pro"
TEMPLATE = DATA_DIR / "_mmp_template_yaml"
TASK = "mobile_mmlu_pro"
NAME = "Mobile-MMLU-Pro"
LETTERS = "ABCD"
CHARS_PER_TOKEN = 4
PRED_FILE = "mobile_mmlu_pro.json"
MISSING = ("Mobile-MMLU-Pro isn't on this server: fetch it with the data step "
           "(scripts/fetch_data.py)")

# the paper's 9 categories over its 80 fields (arXiv 2503.20786, Appendix D,
# Table 11), each field as the file spells it ("strarigies" included)
CATEGORIES = {
    "Academic & Learning": (
        "elementary_mathematics", "high_school_mathematics", "basic_statistics",
        "conceptual_physics", "science_fundamentals", "formal_logic", "logical_fallacies",
        "education_techniques", "reading_and_literature", "writing_skills", "linguistics",
        "social_sciences", "political_systems", "world_history", "geography"),
    "Business & Career": (
        "project_management", "human_resources", "business_management",
        "marketing__and_sales_strarigies", "personal_finance", "e_commerce", "shopping",
        "accounting", "communication_and_public_speaking", "social_etiquette", "public_speaking"),
    "Technology & Digital": (
        "digital_literacy", "technical_help", "mobile_customization", "cybersecurity",
        "online_privacy", "social_media", "digital_detox"),
    "Health & Safety": (
        "mental_health", "physical_fitness", "medical_and_health_knowledge", "ergonomics",
        "first_aid", "outdoor_survival_skills", "automotive_care"),
    "Lifestyle & Personal": (
        "basic_life_skills", "time_management", "conflict_resolution", "event_planning",
        "creativity", "emotional_intelligence", "personal_branding", "career_development",
        "fashion_and_style", "travel_planning", "sports", "gardening_and_horticulture",
        "entertainment", "movie_and_tv_show", "podcasting", "hobbies",
        "photography_basics_and_smartphone_photography"),
    "Home & Family": (
        "home_safety", "pet_care", "waste_management", "home_maintenance", "parenting",
        "relationships", "teens_and_youth", "cooking_and_recipes", "food_safety",
        "nutrition_and_diet"),
    "Culture & Society": (
        "art_techniques_and_architecture", "interior_design", "cultural_awareness",
        "religious_studies", "holidays_and_traditions", "legal_rights", "law", "ethical_living",
        "ethics"),
    "Environment": ("environmental_and_sustainable_living", "weather_forecasting"),
    "Miscellaneous": ("global_facts", "news_and_information"),
}
FIELD_CATEGORY = {f: c for c, fs in CATEGORIES.items() for f in fs}
OTHER = "Other"


def manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def data_dir() -> Path:
    """the server's data folder for Mobile-MMLU-Pro and our key: never the repo"""
    if os.environ.get("MMP_DIR"):
        return Path(os.environ["MMP_DIR"])
    try:
        from service import config
        return Path(config.MMP_DIR)
    except ImportError:
        return REPO / "data" / "mobile_mmlu_pro"


def csv_path() -> Path:
    return data_dir() / manifest()["file"]["file"]


def key_dir() -> Path:
    return data_dir() / "key"


def available() -> str:
    """'' when the pinned file is on this server; else why not, in one line"""
    p, f = csv_path(), manifest()["file"]
    return "" if p.is_file() and p.stat().st_size == f["bytes"] else MISSING


_rows: dict = {}


def load() -> list[dict]:
    """every question — {id, question, A–D, field, category} — read from the
    fetched file once its sha256 is the pinned one; [] when it isn't here"""
    if available():
        return []
    p = csv_path()
    st = p.stat()
    k = (str(p), st.st_mtime_ns, st.st_size)
    if _rows.get("k") != k:
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        want = manifest()["file"]["sha256"]
        if sha != want:
            raise ValueError(f"{p} isn't the pinned file (sha256 {sha[:12]}, pinned "
                             f"{want[:12]}): fetch it again with the data step")
        csv.field_size_limit(10 ** 9)
        rows = []
        for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
            f = (r.get("Field") or "").strip()
            rows.append({"id": r["question_id"].strip(), "question": r["Question"],
                         **{L: r[L] for L in LETTERS}, "field": f,
                         "category": FIELD_CATEGORY.get(f, OTHER)})
        _rows.update(k=k, rows=rows)
    return _rows["rows"]


def by_id() -> dict[str, dict]:
    return {q["id"]: q for q in load()}


# ---------------------------------------------------------------------------
# 14.4: the full Mobile-MMLU — 80 files, one a field
# ---------------------------------------------------------------------------

FULL_DATA_DIR = REPO / "eval_tasks" / "mobile_mmlu"
FULL_TASK = "mobile_mmlu_full"
FULL_NAME = "Mobile-MMLU (full)"
FULL_PRED_FILE = "mobile_mmlu_full.json"
FULL_MISSING = ("Mobile-MMLU (full) isn't on this server: fetch it with the data step "
                "(scripts/fetch_data.py)")
# 14.4.5: MOBILE_MMLU_FULL=0 in .env hides the full set everywhere and refuses
# new runs of it; its files, picks and labels stay on disk
FULL_OFF = "Mobile-MMLU (full) is switched off on this server (MOBILE_MMLU_FULL=0 in .env)"


def full_on() -> bool:
    """the switch: on unless MOBILE_MMLU_FULL is 0 (the service's config, or
    the environment where there is none)"""
    try:
        from service import config
        return bool(config.MOBILE_MMLU_FULL)
    except ImportError:
        return os.environ.get("MOBILE_MMLU_FULL", "1").strip().lower() not in OFF_WORDS


OFF_WORDS = ("0", "no", "off", "false")


def full_manifest() -> dict:
    return json.loads((FULL_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def full_dir() -> Path:
    """the server's data folder for the full Mobile-MMLU: never the repo"""
    if os.environ.get("MMF_DIR"):
        return Path(os.environ["MMF_DIR"])
    try:
        from service import config
        return Path(config.MMF_DIR)
    except ImportError:
        return REPO / "data" / "mobile_mmlu"


def full_files() -> list[tuple[dict, Path]]:
    """each pinned file and where the data step keeps it"""
    return [(f, full_dir() / f["file"]) for f in full_manifest()["files"]]


def full_available() -> str:
    """'' when every pinned file of the full set is on this server; else why
    not, in one line. 14.4.5: switched off, it isn't: everything that asks
    this hides the set or refuses it"""
    return FULL_OFF if not full_on() else _full_missing()


def _full_missing() -> str:
    ok = all(p.is_file() and p.stat().st_size == f["bytes"] for f, p in full_files())
    return "" if ok else FULL_MISSING


_full_rows: dict = {}


def load_full() -> list[dict]:
    """every question of the full set — {id, question, A–D, field, category},
    its field from its file's name — once each file's sha256 is the pinned
    one; [] when the set isn't here (14.4.5: or is switched off)"""
    return _read_full() if full_on() else []


def _read_full() -> list[dict]:
    """the full set's rows whether or not it is switched off: the one key
    keeps its labels either way (pool), so switching it back shows them"""
    if _full_missing():
        return []
    files = full_files()
    k = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for _, p in files)
    if _full_rows.get("k") != k:
        csv.field_size_limit(10 ** 9)
        rows = []
        for f, p in files:
            raw = p.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if sha != f["sha256"]:
                raise ValueError(f"{p} isn't the pinned file (sha256 {sha[:12]}, pinned "
                                 f"{f['sha256'][:12]}): fetch it again with the data step")
            field = Path(f["file"]).stem
            for r in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
                rows.append({"id": r["question_id"].strip(), "question": r["Question"],
                             **{L: r[L] for L in LETTERS}, "field": field,
                             "category": FIELD_CATEGORY.get(field, OTHER)})
        _full_rows.update(k=k, rows=rows)
    return _full_rows["rows"]


def full_by_id() -> dict[str, dict]:
    return {q["id"]: q for q in load_full()}


def overlap(pro: list[dict] | None = None, full: list[dict] | None = None) -> dict:
    """how Pro sits in the full set, by question id and by text: {pro, full,
    pro_in_full, pro_not_in_full, same_wording, worded_apart: [ids],
    same_options, full_only}"""
    pro = load() if pro is None else pro
    full = load_full() if full is None else full
    F = {q["id"]: q for q in full}
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()      # noqa: E731
    both = [q for q in pro if q["id"] in F]
    apart = sorted(q["id"] for q in both if norm(q["question"]) != norm(F[q["id"]]["question"]))
    return {"pro": len(pro), "full": len(full), "pro_in_full": len(both),
            "pro_not_in_full": sorted(q["id"] for q in pro if q["id"] not in F),
            "same_wording": len(both) - len(apart), "worded_apart": apart,
            "same_options": sum(1 for q in both
                                if [q[L] for L in LETTERS] == [F[q["id"]][L] for L in LETTERS]),
            "full_only": len(set(F) - {q["id"] for q in pro})}


def credit() -> dict:
    m = manifest()
    return {"name": NAME, "by": m["by"], "licence": m["licence"], "url": m["source"],
            "revision": m["revision"], "note": m["licence_note"], "answers": m["answers"],
            "portal": m["portal"], "n": m["file"]["n"]}


# ---------------------------------------------------------------------------
# how a question is asked
# ---------------------------------------------------------------------------

MMLU_HEAD = "The following are multiple choice questions (with answers) about {subject}.\n\n"


def subject(field: str) -> str:
    """"marketing__and_sales_strarigies" -> "marketing and sales strarigies" """
    return " ".join(w for w in field.split("_") if w)


def mc_prompt(q: dict) -> str:
    """as lm_eval's mmlu asks it, 0-shot (gguf_data.mmlu_task's text): the
    field's line, the question stripped, each option after its letter, and
    "Answer:" — after which each letter is scored"""
    return (MMLU_HEAD.format(subject=subject(q["field"])) + q["question"].strip() + "\n"
            + "".join(f"{L}. {q[L]}\n" for L in LETTERS) + "Answer:")


# the authors' own prompt (VILA-Lab/Mobile-MMLU, generate_answers.py,
# format_prompt), word for word — its full-width colon too: what a served or
# OpenRouter model is asked, and answers with a letter
ASK_HEAD = ("Answer the following multiple choice question and provide only letter of the "
            "correct answer.\n")


def ask_prompt(q: dict) -> str:
    return (ASK_HEAD + f"{q['question'].strip()}\n" + "".join(f"{L}. {q[L]}\n" for L in LETTERS)
            + "Answer：\n")


_THINK = re.compile(r"(?s)^.*</think>")
_FIRST = re.compile(r"^[\s*_(\[]*([ABCD])(?:$|[\s]*$|[.):\]*,]|\s*\n)")
_SAID = re.compile(r"(?:[Aa]nswer|[Oo]ption|[Cc]hoice)\s*(?:is|:|：)?\s*[*(\[]*([ABCD])(?![A-Za-z])")
_ALONE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
_WORD_AFTER = re.compile(r"\s+[a-z]")


def letter_of(text) -> str | None:
    """the letter a reply gives — "B", "B.", "(B)", "**B**", "The answer is
    B" — after any thinking; None when it gives none, or several"""
    t = _THINK.sub("", str(text or "")).strip()
    if not t:
        return None
    m = _FIRST.match(t) or _SAID.search(t)
    if m:
        return m.group(1)
    found = list(_ALONE.finditer(t))
    if len({x.group(1) for x in found}) != 1:
        return None
    # "A" before a word is the article, not an answer
    return None if all(_WORD_AFTER.match(t, x.end()) for x in found) else found[0].group(1)


# ---------------------------------------------------------------------------
# the labellers' question, and their answers
# ---------------------------------------------------------------------------

LABEL_INSTRUCTION = (
    'Reply with JSON only, as {"answer": "A", "depends_on_now": false}. "answer" is the letter '
    'of the right option: A, B, C or D. "depends_on_now" is true only when the right answer '
    "depends on a current app version, price or date.")
LABEL_MAX_TOKENS = 4000


def label_prompt(q: dict) -> str:
    return (f"{q['question'].strip()}\n\n" + "".join(f"{L}. {q[L]}\n" for L in LETTERS) + "\n"
            + LABEL_INSTRUCTION)


def _last_object(text: str) -> dict | None:
    i = text.rfind("{")
    while i != -1:
        j = text.find("}", i)
        while j != -1:
            try:
                obj = json.loads(text[i:j + 1])
            except ValueError:
                j = text.find("}", j + 1)
                continue
            if isinstance(obj, dict):
                return obj
            break
        i = text.rfind("{", 0, i)
    return None


def parse_label(text) -> dict:
    """{letter, now}: a labeller's letter (None when it gives none) and its
    "depends on now" flag"""
    t = _THINK.sub("", str(text or "")).strip()
    obj = _last_object(t)
    if obj is not None:
        a = str(obj.get("answer") or "").strip().strip("*()[]. ").upper()
        now = obj.get("depends_on_now")
        now = now if isinstance(now, bool) else str(now).strip().lower() in ("true", "yes", "1")
        return {"letter": a if a in tuple(LETTERS) else None, "now": bool(now)}
    return {"letter": letter_of(t), "now": False}


# ---------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------

SLOTS = ("first", "second", "third")
DECIDED = {"agreed": "kept by agreement", "settled": "settled by the third labeller",
           "split": "dropped as split", "time": "dropped as time-sensitive",
           "waiting": "waiting for a labeller"}
KEPT = ("agreed", "settled")


def decide(a: dict | None, b: dict | None, c: dict | None = None) -> dict:
    """one question's key from its labels — a and b the first two labellers',
    c the third's — {key, decision, reason}; key None when it is dropped or
    still waits"""
    if not a or not b:
        return {"key": None, "decision": "waiting",
                "reason": "the first two labellers haven't both answered"}
    if a.get("now") and b.get("now"):
        return {"key": None, "decision": "time",
                "reason": "both first labellers say the right answer depends on now"}
    la, lb = a.get("letter"), b.get("letter")
    if la and la == lb:
        return {"key": la, "decision": "agreed", "reason": f"both chose {la}"}
    said = f"{la or 'no letter'} and {lb or 'no letter'}"
    if not c:
        return {"key": None, "decision": "waiting",
                "reason": f"the first two differ ({said}): the third labeller decides"}
    lc = c.get("letter")
    votes = [la, lb, lc]
    pick = next((L for L in LETTERS if votes.count(L) >= 2), None)
    if pick:
        return {"key": pick, "decision": "settled",
                "reason": f"the first two differ ({said}); the third chose {lc}: two of three"}
    return {"key": None, "decision": "split",
            "reason": f"all three differ ({said}, then {lc or 'no letter'})"}


# 14.4: one pool of labels for both sets, by question — its id and a hash of
# its wording, so a question the two sets word alike (all but one) is labelled
# once, and the one worded apart (9933ec55) twice, once in each wording

def lid(q: dict) -> str:
    """a question's label id: "9933ec55#a556794a" — its id and the first 8 hex
    of its question's sha256"""
    return f"{q['id']}#{hashlib.sha256(q['question'].encode('utf-8')).hexdigest()[:8]}"


def pool(pro: list[dict] | None = None, full: list[dict] | None = None) -> list[dict]:
    """every question either set needs labelled, once each: Pro's first, in
    its order, then the full set's that Pro hasn't (its own, and the one
    worded apart) — each with its "lid" and the sets it's in"""
    pro = load() if pro is None else pro
    full = _read_full() if full is None else full
    out: dict[str, dict] = {}
    for name, rows in (("pro", pro), ("full", full)):
        for q in rows:
            k = lid(q)
            if k in out:
                out[k]["sets"].append(name)
            else:
                out[k] = {**q, "lid": k, "sets": [name]}
    return list(out.values())


def build_key(rows: list[dict], labels: dict, current: dict) -> dict:
    """the key from every label: {items: {lid: {id, key, decision, reason,
    category, sets, labels}}, sets: {pro, full: {counts, version}}, labellers}.
    `rows` is the pool (each with its lid; a row without one is Pro's, as 14.3
    built it). `current` is each slot's labeller now ({slot: {id, version}}):
    only its labels count, so a labeller changed on AI models labels its slot
    again"""
    def lab(slot: str, k: str) -> dict | None:
        x = (labels.get(slot) or {}).get(k)
        cur = current.get(slot) or {}
        return x if x and cur.get("version") and x.get("version") == cur["version"] else None
    items = {}
    for q in rows:
        k = q.get("lid") or lid(q)
        got = {s: lab(s, k) for s in SLOTS}
        items[k] = {**decide(got["first"], got["second"], got["third"]), "id": q["id"],
                    "category": q["category"], "sets": list(q.get("sets") or ["pro"]),
                    "labels": {s: {"letter": x["letter"], "now": x["now"]}
                               for s, x in got.items() if x}}
    out = {"items": items, "pool": True, "labellers": current, "built_at": time.time()}
    out["sets"] = {name: {"counts": v["counts"], "version": v["version"]}
                   for name, v in ((n, set_view(out, n)) for n in SETS)}
    return out


SETS = ("pro", "full")


def set_view(pooled: dict, name: str) -> dict:
    """one set's key, from the pool: {items: {id: item}, counts, version,
    labellers, built_at} — as 14.3's key was, by question id. A key built
    before the pool (items by id, no "pool") is Pro's, as it stands"""
    if not pooled:
        return {}
    if not pooled.get("pool"):
        return pooled if name == "pro" else {}
    items = {it["id"]: it for it in (pooled.get("items") or {}).values()
             if name in (it.get("sets") or [])}
    return {"items": items, "counts": counts(items), "version": key_version(items),
            "labellers": pooled.get("labellers") or {}, "built_at": pooled.get("built_at")}


def counts(items: dict) -> dict:
    """how many were kept by agreement, settled by the third labeller, dropped
    as split, dropped as time-sensitive, and wait — overall and per category"""
    def blank() -> dict:
        return {**{d: 0 for d in DECIDED}, "questions": 0, "kept": 0}
    out, cats = blank(), {c: blank() for c in CATEGORIES}
    for it in items.values():
        for b in (out, cats.setdefault(it.get("category") or OTHER, blank())):
            b[it["decision"]] += 1
            b["questions"] += 1
            b["kept"] += it["decision"] in KEPT
    return {"all": out, "by_category": {c: v for c, v in cats.items() if v["questions"]}}


def agreement(it: dict) -> str:
    """how the labellers decided a question, in words: "kept by agreement
    (both chose B)" """
    if not it:
        return "no answer key yet"
    return f"{DECIDED[it['decision']]} ({it['reason']})"


def kept(items: dict) -> dict[str, str]:
    return {q: it["key"] for q, it in items.items() if it["decision"] in KEPT and it["key"]}


def key_version(items: dict) -> str:
    """a short hash of the kept questions and their letters: a score says
    which key it was scored on, and a GGUF dataset is rebuilt when it moves"""
    k = kept(items)
    if not k:
        return ""
    return hashlib.sha256(json.dumps(sorted(k.items())).encode()).hexdigest()[:10]


def _read(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def read_labels() -> dict:
    """every label kept, {slot: {lid: label}}. 14.4: labels kept by question
    id alone (before the pool) are Pro's, labelled in Pro's wording: moved to
    their label ids once, when Pro's file is here"""
    lab = _read(key_dir() / "labels.json", {})
    old = any("#" not in k for got in lab.values() for k in got)
    if old and not available():
        ids = {q["id"]: lid(q) for q in load()}
        lab = {slot: {(ids.get(k, k) if "#" not in k else k): v for k, v in got.items()}
               for slot, got in lab.items()}
        _write(key_dir() / "labels.json", lab)
    return lab


def add_labels(slot: str, got: dict[str, dict]) -> None:
    """a labeller's answers, kept by question id: {letter, now, model, version, at}"""
    lab = read_labels()
    lab.setdefault(slot, {}).update(got)
    _write(key_dir() / "labels.json", lab)


def read_key() -> dict:
    return _read(key_dir() / "key.json", {})


def write_key(k: dict) -> None:
    _write(key_dir() / "key.json", k)


_keyc: dict = {}


def pooled_key() -> dict:
    """the pool's key as last built, read once per change: {} before any label"""
    p = key_dir() / "key.json"
    try:
        st = p.stat()
    except OSError:
        return {}
    k = (st.st_mtime_ns, st.st_size)
    if _keyc.get("k") != k:
        pooled = read_key()
        _keyc.update(k=k, key=pooled, views={n: set_view(pooled, n) for n in SETS})
    return _keyc["key"]


def current_key() -> dict:
    """Mobile-MMLU-Pro's key, by question id (14.4: its view of the pool)"""
    pooled_key()
    return (_keyc.get("views") or {}).get("pro") or {}


def full_key() -> dict:
    """14.4: the full Mobile-MMLU's key, by question id — the pool's labels of
    its own wording"""
    pooled_key()
    return (_keyc.get("views") or {}).get("full") or {}


def gguf_docs(which: str = "pro") -> list[dict]:
    """the kept questions as lm_eval's mmlu docs — {question, choices,
    subject, answer} — for gguf_data.py: a GGUF is scored on the key as it
    stands when its dataset is built (its sha256 moves with the key). 14.4:
    `which` "full", the full set's, in its own wording, on its own key"""
    full = which == "full"
    key = full_key() if full else current_key()
    items = key.get("items") or {}
    name = FULL_NAME if full else NAME
    if not kept(items):
        raise ValueError(f"{name} has no answer key yet: build it on AI models first")
    why = available() or (full_available() if full else "")
    if why:
        raise ValueError(why)
    k = kept(items)
    return [{"id": q["id"], "question": q["question"], "choices": [q[L] for L in LETTERS],
             "subject": subject(q["field"]), "answer": LETTERS.index(k[q["id"]])}
            for q in (load_full() if full else load()) if q["id"] in k]


# ---------------------------------------------------------------------------
# the run's task, and what a model picked
# ---------------------------------------------------------------------------

def build_tasks(dest: Path) -> Path:
    """the task lm_eval runs, under `dest`: every question's prompt (never a
    key) and the yaml; and the authors' prompts a served model is asked. A
    stand-in with no questions when the file isn't here, so step 4 finds the
    task. Returns the directory for --include_path"""
    dest.mkdir(parents=True, exist_ok=True)
    rows = load()
    items, ask, stamp = dest / f"{TASK}.jsonl", dest / f"{TASK}_ask.jsonl", dest / f"{TASK}.sha256"
    want = manifest()["file"]["sha256"] if rows else "none"
    if not (items.exists() and ask.exists() and stamp.exists()
            and stamp.read_text(encoding="utf-8").strip() == want):
        for path, how in ((items, mc_prompt), (ask, ask_prompt)):
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("".join(json.dumps({"id": q["id"], "field": q["field"],
                                               "prompt": how(q)}, ensure_ascii=False) + "\n"
                                   for q in rows), encoding="utf-8")
            tmp.replace(path)
        stamp.write_text(want + "\n", encoding="utf-8")
    (dest / f"{TASK}.yaml").write_text(
        f"task: {TASK}\n" + TEMPLATE.read_text(encoding="utf-8")
        .replace("__ITEMS_PATH__", str(items.resolve())), encoding="utf-8")
    return dest


def build_full_tasks(dest: Path) -> Path:
    """14.4: the full run's task, under `dest` — every question the full set
    or Pro asks, once each (the pool: the full set's wording, and Pro's of a
    question worded apart, so that one run scores both sets on their own),
    never a key — and the authors' prompts a served model is asked. A
    model's run asks only what it hasn't answered (full_left). Returns the
    directory for --include_path"""
    dest.mkdir(parents=True, exist_ok=True)
    rows = pool()
    items, ask = dest / f"{FULL_TASK}.jsonl", dest / f"{FULL_TASK}_ask.jsonl"
    stamp = dest / f"{FULL_TASK}.sha256"
    want = full_sha() if load_full() else "none"
    if not (items.exists() and ask.exists() and stamp.exists()
            and stamp.read_text(encoding="utf-8").strip() == want):
        for path, how in ((items, mc_prompt), (ask, ask_prompt)):
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("".join(json.dumps({"id": q["id"], "lid": q["lid"], "field": q["field"],
                                               "prompt": how(q)}, ensure_ascii=False) + "\n"
                                   for q in rows), encoding="utf-8")
            tmp.replace(path)
        stamp.write_text(want + "\n", encoding="utf-8")
    (dest / f"{FULL_TASK}.yaml").write_text(
        f"task: {FULL_TASK}\n" + TEMPLATE.read_text(encoding="utf-8")
        .replace("__ITEMS_PATH__", str(items.resolve())), encoding="utf-8")
    return dest


def full_left(model_dir: Path | None) -> list[int]:
    """14.4: the full run's questions this model hasn't answered, as indices
    into the task's rows (the pool's order) — every one for a model with no
    picks; a Pro run's count, so a later full run asks only the rest"""
    return to_ask(model_dir, FULL_TASK)


def to_ask(model_dir: Path | None, task: str = TASK) -> list[int]:
    """14.4: the questions a run of `task` would ask this model — those it
    has no pick for, as indices into the task's rows (Pro's or the pool's
    order). One run, two scores: a full run's picks are Pro's too, so a Pro
    run after it asks none, and a full run after a Pro run only the rest"""
    rows = pool() if task == FULL_TASK else [{**q, "lid": lid(q)} for q in load()]
    have = picks(model_dir)[0] if model_dir and Path(model_dir).is_dir() else {}
    return [i for i, q in enumerate(rows) if q["lid"] not in have]


def samples_file(task_out: Path, task: str, todo: list[int]) -> Path | None:
    """lm_eval's --samples for a run that asks only some of `task`'s
    questions: {task: indices}, beside its answers; None when it asks all"""
    if len(todo) == len(pool() if task == FULL_TASK else load()):
        return None
    task_out.mkdir(parents=True, exist_ok=True)
    path = task_out / "left.json"
    path.write_text(json.dumps({task: todo}), encoding="utf-8")
    return path


def _pick(rec: dict) -> tuple[str | None, str]:
    """(letter, how): the highest log-likelihood of four, or a reply's letter"""
    fr = rec.get("filtered_resps")
    if isinstance(fr, list) and len(fr) == len(LETTERS) and all(
            isinstance(x, (list, tuple, int, float)) for x in fr):
        lls = [float(x[0] if isinstance(x, (list, tuple)) else x) for x in fr]
        return LETTERS[max(range(len(lls)), key=lls.__getitem__)], "log-likelihood"
    text = fr[0] if isinstance(fr, list) and fr else ((rec.get("resps") or [[""]])[0] or [""])[0]
    return letter_of(text), "letter"


def _samples(model_dir: Path, task: str) -> list[Path]:
    return sorted(Path(model_dir).glob(f"{task}_*shot/**/samples_{task}_*.jsonl"),
                  key=lambda p: p.stat().st_mtime)


def picks(model_dir: Path) -> tuple[dict[str, str | None], set[str]]:
    """14.4: every pick the model has made, by label id — from its two pick
    files, then the samples its runs wrote (later over earlier): a Pro run's
    by question id (Pro's wording), a full run's by label id — and how they
    were picked"""
    model_dir = Path(model_dir)
    pro_lid = {q["id"]: lid(q) for q in load()}
    full_lid = {q["id"]: lid(q) for q in load_full()}
    got: dict[str, str | None] = {}
    hows: set[str] = set()
    for f, ids in ((PRED_FILE, pro_lid), (FULL_PRED_FILE, full_lid)):
        old = _read(model_dir / f, None) or {}
        for qid, v in (old.get("predictions") or {}).items():
            if qid in ids:
                got[ids[qid]] = v
        if old.get("how"):
            hows.add(old["how"])
    for task in (TASK, FULL_TASK):
        for f in _samples(model_dir, task):
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                doc = rec.get("doc") or {}
                k = doc.get("lid") or pro_lid.get(doc.get("id"))
                if k:
                    got[k], how = _pick(rec)
                    hows.add(how)
    return got, hows


def _pred_file(rows: list[dict], got: dict, hows: set[str], sha: str) -> dict:
    preds = {q["id"]: got[lid(q)] for q in rows if lid(q) in got}
    return {"how": "log-likelihood" if hows == {"log-likelihood"} else "letter",
            "predictions": preds, "n": len(preds),
            "unreadable": sum(1 for v in preds.values() if v is None),
            "data_sha256": sha, "at": time.time()}


def full_sha() -> str:
    """one hash of the full set as pinned: its 80 files' and Pro's"""
    return hashlib.sha256(json.dumps([manifest()["file"]["sha256"]]
                                     + [f["sha256"] for f in full_manifest()["files"]])
                          .encode()).hexdigest()


def collect(model_dir: Path) -> dict | None:
    """every pick of the model's runs, from the samples they wrote (lm_eval's
    or a served model's, later ones over earlier), into mobile_mmlu_pro.json:
    {how, predictions: {id: letter or None}, n, unreadable, data_sha256, at}.
    14.4: one run, two scores — the picks are kept by question and wording, so
    a full run's picks give Pro's score too (each set on its own wording), and
    a Pro run's count towards a later full run. Once the model has sat the
    full set, mobile_mmlu_full.json too. Returns Pro's"""
    model_dir = Path(model_dir)
    if not (_samples(model_dir, TASK) or _samples(model_dir, FULL_TASK)):
        return None
    got, hows = picks(model_dir)
    pro = _pred_file(load(), got, hows, manifest()["file"]["sha256"])
    if pro["n"]:
        _write(model_dir / PRED_FILE, pro)
    full = load_full()
    if full and (_samples(model_dir, FULL_TASK) or (model_dir / FULL_PRED_FILE).exists()):
        _write(model_dir / FULL_PRED_FILE, _pred_file(full, got, hows, full_sha()))
    return pro if pro["n"] else None


def predictions(model_dir: Path) -> dict | None:
    return _read(Path(model_dir) / PRED_FILE, None)


def full_predictions(model_dir: Path) -> dict | None:
    """14.4: the model's picks on the full set's wording"""
    return _read(Path(model_dir) / FULL_PRED_FILE, None)


def score(preds: dict | None, key: dict | None = None) -> dict | None:
    """the model's accuracy on the kept questions it answered, with n, overall
    and per category, on the key as it stands — None before any key or pick"""
    key = current_key() if key is None else key
    items = key.get("items") or {}
    k = kept(items)
    picks = (preds or {}).get("predictions") or {}
    got = {q: picks[q] for q in k if q in picks}
    if not k or not got:
        return None

    def part(qs: list[str]) -> dict:
        n = len(qs)
        c = sum(1 for q in qs if got[q] == k[q])
        p = c / n if n else None
        return {"acc": p, "n": n, "correct": c,
                "se": math.sqrt(p * (1 - p) / n) if n and p is not None else None}
    cats: dict[str, list[str]] = {}
    for q in got:
        cats.setdefault(items[q].get("category") or OTHER, []).append(q)
    # 14.4: and its 80 fields
    fields: dict[str, list[str]] = {}
    field_of = {q: f for f, qs in _fields_of(items).items() for q in qs}
    for q in got:
        fields.setdefault(field_of.get(q) or OTHER, []).append(q)
    return {**part(list(got)), "of": len(k), "unreadable": sum(1 for v in got.values() if v is None),
            "how": (preds or {}).get("how"), "key": key.get("version") or "",
            "by_category": {c: part(cats[c]) for c in CATEGORIES if c in cats},
            "by_field": {f: part(fields[f]) for f in sorted(fields)}}


def _fields_of(items: dict) -> dict[str, list[str]]:
    """each field's questions, as the sets' rows say (a key's items carry
    only their category)"""
    out: dict[str, list[str]] = {}
    for q in (*load(), *load_full()):
        if q["id"] in items:
            ids = out.setdefault(q["field"], [])
            if q["id"] not in ids:
                ids.append(q["id"])
    return out


def summary(s: dict | None, name: str = NAME) -> str:
    if not s:
        return f"{name}: answered; scored once the answer key is built"
    return f"{name} {100 * s['acc']:.1f}% on {s['n']:,} kept questions"


# ---------------------------------------------------------------------------
# checking the key against the paper, and the portal's file
# ---------------------------------------------------------------------------

def paper_checks(ours: dict[str, dict | None], which: str = "pro") -> dict:
    """ours beside the paper's for its three models: {rows: [{model, paper,
    ours, n, diff, ok}], provisional, within, setting}. The key stays
    provisional until all three land within 3 points. 14.4: `which` set —
    the full set's checks are Table 2's Mobile-MMLU column"""
    pc = (manifest() if which == "pro" else full_manifest())["paper_checks"]
    rows = []
    for model, paper in pc["models"].items():
        s = ours.get(model)
        v = round(100 * s["acc"], 1) if s and s.get("acc") is not None else None
        diff = round(v - paper, 1) if v is not None else None
        rows.append({"model": model, "paper": paper, "ours": v, "n": (s or {}).get("n"),
                     "diff": diff, "ok": diff is not None and abs(diff) <= pc["within"]})
    return {"rows": rows, "provisional": not all(r["ok"] for r in rows),
            "within": pc["within"], "setting": pc["setting"], "table": pc.get("table")}


def portal_scores() -> dict:
    """the scores the authors' portal gave, as masein typed them: {model:
    {score, by, at}}"""
    return _read(key_dir() / "portal.json", {})


def set_portal_score(model: str, score: float, by: str) -> None:
    got = portal_scores()
    got[model] = {"score": score, "by": by, "at": time.time()}
    _write(key_dir() / "portal.json", got)


def portal_csv(preds: dict) -> str:
    """the authors' portal's format: question_id,predicted_answer — the
    model's picks, never our key"""
    rows = [(q, L) for q, L in ((preds or {}).get("predictions") or {}).items() if L]
    return "question_id,predicted_answer\n" + "".join(f"{q},{L}\n" for q, L in rows)


# ---------------------------------------------------------------------------
# what it takes, before Start
# ---------------------------------------------------------------------------

# each labeller's reply, thinking included, about: the mean completion tokens
# of GPT-6 Sol's and Gemini 3.1 Pro's 343 validation calls each in the
# reasoning lab (reasoning-lab data/validation/verdicts.jsonl, medium effort);
# a maker with none on record, a guess
OUT_GUESS = {"openai": 230, "google": 620}
OUT_DEFAULT = 400
# the share of questions the first two split on, before they have answered:
# a guess — the authors kept only questions three frontier models agreed on
THIRD_SHARE = 0.10


def _stats_rows(which: str = "pro") -> list[dict]:
    """a question of the published mean lengths, for an estimate with no file
    (14.4: Pro's, or the full set's for "full" and "all")"""
    mc = (manifest() if which == "pro" else full_manifest())["mean_chars"]
    return [{"id": "", "question": "x" * round(mc["Question"]),
             **{L: "x" * round(mc[L]) for L in LETTERS}, "field": "global_facts"}]


def tokens_in(how, rows: list[dict] | None = None) -> float:
    """the mean prompt tokens of one question, asked `how` (a prompt
    function): over the file, or the published mean lengths"""
    rows = rows or _stats_rows()
    return sum(len(how(q)) for q in rows) / len(rows) / CHARS_PER_TOKEN


def out_guess(model_id: str) -> int:
    return OUT_GUESS.get((model_id or "").split("/")[0].lower(), OUT_DEFAULT)


def _stats_n(which: str) -> int:
    """how many questions a set asks to label, without its files: Pro's; the
    full set's; or both, once each — the full set and Pro's wording apart"""
    if which == "pro":
        return manifest()["file"]["n"]
    fm = full_manifest()
    return fm["n"] + (len(fm["overlap"]["worded_apart"]) if which == "all" else 0)


def label_estimate(labellers: dict, left: dict | None = None, rows: list[dict] | None = None,
                   guess_third: bool | None = None, which: str = "pro") -> dict:
    """the dry run: each labeller's questions, tokens in and out, and cost —
    {slots: {slot: {id, name, questions, tokens_in, tokens_out, usd, …}}, usd}.
    `labellers`: {slot: {id, name, price_in, price_out}}; `left`: each slot's
    questions still to label (every one, with none given). The third answers
    only where the first two split: while they have questions left, its count
    is THIRD_SHARE of theirs, a guess"""
    n = len(rows) if rows else _stats_n(which)
    each_in = tokens_in(label_prompt, rows or _stats_rows(which))
    left = dict(left or {"first": n, "second": n, "third": 0})
    guess_third = True if guess_third is None else guess_third
    if guess_third:
        left["third"] = int(left.get("third") or 0) + round(
            THIRD_SHARE * max(int(left.get("first") or 0), int(left.get("second") or 0)))
    out, total, known = {}, 0.0, True
    for slot in SLOTS:
        lab = labellers.get(slot) or {}
        q = int(left.get(slot) or 0)
        oe = out_guess(lab.get("id", ""))
        tin, tout = round(q * each_in), q * oe
        pin, pout = lab.get("price_in"), lab.get("price_out")
        usd = (pin * tin + pout * tout) / 1e6 if pin is not None and pout is not None else None
        known = known and (usd is not None or q == 0)
        total += usd or 0.0
        out[slot] = {"id": lab.get("id", ""), "name": lab.get("name") or lab.get("id", ""),
                     "questions": q, "tokens_in": tin, "tokens_out": tout, "usd": usd,
                     "price_in": pin, "price_out": pout, "out_each": oe,
                     "in_each": round(each_in), "guess": slot == "third" and guess_third}
    return {"slots": out, "usd": round(total, 2) if known else None,
            "usd_known": round(total, 2), "third_share": THIRD_SHARE if guess_third else None,
            "questions": n, "which": which}


SERVED_GUESS_S = 5.0     # as MobileAIBench's estimate guesses a served answer
HF_GUESS_S = 0.1         # four letters' log-likelihoods after one prompt, on the GPU


def _dur(seconds: float) -> str:
    return f"about {seconds / 3600:.1f} h" if seconds >= 5400 else \
        f"about {max(1, round(seconds / 60))} min"


def run_estimate(served: bool, secs_each: float | None = None, which: str = "pro",
                 model_dir: Path | None = None) -> dict:
    """what a model's Mobile-MMLU-Pro run takes: {questions, answers,
    judgements, seconds, line, missing}. 14.4: `which` "full", the full set's
    run. With `model_dir`, only the questions this model has no pick for —
    one run, two scores: a Pro run counts towards the full set's, and a full
    run's picks are Pro's too: {of, reused} say so"""
    full = which == "full"
    missing = available() or (full_available() if full else "")
    rows = [] if missing else pool() if full else load()
    total = len(rows) or (_stats_n("all") if full else manifest()["file"]["n"])
    n = len(to_ask(model_dir, FULL_TASK if full else TASK)) if rows and model_dir else total
    each = secs_each if secs_each and secs_each > 0 else (SERVED_GUESS_S if served else HF_GUESS_S)
    sec = n * each
    sample = rows or (_stats_rows("full") if full else None)
    reused = total - n
    by = _sat(model_dir) if reused else []
    by_words = (f"its {by[0]} run" if len(by) == 1 else "its Mobile-MMLU runs")
    line = (f"nothing to ask: all {total:,} answered already, by {by_words}" if not n
            else f"{n:,} answers, {_dur(sec)}" + ("" if secs_each and secs_each > 0
                                                  else ", a rough guess")
            + (f" · {reused:,} answered already, by {by_words}" if reused else ""))
    return {"questions": n, "prompts": n, "answers": n, "judgements": 0, "seconds": round(sec),
            "each": each, "measured": bool(secs_each and secs_each > 0), "of": total,
            "reused": reused,
            "tokens_in": round(n * tokens_in(ask_prompt if served else mc_prompt, sample)),
            "tokens_out": n * 2, "line": line, "missing": missing}


def _sat(model_dir: Path | None) -> list[str]:
    """which of the two sets this model has answers to: their names"""
    d = Path(model_dir) if model_dir else None
    if not d or not d.is_dir():
        return []
    return [name for name, task, f in ((NAME, TASK, PRED_FILE), (FULL_NAME, FULL_TASK,
                                                                  FULL_PRED_FILE))
            if (d / f).exists() or _samples(d, task)]


# ---------------------------------------------------------------------------
# the command line, on the server
# ---------------------------------------------------------------------------

SET_WORDS = {"pro": "Mobile-MMLU-Pro alone",
             "full": "the full Mobile-MMLU (Non-commercial: internal research only)",
             "all": "both, once each — what Start sends, Pro's questions first"}


def dry_run_lines(est: dict, source: str) -> list[str]:
    """the dry run, each set apart (14.4: Pro alone, the full set, and both)"""
    sets = est.get("sets") or {"pro": est}
    out = [f"Labelling the Mobile-MMLU answer key — a dry run, nothing is sent ({source}):"]
    for name in ("pro", "full", "all"):
        if name in sets:
            out += [f"{SET_WORDS[name]}:"] + ["  " + x for x in _est_lines(sets[name])]
    return out


def _est_lines(est: dict) -> list[str]:
    out = []
    for slot, s in est["slots"].items():
        cost = f"${s['usd']:,.2f}" if s["usd"] is not None else "price known once it is pinned"
        price = (f" at ${s['price_in']:g} in / ${s['price_out']:g} out per million"
                 if s["price_in"] is not None else "")
        out.append(f"  {slot:<6} {s['name']}: {s['questions']:,} questions · "
                   f"{s['tokens_in']:,} tokens in (about {s['in_each']} each) · "
                   f"{s['tokens_out']:,} out (about {s['out_each']} each, thinking included) · "
                   f"{cost}{price}")
    out.append("  total: " + (f"${est['usd']:,.2f}" if est["usd"] is not None
                               else f"${est['usd_known']:,.2f} and the unpriced"))
    if est.get("third_share"):
        out.append(f"  (the third answers only where the first two differ: "
                   f"{int(100 * est['third_share'])}% is a guess until they have)")
    return out


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Mobile-MMLU-Pro's answer key")
    ap.add_argument("--dry-run", action="store_true",
                    help="each labeller's tokens and cost; nothing is sent")
    ap.add_argument("--stats", action="store_true",
                    help="estimate from the published mean lengths, without the file")
    ap.add_argument("--key", action="store_true", help="the key's counts, overall and per category")
    ap.add_argument("--overlap", action="store_true",
                    help="14.4: how Pro sits in the full set, by question id and by text")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    if a.overlap:
        why = available() or full_available()
        if why:
            print(why)
            return 1
        o = overlap()
        print(f"Pro {o['pro']:,} · full {o['full']:,} · Pro's in the full set {o['pro_in_full']:,}"
              f" (not: {len(o['pro_not_in_full'])}) · worded the same {o['same_wording']:,} · "
              f"worded apart {', '.join(o['worded_apart']) or 'none'} · same options "
              f"{o['same_options']:,} · only in the full set {o['full_only']:,}")
        return 0
    if a.dry_run:
        try:
            from service import mmp_key
            est = mmp_key.estimate(stats=a.stats)
        except ImportError:
            from_defaults = {s: {"id": d["id"], "name": d["name"], "price_in": d.get("price_in"),
                                 "price_out": d.get("price_out")}
                             for s, d in DEFAULT_LABELLERS.items()}
            est = {"sets": {w: label_estimate(from_defaults, which=w) for w in ("pro", "full",
                                                                                 "all")}}
        src = ("the published mean lengths" if a.stats or available()
               else f"{len(load()):,} questions in {csv_path()}"
               + ("" if full_available() else "; the full set's published lengths")
               + ("" if full_available() else "" if not load_full() else
                  f", {len(load_full()):,} in {full_dir()}"))
        print("\n".join(dry_run_lines(est, src)))
        return 0
    if a.key:
        c = (current_key().get("counts") or {})
        if not c:
            print("no key yet")
            return 0
        for name, b in [("all", c["all"]), *c["by_category"].items()]:
            print(f"{name}: {b['questions']:,} questions, {b['kept']:,} kept — "
                  + " · ".join(f"{DECIDED[d]} {b[d]:,}" for d in DECIDED))
        return 0
    ap.print_help()
    return 0


# the default labellers — the two strong models the reasoning lab pinned
# (reasoning-lab rlab/strong.py: dated id, provider, prices), and a third from
# a third maker. Each can be changed on AI models
DEFAULT_LABELLERS = {
    "first": {"id": "openai/gpt-6-sol", "name": "GPT-6 Sol", "version": "openai/gpt-6-sol-20260922",
              "provider": "openai/flex", "price_in": 1.0, "price_out": 5.0},
    "second": {"id": "google/gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro",
               "version": "google/gemini-3.1-pro-preview-20260219",
               "provider": "google-ai-studio/flex", "price_in": 1.0, "price_out": 6.0},
    "third": {"id": "anthropic/claude-sonnet-5.5", "name": "Claude Sonnet 5.5", "version": "",
              "provider": "", "price_in": None, "price_out": None},
}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
