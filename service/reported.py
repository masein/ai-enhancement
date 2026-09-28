"""12m.2: reported scores — numbers from outside the board.

Three sources, each credited wherever its numbers show:
- **Epoch AI's Benchmarking Hub** (CC BY 4.0: "Data: Epoch AI, CC BY 4.0"),
  its zip of CSVs, imported daily;
- **Artificial Analysis's free Data API** (credit: "Data: Artificial
  Analysis"), with a key masein makes, imported daily. Their free tier is
  for internal use: this board is on the tailnet, and these numbers never
  leave it (they are never in the frozen single-file report either);
- **model cards and papers**, typed in: the model, the benchmark, the value,
  their setting (shots, chain of thought or not), the source's URL, the date
  and who entered it.

Never measured here, so never mixed with what is: they live in their own
tables and reach the page on their own endpoint (/api/reported), like the
phone's reports (12f.2) — never a results file, a board column, an average
or a rank. A reported MMLU-Pro 5-shot CoT is not our MMLU-Pro subset.

Each import keeps its file's sha256 and the date: a changed file is a new
import, an unchanged one is only checked. A model's name differs between
sources ("GPT-5.5" at Epoch, "openai/gpt-5.5" at OpenRouter); a small
editable alias table makes them one model — and one measured here too.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import threading
import urllib.request

from . import config, db

SOURCES = {
    "epoch": {"name": "Epoch AI", "credit": "Data: Epoch AI, CC BY 4.0",
              "licence": "CC BY 4.0", "url": "https://epoch.ai/benchmarks"},
    "aa": {"name": "Artificial Analysis", "credit": "Data: Artificial Analysis",
           "licence": "their free Data API: internal use, with credit",
           "url": "https://artificialanalysis.ai"},
    "card": {"name": "model card", "credit": "as the model card or paper reports it",
             "licence": "", "url": ""},
}
NO_AA_KEY = ("Artificial Analysis: no key yet — add ARTIFICIAL_ANALYSIS_API_KEY to the "
             "server's .env, then Import now")

# Artificial Analysis's evaluations, by the names the board shows; an index
# is their points (0–100), a benchmark a share
AA_BENCH = {"artificial_analysis_intelligence_index": ("Intelligence Index", "points"),
            "artificial_analysis_coding_index": ("Coding Index", "points"),
            "artificial_analysis_math_index": ("Math Index", "points"),
            "mmlu_pro": ("MMLU-Pro", "share"), "gpqa": ("GPQA Diamond", "share"),
            "hle": ("Humanity's Last Exam", "share"), "livecodebench": ("LiveCodeBench", "share"),
            "scicode": ("SciCode", "share"), "math_500": ("MATH-500", "share"),
            "aime": ("AIME", "share")}


# ---------------------------------------------------------------------------
# a model's name, across sources
# ---------------------------------------------------------------------------

def slug(s: str) -> str:
    """"GPT-5.5 (high)" -> "gpt-5.5-high": lower case, words joined by -"""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9.]+", "-", str(s).lower())).strip("-")


def key(name: str, maker: str = "") -> str:
    """one model's key: "openai/gpt-5.5". A name that already carries its
    maker ("openai/gpt-5.5", OpenRouter's) keeps it"""
    if "/" in str(name):
        m, n = str(name).split("/", 1)
        return f"{slug(m)}/{slug(n)}"
    return f"{slug(maker) or 'unknown'}/{slug(name)}"


def board_ids() -> list[str]:
    """every model on the board, by its id, from the model_meta.json beside
    its results (served models and GGUF files have one too)"""
    out = []
    for f in sorted(config.OUT_DIR.glob("*/model_meta.json")) if config.OUT_DIR.is_dir() else []:
        try:
            m = json.loads(f.read_text(encoding="utf-8")).get("model")
        except (OSError, ValueError):
            continue
        if m:
            out.append(m)
    return out


def aliases() -> dict[str, str]:
    """alias key -> the key or board model id it is"""
    return {a["alias"]: a["target"] for a in db.reported_aliases()}


def resolve(k: str, board: list[str] | None = None, al: dict | None = None) -> str:
    """what a key is: an alias's target, else a board model with the same
    name (an open model already here: "qwen/qwen3-1.7b" and Qwen/Qwen3-1.7B),
    else itself"""
    al = aliases() if al is None else al
    k = al.get(k, k)
    for mid in board or []:
        if mid == k or slug(mid.split("/")[-1]) == k.split("/")[-1]:
            return mid
    return k


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------

def _fetch(url: str, headers: dict | None = None, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "evalboard", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _begin(source: str, blob: bytes) -> tuple[str, dict | None]:
    """the file's sha256, and the last import when it is the same file"""
    sha = hashlib.sha256(blob).hexdigest()
    last = db.reported_import_last(source)
    return sha, (last if last and last["sha256"] == sha else None)


def _line(source: str, rec: dict | None) -> str:
    name = SOURCES[source]["name"]
    if not rec:
        return f"{name}: not imported yet"
    day = lambda t: _dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d")  # noqa: E731
    checked = rec.get("checked_at") or rec["at"]
    return (f"{name}: {rec['n']} scores for {rec['models']} models, imported {day(rec['at'])} "
            f"· file {rec['sha256'][:12]}"
            + (f" · unchanged when checked {day(checked)}" if day(checked) != day(rec["at"]) else ""))


def keep(rows: list[dict], makers: list[str] | None = None, per: int | None = None,
         board: list[str] | None = None) -> list[dict]:
    """the models kept: each maker's `per` most recent (by release date where
    the source gives one, else in the source's own order), and any model
    already on the board. rows: {key, maker, released?, …} a score"""
    makers = [slug(m) for m in (makers if makers is not None else config.REPORTED_MAKERS)]
    per = config.REPORTED_PER_MAKER if per is None else per
    first: dict[str, dict] = {}
    for i, r in enumerate(rows):
        first.setdefault(r["key"], {"i": i, "maker": slug(r.get("maker") or r["key"].split("/")[0]),
                                    "released": r.get("released") or ""})
    chosen = set()
    for mk in makers:
        ks = sorted((k for k, v in first.items() if v["maker"] == mk or v["maker"].startswith(mk)),
                    key=lambda k: first[k]["i"])
        # newest first; a sort that keeps the source's order among equals
        chosen.update(sorted(ks, key=lambda k: first[k]["released"], reverse=True)[:per])
    al = aliases()
    for k in first:
        if resolve(k, board, al) in (board or []):
            chosen.add(k)
    return [r for r in rows if r["key"] in chosen]


def import_aa(fetch=None, board: list[str] | None = None) -> dict:
    """Artificial Analysis's models and their evaluations. Nothing is asked
    of them without a key, and the line says so"""
    if not config.AA_API_KEY:
        return {"source": "aa", "status": "no key", "line": NO_AA_KEY}
    board = board_ids() if board is None else board
    blob = (fetch or _fetch)(config.AA_URL, {"x-api-key": config.AA_API_KEY})
    sha, same = _begin("aa", blob)
    if same:
        db.reported_import_checked(same["id"])
        return {"source": "aa", "status": "unchanged", "line": _line("aa", same)}
    data = json.loads(blob.decode("utf-8")).get("data") or []
    rows = []
    for m in data:
        maker = ((m.get("model_creator") or {}).get("name") or "").strip()
        name = (m.get("name") or m.get("slug") or "").strip()
        if not name:
            continue
        for field, v in (m.get("evaluations") or {}).items():
            if field not in AA_BENCH or not isinstance(v, (int, float)):
                continue
            label, unit = AA_BENCH[field]
            rows.append({"key": key(name, maker), "name": name, "maker": maker,
                         "released": m.get("release_date") or "",
                         "benchmark": label, "unit": unit,
                         "value": float(v) if unit == "points" or v <= 1 else float(v) / 100,
                         "setting": "Artificial Analysis's own run",
                         "url": "https://artificialanalysis.ai/models/" + (m.get("slug") or slug(name))})
    return _store("aa", sha, keep(rows, board=board))


def _store(source: str, sha: str, rows: list[dict]) -> dict:
    iid = db.reported_import_add(source, sha, rows)
    rec = db.reported_import_last(source)
    return {"source": source, "status": "imported", "id": iid, "line": _line(source, rec)}


def run_all(fetch=None, board: list[str] | None = None) -> list[dict]:
    """every source that imports, once: its line each (an error is a line too)"""
    board = board_ids() if board is None else board
    out = []
    for name, fn in (("aa", import_aa),):
        try:
            out.append(fn(fetch=fetch, board=board))
        except Exception as e:                      # noqa: BLE001 — one source's error is its line
            out.append({"source": name, "status": "failed",
                        "line": f"{SOURCES[name]['name']}: the import failed — {e}"})
    return out


_daily = {"day": None, "lock": threading.Lock()}


def daily() -> bool:
    """once a day, from the queue's idle loop: every source, in a thread of
    its own so the queue never waits. True when it started one"""
    if not config.REPORTED_DAILY:
        return False
    today = _dt.date.today().isoformat()
    with _daily["lock"]:
        if _daily["day"] == today:
            return False
        _daily["day"] = today
    threading.Thread(target=run_all, name="reported-daily", daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# model cards, typed in
# ---------------------------------------------------------------------------

def card_add(f: dict, entered_by: str) -> dict:
    """one number from a model card or paper, as typed — or why not"""
    name, maker = str(f.get("model") or "").strip(), str(f.get("maker") or "").strip()
    bench, setting = str(f.get("benchmark") or "").strip(), str(f.get("setting") or "").strip()
    url, date = str(f.get("url") or "").strip(), str(f.get("date") or "").strip()
    who = (entered_by or "").strip()
    if not name:
        raise ValueError("The model, as its card names it")
    if not bench:
        raise ValueError("The benchmark, as the card names it")
    raw = str(f.get("value") or "").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(%?)", raw)
    if not m:
        raise ValueError("The value, a number: 86.4 or 86.4%")
    v = float(m.group(1))
    value = v / 100 if m.group(2) or v > 1 else v
    if not 0 <= value <= 1:
        raise ValueError("A share from 0 to 100%")
    if not setting:
        raise ValueError("Their setting: the shots, and chain of thought or not (\"5-shot, CoT\")")
    if not re.match(r"https?://", url):
        raise ValueError("The source: the card's or paper's address, https://…")
    try:
        _dt.date.fromisoformat(date)
    except ValueError:
        raise ValueError("The date the card gives it, as 2026-09-28") from None
    if not who:
        raise ValueError("Who entered it")
    row = {"key": key(name, maker), "name": name[:120], "maker": maker[:60],
           "benchmark": bench[:80], "unit": "share", "value": value, "setting": setting[:200],
           "url": url[:500], "date": date, "by": who[:80]}
    row["id"] = db.reported_card_add(row)
    return row


# ---------------------------------------------------------------------------
# the page's view
# ---------------------------------------------------------------------------

def view(board: list[str] | None = None) -> dict:
    """what /api/reported returns: each source's line and credit, the models
    (a board model when one is the same model), and every score"""
    board = board_ids() if board is None else board
    al = aliases()
    scores, models = [], {}
    for r in db.reported_scores():
        mid = resolve(r["key"], board, al)
        here = mid in (board or [])
        rid = mid if here else "reported/" + r["key"]
        models.setdefault(rid, {"id": rid, "key": r["key"], "name": r["name"],
                                "maker": r["maker"], "measured": mid if here else None})
        scores.append({"model": rid, "source": r["source"], "benchmark": r["benchmark"],
                       "value": r["value"], "unit": r["unit"], "setting": r["setting"],
                       "url": r["url"], "date": r.get("date") or "", "by": r.get("by") or ""})
    srcs = {}
    for s, meta in SOURCES.items():
        last = db.reported_import_last(s) if s != "card" else None
        srcs[s] = {**meta, "line": NO_AA_KEY if s == "aa" and not config.AA_API_KEY and not last
                   else _line(s, last) if s != "card" else "",
                   "has_key": bool(config.AA_API_KEY) if s == "aa" else None}
    return {"sources": srcs, "models": list(models.values()), "scores": scores,
            "aliases": [{"alias": a, "target": t} for a, t in sorted(al.items())],
            "settings": {"makers": config.REPORTED_MAKERS, "per_maker": config.REPORTED_PER_MAKER}}


def alias_set(alias: str, target: str, by: str) -> dict:
    a, t = key(alias), str(target or "").strip()
    if not alias.strip() or not t:
        raise ValueError("An alias is a name and the model it is")
    if not by.strip():
        raise ValueError("Who set it")
    # a board model's id as it is; a reported model by its key
    t = t.removeprefix("reported/") if t.startswith("reported/") else t
    db.reported_alias_set(a, t, by.strip()[:80])
    return {"alias": a, "target": t}
