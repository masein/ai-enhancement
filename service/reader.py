"""11g: what the page's Reader reads — every file, in the page.

A person can read a dataset's documents, a rubric, a criteria file, the
practice half of a topic's questions, a run's log and a provenance record
without downloading anything. Each reader has its own JSON here, shaped for
reading, and each keeps the one rule that does not bend: a hidden
(report-half) question — its text, its answer, its qid — never reaches the
page. The bank reader gets the practice half and a count; the log reader
withholds any line that quotes a hidden question.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from . import config, db
from . import proposals as prop

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exam_build  # noqa: E402

PAGE = 50                        # a list reader asks for this many at a time
LOG_MAX = 2000                   # the most lines a log reader can hold


# ---------------------------------------------------------------------------
# dataset documents
# ---------------------------------------------------------------------------

def _items(did: int) -> list[dict]:
    p = prop.dataset_dir(did) / "items.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _labels(did: int, fmt: str, items: list[dict], pv: dict) -> list[dict] | None:
    """Which request — and so which focus label — each kept document came
    from. A dataset made from 11g on has it in items.meta.json, written when
    the batch landed. An older one is rebuilt from its provenance: the
    requests in order, each keeping what it asked for less what it lost. When
    that does not add up, the answer is "not recorded", never a guess."""
    side = prop.dataset_dir(did) / "items.meta.json"
    if side.exists():
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
            if isinstance(meta, list) and len(meta) == len(items):
                return meta
        except (OSError, ValueError):
            pass
    missing = (pv.get("items") or {}).get("missing")
    if not isinstance(missing, list):
        return None
    lost: dict = {}
    for m in missing:
        lost[m.get("request")] = lost.get(m.get("request"), 0) + 1
    labels = pv.get("focus_labels") or []
    plan = pv.get("focus_plan") or []
    count = int(pv.get("count_requested") or (pv.get("items") or {}).get("requested") or len(items))
    tries = []
    for per in {prop.items_per_request(fmt), 1}:
        if labels:
            tries.append(prop.focus_requests_plan(labels, per))
        tries.append(prop.focus_chunks(plan, per, count))
    for chunks in tries:
        out: list[dict] = []
        for k, (focus, n) in enumerate(chunks):
            out += [{"request": k, "focus": focus}] * max(0, n - lost.get(k, 0))
        if len(out) == len(items) and sum(n for _, n in chunks) == count:
            return out
    return None


def dataset_page(did: int, offset: int = 0, limit: int = PAGE, q: str = "") -> dict:
    """The dataset as the Reader lists it: every kept document, numbered, and
    every missing one in its place — after the documents of the request it
    was meant for — with its reason. A search narrows to the documents whose
    title or text holds it."""
    d = db.dataset_get(did)
    if not d:
        raise KeyError(did)
    try:
        pv = json.loads(d.get("provenance") or "{}")
    except (ValueError, TypeError):
        pv = {}
    fmt = d.get("fmt") or "doc"
    items = _items(did) if d["status"] == "ready" else []
    labels = _labels(did, fmt, items, pv) if items else None
    it = pv.get("items") or {}
    missing = it.get("missing") if isinstance(it.get("missing"), list) else None
    requested = int(it.get("requested") or pv.get("count_requested") or d.get("count") or 0)
    entries: list[dict] = []

    def doc(i: int, x: dict) -> dict:
        if fmt == "chat":
            # 12g.2: a request, the reply, and the checks it passed
            try:
                import everyday as _ev
                said = [_ev.describe(c) for c in x.get("checks") or []]
            except Exception:                    # noqa: BLE001 — the words are a courtesy
                said = []
            body = str(x.get("assistant") or "")
            e = {"type": "doc", "n": i + 1, "title": str(x.get("user") or "")[:160],
                 "user": x.get("user") or "", "assistant": body, "checks": said}
        elif fmt == "free":
            body = " ".join(str(x.get(k) or "") for k in ("question", "answer", "rationale"))
            e = {"type": "doc", "n": i + 1, "title": str(x.get("question") or "")[:160],
                 "question": x.get("question") or "", "answer": x.get("answer") or "",
                 "rationale": x.get("rationale") or ""}
        else:
            body = str(x.get("text") or "")
            e = {"type": "doc", "n": i + 1, "title": str(x.get("title") or ""), "text": body}
        e["words"] = len(body.split())
        e["focus"] = (labels[i] or {}).get("focus") if labels else None
        e["request"] = (labels[i] or {}).get("request") if labels else None
        return e

    if labels is not None and missing is not None:
        # in request order: each request's documents, then the ones it lost —
        # a request that kept nothing still has its place
        docs_by: dict = {}
        for i, x in enumerate(items):
            docs_by.setdefault(labels[i].get("request"), []).append(doc(i, x))
        gone_by: dict = {}
        for m in missing:
            gone_by.setdefault(m.get("request"), []).append(
                {"type": "missing", "focus": m.get("focus"), "why": m.get("why"),
                 "request": m.get("request")})
        keys = sorted(set(docs_by) | set(gone_by), key=lambda k: (k is None, k if k is not None else 0))
        for k in keys:
            entries += docs_by.get(k, []) + gone_by.get(k, [])
    else:
        entries = [doc(i, x) for i, x in enumerate(items)]
        if missing is not None:
            entries += [{"type": "missing", "focus": m.get("focus"), "why": m.get("why")}
                        for m in missing]
        else:
            # made before 11a: the gap is real, and its reasons were not kept
            entries += [{"type": "missing", "focus": None,
                         "why": "reasons not recorded (made before 11a)"}] \
                * max(0, requested - len(items))
    qn = (q or "").strip().lower()
    if qn:
        entries = [e for e in entries if e["type"] == "doc" and qn in " ".join(
            str(e.get(k) or "") for k in ("title", "text", "question", "answer", "rationale",
                                          "user", "assistant")).lower()]
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 200))
    return {"id": did, "status": d["status"], "fmt": fmt, "kept": len(items),
            "requested": requested,
            "missing": sum(1 for e in entries if e["type"] == "missing") if not qn
            else max(0, requested - len(items)),
            "labels_recorded": labels is not None, "q": q or "", "offset": offset,
            "limit": limit, "total": len(entries), "entries": entries[offset:offset + limit]}


def write_item_labels(did: int, origin: list[dict], dropped: list[dict]) -> None:
    """items.meta.json beside items.jsonl: the request and focus label of each
    kept document, in order, so the Reader never has to rebuild it."""
    gone = {d.get("index") for d in dropped or []}
    kept = [o for i, o in enumerate(origin) if i not in gone]
    p = prop.dataset_dir(did) / "items.meta.json"
    p.write_text(json.dumps(kept), encoding="utf-8")


# ---------------------------------------------------------------------------
# the bank, practice half only
# ---------------------------------------------------------------------------

def bank_practice(topic: str) -> dict:
    """The practice (diagnose) half of one topic's questions, and of the
    hidden (report) half only how many there are. No hidden question's
    text, answer or qid is in this reply."""
    rows = exam_build.load_bank(config.EXAM_DIR).get(topic) or []
    practice, hidden = [], 0
    for r in sorted(rows, key=lambda r: r["qid"]):
        if exam_build.half_of(r["qid"]) != "diagnose":
            hidden += 1
            continue
        meta = r.get("meta") or {}
        practice.append({"qid": r["qid"], "prompt": r.get("prompt") or "",
                         "reference": r.get("reference") or "",
                         "difficulty": meta.get("difficulty"), "domain": meta.get("domain"),
                         "style": meta.get("style"), "written_by": r.get("accepted_by") or "",
                         "source": r.get("source") or ""})
    return {"topic": topic, "half": "diagnose", "questions": practice, "report_count": hidden}


# ---------------------------------------------------------------------------
# rubric and criteria
# ---------------------------------------------------------------------------

def _judge():
    import judge
    return judge


def rubric_view(name: str) -> dict:
    j = _judge()
    p = j.rubric_path(name)
    if not p:
        raise KeyError(name)
    text = p.read_text(encoding="utf-8")
    m = re.search(r"\(version (\d+)", text)
    head = text.split("\n", 1)[0]
    return {"name": name, "file": p.name, "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "version": m.group(1) if m else "", "status": "draft" if "DRAFT" in head else ""}


def criteria_view(name: str) -> dict:
    """The criteria file exactly as the judge reads it: through the judge's own
    normalise_criteria(), so every layout the author has sent reads the same."""
    j = _judge()
    p = j.rubric_path(name, ".criteria.json")
    if not p:
        raise KeyError(name)
    raw = p.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    norm = j.normalise_criteria(spec)
    flags = []
    for f in norm.get("flags") or []:
        eff = f.get("effect") or ""
        flags.append({**f, "effect_words": j.effect_words(eff) if eff else ""})
    top = {k: v for k, v in norm.items() if k not in ("criteria", "flags", "evaluation_principles")}
    return {"name": name, "file": p.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "criteria": norm.get("criteria") or [], "flags": flags, "top": top,
            "principles": norm.get("evaluation_principles") or [],
            "raw": json.dumps(spec, indent=2, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# a run's log
# ---------------------------------------------------------------------------

_HIDDEN: dict = {"key": None, "qids": set(), "texts": []}


def _hidden_marks() -> tuple[set, list]:
    """Every hidden question's qid, and the start of its prompt and answer:
    what a log line must not carry. Rebuilt when the bank changes."""
    bank_dir = Path(config.EXAM_DIR) / "bank"
    files = sorted(bank_dir.glob("*.jsonl")) if bank_dir.is_dir() else []
    key = tuple((f.name, f.stat().st_mtime) for f in files)
    if _HIDDEN["key"] != key:
        qids, texts = set(), []
        for rows in exam_build.load_bank(config.EXAM_DIR).values():
            for r in rows:
                if exam_build.half_of(r["qid"]) == "diagnose":
                    continue
                qids.add(str(r["qid"]))
                for k in ("prompt", "reference"):
                    t = " ".join(str(r.get(k) or "").split())
                    if len(t) >= 24:
                        texts.append(t[:48].lower())
        _HIDDEN.update(key=key, qids=qids, texts=texts)
    return _HIDDEN["qids"], _HIDDEN["texts"]


def _withheld(line: str, qids: set, texts: list) -> bool:
    flat = " ".join(line.split()).lower()
    if any(t in flat for t in texts):
        return True
    return any(q in line for q in qids if len(q) >= 6)


def log_lines(sid: int, tail: int = 200) -> dict:
    """The last `tail` lines of a run's log (up to 2,000), numbered, with any
    line that quotes a hidden question withheld."""
    sub = db.get(sid)
    if not sub:
        raise KeyError(sid)
    path = config.LOGS_DIR / f"service_{sid}_{sub['hf_id'].replace('/', '__')}.log"
    status = sub.get("status") or ""
    base = {"id": sid, "status": status, "model": sub["hf_id"],
            "active": status in ("preflight", "waiting_gpu", "waiting_lock", "running", "canceling")}
    if not path.exists():
        return {**base, "total": 0, "first": 1, "lines": [], "withheld": 0,
                "note": "no log yet — the run has not started"}
    lines = path.read_text(errors="replace").splitlines()
    n = max(1, min(int(tail), LOG_MAX))
    keep = lines[-n:]
    qids, texts = _hidden_marks()
    out, withheld = [], 0
    for line in keep:
        if _withheld(line, qids, texts):
            withheld += 1
            out.append("[line withheld — it quotes a hidden question]")
        else:
            out.append(line)
    return {**base, "total": len(lines), "first": len(lines) - len(keep) + 1, "lines": out,
            "withheld": withheld}


# ---------------------------------------------------------------------------
# how a model was graded: its judge runs and its judge file's head
# ---------------------------------------------------------------------------

# what a topic's entry in judge.json may show in the Reader: counts, scores
# and fingerprints — never its items, which carry qids of both halves
_TASK_FIELDS = ("topic", "n", "n_report", "n_diagnose", "score_report", "score_diagnose",
                "ungraded", "unparseable", "bank_sha256", "max",
                # 11l: what the answers were generated with, and how many never
                # left a reasoning block — it changes what a score means
                "generation", "no_answer", "no_score")
_HEAD_FIELDS = ("judge", "canary", "split_salt", "correct_at", "preliminary_reasons")


def judge_provenance(model: str) -> dict:
    """"How this was graded": the judge runs recorded for one model (never a
    run's plan, which lists what it graded) and the head of its judge.json,
    with each topic reduced to counts, scores and fingerprints."""
    runs = [r for r in db.judge_runs(500) if r.get("model") == model]
    j = prop._judge_file(config.OUT_DIR / model.replace("/", "__"))
    if not runs and not j:
        raise KeyError(model)
    head = {k: j.get(k) for k in _HEAD_FIELDS if j.get(k) is not None} if j else {}
    tasks = {t: {k: v.get(k) for k in _TASK_FIELDS if v.get(k) is not None}
             for t, v in sorted(((j or {}).get("tasks") or {}).items())}
    return {"model": model, "runs": runs, "judge_file": head, "tasks": tasks}
