"""12d.1: the Playground's side of a chat — the models it offers, practice
questions (the practice half only: a hidden question is never offered, sent,
or reachable by id), a message and its reply, and marking a practice question
sent unedited, on the scored settings, as a chat's first message. A mark here
is the chat's own record: never a score, never on the board."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from . import chat, config, db

BASE_LINE = "Base models aren't listed: they have no chat format."
OWN_CODE_LINE = "Models that run their own code aren't listed: chat doesn't run a model's code."
NOT_JUDGED = "not marked here; the exam's judge marks it in runs"


def _scripts() -> None:
    here = str(Path(__file__).resolve().parent.parent / "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)


def models() -> dict:
    """what the picker lists — each with its scored settings — and the
    line for what it leaves out"""
    rows = chat.board_models()
    offered = [{"id": m["id"], "name": m["name"], "source": m["source"],
                "params": m["params"], "ctx": (m["archinfo"] or {}).get("ctx"),
                "scored": chat.scored_settings(m)} for m in rows if m["chat"]]
    lines = []
    if any(m["why_not"] == "base" for m in rows):
        lines.append(BASE_LINE)
    if any(m["why_not"] == "own code" for m in rows):
        lines.append(OWN_CODE_LINE)
    return {"models": offered, "left_out": lines}


# ---------------------------------------------------------------------------
# practice questions: the practice half only
# ---------------------------------------------------------------------------

def _everyday_practice() -> dict[str, dict]:
    _scripts()
    import everyday as ev
    labels = ev.groups()
    return {q["id"]: {**q, "groupLabel": labels.get(q["group"], q["group"])}
            for q in ev.load_bank() if ev.half(q) == ev.PRACTICE}


def _exam_practice() -> dict[str, dict]:
    _scripts()
    import exam_build as eb
    if not config.EXAM_DIR.is_dir():
        return {}
    out = {}
    for topic, rows in eb.load_bank(config.EXAM_DIR).items():
        for r in rows:
            if eb.half_of(r["qid"]) == "diagnose":
                out[r["qid"]] = {**r, "topic": topic}
    return out


def practice() -> dict:
    """the lists the picker shows: prompts only, the practice half only"""
    evd = sorted(_everyday_practice().values(), key=lambda q: (q["group"], q["id"]))
    exam = sorted(_exam_practice().values(), key=lambda r: (r["topic"], r["qid"]))
    return {"everyday": [{"kind": "everyday", "id": q["id"], "group": q["group"],
                          "groupLabel": q["groupLabel"], "prompt": q["prompt"]} for q in evd],
            "knowledge": [{"kind": "knowledge", "id": r["qid"], "topic": r["topic"],
                           "prompt": r["prompt"]} for r in exam]}


def practice_item(ref: dict | None) -> dict | None:
    """a practice question by {kind, id} — None for anything else, a hidden
    question's id included"""
    if not isinstance(ref, dict):
        return None
    if ref.get("kind") == "everyday":
        return _everyday_practice().get(str(ref.get("id")))
    if ref.get("kind") == "knowledge":
        return _exam_practice().get(str(ref.get("id")))
    return None


def mark(ref: dict | None, text_sent: str, first: bool, scored: bool, reply: str) -> dict | None:
    """how the run would mark it — Everyday's script checks in plain words, or
    the exam's reference, folded — or why it isn't marked"""
    if not ref:
        return None
    q = practice_item(ref)
    if q is None:
        return None
    if text_sent != q["prompt"]:
        return {"unmarked": "edited, so not marked"}
    if not scored:
        return {"unmarked": "settings changed, so not marked"}
    if not first:
        return {"unmarked": "not the chat's first message, so not marked"}
    if ref["kind"] == "knowledge":
        return {"kind": "knowledge", "reference": q.get("reference") or "", "note": NOT_JUDGED}
    _scripts()
    import everyday as ev
    ok, _ = ev.grade(q, reply)
    if ok is True:
        return {"kind": "everyday", "pass": True, "words": "✓ passes"}
    if ok is None:
        return {"kind": "everyday", "pass": None, "words": "✓ passes its script checks",
                "note": "its judge check isn't run here: the run's judge marks it"}
    fails = ev.failures(q, reply)
    return {"kind": "everyday", "pass": False,
            "words": "✗ " + "; ".join(f["why"] for f in fails),
            "checks": [f["check"] for f in fails]}


# ---------------------------------------------------------------------------
# a message and its reply
# ---------------------------------------------------------------------------

def view(c: dict) -> dict:
    row = chat.model_row(c["model"]) or {"id": c["model"], "name": c["model"].split("/")[-1],
                                           "archinfo": {}, "params": None}
    return {**{k: v for k, v in c.items() if k != "who"},
            "name": row["name"], "scored": chat.scored_settings(row),
            "is_scored": chat.is_scored(c["settings"], row)}


def history(c: dict, upto: int) -> list[dict]:
    """what the model is sent: the system message, then each message up to
    `upto`, a reply as the text shown (never its thinking)"""
    row = chat.model_row(c["model"])
    s = chat.effective(c["settings"], row)
    out = [{"role": "system", "content": s["system"]}] if s["system"] else []
    for m in c["messages"][:upto]:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["text"]})
        elif m.get("replies"):
            out.append({"role": "assistant", "content": m["replies"][m.get("shown", 0)]["text"]})
    return out


def _too_long(c: dict, text: str) -> str:
    row = chat.model_row(c["model"])
    s = chat.effective(c["settings"], row)
    fits = chat.context_words(row, s)
    have = sum(len((m.get("text") or (m["replies"][m.get("shown", 0)]["text"]
                                      if m.get("replies") else "")).split())
               for m in c["messages"]) + len(text.split()) + len(s["system"].split())
    return f"Too long for this model: about {fits:,} words fits." if have > fits else ""


def send(chat_id: str, text: str, by: str, practice_ref: dict | None = None) -> dict:
    """a person's message: kept, and its reply started"""
    c = chat.get_chat(chat_id, by)
    text = (text or "").strip()
    if not text:
        raise ValueError("an empty message")
    if any(m["role"] == "assistant" and m.get("pending") for m in c["messages"]):
        raise ValueError("a reply is still coming — stop it first")
    why = _too_long(c, text)
    if why:
        raise ValueError(why)
    first = not c["messages"]
    ref = practice_ref if practice_item(practice_ref) else None
    c["messages"].append({"role": "user", "text": text, "at": time.time(), "practice": ref})
    c["messages"].append({"role": "assistant", "replies": [], "shown": 0, "pending": True})
    c["title"] = c["title"] or chat.title_of(text)
    db.chat_put(c)
    n = len(c["messages"]) - 1
    return _start(c, n, {"ref": ref, "text": text, "first": first})


def again(chat_id: str, n: int, by: str) -> dict:
    """the same message asked again: both replies kept, ‹ 1 of 2 ›"""
    c = chat.get_chat(chat_id, by)
    if not (0 < n < len(c["messages"])) or c["messages"][n]["role"] != "assistant":
        raise ValueError("no reply there")
    if any(m["role"] == "assistant" and m.get("pending") for m in c["messages"]):
        raise ValueError("a reply is still coming — stop it first")
    u = c["messages"][n - 1]
    c["messages"][n]["pending"] = True
    db.chat_put(c)
    return _start(c, n, {"ref": u.get("practice"), "text": u["text"], "first": n == 1})


def _start(c: dict, n: int, sent: dict) -> dict:
    row = chat.model_row(c["model"])
    if not row or not row["chat"]:
        raise ValueError("this model isn't offered here any more")
    settings = chat.effective(c["settings"], row)
    scored = chat.is_scored(c["settings"], row)

    def on_done(reply: dict | None, why: str = "") -> dict | None:
        cc = db.chat_get(c["id"])
        m = cc["messages"][n]
        m.pop("pending", None)
        if reply is None:                   # refused, stopped before it began, or failed
            m["refused"] = why
            db.chat_put(cc)
            return None
        m.pop("refused", None)
        reply["at"] = time.time()
        reply["mark"] = mark(sent["ref"], sent["text"], sent["first"], scored, reply["text"])
        m.setdefault("replies", []).append(reply)
        m["shown"] = len(m["replies"]) - 1
        db.chat_put(cc)
        return reply

    # the stream's id is kept before it starts: a fast reply is stored by
    # on_done, and nothing here may write an older copy over it
    sid = chat.uuid.uuid4().hex
    c["messages"][n]["stream"] = sid
    db.chat_put(c)
    try:
        chat.ENGINE.start(c, history(c, n), settings, row, on_done, stream_id=sid)
    except Exception:
        cc = db.chat_get(c["id"])
        cc["messages"][n].pop("pending", None)
        db.chat_put(cc)
        raise
    return {"stream": sid, "chat": view(db.chat_get(c["id"]))}


def set_settings(chat_id: str, settings: dict, by: str) -> dict:
    c = chat.get_chat(chat_id, by)
    row = chat.model_row(c["model"])
    base = chat.scored_settings(row)
    s = {}
    for k in chat.FIELDS:
        if k not in settings or settings[k] == base[k]:
            continue
        v = settings[k]
        if k == "system":
            v = str(v)[:4000]
        elif k == "thinking":
            # on or off as it was scored is the scored setting, however said
            if not base["can_think"] or v is None or bool(v) == base["thinking_on"]:
                continue
            v = bool(v)
        elif k == "temperature":
            v = max(0.0, min(2.0, float(v)))
        elif k == "max_gen_toks":
            v = max(16, min(int(v), 16384))
        s[k] = v
    c["settings"] = s
    db.chat_put(c)
    return view(c)


def delete(chat_id: str, by: str) -> None:
    chat.get_chat(chat_id, by)
    db.chat_delete(chat_id)
