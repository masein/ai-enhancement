"""12d.1: the Playground's side of a chat — the models it offers, practice
questions (the practice half only: a hidden question is never offered, sent,
or reachable by id), a message and its reply, and marking a practice question
sent unedited, on the scored settings, as a chat's first message. A mark here
is the chat's own record: never a score, never on the board."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from . import chat, config, db

BASE_LINE = "Base models aren't listed: they have no chat format."
OWN_CODE_LINE = "Models that run their own code aren't listed: chat doesn't run a model's code."
# 12m.3: a chat's cost isn't metered the way a run's is
# 16.1: a GGUF file with no server is measured by the host's worker, never chatted with
GGUF_LINE = ("GGUF files aren't here: chat with one through its llama-server, registered "
             "under Add a model ▸ Running on a server.")
OPENROUTER_LINE = ("Models from OpenRouter aren't listed: a chat's cost isn't counted against "
                   "the monthly AI limit.")
NOT_JUDGED = "not marked here; the exam's judge marks it in runs"


def _scripts() -> None:
    here = str(Path(__file__).resolve().parent.parent / "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)


def _trained() -> dict[str, dict]:
    """12d.2: a model trained in Improve ("Trained from" set): its base, and
    the day it was trained — its upload's, else the day it was said"""
    out = {}
    for mid, t in db.trained_from_all().items():
        at = t.get("at") or 0
        if mid.startswith("local/"):
            p = config.ARTIFACTS_DIR / mid[6:]
            try:
                at = p.stat().st_mtime
            except OSError:
                pass
        out[mid] = {"base": t["base"], "date": time.strftime("%Y-%m-%d", time.localtime(at))}
    return out


def states() -> dict:
    """16.2: each offered model's state — {id: {state, why}} — and the GPU's
    line: what the picker asks again every few seconds"""
    from . import gpu
    g = gpu.status()
    return {"states": {m["id"]: chat.ENGINE.state(m) for m in chat.board_models() if m["chat"]},
            "gpu": {"line": g.get("line") or "", "ok": g.get("ok"),
                    "holders": [gpu.holder_words(h) for h in g.get("holders") or []]}}


def models() -> dict:
    """what the picker lists — each with its scored settings, a trained model
    right under its base — and the line for what it leaves out"""
    rows = chat.board_models()
    trained = _trained()
    offered = [{"id": m["id"], "name": m["name"], "source": m["source"],
                "params": m["params"], "ctx": (m["archinfo"] or {}).get("ctx"),
                "scored": chat.scored_settings(m),
                "trained_from": (trained.get(m["id"]) or {}).get("base") or "",
                "trained_on": (trained.get(m["id"]) or {}).get("date") or "",
                # 12d.3: served elsewhere, and a phone build (12f.2): the picker's tags
                "served": bool(m.get("served")), "phone": bool(m.get("phone")),
                # 16.2: whether it can answer now, and why not
                **chat.ENGINE.state(m)}
               for m in rows if m["chat"]]
    ids = {m["id"] for m in offered}
    bases = [m for m in offered if not (m["trained_from"] and m["trained_from"] in ids)]
    order = []
    for m in bases:
        order.append(m)
        order.extend(x for x in offered if x["trained_from"] == m["id"])
    offered = order
    lines = []
    if any(m["why_not"] == "base" for m in rows):
        lines.append(BASE_LINE)
    if any(m["why_not"] == "own code" for m in rows):
        lines.append(OWN_CODE_LINE)
    if any(m["why_not"] == "openrouter" for m in rows):
        lines.append(OPENROUTER_LINE)
    if any(m["why_not"] == "gguf" for m in rows):
        lines.append(GGUF_LINE)
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
    # 16.5: no Knowledge exam question to try while it is switched off
    if not config.KNOWLEDGE_EXAM:
        return {}
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
    row2 = chat.model_row(c["model2"]) if c.get("model2") else None
    return {**{k: v for k, v in c.items() if k != "who"},
            "name": row["name"],
            "name2": (row2 or {}).get("name") or (c.get("model2") or "").split("/")[-1],
            "scored": chat.scored_settings(row),
            "is_scored": chat.is_scored(c["settings"], row)}


def _side(m: dict, col: str) -> dict:
    """an answer's column: "a" is the chat's model, "b" (12d.2) the one it is
    compared with"""
    return m if col == "a" else m.setdefault("b", {"replies": [], "shown": 0})


def history(c: dict, upto: int, col: str = "a") -> list[dict]:
    """what a model is sent: the system message, then each message up to
    `upto`, a reply as the text shown (never its thinking) — its own replies"""
    row = chat.model_row(c["model"] if col == "a" else c["model2"])
    s = chat.effective(c["settings"], row)
    out = [{"role": "system", "content": s["system"]}] if s["system"] else []
    for m in c["messages"][:upto]:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["text"]})
        else:
            side = _side(dict(m), col)
            if side.get("replies"):
                out.append({"role": "assistant",
                            "content": side["replies"][side.get("shown", 0)]["text"]})
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
    if _pending(c):
        raise ValueError("a reply is still coming — stop it first")
    why = _too_long(c, text)
    if why:
        raise ValueError(why)
    first = not c["messages"]
    ref = practice_ref if practice_item(practice_ref) else None
    c["messages"].append({"role": "user", "text": text, "at": time.time(), "practice": ref})
    a = {"role": "assistant", "replies": [], "shown": 0, "pending": True}
    if c.get("model2"):
        a["b"] = {"replies": [], "shown": 0, "pending": True}
    c["messages"].append(a)
    c["title"] = c["title"] or chat.title_of(text)
    db.chat_put(c)
    n = len(c["messages"]) - 1
    return _start(c, n, {"ref": ref, "text": text, "first": first},
                  cols=("a", "b") if c.get("model2") else ("a",))


def _pending(c: dict) -> bool:
    return any(m["role"] == "assistant" and (m.get("pending") or (m.get("b") or {}).get("pending"))
               for m in c["messages"])


def again(chat_id: str, n: int, by: str, col: str = "a") -> dict:
    """the same message asked again of one column's model: both replies
    kept, ‹ 1 of 2 ›"""
    c = chat.get_chat(chat_id, by)
    if not (0 < n < len(c["messages"])) or c["messages"][n]["role"] != "assistant":
        raise ValueError("no reply there")
    if col == "b" and not c.get("model2"):
        raise ValueError("this chat compares no second model")
    if _pending(c):
        raise ValueError("a reply is still coming — stop it first")
    u = c["messages"][n - 1]
    _side(c["messages"][n], col)["pending"] = True
    db.chat_put(c)
    return _start(c, n, {"ref": u.get("practice"), "text": u["text"], "first": n == 1},
                  cols=(col,))


_store = threading.Lock()           # 12d.2: two columns finish at once: one write at a time


def _start(c: dict, n: int, sent: dict, cols=("a",)) -> dict:
    rows = {col: chat.model_row(c["model"] if col == "a" else c["model2"]) for col in cols}
    if any(not r or not r["chat"] for r in rows.values()):
        raise ValueError("this model isn't offered here any more")

    def on_done_for(col: str, scored: bool):
        def on_done(reply: dict | None, why: str = "") -> dict | None:
            with _store:
                cc = db.chat_get(c["id"])
                m = _side(cc["messages"][n], col)
                m.pop("pending", None)
                if reply is None:           # refused, stopped before it began, or failed
                    m["refused"] = why
                    db.chat_put(cc)
                    return None
                m.pop("refused", None)
                reply["at"] = time.time()
                reply["mark"] = mark(sent["ref"], sent["text"], sent["first"], scored,
                                     reply["text"])
                m.setdefault("replies", []).append(reply)
                m["shown"] = len(m["replies"]) - 1
                db.chat_put(cc)
                return reply
        return on_done

    # each stream's id is kept before it starts: a fast reply is stored by
    # on_done, and nothing here may write an older copy over it
    sids = {col: chat.uuid.uuid4().hex for col in cols}
    for col in cols:
        _side(c["messages"][n], col)["stream"] = sids[col]
    db.chat_put(c)
    # 12d.2: both at once when both fit; else the second after the first
    together = len(cols) < 2 or chat.ENGINE.fits_both(rows["a"], rows["b"])
    first = None
    try:
        for col in cols:
            row = rows[col]
            st = chat.ENGINE.start(c, history(c, n, col), chat.effective(c["settings"], row), row,
                                   on_done_for(col, chat.is_scored(c["settings"], row)),
                                   stream_id=sids[col], after=None if together else first)
            first = first or st
    except Exception:
        with _store:
            cc = db.chat_get(c["id"])
            for col in cols:
                _side(cc["messages"][n], col).pop("pending", None)
            db.chat_put(cc)
        raise
    out = {"stream": sids.get("a") or sids["b"], "chat": view(db.chat_get(c["id"]))}
    if "b" in sids:
        out["stream_b"] = sids["b"]
    return out


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
