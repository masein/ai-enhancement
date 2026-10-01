"""14.3: Mobile-MMLU-Pro's answer key — who labels it, and the labelling run.
The rules and the file are scripts/mobile_mmlu.py's.

**The labellers.** Three slots, each a model on OpenRouter pinned as the
judge is (ai_models.pin: its dated version, its first provider, no
fallbacks), chosen on AI models. A slot nobody has chosen for keeps its
default (mobile_mmlu.DEFAULT_LABELLERS), pinned when Start is pressed. No
labeller may be:
- local, or in-house (a Qwen3.6 build): the key must come from outside;
- a model with a Mobile-MMLU-Pro score on the board. One scored later says
  "labelled the key" on its row and isn't ranked;
- from the same maker as another labeller.

**The run.** Nothing is sent until masein presses Start, after the dry run's
tokens and cost. The first two labellers each get, as a batch, every question
they haven't labelled; the third gets the questions they split on, as their
answers land. A batch is OpenRouter's, worked on disk a few requests at a time
(llm.OpenRouterChat): Stop holds it (nothing more is sent; what is in flight
lands), Start carries on where it stopped, and so does a restart. Its spend
counts against the month's AI limit as the "labeller" job, and at the limit
it waits. As a batch lands, its labels are kept by question id and the key is
rebuilt.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from . import ai_models, config, db, llm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mobile_mmlu as mmp  # noqa: E402

KIND = "mmpk"
JOB = "labeller"
SLOT_LABEL = {"first": "First labeller", "second": "Second labeller", "third": "Third labeller"}
SLOT_DOES = {"first": "answers every question on its own",
             "second": "answers every question on its own, from another maker",
             "third": "answers only where the first two differ, from a third maker"}
SLOT_WHY = {"first": "the strong model the reasoning lab validates with",
            "second": "the reasoning lab's other strong model, another maker",
            "third": "a third maker, for the splits"}
IN_HOUSE = re.compile(r"qwen[\s_-]*3[._-]?6", re.I)
STOPPED = "stopped — Start carries on where it stopped"


def _setting(slot: str) -> str:
    return "labeller:" + slot


def chosen(slot: str) -> dict | None:
    return db.ai_get(_setting(slot))


def labeller(slot: str) -> dict:
    """the slot's labeller: chosen on AI models, else its default — {kind:
    openrouter|default, id, name, version, provider, price_in, price_out}"""
    c = chosen(slot)
    if c:
        return c
    d = mmp.DEFAULT_LABELLERS[slot]
    cached = _cached(d["id"]) or {}
    return {"kind": "default", **d,
            "price_in": d["price_in"] if d["price_in"] is not None else cached.get("price_in"),
            "price_out": d["price_out"] if d["price_out"] is not None else cached.get("price_out")}


def labellers() -> dict:
    return {s: labeller(s) for s in mmp.SLOTS}


def _cached(model_id: str) -> dict | None:
    """OpenRouter's list as AI models last fetched it — never asked here"""
    try:
        got = json.loads(ai_models._cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return next((m for m in got.get("models") or [] if m.get("id") == model_id), None)


def scored_ids() -> set[str]:
    """every model with a Mobile-MMLU-Pro score on the board, by its own id
    and (a model from OpenRouter) the id OpenRouter knows it by"""
    out: set[str] = set()
    for d in config.OUT_DIR.iterdir() if config.OUT_DIR.is_dir() else []:
        if not (d / mmp.PRED_FILE).exists():
            continue
        try:
            meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        out.add(meta.get("model") or d.name.replace("__", "/"))
        pin = ((meta.get("served") or {}).get("pin") or {}).get("model")
        if pin:
            out.add(pin)
    return out


def refused(slot: str, model_id: str, others: dict | None = None) -> str:
    """'' when `model_id` may label this slot; else why not, in one line"""
    mid = (model_id or "").strip()
    if mid == ai_models.LOCAL or not mid:
        return ("a labeller can't be local: the key must come from strong models made "
                "elsewhere, never one run on this server")
    if IN_HOUSE.search(mid) or mid.startswith("served/"):
        return ("a labeller can't be an in-house model (a Qwen3.6 build, or a model served "
                "here): choose a model on OpenRouter from another maker")
    if mid in scored_ids():
        return (f"{mid} has a Mobile-MMLU-Pro score on the board: a model can't label the key "
                "it is scored on")
    others = labellers() if others is None else others
    mk = ai_models.maker(mid)
    for s in (x for x in mmp.SLOTS if x != slot):
        o = others.get(s) or {}
        if o.get("id") and ai_models.maker(o["id"]) == mk:
            return (f"the {SLOT_LABEL[s].lower()} is {mk} too: each labeller must come from "
                    "another maker")
    return ""


def save(slot: str, model_id: str, by: str) -> dict:
    """pin a slot to a model on OpenRouter — refused (ValueError) for a model
    that can't label the key"""
    if slot not in mmp.SLOTS:
        raise ValueError(f"no such labeller: {slot}")
    why = refused(slot, model_id)
    if why:
        raise ValueError(why)
    if not ai_models.has_key():
        raise ValueError("OpenRouter has no key on this server (OPENROUTER_API_KEY)")
    value = ai_models.pin(model_id.strip())
    db.ai_set(_setting(slot), value, by)
    return value


def problems() -> list[str]:
    """what stops the labellers set now from labelling, each in one line"""
    labs = labellers()
    out = []
    for s in mmp.SLOTS:
        why = refused(s, labs[s].get("id", ""), labs)
        if why:
            out.append(f"{SLOT_LABEL[s]} ({labs[s].get('name') or labs[s].get('id')}): {why}")
    return out


def labelled_ids() -> set[str]:
    """the labellers' ids: a row of one of them says "labelled the key" """
    return {x.get("id") for x in labellers().values() if x.get("id")}


# ---------------------------------------------------------------------------
# where the run stands
# ---------------------------------------------------------------------------

def _bdir() -> Path:
    return mmp.key_dir() / "batches"


def _meta(batch_id: str) -> dict:
    return json.loads((_bdir() / f"{batch_id}.json").read_text(encoding="utf-8"))


def pending() -> list[dict]:
    """this key's batches still out: [{batch_id, slot, ids, …}]"""
    out = []
    for r in db.batches_pending():
        if r["kind"] == KIND:
            try:
                out.append({**_meta(r["batch_id"]), "batch_id": r["batch_id"],
                            "progress": r.get("progress") or ""})
            except (OSError, ValueError):
                continue
    return out


def current() -> dict:
    """each slot's labeller as its labels are matched: {slot: {id, version}}"""
    return {s: {"id": x.get("id"), "version": x.get("version") or x.get("id")}
            for s, x in labellers().items()}


def rebuild() -> dict:
    """the key from every label kept, with the labellers set now"""
    rows = mmp.load()
    k = mmp.build_key(rows, mmp.read_labels(), current())
    mmp.write_key(k)
    return k


def left(rows: list[dict] | None = None) -> dict:
    """each slot's questions still to label: {slot: [ids]} — none already out
    in a batch"""
    rows = mmp.load() if rows is None else rows
    labels, cur = mmp.read_labels(), current()
    out_now = {}
    for p in pending():
        out_now.setdefault(p["slot"], set()).update(p.get("ids") or [])

    def has(slot: str, qid: str) -> bool:
        x = (labels.get(slot) or {}).get(qid)
        return bool(x) and x.get("version") == cur[slot]["version"]
    todo = {s: [] for s in mmp.SLOTS}
    for q in rows:
        qid = q["id"]
        for s in ("first", "second"):
            if not has(s, qid) and qid not in out_now.get(s, set()):
                todo[s].append(qid)
        if has("first", qid) and has("second", qid) and not has("third", qid) \
                and qid not in out_now.get("third", set()):
            d = mmp.decide(labels["first"][qid], labels["second"][qid])
            if d["decision"] == "waiting":
                todo["third"].append(qid)
    return todo


def estimate(stats: bool = False) -> dict:
    """the dry run: each labeller's questions, tokens and cost — what is still
    to label, the third's a share of the first two's until they have answered.
    Nothing is sent"""
    labs = labellers()
    rows = [] if stats else mmp.load()
    if rows:
        n_left = {s: len(v) for s, v in left(rows).items()}
        # the third's share is a guess while the first two have questions left
        est = mmp.label_estimate(labs, n_left, rows,
                                 guess_third=bool(n_left["first"] or n_left["second"]))
    else:
        est = mmp.label_estimate(labs)
    try:
        est.update(over_limit=ai_models.over_limit(), limit=ai_models.limit(),
                   spent=round(db.spend_this_month(), 2))
    except Exception:                               # noqa: BLE001 — no database here: the CLI's
        est.update(over_limit="", limit=None, spent=None)
    return est


def stopped() -> dict | None:
    try:
        return json.loads((mmp.key_dir() / "stopped.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# the run: Start, Stop, and each batch as it lands
# ---------------------------------------------------------------------------

class LabellerChat(llm.OpenRouterChat):
    """a labeller's batch: OpenRouter's, held while the run is stopped"""

    def waiting(self) -> str:
        return STOPPED if stopped() else super().waiting()


def _backend(pin: dict) -> llm.Backend:
    if not config.OPENROUTER_API_KEY:
        raise llm.LLMError("OpenRouter has no key on this server")
    return LabellerChat(pin["id"], config.OPENROUTER_API_KEY, config.BENCH_ROOT, pin=pin,
                        role=JOB)


def batch_backend(batch_id: str) -> llm.Backend:
    try:
        return _backend(_meta(batch_id)["pin"])
    except (OSError, ValueError, KeyError) as e:
        raise llm.LLMError(f"the key's batch {batch_id} has no record: {e}") from None


def _submit(slot: str, pin: dict, ids: list[str], by: str) -> str:
    qs = mmp.by_id()
    reqs = [llm.Request(f"mmpk:{slot}:{qid}", "", mmp.label_prompt(qs[qid]),
                        max_tokens=min(mmp.LABEL_MAX_TOKENS, config.OPENROUTER_MAX_TOKENS),
                        json=True, meta={"kind": KIND})
            for qid in ids if qid in qs]
    if not reqs:
        return ""
    be = _backend(pin)
    bid = be.submit(reqs)
    p = _bdir() / f"{bid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"slot": slot, "pin": pin, "ids": [r.custom_id.split(":", 2)[2]
                                                               for r in reqs],
                             "by": by, "at": time.time()}), encoding="utf-8")
    db.batch_add(bid, KIND, 0, len(reqs), "openrouter", pin["id"])
    return bid


def _pinned(slot: str, by: str) -> dict:
    """the slot's labeller, pinned: chosen on AI models, or its default pinned
    now and kept as the slot's choice"""
    c = chosen(slot)
    if c:
        return c
    d = mmp.DEFAULT_LABELLERS[slot]
    try:
        value = ai_models.pin(d["id"])
    except ValueError as e:
        raise ValueError(f"{SLOT_LABEL[slot]}: {e} — choose another on AI models") from None
    db.ai_set(_setting(slot), value, by)
    return value


def start(by: str) -> dict:
    """masein's Start, after the dry run: every slot pinned, the stop lifted,
    and what is left sent — the first two's questions, and the third's splits
    once the first two have answered them. Started again, it carries on"""
    why = mmp.available() or ai_models.over_limit() or ("" if ai_models.has_key() else
                                                         "OpenRouter has no key on this server "
                                                         "(OPENROUTER_API_KEY)")
    if why:
        raise ValueError(why)
    bad = problems()
    if bad:
        raise ValueError(bad[0])
    (mmp.key_dir() / "stopped.json").unlink(missing_ok=True)
    mmp._write(mmp.key_dir() / "run.json", {"by": by, "at": time.time()})
    sent = {}
    for slot in ("first", "second"):
        _pinned(slot, by)
    todo = left()
    for slot in ("first", "second"):
        ids = todo[slot]
        if ids:
            sent[slot] = {"batch_id": _submit(slot, chosen(slot), ids, by), "n": len(ids)}
    third = advance(by)
    if third:
        sent["third"] = third
    for p in pending():                       # a stopped batch takes up again
        try:
            batch_backend(p["batch_id"]).status(p["batch_id"])
        except llm.LLMError:
            pass
    return {"sent": sent, "pending": len(pending())}


def advance(by: str = "") -> dict | None:
    """the third labeller's turn: the questions the first two split on that it
    hasn't been asked — never while the run is stopped"""
    if stopped() or mmp.available():
        return None
    ids = left()["third"]
    if not ids:
        return None
    pin = _pinned("third", by or (mmp._read(mmp.key_dir() / "run.json", {}) or {}).get("by", ""))
    return {"batch_id": _submit("third", pin, ids, by), "n": len(ids)}


def stop(by: str) -> dict:
    """masein's Stop: nothing more is sent; what is in flight lands"""
    mmp._write(mmp.key_dir() / "stopped.json", {"by": by, "at": time.time()})
    return {"stopped": True, "pending": len(pending())}


def finish(batch_id: str, results: dict) -> int:
    """a batch landed: its labels kept by question id, the key rebuilt, and
    the third labeller sent what the first two split on"""
    meta = _meta(batch_id)
    pin = meta["pin"]
    got = {}
    for cid, res in results.items():
        qid = cid.split(":", 2)[2]
        if res.error:
            continue                          # asked again by the next Start
        lab = mmp.parse_label(res.text)
        got[qid] = {**lab, "model": pin["id"], "version": pin.get("version") or pin["id"],
                    "at": time.time()}
    mmp.add_labels(meta["slot"], got)
    rebuild()
    try:
        advance(meta.get("by", ""))
    except (ValueError, llm.LLMError) as e:
        print(f"[mmp key] the third labeller waits: {e}")
    return len(got)


def failed(batch_id: str, why: str) -> None:
    """a batch failed whole: what it landed is kept, the rest asked again by
    the next Start"""
    print(f"[mmp key] {batch_id} failed: {why}")


def status() -> dict:
    """the AI models page's card: the labellers, the dry run, the run, the key"""
    labs = labellers()
    k = mmp.current_key()
    out_now = pending()
    return {"available": mmp.available(),
            "labellers": [{"slot": s, "label": SLOT_LABEL[s], "does": SLOT_DOES[s],
                           "why": SLOT_WHY[s], "suggested": mmp.DEFAULT_LABELLERS[s]["id"],
                           "chosen": chosen(s), "now": labs[s],
                           "refused": refused(s, labs[s].get("id", ""), labs)}
                          for s in mmp.SLOTS],
            "problems": problems(),
            "estimate": estimate(stats=bool(mmp.available())),
            "running": [{"slot": p["slot"], "n": len(p.get("ids") or []),
                         "progress": p.get("progress", "")} for p in out_now],
            "stopped": stopped(),
            "key": {"version": k.get("version") or "", "counts": k.get("counts") or {},
                    "built_at": k.get("built_at"), "labels": mmp.DECIDED}}
