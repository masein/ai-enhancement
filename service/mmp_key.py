"""14.3: Mobile-MMLU-Pro's answer key — who labels it, and the labelling run.
The rules and the file are scripts/mobile_mmlu.py's. 14.4: one pool of labels
for Mobile-MMLU-Pro and the full Mobile-MMLU — a question both sets word alike
is labelled once — and Pro's questions go first, so Pro's key is whole before
the full set's own are sent.

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
# 16c: what is labelled — Mobile-MMLU-Pro alone (masein, 5 Oct: the default),
# or Pro and then the full set
SCOPES = {"pro": "Mobile-MMLU-Pro only", "all": "Pro, then the full set"}
SCOPE_SHORT = {"pro": "Pro only", "all": "Pro, then the full set"}


def scope() -> str:
    """"pro" or "all": what the card chose — Pro alone while the full set is
    switched off (14.4.5), and by default"""
    if not mmp.full_on():
        return "pro"
    v = db.ai_get("mmp:scope", "pro")
    return v if v in SCOPES else "pro"


def scope_meta() -> dict:
    m = db.ai_get_meta("mmp:scope") or {}
    return {"value": scope(), "words": SCOPES[scope()], "by": m.get("by") or "",
            "at": m.get("at"), "chosen": bool(m)}


def set_scope(value: str, by: str) -> dict:
    """the card's choice, kept with who chose it and when; Pro only cancels
    what is out for the full set's own questions"""
    if value not in SCOPES:
        raise ValueError(f"no such choice: {value} — {' or '.join(SCOPES)}")
    if value == "all" and not mmp.full_on():
        raise ValueError("the full Mobile-MMLU is switched off on this server "
                         "(MOBILE_MMLU_FULL=0): only Pro can be labelled")
    db.ai_set("mmp:scope", value, by)
    prune()
    return scope_meta()


class LabellerChat(llm.OpenRouterChat):
    """a labeller's batch: OpenRouter's, held while the run is stopped"""
    HALT_TAIL = ", or now with Carry on"

    def waiting(self) -> str:
        return STOPPED if stopped() else super().waiting()


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
    """every model with a Mobile-MMLU-Pro or (14.4) full Mobile-MMLU score on
    the board, by its own id and (a model from OpenRouter) the id OpenRouter
    knows it by"""
    out: set[str] = set()
    for d in config.OUT_DIR.iterdir() if config.OUT_DIR.is_dir() else []:
        if not ((d / mmp.PRED_FILE).exists() or (d / mmp.FULL_PRED_FILE).exists()):
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
        return (f"{mid} has a Mobile-MMLU score on the board: a model can't label the key "
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
    value = ai_models.pin(model_id.strip(), JOB)
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
    """the key from every label kept, with the labellers set now — the pool's,
    and each set's view of it"""
    k = mmp.build_key(mmp.pool(), mmp.read_labels(), current())
    mmp.write_key(k)
    return k


def _out_now() -> dict[str, set]:
    out: dict[str, set] = {}
    for p in pending():
        out.setdefault(p["slot"], set()).update(p.get("ids") or [])
    return out


def _rows() -> list[dict]:
    """the questions labelling may send: the pool — Pro's alone while the full
    set is switched off (14.4.5), or (16c) while Pro alone is chosen. The key
    keeps every label either way"""
    rows = mmp.pool()
    return rows if scope() == "all" else [q for q in rows if "pro" in q["sets"]]


def prune() -> int:
    """16c: a batch out for questions that aren't chosen now is cancelled,
    never left waiting — one held at the monthly limit would take up again by
    itself when the month turns. Returns how many questions were cancelled.
    16c review: nothing is cancelled unless Pro's file was read and the pool
    holds Pro's questions — an unreadable file gives an empty pool, and every
    question out, Pro's included, would have gone"""
    if scope() == "all" or mmp.available():
        return 0
    try:
        mine = {q["lid"] for q in _rows() if "pro" in q["sets"]}
    except Exception:                               # noqa: BLE001 — no file here: nothing out
        return 0
    if not mine:
        return 0
    n = 0
    for p in pending():
        drop = [f"mmpk:{p['slot']}:{k}" for k in p.get("ids") or [] if k not in mine]
        if drop:
            try:
                n += batch_backend(p["batch_id"]).cancel(
                    p["batch_id"], drop, "the full set isn’t chosen (Pro only)")
            except llm.LLMError:
                continue
    return n


def left(rows: list[dict] | None = None) -> dict:
    """each slot's questions still to label, by label id: {slot: [lids]}, in
    the pool's order (Pro's first) — none already out in a batch"""
    rows = _rows() if rows is None else rows
    labels, cur = mmp.read_labels(), current()
    out_now = _out_now()

    def has(slot: str, k: str) -> bool:
        x = (labels.get(slot) or {}).get(k)
        return bool(x) and x.get("version") == cur[slot]["version"]
    todo = {s: [] for s in mmp.SLOTS}
    for q in rows:
        k = q["lid"]
        for s in ("first", "second"):
            if not has(s, k) and k not in out_now.get(s, set()):
                todo[s].append(k)
        if has("first", k) and has("second", k) and not has("third", k) \
                and k not in out_now.get("third", set()):
            d = mmp.decide(labels["first"][k], labels["second"][k])
            if d["decision"] == "waiting":
                todo["third"].append(k)
    return todo


def pro_open(rows: list[dict] | None = None, todo: dict | None = None) -> bool:
    """14.4: Pro's key isn't whole yet: one of Pro's questions is still to
    label, or out in a batch, in any slot"""
    rows = _rows() if rows is None else rows
    pro = {q["lid"] for q in rows if "pro" in q["sets"]}
    todo = left(rows) if todo is None else todo
    labels, cur = mmp.read_labels(), current()

    def landed(slot: str, k: str) -> bool:
        # a batch still on the books as it lands: its labels are kept already
        x = (labels.get(slot) or {}).get(k)
        return bool(x) and x.get("version") == cur[slot]["version"]
    out_now = {s: [k for k in v if not landed(s, k)] for s, v in _out_now().items()}
    return any(k in pro for s in mmp.SLOTS for k in [*todo[s], *out_now.get(s, ())])


def due(rows: list[dict] | None = None) -> dict:
    """14.4: what is sent now, per slot — Pro's questions while Pro's key isn't
    whole, then every question left"""
    rows = _rows() if rows is None else rows
    todo = left(rows)
    if not pro_open(rows, todo):
        return todo
    pro = {q["lid"] for q in rows if "pro" in q["sets"]}
    return {s: [k for k in v if k in pro] for s, v in todo.items()}


def _set_left(rows: list[dict], todo: dict, name: str) -> dict:
    """a set's questions still to label, per slot, as counts"""
    mine = {q["lid"] for q in rows if name in q["sets"]}
    return {s: sum(1 for k in v if k in mine) for s, v in todo.items()}


def estimate(stats: bool = False) -> dict:
    """the dry run: each labeller's questions, tokens and cost — what is still
    to label, the third's a share of the first two's until they have answered.
    Nothing is sent. 14.4: for Pro alone, for the full set, and for both (what
    Start sends, Pro's first): {pro, full, all} — and the top level is "all",
    as 14.3's one estimate was"""
    labs = labellers()
    pro_rows = [] if stats else mmp.load()
    full_rows = [] if stats or not pro_rows else mmp.load_full()
    out: dict = {}
    if pro_rows:
        rows = mmp.pool(pro_rows, full_rows)
        todo = left(rows)
        for name, mine in (("pro", pro_rows), ("full", full_rows), ("all", rows)):
            if not mine:
                continue
            n_left = (_set_left(rows, todo, name) if name != "all"
                      else {s: len(v) for s, v in todo.items()})
            # the third's share is a guess while the first two have questions left
            out[name] = mmp.label_estimate(labs, n_left, mine,
                                           guess_third=bool(n_left["first"] or n_left["second"]))
    if not mmp.full_on():                           # 14.4.5: switched off: Pro alone
        out = {"pro": out.get("pro") or mmp.label_estimate(labs)}
        out["all"] = out["pro"]
    if "full" not in out and mmp.full_on():         # the set's published lengths, without its files
        out.setdefault("pro", mmp.label_estimate(labs))
        out["full"] = {**mmp.label_estimate(labs, which="full"),
                       "missing": mmp.full_available()}
        if "all" not in out:
            out["all"] = mmp.label_estimate(labs, which="all")
    est = {**out["all"], "sets": out}
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
    """a batch of questions, by label id (14.4: each asked in its own wording)"""
    qs = {q["lid"]: q for q in mmp.pool()}
    reqs = [llm.Request(f"mmpk:{slot}:{k}", "", mmp.label_prompt(qs[k]),
                        max_tokens=min(mmp.LABEL_MAX_TOKENS, config.OPENROUTER_MAX_TOKENS),
                        json=True, meta={"kind": KIND})
            for k in ids if k in qs]
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
        value = ai_models.pin(d["id"], JOB)
    except ValueError as e:
        # 16c: in OpenRouter's words (ai_models.refusal), "choose another" once
        raise ValueError(f"{SLOT_LABEL[slot]}: {e}") from None
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
    # 16c: everything checked before anything is sent — all three labellers
    # pinned, and each still the version it was pinned to. One that can't be
    # sends nothing, and says why
    for slot in mmp.SLOTS:
        drift = ai_models.drifted(_pinned(slot, by))
        if drift:
            raise ValueError(f"{SLOT_LABEL[slot]}: {drift}")
    (mmp.key_dir() / "stopped.json").unlink(missing_ok=True)
    prune()
    rows = _rows()
    now = due(rows)
    # 14.4: Pro's questions first; the rest once Pro's key is whole — sent then
    # by advance(), once a Start (16c: only when the full set is chosen)
    first_round_is_all = not pro_open(rows)
    mmp._write(mmp.key_dir() / "run.json", {"by": by, "at": time.time(),
                                             "rest_sent": first_round_is_all})
    sent = {}
    for slot in ("first", "second"):
        ids = now[slot]
        if ids:
            sent[slot] = {"batch_id": _submit(slot, chosen(slot), ids, by), "n": len(ids)}
    third = advance(by)
    if third:
        sent["third"] = third
    for p in pending():                       # a stopped or halted batch takes up again
        try:
            be = batch_backend(p["batch_id"])
            be.resume(p["batch_id"])
            be.status(p["batch_id"])
        except llm.LLMError:
            pass
    return {"sent": sent, "pending": len(pending())}


def advance(by: str = "") -> dict | None:
    """the third labeller's turn: the questions the first two split on that it
    hasn't been asked — Pro's first (14.4) — never while the run is stopped.
    And once Pro's key is whole, the full set's own questions to the first
    two, once a Start (a failed one waits for the next, as 14.3's do)"""
    if stopped() or mmp.available():
        return None
    run = mmp._read(mmp.key_dir() / "run.json", {}) or {}
    by = by or run.get("by", "")
    rows = _rows()
    now = due(rows)
    out = None
    if now["third"]:
        pin = _pinned("third", by)
        out = {"batch_id": _submit("third", pin, now["third"], by), "n": len(now["third"])}
    if not run.get("rest_sent") and not pro_open(rows):
        rest = {}
        for slot in ("first", "second"):
            ids = left(rows)[slot]
            if ids:
                rest[slot] = {"batch_id": _submit(slot, _pinned(slot, by), ids, by),
                              "n": len(ids)}
        mmp._write(mmp.key_dir() / "run.json", {**run, "rest_sent": True})
        if rest:
            out = {**(out or {}), "rest": rest}
    return out


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
        k = cid.split(":", 2)[2]                # 14.4: the question's label id
        if res.error:
            continue                          # asked again by the next Start
        lab = mmp.parse_label(res.text)
        got[k] = {**lab, "model": pin["id"], "version": pin.get("version") or pin["id"],
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


def _progress(p: dict) -> str:
    """16c: a batch out, as it goes: "1,859 of 2,714 · about 9 min left ·
    $27.67 so far" — the pace from its last answers"""
    tl = llm.tally(p["batch_id"])
    if not tl:
        return p.get("progress") or ""
    done = tl["answered"] + tl["failed"] + tl["cancelled"]
    bits = [f"{done:,} of {tl['sent']:,}"]
    r = tl.get("recent") or []
    if done < tl["sent"] and not tl.get("halted") and len(r) >= 5 and r[-1] > r[0]:
        secs = (tl["sent"] - done) * (r[-1] - r[0]) / (len(r) - 1)
        bits.append(f"about {max(1, round(secs / 60)):,} min left" if secs < 5400
                    else f"about {secs / 3600:.1f} h left")
    bits.append(f"${db.spend_of_batch(p['batch_id']):,.2f} so far")
    line = " · ".join(bits)
    why = tl.get("halted") or ""
    return f"{line} · {why}" if why else line


def last_batches() -> dict:
    """16c: each labeller's last batch, in one line (llm.batch_line) — only
    one that failed, part or whole, or waits at a run of refusals"""
    out: dict = {}
    for r in db.batches_list(500):
        if r["kind"] != KIND:
            continue
        try:
            slot = _meta(r["batch_id"])["slot"]
        except (OSError, ValueError, KeyError):
            continue
        if slot in out:
            continue
        tl = llm.tally(r["batch_id"])
        err = r.get("error") if r.get("status") == "failed" else ""
        out[slot] = {"batch_id": r["batch_id"], "status": r.get("status"),
                     "failed": bool((tl or {}).get("failed") or err or (tl or {}).get("halted")),
                     "line": f"{SLOT_LABEL[slot]}’s last batch: {llm.batch_line(tl, err or '')}"}
    return out


def done_for_scope() -> bool:
    """16c: what was chosen is labelled — nothing left to send, nothing out"""
    try:
        return not pending() and not any(left().values())
    except Exception:                               # noqa: BLE001 — no file: not done
        return False


def status() -> dict:
    """the AI models page's card: the labellers, the dry run, the run, the key"""
    labs = labellers()
    k = mmp.current_key()
    out_now = pending()
    f = mmp.full_key()
    est = estimate(stats=bool(mmp.available()))
    sc = scope()
    # 16c: what Start sends now — the chosen set's cost — and what the
    # OpenRouter key may still spend
    sends = ((est.get("sets") or {}).get("pro") if sc == "pro" else est) or est
    return {"available": mmp.available(), "full_available": mmp.full_available(),
            # 14.4.5: switched off, the card shows Pro alone
            "full_on": mmp.full_on(),
            "scope": scope_meta(), "scopes": SCOPES if mmp.full_on() else {"pro": SCOPES["pro"]},
            "sends": {"usd": sends.get("usd"), "usd_known": sends.get("usd_known"),
                      "questions": sends.get("questions"), "words": SCOPE_SHORT[sc]},
            "done": not mmp.available() and done_for_scope(),
            "key_warning": ai_models.more_than_key(sends.get("usd")),
            "key_allowance": ai_models.key_allowance(),
            # 14.4: each set's view of the one key
            "sets": {"pro": {"version": k.get("version") or "", "counts": k.get("counts") or {}},
                     **({"full": {"version": f.get("version") or "",
                                  "counts": f.get("counts") or {}}} if mmp.full_on() else {})},
            "labellers": [{"slot": s, "label": SLOT_LABEL[s], "does": SLOT_DOES[s],
                           "why": SLOT_WHY[s], "suggested": mmp.DEFAULT_LABELLERS[s]["id"],
                           "chosen": chosen(s), "now": labs[s],
                           "refused": refused(s, labs[s].get("id", ""), labs)}
                          for s in mmp.SLOTS],
            "problems": problems(),
            "estimate": est,
            "running": [{"slot": p["slot"], "n": len(p.get("ids") or []),
                         "progress": _progress(p)} for p in out_now],
            "last": last_batches(),
            "stopped": stopped(),
            "key": {"version": k.get("version") or "", "counts": k.get("counts") or {},
                    "built_at": k.get("built_at"), "labels": mmp.DECIDED}}
