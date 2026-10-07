"""17, stage 3: the Frontier benchmarks graded on the server — SimpleQA
Verified and Humanity's Last Exam by their owners' graders, and MATH Level 5
and OTIS Mock AIME given Epoch AI's model check as a second look on what the
code marks wrong (scripts/frontier_graders.py has each prompt and its reading).

**The graders.** One slot each, a model on OpenRouter pinned as the judge is
(ai_models.pin: its dated version, its first provider, no fallbacks), chosen
on AI models. A slot nobody has chosen for keeps its suggestion — the owners'
model where it is still served — pinned when Start is pressed. Never local:
a rented box holds no key, and the server's own judge isn't the owners'.

**The run.** Nothing is sent by itself: an import or a run on the board
scores what code can and leaves the rest waiting. The AI models page shows the
dry run — each grader's answers, tokens and cost — and sends only on masein's
Start; Stop holds it, and Start carries on. A batch is OpenRouter's, worked a
few requests at a time (llm.OpenRouterChat); its spend counts against the
month's AI limit as the "grader" job. As a batch lands, each answer's grade is
kept beside the answers (grades.json: the grader's pin and its prompt's
sha256, and each answer's grade), and the benchmark is scored again. An answer
the grader refuses is kept as refused, in its words, and asked again by the
next Start.

17b, before anyone presses Start:
- **Start once.** Start lists what waits and sends it under a lock, across
  threads and processes: a second press waits, then finds it out.
- **Recorded, then sent.** A batch is on disk and in the database before its
  first request goes; a batch nobody recorded is never sent.
- **A grade is a grade.** An empty reply, one cut at its cap, one that can't
  be read as a grade, or one `finish()` fails on (one answer at a time) is
  listed beside the refusals, and asked again by the next Start. Each
  grader's reasoning is set and its cap sized for it (frontier_graders.ask);
  the dry run counts with the same numbers, and says what the caps allow.
- **The answer it graded.** A grade keeps the sha256 of the answer it was
  given, its grader's version and its prompt's sha256. An answer replaced
  while its batch was out is graded again; a score from more than one
  grader or prompt says so, and isn't final.
- **The card** says why a batch failed and why one waits (Stop, a run of
  refusals, the month's limit), with Carry on; and that choosing a grader
  asks each provider one paid token, counted in the month's spend.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import os
import sys
import threading
import time
from pathlib import Path

from . import ai_models, config, db, llm
from . import frontier as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import frontier as fb  # noqa: E402
import frontier_graders as fg  # noqa: E402

KIND = "frgr"
JOB = "grader"
STOPPED = "stopped — Start carries on where it stopped"
CHARS_A_TOKEN = 4                      # the dry run's count of a prompt, before it is sent
PROBE_WORDS = ("Choosing a grader asks each of its providers one token first, to check it "
               "keeps no prompt (a fraction of a cent each), counted in this month's spend.")
_START = threading.Lock()
# 17c: an answer whose reply was no grade this many times is listed as
# ungraded (counted wrong) and never sent again; a batch whose replies can't be
# recorded is tried this many times before it fails
GRADE_TRIES = sf.GRADE_TRIES


class GraderChat(llm.OpenRouterChat):
    """a grader's batch: OpenRouter's, held while the grading is stopped —
    17f: and stopped, its refusal said, when its first replies all refuse"""
    HALT_TAIL = ", or now with Start"
    FIRST_REFUSALS = 5

    def waiting(self) -> str:
        return STOPPED if stopped() else super().waiting()


def gdir() -> Path:
    return config.BENCH_ROOT / "frontier" / "grading"


def _setting(slot: str) -> str:
    return "grader:" + slot


def chosen(slot: str) -> dict | None:
    return db.ai_get(_setting(slot))


def _cached(model_id: str) -> dict | None:
    try:
        got = json.loads(ai_models._cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return next((m for m in got.get("models") or [] if m.get("id") == model_id), None)


def grader(slot: str) -> dict:
    """the slot's grader: chosen on AI models, else its suggestion — {kind,
    id, name, version, provider, price_in, price_out}"""
    c = chosen(slot)
    if c:
        return c
    g = fg.GRADERS[slot]
    m = _cached(g["suggested"]) or {}
    return {"kind": "default", "id": g["suggested"], "name": m.get("name") or g["suggested"],
            "version": m.get("version") or "", "price_in": m.get("price_in"),
            "price_out": m.get("price_out")}


def save(slot: str, model_id: str, by: str) -> dict:
    """a grader, pinned on OpenRouter — never local"""
    if slot not in fg.GRADERS:
        raise ValueError(f"no such grader: {slot}")
    if model_id == ai_models.LOCAL:
        raise ValueError("a grader is a model on OpenRouter: the server's own judge isn't the "
                         "benchmark owners'")
    value = ai_models.pin(model_id, JOB)
    db.ai_set(_setting(slot), value, by)
    # 17g: nothing moves until Start — the dry run shows what it would do
    # (17f moved the first's grades aside here, and choosing it again priced
    # every answer again)
    return value


def stopped() -> dict | None:
    try:
        return json.loads((gdir() / "stopped.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stop(by: str) -> dict:
    """masein's Stop: nothing more is sent; what is in flight lands"""
    gdir().mkdir(parents=True, exist_ok=True)
    (gdir() / "stopped.json").write_text(json.dumps({"by": by, "at": time.time()}),
                                         encoding="utf-8")
    return {"stopped": True, "pending": len(pending())}


# ---------------------------------------------------------------------------
# what waits for a grader
# ---------------------------------------------------------------------------

def _meta(batch_id: str) -> dict:
    return json.loads((gdir() / "batches" / f"{batch_id}.json").read_text(encoding="utf-8"))


def pending() -> list[dict]:
    """the batches out now, each with its record"""
    out = []
    for r in db.batches_pending():
        if r["kind"] != KIND:
            continue
        try:
            out.append({**_meta(r["batch_id"]), "batch_id": r["batch_id"],
                        "progress": r.get("progress") or ""})
        except (OSError, ValueError):
            continue
    return out


def _model_of(row: Path) -> tuple[str, str]:
    """(the row's model id, the served model it is a row of)"""
    try:
        m = json.loads((row / "model_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        m = {}
    mid = m.get("model") or row.name.replace("__", "/", 1)
    return mid, m.get("base_model") or mid.replace(" · thinking", "")


def _cancelled(batch_id: str) -> set[str]:
    """17c: the answers of a batch whose requests were cancelled, unsent — 17d:
    once its worker is done (a request in flight when it was cancelled lands
    after, as its own reply), and only those whose last word is the cancel"""
    d = llm.batch_dir(batch_id)
    if d is None or _working(batch_id):
        return set()
    return {c.split(":", 1)[1] for c, r in llm.LocalOpenAI._results(d).items()
            if r.get("cancelled") and ":" in c}


def _busy(batch_id: str) -> bool:
    """17i: a batch with replies still on their way — its worker holding its
    lock now and a moment later (a stopped batch's worker, started by a tick,
    ends at once)"""
    if not _working(batch_id):
        return False
    time.sleep(0.2)
    return _working(batch_id)


def held_rows(moving: list[dict] | None = None) -> dict:
    """17i: the rows a batch Start moves still has replies on their way for —
    {(row, task): the grader they are at}: Start leaves them until the replies
    land (they were paid, then the same answers paid again at the grader
    chosen now), and the dry run says so"""
    out = {}
    for p in pending() if moving is None else moving:
        if (moving is not None or _moving(p)) and _busy(p["batch_id"]):
            out[(p["row"], p["task"])] = p["pin"].get("version") or p["pin"].get("id")
    return out


def waiting(view: bool = False, hold: dict | None = None) -> list[dict]:
    """every row's answers a grader is still to see, by benchmark — those out
    in a batch now left out (17c: but not those cancelled from it, unsent):
    [{slot, task, row, model, base, items}]. 17g: `view` — as Start would
    leave them with the grader chosen now (the dry run), each row saying what
    the switch does (`switch`: kept, regrade, match) and how many grades
    the grader gave before come back (`reused`)"""
    # 17h: in the dry run, a batch Start moves to the grader chosen now
    # (stopped, then another chosen) holds nothing — its answers are priced
    # here, at the grader they go to (they were priced at the stopped one's)
    out = pending()
    moving = [p for p in out if view and _moving(p)]
    if view and hold is None:
        hold = held_rows(moving)
    hold = hold or {}
    out_now = {(p["row"], p["task"], k) for p in out if p not in moving
               for k in set(p.get("keys") or []) - _cancelled(p["batch_id"])}
    got = []
    root = Path(config.OUT_DIR)
    for slot, g in fg.GRADERS.items():
        t = g["task"]
        now = chosen(slot) if view else None
        for row in sorted(root.glob(f"*/{t}_0shot/{sf.SUB}")) if root.is_dir() else []:
            d, row = row, row.parent.parent
            if (row.name, t) in hold:
                continue                    # 17i: its replies in flight land first
            seen, how, reused = None, "", 0
            if now:
                seen = copy.deepcopy(sf.read_grades(d))
                # 17h: what a moved batch landed, recorded as Start records it
                for p in moving:
                    if p["row"] == row.name and p["task"] == t:
                        _apply(seen, d, p["slot"], p["pin"], p, _items(t), _landed(p))
                how, reused = _switch(seen, now.get("version") or now.get("id"),
                                      fg.prompt_sha(slot))
            try:
                todo = sf.to_grade(row, t, grades=seen)
            except Exception:                           # noqa: BLE001 — no dataset here
                continue
            todo = [x for x in todo if (row.name, t, sf.gkey(x["id"], x["epoch"])) not in out_now]
            if todo or how:
                mid, base = _model_of(row)
                # 17h: nothing to send, the switch alone — what Start leaves the
                # score as, said before it (a $0 Start could take a final away)
                after = sf.state_of(row, t, seen) if how and not todo else None
                got.append({"slot": slot, "task": t, "row": row.name, "model": mid,
                            "base": base, "items": todo,
                            **({"switch": how, "why": (seen or {}).get("regrade", {}).get("why")
                                or "", "reused": reused} if how else {}),
                            **({"after": after["words"], "after_state": after["state"]}
                               if after else {})})
    return got


def _items(task: str) -> dict:
    return {it["id"]: it for it in fb.load(task, config.BENCH_ROOT)}


def _landed(p: dict) -> dict:
    """a batch's replies on disk, never its cancelled requests — as finish()
    reads them"""
    d = llm.batch_dir(p["batch_id"])
    if d is None:
        return {}
    return {c: llm.Result(text=r.get("text") or "", error=r.get("error") or "",
                          finish=r.get("finish_reason") or "", status=r.get("status"),
                          kind=r.get("kind") or "")
            for c, r in llm.LocalOpenAI._results(d).items() if not r.get("cancelled")}


def _moving(p: dict) -> str:
    """17c/17h: why Start moves a batch out to the grader pinned now — its
    grader drifted on OpenRouter, or another was chosen since — '' if not"""
    now = chosen(p["slot"]) or {}
    return ai_models.drifted(p["pin"]) or (
        "another grader was chosen" if now and now.get("version") != p["pin"].get("version")
        else "")


def _price(g: dict) -> tuple[float | None, float | None]:
    return g.get("price_in"), g.get("price_out")


def structured(g: dict) -> bool | None:
    """17f: whether the grader takes a JSON schema, as OpenRouter lists it"""
    m = _cached(g.get("id") or "") or {}
    return m.get("structured")


def reasons(g: dict) -> bool | None:
    """whether OpenRouter lists the grader as a model that reasons"""
    return (_cached(g.get("id") or "") or {}).get("reasons")


def estimate() -> dict:
    """the dry run: each grader's answers, tokens and cost for what waits.
    Nothing is sent"""
    hold = held_rows()
    est = _estimate(waiting(view=True, hold=hold))
    try:
        est.update(over_limit=ai_models.over_limit(), limit=ai_models.limit(),
                   spent=round(db.spend_this_month(), 2))
    except Exception:                               # noqa: BLE001 — no database here
        est.update(over_limit="", limit=None, spent=None)
    est.update(_limit_words(est, held()))
    if hold:
        est["held"] = _held_words(hold)
    return est


def _held_words(hold: dict) -> list[dict]:
    """17i: the rows Start leaves until their replies land, in words"""
    return [{"row": row, "task": task, "label": fb.BENCH[task]["label"], "at": at,
             "words": f"{fb.BENCH[task]['label']} of {row}: replies on their way at {at} — "
                      "left as it is until they land, then the next Start moves it to the "
                      "grader chosen now (those answers are never asked twice)"}
            for (row, task), at in sorted(hold.items())]


def _limit_words(est: dict, h: dict) -> dict:
    """17i: the month's limit against what Start costs — the dry run's new
    answers and what the batches out still hold (`h`, held()) — {left, short,
    may_stop}: `short` when that is more than what is left (Start refuses it
    unless asked to stop part-way), `may_stop` when only its most could"""
    lim, spent = est.get("limit"), est.get("spent")
    if lim is None or spent is None or not (est.get("answers") or h.get("answers")):
        return {}
    usd = float(est.get("usd") or 0) + float(h.get("usd") or 0)
    left = max(0.0, round(float(lim) - float(spent), 2))
    out: dict = {"left": left, "cost": round(usd, 4)}
    if est.get("usd_known", True) and h.get("usd_known", True) and usd > left:
        out["short"] = (f"This month's AI limit has ${left:,.2f} left (${float(spent):,.2f} of "
                        f"${float(lim):,.2f} spent), and this grading costs about "
                        f"${float(usd):,.2f}: it would stop part-way, at the limit, and carry "
                        "on by itself when the limit is raised. Raise the limit on AI models "
                        "first, or start it knowing it stops there.")
    elif float(est.get("usd_max") or 0) + float(h.get("usd") or 0) > left:
        most = float(est.get("usd_max") or 0) + float(h.get("usd") or 0)
        out["may_stop"] = (f"This month's AI limit has ${left:,.2f} left: about ${usd:,.2f} "
                           f"fits, but at its most (every reply at its cap, asked "
                           f"{GRADE_TRIES} times) this could reach ${most:,.2f} and stop "
                           "part-way.")
    return out


def _short(work: list[dict]) -> str:
    """17i: Start's own check, before anything is sent — the words, or ''"""
    est = _estimate(work)
    try:
        est.update(limit=ai_models.limit(), spent=round(db.spend_this_month(), 2))
    except Exception:                               # noqa: BLE001 — no database here
        return ""
    return _limit_words(est, held()).get("short", "")


def _estimate(ws: list[dict]) -> dict:
    """the dry run's sums over what `waiting(view=True)` gave"""
    fb.set_root(config.BENCH_ROOT)
    per: dict[str, dict] = {}
    rows = []
    for w in ws:
        slot = w["slot"]
        if not w["items"]:
            # 17g: nothing to send — the grades it gave before come back
            rows.append({"slot": slot, "task": w["task"], "label": fb.BENCH[w["task"]]["label"],
                         "model": w["model"], "answers": 0, "usd": 0.0,
                         "switch": w["switch"], "reused": w.get("reused") or 0,
                         "after": w.get("after") or "", "after_state": w.get("after_state")})
            continue
        g = grader(slot)
        items = {it["id"]: it for it in fb.load(w["task"], config.BENCH_ROOT)}
        tin = sum(len(fg.render(slot, items[x["id"]], x)) for x in w["items"]) // CHARS_A_TOKEN
        # 17b: what Start sends with each answer — its reasoning and its cap
        a = fg.ask(slot, reasons(g), structured(g))
        cap = min(a["max_tokens"], config.OPENROUTER_MAX_TOKENS)
        tout = len(w["items"]) * a["out_tokens"]
        tmax = len(w["items"]) * cap
        pin, pout = _price(g)
        # prices are per million tokens, as OpenRouter's list gives them
        usd = ((tin * pin + tout * pout) / 1e6 if pin is not None and pout is not None
               else None)
        # 17d: at most, every reply at its cap, and each answer asked its
        # GRADE_TRIES times (a reply that isn't a grade is asked again)
        most = (GRADE_TRIES * (tin * pin + tmax * pout) / 1e6
                if pin is not None and pout is not None else None)
        e = per.setdefault(slot, {"answers": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0,
                                  "usd_known": True, "tokens_out_max": 0, "usd_max": 0.0,
                                  "cap": cap, "reasoning": a["reasoning"],
                                  # 17f: JSON only, where the grader takes a schema
                                  "json_only": bool(a["schema"]),
                                  "reasoning_words": fg.GRADERS[slot]["reasoning_words"]})
        e["answers"] += len(w["items"])
        e["tokens_in"] += tin
        e["tokens_out"] += tout
        e["tokens_out_max"] += tmax
        if usd is None:
            e["usd_known"] = False
        else:
            e["usd"] += usd
            e["usd_max"] += most
        rows.append({"slot": slot, "task": w["task"], "label": fb.BENCH[w["task"]]["label"],
                     "model": w["model"], "answers": len(w["items"]),
                     "usd": None if usd is None else round(usd, 4),
                     # 17f: graded again whole by the grader chosen now, said.
                     # 17g: as Start would do it; and the grades it gave before
                     **({"regrade": w["why"]} if w.get("switch") in ("regrade", "match")
                        else {}),
                     **({"switch": w["switch"], "reused": w.get("reused") or 0}
                        if w.get("switch") else {})})
    for e in per.values():
        e["usd"], e["usd_max"] = round(e["usd"], 4), round(e["usd_max"], 4)
    total = round(sum(e["usd"] for e in per.values()), 4)
    return {"graders": per, "rows": rows, "answers": sum(e["answers"] for e in per.values()),
            "usd": total, "usd_max": round(sum(e["usd_max"] for e in per.values()), 4),
            "usd_known": all(e["usd_known"] for e in per.values())}


# ---------------------------------------------------------------------------
# the run: Start, Stop, and each batch as it lands
# ---------------------------------------------------------------------------

def _backend(pin: dict) -> llm.Backend:
    if not config.OPENROUTER_API_KEY:
        raise llm.LLMError("OpenRouter has no key on this server")
    return GraderChat(pin["id"], config.OPENROUTER_API_KEY, config.BENCH_ROOT, pin=pin,
                      role=JOB)


def batch_backend(batch_id: str) -> llm.Backend:
    try:
        return _backend(_meta(batch_id)["pin"])
    except (OSError, ValueError, KeyError) as e:
        raise llm.LLMError(f"the grading batch {batch_id} has no record: {e}") from None


def _pinned(slot: str, by: str) -> dict:
    c = chosen(slot)
    if c:
        return c
    try:
        value = ai_models.pin(fg.GRADERS[slot]["suggested"], JOB)
    except ValueError as e:
        raise ValueError(f"{fg.GRADERS[slot]['label']}: {e}") from None
    db.ai_set(_setting(slot), value, by)
    return value


def _submit(w: dict, pin: dict, by: str) -> str:
    slot, task = w["slot"], w["task"]
    items = {it["id"]: it for it in fb.load(task, config.BENCH_ROOT)}
    a = fg.ask(slot, reasons(pin), structured(pin))
    todo = [x for x in w["items"] if x["id"] in items]
    reqs = [llm.Request(f"frgr:{sf.gkey(x['id'], x['epoch'])}", "",
                        fg.render(slot, items[x["id"]], x),
                        max_tokens=min(a["max_tokens"], config.OPENROUTER_MAX_TOKENS),
                        reasoning=a["reasoning"], meta={"kind": KIND}, schema=a["schema"])
            for x in todo]
    if not reqs:
        return ""
    # 17b: on disk and in the database before the first request goes
    be = _backend(pin)
    bid = be.submit(reqs, start=False)
    p = gdir() / "batches" / f"{bid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"slot": slot, "task": task, "row": w["row"], "model": w["model"],
                             "base": w["base"], "pin": pin, "prompt_sha256": fg.prompt_sha(slot),
                             "keys": [r.custom_id.split(":", 1)[1] for r in reqs],
                             # the answers sent: a grade counts for these alone
                             "answers": {sf.gkey(x["id"], x["epoch"]): sf.answer_sha(x["answer"])
                                         for x in todo},
                             "max_tokens": reqs[0].max_tokens, "reasoning": a["reasoning"],
                             "by": by, "at": time.time()}), encoding="utf-8")
    db.batch_add(bid, KIND, 0, len(reqs), "openrouter", pin["id"])
    be.resume(bid)
    return bid


@contextlib.contextmanager
def _one_start():
    """17b: one Start at a time — in this process, and across processes"""
    gdir().mkdir(parents=True, exist_ok=True)
    with _START:
        fd = os.open(gdir() / "start.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)


def start(by: str, partial: bool = False) -> dict:
    """masein's Start, after the dry run: each grader with answers to see
    pinned, the stop lifted, and what waits sent. Started again, it carries on.
    17b: what waits is listed inside the lock — a second press finds the
    first one's batches out, and sends nothing twice"""
    with _one_start():
        return _start(by, partial)


def _start(by: str, partial: bool = False) -> dict:
    why = ai_models.over_limit() or ("" if ai_models.has_key() else
                                     "OpenRouter has no key on this server (OPENROUTER_API_KEY)")
    if why:
        raise ValueError(why)
    # everything checked before anything changes — 17d: before the stop is
    # lifted too, so a refused Start leaves a stopped grading stopped
    moved = [(p, why) for p in pending() for why in [_moving(p)] if why]
    # 17i: a moved batch with replies still on their way: its row waits for
    # them, said — Start waited 30 s, then switched under them, and the same
    # answers were paid again at the grader chosen now
    hold = held_rows([p for p, _ in moved])
    # 17g: what waits as Start leaves it — the grader chosen now taking over
    # (its own grades back, a whole regrade, a top-up) — decided before
    # anything changes, done once the stop is lifted
    work = [w for w in waiting(view=True, hold=hold) if w["items"]]
    # 17i: the month's limit — what this Start costs against what is left,
    # before anything is sent
    if not partial:
        short = _short(work)
        if short:
            raise ValueError(short)
    pins = {}
    for slot in sorted({w["slot"] for w in work} | {p["slot"] for p, _ in moved}):
        pins[slot] = _pinned(slot, by)
        drift = ai_models.drifted(pins[slot])
        if drift:
            raise ValueError(f"{fg.GRADERS[slot]['label']}: {drift}")
    (gdir() / "stopped.json").unlink(missing_ok=True)
    # 17c: a batch out whose grader moved (OpenRouter repointed its id, or
    # another was chosen since) sends nothing more: its unsent requests are
    # cancelled, and go to the grader pinned now, with what waits. 17d: those
    # in flight land first — they are never asked twice. 17h: and what landed
    # is recorded now, before the grader chosen now takes over (recorded
    # later, it met that grader's grades of the same answers)
    cancelled = {p["batch_id"]: _cancel_unsent(p, why) for p, why in moved}
    for p, _ in moved:
        if (p["row"], p["task"]) not in hold:
            _settle([p["batch_id"]], timeout=5.0)
            _close(p["batch_id"])
    for slot in fg.GRADERS:
        pin = pins.get(slot) or chosen(slot)
        if pin:
            _reset_tries(slot, pin, hold)
    work = waiting(hold=hold)
    skipped = []
    # 17e: a slot new to the work — another's batch landed with replies to ask
    # again while this waited — pinned now; one whose pin moved waits, said
    new_slots = sorted({w["slot"] for w in work} - set(pins))
    for slot in new_slots:
        pin = _pinned(slot, by)
        drift = ai_models.drifted(pin)
        if drift:
            skipped.append(f"{fg.GRADERS[slot]['label']}: {drift}")
            continue
        pins[slot] = pin
        _reset_tries(slot, pin, hold)
    if new_slots:
        work = waiting(hold=hold)
    work = [w for w in work if w["slot"] in pins]
    # 17e: moved are those still cancelled once the requests in flight landed
    # (Start said 6 when 4 moved: 2 were in flight, and landed)
    n_moved = sum(_still_cancelled(b, ids) for b, ids in cancelled.items())
    sent = []
    for w in work:
        bid = _submit(w, pins[w["slot"]], by)
        if bid:
            sent.append({"batch_id": bid, "slot": w["slot"], "task": w["task"],
                         "model": w["model"], "n": len(w["items"])})
    for p in pending():                       # a stopped or halted batch takes up again
        try:
            be = batch_backend(p["batch_id"])
            be.resume(p["batch_id"])
            be.status(p["batch_id"])
        except llm.LLMError:
            pass
    return {"sent": sent, "pending": len(pending()), "moved": n_moved,
            **({"skipped": skipped} if skipped else {}),
            **({"held": [{"row": r, "task": t, "at": v} for (r, t), v in hold.items()]}
               if hold else {})}


SETTLE_S = 30.0                         # how long Start waits for requests in flight


def _close(batch_id: str) -> None:
    """17h: a moved batch, every request answered or cancelled, recorded and
    closed now — as the poller would at its next tick"""
    try:
        be = batch_backend(batch_id)
        state, detail = be.status(batch_id)
        if state == "failed":
            # 17i: every request refused — closed as the poller closes it,
            # what landed recorded, its answers free for the grader now (they
            # stayed "out", and the dry run counted one more than Start sent)
            db.batch_finish(batch_id, "failed", detail)
            failed(batch_id, detail)
            return
        if state != "done":
            return
        finish(batch_id, be.fetch(batch_id))
        db.batch_finish(batch_id, "done", "")
    except Exception as e:                              # noqa: BLE001 — the poller tries again
        print(f"[frontier grading] {batch_id}: not closed at Start ({e!r}) — the poller will")


def _working(batch_id: str) -> bool:
    """17d: a batch's worker is still sending (its lock is held)"""
    d = llm.batch_dir(batch_id)
    if d is None or not (d / "worker.lock").exists():
        return False
    fd = os.open(d / "worker.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _settle(batch_ids: list[str], timeout: float | None = None) -> bool:
    """17d: the requests in flight of these batches landed (their workers done)"""
    end = time.time() + (SETTLE_S if timeout is None else timeout)
    while time.time() < end:
        if not any(_working(b) for b in batch_ids):
            return True
        time.sleep(0.1)
    return not any(_working(b) for b in batch_ids)


@contextlib.contextmanager
def grades_lock(d: Path):
    """17e: one task's grades.json read, changed and written by one hand at a
    time — choosing a grader while the poller recorded a batch overwrote the
    batch's grades, and the next Start bought them again"""
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d / "grades.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_grades(d: Path, g: dict) -> None:
    tmp = d / (sf.GRADES + ".part")
    tmp.write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / sf.GRADES)


def _permanent(res) -> bool:
    """17e: a refusal that will never change for this answer — too long (HTTP
    400, 413), its words (403, moderation), unprocessable (422): a try, where
    one that may pass another time (unreached, a limit, a rate, a provider
    down, the data policy) isn't. An answer always refused was sent again on
    every Start, and its benchmark never scored. 17f: its status and kind as
    the request kept them, read from the whole error (`res` a Result, or the
    error's words)"""
    error = res if isinstance(res, str) else res.error
    status, kind = (None, "") if isinstance(res, str) else (res.status, res.kind)
    if status is None:
        m = llm._HTTP_STATUS.search(error or "")
        status = int(m.group(1)) if m else None
    if status is None:
        return False
    # 17h: a refusal about the answer itself (too long, flagged) — its try;
    # one about the grader (an id it doesn't know, a region), the key, the
    # provider or the network never is. A kind kept as "refused" before 17h
    # is read again from its words
    if not kind or kind == "refused":
        kind = ai_models.refusal(status, error)[0]
    return kind == "answer"


def _reset_tries(slot: str, pin: dict, hold: dict | None = None) -> None:
    """17d: another grader or prompt asks again what an earlier one gave no
    grade — the tries count again from none, and its scores wait"""
    from . import served
    version, sha = pin.get("version") or pin.get("id"), fg.prompt_sha(slot)
    root = Path(config.OUT_DIR)
    task = fg.GRADERS[slot]["task"]
    for d in sorted(root.glob(f"*/{task}_0shot/{sf.SUB}")) if root.is_dir() else []:
        if (d.parent.parent.name, task) in (hold or {}):
            continue                        # 17i: its replies in flight land first
        with grades_lock(d):
            g = sf.read_grades(d)
            was = json.dumps(g, sort_keys=True)
            _switch(g, version, sha)
            changed = json.dumps(g, sort_keys=True) != was
            if changed:
                _write_grades(d, g)
        if not changed:
            continue
        row = d.parent.parent
        rec = served.get(_model_of(row)[1])
        if rec:
            try:
                sf.score_task(row, task, rec)
            except Exception as e:                      # noqa: BLE001 — said, scored later
                print(f"[frontier grading] {row.name} {task}: scoring failed: {e!r}")


def _regrade(g: dict, version: str, sha: str) -> bool:
    """17f: a first grader that left more than UNGRADED_SHARE of its answers
    without a grade (the case the card advises another grader for): the one
    chosen now grades the whole benchmark again — its cost in the dry run,
    before Start — and its score is final. The first's grades and no-grades
    are kept aside (`aside`), never mixed in. True when it changed `g`"""
    mine = (version, sha)

    def who(x: dict) -> tuple:
        return (x.get("by"), x.get("prompt_sha256"))
    items, ref = g.get("items") or {}, g.get("refused") or {}
    ungraded = sum(1 for x in ref.values()
                   if who(x) != mine and int(x.get("tries") or 0) >= GRADE_TRIES)
    seen = sum(1 for x in items.values() if who(x) != mine) + ungraded
    if not ungraded or ungraded <= sf.UNGRADED_SHARE * seen:
        return False
    _stash(g, mine)                             # 17g: each under its own name
    g["regrade"] = {"by": version, "prompt_sha256": sha, "at": time.time(),
                    "why": f"the first grader left {ungraded:,} of {seen:,} without a grade"}
    return True


def _mixed(g: dict, mine: tuple) -> bool:
    """17h: grades on the row by more than one grader that aren't one grader
    and its top-up (a second that graded only what the first gave no grade,
    UNGRADED_SHARE of them at most) — never final as they are"""
    items = g.get("items") or {}
    who: dict = {}
    for x in items.values():
        who.setdefault(_who(x), []).append(x)
    if len(who) < 2:
        return False
    if len(who) == 2:
        (a, xa), (b, xb) = sorted(who.items(), key=lambda kv: -len(kv[1]))
        top = all(_who(x.get("after") or {}) == a for x in xb)
        if top and len(xb) <= sf.UNGRADED_SHARE * len(items):
            return False
    return True


def _key(who: tuple) -> str:
    """a grader's name in grades.json: its version and its prompt's sha256"""
    return f"{who[0]} · {who[1]}"


def _who(x: dict) -> tuple:
    return (x.get("by"), x.get("prompt_sha256"))


def _migrate(g: dict) -> None:
    """17f's `aside` (a list, which nothing read) into `kept`, each grade
    under its grader's name"""
    for e in g.pop("aside", None) or []:
        for part in ("items", "refused"):
            for k, x in (e.get(part) or {}).items():
                (g.setdefault("kept", {}).setdefault(_key(_who(x)), {})
                 .setdefault(part, {}).setdefault(k, x))


def _stash(g: dict, mine: tuple) -> None:
    """17g: every grade and no-grade not by `mine` into `kept`, under its own
    grader's name — never mixed in, and back when that grader is chosen"""
    for part in ("items", "refused"):
        keep = {}
        for k, x in (g.get(part) or {}).items():
            if _who(x) == mine:
                keep[k] = x
            else:
                g.setdefault("kept", {}).setdefault(_key(_who(x)), {}).setdefault(part, {})[k] = x
        g[part] = keep


def _switch(g: dict, version: str, sha: str) -> tuple[str, int]:
    """17g: the grader chosen now takes over a benchmark's grades, at Start
    (and in the dry run, on a copy) — ('', 0), or what it did and how many of
    its own grades came back:
    - 'match': a regrade asked for, so that the rows compared are graded by
      one grader (`regrade_to`) — every other grader's set aside by name;
    - 'kept': it graded this benchmark before — what it graded comes back,
      the grades here now kept under their own grader's name;
    - 'regrade': 17f's — the one here left more than UNGRADED_SHARE ungraded.
    Otherwise a top-up: the others' no-grades asked again (17d)"""
    mine = (version, sha)
    _migrate(g)
    kept = g.get("kept") or {}
    back = kept.get(_key(mine)) or {}
    active = [x for part in ("items", "refused") for x in (g.get(part) or {}).values()]
    how = ""
    ask = g.get("regrade_to") or {}
    if ask and (ask.get("by"), ask.get("prompt_sha256")) == mine:
        _stash(g, mine)
        g.pop("regrade_to", None)
        g["regrade"] = {"by": version, "prompt_sha256": sha, "at": time.time(),
                        "why": ask.get("why") or "asked so the rows compared share a grader"}
        how = "match"
    elif back and not any(_who(x) == mine for x in active):
        _stash(g, mine)
        how = "kept"
    elif _regrade(g, version, sha):
        how = "regrade"
    elif _mixed(g, mine):
        # 17h: two graders behind the row that aren't a top-up (a stop, then
        # another grader): the one chosen now grades every answer it didn't —
        # the row was "not final", and no grader sent anything
        _stash(g, mine)
        g["regrade"] = {"by": version, "prompt_sha256": sha, "at": time.time(),
                        "why": "its grades came from two graders: the one chosen now grades "
                               "those it didn't"}
        how = "mixed"
    reused = 0
    if back and how in ("kept", "match"):
        g.setdefault("items", {}).update(back.get("items") or {})
        reused = len(back.get("items") or {})
        back.pop("items", None)
    # 17d: another grader's no-grades are asked again by the one chosen now,
    # from none. 17i: kept under their grader's name with their tries, never
    # zeroed — chosen, changed and chosen back, a grader was paid again for
    # an answer it had failed three times; its own come back
    ref = g.setdefault("refused", {})
    moved = 0
    for k, x in list(ref.items()):
        if _who(x) != mine:
            g.setdefault("kept", {}).setdefault(_key(_who(x)), {}).setdefault(
                "refused", {})[k] = x
            del ref[k]
            moved += 1
    items = g.get("items") or {}
    own = (g.get("kept") or {}).get(_key(mine)) or {}
    for k, x in list((own.get("refused") or {}).items()):
        if k not in items and k not in ref:
            ref[k] = x
            del own["refused"][k]
            moved += 1
    if moved and not how:
        # 17i: nothing to send, and still a change Start makes — the no-grades
        # are this grader's now (its own back, with their tries): the dry
        # run's row says what the score becomes
        how = "no-grades"
    for name in [n for n, v in (g.get("kept") or {}).items()
                 if not any(v.get(part) for part in ("items", "refused"))]:
        g["kept"].pop(name)
    return how, reused


def _no_grade_by_other(g: dict, key: str, mine: tuple) -> dict:
    """17i: another grader's no-grade of `key`, kept under its name — what a
    top-up's grade comes after"""
    for name, v in (g.get("kept") or {}).items():
        x = (v.get("refused") or {}).get(key)
        if x and _who(x) != mine and int(x.get("tries") or 0):
            return x
    return {}


def _cancel_unsent(p: dict, why: str) -> list[str]:
    """a batch's requests not yet sent, cancelled — recorded so, never failed.
    17e: the ids it cancelled (some may still be in flight, and land)"""
    d = llm.batch_dir(p["batch_id"])
    if d is None:
        return []
    landed = set(llm.LocalOpenAI._results(d))
    unsent = [r["custom_id"] for r in llm.LocalOpenAI._requests(d)
              if r["custom_id"] not in landed]
    if not unsent:
        return []
    try:
        batch_backend(p["batch_id"]).cancel(p["batch_id"], unsent, why)
    except llm.LLMError:
        return []
    return unsent


def _still_cancelled(batch_id: str, ids: list[str]) -> int:
    """17e: of `ids`, those whose last word is the cancel — a reply that landed
    beats it (service/llm.py)"""
    d = llm.batch_dir(batch_id)
    if d is None or not ids:
        return 0
    res = llm.LocalOpenAI._results(d)
    return sum(1 for c in ids if (res.get(c) or {}).get("cancelled"))


def grader_record(slot: str, pin: dict, by: str = "", prompt_sha: str = "") -> dict:
    """what the scores say graded them: the model as pinned, and the prompt —
    17c: the prompt the batch was sent with, as its record kept it"""
    g = fg.GRADERS[slot]
    return {"slot": slot, "label": g["label"], "model": pin.get("id"),
            "version": pin.get("version") or pin.get("id"), "provider": pin.get("provider_name")
            or pin.get("provider"), "prompt": g["prompt"], "prompt_words": g["prompt_words"],
            "prompt_sha256": prompt_sha or fg.prompt_sha(slot), "owners": g["owners"], "by": by}


def finish(batch_id: str, results: dict) -> int:
    """a batch landed: each answer's grade kept beside the answers, a refusal
    in its words, and the benchmark scored again"""
    from . import served
    meta = _meta(batch_id)
    slot, task, pin = meta["slot"], meta["task"], meta["pin"]
    row = Path(config.OUT_DIR) / meta["row"]
    d = sf.task_dir(row, task)
    items = {it["id"]: it for it in fb.load(task, config.BENCH_ROOT)}
    with grades_lock(d):
        n = _record(d, slot, pin, meta, items, results)
    rec_served = served.get(meta.get("base") or meta.get("model") or "")
    if rec_served:
        try:
            sf.score_task(row, task, rec_served)
        except Exception as e:                      # noqa: BLE001 — the grades are kept
            print(f"[frontier grading] {meta['row']} {task}: scoring failed: {e!r}")
    return n


def _record(d: Path, slot: str, pin: dict, meta: dict, items: dict, results: dict) -> int:
    """a batch's replies into grades.json, under its lock"""
    g = sf.read_grades(d)
    n = _apply(g, d, slot, pin, meta, items, results)
    _write_grades(d, g)
    return n


def _apply(g: dict, d: Path, slot: str, pin: dict, meta: dict, items: dict,
           results: dict) -> int:
    """17h: a batch's replies into the grades `g` — the dry run applies a
    moved batch's landed replies to a copy, as Start then records them"""
    g.setdefault("items", {})
    g.setdefault("refused", {})
    rec = grader_record(slot, pin, meta.get("by", ""), meta.get("prompt_sha256") or "")
    graders = [x for x in g.get("graders") or [] if x.get("version") != rec["version"]
               or x.get("prompt_sha256") != rec["prompt_sha256"]]
    g["graders"] = [*graders, rec]
    g["grader"] = rec
    sent = meta.get("answers") or {}
    now = {sf.gkey(q, e): sf.answer_sha(a.get("answer") or "")
           for (q, e), a in sf.read_answers(d / sf.ANSWERS).items()}
    cap = meta.get("max_tokens")
    n = 0
    errors = []
    for cid, res in results.items():
        key = cid.split(":", 1)[1]
        qid = key.rsplit("#", 1)[0]
        # 17b: one answer at a time — a reply this can't read fails that
        # answer, never the batch and the replies beside it
        try:
            if key not in sent or sent[key] != now.get(key):
                continue                # the answer changed while out: graded again
            if res.error and res.error.startswith("cancelled"):
                continue                # 17c: never sent: it waits for the next Start
            why = llm.plain_error(res.error) if res.error else ""
            got = None if why else fg.read(slot, res.text, items.get(qid) or {})
            if got is not None and got.get("ok") is None:
                # 17j: a reply that reached its cap is read all the same — a
                # whole object that ended exactly at the cap was a paid try;
                # one the cap cut off reads as no grade, and says the cap
                why = (f"the reply was cut at its cap of {cap or 'its'} tokens"
                       if res.finish == "length" else got["unread"])
            if why and res.error:
                errors.append((key, why, res))          # 17f: counted once the batch is read
                continue
            if why:
                # 17c: counted, for this answer — at GRADE_TRIES it is ungraded.
                # 17d: only a reply that came and isn't a grade is a try; and
                # another grader or prompt starts the count again
                g["refused"][key] = _refusal(g["refused"].get(key), why, rec, sent[key],
                                             counts=True, kind="unread")
                continue
            was = g["refused"].get(key) or _no_grade_by_other(
                g, key, (rec["version"], rec["prompt_sha256"]))
            # 17e: a grade where another grader or prompt gave none says so —
            # a second grader's top-up (service/frontier.py graders_of)
            after = ({"after": {"by": was.get("by"), "prompt_sha256": was.get("prompt_sha256")}}
                     if was and (was.get("by"), was.get("prompt_sha256"))
                     != (rec["version"], rec["prompt_sha256"]) else {})
            g["items"][key] = {**got, "by": rec["version"], "prompt_sha256": rec["prompt_sha256"],
                               "answer_sha256": sent[key], "at": time.time(), **after}
            g["refused"].pop(key, None)
            n += 1
        except Exception as e:                      # noqa: BLE001 — that answer only
            g["refused"][key] = _refusal(g["refused"].get(key),
                                         f"its reply couldn't be read: {e!r}"[:300], rec,
                                         sent.get(key), counts=True, kind="unread")
    # 17e: a refusal that never changes is a try. 17h: only one about the
    # answer itself — a grader's refusal of every answer (a model id the
    # provider doesn't know) is no answer's, and stops the batch (llm)
    for key, why, res in errors:
        g["refused"][key] = _refusal(g["refused"].get(key), why, rec, sent[key],
                                     counts=_permanent(res), kind="error")
    return n


def _refusal(was: dict | None, why: str, rec: dict, answer_sha: str | None, counts: bool,
             kind: str) -> dict:
    """17d: an answer's no-grade, its tries counted only for a reply that came,
    and only while the grader, its prompt and the answer are the same"""
    was = was or {}
    same = (was.get("answer_sha256"), was.get("by"), was.get("prompt_sha256")) == (
        answer_sha, rec["version"], rec["prompt_sha256"])
    tries = (int(was.get("tries") or 0) if same else 0) + (1 if counts else 0)
    return {"words": why, "by": rec["version"], "prompt_sha256": rec["prompt_sha256"],
            "at": time.time(), "kind": kind, "tries": tries, "answer_sha256": answer_sha}


def failed(batch_id: str, why: str) -> None:
    """a batch failed: the rest asked again by the next Start — 17c: and what
    it landed read first, each reply paid for kept as its grade"""
    print(f"[frontier grading] {batch_id} failed: {why}")
    d = llm.batch_dir(batch_id)
    if d is None:
        return
    landed = {c: llm.Result(text=r.get("text") or "", error=r.get("error") or "",
                            finish=r.get("finish_reason") or "")
              for c, r in llm.LocalOpenAI._results(d).items()}
    if landed:
        try:
            finish(batch_id, landed)
        except Exception as e:                          # noqa: BLE001 — said, kept on disk
            print(f"[frontier grading] {batch_id}: its {len(landed)} replies couldn't be "
                  f"recorded: {e!r}")


def retry_finish(batch_id: str, e: Exception) -> bool:
    """17c: a batch whose replies couldn't be recorded (finish() failed outside
    its answers: no question file, a disk error) is tried again by the next
    poll, its replies paid for kept. 17d: for as long as it takes — after 30
    polls it was failed, and the next Start paid for every reply again; the
    card says why it waits. False only when its record is gone"""
    p = gdir() / "batches" / f"{batch_id}.json"
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    meta["finish_tries"] = int(meta.get("finish_tries") or 0) + 1
    meta["finish_error"] = repr(e)[:300]
    p.write_text(json.dumps(meta), encoding="utf-8")
    return True


def _progress(p: dict) -> str:
    tl = llm.tally(p["batch_id"])
    if not tl:
        return p.get("progress") or ""
    done = tl["answered"] + tl["failed"] + tl["cancelled"]
    line = f"{done:,} of {tl['sent']:,} · ${db.spend_of_batch(p['batch_id']):,.2f} so far"
    return f"{line} · {tl['halted']}" if tl.get("halted") else line


def last_failed() -> list[dict]:
    """17b: each grader's last batch, when it failed — whole, or some of its
    requests — in one line (llm.batch_line), as the Mobile-MMLU key's card
    says it"""
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
        out[slot] = {"slot": slot, "batch_id": r["batch_id"], "status": r.get("status"),
                     "failed": bool((tl or {}).get("failed") or err),
                     "line": f"{fg.GRADERS[slot]['label']}’s last batch: "
                             f"{llm.batch_line(tl, err or '')}"}
    return [x for x in out.values() if x["failed"]]


def _unsent(p: dict) -> int:
    """a batch's answers not yet answered, failed or cancelled"""
    tl = llm.tally(p["batch_id"])
    if not tl:
        return len(p.get("keys") or [])
    return max(0, tl["sent"] - tl["answered"] - tl.get("failed", 0) - tl.get("cancelled", 0))


CARRY = "Carry on sends the rest."


def waits() -> list[dict]:
    """17b: why the batches out send nothing now, each in words. 17c: each
    with what Carry on does about it — {why, carry}: Stop and a run of
    refusals, it lifts; a grader OpenRouter has moved, Start sends its unsent
    to the one pinned now; the month's limit, a missing key and replies that
    can't be recorded yet, it can't"""
    out: list[dict] = []
    p = pending()
    if not p:
        return out
    st = stopped()
    if st:
        out.append({"why": f"Stopped by {st.get('by') or 'someone'}: "
                           f"{sum(_unsent(x) for x in p):,} answers wait to be sent",
                    "carry": CARRY})
    limit = ai_models.over_limit()
    if limit:
        out.append({"why": limit, "carry": "Carry on waits until the limit is raised (AI "
                                           "models ▸ the month's limit) or the month turns."})
    if not ai_models.has_key():
        out.append({"why": "OpenRouter has no key on this server (OPENROUTER_API_KEY): the "
                           "batches out wait for it, their replies kept", "carry": ""})
    for x in p:
        label = fg.GRADERS[x["slot"]]["label"]
        h = (llm.tally(x["batch_id"]) or {}).get("halted")
        if h:
            out.append({"why": f"{label}: {h}", "carry": CARRY})
        drift = ai_models.drifted(x.get("pin") or {})
        if drift:
            out.append({"why": f"{label}: {drift}",
                        "carry": f"Start sends its {_unsent(x):,} unsent answers to the "
                                 "grader pinned now."})
        if x.get("finish_error"):
            out.append({"why": f"{label}: its replies couldn't be recorded yet "
                               f"({x['finish_error']}) — tried {x.get('finish_tries')} times; "
                               "kept on disk, and tried again at every poll", "carry": ""})
    return out


def held() -> dict:
    """17c: what the batches out still have to send, and about what it costs
    — what Carry on sends, beside the dry run's new answers"""
    n, usd, known = 0, 0.0, True
    for x in pending():
        if _moving(x):
            continue                        # 17h: the dry run prices it, at the grader now
        d = llm.batch_dir(x["batch_id"])
        if d is None:
            continue
        landed = set(llm.LocalOpenAI._results(d))
        unsent = [r for r in llm.LocalOpenAI._requests(d) if r["custom_id"] not in landed]
        if not unsent:
            continue
        pin = x.get("pin") or {}
        tin = sum(len(r.get("user") or "") for r in unsent) // CHARS_A_TOKEN
        tout = len(unsent) * fg.ask(x["slot"], reasons(pin))["out_tokens"]
        pi, po = _price(pin)
        n += len(unsent)
        if pi is None or po is None:
            known = False
        else:
            usd += (tin * pi + tout * po) / 1e6
    return {"answers": n, "usd": round(usd, 4), "usd_known": known}


def refusals() -> list[dict]:
    """each benchmark's answers its grader gave no grade, with the last words —
    17c: and how many of them are ungraded for good, after GRADE_TRIES"""
    out = []
    root = Path(config.OUT_DIR)
    for slot, g in fg.GRADERS.items():
        for d in sorted(root.glob(f"*/{g['task']}_0shot/{sf.SUB}")) if root.is_dir() else []:
            gr = sf.read_grades(d)
            ref = gr.get("refused") or {}
            if ref:
                last = max(ref.values(), key=lambda x: x.get("at") or 0)
                ungraded = sum(1 for x in ref.values() if int(x.get("tries") or 0) >= GRADE_TRIES)
                seen = len(gr.get("items") or {}) + ungraded
                out.append({"slot": slot, "task": g["task"], "row": d.parent.parent.name,
                            "n": len(ref), "words": last.get("words") or "",
                            "ungraded": ungraded,
                            # 17d: too many for a score — the grader isn't answering in form
                            "form": bool(ungraded and ungraded > sf.UNGRADED_SHARE * seen)})
    return out


def _score_grader(d: Path) -> dict:
    """the grader a benchmark's written score names, {} when none"""
    for f in sorted(d.glob("results_*.json")):
        try:
            return (json.loads(f.read_text(encoding="utf-8")).get("frontier") or {}) \
                .get("grader") or {}
        except (OSError, ValueError):
            continue
    return {}


def _cost(slot: str, g: dict, task: str, todo: list[dict]) -> float | None:
    """what sending `todo` to grader `g` costs, as the dry run prices it"""
    pin, pout = _price(g)
    if pin is None or pout is None:
        return None
    items = {it["id"]: it for it in fb.load(task, config.BENCH_ROOT)}
    a = fg.ask(slot, reasons(g), structured(g))
    tin = sum(len(fg.render(slot, items[x["id"]], x)) for x in todo
              if x["id"] in items) // CHARS_A_TOKEN
    return round((tin * pin + len(todo) * a["out_tokens"] * pout) / 1e6, 4)


def mismatches() -> list[dict]:
    """17g: a benchmark whose rows are scored by different graders, each
    final on its own row — said on every one of them, and each row the grader
    chosen now didn't grade offered its regrade by that grader, with its
    price: [{slot, task, label, grader, rows: [{row, model, version, offer,
    asked}]}]"""
    from . import served
    out = []
    root = Path(config.OUT_DIR)
    fb.set_root(config.BENCH_ROOT)
    for slot, gdef in fg.GRADERS.items():
        t = gdef["task"]
        rows = []
        for d in sorted(root.glob(f"*/{t}_0shot/{sf.SUB}")) if root.is_dir() else []:
            gr = _score_grader(d)
            if gr.get("version"):
                row = d.parent.parent
                mid, base = _model_of(row)
                name = (served.get(base) or {}).get("name") or mid
                rows.append({"row": row.name, "model": mid, "d": d,
                             "name": name + (" · thinking" if mid.endswith(" · thinking")
                                             and not name.endswith(" · thinking") else ""),
                             "version": gr["version"], "prompt_sha256": gr.get("prompt_sha256"),
                             "asked": bool(sf.read_grades(d).get("regrade_to"))})
        if len({(r["version"], r["prompt_sha256"]) for r in rows}) < 2:
            continue
        now = chosen(slot)
        mine = (now.get("version") or now.get("id"), fg.prompt_sha(slot)) if now else None
        for r in rows:
            d = r.pop("d")
            if mine and (r["version"], r["prompt_sha256"]) != mine:
                seen = copy.deepcopy(sf.read_grades(d))
                seen["regrade_to"] = {"by": mine[0], "prompt_sha256": mine[1]}
                _switch(seen, *mine)
                todo = sf.to_grade(d.parent.parent, t, grades=seen)
                r["offer"] = {"answers": len(todo), "usd": _cost(slot, now, t, todo),
                              "to": now.get("version") or now.get("id")}
        out.append({"slot": slot, "task": t, "label": fb.BENCH[t]["label"],
                    "grader": (now or {}).get("version") or (now or {}).get("id"), "rows": rows})
    return out


def regrade_row(row_name: str, task: str, by: str, undo: bool = False) -> dict:
    """17g: a row graded again whole by the grader chosen now, at the next
    Start — so that the rows compared share one grader. Nothing is sent now;
    the dry run shows it, with its price"""
    slot = next((s for s, g in fg.GRADERS.items() if g["task"] == task), None)
    if slot is None:
        raise ValueError(f"{task} has no grader")
    if not row_name or "/" in row_name or row_name.startswith("."):
        raise ValueError(f"{row_name!r} isn't a row")
    d = sf.task_dir(Path(config.OUT_DIR) / row_name, task)
    if not d.is_dir():
        raise ValueError(f"{row_name} has no {fb.BENCH[task]['label']} answers")
    now = chosen(slot)
    if not now and not undo:
        raise ValueError(f"no grader is chosen for {fb.BENCH[task]['label']}")
    with grades_lock(d):
        g = sf.read_grades(d)
        if undo:
            g.pop("regrade_to", None)
        else:
            name = now.get("version") or now.get("id")
            g["regrade_to"] = {"by": name, "prompt_sha256": fg.prompt_sha(slot), "who": by,
                               "at": time.time(),
                               "why": f"asked by {by}, so that the rows compared share {name}"}
        _write_grades(d, g)
    return {"asked": not undo}


def status() -> dict:
    """the AI models page's card: the graders, the dry run, the run"""
    est = estimate()
    return {"graders": [{"slot": s, "label": g["label"], "does": g["does"], "why": g["why"],
                         "task": g["task"], "benchmark": fb.BENCH[g["task"]]["label"],
                         "suggested": g["suggested"], "owners": g["owners"],
                         "prompt": g["prompt_words"], "prompt_sha256": fg.prompt_sha(s),
                         "chosen": chosen(s), "now": grader(s),
                         "ask": fg.ask(s, reasons(grader(s)), structured(grader(s))),
                         # 17j: whether HLE's judge answers in CAIS's JSON schema
                         # (structured outputs) — None when OpenRouter doesn't say
                         **({"json_only": structured(grader(s))} if s == "hle" else {}),
                         "reasoning_words": g["reasoning_words"]}
                        for s, g in fg.GRADERS.items()],
            "estimate": est,
            "key_warning": ai_models.more_than_key(est.get("usd")),
            "running": [{"slot": p["slot"], "task": p["task"], "model": p["model"],
                         "n": len(p.get("keys") or []), "progress": _progress(p)}
                        for p in pending()],
            "refused": refusals(),
            "stopped": stopped(),
            "waits": waits(),
            "held": held(),
            "tries": GRADE_TRIES,
            "last": last_failed(),
            "probe_words": PROBE_WORDS,
            "has_key": ai_models.has_key(),
            # 17g: rows of one benchmark scored by different graders
            "mismatches": mismatches()}
