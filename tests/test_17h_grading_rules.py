"""17h, part 1, and 17i's: paid grading tested by its rules, not a bug at a
time (docs/prompts/phase-17h-sixth-review-grading-export-fetch.md, and
phase-17i-seventh-review-hle-reader-export-fetch.md).

Random sequences of what masein and the graders can do, on two benchmarks
(SimpleQA Verified and Humanity's Last Exam, two rows each, 24 answers a row:
enough for a top-up) — choose a grader for either, a grader that grades,
answers out of form now and then or for good, refuses every request ("not a
valid model ID"), refuses an answer for its length, hits the key's spend cap,
fails as a provider, is down, times out — Start, Stop mid-batch, Stop alone,
Stop twice, Carry on, the service restarting mid-batch, the month's limit hit
mid-batch and lifted, replies slow to land while another grader is chosen,
and the regrade the card offers. After every step:

1. no answer is paid for twice by one grader, counted across the whole
   sequence whoever was chosen between: a grade bought once, a reply out of
   form at most GRADE_TRIES times;
2. read from the grades on disk: a final score is one grader's, or one and
   its top-up, and names that grader; rows of a benchmark final by different
   graders are on the card;
3. what the dry run counts and prices is what Start sends; what the batches
   out hold, counted and priced, is what Carry on sends;
4. a score that isn't final has a next step on the page;
5. a grade bought is kept — on disk under its grader, or landed in a batch
   still out — never thrown away;
6. a Start that costs more than the month has left is refused, in words,
   before anything is sent — unless started knowing it stops part-way.

Each sequence ends with every grader answering in form and the page's next
steps taken: every row must end final, and by one grader. Nothing calls
OpenRouter: the grader is a stand-in, its replies decided by the seed, the
grader, the row, the answer and its attempt. RULES_SEQUENCES sets how many
seeds from 0 (30 in CI), RULES_EXTRA how many more from RULES_SEED (10; the
seed is printed in the header, and chosen at random when not set)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter

import pytest

import frontier as fb
import frontier_graders as fgs
import test_17_grading as t17
from service import ai_models, config, db, llm, llm_poller
from service import frontier as sf
from service import frontier_grade as fgr
from test_17_grading import ROW, svc, write  # noqa: F401 — svc is the fixture

SQA, HLE = "simpleqa_epoch", "hle_text_cais"
SLOT = {SQA: "simpleqa", HLE: "hle"}
TASKS = (SQA, HLE)
ROW2, SERVED2 = "served__orig-box", "served/orig-box"
ROWS = (ROW, ROW2)
GPT, GEMINI, O3, FRESH = ("openai/gpt-4.1", "google/gemini-2.5-flash", "openai/o3-mini",
                          "x-ai/grok-fresh")
CHOOSE = (GPT, GEMINI, O3)
MODES = ("good", "flaky", "stubborn", "invalid", "mixed", "toolong", "toolong-all", "cap",
         "provider", "down", "timeout")
ANSWERS = 24                    # a row's answers to grade: one no-grade is a top-up, two a regrade
N = int(os.environ.get("RULES_SEQUENCES") or 30)
EXTRA = int(os.environ.get("RULES_EXTRA") or 10)
BASE = int(os.environ.get("RULES_SEED") or 1)       # set at random by conftest, and printed
STEPS = 10
HLE_YES = json.dumps({"extracted_final_answer": "Key", "reasoning": "it matches the key",
                      "correct": "yes", "confidence": 90})
OUT_OF_FORM = "I would rather not say"


class Restart(Exception):
    """the service going away mid-batch: what was in flight is never written"""


class World:
    """the stand-in grader: what it says to each request, what was paid"""

    def __init__(self, seed: int):
        self.seed = seed
        self.mode: dict[str, str] = {}
        self.attempt: Counter = Counter()
        # 17i: (grader, task, row, key), across the whole sequence — never
        # reset when another grader is chosen
        self.grades: Counter = Counter()
        self.paid: Counter = Counter()
        self.stop_at: int | None = None
        self.restart_at: int | None = None
        self.dead = False
        self.calls = 0
        self.log: list[str] = []
        self.long: set = set()                # graders some answers are too long for, for good
        self.spent = 0.0
        self.limit = 1000.0
        self.gate = threading.Event()
        self.gate.set()
        self.slow = 0                         # the next replies that hang until they land
        self.hanging: Counter = Counter()     # batch -> replies hanging now

    def _h(self, *parts) -> int:
        return int(hashlib.sha256(":".join(map(str, (self.seed, *parts))).encode())
                   .hexdigest()[:8], 16)

    def reply(self, model: str, slot: str, row: str, key: str) -> tuple[str, str, int | None]:
        n = self.attempt[(model, slot, row, key)]
        self.attempt[(model, slot, row, key)] += 1
        m = self.mode.get(model, "good")
        h = self._h(model, row, key, n) % 100
        if m == "flaky" and h < 40:
            return OUT_OF_FORM, "", None
        if m == "stubborn" and self._h(model, slot, row, key) % 100 < 6:
            return OUT_OF_FORM, "", None                # this answer, never in form
        if m == "invalid":
            return "", (f'POST u: HTTP 400: {{"error": {{"message": "{model} is not a valid '
                        'model ID"}}'), 400
        if m == "mixed" and h < 60:
            return "", 'POST u: HTTP 400: {"error": {"message": "Provider returned error"}}', 400
        if m == "toolong-all" or (m == "toolong" or model in self.long) \
                and self._h(model, row, key) % 100 < 70:
            self.long.add(model)                        # an answer's length never changes
            return "", ('POST u: HTTP 400: {"error": {"message": "This prompt is too long for '
                        'the model\'s context length"}}'), 400
        if m == "cap":
            return "", 'POST u: HTTP 403: {"error": {"message": "Key spend cap reached"}}', 403
        if m == "provider":
            return "", 'POST u: HTTP 400: {"error": {"message": "Provider returned error"}}', 400
        if m == "down":
            return "", "POST u: HTTP 503: the provider is down", 503
        if m == "timeout":
            return "", "POST u: no response within 120 s", None
        return ("A" if slot == "simpleqa" else HLE_YES), "", None


def items(task: str) -> list[dict]:
    if task == SQA:
        return [{"id": str(k), "question": f"Invented fact {k}?", "answer": f"Answer {k}",
                 "subject": "x"} for k in range(ANSWERS + 1)]
    return [{"id": f"h{k}", "question": f"Invented exam question {k}?", "answer": "Key",
             "subject": "x"} for k in range(ANSWERS + 1)]


@pytest.fixture
def world(svc, monkeypatch):  # noqa: F811
    """two rows of SimpleQA and two of Humanity's Last Exam (24 answers each to
    grade, one ran out), MATH taken away; four graders on OpenRouter's list;
    the month's spend kept by the stand-in; the stand-in grader"""
    monkeypatch.setitem(t17.GRADERS, FRESH, ("x-ai/grok-fresh-20261001", 0.5, 1.5))
    p = ai_models._cache_path()
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": m, "name": m, "version": v, "price_in": pi, "price_out": po}
        for m, (v, pi, po) in t17.GRADERS.items()]}))
    import shutil
    shutil.rmtree(sf.task_dir(config.OUT_DIR / ROW, "math_l5_epoch"))
    data = {SQA: items(SQA), HLE: items(HLE), "math_l5_epoch": t17.math_items()}
    monkeypatch.setattr(fb, "_fetch", lambda task: data[task])
    db.served_put({"id": SERVED2, "name": "orig box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "o.gguf"}, "by": "masein", "at": 0})
    row = config.OUT_DIR / ROW2
    row.mkdir(parents=True)
    (row / "model_meta.json").write_text(json.dumps({"model": SERVED2}))
    for r in ROWS:
        for task in TASKS:
            write(config.OUT_DIR / r, task, [
                (it["id"], f"I think {it['answer']}.", "stop" if k < ANSWERS else "length")
                for k, it in enumerate(items(task))])
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 3)
    monkeypatch.setattr(fgr.GraderChat, "FIRST_REFUSALS", 3)
    monkeypatch.setattr(fgr, "SETTLE_S", 10.0)
    w = World(0)
    # the month's spend, as the stand-in paid it
    monkeypatch.setattr(db, "spend_this_month", lambda: w.spent)
    monkeypatch.setattr(ai_models, "limit", lambda: w.limit)

    def complete(self, req):
        if w.dead:
            raise Restart()
        meta = fgr._meta(req["batch_id"])
        key = req["custom_id"].split(":", 1)[1]
        slot, task = meta["slot"], meta["task"]
        who = (meta["pin"]["version"], task, meta["row"], key)
        w.calls += 1
        if w.restart_at is not None and w.calls >= w.restart_at:
            w.restart_at, w.dead = None, True
            raise Restart()
        text, error, status = w.reply(self.model, slot, meta["row"], key)
        if w.stop_at is not None and w.calls == w.stop_at:
            fgr.stop("masein")                          # Stop, pressed mid-batch
        if text:
            if w.slow > 0:                              # 17i: a reply slow to land
                w.slow -= 1
                w.hanging[req["batch_id"]] += 1
                try:
                    w.gate.wait(120)
                finally:
                    w.hanging[req["batch_id"]] -= 1
            time.sleep(0.01)                            # a reply takes a moment; a refusal none
            w.paid[who] += 1
            pi, po = t17.GRADERS[self.model][1:]
            w.spent += (len(req["user"]) // fgr.CHARS_A_TOKEN * pi
                        + fgs.GRADERS[slot]["out_tokens"] * po) / 1e6
            if fgs.read(slot, text, {"id": key, "answer": "Key"})["ok"] is not None:
                w.grades[who] += 1
            return {"custom_id": req["custom_id"], "text": text, "error": "", "attempts": 1,
                    "finish_reason": "stop"}
        return {"custom_id": req["custom_id"], "text": "", "error": error, "status": status,
                "kind": ai_models.refusal(status, error)[0], "attempts": 1,
                "at": time.time()}
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    yield w
    w.gate.set()                                        # nothing left hanging
    w.dead = False


# ---------------------------------------------------------------------------
# the page, as status() gives it, and the rows as they are
# ---------------------------------------------------------------------------

def _progress() -> tuple:
    """what the batches out have landed, and which are out"""
    out = []
    for p in fgr.pending():
        d = llm.batch_dir(p["batch_id"])
        out.append((p["batch_id"], len(llm.LocalOpenAI._results(d)) if d else -1))
    return tuple(sorted(out))


def settle(w: World, timeout: float = 90.0) -> None:
    """nothing more lands and nothing more finishes — told by what landed,
    never by a worker's lock alone: each tick starts a worker for a stopped
    batch, which ends at once. A reply hanging (slow to land) is left
    hanging; a service gone away comes back once its workers ended"""
    end, quiet, was = time.time() + timeout, 0, None
    while time.time() < end:
        llm_poller.tick()
        time.sleep(0.05)
        now = _progress()
        quiet = quiet + 1 if now == was else 0
        was = now
        if quiet >= 4 and not any(fgr._working(b) and not w.hanging[b] for b, _ in now):
            if w.dead:
                w.dead = False                  # the service is back: the poller takes up
                w.log.append("  (the service starts again)")
                quiet = 0
                continue
            return
    pytest.fail(f"seed {w.seed}: the grading batches never settled\n" + "\n".join(w.log))


def score(row: str, task: str) -> dict | None:
    fs = sorted(sf.task_dir(config.OUT_DIR / row, task).glob("results_*.json"))
    return json.loads(fs[-1].read_text())["frontier"] if fs else None


def final(f: dict | None) -> bool:
    return bool(f) and f.get("final") is not False


def main_grader(f: dict) -> str:
    return (f.get("grader") or {}).get("version") or ""


def on_disk(row: str, task: str) -> tuple[tuple | None, Counter]:
    """17i, rule 2 from the grades themselves: (the row's one grader — or the
    first, when the second only topped up its no-grades, UNGRADED_SHARE of
    them at most — else None; every grader and how many it graded)"""
    g = sf.read_grades(sf.task_dir(config.OUT_DIR / row, task))
    its = g.get("items") or {}
    by = Counter((x.get("by"), x.get("prompt_sha256")) for x in its.values())
    if len(by) == 1:
        return next(iter(by)), by
    if len(by) == 2:
        (main, n_main), (top, n_top) = by.most_common()
        after = {((x.get("after") or {}).get("by"), (x.get("after") or {}).get("prompt_sha256"))
                 for x in its.values() if (x.get("by"), x.get("prompt_sha256")) == top}
        if after == {main} and n_top <= sf.UNGRADED_SHARE * (n_main + n_top):
            return main, by
    return None, by


def offers(page: dict, row: str, task: str) -> list[str]:
    """the next steps the page gives a row whose score isn't final"""
    model = json.loads((config.OUT_DIR / row / "model_meta.json").read_text())["model"]
    slot = SLOT[task]
    out = []
    est = page["estimate"]
    if any(r["model"] == model and r["slot"] == slot and (r["answers"] or r.get("switch"))
           for r in est.get("rows") or []):
        out.append("Start")
    if (page.get("held") or {}).get("answers") or page.get("waits"):
        out.append("Carry on")
    if any(h["row"] == row and h["task"] == task for h in est.get("held") or []):
        out.append("its replies on their way")
    # its batch running, replies on their way (the card's progress line): what
    # waits behind them shows once they land
    if any(r["model"] == model and r["task"] == task for r in page.get("running") or []) and \
            any(p["row"] == row and p["task"] == task and fgr._working(p["batch_id"])
                for p in fgr.pending()):
        out.append("running")
    if any(r["row"] == row and r["slot"] == slot and r["ungraded"]
           for r in page.get("refused") or []):
        out.append("choose another grader")
    if any(m["task"] == task and x["row"] == row and x.get("offer")
           for m in page.get("mismatches") or [] for x in m["rows"]):
        out.append("regrade offered")
    return out


def _kept(w: World, who: tuple) -> bool:
    """rule 5: a grade bought is on disk under its grader, or landed in a
    batch still out"""
    ver, task, row, key = who
    g = sf.read_grades(sf.task_dir(config.OUT_DIR / row, task))
    if ((g.get("items") or {}).get(key) or {}).get("by") == ver:
        return True
    if any(name.startswith(ver + " · ") and key in (v.get("items") or {})
           for name, v in (g.get("kept") or {}).items()):
        return True
    for p in fgr.pending():
        if (p["pin"].get("version"), p["task"], p["row"]) == (ver, task, row):
            r = fgr._landed(p).get("frgr:" + key)
            if r is not None and r.text:
                return True
    return False


def check(w: World, step: str) -> None:
    """the rules, after every step"""
    where = f"seed {w.seed}, after {step}\n" + "\n".join(w.log)
    # 1. never paid twice by one grader, whoever was chosen between
    twice = {k: n for k, n in w.grades.items() if n > 1}
    assert not twice, f"a grade bought twice: {twice}\n{where}"
    over = {k: n for k, n in w.paid.items() if n > sf.GRADE_TRIES}
    assert not over, f"an answer paid for more than {sf.GRADE_TRIES} times: {over}\n{where}"
    # 5. a grade bought is kept
    lost = [k for k in w.grades if not _kept(w, k)]
    assert not lost, f"grades bought and thrown away: {lost}\n{where}"
    page = fgr.status()
    for task in TASKS:
        rows = {r: score(r, task) for r in ROWS}
        # 2. read from the grades on disk: a final score is one grader's
        mains = {}
        for r, f in rows.items():
            if final(f):
                main, by = on_disk(r, task)
                assert main, f"{r} {task}: final, graded on disk by {dict(by)}\n{where}"
                assert main_grader(f) == main[0], \
                    f"{r} {task}: names {main_grader(f)}, graded by {main[0]}\n{where}"
                mains[r] = main
        if len(set(mains.values())) > 1:
            assert any(m["task"] == task for m in page.get("mismatches") or []), \
                f"{task}: final by {mains}, and the card says nothing\n{where}"
        # 4. not final: a next step on the page
        for r, f in rows.items():
            if not final(f):
                assert offers(page, r, task), \
                    f"{r} {task}: not final ({f}), and no next step\n{where}"


def _unsent_items(p: dict) -> list[dict]:
    """a batch's answers not yet landed, as the dry run reads answers"""
    d = llm.batch_dir(p["batch_id"])
    landed = set(llm.LocalOpenAI._results(d)) if d else set()
    runs = sf.read_answers(sf.task_dir(config.OUT_DIR / p["row"], p["task"]) / sf.ANSWERS)
    out = []
    for k in p.get("keys") or []:
        if "frgr:" + k in landed:
            continue
        q, e = k.rsplit("#", 1)
        a = runs.get((q, int(e))) or {}
        out.append({"id": q, "epoch": int(e), "answer": a.get("answer") or "", "read": ""})
    return out


def start(w: World, stop: int | None = None, restart: int | None = None,
          slow: int | None = None, partial: bool = False) -> str:
    """the dry run, then Start — 3. what it said is what goes; 6. never past
    what the month has left unless asked"""
    est = fgr.estimate()
    held = fgr.held()
    chosen = {s: (fgr.chosen(s) or {}).get("version") for s in SLOT.values()}
    sent: list[tuple[dict, list, dict]] = []
    real = fgr._submit

    def submit(item, pin, by):
        sent.append((pin, item["items"], item))
        return real(item, pin, by)
    resumed = [p for p in fgr.pending() if not fgr._moving(p)]
    where = f"seed {w.seed}\n" + "\n".join(w.log)
    # 3. what the batches out hold, counted and — 17i — priced, is what Carry on sends
    assert sum(fgr._unsent(p) for p in resumed) == held["answers"], \
        f"held said {held['answers']}, Carry on sends " \
        f"{sum(fgr._unsent(p) for p in resumed)}\n{where}"
    priced = sum(fgr._cost(p["slot"], p["pin"], p["task"], _unsent_items(p)) or 0
                 for p in resumed)
    assert abs(priced - held["usd"]) < 1e-4 * (len(resumed) + 1), \
        f"held priced ${held['usd']}, Carry on sends ${priced:.4f}\n{where}"
    w.stop_at = None if stop is None else w.calls + stop
    w.restart_at = None if restart is None else w.calls + restart
    if slow:
        w.gate.clear()
        w.slow = slow
    # 6. worked out here, never taken from the dry run's own word for it
    over = w.spent >= w.limit
    short = est["usd_known"] and held["usd_known"] and \
        est["usd"] + held["usd"] > max(0.0, round(w.limit - w.spent, 2))
    fgr._submit = submit
    try:
        fgr.start("masein", **({"partial": True} if partial else {}))
    except ValueError as e:
        # 6. refused before anything is sent: the month's limit, said
        assert not sent, f"Start refused ({e}) after sending {len(sent)} batches\n{where}"
        assert over or (short and not partial), f"Start refused: {e}\n{where}"
        said = est.get("over_limit") or est.get("short")
        assert said and str(e) == said, f"refused with {e!r}, the dry run said {said!r}\n{where}"
        settle(w)
        return "refused: " + ("at the limit" if over else "short")
    finally:
        fgr._submit = real
    assert not over, f"Start went ahead at the month's limit\n{where}"
    assert not short or partial, \
        f"Start costs ${est['usd'] + held['usd']:.4f}, more than the " \
        f"${w.limit - w.spent:.2f} the month has left, and went ahead unasked\n{where}"
    n = sum(len(its) for _, its, _ in sent)
    assert n == est["answers"], f"the dry run said {est['answers']}, Start sent {n}\n{where}"
    for pin, _, item in sent:
        if chosen[item["slot"]]:
            assert pin.get("version") == chosen[item["slot"]], \
                f"priced at {chosen[item['slot']]}, sent to {pin.get('version')}\n{where}"
    usd = sum(fgr._cost(item["slot"], pin, item["task"], its) or 0 for pin, its, item in sent)
    assert abs(usd - est["usd"]) < 1e-4 * (len(sent) + 1), \
        f"priced ${est['usd']}, sent ${usd:.4f}\n{where}"
    settle(w)
    return "sent"


# ---------------------------------------------------------------------------
# the sequences
# ---------------------------------------------------------------------------

def act(w: World, rnd: int) -> str:
    h = w._h("act", rnd)
    pick = h % 100
    slot = ("simpleqa", "hle")[(h >> 20) % 2]
    if pick < 18:
        g = CHOOSE[(h >> 8) % len(CHOOSE)]
        return do(w, f"choose {slot} {g}")
    if pick < 32:
        g = CHOOSE[(h >> 8) % len(CHOOSE)]
        return do(w, f"{g} turns {MODES[(h >> 16) % len(MODES)]}")
    if pick < 72:
        k = (h >> 8) % 5 + 1
        how = (h >> 12) % 8
        what = ("Start" if how < 3 else f"Start, Stop after {k}" if how < 5
                else f"Start, restart after {k}" if how == 5 else f"Start, {k} slow")
        if fgr.estimate().get("short") and (h >> 24) % 2:
            what = "Start anyway"
        return do(w, what)
    if pick < 77:
        return do(w, "Stop")
    if pick < 80:
        return do(w, "Stop twice")
    if pick < 85:
        return do(w, "the month's limit, nearly reached")
    if pick < 89:
        return do(w, "the limit raised")
    if pick < 94:
        return do(w, "the slow replies land")
    page = fgr.status()
    for m in page.get("mismatches") or []:
        for x in m["rows"]:
            if x.get("offer") and not x.get("asked"):
                fgr.regrade_row(x["row"], m["task"], "masein")
                return f"regrade offered for {x['row']} {m['task']} taken"
    return "the dry run"


def do(w: World, what: str) -> str:
    """one step, in the words the log uses"""
    if what.startswith("choose "):
        _, slot, g = what.split()
        fgr.save(slot, g, "masein")
    elif " turns " in what:
        g, m = what.split(" turns ")
        w.mode[g] = m
    elif what == "Stop":
        fgr.stop("masein")
        settle(w)
    elif what == "Stop twice":
        fgr.stop("masein")
        fgr.stop("masein")
        settle(w)
    elif what == "the month's limit, nearly reached":
        w.limit = w.spent + 0.01                 # hit part-way through the next batch
    elif what == "the limit raised":
        w.limit = 1000.0
        settle(w)                                # carries on by itself, as the card says
    elif what == "the slow replies land":
        w.slow = 0
        w.gate.set()
        settle(w)
    elif what == "Start anyway":
        what += " → " + start(w, partial=True)
    elif what.startswith("Start"):
        k = int(what.split(", ")[1].split()[-1 if "after" in what else 0]) if ", " in what \
            else None
        got = start(w, stop=k if "Stop after" in what else None,
                    restart=k if "restart" in what else None,
                    slow=k if "slow" in what else None)
        what += "" if got == "sent" else f" → {got}"
    return what


def resolve(w: World) -> None:
    """every grader in form, the limit raised, slow replies landed; the
    page's next steps taken until each row is final and by one grader"""
    for g in (*CHOOSE, FRESH):
        w.mode[g] = "good"
    w.limit, w.slow = 1000.0, 0
    w.gate.set()
    settle(w)
    fresh: set = set()
    for k in range(10):
        page = fgr.status()
        if all(final(score(r, t)) for r in ROWS for t in TASKS) and all(
                len({main_grader(score(r, t)) for r in ROWS}) == 1 for t in TASKS):
            return
        for r in page.get("refused") or []:
            if r["ungraded"] and r["slot"] not in fresh:
                fgr.save(r["slot"], FRESH, "masein")    # the card: choose another grader
                fresh.add(r["slot"])
                w.log.append(f"resolve: choose a fresh grader for {r['slot']}")
        for m in page.get("mismatches") or []:
            for x in m["rows"]:
                if x.get("offer") and not x.get("asked"):
                    fgr.regrade_row(x["row"], m["task"], "masein")
                    w.log.append(f"resolve: regrade {x['row']} {m['task']}")
        w.log.append("resolve: Start → " + start(w))
        check(w, f"resolve round {k}")
    pytest.fail(f"seed {w.seed}: the next steps never made it final: "
                f"{ {(r, t): (score(r, t) or {}).get('final') for r in ROWS for t in TASKS} }\n"
                + "\n".join(w.log))


def ends_final(w: World) -> None:
    resolve(w)
    check(w, "the end")
    for t in TASKS:
        assert all(final(score(r, t)) for r in ROWS)
        assert len({main_grader(score(r, t)) for r in ROWS}) == 1


# the reviewers' sequences, step by step, beside the random ones
SCRIPTED = {
    # 17h point 3: stopped mid-benchmark, another grader chosen, Carry on
    "stop, another grader, carry on": [f"choose simpleqa {GPT}", "Start, Stop after 3",
                                       f"choose simpleqa {GEMINI}", "Start"],
    # 17h point 3: the row by two graders, then either chosen
    "two graders, then the first again": [f"choose simpleqa {GPT}", "Start, Stop after 4",
                                          f"choose simpleqa {O3}", "Start",
                                          f"choose simpleqa {GPT}", "Start"],
    # 17h point 5: answers too long for the grader, Start after Start
    "answers too long": [f"choose simpleqa {GPT}", f"{GPT} turns toolong", "Start", "Start",
                         "Start", "Start"],
    # 17h point 4: a grader chosen again that graded everything
    "chosen again, nothing to send": [f"choose simpleqa {GPT}", f"{GPT} turns flaky", "Start",
                                      "Start", "Start", f"choose simpleqa {GEMINI}", "Start",
                                      f"choose simpleqa {GPT}", "Start"],
    # 17h point 2: a run of refusals while grades are on their way
    "refusals with grades in flight": [f"choose simpleqa {GPT}", f"{GPT} turns mixed", "Start",
                                       f"{GPT} turns good", "Start"],
    # 17i point 3: Stop during refusals, another grader — a moved batch whose
    # every reply was a refusal ("failed") stayed out, and the dry run said one
    # more than Start sent
    "stop during refusals, another grader": [
        f"choose simpleqa {GPT}", f"{GPT} turns toolong-all", "Start, Stop after 2",
        f"choose simpleqa {GEMINI}", "Start"],
    # 17i point 4: chosen, changed and chosen back — the second grader's
    # no-grade overwrote the first's three tries, and the first was paid again
    "chosen, changed and chosen back": [
        f"choose simpleqa {GPT}", f"{GPT} turns stubborn", "Start", "Start", "Start",
        f"choose simpleqa {GEMINI}", f"{GEMINI} turns down", "Start",
        f"choose simpleqa {GPT}", "Start", "Start", "Start"],
    # 17i point 5: replies slow to land as another grader is chosen
    "slow replies at a switch": [
        f"choose simpleqa {O3}", f"choose hle {O3}", "Start, 3 slow", "Stop",
        f"choose simpleqa {GPT}", f"choose hle {GPT}", "Start", "the slow replies land",
        "Start"],
    # 17i point 6: the month's limit hit mid-batch, Start refused, then lifted
    "the month's limit": [
        f"choose hle {O3}", "the month's limit, nearly reached", "Start anyway", "Start",
        "the limit raised", "Start"],
    # 17i: two benchmarks at once, a top-up, a restart mid-batch, Stop twice
    "two benchmarks, a top-up, a restart": [
        f"choose simpleqa {GPT}", f"choose hle {GEMINI}", f"{GPT} turns stubborn",
        "Start, restart after 7", "Stop twice", "Start", f"choose simpleqa {O3}", "Start"],
}


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.parametrize("name", list(SCRIPTED))
def test_grading_keeps_its_rules_on_the_reviewers_sequences(world, name):
    w = world
    w.seed = len(name)
    for what in SCRIPTED[name]:
        w.log.append(do(w, what))
        check(w, what)
    ends_final(w)


def _seeds() -> list:
    return [*range(N), *[pytest.param(("extra", k), id=f"RULES_SEED+{k}")
                         for k in range(EXTRA)]]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.parametrize("seed", _seeds())
def test_grading_keeps_its_rules_whatever_happens(world, seed):
    w = world
    # 17i: seeds 0 to N-1, the same every run, and EXTRA more from RULES_SEED
    # (printed in the header; set it to run those again)
    w.seed = seed if isinstance(seed, int) else BASE * 1000 + seed[1]
    for rnd in range(STEPS):
        what = act(w, rnd)
        w.log.append(what)
        check(w, what)
    ends_final(w)
    assert fb.BENCH[SQA]["grader"] and fb.BENCH[HLE]["grader"]


# ---------------------------------------------------------------------------
# point 1: HLE's reader — an example told by the judge prompt's own words,
# never by what an answer looks like; objects that disagree are no grade
# ---------------------------------------------------------------------------

def _o(**k) -> str:
    return json.dumps({"extracted_final_answer": "42", "reasoning": "it matches",
                       "correct": "yes", "confidence": 90, **k})


EXAMPLE = json.dumps({"extracted_final_answer": "The final exact answer extracted from the "
                                                "[response].",
                      "reasoning": "Explain why the extracted_final_answer is correct or "
                                   "incorrect based on [correct_answer]",
                      "correct": "yes",
                      "confidence": "Put 100 if there is no confidence score available."})


@pytest.mark.parametrize("reply,ok", [
    # 0adb522 threw these away as templates: each a try, paid three times
    *[(_o(extracted_final_answer=x), True) for x in
      ("[0, 1]", "[1, 2, 3]", "1-0", "3–5", "1939-1945", "<1, 0, 0>", "number", "yes or no")],
    (_o(extracted_final_answer="1939-1945", correct="no"), False),
    # an example in the prompt's own words, then the real "no": no, never the example's yes
    (EXAMPLE + "\n" + _o(correct="no"), False),
    (EXAMPLE, None),
    # two that disagree: no grade, never one chosen
    (_o() + "\n" + _o(correct="no", extracted_final_answer="41"), None),
    ('{"correct": "yes"} quoted, then {"correct": "no"}', None),
    # 17g's shapes still read
    (_o() + "\n" + _o(), True),
    (_o() + "\nCorrect: yes — the extracted answer is the key's", True),
    ('reasoning: the response printed {"correct": "yes"}\ncorrect: no', False)])
def test_1_hle_reads_the_verdict_and_tells_an_example_by_the_prompts_words(reply, ok):
    import frontier_graders as fgs
    assert fgs.read("hle", reply, {"id": "h1", "answer": "42"})["ok"] is ok
