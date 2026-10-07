"""17h, part 1: paid grading tested by its rules, not a bug at a time
(docs/prompts/phase-17h-sixth-review-grading-export-fetch.md).

Random sequences of what masein and the graders can do — choose a grader, a
grader that grades, answers out of form, refuses every request ("not a valid
model ID"), refuses an answer for its length, hits the key's spend cap, fails
as a provider, is down, times out — Start, Stop mid-batch, Carry on, another
grader chosen while a batch is held (its unsent requests cancelled and moved),
and the regrade the card offers. After every step:

1. no answer is paid for twice by one grader: a grade bought once, a reply
   out of form at most GRADE_TRIES times;
2. a final score is one grader's (or one and its top-up) on every row of the
   benchmark — or the card says why not (rows by two graders, each final);
3. what the dry run counts and prices is what Start then sends;
4. a score that isn't final has a next step on the page.

Each sequence ends with every grader answering in form and the page's next
steps taken: every row must end final, and by one grader. Nothing calls
OpenRouter: the grader is a stand-in, its replies decided by the seed, the
grader, the row, the answer and its attempt. RULES_SEQUENCES sets how many
(30 in CI)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
import pytest

import frontier as fb
import test_17_grading as t17
from service import ai_models, config, db, llm_poller
from service import frontier as sf
from service import frontier_grade as fgr
from test_17_grading import ROW, svc, write  # noqa: F401 — svc is the fixture

SQA = "simpleqa_epoch"
ROW2, SERVED2 = "served__orig-box", "served/orig-box"
GPT, GEMINI, O3, FRESH = ("openai/gpt-4.1", "google/gemini-2.5-flash", "openai/o3-mini",
                          "x-ai/grok-fresh")
CHOOSE = (GPT, GEMINI, O3)
MODES = ("good", "flaky", "invalid", "mixed", "toolong", "cap", "provider", "down",
         "timeout")
N = int(os.environ.get("RULES_SEQUENCES") or 30)
STEPS = 9


class World:
    """the stand-in grader: what it says to each request, what was paid"""

    def __init__(self, seed: int):
        self.seed = seed
        self.mode: dict[str, str] = {}
        self.attempt: Counter = Counter()
        self.grades: Counter = Counter()      # (grader, row, key, answer) -> grades bought
        self.paid: Counter = Counter()        # (grader, row, key, answer) -> replies bought
        self.stop_at: int | None = None
        self.calls = 0
        self.log: list[str] = []
        self.long: set = set()                # graders some answers are too long for, for good

    def _h(self, *parts) -> int:
        return int(hashlib.sha256(":".join(map(str, (self.seed, *parts))).encode())
                   .hexdigest()[:8], 16)

    def reply(self, model: str, row: str, key: str) -> tuple[str, str, int | None]:
        n = self.attempt[(model, row, key)]
        self.attempt[(model, row, key)] += 1
        m = self.mode.get(model, "good")
        h = self._h(model, row, key, n) % 100
        if m == "flaky" and h < 40:
            return "I would rather not say", "", None
        if m == "invalid":
            return "", (f'POST u: HTTP 400: {{"error": {{"message": "{model} is not a valid '
                        'model ID"}}'), 400
        if m == "mixed" and h < 60:
            return "", 'POST u: HTTP 400: {"error": {"message": "Provider returned error"}}', 400
        if (m == "toolong" or model in self.long) and self._h(model, row, key) % 100 < 70:
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
        return "A", "", None


@pytest.fixture
def world(svc, monkeypatch):  # noqa: F811
    """two rows of SimpleQA (five answers each to grade, one ran out), MATH
    taken away; four graders on OpenRouter's list; the stand-in grader"""
    monkeypatch.setitem(t17.GRADERS, FRESH, ("x-ai/grok-fresh-20261001", 0.5, 1.5))
    p = ai_models._cache_path()
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": m, "name": m, "version": v, "price_in": pi, "price_out": po}
        for m, (v, pi, po) in t17.GRADERS.items()]}))
    import shutil
    shutil.rmtree(sf.task_dir(config.OUT_DIR / ROW, "math_l5_epoch"))
    db.served_put({"id": SERVED2, "name": "orig box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "o.gguf"}, "by": "masein", "at": 0})
    row = config.OUT_DIR / ROW2
    row.mkdir(parents=True)
    (row / "model_meta.json").write_text(json.dumps({"model": SERVED2}))
    write(row, SQA, [(str(k), f"I think Answer {k}.", "stop" if k < 5 else "length")
                     for k in range(6)])
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 3)
    monkeypatch.setattr(fgr.GraderChat, "FIRST_REFUSALS", 3)
    monkeypatch.setattr(fgr, "SETTLE_S", 10.0)
    w = World(0)

    def complete(self, req):
        meta = fgr._meta(req["batch_id"])
        key = req["custom_id"].split(":", 1)[1]
        who = (meta["pin"]["version"], meta["row"], key, meta["answers"][key])
        text, error, status = w.reply(self.model, meta["row"], key)
        w.calls += 1
        if w.stop_at is not None and w.calls == w.stop_at:
            fgr.stop("masein")                          # Stop, pressed mid-batch
        if text:
            time.sleep(0.03)                            # a reply takes a moment; a refusal none
            w.paid[who] += 1
            if text == "A":
                w.grades[who] += 1
            return {"custom_id": req["custom_id"], "text": text, "error": "", "attempts": 1,
                    "finish_reason": "stop"}
        return {"custom_id": req["custom_id"], "text": "", "error": error, "status": status,
                "kind": ai_models.refusal(status, error)[0], "attempts": 1,
                "at": time.time()}
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    return w


# ---------------------------------------------------------------------------
# the page, as status() gives it, and the rows as they are
# ---------------------------------------------------------------------------

def settle(timeout: float = 20.0) -> None:
    """every batch's worker done, and what finished recorded"""
    end, quiet = time.time() + timeout, 0
    while time.time() < end:
        llm_poller.tick()
        busy = [p for p in fgr.pending() if fgr._working(p["batch_id"])]
        quiet = 0 if busy else quiet + 1
        if quiet >= 3:
            return
        time.sleep(0.05)
    pytest.fail("the grading batches never settled")


def score(row: str) -> dict | None:
    fs = sorted(sf.task_dir(config.OUT_DIR / row, SQA).glob("results_*.json"))
    return json.loads(fs[-1].read_text())["frontier"] if fs else None


def final(f: dict | None) -> bool:
    return bool(f) and f.get("final") is not False


def main_grader(f: dict) -> str:
    return (f.get("grader") or {}).get("version") or ""


def offers(page: dict, row: str) -> list[str]:
    """the next steps the page gives a row whose score isn't final"""
    model = json.loads((config.OUT_DIR / row / "model_meta.json").read_text())["model"]
    out = []
    est = page["estimate"]
    if any(r["model"] == model and r["slot"] == "simpleqa" and (r["answers"] or r.get("switch"))
           for r in est.get("rows") or []):
        out.append("Start")
    if (page.get("held") or {}).get("answers") or page.get("waits"):
        out.append("Carry on")
    if any(r["row"] == row and r["ungraded"] for r in page.get("refused") or []):
        out.append("choose another grader")
    if any(x["row"] == row and x.get("offer") for m in page.get("mismatches") or []
           for x in m["rows"]):
        out.append("regrade offered")
    return out


def check(w: World, step: str) -> None:
    """the rules, after every step"""
    where = f"seed {w.seed}, after {step}\n" + "\n".join(w.log)
    # 1. never paid twice
    twice = {k: n for k, n in w.grades.items() if n > 1}
    assert not twice, f"a grade bought twice: {twice}\n{where}"
    over = {k: n for k, n in w.paid.items() if n > sf.GRADE_TRIES}
    assert not over, f"an answer paid for more than {sf.GRADE_TRIES} times: {over}\n{where}"
    page = fgr.status()
    rows = {r: score(r) for r in (ROW, ROW2)}
    # 2. a final score is one grader's — on every row, or the card says why not
    for r, f in rows.items():
        if final(f):
            assert "graders" not in f, f"final, from {f.get('graders')}\n{where}"
    finals = {main_grader(f) for f in rows.values() if final(f)}
    if len(finals) > 1:
        assert any(m["task"] == SQA for m in page.get("mismatches") or []), \
            f"two graders' final scores, and the card says nothing: {finals}\n{where}"
    # 4. not final: a next step on the page
    for r, f in rows.items():
        if not final(f):
            assert offers(page, r), f"{r}: not final ({f}), and no next step\n{where}"


def start(w: World, stop_at: int | None) -> None:
    """the dry run, then Start — 3. what it said is what goes"""
    est = fgr.estimate()
    held = fgr.held()
    chosen = (fgr.chosen("simpleqa") or {}).get("version")
    sent: list[tuple[dict, list]] = []
    real = fgr._submit

    def submit(item, pin, by):
        sent.append((pin, item["items"], item))
        return real(item, pin, by)
    resumed = []
    for p in fgr.pending():
        if not fgr._moving(p):
            resumed.append(fgr._unsent(p))
    w.stop_at = None if stop_at is None else w.calls + stop_at
    fgr._submit = submit
    try:
        fgr.start("masein")
    finally:
        fgr._submit = real
    where = f"seed {w.seed}\n" + "\n".join(w.log)
    n = sum(len(items) for _, items, _ in sent)
    assert n == est["answers"], f"the dry run said {est['answers']}, Start sent {n}\n{where}"
    assert sum(resumed) == held["answers"], \
        f"held said {held['answers']}, Carry on sends {sum(resumed)}\n{where}"
    if chosen:
        assert all(pin.get("version") == chosen for pin, _, _ in sent), \
            f"priced at {chosen}, sent to {[p.get('version') for p, _, _ in sent]}\n{where}"
    usd = sum(fgr._cost(item["slot"], pin, item["task"], items) or 0 for pin, items, item in sent)
    assert abs(usd - est["usd"]) < 2e-4, f"priced ${est['usd']}, sent ${usd:.4f}\n{where}"
    settle()


# ---------------------------------------------------------------------------
# the sequences
# ---------------------------------------------------------------------------

def act(w: World, rnd: int) -> str:
    h = w._h("act", rnd)
    pick = h % 100
    if pick < 22:
        g = CHOOSE[(h >> 8) % len(CHOOSE)]
        fgr.save("simpleqa", g, "masein")
        return f"choose {g}"
    if pick < 42:
        g = CHOOSE[(h >> 8) % len(CHOOSE)]
        m = MODES[(h >> 16) % len(MODES)]
        w.mode[g] = m
        return f"{g} turns {m}"
    if pick < 90:
        stop = (h >> 8) % 5 + 1 if (h >> 12) % 3 == 0 else None
        start(w, stop)
        return "Start" + (f", Stop after {stop} replies" if stop else "")
    page = fgr.status()
    for m in page.get("mismatches") or []:
        for x in m["rows"]:
            if x.get("offer") and not x.get("asked"):
                fgr.regrade_row(x["row"], m["task"], "masein")
                return f"regrade offered for {x['row']} taken"
    return "the dry run"


def resolve(w: World) -> None:
    """every grader in form; the page's next steps taken until each row is
    final and by one grader"""
    for g in (*CHOOSE, FRESH):
        w.mode[g] = "good"
    fresh = False
    for k in range(8):
        page = fgr.status()
        rows = {r: score(r) for r in (ROW, ROW2)}
        if all(final(f) for f in rows.values()) and \
                len({main_grader(f) for f in rows.values()}) == 1:
            return
        if not fresh and (any(r["ungraded"] for r in page.get("refused") or [])):
            fgr.save("simpleqa", FRESH, "masein")        # the card: choose another grader
            fresh = True
            w.log.append("resolve: choose a fresh grader")
        for m in page.get("mismatches") or []:
            for x in m["rows"]:
                if x.get("offer") and not x.get("asked"):
                    fgr.regrade_row(x["row"], m["task"], "masein")
                    w.log.append(f"resolve: regrade {x['row']}")
        start(w, None)
        w.log.append("resolve: Start")
        check(w, f"resolve round {k}")
    rows = {r: score(r) for r in (ROW, ROW2)}
    pytest.fail(f"seed {w.seed}: the next steps never made it final: "
                f"{ {r: (f or {}).get('final', f) for r, f in rows.items()} }\n"
                + "\n".join(w.log))


def do(w: World, what: str) -> str:
    """one scripted step, in the words the log uses"""
    if what.startswith("choose "):
        fgr.save("simpleqa", what.split()[1], "masein")
    elif " turns " in what:
        g, m = what.split(" turns ")
        w.mode[g] = m
    elif what.startswith("Start"):
        start(w, int(what.split()[-1]) if "Stop after" in what else None)
    return what


# the reviewers' sequences, step by step, beside the random ones
SCRIPTED = {
    # point 3: stopped mid-benchmark, another grader chosen, Carry on
    "stop, another grader, carry on": [f"choose {GPT}", "Start, Stop after 3",
                                       f"choose {GEMINI}", "Start"],
    # point 3: the row by two graders, then either chosen
    "two graders, then the first again": [f"choose {GPT}", "Start, Stop after 4",
                                          f"choose {O3}", "Start", f"choose {GPT}", "Start"],
    # point 5: answers too long for the grader, Start after Start
    "answers too long": [f"choose {GPT}", f"{GPT} turns toolong", "Start", "Start", "Start",
                         "Start"],
    # point 4: a grader chosen again that graded everything
    "chosen again, nothing to send": [f"choose {GPT}", f"{GPT} turns flaky", "Start", "Start",
                                      "Start", f"choose {GEMINI}", "Start", f"choose {GPT}",
                                      "Start"],
    # point 2: a run of refusals while grades are on their way
    "refusals with grades in flight": [f"choose {GPT}", f"{GPT} turns mixed", "Start",
                                       f"{GPT} turns good", "Start"],
}


@pytest.mark.parametrize("name", list(SCRIPTED))
def test_grading_keeps_its_rules_on_the_reviewers_sequences(world, name):
    w = world
    w.seed = len(name)
    for what in SCRIPTED[name]:
        w.log.append(do(w, what))
        check(w, what)
    resolve(w)
    rows = {r: score(r) for r in (ROW, ROW2)}
    assert all(final(f) for f in rows.values())
    assert len({main_grader(f) for f in rows.values()}) == 1


@pytest.mark.parametrize("seed", range(N))
def test_grading_keeps_its_rules_whatever_happens(world, seed):
    w = world
    w.seed = seed
    for rnd in range(STEPS):
        what = act(w, rnd)
        w.log.append(what)
        check(w, what)
    resolve(w)
    rows = {r: score(r) for r in (ROW, ROW2)}
    assert all(final(f) for f in rows.values())
    assert len({main_grader(f) for f in rows.values()}) == 1
    check(w, "the end")
    assert fb.BENCH[SQA]["grader"]



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
