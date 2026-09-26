"""12i.1: the judge test — how masein picks a judge, and how the board earns
the right to drop "provisional judge".

1. masein marks answers already on file, one at a time: a mix of Knowledge
   exam topics and models (0–4), and the Everyday questions a judge marks
   (pass or fail). JUDGE_TEST_N of them; the judges' marks are never shown
   while he marks, and every mark is saved as it is given.
2. Up to four candidate judges mark the same answers, with the board's own
   judge prompts, the cost said before anything runs.
3. One table: how often each gives his mark, how often within a point, the
   weighted kappa, and the cost per 1,000 answers. "Use this judge" makes one
   the judge (a new judge version, service/ai_models.py).

When the judge this server runs now has a weighted kappa of JUDGE_KAPPA_MIN
or more against his marks on at least JUDGE_TEST_MIN answers, the
calibration on file says so, and "provisional judge" comes off.

An Everyday pass is a 4 and a fail a 0 here, so one kappa covers both.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
import time
from pathlib import Path

from . import ai_models, config, db, llm

PERSON = "person"
SEED = 1234
KINDS = {"exam": "Knowledge exam", "everyday": "Everyday tasks"}
# a judge request's size, for the estimate before a run: the board's judge
# prompts with a question, a reference and an answer, and a short reply
TOKENS_IN, TOKENS_OUT = 1500, 200
CAL_FILE = "judge_calibration.json"


def _scripts() -> None:
    here = str(Path(__file__).resolve().parent.parent / "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)


def _sample_path() -> Path:
    return config.BENCH_ROOT / "ai" / "judge_test.json"


# ---------------------------------------------------------------------------
# 1. the answers
# ---------------------------------------------------------------------------

def _exam_rows() -> list[dict]:
    _scripts()
    import judge_calibrate as jc
    if not config.OUT_DIR.is_dir():
        return []
    return [{"key": "exam:" + r["id"], "kind": "exam", "model": r["model"].replace("__", "/", 1),
             "task": r["task"], "topic": r["category"], "question": r["prompt"],
             "reference": r["reference"], "answer": r["answer"], "judge": r["judge_score"]}
            for r in jc._judged_rows(config.OUT_DIR, set()) if (r["answer"] or "").strip()]


def _everyday_rows() -> list[dict]:
    _scripts()
    import everyday as ev
    qs = {q["id"]: q for q in ev.load_bank()}
    out = []
    for f in sorted(config.OUT_DIR.glob("*/everyday.json")) if config.OUT_DIR.is_dir() else []:
        e = ev.read(f.parent) or {}
        if e.get("earlier"):
            continue
        for it in e.get("items") or []:
            q = qs.get(it["id"])
            if not q or not it.get("judged") or it.get("pass") is None \
                    or not (it.get("answer_text") or "").strip():
                continue
            rubric = next(c["rubric"] for c in q["checks"] if c["type"] == "judge")
            out.append({"key": f"everyday:{f.parent.name}|{it['id']}", "kind": "everyday",
                        "model": e.get("model") or f.parent.name, "task": it["id"],
                        "topic": ev.groups().get(q["group"], q["group"]), "question": q["prompt"],
                        "reference": q.get("reference") or "", "rubric": rubric,
                        "answer": it["answer_text"], "judge": 4 if it["pass"] else 0})
    return out


def _spread(rows: list[dict], n: int) -> list[dict]:
    """n rows round-robin over (kind, topic, model), each stratum in a fixed
    shuffled order: every topic before any has two"""
    strata: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}|{r['key']}".encode()).hexdigest()):
        strata[(r["kind"], r["topic"], r["model"])].append(r)
    order = sorted(strata)
    out: list[dict] = []
    while len(out) < n and any(strata.values()):
        for k in order:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop(0))
    return out


def answers(n: int | None = None, rebuild: bool = False) -> list[dict]:
    """The answers masein marks: drawn once from what is on file and kept
    (BENCH_ROOT/ai/judge_test.json), so a mark always means the same answer.
    About one in seven are Everyday ones, when there are that many"""
    n = n or config.JUDGE_TEST_N
    p = _sample_path()
    if not rebuild:
        try:
            got = json.loads(p.read_text(encoding="utf-8"))
            if got.get("n") == n:
                return got["answers"]
        except (OSError, ValueError, KeyError):
            pass
    evd = _everyday_rows()
    n_evd = min(len(evd), n // 7)
    picked = _spread(evd, n_evd) + _spread(_exam_rows(), n - n_evd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n": n, "at": time.time(), "answers": picked}), encoding="utf-8")
    return picked


def criteria_of(a: dict) -> dict:
    """what a full answer contains, as the judge is told it: a topic's
    criteria and flags, or its rubric; an Everyday question's rubric"""
    if a["kind"] == "everyday":
        return {"rubric": a["rubric"], "criteria": [], "flags": []}
    _scripts()
    import judge
    rub = judge.rubric_for(a["task"])
    spec = rub.criteria or {}
    return {"rubric": "" if spec else rub.text,
            "criteria": [judge.label_of(c) for c in spec.get("criteria") or []],
            "flags": [judge.label_of(f) for f in spec.get("flags") or []]}


def shown(a: dict) -> dict:
    """one answer as the page shows it while masein marks: never a judge's mark"""
    return {"key": a["key"], "kind": a["kind"], "kind_label": KINDS[a["kind"]],
            "topic": a["topic"], "question": a["question"], "reference": a["reference"],
            "answer": a["answer"], "scale": "pf" if a["kind"] == "everyday" else "0-4",
            **criteria_of(a)}


def mark(key: str, value, by: str) -> None:
    """0–4, "P" or "F" (a 4 or a 0), or None: skipped"""
    if key not in {a["key"] for a in answers()}:
        raise ValueError("not one of the judge test's answers")
    if value in ("P", "F"):
        value = 4 if value == "P" else 0
    if value is not None and value not in range(5):
        raise ValueError("a mark is 0 to 4, P or F, or a skip")
    db.jt_mark(PERSON, key, value, by)


def progress() -> dict:
    got = db.jt_marks(PERSON)
    keys = [a["key"] for a in answers()]
    marked = sum(1 for k in keys if got.get(k) is not None)
    skipped = sum(1 for k in keys if k in got and got[k] is None)
    nxt = next((i for i, k in enumerate(keys) if k not in got), None)
    return {"total": len(keys), "marked": marked, "skipped": skipped, "next": nxt}


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------

def weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float | None:
    """Cohen's kappa with quadratic weights over the scores 0..k-1: a miss
    by two counts four times a miss by one. None with no pairs; 1.0 when the
    two agree and there is no spread to be chance about"""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ra = [sum(row) for row in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * ra[i] * rb[j] / n for i in range(k) for j in range(k))
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return round(1 - num / den, 4)


def agreement(person: dict[str, int | None], judge: dict[str, int | None]) -> dict:
    """{n, exact, within1, kappa} over the answers both marked"""
    keys = [k for k, v in person.items() if v is not None and judge.get(k) is not None]
    a = [person[k] for k in keys]
    b = [judge[k] for k in keys]
    n = len(keys)
    return {"n": n,
            "exact": round(sum(x == y for x, y in zip(a, b)) / n, 4) if n else None,
            "within1": round(sum(abs(x - y) <= 1 for x, y in zip(a, b)) / n, 4) if n else None,
            "kappa": weighted_kappa(a, b)}


# ---------------------------------------------------------------------------
# 2. candidates
# ---------------------------------------------------------------------------

def _runs_dir() -> Path:
    return config.BENCH_ROOT / "ai" / "judge_test_runs"


def candidate(model_id: str) -> dict:
    """a candidate judge, pinned as a job would be: {kind, id, version, name,
    provider, price_in, price_out, key}. `key` names its marks"""
    if model_id == ai_models.LOCAL:
        c = {"kind": ai_models.LOCAL, "id": ai_models.LOCAL, "name": ai_models.local_label(ask=True),
             "version": "local/" + ai_models.local_model(),
             "provider": "", "price_in": 0.0, "price_out": 0.0}
    else:
        m = ai_models.model(model_id)
        if not m:
            raise ValueError(f"{model_id} is not one of OpenRouter's text models")
        prov = ai_models.first_provider(model_id) or {}
        c = {"kind": "openrouter", "id": m["id"], "version": m["version"],
             "name": m["name"].split(": ", 1)[-1], "provider": prov.get("tag") or "",
             "provider_name": prov.get("name") or "", "precision": prov.get("precision"),
             "price_in": prov.get("price_in") if prov.get("price_in") is not None
             else m["price_in"],
             "price_out": prov.get("price_out") if prov.get("price_out") is not None
             else m["price_out"]}
    c["key"] = candidate_key(c)
    return c


def candidate_key(c: dict) -> str:
    _scripts()
    import judge
    ident = {"id": ("openrouter/" + c["version"]) if c["kind"] == "openrouter"
             else c["version"], "model": c["version"], "pin": c.get("provider", "")}
    return judge.version(ident)["key"]


def estimate(model_ids: list[str]) -> dict:
    """the cost before anything runs: per candidate, and in all"""
    n = len(answers())
    out = []
    for mid in model_ids:
        c = candidate(mid)
        usd = n * ((c["price_in"] or 0) * TOKENS_IN + (c["price_out"] or 0) * TOKENS_OUT) / 1e6
        out.append({"id": mid, "name": c["name"], "usd": round(usd, 4)})
    return {"n": n, "candidates": out, "usd": round(sum(x["usd"] for x in out), 4)}


def request_for(a: dict) -> tuple[str, int]:
    """the board's own judge prompt for this answer, and its reply cap"""
    _scripts()
    if a["kind"] == "everyday":
        import everyday as ev
        q = next(q for q in ev.load_bank() if q["id"] == a["task"])
        return ev.judge_prompt(q, a["answer"]), 200
    import judge
    rub = judge.rubric_for(a["task"])
    spec = rub.criteria
    if spec:
        return (judge.build_criteria_prompt(rub.text, spec, a["question"], a["reference"],
                                            a["answer"]), 700)
    return judge.build_prompt(rub.text, a["question"], a["reference"], a["answer"]), 200


def read_mark(a: dict, text: str) -> int | None:
    _scripts()
    if a["kind"] == "everyday":
        import everyday as ev
        v = ev.parse_verdict(text)
        return None if v is None else (4 if v["pass"] else 0)
    import judge
    spec = judge.rubric_for(a["task"]).criteria
    if spec:
        g = judge.parse_grade_criteria(text, spec)
        return None if g is None else judge.fold(g.criteria, g.flags, spec)
    score, _ = judge.parse_grade(text)
    return score


def backend(c: dict) -> llm.Backend:
    if c["kind"] == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OpenRouter has no key on this server")
        return llm.OpenRouterChat(c["id"], config.OPENROUTER_API_KEY, config.BENCH_ROOT,
                                  pin=c, role="judge_test")
    return llm.LocalOpenAI(ai_models.local_model(), "", config.BENCH_ROOT, role="judge")


def run(model_ids: list[str], by: str) -> list[dict]:
    """Each candidate marks every answer, as one batch; the poller finishes
    them (finish()). Up to four at a time"""
    if not model_ids or len(model_ids) > 4:
        raise ValueError("tick one to four candidate judges")
    why = ai_models.over_limit()
    if why:
        raise ValueError(why)
    todo = answers()
    out = []
    for mid in model_ids:
        c = candidate(mid)
        reqs = []
        for i, a in enumerate(todo):
            user, cap = request_for(a)
            reqs.append(llm.Request(custom_id=f"jt:{i}", system="", user=user, max_tokens=cap,
                                    json=True, meta={"kind": "judge_test", "key": a["key"]}))
        be = backend(c)
        bid = be.submit(reqs)
        d = _runs_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{bid}.json").write_text(json.dumps({"candidate": c, "keys": [a["key"] for a in todo],
                                                   "by": by, "at": time.time()}), encoding="utf-8")
        db.batch_add(bid, "judge_test", 0, len(reqs), be.name, c["version"])
        out.append({"batch_id": bid, "candidate": c})
    return out


def batch_backend(batch_id: str) -> llm.Backend:
    """the poller's: the candidate that ran this batch, as it was pinned"""
    meta = json.loads((_runs_dir() / f"{batch_id}.json").read_text(encoding="utf-8"))
    return backend(meta["candidate"])


def finish(batch_id: str, results: dict) -> int:
    """the candidate's marks in; how many it gave"""
    meta = json.loads((_runs_dir() / f"{batch_id}.json").read_text(encoding="utf-8"))
    c, keys = meta["candidate"], meta["keys"]
    by_key = {a["key"]: a for a in answers()}
    n = answered = 0
    for i, key in enumerate(keys):
        res = results.get(f"jt:{i}")
        a = by_key.get(key)
        if a is None or res is None or getattr(res, "error", ""):
            continue
        answered += 1                   # what it was paid for, read or not
        m = read_mark(a, getattr(res, "text", ""))
        if m is not None:
            db.jt_mark(c["key"], key, int(m), meta.get("by", ""))
            n += 1
    runs = db.ai_get("judge_test:candidates", {})
    runs[c["key"]] = {**c, "batch_id": batch_id, "at": time.time(), "marked": n,
                      "answered": answered}
    db.ai_set("judge_test:candidates", runs, meta.get("by", ""))
    calibrate()
    return n


# ---------------------------------------------------------------------------
# 3. the result
# ---------------------------------------------------------------------------

def _spent(batch_id: str) -> float:
    return db.spend_of_batch(batch_id)


def current_marks() -> dict[str, int | None]:
    """the judge this server runs now, on these answers: a candidate run of
    its version if there is one, else its marks on file (judge.json, and
    everyday.json's verdicts) when they are its version's"""
    _scripts()
    import judge
    key = judge.version()["key"]
    got = db.jt_marks(key)
    if got:
        return got
    ident = judge.identity()
    out = {}
    for a in answers():
        jf = config.OUT_DIR / a["model"].replace("/", "__") / (
            "judge.json" if a["kind"] == "exam" else "everyday.json")
        try:
            j = json.loads(jf.read_text(encoding="utf-8")).get("judge") or {}
        except (OSError, ValueError):
            continue
        v = (j.get("version") or {}).get("key")
        if (v and v == key) or (not v and j.get("id") == ident["id"]):
            out[a["key"]] = a["judge"]
    return out


def result() -> dict:
    """the table: the current judge, then each candidate — same mark,
    within 1, weighted kappa, n, cost per 1,000 answers; the best marked"""
    _scripts()
    import judge
    person = db.jt_marks(PERSON)
    rows = []
    # a page shows this: a local judge is named by its weights, asked once
    ai_models.local_name(ask=True)
    cur_key = judge.version()["key"]
    runs = db.ai_get("judge_test:candidates", {})
    cur = current_marks()
    if cur:
        rows.append({"key": cur_key, "name": judge.version()["label"], "current": True,
                     "id": judge.identity().get("slug") or judge.identity()["id"],
                     **agreement(person, cur), "per_1000": None})
    for key, c in runs.items():
        if key == cur_key:
            if rows:
                rows[0]["per_1000"] = _per_1000(c)
            continue
        rows.append({"key": key, "name": c["name"], "current": False, "id": c["id"],
                     "provider": c.get("provider_name") or c.get("provider") or "",
                     **agreement(person, db.jt_marks(key)), "per_1000": _per_1000(c)})
    scored = [r for r in rows if r["kappa"] is not None and r["n"]]
    if scored:
        best = max(scored, key=lambda r: (r["kappa"], r["exact"] or 0))
        best["best"] = True
    return {"rows": rows, "person": progress(), "kappa_min": config.JUDGE_KAPPA_MIN,
            "n_min": config.JUDGE_TEST_MIN}


def _per_1000(c: dict) -> float | None:
    """what 1,000 answers cost it: the batch's spend over what it answered"""
    n = c.get("answered") or c.get("marked") or 0
    return round(_spent(c["batch_id"]) / n * 1000, 2) if n else None


def calibrate() -> dict | None:
    """The calibration the page reads (results/full/judge_calibration.json),
    from the judge test, for the judge this server runs now: calibrated at a
    weighted kappa of JUDGE_KAPPA_MIN or more on JUDGE_TEST_MIN answers"""
    _scripts()
    import judge
    ag = agreement(db.jt_marks(PERSON), current_marks())
    if not ag["n"]:
        return None
    ident = judge.identity()
    out = {"kappa": ag["kappa"], "n": ag["n"], "exact": ag["exact"], "within1": ag["within1"],
           "kappa_min": config.JUDGE_KAPPA_MIN, "n_min": config.JUDGE_TEST_MIN,
           "calibrated": bool(ag["kappa"] is not None and ag["kappa"] >= config.JUDGE_KAPPA_MIN
                              and ag["n"] >= config.JUDGE_TEST_MIN),
           "method": "judge test: weighted kappa (quadratic) against a person's marks",
           "by": (db.ai_get_meta("judge_test:candidates") or {}).get("by") or "masein",
           "judge": {"id": ident["id"], "version": judge.version(ident)["key"]}}
    p = config.OUT_DIR / CAL_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def use(key: str, by: str) -> dict:
    """make a candidate the judge — a new judge version"""
    runs = db.ai_get("judge_test:candidates", {})
    c = runs.get(key)
    if not c:
        raise ValueError("no such candidate in the judge test")
    return ai_models.save("judge", c["id"], by)
