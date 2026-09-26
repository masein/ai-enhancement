"""12i.2: the question builder — new Knowledge exam or Everyday questions,
written by an AI model, checked by another, and reviewed by a person before
they join the bank as a new version. Three steps on one page:

1. What: the kind, the topic or group, the level, how many, the writer and
   the checker, and the writing instructions (their output section locked).
2. Try 10: the writer makes ten; masein accepts, edits or rejects each, with a
   reason, and those reasons go into the prompt for the rest.
3. Make the rest, checked: every question is answered blind by the checker
   and — for the exam — marked by the judge; near-duplicates are found; the
   flagged ones and a tenth of the rest are reviewed; Publish.

A draft is one row in qb_drafts: its spec, the prompt used, every question
with its review and flags. The model calls are batches the poller finishes
(kind "qb"), as every other model call on the board is, so a draft survives a
reload and a restart.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from . import ai_models, config, contamination, db, llm

REPO = Path(__file__).resolve().parent.parent
# the default writing instructions, where the image carries them (docs/ is not
# in it: 12i.3). docs/prompts/phase-12i/ holds the same text, as the brief
# names it, and a test keeps the two the same
PROMPTS = {"knowledge": REPO / "eval_tasks" / "fr" / "question-builder-prompt.md",
           "everyday": REPO / "eval_tasks" / "everyday" / "question-builder-prompt.md"}
KINDS = ("knowledge", "everyday")
LEVELS = ("general public", "specialist")
TRY_N = 10                   # step 2's first ten
MIN_REVIEWED = 5             # "Make the rest" after this many of them
SUGGEST_MIN = 40             # under this, a topic gets no score of its own in Improve
SAMPLE_SHARE = 0.10          # of the unflagged, reviewed at random
REASONS = ("too easy", "two right answers", "trivia", "unclear", "wrong answer", "other")
CONCERNS = ("ambiguous", "time-sensitive", "trivia")
OUTPUT_HEAD = "## Output"
# a request's size, for the estimate before anything runs (tokens)
WRITE_IN, WRITE_OUT = 3000, 350          # per request; per question
LONG_OUT = 1300                          # a Summarise question carries a long text
CHECK_IN, CHECK_OUT = 400, 250
JUDGE_IN, JUDGE_OUT = 1500, 200
ROLE = {"writer": "exam", "checker": "checker", "judge": "judge"}

_lock = threading.RLock()

CHECK_SYSTEM = (
    "Answer the question as an expert would, in 2 to 5 sentences. Then say whether the question "
    "itself has a problem: \"ambiguous\" (it has more than one right answer, or it is unclear "
    "what it asks), \"time-sensitive\" (the answer changes with the news, prices or the year), "
    "or \"trivia\" (it asks for a date, name or number for its own sake). Reply with one JSON "
    "object and nothing else: {\"answer\": <your answer>, \"concerns\": <a list, empty when "
    "there is none>, \"why\": <one short sentence when there is a concern>}.")


def _scripts() -> None:
    here = str(REPO / "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the writing instructions: editable, but for the output section
# ---------------------------------------------------------------------------

def split_prompt(text: str) -> tuple[str, str]:
    """(the editable instructions, the locked "## Output" section)"""
    lines = text.splitlines()
    start = next((i for i, x in enumerate(lines) if x.strip().startswith(OUTPUT_HEAD)), None)
    if start is None:
        return text.strip(), ""
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
               len(lines))
    return ("\n".join(lines[:start] + lines[end:]).strip(),
            "\n".join(lines[start:end]).strip())


def default_prompt(kind: str) -> dict:
    editable, locked = split_prompt(PROMPTS[kind].read_text(encoding="utf-8"))
    return {"editable": editable, "locked": locked, "path": str(PROMPTS[kind].relative_to(REPO))}


def _editable(kind: str, text: str | None) -> str:
    """what masein wrote, with any output section he pasted in taken out: the
    locked one is always the default's, so the result can always be read"""
    if not (text or "").strip():
        return default_prompt(kind)["editable"]
    return split_prompt(text)[0]


def compose(kind: str, editable: str, fill: dict, feedback: str = "") -> str:
    text = editable
    for k, v in fill.items():
        text = text.replace("{" + k + "}", str(v))
    parts = [text.strip()]
    if feedback:
        parts.append(feedback.strip())
    parts.append(default_prompt(kind)["locked"])
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# what can be built: topics, groups, and each job's model
# ---------------------------------------------------------------------------

def suggest_subtopics(topic: str) -> list[str]:
    """the topic's brief (exam_build.TOPIC_BRIEFS) cut at its commas — a start
    masein edits, not a taxonomy"""
    _scripts()
    import exam_build as eb
    brief = eb.TOPIC_BRIEFS.get(topic, "")
    parts = [p.strip() for p in re.split(r"[,;:]", brief)]
    return [p for p in parts if p and not p.lower().startswith("no ")][:8]


def topics() -> list[dict]:
    _scripts()
    import categories
    import exam_build as eb
    bank = eb.load_bank(config.EXAM_DIR) if config.EXAM_DIR.is_dir() else {}
    return [{"name": t, "slug": categories.topic_slug(t), "subtopics": suggest_subtopics(t),
             "bank": len(bank.get(t, []))} for t in eb.TOPICS]


def everyday_groups() -> list[dict]:
    _scripts()
    import everyday as ev
    counts: dict[str, int] = {}
    for q in ev.load_bank():
        counts[q["group"]] = counts.get(q["group"], 0) + 1
    return [{"id": g, "label": label, "bank": counts.get(g, 0)}
            for g, label in ev.groups().items()]


def who(job: str, override: dict | None = None) -> dict:
    """the model doing a job for this batch — the AI models page's, or the one
    chosen here — in words, with its price per million tokens (None: unknown)"""
    if override:
        return {"label": override["name"], "id": override.get("version") or override["id"],
                "kind": override["kind"], "price_in": override.get("price_in"),
                "price_out": override.get("price_out"), "chosen_here": True}
    c = ai_models.choice(job)
    if c and c.get("kind") == "openrouter":
        return {"label": ai_models.label(job), "id": c["version"], "kind": "openrouter",
                "price_in": c.get("price_in"), "price_out": c.get("price_out")}
    p, m = ai_models._env(job)
    free = (c and c.get("kind") == ai_models.LOCAL) or p in (ai_models.LOCAL, "fake", "stub")
    return {"label": ai_models.label(job), "id": f"{p}/{m}" if p else "", "kind": p or "",
            "price_in": 0.0 if free else None, "price_out": 0.0 if free else None}


def _override(model_id: str) -> dict | None:
    """'' keeps the AI models page's choice; else a model pinned for this
    batch only, as the judge test pins a candidate"""
    if not model_id:
        return None
    from . import judge_test
    return judge_test.candidate(model_id)


def _backend(role: str, override: dict | None) -> llm.Backend:
    if override:
        if override["kind"] == "openrouter":
            if not config.OPENROUTER_API_KEY:
                raise ValueError("OpenRouter has no key on this server")
            return llm.OpenRouterChat(override["id"], config.OPENROUTER_API_KEY,
                                      config.BENCH_ROOT, pin=override, role=role)
        return llm.LocalOpenAI(ai_models.local_model(), "", config.BENCH_ROOT, role=role)
    return llm.client(role)


def blocked(job: str, override: dict | None = None) -> str:
    if override:
        if override["kind"] == "openrouter" and not config.OPENROUTER_API_KEY:
            return "OpenRouter has no key on this server"
        return ai_models.over_limit() if override["kind"] == "openrouter" else ""
    return llm.blocked(ROLE[job])


# ---------------------------------------------------------------------------
# the estimate, before anything runs
# ---------------------------------------------------------------------------

def _usd(w: dict, tin: float, tout: float) -> float | None:
    if w.get("price_in") is None or w.get("price_out") is None:
        return None
    return (w["price_in"] * tin + w["price_out"] * tout) / 1e6


def estimate(kind: str, count: int, writer: dict, checker: dict, group: str = "") -> dict:
    count = max(0, int(count))
    per_q = LONG_OUT if group == "summarising" else WRITE_OUT
    per_req = _per_request(kind, group, writer)
    judge = who("judge")
    judged = count if kind == "knowledge" else round(count * 0.25)
    parts = [("writer", _usd(writer, -(-count // per_req) * WRITE_IN, count * per_q)),
             ("checker", _usd(checker, count * CHECK_IN, count * CHECK_OUT)),
             ("judge", _usd(judge, judged * JUDGE_IN, judged * JUDGE_OUT))]
    unknown = [k for k, v in parts if v is None]
    usd = round(sum(v for _, v in parts if v), 4)
    if unknown:
        line = f"cost not known: no price for the {' and '.join(unknown)} here"
    elif usd == 0:
        line = "free: every step runs on the local model"
    else:
        line = f"about ${usd:,.2f}"
    return {"usd": None if unknown else usd, "line": line,
            "parts": {k: (None if v is None else round(v, 4)) for k, v in parts}}


def _per_request(kind: str, group: str, writer: dict) -> int:
    if writer.get("kind") == ai_models.LOCAL:
        return 2
    return 3 if (kind == "everyday" and group == "summarising") else 10


# ---------------------------------------------------------------------------
# a draft
# ---------------------------------------------------------------------------

def get(draft_id: str) -> dict:
    d = db.qb_get(draft_id)
    if not d:
        raise KeyError(draft_id)
    return d


def _put(d: dict) -> dict:
    d["updated_at"] = time.time()
    db.qb_put(d)
    return d


def create(spec: dict, by: str) -> dict:
    """Step 1 done: a draft, and its first ten being written"""
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError("kind is knowledge or everyday")
    try:
        count = int(spec.get("count") or 0)
    except (TypeError, ValueError):
        raise ValueError("how many is a number") from None
    if count < 1:
        raise ValueError("how many: at least 1")
    s: dict = {"kind": kind, "count": count, "dedup": bool(spec.get("dedup", True))}
    if kind == "knowledge":
        names = {t["name"] for t in topics()}
        if spec.get("topic") not in names:
            raise ValueError("choose one of the exam's topics")
        level = spec.get("level") or LEVELS[0]
        if level not in LEVELS:
            raise ValueError("level is general public or specialist")
        subs = [x.strip() for x in (spec.get("subtopics") or []) if str(x).strip()]
        s.update(topic=spec["topic"], level=level,
                 subtopics=subs or suggest_subtopics(spec["topic"]))
    else:
        _scripts()
        import everyday as ev
        g = (spec.get("group") or "").strip()
        known = ev.groups()
        if g == "new":
            label = (spec.get("new_label") or "").strip()
            about = (spec.get("new_about") or "").strip()
            if not label or not about:
                raise ValueError("a new group needs a name and one line on what it tests")
            gid = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:30]
            if not gid or gid in known:
                raise ValueError(f"{label!r} is a group already, or not a name")
            s.update(group=gid, group_label=label, group_about=about, new_group=True)
        elif g in known:
            s.update(group=g, group_label=known[g])
        else:
            raise ValueError("choose one of the Everyday groups, or a new one")
    writer_o, checker_o = _override(spec.get("writer", "")), _override(spec.get("checker", ""))
    why = blocked("writer", writer_o)
    if why:
        raise ValueError(f"the question writer can't run: {why}")
    editable = _editable(kind, spec.get("prompt"))
    d = {"id": uuid.uuid4().hex[:10], "kind": kind, "spec": s, "by": by,
         "created_at": time.time(), "stage": "try", "status": "writing", "error": "",
         "prompt": {"editable": editable,
                    "edited": editable != default_prompt(kind)["editable"],
                    "sha": _sha(editable + "\n" + default_prompt(kind)["locked"])},
         "writer_o": writer_o, "checker_o": checker_o,
         "writer": who("writer", writer_o), "checker": who("checker", checker_o),
         "feedback": "", "items": [], "prompts": {}, "published": None, "empty_writes": 0}
    d["estimate"] = estimate(kind, count, d["writer"], d["checker"], s.get("group", ""))
    with _lock:
        # asked first, stored after: a writer that can't be asked leaves no
        # draft stuck on "writing" (the poller waits on the lock meanwhile)
        _write(d, min(TRY_N, count))
        return _put(d)


def _fill(d: dict, n: int) -> dict:
    s = d["spec"]
    if d["kind"] == "knowledge":
        return {"topic": s["topic"], "subtopics": "; ".join(s["subtopics"]),
                "level": s["level"], "count": n}
    g = s["group"] + (f" — {s['group_about']}" if s.get("new_group") else "")
    return {"group": g, "count": n}


def _write(d: dict, n: int) -> None:
    """ask the writer for n more, as one batch the poller finishes"""
    s = d["spec"]
    per = _per_request(d["kind"], s.get("group", ""), d["writer"])
    start = len(d["items"])
    reqs, prompts = [], {}
    for k in range(0, n, per):
        m = min(per, n - k)
        text = compose(d["kind"], d["prompt"]["editable"], _fill(d, m), d["feedback"])
        sha = _sha(text)
        prompts[sha] = text
        out = (LONG_OUT if s.get("group") == "summarising" else WRITE_OUT) * m + 400
        reqs.append(llm.Request(
            f"qbw:{d['id']}:{start + k}", "", text, max_tokens=min(out, 4000),
            meta={"kind": "qb_write", "draft": d["id"], "draft_kind": d["kind"], "count": m,
                  "start": start + k,
                  "topic": s.get("topic", ""), "group": s.get("group", ""), "sha": sha}))
    d["prompts"].update(prompts)
    _submit(d, "write", "writer", reqs)
    d["status"] = "writing"


def _submit(d: dict, step: str, job: str, reqs: list) -> None:
    override = d.get(f"{job}_o") if job in ("writer", "checker") else None
    be = _backend(ROLE[job], override)
    bid = be.submit(reqs)
    meta = {"draft": d["id"], "step": step, "job": job, "override": override,
            "ids": [r.custom_id for r in reqs],
            "shas": {r.custom_id: r.meta.get("sha", "") for r in reqs}}
    p = _batches_dir() / f"{bid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta), encoding="utf-8")
    db.batch_add(bid, "qb", 0, len(reqs), getattr(be, "name", "?"), getattr(be, "model", ""))
    d["busy"] = bid


def _batches_dir() -> Path:
    return config.BENCH_ROOT / "builder" / "batches"


def batch_backend(batch_id: str) -> llm.Backend:
    meta = json.loads((_batches_dir() / f"{batch_id}.json").read_text(encoding="utf-8"))
    return _backend(ROLE[meta["job"]], meta.get("override"))


# ---------------------------------------------------------------------------
# reading what the writer wrote
# ---------------------------------------------------------------------------

def _lines(text: str) -> list[dict]:
    """JSON Lines, as asked; a model that wrapped them in an array or a fence
    is read too"""
    out = []
    for line in (text or "").splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if isinstance(v, dict):
                out.append(v)
    if not out:
        out = [v for v in (llm.extract_array(text or "") or []) if isinstance(v, dict)]
    return out


def _question(d: dict, raw: dict) -> tuple[dict, str]:
    """(the question in the bank's own shape, '' or why it can't be used)"""
    s = d["spec"]
    if d["kind"] == "knowledge":
        crit = [str(c).strip() for c in (raw.get("criteria") or []) if str(c).strip()]
        q = {"question": str(raw.get("question") or raw.get("prompt") or "").strip(),
             "reference": str(raw.get("reference") or "").strip(), "criteria": crit,
             "subtopic": str(raw.get("subtopic") or "").strip(), "level": s["level"],
             "notes": str(raw.get("notes") or "").strip()}
        if not q["question"] or not q["reference"]:
            return q, "no question or no reference"
        if len(crit) < 2:
            return q, "fewer than two criteria: the judge would have nothing to check"
        return q, ""
    _scripts()
    import everyday as ev
    q = {"group": s["group"], "skill": str(raw.get("skill") or "").strip(),
         "difficulty": raw.get("difficulty") if raw.get("difficulty") in (1, 2, 3) else 2,
         "prompt": str(raw.get("prompt") or "").strip(),
         "reference": str(raw.get("reference") or "").strip(),
         "checks": raw.get("checks") if isinstance(raw.get("checks"), list) else [],
         "notes": str(raw.get("notes") or "").strip()}
    if not q["prompt"] or not q["reference"]:
        return q, "no question or no reference"
    if not q["checks"]:
        return q, "no checks"
    bad = next((why for why in map(ev._bad_check, q["checks"]) if why), "")
    if bad:
        return q, f"a check can't be read: {bad}"
    return q, _self_check(q)


def _self_check(q: dict) -> str:
    """'' when an Everyday question can stand: its reference passes its own
    checks, and pasting the message back fails a Shorten or Summarise one"""
    _scripts()
    import everyday as ev
    for c in q["checks"]:
        ok, why = ev.run_check(c, q["reference"], q["prompt"])
        if ok is False:
            return f"its reference fails its own checks: {why}"
    if q["group"] in ("shorten", "summarising") and not ev.failures(q, q["prompt"]):
        return "pasting the message back passes its checks"
    return ""


def _text(d: dict, q: dict) -> str:
    return q["question"] if d["kind"] == "knowledge" else q["prompt"]


# ---------------------------------------------------------------------------
# the poller's side
# ---------------------------------------------------------------------------

def finish(batch_id: str, results: dict) -> int:
    meta = json.loads((_batches_dir() / f"{batch_id}.json").read_text(encoding="utf-8"))
    with _lock:
        d = get(meta["draft"])
        if d.get("busy") == batch_id:
            d["busy"] = ""
        step = meta["step"]
        try:
            n = {"write": _finish_write, "check": _finish_check,
                 "judge": _finish_judge}[step](d, meta, results)
        except (llm.LLMError, ValueError, OSError) as e:
            # the next step could not start (its model is unavailable, say):
            # what came back is kept, and Resume carries on from it
            d.update(status="failed", error=f"stopped after the {meta['job']}: {e}"[:400])
            n = 0
        _put(d)
        return n


def failed(batch_id: str, why: str) -> None:
    """a batch the provider failed: the draft stops and says so; what it has
    stays, and Resume carries on"""
    try:
        meta = json.loads((_batches_dir() / f"{batch_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    with _lock:
        d = get(meta["draft"])
        d.update(busy="", status="failed", error=f"the {meta['job']} failed: {why}"[:400])
        _put(d)


def _finish_write(d: dict, meta: dict, results: dict) -> int:
    got = 0
    for cid in meta["ids"]:
        res = results.get(cid)
        if res is None or getattr(res, "error", ""):
            continue
        sha = meta.get("shas", {}).get(cid, "")
        for raw in _lines(getattr(res, "text", "")):
            q, why = _question(d, raw)
            d["items"].append({"n": len(d["items"]) + 1, "stage": d["stage"], "q": q,
                               "auto": why, "verdict": None, "reason": "", "flags": [],
                               "answer": None, "mark": None, "sample": False, "edited": False,
                               "prompt_sha": sha})
            got += 1
    d["empty_writes"] = 0 if got else d.get("empty_writes", 0) + 1
    if d["status"] == "cancelled":
        return got
    if d["stage"] == "try":
        if not d["items"]:
            d.update(status="failed", error="nothing in the writer's reply could be read as "
                                             "questions — Resume asks again")
        else:
            d["status"] = "review"
        return got
    left = d["spec"]["count"] - len(d["items"])
    if left > 0 and d["empty_writes"] < 2:
        _write(d, min(TRY_N, left))
    else:
        _check(d)
    return got


def _needs_check(it: dict) -> bool:
    return not it["auto"] and it["verdict"] != "reject" and it["answer"] is None


def _check(d: dict) -> None:
    """every question still standing, answered blind by the checker"""
    todo = [it for it in d["items"] if _needs_check(it)]
    if not todo:
        # answered, but the judge never marked them (it failed, or a cancel)
        waiting = [it for it in d["items"] if it.get("judge_pending") and not it["auto"]]
        if waiting:
            _judge(d, waiting)
        else:
            _finalize(d)
        return
    reqs = []
    for it in todo:
        q = it["q"]
        if d["kind"] == "knowledge":
            reqs.append(llm.Request(f"qbc:{d['id']}:{it['n']}", CHECK_SYSTEM, q["question"],
                                    max_tokens=600, json=True,
                                    meta={"kind": "qb_check", "ref": q["reference"]}))
        else:
            # as a person would type it: no checks, no reference
            reqs.append(llm.Request(f"qbc:{d['id']}:{it['n']}", "", q["prompt"],
                                    max_tokens=1500, meta={"kind": "qb_check",
                                                           "ref": q["reference"]}))
    _submit(d, "check", "checker", reqs)
    d["status"] = "checking"


def _finish_check(d: dict, meta: dict, results: dict) -> int:
    _scripts()
    import everyday as ev
    by_n = {it["n"]: it for it in d["items"]}
    to_judge = []
    for cid in meta["ids"]:
        it = by_n.get(int(cid.rsplit(":", 1)[1]))
        if it is None:
            continue
        res = results.get(cid)
        if res is None or getattr(res, "error", ""):
            it["answer"] = ""
            _flag(it, "checker", "the checker gave no answer: "
                  + (getattr(res, "error", "") or "nothing came back")[:200])
            continue
        text = getattr(res, "text", "") or ""
        if d["kind"] == "knowledge":
            got = llm.extract_json(text) or {}
            it["answer"] = str(got.get("answer") or text).strip() if isinstance(got, dict) \
                else text.strip()
            for c in (got.get("concerns") or []) if isinstance(got, dict) else []:
                if c in CONCERNS:
                    why = str(got.get("why") or "").strip()
                    _flag(it, "concern", f"the checker finds it {c}" + (f": {why}" if why else ""))
            to_judge.append(it)
        else:
            it["answer"] = text.strip()
            ok, why = ev.grade(it["q"], it["answer"])
            if ok is False:
                _flag(it, "checker", f"the checker's answer failed: {why}")
            elif ok is None:
                to_judge.append(it)
    for it in to_judge:
        it["judge_pending"] = True
    if d["status"] == "cancelled":
        return len(meta["ids"])
    if to_judge:
        _judge(d, to_judge)
    else:
        _finalize(d)
    return len(meta["ids"])


def _task(d: dict) -> str:
    _scripts()
    import categories
    return "exam_" + categories.topic_slug(d["spec"]["topic"])


def full_reference(q: dict) -> str:
    """the reference the judge reads, and the bank keeps: the answer, then
    what a full answer contains — the question's own criteria"""
    return q["reference"].strip() + "\n\nA full answer:\n" + "\n".join(
        f"- {c}" for c in q["criteria"])


def _judge_request(d: dict, it: dict) -> tuple[str, int]:
    _scripts()
    if d["kind"] == "everyday":
        import everyday as ev
        return ev.judge_prompt(it["q"], it["answer"]), 200
    import judge
    rub = judge.rubric_for(_task(d))
    ref = full_reference(it["q"])
    if rub.criteria:
        return judge.build_criteria_prompt(rub.text, rub.criteria, it["q"]["question"], ref,
                                           it["answer"]), 700
    return judge.build_prompt(rub.text, it["q"]["question"], ref, it["answer"]), 200


def _judge(d: dict, todo: list[dict]) -> None:
    _scripts()
    import judge
    if judge.is_stub():
        # the stand-in grades at once, as a judged run's stub does
        _read_marks(d, {it["n"]: _stub_reply(d, it) for it in todo})
        _finalize(d)
        return
    reqs = []
    for it in todo:
        user, cap = _judge_request(d, it)
        reqs.append(llm.Request(f"qbj:{d['id']}:{it['n']}", "", user, max_tokens=cap,
                                json=True, meta={"kind": "qb_judge", "draft_kind": d["kind"]}))
    _submit(d, "judge", "judge", reqs)
    d["status"] = "checking"


def _stub_reply(d: dict, it: dict) -> str:
    import judge
    user, _ = _judge_request(d, it)
    if d["kind"] == "everyday":
        import everyday as ev
        return ev.stub_reply(user)
    return judge.StubGrader.reply(user)


def _finish_judge(d: dict, meta: dict, results: dict) -> int:
    texts = {}
    for cid in meta["ids"]:
        res = results.get(cid)
        texts[int(cid.rsplit(":", 1)[1])] = "" if res is None or getattr(res, "error", "") \
            else getattr(res, "text", "")
    _read_marks(d, texts)
    if d["status"] != "cancelled":
        _finalize(d)
    return len(texts)


def _short(text: str, words: int = 25) -> str:
    w = (text or "").split()
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


def _read_marks(d: dict, texts: dict[int, str]) -> None:
    _scripts()
    by_n = {it["n"]: it for it in d["items"]}
    for n, text in texts.items():
        it = by_n.get(n)
        if it is None:
            continue
        it.pop("judge_pending", None)
        if d["kind"] == "everyday":
            import everyday as ev
            v = ev.parse_verdict(text) if text else None
            if v is None:
                _flag(it, "checker", "the judge gave no verdict on the checker's answer")
            elif not v["pass"]:
                _flag(it, "checker", f"the checker's answer failed: {v['reason']}")
            continue
        import judge
        spec = judge.rubric_for(_task(d)).criteria
        mark = None
        if text:
            if spec:
                g = judge.parse_grade_criteria(text, spec)
                mark = None if g is None else judge.fold(g.criteria, g.flags, spec)
            else:
                mark, _ = judge.parse_grade(text)
        it["mark"] = mark
        if mark is None:
            _flag(it, "checker", "the judge gave no mark to the checker's answer")
        elif mark < 3:
            _flag(it, "checker", f"the checker answered “{_short(it['answer'])}” ({mark}/4); "
                  f"the criteria expect: {'; '.join(it['q']['criteria'])}")


def _flag(it: dict, kind: str, text: str, **extra) -> None:
    it["flags"].append({"kind": kind, "text": text, **extra})
    if it["verdict"] in ("accept", "edit"):
        # reviewed in Try 10, flagged since: it is reviewed again
        it["earlier"] = it["verdict"]
        it["verdict"] = None


# ---------------------------------------------------------------------------
# duplicates: this bank, earlier batches, and each other
# ---------------------------------------------------------------------------

def _grams(text: str) -> set[int]:
    toks = contamination.normalize(text)
    if len(toks) < contamination.NGRAM:
        return {hash(" ".join(toks))} if toks else set()
    return {hash(w) for w in contamination.windows(toks)}


def _others(d: dict) -> list[dict]:
    """what a new question may duplicate: {src, id, label, text}"""
    _scripts()
    out = []
    if d["kind"] == "knowledge":
        import exam_build as eb
        for r in eb.load_bank(config.EXAM_DIR).get(d["spec"]["topic"], []) \
                if config.EXAM_DIR.is_dir() else []:
            out.append({"src": "bank", "id": r["qid"], "label": f"bank question {r['qid'][:8]}",
                        "text": r["prompt"]})
    else:
        import everyday as ev
        for q in ev.load_bank():
            out.append({"src": "bank", "id": q["id"], "label": q["id"], "text": q["prompt"]})
    for other in db.qb_list():
        if other["id"] == d["id"] or other["kind"] != d["kind"] or other.get("published"):
            continue
        for it in other["items"]:
            if not it["auto"] and it["verdict"] != "reject":
                out.append({"src": "earlier", "id": f"{other['id']}:{it['n']}",
                            "label": f"#{it['n']} of an earlier batch",
                            "text": _text(other, it["q"])})
    return out


def _embeddings(texts: list[str]) -> list[list[float]] | None:
    """cached by model and text, so the bank is embedded once"""
    p = config.BENCH_ROOT / "builder" / "embeddings.json"
    cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    key = lambda t: _sha(config.OPENROUTER_EMBED_MODEL + "\0" + t)   # noqa: E731
    todo = sorted({t for t in texts if key(t) not in cache})
    if todo:
        got = ai_models.embed(todo)
        if got is None:
            return None
        cache.update({key(t): v for t, v in zip(todo, got)})
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[key(t)] for t in texts]


def _dedup(d: dict) -> None:
    live = [it for it in d["items"] if not it["auto"] and it["verdict"] != "reject"]
    if not live:
        return
    others = _others(d)
    new_texts = [_text(d, it["q"]) for it in live]
    vecs = _embeddings(new_texts + [o["text"] for o in others]) if ai_models.has_key() else None
    d["dedup_how"] = "13-gram and embeddings" if vecs else "13-gram"
    grams_o = [_grams(o["text"]) for o in others]
    grams_n = [_grams(t) for t in new_texts]
    for i, it in enumerate(live):
        if any(f["kind"] == "dup" for f in it["flags"]):
            continue
        best = None
        # the bank and earlier batches, then the questions before it here
        cands = [(o, grams_o[j], vecs[len(live) + j] if vecs else None)
                 for j, o in enumerate(others)]
        cands += [({"src": "batch", "id": str(o["n"]), "label": f"#{o['n']}",
                    "text": new_texts[j]}, grams_n[j], vecs[j] if vecs else None)
                  for j, o in enumerate(live[:i])]
        for o, g, v in cands:
            how = ""
            if grams_n[i] & g:
                how = "13 words in a row" if len(contamination.normalize(o["text"])) \
                    >= contamination.NGRAM else "the same words"
            elif vecs is not None:
                cos = ai_models.cosine(vecs[i], v)
                if cos >= config.QB_DUP_COSINE:
                    how = f"cosine {cos:.2f}"
            if how:
                best = (o, how)
                break
        if best:
            o, how = best
            _flag(it, "dup", f"looks like {o['label']}", other=o, how=how)


def _finalize(d: dict) -> None:
    """checked: duplicates, a tenth of the rest drawn at random, then review"""
    if d["spec"].get("dedup", True):
        _dedup(d)
    pool = [it for it in d["items"] if not it["auto"] and not it["flags"]
            and it["verdict"] is None and not it["sample"]]
    k = min(len(pool), max(1, round(SAMPLE_SHARE * len(pool)))) if pool else 0
    for it in random.Random(d["id"]).sample(pool, k):
        it["sample"] = True
    d["status"] = "review"


# ---------------------------------------------------------------------------
# masein's side: review, the rest, cancel, publish
# ---------------------------------------------------------------------------

def _item(d: dict, n: int) -> dict:
    it = next((x for x in d["items"] if x["n"] == n), None)
    if it is None:
        raise ValueError(f"no question #{n} in this draft")
    return it


def _edit(d: dict, it: dict, edited: dict) -> None:
    q = dict(it["q"])
    for k in (("question", "reference", "criteria", "notes") if d["kind"] == "knowledge"
              else ("prompt", "reference", "checks", "notes")):
        if k in edited:
            q[k] = edited[k]
    if d["kind"] == "knowledge":
        q["criteria"] = [str(c).strip() for c in q["criteria"] if str(c).strip()]
        _, why = _question(d, q)
    else:
        _, why = _question(d, q)
    if why:
        raise ValueError(f"the edit can't stand: {why}")
    it.setdefault("original", it["q"])
    it["q"], it["edited"] = q, True


def review(draft_id: str, n: int, verdict: str, by: str, reason: str = "",
           edited: dict | None = None) -> dict:
    if verdict not in ("accept", "edit", "reject"):
        raise ValueError("accept, edit or reject")
    if reason and reason not in REASONS:
        raise ValueError(f"a reason is one of: {', '.join(REASONS)}")
    with _lock:
        d = get(draft_id)
        it = _item(d, n)
        if it["auto"]:
            raise ValueError(f"#{n} was set aside before review: {it['auto']}")
        if verdict == "edit":
            _edit(d, it, edited or {})
        it.update(verdict=verdict, reason=reason, by=by, at=time.time())
        return _put(d)


def resolve_dup(draft_id: str, n: int, keep: str, by: str) -> dict:
    """keep new, keep old, or keep both — the old one is another question of
    this batch; one already in the bank, or in an earlier batch, stays"""
    with _lock:
        d = get(draft_id)
        it = _item(d, n)
        f = next((f for f in it["flags"] if f["kind"] == "dup"), None)
        if f is None:
            raise ValueError(f"#{n} is not flagged as a duplicate")
        other = f["other"]
        if keep == "old":
            it.update(verdict="reject", reason="", dup="old", by=by, at=time.time())
        elif keep == "new":
            if other["src"] != "batch":
                raise ValueError("the other one is already in the bank or an earlier batch: "
                                 "keep old or keep both")
            o = _item(d, int(other["id"]))
            o.update(verdict="reject", reason="", dup=f"#{n} kept", by=by, at=time.time())
            it.update(verdict="accept", dup="new", by=by, at=time.time())
        elif keep == "both":
            it.update(verdict="accept", dup="both", by=by, at=time.time())
        else:
            raise ValueError("keep new, old or both")
        f["resolved"] = keep
        return _put(d)


def reviewed_try(d: dict) -> int:
    return sum(1 for it in d["items"] if it["stage"] == "try" and not it["auto"]
               and it["verdict"])


def can_rest(d: dict) -> str:
    """'' when "Make the rest" can start, else why not"""
    if d["stage"] != "try" or d["status"] not in ("review", "failed", "cancelled"):
        return "the first ten are still being written" if d["status"] == "writing" else \
            "the rest is already under way"
    live = [it for it in d["items"] if not it["auto"]]
    need = min(MIN_REVIEWED, len(live))
    if reviewed_try(d) < need:
        return f"review {need - reviewed_try(d)} more of the first ten first"
    return ""


def feedback(d: dict) -> str:
    """the first ten's reviews as a short list for the rest: avoid, do more of"""
    avoid, more = [], []
    for it in d["items"]:
        if it["stage"] != "try" or it["auto"]:
            continue
        text = _short(_text(d, it.get("original") or it["q"]), 30)
        if it["verdict"] == "reject":
            avoid.append(f"- {it['reason'] or 'rejected'}: “{text}”")
        elif it["verdict"] == "edit":
            new = _short(_text(d, it["q"]), 30)
            more.append(f"- like this, as it was edited: “{text}” became “{new}”"
                        if new != text else f"- like this, with its answer or checks fixed: “{text}”")
        elif it["verdict"] == "accept" and it["reason"] == "":
            more.append(f"- like this: “{text}”")
    if not avoid and not more:
        return ""
    out = ["## What the reviewer said about the first ten"]
    if avoid:
        out += ["Avoid questions like these:"] + avoid[:8]
    if more:
        out += ["Do more of these:"] + more[:5]
    return "\n".join(out)


def rest(draft_id: str, by: str) -> dict:
    with _lock:
        d = get(draft_id)
        why = can_rest(d)
        if why:
            raise ValueError(why)
        why = blocked("checker", d.get("checker_o"))
        if why:
            raise ValueError(f"the checker can't run: {why} — choose one on AI models, or "
                             "for this batch in step 1")
        d.update(stage="rest", feedback=feedback(d), rest_by=by, error="")
        left = d["spec"]["count"] - len(d["items"])
        if left > 0:
            _write(d, min(TRY_N, left))
        else:
            _check(d)
        return _put(d)


def cancel(draft_id: str, by: str) -> dict:
    with _lock:
        d = get(draft_id)
        if d["status"] not in ("writing", "checking"):
            raise ValueError("nothing is running in this draft")
        d.update(status="cancelled", cancelled_by=by)
        return _put(d)


def resume(draft_id: str, by: str) -> dict:
    """carry on where a cancelled or failed draft stopped"""
    with _lock:
        d = get(draft_id)
        if d["status"] not in ("cancelled", "failed"):
            raise ValueError("this draft is not stopped")
        d.update(error="", empty_writes=0)
        if d["stage"] == "try":
            left = min(TRY_N, d["spec"]["count"]) - len(d["items"])
            if left > 0:
                _write(d, left)
            else:
                d["status"] = "review"
        else:
            left = d["spec"]["count"] - len(d["items"])
            if left > 0:
                _write(d, min(TRY_N, left))
            else:
                _check(d)
        return _put(d)


def to_review(d: dict) -> list[dict]:
    """what must be cleared before Publish: every flagged question, and the
    sample of the rest"""
    return [it for it in d["items"] if not it["auto"] and it["verdict"] is None
            and (it["flags"] or it["sample"])]


def publishable(d: dict) -> list[dict]:
    """ready to publish: checked, not rejected, and not waiting for review"""
    waiting = {it["n"] for it in to_review(d)}
    return [it for it in d["items"] if not it["auto"] and it["verdict"] != "reject"
            and it["answer"] is not None and it["n"] not in waiting]


def can_publish(d: dict) -> str:
    if d.get("published"):
        return "published already"
    if d["stage"] != "rest" or d["status"] != "review":
        return "the questions are still being written or checked"
    left = len(to_review(d))
    if left:
        return f"{left} still to review"
    if not publishable(d):
        return "no question is left to publish"
    return ""


def progress(d: dict) -> dict:
    items = d["items"]
    flagged = sum(1 for it in items if not it["auto"] and it["flags"])
    line = (f"{len(items)} of {d['spec']['count']} written · {flagged} flagged"
            if d["stage"] == "rest" else f"{len(items)} of {min(TRY_N, d['spec']['count'])} written")
    return {"written": len(items), "count": d["spec"]["count"], "flagged": flagged,
            "set_aside": sum(1 for it in items if it["auto"]),
            "checked": sum(1 for it in items if it["answer"] is not None),
            "to_review": len(to_review(d)), "reviewed_try": reviewed_try(d),
            "publishable": len(publishable(d)), "line": line}


def view(d: dict) -> dict:
    """the draft as the page reads it"""
    out = {k: v for k, v in d.items() if k not in ("prompts", "writer_o", "checker_o")}
    if d["kind"] == "everyday":
        # each check in plain words, as the Everyday page shows them
        _scripts()
        import everyday as ev
        out["items"] = [{**it, "checks_words": [_describe(ev, c) for c in it["q"]["checks"]]}
                        for it in d["items"]]
    out["progress"] = progress(d)
    out["can_rest"] = can_rest(d)
    out["can_publish"] = can_publish(d)
    out["locked"] = default_prompt(d["kind"])["locked"]
    return out


def _describe(ev, c) -> str:
    try:
        return ev.describe(c)
    except Exception:                     # noqa: BLE001 — a check set aside as unreadable
        return json.dumps(c)


def publish(draft_id: str, by: str) -> dict:
    """the new bank version: every question kept, with who wrote it, who
    checked it and who approved it, the batch and the prompt's hash"""
    _scripts()
    with _lock:
        d = get(draft_id)
        why = can_publish(d)
        if why:
            raise ValueError(why)
        rows = publishable(d)
        stamp = {"written_by": d["writer"]["label"] + (f" ({d['writer']['id']})"
                                                       if d["writer"]["id"] else ""),
                 "checked_by": d["checker"]["label"] + (f" ({d['checker']['id']})"
                                                        if d["checker"]["id"] else ""),
                 "approved_by": by, "batch_id": d["id"]}
        if d["kind"] == "knowledge":
            out = _publish_exam(d, rows, stamp)
        else:
            out = _publish_everyday(d, rows, stamp)
        d["published"] = {**out, "by": by, "at": time.time(), "n": len(rows)}
        d["status"] = "published"
        _put(d)
        return d["published"]


def _publish_exam(d: dict, rows: list[dict], stamp: dict) -> dict:
    import exam_build as eb
    topic, added = d["spec"]["topic"], 0
    have = eb.bank_qids(config.EXAM_DIR)
    for it in rows:
        q = it["q"]
        qid = eb.qid_of(q["question"])
        if qid in have:
            continue
        eb.append_bank(config.EXAM_DIR, {
            "qid": qid, "topic": topic, "prompt": q["question"], "reference": full_reference(q),
            "notes": q["notes"], "source": "question builder", "drafted_by": stamp["written_by"],
            **stamp, "accepted_by": stamp["approved_by"], "accepted_at": time.time(),
            "prompt_sha256": it["prompt_sha"], "edited": bool(it["edited"]),
            "meta": {"criteria": q["criteria"], "subtopic": q["subtopic"], "level": q["level"]}})
        db.curation_add(f"build:{d['id']}:{it['n']}", topic, qid, "accept",
                        stamp["approved_by"], bool(it["edited"]))
        have.add(qid)
        added += 1
    return {"kind": "knowledge", "topic": topic, "added": added}


def _publish_everyday(d: dict, rows: list[dict], stamp: dict) -> dict:
    import everyday as ev
    s = d["spec"]
    before = ev.version()["hash"]
    dest = ev.built_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if s.get("new_group"):
        gp = ev.built_dir() / "groups.json"
        extra = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
        extra[s["group"]] = s["group_label"]
        gp.write_text(json.dumps(extra, indent=1), encoding="utf-8")
    ids = {q["id"] for q in ev.load_bank()}
    lines = []
    for k, it in enumerate(rows, 1):
        q = it["q"]
        qid = f"everyday-{s['group']}-b{d['id'][:6]}-{k:02d}"
        if qid in ids:
            continue
        lines.append(json.dumps({"id": qid, **{f: q[f] for f in (
            "group", "skill", "difficulty", "prompt", "reference", "checks", "notes")},
            **stamp, "prompt_sha256": it["prompt_sha"], "edited": bool(it["edited"])},
            ensure_ascii=False) + "\n")
    with open(dest, "a", encoding="utf-8") as fh:
        fh.writelines(lines)
    ev.load_bank()                      # a bad row fails here, not on a model's answer
    return {"kind": "everyday", "group": s["group"], "added": len(lines),
            "version": ev.version()["hash"], "before": before}
