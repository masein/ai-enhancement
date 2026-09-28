"""12n.2: SimpleQA Verified — 1,000 short factual questions (Google DeepMind,
2025; MIT), a cleaned-up SimpleQA, pinned in eval_tasks/simpleqa
(manifest.json: the source, its revision, the file's sha256 and the licence).

Asked through the chat template with the Everyday settings, as the question
reads; each answer graded by the board's judge with the dataset's own grader
template (grader_template.txt, as Inspect Evals carries it): correct,
incorrect or not attempted.

The score is Epoch AI's: the share of all the questions answered correctly.
"Not attempted" is a number of its own — a model that says it doesn't know is
honest, not wrong, and the page says so.

A Standard benchmark, shared with the frontier: never in the Avg, never a
training target, never in Improve.

    python scripts/simpleqa.py <results>/<model>     marks again, writes simpleqa.json
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as _judge  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "eval_tasks" / "simpleqa"
CSV_NAME = "simpleqa_verified.csv"
TEMPLATE = DATA_DIR / "_simpleqa_template_yaml"
GRADER = DATA_DIR / "grader_template.txt"
TASK = "simpleqa_verified"
OUT_NAME = "simpleqa.json"
WAITING = "waiting for the judge"
NEVER_FINISHED = "the model never finished its answer"
NOTHING = "the model wrote nothing"
# the grader's letters, as its instructions ask for them
GRADES = {"A": "correct", "B": "incorrect", "C": "not_attempted"}
WORDS = {"correct": "correct", "incorrect": "incorrect", "not_attempted": "not attempted"}
# the page's words for why abstaining is kept apart
HONEST = ("A model that says it doesn't know is honest, not wrong: not attempted is counted "
          "apart, and only a correct answer scores")


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

def manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def data_sha() -> str:
    return manifest()["sources"]["simpleqa_verified"]["files"][CSV_NAME]


def credit() -> dict:
    s = manifest()["sources"]["simpleqa_verified"]
    return {k: s[k] for k in ("name", "licence", "cite", "url", "revision")}


def load() -> list[dict]:
    """the questions, as committed: {id, prompt, answer, topic, answer_type}"""
    text = (DATA_DIR / CSV_NAME).read_text(encoding="utf-8")
    return [{"id": f"sqv-{int(r['original_index']):04d}", "prompt": r["problem"].strip(),
             "answer": r["answer"].strip(), "topic": r.get("topic") or "",
             "answer_type": r.get("answer_type") or ""}
            for r in csv.DictReader(io.StringIO(text))]


def build_tasks(dest: Path) -> Path:
    """the task under `dest`: the questions (never their answers — the harness
    has no use for them) and the yaml, with their absolute path and the
    Everyday settings filled in. Returns the directory for --include_path"""
    from everyday import run_settings              # 12d.1: the settings the Playground shows too
    dest.mkdir(parents=True, exist_ok=True)
    s = run_settings(None)
    items = dest / f"{TASK}.jsonl"
    stamp = dest / f"{TASK}.sha256"
    want = data_sha()
    if not (items.exists() and stamp.exists() and stamp.read_text(encoding="utf-8").strip() == want):
        items.write_text("".join(json.dumps({"id": q["id"], "prompt": q["prompt"]},
                                            ensure_ascii=False) + "\n" for q in load()),
                         encoding="utf-8")
        stamp.write_text(want + "\n", encoding="utf-8")
    yaml = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__ITEMS_PATH__", str(items.resolve()))
            .replace("__UNTIL__", json.dumps(s["until"]))
            .replace("__MAX_GEN_TOKS__", str(s["max_gen_toks"]))
            .replace("__DO_SAMPLE__", "true" if s["do_sample"] else "false")
            .replace("__TEMPERATURE__", f"{float(s['temperature'])}"))
    (dest / f"{TASK}.yaml").write_text(f"task: {TASK}\n" + yaml, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# the grader
# ---------------------------------------------------------------------------

def grader_key() -> str:
    """eight hex digits over the grader's words: a grade is kept only while
    the judge gave it on these"""
    return hashlib.sha256(GRADER.read_bytes()).hexdigest()[:8]


def judge_prompt(q: dict, answer: str) -> str:
    """the dataset's grader template, filled in — by replacement, as its text
    has braces of its own"""
    return (GRADER.read_text(encoding="utf-8").replace("{question}", q["prompt"])
            .replace("{criterion}", q["answer"])
            .replace("{answer}", (answer or "").strip() or "(empty)"))


_LETTER = re.compile(r"(?<![A-Za-z])([ABC])(?![A-Za-z])")
_WORD = re.compile(r"\b(NOT[ _]ATTEMPTED|INCORRECT|CORRECT)\b")


def parse_grade(text: str) -> str | None:
    """the judge's reply: "correct", "incorrect" or "not_attempted", or None
    when it can't be read — its letter first, as its instructions ask"""
    t = (text or "").strip()
    m = _LETTER.search(t)
    if m:
        return GRADES[m.group(1)]
    m = _WORD.search(t.upper())
    if m:
        return {"CORRECT": "correct", "INCORRECT": "incorrect"}.get(m.group(1), "not_attempted")
    return None


# the stand-in judge: deterministic, so tests and a server with
# JUDGE_MODEL=stub grade the same way every time. Correct when every word of
# the gold target is in the answer; not attempted when it says it doesn't know
DOESNT_KNOW = re.compile(r"\b(?:I (?:don't|do not) know|I'm not sure|I am not sure|not sure|"
                         r"I (?:can(?:no|')t|cannot) (?:answer|know|find|say)|no information|"
                         r"unable to (?:answer|find|say))\b", re.I)


def _norm(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def stub_grade(q: dict, answer: str) -> str:
    gold, got = _norm(q["answer"]), set(_norm(answer))
    if gold and all(w in got for w in gold):
        return "correct"
    if DOESNT_KNOW.search(answer or ""):
        return "not_attempted"
    return "incorrect"


def stub_reply(prompt: str) -> str:
    """what the fake judge backend answers to a judge_prompt(): the letter"""
    tail = prompt.rsplit("Question: ", 1)[-1]
    question, rest = tail.split("\nGold target: ", 1)
    gold, rest = rest.split("\nPredicted answer: ", 1)
    answer = rest.split("\n'''", 1)[0]
    g = stub_grade({"prompt": question, "answer": gold}, answer)
    return {v: k for k, v in GRADES.items()}[g]


# ---------------------------------------------------------------------------
# the answers, graded
# ---------------------------------------------------------------------------

def _stamp(f: Path) -> str:
    m = re.search(r"_(\d{4}-\d{2}-\d{2}T[^/]*?)\.jsonl$", f.name)
    return m.group(1) if m else ""


def answers(model_dir: Path) -> dict[str, dict]:
    """{id: the harness's record} from the task's samples, newest last"""
    dirs = [d for d in model_dir.glob(f"{TASK}_*shot") if re.fullmatch(rf"{TASK}_\d+shot", d.name)]
    files = sorted((f for d in dirs for f in d.rglob(f"samples_{TASK}_*.jsonl")), key=_stamp)
    out: dict[str, dict] = {}
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                doc = rec.get("doc") or {}
                if doc.get("id"):
                    out[doc["id"]] = rec
    return out


def read(model_dir: Path) -> dict | None:
    try:
        out = json.loads((model_dir / OUT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def write(model_dir: Path, out: dict) -> Path:
    p = model_dir / OUT_NAME
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def _model_id(model_dir: Path) -> str:
    try:
        return json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))["model"]
    except (OSError, ValueError, KeyError):
        return model_dir.name.replace("__", "/", 1)


def _waits(it: dict) -> bool:
    return it["grade"] is None and not it.get("unjudged")


def _share(items: list[dict], hit) -> dict:
    p = sum(1 for it in items if hit(it)) / len(items)
    return {"n": sum(1 for it in items if hit(it)), "of": len(items), "rate": round(p, 6),
            "se": round(math.sqrt(p * (1 - p) / len(items)), 6)}


def rates(items: list[dict]) -> dict | None:
    """Epoch's number — the share of the questions answered correctly — and
    the shares not attempted and incorrect. An answer the model never
    finished, or left empty, answered nothing: not attempted. One the judge
    couldn't grade is left out (its failure, not the model's). No rates until
    every answer is graded: a share of the first half is another number"""
    items = [it for it in items if not it.get("unjudged")]
    if not items or any(_waits(it) for it in items):
        return None
    return {"correct": _share(items, lambda it: it["grade"] == "correct"),
            "not_attempted": _share(items, lambda it: it["grade"] == "not_attempted"),
            "incorrect": _share(items, lambda it: it["grade"] == "incorrect")}


def mark(model_dir: Path, verdicts: dict[str, str] | None = None,
         judge: dict | None = None) -> dict | None:
    """Every answer, graded: the judge's grade when it has one on this grader
    for this answer, else waiting. None when the harness logged no answers"""
    verdicts = verdicts or {}
    got = answers(model_dir)
    if not got:
        return None
    prev = read(model_dir) or {}
    before = {it["id"]: it for it in prev.get("items") or []}
    key = grader_key()
    items = []
    for q in load():
        rec = got.get(q["id"])
        if rec is None:
            continue
        parts = _judge.answer_parts(rec)
        ans = parts["answer_text"]
        it = {"id": q["id"], "answer_text": ans, "had_reasoning": parts["had_reasoning"]}
        if parts["no_answer"] or not ans.strip():
            it.update(grade="not_attempted", unmarked=True,
                      reason=NEVER_FINISHED if parts["no_answer"] else NOTHING)
            items.append(it)
            continue
        g = verdicts.get(q["id"])
        old = before.get(q["id"]) or {}
        if g is None and old.get("grade") and not old.get("unmarked") \
                and old.get("answer_text") == ans and old.get("grader") == key:
            g = old["grade"]
        if g is None:
            it.update(grade=None, reason=WAITING)
        else:
            it.update(grade=g, reason=WORDS[g], grader=key)
        items.append(it)
    return {"model": _model_id(model_dir), "marked_at": time.time(), "grader": key,
            "asked": {"n": len(got), "of": len(load()), "data": data_sha()},
            "judge": judge if judge is not None else prev.get("judge"),
            "items": items, "rates": rates(items),
            "waiting": sum(1 for it in items if _waits(it))}


def summary(out: dict | None) -> str:
    """the queue row's line: "SimpleQA Verified 14.2% correct · 31% not attempted" """
    if not out:
        return "no answers to grade"
    if out.get("waiting"):
        return f"{len(out['items']) - out['waiting']} of {len(out['items'])} graded · {WAITING}"
    r = out.get("rates") or {}
    if not r:
        return "graded"
    return (f"SimpleQA Verified {100 * r['correct']['rate']:.1f}% correct · "
            f"{100 * r['not_attempted']['rate']:.0f}% not attempted")


def _pending(out: dict) -> list[dict]:
    return [it for it in out["items"] if _waits(it)]


def start(model_dir: Path, submission: int | None = None) -> dict:
    """Grade now; send the judge what waits on it, as one batch. Called by the
    runner straight after the answers. Returns simpleqa.json's content, plus
    `batch_id` when a batch went out"""
    from service import db, llm
    out = mark(model_dir)
    if out is None:
        raise RuntimeError("the harness logged no answers to SimpleQA Verified")
    todo = _pending(out)
    if not todo:
        write(model_dir, out)
        return out
    qs = {q["id"]: q for q in load()}
    ident = _judge.identity()
    if _judge.is_stub():
        out = mark(model_dir, {it["id"]: stub_grade(qs[it["id"]], it["answer_text"])
                               for it in todo},
                   judge={"id": ident["id"], "version": _judge.version(ident)["key"],
                          "provisional": False})
        write(model_dir, out)
        return out
    why = _judge.blocked()
    if why:
        for it in _pending(out):
            it["reason"] = "not graded: the judge is not set up on this server"
        write(model_dir, out)
        return out
    try:
        backend = llm.client("judge")
        stamp = llm.provisional(backend, "graded")
        reqs = [llm.Request(custom_id=f"simpleqa:{submission or 0}:{it['id']}", system="",
                            json=False, max_tokens=20,
                            user=judge_prompt(qs[it["id"]], it["answer_text"]),
                            meta={"kind": "simpleqa", "id": it["id"]})
                for it in todo]
        bid = backend.submit(reqs)
    except Exception as e:                          # noqa: BLE001 — the answers are kept
        for it in _pending(out):
            it["reason"] = "not graded: the judge could not be reached"
        write(model_dir, out)
        out["error"] = str(e)
        return out
    out["judge"] = {"id": ident["id"], "version": _judge.version(ident)["key"],
                    "provisional": bool(stamp), "batch_id": bid}
    write(model_dir, out)
    if submission:
        db.update(submission, judge_batch=bid)
        db.batch_add(bid, "simpleqa", submission, len(reqs), backend.name, backend.model)
        db.batch_progress(bid, f"0/{len(reqs)} done")
    out["batch_id"] = bid
    return out


def finish(model_dir: Path, results: dict) -> dict | None:
    """The poller's half: the judge's replies in, simpleqa.json out"""
    verdicts, unread = {}, {}
    for cid, res in results.items():
        if not str(cid).startswith("simpleqa:"):
            continue
        qid = str(cid).rsplit(":", 1)[-1]
        g = None if getattr(res, "error", None) else parse_grade(getattr(res, "text", ""))
        if g is None:
            unread[qid] = "not graded: the judge's reply could not be read"
        else:
            verdicts[qid] = g
    prev = read(model_dir) or {}
    out = mark(model_dir, verdicts, judge=prev.get("judge"))
    if out is None:
        return None
    for it in out["items"]:
        if it["id"] in unread and _waits(it):
            it.update(reason=unread[it["id"]], unjudged=True)
    out["rates"] = rates(out["items"])
    out["waiting"] = sum(1 for it in out["items"] if _waits(it))
    write(model_dir, out)
    return out


def judge_failed(model_dir: Path, why: str) -> None:
    """the batch failed: every answer that waited on it says so, and waits no more"""
    out = read(model_dir)
    if not out:
        return
    for it in out.get("items") or []:
        if _waits(it):
            it.update(reason=f"not graded: the judge's batch failed ({why[:120]})", unjudged=True)
    out["rates"] = rates(out["items"])
    out["waiting"] = sum(1 for it in out["items"] if _waits(it))
    write(model_dir, out)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    d = Path(argv[0])
    out = start(d)
    print(f"{d.name}: {summary(out)}"
          + (f" · judge batch {out['batch_id']}" if out.get("batch_id") else ""))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))                   # service/, for the judge
    raise SystemExit(main(sys.argv[1:]))
