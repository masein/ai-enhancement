#!/usr/bin/env python3
"""The Everyday tasks pilot (brief 12a): mark what a model answered.

Everyday tasks asks what people type into an assistant on a phone — short,
lowercase, typos, one plain request — and most answers can be marked by a
script, so the score needs no judge. The pilot is five questions, all of
them readable. It is a look, not a benchmark: never ranked, never averaged
into anything, never on the Leaderboard, never read by Propose, never sent
to a generator. Its questions are not split into practice and hidden
halves; each item may carry a `split` later, and nothing here reads one yet.

    python scripts/everyday.py results/full            re-mark every model
    python scripts/everyday.py results/full -m org/x   one model

It reads the generations the harness logged, marks each one on the text
after the reasoning block (judge.answer_parts, #59's split — never the raw
generation), and writes everyday.json beside the model's results. Each
check says pass or fail and one reason in plain words, because the reason
is what the page shows. Question 03 is the one the judge marks, against the
rubric the question carries; re-marking keeps its verdict while the answer is
the same one it read. English only (2026-09-24).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import judge as _judge  # noqa: E402

TASK = "everyday_pilot"
PILOT_DIR = REPO / "eval_tasks" / "everyday"
PILOT_PATH = PILOT_DIR / "pilot.jsonl"
TEMPLATE_PATH = PILOT_DIR / "_everyday_template_yaml"
OUT_NAME = "everyday.json"
# the pilot's five groups, as the page shows them (12b.3: English only —
# Language became Summarising, and Behaviour is Instructions)
GROUPS = {"understanding": "Understanding", "writing": "Writing", "transform": "Transform",
          "summarising": "Summarising", "instructions": "Instructions"}
NEVER_FINISHED = "never finished answering"
WAITING = "waiting for the judge"


def load_pilot(path: Path = PILOT_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# the checks: (pass, reason). The reason is what a person reads on the page.
# ---------------------------------------------------------------------------

def _has(text: str, s: str) -> bool:
    """`s` as a whole word or number: "29" is in "29 days", not in "1929"."""
    return re.search(rf"(?<!\w){re.escape(s)}(?!\w)", text, re.I) is not None


def check_contains(answer: str, spec: dict) -> tuple[bool, str]:
    want = list(spec.get("any") or [])
    hit = next((s for s in want if _has(answer, s)), None)
    return (True, f"says {hit}") if hit else (False, f"didn't say {want[0]}")


_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)```", re.S)


def _first_object(text: str) -> tuple[str, int, int] | None:
    """The first balanced {…} in `text`, and where it sits. Braces inside a
    JSON string do not count."""
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], start, i + 1
        start = text.find("{", start + 1)
    return None


def _values(o) -> list[str]:
    """Every value in the object, flattened, in order — key names are free."""
    if isinstance(o, dict):
        return [v for x in o.values() for v in _values(x)]
    if isinstance(o, list):
        return [v for x in o for v in _values(x)]
    return [] if o is None else [str(o)]


# what the JSON must hold: (the name the reason uses, any of these phrases)
JSON_NEEDS = (("Sara Ahmed", (("sara ahmed",), ("sara", "ahmed"))),
              ("34", (("34",),)),
              ("product manager", (("product manager",),)),
              ("Dubai", (("dubai",),)),
              ("March 2021", (("march 2021",), ("2021-03",))))


def check_json(answer: str, spec: dict | None = None) -> tuple[bool, str]:
    fenced = next((m for m in _FENCE.finditer(answer) if "{" in m.group(1)), None)
    where = fenced.group(1) if fenced else answer
    got = _first_object(where)
    if got is None:
        return False, "not valid JSON"
    blob, a, b = got
    try:
        obj = json.loads(blob)
    except ValueError:
        return False, "not valid JSON"
    flat = " ".join(_values(obj)).lower()
    missing = [name for name, ways in JSON_NEEDS
               if not any(all(_has(flat, w) for w in way) for way in ways)]
    if missing:
        return False, "missing: " + ", ".join(missing)
    # text around it does not fail it, but it is said
    outside = (answer[:fenced.start()] + answer[fenced.end():]) if fenced \
        else answer[:a] + answer[b:]
    return True, ("valid JSON, all five values" if not outside.strip()
                  else "valid JSON, all five values, with text around it")


# (what must be there, any of these) and what must not be
FIXED_NEEDS = (("I am writing", ("i am writing", "i'm writing", "i’m writing")),
               ("regarding", ("regarding", "in regard to", "with regard to", "about")),
               ("sent", ("sent",)),
               ("paid", ("paid",)))
FIXED_WRONG = ("I writing", "sended", "payed")


def check_fixed(answer: str, spec: dict | None = None) -> tuple[bool, str]:
    left = [w for w in FIXED_WRONG if _has(answer, w)]
    if left:
        return False, "still says " + ", ".join(f"'{w}'" for w in left)
    gone = [name for name, ways in FIXED_NEEDS if not any(_has(answer, w) for w in ways)]
    if gone:
        return False, "doesn't say " + ", ".join(f"'{w}'" for w in gone)
    return True, "all four mistakes fixed"


_MARKER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•·])\s*")
_WRAP = re.compile(r"^(?:\*\*|__|[\"'“”‘’`*_])+|(?:\*\*|__|[\"'“”‘’`*_])+$")
# a colon, or a dash with words after it: the name is being explained. A
# hyphen inside a word ("Brew-Ha") is part of the name
_EXPLAINS = re.compile(r":|\s[-–—]+\s*\w|[–—]\s*\w")


def _clean_line(line: str) -> str:
    s = _MARKER.sub("", line.strip())
    prev = None
    while prev != s:
        prev, s = s, _WRAP.sub("", s).strip()
    return s


def check_lines(answer: str, spec: dict | None = None) -> tuple[bool, str]:
    spec = spec or {}
    n, most = int(spec.get("n", 3)), int(spec.get("max_words", 5))
    lines = [_clean_line(x) for x in answer.splitlines() if x.strip()]
    lines = [x for x in lines if x]
    if len(lines) != n:
        return False, f"{len(lines)} line{'' if len(lines) == 1 else 's'}, expected {n}"
    for i, x in enumerate(lines, 1):
        if _EXPLAINS.search(x):
            return False, f"line {i} explains the name"
        if len(x.split()) > most:
            return False, f"line {i} is {len(x.split())} words"
    return True, "three names, nothing else"


CHECKS = {"contains": check_contains, "json": check_json, "fixed": check_fixed,
          "lines": check_lines}


# ---------------------------------------------------------------------------
# the one question the judge marks
# ---------------------------------------------------------------------------

# 12b.3: the question carries its rubric; nothing here knows what it asks
JUDGE_PROMPT = """You are checking ONE answer from an assistant against a rubric. Read the question, the rubric and the answer, and decide whether the answer passes the rubric. Reply with one JSON object and nothing else: {{"pass": <true or false>, "reason": <one short sentence for a person, about the answer>}}.

QUESTION
{question}

RUBRIC
{rubric}

THE ANSWER
{answer}"""


def judge_prompt(item: dict, answer: str) -> str:
    return JUDGE_PROMPT.format(question=item["prompt"], rubric=item["check"]["rubric"],
                               answer=answer.strip() or "(empty)")


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def stub_verdict(answer: str) -> dict:
    """The deterministic stand-in the tests and dry runs use, for the one
    question the judge marks (03, the school notice): at most two sentences,
    11:30, and Thursday."""
    a = (answer or "").strip()
    if not a:
        return {"pass": False, "reason": "no answer"}
    if "11:30" not in a:
        return {"pass": False, "reason": "it doesn't give the 11:30 closing time"}
    if "thursday" not in a.lower():
        return {"pass": False, "reason": "it doesn't say Thursday"}
    if _sentences(a) > 2:
        return {"pass": False, "reason": "more than two sentences"}
    return {"pass": True, "reason": "closes 11:30 on Thursday, in two sentences or fewer"}


def stub_reply(prompt: str) -> str:
    """What the fake judge backend answers to a judge_prompt()."""
    answer = prompt.rsplit("THE ANSWER\n", 1)[-1]
    return json.dumps(stub_verdict(answer), ensure_ascii=False)


def parse_verdict(text: str) -> dict | None:
    from service import llm
    obj = llm.extract_json(text or "")
    if not isinstance(obj, dict) or not isinstance(obj.get("pass"), bool):
        return None
    reason = " ".join(str(obj.get("reason") or "").split())[:200]
    return {"pass": obj["pass"],
            "reason": reason or ("the judge says it is right" if obj["pass"]
                                 else "the judge says it is wrong")}


# ---------------------------------------------------------------------------
# marking
# ---------------------------------------------------------------------------

def records(model_dir: Path) -> dict[str, dict]:
    """The harness's logged samples, by question id."""
    out = {}
    for rec in _judge._records(model_dir, TASK):
        qid = (rec.get("doc") or {}).get("id")
        if qid:
            out[qid] = rec
    return out


def read(model_dir: Path) -> dict | None:
    try:
        return json.loads((model_dir / OUT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_id(model_dir: Path) -> str:
    try:
        return json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))["model"]
    except (OSError, ValueError, KeyError):
        return model_dir.name.replace("__", "/", 1)


def mark(model_dir: Path, verdicts: dict[str, dict] | None = None,
         judge: dict | None = None) -> dict | None:
    """Mark every pilot question the model answered, and return what
    everyday.json holds — None when the harness logged no pilot answers.
    `verdicts`: {question id: {pass, reason}} from the judge. A verdict
    already on file is kept while the answer it read is unchanged."""
    recs = records(model_dir)
    if not recs:
        return None
    prev = read(model_dir) or {}
    before = {it["id"]: it for it in prev.get("items") or []}
    verdicts = verdicts or {}
    items = []
    for q in load_pilot():
        ctype = q["check"]["type"]
        rec = recs.get(q["id"])
        it = {"id": q["id"], "group": q["group"], "check": ctype}
        if rec is None:
            items.append({**it, "pass": False, "reason": "no answer was recorded",
                          "answer_text": "", "had_reasoning": False})
            continue
        parts = _judge.answer_parts(rec)
        ans = parts["answer_text"]
        it.update(answer_text=ans, had_reasoning=parts["had_reasoning"])
        if parts["had_reasoning"]:
            it.update(reasoning_text=parts["reasoning_text"],
                      reasoning_words=_judge.words(parts["reasoning_text"]))
        if parts["no_answer"]:
            it.update({"pass": False, "reason": NEVER_FINISHED, "no_answer": True})
        elif not ans.strip():
            it.update({"pass": False, "reason": "the model wrote nothing"})
        elif ctype == "judge":
            v = verdicts.get(q["id"])
            old = before.get(q["id"]) or {}
            if v is None and old.get("pass") is not None and old.get("answer_text") == ans:
                v = {"pass": old["pass"], "reason": old["reason"]}
            if v is None:
                it.update({"pass": None, "reason": old.get("reason") if (
                    old.get("answer_text") == ans and old.get("pass") is None
                    and old.get("reason")) else WAITING})
            else:
                it.update({"pass": bool(v["pass"]), "reason": v["reason"]})
        else:
            ok, why = CHECKS[ctype](ans, q["check"])
            it.update({"pass": ok, "reason": why})
        items.append(it)
    gen = _judge._generation(model_dir, TASK) or {}
    out = {
        "model": prev.get("model") or _model_id(model_dir),
        "task": TASK,
        "marked_at": time.time(),
        # what the answers were generated with: the chat template always,
        # greedy, and the budget (512, or a reasoning model's 2,048)
        "settings": {"chat_template": True, "greedy": True, **gen},
        "passed": sum(1 for it in items if it["pass"] is True),
        "total": len(items),
        "waiting": sum(1 for it in items if it["pass"] is None),
        "items": items,
    }
    if judge or prev.get("judge"):
        out["judge"] = judge or prev["judge"]
    return out


def write(model_dir: Path, out: dict) -> Path:
    p = model_dir / OUT_NAME
    p.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return p


def summary(out: dict) -> str:
    """The queue row's words."""
    line = f"Everyday pilot: {out['passed']} of {out['total']}"
    if out.get("waiting"):
        line += f" · the judge is marking {out['waiting']}"
    return line


# ---------------------------------------------------------------------------
# the service's side: the task the harness runs, and the judge's one request
# ---------------------------------------------------------------------------

def build_task(dest: Path) -> Path:
    """The pilot as a harness task under `dest`: the items, and the yaml with
    their absolute path filled in. Returns the directory for --include_path."""
    dest.mkdir(parents=True, exist_ok=True)
    items = dest / f"{TASK}.jsonl"
    shutil.copyfile(PILOT_PATH, items)
    yaml = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__ITEMS_PATH__",
                                                             str(items.resolve()))
    (dest / f"{TASK}.yaml").write_text(f"task: {TASK}\n" + yaml, encoding="utf-8")
    return dest


def _pending(out: dict) -> list[dict]:
    """The questions that wait on the judge, with an answer to send it."""
    qs = {q["id"]: q for q in load_pilot()}
    return [qs[it["id"]] for it in out["items"] if it["pass"] is None and it["answer_text"]]


def start(model_dir: Path, submission: int | None = None) -> dict:
    """Mark now; send the judge its one question. Called by the runner
    straight after generation, inside the same run. Returns everyday.json's
    content plus `batch_id` when a judge batch went out."""
    from service import config, db, llm
    out = mark(model_dir)
    if out is None:
        raise RuntimeError("the harness logged no answers for the pilot")
    todo = _pending(out)
    if not todo:
        write(model_dir, out)
        return out
    answers = {it["id"]: it["answer_text"] for it in out["items"]}
    ident = _judge.identity()
    if config.JUDGE_MODEL == "stub":
        out = mark(model_dir, {q["id"]: stub_verdict(answers[q["id"]]) for q in todo},
                   judge={"id": ident["id"], "provisional": False})
        write(model_dir, out)
        return out
    why = _judge.blocked()
    if why:
        for it in out["items"]:
            if it["pass"] is None:
                it["reason"] = "not marked: the judge is not set up on this server"
        write(model_dir, out)
        return out
    try:
        backend = llm.client("judge")
        stamp = llm.provisional(backend, "marked")
        reqs = [llm.Request(custom_id=f"everyday:{submission or 0}:{q['id']}", system="",
                            json=True, max_tokens=200,
                            user=judge_prompt(q, answers[q["id"]]),
                            meta={"kind": "everyday", "id": q["id"]})
                for q in todo]
        bid = backend.submit(reqs)
    except Exception as e:                          # noqa: BLE001 — the answers are marked
        for it in out["items"]:
            if it["pass"] is None:
                it["reason"] = "not marked: the judge could not be reached"
        write(model_dir, out)
        out["error"] = str(e)
        return out
    out["judge"] = {"id": ident["id"], "provisional": bool(stamp), "batch_id": bid}
    write(model_dir, out)
    if submission:
        db.update(submission, judge_batch=bid)
        db.batch_add(bid, "everyday", submission, len(reqs), backend.name, backend.model)
        # the queue reads "grading 0/1" from the moment it goes out
        db.batch_progress(bid, f"0/{len(reqs)} done")
    out["batch_id"] = bid
    return out


def finish(model_dir: Path, results: dict) -> dict | None:
    """The poller's half: the judge's replies in, everyday.json out."""
    verdicts = {}
    for cid, res in results.items():
        if not str(cid).startswith("everyday:"):
            continue
        qid = str(cid).rsplit(":", 1)[-1]
        v = None if getattr(res, "error", None) else parse_verdict(getattr(res, "text", ""))
        verdicts[qid] = v or {"pass": None, "reason": "not marked: the judge's reply could "
                                                      "not be read"}
    prev = read(model_dir) or {}
    out = mark(model_dir, {k: v for k, v in verdicts.items() if v["pass"] is not None},
               judge=prev.get("judge"))
    if out is None:
        return None
    for it in out["items"]:
        v = verdicts.get(it["id"])
        if it["pass"] is None and v is not None:
            it["reason"] = v["reason"]
    out["waiting"] = 0
    write(model_dir, out)
    return out


def judge_failed(model_dir: Path, why: str) -> None:
    """The judge's batch failed: the question says so instead of waiting."""
    out = read(model_dir)
    if not out:
        return
    for it in out["items"]:
        if it["pass"] is None:
            it["reason"] = "not marked: the judge failed — run the pilot again"
    out["waiting"] = 0
    out["judge_error"] = why[:300]
    write(model_dir, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("-m", "--model", action="append", default=[])
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    want = {m.replace("/", "__") for m in a.model}
    n = 0
    for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        out = mark(d)
        if out is None:
            continue
        write(d, out)
        n += 1
        print(f"{out['model']}: {summary(out)}")
    print(f"marked {n} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
