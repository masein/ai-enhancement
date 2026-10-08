"""17: the Frontier benchmarks, asked of a served model — on the board, from
Test this model, and on a rented box, where scripts/remote_gguf.py serves a
GGUF with llama-server and runs the board's own runner. The same code asks
both: the prompts and scorers of scripts/frontier.py, a model's card's
sampling for the thinking setting asked, a seed a question and run, and the
thinking switch said out loud (16b's rule) — through served.answer_one, the
path every served answer takes.

Answers are kept as they land, one line each, in
results/<row>/<task>_0shot/frontier/answers.jsonl: a run stopped — the
process killed, the box stopped — carries on from the next unanswered
question when it is started again. A shard (FRONTIER_SHARD i/n) asks every
n-th question from the i-th.

17b: an answer is only an answer. A question the server failed on (an error
of its own, twice, and again without its chat parsing) is never kept: the
next run asks it again, and more failures than served.item_error_limit stop
the run. Answers made with another setup — another file or build on the
server, another launch, budget, sampling or thinking switch — are never mixed
with the new ones: they are set aside when the run starts, and asked again.
A thinking row with no thinking in any answer, or a thinking-off row with
thinking in its answers, is not scored: the server didn't do what was asked.

When every question of every run is answered, the task is scored by code and
written in lm_eval's layout (results_*.json, samples_*.jsonl beside the
answers), so the board reads its cell as it reads any other: the share right
over all runs, with its error over questions. A rented box doesn't score
(FRONTIER_SCORE_AFTER_RUN=0): the server scores what it imports.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, db, served

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import frontier as fb  # noqa: E402

ANSWERS = "answers.jsonl"
SETUP = "setup.json"
SUB = "frontier"                     # the answers' folder in the task's


def task_dir(row: Path, task: str) -> Path:
    return Path(row) / f"{task}_0shot" / SUB


def read_answers(path: Path) -> dict[tuple[str, int], dict]:
    """every answer kept, by (question, run) — a line a crash cut short is
    asked again"""
    out: dict[tuple[str, int], dict] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
            # 17b: a line the server failed on (an earlier run's) isn't an answer
            if r.get("error"):
                continue
            out[(str(r["id"]), int(r["epoch"]))] = r
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return out


def family_of(rec: dict) -> str:
    return fb.family(rec.get("id") or "", rec.get("name") or "", rec.get("based_on") or "",
                     (rec.get("pin") or {}).get("file") or "",
                     (rec.get("pin") or {}).get("model") or "")


def settings(rec: dict, task: str, on: bool) -> dict:
    """how each question of `task` is asked: its budget, the card's sampling,
    and the thinking switch, said out loud (not to a model from OpenRouter)"""
    spec = fb.BENCH[task]
    s = {"max_tokens": spec["budget"]["on" if on else "off"],
         **fb.sampling(family_of(rec), on)}
    if not served.is_openrouter(rec):
        s["chat_template_kwargs"] = {"enable_thinking": bool(on)}
    return s


def shard() -> tuple[int, int] | None:
    return fb.parse_shard(config.FRONTIER_SHARD)


def _where() -> str:
    return config.FRONTIER_WHERE or "this server"


def _setup(rec: dict, task: str, on: bool, s: dict) -> dict:
    spec = fb.BENCH[task]
    sh = shard()
    pin = rec.get("pin") or {}
    return {
        "version": fb.VERSION, "task": task, "protocol": spec["protocol"],
        "protocol_version": spec["protocol_version"], "source": spec["source"],
        "epochs": spec["epochs"], "thinking": "on" if on else "off",
        "budget": s["max_tokens"], "family": family_of(rec) or "server defaults",
        "sampling": {k: v for k, v in s.items() if k not in ("max_tokens",
                                                            "chat_template_kwargs")},
        "where": _where(), "file": pin.get("file") or "",
        # 17b: what its server serves, and how it was launched
        "server": {"file": pin.get("file") or "", "size": pin.get("size"),
                   "build": pin.get("build") or ""},
        "launch": {"flags": rec.get("flags") or "", "env": rec.get("env") or ""},
        # 17c: the launch as the import compares it — routing and speculative
        # decoding, read from the flags, the environment and "How it's
        # served" — so reordered flags, slots and context don't count
        "launch_setup": launch_setup(rec),
        "shard": {"i": sh[0], "n": sh[1]} if sh else None}


def launch_setup(rec: dict) -> dict:
    """the registered launch, normalised as the import normalises a box's —
    17d: and the flags that change what the server answers (its cache types,
    its chat template, its reasoning budget and format, rope and yarn), the
    board's own resume only; `drafts` None when the server's /slots didn't say"""
    import import_frontier as imf
    got = imf.record_launch(rec)
    sp = rec.get("speculative")
    return {"env": dict(sorted(got["env"].items())), "spec": sorted(got["spec"]),
            "drafts": None if sp is None else bool(sp),
            "answers": imf.answer_flags(rec)}


# 17b: what a task's answers depend on — a change asks them all again. 17c:
# the launch as the import compares it, not its text
SETUP_KEYS = ("version", "protocol_version", "source", "epochs", "thinking", "budget",
              "sampling", "family", "server", "launch_setup")


def setup_differs(old: dict, new: dict) -> list[str]:
    """the parts of a task's setup that changed, by name — 17c: a part the old
    setup.json doesn't hold (one written before it was kept) is unknown, not
    changed (setup_unknown). 17d: the launch part by part, each compared
    only when both sides say it (a /slots probe that failed isn't a change)"""
    out = []
    for k in SETUP_KEYS:
        if k not in old:
            continue
        if k == "launch_setup" and isinstance(old[k], dict) and isinstance(new.get(k), dict):
            if any(old[k][p] != new[k].get(p) for p in old[k]
                   if p in new[k] and old[k][p] is not None and new[k][p] is not None):
                out.append(k)
        elif k == "version" and old.get(k) in fb.ASKED_AS and new.get(k) in fb.ASKED_AS:
            continue                    # 18c: asked the same way; only the reading changed
        elif old.get(k) != new.get(k):
            out.append(k)
    return out


def setup_unknown(old: dict) -> list[str]:
    """the parts an earlier setup.json doesn't say — kept as unknown, and said"""
    return [k for k in SETUP_KEYS if k not in old] + list(old.get("unknown_earlier") or [])


def _read_json(p: Path) -> dict:
    try:
        got = json.loads(Path(p).read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def set_aside(row: Path, task: str, why: str) -> Path | None:
    """a task's answers (and their grades and scores) moved out of the row, to
    results/earlier/<row>/, as an import sets them aside"""
    d = task_dir(row, task)
    if not d.exists():
        return None
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    aside = (Path(config.OUT_DIR).with_name("earlier") / Path(row).name
             / f"{task}_0shot-frontier-{why}-{stamp}")
    # 17d: two set aside within one second each keep their own folder
    k = 1
    while aside.exists():
        k += 1
        aside = aside.with_name(f"{task}_0shot-frontier-{why}-{stamp}-{k}")
    aside.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(aside))
    return aside


# 17d: a run's first answers checked for thinking, on the box as on the board
# — a server that ignores the switch stops in words after these many, not
# three hours later at the import
EARLY = 20


def early_thinking(task: str, on: bool, answers: list[str]) -> str:
    """'' unless the first EARLY answers say the server isn't doing what was
    asked: a thinking run none of whose answers thought, or an off run most
    of whose did"""
    label = fb.BENCH[task]["label"]
    n = sum(1 for a in answers if thought(a))
    if on and not n:
        return (f"{label}: thinking was asked for, and none of the first {len(answers)} answers "
                "holds any — the server isn't thinking (a --reasoning-budget 0, or a chat "
                "template that ignores the switch?): stopped")
    if not on and n > len(answers) / 2:
        return (f"{label}: thinking was off, and {n} of the first {len(answers)} answers hold "
                "thinking — the server thinks anyway: stopped")
    return ""


def ask_task(rec: dict, task: str, row: Path, on: bool, progress=None,
             canceled=lambda: False, log=lambda line: None,
             sid: int | None = None) -> tuple[int, int]:
    """every question of `task` this run (or shard) holds and hasn't answered
    — (answered, of) when it stops. Raises served.ServerStopped when the
    server stops answering: what it answered before is kept"""
    fb.set_root(config.BENCH_ROOT)
    spec = fb.BENCH[task]
    items = fb.shard_of(fb.load(task, config.BENCH_ROOT), shard())
    d = task_dir(row, task)
    s = settings(rec, task, on)
    new = _setup(rec, task, on, s)
    # 17b: answers made with another setup are never mixed with these. 17c:
    # what the earlier setup.json doesn't say is unknown — the answers kept,
    # and said, here and in setup.json from now on
    old = _read_json(d / SETUP)
    if (d / ANSWERS).exists() and read_answers(d / ANSWERS):
        changed = setup_differs(old, new) if old else []
        unknown = sorted(set(setup_unknown(old))) if old else list(SETUP_KEYS)
        if changed:
            aside = set_aside(row, task, "another-setup")
            log(f"[frontier] {task}: its answers were made with another setup "
                f"({', '.join(changed)}): set aside at {aside}, and asked again")
        elif unknown:
            new["unknown_earlier"] = unknown
            log(f"[frontier] {task}: its earlier answers' setup.json "
                + ("is missing or unreadable" if not old else
                   f"doesn't say their {', '.join(unknown)}")
                + ": they are kept, with those unknown")
    d.mkdir(parents=True, exist_ok=True)
    (d / SETUP).write_text(json.dumps(new, indent=1, sort_keys=True), encoding="utf-8")
    path = d / ANSWERS
    done = read_answers(path)
    if config.FRONTIER_ASK_WRITTEN_OFF:
        # 17f: asked again, what earlier runs wrote off (--ask-written-off)
        again = [k for k, r in done.items() if r.get("unanswered")]
        for k in again:
            done.pop(k)
        if again:
            _failed_runs(d, clear=again)
            log(f"[frontier] {task}: {len(again)} written off before, asked again")
    want = [(it, e) for it in items for e in range(spec["epochs"])]
    todo = [(it, e) for it, e in want if (it["id"], e) not in done]
    total, have = len(want), len(want) - len(todo)
    lock = threading.Lock()
    halt: list[Exception] = []
    failed: dict[tuple[str, int], str] = {}
    limit = served.item_error_limit(len(items))
    t0, n0 = time.time(), have
    count = {"n": have}
    early = {"checked": False}

    def check_early() -> str:
        """the first EARLY answers' thinking, once there are that many"""
        if early["checked"]:
            return ""
        got = [r["answer"] for r in read_answers(path).values() if not r.get("unanswered")]
        if len(got) < EARLY:
            return ""
        early["checked"] = True
        return early_thinking(task, on, got[:EARLY])

    why = check_early()                         # a resume: checked before anything is asked
    if why:
        raise served.ServerStopped(have, total, why, refused=why)

    def write(line: dict) -> int:
        if sid:
            line = {**line, "run": sid}             # 17h: the run that asked it
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            count["n"] += 1
            return count["n"]

    def one(job) -> None:
        it, e = job
        if halt or canceled():
            return
        text, _ = fb.prompt(task, it)
        seed = fb.seed_of(task, it["id"], e)
        # HLE's system message, CAIS's
        si = {**s, "seed": seed, **({"system": fb.system_of(task)} if fb.system_of(task) else {})}
        try:
            a = served.answer_one(rec, text, si)
        except served.ServerStopped as x:
            with lock:
                halt.append(x)
            return
        except ValueError as x:                 # the server refused this request outright
            with lock:
                halt.append(served.ServerStopped(0, 0, str(x), refused=(
                    f"the server refused a question ({status_of(x)})")))
            return
        if a.error:
            # 17b: no answer either way (its chat endpoint twice, then without
            # its chat parsing) isn't an answer — and more questions than the
            # limit failing, the server isn't right: stopped, nothing of
            # theirs kept. 17c: the limit counts questions, not runs (a
            # question's eight runs are one question)
            with lock:
                failed[(it["id"], e)] = a.error.get("chat") or ""
                qs = {k[0] for k in failed}
                if len(qs) > limit and not halt:
                    halt.append(served.ServerStopped(0, total, a.error.get("chat") or "", refused=(
                        f"the server failed on {len(qs)} questions, asked its own way and "
                        f"without its chat parsing ({status_of(a.error.get('chat'))}): "
                        "stopped")))
            return
        line = {"id": it["id"], "epoch": e, "seed": seed, "answer": str(a),
                "finish": getattr(a, "finish", None), "tokens": a.tokens,
                "at": round(time.time(), 3)}
        if a.fallback:
            line["fallback"] = a.fallback
        n = write(line)
        if n >= EARLY and not early["checked"]:
            with lock:
                why = check_early()
                if why and not halt:
                    halt.append(served.ServerStopped(0, total, why, refused=why))
        if progress:
            progress(n, total, (time.time() - t0) / max(1, n - n0))

    def stop_if_halted() -> None:
        if halt:
            e = halt[0]
            e.done, e.total = count["n"], total
            raise e

    with ThreadPoolExecutor(max_workers=served.concurrency(rec)) as pool:
        list(pool.map(one, todo))
    stop_if_halted()
    if failed and not canceled():
        # 17c: each one the server failed on is asked once more; failing
        # again, it is written as no answer — counted wrong, named in the log
        # and on the row — and the run carries on to the next benchmark
        again = [(it, e) for it, e in todo if (it["id"], e) in failed]
        failed.clear()
        with ThreadPoolExecutor(max_workers=served.concurrency(rec)) as pool:
            list(pool.map(one, again))
        stop_if_halted()
        if failed and count["n"] == 0:
            # nothing at all answered, here or before: the server, not the
            # questions. 17e: a resume that answered nothing, with answers
            # from before, asks the server one of those first (below) — a
            # benchmark whose last question truly fails stopped here on every
            # resume
            why = next(iter(failed.values()))
            raise served.ServerStopped(count["n"], total, why, refused=(
                f"the server failed on every question it was asked ({len(failed)}), asked its "
                f"own way and without its chat parsing ({status_of(why)}): stopped"))
        # 17d: before a question is written off, the server must still answer
        # one it answered before: a server that went down near the end of a
        # benchmark left its last questions as no answer for good
        if failed and not still_answers(rec, task, path, s, failed):
            why = next(iter(failed.values()))
            raise served.ServerStopped(count["n"], total, why, refused=(
                f"the server stopped answering ({why}): {len(failed)} question(s) it failed on "
                "are asked again by the next run, nothing written for them"))
        # 17f: written off only when the refusal is the question's own (a 4xx —
        # too long for its context — or no answer within the timeout), or
        # when it failed on WRITE_OFF_RUNS separate runs. A 5xx on real
        # requests is the server's, whatever the probe says: kept, and asked
        # again by the next run (20 left that all answered 500 were written
        # off for good while the probe answered)
        runs_failed = _failed_runs(d, add=[k for k, why in failed.items() if not _own(why)])
        kept = 0
        for (qid, e), why in sorted(failed.items()):
            n_runs = runs_failed.get(gkey(qid, e), 0)
            if not _own(why) and n_runs < WRITE_OFF_RUNS:
                kept += 1
                continue
            write({"id": qid, "epoch": e, "seed": fb.seed_of(task, qid, e), "answer": "",
                   "finish": None, "tokens": None, "unanswered": why[:300] or "no reason given",
                   "at": round(time.time(), 3)})
            log(f"[frontier] {task}: question {qid}, run {e}: the server failed on it twice"
                + (f", on {n_runs} separate runs" if not _own(why) else "")
                + f" — written as no answer, counted wrong ({status_of(why)})")   # 18c point 16
        if kept:
            log(f"[frontier] {task}: {kept} question(s) the server failed on with an error that "
                f"isn't theirs (a 5xx) — kept, and asked again by the next run; written off "
                f"after {WRITE_OFF_RUNS} runs")
    return count["n"], total


WRITE_OFF_RUNS = 3
FAILED_RUNS = "failed_runs.json"


def _own(why: str) -> bool:
    """17f: the refusal is the question's own — a 4xx (its prompt and budget
    too long for the context, a request the server can't take), or no answer
    within the timeout from a server that is up — never a 5xx"""
    m = re.match(r"HTTP (\d{3})", why or "")
    if m:
        return 400 <= int(m.group(1)) < 500 and int(m.group(1)) not in (401, 403, 408, 429)
    return "no answer within" in (why or "")


def _failed_runs(d: Path, add: list | None = None, clear: list | None = None) -> dict:
    """17f: how many separate runs each question failed on with a 5xx —
    counted once a run, cleared once it is asked again on purpose"""
    p = d / FAILED_RUNS
    try:
        got = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        got = {}
    for k in clear or []:
        got.pop(gkey(*k), None)
    for k in add or []:
        got[gkey(*k)] = int(got.get(gkey(*k), 0)) + 1
    if add or clear:
        p.write_text(json.dumps(got), encoding="utf-8")
    return got


def still_answers(rec: dict, task: str, path: Path, s: dict, failed: dict) -> bool:
    """17d: the server answers again a question it answered before (a few
    tokens of it) — or, with none answered here, its /health"""
    answered = [r for k, r in read_answers(path).items()
                if k not in failed and not r.get("unanswered")]
    if not answered:
        return served.healthy(rec)
    r = answered[0]
    it = next((x for x in fb.load(task, config.BENCH_ROOT) if x["id"] == r["id"]), None)
    if it is None:
        return served.healthy(rec)
    text, _ = fb.prompt(task, it)
    si = {**s, "max_tokens": 16, "seed": fb.seed_of(task, it["id"], r.get("epoch", 0)),
          **({"system": fb.system_of(task)} if fb.system_of(task) else {})}
    try:
        return not served.answer_one(rec, text, si).error
    except (served.ServerStopped, ValueError):
        return False


# ---------------------------------------------------------------------------
# 17b: the parity check — the same questions on the server and on a box
# ---------------------------------------------------------------------------

def parity_ask(rec: dict, path: Path, progress=None, identity: dict | None = None,
               n: int | None = None, twice: bool = False) -> int:
    """fb.PARITY's questions asked of `rec`, greedy and thinking off, as many at
    a time as it takes — each reply, its letter and the key, to `path`. Raises
    served.ServerStopped when the server stops answering or fails on one.
    17c: the file's first line is `identity` — which side, which file, which
    launch (scripts/frontier_parity.py) — so compare can refuse two sides
    that differ. 17d: `n` questions (PARITY's 500 by default); `twice`, each
    asked a second time (`answer2`) — the box's agreement with itself, which
    compare reports beside its agreement with the server"""
    fb.set_root(config.BENCH_ROOT)
    items = fb.parity_items(config.BENCH_ROOT, n)
    task = fb.PARITY["task"]
    s = {"max_tokens": fb.PARITY["max_tokens"], **fb.PARITY["sampling"]}
    if not served.is_openrouter(rec):
        s["chat_template_kwargs"] = {"enable_thinking": False}
    got: dict[str, dict] = {}
    lock = threading.Lock()
    total = len(items) * (2 if twice else 1)
    done = {"n": 0}

    def ask(it: dict) -> "served.Answer":
        text, _ = fb.prompt(task, it)
        a = served.answer_one(rec, text, s)
        if a.error:
            raise served.ServerStopped(0, len(items), a.error.get("chat") or "", refused=(
                f"the server failed on parity question {it['id']}: {a.error.get('chat')}"))
        with lock:
            done["n"] += 1
            k = done["n"]
        if progress:
            progress(k, total)
        return a

    def one(it: dict) -> None:
        a = ask(it)
        _, need = fb.prompt(task, it)
        with lock:
            got[it["id"]] = {"id": it["id"], "key": need["key"], "answer": str(a),
                             "read": fb.read_mmlu_pro(str(a)), "finish": a.finish,
                             "tokens": a.tokens}

    def again(it: dict) -> None:
        a = ask(it)
        with lock:
            got[it["id"]].update(answer2=str(a), read2=fb.read_mmlu_pro(str(a)))
    with ThreadPoolExecutor(max_workers=served.concurrency(rec)) as pool:
        list(pool.map(one, items))
    if twice:
        with ThreadPoolExecutor(max_workers=served.concurrency(rec)) as pool:
            list(pool.map(again, items))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    head = {**(identity or {}), "n": len(items), "seed": fb.PARITY["seed"], "twice": twice}
    Path(path).write_text(json.dumps({"parity_of": head}, ensure_ascii=False) + "\n"
                          + "".join(json.dumps(got[it["id"]], ensure_ascii=False) + "\n"
                                    for it in items), encoding="utf-8")
    return len(got)


# ---------------------------------------------------------------------------
# scoring: by code, once every question of every run is in — and (stage 3)
# by a grader, or code with Epoch's model check as a second look
# ---------------------------------------------------------------------------

GRADES = "grades.json"
# 17c: an answer whose grader's reply was no grade this many times is ungraded:
# never sent again, counted wrong, and said (service/frontier_grade.py)
GRADE_TRIES = 3
# 17d: a benchmark with more than this share of what its grader was sent left
# ungraded has no score: the grader isn't answering in its form
UNGRADED_SHARE = 0.05


def read_grades(d: Path) -> dict:
    """what a grader (or Epoch's model check) said of each answer, by
    "<question>#<run>", and who said it: {grader: {...}, items: {...}}"""
    try:
        g = json.loads((Path(d) / GRADES).read_text(encoding="utf-8"))
        return g if isinstance(g, dict) else {}
    except (OSError, ValueError):
        return {}


def gkey(qid: str, epoch: int) -> str:
    return f"{qid}#{epoch}"


def answer_sha(answer: str) -> str:
    """17b: the answer a grade was given for — a grade counts only while the
    answer it graded is the one on file"""
    return hashlib.sha256((answer or "").encode("utf-8")).hexdigest()


def marks(row: Path, task: str, items: list[dict] | None = None,
          grades: dict | None = None) -> dict:
    """every answer of the task's, scored: {runs: {question: [run]}, missing:
    [(question, run)], items}. A run: {epoch, ok, code_ok, read, ran_out,
    answer, finish, tokens, error, grade} — `ok` the score the page uses: the
    code's; for a benchmark Epoch checks with a model (`look`), the code's when
    it says right and the check's when it says wrong or can't read (None
    until the check has run); for a graded one, its grader's (None until
    graded). An answer that ran out of room is wrong, and never sent to a
    grader or a check"""
    fb.set_root(config.BENCH_ROOT)
    spec = fb.BENCH[task]
    items = items if items is not None else fb.load(task, config.BENCH_ROOT)
    d = task_dir(row, task)
    # 17g: the grades as Start would leave them (the dry run's), or on file
    gr = read_grades(d) if grades is None else grades
    got, grades = read_answers(d / ANSWERS), (gr.get("items") or {})
    refused = gr.get("refused") or {}
    runs: dict[str, list[dict]] = {}
    missing = []
    for it in items:
        _, need = fb.prompt(task, it)
        for e in range(spec["epochs"]):
            a = got.get((it["id"], e))
            if a is None:
                missing.append((it["id"], e))
                continue
            sc = fb.score(task, a.get("answer") or "", a.get("finish"), need)
            g = grades.get(gkey(it["id"], e))
            if g is not None and g.get("answer_sha256") != answer_sha(a.get("answer") or ""):
                g = None                    # 17b: graded another answer: graded again
            ok = sc["ok"]
            # 17c: no grade after GRADE_TRIES tries, for this answer: ungraded
            no = refused.get(gkey(it["id"], e)) or {}
            ungraded = (g is None and int(no.get("tries") or 0) >= GRADE_TRIES
                        and no.get("answer_sha256") == answer_sha(a.get("answer") or ""))
            if sc["ran_out"] or a.get("unanswered"):
                ok = False                  # 17c: no answer from the server: wrong, never graded
            elif ungraded and (spec.get("grader") or (spec.get("look") and not sc["ok"])):
                ok = False                  # 17c: counted wrong, said, never sent again
            elif spec.get("grader"):
                ok = None if g is None else bool(g.get("ok"))
            elif spec.get("look") and not sc["ok"]:
                # MATH: an answer nothing can be read from is wrong, never asked
                # (Epoch's scorer); OTIS's extractor reads those too
                ok = (False if spec["look"] == "equivalent" and sc["read"] is None
                      else None if g is None else bool(g.get("ok")))
            runs.setdefault(it["id"], []).append({
                "epoch": e, "ok": ok, "code_ok": sc["ok"], "read": sc["read"],
                "ran_out": sc["ran_out"], "answer": a.get("answer") or "",
                "finish": a.get("finish"), "tokens": a.get("tokens"),
                "error": a.get("error"), "unanswered": a.get("unanswered"),
                "ungraded": bool(ungraded and ok is False and not sc["ok"]), "grade": g})
    return {"runs": runs, "missing": missing, "items": items}


def to_grade(row: Path, task: str, grades: dict | None = None) -> list[dict]:
    """stage 3: the answers a grader (or Epoch's model check) is still to see —
    every answer of a graded benchmark, and for one Epoch checks with a model,
    those the code marks wrong or can't read; never one that ran out of room.
    17g: `grades` as Start would leave them, for the dry run"""
    spec = fb.BENCH[task]
    if not (spec.get("grader") or spec.get("look")):
        return []
    m = marks(row, task, grades=grades)
    if m["missing"]:
        return []
    out = []
    for it in m["items"]:
        for r in m["runs"].get(it["id"], []):
            if r["grade"] is None and r["ok"] is None and not r["ran_out"] \
                    and not r.get("ungraded"):
                out.append({"id": it["id"], "epoch": r["epoch"], "answer": r["answer"],
                            "read": r["read"]})
    return out


def _share(per: dict[str, list[float]], task: str, items: list[dict]) -> dict:
    """the benchmark's score from each question's runs: its own aggregate"""
    groups = {it["id"]: fb.group_of(task, it) for it in items}
    return fb.summary(per, task, groups)


# 17c: an off row is scored with at most this share of its answers holding
# thinking — each scored on what follows it, as scoring reads every reply —
# and refused above it. 17j: a quarter, not 1%. On 8 Oct the UD-Q4_K_XL file's
# Humanity's Last Exam run with thinking off was refused for 76 of 2,158
# answers (3.5%): the server did as it was told, and on the hardest questions
# the model opened a thinking block of its own. The share is to catch a server
# that ignores the switch, and only more than a quarter says that
THINKING_OFF_SHARE = 0.25


def status_of(why) -> str:
    """18b: a server's refusal by its status, or our own words for a
    timeout — never its own words, which went out with the exported log"""
    t = str(why or "")
    m = re.search(r"\bHTTP (\d{3})\b", t) or re.search(r"\b([45]\d\d)\b", t)
    if m:
        return f"HTTP {m.group(1)}"
    m = re.search(r"no answer within \d+ s", t)
    return m.group(0) if m else "no status given"


def thought_words(n: int, of: int) -> str:
    """17j: "76 of 2,158 thought anyway, 3.5%" — an off row's answers that
    held thinking, on its score and its run"""
    return f"{n:,} of {of:,} thought anyway, {n / of:.1%}" if of else f"{n:,} thought anyway"


# 18b: above this share of an off row's answers thinking, its score isn't a
# thinking-off score as others publish them — said; THINKING_OFF_SHARE (a
# quarter) stays the refusal line
NOT_COMPARABLE_SHARE = 0.05
NOT_COMPARABLE = "not comparable with thinking-off numbers published elsewhere"


def thought_note(n: int, of: int) -> str:
    """the share with what it means: "76 of 2,158 thought anyway, 3.5%
    (thinking off; scored on what follows the thinking)" — and above 5%, that
    the score isn't comparable"""
    return (thought_words(n, of) + " (thinking off; scored on what follows the thinking)"
            + (f" — {NOT_COMPARABLE}" if of and n / of > NOT_COMPARABLE_SHARE else ""))


def thought(answer: str) -> bool:
    """an answer that holds thinking — 17c: what scoring strips as thinking
    (fb.visible's pattern: everything up to the first </think>, the opening
    tag or not, as a template that opens <think> in the prompt leaves it),
    or a <think> never closed; an empty block, as a template told not to
    think writes, is none"""
    # 18b: every block, wherever it sits — one opened mid-reply was counted as
    # ran out but never in the share
    return bool(fb.thinking_of(answer or ""))


def thinking_refused(task: str, thinking: str | None, answers: list[str]) -> str:
    """'' when the answers are what the thinking setting asked for; else why
    not, in words — a thinking row none of whose answers thought (the server
    was told not to, or its template ignores the switch), or an off row more
    than THINKING_OFF_SHARE of whose answers did"""
    label = fb.BENCH[task]["label"]
    n = sum(1 for a in answers if thought(a))
    if thinking == "on" and answers and not n:
        return (f"{label}: thinking was asked for, and none of its {len(answers):,} answers "
                "holds any — the server didn't think (a --reasoning-budget 0, or a chat "
                "template that ignores the switch?): not scored")
    if thinking == "off" and n > THINKING_OFF_SHARE * len(answers):
        return (f"{label}: thinking was off, and {n:,} of its {len(answers):,} answers hold "
                f"thinking ({n / len(answers):.1%}) — more than a quarter: the server ignored "
                "the thinking switch: not scored")
    return ""


def thinking_kept(thinking: str | None, answers: list[str]) -> int:
    """17c: an off row's answers that hold thinking, at most
    THINKING_OFF_SHARE of them — scored, and said"""
    return sum(1 for a in answers if thought(a)) if thinking == "off" else 0


def stored_score(d: Path, task: str) -> tuple[float | None, str] | None:
    """the score the task's results file holds now, and the version that
    scored it; None when there is none"""
    for p in sorted(d.glob("results_*.json"))[-1:]:
        got = _read_json(p)
        res = (got.get("results") or {}).get(task) or {}
        return res.get("acc,none"), str((got.get("frontier") or {}).get("version") or "")
    return None


def score_words(x: float | None) -> str:
    return "no score" if x is None else f"{100 * x:.1f}%"


def score_task(row: Path, task: str, rec: dict, log=print) -> dict | None:
    """the task's score from its answers, written in lm_eval's layout — None
    while a question of a run is still unanswered. A graded benchmark is
    written once every answer is graded; until then {waiting: n}. One Epoch
    checks with a model is written with the code's score until the check is
    done, and with both after: the page's is Epoch's way. 18c point 12: a
    results file replaced with another score is said ("score X → Y")"""
    spec = fb.BENCH[task]
    m = marks(row, task)
    if m["missing"]:
        return None
    items, runs = m["items"], m["runs"]
    d = task_dir(row, task)
    flat = [r for it in items for r in runs[it["id"]]]
    # 17b: the server did what was asked — thinking on, or off
    thinking = _read_json(d / SETUP).get("thinking")
    why = thinking_refused(task, thinking, [r["answer"] for r in flat])
    if why:
        return {"refused": why}
    held = thinking_kept(thinking, [r["answer"] for r in flat])
    # 17c: the questions the server never answered, counted wrong and named
    never = sorted({q for q, rs in runs.items() for r in rs if r.get("unanswered")})
    out = outcome(spec, flat, d)
    ungraded, waiting = out["ungraded"], out["waiting"]
    ran_out = sum(1 for r in flat if r["ran_out"])
    unread = sum(1 for r in flat if r["read"] is None and not r["ran_out"]
                 and not r.get("unanswered"))
    errors = sum(1 for r in flat if r["error"])
    if out["state"] == "waiting":
        # 17d: answers asked again (another grader chosen) wait: the score made
        # without them is no longer the page's. 18b: the share said here too
        _unwrite(d, task)
        return {"waiting": waiting, "of": len(flat), "label": spec["label"],
                **({"thinking_held": held, "answers": len(flat)} if held else {})}
    if out["state"] == "no score":
        _unwrite(d, task)
        return {"no_score": out["words"], "ungraded": ungraded, "of": out["seen"],
                "label": spec["label"]}
    code = (None if spec.get("grader") else
            _share({q: [1.0 if r["code_ok"] else 0.0 for r in rs] for q, rs in runs.items()},
                   task, items))
    page = (code if waiting else
            _share({q: [1.0 if r["ok"] else 0.0 for r in rs] for q, rs in runs.items()},
                   task, items))
    rows = []
    for k, it in enumerate(items):
        rs = runs[it["id"]]
        _, need = fb.prompt(task, it)
        rows.append({"doc_id": k, "doc": fb.doc_of(task, it),
                     "doc_hash": hashlib.sha256(f"{task}:{it['id']}".encode()).hexdigest(),
                     "arguments": [], "target": fb.target_of(task, need),
                     "resps": [[r["answer"]] for r in rs],
                     "filtered_resps": [r["read"] or "" for r in rs],
                     "acc": sum(1.0 for r in rs if r["ok"]) / len(rs),
                     "frontier": [{k2: v for k2, v in r.items() if k2 != "answer"} for r in rs]})
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S.000000", time.gmtime())
    before = stored_score(d, task)
    for old in [*d.glob("results_*.json"), *d.glob(f"samples_{task}_*.jsonl")]:
        old.unlink()
    changed = before is not None and (before[0] is None) != (page["score"] is None) or (
        before is not None and before[0] is not None and page["score"] is not None
        and abs(before[0] - page["score"]) > 1e-9)
    if changed:
        log(f"[frontier] {Path(row).name} · {spec['label']}: score {score_words(before[0])} → "
            f"{score_words(page['score'])}" + (f" (scored by {before[1]} before, {fb.VERSION} "
                                               "now)" if before[1] != fb.VERSION else ""))
    try:
        setup = json.loads((d / SETUP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        setup = {}
    g, graders = out["grader"], out["graders"]
    looked = sum(1 for r in flat if r["grade"] is not None)
    # 17g: where it ran and the Runs rows it came from, from the row's record
    # of its imports — grading scores again, and dropped what the import marked
    where, came = setup.get("where"), []
    import import_frontier as imf
    if where and where != "this server":
        w, came = imf.where_and_runs(row, task)
        where = w or where
    # 17i: shards imported and held, waiting for the rest — said on the score
    try:
        held_shards = imf.shards_held(row, task)
    except Exception:                                   # noqa: BLE001 — a note, never the score
        held_shards = ""
    detail = {"version": fb.VERSION, "protocol": spec["protocol"], "epochs": spec["epochs"],
              **({"before": {"score": before[0], "version": before[1]}} if changed else {}),
              "questions": len(items), "answers": len(flat),
              "ran_out": ran_out, "unread": unread, "errors": errors,
              # 17c: an off row's few answers that thought anyway, scored
              **({"thinking_held": held} if held else {}),
              **({"ungraded": ungraded} if ungraded else {}),
              **({"unanswered": sum(1 for r in flat if r.get("unanswered")),
                  "unanswered_ids": never[:20]} if never else {}),
              "budget": setup.get("budget"), "sampling": setup.get("sampling"),
              "family": setup.get("family"), "where": where, **({"runs": came} if came else {}),
              **({"shards_held": held_shards} if held_shards else {}),
              "thinking": setup.get("thinking"), "note": spec.get("note") or "",
              "scored_by": ("grader" if spec.get("grader") else "code, then Epoch's model check"
                            if spec.get("look") and not waiting and looked else "code"),
              "code": code, "grader": g,
              # 17b: more than one grader or prompt behind the score: said, not final.
              # 17e: but a second that only graded what the first gave no
              # grade (a top-up) is final, named beside the first
              **({"graders": graders} if len(graders) > 1 and not (g or {}).get("topup")
                 else {}),
              # 17d: and a score with an answer its grader gave no grade isn't final
              **({"final": False} if out["state"] == "not final" else {}),
              **({"look": {"done": looked, "waiting": waiting}} if spec.get("look") else {})}
    res = {"alias": task, "acc,none": page["score"], "acc_stderr,none": page["se"]}
    if spec.get("look") and code:
        res.update({"acc_code,none": code["score"], "acc_code_stderr,none": code["se"]})
    (d / f"results_{stamp}.json").write_text(json.dumps({
        "results": {task: res},
        "group_subtasks": {task: []}, "n-shot": {task: spec.get("shots", 0)},
        "n-samples": {task: {"original": len(items), "effective": len(items)}},
        "higher_is_better": {task: {"acc": True}},
        "configs": {task: {"task": task, "output_type": "generate_until",
                           "num_fewshot": spec.get("shots", 0),
                           "dataset_path": fb.source_name(task),
                           "dataset_kwargs": {"revision": spec["source"]["revision"]}}},
        "config": {"model": "frontier", "model_args": f"pretrained={rec['id']}"
                   + served._row_mark(d.parent)},
        "chat_template": "the server's own", "date": time.time(), "frontier": detail,
        "served": served.view(rec)},
        indent=1), encoding="utf-8")
    (d / f"samples_{task}_{stamp}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {**page, **detail, "epochs": spec["epochs"]}


def outcome(spec: dict, flat: list[dict], d: Path, known: dict | None = None) -> dict:
    """17h: what a task's score is from its marks, decided once — score_task
    writes it, the dry run says it before Start (`known`: the grades as Start
    would leave them). {state: waiting | no score | not final | final,
    words, waiting, ungraded, seen, grader, graders}"""
    ungraded = sum(1 for r in flat if r.get("ungraded"))
    waiting = sum(1 for r in flat if r["ok"] is None)
    seen = sum(1 for r in flat if r["grade"] is not None or r.get("ungraded"))
    g, graders = graders_of(d, flat, known)
    base = {"waiting": waiting, "ungraded": ungraded, "seen": seen, "grader": g,
            "graders": graders}
    if spec.get("grader") and waiting:
        return {**base, "state": "waiting", "words": f"{waiting:,} answers wait for a grade"}
    if ungraded and ungraded > UNGRADED_SHARE * seen:
        return {**base, "state": "no score", "words": (
            f"{spec['label']}: its grader gave no grade on {ungraded:,} of the {seen:,} answers "
            f"it was sent ({_pct(ungraded / seen)}, more than {UNGRADED_SHARE:.0%}) — it isn't "
            "answering in its form: no score; choose another grader on AI models")}
    if len(graders) > 1 and not (g or {}).get("topup"):
        return {**base, "state": "not final",
                "words": f"graded by {len(graders)} graders or prompts: not final"}
    if ungraded:
        return {**base, "state": "not final",
                "words": f"{ungraded:,} its grader gave no grade: not final until another "
                         "grader grades them"}
    return {**base, "state": "final", "words": "final"}


def state_of(row: Path, task: str, grades: dict | None = None) -> dict:
    """17h: `outcome` for a row's task with these grades, nothing written"""
    m = marks(row, task, grades=grades)
    if m["missing"]:
        return {"state": "waiting", "words": "its answers aren't all in"}
    flat = [r for it in m["items"] for r in m["runs"][it["id"]]]
    return outcome(fb.BENCH[task], flat, task_dir(row, task), grades)


def graders_of(d: Path, flat: list[dict], known: dict | None = None
               ) -> tuple[dict | None, list[dict]]:
    """17b: who graded the answers the score counts — (the one grader and
    prompt, or None; every one, each with how many it graded). A grade keeps
    its grader's version and its prompt's sha256. 17e: a second grader that
    graded only answers the first gave no grade, UNGRADED_SHARE of them at
    most, is a top-up — the first is the score's grader, the top-up named
    with it ({..., "topup": {version, prompt_sha256, n}})"""
    used: dict = {}
    after: dict = {}
    for r in flat:
        gr = r.get("grade")
        if gr:
            k = (gr.get("by"), gr.get("prompt_sha256"))
            used[k] = used.get(k, 0) + 1
            was = gr.get("after") or {}
            after.setdefault(k, set()).add((was.get("by"), was.get("prompt_sha256")))
    known = {(x.get("version"), x.get("prompt_sha256")): x
             for x in ((known if known is not None else read_grades(d)).get("graders")
                       or [])}
    every = [{**known.get(k, {"version": k[0], "prompt_sha256": k[1]}), "n": n}
             for k, n in sorted(used.items(), key=lambda kv: -kv[1])]
    if len(every) == 1:
        return {k: v for k, v in every[0].items() if k != "n"}, every
    if len(every) == 2:
        main, top = every
        mk = (main.get("version"), main.get("prompt_sha256"))
        tk = (top.get("version"), top.get("prompt_sha256"))
        if after.get(tk) == {mk} and top["n"] <= UNGRADED_SHARE * (main["n"] + top["n"]):
            return ({**{k: v for k, v in main.items() if k != "n"},
                     "topup": {k: top.get(k) for k in ("version", "prompt_sha256",
                                                       "prompt_words", "n")}}, every)
    return None, every


def _pct(x: float) -> str:
    """17e: a share in words that never rounds onto the line it is set beside —
    51 of 1,000 is 5.1%, not "5%, more than 5%" """
    return f"{x:.0%}" if round(x * 100) != round(UNGRADED_SHARE * 100) else f"{x:.1%}"


def _unwrite(d: Path, task: str) -> None:
    """a task's written score taken away (its answers and grades stay)"""
    for old in [*d.glob("results_*.json"), *d.glob(f"samples_{task}_*.jsonl")]:
        old.unlink()


def words(task: str, sc: dict) -> str:
    """"GPQA Diamond 61.6% ± 3.1 · 4 runs of 198 · 2 ran out" — or, for a
    graded benchmark not yet graded, "SimpleQA Verified: 1,000 answers wait for
    its grader" """
    spec = fb.BENCH[task]
    if sc.get("refused"):
        return sc["refused"]
    if sc.get("no_score"):
        return sc["no_score"]
    if sc.get("waiting") is not None and sc.get("score") is None:
        return (f"{spec['label']}: {sc['waiting']:,} answers wait for its grader (AI models ▸ "
                "Start)" + (f" · {thought_note(sc['thinking_held'], sc.get('answers') or 0)}"
                            if sc.get("thinking_held") else ""))
    bits = [f"{spec['label']} {100 * sc['score']:.1f}% ± {100 * sc['se']:.1f}",
            f"{sc['epochs']} run{'s' if sc['epochs'] != 1 else ''} of {sc['questions']:,}"]
    if sc.get("ran_out"):
        bits.append(f"{sc['ran_out']} ran out of room")
    if sc.get("unread"):
        bits.append(f"{sc['unread']} with no answer read")
    if sc.get("ungraded"):
        bits.append(f"{sc['ungraded']} its grader gave no grade {GRADE_TRIES} times, counted "
                    "wrong")
    if sc.get("unanswered"):
        ids = sc.get("unanswered_ids") or []
        bits.append(f"{sc['unanswered']} the server never answered, counted wrong "
                    f"({', '.join(ids[:5])}{' …' if len(ids) > 5 else ''})")
    if sc.get("thinking_held"):
        # 17j: the share, as the run and the score's cell say it
        bits.append(thought_note(sc["thinking_held"], sc.get("answers") or 0))
    if (sc.get("look") or {}).get("waiting"):
        bits.append(f"code's score: Epoch's model check not run on {sc['look']['waiting']:,}")
    if sc.get("final") is False:
        # 17e: "graded by 0 graders" said for one grader with ungraded answers
        n = len(sc.get("graders") or [])
        bits.append(f"graded by {n} graders or prompts: not final" if n > 1 else
                    "not final until those get a grade (another grader on AI models)")
    return " · ".join(bits)


# ---------------------------------------------------------------------------
# a run: the runner's (run_submission dispatches a served model here)
# ---------------------------------------------------------------------------

def run(sid: int, sub: dict, rec: dict, th: dict, row: Path, log_path: Path) -> tuple[str, str]:
    """(status, line) for a Frontier run of a served model"""
    on = bool(th and th.get("on"))
    # 17e: asked in the suite's order, as the box lists them — the run's
    # record keeps its tasks sorted, which asked MMLU-Pro before SimpleQA:
    # the pilot's stop by hand in MMLU-Pro left SimpleQA never asked
    chosen = set(json.loads(sub.get("tasks") or "[]") or fb.TASKS)
    tasks = [t for t in fb.TASKS if t in chosen]
    sh = shard()

    def log(text: str) -> None:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(text.rstrip("\n") + "\n")
    log(f"[frontier] {rec['id']} · thinking {'on' if on else 'off'} · "
        f"{', '.join(tasks)} · sampling {family_of(rec) or 'the server’s defaults'}"
        + (f" · shard {sh[0]} of {sh[1]}" if sh else "") + f" · on {_where()}")
    lines, incomplete = [], []
    # 17g: the whole run checked before the first question — a benchmark a
    # slot of its server can't hold (the prompt and the budget) is said and
    # left, and every one that fits is asked: at -c 83,968 three were asked,
    # then HLE stopped the run, and two that fit never were
    ctx = (rec.get("pin") or {}).get("ctx")
    big = [t for t in tasks if isinstance(ctx, int) and 0 < ctx < fb.slot_context([t], on)]
    if big:
        skip = (f"{', '.join(fb.BENCH[t]['label'] for t in big)}: its server's context is "
                f"{ctx:,} tokens a slot, and thinking {'on' if on else 'off'} needs "
                + ", ".join(f"{fb.slot_context([t], on):,}" for t in big)
                + " (the budget and the prompt) — start it with a larger -c, or run its GGUF on "
                "a rented GPU. " + ("Nothing was asked" if len(big) == len(tasks) else
                                    "Not asked; the rest are"))
        log(f"[frontier] {skip}")
        if len(big) == len(tasks):
            return "failed", skip
        incomplete.append(skip)
        tasks = [t for t in tasks if t not in big]
    for task in tasks:
        label = fb.BENCH[task]["label"]

        def progress(n: int, total: int, each: float, label=label) -> None:
            left = (total - n) * each
            db.update(sid, status="running",
                      progress=f"{label} {n:,} of {total:,} · {each:.1f} s an answer · "
                               + (f"{left / 3600:.1f} h left" if left >= 3600 else
                                  f"{max(1, round(left / 60))} min left" if n < total
                                  else "all answered"))
        try:
            fb.load(task, config.BENCH_ROOT)
        except Exception as e:                          # noqa: BLE001 — said on the row
            # 17c: in the error's own words — a count that doesn't agree isn't
            # a gated set's terms
            line = f"{fb.load_failed(task, e)}. Nothing was asked"
            log(f"[frontier] {line} ({e!r})")
            return "failed", line
        try:
            n, total = ask_task(rec, task, row, on, progress,
                                canceled=lambda: db.cancel_requested(sid), log=log, sid=sid)
        except served.ServerStopped as e:
            # 17d: a stop with a reason says the reason (the thinking check's
            # starts with the benchmark's name)
            said = getattr(e, "refused", "") or ""
            line = (said if said.startswith(label) else f"{label}: {said}" if said else
                    f"{label}: the server stopped answering at {e.done:,} of {e.total:,}")
            line += served.KEPT_FOR_NEXT
            log(f"[frontier] {line} ({status_of(e.why)})")
            return "failed", line
        if db.cancel_requested(sid):
            return "canceled", f"{label}: stopped at {n:,} of {total:,}; the answers are kept"
        log(f"[frontier] {task}: {n:,} of {total:,} answered")
        if n < total:
            # 17b: the questions the server failed on, not kept: the next run asks
            # them. 17f: and this run carries on to the next benchmark — one
            # question that fails with a 5xx left a box's other benchmarks unasked
            incomplete.append(f"{label}: {n:,} of {total:,} answered — the server failed on "
                              f"{total - n:,}; the next run asks them again")
            continue
        if not config.FRONTIER_SCORE_AFTER_RUN or sh:
            lines.append(f"{label}: {n:,} of {total:,} answered")
            continue
        sc = score_task(row, task, rec, log=log)
        lines.append(words(task, sc) if sc else f"{label}: {n:,} of {total:,} answered")
        if sc:
            log(f"[frontier] {words(task, sc)}")
        if sc and sc.get("refused"):
            return "failed", " · ".join(lines)
    if incomplete:
        return "failed", " · ".join([*lines, *incomplete])
    return "done", " · ".join(lines)
