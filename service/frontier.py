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

When every question of every run is answered, the task is scored by code and
written in lm_eval's layout (results_*.json, samples_*.jsonl beside the
answers), so the board reads its cell as it reads any other: the share right
over all runs, with its error over questions. A rented box doesn't score
(FRONTIER_SCORE_AFTER_RUN=0): the server scores what it imports.
"""

from __future__ import annotations

import hashlib
import json
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
            out[(str(r["id"]), int(r["epoch"]))] = r
        except (ValueError, KeyError, TypeError):
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


def _write_setup(d: Path, rec: dict, task: str, on: bool, s: dict) -> None:
    spec = fb.BENCH[task]
    sh = shard()
    (d / SETUP).write_text(json.dumps({
        "version": fb.VERSION, "task": task, "protocol": spec["protocol"],
        "protocol_version": spec["protocol_version"], "source": spec["source"],
        "epochs": spec["epochs"], "thinking": "on" if on else "off",
        "budget": s["max_tokens"], "family": family_of(rec) or "server defaults",
        "sampling": {k: v for k, v in s.items() if k not in ("max_tokens",
                                                            "chat_template_kwargs")},
        "where": _where(), "file": (rec.get("pin") or {}).get("file") or "",
        "shard": {"i": sh[0], "n": sh[1]} if sh else None}, indent=1, sort_keys=True),
        encoding="utf-8")


def ask_task(rec: dict, task: str, row: Path, on: bool, progress=None,
             canceled=lambda: False) -> tuple[int, int]:
    """every question of `task` this run (or shard) holds and hasn't answered
    — (answered, of) when it stops. Raises served.ServerStopped when the
    server stops answering: what it answered before is kept"""
    fb.set_root(config.BENCH_ROOT)
    spec = fb.BENCH[task]
    items = fb.shard_of(fb.load(task, config.BENCH_ROOT), shard())
    d = task_dir(row, task)
    d.mkdir(parents=True, exist_ok=True)
    s = settings(rec, task, on)
    _write_setup(d, rec, task, on, s)
    path = d / ANSWERS
    done = read_answers(path)
    want = [(it, e) for it in items for e in range(spec["epochs"])]
    todo = [(it, e) for it, e in want if (it["id"], e) not in done]
    total, have = len(want), len(want) - len(todo)
    lock = threading.Lock()
    halt: list[Exception] = []
    t0, n0 = time.time(), have
    count = {"n": have}

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
                    f"the server refused a question: {x}")))
            return
        line = {"id": it["id"], "epoch": e, "seed": seed, "answer": str(a),
                "finish": getattr(a, "finish", None), "tokens": a.tokens,
                "at": round(time.time(), 3)}
        if a.fallback:
            line["fallback"] = a.fallback
        if a.error:
            line["error"] = a.error
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            count["n"] += 1
            n = count["n"]
        if progress:
            progress(n, total, (time.time() - t0) / max(1, n - n0))

    with ThreadPoolExecutor(max_workers=served.concurrency(rec)) as pool:
        list(pool.map(one, todo))
    if halt:
        e = halt[0]
        e.done, e.total = count["n"], total
        raise e
    return count["n"], total


# ---------------------------------------------------------------------------
# scoring: by code, once every question of every run is in — and (stage 3)
# by a grader, or code with Epoch's model check as a second look
# ---------------------------------------------------------------------------

GRADES = "grades.json"


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


def marks(row: Path, task: str, items: list[dict] | None = None) -> dict:
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
    got, grades = read_answers(d / ANSWERS), (read_grades(d).get("items") or {})
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
            ok = sc["ok"]
            if sc["ran_out"]:
                ok = False
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
                "error": a.get("error"), "grade": g})
    return {"runs": runs, "missing": missing, "items": items}


def to_grade(row: Path, task: str) -> list[dict]:
    """stage 3: the answers a grader (or Epoch's model check) is still to see —
    every answer of a graded benchmark, and for one Epoch checks with a model,
    those the code marks wrong or can't read; never one that ran out of room"""
    spec = fb.BENCH[task]
    if not (spec.get("grader") or spec.get("look")):
        return []
    m = marks(row, task)
    if m["missing"]:
        return []
    out = []
    for it in m["items"]:
        for r in m["runs"].get(it["id"], []):
            if r["grade"] is None and r["ok"] is None and not r["ran_out"]:
                out.append({"id": it["id"], "epoch": r["epoch"], "answer": r["answer"],
                            "read": r["read"]})
    return out


def _share(per: dict[str, list[float]], task: str, items: list[dict]) -> dict:
    """the benchmark's score from each question's runs: its own aggregate"""
    groups = {it["id"]: fb.group_of(task, it) for it in items}
    return fb.summary(per, task, groups)


def score_task(row: Path, task: str, rec: dict) -> dict | None:
    """the task's score from its answers, written in lm_eval's layout — None
    while a question of a run is still unanswered. A graded benchmark is
    written once every answer is graded; until then {waiting: n}. One Epoch
    checks with a model is written with the code's score until the check is
    done, and with both after: the page's is Epoch's way"""
    spec = fb.BENCH[task]
    m = marks(row, task)
    if m["missing"]:
        return None
    items, runs = m["items"], m["runs"]
    d = task_dir(row, task)
    flat = [r for it in items for r in runs[it["id"]]]
    waiting = sum(1 for r in flat if r["ok"] is None)
    ran_out = sum(1 for r in flat if r["ran_out"])
    unread = sum(1 for r in flat if r["read"] is None and not r["ran_out"])
    errors = sum(1 for r in flat if r["error"])
    if spec.get("grader") and waiting:
        return {"waiting": waiting, "of": len(flat), "label": spec["label"]}
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
    for old in [*d.glob("results_*.json"), *d.glob(f"samples_{task}_*.jsonl")]:
        old.unlink()
    try:
        setup = json.loads((d / SETUP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        setup = {}
    g = read_grades(d).get("grader")
    looked = sum(1 for r in flat if r["grade"] is not None)
    detail = {"version": fb.VERSION, "protocol": spec["protocol"], "epochs": spec["epochs"],
              "questions": len(items), "answers": len(flat),
              "ran_out": ran_out, "unread": unread, "errors": errors,
              "budget": setup.get("budget"), "sampling": setup.get("sampling"),
              "family": setup.get("family"), "where": setup.get("where"),
              "thinking": setup.get("thinking"), "note": spec.get("note") or "",
              "scored_by": ("grader" if spec.get("grader") else "code, then Epoch's model check"
                            if spec.get("look") and not waiting and looked else "code"),
              "code": code, "grader": g,
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


def words(task: str, sc: dict) -> str:
    """"GPQA Diamond 61.6% ± 3.1 · 4 runs of 198 · 2 ran out" — or, for a
    graded benchmark not yet graded, "SimpleQA Verified: 1,000 answers wait for
    its grader" """
    spec = fb.BENCH[task]
    if sc.get("waiting") is not None and sc.get("score") is None:
        return (f"{spec['label']}: {sc['waiting']:,} answers wait for its grader (AI models ▸ "
                "Start)")
    bits = [f"{spec['label']} {100 * sc['score']:.1f}% ± {100 * sc['se']:.1f}",
            f"{sc['epochs']} run{'s' if sc['epochs'] != 1 else ''} of {sc['questions']:,}"]
    if sc.get("ran_out"):
        bits.append(f"{sc['ran_out']} ran out of room")
    if sc.get("unread"):
        bits.append(f"{sc['unread']} with no answer read")
    if (sc.get("look") or {}).get("waiting"):
        bits.append(f"code's score: Epoch's model check not run on {sc['look']['waiting']:,}")
    return " · ".join(bits)


# ---------------------------------------------------------------------------
# a run: the runner's (run_submission dispatches a served model here)
# ---------------------------------------------------------------------------

def run(sid: int, sub: dict, rec: dict, th: dict, row: Path, log_path: Path) -> tuple[str, str]:
    """(status, line) for a Frontier run of a served model"""
    on = bool(th and th.get("on"))
    tasks = [t for t in (json.loads(sub.get("tasks") or "[]") or fb.TASKS) if t in fb.BENCH]
    sh = shard()

    def log(text: str) -> None:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(text.rstrip("\n") + "\n")
    log(f"[frontier] {rec['id']} · thinking {'on' if on else 'off'} · "
        f"{', '.join(tasks)} · sampling {family_of(rec) or 'the server’s defaults'}"
        + (f" · shard {sh[0]} of {sh[1]}" if sh else "") + f" · on {_where()}")
    lines = []
    for task in tasks:
        label = fb.BENCH[task]["label"]

        def progress(n: int, total: int, each: float, label=label) -> None:
            left = (total - n) * each
            db.update(sid, status="running",
                      progress=f"{label} {n:,} of {total:,} · {each:.1f} s an answer · "
                               + (f"{left / 3600:.1f} h left" if left >= 3600 else
                                  f"{max(1, round(left / 60))} min left" if n < total
                                  else "all answered"))
        # 17: a slot of its server holds the prompt and the budget, or nothing is asked
        ctx, need = (rec.get("pin") or {}).get("ctx"), fb.slot_context([task], on)
        if isinstance(ctx, int) and 0 < ctx < need:
            line = (f"{label}: its server's context is {ctx:,} tokens a slot, and thinking "
                    f"{'on' if on else 'off'} needs {need:,} (the budget and the prompt) — "
                    f"start it with a larger -c, or run its GGUF on a rented GPU. Nothing was "
                    "asked")
            log(f"[frontier] {line}")
            return "failed", line
        try:
            fb.load(task, config.BENCH_ROOT)
        except Exception as e:                          # noqa: BLE001 — said on the row
            line = (f"{label}: its questions could not be fetched"
                    + (" — it is gated: accept its terms on Hugging Face with the account "
                       "whose token this machine has (HF_TOKEN)"
                       if fb.BENCH[task]["source"].get("gated") else "")
                    + ". Nothing was asked")
            log(f"[frontier] {line} ({e!r})")
            return "failed", line
        try:
            n, total = ask_task(rec, task, row, on, progress,
                                canceled=lambda: db.cancel_requested(sid))
        except served.ServerStopped as e:
            line = (f"{label}: {e}" if str(e).startswith("the server refused") else
                    f"{label}: the server stopped answering at {e.done:,} of {e.total:,}")
            line += served.KEPT_FOR_NEXT
            log(f"[frontier] {line} ({e.why})")
            return "failed", line
        if db.cancel_requested(sid):
            return "canceled", f"{label}: stopped at {n:,} of {total:,}; the answers are kept"
        log(f"[frontier] {task}: {n:,} of {total:,} answered")
        if not config.FRONTIER_SCORE_AFTER_RUN or sh:
            lines.append(f"{label}: {n:,} of {total:,} answered")
            continue
        sc = score_task(row, task, rec)
        lines.append(words(task, sc) if sc else f"{label}: {n:,} of {total:,} answered")
        if sc:
            log(f"[frontier] {words(task, sc)}")
    return "done", " · ".join(lines)
