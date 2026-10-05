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
        try:
            a = served.answer_one(rec, text, {**s, "seed": seed})
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
# scoring: by code, once every question of every run is in
# ---------------------------------------------------------------------------

def score_task(row: Path, task: str, rec: dict) -> dict | None:
    """the task's score from its answers, written in lm_eval's layout — None
    while a question of a run is still unanswered"""
    spec = fb.BENCH[task]
    items = fb.load(task, config.BENCH_ROOT)
    d = task_dir(row, task)
    got = read_answers(d / ANSWERS)
    missing = [(it["id"], e) for it in items for e in range(spec["epochs"])
               if (it["id"], e) not in got]
    if missing:
        return None
    per: dict[str, list[float]] = {}
    rows, ran_out, unread, errors = [], 0, 0, 0
    for k, it in enumerate(items):
        _, need = fb.prompt(task, it)
        runs = []
        for e in range(spec["epochs"]):
            a = got[(it["id"], e)]
            sc = fb.score(task, a.get("answer") or "", a.get("finish"), need)
            ran_out += sc["ran_out"]
            unread += (sc["read"] is None and not sc["ran_out"])
            errors += bool(a.get("error"))
            runs.append({"epoch": e, **sc, "tokens": a.get("tokens")})
        per[it["id"]] = [1.0 if r["ok"] else 0.0 for r in runs]
        rows.append({"doc_id": k, "doc": {"id": it["id"]},
                     # never the question: only its id (GPQA's are never shown)
                     "doc_hash": hashlib.sha256(f"{task}:{it['id']}".encode()).hexdigest(),
                     "arguments": [], "target": need.get("key"),
                     "resps": [[got[(it["id"], r["epoch"])].get("answer") or ""]
                               for r in runs],
                     "filtered_resps": [r["read"] or "" for r in runs],
                     "acc": sum(per[it["id"]]) / len(runs), "frontier": runs})
    sm = fb.summary(per)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S.000000", time.gmtime())
    for old in [*d.glob("results_*.json"), *d.glob(f"samples_{task}_*.jsonl")]:
        old.unlink()
    try:
        setup = json.loads((d / SETUP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        setup = {}
    detail = {"version": fb.VERSION, "protocol": spec["protocol"], "epochs": spec["epochs"],
              "questions": len(items), "answers": len(items) * spec["epochs"],
              "ran_out": ran_out, "unread": unread, "errors": errors,
              "budget": setup.get("budget"), "sampling": setup.get("sampling"),
              "family": setup.get("family"), "where": setup.get("where"),
              "thinking": setup.get("thinking")}
    (d / f"results_{stamp}.json").write_text(json.dumps({
        "results": {task: {"alias": task, "acc,none": sm["score"],
                           "acc_stderr,none": sm["se"]}},
        "group_subtasks": {task: []}, "n-shot": {task: 0},
        "n-samples": {task: {"original": len(items), "effective": len(items)}},
        "higher_is_better": {task: {"acc": True}},
        "configs": {task: {"task": task, "output_type": "generate_until", "num_fewshot": 0,
                           "dataset_path": spec["source"]["hf"],
                           "dataset_kwargs": {"revision": spec["source"]["revision"]}}},
        "config": {"model": "frontier", "model_args": f"pretrained={rec['id']}"
                   + served._row_mark(d.parent)},
        "chat_template": "the server's own", "date": time.time(), "frontier": detail,
        "served": served.view(rec)},
        indent=1), encoding="utf-8")
    (d / f"samples_{task}_{stamp}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {**sm, **detail}


def words(task: str, sc: dict) -> str:
    """"GPQA Diamond 61.6% ± 3.1 · 4 runs of 198 · 2 ran out" """
    spec = fb.BENCH[task]
    bits = [f"{spec['label']} {100 * sc['score']:.1f}% ± {100 * sc['se']:.1f}",
            f"{sc['epochs']} run{'s' if sc['epochs'] != 1 else ''} of {sc['questions']:,}"]
    if sc.get("ran_out"):
        bits.append(f"{sc['ran_out']} ran out of room")
    if sc.get("unread"):
        bits.append(f"{sc['unread']} with no ANSWER line")
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
