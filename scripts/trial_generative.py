#!/usr/bin/env python3
"""Try IFEval, MMLU-Pro and MATH-500 on one model, a few items each (12h.1).

    sudo docker compose exec -T bench python scripts/trial_generative.py \\
        --model Qwen/Qwen3-1.7B --limit 20 2>&1 | tail -60

Inside the bench container, before any full run: it asks the model N items of
each benchmark exactly as a run asks them — the service's own preflight, its
lm_eval command, its thinking switch and its backend (vLLM, falling back to
hf) — and prints, per item, what the model wrote, what this board read out of
it, the correct answer and the verdict. So it is the check that the answers
can be read at all, and that the model loads.

Then the time: seconds per item, and what a full MMLU-Pro run (12,032
chain-of-thought answers) would take at that rate. Over three hours, masein
decides between the full run and a seeded subset, which the board labels as a
subset; only a full run is comparable to published numbers. 12a.5b: two rates
per benchmark — with loading the model in it, and generating alone, from
lm_eval's own progress bar; the estimate uses the second, since a run loads
the model once.

It writes only to a temporary folder, holds the GPU lock while it runs (a
queued run waits, as it would for any other), and never touches the results.

12a.5b: IFEval and MATH-500 ask --limit items (20 unless said), MMLU-Pro
--limit per subject (2 unless said: 28 items), and the counts are printed
first. It stops — lm_eval and everything it started killed, the GPU lock
freed, "stopped; GPU free" — on Ctrl+C, when the terminal that started it
goes away, and at --max-minutes (15 unless said); scripts/trial_stop.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import generative as gen  # noqa: E402
import judge as _judge  # noqa: E402
import trial_stop  # noqa: E402
from service import catalog, config, runner  # noqa: E402

FULL_MMLU_PRO = sum(config.MMLU_PRO_SUBJECTS.values())
OVER = 3 * 3600                     # past this, masein chooses full or subset
# 12a.5b: what it asks unless --limit says: items of IFEval and MATH-500, and
# of each MMLU-Pro subject; the order a run asks them, MMLU-Pro last
ITEMS, PER_SUBJECT = 20, 2
ORDER = ["ifeval", "hendrycks_math500", "mmlu_pro"]
STOP: trial_stop.Stop | None = None


def run_harness(cmd: list[str], cwd: Path, log: Path) -> int:
    """lm_eval, as a run starts it; its output goes to `log`. It stops as the
    trial does (trial_stop). The tests put a fake model here."""
    if STOP is not None:
        return STOP.run(cmd, cwd, log)
    with open(log, "a") as lf:
        return subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT).returncode


def duration(seconds: float) -> str:
    """"about 2 h 40 min" — the runner's words"""
    return runner.duration(seconds)


def counts(tasks: list[str], limit: int | None) -> dict[str, str]:
    """what it asks of each, in words: "MMLU-Pro: 28 items (2 per subject × 14)" """
    out = {}
    for t in tasks:
        if t == "mmlu_pro":
            k, n = limit or PER_SUBJECT, len(config.MMLU_PRO_SUBJECTS)
            got = sum(len(v) for v in runner.mmlu_pro_per_subject(k).values())
            out[t] = f"{gen.TASKS[t]}: {got} items ({k} per subject × {n})"
        else:
            out[t] = f"{gen.TASKS[t]}: {limit or ITEMS} items"
    return out


def clip(s: str, n: int = 70) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def show(task: str, recs: list[dict], seconds: float, out=print,
         generating: tuple[int, int] | None = None) -> dict:
    """Per item, what was written, read and expected; then the task's line.
    12a.5b: `generating` — (items, seconds) from lm_eval's own bar, the time
    without loading the model, which a trial of two items is mostly"""
    name = gen.TASKS[task]
    per = seconds / len(recs) if recs else 0.0
    per_gen = generating[1] / generating[0] if generating and generating[0] else None
    out(f"{name} — {len(recs)} items in {seconds:.0f} s, {per:.2f} s per item"
        + (f" · generating {per_gen:.2f} s per item, loading left out"
           if per_gen is not None else ""))
    right = unreadable = ran_out = 0
    for i, rec in enumerate(recs, 1):
        r = gen.read(task, rec)
        right += r["correct"]
        ran_out += r["ran_out"]
        if not r["ran_out"] and r["extracted"] in (None, ""):
            unreadable += 1
        mark = "✓" if r["correct"] else "✗"
        if task == "ifeval":
            read_as = "verdict of the harness's checker"
        else:
            read_as = f"read {r['extracted'] if r['extracted'] not in (None, '') else '—'}"
        why = (" (ran out of room)" if r["ran_out"]
               else " (nothing to read)" if r["extracted"] in (None, "") else "")
        out(f"  {i:>3}  {mark}  {read_as}  ·  answer {clip(str(r['key'] or '—'), 40)}{why}")
        out(f"         wrote: {clip(gen.raw_generation(rec))}")
    out(f"{name}: {right} of {len(recs)} right · {unreadable} unreadable · "
        f"{ran_out} ran out of room")
    return {"n": len(recs), "right": right, "unreadable": unreadable, "ran_out": ran_out,
            "per_item": per, "per_generating": per_gen}


def estimate(per_item: float, out=print) -> float:
    """A full MMLU-Pro run at this rate; the choice it leaves, when over 3 h.
    12a.5b: the generating rate when lm_eval's bar gave one"""
    full = per_item * FULL_MMLU_PRO
    out(f"A full MMLU-Pro run at this rate: {duration(full)} for {FULL_MMLU_PRO:,} "
        f"(the model loads once in a run; a rate with loading in it over-estimates)")
    if full > OVER:
        out("Over 3 hours: masein decides — the full run, or a seeded subset (the "
            "submit form's MMLU-Pro subset), labelled as a subset on the board. Only a "
            "full run is comparable to published numbers.")
    return full


def main(argv=None) -> int:
    global STOP
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models from 12h.1, for --model:\n" + "\n".join(
            f"  {m['id']}" for m in catalog.MODELS))
    ap.add_argument("--model", required=True, help="a Hugging Face id, or local/<name>")
    ap.add_argument("--limit", type=int, default=None,
                    help=f"items of IFEval and MATH-500 (default {ITEMS}), and of each "
                         f"MMLU-Pro subject (default {PER_SUBJECT})")
    ap.add_argument("--max-minutes", type=float, default=trial_stop.MAX_MINUTES,
                    help="stop at this many minutes, keeping what finished (default "
                         f"{trial_stop.MAX_MINUTES:g})")
    ap.add_argument("--thinking", action="store_true",
                    help="think before answering (a model with a switch only)")
    ap.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto",
                    help="auto: vLLM where the model loads in it, else hf")
    ap.add_argument("--tasks", default=",".join(ORDER),
                    help="which of the three (default: all, MMLU-Pro last)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    a = ap.parse_args(argv)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in gen.TASKS]
    if unknown or (a.limit is not None and a.limit < 1) or a.max_minutes <= 0:
        ap.error(f"tasks are {', '.join(gen.TASKS)}; --limit is at least 1; --max-minutes is "
                 f"more than 0")

    from service.hfmeta import PreflightError, preflight
    try:
        meta = preflight(a.model, "instruct")
    except PreflightError as e:
        print(f"{a.model} does not load here: {e}")
        return 2
    if meta["kind"] != "instruct":
        print(config.GEN_INSTRUCT_ONLY)
        return 2
    if meta.get("remote_code"):
        # a trial runs as root, with the network: a model's own code runs only
        # in a queued run, which drops privileges and goes offline first
        print(f"{a.model} runs its own model code (approved at "
              f"{(meta.get('revision') or '')[:12]}). Try it with a queued run, which runs "
              f"that code as the unprivileged user; a trial never does.")
        return 2
    th = runner.gen_thinking({"thinking": a.thinking}, meta)
    backend, why = runner.gen_backend()
    if a.backend != "auto":
        backend, why = a.backend, f"--backend {a.backend}"
    print(f"{a.model} · thinking {'on' if th['on'] else 'off'} ({th['mode']}) · "
          f"{th['budget']} tokens per answer · on {backend}" + (f" ({why})" if why else "")
          )
    for line in counts(tasks, a.limit).values():
        print(line)
    print(f"It stops at {a.max_minutes:g} minutes (--max-minutes), keeping what finished.")

    if not runner.acquire_lock(0):
        print("A run holds the GPU. Try again when the queue is idle.")
        return 3
    work = Path(tempfile.mkdtemp(prefix="trial-generative-"))
    mdir = work / a.model.replace("/", "__")
    pretrained = a.model if not a.model.startswith("local/") else \
        str((config.ARTIFACTS_DIR / a.model[6:]).resolve())
    summary = {}
    stopped = ""
    STOP = trial_stop.Stop(a.max_minutes)
    try:
        STOP.__enter__()
        for task in tasks:
            shots = config.NFEWSHOT.get(task, 0)
            task_out = mdir / f"{task}_{shots}shot"
            task_out.mkdir(parents=True, exist_ok=True)
            samples = None
            if task == "mmlu_pro":        # 12a.5b: k seeded items of each subject
                samples = task_out / "subset.json"
                samples.write_text(json.dumps(runner.mmlu_pro_per_subject(a.limit or PER_SUBJECT)),
                                   encoding="utf-8")

            def cmd_for(be):
                return runner.lm_eval_cmd(
                    runner.gen_model_args(
                        pretrained, th, backend=be,
                        gpu_util=runner.gpu_util_for(runner.gpu_free_mib(),
                                                     runner.gpu_total_mib())
                        if be == "vllm" else None),
                    task, shots, meta["batch"], task_out, chat=True,
                    max_gen_toks=th["budget"], backend=be, samples=samples,
                    limit=None if samples else (a.limit or ITEMS))
            log = work / f"{task}.log"
            t0 = time.time()
            status = run_harness(cmd_for(backend), runner.lm_eval_cwd(task_out), log)
            if status != 0 and backend == "vllm" and a.backend == "auto":
                print(f"{gen.TASKS[task]}: vLLM could not run it (exit {status}); trying hf")
                backend = "hf"
                t0 = time.time()
                status = run_harness(cmd_for("hf"), runner.lm_eval_cwd(task_out), log)
            seconds = time.time() - t0
            recs = _judge._records(mdir, task)
            if status != 0 or not recs:
                tail = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
                print(f"{gen.TASKS[task]}: lm_eval exited {status} with "
                      f"{len(recs)} answers. The end of its output:")
                for ln in tail[-12:]:
                    print(f"    | {ln}")
                summary[task] = None
                continue
            summary[task] = show(task, recs, seconds,
                                 generating=runner.bar_seconds(
                                     log.read_text(errors="replace") if log.exists() else ""))
            print()
        if summary.get("mmlu_pro"):
            s = summary["mmlu_pro"]
            estimate(s["per_generating"] or s["per_item"])
    except trial_stop.Stopped as e:
        stopped = str(e)
    finally:
        # lm_eval is gone before the lock is: Stop.run kills its whole group
        STOP.__exit__(None, None, None)
        STOP = None
        runner.release_lock()
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept: {work}")
    if stopped:
        done = [gen.TASKS[t] for t in tasks if summary.get(t)]
        trial_stop.say(f"{stopped}. Finished: {', '.join(done) if done else 'nothing'}.")
        trial_stop.say("stopped; GPU free")
        return 4
    ok = all(summary.get(t) for t in tasks)
    print("trial OK: every benchmark answered and read" if ok
          else "trial FAILED: " + ", ".join(gen.TASKS[t] for t in tasks if not summary.get(t)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
