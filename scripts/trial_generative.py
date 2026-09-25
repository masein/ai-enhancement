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
subset; only a full run is comparable to published numbers. The rate includes
loading the model, so a short trial over-estimates it.

It writes only to a temporary folder, holds the GPU lock while it runs (a
queued run waits, as it would for any other), and never touches the results.
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
from service import config, runner  # noqa: E402

FULL_MMLU_PRO = sum(config.MMLU_PRO_SUBJECTS.values())
OVER = 3 * 3600                     # past this, masein chooses full or subset


def run_harness(cmd: list[str], cwd: Path, log: Path) -> int:
    """lm_eval, as a run starts it; its output goes to `log`. The tests put a
    fake model here."""
    with open(log, "a") as lf:
        return subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT).returncode


def duration(seconds: float) -> str:
    """"about 2 h 40 min" """
    m = int(round(seconds / 60))
    if m < 60:
        return f"about {max(m, 1)} min"
    return f"about {m // 60} h {m % 60} min" if m % 60 else f"about {m // 60} h"


def clip(s: str, n: int = 70) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def show(task: str, recs: list[dict], seconds: float, out=print) -> dict:
    """Per item, what was written, read and expected; then the task's line"""
    name = gen.TASKS[task]
    per = seconds / len(recs) if recs else 0.0
    out(f"{name} — {len(recs)} items in {seconds:.0f} s, {per:.2f} s per item")
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
            "per_item": per}


def estimate(per_item: float, out=print) -> float:
    """A full MMLU-Pro run at this rate; the choice it leaves, when over 3 h"""
    full = per_item * FULL_MMLU_PRO
    out(f"A full MMLU-Pro run at this rate: {duration(full)} for {FULL_MMLU_PRO:,} "
        f"(the rate includes loading the model, so a short trial over-estimates)")
    if full > OVER:
        out("Over 3 hours: masein decides — the full run, or a seeded subset (the "
            "submit form's MMLU-Pro subset), labelled as a subset on the board. Only a "
            "full run is comparable to published numbers.")
    return full


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a Hugging Face id, or local/<name>")
    ap.add_argument("--limit", type=int, default=20, help="items per benchmark (default 20)")
    ap.add_argument("--thinking", action="store_true",
                    help="think before answering (a model with a switch only)")
    ap.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto",
                    help="auto: vLLM where the model loads in it, else hf")
    ap.add_argument("--tasks", default=",".join(gen.TASKS),
                    help="which of the three (default: all)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    a = ap.parse_args(argv)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in gen.TASKS]
    if unknown or a.limit < 1:
        ap.error(f"tasks are {', '.join(gen.TASKS)}; --limit is at least 1")

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

    if not runner.acquire_lock(0):
        print("A run holds the GPU. Try again when the queue is idle.")
        return 3
    work = Path(tempfile.mkdtemp(prefix="trial-generative-"))
    mdir = work / a.model.replace("/", "__")
    pretrained = a.model if not a.model.startswith("local/") else \
        str((config.ARTIFACTS_DIR / a.model[6:]).resolve())
    summary = {}
    try:
        for task in tasks:
            shots = config.NFEWSHOT.get(task, 0)
            task_out = mdir / f"{task}_{shots}shot"
            task_out.mkdir(parents=True, exist_ok=True)
            samples = None
            if task == "mmlu_pro":        # N items across the subjects, not N per subject
                samples = task_out / "subset.json"
                samples.write_text(json.dumps(runner.mmlu_pro_subset(a.limit)), encoding="utf-8")

            def cmd_for(be):
                return runner.lm_eval_cmd(
                    runner.gen_model_args(
                        pretrained, th, backend=be,
                        gpu_util=runner.gpu_util_for(runner.gpu_free_mib(),
                                                     runner.gpu_total_mib())
                        if be == "vllm" else None),
                    task, shots, meta["batch"], task_out, chat=True,
                    max_gen_toks=th["budget"], backend=be, samples=samples,
                    limit=None if samples else a.limit)
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
            summary[task] = show(task, recs, seconds)
            print()
        if summary.get("mmlu_pro"):
            estimate(summary["mmlu_pro"]["per_item"])
    finally:
        runner.release_lock()
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept: {work}")
    ok = all(summary.get(t) for t in tasks)
    print("trial OK: every benchmark answered and read" if ok
          else "trial FAILED: " + ", ".join(gen.TASKS[t] for t in tasks if not summary.get(t)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
