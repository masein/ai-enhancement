#!/usr/bin/env python3
"""Run Standard tasks on one model again, beside its numbers on the board (12h.1).

    sudo docker compose exec -T bench python scripts/trial_standard.py \\
        --model HuggingFaceTB/SmolLM2-135M-Instruct 2>&1 | tail -12

Inside the bench container, after an image change: the board's numbers for
this model were made on the image before it (12h.1 moves transformers to
5.5.3), and a queued run would not make them again — a run skips every task
it already has. This asks the model each task exactly as a run asks it (the
service's own preflight and lm_eval command) and prints, per task, the new
number beside the board's, and whether the difference is inside the noise.

The quick suite by default (HellaSwag and ARC-Easy), whole, so the numbers
compare. It writes only to a temporary folder, holds the GPU lock while it
runs (a queued run waits, as it would for any other), and never touches the
results.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import report_lm_eval as report  # noqa: E402
from service import config, runner  # noqa: E402


def run_harness(cmd: list[str], cwd: Path, log: Path) -> int:
    """lm_eval, as a run starts it; its output goes to `log`. The tests put a
    fake harness here."""
    with open(log, "a") as lf:
        return subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT).returncode


def scores(root: Path) -> dict[str, tuple[str, float, float]]:
    """task -> (metric, value, stderr): the headline number the board shows"""
    out = {}
    if not root.is_dir():
        return out
    for rec in report.merge_runs(report.load_results(root)).values():
        for task, entry in (rec.get("tasks") or {}).items():
            pm = report.primary_metric(entry)
            if pm:
                out[task] = pm
    return out


def compare(task: str, now: tuple[str, float, float] | None,
            board: tuple[str, float, float] | None) -> str:
    """"hellaswag · acc_norm 0.4123 ± 0.0049 · board 0.4130 ± 0.0049 · −0.0007,
    inside the noise" """
    if now is None:
        return f"{task}: no result — see the end of lm_eval's output above"
    metric, v, se = now
    line = f"{task} · {metric} {v:.4f} ± {se:.4f}"
    if board is None:
        return line + " · not on the board for this model"
    _, b, bse = board
    d = v - b
    z = abs(d) / math.sqrt(se * se + bse * bse) if (se or bse) else 0.0
    return (line + f" · board {b:.4f} ± {bse:.4f} · {d:+.4f}, "
            + ("inside the noise" if z <= 1.96 else f"{z:.1f} standard errors — a real "
               f"difference"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a Hugging Face id on the board")
    ap.add_argument("--tasks", default=",".join(config.QUICK_TASKS),
                    help="Standard tasks (default: the quick suite)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    a = ap.parse_args(argv)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    known = set(config.FULL_TASKS) | set(config.QUICK_TASKS)
    if not tasks or [t for t in tasks if t not in known]:
        ap.error(f"tasks are Standard tasks: {', '.join(config.FULL_TASKS)}")

    from service.hfmeta import PreflightError, preflight
    try:
        meta = preflight(a.model, "auto")
    except PreflightError as e:
        print(f"{a.model} does not load here: {e}")
        return 2
    if meta.get("remote_code"):
        # a trial runs as root, with the network: a model's own code runs only
        # in a queued run, which drops privileges and goes offline first
        print(f"{a.model} runs its own model code; a trial never does. Queue a run instead.")
        return 2
    try:
        tv = md.version("transformers")
    except md.PackageNotFoundError:
        tv = "?"
    board = scores(config.OUT_DIR / a.model.replace("/", "__"))
    print(f"{a.model} · {meta['kind']} · transformers {tv} · beside its numbers on the board")

    if not runner.acquire_lock(0):
        print("A run holds the GPU. Try again when the queue is idle.")
        return 3
    work = Path(tempfile.mkdtemp(prefix="trial-standard-"))
    got: dict[str, tuple[str, float, float] | None] = {}
    try:
        for task in tasks:
            shots = config.NFEWSHOT.get(task, 0)
            task_out = work / a.model.replace("/", "__") / f"{task}_{shots}shot"
            task_out.mkdir(parents=True, exist_ok=True)
            cmd = runner.lm_eval_cmd(f"pretrained={a.model},dtype=bfloat16", task, shots,
                                     meta["batch"], task_out, chat=meta["kind"] == "instruct")
            log = work / f"{task}.log"
            status = run_harness(cmd, runner.lm_eval_cwd(task_out), log)
            got[task] = scores(task_out).get(task) if status == 0 else None
            if got[task] is None:
                tail = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
                print(f"{task}: lm_eval exited {status}. The end of its output:")
                for ln in tail[-12:]:
                    print(f"    | {ln}")
            print(compare(task, got[task], board.get(task)))
    finally:
        runner.release_lock()
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept: {work}")
    ok = all(got.get(t) for t in tasks)
    print("trial OK: every task ran" if ok
          else "trial FAILED: " + ", ".join(t for t in tasks if not got.get(t)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
