#!/usr/bin/env python3
"""Deploy step 4: ask the installed lm_eval to find every task the board runs.

    sudo docker compose exec -T bench python scripts/check_tasks.py

A run hands lm_eval one task at a time, by name. For every task of every
suite (config.SUITES: the standard tasks, the permutation control, each exam
topic, Everyday tasks), this builds the command a run builds
(runner.lm_eval_cmd), goes to the folder a run starts it in
(runner.lm_eval_cwd), and gives the command to lm_eval's own command line.
lm_eval is stopped where it would load the model, after it has chosen its
tasks: no GPU, no model, no dataset, only whether the name finds its task.

Why: every everyday run from #62 to #65 failed right there, before asking a
question. The runner started lm_eval in BENCH_ROOT, which holds the folder
everyday/, and lm_eval 0.4.12 reads a --tasks value naming a folder as a
folder of task files. It found none and selected nothing. The unit suite
cannot see that: lm_eval is not installed where scripts/check.sh runs, and
the exam topics exist only on the server. This asks the real lm_eval, in the
real container, about the real task folders.

It writes only to a temporary folder: Everyday's task is built there from
the deployed bank, as a run builds it, and each command's output folder is
made there. Ends with one line:

    tasks OK: 52 of 52 found by lm_eval 0.4.12

or `tasks FAILED: …` naming each task not found and why, with the end of
lm_eval's output for it, and exits 1.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from service import config, runner  # noqa: E402

MODEL_ARGS = "pretrained=check-tasks/no-model,dtype=bfloat16"   # never loaded


class _Selected(BaseException):
    """raised in place of simple_evaluate: lm_eval has chosen its tasks. Not
    an Exception, so no `except Exception` on the way out can swallow it"""


_chosen: dict = {}


def _stop(*args, **kwargs):
    _chosen["tasks"] = kwargs.get("tasks")
    raise _Selected()


def _harness():
    """The installed lm_eval, stopped at simple_evaluate, with its task index
    built once per include path: every command builds the same index, and
    about fifty of them would build it fifty times. Returns (cli, version)."""
    import lm_eval
    import lm_eval.evaluator as evaluator
    import lm_eval.tasks as lm_tasks
    real = evaluator.simple_evaluate
    # the run command imports simple_evaluate when it runs; anything that took
    # it at import time gets the stop too
    evaluator.simple_evaluate = _stop
    lm_eval.simple_evaluate = _stop
    import lm_eval.__main__ as cli
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] == "lm_eval" and getattr(mod, "simple_evaluate", None) is real:
            mod.simple_evaluate = _stop

    Real = lm_tasks.TaskManager
    built: dict[str, object] = {}

    class _OneIndexPerPath(Real):
        def __new__(cls, *a, **k):
            key = repr((a, sorted(k.items(), key=lambda kv: kv[0])))
            if key not in built:
                built[key] = Real(*a, **k)
            return built[key]
    lm_tasks.TaskManager = _OneIndexPerPath
    return cli, getattr(lm_eval, "__version__", "?")


@contextlib.contextmanager
def _captured():
    """lm_eval's own output for one command, kept aside and shown only when
    the task was not found"""
    got = {"text": ""}
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile() as buf:
        saved = os.dup(1), os.dup(2)
        os.dup2(buf.fileno(), 1)
        os.dup2(buf.fileno(), 2)
        try:
            yield got
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            buf.seek(0)
            got["text"] = buf.read().decode("utf-8", "replace")


def _name(t) -> str | None:
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        return t.get("task")
    return getattr(t, "task", None)


def find(cli, task: str, work: Path) -> tuple[bool, str, str]:
    """(found, why not, lm_eval's output) for one task, asked exactly as a run
    asks: its command, from its folder."""
    shots = config.NFEWSHOT.get(task, 0)
    task_out = work / f"{task}_{shots}shot"
    task_out.mkdir(parents=True, exist_ok=True)
    cmd = runner.lm_eval_cmd(MODEL_ARGS, task, shots, 1, task_out,
                             chat=task == config.EVERYDAY_TASK or task.startswith(("exam_", "fr_")))
    cwd = runner.lm_eval_cwd(task_out)
    here, argv = os.getcwd(), sys.argv
    err = ""
    _chosen.clear()
    with _captured() as out:
        try:
            os.chdir(cwd)
            sys.argv = list(cmd)
            cli.cli_evaluate()
            err = "lm_eval returned without choosing any task"
        except _Selected:
            pass
        except SystemExit as e:
            err = f"lm_eval exited ({e.code})"
        except Exception as e:                     # noqa: BLE001 — said, not raised
            err = f"{type(e).__name__}: {e}"
        finally:
            os.chdir(here)
            sys.argv = argv
    if "tasks" not in _chosen:
        return False, err, out["text"]
    sel = _chosen["tasks"]
    names = [_name(t) for t in sel or []]
    if not sel:
        why = "lm_eval selected nothing"
        if (Path(cwd) / task).is_dir():
            why += (f": {task}/ is a folder in {cwd}, where lm_eval runs, and lm_eval reads "
                    f"a --tasks value that names a folder as a folder of task files")
        return False, why, out["text"]
    if not all(isinstance(t, str) for t in sel):
        return False, (f"lm_eval read task files from a folder or a path instead of finding "
                       f"{task} by name: {names}"), out["text"]
    if task not in names:
        return False, f"lm_eval chose {names}, not {task}", out["text"]
    return True, "", out["text"]


def gpqa_access(check=None) -> str:
    """one line: whether this server's HF account may read GPQA's dataset
    (its terms accepted), from Hugging Face's own check — or why it can't say"""
    try:
        if check is None:
            from huggingface_hub import auth_check as check
        check("Idavidrein/gpqa", repo_type="dataset")
    except ImportError:
        return "gpqa      not checked: this huggingface_hub has no auth_check"
    except Exception as e:                         # noqa: BLE001 — said in one line
        text = f"{type(e).__name__} {e}"
        if re.search(r"gated|authenticat|401|403|GatedRepo", text, re.I):
            return "gpqa      " + config.GPQA_GATED
        return f"gpqa      not checked: {type(e).__name__}"
    return "gpqa      this server's HF account can read GPQA Diamond"


def main() -> int:
    try:
        cli, version = _harness()
    except ImportError as e:
        print(f"lm_eval is not importable here ({e}). This runs inside the bench container: "
              f"sudo docker compose exec -T bench python scripts/check_tasks.py")
        return 2
    print(f"lm_eval {version} · BENCH_ROOT {config.BENCH_ROOT}")
    found = total = 0
    failed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="check-tasks-") as tmp:
        # Everyday's task, built from the deployed bank as every run builds it
        import everyday
        config.EVERYDAY_TASKS_DIR = everyday.build_task(Path(tmp) / "everyday-task")
        # 12k.2: Trust & safety's four, from the pinned files, as a run builds them
        import trust_safety
        config.TRUST_TASKS_DIR = trust_safety.build_tasks(Path(tmp) / "trust-safety-tasks")
        # 12n.2: SimpleQA Verified, from the pinned file
        import simpleqa
        config.SIMPLEQA_TASKS_DIR = simpleqa.build_tasks(Path(tmp) / "simpleqa-task")
        # 12o.3: MobileAIBench's sets, from the pinned files (14.1: Dolly,
        # CNN/DailyMail, XSum and MT-Bench's two turns too)
        import mobileaibench
        config.MAB_TASKS_DIR = mobileaibench.build_tasks(Path(tmp) / "mobileaibench-tasks")
        # 14.3: Mobile-MMLU-Pro, from the fetched file — a stand-in with no
        # questions before the data step has fetched it
        import mobile_mmlu
        config.MMP_TASKS_DIR = mobile_mmlu.build_tasks(Path(tmp) / "mobile-mmlu-tasks")
        # 14.4: and the full Mobile-MMLU's, likewise
        config.MMF_TASKS_DIR = mobile_mmlu.build_full_tasks(Path(tmp) / "mobile-mmlu-full-tasks")
        # 12q: DeviceMark's three, with a stand-in for each question: only whether
        # lm_eval finds the task is asked, and no dataset is fetched for that
        import devicemark
        config.DM_TASKS_DIR = devicemark.build_tasks(Path(tmp) / "devicemark-tasks",
                                                     devicemark.stand_in_items())
        seen: dict[str, tuple[bool, str, str]] = {}
        # the full suite's second choice, all of BBQ, is checked as its own line
        # the mobile suite's judged part (14.1) and the full suite's second choice,
        # all of BBQ, are checked as lines of their own
        for suite, tasks in [(s, config.tasks_for_suite(s)) for s in config.SUITES] + [
                ("mobile, judged", config.tasks_for_suite("mobile", part="judged")),
                ("mobile, trust", config.tasks_for_suite("mobile", part="trust")),
                ("mobile, mmlu", config.tasks_for_suite("mobile", part="mmlu")),
                ("mobile, mmlu_full", config.tasks_for_suite("mobile", part="mmlu_full")),
                ("full, all of BBQ", [config.BBQ_ALL_TASK])]:
            if not tasks:
                note = (" — the exam has not been built (scripts/exam_build.py)"
                        if suite == "judged" else "")
                print(f"{suite:<9} no tasks{note}")
                continue
            ok = 0
            for task in tasks:
                if task not in seen:
                    seen[task] = find(cli, task, Path(tmp) / "out")
                good, why, text = seen[task]
                ok += good
                if not good and task not in failed:
                    failed.append(task)
                    print(f"  {task}: {why}")
                    tail = [ln for ln in text.strip().splitlines() if ln.strip()][-8:]
                    for ln in tail:
                        print(f"    | {ln}")
            # 14.1: MobileAIBench's lines name their sets
            print(f"{suite:<9} {ok} of {len(tasks)} found"
                  + (f" ({', '.join(tasks)})" if suite.startswith("mobile") else ""))
        found = sum(1 for good, _, _ in seen.values() if good)
        total = len(seen)
    # 12n.2: GPQA's dataset is gated — finding its tasks says nothing of whether
    # this server's HF account may read it; this asks Hugging Face
    print(gpqa_access())
    if failed:
        print(f"tasks FAILED: {found} of {total} found by lm_eval {version} — "
              f"not found: {', '.join(failed)}")
        return 1
    print(f"tasks OK: {found} of {total} found by lm_eval {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
