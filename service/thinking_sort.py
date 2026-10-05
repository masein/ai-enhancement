"""16b: answers made before the thinking rule, sorted once at deploy.

Before 16b the Mobile suite asked a model with its chat template's default —
thinking on, for a Qwen3 — on Hugging Face and over a server alike, and
Instruction & maths and Frontier asked a served model as its server does.
Those answers sit in the model's own row, which now holds thinking-off
answers only (runner.THINKING_RULE). This moves each task's thinking-on
answers to the model's thinking row ("· thinking") when that row has none for
the task, else beside the tree (results/earlier/<row>/<task>-thinking-on-
<time>), with the judge's verdicts on them, and makes the rows' Mobile scores
again from what is left. A served model's Instruction & maths and Frontier
answers came through lm_eval, which never saw the thinking (its server keeps
it apart): theirs is what the model was registered with, and "the model
decides" is not known to be off, so they go beside the tree too
(…-thinking-default-<time>). Nothing is asked, nothing is deleted. A run does
the same for itself as it starts a task; this is for the rows no run will
touch soon.

A dry run first, which prints what it would move; then --apply, with the
queue idle:

    sudo docker compose exec -T bench python -m service.thinking_sort
    sudo docker compose exec -T bench python -m service.thinking_sort --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import config, db, runner
from . import served as _served

MOBILE = [*config.MAB_ALL, config.MMP_TASK, config.MMF_TASK]
# 12h.1 asked a Hugging Face model's I&M and Frontier with its thinking off
# already: only a served model's were its server's default
SERVED_TOO = [*config.GEN_TASKS, *config.SHARED_TASKS]


def _meta(row: Path) -> dict:
    try:
        return json.loads((row / "model_meta.json").read_text())
    except (OSError, ValueError):
        return {}


def tasks_of(row: Path) -> tuple[str, list[str], dict | None] | None:
    """(model, the tasks the rule decides, its served record) for a model's
    own row — None for a thinking row, or a model that can't turn its
    thinking off"""
    if row.name.endswith("__thinking") or not row.is_dir():
        return None
    meta = _meta(row)
    mid = meta.get("model")
    if not mid:
        return None
    rec = _served.get(mid) if str(mid).startswith("served/") else None
    if rec:
        return None if _served.is_openrouter(rec) else (mid, MOBILE + SERVED_TOO, rec)
    return (mid, MOBILE, None) if meta.get("thinking") == "switch" else None


def _task_dirs(row: Path, task: str) -> list[Path]:
    """a task's answers in a row: "<task>_<shots>shot", as a run names them"""
    shot = re.compile(re.escape(task) + r"_\d+shot")
    return sorted(p for p in row.iterdir() if p.is_dir() and shot.fullmatch(p.name))


def plan(out_dir: Path | None = None) -> list[dict]:
    """each task in a model's own row whose answers on disk were made with
    the thinking on, or as its server decided: where it would go"""
    out_dir = out_dir or config.OUT_DIR
    found = []
    for row in sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.is_dir() else []:
        got = tasks_of(row)
        if not got:
            continue
        mid, tasks, rec = got
        think_row = row.with_name(row.name + "__thinking")
        for task in tasks:
            # through lm_eval: the generative three and GPQA's chain of thought
            unmarked = runner.unmarked_of(rec, task in SERVED_TOO
                                          and task != config.SIMPLEQA_TASK)
            for task_out in _task_dirs(row, task):
                was = runner.answered_thinking(task_out, unmarked)
                if was in (None, "off"):
                    continue
                found.append({"model": mid, "row": row, "task": task, "task_out": task_out,
                              "was": was, "unmarked": unmarked, "thinking_row": think_row,
                              "to": "thinking row" if was == "on"
                              and not (think_row / task_out.name).exists() else "earlier"})
    return found


def apply(found: list[dict]) -> list[tuple[dict, tuple[str, Path] | None]]:
    done = []
    for f in found:
        meta = _meta(f["row"])
        meta_think = {**meta, "model": f["model"] + " · thinking", "base_model": f["model"]}
        done.append((f, runner.sort_thinking(f["task_out"], f["task"], False, f["thinking_row"],
                                             meta_think, f["unmarked"])))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="move them (without it, a dry run)")
    a = ap.parse_args(argv)
    db.init()
    found = plan()
    if not found:
        print("Nothing to sort: every model's own row holds thinking-off answers only.")
        return 0
    for f in found:
        print(f"{f['model']} · {f['task']}: "
              + ("thinking on" if f["was"] == "on" else "as its server decided") + " → "
              + ("its thinking row" if f["to"] == "thinking row"
                 else "earlier (its thinking row has answers of its own)" if f["was"] == "on"
                 else "earlier (not known to be thinking off)"))
    if not a.apply:
        print(f"\n{len(found)} task(s) to sort. A dry run: nothing moved. Again with --apply, "
              "with the queue idle.")
        return 0
    busy = [s["id"] for s in db.recent(200) if s.get("status") in db.RUNNING]
    if busy:
        print(f"Not now: run {', '.join(f'#{i}' for i in busy)} is going. Again when the queue "
              "is idle.", file=sys.stderr)
        return 2
    for f, got in apply(found):
        print(f"moved {f['model']} · {f['task']} to {got[1]}" if got
              else f"{f['model']} · {f['task']}: already sorted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
