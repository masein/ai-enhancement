import argparse
import glob
import sys
from pathlib import Path


def cli_evaluate():
    p = argparse.ArgumentParser(prog="lm_eval")
    p.add_argument("--tasks")
    p.add_argument("--include_path", default=None)
    args, _ = p.parse_known_args(sys.argv[1:])
    from lm_eval.tasks import TaskManager, names_in
    tm = TaskManager(include_path=args.include_path)
    task_list = args.tasks.split(",")
    if len(task_list) == 1 and Path(task_list[0]).is_dir():
        names = [{"task": sorted(names_in(Path(y).read_text()))[0]}
                 for y in glob.glob(str(Path(task_list[0]) / "*.yaml"))]
    else:
        names = tm.match_tasks(task_list)
        missing = [t for t in task_list if t not in names]
        if missing:
            raise ValueError(f"Tasks not found: {', '.join(missing)}")
    print(f"Selected Tasks: {names}", file=sys.stderr)
    from lm_eval import simple_evaluate
    simple_evaluate(model="hf", tasks=names, task_manager=tm)


if __name__ == "__main__":
    cli_evaluate()
