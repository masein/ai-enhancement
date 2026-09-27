"""A fake llama-perplexity: it reads the files and flags the real one does and
prints what it prints — the lines perplexity.cpp writes, in their formats — with
canned accuracies. No model runs.

Its environment sets the rest:
- FAKE_PPL_ACC: the fraction it gets right (0.6);
- FAKE_PPL_DELAY: seconds a task (0);
- FAKE_PPL_FAIL: print a context-window error and no score;
- FAKE_PPL_LOCK: a path; "lock held" is printed while it exists.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print("version: 9999 (91428471f)", file=sys.stderr)
        print("built with fake for tests", file=sys.stderr)
        return 0
    arg = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("-")}
    model = arg.get("-m", "")
    if not os.path.isfile(model):
        print(f"llama_model_load: error loading model: {model}", file=sys.stderr)
        print("main: unable to load model", file=sys.stderr)
        return 1
    mode = next(m for m in ("--hellaswag", "--winogrande", "--multiple-choice") if m in argv)
    path = arg.get("-bf") if mode == "--multiple-choice" else arg.get("-f")
    data = open(path, "rb").read()
    if mode == "--hellaswag":
        text = data.decode().rstrip("\n")
        count = len(text.split("\n")) // 6
        want = int(arg.get("--hellaswag-tasks", 400))
    elif mode == "--winogrande":
        count = len([x for x in data.decode().split("\n") if x.strip()])
        want = int(arg.get("--winogrande-tasks", 0))
    else:
        (count,) = struct.unpack_from("<I", data, 0)
        want = int(arg.get("--multiple-choice-tasks", 0))
    n = count if not want or want >= count else want
    acc = float(os.environ.get("FAKE_PPL_ACC", "0.6"))
    delay = float(os.environ.get("FAKE_PPL_DELAY", "0"))
    lock = os.environ.get("FAKE_PPL_LOCK", "")
    out = sys.stdout
    if lock and os.path.isdir(lock):
        print("lock held", file=sys.stderr)
    name = {"--hellaswag": "hellaswag_score", "--winogrande": "winogrande_score",
            "--multiple-choice": "multiple_choice_score"}[mode]
    if mode == "--multiple-choice":
        print(f"{name}: there are {count} tasks in prompt", file=sys.stderr)
        if n < count:
            print(f"{name}: selecting {n} random tasks from {count} tasks available",
                  file=sys.stderr)
    else:
        print(f"{name} : loaded {count} tasks from prompt.", file=sys.stderr)
        if mode == "--hellaswag":
            print(f"{name} : selecting {n} randomized tasks.", file=sys.stderr)
        elif n < count:
            print(f"{name} : selecting {n} random tasks", file=sys.stderr)
    if os.environ.get("FAKE_PPL_FAIL"):
        print(f"{name} : task 0 does not fit in the context window (requires 5000 tokens)",
              file=sys.stderr)
        return 0
    print("\ntask\tacc_norm" + ("\t95% confidence interval" if mode == "--hellaswag" else ""),
          file=out, flush=True)
    right = 0
    for i in range(1, n + 1):
        if delay:
            time.sleep(delay)
        if round(acc * i) > right:
            right += 1
        p = right / i
        if mode == "--hellaswag":
            print(f"{i}\t{100 * p:3.8f}%\t[{100 * max(0, p - 0.1):3.4f}%, "
                  f"{100 * min(1, p + 0.1):3.4f}%]", file=out, flush=True)
        elif mode == "--winogrande":
            print(f"{i}\t{100 * p:.4f}\t{-1.5:10.6f}  {-2.5:10.6f}  1  1", file=out, flush=True)
        else:
            print(f"{i}\t{100 * p:.8f}", file=out, flush=True)
    print("", file=out, flush=True)
    sigma = math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else 0.0
    if mode == "--winogrande" and n >= 100:
        print(f"Final Winogrande score({n} tasks): {100 * p:.4f} +/- {100 * sigma:.4f}",
              file=sys.stderr)
    if mode == "--multiple-choice" and (n >= 100 or n == count):
        print(f"Final result: {100 * p:.4f} +/- {100 * sigma:.4f}", file=sys.stderr)
        print(f"Random chance: {25.0:.4f} +/- {1.0:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
