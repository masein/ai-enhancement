#!/usr/bin/env python3
"""
A hyperparameter sweep with no dependencies — the dependency-free twin of `wandb sweep`.

W&B sweeps work like this: you declare a search space in YAML (see configs/sweep.yaml),
`wandb sweep configs/sweep.yaml` registers it and prints a sweep ID, and
`wandb agent <id>` starts a worker that repeatedly asks the server for a config,
runs your script with it, and reports back. Run N agents and you get N-way
parallelism for free. Methods are grid, random, and bayes; bayes needs a declared
`metric` to optimise and is worth it when a single run is expensive.

This script is the same idea with a for-loop instead of a server: it is what you
run in CI, or when you want the sweep to be reproducible from a git checkout with
no account. Every trial lands in the registry, so `python -m aienh dashboard`
renders them together.

Run:  python scripts/sweep_local.py            (grid over lr x width)
      python scripts/sweep_local.py --steps 200
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aienh.dashboard import build_dashboard  # noqa: E402
from aienh.evaluate import DEFAULT_SUITE, RAW_TEMPLATE, run_suite  # noqa: E402
from aienh.model import ModelConfig  # noqa: E402
from aienh.registry import from_train_result, leaderboard, record, render_table  # noqa: E402
from aienh.train import TrainConfig, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--eval-n", type=int, default=150)
args = ap.parse_args()

# The search space. Keep it small and deliberate: a 2x3x4 grid is 24 runs, and 24
# runs of noise is worse than 4 runs you can interpret.
SPACE = {
    "lr": [1e-3, 3e-3],
    "n_embd": [96, 192],
}

trials = list(itertools.product(*SPACE.values()))
print(f"sweep: {len(trials)} trials over {list(SPACE)}  ({args.steps} steps each)")

for i, combo in enumerate(trials, 1):
    params = dict(zip(SPACE, combo))
    print(f"\n########## trial {i}/{len(trials)}: {params} ##########")
    mc = ModelConfig(n_layer=4, n_head=6, n_embd=params["n_embd"], block_size=64)
    tc = TrainConfig(corpus="arithmetic", n_docs=20000, block_size=64, steps=args.steps,
                     micro_batch_size=48, lr=params["lr"], tracker="local",
                     name_prefix="sweep", log_every=args.steps // 2,
                     eval_every=args.steps // 2, tags=["sweep"])
    r = train(mc, tc)
    kw = {t: {"n": args.eval_n} for t in DEFAULT_SUITE if not t.startswith("ppl_")}
    ev = run_suite(r["model"], r["tokenizer"], r["device"], tasks=DEFAULT_SUITE,
                   task_kwargs=kw, template=RAW_TEMPLATE)
    rec = from_train_result(r, kind="sweep", eval_out=ev)
    rec.notes = ", ".join(f"{k}={v}" for k, v in params.items())
    record(rec)

print("\n" + render_table(leaderboard()))
p = build_dashboard(out_path="artifacts/dashboard.html")
print(f"\ndashboard -> {p}")
print("\nRead the table, not just the top row: if the spread across trials is smaller "
      "than the standard error on the metric, the 'best' config is a coin flip.")
