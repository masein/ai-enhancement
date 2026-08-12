#!/usr/bin/env python3
"""
The five W&B calls that matter, in one runnable file.

    wandb.init()      start a run, freeze its config
    run.log()         a point on a time series
    run.summary[]     a final scalar — what run tables and leaderboards sort on
    wandb.Table       sample-level rows (prompt / generation / reward)
    wandb.Artifact    a versioned file, with lineage

Run it two ways:

    AIENH_TRACKER=local python scripts/wandb_example.py     # no account needed
    wandb login && python scripts/wandb_example.py          # real W&B

The repo's Tracker wrapper (src/aienh/tracking.py) is what makes the first line
work: identical call sites, JSONL on disk instead of the network. Do that in CI —
a test suite that needs an API key is a test suite that gets disabled.

Notes for team practice, learned the expensive way:

  * Log `step` consistently — optimizer steps, not epochs, not wall-clock — or
    charts from two runs will not overlay.
  * Put the whole config in `config`, including data version and git SHA. A run
    whose inputs you cannot reconstruct is an anecdote.
  * Use `group` for runs that belong to one experiment and `tags` for slicing.
    Then the run table does the work instead of a spreadsheet.
  * Log a Table of raw samples every time. Aggregates say a number moved; samples
    say why, and about half the time the answer is "the parser broke".
  * Artifacts give you lineage (dataset -> checkpoint -> eval). When someone
    challenges a benchmark number six weeks later, lineage is the only thing that
    settles it.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aienh.tracking import Tracker  # noqa: E402

config = {
    "model": {"n_layer": 4, "n_embd": 192, "moe": False},
    "optim": {"lr": 3e-3, "batch_size": 48, "warmup_frac": 0.05},
    "data": {"corpus": "arithmetic", "version": "v1", "n_docs": 20000},
    "git_sha": "abc1234",
}

with Tracker(project="aienh-demo", name="wandb-example", config=config,
             tags=["demo"], group="tracking-tutorial") as tr:
    for step in range(100):
        loss = 3.0 * math.exp(-step / 30) + 0.35 + 0.02 * math.sin(step)
        tr.log({"train/loss": loss,
                "train/ppl": math.exp(loss),
                "train/lr": 3e-3 * min(1.0, (step + 1) / 5)}, step=step)
        if step % 25 == 0:
            tr.log({"eval/val_loss": loss + 0.04}, step=step)

    tr.set_summary({"eval/final_val_loss": loss + 0.04, "eval/points": 71.4})

    tr.log_table(
        "samples",
        ["prompt", "generation", "gold", "correct"],
        [["17 + 45 =", " 62", 62, True],
         ["8 + 9 =", " 17", 17, True],
         ["73 + 68 =", " 131", 141, False]],
    )

    ckpt = Path("runs/wandb-example/fake_model.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not a real checkpoint")
    tr.log_artifact(ckpt, name="wandb-example-ckpt", kind="model")

print("\nBackend was chosen automatically: W&B if credentials exist, else local JSONL.")
print("Local output: runs/wandb-example/{config.json,metrics.jsonl,summary.json,samples.json}")
