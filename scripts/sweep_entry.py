#!/usr/bin/env python3
"""
Entry point for `wandb agent` (see configs/sweep.yaml).

The agent launches this script once per trial with the sampled hyperparameters on
the command line. wandb.init() picks them up in `wandb.config`; you read them from
there rather than from argparse, which is the one non-obvious part of the API.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import wandb
except ImportError:
    raise SystemExit("this script is for `wandb agent`; run scripts/sweep_local.py instead")

from aienh.evaluate import DEFAULT_SUITE, RAW_TEMPLATE, run_suite
from aienh.model import ModelConfig
from aienh.train import TrainConfig, train

run = wandb.init()
c = wandb.config

mc = ModelConfig(n_layer=4, n_head=6, n_embd=c.n_embd, block_size=64,
                 moe=bool(c.get("moe", False)), n_experts=8, top_k=2)
tc = TrainConfig(corpus="arithmetic", n_docs=20000, block_size=64, steps=800,
                 micro_batch_size=c.micro_batch_size, lr=c.lr,
                 tracker="local", run_name=run.name, log_every=50)
r = train(mc, tc)

kw = {t: {"n": 200} for t in DEFAULT_SUITE if not t.startswith("ppl_")}
ev = run_suite(r["model"], r["tokenizer"], r["device"], tasks=DEFAULT_SUITE,
               task_kwargs=kw, template=RAW_TEMPLATE)
# The sweep optimises whatever is in summary under `metric.name`.
run.summary["eval/points"] = ev["points"]
run.summary["eval/final_val_ppl"] = r["summary"]["eval/final_val_ppl"]
run.finish()
