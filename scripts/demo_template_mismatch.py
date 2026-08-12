#!/usr/bin/env python3
"""
Demo: the same model, the same questions, two prompt formats.

This is the most common false alarm in benchmarking work, and the cheapest one to
prevent. A model fine-tuned on "Q: {q}\\nA:" is a different function from the base
model it came from; prompt it raw and it will underperform for a reason that has
nothing to do with capability.

Run:  python scripts/demo_template_mismatch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aienh.evaluate import CHAT_TEMPLATE, RAW_TEMPLATE, run_suite  # noqa: E402
from aienh.model import ModelConfig  # noqa: E402
from aienh.sft import SFTConfig, sft  # noqa: E402
from aienh.train import TrainConfig, train  # noqa: E402

TASKS = ["arith_exact", "arith_mc4", "format_ok"]

base = train(
    ModelConfig(n_layer=4, n_head=6, n_embd=192, block_size=64),
    TrainConfig(corpus={"arithmetic": 0.8, "stories": 0.2}, n_docs=20000, block_size=64,
                steps=500, micro_batch_size=48, tracker="local", save=False,
                name_prefix="tmpl-base", log_every=250, eval_every=250),
)
tuned = sft(base["model"], base["tokenizer"],
            SFTConfig(n_examples=4000, epochs=3.0, tracker="local", save=False,
                      template=CHAT_TEMPLATE, log_every=100, eval_every=200),
            device=base["device"], parent=base["name"])

print("\n" + "=" * 72)
print("SAME CHECKPOINT, TWO PROMPT FORMATS")
print("=" * 72)
kw = {t: {"n": 200} for t in TASKS}
matched = run_suite(tuned["model"], tuned["tokenizer"], tuned["device"], tasks=TASKS,
                    task_kwargs=kw, template=CHAT_TEMPLATE, verbose=False)
mismatched = run_suite(tuned["model"], tuned["tokenizer"], tuned["device"], tasks=TASKS,
                       task_kwargs=kw, template=RAW_TEMPLATE, verbose=False)

print(f"\n{'task':<14} {'matched (Q:/A:)':>18} {'raw continuation':>18}   delta")
print("-" * 72)
for t in TASKS:
    a = matched["results"][t]["value"]
    b = mismatched["results"][t]["value"]
    print(f"{t:<14} {a:>18.4f} {b:>18.4f}   {b - a:+.4f}")
print(f"{'POINTS':<14} {matched['points']:>18.2f} {mismatched['points']:>18.2f}   "
      f"{mismatched['points'] - matched['points']:+.2f}")
print(f"\nsuite hashes differ ({matched['suite_hash']} vs {mismatched['suite_hash']}) — "
      "which is exactly why the template belongs in the hash.")
print("\nTakeaway: publish the template with the score, and never compare two numbers "
      "whose suite hashes differ.")
