#!/usr/bin/env python3
"""
Demo: a filter threshold silently biases a dataset, and one aggregate metric hides it.

Two identical models. The only difference is one number in the data pipeline:
`min_chars`, which prose corpora want at 12 and arithmetic wants at 5. With
min_chars=12, every "0 + 3 = 3" style document is dropped — so the training set is
systematically missing small operands.

Watch the OVERALL accuracy barely move while the small-operand slice collapses.
That gap is the entire argument for slicing your metrics.

Run:  python scripts/demo_data_bias.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aienh import data as D  # noqa: E402
from aienh.evaluate import task_arith_exact  # noqa: E402
from aienh.model import ModelConfig  # noqa: E402
from aienh.train import TrainConfig, train  # noqa: E402

STEPS = 500
results = {}

for label, preset in [("corpus-aware (min_chars=5)", {"min_chars": 5, "max_repetition": 0.9,
                                                      "min_alpha": 0.5}),
                      ("prose defaults (min_chars=12)", {"min_chars": 12, "max_repetition": 0.4,
                                                         "min_alpha": 0.6})]:
    print("\n" + "=" * 72)
    print(f"PIPELINE: {label}")
    print("=" * 72)
    saved = dict(D.FILTER_PRESETS)
    D.FILTER_PRESETS["arithmetic"] = preset
    try:
        r = train(
            ModelConfig(n_layer=4, n_head=6, n_embd=192, block_size=64),
            TrainConfig(corpus="arithmetic", n_docs=20000, block_size=64, steps=STEPS,
                        micro_batch_size=48, tracker="local", save=False,
                        name_prefix="bias", log_every=250, eval_every=250),
        )
        res = task_arith_exact(r["model"], r["tokenizer"], r["device"], n=300)
        results[label] = res
    finally:
        D.FILTER_PRESETS.clear()
        D.FILTER_PRESETS.update(saved)

print("\n" + "=" * 72)
print("THE AGGREGATE vs THE SLICES")
print("=" * 72)
print(f"{'pipeline':<32} {'overall':>9}   " +
      "  ".join(f"{k:>18}" for k in results[next(iter(results))].extra["by_operand_size"]))
for label, res in results.items():
    slices = "  ".join(f"{(v[0] if v[0] is not None else float('nan')):>18.3f}"
                       for v in res.extra["by_operand_size"].values())
    print(f"{label:<32} {res.value:>9.3f}   {slices}")
print("\nThe overall number is one column. The story is in the others.")
