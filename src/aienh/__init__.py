"""
aienh — a teaching-grade but real LLM training + evaluation pipeline.

Read docs/00-orientation.md first. Every module here maps to one concept:

    tokenizer.py  text  -> integers            (char + byte-level BPE)
    data.py       raw   -> cleaned -> packed   (the preprocessing pipeline)
    model.py      integers -> logits           (dense transformer + MoE)
    train.py      logits -> gradients -> new weights   (pretraining)
    sft.py        same loop, masked labels     (post-training step 1)
    distill.py    teacher distribution -> student      (soft vs hard labels)
    grpo.py       reward -> group advantage -> policy update  (RL post-training)
    evaluate.py   model -> numbers             (the benchmark harness)
    registry.py   numbers -> a leaderboard row (names, configs, points)
    dashboard.py  leaderboard -> HTML          (the automation deliverable)
    tracking.py   anything -> W&B (or a local jsonl fallback)
"""

__version__ = "0.1.0"
