# docs — reading order

Start at **[00-orientation.md](00-orientation.md)**. It contains the map, the
vocabulary, and the three things to internalise before your first team meeting.

| # | Doc | Answers | Code it maps to |
|---|-----|---------|-----------------|
| 00 | [orientation](00-orientation.md) | How does all of this connect? What do the words mean? | — |
| 01 | [how-llms-work](01-how-llms-work.md) | What is the model actually computing? | `model.py` |
| 02 | [tokenization-and-data](02-tokenization-and-data.md) | How does text become training data, and how does that break? | `tokenizer.py`, `data.py` |
| 03 | [training-mechanics](03-training-mechanics.md) | Batch size, forward, backward, optimizer, schedules, precision, distributed | `train.py` |
| 04 | [mixture-of-experts](04-mixture-of-experts.md) | Why MoE, what collapses, what to log, why it changes RL | `model.py:MoEFeedForward` |
| 05 | [post-training](05-post-training.md) | The whole post-training map and what each stage buys | `sft.py` |
| 06 | [rlhf](06-rlhf.md) | Reward models, PPO, DPO, RLAIF, RLVR — and where each fails | — |
| 07 | [grpo](07-grpo.md) | GRPO in full, DAPO / Dr. GRPO / GSPO, and an honest failed run | `grpo.py` |
| 08 | [labels-and-distillation](08-labels-and-distillation.md) | Soft vs hard labels; offline / online / on-policy KD | `distill.py` |
| 09 | [evaluation-and-benchmarking](09-evaluation-and-benchmarking.md) | How to produce numbers people can trust | `evaluate.py`, `registry.py` |
| 10 | [experiment-tracking-wandb](10-experiment-tracking-wandb.md) | W&B, sweeps, artifacts, team conventions | `tracking.py` |
| 11 | [the-pipeline-you-will-build](11-the-pipeline-you-will-build.md) | The design of your actual deliverable, and week-one questions | `pipeline.py` |
| 12 | [reading-list](12-reading-list.md) | The papers, ordered, with what to take from each | — |

## If you have one evening

00 (the map and the vocabulary) → 09 (your actual job) → 03 (the loop) → skim 07.

## If you have a weekend

All of them in order, running the commands as you go. The docs assume you have the
code open next to them; several sections are annotations on a specific function.

## The two demos worth running before you read anything

```bash
python scripts/demo_template_mismatch.py   # the #1 false alarm in eval work
python scripts/demo_data_bias.py           # a data bug that no aggregate metric shows
```

Each takes a few minutes and each demonstrates a class of bug that makes teams
distrust their own numbers.
