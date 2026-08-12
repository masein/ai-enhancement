# ai-enhancement

A working LLM training + evaluation pipeline, small enough to read end to end and
real enough that every mechanism in it is the one production systems use.

Built as a learning path for joining a team that trains models, where the job is
**the pipeline that tells everyone whether the model got better**.

Two halves, meant to be used together:

- **`docs/`** — thirteen documents covering how LLMs work, MoE, data pipelines,
  post-training, RLHF, GRPO, distillation, evaluation, and W&B. Start at
  [`docs/00-orientation.md`](docs/00-orientation.md).
- **`src/aienh/`** — the same material as runnable code: a transformer (dense or
  MoE), a data pipeline with dedup and contamination checks, pretraining, SFT,
  distillation, GRPO, an eval harness, a model registry, and an HTML dashboard.

No GPU required. Everything runs on a laptop CPU; the arithmetic task is small
enough that a 1.8M-parameter model genuinely learns it in a few minutes, and every
corpus is generated rather than downloaded, so the whole repo works offline.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, numpy, pyyaml. wandb is optional.
```

On an Apple Silicon Mac the code uses the `mps` GPU backend automatically.

## Five minutes

```bash
export PYTHONPATH=src

python -m aienh data --corpus dirty      # the data pipeline, with per-stage drop rates
python -m aienh.model                    # dense vs MoE parameter accounting
python tests/test_smoke.py               # 37 tests of the maths, not the plumbing
python -m aienh pipeline --scale smoke   # the whole pipeline, ~3 minutes
open artifacts/dashboard.html
```

## The real run

```bash
python -m aienh pipeline --scale small   # ~20-40 min on a laptop
```

Pretrains a dense model and an MoE at matched budget, fine-tunes, runs GRPO,
distils a smaller student two ways, scores everything with one hashed eval suite,
and writes a self-contained HTML leaderboard.

---

## Layout

```
docs/            the written material — start with 00-orientation.md
src/aienh/
  tokenizer.py   char + byte-level BPE, trained from scratch
  data.py        corpora, normalise/filter/dedup/mix/split/pack, contamination checks
  model.py       decoder-only transformer; RoPE, RMSNorm, dense or MoE feed-forward
  train.py       the pretraining loop, with the four core steps spelled out
  sft.py         supervised fine-tuning (= pretraining + a loss mask)
  distill.py     soft/hard labels, offline / online / on-policy KD
  grpo.py        GRPO with DAPO, Dr. GRPO and GSPO variants behind flags
  evaluate.py    the eval harness: loglikelihood, generation, pass@k, slicing, stderr
  registry.py    append-only run registry: names, configs, points, lineage
  dashboard.py   registry -> one self-contained HTML file
  tracking.py    W&B wrapper with a local JSONL fallback
  pipeline.py    every stage wired together
  cli.py         one entry point for all of it
configs/         commented YAML configs — the unit of reproducibility
scripts/         demos of specific failure modes; sweeps
tests/           37 tests, aimed at the bugs that produce plausible wrong numbers
```

## CLI

```bash
python -m aienh data       --corpus dirty
python -m aienh train      --config configs/pretrain_dense.yaml
python -m aienh train      --moe --config configs/pretrain_moe.yaml
python -m aienh sft        runs/<run>/model.pt --config configs/sft.yaml
python -m aienh grpo       runs/<run>/model.pt --config configs/grpo.yaml
python -m aienh distill    runs/<run>/model.pt --mode offline
python -m aienh eval       runs/<run>/model.pt --tasks arith_exact,arith_mc4 --template chat
python -m aienh pipeline   --scale small
python -m aienh dashboard
python -m aienh leaderboard
python -m aienh sample     runs/<run>/model.pt --prompt "Q: 17 + 25 =" --greedy
```

Any config value can be overridden inline, flat or qualified — and an unknown key is
an error rather than a silent no-op:

```bash
python -m aienh train --set lr=1e-3 --set moe=true --set n_layer=6
python -m aienh train --set model.n_embd=256 --set train.steps=2000
python -m aienh train --set corpus='{"stories":0.8,"code":0.2}'
```

## Demos of specific failure modes

These exist because each one is a class of bug that makes teams distrust their own
numbers:

```bash
python scripts/demo_template_mismatch.py   # same model, two prompt formats, different scores
python scripts/demo_data_bias.py           # a filter threshold biases the data invisibly
python scripts/sweep_local.py              # a hyperparameter sweep, no account needed
python scripts/wandb_example.py            # the five W&B calls that matter
```

## Experiment tracking

W&B if you are logged in, local JSONL otherwise — same call sites either way:

```bash
wandb login && python -m aienh pipeline --scale small   # real W&B
AIENH_TRACKER=local python -m aienh pipeline            # no network at all
```

---

## The task, and why it is a good one

Two-digit addition: `Q: 17 + 25 =\nA: 42`.

It is deliberately trivial, because the *harness* is the subject, not the model. It
also has three properties real benchmarks want and rarely all have:

- **Verifiable.** No judge, no annotators — correctness is `==`.
- **Sliceable.** Accuracy by operand size exposes data bias that an aggregate hides.
- **Small enough to enforce a clean split.** The problem space is 2,500 items, so
  `data.arith_split` partitions the *space* by a stable hash and no training stage
  can see an eval problem. That is decontamination by construction rather than
  measured after the fact — which is what you would want at real scale and rarely get.

## What one 15-minute run actually shows

`python -m aienh pipeline --scale small`, measured on the held-out problem split:

| run | points | `arith_exact` | `ppl_stories` | note |
|---|---|---|---|---|
| pretrain, dense | 36.67 | 0.005 | 1.25 | 1.79M params |
| pretrain, MoE | 37.77 | 0.005 | 1.23 | 2.98× total params, **1.00× active/token** |
| + SFT | 48.43 | **0.365** | **78.32** | target task 73× better, general LM 63× worse |
| + GRPO | 44.13 | 0.270 | 47.48 | RL made it worse — see `docs/07` |
| distilled student | 31.17 | 0.180 | — | 231K params, 13% of the teacher |

Three things in that table are the point of the whole repo: the MoE row is a genuinely
FLOP-matched comparison (same active parameters per token, 3× the total), the SFT row
shows catastrophic forgetting that only an unrelated metric in the suite could catch,
and the GRPO row is a real negative result that the harness diagnoses rather than hides.

## What this repo is honest about

- The corpora are generated, not real text. The pipeline stages are real; the data is
  not web-scale and does not have web-scale problems.
- 1.8M parameters is a toy. Some things invert at this scale — most notably MoE is
  *slower* in wall clock here, because per-expert matmuls are small and routing
  overhead is not amortised. That inversion is real and worth seeing.
- The measured numbers quoted in `docs/` come from runs in this repo on a CPU. Treat
  the directions as the lesson and the magnitudes as illustrative.
- No LLM-as-judge, no distributed training, no KV cache. Each is noted where it would
  go, and why it was left out.
