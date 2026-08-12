# 00 — Orientation: the map

You are joining a team that trains a model. Your job is the pipeline that tells
everyone whether the model got better. That job sits downstream of every decision
anyone else makes, which means you need a working model of the whole stack — not
to build it, but to know what each number you produce is actually measuring.

Read this file first. It exists so the other twelve make sense.

---

## The one-paragraph version of the whole field

A language model is a function that takes a sequence of tokens and returns a
probability distribution over the next token. You train it by showing it enormous
amounts of text and adjusting its weights so the token that actually came next
gets higher probability (**pretraining**). That gives you something that continues
text. You then reshape it: teach it to answer rather than continue
(**supervised fine-tuning**), teach it which answers people prefer
(**RLHF/DPO**), and teach it to succeed at things you can check automatically
(**RLVR / GRPO**). Along the way you may make it cheaper by having a big model
teach a small one (**distillation**) or by making only part of the network run
per token (**mixture of experts**). Everything else — data pipelines, eval
harnesses, dashboards — exists because none of the above is observable without
measurement.

---

## The pipeline, and where each topic lives

```
                    ┌─────────────────────────────────────────────┐
  raw text ────────▶│  DATA PIPELINE                              │ doc 02
                    │  normalise · filter · dedup · mix · pack    │
                    └───────────────────┬─────────────────────────┘
                                        │ tokens
                    ┌───────────────────▼─────────────────────────┐
                    │  PRETRAINING                                │ doc 01, 03
                    │  transformer · attention · MoE              │ doc 04
                    │  batch · forward · backward · optimizer     │
                    └───────────────────┬─────────────────────────┘
                                        │ base model
                    ┌───────────────────▼─────────────────────────┐
                    │  POST-TRAINING                              │ doc 05
                    │  SFT → preference (RLHF/DPO) → RLVR (GRPO)  │ doc 06, 07
                    │  distillation, soft vs hard labels          │ doc 08
                    └───────────────────┬─────────────────────────┘
                                        │ shipped model
                    ┌───────────────────▼─────────────────────────┐
  ★ YOUR JOB ──────▶│  EVALUATION + TRACKING                      │ doc 09, 10
                    │  harness · benchmarks · registry · dashboard│ doc 11
                    └─────────────────────────────────────────────┘
```

| # | Doc | Answers |
|---|-----|---------|
| 01 | [how-llms-work](01-how-llms-work.md) | What is the model actually computing? |
| 02 | [tokenization-and-data](02-tokenization-and-data.md) | How does text become training data, and how does that step break? |
| 03 | [training-mechanics](03-training-mechanics.md) | Batch size, forward, backward, optimizer, schedules, precision |
| 04 | [mixture-of-experts](04-mixture-of-experts.md) | Why MoE, what breaks, what to log |
| 05 | [post-training](05-post-training.md) | The whole post-training map and what each stage buys |
| 06 | [rlhf](06-rlhf.md) | Reward models, PPO, DPO, and where RLHF actually fails |
| 07 | [grpo](07-grpo.md) | GRPO in full, plus DAPO / Dr. GRPO / GSPO |
| 08 | [labels-and-distillation](08-labels-and-distillation.md) | Soft vs hard labels; offline / online / on-policy KD |
| 09 | [evaluation-and-benchmarking](09-evaluation-and-benchmarking.md) | How to produce numbers people can trust |
| 10 | [experiment-tracking-wandb](10-experiment-tracking-wandb.md) | W&B, sweeps, artifacts, team conventions |
| 11 | [the-pipeline-you-will-build](11-the-pipeline-you-will-build.md) | The design of your actual deliverable |
| 12 | [reading-list](12-reading-list.md) | The papers, ordered, with what to take from each |

---

## Read code, not just prose

Every doc points at a file in `src/aienh/`, and the files are written to be read.
The fastest way to stop being confused about attention is to read
`model.py:CausalSelfAttention` — it is 30 lines. The fastest way to understand
GRPO is `grpo.py`, where the objective is 15 lines of tensor code with the paper's
notation in the comments.

Start here, in this order:

```bash
python -m aienh data --corpus dirty          # watch each cleaning stage fire
python -m aienh.model                        # dense vs MoE parameter counts
python -m aienh pipeline --scale smoke       # the whole thing, ~3 minutes
open artifacts/dashboard.html
```

---

## The vocabulary, so you can follow a standup on day one

**token** — the unit the model consumes. Not a word: roughly 0.75 words in English,
one to three characters in code. Everything is counted in tokens.

**context / block size / sequence length** — how many tokens the model can attend
over at once. Costs memory quadratically in the naive implementation.

**embedding dimension / d_model / width** — the size of the vector representing
each position. `n_embd` in this repo.

**logits** — the raw pre-softmax scores over the vocabulary, one per token. The
model's actual output.

**loss** — cross-entropy: `-log P(correct next token)`, averaged over tokens.
**perplexity** = `exp(loss)`, read as "how many equally-likely options was it
effectively choosing between".

**step / iteration** — one optimizer update. The x-axis of every chart. Not an epoch.

**epoch** — one pass over the dataset. Nearly meaningless in pretraining (you do
less than one) and important in fine-tuning (you do 1–3).

**effective batch size** — sequences per optimizer step, *after* gradient
accumulation and across all GPUs. The number that matters; `micro_batch_size` is
just what fits in memory.

**base model** — pretrained only. Continues text; does not answer questions.

**instruct / chat model** — post-trained. Answers in a format.

**checkpoint** — saved weights (+ config + tokenizer, if whoever saved it was
careful).

**active vs total parameters** — identical for dense models; wildly different for
MoE. Quote both or you are misleading someone.

**FLOPs** — the compute unit. A useful approximation for a dense transformer:
forward+backward ≈ `6 × params × tokens`.

**KV cache** — cached keys/values so generating token *n+1* does not recompute the
whole prefix. Turns generation from O(n²) into O(n) per token, at a memory cost.

**temperature / top-k / top-p** — decoding controls. Temperature 0 (greedy) for
measurement; higher for diversity.

**teacher forcing** — training on the ground-truth prefix rather than the model's
own output. Standard, and the source of "exposure bias".

**rollout** — a sampled generation, in an RL context.

**advantage** — how much better a rollout was than a baseline. In GRPO the
baseline is the rollout's own group.

**reward hacking** — the model maximises your reward function without doing the
thing you wanted. Assume it will happen; log reward components separately so you
can see it.

**contamination** — evaluation data present in training data. Inflates scores;
your job to measure it.

**suite / harness** — the code + item set + prompt format that produces a score.
Change any of the three and the score is not comparable to yesterday's.

---

## Three things to internalise before your first meeting

**1. Almost every "the model regressed" alert is a harness bug.** Prompt template
changed, parser changed, decoding changed, item set changed, batch size changed
the padding. Check the harness before you check the model. `docs/09` is a list of
these in priority order.

**2. A number without `n` and a standard error is not a result.** At n=200 and
p≈0.5 the standard error is ±3.5 points. Two runs differing by 2 points are
indistinguishable. You will be the person in the room who knows this, and saying
it early saves weeks.

**3. Perplexity and task metrics answer different questions.** A concrete example
from this repo: on the arithmetic corpus, most of the loss lives in the operand
tokens, which are random by construction and therefore irreducible. The model can
sit at a perfectly respectable loss while getting the *answer* wrong most of the
time, because the answer is 2 of ~12 tokens. Loss is a training-health signal;
task metrics are the product signal. Report both, never substitute one for the other.
