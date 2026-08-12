# 05 — Post-training: the map

Code: **`src/aienh/sft.py`**, **`src/aienh/grpo.py`**, **`src/aienh/distill.py`**.

Pretraining produces a model that *continues text*. Post-training turns it into
something that *does what you asked*. Same architecture, same forward pass, same
optimizer — different data and different objectives.

---

## The stages, and what each one actually buys

```
base model            continues text. "What is 2+2?" → "and other basic questions..."
   │
   ▼ SFT (supervised fine-tuning)
instruct model        answers in a format. Most of the "it's a chatbot now" delta.
   │                  Data: (prompt, ideal response) pairs. Objective: cross-entropy
   │                  on the response only.
   ▼ preference optimisation (RLHF / DPO)
aligned model         picks the response people prefer among plausible ones.
   │                  Data: (prompt, chosen, rejected). Tone, helpfulness, refusals.
   ▼ RLVR (GRPO and friends)
reasoning model       gets verifiable things RIGHT. Data: (prompt, checker).
                      Objective: maximise a computed reward. This is where the
                      recent step change in math/code benchmarks came from.
```

Each stage is smaller and more expensive per token than the one before:

| Stage | Data volume | Cost per example | What it changes |
|---|---|---|---|
| Pretraining | 1e12–1e13 tokens | ~free (scraped) | everything |
| SFT | 1e4–1e6 examples | expensive (written/curated) | format, task-following |
| Preference | 1e4–1e6 pairs | expensive (human comparisons) | which of several good answers |
| RLVR | 1e3–1e5 prompts | cheap prompts, expensive compute | correctness on checkable tasks |

---

## SFT in detail

The mechanical difference from pretraining is **one argument**:

```python
model(x, targets=y)                    # pretraining: loss on every token
model(x, targets=y, loss_mask=mask)    # SFT: loss on response tokens only
```

Everything else — forward, backward, AdamW, cosine schedule — is identical. See
`sft.py:encode_example`, which builds that mask, and
`tests/test_smoke.py:test_sft_mask_covers_response_and_eos_only`, which asserts it
covers the response and the EOS token and nothing else.

**Why mask the prompt.** You want `P(response | prompt)`, not `P(prompt)`. Training
on prompt tokens is not catastrophic — it acts as extra language modelling — but it
dilutes the signal, and on templated data it teaches the model to invent new
questions. `mask_prompt=False` is exposed so you can measure the difference rather
than believe me.

**EOS is inside the mask.** The model must learn to *stop*. Omit it and generations
ramble past the answer forever, which surfaces as parse failures in eval, not as a
training bug.

**Throughput looks terrible and that is expected.** `sft.py` logs
`loss_token_frac` — the share of each batch that contributes to the loss. Measured
in this repo it is ~22%: the prompt is most of every sequence. You are paying for
four times the forward passes you get gradient from. This is normal, and it is why
some teams pack multiple SFT examples per sequence with a block-diagonal mask.

**Hyperparameters, and the reasons:**

- `lr` ≈ 1/10 of pretraining. You are nudging a trained model. Too high causes
  **catastrophic forgetting** — the model learns your format while its general
  ability degrades, which your narrow eval suite will not catch. Keep a general
  capability metric in the suite specifically to detect this.
- `epochs` 1–3 normally. SFT sets are small enough to overfit fast.
- `weight_decay` 0 or tiny for short fine-tunes.
- **The template is a contract.** Whatever you train on is what you must use at
  inference and in evaluation. See below.

### The template trap (measure this once, remember it forever)

A model SFT'd on `"Q: {q}\nA:"` and then evaluated with a raw prompt scores badly
for a reason that has nothing to do with capability. It looks exactly like a
regression on a dashboard.

```bash
python scripts/demo_template_mismatch.py
```

The repo defends against this structurally: `evaluate.suite_hash()` includes the
prompt template, so a score produced under a different template gets a different
suite hash, and the dashboard refuses to put two suite hashes in one table.

---

## Parameter-efficient fine-tuning (LoRA), briefly

Full fine-tuning updates every weight and needs optimizer state for all of them.
**LoRA** freezes the base weights and learns a low-rank update `ΔW = BA` with
`rank ≈ 8–64`, training <1% of the parameters. Two consequences worth knowing:

- Memory drops enormously (optimizer state is on the adapter only), so a 70B model
  fine-tunes on one GPU. QLoRA adds 4-bit base weights on top of that.
- You can serve one base model with many swappable adapters, which is the real
  production argument.

For your job: **an adapter is not a checkpoint.** An eval result must record which
base model an adapter was applied to, or it is unreproducible. This is a common
registry design mistake.

---

## What "aligned" is actually trading against

Every post-training stage has a cost, and knowing the costs is how you interpret an
eval regression instead of panicking about it:

- **The alignment tax.** Post-training often *lowers* raw benchmark scores on
  narrow capability tests while making the model far more useful. Whether that is
  a regression depends on which number your team is optimising, and that is a
  product decision, not a measurement.
- **Mode collapse.** Preference optimisation and RL both reduce output diversity.
  Fine for math, bad for creative writing. If your suite has no diversity metric,
  this happens invisibly.
- **Sycophancy.** Reward models trained on human preference learn that agreement is
  preferred. This is a well-documented, structural effect of the objective, not a
  bug in any particular implementation.
- **Refusal drift** in both directions — over-refusing benign requests, or
  loosening on genuinely harmful ones. Needs its own eval set.

---

## Where distillation fits

Distillation is orthogonal to the ladder: it is how you get a *smaller* model with
similar behaviour, and it can be applied at any stage. In practice, teams:

1. Post-train a large model properly.
2. Distil it into the sizes they actually want to serve.
3. Optionally re-run a short RL stage on the student.

Doc 08 covers soft vs hard labels and the offline/online/on-policy distinction.

---

## Two results from this repo, both worth internalising

Both measured on the arithmetic task with a 1.8M-parameter model, so treat the
magnitudes as illustrative and the *directions* as the lesson.

**1. SFT concentrates the training signal, and that is worth a lot.** The base model
trained on packed `a + b = c` documents spends most of its loss on the operand
tokens, which are random by construction and therefore unlearnable. SFT with a
response-only mask puts every gradient on the tokens you care about. Measured:
exact-match went from ~5% (base) to ~99% (SFT, 12 epochs) on problems the model had
seen. Same architecture, same data distribution — only the loss mask and the epoch
count changed.

**2. The GRPO stage in this repo does not improve held-out accuracy — in four
different configurations.** Starting points from 8.5% to 98.5% exact-match, learning
rates from 3e-5 to 2e-4, 100 to 400 iterations: flat or worse every time. The full
table and the diagnosis are in [doc 07](07-grpo.md#what-actually-happened-when-i-ran-it--the-honest-result).

I am leaving that in rather than tuning until it looks good, because the diagnosis is
the transferable part, and because "the RL run did nothing, work out why" is a
realistic description of the job. Short version: the update direction is verifiably
correct (there is a test), but at 8% accuracy 4–7 of every 8 groups were dropped as
zero-variance, and 3,200 prompt-visits of scalar reward is a far weaker signal than
the 14,000 densely-supervised example-visits the SFT stage it started from used.

The rule that generalises: **RL needs headroom, and the way you measure headroom is
the pass@1 → pass@k gap.** Measure both before starting. A large gap means the
capability is present but unreliable, which is what RL fixes. A gap near zero — at
either end of the range — means RL can only add noise.
`evaluate.py:task_arith_pass_at_k` exists for this measurement, and handing it to an
RL engineer before they burn a week of GPU time is probably the highest-leverage
single thing an eval owner does.

And the corollary for your pipeline: **evaluate a held-out metric *during* the RL
run, not after it.** `grpo.py` does this every `eval_every` iterations and prints a
warning when held-out accuracy drops more than 3 standard errors below the run's own
best. Reward is a claim about the reward function; only a held-out metric is a claim
about the model.
