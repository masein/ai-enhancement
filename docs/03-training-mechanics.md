# 03 — Training mechanics: batch, forward, backward, optimizer

Code: **`src/aienh/train.py`**. The four concepts in this doc's title are four
consecutive lines in its inner loop.

---

## The loop, verbatim

```python
for step in range(steps):
    lr = cosine_lr(step, ...)                  # 1. schedule
    for g in opt.param_groups: g["lr"] = lr
    opt.zero_grad(set_to_none=True)            # 2. clear last step's gradients

    for _ in range(grad_accum):
        x, y = get_batch(...)                  # 3. a micro-batch
        _, loss, aux = model(x, targets=y)     # 4. FORWARD
        (loss / grad_accum).backward()         # 5. BACKWARD (accumulates)

    gnorm = clip_grad_norm_(params, 1.0)       # 6. clip
    opt.step()                                 # 7. OPTIMIZER: update weights
```

Everything else in `train.py` is bookkeeping around those seven lines.

---

## Forward

Run the batch through the model, get logits `[B, T, V]`, compare to targets with
cross-entropy, reduce to **one scalar**. It must be a scalar, because "the gradient"
is only defined for a scalar-valued function.

Along the way PyTorch records every operation in an autograd graph. That graph is
why the forward pass costs memory proportional to activations, not just parameters,
and why `torch.no_grad()` during evaluation roughly halves memory.

## Backward

`loss.backward()` walks that graph in reverse applying the chain rule, and leaves
`d(loss)/d(param)` in `param.grad` for every parameter. Cost: roughly 2× the
forward pass — hence the `6 · params · tokens` FLOP estimate (2 forward, 4 backward).

Two PyTorch behaviours you must know:

**Gradients accumulate.** `.backward()` *adds* into `.grad`. This is convenient
(it is how gradient accumulation works) and dangerous (forget `zero_grad` and you
train on a running sum of every batch you have ever seen — the loss still goes down,
slowly and wrongly).

**`set_to_none=True`** frees the gradient tensors instead of filling them with
zeros. Slightly faster, meaningfully less memory.

## Optimizer

**AdamW.** Per parameter it keeps two running averages: the gradient (momentum,
`beta1`) and the squared gradient (variance, `beta2`). The update is
`lr · m̂ / (sqrt(v̂) + eps)` — every parameter gets its own effective step size,
which is what makes it work on the wildly different gradient scales inside a
transformer.

Costs **2 extra floats per parameter** in optimizer state. A 7B model in fp32:
28 GB weights + 56 GB optimizer state, before activations. This is why optimizer
state sharding (ZeRO / FSDP) exists.

Settings, and why:

- `beta2 = 0.95`, not the 0.999 default. LLM gradients are noisier than vision's;
  a shorter variance window adapts faster and is the near-universal LLM choice.
- `weight_decay = 0.1` on matrices **only**. Not on norm gains, biases, or
  embeddings — decaying those just fights the model's ability to represent scale.
  `model.py:configure_optimizer` splits the parameter groups on `p.dim() >= 2`,
  which is the standard shortcut for "is this a matrix".
- **"W" matters.** AdamW *decouples* weight decay from the adaptive step. Adam's
  original coupling made decay effectively adaptive, which is not what anyone wanted.

## Batch size

The number of sequences whose gradients you average before stepping.

Bigger batch → less noisy gradient → you can use a bigger learning rate → fewer
steps. But only up to the **critical batch size**, past which the gradient is
already accurate and you are spending FLOPs for nothing. Critical batch size grows
as the model and dataset grow, which is why frontier runs use batches of millions
of tokens and your laptop run should not.

Count in **tokens per step**, not sequences:

```
tokens/step = micro_batch_size × grad_accum × block_size × n_gpus
```

`train.py` prints this, plus total tokens and `tokens_per_param`, at the start of
every run. If `tokens_per_param` is ≈0.3, the model is undertrained and any
conclusion about architecture is unsupported.

### Gradient accumulation

You want an effective batch of 512 sequences; 16 fit in memory. Run 32 micro-batches
of 16, call `.backward()` on each, step once. Mathematically identical, ~32× less
memory, same total FLOPs.

**The one trap:** divide each micro-batch's loss by `grad_accum`. Otherwise the
accumulated gradient is a *sum* instead of a *mean*, and your effective learning
rate silently scales with `grad_accum` — so changing accumulation to fit a new GPU
changes your training dynamics.

---

## Learning rate: the schedule

The single most important hyperparameter, and the only one with a near-universal
recipe: **linear warmup, then cosine decay to a small floor.**

**Warmup** (first 0.5–5% of steps): at step 0 the weights are random, gradients are
large and badly conditioned, and a full-size step can wreck the run. Adam's variance
estimate is also near-zero, so its early steps are unreliable. Ramp in.

**Cosine decay**: big steps early to travel, small steps late to settle. Decay to
`0.1 × peak` rather than 0 — a dead tail wastes the last 10% of your budget.

Rough scaling: LR scales *down* with model width. A 1M-parameter model is happy at
3e-3; a 7B model wants ~3e-4; frontier models go lower still. If you take one
number from this doc: **when a run diverges, halve the learning rate first.**

`utils.py:cosine_lr` implements it; `tests/test_smoke.py:test_lr_schedule_shape`
asserts the shape.

---

## Gradient clipping

If the whole gradient vector's norm exceeds a threshold (1.0 is standard), rescale
it down. Cheap insurance against one pathological batch destroying hours of work.

**Watch `grad_norm` as a diagnostic, not just a safety net.** A spike in grad_norm
precedes a loss spike. If it is consistently pinned at the clip value, your learning
rate is too high or your data has a problem. `train.py` logs it every time.

---

## Precision

| dtype | exponent range | notes |
|---|---|---|
| fp32 | wide | the safe default; 4 bytes/param |
| **bf16** | same as fp32 | 2 bytes, no loss scaling needed. The modern standard. |
| fp16 | narrow | 2 bytes but underflows; needs a GradScaler. Legacy. |
| fp8 | very narrow | frontier-scale, needs careful per-tensor scaling |

**Mixed precision** in practice: parameters and optimizer state in fp32, forward and
backward in bf16, loss accumulation in fp32. `torch.autocast` handles the casting.
Roughly 2× throughput and half the activation memory on hardware with bf16 tensor
cores.

`utils.py:pick_dtype` chooses bf16 on CUDA when supported and fp32 elsewhere —
MPS bf16 support is uneven and CPU bf16 is usually *slower*, not faster.

---

## Distributed training, in one page

You will not run this on a laptop, but you must be able to follow the conversation.

**Data parallel (DDP)** — every GPU holds the full model, processes a different
micro-batch, and gradients are all-reduced (averaged) before the step.
Simple, and the effective batch multiplies by GPU count. Fails when the model
does not fit on one GPU.

**FSDP / ZeRO** — shard parameters, gradients, and optimizer state across GPUs;
gather each layer's parameters just in time for its forward. Trades communication
for memory. This is how most large models are trained today.

**Tensor parallel** — split individual matrices across GPUs. High communication, so
it stays inside one node.

**Pipeline parallel** — different layers on different GPUs, micro-batches flowing
through. Introduces "bubbles" (idle GPUs) that careful scheduling minimises.

**Expert parallel** — MoE-specific: different experts on different GPUs, tokens
routed across the network. The all-to-all communication is usually the bottleneck.

Frontier runs combine all of these ("4D parallelism"). The thing to remember for
your job: **the effective batch size is per-step across all devices**, so a
config that looks identical on 8 GPUs and 64 GPUs is training two different models.

---

## What to watch, in priority order

1. **loss** — should fall. A plateau at high loss usually means the LR is too low or
   the data is broken.
2. **grad_norm** — spikes precede instability.
3. **val loss and the train/val gap** — the gap opening up is memorisation.
4. **tokens/sec** — a sudden drop means something started swapping or a fallback
   kernel got hit.
5. **learning rate** — log it. Half of all "why is this run different" questions
   are answered by the LR curve.
6. **MoE balance** (doc 04) — a collapsing router is visible here long before it
   shows up in the loss.

---

## Reproducibility

`utils.py:set_seed` seeds python, numpy and torch. Same seed + same code + same
device = same numbers.

This is not hygiene, it is the foundation of your job: **if a run is not
reproducible you cannot attribute a score change to a model change.** Note also
that changing the device (CPU→GPU) or the batch size changes floating-point
reduction order, so bitwise reproducibility is per-configuration, not absolute. Say
so when you report.
