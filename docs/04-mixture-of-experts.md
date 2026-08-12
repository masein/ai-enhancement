# 04 — Mixture of Experts

Code: **`src/aienh/model.py:MoEFeedForward`**.
Run: `python -m aienh.model` then `python -m aienh train --moe --config configs/pretrain_moe.yaml`

---

## The argument in one sentence

Replace the single MLP in each block with `E` MLPs plus a small router, and run only
the top `k` of them per token: you get `E/k`× the feed-forward parameters at
approximately the same FLOPs per token.

Since roughly two thirds of a dense transformer's parameters live in its MLPs
(doc 01), this is where the leverage is. Attention is left alone — it has to see
every position, so there is nothing to sparsify.

**Why anyone cares:** model quality scales with parameters, but inference cost
scales with parameters *touched per token*. MoE decouples them. That decoupling is
why essentially every frontier-scale open model released in the last two years is
an MoE.

---

## The mechanism

```
for each token's residual vector x:
    logits  = router(x)                      # [E], one score per expert
    probs   = softmax(logits)
    top_v, top_i = probs.topk(k)             # pick k experts
    top_v  /= top_v.sum()                    # renormalise the kept mass
    y = Σ_j  top_v[j] · expert_{top_i[j]}(x)
```

Routing is **per token**, not per sequence. The word "cat" in one sentence and the
same word in another can go to different experts, and consecutive tokens in one
sentence usually do.

Implementation note: `model.py` loops over experts (not tokens), gathering the
tokens routed to each and running one batched matmul per expert. That is the
readable form of what production kernels do with a permutation plus a grouped GEMM.

---

## The failure mode you will actually debug

Routing is a discrete choice, and nothing in the loss makes experts get equal
traffic. Left alone, the router collapses: a few experts get everything, the rest
receive nothing, get no gradient, and stay at their initialisation. You have paid
for `E` experts and are running `k` of them badly.

Two auxiliary losses fix it. Both are in `MoEFeedForward.forward`.

**Load-balancing loss** (Switch Transformer form):

```
aux = E · Σ_i  f_i · P_i
```

where `f_i` is the fraction of routed slots that went to expert *i*, and `P_i` is
the mean router probability for expert *i*. It is minimised when both are uniform
(`1/E`), so it pushes traffic flat. Coefficient ~0.01: large enough to balance,
small enough not to dominate the language-modelling objective. Set it to 0 and you
can watch the collapse happen in the dashboard's expert-utilisation chart.

**Router z-loss**: penalises `logsumexp(router_logits)²`, keeping the logits small
so the softmax stays numerically sane in low precision. Coefficient ~0.001. Cheap
insurance, and the failure it prevents (NaNs deep into a long run) is expensive.

**What to log, every run:**

- `moe/expert_i_frac` — traffic share per expert
- `moe/balance` — 1.0 = perfectly uniform, 0.0 = fully collapsed
- `moe/router_entropy` — falling entropy means the router is committing harder
- `moe/aux_loss` — should be small and stable, not growing

`model.py:moe_stats()` returns all of these; the dashboard charts them. **A router
collapse is visible here for a long time before it shows up in the loss curve.**

---

## The other cost: memory, not compute

The FLOPs story is good. The memory story is not, and this is where MoE claims get
oversold:

- **All** experts must be resident in memory (or paged in), even though only `k`
  run. Total parameters set your memory bill; active parameters set your FLOP bill.
- Distributed MoE ("expert parallel") puts different experts on different GPUs, so
  every token's routing decision becomes network traffic. The **all-to-all** is
  usually the bottleneck, and it is why MoE throughput is sensitive to topology in
  a way dense models are not.
- **Capacity factor**: real kernels give each expert a fixed buffer. Tokens that
  overflow their expert's buffer are *dropped* (passed through unchanged). Watch
  the drop rate; a high one silently degrades quality. This repo's loop-over-experts
  implementation has no capacity limit, which is simpler but not how production works.
- Batch-size sensitivity: with a small batch, each expert receives few tokens, so
  its matmuls are small and inefficient. MoE wants large batches to be fast.

**So when you report an MoE result, quote three numbers**: total parameters, active
parameters per token, and measured wall-clock throughput. Any one alone is
misleading, and "our 400B model" for a 17B-active model is the most common way
teams mislead people (usually themselves).

---

## Design choices you will hear discussed

**Number of experts / top-k.** 8 experts top-2 was the Mixtral-era default;
current large models use far more, finer-grained experts (hundreds) with top-8ish
routing. Finer granularity gives the router more combinations to compose and
generally scores better per active parameter.

**Shared experts** (DeepSeek's design, `n_shared_experts` here): one or more experts
that always run, in addition to the routed ones. The idea is that common patterns
should not have to be duplicated in every specialist. Cheap, and widely adopted.

**MoE in every layer or alternating?** Some architectures make every other block
dense (`moe_every=2` here) to cut memory and communication.

**Router type.** Token-choice (each token picks experts, as here) vs expert-choice
(each expert picks tokens, which guarantees balance by construction but breaks
causality-friendly streaming). Token-choice dominates.

---

## MoE changes RL, and this will matter to you

This is the least-known interaction and the one most likely to bite your team.

In RL post-training you sample rollouts with one policy and update a slightly
different one. GRPO's importance ratio corrects for that mismatch **per token**.
In an MoE, a small weight change can flip *which experts fire* for a token — a
discrete change producing a large per-token probability change that has nothing to
do with the policy being better or worse. The result is high-variance, badly-behaved
ratios and unstable training.

Two responses exist:

- **Routing replay** — force the update pass to reuse the sampling pass's routing
  decisions. Works, but constrains the model and costs memory.
- **GSPO** — use one *sequence-level* importance ratio shared by all tokens in a
  rollout, which averages the noise away. Reported by the Qwen team as removing the
  need for routing replay. Implemented here as `GRPOConfig(ratio_mode="sequence")`;
  see doc 07.

**If your team's model is MoE and it is being RL'd, this is a specific,
high-value question to raise.** It is the kind of thing that shows up as
"RL runs are unstable" and gets blamed on the learning rate for a month.

---

## The comparison to actually run

The pipeline runs dense and MoE at **matched depth, width and step count**, so the
only difference is the feed-forward block:

```bash
python -m aienh pipeline --scale small --stages pretrain_dense,pretrain_moe,dashboard
```

Read `params_total` vs `params_active` in the output, then the val perplexity, then
the wall clock. Expect the MoE to have several times the parameters, similar
active parameters, better loss per step, and *worse* wall-clock throughput at this
tiny scale — because the per-expert matmuls are small and the routing overhead is
not amortised. That inversion at small scale is real and worth seeing: MoE's
advantage is asymptotic, and a laptop-scale experiment can honestly show the
opposite of the paper.

---

## Current-generation numbers, and a caveat

Well-established figures:

| Model | Total | Active/token | Experts | Top-k |
|---|---|---|---|---|
| Mixtral 8x7B | 46.7B | ~12.9B | 8 | 2 |
| DeepSeek-V3 | 671B | 37B | 256 routed + 1 shared | 8 |
| Qwen3-235B-A22B | 235B | 22B | 128 | 8 |

The 2026 generation is bigger again — a mid-2026 survey lists models around
0.7–1.6T total with 30–50B active and 250–400 experts (GLM-5.2, DeepSeek V4-Pro,
Kimi K2.6, MiniMax M3). **I have not verified those per-model figures against
primary sources** — they come from a secondary write-up, and vendor specs for very
recent models move. Check the model card before quoting any of them in a meeting.
The pattern is the reliable part: **total parameters up by ~10×, active parameters
roughly flat.** That is the whole MoE thesis, in one row of a table.
