# 01 — How LLMs work

Code for this doc: **`src/aienh/model.py`**. Read it alongside. It is ~300 lines and
contains every mechanism described here.

---

## The objective is one line

> Given tokens `t_1 … t_n`, predict `t_{n+1}`.

That is the entire training objective of a base LLM. No labels, no annotation —
the text supplies its own supervision, which is why it scales to trillions of
tokens. Everything people find surprising about LLMs (translation, code, chain of
thought) is a side effect of doing this well enough on a broad enough corpus.

The loss is cross-entropy:

```
loss = -log P(actual next token)      averaged over every position
```

Averaged over the whole vocabulary and every position in the batch. Nothing else.

One structural detail that makes it efficient: because attention is **causal**
(position *t* can only see positions ≤ *t*), a single forward pass over a sequence
of 4,096 tokens produces 4,096 separate next-token predictions, all trainable at
once. You are not training on one example per sequence; you are training on one
example per *token*.

---

## The stack, top to bottom

```
tokens        [B, T]              integers, e.g. [[42, 7, 993, ...]]
  │
  ├─ embedding lookup             a learned vector per vocabulary entry
  ▼
x             [B, T, C]           the "residual stream"
  │
  ├─ Block 1 ─┐
  │           ├─ norm → attention → add back to x     ← positions mix here
  │           └─ norm → MLP (or MoE) → add back to x  ← per-position compute here
  ├─ Block 2 …
  ├─ Block N
  ▼
final norm → linear projection to vocabulary size
  ▼
logits        [B, T, V]           a score for every possible next token
  ▼
softmax → probabilities → cross-entropy against the true next token
```

Two intuitions worth carrying:

**The residual stream is a shared workspace.** Every block *reads* from `x`,
computes something, and *adds* it back. Nothing is overwritten. This is why
`x = x + f(norm(x))` rather than `x = f(x)`: it keeps a clean gradient path from
the loss all the way to layer 0, which is what makes 100-layer stacks trainable.

**There are exactly two kinds of layer.** Attention moves information *between*
positions. The MLP transforms each position *independently*. Alternating them is
the whole architecture.

---

## Attention, concretely

Each position produces three vectors from its residual-stream vector:

- **query** — what am I looking for?
- **key** — what do I offer?
- **value** — what will I hand over if selected?

Then, for every position:

```
scores  = q · kᵀ / sqrt(head_dim)      how well does my query match each key
mask    = -inf above the diagonal      you may not look at the future
weights = softmax(scores)              a probability distribution over positions
output  = weights · v                  a weighted average of the values
```

`sqrt(head_dim)` keeps the scores' variance ~1 regardless of dimension; without it
the softmax saturates and gradients vanish.

**Multi-head** means doing this `H` times in parallel with smaller vectors
(`head_dim = C / H`) and concatenating. Different heads specialise — some track
syntax, some track long-range references. It costs the same FLOPs as one big head
and works better, which is the only justification it needs.

The cost: the score matrix is `[B, H, T, T]`. Doubling context quadruples attention
compute and memory. Everything called "efficient attention" attacks this term. In
`model.py` we call `F.scaled_dot_product_attention(..., is_causal=True)`, which
dispatches to a fused kernel (FlashAttention family) that never materialises the
full matrix — same maths, far less memory.

---

## Positions

Attention as described is order-blind: it computes a *set* operation. Position
information must be injected.

**Learned absolute** (GPT-2): add a learned vector per position index. Simple; does
not extrapolate past the trained length at all.

**RoPE — rotary position embeddings** (Llama, Qwen, most current models): rotate
each 2-D slice of q and k by an angle proportional to the position. Then `q · k`
depends only on the *difference* of positions, so attention becomes naturally
relative and the same weights extrapolate (imperfectly, but usefully) to longer
contexts. This repo defaults to RoPE; `pos="learned"` switches back so you can
compare. See `model.py:build_rope_cache` / `apply_rope`.

---

## The MLP is where the parameters are

```
h = GELU(x @ W_in)      # C → 4C
y = h @ W_out           # 4C → C
```

Two matrices, `4C² + 4C² = 8C²` parameters per block, versus `4C²` for attention's
four projections. So roughly **two thirds of a dense model's parameters are in the
MLPs** — which is exactly why mixture-of-experts replaces the MLP and not the
attention (doc 04).

Modern models often use SwiGLU (a gated variant, three matrices at ⅔ width) instead
of GELU. It is a few percent better for the same parameter count and is not
conceptually important; this repo uses GELU to keep the file readable.

---

## Normalisation and the details that matter

**RMSNorm vs LayerNorm.** LayerNorm subtracts the mean and divides by the standard
deviation, then applies a learned gain and bias. RMSNorm skips the mean subtraction
and the bias: `x / rms(x) * gain`. Fewer operations, empirically just as good,
which is why every recent model uses it.

**Pre-norm vs post-norm.** Pre-norm (normalise *before* the sublayer, as here) keeps
the residual path a clean identity and is what makes deep models trainable without
elaborate warmup schemes. Post-norm (the original 2017 Transformer) needs more care.

**Weight tying.** Reusing the embedding matrix as the output projection saves
`V × C` parameters and usually helps small models. At `V=128k, C=4096` that is 500M
parameters — not a footnote. Large models increasingly *untie* them because at scale
the constraint costs more than the parameters.

**Bias terms.** Modern models mostly drop them from linear layers. They cost
parameters and buy nothing measurable.

**Initialisation.** `N(0, 0.02)`, except the projections that write into the
residual stream, which are scaled by `1/sqrt(2 · n_layer)` so the stream's variance
does not grow with depth. Skip this and deep models start unstable.

---

## Generation

Training is parallel over positions; generation is inherently sequential:

```
predict distribution for the next token → sample or argmax → append → repeat
```

Controls:

- **temperature** divides the logits. `<1` sharpens (more deterministic), `>1`
  flattens (more diverse), `0` = greedy argmax. **Use greedy when measuring**, so
  sampling noise does not end up in your benchmark numbers.
- **top-k / top-p (nucleus)** truncate the distribution before sampling, cutting
  off the long tail of nonsense tokens.

The naive loop is O(T²) because it re-runs the whole prefix each step. Real
inference uses a **KV cache**: keys and values for earlier positions never change,
so store them. `model.py:generate` deliberately omits the cache — it would be the
first thing to add if you cared about inference speed, and leaving it out keeps the
function readable.

---

## Scale, and the numbers you should be able to recall

Approximations for a dense transformer, accurate enough for meeting arithmetic:

```
parameters      ≈ 12 · n_layer · C²           (attention 4C² + MLP 8C²)
training FLOPs  ≈ 6 · params · tokens         (fwd 2, bwd 4)
inference FLOPs ≈ 2 · active_params · tokens
```

**Chinchilla-optimal** is ≈ 20 tokens per parameter — the compute-optimal split of
a fixed budget between model size and data. Almost every deployed model is trained
far *past* it (hundreds to thousands of tokens per parameter), because inference
cost dominates training cost over a model's lifetime, so a smaller model trained
longer is the better product even though it is not compute-optimal to train.

`train.py` prints `tokens_per_param` for every run. If it is ≈0.3, the model is
massively undertrained and a bad loss means nothing about the architecture.

---

## What to actually verify by hand, once

1. Run `python -m aienh.model`. Confirm the untrained loss ≈ `ln(vocab_size)` — a
   uniform distribution over the vocabulary. If it starts much higher, your init is
   wrong; much lower, and something is leaking.
2. Read `test_causality` in `tests/test_smoke.py`, then run it. It changes the last
   input token and asserts that no earlier position's logits move. A model that
   fails this can see the future, and every number you produce from it is fiction.
3. Read `test_pack_is_a_shifted_pair`. The off-by-one between inputs and targets is
   the single most common silent bug in a from-scratch training stack.
