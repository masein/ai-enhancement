# 07 — GRPO, and its variants

Code: **`src/aienh/grpo.py`** — a complete, runnable implementation with every
variant below behind a config flag. Read the module docstring, then the loop.

---

## The idea

PPO turns one end-of-sequence reward into per-token advantages using a **critic** —
a second network, policy-sized, trained to predict expected return. Expensive and
finicky.

GRPO deletes the critic and gets the baseline from a **group** instead: sample `G`
completions for the *same* prompt, and score each relative to its peers.

```
for each prompt:
    sample G completions from the current policy
    compute a reward r_i for each
    A_i = (r_i - mean(r)) / std(r)            ← the group is the baseline
    push up the tokens of above-average completions, push down the rest
```

That is the whole trick, and where "group relative" comes from. A useful consequence:
only the *ranking within a group* matters, so a reward function that is monotone but
badly scaled still works.

---

## The objective, matched to the code

```
ratio_{i,t} = π_new(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})

surrogate   = min( ratio · A_i ,  clip(ratio, 1-ε_low, 1+ε_high) · A_i )

loss        = -(Σ_i Σ_t surrogate) / (Σ_i |o_i|)   +   β · KL(π_new ‖ π_ref)
```

Line by line, in `grpo.py`:

**The importance ratio** corrects for the fact that you sampled with `π_old` and are
now updating `π_new`. `token_logprobs()` computes both; `old_logp` is captured under
`no_grad` right after sampling.

**Clipping** is PPO's trust region: it stops one update moving the policy so far
that the samples you collected no longer describe it. **With `mu=1` (one gradient
step per sampled batch) the ratio is exactly 1 and clipping never fires** — it only
earns its keep when you reuse a batch, which is what `mu > 1` does. If you see
`clip_frac = 0.00%` in the logs and `mu=1`, that is correct, not a bug.

**The KL penalty** to a frozen reference (usually the SFT checkpoint) is a leash
against drifting into degenerate text. Implemented with the **k3 estimator**
`exp(d) - d - 1` where `d = log π_ref - log π_new`: always ≥ 0, unbiased, and much
lower variance than the naive squared difference.

**Note the current default: `beta = 0.0`** — no KL penalty. That is what TRL ships
today (verified against the current GRPOTrainer docs), because on verifiable-reward
tasks practitioners found the leash costs more capability than the drift it prevents.
This is a live, contested tradeoff rather than settled practice: keep `beta=0` for
short runs on verifiable rewards, and raise it if you observe general ability
collapsing while reward climbs.

---

## The variants, and the problem each one solves

These names come up constantly. Each is a small, specific fix, and all are flags in
`GRPOConfig`.

### Dr. GRPO — `scale_rewards=False`

Dividing by the group's standard deviation biases the gradient: prompts whose rewards
happen to have low variance get up-weighted for no good reason. Dr. GRPO divides by a
constant instead. Cheap to try; sometimes helps.

### DAPO — four fixes

**1. Clip-higher** (`epsilon_high > epsilon_low`). With a symmetric clip, a token
the old policy assigned low probability cannot be raised much, because the ratio
bumps the ceiling immediately. Raising only the upper bound lets good-but-unlikely
tokens actually get promoted. (The paper frames this as fighting a "Matthew effect"
where already-likely tokens keep winning.)

**2. Dynamic sampling** (`dynamic_sampling=True`). If every completion in a group
gets the same reward, `A_i = 0` for all of them and the group contributes an
identically-zero gradient — while still costing you `G` generations. The paper drops
those groups *and resamples* to keep the batch size constant. This repo drops them
and proceeds with a smaller batch (simpler, and it makes the cost visible in the
logged `groups_skipped`); resampling would be the next thing to add.

**This is not a micro-optimisation, and the repo logs it.** In a measured run on a
nearly-solved task, 3–5 of every 8 groups were zero-variance. If half your groups are
dead, your effective batch size is half what your config says, and none of your
hyperparameters mean what you think.

**3. Token-level loss normalisation** (`loss_norm="token"`). Original GRPO averages
within each sequence, then across sequences — so each token of a long completion
counts for less. DAPO uses one denominator for all tokens in the batch. Matters a
lot when completion lengths vary, i.e. always in reasoning work. **TRL's current
default `loss_type` is `"dapo"`**, i.e. this.

**4. Overlong reward shaping** (`length_penalty_after`). A soft, increasing penalty
for generations past a length threshold, rather than a hard truncation, so the
length signal degrades gracefully instead of cliff-edging.

### GSPO — `ratio_mode="sequence"`

Replace the per-token importance ratio with a single **length-normalised
sequence-level** ratio, shared by every token in the completion:

```
s_i = exp( (1/|o_i|) · Σ_t log( π_new(o_{i,t}) / π_old(o_{i,t}) ) )
```

The motivation is **MoE**, and it is worth understanding precisely. Reward is
assigned to the whole sequence, but token-level ratios spread it unevenly across
tokens. In an MoE, a small weight change can flip *which experts fire* for a token —
a discrete change producing a large per-token probability change that carries no
information about whether the policy improved. Averaging into one sequence ratio
washes that noise out. The Qwen team reports this removes the need for
**routing replay** (forcing the update pass to reuse the sampling pass's routing),
which is the alternative fix.

**If your team's model is MoE and RL runs are unstable, this is the first thing to
raise.** It is the kind of issue that gets misdiagnosed as a learning-rate problem
for weeks.

### SAPO and others

TRL's current `loss_type` also accepts `"sapo"` (a soft trust-region gating variant).
The variant space is moving fast; the useful skill is not memorising the list but
being able to read one and place it: *which term of the objective does this change,
and what pathology motivated it?*

---

## The metrics that tell you an RL run is failing

This is the practical core of the doc. All of these are logged by `grpo.py`.

| Metric | Healthy | What it means when it isn't |
|---|---|---|
| `reward/mean` | climbing | flat → check the reward function actually fires |
| `reward/correct_frac` | climbing with reward | rising total + falling correctness = **reward hacking** |
| `reward/std_within_group` | > 0 | ~0 → all advantages are 0; you are computing nothing |
| `grpo/groups_skipped` | small fraction | large → effective batch is a fiction |
| `policy/clip_frac` | 0% at mu=1; a few % at mu>1 | >20% → steps fighting the trust region |
| `policy/entropy` | slowly falling | collapsing → policy becoming deterministic; reward looks great right up to the point the model can only say one thing |
| `policy/kl_to_ref` | slowly rising | spiking → drifting off the reference |
| `gen/completion_len` | stable | rising with flat quality → **length hacking** |

---

## What actually happened when I ran it — the honest result

I ran the GRPO stage in this repo across four regimes. **It did not improve held-out
accuracy in any of them.** Since that is the opposite of what the paper promises,
here is the full table and the diagnosis, because working out *why* an RL run does
nothing is most of the job.

All numbers are greedy exact-match on the held-out problem split, 1.8M-parameter
model, CPU:

| Starting point | pass@1 | pass@8 | GRPO run | After |
|---|---|---|---|---|
| SFT 4 epochs | 0.085 | 0.517 | 100 iters, lr 3e-5 | **0.050** |
| SFT 4 epochs | 0.085 | 0.517 | 100 iters, lr 1e-4 | **0.065** |
| SFT 8 epochs | 0.365 | 0.817 | 400 iters, lr 5e-5 | **~0.33**, flat |
| SFT 12 epochs* | 0.985 | 0.983 | 150 iters, lr 2e-4 | **0.845** |

\* that row was measured before the train/test split existed, so its accuracy is on
seen problems — it is the "no headroom" case regardless.

**First: it is not a sign error.**
`tests/test_smoke.py:test_grpo_update_moves_probabilities_the_right_way` takes a
positive-advantage completion and a negative-advantage one, runs five GRPO steps, and
asserts the first became more likely and the second less likely. It passes. The
gradient goes the right way; the update is doing what the objective says.

**Row 4 is fully explained by the diagnostics.** At 98.5% there is nothing to
reinforce: 3–5 of every 8 groups were zero-variance (every sample scored identically,
every advantage exactly 0), and entropy fell 0.18 → 0.078. The run had no signal and
a collapsing policy. This is the clean, expected "no headroom" failure.

**Rows 1–3 are the interesting ones**, because the pass@1→pass@8 gap says there
*should* be headroom. Two contributing factors I am confident about, and one
hypothesis I am not:

- **Confident: RL is enormously less data-efficient than SFT here.** 400 iterations ×
  8 prompts = 3,200 prompt-visits, each supervising ~4 answer tokens with a scalar
  reward. The SFT stage it started from did 448 steps × 32 examples = 14,000
  example-visits with dense per-token cross-entropy. RL is buying a much weaker signal
  per unit of compute, which is exactly why real RLVR runs are measured in thousands
  of steps and tens of thousands of prompts.
- **Confident: most groups carry no signal at low accuracy.** At pass@1 ≈ 0.08 and
  temperature 1.0, a group of 8 usually contains zero correct samples, so 4–7 of every
  8 groups were dropped as zero-variance. The effective batch was a fraction of the
  configured one — the logged `groups_skipped` says so directly.
- **Hypothesis, not verified: negative advantages bleed across contexts at this
  scale.** The answer alphabet is 10 digit tokens, and a 1.8M-parameter model has
  little capacity to represent "the token 7 is wrong *in this problem*" separately from
  "the token 7". If suppressing a wrong answer's digits also suppresses those digits
  elsewhere, positive and negative updates partly cancel. This would predict that the
  failure eases with a larger model or a bigger answer vocabulary; I did not run that
  experiment, so treat it as a plausible mechanism rather than a finding.

**What I would try next**, in order: 10× the iterations; curriculum (start RL on
easier operands so groups have variance); `beta > 0` to leash the policy while it
explores; `mu > 1` so each expensive batch of rollouts drives several updates; and a
partial-credit reward (per-digit correctness) so groups stop being all-or-nothing —
that last one directly attacks the zero-variance problem and is what I would do first.

**The transferable lesson is not "GRPO doesn't work".** It is that an RL run needs
three things checked before you trust it, and this repo gives you all three:
a correctness test on the update direction, `groups_skipped` telling you your real
batch size, and a **held-out metric evaluated during the run** so you find out at
iteration 50 rather than after the weekend. `grpo.py` prints exactly that, and warns
when held-out accuracy falls more than 3 standard errors below the run's own best.

---

## When NOT to run GRPO

**RL needs headroom, and headroom is the pass@1 → pass@k gap.** Measure both before
you start:

```bash
python -m aienh eval runs/<sft-run>/model.pt \
  --tasks arith_exact,arith_pass@8 --template chat
```

- `pass@1 = 0.10`, `pass@8 = 0.45` → **large gap.** The capability is present but
  unreliable. This is exactly what RL fixes, because it reinforces the samples that
  already happen to be right.
- `pass@1 ≈ pass@8 ≈ 0.98` → **no gap.** Do not run RL. Nothing to reinforce.
- `pass@1 ≈ pass@8 ≈ 0.02` → **no gap, at the bottom.** RL has nothing to reinforce
  either; almost every rollout fails, so almost every group is zero-variance. Fix
  the SFT stage or the task decomposition first.

Handing an RL engineer this measurement before they burn a week of GPU time is
probably the highest-leverage single thing an eval owner does.

---

## Practical notes on running it

**Generation dominates the wall clock.** An RL step is: sample `G × prompts`
completions, then one or a few gradient steps. Sampling is the expensive part, which
is why real setups run vLLM for rollouts (TRL supports `use_vllm` with colocated or
server modes) and why `mu > 1` is attractive — it amortises generation across more
updates, at the cost of a staler policy, which is what clipping then has to control.

**Temperature matters more than you would expect.** Too low and every sample in a
group is identical, so the variance is zero and there is nothing to learn. Too high
and the samples are noise. 1.0 is the usual starting point for rollouts. Note this
is a *training* parameter and has nothing to do with the temperature you evaluate at
(which should be 0).

**Truncate at EOS.** Tokens after EOS were never meaningfully chosen by the policy;
training on them is training on padding. `sample_rollouts` does this, and
`tests/test_smoke.py:test_grpo_scoring_mask_marks_exactly_the_completion` asserts the
mask covers exactly the sampled tokens.

**Never RL on eval problems.** `grpo.py` draws prompts only from the training side of
`data.arith_split`. At real scale this is benchmark contamination and it is your job
to police it; here it is enforced by construction so the habit is visible.

---

## Try it

```bash
# the default (DAPO-flavoured: clip-higher, dynamic sampling, token normalisation)
python -m aienh grpo runs/<sft-run>/model.pt --config configs/grpo.yaml

# vanilla GRPO: symmetric clip, sequence normalisation, keep dead groups
python -m aienh grpo runs/<sft-run>/model.pt \
  --set epsilon_high=0.2 --set loss_norm=sequence --set dynamic_sampling=false

# Dr. GRPO
python -m aienh grpo runs/<sft-run>/model.pt --set scale_rewards=false

# GSPO (what you would use on an MoE policy)
python -m aienh grpo runs/<sft-run>/model.pt --set ratio_mode=sequence

# make clipping actually do something
python -m aienh grpo runs/<sft-run>/model.pt --set mu=4
```

---

## Sources

- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer) — verified for
  the current defaults quoted above (`beta=0.0`, `loss_type="dapo"`, the reward
  function signature, `loss_type` options including `dr_grpo` and `sapo`).
- [From GRPO to DAPO and GSPO](https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo)
  — the clearest side-by-side of the three objectives, with the formulas.
- [GRPO++: tricks for making RL actually work](https://cameronrwolfe.substack.com/p/grpo-tricks)
  — practitioner-level detail on what breaks.
