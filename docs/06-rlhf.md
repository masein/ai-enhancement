# 06 — RLHF

RLHF is where "make the model better" stops being a supervised problem. There is no
correct next token for "write a polite refusal" — there is only *better* and
*worse*, and only people can say which.

Nothing in this repo implements RLHF (the reward model would need human labels).
This doc exists so you can follow the conversation, and because most GRPO
vocabulary is inherited from here.

---

## Why it exists

SFT can only teach you to imitate demonstrations. Two problems:

1. **Writing good demonstrations is hard, comparing outputs is easy.** Asking a
   human to write the ideal response is slow and produces one sample. Asking them
   which of two responses is better is fast and produces a usable signal.
2. **Imitation caps you at the demonstrator.** Optimising a preference signal lets
   the model exceed the average demonstration quality.

---

## The classic three-stage pipeline

### Stage 1: SFT

As in doc 05. RLHF always starts from an SFT'd model — RL from a base model is
hopeless because random continuations almost never get reward.

### Stage 2: train a reward model

Collect comparisons: for a prompt, show a human two model outputs, record which they
prefer. Then train a model — usually the SFT model with the language head replaced
by a scalar head — on the **Bradley-Terry** objective:

```
loss = -log σ( r(prompt, chosen) - r(prompt, rejected) )
```

It only ever learns *relative* quality, so the absolute scale is arbitrary. That has
a practical consequence: reward values are not comparable across reward models, and
"reward went up" is only meaningful within one RM's lifetime.

Failure modes, all of which you will eventually have to detect:

- **Length bias.** Human raters prefer longer answers, so the RM learns "longer is
  better", so the policy learns to pad. This is the most reliably reproduced
  artefact in RLHF, and it is why length is worth logging as a metric in its own
  right.
- **Distribution shift.** The RM is trained on outputs from the *old* policy. As the
  policy improves it moves off-distribution and the RM's judgements get less
  reliable exactly where you need them most.
- **Annotator disagreement.** Inter-annotator agreement on subjective preference is
  typically 60–75%. That is your ceiling. An RM with 80% "accuracy" on such data is
  partly fitting noise.

### Stage 3: optimise the policy against the reward model

```
maximise   E[ r(prompt, output) ]  -  β · KL( π_new || π_SFT )
```

The KL term is not optional. Without it the policy walks off into whatever
degenerate text maximises the RM's score — this is **reward hacking**, and with a
learned RM it happens quickly and looks like nonsense that scores 9.8/10.

**PPO** is the standard optimiser here, and it needs four models in memory:

| model | role | trainable |
|---|---|---|
| policy | the model being trained | yes |
| reference | frozen SFT model, for the KL term | no |
| reward model | scores outputs | no |
| **critic** (value model) | predicts expected return per token | yes |

The critic is what makes PPO expensive: it is typically the same size as the policy
and needs its own optimizer state. **Deleting the critic is exactly what GRPO does**
(doc 07) — that is the whole idea.

---

## DPO: the same goal without RL

**Direct Preference Optimization** skips the reward model and the RL loop entirely.
The insight is that the optimal policy for a KL-constrained reward objective has a
closed form, which you can rearrange into a loss on preference pairs directly:

```
loss = -log σ( β · [ log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected) ] )
```

Two models in memory (policy + frozen reference), no sampling, no critic, no reward
model. It is a supervised loss on a static dataset — you can run it in an afternoon.

**The tradeoff.** DPO is offline: it only ever sees the preference dataset's
responses, never the current policy's. PPO/GRPO are online: they sample from the
policy being trained, so they get signal about the model's *actual current
behaviour*. Online generally wins given enough compute; DPO wins on effort per unit
of improvement, which is why it is the default first thing to try.

Variants you will hear: **IPO** (fixes a DPO overfitting pathology), **KTO** (works
from thumbs-up/down instead of pairs, so you can use production feedback),
**ORPO** (folds SFT and preference into one stage), **SimPO** (length-normalised,
reference-free). They are refinements of the same idea; the family matters more than
the members.

---

## RLAIF, and why RLVR displaced a lot of this

**RLAIF** replaces the human labeller with a model ("Constitutional AI" is the
best-known instance: a written set of principles, a model that critiques against
them). Far cheaper, scales without a labelling org, and inherits the judge model's
biases wholesale.

**RLVR** (RL with Verifiable Rewards) replaces the judge with *code*: does the unit
test pass, does the answer match, does the output satisfy the constraint. Where you
can construct a verifier, this dominates — no reward model to train, no reward model
to drift, no annotator noise, and reward hacking becomes much harder because the
checker is exact. This is the setting GRPO is used in, and it is what doc 07 and
`src/aienh/grpo.py` implement.

The obvious limitation: most of what people want from a model is not verifiable.
"Write a good email" has no unit test. So real pipelines use both — RLVR for
math/code/format/tool-use, preference methods for everything else.

---

## What this means for your job

**1. Reward is not a metric.** A rising reward curve means the policy is getting
better at the reward function, which is a claim about the reward function, not about
the model. Always pair it with independent held-out evals. Teams that watch only
reward ship reward hackers.

**2. Log reward components separately, always.** A composite reward that rises while
its correctness component falls is the canonical reward-hacking signature, and it is
invisible in the total. `grpo.py:reward_arith` returns
`{total, correct, format}` for exactly this reason, and the training loop logs all
three.

**3. Track output length as a first-class metric.** Length inflation is the most
common artefact of preference optimisation. If length rises while quality metrics
stay flat, the model learned to pad.

**4. Keep a general-capability probe in the suite.** RL on a narrow reward degrades
unrelated abilities, and a suite that only measures the RL target cannot see it. The
KL-to-reference metric is a *proxy* for this; an actual held-out capability eval is
the real check.

**5. Insist on the reference model's identity in every record.** "KL to reference"
is meaningless without knowing which reference. Same for reward values and which RM
produced them. This is registry design, and it is your responsibility.
