# 08 — Soft vs hard labels, and distillation

Code: **`src/aienh/distill.py`**.
See it live: `python -m aienh pipeline --scale smoke --stages pretrain_dense,sft,distill`
prints the teacher's actual distribution before training the student.

---

## Hard vs soft labels

**Hard label** — the one correct token, as a one-hot vector. `"62"` and nothing else.

**Soft label** — the teacher's full distribution over the vocabulary:
`62: 0.70, 61: 0.15, 72: 0.05, 63: 0.04, …`

The soft label carries strictly more information. It says which wrong answers are
near-misses and which are absurd — Hinton's "dark knowledge". Concretely: a hard
label gives you one bit of "this token, not the others"; a soft label gives you the
teacher's entire similarity structure over the vocabulary, for free, at every
position.

**Why that matters:** each token carries more information, so the student needs fewer
tokens to reach a given quality. That is the whole reason distillation works, and why
essentially every small production model is distilled from a big one rather than
pretrained from scratch.

`distill.py:inspect_soft_labels` prints a real teacher's distribution at several
temperatures next to the hard label it would collapse to. Run it once — the
distinction stops being abstract immediately.

### Temperature, and the T² factor

Dividing logits by `T > 1` flattens the distribution and exposes the
low-probability structure that a confident teacher otherwise hides:

```
L = α · T² · KL(teacher_T ‖ student_T)  +  (1-α) · CE(hard label)
```

The `T²` is not cosmetic. Dividing logits by `T` shrinks gradients by `1/T²`, so you
multiply back to keep the soft term's magnitude comparable to the hard term as you
tune `T`. Omit it and changing `T` silently changes your effective learning rate on
the soft term.

Typical values: `T` 1–5, `α` 0.5–0.9. Keeping a nonzero hard-label term
(`α < 1`) anchors the student to ground truth where the teacher is wrong.

### KL direction is a real design choice

`KL(teacher ‖ student)` — **forward**, mode-covering. The student is penalised for
putting low probability anywhere the teacher puts mass, so it tries to cover all of
the teacher's behaviour. Standard for distillation.

`KL(student ‖ teacher)` — **reverse**, mode-seeking. Produces a sharper, less diverse
student that commits to the teacher's dominant modes. Sometimes preferred when you
want a decisive small model.

`distill.py:kd_loss_full` uses forward KL and says so in the docstring. Know which
one your team is using; it visibly changes output diversity.

---

## Offline vs online vs on-policy

This is the distinction in your list, and the three differ in *what data the teacher
is asked about* — which turns out to matter more than the loss function.

### Offline

Run the teacher **once** over the corpus, store its outputs, then train the student
against the stored targets.

- **Pro:** teacher compute is paid once. You can then iterate on student
  architecture, width, learning rate, twenty times, without touching the teacher.
  This is the practical default while you are still searching.
- **Con:** storage, and staleness. The targets only cover the text you ran the
  teacher on, which is text the *teacher* would visit — not text the *student*
  produces.

**The storage arithmetic, which is why nobody stores full logits:**

| what | size for 1B tokens, V=128k, fp16 |
|---|---|
| full distribution | 2 × 128,000 × 1e9 = **256 TB** |
| top-k, k=16 (2B value + 4B index) | 6 × 16 × 1e9 = **96 GB** |

So you store **top-k** and renormalise over the kept `k` at train time.
`distill.py:kd_loss_topk` does this, and prints the real ratio for your run.
The approximation error grows as `k` shrinks — at `k=1` you are back to hard labels
with extra steps, which defeats the point. `k` in 8–64 is the usual range; measure
it rather than guessing, because the right value depends on how peaked your teacher
is.

### Online

Teacher and student both in memory. The teacher runs forward (no grad) on the same
batch as the student, every step.

- **Pro:** no storage, always consistent with whatever data you feed, and you get the
  **full** distribution rather than a top-k approximation.
- **Con:** a teacher forward on every step, forever — typically 2–4× the student's own
  cost — and the teacher has to fit alongside the student.

### On-policy (sequence-level KD / GKD)

The **student generates**, and the teacher scores the student's own samples.

This fixes the deepest problem with both of the above. Teacher-forced training only
ever shows the student states the *teacher* would visit. At inference the student
follows its own outputs, walks into states it never saw, and compounds errors —
**exposure bias**. Training on the student's own trajectories removes that mismatch,
and empirically it is what closes the last chunk of the teacher–student gap.

- **Con:** generation inside the training loop. Slowest option, and it shares the
  wall-clock profile (and the vLLM-shaped infrastructure) of RL.

Note the structural similarity to RL: sample from the current policy, score the
samples with something external, update. On-policy distillation is RL where the
reward is "agree with the teacher".

### Which to use

Offline while iterating on the student (cheap, repeatable, teacher bill sunk).
Online when the teacher co-resides or you need full distributions. On-policy as a
final stage, once the student is otherwise trained, to fix generation-time behaviour.

---

## What else gets distilled

**Logit distillation** — the above. Needs a shared tokenizer and vocabulary, which
is a real constraint: you cannot logit-distil across model families without a
vocabulary alignment step.

**Sequence-level / rejection-sampling distillation** — the teacher generates
completions, you keep the good ones (verified, or filtered by a reward model), and
SFT the student on them. No shared vocabulary needed, dead simple, and extremely
common in practice. Most "distilled" open models are this, not logit distillation.

**Reasoning-trace distillation** — generate long chains of thought with a large
reasoning model, keep the ones with correct final answers, SFT a small model on the
traces. This is how the small "distilled reasoning" models are made, and it works
strikingly well relative to its simplicity.

**Feature / intermediate distillation** — also match hidden states or attention
maps. Requires architectural compatibility; more finicky, sometimes worth it.

---

## Things to check as the eval owner

**1. The tokenizer must match** for logit distillation. Different tokenizers means
different vocabularies means the distributions are not comparable, full stop.

**2. Record the teacher in the registry row.** A student's score is meaningless
without knowing which teacher, at which checkpoint, produced it.
`registry.RunRecord.parent` plus `metrics.teacher_params` do this here.

**3. A student that scores well on the distillation distribution may be much worse
off it.** Offline KD teaches the student the teacher's behaviour *on that corpus*.
Evaluate off-distribution deliberately.

**4. Distillation can transfer the teacher's contamination.** If the teacher
memorised a benchmark, its soft labels encode those answers, and the student inherits
the inflated score without ever seeing the benchmark. This is genuinely hard to
detect from the student's data alone — the only defence is knowing the teacher's
contamination status.

**5. Report the compression ratio next to the score.** "97% of teacher quality at 20%
of the parameters" is the claim distillation makes. Both halves are needed.
`distill.py` prints the student/teacher parameter ratio at the start of every run.

---

## Run it

```bash
# offline: builds a top-k cache, prints the storage arithmetic
python -m aienh distill runs/<teacher>/model.pt --mode offline --config configs/distill_offline.yaml

# online: full distributions, teacher forward every step
python -m aienh distill runs/<teacher>/model.pt --mode online

# on-policy: student generates, teacher grades its own trajectories
python -m aienh distill runs/<teacher>/model.pt --mode on_policy

# and to see soft labels at several temperatures next to the hard label:
python -m aienh pipeline --scale smoke --stages pretrain_dense,sft,distill
```
