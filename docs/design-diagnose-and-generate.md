# Feature design: diagnose a model, then fix it

Draft for discussion — 2026-09-18.

Two ideas on the table:

1. **Diagnose → generate → retrain.** A button on a model that explains its
   evaluation, shows which items it failed, and offers to generate training data
   targeting those weaknesses.
2. **Free-response benchmark.** Ask models good open questions per category, have
   an AI judge grade the written answers, and score from that.

Both are worth building. One of them has a trap in it that has to be designed
out before a single line is written, and the other is worth building *later*
than it looks, for a reason specific to our model sizes. That is most of what
this document is about.

---

## 1. The trap, stated plainly

The loop "see which items the model failed → make training data for those items →
retrain → re-evaluate on the same benchmark" **is training on the test set**. Not
metaphorically. If the generated data is derived from the failed items, the
re-evaluated score goes up and means nothing at all.

This is the mechanised version of Goodhart's law, and it is genuinely easy to do
by accident. A plausible-looking implementation — take the 400 MMLU questions the
model got wrong, hand them to an LLM, ask for "500 more like these" — produces
paraphrases of test items. The model then scores 15 points higher on MMLU and is
not one point better at anything.

The difference between the useful version and the useless one is a single
distinction:

| | What it means | Verdict |
|---|---|---|
| Train on the **items** | data derived from the specific questions it failed | contamination |
| Train on the **skill** | data generated from a description of the weakness, by a generator that never saw the items | legitimate |

"The model fails 4-digit multiplication with carries" is a *skill* statement. You
can generate unlimited fresh arithmetic from it without ever touching MMLU. "The
model failed items 3021, 3044, 3120" is an *item* statement, and anything derived
from it is poison.

**Everything below is built to make the first cheap and the second structurally
difficult.**

---

## 2. The mechanism: a held-out report split

Policy notes do not survive contact with a deadline. The separation has to be in
the data layout.

**Proposal.** At ingest, every benchmark is split deterministically into two
halves by a hash of the item id:

- **`diagnose`** — everything the explain view, the failure analysis and any
  dataset generator is allowed to see.
- **`report`** — what the leaderboard number is computed from. Never shown in the
  explain view, never readable by a generator, never exported.

`hash(item_id) % 2` with a fixed salt: stable across runs, no bookkeeping,
identical for every model so scores stay comparable.

What this buys: a model retrained on data derived from its `diagnose` failures
still has a **clean** `report` score. If the retraining genuinely taught a skill,
`report` moves. If it taught the test, `report` does not move while `diagnose`
shoots up — and that divergence is itself the contamination alarm, visible on
the dashboard for free.

Cost: half the items, so standard errors grow by √2. On MMLU (14,042 items) that
takes stderr from ~0.4 to ~0.6 points. Acceptable. On a 200-item perplexity
slice it is not — those stay whole, and are excluded from the diagnose loop.

**Open question for discussion:** this changes every existing score. Do we
re-run the board once against `report`-only, accepting a discontinuity, or carry
both and mark the old rows? My instinct is re-run — we have the GPU hours and a
board with two incompatible score definitions is worse than a day of compute.

---

## 3. Phase 0 — Explain evaluation (no generation at all)

This is most of the value and it carries none of the risk. Build it first and
ship it alone.

**We already have the data.** `--log_samples` is on, so every run writes
`samples_<task>_<timestamp>.jsonl` with one line per item. Nothing needs
re-running; this phase is reading files we already produce. *(To verify: the
exact fields lm_eval 0.4.12 writes per line — I believe `doc`, `target`,
`filtered_resps`, per-metric values and the raw loglikelihoods, but that should
be confirmed against a real file before we design a schema around it.)*

### What the explain view shows

**Per-subject breakdown.** MMLU has 57 subjects and we currently collapse them to
one number. "48.3%" becomes "72% on us_history, 26% on high_school_mathematics,
24% on formal_logic" — actionable on its own, and available today with no new
evaluation.

**Four categories of failure, which are not the same thing.** This is the part
that makes the view worth building rather than just dumping wrong answers:

| Category | Signature in the logprobs | What it means |
|---|---|---|
| **At chance** | options within noise of each other | knows nothing here |
| **Confidently wrong** | large margin on a wrong option | has learned something false |
| **Near miss** | correct option second, small margin | almost there; more data may help |
| **Format failure** | correct content, wrong extraction | our bug, not the model's |

That last row is not hypothetical. We spent a day this month on a chat template
being applied to a base model — a *formatting* failure that looked exactly like a
capability failure and moved MMLU by 25 points. An explain view that cannot tell
those apart will send people to generate training data for a problem that is in
our harness.

**Items that weaker models got right.** The highest-signal diagnostic we can
compute, and we need our own board to compute it. For each item, rank every model
that attempted it. An item where a 160M model succeeds and our 750M model fails
is an anomaly worth a human look — it is almost never "needs more data".

**Optional, later: item response theory.** With ~30 models × thousands of items
we have enough to fit a 2-parameter IRT model — item difficulty, item
discrimination, model ability. That gives us, properly, the thing the "no
separation" flag currently approximates: which items carry information at all.
Worth doing once, not in v1.

### UX

A **Diagnose** button on the model page, next to the existing tiles. It opens a
view per task: the subject breakdown, the four failure buckets with counts, and a
sample of items from each bucket (from `diagnose` only — the view physically
cannot show a `report` item).

No generation button here. Phase 0 ships without one on purpose, so we can see
what the data actually tells us before deciding what to generate.

---

## 4. Phase 1 — Generation, with the safety rails

Only after Phase 0 has been used for a few weeks and we know what the failures
actually look like.

### The pipeline

```
diagnose failures
   → a SKILL SPEC (human-reviewed, text)
   → a generator (never sees the items)
   → candidate dataset
   → contamination gate
   → provenance record
   → training-taint flag on any model trained with it
```

**The skill spec is the airlock, and a human writes it.** The system proposes
("38 of 51 high_school_mathematics failures involve multi-step arithmetic with
fractions"); a person approves or edits it; the generator receives *only that
sentence*. The failed items never cross the boundary. This is the single design
decision that makes the feature safe, and it is worth the friction of a human in
the loop.

**Contamination gate.** Before a generated dataset can be used, check n-gram
overlap against the full benchmark — both halves. The Pythia/GPT-3 convention is
13-gram overlap; any generated item sharing a 13-gram with any eval item is
dropped, and a dataset losing more than a few percent that way is rejected
outright as a sign the generator is echoing. Cheap to run, and it is the
backstop for when someone eventually bypasses the airlock.

**Provenance, recorded per dataset**: source model, benchmark, split, skill spec
text, generator identity and version, generation timestamp, item count, content
hash, overlap-check result. This is the same discipline as the per-run provenance
we already keep, and for the same reason.

**The taint flag.** A training run that consumes a generated dataset records it.
Any model from that run carries a visible mark on the leaderboard — *"trained on
data derived from mmlu diagnostics"* — and I would argue its MMLU score should be
excluded from the official average, the same way a preliminary model is. The
score stays visible and per-task; it just stops being a ranking claim. This is
not punishment, it is the only honest way to keep a board where some models have
been tuned against it.

### Where generation runs, and with whose keys

Generation wants a capable model, which means an external API or a large local
one. Two constraints from how this server already works:

- We deliberately **withhold tokens** from eval jobs and store HF credentials
  0600 in `$HOME`, never on shared volumes. Putting an LLM API key on a shared
  box, readable by a service that also runs other people's code, contradicts
  that directly.
- The tailnet is the auth boundary and there is no per-user identity on the
  service.

So generation should **not** run on the eval box with a server-held key. Options,
in order of my preference:

1. **Client-side.** The dashboard emits the skill spec; the submitter runs the
   generator on their own machine with their own key and uploads the resulting
   dataset as an artifact, which goes through the contamination gate on arrival.
   No new secret on the shared box, and the person who spends the money is the
   person who wanted the data.
2. **A local generator model** on the 5090, when it is free. No key, fully
   reproducible, but competes with evals for the GPU and a small local generator
   makes weak data.
3. Server-held API key. I would rather not, for the reasons above.

---

## 5. Phase 2 — The free-response judge benchmark

I like this idea too, and I want to make an argument about *when* it pays off,
because I think it is later than it looks.

### The case against building it now

The appeal is "multiple choice cannot separate our models, so let them write."
But for models at our scale the causality runs the other way. **Loglikelihood
multiple choice exists precisely because it extracts signal from models that
cannot generate coherent text.** A 350M base model does not write an answer a
judge can grade; it writes plausible-shaped noise. An LLM judge grading noise
produces noise with a confident number attached — which is worse than no number.

We already have the evidence on our own board: **gsm8k is generative, and it
reads 0.0% for most of our models.** That is not a measurement failure, it is the
honest result of asking a sub-billion model to produce a written answer. A
judged free-response suite would produce the same floor, with more cost and less
reproducibility.

And the thing we actually want — separation among models that multiple choice
cannot distinguish — we already have. It is the perplexity slices, and our own
dashboard says so: *"they are the eval that separates models multiple-choice
cannot."* Perplexity works on base models with no prompt format at all, costs
nothing extra, and is fully deterministic.

**So: build it, but build it knowing it starts paying off somewhere around 1B+
parameters and mainly for instruct-tuned models.** For the current board it will
mostly produce zeros. If the roadmap has us training past 1B, it is worth having
ready before we get there rather than after.

### If and when we build it, the hard parts

**The judge must be pinned, and I would argue it must be local.** A benchmark
whose scores change when a vendor silently updates a model is not a benchmark.
We learned this exact lesson two weeks ago one level down: transformers was
unpinned, drifted 5.15.0 → 5.17.0, and a checkpoint-key rename broke a model
load silently. A judge behind an API is that same failure with no version number
to check and no way to re-run last month's grading. A local judge on the 5090,
pinned by weights hash, is reproducible forever and costs GPU time we control.
The trade is that a local judge good enough to grade is large enough to compete
with evals for the card.

**Known judge biases to design against**, all documented in the literature and
all cheap to mitigate:

- **Position bias** — judges favour the first answer shown. Mitigate by grading
  single answers against a rubric rather than pairwise, or by randomising order
  and averaging both directions.
- **Length bias** — longer answers score higher regardless of quality. Mitigate
  by putting length in the rubric explicitly, and by reporting score-vs-length
  so we can see it if it appears.
- **Self-preference** — a judge scores its own family higher. Mitigate by never
  using a judge from the same family as anything on the board, and by recording
  judge identity in provenance.

**Calibration is not optional.** Before any judged score goes on the board, a
human grades a sample — 100 answers or so — and we report judge/human agreement
(Cohen's κ). Below some agreement threshold the suite is preliminary, the same
way a model missing required tasks is preliminary. Without this number we do not
know what the judge is measuring, and neither does anyone reading the board.

**Determinism.** Temperature 0, pinned prompt, prompt hash in provenance, and the
rubric versioned in git. Two runs of the same model must produce the same score
or the whole thing is decoration.

**Scoring shape.** A rubric score per category (0–4, say, with anchors written
out), not a single overall number. Free-response's advantage over MC is
*diagnostic richness*; collapsing it to one number throws that away.

---

## 6. What I would actually do, in order

1. **Held-out split** (`diagnose` / `report`). Lands before anything else,
   because retrofitting it after generation exists is how contamination gets
   grandfathered in. Includes the decision about re-running the board.
2. **Explain evaluation, read-only.** Subject breakdown, the four failure
   buckets, items-weaker-models-got-right. Uses `--log_samples` data we already
   write. This is the phase with the best value-to-risk ratio and it might be
   enough on its own.
3. **Use it for a few weeks.** Look at what the failures actually are before
   building a machine to fix them. My guess is that a meaningful share turn out
   to be format failures and task artefacts rather than missing knowledge — and
   if so, generating training data would have been the wrong response entirely.
4. **Generation**, with the skill-spec airlock, the contamination gate,
   provenance and the taint flag. Client-side keys.
5. **Judged free-response**, when the models are big enough to be worth judging.
   Local pinned judge, rubric per category, human calibration before it counts.

## 7. Open questions for you

- **Re-run the board against `report` only, or carry both?** I lean re-run.
- **Should a taint-flagged model be excluded from the official average**, or just
  marked? I lean excluded — same treatment as preliminary.
- **Is 1B+ on the roadmap soon?** It decides whether the judge benchmark is next
  quarter's work or next year's.
- **Who writes the skill specs?** The airlock only works if a person actually
  reads them. If that person is always you, it is a bottleneck worth knowing
  about now.
- **Does anything currently on the board already have this problem?** Worth
  asking before we build the detector — if any checkpoint was trained on data
  derived from a benchmark, we should know.
