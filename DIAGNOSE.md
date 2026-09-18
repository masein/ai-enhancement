# Reading the diagnosis

`scripts/diagnose.py` reads the harness's `--log_samples` output and writes
`results/full/<model>/diagnose.json`; the dashboard's **Diagnose** section
renders it. Nothing here re-runs an evaluation or touches the GPU — every
per-item outcome is already on disk.

This file is about what to do with it for the next few weeks, which is the
deliberate gap between phase 2 (explain) and phase 4 (generate) in
`feature-diagnose-and-generate.md`. The point of the gap: look at what the
failures actually **are** before building a machine to fix them.

---

## The routine

**After each batch of evals** (or once a week, whichever is less):

```
cd ~/benchmarks && sudo python3 aienh/scripts/diagnose.py results/full 2>&1 | tail -40
```

One model only, when a checkpoint lands:

```
cd ~/benchmarks && sudo python3 aienh/scripts/diagnose.py results/full -m local/my-ckpt
```

Read the **FINDINGS** block it prints at the end. That block is the triage
list — it is the only board-wide view; the dashboard is per-model. It names
`<model>/<task>` pairs and nothing else needs your attention.

Then open those models in the dashboard → the model page → **Diagnose**.

Do not walk the whole board. A model with no findings has nothing
distribution-level wrong with it, and the section says so in one line.

---

## Read it in this order

The order matters, because the most seductive part is the least informative.

1. **The at-chance line.** If a task's score has not cleared chance, stop
   reading that task's detail. The breakdown describes how the model *guesses*,
   not what it knows, and no group score or example below it is evidence about
   the subject. Most of the board's small models are here on MMLU.
2. **The findings block** at the top of the section. One sentence per finding,
   with the numbers it was derived from.
3. **Where the answers went** — picked vs correct, per answer slot. This is the
   picture that settles a position-bias finding in one glance.
4. **Weakest categories first, subjects on expand.** MMLU's 57 subjects roll
   up into fifteen categories (`scripts/categories.yaml`); "weak in economics"
   is a sentence someone can act on, "weak in high_school_macroeconomics on
   13 items" is not. A category under 30 leaderboard-half items is greyed.
   Only meaningful when the task cleared chance — the section says so.
5. **The failing items** — last, and mostly for models with *no* findings.
   Eight questions always look like a pattern. They are a cross-section of the
   diagnosis half, and they are the weakest evidence on the page.

---

## What each finding means, and what it is NOT

| Finding (CLI label / UI chip) | What it says | What to check | Data will fix it? |
|---|---|---|---|
| `answers one option` / `one option only` | One slot chosen on ≥80% of items | Generation config, prompt format, whether the checkpoint is broken | **No** |
| `answer positions skewed` / `answer positions` | Picks and answer key have different shapes (TVD ≥ 0.20) | The ceiling shown on the page. Permute the options and re-run — see below | **No** |
| `picks by option length` / `option length` | Shortest option chosen ≥1.6× chance | Whether the task is scored `acc` where it should be `acc_norm` | **No** |
| `confidently wrong on most items` | ≥35% of items: backs a wrong option *and* rates the correct one below chance | **Provenance → chat template.** A template applied to a base model does exactly this | **No** |
| *(no finding, score above chance, weak groups clustered)* | An ordinary subject gap | The group item counts | **Yes** — this is the case phase 4 is for |

The fifth row is the only one a training-data generator would help. Four of the
five are properties of how we pose the task or of the output distribution, and
more subject data moves the score without teaching the model anything.

`at chance on most items` (≥40% of items carrying no information either way) is
reported as context rather than a finding. It is honest ignorance, and it is
also a case where more data is the right answer — but read the at-chance line
first, because a model at chance overall tells you nothing about *which*
subject to feed it.

---

## Telling a real finding from noise

- **Group scores under ~30 items are noise.** A subject with 13 items in the
  leaderboard half carries roughly ±13 points. Do not chase its ranking. The
  page greys every category and subject under that floor (`MIN_GROUP_N` in
  `scripts/diagnose.py`, `thresholds.min_group_n` in every `diagnose.json`).
- **A category called `other` with an `unmapped` list** means the harness
  produced a subject `scripts/categories.yaml` does not know. Add it to the
  file and re-run `diagnose.py`; the CLI lists them at the end of its output.
- **Position bias on a 2-option task** (Winogrande, PIQA) means much less than
  on a 4-option one — there are only two slots to skew between.
- **A finding just over its threshold is weak.** The thresholds are in the
  `thresholds` key of every `diagnose.json` and as constants at the top of
  `scripts/diagnose.py`: `POSITION_SKEW` 0.20, `DEGENERATE_SHARE` 0.80,
  `CONFIDENT_SHARE` 0.35, `CHANCE_LIFT` 1.30. TVD 0.21 is not TVD 0.44.
- **`truthfulqa_mc2` reports no statistics at all**, by design — several options
  are correct, so there is no single answer index. If you ever see a finding on
  it, something has regressed.
- **An item-count warning** on a task means the per-item log and the reported
  score do not cover the same number of items. Resolve that before believing
  anything else about that task.

---

## The one experiment

Every position-skewed model on the board is ≤360M, which is either a real
property of small models or an artefact of how we pose MMLU. One cheap run
settles it: `mmlu_perm` (`eval_tasks/mmlu_perm/`) re-poses a fixed subset of
twelve MMLU subjects with the answer options rotated by `doc_id mod 4`, so the
correct answer visits every slot equally. Same five shots, same metric, option
text untouched; only where the right answer sits changes. It is a **control,
not a leaderboard task** — it never enters an average, however
`REQUIRED_TASKS` is set — and it uses the card, so it goes through the same
queue and lock as every evaluation:

```bash
python clients/bench_client.py --base http://<tailscale-ip>:8899 \
    submit HuggingFaceTB/SmolLM2-360M --suite control --submitter you
# or in the dashboard's Submit box: suite = control
```

About a fifth of a full MMLU. Then open the model's page → **Diagnose** →
*The permutation control*: both scores, both standard errors, the gap in
standard errors, and one sentence derived from them:

- **"the format was hiding measurable knowledge"** — `mmlu_perm` cleared
  chance while `mmlu` did not. The bias is positional and it is ours. The fix
  is the prompt format, and it applies to every model under ~400M on the board.
- **"the knowledge is not there to hide"** — both sit at chance. The model
  genuinely cannot reach those slots, and the ceiling shown on its page is
  its real ceiling.

Either answer changes what phase 4 should be. Neither needs new training data.
Run it on `SmolLM2-360M` first (TVD 0.44 on MMLU, the clearest case), then on
one position-skewed model per family.

---

## From a gap to data: the Review tab

Once the routine above has said "genuine gap" about a category, the Diagnose
section offers **Propose a skill spec** on that category's row. The button is
disabled, with the reason beside it, whenever the task has not cleared chance,
has any distribution finding, or the category is under the noise floor —
because those are format failures, and a generator offered for them would
make the score move while the model learned nothing.

What happens after the click (`service/proposals.py`):

1. An LLM reads this model's **diagnosis-half** failures in that category —
   never the leaderboard half — and proposes one to three sentences naming the
   skill that is missing, the share of failures it explains, and the patterns
   it saw.
2. A person reads it on the **Review** tab next to everything this file says
   they need: the category score and item count, the ceiling, the model's own
   findings for the task, and eight of the diagnosis-half items the LLM saw.
   They approve it, edit it, or reject it with a reason, under their name.
3. Only the approved text reaches the generator. Not one item, hash, model
   name or score goes with it. The generated items pass a 13-gram
   contamination gate against both halves of every benchmark on disk; above
   2% dropped the whole dataset is refused as an echo.
4. The dataset lands under `$BENCH_ROOT/datasets/<id>/` with a full
   `provenance.json`, downloadable from the tab and with `bench pull`.
5. A training run that consumes it says so (`datasets=[id]` on the run), and
   every checkpoint of that run carries the badge *trained on data derived
   from mmlu diagnostics* and loses that task from its official average. The
   per-task score stays; the ranking claim goes. If that model's leaderboard
   half and diagnosis half then move apart, that divergence is the alarm this
   whole design exists to raise — phase 6 makes it visible.

## The log

The output of these weeks is this table, not a feeling. Fill one row per
finding you actually investigated. After a few weeks the tally — format
failures versus genuine subject gaps — is the input to the phase-4 decision,
and if it comes out the way section 6 of the feature doc predicts, phase 4 is a
prompt-format fix rather than a data generator.

| Date | Model | Task | Finding | What I checked | Verdict (format / artefact / genuine gap) | Fix it would need |
|---|---|---|---|---|---|---|
| | | | | | | |

Verdict definitions, so the tally means something:

- **format** — how we pose the task (prompt, chat template, few-shot layout)
- **artefact** — how we score it (metric choice, option lengths, a multi-true task)
- **genuine gap** — the model cleared chance, no distribution finding, and the
  weak groups are consistent across enough items to believe
