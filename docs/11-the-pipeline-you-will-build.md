# 11 — The pipeline you will build

Code: **`src/aienh/pipeline.py`**, **`src/aienh/registry.py`**, **`src/aienh/dashboard.py`**,
**`src/aienh/cli.py`**.

You described your job as: *build a pipeline to test the models, work with different
datasets, pretraining and post-training*. This doc is the design of that thing, plus
the decisions worth making before you write any of it.

---

## What "pipeline" has to mean

Not a script that runs the stages. A pipeline is the set of guarantees that make
results comparable across time and people. Four of them:

**1. Self-contained artifacts.** Every stage emits a checkpoint containing the
weights, the model config, *and the tokenizer*. No stage depends on ambient state or
on someone remembering which flags they passed. `train.load_checkpoint(path)` returns
a working model and tokenizer from the file alone — if that is not true of your
checkpoints, every downstream number is provisional.

**2. One suite, hashed.** Every stage is scored by the same task list, at the same
item seed, with the template that stage was trained for — and the suite hash goes into
the result. Two rows with different suite hashes sat different exams. The tooling
should refuse to average them.

**3. Lineage.** Every record names its parent. `pre-dense-lucid-vector →
sft-plain-pillar → grpo-warm-beacon`. Without it, "why did the RL run regress?" has no
answer, because you cannot identify the model it regressed from.

**4. Every stage skippable and resumable.** You will re-run one stage forty times and
the others twice. `--stages sft,grpo` exists for that reason, and it should exist from
day one, not after the first painful week.

---

## The stage graph

```
data ──▶ pretrain dense ──┐
data ──▶ pretrain MoE  ───┼──▶ eval ──▶ registry ──▶ dashboard
                 │
                 └──▶ SFT ──▶ eval
                       ├──▶ GRPO ──▶ eval
                       └──▶ distill (offline + online) ──▶ eval
```

```bash
python -m aienh pipeline --scale smoke     # ~3 min, runs in CI
python -m aienh pipeline --scale small     # the real demo
python -m aienh pipeline --stages pretrain_dense,pretrain_moe,dashboard
```

---

## Design decisions, and why

### Scale presets, with a smoke tier

`SCALES = {smoke, small, full}` — the same code path at three budgets.

**The smoke tier is the important one.** CI runs the *entire* pipeline on every
commit in a couple of minutes. A pipeline exercised only at full scale is broken half
the time, and you find out when you need it. This is the highest-value single design
decision in the whole repo, and it is the one most often skipped.

### Config files, not flags

Every stage takes `--config path.yaml`, with `--set key=value` for one-offs. A run you
cannot relaunch from a file is a run you cannot defend. The configs in `configs/` are
commented with *why* each number is what it is, because in six weeks "why is lr 3e-3?"
is a real question and "it was in the file" is not an answer.

### Names, not IDs

`sft-keen-onyx-3f9a` — adjective, noun, config-hash prefix. People say these out loud
in standups and type them into filters; `run_1729` and a UUID both fail that test. The
config hash in the name means a changed config automatically produces a changed name,
which structurally prevents the worst bookkeeping mistake available (two runs, one
name, different configs).

### Append-only JSONL registry

`runs/registry.jsonl`. Diffable, greppable, mergeable in git, impossible to corrupt
with a bad migration, and trivially readable by anything. Last-write-wins per name on
read, so re-running an eval updates the row while the full history stays in the file.

Move to a real database when you have a reason — concurrent writers, or a UI that
needs queries. Not before.

### A single self-contained HTML dashboard

No server, no build step, no external assets. Attachable to a PR, emailable,
survivable. `dashboard.py` enforces the presentation rules that stop dashboards
misleading people:

- Runs from different suite hashes are never in the same table; a selector switches
  between them.
- Every chart has a table view. Charts persuade; tables let someone check.
- Accuracies carry `n` and stderr.
- Deltas are labelled with an arrow **and a word** ("▲ 0.12 better"), never colour
  alone — colour-blind readers and greyscale printouts both matter.
- The composite score's component weights are rendered next to it.

### Point-in-time everything

`git_sha` in every record; config hash in every name; item seed in every suite hash.
Record the code version alongside the numbers, or the numbers are hearsay.

---

## What I would add next, in priority order

The repo is deliberately incomplete. If this were a real team deliverable, this is
the order I would extend it:

1. **A regression gate in CI.** Run the smoke pipeline on every PR, compare against
   the last known-good registry row, fail the build if a metric drops by more than
   2× its standard error. This is the artefact that changes team behaviour, because it
   moves eval from "something someone remembers to run" to "something that blocks a
   merge".
2. **Bootstrap confidence intervals**, not just the normal-approximation stderr.
   Resample the item set 1,000 times; report the 2.5/97.5 percentiles. More honest at
   small `n` and near 0 or 1, where the normal approximation is worst.
3. **Paired significance testing.** When comparing two models on the *same* items,
   a paired test (McNemar / paired bootstrap) is far more sensitive than comparing two
   independent proportions. Most model comparisons are paired and most teams do not
   exploit it.
4. **Per-item result storage.** Store every item's outcome, not just the aggregate.
   Then "which items did we newly break?" is a query. This is the single most
   requested capability once people start trusting the dashboard.
5. **Real benchmarks via lm-evaluation-harness**, wrapped so its results land in the
   same registry with the same lineage fields. Do not reimplement MMLU.
6. **Judge-based evals with a validated judge** — including the human-agreement
   measurement, without which the judge is a number-generator.
7. **Cost tracking.** GPU-hours and dollars per run, in the registry. Someone will
   ask, and "points per GPU-hour" is a genuinely useful ranking that almost nobody
   computes.

---

## Questions to ask your team in week one

These are the questions whose answers determine your design, and they are all
questions where teams commonly have no written answer:

1. **What is the current eval suite, and where is its item set pinned?** If the answer
   is "a script on someone's machine", that is your first project.
2. **Which prompt template does each model get evaluated with, and is it recorded?**
3. **Is contamination measured anywhere?** If not, measuring it once will be the most
   valuable thing you do in month one — and possibly the least welcome.
4. **What is the pass@1 vs pass@k picture on the tasks being RL'd?** If nobody knows,
   the RL runs are unguided.
5. **Is the model MoE, and if so is GSPO or routing replay being used for RL?**
   (doc 04, doc 07 — this is a specific, high-value question.)
6. **What is the smallest change anyone would want to detect?** That number sets your
   required `n`, and therefore your eval compute budget. Almost nobody works backwards
   from it, and it is the right way to size an eval suite.
7. **Who is allowed to change the suite, and what happens to historical numbers when
   they do?** The answer should be "it gets a new hash and the old rows stay".

---

## How to demo this repo in an interview or a first week

```bash
python -m aienh pipeline --scale small          # 20-40 min on a laptop
open artifacts/dashboard.html
python scripts/demo_template_mismatch.py       # the false-alarm class of bug
python scripts/demo_data_bias.py               # the invisible-data-bug class
python tests/test_smoke.py                     # 37 tests of the maths, not the plumbing
```

The point of the demos is not the model — it is 1.8M parameters doing two-digit
addition. The point is that the harness catches two entire classes of bug that make
teams distrust their own numbers, and that every result is reproducible from a file.
