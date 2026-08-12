# 10 — Experiment tracking with Weights & Biases

Code: **`src/aienh/tracking.py`** (the wrapper), **`scripts/wandb_example.py`** (the
five calls that matter), **`configs/sweep.yaml`** + **`scripts/sweep_entry.py`** (a
real sweep), **`scripts/sweep_local.py`** (the dependency-free equivalent).

---

## What W&B actually is

A database with a web UI, plus a client that pushes numbers to it. That is the whole
product, and knowing that makes its API obvious.

The object model:

| Object | What it is |
|---|---|
| **run** | one execution of your script |
| **config** | the run's hyperparameters. Written once, immutable. |
| **metrics** | a time series you `log()` during the run — the loss curves |
| **summary** | final scalar values. **This is what the run table sorts on.** |
| **artifact** | a versioned file (checkpoint, dataset, eval output) with lineage |
| **sweep** | a search over configs, executed by agents |
| **project / group / tags** | organisation. `group` for one experiment's runs, `tags` for slicing. |

The five calls:

```python
run = wandb.init(project="aienh", name="pre-dense-1", config={...})
run.log({"train/loss": 1.23}, step=42)
run.summary["eval/points"] = 71.4
run.log({"samples": wandb.Table(columns=[...], data=[[...]])})
run.log_artifact(artifact)     # wandb.Artifact(name=..., type="model").add_file(path)
run.finish()
```

(Verified against the current W&B quickstart docs. The docs now favour the context
manager form, `with wandb.init(...) as run:`, which does `finish()` for you.)

---

## Why it is worth using in a team

**1. Config and metrics live together.** "Which run had lr=3e-4 and beta2=0.95?"
becomes a query instead of archaeology through someone's shell history.

**2. The run table is a shared, sortable ground truth.** This is what people mean by
"the dashboard". It ends the era of three engineers with three spreadsheets.

**3. Artifacts give you lineage.** This eval number came from that checkpoint, which
came from that dataset version. When a benchmark result is challenged six weeks later
— and it will be — lineage is the only thing that settles it.

**4. Sweeps are free parallelism.** Declare a space, run N agents.

---

## The wrapper, and why you want one

`tracking.py:Tracker` presents the same interface and picks a backend:

- W&B if it is installed and credentials exist
- otherwise JSONL on disk, in `runs/<name>/metrics.jsonl`

```bash
AIENH_TRACKER=local python -m aienh train      # no account, no network
python -m aienh train                          # real W&B if you are logged in
```

Two reasons this matters beyond convenience:

**CI must not need an API key.** A test suite that requires credentials is a test
suite that gets disabled. The whole pipeline runs at `--scale smoke` with
`AIENH_TRACKER=local` and produces a dashboard from local files only.

**You will not always own the tracking choice.** Teams switch from W&B to MLflow, or
add an internal system. If every call site imports `wandb` directly, that migration
touches every file. One wrapper makes it one file. This is ordinary engineering
judgement, and it is worth applying here even though the ML ecosystem often does not.

---

## Team conventions worth arguing for on day one

**Log `step` consistently — optimizer steps.** Not epochs, not wall-clock, not
"whatever the loop variable was". If two runs use different x-axes their curves
cannot be overlaid, which defeats the purpose.

**Put *everything* in `config`.** Model config, train config, data version, mixture
weights, git SHA, dataset row count, tokenizer hash. A run whose inputs you cannot
reconstruct is an anecdote. `train.py` logs all of that plus the data pipeline's
per-stage drop report, so a filter change is visible in the run config.

**Use `group` and `tags`.** `group` for runs that belong to one experiment (a sweep,
an ablation), `tags` for slicing (`moe`, `sft`, `rlvr`). Then the run table does the
work.

**Log a `Table` of raw samples every run.** Aggregates tell you a number moved;
samples tell you why, and about half the time the answer is "the parser broke".
`grpo.py` logs prompt / completion / reward / advantage for a sample of rollouts.

**Namespace your metric keys.** `train/loss`, `eval/val_ppl`, `moe/balance`,
`reward/mean`. W&B groups charts by prefix, and the panel layout stops being a mess.

**Log system metrics.** W&B captures GPU utilisation automatically. A run at 30% GPU
utilisation is a run you can make three times faster, and nobody notices without the
chart.

**Never reuse a run name.** Two runs with the same name and different configs is the
most expensive bookkeeping mistake available. `utils.run_name()` puts a config hash in
the name so a changed config produces a changed name automatically.

---

## Sweeps

A sweep declares a search space; agents execute it.

```yaml
# configs/sweep.yaml
program: scripts/sweep_entry.py
method: bayes                # grid | random | bayes
metric:
  name: eval/points          # must be something you write to summary
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 3.0e-4
    max: 1.0e-2
  n_embd:
    values: [96, 192, 288]
early_terminate:
  type: hyperband            # kill trials that are clearly losing
  min_iter: 100
```

```bash
wandb sweep configs/sweep.yaml     # prints a sweep id
wandb agent <sweep-id>             # run one worker; run several for parallelism
```

The non-obvious API detail: the agent passes sampled hyperparameters on the command
line, and `wandb.init()` picks them up — you read them from `wandb.config`, not from
argparse. `scripts/sweep_entry.py` shows this.

**Method choice:** `grid` when the space is small and you want completeness.
`random` is a strong default and beats grid for the same budget when some dimensions
do not matter. `bayes` when a single run is expensive enough to justify the
sequential dependency (it cannot parallelise as freely, since later trials depend on
earlier ones).

**The honest caveat, which is your job to raise:** if the spread across trials is
smaller than the standard error on your metric, the "best" config is a coin flip.
`scripts/sweep_local.py` prints exactly that warning at the end. Sweeps are very good
at producing a confident-looking winner from noise.

---

## What the local backend writes

```
runs/<name>/
  config.json        the frozen config
  metrics.jsonl      one JSON object per log() call
  summary.json       final scalars
  rollout_samples.json   sample tables — named after the table, so the filename
                         is whatever you passed to log_table()
  eval.json          the full eval result, including per-task raw samples
  model.pt           the checkpoint (weights + model config + tokenizer)
  artifact_*.json    artifact records (path, size)
```

`dashboard.py` reads these plus `runs/registry.jsonl` and emits one self-contained
HTML file — no network, no build step, no server. That property matters more than it
sounds: it can be attached to a PR, emailed, or dropped into CI artifacts, and it
still works in five years.

---

## Alternatives, briefly

**MLflow** — open source, self-hostable, strong model-registry story. The usual choice
when data cannot leave your infrastructure.

**TensorBoard** — local, free, no accounts. Fine for one person watching one run;
weak at comparing hundreds.

**Neptune / Comet / Aim** — same category as W&B, different tradeoffs.

**W&B Weave** — a separate product in the same ecosystem, aimed at tracing and
evaluating LLM *applications* (prompts, chains, judge-based evals) rather than
training runs. Different tool, different problem; do not conflate the two when
someone says "we use W&B".

For training-run tracking, W&B is the de facto standard in this space, and being
fluent in it is table stakes on a research team.
