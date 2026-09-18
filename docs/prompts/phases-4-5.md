# Brief for Claude Code — from "explain the gap" to "fix the gap", safely

You are working in the `ai-enhancement` repo: a team LLM benchmark platform on a
shared RTX 5090. A FastAPI service (`service/`) queues evaluations, runs
lm-evaluation-harness 0.4.12, writes results under `$BENCH_ROOT/results/full/`,
and serves a single-file dashboard whose Python builds a JSON payload and whose
JS renders it (`scripts/report_lm_eval.py` — one file, no build step, no
framework; keep it that way). `scripts/diagnose.py` reads the harness's
per-item `--log_samples` output and writes `diagnose.json` beside each model's
results; the dashboard's **Diagnose** section renders it.

**Read these first, in this order, before writing anything:**

1. `docs/design-diagnose-and-generate.md` — the design this brief implements.
   Sections 2, 4, 5 and 6 are binding unless this brief overrides them. Note
   the numbering: that document's headings say "Phase 0/1/2"; this brief uses
   the step numbers from its §6, so its §4 ("Phase 1", generation) is Phase 4
   here and its §5 ("Phase 2", the judge) is Phase 5. Its Phase 0 — the split
   and the explain view — is already built and deployed.
2. `DIAGNOSE.md` — how the diagnosis is meant to be read; it tells you what the
   dashboard must make easy.
3. The module docstring of `scripts/diagnose.py` and the comment block above
   `vDiagnose()` in `scripts/report_lm_eval.py`. They explain the one rule
   everything here enforces.
4. `SERVICE.md` and `API.md` — how the service runs, and what the API already
   offers (`/api/truns`, `/api/artifacts`, `/api/submissions`).

This brief is split into phases. **Implement one phase per session, as one
pull request, in order.** If the message that hands you this file names a
phase, do that one only. If it names none, do Phase T. Do not begin a phase
until the previous one's CI is green and merged.

---

## What the user actually wants (keep this in front of you)

> For the models we train: what is the gap, and in what category — economics,
> say — are we weak. Use an LLM to find the gap and create proper data for it.
> A human stays in the loop, and the dashboard gives them enough to decide well.

Everything below serves that sentence. Where a design choice trades polish for
the human being able to decide well, choose the human.

## The one rule

Every benchmark item is assigned, by a hash of its content, to a **report**
half or a **diagnose** half (`scripts/diagnose.py::split_of`, salt
`evalboard-split-v1`). The leaderboard number comes from the report half.

**The report half is never shown to a human, never exported, never placed in
any LLM request, never used to build training data.** Only the diagnose half
may be. This is what lets a model be retrained on what the tool finds and still
have an honest score — and if the two halves ever diverge after training, that
divergence is the alarm, and Phase 6 makes it visible.

Every phase below must include a test that proves it holds for the code that
phase adds. Not a comment. A test.

## Non-negotiables (from the project's standing constraints)

- **Secrets.** The LLM API key lives in `.env` (gitignored), reaches the
  container through docker-compose `${LLM_API_KEY}` interpolation — **never a
  literal in `docker-compose.yml`**, never in git. Add every new secret's
  variable name to the strip list in `service/runner.py::_child_env` so a
  submitted model's own code cannot read it. Document in SERVICE.md that
  `docker inspect` exposes `Config.Env` to anyone with docker access on the
  box: the key is protected from submitted code, not from colleagues.
- **The tailnet is the auth boundary.** No login, no tokens (the owner has
  declined them). Record *who* approved something the way submissions already
  do — a free-text `submitter` field — and say so in the UI.
- **Never execute submitted code** except through the existing gated path.
  Datasets are JSONL, never pickle. Artifacts stay safetensors-only.
- **Bind to the Tailscale IP only.** Never `0.0.0.0` on the host side.
- **Disk.** Anything new lands under `$BENCH_ROOT` (a mounted volume), never
  the root filesystem, and gets a quota like artifacts have.
- **GPU etiquette.** Anything that uses the card (the judge, the permutation
  control) goes through the same lock and queue as evaluations. One job at a
  time.
- **Commit messages carry no AI attribution lines.** Match the existing style:
  `area: what changed`, then a body that explains *why*. Read `git log` for
  the register.
- **Do not add a JS framework, a bundler, or a second page.** The dashboard is
  one file by design. Follow its conventions: `el()`, hash routing via
  `navigate()`, the palette variables, never red-green as the only encoding,
  every bar zero-anchored, every threshold visible on the chart it applies to.
- **Pin what must be reproducible** and record it in provenance: generator
  model id, judge weights hash, prompt hash, rubric version, salt.

---

## Phase T — Tests and CI (prerequisite; small; do first)

There are no tests and no CI. Build the harness every later phase extends.

**Deliverables**

- `pyproject.toml` with `[tool.pytest.ini_options]` and `[tool.ruff]` config;
  `requirements-dev.txt` (pytest, ruff, playwright). Python 3.12.
- `tests/fixtures/make_fixture.py` — a synthetic `results/full` tree with
  `results_*.json`, `model_meta.json`, and `--log_samples` JSONL in the
  harness's **real 0.4.12 schema** (keys: `acc, arguments, doc, doc_hash,
  doc_id, filter, filtered_resps, metrics, prompt_hash, resps, target,
  target_hash`; `filtered_resps` is `[[logprob_str, is_greedy_str], …]`). It
  must cover: a model at chance, one below chance, one answering a single
  option, one skewed onto two of four slots, one picking by option length, a
  multi-true task (`truthfulqa_mc2` with `mc2_targets`), and one model whose
  `n-samples` disagrees with its log. Deterministic (seeded).
- `tests/test_diagnose.py` — unit tests for `split_of` (stable, model-
  independent, ~50/50), `bucket()` on every branch, the multi-true guard, the
  TVD/ceiling arithmetic (`ceiling == 1 - TVD`), `spread_examples()`
  round-robin, and — **the rule** — that no `report`-half `doc_hash` ever
  appears in `examples`.
- `tests/test_report.py` — `build_payload` on the fixture: `anyDiag`,
  `_trim_diag` caps, `_beside()` re-reads a rewritten file and never caches a
  miss (this was a real bug; keep the regression test).
- `tests/test_service.py` — FastAPI `TestClient` against a temp `BENCH_ROOT`:
  `/healthz`, `/api/results` reflects a `diagnose.json` written *after* the
  first request (the `_tree_key` fix), submission validation paths. No GPU, no
  network; mark anything that needs either with `@pytest.mark.gpu` /
  `@pytest.mark.network` and deselect them in CI.
- `tests/test_dashboard.py` — Playwright (Python) smoke against the fixture's
  frozen report: every tab renders with zero console errors; a model page
  shows the Diagnose card; deep link and Back/Forward work; a model with no
  diagnosis shows the "not on file" note; no horizontal scroll at 430px.
- `.github/workflows/ci.yml` — on push and PR: ruff, `py_compile` over
  `scripts/ service/ clients/`, pytest (deselecting gpu/network), Playwright
  chromium install + the dashboard smoke. Cache pip. A separate
  `workflow_dispatch`-only job builds the Docker image (the base image is
  multi-GB; do not run it on every push).
- Repo hygiene in the same PR: `git rm "main,"` (a stray tracked file),
  `.gitignore` gains `Claude outputs/`.

**Done when:** CI is green on the PR; `pytest -q` passes locally in under a
minute; the fixture generator is what every later phase's tests build on.

---

## Phase 3b — Categories, and the permutation control

Two small things the later phases depend on.

### Categories

MMLU's 57 subjects are not the categories a person thinks in. Add
`scripts/categories.yaml`: a mapping from every MMLU subject to one of roughly
fifteen human categories (economics, law, medicine & health, mathematics,
computer science, physics & engineering, chemistry & biology, history,
philosophy & religion, politics & government, psychology & sociology, business
& accounting, geography & world facts, language & logic, other). Every one of
the 57 must appear exactly once; a test enforces it, and a test fails if the
harness's subject list and the YAML ever disagree.

Then:

- `diagnose.py` writes a `categories` block per task alongside `groups`, with
  the same fields, rolled up from the mapping. Unmapped groups roll into
  `other` and are listed so the mapping can be fixed.
- The Diagnose section shows **categories first, subjects on expand**, weakest
  first, with item counts, and greys out any category whose leaderboard-half
  count is under 30 (DIAGNOSE.md's noise floor). Same rule as the existing
  group table: nothing is claimed for a task that has not cleared chance.
- The **leaderboard** gains an optional "by category" view for MMLU so a
  trained model's category profile can be compared against reference models.

### The permutation control

Every position-skewed model on the board is ≤360M. That is either a property
of small models or an artefact of how we pose MMLU, and one run settles it.

- Add a custom lm_eval task group `mmlu_perm` (`eval_tasks/mmlu_perm/`,
  loaded with `--include_path`) that re-poses a fixed subset of MMLU subjects
  with the answer options **cyclically rotated by `doc_id mod 4`**, so the
  correct answer visits every slot equally. Same few-shot count as `mmlu`.
  Deterministic; no `--limit` (the dashboard warns on limit).
- It is a control, not a leaderboard task: `required_tasks` must not include
  it, and it never enters the official average. A test enforces both.
- The Diagnose section, for a model that has both `mmlu` and `mmlu_perm`,
  shows the two scores side by side and one sentence: if `mmlu_perm` cleared
  chance while `mmlu` did not, *"the format was hiding measurable knowledge"*;
  if both sit at chance, *"the knowledge is not there to hide"*. Both numbers,
  their standard errors, and the difference in SEs are on the page.
- Document the run command in DIAGNOSE.md under "The one experiment".

**Done when:** tests cover the mapping completeness, the rollup arithmetic,
the rotation (correct index lands on each slot 25% ± tolerance over the
subset), and the official-average exclusion; the Playwright smoke covers the
category view.

---

## Phase 4 — Find the gap, generate data, keep a human in the loop

This is the feature the user asked for. Build it exactly as the pipeline in
design §4, with the server-side generation the owner chose:

```
diagnose (categories)
  → PROPOSAL: an LLM reads diagnose-half failures for one weak category
             and proposes a skill spec sentence + evidence
  → HUMAN approves / edits / rejects the spec in the dashboard
  → GENERATOR receives ONLY the approved spec text, via a batch API call
  → CONTAMINATION GATE (13-gram overlap vs the full benchmark, both halves)
  → PROVENANCE record; dataset stored as an artifact under $BENCH_ROOT
  → TAINT: any training run that consumes it records it; its checkpoints
           carry a visible mark and lose that task from the official average
```

### The LLM client

`service/llm.py`, provider-agnostic, **batch API only** (the owner's call:
cheaper, and this is never latency-sensitive):

- `submit(requests) -> batch_id`, `status(batch_id)`, `fetch(batch_id)`.
- Backends `anthropic` and `openai`, chosen by `LLM_PROVIDER`; model by
  `LLM_MODEL` (validated at startup, recorded in every provenance record);
  key from `LLM_API_KEY` via `.env` → compose interpolation. Update
  `.env.example`, `docker-compose.yml`, `_child_env`'s strip list, and
  SERVICE.md in the same commit.
- A `fake` backend for tests that returns canned responses and records every
  request body — the tests below inspect those bodies.
- Batch ids persist in SQLite so a restart resumes polling rather than
  re-submitting. A background poller in `service/worker.py`'s style: one
  thread, one loop, no GPU involvement.
- Spend guard: `LLM_MAX_ITEMS_PER_BATCH` and `LLM_DAILY_ITEM_CAP` in env; the
  UI shows today's usage against the cap before anyone clicks generate.

### Proposals (the LLM finds the gap)

- Trigger from the Diagnose section: a weak category row gets a **"Propose a
  skill spec"** action. Disabled, with the reason shown, when: the task has not
  cleared chance; the category's leaderboard-half count is under 30; or the
  task has any distribution finding (`one option only`, `answer positions`,
  `option length`, `confidently wrong`). Those are format failures — the
  button must say so and point at the finding instead of offering data. This
  gating is the single most important piece of UX in the phase.
- `POST /api/proposals` builds the LLM request from **diagnose-half items
  only** for that model × task × category: question, the model's pick, the
  correct answer, the bucket. Asks for: a one-to-three-sentence skill spec
  describing the *skill or knowledge* missing (not the questions), the share
  of failures it explains, and the two or three failure patterns it saw.
  Structured output.
- Table `proposals`: id, model, task, category, spec_text (LLM's), edited_text
  (human's), evidence JSON, status `proposed|approved|rejected`, proposer
  (LLM id), approver (free-text name), timestamps, batch_id.
- **Review tab** in the dashboard (LIVE only, like Training): proposals
  awaiting review, each showing the spec, the evidence counts, the category's
  score and ceiling, the item counts, and the model's Diagnose findings for
  that task — everything DIAGNOSE.md says a person needs. Approve (with edit),
  or reject with a reason. Show up to eight diagnose-half examples the LLM saw,
  labelled as such.

### Generation

- `POST /api/proposals/{id}/generate` — only for `approved`. The request to
  the generator contains: the approved spec text, the category name, the
  requested count and item format (question + answer + short rationale, JSONL,
  MC or free-response as the human chose), and a style constraint to vary
  surface form. **It contains no benchmark item, in any form.**
- Result lands as `$BENCH_ROOT/datasets/<id>/items.jsonl` +
  `provenance.json`. Quota `DATASET_QUOTA_GB`, default 20. Datasets are listed
  and downloadable through `GET /api/datasets`, `GET /api/datasets/{id}` and
  `/items.jsonl`, and can be pulled by `clients/bench_client.py`.

### Contamination gate

`service/contamination.py`: normalise (lower-case, collapse whitespace, strip
punctuation), build the 13-gram set from **every** benchmark item on disk
(both halves — read the `doc` fields of every `samples_*.jsonl`, cached by
tree mtime), and drop any generated item sharing a 13-gram with it. Record
per-dataset: items in, items dropped, share dropped, and the first few offending
n-grams. **Reject the dataset outright above 2% dropped** — that is a generator
echoing the benchmark, not a coincidence. Also check generated items against
each other and drop near-duplicates.

### Provenance

`provenance.json` per dataset: source model id, task, category, split
(`diagnose`), the proposal id and both spec texts, approver, generator
provider + model + batch id, prompt hash, timestamps, items generated /
dropped / kept, content sha256 of `items.jsonl`, the gate result, the split
salt. Shown in full on the dataset's page in the Review tab.

### Taint

- `POST /api/truns` and its config accept `datasets: [ids]`. A training run
  that consumes a generated dataset records it (`examples/train_and_benchmark.py`
  shows how; update it).
- In `build_payload`, a model whose training run (via the existing
  `tevents`/`hf_prefix` join) consumed a dataset derived from task T gains
  `tainted: [T]`. The leaderboard shows a badge — *"trained on data derived
  from mmlu diagnostics"* — and **excludes T from that model's official
  average**, exactly the way a missing required task does. The per-task score
  stays visible. The model page says why in one sentence.

### Tests this phase must include

- **The rule, twice.** With the fake backend, (a) every `doc_hash` referenced
  in a proposal request hashes to `diagnose` under `split_of` — assert over
  the recorded request bodies; (b) no generation request body shares a
  13-gram with any benchmark item in the fixture.
- Gate: a dataset with a planted benchmark sentence is dropped; one with 3%
  planted is rejected; one clean passes; near-duplicates collapse.
- Proposal gating: every disabled-button condition, unit-tested on the
  payload side and covered by the Playwright smoke on the UI side.
- Taint propagates through the `truns` → `tevents` → model join; the official
  average excludes the tainted task; an untainted sibling checkpoint does not
  change.
- Poller resumes a persisted batch id after a simulated restart.
- Quota refusal; provenance file completeness (every field non-empty).
- Secrets: a test greps `docker-compose.yml` and the repo for the literal key
  variable's value pattern and asserts the variable name is in `_child_env`'s
  strip list.

**Done when:** the full flow runs against the fake backend in CI end to end —
propose → approve → generate → gate → provenance → taint on the board — and
the Review tab screenshot is on the PR.

---

## Phase 5 — Judged free response: a benchmark, and the control experiment

Reframed from the design doc, with the owner's agreement. Two jobs:

1. **The benchmark** it was always meant to be — for models at 1B+ and mainly
   instruct-tuned, where multiple choice stops separating them. The design's
   case that it produces zeros for the current board stands; build it so it is
   ready before those models arrive.
2. **The control experiment** for every Diagnose finding. A third of the board
   has an MMLU score that is not a measurement of knowledge. Re-asking the
   **diagnose-half** items as open questions and judging the written answer
   tells us whether the knowledge was there under the broken format. That
   answers "is missing knowledge actually the problem?" per model, per
   category — the question Phase 4 depends on.

### Tasks

- `eval_tasks/fr/` — lm_eval `generate_until` tasks, loaded with
  `--include_path`, one per category: `fr_instruction_following`,
  `fr_factual_accuracy`, `fr_reasoning`, `fr_cultural`. Items in
  `eval_tasks/fr/<category>.jsonl`, **human-authored**. Ship ten clearly
  marked SEED items per category and `eval_tasks/fr/AUTHORING.md` explaining
  the shape of a good item and the target of ~50 per category. Do not
  generate hundreds silently; the owner said the questions must be chosen
  well, and that is a human job.
- `fr_control_mmlu` — built automatically from the diagnose half of MMLU:
  the question only, no options, answered open-ended; the judge grades
  against the gold option's text. Stratified by category, ~10 per category.
  A test asserts every item's `doc_hash` splits to `diagnose`.
- Neither enters `required_tasks` or the official average until calibrated
  (below). A test enforces it.

### The judge

- **Local and pinned.** `scripts/judge.py` loads a judge model from `HF_HOME`
  on the 5090, records the sha256 of its safetensors and its config, runs
  greedy (temperature 0), and writes `judge.json` beside the model's results
  in the same shape-and-place pattern as `diagnose.json` (the dashboard's
  `_beside()` loader and `_tree_key()` freshness key pick it up unchanged —
  add `judge.json` to `_WATCH`).
- **Never the same family as anything on the board.** `JUDGE_MODEL` is
  config; at startup, derive the board's model families from the payload and
  refuse to grade any model whose family matches the judge's, marking that
  cell "not judged — same family as judge". A test covers it. Llama- or
  Mistral-family ~7–8B instruct models are the obvious candidates given what
  is on the board today; the owner picks.
- Runs as a queued job through the same lock as evaluations. Never
  concurrently with an eval.
- **Rubric per category**, 0–4 with written anchors, in `eval_tasks/fr/rubrics/
  <category>.md`, versioned; rubric sha and prompt sha in `judge.json`. Grade
  **single answers against the rubric**, never pairwise (position bias). Put
  answer length in the rubric explicitly and report score-vs-length per
  category so length bias is visible if it appears.
- **Determinism test:** judging the fixture twice yields byte-identical
  `judge.json`.

### Calibration — not optional

- `scripts/judge_calibrate.py export` samples 100 judged answers (stratified
  by category and score) to a CSV with the rubric attached and the judge's
  score hidden; a human fills in their score; `… import` computes Cohen's κ
  per category and overall, stores it beside the judge run, and the dashboard
  shows it next to every judged number.
- **Below κ 0.6 the suite is preliminary** — shown, never ranked, never in an
  average — exactly the existing "preliminary" mechanism. Above it, the
  category scores may enter a *separate* "judged" average; they never enter
  the multiple-choice official average.

### Dashboard

- A **Judged** section on the model page: per-category rubric scores with
  κ, score-vs-length, and the control result: for `fr_control_mmlu`, the
  share of items the model answered correctly open-ended **that it got wrong
  as multiple choice**, per category. One sentence leads: *"Knew it, couldn't
  pick it"* vs *"Didn't know it either way"*, with the counts.
- On the leaderboard, judged columns appear only once κ clears the bar, with
  the κ in the column header.

### Tests

The rule (control set is diagnose-only); the family refusal; determinism;
rubric/prompt hashes present; κ arithmetic against a known table;
preliminary gating; that `judge.json` is picked up by `/api/results` without
a restart.

**Done when:** the judge runs on the fixture with a tiny stand-in model in
tests (mark the real one `gpu`), the calibration round-trip works on a CSV,
and the model page shows the control sentence with real counts.

---

## Phase 6 — Close the loop, and watch the halves

Small, and the payoff.

- `examples/train_and_benchmark.py` pulls an approved dataset by id from the
  service, mixes it into training with a recorded ratio, and registers the
  run with `datasets: [id]` so the taint flows.
- The model page for a tainted model shows, per tainted task, **the report-
  half score and the diagnose-half score before and after training**, side by
  side, with standard errors. One sentence derived from the numbers:
  - both halves moved together → *"the training taught the skill"*
  - the diagnose half moved and the report half did not → *"the training
    taught the test"* — rendered as a warning, with the ratio of the two deltas.
- The category view from Phase 3b shows the same before/after by category, so
  "we were weak in economics" becomes "economics moved by X on the half we
  never touched".
- Tests: the derived sentence on all three outcomes; the join from dataset →
  trun → checkpoint → model; the before/after picks the right parent
  checkpoint.

**Done when:** the fixture contains a tainted model and its parent, and the
model page renders the comparison with the warning variant covered by the
Playwright smoke.

---

## For every phase

- Start by reading the existing code around what you touch; it explains its
  own reasons in comments. Match that: say *why*, not what.
- Extend `tests/fixtures/make_fixture.py` rather than writing a second
  fixture.
- Update `SERVICE.md`, `API.md` and `DIAGNOSE.md` in the same PR as the code
  they describe. New env vars go in `.env.example` with a comment.
- Verify in a browser before you call UI work done: Playwright screenshots of
  the new state in light and dark, at 1240px and 430px, attached to the PR.
- The PR description ends with the exact commands to deploy and smoke-test on
  the server (`cd ~/benchmarks/aienh && git pull origin main && sudo docker
  compose up -d --build`, then what to click and what to expect).
- If a requirement here conflicts with something you find in the code, stop
  and say so in the PR rather than guessing — the owner would rather answer a
  question than review a wrong assumption.
