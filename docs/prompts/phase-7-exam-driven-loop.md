# Brief for Claude Code — make the exam the instrument

Phases T, 3b, 4, 5 and 6 are merged and green (123 passed, 1 skipped, 17
deselected). This brief changes the **shape of the loop**, not its safety
properties. Most of what is there survives; read it before you change it.

**Read first:**

1. `docs/prompts/phases-4-5.md` — what was built. It is now superseded by this
   file wherever the two disagree, and the disagreement is deliberate.
2. `service/proposals.py`'s module docstring — the pipeline and where each
   safety property lives. The properties do not change. Their inputs do.
3. `scripts/judge.py`'s module docstring — the judging contract. Everything in
   it stays true except "local".
4. `DIAGNOSE.md`, `docs/design-diagnose-and-generate.md`.

Implement as **one PR per change below, in order**. Do not start a change until
the previous one's CI is green and merged.

---

## What changed, and why

The old brief had the gap found by reading **failed MMLU questions**, with the
free-response exam as a secondary control experiment off to one side. That is
backwards for what this platform is for.

The loop the owner actually wants:

```
  0. write the exam            an LLM writes questions across every topic,
                               a human curates                     [once]
     ────────────────────────────── then, every cycle ──────────────────────
  1. sit the exam              the checkpoint answers, on our 5090      [free]
  2. judge the answers         an LLM grades them: a score per topic,
                               and written evaluation of what went wrong
  3. choose the topic          a human reads the scores and the judge's
                               evaluation, picks one topic, approves a spec
  4. generate the dataset      an LLM writes training documents for it
  5. fine-tune and resubmit    whoever owns the training run           [free]
  ↻  back to 1 — same exam, and the before/after says whether it worked
```

The judge's own output is the gap-finding signal. There is no separate
mechanism reading MMLU failures to decide what to train, because the exam
already says, per topic, in a person's vocabulary, how good the model is and
why it is not better.

**MMLU and the rest of the standard suite do not go away.** They keep running,
keep costing nothing (loglikelihood on our own GPU), keep their per-item
diagnosis, and stay on the board as the comparable public-protocol number and
as a second opinion — a topic that is weak on the exam *and* at chance on MMLU
is a different problem from one that is weak on the exam alone. They simply
stop being what picks the topic.

---

## What does NOT change (do not touch these)

- **The split rule.** `scripts/diagnose.py::split_of`, salt
  `evalboard-split-v1`. Report half never shown, never exported, never in an
  LLM request, never used to derive training data.
- **The airlock.** A human approves a spec string; the generator receives only
  that string. `generation_requests()` still takes no item, no hash, no model
  name, and the test that proves it from recorded request bodies stays.
- **The contamination gate**, provenance record, dataset quota, taint flag and
  the official-average exclusion.
- **`service/llm.py`** — batch-only client, `anthropic` / `openai` / `fake`
  backends, `Request`/`Result`, `prompt_sha`, `extract_json`, the poller and
  its persisted batch ids. This is the right abstraction and change C2 depends
  on it.
- **`scripts/categories.py` + `categories.yaml`** — the ~15 topics and the
  subject rollup. Change C1 reuses the topic list as the exam's spine.
- **`eval_tasks/mmlu_perm/`** — the permutation control. Independent of all
  this and still worth running.
- **The secrets discipline**: `.env` only, compose interpolation, every secret
  name in `service/runner.py::_child_env`'s strip list, and the grep test.
- **CI, the fixture generator, and every existing test** except where a change
  below explicitly rewrites one.

---

## C1 — The exam becomes the instrument

**Today:** `eval_tasks/fr/` holds four *skill* categories
(`instruction_following`, `factual_accuracy`, `reasoning`, `cultural`) with ten
hand-written seed items each. `scripts/fr_build.py` copies them and builds
`fr_control_mmlu` from MMLU's diagnose half.

**Wanted:** one exam question bank spanning the **topics in
`scripts/categories.yaml`** — economics, law, medicine & health, mathematics,
and the rest — because "which topic are we weak in" is the question the loop
exists to answer, and `instruction_following` is not a topic.

Build:

- **`scripts/exam_build.py`** (new; `fr_build.py`'s control-set logic moves
  here, the rest is replaced). It drafts candidate questions through
  `service/llm.py` — one batch, one request per few candidates, the request
  carrying a topic brief, the rubric, and example questions so drafts land
  on-target and vary in form. It writes candidates to
  `$BENCH_ROOT/exam/candidates/<topic>.jsonl` for curation, **never straight
  into the live bank**.
- **A curation step.** A human accepts, edits or rejects each candidate in the
  dashboard; accepted ones land in `$BENCH_ROOT/exam/bank/<topic>.jsonl` with
  who accepted them and when. Target ~60 per topic; ship with the four existing
  skill categories' 40 items migrated in as-is so nothing is thrown away.
- **The bank is split.** Every accepted question gets a `qid` (sha256 of its
  normalised text) and `split_of(qid)` decides report or diagnose, using the
  same function and salt as the benchmarks. **The published per-topic score
  comes from the report half. Step 3 may read only the diagnose half.** This is
  the single most important line in this brief: the exam is our own generated
  questions, so nothing external protects it, and without the split the loop
  trains on its own test with no outside benchmark to catch it.
- lm_eval task yamls per topic, generated as `fr_build.py` already does
  (absolute `data_files` path, manifest with each file's sha256).
- Keep `fr_control_mmlu` exactly as it is — diagnose-half MMLU re-asked
  open-ended is still the "knew it, couldn't pick it" experiment and it now
  reads naturally as one more topic-shaped signal.

**Tests.** Every accepted question's `qid` splits deterministically and the two
halves are within a few percent of even; no report-half question appears in any
artefact step 3 can read (assert over the recorded request bodies, as
`test_gap.py` already does for MMLU); a candidate never reaches the bank
without a recorded approver; the topic list in the bank matches
`categories.yaml` exactly; the manifest's hashes match the files.

---

## C2 — The judge moves to the API

**Today:** `scripts/judge.py` loads a local model with transformers from
`HF_HOME`, sha256s its safetensors, runs greedy on the GPU.

**Wanted:** the judge is an API call through `service/llm.py`, batch mode.
This is the owner's decision; do not re-argue it. But it costs the property the
local judge was chosen for, so replace that property rather than dropping it.

Keep, unchanged in substance: single answers graded 0–4 against a written
versioned rubric with anchors, never pairwise; length named explicitly in the
rubric and score-vs-length reported per topic; the family check; Cohen's κ
calibration against a human sample, with the suite **preliminary below κ 0.6**
and never entering the multiple-choice official average; rubric sha and prompt
sha in every `judge.json`.

Change:

- Grading goes through `llm.Request` with `JUDGE_PROVIDER`, `JUDGE_MODEL`,
  `JUDGE_API_KEY` in `.env` (separate from `LLM_*`, because C-below requires
  them to differ). Add the new key name to `_child_env`'s strip list and the
  secrets-grep test.
- **Pin a dated model ID**, not a floating alias. Record provider, model ID,
  prompt sha, rubric sha and the batch id in `judge.json` and in provenance.
- **Replace the byte-identical determinism test with a drift canary.** A fixed
  set of ~30 answer scripts with known human marks is re-graded at the start of
  every judge run; `judge.json` records the canary's mean absolute deviation
  from those marks and from the previous run's. The dashboard shows it beside
  the scores, and a canary that moves more than a configurable threshold marks
  the run **preliminary** with the reason stated. Without this, a vendor
  updating the model behind the ID silently re-bases every score — the
  unpinned-`transformers` failure again, one level up.
- `--stub` stays, and the `fake` backend is what CI uses.

**Tests.** The canary arithmetic; preliminary gating on both κ and canary
drift; provider/model/prompt/rubric recorded in `judge.json`; the family
refusal; `judge.json` still picked up by `/api/results` without a restart (the
`_beside`/`_tree_key` behaviour is already tested — extend it, do not
duplicate).

---

## C3 — The gap comes from the judge, not from MMLU

**Today:** `service/proposals.py::failures_for(model_dir, task, category)`
reads MMLU's diagnose-half failures and the proposal prompt is built from
them.

**Wanted:** the proposal is built from the judge's own output for the chosen
topic — the low rubric scores and, more importantly, the **written
justifications**, which already say what went wrong in words.

- `failures_for()` → `weak_topics(model_dir)` returning per-topic report-half
  score, rank and item count from `judge.json`, plus
  `justifications_for(model_dir, topic)` returning the judge's written
  reasoning for that topic's **diagnose-half** low-scoring answers only.
- The proposal request carries those justifications, the topic name and the
  rubric — and **no exam question text**, because the justification is about
  the answer, not the question. If a justification quotes the question, strip
  it: add a test with a planted quotation that must not survive into the
  request body.
- Keep the gating, and re-point it: the "Propose" action is disabled, with the
  reason shown, when the topic's report-half count is under 30, when κ or the
  canary has the judged suite preliminary, or when the model's answers for that
  topic were mostly empty or degenerate (a model that wrote nothing has not
  revealed a topic gap). Show MMLU's distribution findings for the matching
  category alongside as a caution, not as a gate.
- The existing `proposals` table, Review tab, approve/edit/reject flow,
  approver recording and provenance all stay; only the evidence shown changes.

**Tests.** Every justification in a proposal request traces to a diagnose-half
`qid`; the planted-quotation strip; each disabled-button condition; the
proposal row records which judge run it was derived from.

---

## C4 — The dataset becomes documents, not question-and-answer pairs

**Today:** `GEN_SYSTEM` + `STYLE` ask for `mc` or `free` items and
`parse_items()` requires `{question, answer, rationale}`.

**Wanted:** training documents — prose passages that teach the skill in the
spec. A short explainer, a worked discussion, a piece of reference text.
Generating question-and-answer pairs shaped like the exam is the most direct
route to teaching the test there is, and prose does not have that shape.

- New format `doc`: `{"title": ..., "text": ...}`, one JSON object per line.
  Target ~800 tokens of body per document, ~2 per request (they are long).
  Keep the style constraint that forces variation in register, length and
  framing across a set.
- Retire `mc`; keep `free` behind a flag if you want it for comparison, but it
  is not the default and the UI should say why.
- **The contamination gate still applies, and matters more.** Same 13-gram
  check against every benchmark item on disk, **and now also against every exam
  question in both halves**. A document that shares a 13-gram with an exam
  question is dropped; above 2% dropped the dataset is rejected as a generator
  echoing the exam.
- Near-duplicate collapse across documents stays.

**Tests.** `parse_items` on the `doc` format including malformed input; the
gate catching a planted exam question inside a document body; the 2% rejection;
a document set whose documents are near-duplicates collapsing; provenance
recording the format.

---

## C5 — The dashboard follows the loop

- The model page leads with **per-topic exam scores** (report half), weakest
  first, with the judge's summary per topic, κ and the canary state. The
  existing Diagnose section stays below it as the free second opinion.
- The Review tab drives from a topic: pick topic → see its score, its rank
  across models, the judge's justifications (diagnose half, labelled as such) →
  propose → approve → generate.
- Phase 6's before/after gains an exam view: for a tainted model, the chosen
  topic's report-half and diagnose-half exam scores before and after, side by
  side with standard errors, and the same three derived sentences (taught the
  skill / taught the test / no change).
- Keep every dashboard convention: one file, `el()`, hash routing via
  `navigate()`, palette variables, no red-green as the only encoding,
  zero-anchored bars, thresholds visible on the chart they apply to.

**Tests.** Extend `test_review_ui.py` and `test_dashboard.py`; screenshots in
light and dark at 1240px and 430px attached to the PR.

---

## A new requirement: the three LLMs must not be the same family

The exam is written by an LLM, the answers are graded by an LLM, and the
training data is written by an LLM. If all three are the same model family, the
loop grades its own family's questions with its own family's judge and fixes
the result with its own family's data. Self-preference in LLM judges is
documented and large, and here it would be invisible — every number would move
in the right direction for the wrong reason.

So:

- `service/config.py` gains `EXAM_PROVIDER`/`EXAM_MODEL` (C1),
  `JUDGE_PROVIDER`/`JUDGE_MODEL` (C2) and keeps `LLM_PROVIDER`/`LLM_MODEL` for
  generation (C4).
- **Startup refuses to run the judge when its provider matches the exam
  writer's or the generator's**, with the reason stated in the UI rather than a
  crash. A documented override exists for a single-provider trial, and using it
  stamps every judged score with a visible "single-provider loop" caveat.
- The cheapest correct arrangement today is one provider for writing and
  generating, the other for judging. Record all three in every provenance
  record.

**Tests.** The refusal, the override, and that all three identities reach
provenance.

---

## Migration and deploy

- The existing `proposals`, `datasets`, `llm_batches` tables and the two
  `truns` columns stay; add rather than rewrite. New tables for the exam bank
  and candidates, additive, applied on startup like the last set.
- The first deploy after C2 must not silently re-base scores: existing
  `judge.json` files were written by a local judge. Keep them, mark them with
  their judge identity, and show API-judged and locally-judged scores as
  different series rather than merging them.
- Every PR's description ends with the deploy commands and what to click:
  `cd ~/benchmarks/aienh && git pull origin main && sudo docker compose up -d
  --build`, plus any one-off script run.
- A copy of `service.sqlite3` before each deploy that migrates.

## Definition of done

CI green, including the dashboard job. `pytest -q` under a minute for the
non-dashboard suite. The full loop runs end to end against the `fake` backend
in CI: write exam → curate → sit → judge → weak topic → propose → approve →
generate documents → gate → provenance → taint → before/after. And the one
rule holds, proved from recorded request bodies at every step that calls an
LLM: **no report-half question, of the benchmarks or of our own exam, ever
reaches a human or a model.**
