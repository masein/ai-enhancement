# Handoff — the evalboard platform and the improvement loop

*Written 2026-09-19 for whoever picks this up. Read this first; it tells you what
exists, why it is shaped the way it is, what is deployed, what is not, and what
to do next. Every path and command below was verified against the repo and the
server at the time of writing.*

---

## 1. What this is, in one paragraph

A team benchmark platform for small language models on one shared RTX 5090.
People submit a model (a Hugging Face id or an uploaded checkpoint); a service
queues it, runs lm-evaluation-harness on the standard suite (MMLU, HellaSwag,
ARC, Winogrande, PIQA, TruthfulQA, GSM8K, two perplexity slices), and a single
self-contained HTML dashboard shows the leaderboard with per-item diagnosis of
*why* each score is what it is. On top of that sits the **improvement loop**:
an LLM-written, human-curated free-response exam across ~15 topics; an LLM
judge that scores each model per topic and explains the failures; a human who
picks the weakest topic; an LLM that writes training documents for it; someone
fine-tunes on them and resubmits; and the exam is sat again to see whether the
training taught the skill or just the test. Everything is built so that the
half of every test used for the published score is never seen by any person or
model.

---

## 2. Where everything lives

| What | Where |
|---|---|
| GitHub | `Teraformer-LIMITED/evalboard`, branch `main` |
| Development checkout (Mac) | `~/Developer/ai-enhancement` — remote is named **`evalboard`** (there is also an `origin` pointing at a personal repo; ignore it) |
| Server | `teraformer-5090-3`, user `masein` |
| Server checkout | `~/benchmarks/aienh` — **this** is the git repo; remote is named **`origin`** |
| Server data root (`BENCH_ROOT`) | `~/benchmarks` — holds `results/`, `eval_tasks/`, `logs/`, `artifacts/`, `datasets/`, `exam/`, `service.sqlite3` |
| HF cache | `~/hf-cache` (bind-mounted; holds downloaded weights *and* accepted licences, e.g. gemma) |
| Compose file | `~/benchmarks/aienh/docker-compose.yml`, `.env` beside it |
| Dashboard | `http://<tailscale-ip>:8899/` — tailnet only |
| API docs for submitters | `API.md`, `FRIENDS.md` (served at `/guide`), `clients/bench_client.py` (served at `/client`) |

**Push / pull / deploy — the commands that actually work.** (Two earlier
attempts guessed these paths wrong; do not improvise.)

```bash
# Mac
cd ~/Developer/ai-enhancement && git push evalboard main

# Server
cd ~/benchmarks/aienh && git pull origin main && sudo docker compose up -d --build
```

`up -d --build` kills an in-flight evaluation (it re-queues on restart) — check
the queue on the dashboard first. Take a copy of `~/benchmarks/service.sqlite3`
before any deploy that migrates the DB; migrations are additive and automatic.

Scripts that read the results tree run **on the host as root**, not inside the
container (the container image does not carry `scripts/`, and the results tree
is root-owned):

```bash
cd ~/benchmarks && sudo python3 aienh/scripts/diagnose.py results/full
```

As of writing, `main` on the Mac was **1 commit ahead** of `evalboard/main`
(the phase-8 brief). Push it.

---

## 3. The repo in one page

```
service/                 FastAPI service (uvicorn, in Docker)
  app.py                 API + serves the dashboard live (same page as the static report, data fetched)
  worker.py              one daemon thread, one job at a time — the GPU etiquette
  runner.py              runs lm_eval per task; the MISSING-weights guard; _child_env strips secrets
  db.py                  sqlite schema + additive migrations (submissions, truns, proposals, datasets, llm_batches…)
  llm.py                 ONE LLM client, batch API only; backends anthropic / openai / fake; three ROLES (llm, exam, judge)
  llm_poller.py          polls persisted batch ids; survives restarts
  proposals.py           find the gap → spec → generate documents → gate → provenance → taint
  contamination.py       13-gram overlap vs every benchmark + exam item, both halves; near-dup collapse
  hfmeta.py              preflight metadata from the Hub
  config.py              every env knob, with the reasoning in comments — read it
scripts/
  report_lm_eval.py      THE dashboard: one file, Python builds the payload, JS renders it. No framework, no build step. ~5k lines
  diagnose.py            per-item diagnosis from --log_samples; the report/diagnose split (split_of); categories; findings
  categories.py/.yaml    MMLU's 57 subjects → ~15 human topics (economics, law, medicine…). The exam's spine
  exam_build.py          draft candidates via LLM → curate → build lm_eval tasks; the control set from MMLU's diagnose half
  judge.py               grade answer scripts 0–4 against rubrics via the API; canary; family check; writes judge.json
  judge_calibrate.py     export 100 answers to CSV for a human → import → Cohen's κ
  run_benchmarks.sh      the manual CLI path; shares the same lock as the service
  make_ppl_task.py       pinned corpus-perplexity tasks
eval_tasks/
  fr/                    rubrics (exam.md + per-category), canary.jsonl, legacy seed items
  mmlu_perm/             the permutation control (options rotated) — run as suite=control
tests/                   fixture generator in tests/fixtures/make_fixture.py; scripts/check.sh runs them all (§ Checks)
docs/
  design-diagnose-and-generate.md     the original design (phases 0–2 there = T…6 here)
  prompts/phases-4-5.md               brief that built T, 3b, 4, 5, 6 — superseded where phase-7 disagrees
  prompts/phase-7-exam-driven-loop.md brief that turned the loop around — implemented
  prompts/phase-8-local-backend.md    the local vLLM backend + the narrated demo — implemented
  prompts/phase-8b-medicine.md Dr. Hossein's medicine bank + per-criterion grading — implemented
  prompts/phase-8c-dashboard-and-demo-visibility.md the demo's own page, the page as found, import from it — implemented
  prompts/phase-8d-law-topic.md       the law topic, his criteria schema, generalised flags and breakdowns — implemented
  prompts/phase-8e-loop-ux.md         next: the loop's UX
  medicine-criteria-v1.md      his first 15 criteria, as delivered (superseded by the v2 notes)
  medicine-criteria-v2.md     his medicine criteria and the critical-failure rule, as delivered
  law-criteria-v1.md             his law dataset and criteria suggestions, as delivered
  law-criteria-v2.md          his 23 law criteria and two flags, as delivered
DIAGNOSE.md              how to read a diagnosis; the finding→action table; the log to keep
DEMO.md                  the whole loop in one command against the local model — and what it does not prove
SERVICE.md               how to run the service; Docker; troubleshooting
```

**Read in this order if you have an hour:** `service/config.py` (every knob
explains itself) → `scripts/diagnose.py` module docstring → `service/proposals.py`
docstring → `scripts/judge.py` docstring → `docs/prompts/phase-7-exam-driven-loop.md`.

---

## 4. The loop, as Omar specified it

Verbatim: *"check the run text questions evaluations and check the answers
(questions are generated with ai before in different topics). after evaluating
the ai would say the score. then we choose which topic or category to make
better based on the score and evaluation generated by ai. the dataset would get
generated by ai. and this is the cycle so someone would finetune with this data
and resubmit the model and we would test again."*

| # | Step | Who / where | API cost |
|---|---|---|---|
| 0 | **Write the exam** — LLM drafts candidates per topic, a person curates on the Exam tab | `exam_build.py draft` → dashboard | once, ~$6–10 |
| 1 | **Sit the exam** — the checkpoint answers and explains | lm_eval on the 5090, `--suite judged` | free |
| 2 | **Judge** — score per topic + written evaluation | `judge.py` via API | ~$4–6 / cycle |
| 3 | **Choose the topic** — human reads scores + judge's notes, approves a skill spec | Review tab | ~$0.04 |
| 4 | **Generate the dataset** — prose documents for that topic; generator sees only the spec | `proposals.py` via API | ~$5–6 / cycle |
| 5 | **Fine-tune and resubmit** | Roohi, `--gap-dataset <id>` | free |
| ↻ | Back to 1 — same exam; the before/after says whether it worked | dashboard | |

**~$10–12 per cycle at batch prices** (verified 2026-09-18; workbook
`llm-api-budget.xlsx` has the assumptions). Cost is not the constraint —
curation time is (§8).

The standard benchmarks keep running on every submission at zero API cost and
stay on the board as the comparable public number and a second opinion. They no
longer drive the loop.

---

## 5. Non-negotiable constraints

These are decisions Omar made explicitly. Do not relitigate them in code; raise
them with him if you disagree.

**Security**
- **The tailnet is the auth boundary.** No login, no tokens — Omar declined
  authorization explicitly. `BIND` is the Tailscale IP; never `0.0.0.0` on the
  host side. Anyone on the tailnet can do anything; "who approved" is a free-text
  name field.
- **Never execute submitted code** except via the gated path: uploads only
  (never Hub `trust_remote_code`), drop to `EVAL_USER` (`benchjob`), fail closed
  if the drop cannot happen. Artifacts are **safetensors-only** — pickle `.bin`
  executes code on load and is refused.
- **Secrets live in `.env` only.** Compose interpolates them; `docker-compose.yml`
  never carries a literal; every secret's variable name is in
  `service/runner.py::_child_env`'s strip list so a submitted model's code cannot
  read it (a test greps for this). Known limit, documented: `docker inspect`
  shows `Config.Env` to anyone with docker access on the box — keys are protected
  from submitted code, **not from colleagues**.
- HF tokens 0600 in `$HOME`, never on shared volumes. Never put personal SSH
  keys on the shared server — deploy key or repo-scoped PAT only.

**The shared machine**
- One RTX 5090, 32 GB, shared. One lm_eval at a time, enforced by
  `results/.run.lock` (the service and the CLI share it; `pid: "host"` in compose
  keeps the liveness check truthful). Check `nvidia-smi` before launching
  anything. Never take a colleague's memory.
- Never fill the root filesystem. Everything new lands under `BENCH_ROOT`
  with a quota (artifacts 8 GB each / 150 GB total; datasets 20 GB).

**The one rule — the split**
- Every benchmark question *and* every exam question is assigned by
  `sha256(salt + content)` to a **report** half or a **diagnose** half
  (`scripts/diagnose.py::split_of`, salt `evalboard-split-v1`). Identical on every
  machine, for every model, retroactively.
- **Report half: never shown to a person, never placed in an LLM request, never
  used to derive training data.** It produces the published score.
  **Diagnose half:** the only half a human or the proposal step may read.
- Tests prove this from recorded request bodies at every LLM call site. Keep
  those tests. Changing the salt re-splits everything and invalidates every
  published score — it is a constant, not a setting.
- Why it matters more for the exam than for MMLU: the exam is our own generated
  questions; nothing external would catch a loop that trained on its own test.

**The three LLMs**
- Exam writer (`EXAM_*`), generator (`LLM_*`) and judge (`JUDGE_*`) are separate
  identities. **The judge's provider must differ from the other two**, or the loop
  grades its own family's questions with its own family's judge. Enforced at
  startup; `ALLOW_SINGLE_PROVIDER_LOOP=1` overrides for a trial and stamps every
  judged score "single-provider loop".
- **`JUDGE_MODEL` must be a dated id**, never a floating alias — a vendor update
  behind an alias silently re-bases every score (we hit exactly this one level
  down when `transformers` drifted 5.15 → 5.17 and broke a model load).
- Judge scores are **preliminary below Cohen's κ 0.60** against a human sample,
  and when the 30-script canary drifts past `JUDGE_CANARY_MAX_DRIFT` (0.5).
  Preliminary = shown, never ranked, never in an average.

**Data**
- Generated training data is **prose documents**, not question-and-answer pairs
  ("our generated dataset doesnt need to be q nad answer it could be a simple
  document"). Q&A shaped like the exam is the shortest route to teaching the test.
- The contamination gate drops any generated document sharing a 13-gram with any
  benchmark or exam item, both halves; **>2% dropped rejects the dataset** as a
  generator echoing the test.
- A model trained on a generated dataset carries a visible **taint** badge and
  loses that task from its official average. The per-task score stays visible.

**Process**
- **Commit messages carry no AI attribution lines** ("remove the authored claude
  stuff from commit message"). Style: `area: what changed`, body says *why*.
  Match `git log`.
- Batch API only, always (50% off both providers; nothing here is
  latency-sensitive).
- The dashboard is one file. No framework, no bundler, no second page. Follow
  its conventions (`el()`, hash routing via `navigate()`, palette variables,
  never red-green as the only encoding, zero-anchored bars).

---

## 5b. Checks — there is no GitHub CI

Since 2026-09-24 the account's 2,000 Actions minutes are used up.
`.github/workflows/ci.yml` is kept but runs only when started by hand
(`workflow_dispatch`, Actions ▸ ci ▸ Run workflow). A push or a PR starts
nothing, so PRs show no failed job and use no minutes. **The check is
local, and it is the gate.**

1. **Before every merge, run `scripts/check.sh` on the branch head**, with
   nothing left uncommitted.
   - It runs what CI ran, in CI's order:
     - ruff;
     - a compile pass over `scripts/ service/ clients/`;
     - the unit and API tests (`-m "not gpu and not network and not dashboard"`);
     - the browser suite (`-m dashboard`).
   - Every step runs even after one fails, and it ends with two lines:
     ```
     check of b273738 (clean) · 2026-09-24 14:05 +0400
     lint ok · unit 516/516 · browser 398/398 · 19 min
     ```
   - `(UNCOMMITTED CHANGES)` in place of `(clean)` means it checked
     something other than the commit, so it doesn't count.
   - It needs `requirements-dev.txt` installed and
     `playwright install chromium` done once. `PYTHON=.venv/bin/python
     scripts/check.sh` picks the interpreter.
2. **Paste the summary line, the commit and the date into the PR description
   under "Local check".** No summary, no merge. A push after the check needs
   a new check.
3. **After every deploy, run the unit suite inside the running `bench`
   container.** This tests the real image's Python and packages, not only the
   laptop's. It is deploy step 3 (below). The image has carried pytest and
   httpx since 12b.1.
   - **The tests aren't in the image.** They read files the image leaves out
     (the Dockerfile, `.dockerignore`, `docker-compose.yml`, the docs), so
     step 3 streams the commit just deployed into `/tmp/check` inside the
     container with `git archive`.
   - **It runs with an empty environment (`env -i`).** The container's
     environment holds the live `BENCH_ROOT`, the live database and the real
     API keys. `service/config.py` falls back to `BENCH_ROOT` for any path a
     test doesn't redirect, and a key in the environment changes what a
     "not configured" test sees. With `env -i`, every fallback lands in
     `/tmp/check`, as in a fresh CI checkout.

**The deploy steps, from 12b.1 on:**

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
git archive HEAD | sudo docker compose exec -T bench sh -c 'rm -rf /tmp/check && mkdir /tmp/check && cd /tmp/check && tar -x && exec env -i PATH="$PATH" HOME=/tmp/check LANG=C.UTF-8 python -m pytest -q -p no:cacheprovider -m "not gpu and not network and not dashboard"' 2>&1 | tail -15
```

**Step 3's expected output:** the last line reads `N passed, M deselected
in …s`, with no `failed` and no `error`.
- `test_matches_the_installed_harness` skips on a laptop, where lm_eval isn't
  installed. It runs here, because the image has lm_eval.
- A test that fails here and passes in `scripts/check.sh` is a difference in
  the image: its Python, a package version, or running as root. The page is
  already up at that point, so step 3 doesn't block the deploy. Send the
  output, and the fix goes in the next PR.

---

## 6. How we got here — the decisions and their reasons

Chronological. Each of these changed the shape of what was built.

1. **A checkpoint with invented weights scored on the board.** `transformers`
   was unpinned (`>=4.55`), drifted 5.15 → 5.17, and a checkpoint-key rename
   left `lm_head.weight` randomly initialized — the run "succeeded" and produced a
   near-chance score. → The **MISSING-weights guard** in `runner.py` parses the
   load report and discards the task output before the exit-code check.
   `transformers` is still unpinned pending Roohi's minimum version.

2. **Per-item diagnosis before any generation.** Design doc §2: retrofitting a
   held-out split after a generator exists is how contamination gets
   grandfathered in. So the split landed first, then the read-only explain view,
   then a deliberate pause to look at real failures.

3. **What the failures actually were.** Across 33 real models: every
   position-skewed model was ≤360M (SmolLM2 family puts 92% of MMLU answers on
   A/B against a uniform key), pythia-31m picks the shortest option 100% of the
   time, gemma base vs -it reach the same MMLU score by opposite mechanisms.
   Conclusion, stated carefully: **for those models the MMLU score is not a valid
   measurement of knowledge**, so no number from it says what to train. (An
   earlier stronger claim — "not a single flagged model has a knowledge gap" —
   was an overstatement and was withdrawn; the honest version is "we can't tell,
   because the instrument is broken.") The `mmlu_perm` permutation control exists
   to settle whether the bias is ours or the models'. **It has not been run yet.**

4. **The loop got turned the right way round.** The first build (phases T–6)
   found the gap by reading failed MMLU questions, with the exam as a side
   control. Omar's description made the exam *the* instrument: the judge already
   says per topic, in words, what went wrong. Phase 7 rewired it. The safety
   properties did not change; their inputs did.

5. **Judge: local → API.** The design argued for a local pinned judge
   (reproducible forever). Omar chose the API. What was lost — byte-identical
   determinism — was replaced with a **dated model id + prompt/rubric hashes in
   provenance + a 30-script canary re-graded every run**.

6. **Three families, not one.** Once the exam writer, judge and generator were
   all LLMs, self-preference became a loop-level risk. Hence the provider rule.
   It also settles "do we need Claude?" — yes, as the second family.

7. **Cost is not the bottleneck; curation is.** Budget came out at ~$10/cycle.
   But a topic needs **30 report-half questions** before anything can be
   proposed from it (`PROPOSE_MIN_N = 30`, the same noise floor as DIAGNOSE.md),
   the split is even so that is **60 accepted per topic**, and 15 topics is
   ~1,350 candidates to read — roughly 17 hours of expert time. Recommendation
   on record: **start with three topics**, get the loop turning, add topics as
   rubrics get written. The gate is per topic, so this works natively.

---

## 7. Current state (2026-09-19)

**Merged on `main`, CI green, 185 tests (164 pass, 1 skipped for lm_eval, 20
deselected gpu/network/dashboard), non-browser suite ~34 s:**

- Phase T: fixture tree, test suite, CI
- Phase 3b: `categories.yaml` (57 subjects → topics) and `mmlu_perm`
- Phase 4: proposals, generation, human approval, contamination gate, provenance, taint
- Phase 5: judged free response, `fr_control_mmlu` ("knew it, couldn't pick it")
- Phase 6: before/after on both halves for tainted models
- Phase 7 (C1–C5): exam as instrument, split by `qid`; API judge with canary;
  gap from judge's justifications with question text stripped; prose documents;
  dashboard follows the loop

**Deployed on the server:** the diagnose view (releases through the cache fix
`ca7e996`). **Phases T–7 have not been deployed** — the last deploy predates
them. First server session is: pull, `up -d --build`, run `diagnose.py`, then
the rollout in §9.

**Since merged (phase 8 and 8b):** the `local` provider — vLLM behind the batch
interface, everything it produces stamped provisional — the narrated
`scripts/demo_loop.py` and `DEMO.md`, Dr. Hossein's imported medicine bank with
a rubric per topic, and per-criterion grading with the 0–4 folded in code (§8,
§10). **Not built and not planned:** thinking-trace export (was in an early
plan; dropped).

**Never exercised against anything real:** the Anthropic and OpenAI batch
clients and the judge. They have only run against the `fake` backend and stubs.
The first live proposal, judged run and generation are where that glue gets its
test. Budget a day for things not to work the first time.

---

## 8. People

| Who | Role in the loop |
|---|---|
| **Omar Affifi** | Owns the platform and the decisions in §5. Runs Claude Code against the briefs in `docs/prompts/`. |
| **Dr. Hossein** | Owns the exam's substance: signs off the topic list, **writes the rubrics** (what a 0 and a 4 look like, per topic), curates the drafted questions. Rubrics are the critical path. **First delivery is in** — see below. |
| **Roohi** | Trains. Consumes generated datasets via `--gap-dataset`, resubmits checkpoints. **Open: his fine-tuning cycle time, and whether prose-document JSONL drops into his training mix as-is.** Also owns the `transformers` minimum-version answer. |

A rubric is the marking scheme: anchored 0–4 descriptions, a stated priority
(e.g. reasoning chain over right answer), length named explicitly (judges reward
length). Example: `eval_tasks/fr/rubrics/reasoning.md`. Its hash goes into every
`judge.json`; change the rubric and scores before/after are not comparable.

**Dr. Hossein's delivery was in the repo and in the loop (phases 8b and 8d).
Phase 10 retired it** — the files below now live under
`eval_tasks/fr/retired/` (rubric pairs in `retired/rubrics/`), kept as the
record of what their judged runs were graded on; the live exam is the 37
topics of §10c:

| What | Where |
|---|---|
| 100 consumer health questions with metadata, as delivered | `eval_tasks/fr/retired/medicine_v2.json` |
| 100 law questions with metadata, his own difficulty levels and `jurisdiction_required` | `eval_tasks/fr/retired/law_v2.json` |
| 100 questions each for computer science, economics and physics & engineering | `eval_tasks/fr/retired/computer_science_v1.json`, `economics_v1.json`, `physics_engineering_v1.json` |
| their criteria files and **his own prose rubrics** — not drafts, he wrote the 0–4 anchors | `eval_tasks/fr/retired/rubrics/{computer_science,economics,physics_engineering}.{md,criteria.json}` |
| his criteria files, **verbatim** — the platform's schema is his | `eval_tasks/fr/retired/rubrics/medicine_health.criteria.json`, `law.criteria.json` |
| his scoring notes for each, as delivered | `docs/medicine-criteria-v2.md`, `docs/law-criteria-v2.md` |
| the 0–4 rubrics derived from them — **DRAFT**, anchors not yet reviewed | `eval_tasks/fr/retired/rubrics/medicine_health.md`, `law.md` |

They import into their topics with him as the approver (`exam_build.py
import`, AUTHORING.md, or the Exam tab's import panel), the judge grades
those topics criterion by criterion and folds the 0–4 in code, and the page
shows the per-criterion row, one line per flag in words, and one breakdown
table per metadata field the topic carries. A criteria file names its own
flags and their effects (`zero_score`, `cap_at_N_of_4`, `score=N`); nothing
about any topic is special-cased in the code.

**His files go in as he writes them.** Three deliveries have arrived in three
shapes — flags as a list, as `critical_flag`, as `critical_error_flag`; the
criterion slug in `id` or in `name`; effects spelled `zero_score` or
`score=0` — and the loader reads all of them into one internal shape
(`judge.normalise_criteria`). The decision (2026-09-20) is that we extend the
loader, never ask him to rewrite a file. `docs/CRITERIA-SCHEMA.md` is the
table of every variant and what it maps onto; `tests/test_criteria_schema.py`
asserts every delivered file still normalises.

**The question floor is cleared.** Every topic with questions is at 100,
about 50 report-half each (the smallest has 40), over the 30 a topic needs
before anything may be proposed from it. The demo says so rather than asking
for more. That was true of the two banks here, and it is true of the 37 of
§10c.

**What is open with him, and the first is a blocker for calling any score on
these topics a result:**

1. **Rubric sign-off.** *(moot since phase 10)* The two prose rubrics said
   DRAFT: their 0–4 anchors were derived from his criteria and he had not
   reviewed them. Both are retired now; none of the 37 rubrics of the
   37-topic exam says DRAFT. The rule stands for any rubric that does: until
   the word is removed, every judged score for the topic is stamped draft on
   the page, and removing it changes the rubric's hash, which is correct —
   scores from before and after are then not comparable.
2. *(closed)* The law difficulty levels and `jurisdiction_required` arrived
   in `law_v2.json`: his own levels on every item (55 of them differ
   from the id-range mapping we had assumed) and the jurisdiction flag on all
   100. The reference line says the flag in words, so the judge reads it, and
   the page tabulates law's means for the 85 against the 15.

Not yet done and worth planning with him: per-criterion calibration. The
export writes a column per criterion and the flag, and the import reports the
agreement, but κ — the gate — is still on the folded score alone.

---

## 9. What to do next — the rollout order

1. **Deploy** (§2). Copy `service.sqlite3` first. Run `diagnose.py` once so
   every model gains its category block.
2. **Configure the three identities in `.env`** on two different providers.
   Judge on a **dated** model id. The service refuses to start/judge and says
   why if they clash or the id is an alias. Cheapest correct arrangement: one
   provider writes + generates, the other judges.
3. **Exam:** `scripts/exam_build.py migrate` (brings the 40 legacy seed items
   in), `draft --topic <t>` for the first three topics, then **curate on the
   Exam tab** to 60 accepted each. Rubrics for those three topics must exist
   first.
4. **First judged run:** submit a small model with `--suite judged`. Then
   **calibrate**: `judge_calibrate.py export` → a human marks 100 answers →
   `import` → κ. Below 0.60 the suite stays preliminary.
5. **Turn the loop:** weakest topic → propose → approve → generate → hand the
   dataset id to Roohi → resubmit → read both halves on the model page.
6. **Run the permutation control** on SmolLM2-360M (`--suite control`) when the
   card is free. Settles whether small-model position bias is our prompt's fault.

---

## 10. Phase 8 — implemented and merged

Briefs: `docs/prompts/phase-8-local-backend.md` (P0–P3) and
`docs/prompts/phase-8b-medicine.md` (P4a–P4c). Read **DEMO.md** first:
it has the `.env` block, the one command, and what a green run does not prove.
What landed: the `local` provider (`service/llm.py::LocalOpenAI`), the
provisional stamp on every artefact a local identity makes,
`scripts/demo_loop.py`, `exam_build.py import` with a rubric per topic, and
per-criterion grading folded to a 0–4 in code. Phase 8c gave the demo its own
page and the Exam tab an import panel; phase 8d made the criteria-file schema
the author's own, with a list of flags per topic and a breakdown table per
metadata field. What is still true of the box:

- vLLM OpenAI-compatible server at `http://localhost:8000/v1`, **loopback only**
  (tunnel with `ssh -L 8000:localhost:8000`)
- served model id `chat`, weights `google/gemma-4-E4B-it`
- `api_key` required by client libraries, ignored by vLLM; `GET /v1/models`
  lists what is served; honours `temperature`, `max_tokens`,
  `response_format={"type":"json_object"}`
- **Shared card:** vLLM holds 13.3 GB; with Qwen in Ollama the card sits at
  ~94%. Long-context requests can push it over.

What the briefs asked for, and what it became:

- **P0** cleanup: `rm -rf logs/_testenv logs/_repo_snapshot.tgz` (leftovers
  from a test run; gitignored) and add `.claude/` to `.gitignore`.
- **P1** a `local` provider in `service/llm.py`. **The fact that shapes it: vLLM
  has no Batch API** — no `/v1/batches` — so this is *not* `OpenAIBatches` with
  a different base URL. The backend presents the same `submit/status/fetch`
  interface and fulfils it with ordinary chat completions on a worker thread,
  concurrency 2, `max_tokens` capped, retry-then-degrade per request (one failure
  never fails the batch), results persisted so a restart resumes. Adds
  `json: bool` to `Request` → `response_format` (Gemma wraps JSON in prose
  otherwise). **Every artefact from a `local` identity is stamped provisional
  with no off switch** — a local model id is unpinnable and a 4B model is not a
  judge to publish from. Greyed, unranked, never averaged.
- **P2** `scripts/demo_loop.py` — one narrated command that drives draft →
  accept → build → sit → judge → weakest topic → propose → generate → gate on a
  small model, printing every artefact and **asserting live that no exam question
  text reached any request body**. Everything under `$BENCH_ROOT/demo/`, stamped
  `demo`.
- **P3** `DEMO.md`, which must say plainly: this exercises the callers, the
  split, the airlock, the gate and the dashboard. **It does not exercise the
  Anthropic or OpenAI batch clients.** A green demo is not a green production
  path.

- **P4a–P4c** (phase 8b) Dr. Hossein's medicine bank: `exam_build.py import`
  with metadata as the reference, a rubric per topic with a DRAFT stamp,
  per-criterion grading whose 0–4 is folded in code, and the docs above.

All of it is merged. What has NOT happened yet: a full `--sit here` run on the
box against vLLM with a real model on the card, and Dr. Hossein's sign-off plus
the ≥10 further questions (§8). Until both, every medicine score carries two
stamps and counts for nothing.

## 10b. Phase 9 — a dashboard people like using

Brief: `docs/prompts/phase-9-ux.md` (four PRs: 9a–9d). The page was correct
and read like an audit log; phase 9 changes how it reads, never what it says —
the report half, the provisional stamps and taint are untouched.

- **9a — the five asks.** Model search in both model-id boxes
  (`GET /api/models/suggest`: the board's own models first, then the Hub via
  the service, cached, two-second limit). The answers as cards that never
  scroll sideways, criteria as 14 px cells with the three weakest named, a
  "criterion below 0.5" filter. `X-Evalboard-Build` on every `/api/*` answer
  and `<meta name="evalboard-build">` on the page: a page from an older build
  shows a bar and reloads itself after a minute, never mid-typing. **Propose
  with a warning**: the gate is split into `soft` (about the judge) and `hard`
  (about the data), both always evaluated; soft-only is **Propose…** behind a
  dialog, and the proposal, spec, dataset `provenance.json` and taint trail
  carry "proposed over a provisional judge". `ALLOW_PRELIMINARY_OVERRIDE=0`
  restores the old gate word for word. Propose starts only on the topic page;
  the model page and Review link to it. One `pager()` for the answers, queue,
  Models, Leaderboard (ranked first), Provenance and Training.
- **9b — the first screen and the way around.** The checks are one line on
  every tab (`checks` in the payload: key, severity, one-line summary, a Show me
  target; the long text behind a disclosure). Overview skips duplicate rows,
  links the preliminary models, has a Loop card and a Submit button. Six tabs
  and **More ▾**; every old hash still lands. One name, in the header; the
  per-panel boxes are gone. The Loop board is one model's (`/api/loop?model=`),
  the next step is the server's and the same for everyone (no read flag),
  empty topics fold into one row, and buttons carry their context (Import on a
  topic opens the import panel on it; Sit ticks it and focuses the model box).
  No shell commands, container paths or sha prefixes on the page; long text
  behind ⓘ. The refreshed stamp has a green dot, amber when the polls stop.
- **9c — every action answers, every table fits.** Toasts for what changed
  (errors stay inline). Queue rows act: Cancel (a running job too — `canceling`,
  and the runner stops its lm_eval child), Resubmit, Open results. A judged
  Submit picks topics. Imports and accepted questions rebuild the harness tasks
  themselves; **Make new questions sittable** shows only when a judged run was
  in the way. The import panel has labels above its fields and the source as
  text. The Leaderboard shows six task columns (Columns ▾ for the rest), a
  sticky model column, the score over its error, a density toggle, duplicates
  folded under their twin, and compare starting empty. Provenance wraps. The
  model page: one caveat line, one judged topic at a time, no Unknown tile, and
  "last evaluated" counts judged runs. Training opens on the latest run; the
  theme is a menu; chart labels keep the end of a long name.
- **9d — one visual system.** Tokens on the root: spacing 4/8/12/16/24/32,
  type 12/14/16/20/28, radii 6/10, one border colour and one surface; every
  font size in the stylesheet is on the scale (glyphs under 10.5 px excepted).
  Buttons: primary, secondary, quiet, danger; disabled looks disabled and says
  why beside it. One badge with three tones (neutral, warning, danger) and at
  most one warning per row — the Loop board's provisional / single provider /
  draft rubric triplet is one line above it. One table: plain-case headers
  that stick while the page scrolls, 40 px rows, right-aligned tabular
  numbers, the pager. Empty states name the action that fills them; loading
  is a skeleton. A focus ring on every control, and anything with a click
  handler is reachable by Tab and Enter. Checked in light, dark and dim.

Definition of done (brief): automated — no tab scrolls sideways at 1,280 or
1,512 px; the answers list and every paged table fit their container, with
room; the Leaderboard shows at most six task columns and says how many are
hidden; Overview's first screen at 1,512 × 900 is the hero and Top models;
search, pager, override dialog and version bar each have their tests; the
split, provisional, taint and leak tests are unchanged and pass. The seven
steps on the live server (search "smol", queue law, page the queue, see it
land, Propose… on law, approve and generate, the mark in provenance.json) are
for a person on the box.

---

## 10c. Phase 10 — the 37-topic exam

Brief: `docs/prompts/phase-10-knowledge-classification.md` (three PRs:
10a–10c). Omar's decision (2026-09-21): the 37 folders of
`docs/Knowledge Classification/` become the exam's topics, named exactly as
the folders are; 36 held 100 questions, a criteria file and a prose rubric
each, written by masein, and Arts was empty — its three files arrived on
2026-09-22, so all 37 hold them now. The five topics the exam had —
medicine & health, law, economics, computer science, physics & engineering —
are retired: their files and judged runs stay, as history.

- **10a — files, topics, loader, retirement.** The delivered files moved
  verbatim: banks to `eval_tasks/fr/banks/<slug>_v1.json`, criteria and
  rubrics to `eval_tasks/fr/rubrics/<slug>.{criteria.json,md}`; the retired
  five's banks and rubric pairs to `eval_tasks/fr/retired/`, outside every
  path the service reads. `scripts/categories.yaml` is the 37 topics, with
  MMLU's 57 subjects remapped per the brief's table (24 topics have MMLU
  subjects, 13 have none); **General & Multidisciplinary** replaces `other` as
  the fallback. The MMLU control set is 10 per topic with subjects, 240 items
  (it was 150). The loader reads two more flag layouts (`critical_error`, one
  object; `critical_flags`, a list) and the principles under their other
  names; acuity adds `critical` and `high` (emergency → critical → urgent →
  high → moderate → mild → routine). Two commands: `import-dir <dir>
  --approver <name>` and `retire --topic <name> --reason <text>`. **A row
  belongs to the topic its own `topic` field names, exactly**: the retired
  `law` and the new `Law` share `bank/law.jsonl`, and every read filters on
  the stored string; a retired row (`retired_at`) is out of the built tasks,
  the Loop board, the counts and imports, and is never deleted or re-split.
  `build` removes the task files of topics that are no longer in the exam
  (suite=judged runs every yaml in the tasks directory). `build results/full`
  finds the results under `BENCH_ROOT` when the working directory does not
  have them — inside the container that is `/app`, where the documented
  command could not have worked before.
- **10b — results belong to the questions they were given on.** `build`
  writes each task's `bank_sha256` (sha256 over its sorted item keys — qids,
  or MMLU document hashes for the control) into `tasks/manifest.json`. The
  runner writes the same value beside the answers
  (`<task>_<n>shot/bank.sha256`, with `answered_by.json` naming the row) and
  treats a built task as done only when the two agree; answers to another
  question set are moved, whole, to `results/earlier/<model>/`, and the task
  is answered again. A run that answered nothing new says so on its row:
  "answers reused from #46 (same questions) · re-graded". `judge.json`
  records `bank_sha256` and `topic` per task, and a result counts — on the
  Loop board, in the answers, the propose gate, the averages and the
  Leaderboard — only while that fingerprint is the task's current one;
  everything else, including every entry written before this, is **history**:
  kept in `judge.json` under `history` (score, date, judge; no items) and
  shown on the model page under **Earlier exams (retired question sets)**,
  nowhere else. `/api/answers` applies the same rule. The daily item cap is
  **per provider**: `local` is unlimited unless `LOCAL_DAILY_ITEM_CAP` is
  set, a paid provider keeps `LLM_DAILY_ITEM_CAP`, and a judge batch counts
  against the judge's provider; the Review tab shows use per provider. Also
  in 10b, from the live rehearsal: the generation register follows the bank
  — the layperson's only for a topic whose items are at least half people
  asking about their own situation (`conversational`, `context_rich`,
  `telegraphic`, or carrying `subject`), a worked explanation for everything
  else, and a criteria file's `audience` still overrides both; a field whose
  values are a sentence per question (the new banks' `intent`) is never put
  in a request. Opening a topic page ticks exactly that topic in Sit the
  exam; "Review the spec" lands on the proposal and marks it. vLLM is reached
  over a shared Docker network (`LOCAL_BASE_URL=http://gemma-vllm:8000/v1`,
  SERVICE.md § The local model).

- **10c — a board with 36 topics, and the phase-9 findings.** The Loop board
  sorts weakest first for the model in "Results for", has a search box and
  the shared pager at 25; a topic without questions folds into the "topics
  without questions" line (Arts did, until its bank arrived). The Sit panel
  and Queue → judged share one topic picker: a filter,
  All / None over what it shows, and "12 of 37 topics · about 1,200 answers ·
  about 25 min" — the minutes from the last judged runs (`/api/loop` →
  `pace`: GPU plus judge time per graded item over the last five), never a
  constant. The model page picks a judged topic from a searchable select,
  weakest first with its score. The Leaderboard's Columns menu has three
  groups — tasks (six by default), judged topics and MMLU by category, the
  last two hidden — and a **Judged avg** column is always shown (report half,
  current question sets only; a preliminary judge's greyed). The Exam tab's
  rubrics table has a search box and the pager. From the live check: a judged
  run is refused, in plain words naming the URL tried, while the judge does
  not answer `GET /v1/models` (2 s, cached 30 s); the Loop board and topic
  page say **judge offline**, *Queue this run* is disabled with the reason,
  and the header's live dot covers the judge. A row whose grading failed
  leads with "The grading model isn't running. Start it, then Retry grading —
  the answers are kept.", the raw text behind **details**, and a **Retry
  grading** button (0 GPU, by 10b's resume). A toast with a link stays 8 s and
  not while it is hovered or focused. The Queue starts on page 1 when you
  enter it and when a row you queued changes status, and marks that row; its
  pager is built once and updated in place.

**Deploy steps**, once 10a–10c are all merged — as one sequence; the `build`
step is also what writes 10b's question-set fingerprints:

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
for t in "medicine & health" law economics "computer science" "physics & engineering"; do
  sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam retire --topic "$t" --reason "replaced by the 37-topic exam (2026-09-21)"
done
sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam import-dir eval_tasks/fr/banks --approver masein
sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam build results/full
sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam summary
```

**Expected output** (replayed on a bank shaped like the box's: the five old
topics at 100 rows each, and the tasks an earlier build wrote for them):

1. `up -d --build` ends with the container healthy. The image build prints
   `image files OK` — the startup check now covers all 111 delivered files.
2. `retire`, one line per topic. If a bank has more rows under a topic than
   the 100 delivered (questions drafted and accepted there), the counts say
   so; a second run says `retired 0 of 100 rows (100 already retired)`:
   ```
   medicine & health: retired 100 of 100 rows in medicine_health.jsonl · reason: replaced by the 37-topic exam (2026-09-21)
   law: retired 100 of 100 rows in law.jsonl · reason: replaced by the 37-topic exam (2026-09-21)
   economics: retired 100 of 100 rows in economics.jsonl · reason: replaced by the 37-topic exam (2026-09-21)
   computer science: retired 100 of 100 rows in computer_science.jsonl · reason: replaced by the 37-topic exam (2026-09-21)
   physics & engineering: retired 100 of 100 rows in physics_engineering.jsonl · reason: replaced by the 37-topic exam (2026-09-21)
   ```
   A name no row carries is refused with the names the bank holds, and
   exit code 2.
3. `import-dir`, 37 lines and a total (a second run: `imported 0`, `skipped
   3700`):
   ```
   Agriculture                                  imported 100  skipped   0  (report 55 / diagnose 45)  source agriculture_v1
   AI & Machine Learning                        imported 100  skipped   0  (report 49 / diagnose 51)  source ai_machine_learning_v1
   Anthropology & Human Geography               imported 100  skipped   0  (report 49 / diagnose 51)  source anthropology_human_geography_v1
   Architecture & Built Environment             imported 100  skipped   0  (report 44 / diagnose 56)  source architecture_built_environment_v1
   Arts                                         imported 100  skipped   0  (report 56 / diagnose 44)  source arts_v1
   …
   Law                                          imported 100  skipped   0  (report 62 / diagnose 38)  source law_v1
   …
   Technology                                   imported 100  skipped   0  (report 49 / diagnose 51)  source technology_v1
   total: 37 topics, imported 3700, skipped 0 already in the bank — report 1828 / diagnose 1872 · written by masein
   ```
4. `build`: 37 `exam_*` tasks of 100 items, then the control set and what it
   removed:
   ```
   exam_agriculture                                 100 items  (report 55, diagnose 45)
   …
   exam_general_multidisciplinary                   100 items  (report 47, diagnose 53)
   fr_control_mmlu                                  240 items  (report 0, diagnose 240)

   37 exam topics built
   MMLU control set: 240 items — up to 10 from each of the 24 topics with MMLU subjects (the old 15-category exam built 150); the other 13 topics have no MMLU subjects
   removed 2 task(s) no longer in the exam: exam_medicine_health, exam_physics_engineering
   ```
   (`exam_law`, `exam_economics` and `exam_computer_science` are rebuilt in
   place with the new questions. If the bank still had the old `other` topic,
   `exam_other` is removed too, and its 40 migrated skill items stay in
   `bank/other.jsonl`, unread.)
5. `summary`: 37 lines, all at `accepted 100` with their halves as in step 3
   (`Arts  accepted 100 (report  56 / diagnose  44)`). The retired five do not
   appear.

**If the deploy steps already ran before Arts arrived**, the same commands add
it and nothing else: after the pull and `up -d --build`, skip the `retire`
loop (a second run changes nothing anyway); `import-dir` says `Arts … imported
100` and `skipped 100` for the other 36 (`total: 37 topics, imported 100,
skipped 3600`); `build` adds `exam_arts` and rebuilds the rest unchanged —
their fingerprints do not move, so no answer on them is set aside.

---

## 10d. Phase 11 — the LiveBench look

Brief: `docs/prompts/phase-11-livebench-look.md` (four PRs: 11a–11d). masein
pointed at LiveBench and Artificial Analysis for the feel of the page; we
take the ideas and write them ourselves — no code, stylesheet, font, logo,
name or wording from either site. 11a is the five findings of the 37-topic
live check; 11b the visual system; 11c the Leaderboard; 11d the Overview
cards and the model page.

- **11a — the five findings of the live check (2026-09-22).**
  - **One popover component**, for More ▾, Theme ▾ and the name menu (and
    11c's popovers later). The panel is appended to `document.body` and
    placed from its button's rect, because `div.morewrap` sits inside
    `#tabs`, which scrolls sideways: the menu was clipped to 37 px and
    almost nothing in it could be clicked. It flips above the button when
    there is no room below, stays 8 px inside the window, follows scroll and
    resize, closes on an outside mousedown, on Tab and when its button
    scrolls out of view, and carries the ARIA menu-button keyboard pattern
    (↓/Enter/Space open and focus the current or first item, ↑↓ wrap, Home
    and End, Esc closes and gives the button back). It survives a poll: the
    panel is not inside `#view`, and `popReanchor()` finds the button again
    by `data-pop-anchor` after every render.
  - **Documents are spread over the areas whose answers failed.** Dataset #2
    asked for 20 documents and wrote 11 of the 18 about relativistic
    momentum, because every request was identical but for the style seed.
    Now `focus_plan()` counts the **diagnose-half** graded answers below
    `WEAK_SCORE` per `domain` (the label comes from the bank, by qid — the
    report half is never read), allocates the documents by largest remainder
    with at least one each while the count allows, and orders the requests
    round-robin so a batch cut short still covers the spread. Each request
    carries one line, `Focus: <label>` — the label only, no count, no score,
    no qid, no question text. A topic's domains are used only when they are
    a closed set of labels (`is_domain_label_set()`: at most 15 distinct,
    each on at least 2 items, at most 64 characters and 8 words, no `.?!`);
    otherwise there is no `Focus:` line, which is what happened before. The
    plan is on the proposal's Review card **before** Generate, on the
    dataset card, and in provenance as `focus_plan`.
  - **Every document asked for is accounted for.** `parse_items()` dropped an
    item without a title, under 120 words, or in a reply that was not JSON,
    and said nothing; `_finish_generation()` reported an error only when
    every request failed. Provenance now carries `items.requested` and
    `items.missing: [{request, focus, why}]`, with `why` one of `error: …`,
    `reply not JSON`, `no title`, `too short (<n> words)`, `empty reply`,
    `not in the reply` or `dropped by the gate: benchmark|exam|duplicate`.
    The card and the toast say the line — "10 of 12 documents · 2 missing —
    2 too short (60 words)" — and a disclosure lists each missing document
    with the area it was meant to cover. A dataset made before 11a says
    "reasons not recorded (made before 11a)" rather than leaving a blank.
    **No automatic retry**: the local GPU is shared.
  - **A `local/` model whose weights are not on this server.** #53 picked
    `local/qwen35-delta-moe-7d560104-step945` from the search — it is on the
    board because its results came from elsewhere — and the run failed at
    start, after the wait. `suggest.local_candidates()` now sets
    `weights: bool` for every `local/` id; the search greys such an item,
    labels it "results only — weights not on the server", marks it
    `aria-disabled` and skips it with the arrow keys; the submit form shows
    the same sentence and disables **Queue this run**; and `POST
    /api/submissions` refuses with 422 before anything is queued: "local/<name>
    has results on this board but no weights on this server — upload it first
    (POST /api/artifacts/<name>) to run it here." An id that is not on the
    board either is refused in preflight's own words.
  - **"rubrics v?"** on the model page: the judged header built its rubric
    list with `` `v${r.version}` ``, and no author-written rubric carries a
    version. It uses `rubricVersion()` now — "rubrics no version", or "v2,
    no version" for a mix.

- **11b — the visual system.** One set of tokens, one type scale, one bar and
  one table component.
  - **Tokens.** A pale blue-grey page (`--plane` #F6F8FC), white cards, ink
    #14213D and one saturated accent (#2F54EB, 5.9:1 on white). New:
    `--live-text` / `--live-dot` for the status badge, `--bar-bg` for the
    translucent bar, and `--heat-1` … `--heat-5`, five steps of the accent
    mixed into the card surface for 11c's rank tint. Dark and dim keep their
    surfaces and take the lighter accent #7C9BFF with stronger heat steps.
    `--font-sans` and `--font-mono` are tokens now; no web font, as before.
  - **Type.** Prose in the sans face; data and labels in mono — column
    headers, numbers (tabular figures), chips, badges, the status line, the
    footer and the eyebrow. The title is 28px/800 with tight tracking; a
    section title is 20px/700 after a mono index.
  - **The sticky bar.** 56px, translucent with a blur and a hairline under it:
    the title and the `● LIVE · 12:33` badge on the left, the tabs in the
    middle, and the checks pill, the name and Theme ▾ on the right. The badge
    took over every state the old "live · refreshed" chip had (stale, judge
    offline), and a static report has no badge at all. At 720px the title
    shortens and the tabs drop to a second row in the same sticky block.
  - **The checks** are a pill in the bar; their list opens just under it. Still
    on every tab, still never dismissed.
  - **Sections are numbered** 01, 02, … down each tab, drawn from the DOM so a
    card that moves takes its place in the count.
  - **One table component**: mono uppercase sticky headers, 44px one-line rows,
    hairline separators, `--accent-soft` on hover, a 3px accent bar on an
    opened row, right-aligned mono numbers, and a `pin` class for the first
    column (11c pins the model column at phone width).
  - **A status line** above the big tables: "Showing 1–25 of 32 models · 15
    ranked · sorted by Avg ▼ · ● live 12:33".
  - **Back to top**, a mono pill after two screens, which moves focus to the
    bar; **one tooltip**, inverted (ink background, surface text), mono, at
    most 280px, on hover and on focus, wired with `aria-describedby`.
  - **Removed: the comfortable/compact density switch** — the one-line row is
    the compact one. `test_every_action_answers.py` no longer clicks it.

- **11c — the Leaderboard.**
  - **One toolbar row** (a single strip that wraps on a narrow card): the
    topic-group chips `All tasks · Knowledge · Commonsense · Reasoning · Math ·
    Truthfulness · Judged topics` on the left, and on the right the pills
    `Kind: All ▾`, `Size: All ▾`, `Status: All ▾`, `Columns · N hidden ▾`,
    `Models ▾` and `Scale: above chance ▾`, all on 11a's popover. The view is in
    the hash (`#tab=leaderboard&chip=knowledge&kind=instruct&open=<id>`); old
    hashes still land. **Removed:** the Avg-scale, Cells (numbers | heat) and
    View (tasks | MMLU by category) switches, and the Columns `<details>` — the
    Scale pill, the rank tint, the Knowledge chip and the Columns pill replace
    them.
  - **Knowledge** shows MMLU and **MMLU by area**: eight columns, each the
    leaderboard-half items of the MMLU subjects mapped to that area's topics,
    pooled (a mean weighted by item count). The 24 per-topic MMLU columns are
    under Columns. **Judged topics** shows the judged area means, by
    `judged_avg()`'s rules (no tainted topic, at least half the area judged, or
    "—" with "k of n topics judged") — and only once the judge is calibrated;
    until then the chip is disabled and says why.
  - **Two header rows**: the group over its columns, then the name with its
    unit on a muted second line (`5-shot · %`). **One-line cells**: `52.3 ±0.4`,
    the `%` in the header. **The rank tint**: each cell's rank in its column
    over the whole board (so a filter never changes a colour), in five steps
    from `--heat-5`; a cell the z-test cannot tell from the column's best
    shares the top step, in bold, with `≈`. Judged cells are tinted only when
    the judge is calibrated. A new `#` column (the rank, or "—" for a
    preliminary model) replaces "#1/14" under Avg; Params is one line; a 3px
    family bar on the model cell, never colour alone.
  - **Rows open in place**: a click on a row (or ▸, Enter, Space) opens a detail
    row under it — Tasks with each rank and "tied with best", MMLU by area as
    bars above chance, Judged topics as chips by area (grey, with one
    "provisional … not ranked" line and no area mean while the judge is
    provisional), and Links (model page, Provenance, Add to radar, Run exam).
    Open rows are kept by model id in `state` and in the hash: they survive
    polls, paging, sorting and a pasted link.
  - **Avg has a standard error now** (`avgSe` / `avgRawSe` in the payload,
    `avg_se()`): the per-task errors carried through the mean, scaled with the
    score. The frontier's z-test needs it.
  - **Insights** under the table: score against size with a z-tested frontier
    (a lead inside the noise never draws the line; click a point to shade the
    models that are bigger and score lower); Weakest topics for one judged
    model (the one judged last, or chosen) — report half, weakest first, grey
    and hatched while provisional; and the radar with up to five model chips
    and three sources (Tasks, MMLU by area, Judged by area — the last disabled
    until the judge is calibrated). **Removed:** the Capability profile card and
    the compare column — the chips are the comparison. Every chart has "Show
    as table".
  - The long paragraph went behind **How to read this table ▾**, collapsed and
    remembered, with About these benchmarks inside it.
  - **`scripts/areas.yaml`** groups the 37 topics into 8 areas (the brief's
    proposal; masein can change it), read by `categories.areas()` without
    pyyaml; `tests/test_areas.py` holds it to every topic in exactly one area,
    spelled as `categories.yaml` spells it.
  - The Models tab's filters are the same pills: `Kind`, `Source`, `Family`,
    and `Show:` (judged / tainted / preliminary, which combine).

- **11d — Overview and the model page.**
  - **Overview**: the hero (11b), then the four stat tiles as one mono line
    ("10 models · 8 tasks · 137 of 273 gaps are real · 4.62 h of evaluation ·
    …"), then **01 Highlights** — four cards, each a value and one sentence
    derived from the data: BEST MODEL (its lead over the next, in the z-test's
    words — "within noise" or "a real gap (z = 3.8)"), and on a live page
    WEAKEST TOPIC (the model judged last, its own weakest topic, "k of 37
    topics judged — provisional, local judge"; not a ranking, no area mean),
    THE LOOP ("7 / 37 topics judged", last judged when) and JUDGE STEADINESS
    (the canary: "30 / 30 steady", its deviations and the calibration). "Best
    official average" folded into BEST MODEL. **02 Top models** is the table
    component with `#` and the tint, "See the leaderboard →", and the biggest
    real gap as a mono line under it. **03 The loop** as before.
  - **The model page**: a hero — an eyebrow `MODEL · BASE · #3 OF 9`, the name
    as the h1, the id in mono, three cards (Parameters — with the compute
    estimate when known —, the average with its verdict against the next
    model, Tasks) and the sentence underneath. The section nav is sticky mono
    chips under the bar, numbered 01–06, and the one in view is lit
    (IntersectionObserver). The judged topics are grouped under the 8 areas —
    "k of n topics judged", an area mean only for a judge whose scores count,
    never while provisional — weakest first inside each; "the judged suite is
    preliminary" is said **once** above the table instead of on every row; the
    row's action is a small "Propose →". **Score against answer length** is one
    row per topic with a column per length (`0.87 · 86`, mean · items, a
    neutral grey by the mean — not a rank tint — and "few" under five items),
    folded behind "Show the table" past ten topics. The topic page has no such
    table.

**Deploy steps**, after each phase-11 PR merges. This phase has no data
migration: it changes code (11c also adds `scripts/areas.yaml`).

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` rebuilds the image and ends with the container healthy;
   the image build prints `image files OK` (the 111 delivered files).
2. The log grep prints `no errors`. (`grep` exits 1 when it matches nothing,
   so the `|| echo` is the pass case.)
3. On the page, after a hard reload (11a): More ▾ opens fully at 1,280 px
   and at 400 px and Esc returns focus to the button; a model page judged on
   the 37-topic exam reads "rubrics no version", never "v?"; searching for
   `qwen35-delta-moe` shows the checkpoint greyed as "results only — weights
   not on the server", and
   `curl -s -XPOST localhost:8000/api/submissions -H 'Content-Type: application/json' -d '{"hf_id":"local/qwen35-delta-moe-7d560104-step945","suite":"quick"}'`
   returns the 422 sentence with nothing queued.
4. The next Generate on a topic whose questions carry domain labels shows
   its focus plan on the Review card before you press it, and the toast when
   the batch finishes says how many documents came back and why any are
   missing.
5. After 11b: a 56px sticky bar with `● LIVE · <time>`, numbered sections,
   one-line 44px table rows. After 11c: the Leaderboard's one toolbar strip,
   tinted rows that open in place, Insights under the table, and Judged
   topics disabled with its reason while the local judge is uncalibrated.
   After 11d: the Overview's stat line and four highlight cards with their
   verdicts, and the model page's hero and numbered section chips.

### 11e — the second live check

Brief: `docs/prompts/phase-11e-live-check.md` (items 1–12). One PR. The
phase-11 rules are unchanged: only diagnose-half items are read to build a
generation request, and no provisional judged score goes into any average,
column, tint or frontier.

- **Documents spread on every topic** (masein's decision, item 1). The rule
  lives in `service/proposals.py` (`focus_scheme`, `focus_for`):
  - **By area** when a topic's labels are ≤ 25 as written, or ≤ 25 groups
    (the text before the first ` — `, ` – `, ` - `, `: ` or ` / `): the
    documents are allocated over the areas by their failed diagnose-half
    items, round-robin so any first N still spreads. 18 of the 37 banks.
  - **By concept** when even the groups are more than 25: one document per
    failed diagnose-half label, weakest first (ties by qid), round again past
    the end. 19 banks.
  - **Hygiene**: a label goes in a request only when it is ≤ 64 characters,
    ≤ 10 words and has no `.`, `?` or `!`; else its group; else it is skipped.
  - **Shown before Approve** on the proposal card ("Where the documents go"),
    with a "Spread the documents over these" box. **Approve freezes** the plan:
    `proposals.approved_focus` (a new column, added at startup) holds up to
    100 labels; Generate with count N takes the first N. Unticked → no Focus
    lines, `focus_mode: off`. When there is no plan the card says the true
    reason (no labels / every label too long / nothing failed / turned off).
  - A proposal **approved before 11e** keeps 11a's behaviour and says
    "Approved before plans were shown".
  - `provenance.focus_mode` and `provenance.focus_labels` record what was sent.
- **The bar** (2, 3): the checks pill reads `● 6 checks ▾` and is the same
  button as `masein ▾` and `Theme ▾`; "k of n are about the judged suite" is
  the first line of its panel. The pill's 40px height came from the `.warn`
  paragraph style bleeding onto `.dot.warn` (and onto the `provisional`
  badge, item 12) — both reset. The badge and the status line read the
  refresh time (`LIVE · 15:17`), not the timezone.
- **Model cell and opened row** (4, 5): the model cell is one clipped flex
  line — the name gives way first, then a long badge — and the duplicate
  toggle is inside it; the opened row's Links repeat "Same run as …" with the
  toggle. The opened row's blocks are `minmax(min(100%, 300px), 1fr)`.
- **A stale diagnosis** (6): a diagnosis whose MMLU categories are not the
  current `categories.yaml` set is stale. The Knowledge chip says once
  "MMLU by area needs a fresh diagnosis — … made with the old 15 categories",
  and an opened row says it instead of eight empty bars. Area columns that
  are empty for every model on the page are hidden (the fixture's MMLU
  subjects reach 5 of the 8 areas; the live board's reach all 8).
- **Focus in a panel** (7): a render patches the Columns panel's checkboxes
  in place, so the focused control is the same node afterwards.
- **The frontier** (8): no model of the **same size or smaller** beats it by
  a z-tested gap; the line goes through the best point at each size; the
  caption says "No model here is bigger and scores lower than X."
- **Highlights** (9): the value is the number only (`39.3`, `0.79 / 4`,
  `7 / 37`, `30 / 30`); the name is its own 16px line with an ellipsis and
  the full id in the tooltip; the canary numbers have two decimals.
- **400 px** (10): the bar is two rows (≤ 96px) — title, LIVE and a `⋯` menu
  holding the checks, the name and the theme; then the tabs. On the
  Leaderboard the rank and model take ≤ 45% of the scroller, only `prelim`
  stays of the badges, Params drops "· N act" (it stays in the tooltip),
  units wrap under their names, and the right edge fades with "scroll →"
  while there is more; a poll keeps the sideways scroll. The Insights charts
  are never narrower than their viewBox (so 12px text is 12px) and scroll in
  their own box; below 600px Weakest topics are HTML rows.
- **11** — a dataset made before 11a says "2 missing — reasons not recorded
  (made before 11a)" on the topic page too.
- **Polish** (12): 16px under the Insights intro, 12px between "Submit a
  model" and the stats line, the standard 12px `provisional` badge,
  link-styled Links buttons, and the Judged topics chip is `aria-disabled`:
  its reason is the tooltip, its `aria-describedby`, and a one-line note under
  the chips on click — no permanent line.

**Deploy steps, after 11e merges.**

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` rebuilds and ends with the container healthy; the image
   build prints `image files OK`.
2. The log grep prints `no errors`. On start the service adds the
   `approved_focus` column to `proposals` by itself; nothing to run by hand.
3. After a hard reload, at 1,512 px: every tab fully visible and none
   overlapping, `● N checks ▾` the same size as the buttons beside it, the
   badge `LIVE · <time>`, and each highlight value on one line.
4. A new Propose on **Sociology** shows, before Approve, "20 documents, one
   per concept the model missed in the practice half: …"; after Generate the
   dataset's labels are those 20. A new Propose on **Mathematics &
   Statistics** shows "Documents will cover N areas: …".
5. Until the diagnosis is re-run, the Knowledge chip says "MMLU by area needs
   a fresh diagnosis — …". Then, the data step:

   ```bash
   cd ~/benchmarks && sudo python3 aienh/scripts/diagnose.py results/full
   ```

   prints one line per model and then `wrote diagnose.json for N model(s)`,
   with no "could not write" block; after the next poll the Knowledge chip
   shows the 8 area columns with numbers and the sentence is gone.
6. At 400 px: the Leaderboard shows the Avg column, the header is two rows,
   and the chart text is readable.

### 11f — masein's seven

Brief: the second half of `docs/prompts/phase-11e-live-check.md`. One PR.
**The one rule changed on purpose:** the ± error is no longer on every cell.
It is the cell's tooltip (hover or focus), it is in the opened row, and
**Columns ▾ › Show ± errors** (off by default, remembered in this browser)
puts it on every cell. Bold keeps its meaning — best in the column or within
its noise — and no provisional score is bold, tinted, averaged or ranked.

- **The Leaderboard header** is one line of one-word names (`ARC-C`,
  `TRUTHFULQA`, `JUDGED`, `UPDATED`…) under a muted group row with a hairline
  bracket, and the group row is only on **All tasks**. The n-shot, the unit
  and the scale are each name's tooltip (focusable, dotted underline on
  hover); the ⓘ icons are gone. The JUDGED column is there only while some
  model on the page has a judged average. Dates read `15 Sep`, with the year
  only when it is not this one. One mono caption under the table says what
  bold means.
- **Cells** hold the number. `lbLeaders()` replaced the five rank-fifth
  tint steps: a column's leaders — its best, and every score the z-test
  cannot tell from it (1% band for a lower-is-better column) — are bold and
  tinted `--heat-3`; every other cell is plain. Computed over the whole
  board, so a filter never changes them. The ●/≈ glyphs are gone from the
  Leaderboard and Top models, which follows the same rule.
- **Opening a row** is one motion: `grid-template-rows` 0fr→1fr, a fade and
  a 4px rise, 200ms, only when a person toggles it (`state.lbAnim`, used
  once). A poll, a sort or a pasted link draws it open, still. The chevron
  turns 90° over 150ms. Closing plays it back; the row leaves the DOM after
  `transitionend`. The panel reads as the row's continuation (its accent bar,
  a faint wash, rounded below), with **✕ Close**; Esc on the row or ✕ gives
  the chevron the focus back. **Tasks** is a compact list — name, score, a
  thin bar on the column's scale, rank — and **Links** a list of link
  buttons.
- **A new page starts at the top.** `navigate()` and every in-page `#…`
  link push an entry that remembers where it came from; each entry keeps its
  `scrollY` in `history.state`. Back/Forward restore it (retrying while the
  view loads, until the person scrolls); "← Back to …" is Back when the entry
  before is that view. Sorting, paging, chips, opening a row and polls never
  move the scroll; a button aimed at a section still lands on it.
- **One motion system:** `--dur-1/2/3` (120/180/240ms), `--ease`, all 0
  under reduced motion. Button colours ease and a press scales to .98; one
  tab underline slides; popovers come in 4px from the side away from their
  button and leave in half the time (a keyless, inert ghost fades, so
  nothing finds a closed panel); toasts rise and fade; a new view — after a
  navigation, never a poll — fades up 6px; a value a poll changed
  (`data-watch`: Leaderboard cells, Queue status and progress) gets a 1.2s
  wash; a theme change cross-fades for 250ms. Transitions touch only
  opacity, transform, grid rows and colours.
- **One action cell** (`actCell`) for every table with actions: the row's
  next step as its one button — a ghost, except **Retry grading**, which is
  filled — and the rest in a `⋯` menu on the shared popover (Log, Resubmit,
  Copy id, Open model page). Used by the Queue, the Loop board, the topic
  page's datasets and the Review card's decide row (**Reject…** is in its
  `⋯` and asks for its reason in place). The GPU cell never wraps; a long
  failure is two lines and "details ▸".
- **No native `<select>` in the view.** `Select` (short lists: a button and
  a listbox — ↑↓, Home/End, typeahead, Enter, Esc) and `Combobox` (topics and
  models: a search field over a grouped list, `aria-activedescendant`).
  Topics are grouped by the 8 areas, weakest first, with the score and a
  tiny bar; the model page's picker replaced a select and its "find a topic"
  box. Both keep `.value` and fire `change`, so the code around them did not
  change shape. Tests pick with `conftest.choose()` and read with `choice()`.
- Found on the way: a sticky table header sat over the first row inside the
  sideways scroller below 900px, and scrolling inside a tall popover flipped
  it up under the bar. Both fixed.
- The browser tests run with **reduced motion** by default (`page` fixture);
  the motion is tested with it on, in contexts of their own.

**Deploy steps, after 11f merges.** Code only, no data step.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. After a hard reload, at 1,512 px: the Leaderboard header is one line of
   names under a quiet group row; cells hold only numbers; the leaders are
   bold and tinted and there is no blue slab. Opening a row glides open in
   about 0.2s and closes the same way; a poll never replays it.
4. Scroll down any page and open another: it starts at the top, and Back
   returns to where you were.
5. Every Queue row shows one tidy button and `⋯`, and no GPU cell wraps.
   There are no native dropdowns; the model page's topic picker is a
   searchable list grouped by area.
6. With macOS reduced motion on (System Settings › Accessibility › Display),
   nothing moves.

### 11g — read every file in the page

Brief: `docs/prompts/phase-11g-read-and-first-page.md` (11g–11j; committed
with 11g, with masein's two corrections of 22 Sep marked in it). One PR.

- **One Reader** — a sheet from the right, `min(760px, 92vw)`, the whole
  screen below 720px. Title, where the file comes from, **Copy**,
  **Download**, **Open raw ↗** and **✕ Close**. It has an address —
  `…&read=<kind>:<id>[:<n>]`, e.g. `read=dataset:6:3` — so a pasted link opens
  it and Back closes it; it lives in `state.read`, so a poll never closes it;
  Esc closes it and gives the focus back; focus stays inside while it is
  open. Everything is text: markdown goes through a small renderer that
  makes no HTML (`mdRender`), links get `rel="noopener noreferrer"`, and a
  `javascript:` link is only its words.
- **The six kinds** (`service/reader.py`):
  - `dataset:<id>` — `GET /api/datasets/{id}/items?offset=&limit=&q=`: every
    kept document numbered with its focus label and word count, the missing
    ones in their place (after their request's documents) with their
    reason; ← → between documents, a search that marks what it finds, and
    the free format as Question / Answer / Rationale. Each document's label
    is now written beside the documents when a batch lands
    (`items.meta.json`); an older dataset's is rebuilt from its provenance,
    or said to be not recorded. A dataset made before 11a says its missing
    ones' "reasons not recorded". The `items.jsonl` download is unchanged.
  - `rubric:<name>` — the markdown, a contents list, the version (or "no
    version") and the sha.
  - `criteria:<name>` — read through the judge's own `normalise_criteria()`
    (all three layouts the author has sent: `critical_error_flag`,
    `critical_error`, `critical_flags`): the criteria table, the flag(s) with
    what they do and their examples, the top-level fields, the raw JSON.
  - `bank:<slug>` — `GET /api/exam/bank?topic=…&half=diagnose`: the practice
    half only, and of the hidden half **only a count** — no qid either.
    Filters for difficulty, domain and style, and a search.
  - `log:<run id>` — `GET /api/runs/{id}/lines`: numbered lines (up to
    2,000), a search with **next ↓**, bad lines in the warning tone, a wrap
    toggle, **Load earlier lines**; it follows a running job until the
    person scrolls up. A line that quotes a hidden question (its qid, or the
    start of its text) is withheld.
  - `provenance:dataset:<id>` and `provenance:judge:<model id>` —
    `GET /api/judge/provenance?model=…` is the model's judge runs (never a
    run's plan) and its judge file's head, each topic reduced to counts,
    scores and fingerprints (never an item, never a qid). A collapsible
    tree: hashes shortened with a copy button, times local, "Demo only" as a
    badge.
- **Where Read appears**: the topic page's datasets (**Read**, Download
  beside it, Provenance in `⋯`), the Review card's and the Review tab's
  datasets, the Exam tab's rubric table (rubric and criteria, a ↓ beside
  each), the topic header (`arts.md`, `20 criteria`, and **Read the practice
  questions**), the Queue's `⋯` › Log (the raw log is the next item), the
  dataset reader's header (Provenance), and the model page's Judged section
  (**How this was graded ▸**).
- Found on the way: 11f's tab underline measured More ▾ from its own
  wrapper and drew under Overview. Fixed.

**Deploy steps, after 11g merges.** Code only, no data step.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. From the Arts topic page, dataset #6's **Read** opens the reader: its 20
   documents, each with its focus label; ← and → move; the search narrows
   the list. Dataset #2 (Physics) lists its 2 missing documents with
   "reasons not recorded". (A dataset generated before this deploy has its
   labels rebuilt from its provenance.)
4. `arts.md` reads as formatted text; the Arts criteria read as a table of
   20 criteria, then the flag with its examples.
5. **Read the practice questions** on Arts lists its 44 practice questions
   and says "56 hidden questions — never shown, by design".
6. A Queue row's `⋯` › Log opens the reader and follows a running job.
7. Every reader's link, pasted into a new tab, opens it; Back closes it.

### 11h — the first page, and the 11e findings

Brief: `docs/prompts/phase-11g-read-and-first-page.md`, 11h. One PR.

- **One font rule** for the board. Mono is for numbers in tables, cards and
  charts, column headers, eyebrows and section numbers, badges, model ids
  (`.mid`), the LIVE badge and the log reader. Everything read as words —
  pills and buttons, chips, the stats and status lines, captions, card
  links, the footer, the tooltip — is sans.
- **The Overview**: a compact hero on one row (the eyebrow, one line, the
  two guide links; **Submit a model** on the right), no second title (the
  bar has it; the `h1` stays for screen readers only), the stats line in
  sans 14px under it. The highlight cards: the value in sans 28px bold
  with tabular figures, the name 14px on one line, the verdict muted, the
  link in the accent with no underline until hover; four cards of one
  height with their links on one line. The hover on a Top models row
  covers the whole row.
- **One scale**: an area's MMLU number follows the **Scale** pill in its
  column, in the opened row's bars and on the radar (`areaScaled`); the
  column's tooltip and the bars' caption say which scale.
- **No page scrolls sideways**: a Leaderboard wider than its card scrolls in
  its own box at any width (`.hscroll`, decided after each render), with #
  and Model pinned and the fade and "scroll →" on the right; only then does
  its header stop sticking to the page.
- **The phone**: below 720px the six pills are one **Filters ▾** (with a
  count when any is set, e.g. "Filters · 2") that opens a sheet from the
  bottom; the chips are one row that scrolls; the pager is one line.
- **LIVE** shows the time of the last check that worked, so it moves with
  every poll; the tooltip says "data last changed 17:29 · checked 12 s
  ago"; after two missed intervals it turns amber: `STALE · 17:29`.
- **Plain words** (§7) on every main tab: "hidden questions" for the report
  half, "practice questions" for the diagnose half, "the AI", "the missing
  skill", "agreement with a person" for κ, "scored below 3 of 4", "the copy
  check". Task ids, judge.json and fingerprints moved to tooltips, Details
  and the Provenance tab. The propose gate's own reasons follow suit. The
  Review card's blank failure count (`ev.diagnose_wrong`) now reads
  `diagnose_weak`.
- **"Sandbox run" is gone**: the menu item, `GET /demo`, `payload["demo"]`
  and its cache stamp. `scripts/demo_loop.py` and `DEMO.md` stay, as a
  command-line tool; DEMO.md says the dashboard no longer links to it.
- 11h §6 (the opened row's Tasks list) needed nothing: 11f's list is one
  line a task at 1,280px.

**Deploy steps, after 11h merges.** Code only, no data step.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. The Overview: one compact hero row with **Submit a model** on the right,
   no second title, four cards with a sans number, and Top models visible
   without scrolling at 1,280px and up.
4. The Knowledge chip at 1,512px: the page does not scroll sideways; the
   table does, in its own box. Its area numbers match the opened row's bars
   under both scales.
5. At 400px the Leaderboard table starts near the top, behind **Filters ▾**.
6. The LIVE badge's time moves with each check.
7. More ▾ has no "Sandbox run", and `/demo` is a 404.

### 11i — sit the exam from the model page

Brief: `docs/prompts/phase-11g-read-and-first-page.md`, 11i. One PR.

- **Sit the exam** in the model page's hero, beside the three cards, with
  "34 of 37 topics judged" under it. It opens a panel on the model page and
  scrolls to it. So do an opened Leaderboard row's **Run exam**, each model
  row of the Overview's loop card, and a **Sit the exam ▸** beside the Loop
  tab's "Results for".
- **The panel** (`modelSitPanel`, `state.msit`): the 37 topics under the 8
  areas, an area's box ticking the whole area. Each row says where this
  model stands, from its judge file, its history and the queue:
  - `not sat`;
  - `0.79 / 4 · 22 Sep`, with `· provisional` when the judge's scores do not
    count yet;
  - `no hidden questions yet · 22 Sep` for a topic graded on practice
    questions only;
  - `on older questions` (10b's retired sets);
  - `in the queue` or `running…`, with the box disabled;
  - `no questions yet` for a topic with no bank, disabled.
  Quick picks **Not sat yet (n)**, **All n**, **Weakest 5** (this model's own
  judged topics) and **None** tick exactly their set. The **MMLU control**
  has its own box, off by default. A ticked topic already judged on these
  questions says "same questions — re-grade only, no GPU". The summary
  follows the ticks: "12 topics · about 1,200 answers · about 25 min of
  GPU, then grading · the GPU is shared", with "— consider a quiet time"
  over 10. **Queue this run** is disabled with the reason beside it: the
  judge offline, the weights not on this server, the checkpoint's own code.
  It sends exactly the ticked `exam_*` tasks, and `fr_control_mmlu` only
  when ticked. The rows turn to `in the queue`, then `running…`, as the
  queue polls.
- **One picker** (`examPicker`): the Queue form's judged suite uses it too,
  every topic with questions ticked to begin with, the control off. It
  sends the ticks as a list, not an empty "whole suite". The topic page
  keeps its one-topic panel.
- **Own model code, before queueing** (`hfmeta.remote_code_check`,
  `GET /api/models/code`). The search says "ships its own model code", and
  adds "— this server does not run it" when it will not. Where the server
  runs it, the panel and the form show an unticked box ("Run this
  checkpoint's own model code (`modeling_*.py`, sha `ab12…`) — as the
  unprivileged `benchjob` user"). Queue needs it ticked, and it sends
  `allow_remote_code: true`. Where the server does not, the button is
  disabled with the reason, naming `ALLOW_REMOTE_CODE=1`, `EVAL_USER` and
  `SERVICE.md § custom model code`. The API refuses with 422 before
  queueing, in the same words, and never fails at start. A sha off
  `REMOTE_CODE_SHAS` is named, with the sha to add. **Resubmit…** on a row
  that failed for this reason opens the same box in a row under it.
- **The Suite cell** is one line: `judged · 37 topics + MMLU control ▸`, and
  ▸ opens the list grouped by area; one topic stays spelled out,
  `judged · Arts`. It's the same on the model page's runs.

**Deploy steps, after 11i merges.** Code only, no data step.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
sudo docker compose exec -T bench python3 -c "import urllib.request as u; print(u.urlopen('http://localhost:8899/api/models/code?id=local/qwen35-delta-moe-7d560104-step945-v2').read().decode())"
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. The last command prints `"own_code": true`, the checkpoint's `.py` files with
   their shas, and a `why` naming `ALLOW_REMOTE_CODE=1` and `EVAL_USER`
   unless both are set (then `why` is empty).
4. A model page's **Sit the exam** opens the panel with 37 topics under 8
   areas and this model's scores; **Not sat yet** ticks the ones it has not
   sat; **Queue this run** queues them, and their rows say `in the queue`.
5. #56's Suite cell reads `judged · 37 topics + MMLU control ▸` on one line.

### 11j — the Review tab, rebuilt

Brief: `docs/prompts/phase-11g-read-and-first-page.md`, 11j. One PR.

- **Four views, each with its count**, in one row at the top: **To review**,
  **Ready to generate**, **Datasets**, **History** (rejected and failed) —
  and **+ New proposal** on the right. The view lives in the hash
  (`#tab=review&view=datasets`), so a link opens it; the tab opens on To
  review when something is waiting, and on Datasets when nothing is.
- **Each view is a compact list**, one line per row:
  - proposals: topic · model · status · asked by · when · Demo only;
  - datasets: # · topic · model · documents ("20 of 20", or "18 of 20 · 2
    missing") · made by · when · Demo only · **Read** · **Use in training
    ⧉**, which copies `--gap-dataset 8`; the sentence about training runs
    is its tooltip now. The topic page's datasets use the same row.
  - The explanation is one line plus **How this works ▸**, which also holds
    the AI's name, today's usage and the storage line.
  - "Pick a topic" is gone: the Loop board already is that table.
- **A proposal opens as a card in the 11g reader's sheet** (`readProposal`,
  `read=proposal:<id>`), so a pasted link opens it and Back closes it:
  - the header is "Economics · good-750m", with `#8`, one status chip and
    **one** Demo only badge whose tooltip gives the reasons in plain words;
  - **What's missing**, in large type, editable while it is To review;
  - **Why**, in one line: "36 of 44 practice answers scored below 3 of 4 ·
    Arts score 1.46 / 4 (hidden questions)" — the count that read "—"
    because the page looked for `diagnose_wrong` and the service sends
    `diagnose_weak`;
  - **The answers it read (33) ▸**: every one of them, not the first eight,
    each with its practice question, the model's answer, the score and the
    judge's comment, under one line saying the AI saw only the comments
    with the question wording taken out;
  - **Documents will cover**: the plan as chips, the first 8 then "+12
    more", with the spread box;
  - the actions for its status: Approve / Reject… with a reason, or count,
    format and Generate;
  - the datasets made from it, each opening the reader;
  - **Details ▸**: who asked and approved, the AI and the prompt
    fingerprint, the copy check, today's usage and the full reasons.
- **+ New proposal** opens a dialog: a model, then the topics grouped by
  area, weakest first, each with its score — and each blocked one disabled
  with its reason ("not sat yet", "proposal #5 is open", "under 30 hidden
  questions — its score is noise"). The topic page's **Propose…** opens the
  same dialog with both filled in.
- **New endpoint** `GET /api/proposals/{id}/answers`: the practice answers
  the AI read, joined to their questions and the model's answers. Diagnose
  half only, by the qids the proposal recorded when it was made.

**Deploy steps, after 11j merges.** Code only, no data step.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. The Review tab opens on **To review** or **Datasets**, with four counted
   views and **+ New proposal**.
4. Proposal #5 (Arts) opens as a card saying "36 of 44 practice answers
   scored below 3 of 4", with all 33 answers behind **The answers it read**,
   each with its practice question.
5. The Datasets view lists #1–#8, each with **Read** and **Use in training
   ⧉**; the tab is under 1,600 px tall at 1,512 px.

### 11k — the 11f–11j live check

Brief: `docs/prompts/phase-11k-live-check-fixes.md`. One PR. The nine things
masein and I found on the board on the evening of 22 Sep.

- **Propose → opens the New proposal dialog**, filled in, from the model
  page, the Loop board and the topic page; it opens over the page and never
  navigates. Until now it linked to the topic page, which has had no Propose
  of its own since 11j moved proposing into the dialog. Where a proposal for
  that model and topic is already open, the row reads **Review it →** and
  opens that proposal's card. The tooltip flips side and stays 8 px inside
  the window.
- **A run says the stage it is at**: `queued` → `running 2/4` →
  `grading 240/570` → `done`. `done` waits for the grades. While the judge
  works the action cell holds a quiet **Grading… 240/570** chip instead of
  nothing, and `⋯` keeps Log throughout.
- **The dataset reader's "The missing skill"** shows the approved spec again
  (11j's textarea took its class name), falls back to the proposal's own
  words, and says "no spec recorded" when there are none. Closed, it is one
  line tall.
- **Every sticky table clips and scrolls inside its own card** — not only
  the Leaderboard's. The Queue's painted across its card at 1,512 px.
- **Nothing slides the page sideways**: `#view`, `.wrap` and `.card` clip on
  the x axis (`overflow-x: clip`, which makes no scroller, so page-sticky
  headers and body popovers are untouched).
- **Sit the exam's area headings** read "0 ticked · 2 of 5 judged".
- **The answers a proposal read** come ten at a time, with the pager, and
  each answer is clamped to three lines behind **Show the whole answer**.
  The question and the judge's comment stay whole. It was one 14,000 px
  scroll.
- **The focus ring is the keyboard's**: closing a popover with the mouse
  gives the button its focus back without a ring; a Tab still shows one.
- **A list row is a row**: no borders or underlines inside a menu, and the
  chosen one is tinted and ticked.

**Deploy steps, after 11k merges.** Code only, no data step.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. A model page's **Propose →** opens the dialog over that page; a topic
   with an open proposal reads **Review it →**.
4. A judged run shows `grading k/n` and a **Grading…** chip while the judge
   works, and `done` with **Open results** after.
5. Dataset #6's reader shows the missing skill; no table paints over its
   card at 1,280–1,920 px; no page slides sideways at 400 px.


### 11l — thinking models are scored on their answers

Brief: `docs/prompts/phase-11l-thinking-models-and-reads.md`, §1 only; §2–§9
follow as 11m. Run #60 (Qwen3-1.7B) opened `<think>` on every one of 3,730
answers and closed it on none: the 256-token budget ran out inside the
reasoning, and the judge graded cut-off monologues as answers.

- **The reasoning comes out before anything is graded or shown**
  (`judge.split_reasoning`, `judge.answer_parts`). The raw generation stays
  in the harness's own log. `<think>…</think>` is the default; more tags go
  in `REASONING_WRAPPERS`. A closing tag with no opening one closes a block
  the chat template opened in the prompt (the DeepSeek-R1 distillations).
- **An answer that never left its reasoning block is no answer.** It is not
  sent to the judge, is counted (`no_answer`), and is in no mean, no
  distribution and no criterion. A topic where most answers never finished
  has **no score** — a dash on the model page, with "the model never
  finished answering: these questions were not scored", and Propose refuses
  it. Only a reasoning model's answer can be no answer: an empty generation
  from any other model is graded exactly as before.
- **A reasoning model gets room to answer.** Preflight marks a chat template
  that opens a reasoning block (or offers `enable_thinking`); its judged
  runs pass `--gen_kwargs max_gen_toks=2048` (`REASONING_MAX_GEN_TOKS`).
  Every other model keeps 256. The budget each topic's answers were
  generated with is read from the harness's results file, recorded with the
  grades, and shown under **How this was graded ▸**.
- **Run #60 is voided, not deleted.** `judge.py --revalidate` re-reads every
  judged model's answers and takes out of the grades each one that never
  left its reasoning block, keeping what the judge said about it under
  `voided`. It asks no judge, and it touches only topics with such answers.
- **A new check:** "Answers that never finished", amber, naming the model
  and the count; **Show me** opens its page.

**Deploy steps, after 11l merges.** Code, then one data step.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
sudo docker compose exec -T bench python3 scripts/judge.py /home/masein/benchmarks/results/full --revalidate
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. The revalidate step prints
   `Qwen__Qwen3-1.7B: 3730 answers never finished — 37 of 37 topics now have no score`
   (the counts are the run's own), then
   `revalidated: 1 model(s) changed, the rest untouched`. Run again, it
   prints `revalidated: 0 model(s) changed`.
4. Qwen3-1.7B's model page shows a dash on all 37 topics and "37 topics
   were not scored"; the checks bar reads "Answers that never finished:
   Qwen3-1.7B (3,730 of 3,730)". Sit it again to score it.

### 11m — the Exam tab reads the questions, and five smaller fixes

Brief: `docs/prompts/phase-11l-thinking-models-and-reads.md`, §2–§9 (§1 was
11l; §6, the submit form, was already #57).

- **§2 The Exam tab opens a topic's questions.** In **Rubrics and
  criteria**, the bank count, "read the practice half" and a **questions**
  entry beside `rubric` and `criteria` all open the bank reader
  (`read=bank:<topic>`). The cell reads `6 — 2 hidden / 4 practice · read
  the practice half`: "hidden", not "report", by the plain-words rule. Only
  the practice half opens, as everywhere.
- **§3 The model page's by-criterion block is gone**, with its topic picker.
  The criteria strip inside each answer card stays. Three things lived only
  inside that block and went with it: the per-topic flag lines, the
  breakdown tables (by acuity, difficulty, jurisdiction) and the "every
  item is …" constant-field line. Their numbers are still in judge.json,
  and Propose reads judge.json, not the page.
- **§4 One rule for the three header pills** (`.bar .barpill`). The checks
  pill had `3px 0` from `details.checks > summary`; all three are now
  `4px 12px`.
- **§5 Conditional criteria — reported on, not changed.** See below.
- **§7 Question-and-answer datasets have a register of their own.**
  `fmt: free` gets `GEN_SYSTEM_QA` and a question-and-answer register in
  place of the documents' "Prose, never question-and-answer pairs." A
  generator that follows its instructions now returns items the reader can
  parse; with the old prompt the same generator reproduces dataset #9 word
  for word ("the generator returned no parseable items").
- **§8 A failed dataset says 0 of n and why.** DOCUMENTS MADE BY reads
  `0 of 26`; the status says **Failed**; a line under the row says what
  happened ("0 of 26 kept — asked for question-and-answer items, but the
  generator was given the prose register."), and **Details** lists the
  reason for every missing item from `provenance.items.missing`, ten a page.
- **§9 The suite options say what each gets you**, one line each under its
  name, and the form says `full` and `judged` are separate runs — a model
  needs both for an average and a judged score — and that resubmitting is
  free.

**§5, what judge.py does with a conditional criterion.**
- It asks one question per answer, carrying all twenty criteria. There is
  no separate "does this apply?" step. Each conditional criterion's line
  ends "Return null for it when it does not apply to this question."
- The same prompt's header says, twice, that every criterion must come back
  as a number from 0.0 to 1.0. The two instructions contradict each other.
- None of the 504 conditional criteria in the 37 criteria files has an
  `applies_when`; the condition is only inside each definition ("When
  algebra is required, …").
- Everything after the reply handles null correctly: the parser accepts it,
  and the fold and every criterion mean leave it out. The stub judge
  returns null, and the tests pass through it.
- So "100 of 100" on all twelve of Mathematics & Statistics' conditionals
  means the judge never returned null there. Those twelve carry 0.60 of
  that topic's fold weight. Whether they pulled the score down or up
  depends on the number the judge gave a criterion that did not apply.
- Nothing is changed: a new prompt changes its sha256, and every score
  before it stops being comparable with every score after. That is a
  decision for masein, with a re-sit. `judge.py --conditionals` (read only)
  measures it on the server.

**Deploy steps, after 11m merges.** Code only; the last command reads and
writes nothing.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
sudo docker compose exec -T bench python3 scripts/judge.py /home/masein/benchmarks/results/full --conditionals
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK`.
2. The log grep prints `no errors`.
3. One line per judged topic, like
   `<model>  exam_mathematics_statistics: 12 conditional criteria, scored on <P> of <N> answer-criterion pairs (<%>), <k> never skipped; <m> on average when scored, <z>% at 0.0; topic mean <a> / 4 as graded, <b> / 4 with every conditional left out`,
   then a total. **100% and "12 never
   skipped" is the §5 finding**: the judge scores conditionals it should
   skip. Some pairs below 100% means it does skip. The two topic means
   bound how much that moved the score.
4. The Exam tab's **Rubrics and criteria** rows open the practice questions.
   The model page has no by-criterion block. The three header pills match.
   Dataset #9 reads `0 of 26`, **Failed**, and the line about the prose
   register.

### 12a — the Everyday tasks pilot

Brief: `docs/prompts/phase-12a-everyday-pilot.md`, the first of 12a–12f.
Five questions typed the way people type into an assistant on a phone, so
masein can read what four models actually say before a bigger bank is
written. A look, not a benchmark: never ranked, never averaged, on no
leaderboard, read by nothing that proposes or generates. None of the plan's
restructure is here — that is 12b.

- **The questions**: `eval_tasks/everyday/pilot.jsonl`, the brief's five,
  typos included, all readable (no practice/hidden split yet; an item can
  carry `split` later without a rewrite). The harness task is written from
  `eval_tasks/everyday/_everyday_template_yaml` at the start of each run
  (`everyday.build_task`): the question as typed, no `Answer:` suffix,
  greedy, 512 tokens.
- **The suite**: `everyday`, beside quick/full/control/judged. Always
  through the chat template, whatever kind the board lists; a model without
  one is refused at preflight ("This model has no chat template, so it
  can't be asked questions the way a person would."). A reasoning model gets
  #59's 2,048 tokens. The model's listed kind is left as it was. A second
  pilot run answers again; the last answers move to `results/earlier/`.
- **Marking**: `scripts/everyday.py`, straight after the answers, in the same
  run. It marks the text after the thinking (`judge.answer_parts`, #59's
  split); an answer that never left its thinking fails with "never finished
  answering". Four checks are scripts — contains, json, fixed, lines — each
  with one plain reason. The Arabic question is the only one the judge
  marks: the row says `grading 0/1` until it lands. `everyday.json` sits
  beside the model's results; `python3 scripts/everyday.py results/full`
  re-marks every model from the logs, keeping the judge's verdicts.
- **What masein sees**: an **Everyday tasks** block on the model page, above
  the exam (the count, five rows, a row opens the question and the answer,
  the thinking folded), or one line, "Not tested on everyday tasks · Test".
  And `#everyday`, from **Compare models →** and **More ▾ ▸ Everyday
  pilot**: the five questions against the models, each mark opening the
  answer in the side panel (↑↓ questions, ←→ models), and **Run the pilot**,
  which queues one run per ticked model (Qwen3-1.7B, Qwen3-0.6B,
  SmolLM2-360M-Instruct and gemma-3-270m-it are ticked).

**Deploy steps, after 12a merges.** Code only.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

**Expected output:**

1. `up -d --build` ends with the container healthy; the image build prints
   `image files OK` (the build now also checks `eval_tasks/everyday/`).
2. The log grep prints `no errors`.
3. **More ▾ ▸ Everyday pilot** opens `#everyday`, which reads "No model has
   taken the pilot yet" with **Run the pilot**. Run it: four rows queue,
   each finishes in minutes and says `Everyday pilot: n of 5`, with
   `grading 0/1` while the judge marks the Arabic answer.
4. `#everyday` then shows four models × five questions; every mark opens the
   answer. Qwen3-1.7B's answers are the text after its thinking, and its
   thinking is folded. Nothing of it is on the Leaderboard.

### 12b.1 — the five places: the header, Improve, Benchmarks and Models

Brief: `docs/prompts/phase-12b-five-places.md`. It shipped in two PRs, as
the brief allows. **12b.1** is §1–§5 and §8's redirects. **12b.2** is §6
(Home rebuilt), §7 (the model page's tabs, with Run provenance moving to
History) and the doc links. This PR moves things: tables, readers and
dialogs keep their code and change container.

- **The views are still the old tab ids** (`leaderboard`, `loop`, `review`,
  `queue`, `exam`…), so every in-page `navigate({ tab })` still works.
  `PLACES` groups them:
  - Home = `overview`
  - Models = `leaderboard`
  - Improve = `loop` · `review` · `training`
  - Benchmarks = `tasks` · `exam` · `everyday`
  - All runs, Data & sources and Help are pages outside the nav.

  `viewHash`/`viewOfHash` translate between views and addresses, and any
  legacy address is rewritten in place with `replaceState`.
- **The header**: four places, then the run counter, **Test a model**, the
  status dot and the name menu.
  - Below 720px the places are one **Menu ▾**.
  - The name menu holds Theme, Data & sources and Help.
  - There is no More ▾ and no Theme ▾.
  - Test a model is today's Submit form in a dialog. `#tab=submit` opens it.
- **Models is one table.**
  - Its switch is Standard · Knowledge exam · Everyday tasks, kept in
    `lbS().view`. The exam view is the old `chip: 'judged'`, and
    Language modelling is `chip: 'lm'` (the old Perplexity page).
  - All six filters are in Filters ▾. The Models tab's facts are columns
    under Columns ▸ Model facts, off by default.
  - There is no row expansion: a click on a row opens the model page.
  - A model with nothing in the view sits under **Not tested on this (n) ▸**.
    A model graded by a judge that does not count is still a row.

**Deploy steps, after 12b.1 merges.** Code, plus pytest and httpx in the
image, which step 3 needs (§ 5b).

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
git archive HEAD | sudo docker compose exec -T bench sh -c 'rm -rf /tmp/check && mkdir /tmp/check && cd /tmp/check && tar -x && exec env -i PATH="$PATH" HOME=/tmp/check LANG=C.UTF-8 python -m pytest -q -p no:cacheprovider -m "not gpu and not network and not dashboard"' 2>&1 | tail -15
```

**Expected output:**

1. `up -d --build` ends with the container healthy, and the image build
   prints `image files OK`.
2. The log grep prints `no errors`.
3. The header reads **Home · Models · Improve · Benchmarks**, then
   **Runs**, **Test a model**, a green dot and your name.
   - A bookmark to `#tab=leaderboard&chip=math` lands on Models ▸ Math.
   - `#tab=review&view=datasets` lands on Improve ▸ Review ▸ datasets.
   - `#everyday` lands on Benchmarks ▸ Everyday tasks.
4. **Test a model** opens the form as a dialog. Queueing a run shows
   "Run #n queued — follow it →", and the run counter changes to
   **● 1 running** once the run starts.
5. Step 3 ends `N passed, M deselected in …s`, with no `failed` and no
   `error`. It is the first run of the unit suite inside the image, so a
   failure here is news about the image (§ 5b): send the output.

---

## 11. Known gaps, risks, loose ends

- `transformers` unpinned (`>=4.55`); the guard catches the failure mode we saw,
  not every possible one. Pin once Roohi gives a minimum.
- Duplicate leaderboard rows: `local/qwen35-delta-moe-7d560104-step945` and
  `-v2` are byte-identical on 8 tasks and both counted at the top. Resolve.
- `mmlu_perm` control never run.
- Existing `judge.json` files on the server (if any) were written by the
  *local* judge design from phase 5; phase 7 marks them with their judge identity
  and shows them as a different series. Do not merge the two.
- The API-judge path costs reproducibility; the canary is the mitigation, not a
  cure. If the vendor retires the dated id, re-grade a held sample and report
  the drift before trusting new numbers.
- Dashboard payload grows ~30–40 KB per model with full diagnosis; fine over the
  tailnet. Knobs: `_DIAG_EXAMPLES`, `_DIAG_Q` in `report_lm_eval.py`.
- `.git/*.lock.stale*` files in the Mac checkout are harmless debris from a tool
  that could not delete lock files; remove them.
- The device-bridge tool used for some of this work cannot delete files and
  sometimes wrote stale content under reused paths — every write in this project
  was verified by md5 afterwards. If you use similar tooling, do the same.

---

## 12. Documents and artefacts

| File | What it is |
|---|---|
| `HANDOFF.md` | this |
| `DIAGNOSE.md` | operator guide: routine, reading order, finding→action table, the log to keep |
| `DEMO.md` | the loop end to end in one command against the local model, and — plainly — what a green run does not prove |
| `SERVICE.md`, `API.md`, `FRIENDS.md`, `BENCHMARK-RUN.md` | running the service; the API; the submitter guide; the manual CLI path |
| `docs/design-diagnose-and-generate.md` | the original design and its argument for the split |
| `docs/prompts/*.md` | the implementation briefs, in order; all merged as of phase 8b |
| `docs/medicine-criteria-v1.md` | Dr. Hossein's 15 criteria and the critical-failure rule, as delivered — the source `rubrics/medicine_health*.` derive from |
| `llm-api-budget.xlsx` (with Omar) | per-cycle cost model; prices verified 2026-09-18; Steps and Glossary sheets define every term |
| `eval_pipeline_fasttrack.html` (with Omar) | the team deck: loop, the one rule, rollout, curation math, costs, provider rule |

---

## 13. Gotchas learned the hard way

- `~/benchmarks` is **not** a git repo; `~/benchmarks/aienh` is. `docker compose`
  from `~/benchmarks` finds no compose file.
- `docker compose exec` with a heredoc needs `-T`.
- **The image carries only what the Dockerfile copies, and the demo does not
  prove it.** `eval_tasks/fr/` was excluded by `.dockerignore`, so the
  container had no harness task template: both demos passed (they run from
  the bind-mounted checkout) and the first person to press "Rebuild the
  harness tasks" got a 500. `service/startup.py` now lists every repo file a
  request can reach, the app refuses to start without one, the image build
  runs the same check, and `tests/test_image_contents.py` walks the COPY list
  through `.dockerignore` without building anything.
- `scripts/diagnose.py` does not exist inside the container at `/app/scripts/`
  — run it on the host with `sudo`, from `~/benchmarks`.
- The live dashboard and the static report are the **same page and the same
  `build_payload`**. Anything that reads a file beside `results_*.json` must not
  cache misses by path (the service lives for weeks) and must be in
  `app.py::_WATCH` or the payload cache never invalidates. Both bugs happened.
- Examples in `diagnose.json` are sampled round-robin across groups on purpose;
  file order made every example come from whichever subject sorts first.
- Anthropic's 4.7-and-later models use a tokenizer that produces ~30% more
  tokens for the same text; `$/MTok` alone understates their cost. Whether
  Sonnet 5 / Opus 5 are in that family is inferred from version ordering, not
  stated.
