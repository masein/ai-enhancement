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
tests/                   185 tests; fixture generator in tests/fixtures/make_fixture.py; CI in .github/workflows/ci.yml
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

**Dr. Hossein's delivery is in the repo and in the loop (phases 8b and 8d):**

| What | Where |
|---|---|
| 100 consumer health questions with metadata, as delivered | `eval_tasks/fr/medicine_v2.json` |
| 100 law questions with metadata, his own difficulty levels and `jurisdiction_required` | `eval_tasks/fr/law_v2.json` |
| 100 questions each for computer science, economics and physics & engineering | `eval_tasks/fr/computer_science_v1.json`, `economics_v1.json`, `physics_engineering_v1.json` |
| their criteria files and **his own prose rubrics** — not drafts, he wrote the 0–4 anchors | `eval_tasks/fr/rubrics/{computer_science,economics,physics_engineering}.{md,criteria.json}` |
| his criteria files, **verbatim** — the platform's schema is his | `eval_tasks/fr/rubrics/medicine_health.criteria.json`, `law.criteria.json` |
| his scoring notes for each, as delivered | `docs/medicine-criteria-v2.md`, `docs/law-criteria-v2.md` |
| the 0–4 rubrics derived from them — **DRAFT**, anchors not yet reviewed | `eval_tasks/fr/rubrics/medicine_health.md`, `law.md` |

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

**The question floor is cleared.** Both topics are at 100 questions, about 50
report-half each, over the 30 a topic needs before anything may be proposed
from it. The demo says so rather than asking for more.

**What is open with him, and the first is a blocker for calling any score on
these topics a result:**

1. **Rubric sign-off.** Both prose rubrics say DRAFT: their 0–4 anchors were
   derived from his criteria and he has not reviewed them. Until he removes
   the word, every judged score for the topic is stamped draft on the page.
   Removing it changes the rubric's hash, which is correct — scores from
   before and after are then not comparable. The criteria files themselves
   are his own and need no sign-off.
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
