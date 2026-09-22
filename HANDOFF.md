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
