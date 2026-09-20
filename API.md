# Benchmark service API — integrate it into your training loop

For anyone who wants their checkpoints benchmarked automatically: everything the
dashboard does goes through this JSON API, so your training code can do it too.
There are two ways to get a model in, and **you do not need a Hugging Face
account for the first one**:

1. **Upload it to the service's own artifact storage** (`POST /api/artifacts/…`
   or `bench.upload_artifact(...)`) and benchmark it as `local/<name>` — the
   default path for your own checkpoints.
2. Submit a **Hugging Face id** (`org/model`) for anything public on the Hub.

Either way: evaluation runs on the shared GPU, one job at a time, and results
land on the team leaderboard. The pattern is **upload → submit → keep training →
collect scores later**.

Base URL (on the tailnet): `http://100.74.89.105:8899`

Auth: none by default. If the operator sets `SUBMIT_TOKEN`, send it as an
`X-Token` header on POSTs (the dashboard picks it up from `?token=…` in the URL).

The zero-dependency Python client in [`clients/bench_client.py`](clients/bench_client.py)
wraps all of this in one stdlib-only file — vendor it into your repo. The
service also serves it directly (`curl -o bench_client.py
http://100.74.89.105:8899/client`), so you never need repo access to use it.

---

## The 30-second version

Your model is a directory on your disk (anything `save_pretrained()` wrote):

```python
from bench_client import Bench
bench = Bench("http://100.74.89.105:8899")

mid = bench.upload_artifact("run7-step4000", "ckpt_dir/")    # -> "local/run7-step4000"
sid = bench.submit(mid, suite="quick", submitter="you")      # returns immediately
bench.wait(sid, echo=True)                                   # optional: block until done
print(bench.scores(mid))
# {'hellaswag': {'value': 0.412, 'stderr': 0.005, 'metric': 'acc_norm', 'shots': 5, ...}, ...}
```

Or from a shell, one line — upload, benchmark, wait, print the scores:

```bash
python bench_client.py --base http://100.74.89.105:8899 \
    upload run7-step4000 ckpt_dir/ --submit --suite quick --submitter you --wait
```

(For a public Hub model, skip the upload: `bench.submit("myorg/my-model", …)` /
`… submit myorg/my-model --suite quick --wait`.)

---

## Artifact storage — benchmark models without Hugging Face

The service stores your checkpoints itself. Upload a `save_pretrained()`
directory under a name; from then on `local/<name>` works **everywhere a model
id does** — `bench.submit()`, `log_checkpoint()`, `bench.scores()`, and the
dashboard's Submit box.

**SDK** (what most training loops use):

```python
model.save_pretrained("ckpt"); tokenizer.save_pretrained("ckpt")
mid = bench.upload_artifact("run7-step4000", "ckpt")   # -> "local/run7-step4000"
bench.submit(mid, suite="quick", submitter="you")      # or run.log_checkpoint(4000, mid)
```

**CLI**:

```bash
python bench_client.py --base … upload run7-step4000 ckpt/ --submit --suite quick
python bench_client.py --base … artifacts        # names, sizes, quota use
python bench_client.py --base … delete run7-step4000
```

**Raw HTTP** (no client at all) — zip the directory's *contents* and POST it:

```bash
cd ckpt && zip -r ../ckpt.zip . && cd ..
curl -X POST --data-binary @ckpt.zip -H 'Content-Type: application/zip' \
     http://100.74.89.105:8899/api/artifacts/run7-step4000
# -> {"model_id": "local/run7-step4000", ...}
```

### The endpoints

- `POST /api/artifacts/{name}` — body is the zip, streamed to disk. A single
  top-level folder inside the zip is fine (it's stripped); `config.json` must
  end up at the artifact root.
- `GET /api/artifacts` — `{"artifacts": [{"name", "model_id", "bytes",
  "created"}, …], "total_bytes": …, "quota_bytes": …}`.
- `DELETE /api/artifacts/{name}` — frees the disk. Scores already on the
  leaderboard stay; refused (409) while that artifact is queued or running.

### The rules

- **Names are immutable** — one name per checkpoint (`run7-step4000`, not
  `run7`). Re-uploading a taken name is refused with 409; the client pre-checks
  and tells you before wasting the upload.
- **safetensors only** — pickle `.bin` weights execute code on load and are
  refused. Anything modern `save_pretrained()` writes passes by default.
- **Caps**: per-upload `ARTIFACT_MAX_GB` (default 8), shared total quota
  `ARTIFACT_QUOTA_GB` (default 150) — `GET /api/artifacts` shows usage, delete
  what you no longer need.
- Preflight (params cap, architecture capture, `trust_remote_code` refusal)
  applies to uploads exactly as it does to Hub models.

---

## Semantics you should design around

**One job at a time.** The GPU is shared; submissions queue. `quick` on a small
model is minutes; `full` includes GSM8K (generative) and MMLU (14k items) and can
be an hour+ on big-vocab models. Don't block your training loop on `wait()` —
submit and collect later.

**Submitting is idempotent-ish.** A model already queued or running joins the
existing run (you get the same id back). A model already fully benchmarked
finishes in seconds — per-(model, task) results are cached on disk, so re-submits
cost nothing.

**`quick` upgrades to `full`.** Both write into the same results tree; a later
`full` submission runs only the tasks `quick` didn't. Iterate with `quick`,
finish with `full`.

**Preflight rejects fast, before any GPU.** Nonexistent repo, gated repo the
server's account hasn't accepted, models over the parameter cap (default 4B), and
models requiring `trust_remote_code` (the service never executes repo code) all
fail in seconds with a human-readable `error`.

**Kind matters, and ambiguity is refused where it bites.** `kind:"auto"` applies
the chat template when the repo ships one and the model's name corroborates it
(`instruct`, `-it`, `chat`, `sft`, `dpo`…). For an **uploaded artifact**
(`local/<name>`) that ships a template with nothing in its name to corroborate
it, preflight **fails** and asks you to send `kind:"base"` or `kind:"instruct"`
— because a checkpoint saved from an instruct model's tokenizer inherits that
template even when the weights are a base model, and applying it moves
multiple-choice scores by tens of points. For a **Hub repo** in the same
position the template is applied (publishing one usually does mean a chat
model) and the run carries `archinfo.kind_unconfirmed: true`, which the report
turns into a warning. Each run records the template's hash, its source, and the
reason for the decision; they appear in provenance and in CSV exports, and two
runs with different template ids are not comparable.

**Custom architectures.** A checkpoint whose `config.json` has an `auto_map`
needs its own `modeling_*.py` executed to load. Upload it as an artifact (the
`.py` files ride along in the zip) and submit with `allow_remote_code: true`.
The server must be configured for it (`ALLOW_REMOTE_CODE=1`, `EVAL_USER`) — see
SERVICE.md § custom model code for what that buys and what it does not. There
is no separate token for this: the tailnet is the auth boundary. Those runs execute as an
unprivileged user with the Hub offline and the HF token withheld, and every
`.py` is hashed into provenance (`archinfo.code_sha`) and the CSV exports. Hub
repos with `auto_map` are refused regardless. Custom `model_type` values are
accepted on this path; preflight reads shape from `text_config` when the top
level is a wrapper.

**MoE models report both parameter counts.** `params` is the total (it drives
VRAM and the size cap, since every expert is loaded) and
`archinfo.active_params` is per-token (it drives a fair comparison against a
dense model), alongside `experts` and `experts_per_tok`. Taken from
`num_active_params` in the config when present, otherwise estimated from the
expert geometry and labelled `active_src: "estimated"`. The leaderboard shows
total with active beside it.

**Parameter counts for uploads are exact.** They come from the safetensors
headers, not from file size, so an fp32 checkpoint is not counted twice (which
would also inflate it against `MAX_PARAMS_B`). `archinfo.stored_dtype` reports
the precision the weights are saved in — worth checking before a full
fine-tune, since transformers loads a checkpoint in its stored dtype and an
fp16 one trained without a loss scaler is a classic route to a NaN loss.

**Official vs preliminary.** An overall average exists only for models that
completed every task in the required list (`mmlu, hellaswag, arc_challenge,
arc_easy, winogrande, piqa, truthfulqa_mc2` by default; `REQUIRED_TASKS`
overrides). Anything short of that is preliminary: `avg` and `avgRaw` are
`null`, `official` is `false`, and `missing` lists what it still needs. Its
per-task cells are unaffected. `avg` is scaled so chance = 0 (`avgRaw` is the
unscaled mean) — raw accuracy is not comparable across tasks whose guess rates
differ. gsm8k is reported but deliberately excluded from the average: it sits at
~0% below ~1B params and only adds noise to a mean.

**Comparability.** Every run uses the same few-shot counts, seed, dtype and
harness version (lm_eval 0.4.12, pinned). Scores here are comparable to *each
other*, not to public leaderboards (different n-shot conventions).

---

## Endpoints

### POST /api/submissions — queue a model

```json
{"hf_id": "myorg/my-model",     // required — org/name on the HF Hub, OR local/<name> for an uploaded artifact
 "suite": "quick",              // "quick" (hellaswag+arc_easy+perplexity) | "full" (all tasks) — default full
                                // | "control": mmlu_perm only — MMLU with the options rotated (DIAGNOSE.md
                                //   § The one experiment); a control, never in the average
                                // | "judged": the free-response tasks + the local judge (needs JUDGE_MODEL and
                                //   scripts/fr_build.py on the server); never in the MC average
 "kind": "auto",                // "auto" | "base" | "instruct" — default auto
 "submitter": "masein",           // shows on the queue and in provenance
 "note": "run7 step 4000",      // free text, shows as a tooltip
 "allow_remote_code": false}    // uploads with a custom architecture (see below)
```

Returns `{"id": 12, "status": "queued"}` — or, if the model is already active,
`{"id": 9, "status": "running", "note": "already in the queue — joining the existing run"}`.

### GET /api/submissions?limit=100 — the queue, newest first

Each row:

```json
{"id": 12, "hf_id": "myorg/my-model", "kind": "instruct", "suite": "quick",
 "submitter": "masein", "note": "run7 step 4000",
 "status": "running",              // queued | preflight | waiting_lock | waiting_gpu | running | done | failed | canceled
 "progress": "2/4 · arc_easy (5-shot)",
 "error": "",                      // human-readable reason when failed
 "params": 596049920, "vocab": 151936, "batch": 1, "need_gb": 4.2,
 "created_at": 1755500000.1, "started_at": 1755500060.5, "finished_at": null,
 "gpu_seconds": 312.4}
```

Status lifecycle: `queued → preflight → (waiting_lock | waiting_gpu)* → running → done | failed`,
plus `canceled` (only reachable from `queued`). `waiting_lock` means a manual CLI
run holds the GPU; `waiting_gpu` means not enough free VRAM yet — both clear on
their own.

### POST /api/submissions/{id}/cancel

Only while `queued` (409 otherwise — a running job finishes its current task).

### GET /api/runs/{id}/log?tail=200

Plain-text tail of that run's raw lm-eval output. Where you look when `failed`.

### GET /api/results — everything the dashboard renders

The payload your tooling wants. The useful parts:

```jsonc
{
  "models": [ {"id": "myorg/my-model", "name": "my-model", "kind": "instruct",
               "params": 596049920,
               "official": true, "nhave": 7, "nreq": 7, "missing": [],
               "tainted": [],                   // tasks its training data was derived from (see Taint)
               "avg": 0.321, "avgRaw": 0.514,   // required-list mean: above-chance, and raw
               "partialAvg": 0.455, "navg": 8,  // over whatever it ran — a diagnostic, never a rank
               "archinfo": {"arch": "Qwen3ForCausalLM", "hidden": 1024,   // captured at preflight from
                            "layers": 28, "heads": 16,                     // the model's config.json —
                            "ctx": 40960, "vocab": 151936},                // null for CLI-run models
               "minutes": 61.2, "date": "2026-08-19 10:02:11", ...} ],
  "accTasks": ["mmlu", "hellaswag", ...],       // higher-is-better, proportions
  "pplTasks": ["ppl_code", "ppl_fineweb_edu"],  // lower-is-better, no stderr
  "tasks":  { "hellaswag": {"metric": "acc_norm", "lower": false, "chance": 0.25, "control": false}, ... },
              // control: true marks mmlu_perm — shown, never averaged
  "cells":  { "hellaswag": { "myorg/my-model": {"v": 0.412, "se": 0.005, "shots": 5, "n": 10042} } },
  "sig":    { "hellaswag": [ ["modelA", "modelB", 0.062, 3.1, true], ... ] },
              // pairwise [a, b, diff, z, significant_at_95%] — check before claiming a win
  "extra":  [ ["my-model", "mmlu_anatomy", "acc", 0.2519, 0.0021, 5, 135], ... ]  // every raw metric incl. subtasks
}
```

Read a score as `cells[task][hf_id].v ± .se`, with the metric name and direction
from `tasks[task]`.

A model with a per-item diagnosis on file (`scripts/diagnose.py`) carries
`models[].diag`: per task, the leaderboard-half and diagnosis-half scores,
the failure buckets, `groups` (MMLU subjects) and `categories` — the same
fields rolled up through `scripts/categories.yaml`, each with the `groups` it
holds, plus `unmapped` for any subject the file does not know.
`meta.categories` is the category order. Every number in `diag` is computed
from the per-item log; `score_report` is the leaderboard half only. Quote `bits_per_byte` for the perplexity tasks — it's the
tokenizer-independent one.

### The exam

`GET /api/exam` — the exam writer's identity, the bank per topic (accepted,
report half, diagnose half, awaiting curation, target), the tasks built, and
the draft command. `GET /api/exam/candidates?topic=` — drafts awaiting a
decision. `POST /api/exam/candidates/{cid}/accept` `{"approver", "prompt",
"reference", "notes"}` (edits allowed; the accepted text is what gets hashed)
and `…/reject` `{"approver", "reason"}` — a name is required and recorded.
`GET /api/exam/bank?topic=` — the bank with **report-half text withheld**:
those rows carry `qid`, `topic`, `half` and `withheld`, never the question.
`POST /api/exam/build` — write the harness tasks from the bank plus the MMLU
control set (no GPU).

`POST /api/exam/import/preview` and `POST /api/exam/import` —
`{"topic", "approver", "source", "items"` **or** `"text"}`: a bank written by
a person, the same parser and the same records as `scripts/exam_build.py
import`. The preview writes nothing and answers with the counts (`imported`,
`skipped` duplicates by qid, `invalid` with the offending indices), the
`report`/`diagnose` split, the per-acuity and per-intent counts, and the
items — **report-half prompts withheld**, qid and metadata only, in the
response body as well as on the page. The commit is idempotent and records
the decision in `curation`. A name is required. 2 MB cap; anything that is
not a JSON array of objects is refused with 422.

`GET /api/exam/rubrics` — for every topic: the rubric and criteria file the
judge would use right now (the topic's own or the `exam.md` fallback), their
version, sha256, DRAFT status and criterion count, plus `store` — where an
upload would land — and the recent changes.
`GET /api/exam/rubrics/{name}?kind=rubric|criteria` — the file itself.
`POST /api/exam/rubrics/preview` and `POST /api/exam/rubrics`
`{"name", "kind", "content", "approver", "note"}` — the preview validates a
criteria file the way `judge.py` does (criterion and flag ids unique and
lower-case, definitions and conditions present, weights positive, and every
flag effect one this judge can apply — `zero_score` or `cap_at_N_of_4`) or a
prose rubric (heading with a version, five anchors), diffs it against the
file in use and says in words
that a new sha makes earlier judged runs on that topic non-comparable. The
commit writes the file and records who, when and both shas in
`rubric_changes`. It never touches git.

### Judged free response

`GET /api/judge/justifications?model=&topic=&limit=` — what the judge wrote
about that model's answers on that topic: **diagnosis half only**, with any
exam question text the judge quoted already removed. The same function a
proposal is built from, so the page shows exactly what the LLM would be
given.

`GET /api/judge` — the judge's identity (`provider/model`, dated), its
family, whether its provider clashes with the exam writer's or generator's
(`provider_clash`, `single_provider_loop`), the canary threshold, the tasks
built, judged runs in flight, and the calibration on file with the judge it
was made for. `models[].judgeState` in `/api/results` says whether a model's
judged numbers may be ranked and, if not, why in words: uncalibrated, κ
below the line, calibrated for a different judge, judged by a judge other
than the one this server runs now, or a moved canary. `models[].judge.canary`
carries the canary's mean absolute deviation from the human marks and from
the previous run. In `/api/results`, a judged model carries
`models[].judge`: per task `{n, mean, max: 4, dist, score_vs_length,
n_report, score_report, n_diagnose, score_diagnose}` — **`score_report` is
the published number**; the diagnose half is what a proposal may read — and,
for `fr_control_mmlu`, `control` per category (`mc_wrong`, `knew`, `didnt`);
`judgedAvg` is the mean over the four authored categories. `judged`
(top level) lists the tasks, the exam tasks in topic order (`exam`), the
task → topic map (`topics`), the judge, and `calibration` (`kappa`, `n`,
`calibrated`, `per_category`). Below κ 0.60 nothing judged is ranked or
averaged; above it the leaderboard shows judged columns with κ in the
header and a separate judged average — never part of `avg`.

### Find the gap: proposals and generated datasets

The pipeline behind the Diagnose section's **Propose a skill spec** button
(DIAGNOSE.md, phase 4). Off unless the operator set `LLM_PROVIDER`; `GET
/api/llm` says so, and shows today's batch-item use against the daily cap.

- `POST /api/proposals` `{"model", "topic", "requested_by"}` — the LLM reads
  the **judge's written assessments** of this model's **diagnosis-half**
  answers on that exam topic and proposes a skill spec. No exam question text
  goes with them: anything the judge quoted from a question is stripped, and a
  test proves it from the recorded request body. Refused (409) with the page's
  own words when the judged suite is preliminary (κ, a moved canary, a
  different judge), when the topic has under 30 report-half questions, or when
  the model wrote nothing usable on it. MMLU's distribution finding for the
  matching category rides along as `caution` — context, never a gate. Returns
  `{"id", "status": "pending", "batch_id", "task"}`; the poller turns it into
  `proposed` when the batch completes.
- `GET /api/proposals[?status=]`, `GET /api/proposals/{id}` — each with
  `spec_text` (the LLM's), `edited_text` (the human's), `evidence`
  (`share_explained`, `patterns`, the topic's report- and diagnosis-half
  scores and counts, the judge id, the MMLU caution, and up to eight
  diagnosis-half `examples`, each a judge assessment with its score),
  `proposer`, `approver`, `judge_run` (the judge id, batch and prompt sha the
  evidence came from), and its `datasets`. `task` is the exam task
  (`exam_<topic>`) and `category` is the topic.
- `POST /api/proposals/{id}/approve` `{"approver", "edited_text"}` and
  `…/reject` `{"approver", "reason"}` — a name is required: it is the record.
- `POST /api/proposals/{id}/generate` `{"requester", "count", "fmt": "doc"|"free"}`
  — only for an approved proposal. The generator receives the approved spec,
  the topic, the count, the format and a style constraint, **and no benchmark
  item and no exam question in any form**. `doc` is the default and writes
  prose **training documents** (`{"title", "text"}`, around 600 words, two per
  request): question-and-answer pairs shaped like the exam are the most direct
  route to teaching the test, and prose does not have that shape. `free` stays
  for comparison; `mc` is retired and refused with that reason. Returns
  `{"dataset_id", "batch_id"}`. Refused when the spend guard (429), the batch
  cap (422) or the dataset quota (507) says so.
- `GET /api/datasets`, `GET /api/datasets/{id}`, `GET /api/datasets/{id}/items.jsonl`,
  `DELETE /api/datasets/{id}` (refused while a training run references it).
  Every item passed the 13-gram contamination gate against both halves of
  every benchmark on disk **and every exam question in the bank**; the gate
  report names how many were dropped for each (`dropped_benchmark`,
  `dropped_exam`), and a dataset losing more than 2% that way is `rejected`. `provenance` is the full record: source model, task, category,
  split, salt, proposal id and both spec texts, approver, generator provider +
  model + batch id, prompt hash, timestamps, items generated / dropped / kept,
  the gate result, the content sha256.

  Client: `bench.datasets()`, `bench.dataset(id)`, `bench.pull_dataset(id, dest)`;
  CLI `datasets` and `pull <id> <dir>`.

**Taint.** `POST /api/truns` accepts `"datasets": [ids]` and `"parent"` (the model
id the run started from; defaults to `config.base_model`). A dataset derived
from an **exam topic** taints that topic: the score is still shown on the
model page, the topic leaves `judgedAvg` and its leaderboard column, and the
badge says so. A dataset derived from a **multiple-choice** task keeps the
Phase 4 behaviour: the task leaves the official average and the model is
unranked. A training run that
records a generated dataset marks every checkpoint submitted under it (or
carrying its `hf_prefix`) as trained on data derived from that dataset's task:
`models[].tainted: ["mmlu"]` in `/api/results`, a badge on the board, and the
task **excluded from that model's official average exactly the way a missing
required task is** — the per-task score stays visible, the model carries no
rank. `examples/train_and_benchmark.py --gap-dataset <id>` shows the pattern.

**What the training taught.** When the parent is on the board, the tainted
model carries `taintCompare[task]`: both halves before and after
(`{v, n, se}`), `dReport`, `dDiagnose`, their standard errors, a `verdict`
(`skill` | `test` | `none` | `mixed`), the `ratio` of the two deltas, the
derived `text`, and `scale`. `scale` is `"pct"` for a multiple-choice task
(accuracy, with the per-`categories` breakdown) or `"rubric"` for an exam
topic (the judge's 0–4 mean, from the per-half score distributions, and both
sides must come from the same judge or the card says so instead of drawing
it). `test` means the diagnosis half moved and the report half did not — the
training taught the test — and the board carries a warning.

### GET /healthz

`{"ok": true, "queue": 1}` — for your scripts' sanity checks.

### GET /client

`bench_client.py` itself, as plain Python — `curl -o bench_client.py …/client`.
(`GET /guide` is the human quick-start, rendered HTML.)

---

## Run tracking — your training curves, live on the Training tab

The wandb-shaped half of the service. Your training loop streams metrics here and
the **Training** tab shows live curves, run overlay/compare, config diff between
runs, and — the part no external tracker can do — your benchmark scores plotted
on the same step axis as your loss, joined through checkpoints.

```python
from bench_client import Bench
bench = Bench("http://100.74.89.105:8899")

with bench.init("run7", project="llm", submitter="you",
                config={"lr": 3e-4, "batch": 32}) as run:
    for step, batch in enumerate(loader, 1):
        loss = train_step(batch)
        run.log({"loss": loss, "lr": sched.get_last_lr()[0],
                 "gpu_mem_gb": torch.cuda.memory_allocated()/1e9}, step=step)
        if step % 1000 == 0:
            model_id = push_checkpoint(step)          # Hub repo or artifact upload
            run.log_checkpoint(step, model_id)        # marks the step AND queues the benchmark
# leaving the `with` calls run.finish() — status "failed" if an exception escaped
```

Two conventions the dashboard understands: put `batch_size`, `micro_batch_size`
and `grad_accum` in `config=` (they render in the Config card and highlight in
the diff between runs), and log a cumulative **`tokens`** metric
(`step × batch × seq_len`, or your real count) — the run list shows total
trained tokens and it charts like any metric.

Semantics worth knowing: `log()` buffers (64 points or 10 s) and **never raises
into your training loop** — if the service is down it warns once on stderr and
keeps training; any metric name is fine (system stats like `gpu_mem_gb` are just
metrics); values are step-indexed; a run that never calls `finish()` and goes
silent for 5× its usual update gap (at least 30 min) shows as "idle …" in the
list — display-only, cleared by the next `log()`. Raw endpoints, if you'd rather not use
the client: `POST /api/truns` → `{id}`, `POST /api/truns/{id}/log`
`{"metrics":[{"step":n,"name":"loss","value":x},…]}` (≤5000/batch),
`POST /api/truns/{id}/event` `{"step":n,"kind":"checkpoint","detail":"<model id>"}`,
`POST /api/truns/{id}/finish` `{"status":"finished"}`,
`GET /api/truns`, `GET /api/truns/{id}` (downsampled series + events).

The two halves compose at checkpoint time: `upload_artifact()` then
`run.log_checkpoint(step, model_id)` — one call that marks the step on your
curves **and** queues the benchmark. (Storage details in
[Artifact storage](#artifact-storage--benchmark-models-without-hugging-face)
above.)

## A runnable, end-to-end sample

[`examples/train_and_benchmark.py`](examples/train_and_benchmark.py) is this
whole document as working code: it fine-tunes a tiny model for real, streams its loss/lr/throughput curves to the
Training tab, uploads each checkpoint as an artifact (or pushes to the Hub with
--push-to), submits every checkpoint the moment it lands (non-blocking), and
prints a task × step score table at the end.

```bash
# 1) prove the service works — no training, no HF account, ~a minute:
python examples/train_and_benchmark.py --bench http://100.74.89.105:8899 --dry-run

# 2) the full pipeline — checkpoints go to the service's artifact storage,
#    so NO Hugging Face account is needed:
python examples/train_and_benchmark.py --bench http://100.74.89.105:8899 \
    --steps 200 --checkpoint-every 100
# (add --push-to <hf-user>/bench-demo to publish checkpoints to the Hub instead)
```

It's also the template to copy from: the `checkpoint()` function is the
integration in ~15 lines, including the rule that matters (a submit failure
prints a warning and training continues).

## The training-loop pattern

Submit at every checkpoint, don't wait, collect at the end (or from a separate
process). The service dedupes and resumes, so this is cheap and crash-safe:

```python
# during training — after each checkpoint is saved (uploaded artifact or Hub repo)
from bench_client import Bench, BenchError
bench = Bench("http://100.74.89.105:8899")

def on_checkpoint(step: int, repo_id: str):     # repo_id: "local/<name>" or "org/model"
    try:
        bench.submit(repo_id, suite="quick", submitter="masein", note=f"step {step}")
    except BenchError as e:
        print(f"benchmark submit failed (non-fatal): {e}")   # never kill training over this

# after training — collect the curve and log it to your tracker
import wandb
for step, repo_id in checkpoints:                 # your list of (step, hub id)
    for task, s in bench.scores(repo_id).items():
        wandb.log({f"bench/{task}": s["value"]}, step=step)
```

That gives you benchmark-vs-training-step curves next to your loss curves —
the "is it actually getting better on capabilities, not just on loss" plot. If
you prefer fire-and-forget without any client code, it's one curl:

```bash
curl -s -X POST http://100.74.89.105:8899/api/submissions \
     -H 'Content-Type: application/json' \
     -d "{\"hf_id\":\"myorg/run7-step$STEP\",\"suite\":\"quick\",\"submitter\":\"$USER\",\"note\":\"step $STEP\"}"
```

---

## Errors

| code | meaning | what to do |
|---|---|---|
| 401 | server requires `X-Token` | get the token from the operator; dashboard users append `?token=…` |
| 404 | unknown submission id | check `GET /api/submissions` |
| 409 | cancel on a non-queued job | it's already running or finished |
| 422 | bad request shape | `hf_id` must be `org/name` or `local/<name>`; suite `quick|full`; kind `auto|base|instruct` |
| 409 / 507 | artifact upload refused | name already taken (immutable — pick a new one) / size cap or storage quota hit |
| (failed status) | preflight or run failure | read `error` on the row; raw output at `/api/runs/{id}/log` |

## Etiquette

The card is shared with real training jobs. `quick` for iteration; `full` when a
checkpoint matters. Batch your curiosity — every `full` on a big-vocab model is
an hour of shared GPU. And put your name in `submitter`, so the queue answers
"whose job is this?" without archaeology.
