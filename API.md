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
 "kind": "auto",                // "auto" | "base" | "instruct" — default auto
 "submitter": "masein",           // shows on the queue and in provenance
 "note": "run7 step 4000"}      // free text, shows as a tooltip
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
               "avg": 0.321, "avgRaw": 0.514,   // required-list mean: above-chance, and raw
               "partialAvg": 0.455, "navg": 8,  // over whatever it ran — a diagnostic, never a rank
               "archinfo": {"arch": "Qwen3ForCausalLM", "hidden": 1024,   // captured at preflight from
                            "layers": 28, "heads": 16,                     // the model's config.json —
                            "ctx": 40960, "vocab": 151936},                // null for CLI-run models
               "minutes": 61.2, "date": "2026-08-19 10:02:11", ...} ],
  "accTasks": ["mmlu", "hellaswag", ...],       // higher-is-better, proportions
  "pplTasks": ["ppl_code", "ppl_fineweb_edu"],  // lower-is-better, no stderr
  "tasks":  { "hellaswag": {"metric": "acc_norm", "lower": false, "chance": 0.25}, ... },
  "cells":  { "hellaswag": { "myorg/my-model": {"v": 0.412, "se": 0.005, "shots": 5, "n": 10042} } },
  "sig":    { "hellaswag": [ ["modelA", "modelB", 0.062, 3.1, true], ... ] },
              // pairwise [a, b, diff, z, significant_at_95%] — check before claiming a win
  "extra":  [ ["my-model", "mmlu_anatomy", "acc", 0.2519, 0.0021, 5, 135], ... ]  // every raw metric incl. subtasks
}
```

Read a score as `cells[task][hf_id].v ± .se`, with the metric name and direction
from `tasks[task]`. Quote `bits_per_byte` for the perplexity tasks — it's the
tokenizer-independent one.

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
