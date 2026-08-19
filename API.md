# Benchmark service API — integrate it into your training loop

For anyone who wants their checkpoints benchmarked automatically: everything the
dashboard does goes through this JSON API, so your training code can do it too.
The intended pattern is **push a checkpoint to Hugging Face → submit its id here →
keep training → collect scores later**. Evaluation runs on the shared GPU, one
job at a time, and results appear on the team leaderboard.

Base URL (on the tailnet): `http://teraformer-5090-3:8899`

Auth: none by default. If the operator sets `SUBMIT_TOKEN`, send it as an
`X-Token` header on POSTs (the dashboard picks it up from `?token=…` in the URL).

The zero-dependency Python client in [`clients/bench_client.py`](clients/bench_client.py)
wraps all of this in one stdlib-only file — vendor it into your repo.

---

## The 30-second version

```python
from bench_client import Bench
bench = Bench("http://teraformer-5090-3:8899")

sid = bench.submit("myorg/my-model", suite="quick", submitter="you")  # returns immediately
bench.wait(sid, echo=True)                                            # optional: block until done
print(bench.scores("myorg/my-model"))
# {'hellaswag': {'value': 0.412, 'stderr': 0.005, 'metric': 'acc_norm', 'shots': 5, ...}, ...}
```

Or from a shell:

```bash
python bench_client.py --base http://teraformer-5090-3:8899 submit myorg/my-model --suite quick --wait
```

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

**Kind matters.** `kind:"auto"` (default) applies the chat template iff the repo
ships one — the right call almost always. Only override it if you know your
checkpoint is mislabeled; a wrong template moves scores by tens of points.

**Comparability.** Every run uses the same few-shot counts, seed, dtype and
harness version (lm_eval 0.4.12, pinned). Scores here are comparable to *each
other*, not to public leaderboards (different n-shot conventions).

---

## Endpoints

### POST /api/submissions — queue a model

```json
{"hf_id": "myorg/my-model",     // required, org/name on the HF Hub
 "suite": "quick",              // "quick" (hellaswag+arc_easy+perplexity) | "full" (all tasks) — default full
 "kind": "auto",                // "auto" | "base" | "instruct" — default auto
 "submitter": "omar",           // shows on the queue and in provenance
 "note": "run7 step 4000"}      // free text, shows as a tooltip
```

Returns `{"id": 12, "status": "queued"}` — or, if the model is already active,
`{"id": 9, "status": "running", "note": "already in the queue — joining the existing run"}`.

### GET /api/submissions?limit=100 — the queue, newest first

Each row:

```json
{"id": 12, "hf_id": "myorg/my-model", "kind": "instruct", "suite": "quick",
 "submitter": "omar", "note": "run7 step 4000",
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
               "params": 596049920, "avg": 0.393, "navg": 8,   // mean over accuracy tasks it ran, and how many
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

---

## The training-loop pattern

Submit at every checkpoint, don't wait, collect at the end (or from a separate
process). The service dedupes and resumes, so this is cheap and crash-safe:

```python
# during training — after each checkpoint is pushed to the Hub
from bench_client import Bench, BenchError
bench = Bench("http://teraformer-5090-3:8899")

def on_checkpoint(step: int, repo_id: str):
    try:
        bench.submit(repo_id, suite="quick", submitter="omar", note=f"step {step}")
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
curl -s -X POST http://teraformer-5090-3:8899/api/submissions \
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
| 422 | bad request shape | `hf_id` must be `org/name`; suite `quick|full`; kind `auto|base|instruct` |
| (failed status) | preflight or run failure | read `error` on the row; raw output at `/api/runs/{id}/log` |

## Etiquette

The card is shared with real training jobs. `quick` for iteration; `full` when a
checkpoint matters. Batch your curiosity — every `full` on a big-vocab model is
an hour of shared GPU. And put your name in `submitter`, so the queue answers
"whose job is this?" without archaeology.
