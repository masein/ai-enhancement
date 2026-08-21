# The team benchmark — 5-minute guide

One place for the whole loop: **benchmark any model, track your training runs,
store your checkpoints** — on our shared GPU, on the tailnet.

Dashboard: **http://teraformer-5090-3:8899/** (you need to be on the tailnet —
ask Omar for an invite). This guide lives at `/guide` on the same host.

---

## 1 · Benchmark a model in 30 seconds (no code)

Open the dashboard → **Submit & Queue** → paste a model id → Submit.

- Any public Hugging Face id works: `HuggingFaceTB/SmolLM2-135M`
- `suite`: **quick** = hellaswag + arc-easy + perplexity, minutes — use while
  iterating. **full** = all 8 benchmarks + perplexity, up to an hour+ — use when
  a checkpoint matters. A quick run upgrades to full later by running only the
  missing tasks.
- `kind`: leave on **auto** (it detects chat templates; wrong templates move
  scores by tens of points).
- Put **your name** in the submitter field. The queue shows live progress; when
  it's done your model is on the Leaderboard with everyone else's.

Re-submitting an already-benchmarked model is free (results are cached per
task), and submitting something already in the queue just joins that run.

## 2 · Track your training run (2 lines in your loop)

Grab the client (one stdlib-only file, no pip installs):

```bash
curl -O https://raw.githubusercontent.com/masein/ai-enhancement/main/clients/bench_client.py
```

Then in your training code:

```python
from bench_client import Bench
bench = Bench("http://teraformer-5090-3:8899")

run = bench.init("my-run7", submitter="yourname",
                 config={"lr": 3e-4, "batch": 32})     # config shows + diffs in the UI
...
run.log({"loss": loss.item(), "lr": lr}, step=step)     # every step or every N — your call
...
run.finish()
```

Two useful conventions: put `batch_size` / `micro_batch_size` / `grad_accum` in
`config=` (they show and diff in the UI), and log a cumulative `tokens` metric —
the run list then shows total trained tokens.

Your curves appear **live** on the **Training** tab: overlay runs to compare,
smoothing slider, log scale, and a config diff that highlights exactly what you
changed between two runs. Logging is buffered and can never crash your training —
if the service is unreachable it warns once and your loop keeps going. Anything
is a metric: `grad_norm`, `tokens_per_s`, `gpu_mem_gb`, whatever you log.

## 3 · Benchmark your checkpoints — no Hugging Face account needed

When you save a checkpoint, upload it and let the service evaluate it:

```python
model.save_pretrained("ckpt"); tokenizer.save_pretrained("ckpt")
model_id = bench.upload_artifact("my-run7-step4000", "ckpt")   # -> local/my-run7-step4000
run.log_checkpoint(4000, model_id)     # marks the step AND queues the benchmark
```

Now the Training tab shows **benchmark scores plotted against your training
steps**, right under your loss curve — select your run and look for "Benchmarks
along this run". (If you prefer the Hub, push there and pass the repo id to
`log_checkpoint` instead — both work.)

Want the whole thing as working code? The repo has a runnable sample that
trains a tiny model and does all of the above:

```bash
git clone https://github.com/masein/ai-enhancement && cd ai-enhancement
python examples/train_and_benchmark.py --bench http://teraformer-5090-3:8899 --dry-run   # 1-minute check
python examples/train_and_benchmark.py --bench http://teraformer-5090-3:8899 --steps 200 --checkpoint-every 100
```

## 4 · Reading the dashboard

**Overview** — best model, how many differences are statistically real.
Checkpoint evals (anything you uploaded as an artifact) are kept out of the
model comparisons by default so sweeps don't bury the ladder — the
**"+ checkpoints"** switch next to the Base/Instruct filter brings them in,
and searching a checkpoint by name always finds it. Their natural home is the
Training tab (score vs step) and the Evals query bar.
**Training** — your live curves, run compare, config diff, benchmark-vs-step.
The runs list has a search box (name, project or person), a status filter and a
sort menu — "best loss" and "recently updated" are the two you'll live in.
**Submit & Queue** — submit models, watch progress, read failure logs. The
queue is searchable and filterable the same way (find *your* jobs, failures
first); click any column header to sort.
**Leaderboard** — every model × every task, ± standard error, sortable.
**Tasks** — one panel per benchmark; the dashed line is chance. Distance from
chance is the real score: **50% on Winogrande/PIQA is a coin flip, not a pass**,
and MMLU sits at ~25% for every model under ~1B — that's expected, not a bug.
**Perplexity & Loss** — bits/byte on pinned corpora, plus cross-entropy in
nats/byte (same quantity as your training loss, per byte, so it's comparable
across tokenizers — line it up with your loss curve).
**Evals** — full provenance (sorted newest-eval-first; click a column to
re-sort) and a query bar over every raw number (`pythia task:mmlu_ value>0.3`,
`metric:cross_entropy`), with CSV export. The query bar suggests completions
as you type — model, task and metric names straight from the data, so you
never have to remember what a task is called: ↑↓ to pick, Enter to insert.

Two honest-statistics habits the dashboard enforces: every score carries its
standard error, and if two error bars overlap, treat the models as tied.

## 5 · House rules

The GPU is shared with real training jobs, so: one evaluation runs at a time
(your submission queues — that's normal); use **quick** while iterating and save
**full** for checkpoints that matter; models are capped at **4B params**;
uploaded checkpoints must be **safetensors** (pickle `.bin` files are refused —
they execute code on load) and count against a shared storage quota, so delete
old artifacts you don't need (`DELETE /api/artifacts/<name>`, scores stay).
Always set your name as submitter — the queue should answer "whose job is this"
without archaeology.

## FAQ

**My model scored below 25% on MMLU.** Below-chance on a 4-way task means the
prompt format is fighting the model (usually a chat-template mismatch), not that
it "knows negative things". Try `kind: base` vs `instruct`, or ask Omar.

**GSM8K is ~0%.** Correct and expected under ~1B params — written math barely
exists at this scale. It's there so you can see it emerge.

**A gated model (gemma, llama) fails preflight.** The server's HF account must
accept that model's license once — ask Omar, or submit an ungated mirror.

**My run shows "stale?"** Your training stopped logging without calling
`run.finish()` (crash, Ctrl-C). Cosmetic — logging again resumes it.

**My run says "finished" but there are no benchmark scores yet.** "Finished"
means the *training* finished; your checkpoints may still be in the eval queue
(one runs at a time). The run's page shows "benchmarks: X/Y done" and the
Submit & Queue tab shows live progress — scores appear as each one lands.

**Something else broke.** Every failed submission has a `log` link with the raw
error, and the error messages are written to be actionable. If they aren't: Omar.

*Everything here is also an HTTP API — see
[API.md](https://github.com/masein/ai-enhancement/blob/main/API.md) if you'd
rather curl.*
